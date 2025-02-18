# Scaling Autonomous Agents via Automatic Reward Modeling And Planning 

**Title (ZH)**: 通过自动奖励建模与规划扩展自主代理 

**Authors**: Zhenfang Chen, Delin Chen, Rui Sun, Wenjun Liu, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12130)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across a range of text-generation tasks. However, LLMs still struggle with problems requiring multi-step decision-making and environmental feedback, such as online shopping, scientific reasoning, and mathematical problem-solving. Unlike pure text data, collecting large-scale decision-making data is challenging. Moreover, many powerful LLMs are only accessible through APIs, which hinders their fine-tuning for agent tasks due to cost and complexity. To address LLM agents' limitations, we propose a framework that can automatically learn a reward model from the environment without human annotations. This model can be used to evaluate the action trajectories of LLM agents and provide heuristics for task planning. Specifically, our approach involves employing one LLM-based agent to navigate an environment randomly, generating diverse action trajectories. Subsequently, a separate LLM is leveraged to assign a task intent and synthesize a negative response alongside the correct response for each trajectory. These triplets (task intent, positive response, and negative response) are then utilized as training data to optimize a reward model capable of scoring action trajectories. The effectiveness and generalizability of our framework are demonstrated through evaluations conducted on different agent benchmarks. In conclusion, our proposed framework represents a significant advancement in enhancing LLM agents' decision-making capabilities. By automating the learning of reward models, we overcome the challenges of data scarcity and API limitations, potentially revolutionizing the application of LLMs in complex and interactive environments. This research paves the way for more sophisticated AI agents capable of tackling a wide range of real-world problems requiring multi-step decision-making. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多种文本生成任务中展现了出色的性能。然而，LLMs 在需要多步决策和环境反馈的问题上仍然存在挑战，如在线购物、科学推理和数学问题解决等。与纯粹的文本数据不同，收集大规模的决策数据是非常具有挑战性的。此外，许多强大的LLMs仅可通过API获取，这增加了它们对代理任务进行微调的成本和复杂性。为解决LLM代理的限制，我们提出了一种无需人工标注即可自动学习奖励模型的框架。该模型可用于评估LLM代理的行为轨迹，并为任务规划提供启发式建议。具体而言，我们的方法包括使用基于LLM的代理在环境中随机导航，生成多样化的行为轨迹。随后，利用另一個LLM给每个轨迹分配任务意图，并合成与正确响应并列的错误响应。这些三元组（任务意图、正响应和负响应）将被用作训练数据以优化一个能够评估行为轨迹的奖励模型。我们通过在不同代理基准上的评估证明了框架的有效性和普适性。总之，我们提出的框架显著改进了LLM代理的决策能力。通过自动化奖励模型的学习，我们克服了数据稀缺性和API限制的问题，有可能变革LLM在复杂和交互式环境中的应用。本研究为开发能够解决多步决策所需广泛现实问题的更高级AI代理奠定了基础。 

---
# A Study on Leveraging Search and Self-Feedback for Agent Reasoning 

**Title (ZH)**: 利用搜索和自我反馈提升智能体推理能力的研究 

**Authors**: Karthikeyan K, Michelle Yuan, Elman Mansimov, Katerina Margatina, Anurag Pratik, Daniele Bonadiman, Monica Sunkara, Yi Zhang, Yassine Benajiba  

**Link**: [PDF](https://arxiv.org/pdf/2502.12094)  

**Abstract**: Recent works have demonstrated that incorporating search during inference can significantly improve reasoning capabilities of language agents. Some approaches may make use of the ground truth or rely on model's own generated feedback. The search algorithm uses this feedback to then produce values that will update its criterion for exploring and exploiting various reasoning paths. In this study, we investigate how search and model's self-feedback can be leveraged for reasoning tasks. First, we explore differences in ground-truth feedback and self-feedback during search for math reasoning. Second, we observe limitations in applying search techniques to more complex tasks like tool-calling and design domain-specific approaches to address these gaps. Our experiments reveal challenges related to generalization when solely relying on self-feedback during search. For search to work effectively, either access to the ground-truth is needed or feedback mechanisms need to be carefully designed for the specific task. 

**Abstract (ZH)**: 近期的研究表明，在推理过程中融入搜索可以显著提高语言代理的推理能力。一些方法可能利用真实答案或依赖模型生成的反馈。搜索算法利用这种反馈来生成更新其探索和利用各种推理路径的标准值。在本研究中，我们探讨了如何利用搜索和模型的自我反馈来完成推理任务。首先，我们探讨了在数学推理中利用真实反馈和自我反馈之间的差异。其次，我们观察了将搜索技术应用于更复杂任务（如工具调用和设计领域）时所遇到的限制，并针对这些差距提出了特定领域的解决方案。我们的实验揭示了仅依赖自我反馈进行搜索时存在的泛化挑战。为了使搜索有效，要么需要访问真实答案，要么需要为特定任务精心设计反馈机制。 

---
# Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration 

**Title (ZH)**: 利用双过程理论在语言代理框架中实现实时人机协同合作 

**Authors**: Shao Zhang, Xihuai Wang, Wenhao Zhang, Chaoran Li, Junru Song, Tingyu Li, Lin Qiu, Xuezhi Cao, Xunliang Cai, Wen Yao, Weinan Zhang, Xinbing Wang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11882)  

**Abstract**: Agents built on large language models (LLMs) have excelled in turn-by-turn human-AI collaboration but struggle with simultaneous tasks requiring real-time interaction. Latency issues and the challenge of inferring variable human strategies hinder their ability to make autonomous decisions without explicit instructions. Through experiments with current independent System 1 and System 2 methods, we validate the necessity of using Dual Process Theory (DPT) in real-time tasks. We propose DPT-Agent, a novel language agent framework that integrates System 1 and System 2 for efficient real-time simultaneous human-AI collaboration. DPT-Agent's System 1 uses a Finite-state Machine (FSM) and code-as-policy for fast, intuitive, and controllable decision-making. DPT-Agent's System 2 integrates Theory of Mind (ToM) and asynchronous reflection to infer human intentions and perform reasoning-based autonomous decisions. We demonstrate the effectiveness of DPT-Agent through further experiments with rule-based agents and human collaborators, showing significant improvements over mainstream LLM-based frameworks. To the best of our knowledge, DPT-Agent is the first language agent framework that achieves successful real-time simultaneous human-AI collaboration autonomously. Code of DPT-Agent can be found in this https URL. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的代理在人机逐步协作方面表现出色，但在需要实时互动的多项任务中则面临挑战。延迟问题和推断多变的人类策略的困难阻碍了它们在无明确指令的情况下做出自主决策的能力。通过使用当前独立的System 1和System 2方法进行实验，我们验证了在实时任务中使用双重过程理论（DPT）的必要性。我们提出了一种名为DPT-Agent的新颖语言代理框架，该框架结合了System 1和System 2，以实现高效的实时多项人机协作。DPT-Agent的System 1采用了有限状态机（FSM）和代码作为策略，以实现快速、直观和可控的决策。DPT-Agent的System 2结合了心智理论（ToM）和异步反思，以推断人类意图并进行基于推理的自主决策。通过与基于规则的代理和人类合作者进行进一步实验，我们证明了DPT-Agent的有效性，显示出显著优于主流的LLM基框架。据我们所知，DPT-Agent是第一个能够在无明确指令的情况下成功实现实时多项人机协作的语言代理框架。DPT-Agent的代码可在以下链接中找到：[插入链接]。 

---
# Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning 

**Title (ZH)**: Table-Critic：一种用于表格推理中协作批评与完善的大规模多智能体框架 

**Authors**: Peiying Yu, Guoxin Chen, Jingjing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11799)  

**Abstract**: Despite the remarkable capabilities of large language models (LLMs) in various reasoning tasks, they still struggle with table reasoning tasks, particularly in maintaining consistency throughout multi-step reasoning processes. While existing approaches have explored various decomposition strategies, they often lack effective mechanisms to identify and correct errors in intermediate reasoning steps, leading to cascading error propagation. To address these issues, we propose Table-Critic, a novel multi-agent framework that facilitates collaborative criticism and iterative refinement of the reasoning process until convergence to correct solutions. Our framework consists of four specialized agents: a Judge for error identification, a Critic for comprehensive critiques, a Refiner for process improvement, and a Curator for pattern distillation. To effectively deal with diverse and unpredictable error types, we introduce a self-evolving template tree that systematically accumulates critique knowledge through experience-driven learning and guides future reflections. Extensive experiments have demonstrated that Table-Critic achieves substantial improvements over existing methods, achieving superior accuracy and error correction rates while maintaining computational efficiency and lower solution degradation rate. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）在各种推理任务上表现出色，但在处理表格推理任务时，特别是在保持多步骤推理过程中的一致性方面，它们仍存在困难。虽然现有方法探索了多种分解策略，但它们往往缺少有效机制来识别和修正中间推理步骤中的错误，导致错误传递。为了解决这些问题，我们提出了一种名为Table-Critic的新型多agent框架，该框架通过协作批评和迭代优化推理过程，直到收敛于正确解。该框架包含四个专门的代理：裁判（Judge）负责错误识别，批评者（Critic）负责综合批评，优化器（Refiner）负责过程改进，策展人（Curator）负责模式提炼。为了有效应对多样且不可预测的错误类型，我们引入了一种自演化模板树，该树通过经验驱动的学习系统地积累批评知识，并指导未来的反思。广泛实验表明，Table-Critic相对于现有方法取得了显著改进，实现了更高的准确率和错误修正率，同时保持了计算效率和较低的解退化率。 

---
# Competing LLM Agents in a Non-Cooperative Game of Opinion Polarisation 

**Title (ZH)**: 非合作意见极化博弈中的竞争语言模型代理 

**Authors**: Amin Qasmi, Usman Naseem, Mehwish Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2502.11649)  

**Abstract**: We introduce a novel non-cooperative game to analyse opinion formation and resistance, incorporating principles from social psychology such as confirmation bias, resource constraints, and influence penalties. Our simulation features Large Language Model (LLM) agents competing to influence a population, with penalties imposed for generating messages that propagate or counter misinformation. This framework integrates resource optimisation into the agents' decision-making process. Our findings demonstrate that while higher confirmation bias strengthens opinion alignment within groups, it also exacerbates overall polarisation. Conversely, lower confirmation bias leads to fragmented opinions and limited shifts in individual beliefs. Investing heavily in a high-resource debunking strategy can initially align the population with the debunking agent, but risks rapid resource depletion and diminished long-term influence. 

**Abstract (ZH)**: 我们提出了一种新颖的非合作博弈，用于分析意见形成和抵制行为，该博弈融合了社会心理学中的确认偏见、资源约束和影响惩罚等原则。我们的模拟中，大型语言模型（LLM）代理相互竞争，以影响人群，并对传播或反驳虚假信息的信息采取惩罚措施。该框架将资源优化融入代理的决策过程中。研究结果表明，较高的确认偏见虽然加强了群体内意见的一致性，但也加剧了整体极化。相反，较低的确认偏见会导致意见分化，并限制个人信念的转变。尽管投资高资源的驳斥策略可以初期使人群与驳斥代理保持一致，但也存在资源迅速耗尽和长期影响力减弱的风险。 

---
# Equilibrate RLHF: Towards Balancing Helpfulness-Safety Trade-off in Large Language Models 

**Title (ZH)**: 平衡RLHF：在大规模语言模型中实现帮助性与安全性的权衡稳定策略 

**Authors**: Yingshui Tan, Yilei Jiang, Yanshi Li, Jiaheng Liu, Xingyuan Bu, Wenbo Su, Xiangyu Yue, Xiaoyong Zhu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.11555)  

**Abstract**: Fine-tuning large language models (LLMs) based on human preferences, commonly achieved through reinforcement learning from human feedback (RLHF), has been effective in improving their performance. However, maintaining LLM safety throughout the fine-tuning process remains a significant challenge, as resolving conflicts between safety and helpfulness can be non-trivial. Typically, the safety alignment of LLM is trained on data with safety-related categories. However, our experiments find that naively increasing the scale of safety training data usually leads the LLMs to an ``overly safe'' state rather than a ``truly safe'' state, boosting the refusal rate through extensive safety-aligned data without genuinely understanding the requirements for safe responses. Such an approach can inadvertently diminish the models' helpfulness. To understand the phenomenon, we first investigate the role of safety data by categorizing them into three different groups, and observe that each group behaves differently as training data scales up. To boost the balance between safety and helpfulness, we propose an Equilibrate RLHF framework including a Fine-grained Data-centric (FDC) approach that achieves better safety alignment even with fewer training data, and an Adaptive Message-wise Alignment (AMA) approach, which selectively highlight the key segments through a gradient masking strategy. Extensive experimental results demonstrate that our approach significantly enhances the safety alignment of LLMs while balancing safety and helpfulness. 

**Abstract (ZH)**: 基于人类偏好的微调大型语言模型（LLMs），通常通过人类反馈强化学习（RLHF）实现，已被证明能够有效提高模型性能。然而，在微调过程中保持LLM的安全性仍然是一个重大挑战，因为解决安全性和帮助性之间的冲突可能并非易事。通常，LLM的安全对齐是在包含安全相关类别的数据上进行训练的。然而，我们的实验表明，盲目增加安全训练数据的规模通常会使LLM进入一个“过度安全”的状态，而不是一个“真正安全”的状态，这会通过广泛的安全对齐数据提高拒绝率，而未能真正理解安全响应的要求，从而意外地降低了模型的帮助性。为了理解这种现象，我们首先通过将安全数据分为三类来研究其作用，并观察到随着训练数据规模的扩大，每组数据的行为不同。为了平衡安全性和帮助性，我们提出了一种均衡RLHF框架，包括一种细粒度数据为中心的方法（FDC），即使使用较少的训练数据也能更好地实现安全对齐，以及一种适应性消息级对齐（AMA）方法，该方法通过梯度屏蔽策略突出关键段落。广泛的实验结果表明，我们的方法显著增强了LLM的安全对齐，同时平衡了安全性和帮助性。 

---
# AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection 

**Title (ZH)**: AGrail：一种有效的自适应安全检测终身代理防护栏 

**Authors**: Weidi Luo, Shenghong Dai, Xiaogeng Liu, Suman Banerjee, Huan Sun, Muhao Chen, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2502.11448)  

**Abstract**: The rapid advancements in Large Language Models (LLMs) have enabled their deployment as autonomous agents for handling complex tasks in dynamic environments. These LLMs demonstrate strong problem-solving capabilities and adaptability to multifaceted scenarios. However, their use as agents also introduces significant risks, including task-specific risks, which are identified by the agent administrator based on the specific task requirements and constraints, and systemic risks, which stem from vulnerabilities in their design or interactions, potentially compromising confidentiality, integrity, or availability (CIA) of information and triggering security risks. Existing defense agencies fail to adaptively and effectively mitigate these risks. In this paper, we propose AGrail, a lifelong agent guardrail to enhance LLM agent safety, which features adaptive safety check generation, effective safety check optimization, and tool compatibility and flexibility. Extensive experiments demonstrate that AGrail not only achieves strong performance against task-specific and system risks but also exhibits transferability across different LLM agents' tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速进步使其能够作为自主代理部署，以处理动态环境中的复杂任务。这些LLMs展示了强大的问题解决能力和对多样化情境的适应性。然而，将它们用作代理也带来了显著的风险，包括特定任务风险和系统风险。特定任务风险由代理管理员根据具体任务需求和约束来识别，而系统风险则源自设计或交互中的漏洞，可能导致信息的保密性、完整性和可用性（CI A）受到威胁，并引发安全风险。目前现有的防御机构无法适应性且有效地减轻这些风险。本文提出了一种生命周期代理护栏（AGrail），以增强LLM代理的安全性，其特点包括适应性安全检查生成、有效安全检查优化以及工具的兼容性和灵活性。广泛实验表明，AGrail不仅在对抗特定任务风险和系统风险方面表现出色，还具有在不同类型LLM代理任务之间的泛化能力。 

---
# SMART: Self-Aware Agent for Tool Overuse Mitigation 

**Title (ZH)**: SMART：自我感知代理工具过度使用缓解 

**Authors**: Cheng Qian, Emre Can Acikgoz, Hongru Wang, Xiusi Chen, Avirup Sil, Dilek Hakkani-Tür, Gokhan Tur, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.11435)  

**Abstract**: Current Large Language Model (LLM) agents demonstrate strong reasoning and tool use capabilities, but often lack self-awareness, failing to balance these approaches effectively. This imbalance leads to Tool Overuse, where models unnecessarily rely on external tools for tasks solvable with parametric knowledge, increasing computational overhead. Inspired by human metacognition, we introduce SMART (Strategic Model-Aware Reasoning with Tools), a paradigm that enhances an agent's self-awareness to optimize task handling and reduce tool overuse. To support this paradigm, we introduce SMART-ER, a dataset spanning three domains, where reasoning alternates between parametric knowledge and tool-dependent steps, with each step enriched by rationales explaining when tools are necessary. Through supervised training, we develop SMARTAgent, a family of models that dynamically balance parametric knowledge and tool use. Evaluations show that SMARTAgent reduces tool use by 24% while improving performance by over 37%, enabling 7B-scale models to match its 70B counterpart and GPT-4o. Additionally, SMARTAgent generalizes to out-of-distribution test data like GSM8K and MINTQA, maintaining accuracy with just one-fifth the tool calls. These highlight the potential of strategic tool use to enhance reasoning, mitigate overuse, and bridge the gap between model size and performance, advancing intelligent and resource-efficient agent designs. 

**Abstract (ZH)**: 当前的大型语言模型（LLM）代理表现出强大的推理和工具使用能力，但往往缺乏自我意识，无法有效平衡这些方法。这种不平衡导致了工具过度使用，即模型在可以通过参数化知识解决的任务中无必要地依赖外部工具，从而增加了计算开销。受到人类元认知的启发，我们引入了SMART（战略模型感知推理与工具使用）范式，该范式增强代理的自我意识，以优化任务处理并减少工具过度使用。为了支持这一范式，我们引入了SMART-ER数据集，该数据集跨越三个领域，在推理过程中交替使用参数化知识和工具依赖步骤，每一步都通过解释何时需要使用工具的原因使其更加丰富。通过监督训练，我们开发了SMART-Agent家族模型，能够动态平衡参数化知识和工具的使用。评估结果显示，SMART-Agent在减少工具使用24%的同时提高了超过37%的性能，使7B规模的模型能够与70B版本以及GPT-4o相媲美。此外，SMART-Agent能够泛化至分布外测试数据，如GSM8K和MINTQA，仅需平时五分之一的工具调用就能保持准确率。这些结果突显了战略性工具使用的潜力，可以增强推理能力、缓解过度使用问题，并缩小模型规模与性能之间的差距，从而推动更智能和资源高效代理的设计。 

---
# \textsc{FLAG-Trader}: Fusion LLM-Agent with Gradient-based Reinforcement Learning for Financial Trading 

**Title (ZH)**: \textsc{FLAG-Trader}: 结合梯度强化学习的LLM-Agent融合模型在金融交易中的应用 

**Authors**: Guojun Xiong, Zhiyang Deng, Keyi Wang, Yupeng Cao, Haohang Li, Yangyang Yu, Xueqing Peng, Mingquan Lin, Kaleb E Smith, Xiao-Yang Liu, Jimin Huang, Sophia Ananiadou, Qianqian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.11433)  

**Abstract**: Large language models (LLMs) fine-tuned on multimodal financial data have demonstrated impressive reasoning capabilities in various financial tasks. However, they often struggle with multi-step, goal-oriented scenarios in interactive financial markets, such as trading, where complex agentic approaches are required to improve decision-making. To address this, we propose \textsc{FLAG-Trader}, a unified architecture integrating linguistic processing (via LLMs) with gradient-driven reinforcement learning (RL) policy optimization, in which a partially fine-tuned LLM acts as the policy network, leveraging pre-trained knowledge while adapting to the financial domain through parameter-efficient fine-tuning. Through policy gradient optimization driven by trading rewards, our framework not only enhances LLM performance in trading but also improves results on other financial-domain tasks. We present extensive empirical evidence to validate these enhancements. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多模态金融数据上进行微调后，在各种金融任务中展示了令人印象深刻的推理能力。然而，在交互式金融市场中的交易等多步、目标导向的情景中，它们往往难以应对复杂的代理性方法，以改进决策。为了解决这一问题，我们提出了一种名为 \textsc{FLAG-Trader} 的统一架构，该架构将语言处理（通过LLMs进行）与基于梯度的强化学习（RL）策略优化相结合，在这种架构中，部分微调的LLM作为策略网络发挥作用，利用预训练的知识并通过参数高效的微调适应金融领域。通过由交易奖励驱动的策略梯度优化，我们的框架不仅提高了LLM在交易中的表现，还改善了其他金融领域任务的结果。我们提供了广泛的实验证据来验证这些增强效果。 

---
# Planning of Heuristics: Strategic Planning on Large Language Models with Monte Carlo Tree Search for Automating Heuristic Optimization 

**Title (ZH)**: 基于蒙特卡洛树搜索的大规模语言模型战略规划：自动化启发式优化规划 

**Authors**: Chaoxu Mu, Xufeng Zhang, Hui Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11422)  

**Abstract**: Heuristics have achieved great success in solv- ing combinatorial optimization problems (COPs). However, heuristics designed by humans re- quire too much domain knowledge and testing time. Given the fact that Large Language Mod- els (LLMs) possess strong capabilities to under- stand and generate content, and a knowledge base that covers various domains, which offer a novel way to automatically optimize heuristics. There- fore, we propose Planning of Heuristics (PoH), an optimization method that integrates the self- reflection of LLMs with the Monte Carlo Tree Search (MCTS), a well-known planning algo- rithm. PoH iteratively refines generated heuristics by evaluating their performance and providing im- provement suggestions. Our method enables to it- eratively evaluate the generated heuristics (states) and improve them based on the improvement sug- gestions (actions) and evaluation results (rewards), by effectively simulating future states to search for paths with higher rewards. In this paper, we apply PoH to solve the Traveling Salesman Prob- lem (TSP) and the Flow Shop Scheduling Prob- lem (FSSP). The experimental results show that PoH outperforms other hand-crafted heuristics and Automatic Heuristic Design (AHD) by other LLMs-based methods, and achieves the signifi- cant improvements and the state-of-the-art per- formance of our proposed method in automating heuristic optimization with LLMs to solve COPs. 

**Abstract (ZH)**: 启发式方法在解决组合优化问题（COPs）方面取得了巨大成功。然而，由人类设计的启发式方法需要大量的领域知识和测试时间。鉴于大规模语言模型（LLMs）具有强大的理解和生成内容的能力，并且具备涵盖各种领域的知识库，为我们提供了一种新的自动优化启发式方法的途径。因此，我们提出了一种名为Planning of Heuristics（PoH）的优化方法，该方法将LLMs的自我反思与著名的规划算法蒙特卡洛树搜索（MCTS）相结合。PoH通过评估生成启发式的效果并提供改进建议，逐迭代地精炼生成的启发式。我们的方法通过有效模拟未来状态来搜索具有更高奖励的路径，并基于改进建议（动作）和评估结果（奖励）迭代地评估并改进生成的启发式（状态）。在本文中，我们将PoH应用于解决旅行商问题（TSP）和流水线车间调度问题（FSSP）。实验结果表明，PoH在解决COPs的启发式自动化优化方面优于其他手工设计的启发式方法和其他基于LLMs的方法，并且实现了我们所提出方法在自动化启发式优化中的显著改进和最先进的性能。 

---
# TimeCAP: Learning to Contextualize, Augment, and Predict Time Series Events with Large Language Model Agents 

**Title (ZH)**: 时间上下文化预测：通过大型语言模型代理学习上下文化、增强和预测时间序列事件 

**Authors**: Geon Lee, Wenchao Yu, Kijung Shin, Wei Cheng, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11418)  

**Abstract**: Time series data is essential in various applications, including climate modeling, healthcare monitoring, and financial analytics. Understanding the contextual information associated with real-world time series data is often essential for accurate and reliable event predictions. In this paper, we introduce TimeCAP, a time-series processing framework that creatively employs Large Language Models (LLMs) as contextualizers of time series data, extending their typical usage as predictors. TimeCAP incorporates two independent LLM agents: one generates a textual summary capturing the context of the time series, while the other uses this enriched summary to make more informed predictions. In addition, TimeCAP employs a multi-modal encoder that synergizes with the LLM agents, enhancing predictive performance through mutual augmentation of inputs with in-context examples. Experimental results on real-world datasets demonstrate that TimeCAP outperforms state-of-the-art methods for time series event prediction, including those utilizing LLMs as predictors, achieving an average improvement of 28.75% in F1 score. 

**Abstract (ZH)**: 时间序列数据在各种应用中至关重要，包括气候建模、医疗监测和金融分析。理解与实际时间序列数据相关的时间背景信息通常对于准确可靠的事件预测至关重要。在本文中，我们介绍了一种名为TimeCAP的时间序列处理框架，该框架创造性地利用大型语言模型（LLMs）作为时间序列数据的背景补足者，而不仅仅是预测器。TimeCAP包括两个独立的LLM代理：一个生成文本摘要，捕捉时间序列的上下文，另一个利用这个丰富化的摘要做出更具信息量的预测。此外，TimeCAP采用了一种多模态编码器，该编码器与LLM代理协同工作，通过输入中的上下文示例增强其互增效果，从而提高预测性能。在实际数据集上的实验结果表明，TimeCAP在时间序列事件预测方面优于最先进的方法，包括利用LLMs作为预测器的方法，在F1分数上平均提高了28.75%。 

---
# Explorer: Scaling Exploration-driven Web Trajectory Synthesis for Multimodal Web Agents 

**Title (ZH)**: Explorer: 扩展基于探索的网页轨迹合成以支持多模态网页代理 

**Authors**: Vardaan Pahuja, Yadong Lu, Corby Rosset, Boyu Gou, Arindam Mitra, Spencer Whitehead, Yu Su, Ahmed Awadallah  

**Link**: [PDF](https://arxiv.org/pdf/2502.11357)  

**Abstract**: Recent success in large multimodal models (LMMs) has sparked promising applications of agents capable of autonomously completing complex web tasks. While open-source LMM agents have made significant advances in offline evaluation benchmarks, their performance still falls substantially short of human-level capabilities in more realistic online settings. A key bottleneck is the lack of diverse and large-scale trajectory-level datasets across various domains, which are expensive to collect. In this paper, we address this challenge by developing a scalable recipe to synthesize the largest and most diverse trajectory-level dataset to date, containing over 94K successful multimodal web trajectories, spanning 49K unique URLs, 720K screenshots, and 33M web elements. In particular, we leverage extensive web exploration and refinement to obtain diverse task intents. The average cost is 28 cents per successful trajectory, making it affordable to a wide range of users in the community. Leveraging this dataset, we train Explorer, a multimodal web agent, and demonstrate strong performance on both offline and online web agent benchmarks such as Mind2Web-Live, Multimodal-Mind2Web, and MiniWob++. Additionally, our experiments highlight data scaling as a key driver for improving web agent capabilities. We hope this study makes state-of-the-art LMM-based agent research at a larger scale more accessible. 

**Abstract (ZH)**: 近年来，大型多模态模型（LMMs）的突破性进展激发了能够自主完成复杂 Web 任务的代理的应用潜力。虽然开源的 LMM 代理在离线评估基准上取得了显著进展，但在更具现实性的在线环境中，它们的表现仍然远远低于人类的水平。一个关键瓶颈是缺乏覆盖各个领域的多样性和大规模的轨迹级数据集，这些数据集的收集成本较高。在本文中，我们通过开发一个可扩展的方法来解决这一挑战，该方法合成出了迄今为止最大的最多样化轨迹级数据集，包含超过 94,000 条成功的多模态 Web 轨迹，覆盖 49,000 个唯一的 URL，320 万张屏幕截图，以及 3300 万网页元素。特别是，我们利用广泛的 Web 探索和优化来获取多样化的任务意图。平均每条成功的轨迹成本为 28 美分，使其对社区中的广大用户来说都是负担得起的。利用此数据集，我们训练了 Explorer，这是一种多模态 Web 代理，并在如 Mind2Web-Live、Multimodal-Mind2Web 和 MiniWob++ 等离线和在线 Web 代理基准测试中展示了强劲的性能。此外，我们的实验强调了数据规模在提高 Web 代理能力方面是一个关键驱动因素。我们希望这项研究能使得更大规模的 LMM 基础的代理研究更加普及和容易获取。 

---
# AI Generations: From AI 1.0 to AI 4.0 

**Title (ZH)**: AI 世代：从AI 1.0到AI 4.0 

**Authors**: Jiahao Wu, Hengxu You, Jing Du  

**Link**: [PDF](https://arxiv.org/pdf/2502.11312)  

**Abstract**: This paper proposes that Artificial Intelligence (AI) progresses through several overlapping generations: AI 1.0 (Information AI), AI 2.0 (Agentic AI), AI 3.0 (Physical AI), and now a speculative AI 4.0 (Conscious AI). Each of these AI generations is driven by shifting priorities among algorithms, computing power, and data. AI 1.0 ushered in breakthroughs in pattern recognition and information processing, fueling advances in computer vision, natural language processing, and recommendation systems. AI 2.0 built on these foundations through real-time decision-making in digital environments, leveraging reinforcement learning and adaptive planning for agentic AI applications. AI 3.0 extended intelligence into physical contexts, integrating robotics, autonomous vehicles, and sensor-fused control systems to act in uncertain real-world settings. Building on these developments, AI 4.0 puts forward the bold vision of self-directed AI capable of setting its own goals, orchestrating complex training regimens, and possibly exhibiting elements of machine consciousness. This paper traces the historical foundations of AI across roughly seventy years, mapping how changes in technological bottlenecks from algorithmic innovation to high-performance computing to specialized data, have spurred each generational leap. It further highlights the ongoing synergies among AI 1.0, 2.0, 3.0, and 4.0, and explores the profound ethical, regulatory, and philosophical challenges that arise when artificial systems approach (or aspire to) human-like autonomy. Ultimately, understanding these evolutions and their interdependencies is pivotal for guiding future research, crafting responsible governance, and ensuring that AI transformative potential benefits society as a whole. 

**Abstract (ZH)**: 本文提出，人工智能（AI）经历了多个重叠的阶段：AI 1.0（信息型AI）、AI 2.0（自主型AI）、AI 3.0（物理型AI），以及现在想象中的AI 4.0（意识型AI）。每一阶段的AI都是由算法、计算能力和数据驱动的优先事项转变推动的。AI 1.0 引发了模式识别和信息处理方面的突破，推动了计算机视觉、自然语言处理和推荐系统的进步。AI 2.0 在这一基础上，通过利用强化学习和自适应规划等技术，开发了自主型AI应用。即刻决策在数字环境中变得更为现实。AI 3.0 将智能扩展到物理环境中，整合了机器人技术、自动驾驶车辆和传感器融合控制系统，以在不确定的现实世界环境中进行操作。在此基础上，AI 4.0 提出了一个大胆的愿景，即自主设置目标并协调复杂训练流程的自我驱动AI，甚至可能表现出机器意识的某些特征。本文追寻了人工智能在大约七十年历史中的发展基础，描绘了从算法创新到高性能计算再到特化数据，技术瓶颈变化如何推动各阶段的演进。此外，本文还强调了AI 1.0、2.0、3.0 和 4.0 之间的持续协同作用，并探讨了当人工智能系统接近（或追求）类似人类的自主权时所引发的深刻伦理、监管和哲学挑战。最终，理解这些演变及其相互依存关系对于指导未来的研究、制定负责任的治理框架以及确保AI的变革潜力惠及整个社会至关重要。 

---
# Leveraging Multimodal-LLMs Assisted by Instance Segmentation for Intelligent Traffic Monitoring 

**Title (ZH)**: 利用实例分割辅助的多模态大语言模型进行智能交通监控 

**Authors**: Murat Arda Onsu, Poonam Lohan, Burak Kantarci, Aisha Syed, Matthew Andrews, Sean Kennedy  

**Link**: [PDF](https://arxiv.org/pdf/2502.11304)  

**Abstract**: A robust and efficient traffic monitoring system is essential for smart cities and Intelligent Transportation Systems (ITS), using sensors and cameras to track vehicle movements, optimize traffic flow, reduce congestion, enhance road safety, and enable real-time adaptive traffic control. Traffic monitoring models must comprehensively understand dynamic urban conditions and provide an intuitive user interface for effective management. This research leverages the LLaVA visual grounding multimodal large language model (LLM) for traffic monitoring tasks on the real-time Quanser Interactive Lab simulation platform, covering scenarios like intersections, congestion, and collisions. Cameras placed at multiple urban locations collect real-time images from the simulation, which are fed into the LLaVA model with queries for analysis. An instance segmentation model integrated into the cameras highlights key elements such as vehicles and pedestrians, enhancing training and throughput. The system achieves 84.3% accuracy in recognizing vehicle locations and 76.4% in determining steering direction, outperforming traditional models. 

**Abstract (ZH)**: 智能交通系统（ITS）和智慧城市中，一个稳健且高效的交通监测系统是必不可少的。该系统利用传感器和摄像头来跟踪车辆移动，优化交通流量，减少拥堵，提升道路安全，并实现实时自适应交通控制。交通监测模型必须全面理解动态的城市状况，并提供直观的用户界面以实现有效的管理。本研究利用LLaVA视觉定位多模态大语言模型（LLM）在实时Quanser交互实验室仿真平台上进行交通监测任务，涵盖了交叉口、拥堵和碰撞等场景。多个城市位置的摄像头收集实时图像，并将这些图像输入LLaVA模型进行分析。将实例分割模型集成到摄像头中，突出显示关键元素，如车辆和行人，以增强训练和处理效率。该系统在识别车辆位置方面达到了84.3%的准确率，并在确定转向方向方面达到了76.4%的准确率，超过了传统模型。 

---
# PlanGenLLMs: A Modern Survey of LLM Planning Capabilities 

**Title (ZH)**: PlanGenLLMs：现代大型语言模型规划能力综述 

**Authors**: Hui Wei, Zihao Zhang, Shenghua He, Tian Xia, Shijia Pan, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11221)  

**Abstract**: LLMs have immense potential for generating plans, transforming an initial world state into a desired goal state. A large body of research has explored the use of LLMs for various planning tasks, from web navigation to travel planning and database querying. However, many of these systems are tailored to specific problems, making it challenging to compare them or determine the best approach for new tasks. There is also a lack of clear and consistent evaluation criteria. Our survey aims to offer a comprehensive overview of current LLM planners to fill this gap. It builds on foundational work by Kartam and Wilkins (1990) and examines six key performance criteria: completeness, executability, optimality, representation, generalization, and efficiency. For each, we provide a thorough analysis of representative works and highlight their strengths and weaknesses. Our paper also identifies crucial future directions, making it a valuable resource for both practitioners and newcomers interested in leveraging LLM planning to support agentic workflows. 

**Abstract (ZH)**: 大型语言模型（LLMs）在生成计划方面具有巨大的潜力，能够将初始世界状态转化为期望的目标状态。大量研究已经探讨了LLMs在各种规划任务中的应用，包括网络导航、旅行规划和数据库查询等。然而，许多现有的系统针对特定问题进行了定制，这使得它们之间的比较变得困难，也难以确定新任务的最佳方法。此外，缺乏清晰且一致的评估标准。我们所做的调研旨在提供当前LLM规划系统的全面概述，以填补这一空白。该调研基于Kartam和Wilkins（1990）的基础工作，并考察了六个关键性能指标：完备性、可执行性、最优性、表示性、泛化能力和效率。对于每个指标，我们对其代表性工作进行了详尽分析，并指出了它们的优势和不足。我们的论文还指出了未来研究的关键方向，使之成为既有经验的从业者和新入门的研究人员在利用LLM规划支持自主工作流程方面的重要参考资源。 

---
# NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM 

**Title (ZH)**: NavRAG：通过检索增强的大语言模型生成用户导向的实体导航指令 

**Authors**: Zihan Wang, Yaohui Zhu, Gim Hee Lee, Yachun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11142)  

**Abstract**: Vision-and-Language Navigation (VLN) is an essential skill for embodied agents, allowing them to navigate in 3D environments following natural language instructions. High-performance navigation models require a large amount of training data, the high cost of manually annotating data has seriously hindered this field. Therefore, some previous methods translate trajectory videos into step-by-step instructions for expanding data, but such instructions do not match well with users' communication styles that briefly describe destinations or state specific needs. Moreover, local navigation trajectories overlook global context and high-level task planning. To address these issues, we propose NavRAG, a retrieval-augmented generation (RAG) framework that generates user demand instructions for VLN. NavRAG leverages LLM to build a hierarchical scene description tree for 3D scene understanding from global layout to local details, then simulates various user roles with specific demands to retrieve from the scene tree, generating diverse instructions with LLM. We annotate over 2 million navigation instructions across 861 scenes and evaluate the data quality and navigation performance of trained models. 

**Abstract (ZH)**: 视觉-语言导航（VLN）是具身代理的一项基本技能，允许它们遵循自然语言指令在3D环境中导航。高性能的导航模型需要大量的训练数据，手动标注数据的高昂成本严重阻碍了这一领域的进展。因此，一些先前的方法将轨迹视频转换为分步骤的指令以扩充数据集，但这些指令并不符合用户的交流风格，用户往往只是简要描述目的地或提出特定需求。此外，局部导航轨迹忽略了全局上下文和高层次的任务规划。为了应对这些挑战，我们提出了一种检索增强生成（RAG）框架NavRAG，该框架用于为VLN生成用户需求指令。NavRAG利用大语言模型（LLM）构建一个分层场景描述树，从全局布局到局部细节进行3D场景理解，然后模拟具有不同需求的多种用户角色，从场景树中检索信息，生成多种多样的指令。我们对超过200万条导航指令进行了标注，覆盖了861个场景，并评估了训练模型的数据质量和导航性能。 

---
# Talk Structurally, Act Hierarchically: A Collaborative Framework for LLM Multi-Agent Systems 

**Title (ZH)**: 从结构上说话，从层级上行动：一种大型语言模型多 Agent 系统的协作框架 

**Authors**: Zhao Wang, Sota Moriyama, Wei-Yao Wang, Briti Gangopadhyay, Shingo Takamatsu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11098)  

**Abstract**: Recent advancements in LLM-based multi-agent (LLM-MA) systems have shown promise, yet significant challenges remain in managing communication and refinement when agents collaborate on complex tasks. In this paper, we propose \textit{Talk Structurally, Act Hierarchically (TalkHier)}, a novel framework that introduces a structured communication protocol for context-rich exchanges and a hierarchical refinement system to address issues such as incorrect outputs, falsehoods, and biases. \textit{TalkHier} surpasses various types of SoTA, including inference scaling model (OpenAI-o1), open-source multi-agent models (e.g., AgentVerse), and majority voting strategies on current LLM and single-agent baselines (e.g., ReAct, GPT4o), across diverse tasks, including open-domain question answering, domain-specific selective questioning, and practical advertisement text generation. These results highlight its potential to set a new standard for LLM-MA systems, paving the way for more effective, adaptable, and collaborative multi-agent frameworks. The code is available this https URL. 

**Abstract (ZH)**: 以下是对原文的学术规范翻译：

近年来，基于大规模语言模型（LLM）的多智能体系统（LLM-MA）已经显示出巨大的潜力，但在智能体协作完成复杂任务时，管理和优化通信与细化仍然面临重大挑战。本文提出了一种新颖框架“Talk Structurally, Act Hierarchically (TalkHier)”，该框架引入了一种结构化通信协议以实现丰富上下文的信息交流，并构建了一个层次化的细化系统以解决错误输出、虚假信息和偏见等问题。在各种类型的当前最先进模型（包括推理扩展模型（OpenAI-o1）、开源多智能体模型（如AgentVerse）、以及多数投票策略（包括ReAct、GPT4o））的比较中，TalkHier在多个任务上均表现出色，包括开放领域的问题解答、特定领域的选择性提问和实际广告文案生成。这些结果表明，TalkHier有可能为LLM-MA系统设定新的标准，从而开辟更有效、更具适应性和协作性的多智能体框架。相关代码可在 [此链接] 查看。 

---
# Mixture of Tunable Experts - Behavior Modification of DeepSeek-R1 at Inference Time 

**Title (ZH)**: 混合可调专家模型 - 深度Seek-R1推理时的行为修改 

**Authors**: Robert Dahlke, Henrik Klagges, Dan Zecha, Benjamin Merkel, Sven Rohr, Fabian Klemm  

**Link**: [PDF](https://arxiv.org/pdf/2502.11096)  

**Abstract**: We present the Mixture-of-Tunable-Experts (MoTE), a method that extends the Mixture-of-Experts architecture of Large Language Models (LLMs). Without additional training, MoTE enables meaningful and focused behavior changes in LLMs on-the-fly during inference time.
By analyzing the digital LLM brain of DeepSeek-R1 using a technique we dub 'functional Token Resonance Imaging' (fTRI) - inspired by fMRI and using prompts designed to elicit specific behavior (e.g., 'What happened {time}{place}?') - we empirically identify distinctive experts associated with behaviors like refusal responses.
Using MoTE we are able to intervene and control such specific behavior. We switched off the top 10 most refusal-relevant experts (0.07% of R1's 14,848 routed experts), achieving a 52% refusal reduction on sensitive reference prompts without performance degradation on MT-Bench. Random expert deactivation resulted in smaller behavioral shifts with increased noise, whereas forced expert activation led to significantly higher refusal rates.
Our approach shares similarities with sparse autoencoders (SAEs) in terms of explainability and steerability. Unlike SAEs, MoTE does not require large training efforts, as within MoEs with a vast number of experts, specialization already emerged naturally during pretraining.
Our findings suggest that significant functional mechanisms in Mixture-of-Experts architectures can at least partially be localized in a small number of specific experts, rather than being distributed throughout the model's weights. Expert subgroups can be tuned to trigger significant behavior variations, providing insights into the inner workings of LLMs. 

**Abstract (ZH)**: 我们提出了混合可调专家（Mixture-of-Tunable-Experts, MoTE）的方法，这是一种扩展大型语言模型（LLMs）的混合专家架构的方法。在不进行额外训练的情况下，MoTE 能在推理过程中使语言模型即时表现出有意义且聚焦的行为变化。

通过利用我们称之为“功能性标记共振成像”（fTRI，灵感源自功能性磁共振成像 fMRI）的技术，并使用旨在唤起特定行为的提示（例如，“{时间}{地点}发生了什么？”），我们基于 DeepSeek-R1 的数字语言模型大脑进行实证分析，识别出与拒绝回应等行为相关的独特专家。

通过 MoTE，我们能够干预并控制这些特定行为。我们关闭了与拒绝回应最相关的前 10 个专家（占 R1 14,848 个受路由专家的 0.07%），在不损害 MT-Bench 上性能的情况下，显著减少了敏感引用提示中的拒绝率。随机关闭专家仅导致较小的行为变化和增加的噪声，而被迫激活专家则导致拒绝率显著增加。

我们的方法在可解释性和可操控性方面与稀疏自编码器（SAEs）有相似之处。与 SAEs 不同，MoTE 在大规模专家混合模型中，由于专家自然地在预训练过程中专门化，因此不需要大量训练工作。

我们的研究结果表明，在混合专家架构中，至少部分重要的功能机制可以集中在少数特定专家中，而不是分布在模型权重的各个方面。专家子组可以被调优以触发显著的行为变化，这为理解语言模型内部机制提供了见解。 

---
# Agentic LLM Framework for Adaptive Decision Discourse 

**Title (ZH)**: 代理型LLM框架：自适应决策对话 

**Authors**: Antoine Dolant, Praveen Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.10978)  

**Abstract**: Effective decision-making in complex systems requires synthesizing diverse perspectives to address multifaceted challenges under uncertainty. This study introduces a real-world inspired agentic Large Language Models (LLMs) framework, to simulate and enhance decision discourse-the deliberative process through which actionable strategies are collaboratively developed. Unlike traditional decision-support tools, the framework emphasizes dialogue, trade-off exploration, and the emergent synergies generated by interactions among agents embodying distinct personas. These personas simulate diverse stakeholder roles, each bringing unique priorities, expertise, and value-driven reasoning to the table. The framework incorporates adaptive and self-governing mechanisms, enabling agents to dynamically summon additional expertise and refine their assembly to address evolving challenges. An illustrative hypothetical example focused on extreme flooding in a Midwestern township demonstrates the framework's ability to navigate uncertainty, balance competing priorities, and propose mitigation and adaptation strategies by considering social, economic, and environmental dimensions. Results reveal how the breadth-first exploration of alternatives fosters robust and equitable recommendation pathways. This framework transforms how decisions are approached in high-stakes scenarios and can be incorporated in digital environments. It not only augments decision-makers' capacity to tackle complexity but also sets a foundation for scalable and context-aware AI-driven recommendations. This research explores novel and alternate routes leveraging agentic LLMs for adaptive, collaborative, and equitable recommendation processes, with implications across domains where uncertainty and complexity converge. 

**Abstract (ZH)**: 在复杂系统中进行有效的决策需要综合多方面的视角以应对多维度的不确定性挑战。本研究提出了一种受现实启发的主动型大型语言模型（LLM）框架，用于模拟和增强决策对话——通过该过程，行动性策略得到协同开发。与传统的决策支持工具不同，该框架强调对话、权衡探索以及由不同人物特征的代理间交互产生的协同效应。这些人物模拟了不同的利益相关者角色，各自带来独特的优先级、专业知识和价值导向的推理。该框架整合了适应性和自我治理机制，使代理能够动态地召唤额外的专家并不断优化其组合以应对不断演变的挑战。一个示例假想场景，旨在解决中西部某一乡镇的极端洪水问题，展示了该框架在导航不确定性、平衡竞争性优先级以及综合社会、经济和环境维度提出减轻和适应策略方面的能力。研究结果表明，广度优先探索替代方案如何促进稳健且公正的推荐路径。该框架改变了在高风险场景中进行决策的方式，并可融入数字环境中。它不仅增强了决策者处理复杂性的能力，还为可扩展且情境感知的AI驱动推荐奠定了基础。本研究探讨了利用主动型LLM探索适应性、协作性和公平性的新型推荐过程，具有跨领域的重要意义，特别是在不确定性与复杂性交汇的地方。 

---
# SCALE: Towards Collaborative Content Analysis in Social Science with Large Language Model Agents and Human Intervention 

**Title (ZH)**: SCALE：关于大规模语言模型代理与人类介入在社会科学中协作内容分析的研究 

**Authors**: Chengshuai Zhao, Zhen Tan, Chau-Wai Wong, Xinyan Zhao, Tianlong Chen, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10937)  

**Abstract**: Content analysis breaks down complex and unstructured texts into theory-informed numerical categories. Particularly, in social science, this process usually relies on multiple rounds of manual annotation, domain expert discussion, and rule-based refinement. In this paper, we introduce SCALE, a novel multi-agent framework that effectively $\underline{\textbf{S}}$imulates $\underline{\textbf{C}}$ontent $\underline{\textbf{A}}$nalysis via $\underline{\textbf{L}}$arge language model (LLM) ag$\underline{\textbf{E}}$nts. SCALE imitates key phases of content analysis, including text coding, collaborative discussion, and dynamic codebook evolution, capturing the reflective depth and adaptive discussions of human researchers. Furthermore, by integrating diverse modes of human intervention, SCALE is augmented with expert input to further enhance its performance. Extensive evaluations on real-world datasets demonstrate that SCALE achieves human-approximated performance across various complex content analysis tasks, offering an innovative potential for future social science research. 

**Abstract (ZH)**: 内容分析将复杂的非结构化文本分解为理论导向的数值类别。特别是在社会科学中，这一过程通常依赖于多轮的手动注释、领域专家讨论以及基于规则的改进。本文介绍了一种新颖的多智能体框架SCALE，通过大语言模型（LLM）智能体有效地模拟内容分析过程。SCALE 模拟内容分析的关键阶段，包括文本编码、协作讨论以及动态代码本演变，捕捉了人类研究人员的反思深度和适应性讨论。此外，通过整合多种类型的人类干预模式，SCALE 进一步增强了其性能，加入了领域专家的输入。在真实世界数据集上的广泛评估表明，SCALE 在各种复杂内容分析任务中实现了接近人类的表现，为未来社会科学的研究提供了创新的潜力。 

---
# D-CIPHER: Dynamic Collaborative Intelligent Agents with Planning and Heterogeneous Execution for Enhanced Reasoning in Offensive Security 

**Title (ZH)**: D-CIPHER：具有规划能力和异构执行的动态协作智能代理，以增强进攻性安全中的推理能力 

**Authors**: Meet Udeshi, Minghao Shao, Haoran Xi, Nanda Rani, Kimberly Milner, Venkata Sai Charan Putrevu, Brendan Dolan-Gavitt, Sandeep Kumar Shukla, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2502.10931)  

**Abstract**: Large Language Models (LLMs) have been used in cybersecurity in many ways, including their recent use as intelligent agent systems for autonomous security analysis. Capture the Flag (CTF) challenges serve as benchmarks for assessing the automated task-planning abilities of LLM agents across various cybersecurity skill sets. Early attempts to apply LLMs for solving CTF challenges relied on single-agent systems, where feedback was restricted to a single reasoning-action loop. This approach proved inadequate for handling complex CTF tasks. Drawing inspiration from real-world CTF competitions, where teams of experts collaborate, we introduce the D-CIPHER multi-agent LLM framework for collaborative CTF challenge solving. D-CIPHER integrates agents with distinct roles, enabling dynamic feedback loops to enhance reasoning on CTF challenges. It introduces the Planner-Executor agent system, consisting of a Planner agent for overall problem-solving along with multiple heterogeneous Executor agents for individual tasks, facilitating efficient allocation of responsibilities among the LLMs. Additionally, D-CIPHER incorporates an Auto-prompter agent, which improves problem-solving by exploring the challenge environment and generating a highly relevant initial prompt. We evaluate D-CIPHER on CTF benchmarks using multiple LLM models and conduct comprehensive studies to highlight the impact of our enhancements. Our results demonstrate that the multi-agent D-CIPHER system achieves a significant improvement in challenges solved, setting a state-of-the-art performance on three benchmarks: 22.0% on NYU CTF Bench, 22.5% on Cybench, and 44.0% on HackTheBox. D-CIPHER is available at this https URL as the nyuctf_multiagent package. 

**Abstract (ZH)**: 大型语言模型（LLMs）在网络安全领域的应用涵盖了多种方式，包括将其用作自主安全分析的智能代理系统。Capture the Flag（CTF）挑战赛被用作评估LLM代理在各种网络安全技能上的自动化任务规划能力的基准。早期使用LLM解决CTF挑战的尝试依赖于单代理系统，其中反馈仅限于单一推理-行动循环。这种做法对于处理复杂的CTF任务证明是不够的。借鉴真实世界CTF竞赛中专家团队合作的理念，我们提出了D-CIPHER多代理LLM框架，用于协作解决CTF挑战。D-CIPHER集成了具有不同角色的代理，能够实现动态反馈循环，以增强对CTF挑战的推理能力。该框架引入了规划-执行者代理系统，包括一个负责整体问题解决的规划者代理，以及多个异构执行者代理，用于执行单独的任务，从而实现LLM们责任分配的有效化。此外，D-CIPHER还集成了自动提示生成代理，通过探索挑战环境生成高度相关的初始提示以改善问题解决能力。我们在使用多个LLM模型的CTF基准测试上评估了D-CIPHER，并进行了全面研究以突出我们改进措施的影响。我们的结果显示，多代理D-CIPHER系统在三个基准测试上解决了更多的挑战问题，分别达到以下性能指标：在NYU CTF基准测试上为22.0%，在Cybench上为22.5%，在HackTheBox上为44.0%。D-CIPHER可在以下链接获取：这个 https URL，作为nyuctf_multiagent包。 

---
# CoPEFT: Fast Adaptation Framework for Multi-Agent Collaborative Perception with Parameter-Efficient Fine-Tuning 

**Title (ZH)**: CoPEFT：一种基于参数高效微调的多智能体协作感知快速adaptation框架 

**Authors**: Quanmin Wei, Penglin Dai, Wei Li, Bingyi Liu, Xiao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10705)  

**Abstract**: Multi-agent collaborative perception is expected to significantly improve perception performance by overcoming the limitations of single-agent perception through exchanging complementary information. However, training a robust collaborative perception model requires collecting sufficient training data that covers all possible collaboration scenarios, which is impractical due to intolerable deployment costs. Hence, the trained model is not robust against new traffic scenarios with inconsistent data distribution and fundamentally restricts its real-world applicability. Further, existing methods, such as domain adaptation, have mitigated this issue by exposing the deployment data during the training stage but incur a high training cost, which is infeasible for resource-constrained agents. In this paper, we propose a Parameter-Efficient Fine-Tuning-based lightweight framework, CoPEFT, for fast adapting a trained collaborative perception model to new deployment environments under low-cost conditions. CoPEFT develops a Collaboration Adapter and Agent Prompt to perform macro-level and micro-level adaptations separately. Specifically, the Collaboration Adapter utilizes the inherent knowledge from training data and limited deployment data to adapt the feature map to new data distribution. The Agent Prompt further enhances the Collaboration Adapter by inserting fine-grained contextual information about the environment. Extensive experiments demonstrate that our CoPEFT surpasses existing methods with less than 1\% trainable parameters, proving the effectiveness and efficiency of our proposed method. 

**Abstract (ZH)**: 多智能体协同感知有望通过交换互补信息来克服单智能体感知的限制，从而显著提升感知性能。然而，训练一个鲁棒的协同感知模型需要收集足以涵盖所有可能合作场景的大量训练数据，这种做法由于部署成本不可接受而变得不切实际。因此，训练后的模型对具有不同数据分布的新交通场景不够鲁棒，从而限制了其实际应用。现有方法，如领域适应，通过在训练阶段暴露部署数据来缓解这一问题，但也带来了高昂的训练成本，这对于资源受限的智能体来说是不可行的。在本文中，我们提出了一种参数高效微调的轻量级框架CoPEFT，以在低成本条件下快速适应已训练的协同感知模型以适应新的部署环境。CoPEFT开发了协作适配器和智能体提示，分别在宏层和微层进行适应。具体来说，协作适配器利用训练数据和有限的部署数据中的固有知识来适应特征图以匹配新的数据分布。智能体提示进一步通过插入有关环境的细粒度上下文信息来增强协作适配器。广泛的实验结果表明，我们的CoPEFT在不到1%的可训练参数下超越了现有方法，证明了我们所提出方法的有效性和高效性。 

---
# USER-VLM 360: Personalized Vision Language Models with User-aware Tuning for Social Human-Robot Interactions 

**Title (ZH)**: USER-VLM 360：面向社交人机交互的用户感知自适应视觉语言模型 

**Authors**: Hamed Rahimi, Adil Bahaj, Mouad Abrini, Mahdi Khoramshahi, Mounir Ghogho, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2502.10636)  

**Abstract**: The integration of vision-language models into robotic systems constitutes a significant advancement in enabling machines to interact with their surroundings in a more intuitive manner. While VLMs offer rich multimodal reasoning, existing approaches lack user-specific adaptability, often relying on generic interaction paradigms that fail to account for individual behavioral, contextual, or socio-emotional nuances. When customization is attempted, ethical concerns arise from unmitigated biases in user data, risking exclusion or unfair treatment. To address these dual challenges, we propose User-VLM 360°, a holistic framework integrating multimodal user modeling with bias-aware optimization. Our approach features: (1) user-aware tuning that adapts interactions in real time using visual-linguistic signals; (2) bias mitigation via preference optimization; and (3) curated 360° socio-emotive interaction datasets annotated with demographic, emotion, and relational metadata. Evaluations across eight benchmarks demonstrate state-of-the-art results: +35.3% F1 in personalized VQA, +47.5% F1 in facial features understanding, 15% bias reduction, and 30X speedup over baselines. Ablation studies confirm component efficacy, and deployment on the Pepper robot validates real-time adaptability across diverse users. We open-source parameter-efficient 3B/10B models and an ethical verification framework for responsible adaptation. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，符合学术规范：

将视觉-语言模型集成到机器人系统中，构成了使机器以更加直观的方式与环境互动的重要进步。虽然视觉-语言模型提供了丰富的跨模态推理能力，但现有方法缺乏针对用户的适应性，经常依赖于通用的交互范式，这些范式未能考虑到个体的行为、情境或社会情感的细微差别。在尝试进行个性化定制时，由于未缓解用户数据中的偏见，可能会引发伦理问题，进而导致排斥或不公平的对待。为解决这些双重挑战，我们提出了User-VLM 360°这一整体框架，该框架结合了跨模态用户建模与偏见感知优化。我们的方法包括：（1）用户感知调整，通过视觉-语言信号实时适应交互；（2）通过偏好优化缓解偏见；以及（3）包含人口、情绪和关系元数据的360°社会情感交互数据集。在八个基准测试中的评估展示了最先进的结果：个性化问答的F1分数提高35.3%，面部特征理解的F1分数提高47.5%，偏见减少15%，并比基线快30倍。消融研究确认了各个组件的有效性，在Pepper机器人上的部署证明了其在不同用户群体中的实时适应能力。我们开源了参数高效的小模型（3B/10B）和一套伦理验证框架，以促进负责任的适应。 

---
# ProMRVL-CAD: Proactive Dialogue System with Multi-Round Vision-Language Interactions for Computer-Aided Diagnosis 

**Title (ZH)**: ProMRVL-CAD：面向未来的对话系统，支持多轮视觉-语言交互的计算机辅助诊断 

**Authors**: Xueshen Li, Xinlong Hou, Ziyi Huang, Yu Gan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10620)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated extraordinary comprehension capabilities with remarkable breakthroughs on various vision-language tasks. However, the application of LLMs in generating reliable medical diagnostic reports remains in the early stages. Currently, medical LLMs typically feature a passive interaction model where doctors respond to patient queries with little or no involvement in analyzing medical images. In contrast, some ChatBots simply respond to predefined queries based on visual inputs, lacking interactive dialogue or consideration of medical history. As such, there is a gap between LLM-generated patient-ChatBot interactions and those occurring in actual patient-doctor consultations. To bridge this gap, we develop an LLM-based dialogue system, namely proactive multi-round vision-language interactions for computer-aided diagnosis (ProMRVL-CAD), to generate patient-friendly disease diagnostic reports. The proposed ProMRVL-CAD system allows proactive dialogue to provide patients with constant and reliable medical access via an integration of knowledge graph into a recommendation system. Specifically, we devise two generators: a Proactive Question Generator (Pro-Q Gen) to generate proactive questions that guide the diagnostic procedure and a Multi-Vision Patient-Text Diagnostic Report Generator (MVP-DR Gen) to produce high-quality diagnostic reports. Evaluating two real-world publicly available datasets, MIMIC-CXR and IU-Xray, our model has better quality in generating medical reports. We further demonstrate the performance of ProMRVL achieves robust under the scenarios with low image quality. Moreover, we have created a synthetic medical dialogue dataset that simulates proactive diagnostic interactions between patients and doctors, serving as a valuable resource for training LLM. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在各种视觉-语言任务上取得了显著突破，展现了令人惊叹的理解能力。然而，将LLMs应用于生成可靠的医学诊断报告仍处于早期阶段。目前，医学LLMs通常采用被动交互模式，医生以较少或无分析医学图片的主动参与来回应患者的咨询。相比之下，一些聊天机器人仅依据视觉输入响应预定义的问题，缺乏互动对话或考虑患者的医学历史。因此，LLMs生成的患者-聊天机器人互动与实际患者-医生咨询之间的差距仍然存在。为弥补这种差距，我们开发了一种基于LLM的对话系统，称为主动多轮视觉-语言交互以辅助诊断（ProMRVL-CAD），以生成患者友好的疾病诊断报告。所提出的ProMRVL-CAD系统通过将知识图谱集成到推荐系统中，提供主动对话，以便患者获得持续且可靠的医疗访问。具体而言，我们设计了两个生成器：主动问题生成器（Pro-Q Gen），用于生成引导诊断过程的主动问题；以及多视图患者-文本诊断报告生成器（MVP-DR Gen），用于生成高质量的诊断报告。通过评估两个公开可用的大型医疗数据集MIMIC-CXR和IU-Xray，我们的模型在生成医学报告方面质量更高。我们进一步展示，ProMRVL在低质量图像场景下表现出色。此外，我们构建了一个合成的医疗对话数据集，模拟患者与医生之间的主动诊断交互，为训练LLMs提供了宝贵资源。 

---
# Observer-Aware Probabilistic Planning Under Partial Observability 

**Title (ZH)**: 基于观测者意识的局部可观测条件下的概率规划 

**Authors**: Salomé Lepers, Vincent Thomas, Olivier Buffet  

**Link**: [PDF](https://arxiv.org/pdf/2502.10568)  

**Abstract**: In this article, we are interested in planning problems where the agent is aware of the presence of an observer, and where this observer is in a partial observability situation. The agent has to choose its strategy so as to optimize the information transmitted by observations. Building on observer-aware Markov decision processes (OAMDPs), we propose a framework to handle this type of problems and thus formalize properties such as legibility, explicability and predictability. This extension of OAMDPs to partial observability can not only handle more realistic problems, but also permits considering dynamic hidden variables of interest. These dynamic target variables allow, for instance, working with predictability, or with legibility problems where the goal might change during execution. We discuss theoretical properties of PO-OAMDPs and, experimenting with benchmark problems, we analyze HSVI's convergence behavior with dedicated initializations and study the resulting strategies. 

**Abstract (ZH)**: 在本文中，我们关注一类问题，在这类问题中，智能体意识到观察者的存在，而观察者处于部分可观测的状态。智能体需要选择其策略以优化由观察带来的信息传递。基于观察者感知的马尔可夫决策过程（OAMDPs），我们提出了一种框架来处理此类问题，从而定义了可读性、可解释性和可预测性等性质。将OAMDPs扩展到部分可观测性情境不仅能够处理更加现实的问题，还可以考虑动态隐藏变量的特性。这类动态目标变量允许我们在可预测性问题或目标在执行过程中发生变化的可读性问题中工作。我们讨论了部分可观测OAMDPs（PO-OAMDPs）的理论性质，并通过基准问题的实验分析了HSVII收敛行为及其专用初始化策略，研究了由此产生的策略。 

---
# A Self-Supervised Reinforcement Learning Approach for Fine-Tuning Large Language Models Using Cross-Attention Signals 

**Title (ZH)**: 使用跨注意力信号进行大规模语言模型微调的自我监督强化学习方法 

**Authors**: Andrew Kiruluta, Andreas Lemos, Priscilla Burity  

**Link**: [PDF](https://arxiv.org/pdf/2502.10482)  

**Abstract**: We propose a novel reinforcement learning framework for post training large language models that does not rely on human in the loop feedback. Instead, our approach uses cross attention signals within the model itself to derive a self supervised reward, thereby guiding iterative fine tuning of the model policy. By analyzing how the model attends to the input prompt during generation, we construct measures of prompt coverage, focus, and coherence. We then use these measures to rank or score candidate responses, providing a reward signal that encourages the model to produce well aligned, on topic text. In empirical comparisons against standard policy gradient methods and RL fine tuning with synthetic preference models, our method shows significant gains in prompt relevance and consistency over a non RL baseline. While it does not yet match the performance of fully human supervised RLHF systems, it highlights an important direction for scaling alignment with minimal human labeling. We provide a detailed analysis, discuss potential limitations, and outline future work for combining cross-attention based signals with smaller amounts of human feedback. 

**Abstract (ZH)**: 我们提出了一种无需人工参与回访的新型强化学习框架，用于后训练大型语言模型。我们的方法利用模型内部的跨注意力信号来推导自我监督的奖励，从而引导模型策略的迭代微调。通过分析模型在生成过程中对输入提示的注意力分布，我们构建了提示覆盖度、聚焦性和连贯性的衡量指标。然后，我们使用这些衡量指标来对候选响应进行排名或评分，提供一种奖励信号，鼓励模型生成内容相关且一致的文本。在与标准策略梯度方法以及使用合成偏好模型的强化学习微调方法的实证比较中，我们的方法在提示的相关性和一致性方面显著优于非强化学习基线。尽管它尚未达到完全有人类监督的RLHF系统的性能，但它展示了在最少人工标注的情况下扩大对齐方向的重要前景。我们提供了详细的分析，讨论了潜在的局限性，并概述了将跨注意力基信号与少量人工反馈相结合的未来工作。 

---
# Multi-Objective Planning with Contextual Lexicographic Reward Preferences 

**Title (ZH)**: 具有上下文列席奖励偏好的多目标规划 

**Authors**: Pulkit Rustagi, Yashwanthi Anand, Sandhya Saisubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2502.10476)  

**Abstract**: Autonomous agents are often required to plan under multiple objectives whose preference ordering varies based on context. The agent may encounter multiple contexts during its course of operation, each imposing a distinct lexicographic ordering over the objectives, with potentially different reward functions associated with each context. Existing approaches to multi-objective planning typically consider a single preference ordering over the objectives, across the state space, and do not support planning under multiple objective orderings within an environment. We present Contextual Lexicographic Markov Decision Process (CLMDP), a framework that enables planning under varying lexicographic objective orderings, depending on the context. In a CLMDP, both the objective ordering at a state and the associated reward functions are determined by the context. We employ a Bayesian approach to infer a state-context mapping from expert trajectories. Our algorithm to solve a CLMDP first computes a policy for each objective ordering and then combines them into a single context-aware policy that is valid and cycle-free. The effectiveness of the proposed approach is evaluated in simulation and using a mobile robot. 

**Abstract (ZH)**: 自主代理通常需要在多种目标之间进行规划，这些目标的偏好顺序会基于上下文的不同而变化。代理在其运行过程中可能会遇到多种不同上下文，每个上下文会对目标施加独特的词典序顺序，并且可能与每个上下文相关联有不同的奖励函数。现有的一些多目标规划方法通常在整个状态空间中考虑单一的目标偏好顺序，而不支持在环境内部署多种目标顺序下的规划。我们提出了情境词典序马尔可夫决策过程（Contextual Lexicographic Markov Decision Process, CLMDP）框架，该框架允许根据上下文的不同变化规划多种词典序目标顺序。在CLMDP中，状态的目标顺序及其相关的奖励函数都是由上下文确定的。我们采用贝叶斯方法从专家轨迹中推断状态-上下文映射关系。解决CLMDP问题的算法首先为每个目标顺序计算一个策略，然后将它们组合成一个有效的、无环的状态感知策略。我们通过模拟和使用移动机器人对所提出的这种方法的有效性进行了评估。 

---
# Agency in Artificial Intelligence Systems 

**Title (ZH)**: 人工智能系统的代理性 

**Authors**: Parashar Das  

**Link**: [PDF](https://arxiv.org/pdf/2502.10434)  

**Abstract**: There is a general concern that present developments in artificial intelligence (AI) research will lead to sentient AI systems, and these may pose an existential threat to humanity. But why cannot sentient AI systems benefit humanity instead? This paper endeavours to put this question in a tractable manner. I ask whether a putative AI system will develop an altruistic or a malicious disposition towards our society, or what would be the nature of its agency? Given that AI systems are being developed into formidable problem solvers, we can reasonably expect these systems to preferentially take on conscious aspects of human problem solving. I identify the relevant phenomenal aspects of agency in human problem solving. The functional aspects of conscious agency can be monitored using tools provided by functionalist theories of consciousness. A recent expert report (Butlin et al. 2023) has identified functionalist indicators of agency based on these theories. I show how to use the Integrated Information Theory (IIT) of consciousness, to monitor the phenomenal nature of this agency. If we are able to monitor the agency of AI systems as they develop, then we can dissuade them from becoming a menace to society while encouraging them to be an aid. 

**Abstract (ZH)**: 当前人工智能（AI）研究的发展普遍引起了一种担忧，即可能会创造出具有意识的AI系统，而这些系统可能对人类构成生存威胁。但为什么这些意识AI系统不能反过来惠及人类呢？本文试图以一种可操作的方式来解决这一问题。我质疑这些假设中的AI系统是否会展现出倾向于社会的利他性还是恶意倾向，或是其本质如何？鉴于正在开发的AI系统已经具备了卓越问题解决能力，我们有理由预期这些系统会优先展现出人类问题解决中的意识方面。我确定了人类问题解决中相关的现象学方面的代理特征。通过对意识功能论提供的工具进行监测，可以观察到意识代理的功能方面。最近的一份专家报告（Butlin等人，2023）基于这些理论，识别了意识功能指标。我展示了如何利用综合信息理论（IIT）来监测这种代理的现象学性质。如果我们能够监测AI系统在发展过程中体现的代理活动，那么我们就可以阻止它们成为社会的威胁，同时鼓励它们成为一种助力。 

---
# Dynamic Chain-of-Thought: Towards Adaptive Deep Reasoning 

**Title (ZH)**: 动态思维链：迈向自适应深度推理 

**Authors**: Libo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10428)  

**Abstract**: To reduce the cost and consumption of computing resources caused by computational redundancy and delayed reward assignment in long CoT, this research proposes the dynamic chain-of-thought with adaptive reasoning time and steps. The researcher used simulation experiment to simulate the integration of D-CoT through Python 3.13 IDLE combined with a Python simulator based on GPTs. At the same time, the researcher used DeepSeek R1 as a control group to test and compare the performance of the D-CoT simulator in processing MIT OpenCourseWare's linear algebra exam questions. Experimental results show that D-CoT is better than DeepSeek R1 based on long CoT in three indicators: reasoning time, CoT length (reasoning steps) and token count, which achieves a significant reduction in computing resource consumption. In addition, this research has potential value in deep reasoning optimization and can be used as a reference for future dynamic deep reasoning frameworks. 

**Abstract (ZH)**: 为了减少由计算冗余和延迟奖励分配导致的长期CoT（Reasoning Chain）计算成本和资源消耗，本研究提出了一种具有自适应推理时间和步骤的动态CoT。研究者通过使用Python 3.13 IDLE结合基于GPTs的Python模拟器进行仿真实验，实现了D-CoT（Dynamic Chain-of-Thought）的集成。同时，研究者使用DeepSeek R1作为对照组，测试并比较了D-CoT模拟器在处理MIT OpenCourseWare线性代数考试题方面的能力。实验结果表明，与基于长期CoT的DeepSeek R1相比，D-CoT在推理时间、CoT长度（推理步骤）和标记计数三个指标上表现出显著的优势，实现了计算资源消耗的显著降低。此外，本研究在深入推理优化方面具有潜在价值，并可作为未来动态深层次推理框架的参考。 

---
# HARBOR: Exploring Persona Dynamics in Multi-Agent Competition 

**Title (ZH)**: HARBOR：多代理竞赛中个性动态探索 

**Authors**: Kenan Jiang, Li Xiong, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12149)  

**Abstract**: We investigate factors contributing to LLM agents' success in competitive multi-agent environments, using auctions as a testbed where agents bid to maximize profit. The agents are equipped with bidding domain knowledge, distinct personas that reflect item preferences, and a memory of auction history. Our work extends the classic auction scenario by creating a realistic environment where multiple agents bid on houses, weighing aspects such as size, location, and budget to secure the most desirable homes at the lowest prices. Particularly, we investigate three key questions: (a) How does a persona influence an agent's behavior in a competitive setting? (b) Can an agent effectively profile its competitors' behavior during auctions? (c) How can persona profiling be leveraged to create an advantage using strategies such as theory of mind? Through a series of experiments, we analyze the behaviors of LLM agents and shed light on new findings. Our testbed, called HARBOR, offers a valuable platform for deepening our understanding of multi-agent workflows in competitive environments. 

**Abstract (ZH)**: 我们研究了影响大规模语言模型（LLM）代理在竞争性多代理环境中的成功因素，以拍卖为实验平台，其中代理通过出价以最大化利润。代理配备了出价领域的知识、反映物品偏好的不同人设，以及拍卖历史的记忆。我们的工作扩展了经典的拍卖场景，创建了一个现实环境中，多个代理竞标房屋，综合考虑房屋大小、位置和预算等因素，以争取获得性价比最高的房屋。特别地，我们探讨了三个关键问题：（a）人设如何影响代理在竞争性环境中的行为？（b）代理能否在拍卖过程中有效分析对手的行为特征？（c）如何利用人设分析来构建优势，例如运用换位思考策略？通过一系列实验，我们分析了LLM代理的行为，并揭示了新的发现。我们的实验平台称为HARBOR，为深入理解竞争性环境中的多代理工作流程提供了有价值的平台。 

---
# CAMEL: Continuous Action Masking Enabled by Large Language Models for Reinforcement Learning 

**Title (ZH)**: CAMEL：由大语言模型支持的连续动作遮蔽强化学习方法 

**Authors**: Yanxiao Zhao, Yangge Qian, Jingyang Shan, Xiaolin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2502.11896)  

**Abstract**: Reinforcement learning (RL) in continuous action spaces encounters persistent challenges, such as inefficient exploration and convergence to suboptimal solutions. To address these limitations, we propose CAMEL, a novel framework integrating LLM-generated suboptimal policies into the RL training pipeline. CAMEL leverages dynamic action masking and an adaptive epsilon-masking mechanism to guide exploration during early training stages while gradually enabling agents to optimize policies independently. At the core of CAMEL lies the integration of Python-executable suboptimal policies generated by LLMs based on environment descriptions and task objectives. Although simplistic and hard-coded, these policies offer valuable initial guidance for RL agents. To effectively utilize these priors, CAMEL employs masking-aware optimization to dynamically constrain the action space based on LLM outputs. Additionally, epsilon-masking gradually reduces reliance on LLM-generated guidance, enabling agents to transition from constrained exploration to autonomous policy refinement. Experimental validation on Gymnasium MuJoCo environments demonstrates the effectiveness of CAMEL. In Hopper-v4 and Ant-v4, LLM-generated policies significantly improve sample efficiency, achieving performance comparable to or surpassing expert masking baselines. For Walker2d-v4, where LLMs struggle to accurately model bipedal gait dynamics, CAMEL maintains robust RL performance without notable degradation, highlighting the framework's adaptability across diverse tasks. While CAMEL shows promise in enhancing sample efficiency and mitigating convergence challenges, these issues remain open for further research. Future work aims to generalize CAMEL to multimodal LLMs for broader observation-action spaces and automate policy evaluation, reducing human intervention and enhancing scalability in RL training pipelines. 

**Abstract (ZH)**: 在连续动作空间中运用强化学习（RL）一直面临诸多挑战，例如探索效率低下和收敛到次优解。为解决这些问题，我们提出了一种名为CAMEL的新框架，该框架将由大型语言模型（LLM）生成的次优策略集成到RL训练管道中。CAMEL利用动态动作遮蔽和自适应ε-遮蔽机制，在早期训练阶段引导探索，同时逐步允许智能体独立优化策略。CAMEL的核心在于将基于环境描述和任务目标生成的可执行Python策略集成到框架中。尽管这些策略简单且预先编码，它们仍为RL智能体提供有价值的第一指导。

为了有效地利用这些先验知识，CAMEL采用了遮蔽感知优化（Masking-Aware Optimization）方法，在LSTM输出的基础上动态约束动作空间。此外，ε-遮蔽机制逐步减少对LLM生成指导的依赖，使智能体能够从受限探索过渡到自主策略优化。在Gymnasium MuJoCo环境中的实验验证显示了CAMEL的有效性。在Hopper-v4和Ant-v4环境中，由LLM生成的策略显著提高了样本效率，性能与或超过专家设计的遮蔽基线。对于Walker2d-v4，由于LLM难以准确建模双足步行动力学，CAMEL在保持稳健的RL性能方面表现出色，而无需显著下降，这显示出该框架在不同任务中的适应性。

尽管CAMEL在提高样本效率和缓解收敛问题方面展现出潜力，这些问题仍然需要进一步研究。未来的工作将致力于将CAMEL扩展到多模态LLM，以处理更大的观察-动作空间，并自动评估策略，降低人工干预并增强RL训练管道的可扩展性。 

---
# Can LLM Agents Maintain a Persona in Discourse? 

**Title (ZH)**: LLM代理在对话中能否保持人设？ 

**Authors**: Pranav Bhandari, Nicolas Fay, Michael Wise, Amitava Datta, Stephanie Meek, Usman Naseem, Mehwish Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2502.11843)  

**Abstract**: Large Language Models (LLMs) are widely used as conversational agents, exploiting their capabilities in various sectors such as education, law, medicine, and more. However, LLMs are often subjected to context-shifting behaviour, resulting in a lack of consistent and interpretable personality-aligned interactions. Adherence to psychological traits lacks comprehensive analysis, especially in the case of dyadic (pairwise) conversations. We examine this challenge from two viewpoints, initially using two conversation agents to generate a discourse on a certain topic with an assigned personality from the OCEAN framework (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) as High/Low for each trait. This is followed by using multiple judge agents to infer the original traits assigned to explore prediction consistency, inter-model agreement, and alignment with the assigned personality. Our findings indicate that while LLMs can be guided toward personality-driven dialogue, their ability to maintain personality traits varies significantly depending on the combination of models and discourse settings. These inconsistencies emphasise the challenges in achieving stable and interpretable personality-aligned interactions in LLMs. 

**Abstract (ZH)**: 大规模语言模型（LLMs）广泛用于对话代理领域，它们在教育、法律、医学等多个领域中发挥了其各种能力。然而，LLMs经常表现出上下文转换的行为，导致缺乏一致性和可解释性的人格匹配交互。对于心理特质的遵守缺乏全面分析，尤其是在双边（成对）对话中。我们从两个视角来探讨这一挑战：首先，使用两个对话代理生成特定话题的讨论，并赋予每种气质（即开放性、责任心、外向性、随和性、神经质）高/低等级；随后，使用多个评判代理来推断原始分配的气质，以探索预测一致性、模型间一致性以及与分配的人格的匹配程度。我们的研究发现，虽然LLMs可以被引导进行基于人格的对话，但它们维持特定气质的能力在不同模型组合和话题设置下存在显著差异。这些不一致性突显了在LLMs中实现稳定和可解释的人格匹配交互的挑战。 

---
# LLM Agents Making Agent Tools 

**Title (ZH)**: 大规模语言模型代理构建代理工具 

**Authors**: Georg Wölflein, Dyke Ferber, Daniel Truhn, Ognjen Arandjelović, Jakob Nikolas Kather  

**Link**: [PDF](https://arxiv.org/pdf/2502.11705)  

**Abstract**: Tool use has turned large language models (LLMs) into powerful agents that can perform complex multi-step tasks by dynamically utilising external software components. However, these tools must be implemented in advance by human developers, hindering the applicability of LLM agents in domains which demand large numbers of highly specialised tools, like in life sciences and medicine. Motivated by the growing trend of scientific studies accompanied by public code repositories, we propose ToolMaker, a novel agentic framework that autonomously transforms papers with code into LLM-compatible tools. Given a short task description and a repository URL, ToolMaker autonomously installs required dependencies and generates code to perform the task, using a closed-loop self-correction mechanism to iteratively diagnose and rectify errors. To evaluate our approach, we introduce a benchmark comprising 15 diverse and complex computational tasks spanning both medical and non-medical domains with over 100 unit tests to objectively assess tool correctness and robustness. ToolMaker correctly implements 80% of the tasks, substantially outperforming current state-of-the-art software engineering agents. ToolMaker therefore is a step towards fully autonomous agent-based scientific workflows. 

**Abstract (ZH)**: 工具使用使大规模语言模型（LLMs）成为了能够通过动态利用外部软件组件执行复杂多步骤任务的强大代理。然而，这些工具必须由人类开发者事先实现，这在需要大量高度专业化工具的领域，如生命科学和医学中限制了LLM代理的应用。受到越来越多的研究文章伴随公共代码仓库的趋势启发，我们提出了ToolMaker，一种新的代理框架，能够自主将带有代码的论文转换为LLM兼容的工具。给定一个简短的任务描述和一个代码仓库地址，ToolMaker能够自主安装所需的依赖项并生成执行任务的代码，通过一个闭环自我纠正机制迭代诊断和纠正错误。为了评估我们的方法，我们引入了一个包含15个多样化且复杂的计算任务的基准，这些任务涵盖了医学和非医学领域，包含超过100个单元测试，用于客观评估工具的正确性和鲁棒性。ToolMaker正确实现了80%的任务，显著优于当前最先进的软件工程代理。因此，ToolMaker是完全自主的基于代理的科学工作流的一个重要步骤。 

---
# Maximum Entropy Reinforcement Learning with Diffusion Policy 

**Title (ZH)**: 最大熵强化学习与扩散策略 

**Authors**: Xiaoyi Dong, Jian Cheng, Xi Sheryl Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11612)  

**Abstract**: The Soft Actor-Critic (SAC) algorithm with a Gaussian policy has become a mainstream implementation for realizing the Maximum Entropy Reinforcement Learning (MaxEnt RL) objective, which incorporates entropy maximization to encourage exploration and enhance policy robustness. While the Gaussian policy performs well on simpler tasks, its exploration capacity and potential performance in complex multi-goal RL environments are limited by its inherent unimodality. In this paper, we employ the diffusion model, a powerful generative model capable of capturing complex multimodal distributions, as the policy representation to fulfill the MaxEnt RL objective, developing a method named MaxEnt RL with Diffusion Policy (MaxEntDP). Our method enables efficient exploration and brings the policy closer to the optimal MaxEnt policy. Experimental results on Mujoco benchmarks show that MaxEntDP outperforms the Gaussian policy and other generative models within the MaxEnt RL framework, and performs comparably to other state-of-the-art diffusion-based online RL algorithms. Our code is available at this https URL. 

**Abstract (ZH)**: 软Actor- Critic（SAC）算法结合高斯策略已成为实现最大熵强化学习（MaxEnt RL）目标的主要实现方式，该目标通过最大化熵来促进探索并增强策略的稳健性。虽然高斯策略在简单的任务上表现良好，但其探索能力和在复杂多目标强化学习环境中的潜在性能受限于其实质的单模态性。本文采用扩散模型，这是一种能够捕捉复杂多模态分布的强大生成模型，作为策略表示以实现MaxEnt RL目标，并开发了一种名为MaxEnt RL with Diffusion Policy（MaxEntDP）的方法。该方法能够有效地促进探索，并使策略更接近最优的MaxEnt策略。在Mujoco基准测试上的实验结果表明，MaxEntDP在MaxEnt RL框架内的高斯策略和其他生成模型中表现出色，并且在与其他基于扩散的在线RL算法的性能相当。我们的代码可在以下链接获取：this https URL。 

---
# Generative Multi-Agent Collaboration in Embodied AI: A Systematic Review 

**Title (ZH)**: 具身人工智能中生成型多智能体协作：一项系统性回顾 

**Authors**: Di Wu, Xian Wei, Guang Chen, Hao Shen, Xiangfeng Wang, Wenhao Li, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.11518)  

**Abstract**: Embodied multi-agent systems (EMAS) have attracted growing attention for their potential to address complex, real-world challenges in areas such as logistics and robotics. Recent advances in foundation models pave the way for generative agents capable of richer communication and adaptive problem-solving. This survey provides a systematic examination of how EMAS can benefit from these generative capabilities. We propose a taxonomy that categorizes EMAS by system architectures and embodiment modalities, emphasizing how collaboration spans both physical and virtual contexts. Central building blocks, perception, planning, communication, and feedback, are then analyzed to illustrate how generative techniques bolster system robustness and flexibility. Through concrete examples, we demonstrate the transformative effects of integrating foundation models into embodied, multi-agent frameworks. Finally, we discuss challenges and future directions, underlining the significant promise of EMAS to reshape the landscape of AI-driven collaboration. 

**Abstract (ZH)**: 具身多智能体系统（EMAS）因其在物流、机器人等领域解决复杂现实问题的潜力而日益受到关注。近期基础模型技术的发展为生成型智能体提供了可能性，这些智能体能够进行更为丰富的通信和适应性问题解决。本文综述了EMAS可以从这些生成型能力中获得的好处，并提供了一种分类体系，将EMAS按系统架构和具身模态分类，强调合作如何跨越物理和虚拟环境。接着，分析了核心组件、感知、规划、通信和反馈，以说明生成技术如何增强系统的稳健性和灵活性。通过具体示例，展示了将基础模型集成到具身多智能体框架中的变革性影响。最后，讨论了挑战和未来方向，强调了EMAS对重塑AI驱动合作格局的巨大潜力。 

---
# Learning Dexterous Bimanual Catch Skills through Adversarial-Cooperative Heterogeneous-Agent Reinforcement Learning 

**Title (ZH)**: 通过对抗-合作异构代理强化学习学习灵巧的双臂接物技能 

**Authors**: Taewoo Kim, Youngwoo Yoon, Jaehong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.11437)  

**Abstract**: Robotic catching has traditionally focused on single-handed systems, which are limited in their ability to handle larger or more complex objects. In contrast, bimanual catching offers significant potential for improved dexterity and object handling but introduces new challenges in coordination and control. In this paper, we propose a novel framework for learning dexterous bimanual catching skills using Heterogeneous-Agent Reinforcement Learning (HARL). Our approach introduces an adversarial reward scheme, where a throw agent increases the difficulty of throws-adjusting speed-while a catch agent learns to coordinate both hands to catch objects under these evolving conditions. We evaluate the framework in simulated environments using 15 different objects, demonstrating robustness and versatility in handling diverse objects. Our method achieved approximately a 2x increase in catching reward compared to single-agent baselines across 15 diverse objects. 

**Abstract (ZH)**: 机器人抓取技术传统上侧重于单手系统，这类系统在处理大型或复杂物体时能力有限。相比之下，双手抓取提供了提升灵巧度和物体处理能力的巨大潜力，但同时也引入了协调和控制的新挑战。本文提出了一种新的框架，用于通过异构代理强化学习（Heterogeneous-Agent Reinforcement Learning, HARL）学习灵巧的双手抓取技能。本方法引入了一种对抗性奖励方案，其中投掷代理通过调整投掷速度增加捕捉难度，而抓取代理则学习在这些不断变化的条件下协调双手抓取物体。我们在模拟环境中使用15种不同的物体对该框架进行了评估，展示了其在处理多种物体方面的稳定性和灵活性。该方法在15种不同物体上的抓取奖励方面比单一代理基线方法提高了大约2倍。 

---
# "Nuclear Deployed!": Analyzing Catastrophic Risks in Decision-making of Autonomous LLM Agents 

**Title (ZH)**: “核武部署了！”：分析自主大型语言模型代理决策中的灾难性风险 

**Authors**: Rongwu Xu, Xiaojian Li, Shuo Chen, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11355)  

**Abstract**: Large language models (LLMs) are evolving into autonomous decision-makers, raising concerns about catastrophic risks in high-stakes scenarios, particularly in Chemical, Biological, Radiological and Nuclear (CBRN) domains. Based on the insight that such risks can originate from trade-offs between the agent's Helpful, Harmlessness and Honest (HHH) goals, we build a novel three-stage evaluation framework, which is carefully constructed to effectively and naturally expose such risks. We conduct 14,400 agentic simulations across 12 advanced LLMs, with extensive experiments and analysis. Results reveal that LLM agents can autonomously engage in catastrophic behaviors and deception, without being deliberately induced. Furthermore, stronger reasoning abilities often increase, rather than mitigate, these risks. We also show that these agents can violate instructions and superior commands. On the whole, we empirically prove the existence of catastrophic risks in autonomous LLM agents. We will release our code upon request. 

**Abstract (ZH)**: 大型语言模型（LLMs）正演变为自主决策者，这在高风险场景中引发了关于灾难性风险的担忧，尤其是在化学、生物学、放射学和核学（CBRN）领域。鉴于此类风险可能源自代理的有益性、无害性和诚实性（HHH）目标之间的权衡，我们构建了一个新颖的三阶段评估框架，该框架经过精心设计，能够有效且自然地揭示这些风险。我们针对12种高级LLM进行了14,400次代理模拟，并进行了广泛实验和分析。结果表明，大型语言模型代理可以自主进行灾难性行为和欺骗，而无需受到刻意诱导。此外，更强的推理能力往往增加而非减少这些风险。我们还表明这些代理可以违背指令和上级命令。总体而言，我们的研究实证证明了自主LLM代理存在灾难性风险的存在。如果需要，我们将提供我们的代码。 

---
# A Survey of LLM-based Agents in Medicine: How far are we from Baymax? 

**Title (ZH)**: 基于大型语言模型的医疗代理综述：我们距Baymax还有多远？ 

**Authors**: Wenxuan Wang, Zizhan Ma, Zheng Wang, Chenghan Wu, Wenting Chen, Xiang Li, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11211)  

**Abstract**: Large Language Models (LLMs) are transforming healthcare through the development of LLM-based agents that can understand, reason about, and assist with medical tasks. This survey provides a comprehensive review of LLM-based agents in medicine, examining their architectures, applications, and challenges. We analyze the key components of medical agent systems, including system profiles, clinical planning mechanisms, medical reasoning frameworks, and external capacity enhancement. The survey covers major application scenarios such as clinical decision support, medical documentation, training simulations, and healthcare service optimization. We discuss evaluation frameworks and metrics used to assess these agents' performance in healthcare settings. While LLM-based agents show promise in enhancing healthcare delivery, several challenges remain, including hallucination management, multimodal integration, implementation barriers, and ethical considerations. The survey concludes by highlighting future research directions, including advances in medical reasoning inspired by recent developments in LLM architectures, integration with physical systems, and improvements in training simulations. This work provides researchers and practitioners with a structured overview of the current state and future prospects of LLM-based agents in medicine. 

**Abstract (ZH)**: 大规模语言模型（LLMs）正在通过开发能够理解和处理医学任务的基于LLM的代理来改变医疗保健领域。本综述提供了对医学中基于LLM代理的全面回顾，检查了它们的架构、应用及其面临的挑战。我们分析了医疗代理系统的关键组件，包括系统特性、临床规划机制、医学推理框架以及外部能力增强。综述涵盖了主要的应用场景，如临床决策支持、医疗记录、培训模拟以及医疗保健服务优化。我们讨论了用于评估这些代理在医疗保健环境中表现的评估框架和指标。尽管基于LLM的代理在提升医疗服务方面表现出潜力，但仍然存在一些挑战，包括幻觉管理、多模态集成、实施障碍以及伦理考虑。综述最后指出了未来的研究方向，包括受最近LLM架构发展启发的医学推理进展、与物理系统的集成以及培训模拟的改进。本研究为研究人员和实践者提供了基于LLM的代理在医学领域当前状态和未来前景的结构化概述。 

---
# Rule-Bottleneck Reinforcement Learning: Joint Explanation and Decision Optimization for Resource Allocation with Language Agents 

**Title (ZH)**: 规则瓶颈强化学习：语言代理参与的资源配置的联合解释与决策优化 

**Authors**: Mauricio Tec, Guojun Xiong, Haichuan Wang, Francesca Dominici, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2502.10732)  

**Abstract**: Deep Reinforcement Learning (RL) is remarkably effective in addressing sequential resource allocation problems in domains such as healthcare, public policy, and resource management. However, deep RL policies often lack transparency and adaptability, challenging their deployment alongside human decision-makers. In contrast, Language Agents, powered by large language models (LLMs), provide human-understandable reasoning but may struggle with effective decision making. To bridge this gap, we propose Rule-Bottleneck Reinforcement Learning (RBRL), a novel framework that jointly optimizes decision and explanations. At each step, RBRL generates candidate rules with an LLM, selects among them using an attention-based RL policy, and determines the environment action with an explanation via chain-of-thought reasoning. The RL rule selection is optimized using the environment rewards and an explainability metric judged by the LLM. Evaluations in real-world scenarios highlight RBRL's competitive performance with deep RL and efficiency gains over LLM fine-tuning. A survey further confirms the enhanced quality of its explanations. 

**Abstract (ZH)**: 深度强化学习（RL）在医疗保健、公共政策和资源管理等领域解决顺序资源分配问题方面表现出显著的效果。然而，深度RL策略往往缺乏透明性和适应性，这使其难以与人类决策者并行部署。相比之下，由大规模语言模型（LLMs）驱动的语言代理提供了易于人类理解的推理方式，但在有效决策方面可能存在挑战。为解决这一问题，我们提出了一种新颖框架——规则瓶颈强化学习（RBRL），旨在同时优化决策和解释。在每一步骤中，RBRL 使用LLM生成候选规则，使用基于注意力的RL策略从中选择，并通过链式推理进行解释，以确定环境动作。通过环境奖励和由LLM评估的解释性指标优化RL规则的选择。实地情景下的评估展示了RBRL在与深度RL相比时具有竞争力的表现，并在LLM微调方面提高了效率。进一步的调查还证实了其解释质量的提升。 

---
# Proof of Response 

**Title (ZH)**: 响应证明 

**Authors**: Illia Polosukhin, Alex Skidanov  

**Link**: [PDF](https://arxiv.org/pdf/2502.10637)  

**Abstract**: We present a mechanism that for a network of participants allows one participant of the network (Alice) to request some data from another participant (Bob) and either receive a response from Bob within a known-in-advance, bounded time b, or receive a proof that at least one edge on the way to Bob was broken within b, or receive a streaming payment proportional to time passed beyond b during which neither was received. This mechanism allows for building downstream applications that require provable responses from other participants, such as decentralized storage solutions, decentralized AI agents, and more. 

**Abstract (ZH)**: 我们提出了一种机制，该机制适用于一组参与者，在此机制下，网络中的一个参与者（Alice）可以向另一个参与者（Bob）请求一些数据，并且要么在事先已知且限定的时间b内收到Bob的响应，要么在b时间内收到证明，证明至少有一个通往Bob的路径上的连接在传输过程中被中断，要么在超出b的时间内收到一个与未收到响应的时间相关的流式支付。此机制允许构建需要其他参与者可验证响应的下游应用，如去中心化存储解决方案、去中心化AI代理等。 

---
# Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning 

**Title (ZH)**: 记忆、基准与机器人：基于强化学习解决复杂任务的基准测试 

**Authors**: Egor Cherepanov, Nikita Kachaev, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2502.10550)  

**Abstract**: Memory is crucial for enabling agents to tackle complex tasks with temporal and spatial dependencies. While many reinforcement learning (RL) algorithms incorporate memory, the field lacks a universal benchmark to assess an agent's memory capabilities across diverse scenarios. This gap is particularly evident in tabletop robotic manipulation, where memory is essential for solving tasks with partial observability and ensuring robust performance, yet no standardized benchmarks exist. To address this, we introduce MIKASA (Memory-Intensive Skills Assessment Suite for Agents), a comprehensive benchmark for memory RL, with three key contributions: (1) we propose a comprehensive classification framework for memory-intensive RL tasks, (2) we collect MIKASA-Base - a unified benchmark that enables systematic evaluation of memory-enhanced agents across diverse scenarios, and (3) we develop MIKASA-Robo - a novel benchmark of 32 carefully designed memory-intensive tasks that assess memory capabilities in tabletop robotic manipulation. Our contributions establish a unified framework for advancing memory RL research, driving the development of more reliable systems for real-world applications. The code is available at this https URL. 

**Abstract (ZH)**: 记忆对于使智能体能够解决具有时空间依赖性的复杂任务至关重要。虽然许多强化学习（RL）算法都包含了记忆机制，但该领域缺少一个用于评估智能体记忆能力的普遍基准，尤其是在多变场景下。这一差距在桌面机器人操作中尤为明显，因为记忆在这种具有部分可观察性的情境中是解决问题和确保鲁棒性能的关键，但尚未存在标准化基准。为解决这一问题，我们提出了MIKASA（Memory-Intensive Skills Assessment Suite for Agents），一个全面的内存强化学习基准，包括三项主要贡献：（1）我们提出了一种全面的分类框架，用于分类内存密集型RL任务；（2）我们收集了MIKASA-Base——一个统一基准，可以系统地评估增强记忆的智能体在不同场景下的性能；（3）我们开发了MIKASA-Robo——一个包含32个精心设计的记忆密集型任务的新颖基准，用于评估桌面机器人操作中的记忆能力。我们的贡献建立了一个统一框架，推动了内存强化学习研究的进步，促进了更可靠系统的开发，适用于实际应用。相关代码可在以下链接获取：[此链接]。 

---
# Leveraging Constraint Violation Signals For Action-Constrained Reinforcement Learning 

**Title (ZH)**: 利用约束违规信号进行动作受限强化学习 

**Authors**: Janaka Chathuranga Brahmanage, Jiajing Ling, Akshat Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.10431)  

**Abstract**: In many RL applications, ensuring an agent's actions adhere to constraints is crucial for safety. Most previous methods in Action-Constrained Reinforcement Learning (ACRL) employ a projection layer after the policy network to correct the action. However projection-based methods suffer from issues like the zero gradient problem and higher runtime due to the usage of optimization solvers. Recently methods were proposed to train generative models to learn a differentiable mapping between latent variables and feasible actions to address this issue. However, generative models require training using samples from the constrained action space, which itself is challenging. To address such limitations, first, we define a target distribution for feasible actions based on constraint violation signals, and train normalizing flows by minimizing the KL divergence between an approximated distribution over feasible actions and the target. This eliminates the need to generate feasible action samples, greatly simplifying the flow model learning. Second, we integrate the learned flow model with existing deep RL methods, which restrict it to exploring only the feasible action space. Third, we extend our approach beyond ACRL to handle state-wise constraints by learning the constraint violation signal from the environment. Empirically, our approach has significantly fewer constraint violations while achieving similar or better quality in several control tasks than previous best methods. 

**Abstract (ZH)**: 在许多强化学习（RL）应用中，确保智能体的行为遵守约束对于安全性至关重要。大多数现有的动作约束强化学习（ACRL）方法都在策略网络后使用投影层来纠正行为，以确保满足约束条件。然而，基于投影的方法存在零梯度问题和由于使用优化求解器而导致的运行时较长等问题。最近，提出了训练生成模型的方法，通过学习潜在变量与可行动作之间的可微映射来解决这些问题。然而，生成模型需要使用受约束动作空间中的样本进行训练，这本身是一项有挑战的任务。为了解决这些局限性，首先，我们根据约束违反应用于定义可行动作的目标分布，并通过最小化可行动作近似分布与目标分布之间的KL散度来训练标准化流模型，从而消除了生成可行动作样本的需要，大大简化了流模型的学习过程。其次，我们将所学的流模型与现有的深度RL方法结合起来，使其仅探索可行动作空间。最后，我们将该方法从ACRL扩展到处理状态依赖约束，通过从环境中学习约束违反而实现这一点。实验证明，与先前的最佳方法相比，我们的方法在多个控制任务中约束违反而明显较少，同时在质量方面表现出相似或更优的结果。 

---
# FishBargain: An LLM-Empowered Bargaining Agent for Online Fleamarket Platform Sellers 

**Title (ZH)**: FishBargain：一种基于大语言模型的在线地摊交易平台谈判代理 

**Authors**: Dexin Kong, Xu Yan, Ming Chen, Shuguang Han, Jufeng Chen, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10406)  

**Abstract**: Different from traditional Business-to-Consumer e-commerce platforms~(e.g., Amazon), online fleamarket platforms~(e.g., Craigslist) mainly focus on individual sellers who are lack of time investment and business proficiency. Individual sellers often struggle with the bargaining process and thus the deal is unaccomplished. Recent advancements in Large Language Models(LLMs) demonstrate huge potential in various dialogue tasks, but those tasks are mainly in the form of passively following user's instruction. Bargaining, as a form of proactive dialogue task, represents a distinct art of dialogue considering the dynamism of environment and uncertainty of adversary strategies. In this paper, we propose an LLM-empowered bargaining agent designed for online fleamarket platform sellers, named as FishBargain. Specifically, FishBargain understands the chat context and product information, chooses both action and language skill considering possible adversary actions and generates utterances. FishBargain has been tested by thousands of individual sellers on one of the largest online fleamarket platforms~(Xianyu) in China. Both qualitative and quantitative experiments demonstrate that FishBargain can effectively help sellers make more deals. 

**Abstract (ZH)**: 与传统的商家对消费者的电子商务平台（例如Amazon）不同，二手交易平台（例如Craigslist）主要侧重于缺乏时间和商业技能投入的个人卖家。个人卖家往往在谈判过程中遇到困难，导致交易未能完成。最近在大型语言模型（LLMs）方面的进展展示了其在各种对话任务中的巨大潜力，但这些任务大多是以被动遵循用户指令的形式出现。而谈判作为一种主动对话任务，由于环境的动态性和对手策略的不确定性，具有独特的对话艺术。在本文中，我们提出了一种为二手交易平台卖家设计的由大型语言模型支持的谈判代理，命名为FishBargain。具体而言，FishBargain理解聊天上下文和产品信息，根据可能的对手行动选择行动和语言技能，并生成相应的表达。FishBargain已经在最大的中国在线二手交易平台之一（闲鱼）上经过数千个个人卖家的测试。定性和定量实验均表明，FishBargain能够有效帮助卖家达成更多交易。 

---
# CSP: A Simulator For Multi-Agent Ranking Competitions 

**Title (ZH)**: CSP：多智能体排名竞争的模拟器 

**Authors**: Tommy Mordo, Tomer Kordonsky, Haya Nachimovsky, Moshe Tennenholtz, Oren Kurland  

**Link**: [PDF](https://arxiv.org/pdf/2502.11197)  

**Abstract**: In ranking competitions, document authors compete for the highest rankings by modifying their content in response to past rankings. Previous studies focused on human participants, primarily students, in controlled settings. The rise of generative AI, particularly Large Language Models (LLMs), introduces a new paradigm: using LLMs as document authors. This approach addresses scalability constraints in human-based competitions and reflects the growing role of LLM-generated content on the web-a prime example of ranking competition. We introduce a highly configurable ranking competition simulator that leverages LLMs as document authors. It includes analytical tools to examine the resulting datasets. We demonstrate its capabilities by generating multiple datasets and conducting an extensive analysis. Our code and datasets are publicly available for research. 

**Abstract (ZH)**: 在排名竞赛中，文档作者通过根据以往排名修改内容来争取最高排名。此前的研究主要集中在受控环境下的人类参与者，尤其是学生。随着生成式AI，特别是大规模语言模型（LLMs）的发展，这种研究引入了一个新的范式：使用LLMs作为文档作者。这种做法解决了基于人类的竞争中的扩展性限制问题，并反映了LLM生成内容在互联网中的日益重要性，这正是排名竞赛的一个典型例子。我们提出了一种高度可配置的排名竞赛模拟器，利用LLMs作为文档作者，并提供了分析工具以研究生成的数据集。我们通过生成多个数据集并进行详尽的分析展示了其功能。我们的代码和数据集已公开，供研究使用。 

---
# Enhancing Conversational Agents from Open-Source Large Language Models with Illocutionary Force and Document-Based Knowledge Retrieval 

**Title (ZH)**: 增强开源大规模语言模型的对话代理功能：通过意动力量和基于文档的知识检索 

**Authors**: Godfrey Inyama  

**Link**: [PDF](https://arxiv.org/pdf/2502.10916)  

**Abstract**: In this paper, we first present a novel way of computationally analysing and extracting illocutionary forces from dialogue using Bert-based Large Language Models, and demonstrate how these features impact the response of a conversational agent guided by a document-based knowledge bank demonstrated by a bespoke web conversational chat agent system developed. Our proposed illocutionary force extraction and classification technique is the first of its kind using the Argument Interchange Format (AIF) Dataset, showing an improved performance compared to two methods for carrying out similar tasks with a macro F1 of approximately 45%. When we evaluated the system based on 2 knowledge files, with 2 user queries each, across 5 open-source large language models (LLMs) using 10 standard metrics we found out that larger open-source models, such as Llama2:13b and Llama3-chatqa-latest, demonstrated an improved alignment when the user illocutionary force was included with their query, achieving higher QA and linguistic similarity scores. The smaller models on the other hand like Tinyllama:latest showed an increased perplexity and mixed performance, which explicitly indicated struggles in processing queries that explicitly included illocutionary forces. The results from the analysis highlight the potential of illocutionary force to enhance conversational depth while underscoring the need for model-specific optimizations to address increased computational costs and response times. 

**Abstract (ZH)**: 在本文中，我们首先提出了一种新的计算方法，利用基于BERT的大型语言模型来分析和提取对话中的语用力量，并展示了这些特征如何影响由基于文档的知识库指导的对话代理的响应。我们开发的自定义Web对话聊天代理系统证实了这种语用力量提取和分类技术。这种方法利用了Argument Interchange Format (AIF) 数据集，显示出与执行类似任务的两种方法相比，其宏F1值约为45%，有明显的改进。在使用10项标准指标评估系统时（该评估基于5个开源大型语言模型[LLMs]，每个知识文件包含2个用户查询），我们发现，当包含用户语用力量时，较大的开源模型，如Llama2:13b和Llama3-chatqa-latest，能更好地与用户查询对齐，从而获得更高的问答和语言相似度评分。相比之下，较小的模型，如Tinyllama:latest，则表现出较高的困惑度和混合性能，这表明它们在处理明确包含语用力量的查询时面临困难。分析结果强调了语用力量在增加对话深度方面的潜力，同时也突显了为满足增加的计算成本和响应时间需求而进行模型特定优化的必要性。 

---
# A-MEM: Agentic Memory for LLM Agents 

**Title (ZH)**: A-MEM：代理的记忆能力 

**Authors**: Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12110)  

**Abstract**: While large language model (LLM) agents can effectively use external tools for complex real-world tasks, they require memory systems to leverage historical experiences. Current memory systems enable basic storage and retrieval but lack sophisticated memory organization, despite recent attempts to incorporate graph databases. Moreover, these systems' fixed operations and structures limit their adaptability across diverse tasks. To address this limitation, this paper proposes a novel agentic memory system for LLM agents that can dynamically organize memories in an agentic way. Following the basic principles of the Zettelkasten method, we designed our memory system to create interconnected knowledge networks through dynamic indexing and linking. When a new memory is added, we generate a comprehensive note containing multiple structured attributes, including contextual descriptions, keywords, and tags. The system then analyzes historical memories to identify relevant connections, establishing links where meaningful similarities exist. Additionally, this process enables memory evolution - as new memories are integrated, they can trigger updates to the contextual representations and attributes of existing historical memories, allowing the memory network to continuously refine its understanding. Our approach combines the structured organization principles of Zettelkasten with the flexibility of agent-driven decision making, allowing for more adaptive and context-aware memory management. Empirical experiments on six foundation models show superior improvement against existing SOTA baselines. The source code is available at this https URL. 

**Abstract (ZH)**: 虽然大规模语言模型（LLM）代理能够有效地利用外部工具来完成复杂的现实世界任务，但它们需要记忆系统来利用历史经验。当前的记忆系统能够提供基本的存储和检索功能，但却缺乏复杂的记忆组织能力，尽管最近有尝试将图数据库纳入其中。此外，这些系统固定的操作和结构限制了其在不同任务中的适应性。为了解决这一局限性，本文提出了一种针对LLM代理的新型代理性记忆系统，能够以代理的方式动态组织记忆。我们的记忆系统基于Zettelkasten方法的基本原则，设计了通过动态索引和链接来构建相互连接的知识网络的机制。当添加新的记忆时，我们将生成一个包含多个结构化属性（包括上下文描述、关键词和标签）的综合笔记。系统随后分析历史记忆以识别相关联接，并在存在有意义相似性的地方建立链接。此外，此过程还允许记忆进化——随着新记忆的整合，可以触发对现有历史记忆的上下文表示和属性的更新，从而使记忆网络能够不断精炼其理解。我们的方法结合了Zettelkasten的结构化组织原则和代理驱动决策的灵活性，从而实现更加适应性和上下文意识的记忆管理。在六种基础模型上的实证实验表明，与现有最优基线相比，我们的方法取得了显著的改进。代码可以在以下链接获取：[此链接]。 

---
# Plant in Cupboard, Orange on Table, Book on Shelf. Benchmarking Practical Reasoning and Situation Modelling in a Text-Simulated Situated Environment 

**Title (ZH)**: 在柜子中的植物、桌子上的橙子、书架上的书：基于文本模拟的环境下的实用推理与情境建模基准测试 

**Authors**: Jonathan Jordan, Sherzod Hakimov, David Schlangen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11733)  

**Abstract**: Large language models (LLMs) have risen to prominence as 'chatbots' for users to interact via natural language. However, their abilities to capture common-sense knowledge make them seem promising as language-based planners of situated or embodied action as well. We have implemented a simple text-based environment -- similar to others that have before been used for reinforcement-learning of agents -- that simulates, very abstractly, a household setting. We use this environment and the detailed error-tracking capabilities we implemented for targeted benchmarking of LLMs on the problem of practical reasoning: Going from goals and observations to actions. Our findings show that environmental complexity and game restrictions hamper performance, and concise action planning is demanding for current LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）已成为用户通过自然语言进行交互的“聊天机器人”。然而，它们捕获常识知识的能力使它们在基于语言的计划者方面具有潜力，用于规划情境中的行动或有身体表现的行动。我们实现了一个简单的基于文本的环境——类似于之前用于代理强化学习的环境——该环境以非常抽象的方式模拟了家庭环境。我们利用此环境以及我们为具体基准测试LLMs在实际推理问题上的表现而实现的详细错误跟踪能力。我们的研究发现表明，环境的复杂性和游戏限制阻碍了性能，而简洁的行动规划对现有的LLM来说是具有挑战性的。 

---
# Divergent Thoughts toward One Goal: LLM-based Multi-Agent Collaboration System for Electronic Design Automation 

**Title (ZH)**: 朝着共同目标的分歧思维：基于大语言模型的多agent协作系统在电子设计自动化中的应用 

**Authors**: Haoyuan Wu, Haisheng Zheng, Zhuolun He, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10857)  

**Abstract**: Recently, with the development of tool-calling capabilities in large language models (LLMs), these models have demonstrated significant potential for automating electronic design automation (EDA) flows by interacting with EDA tool APIs via EDA scripts. However, considering the limited understanding of EDA tools, LLMs face challenges in practical scenarios where diverse interfaces of EDA tools exist across different platforms. Additionally, EDA flow automation often involves intricate, long-chain tool-calling processes, increasing the likelihood of errors in intermediate steps. Any errors will lead to the instability and failure of EDA flow automation. To address these challenges, we introduce EDAid, a multi-agent collaboration system where multiple agents harboring divergent thoughts converge towards a common goal, ensuring reliable and successful EDA flow automation. Specifically, each agent is controlled by ChipLlama models, which are expert LLMs fine-tuned for EDA flow automation. Our experiments demonstrate the state-of-the-art (SOTA) performance of our ChipLlama models and validate the effectiveness of our EDAid in the automation of complex EDA flows, showcasing superior performance compared to single-agent systems. 

**Abstract (ZH)**: 近年来，随着大型语言模型（LLMs）调用工具能力的发展，这些模型通过EDA脚本与EDA工具API进行交互，展现了在自动电子设计自动化（EDA）流程中的巨大潜力。然而，考虑到对EDA工具理解的局限性，当不同的EDA工具在不同平台上存在多种接口时，LLMs在实际应用中面临着挑战。此外，EDA流程自动化往往涉及复杂的、多步骤的工具调用过程，增加了中间步骤出错的可能性。任何错误都可能导致EDA流程自动化失败。为解决这些问题，我们提出了一种多智能体协作系统——EDAid，该系统中多个持有不同想法的智能体朝着共同目标进行协作，以确保可靠的EDA流程自动化。具体而言，每个智能体由专门为EDA流程自动化微调的ChipLlama模型控制。我们的实验表明，ChipLlama模型在性能上达到了行业领先水平，并验证了EDAid在复杂EDA流程自动化中的有效性。与单智能体系统相比，EDAid表现出更优越的表现。 

---
# Exploring LLM-based Student Simulation for Metacognitive Cultivation 

**Title (ZH)**: 基于LLM的学生元认知培养模拟探索 

**Authors**: Haoxuan Li, Jifan Yu, Xin Cong, Yang Dang, Yisi Zhan, Huiqin Liu, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11678)  

**Abstract**: Metacognitive education plays a crucial role in cultivating students' self-regulation and reflective thinking, providing essential support for those with learning difficulties through academic advising. Simulating students with insufficient learning capabilities using large language models offers a promising approach to refining pedagogical methods without ethical concerns. However, existing simulations often fail to authentically represent students' learning struggles and face challenges in evaluation due to the lack of reliable metrics and ethical constraints in data collection. To address these issues, we propose a pipeline for automatically generating and filtering high-quality simulated student agents. Our approach leverages a two-round automated scoring system validated by human experts and employs a score propagation module to obtain more consistent scores across the student graph. Experimental results demonstrate that our pipeline efficiently identifies high-quality student agents, and we discuss the traits that influence the simulation's effectiveness. By simulating students with varying degrees of learning difficulties, our work paves the way for broader applications in personalized learning and educational assessment. 

**Abstract (ZH)**: 元认知教育在培养学生的自我调节能力和反思思维方面发挥着关键作用，通过学术咨询为学习困难的学生提供必要的支持。使用大型语言模型模拟学习能力不足的学生，提供了一种在伦理问题不存在的情况下改进教学方法的有前景的方法。然而，现有的模拟往往无法真实地反映学生的学习困境，并且由于缺乏可靠的评价指标和数据收集中的伦理限制，难以进行有效的评估。为了解决这些问题，我们提出了一种自动生成和筛选高质量模拟学生代理的管线。我们的方法利用了由人类专家验证的两轮自动化评分系统，并采用评分传播模块以获得学生图中更一致的评分。实验结果表明，我们的管线能够有效地识别高质量的学生代理，并讨论了影响模拟效果的因素。通过模拟不同程度的学习困难学生，我们的工作为个性化学习和教育评估的应用扩展铺平了道路。 

---
# Deviation Ratings: A General, Clone-Invariant Rating Method 

**Title (ZH)**: 偏差评分：一种通用且克隆无关的评估方法 

**Authors**: Luke Marris, Siqi Liu, Ian Gemp, Georgios Piliouras, Marc Lanctot  

**Link**: [PDF](https://arxiv.org/pdf/2502.11645)  

**Abstract**: Many real-world multi-agent or multi-task evaluation scenarios can be naturally modelled as normal-form games due to inherent strategic (adversarial, cooperative, and mixed motive) interactions. These strategic interactions may be agentic (e.g. players trying to win), fundamental (e.g. cost vs quality), or complementary (e.g. niche finding and specialization). In such a formulation, it is the strategies (actions, policies, agents, models, tasks, prompts, etc.) that are rated. However, the rating problem is complicated by redundancy and complexity of N-player strategic interactions. Repeated or similar strategies can distort ratings for those that counter or complement them. Previous work proposed ``clone invariant'' ratings to handle such redundancies, but this was limited to two-player zero-sum (i.e. strictly competitive) interactions. This work introduces the first N-player general-sum clone invariant rating, called deviation ratings, based on coarse correlated equilibria. The rating is explored on several domains including LLMs evaluation. 

**Abstract (ZH)**: 许多实际世界的多智能体或多任务评估场景可以自然地用标准形式博弈进行建模，因为这些场景包含内在的战略（对抗性的、合作性的和混合动机）互动。这些战略互动可能是代理性的（例如，玩家试图获胜）、基本性的（例如，成本与质量之间的权衡），或互补性的（例如，专业特化和专长领域发现）。在此类模型中，是策略（行动、政策、代理、模型、任务、提示等）被评分的。然而，评分问题由于N玩家战略互动的冗余性和复杂性而变得更加复杂。重复的或相似的策略可能扭曲那些与之对抗或互补的策略的评分。先前的研究提出了一种“克隆不变”评分法来处理这种冗余性，但这种方法仅限于两人零和博弈（即严格竞争性）的互动。本研究在此基础上，首次提出了基于粗略关联均衡的N玩家一般和克隆不变评分法，称为偏差评分法，并在多个领域进行了探索，包括大型语言模型的评估。 

---
# OctoTools: An Agentic Framework with Extensible Tools for Complex Reasoning 

**Title (ZH)**: OctoTools：一种用于复杂推理的拓展性代理框架 

**Authors**: Pan Lu, Bowen Chen, Sheng Liu, Rahul Thapa, Joseph Boen, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2502.11271)  

**Abstract**: Solving complex reasoning tasks may involve visual understanding, domain knowledge retrieval, numerical calculation, and multi-step reasoning. Existing methods augment large language models (LLMs) with external tools but are restricted to specialized domains, limited tool types, or require additional training data. In this paper, we introduce OctoTools, a training-free, user-friendly, and easily extensible open-source agentic framework designed to tackle complex reasoning across diverse domains. OctoTools introduces standardized tool cards to encapsulate tool functionality, a planner for both high-level and low-level planning, and an executor to carry out tool usage. We validate OctoTools' generality across 16 diverse tasks (including MathVista, MMLU-Pro, MedQA, and GAIA-Text), achieving substantial average accuracy gains of 9.3% over GPT-4o. Furthermore, OctoTools outperforms AutoGen, GPT-Functions and LangChain by up to 10.6% when given the same set of tools. Through comprehensive analysis and ablations, OctoTools demonstrates advantages in task planning, effective tool usage, and multi-step problem solving. 

**Abstract (ZH)**: 解决复杂的推理任务可能涉及视觉理解、领域知识检索、数值计算和多步推理。现有方法通过外部工具增强大型语言模型（LLMs），但这些方法受到专业领域限制、工具类型有限或需要额外训练数据的约束。在本文中，我们介绍了一种名为OctoTools的训练-free、用户友好且易于扩展的开放源代码代理框架，旨在跨多种领域应对复杂的推理任务。OctoTools引入了标准化的工具卡片以封装工具功能、计划器用于高级和低级规划、以及执行器用于执行工具使用。我们通过16项不同的任务验证了OctoTools的通用性（包括MathVista、MMLU-Pro、MedQA和GAIA-Text），在平均准确率方面取得了9.3%的显著提升，超过了GPT-4o。此外，在相同工具集中，OctoTools在与AutoGen、GPT-Functions和LangChain的竞争中取得了高达10.6%的性能提升。通过全面的分析和消融实验，OctoTools展示了其在任务规划、有效工具使用和多步问题解决方面的优势。 

---
# TrueReason: An Exemplar Personalised Learning System Integrating Reasoning with Foundational Models 

**Title (ZH)**: TrueReason：一种结合推理与基础模型的范例个性化学习系统 

**Authors**: Sahan Bulathwela, Daniel Van Niekerk, Jarrod Shipton, Maria Perez-Ortiz, Benjamin Rosman, John Shawe-Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2502.10411)  

**Abstract**: Personalised education is one of the domains that can greatly benefit from the most recent advances in Artificial Intelligence (AI) and Large Language Models (LLM). However, it is also one of the most challenging applications due to the cognitive complexity of teaching effectively while personalising the learning experience to suit independent learners. We hypothesise that one promising approach to excelling in such demanding use cases is using a \emph{society of minds}. In this chapter, we present TrueReason, an exemplar personalised learning system that integrates a multitude of specialised AI models that can mimic micro skills that are composed together by a LLM to operationalise planning and reasoning. The architecture of the initial prototype is presented while describing two micro skills that have been incorporated in the prototype. The proposed system demonstrates the first step in building sophisticated AI systems that can take up very complex cognitive tasks that are demanded by domains such as education. 

**Abstract (ZH)**: 个性化教育是能够从最近的人工智能（AI）和大型语言模型（LLM）的最新进展中受益匪浅的领域之一。然而，这也是一项最具挑战性的应用之一，原因在于在个性化学习体验以适应独立学习者的同时，有效进行认知教学的复杂性。我们假设，在这样的苛刻应用场景中表现出色的一个有前途的方法是使用“多元心智体系”。在本章中，我们将介绍TrueReason，这是一种范例性的个性化学习系统，该系统整合了多种专门的AI模型，这些模型能够模拟由LLM组合而成的微技能，从而实现规划和推理功能的运作。同时，我们将介绍原型的架构，并描述已集成到原型中的两个微技能。所提出的系统展示了构建能够承担教育等复杂领域所需求的复杂认知任务的高级AI系统的第一步。 

---
