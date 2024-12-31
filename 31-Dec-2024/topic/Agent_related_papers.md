# Exploring and Controlling Diversity in LLM-Agent Conversation 

**Title (ZH)**: 探索和控制大规模语言模型-代理对话中的多样性 

**Authors**: KuanChao Chu, Yi-Pei Chen, Hideki Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2412.21102)  

**Abstract**: Diversity is a critical aspect of multi-agent communication. In this paper, we focus on controlling and exploring diversity in the context of open-domain multi-agent conversations, particularly for world simulation applications. We propose Adaptive Prompt Pruning (APP), a novel method that dynamically adjusts the content of the utterance generation prompt to control diversity using a single parameter, lambda. Through extensive experiments, we show that APP effectively controls the output diversity across models and datasets, with pruning more information leading to more diverse output. We comprehensively analyze the relationship between prompt content and conversational diversity. Our findings reveal that information from all components of the prompt generally constrains the diversity of the output, with the Memory block exerting the most significant influence. APP is compatible with established techniques like temperature sampling and top-p sampling, providing a versatile tool for diversity management. To address the trade-offs of increased diversity, such as inconsistencies with omitted information, we incorporate a post-generation correction step, which effectively balances diversity enhancement with output consistency. Additionally, we examine how prompt structure, including component order and length, impacts diversity. This study addresses key questions surrounding diversity in multi-agent world simulation, offering insights into its control, influencing factors, and associated trade-offs. Our contributions lay the foundation for systematically engineering diversity in LLM-based multi-agent collaborations, advancing their effectiveness in real-world applications. 

**Abstract (ZH)**: 多样性是多agents通信的一个关键方面。本文重点探讨在开放领域多agents对话中控制和探索多样性，特别是在世界模拟应用程序中的应用。我们提出了一种名为自适应提示修剪（APP）的新方法，该方法通过调整单个参数lambda来动态调整生成语句的提示内容，从而控制多样性。通过广泛的实验，我们证明APP能够在不同模型和数据集中有效地控制输出多样性，修剪更多的信息会导致更广泛的输出。我们全面分析了提示内容与对话多样性的关系。研究结果表明，提示中所有部分的信息通常都会限制输出的多样性，而Memory块的影响最为显著。APP与现有的技术和方法（如温度采样和top-p采样）兼容，提供了一种广泛适用的多样性管理工具。为应对增加多样性所带来的矛盾，如遗漏信息导致的一致性问题，我们引入了事后生成校正步骤，这有效地平衡了多样性提升与输出一致性之间的关系。此外，我们还研究了提示结构（包括各组件的顺序和长度）对多样性的影响。本研究针对多agents世界模拟中多样性控制的关键问题提供了解析，并探讨了其影响因素及其权衡。我们提出的贡献为系统地构建基于LLM的多agents合作中的多样性奠定了基础，提高了其在实际应用中的效果。 

---
# Plancraft: an evaluation dataset for planning with LLM agents 

**Title (ZH)**: PlanCraft：用于评估基于LLM代理的规划数据集 

**Authors**: Gautier Dagan, Frank Keller, Alex Lascarides  

**Link**: [PDF](https://arxiv.org/pdf/2412.21033)  

**Abstract**: We present Plancraft, a multi-modal evaluation dataset for LLM agents. Plancraft has both a text-only and multi-modal interface, based on the Minecraft crafting GUI. We include the Minecraft Wiki to evaluate tool use and Retrieval Augmented Generation (RAG), as well as an oracle planner and oracle RAG information extractor, to ablate the different components of a modern agent architecture. To evaluate decision-making, Plancraft also includes a subset of examples that are intentionally unsolvable, providing a realistic challenge that requires the agent not only to complete tasks but also to decide whether they are solvable at all. We benchmark both open-source and closed-source LLMs and strategies on our task and compare their performance to a handcrafted planner. We find that LLMs and VLMs struggle with the planning problems that Plancraft introduces, and we offer suggestions on how to improve their capabilities. 

**Abstract (ZH)**: 我们介绍了Plancraft，这是一个针对大规模语言模型（LLM）代理的跨模态评估数据集。Plancraft 支持文本-only 和跨模态两种界面，基于《我的世界》（Minecraft）的制作用户界面（GUI）。我们包含了《我的世界》维基，用于评估工具使用和检索增强生成（RAG），同时也提供了一个 oracle 计划者和 oracle RAG 信息提取器，以消除现代代理架构中不同组件的影响。为了评估决策能力，Plancraft 还包括了一部分故意无法解决的示例，这些示例为代理提供了真实的挑战，不仅需要代理完成任务，还需要代理判断这些任务是否可解。我们对开源和封闭源代码的 LLM 和策略进行了基准测试，并将它们的表现与手工设计的计划者进行了比较。我们发现，LLM 和视觉-语言模型在处理 Plancraft 引入的规划问题时表现不佳，并提出了提高其能力的建议。 

---
# Efficient Multi-Agent Collaboration with Tool Use for Online Planning in Complex Table Question Answering 

**Title (ZH)**: 高效的多代理协作与工具使用在复杂表格问答中的在线规划 

**Authors**: Wei Zhou, Mohsen Mesgar, Annemarie Friedrich, Heike Adel  

**Link**: [PDF](https://arxiv.org/pdf/2412.20145)  

**Abstract**: Complex table question answering (TQA) aims to answer questions that require complex reasoning, such as multi-step or multi-category reasoning, over data represented in tabular form. Previous approaches demonstrated notable performance by leveraging either closed-source large language models (LLMs) or fine-tuned open-weight LLMs. However, fine-tuning LLMs requires high-quality training data, which is costly to obtain, and utilizing closed-source LLMs poses accessibility challenges and leads to reproducibility issues. In this paper, we propose Multi-Agent Collaboration with Tool use (MACT), a framework that requires neither closed-source models nor fine-tuning. In MACT, a planning agent and a coding agent that also make use of tools collaborate to answer questions. Our experiments on four TQA benchmarks show that MACT outperforms previous SoTA systems on three out of four benchmarks and that it performs comparably to the larger and more expensive closed-source model GPT-4 on two benchmarks, even when using only open-weight models without any fine-tuning. We conduct extensive analyses to prove the effectiveness of MACT's multi-agent collaboration in TQA. 

**Abstract (ZH)**: 复杂表格问答（TQA）的目标是通过对以表格形式表示的数据进行多步或多类别推理以回答问题。之前的方法通过利用闭源大型语言模型（LLMs）或微调的开源权重LLMs展现了显著的表现。然而，微调LLMs需要高质量的训练数据，这往往成本高昂，而利用闭源LLMs则带来了访问性和可再现性的问题。在本文中，我们提出了多agent协作与工具使用（MACT）框架，该框架既不需要闭源模型，也不需要微调。在MACT中，规划agent和编写agent通过使用工具协同工作以回答问题。我们在四个TQA基准上的实验表明，MACT在三个基准上优于之前的SOTA系统，并且在两个基准上与更大且更昂贵的闭源模型GPT-4具有相似的表现，即使仅使用未经过微调的开源模型也是如此。我们进行了广泛的分析以证明MACT在TQA中的多agent协作的有效性。 

---
# OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System 

**Title (ZH)**: OneKE：一种基于模式指导的LLM代理知识提取系统（Docker 化版本） 

**Authors**: Yujie Luo, Xiangyuan Ru, Kangwei Liu, Lin Yuan, Mengshu Sun, Ningyu Zhang, Lei Liang, Zhiqiang Zhang, Jun Zhou, Lanning Wei, Da Zheng, Haofen Wang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.20005)  

**Abstract**: We introduce OneKE, a dockerized schema-guided knowledge extraction system, which can extract knowledge from the Web and raw PDF Books, and support various domains (science, news, etc.). Specifically, we design OneKE with multiple agents and a configure knowledge base. Different agents perform their respective roles, enabling support for various extraction scenarios. The configure knowledge base facilitates schema configuration, error case debugging and correction, further improving the performance. Empirical evaluations on benchmark datasets demonstrate OneKE's efficacy, while case studies further elucidate its adaptability to diverse tasks across multiple domains, highlighting its potential for broad applications. We have open-sourced the Code at this https URL and released a Video at this http URL. 

**Abstract (ZH)**: 我们介绍了OneKE，这是一个容器化的基于模式的知识提取系统，能够从网页和原始PDF书籍中提取知识，并支持多种领域（如科学、新闻等）。具体而言，我们设计了OneKE，使其包含多个代理和一个配置的知识库。各个代理承担不同的角色，从而支持多种提取场景。配置的知识库有助于模式配置、错误案例调试和修正，进而提高系统的性能。基准数据集上的实证评估展示了OneKE的有效性，而案例研究进一步阐明了其在多个领域的多种任务中适应性的特点，突显了其广泛的应用潜力。我们已将源代码开源发布在 <https://this-url.com/>，并在 <https://this-url.com/> 上传了演示视频。 

---
# Distributed Mixture-of-Agents for Edge Inference with Large Language Models 

**Title (ZH)**: 基于边缘推理的大规模语言模型混合代理分布式系统 

**Authors**: Purbesh Mitra, Priyanka Kaswan, Sennur Ulukus  

**Link**: [PDF](https://arxiv.org/pdf/2412.21200)  

**Abstract**: Mixture-of-Agents (MoA) has recently been proposed as a method to enhance performance of large language models (LLMs), enabling multiple individual LLMs to work together for collaborative inference. This collaborative approach results in improved responses to user prompts compared to relying on a single LLM. In this paper, we consider such an MoA architecture in a distributed setting, where LLMs operate on individual edge devices, each uniquely associated with a user and equipped with its own distributed computing power. These devices exchange information using decentralized gossip algorithms, allowing different device nodes to talk without the supervision of a centralized server. In the considered setup, different users have their own LLM models to address user prompts. Additionally, the devices gossip either their own user-specific prompts or augmented prompts to generate more refined answers to certain queries. User prompts are temporarily stored in the device queues when their corresponding LLMs are busy. Given the memory limitations of edge devices, it is crucial to ensure that the average queue sizes in the system remain bounded. In this paper, we address this by theoretically calculating the queuing stability conditions for the device queues under reasonable assumptions, which we validate experimentally as well. Further, we demonstrate through experiments, leveraging open-source LLMs for the implementation of distributed MoA, that certain MoA configurations produce higher-quality responses compared to others, as evaluated on AlpacaEval 2.0 benchmark. The implementation is available at: this https URL. 

**Abstract (ZH)**: 以下是对这段内容的翻译，符合学术规范：

Mixture-of-Agents (MoA) 近期被提出作为一种增强大规模语言模型 (LLMs) 性能的方法，使多个独立的 LLM 能够协同工作进行联合推理。这种协同方法使得用户提示的响应效果优于仅依赖单一 LLM。在本文中，我们考虑在分布式环境中 MoA 架构的应用，其中 LLM 在个体边缘设备上运行，每台设备都唯一关联于一个用户，并配备了独立的分布式计算能力。这些设备通过去中心化的闲聊算法 (gossip algorithms) 交换信息，允许不同的设备节点在无需中央服务器监督的情况下相互交流。在这个设置中，不同用户都有各自的 LLM 模型来处理用户提示。此外，设备相互闲聊时可能传播各自特定用户的提示或增强后的提示，以生成更精细的回答来解决特定查询。当用户提示对应的 LLM 正忙时，这些提示将被暂时存储在设备队列中。鉴于边缘设备的内存限制，确保系统中平均队列大小保持在界限内至关重要。本文通过在合理假设下理论计算设备队列的排队稳定条件，并通过实验验证了这些条件。此外，我们通过实验展示了，在利用开源 LLM 实现分布式 MoA 的情况下，某些 MoA 配置相比其他配置在 AlpacaEval 2.0 基准上的响应质量更高。该实现可参见：this [网站链接]。

请注意将"this https URL"替换为实际的访问链接。 

---
# Aviary: training language agents on challenging scientific tasks 

**Title (ZH)**: Aviary：在具有挑战性的科学任务中训练语言代理 

**Authors**: Siddharth Narayanan, James D. Braza, Ryan-Rhys Griffiths, Manu Ponnapati, Albert Bou, Jon Laurent, Ori Kabeli, Geemi Wellawatte, Sam Cox, Samuel G. Rodriques, Andrew D. White  

**Link**: [PDF](https://arxiv.org/pdf/2412.21154)  

**Abstract**: Solving complex real-world tasks requires cycles of actions and observations. This is particularly true in science, where tasks require many cycles of analysis, tool use, and experimentation. Language agents are promising for automating intellectual tasks in science because they can interact with tools via natural language or code. Yet their flexibility creates conceptual and practical challenges for software implementations, since agents may comprise non-standard components such as internal reasoning, planning, tool usage, as well as the inherent stochasticity of temperature-sampled language models. Here, we introduce Aviary, an extensible gymnasium for language agents. We formalize agents as policies solving language-grounded partially observable Markov decision processes, which we term language decision processes. We then implement five environments, including three challenging scientific environments: (1) manipulating DNA constructs for molecular cloning, (2) answering research questions by accessing scientific literature, and (3) engineering protein stability. These environments were selected for their focus on multi-step reasoning and their relevance to contemporary biology research. Finally, with online training and scaling inference-time compute, we show that language agents backed by open-source, non-frontier LLMs can match and exceed both frontier LLM agents and human experts on multiple tasks at up to 100x lower inference cost. 

**Abstract (ZH)**: 解决复杂的实际任务需要一系列的动作和观察。这一点在科学领域表现尤为明显，因为科学任务通常需要多次分析、工具使用和实验循环。语言代理在科学领域自动化智力任务方面具有巨大潜力，因为它们可以通过自然语言或代码与工具进行交互。然而，语言代理的灵活性也给软件实现带来了概念和实践上的挑战，因为这些代理可能包括非标准组件，如内部推理、规划、工具使用以及基于温度采样的语言模型固有的随机性。在此背景下，我们介绍了Aviary，一个灵活的语言代理实验平台。我们将代理精确定义为解决语言驱动的部分可观测马尔可夫决策过程的策略，我们将其称为语言决策过程。然后，我们实现了五个环境，包括三个具有挑战性的科学环境：（1）进行分子克隆时的核酸结构操作；（2）通过访问科学文献回答研究问题；（3）工程蛋白质稳定性。这些环境之所以被选择，是因为它们强调多步推理，并且与当前的生物学研究密切相关。最后，通过在线训练和扩展推理时间计算资源，我们证明基于开源非前沿的大语言模型（LLM）的语言代理能够在多个任务上达到甚至超过前沿LLM代理和人类专家的表现，且推理成本最多可降低100倍。 

---
# Training Software Engineering Agents and Verifiers with SWE-Gym 

**Title (ZH)**: 使用 SWE-Gym 训练软件工程代理和验证器 

**Authors**: Jiayi Pan, Xingyao Wang, Graham Neubig, Navdeep Jaitly, Heng Ji, Alane Suhr, Yizhe Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.21139)  

**Abstract**: We present SWE-Gym, the first environment for training real-world software engineering (SWE) agents. SWE-Gym contains 2,438 real-world Python task instances, each comprising a codebase with an executable runtime environment, unit tests, and a task specified in natural language. We use SWE-Gym to train language model based SWE agents , achieving up to 19% absolute gains in resolve rate on the popular SWE-Bench Verified and Lite test sets. We also experiment with inference-time scaling through verifiers trained on agent trajectories sampled from SWE-Gym. When combined with our fine-tuned SWE agents, we achieve 32.0% and 26.0% on SWE-Bench Verified and Lite, respectively, reflecting a new state-of-the-art for open-weight SWE agents. To facilitate further research, we publicly release SWE-Gym, models, and agent trajectories. 

**Abstract (ZH)**: 我们提出了SWE-Gym，这是首个用于训练现实世界软件工程（SWE）代理的环境。SWE-Gym包含2,438个实际存在的Python任务实例，每个实例都包含一个具有可执行运行时环境的代码库、单元测试以及用自然语言指定的任务。我们使用SWE-Gym来训练基于语言模型的SWE代理，在流行的SWE-Bench Verified和Lite测试集中，达到了高达19%的解决率绝对提升。我们还通过使用从SWE-Gym中采样的代理轨迹训练验证器，进行了推理时间的扩展实验。将这些验证器与我们微调的SWE代理结合使用后，在SWE-Bench Verified和Lite测试集中分别取得了32.0%和26.0%的成绩，这反映了开放式权重SWE代理的新最佳性能。为了促进进一步的研究，我们已将SWE-Gym、模型和代理轨迹公开发布。 

---
# Planning, Living and Judging: A Multi-agent LLM-based Framework for Cyclical Urban Planning 

**Title (ZH)**: 规划、居住与评判：基于多智能体LLM框架的循环城市规划系统 

**Authors**: Hang Ni, Yuzhi Wang, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20505)  

**Abstract**: Urban regeneration presents significant challenges within the context of urbanization, requiring adaptive approaches to tackle evolving needs. Leveraging advancements in large language models (LLMs), we propose Cyclical Urban Planning (CUP), a new paradigm that continuously generates, evaluates, and refines urban plans in a closed-loop. Specifically, our multi-agent LLM-based framework consists of three key components: (1) Planning, where LLM agents generate and refine urban plans based on contextual data; (2) Living, where agents simulate the behaviors and interactions of residents, modeling life in the urban environment; and (3) Judging, which involves evaluating plan effectiveness and providing iterative feedback for improvement. The cyclical process enables a dynamic and responsive planning approach. Experiments on the real-world dataset demonstrate the effectiveness of our framework as a continuous and adaptive planning process. 

**Abstract (ZH)**: 城市再开发在城市化进程中面临显著挑战，要求采用适应性策略来应对不断变化的需求。借助大型语言模型（LLMs）的进步，我们提出了一种新的范式——循环城市规划（Cyclical Urban Planning，CUP），该范式通过闭环不断生成、评估和优化城市规划。具体而言，我们的基于多智能体的大型语言模型框架包括三个关键组成部分：（1）规划阶段，LLM智能体基于上下文数据生成和优化城市规划；（2）生活阶段，智能体模拟居民的行为和互动，模型化城市环境中的人类生活；（3）评估阶段，涉及评估规划的有效性并提供迭代反馈以进行改进。循环过程使得规划方法具有动态和响应性。实验结果表明，我们的框架作为连续且适应性强的规划过程是有效的。 

---
# Action-Agnostic Point-Level Supervision for Temporal Action Detection 

**Title (ZH)**: 基于动作无关的点级别监督的时空动作检测 

**Authors**: Shuhei M. Yoshida, Takashi Shibata, Makoto Terao, Takayuki Okatani, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2412.21205)  

**Abstract**: We propose action-agnostic point-level (AAPL) supervision for temporal action detection to achieve accurate action instance detection with a lightly annotated dataset. In the proposed scheme, a small portion of video frames is sampled in an unsupervised manner and presented to human annotators, who then label the frames with action categories. Unlike point-level supervision, which requires annotators to search for every action instance in an untrimmed video, frames to annotate are selected without human intervention in AAPL supervision. We also propose a detection model and learning method to effectively utilize the AAPL labels. Extensive experiments on the variety of datasets (THUMOS '14, FineAction, GTEA, BEOID, and ActivityNet 1.3) demonstrate that the proposed approach is competitive with or outperforms prior methods for video-level and point-level supervision in terms of the trade-off between the annotation cost and detection performance. 

**Abstract (ZH)**: 我们提出了一种适用于时间动作检测的动作无关点级（AAPL）监督方法，以实现使用少量标注数据的精确动作实例检测。在所提出的方法中，一小部分视频帧以无监督的方式进行采样并呈现给人类标注者，随后他们使用动作类别对这些帧进行标注。与点级监督不同，后者要求标注者在未剪裁的视频中搜索每个动作实例，而AAPL监督在选择待标注的帧时不依赖人工干预。此外，我们还提出了一种检测模型和学习方法，以有效利用AAPL标签。在THUMOS '14、FineAction、GTEA、BEOD和ActivityNet 1.3等多个数据集上的广泛实验表明，所提出的方法在标注成本和检测性能的权衡上与或优于早期的视频级和点级监督方法。 

---
# UBER: Uncertainty-Based Evolution with Large Language Models for Automatic Heuristic Design 

**Title (ZH)**: UBER：基于不确定性的大语言模型自动启发式设计演化方法 

**Authors**: Zijie Chen, Zhanchao Zhou, Yu Lu, Renjun Xu, Lili Pan, Zhenzhong Lan  

**Link**: [PDF](https://arxiv.org/pdf/2412.20694)  

**Abstract**: NP-hard problem-solving traditionally relies on heuristics, but manually crafting effective heuristics for complex problems remains challenging. While recent work like FunSearch has demonstrated that large language models (LLMs) can be leveraged for heuristic design in evolutionary algorithm (EA) frameworks, their potential is not fully realized due to its deficiency in exploitation and exploration. We present UBER (Uncertainty-Based Evolution for Refinement), a method that enhances LLM+EA methods for automatic heuristic design by integrating uncertainty on top of the FunSearch framework. UBER introduces two key innovations: an Uncertainty-Inclusive Evolution Process (UIEP) for adaptive exploration-exploitation balance, and a principled Uncertainty-Inclusive Island Reset (UIIS) strategy for maintaining population diversity. Through extensive experiments on challenging NP-complete problems, UBER demonstrates significant improvements over FunSearch. Our work provides a new direction for the synergy of LLMs and EA, advancing the field of automatic heuristic design. 

**Abstract (ZH)**: 传统的NP-hard问题求解依赖于启发式方法，但为复杂问题手动设计有效的启发式方法仍然具有挑战性。虽然最近的研究，如FunSearch，已经证明大规模语言模型（LLMs）可以在进化算法（EAs）框架中用于启发式设计，但它们的潜力并未完全发挥，因为存在探索和利用能力的不足。我们提出了UBER（基于不确定性 refinement的进化算法），这是一种通过在FunSearch框架上集成不确定性来增强LLM+EA方法以实现自动启发式设计的方法。UBER引入了两项关键创新：一种包含不确定性的进化过程（UIEP），以实现自适应的探索和利用平衡，以及一种原则性的包含不确定性的岛群重启策略（UIIS），以维持种群多样性。通过在NP完全问题上的广泛实验，UBER在FunSearch的基础上取得了显著的进步。我们的工作为LLMs和EAs的协同作用提供了新的方向，推动了自动启发式设计领域的发展。 

---
# Game Theory and Multi-Agent Reinforcement Learning : From Nash Equilibria to Evolutionary Dynamics 

**Title (ZH)**: 博弈论与多智能体强化学习：从纳什均衡到进化动力学 

**Authors**: Neil De La Fuente, Miquel Noguer i Alonso, Guim Casadellà  

**Link**: [PDF](https://arxiv.org/pdf/2412.20523)  

**Abstract**: This paper explores advanced topics in complex multi-agent systems building upon our previous work. We examine four fundamental challenges in Multi-Agent Reinforcement Learning (MARL): non-stationarity, partial observability, scalability with large agent populations, and decentralized learning. The paper provides mathematical formulations and analysis of recent algorithmic advancements designed to address these challenges, with a particular focus on their integration with game-theoretic concepts. We investigate how Nash equilibria, evolutionary game theory, correlated equilibrium, and adversarial dynamics can be effectively incorporated into MARL algorithms to improve learning outcomes. Through this comprehensive analysis, we demonstrate how the synthesis of game theory and MARL can enhance the robustness and effectiveness of multi-agent systems in complex, dynamic environments. 

**Abstract (ZH)**: 本文在我们之前工作的基础上，探讨了复杂多智能体系统的高级主题。我们研究了多智能体强化学习（MARL）中四个基本挑战：非平稳性、部分可观测性、大规模智能体群体的可扩展性以及去中心化学习。文章提供了对旨在解决这些挑战的最新算法进展的数学建模和分析，并特别关注这些算法与博弈论概念的集成。我们探讨了如何有效地将纳什均衡、进化博弈论、相关均衡和对抗动态纳入MARL算法中，以改善学习结果。通过对这些内容的全面分析，我们展示了将博弈论与MARL的结合如何增强复杂动态环境中多智能体系统的稳健性和有效性。 

---
# Multi-Scenario Reasoning: Unlocking Cognitive Autonomy in Humanoid Robots for Multimodal Understanding 

**Title (ZH)**: 多场景推理：在类人机器人中实现多模态理解的认知自主性 

**Authors**: Libo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20429)  

**Abstract**: To improve the cognitive autonomy of humanoid robots, this research proposes a multi-scenario reasoning architecture to solve the technical shortcomings of multi-modal understanding in this field. It draws on simulation based experimental design that adopts multi-modal synthesis (visual, auditory, tactile) and builds a simulator "Maha" to perform the experiment. The findings demonstrate the feasibility of this architecture in multimodal data. It provides reference experience for the exploration of cross-modal interaction strategies for humanoid robots in dynamic environments. 

**Abstract (ZH)**: 为了提高类人机器人的情境认知自主能力，本研究提出了一种多场景推理架构，以解决该领域多模态理解的技术短板。该架构借鉴了基于仿真的实验设计，采用了多模态合成（视觉、听觉、触觉），并构建了一个名为“Maha”的模拟器来执行实验。研究发现表明了该架构在多模态数据上的可行性，为其在动态环境下类人机器人跨模态交互策略的探索提供了参考经验。 

---
# Safe Multiagent Coordination via Entropic Exploration 

**Title (ZH)**: 通过熵驱动的探索实现安全多智能体协调 

**Authors**: Ayhan Alp Aydeniz, Enrico Marchesini, Robert Loftin, Christopher Amato, Kagan Tumer  

**Link**: [PDF](https://arxiv.org/pdf/2412.20361)  

**Abstract**: Many real-world multiagent learning problems involve safety concerns. In these setups, typical safe reinforcement learning algorithms constrain agents' behavior, limiting exploration -- a crucial component for discovering effective cooperative multiagent behaviors. Moreover, the multiagent literature typically models individual constraints for each agent and has yet to investigate the benefits of using joint team constraints. In this work, we analyze these team constraints from a theoretical and practical perspective and propose entropic exploration for constrained multiagent reinforcement learning (E2C) to address the exploration issue. E2C leverages observation entropy maximization to incentivize exploration and facilitate learning safe and effective cooperative behaviors. Experiments across increasingly complex domains show that E2C agents match or surpass common unconstrained and constrained baselines in task performance while reducing unsafe behaviors by up to $50\%$. 

**Abstract (ZH)**: 许多现实世界的多代理学习问题涉及安全方面的考虑。在这种设置中，典型的安全强化学习算法会约束代理的行为，限制探索——这是发现有效合作行为的关键环节。此外，多代理文献通常为每个代理建模个体约束，尚未探讨使用联合团队约束所带来的益处。在本工作中，我们从理论和实践的角度分析了团队约束，并提出了受限多代理强化学习中的熵探索方法（E2C）来解决探索问题。E2C 利用观察熵最大化来激励探索，并促进学习安全有效的合作行为。实验结果表明，在逐渐复杂的任务领域中，E2C 代理在任务性能上与常见的未受约束和受约束基准相当甚至超越，并将不安全行为减少了高达 50%。 

---
# Leveraging Large Language Models for Enhancing Autonomous Vehicle Perception 

**Title (ZH)**: 利用大型语言模型增强自主车辆感知 

**Authors**: Athanasios Karagounis  

**Link**: [PDF](https://arxiv.org/pdf/2412.20230)  

**Abstract**: Autonomous vehicles (AVs) rely on sophisticated perception systems to interpret their surroundings, a cornerstone for safe navigation and decision-making. The integration of Large Language Models (LLMs) into AV perception frameworks offers an innovative approach to address challenges in dynamic environments, sensor fusion, and contextual reasoning. This paper presents a novel framework for incorporating LLMs into AV perception, enabling advanced contextual understanding, seamless sensor integration, and enhanced decision support. Experimental results demonstrate that LLMs significantly improve the accuracy and reliability of AV perception systems, paving the way for safer and more intelligent autonomous driving technologies. By expanding the scope of perception beyond traditional methods, LLMs contribute to creating a more adaptive and human-centric driving ecosystem, making autonomous vehicles more reliable and transparent in their operations. These advancements redefine the relationship between human drivers and autonomous systems, fostering trust through enhanced understanding and personalized decision-making. Furthermore, by integrating memory modules and adaptive learning mechanisms, LLMs introduce continuous improvement in AV perception, enabling vehicles to evolve with time and adapt to changing environments and user preferences. 

**Abstract (ZH)**: 自主驾驶车辆（AVs）依赖于复杂的感知系统来解释其周围环境，这是确保安全导航和决策的基础。将大型语言模型（LLMs）整合到AV感知框架中为应对动态环境、传感器融合和上下文推理的挑战提供了一种创新的方法。本文提出了一种新的框架，用于将LLMs整合到AV感知中，从而实现高级上下文理解、无缝传感器集成和增强的决策支持。实验结果表明，LLMs显著提高了AV感知系统的准确性和可靠性，为更安全、更智能的自主驾驶技术铺平了道路。通过扩展感知范围超越传统方法，LLMs有助于创建一个更具适应性和以人为中心的驾驶生态系统，使自主车辆在操作中更为可靠和透明。这些进步重新定义了人类驾驶员与自主系统之间的关系，通过增强理解和个性化决策来培养信任。此外，通过整合记忆模块和适应性学习机制，LLMs为AV感知提供持续改进的能力，使车辆能够随着时间的推移而演变，并适应不断变化的环境和用户偏好。 

---
# TradingAgents: Multi-Agents LLM Financial Trading Framework 

**Title (ZH)**: 交易代理：多智能体LLM金融交易框架

注：在这个翻译中，“TradingAgents”被译为“交易代理”，“Multi-Agents”被译为“多智能体”，“LLM”被解释为“大语言模型”，考虑到具体上下文，“LLM”也有可能指的是“长期记忆模型”或其他特定含义，需要根据实际场景进一步确认。此处保持了“LLM”未译，保持原文中的缩写形式，同时在翻译中注释其可能的含义。总体来说，整句话翻译保持了原文的学术风格和专业术语。 

**Authors**: Yijia Xiao, Edward Sun, Di Luo, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.20138)  

**Abstract**: Significant progress has been made in automated problem-solving using societies of agents powered by large language models (LLMs). In finance, efforts have largely focused on single-agent systems handling specific tasks or multi-agent frameworks independently gathering data. However, multi-agent systems' potential to replicate real-world trading firms' collaborative dynamics remains underexplored. TradingAgents proposes a novel stock trading framework inspired by trading firms, featuring LLM-powered agents in specialized roles such as fundamental analysts, sentiment analysts, technical analysts, and traders with varied risk profiles. The framework includes Bull and Bear researcher agents assessing market conditions, a risk management team monitoring exposure, and traders synthesizing insights from debates and historical data to make informed decisions. By simulating a dynamic, collaborative trading environment, this framework aims to improve trading performance. Detailed architecture and extensive experiments reveal its superiority over baseline models, with notable improvements in cumulative returns, Sharpe ratio, and maximum drawdown, highlighting the potential of multi-agent LLM frameworks in financial trading. 

**Abstract (ZH)**: 在使用大规模语言模型（LLMs）驱动的代理社会进行自动化问题解决方面已经取得显著进展。在金融领域，大部分努力主要集中在处理特定任务的单代理系统或独立收集数据的多代理框架上。然而，多代理系统在复制现实世界交易公司的协同动态方面具有巨大潜力，这一领域尚未充分探索。TradingAgents 提出了一种受交易公司启发的新型股票交易框架，该框架中的代理由LLM驱动，扮演不同的专业角色，如基本面分析师、情绪分析师、技术分析师和不同风险偏好级别的交易者。该框架包括牛市和熊市研究员代理评估市场状况，风险管理部门监控风险敞口，以及交易者通过辩论和历史数据综合获得的见解来做决策。通过模拟动态且协作的交易环境，该框架旨在提高交易表现。详细的架构和大量实验表明，与基准模型相比，该框架在累积回报、夏普比率和最大回撤等方面具有显著优势，这表明多代理LLM框架在金融市场交易中的潜力。 

---
# AnalogXpert: Automating Analog Topology Synthesis by Incorporating Circuit Design Expertise into Large Language Models 

**Title (ZH)**: AnalogXpert: 通过将电路设计专业知识融入大型语言模型来自动化模拟拓扑合成 

**Authors**: Haoyi Zhang, Shizhao Sun, Yibo Lin, Runsheng Wang, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2412.19824)  

**Abstract**: Analog circuits are crucial in modern electronic systems, and automating their design has attracted significant research interest. One of major challenges is topology synthesis, which determines circuit components and their connections. Recent studies explore large language models (LLM) for topology synthesis. However, the scenarios addressed by these studies do not align well with practical applications. Specifically, existing work uses vague design requirements as input and outputs an ideal model, but detailed structural requirements and device-level models are more practical. Moreover, current approaches either formulate topology synthesis as graph generation or Python code generation, whereas practical topology design is a complex process that demands extensive design knowledge. In this work, we propose AnalogXpert, a LLM-based agent aiming at solving practical topology synthesis problem by incorporating circuit design expertise into LLMs. First, we represent analog topology as SPICE code and introduce a subcircuit library to reduce the design space, in the same manner as experienced designers. Second, we decompose the problem into two sub-task (i.e., block selection and block connection) through the use of CoT and incontext learning techniques, to mimic the practical design process. Third, we introduce a proofreading strategy that allows LLMs to incrementally correct the errors in the initial design, akin to human designers who iteratively check and adjust the initial topology design to ensure accuracy. Finally, we construct a high-quality benchmark containing both real data (30) and synthetic data (2k). AnalogXpert achieves 40% and 23% success rates on the synthetic dataset and real dataset respectively, which is markedly better than those of GPT-4o (3% on both the synthetic dataset and the real dataset). 

**Abstract (ZH)**: 现代电子系统中，模拟电路至关重要，其自动化设计吸引了大量研究兴趣。其中一项主要挑战是拓扑合成，它决定了电路元件及其连接方式。近期的研究探讨了通过大型语言模型（LLM）进行拓扑合成的可行性，但是现有研究中的应用场景与实际应用不完全匹配。具体来说，现有工作使用模糊的设计要求作为输入，输出理想模型，但实际上，详细的结构要求和器件级模型更为实用。此外，当前的方法将拓扑合成要么形式化为图生成，要么形式化为Python代码生成，而实际的拓扑设计是一个复杂的过程，需要广泛的设计知识。在本文中，我们提出了AnalogXpert，这是一种基于LLM的代理，旨在通过将电路设计专业知识整合到LLM中来解决实际的拓扑合成问题。首先，我们将模拟拓扑表示为SPICE代码，并引入子电路库以减少设计空间，类似于经验丰富的设计师的做法。其次，我们通过使用CoT和上下文学习技术将问题分解为两个子任务（即模块选择和模块连接），以模拟实际设计过程。第三，我们引入了一种校对策略，使LLM能够逐步修正初始设计中的错误，类似于人类设计师通过迭代检查和调整初始拓扑设计来确保准确性。最后，我们构建了一个高性能基准，包含实际数据（30个）和合成数据（2000个）。AnalogXpert在合成数据集和实际数据集上的成功率分别为40%和23%，这比GPT-4o的性能要好得多（在合成数据集和实际数据集上均为3%）。 

---
