# TheoremExplainAgent: Towards Multimodal Explanations for LLM Theorem Understanding 

**Title (ZH)**: 《定理解释代理：面向多模态定理理解的解释方法》 

**Authors**: Max Ku, Thomas Chong, Jonathan Leung, Krish Shah, Alvin Yu, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.19400)  

**Abstract**: Understanding domain-specific theorems often requires more than just text-based reasoning; effective communication through structured visual explanations is crucial for deeper comprehension. While large language models (LLMs) demonstrate strong performance in text-based theorem reasoning, their ability to generate coherent and pedagogically meaningful visual explanations remains an open challenge. In this work, we introduce TheoremExplainAgent, an agentic approach for generating long-form theorem explanation videos (over 5 minutes) using Manim animations. To systematically evaluate multimodal theorem explanations, we propose TheoremExplainBench, a benchmark covering 240 theorems across multiple STEM disciplines, along with 5 automated evaluation metrics. Our results reveal that agentic planning is essential for generating detailed long-form videos, and the o3-mini agent achieves a success rate of 93.8% and an overall score of 0.77. However, our quantitative and qualitative studies show that most of the videos produced exhibit minor issues with visual element layout. Furthermore, multimodal explanations expose deeper reasoning flaws that text-based explanations fail to reveal, highlighting the importance of multimodal explanations. 

**Abstract (ZH)**: 理解特定领域的定理通常不仅需要基于文本的推理，有效的沟通还需要通过结构化的视觉解释来深化理解。虽然大型语言模型（LLMs）在基于文本的定理推理方面表现出色，但生成连贯且有教育意义的视觉解释仍然是一个开放的挑战。在本工作中，我们引入了TheoremExplainAgent，这是一种使用Manim动画生成长格式定理解释视频（超过5分钟）的方法。为了系统性地评估多模态定理解释，我们提出了TheoremExplainBench，这是一个涵盖240个定理（涉及多个STEM学科）的基准，同时附有5个自动化评估指标。我们的结果显示，有意识的规划对于生成详细长视频至关重要，o3-mini代理的成功率为93.8%，总体评分为0.77。然而，我们的定量和定性研究显示，大多数生成的视频在视觉元素布局方面存在一些小问题。此外，多模态解释揭示了基于文本的解释无法揭示的深层次推理缺陷，突显了多模态解释的重要性。 

---
# Joint Optimal Transport and Embedding for Network Alignment 

**Title (ZH)**: 联合最优传输与嵌入在网络对齐中的应用 

**Authors**: Qi Yu, Zhichen Zeng, Yuchen Yan, Lei Ying, R. Srikant, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2502.19334)  

**Abstract**: Network alignment, which aims to find node correspondence across different networks, is the cornerstone of various downstream multi-network and Web mining tasks. Most of the embedding-based methods indirectly model cross-network node relationships by contrasting positive and negative node pairs sampled from hand-crafted strategies, which are vulnerable to graph noises and lead to potential misalignment of nodes. Another line of work based on the optimal transport (OT) theory directly models cross-network node relationships and generates noise-reduced alignments. However, OT methods heavily rely on fixed, pre-defined cost functions that prohibit end-to-end training and are hard to generalize. In this paper, we aim to unify the embedding and OT-based methods in a mutually beneficial manner and propose a joint optimal transport and embedding framework for network alignment named JOENA. For one thing (OT for embedding), through a simple yet effective transformation, the noise-reduced OT mapping serves as an adaptive sampling strategy directly modeling all cross-network node pairs for robust embedding this http URL another (embedding for OT), on top of the learned embeddings, the OT cost can be gradually trained in an end-to-end fashion, which further enhances the alignment quality. With a unified objective, the mutual benefits of both methods can be achieved by an alternating optimization schema with guaranteed convergence. Extensive experiments on real-world networks validate the effectiveness and scalability of JOENA, achieving up to 16% improvement in MRR and 20x speedup compared with the state-of-the-art alignment methods. 

**Abstract (ZH)**: 网络对齐（network alignment）旨在找到不同网络之间节点的对应关系，是多种下游多网络和Web挖掘任务的基础。大部分基于嵌入的方法通过从手工构建的策略中采样正负节点对来间接建模跨网络节点关系，这使其容易受到图噪声的影响，并可能导致节点的潜在错位。另一种基于最优传输（Optimal Transport, OT）理论的方法直接建模跨网络节点关系并生成降噪对齐。然而，OT方法高度依赖固定的、先验定义的成本函数，这阻碍了端到端的训练，并且难以泛化。本文旨在以一种互利的方式统一嵌入方法和OT方法，并提出了一种名为JOENA的联合最优传输和嵌入框架，用于网络对齐。首先，在嵌入方面，通过一个简单有效的转换，降噪的OT映射充当一种自适应的采样策略，直接建模所有跨网络节点对，以实现鲁棒的嵌入。其次，在学习到的嵌入的基础上，OT成本可以逐步以端到端的方式训练，进一步提高对齐质量。通过一个统一的目标函数，交替优化方案可以保证收敛并同时实现两种方法的互惠互利。在真实世界网络上的广泛实验验证了JOENA的有效性和可扩展性，与最先进的对齐方法相比，MRR提高了最高16%，速度提高了20倍。 

---
# WOFOSTGym: A Crop Simulator for Learning Annual and Perennial Crop Management Strategies 

**Title (ZH)**: WOFOST-Gym：一种用于学习年度和多年生作物管理策略的作物模拟器 

**Authors**: William Solow, Sandhya Saisubramanian, Alan Fern  

**Link**: [PDF](https://arxiv.org/pdf/2502.19308)  

**Abstract**: We introduce WOFOSTGym, a novel crop simulation environment designed to train reinforcement learning (RL) agents to optimize agromanagement decisions for annual and perennial crops in single and multi-farm settings. Effective crop management requires optimizing yield and economic returns while minimizing environmental impact, a complex sequential decision-making problem well suited for RL. However, the lack of simulators for perennial crops in multi-farm contexts has hindered RL applications in this domain. Existing crop simulators also do not support multiple annual crops. WOFOSTGym addresses these gaps by supporting 23 annual crops and two perennial crops, enabling RL agents to learn diverse agromanagement strategies in multi-year, multi-crop, and multi-farm settings. Our simulator offers a suite of challenging tasks for learning under partial observability, non-Markovian dynamics, and delayed feedback. WOFOSTGym's standard RL interface allows researchers without agricultural expertise to explore a wide range of agromanagement problems. Our experiments demonstrate the learned behaviors across various crop varieties and soil types, highlighting WOFOSTGym's potential for advancing RL-driven decision support in agriculture. 

**Abstract (ZH)**: 我们介绍了一种新的作物模拟环境——WOFOSTGym，该环境旨在训练强化学习（RL）代理，以优化一年生和多年生作物在单农场和多农场环境中的农作管理决策。有效的作物管理要求优化产量和经济效益，同时最小化环境影响，这是一项复杂的序贯决策问题，非常适合使用RL。然而，缺乏适用于多年生作物和多农场环境的模拟器阻碍了该领域的RL应用。现有的作物模拟器也不支持多种一年生作物。WOFOSTGym 通过支持23种一年生作物和两种多年生作物，填补了这些空白，使RL代理能够在多年的、多作物和多农场环境中学习多样的农作管理策略。我们的模拟器提供了一系列具有挑战性的任务，用于在部分可观测性、非马尔可夫动态以及延迟反馈的情境下学习。WOFOSTGym 的标准RL接口使没有农业背景的科研人员也能探索广泛范围的农作管理问题。我们的实验展示了在各种作物品种和土壤类型下的学习行为，突显了WOFOSTGym 在推动基于RL的农业决策支持方面的潜力。 

---
# Complex LLM Planning via Automated Heuristics Discovery 

**Title (ZH)**: 通过自动启发式发现进行复杂的LLM规划 

**Authors**: Hongyi Ling, Shubham Parashar, Sambhav Khurana, Blake Olson, Anwesha Basu, Gaurangi Sinha, Zhengzhong Tu, James Caverlee, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.19295)  

**Abstract**: We consider enhancing large language models (LLMs) for complex planning tasks. While existing methods allow LLMs to explore intermediate steps to make plans, they either depend on unreliable self-verification or external verifiers to evaluate these steps, which demand significant data and computations. Here, we propose automated heuristics discovery (AutoHD), a novel approach that enables LLMs to explicitly generate heuristic functions to guide inference-time search, allowing accurate evaluation of intermediate states. These heuristic functions are further refined through a heuristic evolution process, improving their robustness and effectiveness. Our proposed method requires no additional model training or fine-tuning, and the explicit definition of heuristic functions generated by the LLMs provides interpretability and insights into the reasoning process. Extensive experiments across diverse benchmarks demonstrate significant gains over multiple baselines, including nearly twice the accuracy on some datasets, establishing our approach as a reliable and interpretable solution for complex planning tasks. 

**Abstract (ZH)**: 我们考虑增强大型语言模型（LLMs）以应对复杂的规划任务。虽然现有的方法允许LLMs探索中间步骤以制定计划，但这些方法要么依赖于不可靠的自我验证，要么依赖外部验证者来评估这些步骤，这些过程需要大量的数据和计算资源。在此，我们提出了自动启发式发现（AutoHD）这一新颖方法，该方法使LLMs能够明确生成启发式函数以引导推理时的搜索，并允许对中间状态进行准确的评估。通过启发式进化过程进一步优化这些启发式函数，提高了它们的稳健性和有效性。我们提出的方法不需要额外的模型训练或微调，由LLMs生成的明确定义的启发式函数提供了可解释性和对推理过程的洞察。在多种基准测试中的广泛实验表明，与多个基线相比，我们的方法在某些数据集上几乎提高了两倍的准确性，确立了我们的方法作为复杂规划任务的一种可靠且可解释的解决方案。 

---
# Multi-Agent Security Tax: Trading Off Security and Collaboration Capabilities in Multi-Agent Systems 

**Title (ZH)**: 多代理安全税：多代理系统中安全与协作能力的权衡 

**Authors**: Pierre Peigne-Lefebvre, Mikolaj Kniejski, Filip Sondej, Matthieu David, Jason Hoelscher-Obermaier, Christian Schroeder de Witt, Esben Kran  

**Link**: [PDF](https://arxiv.org/pdf/2502.19145)  

**Abstract**: As AI agents are increasingly adopted to collaborate on complex objectives, ensuring the security of autonomous multi-agent systems becomes crucial. We develop simulations of agents collaborating on shared objectives to study these security risks and security trade-offs. We focus on scenarios where an attacker compromises one agent, using it to steer the entire system toward misaligned outcomes by corrupting other agents. In this context, we observe infectious malicious prompts - the multi-hop spreading of malicious instructions. To mitigate this risk, we evaluated several strategies: two "vaccination" approaches that insert false memories of safely handling malicious input into the agents' memory stream, and two versions of a generic safety instruction strategy. While these defenses reduce the spread and fulfillment of malicious instructions in our experiments, they tend to decrease collaboration capability in the agent network. Our findings illustrate potential trade-off between security and collaborative efficiency in multi-agent systems, providing insights for designing more secure yet effective AI collaborations. 

**Abstract (ZH)**: 随着人工智能代理被越来越多地用于协同实现复杂的任务目标，确保自主多智能体系统的安全性变得至关重要。我们开发了模拟代理协作实现共享目标的仿真，以研究这些安全风险和安全权衡。我们专注于一种场景，在这种场景中，攻击者控制一个代理，利用该代理引导整个系统朝目标偏差的方向发展，从而破坏其他代理。在这种背景下，我们观察到恶意指令的传染性——多跳传播的恶意指令。为了减轻这一风险，我们评估了多种策略：两种“疫苗”方法，即向代理的记忆流中插入虚假记忆，使其认为已经安全地处理了恶意输入，以及两种通用安全性指令策略的版本。虽然这些防御措施在我们的实验中减轻了恶意指令的传播与实现，但它们往往会降低代理网络的协作能力。我们的研究结果展示了在多智能体系统中安全性和协作效率之间的潜在权衡，并为设计更安全且有效的AI协作提供了洞见。 

---
# A Temporal Planning Framework for Multi-Agent Systems via LLM-Aided Knowledge Base Management 

**Title (ZH)**: 通过LLM辅助知识库管理的多agent系统时间规划框架 

**Authors**: Enrico Saccon, Ahmet Tikna, Davide De Martini, Edoardo Lamon, Luigi Palopoli, Marco Roveri  

**Link**: [PDF](https://arxiv.org/pdf/2502.19135)  

**Abstract**: This paper presents a novel framework, called PLANTOR (PLanning with Natural language for Task-Oriented Robots), that integrates Large Language Models (LLMs) with Prolog-based knowledge management and planning for multi-robot tasks. The system employs a two-phase generation of a robot-oriented knowledge base, ensuring reusability and compositional reasoning, as well as a three-step planning procedure that handles temporal dependencies, resource constraints, and parallel task execution via mixed-integer linear programming. The final plan is converted into a Behaviour Tree for direct use in ROS2. We tested the framework in multi-robot assembly tasks within a block world and an arch-building scenario. Results demonstrate that LLMs can produce accurate knowledge bases with modest human feedback, while Prolog guarantees formal correctness and explainability. This approach underscores the potential of LLM integration for advanced robotics tasks requiring flexible, scalable, and human-understandable planning. 

**Abstract (ZH)**: 本文提出了一种新的框架，称为PLANTOR（基于自然语言的面向任务机器人规划），该框架将大型语言模型（LLMs）与基于Prolog的知识管理和规划相结合，用于多机器人任务。该系统采用了两阶段机器人定向知识库生成方式，确保了可重用性和组合推理能力，以及包含三个步骤的规划流程，通过混合整数线性规划处理时间依赖性、资源约束和并行任务执行问题。最终的计划被转换成行为树，可以直接用于ROS2。我们在块世界和拱结构建造场景中的多机器人装配任务中测试了该框架。结果表明，LLMs能够在适度的人工反馈下生成准确的知识库，而Prolog则保证了形式正确性和可解释性。这一方法突显了LLMs在实现复杂、可扩展且易于人类理解的规划方面潜在的优势，适用于高级机器人任务。 

---
# Nexus: A Lightweight and Scalable Multi-Agent Framework for Complex Tasks Automation 

**Title (ZH)**: Nexus：一种轻量级且可扩展的多代理框架，用于复杂任务自动化 

**Authors**: Humza Sami, Mubashir ul Islam, Samy Charas, Asav Gandhi, Pierre-Emmanuel Gaillardon, Valerio Tenace  

**Link**: [PDF](https://arxiv.org/pdf/2502.19091)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have substantially evolved Multi-Agent Systems (MASs) capabilities, enabling systems that not only automate tasks but also leverage near-human reasoning capabilities. To achieve this, LLM-based MASs need to be built around two critical principles: (i) a robust architecture that fully exploits LLM potential for specific tasks -- or related task sets -- and ($ii$) an effective methodology for equipping LLMs with the necessary capabilities to perform tasks and manage information efficiently. It goes without saying that a priori architectural designs can limit the scalability and domain adaptability of a given MAS.
To address these challenges, in this paper we introduce Nexus: a lightweight Python framework designed to easily build and manage LLM-based MASs. Nexus introduces the following innovations: (i) a flexible multi-supervisor hierarchy, (ii) a simplified workflow design, and (iii) easy installation and open-source flexibility: Nexus can be installed via pip and is distributed under a permissive open-source license, allowing users to freely modify and extend its capabilities.
Experimental results demonstrate that architectures built with Nexus exhibit state-of-the-art performance across diverse domains. In coding tasks, Nexus-driven MASs achieve a 99% pass rate on HumanEval and a flawless 100% on VerilogEval-Human, outperforming cutting-edge reasoning language models such as o3-mini and DeepSeek-R1. Moreover, these architectures display robust proficiency in complex reasoning and mathematical problem solving, achieving correct solutions for all randomly selected problems from the MATH dataset. In the realm of multi-objective optimization, Nexus-based architectures successfully address challenging timing closure tasks on designs from the VTR benchmark suite, while guaranteeing, on average, a power saving of nearly 30%. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的最新进展显著提升了多代理系统（MASs）的能力，使其不仅能够自动化任务，还能利用接近人类的推理能力。为了实现这一点，基于LLM的MASs需要围绕两个关键原则进行构建：（i）一个稳健的架构，充分利用LLM在特定任务或相关任务集中的潜力；和（ii）一种有效的方案，使LLM具备必要的能力，以高效地执行任务和管理信息。不言而喻，先验的设计可能会限制给定MAS的扩展性和领域适应性。

为了解决这些问题，在本文中我们提出了Nexus：一个轻量级的Python框架，用于轻松构建和管理基于LLM的MASs。Nexus的创新包括：（i）灵活的多监督器层次结构，（ii）简化的流程设计，以及（iii）易于安装和开源灵活性：Nexus可以通过pip安装，并采用宽松的开源许可协议进行分发，允许用户自由修改和扩展其功能。

实验结果表明，使用Nexus构建的架构在多个领域都表现出最先进的性能。在编程任务中，Nexus驱动的MASs在HumanEval基准测试中达到了99%的通过率，并在VerilogEval-Human基准测试中达到了100%的满分，超越了诸如o3-mini和DeepSeek-R1等最先进的推理语言模型。此外，这些架构在复杂的推理和数学问题解决方面表现出了强大的能力，对于从MATH数据集中随机选择的所有问题都实现了正确的解决方案。在多目标优化领域，基于Nexus的架构成功应对了VTR基准套件中的复杂定时闭合任务，并平均实现了近30%的功率节省。 

---
# Dealing with Inconsistency for Reasoning over Knowledge Graphs: A Survey 

**Title (ZH)**: 处理知识图谱推理中的不一致性：一种综述 

**Authors**: Anastasios Nentidis, Charilaos Akasiadis, Angelos Charalambidis, Alexander Artikis  

**Link**: [PDF](https://arxiv.org/pdf/2502.19023)  

**Abstract**: In Knowledge Graphs (KGs), where the schema of the data is usually defined by particular ontologies, reasoning is a necessity to perform a range of tasks, such as retrieval of information, question answering, and the derivation of new knowledge. However, information to populate KGs is often extracted (semi-) automatically from natural language resources, or by integrating datasets that follow different semantic schemas, resulting in KG inconsistency. This, however, hinders the process of reasoning. In this survey, we focus on how to perform reasoning on inconsistent KGs, by analyzing the state of the art towards three complementary directions: a) the detection of the parts of the KG that cause the inconsistency, b) the fixing of an inconsistent KG to render it consistent, and c) the inconsistency-tolerant reasoning. We discuss existing work from a range of relevant fields focusing on how, and in which cases they are related to the above directions. We also highlight persisting challenges and future directions. 

**Abstract (ZH)**: 在知识图谱（KGs）中，数据结构通常由特定的本体定义。在执行信息检索、问答和新知识推导等一系列任务时，推理是必要的。然而，用于填充KG的信息往往通过半自动的方式从自然语言资源中提取，或者通过集成遵循不同语义模式的数据集，这可能导致KG的一致性问题。这在一定程度上阻碍了推理过程。在本综述中，我们专注于如何在不一致的KG上进行推理，通过对三个互补方向的研究现状进行分析：a）检测导致不一致的部分KG，b）修复不一致的KG以使其一致，c）容忍不一致性的推理。我们从多个相关领域出发，讨论现有工作的实现方式及其与上述方向的相关性，并强调存在的挑战及未来的研究方向。 

---
# Talking like Piping and Instrumentation Diagrams (P&IDs) 

**Title (ZH)**: 模仿管道和仪表图（P&IDs）的表达方式 

**Authors**: Achmad Anggawirya Alimin, Dominik P. Goldstein, Lukas Schulze Balhorn, Artur M. Schweidtmann  

**Link**: [PDF](https://arxiv.org/pdf/2502.18928)  

**Abstract**: We propose a methodology that allows communication with Piping and Instrumentation Diagrams (P&IDs) using natural language. In particular, we represent P&IDs through the DEXPI data model as labeled property graphs and integrate them with Large Language Models (LLMs). The approach consists of three main parts: 1) P&IDs are cast into a graph representation from the DEXPI format using our pyDEXPI Python package. 2) A tool for generating P&ID knowledge graphs from pyDEXPI. 3) Integration of the P&ID knowledge graph to LLMs using graph-based retrieval augmented generation (graph-RAG). This approach allows users to communicate with P&IDs using natural language. It extends LLM's ability to retrieve contextual data from P&IDs and mitigate hallucinations. Leveraging the LLM's large corpus, the model is also able to interpret process information in PIDs, which could help engineers in their daily tasks. In the future, this work will also open up opportunities in the context of other generative Artificial Intelligence (genAI) solutions on P&IDs, and AI-assisted HAZOP studies. 

**Abstract (ZH)**: 我们提出了一种方法论，允许通过工艺和管道图（P&IDs）使用自然语言进行通信。特别地，我们通过DEXPI数据模型将P&IDs表示为带标签的属性图，并将其与大规模语言模型（LLMs）集成。该方法主要包括三个主要部分：1）使用我们的pyDEXPI Python包将P&IDs转换为从DEXPI格式到图表示。2）一种从pyDEXPI生成P&ID知识图的工具。3）通过基于图的检索增强生成（graph-RAG）将P&ID知识图与LLMs集成。该方法允许用户使用自然语言与P&IDs进行通信。它扩展了LLM从P&IDs检索上下文数据的能力，并减轻了幻觉现象。利用LLM庞大的语料库，该模型还能够解释PID中的过程信息，这可能有助于工程师完成日常任务。未来，这项工作还将为PID上下文中的其他生成型人工智能（genAI）解决方案以及AI辅助的HAZOP研究开辟新的机会。 

---
# Multi-LLM Collaborative Search for Complex Problem Solving 

**Title (ZH)**: 多大型语言模型协作搜索在复杂问题解决中的应用 

**Authors**: Sen Yang, Yafu Li, Wai Lam, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18873)  

**Abstract**: Large language models (LLMs) often struggle with complex reasoning tasks due to their limitations in addressing the vast reasoning space and inherent ambiguities of natural language. We propose the Mixture-of-Search-Agents (MoSA) paradigm, a novel approach leveraging the collective expertise of multiple LLMs to enhance search-based reasoning. MoSA integrates diverse reasoning pathways by combining independent exploration with iterative refinement among LLMs, mitigating the limitations of single-model approaches. Using Monte Carlo Tree Search (MCTS) as a backbone, MoSA enables multiple agents to propose and aggregate reasoning steps, resulting in improved accuracy. Our comprehensive evaluation across four reasoning benchmarks demonstrates MoSA's consistent performance improvements over single-agent and other multi-agent baselines, particularly in complex mathematical and commonsense reasoning tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂推理任务中往往表现不佳，因为它们在处理广泛推理空间和自然语言固有的歧义性方面存在局限性。我们提出了搜索代理混合体（Mixture-of-Search-Agents, MoSA）范式，这是一种利用多个LLM集体专业知识来增强基于搜索的推理的新方法。MoSA 通过结合独立探索和LLMs之间的迭代完善，整合了多样的推理路径，从而减轻了单模型方法的局限性。MoSA 以蒙特卡洛树搜索（MCTS）为基础，使多个代理能够提出并聚合推理步骤，从而提高准确性。我们的全面评估表明，MoSA 在四个推理基准上的表现优于单代理和其它多代理基线，特别是在复杂的数学和常识推理任务中表现出显著改进。 

---
# Towards an AI co-scientist 

**Title (ZH)**: Towards 一位AI合作者 

**Authors**: Juraj Gottweis, Wei-Hung Weng, Alexander Daryin, Tao Tu, Anil Palepu, Petar Sirkovic, Artiom Myaskovsky, Felix Weissenberger, Keran Rong, Ryutaro Tanno, Khaled Saab, Dan Popovici, Jacob Blum, Fan Zhang, Katherine Chou, Avinatan Hassidim, Burak Gokturk, Amin Vahdat, Pushmeet Kohli, Yossi Matias, Andrew Carroll, Kavita Kulkarni, Nenad Tomasev, Yuan Guan, Vikram Dhillon, Eeshit Dhaval Vaishnav, Byron Lee, Tiago R D Costa, José R Penadés, Gary Peltz, Yunhan Xu, Annalisa Pawlosky, Alan Karthikesalingam, Vivek Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18864)  

**Abstract**: Scientific discovery relies on scientists generating novel hypotheses that undergo rigorous experimental validation. To augment this process, we introduce an AI co-scientist, a multi-agent system built on Gemini 2.0. The AI co-scientist is intended to help uncover new, original knowledge and to formulate demonstrably novel research hypotheses and proposals, building upon prior evidence and aligned to scientist-provided research objectives and guidance. The system's design incorporates a generate, debate, and evolve approach to hypothesis generation, inspired by the scientific method and accelerated by scaling test-time compute. Key contributions include: (1) a multi-agent architecture with an asynchronous task execution framework for flexible compute scaling; (2) a tournament evolution process for self-improving hypotheses generation. Automated evaluations show continued benefits of test-time compute, improving hypothesis quality. While general purpose, we focus development and validation in three biomedical areas: drug repurposing, novel target discovery, and explaining mechanisms of bacterial evolution and anti-microbial resistance. For drug repurposing, the system proposes candidates with promising validation findings, including candidates for acute myeloid leukemia that show tumor inhibition in vitro at clinically applicable concentrations. For novel target discovery, the AI co-scientist proposed new epigenetic targets for liver fibrosis, validated by anti-fibrotic activity and liver cell regeneration in human hepatic organoids. Finally, the AI co-scientist recapitulated unpublished experimental results via a parallel in silico discovery of a novel gene transfer mechanism in bacterial evolution. These results, detailed in separate, co-timed reports, demonstrate the potential to augment biomedical and scientific discovery and usher an era of AI empowered scientists. 

**Abstract (ZH)**: 科学发现依赖于科学家提出新颖的假设并通过严格的实验验证。为了增强这一过程，我们引入了一个AI合作者，这是一种基于Gemini 2.0的多智能体系统。AI合作者旨在帮助发现新的原创知识，并根据先前的证据和科学家提供的研究目标和指导，提出可证明的新颖研究假设和建议。系统的设计采用了生成、辩论和进化的方法来生成假设，这一方法借鉴了科学方法，并通过扩展测试时计算的规模得到了加速。主要贡献包括：（1）一种具有异步任务执行框架的多智能体架构，实现灵活的计算扩展；（2）一种锦标赛进化过程以自我改善假设生成。自动评估显示，测试时计算继续带来益处，提高了假设的质量。虽然具有通用目的，但我们将开发和验证集中在三个生物医学领域：药物重新定位、新的靶点发现以及解释细菌进化和抗生素耐药机制。在药物重新定位方面，该系统提出了具有有前景验证结果的候选药物，包括在临床适用浓度下显示对急性髓系白血病抑制作用的候选药物。在新的靶点发现方面，AI合作者提出了新的表观遗传靶点用于肝纤维化，这些靶点通过抗纤维化活性和人类肝类器官中的肝细胞再生进行了验证。最后，AI合作者通过平行的计算发现了一个新的基因转移机制，重现了细菌进化的未发表实验结果。这些结果，分别在单独的、同步发布的报告中详述，展示了其在生物医学和科学研究中增强发现潜力，并开启了AI赋能科学家的新时期。 

---
# Intelligence Test 

**Title (ZH)**: 智力测验 

**Authors**: Jingtao Zhan, Jiahao Zhao, Jiayu Li, Yiqun Liu, Bo Zhang, Qingyao Ai, Jiaxin Mao, Hongning Wang, Min Zhang, Shaoping Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.18858)  

**Abstract**: How does intelligence emerge? We propose that intelligence is not a sudden gift or random occurrence, but rather a necessary trait for species to survive through Natural Selection. If a species passes the test of Natural Selection, it demonstrates the intelligence to survive in nature. Extending this perspective, we introduce Intelligence Test, a method to quantify the intelligence of any subject on any task. Like how species evolve by trial and error, Intelligence Test quantifies intelligence by the number of failed attempts before success. Fewer failures correspond to higher intelligence. When the expectation and variance of failure counts are both finite, it signals the achievement of an autonomous level of intelligence. Using Intelligence Test, we comprehensively evaluate existing AI systems. Our results show that while AI systems achieve a level of autonomy in simple tasks, they are still far from autonomous in more complex tasks, such as vision, search, recommendation, and language. While scaling model size might help, this would come at an astronomical cost. Projections suggest that achieving general autonomy would require unimaginable $10^{26}$ parameters. Even if Moore's Law continuously holds, such a parameter scale would take $70$ years. This staggering cost highlights the complexity of human tasks and the inadequacies of current AI. To further understand this phenomenon, we conduct a theoretical analysis. Our simulations suggest that human tasks possess a criticality property. As a result, autonomy requires a deep understanding of the task's underlying mechanisms. Current AI, however, does not fully grasp these mechanisms and instead relies on superficial mimicry, making it difficult to reach an autonomous level. We believe Intelligence Test can not only guide the future development of AI but also offer profound insights into the intelligence of humans ourselves. 

**Abstract (ZH)**: 智力是如何产生的？我们认为，智力并不是突然获得的恩赐或随机发生的事件，而是通过自然选择生存下来的物种所必需的一种特性。如果一个物种能够通过自然选择的考验，这便表明了其在自然界中的生存智能。从这一视角出发，我们引入了一种智能测试的方法，用于衡量任何受试者在任何任务中的智能水平。就像物种通过试错逐渐进化一样，智能测试则是通过计算成功之前的失败次数来衡量智能水平。失败次数越少，智能水平越高。当失败次数的期望值和方差都有限时，这表明已经达到了一定程度的自主智能。利用智能测试方法，我们全面评估了现有的AI系统。结果显示，虽然AI系统在简单任务中实现了一定程度的自主性，但在更复杂的任务如视觉识别、搜索、推荐和语言处理方面仍然远未达到自主性水平。虽然扩大模型规模可能会有所帮助，但这需要极其高昂的成本。预测显示，要实现通用自主性可能需要难以想象的 $10^{26}$ 个参数，即使按照摩尔定律持续发展，这一参数规模也需要70年。这一惊人的成本凸显了人类任务的复杂性和当前AI的不足。为了进一步理解这一现象，我们进行了理论分析。模拟结果显示，人类任务具有关键性特征，因此自主性需要深入理解任务的内在机制。然而，现有的AI并没有完全掌握这些机制，而是依赖于表面的模仿，这使得它们难以达到自主的水平。我们认为，智能测试不仅可以指导未来AI的发展，还可以对人类自身的智能提供深刻的见解。 

---
# REALM-Bench: A Real-World Planning Benchmark for LLMs and Multi-Agent Systems 

**Title (ZH)**: REALM-Bench：面向LLMs和多智能体系统的实时规划基准测试 

**Authors**: Longling Geng, Edward Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18836)  

**Abstract**: This benchmark suite provides a comprehensive evaluation framework for assessing both individual LLMs and multi-agent systems in real-world planning scenarios. The suite encompasses eleven designed problems that progress from basic to highly complex, incorporating key aspects such as multi-agent coordination, inter-agent dependencies, and dynamic environmental disruptions. Each problem can be scaled along three dimensions: the number of parallel planning threads, the complexity of inter-dependencies, and the frequency of unexpected disruptions requiring real-time adaptation. The benchmark includes detailed specifications, evaluation metrics, and baseline implementations using contemporary frameworks like LangGraph, enabling rigorous testing of both single-agent and multi-agent planning capabilities. Through standardized evaluation criteria and scalable complexity, this benchmark aims to drive progress in developing more robust and adaptable AI planning systems for real-world applications. 

**Abstract (ZH)**: 本基准套件为评估单一大规模语言模型（LLM）和多智能体系统在现实规划场景中的表现提供了一个全面的评估框架。该套件包含十一个设计问题，从基础问题逐步过渡到高度复杂的复杂问题，涵盖了多智能体协调、智能体间依赖关系以及动态环境干扰等关键方面。每个问题可以在三个维度上进行扩展：并行规划线程的数量、相互依赖关系的复杂性以及需要实时适应的意外干扰的频率。基准测试包括详细的规范、评价指标以及使用现代框架（如LangGraph）的基线实现，从而能够对单一智能体和多智能体规划能力进行严格的测试。通过标准化的评估标准和可扩展的复杂性，该基准测试旨在推动开发更 robust 和适应性强的 AI 规划系统，以应用于实际场景。 

---
# Data-Efficient Multi-Agent Spatial Planning with LLMs 

**Title (ZH)**: 基于LLM的高数据效率多智能体空间规划 

**Authors**: Huangyuan Su, Aaron Walsman, Daniel Garces, Sham Kakade, Stephanie Gil  

**Link**: [PDF](https://arxiv.org/pdf/2502.18822)  

**Abstract**: In this project, our goal is to determine how to leverage the world-knowledge of pretrained large language models for efficient and robust learning in multiagent decision making. We examine this in a taxi routing and assignment problem where agents must decide how to best pick up passengers in order to minimize overall waiting time. While this problem is situated on a graphical road network, we show that with the proper prompting zero-shot performance is quite strong on this task. Furthermore, with limited fine-tuning along with the one-at-a-time rollout algorithm for look ahead, LLMs can out-compete existing approaches with 50 times fewer environmental interactions. We also explore the benefits of various linguistic prompting approaches and show that including certain easy-to-compute information in the prompt significantly improves performance. Finally, we highlight the LLM's built-in semantic understanding, showing its ability to adapt to environmental factors through simple prompts. 

**Abstract (ZH)**: 在本项目中，我们的目标是探讨如何利用预训练大规模语言模型的世界知识，以实现多智能体决策中的高效和稳健学习。我们在一个出租车路线分配问题上进行研究，该问题要求智能体决定如何最有效地接载乘客，以最小化整体等待时间。虽然该问题基于图形化的道路网络，我们发现通过适当的提示，预训练模型在该任务上的零样本性能非常强劲。此外，通过有限的微调与一次接一个智能体的前瞻算法（one-at-a-time rollout algorithm）相结合，语言模型可以在仅需现有方法五分之一的环境交互次数的情况下，超越现有方法。我们还探讨了不同语言提示方法的优势，并表明将某些易于计算的信息包含在提示中可以显著提高性能。最后，我们展示了语言模型内置的语义理解能力，表明通过简单的提示，它可以适应环境因素。 

---
# Holistic Audit Dataset Generation for LLM Unlearning via Knowledge Graph Traversal and Redundancy Removal 

**Title (ZH)**: 基于知识图谱遍历和冗余删除的全面审计数据集生成以实现LLM去学习 

**Authors**: Weipeng Jiang, Juan Zhai, Shiqing Ma, Ziyan Lei, Xiaofei Xie, Yige Wang, Chao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18810)  

**Abstract**: In recent years, Large Language Models (LLMs) have faced increasing demands to selectively remove sensitive information, protect privacy, and comply with copyright regulations through unlearning, by Machine Unlearning. While evaluating unlearning effectiveness is crucial, existing benchmarks are limited in scale and comprehensiveness, typically containing only a few hundred test cases. We identify two critical challenges in generating holistic audit datasets: ensuring audit adequacy and handling knowledge redundancy between forget and retain dataset. To address these challenges, we propose HANKER, an automated framework for holistic audit dataset generation leveraging knowledge graphs to achieve fine-grained coverage and eliminate redundant knowledge. Applying HANKER to the popular MUSE benchmark, we successfully generated over 69,000 and 111,000 audit cases for the News and Books datasets respectively, identifying thousands of knowledge memorization instances that the previous benchmark failed to detect. Our empirical analysis uncovers how knowledge redundancy significantly skews unlearning effectiveness metrics, with redundant instances artificially inflating the observed memorization measurements ROUGE from 19.7% to 26.1% and Entailment Scores from 32.4% to 35.2%, highlighting the necessity of systematic deduplication for accurate assessment. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）面临着越来越大的需求，即通过机器遗忘（Machine Unlearning）来选择性地删除敏感信息、保护隐私并遵守版权规定。评估遗忘效果至关重要，但现有的基准在规模和全面性上存在局限性，通常仅包含几百个测试案例。我们确定了生成全面审计数据集的两大关键挑战：确保审计充分性和处理遗忘集和保留集之间的知识冗余。为应对这些挑战，我们提出了一种名为HANKER的自动化框架，该框架利用知识图谱实现了细粒度的覆盖并消除了冗余知识。将HANKER应用于流行的MUSE基准，我们成功为新闻和书籍数据集分别生成了超过69,000个和111,000个审计案例，发现了先前基准未能检测到的数千个知识记忆实例。我们的实证分析揭示了知识冗余如何显著歪曲遗忘效果的度量指标，冗余实例人为地将ROUGE测量值从19.7%提高到26.1%，将逻辑蕴含分数从32.4%提高到35.2%。这强调了系统去重对于准确评估的必要性。 

---
# Like Father, Like Son: Kinship-Aware Preference Mapping (KARMA) for Automatic Alignment in Large Language Models 

**Title (ZH)**: 亲如父子：基于亲缘关系的偏好映射（KARMA）在大规模语言模型中实现自动对齐 

**Authors**: Jeesu Jung, Chanjun Park, Sangkeun Jung  

**Link**: [PDF](https://arxiv.org/pdf/2502.18744)  

**Abstract**: Recent advancements in Large Language Model (LLM) alignment have sought to mitigate the cost of human annotations by leveraging pretrained models to generate preference data. However, existing methods often compare responses from models with substantially different capabilities, yielding superficial distinctions that fail to provide meaningful guidance on what constitutes a superior response. To address this limitation, we propose Kinship-Aware pReference MApping (KARMA), a novel framework that systematically pairs responses from models with comparable competencies. By constraining preference comparisons to outputs of similar complexity and quality, KARMA enhances the informativeness of preference data and improves the granularity of alignment signals. Empirical evaluations demonstrate that our kinship-aware approach leads to more consistent and interpretable alignment outcomes, ultimately facilitating a more principled and reliable pathway for aligning LLM behavior with human preferences. 

**Abstract (ZH)**: 近年来，大型语言模型（LLM）对齐领域的一项重要进展是通过利用预训练模型生成偏好数据来减轻human标注的成本。然而，现有方法常常比较具有显著不同能力的模型的回答，这些比较往往仅提供肤浅的区别，未能提供关于何为优秀回答的有效指导。为应对这一局限，我们提出了一种新的框架——家族意识偏好映射（Kinship-Aware Reference Mapping，简称KARMA），该框架系统性地将具有相似能力的模型的回答进行配对。通过将偏好比较限定在相似复杂度和质量的输出上，KARMA提高了偏好数据的信息含量，并增强了对齐信号的精细度。实验证据表明，我们的家族意识方法能产生更加一致和可解释的对齐结果，最终促进了更加原则性和可靠的LLM行为与人类偏好对齐途径的形成。 

---
# Talking to the brain: Using Large Language Models as Proxies to Model Brain Semantic Representation 

**Title (ZH)**: 与大脑对话：使用大规模语言模型作为代理模型构建大脑语义表示 

**Authors**: Xin Liu, Ziyue Zhang, Jingxin Nie  

**Link**: [PDF](https://arxiv.org/pdf/2502.18725)  

**Abstract**: Traditional psychological experiments utilizing naturalistic stimuli face challenges in manual annotation and ecological validity. To address this, we introduce a novel paradigm leveraging multimodal large language models (LLMs) as proxies to extract rich semantic information from naturalistic images through a Visual Question Answering (VQA) strategy for analyzing human visual semantic representation. LLM-derived representations successfully predict established neural activity patterns measured by fMRI (e.g., faces, buildings), validating its feasibility and revealing hierarchical semantic organization across cortical regions. A brain semantic network constructed from LLM-derived representations identifies meaningful clusters reflecting functional and contextual associations. This innovative methodology offers a powerful solution for investigating brain semantic organization with naturalistic stimuli, overcoming limitations of traditional annotation methods and paving the way for more ecologically valid explorations of human cognition. 

**Abstract (ZH)**: 传统的利用自然场景刺激的心理学实验在手动标注和生态效度方面面临挑战。为了解决这些问题，我们引入了一种新的范式，利用多模态大规模语言模型（LLMs）作为代理，通过视觉问答（VQA）策略从自然图像中提取丰富的语义信息，以分析人类视觉语义表征。由LLM衍生的表征能够成功预测通过fMRI测量的已确立的神经活动模式（例如，面孔、建筑物），这验证了其可行性，并揭示了跨皮层区域的层次语义组织。从LLM衍生表征构建的大脑语义网络识别出反映功能和上下文关联的有意义的聚类。这种创新的方法论为利用自然场景刺激研究大脑语义组织提供了有力的解决方案，克服了传统标注方法的局限性，并为更生态有效的探索人类认知开辟了道路。 

---
# TrajLLM: A Modular LLM-Enhanced Agent-Based Framework for Realistic Human Trajectory Simulation 

**Title (ZH)**: TrajLLM：一种模块化的大语言模型增强型基于代理的现实人类轨迹仿真框架 

**Authors**: Chenlu Ju, Jiaxin Liu, Shobhit Sinha, Hao Xue, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2502.18712)  

**Abstract**: This work leverages Large Language Models (LLMs) to simulate human mobility, addressing challenges like high costs and privacy concerns in traditional models. Our hierarchical framework integrates persona generation, activity selection, and destination prediction, using real-world demographic and psychological data to create realistic movement patterns. Both physical models and language models are employed to explore and demonstrate different methodologies for human mobility simulation. By structuring data with summarization and weighted density metrics, the system ensures scalable memory management while retaining actionable insights. Preliminary results indicate that LLM-driven simulations align with observed real-world patterns, offering scalable, interpretable insights for social problems such as urban planning, traffic management, and public health. The framework's ability to dynamically generate personas and activities enables it to provide adaptable and realistic daily routines. This study demonstrates the transformative potential of LLMs in advancing mobility modeling for societal and urban applications. The source code and interactive demo for our framework are available at this https URL. 

**Abstract (ZH)**: 本研究利用大规模语言模型（LLMs）模拟人类移动，解决了传统模型中成本高和隐私保护等挑战。我们的分层框架整合了个性生成、活动选择和目的地预测，并利用真实世界的人口统计和心理数据创建真实的人类移动模式。在使用物理模型和语言模型探索和演示不同的移动模式模拟方法的同时，系统通过使用摘要技术和加权密度度量来结构化数据，确保了可扩展的记忆管理并保留了可操作的信息。初步结果表明，由LLM驱动的模拟与现实世界观察到的模式一致，为城市规划、交通管理以及公共健康等社会问题提供了可扩展且可解释的见解。该框架能够动态生成个性和活动，使其能够提供适应性强且真实的日常活动模式。本研究展示了LLMs在推进社会和城市应用中的移动建模方面的变革潜力。我们的框架的源代码和互动演示可在以下链接获取：[这个网址]。 

---
# Hybrid Voting-Based Task Assignment in Role-Playing Games 

**Title (ZH)**: 基于角色扮演游戏中的混合投票任务分配算法 

**Authors**: Daniel Weiner, Raj Korpan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18690)  

**Abstract**: In role-playing games (RPGs), the level of immersion is critical-especially when an in-game agent conveys tasks, hints, or ideas to the player. For an agent to accurately interpret the player's emotional state and contextual nuances, a foundational level of understanding is required, which can be achieved using a Large Language Model (LLM). Maintaining the LLM's focus across multiple context changes, however, necessitates a more robust approach, such as integrating the LLM with a dedicated task allocation model to guide its performance throughout gameplay. In response to this need, we introduce Voting-Based Task Assignment (VBTA), a framework inspired by human reasoning in task allocation and completion. VBTA assigns capability profiles to agents and task descriptions to tasks, then generates a suitability matrix that quantifies the alignment between an agent's abilities and a task's requirements. Leveraging six distinct voting methods, a pre-trained LLM, and integrating conflict-based search (CBS) for path planning, VBTA efficiently identifies and assigns the most suitable agent to each task. While existing approaches focus on generating individual aspects of gameplay, such as single quests, or combat encounters, our method shows promise when generating both unique combat encounters and narratives because of its generalizable nature. 

**Abstract (ZH)**: 在角色扮演游戏（RPG）中，沉浸感的水平至关重要，尤其是在游戏中的人物代理向玩家传达任务、提示或想法时。为了使代理能够准确地解释玩家的情绪状态和情境细微差别，需要具备一定的理解和认知基础，这可以通过使用大规模语言模型（LLM）来实现。然而，维持代理在多个情境变化下的专注度需要更为稳固的方法，例如将LLM与专门的任务分配模型相结合，以在整个游戏过程中引导其表现。为此，我们提出了基于投票的任务分配框架（Voting-Based Task Assignment, VBTA），该框架借鉴了人类在任务分配和完成中的推理机制。VBTA将能力配置文件分配给代理，将任务描述分配给任务，然后生成一个适合度矩阵，量化代理能力与任务需求之间的匹配程度。通过利用六种不同的投票方法、预训练的LLM以及冲突基于搜索（CBS）进行路径规划，VBTA能够高效地识别并分配最适合的代理给每个任务。虽然现有的方法主要集中在生成游戏的个别方面，例如单个任务或战斗遭遇，但我们的方法由于其通用性，显示出在生成独特战斗遭遇和叙述方面的潜力。 

---
# Speaking the Right Language: The Impact of Expertise Alignment in User-AI Interactions 

**Title (ZH)**: 说得恰到好处：专家水平契合对用户-AI交互的影响 

**Authors**: Shramay Palta, Nirupama Chandrasekaran, Rachel Rudinger, Scott Counts  

**Link**: [PDF](https://arxiv.org/pdf/2502.18685)  

**Abstract**: Using a sample of 25,000 Bing Copilot conversations, we study how the agent responds to users of varying levels of domain expertise and the resulting impact on user experience along multiple dimensions. Our findings show that across a variety of topical domains, the agent largely responds at proficient or expert levels of expertise (77% of conversations) which correlates with positive user experience regardless of the user's level of expertise. Misalignment, such that the agent responds at a level of expertise below that of the user, has a negative impact on overall user experience, with the impact more profound for more complex tasks. We also show that users engage more, as measured by the number of words in the conversation, when the agent responds at a level of expertise commensurate with that of the user. Our findings underscore the importance of alignment between user and AI when designing human-centered AI systems, to ensure satisfactory and productive interactions. 

**Abstract (ZH)**: 通过研究一个包含25,000次必应AI助手对话的样本，我们探讨了该智能代理在面对不同专业水平用户时的响应方式，以及这种响应对用户体验在多个维度上的影响。我们的研究发现，在各种主题领域中，智能代理的响应大多达到了熟练或专家级别的专业水平（占77%的对话），这与用户的体验满意度相关，无论用户的专业水平如何。当智能代理的响应专业水平低于用户时，这种不匹配会对用户体验产生负面影响，尤其在复杂任务中表现得更为明显。此外，我们还发现，当智能代理的响应专业水平与用户相当时，用户的对话参与度会更高，表现为对话中词汇量的增加。我们的研究强调，在设计以人为本的人工智能系统时，保持用户和AI之间的匹配性对于确保满意的和高效的交互至关重要。 

---
# Independent Mobility GPT (IDM-GPT): A Self-Supervised Multi-Agent Large Language Model Framework for Customized Traffic Mobility Analysis Using Machine Learning Models 

**Title (ZH)**: 独立移动GPT（IDM-GPT）：一种基于自我监督的多代理大型语言模型框架，用于使用机器学习模型进行定制化交通移动分析 

**Authors**: Fengze Yang, Xiaoyue Cathy Liu, Lingjiu Lu, Bingzhang Wang, Chenxi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18652)  

**Abstract**: With the urbanization process, an increasing number of sensors are being deployed in transportation systems, leading to an explosion of big data. To harness the power of this vast transportation data, various machine learning (ML) and artificial intelligence (AI) methods have been introduced to address numerous transportation challenges. However, these methods often require significant investment in data collection, processing, storage, and the employment of professionals with expertise in transportation and ML. Additionally, privacy issues are a major concern when processing data for real-world traffic control and management. To address these challenges, the research team proposes an innovative Multi-agent framework named Independent Mobility GPT (IDM-GPT) based on large language models (LLMs) for customized traffic analysis, management suggestions, and privacy preservation. IDM-GPT efficiently connects users, transportation databases, and ML models economically. IDM-GPT trains, customizes, and applies various LLM-based AI agents for multiple functions, including user query comprehension, prompts optimization, data analysis, model selection, and performance evaluation and enhancement. With IDM-GPT, users without any background in transportation or ML can efficiently and intuitively obtain data analysis and customized suggestions in near real-time based on their questions. Experimental results demonstrate that IDM-GPT delivers satisfactory performance across multiple traffic-related tasks, providing comprehensive and actionable insights that support effective traffic management and urban mobility improvement. 

**Abstract (ZH)**: 随着城市化进程的推进，越来越多的传感器被部署在交通系统中，产生了大量数据。为了利用这些海量的交通数据，各种机器学习（ML）和人工智能（AI）方法被引入以应对各种交通挑战。然而，这些方法往往需要大量投资于数据采集、处理、存储以及需要具备交通和ML专业知识的专业人员。此外，在处理用于实时交通控制和管理的数据时，隐私问题也是一项主要关注点。为了解决这些挑战，研究团队提出了一种基于大规模语言模型（LLMs）的创新多代理框架，名为独立移动GPT（IDM-GPT），用于定制化的交通分析、管理建议以及隐私保护。IDM-GPT有效地将用户、交通数据库和ML模型经济地连接起来。IDM-GPT训练、定制并应用于多种基于LLM的AI代理进行多种功能，包括用户查询理解、提示优化、数据分析、模型选择、以及性能评估和提升。借助IDM-GPT，即使没有交通或ML背景的用户，也可以基于问题获得高效且直观的数据分析和个性化建议，实现近乎实时的响应。实验结果表明，IDM-GPT在多个与交通相关的任务中提供了令人满意的表现，提供了全面且可操作的见解，支持有效的交通管理和城市流动性改进。 

---
# Automated Knowledge Component Generation and Knowledge Tracing for Coding Problems 

**Title (ZH)**: 自动知识组件生成与编码问题的知识追踪 

**Authors**: Zhangqi Duan, Nigel Fernandez, Sri Kanakadandi, Bita Akram, Andrew Lan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18632)  

**Abstract**: Knowledge components (KCs) mapped to problems help model student learning, tracking their mastery levels on fine-grained skills thereby facilitating personalized learning and feedback in online learning platforms. However, crafting and tagging KCs to problems, traditionally performed by human domain experts, is highly labor-intensive. We present a fully automated, LLM-based pipeline for KC generation and tagging for open-ended programming problems. We also develop an LLM-based knowledge tracing (KT) framework to leverage these LLM-generated KCs, which we refer to as KCGen-KT. We conduct extensive quantitative and qualitative evaluations validating the effectiveness of KCGen-KT. On a real-world dataset of student code submissions to open-ended programming problems, KCGen-KT outperforms existing KT methods. We investigate the learning curves of generated KCs and show that LLM-generated KCs have a comparable level-of-fit to human-written KCs under the performance factor analysis (PFA) model. We also conduct a human evaluation to show that the KC tagging accuracy of our pipeline is reasonably accurate when compared to that by human domain experts. 

**Abstract (ZH)**: 知识组件（KCs）映射到问题可以帮助建模学生的学习过程，通过追踪他们在细粒度技能上的掌握水平，从而在在线学习平台上促进个性化学习和反馈。然而，将KCs与问题联系起来的传统工作——通常由人类领域专家完成——是非常劳动密集型的。我们提出了一种基于LLM的全自动流水线，用于生成和标注开放性编程问题的KCs。我们还开发了一种基于LLM的知识追踪（KT）框架，利用这些LLM生成的KCs，我们称之为KCGen-KT。我们进行了广泛的定量和定性评估，验证了KCGen-KT的有效性。在实际的学生代码提交数据集上，KCGen-KT比现有方法表现更优。我们研究了生成的KCs的学习曲线，并显示了在绩效因素分析（PFA）模型下，LLM生成的KCs与手写KCs具有相当的拟合度。我们还进行了一项人力评估，结果表明，与人类领域专家相比，我们流水线的KCs标注准确率是合理的。 

---
# CuDIP: Enhancing Theorem Proving in LLMs via Curriculum Learning-based Direct Preference Optimization 

**Title (ZH)**: CuDIP：通过基于课程学习的直接偏好优化提升大语言模型的定理证明能力 

**Authors**: Shuming Shi, Ruobing Zuo, Gaolei He, Jianlin Wang, Chenyang Xu, Zhengfeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18532)  

**Abstract**: Automated theorem proving (ATP) is one of the most challenging mathematical reasoning tasks for Large Language Models (LLMs). Most existing LLM-based ATP methods rely on supervised fine-tuning, which results in a limited alignment between the theorem proving process and human preferences. Direct Preference Optimization (DPO), which aligns LLMs with human preferences, has shown positive effects for certain tasks. However, the lack of high-quality preference data for theorem proving presents a significant challenge. In this paper, we innovatively apply DPO to formal automated theorem proving and introduces a Curriculum Learning-based DPO Iterative Theorem Proving (CuDIP) method. Specifically, we propose a method for constructing preference data which utilizes LLMs and existing theorem proving data to enhance the diversity of the preference data while reducing the reliance on human preference annotations. We then integrate this preference data construction method with curriculum learning to iteratively fine-tune the theorem proving model through DPO. Experimental results on the MiniF2F and ProofNet datasets demonstrate the effectiveness of the proposed method. 

**Abstract (ZH)**: 自动定理证明（ATP）是大型语言模型（LLMs）面临的最具挑战性的数学推理任务之一。现有的大多数基于LLM的ATP方法依赖于监督微调，这导致定理证明过程与人类偏好之间的对齐有限。直接受偏好优化（Direct Preference Optimization, DPO），该方法通过调整LLM与人类偏好之间的对齐来改善某些任务的效果。然而，高质量的偏好数据的缺乏为定理证明带来了重大挑战。在本文中，我们创新地将DPO应用到形式化的自动定理证明中，并提出了一种基于课程学习的DPO迭代定理证明（Curriculum-based Direct Preference Optimization Iterative Theorem Proving, CuDIP）方法。具体而言，我们提出了一种利用LLM和现有定理证明数据构建偏好数据的方法，以增强偏好数据的多样性，同时减少对人类偏好注释的依赖。然后，我们将这种偏好数据构建方法与课程学习结合，通过DPO迭代微调定理证明模型。在MiniF2F和ProofNet数据集上的实验结果表明了所提方法的有效性。 

---
# Enhancing Hepatopathy Clinical Trial Efficiency: A Secure, Large Language Model-Powered Pre-Screening Pipeline 

**Title (ZH)**: 增强肝脏疾病临床试验效率：一种安全的大规模语言模型驱动的预筛查流程 

**Authors**: Xiongbin Gui, Hanlin Lv, Xiao Wang, Longting Lv, Yi Xiao, Lei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18531)  

**Abstract**: Background: Recruitment for cohorts involving complex liver diseases, such as hepatocellular carcinoma and liver cirrhosis, often requires interpreting semantically complex criteria. Traditional manual screening methods are time-consuming and prone to errors. While AI-powered pre-screening offers potential solutions, challenges remain regarding accuracy, efficiency, and data privacy. Methods: We developed a novel patient pre-screening pipeline that leverages clinical expertise to guide the precise, safe, and efficient application of large language models. The pipeline breaks down complex criteria into a series of composite questions and then employs two strategies to perform semantic question-answering through electronic health records - (1) Pathway A, Anthropomorphized Experts' Chain of Thought strategy, and (2) Pathway B, Preset Stances within an Agent Collaboration strategy, particularly in managing complex clinical reasoning scenarios. The pipeline is evaluated on three key metrics-precision, time consumption, and counterfactual inference - at both the question and criterion levels. Results: Our pipeline achieved high precision (0.921, in criteria level) and efficiency (0.44s per task). Pathway B excelled in complex reasoning, while Pathway A was effective in precise data extraction with faster processing times. Both pathways achieved comparable precision. The pipeline showed promising results in hepatocellular carcinoma (0.878) and cirrhosis trials (0.843). Conclusions: This data-secure and time-efficient pipeline shows high precision in hepatopathy trials, providing promising solutions for streamlining clinical trial workflows. Its efficiency and adaptability make it suitable for improving patient recruitment. And its capability to function in resource-constrained environments further enhances its utility in clinical settings. 

**Abstract (ZH)**: 背景：涉及复杂肝脏疾病的队列研究，如肝细胞癌和肝硬化，常常需要解读复杂的筛选标准。传统的手工筛查方法耗时且容易出错。虽然基于人工智能的预筛查提供了潜在解决方案，但在准确性和效率以及数据隐私方面仍然存在挑战。方法：我们开发了一种新的患者预筛查管道，该管道利用临床知识指导大型语言模型的精确、安全和高效应用。该管道将复杂的筛选标准分解为一系列复合问题，然后通过电子健康记录执行语义问题回答，具体策略包括：（1）途径A：拟人化专家的思维链策略；（2）途径B：代理合作中的预设立场策略，特别适用于处理复杂的临床推理情景。该管道从问题和标准层面分别以精度、耗时和反事实推理为评价指标进行了评估。结果：我们的管道在标准层面达到了高精度（0.921）和高效率（每任务0.44秒）。途径B在复杂推理方面表现出色，而途径A在精确数据提取方面更有效，且处理时间更短。两种途径在精度上表现相当。该管道在肝细胞癌（0.878）和肝硬化临床试验（0.843）中显示出有前景的结果。结论：这种数据安全且高效的工作流管道在肝病临床试验中显示出了高精度，为简化临床试验流程提供了有前景的解决方案。其高效性和适应性使其适用于提高患者招募。此外，其在资源受限环境中运行的能力进一步增强了其在临床环境中的实用性。 

---
# Hi Robot: Open-Ended Instruction Following with Hierarchical Vision-Language-Action Models 

**Title (ZH)**: 《嗨，机器人：基于层次视觉-语言-行动模型的开放指令跟随》 

**Authors**: Lucy Xiaoyang Shi, Brian Ichter, Michael Equi, Liyiming Ke, Karl Pertsch, Quan Vuong, James Tanner, Anna Walling, Haohuan Wang, Niccolo Fusai, Adrian Li-Bell, Danny Driess, Lachy Groom, Sergey Levine, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2502.19417)  

**Abstract**: Generalist robots that can perform a range of different tasks in open-world settings must be able to not only reason about the steps needed to accomplish their goals, but also process complex instructions, prompts, and even feedback during task execution. Intricate instructions (e.g., "Could you make me a vegetarian sandwich?" or "I don't like that one") require not just the ability to physically perform the individual steps, but the ability to situate complex commands and feedback in the physical world. In this work, we describe a system that uses vision-language models in a hierarchical structure, first reasoning over complex prompts and user feedback to deduce the most appropriate next step to fulfill the task, and then performing that step with low-level actions. In contrast to direct instruction following methods that can fulfill simple commands ("pick up the cup"), our system can reason through complex prompts and incorporate situated feedback during task execution ("that's not trash"). We evaluate our system across three robotic platforms, including single-arm, dual-arm, and dual-arm mobile robots, demonstrating its ability to handle tasks such as cleaning messy tables, making sandwiches, and grocery shopping. 

**Abstract (ZH)**: 能够在开放环境中执行多种不同任务的通用机器人不仅需要推理出完成目标所需的步骤，还必须能够处理复杂指令、提示以及在任务执行过程中提供的反馈。复杂的指令（例如，“能为我做一个素食三明治吗？”或“我不喜欢那个”）不仅需要执行个体步骤的能力，还需要将复杂的命令和反馈置于物理世界中。在这项工作中，我们描述了一个利用视觉-语言模型分层结构的系统，首先通过推理复杂的提示和用户反馈来推导出完成任务的最优下一步，然后通过低级动作执行该步骤。与仅能执行简单指令（如“拿起杯子”）的方法不同，我们的系统能够通过复杂的提示进行推理，并在任务执行过程中整合环境反馈（如“那不是垃圾”）。我们将在三个不同的机器人平台上评估该系统，包括单臂机器人、双臂机器人和双臂移动机器人，展示了其完成清理杂乱的桌子、制作三明治和采购杂货等任务的能力。 

---
# Norm Growth and Stability Challenges in Localized Sequential Knowledge Editing 

**Title (ZH)**: 局部顺序知识编辑中的范式增长与稳定性挑战 

**Authors**: Akshat Gupta, Christine Fang, Atahan Ozdemir, Maochuan Lu, Ahmed Alaa, Thomas Hartvigsen, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2502.19416)  

**Abstract**: This study investigates the impact of localized updates to large language models (LLMs), specifically in the context of knowledge editing - a task aimed at incorporating or modifying specific facts without altering broader model capabilities. We first show that across different post-training interventions like continuous pre-training, full fine-tuning and LORA-based fine-tuning, the Frobenius norm of the updated matrices always increases. This increasing norm is especially detrimental for localized knowledge editing, where only a subset of matrices are updated in a model . We reveal a consistent phenomenon across various editing techniques, including fine-tuning, hypernetwork-based approaches, and locate-and-edit methods: the norm of the updated matrix invariably increases with successive updates. Such growth disrupts model balance, particularly when isolated matrices are updated while the rest of the model remains static, leading to potential instability and degradation of downstream performance. Upon deeper investigations of the intermediate activation vectors, we find that the norm of internal activations decreases and is accompanied by shifts in the subspaces occupied by these activations, which shows that these activation vectors now occupy completely different regions in the representation space compared to the unedited model. With our paper, we highlight the technical challenges with continuous and localized sequential knowledge editing and their implications for maintaining model stability and utility. 

**Abstract (ZH)**: 本研究探讨了大型语言模型（LLMs）局部更新对其知识编辑任务的影响，即在不改变模型更广泛能力的前提下，将特定事实纳入或修改特定知识的任务。我们首先表明，在不同后续训练干预措施（如连续前训练、全程微调和LORA基微调）中，更新矩阵的Frobenius范数始终增加。这种范数的增加尤其对局部知识编辑有害，在这种编辑中，只有模型的一部分矩阵被更新。无论使用哪种编辑技术（包括微调、超网络方法以及定位并编辑方法），我们一致发现了这样一个现象：随着更新的连续进行，更新矩阵的范数始终增加。这种增长扰乱了模型的平衡，尤其是仅更新孤立的矩阵而模型的其余部分保持静态时，可能导致模型不稳定和下游性能下降。在更深入地研究中间激活向量后，我们发现内部激活的范数减小，并伴随这些激活占有的子空间发生了变化，表明这些激活向量现在在表示空间中占据了与未编辑模型完全不同的区域。通过本文，我们强调了持续性和局部性序列知识编辑的技术挑战及其对保持模型稳定性和实用性的潜在影响。 

---
# Project Alexandria: Towards Freeing Scientific Knowledge from Copyright Burdens via LLMs 

**Title (ZH)**: 亚历山大项目：通过大语言模型缓解版权负担，释放科学知识的研究方向 

**Authors**: Christoph Schuhmann, Gollam Rabby, Ameya Prabhu, Tawsif Ahmed, Andreas Hochlehnert, Huu Nguyen, Nick Akinci Heidrich, Ludwig Schmidt, Robert Kaczmarczyk, Sören Auer, Jenia Jitsev, Matthias Bethge  

**Link**: [PDF](https://arxiv.org/pdf/2502.19413)  

**Abstract**: Paywalls, licenses and copyright rules often restrict the broad dissemination and reuse of scientific knowledge. We take the position that it is both legally and technically feasible to extract the scientific knowledge in scholarly texts. Current methods, like text embeddings, fail to reliably preserve factual content, and simple paraphrasing may not be legally sound. We urge the community to adopt a new idea: convert scholarly documents into Knowledge Units using LLMs. These units use structured data capturing entities, attributes and relationships without stylistic content. We provide evidence that Knowledge Units: (1) form a legally defensible framework for sharing knowledge from copyrighted research texts, based on legal analyses of German copyright law and U.S. Fair Use doctrine, and (2) preserve most (~95%) factual knowledge from original text, measured by MCQ performance on facts from the original copyrighted text across four research domains. Freeing scientific knowledge from copyright promises transformative benefits for scientific research and education by allowing language models to reuse important facts from copyrighted text. To support this, we share open-source tools for converting research documents into Knowledge Units. Overall, our work posits the feasibility of democratizing access to scientific knowledge while respecting copyright. 

**Abstract (ZH)**: 付费墙、许可协议和版权规则常常限制科学知识的广泛传播和再利用。我们认为，从学术文本中提取科学知识在法律和技术上都是可行的。当前的方法，如文本嵌入，无法可靠地保留事实内容，而简单的改写可能在法律上也不足为据。我们呼吁学术界采用一个新思路：利用大型语言模型（LLM）将学术文档转换为知识单元（Knowledge Units）。这些单元使用结构化数据来捕捉实体、属性和关系，而不包含风格性内容。我们提供了证据表明，知识单元：（1）构成了一种基于德国著作权法和美国合理使用原则的法律防御框架，用于分享受版权保护的研究文本中的知识；（2）从原始版权文本中保留了大多数（约95%）的事实知识，通过在一个研究领域中的多项选择题（MCQ）测试来衡量这些事实知识的绩效。从版权中解放科学知识有望为科学研究和教育带来变革性的益处，允许语言模型重新利用受版权保护文本中的重要事实。为了支持这一点，我们分享了将研究文档转换为知识单元的开源工具。总体而言，我们的工作提出了在尊重版权的同时实现科学知识民主化访问的可能性。 

---
# Code to Think, Think to Code: A Survey on Code-Enhanced Reasoning and Reasoning-Driven Code Intelligence in LLMs 

**Title (ZH)**: 从代码中思考，从思考中编码：关于代码增强推理与推理驱动的代码智能在大语言模型中的综述 

**Authors**: Dayu Yang, Tianyang Liu, Daoan Zhang, Antoine Simoulin, Xiaoyi Liu, Yuwei Cao, Zhaopu Teng, Xin Qian, Grey Yang, Jiebo Luo, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2502.19411)  

**Abstract**: In large language models (LLMs), code and reasoning reinforce each other: code offers an abstract, modular, and logic-driven structure that supports reasoning, while reasoning translates high-level goals into smaller, executable steps that drive more advanced code intelligence. In this study, we examine how code serves as a structured medium for enhancing reasoning: it provides verifiable execution paths, enforces logical decomposition, and enables runtime validation. We also explore how improvements in reasoning have transformed code intelligence from basic completion to advanced capabilities, enabling models to address complex software engineering tasks through planning and debugging. Finally, we identify key challenges and propose future research directions to strengthen this synergy, ultimately improving LLM's performance in both areas. 

**Abstract (ZH)**: 在大规模语言模型（LLMs）中，代码和推理相互强化：代码提供了抽象、模块化和逻辑驱动的结构，支持推理；而推理则将高层次的目标转化为更小的、可执行的步骤，驱动更为高级的代码智能。在这项研究中，我们探讨了代码作为增强推理的结构化媒介的作用：它提供了可验证的执行路径，强制执行逻辑分解，并允许运行时验证。我们还研究了推理改善如何将代码智能从基本完成提升到高级能力，从而使模型能够通过规划和调试来应对复杂的软件工程任务。最后，我们指出了关键挑战，并提出了未来的研究方向，以加强这种协同作用，最终提高LLMs在这两个方面的性能。 

---
# Less or More: Towards Glanceable Explanations for LLM Recommendations Using Ultra-Small Devices 

**Title (ZH)**: 更少还是更多：面向超小型设备的LLM推荐解释的凝视可读性探索 

**Authors**: Xinru Wang, Mengjie Yu, Hannah Nguyen, Michael Iuzzolino, Tianyi Wang, Peiqi Tang, Natasha Lynova, Co Tran, Ting Zhang, Naveen Sendhilnathan, Hrvoje Benko, Haijun Xia, Tanya Jonker  

**Link**: [PDF](https://arxiv.org/pdf/2502.19410)  

**Abstract**: Large Language Models (LLMs) have shown remarkable potential in recommending everyday actions as personal AI assistants, while Explainable AI (XAI) techniques are being increasingly utilized to help users understand why a recommendation is given. Personal AI assistants today are often located on ultra-small devices such as smartwatches, which have limited screen space. The verbosity of LLM-generated explanations, however, makes it challenging to deliver glanceable LLM explanations on such ultra-small devices. To address this, we explored 1) spatially structuring an LLM's explanation text using defined contextual components during prompting and 2) presenting temporally adaptive explanations to users based on confidence levels. We conducted a user study to understand how these approaches impacted user experiences when interacting with LLM recommendations and explanations on ultra-small devices. The results showed that structured explanations reduced users' time to action and cognitive load when reading an explanation. Always-on structured explanations increased users' acceptance of AI recommendations. However, users were less satisfied with structured explanations compared to unstructured ones due to their lack of sufficient, readable details. Additionally, adaptively presenting structured explanations was less effective at improving user perceptions of the AI compared to the always-on structured explanations. Together with users' interview feedback, the results led to design implications to be mindful of when personalizing the content and timing of LLM explanations that are displayed on ultra-small devices. 

**Abstract (ZH)**: 大型语言模型（LLMs）在作为个人AI助手推荐日常生活行动方面显示出了显著的潜力，而可解释的人工智能（XAI）技术也越来越被用于帮助用户理解为什么给出某种建议。当前，个人AI助手通常位于如智能手表这样的超小型设备上，这些设备的屏幕空间有限。然而，LLM生成的解释过于冗长，使得在超小型设备上提供简洁明了的LLM解释变得具有挑战性。为了解决这个问题，我们探索了以下两种方法：1）在提示过程中通过定义的上下文组件对LLM的解释文本进行空间结构化，并2）基于用户自信水平提供时间适应性解释。我们进行了一项用户研究，以了解这些方法在用户与超小型设备上的LLM建议和解释交互时如何影响用户体验。研究结果表明，结构化解释能够减少用户阅读解释所需的时间和认知负担。持续提供结构化解释能够增加用户对AI建议的接受度。然而，与非结构化解释相比，用户对结构化解释的满意度较低，因为它们缺乏足够的可读细节。此外，适应性呈现结构化解释对提高用户对AI的感知效果不如持续提供结构化解释显著。结合用户访谈反馈，研究结果为在超小型设备上个性化展示LLM解释的内容和时间提供了设计启示。 

---
# Multi-modal Contrastive Learning for Tumor-specific Missing Modality Synthesis 

**Title (ZH)**: 多模态对比学习在肿瘤特异性缺失模态合成中的应用 

**Authors**: Minjoo Lim, Bogyeong Kang, Tae-Eui Kam  

**Link**: [PDF](https://arxiv.org/pdf/2502.19390)  

**Abstract**: Multi-modal magnetic resonance imaging (MRI) is essential for providing complementary information about brain anatomy and pathology, leading to more accurate diagnoses. However, obtaining high-quality multi-modal MRI in a clinical setting is difficult due to factors such as time constraints, high costs, and patient movement artifacts. To overcome this difficulty, there is increasing interest in developing generative models that can synthesize missing target modality images from the available source ones. Therefore, we design a generative model for missing MRI that integrates multi-modal contrastive learning with a focus on critical tumor regions. Specifically, we integrate multi-modal contrastive learning, tailored for multiple source modalities, and enhance its effectiveness by selecting features based on entropy during the contrastive learning process. Additionally, our network not only generates the missing target modality images but also predicts segmentation outputs, simultaneously. This approach improves the generator's capability to precisely generate tumor regions, ultimately improving performance in downstream segmentation tasks. By leveraging a combination of contrastive, segmentation, and additional self-representation losses, our model effectively reflects target-specific information and generate high-quality target images. Consequently, our results in the Brain MR Image Synthesis challenge demonstrate that the proposed model excelled in generating the missing modality. 

**Abstract (ZH)**: 多模态磁共振成像（MRI）对于提供关于大脑解剖结构和病理的互补信息至关重要，有助于更准确的诊断。然而，在临床环境中获得高质量的多模态MRI面临时间限制、高成本和患者运动伪影等挑战。为了克服这些困难，越来越多的研究兴趣集中在开发生成模型，可以从现有的模态中合成缺失的目标模态图像。因此，我们设计了一个生成模型，将多模态对比学习与关键肿瘤区域的聚焦相结合。具体来说，我们将针对多种源模态定制的多模态对比学习与对比学习过程中基于熵的选择特征相结合，以增强其有效性。此外，我们的网络不仅生成缺失的目标模态图像，还同时预测分割输出。这种做法提高了生成器精确生成肿瘤区域的能力，最终提高了下游分割任务的性能。通过利用对比、分割和附加自我表示损失的组合，我们的模型有效地反映了目标特定的信息，生成高质量的目标图像。因此，我们在Brain MR图像合成挑战中的结果表明，所提出的模型在生成缺失的模态方面表现优异。 

---
# Efficient 4D fMRI ASD Classification using Spatial-Temporal-Omics-based Learning Framework 

**Title (ZH)**: 使用空间-时间-组学生化学习框架的高效4D fMRI ASD分类 

**Authors**: Ziqiao Weng, Weidong Cai, Bo Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.19386)  

**Abstract**: Autism Spectrum Disorder (ASD) is a neurodevelopmental disorder impacting social and behavioral development. Resting-state fMRI, a non-invasive tool for capturing brain connectivity patterns, aids in early ASD diagnosis and differentiation from typical controls (TC). However, previous methods, which rely on either mean time series or full 4D data, are limited by a lack of spatial information or by high computational costs. This underscores the need for an efficient solution that preserves both spatial and temporal information. In this paper, we propose a novel, simple, and efficient spatial-temporal-omics learning framework designed to efficiently extract spatio-temporal features from fMRI for ASD classification. Our approach addresses these limitations by utilizing 3D time-domain derivatives as the spatial-temporal inter-voxel omics, which preserve full spatial resolution while capturing diverse statistical characteristics of the time series at each voxel. Meanwhile, functional connectivity features serve as the spatial-temporal inter-regional omics, capturing correlations across brain regions. Extensive experiments and ablation studies on the ABIDE dataset demonstrate that our framework significantly outperforms previous methods while maintaining computational efficiency. We believe our research offers valuable insights that will inform and advance future ASD studies, particularly in the realm of spatial-temporal-omics-based learning. 

**Abstract (ZH)**: 自闭症谱系障碍（ASD）是一种影响社交和行为发展的神经发育障碍。静息状态fMRI是一种非侵入性的工具，用于捕捉大脑连接模式，有助于ASD的早期诊断以及与典型对照组（TC）的区分。然而，以往的方法要么依赖于平均时间序列数据，要么使用完整的4D数据，这两种方法分别受限于缺乏空间信息或高计算成本。这突显了需要一个既保留空间信息又保留时间信息的高效解决方案。本文提出了一种新颖、简单且高效的时空组学学习框架，旨在高效地从fMRI中提取时空特征用于ASD分类。我们的方法通过利用3D时间域导数作为时空体素间的组学特征，同时保留全空间分辨率并捕捉每个体素时间序列中多样的统计特性，从而解决了这些限制。同时，功能连接特征作为时空区域间的组学特征，用于捕捉大脑区域间的相关性。在ABIDE数据集上的广泛实验和消融研究表明，我们的框架在保持计算效率的同时显著优于之前的 方法。我们相信，我们的研究为未来的ASD研究提供了有价值的见解，并特别在时空组学基础的学习方法方面有所推进。 

---
# Preference-Based Gradient Estimation for ML-Based Approximate Combinatorial Optimization 

**Title (ZH)**: 基于偏好梯度估计的机器学习驱动近似组合优化方法 

**Authors**: Arman Mielke, Uwe Bauknecht, Thilo Strauss, Mathias Niepert  

**Link**: [PDF](https://arxiv.org/pdf/2502.19377)  

**Abstract**: Combinatorial optimization (CO) problems arise in a wide range of fields from medicine to logistics and manufacturing. While exact solutions are often not necessary, many applications require finding high-quality solutions quickly. For this purpose, we propose a data-driven approach to improve existing non-learned approximation algorithms for CO. We parameterize the approximation algorithm and train a graph neural network (GNN) to predict parameter values that lead to the best possible solutions. Our pipeline is trained end-to-end in a self-supervised fashion using gradient estimation, treating the approximation algorithm as a black box. We propose a novel gradient estimation scheme for this purpose, which we call preference-based gradient estimation. Our approach combines the benefits of the neural network and the non-learned approximation algorithm: The GNN leverages the information from the dataset to allow the approximation algorithm to find better solutions, while the approximation algorithm guarantees that the solution is feasible. We validate our approach on two well-known combinatorial optimization problems, the travelling salesman problem and the minimum k-cut problem, and show that our method is competitive with state of the art learned CO solvers. 

**Abstract (ZH)**: 组合优化（CO）问题广泛存在于医学、物流和制造业等多个领域。虽然精确解往往不是必需的，但许多应用需要快速找到高质量的解。为此，我们提出了一种数据驱动的方法，以改进现有的非学习近似算法。我们将近似算法参数化，并训练图神经网络（GNN）来预测能够得到最优解的参数值。我们的方法端到端地在半监督学习框架下进行训练，使用梯度估计来处理近似算法中的黑盒问题。为此，我们提出了一种新的梯度估计方案，称为偏好驱动的梯度估计。我们的方法结合了神经网络和非学习近似算法的优点：GNN利用数据集中的信息来使近似算法找到更好的解，而近似算法保证解的可行性。我们在旅行商问题和最小k割问题这两种广为人知的组合优化问题上验证了我们的方法，并展示了我们的方法在性能上与最先进的学习型组合优化求解器相竞争。 

---
# DataMan: Data Manager for Pre-training Large Language Models 

**Title (ZH)**: DataMan：大规模语言模型预先训练的数据管理器 

**Authors**: Ru Peng, Kexin Yang, Yawen Zeng, Junyang Lin, Dayiheng Liu, Junbo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.19363)  

**Abstract**: The performance emergence of large language models (LLMs) driven by data scaling laws makes the selection of pre-training data increasingly important. However, existing methods rely on limited heuristics and human intuition, lacking comprehensive and clear guidelines. To address this, we are inspired by ``reverse thinking'' -- prompting LLMs to self-identify which criteria benefit its performance. As its pre-training capabilities are related to perplexity (PPL), we derive 14 quality criteria from the causes of text perplexity anomalies and introduce 15 common application domains to support domain mixing. In this paper, we train a Data Manager (DataMan) to learn quality ratings and domain recognition from pointwise rating, and use it to annotate a 447B token pre-training corpus with 14 quality ratings and domain type. Our experiments validate our approach, using DataMan to select 30B tokens to train a 1.3B-parameter language model, demonstrating significant improvements in in-context learning (ICL), perplexity, and instruction-following ability over the state-of-the-art baseline. The best-performing model, based on the Overall Score l=5 surpasses a model trained with 50% more data using uniform sampling. We continue pre-training with high-rated, domain-specific data annotated by DataMan to enhance domain-specific ICL performance and thus verify DataMan's domain mixing ability. Our findings emphasize the importance of quality ranking, the complementary nature of quality criteria, and their low correlation with perplexity, analyzing misalignment between PPL and ICL performance. We also thoroughly analyzed our pre-training dataset, examining its composition, the distribution of quality ratings, and the original document sources. 

**Abstract (ZH)**: 大型语言模型（LLMs）性能的提升，源于数据规模定律的影响，使得预训练数据的选择显得日益重要。然而，现有的方法依赖于有限的经验法则和人类直觉，缺乏全面且清晰的指导方针。为了解决这一问题，我们受到“逆向思维”的启发——促使LLMs自我识别哪些标准对其性能有益。由于其预训练能力与困惑度（PPL）密切相关，我们从文本困惑度异常的原因中提炼了14个质量标准，并引入了15个常见的应用领域以支持跨领域混合。在本文中，我们训练了一个数据管理器（DataMan），使其从点wise评分中学习质量评分和领域识别能力，并使用其对一个包含447B个标记的预训练语料库进行标注，赋予其14个质量评分和领域类型。我们的实验验证了这种方法的有效性，通过使用DataMan选择30B个标记以培训一个参数量为1.3B的语言模型，展示了在上下文学习（ICL）、困惑度和指令跟随能力等方面相较于现有基准的重大改进。基于整体评分l=5的最优模型，超越了使用均匀采样训练的数据量多50%的模型。我们继续使用DataMan标注的高评分、领域特定的数据进行预训练，以增强特定领域的ICL性能，从而验证了DataMan的跨领域能力。我们的研究结果强调了质量评分的重要性、质量标准之间的互补性质以及它们与困惑度的低相关性，并分析了PPL与ICL性能之间的不匹配。我们还详细分析了我们的预训练数据集，探讨了其构成、质量评分的分布以及原始文档来源。 

---
# Physics-Based Hybrid Machine Learning for Critical Heat Flux Prediction with Uncertainty Quantification 

**Title (ZH)**: 基于物理的混合机器学习在不确定性量化下的关键热流预测 

**Authors**: Aidan Furlong, Xingang Zhao, Robert Salko, Xu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19357)  

**Abstract**: Critical heat flux is a key quantity in boiling system modeling due to its impact on heat transfer and component temperature and performance. This study investigates the development and validation of an uncertainty-aware hybrid modeling approach that combines machine learning with physics-based models in the prediction of critical heat flux in nuclear reactors for cases of dryout. Two empirical correlations, Biasi and Bowring, were employed with three machine learning uncertainty quantification techniques: deep neural network ensembles, Bayesian neural networks, and deep Gaussian processes. A pure machine learning model without a base model served as a baseline for comparison. This study examines the performance and uncertainty of the models under both plentiful and limited training data scenarios using parity plots, uncertainty distributions, and calibration curves. The results indicate that the Biasi hybrid deep neural network ensemble achieved the most favorable performance (with a mean absolute relative error of 1.846% and stable uncertainty estimates), particularly in the plentiful data scenario. The Bayesian neural network models showed slightly higher error and uncertainty but superior calibration. By contrast, deep Gaussian process models underperformed by most metrics. All hybrid models outperformed pure machine learning configurations, demonstrating resistance against data scarcity. 

**Abstract (ZH)**: 临界热通量是沸腾系统建模中的关键参数，因为它对传热以及组件的温度和性能有很大影响。本研究探讨了一种结合机器学习与物理基础模型的不确定性感知混合建模方法的发展与验证，该方法用于核反应堆中干涸情况下临界热通量的预测。研究中使用了两种经验关联式：Biasi 和 Bowring，并采用了三种机器学习不确定性量化技术：深度神经网络集成（Ensembles of Deep Neural Networks, DNNs）、贝叶斯神经网络（Bayesian Neural Networks, BNNs）和深度高斯过程（Deep Gaussian Processes, DGPs）。还构建了一个不包含物理基础模型的纯机器学习模型作为基线进行比较。研究重点在充分和有限训练数据场景下，通过一对一相同数量级的比较，使用一致性图、不确定性分布和校准曲线来评估模型的性能和不确定性。研究结果表明，在充分数据场景下，Biasi 混合深度神经网络集成表现出最佳性能（平均绝对相对误差为 1.846%，且具有稳定的不确定性估计）。贝叶斯神经网络模型虽然误差和不确定性略高，但在校准方面表现更优。相比之下，深度高斯过程模型在大多数指标上表现不佳。所有混合模型的性能都优于纯机器学习配置，显示出对数据稀缺性的抵御能力。 

---
# Deep Learning-Based Transfer Learning for Classification of Cassava Disease 

**Title (ZH)**: 基于深度学习的迁移学习在甘蔗病害分类中的应用 

**Authors**: Ademir G. Costa Junior, Fábio S. da Silva, Ricardo Rios  

**Link**: [PDF](https://arxiv.org/pdf/2502.19351)  

**Abstract**: This paper presents a performance comparison among four Convolutional Neural Network architectures (EfficientNet-B3, InceptionV3, ResNet50, and VGG16) for classifying cassava disease images. The images were sourced from an imbalanced dataset from a competition. Appropriate metrics were employed to address class imbalance. The results indicate that EfficientNet-B3 achieved on this task accuracy of 87.7%, precision of 87.8%, revocation of 87.8% and F1-Score of 87.7%. These findings suggest that EfficientNet-B3 could be a valuable tool to support Digital Agriculture. 

**Abstract (ZH)**: 本文比较了四种卷积神经网络架构（EfficientNet-B3、InceptionV3、ResNet50 和 VGG16）在分类甘蔗病害图像方面的性能，所用数据集来自一个竞赛，数据不平衡。采用了适当的评估指标来解决类别不平衡问题。结果表明，EfficientNet-B3 在此任务中的准确率为 87.7%，精确率为 87.8%，召回率为 87.8%，F1 分数为 87.7%。这些发现表明，EfficientNet-B3 可能是一种支持数字农业的有效工具。 

---
# Controlled Diversity: Length-optimized Natural Language Generation 

**Title (ZH)**: 控制性多样性：长度优化的自然语言生成 

**Authors**: Diana Marie Schenke, Timo Baumann  

**Link**: [PDF](https://arxiv.org/pdf/2502.19347)  

**Abstract**: LLMs are not generally able to adjust the length of their outputs based on strict length requirements, a capability that would improve their usefulness in applications that require adherence to diverse user and system requirements. We present an approach to train LLMs to acquire this capability by augmenting existing data and applying existing fine-tuning techniques, which we compare based on the trained models' adherence to the length requirement and overall response quality relative to the baseline model. Our results demonstrate that these techniques can be successfully applied to train LLMs to adhere to length requirements, with the trained models generating texts which better align to the length requirements. Our results indicate that our method may change the response quality when using training data that was not generated by the baseline model. This allows simultaneous alignment to another training objective in certain scenarios, but is undesirable otherwise. Training on a dataset containing the model's own responses eliminates this issue. 

**Abstract (ZH)**: 大型语言模型通常不具备根据严格长度要求调整其输出长度的能力，这种能力将提高其在需要遵守多样化用户和系统要求的应用中的实用性。我们提出了一种通过扩展现有数据并应用现有的微调技术来训练大型语言模型获得这种能力的方法，并且我们在训练模型在长度要求方面的遵守程度及其整体响应质量方面与基线模型进行了比较。实验结果表明，这些技术可以成功应用于训练大型语言模型以遵守长度要求，从而使训练后的模型生成的文本更好地符合长度要求。实验结果表明，当使用非基线模型生成的训练数据时，我们的方法可能改变响应质量。这在某些场景下可以同时实现对另一个训练目标的对齐，但在其他情况下是不希望的。在包含模型自身响应的数据集上进行训练可以解决这个问题。 

---
# Agentic Reward Modeling: Integrating Human Preferences with Verifiable Correctness Signals for Reliable Reward Systems 

**Title (ZH)**: 代理奖励建模：将人类偏好与可验证的正确性信号集成以构建可靠的奖励系统 

**Authors**: Hao Peng, Yunjia Qi, Xiaozhi Wang, Zijun Yao, Bin Xu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.19328)  

**Abstract**: Reward models (RMs) are crucial for the training and inference-time scaling up of large language models (LLMs). However, existing reward models primarily focus on human preferences, neglecting verifiable correctness signals which have shown strong potential in training LLMs. In this paper, we propose agentic reward modeling, a reward system that combines reward models with verifiable correctness signals from different aspects to provide reliable rewards. We empirically implement a reward agent, named RewardAgent, that combines human preference rewards with two verifiable signals: factuality and instruction following, to provide more reliable rewards. We conduct comprehensive experiments on existing reward model benchmarks and inference time best-of-n searches on real-world downstream tasks. RewardAgent significantly outperforms vanilla reward models, demonstrating its effectiveness. We further construct training preference pairs using RewardAgent and train an LLM with the DPO objective, achieving superior performance on various NLP benchmarks compared to conventional reward models. Our codes are publicly released to facilitate further research (this https URL). 

**Abstract (ZH)**: 人工智能代理（Agent-based Reward Modeling, ARM）对于大型语言模型（LLMs）的训练和推理时扩展至关重要。然而，现有的奖励模型主要关注人类偏好，忽视了在训练LLMs过程中展现出强大潜力的可验证正确性信号。本文中，我们提出了一种结合不同方面可验证正确性信号的奖励模型——代理人奖励建模（agentic reward modeling），以提供可靠的奖励。我们实验性地实现了一个名为RewardAgent的奖励代理，它结合了人类偏好奖励与两种可验证信号：事实性（factuality）和指令遵循（instruction following），以提供更可靠的奖励。我们在现有的奖励模型基准以及实际下游任务的最佳推理时间搜索中进行了全面实验。与传统奖励模型相比，RewardAgent显著表现出更高的有效性。为进一步研究，我们使用RewardAgent构建训练偏好对，并利用DPO目标训练了一个LLM，其在各种NLP基准测试中的表现优于传统奖励模型。我们的代码已公开发布，以促进进一步研究（参见：<https://github.com/PseudoPatrick/Reward-Agent>）。 

---
# Partition Tree Weighting for Non-Stationary Stochastic Bandits 

**Title (ZH)**: 非稳态随机多臂bandits的分区树加权方法 

**Authors**: Joel Veness, Marcus Hutter, Andras Gyorgy, Jordi Grau-Moya  

**Link**: [PDF](https://arxiv.org/pdf/2502.19325)  

**Abstract**: This paper considers a generalisation of universal source coding for interaction data, namely data streams that have actions interleaved with observations. Our goal will be to construct a coding distribution that is both universal \emph{and} can be used as a control policy. Allowing for action generation needs careful treatment, as naive approaches which do not distinguish between actions and observations run into the self-delusion problem in universal settings. We showcase our perspective in the context of the challenging non-stationary stochastic Bernoulli bandit problem. Our main contribution is an efficient and high performing algorithm for this problem that generalises the Partition Tree Weighting universal source coding technique for passive prediction to the control setting. 

**Abstract (ZH)**: 本文探讨了一种通用源编码在交互数据中的拓展，特别是将操作与观测交织的数据流。我们的目标是构建一个既是通用的，又能作为控制策略的编码分布。允许操作生成需要谨慎处理，因为在通用场景下，不经区分地对待操作与观测的天真方法会遇到自我错觉问题。本文在非平稳随机伯努利多臂赌博机这一具有挑战性的问题背景下展示了我们的视角。我们的主要贡献是将被动预测中的分区树加权通用源编码技术拓展到控制场景，提出了一种高效且性能出色的算法。 

---
# Shh, don't say that! Domain Certification in LLMs 

**Title (ZH)**: 不要说那个！LLM领域的认证 

**Authors**: Cornelius Emde, Alasdair Paren, Preetham Arvind, Maxime Kayser, Tom Rainforth, Thomas Lukasiewicz, Bernard Ghanem, Philip H.S. Torr, Adel Bibi  

**Link**: [PDF](https://arxiv.org/pdf/2502.19320)  

**Abstract**: Large language models (LLMs) are often deployed to perform constrained tasks, with narrow domains. For example, customer support bots can be built on top of LLMs, relying on their broad language understanding and capabilities to enhance performance. However, these LLMs are adversarially susceptible, potentially generating outputs outside the intended domain. To formalize, assess, and mitigate this risk, we introduce domain certification; a guarantee that accurately characterizes the out-of-domain behavior of language models. We then propose a simple yet effective approach, which we call VALID that provides adversarial bounds as a certificate. Finally, we evaluate our method across a diverse set of datasets, demonstrating that it yields meaningful certificates, which bound the probability of out-of-domain samples tightly with minimum penalty to refusal behavior. 

**Abstract (ZH)**: 大型语言模型（LLMs）通常被部署执行受限任务，涉及狭窄领域。例如，可以基于LLMs构建客服机器人，利用它们广泛的语言理解和能力来提升性能。然而，这些LLMs对抗性脆弱，可能会生成超出预期领域的输出。为了形式化、评估和缓解这一风险，我们引入了域认证；这是一种准确表征语言模型域外行为的保证。随后，我们提出了一种简单而有效的方法，我们称之为VALID，它提供对抗界限作为证书。最后，我们在多种多样且具有代表性的数据集上评估了我们的方法，结果显示，该方法产生了有意义的证书，可以紧密限制域外样本的概率，同时将拒绝行为的代价降到最低。 

---
# FSPO: Few-Shot Preference Optimization of Synthetic Preference Data in LLMs Elicits Effective Personalization to Real Users 

**Title (ZH)**: FSPO: 少量示例的合成偏好优化在大模型中激发对真实用户的有效个性化 

**Authors**: Anikait Singh, Sheryl Hsu, Kyle Hsu, Eric Mitchell, Stefano Ermon, Tatsunori Hashimoto, Archit Sharma, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2502.19312)  

**Abstract**: Effective personalization of LLMs is critical for a broad range of user-interfacing applications such as virtual assistants and content curation. Inspired by the strong in-context learning capabilities of LLMs, we propose Few-Shot Preference Optimization (FSPO), which reframes reward modeling as a meta-learning problem. Under this framework, an LLM learns to quickly adapt to a user via a few labeled preferences from that user, constructing a personalized reward function for them. Additionally, since real-world preference data is scarce and challenging to collect at scale, we propose careful design choices to construct synthetic preference datasets for personalization, generating over 1M synthetic personalized preferences using publicly available LLMs. In particular, to successfully transfer from synthetic data to real users, we find it crucial for the data to exhibit both high diversity and coherent, self-consistent structure. We evaluate FSPO on personalized open-ended generation for up to 1,500 synthetic users across across three domains: movie reviews, pedagogical adaptation based on educational background, and general question answering, along with a controlled human study. Overall, FSPO achieves an 87% Alpaca Eval winrate on average in generating responses that are personalized to synthetic users and a 72% winrate with real human users in open-ended question answering. 

**Abstract (ZH)**: 有效个性化大语言模型（LLM）对于诸如虚拟助手和内容个性化等广泛的应用至关重要。受LLM强大上下文适应能力的启发，我们提出了一种少样本偏好优化（FSPO）方法，将其重新定义为元学习问题。在此框架下，LLM能够通过少量标记的用户偏好快速适应用户，构建个性化奖励函数。此外，由于实际偏好数据稀缺且大规模收集困难，我们提出了一些精心设计的选择来构建用于个性化处理的合成偏好数据集，并使用公开的LLM生成了超过100万条合成个性化偏好数据。特别地，为了成功地从合成数据转移到真实用户，我们发现数据需要同时具备高度的多样性与一致且自我一致的结构。我们在三个领域（电影评论、基于教育背景的教育适应、一般问题回答）的1,500个合成用户上，以及一次受控的人类研究中对FSPO进行了评估。总体而言，FSPO在生成个性化响应方面对合成用户的平均Alpaca评估胜率为87%，在开放性问题回答中，对真实人类用户的胜率为72%。 

---
# Faithful Logic Embeddings in HOL -- A recipe to have it all: deep and shallow, automated and interactive, heavy and light, proofs and counterexamples, meta and object level 

**Title (ZH)**: HOL 中忠实的逻辑嵌入——兼备深浅度、自动化与交互性、重量级与轻量级、证明与反例、元层次与对象层次的方法 

**Authors**: Christoph Benzmüller  

**Link**: [PDF](https://arxiv.org/pdf/2502.19311)  

**Abstract**: Deep and shallow embeddings of non-classical logics in classical higher-order logic have been explored, implemented, and used in various automated reasoning tools in recent years. This paper presents a recipe for the simultaneous deployment of different forms of deep and shallow embeddings in classical higher-order logic, enabling not only flexible interactive and automated theorem proving and counterexample finding at meta and object level, but also automated faithfulness proofs between the logic embeddings. The approach, which is fruitful for logic education, research and application, is deliberately illustrated here using simple propositional modal logic. However, the work presented is conceptual in nature and not limited to such a simple logic context. 

**Abstract (ZH)**: 近年来，非经典逻辑在经典高阶逻辑中的深层和浅层嵌入已被研究、实现并在多种自动推理工具中得到应用。本文提供了一种在经典高阶逻辑中同时部署不同形式的深层和浅层嵌入的方法，不仅能够灵活地进行元级和对象级的交互式和自动定理证明及反例查找，还能进行逻辑嵌入之间的自动忠实性证明。该方法对于逻辑教育、研究和应用都具有成果性，本文故意使用简单的命题模态逻辑来说明这一方法。然而，所展示的工作是概念性的，并不限于这种简单的逻辑环境。 

---
# Anomaly Detection in Complex Dynamical Systems: A Systematic Framework Using Embedding Theory and Physics-Inspired Consistency 

**Title (ZH)**: 复杂动力系统中的异常检测：基于嵌入理论和物理启发一致性的一种系统性框架 

**Authors**: Michael Somma, Thomas Gallien, Branka Stojanovic  

**Link**: [PDF](https://arxiv.org/pdf/2502.19307)  

**Abstract**: Anomaly detection in complex dynamical systems is essential for ensuring reliability, safety, and efficiency in industrial and cyber-physical infrastructures. Predictive maintenance helps prevent costly failures, while cybersecurity monitoring has become critical as digitized systems face growing threats. Many of these systems exhibit oscillatory behaviors and bounded motion, requiring anomaly detection methods that capture structured temporal dependencies while adhering to physical consistency principles. In this work, we propose a system-theoretic approach to anomaly detection, grounded in classical embedding theory and physics-inspired consistency principles. We build upon the Fractal Whitney Embedding Prevalence Theorem, extending traditional embedding techniques to complex system dynamics. Additionally, we introduce state-derivative pairs as an embedding strategy to capture system evolution. To enforce temporal coherence, we develop a Temporal Differential Consistency Autoencoder (TDC-AE), incorporating a TDC-Loss that aligns the approximated derivatives of latent variables with their dynamic representations. We evaluate our method on the C-MAPSS dataset, a benchmark for turbofan aeroengine degradation. TDC-AE outperforms LSTMs and Transformers while achieving a 200x reduction in MAC operations, making it particularly suited for lightweight edge computing. Our findings support the hypothesis that anomalies disrupt stable system dynamics, providing a robust, interpretable signal for anomaly detection. 

**Abstract (ZH)**: 在复杂动态系统中进行异常检测对于确保工业和网络物理基础设施的安全性、可靠性和效率至关重要。预测性维护有助于预防昂贵的故障，而随着数字化系统的威胁增加，网络安全监控变得越来越关键。许多这些系统表现出振荡行为和受限运动，要求异常检测方法能够捕捉到结构化的时序依赖关系，同时遵守物理一致性原则。在本研究中，我们提出了一种基于系统理论的方法来进行异常检测，该方法以经典的嵌入理论和物理启发的一致性原则为基础。我们在此基础上扩展了传统的嵌入技术，以适应复杂系统的动态特性。此外，我们引入了状态及其导数对作为嵌入策略，以捕捉系统的演化过程。为了确保时序的一致性，我们开发了一种时序微分一致性自编码器（TDC-AE），并引入了TDC损失函数，该损失函数将潜在变量的逼近导数与它们的动态表示对齐。我们在C-MAPSS数据集上评估了该方法，该数据集是涡扇发动机退化的基准数据集。实验结果表明，TDC-AE在MAC运算次数减少200倍的情况下优于LSTMs和Transformers，使其特别适用于轻量级边缘计算。我们的研究结果支持了这样的假设，即异常破坏了稳定系统的动态行为，提供了稳健且可解释的异常检测信号。 

---
# Corporate Fraud Detection in Rich-yet-Noisy Financial Graph 

**Title (ZH)**: 富但嘈杂的财务图中的企业欺诈检测 

**Authors**: Shiqi Wang, Zhibo Zhang, Libing Fang, Cam-Tu Nguyen, Wenzhon Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.19305)  

**Abstract**: Corporate fraud detection aims to automatically recognize companies that conduct wrongful activities such as fraudulent financial statements or illegal insider trading. Previous learning-based methods fail to effectively integrate rich interactions in the company network. To close this gap, we collect 18-year financial records in China to form three graph datasets with fraud labels. We analyze the characteristics of the financial graphs, highlighting two pronounced issues: (1) information overload: the dominance of (noisy) non-company nodes over company nodes hinders the message-passing process in Graph Convolution Networks (GCN); and (2) hidden fraud: there exists a large percentage of possible undetected violations in the collected data. The hidden fraud problem will introduce noisy labels in the training dataset and compromise fraud detection results. To handle such challenges, we propose a novel graph-based method, namely, Knowledge-enhanced GCN with Robust Two-stage Learning (${\rm KeGCN}_{R}$), which leverages Knowledge Graph Embeddings to mitigate the information overload and effectively learns rich representations. The proposed model adopts a two-stage learning method to enhance robustness against hidden frauds. Extensive experimental results not only confirm the importance of interactions but also show the superiority of ${\rm KeGCN}_{R}$ over a number of strong baselines in terms of fraud detection effectiveness and robustness. 

**Abstract (ZH)**: 公司欺诈检测旨在自动识别执行欺诈性财务报表或非法内部交易等不当行为的公司。以往基于学习的方法无法有效整合公司网络中的丰富交互。为缩小这一差距，我们收集了18年的中国财务记录，形成了三个带有欺诈标签的图数据集。我们分析了财务图的特性，突出了两个显著问题：（1）信息泛滥：不相关的（噪声）非公司节点在公司节点中占据主导地位，阻碍了图卷积网络（GCN）中的消息传递过程；（2）隐藏的欺诈：收集的数据中存在大量未被检测到的违规行为。隐藏的欺诈问题将在训练数据集中引入噪声标签并影响欺诈检测结果。为应对这些挑战，我们提出了一种新的图基方法，即增强知识的两阶段学习图卷积网络（$\mathrm{KeGCN}_{R}$），该方法利用知识图嵌入来缓解信息泛滥问题，并有效学习丰富的表示。所提模型采用两阶段学习方法来增强对隐藏欺诈的鲁棒性。广泛的实验结果不仅证实了交互的重要性，还显示了$\mathrm{KeGCN}_{R}$在多个强大的基线方法中，在欺诈检测效果和鲁棒性方面均具有优势。 

---
# Combining Planning and Reinforcement Learning for Solving Relational Multiagent Domains 

**Title (ZH)**: 将规划与强化学习相结合解决关系型多智能体领域问题 

**Authors**: Nikhilesh Prabhakar, Ranveer Singh, Harsha Kokel, Sriraam Natarajan, Prasad Tadepalli  

**Link**: [PDF](https://arxiv.org/pdf/2502.19297)  

**Abstract**: Multiagent Reinforcement Learning (MARL) poses significant challenges due to the exponential growth of state and action spaces and the non-stationary nature of multiagent environments. This results in notable sample inefficiency and hinders generalization across diverse tasks. The complexity is further pronounced in relational settings, where domain knowledge is crucial but often underutilized by existing MARL algorithms. To overcome these hurdles, we propose integrating relational planners as centralized controllers with efficient state abstractions and reinforcement learning. This approach proves to be sample-efficient and facilitates effective task transfer and generalization. 

**Abstract (ZH)**: 多智能体强化学习（MARL）由于状态空间和动作空间的指数增长以及多智能体环境的非站定特性，面临着重大挑战。这导致了显著的样本效率低下，并阻碍了在多样任务上的泛化。在关系型设置中，这一复杂性进一步增加，此时领域知识至关重要但常常被现有的MARL算法所忽视。为克服这些难题，我们提议将关系规划器作为中心控制器与高效的状态抽象结合使用，并与强化学习相结合。这种 approach 证明具有样本高效性，并促进了有效的任务迁移和泛化。 

---
# Integrating Biological and Machine Intelligence: Attention Mechanisms in Brain-Computer Interfaces 

**Title (ZH)**: 将以下论文内容或标题翻译成中文，同时确保符合学术规范：

Integrating Biological and Machine Intelligence: Attention Mechanisms in Brain-Computer Interfaces

生物智能与机器智能的融合：脑机接口中的注意力机制 

**Authors**: Jiyuan Wang, Weishan Ye, Jialin He, Li Zhang, Gan Huang, Zhuliang Yu, Zhen Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19281)  

**Abstract**: With the rapid advancement of deep learning, attention mechanisms have become indispensable in electroencephalography (EEG) signal analysis, significantly enhancing Brain-Computer Interface (BCI) applications. This paper presents a comprehensive review of traditional and Transformer-based attention mechanisms, their embedding strategies, and their applications in EEG-based BCI, with a particular emphasis on multimodal data fusion. By capturing EEG variations across time, frequency, and spatial channels, attention mechanisms improve feature extraction, representation learning, and model robustness. These methods can be broadly categorized into traditional attention mechanisms, which typically integrate with convolutional and recurrent networks, and Transformer-based multi-head self-attention, which excels in capturing long-range dependencies. Beyond single-modality analysis, attention mechanisms also enhance multimodal EEG applications, facilitating effective fusion between EEG and other physiological or sensory data. Finally, we discuss existing challenges and emerging trends in attention-based EEG modeling, highlighting future directions for advancing BCI technology. This review aims to provide valuable insights for researchers seeking to leverage attention mechanisms for improved EEG interpretation and application. 

**Abstract (ZH)**: 随着深度学习的迅速发展，注意力机制在脑电图（EEG）信号分析中变得不可或缺，显著增强了脑-计算机接口（BCI）的应用。本文对传统的和基于Transformer的注意力机制、它们的嵌入策略及其在基于EEG的BCI中的应用进行了全面回顾，特别强调了多模态数据融合。通过捕捉时间、频率和空间通道中的EEG变化，注意力机制提高了特征提取、表示学习和模型稳健性。这些方法可以大致分为传统的注意力机制，它们通常与卷积和循环网络结合使用，以及基于Transformer的多头自注意力机制，在长依赖关系捕获方面表现出色。除了单模态分析，注意力机制还增强了多模态EEG的应用，促进了EEG与其他生理或感知数据的有效融合。最后，我们讨论了基于注意力的EEG建模存在的挑战和新兴趋势，强调了BCI技术发展的未来方向。本评论旨在为希望利用注意力机制以改进EEG解释和应用的研究人员提供有价值的见解。 

---
# Multiview graph dual-attention deep learning and contrastive learning for multi-criteria recommender systems 

**Title (ZH)**: 多视角图双注意深度学习与对比学习在多准则推荐系统中的应用 

**Authors**: Saman Forouzandeh, Pavel N. Krivitsky, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2502.19271)  

**Abstract**: Recommender systems leveraging deep learning models have been crucial for assisting users in selecting items aligned with their preferences and interests. However, a significant challenge persists in single-criteria recommender systems, which often overlook the diverse attributes of items that have been addressed by Multi-Criteria Recommender Systems (MCRS). Shared embedding vector for multi-criteria item ratings but have struggled to capture the nuanced relationships between users and items based on specific criteria. In this study, we present a novel representation for Multi-Criteria Recommender Systems (MCRS) based on a multi-edge bipartite graph, where each edge represents one criterion rating of items by users, and Multiview Dual Graph Attention Networks (MDGAT). Employing MDGAT is beneficial and important for adequately considering all relations between users and items, given the presence of both local (criterion-based) and global (multi-criteria) relations. Additionally, we define anchor points in each view based on similarity and employ local and global contrastive learning to distinguish between positive and negative samples across each view and the entire graph. We evaluate our method on two real-world datasets and assess its performance based on item rating predictions. The results demonstrate that our method achieves higher accuracy compared to the baseline method for predicting item ratings on the same datasets. MDGAT effectively capture the local and global impact of neighbours and the similarity between nodes. 

**Abstract (ZH)**: 利用深度学习模型的推荐系统在帮助用户选择符合其偏好的项目方面发挥了关键作用。然而，单一标准推荐系统仍面临重大挑战，这些系统往往忽略了多标准推荐系统（MCRS）已经考虑的项目多样化属性。尽管MCRS能够共享多标准项评分的嵌入向量，但在基于特定标准捕捉用户与项目间复杂关系方面仍存在困难。在本研究中，我们提出了一种基于多边二部图的新颖表示方法，该图中的每条边代表用户对项目的一个标准评分，并结合了多视图双图注意网络（MDGAT）。利用MDGAT有助于充分考虑用户与项目间的所有关系，这些关系既包括局部（基于标准）关系，也包括全局（多标准）关系。此外，我们根据相似性在每个视图中定义锚点，并采用局部和全局对比学习，以区分每个视图和整个图中正负样本之间的差异。我们在两个实际数据集上评估了该方法，并根据项目评分预测评估其性能。结果表明，与基线方法相比，该方法在相同的数据集上预测项目评分时具有更高的准确性。MDGAT有效地捕捉了邻近节点的局部和全局影响以及节点之间的相似性。 

---
# Drop-Upcycling: Training Sparse Mixture of Experts with Partial Re-initialization 

**Title (ZH)**: 滴定升级：通过部分重新初始化训练稀疏专家混合模型 

**Authors**: Taishi Nakamura, Takuya Akiba, Kazuki Fujii, Yusuke Oda, Rio Yokota, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2502.19261)  

**Abstract**: The Mixture of Experts (MoE) architecture reduces the training and inference cost significantly compared to a dense model of equivalent capacity. Upcycling is an approach that initializes and trains an MoE model using a pre-trained dense model. While upcycling leads to initial performance gains, the training progresses slower than when trained from scratch, leading to suboptimal performance in the long term. We propose Drop-Upcycling - a method that effectively addresses this problem. Drop-Upcycling combines two seemingly contradictory approaches: utilizing the knowledge of pre-trained dense models while statistically re-initializing some parts of the weights. This approach strategically promotes expert specialization, significantly enhancing the MoE model's efficiency in knowledge acquisition. Extensive large-scale experiments demonstrate that Drop-Upcycling significantly outperforms previous MoE construction methods in the long term, specifically when training on hundreds of billions of tokens or more. As a result, our MoE model with 5.9B active parameters achieves comparable performance to a 13B dense model in the same model family, while requiring approximately 1/4 of the training FLOPs. All experimental resources, including source code, training data, model checkpoints and logs, are publicly available to promote reproducibility and future research on MoE. 

**Abstract (ZH)**: 与等效容量的密集模型相比，《专家混合》（Mixture of Experts, MoE）架构显著降低了训练和推理的成本。循环使用（Upcycling）是一种方法，通过使用预训练的密集模型对MoE模型进行初始化和训练。尽管循环使用能够带来初始性能的提升，但其训练进度比从零开始训练要慢，长期来看会导致性能不佳。我们提出了一种名为Drop-Upcycling的方法，有效地解决了这个问题。

Drop-Upcycling结合了两个看似矛盾的方法：利用预训练密集模型的知识，同时统计性地重新初始化部分权重。这种方法战略性地促进专家的专业化，显著提高了MoE模型在知识获取方面的效率。大规模的实验结果显示，Drop-Upcycling在长期性能上明显优于之前的MoE构建方法，特别是在训练超过数十亿个令牌时。因此，我们的MoE模型在5.9B活跃参数下，实现了与同一模型家族中13B参数密集模型相当的性能，但训练FLOPs仅为其四分之一。我们提供所有实验资源，包括源代码、训练数据、模型检查点和日志，以促进MoE的可再现性和未来研究。

这些资源均已公开，以便促进MoE的可再现性和未来研究。 

---
# EMT: A Visual Multi-Task Benchmark Dataset for Autonomous Driving in the Arab Gulf Region 

**Title (ZH)**: EMT：阿拉伯海湾地区自动驾驶的多任务视觉基准数据集 

**Authors**: Nadya Abdel Madjid, Murad Mebrahtu, Abdelmoamen Nasser, Bilal Hassan, Naoufel Werghi, Jorge Dias, Majid Khonji  

**Link**: [PDF](https://arxiv.org/pdf/2502.19260)  

**Abstract**: This paper introduces the Emirates Multi-Task (EMT) dataset - the first publicly available dataset for autonomous driving collected in the Arab Gulf region. The EMT dataset captures the unique road topology, high traffic congestion, and distinctive characteristics of the Gulf region, including variations in pedestrian clothing and weather conditions. It contains over 30,000 frames from a dash-camera perspective, along with 570,000 annotated bounding boxes, covering approximately 150 kilometers of driving routes. The EMT dataset supports three primary tasks: tracking, trajectory forecasting and intention prediction. Each benchmark dataset is complemented with corresponding evaluations: (1) multi-agent tracking experiments, focusing on multi-class scenarios and occlusion handling; (2) trajectory forecasting evaluation using deep sequential and interaction-aware models; and (3) intention benchmark experiments conducted for predicting agents intentions from observed trajectories. The dataset is publicly available at this https URL, and pre-processing scripts along with evaluation models can be accessed at this https URL. 

**Abstract (ZH)**: 本文介绍了阿拉伯湾地区首个公开的自动驾驶数据集——Emirates Multi-Task (EMT) 数据集。EMT 数据集捕捉了阿拉伯湾地区的独特道路拓扑结构、高交通拥挤情况以及该地区的特色，包括行人的服饰差异和天气条件的变化。该数据集包含超过 30,000 帧驾驶视角的图像，以及 570,000 个标注的边界框，覆盖约 150 公里的驾驶路线。EMT 数据集支持三项主要任务：追踪、轨迹预测和意图预测。每个基准数据集都配备了相应的评估方法：(1) 多智能体追踪实验，关注多类别场景和遮挡处理；(2) 使用深度序列和交互感知模型的轨迹预测评估；以及 (3) 从观测轨迹预测智能体意图的意图基准实验。该数据集在以下链接公开获取：此 [https URL]，预处理脚本和评估模型则可以在以下链接访问：此 [https URL]。 

---
# Poster: Long PHP webshell files detection based on sliding window attention 

**Title (ZH)**: 海报：基于滑动窗口注意力的长期PHP Web_shell文件检测 

**Authors**: Zhiqiang Wang, Haoyu Wang, Lu Hao  

**Link**: [PDF](https://arxiv.org/pdf/2502.19257)  

**Abstract**: Webshell is a type of backdoor, and web applications are widely exposed to webshell injection attacks. Therefore, it is important to study webshell detection techniques. In this study, we propose a webshell detection method. We first convert PHP source code to opcodes and then extract Opcode Double-Tuples (ODTs). Next, we combine CodeBert and FastText models for feature representation and classification. To address the challenge that deep learning methods have difficulty detecting long webshell files, we introduce a sliding window attention mechanism. This approach effectively captures malicious behavior within long files. Experimental results show that our method reaches high accuracy in webshell detection, solving the problem of traditional methods that struggle to address new webshell variants and anti-detection techniques. 

**Abstract (ZH)**: Webshell 是一种后门，而 web 应用程序普遍易受到 Webshell 注入攻击。因此，研究 Webshell 检测技术至关重要。本研究提出了一种 Webshell 检测方法。首先，我们将 PHP 源代码转换为操作码，然后提取操作码双元组（ODTs）。接着，我们结合 CodeBert 和 FastText 模型进行特征表示和分类。为解决深度学习方法难以检测长 Webshell 文件的问题，我们引入了一种滑动窗口注意力机制。该方法能够有效捕捉长文件中的恶意行为。实验结果表明，本方法在 Webshell 检测中达到了高精度，解决了传统方法难以应对新型 Webshell 变种和反检测技术的问题。 

---
# Can RLHF be More Efficient with Imperfect Reward Models? A Policy Coverage Perspective 

**Title (ZH)**: imperfect奖励模型下，RLHF能否更加高效？一种策略覆盖视角 

**Authors**: Jiawei Huang, Bingcong Li, Christoph Dann, Niao He  

**Link**: [PDF](https://arxiv.org/pdf/2502.19255)  

**Abstract**: Sample efficiency is critical for online Reinforcement Learning from Human Feedback (RLHF). While existing works investigate sample-efficient online exploration strategies, the potential of utilizing misspecified yet relevant reward models to accelerate learning remains underexplored. This paper studies how to transfer knowledge from those imperfect reward models in online RLHF. We start by identifying a novel property of the KL-regularized RLHF objective: \emph{a policy's ability to cover the optimal policy is captured by its sub-optimality}. Building on this insight, we propose a theoretical transfer learning algorithm with provable benefits compared to standard online learning. Our approach achieves low regret in the early stage by quickly adapting to the best available source reward models without prior knowledge of their quality, and over time, it attains an $\tilde{O}(\sqrt{T})$ regret bound \emph{independent} of structural complexity measures. Inspired by our theoretical findings, we develop an empirical algorithm with improved computational efficiency, and demonstrate its effectiveness empirically in summarization tasks. 

**Abstract (ZH)**: 在线人类反馈强化学习（RLHF）中的样本效率对于训练至关重要。尽管现有工作探讨了样本高效的在线探索策略，但利用不完全正确但相关性高的奖励模型来加速学习的潜力尚未得到充分挖掘。本文研究了如何在在线RLHF中利用这些不完美的奖励模型的知识。我们首先识别出KL正则化RLHF目标的一个新颖性质：**策略覆盖最优策略的能力与其次优性的关系被捕捉**。基于这一洞察，我们提出了一种具有可证明优势的理论迁移学习算法，相较于标准的在线学习，该算法在早期阶段能够通过快速适应现有的最佳源奖励模型从而实现低后悔值，同时随着时间的推移，它能够获得与结构复杂度度量无关的$\tilde{O}(\sqrt{T})$后悔界。受到我们理论发现的启发，我们开发了一个具有改进计算效率的实证算法，并通过摘要任务的实证结果验证了其有效性。 

---
# GraphBridge: Towards Arbitrary Transfer Learning in GNNs 

**Title (ZH)**: GraphBridge：面向GNN中的任意迁移学习 

**Authors**: Li Ju, Xingyi Yang, Qi Li, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19252)  

**Abstract**: Graph neural networks (GNNs) are conventionally trained on a per-domain, per-task basis. It creates a significant barrier in transferring the acquired knowledge to different, heterogeneous data setups. This paper introduces GraphBridge, a novel framework to enable knowledge transfer across disparate tasks and domains in GNNs, circumventing the need for modifications to task configurations or graph structures. Specifically, GraphBridge allows for the augmentation of any pre-trained GNN with prediction heads and a bridging network that connects the input to the output layer. This architecture not only preserves the intrinsic knowledge of the original model but also supports outputs of arbitrary dimensions. To mitigate the negative transfer problem, GraphBridg merges the source model with a concurrently trained model, thereby reducing the source bias when applied to the target domain. Our method is thoroughly evaluated across diverse transfer learning scenarios, including Graph2Graph, Node2Node, Graph2Node, and graph2point-cloud. Empirical validation, conducted over 16 datasets representative of these scenarios, confirms the framework's capacity for task- and domain-agnostic transfer learning within graph-like data, marking a significant advancement in the field of GNNs. 

**Abstract (ZH)**: 图神经网络（GNNs）传统的训练方法是针对特定领域和特定任务进行的。这在将获取的知识转移到不同且异构的数据设置中时造成了显著的障碍。本文提出了一种名为GraphBridge的新框架，能够在GNNs中跨异构任务和领域实现知识迁移，无需对任务配置或图结构进行修改。具体而言，GraphBridge 允许将任何预训练的GNN与预测头部和连接输入层和输出层的桥梁网络相结合。该架构不仅保留了原始模型的内在知识，还支持任意维度的输出。为了解决负迁移问题，GraphBridge 通过与目标领域中同时训练的模型合并源模型，从而减轻在目标域应用时的来源偏见。我们的方法在图到图、节点到节点、图到节点和图到点云的多种迁移学习场景中进行了全面评估。在16个代表这些场景的数据库上的实验证据验证了该框架在图型数据中实现任务和领域无关的迁移学习的能力，标志着GNN领域的重要进展。 

---
# Between Circuits and Chomsky: Pre-pretraining on Formal Languages Imparts Linguistic Biases 

**Title (ZH)**: 介于电路与乔姆斯基之间：在形式语言上的预预训练植入了语言偏见 

**Authors**: Michael Y. Hu, Jackson Petty, Chuan Shi, William Merrill, Tal Linzen  

**Link**: [PDF](https://arxiv.org/pdf/2502.19249)  

**Abstract**: Pretraining language models on formal languages can improve their acquisition of natural language, but it is unclear which features of the formal language impart an inductive bias that leads to effective transfer. Drawing on insights from linguistics and complexity theory, we hypothesize that effective transfer occurs when the formal language both captures dependency structures in natural language and remains within the computational limitations of the model architecture. Focusing on transformers, we find that formal languages with both these properties enable language models to achieve lower loss on natural language and better linguistic generalization compared to other languages. In fact, pre-pretraining, or training on formal-then-natural language, reduces loss more efficiently than the same amount of natural language. For a 1B-parameter language model trained on roughly 1.6B tokens of natural language, pre-pretraining achieves the same loss and better linguistic generalization with a 33% smaller token budget. We also give mechanistic evidence of cross-task transfer from formal to natural language: attention heads acquired during formal language pretraining remain crucial for the model's performance on syntactic evaluations. 

**Abstract (ZH)**: 将形式语言用于预训练语言模型可以提高其对自然语言的掌握能力，但目前尚不清楚形式语言的哪些特征能够赋予模型有效的诱导偏置，从而实现有效的迁移学习。基于语言学和复杂性理论的洞察，我们假设有效的迁移学习发生在形式语言既能捕捉自然语言中的依赖结构，又不超出模型架构的计算限制之时。我们将关注于变压器模型，发现符合这两个特征的形式语言能够让语言模型在自然语言上的损失更低，并且在语言通用性方面表现更佳。实际上，使用形式语言预训练再进行自然语言训练（即预预训练），比单纯进行自然语言训练在相同数据量下能更高效地降低损失。例如，对于一个参数量约为10亿的模型，如果它在约16亿个自然语言标记上进行了自然语言训练，那么通过先进行形式语言预训练再进行自然语言训练，可以在减少33%标记预算的情况下达到相同的损失值并实现更好的语言通用性。此外，我们还提供了形式语言和自然语言之间跨任务迁移的机制性证据：在形式语言预训练过程中获得的注意力头对于模型在句法评估中的表现仍然是至关重要的。 

---
# AI-Powered Bayesian Inference 

**Title (ZH)**: AI赋能的贝叶斯推理 

**Authors**: Veronika Ročková, Sean O'Hagan  

**Link**: [PDF](https://arxiv.org/pdf/2502.19231)  

**Abstract**: The advent of Generative Artificial Intelligence (GAI) has heralded an inflection point that changed how society thinks about knowledge acquisition. While GAI cannot be fully trusted for decision-making, it may still provide valuable information that can be integrated into a decision pipeline. Rather than seeing the lack of certitude and inherent randomness of GAI as a problem, we view it as an opportunity. Indeed, variable answers to given prompts can be leveraged to construct a prior distribution which reflects assuredness of AI predictions. This prior distribution may be combined with tailored datasets for a fully Bayesian analysis with an AI-driven prior. In this paper, we explore such a possibility within a non-parametric Bayesian framework. The basic idea consists of assigning a Dirichlet process prior distribution on the data-generating distribution with AI generative model as its baseline. Hyper-parameters of the prior can be tuned out-of-sample to assess the informativeness of the AI prior. Posterior simulation is achieved by computing a suitably randomized functional on an augmented data that consists of observed (labeled) data as well as fake data whose labels have been imputed using AI. This strategy can be parallelized and rapidly produces iid samples from the posterior by optimization as opposed to sampling from conditionals. Our method enables (predictive) inference and uncertainty quantification leveraging AI predictions in a coherent probabilistic manner. 

**Abstract (ZH)**: 生成式人工智能（GAI）的出现标志着一个转折点，改变了社会对知识获取方式的认知。尽管GAI不能完全依赖于决策制定，但它仍然可以提供有价值的信息，这些信息可以集成到决策流程中。我们不应将GAI预测中的不确定性与内在随机性视为问题，而应将其视为一种机会。实际上，针对给定提示得到的多种答案可以被利用来构建先验分布，反映AI预测的信心水平。这种先验分布可以与定制的数据集结合，进行以AI驱动先验的完整贝叶斯分析。在本文中，我们探讨了在非参数贝叶斯框架下实现这一可能性的可行性。基本思想是，在数据生成分布上分配狄利克雷过程先验，并以AI生成模型为基础。先验的超参数可以通过未观察到的数据进行调整，以评估AI先验的信息量。后验模拟通过在扩展数据上计算一个适当随机化的函数来实现，扩展数据包括观察到（标记）数据以及使用AI推断出标签的虚假数据。该策略可以并行化，通过优化而非从条件中采样，快速生成后验的独立同分布样本。本文的方法使我们能够以一致的概率方式利用AI预测进行（预测）推断和不确定性量化。 

---
# Enhancing the Scalability and Applicability of Kohn-Sham Hamiltonians for Molecular Systems 

**Title (ZH)**: 增强Kohn-Sham哈密顿量在分子系统中的可扩展性和适用性 

**Authors**: Yunyang Li, Zaishuo Xia, Lin Huang, Xinran Wei, Han Yang, Sam Harshe, Zun Wang, Chang Liu, Jia Zhang, Bin Shao, Mark B. Gerstein  

**Link**: [PDF](https://arxiv.org/pdf/2502.19227)  

**Abstract**: Density Functional Theory (DFT) is a pivotal method within quantum chemistry and materials science, with its core involving the construction and solution of the Kohn-Sham Hamiltonian. Despite its importance, the application of DFT is frequently limited by the substantial computational resources required to construct the Kohn-Sham Hamiltonian. In response to these limitations, current research has employed deep-learning models to efficiently predict molecular and solid Hamiltonians, with roto-translational symmetries encoded in their neural networks. However, the scalability of prior models may be problematic when applied to large molecules, resulting in non-physical predictions of ground-state properties. In this study, we generate a substantially larger training set (PubChemQH) than used previously and use it to create a scalable model for DFT calculations with physical accuracy. For our model, we introduce a loss function derived from physical principles, which we call Wavefunction Alignment Loss (WALoss). WALoss involves performing a basis change on the predicted Hamiltonian to align it with the observed one; thus, the resulting differences can serve as a surrogate for orbital energy differences, allowing models to make better predictions for molecular orbitals and total energies than previously possible. WALoss also substantially accelerates self-consistent-field (SCF) DFT calculations. Here, we show it achieves a reduction in total energy prediction error by a factor of 1347 and an SCF calculation speed-up by a factor of 18%. These substantial improvements set new benchmarks for achieving accurate and applicable predictions in larger molecular systems. 

**Abstract (ZH)**: 密度泛函理论（DFT）是量子化学和材料科学中的关键方法，其核心在于构建和求解科恩-沙姆（Kohn-Sham）哈密顿量。尽管DFT非常重要，但在构造科恩-沙姆哈密顿量时所需的巨大计算资源常常限制了其应用。为应对这些限制，当前的研究利用深度学习模型来高效预测分子和固体的哈密顿量，并在其神经网络中编码了转旋对称性。然而，之前模型的可扩展性在应用于大分子时可能会出现问题，导致预测基态性质不物理。在此研究中，我们生成了一个比以往更庞大的训练集（PubChemQH），并使用它来创建一个具有物理准确性且可扩展的DFT计算模型。对于我们的模型，我们引入了一个从物理原理出发的损失函数，称之为波函数对齐损失（WALoss）。WALoss涉及对预测的哈密顿量进行基底变换，使其与观测的哈密顿量对齐；因此，结果中的差异可以作为轨道能量差的替代物，从而使模型能够比以往更好地预测分子轨道和总能量。WALoss还极大地加速了自洽场（SCF）DFT计算。在此，我们展示了它可以将总能量预测误差减少1347倍，并将SCF计算速度提升18%。这些显著的改进为在更大分子体系中实现准确且适用的预测设定了新的基准。 

---
# A Lightweight and Extensible Cell Segmentation and Classification Model for Whole Slide Images 

**Title (ZH)**: 一种适用于全视野图像的轻量级可扩展细胞分割与分类模型 

**Authors**: Nikita Shvetsov, Thomas K. Kilvaer, Masoud Tafavvoghi, Anders Sildnes, Kajsa Møllersen, Lill-Tove Rasmussen Busund, Lars Ailo Bongo  

**Link**: [PDF](https://arxiv.org/pdf/2502.19217)  

**Abstract**: Developing clinically useful cell-level analysis tools in digital pathology remains challenging due to limitations in dataset granularity, inconsistent annotations, high computational demands, and difficulties integrating new technologies into workflows. To address these issues, we propose a solution that enhances data quality, model performance, and usability by creating a lightweight, extensible cell segmentation and classification model. First, we update data labels through cross-relabeling to refine annotations of PanNuke and MoNuSAC, producing a unified dataset with seven distinct cell types. Second, we leverage the H-Optimus foundation model as a fixed encoder to improve feature representation for simultaneous segmentation and classification tasks. Third, to address foundation models' computational demands, we distill knowledge to reduce model size and complexity while maintaining comparable performance. Finally, we integrate the distilled model into QuPath, a widely used open-source digital pathology platform. Results demonstrate improved segmentation and classification performance using the H-Optimus-based model compared to a CNN-based model. Specifically, average $R^2$ improved from 0.575 to 0.871, and average $PQ$ score improved from 0.450 to 0.492, indicating better alignment with actual cell counts and enhanced segmentation quality. The distilled model maintains comparable performance while reducing parameter count by a factor of 48. By reducing computational complexity and integrating into workflows, this approach may significantly impact diagnostics, reduce pathologist workload, and improve outcomes. Although the method shows promise, extensive validation is necessary prior to clinical deployment. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

在数字病理学中开发具有临床实用性的细胞级分析工具依然具有挑战性，这主要是由于数据集粒度限制、不一致的标注、高计算需求以及将新技术集成到工作流程中的困难。为解决这些问题，我们提出了一种通过提高数据质量、模型性能和使用便捷性来增强细胞分割和分类模型的方法。首先，我们通过交叉重新标注更新数据标签，从而精炼 PanNuke 和 MoNuSAC 的标注，生成包含七种不同细胞类型的统一数据集。其次，我们利用 H-Optimus 基础模型作为固定编码器，以提高同时进行分割和分类任务的特征表示。第三，为了解决基础模型的高计算需求，我们通过知识蒸馏来减小模型规模和复杂性，同时保持相当的性能。最后，我们将蒸馏后的模型整合到 QuPath 这一广泛应用的开源数字病理学平台中。结果显示，基于 H-Optimus 的模型在分割和分类性能上优于基于 CNN 的模型。具体来说，平均 \(R^2\) 从 0.575 提高到 0.871，平均 \(PQ\) 分数从 0.450 提高到 0.492，表明更好的细胞计数对齐和更高的分割质量。蒸馏后的模型保持了相当的性能，但参数量减少了 48 倍。通过降低计算复杂性和整合到工作流程中，该方法可能显著影响诊断效果，减轻病理学家的工作负担，并提高诊疗结果。尽管该方法显示出良好的前景，但其在临床部署前仍需进行广泛的验证。 

---
# FaithUn: Toward Faithful Forgetting in Language Models by Investigating the Interconnectedness of Knowledge 

**Title (ZH)**: FaithUn：通过探究知识的关联性以实现语言模型的忠实遗忘 

**Authors**: Nakyeong Yang, Minsung Kim, Seunghyun Yoon, Joongbo Shin, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2502.19207)  

**Abstract**: Various studies have attempted to remove sensitive or private knowledge from a language model to prevent its unauthorized exposure. However, prior studies have overlooked the complex and interconnected nature of knowledge, where related knowledge must be carefully examined. Specifically, they have failed to evaluate whether an unlearning method faithfully erases interconnected knowledge that should be removed, retaining knowledge that appears relevant but exists in a completely different context. To resolve this problem, we first define a new concept called superficial unlearning, which refers to the phenomenon where an unlearning method either fails to erase the interconnected knowledge it should remove or unintentionally erases irrelevant knowledge. Based on the definition, we introduce a new benchmark, FaithUn, to analyze and evaluate the faithfulness of unlearning in real-world knowledge QA settings. Furthermore, we propose a novel unlearning method, KLUE, which updates only knowledge-related neurons to achieve faithful unlearning. KLUE identifies knowledge neurons using an explainability method and updates only those neurons using selected unforgotten samples. Experimental results demonstrate that widely-used unlearning methods fail to ensure faithful unlearning, while our method shows significant effectiveness in real-world QA unlearning. 

**Abstract (ZH)**: 各种研究尝试从语言模型中删除敏感或私人信息，以防止其未经授权的暴露。然而，先前的研究忽视了知识的复杂性和相互关联性，未能仔细检查相关知识。具体来说，这些研究未能评估去学习方法是否忠实地删除应去除的知识，而是保留了看似相关但实际上存在于完全不同的上下文中的知识。为解决这一问题，我们首先定义了一个新的概念——表面去学习，指的是去学习方法未能有效删除应去除的相关知识，或者无意中删除了不相关知识的现象。基于这一定义，我们引入了一个新的基准，FaithUn，用于分析和评估去学习方法在实际知识问答（Knowledge QA）场景中的忠实地去除效果。此外，我们提出了一种新的去学习方法——KLUE，该方法仅更新与知识相关的神经元以实现忠实地去除。KLUE 使用可解释的方法来识别知识神经元，并仅使用选定的记忆样本来更新这些神经元。实验结果表明，广泛使用的去学习方法无法确保忠实地去除知识，而我们的方法在实际知识问答去学习中显示出显著的效果。 

---
# EGR-Net: A Novel Embedding Gramian Representation CNN for Intelligent Fault Diagnosis 

**Title (ZH)**: EGR-Net：一种新型嵌入Gramian表示CNN用于智能故障诊断 

**Authors**: Linshan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2502.19199)  

**Abstract**: Feature extraction is crucial in intelligent fault diagnosis of rotating machinery. It is easier for convolutional neural networks(CNNs) to visually recognize and learn fault features by converting the complicated one-dimensional (1D) vibrational signals into two-dimensional (2D) images with simple textures. However, the existing representation methods for encoding 1D signals as images have two main problems, including complicated computation and low separability. Meanwhile, the existing 2D-CNN fault diagnosis methods taking 2D images as the only inputs still suffer from the inevitable information loss because of the conversion process. Considering the above issues, this paper proposes a new 1D-to-2D conversion method called Embedding Gramian Representation (EGR), which is easy to calculate and shows good separability. In EGR, 1D signals are projected in the embedding space and the intrinsic periodicity of vibrational signals is captured enabling the faulty characteristics contained in raw signals to be uncovered. Second, aiming at the information loss problem of existing CNN models with the single input of converted images, a double-branch EGR-based CNN, called EGR-Net, is proposed to learn faulty features from both raw signal feature maps and their corresponding EGRs. The bridge connection is designed to improve the feature learning interaction between the two branches. Widely used open domain gearbox dataset and bearing dataset are used to verify the effectiveness and efficiency of the proposed methods. EGR-Net is compared with traditional and state-of-the-art approaches, and the results show that the proposed method can deliver enhanced performance. 

**Abstract (ZH)**: 智能旋转机械故障诊断中的特征提取至关重要。卷积神经网络（CNNs）通过将复杂的1D振动信号转换为具有简单纹理的2D图像，更容易实现视觉识别和学习故障特征。然而，目前将1D信号编码为图像的表示方法存在两大主要问题，包括计算复杂度高和分离性低。同时，现有的基于2D图像输入的2D-CNN故障诊断方法在转换过程中不可避免地会丢失一些信息。鉴于上述问题，本文提出了一种新的1D到2D变换方法，称为嵌入 gramian表示（EGR），该方法易于计算且具有良好的分离性。在EGR中，1D信号在嵌入空间中进行投影，并捕捉振动信号的固有周期性，从而揭示原始信号中包含的故障特征。其次，针对现有单输入图像转换的CNN模型中存在的信息丢失问题，本文提出了一种基于双支EGR的CNN，称为EGR-Net，该模型能够从原始信号特征图及其对应的EGR中学习故障特征。设计了桥梁连接以提高两支之间的特征学习交互。广泛使用的开放域齿轮箱数据集和轴承数据集被用来验证所提出方法的有效性和效率。EGR-Net与传统方法和最先进的方法进行了比较，结果显示所提出的方法具有增强的性能。 

---
# Simulation of Language Evolution under Regulated Social Media Platforms: A Synergistic Approach of Large Language Models and Genetic Algorithms 

**Title (ZH)**: 在监管社交媒体平台下语言演化模拟：大型语言模型与遗传算法的协同方法 

**Authors**: Jinyu Cai, Yusei Ishimizu, Mingyue Zhang, Munan Li, Jialong Li, Kenji Tei  

**Link**: [PDF](https://arxiv.org/pdf/2502.19193)  

**Abstract**: Social media platforms frequently impose restrictive policies to moderate user content, prompting the emergence of creative evasion language strategies. This paper presents a multi-agent framework based on Large Language Models (LLMs) to simulate the iterative evolution of language strategies under regulatory constraints. In this framework, participant agents, as social media users, continuously evolve their language expression, while supervisory agents emulate platform-level regulation by assessing policy violations. To achieve a more faithful simulation, we employ a dual design of language strategies (constraint and expression) to differentiate conflicting goals and utilize an LLM-driven GA (Genetic Algorithm) for the selection, mutation, and crossover of language strategies. The framework is evaluated using two distinct scenarios: an abstract password game and a realistic simulated illegal pet trade scenario. Experimental results demonstrate that as the number of dialogue rounds increases, both the number of uninterrupted dialogue turns and the accuracy of information transmission improve significantly. Furthermore, a user study with 40 participants validates the real-world relevance of the generated dialogues and strategies. Moreover, ablation studies validate the importance of the GA, emphasizing its contribution to long-term adaptability and improved overall results. 

**Abstract (ZH)**: 社交媒体平台经常实施限制性的政策来规范用户内容，从而催生了创造性的规避语言策略。本文基于大型语言模型（LLMs）提出了一种多智能体框架，用于模拟在监管约束下的语言策略的迭代进化过程。在这个框架中，参与者智能体作为社交媒体用户，不断进化其语言表达方式，而监督智能体则通过评估政策违规行为来模拟平台级别的监管。为了实现更真实的模拟，我们采用了一种双语言策略设计（约束与表达），以区分相冲突的目标，并利用基于LLM的GA（遗传算法）来进行语言策略的选择、突变和杂交。该框架通过两种不同的场景进行了评估：一个抽象的密码游戏和一个现实中的非法宠物交易模拟场景。实验结果表明，随着对话回合的增加，未中断的对话回合数和信息传递的准确性显著提高。此外，40名参与者的用户研究验证了生成的对话和策略的现实相关性。此外，消融研究验证了GA的重要性，突显了其在长期适应性和提高整体结果方面的贡献。 

---
# Provocations from the Humanities for Generative AI Research 

**Title (ZH)**: 人文领域对生成式人工智能研究的挑战与启示 

**Authors**: Lauren Klein, Meredith Martin, André Brock, Maria Antoniak, Melanie Walsh, Jessica Marie Johnson, Lauren Tilton, David Mimno  

**Link**: [PDF](https://arxiv.org/pdf/2502.19190)  

**Abstract**: This paper presents a set of provocations for considering the uses, impact, and harms of generative AI from the perspective of humanities researchers. We provide a working definition of humanities research, summarize some of its most salient theories and methods, and apply these theories and methods to the current landscape of AI. Drawing from foundational work in critical data studies, along with relevant humanities scholarship, we elaborate eight claims with broad applicability to current conversations about generative AI: 1) Models make words, but people make meaning; 2) Generative AI requires an expanded definition of culture; 3) Generative AI can never be representative; 4) Bigger models are not always better models; 5) Not all training data is equivalent; 6) Openness is not an easy fix; 7) Limited access to compute enables corporate capture; and 8) AI universalism creates narrow human subjects. We conclude with a discussion of the importance of resisting the extraction of humanities research by computer science and related fields. 

**Abstract (ZH)**: 本文从人文学科研究人员的角度出发，提出了一系列关于生成性人工智能的用途、影响及其潜在危害的质疑。我们提供了一个人文研究的工作定义，总结了一些最显著的理论和方法，并将这些理论和方法应用于当前的人工智能领域。借鉴批判性数据研究领域的基础工作和相关的人文学科研究成果，我们提出了八个具有广泛适用性的主张，这些主张适用于当前关于生成性人工智能的讨论：1）模型生成文字，但人们赋予其意义；2）生成性人工智能需要扩展其文化定义；3）生成性人工智能永远不可能具代表性；4）更大的模型并不总是更好的模型；5）并非所有训练数据都等同；6）开放性并不能轻易解决；7）有限的计算资源可能导致企业垄断；8）人工智能普遍性造就了狭窄的人类主体。最后，我们讨论了抵制计算机科学及相关领域对人文学科研究的榨取的重要性。 

---
# AutoML for Multi-Class Anomaly Compensation of Sensor Drift 

**Title (ZH)**: 自动机器学习在传感器漂移多类异常补偿中的应用 

**Authors**: Melanie Schaller, Mathis Kruse, Antonio Ortega, Marius Lindauer, Bodo Rosenhahn  

**Link**: [PDF](https://arxiv.org/pdf/2502.19180)  

**Abstract**: Addressing sensor drift is essential in industrial measurement systems, where precise data output is necessary for maintaining accuracy and reliability in monitoring processes, as it progressively degrades the performance of machine learning models over time. Our findings indicate that the standard cross-validation method used in existing model training overestimates performance by inadequately accounting for drift. This is primarily because typical cross-validation techniques allow data instances to appear in both training and testing sets, thereby distorting the accuracy of the predictive evaluation. As a result, these models are unable to precisely predict future drift effects, compromising their ability to generalize and adapt to evolving data conditions. This paper presents two solutions: (1) a novel sensor drift compensation learning paradigm for validating models, and (2) automated machine learning (AutoML) techniques to enhance classification performance and compensate sensor drift. By employing strategies such as data balancing, meta-learning, automated ensemble learning, hyperparameter optimization, feature selection, and boosting, our AutoML-DC (Drift Compensation) model significantly improves classification performance against sensor drift. AutoML-DC further adapts effectively to varying drift severities. 

**Abstract (ZH)**: 在工业测量系统中，解决传感器漂移至关重要，因为精确的数据输出对于保持监测过程中的准确性和可靠性是必不可少的，而传感器漂移会随着时间逐渐削弱机器学习模型的性能。我们的研究发现，现有模型训练中常用的交叉验证方法会由于未能充分考虑漂移而导致性能高估。这是因为传统交叉验证技术允许数据实例同时出现在训练集和测试集中，从而扭曲预测评估的准确性。因此，这些模型无法精确预测未来的漂移效应，从而削弱了它们适应不断变化的数据条件的能力。本文提出了两种解决方案：（1）一种新型的传感器漂移补偿学习范式，用于验证模型，以及（2）增强机器学习（AutoML）技术，以提高分类性能并补偿传感器漂移。通过采用数据平衡、元学习、自动化集成学习、超参数优化、特征选择和提升等策略，我们的AutoML-DC（漂移补偿）模型显著提高了对传感器漂移的分类性能。AutoML-DC 进一步能够有效适应不同的漂移严重程度。 

---
# MEDDxAgent: A Unified Modular Agent Framework for Explainable Automatic Differential Diagnosis 

**Title (ZH)**: MEDDxAgent: 一个统一的模块化代理框架，用于可解释的自动差异诊断 

**Authors**: Daniel Rose, Chia-Chien Hung, Marco Lepri, Israa Alqassem, Kiril Gashteovski, Carolin Lawrence  

**Link**: [PDF](https://arxiv.org/pdf/2502.19175)  

**Abstract**: Differential Diagnosis (DDx) is a fundamental yet complex aspect of clinical decision-making, in which physicians iteratively refine a ranked list of possible diseases based on symptoms, antecedents, and medical knowledge. While recent advances in large language models have shown promise in supporting DDx, existing approaches face key limitations, including single-dataset evaluations, isolated optimization of components, unrealistic assumptions about complete patient profiles, and single-attempt diagnosis. We introduce a Modular Explainable DDx Agent (MEDDxAgent) framework designed for interactive DDx, where diagnostic reasoning evolves through iterative learning, rather than assuming a complete patient profile is accessible. MEDDxAgent integrates three modular components: (1) an orchestrator (DDxDriver), (2) a history taking simulator, and (3) two specialized agents for knowledge retrieval and diagnosis strategy. To ensure robust evaluation, we introduce a comprehensive DDx benchmark covering respiratory, skin, and rare diseases. We analyze single-turn diagnostic approaches and demonstrate the importance of iterative refinement when patient profiles are not available at the outset. Our broad evaluation demonstrates that MEDDxAgent achieves over 10% accuracy improvements in interactive DDx across both large and small LLMs, while offering critical explainability into its diagnostic reasoning process. 

**Abstract (ZH)**: 差异诊断（DDx）是临床决策过程中一个基础但复杂的方面，在这一过程中，医生根据症状、病史和医学知识，逐步细化可能疾病的排名列表。虽然近期大规模语言模型在支持差异诊断方面展现出了潜力，但现有方法仍面临一些关键限制，包括单一数据集评估、模型组件孤立优化、不切实际的完整患者资料假设，以及一次性诊断。我们提出了一种模块化可解释的差异诊断代理（MEDDxAgent）框架，专为互动式差异诊断而设计，诊断推理通过迭代学习演进，而非假设完整患者资料可获取。MEDDxAgent 集成了三个模块化的组件：（1）协调者（DDxDriver），（2）病史采集模拟器，以及（3）两个专门用于知识检索和诊断策略的代理。为了确保稳健的评估，我们引入了一个涵盖呼吸系统、皮肤疾病和罕见疾病的综合差异诊断基准。分析单轮诊断方法，并展示了在初始阶段患者资料不可用时迭代细化的重要性。广泛的评估表明，在大型和小型语言模型中，MEDDxAgent 在互动差异诊断中的准确性提高了超过10%，并对其诊断推理过程提供了关键的可解释性。 

---
# TestNUC: Enhancing Test-Time Computing Approaches through Neighboring Unlabeled Data Consistency 

**Title (ZH)**: TestNUC：通过邻近未标记数据一致性增强测试时计算方法 

**Authors**: Henry Peng Zou, Zhengyao Gu, Yue Zhou, Yankai Chen, Weizhi Zhang, Liancheng Fang, Yibo Wang, Yangning Li, Kay Liu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19163)  

**Abstract**: Test-time computing approaches, which leverage additional computational resources during inference, have been proven effective in enhancing large language model performance. This work introduces a novel, linearly scaling approach, TestNUC, that improves test-time predictions by leveraging the local consistency of neighboring unlabeled data-it classifies an input instance by considering not only the model's prediction on that instance but also on neighboring unlabeled instances. We evaluate TestNUC across eight diverse datasets, spanning intent classification, topic mining, domain discovery, and emotion detection, demonstrating its consistent superiority over baseline methods such as standard prompting and self-consistency. Furthermore, TestNUC can be seamlessly integrated with existing test-time computing approaches, substantially boosting their performance. Our analysis reveals that TestNUC scales effectively with increasing amounts of unlabeled data and performs robustly across different embedding models, making it practical for real-world applications. Our code is available at this https URL. 

**Abstract (ZH)**: 在推理时利用额外计算资源的方法已被证明能够有效提升大规模语言模型的性能。本文介绍了一种新颖的线性扩展方法——TestNUC，通过利用邻近未标注数据的局部一致性来改进测试时的预测。这种方法不仅考虑模型对该输入实例的预测，还考虑其对邻近未标注实例的预测来进行分类。我们在八个不同的数据集中对TestNUC进行了评估，涵盖意向分类、主题挖掘、领域发现和情绪检测等领域，结果显示TestNUC在各种基线方法（如标准提示和自一致性）上表现出一致的优越性。此外，TestNUC可以无缝集成到现有的测试时计算方法中，从而显著提升其性能。我们的分析表明，TestNUC能够有效扩展处理不断增加的未标注数据量，并在不同嵌入模型下表现出很高的鲁棒性，使其在实际应用中具有可行性。我们的代码已发布在以下链接：[提供链接处]。 

---
# Detecting Linguistic Indicators for Stereotype Assessment with Large Language Models 

**Title (ZH)**: 使用大规模语言模型检测 stereotypes 的语言指标 

**Authors**: Rebekka Görge, Michael Mock, Héctor Allende-Cid  

**Link**: [PDF](https://arxiv.org/pdf/2502.19160)  

**Abstract**: Social categories and stereotypes are embedded in language and can introduce data bias into Large Language Models (LLMs). Despite safeguards, these biases often persist in model behavior, potentially leading to representational harm in outputs. While sociolinguistic research provides valuable insights into the formation of stereotypes, NLP approaches for stereotype detection rarely draw on this foundation and often lack objectivity, precision, and interpretability. To fill this gap, in this work we propose a new approach that detects and quantifies the linguistic indicators of stereotypes in a sentence. We derive linguistic indicators from the Social Category and Stereotype Communication (SCSC) framework which indicate strong social category formulation and stereotyping in language, and use them to build a categorization scheme. To automate this approach, we instruct different LLMs using in-context learning to apply the approach to a sentence, where the LLM examines the linguistic properties and provides a basis for a fine-grained assessment. Based on an empirical evaluation of the importance of different linguistic indicators, we learn a scoring function that measures the linguistic indicators of a stereotype. Our annotations of stereotyped sentences show that these indicators are present in these sentences and explain the strength of a stereotype. In terms of model performance, our results show that the models generally perform well in detecting and classifying linguistic indicators of category labels used to denote a category, but sometimes struggle to correctly evaluate the associated behaviors and characteristics. Using more few-shot examples within the prompts, significantly improves performance. Model performance increases with size, as Llama-3.3-70B-Instruct and GPT-4 achieve comparable results that surpass those of Mixtral-8x7B-Instruct, GPT-4-mini and Llama-3.1-8B-Instruct. 

**Abstract (ZH)**: 社会类别和刻板印象嵌入在语言中，并可能在大型语言模型（LLMs）的数据中引入偏差。尽管采取了防护措施，这些偏差常常在模型的行为中持续存在，潜在地导致输出中的代表性伤害。尽管社会语言学研究为刻板印象的形成提供了宝贵的知识，但用于刻板印象检测的NLP方法很少以此为基础，并且往往缺乏客观性、精确性和可解释性。为弥补这一缺口，本文提出了一种新方法，用于检测和量化句子中的刻板印象语言指标。我们从社会类别和刻板印象交流（SCSC）框架中提取语言指标，这些指标显示了语言中强烈的社会类别构建和刻板印象，利用这些指标构建分类方案。为了自动化这一方法，使用基于上下文学习的方法对不同的LLM进行指令，使其将该方法应用于一个句子，在此过程中，LLM检查语言属性并提供细粒度评估的基础。通过对不同语言指标重要性的实证评估，我们学习了一个评分函数，用于衡量一种刻板印象的语言指标。我们的对带有刻板印象句子的标注表明，这些指标在这类句子中存在，并解释了刻板印象的强度。从模型性能来看，我们的结果显示，模型在检测和分类用以表示某一类别的类别标签的语言指标方面表现普遍良好，但在正确评估相关行为和特征方面存在问题。在提示中使用更少的示范示例显著提高了性能。模型性能随规模增加而提高，Llama-3.3-70B-Instruct和GPT-4的性能达到了可比的结果，超过了Mixtral-8x7B-Instruct、GPT-4-mini和Llama-3.1-8B-Instruct的表现。 

---
# When Personalization Meets Reality: A Multi-Faceted Analysis of Personalized Preference Learning 

**Title (ZH)**: 当个性化遭遇现实：个性化偏好学习的多维度分析 

**Authors**: Yijiang River Dong, Tiancheng Hu, Yinhong Liu, Ahmet Üstün, Nigel Collier  

**Link**: [PDF](https://arxiv.org/pdf/2502.19158)  

**Abstract**: While Reinforcement Learning from Human Feedback (RLHF) is widely used to align Large Language Models (LLMs) with human preferences, it typically assumes homogeneous preferences across users, overlooking diverse human values and minority viewpoints. Although personalized preference learning addresses this by tailoring separate preferences for individual users, the field lacks standardized methods to assess its effectiveness. We present a multi-faceted evaluation framework that measures not only performance but also fairness, unintended effects, and adaptability across varying levels of preference divergence. Through extensive experiments comparing eight personalization methods across three preference datasets, we demonstrate that performance differences between methods could reach 36% when users strongly disagree, and personalization can introduce up to 20% safety misalignment. These findings highlight the critical need for holistic evaluation approaches to advance the development of more effective and inclusive preference learning systems. 

**Abstract (ZH)**: 尽管人类反馈强化学习（RLHF）广泛应用于使大型语言模型（LLMs）与人类偏好相一致，它通常假设用户之间具有 homogenous 的偏好，忽视了多样的人类价值观和 minority 观点。虽然个性化的偏好学习通过为每个用户量身定制不同的偏好来解决这一问题，但该领域缺乏标准化的方法来评估其有效性。我们提出了一种多维度评估框架，不仅衡量性能，还衡量公平性、意外影响以及在不同偏好分歧程度下的适应性。通过对来自三个偏好数据集的八种个性化方法进行广泛的实验比较，我们证明了当用户之间分歧较大时，不同方法之间的性能差异可能达到 36%，个性化方法可能会引入多达 20% 的安全不对齐风险。这些发现强调了全面评估方法对于促进更有效和包容性偏好学习系统发展的迫切需求。 

---
# Voting or Consensus? Decision-Making in Multi-Agent Debate 

**Title (ZH)**: 投票还是共识？多智能体辩论中的决策机制 

**Authors**: Lars Benedikt Kaesberg, Jonas Becker, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2502.19130)  

**Abstract**: Much of the success of multi-agent debates depends on carefully choosing the right parameters. Among them, the decision-making protocol stands out. Systematic comparison of decision protocols is difficult because studies alter multiple discussion parameters beyond the protocol. So far, it has been largely unknown how decision-making addresses the challenges of different tasks. This work systematically evaluates the impact of seven decision protocols (e.g., majority voting, unanimity consensus). We change only one variable at a time (i.e., decision protocol) to analyze how different methods affect the collaboration between agents and test different protocols on knowledge (MMLU, MMLU-Pro, GPQA) and reasoning datasets (StrategyQA, MuSR, SQuAD 2.0). Our results show that voting protocols improve performance by 13.2% in reasoning tasks and consensus protocols by 2.8% in knowledge tasks over the other decision protocol. Increasing the number of agents improves performance, while more discussion rounds before voting reduces it. To improve decision-making by increasing answer diversity, we propose two new methods, All-Agents Drafting (AAD) and Collective Improvement (CI). Our methods improve task performance by up to 3.3% with AAD and up to 7.4% with CI. This work demonstrates the importance of decision-making in multi-agent debates beyond scaling. 

**Abstract (ZH)**: 多智能体辩论的成功很大程度上取决于正确选择参数。其中，决策协议尤为关键。由于研究中通常会同时改变多个讨论参数而不仅仅是协议本身，因此系统性的比较这些协议比较困难。到目前为止，决策方法如何应对不同任务的挑战尚不明确。本研究系统地评估了七种决策协议（如多数投票、一致同意）的影响。我们每次只改变一个变量（即决策协议），以分析不同的方法如何影响智能体之间的协作，并在知识（MMLU、MMLU-Pro、GPQA）和推理数据集（StrategyQA、MuSR、SQuAD 2.0）上测试不同的协议。结果显示，投票协议在推理任务中的性能提高了13.2%，而一致协议在知识任务中的性能提高了2.8%。增加智能体的数量可以提高性能，而在投票前进行更多的讨论回合则会降低性能。为了通过增加答案多样性来改进决策，我们提出了两种新方法：全员草案（All-Agents Drafting, AAD）和集体改进（Collective Improvement, CI）。我们的方法分别通过AAD将任务性能提高了3.3%，通过CI提高了7.4%。本研究证明，多智能体辩论中的决策方法的重要性远远超出了简单的扩展。 

---
# From Traditional to Deep Learning Approaches in Whole Slide Image Registration: A Methodological Review 

**Title (ZH)**: 从传统方法到深度学习在全视野组织图像配准中的应用：方法学综述 

**Authors**: Behnaz Elhaminia, Abdullah Alsalemi, Esha Nasir, Mostafa Jahanifar, Ruqayya Awan, Lawrence S. Young, Nasir M. Rajpoot, Fayyaz Minhas, Shan E Ahmed Raza  

**Link**: [PDF](https://arxiv.org/pdf/2502.19123)  

**Abstract**: Whole slide image (WSI) registration is an essential task for analysing the tumour microenvironment (TME) in histopathology. It involves the alignment of spatial information between WSIs of the same section or serial sections of a tissue sample. The tissue sections are usually stained with single or multiple biomarkers before imaging, and the goal is to identify neighbouring nuclei along the Z-axis for creating a 3D image or identifying subclasses of cells in the TME. This task is considerably more challenging compared to radiology image registration, such as magnetic resonance imaging or computed tomography, due to various factors. These include gigapixel size of images, variations in appearance between differently stained tissues, changes in structure and morphology between non-consecutive sections, and the presence of artefacts, tears, and deformations. Currently, there is a noticeable gap in the literature regarding a review of the current approaches and their limitations, as well as the challenges and opportunities they present. We aim to provide a comprehensive understanding of the available approaches and their application for various purposes. Furthermore, we investigate current deep learning methods used for WSI registration, emphasising their diverse methodologies. We examine the available datasets and explore tools and software employed in the field. Finally, we identify open challenges and potential future trends in this area of research. 

**Abstract (ZH)**: 全视野图像（WSI）注册是病理学分析肿瘤微环境（TME）不可或缺的一项任务。它涉及同一切片或组织样本连续切片的WSI之间的空间信息对齐。通常在成像之前，组织切片会使用单个或多个生物标记物进行染色，目标是在Z轴上识别邻近的细胞核，以生成3D图像或识别TME中的细胞亚类。与放射学图像对齐（如磁共振成像或计算机断层扫描）相比，这一任务更加具有挑战性，原因包括大分辨率图像、不同染色组织间的外观差异、非连续切片之间结构和形态的变化，以及存在伪影、撕裂和变形等现象。目前，文献中关于现有方法及其局限性的综述较为缺乏，同时也缺乏对其所面临的挑战和提供的机遇的探讨。我们旨在提供对现有方法及其应用的全面理解。此外，我们将探讨当前用于WSI注册的深度学习方法，强调其多样化的技术方法。我们还将检查可用的数据集，并探索该领域使用的工具和软件。最后，我们将识别该研究领域的开放挑战和可能的未来趋势。 

---
# Chemical knowledge-informed framework for privacy-aware retrosynthesis learning 

**Title (ZH)**: 面向隐私意识的逆合成学习的化学知识驱动框架 

**Authors**: Guikun Chen, Xu Zhang, Yi Yang, Wenguan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19119)  

**Abstract**: Chemical reaction data is a pivotal asset, driving advances in competitive fields such as pharmaceuticals, materials science, and industrial chemistry. Its proprietary nature renders it sensitive, as it often includes confidential insights and competitive advantages organizations strive to protect. However, in contrast to this need for confidentiality, the current standard training paradigm for machine learning-based retrosynthesis gathers reaction data from multiple sources into one single edge to train prediction models. This paradigm poses considerable privacy risks as it necessitates broad data availability across organizational boundaries and frequent data transmission between entities, potentially exposing proprietary information to unauthorized access or interception during storage and transfer. In the present study, we introduce the chemical knowledge-informed framework (CKIF), a privacy-preserving approach for learning retrosynthesis models. CKIF enables distributed training across multiple chemical organizations without compromising the confidentiality of proprietary reaction data. Instead of gathering raw reaction data, CKIF learns retrosynthesis models through iterative, chemical knowledge-informed aggregation of model parameters. In particular, the chemical properties of predicted reactants are leveraged to quantitatively assess the observable behaviors of individual models, which in turn determines the adaptive weights used for model aggregation. On a variety of reaction datasets, CKIF outperforms several strong baselines by a clear margin (e.g., ~20% performance improvement over FedAvg on USPTO-50K), showing its feasibility and superiority to stimulate further research on privacy-preserving retrosynthesis. 

**Abstract (ZH)**: 化学反应数据是关键资产，推动制药、材料科学和工业化学等竞争领域的发展。由于其专有性，这种数据往往包含公司试图保护的机密见解和竞争优势。然而，在这一保密需求与基于机器学习的逆合成分析标准训练范式之间存在矛盾，后者需要将来自多个来源的反应数据整合到单一边中进行预测模型的训练。这种范式带来很大的隐私风险，因为它需要跨组织边界广泛的数据可用性，并且频繁的数据传输可能在存储和传输过程中使专有信息面临未经授权的访问或拦截。在本研究中，我们提出了一种化学知识指导的框架（CKIF），这是一种保护隐私的学习逆合成分析模型的方法。CKIF能够在不泄露专有反应数据的情况下，允许多个化学组织之间的分布式训练。CKIF 不是收集原始反应数据，而是通过迭代地、基于化学知识的模型参数聚合来学习逆合成分析模型。特别地，预测底物的化学性质被用来定量评估单个模型的可观察行为，进而决定用于模型聚合的自适应权重。在多种反应数据集上，CKIF 在多个基准模型上表现出显著优势（例如，在USPTO-50K数据集上对FedAvg的性能提升达到约20%），这表明其可行性和优越性，有望促进进一步研究以推进保护隐私的逆合成分析。 

---
# Improving customer service with automatic topic detection in user emails 

**Title (ZH)**: 通过自动主题检测提升客户服务 

**Authors**: Bojana Bašaragin, Darija Medvecki, Gorana Gojić, Milena Oparnica, Dragiša Mišković  

**Link**: [PDF](https://arxiv.org/pdf/2502.19115)  

**Abstract**: This study introduces a novel Natural Language Processing pipeline that enhances customer service efficiency at Telekom Srbija, a leading Serbian telecommunications company, through automated email topic detection and labelling. Central to the pipeline is BERTopic, a modular architecture that allows unsupervised topic modelling. After a series of preprocessing and post-processing steps, we assign one of 12 topics and several additional labels to incoming emails, allowing customer service to filter and access them through a custom-made application. The model's performance was evaluated by assessing the speed and correctness of the automatically assigned topics across a test dataset of 100 customer emails. The pipeline shows broad applicability across languages, particularly for those that are low-resourced and morphologically rich. The system now operates in the company's production environment, streamlining customer service operations through automated email classification. 

**Abstract (ZH)**: 本研究介绍了一种新颖的自然语言处理流水线，通过自动检测和标记电子邮件主题，显著提升了塞尔维亚领先的电信公司Telekom Srbija的客户服务效率。该流水线的核心是BERTopic，这是一种可模块化配置的架构，允许进行无监督的主题建模。经过一系列预处理和后处理步骤后，我们为每封新邮件分配了12个主题之一及多个额外标签，这使得客户服务可以通过自定义应用程序进行筛选和访问。模型的性能通过在100封客户电子邮件的测试数据集上评估自动分配的主题的速度和准确性来衡量。该流水线具有广泛的适用性，特别是在低资源语言和形态丰富的语言方面。该系统现已被部署在公司的生产环境中，通过自动邮件分类简化了客户服务操作。 

---
# The Shady Light of Art Automation 

**Title (ZH)**: 艺术自动化之光——一片阴郁的天空 

**Authors**: Dejan Grba  

**Link**: [PDF](https://arxiv.org/pdf/2502.19107)  

**Abstract**: Generative artificial intelligence (generative AI) has entered the mainstream culture and become a subject of extensive academic investigation. However, the character and background of its impact on art require subtler scrutiny and more nuanced contextualization. This paper summarizes a broader study of the roles that AI's conceptual and ideological substrata play in influencing art notions. The focus is on divergent but coalescing and often questionable ideas, values, and political views that generative AI and other art-related AI technologies propagate from the computer science and AI/tech industry to the contemporary art and culture. The paper maps the main areas of this complex relationship and concisely critiques their key aspects. 

**Abstract (ZH)**: 生成性人工智能（生成AI）已进入主流文化，并成为广泛的学术研究对象。然而，其对艺术的影响特征及其背景需要更加细腻的审视和更加细致的情境化分析。本文总结了更广泛研究的内容，探讨了AI的概念和意识形态基础如何影响艺术观念。研究的重点在于这些分歧但相互交织、常具争议的想法、价值观和政治观点，它们从计算机科学和AI/技术行业传播到当代艺术和文化。本文绘制了这一复杂关系的主要领域，并简洁地批评了它们的关键方面。 

---
# XSS Adversarial Attacks Based on Deep Reinforcement Learning: A Replication and Extension Study 

**Title (ZH)**: 基于深度强化学习的XSS对手攻击：一项复制与扩展研究 

**Authors**: Samuele Pasini, Gianluca Maragliano, Jinhan Kim, Paolo Tonella  

**Link**: [PDF](https://arxiv.org/pdf/2502.19095)  

**Abstract**: Cross-site scripting (XSS) poses a significant threat to web application security. While Deep Learning (DL) has shown remarkable success in detecting XSS attacks, it remains vulnerable to adversarial attacks due to the discontinuous nature of its input-output mapping. These adversarial attacks employ mutation-based strategies for different components of XSS attack vectors, allowing adversarial agents to iteratively select mutations to evade detection. Our work replicates a state-of-the-art XSS adversarial attack, highlighting threats to validity in the reference work and extending it toward a more effective evaluation strategy. Moreover, we introduce an XSS Oracle to mitigate these threats. The experimental results show that our approach achieves an escape rate above 96% when the threats to validity of the replicated technique are addressed. 

**Abstract (ZH)**: 跨站点脚本（XSS）对网络应用程序的安全构成了显著威胁。尽管深度学习（DL）在检测XSS攻击方面取得了显著成效，但由于其输入-输出映射的不连续性，它仍然容易受到对抗性攻击。这些对抗性攻击通过针对XSS攻击向量的不同组件采用变异策略，使得对抗性代理能够逐步选择变异以逃避检测。我们的工作重现了一种最先进的XSS对抗性攻击，并指出了参考作品中的有效性的潜在威胁，并进一步提出了更有效的评估策略。此外，我们引入了一个XSS Oracle来应对这些威胁。实验结果表明，在解决了重现技术的有效性威胁后，我们的方法实现了超过96%的逃逸率。 

---
# InternVQA: Advancing Compressed Video QualityAssessment with Distilling Large Foundation Model 

**Title (ZH)**: InternVQA：通过提炼大基数基础模型推动压缩视频质量评估发展 

**Authors**: Fengbin Guan, Zihao Yu, Yiting Lu, Xin Li, Zhibo Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.19026)  

**Abstract**: Video quality assessment tasks rely heavily on the rich features required for video understanding, such as semantic information, texture, and temporal motion. The existing video foundational model, InternVideo2, has demonstrated strong potential in video understanding tasks due to its large parameter size and large-scale multimodal data pertaining. Building on this, we explored the transferability of InternVideo2 to video quality assessment under compression scenarios. To design a lightweight model suitable for this task, we proposed a distillation method to equip the smaller model with rich compression quality priors. Additionally, we examined the performance of different backbones during the distillation process. The results showed that, compared to other methods, our lightweight model distilled from InternVideo2 achieved excellent performance in compression video quality assessment. 

**Abstract (ZH)**: 视频质量评估任务依赖于丰富的视频理解特征，如语义信息、纹理和 temporal 运动。现有的视频基础模型 InternVideo2 在视频理解任务中展示了强大的潜力，这得益于其庞大的参数量和大规模多模态数据。基于此，我们探讨了 InternVideo2 在压缩场景下的可迁移性应用于视频质量评估。为了设计适合此任务的轻量级模型，我们提出了一种蒸馏方法，使小模型具备丰富的压缩质量先验知识。此外，我们还在蒸馏过程中考察了不同主干网络的表现。结果表明，与其它方法相比，从 InternVideo2 蒸馏出的轻量级模型在压缩视频质量评估任务中取得了优异的性能。 

---
# Ground-level Viewpoint Vision-and-Language Navigation in Continuous Environments 

**Title (ZH)**: 连续环境下的地面视点视觉-语言导航 

**Authors**: Zerui Li, Gengze Zhou, Haodong Hong, Yanyan Shao, Wenqi Lyu, Yanyuan Qiao, Qi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19024)  

**Abstract**: Vision-and-Language Navigation (VLN) empowers agents to associate time-sequenced visual observations with corresponding instructions to make sequential decisions. However, generalization remains a persistent challenge, particularly when dealing with visually diverse scenes or transitioning from simulated environments to real-world deployment. In this paper, we address the mismatch between human-centric instructions and quadruped robots with a low-height field of view, proposing a Ground-level Viewpoint Navigation (GVNav) approach to mitigate this issue. This work represents the first attempt to highlight the generalization gap in VLN across varying heights of visual observation in realistic robot deployments. Our approach leverages weighted historical observations as enriched spatiotemporal contexts for instruction following, effectively managing feature collisions within cells by assigning appropriate weights to identical features across different viewpoints. This enables low-height robots to overcome challenges such as visual obstructions and perceptual mismatches. Additionally, we transfer the connectivity graph from the HM3D and Gibson datasets as an extra resource to enhance spatial priors and a more comprehensive representation of real-world scenarios, leading to improved performance and generalizability of the waypoint predictor in real-world environments. Extensive experiments demonstrate that our Ground-level Viewpoint Navigation (GVnav) approach significantly improves performance in both simulated environments and real-world deployments with quadruped robots. 

**Abstract (ZH)**: 视觉和语言导航（VLN）赋予代理将时间序列的视觉观察与相应的指令关联起来，从而做出序列决策的能力。然而，在处理视觉多样化的场景或从模拟环境过渡到实际部署时，泛化仍然是一个持续的挑战。本文通过解决人类中心指令与低视野四足机器人的匹配问题，提出了一种地面视角导航（Ground-level Viewpoint Navigation, GVNav）方法来缓解这一问题。这项工作代表了首次尝试在现实机器人的不同类型视觉观察的高度范围内强调VLN的泛化差距。我们的方法利用加权的历史观察来增强时空上下文，有效地通过为不同视角中的相同特征分配适当的权重来管理特征碰撞，从而帮助低视高机器人克服视觉遮挡和感知不匹配等挑战。此外，我们从HM3D和Gibson数据集中转移连接图作为额外资源，增强空间先验并提供更全面的现实世界场景表示，从而提高路径点预测模型在现实世界环境中的性能和泛化能力。广泛的实验表明，我们的地面视角导航（GVNav）方法在四足机器人模拟环境和实际部署中显著提高了性能。 

---
# Robust Over-the-Air Computation with Type-Based Multiple Access 

**Title (ZH)**: 基于类型选择的稳健空中计算 

**Authors**: Marc Martinez-Gost, Ana Pérez-Neira, Miguel Ángel Lagunas  

**Link**: [PDF](https://arxiv.org/pdf/2502.19014)  

**Abstract**: This paper utilizes the properties of type-based multiple access (TBMA) to investigate its effectiveness as a robust approach for over-the-air computation (AirComp) in the presence of Byzantine attacks, this is, adversarial strategies where malicious nodes intentionally distort their transmissions to corrupt the aggregated result. Unlike classical direct aggregation (DA) AirComp, which aggregates data in the amplitude of the signals and are highly vulnerable to attacks, TBMA distributes data over multiple radio resources, enabling the receiver to construct a histogram representation of the transmitted data. This structure allows the integration of classical robust estimators and supports the computation of diverse functions beyond the arithmetic mean, which is not feasible with DA. Through extensive simulations, we demonstrate that robust TBMA significantly outperforms DA, maintaining high accuracy even under adversarial conditions, and showcases its applicability in federated learning (FEEL) scenarios. Additionally, TBMA reduces channel state information (CSI) requirements, lowers energy consumption, and enhances resiliency by leveraging the diversity of the transmitted data. These results establish TBMA as a scalable and robust solution for AirComp, paving the way for secure and efficient aggregation in next-generation networks. 

**Abstract (ZH)**: 本文利用基于类型的选择性多址接入（TBMA）的特性，探讨其在 Byzantine 攻击环境下作为空中计算（AirComp）的稳健方法的有效性。在这种攻击策略中，恶意节点故意对其传输数据进行扭曲以破坏聚合结果。与直接调幅信号聚合（DA）的 AirComp 不同，DA 在信号幅度上进行数据聚合，非常容易受到攻击的影响，而 TBMA 将数据分散到多个射频资源上，使得接收端能够构建传输数据的直方图表示。这种结构允许集成经典的稳健估计器，并支持计算算术平均值之外的多种函数，而 DA 则无法实现这一点。通过广泛的仿真实验，我们证明了稳健的 TBMA 在对抗条件下显著优于 DA，并展示了其在联邦学习（FEEL）场景中的应用潜力。此外，TBMA 还降低了信道状态信息（CSI）的要求，减少了能量消耗，并通过利用传输数据的多样性增强了系统韧性。这些结果确立了 TBMA 作为 AirComp 的可扩展且稳健的解决方案，为下一代网络中的安全高效聚合铺平了道路。 

---
# Distilling Reinforcement Learning Algorithms for In-Context Model-Based Planning 

**Title (ZH)**: 将强化学习算法.cli融入基于上下文的模型导向规划中 

**Authors**: Jaehyeon Son, Soochan Lee, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.19009)  

**Abstract**: Recent studies have shown that Transformers can perform in-context reinforcement learning (RL) by imitating existing RL algorithms, enabling sample-efficient adaptation to unseen tasks without parameter updates. However, these models also inherit the suboptimal behaviors of the RL algorithms they imitate. This issue primarily arises due to the gradual update rule employed by those algorithms. Model-based planning offers a promising solution to this limitation by allowing the models to simulate potential outcomes before taking action, providing an additional mechanism to deviate from the suboptimal behavior. Rather than learning a separate dynamics model, we propose Distillation for In-Context Planning (DICP), an in-context model-based RL framework where Transformers simultaneously learn environment dynamics and improve policy in-context. We evaluate DICP across a range of discrete and continuous environments, including Darkroom variants and Meta-World. Our results show that DICP achieves state-of-the-art performance while requiring significantly fewer environment interactions than baselines, which include both model-free counterparts and existing meta-RL methods. 

**Abstract (ZH)**: 近期的研究表明，Transformer能够在上下文环境下通过模仿现有的强化学习（RL）算法来进行强化学习，从而在不需要参数更新的情况下，高效地适应未见过的任务。然而，这些模型也会继承它们所模仿的RL算法中的次优行为。这一问题主要源于这些算法采用的逐步更新规则。基于模型的规划方法为解决这一局限性提供了可能的解决方案，因为这种方法允许模型在采取行动之前模拟潜在的结果，提供了一种额外的机制来避免次优行为。我们不是学习单独的动力学模型，而是提出了在上下文环境下同时学习环境动力学和改进策略的Distillation for In-Context Planning（DICP）框架。我们在一系列离散和连续环境（包括不同的Darkroom变体和Meta-World）中对DICP进行了评估。实验结果表明，DICP不仅能达到现有的最先进技术的性能水平，而且所需的环境交互次数远低于基线方法，这些基线方法包括无模型方法和现有的元强化学习方法。 

---
# Binary Neural Networks for Large Language Model: A Survey 

**Title (ZH)**: 大型语言模型中的二值神经网络：一个综述 

**Authors**: Liangdong Liu, Zhitong Zheng, Cong Wang, Tianhuang Su, Zhenyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19008)  

**Abstract**: Large language models (LLMs) have wide applications in the field of natural language processing(NLP), such as GPT-4 and Llama. However, with the exponential growth of model parameter sizes, LLMs bring significant resource overheads. Low-bit quantization, as a key technique, reduces memory usage and computational demands by decreasing the bit-width of model parameters, activations, and gradients. Previous quantization methods for LLMs have largely employed Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT). PTQ does not require any retraining of the original model, while QAT involves optimizing precision during training to achieve the best quantization parameters. The BitNet team proposed a radically different approach, where quantization is performed from the start of model training, utilizing low-precision binary weights during the training process. This approach has led to the emergence of many binary quantization techniques for large language models. This paper provides a comprehensive review of these binary quantization techniques. Specifically, we will introduce binary quantization techniques in deep neural networks and further explore their application to LLMs, reviewing their various contributions, implementations, and applications. 

**Abstract (ZH)**: 大语言模型（LLMs）在自然语言处理（NLP）领域有着广泛的应用，如GPT-4和Llama。然而，随着模型参数量的指数级增长，LLMs带来了显著的资源开销。低比特量化作为一种关键技术，通过降低模型参数、激活值和梯度的位宽来减少内存使用和计算需求。以往针对LLMs的量化方法主要采用后训练量化（PTQ）和量化感知训练（QAT）。PTQ不需要对原始模型进行任何重新训练，而QAT则在训练过程中优化精度以获得最佳的量化参数。BitNet团队提出了一个完全不同的方法，在模型训练开始时进行量化，并在训练过程中使用低精度的二进制权重。这一方法促成了许多针对大型语言模型的二进制量化技术的出现。本文对该领域的各种二进制量化技术进行了全面综述。具体而言，我们将介绍二进制量化技术在深度神经网络中的应用，并进一步探讨其在LLMs中的应用，回顾它们的各种贡献、实现方法和应用场景。 

---
# A Multi-Agent DRL-Based Framework for Optimal Resource Allocation and Twin Migration in the Multi-Tier Vehicular Metaverse 

**Title (ZH)**: 基于多智能体深度强化学习的多层车辆元宇宙资源最优分配与双生迁移框架 

**Authors**: Nahom Abishu Hayla, A. Mohammed Seid, Aiman Erbad, Tilahun M. Getu, Ala Al-Fuqaha, Mohsen Guizani  

**Link**: [PDF](https://arxiv.org/pdf/2502.19004)  

**Abstract**: Although multi-tier vehicular Metaverse promises to transform vehicles into essential nodes -- within an interconnected digital ecosystem -- using efficient resource allocation and seamless vehicular twin (VT) migration, this can hardly be achieved by the existing techniques operating in a highly dynamic vehicular environment, since they can hardly balance multi-objective optimization problems such as latency reduction, resource utilization, and user experience (UX). To address these challenges, we introduce a novel multi-tier resource allocation and VT migration framework that integrates Graph Convolutional Networks (GCNs), a hierarchical Stackelberg game-based incentive mechanism, and Multi-Agent Deep Reinforcement Learning (MADRL). The GCN-based model captures both spatial and temporal dependencies within the vehicular network; the Stackelberg game-based incentive mechanism fosters cooperation between vehicles and infrastructure; and the MADRL algorithm jointly optimizes resource allocation and VT migration in real time. By modeling this dynamic and multi-tier vehicular Metaverse as a Markov Decision Process (MDP), we develop a MADRL-based algorithm dubbed the Multi-Objective Multi-Agent Deep Deterministic Policy Gradient (MO-MADDPG), which can effectively balances the various conflicting objectives. Extensive simulations validate the effectiveness of this algorithm that is demonstrated to enhance scalability, reliability, and efficiency while considerably improving latency, resource utilization, migration cost, and overall UX by 12.8%, 9.7%, 14.2%, and 16.1%, respectively. 

**Abstract (ZH)**: 尽管多层次 vehicular Metaverse 有望通过高效的资源分配和无缝的 vehicular twin (VT) 迁移，将车辆转变为互联数字生态系统中的关键节点，现有的技术在高度动态的 vehicular 环境下难以实现这一点，因为它们很难在减少延迟、资源利用和用户体验 (UX) 等多目标优化问题之间取得平衡。为应对这些挑战，我们提出了一种新颖的多层次资源分配和 VT 迁移框架，该框架整合了 Graph Convolutional Networks（GCNs）、基于分层 Stackelberg 博弈的激励机制以及 Multi-Agent Deep Reinforcement Learning（MADRL）。

基于 GCN 的模型可以捕获 vehicular 网络中的时空依赖关系；基于 Stackelberg 博弈的激励机制促进了车辆与基础设施之间的合作；MADRL 算法可实现实时资源分配和 VT 迁移的联合优化。通过将这一动态的多层次 vehicular Metaverse 模型化为马尔可夫决策过程（MDP），我们开发了一个基于 MADRL 的算法，即 Multi-Objective Multi-Agent Deep Deterministic Policy Gradient（MO-MADDPG），该算法能够有效平衡各种相互冲突的目标。广泛仿真实验验证了该算法的有效性，该算法在延迟、资源利用、迁移成本和总体 UX 方面分别提高了 12.8%、9.7%、14.2% 和 16.1%，显著增强了可扩展性、可靠性和效率。 

---
# The Sharpness Disparity Principle in Transformers for Accelerating Language Model Pre-Training 

**Title (ZH)**: Transformer中判别sharpness的原则在加速语言模型预训练中的应用 

**Authors**: Jinbo Wang, Mingze Wang, Zhanpeng Zhou, Junchi Yan, Weinan E, Lei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19002)  

**Abstract**: Transformers consist of diverse building blocks, such as embedding layers, normalization layers, self-attention mechanisms, and point-wise feedforward networks. Thus, understanding the differences and interactions among these blocks is important. In this paper, we uncover a clear Sharpness Disparity across these blocks, which emerges early in training and intriguingly persists throughout the training process. Motivated by this finding, we propose Blockwise Learning Rate (LR), a strategy that tailors the LR to each block's sharpness, accelerating large language model (LLM) pre-training. By integrating Blockwise LR into AdamW, we consistently achieve lower terminal loss and nearly $2\times$ speedup compared to vanilla AdamW. We demonstrate this acceleration across GPT-2 and LLaMA, with model sizes ranging from 0.12B to 1.1B and datasets of OpenWebText and MiniPile. Finally, we incorporate Blockwise LR into Adam-mini (Zhang et al., 2024), a recently proposed memory-efficient variant of Adam, achieving a combined $2\times$ speedup and $2\times$ memory saving. These results underscore the potential of exploiting the sharpness disparity to improve LLM training. 

**Abstract (ZH)**: 本文将以下内容翻译成中文，并保持学术规范：

变压器由多种构建块组成，如嵌入层、规范化层、自注意力机制和点积前向网络。因此，理解这些构建块之间的差异及其相互作用是重要的。在本文中，我们揭示了这些构建块之间的清晰的清晰度差异（Sharpness Disparity），这种差异在训练早期出现，并且出人意料地在整个训练过程中持续存在。鉴于这一发现，我们提出了一种构建块级学习率（Blockwise Learning Rate, B-LR）策略，该策略针对每个构建块的清晰度调整学习率，从而加速大型语言模型（LLM）的预训练。通过将构建块级学习率整合到AdamW中，我们可以始终如一地实现更低的终端损失，并且比常规的AdamW快近两倍。我们通过在GPT-2和LLaMA上进行实验，展示了这种加速效果，这些模型的规模从0.12B到1.1B不等，数据集分别为OpenWebText和MiniPile。最后，我们将构建块级学习率整合到Adam-mini（Zhang et al., 2024），这是一种最近提出的新记忆高效变体中，实现了整体增速2倍和内存减少2倍的效果。这些结果强调了利用清晰度差异以改进LLM训练的潜在可能性。

注释：
- **Sharpness Disparity** 翻译为“清晰度差异”。
- **Blockwise Learning Rate (LR)** 翻译为“构建块级学习率”。
- **Blockwise LR** 翻译为“B-LR”。
- **AdamW** 翻译为“AdamW”。
- **Large Language Model (LLM)** 翻译为“大型语言模型”。
- **OpenWebText** 翻译为“OpenWebText”。
- **MiniPile** 翻译为“MiniPile”。
- **Adam-mini** 翻译为“Adam-mini”。 

---
# PEToolLLM: Towards Personalized Tool Learning in Large Language Models 

**Title (ZH)**: PEToolLLM：面向大型语言模型中的个性化工具学习 

**Authors**: Qiancheng Xu, Yongqi Li, Heming Xia, Fan Liu, Min Yang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18980)  

**Abstract**: Tool learning has emerged as a promising direction by extending Large Language Models' (LLMs) capabilities with external tools. Existing tool learning studies primarily focus on the general-purpose tool-use capability, which addresses explicit user requirements in instructions. However, they overlook the importance of personalized tool-use capability, leading to an inability to handle implicit user preferences. To address the limitation, we first formulate the task of personalized tool learning, which integrates user's interaction history towards personalized tool usage. To fill the gap of missing benchmarks, we construct PEToolBench, featuring diverse user preferences reflected in interaction history under three distinct personalized settings, and encompassing a wide range of tool-use scenarios. Moreover, we propose a framework PEToolLLaMA to adapt LLMs to the personalized tool learning task, which is trained through supervised fine-tuning and direct preference optimization. Extensive experiments on PEToolBench demonstrate the superiority of PEToolLLaMA over existing LLMs. 

**Abstract (ZH)**: 工具学习作为一种有前景的方向，通过扩展大型语言模型（LLMs）的功能，使其能够使用外部工具。现有的工具学习研究主要集中在通用工具使用能力上，该能力针对指令中的明确用户需求。然而，它们未能重视个性化工具使用能力的重要性，导致无法处理用户的隐含偏好。为了弥补这一局限，我们首先制定了个性化工具学习任务，该任务结合了用户交互历史以实现个性化工具使用。为填补缺乏基准数据的空白，我们构建了PEToolBench，该基准涵盖了三个不同个性化设置下的多样用户偏好，并包括多种工具使用场景。此外，我们提出了一种框架PEToolLLaMA，以适应个性化工具学习任务，并通过监督微调和直接偏好优化进行训练。PEToolBench上的广泛实验表明，PEToolLLaMA在性能上优于现有语言模型。 

---
# Low-Confidence Gold: Refining Low-Confidence Samples for Efficient Instruction Tuning 

**Title (ZH)**: 低置信度金数据：用于高效指令 tuning 的低置信度样本精炼 

**Authors**: Hongyi Cal, ie Li, Wenzhen Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.18978)  

**Abstract**: The effectiveness of instruction fine-tuning for Large Language Models is fundamentally constrained by the quality and efficiency of training datasets. This work introduces Low-Confidence Gold (LCG), a novel filtering framework that employs centroid-based clustering and confidence-guided selection for identifying valuable instruction pairs. Through a semi-supervised approach using a lightweight classifier trained on representative samples, LCG curates high-quality subsets while preserving data diversity. Experimental evaluation demonstrates that models fine-tuned on LCG-filtered subsets of 6K samples achieve superior performance compared to existing methods, with substantial improvements on MT-bench and consistent gains across comprehensive evaluation metrics. The framework's efficacy while maintaining model performance establishes a promising direction for efficient instruction tuning. 

**Abstract (ZH)**: 大型语言模型的指令微调效果从根本上受到训练数据集的质量和效率的限制。本文引入了一种名为低置信度黄金（LCG）的新型过滤框架，该框架结合了基于质心的聚类和置信度导向的选择，以识别有价值的指令对。通过使用轻量级分类器对代表性样本进行半监督训练，LCG 能够在保持数据多样性的前提下筛选出高质量的子集。实验评估表明，使用 LCG 过滤后的6K样本子集微调的模型在 MT-bench 等基准评测上表现更优，并在全面的评估指标上实现了显著改进。该框架在保持模型性能的同时依然有效，为高效的指令微调提供了有希望的方向。 

---
# (Mis)Fitting: A Survey of Scaling Laws 

**Title (ZH)**: 不匹配性探究：缩放定律综述

这个翻译在保持原意的同时，采用了更加符合中文学术规范的表达方式。原文中的“(Mis)Fitting”被翻译为“不匹配性”，同时“Scaling Laws”被翻译为“缩放定律”，并且在翻译成中文时使用了更加自然的表达方式“不匹配性探究：缩放定律综述”。 

**Authors**: Margaret Li, Sneha Kudugunta, Luke Zettlemoyer  

**Link**: [PDF](https://arxiv.org/pdf/2502.18969)  

**Abstract**: Modern foundation models rely heavily on using scaling laws to guide crucial training decisions. Researchers often extrapolate the optimal architecture and hyper parameters settings from smaller training runs by describing the relationship between, loss, or task performance, and scale. All components of this process vary, from the specific equation being fit, to the training setup, to the optimization method. Each of these factors may affect the fitted law, and therefore, the conclusions of a given study. We discuss discrepancies in the conclusions that several prior works reach, on questions such as the optimal token to parameter ratio. We augment this discussion with our own analysis of the critical impact that changes in specific details may effect in a scaling study, and the resulting altered conclusions. Additionally, we survey over 50 papers that study scaling trends: while 45 of these papers quantify these trends using a power law, most under-report crucial details needed to reproduce their findings. To mitigate this, we we propose a checklist for authors to consider while contributing to scaling law research. 

**Abstract (ZH)**: 现代基础模型在训练决策中高度依赖于使用放大定律。研究者们通常通过描述损失、任务性能与规模之间的关系，从较小的训练样本中推断出最优架构和超参数设置。这一过程中涉及诸多变量，包括拟合的具体方程、训练设置和优化方法等。每一种因素都可能影响放大定律的拟合结果，从而影响研究的结论。我们讨论了先前研究在诸如最优token参数比等关键问题上得出的不同结论。我们通过自身的分析补充了这一讨论，探讨了特定细节变化对放大研究结果的影响和由此产生的不同结论。此外，我们调研了超过50篇研究放大趋势的论文：其中45篇论文量化这些趋势使用了幂律模型，但多数论文未能充分报告重现其结果所需的关键细节。为解决这一问题，我们提出了一个适用于放大定律研究的作者检查清单。 

---
# DualSpec: Text-to-spatial-audio Generation via Dual-Spectrogram Guided Diffusion Model 

**Title (ZH)**: DualSpec：基于双谱图引导扩散模型的文本到空间音频生成 

**Authors**: Lei Zhao, Sizhou Chen, Linfeng Feng, Xiao-Lei Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18952)  

**Abstract**: Text-to-audio (TTA), which generates audio signals from textual descriptions, has received huge attention in recent years. However, recent works focused on text to monaural audio only. As we know, spatial audio provides more immersive auditory experience than monaural audio, e.g. in virtual reality. To address this issue, we propose a text-to-spatial-audio (TTSA) generation framework named this http URL, it first trains variational autoencoders (VAEs) for extracting the latent acoustic representations from sound event audio. Then, given text that describes sound events and event directions, the proposed method uses the encoder of a pretrained large language model to transform the text into text features. Finally, it trains a diffusion model from the latent acoustic representations and text features for the spatial audio generation. In the inference stage, only the text description is needed to generate spatial audio. Particularly, to improve the synthesis quality and azimuth accuracy of the spatial sound events simultaneously, we propose to use two kinds of acoustic features. One is the Mel spectrograms which is good for improving the synthesis quality, and the other is the short-time Fourier transform spectrograms which is good at improving the azimuth accuracy. We provide a pipeline of constructing spatial audio dataset with text prompts, for the training of the VAEs and diffusion model. We also introduce new spatial-aware evaluation metrics to quantify the azimuth errors of the generated spatial audio recordings. Experimental results demonstrate that the proposed method can generate spatial audio with high directional and event consistency. 

**Abstract (ZH)**: 文本到音频（TTA），即从文本描述生成音频信号，近年来受到了极大的关注。然而，现有的许多工作仅关注文本到单声道音频的转换。我们知道，立体声音频相较于单声道音频提供了更加沉浸的听觉体验，例如在虚拟现实环境中。为了解决这一问题，我们提出了一种名为 this http URL 的文本到立体声音频（TTSA）生成框架。该框架首先训练变分自编码器（VAEs），以从声音事件音频中提取潜在的声学表示。接着，给定描述声音事件及其方向的文本，本方法使用预训练大型语言模型的编码器将文本转化为文本特征。最后，从潜在的声学表示和文本特征中训练扩散模型，用于立体声音频的生成。在推理阶段，仅需提供文本描述即可生成立体声音频。特别地，为同时提高生成音频的质量和方位准确性，我们提出使用两种类型的声学特征。一种是梅尔频谱图，有助于提高生成质量；另一种是短时傅里叶变换频谱图，有助于提高方位准确性。我们提供了一个基于文本提示构建立体声音频数据集的流程，用于训练VAEs和扩散模型。我们还引入了新的空间感知评估指标，以量化生成的立体声音频记录的方位误差。实验结果表明，所提出的方法能够生成方向性和事件一致性高的立体声音频。 

---
# MathTutorBench: A Benchmark for Measuring Open-ended Pedagogical Capabilities of LLM Tutors 

**Title (ZH)**: MathTutorBench: 一个用于评估大型语言模型辅导工具开放性教学能力的标准测试集 

**Authors**: Jakub Macina, Nico Daheim, Ido Hakimi, Manu Kapur, Iryna Gurevych, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18940)  

**Abstract**: Evaluating the pedagogical capabilities of AI-based tutoring models is critical for making guided progress in the field. Yet, we lack a reliable, easy-to-use, and simple-to-run evaluation that reflects the pedagogical abilities of models. To fill this gap, we present MathTutorBench, an open-source benchmark for holistic tutoring model evaluation. MathTutorBench contains a collection of datasets and metrics that broadly cover tutor abilities as defined by learning sciences research in dialog-based teaching. To score the pedagogical quality of open-ended teacher responses, we train a reward model and show it can discriminate expert from novice teacher responses with high accuracy. We evaluate a wide set of closed- and open-weight models on MathTutorBench and find that subject expertise, indicated by solving ability, does not immediately translate to good teaching. Rather, pedagogy and subject expertise appear to form a trade-off that is navigated by the degree of tutoring specialization of the model. Furthermore, tutoring appears to become more challenging in longer dialogs, where simpler questioning strategies begin to fail. We release the benchmark, code, and leaderboard openly to enable rapid benchmarking of future models. 

**Abstract (ZH)**: 评估基于人工智能的辅导模型的教学能力对于推动该领域的进展至关重要。然而，我们缺乏一种可靠、易于使用且简便的操作的评估方法来反映模型的教学能力。为填补这一空白，我们提出了MathTutorBench，这是一个开源基准，用于全面评估辅导模型。MathTutorBench 包含了广泛覆盖由学习科学研究定义的基于对话教学的辅导能力的各类数据集和指标。为了评估开放型教师响应的教学质量，我们训练了一个奖励模型，并展示了它能够在高精度下区分专家与新手教师的响应。我们在 MathTutorBench 上评估了一系列闭合和开放权重模型，发现解决问题的能力所反映的学科专长并不直接转化为良好的教学。相反，教学能力和学科专长似乎形成一个权衡，这种权衡由模型的辅导专业化程度来调节。此外，在较长的对话中，辅导似乎变得更加具有挑战性，简单的提问策略开始失效。我们公开发布了基准数据、代码和排行榜，以促进未来模型的快速基准测试。 

---
# JailBench: A Comprehensive Chinese Security Assessment Benchmark for Large Language Models 

**Title (ZH)**: JailBench：面向大规模语言模型的综合性中文安全评估基准 

**Authors**: Shuyi Liu, Simiao Cui, Haoran Bu, Yuming Shang, Xi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18935)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across various applications, highlighting the urgent need for comprehensive safety evaluations. In particular, the enhanced Chinese language proficiency of LLMs, combined with the unique characteristics and complexity of Chinese expressions, has driven the emergence of Chinese-specific benchmarks for safety assessment. However, these benchmarks generally fall short in effectively exposing LLM safety vulnerabilities. To address the gap, we introduce JailBench, the first comprehensive Chinese benchmark for evaluating deep-seated vulnerabilities in LLMs, featuring a refined hierarchical safety taxonomy tailored to the Chinese context. To improve generation efficiency, we employ a novel Automatic Jailbreak Prompt Engineer (AJPE) framework for JailBench construction, which incorporates jailbreak techniques to enhance assessing effectiveness and leverages LLMs to automatically scale up the dataset through context-learning. The proposed JailBench is extensively evaluated over 13 mainstream LLMs and achieves the highest attack success rate against ChatGPT compared to existing Chinese benchmarks, underscoring its efficacy in identifying latent vulnerabilities in LLMs, as well as illustrating the substantial room for improvement in the security and trustworthiness of LLMs within the Chinese context. Our benchmark is publicly available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种应用中展现了杰出的能力，突显了全面安全评估的迫切需求。特别是，LLMs在中文语言能力上的提升，结合中文表达的独特特性和复杂性，推动了专为中国市场定制的安全评估基准的出现。然而，这些基准通常在有效揭示LLM安全漏洞方面做得并不充分。为了弥补这一差距，我们引入了JailBench，这是第一个全面评估LLMs深层漏洞的中文基准，其包含适合中文环境的细化分层安全分类学。为了提高生成效率，我们提出了一种新颖的自动监狱逃脱提示工程师（AJPE）框架，用于构建JailBench，它利用了监狱逃脱技术来增强评估效果，并通过上下文学习自动扩展数据集规模。我们提出的JailBench在13种主流LLMs上进行了广泛的评估，并在对抗ChatGPT的攻击成功率上超越了现有的中文基准，这表明它在识别LLMs中的潜在漏洞方面效果显著，同时表明在中文背景下提高LLMs的安全性和可信度仍有很大的改进空间。我们的基准已公开发布于此：[请提供一个具体的URL，这里使用了“this https URL”作为占位符]。 

---
# SLAM in the Dark: Self-Supervised Learning of Pose, Depth and Loop-Closure from Thermal Images 

**Title (ZH)**: 在黑暗中进行SLAM：从红外图像中自监督学习姿态、深度和环回闭合 

**Authors**: Yangfan Xu, Qu Hao, Lilian Zhang, Jun Mao, Xiaofeng He, Wenqi Wu, Changhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18932)  

**Abstract**: Visual SLAM is essential for mobile robots, drone navigation, and VR/AR, but traditional RGB camera systems struggle in low-light conditions, driving interest in thermal SLAM, which excels in such environments. However, thermal imaging faces challenges like low contrast, high noise, and limited large-scale annotated datasets, restricting the use of deep learning in outdoor scenarios. We present DarkSLAM, a noval deep learning-based monocular thermal SLAM system designed for large-scale localization and reconstruction in complex lighting this http URL approach incorporates the Efficient Channel Attention (ECA) mechanism in visual odometry and the Selective Kernel Attention (SKA) mechanism in depth estimation to enhance pose accuracy and mitigate thermal depth degradation. Additionally, the system includes thermal depth-based loop closure detection and pose optimization, ensuring robust performance in low-texture thermal scenes. Extensive outdoor experiments demonstrate that DarkSLAM significantly outperforms existing methods like SC-Sfm-Learner and Shin et al., delivering precise localization and 3D dense mapping even in challenging nighttime environments. 

**Abstract (ZH)**: 视觉SLAM对于移动机器人、无人机导航和VR/AR至关重要，但传统RGB摄像系统在低光条件下表现不佳，从而推动了对热SLAM技术的需求，后者在低光环境下表现出色。然而，热成像技术面临着对比度低、噪声高以及大规模注释数据集有限等挑战，这限制了深度学习在户外场景中的应用。为此，我们提出了一种名为DarkSLAM的新型基于深度学习的单目热SLAM系统，旨在复杂光照条件下实现大规模定位和重建（请参见原文链接）。该方法结合了Efficient Channel Attention (ECA)机制在视觉航位推算中的应用和Selective Kernel Attention (SKA)机制在深度估计中的应用，以提高姿态精度并减轻热深度降解问题。此外，系统还包括基于热深度的环形闭合检测和姿态优化功能，确保在低纹理热场景中的稳健性能。广泛的户外实验表明，DarkSLAM在夜间等挑战性环境下显著优于现有的方法，如SC-Sfm-Learner和Shin等人，能够提供精确的定位和3D稠密映射。 

---
# BeamVQ: Beam Search with Vector Quantization to Mitigate Data Scarcity in Physical Spatiotemporal Forecasting 

**Title (ZH)**: BeamVQ：基于向量量化的时间序列预测中物理时空数据稀缺性缓解的束搜索方法 

**Authors**: Weiyan Wang, Xingjian Shi, Ruiqi Shu, Yuan Gao, Rui Ray Chen, Kun Wang, Fan Xu, Jinbao Xue, Shuaipeng Li, Yangyu Tao, Di Wang, Hao Wu, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18925)  

**Abstract**: In practice, physical spatiotemporal forecasting can suffer from data scarcity, because collecting large-scale data is non-trivial, especially for extreme events. Hence, we propose \method{}, a novel probabilistic framework to realize iterative self-training with new self-ensemble strategies, achieving better physical consistency and generalization on extreme events. Following any base forecasting model, we can encode its deterministic outputs into a latent space and retrieve multiple codebook entries to generate probabilistic outputs. Then BeamVQ extends the beam search from discrete spaces to the continuous state spaces in this field. We can further employ domain-specific metrics (e.g., Critical Success Index for extreme events) to filter out the top-k candidates and develop the new self-ensemble strategy by combining the high-quality candidates. The self-ensemble can not only improve the inference quality and robustness but also iteratively augment the training datasets during continuous self-training. Consequently, BeamVQ realizes the exploration of rare but critical phenomena beyond the original dataset. Comprehensive experiments on different benchmarks and backbones show that BeamVQ consistently reduces forecasting MSE (up to 39%), enhancing extreme events detection and proving its effectiveness in handling data scarcity. 

**Abstract (ZH)**: 在实践中，物理时空预测可能会受到数据稀缺性的困扰，因为大规模数据的收集并不容易，尤其是在极端事件的情况下。因此，我们提出了一种名为**Method**的新颖概率框架，以实现迭代自我训练，并通过新的自我集成策略实现更好的物理一致性和泛化能力。在任何基础预测模型之后，我们都可以将该模型的确定性输出编码到一个潜在空间中，并检索多个码本条目以生成概率输出。随后，BeamVQ将在此领域的离散空间中的 beam 搜索扩展到连续状态空间中。我们还可以使用特定领域的度量（例如极端事件的批判成功率）来筛选出前k个候选者，并通过结合高质量的候选者来发展新的自我集成策略。自我集成不仅能够提高推理质量和鲁棒性，还能在连续自我训练过程中逐步增加训练数据集。因此，BeamVQ 实现了对原始数据集之外的稀有但关键现象的探索。在不同基准和骨干网络上的全面实验表明，BeamVQ 一致地降低了预测 MSE（最多降低39%），增强了极端事件的检测，并证明了其在处理数据稀缺性方面的有效性。 

---
# END: Early Noise Dropping for Efficient and Effective Context Denoising 

**Title (ZH)**: END：早期噪声剔除以实现高效有效的上下文去噪 

**Authors**: Hongye Jin, Pei Chen, Jingfeng Yang, Zhengyang Wang, Meng Jiang, Yifan Gao, Binxuan Huang, Xinyang Zhang, Zheng Li, Tianyi Liu, Huasheng Li, Bing Yin  

**Link**: [PDF](https://arxiv.org/pdf/2502.18915)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks. However, they are often distracted by irrelevant or noisy context in input sequences that degrades output quality. This problem affects both long- and short-context scenarios, such as retrieval-augmented generation, table question-answering, and in-context learning. We reveal that LLMs can implicitly identify whether input sequences contain useful information at early layers, prior to token generation. Leveraging this insight, we introduce Early Noise Dropping (\textsc{END}), a novel approach to mitigate this issue without requiring fine-tuning the LLMs. \textsc{END} segments input sequences into chunks and employs a linear prober on the early layers of LLMs to differentiate between informative and noisy chunks. By discarding noisy chunks early in the process, \textsc{END} preserves critical information, reduces distraction, and lowers computational overhead. Extensive experiments demonstrate that \textsc{END} significantly improves both performance and efficiency across different LLMs on multiple evaluation datasets. Furthermore, by investigating LLMs' implicit understanding to the input with the prober, this work also deepens understanding of how LLMs do reasoning with contexts internally. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种自然语言处理任务中展现了卓越的性能。然而，它们常常会被输入序列中的无关或噪声信息所干扰，从而降低输出质量。这个问题影响了长和短上下文场景，包括检索增强生成、表格问答和上下文学习。我们发现，LLMs可以在生成标记之前的早期层面上隐式地识别输入序列中是否包含有用信息。利用这一洞察，我们提出了一种名为早期噪声丢弃（\textsc{END}）的新方法，该方法无需对LLMs进行微调即可缓解这一问题。\textsc{END}将输入序列分段，并在LLMs的早期层面使用线性检测器来区分信息性和噪声性片段。通过在过程早期丢弃噪声性片段，\textsc{END}保留了关键信息，减少了干扰，并降低了计算开销。广泛实验表明，\textsc{END}在多个评估数据集上显著提高了不同LLMs的性能和效率。此外，通过使用探针探索LLMs对输入的隐式理解，这项工作还加深了对LLMs如何在内部进行上下文推理的理解。 

---
# Dynamic Classification: Leveraging Self-Supervised Classification to Enhance Prediction Performance 

**Title (ZH)**: 动态分类：利用自我监督分类增强预测性能 

**Authors**: Ziyuan Zhong, Junyang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.18891)  

**Abstract**: In this paper, we propose an innovative dynamic classification algorithm designed to achieve the objective of zero missed detections and minimal false positives. The algorithm partitions the data into N equivalent training subsets and N prediction subsets using a supervised model, followed by independent predictions from N separate predictive models. This enables each predictive model to operate within a smaller data range, thereby improving overall accuracy. Additionally, the algorithm leverages data generated through supervised learning to further refine prediction results, filtering out predictions that do not meet accuracy requirements without the need to introduce additional models. Experimental results demonstrate that, when data partitioning errors are minimal, the dynamic classification algorithm achieves exceptional performance with zero missed detections and minimal false positives, significantly outperforming existing model ensembles. Even in cases where classification errors are larger, the algorithm remains comparable to state of the art models. The key innovations of this study include self-supervised classification learning, the use of small-range subset predictions, and the direct rejection of substandard predictions. While the current algorithm still has room for improvement in terms of automatic parameter tuning and classification model efficiency, it has demonstrated outstanding performance across multiple datasets. Future research will focus on optimizing the classification component to further enhance the algorithm's robustness and adaptability. 

**Abstract (ZH)**: 在本文中，我们提出了一种创新的动态分类算法，旨在实现零漏检和最小误报的目标。该算法通过监督模型将数据划分为N个等价的训练子集和N个预测子集，随后由N个独立的预测模型进行独立预测。这样可以使得每个预测模型在较小的数据范围内运作，从而提高整体准确性。此外，该算法利用监督学习生成的数据进一步优化预测结果，通过过滤掉不符合精度要求的预测结果，无需引入额外的模型。实验结果表明，当数据划分误差较小时，动态分类算法能够实现零漏检和最小误报，其性能显著优于现有模型组合。即使在分类错误更大的情况下，该算法仍与最新模型相当。本研究的关键创新包括自监督分类学习、小范围子集预测以及直接拒绝劣质预测。尽管当前算法在自动参数调整和分类模型效率方面仍有改进空间，但已在多个数据集中展示了出色性能。未来的研究将集中于优化分类组件，进一步提高算法的鲁棒性和适应性。 

---
# Clip-TTS: Contrastive Text-content and Mel-spectrogram, A High-Huality Text-to-Speech Method based on Contextual Semantic Understanding 

**Title (ZH)**: Clip-TTS：对比文本内容和梅尔谱图，基于情境语义理解的高质量文本到语音方法 

**Authors**: Tianyun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18889)  

**Abstract**: Traditional text-to-speech (TTS) methods primarily focus on establishing a mapping between phonemes and mel-spectrograms. However, during the phoneme encoding stage, there is often a lack of real mel-spectrogram auxiliary information, which results in the encoding process lacking true semantic understanding. At the same time, traditional TTS systems often struggle to balance the inference speed of the model with the quality of the synthesized speech. Methods that generate high-quality synthesized speech tend to have slower inference speeds, while faster inference methods often sacrifice speech quality. In this paper, I propose Clip-TTS, a TTS method based on the Clip architecture. This method uses the Clip framework to establish a connection between text content and real mel-spectrograms during the text encoding stage, enabling the text encoder to directly learn the true semantics of the global context, thereby ensuring the quality of the synthesized speech. In terms of model architecture, I adopt the basic structure of Transformer, which allows Clip-TTS to achieve fast inference speeds. Experimental results show that on the LJSpeech and Baker datasets, the speech generated by Clip-TTS achieves state-of-the-art MOS scores, and it also performs excellently on multi-emotion this http URL samples are available at: this https URL. 

**Abstract (ZH)**: 传统文本到语音（TTS）方法主要集中在建立音素与梅尔频谱图之间的映射。然而，在音素编码阶段，通常缺乏真实的梅尔频谱图辅助信息，导致编码过程缺乏真正的语义理解。同时，传统的TTS系统往往难以在模型推理速度与合成语音质量之间取得平衡。生成高质量合成语音的方法往往会牺牲推理速度，而提速的方法则往往以牺牲语音质量为代价。在本文中，我提出了基于Clip架构的Clip-TTS方法。该方法在文本编码阶段通过Clip框架建立文本内容与真实梅尔频谱图之间的连接，使文本编码器能够直接学习全局语义的真实含义，从而保证合成语音的质量。在模型架构方面，我采用了Transformer的基本结构，使得Clip-TTS能够实现快速推理速度。实验结果显示，在LJSpeech和Baker数据集上，由Clip-TTS生成的语音在MOS评分上达到了最优水平，并且在多情感合成方面也表现出色。相关语音样本可在以下链接获取：[相关链接]。 

---
# SE(3)-Equivariant Ternary Complex Prediction Towards Target Protein Degradation 

**Title (ZH)**: 面向靶向蛋白降解的SE(3)-不变三元复杂预测 

**Authors**: Fanglei Xue, Meihan Zhang, Shuqi Li, Xinyu Gao, James A. Wohlschlegel, Wenbing Huang, Yi Yang, Weixian Deng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18875)  

**Abstract**: Targeted protein degradation (TPD) induced by small molecules has emerged as a rapidly evolving modality in drug discovery, targeting proteins traditionally considered "undruggable". Proteolysis-targeting chimeras (PROTACs) and molecular glue degraders (MGDs) are the primary small molecules that induce TPD. Both types of molecules form a ternary complex linking an E3 ligase with a target protein, a crucial step for drug discovery. While significant advances have been made in binary structure prediction for proteins and small molecules, ternary structure prediction remains challenging due to obscure interaction mechanisms and insufficient training data. Traditional methods relying on manually assigned rules perform poorly and are computationally demanding due to extensive random sampling. In this work, we introduce DeepTernary, a novel deep learning-based approach that directly predicts ternary structures in an end-to-end manner using an encoder-decoder architecture. DeepTernary leverages an SE(3)-equivariant graph neural network (GNN) with both intra-graph and ternary inter-graph attention mechanisms to capture intricate ternary interactions from our collected high-quality training dataset, TernaryDB. The proposed query-based Pocket Points Decoder extracts the 3D structure of the final binding ternary complex from learned ternary embeddings, demonstrating state-of-the-art accuracy and speed in existing PROTAC benchmarks without prior knowledge from known PROTACs. It also achieves notable accuracy on the more challenging MGD benchmark under the blind docking protocol. Remarkably, our experiments reveal that the buried surface area calculated from predicted structures correlates with experimentally obtained degradation potency-related metrics. Consequently, DeepTernary shows potential in effectively assisting and accelerating the development of TPDs for previously undruggable targets. 

**Abstract (ZH)**: 靶向蛋白降解（TPD）由小分子诱导的机制已成为药物发现领域快速发展的范式之一，能够靶向传统认为“无法成药”的蛋白质。蛋白质降解接合子（PROTACs）和分子胶降解剂（MGDs）是主要通过形成这种机制诱导TPD的两类小分子。这两种类型的分子通过形成包含E3连接酶和靶标蛋白的三元复合物，促进药物发现。尽管二元结构预测（即蛋白质和小分子的预测）取得了显著进展，但三元结构预测仍然具有挑战性，原因在于复杂的相互作用机制和缺乏足够的训练数据。传统的依赖于手动规则的方法表现不佳，且计算成本高，因需要广泛的随机取样。在此项工作中，我们引入了DeepTernary，一种新颖的基于深度学习的方法，能够以端到端的方式直接预测三元结构。DeepTernary利用SE(3)-不变图神经网络（GNN）结合内部图和三元交叉图注意力机制，从我们收集的高质量训练数据集TernaryDB中捕捉复杂的三元相互作用。提出的基于查询的口袋点解码器从学习到的三元嵌入中提取最终绑定三元复合物的3D结构，展示了在现有 PROTAC 指标上的最先进的准确性和速度，无需先验的已知 PROTAC 知识。在盲对接协议下，它也展示了在更具挑战性的 MGD 指标的准确度。值得注意的是，我们的实验证明了从预测结构计算的埋藏表面积与实验获得的降解效力相关性相关度存在显著关联。因此，DeepTernary有可能有效辅助和加速对传统意义上的“无法成药”目标的TPD开发。 

---
# Learning to Align Multi-Faceted Evaluation: A Unified and Robust Framework 

**Title (ZH)**: 学习多方面评价对齐：一个统一且 robust 的框架 

**Authors**: Kaishuai Xu, Tiezheng Yu, Wenjun Hou, Yi Cheng, Liangyou Li, Xin Jiang, Lifeng Shang, Qun Liu, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18874)  

**Abstract**: Large Language Models (LLMs) are being used more and more extensively for automated evaluation in various scenarios. Previous studies have attempted to fine-tune open-source LLMs to replicate the evaluation explanations and judgments of powerful proprietary models, such as GPT-4. However, these methods are largely limited to text-based analyses under predefined general criteria, resulting in reduced adaptability for unseen instructions and demonstrating instability in evaluating adherence to quantitative and structural constraints. To address these limitations, we propose a novel evaluation framework, ARJudge, that adaptively formulates evaluation criteria and synthesizes both text-based and code-driven analyses to evaluate LLM responses. ARJudge consists of two components: a fine-tuned Analyzer that generates multi-faceted evaluation analyses and a tuning-free Refiner that combines and refines all analyses to make the final judgment. We construct a Composite Analysis Corpus that integrates tasks for evaluation criteria generation alongside text-based and code-driven analysis generation to train the Analyzer. Our results demonstrate that ARJudge outperforms existing fine-tuned evaluators in effectiveness and robustness. Furthermore, it demonstrates the importance of multi-faceted evaluation and code-driven analyses in enhancing evaluation capabilities. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种场景中被越来越多地用于自动化评估。先前的研究试图对开放源代码的LLMs进行微调，以复制强私有模型（如GPT-4）的评估解释和判断。然而，这些方法主要限于在预定的通用标准下的文本分析，导致在处理未见过的指令时的适应性降低，并在评估遵守定量和结构约束方面表现出不稳定。为解决这些局限性，我们提出了一种新的评估框架ARJudge，该框架能够适配地制定评估标准，并综合文本驱动和编码驱动的分析来评估LLM的响应。ARJudge由两个组件组成：一个微调的Analyzer生成多维度的评估分析，以及一个无需微调的Refiner综合并精炼所有分析以做出最终判断。我们构建了一个综合分析语料库，该语料库整合了评估标准生成任务以及文本驱动和编码驱动分析的生成任务，用于训练Analyzer。我们的结果显示，ARJudge在效果和稳健性方面优于现有的微调评估器。此外，它还强调了多维度评估和编码驱动分析在提高评估能力方面的重要性。 

---
# Inscanner: Dual-Phase Detection and Classification of Auxiliary Insulation Using YOLOv8 Models 

**Title (ZH)**: Inscanner：使用YOLOv8模型的辅助绝缘双重检测与分类 

**Authors**: Youngtae Kim, Soonju Jeong, Sardar Arslan, Dhananjay Agnihotri, Yahya Ahmed, Ali Nawaz, Jinhee Song, Hyewon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.18871)  

**Abstract**: This study proposes a two-phase methodology for detecting and classifying auxiliary insulation in structural components. In the detection phase, a YOLOv8x model is trained on a dataset of complete structural blueprints, each annotated with bounding boxes indicating areas that should contain insulation. In the classification phase, these detected insulation patches are cropped and categorized into two classes: present or missing. These are then used to train a YOLOv8x-CLS model that determines the presence or absence of auxiliary insulation. Preprocessing steps for both datasets included annotation, augmentation, and appropriate cropping of the insulation regions. The detection model achieved a mean average precision (mAP) score of 82%, while the classification model attained an accuracy of 98%. These findings demonstrate the effectiveness of the proposed approach in automating insulation detection and classification, providing a foundation for further advancements in this domain. 

**Abstract (ZH)**: 本研究提出了一种两阶段方法，用于检测和分类结构组件中的辅助绝缘。在检测阶段，使用包含完整结构蓝图的数据集训练YOLOv8x模型，每个蓝图都标注有表示应包含绝缘区域的边界框。在分类阶段，检测到的绝缘区域被裁剪并分为两类：存在或缺失。然后使用分类模型YOLOv8x-CLS来确定辅助绝缘的有无。对于两个数据集的预处理步骤包括标注、数据增强和适当的绝缘区域裁剪。检测模型的平均准确率为82%，而分类模型的准确率为98%。这些发现证明了所提出方法在自动化绝缘检测和分类方面的有效性，并为该领域的进一步发展奠定了基础。 

---
# A Theoretical Perspective: How to Prevent Model Collapse in Self-consuming Training Loops 

**Title (ZH)**: 一个理论视角：如何防止自我消耗训练循环中的模型崩溃 

**Authors**: Shi Fu, Yingjie Wang, Yuzhu Chen, Xinmei Tian, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18865)  

**Abstract**: High-quality data is essential for training large generative models, yet the vast reservoir of real data available online has become nearly depleted. Consequently, models increasingly generate their own data for further training, forming Self-consuming Training Loops (STLs). However, the empirical results have been strikingly inconsistent: some models degrade or even collapse, while others successfully avoid these failures, leaving a significant gap in theoretical understanding to explain this discrepancy. This paper introduces the intriguing notion of recursive stability and presents the first theoretical generalization analysis, revealing how both model architecture and the proportion between real and synthetic data influence the success of STLs. We further extend this analysis to transformers in in-context learning, showing that even a constant-sized proportion of real data ensures convergence, while also providing insights into optimal synthetic data sizing. 

**Abstract (ZH)**: 高质量的数据对于训练大型生成模型至关重要，然而可供在线使用的现实数据资源几乎枯竭。因此，模型越来越多地生成自己的数据以供进一步训练，形成了自消耗训练循环（Self-consuming Training Loops, STLs）。然而，实验结果异常不一致：一些模型性能下降甚至崩溃，而另一些模型能够避免这些失败，其中理论理解上的巨大差距尚未解释这一差异。本文提出了递归稳定性这一引人入胜的概念，并首次进行了理论上的泛化分析，揭示了模型结构和真实数据与合成数据比例如何影响STLs的成功率。我们进一步将这一分析扩展到上下文学习中的变压器模型中，表明即使保持恒定比例的真实数据也能确保收敛，并提供了关于最优合成数据规模的见解。 

---
# Sherlock: Towards Multi-scene Video Abnormal Event Extraction and Localization via a Global-local Spatial-sensitive LLM 

**Title (ZH)**: Sherlock：通过全局-局部空间敏感的大语言模型实现多场景视频异常事件提取与定位 

**Authors**: Junxiao Ma, Jingjing Wang, Jiamin Luo, Peiying Yu, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.18863)  

**Abstract**: Prior studies on Video Anomaly Detection (VAD) mainly focus on detecting whether each video frame is abnormal or not in the video, which largely ignore the structured video semantic information (i.e., what, when, and where does the abnormal event happen). With this in mind, we propose a new chat-paradigm \textbf{M}ulti-scene Video Abnormal Event Extraction and Localization (M-VAE) task, aiming to extract the abnormal event quadruples (i.e., subject, event type, object, scene) and localize such event. Further, this paper believes that this new task faces two key challenges, i.e., global-local spatial modeling and global-local spatial balancing. To this end, this paper proposes a Global-local Spatial-sensitive Large Language Model (LLM) named Sherlock, i.e., acting like Sherlock Holmes to track down the criminal events, for this M-VAE task. Specifically, this model designs a Global-local Spatial-enhanced MoE (GSM) module and a Spatial Imbalance Regulator (SIR) to address the two challenges respectively. Extensive experiments on our M-VAE instruction dataset show the significant advantages of Sherlock over several advanced Video-LLMs. This justifies the importance of global-local spatial information for the M-VAE task and the effectiveness of Sherlock in capturing such information. 

**Abstract (ZH)**: 以下是论文内容或标题的中文翻译，符合学术规范：

先前对视频异常检测（VAD）的研究主要关注检测视频中的每一帧是否异常，很大程度上忽视了视频的结构化语义信息（即异常事件发生的时间、地点及具体内容）。基于这一点，本文提出了一种新的对话范式任务——多场景视频异常事件提取与定位（M-VAE），旨在提取异常事件四元组（即主体、事件类型、对象、场景），并定位这些事件。进一步地，本文认为这一新任务面临两个关键挑战，即全局与局部空间建模以及全局与局部空间平衡。为解决这些问题，本文提出了一种全局与局部空间敏感的大语言模型（LLM）——Sherlock，即类似福尔摩斯追踪犯罪事件，专门用于M-VAE任务。具体而言，该模型设计了一个增强全局与局部空间的混音模块（GSM）和一个空间不平衡调节器（SIR），分别解决上述两个挑战。对于我们在M-VAE指令数据集上进行的广泛实验表明，Sherlock在与几种先进的视频LLM相比时显示出显著的优势。这证实了全局与局部空间信息对于M-VAE任务的重要性以及Sherlock在捕捉这些信息方面的有效性。 

---
# Investigating Generalization of One-shot LLM Steering Vectors 

**Title (ZH)**: 探究一-shot LLM导向向量的泛化能力 

**Authors**: Jacob Dunefsky, Arman Cohan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18862)  

**Abstract**: Steering vectors have emerged as a promising approach for interpreting and controlling LLMs, but current methods typically require large contrastive datasets that are often impractical to construct and may capture spurious correlations. We propose directly optimizing steering vectors through gradient descent on a single training example, and systematically investigate how these vectors generalize. We consider several steering optimization techniques, including multiple novel ones, and find that the resulting vectors effectively mediate safety-relevant behaviors in multiple models. Indeed, in experiments on an alignment-faking model, we are able to optimize one-shot steering vectors that induce harmful behavior on benign examples and whose negations suppress harmful behavior on malign examples. And in experiments on refusal suppression, we demonstrate that one-shot optimized steering vectors can transfer across inputs, yielding a Harmbench attack success rate of 96.9%. Furthermore, to quantitatively assess steering effectiveness in instruction-tuned models, we develop a novel evaluation framework using sequence probabilities from the corresponding base model. With this framework, we analyze how steering vectors modulate an instruction-tuned LLM's ability to recover from outputting false information, and find that this ability derives from the base model. Overall, our findings suggest that optimizing steering vectors on a single example can mediate misaligned behavior in LLMs, and provide a path toward better understanding the relationship between LLM behavior and activation space structure. 

**Abstract (ZH)**: 引导向量已成为解释和控制大型语言模型（LLMs）的一种有前途的方法，但当前的方法通常需要大量的对比数据集，这些数据集往往难以构建，并且可能会捕获虚假的相关性。我们提出了一种直接通过单个训练示例的梯度下降来优化引导向量的方法，并系统地研究了这些向量的一般化能力。我们考虑了几种引导优化技术，包括几种新颖的方法，并发现这些生成的向量在多种模型中有效地调节了与安全性相关的行为。实际上，在针对对齐欺骗模型的实验中，我们能够优化出能够诱导有害行为的单次射击引导向量，并通过其否定有效地抑制有害行为。在拒绝抑制实验中，我们展示了单次射击优化的引导向量可以在不同输入之间转移，使其对Harmbench的攻击成功率达到了96.9%。此外，为了量化指令调整模型中引导的有效性，我们开发了一种新颖的评估框架，使用与之对应的基模型的序列概率。通过这个框架，我们分析了引导向量如何调节一个指令调整的LLM从输出错误信息中恢复的能力，并发现这种能力来自于基模型。总体而言，我们的发现表明，通过对单个示例进行引导向量的优化可以调节LLMs中的未对齐行为，并为更好地理解LLM行为与其激活空间结构之间的关系提供了路径。 

---
# Reimagining Personal Data: Unlocking the Potential of AI-Generated Images in Personal Data Meaning-Making 

**Title (ZH)**: 重新想象个人数据：解锁AI生成图像在个人数据意义构建中的潜力 

**Authors**: Soobin Park, Hankyung Kim, Youn-kyung Lim  

**Link**: [PDF](https://arxiv.org/pdf/2502.18853)  

**Abstract**: Image-generative AI provides new opportunities to transform personal data into alternative visual forms. In this paper, we illustrate the potential of AI-generated images in facilitating meaningful engagement with personal data. In a formative autobiographical design study, we explored the design and use of AI-generated images derived from personal data. Informed by this study, we designed a web-based application as a probe that represents personal data through generative images utilizing Open AI's GPT-4 model and DALL-E 3. We then conducted a 21-day diary study and interviews using the probe with 16 participants to investigate users' in-depth experiences with images generated by AI in everyday lives. Our findings reveal new qualities of experiences in users' engagement with data, highlighting how participants constructed personal meaning from their data through imagination and speculation on AI-generated images. We conclude by discussing the potential and concerns of leveraging image-generative AI for personal data meaning-making. 

**Abstract (ZH)**: 生成图像的人工智能为将个人数据转换为替代视觉形式提供了新的机遇。在本文中，我们探讨了人工智能生成图像在促进有意义的个人数据互动中的潜力。通过一个形成性的自传式设计研究，我们探讨了从个人数据中生成的图像的设计与应用。基于这一研究，我们设计了一个基于 web 的应用程序作为探针，通过使用 OpenAI 的 GPT-4 模型和 DALL-E 3 来代表个人数据并通过生成图像进行展示。随后，我们使用该探针对 16 名参与者进行了为期 21 天的日志研究和访谈，以调查用户在日常生活中对 AI 生成图像的体验。我们的研究结果揭示了用户与数据互动中的新体验品质，突显了参与者如何通过对 AI 生成图像的遐想和推测来构建个人意义。最后，我们讨论了利用生成图像的人工智能进行个人数据意义构建的潜在价值和关切。 

---
# Marking Code Without Breaking It: Code Watermarking for Detecting LLM-Generated Code 

**Title (ZH)**: 在不破坏代码的情况下标记代码：用于检测大语言模型生成代码的代码水印技术 

**Authors**: Jungin Kim, Shinwoo Park, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.18851)  

**Abstract**: Code watermarking identifies AI-generated code by embedding patterns into the code during generation. Effective watermarking requires meeting two key conditions: the watermark should be reliably detectable, and the code should retain its original functionality. However, existing methods often modify tokens that are critical for program logic, such as keywords in conditional expressions or operators in arithmetic computations. These modifications can cause syntax errors or functional failures, limiting the practical use of watermarking. We present STONE, a method that preserves functional integrity by selectively inserting watermarks only into non-syntax tokens. By excluding tokens essential for code execution, STONE minimizes the risk of functional degradation.
In addition, we introduce CWEM, a comprehensive evaluation metric that evaluates watermarking techniques based on correctness, detectability, and naturalness. While correctness and detectability have been widely used, naturalness remains underexplored despite its importance. Unnatural patterns can reveal the presence of a watermark, making it easier for adversaries to remove. We evaluate STONE using CWEM and compare its performance with the state-of-the-art approach. The results show that STONE achieves an average improvement of 7.69% in CWEM across Python, C++, and Java. Our code is available in this https URL. 

**Abstract (ZH)**: 代码水印技术通过在生成代码时嵌入模式来识别AI生成的代码。有效的水印需要满足两个关键条件：水印应该能够可靠地检测到，同时代码应保持其原始功能。然而，现有方法常常修改对程序逻辑至关重要的标记，如条件表达式中的关键字或算术计算中的运算符。这些修改可能导致语法错误或功能故障，限制了水印技术的实际应用。我们提出了一种名为STONE的方法，通过选择性地将水印插入非语法标记中以保留功能完整性。通过排除对代码执行至关重要的标记，STONE将功能退化的风险降至最低。

此外，我们引入了CWEM综合评估指标，该指标基于正确性、检测能力和自然性来评估水印技术。虽然正确性和检测能力已被广泛应用，但自然性仍然未被充分探索，尽管它非常重要。不自然的模式可能会揭示水印的存在，从而使得对手更容易移除。我们使用CWEM评估了STONE，并将其性能与当前最先进的方法进行了比较。结果显示，STONE在Python、C++和Java中均实现了平均7.69%的CWEM改进。我们的代码可以在以下链接获取：[提供链接处]。 

---
# A Causal Lens for Evaluating Faithfulness Metrics 

**Title (ZH)**: 用于评估忠真度度量的因果视角 

**Authors**: Kerem Zaman, Shashank Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2502.18848)  

**Abstract**: Large Language Models (LLMs) offer natural language explanations as an alternative to feature attribution methods for model interpretability. However, despite their plausibility, they may not reflect the model's internal reasoning faithfully, which is crucial for understanding the model's true decision-making processes. Although several faithfulness metrics have been proposed, a unified evaluation framework remains absent. To address this gap, we present Causal Diagnosticity, a framework to evaluate faithfulness metrics for natural language explanations. Our framework employs the concept of causal diagnosticity, and uses model-editing methods to generate faithful-unfaithful explanation pairs. Our benchmark includes four tasks: fact-checking, analogy, object counting, and multi-hop reasoning. We evaluate a variety of faithfulness metrics, including post-hoc explanation and chain-of-thought-based methods. We find that all tested faithfulness metrics often fail to surpass a random baseline. Our work underscores the need for improved metrics and more reliable interpretability methods in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）提供自然语言解释作为模型可解释性的替代方案，替代特征归因方法。然而，尽管具有良好的理论基础，这些解释可能并不忠实于模型内部的推理过程，这对于理解模型的真正决策过程至关重要。尽管已经提出了一些忠实度度量，但缺少一个统一的评估框架。为了解决这一问题，我们提出了一种因果诊断性框架，用于评估自然语言解释的忠实度度量。该框架采用因果诊断性的概念，并使用模型编辑方法生成忠实度解释配对。我们的基准包括四个任务：事实核查、类比、物体计数和多跳推理。我们评估了多种忠实度度量，包括事后解释和基于推理链的方法。我们发现，所有测试的忠实度度量通常无法超越随机基线。我们的工作强调了在LLMs中需要改进度量标准和更可靠解释方法的重要性。 

---
# Sliding Window Attention Training for Efficient Large Language Models 

**Title (ZH)**: 高效的大型语言模型的滑动窗口注意力训练 

**Authors**: Zichuan Fu, Wentao Song, Yejing Wang, Xian Wu, Yefeng Zheng, Yingying Zhang, Derong Xu, Xuetao Wei, Tong Xu, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18845)  

**Abstract**: Recent advances in transformer-based Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks. However, their quadratic computational complexity concerning sequence length remains a significant bottleneck for processing long documents. As a result, many efforts like sparse attention and state space models have been proposed to improve the efficiency of LLMs over long sequences. Though effective, these approaches compromise the performance or introduce structural complexity. This calls for a simple yet efficient model that preserves the fundamental Transformer architecture. To this end, we introduce SWAT, which enables efficient long-context handling via Sliding Window Attention Training. This paper first attributes the inefficiency of Transformers to the attention sink phenomenon resulting from the high variance of softmax operation. Then, we replace softmax with the sigmoid function and utilize a balanced ALiBi and Rotary Position Embedding for efficient information compression and retention. Experiments demonstrate that SWAT achieves SOTA performance compared with state-of-the-art linear recurrent architectures on eight benchmarks. Code is available at this https URL. 

**Abstract (ZH)**: 基于变换器的大型语言模型（LLMs）在各种任务中展现了显著的能力。然而，它们与序列长度相关的二次计算复杂性仍然是处理长文档的一个重要瓶颈。因此，许多努力，如稀疏注意机制和状态空间模型，已被提出以提高LLMs在长序列上的效率。尽管这些方法有效，但它们会牺牲性能或者增加结构复杂性。因此，需要一种简单而高效的模型，同时保留基础的变换器架构。为了解决这一问题，我们引入了SWAT，通过滑动窗口注意训练实现高效的长上下文处理。本文首先将变换器的低效归因于softmax操作的高方差导致的注意陷阱现象。然后，我们用sigmoid函数替代softmax，并利用平衡的ALiBi和旋转位置嵌入来实现高效的信息压缩和保留。实验表明，SWAT在八个基准测试上实现了与最新线性递归架构相比的最优性能。代码可在以下链接获取：[此处链接]。 

---
# BarkXAI: A Lightweight Post-Hoc Explainable Method for Tree Species Classification with Quantifiable Concepts 

**Title (ZH)**: BarkXAI：一种基于可量化概念的轻量级后验可解释方法，用于树种分类 

**Authors**: Yunmei Huang, Songlin Hou, Zachary Nelson Horve, Songlin Fei  

**Link**: [PDF](https://arxiv.org/pdf/2502.18844)  

**Abstract**: The precise identification of tree species is fundamental to forestry, conservation, and environmental monitoring. Though many studies have demonstrated that high accuracy can be achieved using bark-based species classification, these models often function as "black boxes", limiting interpretability, trust, and adoption in critical forestry applications. Attribution-based Explainable AI (XAI) methods have been used to address this issue in related works. However, XAI applications are often dependent on local features (such as a head shape or paw in animal applications) and cannot describe global visual features (such as ruggedness or smoothness) that are present in texture-dominant images such as tree bark. Concept-based XAI methods, on the other hand, offer explanations based on global visual features with concepts, but they tend to require large overhead in building external concept image datasets and the concepts can be vague and subjective without good means of precise quantification. To address these challenges, we propose a lightweight post-hoc method to interpret visual models for tree species classification using operators and quantifiable concepts. Our approach eliminates computational overhead, enables the quantification of complex concepts, and evaluates both concept importance and the model's reasoning process. To the best of our knowledge, our work is the first study to explain bark vision models in terms of global visual features with concepts. Using a human-annotated dataset as ground truth, our experiments demonstrate that our method significantly outperforms TCAV and Llama3.2 in concept importance ranking based on Kendall's Tau, highlighting its superior alignment with human perceptions. 

**Abstract (ZH)**: 树木种属的精确诊定是林业、保护和环境监测的基础。尽管许多研究已经证明，基于树皮的种属分类可以实现高精度，但这些模型往往作为“黑箱”运作，限制了其在关键林业应用中的解释性、可信度和采用率。基于归因的可解释AI（XAI）方法已在相关研究中用于解决这一问题。然而，XAI的应用往往依赖于局部特征（如动物应用中的头部形状或爪子），而无法描述存在于纹理主导图像（如树皮）中的全局视觉特征，如粗糙度或平滑度。另一方面，基于概念的XAI方法提供了基于全局视觉特征和概念的解释，但它们往往需要在构建外部概念图像数据集方面付出巨大代价，并且在缺乏精确量化手段的情况下，这些概念往往是模糊和主观的。为应对这些挑战，我们提出了一种使用操作符和可量化的概念来解释树皮分类视觉模型的轻量级事后方法。我们的方法消除了计算负担，允许对复杂概念进行量化，并评估概念的重要性以及模型的推理过程。据我们所知，我们的工作是第一项以全局视觉特征为基础，通过概念解释树皮视觉模型的研究。使用人工标注的数据集作为基准，我们的实验表明，我们的方法在基于肯德尔τ的相关性排序中概念重要性上显著优于TCAV和Llama3.2，突显了其与人类感知的高度一致。 

---
# Attention-Guided Integration of CLIP and SAM for Precise Object Masking in Robotic Manipulation 

**Title (ZH)**: 基于注意力引导的CLIP与SAM集成方法在机器人操作中实现精确对象遮罩 

**Authors**: Muhammad A. Muttaqien, Tomohiro Motoda, Ryo Hanai, Domae Yukiyasu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18842)  

**Abstract**: This paper introduces a novel pipeline to enhance the precision of object masking for robotic manipulation within the specific domain of masking products in convenience stores. The approach integrates two advanced AI models, CLIP and SAM, focusing on their synergistic combination and the effective use of multimodal data (image and text). Emphasis is placed on utilizing gradient-based attention mechanisms and customized datasets to fine-tune performance. While CLIP, SAM, and Grad- CAM are established components, their integration within this structured pipeline represents a significant contribution to the field. The resulting segmented masks, generated through this combined approach, can be effectively utilized as inputs for robotic systems, enabling more precise and adaptive object manipulation in the context of convenience store products. 

**Abstract (ZH)**: 本文介绍了一种新颖的工作流程，旨在提高机器人在便利店产品掩码中的操作精度。该方法整合了两种先进的AI模型——CLIP和SAM，重点关注它们的协同作用及其对多模态数据（图像和文本）的有效利用。文中强调了利用基于梯度的注意力机制和定制数据集以优化性能。尽管CLIP、SAM和Grad-CAM已经在各自领域内被广泛应用，但它们在这结构化工作流程中的整合为该领域做出了重要的贡献。通过这种综合方法生成的分割掩码可作为机器人系统输入，使机器人能在便利店产品操作中实现更精确和适应性的对象操作。 

---
# BatteryLife: A Comprehensive Dataset and Benchmark for Battery Life Prediction 

**Title (ZH)**: BatteryLife：一个全面的数据集和基准测试，用于电池寿命预测 

**Authors**: Ruifeng Tan, Weixiang Hong, Jiayue Tang, Xibin Lu, Ruijun Ma, Xiang Zheng, Jia Li, Jiaqiang Huang, Tong-Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18807)  

**Abstract**: Battery Life Prediction (BLP), which relies on time series data produced by battery degradation tests, is crucial for battery utilization, optimization, and production. Despite impressive advancements, this research area faces three key challenges. Firstly, the limited size of existing datasets impedes insights into modern battery life data. Secondly, most datasets are restricted to small-capacity lithium-ion batteries tested under a narrow range of diversity in labs, raising concerns about the generalizability of findings. Thirdly, inconsistent and limited benchmarks across studies obscure the effectiveness of baselines and leave it unclear if models popular in other time series fields are effective for BLP. To address these challenges, we propose BatteryLife, a comprehensive dataset and benchmark for BLP. BatteryLife integrates 16 datasets, offering a 2.4 times sample size compared to the previous largest dataset, and provides the most diverse battery life resource with batteries from 8 formats, 80 chemical systems, 12 operating temperatures, and 646 charge/discharge protocols, including both laboratory and industrial tests. Notably, BatteryLife is the first to release battery life datasets of zinc-ion batteries, sodium-ion batteries, and industry-tested large-capacity lithium-ion batteries. With the comprehensive dataset, we revisit the effectiveness of baselines popular in this and other time series fields. Furthermore, we propose CyclePatch, a plug-in technique that can be employed in a series of neural networks. Extensive benchmarking of 18 methods reveals that models popular in other time series fields can be unsuitable for BLP, and CyclePatch consistently improves model performance establishing state-of-the-art benchmarks. Moreover, BatteryLife evaluates model performance across aging conditions and domains. BatteryLife is available at this https URL. 

**Abstract (ZH)**: 电池寿命预测（BLP），依赖于电池退化测试产生的时间序列数据，在电池使用、优化和生产中至关重要。尽管取得了显著进展，该研究领域仍面临三大关键挑战。首先，现有数据集规模有限，阻碍了对现代电池寿命数据的理解。其次，大多数数据集仅包含实验室中窄范围条件下的少量锂离子电池，这使得研究成果的普适性受到质疑。第三，研究之间不一致且有限的基准测试掩盖了基线的有效性，不清楚其他时间序列领域流行的模型是否适用于BLP。为应对这些挑战，我们提出了一种名为BatteryLife的综合数据集和基准测试，旨在解决BLP问题。BatteryLife集成了16个数据集，相比之前最大的数据集样本量增加了2.4倍，并提供了最多样化的电池寿命资源，涵盖8种电池类型、80种化学体系、12种操作温度和646种充放电协议，包括实验室和工业测试数据。值得注意的是，BatteryLife是首次发布锌离子电池、钠离子电池和工业测试的大容量锂离子电池寿命数据集。借助综合数据集，我们重新评估了其他时间序列领域流行的基线的有效性。此外，我们提出了CyclePatch插件技术，可以应用于一系列神经网络。对18种方法的全面基准测试表明，其他时间序列领域的流行模型可能并不适用于BLP，而CyclePatch能够在多种模型中持续提升性能，建立了最新的基准标准。此外，BatteryLife评估了模型在不同老化条件和领域中的性能。BatteryLife可在以下网址访问：[该网址](this https URL)。 

---
# ANPMI: Assessing the True Comprehension Capabilities of LLMs for Multiple Choice Questions 

**Title (ZH)**: ANPMI：评估大型语言模型在多项选择题中的真实理解能力 

**Authors**: Gyeongje Cho, Yeonkyoung So, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.18798)  

**Abstract**: Multiple-choice benchmarks, consisting of various prompts and choices, are among the most widely used methods to assess a language model's natural language understanding capability. Given a specific prompt, we typically compute $P(Choice|Prompt)$ to evaluate how likely a language model is to generate the correct choice compared to incorrect ones. However, we observe that performance measured using this approach reflects not only the model's comprehension of the prompt but also its inherent biases for certain choices regardless of the prompt. This issue makes it challenging to accurately measure a model's natural language understanding, as models may select the answer without fully understanding the prompt. To address this limitation, we propose a novel metric called ANPMI, which normalizes Pointwise Mutual Information (PMI) by $-\log P(Choice)$. ANPMI provides a more accurate assessment of the model's natural language understanding by ensuring that it is challenging to answer a question without properly understanding the prompt. 

**Abstract (ZH)**: 多选基准测试由各种提示和选项组成，是评估语言模型自然语言理解能力的最常用方法之一。给定一个特定的提示，我们通常计算 $P(\text{选择}| \text{提示})$，以评估语言模型生成正确答案而非错误答案的可能性。然而，我们观察到，使用这种方法衡量的性能不仅反映了模型对提示的理解能力，还反映了模型对某些选项固有的偏好，这与提示无关。这一问题使得准确衡量模型的自然语言理解能力变得困难，因为模型可能会在未完全理解提示的情况下选择答案。为解决这一局限性，我们提出了一种新的度量标准，称为ANPMI（Adaptive Normalized Pointwise Mutual Information），该度量标准通过 $-\log P(\text{选择})$ 对点互信息（PMI）进行归一化。ANPMI 通过确保在不正确理解提示的情况下很难回答问题，提供了更准确的语言模型自然语言理解评估。 

---
# Seeing the Forest for the Trees: A Large Scale, Continuously Updating Meta-Analysis of Frontier LLMs 

**Title (ZH)**: 从树木中见森林：前沿大语言模型的大型持续更新元分析 

**Authors**: Jungsoo Park, Junmo Kang, Gabriel Stanovsky, Alan Ritter  

**Link**: [PDF](https://arxiv.org/pdf/2502.18791)  

**Abstract**: The surge of LLM studies makes synthesizing their findings challenging. Meta-analysis can uncover important trends across studies, but its use is limited by the time-consuming nature of manual data extraction. Our study presents a semi-automated approach for meta-analysis that accelerates data extraction using LLMs. It automatically identifies relevant arXiv papers, extracts experimental results and related attributes, and organizes them into a structured dataset. We conduct a comprehensive meta-analysis of frontier LLMs using an automatically extracted dataset, reducing the effort of paper surveying and data extraction by more than 93\% compared to manual approaches. We validate our dataset by showing that it reproduces key findings from a recent manual meta-analysis about Chain-of-Thought (CoT), and also uncovers new insights that go beyond it, showing for example that in-context examples benefit multimodal tasks but offer limited gains in mathematical tasks compared to CoT. Our automatically updatable dataset enables continuous tracking of target models by extracting evaluation studies as new data becomes available. Through our scientific artifacts and empirical analysis, we provide novel insights into LLMs while facilitating ongoing meta-analyses of their behavior. 

**Abstract (ZH)**: 大型语言模型（LLM）研究的激增使得综合其研究成果变得颇具挑战性。元分析能够揭示研究中的重要趋势，但其应用受限于手动数据提取的耗时性质。本研究提出了一种半自动化的方法，利用LLM加速数据提取过程，自动识别相关的arXiv论文，提取实验结果及相关属性，并将它们组织成结构化的数据集。我们利用自动提取的数据集对前沿LLM进行了全面的元分析，与手动方法相比，减少了93%以上的文章查阅和数据提取工作量。我们通过验证数据集，展示了它能够重现最近手动元分析中关于链式思维（CoT）的关键发现，并且揭示了新的见解，例如内部示例在多模态任务中受益，但在数学任务中相比于CoT提供的增益有限。我们自动生成并更新的数据集能够持续追踪目标模型，随着新数据的可用性不断增加新的评估研究。通过我们的科学制品和实证分析，我们不仅提供了关于LLM的新颖见解，还促进了对其行为的持续元分析。 

---
# NeuroTree: Hierarchical Functional Brain Pathway Decoding for Mental Health Disorders 

**Title (ZH)**: 神经树：分层次的功能脑路径解码在精神健康障碍中的应用 

**Authors**: Jun-En Ding, Dongsheng Luo, Anna Zilverstand, Feng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18786)  

**Abstract**: Analyzing functional brain networks using functional magnetic resonance imaging (fMRI) is crucial for understanding psychiatric disorders and addictive behaviors. While existing fMRI-based graph convolutional networks (GCNs) show considerable promise for feature extraction, they often fall short in characterizing complex relationships between brain regions and demographic factors and accounting for interpretable variables linked to psychiatric conditions. We propose NeuroTree to overcome these limitations, integrating a k-hop AGE-GCN with neural ordinary differential equations (ODEs). This framework leverages an attention mechanism to optimize functional connectivity (FC), thereby enhancing dynamic FC feature learning for brain disease classification. Furthermore, NeuroTree effectively decodes fMRI network features into tree structures, which improves the capture of high-order brain regional pathway features and enables the identification of hierarchical neural behavioral patterns essential for understanding disease-related brain subnetworks. Our empirical evaluations demonstrate that NeuroTree achieves state-of-the-art performance across two distinct mental disorder datasets and provides valuable insights into age-related deterioration patterns. These findings underscore the model's efficacy in predicting psychiatric disorders and elucidating their underlying neural mechanisms. 

**Abstract (ZH)**: 使用功能性磁共振成像（fMRI）分析功能性脑网络对于理解精神疾病和成瘾行为至关重要。尽管现有的基于fMRI的图卷积网络（GCNs）在特征提取方面表现出明显的潜力，但在表征脑区之间及其与人口统计学因素的复杂关系以及考虑与精神疾病相关的可解释变量方面常常存在不足。为此，我们提出了一种名为NeuroTree的方法，该方法将k-hop AGE-GCN与神经常微分方程（ODEs）相结合。该框架利用注意力机制优化功能性连接（FC），从而增强动态FC特征学习，提高脑疾病分类的性能。此外，NeuroTree有效将fMRI网络特征解码为树结构，这有助于捕捉高阶脑区路径特征，识别出理解与疾病相关的脑网络所必需的层次神经行为模式。我们的实证评估表明，NeuroTree在两个不同的精神障碍数据集中均达到了当前最佳性能，并提供了关于年龄相关恶化模式的重要见解。这些发现强调了该模型在预测精神疾病和阐明其潜在神经机制方面的有效性。 

---
# Research on Edge Computing and Cloud Collaborative Resource Scheduling Optimization Based on Deep Reinforcement Learning 

**Title (ZH)**: 基于深度强化学习的边缘计算与云协作资源调度优化研究 

**Authors**: Yuqing Wang, Xiao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18773)  

**Abstract**: This study addresses the challenge of resource scheduling optimization in edge-cloud collaborative computing using deep reinforcement learning (DRL). The proposed DRL-based approach improves task processing efficiency, reduces overall processing time, enhances resource utilization, and effectively controls task migrations. Experimental results demonstrate the superiority of DRL over traditional scheduling algorithms, particularly in managing complex task allocation, dynamic workloads, and multiple resource constraints. Despite its advantages, further improvements are needed to enhance learning efficiency, reduce training time, and address convergence issues. Future research should focus on increasing the algorithm's fault tolerance to handle more complex and uncertain scheduling scenarios, thereby advancing the intelligence and efficiency of edge-cloud computing systems. 

**Abstract (ZH)**: 本研究通过深度强化学习（DRL）解决了边缘-云协同计算中的资源调度优化挑战。提出的基于DRL的方法提高了任务处理效率、减少了整体处理时间、提升了资源利用率，并有效地控制了任务迁移。实验结果表明，DRL在处理复杂任务分配、动态工作负载和多种资源约束方面优于传统调度算法。尽管DRL具有许多优势，但仍需进一步改进以提高学习效率、减少训练时间并解决收敛问题。未来的研究应集中于提高算法的鲁棒性，以应对更复杂和不确定的调度场景，从而推动边缘-云计算系统的智能化和高效化。 

---
# Reward Shaping to Mitigate Reward Hacking in RLHF 

**Title (ZH)**: 使用回报塑型来缓解RLHF中的回报劫持问题 

**Authors**: Jiayi Fu, Xuandong Zhao, Chengyuan Yao, Heng Wang, Qi Han, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18770)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is essential for aligning large language models (LLMs) with human values. However, RLHF is susceptible to reward hacking, where the agent exploits flaws in the reward function rather than learning the intended behavior, thus degrading alignment. While reward shaping helps stabilize RLHF and partially mitigate reward hacking, a systematic investigation into shaping techniques and their underlying principles remains lacking. To bridge this gap, we present a comprehensive study of the prevalent reward shaping methods. Our analysis suggests three key design principles: (1) RL reward is ideally bounded, (2) RL benefits from rapid initial growth followed by gradual convergence, and (3) RL reward is best formulated as a function of centered reward. Guided by these insights, we propose Preference As Reward (PAR), a novel approach that leverages the latent preferences embedded within the reward model itself as the signal for reinforcement learning. We evaluated PAR on two base models, Gemma2-2B and Llama3-8B, using two datasets, Ultrafeedback-Binarized and HH-RLHF. Experimental results demonstrate PAR's superior performance over other reward shaping methods. On the AlpacaEval 2.0 benchmark, PAR achieves a win rate at least 5 percentage points higher than competing approaches. Furthermore, PAR exhibits remarkable data efficiency, requiring only a single reference reward for optimal performance, and maintains robustness against reward hacking even after two full epochs of training. Code is available at this https URL. 

**Abstract (ZH)**: 人类反馈强化学习（Reinforcement Learning from Human Feedback，RLHF）对于使大型语言模型（Large Language Models，LLMs）与人类价值观保持一致至关重要。然而，RLHF 对于奖励作弊（reward hacking）较为敏感，在这种情况下，代理利用奖励函数中的漏洞而非学习预期的行为，从而导致一致性降低。虽然奖励塑造有助于稳定 RLHF 并部分缓解奖励作弊问题，但对于塑造技术及其内在原理的系统研究仍然缺乏。为弥补这一不足，我们进行了针对常见奖励塑造方法的全面研究。我们的分析提出了三条关键设计原则：（1）理想情况下，强化学习奖励应有界，（2）强化学习受益于初始快速增长后逐渐收敛，以及（3）强化学习奖励最好以中心化奖励函数的形式表示。根据这些见解，我们提出了偏好作为奖励（Preference As Reward，PAR）这一创新方法，该方法利用奖励模型本身嵌入的潜在偏好作为强化学习信号。我们使用 Gemma2-2B 和 Llama3-8B 两种基础模型，在 Ultrafeedback-Binarized 和 HH-RLHF 两个数据集上评估了 PAR。实验结果表明，PAR 的性能优于其他奖励塑造方法。在 AlpacaEval 2.0 基准测试中，PAR 在至少比竞争对手高 5 个百分点的胜率上表现出色。此外，PAR 设计表现出显著的数据效率，在最佳性能下仅需一个参考奖励，并且即使经过完整的两轮训练也能保持对抗奖励作弊的稳健性。代码可以在以下网址获取：this https URL。 

---
# Online Prototypes and Class-Wise Hypergradients for Online Continual Learning with Pre-Trained Models 

**Title (ZH)**: 基于预训练模型的在线连续学习中在线原型和类内超梯度方法 

**Authors**: Nicolas Michel, Maorong Wang, Jiangpeng He, Toshihiko Yamasaki  

**Link**: [PDF](https://arxiv.org/pdf/2502.18762)  

**Abstract**: Continual Learning (CL) addresses the problem of learning from a data sequence where the distribution changes over time. Recently, efficient solutions leveraging Pre-Trained Models (PTM) have been widely explored in the offline CL (offCL) scenario, where the data corresponding to each incremental task is known beforehand and can be seen multiple times. However, such solutions often rely on 1) prior knowledge regarding task changes and 2) hyper-parameter search, particularly regarding the learning rate. Both assumptions remain unavailable in online CL (onCL) scenarios, where incoming data distribution is unknown and the model can observe each datum only once. Therefore, existing offCL strategies fall largely behind performance-wise in onCL, with some proving difficult or impossible to adapt to the online scenario. In this paper, we tackle both problems by leveraging Online Prototypes (OP) and Class-Wise Hypergradients (CWH). OP leverages stable output representations of PTM by updating its value on the fly to act as replay samples without requiring task boundaries or storing past data. CWH learns class-dependent gradient coefficients during training to improve over sub-optimal learning rates. We show through experiments that both introduced strategies allow for a consistent gain in accuracy when integrated with existing approaches. We will make the code fully available upon acceptance. 

**Abstract (ZH)**: 持续学习（CL）解决了数据序列中分布随时间变化的学习问题。近年来，在离线持续学习（offCL）场景中，利用预训练模型（PTM）的有效解决方案得到了广泛探索，其中每个增量任务的数据在发生前已知，并且可以多次查看。然而，这些解决方案通常依赖于1）任务变化的先验知识和2）超参数搜索，特别是关于学习率的选择。而在在线持续学习（onCL）场景中，进入的数据分布未知，模型只能观察每个数据点一次，这两种假设均不适用。因此，现有的offCL策略在onCL方面在性能上存在明显差距，有些方法难以或不可能适应在线场景。本文通过利用在线原型（OP）和类别相关梯度（CWH）来解决这两个问题。OP通过实时更新其值来利用PTM的稳定输出表示，充当重放样本，而无需任务边界或存储过去的数据。CWH在训练过程中学习类别相关的梯度系数，以改进次优的学习率。实验结果显示，这两种引入的方法在与现有方法集成时能提供一致的准确性提升。我们将在接受后完全开放代码。 

---
# Learning Autonomy: Off-Road Navigation Enhanced by Human Input 

**Title (ZH)**: 增强人类输入的离路导航自主学习 

**Authors**: Akhil Nagariya, Dimitar Filev, Srikanth Saripalli, Gaurav Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2502.18760)  

**Abstract**: In the area of autonomous driving, navigating off-road terrains presents a unique set of challenges, from unpredictable surfaces like grass and dirt to unexpected obstacles such as bushes and puddles. In this work, we present a novel learning-based local planner that addresses these challenges by directly capturing human driving nuances from real-world demonstrations using only a monocular camera. The key features of our planner are its ability to navigate in challenging off-road environments with various terrain types and its fast learning capabilities. By utilizing minimal human demonstration data (5-10 mins), it quickly learns to navigate in a wide array of off-road conditions. The local planner significantly reduces the real world data required to learn human driving preferences. This allows the planner to apply learned behaviors to real-world scenarios without the need for manual fine-tuning, demonstrating quick adjustment and adaptability in off-road autonomous driving technology. 

**Abstract (ZH)**: 在自动驾驶领域，穿越非 paved 地形为车辆带来了独特的挑战，包括多变的路面（如草地和泥土）以及突如其来的障碍物（如灌木丛和水坑）。本文中，我们提出了一个新颖的学习型局部路径规划器，该规划器通过仅使用单目摄像头的数据直接捕捉真实世界中的驾驶细微差异，以应对这些挑战。我们的规划器的关键特征在于其能够处理各种非 paved 地形条件下的挑战性环境，并且具有快速学习的能力。通过使用少量的人类演示数据（5-10 分钟），该规划器能够迅速学会如何在多种非 paved 地形条件下驾驶。该局部规划器极大地减少了学习人类驾驶偏好的实际所需数据量。这使得规划器能够在无需人工微调的情况下将学习到的行为应用到实际场景中，展示了在非 paved 地形自动驾驶技术中的快速调整和适应能力。 

---
# AgentSociety Challenge: Designing LLM Agents for User Modeling and Recommendation on Web Platforms 

**Title (ZH)**: AgentSociety 挑战：设计面向网络平台用户建模和推荐的大型语言模型代理 

**Authors**: Yuwei Yan, Yu Shang, Qingbin Zeng, Yu Li, Keyu Zhao, Zhiheng Zheng, Xuefei Ning, Tianji Wu, Shengen Yan, Yu Wang, Fengli Xu, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18754)  

**Abstract**: The AgentSociety Challenge is the first competition in the Web Conference that aims to explore the potential of Large Language Model (LLM) agents in modeling user behavior and enhancing recommender systems on web platforms. The Challenge consists of two tracks: the User Modeling Track and the Recommendation Track. Participants are tasked to utilize a combined dataset from Yelp, Amazon, and Goodreads, along with an interactive environment simulator, to develop innovative LLM agents. The Challenge has attracted 295 teams across the globe and received over 1,400 submissions in total over the course of 37 official competition days. The participants have achieved 21.9% and 20.3% performance improvement for Track 1 and Track 2 in the Development Phase, and 9.1% and 15.9% in the Final Phase, representing a significant accomplishment. This paper discusses the detailed designs of the Challenge, analyzes the outcomes, and highlights the most successful LLM agent designs. To support further research and development, we have open-sourced the benchmark environment at this https URL. 

**Abstract (ZH)**: 《AgentSociety挑战赛》是Web Conference上首次旨在探索大型语言模型（LLM）代理在模拟用户行为和提升网页平台推荐系统方面潜力的竞赛。该挑战包含两个赛道：用户建模赛道和推荐赛道。参赛者被要求利用从Yelp、Amazon和Goodreads获取的综合数据集以及互动环境模拟器来开发创新性的LLM代理。该挑战吸引了来自全球的295支队伍，在37个正式竞赛日中收到了超过1,400份提交。开发阶段，赛道1和赛道2的参与队伍分别取得了21.9%和20.3%的性能提升，在最终阶段，这两个赛道的参赛队伍分别实现了9.1%和15.9%的性能提升，代表了显著的成就。本文详细讨论了挑战的设计、分析了结果，并突出了最成功的LLM代理设计。为了支持进一步的研究和开发，我们已在以下网址开源了基准环境：[请插入网址]。 

---
# Intent Tagging: Exploring Micro-Prompting Interactions for Supporting Granular Human-GenAI Co-Creation Workflows 

**Title (ZH)**: 意图标注：探索微提示交互以支持精细的人工智能协作工作流 

**Authors**: Frederic Gmeiner, Nicolai Marquardt, Michael Bentley, Hugo Romat, Michel Pahud, David Brown, Asta Roseway, Nikolas Martelaro, Kenneth Holstein, Ken Hinckley, Nathalie Riche  

**Link**: [PDF](https://arxiv.org/pdf/2502.18737)  

**Abstract**: Despite Generative AI (GenAI) systems' potential for enhancing content creation, users often struggle to effectively integrate GenAI into their creative workflows. Core challenges include misalignment of AI-generated content with user intentions (intent elicitation and alignment), user uncertainty around how to best communicate their intents to the AI system (prompt formulation), and insufficient flexibility of AI systems to support diverse creative workflows (workflow flexibility). Motivated by these challenges, we created IntentTagger: a system for slide creation based on the notion of Intent Tags - small, atomic conceptual units that encapsulate user intent - for exploring granular and non-linear micro-prompting interactions for Human-GenAI co-creation workflows. Our user study with 12 participants provides insights into the value of flexibly expressing intent across varying levels of ambiguity, meta-intent elicitation, and the benefits and challenges of intent tag-driven workflows. We conclude by discussing the broader implications of our findings and design considerations for GenAI-supported content creation workflows. 

**Abstract (ZH)**: 尽管生成式AI（GenAI）系统在内容创作方面具有潜在优势，用户在将GenAI有效地集成到其创意工作流程中时往往遇到困难。核心挑战包括AI生成内容与用户意图之间的不一致（意图提取和对齐）、用户对如何最好地向AI系统传达其意图的不确定性（提示制定），以及AI系统缺乏灵活性以支持多样的创意工作流程（工作流程灵活性）。鉴于这些挑战，我们创建了基于“意图标签”概念的IntentTagger系统——这是一种小型、原子化的概念单位，可以封装用户意图，用于探索细化和非线性的微提示互动，以支持人类与GenAI的共创作工作流程。我们的12名参与者的研究提供了关于在不同模糊程度下灵活表达意图的价值、元意图提取以及基于意图标签工作流程的优缺点的见解。最后，我们讨论了研究结果的更广泛含义，并提出了GenAI支持的内容创作工作流程的设计考虑因素。 

---
# AI-Instruments: Embodying Prompts as Instruments to Abstract & Reflect Graphical Interface Commands as General-Purpose Tools 

**Title (ZH)**: AI-Instruments：将提示 embodied 为乐器以抽象和反思图形界面命令的通用工具 

**Authors**: Nathalie Riche, Anna Offenwanger, Frederic Gmeiner, David Brown, Hugo Romat, Michel Pahud, Nicolai Marquardt, Kori Inkpen, Ken Hinckley  

**Link**: [PDF](https://arxiv.org/pdf/2502.18736)  

**Abstract**: Chat-based prompts respond with verbose linear-sequential texts, making it difficult to explore and refine ambiguous intents, back up and reinterpret, or shift directions in creative AI-assisted design work. AI-Instruments instead embody "prompts" as interface objects via three key principles: (1) Reification of user-intent as reusable direct-manipulation instruments; (2) Reflection of multiple interpretations of ambiguous user-intents (Reflection-in-intent) as well as the range of AI-model responses (Reflection-in-response) to inform design "moves" towards a desired result; and (3) Grounding to instantiate an instrument from an example, result, or extrapolation directly from another instrument. Further, AI-Instruments leverage LLM's to suggest, vary, and refine new instruments, enabling a system that goes beyond hard-coded functionality by generating its own instrumental controls from content. We demonstrate four technology probes, applied to image generation, and qualitative insights from twelve participants, showing how AI-Instruments address challenges of intent formulation, steering via direct manipulation, and non-linear iterative workflows to reflect and resolve ambiguous intents. 

**Abstract (ZH)**: 基于聊天的提示生成冗长的线性文本响应，这使得在创意AI辅助设计工作中探索和精炼含糊不清的意图、回溯重释或调整方向变得困难。相比之下，AI工具通过三个关键原则将“提示”体现为界面对象：(1) 将用户意图具象化为可重复使用的直接操作工具；(2) 反映模糊用户意图的多种解读（意图内反思）以及AI模型响应的范围（响应内反思），以指导设计向目标结果的方向发展；(3) 从示例、结果或另一个工具的推断中直接实例化工具。此外，AI工具利用大型语言模型（LLM）建议、变化和精炼新的工具，从而生成自己的操作控制工具，超越了硬编码的功能性。我们通过应用于图像生成的四项技术探针，并基于十二名参与者提供的定性见解，展示了AI工具如何解决意图表述、直接操作引导以及非线性迭代工作流程中的挑战，以反映和解决含糊不清的意图。 

---
# Cross-Modality Investigation on WESAD Stress Classification 

**Title (ZH)**: 跨模态研究在WESAD压力分类中的应用 

**Authors**: Eric Oliver, Sagnik Dakshit  

**Link**: [PDF](https://arxiv.org/pdf/2502.18733)  

**Abstract**: Deep learning's growing prevalence has driven its widespread use in healthcare, where AI and sensor advancements enhance diagnosis, treatment, and monitoring. In mobile health, AI-powered tools enable early diagnosis and continuous monitoring of conditions like stress. Wearable technologies and multimodal physiological data have made stress detection increasingly viable, but model efficacy depends on data quality, quantity, and modality. This study develops transformer models for stress detection using the WESAD dataset, training on electrocardiograms (ECG), electrodermal activity (EDA), electromyography (EMG), respiration rate (RESP), temperature (TEMP), and 3-axis accelerometer (ACC) signals. The results demonstrate the effectiveness of single-modality transformers in analyzing physiological signals, achieving state-of-the-art performance with accuracy, precision and recall values in the range of $99.73\%$ to $99.95\%$ for stress detection. Furthermore, this study explores cross-modal performance and also explains the same using 2D visualization of the learned embedding space and quantitative analysis based on data variance. Despite the large body of work on stress detection and monitoring, the robustness and generalization of these models across different modalities has not been explored. This research represents one of the initial efforts to interpret embedding spaces for stress detection, providing valuable information on cross-modal performance. 

**Abstract (ZH)**: 深度学习的广泛应用已使其在医疗健康领域的应用日益增多，其中的人工智能和传感器进步提高了诊断、治疗和监测的效率。在移动健康领域，基于人工智能的工具能够实现早期诊断和持续监测，如压力等条件的监控。穿戴设备和多模态生理数据使得压力检测日益可行，但模型的有效性取决于数据质量、数量和模态。本研究采用WESAD数据集开发了用于压力检测的变压器模型，并在心电图（ECG）、皮肤电活动（EDA）、肌电图（EMG）、呼吸率（RESP）、温度（TEMP）和三轴加速度计（ACC）信号上进行训练。研究结果表明，单模态变压器在分析生理信号方面具有显著效果，在压力检测中实现了接近100%的性能，准确率、精确率和召回率范围分别为99.73%至99.95%。此外，本研究还探讨了跨模态性能，并通过2D可视化表示学习嵌入空间和基于数据方差的定量分析进行了解释。尽管在压力检测和监控方面的研究已有一定基础，但这些模型在不同模态下的鲁棒性和泛化能力尚未被充分探讨。本研究是解析压力检测嵌入空间的初期尝试，为跨模态性能提供了有价值的信息。 

---
# Deep-Bench: Deep Learning Benchmark Dataset for Code Generation 

**Title (ZH)**: Deep-Bench：用于代码生成的深度学习基准数据集 

**Authors**: Alireza Daghighfarsoodeh, Chung-Yu Wang, Hamed Taherkhani, Melika Sepidband, Mohammad Abdollahi, Hadi Hemmati, Hung Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2502.18726)  

**Abstract**: Deep learning (DL) has revolutionized areas such as computer vision, natural language processing, and more. However, developing DL systems is challenging due to the complexity of DL workflows. Large Language Models (LLMs), such as GPT, Claude, Llama, Mistral, etc., have emerged as promising tools to assist in DL code generation, offering potential solutions to these challenges. Despite this, existing benchmarks such as DS-1000 are limited, as they primarily focus on small DL code snippets related to pre/post-processing tasks and lack a comprehensive coverage of the full DL pipeline, including different DL phases and input data types.
To address this, we introduce DeepBench, a novel benchmark dataset designed for function-level DL code generation. DeepBench categorizes DL problems based on three key aspects: phases such as pre-processing, model construction, and training; tasks, including classification, regression, and recommendation; and input data types such as tabular, image, and text.
GPT-4o -- the state-of-the-art LLM -- achieved 31% accuracy on DeepBench, significantly lower than its 60% on DS-1000. We observed similar difficulty for other LLMs (e.g., 28% vs. 54% for Claude, 21% vs. 41% for LLaMA, and 15% vs. 20% for Mistral). This result underscores DeepBench's greater complexity. We also construct a taxonomy of issues and bugs found in LLM-generated DL code, which highlights the distinct challenges that LLMs face when generating DL code compared to general code.
Furthermore, our analysis also reveals substantial performance variations across categories, with differences of up to 7% among phases and 37% among tasks. These disparities suggest that DeepBench offers valuable insights into the LLMs' performance and areas for potential improvement in the DL domain. 

**Abstract (ZH)**: 深度学习（DL）已经在计算机视觉、自然语言处理等领域引发了革命性的变化。然而，由于DL工作流程的复杂性，开发DL系统面临着诸多挑战。大型语言模型（LLMs），如GPT、Claude、Llama、Mistral等，已经成为了辅助DL代码生成的有希望的工具，为解决这些挑战提供了潜在的解决方案。尽管如此，现有基准测试，如DS-1000，仍然有限，因为它们主要关注与预处理和后处理任务相关的较小的DL代码片段，并且缺乏对完整DL管道的全面覆盖，包括不同的DL阶段和输入数据类型。

为了解决这个问题，我们引入了DeepBench，一个用于函数级DL代码生成的新基准数据集。DeepBench根据三个关键方面对DL问题进行分类：包括预处理、模型构建和训练在内的阶段；包括分类、回归和推荐在内的任务；以及包括表格、图像和文本在内的输入数据类型。

最新的LLM GPT-4o 在DeepBench上的准确率仅为31%，远低于其在DS-1000上的60%。我们还观察到其他LLM（例如Claude的28% vs 54%，LLaMA的21% vs 41%，Mistral的15% vs 20%）同样面临类似的挑战。这一结果显示了DeepBench更高的复杂度。我们还构建了一个LLM生成的DL代码中问题和错误的分类体系，这强调了LLM在生成DL代码时与生成普通代码时面临的独特挑战。

此外，我们的分析还揭示了不同类别之间显著的性能差异，包括在不同阶段之间高达7%的差异和在不同任务之间高达37%的差异。这些差异表明，DeepBench为理解LLM在DL领域的性能提供了宝贵见解，并指出了可能改进的领域。 

---
# Bridging Critical Gaps in Convergent Learning: How Representational Alignment Evolves Across Layers, Training, and Distribution Shifts 

**Title (ZH)**: 弥合收敛学习中的关键缺口：表征对齐如何跨层、训练过程和分布转移演进 

**Authors**: Chaitanya Kapoor, Sudhanshu Srivastava, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2502.18710)  

**Abstract**: Understanding convergent learning -- the extent to which artificial and biological neural networks develop similar representations -- is crucial for neuroscience and AI, as it reveals shared learning principles and guides brain-like model design. While several studies have noted convergence in early and late layers of vision networks, key gaps remain. First, much existing work relies on a limited set of metrics, overlooking transformation invariances required for proper alignment. We compare three metrics that ignore specific irrelevant transformations: linear regression (ignoring affine transformations), Procrustes (ignoring rotations and reflections), and permutation/soft-matching (ignoring unit order). Notably, orthogonal transformations align representations nearly as effectively as more flexible linear ones, and although permutation scores are lower, they significantly exceed chance, indicating a robust representational basis. A second critical gap lies in understanding when alignment emerges during training. Contrary to expectations that convergence builds gradually with task-specific learning, our findings reveal that nearly all convergence occurs within the first epoch -- long before networks achieve optimal performance. This suggests that shared input statistics, architectural biases, or early training dynamics drive convergence rather than the final task solution. Finally, prior studies have not systematically examined how changes in input statistics affect alignment. Our work shows that out-of-distribution (OOD) inputs consistently amplify differences in later layers, while early layers remain aligned for both in-distribution and OOD inputs, suggesting that this alignment is driven by generalizable features stable across distribution shifts. These findings fill critical gaps in our understanding of representational convergence, with implications for neuroscience and AI. 

**Abstract (ZH)**: 理解收敛学习——即人工神经网络和生物神经网络在何种程度上发展出相似的表示——对于神经科学和人工智能至关重要，因为它揭示了共享的学习原则，并指导类脑模型的设计。尽管已有研究指出视觉网络早期和晚期层中存在收敛现象，但仍有许多关键缺口。首先，现有工作往往依赖于有限的度量标准，忽视了实现适当对齐所需的变换不变性。我们比较了三种忽略特定无关变换的度量：线性回归（忽略仿射变换）、Procrustes（忽略旋转和镜像变换）、排列/软匹配（忽略单位序序）。

值得注意的是，正交变换在对齐表示方面的效果几乎与更具灵活性的线性变换相当；尽管排列得分较低，但它们远超随机水平，表明存在一个稳健的表示基础。其次，另一个关键缺口在于了解对齐现象在训练过程中何时出现。与逐渐通过特定任务的学习来建立收敛的预期相反，我们的发现表明几乎所有收敛现象在一整个训练周期内就已经出现了——远在网络达到最佳性能之前。这表明共有的输入统计特性、架构偏见或早期训练动态驱动了收敛现象，而不是最终的任务解决方案。最后，先前的研究并没有系统地探讨输入统计特性变化如何影响对齐。我们的研究表明，对于分布外（OOD）输入而言，收敛现象通常加重了后期层间的差异，而早期层则在对于分布内和分布外输入情况下保持对齐，这表明这种对齐是由在分布转移中稳定的可泛化特征驱动的。

这些发现填补了我们对表示收敛理解的关键缺口，对于神经科学和人工智能领域具有重要意义。 

---
# H-FLTN: A Privacy-Preserving Hierarchical Framework for Electric Vehicle Spatio-Temporal Charge Prediction 

**Title (ZH)**: H-FLTN：一种保护隐私的分级框架，用于电动汽车时空充电预测 

**Authors**: Robert Marlin, Raja Jurdak, Alsharif Abuadbba  

**Link**: [PDF](https://arxiv.org/pdf/2502.18697)  

**Abstract**: The widespread adoption of Electric Vehicles (EVs) poses critical challenges for energy providers, particularly in predicting charging time (temporal prediction), ensuring user privacy, and managing resources efficiently in mobility-driven networks. This paper introduces the Hierarchical Federated Learning Transformer Network (H-FLTN) framework to address these challenges. H-FLTN employs a three-tier hierarchical architecture comprising EVs, community Distributed Energy Resource Management Systems (DERMS), and the Energy Provider Data Centre (EPDC) to enable accurate spatio-temporal predictions of EV charging needs while preserving privacy. Temporal prediction is enhanced using Transformer-based learning, capturing complex dependencies in charging behavior. Privacy is ensured through Secure Aggregation, Additive Secret Sharing, and Peer-to-Peer (P2P) Sharing with Augmentation, which allow only secret shares of model weights to be exchanged while securing all transmissions. To improve training efficiency and resource management, H-FLTN integrates Dynamic Client Capping Mechanism (DCCM) and Client Rotation Management (CRM), ensuring that training remains both computationally and temporally efficient as the number of participating EVs increases. DCCM optimises client participation by limiting excessive computational loads, while CRM balances training contributions across epochs, preventing imbalanced participation. Our simulation results based on large-scale empirical vehicle mobility data reveal that DCCM and CRM reduce the training time complexity with increasing EVs from linear to constant. Its integration into real-world smart city infrastructure enhances energy demand forecasting, resource allocation, and grid stability, ensuring reliability and sustainability in future mobility ecosystems. 

**Abstract (ZH)**: 电动汽车（EVs）的广泛应用对能源供应商提出了严峻挑战，特别是在预测充电时间（时间预测）、保证用户隐私以及在以移动为主导的网络中高效管理资源方面。本文提出了一种层级联邦学习变换器网络（H-FLTN）框架，以应对这些挑战。H-FLTN采用三层层级架构，包括电动汽车、社区分布式能源资源管理系统（DERMS）和能源提供商数据中心（EPDC），以实现对电动汽车充电需求的精确空间-时间预测，同时保护隐私。通过基于变换器的学习增强时间预测，捕捉充电行为中的复杂依赖关系。隐私通过安全聚合、加性秘密分享以及增强的点对点（P2P）共享得以保障，仅交换模型权重的秘密份额，确保所有传输的安全性。为了提高训练效率和资源管理，H-FLTN集成了动态客户端限制机制（DCCM）和客户端轮换管理（CRM），确保随着参与的电动汽车数量增加，训练既在计算上又在时间上保持高效。DCCM通过限制过重的计算负荷优化客户端参与，而CRM在每个训练周期内平衡训练贡献，防止参与不均衡。基于大规模实证车辆移动数据的仿真结果表明，DCCM和CRM随着电动汽车数量增加，将训练时间复杂度从线性降低到常数。其集成到现实世界的智能城市基础设施中，增强了能源需求预测、资源分配和电网稳定性，确保未来移动生态系统中的可靠性和可持续性。 

---
# Policy-as-Prompt: Rethinking Content Moderation in the Age of Large Language Models 

**Title (ZH)**: 政策即提示：在大规模语言模型时代重新思考内容审核 

**Authors**: Konstantina Palla, José Luis Redondo García, Claudia Hauff, Francesco Fabbri, Henrik Lindström, Daniel R. Taber, Andreas Damianou, Mounia Lalmas  

**Link**: [PDF](https://arxiv.org/pdf/2502.18695)  

**Abstract**: Content moderation plays a critical role in shaping safe and inclusive online environments, balancing platform standards, user expectations, and regulatory frameworks. Traditionally, this process involves operationalising policies into guidelines, which are then used by downstream human moderators for enforcement, or to further annotate datasets for training machine learning moderation models. However, recent advancements in large language models (LLMs) are transforming this landscape. These models can now interpret policies directly as textual inputs, eliminating the need for extensive data curation. This approach offers unprecedented flexibility, as moderation can be dynamically adjusted through natural language interactions. This paradigm shift raises important questions about how policies are operationalised and the implications for content moderation practices. In this paper, we formalise the emerging policy-as-prompt framework and identify five key challenges across four domains: Technical Implementation (1. translating policy to prompts, 2. sensitivity to prompt structure and formatting), Sociotechnical (3. the risk of technological determinism in policy formation), Organisational (4. evolving roles between policy and machine learning teams), and Governance (5. model governance and accountability). Through analysing these challenges across technical, sociotechnical, organisational, and governance dimensions, we discuss potential mitigation approaches. This research provides actionable insights for practitioners and lays the groundwork for future exploration of scalable and adaptive content moderation systems in digital ecosystems. 

**Abstract (ZH)**: 内容审查在塑造安全和平等的在线环境方面发挥着关键作用，平衡平台标准、用户期望和监管框架。传统上，这一过程涉及将政策操作化为指导原则，然后下游的人类审核员使用这些指导原则执行任务或进一步标注数据集以训练机器学习内容审查模型。然而，近期大型语言模型（LLMs）的发展正在改变这一格局。这些模型现在可以直接将政策解释为文本输入，从而消除大量数据整理的需要。这种方法提供了前所未有的灵活性，因为可以通过自然语言互动动态调整内容审查。这一范式转变提出了关于政策操作化及其对内容审查实践影响的重要问题。本文正式提出了新兴的政策即提示（Policy-as-Prompt）框架，并在其四个领域中识别了五个关键挑战：技术实施（1. 将政策转化为提示，2. 提示结构和格式的敏感性），社会技术（3. 政策形成的科技决定论风险），组织（4. 政策与机器学习团队间角色的演变），治理（5. 模型治理与问责制）。通过在技术、社会技术、组织和治理维度上分析这些挑战，我们讨论了潜在的缓解方法。本文为实践者提供了可操作的见解，并为未来关于可扩展和适应性强的内容审查系统的探索奠定了基础。 

---
# AI Mismatches: Identifying Potential Algorithmic Harms Before AI Development 

**Title (ZH)**: AI 不匹配：在AI开发之前识别潜在算法危害 

**Authors**: Devansh Saxena, Ji-Youn Jung, Jodi Forlizzi, Kenneth Holstein, John Zimmerman  

**Link**: [PDF](https://arxiv.org/pdf/2502.18682)  

**Abstract**: AI systems are often introduced with high expectations, yet many fail to deliver, resulting in unintended harm and missed opportunities for benefit. We frequently observe significant "AI Mismatches", where the system's actual performance falls short of what is needed to ensure safety and co-create value. These mismatches are particularly difficult to address once development is underway, highlighting the need for early-stage intervention. Navigating complex, multi-dimensional risk factors that contribute to AI Mismatches is a persistent challenge. To address it, we propose an AI Mismatch approach to anticipate and mitigate risks early on, focusing on the gap between realistic model performance and required task performance. Through an analysis of 774 AI cases, we extracted a set of critical factors, which informed the development of seven matrices that map the relationships between these factors and highlight high-risk areas. Through case studies, we demonstrate how our approach can help reduce risks in AI development. 

**Abstract (ZH)**: 人工智能系统常常伴随着高涨的期望被引入，但许多系统未能达到预期，导致未预见的危害和错失的好处。我们经常观察到显著的“AI不匹配”现象，即系统的实际性能无法满足确保安全和创造价值所需的标准。这些不匹配在开发过程中一旦形成，就变得难以解决，突显出早期干预的必要性。应对复杂的多维风险因素，导致AI不匹配的现象始终是一个持续性的挑战。为解决这一问题，我们提出了一种“AI不匹配”方法，旨在早期识别和缓解风险，集中关注实际模型性能与所需任务性能之间的差距。通过对774个AI案例的分析，我们提取了一组关键因素，并据此开发了七个矩阵，以映射这些因素之间的关系并突出高风险区域。通过案例研究，我们展示了这种方法如何帮助降低AI开发过程中的风险。 

---
# Comparing Native and Non-native English Speakers' Behaviors in Collaborative Writing through Visual Analytics 

**Title (ZH)**: 通过可视化分析比较母语者与非母语者在协作写作中的行为差异 

**Authors**: Yuexi Chen, Yimin Xiao, Kazi Tasnim Zinat, Naomi Yamashita, Ge Gao, Zhicheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18681)  

**Abstract**: Understanding collaborative writing dynamics between native speakers (NS) and non-native speakers (NNS) is critical for enhancing collaboration quality and team inclusivity. In this paper, we partnered with communication researchers to develop visual analytics solutions for comparing NS and NNS behaviors in 162 writing sessions across 27 teams. The primary challenges in analyzing writing behaviors are data complexity and the uncertainties introduced by automated methods. In response, we present \textsc{COALA}, a novel visual analytics tool that improves model interpretability by displaying uncertainties in author clusters, generating behavior summaries using large language models, and visualizing writing-related actions at multiple granularities. We validated the effectiveness of \textsc{COALA} through user studies with domain experts (N=2+2) and researchers with relevant experience (N=8). We present the insights discovered by participants using \textsc{COALA}, suggest features for future AI-assisted collaborative writing tools, and discuss the broader implications for analyzing collaborative processes beyond writing. 

**Abstract (ZH)**: 理解母语使用者（NS）与非母语使用者（NNS）之间的协作写作动态对于提高协作质量和团队包容性至关重要。本文与通信研究者合作，开发了一种可视化分析解决方案，用于比较27个团队共162次写作会话中NS和NNS的行为。在分析写作行为时的主要挑战是数据复杂性和自动化方法引入的不确定性。为此，我们提出了COALA（注：原文中COALA是一个专有名词，具体含义可能需要进一步确认）这一新型可视化分析工具，通过显示作者群组中的不确定性、使用大规模语言模型生成行为摘要以及以多个粒度级别可视化与写作相关的行为，来提高模型的可解释性。通过与领域专家（共4人）和具备相关经验的研究者（共8人）进行用户研究，我们验证了COALA的有效性。我们展示了参与者使用COALA获得的见解，建议了未来辅助协作写作的AI工具应包含的特性，并讨论了分析协作过程（不仅仅是写作）的更广泛意义。 

---
# Assistance or Disruption? Exploring and Evaluating the Design and Trade-offs of Proactive AI Programming Support 

**Title (ZH)**: 辅助还是扰乱？探索和评估主动AI编程支持的设计及其权衡 

**Authors**: Kevin Pu, Daniel Lazaro, Ian Arawjo, Haijun Xia, Ziang Xiao, Tovi Grossman, Yan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18658)  

**Abstract**: AI programming tools enable powerful code generation, and recent prototypes attempt to reduce user effort with proactive AI agents, but their impact on programming workflows remains unexplored. We introduce and evaluate Codellaborator, a design probe LLM agent that initiates programming assistance based on editor activities and task context. We explored three interface variants to assess trade-offs between increasingly salient AI support: prompt-only, proactive agent, and proactive agent with presence and context (Codellaborator). In a within-subject study (N=18), we find that proactive agents increase efficiency compared to prompt-only paradigm, but also incur workflow disruptions. However, presence indicators and \revise{interaction context support} alleviated disruptions and improved users' awareness of AI processes. We underscore trade-offs of Codellaborator on user control, ownership, and code understanding, emphasizing the need to adapt proactivity to programming processes. Our research contributes to the design exploration and evaluation of proactive AI systems, presenting design implications on AI-integrated programming workflow. 

**Abstract (ZH)**: AI 编程工具能够生成强大的代码，最近的原型尝试通过主动的AI代理减少用户的 effort。然而，它们对编程工作流的影响尚未得到探索。我们介绍并评估了 Codellaborator，这是一种基于编辑器活动和任务上下文自动启动编程辅助的 LLM 代理。我们探讨了三种界面变体，以评估逐渐显性的 AI 支持之间的权衡：仅提示、主动代理和带有存在感和交互上下文支持的主动代理（Codellaborator）。在一项针对单被试的用户研究（N=18）中，我们发现主动代理相比于仅提示的范式能够提高效率，但也导致了工作流程中断。然而，存在感指示器和交互上下文支持减轻了中断，并提高了用户对 AI 过程的意识。我们强调 Codellaborator 在用户控制、所有权和代码理解方面的权衡，并强调需要根据不同编程过程调整主动性的必要性。我们的研究为探索和评估主动AI系统的设 计做出了贡献，并提出了 AI 集成编程工作流的设计启示。 

---
# Enhancing Text Classification with a Novel Multi-Agent Collaboration Framework Leveraging BERT 

**Title (ZH)**: 利用BERT的新型多代理协作框架增强文本分类 

**Authors**: Hediyeh Baban, Sai A Pidapar, Aashutosh Nema, Sichen Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18653)  

**Abstract**: We introduce a novel multi-agent collaboration framework designed to enhance the accuracy and robustness of text classification models. Leveraging BERT as the primary classifier, our framework dynamically escalates low-confidence predictions to a specialized multi-agent system comprising Lexical, Contextual, Logic, Consensus, and Explainability agents. This collaborative approach allows for comprehensive analysis and consensus-driven decision-making, significantly improving classification performance across diverse text classification tasks. Empirical evaluations on benchmark datasets demonstrate that our framework achieves a 5.5% increase in accuracy compared to standard BERT-based classifiers, underscoring its effectiveness and academic novelty in advancing multi-agent systems within natural language processing. 

**Abstract (ZH)**: 我们提出了一种新型多智能体合作框架，旨在提高文本分类模型的准确性和鲁棒性。该框架采用BERT作为主要分类器，并动态提升低置信度的预测至一个专门的多智能体系统，包含词汇学、上下文、逻辑、共识和可解释性智能体。这种合作方法允许进行全面分析并基于共识进行决策，显著提升了各类文本分类任务的表现。在基准数据集上的实证研究表明，与标准的BERT分类器相比，我们的框架在准确率上提高了5.5%，凸显了其在自然语言处理领域内推动多智能体系统发展的有效性和学术创新性。 

---
# WhatELSE: Shaping Narrative Spaces at Configurable Level of Abstraction for AI-bridged Interactive Storytelling 

**Title (ZH)**: WhatELSE：通过可配置抽象层次构建人工智能桥接互动叙事的空间叙事结构 

**Authors**: Zhuoran Lu, Qian Zhou, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18641)  

**Abstract**: Generative AI significantly enhances player agency in interactive narratives (IN) by enabling just-in-time content generation that adapts to player actions. While delegating generation to AI makes IN more interactive, it becomes challenging for authors to control the space of possible narratives - within which the final story experienced by the player emerges from their interaction with AI. In this paper, we present WhatELSE, an AI-bridged IN authoring system that creates narrative possibility spaces from example stories. WhatELSE provides three views (narrative pivot, outline, and variants) to help authors understand the narrative space and corresponding tools leveraging linguistic abstraction to control the boundaries of the narrative space. Taking innovative LLM-based narrative planning approaches, WhatELSE further unfolds the narrative space into executable game events. Through a user study (N=12) and technical evaluations, we found that WhatELSE enables authors to perceive and edit the narrative space and generates engaging interactive narratives at play-time. 

**Abstract (ZH)**: 生成式人工智能显著增强了交互叙事（IN）中的玩家主动权，通过实现即时内容生成，这种生成能够根据玩家的行为进行调整。虽然将生成任务委托给人工智能使互动叙事更具互动性，但这也使得作者难以控制可能故事的空间——最终由玩家与人工智能互动产生的故事就诞生于这个空间中。在本文中，我们介绍了一种名为WhatELSE的AI桥梁交互叙事创作系统，该系统能够从示例故事中创建叙事可能性空间。WhatELSE提供了三种视图（叙事枢轴、大纲和变体）以帮助作者理解叙事空间，并通过语言抽象技术提供相应的工具，以控制叙事空间的边界。通过利用创新的基于大规模语言模型（LLM）的叙事规划方法，WhatELSE进一步将叙事空间扩展为可执行的游戏事件。通过一项包含12名用户的研究和技术评估，我们发现WhatELSE使作者能够感知和编辑叙事空间，并在游戏过程中生成具有互动性的故事。 

---
# Quantum Machine Learning in Precision Medicine and Drug Discovery -- A Game Changer for Tailored Treatments? 

**Title (ZH)**: 精准医疗和药物发现中的量子机器学习——量身定制治疗的革命性突破？ 

**Authors**: Markus Bertl, Alan Mott, Salvatore Sinno, Bhavika Bhalgamiya  

**Link**: [PDF](https://arxiv.org/pdf/2502.18639)  

**Abstract**: The digitization of healthcare presents numerous challenges, including the complexity of biological systems, vast data generation, and the need for personalized treatment plans. Traditional computational methods often fall short, leading to delayed and sometimes ineffective diagnoses and treatments. Quantum Computing (QC) and Quantum Machine Learning (QML) offer transformative advancements with the potential to revolutionize medicine. This paper summarizes areas where QC promises unprecedented computational power, enabling faster, more accurate diagnostics, personalized treatments, and enhanced drug discovery processes. However, integrating quantum technologies into precision medicine also presents challenges, including errors in algorithms and high costs. We show that mathematically-based techniques for specifying, developing, and verifying software (formal methods) can enhance the reliability and correctness of QC. By providing a rigorous mathematical framework, formal methods help to specify, develop, and verify systems with high precision. In genomic data analysis, formal specification languages can precisely (1) define the behavior and properties of quantum algorithms designed to identify genetic markers associated with diseases. Model checking tools can systematically explore all possible states of the algorithm to (2) ensure it behaves correctly under all conditions, while theorem proving techniques provide mathematical (3) proof that the algorithm meets its specified properties, ensuring accuracy and reliability. Additionally, formal optimization techniques can (4) enhance the efficiency and performance of quantum algorithms by reducing resource usage, such as the number of qubits and gate operations. Therefore, we posit that formal methods can significantly contribute to enabling QC to realize its full potential as a game changer in precision medicine. 

**Abstract (ZH)**: 医疗领域的数字化面临着诸多挑战，包括生物系统的复杂性、数据量的庞大以及需要个性化的治疗方案。传统计算方法往往难以满足这些需求，导致诊断和治疗延迟甚至无效。量子计算（QC）和量子机器学习（QML）为医疗领域带来了变革性的进步，有可能彻底改变医学。本文总结了量子计算在哪些领域有望提供前所未有的计算能力，从而实现更快、更准确的诊断、个性化治疗和改进药物发现过程。然而，将量子技术整合到精准医学中也带来了挑战，包括算法错误和高成本问题。通过使用基于数学的方法来规范、开发和验证软件（形式化方法），可以提高量子计算的可靠性和正确性。形式化方法提供了一个严格的数学框架，有助于精确地规范、开发和验证系统。在基因组数据分析中，形式化规格说明语言可以精确地描述（1）设计用于识别与疾病相关的遗传标记的量子算法的行为和属性。模型检查工具可以系统地探索算法的所有可能状态，以（2）确保在所有情况下均正确行为，而定理证明技术则提供了一种（3）数学证明，证明算法满足其规格属性，确保准确性和可靠性。此外，形式化优化技术可以通过减少资源使用（例如量子位和门操作的数量）来（4）提高量子算法的效率和性能。因此，我们认为形式化方法可以显著促进量子计算在精准医学中充分发挥其游戏规则改变者的作用。 

---
# Faster, Cheaper, Better: Multi-Objective Hyperparameter Optimization for LLM and RAG Systems 

**Title (ZH)**: 更快、更经济、更优：面向LLM和RAG系统的多目标超参数优化 

**Authors**: Matthew Barker, Andrew Bell, Evan Thomas, James Carr, Thomas Andrews, Umang Bhatt  

**Link**: [PDF](https://arxiv.org/pdf/2502.18635)  

**Abstract**: While Retrieval Augmented Generation (RAG) has emerged as a popular technique for improving Large Language Model (LLM) systems, it introduces a large number of choices, parameters and hyperparameters that must be made or tuned. This includes the LLM, embedding, and ranker models themselves, as well as hyperparameters governing individual RAG components. Yet, collectively optimizing the entire configuration in a RAG or LLM system remains under-explored - especially in multi-objective settings - due to intractably large solution spaces, noisy objective evaluations, and the high cost of evaluations. In this work, we introduce the first approach for multi-objective parameter optimization of cost, latency, safety and alignment over entire LLM and RAG systems. We find that Bayesian optimization methods significantly outperform baseline approaches, obtaining a superior Pareto front on two new RAG benchmark tasks. We conclude our work with important considerations for practitioners who are designing multi-objective RAG systems, highlighting nuances such as how optimal configurations may not generalize across tasks and objectives. 

**Abstract (ZH)**: 随著检索增强生成（RAG）成为提高大型语言模型（LLM）系统性能的一种流行技术，它引入了大量的选择项、参数和超参数，需要进行选择或调整。这包括LLM、嵌入和排名模型本身，以及管理RAG各个组件的超参数。然而，由于解的空间巨大、目标评估的噪声以及评估的高成本，集体优化整个RAG或LLM系统的配置仍然未被充分探索，尤其在多目标设置中更是如此。在这项工作中，我们提出了首个在LLM和RAG系统的整个配置中实现成本、延迟、安全性和对齐的多目标参数优化的方法。我们发现，贝叶斯优化方法显著优于基线方法，在两个新的RAG基准任务上取得了更优的帕累托前沿。最后，我们将有关如何设计多目标RAG系统的注意事项提供给实践者，强调最优配置可能无法在不同任务和目标之间泛化的细微之处。 

---
# Diffusion Models for conditional MRI generation 

**Title (ZH)**: 基于扩散模型的条件MRI生成方法 

**Authors**: Miguel Herencia García del Castillo, Ricardo Moya Garcia, Manuel Jesús Cerezo Mazón, Ekaitz Arriola Garcia, Pablo Menéndez Fernández-Miranda  

**Link**: [PDF](https://arxiv.org/pdf/2502.18620)  

**Abstract**: In this article, we present a Latent Diffusion Model (LDM) for the generation of brain Magnetic Resonance Imaging (MRI), conditioning its generation based on pathology (Healthy, Glioblastoma, Sclerosis, Dementia) and acquisition modality (T1w, T1ce, T2w, Flair, PD).
To evaluate the quality of the generated images, the Fréchet Inception Distance (FID) and Multi-Scale Structural Similarity Index (MS-SSIM) metrics were employed. The results indicate that the model generates images with a distribution similar to real ones, maintaining a balance between visual fidelity and diversity. Additionally, the model demonstrates extrapolation capability, enabling the generation of configurations that were not present in the training data.
The results validate the potential of the model to increase in the number of samples in clinical datasets, balancing underrepresented classes, and evaluating AI models in medicine, contributing to the development of diagnostic tools in radiology without compromising patient privacy. 

**Abstract (ZH)**: 在本文中，我们提出了一种潜扩散模型（Latent Diffusion Model, LDM），用于生成脑磁共振成像（MRI），并根据病理类型（健康、胶质母细胞瘤、硬化、痴呆）和成像模态（T1w、T1ce、T2w、FLAIR、PD）进行条件生成。
为了评估生成图像的质量，我们使用了Fréchet Inception Distance（FID）和多尺度结构相似性指数（MS-SSIM）作为评估指标。结果表明，该模型能够生成分布与真实图像相似的图像，既保持了视觉保真度，又保持了多样性。此外，该模型还展示了外推能力，能够生成训练数据中未出现的配置。
这些结果验证了该模型在增加临床数据集样本数量、平衡稀疏类别以及评估医学中的人工智能模型方面的潜力，同时有助于在不侵犯患者隐私的情况下推进放射诊断工具的发展。 

---
# Mind the Gap: Bridging the Divide Between AI Aspirations and the Reality of Autonomous Characterization 

**Title (ZH)**: 注意差距：弥合人工智能期望与自主特征化现实之间的鸿沟 

**Authors**: Grace Guinan, Addison Salvador, Michelle A. Smeaton, Andrew Glaws, Hilary Egan, Brian C. Wyatt, Babak Anasori, Kevin R. Fiedler, Matthew J. Olszta, Steven R. Spurgeon  

**Link**: [PDF](https://arxiv.org/pdf/2502.18604)  

**Abstract**: What does materials science look like in the "Age of Artificial Intelligence?" Each materials domain-synthesis, characterization, and modeling-has a different answer to this question, motivated by unique challenges and constraints. This work focuses on the tremendous potential of autonomous characterization within electron microscopy. We present our recent advancements in developing domain-aware, multimodal models for microscopy analysis capable of describing complex atomic systems. We then address the critical gap between the theoretical promise of autonomous microscopy and its current practical limitations, showcasing recent successes while highlighting the necessary developments to achieve robust, real-world autonomy. 

**Abstract (ZH)**: 人工智能时代，材料科学呈现出怎样的面貌？每个材料领域——合成、表征和建模——对这一问题的回答各有千秋，这源于它们各自独特的挑战和限制。本研究重点关注自主表征在电子显微镜中的巨大潜力。我们介绍了在显微镜分析中开发领域感知型多模态模型的最新进展，这些模型能够描述复杂的原子系统。随后，我们探讨了自主显微镜的理论前景与其当前实践限制之间的关键差距，展示了最近取得的成果，并指出了实现可靠、实用的自主性的必要发展。 

---
# Autonomous Vision-Guided Resection of Central Airway Obstruction 

**Title (ZH)**: 自动视觉引导的中央气道梗阻切除术 

**Authors**: M. E. Smith, N. Yilmaz, T. Watts, P. M. Scheikl, J. Ge, A. Deguet, A. Kuntz, A. Krieger  

**Link**: [PDF](https://arxiv.org/pdf/2502.18586)  

**Abstract**: Existing tracheal tumor resection methods often lack the precision required for effective airway clearance, and robotic advancements offer new potential for autonomous resection. We present a vision-guided, autonomous approach for palliative resection of tracheal tumors. This system models the tracheal surface with a fifth-degree polynomial to plan tool trajectories, while a custom Faster R-CNN segmentation pipeline identifies the trachea and tumor boundaries. The electrocautery tool angle is optimized using handheld surgical demonstrations, and trajectories are planned to maintain a 1 mm safety clearance from the tracheal surface. We validated the workflow successfully in five consecutive experiments on ex-vivo animal tissue models, successfully clearing the airway obstruction without trachea perforation in all cases (with more than 90% volumetric tumor removal). These results support the feasibility of an autonomous resection platform, paving the way for future developments in minimally-invasive autonomous resection. 

**Abstract (ZH)**: 现有的气管肿瘤切除方法往往缺乏有效气道清除所需的精确度，而机器人技术的进步为自主切除提供了新的可能性。我们提出了一种基于视觉引导的自主方法，用于姑息性切除气管肿瘤。该系统使用五次多项式模型气管表面，以规划工具轨迹，而自定义的Faster R-CNN分割管道用于识别气管和肿瘤边界。电凝工具的角度通过手持手术演示进行优化，并规划路径以保持与气管表面1毫米的安全距离。我们在五个连续的体外动物组织模型实验中成功验证了该工作流程，成功地清除了气道阻塞且未造成气管穿孔（肿瘤体积去除率超过90%）。这些结果支持自主切除平台的可行性，为未来微创自主切除的发展铺平了道路。 

---
# Scalable Best-of-N Selection for Large Language Models via Self-Certainty 

**Title (ZH)**: 通过自我 certainty 方式实现大规模语言模型的可扩展最佳选项选择 

**Authors**: Zhewei Kang, Xuandong Zhao, Dawn Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.18581)  

**Abstract**: Best-of-N selection is a key technique for improving the reasoning performance of Large Language Models (LLMs) through increased test-time computation. Current state-of-the-art methods often employ computationally intensive reward models for response evaluation and selection. Reward-free alternatives, like self-consistency and universal self-consistency, are limited in their ability to handle open-ended generation tasks or scale effectively. To address these limitations, we propose self-certainty, a novel and efficient metric that leverages the inherent probability distribution of LLM outputs to estimate response quality without requiring external reward models. We hypothesize that higher distributional self-certainty, aggregated across multiple samples, correlates with improved response accuracy, as it reflects greater confidence in the generated output. Through extensive experiments on various reasoning tasks, we demonstrate that self-certainty (1) scales effectively with increasing sample size $N$, akin to reward models but without the computational overhead; (2) complements chain-of-thought, improving reasoning performance beyond greedy decoding; and (3) generalizes to open-ended tasks where traditional self-consistency methods fall short. Our findings establish self-certainty as a practical and efficient way for improving LLM reasoning capabilities. The code is available at this https URL 

**Abstract (ZH)**: 最佳-of-N选择是一种关键的技术，通过增加测试时的计算量，显著提升大型语言模型（LLMs）的推理性能。当前最先进的方法通常采用计算密集型的奖励模型来评估和选择响应。自无奖励选择方法，如自一致性方法和通用自一致性方法，在处理开放生成任务或有效扩展方面能力有限。为了克服这些限制，我们提出了一种新颖且高效的度量标准——自我确定性，该标准利用LLM输出的固有概率分布来估计响应质量，无需外部奖励模型。我们假设多个样本的分布自我确定性越高，生成的输出越准确。这反映了对生成输出更大的信心。通过在多种推理任务上的广泛实验，我们证明了自我确定性（1）能够有效地随样本数量N的增加而扩展，类似于奖励模型，但不需要额外的计算开销；（2）能够补充思维链方法，超越贪婪解码，提升推理性能；（3）适用于传统自一致性方法难以胜任的开放生成任务。我们的研究结果确立了自我确定性作为一种实用且有效的手段，可以提升LLM的推理能力。相关代码可在以下链接中获取：this https URL 

---
# Differentially Private Iterative Screening Rules for Linear Regression 

**Title (ZH)**: 差分隐私迭代筛选规则用于线性回归 

**Authors**: Amol Khanna, Fred Lu, Edward Raff  

**Link**: [PDF](https://arxiv.org/pdf/2502.18578)  

**Abstract**: Linear $L_1$-regularized models have remained one of the simplest and most effective tools in data science. Over the past decade, screening rules have risen in popularity as a way to eliminate features when producing the sparse regression weights of $L_1$ models. However, despite the increasing need of privacy-preserving models for data analysis, to the best of our knowledge, no differentially private screening rule exists. In this paper, we develop the first private screening rule for linear regression. We initially find that this screening rule is too strong: it screens too many coefficients as a result of the private screening step. However, a weakened implementation of private screening reduces overscreening and improves performance. 

**Abstract (ZH)**: 线性 $L_1$-正则化模型一直是数据科学中最简单且最有效的方法之一。在过去十年中，筛选规则因其能消除线性 $L_1$ 模型中的稀疏回归权重时的特征而受到越来越多的关注。然而，尽管隐私保护模型在数据分析中的需求日益增加，据我们所知，目前尚不存在差分隐私的筛选规则。本文旨在开发首个差分隐私的筛选规则用于线性回归。我们最初发现，该筛选规则过于严格：由于隐私筛选步骤的影响，它会过多地筛选系数。然而，通过削弱隐私筛选的实施，可以减少不必要的筛选并提升性能。 

---
# FactReasoner: A Probabilistic Approach to Long-Form Factuality Assessment for Large Language Models 

**Title (ZH)**: FactReasoner：一种用于大型语言模型长篇事实性评估的概率方法 

**Authors**: Radu Marinescu, Debarun Bhattacharjya, Junkyu Lee, Tigran Tchrakian, Javier Carnerero Cano, Yufang Hou, Elizabeth Daly, Alessandra Pascale  

**Link**: [PDF](https://arxiv.org/pdf/2502.18573)  

**Abstract**: Large language models (LLMs) have demonstrated vast capabilities on generative tasks in recent years, yet they struggle with guaranteeing the factual correctness of the generated content. This makes these models unreliable in realistic situations where factually accurate responses are expected. In this paper, we propose FactReasoner, a new factuality assessor that relies on probabilistic reasoning to assess the factuality of a long-form generated response. Specifically, FactReasoner decomposes the response into atomic units, retrieves relevant contexts for them from an external knowledge source, and constructs a joint probability distribution over the atoms and contexts using probabilistic encodings of the logical relationships (entailment, contradiction) between the textual utterances corresponding to the atoms and contexts. FactReasoner then computes the posterior probability of whether atomic units in the response are supported by the retrieved contexts. Our experiments on labeled and unlabeled benchmark datasets demonstrate clearly that FactReasoner improves considerably over state-of-the-art prompt-based approaches in terms of both factual precision and recall. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在生成任务上展现了巨大的能力，但在保证生成内容的准确性方面却存在困难。这使得这些模型在期望获得事实准确回答的真实场景中不够可靠。本文提出了一种新的事实性评估器——FactReasoner，它依赖于概率推理来评估长文本生成响应的事实性。具体而言，FactReasoner 将响应分解为原子单位，从外部知识源检索与这些原子单位相关的内容，并通过逻辑关系（蕴含、矛盾）的概率编码构造原子和上下文的联合概率分布。FactReasoner 然后计算检索到的上下文是否支持响应中原子单位的概率。我们在有标签和无标签基准数据集上的实验清楚地表明，与基于提示的方法相比，FactReasoner 在事实精密性和召回率方面有了显著改进。 

---
# Application of Attention Mechanism with Bidirectional Long Short-Term Memory (BiLSTM) and CNN for Human Conflict Detection using Computer Vision 

**Title (ZH)**: 使用计算机视觉的人体冲突检测中基于注意机制的双向长短期记忆(BiLSTM)和卷积神经网络(CNN)的应用 

**Authors**: Erick da Silva Farias, Eduardo Palhares Junior  

**Link**: [PDF](https://arxiv.org/pdf/2502.18555)  

**Abstract**: The automatic detection of human conflicts through videos is a crucial area in computer vision, with significant applications in monitoring and public safety policies. However, the scarcity of public datasets and the complexity of human interactions make this task challenging. This study investigates the integration of advanced deep learning techniques, including Attention Mechanism, Convolutional Neural Networks (CNNs), and Bidirectional Long ShortTerm Memory (BiLSTM), to improve the detection of violent behaviors in videos. The research explores how the use of the attention mechanism can help focus on the most relevant parts of the video, enhancing the accuracy and robustness of the model. The experiments indicate that the combination of CNNs with BiLSTM and the attention mechanism provides a promising solution for conflict monitoring, offering insights into the effectiveness of different strategies. This work opens new possibilities for the development of automated surveillance systems that can operate more efficiently in real-time detection of violent events. 

**Abstract (ZH)**: 通过视频自动检测人类冲突是计算机视觉领域的一个关键研究方向，具有在监控和公共安全政策中的重要应用。然而，公共数据集的缺乏以及人类互动的复杂性使这项任务面临挑战。本研究探讨了将先进的深度学习技术，包括注意力机制、卷积神经网络（CNNs）和双向长短期记忆网络（BiLSTM）相结合，以提高视频中暴力行为检测的性能。研究探讨了注意力机制如何帮助聚焦于视频中最具相关性的部分，从而提高模型的准确性和鲁棒性。实验结果表明，CNNs与BiLSTM以及注意力机制的结合提供了一种有前景的解决方案，用以监测冲突，揭示了不同策略的有效性。这项工作为开发能够更高效地实时检测暴力事件的自动化监控系统打开了新的可能性。 

---
# Applications of Statistical Field Theory in Deep Learning 

**Title (ZH)**: 统计场理论在深度学习中的应用 

**Authors**: Zohar Ringel, Noa Rubin, Edo Mor, Moritz Helias, Inbar Seroussi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18553)  

**Abstract**: Deep learning algorithms have made incredible strides in the past decade yet due to the complexity of these algorithms, the science of deep learning remains in its early stages. Being an experimentally driven field, it is natural to seek a theory of deep learning within the physics paradigm. As deep learning is largely about learning functions and distributions over functions, statistical field theory, a rich and versatile toolbox for tackling complex distributions over functions (fields) is an obvious choice of formalism. Research efforts carried out in the past few years have demonstrated the ability of field theory to provide useful insights on generalization, implicit bias, and feature learning effects. Here we provide a pedagogical review of this emerging line of research. 

**Abstract (ZH)**: 在过去十年中，深度学习算法取得了令人瞩目的进步，但由于这些算法的复杂性，深度学习的科学仍处于早期阶段。由于深度学习是一个以实验为主导的领域，自然地，人们希望在物理学框架下发展出深度学习的理论。鉴于深度学习主要涉及学习函数及其分布，统计场论作为一种适用于处理复杂函数（场）分布的强大且多功能的工具箱，成为一种自然的选择。近年来的研究表明，场论能够提供有关泛化、隐式偏见和特征学习效应的有用洞见。在这里，我们提供了一个关于这一新兴研究领域的教学性综述。 

---
# What is the Alignment Objective of GRPO? 

**Title (ZH)**: GRPO的目标对齐是什么？ 

**Authors**: Milan Vojnovic, Se-Young Yun  

**Link**: [PDF](https://arxiv.org/pdf/2502.18548)  

**Abstract**: In this note, we examine the aggregation of preferences achieved by the Group Policy Optimisation (GRPO) algorithm, a reinforcement learning method used to train advanced artificial intelligence models such as DeepSeek-R1-Zero and DeepSeekMath. The GRPO algorithm trains a policy using a reward preference model, which is computed by sampling a set of outputs for a given context, observing the corresponding rewards, and applying shift-and-scale normalisation to these reward values. Additionally, it incorporates a penalty function to discourage deviations from a reference policy.
We present a framework that enables us to characterise the stationary policies of the GRPO algorithm. This analysis reveals that the aggregation of preferences differs fundamentally from standard logarithmic pooling, which is implemented by other approaches such as RLHF. The precise form of preference aggregation arises from the way the reward preference model is defined and from the penalty function, which we show to essentially correspond to the reverse Kullback-Leibler (KL) divergence between the aggregation policy and the reference policy.
Interestingly, we demonstrate that for groups of size two, the reward preference model corresponds to pairwise comparison preferences, similar to those in other alignment methods based on pairwise comparison feedback. We provide explicit characterisations of the aggregate preference for binary questions, for groups of size two, and in the limit of large group size. This provides insights into the dependence of the aggregate preference on parameters such as the regularisation constant and the confidence margin of question answers.
Finally, we discuss the aggregation of preferences obtained by modifying the GRPO algorithm to use direct KL divergence as the penalty or to use rewards without scale normalisation. 

**Abstract (ZH)**: 在本文中，我们探讨了Group Policy Optimisation (GRPO)算法实现的偏好聚合，该算法是一种用于训练先进人工智能模型（例如DeepSeek-R1-Zero和DeepSeekMath）的强化学习方法。GRPO算法通过使用奖励偏好模型来训练策略，该模型通过为给定上下文采样一组输出，观察相应的奖励，并对这些奖励值应用偏移和缩放规范化来计算。此外，该算法还包含一个惩罚函数，以防止偏离参考策略。

我们提出了一种框架，使我们能够描述GRPO算法的稳态策略。这种分析揭示了偏好聚合方式与标准对数池化（如在RLHF等其他方法中实现的方式）存在根本不同。偏好聚合的具体形式取决于奖励偏好模型的定义方式以及惩罚函数，后者实际上对应于聚合策略与参考策略之间的反Kullback-Leibler (KL)散度。

有趣的是，我们证明，在组大小为2的情况下，奖励偏好模型对应于类似于基于成对比较反馈的其他对齐方法中的成对偏好比较。我们具体描述了二元问题的聚合偏好、组大小为2的情况，以及在组大小趋近于无穷大的情况下的聚合偏好。这为聚合偏好对正则化常数和问题答案置信度边际等参数的依赖性提供了解释。

最后，我们讨论了通过修改GRPO算法来使用直接KL散度作为惩罚或不使用缩放规范化奖励来获得的偏好聚合方式。 

---
# Steganography Beyond Space-Time With Chain of Multimodal AI Agents 

**Title (ZH)**: 时空之外的隐写术：多模态AI代理链 

**Authors**: Ching-Chun Chang, Isao Echizen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18547)  

**Abstract**: Steganography is the art and science of covert writing, with a broad range of applications interwoven within the realm of cybersecurity. As artificial intelligence continues to evolve, its ability to synthesise realistic content emerges as a threat in the hands of cybercriminals who seek to manipulate and misrepresent the truth. Such synthetic content introduces a non-trivial risk of overwriting the subtle changes made for the purpose of steganography. When the signals in both the spatial and temporal domains are vulnerable to unforeseen overwriting, it calls for reflection on what can remain invariant after all. This study proposes a paradigm in steganography for audiovisual media, where messages are concealed beyond both spatial and temporal domains. A chain of multimodal agents is developed to deconstruct audiovisual content into a cover text, embed a message within the linguistic domain, and then reconstruct the audiovisual content through synchronising both aural and visual modalities with the resultant stego text. The message is encoded by biasing the word sampling process of a language generation model and decoded by analysing the probability distribution of word choices. The accuracy of message transmission is evaluated under both zero-bit and multi-bit capacity settings. Fidelity is assessed through both biometric and semantic similarities, capturing the identities of the recorded face and voice, as well as the core ideas conveyed through the media. Secrecy is examined through statistical comparisons between cover and stego texts. Robustness is tested across various scenarios, including audiovisual compression, face-swapping, voice-cloning and their combinations. 

**Abstract (ZH)**: 隐写术是关于隐蔽书写的艺术和技术，其在网络安全领域有着广泛的应用。随着人工智能的不断发展，其生成逼真内容的能力成为网络犯罪分子操纵和歪曲事实时的一个潜在威胁。这种合成内容会增加对通过隐写术进行细微修改的信号进行意外覆盖的风险。当音频-视频空间域和时间域中的信号都可能遭受无法预见的覆盖时，这就需要我们反思在所有这些情况下，还剩下什么不变。本研究提出了一种音频-视频隐写术的新范式，在空间域和时间域之外隐藏消息。我们开发了一条多模态代理链，将音频-视频内容分解为隐藏文本，将信息嵌入到语言领域，再通过同步听觉和视觉模态与生成的隐写文本重建音频-视频内容。信息通过偏差语言生成模型的词采样过程进行编码，通过分析词汇选择的概率分布进行解码。研究在零位容量和多位容量设置下评估了信息传递的准确性。通过生物特性和语义相似度对保真度进行评估，捕捉记录的面孔和声音的身份，以及通过媒介传达的核心思想。通过统计对比隐藏文本和传输文本之间的差异来评估安全性。我们测试了该隐写术方法在多种场景下的鲁棒性，包括音频视频压缩、换脸、变声及其组合。 

---
# PII-Bench: Evaluating Query-Aware Privacy Protection Systems 

**Title (ZH)**: PII-Bench：评估查询感知的隐私保护系统 

**Authors**: Hao Shen, Zhouhong Gu, Haokai Hong, Weili Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.18545)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) has raised significant privacy concerns regarding the exposure of personally identifiable information (PII) in user prompts. To address this challenge, we propose a query-unrelated PII masking strategy and introduce PII-Bench, the first comprehensive evaluation framework for assessing privacy protection systems. PII-Bench comprises 2,842 test samples across 55 fine-grained PII categories, featuring diverse scenarios from single-subject descriptions to complex multi-party interactions. Each sample is carefully crafted with a user query, context description, and standard answer indicating query-relevant PII. Our empirical evaluation reveals that while current models perform adequately in basic PII detection, they show significant limitations in determining PII query relevance. Even state-of-the-art LLMs struggle with this task, particularly in handling complex multi-subject scenarios, indicating substantial room for improvement in achieving intelligent PII masking. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的广泛应用引起了人们对用户提示中个人可识别信息（PII）暴露的严重隐私担忧。为应对这一挑战，我们提出了一种与查询无关的PII屏蔽策略，并引入了PII-Bench，这是首个全面评估隐私保护系统的框架。PII-Bench 包含2,842个测试样本，覆盖55个细粒度的PII类别，展示了从单主体描述到复杂多方互动的各种场景。每个样本都精心构造了用户查询、背景描述和标准答案，其中包含与查询相关的PII。我们的实证评估表明，当前模型在基本PII检测方面表现尚可，但在确定查询相关PII方面存在显著局限性。即使最先进的LLMs在这一任务上也面临挑战，尤其是在处理复杂多方场景时，这表明在实现智能PII屏蔽方面仍有很大的改进空间。 

---
# MA-GTS: A Multi-Agent Framework for Solving Complex Graph Problems in Real-World Applications 

**Title (ZH)**: MA-GTS：一种用于解决实际应用中复杂图形问题的多智能体框架 

**Authors**: Zike Yuan, Ming Liu, Hui Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2502.18540)  

**Abstract**: Graph-theoretic problems arise in real-world applications like logistics, communication networks, and traffic optimization. These problems are often complex, noisy, and irregular, posing challenges for traditional algorithms. Large language models (LLMs) offer potential solutions but face challenges, including limited accuracy and input length constraints. To address these challenges, we propose MA-GTS (Multi-Agent Graph Theory Solver), a multi-agent framework that decomposes these complex problems through agent collaboration. MA-GTS maps the implicitly expressed text-based graph data into clear, structured graph representations and dynamically selects the most suitable algorithm based on problem constraints and graph structure scale. This approach ensures that the solution process remains efficient and the resulting reasoning path is interpretable. We validate MA-GTS using the G-REAL dataset, a real-world-inspired graph theory dataset we created. Experimental results show that MA-GTS outperforms state-of-the-art approaches in terms of efficiency, accuracy, and scalability, with strong results across multiple benchmarks (G-REAL 94.2%, GraCoRe 96.9%, NLGraph 98.4%).MA-GTS is open-sourced at this https URL. 

**Abstract (ZH)**: 图论问题在物流、通信网络和交通优化等实际应用中普遍存在。这些问题往往复杂、嘈杂且不规则，给传统的算法带来了挑战。大型语言模型（LLMs）提供了潜在的解决方案，但面临着准确性有限和输入长度限制等挑战。为了应对这些挑战，我们提出了一种名为MA-GTS（多智能体图理论求解器）的多智能体框架，通过智能体协作来分解这些复杂问题。MA-GTS将隐含表示的文字图数据映射为清晰的结构化图表示，并根据问题约束和图结构规模动态选择最合适的算法。这种方法确保了解决过程的高效性，并且推理路径具有可解释性。我们使用自己创建的G-REAL数据集对MA-GTS进行了验证，这是一个基于现实世界的图论数据集。实验结果表明，MA-GTS在效率、准确性和可扩展性方面均优于现有最先进的方法，在多个基准测试中表现出色（G-REAL 94.2%，GraCoRe 96.9%，NLGraph 98.4%）。MA-GTS已开源，可通过以下链接访问：[在这里插入链接]。 

---
# Revisiting Convolution Architecture in the Realm of DNA Foundation Models 

**Title (ZH)**: 在DNA基础模型领域 revisiting 卷积架构 

**Authors**: Yu Bo, Weian Mao, Yanjun Shao, Weiqiang Bai, Peng Ye, Xinzhu Ma, Junbo Zhao, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18538)  

**Abstract**: In recent years, a variety of methods based on Transformer and state space model (SSM) architectures have been proposed, advancing foundational DNA language models. However, there is a lack of comparison between these recent approaches and the classical architecture convolutional networks (CNNs) on foundation model benchmarks. This raises the question: are CNNs truly being surpassed by these recent approaches based on transformer and SSM architectures? In this paper, we develop a simple but well-designed CNN-based method termed ConvNova. ConvNova identifies and proposes three effective designs: 1) dilated convolutions, 2) gated convolutions, and 3) a dual-branch framework for gating mechanisms. Through extensive empirical experiments, we demonstrate that ConvNova significantly outperforms recent methods on more than half of the tasks across several foundation model benchmarks. For example, in histone-related tasks, ConvNova exceeds the second-best method by an average of 5.8%, while generally utilizing fewer parameters and enabling faster computation. In addition, the experiments observed findings that may be related to biological characteristics. This indicates that CNNs are still a strong competitor compared to Transformers and SSMs. We anticipate that this work will spark renewed interest in CNN-based methods for DNA foundation models. 

**Abstract (ZH)**: 近年来，基于Transformer和状态空间模型（SSM）架构的各种方法被提出，推动了DNA语言模型的基础建设。然而，这些近期的方法与经典架构卷积神经网络（CNNs）在基础模型基准上的对比研究仍然不足。这就引出了一个问题：基于Transformer和SSM架构的这些近期方法是否真的超越了成熟的CNNs？在本文中，我们开发了一种简单而精心设计的基于CNN的方法，称为ConvNova。ConvNova提出了三种有效的设计：1）扩张卷积，2）门控卷积，以及3）双分支框架来实现门控机制。通过广泛的实验研究，我们证明了在多个基础模型基准上的多项任务中，ConvNova相较于近期方法显著优于其中一半以上的方法。例如，在与组蛋白相关的任务中，ConvNova的平均性能比第二优方法高出5.8%，同时参数量较少且计算速度更快。此外，实验还观察到了可能与生物特性相关的现象，这表明CNNs仍然是与Transformer和SSMs相匹敌的强有力竞争对手。我们认为这项工作将重新激发对基于CNN的方法在DNA基础模型中的兴趣。 

---
# A Survey of Zero-Knowledge Proof Based Verifiable Machine Learning 

**Title (ZH)**: 基于零知识证明的可验证机器学习综述 

**Authors**: Zhizhi Peng, Taotao Wang, Chonghe Zhao, Guofu Liao, Zibin Lin, Yifeng Liu, Bin Cao, Long Shi, Qing Yang, Shengli Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18535)  

**Abstract**: As machine learning technologies advance rapidly across various domains, concerns over data privacy and model security have grown significantly. These challenges are particularly pronounced when models are trained and deployed on cloud platforms or third-party servers due to the computational resource limitations of users' end devices. In response, zero-knowledge proof (ZKP) technology has emerged as a promising solution, enabling effective validation of model performance and authenticity in both training and inference processes without disclosing sensitive data. Thus, ZKP ensures the verifiability and security of machine learning models, making it a valuable tool for privacy-preserving AI. Although some research has explored the verifiable machine learning solutions that exploit ZKP, a comprehensive survey and summary of these efforts remain absent. This survey paper aims to bridge this gap by reviewing and analyzing all the existing Zero-Knowledge Machine Learning (ZKML) research from June 2017 to December 2024. We begin by introducing the concept of ZKML and outlining its ZKP algorithmic setups under three key categories: verifiable training, verifiable inference, and verifiable testing. Next, we provide a comprehensive categorization of existing ZKML research within these categories and analyze the works in detail. Furthermore, we explore the implementation challenges faced in this field and discuss the improvement works to address these obstacles. Additionally, we highlight several commercial applications of ZKML technology. Finally, we propose promising directions for future advancements in this domain. 

**Abstract (ZH)**: 随着机器学习技术在各个领域的快速发展，数据隐私和模型安全方面的担忧也显著增加。特别是在使用云平台或第三方服务器进行模型训练和部署时，用户终端设备的计算资源限制使这些挑战尤为突出。为应对这一问题，零知识证明（ZKP）技术作为一种有前景的解决方案应运而生，它能够在不泄露敏感数据的情况下，有效地验证模型的性能和真实性，这不仅确保了机器学习模型的可验证性和安全性，而且使其成为保护隐私的人工智能工具。尽管已有部分研究探索了利用ZKP实现可验证机器学习的解决方案，但对这些努力的全面调研和总结仍然缺失。本文旨在填补这一空白，通过回顾和分析2017年6月至2024年12月期间的所有零知识机器学习（ZKML）研究来开展这项调查性研究。首先，我们介绍了ZKML的概念，并按照三个关键类别概述了其ZKP算法设置：可验证训练、可验证推断和可验证测试。接着，我们对这些类别中的现有ZKML研究进行了全面分类，并详细分析了这些研究工作。此外，我们探讨了该领域实施中的挑战，并讨论了改进工作以解决这些障碍。同时，我们还强调了几种ZKML技术的商业应用案例。最后，我们提出了该领域未来发展方向的建议。 

---
# MAFE: Multi-Agent Fair Environments for Decision-Making Systems 

**Title (ZH)**: MAFE：多智能体公平环境决策系统 

**Authors**: Zachary McBride Lazri, Anirudh Nakra, Ivan Brugere, Danial Dervovic, Antigoni Polychroniadou, Furong Huang, Dana Dachman-Soled, Min Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18534)  

**Abstract**: Fairness constraints applied to machine learning (ML) models in static contexts have been shown to potentially produce adverse outcomes among demographic groups over time. To address this issue, emerging research focuses on creating fair solutions that persist over time. While many approaches treat this as a single-agent decision-making problem, real-world systems often consist of multiple interacting entities that influence outcomes. Explicitly modeling these entities as agents enables more flexible analysis of their interventions and the effects they have on a system's underlying dynamics. A significant challenge in conducting research on multi-agent systems is the lack of realistic environments that leverage the limited real-world data available for analysis. To address this gap, we introduce the concept of a Multi-Agent Fair Environment (MAFE) and present and analyze three MAFEs that model distinct social systems. Experimental results demonstrate the utility of our MAFEs as testbeds for developing multi-agent fair algorithms. 

**Abstract (ZH)**: 在静态背景下应用到机器学习（ML）模型的公平性约束已被证明可能随着时间对不同人口群体产生不利影响。为了解决这个问题，新兴的研究侧重于创造能够持久保持公平性的解决方案。虽然许多方法将其视为单代理决策问题，但现实世界的系统通常由多个相互影响的实体组成，这些实体会影响结果。将这些实体明确建模为代理可以更灵活地分析它们的干预及其对系统潜在动态的影响。在多代理系统研究中，一个显著的挑战是缺乏能够利用可用于分析的有限真实世界数据的现实环境。为了解决这一差距，我们引入了多代理公平环境（MAFE）的概念，并介绍了并分析了三种模拟不同社会系统的MAFE。实验结果表明，我们的MAFE为开发多代理公平算法提供了有用的测试平台。 

---
# Heterogeneous Decision Making in Mixed Traffic: Uncertainty-aware Planning and Bounded Rationality 

**Title (ZH)**: 混合交通中的异质性决策：不确定性感知规划与有限理性 

**Authors**: Hang Wang, Qiaoyi Fang, Junshan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18529)  

**Abstract**: The past few years have witnessed a rapid growth of the deployment of automated vehicles (AVs). Clearly, AVs and human-driven vehicles (HVs) will co-exist for many years, and AVs will have to operate around HVs, pedestrians, cyclists, and more, calling for fundamental breakthroughs in AI designed for mixed traffic to achieve mixed autonomy. Thus motivated, we study heterogeneous decision making by AVs and HVs in a mixed traffic environment, aiming to capture the interactions between human and machine decision-making and develop an AI foundation that enables vehicles to operate safely and efficiently. There are a number of challenges to achieve mixed autonomy, including 1) humans drivers make driving decisions with bounded rationality, and it remains open to develop accurate models for HVs' decision making; and 2) uncertainty-aware planning plays a critical role for AVs to take safety maneuvers in response to the human behavior. In this paper, we introduce a formulation of AV-HV interaction, where the HV makes decisions with bounded rationality and the AV employs uncertainty-aware planning based on the prediction on HV's future actions. We conduct a comprehensive analysis on AV and HV's learning regret to answer the questions: 1) {How does the learning performance depend on HV's bounded rationality and AV's planning}; 2) {How do different decision making strategies impact the overall learning performance}? Our findings reveal some intriguing phenomena, such as Goodhart's Law in AV's learning performance and compounding effects in HV's decision making process. By examining the dynamics of the regrets, we gain insights into the interplay between human and machine decision making. 

**Abstract (ZH)**: 近年来，自动驾驶车辆（AVs）的部署迅速增长。显然，AVs 和人类驾驶车辆（HVs）将在相当长的一段时间内共存，且AVs 将不得不在混合交通环境中与HV、行人和骑行者同时运行，这要求我们在设计针对混合交通环境的AI方面取得根本性的突破，以实现混合自主控制。受到这一目标的启发，我们研究了在混合交通环境中AVs 和HV之间异构决策问题，旨在捕捉人类和机器决策之间的交互，并开发一种使车辆能够安全高效运行的AI基础。实现混合自主控制有许多挑战，包括1)人类驾驶员在有限理性下做出驾驶决策，目前尚无准确模型来描述HV的决策过程；2)不确定性意识规划在AVs根据人类行为采取安全措施时发挥关键作用。在本文中，我们引入了一种AV-HV交互的建模形式，其中HV在有限理性的条件下做出决策，而AV则基于对HV未来行动的预测采用不确定性意识规划。我们对AV和HV的学习遗憾进行了全面分析，以回答以下问题：1)学习性能如何依赖于HV的有限理性和AV的规划；2)不同的决策策略如何影响整体学习性能？我们的研究发现揭示了一些有趣的模式，如AV学习性能中的Goodhart定律和HV决策过程中累积效应。通过分析遗憾的动力学，我们对人类和机器决策之间的相互作用有了新的认识。 

---
# ARACNE: An LLM-Based Autonomous Shell Pentesting Agent 

**Title (ZH)**: ARACNE：基于LLM的自主 Shell 渗透测试代理 

**Authors**: Tomas Nieponice, Veronica Valeros, Sebastian Garcia  

**Link**: [PDF](https://arxiv.org/pdf/2502.18528)  

**Abstract**: We introduce ARACNE, a fully autonomous LLM-based pentesting agent tailored for SSH services that can execute commands on real Linux shell systems. Introduces a new agent architecture with multi-LLM model support. Experiments show that ARACNE can reach a 60\% success rate against the autonomous defender ShelLM and a 57.58\% success rate against the Over The Wire Bandit CTF challenges, improving over the state-of-the-art. When winning, the average number of actions taken by the agent to accomplish the goals was less than 5. The results show that the use of multi-LLM is a promising approach to increase accuracy in the actions. 

**Abstract (ZH)**: 我们介绍了ARACNE，这是一种专为SSH服务设计的全自动LLM（大型语言模型）基渗透测试代理，能够执行真实的LinuxShell系统命令。ARACNE引入了一种新的代理架构，支持多LLM模型。实验结果显示，ARACNE在对抗自主防御者ShelLM时的成功率为60%，在对抗OVER THE WIRE Bandit CTF挑战时的成功率为57.58%，均优于现有最先进的方法。当获胜时，代理完成目标所需的平均行动次数少于5次。结果表明，多LLM的使用是提高动作准确性的有前景的方法。 

---
# GOD model: Privacy Preserved AI School for Personal Assistant 

**Title (ZH)**: GOD模型：保留隐私的人工智能个人助手学校 

**Authors**: PIN AI Team, Bill Qingyun Sun, Laura Florescu, Boliang Zhang, Regan Peng, Smile Hu, Shouqiao Wang, Ben Wu, Xi Wang, Davide Crapis, Gavin Zhen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.18527)  

**Abstract**: Personal AI assistants (e.g., Apple Intelligence, Meta AI) offer proactive recommendations that simplify everyday tasks, but their reliance on sensitive user data raises concerns about privacy and trust. To address these challenges, we introduce the Guardian of Data (GOD), a secure, privacy-preserving framework for training and evaluating AI assistants directly on-device. Unlike traditional benchmarks, the GOD model measures how well assistants can anticipate user needs-such as suggesting gifts-while protecting user data and autonomy. Functioning like an AI school, it addresses the cold start problem by simulating user queries and employing a curriculum-based approach to refine the performance of each assistant. Running within a Trusted Execution Environment (TEE), it safeguards user data while applying reinforcement and imitation learning to refine AI recommendations. A token-based incentive system encourages users to share data securely, creating a data flywheel that drives continuous improvement. By integrating privacy, personalization, and trust, the GOD model provides a scalable, responsible path for advancing personal AI assistants. For community collaboration, part of the framework is open-sourced at this https URL. 

**Abstract (ZH)**: 个人AI助手（例如Apple Intelligence、Meta AI）提供主动推荐，简化日常任务，但它们对敏感用户数据的依赖引发了隐私和信任方面的担忧。为应对这些挑战，我们引入了一种名为Guardian of Data（GOD）的安全、隐私保护框架，用于直接在设备上训练和评估AI助手。与传统的基准测试不同，GOD模型评估助手预测用户需求（例如推荐礼物）的能力，同时保护用户数据和自主权。该框架像一所AI学校，通过模拟用户查询并采用基于 curriculum 的方法来优化每个助手的性能，解决了冷启动问题。运行在受信任的执行环境中（TEE），该框架保障了用户数据的安全性，同时利用强化学习和模仿学习来优化AI建议。基于代币的激励机制鼓励用户安全地共享数据，从而形成一个数据飞轮，推动持续改进。通过整合隐私、个性化和信任，GOD模型提供了一条可扩展且负责任的路径，促进个人AI助手的发展。为促进社区合作，部分框架在以下地址开源：[该 https URL]。 

---
# Reinforcement Learning-based Approach for Vehicle-to-Building Charging with Heterogeneous Agents and Long Term Rewards 

**Title (ZH)**: 基于强化学习的方法：考虑异质代理人和长期奖励的车辆到建筑充电策略 

**Authors**: Fangqi Liu, Rishav Sen, Jose Paolo Talusan, Ava Pettet, Aaron Kandel, Yoshinori Suzue, Ayan Mukhopadhyay, Abhishek Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2502.18526)  

**Abstract**: Strategic aggregation of electric vehicle batteries as energy reservoirs can optimize power grid demand, benefiting smart and connected communities, especially large office buildings that offer workplace charging. This involves optimizing charging and discharging to reduce peak energy costs and net peak demand, monitored over extended periods (e.g., a month), which involves making sequential decisions under uncertainty and delayed and sparse rewards, a continuous action space, and the complexity of ensuring generalization across diverse conditions. Existing algorithmic approaches, e.g., heuristic-based strategies, fall short in addressing real-time decision-making under dynamic conditions, and traditional reinforcement learning (RL) models struggle with large state-action spaces, multi-agent settings, and the need for long-term reward optimization. To address these challenges, we introduce a novel RL framework that combines the Deep Deterministic Policy Gradient approach (DDPG) with action masking and efficient MILP-driven policy guidance. Our approach balances the exploration of continuous action spaces to meet user charging demands. Using real-world data from a major electric vehicle manufacturer, we show that our approach comprehensively outperforms many well-established baselines and several scalable heuristic approaches, achieving significant cost savings while meeting all charging requirements. Our results show that the proposed approach is one of the first scalable and general approaches to solving the V2B energy management challenge. 

**Abstract (ZH)**: 将电动汽车电池作为能量存储进行战略性聚合可以优化电网需求，使智能互联社区受益，特别是提供工作场所充电的大型办公楼。这涉及到在长时间段内（例如一个月）优化充电和放电，以减少峰值能源成本和净峰值需求，这需要在不确定性条件下做出 Sequential 决策，并考虑延迟和稀疏奖励、连续动作空间以及确保在多种条件下的泛化复杂性。现有的算法方法，例如基于启发式策略，难以应对动态条件下的实时决策，而传统的强化学习（RL）模型则难以处理庞大的状态-动作空间、多智能体设置以及长期奖励优化的需求。为了解决这些问题，我们提出了一种新的 RL 框架，结合了深度确定性策略梯度方法（DDPG）、动作遮掩以及高效的 MILP 驱动策略指导。我们的方法能够在满足用户充电需求的同时探索连续的动作空间。通过使用一家主要电动汽车制造商的真实数据，我们证明了我们的方法在各个方面都全面优于许多已建立的基准方法和几种可扩展的启发式方法，实现了显著的成本节约，同时满足所有充电要求。我们的结果表明，所提出的方法是解决 V2B 能源管理挑战的首批可扩展和通用方法之一。 

---
# End-to-End Deep Learning for Structural Brain Imaging: A Unified Framework 

**Title (ZH)**: 端到端深度学习在结构脑成像中的应用：统一框架 

**Authors**: Yao Su, Keqi Han, Mingjie Zeng, Lichao Sun, Liang Zhan, Carl Yang, Lifang He, Xiangnan Kong  

**Link**: [PDF](https://arxiv.org/pdf/2502.18523)  

**Abstract**: Brain imaging analysis is fundamental in neuroscience, providing valuable insights into brain structure and function. Traditional workflows follow a sequential pipeline-brain extraction, registration, segmentation, parcellation, network generation, and classification-treating each step as an independent task. These methods rely heavily on task-specific training data and expert intervention to correct intermediate errors, making them particularly burdensome for high-dimensional neuroimaging data, where annotations and quality control are costly and time-consuming. We introduce UniBrain, a unified end-to-end framework that integrates all processing steps into a single optimization process, allowing tasks to interact and refine each other. Unlike traditional approaches that require extensive task-specific annotations, UniBrain operates with minimal supervision, leveraging only low-cost labels (i.e., classification and extraction) and a single labeled atlas. By jointly optimizing extraction, registration, segmentation, parcellation, network generation, and classification, UniBrain enhances both accuracy and computational efficiency while significantly reducing annotation effort. Experimental results demonstrate its superiority over existing methods across multiple tasks, offering a more scalable and reliable solution for neuroimaging analysis. Our code and data can be found at this https URL 

**Abstract (ZH)**: 脑成像分析是神经科学中的基础工具，为深入了解大脑结构和功能提供了宝贵见解。传统的分析工作流遵循一个顺序管道——脑部提取、注册、分割、分段、网络生成和分类，将每个步骤视为独立的任务。这些方法高度依赖于特定任务的训练数据和专家干预以纠正中间错误，特别是在高维度神经成像数据中，注释和质量控制极为昂贵且耗时。我们提出了UniBrain，这是一种统一的端到端框架，将所有处理步骤整合到单一的优化过程中，使得任务之间能够相互交互和相互校正。与传统方法需要大量特定任务的注释不同，UniBrain只需最少的监督，利用低成本标签（即分类和提取）以及单一标注的解剖图谱作为输入。通过联合优化提取、注册、分割、分段、网络生成和分类，UniBrain提高了准确性和计算效率，显著减少了标注工作量。实验结果表明，UniBrain在多个任务上优于现有方法，为神经成像分析提供了更可扩展和可靠的选择。我们的代码和数据可以在以下网址找到：[请填写网址] 

---
# Class-Conditional Neural Polarizer: A Lightweight and Effective Backdoor Defense by Purifying Poisoned Features 

**Title (ZH)**: 基于类条件神经去极化的轻量级有效后门防御方法：净化中毒特征 

**Authors**: Mingli Zhu, Shaokui Wei, Hongyuan Zha, Baoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18520)  

**Abstract**: Recent studies have highlighted the vulnerability of deep neural networks to backdoor attacks, where models are manipulated to rely on embedded triggers within poisoned samples, despite the presence of both benign and trigger information. While several defense methods have been proposed, they often struggle to balance backdoor mitigation with maintaining benign this http URL this work, inspired by the concept of optical polarizer-which allows light waves of specific polarizations to pass while filtering others-we propose a lightweight backdoor defense approach, NPD. This method integrates a neural polarizer (NP) as an intermediate layer within the compromised model, implemented as a lightweight linear transformation optimized via bi-level optimization. The learnable NP filters trigger information from poisoned samples while preserving benign content. Despite its effectiveness, we identify through empirical studies that NPD's performance degrades when the target labels (required for purification) are inaccurately estimated. To address this limitation while harnessing the potential of targeted adversarial mitigation, we propose class-conditional neural polarizer-based defense (CNPD). The key innovation is a fusion module that integrates the backdoored model's predicted label with the features to be purified. This architecture inherently mimics targeted adversarial defense mechanisms without requiring label estimation used in NPD. We propose three implementations of CNPD: the first is r-CNPD, which trains a replicated NP layer for each class and, during inference, selects the appropriate NP layer for defense based on the predicted class from the backdoored model. To efficiently handle a large number of classes, two variants are designed: e-CNPD, which embeds class information as additional features, and a-CNPD, which directs network attention using class information. 

**Abstract (ZH)**: 近年来，关于深度神经网络对后门攻击的脆弱性的研究引起了广泛的关注。在这些攻击中，模型被操纵使其依赖于受污染样本中嵌入的触发器，即使存在触发信息和无害信息。尽管已经提出了多种防御方法，但它们往往难以在抑制后门攻击的同时保持模型对无害输入的性能。基于这一背景，本研究受到光学偏振器概念的启发（允许特定极化状态的光波通过，同时过滤其他极化状态的光波），提出了一种轻量级后门防御方法——NPD。该方法在受损模型中引入了一个神经偏振器（NP）作为中间层，该偏振器通过二阶优化实现轻量级线性变换并可学习。可学习的NP过滤受污染样本中的触发信息，同时保留无害内容。尽管NPD具有显著效果，但实验证实其性能在目标标签（用于净化过程）不准确估计时会下降。为解决这一局限性并充分利用定向对抗防御的潜力，本文提出了条件神经偏振器基于防御（CNPD）。关键创新在于融合模块，该模块将后门模型预测的标签与待净化特征相结合。该架构能够自然模仿定向对抗防御机制，而无需使用NPD中的标签估计。本文提出了CNPD的三种实现方案：第一种是r-CNPD，其训练每个类别的复制NP层，并在推理过程中根据后门模型预测的类别选择适当的NP层进行防御。为高效处理大量类别，我们设计了两种变体：e-CNPD，其将类别信息作为附加特征嵌入；a-CNPD，其使用类别信息引导网络注意力。 

---
# FreeTumor: Large-Scale Generative Tumor Synthesis in Computed Tomography Images for Improving Tumor Recognition 

**Title (ZH)**: FreeTumor：计算机断层扫描图像中大规模生成性肿瘤合成以提高肿瘤识别效果 

**Authors**: Linshan Wu, Jiaxin Zhuang, Yanning Zhou, Sunan He, Jiabo Ma, Luyang Luo, Xi Wang, Xuefeng Ni, Xiaoling Zhong, Mingxiang Wu, Yinghua Zhao, Xiaohui Duan, Varut Vardhanabhuti, Pranav Rajpurkar, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18519)  

**Abstract**: Tumor is a leading cause of death worldwide, with an estimated 10 million deaths attributed to tumor-related diseases every year. AI-driven tumor recognition unlocks new possibilities for more precise and intelligent tumor screening and diagnosis. However, the progress is heavily hampered by the scarcity of annotated datasets, which demands extensive annotation efforts by radiologists. To tackle this challenge, we introduce FreeTumor, an innovative Generative AI (GAI) framework to enable large-scale tumor synthesis for mitigating data scarcity. Specifically, FreeTumor effectively leverages a combination of limited labeled data and large-scale unlabeled data for tumor synthesis training. Unleashing the power of large-scale data, FreeTumor is capable of synthesizing a large number of realistic tumors on images for augmenting training datasets. To this end, we create the largest training dataset for tumor synthesis and recognition by curating 161,310 publicly available Computed Tomography (CT) volumes from 33 sources, with only 2.3% containing annotated tumors. To validate the fidelity of synthetic tumors, we engaged 13 board-certified radiologists in a Visual Turing Test to discern between synthetic and real tumors. Rigorous clinician evaluation validates the high quality of our synthetic tumors, as they achieved only 51.1% sensitivity and 60.8% accuracy in distinguishing our synthetic tumors from real ones. Through high-quality tumor synthesis, FreeTumor scales up the recognition training datasets by over 40 times, showcasing a notable superiority over state-of-the-art AI methods including various synthesis methods and foundation models. These findings indicate promising prospects of FreeTumor in clinical applications, potentially advancing tumor treatments and improving the survival rates of patients. 

**Abstract (ZH)**: 肿瘤是全球导致死亡的主要原因之一，每年约有1000万人因为肿瘤相关疾病去世。基于人工智能的肿瘤识别技术为更精确和智能的肿瘤筛查与诊断提供了新的可能性。然而，这一进展受到标注数据稀缺性的严重限制，这需要放射科医生投入大量的标注工作。为应对这一挑战，我们提出了FreeTumor，一种创新的生成式人工智能（GAI）框架，以缓解数据稀缺问题。具体而言，FreeTumor有效地结合了有限的标注数据和大规模的未标注数据进行肿瘤合成训练。凭借大规模数据的力量，FreeTumor能够生成大量真实感强的肿瘤图像，从而增强训练数据集。为此，我们创建了迄今最大的肿瘤合成和识别训练数据集，通过整理来自33个来源的161,310个公开的计算机断层扫描（CT）体积，其中仅有2.3%包含标注的肿瘤。为了验证合成肿瘤的真实性，我们邀请了13名经过认证的放射科医生进行视觉图灵测试，以区分合成肿瘤与真实肿瘤。严格的临床评估证明了我们合成肿瘤的高质量，它们在区分合成肿瘤与真实肿瘤方面的敏感性和准确性分别仅为51.1%和60.8%。通过高质量的肿瘤合成，FreeTumor将识别训练数据集扩展了40多倍，展示了在各种生成方法和基础模型等先进AI方法方面的显著优越性。这些发现表明FreeTumor在临床应用中具有广阔的前景，有可能推动肿瘤治疗的发展并提高患者的生存率。 

---
# Swallowing the Poison Pills: Insights from Vulnerability Disparity Among LLMs 

**Title (ZH)**: 吞咽毒丸：来自LLMs之间漏洞差异的见解

注释：这里的“毒丸”（Poison Pill）借用的是其比喻含义，指那些可能带来负面影响的事物。在LLM（大型语言模型）的上下文中，这是用来比喻模型中存在的漏洞或安全性问题。同时，“吞咽毒丸”这一表达形象地说明了模型在面对自身存在的漏洞时可能采取的应对方式。 

**Authors**: Peng Yifeng, Wu Zhizheng, Chen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18518)  

**Abstract**: Modern large language models (LLMs) exhibit critical vulnerabilities to poison pill attacks: localized data poisoning that alters specific factual knowledge while preserving overall model utility. We systematically demonstrate these attacks exploit inherent architectural properties of LLMs, achieving 54.6% increased retrieval inaccuracy on long-tail knowledge versus dominant topics and up to 25.5% increase retrieval inaccuracy on compressed models versus original architectures. Through controlled mutations (e.g., temporal/spatial/entity alterations) and, our method induces localized memorization deterioration with negligible impact on models' performance on regular standard benchmarks (e.g., <2% performance drop on MMLU/GPQA), leading to potential detection evasion. Our findings suggest: (1) Disproportionate vulnerability in long-tail knowledge may result from reduced parameter redundancy; (2) Model compression may increase attack surfaces, with pruned/distilled models requiring 30% fewer poison samples for equivalent damage; (3) Associative memory enables both spread of collateral damage to related concepts and amplification of damage from simultaneous attack, particularly for dominant topics. These findings raise concerns over current scaling paradigms since attack costs are lowering while defense complexity is rising. Our work establishes poison pills as both a security threat and diagnostic tool, revealing critical security-efficiency trade-offs in language model compression that challenges prevailing safety assumptions. 

**Abstract (ZH)**: 现代大型语言模型（LLMs）对“毒丸攻击”表现出关键性的脆弱性：这种局部数据污染会改变特定的 factual 知识，同时保持整体模型的功能。我们系统地证明了这些攻击利用了 LLMs 内在的架构特性，在长尾知识上的检索准确性提高了 54.6%，而在压缩模型上的检索准确性提高了最多 25.5%。通过受控的变异（例如，时间/空间/实体的改动），我们的方法导致了局部记忆的退化，同时对模型在常规标准基准上的性能影响很小（例如，在 MMLU/GPQA 中的性能下降不到 2%），从而可能导致潜在的检测规避。我们的发现表明：（1）长尾知识中不成比例的脆弱性可能源于参数冗余度的减少；（2）模型压缩可能增加攻击面，剪枝/精简模型需要更少的毒样本（少 30%）以达到等效的破坏；（3）关联记忆不仅能够将损害传播到相关概念，还能够放大同时攻击带来的损害，特别是在处理主流主题时。这些发现对当前的扩展范式提出了担忧，因为攻击成本在下降，而防御的复杂度在上升。我们的研究将“毒丸攻击”确立为一种安全威胁和诊断工具，揭示了语言模型压缩中的关键安全与效率权衡，这挑战了当前普遍存在的安全性假设。 

---
# RewardDS: Privacy-Preserving Fine-Tuning for Large Language Models via Reward Driven Data Synthesis 

**Title (ZH)**: RewardDS：通过奖励驱动的数据合成实现大型语言模型的隐私保护微调 

**Authors**: Jianwei Wang, Junyao Yang, Haoran Li, Huiping Zhuang, Cen Chen, Ziqian Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18517)  

**Abstract**: The success of large language models (LLMs) has attracted many individuals to fine-tune them for domain-specific tasks by uploading their data. However, in sensitive areas like healthcare and finance, privacy concerns often arise. One promising solution is to sample synthetic data with Differential Privacy (DP) guarantees to replace private data. However, these synthetic data contain significant flawed data, which are considered as noise. Existing solutions typically rely on naive filtering by comparing ROUGE-L scores or embedding similarities, which are ineffective in addressing the noise. To address this issue, we propose RewardDS, a novel privacy-preserving framework that fine-tunes a reward proxy model and uses reward signals to guide the synthetic data generation. Our RewardDS introduces two key modules, Reward Guided Filtering and Self-Optimizing Refinement, to both filter and refine the synthetic data, effectively mitigating the noise. Extensive experiments across medical, financial, and code generation domains demonstrate the effectiveness of our method. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的成功吸引了许多个人通过上传其数据来针对特定领域进行微调。然而，在医疗和金融等敏感领域，隐私问题往往随之而来。一种有前景的解决方案是使用差分隐私（DP）保证的合成数据来替代私人数据。然而，这些合成数据中包含大量被视为噪声的错误数据。现有的解决方案通常依赖于比较ROUGE-L分数或嵌入相似性进行简单的过滤，这在消除噪声方面效果不佳。为了解决这个问题，我们提出了RewardDS，这是一种新颖的隐私保护框架，通过微调一个奖励代理模型并使用奖励信号来指导合成数据的生成。我们的RewardDS引入了两个关键模块——奖励引导过滤和自我优化精炼，既能过滤又能精炼合成数据，有效地减少了噪声。在医疗、金融和代码生成等多个领域的广泛实验中，验证了我们方法的有效性。 

---
# A Multi-Agent Framework for Automated Vulnerability Detection and Repair in Solidity and Move Smart Contracts 

**Title (ZH)**: 一种基于多代理系统的Solidity和Move智能合约自动化漏洞检测与修复框架 

**Authors**: Rabimba Karanjai, Sam Blackshear, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18515)  

**Abstract**: The rapid growth of the blockchain ecosystem and the increasing value locked in smart contracts necessitate robust security measures. While languages like Solidity and Move aim to improve smart contract security, vulnerabilities persist. This paper presents Smartify, a novel multi-agent framework leveraging Large Language Models (LLMs) to automatically detect and repair vulnerabilities in Solidity and Move smart contracts. Unlike traditional methods that rely solely on vast pre-training datasets, Smartify employs a team of specialized agents working on different specially fine-tuned LLMs to analyze code based on underlying programming concepts and language-specific security principles. We evaluated Smartify on a dataset for Solidity and a curated dataset for Move, demonstrating its effectiveness in fixing a wide range of vulnerabilities. Our results show that Smartify (Gemma2+codegemma) achieves state-of-the-art performance, surpassing existing LLMs and enhancing general-purpose models' capabilities, such as Llama 3.1. Notably, Smartify can incorporate language-specific knowledge, such as the nuances of Move, without requiring massive language-specific pre-training datasets. This work offers a detailed analysis of various LLMs' performance on smart contract repair, highlighting the strengths of our multi-agent approach and providing a blueprint for developing more secure and reliable decentralized applications in the growing blockchain landscape. We also provide a detailed recipe for extending this to other similar use cases. 

**Abstract (ZH)**: 区块链生态系统的迅速发展和智能合约中锁定价值的不断增加，迫切需要更为稳健的安全措施。尽管像Solidity和Move这样的编程语言旨在改进智能合约的安全性，但仍然存在漏洞。本文提出了一种名为Smartify的新型多代理框架，该框架利用大型语言模型（LLMs）自动检测和修复Solidity和Move智能合约中的漏洞。不同于传统的依赖于广泛预训练数据集的方法，Smartify采用一个专门化的代理团队，各自使用针对特定编程概念和语言特定安全原则微调过的LLMs来分析代码。我们通过对Solidity和Move的精心标注数据集进行了评估，展示了Smartify在多种漏洞修复方面的有效性。我们的结果显示，Smartify（Gemma2+codegemma）在性能上达到了最先进的水平，超越了现有的语言模型并增强了通用模型的能力，如Llama 3.1。值得注意的是，Smartify可以融入语言特定的知识，例如Move的细微差别，而无需进行大规模的语言特定预训练数据集。本文详细分析了多种语言模型在智能合约修复方面的性能，突显了我们多代理方法的优势，并为开发更安全可靠的去中心化应用程序提供了蓝图，特别是在不断增长的区块链领域。我们还提供了一种详细的配方，以扩展到其他类似的应用场景。 

---
# FCoT-VL:Advancing Text-oriented Large Vision-Language Models with Efficient Visual Token Compression 

**Title (ZH)**: FCoT-VL：高效视觉令牌压缩促进面向文本的大规模视觉语言模型发展 

**Authors**: Jianjian Li, Junquan Fan, Feng Tang, Gang Huang, Shitao Zhu, Songlin Liu, Nian Xie, Wulong Liu, Yong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18512)  

**Abstract**: The rapid success of Vision Large Language Models (VLLMs) often depends on the high-resolution images with abundant visual tokens, which hinders training and deployment efficiency. Current training-free visual token compression methods exhibit serious performance degradation in tasks involving high-resolution, text-oriented image understanding and reasoning. In this paper, we propose an efficient visual token compression framework for text-oriented VLLMs in high-resolution scenarios. In particular, we employ a light-weight self-distillation pre-training stage to compress the visual tokens, requiring a limited numbers of image-text pairs and minimal learnable parameters. Afterwards, to mitigate potential performance degradation of token-compressed models, we construct a high-quality post-train stage. To validate the effectiveness of our method, we apply it to an advanced VLLMs, InternVL2. Experimental results show that our approach significantly reduces computational overhead while outperforming the baselines across a range of text-oriented benchmarks. We will release the models and code soon. 

**Abstract (ZH)**: 视觉大语言模型（VLLMs）的快速成功往往依赖于高分辨率图像和丰富的视觉标记，这阻碍了训练和部署效率。当前的无需训练的视觉标记压缩方法在涉及高分辨率和文本导向图像理解与推理的任务中表现出严重的性能退化。在本文中，我们提出了一种针对高分辨率场景中的文本导向VLLMs的高效视觉标记压缩框架。特别是，我们采用了一个轻量级的自我精炼预训练阶段来压缩视觉标记，仅需少量的图像-文本对和最少的学习参数。随后，为了缓解标记压缩模型潜在的性能退化，我们构建了一个高质量的后训练阶段。为了验证我们方法的有效性，我们将该方法应用到了先进的VLLMs——InternVL2中。实验结果表明，我们的方法显著减少了计算开销，并在多种文本导向基准测试中优于基线方法。我们将很快发布模型和代码。 

---
# ELBA-Bench: An Efficient Learning Backdoor Attacks Benchmark for Large Language Models 

**Title (ZH)**: ELBA-Bench：一种面向大型语言模型的有效后门攻击基准测试 

**Authors**: Xuxu Liu, Siyuan Liang, Mengya Han, Yong Luo, Aishan Liu, Xiantao Cai, Zheng He, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18511)  

**Abstract**: Generative large language models are crucial in natural language processing, but they are vulnerable to backdoor attacks, where subtle triggers compromise their behavior. Although backdoor attacks against LLMs are constantly emerging, existing benchmarks remain limited in terms of sufficient coverage of attack, metric system integrity, backdoor attack alignment. And existing pre-trained backdoor attacks are idealized in practice due to resource access constraints. Therefore we establish $\textit{ELBA-Bench}$, a comprehensive and unified framework that allows attackers to inject backdoor through parameter efficient fine-tuning ($\textit{e.g.,}$ LoRA) or without fine-tuning techniques ($\textit{e.g.,}$ In-context-learning). $\textit{ELBA-Bench}$ provides over 1300 experiments encompassing the implementations of 12 attack methods, 18 datasets, and 12 LLMs. Extensive experiments provide new invaluable findings into the strengths and limitations of various attack strategies. For instance, PEFT attack consistently outperform without fine-tuning approaches in classification tasks while showing strong cross-dataset generalization with optimized triggers boosting robustness; Task-relevant backdoor optimization techniques or attack prompts along with clean and adversarial demonstrations can enhance backdoor attack success while preserving model performance on clean samples. Additionally, we introduce a universal toolbox designed for standardized backdoor attack research, with the goal of propelling further progress in this vital area. 

**Abstract (ZH)**: 生成式大型语言模型在自然语言处理中至关重要，但它们容易受到后门攻击的威胁，其中微妙的触发器可能会改变其行为。尽管针对大型语言模型（LLMs）的后门攻击持续涌现，但现有基准在攻击覆盖范围、度量系统完整性以及后门攻击对齐等方面仍然有限。此外，由于资源访问限制，现有的预训练后门攻击在实际应用中往往是理想化的。因此，我们建立了一个全面且统一的框架——$\textit{ELBA-Bench}$，允许攻击者通过参数高效的微调（例如LoRA）或不使用微调技术（例如上下文学习）来注入后门。$\textit{ELBA-Bench}$ 包含了超过1300个实验，涵盖了12种攻击方法、18个数据集和12个大型语言模型的实现。广泛的实验提供了关于各种攻击策略优点和局限性的新见解。例如，在分类任务中，PEFT攻击在没有微调的情况下表现一致地更好，并且通过优化触发器提高了鲁棒性，实现了跨数据集的强大泛化表现；任务相关后门优化技术或攻击提示，以及干净和对抗性示例，可以增强后门攻击的成功率，同时保持模型在干净样本上的性能。此外，我们还引入了一个通用工具箱，旨在标准化后门攻击研究，从而推动这一关键领域进一步的发展。 

---
# Protecting Users From Themselves: Safeguarding Contextual Privacy in Interactions with Conversational Agents 

**Title (ZH)**: 保护用户免受自身风险：在与对话代理交互中 safeguarding 上下文隐私 

**Authors**: Ivoline Ngong, Swanand Kadhe, Hao Wang, Keerthiram Murugesan, Justin D. Weisz, Amit Dhurandhar, Karthikeyan Natesan Ramamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2502.18509)  

**Abstract**: Conversational agents are increasingly woven into individuals' personal lives, yet users often underestimate the privacy risks involved. The moment users share information with these agents (e.g., LLMs), their private information becomes vulnerable to exposure. In this paper, we characterize the notion of contextual privacy for user interactions with LLMs. It aims to minimize privacy risks by ensuring that users (sender) disclose only information that is both relevant and necessary for achieving their intended goals when interacting with LLMs (untrusted receivers). Through a formative design user study, we observe how even "privacy-conscious" users inadvertently reveal sensitive information through indirect disclosures. Based on insights from this study, we propose a locally-deployable framework that operates between users and LLMs, and identifies and reformulates out-of-context information in user prompts. Our evaluation using examples from ShareGPT shows that lightweight models can effectively implement this framework, achieving strong gains in contextual privacy while preserving the user's intended interaction goals through different approaches to classify information relevant to the intended goals. 

**Abstract (ZH)**: 对话代理越来越多地融入个人生活中，但用户常常低估了其中的隐私风险。一旦用户向这些代理（例如大型语言模型）提供信息，他们的私人信息就变得容易泄露。本文中，我们定义了用户与大型语言模型互动时的上下文隐私概念。其目标是通过确保用户仅披露与实现其预期目标直接相关且必要的信息，从而最小化隐私风险，即使在与不可信接收方（大型语言模型）互动时也是如此。通过一种形式化设计用户研究，我们观察到即使是“重视隐私”的用户也可能因间接披露信息而无意中暴露敏感信息。基于此研究的见解，我们提出了一种可在用户和大型语言模型之间本地部署的框架，能够识别并重新制定超出上下文的信息。通过使用来自ShareGPT的示例进行评估，我们发现轻量级模型能够有效地实施此框架，在不同方法分类与预期目标相关的信息的同时，显著提升上下文隐私，同时保留用户预期的交互目标。 

---
# REFINE: Inversion-Free Backdoor Defense via Model Reprogramming 

**Title (ZH)**: REFINE：通过模型重新编程实现无反转后门防御 

**Authors**: Yukun Chen, Shuo Shao, Enhao Huang, Yiming Li, Pin-Yu Chen, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.18508)  

**Abstract**: Backdoor attacks on deep neural networks (DNNs) have emerged as a significant security threat, allowing adversaries to implant hidden malicious behaviors during the model training phase. Pre-processing-based defense, which is one of the most important defense paradigms, typically focuses on input transformations or backdoor trigger inversion (BTI) to deactivate or eliminate embedded backdoor triggers during the inference process. However, these methods suffer from inherent limitations: transformation-based defenses often fail to balance model utility and defense performance, while BTI-based defenses struggle to accurately reconstruct trigger patterns without prior knowledge. In this paper, we propose REFINE, an inversion-free backdoor defense method based on model reprogramming. REFINE consists of two key components: \textbf{(1)} an input transformation module that disrupts both benign and backdoor patterns, generating new benign features; and \textbf{(2)} an output remapping module that redefines the model's output domain to guide the input transformations effectively. By further integrating supervised contrastive loss, REFINE enhances the defense capabilities while maintaining model utility. Extensive experiments on various benchmark datasets demonstrate the effectiveness of our REFINE and its resistance to potential adaptive attacks. 

**Abstract (ZH)**: 深度神经网络（DNNs）中的后门攻击已成为一个重要的安全威胁，攻击者可以在模型训练阶段植入隐蔽的恶意行为。基于预处理的防御方法是最重要的防御范式之一，通常关注输入变换或后门触发器逆转（BTI）来解除或消除嵌入的后门触发器。然而，这些方法存在固有的局限性：基于变换的防御方法往往难以在模型性能和防御效果之间找到平衡点，而基于BTI的防御方法则难以在缺乏先验知识的情况下准确重建触发模式。在本文中，我们提出了一种基于模型重构的无逆变换后门防御方法REFINE。REFINE由两个关键组件构成：\textbf{(1)} 输入变换模块，扰乱正常模式和后门模式，生成新的正常特征；\textbf{(2)} 输出重新映射模块，重新定义模型的输出域，以有效地引导输入变换。通过进一步结合监督对比损失，REFINE增强了防御能力，同时保持了模型的实用性。在多种基准数据集上的广泛实验表明，REFINE的有效性和其对潜在适应性攻击的抵抗力。 

---
# Exploring Patient Data Requirements in Training Effective AI Models for MRI-based Breast Cancer Classification 

**Title (ZH)**: 探索基于MRI的乳腺癌分类中有效训练AI模型所需的患者数据需求 

**Authors**: Solha Kang, Wesley De Neve, Francois Rameau, Utku Ozbulak  

**Link**: [PDF](https://arxiv.org/pdf/2502.18506)  

**Abstract**: The past decade has witnessed a substantial increase in the number of startups and companies offering AI-based solutions for clinical decision support in medical institutions. However, the critical nature of medical decision-making raises several concerns about relying on external software. Key issues include potential variations in image modalities and the medical devices used to obtain these images, potential legal issues, and adversarial attacks. Fortunately, the open-source nature of machine learning research has made foundation models publicly available and straightforward to use for medical applications. This accessibility allows medical institutions to train their own AI-based models, thereby mitigating the aforementioned concerns. Given this context, an important question arises: how much data do medical institutions need to train effective AI models? In this study, we explore this question in relation to breast cancer detection, a particularly contested area due to the prevalence of this disease, which affects approximately 1 in every 8 women. Through large-scale experiments on various patient sizes in the training set, we show that medical institutions do not need a decade's worth of MRI images to train an AI model that performs competitively with the state-of-the-art, provided the model leverages foundation models. Furthermore, we observe that for patient counts greater than 50, the number of patients in the training set has a negligible impact on the performance of models and that simple ensembles further improve the results without additional complexity. 

**Abstract (ZH)**: 过去十年见证了为医疗机构提供基于人工智能的临床决策支持解决方案的初创企业和公司的显著增加。然而，医疗决策的关键性引发了对外部软件依赖的一些担忧。主要问题包括可能存在的图像模态和用于获取这些图像的医疗器械之间的差异、潜在的法律问题以及对抗性攻击。幸运的是，机器学习研究的开源性质使得基础模型可以公开提供，并且易于用于医疗应用。这种可访问性使医疗机构能够训练自己的基于人工智能的模型，从而减轻了上述担忧。在这一背景下，一个重要的问题出现了：医疗机构需要多少数据来训练有效的AI模型？在本研究中，我们将这一问题与乳腺癌检测联系起来，这是一个特别有争议的领域，由于该疾病的高度普遍性，影响了大约每8名女性中就有1人。通过在训练集中的不同患者数量上进行大规模实验，我们表明，只要模型利用基础模型，医疗机构并不需要数十年的MRI图像来训练能够与最先进的模型竞争的AI模型。此外，我们观察到，在患者数量超过50的情况下，训练集中的患者数量对模型性能几乎没有影响，并且简单的集成进一步提高了结果，而没有增加复杂性。 

---
# Comprehensive Analysis of Transparency and Accessibility of ChatGPT, DeepSeek, And other SoTA Large Language Models 

**Title (ZH)**: 全面分析ChatGPT、DeepSeek及其他领先大型语言模型的透明度与可访问性 

**Authors**: Ranjan Sapkota, Shaina Raza, Manoj Karkee  

**Link**: [PDF](https://arxiv.org/pdf/2502.18505)  

**Abstract**: Despite increasing discussions on open-source Artificial Intelligence (AI), existing research lacks a discussion on the transparency and accessibility of state-of-the-art (SoTA) Large Language Models (LLMs). The Open Source Initiative (OSI) has recently released its first formal definition of open-source software. This definition, when combined with standard dictionary definitions and the sparse published literature, provide an initial framework to support broader accessibility to AI models such as LLMs, but more work is essential to capture the unique dynamics of openness in AI. In addition, concerns about open-washing, where models claim openness but lack full transparency, has been raised, which limits the reproducibility, bias mitigation, and domain adaptation of these models. In this context, our study critically analyzes SoTA LLMs from the last five years, including ChatGPT, DeepSeek, LLaMA, and others, to assess their adherence to transparency standards and the implications of partial openness. Specifically, we examine transparency and accessibility from two perspectives: open-source vs. open-weight models. Our findings reveal that while some models are labeled as open-source, this does not necessarily mean they are fully open-sourced. Even in the best cases, open-source models often do not report model training data, and code as well as key metrics, such as weight accessibility, and carbon emissions. To the best of our knowledge, this is the first study that systematically examines the transparency and accessibility of over 100 different SoTA LLMs through the dual lens of open-source and open-weight models. The findings open avenues for further research and call for responsible and sustainable AI practices to ensure greater transparency, accountability, and ethical deployment of these models.(DeepSeek transparency, ChatGPT accessibility, open source, DeepSeek open source) 

**Abstract (ZH)**: 尽管关于开源人工智能（AI）的讨论越来越多，现有研究仍然缺乏对最新先进（SoTA）大型语言模型（LLMs）透明度和可访问性的讨论。开源倡议（OSI）最近发布了其首个开源软件的正式定义。结合标准词典定义和少数已发表的研究成果，该定义可以作为初步框架，支持对AI模型如LLMs的更广泛访问，但还需要更多的工作来捕捉AI领域开放性的独特动态。此外，关于“开源洗牌”的担忧也引起了重视，即尽管模型声称是开源的，但缺乏完全透明性，这限制了这些模型的可重复性、偏见缓解和领域适应性。在此背景下，我们研究批判性地分析了过去五年中的SoTA LLMs，包括ChatGPT、DeepSeek、LLaMA等，以评估其对透明度标准的遵循程度及其部分开源的影响。具体而言，我们从两个视角——开源模型与开放权重模型——审视了透明度和可访问性：一是开源 vs. 开放权重模型。研究结果表明，虽然某些模型被标榜为开源，但这并不意味着它们完全实现了开源。即使是最佳情况，开源模型也常常不报告模型训练数据、代码以及重要指标如权重的可访问性、碳排放等。据我们所知，这是首次系统性研究超100种SoTA LLMs的透明度和可访问性，通过开源和开放权重模型的双重视角进行评估。研究发现为未来的研究打开了新途径，并呼吁采取负责任和可持续的AI实践，以确保这些模型的更大透明度、可问责性和伦理应用。（DeepSeek透明度，ChatGPT可访问性，开源，DeepSeek开源） 

---
# TurboFuzzLLM: Turbocharging Mutation-based Fuzzing for Effectively Jailbreaking Large Language Models in Practice 

**Title (ZH)**: TurboFuzzLLM：基于变异的 fuzzing 技术加速在实际中有效破解大型语言模型 

**Authors**: Aman Goel, Xian Carrie Wu, Zhe Wang, Dmitriy Bespalov, Yanjun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2502.18504)  

**Abstract**: Jailbreaking large-language models (LLMs) involves testing their robustness against adversarial prompts and evaluating their ability to withstand prompt attacks that could elicit unauthorized or malicious responses. In this paper, we present TurboFuzzLLM, a mutation-based fuzzing technique for efficiently finding a collection of effective jailbreaking templates that, when combined with harmful questions, can lead a target LLM to produce harmful responses through black-box access via user prompts. We describe the limitations of directly applying existing template-based attacking techniques in practice, and present functional and efficiency-focused upgrades we added to mutation-based fuzzing to generate effective jailbreaking templates automatically. TurboFuzzLLM achieves $\geq$ 95\% attack success rates (ASR) on public datasets for leading LLMs (including GPT-4o \& GPT-4 Turbo), shows impressive generalizability to unseen harmful questions, and helps in improving model defenses to prompt attacks. 

**Abstract (ZH)**: 将大型语言模型（LLMs）破解以测试其面对恶意提示的鲁棒性，并评估其抵御可能引发未经授权或恶意响应的提示攻击的能力。在本文中，我们提出了一种基于变异的模糊测试技术——TurboFuzzLLM，该技术能够高效地找到一组有效的破解模板，当这些模板与有害问题结合使用时，可以使目标LLM通过用户提示的黑盒访问生成有害响应。我们讨论了直接应用现有模板攻击技术的实际局限性，并介绍了我们为变异模糊测试添加的功能性和效率性改进，以便自动生成有效的破解模板。TurboFuzzLLM在主要LLM（包括GPT-4o及GPT-4 Turbo）的公开数据集上实现了成功率不低于95%（ASR），具有良好的泛化能力以应对未见过的有害问题，并有助于改进模型对提示攻击的防御能力。 

---
# Deep Learning-based Dual Watermarking for Image Copyright Protection and Authentication 

**Title (ZH)**: 基于深度学习的双重水印技术在图像版权保护与验证中的应用 

**Authors**: Sudev Kumar Padhi, Archana Tiwari, Sk. Subidh Ali  

**Link**: [PDF](https://arxiv.org/pdf/2502.18501)  

**Abstract**: Advancements in digital technologies make it easy to modify the content of digital images. Hence, ensuring digital images integrity and authenticity is necessary to protect them against various attacks that manipulate them. We present a Deep Learning (DL) based dual invisible watermarking technique for performing source authentication, content authentication, and protecting digital content copyright of images sent over the internet. Beyond securing images, the proposed technique demonstrates robustness to content-preserving image manipulations. It is also impossible to imitate or overwrite watermarks because the cryptographic hash of the image and the dominant features of the image in the form of perceptual hash are used as watermarks. We highlighted the need for source authentication to safeguard image integrity and authenticity, along with identifying similar content for copyright protection. After exhaustive testing, we obtained a high peak signal-to-noise ratio (PSNR) and structural similarity index measure (SSIM), which implies there is a minute change in the original image after embedding our watermarks. Our trained model achieves high watermark extraction accuracy and to the best of our knowledge, this is the first deep learning-based dual watermarking technique proposed in the literature. 

**Abstract (ZH)**: 数字技术的进步使得修改数字图像的内容变得十分容易。因此，确保数字图像的完整性和真实性变得至关重要，以抵御各种可能篡改图像的攻击。本文提出了一种基于深度学习（DL）的双隐藏水印技术，用于进行源认证、内容认证及保护通过互联网传输的图像的版权。除了保护图像外，该技术还显示出对保留内容的图像篡改的鲁棒性。由于使用了图像的加密哈希值和感知哈希值的形式表示的图像的主要特征作为水印，因此不可能伪造或覆盖水印。我们强调了源认证的重要性，以确保图像的完整性和真实性，并识别相似内容以保护版权。经过彻底测试后，我们获得了较高的峰值信噪比（PSNR）和结构相似性指数（SSIM），这表明在嵌入水印后，原始图像几乎没有变化。我们的训练模型实现了高水印提取精度，据我们所知，这是文献中首个基于深度学习的双水印技术。 

---
# Mechanistic Understanding of Language Models in Syntactic Code Completion 

**Title (ZH)**: 语言模型在句法代码补全中的工作机制理解 

**Authors**: Samuel Miller, Daking Rai, Ziyu Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18499)  

**Abstract**: Recently, language models (LMs) have shown impressive proficiency in code generation tasks, especially when fine-tuned on code-specific datasets, commonly known as Code LMs. However, our understanding of the internal decision-making processes of Code LMs, such as how they use their (syntactic or semantic) knowledge, remains limited, which could lead to unintended harm as they are increasingly used in real life. This motivates us to conduct one of the first Mechanistic Interpretability works to understand how Code LMs perform a syntactic completion task, specifically the closing parenthesis task, on the CodeLlama-7b model (Roziere et al. 2023). Our findings reveal that the model requires middle-later layers until it can confidently predict the correct label for the closing parenthesis task. Additionally, we identify that while both multi-head attention (MHA) and feed-forward (FF) sub-layers play essential roles, MHA is particularly crucial. Furthermore, we also discover attention heads that keep track of the number of already closed parentheses precisely but may or may not promote a correct number of closing parentheses that are still missing, leading to a positive or negative impact on the model's performance. 

**Abstract (ZH)**: 近年来，语言模型（LMs）在代码生成任务中展示了令人印象深刻的性能，尤其是在经过特定代码数据集（通常称为代码LMs）微调后更是如此。然而，我们对代码LMs内部决策过程的理解仍然有限，这包括它们如何利用自身的（语法或语义）知识，这种理解的缺乏可能导致它们在现实生活中的应用出现未预见的风险。因此，我们旨在进行其中一项首次基于机制可解释性的工作，以探索Code LMs在CodeLlama-7b模型中（Roziere等，2023）执行语法补全任务（具体来说是右括号任务）的方式。我们的研究发现表明，模型需要中间或后期的层级才能自信地预测右括号任务的正确标签。此外，我们还发现多头注意力（MHA）和前向（FF）子层都扮演着至关重要的角色，但MHA尤其重要。进一步地，我们还发现了能够精确跟踪已闭合括号数量的注意力头，但这些头可能或可能不会推动正确的缺失括号数量，从而对模型性能产生正向或负向影响。 

---
# A Comprehensive Survey on Composed Image Retrieval 

**Title (ZH)**: 全面综述合成图像检索 

**Authors**: Xuemeng Song, Haoqiang Lin, Haokun Wen, Bohan Hou, Mingzhu Xu, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2502.18495)  

**Abstract**: Composed Image Retrieval (CIR) is an emerging yet challenging task that allows users to search for target images using a multimodal query, comprising a reference image and a modification text specifying the user's desired changes to the reference image. Given its significant academic and practical value, CIR has become a rapidly growing area of interest in the computer vision and machine learning communities, particularly with the advances in deep learning. To the best of our knowledge, there is currently no comprehensive review of CIR to provide a timely overview of this field. Therefore, we synthesize insights from over 120 publications in top conferences and journals, including ACM TOIS, SIGIR, and CVPR In particular, we systematically categorize existing supervised CIR and zero-shot CIR models using a fine-grained taxonomy. For a comprehensive review, we also briefly discuss approaches for tasks closely related to CIR, such as attribute-based CIR and dialog-based CIR. Additionally, we summarize benchmark datasets for evaluation and analyze existing supervised and zero-shot CIR methods by comparing experimental results across multiple datasets. Furthermore, we present promising future directions in this field, offering practical insights for researchers interested in further exploration. 

**Abstract (ZH)**: 合成图像检索（CIR）是一项新兴且具有挑战性的任务，允许用户使用包含参考图像和修改文本的多模态查询来搜索目标图像。其中，修改文本指定了用户希望对参考图像进行的更改。鉴于其在学术和实践方面的重大价值，CIR 已成为计算机视觉和机器学习社区的迅速发展的研究领域，特别是在深度学习的推动下。据我们所知，目前尚未有关于 CIR 的全面回顾，以提供对该领域的及时概述。因此，我们综合了超过 120 篇发表在顶级会议和期刊上的文献的见解，包括 ACM TOIS、SIGIR 和 CVPR。特别是在此过程中，我们系统地使用细粒度分类法对现有的监督 CIR 和零样本 CIR 模型进行了分类。为了进行全面回顾，我们还简要讨论了与 CIR 密切相关的任务，如基于属性的 CIR 和基于对话的 CIR。此外，我们总结了用于评估的基准数据集，并通过多个数据集的实验结果比较分析现有的监督和零样本 CIR 方法。最后，我们提出了该领域的未来发展方向，为对该领域进一步探索感兴趣的科研人员提供了实用建议。 

---
# Rule-based autocorrection of Piping and Instrumentation Diagrams (P&IDs) on graphs 

**Title (ZH)**: 基于规则的管道和仪表图（P&ID）图上自校正技术 

**Authors**: Lukas Schulze Balhorn, Niels Seijsener, Kevin Dao, Minji Kim, Dominik P. Goldstein, Ge H. M. Driessen, Artur M. Schweidtmann  

**Link**: [PDF](https://arxiv.org/pdf/2502.18493)  

**Abstract**: A piping and instrumentation diagram (P&ID) is a central reference document in chemical process engineering. Currently, chemical engineers manually review P&IDs through visual inspection to find and rectify errors. However, engineering projects can involve hundreds to thousands of P&ID pages, creating a significant revision workload. This study proposes a rule-based method to support engineers with error detection and correction in P&IDs. The method is based on a graph representation of P&IDs, enabling automated error detection and correction, i.e., autocorrection, through rule graphs. We use our pyDEXPI Python package to generate P&ID graphs from DEXPI-standard P&IDs. In this study, we developed 33 rules based on chemical engineering knowledge and heuristics, with five selected rules demonstrated as examples. A case study on an illustrative P&ID validates the reliability and effectiveness of the rule-based autocorrection method in revising P&IDs. 

**Abstract (ZH)**: 工艺和仪表图（P&ID）是化工过程工程中的核心参考文档。目前，化工工程师通过视觉检查手动审查P&ID以查找并纠正错误。然而，工程项目可能涉及数百到数千页的P&ID，从而产生大量的修订工作量。本研究提出了一种基于规则的方法，以支持工程师在P&ID中进行错误检测和纠正。该方法基于P&ID的图表示，通过规则图实现自动化错误检测和纠正，即自动纠错。我们使用自主研发的pyDEXPI Python包从DEXPI标准的P&ID生成P&ID图。在本研究中，我们基于化学工程知识和启发式方法制定了33条规则，并选取了五条规则作为示例。针对一个示例P&ID的案例研究验证了基于规则的自动纠错方法在修订P&ID方面的可靠性和有效性。 

---
# LLM4EFFI: Leveraging Large Language Models to Enhance Code Efficiency and Correctness 

**Title (ZH)**: LLM4EFFI：利用大型语言模型提升代码效率和正确性 

**Authors**: Tong Ye, Weigang Huang, Xuhong Zhang, Tengfei Ma, Peiyu Liu, Jianwei Yin, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18489)  

**Abstract**: Large Language Models (LLMs), particularly Code LLMs, have demonstrated impressive performance in code generation. Current research primarily focuses on the correctness of generated code, while efficiency remains less explored. Recent works have focused on modifying the initial version of the code to improve its efficiency. However, such refinements are limited by the algorithmic design and overall logic of the initial code, resulting in only incremental improvements. In contrast, when human developers write high-quality code, they typically begin by designing several potential solutions at the logical level, evaluating various algorithms and their complexities, and then proceeding to implement and optimize the solution. In this study, we introduce \tool: \uline{L}arge \uline{L}anguage \uline{M}odel for Code \uline{Effi}ciency, a novel framework that enables LLMs to generate code that balances both efficiency and correctness. Specifically, \tool divides the efficiency optimization process into two domains: algorithmic exploration in the logic domain and implementation optimization in the code domain. The correctness of the code is then guaranteed through a synthetic test case refinement process. This approach, which prioritizes efficiency before ensuring correctness, offers a new paradigm for efficient code generation. Experiments demonstrate that \tool consistently improves both efficiency and correctness, achieving new state-of-the-art performance in code efficiency benchmarks across various LLM backbones. 

**Abstract (ZH)**: 大型语言模型（LLMs），尤其是代码LLMs，在代码生成方面展现了令人印象深刻的表现。当前的研究主要集中在生成代码的正确性上，而效率则较少被探讨。最近的研究重点在于修改初始代码版本以提高其效率。然而，这些改进受到初始代码的算法设计和总体逻辑的限制，导致只能实现渐进式的改进。相比之下，当人类开发者编写高质量代码时，他们通常会从逻辑层面开始设计几种潜在解决方案，评估不同的算法及其复杂度，然后再进行实现和优化。在这项研究中，我们引入了一个名为 \tool 的新框架：大型语言模型用于代码效率，该框架使LLMs能够生成既高效又正确的代码。具体而言，\tool 将效率优化过程分为两个领域：逻辑域中的算法探索和代码域中的实现优化。代码的正确性则通过合成测试用例的改进过程来保证。该方法在优先考虑效率后再确保正确性的策略，为高效的代码生成提供了一种新的范式。实验结果显示，\tool 一致地提高了效率和正确性，在不同LLM骨干模型的代码效率基准测试中取得了新的最先进性能。 

---
# AuPair: Golden Example Pairs for Code Repair 

**Title (ZH)**: AuPair：代码修复的优秀实例对 

**Authors**: Aditi Mavalankar, Hassan Mansoor, Zita Marinho, Masha Samsikova, Tom Schaul  

**Link**: [PDF](https://arxiv.org/pdf/2502.18487)  

**Abstract**: Scaling up inference-time compute has proven to be a valuable strategy in improving the performance of Large Language Models (LLMs) without fine-tuning. An important task that can benefit from additional inference-time compute is self-repair; given an initial flawed response, or guess, the LLM corrects its own mistake and produces an improved response, or fix. We leverage the in-context learning ability of LLMs to perform self-repair in the coding domain. The key contribution of our paper is an approach that synthesises and selects an ordered set of golden example pairs, or AuPairs, of these initial guesses and subsequent fixes for the corresponding problems. Each such AuPair is provided as a single in-context example at inference time to generate a repaired solution. For an inference-time compute budget of $N$ LLM calls per problem, $N$ AuPairs are used to generate $N$ repaired solutions, out of which the highest-scoring solution is selected as the final answer. The underlying intuition is that if the LLM is given a different example of fixing an incorrect guess each time, it can subsequently generate a diverse set of repaired solutions. Our algorithm selects these AuPairs in a manner that maximises complementarity and usefulness. We demonstrate the results of our algorithm on 5 LLMs across 7 competitive programming datasets for the code repair task. Our algorithm yields a significant boost in performance compared to best-of-$N$ and self-repair, and also exhibits strong generalisation across datasets and models. Moreover, our approach shows significantly stronger scaling with inference-time compute budget compared to baselines. 

**Abstract (ZH)**: 在推理时间计算上进行扩展已被证明是提高大型语言模型（LLMs）性能的一种有价值的战略，无需进行微调。一个可以从额外推理时间计算中受益的重要任务是自修复；给定一个初始的错误响应或猜测，LLM自行修正错误并生成改进的响应或修复。我们利用LLMs的在上下文学习能力，在编程领域执行自修复任务。我们论文的关键贡献是一种方法，该方法综合并选择了一组有序的黄金示例对，或AuPairs，这些初始猜测和后续修复对应于相应的问题。每个这样的AuPair在推理时作为一个单一的在上下文示例提供，以生成修复解决方案。对于每个问题的推理时间计算预算为$N$次LLM调用，使用$N$个AuPairs生成$N$个修复解决方案，最终选择得分最高的解决方案作为最终答案。基本直觉是，如果每次给LLM提供一个不同修复错误猜测的示例，它可以随后生成多样化的修复解决方案。我们的算法以最大化互补性和有用性的方式选择这些AuPairs。我们在7个具有竞争力的编程数据集上的5个LLM上对我们的算法进行了实验。与最佳选择和自修复相比，我们的算法在代码修复任务上的性能提升显著，并且在数据集和模型之间表现出良好的泛化能力。此外，与基线相比，我们的方法在推理时间计算预算方面的扩展性更强。 

---
# AI Enhanced Ontology Driven NLP for Intelligent Cloud Resource Query Processing Using Knowledge Graphs 

**Title (ZH)**: AI增强的知识驱动自然语言处理在知识图谱支持下的智能云资源查询处理 

**Authors**: Krishna Chaitanya Sunkara, Krishnaiah Narukulla  

**Link**: [PDF](https://arxiv.org/pdf/2502.18484)  

**Abstract**: The conventional resource search in cloud infrastructure relies on keyword-based searches or GUIDs, which demand exact matches and significant user effort to locate resources. These conventional search approaches often fail to interpret the intent behind natural language queries, making resource discovery inefficient and inaccessible to users. Though there exists some form of NLP based search engines, they are limited and focused more on analyzing the NLP query itself and extracting identifiers to find the resources. But they fail to search resources based on their behavior or operations or their capabilities or relationships or features or business relevance or the dynamic changing state or the knowledge these resources have. The search criteria has been changing with the inundation of AI based services which involved discovering not just the requested resources and identifiers but seeking insights. The real intent of a search has never been to just to list the resources but with some actual context such as to understand causes of some behavior in the system, compliance checks, capacity estimations, network constraints, or troubleshooting or business insights. This paper proposes an advanced Natural Language Processing (NLP) enhanced by ontology-based semantics to enable intuitive, human-readable queries which allows users to actually discover the intent-of-search itself. By constructing an ontology of cloud resources, their interactions, and behaviors, the proposed framework enables dynamic intent extraction and relevance ranking using Latent Semantic Indexing (LSI) and AI models. It introduces an automated pipeline which integrates ontology extraction by AI powered data crawlers, building a semantic knowledge base for context aware resource discovery. 

**Abstract (ZH)**: 云基础设施中的传统资源搜索依赖于关键词搜索或全局唯一标识符(GUID)，这需要完全匹配并要求用户付出大量努力来定位资源。这些传统搜索方法往往无法正确解释自然语言查询背后的意图，导致资源发现效率低下且对用户不友好。尽管存在一些基于自然语言处理(NLP)的搜索引擎，它们主要关注于分析NLP查询本身并提取标识符以查找资源。然而，这些搜索方法并未基于资源的行为、操作、能力、关系、功能或业务相关性，以及这些资源所具备的知识来进行搜索。随着基于AI的服务的普及，搜索标准也发生了改变，不再仅仅局限于列出所需资源和标识符，而是要寻求更深入的洞察。搜索的真实意图不仅仅是列出资源，而是需要有一定的上下文，如理解系统中某些行为的原因、合规检查、容量估算、网络约束、故障排除或业务洞察。本文提出了一种由本体增强的高级自然语言处理(NLP)技术，使得用户可以提出直观、易于理解的查询，从而真正发现用户背后的搜索意图。通过构建云计算资源的本体、它们之间的交互和行为，本框架使用潜在语义索引(LSI)和AI模型实现动态的意图提取和相关性排名，并引入了一个自动化工作流，该工作流结合了由AI驱动的数据抓取器进行本体提取，并构建了一个语义知识库以实现上下文感知的资源发现。 

---
# Modeling Churn in Recommender Systems with Aggregated Preferences 

**Title (ZH)**: 使用聚合偏好建模推荐系统中的客户流失 

**Authors**: Gur Keinan, Omer Ben-Porat  

**Link**: [PDF](https://arxiv.org/pdf/2502.18483)  

**Abstract**: While recommender systems (RSs) traditionally rely on extensive individual user data, regulatory and technological shifts necessitate reliance on aggregated user information. This shift significantly impacts the recommendation process, requiring RSs to engage in intensive exploration to identify user preferences. However, this approach risks user churn due to potentially unsatisfactory recommendations. In this paper, we propose a model that addresses the dual challenges of leveraging aggregated user information and mitigating churn risk. Our model assumes that the RS operates with a probabilistic prior over user types and aggregated satisfaction levels for various content types. We demonstrate that optimal policies naturally transition from exploration to exploitation in finite time, develop a branch-and-bound algorithm for computing these policies, and empirically validate its effectiveness. 

**Abstract (ZH)**: 尽管传统的推荐系统（RS）依赖于大量的个体用户数据，但监管和技术的变化要求系统依赖聚合的用户信息。这种转变显著影响了推荐过程，使得RS需要进行密集的探索以识别用户偏好。然而，这种 approach 可能因推荐结果不尽如人意而导致用户流失。在本文中，我们提出了一种模型，以应对利用聚合用户信息和缓解用户流失风险的双重挑战。我们的模型假设RS在不同类型的内容上具有关于用户类型及其聚合满意度的概率先验信息。我们证明了最优策略会在有限时间内自然从探索过渡到利用，并开发了一种分支定界算法来计算这些策略，并通过实证研究验证其有效性。 

---
# MixLLM: Dynamic Routing in Mixed Large Language Models 

**Title (ZH)**: MixLLM：混合大型语言模型中的动态路由 

**Authors**: Xinyuan Wang, Yanchi Liu, Wei Cheng, Xujiang Zhao, Zhengzhang Chen, Wenchao Yu, Yanjie Fu, Haifeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18482)  

**Abstract**: Large Language Models (LLMs) exhibit potential artificial generic intelligence recently, however, their usage is costly with high response latency. Given mixed LLMs with their own strengths and weaknesses, LLM routing aims to identify the most suitable model for each query in the stream to maximize response quality and minimize cost and latency. However, the challenges involve: (1) dynamic trade-offs among quality, cost, and latency; (2) enabling continual learning in deployed systems; and (3) navigating a varying (e.g., new LLM addition or old LLM removal) set of LLM candidates over time. To bridge these gaps, we develop MixLLM, a dynamic contextual-bandit-based routing system for query-LLM assignment. Specifically, we first leverage query tags to enhance query embeddings for the routing task. Next, we design lightweight prediction models to estimate the response qualities and costs of queries over LLMs. We then devise a meta-decision maker to choose the query-LLM assignments to best tradeoff response quality, cost, and latency. Finally, the system benefits from continual training, allowing it to adapt to evolving queries and user feedback over time. Our extensive experiments show that MixLLM achieves the best trade-offs in response quality, cost, and latency (97.25% of GPT-4's quality at 24.18% of the cost under the time constraint). 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）展现出了潜在的人工通用智能能力，但其使用成本较高且响应延迟较长。鉴于混合LLMs各自具有优势和劣势，LLM路由旨在识别流查询中最合适的模型，以最大化响应质量并最小化成本和延迟。然而，面临的挑战包括：（1）在质量、成本和延迟之间动态权衡；（2）在部署系统中实现持续学习；以及（3）随着时间变化导航LLM候选集合的变化（例如，新LLM的添加或旧LLM的移除）。为了弥补这些差距，我们开发了MixLLM，这是一种基于动态上下文臂拍策略的查询-LLM路由系统。具体来说，我们首先利用查询标签增强查询嵌入以优化路由任务。接下来，我们设计了轻量级预测模型来估计查询在不同LLM上的响应质量和成本。然后，我们设计了一个元决策者来选择最佳权衡响应质量、成本和延迟的查询-LLM组合。最后，该系统得益于持续训练，使其能够适应不断变化的查询和用户反馈。我们的广泛实验表明，MixLLM在响应质量和延迟方面实现了最佳权衡（在时间约束条件下，成本仅为GPT-4的24.18%，而响应质量达到97.25%）。 

---
# MDE: Modality Discrimination Enhancement for Multi-modal Recommendation 

**Title (ZH)**: MDE：多模态推荐中的模态鉴别增强 

**Authors**: Hang Zhou, Yucheng Wang, Huijing Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18481)  

**Abstract**: Multi-modal recommendation systems aim to enhance performance by integrating an item's content features across various modalities with user behavior data. Effective utilization of features from different modalities requires addressing two challenges: preserving semantic commonality across modalities (modality-shared) and capturing unique characteristics for each modality (modality-specific). Most existing approaches focus on aligning feature spaces across modalities, which helps represent modality-shared features. However, modality-specific distinctions are often neglected, especially when there are significant semantic variations between modalities. To address this, we propose a Modality Distinctiveness Enhancement (MDE) framework that prioritizes extracting modality-specific information to improve recommendation accuracy while maintaining shared features. MDE enhances differences across modalities through a novel multi-modal fusion module and introduces a node-level trade-off mechanism to balance cross-modal alignment and differentiation. Extensive experiments on three public datasets show that our approach significantly outperforms other state-of-the-art methods, demonstrating the effectiveness of jointly considering modality-shared and modality-specific features. 

**Abstract (ZH)**: 多模态推荐系统旨在通过综合项目内容特征（跨多种模态）与用户行为数据来提升性能。有效利用不同模态的特征需要解决两个挑战：保留模态间的语义一致性（跨模态共享）和捕捉每个模态的独特特性（模态特定）。现有的大多数方法侧重于在模态间对齐特征空间，从而有助于表示跨模态共享特征。然而，模态特定的差异往往被忽视，尤其是在模态之间存在显著语义差异时。为此，我们提出了一个模态差异增强（MDE）框架，该框架优先提取模态特定的信息以提高推荐准确性，同时保持共享特征。MDE 通过一个新的多模态融合模块增强不同模态之间的差异，并引入节点级别的权衡机制来平衡跨模态对齐与差异化。在三个公开数据集上的广泛实验表明，我们的方法显著优于其他最先进的方法，证明了同时考虑模态共享和模态特定特征的有效性。 

---
# QExplorer: Large Language Model Based Query Extraction for Toxic Content Exploration 

**Title (ZH)**: QExplorer：基于大型语言模型的有毒内容查询提取 

**Authors**: Shaola Ren, Li Ke, Longtao Huang, Dehong Gao, Hui Xue  

**Link**: [PDF](https://arxiv.org/pdf/2502.18480)  

**Abstract**: Automatically extracting effective queries is challenging in information retrieval, especially in toxic content exploration, as such content is likely to be disguised. With the recent achievements in generative Large Language Model (LLM), we are able to leverage the capabilities of LLMs to extract effective queries for similar content exploration directly. This study proposes QExplorer, an approach of large language model based Query Extraction for toxic content Exploration. The QExplorer approach involves a 2-stage training process: instruction Supervised FineTuning (SFT) and preference alignment using Direct Preference Optimization (DPO), as well as the datasets construction with feedback of search system. To verify the effectiveness of QExplorer, a series of offline and online experiments are conducted on our real-world system. The offline empirical results demonstrate that the performance of our automatic query extraction outperforms that of several LLMs and humans. The online deployment shows a significant increase in the detection of toxic items. 

**Abstract (ZH)**: 在信息检索中，自动提取有效的查询具有挑战性，特别是在有毒内容探索方面，因为这类内容往往会被伪装。得益于近期生成型大语言模型（LLM）的进展，我们能够利用LLM的能力直接提取用于类似内容探索的有效查询。本研究提出了一种基于大语言模型的查询提取方法——QExplorer，用于有毒内容探索。QExplorer方法包含一个两阶段的训练过程：指令监督微调（SFT）和基于直接偏好优化（DPO）的偏好对齐，同时也包括使用搜索引擎反馈构建数据集的过程。为了验证QExplorer的有效性，在我们实际系统上进行了离线和在线实验。离线实验结果表明，我们的自动查询提取性能优于几种LLM和人类的表现。在线部署结果显示，有毒物品的检测能力显著提高。 

---
# Beyond Self-Consistency: Loss-Balanced Perturbation-Based Regularization Improves Industrial-Scale Ads Ranking 

**Title (ZH)**: 超越自我一致性：损失平衡扰动正则化改进工业规模广告排名 

**Authors**: Ilqar Ramazanli, Hamid Eghbalzadeh, Xiaoyi Liu, Yang Wang, Jiaxiang Fu, Kaushik Rangadurai, Sem Park, Bo Long, Xue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18478)  

**Abstract**: Perturbation-based regularization techniques address many challenges in industrial-scale large models, particularly with sparse labels, and emphasize consistency and invariance for perturbation in model predictions. One of the popular regularization techniques has been various forms of self-consistency, which involve making small modifications to input data while preserving contextual information and enforcing similar predictions through auxiliary loss functions. In this work, we explore the first successful application of perturbation-based regularization algorithms in large-scale ads ranking models, and further propose a novel regularization algorithm, namely, Loss-Balanced Small Perturbation Regularization (LSPR) that can be used in potentially any deep learning model. We have successfully demonstrate that both Self-Consistency Regularization approaches (SCR) and LSPR are scalable and can improve ads delivery systems. By conducting industrial-scale experiments, and numerical analysis, we additionally show that our proposed LSPR, performs consistently better compared to SCR, across various groups and signal availability setups. Finally, we report a successful application of the proposed LSPR in a billion-scale industrial ranking system, which to the best of our knowledge, is the first of its kind, and it is specially designed to address the various scalability challenges (e.g, various surfaces, geological locations, clients and so on) as we will mention in this paper. 

**Abstract (ZH)**: 基于扰动的正则化技术在工业规模的大型模型中解决了许多挑战，特别是在稀疏标签的情况下，并强调扰动在模型预测中的一致性和不变性。其中，流行的一种正则化技术是各种形式的自我一致性，它涉及对输入数据进行轻微修改，同时保留上下文信息，并通过辅助损失函数强制执行相似的预测。在本文中，我们探讨了基于扰动的正则化算法首次成功应用于大规模广告排名模型，并进一步提出了一种新颖的正则化算法，即损失平衡的小扰动正则化（LSPR），该算法可以在任何深度学习模型中使用。我们成功地证明了自我一致性正则化（Self-Consistency Regularization，SCR）方法和LSPR在扩大广告交付系统方面都是可扩展的，并能够提高广告投放系统的效果。通过进行工业规模的实验和数值分析，我们还表明，与SCR相比，我们提出的LSPR在各种群体和信号可用性设置下表现更加一致。最后，我们在十亿规模的工业排名系统中成功应用了LSPR，据我们所知，这是此类系统的首次应用，特别设计用于解决各种可扩展性挑战（例如，各种表面、地质位置、客户端等），如本文中将提到的。 

---
# Recommendations Beyond Catalogs: Diffusion Models for Personalized Generation 

**Title (ZH)**: 超越目录的推荐：用于个性化生成的扩散模型 

**Authors**: Gabriel Patron, Zhiwei Xu, Ishan Kapnadak, Felipe Maia Polo  

**Link**: [PDF](https://arxiv.org/pdf/2502.18477)  

**Abstract**: Modern recommender systems follow the guiding principle of serving the right user, the right item at the right time. One of their main limitations is that they are typically limited to items already in the catalog. We propose REcommendations BEyond CAtalogs, REBECA, a new class of probabilistic diffusion-based recommender systems that synthesize new items tailored to individual tastes rather than retrieve items from the catalog. REBECA combines efficient training in embedding space with a novel diffusion prior that only requires users' past ratings of items. We evaluate REBECA on real-world data and propose novel personalization metrics for generative recommender systems. Extensive experiments demonstrate that REBECA produces high-quality, personalized recommendations, generating images that align with users' unique preferences. 

**Abstract (ZH)**: 现代推荐系统遵循“在正确的时间为正确的用户推荐正确的项目”的指导原则。它们的主要限制之一是通常仅限于目录中的现有项目。我们提出了一种新的概率扩散型推荐系统RERecommendations BEyond CAtalogs (REBECA)，该系统能够合成符合个人喜好的新项目，而不是从目录中检索项目。REBECA 结合了在嵌入空间中的高效训练以及一种新颖的扩散先验，只需利用用户对项目的过往评分。我们在真实数据上评估了 REBECA，并提出了一种新的生成推荐系统的个性化评估指标。大量实验表明，REBECA 能够生成高质量且个性化的推荐，生成的图像与用户的独特偏好高度一致。 

---
# A Contemporary Survey of Large Language Model Assisted Program Analysis 

**Title (ZH)**: 大型语言模型辅助程序分析的当代调查 

**Authors**: Jiayimei Wang, Tao Ni, Wei-Bin Lee, Qingchuan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18474)  

**Abstract**: The increasing complexity of software systems has driven significant advancements in program analysis, as traditional methods unable to meet the demands of modern software development. To address these limitations, deep learning techniques, particularly Large Language Models (LLMs), have gained attention due to their context-aware capabilities in code comprehension. Recognizing the potential of LLMs, researchers have extensively explored their application in program analysis since their introduction. Despite existing surveys on LLM applications in cybersecurity, comprehensive reviews specifically addressing their role in program analysis remain scarce. In this survey, we systematically review the application of LLMs in program analysis, categorizing the existing work into static analysis, dynamic analysis, and hybrid approaches. Moreover, by examining and synthesizing recent studies, we identify future directions and challenges in the field. This survey aims to demonstrate the potential of LLMs in advancing program analysis practices and offer actionable insights for security researchers seeking to enhance detection frameworks or develop domain-specific models. 

**Abstract (ZH)**: 软件系统的日益复杂性推动了程序分析领域的显著进步，传统的分析方法已无法满足现代软件开发的需求。为应对这些局限性，深度学习技术，特别是大型语言模型（LLMs），因其在代码理解方面的上下文感知能力而引起关注。鉴于LLMs的潜在价值，研究人员在其引入后广泛探索了它们在程序分析中的应用。尽管已有对LLMs在网络安全应用的综述，但专门讨论其在程序分析中作用的全面回顾仍然很少见。在这篇综述中，我们将系统地回顾LLMs在程序分析中的应用，并将现有工作分为静态分析、动态分析和混合方法等类别。通过对近期研究的审核和综合，我们还确定了该领域的未来方向和挑战。本文旨在展示LLMs在推动程序分析实践方面的潜力，并为寻求增强检测框架或开发领域特定模型的网络安全研究人员提供可操作的见解。 

---
# FinBloom: Knowledge Grounding Large Language Model with Real-time Financial Data 

**Title (ZH)**: FinBloom：基于实时金融数据的大规模语言模型知识 grounding 

**Authors**: Ankur Sinha, Chaitanya Agarwal, Pekka Malo  

**Link**: [PDF](https://arxiv.org/pdf/2502.18471)  

**Abstract**: Large language models (LLMs) excel at generating human-like responses but often struggle with interactive tasks that require access to real-time information. This limitation poses challenges in finance, where models must access up-to-date information, such as recent news or price movements, to support decision-making. To address this, we introduce Financial Agent, a knowledge-grounding approach for LLMs to handle financial queries using real-time text and tabular data. Our contributions are threefold: First, we develop a Financial Context Dataset of over 50,000 financial queries paired with the required context. Second, we train FinBloom 7B, a custom 7 billion parameter LLM, on 14 million financial news articles from Reuters and Deutsche Presse-Agentur, alongside 12 million Securities and Exchange Commission (SEC) filings. Third, we fine-tune FinBloom 7B using the Financial Context Dataset to serve as a Financial Agent. This agent generates relevant financial context, enabling efficient real-time data retrieval to answer user queries. By reducing latency and eliminating the need for users to manually provide accurate data, our approach significantly enhances the capability of LLMs to handle dynamic financial tasks. Our proposed approach makes real-time financial decisions, algorithmic trading and other related tasks streamlined, and is valuable in contexts with high-velocity data flows. 

**Abstract (ZH)**: 大型语言模型（LLMs）在生成类人回复方面表现出色，但在处理需要实时信息访问的交互任务时往往显得力不从心。这种局限性在金融领域尤为突出，因为在金融领域，模型必须访问最新的信息（如最近的新闻或价格变动）以支持决策。为了解决这一问题，我们引入了“金融代理”这一知识落地方法，该方法允许LLMs使用实时文本和表格数据来处理金融查询。我们的贡献主要包括三个方面：首先，我们开发了一个包含超过50,000个金融查询及其所需上下文的数据集。其次，我们在来自路透社和德新社的1400万篇金融新闻文章以及1200万篇美国证券交易委员会（SEC）文件上训练了一个70亿参数的定制模型——FinBloom 7B。最后，我们使用财务上下文数据集对FinBloom 7B进行微调，使之成为金融代理。该代理能够生成相关的财务上下文信息，从而高效地实现实时数据检索，以回答用户查询。通过减少延迟并消除用户手动提供准确数据的需求，我们的方法显著增强了LLMs处理动态金融任务的能力。我们提出的方法使得实时金融决策、算法交易及其他相关任务得以简化，尤其适用于数据流快速的应用场景。 

---
# SOK: Exploring Hallucinations and Security Risks in AI-Assisted Software Development with Insights for LLM Deployment 

**Title (ZH)**: SOK：探索AI辅助软件开发中的幻觉和安全风险及其对大规模语言模型部署的见解 

**Authors**: Ariful Haque, Sunzida Siddique, Md. Mahfuzur Rahman, Ahmed Rafi Hasan, Laxmi Rani Das, Marufa Kamal, Tasnim Masura, Kishor Datta Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2502.18468)  

**Abstract**: The integration of Large Language Models (LLMs) such as GitHub Copilot, ChatGPT, Cursor AI, and Codeium AI into software development has revolutionized the coding landscape, offering significant productivity gains, automation, and enhanced debugging capabilities. These tools have proven invaluable for generating code snippets, refactoring existing code, and providing real-time support to developers. However, their widespread adoption also presents notable challenges, particularly in terms of security vulnerabilities, code quality, and ethical concerns. This paper provides a comprehensive analysis of the benefits and risks associated with AI-powered coding tools, drawing on user feedback, security analyses, and practical use cases. We explore the potential for these tools to replicate insecure coding practices, introduce biases, and generate incorrect or non-sensical code (hallucinations). In addition, we discuss the risks of data leaks, intellectual property violations and the need for robust security measures to mitigate these threats. By comparing the features and performance of these tools, we aim to guide developers in making informed decisions about their use, ensuring that the benefits of AI-assisted coding are maximized while minimizing associated risks. 

**Abstract (ZH)**: 将大型语言模型（LLMs）如GitHub Copilot、ChatGPT、Cursor AI和Codeium AI集成到软件开发中，已经彻底改变了编码格局，提供了显著的生产率提升、自动化和增强的调试能力。这些工具已被证明在生成代码片段、重构现有代码和为开发人员提供实时支持方面具有巨大的价值。然而，它们的广泛采用也提出了显著的挑战，特别是在安全漏洞、代码质量以及伦理问题方面。本文对人工智能驱动的编程工具的优势和风险进行了全面分析，基于用户反馈、安全分析和实用案例。我们探讨了这些工具可能复制不安全的编程实践、引入偏见以及生成不正确或不通顺代码（幻觉）的可能性。此外，我们讨论了数据泄露、知识产权侵权以及需要采取稳健的安全措施以减轻这些威胁的风险。通过比较这些工具的功能和性能，我们旨在指导开发人员做出明智的决策，确保在最大限度地利用人工智能辅助编程的优势的同时，最大程度地降低相关风险。 

---
# ChatGPT vs. DeepSeek: A Comparative Study on AI-Based Code Generation 

**Title (ZH)**: 基于ChatGPT与DeepSeek的AI代码生成比较研究 

**Authors**: Md Motaleb Hossen Manik  

**Link**: [PDF](https://arxiv.org/pdf/2502.18467)  

**Abstract**: Background: AI-powered code generation, fueled by Large Language Models (LLMs), is revolutionizing software development. Models like OpenAI's Codex and GPT-4, alongside DeepSeek, leverage vast code and natural language datasets. However, ensuring code quality, correctness, and managing complex tasks remains challenging, necessitating thorough evaluation. Methodology: This research compares ChatGPT (version o1) and DeepSeek (version R1) for Python code generation using online judge coding challenges. It evaluates correctness (online judge verdicts, up to three attempts), code quality (Pylint/Flake8), and efficiency (execution time/memory usage). Results: DeepSeek demonstrated higher correctness, particularly on algorithmic tasks, often achieving 'Accepted' on the first attempt. ChatGPT sometimes requires multiple attempts or failures. ChatGPT encountered fewer issues, used comparable or slightly less memory, consumed less execution times and wrote fewer lines of code. Conclusion: DeepSeek exhibited superior correctness in Python code generation, often requiring fewer attempts, suggesting an advantage in algorithmic problem-solving. Both models showed almost similar efficiency in execution time and memory use. Finally, this research provides insights for developers choosing AI coding assistants and informs future AI-driven software development research. 

**Abstract (ZH)**: 背景：以大型语言模型（LLMs）为驱动力的人工智能（AI）编码生成正在革新软件开发。像OpenAI的Codex和GPT-4这样的模型，以及DeepSeek，利用了大量的代码和自然语言数据集。然而，确保代码质量、正确性并有效处理复杂任务仍具有挑战性，因此需要进行彻底的评估。

方法：本研究将ChatGPT（版本o1）和DeepSeek（版本R1）用于Python代码生成的在线校判编程挑战，评估其正确性（在线校判结果，最多三次尝试）、代码质量（Pylint/Flake8）和效率（执行时间和内存使用情况）。

结果：DeepSeek在正确性上表现更优，尤其是在算法任务方面，往往能在第一次尝试就获得“通过”。ChatGPT有时需要多次尝试或失败。ChatGPT遇到的问题更少，使用的内存与DeepSeek相当或稍少，执行时间较短，生成的代码行数也较少。

结论：DeepSeek在Python代码生成的正确性方面表现出色，往往需要较少的尝试次数，这表明其在求解算法问题上有优势。两个模型在执行时间和内存使用方面显示出几乎相似的效率。最后，本研究为选择AI编程助手的开发者提供了见解，并为未来基于AI的软件开发研究提供了信息。 

---
# MLScent A tool for Anti-pattern detection in ML projects 

**Title (ZH)**: MLScent：一种用于检测机器学习项目中反模式的工具 

**Authors**: Karthik Shivashankar, Antonio Martini  

**Link**: [PDF](https://arxiv.org/pdf/2502.18466)  

**Abstract**: Machine learning (ML) codebases face unprecedented challenges in maintaining code quality and sustainability as their complexity grows exponentially. While traditional code smell detection tools exist, they fail to address ML-specific issues that can significantly impact model performance, reproducibility, and maintainability.
This paper introduces MLScent, a novel static analysis tool that leverages sophisticated Abstract Syntax Tree (AST) analysis to detect anti-patterns and code smells specific to ML projects.
MLScent implements 76 distinct detectors across major ML frameworks including TensorFlow (13 detectors), PyTorch (12 detectors), Scikit-learn (9 detectors), and Hugging Face (10 detectors), along with data science libraries like Pandas and NumPy (8 detectors each). The tool's architecture also integrates general ML smell detection (16 detectors), and specialized analysis for data preprocessing and model training workflows.
Our evaluation demonstrates MLScent's effectiveness through both quantitative classification metrics and qualitative assessment via user studies feedback with ML practitioners. Results show high accuracy in identifying framework-specific anti-patterns, data handling issues, and general ML code smells across real-world projects. 

**Abstract (ZH)**: 随着机器学习（ML）代码复杂性的指数级增长，维持代码质量和可持续性面临着前所未有的挑战。虽然传统的代码异味检测工具已经存在，但它们无法解决对模型性能、可再现性和可维护性有重大影响的特定于ML的问题。

本文介绍了一种名为MLScent的新颖静态分析工具，该工具利用复杂的抽象语法树（AST）分析来检测特定于ML项目的反模式和代码异味。

MLScent在TensorFlow（13个检测器）、PyTorch（12个检测器）、Scikit-learn（9个检测器）、Hugging Face（10个检测器）以及其他数据科学库Pandas和NumPy（每个8个检测器）等主要ML框架中实现了76个独特的检测器。该工具的架构还集成了针对一般ML代码异味检测的16个检测器，以及数据预处理和模型训练流程的专门分析。

我们的评估通过定量分类指标和用户研究反馈的定性评估，证明了MLScent的有效性。结果显示，MLScent在识别特定于框架的反模式、数据处理问题以及通用ML代码异味方面具有很高的准确性，这些在实际项目中得到了验证。 

---
