# Multi-Agent Verification: Scaling Test-Time Compute with Multiple Verifiers 

**Title (ZH)**: 多智能体验证：通过多种验证者扩展测试时的计算能力 

**Authors**: Shalev Lifshitz, Sheila A. McIlraith, Yilun Du  

**Link**: [PDF](https://arxiv.org/pdf/2502.20379)  

**Abstract**: By utilizing more computational resources at test-time, large language models (LLMs) can improve without additional training. One common strategy uses verifiers to evaluate candidate outputs. In this work, we propose a novel scaling dimension for test-time compute: scaling the number of verifiers. We introduce Multi-Agent Verification (MAV) as a test-time compute paradigm that combines multiple verifiers to improve performance. We propose using Aspect Verifiers (AVs), off-the-shelf LLMs prompted to verify different aspects of outputs, as one possible choice for the verifiers in a MAV system. AVs are a convenient building block for MAV since they can be easily combined without additional training. Moreover, we introduce BoN-MAV, a simple multi-agent verification algorithm that combines best-of-n sampling with multiple verifiers. BoN-MAV demonstrates stronger scaling patterns than self-consistency and reward model verification, and we demonstrate both weak-to-strong generalization, where combining weak verifiers improves even stronger LLMs, and self-improvement, where the same base model is used to both generate and verify outputs. Our results establish scaling the number of verifiers as a promising new dimension for improving language model performance at test-time. 

**Abstract (ZH)**: 通过在测试时利用更多的计算资源，大型语言模型（LLMs）可以在无需额外训练的情况下得到改进。一种常见的策略是使用验证器来评估候选输出。本研究提出了一种新的测试时计算维度：增加验证器的数量。我们介绍了一种名为多智能体验证（MAV）的测试时计算范式，该范式结合了多个验证器以提升性能。我们提出使用方面验证器（AVs），即预先配置以验证输出不同方面并可即插即用的现成语言模型，作为MAV系统中验证器的一种可能选择。由于AVs可以在无需额外训练的情况下轻松组合，因此它们成为构建MAV的理想组件。此外，我们引入了一种简单的多智能体验证算法BoN-MAV，该算法结合了最优-n采样和多个验证器。BoN-MAV在不同验证器数量上的性能表现优于自我一致性验证和奖励模型验证。我们证明了从弱到强的泛化能力，即结合弱验证器能提升更强的LLMs，并展示了自改进能力，即使用同一个基础模型同时生成和验证输出。我们的研究结果证明，增加验证器的数量是一个值得探索的新维度，有助于提高语言模型在测试时的性能。 

---
# EAIRA: Establishing a Methodology for Evaluating AI Models as Scientific Research Assistants 

**Title (ZH)**: EAIRA：建立评估人工智能模型作为科学研究助手的方法论 

**Authors**: Franck Cappello, Sandeep Madireddy, Robert Underwood, Neil Getty, Nicholas Lee-Ping Chia, Nesar Ramachandra, Josh Nguyen, Murat Keceli, Tanwi Mallick, Zilinghan Li, Marieme Ngom, Chenhui Zhang, Angel Yanguas-Gil, Evan Antoniuk, Bhavya Kailkhura, Minyang Tian, Yufeng Du, Yuan-Sen Ting, Azton Wells, Bogdan Nicolae, Avinash Maurya, M. Mustafa Rafique, Eliu Huerta, Bo Li, Ian Foster, Rick Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2502.20309)  

**Abstract**: Recent advancements have positioned AI, and particularly Large Language Models (LLMs), as transformative tools for scientific research, capable of addressing complex tasks that require reasoning, problem-solving, and decision-making. Their exceptional capabilities suggest their potential as scientific research assistants but also highlight the need for holistic, rigorous, and domain-specific evaluation to assess effectiveness in real-world scientific applications. This paper describes a multifaceted methodology for Evaluating AI models as scientific Research Assistants (EAIRA) developed at Argonne National Laboratory. This methodology incorporates four primary classes of evaluations. 1) Multiple Choice Questions to assess factual recall; 2) Open Response to evaluate advanced reasoning and problem-solving skills; 3) Lab-Style Experiments involving detailed analysis of capabilities as research assistants in controlled environments; and 4) Field-Style Experiments to capture researcher-LLM interactions at scale in a wide range of scientific domains and applications. These complementary methods enable a comprehensive analysis of LLM strengths and weaknesses with respect to their scientific knowledge, reasoning abilities, and adaptability. Recognizing the rapid pace of LLM advancements, we designed the methodology to evolve and adapt so as to ensure its continued relevance and applicability. This paper describes the methodology state at the end of February 2025. Although developed within a subset of scientific domains, the methodology is designed to be generalizable to a wide range of scientific domains. 

**Abstract (ZH)**: 近年来，人工智能，尤其是大型语言模型（LLMs），已被视为推动科学研究变革的工具，能够在需要推理、问题解决和决策的任务中发挥作用。它们的出色能力表明它们在科学研究助理方面具有巨大的潜力，但也强调了对其进行全面、严谨且领域特定的评估的必要性，以评估其在实际科学应用中的有效性。本文描述了阿贡国家实验室开发的一种多维度方法，用于评估人工智能模型作为科学研究助理（EAIRA）。该方法包含四种主要的评估类别。1) 多选题以评估事实记忆；2) 开放式回答以评估高级推理和问题解决能力；3) 实验室风格的实验，通过在受控环境中详细分析其作为研究助理的能力来评估其能力；4) 现场风格的实验，以在广泛的科学领域和应用中捕捉研究人员与LLM的交互情况。这些互补的方法能够全面分析LLMs在科学知识、推理能力和适应性方面的优势和劣势。考虑到LLMs快速发展的步伐，我们设计了该方法以便于其不断进化和适应，以确保其持续的相关性和适用性。本文描述了截至2025年2月底的方法状态。尽管该方法是在某些科学领域内开发的，但设计目的是使其能够广泛应用于各种科学领域。 

---
# Meta-Reasoner: Dynamic Guidance for Optimized Inference-time Reasoning in Large Language Models 

**Title (ZH)**: 元推理器：用于大型语言模型优化推理时推理的动态指导 

**Authors**: Yuan Sui, Yufei He, Tri Cao, Simeng Han, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2502.19918)  

**Abstract**: Large Language Models (LLMs) increasingly rely on prolonged reasoning chains to solve complex tasks. However, this trial-and-error approach often leads to high computational overhead and error propagation, where early mistakes can derail subsequent steps. To address these issues, we introduce Meta-Reasoner, a framework that dynamically optimizes inference-time reasoning by enabling LLMs to "think about how to think." Drawing inspiration from human meta-cognition and dual-process theory, Meta-Reasoner operates as a strategic advisor, decoupling high-level guidance from step-by-step generation. It employs "contextual multi-armed bandits" to iteratively evaluate reasoning progress, and select optimal strategies (e.g., backtrack, clarify ambiguity, restart from scratch, or propose alternative approaches), and reallocates computational resources toward the most promising paths. Our evaluations on mathematical reasoning and puzzles highlight the potential of dynamic reasoning chains to overcome inherent challenges in the LLM reasoning process and also show promise in broader applications, offering a scalable and adaptable solution for reasoning-intensive tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地依赖于长期的推理链来解决复杂的任务。然而，这种试错方法通常会导致计算开销高昂并产生错误传播的问题，即早期的错误可能会影响后续的步骤。为了解决这些问题，我们提出了一种名为Meta-Reasoner的框架，该框架通过使LLMs能够“思考如何思考”来动态优化推理过程。受到人类元认知和双重过程理论的启发，Meta-Reasoner充当一个策略顾问，将高层次的指导与逐步生成过程分离。它使用“上下文多臂老虎机”来逐步评估推理进展，并选择最优策略（例如回退、澄清歧义、从头开始或提出替代方法），并将计算资源重新分配到最有前途的路径上。我们对数学推理和谜题的评估突显了动态推理链在克服LLMs推理过程中的固有挑战方面的潜力，并表明在更广泛的应用中具有前景，提供了一种可扩展和适应性强的解决方案，适用于涉及大量推理的任务。 

---
# Optimus-2: Multimodal Minecraft Agent with Goal-Observation-Action Conditioned Policy 

**Title (ZH)**: Optimus-2：基于目标-观察-动作条件策略的多模态Minecraft代理 

**Authors**: Zaijing Li, Yuquan Xie, Rui Shao, Gongwei Chen, Dongmei Jiang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2502.19902)  

**Abstract**: Building an agent that can mimic human behavior patterns to accomplish various open-world tasks is a long-term goal. To enable agents to effectively learn behavioral patterns across diverse tasks, a key challenge lies in modeling the intricate relationships among observations, actions, and language. To this end, we propose Optimus-2, a novel Minecraft agent that incorporates a Multimodal Large Language Model (MLLM) for high-level planning, alongside a Goal-Observation-Action Conditioned Policy (GOAP) for low-level control. GOAP contains (1) an Action-guided Behavior Encoder that models causal relationships between observations and actions at each timestep, then dynamically interacts with the historical observation-action sequence, consolidating it into fixed-length behavior tokens, and (2) an MLLM that aligns behavior tokens with open-ended language instructions to predict actions auto-regressively. Moreover, we introduce a high-quality Minecraft Goal-Observation-Action (MGOA)} dataset, which contains 25,000 videos across 8 atomic tasks, providing about 30M goal-observation-action pairs. The automated construction method, along with the MGOA dataset, can contribute to the community's efforts to train Minecraft agents. Extensive experimental results demonstrate that Optimus-2 exhibits superior performance across atomic tasks, long-horizon tasks, and open-ended instruction tasks in Minecraft. 

**Abstract (ZH)**: 构建能够模拟人类行为模式的代理以完成各种开放世界任务是一项长期目标。为了使代理能够有效地跨多种任务学习行为模式，一个关键挑战在于建模观测、动作和语言之间的复杂关系。为此，我们提出了Optimus-2，这是一种结合了多模态大型语言模型（MLLM）进行高层次规划，并结合了目标-观测-动作条件策略（GOAP）进行低层次控制的新型Minecraft代理。GOAP 包含以下两个组成部分：(1) 行动导向的行为编码器，该编码器在每个时间步长中建模观测与动作之间的因果关系，然后与历史观测-动作序列动态交互，将其合并成固定长度的行为令牌；(2) MLLM，该模型将行为令牌与开放性语言指令对齐以自回归地预测动作。此外，我们还引入了一个高质量的Minecraft目标-观测-动作（MGOA）数据集，该数据集包含25,000个视频，横跨8个原子任务，提供了大约3000万个目标-观测-动作对。自动化构建方法以及MGOA数据集能够为Minecraft代理的训练工作做出贡献。广泛的实验结果表明，Optimus-2在Minecraft中的原子任务、长时任务和开放式指令任务中均表现出卓越的性能。 

---
# Agentic Mixture-of-Workflows for Multi-Modal Chemical Search 

**Title (ZH)**: 用于多模态化学搜索的代理工作流混合模型 

**Authors**: Tiffany J. Callahan, Nathaniel H. Park, Sara Capponi  

**Link**: [PDF](https://arxiv.org/pdf/2502.19629)  

**Abstract**: The vast and complex materials design space demands innovative strategies to integrate multidisciplinary scientific knowledge and optimize materials discovery. While large language models (LLMs) have demonstrated promising reasoning and automation capabilities across various domains, their application in materials science remains limited due to a lack of benchmarking standards and practical implementation frameworks. To address these challenges, we introduce Mixture-of-Workflows for Self-Corrective Retrieval-Augmented Generation (CRAG-MoW) - a novel paradigm that orchestrates multiple agentic workflows employing distinct CRAG strategies using open-source LLMs. Unlike prior approaches, CRAG-MoW synthesizes diverse outputs through an orchestration agent, enabling direct evaluation of multiple LLMs across the same problem domain. We benchmark CRAG-MoWs across small molecules, polymers, and chemical reactions, as well as multi-modal nuclear magnetic resonance (NMR) spectral retrieval. Our results demonstrate that CRAG-MoWs achieve performance comparable to GPT-4o while being preferred more frequently in comparative evaluations, highlighting the advantage of structured retrieval and multi-agent synthesis. By revealing performance variations across data types, CRAG-MoW provides a scalable, interpretable, and benchmark-driven approach to optimizing AI architectures for materials discovery. These insights are pivotal in addressing fundamental gaps in benchmarking LLMs and autonomous AI agents for scientific applications. 

**Abstract (ZH)**: 巨大的材料设计空间要求采用创新策略整合多学科科学知识并优化材料发现过程。虽然大型语言模型（LLMs）在诸多领域展现了有希望的推理和自动化能力，但它们在材料科学中的应用仍然受限，主要是由于缺乏基准测试标准和实际实施框架。为应对这些挑战，我们引入了一种新的范式——混合工作流自纠正检索增强生成（Mixture-of-Workflows for Self-Corrective Retrieval-Augmented Generation, CRAG-MoW）——该范式使用开源LLMs协调多个独立的CRAG策略。与之前的 approached 不同，CRAG-MoW 通过一个调和代理将不同的输出进行集成，从而使多种LLMs能够直接在同一问题域内进行评估。我们对小分子、聚合物、化学反应以及多模态核磁共振（NMR）光谱检索进行了基准测试。结果显示，CRAG-MoWs 在性能上与GPT-4o相当，并且在比较评估中被更频繁地优先选择，这突显了结构化检索和多代理合成的优势。通过揭示不同数据类型下的性能差异，CRAG-MoW 提供了一种可扩展、可解释并以基准驱动的方法，以优化材料发现中的AI架构。这些见解对于弥合LLMs和自主AI代理在科学应用领域的基准测试基本差距至关重要。 

---
# Self-rewarding correction for mathematical reasoning 

**Title (ZH)**: 数学推理中的自我奖励修正 

**Authors**: Wei Xiong, Hanning Zhang, Chenlu Ye, Lichang Chen, Nan Jiang, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19613)  

**Abstract**: We study self-rewarding reasoning large language models (LLMs), which can simultaneously generate step-by-step reasoning and evaluate the correctness of their outputs during the inference time-without external feedback. This integrated approach allows a single model to independently guide its reasoning process, offering computational advantages for model deployment. We particularly focus on the representative task of self-correction, where models autonomously detect errors in their responses, revise outputs, and decide when to terminate iterative refinement loops. To enable this, we propose a two-staged algorithmic framework for constructing self-rewarding reasoning models using only self-generated data. In the first stage, we employ sequential rejection sampling to synthesize long chain-of-thought trajectories that incorporate both self-rewarding and self-correction mechanisms. Fine-tuning models on these curated data allows them to learn the patterns of self-rewarding and self-correction. In the second stage, we further enhance the models' ability to assess response accuracy and refine outputs through reinforcement learning with rule-based signals. Experiments with Llama-3 and Qwen-2.5 demonstrate that our approach surpasses intrinsic self-correction capabilities and achieves performance comparable to systems that rely on external reward models. 

**Abstract (ZH)**: 我们研究了一种自我奖励推理的大语言模型（LLMs），它能够在推理过程中同时生成逐步推理过程并在输出时评估其正确性，无需外部反馈。这种集成方法允许单个模型独立指导其推理过程，为模型部署提供了计算上的优势。我们特别关注自我修正任务，即模型能够自主检测其回复中的错误、修正输出，并决定何时终止迭代精炼循环。为了实现这一目标，我们提出了一种仅使用自动生成数据的两阶段算法框架，以构建自我奖励推理模型。在第一阶段，我们采用序列拒绝采样方法合成包含自我奖励和自我修正机制的长推理轨迹。通过在这些精心筛选的数据上微调模型，可以让模型学习自我奖励和自我修正的模式。在第二阶段，我们进一步通过基于规则的信号进行强化学习，增强模型评估响应准确性和精炼输出的能力。实验结果显示，我们的方法超越了内在的自我修正能力，并且达到了与依赖外部奖励模型的系统相当的性能。 

---
# Program Synthesis Dialog Agents for Interactive Decision-Making 

**Title (ZH)**: 交互决策中的程序合成对话代理 

**Authors**: Matthew Toles, Nikhil Balwani, Rattandeep Singh, Valentina Giulia Sartori Rodriguez, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.19610)  

**Abstract**: Many real-world eligibility problems, ranging from medical diagnosis to tax planning, can be mapped to decision problems expressed in natural language, wherein a model must make a binary choice based on user features. Large-scale domains such as legal codes or frequently updated funding opportunities render human annotation (e.g., web forms or decision trees) impractical, highlighting the need for agents that can automatically assist in decision-making. Since relevant information is often only known to the user, it is crucial that these agents ask the right questions. As agents determine when to terminate a conversation, they face a trade-off between accuracy and the number of questions asked, a key metric for both user experience and cost. To evaluate this task, we propose BeNYfits, a new benchmark for determining user eligibility for multiple overlapping social benefits opportunities through interactive decision-making. Our experiments show that current language models struggle with frequent hallucinations, with GPT-4o scoring only 35.7 F1 using a ReAct-style chain-of-thought. To address this, we introduce ProADA, a novel approach that leverages program synthesis to assist in decision-making by mapping dialog planning to a code generation problem and using gaps in structured data to determine the best next action. Our agent, ProADA, improves the F1 score to 55.6 while maintaining nearly the same number of dialog turns. 

**Abstract (ZH)**: 许多实际应用中的资格问题，从医疗诊断到税务规划，都可以映射为自然语言表达的决策问题，在这些问题中，模型必须基于用户特征做出二元选择。在法律条文或经常更新的资金机会等大规模领域中，手动标注（例如网页表单或决策树）变得不切实际，突显了能够自动辅助决策的代理的需求。由于相关信息通常只有用户知晓，因此这些代理提出正确问题至关重要。随着代理决定何时终止对话，他们需要在准确性和提出的问题数量之间进行权衡，这是用户经验和成本的重要指标。为了评估这一任务，我们提出了BeNYfits基准，这是一个新的交互式决策方案，用于确定用户是否符合多个重叠的社会福利机会的资格。实验结果显示，当前的语言模型在频繁出现幻觉方面表现不佳，GPT-4o仅获得35.7的F1得分，使用ReAct风格的推理链。为了解决这个问题，我们提出了ProADA，这是一种新型方法，利用程序合成来辅助决策，通过将对话规划映射为代码生成问题，并利用结构化数据中的空白来决定最佳下一步行动。我们的代理ProADA将F1得分提高到55.6，同时基本保持了相同的对话轮次。 

---
# Sim-to-Real Reinforcement Learning for Vision-Based Dexterous Manipulation on Humanoids 

**Title (ZH)**: 基于视觉的灵巧操作中类转真实强化学习方法研究（应用于类人机器人） 

**Authors**: Toru Lin, Kartik Sachdev, Linxi Fan, Jitendra Malik, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20396)  

**Abstract**: Reinforcement learning has delivered promising results in achieving human- or even superhuman-level capabilities across diverse problem domains, but success in dexterous robot manipulation remains limited. This work investigates the key challenges in applying reinforcement learning to solve a collection of contact-rich manipulation tasks on a humanoid embodiment. We introduce novel techniques to overcome the identified challenges with empirical validation. Our main contributions include an automated real-to-sim tuning module that brings the simulated environment closer to the real world, a generalized reward design scheme that simplifies reward engineering for long-horizon contact-rich manipulation tasks, a divide-and-conquer distillation process that improves the sample efficiency of hard-exploration problems while maintaining sim-to-real performance, and a mixture of sparse and dense object representations to bridge the sim-to-real perception gap. We show promising results on three humanoid dexterous manipulation tasks, with ablation studies on each technique. Our work presents a successful approach to learning humanoid dexterous manipulation using sim-to-real reinforcement learning, achieving robust generalization and high performance without the need for human demonstration. 

**Abstract (ZH)**: 强化学习在多个问题领域取得了令人鼓舞的结果，实现了接近甚至超越人类的性能，但在灵巧机器人操作方面取得的成功仍然有限。本研究探讨了将强化学习应用于解决人形机器人 embodiment 上的接触丰富操作任务的关键挑战，并引入了新的技术以进行实证验证。我们的主要贡献包括：一种自动化的实到仿真实验模块，使仿真的环境更加接近真实世界；一种通用的奖励设计方案，简化了长期时间接触丰富操作任务的奖励工程；一种分而治之的教学过程，提高了难以探索问题的学习效率，同时保持了仿真的真实性能；以及一种稀疏和密集表示的混合，以弥合仿真与现实之间的感知差距。我们在三个灵巧操作任务中展示了有前景的结果，并对每种技术进行了消融研究。我们的工作提出了一种利用仿真到现实的强化学习进行人形灵巧操作学习的成功方法，实现了鲁棒泛化和高性能，无需人类演示。 

---
# Bridging Legal Knowledge and AI: Retrieval-Augmented Generation with Vector Stores, Knowledge Graphs, and Hierarchical Non-negative Matrix Factorization 

**Title (ZH)**: 法律知识与人工智能融合：基于向量存储、知识图谱和层次非负矩阵分解的检索增强生成 

**Authors**: Ryan C. Barron, Maksim E. Eren, Olga M. Serafimova, Cynthia Matuszek, Boian S. Alexandrov  

**Link**: [PDF](https://arxiv.org/pdf/2502.20364)  

**Abstract**: Agentic Generative AI, powered by Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG), Knowledge Graphs (KGs), and Vector Stores (VSs), represents a transformative technology applicable to specialized domains such as legal systems, research, recommender systems, cybersecurity, and global security, including proliferation research. This technology excels at inferring relationships within vast unstructured or semi-structured datasets. The legal domain here comprises complex data characterized by extensive, interrelated, and semi-structured knowledge systems with complex relations. It comprises constitutions, statutes, regulations, and case law. Extracting insights and navigating the intricate networks of legal documents and their relations is crucial for effective legal research. Here, we introduce a generative AI system that integrates RAG, VS, and KG, constructed via Non-Negative Matrix Factorization (NMF), to enhance legal information retrieval and AI reasoning and minimize hallucinations. In the legal system, these technologies empower AI agents to identify and analyze complex connections among cases, statutes, and legal precedents, uncovering hidden relationships and predicting legal trends-challenging tasks that are essential for ensuring justice and improving operational efficiency. Our system employs web scraping techniques to systematically collect legal texts, such as statutes, constitutional provisions, and case law, from publicly accessible platforms like Justia. It bridges the gap between traditional keyword-based searches and contextual understanding by leveraging advanced semantic representations, hierarchical relationships, and latent topic discovery. This framework supports legal document clustering, summarization, and cross-referencing, for scalable, interpretable, and accurate retrieval for semi-structured data while advancing computational law and AI. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的代理生成人工智能，借助检索增强生成（RAG）、知识图谱（KGs）和向量存储（VSs），代表了一种变革性的技术，适用于诸如法律系统、研究、推荐系统、网络安全以及大规模安全，包括扩散研究等专门领域。该技术擅长推断大量非结构化或半结构化数据集中的关系。这里的法律领域包括复杂的数据，这些数据具有广泛的、相互关联的和半结构化的知识系统，具有复杂的关系。它包括宪法、法律法规、规章制度和判例法。从复杂的法律文件和其关系中提取洞察力并导航它们的网络对于有效的法律研究至关重要。为了解决这个问题，我们介绍了一种集成了RAG、VS和KG的生成AI系统，通过非负矩阵分解（NMF）进行构建，以增强法律信息检索和AI推理并减少幻觉。在法律系统中，这些技术赋予AI代理识别和分析案件、法律条文和先例之间的复杂联系的能力，揭示隐藏的关系并预测法律趋势，这是确保公正和提高运营效率必不可少的任务。我们的系统采用网页抓取技术系统地收集法律文本，如法律法规、宪法条款和判例法，来自类似于Justia的公共访问平台。它通过利用先进的语义表示、层次关系和潜在主题发现，弥补了传统关键词搜索与上下文理解之间的差距。该框架支持法律文件聚类、摘要和跨参照，以实现半结构化数据的大规模、可解释和准确检索，从而推动计算法学和人工智能的发展。 

---
# Deep Reinforcement Learning based Autonomous Decision-Making for Cooperative UAVs: A Search and Rescue Real World Application 

**Title (ZH)**: 基于深度强化学习的自主决策方法在协作无人机搜救应用中的研究 

**Authors**: Thomas Hickling, Maxwell Hogan, Abdulla Tammam, Nabil Aouf  

**Link**: [PDF](https://arxiv.org/pdf/2502.20326)  

**Abstract**: This paper proposes a holistic framework for autonomous guidance, navigation, and task distribution among multi-drone systems operating in Global Navigation Satellite System (GNSS)-denied indoor settings. We advocate for a Deep Reinforcement Learning (DRL)-based guidance mechanism, utilising the Twin Delayed Deep Deterministic Policy Gradient algorithm. To improve the efficiency of the training process, we incorporate an Artificial Potential Field (APF)-based reward structure, enabling the agent to refine its movements, thereby promoting smoother paths and enhanced obstacle avoidance in indoor contexts. Furthermore, we tackle the issue of task distribution among cooperative UAVs through a DRL-trained Graph Convolutional Network (GCN). This GCN represents the interactions between drones and tasks, facilitating dynamic and real-time task allocation that reflects the current environmental conditions and the capabilities of the drones. Such an approach fosters effective coordination and collaboration among multiple drones during search and rescue operations or other exploratory endeavours. Lastly, to ensure precise odometry in environments lacking GNSS, we employ Light Detection And Ranging Simultaneous Localisation and Mapping complemented by a depth camera to mitigate the hallway problem. This integration offers robust localisation and mapping functionalities, thereby enhancing the systems dependability in indoor navigation. The proposed multi-drone framework not only elevates individual navigation capabilities but also optimises coordinated task allocation in complex, obstacle-laden environments. Experimental evaluations conducted in a setup tailored to meet the requirements of the NATO Sapience Autonomous Cooperative Drone Competition demonstrate the efficacy of the proposed system, yielding outstanding results and culminating in a first-place finish in the 2024 Sapience competition. 

**Abstract (ZH)**: 本文提出了一种整体框架，用于在拒绝全球导航卫星系统（GNSS）的室内环境中进行多无人机系统的自主导航、制导和任务分配。我们主张采用基于深度强化学习（DRL）的制导机制，并利用双延迟深度确定性策略梯度（TD3）算法。为了提高训练效率，我们引入了一种基于人工势场（APF）的奖励结构，使智能体能够优化其移动方式，从而在室内环境中促进更平滑的路径和更有效的障碍物规避。此外，我们通过基于DRL训练的图卷积网络（GCN）来解决协作无人机间任务分配的问题。此GCN表示无人机与任务之间的交互关系，从而实现动态和实时的任务分配，反映当前的环境状况和无人机的能力。这种方法在搜索与救援等探索性任务中促进了多无人机之间的有效协调与合作。最后，为了在没有GNSS信号的环境中确保精确的里程计，我们采用了结合深度相机的激光探测与测距（LIDAR）同步定位与地图构建（SLAM），以解决走廊问题。这种集成提供了稳健的定位与绘图功能，从而提高了系统在室内导航中的可靠性。所提出的多无人机框架不仅提升了个体的导航能力，还优化了复杂、障碍多的环境中的协作任务分配。在专门为符合北约Sapience自主协同无人机比赛要求的情境下进行的实验评估证明了所提系统的有效性，取得了优秀的结果，并在2024年Sapience比赛中获得第一名。 

---
# M^3Builder: A Multi-Agent System for Automated Machine Learning in Medical Imaging 

**Title (ZH)**: M^3Builder：一种用于医学影像领域自动化机器学习的多智能体系统 

**Authors**: Jinghao Feng, Qiaoyu Zheng, Chaoyi Wu, Ziheng Zhao, Ya Zhang, Yanfeng Wang, Weidi Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.20301)  

**Abstract**: Agentic AI systems have gained significant attention for their ability to autonomously perform complex tasks. However, their reliance on well-prepared tools limits their applicability in the medical domain, which requires to train specialized models. In this paper, we make three contributions: (i) We present M3Builder, a novel multi-agent system designed to automate machine learning (ML) in medical imaging. At its core, M3Builder employs four specialized agents that collaborate to tackle complex, multi-step medical ML workflows, from automated data processing and environment configuration to self-contained auto debugging and model training. These agents operate within a medical imaging ML workspace, a structured environment designed to provide agents with free-text descriptions of datasets, training codes, and interaction tools, enabling seamless communication and task execution. (ii) To evaluate progress in automated medical imaging ML, we propose M3Bench, a benchmark comprising four general tasks on 14 training datasets, across five anatomies and three imaging modalities, covering both 2D and 3D data. (iii) We experiment with seven state-of-the-art large language models serving as agent cores for our system, such as Claude series, GPT-4o, and DeepSeek-V3. Compared to existing ML agentic designs, M3Builder shows superior performance on completing ML tasks in medical imaging, achieving a 94.29% success rate using Claude-3.7-Sonnet as the agent core, showing huge potential towards fully automated machine learning in medical imaging. 

**Abstract (ZH)**: 自主智能代理系统因其自主完成复杂任务的能力而备受关注。然而，它们对精心准备的工具的依赖限制了其在医学领域的应用，而医学领域需要训练专门的模型。本论文作出了三项贡献：（i）我们提出了一种名为M3Builder的新型多智能体系统，旨在自动化医学影像领域的机器学习（ML）流程。M3Builder的核心是一个由四个专门智能体组成的协作框架，它们能够处理从自动数据处理和环境配置到自我封装的自动调试和模型训练等复杂、多步骤的医学影像ML工作流程。这些智能体在一种结构化的医学影像ML工作环境中运行，该环境为智能体提供了数据集、训练代码和交互工具的自由文本描述，从而实现无缝的通信和任务执行。（ii）为了评估自动化医学影像ML的进展，我们提出了M3Bench基准测试，该基准测试包含跨越五个解剖部位、三种成像模态（包括2D和3D数据）的14个训练数据集上的四个通用任务。（iii）我们试验了七种最先进的大规模语言模型作为系统的核心智能体，例如Claude系列、GPT-4o和DeepSeek-V3。与现有的ML智能体设计相比，M3Builder在医学影像领域的ML任务完成上表现出更优越的性能，使用Claude-3.7-Sonnet作为核心智能体时，任务成功率达94.29%，展现出向完全自动化的医学影像机器学习方向的巨大潜力。 

---
# Highly Parallelized Reinforcement Learning Training with Relaxed Assignment Dependencies 

**Title (ZH)**: 高并行化的强化学习训练与宽松的任务依赖关系 

**Authors**: Zhouyu He, Peng Qiao, Rongchun Li, Yong Dou, Yusong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2502.20190)  

**Abstract**: As the demands for superior agents grow, the training complexity of Deep Reinforcement Learning (DRL) becomes higher. Thus, accelerating training of DRL has become a major research focus. Dividing the DRL training process into subtasks and using parallel computation can effectively reduce training costs. However, current DRL training systems lack sufficient parallelization due to data assignment between subtask components. This assignment issue has been ignored, but addressing it can further boost training efficiency. Therefore, we propose a high-throughput distributed RL training system called TianJi. It relaxes assignment dependencies between subtask components and enables event-driven asynchronous communication. Meanwhile, TianJi maintains clear boundaries between subtask components. To address convergence uncertainty from relaxed assignment dependencies, TianJi proposes a distributed strategy based on the balance of sample production and consumption. The strategy controls the staleness of samples to correct their quality, ensuring convergence. We conducted extensive experiments. TianJi achieves a convergence time acceleration ratio of up to 4.37 compared to related comparison systems. When scaled to eight computational nodes, TianJi shows a convergence time speedup of 1.6 and a throughput speedup of 7.13 relative to XingTian, demonstrating its capability to accelerate training and scalability. In data transmission efficiency experiments, TianJi significantly outperforms other systems, approaching hardware limits. TianJi also shows effectiveness in on-policy algorithms, achieving convergence time acceleration ratios of 4.36 and 2.95 compared to RLlib and XingTian. TianJi is accessible at this https URL. 

**Abstract (ZH)**: 随着对更优秀代理的需求增加，深度强化学习（DRL）的训练复杂性也随之提高。因此，加速DRL的训练已成为主要研究方向。将DRL训练过程分解为子任务并利用并行计算可以有效降低训练成本。然而，当前的DRL训练系统由于子任务组件之间的数据分配不足，缺乏充分的并行化。这一分配问题已经被忽视，但解决这个问题可以进一步提升训练效率。因此，我们提出了一种高吞吐量的分布式强化学习训练系统——天机（TianJi）。该系统放松了子任务组件之间的数据分配依赖，并支持事件驱动的异步通信。同时，天机保持子任务组件之间的清晰边界。为了解决放松数据分配依赖带来的收敛不确定性，天机提出了一种基于样本生产和消费平衡的分布式策略。该策略控制样本的陈旧度，以纠正其质量，确保收敛。我们进行了广泛实验。相比相关比较系统，天机将收敛时间加速比提高到4.37。在扩展到八台计算节点时，与星天（XingTian）相比，天机的收敛时间加速比为1.6，吞吐量加速比为7.13，展示了其加速训练和扩展能力。在数据传输效率实验中，天机显著优于其他系统，接近硬件极限。天机在使用策略上的算法上也表现出有效性，相比RLlib和星天，其收敛时间加速比分别为4.36和2.95。天机的源代码可通过以下链接访问：[这里](提供链接)。 

---
# Collab-Overcooked: Benchmarking and Evaluating Large Language Models as Collaborative Agents 

**Title (ZH)**: Collab-Overcooked: 评估大型语言模型作为协作代理的基准测试与评估 

**Authors**: Haochen Sun, Shuwen Zhang, Lei Ren, Hao Xu, Hao Fu, Caixia Yuan, Xiaojie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20073)  

**Abstract**: Large language models (LLMs) based agent systems have made great strides in real-world applications beyond traditional NLP tasks. This paper proposes a new LLM-powered Multi-Agent System (LLM-MAS) benchmark, Collab-Overcooked, built on the popular Overcooked-AI game with more applicable and challenging tasks in interactive environments. Collab-Overcooked extends existing benchmarks from two novel perspectives. First, it provides a multi-agent framework supporting diverse tasks and objectives and encourages collaboration through natural language communication. Second, it introduces a spectrum of process-oriented evaluation metrics to assess the fine-grained collaboration capabilities of different LLM agents, a dimension often overlooked in prior work. We conduct extensive experiments over 10 popular LLMs and show that, while the LLMs present a strong ability in goal interpretation, there is a significant discrepancy in active collaboration and continuous adaption that are critical for efficiently fulfilling complicated tasks. Notably, we highlight the strengths and weaknesses in LLM-MAS and provide insights for improving and evaluating LLM-MAS on a unified and open-sourced benchmark. Environments, 30 open-ended tasks, and an integrated evaluation package are now publicly available at this https URL. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的代理系统已经在超出传统NLP任务的实际应用中取得了显著进展。本文提出了一种新的LLM赋能的多代理系统（LLM-MAS）基准测试——Collab-Overcooked，该基准系统基于流行的Overcooked-AI游戏，并引入了更多适用且具有挑战性的交互环境任务。Collab-Overcooked 从两个新颖的角度扩展了现有的基准测试。首先，它提供了一个支持多种任务和目标的多代理框架，并通过自然语言通信促进协作。其次，它引入了一系列过程导向的评估指标，用以评估不同LLM代理的细粒度协作能力，这是以往工作中经常被忽视的一个维度。我们对10种流行的LLM进行了广泛的实验，并展示了虽然这些LLM在目标理解方面表现出强大的能力，但在主动协作和持续适应性方面仍存在显著差异，这些能力对于高效完成复杂任务至关重要。值得注意的是，我们指出了LLM-MAS的优势和不足，并提供了一种统一和开源的基准来改进和评估LLM-MAS。当前，环境、30项开放任务以及集成评估包已公开，访问链接为：this https URL。 

---
# MIND: Towards Immersive Psychological Healing with Multi-agent Inner Dialogue 

**Title (ZH)**: MIND：走向多代理内心对话的沉浸式心理疗愈 

**Authors**: Yujia Chen, Changsong Li, Yiming Wang, Qingqing Xiao, Nan Zhang, Zifan Kong, Peng Wang, Binyu Yan  

**Link**: [PDF](https://arxiv.org/pdf/2502.19860)  

**Abstract**: Mental health issues are worsening in today's competitive society, such as depression and anxiety. Traditional healings like counseling and chatbots fail to engage effectively, they often provide generic responses lacking emotional depth. Although large language models (LLMs) have the potential to create more human-like interactions, they still struggle to capture subtle emotions. This requires LLMs to be equipped with human-like adaptability and warmth. To fill this gap, we propose the MIND (Multi-agent INner Dialogue), a novel paradigm that provides more immersive psychological healing environments. Considering the strong generative and role-playing ability of LLM agents, we predefine an interactive healing framework and assign LLM agents different roles within the framework to engage in interactive inner dialogues with users, thereby providing an immersive healing experience. We conduct extensive human experiments in various real-world healing dimensions, and find that MIND provides a more user-friendly experience than traditional paradigms. This demonstrates that MIND effectively leverages the significant potential of LLMs in psychological healing. 

**Abstract (ZH)**: 当今竞争激烈的社会中，心理健康问题如抑郁和焦虑正在加剧。传统的治疗方法，如咨询和聊天机器人，往往无法有效吸引患者，并且常常提供缺乏情感深度的通用回应。尽管大型语言模型（LLMs）有可能创造更加人机互动的体验，但它们仍然难以捕捉微妙的情感。因此，亟需让LLMs具备类似人类的适应性和暖意。为填补这一空白，我们提出了一种名为MIND（Multi-agent INner Dialogue）的新颖范式，旨在提供更沉浸式的心理疗愈环境。考虑到LLM代理的强大生成能力和角色扮演能力，我们预先定义了一个互动疗愈框架，并将不同的角色分配给LLM代理，使其能够与用户进行互动的内心对话，从而提供更加沉浸式的疗愈体验。我们在多个现实世界的疗愈维度中进行了广泛的人类实验，并发现MIND为用户提供了一种更为友好且更具沉浸感的体验。这表明，MIND能够有效利用LLMs在心理疗愈领域的巨大潜力。 

---
# Exponential Topology-enabled Scalable Communication in Multi-agent Reinforcement Learning 

**Title (ZH)**: 基于拓扑结构的指数级可扩展通信在多智能体强化学习中的应用 

**Authors**: Xinran Li, Xiaolu Wang, Chenjia Bai, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19717)  

**Abstract**: In cooperative multi-agent reinforcement learning (MARL), well-designed communication protocols can effectively facilitate consensus among agents, thereby enhancing task performance. Moreover, in large-scale multi-agent systems commonly found in real-world applications, effective communication plays an even more critical role due to the escalated challenge of partial observability compared to smaller-scale setups. In this work, we endeavor to develop a scalable communication protocol for MARL. Unlike previous methods that focus on selecting optimal pairwise communication links-a task that becomes increasingly complex as the number of agents grows-we adopt a global perspective on communication topology design. Specifically, we propose utilizing the exponential topology to enable rapid information dissemination among agents by leveraging its small-diameter and small-size properties. This approach leads to a scalable communication protocol, named ExpoComm. To fully unlock the potential of exponential graphs as communication topologies, we employ memory-based message processors and auxiliary tasks to ground messages, ensuring that they reflect global information and benefit decision-making. Extensive experiments on large-scale cooperative benchmarks, including MAgent and Infrastructure Management Planning, demonstrate the superior performance and robust zero-shot transferability of ExpoComm compared to existing communication strategies. The code is publicly available at this https URL. 

**Abstract (ZH)**: 在合作多智能体强化学习（MARL）中，精心设计的通信协议可以有效地促进智能体之间的共识，从而提高任务性能。此外，在现实应用中常见的大规模多智能体系统中，由于与小规模设置相比更容易受到部分可观测性的挑战，有效的通信变得更加关键。在此项工作中，我们致力于开发一种可扩展的通信协议，以应对MARL中的通信挑战。不同于以往方法主要集中在选择最优的成对通信连接——随着智能体数量的增长，这项任务变得越来越复杂——我们从全局角度设计通信拓扑结构。具体而言，我们提出了利用指数拓扑结构来通过其直径小和规模小的特性促进快速信息传播。这种方法导致了一个可扩展的通信协议，名为ExpoComm。为了充分发挥指数图作为通信拓扑结构的潜力，我们采用了基于内存的消息处理器和辅助任务来确保消息能够反映全局信息，并促进决策过程。在MAgent和基础设施管理规划等大规模合作基准测试中的广泛实验表明，ExpoComm在现有的通信策略中具有优越的性能和稳健的零样本迁移能力。该代码在本链接公开：https://github.com/your-username/expocomm。 

---
# Med-RLVR: Emerging Medical Reasoning from a 3B base model via reinforcement Learning 

**Title (ZH)**: Med-RLVR：通过强化学习从3B基础模型中涌现医学推理 

**Authors**: Sheng Zhang, Qianchu Liu, Guanghui Qin, Tristan Naumann, Hoifung Poon  

**Link**: [PDF](https://arxiv.org/pdf/2502.19655)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) has recently gained attention for its ability to elicit self-evolved reasoning capabilitie from base language models without explicit reasoning supervisions, as demonstrated by DeepSeek-R1. While prior work on RLVR has primarily focused on mathematical and coding domains, its applicability to other tasks and domains remains unexplored. In this work, we investigate whether medical reasoning can emerge from RLVR. We introduce Med-RLVR as an initial study of RLVR in the medical domain leveraging medical multiple-choice question answering (MCQA) data as verifiable labels. Our results demonstrate that RLVR is not only effective for math and coding but also extends successfully to medical question answering. Notably, Med-RLVR achieves performance comparable to traditional supervised fine-tuning (SFT) on in-distribution tasks while significantly improving out-of-distribution generalization, with an 8-point accuracy gain. Further analysis of training dynamics reveals that, with no explicit reasoning supervision, reasoning emerges from the 3B-parameter base model. These findings underscore the potential of RLVR in domains beyond math and coding, opening new avenues for its application in knowledge-intensive fields such as medicine. 

**Abstract (ZH)**: 验证奖励的强化学习（RLVR）最近因其能够从基础语言模型中激发自我进化的推理能力而引起关注，而不需要显式的推理监督，如DeepSeek-R1所证明的那样。尽管RLVR的早期工作主要集中在数学和编程领域，但其在其他任务和领域的应用尚未得到探索。本研究旨在探讨RLVR是否能在医学推理中发挥作用。我们引入了Med-RLVR作为在医学领域利用医学选择题答案（MCQA）数据作为验证标签的RLVR初步研究。我们的结果表明，除了数学和编程外，RLVR也能成功应用于医学问答任务。值得注意的是，Med-RLVR在同分布任务上的性能与传统的监督微调（SFT）相当，但其在异分布泛化方面的表现显著提高，准确率提高了8个百分点。进一步的训练动力学分析表明，在没有任何显式推理监督的情况下，推理能力从3B参数的基础模型中自发产生。这些发现表明，RLVR在数学和编程领域之外也具有潜在的应用价值，为在知识密集型领域如医学中应用RLVR开辟了新的途径。 

---
# Robust Gymnasium: A Unified Modular Benchmark for Robust Reinforcement Learning 

**Title (ZH)**: 稳健的健身房：统一的模块化稳健强化学习基准测试 

**Authors**: Shangding Gu, Laixi Shi, Muning Wen, Ming Jin, Eric Mazumdar, Yuejie Chi, Adam Wierman, Costas Spanos  

**Link**: [PDF](https://arxiv.org/pdf/2502.19652)  

**Abstract**: Driven by inherent uncertainty and the sim-to-real gap, robust reinforcement learning (RL) seeks to improve resilience against the complexity and variability in agent-environment sequential interactions. Despite the existence of a large number of RL benchmarks, there is a lack of standardized benchmarks for robust RL. Current robust RL policies often focus on a specific type of uncertainty and are evaluated in distinct, one-off environments. In this work, we introduce Robust-Gymnasium, a unified modular benchmark designed for robust RL that supports a wide variety of disruptions across all key RL components-agents' observed state and reward, agents' actions, and the environment. Offering over sixty diverse task environments spanning control and robotics, safe RL, and multi-agent RL, it provides an open-source and user-friendly tool for the community to assess current methods and foster the development of robust RL algorithms. In addition, we benchmark existing standard and robust RL algorithms within this framework, uncovering significant deficiencies in each and offering new insights. 

**Abstract (ZH)**: 受到固有的不确定性和仿真到现实差距的驱使，鲁棒强化学习（RL）旨在提高代理在与环境进行顺序交互时对其复杂性和变化性的抗性。尽管存在大量的RL基准测试，但鲁棒RL的标准基准仍然缺乏。当前的鲁棒RL策略往往专注于特定类型的不确定性，并且在一个接一个独立的环境中进行评估。在本工作中，我们引入了Robust-Gymnasium，这是一个统一的模块化基准测试，专为鲁棒RL设计，支持在所有关键的RL组件（代理观测到的状态和奖励、代理的动作以及环境）上广泛的干扰。该基准测试提供了六十多个涵盖控制、机器人技术、安全RL和多代理RL的多样任务环境，为社区提供了一个开源且用户友好的工具，用于评估当前的方法并促进鲁棒RL算法的发展。此外，我们在此框架内对现有的标准RL和鲁棒RL算法进行了基准测试，揭示了每种算法中存在的显著缺陷，并提供了新的见解。 

---
# Winning Big with Small Models: Knowledge Distillation vs. Self-Training for Reducing Hallucination in QA Agents 

**Title (ZH)**: 小模型的一大优势：知识蒸馏与自我训练在减少问答代理幻觉中的比较 

**Authors**: Ashley Lewis, Michael White, Jing Liu, Toshiaki Koike-Akino, Kieran Parsons, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19545)  

**Abstract**: The deployment of Large Language Models (LLMs) in customer support is constrained by hallucination-generating false information-and the high cost of proprietary models. To address these challenges, we propose a retrieval-augmented question-answering (QA) pipeline and explore how to balance human input and automation. Using a dataset of questions about a Samsung Smart TV user manual, we demonstrate that synthetic data generated by LLMs outperforms crowdsourced data in reducing hallucination in finetuned models. We also compare self-training (fine-tuning models on their own outputs) and knowledge distillation (fine-tuning on stronger models' outputs, e.g., GPT-4o), and find that self-training achieves comparable hallucination reduction. We conjecture that this surprising finding can be attributed to increased exposure bias issues in the knowledge distillation case and support this conjecture with post hoc analysis. We also improve robustness to unanswerable questions and retrieval failures with contextualized "I don't know" responses. These findings show that scalable, cost-efficient QA systems can be built using synthetic data and self-training with open-source models, reducing reliance on proprietary tools or costly human annotations. 

**Abstract (ZH)**: 将大型语言模型（LLMs）部署到客户服务中受到生成虚假信息的幻觉和专有模型成本高的限制。为解决这些挑战，我们提出了一种检索增强的问答（QA）流水线，并探讨了如何在人工输入和自动化之间进行平衡。利用关于三星智能电视用户手册的问题数据集，我们展示了由LLMs生成的合成数据在减少微调模型中的幻觉方面优于众包数据。我们还将自我训练（在其自身输出上微调模型）与知识蒸馏（在其更强模型的输出上进行微调，例如GPT-4o）进行了比较，并发现自我训练在减少幻觉方面达到了可相比的效果。我们推测这一意外发现可能是由于知识蒸馏情况下增强了的暴露偏差问题所致，并通过事后分析支持了这一推测。我们还通过上下文相关的“不知道”回答来提高对无法回答的问题和检索失败的鲁棒性。这些发现表明，可以使用合成数据和开源模型的自我训练构建可扩展且成本效率高的QA系统，从而减少对专有工具或昂贵的人工标注数据的依赖。 

---
# Stay Focused: Problem Drift in Multi-Agent Debate 

**Title (ZH)**: 保持专注：多智能体辩论中的问题漂移 

**Authors**: Jonas Becker, Lars Benedikt Kaesberg, Andreas Stephan, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2502.19559)  

**Abstract**: Multi-agent debate - multiple instances of large language models discussing problems in turn-based interaction - has shown promise for solving knowledge and reasoning tasks. However, these methods show limitations, particularly when scaling them to longer reasoning chains. In this study, we unveil a new issue of multi-agent debate: discussions drift away from the initial problem over multiple turns. We define this phenomenon as problem drift and quantify its presence across ten tasks (i.e., three generative, three knowledge, three reasoning, and one instruction-following task). To identify the reasons for this issue, we perform a human study with eight experts on discussions suffering from problem drift, who find the most common issues are a lack of progress (35% of cases), low-quality feedback (26% of cases), and a lack of clarity (25% of cases). To systematically address the issue of problem drift, we propose DRIFTJudge, a method based on LLM-as-a-judge, to detect problem drift at test-time. We further propose DRIFTPolicy, a method to mitigate 31% of problem drift cases. Our study can be seen as a first step to understanding a key limitation of multi-agent debate, highlighting pathways for improving their effectiveness in the future. 

**Abstract (ZH)**: 多智能体辩论——多个大型语言模型在轮换交互中讨论问题——已被证明在解决知识和推理任务方面具有潜力。然而，在将这些方法扩展到更长的推理链时，它们显示出一些限制。在此研究中，我们揭示了一个新的多智能体辩论问题：讨论会在多个回合后偏离初始问题。我们将这一现象定义为问题漂移，并通过十个任务（即三项生成性任务、三项知识性任务、三项推理性任务和一项指令遵循任务）来量化其存在。为了找出这个问题的原因，我们在八位专家的指导下进行了一项研究，这些专家发现讨论中问题漂移的主要问题是缺乏进展（占35%的情况）、低质量反馈（占26%的情况）和缺乏清晰性（占25%的情况）。为了系统地解决问题漂移的问题，我们提出了一种基于LLM-as-a-judge的方法（DRIFTJudge），在测试时检测问题漂移。我们进一步提出了一种方法DRIFTPolicy，有助于减轻31%的问题漂移情况。我们的研究可以被视为理解多智能体辩论的关键局限性的一个初步步骤，并强调了未来提高其有效性的方法。 

---
# Why Are Web AI Agents More Vulnerable Than Standalone LLMs? A Security Analysis 

**Title (ZH)**: 为什么网络AI代理比独立的大语言模型更易受攻击？一项安全性分析 

**Authors**: Jeffrey Yang Fan Chiang, Seungjae Lee, Jia-Bin Huang, Furong Huang, Yizheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20383)  

**Abstract**: Recent advancements in Web AI agents have demonstrated remarkable capabilities in addressing complex web navigation tasks. However, emerging research shows that these agents exhibit greater vulnerability compared to standalone Large Language Models (LLMs), despite both being built upon the same safety-aligned models. This discrepancy is particularly concerning given the greater flexibility of Web AI Agent compared to standalone LLMs, which may expose them to a wider range of adversarial user inputs. To build a scaffold that addresses these concerns, this study investigates the underlying factors that contribute to the increased vulnerability of Web AI agents. Notably, this disparity stems from the multifaceted differences between Web AI agents and standalone LLMs, as well as the complex signals - nuances that simple evaluation metrics, such as success rate, often fail to capture. To tackle these challenges, we propose a component-level analysis and a more granular, systematic evaluation framework. Through this fine-grained investigation, we identify three critical factors that amplify the vulnerability of Web AI agents; (1) embedding user goals into the system prompt, (2) multi-step action generation, and (3) observational capabilities. Our findings highlights the pressing need to enhance security and robustness in AI agent design and provide actionable insights for targeted defense strategies. 

**Abstract (ZH)**: 最近在Web AI代理方面的进展展示了其在处理复杂网络导航任务方面的显著能力。然而，新兴的研究表明，这些代理相较独立的大规模语言模型（LLMs）更加脆弱，尽管两者都是基于相同的安全对齐模型构建的。鉴于Web AI代理相较于独立的大规模语言模型具有更大的灵活性，这可能会使它们暴露在更广泛的攻击性用户输入之下，因此这种差异尤其令人担忧。为了解决这些问题，本研究调查了导致Web AI代理脆弱性的根本因素。值得注意的是，这种差异源于Web AI代理和独立的大规模语言模型之间的多方面差异，以及复杂的信号——这种差异往往无法通过简单的评估指标，如成功率来捕捉。为了应对这些挑战，我们提出了一种组件级分析和更细致、系统化的评估框架。通过这种精密的调查，我们确定了三个关键因素，这些因素会加剧Web AI代理的脆弱性；（1）将用户目标嵌入系统提示，（2）多步动作生成，（3）观察能力。我们的发现突显了在AI代理设计中增强安全性和鲁棒性的紧迫需求，并提供了针对性防御策略的具体建议。 

---
# Voting or Consensus? Decision-Making in Multi-Agent Debate 

**Title (ZH)**: 投票还是共识？多智能体辩论中的决策机制 

**Authors**: Lars Benedikt Kaesberg, Jonas Becker, Jan Philip Wahle, Terry Ruas, Bela Gipp  

**Link**: [PDF](https://arxiv.org/pdf/2502.19130)  

**Abstract**: Much of the success of multi-agent debates depends on carefully choosing the right parameters. Among them, the decision-making protocol stands out. Systematic comparison of decision protocols is difficult because studies alter multiple discussion parameters beyond the protocol. So far, it has been largely unknown how decision-making addresses the challenges of different tasks. This work systematically evaluates the impact of seven decision protocols (e.g., majority voting, unanimity consensus). We change only one variable at a time (i.e., decision protocol) to analyze how different methods affect the collaboration between agents and test different protocols on knowledge (MMLU, MMLU-Pro, GPQA) and reasoning datasets (StrategyQA, MuSR, SQuAD 2.0). Our results show that voting protocols improve performance by 13.2% in reasoning tasks and consensus protocols by 2.8% in knowledge tasks over the other decision protocol. Increasing the number of agents improves performance, while more discussion rounds before voting reduces it. To improve decision-making by increasing answer diversity, we propose two new methods, All-Agents Drafting (AAD) and Collective Improvement (CI). Our methods improve task performance by up to 3.3% with AAD and up to 7.4% with CI. This work demonstrates the importance of decision-making in multi-agent debates beyond scaling. 

**Abstract (ZH)**: 多智能体辩论的成功很大程度上取决于正确选择参数，其中决策协议尤为关键。系统地比较决策协议的差异颇具挑战性，因为研究中会同时改变多项讨论参数，而不仅仅是决策协议本身。目前，决策如何应对不同任务的挑战尚不明确。本研究系统地评估了七种决策协议（例如，多数投票、一致同意）的影响。我们每次只改变一个变量（即，决策协议），分析不同方法对智能体之间协作的影响，并在知识（MMLU、MMLU-Pro、GPQA）和推理数据集（StrategyQA、MuSR、SQuAD 2.0）上测试不同的协议。结果显示，投票协议在推理任务中的性能提高了13.2%，而在知识任务中的共识协议则提高了2.8%。增加智能体的数量可以提升性能，而投票前的讨论轮数增多则会降低性能。为了通过增加答案多样性来提高决策质量，我们提出了两种新方法：All-Agents Drafting (AAD) 和 Collective Improvement (CI)。我们的方法在AAD下的任务性能提高了最多3.3%，在CI下则提高了最多7.4%。本研究展示了多智能体辩论中决策的重要性，不仅在于规模的扩大。 

---
