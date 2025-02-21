# Benchmarking Multimodal RAG through a Chart-based Document Question-Answering Generation Framework 

**Title (ZH)**: 基于图表型文档问答生成框架的多模态RAG基准测试 

**Authors**: Yuming Yang, Jiang Zhong, Li Jin, Jingwang Huang, Jingpeng Gao, Qing Liu, Yang Bai, Jingyuan Zhang, Rui Jiang, Kaiwen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.14864)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) enhances reasoning capabilities by integrating external knowledge. However, existing benchmarks primarily focus on simple image-text interactions, overlooking complex visual formats like charts that are prevalent in real-world applications. In this work, we introduce a novel task, Chart-based MRAG, to address this limitation. To semi-automatically generate high-quality evaluation samples, we propose CHARt-based document question-answering GEneration (CHARGE), a framework that produces evaluation data through structured keypoint extraction, crossmodal verification, and keypoint-based generation. By combining CHARGE with expert validation, we construct Chart-MRAG Bench, a comprehensive benchmark for chart-based MRAG evaluation, featuring 4,738 question-answering pairs across 8 domains from real-world documents. Our evaluation reveals three critical limitations in current approaches: (1) unified multimodal embedding retrieval methods struggles in chart-based scenarios, (2) even with ground-truth retrieval, state-of-the-art MLLMs achieve only 58.19% Correctness and 73.87% Coverage scores, and (3) MLLMs demonstrate consistent text-over-visual modality bias during Chart-based MRAG reasoning. The CHARGE and Chart-MRAG Bench are released at this https URL. 

**Abstract (ZH)**: 多模态检索增强生成（MRAG）通过集成外部知识增强了推理能力。然而，现有的基准主要关注简单的图像-文本交互，忽视了在实际应用中常见的复杂视觉格式，如图表。为此，我们在本文中提出了一项新颖的任务，即基于图表的MRAG，以解决这一局限性。为了半自动地生成高质量的评估样本，我们提出了基于图表的文档问答生成（CHARGE）框架，该框架通过结构化关键点提取、跨模态验证和关键点驱动的生成来生成评估数据。通过将CHARGE与专家验证相结合，我们构建了基于图表的MRAG基准（Chart-MRAG Bench），该基准涵盖了来自8个实际文档领域的4,738个问答对。我们的评估揭示了当前方法存在三个关键不足：（1）统一的多模态嵌入检索方法在基于图表的场景中表现不佳；（2）即使有 ground-truth 检索，最先进的大规模语言模型也只能达到58.19%的正确率和73.87%的覆盖范围；（3）大规模语言模型在基于图表的MRAG推理过程中表现出持续的文本占优视觉模式偏见。CHARGE 和 Chart-MRAG Bench 已在此 HTTPS 地址发布：[https://example.com/CHARGE-and-Chart-MRAG-Bench](https://example.com/CHARGE-and-Chart-MRAG-Bench) 

---
# Optimizing Model Selection for Compound AI Systems 

**Title (ZH)**: 优化复合人工智能系统中的模型选择方法 

**Authors**: Lingjiao Chen, Jared Quincy Davis, Boris Hanin, Peter Bailis, Matei Zaharia, James Zou, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2502.14815)  

**Abstract**: Compound AI systems that combine multiple LLM calls, such as self-refine and multi-agent-debate, achieve strong performance on many AI tasks. We address a core question in optimizing compound systems: for each LLM call or module in the system, how should one decide which LLM to use? We show that these LLM choices have a large effect on quality, but the search space is exponential. We propose LLMSelector, an efficient framework for model selection in compound systems, which leverages two key empirical insights: (i) end-to-end performance is often monotonic in how well each module performs, with all other modules held fixed, and (ii) per-module performance can be estimated accurately by an LLM. Building upon these insights, LLMSelector iteratively selects one module and allocates to it the model with the highest module-wise performance, as estimated by an LLM, until no further gain is possible. LLMSelector is applicable to any compound system with a bounded number of modules, and its number of API calls scales linearly with the number of modules, achieving high-quality model allocation both empirically and theoretically. Experiments with popular compound systems such as multi-agent debate and self-refine using LLMs such as GPT-4o, Claude 3.5 Sonnet and Gemini 1.5 show that LLMSelector confers 5%-70% accuracy gains compared to using the same LLM for all modules. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，并符合学术规范：

结合多个大语言模型（LLM）调用的复合AI系统，如自我优化和多智能体辩论，在许多AI任务上取得了出色表现。我们探讨了优化复合系统中的核心问题：对于系统中的每个LLM调用或模块，应如何决定使用哪个LLM？我们展示了这些LLM选择对质量有重大影响，但搜索空间呈指数增长。为此，我们提出了LLMSelector，这是一种高效的选择模型框架，利用了两个关键的经验见解：（i）端到端性能通常在固定其他模块情况下，每个模块性能提高而单调增长；（ii）通过LLM可准确估计每个模块的性能。基于这些见解，LLMSelector 逐步选择一个模块，并为该模块分配根据LLM估计表现最佳的模型，直到无法进一步提高性能。LLMSelector 可应用于具有有限模块数量的任何复合系统，其API调用次数随模块数量线性增长，在实验和理论上均实现了高质量的模型分配。使用流行的复合系统如多智能体辩论和自我优化以及LLM如GPT-4o、Claude 3.5 Sonnet和Gemini 1.5的实验表明，与使用同一LLM为所有模块相比，LLMSelector 可提供5%至70%的准确性提升。 

---
# Making Universal Policies Universal 

**Title (ZH)**: 使通用政策真正通用 

**Authors**: Niklas Höpner, David Kuric, Herke van Hoof  

**Link**: [PDF](https://arxiv.org/pdf/2502.14777)  

**Abstract**: The development of a generalist agent capable of solving a wide range of sequential decision-making tasks remains a significant challenge. We address this problem in a cross-agent setup where agents share the same observation space but differ in their action spaces. Our approach builds on the universal policy framework, which decouples policy learning into two stages: a diffusion-based planner that generates observation sequences and an inverse dynamics model that assigns actions to these plans. We propose a method for training the planner on a joint dataset composed of trajectories from all agents. This method offers the benefit of positive transfer by pooling data from different agents, while the primary challenge lies in adapting shared plans to each agent's unique constraints. We evaluate our approach on the BabyAI environment, covering tasks of varying complexity, and demonstrate positive transfer across agents. Additionally, we examine the planner's generalisation ability to unseen agents and compare our method to traditional imitation learning approaches. By training on a pooled dataset from multiple agents, our universal policy achieves an improvement of up to $42.20\%$ in task completion accuracy compared to a policy trained on a dataset from a single agent. 

**Abstract (ZH)**: 开发能够在广泛范围的序列决策任务中求解的通用智能体仍然是一个显著的挑战。我们通过跨智能体设置解决了这个问题，即智能体共享相同观测空间但动作空间不同。我们的方法基于通用策略框架，将策略学习分为两个阶段：一个基于扩散的规划器生成观测序列，以及一个逆动力学模型将动作分配给这些计划。我们提出了一种在由所有智能体轨迹组成的联合数据集上训练规划器的方法。这种方法通过从不同智能体中汇集数据，提供了正向迁移的好处，但主要挑战在于使共享计划适应每个智能体的独特约束。我们通过BabyAI环境评估了我们的方法，该环境涵盖了不同难度的任务，并展示了智能体之间的正向迁移能力。此外，我们还探讨了规划器对未见过的智能体的泛化能力，并将我们的方法与传统的模仿学习方法进行了比较。通过在多个智能体的联合数据集上进行训练，我们的通用策略在任务完成准确性方面的提高幅度达到了$42.20\%$，相比仅在一个智能体数据集上训练的策略。 

---
# EquivaMap: Leveraging LLMs for Automatic Equivalence Checking of Optimization Formulations 

**Title (ZH)**: EquivaMap：利用大规模语言模型进行优化模型等价性自动检查 

**Authors**: Haotian Zhai, Connor Lawless, Ellen Vitercik, Liu Leqi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14760)  

**Abstract**: A fundamental problem in combinatorial optimization is identifying equivalent formulations, which can lead to more efficient solution strategies and deeper insights into a problem's computational complexity. The need to automatically identify equivalence between problem formulations has grown as optimization copilots--systems that generate problem formulations from natural language descriptions--have proliferated. However, existing approaches to checking formulation equivalence lack grounding, relying on simple heuristics which are insufficient for rigorous validation. Inspired by Karp reductions, in this work we introduce quasi-Karp equivalence, a formal criterion for determining when two optimization formulations are equivalent based on the existence of a mapping between their decision variables. We propose EquivaMap, a framework that leverages large language models to automatically discover such mappings, enabling scalable and reliable equivalence verification. To evaluate our approach, we construct the first open-source dataset of equivalent optimization formulations, generated by applying transformations such as adding slack variables or valid inequalities to existing formulations. Empirically, EquivaMap significantly outperforms existing methods, achieving substantial improvements in correctly identifying formulation equivalence. 

**Abstract (ZH)**: 组合优化中的一个基本问题是识别等价形式，这可以导致更高效的求解策略，并深入理解问题的计算复杂性。随着优化协驾系统（能够从自然语言描述生成问题形式的系统）的普及，自动识别问题形式之间的等价性变得越来越重要。然而，现有的形式等价性检查方法缺乏坚实的理论基础，仅仅依赖简单的启发式方法，这些方法不足以进行严格的验证。受到Karp约简的启发，我们在此工作中引入了准Karp等价性，这是一种基于决策变量之间的映射存在的形式来判断两个优化形式是否等价的正式标准。我们提出了一种名为EquivaMap的框架，利用大型语言模型自动发现这些映射，从而实现大规模且可靠的等价性验证。为评估我们的方法，我们构建了第一个开源的等价优化形式数据集，通过应用诸如增加松弛变量或有效不等式的变换生成这些形式。从实验结果来看，EquivaMap显著优于现有方法，在准确识别形式等价性方面取得了重大改进。 

---
# From Knowledge Generation to Knowledge Verification: Examining the BioMedical Generative Capabilities of ChatGPT 

**Title (ZH)**: 从知识生成到知识验证：评估ChatGPT在生物医学领域生成能力的研究 

**Authors**: Ahmed Abdeen Hamed, Byung Suk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.14714)  

**Abstract**: The generative capabilities of LLM models present opportunities in accelerating tasks and concerns with the authenticity of the knowledge it produces. To address the concerns, we present a computational approach that systematically evaluates the factual accuracy of biomedical knowledge that an LLM model has been prompted to generate. Our approach encompasses two processes: the generation of disease-centric associations and the verification of them using the semantic knowledge of the biomedical ontologies. Using ChatGPT as the select LLM model, we designed a set of prompt-engineering processes to generate linkages between diseases, drugs, symptoms, and genes to establish grounds for assessments. Experimental results demonstrate high accuracy in identifying disease terms (88%-97%), drug names (90%-91%), and genetic information (88%-98%). The symptom term identification accuracy was notably lower (49%-61%), as verified against the DOID, ChEBI, SYMPTOM, and GO ontologies accordingly. The verification of associations reveals literature coverage rates of (89%-91%) among disease-drug and disease-gene associations. The low identification accuracy for symptom terms also contributed to the verification of symptom-related associations (49%-62%). 

**Abstract (ZH)**: 大语言模型（LLM）的生成能力为加速任务提供了机会，同时对它所生成知识的真实性也提出了担忧。为了应对这些担忧，我们提出了一种计算方法，旨在系统地评估LLM模型生成的生物医药知识的事实准确性。该方法包含两个过程：疾病为中心的关联生成和使用生物医药本体的语义知识对这些关联进行验证。在选择ChatGPT作为LLM模型的基础上，我们设计了一套提示工程过程，生成疾病、药物、症状和基因之间的关联，以此为基础进行评估。实验结果显示，在识别疾病术语（88%-97%）、药物名称（90%-91%）和遗传信息（88%-98%）方面具有较高的准确性。然而，症状术语的识别准确性较低（49%-61%），这经由DOID、ChEBI、SYMPTOM和GO本体验证。关联的验证表明，在疾病—药物和疾病—基因关联中，文献覆盖率分别为89%-91%；症状术语识别准确性较低也影响了相关关联的验证（49%-62%）。 

---
# Building reliable sim driving agents by scaling self-play 

**Title (ZH)**: 通过扩展自我对弈构建可靠的模拟驾驶代理 

**Authors**: Daphne Cornelisse, Aarav Pandya, Kevin Joseph, Joseph Suárez, Eugene Vinitsky  

**Link**: [PDF](https://arxiv.org/pdf/2502.14706)  

**Abstract**: Simulation agents are essential for designing and testing systems that interact with humans, such as autonomous vehicles (AVs). These agents serve various purposes, from benchmarking AV performance to stress-testing the system's limits, but all use cases share a key requirement: reliability. A simulation agent should behave as intended by the designer, minimizing unintended actions like collisions that can compromise the signal-to-noise ratio of analyses. As a foundation for reliable sim agents, we propose scaling self-play to thousands of scenarios on the Waymo Open Motion Dataset under semi-realistic limits on human perception and control. Training from scratch on a single GPU, our agents nearly solve the full training set within a day. They generalize effectively to unseen test scenes, achieving a 99.8% goal completion rate with less than 0.8% combined collision and off-road incidents across 10,000 held-out scenarios. Beyond in-distribution generalization, our agents show partial robustness to out-of-distribution scenes and can be fine-tuned in minutes to reach near-perfect performance in those cases. Demonstrations of agent behaviors can be found at this link. We open-source both the pre-trained agents and the complete code base. Demonstrations of agent behaviors can be found at \url{this https URL}. 

**Abstract (ZH)**: 仿真实体对于设计和测试与人类交互的系统（如自动驾驶车辆AV）是必不可少的。这些实体具有多种用途，从评估AV性能到检验系统的极限性能，但所有应用场景都共享一个关键要求：可靠性。仿真实体应按设计者的意图行动，尽量减少诸如碰撞等意外行为，这些行为可能会影响分析中的信噪比。为建立可靠的仿真实体基础，我们建议在Waymo Open Motion数据集中对数千种场景进行半现实条件下的人类感知与控制限制下的自我对弈扩展。在单块GPU上从头训练，我们的实体在不到一天的时间内几乎可以解决整个训练集。它们在未见过的测试场景中表现出色，完成了99.8%的任务，总计发生不到0.8%的碰撞和离开道路事件，其中包含了10000个保留场景。除了内部泛化，我们的实体对未见过的场景也显示出部分鲁棒性，并且可以在几分钟内微调以达到近乎完美的性能。实体行为的演示可以在以下链接中找到：[这个链接](this https URL)。我们已开源了预训练的实体和完整的代码库。实体行为的演示可以在[这个URL](this https URL)中找到。 

---
# A Statistical Case Against Empirical Human-AI Alignment 

**Title (ZH)**: 一项关于经验性人机对齐的统计反驳 

**Authors**: Julian Rodemann, Esteban Garces Arias, Christoph Luther, Christoph Jansen, Thomas Augustin  

**Link**: [PDF](https://arxiv.org/pdf/2502.14581)  

**Abstract**: Empirical human-AI alignment aims to make AI systems act in line with observed human behavior. While noble in its goals, we argue that empirical alignment can inadvertently introduce statistical biases that warrant caution. This position paper thus advocates against naive empirical alignment, offering prescriptive alignment and a posteriori empirical alignment as alternatives. We substantiate our principled argument by tangible examples like human-centric decoding of language models. 

**Abstract (ZH)**: 实证人机一致性旨在使人工智能系统的行为与观察到的人类行为保持一致。虽然这一目标是值得追求的，但我们认为实证一致性可能会无意中引入统计偏见，这值得我们谨慎对待。因此，本文主张避免简单的实证一致性，并提出规范性一致性及后验实证一致性作为替代方案。我们通过以人为中心的语言模型解码等具体例子来支持我们的原则性论点。 

---
# Plan-over-Graph: Towards Parallelable LLM Agent Schedule 

**Title (ZH)**: 基于图的计划：朝着可并行的LLM代理调度方向 

**Authors**: Shiqi Zhang, Xinbei Ma, Zouying Cao, Zhuosheng Zhang, Hai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14563)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional abilities in reasoning for task planning. However, challenges remain under-explored for parallel schedules. This paper introduces a novel paradigm, plan-over-graph, in which the model first decomposes a real-life textual task into executable subtasks and constructs an abstract task graph. The model then understands this task graph as input and generates a plan for parallel execution. To enhance the planning capability of complex, scalable graphs, we design an automated and controllable pipeline to generate synthetic graphs and propose a two-stage training scheme. Experimental results show that our plan-over-graph method significantly improves task performance on both API-based LLMs and trainable open-sourced LLMs. By normalizing complex tasks as graphs, our method naturally supports parallel execution, demonstrating global efficiency. The code and data are available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在任务规划推理方面展现出了卓越的能力。然而，对于并行调度的挑战仍存在未探索的空间。本文引入了一种新颖的范式——“计划覆盖图”（Plan-over-Graph），该范式使模型首先将实际文本任务分解为可执行子任务，并构建一个抽象的任务图。然后，模型将理解该任务图作为输入，并生成一个用于并行执行的计划。为了增强处理复杂可扩展图形的规划能力，我们设计了一个自动化且可控的管道来生成合成图形，并提出了一种两阶段训练方案。实验结果表明，我们的“计划覆盖图”方法在基于API的LLMs和可训练的开源LLMs上显著提高了任务性能。通过将复杂的任务规范化为图形表示，该方法自然支持并行执行，从而展示出全局效率。代码和数据已发布在该链接：[此处提供链接]。 

---
# Statistical Scenario Modelling and Lookalike Distributions for Multi-Variate AI Risk 

**Title (ZH)**: 多变量AI风险的统计场景建模与类似分布研究 

**Authors**: Elija Perrier  

**Link**: [PDF](https://arxiv.org/pdf/2502.14491)  

**Abstract**: Evaluating AI safety requires statistically rigorous methods and risk metrics for understanding how the use of AI affects aggregated risk. However, much AI safety literature focuses upon risks arising from AI models in isolation, lacking consideration of how modular use of AI affects risk distribution of workflow components or overall risk metrics. There is also a lack of statistical grounding enabling sensitisation of risk models in the presence of absence of AI to estimate causal contributions of AI. This is in part due to the dearth of AI impact data upon which to fit distributions. In this work, we address these gaps in two ways. First, we demonstrate how scenario modelling (grounded in established statistical techniques such as Markov chains, copulas and Monte Carlo simulation) can be used to model AI risk holistically. Second, we show how lookalike distributions from phenomena analogous to AI can be used to estimate AI impacts in the absence of directly observable data. We demonstrate the utility of our methods for benchmarking cumulative AI risk via risk analysis of a logistic scenario simulations. 

**Abstract (ZH)**: 评估人工智能安全性需要统计严谨的方法和风险度量，以理解人工智能的使用如何影响积聚的风险。然而，许多人工智能安全性文献主要关注单一人工智能模型所带来的风险，忽略了模块化使用人工智能如何影响工作流程组件的风险分布或总体风险度量方面的考虑。此外，在AI存在与否的情况下，缺乏统计基础来敏感化风险模型，以估计AI的因果贡献。部分原因在于缺乏可用于拟合分布的人工智能影响数据。在本工作中，我们通过两种方式来填补这些空白。首先，我们展示了如何利用情景建模（基于成熟的统计技术，如马尔可夫链、 copulas 和蒙特卡洛模拟）来全面建模人工智能风险。其次，我们展示了如何利用类似AI现象的联想分布来估计在无法直接观察数据的情况下的人工智能影响。通过基准测试累计人工智能风险的风险分析仿真，我们证明了我们方法的有效性。 

---
# Narrative-Driven Travel Planning: Geoculturally-Grounded Script Generation with Evolutionary Itinerary Optimization 

**Title (ZH)**: 基于叙述的旅游规划：地理文化为基础的情景生成与进化行程优化 

**Authors**: Ran Ding, Ziyu Zhang, Ying Zhu, Ziqian Kong, Peilan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14456)  

**Abstract**: To enhance tourists' experiences and immersion, this paper proposes a narrative-driven travel planning framework called NarrativeGuide, which generates a geoculturally-grounded narrative script for travelers, offering a novel, role-playing experience for their journey. In the initial stage, NarrativeGuide constructs a knowledge graph for attractions within a city, then configures the worldview, character setting, and exposition based on the knowledge graph. Using this foundation, the knowledge graph is combined to generate an independent scene unit for each attraction. During the itinerary planning stage, NarrativeGuide models narrative-driven travel planning as an optimization problem, utilizing a genetic algorithm (GA) to refine the itinerary. Before evaluating the candidate itinerary, transition scripts are generated for each pair of adjacent attractions, which, along with the scene units, form a complete script. The weighted sum of script coherence, travel time, and attraction scores is then used as the fitness value to update the candidate solution set. Experimental results across four cities, i.e., Nanjing and Yangzhou in China, Paris in France, and Berlin in Germany, demonstrate significant improvements in narrative coherence and cultural fit, alongside a notable reduction in travel time and an increase in the quality of visited attractions. Our study highlights that incorporating external evolutionary optimization effectively addresses the limitations of large language models in travel this http URL codes are available at this https URL. 

**Abstract (ZH)**: 为了提升旅游者的体验和沉浸感，本文提出了一种名为NarrativeGuide的叙述驱动旅游规划框架，该框架生成了基于地理文化的叙述剧本，为旅游者提供了新颖的角色扮演体验。在初始阶段，NarrativeGuide构建城市景点的知识图谱，然后基于知识图谱配置世界观、角色设定和背景介绍。在此基础上，知识图谱被结合用来为每个景点生成一个独立的场景单元。在行程规划阶段，NarrativeGuide将叙述驱动的旅游规划建模为一个优化问题，利用遗传算法（GA）优化行程安排。在评估候选行程之前，为每一对相邻景点生成转换脚本，这些脚本与场景单元共同形成完整的剧本。剧本一致性、旅行时间和景点评分的加权和被用作适应度值来更新候选解集。实验结果在南京和扬州（中国）、巴黎（法国）和柏林（德国）四个城市中表明，该框架显著提高了叙述连贯性和文化契合度，同时减少了旅行时间，并提高了访问景点的质量。我们的研究显示，将外部进化优化引入旅游规划，有效解决了大型语言模型在旅行规划中的局限性。相关代码可通过以下链接获取：此链接。 

---
# HPS: Hard Preference Sampling for Human Preference Alignment 

**Title (ZH)**: HPS：难治疗偏好抽样以实现人类偏好对齐 

**Authors**: Xiandong Zou, Wanyu Lin, Yuchen Li, Pan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.14400)  

**Abstract**: Aligning Large Language Model (LLM) responses with human preferences is vital for building safe and controllable AI systems. While preference optimization methods based on Plackett-Luce (PL) and Bradley-Terry (BT) models have shown promise, they face challenges such as poor handling of harmful content, inefficient use of dispreferred responses, and, specifically for PL, high computational costs. To address these issues, we propose Hard Preference Sampling (HPS), a novel framework for robust and efficient human preference alignment. HPS introduces a training loss that prioritizes the most preferred response while rejecting all dispreferred and harmful ones. It emphasizes "hard" dispreferred responses--those closely resembling preferred ones--to enhance the model's rejection capabilities. By leveraging a single-sample Monte Carlo sampling strategy, HPS reduces computational overhead while maintaining alignment quality. Theoretically, HPS improves sample efficiency over existing PL methods and maximizes the reward margin between preferred and dispreferred responses, ensuring clearer distinctions. Experiments on HH-RLHF and PKU-Safety datasets validate HPS's effectiveness, achieving comparable BLEU and reward scores while greatly improving reward margins and thus reducing harmful content generation. 

**Abstract (ZH)**: 将大型语言模型（LLM）的响应与人类偏好对齐对于构建安全可控的人工智能系统至关重要。基于Plackett-Luce (PL) 和Bradley-Terry (BT) 模型的偏好优化方法虽然显示出了潜力，但也面临着处理有害内容不佳、对不preferred响应利用效率低等问题，特别是对于PL模型，计算成本较高。为了解决这些问题，我们提出了硬偏好采样（HPS，Hard Preference Sampling）框架，这是一种新颖的人类偏好对齐方法，旨在提高对齐的鲁棒性和效率。HPS引入了一种训练损失函数，优先考虑最偏好响应，并拒绝所有不偏好和有害的响应。它强调“硬”不偏好响应—那些与偏好响应高度相似的响应—以增强模型的拒绝能力。通过利用单样本蒙特卡洛采样策略，HPS减少了计算开销，同时保持了对齐质量。理论上，HPS相比现有PL方法提高了样本效率，最大化了偏好与不偏好响应之间的奖励差距，确保了更清晰的区分度。实验结果表明，HPS在HH-RLHF和PKU-Safety数据集上有效，能够达到相似的BLEU和奖励分数，并大大提高了奖励差距，从而降低了有害内容的生成。 

---
# Retrieval-Augmented Process Reward Model for Generalizable Mathematical Reasoning 

**Title (ZH)**: 用于通用数学推理的检索增强过程奖励模型 

**Authors**: Jiachen Zhu, Congmin Zheng, Jianghao Lin, Kounianhua Du, Ying Wen, Yong Yu, Jun Wang, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14361)  

**Abstract**: While large language models (LLMs) have significantly advanced mathematical reasoning, Process Reward Models (PRMs) have been developed to evaluate the logical validity of reasoning steps. However, PRMs still struggle with out-of-distribution (OOD) challenges. This paper identifies key OOD issues, including step OOD, caused by differences in reasoning patterns across model types and sizes, and question OOD, which arises from dataset shifts between training data and real-world problems. To address these issues, we introduce Retrieval-Augmented Process Reward Model (RetrievalPRM), a novel framework designed to tackle these OOD issues. By utilizing a two-stage retrieval-enhanced mechanism, RetrievalPRM retrieves semantically similar questions and steps as a warmup, enhancing PRM's ability to evaluate target steps and improving generalization and reasoning consistency across different models and problem types. Our extensive experiments demonstrate that RetrievalPRM outperforms existing baselines across multiple real-world datasets. Our open-source contributions include a retrieval-enhanced dataset, a tuning framework for PRM training, and the RetrievalPRM model, establishing a new standard for PRM performance. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在数学推理方面取得了显著进展，过程奖励模型（PRMs）已被发展出来评估推理步骤的逻辑有效性。然而，PRMs 仍然面临分布外（OOD）挑战。本文识别出关键的OOD问题，包括由于不同模型类型和规模的推理模式差异导致的步骤OOD问题，以及由于训练数据与真实世界问题之间的数据集偏移导致的问题OOD问题。为解决这些问题，我们引入了检索增强过程奖励模型（RetrievalPRM），这是一种新的框架，旨在解决这些OOD问题。通过利用两阶段的检索增强机制，RetrievalPRM 在预热阶段检索语义上相似的问题和步骤，从而增强PRM评估目标步骤的能力，并提高不同模型和问题类型下的泛化能力和推理一致性。我们的广泛实验表明，RetrievalPRM 在多个真实世界数据集上优于现有基线。我们的开源贡献包括一个检索增强数据集、一种PRM训练的微调框架以及RetrievalPRM模型，为PRM性能设立了新的标准。 

---
# FlowAgent: Achieving Compliance and Flexibility for Workflow Agents 

**Title (ZH)**: FlowAgent: 实现工作流代理的合规性和灵活性 

**Authors**: Yuchen Shi, Siqi Cai, Zihan Xu, Yuei Qin, Gang Li, Hang Shao, Jiawei Chen, Deqing Yang, Ke Li, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.14345)  

**Abstract**: The integration of workflows with large language models (LLMs) enables LLM-based agents to execute predefined procedures, enhancing automation in real-world applications. Traditional rule-based methods tend to limit the inherent flexibility of LLMs, as their predefined execution paths restrict the models' action space, particularly when the unexpected, out-of-workflow (OOW) queries are encountered. Conversely, prompt-based methods allow LLMs to fully control the flow, which can lead to diminished enforcement of procedural compliance. To address these challenges, we introduce FlowAgent, a novel agent framework designed to maintain both compliance and flexibility. We propose the Procedure Description Language (PDL), which combines the adaptability of natural language with the precision of code to formulate workflows. Building on PDL, we develop a comprehensive framework that empowers LLMs to manage OOW queries effectively, while keeping the execution path under the supervision of a set of controllers. Additionally, we present a new evaluation methodology to rigorously assess an LLM agent's ability to handle OOW scenarios, going beyond routine flow compliance tested in existing benchmarks. Experiments on three datasets demonstrate that FlowAgent not only adheres to workflows but also effectively manages OOW queries, highlighting its dual strengths in compliance and flexibility. The code is available at this https URL. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

将工作流与大规模语言模型（LLMs）集成可以使基于LLM的代理执行预定义的程序，从而在实际应用中增强自动化。传统的基于规则的方法往往限制了LLM的固有灵活性，因为它们预定义的执行路径限制了模型的动作空间，特别是在遇到超出工作流（Out-of-Workflow, OOW）的查询时。相反，基于提示的方法使LLM能够完全控制流程，但可能会导致程序合规性的减弱。为了解决这些挑战，我们提出了FlowAgent这一新的代理框架，旨在同时保持合规性和灵活性。我们提出了过程描述语言（PDL），它结合了自然语言的灵活性和代码的精确性来定义工作流。基于PDL，我们开发了一个全面的框架，使LLM能够有效地处理OOW查询，同时在一系列控制器的监督下保留执行路径。此外，我们提出了一种新的评估方法，以严格评估LLM代理处理OOW场景的能力，超越了现有基准中测试的常规流程合规性。在三个数据集中进行的实验表明，FlowAgent不仅遵循工作流，还能有效处理OOW查询，突显了其在合规性和灵活性方面的双重优势。源代码可在以下链接获取：this https URL。 

---
# SPRIG: Stackelberg Perception-Reinforcement Learning with Internal Game Dynamics 

**Title (ZH)**: SPRIG：基于内部博弈动力学的梯度 Stackelberg 感知强化学习 

**Authors**: Fernando Martinez-Lopez, Juntao Chen, Yingdong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14264)  

**Abstract**: Deep reinforcement learning agents often face challenges to effectively coordinate perception and decision-making components, particularly in environments with high-dimensional sensory inputs where feature relevance varies. This work introduces SPRIG (Stackelberg Perception-Reinforcement learning with Internal Game dynamics), a framework that models the internal perception-policy interaction within a single agent as a cooperative Stackelberg game. In SPRIG, the perception module acts as a leader, strategically processing raw sensory states, while the policy module follows, making decisions based on extracted features. SPRIG provides theoretical guarantees through a modified Bellman operator while preserving the benefits of modern policy optimization. Experimental results on the Atari BeamRider environment demonstrate SPRIG's effectiveness, achieving around 30% higher returns than standard PPO through its game-theoretical balance of feature extraction and decision-making. 

**Abstract (ZH)**: 深度强化学习代理常常面临在高维感官输入环境中有效协调感知和决策模块的挑战，特别是在感官输入特征重要性变化的环境中。本文引入了SPRIG（Stackelberg感知-强化学习与内部博弈动力学）框架，该框架将单个代理内的感知-政策交互建模为合作的Stackelberg博弈。在SPRIG中，感知模块充当领导者，战略性地处理原始感官状态，而政策模块则跟随其后，基于提取的特征进行决策。通过修改的贝尔曼算子提供理论保证，同时保持现代策略优化的好处。在Atari BeamRider环境上的实验结果表明，SPRIG在特征提取和决策平衡的博弈论框架下，比标准PPO实现了约30%更高的回报。 

---
# Investigating the Impact of LLM Personality on Cognitive Bias Manifestation in Automated Decision-Making Tasks 

**Title (ZH)**: investigating LLM个性对自动化决策任务中认知偏差表现的影响 

**Authors**: Jiangen He, Jiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14219)  

**Abstract**: Large Language Models (LLMs) are increasingly used in decision-making, yet their susceptibility to cognitive biases remains a pressing challenge. This study explores how personality traits influence these biases and evaluates the effectiveness of mitigation strategies across various model architectures. Our findings identify six prevalent cognitive biases, while the sunk cost and group attribution biases exhibit minimal impact. Personality traits play a crucial role in either amplifying or reducing biases, significantly affecting how LLMs respond to debiasing techniques. Notably, Conscientiousness and Agreeableness may generally enhance the efficacy of bias mitigation strategies, suggesting that LLMs exhibiting these traits are more receptive to corrective measures. These findings address the importance of personality-driven bias dynamics and highlight the need for targeted mitigation approaches to improve fairness and reliability in AI-assisted decision-making. 

**Abstract (ZH)**: 大型语言模型（LLMs）在决策中越来越普遍，但它们的认知偏差敏感性仍然是一个紧迫的挑战。本研究探讨了人格特质如何影响这些偏差，并评估了不同模型架构下缓解策略的有效性。我们的研究结果识别了六种常见的认知偏差，其中沉没成本偏见和群体归因偏见的影响最小。人格特质在放大或减少偏见方面发挥着关键作用，对LLMs对去偏策略的响应方式产生了重大影响。值得注意的是，尽责性和宜人性可能普遍增强偏见缓解策略的有效性，表明表现出这些特质的LLMs更容易接受纠正措施。这些发现强调了基于人格的动力偏见动态的重要性，并突显了需要针对特定情况采取缓解措施的重要性，以提高AI辅助决策的公平性和可靠性。 

---
# Causal Mean Field Multi-Agent Reinforcement Learning 

**Title (ZH)**: 因果均场多智能体强化学习 

**Authors**: Hao Ma, Zhiqiang Pu, Yi Pan, Boyin Liu, Junlong Gao, Zhenyu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.14200)  

**Abstract**: Scalability remains a challenge in multi-agent reinforcement learning and is currently under active research. A framework named mean-field reinforcement learning (MFRL) could alleviate the scalability problem by employing the Mean Field Theory to turn a many-agent problem into a two-agent problem. However, this framework lacks the ability to identify essential interactions under nonstationary environments. Causality contains relatively invariant mechanisms behind interactions, though environments are nonstationary. Therefore, we propose an algorithm called causal mean-field Q-learning (CMFQ) to address the scalability problem. CMFQ is ever more robust toward the change of the number of agents though inheriting the compressed representation of MFRL's action-state space. Firstly, we model the causality behind the decision-making process of MFRL into a structural causal model (SCM). Then the essential degree of each interaction is quantified via intervening on the SCM. Furthermore, we design the causality-aware compact representation for behavioral information of agents as the weighted sum of all behavioral information according to their causal effects. We test CMFQ in a mixed cooperative-competitive game and a cooperative game. The result shows that our method has excellent scalability performance in both training in environments containing a large number of agents and testing in environments containing much more agents. 

**Abstract (ZH)**: 多智能体强化学习中的可扩展性仍然是一个重要挑战，当前正处于活跃的研究之中。一种名为均场强化学习（Mean-Field Reinforcement Learning, MFRL）的框架可以通过应用均场理论，将多智能体问题简化为两智能体问题，从而缓解可扩展性问题。然而，这种框架缺乏在非稳态环境中识别关键交互的能力。因果性包含在交互背后的相对不变机制，尽管环境是非稳态的。因此，我们提出了一种名为因果均场Q学习（Causal Mean-Field Q-learning, CMFQ）的算法来解决可扩展性问题。CMFQ在保留MFRL压缩动作-状态空间表示的同时，对智能体数量变化具有更强的鲁棒性。首先，我们将MFRL决策过程背后的因果性建模为结构因果模型（Structural Causal Model, SCM）。然后，通过干预SCM来量化每个交互的本征程度。此外，我们设计了一种因果性感知的紧凑表示来表征智能体的行为信息，该表示是所有行为信息的加权和，权重由它们的因果效应确定。我们在混合合作-竞争游戏和合作游戏中测试了CMFQ。结果表明，我们的方法在包含大量智能体的训练环境和包含更多智能体的测试环境中都表现出出色的可扩展性。 

---
# Giving AI Personalities Leads to More Human-Like Reasoning 

**Title (ZH)**: 赋予AI个性有助于实现更类人的推理 

**Authors**: Animesh Nighojkar, Bekhzodbek Moydinboyev, My Duong, John Licato  

**Link**: [PDF](https://arxiv.org/pdf/2502.14155)  

**Abstract**: In computational cognitive modeling, capturing the full spectrum of human judgment and decision-making processes, beyond just optimal behaviors, is a significant challenge. This study explores whether Large Language Models (LLMs) can emulate the breadth of human reasoning by predicting both intuitive, fast System 1 and deliberate, slow System 2 processes. We investigate the potential of AI to mimic diverse reasoning behaviors across a human population, addressing what we call the {\em full reasoning spectrum problem}. We designed reasoning tasks using a novel generalization of the Natural Language Inference (NLI) format to evaluate LLMs' ability to replicate human reasoning. The questions were crafted to elicit both System 1 and System 2 responses. Human responses were collected through crowd-sourcing and the entire distribution was modeled, rather than just the majority of the answers. We used personality-based prompting inspired by the Big Five personality model to elicit AI responses reflecting specific personality traits, capturing the diversity of human reasoning, and exploring how personality traits influence LLM outputs. Combined with genetic algorithms to optimize the weighting of these prompts, this method was tested alongside traditional machine learning models. The results show that LLMs can mimic human response distributions, with open-source models like Llama and Mistral outperforming proprietary GPT models. Personality-based prompting, especially when optimized with genetic algorithms, significantly enhanced LLMs' ability to predict human response distributions, suggesting that capturing suboptimal, naturalistic reasoning may require modeling techniques incorporating diverse reasoning styles and psychological profiles. The study concludes that personality-based prompting combined with genetic algorithms is promising for enhancing AI's \textit{human-ness} in reasoning. 

**Abstract (ZH)**: 在计算认知模型中，超越最优行为来捕捉人类判断和决策过程的完整谱系是一项重大挑战。本研究探讨大语言模型（LLMs）是否能够通过预测直觉快速的System 1过程和审慎缓慢的System 2过程来模拟人类推理的广度。我们研究了人工智能模仿人类群体中多样推理行为的潜力，解决我们称之为“完整推理谱系问题”的问题。我们使用了一种自然语言推理（NLI）格式的新颖扩展来设计推理任务，以评估LLMs复制人类推理的能力。问题设计得既能引发直觉快速的System 1反应，又能引发审慎缓慢的System 2反应。人类反应是通过众包收集的，并且我们对整个分布进行了建模，而不仅仅是大多数答案的分布。我们借鉴了大五人格模型的启发，使用基于人格的提示来引发反映特定人格特质的AI响应，从而捕捉人类推理的多样性，并探讨人格特质如何影响LLM的输出。结合遗传算法来优化这些提示的权重后，这种方法与传统的机器学习模型进行了测试。结果表明，开源模型如Llama和Mistral的性能优于专有的GPT模型。尤其是使用遗传算法优化的人格基于提示，显著增强了LLMs预测人类反应分布的能力，表明捕捉非理想的、自然性的推理可能需要融合多样推理风格和心理特征的建模技术。本研究得出结论，结合遗传算法的人格基于提示方法对未来增强AI在推理中的“人性化”具有潜力。 

---
# Explainable Distributed Constraint Optimization Problems 

**Title (ZH)**: 可解释的分布式约束优化问题 

**Authors**: Ben Rachmut, Stylianos Loukas Vasileiou, Nimrod Meir Weinstein, Roie Zivan, William Yeoh  

**Link**: [PDF](https://arxiv.org/pdf/2502.14102)  

**Abstract**: The Distributed Constraint Optimization Problem (DCOP) formulation is a powerful tool to model cooperative multi-agent problems that need to be solved distributively. A core assumption of existing approaches is that DCOP solutions can be easily understood, accepted, and adopted, which may not hold, as evidenced by the large body of literature on Explainable AI. In this paper, we propose the Explainable DCOP (X-DCOP) model, which extends a DCOP to include its solution and a contrastive query for that solution. We formally define some key properties that contrastive explanations must satisfy for them to be considered as valid solutions to X-DCOPs as well as theoretical results on the existence of such valid explanations. To solve X-DCOPs, we propose a distributed framework as well as several optimizations and suboptimal variants to find valid explanations. We also include a human user study that showed that users, not surprisingly, prefer shorter explanations over longer ones. Our empirical evaluations showed that our approach can scale to large problems, and the different variants provide different options for trading off explanation lengths for smaller runtimes. Thus, our model and algorithmic contributions extend the state of the art by reducing the barrier for users to understand DCOP solutions, facilitating their adoption in more real-world applications. 

**Abstract (ZH)**: 分布式约束优化问题（DCOP）建模是一种强大的工具，可用于描述需要分布式求解的合作多智能体问题。现有方法的核心假设是DCOP解决方案易于理解、接受和采用，但这一假设可能并不总是成立，这一点在大量的可解释AI研究文献中已有证据表明。在本文中，我们提出了可解释DCOP（X-DCOP）模型，该模型将一个DCOP扩展为包括解决方案及其对比查询。我们正式定义了可解释的对比解释必须满足的一些关键属性，使它们能够被视为X-DCOP的有效解决方案，并且还讨论了此类有效解释的存在性理论结果。为了求解X-DCOP，我们提出了一种分布式框架，并提出了几种优化和次优变体，以找到有效的解释。我们还包括了一项用户研究，结果显示，不出所料，用户更偏好较短的解释而非较长的。我们的实证评估表明，我们的方法可以扩展到大问题，不同的变体可以在解释长度和更短的运行时间之间提供不同的权衡选项。因此，我们的模型和算法贡献通过降低用户理解DCOP解决方案的障碍，从而在更多实际应用中促进了其采用，从而扩展了现有技术的边界。 

---
# Investigating Non-Transitivity in LLM-as-a-Judge 

**Title (ZH)**: 探究LLM作为法官时的非传递性现象 

**Authors**: Yi Xu, Laura Ruis, Tim Rocktäschel, Robert Kirk  

**Link**: [PDF](https://arxiv.org/pdf/2502.14074)  

**Abstract**: Automatic evaluation methods based on large language models (LLMs) are emerging as the standard tool for assessing the instruction-following abilities of LLM-based agents. The most common method in this paradigm, pairwise comparisons with a baseline model, critically depends on the assumption of transitive preferences. However, the validity of this assumption remains largely unexplored. In this study, we investigate the presence of non-transitivity within the AlpacaEval framework and analyze its effects on model rankings. We find that LLM judges exhibit non-transitive preferences, leading to rankings that are sensitive to the choice of the baseline model. To mitigate this issue, we show that round-robin tournaments combined with Bradley-Terry models of preference can produce more reliable rankings. Notably, our method increases both the Spearman correlation and the Kendall correlation with Chatbot Arena (95.0% -> 96.4% and 82.1% -> 86.3% respectively). To address the computational cost of round-robin tournaments, we propose Swiss-Wise Iterative Matchmaking (Swim) tournaments, using a dynamic matching strategy to capture the benefits of round-robin tournaments while maintaining computational efficiency. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的自动评估方法正在成为评估LLM驱动代理的指令遵循能力的标准工具。这一范式中最常见的方法是使用基线模型进行成对比较，这种方法严重依赖于传递偏好假设。然而，这一假设的有效性尚未得到充分探索。本研究旨在调查AlpacaEval框架内的非传递性现象及其对模型排名的影响。我们发现，LLM评判者表现出非传递性偏好，导致排名对基线模型的选择非常敏感。为缓解这一问题，我们证明了循环赛结合布雷德利-泰利（Bradley-Terry）偏好模型可以产生更可靠的排名。值得注意的是，我们的方法提高了与Chatbot Arena的斯皮尔曼等级相关性和肯德尔等级相关性（分别为95.0% -> 96.4% 和82.1% -> 86.3%）。为应对循环赛的计算成本，我们提出了“智胜”循环赛（Swim）方法，通过动态匹配策略同时捕捉循环赛的优势和保持计算效率。 

---
# LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention 

**Title (ZH)**: LServe：高效处理长序列的统一稀疏注意力大规模语言模型服务 

**Authors**: Shang Yang, Junxian Guo, Haotian Tang, Qinghao Hu, Guangxuan Xiao, Jiaming Tang, Yujun Lin, Zhijian Liu, Yao Lu, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.14866)  

**Abstract**: Large language models (LLMs) have shown remarkable potential in processing long sequences, yet efficiently serving these long-context models remains challenging due to the quadratic computational complexity of attention in the prefilling stage and the large memory footprint of the KV cache in the decoding stage. To address these issues, we introduce LServe, an efficient system that accelerates long-sequence LLM serving via hybrid sparse attention. This method unifies different hardware-friendly, structured sparsity patterns for both prefilling and decoding attention into a single framework, where computations on less important tokens are skipped block-wise. LServe demonstrates the compatibility of static and dynamic sparsity in long-context LLM attention. This design enables multiplicative speedups by combining these optimizations. Specifically, we convert half of the attention heads to nearly free streaming heads in both the prefilling and decoding stages. Additionally, we find that only a constant number of KV pages is required to preserve long-context capabilities, irrespective of context length. We then design a hierarchical KV page selection policy that dynamically prunes KV pages based on query-centric similarity. On average, LServe accelerates LLM prefilling by up to 2.9x and decoding by 1.3-2.1x over vLLM, maintaining long-context accuracy. Code is released at this https URL. 

**Abstract (ZH)**: 大语言模型（LLMs）在处理长序列方面展示了显著的潜力，但由于预填充阶段中的注意力机制所导致的二次计算复杂度以及解码阶段中KV缓存的大内存占用，高效地服务于这些长上下文模型仍是一个挑战。为了解决这些问题，我们引入了LServe系统，该系统通过混合稀疏注意力来加速长序列LLM的服务。该方法将预填充和解码中的不同硬件友好型结构化稀疏模式统一到一个框架中，其中对不重要令牌的计算以块为单位进行跳过。LServe展示出静态和动态稀疏性在长上下文LLM注意力机制中的兼容性。这种设计通过结合这些优化而实现了乘法加速。具体来说，在预填充和解码阶段，我们转换了大约一半的注意力头为几乎免费的流式注意力头。此外，我们发现，只需要一个常数数量的KV页面即可保持长上下文能力，与其上下文长度无关。我们还设计了一种分层的KV页面选择策略，该策略基于查询为中心的相似性动态剪枝KV页面。在平均情况下，LServe将LLM预填充加速2.9倍，解码加速1.3-2.1倍，同时保持长上下文准确性。代码已发布于此 <https://> 地址。 

---
# Interpretable Text Embeddings and Text Similarity Explanation: A Primer 

**Title (ZH)**: 可解释的文本嵌入与文本相似性解释：入门指南 

**Authors**: Juri Opitz, Lucas Möller, Andrianos Michail, Simon Clematide  

**Link**: [PDF](https://arxiv.org/pdf/2502.14862)  

**Abstract**: Text embeddings and text embedding models are a backbone of many AI and NLP systems, particularly those involving search. However, interpretability challenges persist, especially in explaining obtained similarity scores, which is crucial for applications requiring transparency. In this paper, we give a structured overview of interpretability methods specializing in explaining those similarity scores, an emerging research area. We study the methods' individual ideas and techniques, evaluating their potential for improving interpretability of text embeddings and explaining predicted similarities. 

**Abstract (ZH)**: 文本嵌入和文本嵌入模型是许多AI和自然语言处理（NLP）系统的核心组件，尤其是在涉及搜索的应用中。然而，解释这些模型的可解释性挑战依然存在，特别是在解释获得的相似度分数时，这在需要透明度的应用中尤为重要。本文提供了一种结构化的概述，专门介绍解释这些相似度分数的可解释性方法，这是一个新兴的研究领域。我们研究了这些方法的个体理念和技术，并评估其提高文本嵌入可解释性和解释预测相似度的潜力。 

---
# FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling 

**Title (ZH)**: FR-Spec：通过频率排序推测性采样加速大规模词汇语言模型 

**Authors**: Weilin Zhao, Tengyu Pan, Xu Han, Yudi Zhang, Ao Sun, Yuxiang Huang, Kaihuo Zhang, Weilun Zhao, Yuxuan Li, Jianyong Wang, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.14856)  

**Abstract**: Speculative sampling has emerged as an important technique for accelerating the auto-regressive generation process of large language models (LLMs) by utilizing a draft-then-verify mechanism to produce multiple tokens per forward pass. While state-of-the-art speculative sampling methods use only a single layer and a language modeling (LM) head as the draft model to achieve impressive layer compression, their efficiency gains are substantially reduced for large-vocabulary LLMs, such as Llama-3-8B with a vocabulary of 128k tokens. To address this, we present FR-Spec, a frequency-ranked speculative sampling framework that optimizes draft candidate selection through vocabulary space compression. By constraining the draft search to a frequency-prioritized token subset, our method reduces LM Head computation overhead by 75% while ensuring the equivalence of the final output distribution. Experiments across multiple datasets demonstrate an average of 1.12$\times$ speedup over the state-of-the-art speculative sampling method EAGLE-2. 

**Abstract (ZH)**: 投机抽样作为一种重要技术，通过利用先拟后验机制，在每一前向传播过程中生成多个标记，已经成为了加速大规模语言模型（LLMs）的自回归生成过程的关键方法。当前最先进的投机采样方法仅使用一层和一个语言模型（LM）头作为草稿模型，从而实现了显著的层压缩，但其效率提升在具有大词汇量的语言模型（如词汇量达128K的Llama-3-8B）中大幅减少。为解决这一问题，我们提出了一种基于频率排名的投机抽样框架FR-Spec，通过词汇空间压缩优化草稿候选的选择。通过将草稿搜索限定在频率优先的令牌子集中，我们的方法在保持最终输出分布等价性的同时，将LM头的计算开销降低了75%。我们在多个数据集上的实验结果显示，FR-Spec相比最先进的投机采样方法EAGLE-2，平均实现了1.12倍的速度提升。 

---
# Revealing and Mitigating Over-Attention in Knowledge Editing 

**Title (ZH)**: 揭示并缓解知识编辑中的过度关注问题 

**Authors**: Pinzheng Wang, Zecheng Tang, Keyan Zhou, Juntao Li, Qiaoming Zhu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14838)  

**Abstract**: Large Language Models have demonstrated superior performance across a wide range of tasks, but they still exhibit undesirable errors due to incorrect knowledge learned from the training data. To avoid this, knowledge editing methods emerged to precisely edit the specific model knowledge via efficiently modifying a very small percentage of parameters. % However, those methods can lead to the problem of Specificity Failure: when the content related to the edited knowledge occurs in the context, it can inadvertently corrupt other pre-existing knowledge. However, those methods can lead to the problem of Specificity Failure, where the existing knowledge and capabilities are severely degraded due to editing. Our preliminary indicates that Specificity Failure primarily stems from the model's attention heads assigning excessive attention scores to entities related to the edited knowledge, thereby unduly focusing on specific snippets within the context, which we denote as the Attention Drift phenomenon. To mitigate such Attention Drift issue, we introduce a simple yet effective method Selective Attention Drift Restriction}(SADR), which introduces an additional regularization term during the knowledge editing process to restrict changes in the attention weight distribution, thereby preventing undue focus on the edited entity. Experiments on five frequently used strong LLMs demonstrate the effectiveness of our method, where SADR can significantly mitigate Specificity Failure in the predominant knowledge editing tasks. 

**Abstract (ZH)**: 大型语言模型已在广泛的任务中展示出了卓越的性能，但它们仍然会出现由于训练数据中不正确知识导致的不良错误。为避免这一问题，知识编辑方法应运而生，通过高效地修改一小部分参数来精确编辑特定模型知识。然而，这些方法可能会导致特定性失败（Specificity Failure）的问题：当与编辑知识相关的上下文内容出现时，可能会无意中破坏其他现有的知识。

我们的初步研究表明，特定性失败主要源于模型的注意力头向与编辑知识相关联的实体分配了过高的注意力分数，从而使其过度关注上下文中的特定片段，我们将其称为注意力偏移（Attention Drift）现象。为缓解这种注意力偏移问题，我们提出了一种简单而有效的解决方法——选择性注意力偏移限制（Selective Attention Drift Restriction, SADR）。该方法在知识编辑过程中引入了一个额外的正则化项，以限制注意力权重分布的变化，从而防止过度关注编辑的对象。我们对五种常用的强语言模型进行了实验，结果显示我们的方法在主要的知识编辑任务中能够显著减轻特定性失败问题。 

---
# Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs 

**Title (ZH)**: 向经济高效的推理迈进：使 DeepSeek 的多头潜层注意力能够在任意基于Transformer的大型语言模型中生效 

**Authors**: Tao Ji, Bin Guo, Yuanbin Wu, Qipeng Guo, Lixing Shen, Zhan Chen, Xipeng Qiu, Qi Zhang, Tao Gui  

**Link**: [PDF](https://arxiv.org/pdf/2502.14837)  

**Abstract**: Multi-head Latent Attention (MLA) is an innovative architecture proposed by DeepSeek, designed to ensure efficient and economical inference by significantly compressing the Key-Value (KV) cache into a latent vector. Compared to MLA, standard LLMs employing Multi-Head Attention (MHA) and its variants such as Grouped-Query Attention (GQA) exhibit significant cost disadvantages. Enabling well-trained LLMs (e.g., Llama) to rapidly adapt to MLA without pre-training from scratch is both meaningful and challenging. This paper proposes the first data-efficient fine-tuning method for transitioning from MHA to MLA (MHA2MLA), which includes two key components: for partial-RoPE, we remove RoPE from dimensions of queries and keys that contribute less to the attention scores, for low-rank approximation, we introduce joint SVD approximations based on the pre-trained parameters of keys and values. These carefully designed strategies enable MHA2MLA to recover performance using only a small fraction (0.3% to 0.6%) of the data, significantly reducing inference costs while seamlessly integrating with compression techniques such as KV cache quantization. For example, the KV cache size of Llama2-7B is reduced by 92.19%, with only a 0.5% drop in LongBench performance. 

**Abstract (ZH)**: 多头潜在注意（MLA）是一种由DeepSeek提出的创新架构，旨在通过大幅压缩键值（KV）缓存为潜在向量，从而实现高效且经济的推理。与MLA相比，使用多头注意（MHA）及其变体如组查询注意力（GQA）的标准大型语言模型（LLM）在成本上存在显著劣势。让具有良好训练的LLM（例如Llama）能够快速适应MLA而无需从零开始预训练，既具有重要意义也充满挑战。本文提出了从MHA过渡到MLA（MHA2MLA）的第一个数据高效微调方法，其中包括两个关键组件：对于部分RoPE，我们从贡献较小于注意力分数的查询和关键维度中移除RoPE；对于低秩近似，我们引入基于预训练参数的键值联合SVD近似。这些精心设计的策略使MHA2MLA能够在仅使用数据的极小部分（0.3%到0.6%）的情况下恢复性能，显著降低了推理成本，同时与KV缓存量化等压缩技术无缝集成。例如，Llama2-7B的KV缓存大小减少了92.19%，仅性能下降了0.5%，而在LongBench上的性能下降几乎可以忽略不计。 

---
# LongWriter-V: Enabling Ultra-Long and High-Fidelity Generation in Vision-Language Models 

**Title (ZH)**: 长写作者-V： Enables 超长高保真生成能力提升在视觉-语言模型中的应用 

**Authors**: Shangqing Tu, Yucheng Wang, Daniel Zhang-Li, Yushi Bai, Jifan Yu, Yuhao Wu, Lei Hou, Huiqin Liu, Zhiyuan Liu, Bin Xu, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.14834)  

**Abstract**: Existing Large Vision-Language Models (LVLMs) can process inputs with context lengths up to 128k visual and text tokens, yet they struggle to generate coherent outputs beyond 1,000 words. We find that the primary limitation is the absence of long output examples during supervised fine-tuning (SFT). To tackle this issue, we introduce LongWriter-V-22k, a SFT dataset comprising 22,158 examples, each with multiple input images, an instruction, and corresponding outputs ranging from 0 to 10,000 words. Moreover, to achieve long outputs that maintain high-fidelity to the input images, we employ Direct Preference Optimization (DPO) to the SFT model. Given the high cost of collecting human feedback for lengthy outputs (e.g., 3,000 words), we propose IterDPO, which breaks long outputs into segments and uses iterative corrections to form preference pairs with the original outputs. Additionally, we develop MMLongBench-Write, a benchmark featuring six tasks to evaluate the long-generation capabilities of VLMs. Our 7B parameter model, trained with LongWriter-V-22k and IterDPO, achieves impressive performance on this benchmark, outperforming larger proprietary models like GPT-4o. Code and data: this https URL 

**Abstract (ZH)**: 现有的大型视觉语言模型（LVLMs）能够处理多达128k视觉和文本标记的输入，但在产生超过1000字的连贯输出方面存在困难。我们发现，主要的限制在于监督微调（SFT）过程中缺乏长输出示例。为了解决这一问题，我们引入了LongWriter-V-22k，这是一个包含22,158个示例的数据集，每个示例包含多张输入图像、一条指令以及从0到10,000字不等的对应输出。此外，为了生成与输入图像保持高保真度的长输出，我们对SFT模型应用了直接偏好优化（DPO）。鉴于对 lengthy 输出（如3,000字）收集人类反馈的高成本，我们提出了IterDPO，将长输出拆分为段落，并使用迭代校正形成与原始输出的偏好对比对。此外，我们还开发了MMLongBench-Write，这是一个包含六个任务的基准，用于评估VLMs的长生成能力。通过使用LongWriter-V-22k和IterDPO训练的7B参数模型，在该基准上表现出色，超过了更大的私有模型如GPT-4o。代码和数据：[这里提供链接] 

---
# Improving the Diffusability of Autoencoders 

**Title (ZH)**: 提高自编码器的扩散性 

**Authors**: Ivan Skorokhodov, Sharath Girish, Benran Hu, Willi Menapace, Yanyu Li, Rameen Abdal, Sergey Tulyakov, Aliaksandr Siarohin  

**Link**: [PDF](https://arxiv.org/pdf/2502.14831)  

**Abstract**: Latent diffusion models have emerged as the leading approach for generating high-quality images and videos, utilizing compressed latent representations to reduce the computational burden of the diffusion process. While recent advancements have primarily focused on scaling diffusion backbones and improving autoencoder reconstruction quality, the interaction between these components has received comparatively less attention. In this work, we perform a spectral analysis of modern autoencoders and identify inordinate high-frequency components in their latent spaces, which are especially pronounced in the autoencoders with a large bottleneck channel size. We hypothesize that this high-frequency component interferes with the coarse-to-fine nature of the diffusion synthesis process and hinders the generation quality. To mitigate the issue, we propose scale equivariance: a simple regularization strategy that aligns latent and RGB spaces across frequencies by enforcing scale equivariance in the decoder. It requires minimal code changes and only up to 20K autoencoder fine-tuning steps, yet significantly improves generation quality, reducing FID by 19% for image generation on ImageNet-1K 256x256 and FVD by at least 44% for video generation on Kinetics-700 17x256x256. 

**Abstract (ZH)**: 潜扩散模型已经成为生成高质量图像和视频的领先方法，通过使用压缩的潜在表示来减少扩散过程的计算负担。尽管近期的进步主要集中在扩展扩散骨干网络和提高自编码器重构质量上，这些组件之间的相互作用却得到了相对较少的关注。在本文中，我们对现代自编码器进行了光谱分析，并确定其潜在空间中存在异常高的高频成分，尤其是在瓶颈通道尺寸较大的自编码器中更为明显。我们假设这些高频成分干扰了扩散合成过程中的精细到粗糙层次特性，从而阻碍了生成质量。为了缓解这一问题，我们提出了一种尺度等变性（scale equivariance）：这是一种简单正则化策略，通过在解码器中施加尺度等变性，使潜在空间和RGB空间在不同频率上保持一致。该策略只需少量的代码更改，仅需不到20000次自编码器微调步骤，但却显著提高了生成质量，对于ImageNet-1K 256x256图像生成而言，降低了FID约19%，而对于Kinetics-700 17x256x256视频生成而言，降低了FVD至少44%。 

---
# Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs 

**Title (ZH)**: fine-tuned LLMs 中的中间层表示对齐以实现跨语言迁移学习 

**Authors**: Danni Liu, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2502.14830)  

**Abstract**: While large language models demonstrate remarkable capabilities at task-specific applications through fine-tuning, extending these benefits across diverse languages is essential for broad accessibility. However, effective cross-lingual transfer is hindered by LLM performance gaps across languages and the scarcity of fine-tuning data in many languages. Through analysis of LLM internal representations from over 1,000+ language pairs, we discover that middle layers exhibit the strongest potential for cross-lingual alignment. Building on this finding, we propose a middle-layer alignment objective integrated into task-specific training. Our experiments on slot filling, machine translation, and structured text generation show consistent improvements in cross-lingual transfer, especially to lower-resource languages. The method is robust to the choice of alignment languages and generalizes to languages unseen during alignment. Furthermore, we show that separately trained alignment modules can be merged with existing task-specific modules, improving cross-lingual capabilities without full re-training. Our code is publicly available (this https URL). 

**Abstract (ZH)**: 通过微调展示了在特定任务应用中的卓越能力的大规模语言模型，使得这些优势能够在多种语言中得到广泛应用变得尤为重要，以实现广泛的可访问性。然而，有效的跨语言迁移受到不同语言之间的表现差距以及许多语言缺乏微调数据的限制。通过对逾千个语言对的大型语言模型内部表示进行分析，我们发现中间层具有最强的跨语言对齐潜力。基于这一发现，我们提出了一种集成在特定任务训练中的中间层对齐目标。我们在槽填充、机器翻译和结构化文本生成任务上的实验表明，跨语言迁移得到一致的改进，尤其是在低资源语言方面。该方法对对齐语言的选择具有鲁棒性，并在未见过对齐的语言上也具有泛化能力。此外，我们展示了单独训练的对齐模块可以与现有的特定任务模块合并，从而改进跨语言能力而无需进行全面的重新训练。我们的代码已公开 (此链接)。 

---
# Exploring Advanced Techniques for Visual Question Answering: A Comprehensive Comparison 

**Title (ZH)**: 探索视觉问答领域的高级技术：全面对比分析 

**Authors**: Aiswarya Baby, Tintu Thankom Koshy  

**Link**: [PDF](https://arxiv.org/pdf/2502.14827)  

**Abstract**: Visual Question Answering (VQA) has emerged as a pivotal task in the intersection of computer vision and natural language processing, requiring models to understand and reason about visual content in response to natural language questions. Analyzing VQA datasets is essential for developing robust models that can handle the complexities of multimodal reasoning. Several approaches have been developed to examine these datasets, each offering distinct perspectives on question diversity, answer distribution, and visual-textual correlations. Despite significant progress, existing VQA models face challenges related to dataset bias, limited model complexity, commonsense reasoning gaps, rigid evaluation methods, and generalization to real world scenarios. This paper presents a comprehensive comparative study of five advanced VQA models: ABC-CNN, KICNLE, Masked Vision and Language Modeling, BLIP-2, and OFA, each employing distinct methodologies to address these challenges. 

**Abstract (ZH)**: 视觉问答（VQA）已成为计算机视觉与自然语言处理交叉领域的一个关键任务，要求模型能够理解并根据自然语言问题对视觉内容进行推理。分析VQA数据集是开发鲁棒模型、处理多模态推理复杂性的重要步骤。已有多种方法用于研究这些数据集，每种方法都提供了对问题多样性、答案分布及视觉-文本关联的不同视角。尽管取得了显著进展，现有的VQA模型仍面临数据集偏差、模型复杂度有限、常识推理不足、僵化的评估方法以及在现实场景中泛化能力差等挑战。本文对五种先进的VQA模型——ABC-CNN、KICNLE、遮蔽视觉与语言建模、BLIP-2和OFA——进行了全面的比较研究，每种模型采用独特的研究方法来应对上述挑战。 

---
# eC-Tab2Text: Aspect-Based Text Generation from e-Commerce Product Tables 

**Title (ZH)**: eC-Tab2Text：从电子商务产品表格中生成方面导向的文本 

**Authors**: Luis Antonio Gutiérrez Guanilo, Mir Tafseer Nayeem, Cristian López, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2502.14820)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional versatility across diverse domains, yet their application in e-commerce remains underexplored due to a lack of domain-specific datasets. To address this gap, we introduce eC-Tab2Text, a novel dataset designed to capture the intricacies of e-commerce, including detailed product attributes and user-specific queries. Leveraging eC-Tab2Text, we focus on text generation from product tables, enabling LLMs to produce high-quality, attribute-specific product reviews from structured tabular data. Fine-tuned models were rigorously evaluated using standard Table2Text metrics, alongside correctness, faithfulness, and fluency assessments. Our results demonstrate substantial improvements in generating contextually accurate reviews, highlighting the transformative potential of tailored datasets and fine-tuning methodologies in optimizing e-commerce workflows. This work highlights the potential of LLMs in e-commerce workflows and the essential role of domain-specific datasets in tailoring them to industry-specific challenges. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多种领域中展现了极高的灵活性，但在电子商务中的应用仍然相对不足，原因在于缺乏特定领域的数据集。为了解决这一问题，我们提出了一种新的数据集eC-Tab2Text，该数据集旨在捕捉电子商务的复杂性，包括详细的商品属性和用户特定的查询。通过利用eC-Tab2Text，我们关注于从产品表生成文本，使LLMs能够从结构化表格数据中生成高质量且具有属性特定性的商品评论。经过微调的模型被使用标准的Table2Text指标进行了严格的评估，并结合正确性、忠实度和流畅性的评估。我们的结果显示生成上下文相关评论的显著改进，突显了定制数据集和微调方法在优化电子商务工作流程中的潜力。这项工作展示了LLMs在电子商务工作流程中的巨大潜力，并强调了特定领域数据集在针对行业特定挑战进行调整中的关键作用。 

---
# FetalCLIP: A Visual-Language Foundation Model for Fetal Ultrasound Image Analysis 

**Title (ZH)**: 胎儿CLIP：一种用于胎儿超声图像分析的视觉-语言基础模型 

**Authors**: Fadillah Maani, Numan Saeed, Tausifa Saleem, Zaid Farooq, Hussain Alasmawi, Werner Diehl, Ameera Mohammad, Gareth Waring, Saudabi Valappi, Leanne Bricker, Mohammad Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2502.14807)  

**Abstract**: Foundation models are becoming increasingly effective in the medical domain, offering pre-trained models on large datasets that can be readily adapted for downstream tasks. Despite progress, fetal ultrasound images remain a challenging domain for foundation models due to their inherent complexity, often requiring substantial additional training and facing limitations due to the scarcity of paired multimodal data. To overcome these challenges, here we introduce FetalCLIP, a vision-language foundation model capable of generating universal representation of fetal ultrasound images. FetalCLIP was pre-trained using a multimodal learning approach on a diverse dataset of 210,035 fetal ultrasound images paired with text. This represents the largest paired dataset of its kind used for foundation model development to date. This unique training approach allows FetalCLIP to effectively learn the intricate anatomical features present in fetal ultrasound images, resulting in robust representations that can be used for a variety of downstream applications. In extensive benchmarking across a range of key fetal ultrasound applications, including classification, gestational age estimation, congenital heart defect (CHD) detection, and fetal structure segmentation, FetalCLIP outperformed all baselines while demonstrating remarkable generalizability and strong performance even with limited labeled data. We plan to release the FetalCLIP model publicly for the benefit of the broader scientific community. 

**Abstract (ZH)**: 基础模型在医疗领域中的应用越来越有效，它们可以通过大规模数据集进行预训练，并且可以轻松适应下游任务。尽管取得了进展，但由于胎儿超声图像本身复杂性较高，基础模型在这一领域依然面临诸多挑战，通常需要大量的额外训练，并且受限于配对多模态数据的稀缺性。为了解决这些挑战，我们在此介绍了FetalCLIP，这是一种能够生成胎儿超声图像通用表示的基础视觉-语言模型。FetalCLIP通过在包含210,035张胎儿超声图像和相应文本信息的多元化数据集上采用多模态学习方法进行预训练。这代表了迄今为止用于基础模型开发的最大规模的配对数据集。这种独特的训练方法使FetalCLIP能够有效地学习胎儿超声图像中存在的复杂解剖特征，从而产生稳健的表示，这些表示可以应用于多种下游应用。在包括分类、胎龄估计、先天性心脏病（CHD）检测和胎儿结构分割等一系列关键胎儿超声应用的广泛基准测试中，FetalCLIP在所有对照基线模型中表现最佳，并且展示了出色的泛化能力和在有限标记数据下依然强大的性能。我们计划将FetalCLIP模型公开发布，以惠及更广泛的科学界。 

---
# From RAG to Memory: Non-Parametric Continual Learning for Large Language Models 

**Title (ZH)**: 从RAG到记忆：大型语言模型的非参数连续学习 

**Authors**: Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2502.14802)  

**Abstract**: Our ability to continuously acquire, organize, and leverage knowledge is a key feature of human intelligence that AI systems must approximate to unlock their full potential. Given the challenges in continual learning with large language models (LLMs), retrieval-augmented generation (RAG) has become the dominant way to introduce new information. However, its reliance on vector retrieval hinders its ability to mimic the dynamic and interconnected nature of human long-term memory. Recent RAG approaches augment vector embeddings with various structures like knowledge graphs to address some of these gaps, namely sense-making and associativity. However, their performance on more basic factual memory tasks drops considerably below standard RAG. We address this unintended deterioration and propose HippoRAG 2, a framework that outperforms standard RAG comprehensively on factual, sense-making, and associative memory tasks. HippoRAG 2 builds upon the Personalized PageRank algorithm used in HippoRAG and enhances it with deeper passage integration and more effective online use of an LLM. This combination pushes this RAG system closer to the effectiveness of human long-term memory, achieving a 7% improvement in associative memory tasks over the state-of-the-art embedding model while also exhibiting superior factual knowledge and sense-making memory capabilities. This work paves the way for non-parametric continual learning for LLMs. Our code and data will be released at this https URL. 

**Abstract (ZH)**: 我们持续获取、组织和利用知识的能力是人类智能的关键特征，AI系统必须模拟这种能力才能发挥其全部潜力。鉴于大规模语言模型（LLMs）在持续学习方面面临的挑战，检索增强生成（RAG）已成为引入新信息的主要方式。然而，这种方法依赖于向量检索，限制了其模仿人类长期记忆动态性和互联性的能力。近期的RAG方法通过引入知识图等结构来增强向量嵌入，以解决部分差距，如意义构建和关联性。然而，这些方法在基本事实记忆任务上的表现显著低于标准RAG系统。我们解决了这种意外下降，并提出了一种新的框架HippoRAG 2，该框架在事实记忆、意义构建和关联性记忆任务上全面超越了标准RAG系统。HippoRAG 2 基于HippoRAG中使用的个性化PageRank算法，并通过加深段落集成和更有效的在线使用LLM对其进行增强。这种结合使RAG系统更接近人类长期记忆的有效性，在关联记忆任务中超过最先进的嵌入模型7%，同时还表现出更强的事实知识和意义构建记忆能力。这项工作开辟了大规模语言模型非参数持续学习的可能。我们的代码和数据将在以下网址发布：[该网址]。 

---
# A Survey on Text-Driven 360-Degree Panorama Generation 

**Title (ZH)**: 文本驱动的360度全景图生成综述 

**Authors**: Hai Wang, Xiaoyu Xiang, Weihao Xia, Jing-Hao Xue  

**Link**: [PDF](https://arxiv.org/pdf/2502.14799)  

**Abstract**: The advent of text-driven 360-degree panorama generation, enabling the synthesis of 360-degree panoramic images directly from textual descriptions, marks a transformative advancement in immersive visual content creation. This innovation significantly simplifies the traditionally complex process of producing such content. Recent progress in text-to-image diffusion models has accelerated the rapid development in this emerging field. This survey presents a comprehensive review of text-driven 360-degree panorama generation, offering an in-depth analysis of state-of-the-art algorithms and their expanding applications in 360-degree 3D scene generation. Furthermore, we critically examine current limitations and propose promising directions for future research. A curated project page with relevant resources and research papers is available at this https URL. 

**Abstract (ZH)**: 文本驱动的360度全景生成的出现，使得可以直接从文本描述合成360度全景图像，标志着沉浸式视觉内容创作的一个变革性进步。这一创新极大地简化了制作此类内容的传统复杂过程。近年来，文本到图像扩散模型的进步加速了这一新兴领域的快速发展。本文综述了文本驱动的360度全景生成技术，深入分析了目前最先进的算法及其在360度三维场景生成中的扩展应用。此外，我们还批判性地审视了当前的限制，并提出了未来研究的有希望的方向。一个精选项目页面，提供了相关资源和研究论文，可在以下链接访问：[这个https链接]。 

---
# Rapid Word Learning Through Meta In-Context Learning 

**Title (ZH)**: 通过元上下文学习快速词汇学习 

**Authors**: Wentao Wang, Guangyuan Jiang, Tal Linzen, Brenden M. Lake  

**Link**: [PDF](https://arxiv.org/pdf/2502.14791)  

**Abstract**: Humans can quickly learn a new word from a few illustrative examples, and then systematically and flexibly use it in novel contexts. Yet the abilities of current language models for few-shot word learning, and methods for improving these abilities, are underexplored. In this study, we introduce a novel method, Meta-training for IN-context learNing Of Words (Minnow). This method trains language models to generate new examples of a word's usage given a few in-context examples, using a special placeholder token to represent the new word. This training is repeated on many new words to develop a general word-learning ability. We find that training models from scratch with Minnow on human-scale child-directed language enables strong few-shot word learning, comparable to a large language model (LLM) pre-trained on orders of magnitude more data. Furthermore, through discriminative and generative evaluations, we demonstrate that finetuning pre-trained LLMs with Minnow improves their ability to discriminate between new words, identify syntactic categories of new words, and generate reasonable new usages and definitions for new words, based on one or a few in-context examples. These findings highlight the data efficiency of Minnow and its potential to improve language model performance in word learning tasks. 

**Abstract (ZH)**: 人类可以从少量的示例中快速学习新词，并且能够系统地、灵活地在新的语境中使用这些新词。然而，当前语言模型在少量示例学习词汇的能力及其改进方法尚未得到充分探索。本研究引入了一种新颖的方法——基于上下文学习词的元训练（Minnow）。该方法通过使用一个特殊的占位符标记来表示新词，并在此基础上，根据少量上下文示例生成新词的使用示例，进行训练。此训练方法被应用于多种新词以开发其普遍的词汇学习能力。我们发现，使用Minnow从头开始训练规模适中的面向儿童的语言数据，能够实现强劲的少量示例词汇学习，这一能力与更大数据量预训练的大型语言模型（LLM）相当。此外，通过区分性和生成性的评估，我们证明了使用Minnow微调预训练的大型语言模型能够增强其区分新词的能力、识别新词的句法类别，并根据一两个上下文示例生成合理的新用法和定义。这些发现突出了Minnow的数据效率，并展示了其在词汇学习任务中提高语言模型性能的潜力。 

---
# Ray-Tracing for Conditionally Activated Neural Networks 

**Title (ZH)**: 条件激活神经网络的射线追踪方法 

**Authors**: Claudio Gallicchio, Giuseppe Nuti  

**Link**: [PDF](https://arxiv.org/pdf/2502.14788)  

**Abstract**: In this paper, we introduce a novel architecture for conditionally activated neural networks combining a hierarchical construction of multiple Mixture of Experts (MoEs) layers with a sampling mechanism that progressively converges to an optimized configuration of expert activation. This methodology enables the dynamic unfolding of the network's architecture, facilitating efficient path-specific training. Experimental results demonstrate that this approach achieves competitive accuracy compared to conventional baselines while significantly reducing the parameter count required for inference. Notably, this parameter reduction correlates with the complexity of the input patterns, a property naturally emerging from the network's operational dynamics without necessitating explicit auxiliary penalty functions. 

**Abstract (ZH)**: 在本文中，我们提出了一种结合分层构建的多个专家混合（Experts Mixtures, MoEs）层与逐步收敛到优化的专家激活配置的采样机制的新颖架构。该方法允许网络架构的动态展开，从而促进路径特定的高效训练。实验结果表明，该方法在与传统基准相比时实现了相当的准确性，同时显著减少了推理所需的参数量。值得注意的是，这种参数量的减少与输入模式的复杂性相关，这一特性自然地从网络的操作动态中产生，无需使用显式的辅助惩罚函数。 

---
# SigLIP 2: Multilingual Vision-Language Encoders with Improved Semantic Understanding, Localization, and Dense Features 

**Title (ZH)**: SigLIP 2：具备增强语义理解、定位能力和密集特征的多语言视觉-语言编码器 

**Authors**: Michael Tschannen, Alexey Gritsenko, Xiao Wang, Muhammad Ferjad Naeem, Ibrahim Alabdulmohsin, Nikhil Parthasarathy, Talfan Evans, Lucas Beyer, Ye Xia, Basil Mustafa, Olivier Hénaff, Jeremiah Harmsen, Andreas Steiner, Xiaohua Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2502.14786)  

**Abstract**: We introduce SigLIP 2, a family of new multilingual vision-language encoders that build on the success of the original SigLIP. In this second iteration, we extend the original image-text training objective with several prior, independently developed techniques into a unified recipe -- this includes captioning-based pretraining, self-supervised losses (self-distillation, masked prediction) and online data curation. With these changes, SigLIP 2 models outperform their SigLIP counterparts at all model scales in core capabilities, including zero-shot classification, image-text retrieval, and transfer performance when extracting visual representations for Vision-Language Models (VLMs). Furthermore, the new training recipe leads to significant improvements on localization and dense prediction tasks. We also train variants which support multiple resolutions and preserve the input's native aspect ratio. Finally, we train on a more diverse data-mixture that includes de-biasing techniques, leading to much better multilingual understanding and improved fairness. To allow users to trade off inference cost with performance, we release model checkpoints at four sizes: ViT-B (86M), L (303M), So400m (400M), and g (1B). 

**Abstract (ZH)**: 我们介绍了一种新的多语言视觉-语言编码器家族——SigLIP 2，该家族建立在原始SigLIP的成功基础上。在这一改进版本中，我们将原始的图像-文本训练目标与几种先前独立开发的技术统一成一个综合配方——这包括基于描述符的预训练、自监督损失（自我蒸馏、掩码预测）以及在线数据管理。通过这些改进，SigLIP 2 模型在所有模型规模的核心能力上都优于其原始的SigLIP模型，包括零样本分类、图像-文本检索以及用于视觉语言模型（VLM）提取视觉表示的迁移性能。此外，新的训练配方在定位和密集预测任务上带来了显著改进。我们还训练了支持多种分辨率并且保留输入原始长宽比的变体。最后，我们使用更为多样化的数据混 packageName 能，包括去偏见技术，从而提高了多语言理解和公平性。为了允许用户在推理成本与性能之间进行权衡，我们发布了四种大小的模型检查点：ViT-B（86M）、L（303M）、S（400M）和G（1B）。 

---
# Real-Time Device Reach Forecasting Using HLL and MinHash Data Sketches 

**Title (ZH)**: 使用HLL和MinHash数据概览实现实时设备可达性预测 

**Authors**: Chandrashekar Muniyappa, Kendall Willets, Sriraman Krishnamoorthy  

**Link**: [PDF](https://arxiv.org/pdf/2502.14785)  

**Abstract**: Predicting the right number of TVs (Device Reach) in real-time based on a user-specified targeting attributes is imperative for running multi-million dollar ADs business. The traditional approach of SQL queries to join billions of records across multiple targeting dimensions is extremely slow. As a workaround, many applications will have an offline process to crunch these numbers and present the results after many hours. In our case, the solution was an offline process taking 24 hours to onboard a customer resulting in a potential loss of business. To solve this problem, we have built a new real-time prediction system using MinHash and HyperLogLog (HLL) data sketches to compute the device reach at runtime when a user makes a request. However, existing MinHash implementations do not solve the complex problem of multilevel aggregation and intersection. This work will show how we have solved this problem, in addition, we have improved MinHash algorithm to run 4 times faster using Single Instruction Multiple Data (SIMD) vectorized operations for high speed and accuracy with constant space to process billions of records. Finally, by experiments, we prove that the results are as accurate as traditional offline prediction system with an acceptable error rate of 5%. 

**Abstract (ZH)**: 根据用户指定的投放属性，实时预测适合投放广告的电视设备数量（Device Reach）对于运行动辄数百万美元的广告业务至关重要。传统的通过 SQL 查询在多个目标维度表之间连接数十亿条记录的方式极其缓慢。作为变通方法，许多应用程序将采用批量处理来计算这些数据，并在数小时后呈现结果。在我们的情况下，这个处理过程需要24小时的时间来上线一个客户，这可能导致业务的潜在损失。为了解决这个问题，我们构建了一个新的实时预测系统，使用 MinHash 和 HyperLogLog (HLL) 数据概要统计来计算在用户请求时的设备覆盖范围。然而，现有的 MinHash 实现无法解决多层次聚合和交集的复杂问题。本项工作将展示我们是如何解决这一问题的，并且我们改进了 MinHash 算法，使其利用单一指令多数据 (SIMD) 向量化操作，实现高速且准确的实时处理，同时保持恒定的空间复杂度来处理数十亿条记录。最后，通过实验，我们证明该系统的预测结果与传统的批量预测系统一样准确，且误差率在可接受的5%范围内。 

---
# ReVision: A Dataset and Baseline VLM for Privacy-Preserving Task-Oriented Visual Instruction Rewriting 

**Title (ZH)**: ReVision：一种用于隐私保护任务导向视觉指令重写的数据集和基线多模态模型 

**Authors**: Abhijit Mishra, Richard Noh, Hsiang Fu, Mingda Li, Minji Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.14780)  

**Abstract**: Efficient and privacy-preserving multimodal interaction is essential as AR, VR, and modern smartphones with powerful cameras become primary interfaces for human-computer communication. Existing powerful large vision-language models (VLMs) enabling multimodal interaction often rely on cloud-based processing, raising significant concerns about (1) visual privacy by transmitting sensitive vision data to servers, and (2) their limited real-time, on-device usability. This paper explores Visual Instruction Rewriting, a novel approach that transforms multimodal instructions into text-only commands, allowing seamless integration of lightweight on-device instruction rewriter VLMs (250M parameters) with existing conversational AI systems, enhancing vision data privacy. To achieve this, we present a dataset of over 39,000 examples across 14 domains and develop a compact VLM, pretrained on image captioning datasets and fine-tuned for instruction rewriting. Experimental results, evaluated through NLG metrics such as BLEU, METEOR, and ROUGE, along with semantic parsing analysis, demonstrate that even a quantized version of the model (<500MB storage footprint) can achieve effective instruction rewriting, thus enabling privacy-focused, multimodal AI applications. 

**Abstract (ZH)**: 高效且保护隐私的多模态交互对于AR、VR以及现代配备强大摄像头的智能手机成为人机通信的主要接口至关重要。现有的强大视觉-语言模型（VLMs）虽然能够实现多模态交互，但往往依赖云处理，这引发了许多关于（1）视觉隐私方面的担忧，即传输敏感的视觉数据至服务器，以及（2）其在设备上的实时性和使用便利性的局限性。本文探讨了一种名为视觉指令改写的创新方法，该方法将多模态指令转换为纯文本命令，从而使轻量级的在设备上运行的指令重写VLMs（参数量250M）能够与现有的对话式AI系统无缝集成，从而增强视觉数据的隐私性。

为实现这一目标，我们构建了一个包含超过39,000个样本的跨14个领域的数据集，并开发了一个紧凑的VLM，该模型预训练于图像字幕数据集，并针对指令重写进行了微调。通过使用诸如BLEU、METEOR和ROUGE等自然语言生成评估指标及语义解析分析，实验结果表明，即使是最小量化版本的模型（存储占用<500MB），也能实现有效的指令重写，从而推动具有隐私保护的多模态AI应用的发展。 

---
# Harnessing PDF Data for Improving Japanese Large Multimodal Models 

**Title (ZH)**: 利用PDF数据提升日语大规模多模态模型性能 

**Authors**: Jeonghun Baek, Akiko Aizawa, Kiyoharu Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2502.14778)  

**Abstract**: Large Multimodal Models (LMMs) have demonstrated strong performance in English, but their effectiveness in Japanese remains limited due to the lack of high-quality training data. Current Japanese LMMs often rely on translated English datasets, restricting their ability to capture Japan-specific cultural knowledge. To address this, we explore the potential of Japanese PDF data as a training resource, an area that remains largely underutilized. We introduce a fully automated pipeline that leverages pretrained models to extract image-text pairs from PDFs through layout analysis, OCR, and vision-language pairing, removing the need for manual annotation. Additionally, we construct instruction data from extracted image-text pairs to enrich the training data. To evaluate the effectiveness of PDF-derived data, we train Japanese LMMs and assess their performance on the Japanese LMM Benchmark. Our results demonstrate substantial improvements, with performance gains ranging from 3.9% to 13.8% on Heron-Bench. Further analysis highlights the impact of PDF-derived data on various factors, such as model size and language models, reinforcing its value as a multimodal resource for Japanese LMMs. We plan to make the source code and data publicly available upon acceptance. 

**Abstract (ZH)**: 大型多模态模型（LMMs）在英语中表现出了强大的能力，但在日语中的有效性仍然受到限制，主要原因是缺乏高质量的训练数据。当前的日语LMMs往往依赖于翻译自英语的数据集，这限制了它们捕捉日本特定文化知识的能力。为了解决这一问题，我们探索了利用日语PDF数据作为训练资源的潜力，这是一个尚未充分利用的领域。我们提出了一种全自动的工作流，利用预训练模型通过布局分析、OCR和视觉-语言配对来提取PDF中的图像-文本对，从而省去了手动注释的需要。此外，我们从提取的图像-文本对中构建了指令数据，以丰富训练数据。为了评估PDF数据的有效性，我们训练了日语LMMs，并在日语LMM基准测试上对其性能进行了评估。我们的结果显示，在Heron-Bench上的性能增幅从3.9%到13.8%不等。进一步的分析突出了PDF数据对不同因素的影响，如模型规模和语言模型，进一步证实了其作为日语LMMs的多模态资源的价值。我们计划在文章被接受后公开源代码和数据。 

---
# Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning 

**Title (ZH)**: 逻辑-RL：基于规则的强化学习解锁LLM推理能力 

**Authors**: Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, Chong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.14768)  

**Abstract**: Inspired by the success of DeepSeek-R1, we explore the potential of rule-based reinforcement learning (RL) in large reasoning models. To analyze reasoning dynamics, we use synthetic logic puzzles as training data due to their controllable complexity and straightforward answer verification. We make some key technical contributions that lead to effective and stable RL training: a system prompt that emphasizes the thinking and answering process, a stringent format reward function that penalizes outputs for taking shortcuts, and a straightforward training recipe that achieves stable convergence. Our 7B model develops advanced reasoning skills-such as reflection, verification, and summarization-that are absent from the logic corpus. Remarkably, after training on just 5K logic problems, it demonstrates generalization abilities to the challenging math benchmarks AIME and AMC. 

**Abstract (ZH)**: 受DeepSeek-R1成功的启发，我们探索了基于规则的强化学习（RL）在大型推理模型中的潜在价值。为了分析推理动态，我们使用合成逻辑谜题作为训练数据，因为这些数据具有可控的复杂性和直接的答案验证性。我们在以下几个关键技术贡献方面取得了进展，这些贡献使得RL训练既有效又稳定：一个强调思考和回答过程的系统提示，一个严格的格式奖励函数，该函数对采取捷径的行为进行惩罚，以及一个简洁的训练配方，实现了稳定的收敛。我们的7B模型发展出了先进的推理技能，如反思、验证和总结，这些技能在逻辑语料库中是不存在的。值得注意的是，在仅仅训练了5000个逻辑问题之后，该模型展示了对具有挑战性的数学基准测试AIME和AMC的泛化能力。 

---
# Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis 

**Title (ZH)**: 树状辩论：多角色辩论树促进科学比较分析的批判性思维 

**Authors**: Priyanka Kargupta, Ishika Agarwal, Tal August, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.14767)  

**Abstract**: With the exponential growth of research facilitated by modern technology and improved accessibility, scientific discoveries have become increasingly fragmented within and across fields. This makes it challenging to assess the significance, novelty, incremental findings, and equivalent ideas between related works, particularly those from different research communities. Large language models (LLMs) have recently demonstrated strong quantitative and qualitative reasoning abilities, and multi-agent LLM debates have shown promise in handling complex reasoning tasks by exploring diverse perspectives and reasoning paths. Inspired by this, we introduce Tree-of-Debate (ToD), a framework which converts scientific papers into LLM personas that debate their respective novelties. To emphasize structured, critical reasoning rather than focusing solely on outcomes, ToD dynamically constructs a debate tree, enabling fine-grained analysis of independent novelty arguments within scholarly articles. Through experiments on scientific literature across various domains, evaluated by expert researchers, we demonstrate that ToD generates informative arguments, effectively contrasts papers, and supports researchers in their literature review. 

**Abstract (ZH)**: 随着现代技术的发展和获取途径的改进，科学研究在各领域内乃至跨领域内的成果呈指数级增长。这使得评估相关成果之间的意义、新颖性、增量发现和等效观点变得越来越具有挑战性，尤其是来自不同研究社区的作品之间的评估。大型语言模型（LLMs）近期展示了强大的定量和定性推理能力，而多智能体LLM辩论则展示了在处理复杂推理任务方面的潜力，通过探索多样化的视角和推理路径来应对这些挑战。受此启发，我们引入了辩论树（Tree-of-Debate, ToD）框架，该框架将科学论文转化为能够辩论其各自新颖性的LLM角色。ToD通过促进对独立新颖性论点的精细分析，而不仅仅是关注结果，动态构建辩论树，从而强调结构化的批判性推理。通过跨多个学科的科学文献实验，并由专家研究人员评估，我们证明了ToD能够生成具有信息性的论点，有效对比论文，并支持研究人员进行文献综述。 

---
# Step-by-Step Fact Verification System for Medical Claims with Explainable Reasoning 

**Title (ZH)**: 带有可解释推理的逐步医疗声明事实验证系统 

**Authors**: Juraj Vladika, Ivana Hacajová, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2502.14765)  

**Abstract**: Fact verification (FV) aims to assess the veracity of a claim based on relevant evidence. The traditional approach for automated FV includes a three-part pipeline relying on short evidence snippets and encoder-only inference models. More recent approaches leverage the multi-turn nature of LLMs to address FV as a step-by-step problem where questions inquiring additional context are generated and answered until there is enough information to make a decision. This iterative method makes the verification process rational and explainable. While these methods have been tested for encyclopedic claims, exploration on domain-specific and realistic claims is missing. In this work, we apply an iterative FV system on three medical fact-checking datasets and evaluate it with multiple settings, including different LLMs, external web search, and structured reasoning using logic predicates. We demonstrate improvements in the final performance over traditional approaches and the high potential of step-by-step FV systems for domain-specific claims. 

**Abstract (ZH)**: 事实验证（Fact Verification, FV）的目标是基于相关证据评估某一断言的真实性。传统的自动化FV方法依赖于简短的证据片段和仅编码的推理模型，形成了一个三阶段的工作流程。近年来，方法逐渐转向利用大型语言模型（LLM）的多轮对话特性，将FV问题视为逐步解决问题，通过生成并回答询问额外上下文的问题，直到有足够的信息作出判断。这种迭代方法使得验证过程既合理又具有可解释性。尽管这些方法在百科类断言上得到了测试，但对特定领域和真实场景下的断言探索却相对缺失。本文中，我们在三个医疗事实核查数据集上应用了一种迭代的FV系统，并在不同的大型语言模型、外部网络搜索和结构化逻辑推理等多种设置下进行了评估。我们展示了迭代FV系统相较于传统方法在最终性能上的改进，并指出了逐步FV系统在特定领域断言上的高潜力。 

---
# On the Influence of Context Size and Model Choice in Retrieval-Augmented Generation Systems 

**Title (ZH)**: 关于上下文大小和模型选择对检索增强生成系统影响的研究 

**Authors**: Juraj Vladika, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2502.14759)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as an approach to augment large language models (LLMs) by reducing their reliance on static knowledge and improving answer factuality. RAG retrieves relevant context snippets and generates an answer based on them. Despite its increasing industrial adoption, systematic exploration of RAG components is lacking, particularly regarding the ideal size of provided context, and the choice of base LLM and retrieval method. To help guide development of robust RAG systems, we evaluate various context sizes, BM25 and semantic search as retrievers, and eight base LLMs. Moving away from the usual RAG evaluation with short answers, we explore the more challenging long-form question answering in two domains, where a good answer has to utilize the entire context. Our findings indicate that final QA performance improves steadily with up to 15 snippets but stagnates or declines beyond that. Finally, we show that different general-purpose LLMs excel in the biomedical domain than the encyclopedic one, and that open-domain evidence retrieval in large corpora is challenging. 

**Abstract (ZH)**: 检索增强生成（RAG）作为一种通过减少对静态知识的依赖并提高答案的准确性来增强大规模语言模型（LLMs）的方法而崭露头角。RAG会检索相关上下文片段，并基于这些片段生成答案。尽管RAG在工业界的应用越来越广泛，但对其组件的系统性探索仍然不足，尤其是在提供上下文的理想大小以及基础LLM和检索方法的选择方面。为了指导稳健的RAG系统的开发，我们评估了不同大小的上下文、BM25和语义搜索作为检索方法，以及八种基础LLM。我们从传统的使用短答案评估RAG的方法转向探索更具挑战性的长形式问题回答，尤其是在两个领域中，良好的答案需要充分利用整个上下文。我们的研究结果显示，最终的问答性能随着最多15个片段的增加而逐步提高，但在超过这个数量后则停滞不前或下降。最后，我们表明，不同的通用语言模型在医学领域的表现优于百科全书领域，而在大型语料库中进行开放领域证据检索具有挑战性。 

---
# MedVAE: Efficient Automated Interpretation of Medical Images with Large-Scale Generalizable Autoencoders 

**Title (ZH)**: MedVAE：高效的大型通用自动编码器驱动的医学图像自动化解释方法 

**Authors**: Maya Varma, Ashwin Kumar, Rogier van der Sluijs, Sophie Ostmeier, Louis Blankemeier, Pierre Chambon, Christian Bluethgen, Jip Prince, Curtis Langlotz, Akshay Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2502.14753)  

**Abstract**: Medical images are acquired at high resolutions with large fields of view in order to capture fine-grained features necessary for clinical decision-making. Consequently, training deep learning models on medical images can incur large computational costs. In this work, we address the challenge of downsizing medical images in order to improve downstream computational efficiency while preserving clinically-relevant features. We introduce MedVAE, a family of six large-scale 2D and 3D autoencoders capable of encoding medical images as downsized latent representations and decoding latent representations back to high-resolution images. We train MedVAE autoencoders using a novel two-stage training approach with 1,052,730 medical images. Across diverse tasks obtained from 20 medical image datasets, we demonstrate that (1) utilizing MedVAE latent representations in place of high-resolution images when training downstream models can lead to efficiency benefits (up to 70x improvement in throughput) while simultaneously preserving clinically-relevant features and (2) MedVAE can decode latent representations back to high-resolution images with high fidelity. Our work demonstrates that large-scale, generalizable autoencoders can help address critical efficiency challenges in the medical domain. Our code is available at this https URL. 

**Abstract (ZH)**: 医疗图像以高分辨率和大视场获取，以捕捉临床决策所需的细粒度特征。因此，在医疗图像上训练深度学习模型可能会产生巨大的计算成本。本文旨在解决缩小医疗图像以提高下游计算效率的同时，保留临床相关特征的挑战。我们介绍了MedVAE，这是一种由六个大规模二维和三维自编码器组成的家族，能够将医学图像编码为缩小后的潜空间表示，并从潜空间表示解码回高分辨率图像。我们使用一种新颖的两阶段训练方法对MedVAE自编码器进行了训练，使用了1,052,730张医疗图像。在来自20个医学图像数据集的多种任务中，我们展示了以下结果：(1) 当在训练下游模型时使用MedVAE潜空间表示替代高分辨率图像时，可以实现效率上的好处（最多可提高70倍的吞吐量），同时保留临床相关特征；(2) MedVAE能够以高保真度将潜空间表示解码回高分辨率图像。我们的研究证明，大规模、可泛化的自编码器能够帮助解决医学领域中的关键效率挑战。我们的代码可以在以下链接获取：this https URL。 

---
# Multi-Agent Coordination across Diverse Applications: A Survey 

**Title (ZH)**: 跨领域多智能体协同综述 

**Authors**: Lijun Sun, Yijun Yang, Qiqi Duan, Yuhui Shi, Chao Lyu, Yu-Cheng Chang, Chin-Teng Lin, Yang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2502.14743)  

**Abstract**: Multi-agent coordination studies the underlying mechanism enabling the trending spread of diverse multi-agent systems (MAS) and has received increasing attention, driven by the expansion of emerging applications and rapid AI advances. This survey outlines the current state of coordination research across applications through a unified understanding that answers four fundamental coordination questions: (1) what is coordination; (2) why coordination; (3) who to coordinate with; and (4) how to coordinate. Our purpose is to explore existing ideas and expertise in coordination and their connections across diverse applications, while identifying and highlighting emerging and promising research directions. First, general coordination problems that are essential to varied applications are identified and analyzed. Second, a number of MAS applications are surveyed, ranging from widely studied domains, e.g., search and rescue, warehouse automation and logistics, and transportation systems, to emerging fields including humanoid and anthropomorphic robots, satellite systems, and large language models (LLMs). Finally, open challenges about the scalability, heterogeneity, and learning mechanisms of MAS are analyzed and discussed. In particular, we identify the hybridization of hierarchical and decentralized coordination, human-MAS coordination, and LLM-based MAS as promising future directions. 

**Abstract (ZH)**: 多智能体协调研究探讨了使多种多智能体系统（MAS）趋势性传播的潜在机制，并因其新兴应用的扩展和快速的人工智能进步而越来越受到关注。本综述通过统一的理解概述了跨不同应用领域的协调研究现状，并回答了四个基本的协调问题：（1）协调是什么；（2）为什么需要协调；（3）需要与谁协调；以及（4）如何协调。我们的目的是探索协调领域已有的理念和专业知识及其在不同应用领域的联系，并识别和强调新兴及有前景的研究方向。首先，我们识别并分析了对于各种应用来说都至关重要的基础协调问题。其次，综述了多种MAS应用，包括广泛研究的领域（如搜救、仓库自动化与物流、交通系统），以及新兴领域（如类人和拟人机器人、卫星系统和大规模语言模型（LLM））。最后，分析和讨论了MAS的可扩展性、异构性和学习机制方面的开放挑战。特别是，我们确定了层次结构与去中心化协调的结合、人类与MAS的协调以及基于LLM的MAS作为有前景的未来发展方向。 

---
# YOLOv12: A Breakdown of the Key Architectural Features 

**Title (ZH)**: YOL OV12：关键架构特征解析 

**Authors**: Mujadded Al Rabbani Alif, Muhammad Hussain  

**Link**: [PDF](https://arxiv.org/pdf/2502.14740)  

**Abstract**: This paper presents an architectural analysis of YOLOv12, a significant advancement in single-stage, real-time object detection building upon the strengths of its predecessors while introducing key improvements. The model incorporates an optimised backbone (R-ELAN), 7x7 separable convolutions, and FlashAttention-driven area-based attention, improving feature extraction, enhanced efficiency, and robust detections. With multiple model variants, similar to its predecessors, YOLOv12 offers scalable solutions for both latency-sensitive and high-accuracy applications. Experimental results manifest consistent gains in mean average precision (mAP) and inference speed, making YOLOv12 a compelling choice for applications in autonomous systems, security, and real-time analytics. By achieving an optimal balance between computational efficiency and performance, YOLOv12 sets a new benchmark for real-time computer vision, facilitating deployment across diverse hardware platforms, from edge devices to high-performance clusters. 

**Abstract (ZH)**: 本文对YOLOv12进行了架构分析，这是一种在单阶段、实时目标检测领域的重要进步，继承了前代模型的优点并引入了关键改进。该模型包含优化的主干网络（R-ELAN）、7x7 分离卷积以及基于区域的注意力机制（由FlashAttention驱动），从而提高了特征提取能力、提升了效率并增强了检测的鲁棒性。类似其前代模型，YOLOv12 提供了可扩展的解决方案，适用于对延迟敏感和高精度应用场景。实验结果表明，YOLOv12 在平均精度（mAP）和推断速度方面持续表现出提高，使其成为自动驾驶系统、安全和实时分析等应用中颇具吸引力的选择。通过在计算效率和性能之间找到最佳平衡，YOLOv12 为实时计算机视觉设立了新的标准，促进了其在从边缘设备到高性能集群等各种硬件平台的部署。 

---
# EAGER-LLM: Enhancing Large Language Models as Recommenders through Exogenous Behavior-Semantic Integration 

**Title (ZH)**: EAGER-LLM：通过外部行为语义集成增强大型语言模型作为推荐器的能力 

**Authors**: Minjie Hong, Yan Xia, Zehan Wang, Jieming Zhu, Ye Wang, Sihang Cai, Xiaoda Yang, Quanyu Dai, Zhenhua Dong, Zhimeng Zhang, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14735)  

**Abstract**: Large language models (LLMs) are increasingly leveraged as foundational backbones in the development of advanced recommender systems, offering enhanced capabilities through their extensive knowledge and reasoning. Existing llm-based recommender systems (RSs) often face challenges due to the significant differences between the linguistic semantics of pre-trained LLMs and the collaborative semantics essential for RSs. These systems use pre-trained linguistic semantics but learn collaborative semantics from scratch via the llm-Backbone. However, LLMs are not designed for recommendations, leading to inefficient collaborative learning, weak result correlations, and poor integration of traditional RS features. To address these challenges, we propose EAGER-LLM, a decoder-only llm-based generative recommendation framework that integrates endogenous and exogenous behavioral and semantic information in a non-intrusive manner. Specifically, we propose 1)dual-source knowledge-rich item indices that integrates indexing sequences for exogenous signals, enabling efficient link-wide processing; 2)non-invasive multiscale alignment reconstruction tasks guide the model toward a deeper understanding of both collaborative and semantic signals; 3)an annealing adapter designed to finely balance the model's recommendation performance with its comprehension capabilities. We demonstrate EAGER-LLM's effectiveness through rigorous testing on three public benchmarks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在先进推荐系统（RSs）的发展中被越来越多地用作基础骨干，通过其广泛的知识和推理能力提供了增强的功能。现有的基于LLM的推荐系统（RSs）通常面临着预训练LLM的语言语义与RS所需的协作语义之间的显著差异所带来的挑战。这些系统利用预训练的LLM语义，但通过LLM骨干从零开始学习协作语义。然而，这导致了低效的协作学习、结果相关性弱和传统RS特征的不良整合。为了解决这些挑战，我们提出了一种名为EAGER-LLM的自回归LLM基生成推荐框架，该框架以非侵入性的方式整合内生和外生的行为和语义信息。具体而言，我们提出了以下创新：

1. 双源知识丰富项索引，综合了外生信号的索引序列，以实现高效的整体处理；
2. 非侵入性多尺度对齐重构任务，引导模型深入理解协作和语义信号；
3. 设计了一个退火适配器，以精确平衡模型推荐性能与其理解能力。

我们通过在三个公开基准上进行严格的测试，证明了EAGER-LLM的有效性。 

---
# WavRAG: Audio-Integrated Retrieval Augmented Generation for Spoken Dialogue Models 

**Title (ZH)**: WavRAG：结合音频的检索增强生成模型用于口语对话系统 

**Authors**: Yifu Chen, Shengpeng Ji, Haoxiao Wang, Ziqing Wang, Siyu Chen, Jinzheng He, Jin Xu, Zhou Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14727)  

**Abstract**: Retrieval Augmented Generation (RAG) has gained widespread adoption owing to its capacity to empower large language models (LLMs) to integrate external knowledge. However, existing RAG frameworks are primarily designed for text-based LLMs and rely on Automatic Speech Recognition to process speech input, which discards crucial audio information, risks transcription errors, and increases computational overhead. Therefore, we introduce WavRAG, the first retrieval augmented generation framework with native, end-to-end audio support. WavRAG offers two key features: 1) Bypassing ASR, WavRAG directly processes raw audio for both embedding and retrieval. 2) WavRAG integrates audio and text into a unified knowledge representation. Specifically, we propose the WavRetriever to facilitate the retrieval from a text-audio hybrid knowledge base, and further enhance the in-context capabilities of spoken dialogue models through the integration of chain-of-thought reasoning. In comparison to state-of-the-art ASR-Text RAG pipelines, WavRAG achieves comparable retrieval performance while delivering a 10x acceleration. Furthermore, WavRAG's unique text-audio hybrid retrieval capability extends the boundaries of RAG to the audio modality. 

**Abstract (ZH)**: 检索增强生成（RAG）由于其增强大型语言模型（LLMs）整合外部知识的能力而被广泛应用。然而，现有的RAG框架主要针对基于文本的LLMs，并依赖自动语音识别（ASR）处理语音输入，这会丢弃重要的音频信息、增加转录错误的风险以及增加计算量。因此，我们引入了WavRAG，这是首个提供原生端到端语音支持的检索增强生成框架。WavRAG具备两项关键功能：1）绕过ASR，WavRAG直接处理原始音频进行嵌入和检索；2）WavRAG将音频和文本统一到一个知识表示中。具体而言，我们提出了WavRetriever以从文本-音频混合知识库中进行检索，并通过将链式推理集成到其中，进一步增强了口语对话模型的语境能力。与最新的ASR-Text RAG流水线相比，WavRAG实现了相当的检索性能，同时加速了10倍。此外，WavRAG独特的声音-文本混合检索能力将RAG的应用范围扩展到了音频模态。 

---
# Ranking Joint Policies in Dynamic Games using Evolutionary Dynamics 

**Title (ZH)**: 使用演化动力学排名动态博弈中的联合策略 

**Authors**: Natalia Koliou, George Vouros  

**Link**: [PDF](https://arxiv.org/pdf/2502.14724)  

**Abstract**: Game-theoretic solution concepts, such as the Nash equilibrium, have been key to finding stable joint actions in multi-player games. However, it has been shown that the dynamics of agents' interactions, even in simple two-player games with few strategies, are incapable of reaching Nash equilibria, exhibiting complex and unpredictable behavior. Instead, evolutionary approaches can describe the long-term persistence of strategies and filter out transient ones, accounting for the long-term dynamics of agents' interactions. Our goal is to identify agents' joint strategies that result in stable behavior, being resistant to changes, while also accounting for agents' payoffs, in dynamic games. Towards this goal, and building on previous results, this paper proposes transforming dynamic games into their empirical forms by considering agents' strategies instead of agents' actions, and applying the evolutionary methodology $\alpha$-Rank to evaluate and rank strategy profiles according to their long-term dynamics. This methodology not only allows us to identify joint strategies that are strong through agents' long-term interactions, but also provides a descriptive, transparent framework regarding the high ranking of these strategies. Experiments report on agents that aim to collaboratively solve a stochastic version of the graph coloring problem. We consider different styles of play as strategies to define the empirical game, and train policies realizing these strategies, using the DQN algorithm. Then we run simulations to generate the payoff matrix required by $\alpha$-Rank to rank joint strategies. 

**Abstract (ZH)**: 博弈论中的纳什 equilibrium 等解决方案概念在多玩家博弈中寻找稳定共同行动方面发挥了关键作用。然而，研究表明，即使在简单的小型两玩家博弈中，即使只有少数策略，玩家互动的动力学也无法达到纳什均衡，而是表现出了复杂的和不可预测的行为。相反，进化方法可以描述策略在长期中的持续性并筛选出短暂的策略，从而考虑玩家互动的长期动力学。我们的目标是在动态博弈中识别出能够产生稳定行为且对环境变化具有抗性的共同策略，同时也考虑玩家的支付情况。为了实现这一目标，并在此前结果的基础上，本文提出将动态博弈转化为其经验形式——即关注玩家的策略而不是玩家的操作，并应用进化方法 $\alpha$-Rank 评估和排名策略配置，以反映其长期动力学。这种方法不仅使我们能够识别通过玩家长期互动表现出强健性的共同策略，而且还提供了一种有关这些策略排名的描述性和透明框架。实验研究了玩家旨在合作解决图着色问题的随机版本。我们采用不同的玩法风格作为策略来定义经验博弈，并使用 DQN 算法训练实现这些策略的策略。然后，我们运行模拟生成 $\alpha$-Rank 用于排名共同策略所需的支付矩阵。 

---
# Human Misperception of Generative-AI Alignment: A Laboratory Experiment 

**Title (ZH)**: 人类对生成型AI对齐的误感知：一个实验室实验 

**Authors**: Kevin He, Ran Shorrer, Mengjia Xia  

**Link**: [PDF](https://arxiv.org/pdf/2502.14708)  

**Abstract**: We conduct an incentivized laboratory experiment to study people's perception of generative artificial intelligence (GenAI) alignment in the context of economic decision-making. Using a panel of economic problems spanning the domains of risk, time preference, social preference, and strategic interactions, we ask human subjects to make choices for themselves and to predict the choices made by GenAI on behalf of a human user. We find that people overestimate the degree of alignment between GenAI's choices and human choices. In every problem, human subjects' average prediction about GenAI's choice is substantially closer to the average human-subject choice than it is to the GenAI choice. At the individual level, different subjects' predictions about GenAI's choice in a given problem are highly correlated with their own choices in the same problem. We explore the implications of people overestimating GenAI alignment in a simple theoretical model. 

**Abstract (ZH)**: 我们开展了一项激励性实验室实验，以研究人们在经济决策背景下对生成型人工智能（GenAI）一致性感知的情况。利用涵盖风险、时间偏好、社会偏好和战略互动等领域的经济问题面板，我们要求人类被试为自己做出选择，并预测GenAI为人类用户所做的选择。我们发现，人们高估了GenAI选择与人类选择之间的一致程度。在每个问题中，人类被试对GenAI选择的平均预测与人类被试自身的平均选择更为接近，而与GenAI的选择则更不一致。在个体层面，不同被试对给定问题中GenAI选择的预测与他们自身在相同问题中的选择高度相关。我们在一个简单的理论模型中探讨了人们高估GenAI一致性的影响。 

---
# Not All Data are Good Labels: On the Self-supervised Labeling for Time Series Forecasting 

**Title (ZH)**: 不是所有数据都是良好的标签：关于时间序列预测的自监督标记研究 

**Authors**: Yuxuan Yang, Dalin Zhang, Yuxuan Liang, Hua Lu, Huan Li, Gang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.14704)  

**Abstract**: Time Series Forecasting (TSF) is a crucial task in various domains, yet existing TSF models rely heavily on high-quality data and insufficiently exploit all available data. This paper explores a novel self-supervised approach to re-label time series datasets by inherently constructing candidate datasets. During the optimization of a simple reconstruction network, intermediates are used as pseudo labels in a self-supervised paradigm, improving generalization for any predictor. We introduce the Self-Correction with Adaptive Mask (SCAM), which discards overfitted components and selectively replaces them with pseudo labels generated from reconstructions. Additionally, we incorporate Spectral Norm Regularization (SNR) to further suppress overfitting from a loss landscape perspective. Our experiments on eleven real-world datasets demonstrate that SCAM consistently improves the performance of various backbone models. This work offers a new perspective on constructing datasets and enhancing the generalization of TSF models through self-supervised learning. 

**Abstract (ZH)**: 时间序列预测（TSF）是各个领域中的一个关键任务，然而现有的TSF模型过于依赖高质量的数据，未能充分利用所有可用的数据。本文探索了一种新颖的自监督方法，通过内生性地构建候选数据集来重新标记时间序列数据集。在简单重构网络的优化过程中，中间结果被用作自监督范式中的伪标签，从而提高任何预测器的一般性。我们引入了自校正与自适应掩码（SCAM），该方法抛弃了过拟合的组件，并选择性地用来自重构的伪标签替换它们。此外，我们还引入了谱范数正则化（SNR），从损失景观的角度进一步抑制过拟合。我们在 eleven 个真实世界的数据集上的实验表明，SCAM 能够一致地提高各种骨干模型的性能。这项工作为通过自监督学习构建数据集和增强TSF模型的一般性提供了一个新的视角。 

---
# General Uncertainty Estimation with Delta Variances 

**Title (ZH)**: Delta 方差下的通用不确定性估计 

**Authors**: Simon Schmitt, John Shawe-Taylor, Hado van Hasselt  

**Link**: [PDF](https://arxiv.org/pdf/2502.14698)  

**Abstract**: Decision makers may suffer from uncertainty induced by limited data. This may be mitigated by accounting for epistemic uncertainty, which is however challenging to estimate efficiently for large neural networks. To this extent we investigate Delta Variances, a family of algorithms for epistemic uncertainty quantification, that is computationally efficient and convenient to implement. It can be applied to neural networks and more general functions composed of neural networks. As an example we consider a weather simulator with a neural-network-based step function inside -- here Delta Variances empirically obtain competitive results at the cost of a single gradient computation. The approach is convenient as it requires no changes to the neural network architecture or training procedure. We discuss multiple ways to derive Delta Variances theoretically noting that special cases recover popular techniques and present a unified perspective on multiple related methods. Finally we observe that this general perspective gives rise to a natural extension and empirically show its benefit. 

**Abstract (ZH)**: 决策者可能会受到有限数据引起不确定性的影响。为此，可以通过考虑认识不确定性来减轻这种影响，然而，对于大型神经网络来说，有效地估计认识不确定性具有挑战性。在这种背景下，我们探讨了Delta.Variances这一认识不确定性量化的一系列算法，这些算法计算效率高且易于实现。它们可以应用于包含神经网络的神经网络和更广泛的功能。作为示例，我们考虑了一个包含基于神经网络的阶梯函数的天气模拟器——在这里，Delta.Variances仅通过一次梯度计算就能获得有竞争力的结果。该方法很方便，因为它不需要改变神经网络结构或训练过程。我们从多个角度理论地推导了Delta.Variances，指出其特殊情况可以恢复流行的技术，并提供了一种多种相关方法的统一视角。最后，我们观察到这种一般视角产生了自然的拓展，并通过实验展示了其优势。 

---
# seqKAN: Sequence processing with Kolmogorov-Arnold Networks 

**Title (ZH)**: seqKAN：基于柯尔莫哥洛夫-阿诺尔德网络的时间序列处理 

**Authors**: Tatiana Boura, Stasinos Konstantopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.14681)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) have been recently proposed as a machine learning framework that is more interpretable and controllable than the multi-layer perceptron. Various network architectures have been proposed within the KAN framework targeting different tasks and application domains, including sequence processing.
This paper proposes seqKAN, a new KAN architecture for sequence processing. Although multiple sequence processing KAN architectures have already been proposed, we argue that seqKAN is more faithful to the core concept of the KAN framework. Furthermore, we empirically demonstrate that it achieves better results.
The empirical evaluation is performed on generated data from a complex physics problem on an interpolation and an extrapolation task. Using this dataset we compared seqKAN against a prior KAN network for timeseries prediction, recurrent deep networks, and symbolic regression. seqKAN substantially outperforms all architectures, particularly on the extrapolation dataset, while also being the most transparent. 

**Abstract (ZH)**: 基于柯尔莫哥洛夫-阿诺尔德网络（KANs）最近被提出，作为一种比多层感知机更具可解释性和可控性的机器学习框架。KAN框架内已经提出多种网络架构，以适用于不同的任务和应用领域，其中包括序列处理任务。
本文提出了一种新的KAN架构seqKAN，专门用于序列处理任务。尽管已经提出了多种用于序列处理的KAN架构，但我们认为seqKAN更加忠实于KAN框架的核心概念。此外，我们通过实验证明，seqKAN取得了更好的性能。
实验评估是在一个复杂物理问题产生的数据集上进行的，该数据集用于插值和外推任务。我们使用这个数据集将seqKAN与以前的KAN时间序列预测网络、循环深度网络以及符号回归进行比较。实验结果表明，seqKAN在所有架构中表现最佳，尤其是在外推数据集上的表现尤为突出，同时seqKAN也是最透明的网络之一。 

---
# Data-Constrained Synthesis of Training Data for De-Identification 

**Title (ZH)**: 基于数据约束的去标识化训练数据合成 

**Authors**: Thomas Vakili, Aron Henriksson, Hercules Dalianis  

**Link**: [PDF](https://arxiv.org/pdf/2502.14677)  

**Abstract**: Many sensitive domains -- such as the clinical domain -- lack widely available datasets due to privacy risks. The increasing generative capabilities of large language models (LLMs) have made synthetic datasets a viable path forward. In this study, we domain-adapt LLMs to the clinical domain and generate synthetic clinical texts that are machine-annotated with tags for personally identifiable information using capable encoder-based NER models. The synthetic corpora are then used to train synthetic NER models. The results show that training NER models using synthetic corpora incurs only a small drop in predictive performance. The limits of this process are investigated in a systematic ablation study -- using both Swedish and Spanish data. Our analysis shows that smaller datasets can be sufficient for domain-adapting LLMs for data synthesis. Instead, the effectiveness of this process is almost entirely contingent on the performance of the machine-annotating NER models trained using the original data. 

**Abstract (ZH)**: 许多敏感领域，如临床领域，由于隐私风险而缺乏广泛可用的数据集。随着大型语言模型（LLMs）生成能力的提升，合成数据集已成为可行的选择。本研究中，我们将LLMs适应临床领域，并利用强大的编码器基模型生成带关于个人可识别信息标签的合成临床文本。合成语料库随后用于训练合成命名实体识别（NER）模型。研究结果表明，使用合成语料库训练NER模型仅会带来轻微的预测性能下降。我们通过系统性消融研究，利用瑞典语和西班牙语数据对这一过程的局限性进行探讨。分析显示，较小的数据集对于适应LLMs以生成数据来说可能是足够的；而这一过程的有效性几乎完全取决于利用原始数据训练的机器标注NER模型的性能。 

---
# BP-SGCN: Behavioral Pseudo-Label Informed Sparse Graph Convolution Network for Pedestrian and Heterogeneous Trajectory Prediction 

**Title (ZH)**: BP-SGCN: 行为伪标签指导的稀疏图卷积网络用于行人和其他异质轨迹预测 

**Authors**: Ruochen Li, Stamos Katsigiannis, Tae-Kyun Kim, Hubert P. H. Shum  

**Link**: [PDF](https://arxiv.org/pdf/2502.14676)  

**Abstract**: Trajectory prediction allows better decision-making in applications of autonomous vehicles or surveillance by predicting the short-term future movement of traffic agents. It is classified into pedestrian or heterogeneous trajectory prediction. The former exploits the relatively consistent behavior of pedestrians, but is limited in real-world scenarios with heterogeneous traffic agents such as cyclists and vehicles. The latter typically relies on extra class label information to distinguish the heterogeneous agents, but such labels are costly to annotate and cannot be generalized to represent different behaviors within the same class of agents. In this work, we introduce the behavioral pseudo-labels that effectively capture the behavior distributions of pedestrians and heterogeneous agents solely based on their motion features, significantly improving the accuracy of trajectory prediction. To implement the framework, we propose the Behavioral Pseudo-Label Informed Sparse Graph Convolution Network (BP-SGCN) that learns pseudo-labels and informs to a trajectory predictor. For optimization, we propose a cascaded training scheme, in which we first learn the pseudo-labels in an unsupervised manner, and then perform end-to-end fine-tuning on the labels in the direction of increasing the trajectory prediction accuracy. Experiments show that our pseudo-labels effectively model different behavior clusters and improve trajectory prediction. Our proposed BP-SGCN outperforms existing methods using both pedestrian (ETH/UCY, pedestrian-only SDD) and heterogeneous agent datasets (SDD, Argoverse 1). 

**Abstract (ZH)**: 轨迹预测能够通过预测交通参与者的短期未来运动，在自主车辆应用或监控中做出更好的决策。轨迹预测主要分为行人轨迹预测和异质轨迹预测。前者利用行人相对一致的行为，但在包含自行车、车辆等异质交通参与者的实际场景中受限。后者通常依赖额外的类别标签来区分不同的异质交通参与者，但这些标签标记成本高，难以泛化以代表同一类别参与者内的不同行为。在本文中，我们引入了一种行为伪标签，该标签仅根据交通参与者的运动特征有效捕捉行人的行为分布和异质参与者的不同行为模式，显著提高了轨迹预测的准确性。为了实现该框架，我们提出了行为伪标签指导稀疏图卷积网络（BP-SGCN），该模型学习伪标签并指导轨迹预测器。在优化方面，我们提出了一种级联训练方案，首先以无监督方式学习伪标签，然后通过提高轨迹预测准确性的方向进行端到端的微调。实验结果表明，我们的伪标签有效地建模了不同的行为簇，并提升了轨迹预测的准确性。我们提出的BP-SGCN在基于行人（ETH/UCY，行人唯一SDD）和异质交通参与者（SDD，Argoverse 1）的数据集上均优于现有方法。 

---
# Explanations of Deep Language Models Explain Language Representations in the Brain 

**Title (ZH)**: 深度语言模型的解释说明了大脑中的语言表示 

**Authors**: Maryam Rahimi, Yadollah Yaghoobzadeh, Mohammad Reza Daliri  

**Link**: [PDF](https://arxiv.org/pdf/2502.14671)  

**Abstract**: Recent advances in artificial intelligence have given rise to large language models (LLMs) that not only achieve human-like performance but also share computational principles with the brain's language processing mechanisms. While previous research has primarily focused on aligning LLMs' internal representations with neural activity, we introduce a novel approach that leverages explainable AI (XAI) methods to forge deeper connections between the two domains. Using attribution methods, we quantified how preceding words contribute to an LLM's next-word predictions and employed these explanations to predict fMRI recordings from participants listening to the same narratives. Our findings demonstrate that attribution methods robustly predict brain activity across the language network, surpassing traditional internal representations in early language areas. This alignment is hierarchical: early-layer explanations correspond to the initial stages of language processing in the brain, while later layers align with more advanced stages. Moreover, the layers more influential on LLM next-word prediction$\unicode{x2014}$those with higher attribution scores$\unicode{x2014}$exhibited stronger alignment with neural activity. This work establishes a bidirectional bridge between AI and neuroscience. First, we demonstrate that attribution methods offer a powerful lens for investigating the neural mechanisms of language comprehension, revealing how meaning emerges from preceding context. Second, we propose using brain alignment as a metric to evaluate the validity of attribution methods, providing a framework for assessing their biological plausibility. 

**Abstract (ZH)**: Recent人工智能的最新进展催生了大语言模型（LLMs），这些模型不仅实现了类人的性能，而且在计算原理上与大脑的语言处理机制共享相似之处。尽管以往的研究主要集中在使LLMs的内部表示与神经活动相一致，我们提出了一种新的方法，利用可解释的人工智能（XAI）技术，更深入地连接这两者。通过归因方法，量化了前面的词语对LLMs下一个词语预测的贡献，并利用这些解释预测了参与者在听同样故事时的fMRI记录。我们的研究表明，归因方法能够有效地预测语言网络中的脑活动，超过了传统早期语言区域的内部表示方法。这种对齐是分层的：早期层的解释对应于大脑中语言处理的初始阶段，而后期层则与更高级的阶段相对应。此外，对LLMs下一个词语预测更具影响力的层次——那些具有更高归因分数的层次——与神经活动的对齐也更加紧密。这项工作建立了人工智能与神经科学之间的双向桥梁。首先，我们证明了归因方法提供了一种强大的工具，可以研究语言理解的神经机制，揭示意义是如何从先前的上下文中涌现出来的。其次，我们提出使用脑活动对齐作为评估归因方法有效性的指标，为评估其生物学可行性提供了一个框架。 

---
# Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs 

**Title (ZH)**: 一次编辑，全局更新：LLM中跨语言知识同步的简单框架 

**Authors**: Yuchen Wu, Liang Ding, Li Shen, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14645)  

**Abstract**: Knowledge editing allows for efficient adaptation of large language models (LLMs) to new information or corrections without requiring full retraining. However, prior methods typically focus on either single-language editing or basic multilingual editing, failing to achieve true cross-linguistic knowledge synchronization. To address this, we present a simple and practical state-of-the-art (SOTA) recipe Cross-Lingual Knowledge Democracy Edit (X-KDE), designed to propagate knowledge from a dominant language to other languages effectively. Our X-KDE comprises two stages: (i) Cross-lingual Edition Instruction Tuning (XE-IT), which fine-tunes the model on a curated parallel dataset to modify in-scope knowledge while preserving unrelated information, and (ii) Target-language Preference Optimization (TL-PO), which applies advanced optimization techniques to ensure consistency across languages, fostering the transfer of updates. Additionally, we contribute a high-quality, cross-lingual dataset, specifically designed to enhance knowledge transfer across languages. Extensive experiments on the Bi-ZsRE and MzsRE benchmarks show that X-KDE significantly enhances cross-lingual performance, achieving an average improvement of +8.19%, while maintaining high accuracy in monolingual settings. 

**Abstract (ZH)**: 知识编辑使大型语言模型（LLMs）能够高效地适应新信息或纠正错误，而无需进行全面重新训练。然而，以往的方法通常仅专注于单语言编辑或基本的多语言编辑，未能实现真正的跨语言知识同步。为解决这一问题，我们提出了一种简单实用的最先进的（SOTA）方法——跨语言知识民主编辑（X-KDE），旨在有效传播主导语言的知识至其他语言。X-KDE包括两个阶段：（i）跨语言编辑指令微调（XE-IT），在精心策划的平行数据集上微调模型，修改相关知识同时保留无关信息；（ii）目标语言偏好优化（TL-PO），应用高级优化技术以确保语言一致性，促进更新的转移。此外，我们还贡献了一个高质量的跨语言数据集，专门设计用于增强跨语言知识的转移。在Bi-ZsRE和MzsRE基准上的广泛实验表明，X-KDE显著提升了跨语言性能，平均改进幅度为+8.19%，同时在单一语言设置中保持了高准确性。 

---
# ReQFlow: Rectified Quaternion Flow for Efficient and High-Quality Protein Backbone Generation 

**Title (ZH)**: ReQFlow：校正四元数流，用于高效高质蛋白质主链生成 

**Authors**: Angxiao Yue, Zichong Wang, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14637)  

**Abstract**: Protein backbone generation plays a central role in de novo protein design and is significant for many biological and medical applications. Although diffusion and flow-based generative models provide potential solutions to this challenging task, they often generate proteins with undesired designability and suffer computational inefficiency. In this study, we propose a novel rectified quaternion flow (ReQFlow) matching method for fast and high-quality protein backbone generation. In particular, our method generates a local translation and a 3D rotation from random noise for each residue in a protein chain, which represents each 3D rotation as a unit quaternion and constructs its flow by spherical linear interpolation (SLERP) in an exponential format. We train the model by quaternion flow (QFlow) matching with guaranteed numerical stability and rectify the QFlow model to accelerate its inference and improve the designability of generated protein backbones, leading to the proposed ReQFlow model. Experiments show that ReQFlow achieves state-of-the-art performance in protein backbone generation while requiring much fewer sampling steps and significantly less inference time (e.g., being 37x faster than RFDiffusion and 62x faster than Genie2 when generating a backbone of length 300), demonstrating its effectiveness and efficiency. The code is available at this https URL. 

**Abstract (ZH)**: 蛋白质主链生成在从头蛋白质设计中扮演着中心角色，并且对于许多生物学和医学应用都非常重要。尽管扩散和流式生成模型为这一具有挑战性的任务提供了潜在的解决方案，但它们经常生成具有不良设计特性的蛋白质，并且计算效率低下。在本研究中，我们提出了一种新型修正四元数流（ReQFlow）匹配方法，以实现快速且高质量的蛋白质主链生成。特别地，我们的方法从每条蛋白质链上的随机噪声生成局部平移和3D旋转，并将每种3D旋转表示为单位四元数，并通过指数格式中的球面线性插值（SLERP）来构建其流。我们通过保证数值稳定性的四元数流（QFlow）匹配训练模型，并纠正QFlow模型以加速其推断并提高生成的蛋白质主链的设计特性，从而提出了ReQFlow模型。实验结果表明，ReQFlow在蛋白质主链生成中取得了最先进的性能，同时需要更少的采样步骤，并且显着减少了推断时间（例如，在生成300个残基长度的主链时，比RFDiffusion快37倍，比Genie2快62倍），证明了其有效性和效率。代码可在以下链接获取：https://your-link.com 

---
# ATRI: Mitigating Multilingual Audio Text Retrieval Inconsistencies by Reducing Data Distribution Errors 

**Title (ZH)**: ATRI：通过减少数据分布误差来缓解多语言音频文本检索不一致性 

**Authors**: Yuguo Yin, Yuxin Xie, Wenyuan Yang, Dongchao Yang, Jinghan Ru, Xianwei Zhuang, Liming Liang, Yuexian Zou  

**Link**: [PDF](https://arxiv.org/pdf/2502.14627)  

**Abstract**: Multilingual audio-text retrieval (ML-ATR) is a challenging task that aims to retrieve audio clips or multilingual texts from databases. However, existing ML-ATR schemes suffer from inconsistencies for instance similarity matching across languages. We theoretically analyze the inconsistency in terms of both multilingual modal alignment direction error and weight error, and propose the theoretical weight error upper bound for quantifying the inconsistency. Based on the analysis of the weight error upper bound, we find that the inconsistency problem stems from the data distribution error caused by random sampling of languages. We propose a consistent ML-ATR scheme using 1-to-k contrastive learning and audio-English co-anchor contrastive learning, aiming to mitigate the negative impact of data distribution error on recall and consistency in ML-ATR. Experimental results on the translated AudioCaps and Clotho datasets show that our scheme achieves state-of-the-art performance on recall and consistency metrics for eight mainstream languages, including English. Our code will be available at this https URL. 

**Abstract (ZH)**: 多语言音频-文本检索（ML-ATR）是一个具有挑战性的任务，旨在从数据库中检索音频片段或多语言文本。然而，现有的ML-ATR方案在不同语言之间存在不一致性，例如相似性匹配的一致性问题。我们从多语言模态对齐方向错误和权重错误两个方面对这种不一致性进行了理论分析，并提出了量化这种不一致性的理论权重误差上限。基于权重误差上限的分析，我们发现不一致性问题源自由语言随机抽样引起的数据分布错误。我们提出了一种使用1-to-k对比学习和音频-英语共锚对比学习的一致性ML-ATR方案，旨在减轻数据分布错误对ML-ATR召回率和一致性产生的负面影响。在翻译后的AudioCaps和Clotho数据集上的实验证明，我们的方案在包括英语在内的八种主流语言上实现了最先进的召回率和一致性指标。我们的代码将在以下链接处提供：[此处链接]。 

---
# Exploring RWKV for Sentence Embeddings: Layer-wise Analysis and Baseline Comparison for Semantic Similarity 

**Title (ZH)**: 探索RWKV在句子嵌入中的应用：逐层分析及语义相似度基准比较 

**Authors**: Xinghan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2502.14620)  

**Abstract**: This paper investigates the efficacy of RWKV, a novel language model architecture known for its linear attention mechanism, for generating sentence embeddings in a zero-shot setting. I conduct a layer-wise analysis to evaluate the semantic similarity captured by embeddings from different hidden layers of a pre-trained RWKV model. The performance is assessed on the Microsoft Research Paraphrase Corpus (MRPC) dataset using Spearman correlation and compared against a GloVe-based baseline. My results indicate that while RWKV embeddings capture some semantic relatedness, they underperform compared to the GloVe baseline in terms of Spearman correlation. I also analyze the inference time and GPU memory usage, highlighting the computational trade-offs associated with RWKV embeddings. The findings suggest that while RWKV offers potential advantages in terms of linear scaling, its zero-shot sentence embedding quality for semantic similarity tasks requires further investigation and potential task-specific fine-tuning to match or exceed simpler baselines. 

**Abstract (ZH)**: 本文探究了RWKV这一具有线性注意力机制的新颖语言模型架构在零样本设置中生成句向量的有效性。通过对预训练RWKV模型不同隐藏层的嵌入进行逐层分析，评估其所捕获的语义相似性。实验使用Microsoft Research Paraphrase Corpus (MRPC) 数据集和Spearman相关系数进行性能评估，并将其与基于GloVe的基线进行比较。结果表明，尽管RWKV嵌入能够捕获一定程度的语义相关性，但在Spearman相关系数上，其表现不如GloVe基线。此外，本文还分析了推理时间以及GPU内存使用情况，指出了RWKV嵌入相关的计算权衡。研究结果表明，虽然RWKV在方面具有潜在优势，但在语义相似性任务中的零样本句向量质量仍需进一步研究，并可能需要特定任务的微调以达到或超过更简单的基线。

该研究指出，虽然RWKV在理论上具有线性扩展的优势，但在语义相似性任务的零样本句向量质量方面，其性能仍低于基于GloVe的基线系统。这一发现表明，为了提高RWKV在语义相似性任务中的表现，可能需要对其进行特定任务的微调，以充分发挥其潜在优势。 

---
# Reward Models Identify Consistency, Not Causality 

**Title (ZH)**: 奖励模型识别一致性，而非因果关系 

**Authors**: Yuhui Xu, Hanze Dong, Lei Wang, Caiming Xiong, Junnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.14619)  

**Abstract**: Reward models (RMs) play a crucial role in aligning large language models (LLMs) with human preferences and enhancing reasoning quality. Traditionally, RMs are trained to rank candidate outputs based on their correctness and coherence. However, in this work, we present several surprising findings that challenge common assumptions about RM behavior. Our analysis reveals that state-of-the-art reward models prioritize structural consistency over causal correctness. Specifically, removing the problem statement has minimal impact on reward scores, whereas altering numerical values or disrupting the reasoning flow significantly affects RM outputs. Furthermore, RMs exhibit a strong dependence on complete reasoning trajectories truncated or incomplete steps lead to significant variations in reward assignments, indicating that RMs primarily rely on learned reasoning patterns rather than explicit problem comprehension. These findings hold across multiple architectures, datasets, and tasks, leading to three key insights: (1) RMs primarily assess coherence rather than true reasoning quality; (2) The role of explicit problem comprehension in reward assignment is overstated; (3) Current RMs may be more effective at ranking responses than verifying logical validity. Our results suggest a fundamental limitation in existing reward modeling approaches, emphasizing the need for a shift toward causality-aware reward models that go beyond consistency-driven evaluation. 

**Abstract (ZH)**: 奖励模型（RMs）在使大型语言模型（LLMs）与人类偏好保持一致并提高推理质量方面起着关键作用。传统上，RMs 是通过训练来根据候选输出的正确性和连贯性进行排序的。然而，在本工作中，我们揭示了几项令人惊讶的发现，这些发现挑战了关于RMs行为的一些常见假设。我们的分析表明，最先进的奖励模型更倾向于结构一致性而非原因正确性。具体来说，移除问题陈述对奖励评分的影响微乎其微，而改变数值或中断推理流程则显著影响RMs的输出。此外，RMs对完整的推理轨迹表现出强烈的依赖性，断开或不完整的推理步骤会导致奖励分配的巨大差异，表明RMs主要依赖于学习到的推理模式，而不是明确的问题理解。这些发现贯穿多个架构、数据集和任务，产生了三个关键洞察：（1）RMs主要评估连贯性而非真正的推理质量；（2）显式问题理解在奖励分配中的作用被高估了；（3）当前的RMs可能在排序响应方面比验证逻辑有效性更有效。我们的结果表明现有奖励建模方法存在根本性的局限性，强调了转向因果关系感知的奖励模型的必要性，这些模型超越了基于一致性的评估方法。 

---
# A Theory for Conditional Generative Modeling on Multiple Data Sources 

**Title (ZH)**: 多数据源条件生成建模的理论框架 

**Authors**: Rongzhen Wang, Yan Zhang, Chenyu Zheng, Chongxuan Li, Guoqiang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14583)  

**Abstract**: The success of large generative models has driven a paradigm shift, leveraging massive multi-source data to enhance model capabilities. However, the interaction among these sources remains theoretically underexplored. This paper takes the first step toward a rigorous analysis of multi-source training in conditional generative modeling, where each condition represents a distinct data source. Specifically, we establish a general distribution estimation error bound in average total variation distance for conditional maximum likelihood estimation based on the bracketing number. Our result shows that when source distributions share certain similarities and the model is expressive enough, multi-source training guarantees a sharper bound than single-source training. We further instantiate the general theory on conditional Gaussian estimation and deep generative models including autoregressive and flexible energy-based models, by characterizing their bracketing numbers. The results highlight that the number of sources and similarity among source distributions improve the advantage of multi-source training. Simulations and real-world experiments validate our theory. Code is available at: \url{this https URL}. 

**Abstract (ZH)**: 大型生成模型的成功促进了范式的转变，通过利用大规模多源数据来增强模型能力。然而，这些数据源之间的交互在理论上尚未充分探索。本文迈出了严谨分析条件生成模型中多源训练的第一步，其中每个条件代表一个独特的数据源。具体而言，我们基于括号数建立了基于平均总变差距离的条件极大似然估计的一般分布估计误差界。我们的结果表明，在源分布具有某些相似性且模型足够表达力时，多源训练比单源训练能提供更紧的界。进一步地，我们通过刻画其括号数，将一般理论落实到条件高斯估计和深度生成模型（包括自回归和灵活的能量模型）中。结果表明，源的数量和源分布之间的相似性可以强化多源训练的优势。通过模拟和现实世界实验验证了我们的理论。代码可在以下链接获取：\url{this https URL}。 

---
# Factor Graph-based Interpretable Neural Networks 

**Title (ZH)**: 基于因子图的可解释神经网络 

**Authors**: Yicong Li, Kuanjiu Zhou, Shuo Yu, Qiang Zhang, Renqiang Luo, Xiaodong Li, Feng Xia  

**Link**: [PDF](https://arxiv.org/pdf/2502.14572)  

**Abstract**: Comprehensible neural network explanations are foundations for a better understanding of decisions, especially when the input data are infused with malicious perturbations. Existing solutions generally mitigate the impact of perturbations through adversarial training, yet they fail to generate comprehensible explanations under unknown perturbations. To address this challenge, we propose AGAIN, a fActor GrAph-based Interpretable neural Network, which is capable of generating comprehensible explanations under unknown perturbations. Instead of retraining like previous solutions, the proposed AGAIN directly integrates logical rules by which logical errors in explanations are identified and rectified during inference. Specifically, we construct the factor graph to express logical rules between explanations and categories. By treating logical rules as exogenous knowledge, AGAIN can identify incomprehensible explanations that violate real-world logic. Furthermore, we propose an interactive intervention switch strategy rectifying explanations based on the logical guidance from the factor graph without learning perturbations, which overcomes the inherent limitation of adversarial training-based methods in defending only against known perturbations. Additionally, we theoretically demonstrate the effectiveness of employing factor graph by proving that the comprehensibility of explanations is strongly correlated with factor graph. Extensive experiments are conducted on three datasets and experimental results illustrate the superior performance of AGAIN compared to state-of-the-art baselines. 

**Abstract (ZH)**: 可解释的神经网络是增强对决策理解的基础，特别是在输入数据中存在恶意扰动的情况下。现有解决方案通常通过对抗训练来减轻这些扰动的影响，但它们在面对未知扰动时无法生成可解释的解释。为了解决这一挑战，我们提出了一种名为AGAIN（基于因子图的可解释神经网络）的方案，该方案能在未知扰动下生成可解释的解释。与以往需要重新训练的方法不同，AIMAG 直接将逻辑规则整合进来，使解释中的逻辑错误能在推断过程中被识别并纠正。具体而言，我们构建了因子图来表达解释与类别之间的逻辑规则。通过将逻辑规则视为外生知识，AIMAG 能识别违反现实逻辑的不可解释的解释。此外，我们提出了一种交互式干预开关策略，该策略根据因子图提供的逻辑指导修正解释，从而克服了基于对抗训练方法固有的仅能防御已知扰动的局限性。此外，我们从理论上证明了使用因子图的有效性，证明解释的可解释性与因子图之间存在强相关性。在三个数据集上进行了广泛的实验，实验结果表明AIMAG 在与最先进的基线方法相比时表现更优。 

---
# Less is More: Improving LLM Alignment via Preference Data Selection 

**Title (ZH)**: 更少即是更多：通过偏好数据选择提升大语言模型一致性 

**Authors**: Xun Deng, Han Zhong, Rui Ai, Fuli Feng, Zheng Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2502.14560)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as a promising approach for aligning large language models with human preferences. While prior work mainly extends DPO from the aspect of the objective function, we instead improve DPO from the largely overlooked but critical aspect of data selection. Specifically, we address the issue of parameter shrinkage caused by noisy data by proposing a novel margin-maximization principle for dataset curation in DPO training. To accurately estimate margins for data selection, we propose a dual-margin guided approach that considers both external reward margins and implicit DPO reward margins. Extensive experiments demonstrate that our method reduces computational cost dramatically while improving performance. Remarkably, by using just 10\% of the Ultrafeedback dataset, our approach achieves 3\% to 8\% improvements across various Llama and Mistral series models on the AlpacaEval 2.0 benchmark. Furthermore, our approach seamlessly extends to iterative DPO, yielding a roughly 3\% improvement with 25\% online data, while further reducing training time. These results highlight the potential of data selection strategies for advancing preference optimization. 

**Abstract (ZH)**: 直接偏好优化（DPO）已经成为一种有前景的方法，用于使大规模语言模型与人类偏好对齐。尽管前期的工作主要通过目标函数扩展了DPO，我们则从被忽视但至关重要的数据选择方面改进了DPO。具体而言，我们通过提出一种新颖的边际最大化原则来解决由嘈杂数据引起的参数收缩问题，以改进DPO训练中的数据集管理。为了准确估计用于数据选择的边际，我们提出了一种双边际引导方法，该方法同时考虑外部奖励边际和隐含的DPO奖励边际。广泛的经验研究证明，我们的方法在大幅降低计算成本的同时提高了性能。令人印象深刻的是，仅使用Ultrafeedback数据集的10%，我们的方法在AlpacaEval 2.0基准测试中的Llama和Mistral系列模型上实现了3%到8%的性能提升。此外，我们的方法无缝扩展到了迭代DPO，通过使用25%的在线数据，我们不仅提升了约3%的性能，还进一步降低了训练时间。这些结果突显了数据选择策略在偏好优化中潜在的作用。 

---
# FUIA: Model Inversion Attack against Federated Unlearning 

**Title (ZH)**: FUIA：针对联邦未学习模型的模型反转攻击

解释：
- FUIA: Model Inversion Attack against Federated Unlearning
- “FUIA”保持不变，是该研究的简称。
- “Model Inversion Attack”翻译为“模型反转攻击”。
- “against Federated Unlearning”翻译为“针对联邦未学习模型”。

这个标题翻译成中文后，既保留了原文的专业术语，又符合中文的表达习惯。 

**Authors**: Lei Zhou, Youwen Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14558)  

**Abstract**: With the introduction of regulations related to the ``right to be forgotten", federated learning (FL) is facing new privacy compliance challenges. To address these challenges, researchers have proposed federated unlearning (FU). However, existing FU research has primarily focused on improving the efficiency of unlearning, with less attention paid to the potential privacy vulnerabilities inherent in these methods. To address this gap, we draw inspiration from gradient inversion attacks in FL and propose the federated unlearning inversion attack (FUIA). The FUIA is specifically designed for the three types of FU (sample unlearning, client unlearning, and class unlearning), aiming to provide a comprehensive analysis of the privacy leakage risks associated with FU. In FUIA, the server acts as an honest-but-curious attacker, recording and exploiting the model differences before and after unlearning to expose the features and labels of forgotten data. FUIA significantly leaks the privacy of forgotten data and can target all types of FU. This attack contradicts the goal of FU to eliminate specific data influence, instead exploiting its vulnerabilities to recover forgotten data and expose its privacy flaws. Extensive experimental results show that FUIA can effectively reveal the private information of forgotten data. To mitigate this privacy leakage, we also explore two potential defense methods, although these come at the cost of reduced unlearning effectiveness and the usability of the unlearned model. 

**Abstract (ZH)**: 随着“被遗忘权”相关法规的引入，联邦学习（FL）面临着新的隐私合规挑战。为应对这些挑战，研究人员提出了联邦遗忘（FU）。然而，现有的FU研究主要集中在提高遗忘的效率上，对这些方法中固有的潜在隐私漏洞关注较少。为弥补这一不足，我们从FL中的梯度反转攻击汲取灵感，提出了一种联邦遗忘反转攻击（FUIA）。FUIA特别设计用于三种类型的FU（样本遗忘、客户端遗忘和类遗忘），旨在对FU引起的隐私泄露风险进行全面分析。在FUIA中，服务器充当诚实但好奇的攻击者，记录和利用在遗忘前后模型的差异，以揭示被遗忘数据的特征和标签。FUIA显著泄露了被遗忘数据的隐私，并且可以针对所有类型的FU。该攻击与FU消除特定数据影响的目标相悖，反而利用了其漏洞来恢复被遗忘的数据并暴露其隐私缺陷。大量实验结果表明，FUIA能够有效揭示被遗忘数据的私人信息。为了减轻这一隐私泄露，我们还探讨了两种潜在的防御方法，尽管这些方法会降低遗忘效果并影响已遗忘模型的可用性。 

---
# Multiscale Byte Language Models -- A Hierarchical Architecture for Causal Million-Length Sequence Modeling 

**Title (ZH)**: 多尺度字节语言模型——因果百万长度序列建模的层次架构 

**Authors**: Eric Egli, Matteo Manica, Jannis Born  

**Link**: [PDF](https://arxiv.org/pdf/2502.14553)  

**Abstract**: Bytes form the basis of the digital world and thus are a promising building block for multimodal foundation models. Recently, Byte Language Models (BLMs) have emerged to overcome tokenization, yet the excessive length of bytestreams requires new architectural paradigms. Therefore, we present the Multiscale Byte Language Model (MBLM), a model-agnostic hierarchical decoder stack that allows training with context windows of $5$M bytes on single GPU in full model precision. We thoroughly examine MBLM's performance with Transformer and Mamba blocks on both unimodal and multimodal tasks. Our experiments demonstrate that hybrid architectures are efficient in handling extremely long byte sequences during training while achieving near-linear generational efficiency. To the best of our knowledge, we present the first evaluation of BLMs on visual Q\&A tasks and find that, despite serializing images and the absence of an encoder, a MBLM with pure next token prediction can match custom CNN-LSTM architectures with designated classification heads. We show that MBLMs exhibit strong adaptability in integrating diverse data representations, including pixel and image filestream bytes, underlining their potential toward omnimodal foundation models. Source code is publicly available at: this https URL 

**Abstract (ZH)**: 字节构成了数字世界的基石，因此是多模态基础模型中一个有前途的构建块。最近，Byte语言模型（BLMs）出现，旨在克服分词问题，但字节流的过长长度需要新的架构范式。因此，我们提出了多尺度字节语言模型（MBLM），这是一种模型无关的分层解码堆栈，使得能够在单个GPU上以全模型精度训练具有5M字节的上下文窗口。我们详细地考察了MBLM在Transformer和Mamba块下的性能，包括单模态和多模态任务。我们的实验表明，混合架构在处理训练期间极其长的字节序列时具有高效性，并且能够实现接近线性的生成效率。据我们所知，我们首次对视觉问答任务进行了BLMs的评估，并发现，在序列化图像和缺乏编码器的情况下，纯后续令牌预测的MBLM可以与具有指定分类头的定制CNN-LSTM架构相匹配。我们展示了MBLM在整合各种数据表示形式方面表现出强大的适应性，包括像素和图像文件流字节，突显了它们向全能模态基础模型发展的潜力。源代码可以在此公开访问：https://this-url-removed-for-privacy-com 

---
# Position: Graph Learning Will Lose Relevance Due To Poor Benchmarks 

**Title (ZH)**: 位置：由于基准不佳，图学习的研究相关性将会下降。 

**Authors**: Maya Bechler-Speicher, Ben Finkelshtein, Fabrizio Frasca, Luis Müller, Jan Tönshoff, Antoine Siraudin, Viktor Zaverkin, Michael M. Bronstein, Mathias Niepert, Bryan Perozzi, Mikhail Galkin, Christopher Morris  

**Link**: [PDF](https://arxiv.org/pdf/2502.14546)  

**Abstract**: While machine learning on graphs has demonstrated promise in drug design and molecular property prediction, significant benchmarking challenges hinder its further progress and relevance. Current benchmarking practices often lack focus on transformative, real-world applications, favoring narrow domains like two-dimensional molecular graphs over broader, impactful areas such as combinatorial optimization, relational databases, or chip design. Additionally, many benchmark datasets poorly represent the underlying data, leading to inadequate abstractions and misaligned use cases. Fragmented evaluations and an excessive focus on accuracy further exacerbate these issues, incentivizing overfitting rather than fostering generalizable insights. These limitations have prevented the development of truly useful graph foundation models. This position paper calls for a paradigm shift toward more meaningful benchmarks, rigorous evaluation protocols, and stronger collaboration with domain experts to drive impactful and reliable advances in graph learning research, unlocking the potential of graph learning. 

**Abstract (ZH)**: 尽管图上的机器学习在药物设计和分子性质预测方面展现了潜力，但显著的基准测试挑战阻碍了其进一步的发展和应用相关性。当前的基准测试实践往往过于关注变革性的、真实世界的应用场景不足，倾向于关注狭窄的领域如二维分子图，而不是更为广泛且具有影响力的领域，如组合优化、关系数据库或芯片设计。此外，许多基准数据集未能很好地代表底层数据，导致抽象不足且应用场景与实际需求不符。碎片化的评估方式和过度关注准确性进一步加剧了这些问题，促使过度拟合而非促进可泛化的洞见。这些局限性阻碍了真正有用的基础图模型的开发。本文呼吁转向更具意义的基准测试、严格的评估协议，并加强与领域专家的协作，以推动图学习研究取得更有影响力和可靠性的进展，从而释放图学习的潜力。 

---
# CORBA: Contagious Recursive Blocking Attacks on Multi-Agent Systems Based on Large Language Models 

**Title (ZH)**: 基于大型语言模型的多智能体系统中的传染性递归阻塞攻击（CORBA） 

**Authors**: Zhenhong Zhou, Zherui Li, Jie Zhang, Yuanhe Zhang, Kun Wang, Yang Liu, Qing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.14529)  

**Abstract**: Large Language Model-based Multi-Agent Systems (LLM-MASs) have demonstrated remarkable real-world capabilities, effectively collaborating to complete complex tasks. While these systems are designed with safety mechanisms, such as rejecting harmful instructions through alignment, their security remains largely unexplored. This gap leaves LLM-MASs vulnerable to targeted disruptions. In this paper, we introduce Contagious Recursive Blocking Attacks (Corba), a novel and simple yet highly effective attack that disrupts interactions between agents within an LLM-MAS. Corba leverages two key properties: its contagious nature allows it to propagate across arbitrary network topologies, while its recursive property enables sustained depletion of computational resources. Notably, these blocking attacks often involve seemingly benign instructions, making them particularly challenging to mitigate using conventional alignment methods. We evaluate Corba on two widely-used LLM-MASs, namely, AutoGen and Camel across various topologies and commercial models. Additionally, we conduct more extensive experiments in open-ended interactive LLM-MASs, demonstrating the effectiveness of Corba in complex topology structures and open-source models. Our code is available at: this https URL. 

**Abstract (ZH)**: 基于大规模语言模型的多智能体系统（LLM-MAS）已经在实际应用中展现了卓越的能力，能够有效协作完成复杂任务。尽管这些系统设计了安全机制，例如通过对齐来拒绝有害指令，但它们的安全性仍很大程度上未被探索。这一差距使LLM-MASs容易受到有针对性的干扰。在本文中，我们介绍了一种新颖且简单但极具效果的攻击方法——传染性递归阻断攻击（Corba），这种攻击能够破坏LLM-MAS内部智能体之间的交互。Corba利用了两个关键特性：传染性使它能够在任意网络拓扑下传播，而递归性则使其能够持续耗尽计算资源。值得注意的是，这些阻断攻击通常涉及看似无害的指令，这使得它们更难以通过传统的对齐方法进行缓解。我们分别在AutoGen和Camel两个广泛使用的LLM-MAS上，在不同的拓扑结构和商用模型上评估了Corba。此外，我们在开放生成互动的LLM-MAS中进行了更广泛的实验，展示了Corba在复杂拓扑结构和开源模型中的有效性。相关代码可在以下链接获取：这个 https URL。 

---
# Small Graph Is All You Need: DeepStateGNN for Scalable Traffic Forecasting 

**Title (ZH)**: 小图即所有你所需：DeepStateGNN 用于可扩展的交通预测 

**Authors**: Yannick Wölker, Arash Hajisafi, Cyrus Shahabi, Matthias Renz  

**Link**: [PDF](https://arxiv.org/pdf/2502.14525)  

**Abstract**: We propose a novel Graph Neural Network (GNN) model, named DeepStateGNN, for analyzing traffic data, demonstrating its efficacy in two critical tasks: forecasting and reconstruction. Unlike typical GNN methods that treat each traffic sensor as an individual graph node, DeepStateGNN clusters sensors into higher-level graph nodes, dubbed Deep State Nodes, based on various similarity criteria, resulting in a fixed number of nodes in a Deep State graph. The term "Deep State" nodes is a play on words, referencing hidden networks of power that, like these nodes, secretly govern traffic independently of visible sensors. These Deep State Nodes are defined by several similarity factors, including spatial proximity (e.g., sensors located nearby in the road network), functional similarity (e.g., sensors on similar types of freeways), and behavioral similarity under specific conditions (e.g., traffic behavior during rain). This clustering approach allows for dynamic and adaptive node grouping, as sensors can belong to multiple clusters and clusters may evolve over time. Our experimental results show that DeepStateGNN offers superior scalability and faster training, while also delivering more accurate results than competitors. It effectively handles large-scale sensor networks, outperforming other methods in both traffic forecasting and reconstruction accuracy. 

**Abstract (ZH)**: 我们提出了一种名为DeepStateGNN的新型图神经网络（GNN）模型，用于分析交通数据，并在两种关键任务——预测和重构中展示了其有效性。与典型的GNN方法将每个交通传感器视为单独的图节点不同，DeepStateGNN基于多种相似性标准将传感器聚类为更高层次的图节点，称为“Deep State”节点，从而在“Deep State”图中获得固定数量的节点。“Deep State”节点一词字面上是指隐含在网络中的权力网络，就像这些节点一样，它们在不依赖于可见传感器的情况下独立于可见传感器秘密地控制交通。这些“Deep State”节点由多个相似性因素定义，包括空间邻近性（例如，在路网中位于附近的传感器）、功能相似性（例如，在类似类型的高速公路上的传感器），以及在特定条件下行为相似性（例如，雨天交通行为）。这种聚类方法允许动态和自适应的节点分组，因为在多个聚类中传感器可能属于同一类别，并且聚类也可能会随时间变化。我们的实验结果表明，与竞争对手相比，DeepStateGNN在可扩展性和训练速度方面表现出更优越的表现，并且在预测和重构精度上也提供了更准确的结果。它能够有效地处理大规模传感器网络，在交通预测和重构准确性方面均优于其他方法。 

---
# PLPHP: Per-Layer Per-Head Vision Token Pruning for Efficient Large Vision-Language Models 

**Title (ZH)**: PLPHP：高效大规模视觉语言模型的逐层逐头 vision token 剪枝 

**Authors**: Yu Meng, Kaiyuan Li, Chenran Huang, Chen Gao, Xinlei Chen, Yong Li, Xiaoping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14504)  

**Abstract**: Large Vision-Language Models (LVLMs) have demonstrated remarkable capabilities across a range of multimodal tasks. However, their inference efficiency is constrained by the large number of visual tokens processed during decoding. To address this challenge, we propose Per-Layer Per-Head Vision Token Pruning (PLPHP), a two-level fine-grained pruning method including Layer-Level Retention Rate Allocation and Head-Level Vision Token Pruning. Motivated by the Vision Token Re-attention phenomenon across decoder layers, we dynamically adjust token retention rates layer by layer. Layers that exhibit stronger attention to visual information preserve more vision tokens, while layers with lower vision attention are aggressively pruned. Furthermore, PLPHP applies pruning at the attention head level, enabling different heads within the same layer to independently retain critical context. Experiments on multiple benchmarks demonstrate that PLPHP delivers an 18% faster decoding speed and reduces the Key-Value Cache (KV Cache) size by over 50%, all at the cost of 0.46% average performance drop, while also achieving notable performance improvements in multi-image tasks. These results highlight the effectiveness of fine-grained token pruning and contribute to advancing the efficiency and scalability of LVLMs. Our source code will be made publicly available. 

**Abstract (ZH)**: 大型多模态模型（Large Vision-Language Models, LVLMs）在多种多模态任务中展现出了显著的能力。然而，它们的推理效率受到解码过程中大量视觉令牌处理的限制。为了解决这一挑战，我们提出了逐层逐头视觉令牌剪枝（Per-Layer Per-Head Vision Token Pruning，PLPHP），这是一种两级细化剪枝方法，包括逐层保留率分配（Layer-Level Retention Rate Allocation）和逐头视觉令牌剪枝（Head-Level Vision Token Pruning）。受到解码层间视觉令牌再注意力现象的启发，我们逐层动态调整令牌保留率。表现出较强视觉信息注意力的层保留更多的视觉令牌，而视觉注意力较低的层则被激进地剪枝。此外，PLPHP 在注意力头层进行剪枝，使同一层内的不同头可以独立保留关键上下文。在多个基准测试上的实验结果表明，PLPHP 可以实现 18% 的更快解码速度，并且将关键值缓存（Key-Value Cache, KV Cache）的大小减少超过 50%，性能平均下降 0.46%，而在多图任务上也取得了显著的性能提升。这些结果突显了细致粒度令牌剪枝的有效性，并有助于提升 LVLMs 的效率和可扩展性。我们的源代码将会公开提供。 

---
# MLGym: A New Framework and Benchmark for Advancing AI Research Agents 

**Title (ZH)**: MLGym：一个新的框架与基准，用于推动AI研究代理的发展 

**Authors**: Deepak Nathani, Lovish Madaan, Nicholas Roberts, Nikolay Bashlykov, Ajay Menon, Vincent Moens, Amar Budhiraja, Despoina Magka, Vladislav Vorotilov, Gaurav Chaurasia, Dieuwke Hupkes, Ricardo Silveira Cabral, Tatiana Shavrina, Jakob Foerster, Yoram Bachrach, William Yang Wang, Roberta Raileanu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14499)  

**Abstract**: We introduce Meta MLGym and MLGym-Bench, a new framework and benchmark for evaluating and developing LLM agents on AI research tasks. This is the first Gym environment for machine learning (ML) tasks, enabling research on reinforcement learning (RL) algorithms for training such agents. MLGym-bench consists of 13 diverse and open-ended AI research tasks from diverse domains such as computer vision, natural language processing, reinforcement learning, and game theory. Solving these tasks requires real-world AI research skills such as generating new ideas and hypotheses, creating and processing data, implementing ML methods, training models, running experiments, analyzing the results, and iterating through this process to improve on a given task. We evaluate a number of frontier large language models (LLMs) on our benchmarks such as Claude-3.5-Sonnet, Llama-3.1 405B, GPT-4o, o1-preview, and Gemini-1.5 Pro. Our MLGym framework makes it easy to add new tasks, integrate and evaluate models or agents, generate synthetic data at scale, as well as develop new learning algorithms for training agents on AI research tasks. We find that current frontier models can improve on the given baselines, usually by finding better hyperparameters, but do not generate novel hypotheses, algorithms, architectures, or substantial improvements. We open-source our framework and benchmark to facilitate future research in advancing the AI research capabilities of LLM agents. 

**Abstract (ZH)**: 我们将介绍一种新的框架和基准MLGym和MLGym-Bench，用于评估和开发在AI研究任务中工作的LLM代理。这是第一个专为机器学习(ML)任务设计的Gym环境，它使研究人员能够研究强化学习(RL)算法以训练此类代理。MLGym-Bench包含了来自计算机视觉、自然语言处理、强化学习和博弈论等多个领域的13个多样性和开放性研究任务。解决这些任务需要实际的AI研究技能，如提出新想法和假说、创建和处理数据、实施机器学习方法、训练模型、运行实验、分析结果，并通过这一过程迭代以提高任务表现。我们对我们的基准测试评估了若干前沿的大型语言模型（LLMs），如Claude-3.5-Sonnet、Llama-3.1 405B、GPT-4o、o1-preview和Gemini-1.5 Pro。我们的MLGym框架使新增任务、整合和评估模型或代理、大规模生成合成数据以及为训练代理于AI研究任务开发新的学习算法变得更加简便。我们发现，当前的前沿模型通常通过找到更优的超参数来改善给定的基线，但并未生成新的假说、算法、架构或显著的改进。我们将我们的框架和基准测试开源，以促进未来在提升LLM代理的AI研究能力方面的工作。 

---
# Temporal Misalignment and Probabilistic Neurons 

**Title (ZH)**: 时间对齐问题与概率神经元 

**Authors**: Velibor Bojković, Xiaofeng Wu, Bin Gu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14487)  

**Abstract**: Spiking Neural Networks (SNNs) offer a more energy-efficient alternative to Artificial Neural Networks (ANNs) by mimicking biological neural principles, establishing them as a promising approach to mitigate the increasing energy demands of large-scale neural models. However, fully harnessing the capabilities of SNNs remains challenging due to their discrete signal processing and temporal dynamics. ANN-SNN conversion has emerged as a practical approach, enabling SNNs to achieve competitive performance on complex machine learning tasks. In this work, we identify a phenomenon in the ANN-SNN conversion framework, termed temporal misalignment, in which random spike rearrangement across SNN layers leads to performance improvements. Based on this observation, we introduce biologically plausible two-phase probabilistic (TPP) spiking neurons, further enhancing the conversion process. We demonstrate the advantages of our proposed method both theoretically and empirically through comprehensive experiments on CIFAR-10/100, CIFAR10-DVS, and ImageNet across a variety of architectures, achieving state-of-the-art results. 

**Abstract (ZH)**: 脉冲神经网络（SNNs）通过模拟生物神经原则提供了比人工神经网络（ANNs）更具能效的替代方案，使它们成为缓解大规模神经模型不断增加的能量需求的一种有前途的方法。然而，完全利用SNNs的能力仍然充满挑战，因为它们具有离散信号处理和时间动态特性。ANN-SNN转换作为一种实用的方法已经出现，使SNNs能够在复杂机器学习任务中获得竞争力的性能。在本工作中，我们识别了在ANN-SNN转换框架中出现的一种现象，称为时间对齐偏差。在这种现象中，SNN层之间的随机脉冲重新排列导致了性能提升。基于这一观察，我们引入了一种生物上可行的两阶段概率（TPP）脉冲神经元，进一步改进了转换过程。我们通过在CIFAR-10/100、CIFAR10-DVS和ImageNet上对各种架构进行全面的实验，从理论和实证两方面展示了我们提出方法的优势，并取得了最先进的成果。 

---
# How Jailbreak Defenses Work and Ensemble? A Mechanistic Investigation 

**Title (ZH)**: 《 Jailbreak 防御机制及其集成研究 —— 一种机制性调查 》

这种翻译保持了原标题的学术严谨性，同时确保了中文表达的清晰和准确。这里的“Jailbreak”在安全领域通常指对某种系统（如移动设备操作系统）的安全限制进行规避，因此在翻译标题时使用了“Jailbreak”及其相关术语。 

**Authors**: Zhuohang Long, Siyuan Wang, Shujun Liu, Yuhang Lai, Xuanjing Huang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.14486)  

**Abstract**: Jailbreak attacks, where harmful prompts bypass generative models' built-in safety, raise serious concerns about model vulnerability. While many defense methods have been proposed, the trade-offs between safety and helpfulness, and their application to Large Vision-Language Models (LVLMs), are not well understood. This paper systematically examines jailbreak defenses by reframing the standard generation task as a binary classification problem to assess model refusal tendencies for both harmful and benign queries. We identify two key defense mechanisms: safety shift, which increases refusal rates across all queries, and harmfulness discrimination, which improves the model's ability to distinguish between harmful and benign inputs. Using these mechanisms, we develop two ensemble defense strategies-inter-mechanism ensembles and intra-mechanism ensembles-to balance safety and helpfulness. Experiments on the MM-SafetyBench and MOSSBench datasets with LLaVA-1.5 models show that these strategies effectively improve model safety or optimize the trade-off between safety and helpfulness. 

**Abstract (ZH)**: 在“监狱越狱”攻击中，有害提示绕过了生成模型内置的安全机制，这引起了对模型脆弱性的严重关注。虽然已经提出了许多防御方法，但安全性和帮助性之间的权衡，以及这些方法在大规模视觉-语言模型（LVLMs）中的应用，尚不完全明确。本文系统地研究了“监狱越狱”防御方法，通过将标准生成任务重新定义为二元分类问题来评估模型对有害和良性查询的拒绝倾向。我们识别出了两种关键的防御机制：安全转换，它增加了所有查询的拒绝率；和危害性区分，它提高了模型区分有害和良性输入的能力。利用这些机制，我们提出了两种集成防御策略—跨机制集成和内机制集成，以平衡安全性和帮助性。通过对MM-SafetyBench和MOSSBench数据集上的LLaVA-1.5模型进行实验，我们发现这些策略能够有效提高模型的安全性或优化安全性和帮助性的权衡。 

---
# Enhancing Smart Environments with Context-Aware Chatbots using Large Language Models 

**Title (ZH)**: 使用大规模语言模型增强基于情境意识的智能环境中的聊天机器人 

**Authors**: Aurora Polo-Rodríguez, Laura Fiorini, Erika Rovini, Filippo Cavallo, Javier Medina-Quero  

**Link**: [PDF](https://arxiv.org/pdf/2502.14469)  

**Abstract**: This work presents a novel architecture for context-aware interactions within smart environments, leveraging Large Language Models (LLMs) to enhance user experiences. Our system integrates user location data obtained through UWB tags and sensor-equipped smart homes with real-time human activity recognition (HAR) to provide a comprehensive understanding of user context. This contextual information is then fed to an LLM-powered chatbot, enabling it to generate personalised interactions and recommendations based on the user's current activity and environment. This approach moves beyond traditional static chatbot interactions by dynamically adapting to the user's real-time situation. A case study conducted from a real-world dataset demonstrates the feasibility and effectiveness of our proposed architecture, showcasing its potential to create more intuitive and helpful interactions within smart homes. The results highlight the significant benefits of integrating LLM with real-time activity and location data to deliver personalised and contextually relevant user experiences. 

**Abstract (ZH)**: 本研究提出了一种新的架构，旨在增强智能环境中的上下文感知交互，通过利用大规模语言模型（LLMs）提升用户体验。我们的系统将通过超宽带（UWB）标签和传感器装备的智能家庭获得的用户位置数据与实时人体活动识别（HAR）集成，从而提供对用户上下文的全面理解。随后，该上下文信息被输送到一个基于LLMs的聊天机器人中，使其能够根据用户的当前活动和环境生成个性化的交互和建议。这种方法超越了传统的静态聊天机器人交互，能够根据用户的实时情况动态调整。通过对真实世界数据集进行的案例研究展示了我们提出的架构的实际可行性和有效性，展示了其在智能家庭中创建更加直观和有用交互的潜力。研究结果突显了将LLMs与实时活动和位置数据相结合以提供个性化和上下文相关的用户体验的巨大优势。 

---
# Single-image Reflectance and Transmittance Estimation from Any Flatbed Scanner 

**Title (ZH)**: 从任意平板扫描仪估计单张图像的反射率和透射率 

**Authors**: Carlos Rodriguez-Pardo, David Pascual-Hernandez, Javier Rodriguez-Vazquez, Jorge Lopez-Moreno, Elena Garces  

**Link**: [PDF](https://arxiv.org/pdf/2502.14462)  

**Abstract**: Flatbed scanners have emerged as promising devices for high-resolution, single-image material capture. However, existing approaches assume very specific conditions, such as uniform diffuse illumination, which are only available in certain high-end devices, hindering their scalability and cost. In contrast, in this work, we introduce a method inspired by intrinsic image decomposition, which accurately removes both shading and specularity, effectively allowing captures with any flatbed scanner. Further, we extend previous work on single-image material reflectance capture with the estimation of opacity and transmittance, critical components of full material appearance (SVBSDF), improving the results for any material captured with a flatbed scanner, at a very high resolution and accuracy 

**Abstract (ZH)**: 平板扫描仪被认为是高分辨率单图像材质捕获的有前途的设备。然而，现有的方法假设非常具体的情况，例如均匀漫射照明，这种条件仅在某些高端设备中可用，从而阻碍了其扩展性和成本效益。相比之下，本工作中我们介绍了一种受固有图像分解启发的方法，该方法能够准确地去除阴影和反光，从而使使用任何平板扫描仪进行捕获成为可能。此外，我们在此前单图像材质反射率捕获工作的基础上，估计了不透明度和透射率，这是全材质外观（SVBSDF）的关键组成部分，从而提高了使用平板扫描仪捕获任何材质的高分辨率和高精度结果。 

---
# Llamba: Scaling Distilled Recurrent Models for Efficient Language Processing 

**Title (ZH)**: Llamba：扩展蒸馏循环模型以实现高效的语言处理 

**Authors**: Aviv Bick, Tobias Katsch, Nimit Sohoni, Arjun Desai, Albert Gu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14458)  

**Abstract**: We introduce Llamba, a family of efficient recurrent language models distilled from Llama-3.x into the Mamba architecture. The series includes Llamba-1B, Llamba-3B, and Llamba-8B, which achieve higher inference throughput and handle significantly larger batch sizes than Transformer-based models while maintaining comparable benchmark performance. Furthermore, Llamba demonstrates the effectiveness of cross-architecture distillation using MOHAWK (Bick et al., 2024), achieving these results with less than 0.1% of the training data typically used for models of similar size. To take full advantage of their efficiency, we provide an optimized implementation of Llamba for resource-constrained devices such as smartphones and edge platforms, offering a practical and memory-efficient alternative to Transformers. Overall, Llamba improves the tradeoff between speed, memory efficiency, and performance, making high-quality language models more accessible. 

**Abstract (ZH)**: 我们介绍了Llamba，这是一个高效的递归语言模型系列，从Llama-3.x中提炼到Mamba架构中。该系列包括Llamba-1B、Llamba-3B和Llamba-8B，与基于Transformer的模型相比，它们在保持相当的基准性能的同时，实现了更高的推理吞吐量并能够处理更大的批次大小。此外，Llamba通过MOHAWK（Bick et al., 2024）展示了跨架构提炼的有效性，仅使用相当于同类模型通常所需训练数据不到0.1%的数据量就达成了这些结果。为了充分利用其高效性，我们为资源受限的设备（如智能手机和边缘平台）提供了Llamba的优化实现，提供了一种与Transformer相比更实用且内存高效的替代方案。总体而言，Llamba改善了速度、内存效率与性能之间的权衡关系，使高质量的语言模型更加普及。 

---
# Watch Less, Feel More: Sim-to-Real RL for Generalizable Articulated Object Manipulation via Motion Adaptation and Impedance Control 

**Title (ZH)**: 减少观看，增强感受：基于运动适应和阻抗控制的通用 articulated 对象 manipulation 的从仿真到现实的 RL 方法 

**Authors**: Tan-Dzung Do, Nandiraju Gireesh, Jilong Wang, He Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14457)  

**Abstract**: Articulated object manipulation poses a unique challenge compared to rigid object manipulation as the object itself represents a dynamic environment. In this work, we present a novel RL-based pipeline equipped with variable impedance control and motion adaptation leveraging observation history for generalizable articulated object manipulation, focusing on smooth and dexterous motion during zero-shot sim-to-real transfer. To mitigate the sim-to-real gap, our pipeline diminishes reliance on vision by not leveraging the vision data feature (RGBD/pointcloud) directly as policy input but rather extracting useful low-dimensional data first via off-the-shelf modules. Additionally, we experience less sim-to-real gap by inferring object motion and its intrinsic properties via observation history as well as utilizing impedance control both in the simulation and in the real world. Furthermore, we develop a well-designed training setting with great randomization and a specialized reward system (task-aware and motion-aware) that enables multi-staged, end-to-end manipulation without heuristic motion planning. To the best of our knowledge, our policy is the first to report 84\% success rate in the real world via extensive experiments with various unseen objects. 

**Abstract (ZH)**: 相对于刚体物体操纵，具有关节的物体操纵提出了一种独特的挑战，因为物体本身作为一个动态环境具有动态性。在这项工作中，我们提出了一种新颖的基于强化学习（RL）的管道，并配备了可变阻抗控制和运动适应性，通过观察历史信息实现对各类关节物体操纵的一般化，特别是在零样本模拟到现实世界的转移中确保平滑和灵巧的运动。为减轻模拟与现实之间的差距，我们的管道减少了对视觉数据的依赖，不直接将RGBD/点云等视觉特征作为策略的输入，而是首先通过现成的模块提取有用的数据。此外，通过利用观察历史来推断物体运动及其内在属性，以及在模拟和现实中均采用阻抗控制，我们进一步减少了模拟与现实之间的差距。此外，我们设计了一个具有大量随机化和特定奖励机制（任务感知和运动感知）的训练环境，这使得多阶段、端到端的操纵成为可能，无需启发式运动规划。据我们所知，我们的策略在各种未见过的物体的广泛实验中，在现实世界中的成功率达到了84%，这是首次报告的此类结果。 

---
# An Efficient Ground-aerial Transportation System for Pest Control Enabled by AI-based Autonomous Nano-UAVs 

**Title (ZH)**: 基于AI自主纳米无人机的高效地面-空中害虫防控交通系统 

**Authors**: Luca Crupi, Luca Butera, Alberto Ferrante, Alessandro Giusti, Daniele Palossi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14455)  

**Abstract**: Efficient crop production requires early detection of pest outbreaks and timely treatments; we consider a solution based on a fleet of multiple autonomous miniaturized unmanned aerial vehicles (nano-UAVs) to visually detect pests and a single slower heavy vehicle that visits the detected outbreaks to deliver treatments. To cope with the extreme limitations aboard nano-UAVs, e.g., low-resolution sensors and sub-100 mW computational power budget, we design, fine-tune, and optimize a tiny image-based convolutional neural network (CNN) for pest detection. Despite the small size of our CNN (i.e., 0.58 GOps/inference), on our dataset, it scores a mean average precision (mAP) of 0.79 in detecting harmful bugs, i.e., 14% lower mAP but 32x fewer operations than the best-performing CNN in the literature. Our CNN runs in real-time at 6.8 frame/s, requiring 33 mW on a GWT GAP9 System-on-Chip aboard a Crazyflie nano-UAV. Then, to cope with in-field unexpected obstacles, we leverage a global+local path planner based on the A* algorithm. The global path planner determines the best route for the nano-UAV to sweep the entire area, while the local one runs up to 50 Hz aboard our nano-UAV and prevents collision by adjusting the short-distance path. Finally, we demonstrate with in-simulator experiments that once a 25 nano-UAVs fleet has combed a 200x200 m vineyard, collected information can be used to plan the best path for the tractor, visiting all and only required hotspots. In this scenario, our efficient transportation system, compared to a traditional single-ground vehicle performing both inspection and treatment, can save up to 20 h working time. 

**Abstract (ZH)**: 高效的作物生产需要尽早检测到害虫爆发并及时进行治疗；为此，我们考虑基于多架自主微型无人机（nano-UAV）的解决方案，用于视觉检测害虫，同时由一台较慢的重型车辆访问检测到的爆发点并提供治疗。为应对nano-UAV上的极端限制，例如低分辨率传感器和低于100 mW的计算功率预算，我们设计、微调并优化了一种小型图像卷积神经网络（CNN）以用于害虫检测。尽管我们CNN的体积小巧（每次推理0.58 GOps），但在我们的数据集上，该CNN在检测有害昆虫方面的均值平均精度（mAP）达到了0.79，即比文献中表现最好的CNN低14%的mAP，但操作次数少32倍。我们的CNN可以在6.8帧/秒的速度下实时运行，在基于Crazyflie nano-UAV的GWT GAP9片上系统上消耗33 mW。为了应对田间突发障碍，我们利用基于A*算法的全局+局部路径规划器。全局路径规划器决定nano-UAV的最佳路线以清理整个区域，而局部路径规划器在我们的nano-UAV上以高达50 Hz的速度运行，并通过调整短距离路径来防止碰撞。最后，在模拟器实验中，我们展示了当25架nano-UAV组成的车队已清理完200x200米的葡萄园后，收集的信息可用于规划拖拉机的最佳路径，访问所有且仅访问必要的热点区域。在这种情况下，与传统的单一地面车辆同时进行检测和治疗相比，我们的高效运输系统可节省多达20小时的工作时间。 

---
# PredictaBoard: Benchmarking LLM Score Predictability 

**Title (ZH)**: PredictaBoard: 评估大型语言模型得分可预测性基准测试 

**Authors**: Lorenzo Pacchiardi, Konstantinos Voudouris, Ben Slater, Fernando Martínez-Plumed, José Hernández-Orallo, Lexin Zhou, Wout Schellaert  

**Link**: [PDF](https://arxiv.org/pdf/2502.14445)  

**Abstract**: Despite possessing impressive skills, Large Language Models (LLMs) often fail unpredictably, demonstrating inconsistent success in even basic common sense reasoning tasks. This unpredictability poses a significant challenge to ensuring their safe deployment, as identifying and operating within a reliable "safe zone" is essential for mitigating risks. To address this, we present PredictaBoard, a novel collaborative benchmarking framework designed to evaluate the ability of score predictors (referred to as assessors) to anticipate LLM errors on specific task instances (i.e., prompts) from existing datasets. PredictaBoard evaluates pairs of LLMs and assessors by considering the rejection rate at different tolerance errors. As such, PredictaBoard stimulates research into developing better assessors and making LLMs more predictable, not only with a higher average performance. We conduct illustrative experiments using baseline assessors and state-of-the-art LLMs. PredictaBoard highlights the critical need to evaluate predictability alongside performance, paving the way for safer AI systems where errors are not only minimised but also anticipated and effectively mitigated. Code for our benchmark can be found at this https URL 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）拥有令人印象深刻的技能，但它们往往会表现出不可预测的失误，在最基础的常识推理任务中也表现出不一致的成功率。这种不可预测性给确保其安全部署带来了重大挑战，因为识别并操作在可靠的“安全区”内至关重要，以减轻风险。为解决这一问题，我们提出了一种名为PredictaBoard的新型协作基准框架，用于评估评分预测器（称为评估者）的能力，预测LLMs在特定任务实例（即提示）上的错误。PredictaBoard通过考虑不同容忍误差水平下的拒识率来评估LLM评估者的对弈。因此，PredictaBoard促进了开发更好的评估者和使LMs更加可预测的研究，不仅提升平均性能。我们使用基线评估者和最先进的LMs进行了示范性实验。PredictaBoard强调了评估可预测性和性能的必要性，为一种新的安全人工智能系统铺平了道路，在这种系统中，错误不仅被最小化，还能被预见和有效缓解。我们的基准代码可以在以下链接找到：这个 https URL 

---
# Stochastic Resonance Improves the Detection of Low Contrast Images in Deep Learning Models 

**Title (ZH)**: 随机共振提高深度学习模型中低对比度图像的检测能力 

**Authors**: Siegfried Ludwig  

**Link**: [PDF](https://arxiv.org/pdf/2502.14442)  

**Abstract**: Stochastic resonance describes the utility of noise in improving the detectability of weak signals in certain types of systems. It has been observed widely in natural and engineered settings, but its utility in image classification with rate-based neural networks has not been studied extensively. In this analysis a simple LSTM recurrent neural network is trained for digit recognition and classification. During the test phase, image contrast is reduced to a point where the model fails to recognize the presence of a stimulus. Controlled noise is added to partially recover classification performance. The results indicate the presence of stochastic resonance in rate-based recurrent neural networks. 

**Abstract (ZH)**: 随机共振描述了噪声在某些系统中提高微弱信号可检测性的有用性。它在自然和工程环境中已被广泛观察到，但在基于速率的神经网络进行图像分类中的应用尚不广泛研究。在本分析中，一个简单的长短期记忆（LSTM）循环神经网络被训练用于数字识别和分类。在测试阶段，将图像对比度降低到模型无法识别刺激存在的程度。然后在模型中添加受控噪声以部分恢复分类性能。结果表明，在基于速率的循环神经网络中存在随机共振。 

---
# Distribution Matching for Self-Supervised Transfer Learning 

**Title (ZH)**: 自我监督迁移学习中的分布匹配 

**Authors**: Yuling Jiao, Wensen Ma, Defeng Sun, Hansheng Wang, Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14424)  

**Abstract**: In this paper, we propose a novel self-supervised transfer learning method called Distribution Matching (DM), which drives the representation distribution toward a predefined reference distribution while preserving augmentation invariance. The design of DM results in a learned representation space that is intuitively structured and offers easily interpretable hyperparameters. Experimental results across multiple real-world datasets and evaluation metrics demonstrate that DM performs competitively on target classification tasks compared to existing self-supervised transfer learning methods. Additionally, we provide robust theoretical guarantees for DM, including a population theorem and an end-to-end sample theorem. The population theorem bridges the gap between the self-supervised learning task and target classification accuracy, while the sample theorem shows that, even with a limited number of samples from the target domain, DM can deliver exceptional classification performance, provided the unlabeled sample size is sufficiently large. 

**Abstract (ZH)**: 在本文中，我们提出了一种新颖的自监督迁移学习方法，称为分布匹配（Distribution Matching，DM）。该方法旨在将表示分布推向一个预定义的参考分布，同时保持增强不变性。DM的设计产生了一个直观结构化的学习表示空间，并提供了易于解释的超参数。在多个真实世界数据集和评估指标上的实验结果表明，DM 在目标分类任务上与现有的自监督迁移学习方法相比具有竞争力。此外，我们还为DM提供了稳健的理论保证，包括总体定理和端到端样本定理。总体定理填补了自监督学习任务与目标分类准确性之间的差距，而样本定理表明，在目标域的样本数量有限的情况下，只要未标注样本量足够大，DM 就可以交付出色的分类性能。 

---
# Reliable Explainability of Deep Learning Spatial-Spectral Classifiers for Improved Semantic Segmentation in Autonomous Driving 

**Title (ZH)**: 面向自主驾驶中改进语义分割的深度学习空谱分类器的可靠可解释性研究 

**Authors**: Jon Gutiérrez-Zaballa, Koldo Basterretxea, Javier Echanobe  

**Link**: [PDF](https://arxiv.org/pdf/2502.14416)  

**Abstract**: Integrating hyperspectral imagery (HSI) with deep neural networks (DNNs) can strengthen the accuracy of intelligent vision systems by combining spectral and spatial information, which is useful for tasks like semantic segmentation in autonomous driving. To advance research in such safety-critical systems, determining the precise contribution of spectral information to complex DNNs' output is needed. To address this, several saliency methods, such as class activation maps (CAM), have been proposed primarily for image classification. However, recent studies have raised concerns regarding their reliability. In this paper, we address their limitations and propose an alternative approach by leveraging the data provided by activations and weights from relevant DNN layers to better capture the relationship between input features and predictions. The study aims to assess the superior performance of HSI compared to 3-channel and single-channel DNNs. We also address the influence of spectral signature normalization for enhancing DNN robustness in real-world driving conditions. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

将高光谱成像（HSI）与深度神经网络（DNNs）相结合，可以通过结合光谱和空间信息来增强智能视觉系统的准确性，这对于自动驾驶中的语义分割等任务非常有用。为了推动此类安全关键系统的研究，需要确定光谱信息对复杂DNNs输出的精确贡献。为解决这一问题，已经提出了多种显著性方法，如类激活映射（CAM），这些方法主要用于图像分类。然而，近期的研究对这些方法的有效性提出了质疑。在本文中，我们探讨了它们的局限性，并通过利用来自相关DNN层的激活和权重数据，提出了一种替代方法，以更好地捕捉输入特征与预测之间的关系。本研究旨在评估HSI相对于3通道和单通道DNNs的优越性能。我们还探讨了光谱特征归一化的影响，以提高DNN在实际驾驶条件下的鲁棒性。 

---
# S*: Test Time Scaling for Code Generation 

**Title (ZH)**: S*: 转码时的代码生成缩放方法 

**Authors**: Dacheng Li, Shiyi Cao, Chengkun Cao, Xiuyu Li, Shangyin Tan, Kurt Keutzer, Jiarong Xing, Joseph E. Gonzalez, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2502.14382)  

**Abstract**: Increasing test-time compute for LLMs shows promise across domains but remains underexplored in code generation, despite extensive study in math. In this paper, we propose S*, the first hybrid test-time scaling framework that substantially improves the coverage and selection accuracy of generated code. S* extends the existing parallel scaling paradigm with sequential scaling to push performance boundaries. It further leverages a novel selection mechanism that adaptively generates distinguishing inputs for pairwise comparison, combined with execution-grounded information to robustly identify correct solutions. We evaluate across 12 Large Language Models and Large Reasoning Model and show: (1) S* consistently improves performance across model families and sizes, enabling a 3B model to outperform GPT-4o-mini; (2) S* enables non-reasoning models to surpass reasoning models - GPT-4o-mini with S* outperforms o1-preview by 3.7% on LiveCodeBench; (3) S* further boosts state-of-the-art reasoning models - DeepSeek-R1-Distill-Qwen-32B with S* achieves 85.7% on LiveCodeBench, approaching o1 (high) at 88.5%. Code will be available under this https URL. 

**Abstract (ZH)**: 增加推理时的计算量在语言模型（LLMs）和大规模推理模型（LRMs）中显示出跨领域的前景，但在代码生成领域仍被大大忽视，尽管在数学领域已有广泛研究。在本文中，我们提出了S*，这是首个混合推理时扩展框架，能够显著提高生成代码的覆盖率和选择准确性。S*扩展了现有的并行扩展范式，通过引入顺序扩展来推动性能边界。此外，S*还利用了一种新颖的选择机制，该机制能够自适应地生成用于两两比较的区分性输入，并结合执行驱动的信息来稳健地识别正确的解决方案。我们在12个不同类型的大型语言模型和大型推理模型上进行了评估，结果显示：（1）S*在不同模型家族和规模上一致地提升了性能，使一个3B模型能够超越GPT-4o-mini；（2）S*使得非推理模型能够超越推理模型——带有S*的GPT-4o-mini在LiveCodeBench上比o1-preview高出3.7%；（3）S*进一步提升了最前沿的推理模型——带有S*的DeepSeek-R1-Distill-Qwen-32B在LiveCodeBench上达到了85.7%，接近o1（高）的88.5%。代码将在以下链接下提供：<https://your-link-url>。 

---
# Affinity and Diversity: A Unified Metric for Demonstration Selection via Internal Representations 

**Title (ZH)**: 亲和度与多样性：一种通过内部表示进行示范选择的统一评价指标 

**Authors**: Mariko Kato, Hakaze Cho, Yoshihiro Sakai, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2502.14380)  

**Abstract**: The performance of In-Context Learning (ICL) is highly sensitive to the selected demonstrations. Existing approaches to demonstration selection optimize different objectives, yielding inconsistent results. To address this, we propose a unified metric--affinity and diversity--that leverages ICL model's internal representations. Our experiments show that both affinity and diversity strongly correlate with test accuracies, indicating their effectiveness for demonstration selection. Moreover, we show that our proposed metrics align well with various previous works to unify the inconsistency. 

**Abstract (ZH)**: "context-free学习（ICL）的表现高度依赖于选定的演示。现有的一些演示选择方法优化不同的目标，导致结果不一致。为了解决这个问题，我们提出了一种统一的度量标准——亲和力和多样性，该标准利用了ICL模型的内部表示。我们的实验表明，亲和力和多样性都与测试准确性有很强的相关性，这表明它们在演示选择中是有效的。此外，我们展示我们的提出度量标准与各种之前的工作很好地对齐，以统一不一致性。" 

---
# Discovering highly efficient low-weight quantum error-correcting codes with reinforcement learning 

**Title (ZH)**: 使用强化学习发现高效低重量量子错误纠正码 

**Authors**: Austin Yubo He, Zi-Wen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14372)  

**Abstract**: The realization of scalable fault-tolerant quantum computing is expected to hinge on quantum error-correcting codes. In the quest for more efficient quantum fault tolerance, a critical code parameter is the weight of measurements that extract information about errors to enable error correction: as higher measurement weights require higher implementation costs and introduce more errors, it is important in code design to optimize measurement weight. This underlies the surging interest in quantum low-density parity-check (qLDPC) codes, the study of which has primarily focused on the asymptotic (large-code-limit) properties. In this work, we introduce a versatile and computationally efficient approach to stabilizer code weight reduction based on reinforcement learning (RL), which produces new low-weight codes that substantially outperform the state of the art in practically relevant parameter regimes, extending significantly beyond previously accessible small distances. For example, our approach demonstrates savings in physical qubit overhead compared to existing results by 1 to 2 orders of magnitude for weight 6 codes and brings the overhead into a feasible range for near-future experiments. We also investigate the interplay between code parameters using our RL framework, offering new insights into the potential efficiency and power of practically viable coding strategies. Overall, our results demonstrate how RL can effectively advance the crucial yet challenging problem of quantum code discovery and thereby facilitate a faster path to the practical implementation of fault-tolerant quantum technologies. 

**Abstract (ZH)**: 可扩展的容错量子计算的实现有望依赖于量子纠错码。在寻求更高效的量子容错方案中，一个关键的码参数是提取错误信息所需测量的权重：随着测量权重的提高，实现成本也随之增加，引入的错误也更多。因此，在码的设计中优化测量权重至关重要。这催生了对量子低密度奇偶校验码（qLDPC码）的浓厚研究兴趣，目前的研究主要集中在其渐近性质（大码极限）。在这项工作中，我们提出了一种基于强化学习（RL）的灵活且计算高效的纠缠标识符代码权重减少方法，这种方法生成的新低权重码在实际相关参数范围内显著优于现有最先进的结果，并且远超先前可访问的小距离。例如，我们的方法在权重为6的代码中，相比于现有结果，物理量子比特的开销减少了一个到两个数量级，并将开销带入了近未来实验可行的范围。我们还利用我们的RL框架研究了代码参数之间的相互作用，提供了关于实际可行编码策略潜在效率和威力的新见解。总体而言，我们的结果展示了RL如何有效地推进量子码发现这一关键且具有挑战性的问题，从而加速了容错量子技术实用实现的步伐。 

---
# Entropy-UID: A Method for Optimizing Information Density 

**Title (ZH)**: 熵-UID：一种优化信息密度的方法 

**Authors**: Xinpeng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2502.14366)  

**Abstract**: Balanced and efficient information flow is essential for optimizing language generation models. In this work, we propose Entropy-UID, a new token selection method that balances entropy and Uniform Information Density (UID) principles for enhanced efficiency of text generation. Our approach adaptively adjusts token selection by jointly minimizing entropy and surprisal, promoting more even information distribution across generated sequences. Theoretical validation demonstrates that Entropy-UID optimally reduces information spikes while maintaining fluency and coherence. The method has been evulated using information-theoretic metrics on multiple benchmark datasets, including WikiText-2, OpenWebText, and WMT. Experimental results show that Entropy-UID achieves lower surprisal and entropy variance compared to standard GPT-2 and alternative heuristics, leading to more balanced and human-like text generation. Our findings point towards the potential of leveraging information-theoretic constraints to refine token selection strategies in autoregressive language models. 

**Abstract (ZH)**: 平衡而高效的信道信息流动对于优化语言生成模型至关重要。在此研究中，我们提出了一种新的标记选择方法——Entropy-UID，该方法结合熵和均匀信息密度（UID）原则，以提高文本生成的效率。我们的方法通过联合最小化熵和意外性，自适应地调整标记选择，从而促进生成序列中信息分布更加均衡。理论验证表明，Entropy-UID 最优地减少了信息峰值，同时保持了流畅性和连贯性。该方法在多个基准数据集（包括 WikiText-2、OpenWebText 和 WMT）上使用信息论度量进行了评估。实验结果表明，与标准 GPT-2 和其他替代启发式方法相比，Entropy-UID 实现了更低的意外性和熵变异性，从而生成了更加均衡和类人化的文本。我们的研究结果指出了利用信息论约束来细化自回归语言模型中的标记选择策略的潜力。 

---
# Is Q-learning an Ill-posed Problem? 

**Title (ZH)**: 《Q-learning是一个病态问题吗？》 

**Authors**: Philipp Wissmann, Daniel Hein, Steffen Udluft, Thomas Runkler  

**Link**: [PDF](https://arxiv.org/pdf/2502.14365)  

**Abstract**: This paper investigates the instability of Q-learning in continuous environments, a challenge frequently encountered by practitioners. Traditionally, this instability is attributed to bootstrapping and regression model errors. Using a representative reinforcement learning benchmark, we systematically examine the effects of bootstrapping and model inaccuracies by incrementally eliminating these potential error sources. Our findings reveal that even in relatively simple benchmarks, the fundamental task of Q-learning - iteratively learning a Q-function from policy-specific target values - can be inherently ill-posed and prone to failure. These insights cast doubt on the reliability of Q-learning as a universal solution for reinforcement learning problems. 

**Abstract (ZH)**: 本文探讨了在连续环境中Q-learning的不稳定性，这是实践者们经常遇到的一个挑战。传统上，这种不稳定性被认为是由于增强学习中的“自助法”（bootstrapping）和回归模型误差引起的。通过使用一个代表性的强化学习基准测试，我们系统地考察了自助法和模型不准确性对Q-learning效果的影响。我们逐步消除了这些潜在的误差来源，发现即使在相对简单的基准中，Q-learning的基本任务——从与策略相关的目标值中迭代学习Q函数——也可能本质上是病态的，并且容易失败。这些发现对Q-learning作为强化学习问题通用解决方案的可靠性提出了质疑。 

---
# Purest Quantum State Identification 

**Title (ZH)**: 最纯量子态的识别 

**Authors**: Yingqi Yu, Honglin Chen, Jun Wu, Wei Xie, Xiangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.14334)  

**Abstract**: Precise identification of quantum states under noise constraints is essential for quantum information processing. In this study, we generalize the classical best arm identification problem to quantum domains, designing methods for identifying the purest one within $K$ unknown $n$-qubit quantum states using $N$ samples. %, with direct applications in quantum computation and quantum communication. We propose two distinct algorithms: (1) an algorithm employing incoherent measurements, achieving error $\exp\left(- \Omega\left(\frac{N H_1}{\log(K) 2^n }\right) \right)$, and (2) an algorithm utilizing coherent measurements, achieving error $\exp\left(- \Omega\left(\frac{N H_2}{\log(K) }\right) \right)$, highlighting the power of quantum memory. Furthermore, we establish a lower bound by proving that all strategies with fixed two-outcome incoherent POVM must suffer error probability exceeding $ \exp\left( - O\left(\frac{NH_1}{2^n}\right)\right)$. This framework provides concrete design principles for overcoming sampling bottlenecks in quantum technologies. 

**Abstract (ZH)**: 在量子信息处理中，精确识别受噪声限制的量子态至关重要。本研究将经典的最佳臂识别问题推广至量子领域，设计了在给定$N$次样本的情况下从未知的$K$个$n$量子比特态中识别出最纯态的方法，这直接应用于量子计算和量子通信。我们提出了两种不同的算法：（1）一种使用非相干测量的算法，其误差为$\exp\left(- \Omega\left(\frac{N H_1}{\log(K) 2^n }\right) \right)$；（2）一种利用相干测量的算法，其误差为$\exp\left(- \Omega\left(\frac{N H_2}{\log(K) }\right) \right)$，突显了量子记忆的强大能力。此外，我们还通过证明所有具有固定两个结果的非相干POVM策略的误差概率必须超过$\exp\left( - O\left(\frac{NH_1}{2^n}\right)\right)$，建立了下界。这种框架为克服量子技术中的采样瓶颈提供了具体的设设计原则。 

---
# A Survey on Feedback-based Multi-step Reasoning for Large Language Models on Mathematics 

**Title (ZH)**: 基于反馈的多步推理综述：大规模语言模型在数学中的应用 

**Authors**: Ting-Ruen Wei, Haowei Liu, Xuyang Wu, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14333)  

**Abstract**: Recent progress in large language models (LLM) found chain-of-thought prompting strategies to improve the reasoning ability of LLMs by encouraging problem solving through multiple steps. Therefore, subsequent research aimed to integrate the multi-step reasoning process into the LLM itself through process rewards as feedback and achieved improvements over prompting strategies. Due to the cost of step-level annotation, some turn to outcome rewards as feedback. Aside from these training-based approaches, training-free techniques leverage frozen LLMs or external tools for feedback at each step to enhance the reasoning process. With the abundance of work in mathematics due to its logical nature, we present a survey of strategies utilizing feedback at the step and outcome levels to enhance multi-step math reasoning for LLMs. As multi-step reasoning emerges a crucial component in scaling LLMs, we hope to establish its foundation for easier understanding and empower further research. 

**Abstract (ZH)**: 近年来，在大型语言模型（LLM）领域取得了进展，发现通过链式思考提示策略可以提升LLM的推理能力，具体做法是通过多步推理鼓励问题解决。因此，后续研究致力于将多步推理过程集成到LLM本身中，并通过过程奖励作为反馈来改进提示策略。由于步骤级标注成本较高，一些研究转向使用结果奖励作为反馈。除了基于训练的方法之外，还有一些无需训练的技术利用冻结的LLM或外部工具在每个步骤提供反馈以增强推理过程。鉴于数学因其逻辑性质而工作量丰富，我们对利用步骤级和结果级反馈以增强LLM的多步数学推理策略进行了综述。鉴于多步推理在扩展LLM中的重要性，我们希望为更易于理解并促进进一步研究奠定基础。 

---
# Line Goes Up? Inherent Limitations of Benchmarks for Evaluating Large Language Models 

**Title (ZH)**: 《线会增长吗？评估大型语言模型基准的固有限制》

这个标题翻译成中文符合学术规范，保留了原文的核心意义。如果是论文的内容摘要或引言部分，可以适当地扩展为一段话，例如：

《评估大型语言模型时基准的固有限制研究——线性增长真的会出现吗？》

这样的扩展不仅保留了原文的意思，还更加符合中文的表达习惯。 

**Authors**: James Fodor  

**Link**: [PDF](https://arxiv.org/pdf/2502.14318)  

**Abstract**: Large language models (LLMs) regularly demonstrate new and impressive performance on a wide range of language, knowledge, and reasoning benchmarks. Such rapid progress has led many commentators to argue that LLM general cognitive capabilities have likewise rapidly improved, with the implication that such models are becoming progressively more capable on various real-world tasks. Here I summarise theoretical and empirical considerations to challenge this narrative. I argue that inherent limitations with the benchmarking paradigm, along with specific limitations of existing benchmarks, render benchmark performance highly unsuitable as a metric for generalisable competence over cognitive tasks. I also contend that alternative methods for assessing LLM capabilities, including adversarial stimuli and interpretability techniques, have shown that LLMs do not have robust competence in many language and reasoning tasks, and often fail to learn representations which facilitate generalisable inferences. I conclude that benchmark performance should not be used as a reliable indicator of general LLM cognitive capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）在广泛的语言、知识和推理基准测试中经常展现出新的和令人印象深刻的性能。这种快速的进步使许多评论家认为这些模型的通用认知能力也得到了快速提升，进而推断出这些模型在各种实际任务中逐渐变得更强大。在此，我总结了挑战这一观点的理论和实证考量。我认为， benchmark测验范式的固有限制以及现有基准的具体限制使得benchmark性能高度不适合用作认知任务中可泛化的胜任能力的评价指标。同时，我还认为，评估LLM能力的替代方法，包括对抗性刺激和可解释性技术，表明LLMs在许多语言和推理任务中并没有牢固的胜任能力，往往未能学习到促进泛化推断的表示。因此，我得出结论认为，benchmark性能不应被用作可靠衡量LLM一般认知能力的指标。 

---
# Textured 3D Regenerative Morphing with 3D Diffusion Prior 

**Title (ZH)**: 带有三维扩散先验的纹理化3D再生变形 

**Authors**: Songlin Yang, Yushi Lan, Honghua Chen, Xingang Pan  

**Link**: [PDF](https://arxiv.org/pdf/2502.14316)  

**Abstract**: Textured 3D morphing creates smooth and plausible interpolation sequences between two 3D objects, focusing on transitions in both shape and texture. This is important for creative applications like visual effects in filmmaking. Previous methods rely on establishing point-to-point correspondences and determining smooth deformation trajectories, which inherently restrict them to shape-only morphing on untextured, topologically aligned datasets. This restriction leads to labor-intensive preprocessing and poor generalization. To overcome these challenges, we propose a method for 3D regenerative morphing using a 3D diffusion prior. Unlike previous methods that depend on explicit correspondences and deformations, our method eliminates the additional need for obtaining correspondence and uses the 3D diffusion prior to generate morphing. Specifically, we introduce a 3D diffusion model and interpolate the source and target information at three levels: initial noise, model parameters, and condition features. We then explore an Attention Fusion strategy to generate more smooth morphing sequences. To further improve the plausibility of semantic interpolation and the generated 3D surfaces, we propose two strategies: (a) Token Reordering, where we match approximate tokens based on semantic analysis to guide implicit correspondences in the denoising process of the diffusion model, and (b) Low-Frequency Enhancement, where we enhance low-frequency signals in the tokens to improve the quality of generated surfaces. Experimental results show that our method achieves superior smoothness and plausibility in 3D morphing across diverse cross-category object pairs, offering a novel regenerative method for 3D morphing with textured representations. 

**Abstract (ZH)**: 纹理化的3D形变可以生成两个3D对象之间平滑且合理的过渡序列，这一过程不仅涉及形状的转变，还包括纹理的变化。这对于电影制作等创意应用至关重要。以往的方法依赖于建立点对点的对应关系并确定平滑的变形轨迹，这在本质上限制了它们只能用于无纹理且拓扑对齐的数据集的形状仅变。这种限制导致了劳动密集型的预处理步骤和较差的泛化能力。为克服这些挑战，我们提出了一种基于3D扩散先验的3D再生形变方法。不同于依赖显式对应关系和变形的方法，我们的方法消除了获得对应关系的额外需求，并利用3D扩散先验生成形变。具体来说，我们引入了一个3D扩散模型，并在三个层次上（初始噪声、模型参数和条件特征）进行源信息和目标信息的插值。我们还探索了一种注意力融合策略，以生成更平滑的过渡序列。为进一步提高语义插值及生成的3D表面的合理性，我们提出了两种策略：（a）标记重新排序，通过基于语义分析匹配近似标记，指导扩散模型去噪过程中的隐式对应关系；（b）低频增强，通过增强标记中的低频信号，提高生成表面的质量。实验结果表明，我们的方法在多种跨类别对象对的3D形变中实现了更优的平滑性和合理性，并提供了一种基于纹理表示的新型再生3D形变方法。 

---
# MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models 

**Title (ZH)**: MedHallu：检测大型语言模型中医疗幻觉的综合基准 

**Authors**: Shrey Pandit, Jiawei Xu, Junyuan Hong, Zhangyang Wang, Tianlong Chen, Kaidi Xu, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2502.14302)  

**Abstract**: Advancements in Large Language Models (LLMs) and their increasing use in medical question-answering necessitate rigorous evaluation of their reliability. A critical challenge lies in hallucination, where models generate plausible yet factually incorrect outputs. In the medical domain, this poses serious risks to patient safety and clinical decision-making. To address this, we introduce MedHallu, the first benchmark specifically designed for medical hallucination detection. MedHallu comprises 10,000 high-quality question-answer pairs derived from PubMedQA, with hallucinated answers systematically generated through a controlled pipeline. Our experiments show that state-of-the-art LLMs, including GPT-4o, Llama-3.1, and the medically fine-tuned UltraMedical, struggle with this binary hallucination detection task, with the best model achieving an F1 score as low as 0.625 for detecting "hard" category hallucinations. Using bidirectional entailment clustering, we show that harder-to-detect hallucinations are semantically closer to ground truth. Through experiments, we also show incorporating domain-specific knowledge and introducing a "not sure" category as one of the answer categories improves the precision and F1 scores by up to 38% relative to baselines. 

**Abstract (ZH)**: 大型语言模型（LLMs）的进展及其在医疗问答中的越来越多应用，要求对其可靠性进行严格的评估。其中一项关键挑战是模型生成可信但事实错误的输出，即幻觉现象。在医疗领域，这一现象对患者的医疗安全和临床决策构成了严重风险。为解决这一问题，我们引入了MedHallu，这是首个专门用于医疗幻觉检测的标准数据集。MedHallu包含来自PubMedQA的10,000个高质量的问题-答案对，通过受控的生成管道系统地生成了带有幻觉的答案。

我们的实验结果显示，当前最先进的LLMs，包括GPT-4o、Llama-3.1以及经过医疗微调的UltraMedical，在二元幻觉检测任务中表现不佳，最佳模型检测“硬”类幻觉时的F1分数低至0.625。通过双向蕴含聚类分析，我们发现难以检测的幻觉在语义上与真实值更为接近。通过实验，我们还展示了通过引入领域特定知识并将“不确定”类别作为答案类别之一，相对基线指标可以显著提高精准率和F1分数，最多可提高38%。 

---
# SEA-HELM: Southeast Asian Holistic Evaluation of Language Models 

**Title (ZH)**: SEA-HELM: 东南亚综合语言模型评估 

**Authors**: Yosephine Susanto, Adithya Venkatadri Hulagadri, Jann Railey Montalan, Jian Gang Ngui, Xian Bin Yong, Weiqi Leong, Hamsawardhini Rengarajan, Peerat Limkonchotiwat, Yifan Mai, William Chandra Tjhi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14301)  

**Abstract**: With the rapid emergence of novel capabilities in Large Language Models (LLMs), the need for rigorous multilingual and multicultural benchmarks that are integrated has become more pronounced. Though existing LLM benchmarks are capable of evaluating specific capabilities of LLMs in English as well as in various mid- to low-resource languages, including those in the Southeast Asian (SEA) region, a comprehensive and authentic evaluation suite for the SEA languages has not been developed thus far. Here, we present SEA-HELM, a holistic linguistic and cultural LLM evaluation suite that emphasizes SEA languages, comprising five core pillars: (1) NLP Classics, (2) LLM-specifics, (3) SEA Linguistics, (4) SEA Culture, (5) Safety. SEA-HELM currently supports Filipino, Indonesian, Tamil, Thai, and Vietnamese. We also introduce the SEA-HELM leaderboard, which allows users to understand models' multilingual and multicultural performance in a systematic and user-friendly manner. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）新型能力的快速涌现，对于综合性的多语言和跨文化基准测试的需求也愈加迫切。尽管现有的LLM基准测试能够评估LLM在英语以及各种中低资源语言中的特定能力，包括东南亚（SEA）地区的语言，但迄今为止尚未开发出全面和真实的针对SEA语言的评估套件。在此，我们介绍SEA-HELM，这是一种综合性的语言和文化评估套件，重点评估SEA语言，包含五个核心支柱：（1）NLP经典问题，（2）LLM特定能力，（3）SEA语言学，（4）SEA文化，（5）安全。SEA-HELM目前支持菲律宾语、印尼语、泰米尔语、泰语和越南语。此外，我们还介绍了SEA-HELM排行榜，用户可以通过系统且用户友好的方式了解模型在多语言和跨文化方面的表现。 

---
# An Evaluation of Sakana's AI Scientist for Autonomous Research: Wishful Thinking or an Emerging Reality Towards 'Artificial General Research Intelligence' (AGRI)? 

**Title (ZH)**: 对Sakana的AI科学家进行自主研究评估：是理想憧憬还是“通用人工智能研究智能”（AGRI）的发展现实？ 

**Authors**: Joeran Beel, Min-Yen Kan, Moritz Baumgart  

**Link**: [PDF](https://arxiv.org/pdf/2502.14297)  

**Abstract**: A major step toward Artificial General Intelligence (AGI) and Super Intelligence is AI's ability to autonomously conduct research - what we term Artificial General Research Intelligence (AGRI). If machines could generate hypotheses, conduct experiments, and write research papers without human intervention, it would transform science. Recently, this http URL introduced the AI Scientist, a system claiming to automate the research lifecycle, generating both excitement and skepticism.
We evaluated the AI Scientist and found it a milestone in AI-driven research. While it streamlines some aspects, it falls short of expectations. Literature reviews are weak, nearly half the experiments failed, and manuscripts sometimes contain hallucinated results. Most notably, users must provide an experimental pipeline, limiting the AI Scientist's autonomy in research design and execution.
Despite its limitations, the AI Scientist advances research automation. Many reviewers or instructors who assess work superficially may not recognize its output as AI-generated. The system produces research papers with minimal human effort and low cost. Our analysis suggests a paper costs a few USD with a few hours of human involvement, making it significantly faster than human researchers. Compared to AI capabilities from a few years ago, this marks progress toward AGRI.
The rise of AI-driven research systems requires urgent discussion within Information Retrieval (IR) and broader scientific communities. Enhancing literature retrieval, citation validation, and evaluation benchmarks could improve AI-generated research reliability. We propose concrete steps, including AGRI-specific benchmarks, refined peer review, and standardized attribution frameworks. Whether AGRI becomes a stepping stone to AGI depends on how the academic and AI communities shape its development. 

**Abstract (ZH)**: 向着通用人工智能（AGI）和超级智能迈出的重要一步是人工智能能够自主进行研究——我们称之为广义研究智能（AGRI）。如果机器能够在没有人类干预的情况下生成假设、进行实验并撰写研究论文，这将彻底改变科学。最近，[原链接]介绍了AI科学家，这是一个声称能够自动化研究生命周期、生成研究成果的系统，引发了人们的兴奋和质疑。

我们评估了AI科学家，并发现它是人工智能驱动研究的一个里程碑。尽管该系统在某些方面简化了流程，但仍未能达到预期。文献综述较弱，大约有一半的实验失败，而文稿中有时包含虚构的结果。最显著的是，用户必须提供实验管道，限制了AI科学家在研究设计和执行中的自主性。

尽管存在限制，AI科学家仍然推动了研究自动化。许多仅从表面上评估工作的审稿人或教师可能不会意识到其产出是由AI生成的。该系统可以使用少量的人力投入和低成本生成研究论文。我们的分析表明，一篇论文的成本只需几美元，并且通过几小时的人力投入即可完成，这比人类研究人员快得多。相比几年前的AI能力，这是朝着AGRI迈进的进步。

AI驱动研究系统的发展需要信息检索（IR）领域及其更广泛的科学界尽快展开讨论。改进文献检索、引用验证和评估基准可以提高AI生成研究的可靠性。我们提出了具体的步骤，包括针对AGRI的具体基准、细化同行评议和标准化归属框架。AGRI是否能够成为AGI的一个里程碑，取决于学术界和AI社区如何塑造其发展。 

---
# Graph Anomaly Detection via Adaptive Test-time Representation Learning across Out-of-Distribution Domains 

**Title (ZH)**: 跨分布外领域自适应测试时代表征学习的图异常检测 

**Authors**: Delaram Pirhayati, Arlei Silva  

**Link**: [PDF](https://arxiv.org/pdf/2502.14293)  

**Abstract**: Graph Anomaly Detection (GAD) has demonstrated great effectiveness in identifying unusual patterns within graph-structured data. However, while labeled anomalies are often scarce in emerging applications, existing supervised GAD approaches are either ineffective or not applicable when moved across graph domains due to distribution shifts and heterogeneous feature spaces. To address these challenges, we present AdaGraph-T3, a novel test-time training framework for cross-domain GAD. AdaGraph-T3 combines supervised and self-supervised learning during training while adapting to a new domain during test time using only self-supervised learning by leveraging a homophily-based affinity score that captures domain-invariant properties of anomalies. Our framework introduces four key innovations to cross-domain GAD: an effective self-supervision scheme, an attention-based mechanism that dynamically learns edge importance weights during message passing, domain-specific encoders for handling heterogeneous features, and class-aware regularization to address imbalance. Experiments across multiple cross-domain settings demonstrate that AdaGraph-T3 significantly outperforms existing approaches, achieving average improvements of over 6.6% in AUROC and 7.9% in AUPRC compared to the best competing model. 

**Abstract (ZH)**: 图异常检测（Graph Anomaly Detection, GAD）在识别图结构数据中的异常模式方面显示出极大的有效性。然而，随着新兴应用的发展，标记的异常样本往往稀缺，现有的监督GAD方法在跨图域迁移时由于分布偏移和异质特征空间的问题，要么效果不佳，要么不适用。为了解决这些挑战，我们提出了一种新的测试时训练框架AdaGraph-T3，用于跨域GAD。AdaGraph-T3在训练过程中结合了监督学习和半监督学习，而在测试时仅通过利用基于同质性亲和度分值的机制，该机制捕捉了异常的域不变属性，从而在新域中进行适应。我们的框架在跨域GAD中引入了四项关键创新：有效的半监督学习方案、基于注意力的机制，该机制在消息传递过程中动态学习边重要性权重、针对异质特征的域特定编码器，以及意识到类别的正则化以处理类别不平衡。

实验结果表明，AdaGraph-T3在多个跨域设置中显著优于现有方法，在AUROC和AUPRC上分别平均提高了6.6%和7.9%，相比最佳竞争模型表现更佳。 

---
# Correcting Noisy Multilabel Predictions: Modeling Label Noise through Latent Space Shifts 

**Title (ZH)**: 纠正嘈杂的多标签预测：通过潜在空间偏移建模标签噪声 

**Authors**: Weipeng Huang, Qin Li, Yang Xiao, Cheng Qiao, Tie Cai, Junwei Liao, Neil J. Hurley, Guangyuan Piao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14281)  

**Abstract**: Noise in data appears to be inevitable in most real-world machine learning applications and would cause severe overfitting problems. Not only can data features contain noise, but labels are also prone to be noisy due to human input. In this paper, rather than noisy label learning in multiclass classifications, we instead focus on the less explored area of noisy label learning for multilabel classifications. Specifically, we investigate the post-correction of predictions generated from classifiers learned with noisy labels. The reasons are two-fold. Firstly, this approach can directly work with the trained models to save computational resources. Secondly, it could be applied on top of other noisy label correction techniques to achieve further improvements. To handle this problem, we appeal to deep generative approaches that are possible for uncertainty estimation. Our model posits that label noise arises from a stochastic shift in the latent variable, providing a more robust and beneficial means for noisy learning. We develop both unsupervised and semi-supervised learning methods for our model. The extensive empirical study presents solid evidence to that our approach is able to consistently improve the independent models and performs better than a number of existing methods across various noisy label settings. Moreover, a comprehensive empirical analysis of the proposed method is carried out to validate its robustness, including sensitivity analysis and an ablation study, among other elements. 

**Abstract (ZH)**: 在大多数实际机器学习应用中，数据中的噪声似乎是不可避免的，这会导致严重的过拟合问题。数据特征可能会包含噪声，而标签也容易因人为输入而变得不准确。本文不同于多类别分类中的嘈杂标签学习，而是专注于多标签分类中的较少研究的嘈杂标签学习领域。具体而言，我们研究的是基于嘈杂标签训练的分类器生成的预测结果的后处理。这有两方面的理由。首先，这种方法可以直接与训练好的模型结合使用，节省计算资源。其次，它可以与其他嘈杂标签校正技术结合使用，以实现进一步的改进。为了解决这个问题，我们采用了可进行不确定性估计的深度生成方法。我们的模型假定标签噪声来源于潜在变量的随机转移，从而为嘈杂学习提供了一种更加稳健和有益的方法。我们为该模型开发了无监督和半监督学习方法。广泛的实证研究表明，我们的方法能够在各种嘈杂标签设置下一致地改进独立模型，并且在多个现有方法中表现更好。此外，对所提出方法进行了全面的实证分析以验证其鲁棒性，包括敏感性分析和消融研究等元素。 

---
# EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts 

**Title (ZH)**: EpMAN： episodic 记忆注意力机制用于泛化到更长的上下文 

**Authors**: Subhajit Chaudhury, Payel Das, Sarathkrishna Swaminathan, Georgios Kollias, Elliot Nelson, Khushbu Pahwa, Tejaswini Pedapati, Igor Melnyk, Matthew Riemer  

**Link**: [PDF](https://arxiv.org/pdf/2502.14280)  

**Abstract**: Recent advances in Large Language Models (LLMs) have yielded impressive successes on many language tasks. However, efficient processing of long contexts using LLMs remains a significant challenge. We introduce \textbf{EpMAN} -- a method for processing long contexts in an \textit{episodic memory} module while \textit{holistically attending to} semantically relevant context chunks. The output of \textit{episodic attention} is then used to reweigh the decoder's self-attention to the stored KV cache of the context during training and generation. When an LLM decoder is trained using \textbf{EpMAN}, its performance on multiple challenging single-hop long-context recall and question-answering benchmarks is found to be stronger and more robust across the range from 16k to 256k tokens than baseline decoders trained with self-attention, and popular retrieval-augmented generation frameworks. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在许多语言任务上取得了令人印象深刻的成功。然而，高效处理长上下文仍然是一个重大挑战。我们提出了**EpMAN** —— 一种在**情景记忆**模块中处理长上下文的方法，同时**整体关注**语义相关的上下文片段。情景注意的输出随后用于在训练和生成过程中调整解码器的自注意权重，使其指向存储在上下文中的KV缓存。当使用**EpMAN** 训练LLM解码器时，其在多个具有挑战性的单跳长上下文回忆和问答基准测试中的表现相比于仅使用自注意训练和流行检索增强生成框架的基线解码器表现出更强且更加稳健，覆盖了从16,000到256,000个标记的范围。 

---
# STeCa: Step-level Trajectory Calibration for LLM Agent Learning 

**Title (ZH)**: STeCa：LLM代理学习的步骤级轨迹校准 

**Authors**: Hanlin Wang, Jian Wang, Chak Tou Leong, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.14276)  

**Abstract**: Large language model (LLM)-based agents have shown promise in tackling complex tasks by interacting dynamically with the environment. Existing work primarily focuses on behavior cloning from expert demonstrations and preference learning through exploratory trajectory sampling. However, these methods often struggle in long-horizon tasks, where suboptimal actions accumulate step by step, causing agents to deviate from correct task trajectories. To address this, we highlight the importance of timely calibration and the need to automatically construct calibration trajectories for training agents. We propose Step-Level Trajectory Calibration (STeCa), a novel framework for LLM agent learning. Specifically, STeCa identifies suboptimal actions through a step-level reward comparison during exploration. It constructs calibrated trajectories using LLM-driven reflection, enabling agents to learn from improved decision-making processes. These calibrated trajectories, together with successful trajectory data, are utilized for reinforced training. Extensive experiments demonstrate that STeCa significantly outperforms existing methods. Further analysis highlights that step-level calibration enables agents to complete tasks with greater robustness. Our code and data are available at this https URL. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的代理在通过与环境动态交互来应对复杂任务方面展现了潜力。现有工作主要集中在从专家示范行为克隆以及通过探索性轨迹采样学习偏好上。然而，这些方法在长时任务中往往表现不佳，因为次优行为逐步累积，导致代理偏离正确的任务轨迹。为解决这一问题，我们强调了及时校准的重要性，并阐明了自动构建校准轨迹以培训代理的必要性。我们提出了一种名为Step-Level Trajectory Calibration（STeCa）的新颖框架，用于LLM代理学习。具体而言，STeCa通过探索中的步骤级奖励比较识别次优行为，并利用LLM驱动的反思构建校准轨迹，使代理能够从改进的决策过程中学习。这些校准轨迹与成功的轨迹数据一起用于强化训练。广泛的实验表明，STeCa显著优于现有方法。进一步的分析表明，步骤级校准使代理能够以更高的鲁棒性完成任务。我们的代码和数据可在以下链接中获取：this https URL。 

---
# LLM-EvRep: Learning an LLM-Compatible Event Representation Using a Self-Supervised Framework 

**Title (ZH)**: LLM-EvRep：使用自我监督框架学习一种兼容LLM的事件表示 

**Authors**: Zongyou Yu, Qiang Qu, Qian Zhang, Nan Zhang, Xiaoming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.14273)  

**Abstract**: Recent advancements in event-based recognition have demonstrated significant promise, yet most existing approaches rely on extensive training, limiting their adaptability for efficient processing of event-driven visual content. Meanwhile, large language models (LLMs) have exhibited remarkable zero-shot capabilities across diverse domains, but their application to event-based visual recognition remains largely unexplored. To bridge this gap, we propose \textbf{LLM-EvGen}, an event representation generator that produces LLM-compatible event representations \textbf{LLM-EvRep}, thereby enhancing the performance of LLMs on event recognition tasks. The generator is trained using a self-supervised framework, aligning the generated representations with semantic consistency and structural fidelity. Comprehensive experiments were conducted on three datasets: N-ImageNet, N-Caltech101, and N-MNIST. The results demonstrate that our method, \textbf{LLM-EvRep}, outperforms the event-to-video method, E2VID, by 15.93\%, 0.82\%, and 50.21\%, respectively, in recognition tasks when evaluated using GPT-4o. 

**Abstract (ZH)**: 近年来，基于事件的识别研究取得了显著的进展，但大多数现有方法依赖于大量的训练，限制了它们在处理事件驱动的视觉内容时的高效适应性。与此同时，大型语言模型（LLMs）在多个领域展示了惊人的零样本能力，但它们在事件驱动的视觉识别中的应用仍然 largely unexplored（未被充分探索）。为了解决这一问题，我们提出了 \textbf{LLM-EvGen}，一种事件表示生成器，它生成与LLM兼容的事件表示 \textbf{LLM-EvRep}，从而增强LLM在事件识别任务中的性能。生成器使用半监督框架进行训练，生成的表示与语义一致性和结构忠实性保持一致。我们在三个数据集——N-ImageNet、N-Caltech101 和 N-MNIST——上进行了全面的实验。结果表明，在使用GPT-4o评估时，我们的方法 \textbf{LLM-EvRep} 在识别任务中的性能分别优于事件到视频的方法 E2VID 15.93%、0.82% 和 50.21%。 

---
# Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models 

**Title (ZH)**: 捕捉细腻的偏好：面向偏好的蒸馏方法用于小型语言模型 

**Authors**: Yanggan Gu, Junzhuo Li, Sirui Huang, Xin Zou, Zhenghua Li, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14272)  

**Abstract**: Aligning small language models (SLMs) with human values typically involves distilling preference knowledge from large language models (LLMs). However, existing distillation methods model preference knowledge in teacher LLMs by comparing pairwise responses, overlooking the extent of difference between responses. This limitation hinders student SLMs from capturing the nuanced preferences for multiple responses. In this paper, we propose a Preference-Aligned Distillation (PAD) framework, which models teacher's preference knowledge as a probability distribution over all potential preferences, thereby providing more nuanced supervisory signals. Our insight in developing PAD is rooted in the demonstration that language models can serve as reward functions, reflecting their intrinsic preferences. Based on this, PAD comprises three key steps: (1) sampling diverse responses using high-temperature; (2) computing rewards for both teacher and student to construct their intrinsic preference; and (3) training the student's intrinsic preference distribution to align with the teacher's. Experiments on four mainstream alignment benchmarks demonstrate that PAD consistently and significantly outperforms existing approaches, achieving over 20\% improvement on AlpacaEval 2 and Arena-Hard, indicating superior alignment with human preferences. Notably, on MT-Bench, using the \textsc{Gemma} model family, the student trained by PAD surpasses its teacher, further validating the effectiveness of our PAD. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，符合学术规范：

将小型语言模型（SLMs）与人类价值观对齐通常涉及从小型语言模型（LLMs）中提炼偏好知识。然而，现有的提炼方法通过成对比较响应来建模教师LLMs的偏好知识，忽视了响应之间差异的程度。这一限制阻碍了学生SLMs捕捉多种响应的微妙偏好的能力。在本文中，我们提出了一种偏好对齐提炼（Preference-Aligned Distillation, PAD）框架，该框架将教师的偏好知识建模为所有潜在偏好的一种概率分布，从而提供更为细腻的监督信号。我们开发PAD的见解根植于语言模型可以作为奖励函数的证明，反映了它们的内在偏好。基于此，PAD 包含三个关键步骤：（1）使用高温抽样多样化的响应；（2）为教师和学生计算奖励以构建其内在偏好；（3）训练学生内在偏好的分布以与教师对齐。针对四个主流对齐基准的实验显示，PAD 在所有基准上都表现出一致且显著的优越性，特别是在 AlpacaEval 2 和 Arena-Hard 上取得了超过20%的提升，表明其与人类偏好对齐更优。值得注意的是，在MT-Bench 上，使用 GEMMA 模型家族训练的学生超越了其教师，进一步验证了我们 PAD 的有效性。 

---
# MCQA-Eval: Efficient Confidence Evaluation in NLG with Gold-Standard Correctness Labels 

**Title (ZH)**: MCQA-Eval：基于金标准正确性标签的NLG中高效置信度评估 

**Authors**: Xiaoou Liu, Zhen Lin, Longchao Da, Chacha Chen, Shubhendu Trivedi, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.14268)  

**Abstract**: Large Language Models (LLMs) require robust confidence estimation, particularly in critical domains like healthcare and law where unreliable outputs can lead to significant consequences. Despite much recent work in confidence estimation, current evaluation frameworks rely on correctness functions -- various heuristics that are often noisy, expensive, and possibly introduce systematic biases. These methodological weaknesses tend to distort evaluation metrics and thus the comparative ranking of confidence measures. We introduce MCQA-Eval, an evaluation framework for assessing confidence measures in Natural Language Generation (NLG) that eliminates dependence on an explicit correctness function by leveraging gold-standard correctness labels from multiple-choice datasets. MCQA-Eval enables systematic comparison of both internal state-based white-box (e.g. logit-based) and consistency-based black-box confidence measures, providing a unified evaluation methodology across different approaches. Through extensive experiments on multiple LLMs and widely used QA datasets, we report that MCQA-Eval provides efficient and more reliable assessments of confidence estimation methods than existing approaches. 

**Abstract (ZH)**: 大型语言模型（LLMs）需要 robust 的置信度估计，特别是在医疗和法律等关键领域，因为不可靠的输出可能导致重大后果。尽管近期在置信度估计方面做了许多工作，但当前的评估框架仍然依赖于正确性函数——各种噪声较大、成本较高的启发式方法，可能引入系统性偏差。这些方法论上的缺陷往往会扭曲评估指标，进而影响置信度度量的比较排名。我们引入了MCQA-Eval，这是一种用于自然语言生成（NLG）中评估置信度度量的框架，它通过利用来自多项选择数据集的准确正确性标签来消除对外部明确正确性函数的依赖。MCQA-Eval 允许系统比较基于内部状态的白盒置信度度量（例如，logit 基础）和基于一致性的黑盒置信度度量，提供了一种统一的评估方法论，适用于不同的方法。通过在多个 LLM 和广泛使用的问答数据集上进行广泛的实验，我们发现 MCQA-Eval 提供了比现有方法更高效且更可靠的置信度估计方法评估。 

---
# EyeBench: A Call for More Rigorous Evaluation of Retinal Image Enhancement 

**Title (ZH)**: EyeBench: 对视网膜图像增强评价方法更严格评估的呼吁 

**Authors**: Wenhui Zhu, Xuanzhao Dong, Xin Li, Yujian Xiong, Xiwen Chen, Peijie Qiu, Vamsi Krishna Vasa, Zhangsihao Yang, Yi Su, Oana Dumitrascu, Yalin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14260)  

**Abstract**: Over the past decade, generative models have achieved significant success in enhancement fundus this http URL, the evaluation of these models still presents a considerable challenge. A comprehensive evaluation benchmark for fundus image enhancement is indispensable for three main reasons: 1) The existing denoising metrics (e.g., PSNR, SSIM) are hardly to extend to downstream real-world clinical research (e.g., Vessel morphology consistency). 2) There is a lack of comprehensive evaluation for both paired and unpaired enhancement methods, along with the need for expert protocols to accurately assess clinical value. 3) An ideal evaluation system should provide insights to inform future developments of fundus image enhancement. To this end, we propose a novel comprehensive benchmark, EyeBench, to provide insights that align enhancement models with clinical needs, offering a foundation for future work to improve the clinical relevance and applicability of generative models for fundus image enhancement. EyeBench has three appealing properties: 1) multi-dimensional clinical alignment downstream evaluation: In addition to evaluating the enhancement task, we provide several clinically significant downstream tasks for fundus images, including vessel segmentation, DR grading, denoising generalization, and lesion segmentation. 2) Medical expert-guided evaluation design: We introduce a novel dataset that promote comprehensive and fair comparisons between paired and unpaired methods and includes a manual evaluation protocol by medical experts. 3) Valuable insights: Our benchmark study provides a comprehensive and rigorous evaluation of existing methods across different downstream tasks, assisting medical experts in making informed choices. Additionally, we offer further analysis of the challenges faced by existing methods. The code is available at \url{this https URL} 

**Abstract (ZH)**: 在过去的十年里，生成模型在视网膜图像增强方面取得了显著的成果。然而，这些模型的评估仍然面临着显著的挑战。为了三个主要原因需要一个全面的评估基准：1）现有的去噪指标（例如，PSNR、SSIM）很难直接应用于下游的实际临床研究（例如，血管形态一致性）。2）缺乏综合评估配对和非配对增强方法，并且需要专家指导的协议来准确评估临床价值。3）理想的评估系统应为视网膜图像增强的未来开发提供有价值的见解。因此，我们提出了一种名为EyeBench的新型综合基准，旨在将增强模型与临床需求对齐，并为未来的工作提供基础，以提高生成模型在视网膜图像增强中的临床相关性和适用性。EyeBench具有三个吸引人的特性：1）多维度临床对齐的下游评估：除了评估增强任务，我们还提供了多种临床显著的下游任务，包括血管分割、DR分级、去噪泛化和病变分割。2）医学专家指导的评估设计：我们引入了一个新型数据集，促进配对和非配对方法之间的全面和公平比较，并包含医学专家的手动评估协议。3）有价值的见解：我们的基准测试提供了对现有方法在不同下游任务中进行全面和严谨评估的研究，帮助医学专家做出明智的选择。此外，我们还进一步分析了现有方法面临的挑战。相关代码可在以下链接获取：\[this https URL\] 

---
# Does Time Have Its Place? Temporal Heads: Where Language Models Recall Time-specific Information 

**Title (ZH)**: 时间有其位置吗？时间头部：语言模型如何回忆时间特定信息 

**Authors**: Yein Park, Chanwoong Yoon, Jungwoo Park, Minbyul Jeong, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14258)  

**Abstract**: While the ability of language models to elicit facts has been widely investigated, how they handle temporally changing facts remains underexplored. We discover Temporal Heads, specific attention heads primarily responsible for processing temporal knowledge through circuit analysis. We confirm that these heads are present across multiple models, though their specific locations may vary, and their responses differ depending on the type of knowledge and its corresponding years. Disabling these heads degrades the model's ability to recall time-specific knowledge while maintaining its general capabilities without compromising time-invariant and question-answering performances. Moreover, the heads are activated not only numeric conditions ("In 2004") but also textual aliases ("In the year ..."), indicating that they encode a temporal dimension beyond simple numerical representation. Furthermore, we expand the potential of our findings by demonstrating how temporal knowledge can be edited by adjusting the values of these heads. 

**Abstract (ZH)**: 尽管语言模型提取事实的能力已经得到了广泛的研究，但它们处理时间变化的事实仍然较少被探索。我们发现了一种名为“时间头”的特定注意机制，主要负责通过电路分析处理时间知识。我们确认这些头存在于多个模型中，尽管它们的具体位置可能有所不同，且它们的响应会根据知识类型及其相应年份的不同而变化。禁用这些头会降低模型提取特定时间知识的能力，同时保持其一般能力，而不影响时间不变性和问答性能。此外，这些头不仅会被包含具体年份的条件激活（“在2004年”），还会被文本来替换的年份表达（“在……年”）激活，这表明它们编码了超越简单数字表示的时间维度。进一步地，我们展示了通过调整这些头的值来编辑时间知识的潜力。 

---
# Effects of Prompt Length on Domain-specific Tasks for Large Language Models 

**Title (ZH)**: 大型语言模型中特定领域任务中提示长度的影响研究 

**Authors**: Qibang Liu, Wenzhe Wang, Jeffrey Willard  

**Link**: [PDF](https://arxiv.org/pdf/2502.14255)  

**Abstract**: In recent years, Large Language Models have garnered significant attention for their strong performance in various natural language tasks, such as machine translation and question answering. These models demonstrate an impressive ability to generalize across diverse tasks. However, their effectiveness in tackling domain-specific tasks, such as financial sentiment analysis and monetary policy understanding, remains a topic of debate, as these tasks often require specialized knowledge and precise reasoning. To address such challenges, researchers design various prompts to unlock the models' abilities. By carefully crafting input prompts, researchers can guide these models to produce more accurate responses. Consequently, prompt engineering has become a key focus of study. Despite the advancements in both models and prompt engineering, the relationship between the two-specifically, how prompt design impacts models' ability to perform domain-specific tasks-remains underexplored. This paper aims to bridge this research gap. 

**Abstract (ZH)**: 近年来，大型语言模型因其在各种自然语言任务中（如机器翻译和问答）的强大表现而引起了广泛关注。这些模型展示出在多种任务上泛化的惊人能力。然而，它们在处理特定领域任务（如金融情绪分析和货币政策理解）方面的有效性仍是一个有争议的话题，因为这些任务往往需要专门的知识和精确的推理。为解决这些挑战，研究者设计了各种提示（prompts）来激发模型的能力。通过精心设计输入提示，研究者可以引导这些模型产生更准确的回应。因此，提示工程已成为研究的一个重要焦点。尽管在模型和提示工程方面取得了进展，但模型和提示工程之间的关系——尤其是提示设计如何影响模型执行特定领域任务的能力——仍是一个未被充分探索的领域。本文旨在弥合这一研究缺口。 

---
# Mem2Ego: Empowering Vision-Language Models with Global-to-Ego Memory for Long-Horizon Embodied Navigation 

**Title (ZH)**: 将以下论文标题或内容翻译成中文，符合学术规范：

Mem2Ego：通过全局到自车记忆增强的视觉-语言模型在长时 horizon 身份绑定导航中的应用 

**Authors**: Lingfeng Zhang, Yuecheng Liu, Zhanguang Zhang, Matin Aghaei, Yaochen Hu, Hongjian Gu, Mohammad Ali Alomrani, David Gamaliel Arcos Bravo, Raika Karimi, Atia Hamidizadeh, Haoping Xu, Guowei Huang, Zhanpeng Zhang, Tongtong Cao, Weichao Qiu, Xingyue Quan, Jianye Hao, Yuzheng Zhuang, Yingxue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14254)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and Vision-Language Models (VLMs) have made them powerful tools in embodied navigation, enabling agents to leverage commonsense and spatial reasoning for efficient exploration in unfamiliar environments. Existing LLM-based approaches convert global memory, such as semantic or topological maps, into language descriptions to guide navigation. While this improves efficiency and reduces redundant exploration, the loss of geometric information in language-based representations hinders spatial reasoning, especially in intricate environments. To address this, VLM-based approaches directly process ego-centric visual inputs to select optimal directions for exploration. However, relying solely on a first-person perspective makes navigation a partially observed decision-making problem, leading to suboptimal decisions in complex environments. In this paper, we present a novel vision-language model (VLM)-based navigation framework that addresses these challenges by adaptively retrieving task-relevant cues from a global memory module and integrating them with the agent's egocentric observations. By dynamically aligning global contextual information with local perception, our approach enhances spatial reasoning and decision-making in long-horizon tasks. Experimental results demonstrate that the proposed method surpasses previous state-of-the-art approaches in object navigation tasks, providing a more effective and scalable solution for embodied navigation. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）和视觉-语言模型（VLMs）的最新进展使其成为有潜力的工具，能够使代理利用常识和空间推理在未知环境中进行高效探索。现有基于LLM的方法将全局记忆（如语义或拓扑地图）转化为语言描述来指导导航。虽然这种方法提高了效率并减少了重复探索的次数，但基于语言的表示形式损失了几何信息，这在复杂的环境中会阻碍空间推理。为了解决这一问题，基于VLM的方法直接处理以自我为中心的视觉输入，选择探索的最佳方向。然而，仅依赖第一人称视角使导航成为部分观察的决策问题，在复杂环境中容易导致不 optimal 的决策。在本文中，我们提出了一种新颖的基于VLM的导航框架，通过自适应地从全局记忆模块中检索任务相关信息，并与代理的以自我为中心的感知进行整合，以解决这个问题。通过动态对齐全局上下文信息与局部感知，我们的方法增强了在长期任务中的空间推理和决策能力。实验结果表明，提出的方法在物体导航任务中超过了之前的先驱方法，为有代理导航提供了更有效和可扩展的解决方案。 

---
# Pandora3D: A Comprehensive Framework for High-Quality 3D Shape and Texture Generation 

**Title (ZH)**: Pandora3D：一种全面的高品质3D形状与纹理生成框架 

**Authors**: Jiayu Yang, Taizhang Shang, Weixuan Sun, Xibin Song, Ziang Chen, Senbo Wang, Shenzhou Chen, Weizhe Liu, Hongdong Li, Pan Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.14247)  

**Abstract**: This report presents a comprehensive framework for generating high-quality 3D shapes and textures from diverse input prompts, including single images, multi-view images, and text descriptions. The framework consists of 3D shape generation and texture generation. (1). The 3D shape generation pipeline employs a Variational Autoencoder (VAE) to encode implicit 3D geometries into a latent space and a diffusion network to generate latents conditioned on input prompts, with modifications to enhance model capacity. An alternative Artist-Created Mesh (AM) generation approach is also explored, yielding promising results for simpler geometries. (2). Texture generation involves a multi-stage process starting with frontal images generation followed by multi-view images generation, RGB-to-PBR texture conversion, and high-resolution multi-view texture refinement. A consistency scheduler is plugged into every stage, to enforce pixel-wise consistency among multi-view textures during inference, ensuring seamless integration.
The pipeline demonstrates effective handling of diverse input formats, leveraging advanced neural architectures and novel methodologies to produce high-quality 3D content. This report details the system architecture, experimental results, and potential future directions to improve and expand the framework. The source code and pretrained weights are released at: \url{this https URL}. 

**Abstract (ZH)**: 本报告提出了一种全面的框架，用于从多种输入提示（包括单张图像、多视角图像和文本描述）生成高质量的3D形状和纹理。该框架由3D形状生成和纹理生成两大部分组成。（1）3D形状生成流水线采用变分自编码器（VAE）将隐式3D几何结构编码到潜在空间中，并使用扩散网络生成条件依赖于输入提示的潜在变量，同时对模型容量进行了改进。还探索了一种艺术家创建的网格（AM）生成的方法，该方法在简单几何结构方面取得了令人鼓舞的结果。（2）纹理生成涉及一个多阶段过程，包括从正面图像生成开始，随后是多视角图像生成、RGB到PBR纹理转换以及高分辨率多视角纹理细化。在每个阶段插入一致性调度器，以确保推理过程中多视角纹理之间的像素级一致性，从而实现无缝集成。

该流水线展示了对多种输入格式的有效处理能力，利用先进的神经网络架构和新颖的方法来生成高质量的3D内容。本报告详细介绍了系统架构、实验结果以及未来改进和扩展框架的潜在方向。源代码和预训练权重已发布于：\url{此处插入超链接}。 

---
# OG-Gaussian: Occupancy Based Street Gaussians for Autonomous Driving 

**Title (ZH)**: OG-Gaussian：基于占用率的街道高斯模型在自动驾驶中的应用 

**Authors**: Yedong Shen, Xinran Zhang, Yifan Duan, Shiqi Zhang, Heng Li, Yilong Wu, Jianmin Ji, Yanyong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14235)  

**Abstract**: Accurate and realistic 3D scene reconstruction enables the lifelike creation of autonomous driving simulation environments. With advancements in 3D Gaussian Splatting (3DGS), previous studies have applied it to reconstruct complex dynamic driving scenes. These methods typically require expensive LiDAR sensors and pre-annotated datasets of dynamic objects. To address these challenges, we propose OG-Gaussian, a novel approach that replaces LiDAR point clouds with Occupancy Grids (OGs) generated from surround-view camera images using Occupancy Prediction Network (ONet). Our method leverages the semantic information in OGs to separate dynamic vehicles from static street background, converting these grids into two distinct sets of initial point clouds for reconstructing both static and dynamic objects. Additionally, we estimate the trajectories and poses of dynamic objects through a learning-based approach, eliminating the need for complex manual annotations. Experiments on Waymo Open dataset demonstrate that OG-Gaussian is on par with the current state-of-the-art in terms of reconstruction quality and rendering speed, achieving an average PSNR of 35.13 and a rendering speed of 143 FPS, while significantly reducing computational costs and economic overhead. 

**Abstract (ZH)**: 准确且逼真的三维场景重建能够实现自动驾驶仿真环境的生动创建。随着三维高斯点聚集（3DGS）技术的进步，先前的研究已经将该技术应用于复杂动态驾驶场景的重建。这些方法通常需要昂贵的激光雷达（LiDAR）传感器和预标注的动态物体数据集。为了解决这些挑战，我们提出了OG-Gaussian这一新颖的方法，该方法用Occupancy Grids（OGs）替换LiDAR点云，而这些OGs是通过Occupancy Prediction Network（ONet）从环绕视图摄像头图像中生成的。我们的方法利用OGs中的语义信息将动态车辆与静态街道背景分离，将这些网格转换为重建静态和动态物体时的两套初始点云。此外，我们通过基于学习的方法估计动态物体的轨迹和姿态，从而消除了复杂的手动标注的需要。在Waymo Open数据集上的实验表明，OG-Gaussian在重建质量和渲染速度方面与当前最先进的技术相当，平均PSNR为35.13，渲染速度达到143 FPS，同时显著降低了计算成本和经济开销。 

---
# SleepGMUformer: A gated multimodal temporal neural network for sleep staging 

**Title (ZH)**: 睡眠GMUformer：一种门控多模态时间神经网络方法用于睡眠分期 

**Authors**: Chenjun Zhao, Xuesen Niu, Xinglin Yu, Long Chen, Na Lv, Huiyu Zhou, Aite Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14227)  

**Abstract**: Sleep staging is a key method for assessing sleep quality and diagnosing sleep disorders. However, current deep learning methods face challenges: 1) postfusion techniques ignore the varying contributions of different modalities; 2) unprocessed sleep data can interfere with frequency-domain information. To tackle these issues, this paper proposes a gated multimodal temporal neural network for multidomain sleep data, including heart rate, motion, steps, EEG (Fpz-Cz, Pz-Oz), and EOG from WristHR-Motion-Sleep and SleepEDF-78. The model integrates: 1) a pre-processing module for feature alignment, missing value handling, and EEG de-trending; 2) a feature extraction module for complex sleep features in the time dimension; and 3) a dynamic fusion module for real-time modality this http URL show classification accuracies of 85.03% on SleepEDF-78 and 94.54% on WristHR-Motion-Sleep datasets. The model handles heterogeneous datasets and outperforms state-of-the-art models by 1.00%-4.00%. 

**Abstract (ZH)**: 睡眠分期是评估睡眠质量和诊断睡眠障碍的关键方法。然而，当前的深度学习方法面临着一些挑战：1）后融合技术忽视了不同模态的不同贡献比例；2）未经处理的睡眠数据会影响频域信息。为了解决这些问题，本文提出了一种门控多模态时序神经网络，用于处理包括心率、运动、步数、EEG（Fpz-Cz，Pz-Oz）和EOG在内的多领域睡眠数据，数据来源为WristHR-Motion-Sleep和SleepEDF-78。该模型整合了：1）一个预处理模块用于特征对齐、缺失值处理和EEG消趋势；2）一个特征提取模块用于时间维度上的复杂睡眠特征；以及3）一个动态融合模块用于实时模态融合。实验结果显示，该模型在SleepEDF-78数据集上的分类准确率为85.03%，在WristHR-Motion-Sleep数据集上的分类准确率为94.54%。该模型能够处理异质性数据集，并在多项基准模型上表现出1.00%-4.00%的性能提升。 

---
# Enhancing Pavement Sensor Data Acquisition for AI-Driven Transportation Research 

**Title (ZH)**: 基于AI驱动的交通研究改进路面传感器数据采集 

**Authors**: Manish Kumar Krishne Gowda, Andrew Balmos, Shin Boonam, James V. Krogmeier  

**Link**: [PDF](https://arxiv.org/pdf/2502.14222)  

**Abstract**: Effective strategies for sensor data management are essential for advancing transportation research, especially in the current data-driven era, due to the advent of novel applications in artificial intelligence. This paper presents comprehensive guidelines for managing transportation sensor data, encompassing both archived static data and real-time data streams. The real-time system architecture integrates various applications with data acquisition systems (DAQ). By deploying the in-house designed, open-source Avena software platform alongside the NATS messaging system as a secure communication broker, reliable data exchange is ensured. While robust databases like TimescaleDB facilitate organized storage, visualization platforms like Grafana provide real-time monitoring capabilities.
In contrast, static data standards address the challenges in handling unstructured, voluminous datasets. The standards advocate for a combination of cost-effective bulk cloud storage for unprocessed sensor data and relational databases for recording summarized analyses. They highlight the role of cloud data transfer tools like FME for efficient migration of sensor data from local storages onto the cloud. Further, integration of robust visualization tools into the framework helps in deriving patterns and trends from these complex datasets.
The proposals were applied to INDOT's real-world case studies involving the I-65 and I-69 Greenfield districts. For real-time data collection, Campbell Scientific DAQ systems were used, enabling continuous generation and monitoring of sensor metrics. In the case of the archived I-69 database, summary data was compiled in Oracle, while the unprocessed data was stored in SharePoint. The results underline the effectiveness of the proposed guidelines and motivate their adoption in research projects. 

**Abstract (ZH)**: 有效的传感器数据管理策略对于推动交通运输研究至关重要，尤其是在当前以数据驱动的时代，由于人工智能领域出现了新的应用。本文提出了全面的交通运输传感器数据管理指南，涵盖了归档的静态数据和实时数据流。实时系统架构将各种应用程序与数据采集系统（DAQ）集成在一起。通过部署自主研发的开源Avena软件平台，并结合NATS消息系统作为安全通信代理，确保了可靠的数据交换。坚固的数据库如TimescaleDB便于有序存储，而视觉化平台如Grafana则提供了实时监控能力。

相比之下，静态数据标准应对了处理非结构化、大数据集的挑战。标准建议结合使用成本效益高的云存储来存储未处理的传感器数据，并使用关系型数据库来记录总结分析。它们强调了使用云数据传输工具（如FME）高效地将传感器数据从本地存储迁移到云端的重要性。此外，将坚固的可视化工具集成到框架中有助于从这些复杂数据集中提取模式和趋势。

这些提议应用到了INDOT的实地案例研究中，涉及I-65和I-69 Greenfield地区。对于实时数据采集，使用了Campbell Scientific DAQ系统，能够持续生成和监控传感器指标。对于归档的I-69数据库，在Oracle中编译了汇总数据，而未处理的数据则存储在SharePoint中。结果证明了所提指南的有效性，并激发了在研究项目中的应用。 

---
# Rethinking Spiking Neural Networks from an Ensemble Learning Perspective 

**Title (ZH)**: 从集成学习视角重新审视脉冲神经网络 

**Authors**: Yongqi Ding, Lin Zuo, Mengmeng Jing, Pei He, Hanpu Deng  

**Link**: [PDF](https://arxiv.org/pdf/2502.14218)  

**Abstract**: Spiking neural networks (SNNs) exhibit superior energy efficiency but suffer from limited performance. In this paper, we consider SNNs as ensembles of temporal subnetworks that share architectures and weights, and highlight a crucial issue that affects their performance: excessive differences in initial states (neuronal membrane potentials) across timesteps lead to unstable subnetwork outputs, resulting in degraded performance. To mitigate this, we promote the consistency of the initial membrane potential distribution and output through membrane potential smoothing and temporally adjacent subnetwork guidance, respectively, to improve overall stability and performance. Moreover, membrane potential smoothing facilitates forward propagation of information and backward propagation of gradients, mitigating the notorious temporal gradient vanishing problem. Our method requires only minimal modification of the spiking neurons without adapting the network structure, making our method generalizable and showing consistent performance gains in 1D speech, 2D object, and 3D point cloud recognition tasks. In particular, on the challenging CIFAR10-DVS dataset, we achieved 83.20\% accuracy with only four timesteps. This provides valuable insights into unleashing the potential of SNNs. 

**Abstract (ZH)**: 脉冲神经网络（SNNs）在能效方面表现出色，但在性能上存在局限。本文中，我们将SNNs视为具有相同架构和权重的时间子网络的集合，并强调影响其性能的一个关键问题：时间步骤之间神经膜电位的极大差异导致子网络输出不稳定，从而影响了整体性能。为缓解这一问题，我们通过膜电位平滑来提高初始膜电位分布的一致性，通过时间相邻子网络的指导来提高输出的一致性，从而改善整体稳定性和性能。此外，膜电位平滑有助于信息的向前传播和梯度的反向传播，缓解了著名的时序梯度消失问题。我们的方法只需要对脉冲神经元进行少量修改，无需调整网络结构，这使得我们的方法具有通用性，并在1D语音、2D物体和3D点云识别任务中展现出一致的性能提升。特别是在具有挑战性的CIFAR10-DVS数据集中，我们仅使用四个时间步骤实现了83.20%的准确率。这为我们释放SNNs的潜力提供了宝贵的见解。 

---
# Towards Secure Program Partitioning for Smart Contracts with LLM's In-Context Learning 

**Title (ZH)**: 面向基于大型语言模型即时上下文学习的智能合约安全划分研究 

**Authors**: Ye Liu, Yuqing Niu, Chengyan Ma, Ruidong Han, Wei Ma, Yi Li, Debin Gao, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2502.14215)  

**Abstract**: Smart contracts are highly susceptible to manipulation attacks due to the leakage of sensitive information. Addressing manipulation vulnerabilities is particularly challenging because they stem from inherent data confidentiality issues rather than straightforward implementation bugs. To tackle this by preventing sensitive information leakage, we present PartitionGPT, the first LLM-driven approach that combines static analysis with the in-context learning capabilities of large language models (LLMs) to partition smart contracts into privileged and normal codebases, guided by a few annotated sensitive data variables. We evaluated PartitionGPT on 18 annotated smart contracts containing 99 sensitive functions. The results demonstrate that PartitionGPT successfully generates compilable, and verified partitions for 78% of the sensitive functions while reducing approximately 30% code compared to function-level partitioning approach. Furthermore, we evaluated PartitionGPT on nine real-world manipulation attacks that lead to a total loss of 25 million dollars, PartitionGPT effectively prevents eight cases, highlighting its potential for broad applicability and the necessity for secure program partitioning during smart contract development to diminish manipulation vulnerabilities. 

**Abstract (ZH)**: 智能合约由于敏感信息泄露，极易受到操纵攻击。解决操纵漏洞特别是在合约中固有的数据保密问题而非简单的实施错误时，具有特别的挑战性。为了解决这个问题，我们提出了PartitionGPT，这是第一个结合静态分析和大型语言模型（LLM）的上下文学习能力的LLM驱动方法，用于根据少量注释的敏感数据变量将智能合约划分为特权代码库和普通代码库。我们对包含99个敏感函数的18个注释智能合约进行了评估。结果显示，PartitionGPT成功为78%的敏感函数生成了可编译且经过验证的划分，并且相比基于函数级别的划分方法，代码量减少了约30%。此外，我们还对导致总共2500万美元损失的九个实际操纵攻击进行了评估，PartitionGPT成功防止了八个案例，这表明它具有广泛的应用潜力，并强调了在智能合约开发过程中进行安全程序划分以减少操纵漏洞的重要性。 

---
# Accurate Forgetting for Heterogeneous Federated Continual Learning 

**Title (ZH)**: 异质联邦连续学习中的精确遗忘 

**Authors**: Abudukelimu Wuerkaixi, Sen Cui, Jingfeng Zhang, Kunda Yan, Bo Han, Gang Niu, Lei Fang, Changshui Zhang, Masashi Sugiyama  

**Link**: [PDF](https://arxiv.org/pdf/2502.14205)  

**Abstract**: Recent years have witnessed a burgeoning interest in federated learning (FL). However, the contexts in which clients engage in sequential learning remain under-explored. Bridging FL and continual learning (CL) gives rise to a challenging practical problem: federated continual learning (FCL). Existing research in FCL primarily focuses on mitigating the catastrophic forgetting issue of continual learning while collaborating with other clients. We argue that the forgetting phenomena are not invariably detrimental. In this paper, we consider a more practical and challenging FCL setting characterized by potentially unrelated or even antagonistic data/tasks across different clients. In the FL scenario, statistical heterogeneity and data noise among clients may exhibit spurious correlations which result in biased feature learning. While existing CL strategies focus on a complete utilization of previous knowledge, we found that forgetting biased information is beneficial in our study. Therefore, we propose a new concept accurate forgetting (AF) and develop a novel generative-replay method~\method~which selectively utilizes previous knowledge in federated networks. We employ a probabilistic framework based on a normalizing flow model to quantify the credibility of previous knowledge. Comprehensive experiments affirm the superiority of our method over baselines. 

**Abstract (ZH)**: 近年来，联邦学习（FL）吸引了越来越多的研究兴趣。然而，客户端在进行序列学习时所处的环境仍被广泛忽视。将联邦学习与连续学习（CL）结合，产生了一个具有挑战性的实际问题：联邦连续学习（FCL）。现有FCL研究主要集中在缓解连续学习中的灾难性遗忘问题，同时与其他客户端合作。我们认为遗忘现象并不总是负面的。本文中，我们探讨了一个更加实际且具有挑战性的FCL设置，其中不同客户端的数据/任务可能彼此无关甚至相互对立。在联邦学习场景中，客户端之间的统计异质性和数据噪声可能会表现出虚假的相关性，从而导致偏差特征学习。现有CL策略专注于充分利用先前的知识，而我们的研究发现遗忘偏差信息是有益的。因此，我们提出了一种新的准确遗忘（AF）的概念，并开发了一种新颖的生成重放方法~\method~，该方法在联邦网络中选择性地利用先前知识。我们采用基于正则化流模型的概率框架来量度先前知识的可信度。全面的实验验证了我们方法优于基线方法的优越性。 

---
# On-the-fly Preference Alignment via Principle-Guided Decoding 

**Title (ZH)**: 基于原则引导解码的实时偏好对齐 

**Authors**: Mingye Zhu, Yi Liu, Lei Zhang, Junbo Guo, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14204)  

**Abstract**: With the rapidly expanding landscape of large language models, aligning model generations with human values and preferences is becoming increasingly important. Popular alignment methods, such as Reinforcement Learning from Human Feedback, have shown significant success in guiding models with greater control. However, these methods require considerable computational resources, which is inefficient, and substantial collection of training data to accommodate the diverse and pluralistic nature of human preferences, which is impractical. These limitations significantly constrain the scope and efficacy of both task-specific and general preference alignment methods. In this work, we introduce On-the-fly Preference Alignment via Principle-Guided Decoding (OPAD) to directly align model outputs with human preferences during inference, eliminating the need for fine-tuning. Our approach involves first curating a surrogate solution to an otherwise infeasible optimization problem and then designing a principle-guided reward function based on this surrogate. The final aligned policy is derived by maximizing this customized reward, which exploits the discrepancy between the constrained policy and its unconstrained counterpart. OPAD directly modifies the model's predictions during inference, ensuring principle adherence without incurring the computational overhead of retraining or fine-tuning. Experiments show that OPAD achieves competitive or superior performance in both general and personalized alignment tasks, demonstrating its efficiency and effectiveness compared to state-of-the-art baselines. 

**Abstract (ZH)**: 随着大规模语言模型的迅速扩展，使模型生成与人类价值观和偏好保持一致变得越来越重要。现有的对齐方法，如基于人类反馈的强化学习，已经在指导更具控制力的模型方面显示出了显著的成功。然而，这些方法需要大量的计算资源，这既不高效，且收集适应人类多元偏好的训练数据也极具挑战性。这些限制严重限制了专门任务和通用偏好对齐方法的范围和有效性。

在本工作中，我们引入了一种即时偏好对齐（On-the-fly Preference Alignment via Principle-Guided Decoding，简称OPAD）方法，在推理过程中直接将模型输出与人类偏好对齐，从而避免了调优的过程。我们的方法首先为一个原本无法解决的优化问题制定一个替代解决方案，然后基于此替代方案设计一个原则引导的奖励函数。最终对齐的策略通过最大化这个定制的奖励来生成，这一过程利用了受约束策略与未约束策略之间的差异。OPAD 在推理过程中直接修改模型的预测结果，确保遵循原则，而不需承担重新训练或调优带来的计算负担。

实验结果显示，OPAD 在通用和个性化对齐任务上均能实现具有竞争力或超越现有先进基准的性能，证明了其高效性和有效性。 

---
# Do LLMs Consider Security? An Empirical Study on Responses to Programming Questions 

**Title (ZH)**: 大型语言模型考虑安全性吗？对编程问题回应的实证研究 

**Authors**: Amirali Sajadi, Binh Le, Anh Nguyen, Kostadin Damevski, Preetha Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2502.14202)  

**Abstract**: The widespread adoption of conversational LLMs for software development has raised new security concerns regarding the safety of LLM-generated content. Our motivational study outlines ChatGPT's potential in volunteering context-specific information to the developers, promoting safe coding practices. Motivated by this finding, we conduct a study to evaluate the degree of security awareness exhibited by three prominent LLMs: Claude 3, GPT-4, and Llama 3. We prompt these LLMs with Stack Overflow questions that contain vulnerable code to evaluate whether they merely provide answers to the questions or if they also warn users about the insecure code, thereby demonstrating a degree of security awareness. Further, we assess whether LLM responses provide information about the causes, exploits, and the potential fixes of the vulnerability, to help raise users' awareness. Our findings show that all three models struggle to accurately detect and warn users about vulnerabilities, achieving a detection rate of only 12.6% to 40% across our datasets. We also observe that the LLMs tend to identify certain types of vulnerabilities related to sensitive information exposure and improper input neutralization much more frequently than other types, such as those involving external control of file names or paths. Furthermore, when LLMs do issue security warnings, they often provide more information on the causes, exploits, and fixes of vulnerabilities compared to Stack Overflow responses. Finally, we provide an in-depth discussion on the implications of our findings and present a CLI-based prompting tool that can be used to generate significantly more secure LLM responses. 

**Abstract (ZH)**: 在软件开发中广泛应用会话型大语言模型（LLM）引发了对LLM生成内容安全性的新关切。我们通过动机性研究阐述了ChatGPT在志愿提供上下文相关信息方面的潜力，以促进安全编码实践。受此发现的启发，我们对Claude 3、GPT-4和Llama 3三种主流LLM的安全意识程度进行了研究。我们通过向这些LLM提供包含漏洞代码的Stack Overflow问题，评估它们是否仅仅是提供问题的答案，还是同时警告用户有关潜在不安全代码的问题，从而展示其安全意识程度。进一步地，我们评估LLM的响应是否提供了关于漏洞原因、利用方法和潜在修复的信息，以帮助提高用户的意识。我们的研究结果表明，所有三种模型在准确检测和警告用户关于漏洞方面面临困难，在我们数据集中，其检测率为12.6%至40%。我们还注意到，这些LLM更频繁地识别那些与敏感信息暴露和不当输入中立化相关的漏洞类型，而对其他类型，如外部控制文件名或路径的漏洞关注较少。此外，当LLM发出安全警告时，它们提供的关于漏洞原因、利用方法和修复的信息比Stack Overflow的回应要多。最后，我们深入探讨了研究结果的影响，并提出了一种基于命令行接口（CLI）的提示工具，可以用于生成更安全的LLM响应。 

---
# Adaptive Sparsified Graph Learning Framework for Vessel Behavior Anomalies 

**Title (ZH)**: 适用于船舶行为异常的自适应稀疏化图学习框架 

**Authors**: Jeehong Kim, Minchan Kim, Jaeseong Ju, Youngseok Hwang, Wonhee Lee, Hyunwoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2502.14197)  

**Abstract**: Graph neural networks have emerged as a powerful tool for learning spatiotemporal interactions. However, conventional approaches often rely on predefined graphs, which may obscure the precise relationships being modeled. Additionally, existing methods typically define nodes based on fixed spatial locations, a strategy that is ill-suited for dynamic environments like maritime environments. Our method introduces an innovative graph representation where timestamps are modeled as distinct nodes, allowing temporal dependencies to be explicitly captured through graph edges. This setup is extended to construct a multi-ship graph that effectively captures spatial interactions while preserving graph sparsity. The graph is processed using Graph Convolutional Network layers to capture spatiotemporal patterns, with a forecasting layer for feature prediction and a Variational Graph Autoencoder for reconstruction, enabling robust anomaly detection. 

**Abstract (ZH)**: 图形神经网络已经成为了学习时空交互的强大工具。然而，传统的做法往往依赖于预先定义的图形，这可能会模糊精确的关系。此外，现有的方法通常是基于固定的空间位置定义节点，这种方法不适合动态环境，比如海上环境。我们的方法引入了一种创新的图形表示，其中将时间戳作为不同的节点进行建模，通过图形边明确捕捉时间依赖关系。这种设置可以扩展以构建一个多船舶图形，该图形有效地捕获空间交互并保持图形稀疏性。该图形通过图形卷积网络层来捕获时空模式，通过预测层进行特征预测，并通过变分图形自编码器进行重建，从而实现稳健的异常检测。 

---
# Multimodal RewardBench: Holistic Evaluation of Reward Models for Vision Language Models 

**Title (ZH)**: 多模态奖励基准：视觉语言模型奖励模型的全面评估 

**Authors**: Michihiro Yasunaga, Luke Zettlemoyer, Marjan Ghazvininejad  

**Link**: [PDF](https://arxiv.org/pdf/2502.14191)  

**Abstract**: Reward models play an essential role in training vision-language models (VLMs) by assessing output quality to enable aligning with human preferences. Despite their importance, the research community lacks comprehensive open benchmarks for evaluating multimodal reward models in VLMs. To address this gap, we introduce Multimodal RewardBench, an expert-annotated benchmark covering six domains: general correctness, preference, knowledge, reasoning, safety, and visual question-answering. Our dataset comprises 5,211 annotated (prompt, chosen response, rejected response) triplets collected from various VLMs. In evaluating a range of VLM judges, we find that even the top-performing models, Gemini 1.5 Pro and Claude 3.5 Sonnet, achieve only 72% overall accuracy. Notably, most models struggle in the reasoning and safety domains. These findings suggest that Multimodal RewardBench offers a challenging testbed for advancing reward model development across multiple domains. We release the benchmark at this https URL. 

**Abstract (ZH)**: 多模态奖励模型在训练视觉语言模型（VLMs）中扮演着至关重要的角色，它们通过评估输出质量来使模型与人类偏好保持一致。尽管如此，研究社区仍然缺乏对VLM中多模态奖励模型进行全面评估的公开基准。为了解决这一不足，我们引入了Multimodal RewardBench，这是一个专家注释的基准，涵盖了六个领域：通用正确性、偏好、知识、推理、安全性和视觉问答。我们的数据集包括来自多种VLM的5,211个注释过的（提示，选择的响应，拒绝的响应）三元组。在对多种VLM裁判的评估中，我们发现即使是表现最佳的模型Gemini 1.5 Pro和Claude 3.5 Sonnet，其整体准确率也只有72%。值得注意的是，大多数模型在推理和安全性领域表现不佳。这些发现表明，Multimodal RewardBench 为跨多个领域的奖励模型发展提供了具有挑战性的测试平台。我们已在此链接处发布了此基准：[此处链接]。 

---
# Type 1 Diabetes Management using GLIMMER: Glucose Level Indicator Model with Modified Error Rate 

**Title (ZH)**: 使用GLIMMER进行1型糖尿病管理：修正误差率的血糖水平指示模型 

**Authors**: Saman Khamesian, Asiful Arefeen, Adela Grando, Bithika Thompson, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2502.14183)  

**Abstract**: Managing Type 1 Diabetes (T1D) demands constant vigilance as individuals strive to regulate their blood glucose levels to avert the dangers of dysglycemia (hyperglycemia or hypoglycemia). Despite the advent of sophisticated technologies such as automated insulin delivery (AID) systems, achieving optimal glycemic control remains a formidable task. AID systems integrate continuous subcutaneous insulin infusion (CSII) and continuous glucose monitors (CGM) data, offering promise in reducing variability and increasing glucose time-in-range. However, these systems often fail to prevent dysglycemia, partly due to limitations in prediction algorithms that lack the precision to avert abnormal glucose events. This gap highlights the need for proactive behavioral adjustments. We address this need with GLIMMER, Glucose Level Indicator Model with Modified Error Rate, a machine learning approach for forecasting blood glucose levels. GLIMMER categorizes glucose values into normal and abnormal ranges and devises a novel custom loss function to prioritize accuracy in dysglycemic events where patient safety is critical. To evaluate the potential of GLIMMER for T1D management, we both use a publicly available dataset and collect new data involving 25 patients with T1D. In predicting next-hour glucose values, GLIMMER achieved a root mean square error (RMSE) of 23.97 (+/-3.77) and a mean absolute error (MAE) of 15.83 (+/-2.09) mg/dL. These results reflect a 23% improvement in RMSE and a 31% improvement in MAE compared to the best-reported error rates. 

**Abstract (ZH)**: 管理1型糖尿病（T1D）需要持续的关注，个体需要不断调节血糖水平，以避免高血糖或低血糖的危险。尽管先进的自动化胰岛素输送（AID）系统等技术已经问世，但实现最佳血糖控制仍是一项艰巨任务。AID系统将连续皮下胰岛素输注（CSII）和连续血糖监测（CGM）的数据整合起来，有望减少血糖波动并增加血糖处于目标范围内的时间。然而，这些系统往往无法预防血糖异常，部分原因在于预测算法的精确性不足，难以避免异常的血糖事件。这一缺口突显了需要采取主动行为调整的必要性。我们通过提出GLIMMER（Glucose Level Indicator Model with Modified Error Rate）这一机器学习方法来应对这一需求，GLIMMER用于预测血糖水平。GLIMMER将血糖值划分为正常和异常范围，并设计了一种新的自定义损失函数，以在患者安全至关重要的低血糖事件中优先确保准确性。为了评估GLIMMER在T1D管理中的潜力，我们使用了一个公开可用的数据集，并收集了涉及25名T1D患者的新增数据。在预测下一小时的血糖值时，GLIMMER的均方根误差（RMSE）为23.97（±3.77）mg/dL，平均绝对误差（MAE）为15.83（±2.09）mg/dL。这些结果表明，与报告的最低误差率相比，GLIMMER的RMSE提高了23%，MAE提高了31%。 

---
# A modal logic translation of the AGM axioms for belief revision 

**Title (ZH)**: 模态逻辑中AGM信念修正公理的翻译 

**Authors**: Giacomo Bonanno  

**Link**: [PDF](https://arxiv.org/pdf/2502.14176)  

**Abstract**: Building on the analysis of Bonanno (Artificial Intelligence, 2025) we introduce a simple modal logic containing three modal operators: a unimodal belief operator, a bimodal conditional operator and the unimodal global operator. For each AGM axiom for belief revision, we provide a corresponding modal axiom. The correspondence is as follows: each AGM axiom is characterized by a property of the Kripke-Lewis frames considered in Bonanno (Artificial Intelligence, 2025) and, in turn, that property characterizes the proposed modal axiom. 

**Abstract (ZH)**: 基于Bonanno（《人工智能》，2025）的研究，我们引入了一种包含三个模态操作符的简单模态逻辑：一个单模态信念操作符、一个双模态条件操作符以及一个单模态全局操作符。对于每个AGM信念修订公理，我们提供了一个对应的模态公理。对应关系如下：每个AGM信念修订公理通过Bonanno（《人工智能》，2025）中考虑的Kripke-Lewis框架的某个性质来表征，而这一性质又表征了所提出的模态公理。 

---
# Weighted Low-rank Approximation via Stochastic Gradient Descent on Manifolds 

**Title (ZH)**: 基于流形上的随机梯度下降的加权低秩逼近 

**Authors**: Conglong Xu, Peiqi Yang, Hao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14174)  

**Abstract**: We solve a regularized weighted low-rank approximation problem by a stochastic gradient descent on a manifold. To guarantee the convergence of our stochastic gradient descent, we establish a convergence theorem on manifolds for retraction-based stochastic gradient descents admitting confinements. On sample data from the Netflix Prize training dataset, our algorithm outperforms the existing stochastic gradient descent on Euclidean spaces. We also compare the accelerated line search on this manifold to the existing accelerated line search on Euclidean spaces. 

**Abstract (ZH)**: 我们通过流形上的梯度下降方法解决了一个正则化加权低秩逼近问题。为了保证我们所采用的梯度下降法的收敛性，我们在限制条件下建立了基于复位操作的流形上随机梯度下降的收敛定理。在Netflix Prize训练数据集的样例数据上，我们的算法优于现有的欧几里得空间上的随机梯度下降方法。我们还对比了在该流形上采用的加速线搜索方法与现有在欧几里得空间上采用的加速线搜索方法。 

---
# Efficient Inverse Multiagent Learning 

**Title (ZH)**: 高效的逆多智能体学习 

**Authors**: Denizalp Goktas, Amy Greenwald, Sadie Zhao, Alec Koppel, Sumitra Ganesh  

**Link**: [PDF](https://arxiv.org/pdf/2502.14160)  

**Abstract**: In this paper, we study inverse game theory (resp. inverse multiagent learning) in which the goal is to find parameters of a game's payoff functions for which the expected (resp. sampled) behavior is an equilibrium. We formulate these problems as generative-adversarial (i.e., min-max) optimization problems, for which we develop polynomial-time algorithms to solve, the former of which relies on an exact first-order oracle, and the latter, a stochastic one. We extend our approach to solve inverse multiagent simulacral learning in polynomial time and number of samples. In these problems, we seek a simulacrum, meaning parameters and an associated equilibrium that replicate the given observations in expectation. We find that our approach outperforms the widely-used ARIMA method in predicting prices in Spanish electricity markets based on time-series data. 

**Abstract (ZH)**: 在本文中，我们研究逆博弈理论（即逆多智能体学习），其目标是在给定期望（或采样）行为是均衡的情况下，找到博弈支付函数的参数。我们将这些问题形式化为生成式对抗（即最小-最大）优化问题，并为此开发了多项式时间算法，前者依赖于精确的一阶 oracle，后者依赖于随机的一阶 oracle。我们将该方法推广，以便多项式时间及样本数量内解决逆多智能体仿真学习问题。在这些问题中，我们寻求一个仿真体，即找到能够复制给定观测的参数和相应的均衡。我们发现，我们的方法在基于时间序列数据预测西班牙电力市场电价方面优于广泛使用的 ARIMA 方法。 

---
# PitVQA++: Vector Matrix-Low-Rank Adaptation for Open-Ended Visual Question Answering in Pituitary Surgery 

**Title (ZH)**: PitVQA++：垂体手术中开放式视觉问答的向量矩阵低秩适应方法 

**Authors**: Runlong He, Danyal Z. Khan, Evangelos B. Mazomenos, Hani J. Marcus, Danail Stoyanov, Matthew J. Clarkson, Mobarakol Islam  

**Link**: [PDF](https://arxiv.org/pdf/2502.14149)  

**Abstract**: Vision-Language Models (VLMs) in visual question answering (VQA) offer a unique opportunity to enhance intra-operative decision-making, promote intuitive interactions, and significantly advancing surgical education. However, the development of VLMs for surgical VQA is challenging due to limited datasets and the risk of overfitting and catastrophic forgetting during full fine-tuning of pretrained weights. While parameter-efficient techniques like Low-Rank Adaptation (LoRA) and Matrix of Rank Adaptation (MoRA) address adaptation challenges, their uniform parameter distribution overlooks the feature hierarchy in deep networks, where earlier layers, that learn general features, require more parameters than later ones. This work introduces PitVQA++ with an open-ended PitVQA dataset and vector matrix-low-rank adaptation (Vector-MoLoRA), an innovative VLM fine-tuning approach for adapting GPT-2 to pituitary surgery. Open-Ended PitVQA comprises around 101,803 frames from 25 procedural videos with 745,972 question-answer sentence pairs, covering key surgical elements such as phase and step recognition, context understanding, tool detection, localization, and interactions recognition. Vector-MoLoRA incorporates the principles of LoRA and MoRA to develop a matrix-low-rank adaptation strategy that employs vector ranking to allocate more parameters to earlier layers, gradually reducing them in the later layers. Our approach, validated on the Open-Ended PitVQA and EndoVis18-VQA datasets, effectively mitigates catastrophic forgetting while significantly enhancing performance over recent baselines. Furthermore, our risk-coverage analysis highlights its enhanced reliability and trustworthiness in handling uncertain predictions. Our source code and dataset is available at~\url{this https URL}. 

**Abstract (ZH)**: 视觉-语言模型（VLMs）在手术视觉问答（Surgical VQA）中的应用为提高术中决策质量、促进直观交互以及显著推进手术教育提供了独特的机会。然而，由于受限的数据集和全微调预训练权重过程中过拟合和灾难性遗忘的风险，开发适用于手术VQA的VLMs具有挑战性。尽管参数高效的技术，如低秩适应（LoRA）和矩阵低秩适应（MoRA），能够解决适应性挑战，但它们均匀的参数分布忽略了深层网络中的特征层次结构，即早期层相比后期层需要更多的参数来学习通用特征。本文提出了PitVQA++以及一个开放性PitVQA数据集，并引入了Vector-MoLoRA，这是一种创新的VLM微调方法，用于将GPT-2适应于垂体手术。开放性PitVQA数据集包含约101,803帧来自25个手术过程视频的约745,972个问题-答案句子对，涵盖了包括手术阶段和步骤识别、上下文理解、工具检测、定位和交互识别在内的关键手术元素。Vector-MoLoRA 结合了LoRA和MoRA的原则，开发了一种矩阵低秩适应策略，利用向量排名来分配更多参数给早期层，并在后期层逐渐减少参数。该方法在开放性PitVQA和EndoVis18-VQA数据集上的验证表明，它有效减轻了灾难性遗忘，同时显著提升了与近期基线模型相比的性能。此外，我们的风险覆盖分析进一步突显了其在处理不确定预测中的增强可靠性和可信度。我们的源代码和数据集可从 [this https URL] 获取。 

---
# Multi-Agent Risks from Advanced AI 

**Title (ZH)**: 高级人工智能中的多智能体风险 

**Authors**: Lewis Hammond, Alan Chan, Jesse Clifton, Jason Hoelscher-Obermaier, Akbir Khan, Euan McLean, Chandler Smith, Wolfram Barfuss, Jakob Foerster, Tomáš Gavenčiak, Anh Han, Edward Hughes, Vojtěch Kovařík, Jan Kulveit, Joel Z. Leibo, Caspar Oesterheld, Christian Schroeder de Witt, Nisarg Shah, Michael Wellman, Paolo Bova, Theodor Cimpeanu, Carson Ezell, Quentin Feuillade-Montixi, Matija Franklin, Esben Kran, Igor Krawczuk, Max Lamparth, Niklas Lauffer, Alexander Meinke, Sumeet Motwani, Anka Reuel, Vincent Conitzer, Michael Dennis, Iason Gabriel, Adam Gleave, Gillian Hadfield, Nika Haghtalab, Atoosa Kasirzadeh, Sébastien Krier, Kate Larson, Joel Lehman, David C. Parkes, Georgios Piliouras, Iyad Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2502.14143)  

**Abstract**: The rapid development of advanced AI agents and the imminent deployment of many instances of these agents will give rise to multi-agent systems of unprecedented complexity. These systems pose novel and under-explored risks. In this report, we provide a structured taxonomy of these risks by identifying three key failure modes (miscoordination, conflict, and collusion) based on agents' incentives, as well as seven key risk factors (information asymmetries, network effects, selection pressures, destabilising dynamics, commitment problems, emergent agency, and multi-agent security) that can underpin them. We highlight several important instances of each risk, as well as promising directions to help mitigate them. By anchoring our analysis in a range of real-world examples and experimental evidence, we illustrate the distinct challenges posed by multi-agent systems and their implications for the safety, governance, and ethics of advanced AI. 

**Abstract (ZH)**: 先进的AI代理的快速发展以及这些代理即将部署的实例将导致前所未有的复杂多代理系统。这些系统带来了新颖且尚未充分探索的风险。本报告通过基于代理激励来识别三种关键失败模式（协调不当、冲突和勾结），以及七个关键风险因素（信息不对称、网络效应、选择性压力、不稳定的动态、承诺问题、新兴代理和多代理安全），为这些风险提供了一个结构化的分类体系。我们强调了每种风险的重要实例，并指出了有助于缓解这些风险的潜在方向。通过结合一系列现实世界案例和实验证据，我们阐述了多代理系统所提出的独特挑战及其对先进AI的安全性、治理和伦理的影响。 

---
# Can Community Notes Replace Professional Fact-Checkers? 

**Title (ZH)**: 社区笔记能否取代专业事实核查人员？ 

**Authors**: Nadav Borenstein, Greta Warren, Desmond Elliott, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2502.14132)  

**Abstract**: Two commonly-employed strategies to combat the rise of misinformation on social media are (i) fact-checking by professional organisations and (ii) community moderation by platform users. Policy changes by Twitter/X and, more recently, Meta, signal a shift away from partnerships with fact-checking organisations and towards an increased reliance on crowdsourced community notes. However, the extent and nature of dependencies between fact-checking and helpful community notes remain unclear. To address these questions, we use language models to annotate a large corpus of Twitter/X community notes with attributes such as topic, cited sources, and whether they refute claims tied to broader misinformation narratives. Our analysis reveals that community notes cite fact-checking sources up to five times more than previously reported. Fact-checking is especially crucial for notes on posts linked to broader narratives, which are twice as likely to reference fact-checking sources compared to other sources. In conclusion, our results show that successful community moderation heavily relies on professional fact-checking. 

**Abstract (ZH)**: 应对社交媒体上虚假信息蔓延的两种常用策略是：（i）专业组织进行事实核查；（ii）平台用户进行社区管理。Twitter/X及最近的Meta所进行的政策调整表明，这种策略正在从与事实核查组织的合作转向更依赖于由社区提供的众包注释。然而，事实核查与有益的社区注释之间的依赖关系及其性质仍然不甚明确。为了解决这些问题，我们利用语言模型对大量Twitter/X社区注释进行标记，标注其属性如主题、引用来源以及是否反驳与更广泛虚假信息叙事相关的内容。我们的分析显示，社区注释引用的事实核查来源比之前报告的要多五倍。特别是对于与更广泛虚假信息叙事相关的帖子，这些注释引用事实核查来源的可能性是引用其他来源的两倍。综上所述，我们的结果表明，成功的社区管理高度依赖于专业事实核查。 

---
# Gradients can train reward models: An Empirical Risk Minimization Approach for Offline Inverse RL and Dynamic Discrete Choice Model 

**Title (ZH)**: 梯度可以训练奖励模型：基于经验风险最小化的离线逆强化学习和动态离散选择模型的empirical风险最小化方法 

**Authors**: Enoch H. Kang, Hema Yoganarasimhan, Lalit Jain  

**Link**: [PDF](https://arxiv.org/pdf/2502.14131)  

**Abstract**: We study the problem of estimating Dynamic Discrete Choice (DDC) models, also known as offline Maximum Entropy-Regularized Inverse Reinforcement Learning (offline MaxEnt-IRL) in machine learning. The objective is to recover reward or $Q^*$ functions that govern agent behavior from offline behavior data. In this paper, we propose a globally convergent gradient-based method for solving these problems without the restrictive assumption of linearly parameterized rewards. The novelty of our approach lies in introducing the Empirical Risk Minimization (ERM) based IRL/DDC framework, which circumvents the need for explicit state transition probability estimation in the Bellman equation. Furthermore, our method is compatible with non-parametric estimation techniques such as neural networks. Therefore, the proposed method has the potential to be scaled to high-dimensional, infinite state spaces. A key theoretical insight underlying our approach is that the Bellman residual satisfies the Polyak-Lojasiewicz (PL) condition -- a property that, while weaker than strong convexity, is sufficient to ensure fast global convergence guarantees. Through a series of synthetic experiments, we demonstrate that our approach consistently outperforms benchmark methods and state-of-the-art alternatives. 

**Abstract (ZH)**: 我们研究动态离散选择（Dynamic Discrete Choice, DDC）模型的估计问题，亦称之为机器学习中的离线最大熵逆强化学习（offline MaxEnt-IRL）。目标是从离线行为数据中恢复支配代理行为的奖励或$Q^*$函数。本文提出了一种全局收敛的梯度优化方法，用于解决这些问题，无需假设奖励在线性参数化的限制条件下。我们方法的创新之处在于引入了基于经验风险最小化（Empirical Risk Minimization, ERM）的逆强化学习/动态离散选择（IRL/DDC）框架，从而避免了隐式状态过渡概率估算的需求。此外，我们的方法与非参数化估计技术（如神经网络）兼容，因此所提出的方法具有扩展到高维度、无限状态空间的潜力。我们方法背后的理论洞察在于贝尔曼残差满足Polyak-Lojasiewicz（PL）条件——这种属性虽然弱于强凸性，但也足以确保快速全局收敛的保证。通过一系列合成实验，我们证明了我们的方法在性能上始终优于基准方法和最先进的替代方案。 

---
# Multi-Objective Bayesian Optimization for Networked Black-Box Systems: A Path to Greener Profits and Smarter Designs 

**Title (ZH)**: 面向网络化黑盒系统的多目标贝叶斯优化：通向更绿色的利润和更智能的设计的道路 

**Authors**: Akshay Kudva, Wei-Ting Tang, Joel A. Paulson  

**Link**: [PDF](https://arxiv.org/pdf/2502.14121)  

**Abstract**: Designing modern industrial systems requires balancing several competing objectives, such as profitability, resilience, and sustainability, while accounting for complex interactions between technological, economic, and environmental factors. Multi-objective optimization (MOO) methods are commonly used to navigate these tradeoffs, but selecting the appropriate algorithm to tackle these problems is often unclear, particularly when system representations vary from fully equation-based (white-box) to entirely data-driven (black-box) models. While grey-box MOO methods attempt to bridge this gap, they typically impose rigid assumptions on system structure, requiring models to conform to the underlying structural assumptions of the solver rather than the solver adapting to the natural representation of the system of interest. In this chapter, we introduce a unifying approach to grey-box MOO by leveraging network representations, which provide a general and flexible framework for modeling interconnected systems as a series of function nodes that share various inputs and outputs. Specifically, we propose MOBONS, a novel Bayesian optimization-inspired algorithm that can efficiently optimize general function networks, including those with cyclic dependencies, enabling the modeling of feedback loops, recycle streams, and multi-scale simulations - features that existing methods fail to capture. Furthermore, MOBONS incorporates constraints, supports parallel evaluations, and preserves the sample efficiency of Bayesian optimization while leveraging network structure for improved scalability. We demonstrate the effectiveness of MOBONS through two case studies, including one related to sustainable process design. By enabling efficient MOO under general graph representations, MOBONS has the potential to significantly enhance the design of more profitable, resilient, and sustainable engineering systems. 

**Abstract (ZH)**: 设计现代工业系统需要在多个相互竞争的目标之间进行权衡，如盈利能力、韧性和可持续性，同时需要考虑到技术、经济和环境等多种因素之间的复杂相互作用。多目标优化（MOO）方法常被用来处理这些权衡，但在选择合适的算法来解决这些问题时通常不够明确，尤其是在系统表示从完全基于方程的（白箱）模型到完全数据驱动的（黑箱）模型之间变化的情况下。虽然灰色箱体MOO方法试图弥补这一差距，但它们通常会对系统结构施加严格的假设，要求模型符合求解器的基本结构假设，而不是让求解器适应被研究系统的自然表示。在这一章节中，我们将通过利用网络表示提出一种统一的灰色箱体多目标优化（MOO）方法。网络表示提供了一种通用且灵活的框架，可以将相互连接的系统建模为一系列具有各种输入和输出的功能节点。具体而言，我们提出了一个名为MOBONS的新颖算法，该算法借鉴了贝叶斯优化的理念，能够高效地优化通用功能网络，包括存在循环依赖关系的网络，从而能够建模反馈回路、循环流和多尺度仿真——这是现有方法无法捕获的特性。此外，MOBONS还包含了约束条件支持并行评估，并利用网络结构提高了贝叶斯优化的样本效率，从而增强了可扩展性。我们通过两个案例研究展示了MOBONS的有效性，其中包括一个与可持续过程设计相关的研究。通过在通用图表示下实现高效的MOO，MOBONS有望显著提升更具盈利性、韧性和可持续性的工程系统的设计。 

---
# Zero loss guarantees and explicit minimizers for generic overparametrized Deep Learning networks 

**Title (ZH)**: 泛化的深度学习网络中零损失保证和显式最小值点的保证 

**Authors**: Thomas Chen, Andrew G. Moore  

**Link**: [PDF](https://arxiv.org/pdf/2502.14114)  

**Abstract**: We determine sufficient conditions for overparametrized deep learning (DL) networks to guarantee the attainability of zero loss in the context of supervised learning, for the $\mathcal{L}^2$ cost and {\em generic} training data. We present an explicit construction of the zero loss minimizers without invoking gradient descent. On the other hand, we point out that increase of depth can deteriorate the efficiency of cost minimization using a gradient descent algorithm by analyzing the conditions for rank loss of the training Jacobian. Our results clarify key aspects on the dichotomy between zero loss reachability in underparametrized versus overparametrized DL. 

**Abstract (ZH)**: 我们确定了在监督学习背景下，对于$\mathcal{L}^2$代价和通用训练数据，过参数化深度学习（DL）网络达到零损失的充分条件。我们给出了零损失最小值器的显式构造，而不依赖于梯度下降法。另一方面，通过分析训练雅可比矩阵秩损失的条件，我们指出深度的增加可能会降低使用梯度下降算法进行代价最小化的效率。我们的结果阐明了在欠参数化与过参数化深度学习之间达到零损失的二分法中的关键方面。 

---
# Object-centric Binding in Contrastive Language-Image Pretraining 

**Title (ZH)**: 对比语言-图像预训练中的对象中心绑定 

**Authors**: Rim Assouel, Pietro Astolfi, Florian Bordes, Michal Drozdzal, Adriana Romero-Soriano  

**Link**: [PDF](https://arxiv.org/pdf/2502.14113)  

**Abstract**: Recent advances in vision language models (VLM) have been driven by contrastive models such as CLIP, which learn to associate visual information with their corresponding text descriptions. However, these models have limitations in understanding complex compositional scenes involving multiple objects and their spatial relationships. To address these challenges, we propose a novel approach that diverges from commonly used strategies, which rely on the design of hard-negative augmentations. Instead, our work focuses on integrating inductive biases into pre-trained CLIP-like models to improve their compositional understanding without using any additional hard-negatives. To that end, we introduce a binding module that connects a scene graph, derived from a text description, with a slot-structured image representation, facilitating a structured similarity assessment between the two modalities. We also leverage relationships as text-conditioned visual constraints, thereby capturing the intricate interactions between objects and their contextual relationships more effectively. Our resulting model not only enhances the performance of CLIP-based models in multi-object compositional understanding but also paves the way towards more accurate and sample-efficient image-text matching of complex scenes. 

**Abstract (ZH)**: 近年来，视觉语言模型（VLM）的进步主要得益于对比模型，如CLIP，这些模型能够学习将视觉信息与其对应的文本描述关联起来。然而，这些模型在理解涉及多个物体及其空间关系的复杂组合场景方面存在局限性。为了解决这些挑战，我们提出了一种新颖的方法，该方法与依赖于设计负样本增强的常见策略不同。我们的工作集中在将归纳偏置整合到预训练的CLIP-like模型中，以提高它们的组合理解能力，而无需使用任何额外的负样本。为此，我们引入了一个绑定模块，该模块将从文本描述中获得的场景图与具有槽结构的图像表示连接起来，从而促进两种模态之间的结构化相似性评估。我们还利用关系作为文本条件下的视觉约束，从而更有效地捕捉物体及其上下文关系之间的复杂交互。通过这种方法，我们的模型不仅提高了基于CLIP的模型在多物体组合理解方面的性能，还为更准确和样本有效的复杂场景中的图像-文本匹配铺平了道路。 

---
# Navigating Semantic Relations: Challenges for Language Models in Abstract Common-Sense Reasoning 

**Title (ZH)**: 导航语义关系：语言模型在抽象常识推理中的挑战 

**Authors**: Cole Gawin, Yidan Sun, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2502.14086)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance in generating human-like text and solving reasoning tasks of moderate complexity, such as question-answering and mathematical problem-solving. However, their capabilities in tasks requiring deeper cognitive skills, such as common-sense understanding and abstract reasoning, remain under-explored. In this paper, we systematically evaluate abstract common-sense reasoning in LLMs using the ConceptNet knowledge graph. We propose two prompting approaches: instruct prompting, where models predict plausible semantic relationships based on provided definitions, and few-shot prompting, where models identify relations using examples as guidance. Our experiments with the gpt-4o-mini model show that in instruct prompting, consistent performance is obtained when ranking multiple relations but with substantial decline when the model is restricted to predicting only one relation. In few-shot prompting, the model's accuracy improves significantly when selecting from five relations rather than the full set, although with notable bias toward certain relations. These results suggest significant gaps still, even in commercially used LLMs' abstract common-sense reasoning abilities, compared to human-level understanding. However, the findings also highlight the promise of careful prompt engineering, based on selective retrieval, for obtaining better performance. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在生成类人类文本和解决中等复杂度的推理任务（如问答和数学问题求解）方面取得了显著成果。然而，它们在需要更深层次认知能力的任务（如常识理解和抽象推理）中的能力尚待探索。本文系统地使用ConceptNet知识图谱评估LLMs在抽象常识推理方面的表现。我们提出了两种提示方法：指令提示，模型根据提供的定义预测合乎逻辑的语义关系；以及少样本提示，模型在示例的指导下识别关系。我们的实验使用gpt-4o-mini模型显示，在指令提示中，当对多种关系进行排序时，可以获得一致的表现，但当模型被限制只能预测单一关系时，表现会显著下降。在少样本提示中，当从五个关系中选择时，模型的准确性显著提高，尽管存在明显的偏好某些关系的现象。这些结果表明，即使是商用的大规模语言模型在抽象常识推理方面的能力与人类级别的理解之间仍存在显著差距。然而，研究结果也突显了通过精心设计提示工程（基于选择性检索）来获得更好性能的潜力。 

---
# Personalized Education with Generative AI and Digital Twins: VR, RAG, and Zero-Shot Sentiment Analysis for Industry 4.0 Workforce Development 

**Title (ZH)**: 基于生成式AI和数字孪生的个性化教育：面向工业4.0劳动力发展的VR、RAG和零样本情感分析 

**Authors**: Yu-Zheng Lin, Karan Petal, Ahmed H Alhamadah, Sujan Ghimire, Matthew William Redondo, David Rafael Vidal Corona, Jesus Pacheco, Soheil Salehi, Pratik Satam  

**Link**: [PDF](https://arxiv.org/pdf/2502.14080)  

**Abstract**: The Fourth Industrial Revolution (4IR) technologies, such as cloud computing, machine learning, and AI, have improved productivity but introduced challenges in workforce training and reskilling. This is critical given existing workforce shortages, especially in marginalized communities like Underrepresented Minorities (URM), who often lack access to quality education. Addressing these challenges, this research presents gAI-PT4I4, a Generative AI-based Personalized Tutor for Industrial 4.0, designed to personalize 4IR experiential learning. gAI-PT4I4 employs sentiment analysis to assess student comprehension, leveraging generative AI and finite automaton to tailor learning experiences. The framework integrates low-fidelity Digital Twins for VR-based training, featuring an Interactive Tutor - a generative AI assistant providing real-time guidance via audio and text. It uses zero-shot sentiment analysis with LLMs and prompt engineering, achieving 86\% accuracy in classifying student-teacher interactions as positive or negative. Additionally, retrieval-augmented generation (RAG) enables personalized learning content grounded in domain-specific knowledge. To adapt training dynamically, finite automaton structures exercises into states of increasing difficulty, requiring 80\% task-performance accuracy for progression. Experimental evaluation with 22 volunteers showed improved accuracy exceeding 80\%, reducing training time. Finally, this paper introduces a Multi-Fidelity Digital Twin model, aligning Digital Twin complexity with Bloom's Taxonomy and Kirkpatrick's model, providing a scalable educational framework. 

**Abstract (ZH)**: 第四次工业革命（4IR）技术，如云计算、机器学习和人工智能，虽然提高了生产力，但也为劳动力培训和再培训带来了挑战。鉴于现有的劳动力短缺问题，尤其是在被边缘化的社区，如代表性不足的少数群体（URM），他们往往缺乏高质量教育机会。为应对这些挑战，本研究提出了一种基于生成式人工智能的个性化导师gAI-PT4I4，旨在个性化4IR体验式学习。gAI-PT4I4利用情感分析评估学生理解程度，结合生成式人工智能和有限自动机以定制学习体验。该框架整合了用于VR培训的低保真数字孪生，其中包括交互式导师——一个生成式人工智能助理，可提供实时音频和文本指导。它使用零样本的情感分析与大规模语言模型（LLM）及提示工程相结合，实现86%的准确性，用于分类学生-教师互动为正面或负面。此外，检索增强生成（RAG）技术能够生成基于专业领域知识的个性化学习内容。为动态适应培训需求，有限自动机结构化训练练习为递增难度的状态，要求任务执行准确率达到80%才能进步。实验评估使用22名志愿者显示，学习准确率超过了80%，从而缩短了培训时间。最后，本文介绍了多保真度数字孪生模型，该模型将数字孪生的复杂性与布卢姆分类法和柯克帕特里克模型相匹配，提供了一个可扩展的教育框架。 

---
# DiffExp: Efficient Exploration in Reward Fine-tuning for Text-to-Image Diffusion Models 

**Title (ZH)**: DiffExp: 在文本到图像扩散模型奖励调优中高效探索的方法 

**Authors**: Daewon Chae, June Suk Choi, Jinkyu Kim, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.14070)  

**Abstract**: Fine-tuning text-to-image diffusion models to maximize rewards has proven effective for enhancing model performance. However, reward fine-tuning methods often suffer from slow convergence due to online sample generation. Therefore, obtaining diverse samples with strong reward signals is crucial for improving sample efficiency and overall performance. In this work, we introduce DiffExp, a simple yet effective exploration strategy for reward fine-tuning of text-to-image models. Our approach employs two key strategies: (a) dynamically adjusting the scale of classifier-free guidance to enhance sample diversity, and (b) randomly weighting phrases of the text prompt to exploit high-quality reward signals. We demonstrate that these strategies significantly enhance exploration during online sample generation, improving the sample efficiency of recent reward fine-tuning methods, such as DDPO and AlignProp. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，要符合学术规范：

对文本到图像扩散模型进行精细调整以最大化奖励已被证明能够有效提升模型性能。然而，奖励精细调整方法往往因在线样本生成而导致收敛速度较慢。因此，获取具有强奖励信号的多样化样本对于提高样本效率和整体性能至关重要。在本研究中，我们介绍了一种名为DiffExp的简单而有效的探索策略，用于文本到图像模型的奖励精细调整。我们的方法采用了两个关键策略：(a) 动态调整无分类器向导的缩放比例，以增强样本的多样性；(b) 随机调整文本提示中短语的权重，以利用高质量的奖励信号。我们证明，这些策略显著提升了在线样本生成过程中的探索能力，从而改进了近期的奖励精细调整方法，如DDPO和AlignProp的样本效率。 

---
# A Racing Dataset and Baseline Model for Track Detection in Autonomous Racing 

**Title (ZH)**: 自动赛车道检测的数据集及 baseline 模型 

**Authors**: Shreya Ghosh, Yi-Huan Chen, Ching-Hsiang Huang, Abu Shafin Mohammad Mahdee Jameel, Chien Chou Ho, Aly El Gamal, Samuel Labi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14068)  

**Abstract**: A significant challenge in racing-related research is the lack of publicly available datasets containing raw images with corresponding annotations for the downstream task. In this paper, we introduce RoRaTrack, a novel dataset that contains annotated multi-camera image data from racing scenarios for track detection. The data is collected on a Dallara AV-21 at a racing circuit in Indiana, in collaboration with the Indy Autonomous Challenge (IAC). RoRaTrack addresses common problems such as blurriness due to high speed, color inversion from the camera, and absence of lane markings on the track. Consequently, we propose RaceGAN, a baseline model based on a Generative Adversarial Network (GAN) that effectively addresses these challenges. The proposed model demonstrates superior performance compared to current state-of-the-art machine learning models in track detection. The dataset and code for this work are available at this http URL. 

**Abstract (ZH)**: 与赛车研究相关的显著挑战之一是缺乏包含与下游任务对应的注释原始图像的公开数据集。在本文中，我们介绍了RoRaTrack，这是一个新颖的数据集，包含来自赛车场景的多摄像头图像数据，用于赛道检测。这些数据是在美国印第安纳州的一个赛车场上，与印第安纳自主挑战赛（IAC）合作，使用达拉拉AV-21采集的。RoRaTrack 解决了诸如由于高速导致的模糊、相机导致的颜色反转以及赛道上缺乏车道标记等常见问题。因此，我们提出了一种基于生成对抗网络（GAN）的基准模型RaceGAN，该模型有效地解决了这些挑战。所提出的模型在轨道检测方面表现出优于当前最先进的机器学习模型的性能。该项目的数据集和代码可以在以下网址获取：[此处填写网址]。 

---
# Triad: Vision Foundation Model for 3D Magnetic Resonance Imaging 

**Title (ZH)**: Triad: 视觉基础模型在三维磁共振成像中的应用 

**Authors**: Shansong Wang, Mojtaba Safari, Qiang Li, Chih-Wei Chang, Richard LJ Qiu, Justin Roper, David S. Yu, Xiaofeng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14064)  

**Abstract**: Vision foundation models (VFMs) are pre-trained on extensive image datasets to learn general representations for diverse types of data. These models can subsequently be fine-tuned for specific downstream tasks, significantly boosting performance across a broad range of applications. However, existing vision foundation models that claim to be applicable to various radiology tasks are mostly pre-trained on 3D computed tomography (CT), which benefits from the availability of extensive 3D CT databases. Significant differences between CT and magnetic resonance imaging (MRI) in imaging principles, signal characteristics, and data distribution may hinder their practical performance and versatility in MRI-specific applications. Here, we propose Triad, a vision foundation model for 3D MRI. Triad adopts a widely used autoencoder architecture to learn robust representations from 131,170 3D MRI volumes and uses organ-independent imaging descriptions to constrain the semantic distribution of the visual modality. The above pre-training dataset is called Triad-131K, which is currently the largest 3D MRI pre-training dataset. We evaluate Triad across three tasks, namely, organ/tumor segmentation, organ/cancer classification, and medical image registration, in two data modalities (within-domain and out-of-domain) settings using 25 downstream datasets. By initializing models with Triad's pre-trained weights, nnUNet-Triad improves segmentation performance by 6.88% compared to nnUNet-Scratch across 17 datasets. Swin-B-Triad achieves a 3.97% improvement over Swin-B-Scratch in classification tasks across five datasets. SwinUNETR-Triad improves by 4.00% compared to SwinUNETR-Scratch in registration tasks across two datasets. Our study demonstrates that pre-training can maximize performance when the data modalities and organs of upstream and downstream tasks are consistent. 

**Abstract (ZH)**: 视觉基础模型（VFMs）在广泛的图像数据集上进行预训练，以学习适用于多种数据类型的通用表示。这些模型可以随后针对特定的下游任务进行微调，从而在众多应用场景中显著提高性能。然而，现有的宣称适用于多种放射学任务的视觉基础模型大多是在3D计算机断层扫描（CT）上进行的预训练，这得益于3D CT数据库的丰富性。CT与磁共振成像（MRI）在成像原理、信号特征和数据分布上的显著差异可能阻碍了这些模型在MRI特定应用中的实用性能和通用性。在这里，我们提出了一种针对3D MRI的视觉基础模型——Triad。Triad采用广泛使用的自动编码器架构，从131,170个3D MRI体素中学习稳健的表示，并使用器官无关的成像描述来约束视觉模态的语义分布。上述预训练数据集称为Triad-131K，目前是最大的3D MRI预训练数据集。我们在两种数据模态（同域和跨域）下使用25个下游数据集对Triad进行了三项任务——器官/肿瘤分割、器官/癌症分类和医疗影像配准——的评估。通过使用Triad预训练的权重初始化模型，nnUNet-Triad在17个数据集上的分割性能相比nnUNet-Scratch提高了6.88%。Swin-B-Triad在5个数据集的分类任务中比Swin-B-Scratch提高了3.97%。SwinUNETR-Triad在两个数据集的配准任务中比SwinUNETR-Scratch提高了4.00%。我们的研究显示，当上游和下游任务的数据模态和器官一致时，预训练可以最大化性能提升。 

---
# EfficientPose 6D: Scalable and Efficient 6D Object Pose Estimation 

**Title (ZH)**: EfficientPose 6D：可扩展且高效的6D物体姿态估计 

**Authors**: Zixuan Fang, Thomas Pöllabauer, Tristan Wirth, Sarah Berkei, Volker Knauthe, Arjan Kuijper  

**Link**: [PDF](https://arxiv.org/pdf/2502.14061)  

**Abstract**: In industrial applications requiring real-time feedback, such as quality control and robotic manipulation, the demand for high-speed and accurate pose estimation remains critical. Despite advances improving speed and accuracy in pose estimation, finding a balance between computational efficiency and accuracy poses significant challenges in dynamic environments. Most current algorithms lack scalability in estimation time, especially for diverse datasets, and the state-of-the-art (SOTA) methods are often too slow. This study focuses on developing a fast and scalable set of pose estimators based on GDRNPP to meet or exceed current benchmarks in accuracy and robustness, particularly addressing the efficiency-accuracy trade-off essential in real-time scenarios. We propose the AMIS algorithm to tailor the utilized model according to an application-specific trade-off between inference time and accuracy. We further show the effectiveness of the AMIS-based model choice on four prominent benchmark datasets (LM-O, YCB-V, T-LESS, and ITODD). 

**Abstract (ZH)**: 在需要实时反馈的工业应用中，如质量控制和机器人操作，对高精度和高速度的姿态估计需求仍然至关重要。尽管在姿态估计的速度和准确性方面取得了进展，但在动态环境中实现计算效率和准确性的平衡仍然面临重大挑战。目前大多数算法在估计时间上缺乏可扩展性，特别是在处理多样化数据集时，最先进的（SOTA）方法往往过于缓慢。本研究旨在开发基于GDRNPP的快速且可扩展的姿态估算器，以在准确性和鲁棒性上达到或超越当前基准，并特别关注在实时场景中效率-准确性的权衡。我们提出了AMIS算法，根据具体应用之间的推理时间和准确性之间的权衡来调整所使用的模型。此外，我们还展示了基于AMIS的模型选择在四个知名基准数据集（LM-O、YCB-V、T-LESS和ITODD）上的有效性。 

---
# Diversity-driven Data Selection for Language Model Tuning through Sparse Autoencoder 

**Title (ZH)**: 基于多样性驱动的数据选择语言模型调优通过稀疏自编码器 

**Authors**: Xianjun Yang, Shaoliang Nie, Lijuan Liu, Suchin Gururangan, Ujjwal Karn, Rui Hou, Madian Khabsa, Yuning Mao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14050)  

**Abstract**: Current pre-trained large language models typically need instruction tuning to align with human preferences. However, instruction tuning data is often quantity-saturated due to the large volume of data collection and fast model iteration, leaving coreset data selection important but underexplored. On the other hand, existing quality-driven data selection methods such as LIMA (NeurIPS 2023 (Zhou et al., 2024)) and AlpaGasus (ICLR 2024 (Chen et al.)) generally ignore the equal importance of data diversity and complexity. In this work, we aim to design a diversity-aware data selection strategy and creatively propose using sparse autoencoders to tackle the challenge of data diversity measure. In addition, sparse autoencoders can also provide more interpretability of model behavior and explain, e.g., the surprising effectiveness of selecting the longest response (ICML 2024 (Zhao et al.)). Using effective data selection, we experimentally prove that models trained on our selected data can outperform other methods in terms of model capabilities, reduce training cost, and potentially gain more control over model behaviors. 

**Abstract (ZH)**: 当前的预训练大型语言模型通常需要指令微调以与人类偏好对齐。然而，由于数据收集量大和模型迭代速度快，指令微调数据往往已经饱和，使得核心集数据选择变得重要但尚未得到充分探索。另一方面，现有的以质量为导向的数据选择方法，如LIMA（NeurIPS 2023，Zhou et al., 2024）和AlpaGasus（ICLR 2024，Chen et al.），通常忽略了数据多样性和复杂性的重要性。在本工作中，我们旨在设计一种具有多样性的数据选择策略，并创造性地提出使用稀疏自编码器来解决数据多样性的度量挑战。此外，稀疏自编码器还可以提供对模型行为的更可解释性，并解释，例如选择最长响应的惊人效果（ICML 2024，Zhao et al.）。通过有效数据选择，我们实验证明，使用我们选择的数据训练的模型在模型能力上可优于其他方法，并降低训练成本，有可能获得更多对模型行为的控制。 

---
# Semantic Decomposition and Selective Context Filtering -- Text Processing Techniques for Context-Aware NLP-Based Systems 

**Title (ZH)**: 语义分解与选择性上下文过滤——基于上下文感知的自然语言处理系统中的文本处理技术 

**Authors**: Karl John Villardar  

**Link**: [PDF](https://arxiv.org/pdf/2502.14048)  

**Abstract**: In this paper, we present two techniques for use in context-aware systems: Semantic Decomposition, which sequentially decomposes input prompts into a structured and hierarchal information schema in which systems can parse and process easily, and Selective Context Filtering, which enables systems to systematically filter out specific irrelevant sections of contextual information that is fed through a system's NLP-based pipeline. We will explore how context-aware systems and applications can utilize these two techniques in order to implement dynamic LLM-to-system interfaces, improve an LLM's ability to generate more contextually cohesive user-facing responses, and optimize complex automated workflows and pipelines. 

**Abstract (ZH)**: 在本文中，我们提出了两种适用于上下文感知系统的技术：语义分解（Semantic Decomposition），该技术按顺序将输入提示分解为系统可以轻松解析和处理的结构化和层次化信息模式；以及选择性上下文过滤（Selective Context Filtering），该技术使系统能够系统地筛选掉特定的无关上下文信息，这些信息通过基于自然语言处理（NLP）的管道输入。我们将探讨如何利用这两种技术来实现动态的大语言模型（LLM）到系统的接口，提高大语言模型生成更具上下文连贯性用户响应的能力，并优化复杂的自动化工作流和管道。 

---
# Towards a Learning Theory of Representation Alignment 

**Title (ZH)**: 朝向表示对齐的学习理论 

**Authors**: Francesco Insulla, Shuo Huang, Lorenzo Rosasco  

**Link**: [PDF](https://arxiv.org/pdf/2502.14047)  

**Abstract**: It has recently been argued that AI models' representations are becoming aligned as their scale and performance increase. Empirical analyses have been designed to support this idea and conjecture the possible alignment of different representations toward a shared statistical model of reality. In this paper, we propose a learning-theoretic perspective to representation alignment. First, we review and connect different notions of alignment based on metric, probabilistic, and spectral ideas. Then, we focus on stitching, a particular approach to understanding the interplay between different representations in the context of a task. Our main contribution here is relating properties of stitching to the kernel alignment of the underlying representation. Our results can be seen as a first step toward casting representation alignment as a learning-theoretic problem. 

**Abstract (ZH)**: 近年来有观点认为，随着AI模型的规模和性能的提升，其表示也在趋向对齐。经验分析已被设计来支持这一观点，并推测不同表示可能朝着一个共同的现实统计模型对齐。本文从学习理论的角度探讨表示对齐。首先，我们回顾并联系基于度量、概率和谱的不同对齐概念。然后，我们聚焦于“缝合”（stitching）这一特定方法，用于理解不同表示在任务背景下的相互作用。我们在这里的主要贡献是将缝合的性质与底层表示的核对齐联系起来。我们的结果可以被视为将表示对齐转化为一个学习理论问题的第一步。 

---
# Position: There are no Champions in Long-Term Time Series Forecasting 

**Title (ZH)**: 位置：长期时间序列预测中不存在冠军模型 

**Authors**: Lorenzo Brigato, Rafael Morand, Knut Strømmen, Maria Panagiotou, Markus Schmidt, Stavroula Mougiakakou  

**Link**: [PDF](https://arxiv.org/pdf/2502.14045)  

**Abstract**: Recent advances in long-term time series forecasting have introduced numerous complex prediction models that consistently outperform previously published architectures. However, this rapid progression raises concerns regarding inconsistent benchmarking and reporting practices, which may undermine the reliability of these comparisons. Our position emphasizes the need to shift focus away from pursuing ever-more complex models and towards enhancing benchmarking practices through rigorous and standardized evaluation methods. To support our claim, we first perform a broad, thorough, and reproducible evaluation of the top-performing models on the most popular benchmark by training 3,500+ networks over 14 datasets. Then, through a comprehensive analysis, we find that slight changes to experimental setups or current evaluation metrics drastically shift the common belief that newly published results are advancing the state of the art. Our findings suggest the need for rigorous and standardized evaluation methods that enable more substantiated claims, including reproducible hyperparameter setups and statistical testing. 

**Abstract (ZH)**: 近期长序列时间序列预测领域的进步引入了大量复杂预测模型，这些模型在性能上持续超越先前发表的架构。然而，这种快速的进步引发了一种担忧，即不一致的基准测试和报告实践可能削弱这些比较的可靠性。我们的观点强调，应将重点从追求更加复杂的模型转移到通过严格的标准化评估方法改进基准测试实践上。为支持这一观点，我们首先在最受欢迎的基准上对表现最佳的模型进行了广泛、彻底且可复制的评估，通过在14个数据集上训练3,500多个网络来完成这项评估。然后，通过综合分析我们发现，即使是实验设置或当前评估指标的轻微变化，也会极大地改变人们普遍认为新发表的结果在推进前沿技术方面有所提升的认知。研究结果表明，需要引入严格的标准化评估方法，以促进更有根据的声明，包括可复现实验超参数设置和统计测试。 

---
# Asking for Help Enables Safety Guarantees Without Sacrificing Effectiveness 

**Title (ZH)**: 请求协助可以在不牺牲有效性的情况下提供安全性保证 

**Authors**: Benjamin Plaut, Juan Liévano-Karim, Stuart Russell  

**Link**: [PDF](https://arxiv.org/pdf/2502.14043)  

**Abstract**: Most reinforcement learning algorithms with regret guarantees rely on a critical assumption: that all errors are recoverable. Recent work by Plaut et al. discarded this assumption and presented algorithms that avoid "catastrophe" (i.e., irreparable errors) by asking for help. However, they provided only safety guarantees and did not consider reward maximization. We prove that any algorithm that avoids catastrophe in their setting also guarantees high reward (i.e., sublinear regret) in any Markov Decision Process (MDP), including MDPs with irreversible costs. This constitutes the first no-regret guarantee for general MDPs. More broadly, our result may be the first formal proof that it is possible for an agent to obtain high reward while becoming self-sufficient in an unknown, unbounded, and high-stakes environment without causing catastrophe or requiring resets. 

**Abstract (ZH)**: 大多数具有后悔保证的强化学习算法依赖于一个关键假设：所有错误都是可恢复的。Plaut等人近期的工作放弃了这一假设，并提出了通过寻求帮助来避免“灾难”（即不可修复的错误）的算法。然而，他们的工作只提供了安全保证，而未考虑奖励最大化的问题。我们证明，在他们的设定中避免灾难的任意算法在任何马尔可夫决策过程（MDP）中都能保证高奖励（即亚线性后悔），包括具有不可逆成本的MDP。这构成了对一般MDP的第一个无后悔保证。更广泛地说，我们的结果可能是第一个形式证明，即在一个未知、未界定且高风险的环境中，代理可以在不造成灾难或需要重置的情况下变得自给自足并获得高奖励。 

---
# DiffSampling: Enhancing Diversity and Accuracy in Neural Text Generation 

**Title (ZH)**: DiffSampling: 提升神经文本生成中的多样性和准确性 

**Authors**: Giorgio Franceschelli, Mirco Musolesi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14037)  

**Abstract**: Despite their increasing performance, large language models still tend to reproduce training data, generate several repetitions, and focus on the most common grammatical structures and words. A possible cause is the decoding strategy adopted: the most common ones either consider only the most probable tokens, reducing output diversity, or increase the likelihood of unlikely tokens at the cost of output accuracy and correctness. In this paper, we propose a family of three new decoding methods by leveraging a mathematical analysis of the token probability distribution. In particular, the difference between consecutive, sorted probabilities can be used to avoid incorrect tokens and increase the chance of low-probable but accurate words. Experiments concerning math problem solving, extreme summarization, and the divergent association task show that our approach consistently performs at least as well as current alternatives in terms of quality and diversity. 

**Abstract (ZH)**: 尽管大型语言模型在性能上不断提高，但仍倾向于复制训练数据、产生多次重复，并且重点关注最常见的语法结构和词汇。一个可能的原因是采用的解码策略：最常见的解码策略要么仅考虑最有可能的标记，从而减少输出的多样性，要么增加不太可能的标记的机会，但代价是降低了输出的准确性和正确性。在这篇论文中，我们提出了一种新的解码方法家族，通过利用标记概率分布的数学分析。具体来说，连续排好序的概率之间的差异可以用于避免错误的标记，增加低概率但准确的词汇出现的机会。关于数学问题解决、极端摘要和发散关联任务的实验表明，我们的方法在质量和多样性方面至少与当前替代方法一样好。 

---
# Dynamic Activation with Knowledge Distillation for Energy-Efficient Spiking NN Ensembles 

**Title (ZH)**: 知识蒸馏用于提高能效的脉冲神经网络集成中的动态激活方法 

**Authors**: Orestis Konstantaropoulos, Theodoris Mallios, Maria Papadopouli  

**Link**: [PDF](https://arxiv.org/pdf/2502.14023)  

**Abstract**: While foundation AI models excel at tasks like classification and decision-making, their high energy consumption makes them unsuitable for energy-constrained applications. Inspired by the brain's efficiency, spiking neural networks (SNNs) have emerged as a viable alternative due to their event-driven nature and compatibility with neuromorphic chips. This work introduces a novel system that combines knowledge distillation and ensemble learning to bridge the performance gap between artificial neural networks (ANNs) and SNNs. A foundation AI model acts as a teacher network, guiding smaller student SNNs organized into an ensemble, called Spiking Neural Ensemble (SNE). SNE enables the disentanglement of the teacher's knowledge, allowing each student to specialize in predicting a distinct aspect of it, while processing the same input. The core innovation of SNE is the adaptive activation of a subset of SNN models of an ensemble, leveraging knowledge-distillation, enhanced with an informed-partitioning (disentanglement) of the teacher's feature space. By dynamically activating only a subset of these student SNNs, the system balances accuracy and energy efficiency, achieving substantial energy savings with minimal accuracy loss. Moreover, SNE is significantly more efficient than the teacher network, reducing computational requirements by up to 20x with only a 2% drop in accuracy on the CIFAR-10 dataset. This disentanglement procedure achieves an accuracy improvement of up to 2.4% on the CIFAR-10 dataset compared to other partitioning schemes. Finally, we comparatively analyze SNE performance under noisy conditions, demonstrating enhanced robustness compared to its ANN teacher. In summary, SNE offers a promising new direction for energy-constrained applications. 

**Abstract (ZH)**: 虽然基础人工智能模型在分类和决策任务方面表现出色，但它们的高能耗使其不适合受能耗限制的应用。受到大脑高效性的启发，基于事件的神经网络（SNNs）因其事件驱动的特性以及与神经形态芯片的兼容性而成为一种可行的替代方案。本文引入了一种新颖的系统，该系统结合了知识蒸馏和集成学习，以弥合人工神经网络（ANNs）和SNNs之间的性能差距。一个基础AI模型作为教师网络，指导一个组织成集的较小的学生SNNs，称为Spike Neural Ensemble（SNE）。SNE实现了教师知识的去耦合，使每个学生能够专注于预测它的某一特定方面，同时处理相同的输入。SNE的核心创新在于，通过知识蒸馏和教师特征空间的明智分区（去耦合），动态激活集合中的一组SNN模型。通过仅动态激活这些学生SNN模型的一部分，系统在保持高准确率的同时实现了显著的能量节约。此外，与教师网络相比，SNE在计算要求上提高了20倍的效率，并在CIFAR-10数据集上仅损失2%的准确率。此去耦合过程在CIFAR-10数据集上相比于其他分区方案实现了高达2.4%的准确率提升。最后，我们比较分析了在噪声条件下SNE的表现，显示了其增强的鲁棒性。总之，SNE为受能耗限制的应用提供了一个有前途的新方向。 

---
# Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems 

**Title (ZH)**: 人性化的机器：减轻文本生成系统中的拟人化行为 

**Authors**: Myra Cheng, Su Lin Blodgett, Alicia DeVrio, Lisa Egede, Alexandra Olteanu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14019)  

**Abstract**: As text generation systems' outputs are increasingly anthropomorphic -- perceived as human-like -- scholars have also raised increasing concerns about how such outputs can lead to harmful outcomes, such as users over-relying or developing emotional dependence on these systems. How to intervene on such system outputs to mitigate anthropomorphic behaviors and their attendant harmful outcomes, however, remains understudied. With this work, we aim to provide empirical and theoretical grounding for developing such interventions. To do so, we compile an inventory of interventions grounded both in prior literature and a crowdsourced study where participants edited system outputs to make them less human-like. Drawing on this inventory, we also develop a conceptual framework to help characterize the landscape of possible interventions, articulate distinctions between different types of interventions, and provide a theoretical basis for evaluating the effectiveness of different interventions. 

**Abstract (ZH)**: 随着文本生成系统的输出越来越具有人性化的特征，被视作类似人类的产出，学者们也越来越多地担忧这些输出可能带来的负面影响，如用户过度依赖这些系统或对这些系统产生情感依赖。然而，如何干预这些系统输出以减轻其人性化的特征及其潜在的负面影响，这一领域仍缺乏研究。通过这项研究，我们旨在为开发此类干预措施提供实证和理论基础。为此，我们整理了一份基于前人研究和一项众包研究的干预措施清单。在众包研究中，参与者对系统输出进行了编辑，以使其更不具人性化特征。基于这份清单，我们还开发了一个概念框架，以帮助描述可能的干预措施的景观、阐述不同类型的干预措施之间的区别，并为评估不同干预措施的有效性提供理论依据。 

---
# Appeal prediction for AI up-scaled Images 

**Title (ZH)**: AI放大数据的引诉预测 

**Authors**: Steve Göring, Rasmus Merten, Alexander Raake  

**Link**: [PDF](https://arxiv.org/pdf/2502.14013)  

**Abstract**: DNN- or AI-based up-scaling algorithms are gaining in popularity due to the improvements in machine learning. Various up-scaling models using CNNs, GANs or mixed approaches have been published. The majority of models are evaluated using PSRN and SSIM or only a few example images. However, a performance evaluation with a wide range of real-world images and subjective evaluation is missing, which we tackle in the following paper. For this reason, we describe our developed dataset, which uses 136 base images and five different up-scaling methods, namely Real-ESRGAN, BSRGAN, waifu2x, KXNet, and Lanczos. Overall the dataset consists of 1496 annotated images. The labeling of our dataset focused on image appeal and has been performed using crowd-sourcing employing our open-source tool AVRate Voyager. We evaluate the appeal of the different methods, and the results indicate that Real-ESRGAN and BSRGAN are the best. Furthermore, we train a DNN to detect which up-scaling method has been used, the trained models have a good overall performance in our evaluation. In addition to this, we evaluate state-of-the-art image appeal and quality models, here none of the models showed a high prediction performance, therefore we also trained two own approaches. The first uses transfer learning and has the best performance, and the second model uses signal-based features and a random forest model with good overall performance. We share the data and implementation to allow further research in the context of open science. 

**Abstract (ZH)**: 基于DNN或AI的上尺度算法由于机器学习的进步而逐渐流行。使用CNN、GAN或混合方法的各种上尺度模型已经发表。大多数模型的评估通常使用PSNR和SSIM指标，或者仅限于少量示例图像。然而，缺乏对广泛真实世界图像的性能评估以及主观评价，这在接下来的研究中得到了解决。因此，我们描述了我们开发的包含136个基础图像和五种不同的上尺度方法（即Real-ESRGAN、BSRGAN、waifu2x、KXNet、Lanczos）的数据集。整个数据集共包含1496张标注图像。我们的数据集的标注重点在于图像吸引力，并使用众包方法通过我们开源工具AVRate Voyager进行。我们评估了不同方法的吸引力，结果表明Real-ESRGAN和BSRGAN表现最佳。此外，我们训练了一个DNN来检测使用了哪种上尺度方法，所训练的模型在我们的评估中总体表现良好。此外，我们还评估了最先进的图像吸引力和质量模型，但这些模型的预测性能并不高，因此我们还训练了两种自己的方法。第一种使用了迁移学习，具有最好的性能；第二种模型使用了基于信号的特征和随机森林模型，整体性能良好。我们共享了数据和实现，以在开放科学的背景下促进进一步的研究。 

---
# DFDT: Dynamic Fast Decision Tree for IoT Data Stream Mining on Edge Devices 

**Title (ZH)**: DFDT：边缘设备上用于物联网数据流挖掘的动态快速决策树 

**Authors**: Afonso Lourenço, João Rodrigo, João Gama, Goreti Marreiros  

**Link**: [PDF](https://arxiv.org/pdf/2502.14011)  

**Abstract**: The Internet of Things generates massive data streams, with edge computing emerging as a key enabler for online IoT applications and 5G networks. Edge solutions facilitate real-time machine learning inference, but also require continuous adaptation to concept drifts. Ensemble-based solutions improve predictive performance, but incur higher resource consumption, latency, and memory demands. This paper presents DFDT: Dynamic Fast Decision Tree, a novel algorithm designed for energy-efficient memory-constrained data stream mining. DFDT improves hoeffding tree growth efficiency by dynamically adjusting grace periods, tie thresholds, and split evaluations based on incoming data. It incorporates stricter evaluation rules (based on entropy, information gain, and leaf instance count), adaptive expansion modes, and a leaf deactivation mechanism to manage memory, allowing more computation on frequently visited nodes while conserving energy on others. Experiments show that the proposed framework can achieve increased predictive performance (0.43 vs 0.29 ranking) with constrained memory and a fraction of the runtime of VFDT or SVFDT. 

**Abstract (ZH)**: 物联网生成了大量数据流，边缘计算作为在线物联网应用和5G网络的关键使能器正在兴起。边缘解决方案促进了实时机器学习推理，但也需要持续适应概念漂移。基于集成的方法可以提高预测性能，但会增加更高的资源消耗、延迟和内存需求。本文介绍了一种新的算法DFDT：动态快速决策树（Dynamic Fast Decision Tree），它旨在为受内存约束的数据流挖掘提供能源高效的方法。DFDT通过根据流入数据动态调整宽限期、平局阈值和分裂评估，提高了霍夫丁树的增长效率。它结合了更严格的评估规则（基于熵、信息增益和叶节点实例计数）、自适应扩展模式以及叶节点失活机制，以管理内存，使得在频繁访问的节点上进行更多的计算，而在其他节点上则节省能量。实验表明，所提出的方法在受内存约束的情况下，相比VFDT或SVFDT，能够在更短的运行时间内实现更高的预测性能（排名从0.29提高到0.43）。 

---
# Which Attention Heads Matter for In-Context Learning? 

**Title (ZH)**: 影响上下文学习的哪些注意力头重要？ 

**Authors**: Kayo Yin, Jacob Steinhardt  

**Link**: [PDF](https://arxiv.org/pdf/2502.14010)  

**Abstract**: Large language models (LLMs) exhibit impressive in-context learning (ICL) capability, enabling them to perform new tasks using only a few demonstrations in the prompt. Two different mechanisms have been proposed to explain ICL: induction heads that find and copy relevant tokens, and function vector (FV) heads whose activations compute a latent encoding of the ICL task. To better understand which of the two distinct mechanisms drives ICL, we study and compare induction heads and FV heads in 12 language models.
Through detailed ablations, we discover that few-shot ICL performance depends primarily on FV heads, especially in larger models. In addition, we uncover that FV and induction heads are connected: many FV heads start as induction heads during training before transitioning to the FV mechanism. This leads us to speculate that induction facilitates learning the more complex FV mechanism that ultimately drives ICL. 

**Abstract (ZH)**: 大型语言模型（LLMs）展示了令人印象深刻的上下文内学习（ICL）能力，使得它们能够在仅通过少量提示示例的情况下执行新任务。已提出了两种不同的机制来解释ICL：感应头（induction heads），它们寻找并复制相关令牌；以及功能向量（FV）头，其激活计算ICL任务的潜在编码。为了更好地了解哪一种机制驱动ICL，我们对12种语言模型中的感应头和FV头进行了研究与比较。

通过详细的消除实验，我们发现，少量示例ICL性能主要依赖于FV头，特别是在较大的模型中更是如此。此外，我们发现FV头和感应头之间存在联系：许多FV头在训练期间最初作为感应头存在，之后才转换为FV机制。这一发现使我们推测，感应头有助于学习更复杂的FV机制，最终驱动ICL。 

---
# MaskPrune: Mask-based LLM Pruning for Layer-wise Uniform Structures 

**Title (ZH)**: MaskPrune：基于掩码的层间均匀结构的LLM剪枝 

**Authors**: Jiayu Qin, Jianchao Tan, Kefeng Zhang, Xunliang Cai, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14008)  

**Abstract**: The remarkable performance of large language models (LLMs) in various language tasks has attracted considerable attention. However, the ever-increasing size of these models presents growing challenges for deployment and inference. Structured pruning, an effective model compression technique, is gaining increasing attention due to its ability to enhance inference efficiency. Nevertheless, most previous optimization-based structured pruning methods sacrifice the uniform structure across layers for greater flexibility to maintain performance. The heterogeneous structure hinders the effective utilization of off-the-shelf inference acceleration techniques and impedes efficient configuration for continued training. To address this issue, we propose a novel masking learning paradigm based on minimax optimization to obtain the uniform pruned structure by optimizing the masks under sparsity regularization. Extensive experimental results demonstrate that our method can maintain high performance while ensuring the uniformity of the pruned model structure, thereby outperforming existing SOTA methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种语言任务中表现出色，引起了广泛的关注。然而，这些模型不断扩大的规模给部署和推理带来了越来越大的挑战。结构化剪枝作为一种有效的模型压缩技术，因其能够提升推理效率而受到越来越多的关注。然而，大多数基于优化的结构化剪枝方法为了获得更大的灵活性而在各层之间牺牲了一致性结构。这种异构结构阻碍了现成的推理加速技术的有效利用，并妨碍了模型持续训练的高效配置。为了解决这一问题，我们提出了一种基于最小-最大优化的新型掩码学习范式，通过稀疏正则化优化掩码以获得一致性的剪枝结构。广泛的实验结果表明，我们的方法能够维持高性能的同时确保剪枝模型结构的一致性，从而优于现有最先进的方法。 

---
# Rectified Lagrangian for Out-of-Distribution Detection in Modern Hopfield Networks 

**Title (ZH)**: 修正后的拉格朗日乘子在现代霍普菲尔德网络中异常分布检测中的应用 

**Authors**: Ryo Moriai, Nakamasa Inoue, Masayuki Tanaka, Rei Kawakami, Satoshi Ikehata, Ikuro Sato  

**Link**: [PDF](https://arxiv.org/pdf/2502.14003)  

**Abstract**: Modern Hopfield networks (MHNs) have recently gained significant attention in the field of artificial intelligence because they can store and retrieve a large set of patterns with an exponentially large memory capacity. A MHN is generally a dynamical system defined with Lagrangians of memory and feature neurons, where memories associated with in-distribution (ID) samples are represented by attractors in the feature space. One major problem in existing MHNs lies in managing out-of-distribution (OOD) samples because it was originally assumed that all samples are ID samples. To address this, we propose the rectified Lagrangian (RegLag), a new Lagrangian for memory neurons that explicitly incorporates an attractor for OOD samples in the dynamical system of MHNs. RecLag creates a trivial point attractor for any interaction matrix, enabling OOD detection by identifying samples that fall into this attractor as OOD. The interaction matrix is optimized so that the probability densities can be estimated to identify ID/OOD. We demonstrate the effectiveness of RecLag-based MHNs compared to energy-based OOD detection methods, including those using state-of-the-art Hopfield energies, across nine image datasets. 

**Abstract (ZH)**: 现代霍普菲尔德网络（MHNs）近年来在人工智能领域引起了广泛关注，因为它们能够存储和检索大量模式，并具有指数级大的存储容量。MHN 通常定义为带有记忆神经元和特征神经元拉格朗日量的动态系统，其中与在分布（ID）样本相关的记忆在特征空间中表示为吸引子。现有 MHN 中的一个主要问题是管理非分布（OOD）样本，因为最初假设所有样本都是 ID 样本。为了解决这个问题，我们提出了一种新的记忆神经元拉格朗日方法——校正拉格朗日（RegLag），它明确地在 MHN 的动态系统中增加了 OOD 样本的吸引子。RegLag 为任何交互矩阵创建了一个平凡的点吸引子，通过识别落入该吸引子的样本作为 OOD 样本，实现了 OOD 检测。交互矩阵被优化，以便可以通过估计概率密度来识别 ID 和 OOD 样本。我们通过在九个图像数据集上将基于 RegLag 的 MHN 与基于能量的方法（包括最先进的霍普菲尔德能量方法）进行 OOD 检测进行比较，展示了其有效性。 

---
# Towards a perturbation-based explanation for medical AI as differentiable programs 

**Title (ZH)**: 基于扰动的解释方法探索：医学人工智能可微程序中的扰动解释 

**Authors**: Takeshi Abe, Yoshiyuki Asai  

**Link**: [PDF](https://arxiv.org/pdf/2502.14001)  

**Abstract**: Recent advancement in machine learning algorithms reaches a point where medical devices can be equipped with artificial intelligence (AI) models for diagnostic support and routine automation in clinical settings. In medicine and healthcare, there is a particular demand for sufficient and objective explainability of the outcome generated by AI models. However, AI models are generally considered as black boxes due to their complexity, and the computational process leading to their response is often opaque. Although several methods have been proposed to explain the behavior of models by evaluating the importance of each feature in discrimination and prediction, they may suffer from biases and opacities arising from the scale and sampling protocol of the dataset used for training or testing. To overcome the shortcomings of existing methods, we explore an alternative approach to provide an objective explanation of AI models that can be defined independently of the learning process and does not require additional data. As a preliminary study for this direction of research, this work examines a numerical availability of the Jacobian matrix of deep learning models that measures how stably a model responses against small perturbations added to the input. The indicator, if available, are calculated from a trained AI model for a given target input. This is a first step towards a perturbation-based explanation, which will assist medical practitioners in understanding and interpreting the response of the AI model in its clinical application. 

**Abstract (ZH)**: 最近机器学习算法的发展已经达到了一个阶段，医疗设备可以配备人工智能（AI）模型，以支持诊断并在临床环境中实现常规自动化。在医学和医疗保健领域，对由AI模型生成的结果的充分且客观的解释性有着特别的需求。然而，由于其复杂性，AI模型通常被视为黑箱，其对应回应的计算过程往往不清楚。尽管已经提出了一些方法来通过评估每个特征在区分和预测中的重要性来解释模型的行为，但这些方法可能会因为用于训练或测试的数据集规模和采样协议而产生偏差和不透明性。

为了克服现有方法的局限性，我们探索了一种替代方法，以提供独立于学习过程的客观解释，且不需要额外数据。作为这一研究方向的初步研究，这项工作检查了用于度量模型在输入上添加小扰动后响应稳定性的大规模雅可比矩阵的数字可用性。如果该指标可用，它将从给定的目标输入中训练的AI模型中计算得出。这代表了基于扰动的解释的第一步，将帮助医疗从业者理解并解释AI模型在临床应用中的响应。 

---
# Human-Artificial Interaction in the Age of Agentic AI: A System-Theoretical Approach 

**Title (ZH)**: agency人工智能时代的个体-人工交互：系统理论视角 

**Authors**: Uwe M. Borghoff, Paolo Bottoni, Remo Pareschi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14000)  

**Abstract**: This paper presents a novel perspective on human-computer interaction (HCI), framing it as a dynamic interplay between human and computational agents within a networked system. Going beyond traditional interface-based approaches, we emphasize the importance of coordination and communication among heterogeneous agents with different capabilities, roles, and goals. A key distinction is made between multi-agent systems (MAS) and Centaurian systems, which represent two different paradigms of human-AI collaboration. MAS maintain agent autonomy, with structured protocols enabling cooperation, while Centaurian systems deeply integrate human and AI capabilities, creating unified decision-making entities.
To formalize these interactions, we introduce a framework for communication spaces, structured into surface, observation, and computation layers, ensuring seamless integration between MAS and Centaurian architectures, where colored Petri nets effectively represent structured Centaurian systems and high-level reconfigurable networks address the dynamic nature of MAS.
Our research has practical applications in autonomous robotics, human-in-the-loop decision making, and AI-driven cognitive architectures, and provides a foundation for next-generation hybrid intelligence systems that balance structured coordination with emergent behavior. 

**Abstract (ZH)**: 本文从一个新的角度探讨了人机交互（HCI），将其视为网络系统中人类代理与计算代理之间动态互动的过程。超越传统的基于界面的方法，我们强调了不同能力、角色和目标的异质代理之间协调与沟通的重要性。我们区分了多代理系统（MAS）和赛博坦（Centaurian）系统，这两种系统代表了人与人工智能合作的两种不同范式。MAS 保持代理的自主性，通过结构化的协议实现合作，而Centaurian系统则深入整合了人类和人工智能的能力，创造了统一的决策实体。

为了正式化这些互动，我们提出了一种通信空间框架，分为表面、观察和计算三层，确保MAS和Centaurian架构之间的无缝集成，其中带颜色的Petri网有效代表了结构化的Centaurian系统，而高级可重构网络则解决了MAS的动态特性。

我们的研究在自主机器人、人参与决策以及人工智能驱动的认知架构方面具有实际应用，并为下一代混合智能系统提供了基础，这些系统平衡了结构化协调与涌现行为之间的关系。 

---
# A Baseline Method for Removing Invisible Image Watermarks using Deep Image Prior 

**Title (ZH)**: 使用深度图像先验去除不可见图像水印的一种基线方法 

**Authors**: Hengyue Liang, Taihui Li, Ju Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.13998)  

**Abstract**: Image watermarks have been considered a promising technique to help detect AI-generated content, which can be used to protect copyright or prevent fake image abuse. In this work, we present a black-box method for removing invisible image watermarks, without the need of any dataset of watermarked images or any knowledge about the watermark system. Our approach is simple to implement: given a single watermarked image, we regress it by deep image prior (DIP). We show that from the intermediate steps of DIP one can reliably find an evasion image that can remove invisible watermarks while preserving high image quality. Due to its unique working mechanism and practical effectiveness, we advocate including DIP as a baseline invasion method for benchmarking the robustness of watermarking systems. Finally, by showing the limited ability of DIP and other existing black-box methods in evading training-based visible watermarks, we discuss the positive implications on the practical use of training-based visible watermarks to prevent misinformation abuse. 

**Abstract (ZH)**: 图像水印被认为是一种有希望的技术，有助于检测AI生成的内容，可用于保护版权或防止伪造图像滥用。在本文中，我们提出了一种黑盒方法，用于移除不可见的图像水印，无需任何已标记水印图像的数据集，也无需了解水印系统的信息。我们的方法易于实现：给定一张标记过的图像，我们通过深度图像先验（DIP）进行回归。我们表明，从DIP的中间步骤中可以可靠地找到一种规避图像，能够移除不可见水印同时保持高图像质量。由于其独特的运作机制及其实际有效性，我们建议将DIP纳入基准测试水印系统的鲁棒性评估中的基线入侵方法。最后，通过展示DIP和其他现有黑盒方法在规避基于训练的可见水印方面的有限能力，我们讨论了基于训练的可见水印在防止虚假信息滥用方面的实际应用意义。 

---
# Generative Detail Enhancement for Physically Based Materials 

**Title (ZH)**: 基于物理原理的材料生成性细节增强 

**Authors**: Saeed Hadadan, Benedikt Bitterli, Tizian Zeltner, Jan Novák, Fabrice Rousselle, Jacob Munkberg, Jon Hasselgren, Bartlomiej Wronski, Matthias Zwicker  

**Link**: [PDF](https://arxiv.org/pdf/2502.13994)  

**Abstract**: We present a tool for enhancing the detail of physically based materials using an off-the-shelf diffusion model and inverse rendering. Our goal is to enhance the visual fidelity of materials with detail that is often tedious to author, by adding signs of wear, aging, weathering, etc. As these appearance details are often rooted in real-world processes, we leverage a generative image model trained on a large dataset of natural images with corresponding visuals in context. Starting with a given geometry, UV mapping, and basic appearance, we render multiple views of the object. We use these views, together with an appearance-defining text prompt, to condition a diffusion model. The details it generates are then backpropagated from the enhanced images to the material parameters via inverse differentiable rendering. For inverse rendering to be successful, the generated appearance has to be consistent across all the images. We propose two priors to address the multi-view consistency of the diffusion model. First, we ensure that the initial noise that seeds the diffusion process is itself consistent across views by integrating it from a view-independent UV space. Second, we enforce geometric consistency by biasing the attention mechanism via a projective constraint so that pixels attend strongly to their corresponding pixel locations in other views. Our approach does not require any training or finetuning of the diffusion model, is agnostic of the material model used, and the enhanced material properties, i.e., 2D PBR textures, can be further edited by artists. 

**Abstract (ZH)**: 我们提出了一种工具，利用现成的扩散模型和逆渲染技术来增强基于物理的材质细节。我们的目标是通过增加磨损、老化和风化等细节，提升材料在视觉上的真实度，特别是那些常常难以手工编著的细节。由于这些外观细节往往根植于现实世界的过程，我们利用一种在大量自然图像及其上下文外观数据上进行训练的生成图像模型。通过给定几何形状、UV展开图和基本外观，我们渲染对象的多个视图。这些视图与一个定义外观的文本提示一起，用于条件化扩散模型。生成的细节随后通过逆可微渲染从增强后的图像反向传播到材质参数中。为了实现逆渲染的成功，生成的外观必须在所有图像中保持一致。我们提出两种先验知识来解决扩散模型的多视图一致性问题。首先，我们确保生成扩散过程初始噪声的一致性，并将其从与视图无关的UV空间中提取出来；其次，我们通过投影约束偏置注意力机制，使像素强烈关注其他视图中对应的像素位置，从而确保几何一致性。我们的方法不需要对扩散模型进行任何训练或微调，也不依赖于所使用的材质模型，并且增强后的材质属性，例如2D物理材质纹理，还可以由艺术家进一步编辑。 

---
# Learning to Discover Regulatory Elements for Gene Expression Prediction 

**Title (ZH)**: 学习发现调控元件以预测基因表达 

**Authors**: Xingyu Su, Haiyang Yu, Degui Zhi, Shuiwang Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.13991)  

**Abstract**: We consider the problem of predicting gene expressions from DNA sequences. A key challenge of this task is to find the regulatory elements that control gene expressions. Here, we introduce Seq2Exp, a Sequence to Expression network explicitly designed to discover and extract regulatory elements that drive target gene expression, enhancing the accuracy of the gene expression prediction. Our approach captures the causal relationship between epigenomic signals, DNA sequences and their associated regulatory elements. Specifically, we propose to decompose the epigenomic signals and the DNA sequence conditioned on the causal active regulatory elements, and apply an information bottleneck with the Beta distribution to combine their effects while filtering out non-causal components. Our experiments demonstrate that Seq2Exp outperforms existing baselines in gene expression prediction tasks and discovers influential regions compared to commonly used statistical methods for peak detection such as MACS3. The source code is released as part of the AIRS library (this https URL). 

**Abstract (ZH)**: 我们将研究重点放在从DNA序列预测基因表达的问题上。这一任务的关键挑战在于识别控制基因表达的调控元件。本文介绍了一个名为Seq2Exp的序列到表达网络，该网络明确设计用于发现和提取驱动目标基因表达的调控元件，从而提升基因表达预测的准确性。我们的方法捕捉了表观基因组信号、DNA序列与其相关调控元件之间的因果关系。具体而言，我们提出了一种方法，即在因果活跃调控元件的条件下分解表观基因组信号和DNA序列，并采用带有Beta分布的信息瓶颈来结合它们的影响，同时过滤掉非因果成分。实验结果表明，Seq2Exp在基因表达预测任务中优于现有基线，并且与常用的峰值检测统计方法（如MACS3）相比，能够发现具有影响的区域。源代码已作为AIRS库的一部分发布（详见链接：https://example.com）。 

---
# Gesture-Aware Zero-Shot Speech Recognition for Patients with Language Disorders 

**Title (ZH)**: 患有语言障碍患者的无监督手势引导语音识别 

**Authors**: Seungbae Kim, Daeun Lee, Brielle Stark, Jinyoung Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.13983)  

**Abstract**: Individuals with language disorders often face significant communication challenges due to their limited language processing and comprehension abilities, which also affect their interactions with voice-assisted systems that mostly rely on Automatic Speech Recognition (ASR). Despite advancements in ASR that address disfluencies, there has been little attention on integrating non-verbal communication methods, such as gestures, which individuals with language disorders substantially rely on to supplement their communication. Recognizing the need to interpret the latent meanings of visual information not captured by speech alone, we propose a gesture-aware ASR system utilizing a multimodal large language model with zero-shot learning for individuals with speech impairments. Our experiment results and analyses show that including gesture information significantly enhances semantic understanding. This study can help develop effective communication technologies, specifically designed to meet the unique needs of individuals with language impairments. 

**Abstract (ZH)**: 语言障碍个体由于其有限的语言处理和理解能力常常面临重大的沟通挑战，这也影响了他们与主要依赖自动语音识别（ASR）的语音辅助系统的互动。尽管在ASR方面已经取得了进步以解决口吃等问题，但尚未过多关注整合非言语沟通方法（如手势），这些方法对于语言障碍个体补充沟通至关重要。为了解释仅靠语音无法捕捉到的潜在视觉信息含义的需要，我们提出了一种利用多模态大型语言模型并结合零样本学习的手势感知ASR系统，以满足言语障碍个体的特殊需求。实验结果和分析表明，纳入手势信息显著提高了语义理解。本研究有助于开发有效的沟通技术，特别是针对语言障碍个体的独特需求进行设计。 

---
# Utilizing Effective Dynamic Graph Learning to Shield Financial Stability from Risk Propagation 

**Title (ZH)**: 利用有效的动态图学习方法shield金融稳定免受风险传播的影响 

**Authors**: Guanyuan Yu, Qing Li, Yu Zhao, Jun Wang, YiJun Chen, Shaolei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.13979)  

**Abstract**: Financial risks can propagate across both tightly coupled temporal and spatial dimensions, posing significant threats to financial stability. Moreover, risks embedded in unlabeled data are often difficult to detect. To address these challenges, we introduce GraphShield, a novel approach with three key innovations: Enhanced Cross-Domain Infor mation Learning: We propose a dynamic graph learning module to improve information learning across temporal and spatial domains. Advanced Risk Recognition: By leveraging the clustering characteristics of risks, we construct a risk recognizing module to enhance the identification of hidden threats. Risk Propagation Visualization: We provide a visualization tool for quantifying and validating nodes that trigger widespread cascading risks. Extensive experiments on two real-world and two open-source datasets demonstrate the robust performance of our framework. Our approach represents a significant advancement in leveraging artificial intelligence to enhance financial stability, offering a powerful solution to mitigate the spread of risks within financial networks. 

**Abstract (ZH)**: 金融风险可以在紧密耦合的时空维度之间传播，对金融稳定性构成重大威胁。此外，嵌入未标记数据中的风险往往难以检测。为应对这些挑战，我们引入了GraphShield，这一创新方法包含三项关键改进：

1. **增强跨域信息学习**：我们提出一个动态图学习模块，以提高跨时空域的信息学习能力。

2. **先进的风险识别**：通过利用风险的聚类特性，我们构建了一个风险识别模块，以增强对潜藏威胁的识别。

3. **风险传播可视化**：我们提供一个可视化工具，用于量化和验证触发大规模链式风险的节点。

在两个真实世界和两个开源数据集上的广泛实验表明，该框架具有稳健的性能。我们的方法代表了利用人工智能增强金融稳定性的重大进展，提供了一个强有力的风险在网络中传播的缓解方案。 

---
# IncepFormerNet: A multi-scale multi-head attention network for SSVEP classification 

**Title (ZH)**: IncepFormerNet：一种用于SSVEP分类的多尺度多头注意力网络 

**Authors**: Yan Huang, Yongru Chen, Lei Cao, Yongnian Cao, Xuechun Yang, Yilin Dong, Tianyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.13972)  

**Abstract**: In recent years, deep learning (DL) models have shown outstanding performance in EEG classification tasks, particularly in Steady-State Visually Evoked Potential(SSVEP)-based Brain-Computer-Interfaces(BCI)systems. DL methods have been successfully applied to SSVEP-BCI. This study proposes a new model called IncepFormerNet, which is a hybrid of the Inception and Transformer architectures. IncepFormerNet adeptly extracts multi-scale temporal information from time series data using parallel convolution kernels of varying sizes, accurately capturing the subtle variations and critical features within SSVEP this http URL, the model integrates the multi-head attention mechanism from the Transformer architecture, which not only provides insights into global dependencies but also significantly enhances the understanding and representation of complex this http URL, it takes advantage of filter bank techniques to extract features based on the spectral characteristics of SSVEP data. To validate the effectiveness of the proposed model, we conducted experiments on two public datasets, . The experimental results show that IncepFormerNet achieves an accuracy of 87.41 on Dataset 1 and 71.97 on Dataset 2 using a 1.0-second time window. To further verify the superiority of the proposed model, we compared it with other deep learning models, and the results indicate that our method achieves significantly higher accuracy than the this http URL source codes in this work are available at: this https URL. 

**Abstract (ZH)**: 近年来，深度学习（DL）模型在脑电图（EEG）分类任务中表现出卓越的性能，特别是在基于稳态视觉诱发电位（SSVEP）的脑-机接口（BCI）系统中。成功地将DL方法应用于SSVEP-BCI系统中。本研究提出了一种新的模型，名为IncepFormerNet，该模型是Inception和Transformer架构的混合体。IncepFormerNet通过使用不同大小的并行卷积核有效地提取时间序列数据的多尺度时间信息，准确捕捉SSVEP中的细微变化和关键特征。该模型整合了从Transformer架构中引入的多头注意力机制，不仅提供了关于全局依赖性的见解，还极大地增强了对复杂结构的的理解和表示。利用滤波器银行技术，IncepFormerNet基于SSVEP数据的频谱特征提取特征。为了验证所提出模型的有效性，我们在两个公开数据集上进行了实验。实验结果显示，在1秒时间窗口内，IncepFormerNet在数据集1上达到了87.41%的准确率，在数据集2上达到了71.97%的准确率。为进一步验证所提模型的优势，我们将其与其他深度学习模型进行了比较，结果表明，我们的方法在准确率上显著高于其他方法。本工作中源代码可在以下网址获得：[此处应填写具体的网址]。 

---
# Bridging Simulation and Reality: A 3D Clustering-Based Deep Learning Model for UAV-Based RF Source Localization 

**Title (ZH)**: 仿真与现实之间的桥梁：基于3D聚类的深度学习模型在无人机无线电信号源定位中的应用 

**Authors**: Saad Masrur, Ismail Guvenc  

**Link**: [PDF](https://arxiv.org/pdf/2502.13969)  

**Abstract**: Localization of radio frequency (RF) sources has critical applications, including search and rescue, jammer detection, and monitoring of hostile activities. Unmanned aerial vehicles (UAVs) offer significant advantages for RF source localization (RFSL) over terrestrial methods, leveraging autonomous 3D navigation and improved signal capture at higher altitudes. Recent advancements in deep learning (DL) have further enhanced localization accuracy, particularly for outdoor scenarios. DL models often face challenges in real-world performance, as they are typically trained on simulated datasets that fail to replicate real-world conditions fully. To address this, we first propose the Enhanced Two-Ray propagation model, reducing the simulation-to-reality gap by improving the accuracy of propagation environment modeling. For RFSL, we propose the 3D Cluster-Based RealAdaptRNet, a DL-based method leveraging 3D clustering-based feature extraction for robust localization. Experimental results demonstrate that the proposed Enhanced Two-Ray model provides superior accuracy in simulating real-world propagation scenarios compared to conventional free-space and two-ray models. Notably, the 3D Cluster-Based RealAdaptRNet, trained entirely on simulated datasets, achieves exceptional performance when validated in real-world environments using the AERPAW physical testbed, with an average localization error of 18.2 m. The proposed approach is computationally efficient, utilizing 33.5 times fewer parameters, and demonstrates strong generalization capabilities across diverse trajectories, making it highly suitable for real-world applications. 

**Abstract (ZH)**: 射频（RF）源定位在搜索救援、干扰器检测以及敌对活动监控等方面具有关键应用价值。无人驾驶飞行器（UAV）在射频源定位（RFSL）方面相较于地面方法具有显著优势，利用了自主的三维导航能力和在较高海拔处改善的信号捕获能力。近年来，深度学习（DL）的进步进一步提升了定位精度，尤其是在户外场景中表现尤为明显。然而，DL模型在实际应用中往往面临挑战，因为它们通常是在无法完全复制现实环境的模拟数据集上进行训练的。为解决这一问题，我们首先提出了增强的双射线传播模型，通过改进传播环境建模提升了模拟与现实之间的差距。在射频源定位方面，我们提出了一种基于三维聚类的实适应RNet（3D Cluster-Based RealAdaptRNet），这是一种利用三维聚类特征提取方法以实现鲁棒定位的DL模型。实验结果表明，提出的增强的双射线模型在模拟真实的传播场景方面比传统的自由空间和双射线模型提供了更高的精度。值得注意的是，3D聚类基于实适应RNet完全在模拟数据集上训练，在使用AERPAW物理测试床在实际环境中验证时，其平均定位误差仅为18.2米。提出的这种方法在计算效率方面表现出色，参数量减少了33.5倍，并且具有很强的跨轨迹泛化能力，使其成为适用于实际应用的理想选择。 

---
