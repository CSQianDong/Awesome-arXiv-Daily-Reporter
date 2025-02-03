# Do LLMs Strategically Reveal, Conceal, and Infer Information? A Theoretical and Empirical Analysis in The Chameleon Game 

**Title (ZH)**: 大型语言模型（LLMs）在《变色龙博弈》中战略性地揭示、隐瞒和推断信息吗？理论与实证分析 

**Authors**: Mustafa O. Karabag, Ufuk Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2501.19398)  

**Abstract**: Large language model-based (LLM-based) agents have become common in settings that include non-cooperative parties. In such settings, agents' decision-making needs to conceal information from their adversaries, reveal information to their cooperators, and infer information to identify the other agents' characteristics. To investigate whether LLMs have these information control and decision-making capabilities, we make LLM agents play the language-based hidden-identity game, The Chameleon. In the game, a group of non-chameleon agents who do not know each other aim to identify the chameleon agent without revealing a secret. The game requires the aforementioned information control capabilities both as a chameleon and a non-chameleon. The empirical results show that while non-chameleon LLM agents identify the chameleon, they fail to conceal the secret from the chameleon, and their winning probability is far from the levels of even trivial strategies. To formally explain this behavior, we give a theoretical analysis for a spectrum of strategies, from concealing to revealing, and provide bounds on the non-chameleons' winning probability. Based on the empirical results and theoretical analysis of different strategies, we deduce that LLM-based non-chameleon agents reveal excessive information to agents of unknown identities. Our results point to a weakness of contemporary LLMs, including GPT-4, GPT-4o, Gemini 1.5, and Claude 3.5 Sonnet, in strategic interactions. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的代理在包含非合作方的环境中已变得常见。在这种环境中，代理的决策需要从其对手隐藏信息，向其合作者揭示信息，并推断信息以识别其他代理的特征。为了调查LLM是否具备这些信息控制和决策能力，我们让LLM代理参与基于语言的隐藏身份游戏《变色龙》（The Chameleon）。在这个游戏中，一群未知彼此身份的非变色龙代理试图识别变色龙代理，同时不泄露秘密。该游戏要求变色龙及其非变色龙代理都具备上述信息控制能力。实验结果显示，非变色龙LLM代理能够识别出变色龙代理，但无法从变色龙代理那里隐藏秘密，它们的获胜概率远低于甚至简单策略的水平。为了正式解释这种行为，我们为从隐藏到揭示不同策略谱系进行了理论分析，并提供了非变色龙代理的获胜概率边界。基于实验结果和不同策略的理论分析，我们推断出基于LLM的非变色龙代理向未知身份的代理泄露了过多的信息。我们的研究结果指出了当今包括GPT-4、GPT-4o、Gemini 1.5和Claude 3.5 Sonnet在内的LLM在战略互动中存在的一项弱点。 

---
# SETS: Leveraging Self-Verification and Self-Correction for Improved Test-Time Scaling 

**Title (ZH)**: SETS：利用自我验证和自我修正以提高测试时扩展性 

**Authors**: Jiefeng Chen, Jie Ren, Xinyun Chen, Chengrun Yang, Ruoxi Sun, Sercan Ö Arık  

**Link**: [PDF](https://arxiv.org/pdf/2501.19306)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have created new opportunities to enhance performance on complex reasoning tasks by leveraging test-time computation. However, conventional approaches such as repeated sampling with majority voting or reward model scoring, often face diminishing returns as test-time compute scales, in addition to requiring costly task-specific reward model training. In this paper, we present Self-Enhanced Test-Time Scaling (SETS), a novel method that leverages the self-verification and self-correction capabilities of recent advanced LLMs to overcome these limitations. SETS integrates sampling, self-verification, and self-correction into a unified framework, enabling efficient and scalable test-time computation for improved capabilities at complex tasks. Through extensive experiments on challenging planning and reasoning benchmarks, compared to the alternatives, we demonstrate that SETS achieves significant performance improvements and more favorable test-time scaling laws. 

**Abstract (ZH)**: 最近的大语言模型（LLMs）进展为通过利用测试时计算来增强复杂推理任务的性能提供了新的机会。然而，传统的重复抽样与多数投票或奖励模型评分方法，在测试时计算规模扩大时往往会遭遇递减的回报，并且还需要进行昂贵的任务特定奖励模型训练。本文提出了自我增强测试时缩放（SETS），这是一种利用现代先进LLMs的自我验证和自我修正能力的新方法，以克服这些限制。SETS将采样、自我验证和自我修正整合到一个统一框架中，从而实现高效的可扩展测试时计算，以提高复杂任务的能力。通过在具挑战性的规划和推理基准上的广泛实验，与替代方法相比，我们证明SETS在性能上取得了显著的改进，并且具有更有利于测试时缩放的规律。 

---
# Synthetic User Behavior Sequence Generation with Large Language Models for Smart Homes 

**Title (ZH)**: 使用大型语言模型生成智能家庭中合成用户行为序列 

**Authors**: Zhiyao Xu, Dan Zhao, Qingsong Zou, Jingyu Xiao, Yong Jiang, Zhenhui Yuan, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.19298)  

**Abstract**: In recent years, as smart home systems have become more widespread, security concerns within these environments have become a growing threat. Currently, most smart home security solutions, such as anomaly detection and behavior prediction models, are trained using fixed datasets that are precollected. However, the process of dataset collection is time-consuming and lacks the flexibility needed to adapt to the constantly evolving smart home environment. Additionally, the collection of personal data raises significant privacy concerns for users. Lately, large language models (LLMs) have emerged as a powerful tool for a wide range of tasks across diverse application domains, thanks to their strong capabilities in natural language processing, reasoning, and problem-solving. In this paper, we propose an LLM-based synthetic dataset generation IoTGen framework to enhance the generalization of downstream smart home intelligent models. By generating new synthetic datasets that reflect changes in the environment, smart home intelligent models can be retrained to overcome the limitations of fixed and outdated data, allowing them to better align with the dynamic nature of real-world home environments. Specifically, we first propose a Structure Pattern Perception Compression (SPPC) method tailored for IoT behavior data, which preserves the most informative content in the data while significantly reducing token consumption. Then, we propose a systematic approach to create prompts and implement data generation to automatically generate IoT synthetic data with normative and reasonable properties, assisting task models in adaptive training to improve generalization and real-world performance. 

**Abstract (ZH)**: 近年来，随着智能家居系统的普及，这些环境中存在的安全问题已成为日益突出的威胁。目前，大多数智能家居安全解决方案，如异常检测和行为预测模型，都是通过预先收集的数据集进行训练的。然而，数据集收集的过程耗时且缺乏适应不断演变的智能家居环境所需的灵活性。此外，收集个人数据引发了用户的重要隐私担忧。最近，大型语言模型（LLMs）因其在自然语言处理、推理和问题解决方面强大的能力，在多种应用领域中被证明是强有力的工具。在这篇论文中，我们提出了一种基于大型语言模型的合成数据集生成框架IoTGen，以增强下游智能家居智能模型的一般性。通过生成反映环境变化的新合成数据集，智能家居智能模型可以重新训练以克服固定和过时数据的局限性，使其更好地适应智能家居环境的动态特性。具体来说，我们首先提出了一种针对IoT行为数据的结构模式感知压缩（SPPC）方法，该方法在显著减少标记消耗的同时，保留了数据中最具有信息性的内容。然后，我们提出了一种系统的方法来创造提示并实施数据生成，以自动生成具有规范性和合理性的IoT合成数据，帮助任务模型适应性训练，从而提高泛化能力和实际性能。 

---
# Bridging the Reasoning Gap: Small LLMs Can Plan with Generalised Strategies 

**Title (ZH)**: 弥合推理差距：小型语言模型可以通过通用策略进行规划 

**Authors**: Andrey Borro, Patricia J Riddle, Michael W Barley, Michael J Witbrock  

**Link**: [PDF](https://arxiv.org/pdf/2501.18817)  

**Abstract**: Recent advancements in the reasoning skills of Large Language Models (LLMs) demonstrate an increase in the ability of LLMs to solve simple planning tasks. However, as long as the driving force behind improved reasoning capability is the size and complexity of the model, the financial and computational costs associated with running them will also increase. This trend raises questions about continued accessibility and whether these improvements will increase at the same pace as models continue to grow in size and expense. We propose two approaches to enhance the reasoning ability of less resource-intensive LLMs. (1) Provide them with a generalised strategy for solving tasks within a given domain, generated by a more resource-intensive LLM. (2) Exploit their cost-effectiveness by iteratively prompting these models to correct errors in their proposed solutions. Our empirical results from planning and mathematical reasoning tasks demonstrate that these methods improve the performance of less resource-intensive LLMs to levels comparable with their more resource-intensive counterparts, at a fraction of the cost. Additionally, we show that the utilisation of generalised strategies in our experiments reduced the cost of the less resource-intensive model by nearly 30 percent on average. 

**Abstract (ZH)**: 近期大型语言模型（LLMs）推理能力的进展表明其解决简单规划任务的能力得到了增强。然而，只要提升推理能力的动力依然是模型的规模和复杂度增加，随之而来的便是运行成本的上升。这一趋势引发了关于持续可及性的问题，以及随着模型继续增大和昂贵，这些改进是否会以相同的速度发生。我们提出了两种增强较少资源消耗的LLMs推理能力的方法。（1）通过更有资源消耗的LLMs生成一个通用策略来解决给定领域内的任务。（2）利用它们的成本效益，通过迭代提示这些模型纠正它们提出的解决方案中的错误。我们从规划和数学推理任务中获得的经验结果表明，这些方法能够将较少资源消耗的LLMs的性能提升到与其更资源消耗的同类相媲美的水平，而且成本仅为它们的一小部分。此外，我们还展示了在实验中使用通用策略将较少资源消耗的模型的成本平均降低了近30%。 

---
# LLM-Generated Heuristics for AI Planning: Do We Even Need Domain-Independence Anymore? 

**Title (ZH)**: 生成的LLM启发式方法在AI规划中的应用：我们还需要领域无关性吗？ 

**Authors**: Alexander Tuisov, Yonatan Vernik, Alexander Shleyfman  

**Link**: [PDF](https://arxiv.org/pdf/2501.18784)  

**Abstract**: Domain-independent heuristics have long been a cornerstone of AI planning, offering general solutions applicable across a wide range of tasks without requiring domain-specific engineering. However, the advent of large language models (LLMs) presents an opportunity to generate heuristics tailored to specific planning problems, potentially challenging the necessity of domain independence as a strict design principle. In this paper, we explore the use of LLMs to automatically derive planning heuristics from task descriptions represented as successor generators and goal tests written in general purpose programming language. We investigate the trade-offs between domain-specific LLM-generated heuristics and traditional domain-independent methods in terms of computational efficiency and explainability. Our experiments demonstrate that LLMs can create heuristics that achieve state-of-the-art performance on some standard IPC domains, as well as their ability to solve problems that lack an adequate Planning Domain Definition Language ({\sc pddl}) representation. We discuss whether these results signify a paradigm shift and how they can complement existing approaches. 

**Abstract (ZH)**: 领域无关的经验在人工智能规划中一直是一个基石，提供了一类适用于广泛任务的通用解决方案，无需针对特定领域进行工程设计。然而，大型语言模型（LLMs）的出现为生成针对特定规划问题定制的经验提供了一个机会，可能挑战领域无关性作为严格设计原则的必要性。在本文中，我们探讨了使用LLMs从以通用编程语言编写的任务描述（表示为后续生成器和目标测试）中自动推导规划经验的方法。我们研究了领域特定的LLM生成经验与传统领域无关方法之间的权衡，尤其是在计算效率和可解释性方面的权衡。实验结果表明，LLMs能够创建在一些标准IPC领域中达到最佳性能的经验，并展示了它们解决缺乏适当规划领域定义语言（PDDL）表示的问题的能力。我们讨论了这些结果是否标志着范式转变，并探讨了它们如何补充现有的方法。 

---
# Simulation Streams: A Programming Paradigm for Controlling Large Language Models and Building Complex Systems with Generative AI 

**Title (ZH)**: 仿真流：控制大规模语言模型和构建基于生成式AI复杂系统的编程范式 

**Authors**: Peter Sunehag, Joel Z. Leibo  

**Link**: [PDF](https://arxiv.org/pdf/2501.18668)  

**Abstract**: We introduce Simulation Streams, a programming paradigm designed to efficiently control and leverage Large Language Models (LLMs) for complex, dynamic simulations and agentic workflows. Our primary goal is to create a minimally interfering framework that harnesses the agentic abilities of LLMs while addressing their limitations in maintaining consistency, selectively ignoring/including information, and enforcing strict world rules. Simulation Streams achieves this through a state-based approach where variables are modified in sequential steps by "operators," producing output on a recurring format and adhering to consistent rules for state variables. This approach focus the LLMs on defined tasks, while aiming to have the context stream remain "in-distribution". The approach incorporates an Entity-Component-System (ECS) architecture to write programs in a more intuitive manner, facilitating reuse of workflows across different components and entities. This ECS approach enhances the modularity of the output stream, allowing for complex, multi-entity simulations while maintaining format consistency, information control, and rule enforcement. It is supported by a custom editor that aids in creating, running, and analyzing simulations. We demonstrate the versatility of simulation streams through an illustrative example of an ongoing market economy simulation, a social simulation of three characters playing a game of catch in a park and a suite of classical reinforcement learning benchmark tasks. These examples showcase Simulation Streams' ability to handle complex, evolving scenarios over 100s-1000s of iterations, facilitate comparisons between different agent workflows and models, and maintain consistency and continued interesting developments in LLM-driven simulations. 

**Abstract (ZH)**: 我们引入了Simulation Streams编程范式，该范式旨在高效控制和利用大型语言模型（LLMs）进行复杂的动态模拟和有代理性的工作流程。我们的主要目标是创建一个干扰最小的框架，充分利用LLMs的代理能力，同时解决它们在保持一致性、选择性忽略或包含信息以及强制执行严格的世界规则方面的限制。Simulation Streams 通过基于状态的方法实现这一目标，这种方法通过“操作符”在顺序步骤中修改变量，以一致的格式输出，并遵循状态变量的一致规则。这种方法将重点放在定义的任务上，同时让上下文流保持“在分布”状态。该方法采用实体-组件-系统（ECS）架构，使编程更加直观，有利于在不同组件和实体之间重用工作流程。这种ECS方法增强了输出流的模块性，可以在保持格式一致性、信息控制和规则执行的前提下，进行复杂、多实体的模拟。该方法还借助自定义编辑器来辅助创建、运行和分析模拟。我们通过以下示例展示了Simulation Streams的多功能性：一个持续的市场经济模拟、三个角色在公园里玩接球游戏的社会模拟，以及一系列经典的强化学习基准任务。这些示例展示了Simulation Streams在数百到数千次迭代中处理复杂、演变场景的能力，支持不同代理工作流程和模型之间的比较，并在LLM驱动的模拟中保持一致性及持续的有趣发展。 

---
# Enhancing Large Language Model Efficiencyvia Symbolic Compression: A Formal Approach Towards Interpretability 

**Title (ZH)**: 通过符号压缩提升大型语言模型效率：通向可解释性的形式化方法 

**Authors**: Lumen AI, Tengzhou No. 1 Middle School, Shihao Ji, Zihui Song, Fucheng Zhong, Jisen Jia, Zhaobo Wu, Zheyi Cao, Tianhao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.18657)  

**Abstract**: Large language models (LLMs) face significant token efficiency bottlenecks in code generation and logical reasoning tasks, a challenge that directly impacts inference cost and model interpretability. This paper proposes a formal framework based on symbolic compression,integrating combinatory logic, information-theoretic optimal encoding, and context-aware inference techniques to achieve a step-change improvement in token efficiency while preserving semantic integrity. We establish a mathematical framework within a functional programming paradigm, derive the quantitative relationship between symbolic density and model interpretability, and propose a differentiable compression factor metric to evaluate encoding efficiency. Furthermore, we leverage parameter-efficient fine-tuning (PEFT) techniques to achieve a low-cost application of the GAEL language. Experimental results show that this method achieves a 78.3% token compression rate in code generation tasks while improving logical traceability by 62% through structural explicitness. This research provides new theoretical tools for efficient inference in LLMs and opens a symbolic path for modelinterpretability research. 

**Abstract (ZH)**: 大语言模型（LLMs）在代码生成和逻辑推理等任务中面临显著的标记效率瓶颈，这直接影响推理成本和模型可解释性。本文提出了一种基于符号压缩的形式化框架，该框架结合了组合逻辑、信息论最优编码和上下文感知推理技术，以在保持语义完整性的同时实现标记效率的大幅改进。我们在此功能式编程范式下建立了一个数学框架，推导了符号密度与模型可解释性之间的定量关系，并提出了一种可微分压缩因子度量来评估编码效率。此外，我们利用参数效率微调（PEFT）技术实现GAEL语言的低成本应用。实验结果表明，该方法在代码生成任务中的标记压缩率达到78.3%，并通过结构明确性提高了逻辑追溯性62%。这项研究为LLMs的高效推理提供了新的理论工具，并为模型可解释性研究开辟了一条符号途径。 

---
# Reward-Guided Speculative Decoding for Efficient LLM Reasoning 

**Title (ZH)**: 基于奖励引导的推测性解码以实现高效的大型语言模型推理 

**Authors**: Baohao Liao, Yuhui Xu, Hanze Dong, Junnan Li, Christof Monz, Silvio Savarese, Doyen Sahoo, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.19324)  

**Abstract**: We introduce Reward-Guided Speculative Decoding (RSD), a novel framework aimed at improving the efficiency of inference in large language models (LLMs). RSD synergistically combines a lightweight draft model with a more powerful target model, incorporating a controlled bias to prioritize high-reward outputs, in contrast to existing speculative decoding methods that enforce strict unbiasedness. RSD employs a process reward model to evaluate intermediate decoding steps and dynamically decide whether to invoke the target model, optimizing the trade-off between computational cost and output quality. We theoretically demonstrate that a threshold-based mixture strategy achieves an optimal balance between resource utilization and performance. Extensive evaluations on challenging reasoning benchmarks, including Olympiad-level tasks, show that RSD delivers significant efficiency gains against decoding with the target model only (up to 4.4x fewer FLOPs), while achieving significant better accuracy than parallel decoding method on average (up to +3.5). These results highlight RSD as a robust and cost-effective approach for deploying LLMs in resource-intensive scenarios. 

**Abstract (ZH)**: 我们提出了奖励引导的推测解码 (RSD) 框架，这是一种旨在提高大语言模型 (LLMs) 推断效率的新颖方法。RSD 通过结合一个轻量级的草稿模型和一个更强大的目标模型，并引入可控偏见以优先考虑高奖励输出，从而协同工作，与现有的推测解码方法不同，后者强制实现严格的无偏性。RSD 利用一个过程奖励模型来评估中间解码步骤，并动态决定是否调用目标模型，从而优化计算成本与输出质量之间的权衡。我们从理论上证明，基于阈值的混合策略在资源利用和性能之间实现了最优平衡。在包括奥林匹克级别任务在内的复杂推理基准测试中的广泛评估显示，与仅使用目标模型的解码相比，RSD 能够实现显著的效率改进（最多可减少 4.4 倍的 FLOPs），同时与并行解码方法相比，在平均准确度上也有显著的提升（最多可提高 3.5%）。这些结果突显了 RSD 在资源密集型场景中部署 LLM 的稳健性和成本效益。 

---
# Analysis of LLMs vs Human Experts in Requirements Engineering 

**Title (ZH)**: 分析大型语言模型与人类专家在需求工程中的表现 

**Authors**: Cory Hymel, Hiroe Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2501.19297)  

**Abstract**: The majority of research around Large Language Models (LLM) application to software development has been on the subject of code generation. There is little literature on LLMs' impact on requirements engineering (RE), which deals with the process of developing and verifying the system requirements. Within RE, there is a subdiscipline of requirements elicitation, which is the practice of discovering and documenting requirements for a system from users, customers, and other stakeholders. In this analysis, we compare LLM's ability to elicit requirements of a software system, as compared to that of a human expert in a time-boxed and prompt-boxed study. We found LLM-generated requirements were evaluated as more aligned (+1.12) than human-generated requirements with a trend of being more complete (+10.2%). Conversely, we found users tended to believe that solutions they perceived as more aligned had been generated by human experts. Furthermore, while LLM-generated documents scored higher and performed at 720x the speed, their cost was, on average, only 0.06% that of a human expert. Overall, these findings indicate that LLMs will play an increasingly important role in requirements engineering by improving requirements definitions, enabling more efficient resource allocation, and reducing overall project timelines. 

**Abstract (ZH)**: 关于大型语言模型（LLM）在软件开发中的应用，大多数研究集中在代码生成方面。对于需求工程（RE），即涉及系统需求开发和验证的过程，有关LLM影响的研究较少。需求工程中有一分支是需求获取，即从用户、客户和其他利益相关方中发现并记录系统需求的过程。在本分析中，我们比较了LLM与人类专家在限定时间和提示箱条件下获取软件系统需求的能力。我们发现，LLM生成的需求与人类生成的需求相比，更符合预期（平均评价得分为+1.12），且呈现更为完整（平均增幅+10.2%）的趋势。然而，用户倾向于认为那些看起来更符合预期的需求是由人类专家生成的。此外，尽管LLM生成的文档评分更高，执行速度是人类专家的720倍，但其平均成本仅为人类专家的0.06%。总体而言，这些发现表明，LLM将在需求工程中发挥越来越重要的作用，通过改进需求定义、提高资源分配效率并缩短项目时间线来助力需求工程的发展。 

---
# A Zero-Shot Generalization Framework for LLM-Driven Cross-Domain Sequential Recommendation 

**Title (ZH)**: 基于大语言模型驱动的跨域序列推荐的零样本泛化框架 

**Authors**: Yunzhe Li, Junting Wang, Hari Sundaram, Zhining Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.19232)  

**Abstract**: Zero-shot cross-domain sequential recommendation (ZCDSR) enables predictions in unseen domains without the need for additional training or fine-tuning, making it particularly valuable in data-sparse environments where traditional models struggle. Recent advancements in large language models (LLMs) have greatly improved ZCDSR by leveraging rich pretrained representations to facilitate cross-domain knowledge transfer. However, a key challenge persists: domain semantic bias, which arises from variations in vocabulary and content focus across domains. This misalignment leads to inconsistencies in item embeddings and hinders generalization.
To address this issue, we propose a novel framework designed to enhance LLM-based ZCDSR by improving cross-domain alignment at both the item and sequential levels. At the item level, we introduce a generalization loss that promotes inter-domain compactness by aligning embeddings of similar items across domains while maintaining intra-domain diversity to preserve unique item characteristics. This prevents embeddings from becoming overly generic while ensuring effective transferability. At the sequential level, we develop a method for transferring user behavioral patterns by clustering user sequences in the source domain and applying attention-based aggregation for target domain inference. This dynamic adaptation of user embeddings allows effective zero-shot recommendations without requiring target-domain interactions.
Comprehensive experiments across multiple datasets and domains demonstrate that our framework significantly improves sequential recommendation performance in the ZCDSR setting. By mitigating domain bias and enhancing the transferability of sequential patterns, our method provides a scalable and robust approach for achieving more effective zero-shot recommendations across domains. 

**Abstract (ZH)**: 无监督跨域序列推荐（Zero-shot Cross-Domain Sequential Recommendation, ZCDSR）能够在未见过的领域进行预测，无需额外的训练或微调，尤其适用于传统模型难以应对的数据稀疏环境。最近在大型语言模型（Large Language Models, LLMs）方面的进展大幅提升了ZCDSR，通过利用丰富的预训练表示来促进跨域知识迁移。然而，一个关键挑战依然存在：领域语义偏差，这源于不同领域词汇和内容关注点的差异导致的不一致。这种不一致导致项目嵌入的不一致性，从而阻碍了一般化能力。

为解决这一问题，我们提出了一种新的框架，旨在通过改进LLM基础的ZCDSR来提高跨域对齐。在项目层面，我们引入了一种泛化损失，通过跨领域对类似项目的嵌入进行对齐，以增加它们的紧凑性，同时保持 intra-domain 的多样性，以便保留项目的独特特征。这防止嵌入变得过于通用，同时确保有效迁移能力。在序列层面，我们开发了一种方法，通过在源领域聚类用户序列表征，并应用注意力聚合来推断目标领域的用户行为模式。这种动态的用户嵌入适应能力使得在无需目标领域交互的情况下，能够实现有效的无监督推荐。

我们通过多个数据集和领域的全面实验表明，我们的框架显著提高了ZCDSR设置中的序列推荐性能。通过缓解领域偏差并增强序列模式的迁移性，我们的方法提供了一种可扩展且稳健的方法，能够在不同领域中实现更有效的无监督推荐。 

---
# Efficient Reasoning with Hidden Thinking 

**Title (ZH)**: 高效的隐性思维推理 

**Authors**: Xuan Shen, Yizhou Wang, Xiangxi Shi, Yanzhi Wang, Pu Zhao, Jiuxiang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2501.19201)  

**Abstract**: Chain-of-Thought (CoT) reasoning has become a powerful framework for improving complex problem-solving capabilities in Multimodal Large Language Models (MLLMs). However, the verbose nature of textual reasoning introduces significant inefficiencies. In this work, we propose $\textbf{Heima}$ (as hidden llama), an efficient reasoning framework that leverages reasoning CoTs at hidden latent space. We design the Heima Encoder to condense each intermediate CoT into a compact, higher-level hidden representation using a single thinking token, effectively minimizing verbosity and reducing the overall number of tokens required during the reasoning process. Meanwhile, we design corresponding Heima Decoder with traditional Large Language Models (LLMs) to adaptively interpret the hidden representations into variable-length textual sequence, reconstructing reasoning processes that closely resemble the original CoTs. Experimental results across diverse reasoning MLLM benchmarks demonstrate that Heima model achieves higher generation efficiency while maintaining or even better zero-shot task accuracy. Moreover, the effective reconstruction of multimodal reasoning processes with Heima Decoder validates both the robustness and interpretability of our approach. 

**Abstract (ZH)**: Chain-of-Thought (CoT)推理已成为增强多模态大型语言模型（MLLMs）复杂问题解决能力的一种强大框架。然而，文本推理的冗长性引入了显著的效率问题。在本文中，我们提出了**Heima**（隐灵马）这一高效的推理框架，该框架利用在隐潜空间中进行推理CoT。我们设计了Heima编码器，通过一个思考标记将每个中间的CoT紧凑地表示为更高层次的隐藏表示，有效减少了冗余性，减少了推理过程中所需的标记数量。同时，我们设计了相应的Heima解码器与传统的大型语言模型（LLMs）相结合，以自适应方式将隐藏表示解释为可变长度的文本序列，重构出接近原始CoT的推理过程。在多样性的推理MLLM基准测试中的实验结果表明，Heima模型在保持或甚至在零样本任务准确性方面实现了更高的生成效率。此外，Heima解码器对多模态推理过程的有效重构验证了我们方法的鲁棒性和可解释性。 

---
# Enhancing Model Defense Against Jailbreaks with Proactive Safety Reasoning 

**Title (ZH)**: 增强模型防御以对抗 Jailbreak 攻击：基于前瞻性安全性推理的方法 

**Authors**: Xianglin Yang, Gelei Deng, Jieming Shi, Tianwei Zhang, Jin Song Dong  

**Link**: [PDF](https://arxiv.org/pdf/2501.19180)  

**Abstract**: Large language models (LLMs) are vital for a wide range of applications yet remain susceptible to jailbreak threats, which could lead to the generation of inappropriate responses. Conventional defenses, such as refusal and adversarial training, often fail to cover corner cases or rare domains, leaving LLMs still vulnerable to more sophisticated attacks. We propose a novel defense strategy, Safety Chain-of-Thought (SCoT), which harnesses the enhanced \textit{reasoning capabilities} of LLMs for proactive assessment of harmful inputs, rather than simply blocking them. SCoT augments any refusal training datasets to critically analyze the intent behind each request before generating answers. By employing proactive reasoning, SCoT enhances the generalization of LLMs across varied harmful queries and scenarios not covered in the safety alignment corpus. Additionally, it generates detailed refusals specifying the rules violated. Comparative evaluations show that SCoT significantly surpasses existing defenses, reducing vulnerability to out-of-distribution issues and adversarial manipulations while maintaining strong general capabilities. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在广泛的应用中发挥着关键作用，但仍然容易受到脱管攻击的威胁，这可能导致生成不适当的回答。传统防御措施，如拒绝和对抗训练，往往难以涵盖边缘案例或稀有领域，导致LLMs仍然容易受到更复杂的攻击。我们提出了一种新颖的防御策略，安全链式思考（SCoT），它利用了LLMs增强的推理能力进行主动评估有害输入，而不是简单地拒绝它们。SCoT 增强了任何拒绝训练数据集，在生成答案之前，针对每个请求认真分析其背后的目的。通过采用主动推理，SCoT 提高了LLMs在各种未涵盖在安全性对齐语料中的有害查询和场景中的泛化能力。此外，它还生成详细的拒绝理由，明确说明违反了哪些规则。比较评估表明，SCoT 显著超越了现有防御措施，减少了对分布外问题和对抗操纵的脆弱性，同时保持了强大的通用能力。 

---
# BRiTE: Bootstrapping Reinforced Thinking Process to Enhance Language Model Reasoning 

**Title (ZH)**: BRiTE: 通过强化思考过程来提升语言模型推理能力的自增强方法 

**Authors**: Han Zhong, Yutong Yin, Shenao Zhang, Xiaojun Xu, Yuanxin Liu, Yifei Zuo, Zhihan Liu, Boyi Liu, Sirui Zheng, Hongyi Guo, Liwei Wang, Mingyi Hong, Zhaoran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.18858)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in complex reasoning tasks, yet generating reliable reasoning processes remains a significant challenge. We present a unified probabilistic framework that formalizes LLM reasoning through a novel graphical model incorporating latent thinking processes and evaluation signals. Within this framework, we introduce the Bootstrapping Reinforced Thinking Process (BRiTE) algorithm, which works in two steps. First, it generates high-quality rationales by approximating the optimal thinking process through reinforcement learning, using a novel reward shaping mechanism. Second, it enhances the base LLM by maximizing the joint probability of rationale generation with respect to the model's parameters. Theoretically, we demonstrate BRiTE's convergence at a rate of $1/T$ with $T$ representing the number of iterations. Empirical evaluations on math and coding benchmarks demonstrate that our approach consistently improves performance across different base models without requiring human-annotated thinking processes. In addition, BRiTE demonstrates superior performance compared to existing algorithms that bootstrap thinking processes use alternative methods such as rejection sampling, and can even match or exceed the results achieved through supervised fine-tuning with human-annotated data. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂推理任务中展现了显著的能力，然而生成可靠的推理过程仍然是一个显著的挑战。我们提出了一种统一的概率框架，通过一种新颖的图模型来正式化LLM的推理过程，该图模型结合了潜在的思考过程和评价信号。在此框架内，我们引入了Bootstrapping Reinforced Thinking Process（BRiTE）算法，该算法分为两个步骤。首先，通过强化学习来近似最优的思考过程，生成高质量的推理，利用一种新颖的奖励塑造机制。其次，通过最大化推理生成与模型参数的联合概率来增强基础的LLM。理论上，我们证明了BRiTE在每迭代一次的收敛速率为$1/T$，其中$T$表示迭代次数。对数学和编程基准的实证评估表明，我们的方法在不同的基础模型上都能持续提升性能，而不需要标注过的思考过程。此外，与使用拒绝采样等其他方法进行思考过程的自助提取算法相比，BRiTE展现了更优的性能，并且甚至可以达到或超过使用人类标注数据进行监督微调所获得的结果。 

---
# Large Language Models as Common-Sense Heuristics 

**Title (ZH)**: 大型语言模型作为常识启发式方法 

**Authors**: Andrey Borro, Patricia J Riddle, Michael W Barley, Michael J Witbrock  

**Link**: [PDF](https://arxiv.org/pdf/2501.18816)  

**Abstract**: While systems designed for solving planning tasks vastly outperform Large Language Models (LLMs) in this domain, they usually discard the rich semantic information embedded within task descriptions. In contrast, LLMs possess parametrised knowledge across a wide range of topics, enabling them to leverage the natural language descriptions of planning tasks in their solutions. However, current research in this direction faces challenges in generating correct and executable plans. Furthermore, these approaches depend on the LLM to output solutions in an intermediate language, which must be translated into the representation language of the planning task. We introduce a novel planning method, which leverages the parametrised knowledge of LLMs by using their output as a heuristic for Hill-Climbing Search. This approach is further enhanced by prompting the LLM to generate a solution estimate to guide the search. Our method outperforms the task success rate of similar systems within a common household environment by 22 percentage points, with consistently executable plans. All actions are encoded in their original representation, demonstrating that strong results can be achieved without an intermediate language, thus eliminating the need for a translation step. 

**Abstract (ZH)**: 虽然为了解决规划任务而设计的系统在这一领域远远超过大型语言模型（LLMs），但它们通常会丢弃任务描述中包含的丰富语义信息。相比之下，LLMs具有跨多种主题的参数化知识，使其能够利用规划任务的自然语言描述来解决这些问题。然而，当前在这方面的研究在生成正确且可执行的计划方面面临挑战。此外，这些方法依赖于LLMs以中间语言的形式输出解决方案，并需要将其转换为规划任务的表示语言。我们提出了一种新颖的规划方法，该方法通过利用LLMs的参数化知识，使用其输出作为Hill-Climbing搜索的启发式信息。该方法进一步通过促使LLMs生成解决方案估计值来引导搜索过程。我们的方法在常见家庭环境中的任务成功率方面比类似系统高出22个百分点，且始终能够生成可执行的计划。所有操作均以原始表示形式编码，这表明无需中间语言即可取得强劲结果，从而消除了转换步骤的需要。 

---
# Survey and Improvement Strategies for Gene Prioritization with Large Language Models 

**Title (ZH)**: 大型语言模型在基因优先级确定中的调查与改进策略 

**Authors**: Matthew Neeley, Guantong Qi, Guanchu Wang, Ruixiang Tang, Dongxue Mao, Chaozhong Liu, Sasidhar Pasupuleti, Bo Yuan, Fan Xia, Pengfei Liu, Zhandong Liu, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.18794)  

**Abstract**: Rare diseases are challenging to diagnose due to limited patient data and genetic diversity. Despite advances in variant prioritization, many cases remain undiagnosed. While large language models (LLMs) have performed well in medical exams, their effectiveness in diagnosing rare genetic diseases has not been assessed. To identify causal genes, we benchmarked various LLMs for gene prioritization. Using multi-agent and Human Phenotype Ontology (HPO) classification, we categorized patients based on phenotypes and solvability levels. As gene set size increased, LLM performance deteriorated, so we used a divide-and-conquer strategy to break the task into smaller subsets. At baseline, GPT-4 outperformed other LLMs, achieving near 30% accuracy in ranking causal genes correctly. The multi-agent and HPO approaches helped distinguish confidently solved cases from challenging ones, highlighting the importance of known gene-phenotype associations and phenotype specificity. We found that cases with specific phenotypes or clear associations were more accurately solved. However, we observed biases toward well-studied genes and input order sensitivity, which hindered gene prioritization. Our divide-and-conquer strategy improved accuracy by overcoming these biases. By utilizing HPO classification, novel multi-agent techniques, and our LLM strategy, we improved causal gene identification accuracy compared to our baseline evaluation. This approach streamlines rare disease diagnosis, facilitates reanalysis of unsolved cases, and accelerates gene discovery, supporting the development of targeted diagnostics and therapies. 

**Abstract (ZH)**: 罕见疾病由于患者数据有限和遗传多样性高，诊断起来具有挑战性。尽管在变体优先级排序方面取得了进展，但仍有许多病例未能确诊。虽然大型语言模型（LLMs）在医学考试中表现出色，但它们在诊断罕见遗传疾病方面的有效性尚未得到评估。为了识别致病变异，我们对各种LLM进行了基因优先级排序的基准测试。利用多智能体系统和人类表型ontology（HPO）分类，我们根据表型和解决难度将患者分为不同的类别。随着基因组集的增大，LLM的表现逐渐下降，因此我们采用分而治之的策略将任务分解成更小的子集。基线测试中，GPT-4 在排序致病变异方面表现优于其他LLM，准确率接近30%。多智能体系统和HPO方法有助于区分易解和难解的病例，突显了已知基因-表型关联和表型特异性的重要性。我们发现，具有特定表型或明确关联的病例更易得到准确解决。然而，我们观察到对研究较多的基因存在偏差，并且输入顺序的敏感性影响了基因优先级排序。我们的分而治之策略通过克服这些偏差提高了准确率。通过利用HPO分类、新型多智能体技术和我们的LLM策略，我们提高了致病变异识别的准确性，与基线评估相比有所改进。这种 approach 简化了罕见疾病的诊断，促进了未解病例的重新分析，加速了基因发现，支持了针对性诊断和治疗的发展。 

---
# Fake News Detection After LLM Laundering: Measurement and Explanation 

**Title (ZH)**: 经过LLM清洗后的假新闻检测：测量与解释 

**Authors**: Rupak Kumar Das, Jonathan Dodge  

**Link**: [PDF](https://arxiv.org/pdf/2501.18649)  

**Abstract**: With their advanced capabilities, Large Language Models (LLMs) can generate highly convincing and contextually relevant fake news, which can contribute to disseminating misinformation. Though there is much research on fake news detection for human-written text, the field of detecting LLM-generated fake news is still under-explored. This research measures the efficacy of detectors in identifying LLM-paraphrased fake news, in particular, determining whether adding a paraphrase step in the detection pipeline helps or impedes detection. This study contributes: (1) Detectors struggle to detect LLM-paraphrased fake news more than human-written text, (2) We find which models excel at which tasks (evading detection, paraphrasing to evade detection, and paraphrasing for semantic similarity). (3) Via LIME explanations, we discovered a possible reason for detection failures: sentiment shift. (4) We discover a worrisome trend for paraphrase quality measurement: samples that exhibit sentiment shift despite a high BERTSCORE. (5) We provide a pair of datasets augmenting existing datasets with paraphrase outputs and scores. The dataset is available on GitHub 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的先进能力，它们可以生成高度逼真且与上下文相关的假新闻，这有助于传播错误信息。尽管已有大量有关检测人类撰写的假新闻的研究，但检测LLM生成的假新闻的领域仍相对未被充分探索。本研究评估了检测器在识别LLM重述的假新闻方面的有效性，特别是在确定检测管道中是否加入重述步骤是否有助于或妨碍检测方面发挥了作用。本研究的贡献在于：（1）检测器在识别LLM重述的假新闻方面比识别人类撰写的假新闻更加困难；（2）我们发现哪些模型在哪些任务中表现优异（逃避检测、通过重述逃避检测以及为了语义相似性进行重述）；（3）通过LIME解释，我们发现检测失败可能的一个原因是情感转变；（4）我们发现重述质量测量的一个令人担忧的趋势：尽管BERTSCORE很高，但样本仍表现出情感转变；（5）我们提供了一对数据集，这些数据集通过加入重述输出和评分增强了现有数据集。该数据集可在GitHub上获取。 

---
# Layered Chain-of-Thought Prompting for Multi-Agent LLM Systems: A Comprehensive Approach to Explainable Large Language Models 

**Title (ZH)**: 多层思维链提示在多代理大语言模型系统中的应用：一种全面的可解释大语言模型方法 

**Authors**: Manish Sanwal  

**Link**: [PDF](https://arxiv.org/pdf/2501.18645)  

**Abstract**: Large Language Models (LLMs) leverage chain-of-thought (CoT) prompting to provide step-by-step rationales, improving performance on complex tasks. Despite its benefits, vanilla CoT often fails to fully verify intermediate inferences and can produce misleading explanations. In this work, we propose Layered Chain-of-Thought (Layered-CoT) Prompting, a novel framework that systematically segments the reasoning process into multiple layers, each subjected to external checks and optional user feedback. We expand on the key concepts, present three scenarios -- medical triage, financial risk assessment, and agile engineering -- and demonstrate how Layered-CoT surpasses vanilla CoT in terms of transparency, correctness, and user engagement. By integrating references from recent arXiv papers on interactive explainability, multi-agent frameworks, and agent-based collaboration, we illustrate how Layered-CoT paves the way for more reliable and grounded explanations in high-stakes domains. 

**Abstract (ZH)**: 大型语言模型（LLMs）利用链式思考（CoT）提示提供逐步的推理过程，从而在复杂任务上表现出色。尽管具有诸多优势，传统的CoT往往未能完全验证中间推理，且可能生成误导性的解释。在这项研究中，我们提出了一种新颖的框架——分层链式思考（Layered-CoT）提示，该框架系统地将推理过程划分为多个层次，每个层次都接受外部检查并可选地接受用户反馈。我们详细介绍了核心概念，并提出了三个应用场景——医疗分诊、金融风险评估和敏捷工程——展示了Layered-CoT在透明度、正确性和用户参与度方面如何超越传统的CoT。通过结合近期arXiv论文中关于交互式解释、多智能体框架以及基于代理的合作研究中的参考，我们阐述了Layered-CoT如何为高风险领域中的更可靠和具体的解释开辟道路。 

---
# Indiana Jones: There Are Always Some Useful Ancient Relics 

**Title (ZH)**: 印第安纳·琼斯：总是有一些 Useful 的古遗物 

**Authors**: Junchen Ding, Jiahao Zhang, Yi Liu, Ziqi Ding, Gelei Deng, Yuekang Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.18628)  

**Abstract**: This paper introduces Indiana Jones, an innovative approach to jailbreaking Large Language Models (LLMs) by leveraging inter-model dialogues and keyword-driven prompts. Through orchestrating interactions among three specialised LLMs, the method achieves near-perfect success rates in bypassing content safeguards in both white-box and black-box LLMs. The research exposes systemic vulnerabilities within contemporary models, particularly their susceptibility to producing harmful or unethical outputs when guided by ostensibly innocuous prompts framed in historical or contextual contexts. Experimental evaluations highlight the efficacy and adaptability of Indiana Jones, demonstrating its superiority over existing jailbreak methods. These findings emphasise the urgent need for enhanced ethical safeguards and robust security measures in the development of LLMs. Moreover, this work provides a critical foundation for future studies aimed at fortifying LLMs against adversarial exploitation while preserving their utility and flexibility. 

**Abstract (ZH)**: 本文介绍了一种创新方法 Indiana Jones，该方法通过利用模型间对话和关键词驱动的提示来破解大规模语言模型（LLMs）。通过协调三个专门化 LLMs 之间的交互，该方法在透明和不透明 LLMs 中几乎能够完美地绕过内容保护措施。研究揭示了当代模型中的系统性漏洞，尤其是当模型受到看似无害的提示引导时，容易产生有害或不道德的输出。实验评估突显了 Indiana Jones 的有效性和适应性，证明了其在现有破解方法中的优越性。这些发现强调了在开发 LLMs 过程中加强伦理保护和安全性措施的紧迫需求。此外，本文为基础研究提供了关键框架，旨在加强 LLMs 的防御能力以抵御对抗性利用，同时保留其实用性和灵活性。 

---
# TableMaster: A Recipe to Advance Table Understanding with Language Models 

**Title (ZH)**: TableMaster：一种利用语言模型推动表格理解的方法 

**Authors**: Lang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2501.19378)  

**Abstract**: Tables serve as a fundamental format for representing structured relational data. While current language models (LMs) excel at many text-based tasks, they still face challenges in table understanding due to the complex characteristics of tabular data, such as their structured nature. In this paper, we aim to enhance LMs for improved table understanding. We identify four key challenges: 1) difficulty in locating target data, 2) deficiency in table semantics, 3) numerical inaccuracies in textual reasoning, and 4) semantic inflexibility in symbolic reasoning. To address these issues, we propose TableMaster, a recipe and comprehensive framework that integrates multiple solutions to overcome these obstacles. TableMaster first extracts relevant table content and verbalizes it with enriched semantic context. Additionally, we introduce adaptive reasoning, a flexible approach that dynamically adjusts between textual and symbolic reasoning, tailoring the reasoning process to each query. Extensive analyses and experiments demonstrate our findings and the effectiveness of TableMaster. On the WikiTQ dataset, TableMaster achieves an accuracy of 78.13% using GPT-4o-mini, surpassing existing baselines. 

**Abstract (ZH)**: 表格是表示结构化关系数据的基本格式。尽管当前的语言模型（LMs）在许多文本任务中表现出色，但它们在理解表格方面仍然面临着挑战，这主要是由于表格数据的复杂特性，如其结构化性质。在本文中，我们旨在通过增强语言模型来改善表格理解。我们识别出四个关键挑战：1）目标数据定位困难，2）表格语义不足，3）文本推理中的数值不准确性，4）符号推理中的语义灵活性不足。为了解决这些问题，我们提出了TableMaster，这是一种集成了多种解决方案的食谱和综合框架，以克服这些障碍。TableMaster 首先抽取相关表格内容，并以丰富的语义上下文形式进行表述。此外，我们引入了适应性推理，这是一种灵活的方法，能够根据查询动态调整文本和符号推理之间的平衡，使推理过程适应每个查询。广泛的分析和实验验证了我们的发现和TableMaster的有效性。在WikiTQ数据集中，使用GPT-4o-mini时，TableMaster 的准确率为78.13%，超过了现有基准。 

---
# LLM-based Affective Text Generation Quality Based on Different Quantization Values 

**Title (ZH)**: 基于不同量化值的LLM情感文本生成质量研究 

**Authors**: Yarik Menchaca Resendiz, Roman Klinger  

**Link**: [PDF](https://arxiv.org/pdf/2501.19317)  

**Abstract**: Large language models exhibit a remarkable capacity in language generation and comprehension. These advances enable AI systems to produce more human-like and emotionally engaging text. However, these models rely on a large number of parameters, requiring significant computational resources for training and inference. In some scenarios, accessing these resources can be challenging (e.g., budget or hardware limitations). Techniques like reducing precision bits can make models more memory-efficient, reducing the computational resources needed, at the cost of reduced accuracy. This paper addresses the trade-off between different quantization values, GPU RAM utilization, and text quality in affective text generation (e.g., "I really enjoy running in the snow-covered forest"). To evaluate, we use an emotion classifier and ten seed prompts to generate affective text. We test three setups of precision bits (8, 16, and 32) across five open-weight language models from two different families. Our findings demonstrate that bit reductions lead to memory savings, achieving a reduction of 76%. However, this optimization comes with a trade-off, leading to a decrease of up to 10 pp in F1 score for larger models and an increase of 10 pp for smaller models, along with roughly double the inference time. In terms of text quality, larger models at lower quantization levels generally outperform smaller, higher-precision models -- while requiring similar memory. 

**Abstract (ZH)**: 大型语言模型在语言生成和理解方面表现出非凡的能力。这些进步使得AI系统能够生成更加接近人类和情感化的文本。然而，这些模型依赖大量的参数，需要大量的计算资源进行训练和推断。在某些场景中，获取这些资源可能会有挑战（例如预算或硬件限制）。通过减少精度位数等技术可以提高模型的内存效率，减少所需的计算资源，但会牺牲一定的准确性。本文探讨了不同量化值、GPU RAM利用率与情感文本生成质量之间的权衡问题（例如，“我真的很喜欢在覆雪的森林里跑步”）。为评估这一点，我们使用情感分类器和十个种子提示词生成情感化的文本。我们测试了两种不同家族的五种开源权重语言模型在三种精度位数设置（8位、16位和32位）下的情况。我们的研究结果表明，减少精度位数可以节省内存，实现76%的内存缩减。然而，这种优化伴随着准确性下降的代价：对于大型模型，准确率下降最多可达10个百分点；对于小型模型则提高10个百分点，同时推断时间大约增加一倍。在文本质量方面，较低量化级别的大型模型通常优于较高精度的小型模型，而占用的内存却相似。 

---
# Pheromone-based Learning of Optimal Reasoning Paths 

**Title (ZH)**: 基于pheromone的学习最优推理路径 

**Authors**: Anirudh Chari, Aditya Tiwari, Richard Lian, Suraj Reddy, Brian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.19278)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable reasoning capabilities through chain-of-thought prompting, yet discovering effective reasoning methods for complex problems remains challenging due to the vast space of possible intermediate steps. We introduce Ant Colony Optimization-guided Tree of Thought (ACO-ToT), a novel algorithm that combines ACO with LLMs to discover optimal reasoning paths for complex problems efficiently. Drawing inspiration from Hebbian learning in neurological systems, our method employs a collection of distinctly fine-tuned LLM "ants" to traverse and lay pheromone trails through a centralized tree of thought, with each ant's movement governed by a weighted combination of existing pheromone trails and its own specialized expertise. The algorithm evaluates complete reasoning paths using a mixture-of-experts-based scoring function, with pheromones reinforcing productive reasoning paths across iterations. Experiments on three challenging reasoning tasks (GSM8K, ARC-Challenge, and MATH) demonstrate that ACO-ToT performs significantly better than existing chain-of-thought optimization approaches, suggesting that incorporating biologically inspired collective search mechanisms into LLM inference can substantially enhance reasoning capabilities. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过链式思考提示展示了非凡的推理能力，但发现复杂问题的有效推理方法仍然是挑战性的，因为可能存在大量的中间步骤。我们引入了一种名为蚁群优化引导的思想树（ACO-ToT）的新算法，该算法结合了蚁群优化（ACO）和大规模语言模型（LLMs），以高效地发现复杂问题的最佳推理路径。受到神经学系统中的Hebbian学习启发，我们的方法使用一组不同微调的大规模语言模型“蚂蚁”在中央思想树中遍历并铺设信息素路径，每只蚂蚁的移动由现有信息素路径和其自身专业知识的加权组合来决定。算法使用基于专家群体的评分函数评估完整的推理路径，并通过多次迭代加强有效的推理路径。在三个具有挑战性的推理任务（GSM8K、ARC-Challenge和MATH）上的实验表明，ACO-ToT在链式思考优化方法方面表现出显著优越性，这表明将生物启发的集体搜索机制纳入LLM推理中可以显著增强推理能力。 

---
# Improving the Robustness of Representation Misdirection for Large Language Model Unlearning 

**Title (ZH)**: 提高大型语言模型脱训中表示误导的稳健性 

**Authors**: Dang Huu-Tien, Hoang Thanh-Tung, Le-Minh Nguyen, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2501.19202)  

**Abstract**: Representation Misdirection (RM) and variants are established large language model (LLM) unlearning methods with state-of-the-art performance. In this paper, we show that RM methods inherently reduce models' robustness, causing them to misbehave even when a single non-adversarial forget-token is in the retain-query. Toward understanding underlying causes, we reframe the unlearning process as backdoor attacks and defenses: forget-tokens act as backdoor triggers that, when activated in retain-queries, cause disruptions in RM models' behaviors, similar to successful backdoor attacks. To mitigate this vulnerability, we propose Random Noise Augmentation -- a model and method agnostic approach with theoretical guarantees for improving the robustness of RM methods. Extensive experiments demonstrate that RNA significantly improves the robustness of RM models while enhancing the unlearning performances. 

**Abstract (ZH)**: Representation Misdirection (RM) 和其变体是已知的大规模语言模型（LLM）遗忘方法，具有最先进的性能。本文表明，RM 方法本质上会降低模型的稳健性，即使在保留查询中仅包含一个非对抗性遗忘标记，这些模型也会表现出异常行为。为了理解其根本原因，我们将遗忘过程重新框架为后门攻击和防御：遗忘标记作为后门触发器，在保留查询中激活时，会引发 RM 模型行为的中断，类似于成功的后门攻击。为缓解这一漏洞，我们提出了随机噪声增强——一种模型和方法无关的方法，并具有理论上保证的提高 RM 方法稳健性的能力。大量实验表明，随机噪声增强在提高 RM 模型的稳健性的同时，也提升了遗忘性能。 

---
# Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming 

**Title (ZH)**: 宪法分类器：防御跨越数千小时红队攻击的通用破解攻击 

**Authors**: Mrinank Sharma, Meg Tong, Jesse Mu, Jerry Wei, Jorrit Kruthoff, Scott Goodfriend, Euan Ong, Alwin Peng, Raj Agarwal, Cem Anil, Amanda Askell, Nathan Bailey, Joe Benton, Emma Bluemke, Samuel R. Bowman, Eric Christiansen, Hoagy Cunningham, Andy Dau, Anjali Gopal, Rob Gilson, Logan Graham, Logan Howard, Nimit Kalra, Taesung Lee, Kevin Lin, Peter Lofgren, Francesco Mosconi, Clare O'Hara, Catherine Olsson, Linda Petrini, Samir Rajani, Nikhil Saxena, Alex Silverstein, Tanya Singh, Theodore Sumers, Leonard Tang, Kevin K. Troy, Constantin Weisser, Ruiqi Zhong, Giulio Zhou, Jan Leike, Jared Kaplan, Ethan Perez  

**Link**: [PDF](https://arxiv.org/pdf/2501.18837)  

**Abstract**: Large language models (LLMs) are vulnerable to universal jailbreaks-prompting strategies that systematically bypass model safeguards and enable users to carry out harmful processes that require many model interactions, like manufacturing illegal substances at scale. To defend against these attacks, we introduce Constitutional Classifiers: safeguards trained on synthetic data, generated by prompting LLMs with natural language rules (i.e., a constitution) specifying permitted and restricted content. In over 3,000 estimated hours of red teaming, no red teamer found a universal jailbreak that could extract information from an early classifier-guarded LLM at a similar level of detail to an unguarded model across most target queries. On automated evaluations, enhanced classifiers demonstrated robust defense against held-out domain-specific jailbreaks. These classifiers also maintain deployment viability, with an absolute 0.38% increase in production-traffic refusals and a 23.7% inference overhead. Our work demonstrates that defending against universal jailbreaks while maintaining practical deployment viability is tractable. 

**Abstract (ZH)**: 大型语言模型（LLMs）容易受到通用突破策略的攻击，这些策略系统地绕过了模型的安全防护，使用户能够执行需要大量模型交互的有害过程，例如大规模制造非法物质。为了防御这些攻击，我们引入了宪法分类器：这些防护措施是基于合成数据训练的，而合成数据是由提示LLM使用自然语言规则（即宪法）生成的，这些规则明确了允许和限制的内容。在超过3000个估计的红队测试小时里，没有一位红队成员能够找到能够从早期分类器守护的LLM中提取与未受保护模型在大多数目标查询中相似详细程度信息的通用突破策略。在自动化评估中，增强后的分类器展示了对域特定的隔离突破具有稳健的防御能力。这些分类器还保持了部署可行性，增加了绝对0.38%的生产流量拒绝率和23.7%的推理开销。我们的工作证明了在保持实用性的同时防御通用突破策略是可行的。 

---
# Structural Embedding Projection for Contextual Large Language Model Inference 

**Title (ZH)**: 基于结构嵌入投影的上下文大型语言模型推理 

**Authors**: Vincent Enoasmo, Cedric Featherstonehaugh, Xavier Konstantinopoulos, Zacharias Huntington  

**Link**: [PDF](https://arxiv.org/pdf/2501.18826)  

**Abstract**: Structured embedding transformations offer a promising approach for enhancing the efficiency and coherence of language model inference. The introduction of Structural Embedding Projection (SEP) provides a mechanism for refining token representations through projection matrices that integrate hierarchical and relational dependencies. The mathematical formulation of SEP enables embedding spaces to capture structured contextual relationships, thereby improving semantic fidelity without significantly increasing computational overhead. Experimental evaluations conducted on a range of linguistic datasets revealed that SEP contributed to reductions in perplexity and enhanced contextual coherence, demonstrating its potential to refine language model outputs. Computational efficiency assessments highlighted variations across different datasets, suggesting that the integration of structured embeddings introduced dataset-dependent trade-offs between inference speed and representational richness. The qualitative analysis of generated responses indicated that SEP enhanced narrative consistency and topic alignment, leading to improved fluency in multi-sentence text generation. The modifications to embedding layers required precise optimization to ensure stable training dynamics, as the introduction of structured transformations altered the traditional representation-learning process. The architectural adjustments necessary for SEP implementation influenced inference latency and memory consumption, requiring a balance between efficiency gains and additional processing demands. The impact of SEP on lexical diversity suggested that embedding modifications influenced the model's vocabulary usage, reflecting a more context-aware selection of generated tokens. 

**Abstract (ZH)**: 结构嵌入变换为提高语言模型推理的效率和连贯性提供了有希望的方法。Structural Embedding Projection (SEP) 的引入提供了一种机制，通过结合层次和关系依赖性的投影矩阵细化词元表示。SEP的数学表述使嵌入空间能够捕捉结构化的上下文关系，从而在不显著增加计算开销的情况下提高语义保真度。在多种语言数据集上的实验评估表明，SEP有助于减少困惑度并增强上下文连贯性，证明了其在提升语言模型输出方面的潜力。计算效率评估揭示了不同数据集间的差异，表明结构嵌入的集成在推理速度和表示丰富性之间引入了数据集依赖的权衡。生成响应的定性分析表明，SEP增强了叙事一致性并改善了主题对齐，从而提高了多句文本生成的流畅度。嵌入层的修改要求精确优化以确保稳定的训练动态，因为结构变换的引入改变了传统的表征学习过程。SEP实施所需的架构调整影响了推理延迟和内存消耗，需要在效率提升和额外处理需求之间寻求平衡。SEP对词汇多样性的影晌表明，嵌入层的修改影响了模型词汇的使用情况，反映了对生成词元更具上下文意识的选择。 

---
# Examining the Robustness of Large Language Models across Language Complexity 

**Title (ZH)**: 考察大型语言模型在不同语言复杂性下的稳健性 

**Authors**: Jiayi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.18738)  

**Abstract**: With the advancement of large language models (LLMs), an increasing number of student models have leveraged LLMs to analyze textual artifacts generated by students to understand and evaluate their learning. These student models typically employ pre-trained LLMs to vectorize text inputs into embeddings and then use the embeddings to train models to detect the presence or absence of a construct of interest. However, how reliable and robust are these models at processing language with different levels of complexity? In the context of learning where students may have different language backgrounds with various levels of writing skills, it is critical to examine the robustness of such models to ensure that these models work equally well for text with varying levels of language complexity. Coincidentally, a few (but limited) research studies show that the use of language can indeed impact the performance of LLMs. As such, in the current study, we examined the robustness of several LLM-based student models that detect student self-regulated learning (SRL) in math problem-solving. Specifically, we compared how the performance of these models vary using texts with high and low lexical, syntactic, and semantic complexity measured by three linguistic measures. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）的进步，越来越多的学生模型利用LLMs来分析学生生成的文本，以理解和评估其学习情况。这些学生模型通常采用预训练的LLMs将文本输入矢量化为嵌入表示，然后使用这些嵌入来训练模型以检测感兴趣的结构是否存在。然而，这些模型在处理不同复杂度的语言时有多可靠和稳健？在学生可能具有不同语言背景和不同写作技能的学习情境下，确保这些模型能够有效地处理不同语言复杂度的文本尤为重要。巧合的是，少数（但有限）的研究表明，语言的使用确实会影响LLMs的性能。因此，在本研究中，我们考察了几种基于LLM的学生模型在检测数学问题解决中的自我调节学习（SRL）时的稳健性。具体而言，我们通过使用基于三项语言学度量测量的高复杂度和低复杂度的文本，比较了这些模型的性能差异。 

---
# Zero-shot Large Language Models for Long Clinical Text Summarization with Temporal Reasoning 

**Title (ZH)**: 零样本大型语言模型在具有时间推理的长临床文本摘要中的应用 

**Authors**: Maya Kruse, Shiyue Hu, Nicholas Derby, Yifu Wu, Samantha Stonbraker, Bingsheng Yao, Dakuo Wang, Elizabeth Goldberg, Yanjun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2501.18724)  

**Abstract**: Recent advancements in large language models (LLMs) have shown potential for transforming data processing in healthcare, particularly in understanding complex clinical narratives. This study evaluates the efficacy of zero-shot LLMs in summarizing long clinical texts that require temporal reasoning, a critical aspect for comprehensively capturing patient histories and treatment trajectories. We applied a series of advanced zero-shot LLMs to extensive clinical documents, assessing their ability to integrate and accurately reflect temporal dynamics without prior task-specific training. While the models efficiently identified key temporal events, they struggled with chronological coherence over prolonged narratives. The evaluation, combining quantitative and qualitative methods, highlights the strengths and limitations of zero-shot LLMs in clinical text summarization. The results suggest that while promising, zero-shot LLMs require further refinement to effectively support clinical decision-making processes, underscoring the need for enhanced model training approaches that better capture the nuances of temporal information in long context medical documents. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的发展显示出其在医疗健康数据处理中的潜在潜力，特别是在理解复杂的临床叙述方面。本研究评估了零样本LLMs在总结需要时间推理的长篇临床文本方面的有效性，这是全面捕捉患者病史和治疗轨迹的关键方面。我们应用了一系列先进的零样本LLMs对大量临床文件进行了评估，测试了它们在不进行特定任务训练的情况下整合并准确反映时间动态的能力。虽然这些模型能够高效地识别关键的时间事件，但在长时间叙述中保持时间顺序连贯性方面存在困难。本评估综合了定量和定性方法，突显了零样本LLMs在临床文本总结方面的优势和局限性。研究结果表明，虽然零样本LLMs前景广阔，但它们需要进一步优化，以更有效地支持临床决策过程，强调了需要采用更好的模型训练方法来更好地捕捉长上下文医学文档中的时间信息的重要性。 

---
# SELMA: A Speech-Enabled Language Model for Virtual Assistant Interactions 

**Title (ZH)**: SELMA：一种语音启用的语言模型，适用于虚拟助手交互 

**Authors**: Dominik Wagner, Alexander Churchill, Siddarth Sigtia, Erik Marchi  

**Link**: [PDF](https://arxiv.org/pdf/2501.19377)  

**Abstract**: In this work, we present and evaluate SELMA, a Speech-Enabled Language Model for virtual Assistant interactions that integrates audio and text as inputs to a Large Language Model (LLM). SELMA is designed to handle three primary and two auxiliary tasks related to interactions with virtual assistants simultaneously within a single end-to-end model. We employ low-rank adaptation modules for parameter-efficient training of both the audio encoder and the LLM. Additionally, we implement a feature pooling strategy enabling the system to recognize global patterns and improve accuracy on tasks less reliant on individual sequence elements. Experimental results on Voice Trigger (VT) detection, Device-Directed Speech Detection (DDSD), and Automatic Speech Recognition (ASR), demonstrate that our approach both simplifies the typical input processing pipeline of virtual assistants significantly and also improves performance compared to dedicated models for each individual task. SELMA yields relative Equal-Error Rate improvements of 64% on the VT detection task, and 22% on DDSD, while also achieving word error rates close to the baseline. 

**Abstract (ZH)**: 在本文中，我们介绍了并评估了SELMA（Speech-Enabled Language Model for Virtual Assistant Interactions），这是一种结合了音频和文本输入的大语言模型（LLM），用于虚拟助手交互。SELMA 设计用于在同一端到端模型中同时处理与虚拟助手交互相关的三大主要任务和两大辅助任务。我们采用低秩适应模块对音频编码器和大语言模型进行参数高效的训练。此外，我们实现了一种特征聚合策略，使系统能够识别全局模式并提高对较少依赖于个体序列元素的任务的准确性。在Voice Trigger（VT）检测、Device-Directed Speech Detection（DDSD）和自动语音识别（ASR）等实验中的结果表明，我们的方法不仅显著简化了虚拟助手的典型输入处理流程，还显著提高了性能，相较于针对每个单独任务的专用模型。在VT检测任务中，SELMA 的相对等错误率（Equal-Error Rate, EER）改进了64%，在DDSD任务中改进了22%，同时在单词错误率（Word Error Rate, WER）方面接近基线水平。 

---
# Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment 

**Title (ZH)**: 法官解码：更快的推测采样需要超越模型对齐 

**Authors**: Gregor Bachmann, Sotiris Anagnostidis, Albert Pumarola, Markos Georgopoulos, Artsiom Sanakoyeu, Yuming Du, Edgar Schönfeld, Ali Thabet, Jonas Kohler  

**Link**: [PDF](https://arxiv.org/pdf/2501.19309)  

**Abstract**: The performance of large language models (LLMs) is closely linked to their underlying size, leading to ever-growing networks and hence slower inference. Speculative decoding has been proposed as a technique to accelerate autoregressive generation, leveraging a fast draft model to propose candidate tokens, which are then verified in parallel based on their likelihood under the target model. While this approach guarantees to reproduce the target output, it incurs a substantial penalty: many high-quality draft tokens are rejected, even when they represent objectively valid continuations. Indeed, we show that even powerful draft models such as GPT-4o, as well as human text cannot achieve high acceptance rates under the standard verification scheme. This severely limits the speedup potential of current speculative decoding methods, as an early rejection becomes overwhelmingly likely when solely relying on alignment of draft and target.
We thus ask the following question: Can we adapt verification to recognize correct, but non-aligned replies? To this end, we draw inspiration from the LLM-as-a-judge framework, which demonstrated that LLMs are able to rate answers in a versatile way. We carefully design a dataset to elicit the same capability in the target model by training a compact module on top of the embeddings to produce ``judgements" of the current continuation. We showcase our strategy on the Llama-3.1 family, where our 8b/405B-Judge achieves a speedup of 9x over Llama-405B, while maintaining its quality on a large range of benchmarks. These benefits remain present even in optimized inference frameworks, where our method reaches up to 141 tokens/s for 8B/70B-Judge and 129 tokens/s for 8B/405B on 2 and 8 H100s respectively. 

**Abstract (ZH)**: 大型语言模型（LLMs）的性能与其基础规模密切相关，导致了网络规模的不断增长，从而引发了更快的推理速度。推测性解码已被提出作为一种技术，用于加速自回归生成，通过使用快速草稿模型提出候选token，然后在并行验证下基于其在目标模型下的可能性进行验证。尽管这种方法保证能够复制目标输出，但也导致了显著的代价：许多高质量的草稿token被拒绝，即使它们代表了客观上有效的扩展。事实上，我们展示了即使像GPT-4o这样的强大草稿模型，甚至人类文本，在标准验证方案下也无法实现高接受率。这严重限制了当前推测性解码方法的加速潜力，仅依赖草稿和目标的对齐会导致过早拒绝变得极为常见。

因此，我们提出以下问题：我们能否调整验证过程，以识别正确但未对齐的回答？为此，我们借鉴LLM作为裁判的框架，该框架证明了LLM能够以灵活的方式评估答案。我们精心设计了一个数据集，通过在嵌入层上训练一个紧凑模块来产生当前扩展的“判决”，以使目标模型具备同样的能力。我们在Llama-3.1家族中展示了我们的策略，我们的8b/405B-裁判模型相对于Llama-405B实现了9倍的加速，同时在大量benchmark上保持了质量。即使在优化推理框架中，我们的方法也能达到141 token/s（使用2个H100）和129 token/s（使用8个H100），对于8b/70B-裁判和8b/405B-裁判模型。 

---
# Partially Rewriting a Transformer in Natural Language 

**Title (ZH)**: 部分重构自然语言变压器模型 

**Authors**: Gonçalo Paulo, Nora Belrose  

**Link**: [PDF](https://arxiv.org/pdf/2501.18838)  

**Abstract**: The greatest ambition of mechanistic interpretability is to completely rewrite deep neural networks in a format that is more amenable to human understanding, while preserving their behavior and performance. In this paper, we attempt to partially rewrite a large language model using simple natural language explanations. We first approximate one of the feedforward networks in the LLM with a wider MLP with sparsely activating neurons - a transcoder - and use an automated interpretability pipeline to generate explanations for these neurons. We then replace the first layer of this sparse MLP with an LLM-based simulator, which predicts the activation of each neuron given its explanation and the surrounding context. Finally, we measure the degree to which these modifications distort the model's final output. With our pipeline, the model's increase in loss is statistically similar to entirely replacing the sparse MLP output with the zero vector. We employ the same protocol, this time using a sparse autoencoder, on the residual stream of the same layer and obtain similar results. These results suggest that more detailed explanations are needed to improve performance substantially above the zero ablation baseline. 

**Abstract (ZH)**: 机械可解释性的最大抱负是将深度神经网络完全重新书写成一种更易于人类理解的格式，同时保留其行为和性能。在本文中，我们尝试使用简单的自然语言解释部分重写一个大规模语言模型。首先，我们用一个稀疏激活神经元的较宽多层感知机（MLP）近似LLM中的一个前向网络 —— 这个MLP被称为转换器，并使用自动可解释性流水线为这些神经元生成解释。然后，我们将这个稀疏MLP的第一层替换为基于LLM的模拟器，该模拟器根据给定的解释和上下文预测每个神经元的激活情况。最后，我们测量这些修改对模型最终输出的扭曲程度。通过我们的流水线，模型的损失量增加在统计上与完全用零向量替换稀疏MLP输出相当。我们使用相同的协议，这次使用稀疏自编码器，应用于相同层的残差流，并获得类似的结果。这些结果表明，为了在零删除基线之上显著提高性能，需要更详细的解释。 

---
# The TIP of the Iceberg: Revealing a Hidden Class of Task-In-Prompt Adversarial Attacks on LLMs 

**Title (ZH)**: 冰山一角：揭示LLMs中隐藏的任务嵌入提示对抗攻击类别 

**Authors**: Sergey Berezin, Reza Farahbakhsh, Noel Crespi  

**Link**: [PDF](https://arxiv.org/pdf/2501.18626)  

**Abstract**: We present a novel class of jailbreak adversarial attacks on LLMs, termed Task-in-Prompt (TIP) attacks. Our approach embeds sequence-to-sequence tasks (e.g., cipher decoding, riddles, code execution) into the model's prompt to indirectly generate prohibited inputs. To systematically assess the effectiveness of these attacks, we introduce the PHRYGE benchmark. We demonstrate that our techniques successfully circumvent safeguards in six state-of-the-art language models, including GPT-4o and LLaMA 3.2. Our findings highlight critical weaknesses in current LLM safety alignments and underscore the urgent need for more sophisticated defence strategies.
Warning: this paper contains examples of unethical inquiries used solely for research purposes. 

**Abstract (ZH)**: 我们提出了一种针对大规模语言模型（LLMs）的新颖类别 jailbreak 对抗攻击，称为任务在提示（Task-in-Prompt，TIP）攻击。我们的方法将序列到序列的任务（例如，密码解码、谜语、代码执行）嵌入到模型的提示中，以间接生成禁止输入。为了系统评估这些攻击的有效性，我们引入了 PHRYGE 基准。我们证明，我们的技术成功绕过了包括 GPT-4o 和 LLaMA 3.2 在内的六种最先进的语言模型的安全防护。我们的研究结果突显了当前 LLM 安全对齐中的关键弱点，并强调了需要更高级防御策略的紧迫性。
请注意：本文包含仅供研究使用的目的不道德的查询示例。 

---
