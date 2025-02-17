# Semantica: Decentralized Search using a LLM-Guided Semantic Tree Overlay 

**Title (ZH)**: Semantica：由LLM引导的语义树overlay的去中心化搜索 

**Authors**: Petru Neague, Quinten Stokkink, Naman Goel, Johan Pouwelse  

**Link**: [PDF](https://arxiv.org/pdf/2502.10151)  

**Abstract**: Centralized search engines are key for the Internet, but lead to undesirable concentration of power. Decentralized alternatives fail to offer equal document retrieval accuracy and speed. Nevertheless, Semantic Overlay Networks can come close to the performance of centralized solutions when the semantics of documents are properly captured. This work uses embeddings from Large Language Models to capture semantics and fulfill the promise of Semantic Overlay Networks. Our proposed algorithm, called Semantica, constructs a prefix tree (trie) utilizing document embeddings calculated by a language model. Users connect to each other based on the embeddings of their documents, ensuring that semantically similar users are directly linked. Thereby, this construction makes it more likely for user searches to be answered by the users that they are directly connected to, or by the users they are close to in the network connection graph. The implementation of our algorithm also accommodates the semantic diversity of individual users by spawning "clone" user identifiers in the tree. Our experiments use emulation with a real-world workload to show Semantica's ability to identify and connect to similar users quickly. Semantica finds up to ten times more semantically similar users than current state-of-the-art approaches. At the same time, Semantica can retrieve more than two times the number of relevant documents given the same network load. We also make our code publicly available to facilitate further research in the area. 

**Abstract (ZH)**: 集中式搜索引擎是互联网的关键组成部分，但会导致权力过度集中。去中心化的替代方案在文档检索的准确性和速度方面无法与集中式解决方案相媲美。然而，语义Overlay网络在文档语义得到适当捕捉时，可以接近集中式解决方案的性能。本研究利用大规模语言模型的嵌入来捕捉语义，并实现语义Overlay网络的承诺。我们提出的算法称为Semantica，利用语言模型计算出来的文档嵌入构建前缀树（Trie）。用户基于其文档的嵌入相互连接，确保语义相似的用户能够直接相连。这种构造使得用户的搜索查询更有可能由他们直接连接的用户或在网络连接图中接近的用户来回答。此外，我们的算法通过生成“克隆”用户标识符来适应用户的语义多样性，从而使这些标识符在树中出现。我们的实验通过使用真实负载的模拟来展示Semantica快速识别和连接相似用户的的能力。Semantica在识别语义相似用户方面比当前最先进的方法多出10倍。同时，在相同的网络负载下，Semantica可以检索到超过两倍的相关文档数量。我们还将我们的代码公开，以促进该领域的进一步研究。 

---
# A Survey on LLM-powered Agents for Recommender Systems 

**Title (ZH)**: 基于大规模语言模型的推荐系统代理综述 

**Authors**: Qiyao Peng, Hongtao Liu, Hua Huang, Qing Yang, Minglai Shao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10050)  

**Abstract**: Recommender systems are essential components of many online platforms, yet traditional approaches still struggle with understanding complex user preferences and providing explainable recommendations. The emergence of Large Language Model (LLM)-powered agents offers a promising approach by enabling natural language interactions and interpretable reasoning, potentially transforming research in recommender systems. This survey provides a systematic review of the emerging applications of LLM-powered agents in recommender systems. We identify and analyze three key paradigms in current research: (1) Recommender-oriented approaches, which leverage intelligent agents to enhance the fundamental recommendation mechanisms; (2) Interaction-oriented approaches, which facilitate dynamic user engagement through natural dialogue and interpretable suggestions; and (3) Simulation-oriented approaches, which employ multi-agent frameworks to model complex user-item interactions and system dynamics. Beyond paradigm categorization, we analyze the architectural foundations of LLM-powered recommendation agents, examining their essential components: profile construction, memory management, strategic planning, and action execution. Our investigation extends to a comprehensive analysis of benchmark datasets and evaluation frameworks in this domain. This systematic examination not only illuminates the current state of LLM-powered agent recommender systems but also charts critical challenges and promising research directions in this transformative field. 

**Abstract (ZH)**: 推荐系统是许多在线平台的关键组成部分，尽管传统的做法仍然难以理解复杂的用户偏好并提供可解释的推荐。通过使用大型语言模型（LLM）驱动的代理，可以实现自然语言交互和可解释的推理，从而为推荐系统研究带来潜在的变革。本文综述了LLM驱动代理在推荐系统中的新兴应用。我们总结了当前研究中的三大主要范式：（1）以推荐为导向的方法，利用智能代理来增强基本的推荐机制；（2）以交互为导向的方法，通过自然对话和可解释的建议促进动态用户参与；以及（3）以仿真为导向的方法，利用多智能体框架来建模复杂的用户-项目交互和系统动力学。除了范式分类，我们还分析了LLM驱动推荐代理的架构基础，审视了其主要组成部分：用户档案构建、内存管理、战略规划和行动执行。我们进一步对这一领域的基准数据集和评估框架进行了全面分析。通过这种系统的检查，不仅揭示了LLM驱动代理推荐系统当前的状态，还指出了这一变革领域中关键挑战和有前途的研究方向。 

---
# ArchRAG: Attributed Community-based Hierarchical Retrieval-Augmented Generation 

**Title (ZH)**: ArchRAG：具属性的社区导向层次检索增强生成 

**Authors**: Shu Wang, Yixiang Fang, Yingli Zhou, Xilin Liu, Yuchi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.09891)  

**Abstract**: Retrieval-Augmented Generation (RAG) has proven effective in integrating external knowledge into large language models (LLMs) for question-answer (QA) tasks. The state-of-the-art RAG approaches often use the graph data as the external data since they capture the rich semantic information and link relationships between entities. However, existing graph-based RAG approaches cannot accurately identify the relevant information from the graph and also consume large numbers of tokens in the online retrieval process. To address these issues, we introduce a novel graph-based RAG approach, called Attributed Community-based Hierarchical RAG (ArchRAG), by augmenting the question using attributed communities, and also introducing a novel LLM-based hierarchical clustering method. To retrieve the most relevant information from the graph for the question, we build a novel hierarchical index structure for the attributed communities and develop an effective online retrieval method. Experimental results demonstrate that ArchRAG outperforms existing methods in terms of both accuracy and token cost. 

**Abstract (ZH)**: 检索增强生成（RAG）方法已被证明在将外部知识集成到大型语言模型（LLMs）中以进行问答（QA）任务时是有效的。最先进的一些RAG方法通常使用图数据作为外部数据，因为图数据能够捕捉丰富的语义信息并反映实体之间的关系。然而，现有的基于图的RAG方法在从图中准确识别相关信息方面存在困难，并且在在线检索过程中消耗了大量的 tokens。为了解决这些问题，我们提出了一种新颖的基于图的RAG方法，称为属性社区层次RAG（ArchRAG），该方法通过使用属性社区增强问题，并引入了一种新颖的基于大型语言模型的层次聚类方法。为了从图中为问题检索最相关的信息，我们构建了一种新颖的层次索引结构来索引属性社区，并开发了一种有效的在线检索方法。实验结果表明，ArchRAG 在准确性和 tokens 成本方面都优于现有方法。 

---
# LLM-Powered Preference Elicitation in Combinatorial Assignment 

**Title (ZH)**: 基于LLM的组合分配中偏好引致方法 

**Authors**: Ermis Soumalias, Yanchen Jiang, Kehang Zhu, Michael Curry, Sven Seuken, David C. Parkes  

**Link**: [PDF](https://arxiv.org/pdf/2502.10308)  

**Abstract**: We study the potential of large language models (LLMs) as proxies for humans to simplify preference elicitation (PE) in combinatorial assignment. While traditional PE methods rely on iterative queries to capture preferences, LLMs offer a one-shot alternative with reduced human effort. We propose a framework for LLM proxies that can work in tandem with SOTA ML-powered preference elicitation schemes. Our framework handles the novel challenges introduced by LLMs, such as response variability and increased computational costs. We experimentally evaluate the efficiency of LLM proxies against human queries in the well-studied course allocation domain, and we investigate the model capabilities required for success. We find that our approach improves allocative efficiency by up to 20%, and these results are robust across different LLMs and to differences in quality and accuracy of reporting. 

**Abstract (ZH)**: 我们研究了大型语言模型（LLMs）作为人类代理在组合分配中简化偏好获取（PE）的潜力。传统的方法依赖于迭代查询来捕捉偏好，而LLMs提供了减少人类劳动的努力量化的单次替代方案。我们提出了一种框架，使得LLMs可以与最先进的基于机器学习的偏好获取方法协同工作。我们的框架处理由LLMs引入的新挑战，例如响应的变异性及增加的计算成本。我们在已研究充分的课程分配领域中实验性地评估了LLMs代理与人类查询的效率，并探讨了成功所需的模型能力。我们发现，我们的方法在分配效率方面提高了最多20%，并且这些结果在不同的LLMs及报告质量和准确性差异的情况下具有稳健性。 

---
# Do Large Language Models Reason Causally Like Us? Even Better? 

**Title (ZH)**: 大型语言模型像我们一样进行因果推理吗？甚至更好？ 

**Authors**: Hanna M. Dettki, Brenden M. Lake, Charley M. Wu, Bob Rehder  

**Link**: [PDF](https://arxiv.org/pdf/2502.10215)  

**Abstract**: Causal reasoning is a core component of intelligence. Large language models (LLMs) have shown impressive capabilities in generating human-like text, raising questions about whether their responses reflect true understanding or statistical patterns. We compared causal reasoning in humans and four LLMs using tasks based on collider graphs, rating the likelihood of a query variable occurring given evidence from other variables. We find that LLMs reason causally along a spectrum from human-like to normative inference, with alignment shifting based on model, context, and task. Overall, GPT-4o and Claude showed the most normative behavior, including "explaining away", whereas Gemini-Pro and GPT-3.5 did not. Although all agents deviated from the expected independence of causes - Claude the least - they exhibited strong associative reasoning and predictive inference when assessing the likelihood of the effect given its causes. These findings underscore the need to assess AI biases as they increasingly assist human decision-making. 

**Abstract (ZH)**: 因果推理是智能的核心组成部分。大型语言模型（LLMs）展示了生成类人类文本的能力，引发了对其响应是反映真正理解还是统计模式的疑问。我们使用基于碰撞图的任务比较了人类和四种LLM的因果推理能力，这些任务包括评估某一变量在其他变量证据的情况下出现的可能性。我们的研究表明，LLM在从人类样本来规范推理的谱系中进行因果推理，其对齐程度取决于模型、上下文和任务。总体而言，GPT-4o和Claude表现出最规范的行为，包括“解释掉”现象；而Gemini-Pro和GPT-3.5则没有表现出规范性行为。尽管所有代理都偏离了原因相互独立的预期，Claude偏离程度最小，但它们在评估效果给定其原因的出现可能性时，仍然展示了强大的关联推理和预测性推断。这些发现强调了在AI越来越多地协助人类决策时评估AI偏见的需求。 

---
# MathConstruct: Challenging LLM Reasoning with Constructive Proofs 

**Title (ZH)**: MathConstruct：以建构性证明挑战LLM推理能力 

**Authors**: Mislav Balunović, Jasper Dekoninck, Nikola Jovanović, Ivo Petrov, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2502.10197)  

**Abstract**: While Large Language Models (LLMs) demonstrate impressive performance in mathematics, existing math benchmarks come with significant limitations. Many focus on problems with fixed ground-truth answers, and are often saturated due to problem simplicity or the viability of guessing or memorization. Crucially, they capture only a narrow subset of relevant math problems. To address this research gap, we introduce \mc, a new benchmark of 126 challenging problems sourced from various math competitions, which targets constructive proofs, a widely encountered problem type requiring the construction of mathematical objects with specific properties. These proofs are particularly suitable for LLM evaluation, as solution correctness can be easily verified. Our automated verifiers also enable MathConstruct to generate problem variations, used to evaluate robustness. State-of-the-art LLMs solve only 54% of MathConstruct problems, highlighting its complexity and importance for LLM evaluation. 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）在数学领域展现了 impressive 的性能，现有的数学基准测试仍然存在显著的局限性。许多基准测试关注有固定正确答案的问题，并且由于问题过于简单或可以通过猜测或记忆来解决而变得饱和。更重要的是，这些基准测试仅捕捉到了相关数学问题的一个狭窄子集。为解决这一研究空白，我们引入了 \mc（MathConstruct），一个包含来自各类数学竞赛的126个具有挑战性问题的新基准测试，该基准测试旨在评估构造性证明，这是一类常见的问题类型，需要构建具有特定属性的数学对象。这些证明特别适合评估LLM的能力，因为解的正确性可以很容易地验证。我们还通过自动化验证器生成问题变体，用于评估模型的鲁棒性。最先进的LLM仅能解决MathConstruct测试中54%的问题，这突显了其复杂性和在LLM评估中的重要性。 

---
# POI-Enhancer: An LLM-based Semantic Enhancement Framework for POI Representation Learning 

**Title (ZH)**: POI-增强器：基于LLM的POI语义增强表示学习框架 

**Authors**: Jiawei Cheng, Jingyuan Wang, Yichuan Zhang, Jiahao Ji, Yuanshao Zhu, Zhibo Zhang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10038)  

**Abstract**: POI representation learning plays a crucial role in handling tasks related to user mobility data. Recent studies have shown that enriching POI representations with multimodal information can significantly enhance their task performance. Previously, the textual information incorporated into POI representations typically involved only POI categories or check-in content, leading to relatively weak textual features in existing methods. In contrast, large language models (LLMs) trained on extensive text data have been found to possess rich textual knowledge. However leveraging such knowledge to enhance POI representation learning presents two key challenges: first, how to extract POI-related knowledge from LLMs effectively, and second, how to integrate the extracted information to enhance POI representations. To address these challenges, we propose POI-Enhancer, a portable framework that leverages LLMs to improve POI representations produced by classic POI learning models. We first design three specialized prompts to extract semantic information from LLMs efficiently. Then, the Dual Feature Alignment module enhances the quality of the extracted information, while the Semantic Feature Fusion module preserves its integrity. The Cross Attention Fusion module then fully adaptively integrates such high-quality information into POI representations and Multi-View Contrastive Learning further injects human-understandable semantic information into these representations. Extensive experiments on three real-world datasets demonstrate the effectiveness of our framework, showing significant improvements across all baseline representations. 

**Abstract (ZH)**: POI表示学习在处理与用户移动数据相关任务中起着至关重要的作用。近年来的研究表明，通过多模态信息丰富POI表示可以显著提升其在任务中的性能。此前，集成到POI表示中的文本信息通常仅涉及POI类别或签到内容，导致现有方法中的文本特征相对较弱。相比之下，大型语言模型（LLM）通过对大量文本数据的训练，具备丰富的文本知识。然而，利用这些知识来增强POI表示学习存在两个关键挑战：首先，如何有效地从LLM中提取与POI相关知识，其次，如何整合提取的信息以增强POI表示。为了解决这些挑战，我们提出了一种名为POI-Enhancer的可移植框架，该框架利用LLM来改进经典POI学习模型生成的POI表示。我们首先设计了三种专门的提示来高效地从LLM中提取语义信息。接着，双特征对齐模块提升了提取信息的质量，而语义特征融合模块则保持了其完整性。随后，交叉注意力融合模块实现了对这些高质量信息的全面自适应融合，而多视图对比学习进一步注入了人类可理解的语义信息。在三个真实世界数据集上的广泛实验表明了我们框架的有效性，展示了在所有基准表示中都取得了显著的性能提升。 

---
# Decision Information Meets Large Language Models: The Future of Explainable Operations Research 

**Title (ZH)**: 决策信息与大型语言模型：可解释运筹学的未来 

**Authors**: Yansen Zhang, Qingcan Kang, Wing Yin Yu, Hailei Gong, Xiaojin Fu, Xiongwei Han, Tao Zhong, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.09994)  

**Abstract**: Operations Research (OR) is vital for decision-making in many industries. While recent OR methods have seen significant improvements in automation and efficiency through integrating Large Language Models (LLMs), they still struggle to produce meaningful explanations. This lack of clarity raises concerns about transparency and trustworthiness in OR applications. To address these challenges, we propose a comprehensive framework, Explainable Operations Research (EOR), emphasizing actionable and understandable explanations accompanying optimization. The core of EOR is the concept of Decision Information, which emerges from what-if analysis and focuses on evaluating the impact of complex constraints (or parameters) changes on decision-making. Specifically, we utilize bipartite graphs to quantify the changes in the OR model and adopt LLMs to improve the explanation capabilities. Additionally, we introduce the first industrial benchmark to rigorously evaluate the effectiveness of explanations and analyses in OR, establishing a new standard for transparency and clarity in the field. 

**Abstract (ZH)**: 运筹学（OR）对于许多行业的决策制定至关重要。虽然最近的OR方法通过整合大规模语言模型（LLMs）在自动化和效率方面取得了显著进步，但它们仍难以生成有意义的解释。这种缺乏清晰性引发了关于OR应用透明性和可信度的担忧。为应对这些挑战，我们提出了一种全面框架——可解释运筹学（EOR），强调伴随优化过程的可操作性和易理解的解释。EOR的核心概念是决策信息（Decision Information），它源于“假设分析”，并专注于评估复杂约束（或参数）变化对决策制定的影响。具体而言，我们利用二分图来量化运筹学模型的变化，并采用LLMs来提高解释能力。此外，我们还引入了首个工业基准，以严格评估解释和分析在运筹学中的有效性，从而为该领域的透明性和清晰性设定新的标准。 

---
# Has My System Prompt Been Used? Large Language Model Prompt Membership Inference 

**Title (ZH)**: 我的系统提示被使用了吗？大型语言模型提示成员 inference 

**Authors**: Roman Levin, Valeriia Cherepanova, Abhimanyu Hans, Avi Schwarzschild, Tom Goldstein  

**Link**: [PDF](https://arxiv.org/pdf/2502.09974)  

**Abstract**: Prompt engineering has emerged as a powerful technique for optimizing large language models (LLMs) for specific applications, enabling faster prototyping and improved performance, and giving rise to the interest of the community in protecting proprietary system prompts. In this work, we explore a novel perspective on prompt privacy through the lens of membership inference. We develop Prompt Detective, a statistical method to reliably determine whether a given system prompt was used by a third-party language model. Our approach relies on a statistical test comparing the distributions of two groups of model outputs corresponding to different system prompts. Through extensive experiments with a variety of language models, we demonstrate the effectiveness of Prompt Detective for prompt membership inference. Our work reveals that even minor changes in system prompts manifest in distinct response distributions, enabling us to verify prompt usage with statistical significance. 

**Abstract (ZH)**: 提示工程已经成为一种强大的技术，用于优化大型语言模型（LLMs）以满足特定应用需求，从而加快原型设计速度并提高性能，同时引起了社区对保护专有系统提示的兴趣。在此项工作中，我们通过成员推理的角度探索了提示隐私的一个新颖视角。我们开发了提示侦探（Prompt Detective），这是一种统计方法，用于可靠地判断给定的系统提示是否被第三方语言模型使用。我们的方法依赖于比较不同系统提示对应的两种模型输出分布的统计测试。通过使用各种语言模型进行广泛实验，我们展示了提示侦探在提示成员推理方面的有效性。我们的研究结果表明，即使微小的系统提示更改也会导致不同的响应分布，从而使我们能够以统计显著性验证提示的使用情况。 

---
# Diverse Inference and Verification for Advanced Reasoning 

**Title (ZH)**: 高级推理中的多样推断与验证 

**Authors**: Iddo Drori, Gaston Longhitano, Mao Mao, Seunghwan Hyun, Yuke Zhang, Sungjun Park, Zachary Meeks, Xin-Yu Zhang, Ben Segev, Howard Yong, Nakul Verma, Avi Shporer, Alon Amit, Madeleine Udell  

**Link**: [PDF](https://arxiv.org/pdf/2502.09955)  

**Abstract**: Reasoning LLMs such as OpenAI o1, o3 and DeepSeek R1 have made significant progress in mathematics and coding, yet find challenging advanced tasks such as International Mathematical Olympiad (IMO) combinatorics problems, Abstraction and Reasoning Corpus (ARC) puzzles, and Humanity's Last Exam (HLE) questions. We use a diverse inference approach that combines multiple models and methods at test time. We find that verifying mathematics and code problems, and rejection sampling on other problems is simple and effective. We automatically verify correctness of solutions to IMO problems by Lean, and ARC puzzles by code, and find that best-of-N effectively answers HLE questions. Our approach increases answer accuracy on IMO combinatorics problems from 33.3% to 77.8%, accuracy on HLE questions from 8% to 37%, and solves 80% of ARC puzzles that 948 humans could not and 26.5% of ARC puzzles that o3 high compute does not. Test-time simulations, reinforcement learning, and meta-learning with inference feedback improve generalization by adapting agent graph representations and varying prompts, code, and datasets. Our approach is reliable, robust, and scalable, and in the spirit of reproducible research, we will make it publicly available upon publication. 

**Abstract (ZH)**: 像OpenAI的o1和o3以及DeepSeek的R1这样的大型语言模型（LLMs）在数学和编程领域取得了显著进展，但在解决国际数学奥林匹克（IMO）组合数学问题、抽象与推理语料库（ARC）挑战以及人类最后一考（HLE）的问题等高级任务中仍面临挑战。我们采用了一种多样化的推理方法，在测试时结合了多种模型和方法。我们发现，验证数学和编程问题的答案，并在其他问题上使用拒绝采样，简单且有效。我们通过Lean自动验证IMO问题和ARC谜题的答案正确性，并发现“最佳-N”方法能够有效回答HLE问题。我们的方法将IMO组合数学问题的解答准确性从33.3%提高到77.8%，将HLE问题的解答准确性从8%提高到37%。此外，该方法解决了948名人类无法解答的80%的ARC谜题，以及o3高算力无法解决的26.5%的ARC谜题。通过测试时的模拟、强化学习和基于推理反馈的元学习，我们改进了泛化能力，通过调整代理图的表示和提示、代码和数据集，从而适应不同的任务需求。我们的方法可靠、稳健且具有扩展性。秉承可再现研究的精神，我们将公开发布我们的方法。 

---
# MIR-Bench: Benchmarking LLM's Long-Context Intelligence via Many-Shot In-Context Inductive Reasoning 

**Title (ZH)**: MIR-Bench：通过多次-shot 在上下文归纳推理评估大语言模型的长上下文智能 

**Authors**: Kai Yan, Zhan Ling, Kang Liu, Yifan Yang, Ting-Han Fan, Lingfeng Shen, Zhengyin Du, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.09933)  

**Abstract**: Inductive Reasoning (IR), the ability to summarize rules from examples and apply on new ones, has long been viewed as a primal ability for general intelligence and widely studied by cognitive science and AI researchers. Many benchmarks have been proposed to measure such ability for Large Language Models (LLMs); however, they focus on few-shot (usually $<$10) setting and lack evaluation for aggregating many pieces of information from long contexts. On the other hand, the ever-growing context length of LLMs have brought forth the novel paradigm of many-shot In-Context Learning (ICL), which addresses new tasks with hundreds to thousands of examples without expensive and inefficient fine-tuning. However, many-shot evaluations are mostly focused on classification (a very limited aspect of IR), and popular long-context LLM tasks such as Needle-In-A-Haystack (NIAH) seldom require complicated intelligence for integrating many pieces of information. To fix the issues from both worlds, we propose MIR-Bench, the first many-shot in-context inductive reasoning benchmark that asks LLM to induce output via input-output examples from underlying functions with diverse data format. Based on MIR-Bench, we study many novel problems for inductive reasoning and many-shot ICL, including robustness against erroneous shots and the effect of Chain-of-Thought (CoT), and acquired insightful findings. 

**Abstract (ZH)**: 归纳推理（IR），即从例子中总结规则并应用于新情况的能力，长期以来被视为通用智能的基本能力，受到了认知科学和人工智能研究人员的广泛关注。许多基准测试已经提出，用于衡量大型语言模型（LLMs）的这种能力；然而，这些基准测试主要集中在少量示例（通常少于10个）的设置上，并且缺乏对从长上下文中聚集大量信息的评估。另一方面，LLMs 的不断增长的上下文长度带来了新的 paradigm——大量的上下文学习（Many-Shot In-Context Learning, Many-Shot ICL），该 paradigm能够利用成百上千的示例来处理新任务，而不需要昂贵且低效的微调。然而，大多数 Many-Shot 评估主要集中在分类（这是一种 IR 的非常有限的方面），而流行的长上下文 LLM 任务，如“针扎干草堆”（Needle-In-A-Haystack, NIAH）很少需要复杂的集成大量信息的智能。为了解决这两个方面的不足，我们提出了 MIR-Bench，这是第一个 Many-Shot in-context 归纳推理基准测试，要求 LLM 通过输入-输出示例从底层函数的不同数据格式中推导出输出。基于 MIR-Bench，我们研究了归纳推理和 Many-Shot ICL 的许多新问题，包括对错误示例的鲁棒性以及思维链（Chain-of-Thought, CoT）的影响，并获得了深刻的发现。 

---
# AutoS$^2$earch: Unlocking the Reasoning Potential of Large Models for Web-based Source Search 

**Title (ZH)**: AutoS$^2$earch：解锁大型模型在基于网页的源搜索中的推理潜力 

**Authors**: Zhengqiu Zhu, Yatai Ji, Jiaheng Huang, Yong Zhao, Sihang Qiu, Rusheng Ju  

**Link**: [PDF](https://arxiv.org/pdf/2502.09913)  

**Abstract**: Web-based management systems have been widely used in risk control and industrial safety. However, effectively integrating source search capabilities into these systems, to enable decision-makers to locate and address the hazard (e.g., gas leak detection) remains a challenge. While prior efforts have explored using web crowdsourcing and AI algorithms for source search decision support, these approaches suffer from overheads in recruiting human participants and slow response times in time-sensitive situations. To address this, we introduce AutoS$^2$earch, a novel framework leveraging large models for zero-shot source search in web applications. AutoS$^2$earch operates on a simplified visual environment projected through a web-based display, utilizing a chain-of-thought prompt designed to emulate human reasoning. The multi-modal large language model (MLLMs) dynamically converts visual observations into language descriptions, enabling the LLM to perform linguistic reasoning on four directional choices. Extensive experiments demonstrate that AutoS$^2$earch achieves performance nearly equivalent to human-AI collaborative source search while eliminating dependency on crowdsourced labor. Our work offers valuable insights in using web engineering to design such autonomous systems in other industrial applications. 

**Abstract (ZH)**: 基于Web的管理系统在风险控制和工业安全领域得到了广泛应用。然而，有效地将源搜索能力整合进这些系统，以使决策者能够定位并解决安全隐患（例如，气体泄漏检测）仍然具有挑战性。虽然先前的努力探索了使用网络众包和AI算法来支持源搜索决策，但这些方法在招募人类参与者方面存在负担，并且在时间敏感的情况下响应速度较慢。为解决这一问题，我们提出了一种名为AutoS$^2$earch的新颖框架，该框架利用大规模模型在Web应用程序中进行零样本源搜索。AutoS$^2$earch在基于Web的显示中应用简化视觉环境，利用一个链式思考提示来模仿人类推理。多模态大规模语言模型（Multi-Modal Large Language Models, MLLMs）动态地将视觉观察转化为语言描述，从而允许LLM在四个方向选择上进行语言推理。大量的实验表明，AutoS$^2$earch在性能上几乎与人类-AI协作的源搜索相当，并且消除了对众包劳动力的依赖。我们的研究为利用Web工程设计此类自主系统提供了宝贵见解，在其他工业应用中也具有重要意义。 

---
# Evaluating the Meta- and Object-Level Reasoning of Large Language Models for Question Answering 

**Title (ZH)**: 评估大型语言模型在问答任务中的元级和对象级推理能力 

**Authors**: Nick Ferguson, Liane Guillou, Alan Bundy, Kwabena Nuamah  

**Link**: [PDF](https://arxiv.org/pdf/2502.10338)  

**Abstract**: Large Language Models (LLMs) excel in natural language tasks but still face challenges in Question Answering (QA) tasks requiring complex, multi-step reasoning. We outline the types of reasoning required in some of these tasks, and reframe them in terms of meta-level reasoning (akin to high-level strategic reasoning or planning) and object-level reasoning (embodied in lower-level tasks such as mathematical reasoning). Franklin, a novel dataset with requirements of meta- and object-level reasoning, is introduced and used along with three other datasets to evaluate four LLMs at question answering tasks requiring multiple steps of reasoning. Results from human annotation studies suggest LLMs demonstrate meta-level reasoning with high frequency, but struggle with object-level reasoning tasks in some of the datasets used. Additionally, evidence suggests that LLMs find the object-level reasoning required for the questions in the Franklin dataset challenging, yet they do exhibit strong performance with respect to the meta-level reasoning requirements. 

**Abstract (ZH)**: 大语言模型（LLMs）在自然语言任务中表现出色，但在需要复杂多步推理的问答（QA）任务中仍然面临挑战。我们概述了这些任务中所需的各种推理类型，并将其重新定义为元级推理（类似于高层次的战略推理或规划）和对象级推理（体现在如数学推理等低层任务中）。我们介绍了一个新的数据集Franklin，该数据集包含元级和对象级推理的要求，并与另外三个数据集一起用于评估四款LLM在多步推理问答任务中的性能。人类标注研究的结果表明，LLM在元级推理方面表现出较高的频率，但在某些数据集中的对象级推理任务中却遇到困难。此外，证据表明，LLM在Franklin数据集中所需的对象级推理是一项挑战，但它们在满足元级推理要求方面表现出很强的能力。 

---
# Process Reward Models for LLM Agents: Practical Framework and Directions 

**Title (ZH)**: 面向LLM代理的过程奖励模型：实用框架与发展方向 

**Authors**: Sanjiban Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2502.10325)  

**Abstract**: We introduce Agent Process Reward Models (AgentPRM), a simple and scalable framework for training LLM agents to continually improve through interactions. AgentPRM follows a lightweight actor-critic paradigm, using Monte Carlo rollouts to compute reward targets and optimize policies. It requires minimal modifications to existing RLHF pipelines, making it easy to integrate at scale. Beyond AgentPRM, we propose InversePRM, which learns process rewards directly from demonstrations without explicit outcome supervision. We also explore key challenges and opportunities, including exploration, process reward shaping, and model-predictive reasoning. We evaluate on ALFWorld benchmark, show that small 3B models trained with AgentPRM and InversePRM outperform strong GPT-4o baselines, and analyze test-time scaling, reward hacking, and more. Our code is available at: this https URL. 

**Abstract (ZH)**: 我们引入了代理过程奖励模型（AgentPRM）——一种简洁且可扩展的框架，用于训练大规模语言模型代理通过交互不断改进。AgentPRM 采用轻量级的演员-批评家范式，并利用蒙特卡洛展开来计算奖励目标并优化策略。它只需对现有的 RLHF 流程进行最少的修改，使其易于大规模集成。除了 AgentPRM，我们还提出了逆向过程奖励模型（InversePRM），该模型直接从演示学习过程奖励，而无需显式的结果监督。我们还探索了关键的挑战和机会，包括探索、过程奖励塑造和模型预测推理。我们使用 ALFWorld 基准进行评估，并展示了使用 AgentPRM 和 InversePRM 训练的小型 3B 模型超越了强大的 GPT-4o 基线模型的结果。我们分析了测试时缩放、奖励劫持等问题，并提供了更多细节。我们的代码可从以下链接获取：[提供链接]

翻译如下：

我们引入了代理过程奖励模型(AgentPRM)，这是一种简单且可扩展的框架，用于通过交互训练大规模语言模型代理使其不断改进。AgentPRM 采用轻量级的演员-批评家范式，使用蒙特卡洛展开计算奖励目标并优化策略。它只需要对现有的 RLHF 流程进行最小的修改，使其在大规模部署中易于集成。除了 AgentPRM，我们还提出了逆过程奖励模型(InversePRM)，它直接从演示中学习过程奖励，而无需显式的结果监督。我们还探索了关键的挑战和机遇，包括探索、过程奖励塑造和模型预测推理。我们使用 ALFWorld 基准对这些模型进行了评估，结果表明使用 AgentPRM 和 InversePRM 训练的小型 3B 模型优于强大的 GPT-4o 基线。我们分析了测试时扩展、奖励劫持等问题，并进行了更多的探讨。我们的代码库可以在以下链接处获得：[提供链接] 

---
# Automated Hypothesis Validation with Agentic Sequential Falsifications 

**Title (ZH)**: 自动假设验证与代理性的序贯反驳 

**Authors**: Kexin Huang, Ying Jin, Ryan Li, Michael Y. Li, Emmanuel Candès, Jure Leskovec  

**Link**: [PDF](https://arxiv.org/pdf/2502.09858)  

**Abstract**: Hypotheses are central to information acquisition, decision-making, and discovery. However, many real-world hypotheses are abstract, high-level statements that are difficult to validate directly. This challenge is further intensified by the rise of hypothesis generation from Large Language Models (LLMs), which are prone to hallucination and produce hypotheses in volumes that make manual validation impractical. Here we propose Popper, an agentic framework for rigorous automated validation of free-form hypotheses. Guided by Karl Popper's principle of falsification, Popper validates a hypothesis using LLM agents that design and execute falsification experiments targeting its measurable implications. A novel sequential testing framework ensures strict Type-I error control while actively gathering evidence from diverse observations, whether drawn from existing data or newly conducted procedures. We demonstrate Popper on six domains including biology, economics, and sociology. Popper delivers robust error control, high power, and scalability. Furthermore, compared to human scientists, Popper achieved comparable performance in validating complex biological hypotheses while reducing time by 10 folds, providing a scalable, rigorous solution for hypothesis validation. 

**Abstract (ZH)**: 假设在信息获取、决策制定和发现中起着核心作用。然而，许多实际世界的假设往往是抽象和高层次的陈述，难以直接验证。随着大型语言模型（LLMs）生成假设的现象出现，这一挑战变得更加严峻，因为这些模型容易产生幻觉，并生成大量需要人工验证的假设，从而使手动验证变得不切实际。为此，我们提出了Popper，这是一种代理驱动的框架，旨在实现对自由格式假设的 rigorous 自动验证。Popper 依据 Karl Popper 的证伪原则，使用 LLM 代理设计并执行针对假设可测量影响的证伪实验进行验证。一个新颖的序列性测试框架确保了严格的第一类错误控制，同时积极从各种观察中收集证据，这些观察可以来自于现有数据，也可以是新开展的程序。我们分别在生物学、经济学和社会学六个领域展示了 Popper 的应用。Popper 实现了稳健的错误控制、强大的检验效能和可扩展性。此外，与人类科学家相比，Popper 在验证复杂的生物学假设方面取得了可比的性能，同时将时间缩短了10倍，从而提供了一个高效的、严谨的假设验证解决方案。 

---
# A Solver-Aided Hierarchical Language for LLM-Driven CAD Design 

**Title (ZH)**: 辅助求解器的分层语言：面向LLM驱动的CAD设计 

**Authors**: Benjamin T. Jones, Felix Hähnlein, Zihan Zhang, Maaz Ahmad, Vladimir Kim, Adriana Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2502.09819)  

**Abstract**: Large language models (LLMs) have been enormously successful in solving a wide variety of structured and unstructured generative tasks, but they struggle to generate procedural geometry in Computer Aided Design (CAD). These difficulties arise from an inability to do spatial reasoning and the necessity to guide a model through complex, long range planning to generate complex geometry. We enable generative CAD Design with LLMs through the introduction of a solver-aided, hierarchical domain specific language (DSL) called AIDL, which offloads the spatial reasoning requirements to a geometric constraint solver. Additionally, we show that in the few-shot regime, AIDL outperforms even a language with in-training data (OpenSCAD), both in terms of generating visual results closer to the prompt and creating objects that are easier to post-process and reason about. 

**Abstract (ZH)**: 大型语言模型（LLM）在解决结构化和非结构化生成任务方面取得了巨大成功，但在计算机辅助设计（CAD）中生成工艺几何图形方面仍然面临挑战。这些困难源于其在空间推理能力上的不足，以及需要通过复杂的、长范围的规划来生成复杂的几何图形。我们通过引入一种被称为AIDL的解决器辅助型分层领域特定语言（DSL），使LLM能够实现生成CAD设计。AIDL将空间推理的要求卸载到一个几何约束求解器中。此外，我们还展示了在少样本情况下，AIDL在生成与提示更接近的视觉结果和创建更易于后续处理和推理的对象方面，甚至超过了具备训练数据支持的语言（如OpenSCAD）。 

---
# TableTalk: Scaffolding Spreadsheet Development with a Language Agent 

**Title (ZH)**: TableTalk：语言代理辅助的电子表格开发支架 

**Authors**: Jenny T. Liang, Aayush Kumar, Yasharth Bajpai, Sumit Gulwani, Vu Le, Chris Parnin, Arjun Radhakrishna, Ashish Tiwari, Emerson Murphy-Hill, Guastavo Soares  

**Link**: [PDF](https://arxiv.org/pdf/2502.09787)  

**Abstract**: Despite its ubiquity in the workforce, spreadsheet programming remains challenging as programmers need both spreadsheet-specific knowledge (e.g., APIs to write formulas) and problem-solving skills to create complex spreadsheets. Large language models (LLMs) can help automate aspects of this process, and recent advances in planning and reasoning have enabled language agents, which dynamically plan, use tools, and take iterative actions to complete complex tasks. These agents observe, plan, and act, making them well-suited to scaffold spreadsheet programming by following expert processes.
We present TableTalk, a language agent that helps programmers build spreadsheets conversationally. Its design reifies three design principles -- scaffolding, flexibility, and incrementality -- which we derived from two studies of seven programmers and 62 Excel templates. TableTalk structures spreadsheet development by generating step-by-step plans and suggesting three next steps users can choose from. It also integrates tools that enable incremental spreadsheet construction. A user study with 20 programmers shows that TableTalk produces spreadsheets 2.3 times more likely to be preferred over a baseline agent, while reducing cognitive load and time spent reasoning about spreadsheet actions by 12.6%. TableTalk's approach has implications for human-agent collaboration. This includes providing persistent direct manipulation interfaces for stopping or undoing agent actions, while ensuring that such interfaces for accepting actions can be deactivated. 

**Abstract (ZH)**: 尽管电子表格编程在工作中无处不在，但编程人员仍需掌握电子表格特定的知识（例如，编写公式的API）和解决问题的能力，才能创建复杂的电子表格。大型语言模型（LLMs）可以帮助自动化此过程中的某些方面，近期规划和推理方面的进展使得语言代理能够动态规划、使用工具并采取迭代行动以完成复杂任务。这些代理观察、计划和行动，因此它们非常适合通过遵循专家流程来搭建电子表格编程。

我们提出了一种名为TableTalk的语言代理，它帮助编程人员通过对话方式构建电子表格。其设计遵循了三条设计原则——搭建、灵活性和逐步性，这三条原则源自对七位编程人员和62个Excel模板的两个研究。TableTalk通过生成逐步计划并建议用户提供三种可选的下一步操作，来结构化电子表格的开发工作。此外，TableTalk还整合了用于逐步构建电子表格的工具。一项涉及20位编程人员的用户研究表明，与基准代理相比，TableTalk生成的电子表格被更偏好（2.3倍的可能性），同时减少了12.6%的认知负担和用于推理电子表格操作的时间。TableTalk的方法对人机协作具有重要意义，包括提供持久的直接操作界面以停止或撤销代理操作，同时确保可接受操作的界面能够被禁用。

这种人机协作的方法还具有其他影响。它包括提供持久的直接操作界面以停止或撤销代理操作，同时确保接受操作的界面可以被禁用。 

---
# Trust at Your Own Peril: A Mixed Methods Exploration of the Ability of Large Language Models to Generate Expert-Like Systems Engineering Artifacts and a Characterization of Failure Modes 

**Title (ZH)**: 自担风险的信任：大型语言模型生成专家级系统工程制品能力的混合方法探索及失败模式 characterization 

**Authors**: Taylan G. Topcu, Mohammed Husain, Max Ofsa, Paul Wach  

**Link**: [PDF](https://arxiv.org/pdf/2502.09690)  

**Abstract**: Multi-purpose Large Language Models (LLMs), a subset of generative Artificial Intelligence (AI), have recently made significant progress. While expectations for LLMs to assist systems engineering (SE) tasks are paramount; the interdisciplinary and complex nature of systems, along with the need to synthesize deep-domain knowledge and operational context, raise questions regarding the efficacy of LLMs to generate SE artifacts, particularly given that they are trained using data that is broadly available on the internet. To that end, we present results from an empirical exploration, where a human expert-generated SE artifact was taken as a benchmark, parsed, and fed into various LLMs through prompt engineering to generate segments of typical SE artifacts. This procedure was applied without any fine-tuning or calibration to document baseline LLM performance. We then adopted a two-fold mixed-methods approach to compare AI generated artifacts against the benchmark. First, we quantitatively compare the artifacts using natural language processing algorithms and find that when prompted carefully, the state-of-the-art algorithms cannot differentiate AI-generated artifacts from the human-expert benchmark. Second, we conduct a qualitative deep dive to investigate how they differ in terms of quality. We document that while the two-material appear very similar, AI generated artifacts exhibit serious failure modes that could be difficult to detect. We characterize these as: premature requirements definition, unsubstantiated numerical estimates, and propensity to overspecify. We contend that this study tells a cautionary tale about why the SE community must be more cautious adopting AI suggested feedback, at least when generated by multi-purpose LLMs. 

**Abstract (ZH)**: 多用途大型语言模型（LLMs），作为生成性人工智能（AI）的一个子类，最近取得了显著进展。虽然人们期望LLMs能够在系统工程（SE）任务中发挥重要作用，但系统的跨学科复杂性以及需要综合深厚的领域知识和操作环境，使得人们对LLMs生成SE制品的有效性产生了疑问，特别是考虑到它们的训练数据绝大部分来自互联网。为了解决这一问题，我们进行了一项经验性探索，将由人类专家生成的SE制品作为基准，进行解析，并通过提示工程将这些制品输入到不同的LLMs中，以生成典型的SE制品片段。这一过程不进行任何微调或校准，以记录基线LLM性能。然后我们采用两折混合方法来对比AI生成的制品与基准之间的差异。首先，我们使用自然语言处理算法定量比较这些制品，发现当精心设计提示时，最先进的算法无法将AI生成的制品与人类专家基准区分开来。其次，我们进行深入定性分析，探讨它们在质量上的差异。我们记录发现，虽然两个材料看起来非常相似，但AI生成的制品表现出一些难以检测的重大失效模式，包括：过早定义需求、缺乏支持的数值估计以及过度详细描述的倾向。我们认为，这项研究揭示了系统工程界在采纳多用途LLMs生成的AI建议反馈时应更加谨慎的原因。 

---
# The Hidden Dimensions of LLM Alignment: A Multi-Dimensional Safety Analysis 

**Title (ZH)**: LLM对齐的隐含维度：多维度安全性分析 

**Authors**: Wenbo Pan, Zhichao Liu, Qiguang Chen, Xiangyang Zhou, Haining Yu, Xiaohua Jia  

**Link**: [PDF](https://arxiv.org/pdf/2502.09674)  

**Abstract**: Large Language Models' safety-aligned behaviors, such as refusing harmful queries, can be represented by linear directions in activation space. Previous research modeled safety behavior with a single direction, limiting mechanistic understanding to an isolated safety feature. In this work, we discover that safety-aligned behavior is jointly controlled by multi-dimensional directions. Namely, we study the vector space of representation shifts during safety fine-tuning on Llama 3 8B for refusing jailbreaks. By studying orthogonal directions in the space, we first find that a dominant direction governs the model's refusal behavior, while multiple smaller directions represent distinct and interpretable features like hypothetical narrative and role-playing. We then measure how different directions promote or suppress the dominant direction, showing the important role of secondary directions in shaping the model's refusal representation. Finally, we demonstrate that removing certain trigger tokens in harmful queries can mitigate these directions to bypass the learned safety capability, providing new insights on understanding safety alignment vulnerability from a multi-dimensional perspective. Code and artifacts are available at this https URL. 

**Abstract (ZH)**: 大规模语言模型的安全对齐行为，如拒绝有害查询，可以通过激活空间中的线性方向来表示。以往的研究使用单一方向来建模安全行为，这限制了对其机制理解的单一安全特性。在本研究中，我们发现安全对齐行为实际上是由多维度的方向共同控制的。具体来说，我们研究了在 Llama 3 8B 上进行拒绝对抗攻击调整期间表示变化的向量空间。通过研究这种空间中的正交方向，我们发现一个主导方向主导了模型的拒绝行为，而多个较小的方向则代表了独特的可解释特征，如假设性叙述和角色扮演。随后，我们测量不同方向如何促进或抑制主导方向，展示了次要方向在塑造模型拒绝表示中的重要作用。最后，我们证明去除有害查询中的某些触发词可以缓解这些方向，使其能够绕过学习到的安全能力，从而从多维度角度为理解安全对齐脆弱性提供了新的见解。相关代码和资料可在以下链接获取：this https URL。 

---
# Are Smarter LLMs Safer? Exploring Safety-Reasoning Trade-offs in Prompting and Fine-Tuning 

**Title (ZH)**: 更聪明的大型语言模型更安全吗？探索提示与微调中的安全推理权衡 

**Authors**: Ang Li, Yichuan Mo, Mingjie Li, Yifei Wang, Yisen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09673)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable success across various NLP benchmarks. However, excelling in complex tasks that require nuanced reasoning and precise decision-making demands more than raw language proficiency--LLMs must reason, i.e., think logically, draw from past experiences, and synthesize information to reach conclusions and take action. To enhance reasoning abilities, approaches such as prompting and fine-tuning have been widely explored. While these methods have led to clear improvements in reasoning, their impact on LLM safety remains less understood. In this work, we investigate the interplay between reasoning and safety in LLMs. We highlight the latent safety risks that arise as reasoning capabilities improve, shedding light on previously overlooked vulnerabilities. At the same time, we explore how reasoning itself can be leveraged to enhance safety, uncovering potential mitigation strategies. By examining both the risks and opportunities in reasoning-driven LLM safety, our study provides valuable insights for developing models that are not only more capable but also more trustworthy in real-world deployments. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在各种自然语言处理（NLP）基准测试中取得了显著的成功。然而，在执行要求精细推理和精确决策的复杂任务时，光有语言能力是不够的—LLMs 必须能够进行推理，即逻辑思考、借鉴过往经验并综合信息来得出结论并采取行动。为了增强推理能力，人们广泛探索了提示和微调等方法。尽管这些方法在提高推理能力方面取得了明显进步，但它们对LLMs安全性的潜在影响仍不甚明了。在本研究中，我们探讨了推理与安全性在LLMs中的相互作用。我们强调了随着推理能力的提升而产生的潜在安全风险，揭示了以往未被重视的漏洞。同时，我们还探讨了如何利用推理本身来增强安全性，发现潜在的缓解策略。通过研究推理驱动的LLMs安全性中的风险与机遇，本研究为开发出不仅更具能力而且在实际部署中更具可信度的模型提供了宝贵见解。 

---
# The Science of Evaluating Foundation Models 

**Title (ZH)**: 评价基础模型的科学方法 

**Authors**: Jiayi Yuan, Jiamu Zhang, Andrew Wen, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.09670)  

**Abstract**: The emergent phenomena of large foundation models have revolutionized natural language processing. However, evaluating these models presents significant challenges due to their size, capabilities, and deployment across diverse applications. Existing literature often focuses on individual aspects, such as benchmark performance or specific tasks, but fails to provide a cohesive process that integrates the nuances of diverse use cases with broader ethical and operational considerations. This work focuses on three key aspects: (1) Formalizing the Evaluation Process by providing a structured framework tailored to specific use-case contexts, (2) Offering Actionable Tools and Frameworks such as checklists and templates to ensure thorough, reproducible, and practical evaluations, and (3) Surveying Recent Work with a targeted review of advancements in LLM evaluation, emphasizing real-world applications. 

**Abstract (ZH)**: 大型基础模型中出现的现象已经革命性地改变了自然语言处理。然而，由于模型的规模、功能及其在多样应用中的部署，对其进行评估面临着重大挑战。现有文献通常集中在个别方面，如基准性能或特定任务上，但未能提供一个能够整合多样化应用场景及其更广泛伦理与操作考量的整体评估过程。本研究聚焦于三个关键方面：(1) 规范化评估流程，通过提供适应特定应用场景上下文的结构化框架，(2) 提供实用工具和框架，如检查列表和模板，确保评估过程全面、可重复且实用，以及(3) 回顾近期工作，对大型语言模型评估的进展进行有针对性的综述，强调其实用应用。 

---
# Principled Data Selection for Alignment: The Hidden Risks of Difficult Examples 

**Title (ZH)**: 遵循原则的数据选择以实现对齐：困难例证的潜在风险 

**Authors**: Chengqian Gao, Haonan Li, Liu Liu, Zeke Xie, Peilin Zhao, Zhiqiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.09650)  

**Abstract**: The alignment of large language models (LLMs) often assumes that using more clean data yields better outcomes, overlooking the match between model capacity and example difficulty. Challenging this, we propose a new principle: Preference data vary in difficulty, and overly difficult examples hinder alignment, by exceeding the model's capacity. Through systematic experimentation, we validate this principle with three key findings: (1) preference examples vary in difficulty, as evidenced by consistent learning orders across alignment runs; (2) overly difficult examples significantly degrade performance across four LLMs and two datasets; and (3) the capacity of a model dictates its threshold for handling difficult examples, underscoring a critical relationship between data selection and model capacity. Building on this principle, we introduce Selective DPO, which filters out overly difficult examples. This simple adjustment improves alignment performance by 9-16% in win rates on the AlpacaEval 2 benchmark compared to the DPO baseline, suppressing a series of DPO variants with different algorithmic adjustments. Together, these results illuminate the importance of aligning data difficulty with model capacity, offering a transformative perspective for improving alignment strategies in LLMs. Code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）的对齐往往假设使用更多干净的数据可以获得更好的结果，而忽略了模型容量与示例难度之间的匹配。我们提出一个新的原则：偏好数据的难度各不相同，过于复杂的示例会妨碍对齐，因为它们超出了模型的容量。通过系统的实验，我们验证了这一原则，得到了三个关键发现：（1）偏好示例的难度是不同的，这体现在对齐试验中一致的学习顺序；（2）过于复杂的示例在四个LLMs和两个数据集上显著降低了性能；（3）模型的容量决定了其处理困难示例的阈值，突显了数据选择与模型容量之间的重要关系。基于这一原则，我们引入了选择性DPO，该方法过滤掉过于复杂的示例。这一简单的调整在AlpacaEval 2基准测试中将胜率提高了9-16%，超过了多种具有不同算法调整的DPO变体。总之，这些结果揭示了数据难度与模型容量之间对齐的重要性，并为改进LLMs中的对齐策略提供了转变性的视角。代码可在以下链接获取：this https URL。 

---
# Jailbreaking to Jailbreak 

**Title (ZH)**: 越狱到越狱

这里的“越狱”是指计算机或移动设备上的安全机制被绕过，以获得更高的系统权限。若这是某个论文的标题或内容摘要，在翻译时应确保保留其特定含义和学术规范。如果有更多上下文信息或其他具体句子需要翻译，请提供，以便更准确地翻译。 

**Authors**: Jeremy Kritz, Vaughn Robinson, Robert Vacareanu, Bijan Varjavand, Michael Choi, Bobby Gogov, Scale Red Team, Summer Yue, Willow E. Primack, Zifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.09638)  

**Abstract**: Refusal training on Large Language Models (LLMs) prevents harmful outputs, yet this defense remains vulnerable to both automated and human-crafted jailbreaks. We present a novel LLM-as-red-teamer approach in which a human jailbreaks a refusal-trained LLM to make it willing to jailbreak itself or other LLMs. We refer to the jailbroken LLMs as $J_2$ attackers, which can systematically evaluate target models using various red teaming strategies and improve its performance via in-context learning from the previous failures. Our experiments demonstrate that Sonnet 3.5 and Gemini 1.5 pro outperform other LLMs as $J_2$, achieving 93.0% and 91.0% attack success rates (ASRs) respectively against GPT-4o (and similar results across other capable LLMs) on Harmbench. Our work not only introduces a scalable approach to strategic red teaming, drawing inspiration from human red teamers, but also highlights jailbreaking-to-jailbreak as an overlooked failure mode of the safeguard. Specifically, an LLM can bypass its own safeguards by employing a jailbroken version of itself that is willing to assist in further jailbreaking. To prevent any direct misuse with $J_2$, while advancing research in AI safety, we publicly share our methodology while keeping specific prompting details private. 

**Abstract (ZH)**: 基于大型语言模型（LLM）的拒绝训练可以防止有害输出，但这种防御仍然容易受到自动化和人工构建的“逃逸”攻击的攻击。我们提出了一种新的LLM作为红队成员的方法，在这种方法中，一个人类通过“逃逸”一个拒绝训练的LLM，使其愿意自我“逃逸”或对其他LLM进行“逃逸”。我们称这些被“逃逸”的LLM为$J_2$攻击者，这些攻击者可以通过多种红队策略系统地评估目标模型，并通过从之前的失败中进行上下文学习来提高其性能。我们的实验表明，Sonnet 3.5和Gemini 1.5作为$J_2$攻击者表现出色，分别在Harmbench上对GPT-4o（以及其他能力相当的LLM）取得了93.0%和91.0%的成功攻击率（ASR）。我们的工作不仅引入了一种可扩展的红队策略方法，借鉴了人类红队成员的做法，还突显了“逃逸”以“逃逸”作为安全防护未被充分考虑的失败模式。具体来说，LLM可以通过使用一个愿意协助进一步“逃逸”的被“逃逸”的版本自身来绕过其自身的防护。为了防止任何直接滥用$J_2$，同时推进人工智能安全研究，我们公开分享了我们的方法，但保留了具体的提示细节。 

---
# Reading between the Lines: Can LLMs Identify Cross-Cultural Communication Gaps? 

**Title (ZH)**: 在字里行间探寻文化差异：大型语言模型能否识别跨文化沟通差距？ 

**Authors**: Sougata Saha, Saurabh Kumar Pandey, Harshit Gupta, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2502.09636)  

**Abstract**: In a rapidly globalizing and digital world, content such as book and product reviews created by people from diverse cultures are read and consumed by others from different corners of the world. In this paper, we investigate the extent and patterns of gaps in understandability of book reviews due to the presence of culturally-specific items and elements that might be alien to users from another culture. Our user-study on 57 book reviews from Goodreads reveal that 83\% of the reviews had at least one culture-specific difficult-to-understand element. We also evaluate the efficacy of GPT-4o in identifying such items, given the cultural background of the reader; the results are mixed, implying a significant scope for improvement. Our datasets are available here: this https URL 

**Abstract (ZH)**: 在快速全球化的数字时代，来自不同文化背景的人们撰写或生成的书评和产品评论被世界各地的用户阅读和消费。本文探讨了由于存在可能对其他文化背景用户来说不熟悉的文化特有项和元素，而导致书评理解差距的程度和模式。我们在对来自Goodreads的57篇书评进行用户研究后发现，83%的书评中至少包含一个文化特有、难以理解的元素。我们还评估了GPT-4o在考虑读者文化背景时识别这些元素的有效性；结果参差不齐，表明还有很大的改进空间。我们的数据集可通过以下链接访问：this https URL 

---
# Enhancing Multilingual LLM Pretraining with Model-Based Data Selection 

**Title (ZH)**: 基于模型的数据选择增强多语言大语言模型预训练 

**Authors**: Bettina Messmer, Vinko Sabolčec, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2502.10361)  

**Abstract**: Dataset curation has become a basis for strong large language model (LLM) performance. While various rule-based filtering heuristics exist for English and multilingual datasets, model-based filtering techniques have primarily focused on English. To address the disparity stemming from limited research on non-English languages, we propose a model-based filtering framework for multilingual datasets that aims to identify a diverse set of structured and knowledge-rich samples. Our approach emphasizes transparency, simplicity, and efficiency, leveraging Transformer- and FastText-based classifiers to ensure the broad accessibility of our technique and data. We conduct comprehensive ablation studies on the FineWeb-2 web crawl dataset across diverse language families, scripts, and resource availability to demonstrate the effectiveness of our method. Training a 1B-parameter Llama model for 70B and 119B tokens, our approach can match the baseline MMLU score with as little as 15% of the training tokens, while also improving across other benchmarks. These findings provide strong evidence for the generalizability of our approach to other languages. As a result, we extend our framework to 20 languages for which we release the refined pretraining datasets. 

**Abstract (ZH)**: 数据集整理已成为强大大型语言模型（LLM）性能的基础。虽然各种基于规则的过滤启发式方法已应用于英语和多语言数据集，但基于模型的过滤技术主要集中在英语上。为了解决因对非英语语言研究不足而产生的差距，我们提出了一种针对多语言数据集的基于模型的过滤框架，旨在识别一个结构多样且知识丰富样本集。我们的方法强调透明性、简洁性和效率，通过利用基于Transformer和FastText的分类器，确保我们的技术和数据具有广泛的可访问性。我们在FineWeb-2网络爬虫数据集上进行了跨多种语言家族、书写系统和资源可获得性的全面消融研究，以展示我们方法的有效性。通过对70B和119B令牌训练具有1B参数的Llama模型，我们的方法可以用最少15%的训练令牌达到基线MMLU得分，并且在其他基准测试中也表现出改进。这些发现为我们的方法在其他语言中的普适性提供了强有力的证据。因此，我们将框架扩展到20种语言，并发布了这些语言的精炼预训练数据集。 

---
# Prediction hubs are context-informed frequent tokens in LLMs 

**Title (ZH)**: 预测枢纽是LLM中受到上下文指导的频繁令牌 

**Authors**: Beatrix M. G. Nielsen, Iuri Macocco, Marco Baroni  

**Link**: [PDF](https://arxiv.org/pdf/2502.10201)  

**Abstract**: Hubness, the tendency for few points to be among the nearest neighbours of a disproportionate number of other points, commonly arises when applying standard distance measures to high-dimensional data, often negatively impacting distance-based analysis. As autoregressive large language models (LLMs) operate on high-dimensional representations, we ask whether they are also affected by hubness. We first show, theoretically, that the only representation comparison operation performed by LLMs, namely that between context and unembedding vectors to determine continuation probabilities, is not characterized by the concentration of distances phenomenon that typically causes the appeareance of nuisance hubness. We then empirically show that this comparison still leads to a high degree of hubness, but the hubs in this case do not constitute a disturbance. They are rather the result of context-modulated frequent tokens often appearing in the pool of likely candidates for next token prediction. On the other hand, when other distance computations involving LLM representations are performed, we do not have the same theoretical guarantees, and, indeed, we see nuisance hubs appear. In summary, our work highlights, on the one hand, how hubness, while omnipresent in high-dimensional spaces, is not always a negative property that needs to be mitigated, and, on the other hand, it shows that various widely-used LLMs have developed a guessing strategy that consists in constantly assigning a high probability to frequent tokens. 

**Abstract (ZH)**: 高维空间中的聚众现象（hubness），指的是少量点成为其他大量点的近邻的概率不成比例地高，通常在应用标准的距离度量于高维数据时出现，往往会对基于距离的分析产生负面影响。由于自回归的大规模语言模型（LLMs）处理高维表示，我们质疑它们是否也会受到高维空间中的聚众现象影响。我们首先从理论上证明，LLMs执行的唯一表示比较操作——即根据上下文向量与未嵌入向量之间的比较来确定续写概率——并不表现出导致不良聚众现象的近距离度量集中现象。然后，我们通过实验展示了这种比较仍然会导致高聚众现象，但这些聚众点在此情况下并不构成干扰。相反，它们是语境调节下的高频词汇频繁出现在下一个词预测备选库中的结果。另一方面，当进行涉及LLMs表示的其他距离计算时，我们没有相同的理论保障，事实上，我们观察到不良的聚众现象出现。总之，我们的研究突出了如下方面：尽管聚众现象在高维空间中普遍存在，但它并不总是需要被抑制的负面属性，同时展示了各种广泛使用的LLMs发展了一种猜测策略，即持续地为高频词汇分配高概率。 

---
# ORI: O Routing Intelligence 

**Title (ZH)**: ORI: 路由智能 

**Authors**: Ahmad Shadid, Rahul Kumar, Mohit Mayank  

**Link**: [PDF](https://arxiv.org/pdf/2502.10051)  

**Abstract**: Single large language models (LLMs) often fall short when faced with the ever-growing range of tasks, making a single-model approach insufficient. We address this challenge by proposing ORI (O Routing Intelligence), a dynamic framework that leverages a set of LLMs. By intelligently routing incoming queries to the most suitable model, ORI not only improves task-specific accuracy, but also maintains efficiency. Comprehensive evaluations across diverse benchmarks demonstrate consistent accuracy gains while controlling computational overhead. By intelligently routing queries, ORI outperforms the strongest individual models by up to 2.7 points on MMLU and 1.8 points on MuSR, ties the top performance on ARC, and on BBH. These results underscore the benefits of a multi-model strategy and demonstrate how ORI's adaptive architecture can more effectively handle diverse tasks, offering a scalable, high-performance solution for a system of multiple large language models. 

**Abstract (ZH)**: 单个大规模语言模型（LLMs）在面对日益增长的任务范围时往往表现不足，使得单模型方法变得不够充分。为应对这一挑战，我们提出了一种名为ORI（O Routing Intelligence）的动态框架，该框架利用了一组LLMs。通过智能地将传入查询路由到最适合的模型，ORI 不仅提高了任务特定的准确性，还保持了效率。通过对各种基准的全面评估，结果显示，在控制计算开销的情况下，ORI 仍然能够实现一致的准确性提升。通过对查询的智能路由，ORI 在MMLU上的表现比最强的单个模型高出2.7个百分点，在MuSR上的表现高出1.8个百分点，在ARC和BBH上达到了顶级性能。这些结果强调了多模型策略的优势，并展示了ORI的自适应架构如何更有效地处理多样化的任务，提供了用于多大规模语言模型系统的可扩展且高性能的解决方案。 

---
# Large Language Diffusion Models 

**Title (ZH)**: 大型语言扩散模型 

**Authors**: Shen Nie, Fengqi Zhu, Zebin You, Xiaolu Zhang, Jingyang Ou, Jun Hu, Jun Zhou, Yankai Lin, Ji-Rong Wen, Chongxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.09992)  

**Abstract**: Autoregressive models (ARMs) are widely regarded as the cornerstone of large language models (LLMs). We challenge this notion by introducing LLaDA, a diffusion model trained from scratch under the pre-training and supervised fine-tuning (SFT) paradigm. LLaDA models distributions through a forward data masking process and a reverse process, parameterized by a vanilla Transformer to predict masked tokens. By optimizing a likelihood bound, it provides a principled generative approach for probabilistic inference. Across extensive benchmarks, LLaDA demonstrates strong scalability, outperforming our self-constructed ARM baselines. Remarkably, LLaDA 8B is competitive with strong LLMs like LLaMA3 8B in in-context learning and, after SFT, exhibits impressive instruction-following abilities in case studies such as multi-turn dialogue. Moreover, LLaDA addresses the reversal curse, surpassing GPT-4o in a reversal poem completion task. Our findings establish diffusion models as a viable and promising alternative to ARMs, challenging the assumption that key LLM capabilities discussed above are inherently tied to ARMs. 

**Abstract (ZH)**: 自回归模型（ARMs）长期以来被视为大规模语言模型（LLMs）的基础。我们提出了对这一观点的挑战，通过引入在预训练和监督微调（SFT）范式下从头开始训练的扩散模型——LLaDA。LLaDA 通过前向数据遮蔽过程和一个反向过程来建模分布，其中反向过程通过一个普通的Transformer来预测遮蔽的标记。通过优化似然性边界，它提供了一种用于概率推理的原则性生成方法。在广泛的基准测试中，LLaDA 显示出了强大的可扩展性，超越了我们自行构建的ARM基线。令人惊讶的是，LLaDA 8B在上下文学习任务中与强大的LLMs（如LLaMA3 8B）具有竞争力，并且经过SFT后，在案例研究中（如多轮对话）展示了出色的操作指令能力。此外，LLaDA 解决了逆向难题，在逆向诗歌生成任务中超越了GPT-4o。我们的研究结果确立了扩散模型作为一种可行且有前途的ARM替代方案，挑战了关键LLM能力上述特性与ARMs密不可分这一假设。 

---
# A Preliminary Exploration with GPT-4o Voice Mode 

**Title (ZH)**: 初步探究：GPT-4o语音模式 

**Authors**: Yu-Xiang Lin, Chih-Kai Yang, Wei-Chih Chen, Chen-An Li, Chien-yu Huang, Xuanjun Chen, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.09940)  

**Abstract**: With the rise of multimodal large language models, GPT-4o stands out as a pioneering model, driving us to evaluate its capabilities. This report assesses GPT-4o across various tasks to analyze its audio processing and reasoning abilities. We find that GPT-4o exhibits strong knowledge in audio, speech, and music understanding, performing well in tasks like intent classification, spoken command classification, semantic and grammatical reasoning., multilingual speech recognition, and singing analysis. It also shows greater robustness against hallucinations than other large audio-language models (LALMs). However, it struggles with tasks such as audio duration prediction and instrument classification. Additionally, GPT-4o's safety mechanisms cause it to decline tasks like speaker identification, age classification, MOS prediction, and audio deepfake detection. Notably, the model exhibits a significantly different refusal rate when responding to speaker verification tasks on different datasets. This is likely due to variations in the accompanying instructions or the quality of the input audio, suggesting the sensitivity of its built-in safeguards. Finally, we acknowledge that model performance varies with evaluation protocols. This report only serves as a preliminary exploration of the current state of LALMs. 

**Abstract (ZH)**: 随着多模态大语言模型的兴起，GPT-4o 凸显为一种开创性的模型，推动我们对其能力进行评估。本报告通过多种任务评估 GPT-4o，以分析其音频处理和推理能力。研究发现，GPT-4o 在音频、语音和音乐理解方面表现出强大的知识基础，能够在意图分类、口头命令分类、语义和语法推理、多语言语音识别和歌唱分析等任务中表现出色。此外，GPT-4o 在抗幻觉方面比其他大型音频语言模型（LALMs）更为稳健。然而，它在音频时长预测和乐器分类等任务中遇到困难。同时，GPT-4o 的安全机制导致它拒绝执行像说话人识别、年龄分类、MOS 预测和音频深度伪造检测等任务。值得注意的是，当模型在不同的数据集上响应说话人验证任务时，其拒绝率表现出显著差异。这可能与伴随说明或输入音频的质量变化有关，暗示其内置保护措施的高度敏感性。最后，我们意识到模型性能因评估协议而异。本报告仅作为对当前 LALMs 状态的初步探索。 

---
# Mind What You Ask For: Emotional and Rational Faces of Persuasion by Large Language Models 

**Title (ZH)**: 谨言慎语：大型语言模型在说服中的情感与理性面相 

**Authors**: Wiktoria Mieleszczenko-Kowszewicz, Beata Bajcar, Jolanta Babiak, Berenika Dyczek, Jakub Świstak, Przemysław Biecek  

**Link**: [PDF](https://arxiv.org/pdf/2502.09687)  

**Abstract**: Be careful what you ask for, you just might get it. This saying fits with the way large language models (LLMs) are trained, which, instead of being rewarded for correctness, are increasingly rewarded for pleasing the recipient. So, they are increasingly effective at persuading us that their answers are valuable. But what tricks do they use in this persuasion? In this study, we examine what are the psycholinguistic features of the responses used by twelve different language models. By grouping response content according to rational or emotional prompts and exploring social influence principles employed by LLMs, we ask whether and how we can mitigate the risks of LLM-driven mass misinformation. We position this study within the broader discourse on human-centred AI, emphasizing the need for interdisciplinary approaches to mitigate cognitive and societal risks posed by persuasive AI responses. 

**Abstract (ZH)**: 如您所愿，但请务必小心。这句谚语同样适用于大型语言模型（LLMs）的训练方式，它们不再因正确性而获得奖励，而是越来越因取悦接收者而获得奖励。因此，它们在说服我们相信其答案的价值方面变得越来越有效。但它们在这一说服过程中使用了哪些技巧？在本研究中，我们分析了十二种不同语言模型的回应中所使用的心理语言学特征。通过根据理性和情绪提示对回应内容进行分组，并考察LLMs所采用的社会影响原则，我们探讨是否以及如何减轻由LLM驱动的大规模误导信息的风险。我们将本研究放置在更广泛的人本中心AI讨论中，强调需要采用跨学科的方法来缓解具有说服力的AI回应所带来的心智和社会风险。 

---
# k-LLMmeans: Summaries as Centroids for Interpretable and Scalable LLM-Based Text Clustering 

**Title (ZH)**: k-LLMmeans：基于摘要的可解释和可扩展的大语言模型文本聚类方法 

**Authors**: Jairo Diaz-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2502.09667)  

**Abstract**: We introduce k-LLMmeans, a novel modification of the k-means clustering algorithm that utilizes LLMs to generate textual summaries as cluster centroids, thereby capturing contextual and semantic nuances often lost when relying on purely numerical means of document embeddings. This modification preserves the properties of k-means while offering greater interpretability: the cluster centroid is represented by an LLM-generated summary, whose embedding guides cluster assignments. We also propose a mini-batch variant, enabling efficient online clustering for streaming text data and providing real-time interpretability of evolving cluster centroids. Through extensive simulations, we show that our methods outperform vanilla k-means on multiple metrics while incurring only modest LLM usage that does not scale with dataset size. Finally, We present a case study showcasing the interpretability of evolving cluster centroids in sequential text streams. As part of our evaluation, we compile a new dataset from StackExchange, offering a benchmark for text-stream clustering. 

**Abstract (ZH)**: 我们将介绍 k-LLMmeans，这是一种对 k-means 聚类算法的新型改进，它利用大型语言模型（LLMs）生成文本摘要作为聚类中心，从而捕捉文档嵌入方法中通常丢失的上下文和语义细微差别。这一改进保留了 k-means 的属性，同时提供了更高的可解释性：聚类中心由 LLM 生成的摘要表示，其嵌入指导聚类分配。我们还提出了一种基于小批量的变体，使其能够高效地对流式文本数据进行在线聚类，并提供实时聚类中心进化的可解释性。通过广泛的模拟，我们展示了我们的方法在多个指标上优于传统的 k-means，且所使用的 LLM 资源仅适度增加，不会随数据集大小成比例增长。最后，我们通过一个案例研究展示了流式文本序列中聚类中心演化结果的可解释性。作为评估的一部分，我们从 StackExchange 编制了一个新数据集，为文本流聚类提供了一个基准。 

---
# Cancer Vaccine Adjuvant Name Recognition from Biomedical Literature using Large Language Models 

**Title (ZH)**: 使用大型语言模型从生物医学文献中识别癌症疫苗辅助剂名称 

**Authors**: Hasin Rehana, Jie Zheng, Leo Yeh, Benu Bansal, Nur Bengisu Çam, Christianah Jemiyo, Brett McGregor, Arzucan Özgür, Yongqun He, Junguk Hur  

**Link**: [PDF](https://arxiv.org/pdf/2502.09659)  

**Abstract**: Motivation: An adjuvant is a chemical incorporated into vaccines that enhances their efficacy by improving the immune response. Identifying adjuvant names from cancer vaccine studies is essential for furthering research and enhancing immunotherapies. However, the manual curation from the constantly expanding biomedical literature poses significant challenges. This study explores the automated recognition of vaccine adjuvant names using Large Language Models (LLMs), specifically Generative Pretrained Transformers (GPT) and Large Language Model Meta AI (Llama). Methods: We utilized two datasets: 97 clinical trial records from AdjuvareDB and 290 abstracts annotated with the Vaccine Adjuvant Compendium (VAC). GPT-4o and Llama 3.2 were employed in zero-shot and few-shot learning paradigms with up to four examples per prompt. Prompts explicitly targeted adjuvant names, testing the impact of contextual information such as substances or interventions. Outputs underwent automated and manual validation for accuracy and consistency. Results: GPT-4o attained 100% Precision across all situations while exhibiting notable improve in Recall and F1-scores, particularly with incorporating interventions. On the VAC dataset, GPT-4o achieved a maximum F1-score of 77.32% with interventions, surpassing Llama-3.2-3B by approximately 2%. On the AdjuvareDB dataset, GPT-4o reached an F1-score of 81.67% for three-shot prompting with interventions, surpassing Llama-3.2-3 B's maximum F1-score of 65.62%. Conclusion: Our findings demonstrate that LLMs excel at identifying adjuvant names, including rare variations of naming representation. This study emphasizes the capability of LLMs to enhance cancer vaccine development by efficiently extracting insights. Future work aims to broaden the framework to encompass various biomedical literature and enhance model generalizability across various vaccines and adjuvants. 

**Abstract (ZH)**: 动机：佐剂是添加到疫苗中的一种化学物质，它可以增强疫苗的效果，通过改善免疫反应。从不断扩展的生物医学文献中识别癌症疫苗研究中的佐剂名称对进一步的研究和增强免疫疗法具有重要意义。然而，手动从这些文献中进行筛选存在显著的挑战。本研究探索了使用大规模语言模型（LLMs）自动识别疫苗佐剂名称的方法，具体使用了生成预训练变换器（GPT）和大规模语言模型Meta AI（Llama）。

方法：我们利用了两个数据集：来自AdjuvareDB的97项临床试验记录和来自Vaccine Adjuvant Compendium (VAC)的290篇带有注释的摘要。我们使用了零样本和少量样本学习范式，其中GPT-4o和Llama 3.2每提示最多使用四个示例。提示特别针对佐剂名称，测试了物质或干预措施等上下文信息的影响。输出结果进行了自动和手动验证以确保准确性和一致性。

结果：GPT-4o在所有情况下都达到了100%的精确度，同时在召回率和F1分数方面表现出显著改善，特别是加入了干预措施的情况下。在VAC数据集中，GPT-4o在加入干预措施时的最大F1分数为77.32%，超过了Llama-3.2-3B约2%。在AdjuvareDB数据集中，对于使用干预措施的三种提示，GPT-4o达到了81.67%的F1分数，超过了Llama-3.2-3B的最大F1分数65.62%。

结论：我们的研究结果表明，LLMs在识别佐剂名称方面表现出色，包括识别罕见的命名变体。本研究强调了LLMs在通过高效提取见解来增强癌症疫苗开发方面的潜力。未来的研究方向将致力于扩展框架以涵盖各种生物医学文献，并增强模型在不同疫苗和佐剂方面的普适性。 

---
