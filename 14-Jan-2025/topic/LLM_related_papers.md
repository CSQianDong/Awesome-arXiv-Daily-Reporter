# Lifelong Learning of Large Language Model based Agents: A Roadmap 

**Title (ZH)**: 基于代理的大型语言模型终身学习：一条路线图 

**Authors**: Junhao Zheng, Chengming Shi, Xidi Cai, Qiuke Li, Duzhen Zhang, Chenxing Li, Dong Yu, Qianli Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.07278)  

**Abstract**: Lifelong learning, also known as continual or incremental learning, is a crucial component for advancing Artificial General Intelligence (AGI) by enabling systems to continuously adapt in dynamic environments. While large language models (LLMs) have demonstrated impressive capabilities in natural language processing, existing LLM agents are typically designed for static systems and lack the ability to adapt over time in response to new challenges. This survey is the first to systematically summarize the potential techniques for incorporating lifelong learning into LLM-based agents. We categorize the core components of these agents into three modules: the perception module for multimodal input integration, the memory module for storing and retrieving evolving knowledge, and the action module for grounded interactions with the dynamic environment. We highlight how these pillars collectively enable continuous adaptation, mitigate catastrophic forgetting, and improve long-term performance. This survey provides a roadmap for researchers and practitioners working to develop lifelong learning capabilities in LLM agents, offering insights into emerging trends, evaluation metrics, and application scenarios. Relevant literature and resources are available at \href{this url}{this https URL}. 

**Abstract (ZH)**: 终身学习，也称为持续学习或增量学习，是推进通用人工智能（AGI）的关键组件，它使系统能够适应动态环境。虽然大型语言模型（LLMs）在自然语言处理方面表现出色，但现有的LLM代理通常设计为静态系统，缺乏根据新挑战不断调整的能力。本文是首次系统总结将终身学习引入LLM基代理潜在技术的综述。我们将这些代理的核心组件分为三个模块：感知模块用于多模态输入集成、记忆模块用于存储和检索不断演变的知识，以及动作模块用于与动态环境进行具身交互。我们强调了这些支柱如何共同实现持续适应、防止灾难性遗忘以及提升长期性能。本文为研究人员和实践者开发LLM代理的终身学习能力提供了指导，提供了关于新兴趋势、评估指标和应用场景的见解。相关文献和资源详见 [该网址](this https URL)。 

---
# Lessons From Red Teaming 100 Generative AI Products 

**Title (ZH)**: 《红队测试100个生成型AI产品所得教训》 

**Authors**: Blake Bullwinkel, Amanda Minnich, Shiven Chawla, Gary Lopez, Martin Pouliot, Whitney Maxwell, Joris de Gruyter, Katherine Pratt, Saphir Qi, Nina Chikanov, Roman Lutz, Raja Sekhar Rao Dheekonda, Bolor-Erdene Jagdagdorj, Eugenia Kim, Justin Song, Keegan Hines, Daniel Jones, Giorgio Severi, Richard Lundeen, Sam Vaughan, Victoria Westerhoff, Pete Bryan, Ram Shankar Siva Kumar, Yonatan Zunger, Chang Kawaguchi, Mark Russinovich  

**Link**: [PDF](https://arxiv.org/pdf/2501.07238)  

**Abstract**: In recent years, AI red teaming has emerged as a practice for probing the safety and security of generative AI systems. Due to the nascency of the field, there are many open questions about how red teaming operations should be conducted. Based on our experience red teaming over 100 generative AI products at Microsoft, we present our internal threat model ontology and eight main lessons we have learned:
1. Understand what the system can do and where it is applied
2. You don't have to compute gradients to break an AI system
3. AI red teaming is not safety benchmarking
4. Automation can help cover more of the risk landscape
5. The human element of AI red teaming is crucial
6. Responsible AI harms are pervasive but difficult to measure
7. LLMs amplify existing security risks and introduce new ones
8. The work of securing AI systems will never be complete
By sharing these insights alongside case studies from our operations, we offer practical recommendations aimed at aligning red teaming efforts with real world risks. We also highlight aspects of AI red teaming that we believe are often misunderstood and discuss open questions for the field to consider. 

**Abstract (ZH)**: 近年来，AI红队已经成为评估生成式AI系统安全性和安全性的一种实践方法。由于该领域尚处于起步阶段，关于如何进行红队操作仍然存在许多开放性问题。基于我们对微软100多种生成式AI产品的红队经验，我们提出了我们的内部威胁模型本体以及从这些经验中得出的八条主要教训：

1. 了解系统的能力及其应用领域
2. 并非只有通过计算梯度才能破坏AI系统
3. AI红队与安全基准测试不同
4. 自动化可以帮助覆盖更多的风险领域
5. 人类元素在AI红队中至关重要
6. 道德AI的危害普遍存在但难以衡量
7. 大型语言模型放大了现有的安全风险并引入了新的风险
8. 保障AI系统安全的工作永远不可能完成

通过结合我们运营中的案例研究分享这些见解，我们提供了一些建议，旨在使红队操作与现实世界的风险相一致。我们还强调了我们认为常常被误解的AI红队方面的内容，并讨论了该领域需要考虑的开放性问题。 

---
# How GPT learns layer by layer 

**Title (ZH)**: 如何逐层学习：GPT的层层学习机制 

**Authors**: Jason Du, Kelly Hong, Alishba Imran, Erfan Jahanparast, Mehdi Khfifi, Kaichun Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2501.07108)  

**Abstract**: Large Language Models (LLMs) excel at tasks like language processing, strategy games, and reasoning but struggle to build generalizable internal representations essential for adaptive decision-making in agents. For agents to effectively navigate complex environments, they must construct reliable world models. While LLMs perform well on specific benchmarks, they often fail to generalize, leading to brittle representations that limit their real-world effectiveness. Understanding how LLMs build internal world models is key to developing agents capable of consistent, adaptive behavior across tasks. We analyze OthelloGPT, a GPT-based model trained on Othello gameplay, as a controlled testbed for studying representation learning. Despite being trained solely on next-token prediction with random valid moves, OthelloGPT shows meaningful layer-wise progression in understanding board state and gameplay. Early layers capture static attributes like board edges, while deeper layers reflect dynamic tile changes. To interpret these representations, we compare Sparse Autoencoders (SAEs) with linear probes, finding that SAEs offer more robust, disentangled insights into compositional features, whereas linear probes mainly detect features useful for classification. We use SAEs to decode features related to tile color and tile stability, a previously unexamined feature that reflects complex gameplay concepts like board control and long-term planning. We study the progression of linear probe accuracy and tile color using both SAE's and linear probes to compare their effectiveness at capturing what the model is learning. Although we begin with a smaller language model, OthelloGPT, this study establishes a framework for understanding the internal representations learned by GPT models, transformers, and LLMs more broadly. Our code is publicly available: this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在语言处理、策略游戏和推理方面表现出色，但在构建适应性决策所需的关键内部表示方面却遇到困难。为了使智能体能够有效地导航复杂环境，它们必须构建可靠的世界模型。尽管LLMs在特定基准测试中表现良好，但它们常常难以泛化，导致脆弱的表示，限制了其在现实世界中的有效性。理解LLMs如何构建内部世界模型是开发能够在各种任务中表现一致且具有适应性的智能体的关键。我们分析了基于GPT的OthelloGPT模型，该模型在国际象棋游戏数据上进行训练，作为研究表示学习的控制实验床。尽管仅基于下一步骤预测并使用随机有效移动进行训练，OthelloGPT在理解棋盘状态和游戏方面的逐层进展具有重要意义。早期层捕捉静态属性，如棋盘边缘，而深层层反映了动态瓷砖的变化。为了解读这些表示，我们对比了稀疏自编码器（SAEs）和线性探针，发现SAEs提供了关于组合特征的更加稳健且解耦的理解，而线性探针主要检测有助于分类的特征。我们使用SAEs解码与瓷砖颜色和稳定性相关的特征，这是之前未曾研究过的特征，反映了如棋盘控制和长远规划等复杂的游戏概念。我们研究了使用SAEs和线性探针的线性探针精度和瓷砖颜色的进展，以比较它们在捕捉模型所学内容方面的有效性。尽管我们从较小的语言模型OthelloGPT开始，但本研究建立了一种框架，用以理解GPT模型、变换器及更广泛的LLMs所学到的内部表示。我们的代码已公开：[此处插入URL]。 

---
# Value Compass Leaderboard: A Platform for Fundamental and Validated Evaluation of LLMs Values 

**Title (ZH)**: 价值定向领导者排行榜：一个用于评估大型语言模型价值观的基础且验证过的平台 

**Authors**: Jing Yao, Xiaoyuan Yi, Shitong Duan, Jindong Wang, Yuzhuo Bai, Muhua Huang, Peng Zhang, Tun Lu, Zhicheng Dou, Maosong Sun, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2501.07071)  

**Abstract**: As Large Language Models (LLMs) achieve remarkable breakthroughs, aligning their values with humans has become imperative for their responsible development and customized applications. However, there still lack evaluations of LLMs values that fulfill three desirable goals. (1) Value Clarification: We expect to clarify the underlying values of LLMs precisely and comprehensively, while current evaluations focus narrowly on safety risks such as bias and toxicity. (2) Evaluation Validity: Existing static, open-source benchmarks are prone to data contamination and quickly become obsolete as LLMs evolve. Additionally, these discriminative evaluations uncover LLMs' knowledge about values, rather than valid assessments of LLMs' behavioral conformity to values. (3) Value Pluralism: The pluralistic nature of human values across individuals and cultures is largely ignored in measuring LLMs value alignment. To address these challenges, we presents the Value Compass Leaderboard, with three correspondingly designed modules. It (i) grounds the evaluation on motivationally distinct \textit{basic values to clarify LLMs' underlying values from a holistic view; (ii) applies a \textit{generative evolving evaluation framework with adaptive test items for evolving LLMs and direct value recognition from behaviors in realistic scenarios; (iii) propose a metric that quantifies LLMs alignment with a specific value as a weighted sum over multiple dimensions, with weights determined by pluralistic values. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）取得突出进展，使其价值与人类价值观相匹配已成为负责任开发和定制应用的必要条件。然而，当前缺乏满足三个理想目标的LLMs价值观评估。具体来说：

1. **价值澄清**：我们期望精确且全面地阐明LLMs的内在价值观，而现有的评估主要集中在偏见和毒性等安全风险方面。

2. **评估有效性**：现有的静态开源基准易受数据污染的影响，并且随着LLMs的发展而迅速过时。此外，这些区分性的评估揭示了LLMs对于价值观的知识，而不是对其行为符合价值观的有效评估。

3. **价值多元性**：衡量LLMs的价值对齐时，人类价值观在个体之间和文化之间的多元性被很大程度上忽视了。

为解决这些挑战，我们提出了价值指南针排行榜，并设计了三个相应的模块。它（i）基于动机上不同的基本价值观，从全局视角阐明LLMs的内在价值观；(ii)应用一个生成性的不断发展评估框架，该框架具有自适应测试项目，适用于演化的LLMs，并直接从现实场景中的行为中识别价值观；(iii)提出一个量化LLMs与特定价值观对齐程度的指标，该指标是多维度权重的加权总和，权重由多元价值观确定。 

---
# PoAct: Policy and Action Dual-Control Agent for Generalized Applications 

**Title (ZH)**: PoAct：通用应用的策略与动作双控智能体 

**Authors**: Guozhi Yuan, Youfeng Liu, Jingli Yang, Wei Jia, Kai Lin, Yansong Gao, Shan He, Zilin Ding, Haitao Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.07054)  

**Abstract**: Based on their superior comprehension and reasoning capabilities, Large Language Model (LLM) driven agent frameworks have achieved significant success in numerous complex reasoning tasks. ReAct-like agents can solve various intricate problems step-by-step through progressive planning and tool calls, iteratively optimizing new steps based on environmental feedback. However, as the planning capabilities of LLMs improve, the actions invoked by tool calls in ReAct-like frameworks often misalign with complex planning and challenging data organization. Code Action addresses these issues while also introducing the challenges of a more complex action space and more difficult action organization. To leverage Code Action and tackle the challenges of its complexity, this paper proposes Policy and Action Dual-Control Agent (PoAct) for generalized applications. The aim is to achieve higher-quality code actions and more accurate reasoning paths by dynamically switching reasoning policies and modifying the action space. Experimental results on the Agent Benchmark for both legal and generic scenarios demonstrate the superior reasoning capabilities and reduced token consumption of our approach in complex tasks. On the LegalAgentBench, our method shows a 20 percent improvement over the baseline while requiring fewer tokens. We conducted experiments and analyses on the GPT-4o and GLM-4 series models, demonstrating the significant potential and scalability of our approach to solve complex problems. 

**Abstract (ZH)**: 基于其卓越的理解能力和推理能力，由大语言模型（LLM）驱动的智能体框架在众多复杂的推理任务中取得了显著的成功。类似于ReAct的智能体可以通过逐步的规划和工具调用解决各种复杂问题，并根据环境反馈迭代优化新的步骤。然而，随着LLM规划能力的提高，ReAct框架中由工具调用引发的动作往往与复杂的规划和具有挑战性的数据组织不相匹配。代码执行（Code Action）解决这些问题的同时，也带来了动作空间更加复杂以及动作组织更加困难的挑战。为了利用代码执行并应对这些复杂性挑战，本文提出了一种策略与行动双控制智能体（Policy and Action Dual-Control Agent, PoAct），以实现泛化的应用。目标是通过动态切换推理策略并调整动作空间，实现更高的代码执行质量及更准确的推理路径。在代理基准测试中，无论是针对法律场景还是通用场景，我们的方法都展示了更强的推理能力和更少的标记消耗。在LegalAgentBench上，我们的方法相比基准方法提高了20%，且消耗的标记更少。我们在GPT-4o和GLM-4系列模型上进行了实验和分析，验证了我们方法的巨大潜力和可扩展性，可用于解决复杂问题。 

---
# A Proposed Large Language Model-Based Smart Search for Archive System 

**Title (ZH)**: 一种基于大型语言模型的智能存档系统搜索方法 

**Authors**: Ha Dung Nguyen, Thi-Hoang Anh Nguyen, Thanh Binh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2501.07024)  

**Abstract**: This study presents a novel framework for smart search in digital archival systems, leveraging the capabilities of Large Language Models (LLMs) to enhance information retrieval. By employing a Retrieval-Augmented Generation (RAG) approach, the framework enables the processing of natural language queries and transforming non-textual data into meaningful textual representations. The system integrates advanced metadata generation techniques, a hybrid retrieval mechanism, a router query engine, and robust response synthesis, the results proved search precision and relevance. We present the architecture and implementation of the system and evaluate its performance in four experiments concerning LLM efficiency, hybrid retrieval optimizations, multilingual query handling, and the impacts of individual components. Obtained results show significant improvements over conventional approaches and have demonstrated the potential of AI-powered systems to transform modern archival practices. 

**Abstract (ZH)**: 本研究提出了一种用于数字档案系统智能搜索的新框架，充分利用大型语言模型（LLMs）的能力来增强信息检索。通过采用检索增强生成（RAG）方法，该框架能够处理自然语言查询，并将非文本数据转换为有意义的文本表示。该系统集成了高级元数据生成技术、混合检索机制、路由器查询引擎以及强大的响应合成方法，结果显示了搜索精度和相关性的提升。我们展示了该系统的架构和实现，并在四个实验中评估了其性能，涉及大型语言模型效率、混合检索优化、多语言查询处理以及各组件的影响。所获得的结果表明，该方法显著优于传统方法，展示了基于人工智能的系统在转变现代档案实践方面的巨大潜力。 

---
# Enhancing Patient-Centric Communication: Leveraging LLMs to Simulate Patient Perspectives 

**Title (ZH)**: 提升以患者为中心的沟通：利用大语言模型模拟患者视角 

**Authors**: Xinyao Ma, Rui Zhu, Zihao Wang, Jingwei Xiong, Qingyu Chen, Haixu Tang, L. Jean Camp, Lucila Ohno-Machado  

**Link**: [PDF](https://arxiv.org/pdf/2501.06964)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in role-playing scenarios, particularly in simulating domain-specific experts using tailored prompts. This ability enables LLMs to adopt the persona of individuals with specific backgrounds, offering a cost-effective and efficient alternative to traditional, resource-intensive user studies. By mimicking human behavior, LLMs can anticipate responses based on concrete demographic or professional profiles. In this paper, we evaluate the effectiveness of LLMs in simulating individuals with diverse backgrounds and analyze the consistency of these simulated behaviors compared to real-world outcomes. In particular, we explore the potential of LLMs to interpret and respond to discharge summaries provided to patients leaving the Intensive Care Unit (ICU). We evaluate and compare with human responses the comprehensibility of discharge summaries among individuals with varying educational backgrounds, using this analysis to assess the strengths and limitations of LLM-driven simulations. Notably, when LLMs are primed with educational background information, they deliver accurate and actionable medical guidance 88% of the time. However, when other information is provided, performance significantly drops, falling below random chance levels. This preliminary study shows the potential benefits and pitfalls of automatically generating patient-specific health information from diverse populations. While LLMs show promise in simulating health personas, our results highlight critical gaps that must be addressed before they can be reliably used in clinical settings. Our findings suggest that a straightforward query-response model could outperform a more tailored approach in delivering health information. This is a crucial first step in understanding how LLMs can be optimized for personalized health communication while maintaining accuracy. 

**Abstract (ZH)**: 大型语言模型（LLMs）在角色扮演场景中展现了令人印象深刻的能力，特别是在使用特定提示模拟特定领域专家方面。这种能力使LLMs能够扮演具有特定背景的个体角色，提供一种成本效益高且高效的替代传统资源密集型用户研究的方法。通过模仿人类行为，LLMs可以根据具体的年龄、性别或职业特征预测响应。在本文中，我们评估了LLMs在模拟具有不同背景的个体方面的有效性，并分析了这些模拟行为与现实结果的一致性。特别地，我们探索了LLMs在解释和响应患者从重症监护室（ICU）出院时所收到的出院总结方面的潜力。我们使用不同教育背景的个体来评估出院总结的可理解性，并根据这些分析评估由LLM驱动的模拟的优缺点。值得注意的是，当LLMs预加载教育背景信息时，它们有88%的时间提供准确且实用的医疗指导。然而，当提供其他信息时，性能显著下降，甚至低于随机猜测的水平。这项初步研究展示了自动从多元化人群中生成患者特定健康信息的潜在利弊。尽管LLMs在模拟健康人格方面表现出前景，但我们的结果突显了临床应用前必须解决的关键差距。我们的研究结果表明，一个简单的查询-响应模型可能在传递健康信息方面优于更加定制化的模型。这是理解如何优化LLMs以实现个性化的健康沟通，同时保持准确性的一个关键步骤。 

---
# Risk-Averse Finetuning of Large Language Models 

**Title (ZH)**: 大型语言模型的风险规避微调 

**Authors**: Sapana Chaudhary, Ujwal Dinesha, Dileep Kalathil, Srinivas Shakkottai  

**Link**: [PDF](https://arxiv.org/pdf/2501.06911)  

**Abstract**: We consider the challenge of mitigating the generation of negative or toxic content by the Large Language Models (LLMs) in response to certain prompts. We propose integrating risk-averse principles into LLM fine-tuning to minimize the occurrence of harmful outputs, particularly rare but significant events. By optimizing the risk measure of Conditional Value at Risk (CVaR), our methodology trains LLMs to exhibit superior performance in avoiding toxic outputs while maintaining effectiveness in generative tasks. Empirical evaluations on sentiment modification and toxicity mitigation tasks demonstrate the efficacy of risk-averse reinforcement learning with human feedback (RLHF) in promoting a safer and more constructive online discourse environment. 

**Abstract (ZH)**: 我们探讨了通过大型语言模型（LLMs）对特定提示的响应，减少生成负面或有害内容的挑战。我们提出将规避风险的原则融入LLM的微调中，以最小化有害输出（尤其是罕见但重要的事件）的发生率。通过优化条件值风险（CVaR）的风险度量，我们的方法训练LLM在避免有害输出的同时，在生成任务中保持高效。在情感修饰和有害内容减轻任务上的实证评估表明，风险规避的强化学习（结合人类反馈的RLHF）在促进更安全和建设性的在线对话环境中具有有效性。 

---
# An efficient approach to represent enterprise web application structure using Large Language Model in the service of Intelligent Quality Engineering 

**Title (ZH)**: 使用大型语言模型为企业_web应用程序结构表示的一种高效方法——智能质量工程的服务应用 

**Authors**: Zaber Al Hassan Ayon, Gulam Husain, Roshankumar Bisoi, Waliur Rahman, Dr Tom Osborn  

**Link**: [PDF](https://arxiv.org/pdf/2501.06837)  

**Abstract**: This paper presents a novel approach to represent enterprise web application structures using Large Language Models (LLMs) to enable intelligent quality engineering at scale. We introduce a hierarchical representation methodology that optimizes the few-shot learning capabilities of LLMs while preserving the complex relationships and interactions within web applications. The approach encompasses five key phases: comprehensive DOM analysis, multi-page synthesis, test suite generation, execution, and result analysis. Our methodology addresses existing challenges around usage of Generative AI techniques in automated software testing by developing a structured format that enables LLMs to understand web application architecture through in-context learning. We evaluated our approach using two distinct web applications: an e-commerce platform (Swag Labs) and a healthcare application (MediBox) which is deployed within Atalgo engineering environment. The results demonstrate success rates of 90\% and 70\%, respectively, in achieving automated testing, with high relevance scores for test cases across multiple evaluation criteria. The findings suggest that our representation approach significantly enhances LLMs' ability to generate contextually relevant test cases and provide better quality assurance overall, while reducing the time and effort required for testing. 

**Abstract (ZH)**: 本文提出了一种新的方法，利用大规模语言模型（LLMs）来表示企业Web应用程序结构，从而实现大规模的智能质量工程。我们引入了一种层次化表示方法，优化了LLMs的少样本学习能力，同时保留了Web应用程序内部的复杂关系和交互。该方法包括五个关键阶段：全面的DOM分析、多页面合成、测试用例生成、执行和结果分析。我们的方法通过上下文学习开发了一种结构化格式，解决了在自动化软件测试中使用生成式AI技术的现有挑战，使LLMs能够理解Web应用程序架构。我们使用两个不同的Web应用程序评估了该方法：一个电子商务平台（Swag Labs）和一个部署在Atalgo工程环境中的医疗应用（MediBox）。结果表明，在自动化测试方面分别达到了90%和70%的成功率，且测试用例在多个评估标准下的相关性评分都很高。研究结果表明，我们的表示方法显著增强了LLMs生成上下文相关测试用例的能力，并整体提高了质量保证的水平，同时减少了测试所需的时间和努力。 

---
# LLMs Model Non-WEIRD Populations: Experiments with Synthetic Cultural Agents 

**Title (ZH)**: LLMs 模型化非 WEIRD 人群：合成文化代理的实验研究 

**Authors**: Augusto Gonzalez-Bonorino, Monica Capra, Emilio Pantoja  

**Link**: [PDF](https://arxiv.org/pdf/2501.06834)  

**Abstract**: Despite its importance, studying economic behavior across diverse, non-WEIRD (Western, Educated, Industrialized, Rich, and Democratic) populations presents significant challenges. We address this issue by introducing a novel methodology that uses Large Language Models (LLMs) to create synthetic cultural agents (SCAs) representing these populations. We subject these SCAs to classic behavioral experiments, including the dictator and ultimatum games. Our results demonstrate substantial cross-cultural variability in experimental behavior. Notably, for populations with available data, SCAs' behaviors qualitatively resemble those of real human subjects. For unstudied populations, our method can generate novel, testable hypotheses about economic behavior. By integrating AI into experimental economics, this approach offers an effective and ethical method to pilot experiments and refine protocols for hard-to-reach populations. Our study provides a new tool for cross-cultural economic studies and demonstrates how LLMs can help experimental behavioral research. 

**Abstract (ZH)**: 尽管其重要性不言而喻，研究跨多元、非WEIRD（西方、受过教育、工业化、富裕、民主）人口的经济行为存在显著挑战。为应对这一问题，我们引入了一种新的方法，该方法利用大型语言模型（LLMs）创建合成文化代理（SCAs），以代表这些人群。我们将这些SCAs置于经典的经济行为实验中，包括分配者游戏和 ultimatum 游戏。研究结果表明，这些实验中的行为在跨文化上存在显著差异。值得注意的是，对于已有数据的人群，SCAs 的行为在定性上与真实人类被试相似。对于未被研究的人群，我们的方法可以产生新的、可测试的关于经济行为的假设。通过将AI引入实验经济学，该方法提供了一种有效且伦理的方式，以试点实验并改进难以接触人群的研究方案。我们的研究提供了一种新的工具，用于跨文化经济研究，并展示了LLMs如何帮助实验行为研究。 

---
# Leveraging Taxonomy and LLMs for Improved Multimodal Hierarchical Classification 

**Title (ZH)**: 利用分类体系和大规模语言模型以改进多模态层次分类 

**Authors**: Shijing Chen, Mohamed Reda Bouadjenek, Shoaib Jameel, Usman Naseem, Basem Suleiman, Flora D. Salim, Hakim Hacid, Imran Razzak  

**Link**: [PDF](https://arxiv.org/pdf/2501.06827)  

**Abstract**: Multi-level Hierarchical Classification (MLHC) tackles the challenge of categorizing items within a complex, multi-layered class structure. However, traditional MLHC classifiers often rely on a backbone model with independent output layers, which tend to ignore the hierarchical relationships between classes. This oversight can lead to inconsistent predictions that violate the underlying taxonomy. Leveraging Large Language Models (LLMs), we propose a novel taxonomy-embedded transitional LLM-agnostic framework for multimodality classification. The cornerstone of this advancement is the ability of models to enforce consistency across hierarchical levels. Our evaluations on the MEP-3M dataset - a multi-modal e-commerce product dataset with various hierarchical levels - demonstrated a significant performance improvement compared to conventional LLM structures. 

**Abstract (ZH)**: 多层层次分类（MLHC）解决了在复杂多层类结构中对项目进行分类的挑战。然而，传统的MLHC分类器往往依赖于具有独立输出层的骨干模型，这些模型倾向于忽略类之间的层次关系。这种忽视可能导致违反底层分类体系的一致性预测。利用大型语言模型（LLMs），我们提出了一种新颖的嵌入分类学的过渡LLM无关框架，用于多模态分类。这一进展的核心在于模型能够确保在不同层次上的一致性。我们对MEP-3M数据集（这是一个包含多种层次结构的多模态电子商务产品数据集）进行的评估表明，与传统的LLM结构相比，该框架在性能上取得了显著的提升。 

---
# Eliza: A Web3 friendly AI Agent Operating System 

**Title (ZH)**: 艾莉莎：一个面向Web3的AI代理操作系统 

**Authors**: Shaw Walters, Sam Gao, Shakker Nerd, Feng Da, Warren Williams, Ting-Chien Meng, Hunter Han, Frank He, Allen Zhang, Ming Wu, Timothy Shen, Maxwell Hu, Jerry Yan  

**Link**: [PDF](https://arxiv.org/pdf/2501.06781)  

**Abstract**: AI Agent, powered by large language models (LLMs) as its cognitive core, is an intelligent agentic system capable of autonomously controlling and determining the execution paths under user's instructions. With the burst of capabilities of LLMs and various plugins, such as RAG, text-to-image/video/3D, etc., the potential of AI Agents has been vastly expanded, with their capabilities growing stronger by the day. However, at the intersection between AI and web3, there is currently no ideal agentic framework that can seamlessly integrate web3 applications into AI agent functionalities. In this paper, we propose Eliza, the first open-source web3-friendly Agentic framework that makes the deployment of web3 applications effortless. We emphasize that every aspect of Eliza is a regular Typescript program under the full control of its user, and it seamlessly integrates with web3 (i.e., reading and writing blockchain data, interacting with smart contracts, etc.). Furthermore, we show how stable performance is achieved through the pragmatic implementation of the key components of Eliza's runtime. Our code is publicly available at this https URL. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的认知核心，AI 剂机是一种能够在用户指令下自主控制和决定执行路径的智能代理系统。随着大型语言模型能力的爆发式增长以及各种插件（如RAG、文本转图像/视频/3D等）的应用，AI 剂机的潜力得到了极大扩展，其功能日渐增强。然而，在AI与Web3的交汇点上，目前尚缺乏一个能够无缝集成Web3应用到AI剂机功能的理想框架。本文中，我们提出Eliza——首个开源的Web3友好型代理框架，使得部署Web3应用变得轻而易举。我们强调，Eliza 的每一部分都是用户完全控制下的常规TypeScript程序，并且能够无缝集成Web3（如读取和写入区块链数据、与智能合约互动等）。此外，我们展示了通过Pragmatic实现实现Eliza运行时关键组件的稳定性能。我们的代码已公开，可在以下链接访问：[此 https URL](此 https URL)。 

---
# Fine-tuning ChatGPT for Automatic Scoring of Written Scientific Explanations in Chinese 

**Title (ZH)**: 将ChatGPT微调以自动评分中文书面科学解释 

**Authors**: Jie Yang, Ehsan Latif, Yuze He, Xiaoming Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2501.06704)  

**Abstract**: The development of explanations for scientific phenomena is essential in science assessment, but scoring student-written explanations remains challenging and resource-intensive. Large language models (LLMs) have shown promise in addressing this issue, particularly in alphabetic languages like English. However, their applicability to logographic languages is less explored. This study investigates the potential of fine-tuning ChatGPT, a leading LLM, to automatically score scientific explanations written in Chinese. Student responses to seven scientific explanation tasks were collected and automatically scored, with scoring accuracy examined in relation to reasoning complexity using the Kendall correlation. A qualitative analysis explored how linguistic features influenced scoring accuracy. The results show that domain-specific adaptation enables ChatGPT to score Chinese scientific explanations with accuracy. However, scoring accuracy correlates with reasoning complexity: a negative correlation for lower-level responses and a positive one for higher-level responses. The model overrates complex reasoning in low-level responses with intricate sentence structures and underrates high-level responses using concise causal reasoning. These correlations stem from linguistic features--simplicity and clarity enhance accuracy for lower-level responses, while comprehensiveness improves accuracy for higher-level ones. Simpler, shorter responses tend to score more accurately at lower levels, whereas longer, information-rich responses yield better accuracy at higher levels. These findings demonstrate the effectiveness of LLMs in automatic scoring within a Chinese context and emphasize the importance of linguistic features and reasoning complexity in fine-tuning scoring models for educational assessments. 

**Abstract (ZH)**: 科学现象解释的发展对于科学评估至关重要，但对学生撰写的解释进行评分仍然是一项具有挑战性和资源密集性的任务。大型语言模型（LLMs）在此方面展现出了潜力，尤其是在像英语这样的字母语言方面。然而，它们在象形文字语言如中文方面的应用尚缺乏探讨。本研究旨在探讨微调领先的大规模语言模型ChatGPT自动评分中文科学解释的潜力。收集了学生对七项科学解释任务的响应，并自动进行了评分，评分准确度与推理复杂性之间的关系通过Kendall相关性进行了考察。定性分析进一步探讨了语言特征如何影响评分准确度。结果显示，领域特定的适应性使ChatGPT能够准确评分中文科学解释。然而，评分准确度与推理复杂性之间存在相关性：低层级响应呈现负相关，而高层级响应则呈现正相关。该模型在低层级复杂推理的复杂句结构中高估了推理，在高层级使用简洁因果推理的响应中低估了评分准确度。这些相关性源于语言特征——简洁性和清晰性增强了低层级响应的准确度，而综合性和全面性则改进了高层级响应的准确度。简短的回答通常在低层级更准确，而较长的信息丰富回答在高层级更准确。这些发现表明，大规模语言模型在中文背景下自动评分的有效性，并强调了语言特征和推理复杂性在微调评分模型方面的重要性，以提高教育评估的准确性。 

---
# DVM: Towards Controllable LLM Agents in Social Deduction Games 

**Title (ZH)**: DVM：面向社交推理游戏中的可控大型语言模型代理 

**Authors**: Zheng Zhang, Yihuai Lan, Yangsen Chen, Lei Wang, Xiang Wang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06695)  

**Abstract**: Large Language Models (LLMs) have advanced the capability of game agents in social deduction games (SDGs). These games rely heavily on conversation-driven interactions and require agents to infer, make decisions, and express based on such information. While this progress leads to more sophisticated and strategic non-player characters (NPCs) in SDGs, there exists a need to control the proficiency of these agents. This control not only ensures that NPCs can adapt to varying difficulty levels during gameplay, but also provides insights into the safety and fairness of LLM agents. In this paper, we present DVM, a novel framework for developing controllable LLM agents for SDGs, and demonstrate its implementation on one of the most popular SDGs, Werewolf. DVM comprises three main components: Predictor, Decider, and Discussor. By integrating reinforcement learning with a win rate-constrained decision chain reward mechanism, we enable agents to dynamically adjust their gameplay proficiency to achieve specified win rates. Experiments show that DVM not only outperforms existing methods in the Werewolf game, but also successfully modulates its performance levels to meet predefined win rate targets. These results pave the way for LLM agents' adaptive and balanced gameplay in SDGs, opening new avenues for research in controllable game agents. 

**Abstract (ZH)**: 大型语言模型（LLMs）在提升桌游推理游戏中人工智能角色（AI角色）的能力方面取得了显著进展。这类游戏依赖于对话驱动的互动，要求AI角色根据相关信息进行推理、决策和表达。虽然这些进步使得桌游推理游戏中的非玩家角色（NPC）变得更加复杂和策略性，但需要控制这些角色的专业水平。这种控制不仅确保AI角色在不同难度等级的游戏中能够适应变化，还为Lyling模型（LLM）代理的安全性和公平性提供了见解。本文提出了一种名为DVM的新颖框架，用于开发可控制的大型语言模型代理，该框架已在最受欢迎的桌游推理游戏之一“狼人杀”中进行了实现。DVM的主要组成部分包括预测器、决策器和讨论者。通过结合强化学习和胜率约束决策链奖励机制，我们使代理能够动态调节其游戏水平以实现指定的胜率。实验表明，DVM不仅在“狼人杀”游戏中优于现有方法，而且成功地调节其性能水平以达到预定义的胜率目标。这些结果为大型语言模型代理在桌游推理游戏中的适应性和平衡游戏提供了可能，为可控游戏代理的研究开辟了新的途径。 

---
# Guided Code Generation with LLMs: A Multi-Agent Framework for Complex Code Tasks 

**Title (ZH)**: 基于LLM的引导式代码生成：复杂代码任务的多代理框架 

**Authors**: Amr Almorsi, Mohanned Ahmed, Walid Gomaa  

**Link**: [PDF](https://arxiv.org/pdf/2501.06625)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in code generation tasks, yet they face significant limitations in handling complex, long-context programming challenges and demonstrating complex compositional reasoning abilities. This paper introduces a novel agentic framework for ``guided code generation'' that tries to address these limitations through a deliberately structured, fine-grained approach to code generation tasks. Our framework leverages LLMs' strengths as fuzzy searchers and approximate information retrievers while mitigating their weaknesses in long sequential reasoning and long-context understanding. Empirical evaluation using OpenAI's HumanEval benchmark with Meta's Llama 3.1 8B model (int4 precision) demonstrates a 23.79\% improvement in solution accuracy compared to direct one-shot generation. Our results indicate that structured, guided approaches to code generation can significantly enhance the practical utility of LLMs in software development while overcoming their inherent limitations in compositional reasoning and context handling. 

**Abstract (ZH)**: 大型语言模型（LLMs）在代码生成任务中展现了显著的能力，但在处理复杂的、长上下文的编程挑战以及展示复杂的组合推理能力方面存在重大局限。本文介绍了一种新的自主框架，旨在通过精细结构化的方法解决这些局限性，以实现“引导式代码生成”。我们的框架利用了LLMs作为模糊搜索者和近似信息检索器的优势，同时减轻了它们在长顺序推理和长上下文理解方面的弱点。使用OpenAI的HumanEval基准测试和Meta的Llama 3.1 8B模型（int4精度）进行的实证评估表明，与直接一对一生成相比，该框架在解决方案准确性上提高了23.79%。我们的结果表明，结构化和引导式的代码生成方法可以显著提高LLMs在软件开发中的实用价值，同时克服它们在组合推理和上下文处理方面的固有局限性。 

---
# Sustainable and Intelligent Public Facility Failure Management System Based on Large Language Models 

**Title (ZH)**: 基于大型语言模型的可持续与智能公共设施故障管理系统 

**Authors**: Siguo Bi, Jilong Zhang, Wei Ni  

**Link**: [PDF](https://arxiv.org/pdf/2501.06231)  

**Abstract**: This paper presents a new Large Language Model (LLM)-based Smart Device Management framework, a pioneering approach designed to address the intricate challenges of managing intelligent devices within public facilities, with a particular emphasis on applications to libraries. Our framework leverages state-of-the-art LLMs to analyze and predict device failures, thereby enhancing operational efficiency and reliability. Through prototype validation in real-world library settings, we demonstrate the framework's practical applicability and its capacity to significantly reduce budgetary constraints on public facilities. The advanced and innovative nature of our model is evident from its successful implementation in prototype testing. We plan to extend the framework's scope to include a wider array of public facilities and to integrate it with cutting-edge cybersecurity technologies, such as Internet of Things (IoT) security and machine learning algorithms for threat detection and response. This will result in a comprehensive and proactive maintenance system that not only bolsters the security of intelligent devices but also utilizes machine learning for automated analysis and real-time threat mitigation. By incorporating these advanced cybersecurity elements, our framework will be well-positioned to tackle the dynamic challenges of modern public infrastructure, ensuring robust protection against potential threats and enabling facilities to anticipate and prevent failures, leading to substantial cost savings and enhanced service quality. 

**Abstract (ZH)**: 本文提出了一个新的基于大型语言模型（LLM）的智能设备管理框架，这是一种开创性的方法，旨在解决在公共设施中管理智能设备的复杂挑战，尤其着重于图书馆的应用。该框架利用最先进的LLM对设备故障进行分析和预测，从而提高运营效率和可靠性。通过在真实图书馆环境中的原型验证，我们展示了该框架的实用性和其显著减少公共设施预算限制的能力。模型的先进性和创新性也体现在其在原型测试中的成功实施。我们计划将该框架的应用范围扩展到更多的公共设施，并将其与最新的网络安全技术相结合，如物联网（IoT）安全和机器学习算法，以进行威胁检测与响应。这将形成一个全面且前瞻性的维护系统，不仅增强智能设备的安全性，还利用机器学习进行自动化分析和实时威胁缓解。通过整合这些先进的网络安全元素，我们的框架将能够应对现代公共基础设施的动态挑战，确保对潜在威胁的 robust 保护，并使设施能够预见和预防故障，从而实现显著的成本节约和提升服务质量。 

---
# A Novel Task-Driven Method with Evolvable Interactive Agents Using Event Trees for Enhanced Emergency Decision Support 

**Title (ZH)**: 一种基于任务的新型方法，利用事件树中的可进化互动代理，以增强应急管理决策支持 

**Authors**: Xingyu Xiao, Peng Chen, Ben Qi, Jingang Liang, Jiejuan Tong, Haitao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06193)  

**Abstract**: As climate change and other global challenges increase the likelihood of unforeseen emergencies, the limitations of human-driven strategies in critical situations become more pronounced. Inadequate pre-established emergency plans can lead operators to become overwhelmed during complex systems malfunctions. This study addresses the urgent need for agile decision-making in response to various unforeseen incidents through a novel approach, EvoTaskTree (a task-driven method with evolvable interactive agents using event trees for emergency decision support). This advanced approach integrates two types of agents powered by large language models (LLMs): task executors, responsible for executing critical procedures, and task validators, ensuring the efficacy of those actions. By leveraging insights from event tree analysis, our framework encompasses three crucial tasks: initiating event subevent analysis, event tree header event analysis, and decision recommendations. The agents learn from both successful and unsuccessful responses from these tasks. Finally, we use nuclear power plants as a demonstration of a safety-critical system. Our findings indicate that the designed agents are not only effective but also outperform existing approaches, achieving an impressive accuracy rate of up to 100 % in processing previously unencoun32 tered incident scenarios. This paper demonstrates that EvoTaskTree significantly enhances the rapid formulation of emergency decision-making. 

**Abstract (ZH)**: 随着气候变化和其他全球挑战增加未预见紧急情况的可能性，人类驱动的战略在关键时刻的局限性变得更加明显。缺乏充分的应急预案可能导致操作人员在复杂系统故障期间感到不知所措。本研究通过一种新颖的方法EvoTaskTree（一种基于任务的可进化交互代理方法，结合事件树支持应急决策），应对各种未预见的事件对敏捷决策的迫切需求。这种方法结合了由大型语言模型（LLMs）驱动的两种类型的代理：任务执行器，负责执行关键程序；以及任务验证器，确保这些行动的有效性。通过利用事件树分析的见解，我们的框架涵盖了三个关键任务：触发事件子事件分析、事件树Head事件分析和决策建议。代理从这些任务的成功和失败响应中学习。最后，我们使用核电厂作为安全关键系统的演示案例。研究发现，设计的代理不仅有效，而且在处理之前未遇见过的事件场景时比现有方法更为出色，实现了高达100%的处理精度。本文展示了EvoTaskTree显著提高了应急决策的快速制定能力。 

---
# A Multimodal Social Agent 

**Title (ZH)**: 多模态社会代理模型 

**Authors**: Athina Bikaki, Ioannis A. Kakadiaris  

**Link**: [PDF](https://arxiv.org/pdf/2501.06189)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated remarkable progress in common-sense reasoning tasks. This ability is fundamental to understanding social dynamics, interactions, and communication. However, the potential of integrating computers with these social capabilities is still relatively unexplored. However, the potential of integrating computers with these social capabilities is still relatively unexplored. This paper introduces MuSA, a multimodal LLM-based agent that analyzes text-rich social content tailored to address selected human-centric content analysis tasks, such as question answering, visual question answering, title generation, and categorization. It uses planning, reasoning, acting, optimizing, criticizing, and refining strategies to complete a task. Our approach demonstrates that MuSA can automate and improve social content analysis, helping decision-making processes across various applications. We have evaluated our agent's capabilities in question answering, title generation, and content categorization tasks. MuSA performs substantially better than our baselines. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在常识推理任务中取得了显著进展。这种能力对于理解社会动态、交互和沟通至关重要。然而，将计算机与这些社会能力整合的可能性仍然相对未被充分探索。本文介绍了一种基于多模态LLM的代理MuSA，该代理专门用于分析富含文本的社会内容，以应对诸如问答、视觉问答、标题生成和内容分类等人本中心的内容分析任务。MuSA 使用规划、推理、执行、优化、批评和改进等策略来完成任务。我们的方法表明，MuSA 可以自动化并提升社会内容分析，帮助各类应用中的决策过程。我们已在问答、标题生成和内容分类任务上评估了该代理的能力。MuSA 在这些任务上的表现显著优于我们的基线模型。 

---
# RadAlign: Advancing Radiology Report Generation with Vision-Language Concept Alignment 

**Title (ZH)**: RadAlign：通过视觉-语言概念对齐提升放射学报告生成 

**Authors**: Difei Gu, Yunhe Gao, Yang Zhou, Mu Zhou, Dimitris Metaxas  

**Link**: [PDF](https://arxiv.org/pdf/2501.07525)  

**Abstract**: Automated chest radiographs interpretation requires both accurate disease classification and detailed radiology report generation, presenting a significant challenge in the clinical workflow. Current approaches either focus on classification accuracy at the expense of interpretability or generate detailed but potentially unreliable reports through image captioning techniques. In this study, we present RadAlign, a novel framework that combines the predictive accuracy of vision-language models (VLMs) with the reasoning capabilities of large language models (LLMs). Inspired by the radiologist's workflow, RadAlign first employs a specialized VLM to align visual features with key medical concepts, achieving superior disease classification with an average AUC of 0.885 across multiple diseases. These recognized medical conditions, represented as text-based concepts in the aligned visual-language space, are then used to prompt LLM-based report generation. Enhanced by a retrieval-augmented generation mechanism that grounds outputs in similar historical cases, RadAlign delivers superior report quality with a GREEN score of 0.678, outperforming state-of-the-art methods' 0.634. Our framework maintains strong clinical interpretability while reducing hallucinations, advancing automated medical imaging and report analysis through integrated predictive and generative AI. Code is available at this https URL. 

**Abstract (ZH)**: 自动化胸片解释要求同时具备准确的疾病分类和详细的放射学报告生成能力，这在临床工作流程中提出了重大挑战。当前的方法要么侧重于提高分类准确性而牺牲可解释性，要么通过图像字幕技术生成详细的但可能不稳定的报告。在此研究中，我们提出了RadAlign，这是一个新型框架，结合了视觉-语言模型（VLMs）的预测准确性与大型语言模型（LLMs）的推理能力。RadAlign受到放射科医生工作流程的启发，首先使用一种专门的VLM将视觉特征与关键医学概念对齐，实现了多种疾病平均AUC达0.885的优秀疾病分类。这些识别出的医学条件以文本形式表示在对齐的视觉-语言空间中，随后用于触发基于LLM的报告生成。通过一种检索增强生成机制，将输出与类似的历史病例关联起来，RadAlign提供了质量更优的报告，其GREEN评分为0.678，优于现有方法的0.634。我们的框架保持了强烈的临床可解释性，同时减少了幻觉现象，通过整合预测和生成AI技术，推动了自动化医学影像和报告分析的发展。代码可在以下链接获取：this https URL。 

---
# Emergent effects of scaling on the functional hierarchies within large language models 

**Title (ZH)**: 大规模语言模型内部功能层次结构中的扩展示效 

**Authors**: Paul C. Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2501.07359)  

**Abstract**: Large language model (LLM) architectures are often described as functionally hierarchical: Early layers process syntax, middle layers begin to parse semantics, and late layers integrate information. The present work revisits these ideas. This research submits simple texts to an LLM (e.g., "A church and organ") and extracts the resulting activations. Then, for each layer, support vector machines and ridge regressions are fit to predict a text's label and thus examine whether a given layer encodes some information. Analyses using a small model (Llama-3.2-3b; 28 layers) partly bolster the common hierarchical perspective: Item-level semantics are most strongly represented early (layers 2-7), then two-item relations (layers 8-12), and then four-item analogies (layers 10-15). Afterward, the representation of items and simple relations gradually decreases in deeper layers that focus on more global information. However, several findings run counter to a steady hierarchy view: First, although deep layers can represent document-wide abstractions, deep layers also compress information from early portions of the context window without meaningful abstraction. Second, when examining a larger model (Llama-3.3-70b-Instruct), stark fluctuations in abstraction level appear: As depth increases, two-item relations and four-item analogies initially increase in their representation, then markedly decrease, and afterward increase again momentarily. This peculiar pattern consistently emerges across several experiments. Third, another emergent effect of scaling is coordination between the attention mechanisms of adjacent layers. Across multiple experiments using the larger model, adjacent layers fluctuate between what information they each specialize in representing. In sum, an abstraction hierarchy often manifests across layers, but large models also deviate from this structure in curious ways. 

**Abstract (ZH)**: 大型语言模型（LLM）架构常被描述为功能分层的：早期层处理句法，中期层开始解析语义，晚期层整合信息。本研究重新探讨了这些观点。本研究将简单文本（例如，“一座教堂和一个风琴”）提交给LLM，并提取生成的激活信号。然后，对每一层，使用支持向量机和岭回归来预测文本的标签，从而检查给定层是否编码了一些信息。使用小型模型（Llama-3.2-3b，28层）的分析部分支持了常见的分层观点：项目级别的语义在早期（第2层至第7层）最为强烈地表征，然后是两项目关系（第8层至第12层），接着是四项目类比（第10层至第15层）。之后，在专注于更全局信息的深层层中，代表性项目和简单关系逐渐减少。然而，一些发现与稳定的分层观点相矛盾：首先，尽管深层层可以表示文档级抽象，但它们也会压缩来自上下文窗口早期部分的信息，而这种压缩缺乏有意义的抽象。其次，在研究一个较大的模型（Llama-3.3-70b-Instruct）时，观察到抽象层次显著波动：随着深度增加，两项目关系和四项目类比的表示最初增加，随后大幅减少，之后又短暂增加。这种奇怪的模式在多个实验中持续出现。第三，随着模型规模的扩大，另一现象是相邻层之间的注意力机制协调。在使用较大模型进行的多个实验中，相邻层在它们各自专长表示的信息上波动。综上所述，抽象层次经常在各层间表现出来，但大型模型也会以奇怪的方式偏离这种结构。 

---
# The Lessons of Developing Process Reward Models in Mathematical Reasoning 

**Title (ZH)**: 在数学推理中开发过程奖励模型的启示 

**Authors**: Zhenru Zhang, Chujie Zheng, Yangzhen Wu, Beichen Zhang, Runji Lin, Bowen Yu, Dayiheng Liu, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.07301)  

**Abstract**: Process Reward Models (PRMs) emerge as a promising approach for process supervision in mathematical reasoning of Large Language Models (LLMs), which aim to identify and mitigate intermediate errors in the reasoning processes. However, the development of effective PRMs faces significant challenges, particularly in data annotation and evaluation methodologies. In this paper, through extensive experiments, we demonstrate that commonly used Monte Carlo (MC) estimation-based data synthesis for PRMs typically yields inferior performance and generalization compared to LLM-as-a-judge and human annotation methods. MC estimation relies on completion models to evaluate current-step correctness, leading to inaccurate step verification. Furthermore, we identify potential biases in conventional Best-of-N (BoN) evaluation strategies for PRMs: (1) The unreliable policy models generate responses with correct answers but flawed processes, leading to a misalignment between the evaluation criteria of BoN and the PRM objectives of process verification. (2) The tolerance of PRMs of such responses leads to inflated BoN scores. (3) Existing PRMs have a significant proportion of minimum scores concentrated on the final answer steps, revealing the shift from process to outcome-based assessment in BoN Optimized PRMs. To address these challenges, we develop a consensus filtering mechanism that effectively integrates MC estimation with LLM-as-a-judge and advocates a more comprehensive evaluation framework that combines response-level and step-level metrics. Based on the mechanisms, we significantly improve both model performance and data efficiency in the BoN evaluation and the step-wise error identification task. Finally, we release a new state-of-the-art PRM that outperforms existing open-source alternatives and provides practical guidelines for future research in building process supervision models. 

**Abstract (ZH)**: 过程奖励模型（PRMs）作为一种有前景的方法，在大型语言模型（LLMs）的数学推理过程中实现过程监督，旨在识别和缓解推理过程中的中间错误。然而，有效PRMs的发展面临着重大的挑战，尤其是在数据标注和评估方法方面。本文通过大量的实验表明，通常用于PRMs的数据合成方法——基于蒙特卡洛（MC）估计的方法，通常在性能和泛化能力上劣于LLM-as-a-judge和人工标注方法。MC估计依赖于完成模型来评估当前步骤的正确性，导致步骤验证不够准确。此外，我们还发现了常规的Best-of-N（BoN）评估策略在PRMs中的潜在偏差：（1）不可靠的策略模型生成正确答案但过程有误的答案，导致BoN的评估标准与PRM的过程验证目标不一致。（2）PRMs对这种答案的容忍度导致BoN得分膨胀。（3）现有的PRMs在最终答案步骤上有相当大的低分比例，显示出BoN优化后的PRMs从过程导向评估转向结果导向评估。为解决这些问题，我们开发了一种共识过滤机制，有效地将MC估计与LLM-as-a-judge相结合，并倡导一种更全面的评估框架，结合响应级和步骤级指标。基于这种机制，我们在BoN评估和逐阶错误识别任务中显著提高了模型性能和数据效率。最后，我们发布了一个新的状态最先进PRM，优于现有开源选项，并提供了未来研究中构建过程监督模型的实用指南。 

---
# Logic Meets Magic: LLMs Cracking Smart Contract Vulnerabilities 

**Title (ZH)**: 逻辑遇魔法：大规模语言模型破解智能合约漏洞 

**Authors**: ZeKe Xiao, Qin Wang, Hammond Pearce, Shiping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.07058)  

**Abstract**: Smart contract vulnerabilities caused significant economic losses in blockchain applications. Large Language Models (LLMs) provide new possibilities for addressing this time-consuming task. However, state-of-the-art LLM-based detection solutions are often plagued by high false-positive rates.
In this paper, we push the boundaries of existing research in two key ways. First, our evaluation is based on Solidity v0.8, offering the most up-to-date insights compared to prior studies that focus on older versions (v0.4). Second, we leverage the latest five LLM models (across companies), ensuring comprehensive coverage across the most advanced capabilities in the field.
We conducted a series of rigorous evaluations. Our experiments demonstrate that a well-designed prompt can reduce the false-positive rate by over 60%. Surprisingly, we also discovered that the recall rate for detecting some specific vulnerabilities in Solidity v0.8 has dropped to just 13% compared to earlier versions (i.e., v0.4). Further analysis reveals the root cause of this decline: the reliance of LLMs on identifying changes in newly introduced libraries and frameworks during detection. 

**Abstract (ZH)**: 智能合约漏洞造成了区块链应用中的重大经济损失。大规模语言模型（LLMs）为解决这一耗时任务提供了新的可能性。然而，现有的基于LLM的检测解决方案通常存在较高的误报率。
本文在以下两个关键方面推动了现有研究的边界。首先，我们的评估基于Solidity v0.8，提供了比以前依赖于较旧版本（v0.4）的研究更为前沿的洞察。其次，我们利用了最新的五种LLM模型（来自不同公司），确保在最先进的能力范围内实现全面覆盖。
我们进行了系列严格的评估。实验结果显示，精心设计的提示可以将误报率降低超过60%。令我们惊讶的是，我们还发现，在检测某些特定漏洞方面，对于Solidity v0.8的召回率已下降至13%，而早期版本（例如v0.4）的召回率更高。进一步的分析揭示了这种下降的原因：LLM在检测过程中依赖于识别新引入的库和框架的变化。 

---
# Combining LLM decision and RL action selection to improve RL policy for adaptive interventions 

**Title (ZH)**: 将大规模语言模型的决策与强化学习的动作选择相结合，以改善自适应干预的强化学习策略 

**Authors**: Karine Karine, Benjamin M. Marlin  

**Link**: [PDF](https://arxiv.org/pdf/2501.06980)  

**Abstract**: Reinforcement learning (RL) is increasingly being used in the healthcare domain, particularly for the development of personalized health adaptive interventions. Inspired by the success of Large Language Models (LLMs), we are interested in using LLMs to update the RL policy in real time, with the goal of accelerating personalization. We use the text-based user preference to influence the action selection on the fly, in order to immediately incorporate the user preference. We use the term "user preference" as a broad term to refer to a user personal preference, constraint, health status, or a statement expressing like or dislike, etc. Our novel approach is a hybrid method that combines the LLM response and the RL action selection to improve the RL policy. Given an LLM prompt that incorporates the user preference, the LLM acts as a filter in the typical RL action selection. We investigate different prompting strategies and action selection strategies. To evaluate our approach, we implement a simulation environment that generates the text-based user preferences and models the constraints that impact behavioral dynamics. We show that our approach is able to take into account the text-based user preferences, while improving the RL policy, thus improving personalization in adaptive intervention. 

**Abstract (ZH)**: 强化学习（RL）在医疗健康领域的应用越来越广泛，尤其是在个性化健康适应性干预的发展中。受大型语言模型（LLMs）成功应用的启发，我们感兴趣的是使用LLMs实时更新RL策略，以加速个性化过程。通过文本形式的用户偏好影响即时动作选择，以便立即纳入用户偏好。我们将“用户偏好”定义为用户的个人偏好、约束条件、健康状况，或表达喜欢和不喜欢的陈述等广泛的内容。我们的新颖方法是一种将LLMs响应与RL动作选择相结合的混合方法，以改进RL策略。给定包含用户偏好的LLM提示，LLM作为典型的RL动作选择过程中的筛选器发挥作用。我们研究了不同的提示策略和动作选择策略。为了评估我们的方法，我们构建了一个仿真环境，生成基于文本的用户偏好，并建模影响行为动力学的各种限制条件。结果显示，我们的方法能够考虑到基于文本的用户偏好，从而在适应性干预中提高个性化水平。 

---
# Transfer Learning of Tabular Data by Finetuning Large Language Models 

**Title (ZH)**: 通过微调大规模语言模型进行表格数据的迁移学习 

**Authors**: Shourav B. Rabbani, Ibna Kowsar, Manar D. Samad  

**Link**: [PDF](https://arxiv.org/pdf/2501.06863)  

**Abstract**: Despite the artificial intelligence (AI) revolution, deep learning has yet to achieve much success with tabular data due to heterogeneous feature space and limited sample sizes without viable transfer learning. The new era of generative AI, powered by large language models (LLM), brings unprecedented learning opportunities to diverse data and domains. This paper investigates the effectiveness of an LLM application programming interface (API) and transfer learning of LLM in tabular data classification. LLM APIs respond to input text prompts with tokenized data and instructions, whereas transfer learning finetunes an LLM for a target classification task. This paper proposes an end-to-end finetuning of LLM to demonstrate cross-data transfer learning on ten benchmark data sets when large pre-trained tabular data models do not exist to facilitate transfer learning. The proposed LLM finetuning method outperforms state-of-the-art machine and deep learning methods on tabular data with less than ten features - a standard feature size for tabular data sets. The transfer learning approach uses a fraction of the computational cost of other deep learning or API-based solutions while ensuring competitive or superior classification performance. 

**Abstract (ZH)**: 尽管人工智能（AI）革命已经在多个领域取得了显著进展，但深层学习在处理表格数据方面尚未取得显著成功，原因在于特征空间的多样性以及样本量有限，缺乏有效的迁移学习方法。由大规模语言模型（LLM）驱动的生成型AI新纪元为各种数据和领域带来了前所未有的学习机会。本文探讨了大规模语言模型应用编程接口（API）及其在表格数据分类中的迁移学习效果。大规模语言模型API根据输入文本提示生成标记化数据和指令，而迁移学习则针对目标分类任务对大规模语言模型进行微调。本文提出了一种端到端的迁移学习方法，利用大规模预训练表格数据模型的缺失来在十个基准数据集上展示跨数据集的迁移学习效果。所提出的大规模语言模型微调方法在特征数少于十个（表格数据集的标准特征数量）的表格数据分类任务中优于最先进的机器学习和深度学习方法。迁移学习方法使用其他深度学习或基于API方法计算成本的一小部分，同时仍能确保具有竞争力或优越的分类性能。 

---
# Bridging the Fairness Gap: Enhancing Pre-trained Models with LLM-Generated Sentences 

**Title (ZH)**: 填补公平性差距：通过LLM生成的句子提升预训练模型 

**Authors**: Liu Yu, Ludie Guo, Ping Kuang, Fan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.06795)  

**Abstract**: Pre-trained language models (PLMs) are trained on data that inherently contains gender biases, leading to undesirable impacts. Traditional debiasing methods often rely on external corpora, which may lack quality, diversity, or demographic balance, affecting the effectiveness of debiasing. With the rise of large language models and their extensive knowledge, we propose enhancing fairness (Fair-Gender) in PLMs by absorbing coherent, attribute-balanced, and semantically rich sentences. However, these sentences cannot be directly used for debiasing due to alignment issues and the risk of negative transfer. We address this by applying causal analysis to estimate causal effects, filtering out unaligned sentences, and identifying aligned ones for incorporation into PLMs, thereby ensuring positive transfer. Experiments show that our approach significantly reduces gender biases in PLMs while preserving their language expressiveness. 

**Abstract (ZH)**: 预训练语言模型（PLMs）在训练过程中通常会包含固有的性别偏见，这会导致不良影响。传统的去偏见方法往往依赖外部语料库，而这些外部语料库可能在质量、多样性和人口统计学平衡方面存在不足，从而影响去偏见的有效性。随着大规模语言模型的兴起及其广泛的知识覆盖，我们提出通过吸收一致、属性平衡且语义丰富的句子来增强PLMs的公平性（Fair-Gender）。然而，这些句子不能直接用于去偏见，因为这可能会导致对齐问题和负向迁移的风险。为此，我们采用因果分析估计因果效应，过滤掉对齐不上的句子，并确定对齐的句子以融入PLMs，从而确保正向迁移。实验结果表明，我们的方法能够在显著减少PLMs中的性别偏见的同时，保持其语言表达能力。 

---
# ZNO-Eval: Benchmarking reasoning capabilities of large language models in Ukrainian 

**Title (ZH)**: ZNO-Eval：评估大型语言模型在乌克兰语中的推理能力 

**Authors**: Mykyta Syromiatnikov, Victoria Ruvinskaya, Anastasiya Troynina  

**Link**: [PDF](https://arxiv.org/pdf/2501.06715)  

**Abstract**: As the usage of large language models for problems outside of simple text understanding or generation increases, assessing their abilities and limitations becomes crucial. While significant progress has been made in this area over the last few years, most research has focused on benchmarking English, leaving other languages underexplored. This makes evaluating the reasoning and robustness level of language models in Ukrainian particularly challenging. The purpose of this work is to establish a comprehensive benchmark for the reasoning capabilities evaluation of large language models in the Ukrainian language. This paper presents the ZNO-Eval benchmark based on real exam tasks from Ukraine's standardized educational testing system: the External Independent Evaluation and the National Multi-subject Test. With single-answer options, multiple-choice, matching, and open-ended questions from diverse subjects, including Ukrainian language, mathematics, history, and geography, this dataset paves the way toward a thorough analysis of reasoning capabilities across different domains and complexities. Evaluation of several well-known language models, such as GPT-3.5-Turbo, GPT-4o, GPT-4-Turbo, Mistral Large, Claude 3 Opus, and Gemini-1.5 Pro on this benchmark demonstrated the superiority of GPT-4o in both common knowledge reasoning and intricate language tasks. At the same time, Gemini Pro and GPT-4 Turbo excelled in the arithmetic domain, leading in single-answer and open-ended math problems. While all models were close to max performance in text-only common knowledge tasks like history and geography, there still is a gap for Ukrainian language and math, thus highlighting the importance of developing specialized language benchmarks for more accurate assessments of model capabilities and limitations across different languages and contexts. 

**Abstract (ZH)**: 随着大型语言模型在复杂文本理解和生成之外的问题上的应用增加，评估其能力和限制变得尤为重要。尽管在过去几年间该领域取得了显著进展，但大多数研究仍集中在基准测试英语上，而其他语言则较少受到关注。这使得评估乌克兰语语言模型的推理能力和稳健性变得尤为具有挑战性。本文的目的是建立一个全面的基准，以评估大型语言模型在乌克兰语中的推理能力。本文基于乌克兰标准化教育评估系统——外部独立评估和国家多学科测试的真实考试任务，提出了ZNO-Eval基准。该数据集包含多个答案选项的选择题、匹配题、开放性问题以及来自不同学科的题型，包括乌克兰语、数学、历史和地理等，为跨不同领域和复杂性层次的推理能力进行深入分析铺平了道路。

对该基准上几种知名语言模型（如GPT-3.5-Turbo、GPT-4o、GPT-4-Turbo、Mistral Large、Claude 3 Opus 和 Gemini-1.5 Pro）的评估表明，GPT-4o在常识推理和复杂语言任务方面均表现优越。同时，Gemini Pro 和 GPT-4 Turbo 在数学领域表现出色，领先于单选题和开放性数学问题。尽管所有模型在仅涉及文本的常识任务（如历史和地理）中的表现均接近最佳水平，但乌克兰语和数学领域仍然存在差距，这突显了开发针对不同语言和应用场景的专业化语言基准以准确评估模型能力及限制的重要性。 

---
# ChemAgent: Self-updating Library in Large Language Models Improves Chemical Reasoning 

**Title (ZH)**: ChemAgent：在大型语言模型中自更新的化学知识库增强化学推理 

**Authors**: Xiangru Tang, Tianyu Hu, Muyang Ye, Yanjun Shao, Xunjian Yin, Siru Ouyang, Wangchunshu Zhou, Pan Lu, Zhuosheng Zhang, Yilun Zhao, Arman Cohan, Mark Gerstein  

**Link**: [PDF](https://arxiv.org/pdf/2501.06590)  

**Abstract**: Chemical reasoning usually involves complex, multi-step processes that demand precise calculations, where even minor errors can lead to cascading failures. Furthermore, large language models (LLMs) encounter difficulties handling domain-specific formulas, executing reasoning steps accurately, and integrating code effectively when tackling chemical reasoning tasks. To address these challenges, we present ChemAgent, a novel framework designed to improve the performance of LLMs through a dynamic, self-updating library. This library is developed by decomposing chemical tasks into sub-tasks and compiling these sub-tasks into a structured collection that can be referenced for future queries. Then, when presented with a new problem, ChemAgent retrieves and refines pertinent information from the library, which we call memory, facilitating effective task decomposition and the generation of solutions. Our method designs three types of memory and a library-enhanced reasoning component, enabling LLMs to improve over time through experience. Experimental results on four chemical reasoning datasets from SciBench demonstrate that ChemAgent achieves performance gains of up to 46% (GPT-4), significantly outperforming existing methods. Our findings suggest substantial potential for future applications, including tasks such as drug discovery and materials science. Our code can be found at this https URL 

**Abstract (ZH)**: 化学推理通常涉及复杂、多步的过程，需要精确的计算，其中即使是轻微的错误也可能导致一系列的失败。此外，大型语言模型（LLMs）在处理特定领域的公式、准确执行推理步骤以及有效集成代码时，在应对化学推理任务时遇到困难。为了解决这些挑战，我们提出了ChemAgent，这是一种新的框架，旨在通过动态的自我更新库来提高LLMs的表现。该库通过将化学任务分解为子任务，并将这些子任务编译成一个有结构的集合来进行开发，该集合可以为未来的查询提供参考。当面临新的问题时，ChemAgent会从库（我们称为记忆）中检索和细化相关信息，从而促进有效的任务分解和解决方案的生成。我们的方法设计了三种类型的记忆和一个增强库的推理组件，使LLMs能够在经验中不断提高。SciBench的四种化学推理数据集的实验结果表明，ChemAgent可实现高达46%（GPT-4）的性能提升，显著优于现有方法。我们的研究结果表明，ChemAgent在药物发现和材料科学等领域的未来应用具有巨大的潜力。相关的代码可以在以下链接找到：[这个链接] 

---
# PASS: Presentation Automation for Slide Generation and Speech 

**Title (ZH)**: PASS：幻灯片生成与演讲的自动化呈现 

**Authors**: Tushar Aggarwal, Aarohi Bhand  

**Link**: [PDF](https://arxiv.org/pdf/2501.06497)  

**Abstract**: In today's fast-paced world, effective presentations have become an essential tool for communication in both online and offline meetings. The crafting of a compelling presentation requires significant time and effort, from gathering key insights to designing slides that convey information clearly and concisely. However, despite the wealth of resources available, people often find themselves manually extracting crucial points, analyzing data, and organizing content in a way that ensures clarity and impact. Furthermore, a successful presentation goes beyond just the slides; it demands rehearsal and the ability to weave a captivating narrative to fully engage the audience. Although there has been some exploration of automating document-to-slide generation, existing research is largely centered on converting research papers. In addition, automation of the delivery of these presentations has yet to be addressed. We introduce PASS, a pipeline used to generate slides from general Word documents, going beyond just research papers, which also automates the oral delivery of the generated slides. PASS analyzes user documents to create a dynamic, engaging presentation with an AI-generated voice. Additionally, we developed an LLM-based evaluation metric to assess our pipeline across three critical dimensions of presentations: relevance, coherence, and redundancy. The data and codes are available at this https URL. 

**Abstract (ZH)**: 在当今快节奏的世界中，有效的演示文稿已成为在线和线下会议中沟通的重要工具。精心制作的引人入胜的演示文稿需要大量时间和努力，从收集关键见解到设计能够清晰简洁地传达信息的幻灯片。然而，尽管有大量的资源可用，人们常常发现自己在手动提取关键点、分析数据和组织内容以确保清晰性和影响力上花费大量时间。此外，成功的演示不仅仅局限于幻灯片；它还需要排练以及能够编织引人入胜的故事的能力，以完全吸引观众。尽管已经有一些自动化文档到幻灯片生成的研究，但现有研究主要集中在转换研究论文上。此外，对这些演示文稿的自动化呈现还未被解决。我们引入了PASS，这是一种用于从通用Word文档生成幻灯片的管道，不仅限于研究论文，同时还可以自动化生成的幻灯片的口头呈现。PASS通过生成一个由AI生成声音的动态和引人入胜的演示文稿来分析用户文档。此外，我们还开发了一种基于大语言模型的评估指标，用于从三个关键维度评估我们的管道：相关性、连贯性和冗余性。数据和代码可在以下网址获取：**此处网址**。 

---
# MedCT: A Clinical Terminology Graph for Generative AI Applications in Healthcare 

**Title (ZH)**: MedCT：医疗术语图谱在医疗健康生成式AI应用中的临床术语图 

**Authors**: Ye Chen, Dongdong Huang, Haoyun Xu, Cong Fu, Lin Sheng, Qingli Zhou, Yuqiang Shen, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06465)  

**Abstract**: We introduce the world's first clinical terminology for the Chinese healthcare community, namely MedCT, accompanied by a clinical foundation model MedBERT and an entity linking model MedLink. The MedCT system enables standardized and programmable representation of Chinese clinical data, successively stimulating the development of new medicines, treatment pathways, and better patient outcomes for the populous Chinese community. Moreover, the MedCT knowledge graph provides a principled mechanism to minimize the hallucination problem of large language models (LLMs), therefore achieving significant levels of accuracy and safety in LLM-based clinical applications. By leveraging the LLMs' emergent capabilities of generativeness and expressiveness, we were able to rapidly built a production-quality terminology system and deployed to real-world clinical field within three months, while classical terminologies like SNOMED CT have gone through more than twenty years development. Our experiments show that the MedCT system achieves state-of-the-art (SOTA) performance in semantic matching and entity linking tasks, not only for Chinese but also for English. We also conducted a longitudinal field experiment by applying MedCT and LLMs in a representative spectrum of clinical tasks, including electronic health record (EHR) auto-generation and medical document search for diagnostic decision making. Our study shows a multitude of values of MedCT for clinical workflows and patient outcomes, especially in the new genre of clinical LLM applications. We present our approach in sufficient engineering detail, such that implementing a clinical terminology for other non-English societies should be readily reproducible. We openly release our terminology, models and algorithms, along with real-world clinical datasets for the development. 

**Abstract (ZH)**: 我们向中国的医疗社区引入了世界首款临床术语体系，称之为MedCT，并配以临床基础模型MedBERT和实体链接模型MedLink。MedCT系统能够规范且编程式的表示中文临床数据，从而驱动新的药物研发、治疗路径设计以及改善庞大中国患者群体的健康结果。此外，MedCT知识图谱提供了一种从根本上减少大型语言模型（LLMs）幻觉问题的机制，从而在基于LLMs的临床应用中实现了显著的准确性和安全性。通过利用LLMs生成性和表达性的新兴能力，我们能够在三个月内快速构建出生产级的术语体系，并将其部署到真实的临床领域，而像SNOMED CT这样的传统术语体系则经历了超过二十年的发展。我们的实验表明，MedCT系统在语义匹配和实体链接任务上均达到了现有最佳（SOTA）性能，不仅适用于中文，也适用于英文。我们还进行了长期现场实验，将MedCT和LLMs应用于代表性的多种临床任务，包括电子健康记录（EHR）自动化生成和医学文档搜索以支持诊断决策。研究结果表明，MedCT在临床工作流程和患者结果方面具有多种价值，尤其是在临床LLMs应用的新领域。我们以足够的工程技术细节介绍了我们的方法，使得为其他非英文社会构建临床术语体系变得可重现。我们公开发布了我们的术语体系、模型和算法，并提供了实际的临床数据集，以促进其发展。 

---
# Towards a Probabilistic Framework for Analyzing and Improving LLM-Enabled Software 

**Title (ZH)**: 面向LLM赋能软件分析与改进的概率框架研究 

**Authors**: Juan Manuel Baldonado, Flavia Bonomo-Braberman, Víctor Adrián Braberman  

**Link**: [PDF](https://arxiv.org/pdf/2501.06370)  

**Abstract**: Ensuring the reliability and verifiability of large language model (LLM)-enabled systems remains a significant challenge in software engineering. We propose a probabilistic framework for systematically analyzing and improving these systems by modeling and refining distributions over clusters of semantically equivalent outputs. This framework facilitates the evaluation and iterative improvement of Transference Models -- key software components that utilize LLMs to transform inputs into outputs for downstream tasks. To illustrate its utility, we apply the framework to the autoformalization problem, where natural language documentation is transformed into formal program specifications. Our case illustrates how probabilistic analysis enables the identification of weaknesses and guides focused alignment improvements, resulting in more reliable and interpretable outputs. This principled approach offers a foundation for addressing critical challenges in the development of robust LLM-enabled systems. 

**Abstract (ZH)**: 确保大型语言模型（LLM）驱动系统的可靠性和可验证性仍然是软件工程中的一个重大挑战。我们提出了一种概率框架，通过建模和细化语义等效输出的分布来系统地分析和改进这些系统。该框架有助于评估和迭代改进迁移模型——这些模型利用LLM将输入转换为下游任务的输出。为了展示其实用性，我们将该框架应用于自动形式化问题，即将自然语言文档转换为形式化程序规范。我们的案例研究展示了概率分析如何帮助识别薄弱环节并指导集中对齐改进，从而产生更可靠和可解释的输出。这种原则性的方法为解决稳健的LLM驱动系统开发中的关键挑战奠定了基础。 

---
# Gender-Neutral Large Language Models for Medical Applications: Reducing Bias in PubMed Abstracts 

**Title (ZH)**: 面向医疗应用的性别中立大型语言模型：减少PubMed摘要中的偏见 

**Authors**: Elizabeth Schaefer, Kirk Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2501.06365)  

**Abstract**: This paper presents a pipeline for mitigating gender bias in large language models (LLMs) used in medical literature by neutralizing gendered occupational pronouns. A dataset of 379,000 PubMed abstracts from 1965-1980 was processed to identify and modify pronouns tied to professions. We developed a BERT-based model, ``Modern Occupational Bias Elimination with Refined Training,'' or ``MOBERT,'' trained on these neutralized abstracts, and compared its performance with ``1965Bert,'' trained on the original dataset. MOBERT achieved a 70\% inclusive replacement rate, while 1965Bert reached only 4\%. A further analysis of MOBERT revealed that pronoun replacement accuracy correlated with the frequency of occupational terms in the training data. We propose expanding the dataset and refining the pipeline to improve performance and ensure more equitable language modeling in medical applications. 

**Abstract (ZH)**: 本文提出了一种管道方法，用于减轻医学文献中大型语言模型（LLMs）中的性别偏差，通过对职业性代词进行中性化处理。我们处理了一个包含1965年至1980年间379,000篇PubMed摘要的数据集，以识别并修改与职业相关的代词。我们开发了一个基于BERT的模型，名为“改进训练的现代职业偏见消除”，或简称“MOBERT”，该模型在这些中性化摘要上进行了训练，并将其性能与在原始数据集上训练的“1965Bert”进行了比较。MOBERT实现了70%的包容性替换率，而1965Bert仅为4%。进一步的分析表明，代词替换的准确性与训练数据中职业术语的频率相关。我们建议扩大数据集并完善管道，以提高性能并确保在医学应用中的更公平的语言模型。 

---
# Bactrainus: Optimizing Large Language Models for Multi-hop Complex Question Answering Tasks 

**Title (ZH)**: Bactrainus：优化大型语言模型以应对多跳复杂问答任务 

**Authors**: Iman Barati, Arash Ghafouri, Behrouz Minaei-Bidgoli  

**Link**: [PDF](https://arxiv.org/pdf/2501.06286)  

**Abstract**: In recent years, the use of large language models (LLMs) has significantly increased, and these models have demonstrated remarkable performance in a variety of general language tasks. However, the evaluation of their performance in domain-specific tasks, particularly those requiring deep natural language understanding, has received less attention. In this research, we evaluate the ability of large language models in performing domain-specific tasks, focusing on the multi-hop question answering (MHQA) problem using the HotpotQA dataset. This task, due to its requirement for reasoning and combining information from multiple textual sources, serves as a challenging benchmark for assessing the language comprehension capabilities of these models. To tackle this problem, we have designed a two-stage selector-reader architecture, where each stage utilizes an independent LLM. In addition, methods such as Chain of Thought (CoT) and question decomposition have been employed to investigate their impact on improving the model's performance. The results of the study show that the integration of large language models with these techniques can lead to up to a 4% improvement in F1 score for finding answers, providing evidence of the models' ability to handle domain-specific tasks and their understanding of complex language. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）的使用显著增加，这些模型在各种通用语言任务中表现出色。然而，这些模型在领域特定任务中的评估，尤其是在要求深度自然语言理解的情况下，受到的关注较少。本研究旨在评估大规模语言模型在执行领域特定任务的能力，特别是在使用HotpotQA数据集评估多跳问答（MHQA）问题方面的表现。由于该任务需要进行推理并结合来自多个文本源的信息，它为评估这些模型的语言理解能力提供了一个具有挑战性的基准。为了解决这个问题，我们设计了一个两阶段选择器-阅读器架构，其中每个阶段都使用独立的LLM。此外，我们还采用了链式思维（Chain of Thought，CoT）和问题分解等方法，以研究这些方法对提升模型性能的影响。研究结果表明，将这些技术与大规模语言模型相结合，可以带来高达4%的F1分数改进，这证明了这些模型能够处理领域特定任务并对复杂语言的理解能力。 

---
# MinMo: A Multimodal Large Language Model for Seamless Voice Interaction 

**Title (ZH)**: MinMo：一种支持无缝语音交互的多模态大型语言模型 

**Authors**: Qian Chen, Yafeng Chen, Yanni Chen, Mengzhe Chen, Yingda Chen, Chong Deng, Zhihao Du, Ruize Gao, Changfeng Gao, Zhifu Gao, Yabin Li, Xiang Lv, Jiaqing Liu, Haoneng Luo, Bin Ma, Chongjia Ni, Xian Shi, Jialong Tang, Hui Wang, Hao Wang, Wen Wang, Yuxuan Wang, Yunlan Xu, Fan Yu, Zhijie Yan, Yexin Yang, Baosong Yang, Xian Yang, Guanrou Yang, Tianyu Zhao, Qinglin Zhang, Shiliang Zhang, Nan Zhao, Pei Zhang, Chong Zhang, Jinren Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.06282)  

**Abstract**: Recent advancements in large language models (LLMs) and multimodal speech-text models have laid the groundwork for seamless voice interactions, enabling real-time, natural, and human-like conversations. Previous models for voice interactions are categorized as native and aligned. Native models integrate speech and text processing in one framework but struggle with issues like differing sequence lengths and insufficient pre-training. Aligned models maintain text LLM capabilities but are often limited by small datasets and a narrow focus on speech tasks. In this work, we introduce MinMo, a Multimodal Large Language Model with approximately 8B parameters for seamless voice interaction. We address the main limitations of prior aligned multimodal models. We train MinMo through multiple stages of speech-to-text alignment, text-to-speech alignment, speech-to-speech alignment, and duplex interaction alignment, on 1.4 million hours of diverse speech data and a broad range of speech tasks. After the multi-stage training, MinMo achieves state-of-the-art performance across various benchmarks for voice comprehension and generation while maintaining the capabilities of text LLMs, and also facilitates full-duplex conversation, that is, simultaneous two-way communication between the user and the system. Moreover, we propose a novel and simple voice decoder that outperforms prior models in voice generation. The enhanced instruction-following capabilities of MinMo supports controlling speech generation based on user instructions, with various nuances including emotions, dialects, and speaking rates, and mimicking specific voices. For MinMo, the speech-to-text latency is approximately 100ms, full-duplex latency is approximately 600ms in theory and 800ms in practice. The MinMo project web page is this https URL, and the code and models will be released soon. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）和多模态语音-文本模型的发展为无缝语音交互打下了基础，使其能够实现实时、自然和类人的对话。此前用于语音交互的模型主要分为两类：原生模型和对齐模型。原生模型将语音和文本处理融合在一个框架中，但面临序列长度不一致和预训练不足等问题。对齐模型保留了文本LLM的能力，但常常受限于小的数据集和范围狭窄的语音任务。在这项工作中，我们提出了一种名为MinMo的多模态大型语言模型，参数量约为8亿，旨在实现无缝语音交互。我们解决了此前对齐多模态模型的主要限制。通过多次阶段的语音到文本对齐、文本到语音对齐、语音到语音对齐以及双向交互对齐训练，MinMo在140万小时的多样语音数据和广泛的语音任务上进行了训练。经过多阶段训练后，MinMo在各种语音理解和生成基准测试中表现出最先进的性能，同时保持了文本LLM的能力，并实现了全双工对话，即用户与系统之间的双向通信。此外，我们提出了一种新的简单语音解码器，表现优于之前的模型。MinMo增强的指令遵循能力支持基于用户指令控制语音生成，包括情感、方言、说话速度等多种细微差别，并能够模拟特定的声音。对于MinMo，语音到文本的延迟约为100毫秒，理论上的全双工延迟约为600毫秒，实际应用中约为800毫秒。MinMo项目的网页地址是 <https://XXXXX>，代码和模型将在不久后发布。 

---
# $\text{Transformer}^2$: Self-adaptive LLMs 

**Title (ZH)**: $\text{Transformer}^2$: 自适应 LARGE LANGUAGE MODELS（或：$\text{Transformer}^2$: 自适应大语言模型） 

**Authors**: Qi Sun, Edoardo Cetin, Yujin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06252)  

**Abstract**: Self-adaptive large language models (LLMs) aim to solve the challenges posed by traditional fine-tuning methods, which are often computationally intensive and static in their ability to handle diverse tasks. We introduce \implname, a novel self-adaptation framework that adapts LLMs for unseen tasks in real-time by selectively adjusting only the singular components of their weight matrices. During inference, \implname employs a two-pass mechanism: first, a dispatch system identifies the task properties, and then task-specific "expert" vectors, trained using reinforcement learning, are dynamically mixed to obtain targeted behavior for the incoming prompt. Our method outperforms ubiquitous approaches such as LoRA, with fewer parameters and greater efficiency. \implname demonstrates versatility across different LLM architectures and modalities, including vision-language tasks. \implname represents a significant leap forward, offering a scalable, efficient solution for enhancing the adaptability and task-specific performance of LLMs, paving the way for truly dynamic, self-organizing AI systems. 

**Abstract (ZH)**: 自适应大型语言模型（LLMs）旨在解决传统微调方法所带来的挑战，这些方法通常计算密集且在处理多样化任务方面具有静态特性。我们引入了\implname，这是一种新颖的自适应框架，能够在实时环境下根据需要仅选择性地调整权重矩阵的单一组件，从而适应未见过的任务。在推理过程中，\implname 采用两步机制：首先，调度系统确定任务属性；然后，使用强化学习训练的任务特定“专家”向量会动态混合，以针对输入提示获得特定行为。我们的方法在参数更少且更高效的前提下，超越了诸如LoRA等通用方法。\implname 在不同LLM架构和模态（包括视觉-语言任务）中展示了适应性和灵活性。\implname 代表着一个重要的进步，提供了一种可扩展且高效的解决方案，以增强LLMs的自适应性和任务特定性能，为真正动态和自组织的人工智能系统铺平了道路。 

---
# FLAME: Financial Large-Language Model Assessment and Metrics Evaluation 

**Title (ZH)**: FLAME：金融大型语言模型评估与指标评价 

**Authors**: Jiayu Guo, Yu Guo, Martha Li, Songtao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2501.06211)  

**Abstract**: LLMs have revolutionized NLP and demonstrated potential across diverse domains. More and more financial LLMs have been introduced for finance-specific tasks, yet comprehensively assessing their value is still challenging. In this paper, we introduce FLAME, a comprehensive financial LLMs evaluation system in Chinese, which includes two core evaluation benchmarks: FLAME-Cer and FLAME-Sce. FLAME-Cer covers 14 types of authoritative financial certifications, including CPA, CFA, and FRM, with a total of approximately 16,000 carefully selected questions. All questions have been manually reviewed to ensure accuracy and representativeness. FLAME-Sce consists of 10 primary core financial business scenarios, 21 secondary financial business scenarios, and a comprehensive evaluation set of nearly 100 tertiary financial application tasks. We evaluate 6 representative LLMs, including GPT-4o, GLM-4, ERNIE-4.0, Qwen2.5, XuanYuan3, and the latest Baichuan4-Finance, revealing Baichuan4-Finance excels other LLMs in most tasks. By establishing a comprehensive and professional evaluation system, FLAME facilitates the advancement of financial LLMs in Chinese contexts. Instructions for participating in the evaluation are available on GitHub: this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已经革新了自然语言处理（NLP），并在诸多领域展示了其潜力。越来越多的专用于金融任务的大规模语言模型被引入，但全面评估它们的价值仍然颇具挑战性。本文介绍了FLAME，这是一种全面的中文金融大规模语言模型评估系统，其中包括两个核心评估基准：FLAME-Cer和FLAME-Sce。FLAME-Cer涵盖了14种权威的金融认证类型，包括注册会计师（CPA）、特许金融分析师（CFA）和金融风险管理师（FRM），共计约16,000个精心挑选的问题。所有问题均经过人工审核，以确保准确性和代表性。FLAME-Sce包括10个主要核心金融业务场景、21个次要金融业务场景以及近100个三级金融应用任务的全面评价集。我们评估了6个代表性的LLM，包括GPT-4o、GLM-4、ERNIE-4.0、Qwen2.5、XuanYuan3以及最新的Baichuan4-Finance，结果显示Baichuan4-Finance在大多数任务中表现优异。通过建立一个全面且专业的评价系统，FLAME促进了中文背景下金融LLM的发展。有关评估指南可在GitHub上获取：[此链接]。

请注意，将原文中的链接替换成实际的GitHub链接地址。 

---
# Analyzing the Role of Context in Forecasting with Large Language Models 

**Title (ZH)**: 分析上下文在使用大规模语言模型进行预测中的作用 

**Authors**: Gerrit Mutschlechner, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2501.06496)  

**Abstract**: This study evaluates the forecasting performance of recent language models (LLMs) on binary forecasting questions. We first introduce a novel dataset of over 600 binary forecasting questions, augmented with related news articles and their concise question-related summaries. We then explore the impact of input prompts with varying level of context on forecasting performance. The results indicate that incorporating news articles significantly improves performance, while using few-shot examples leads to a decline in accuracy. We find that larger models consistently outperform smaller models, highlighting the potential of LLMs in enhancing automated forecasting. 

**Abstract (ZH)**: 本研究评估了近期语言模型（LLMs）在二元预测问题上的预测性能。我们首先引入了一个包含超过600个二元预测问题的新颖数据集，该数据集包含相关的新闻文章及其简洁的问题相关摘要。随后，我们探讨了不同水平上下文输入提示对预测性能的影响。结果表明，整合新闻文章显著提高了预测性能，而使用少量示例则导致准确率下降。我们发现，较大的模型持续优于较小的模型，这突显了LLMs在增强自动化预测方面的潜力。 

---
# Imagine while Reasoning in Space: Multimodal Visualization-of-Thought 

**Title (ZH)**: 在空间推理中的多模态可视化思维：Imagine while Reasoning in Space 

**Authors**: Chengzu Li, Wenshan Wu, Huanyu Zhang, Yan Xia, Shaoguang Mao, Li Dong, Ivan Vulić, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.07542)  

**Abstract**: Chain-of-Thought (CoT) prompting has proven highly effective for enhancing complex reasoning in Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs). Yet, it struggles in complex spatial reasoning tasks. Nonetheless, human cognition extends beyond language alone, enabling the remarkable capability to think in both words and images. Inspired by this mechanism, we propose a new reasoning paradigm, Multimodal Visualization-of-Thought (MVoT). It enables visual thinking in MLLMs by generating image visualizations of their reasoning traces. To ensure high-quality visualization, we introduce token discrepancy loss into autoregressive MLLMs. This innovation significantly improves both visual coherence and fidelity. We validate this approach through several dynamic spatial reasoning tasks. Experimental results reveal that MVoT demonstrates competitive performance across tasks. Moreover, it exhibits robust and reliable improvements in the most challenging scenarios where CoT fails. Ultimately, MVoT establishes new possibilities for complex reasoning tasks where visual thinking can effectively complement verbal reasoning. 

**Abstract (ZH)**: 链式推理（CoT）提示在增强大规模语言模型（LLMs）和多模态大规模语言模型（MLLMs）的复杂推理能力方面已被证明非常有效。然而，它在复杂的空间推理任务中表现不佳。尽管如此，人类认知不仅依赖语言，还能够同时进行语言和图像的思维活动。受这一机制的启发，我们提出了一种新的推理范式——多模态思维可视化的（MVoT）。通过生成图像化的推理痕迹，MVoT允许MLLMs进行图像思维。为了保证高质量的可视化效果，我们引入了标记差异损失（token discrepancy loss）到自回归MLLMs中。这一创新显著提高了图像的连贯性和准确性。通过多种动态空间推理任务的验证，实验结果表明，MVoT在多个任务中表现出竞争力。此外，在CoT失败的最具有挑战性的场景中，MVoT也表现出稳健可靠的改进。最终，MVoT为复杂推理任务开辟了新的可能性，在这些任务中，图像思维能够有效地补充言语推理。 

---
# Investigating Large Language Models in Inferring Personality Traits from User Conversations 

**Title (ZH)**: 探究大型语言模型在从用户对话中推断人格特质方面的应用 

**Authors**: Jianfeng Zhu, Ruoming Jin, Karin G. Coifman  

**Link**: [PDF](https://arxiv.org/pdf/2501.07532)  

**Abstract**: Large Language Models (LLMs) are demonstrating remarkable human like capabilities across diverse domains, including psychological assessment. This study evaluates whether LLMs, specifically GPT-4o and GPT-4o mini, can infer Big Five personality traits and generate Big Five Inventory-10 (BFI-10) item scores from user conversations under zero-shot prompting conditions. Our findings reveal that incorporating an intermediate step--prompting for BFI-10 item scores before calculating traits--enhances accuracy and aligns more closely with the gold standard than direct trait inference. This structured approach underscores the importance of leveraging psychological frameworks in improving predictive precision. Additionally, a group comparison based on depressive symptom presence revealed differential model performance. Participants were categorized into two groups: those experiencing at least one depressive symptom and those without symptoms. GPT-4o mini demonstrated heightened sensitivity to depression-related shifts in traits such as Neuroticism and Conscientiousness within the symptom-present group, whereas GPT-4o exhibited strengths in nuanced interpretation across groups. These findings underscore the potential of LLMs to analyze real-world psychological data effectively, offering a valuable foundation for interdisciplinary research at the intersection of artificial intelligence and psychology. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多个领域内展示了令人惊叹的人类化能力，包括心理评估。本研究评估了在零样本推理条件下，LLMs，特别是GPT-4o和GPT-4o mini，能否从用户对话中推断大五人格特质并生成大五人格量表-10（BFI-10）项目分数。研究发现，在直接推断特质之前增加一个中间步骤——先提示生成BFI-10项目分数——可提高准确性和与黄金标准的一致性。这种结构化的方法突显了利用心理学框架以提高预测精度的重要性。此外，基于抑郁症状存在的分组比较揭示了不同模型的表现差异。参与者被分为两组：至少存在一种抑郁症状的组和没有症状的组。GPT-4o mini 在症状存在组中对与抑郁相关的神经质和尽责性特质变化表现出更高的敏感性，而GPT-4o 则在不同组中展示了更细致的解释能力。这些 findings 强调了LLMs 在有效分析现实世界心理数据方面的潜力，为人工智能与心理学交叉领域的跨学科研究奠定了坚实的基础。 

---
# FinerWeb-10BT: Refining Web Data with LLM-Based Line-Level Filtering 

**Title (ZH)**: FinerWeb-10BT：基于LLM的行级过滤细化网络数据 

**Authors**: Erik Henriksson, Otto Tarkka, Filip Ginter  

**Link**: [PDF](https://arxiv.org/pdf/2501.07314)  

**Abstract**: Data quality is crucial for training Large Language Models (LLMs). Traditional heuristic filters often miss low-quality text or mistakenly remove valuable content. In this paper, we introduce an LLM-based line-level filtering method to enhance training data quality. We use GPT-4o mini to label a 20,000-document sample from FineWeb at the line level, allowing the model to create descriptive labels for low-quality lines. These labels are grouped into nine main categories, and we train a DeBERTa-v3 classifier to scale the filtering to a 10B-token subset of FineWeb. To test the impact of our filtering, we train GPT-2 models on both the original and the filtered datasets. The results show that models trained on the filtered data achieve higher accuracy on the HellaSwag benchmark and reach their performance targets faster, even with up to 25\% less data. This demonstrates that LLM-based line-level filtering can significantly improve data quality and training efficiency for LLMs. We release our quality-annotated dataset, FinerWeb-10BT, and the codebase to support further work in this area. 

**Abstract (ZH)**: 数据质量对于训练大型语言模型（LLMs）至关重要。传统的启发式过滤方法往往无法检测到低质量的文本，或者错误地删除有价值的内容。在这篇论文中，我们介绍了一种基于LLM的行级过滤方法，以提升训练数据的质量。我们使用GPT-4o mini对FineWeb数据集中2万个文档样本进行行级标注，允许模型为低质量行生成描述性标签。这些标签被归类为九个主要类别，并训练一个DeBERTa-v3分类器以将过滤扩展到FineWeb的100亿令牌子集。为了测试我们过滤方法的影响，我们在未过滤的数据集和经过过滤的数据集上分别训练了GPT-2模型。结果表明，使用经过过滤的数据集训练的模型在HellaSwag基准测试中获得了更高的准确率，并且即使数据量减少多达25%，也能更快地达到预期的性能目标。这表明基于LLM的行级过滤方法可以显著提高LLMs的数据质量和训练效率。我们发布了包含质量标注的数据集FinerWeb-10BT及其代码库，以支持该领域的进一步研究。 

---
# Hierarchical Divide-and-Conquer for Fine-Grained Alignment in LLM-Based Medical Evaluation 

**Title (ZH)**: 基于LLM的医疗评价中细粒度对齐的分级分而治之方法 

**Authors**: Shunfan Zheng, Xiechi Zhang, Gerard de Melo, Xiaoling Wang, Linlin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06741)  

**Abstract**: In the rapidly evolving landscape of large language models (LLMs) for medical applications, ensuring the reliability and accuracy of these models in clinical settings is paramount. Existing benchmarks often focus on fixed-format tasks like multiple-choice QA, which fail to capture the complexity of real-world clinical diagnostics. Moreover, traditional evaluation metrics and LLM-based evaluators struggle with misalignment, often providing oversimplified assessments that do not adequately reflect human judgment. To address these challenges, we introduce HDCEval, a Hierarchical Divide-and-Conquer Evaluation framework tailored for fine-grained alignment in medical evaluation. HDCEval is built on a set of fine-grained medical evaluation guidelines developed in collaboration with professional doctors, encompassing Patient Question Relevance, Medical Knowledge Correctness, and Expression. The framework decomposes complex evaluation tasks into specialized subtasks, each evaluated by expert models trained through Attribute-Driven Token Optimization (ADTO) on a meticulously curated preference dataset. This hierarchical approach ensures that each aspect of the evaluation is handled with expert precision, leading to a significant improvement in alignment with human evaluators. 

**Abstract (ZH)**: 在大型语言模型（LLMs）在医疗应用的快速演变背景下，确保这些模型在临床环境中的可靠性和准确性至关重要。现有的基准测试通常侧重于固定格式的任务，如多项选择型问答，这未能捕捉到实际临床诊断的复杂性。此外，传统的评估指标和基于LLM的评估工具经常面临偏差问题，往往提供过于简化的评估，未能充分反映人类的判断。为应对这些挑战，我们引入了HDCEval，这是一种针对医疗评估细分对齐的分层征服评估框架。HDCEval基于与专业医生合作开发的一套细粒度的医疗评估指南，涵盖患者问题相关性、医学知识正确性和表达等方面。该框架将复杂的评估任务分解为专门的子任务，每个子任务由通过特性驱动的标记优化（ADTO）在精心策划的偏好数据集上训练的专家模型进行评估。这种分层方法确保每个评估方面都能以专家级别的精确度进行处理，从而显著改善与人类评估者的对齐。 

---
# Fine-tuning Large Language Models for Improving Factuality in Legal Question Answering 

**Title (ZH)**: 针对提高法律问答事实准确性的大规模语言模型微调 

**Authors**: Yinghao Hu, Leilei Gan, Wenyi Xiao, Kun Kuang, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.06521)  

**Abstract**: Hallucination, or the generation of incorrect or fabricated information, remains a critical challenge in large language models (LLMs), particularly in high-stake domains such as legal question answering (QA). In order to mitigate the hallucination rate in legal QA, we first introduce a benchmark called LegalHalBench and three automatic metrics to evaluate the common hallucinations when LLMs answer legal questions. We then propose a hallucination mitigation method that integrates behavior cloning and a novel Hard Sample-aware Iterative Direct Preference Optimization (HIPO). We conduct extensive real-data experiments to validate the effectiveness of our approach. Our results demonstrate remarkable improvements in various metrics, including the newly proposed Non-Hallucinated Statute Rate, Statute Relevance Rate, Legal Claim Truthfulness, as well as traditional metrics such as METEOR, BERTScore, ROUGE-L, and win rates. 

**Abstract (ZH)**: 幻觉，即生成错误或虚构的信息，仍然是大型语言模型（LLMs）中的一个关键挑战，尤其是在法律问题回答（QA）等高风险领域。为了降低法律QA中的幻觉率，我们首先引入了一个基准称为LegalHalBench，并提出了三种自动指标来评估LLMs在回答法律问题时常见的幻觉。然后，我们提出了一种结合行为克隆和一种新颖的硬样本意识迭代直接偏好优化（HIPO）的幻觉缓解方法。我们进行了广泛的实证实验以验证我们方法的有效性。实验结果表明，我们在多个指标领域，包括新提出的无幻觉成文率、成文相关率、法律主张的真实性，以及传统的指标如METEOR、BERTScore、ROUGE-L和胜率等方面均取得了显著改进。 

---
# O1 Replication Journey -- Part 3: Inference-time Scaling for Medical Reasoning 

**Title (ZH)**: O1 复现之旅——第3部分：推理时的扩展性在医疗推理中的应用 

**Authors**: Zhongzhen Huang, Gui Geng, Shengyi Hua, Zhen Huang, Haoyang Zou, Shaoting Zhang, Pengfei Liu, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.06458)  

**Abstract**: Building upon our previous investigations of O1 replication (Part 1: Journey Learning [Qin et al., 2024] and Part 2: Distillation [Huang et al., 2024]), this work explores the potential of inference-time scaling in large language models (LLMs) for medical reasoning tasks, ranging from diagnostic decision-making to treatment planning. Through extensive experiments on medical benchmarks of varying complexity (MedQA, Medbullets, and JAMA Clinical Challenges), our investigation reveals several key insights: (1) Increasing inference time does lead to improved performance. With a modest training set of 500 samples, our model yields substantial performance improvements of 6%-11%. (2) Task complexity directly correlates with the required length of reasoning chains, confirming the necessity of extended thought processes for challenging problems. (3) The differential diagnoses generated by our model adhere to the principles of the hypothetico-deductive method, producing a list of potential conditions that may explain a patient's symptoms and systematically narrowing these possibilities by evaluating the evidence. These findings demonstrate the promising synergy between inference-time scaling and journey learning in advancing LLMs' real-world clinical reasoning capabilities. 

**Abstract (ZH)**: 基于我们之前对O1复制的研究（Part 1：旅程学习 [Qin et al., 2024] 和 Part 2：知识萃取 [Huang et al., 2024]），本研究探讨了大型语言模型（LLMs）在医疗推理任务中的推理时扩展潜力，涵盖从诊断决策到治疗计划的多个方面。通过在不同复杂度的医学基准数据集（MedQA、Medbullets 和 JAMA Clinical Challenges）上进行广泛实验，我们的研究表明以下几个关键见解：（1）增加推理时间确实可以提高性能。仅用500个样本的适度训练集，我们的模型在性能上获得了6%-11%的重大提升。（2）任务复杂性直接与所需的推理链长度相关，证实了对于复杂问题有必要进行更长的思考过程。（3）我们的模型生成的鉴别诊断方案遵循假设演绎方法的原则，产生一系列可能解释患者症状的条件，并通过评估证据系统地缩小这些可能性。这些发现展示了推理时扩展与旅程学习之间有前景的协同作用，有助于推动LLMs在临床推理中的实际应用能力。 

---
# AFRIDOC-MT: Document-level MT Corpus for African Languages 

**Title (ZH)**: AFRIDOC-MT：非洲语言文档级机器翻译语料库 

**Authors**: Jesujoba O. Alabi, Israel Abebe Azime, Miaoran Zhang, Cristina España-Bonet, Rachel Bawden, Dawei Zhu, David Ifeoluwa Adelani, Clement Oyeleke Odoje, Idris Akinade, Iffat Maab, Davis David, Shamsuddeen Hassan Muhammad, Neo Putini, David O. Ademuyiwa, Andrew Caines, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2501.06374)  

**Abstract**: This paper introduces AFRIDOC-MT, a document-level multi-parallel translation dataset covering English and five African languages: Amharic, Hausa, Swahili, Yorùbá, and Zulu. The dataset comprises 334 health and 271 information technology news documents, all human-translated from English to these languages. We conduct document-level translation benchmark experiments by evaluating neural machine translation (NMT) models and large language models (LLMs) for translations between English and these languages, at both the sentence and pseudo-document levels. These outputs are realigned to form complete documents for evaluation. Our results indicate that NLLB-200 achieved the best average performance among the standard NMT models, while GPT-4o outperformed general-purpose LLMs. Fine-tuning selected models led to substantial performance gains, but models trained on sentences struggled to generalize effectively to longer documents. Furthermore, our analysis reveals that some LLMs exhibit issues such as under-generation, repetition of words or phrases, and off-target translations, especially for African languages. 

**Abstract (ZH)**: 本文介绍了AFRIDOC-MT数据集，这是一个涵盖英语和五个非洲语言（阿姆哈拉语、豪萨语、斯瓦希里语、约鲁巴语和祖鲁语）的文档级多平行翻译数据集。该数据集包含334篇健康和271篇信息技术新闻文档，所有文档均为从英语翻译至这些语言的人工翻译。我们通过评估神经机器翻译（NMT）模型和大型语言模型（LLMs），在英语与这些语言之间的翻译进行了文档级基准实验，评估对象包括句子级和伪文档级翻译。这些输出被重新对齐，以形成完整的文档进行评估。结果显示，NLLB-200 在标准NMT模型中获得了最佳平均性能，而GPT-4o在通用LLM中表现更佳。对某些模型进行微调后取得了显著的性能提升，但以句子为单位训练的模型在处理更长文档时难以有效泛化。此外，我们的分析发现，部分LLM存在生成不足、重复词语或短语、目标外翻译等问题，尤其是在处理非洲语言时更为明显。 

---
# Large Language Models Share Representations of Latent Grammatical Concepts Across Typologically Diverse Languages 

**Title (ZH)**: 大型语言模型在类型学上不同的语言中共享潜在语法概念的表示 

**Authors**: Jannik Brinkmann, Chris Wendler, Christian Bartelt, Aaron Mueller  

**Link**: [PDF](https://arxiv.org/pdf/2501.06346)  

**Abstract**: Human bilinguals often use similar brain regions to process multiple languages, depending on when they learned their second language and their proficiency. In large language models (LLMs), how are multiple languages learned and encoded? In this work, we explore the extent to which LLMs share representations of morphosyntactic concepts such as grammatical number, gender, and tense across languages. We train sparse autoencoders on Llama-3-8B and Aya-23-8B, and demonstrate that abstract grammatical concepts are often encoded in feature directions shared across many languages. We use causal interventions to verify the multilingual nature of these representations; specifically, we show that ablating only multilingual features decreases classifier performance to near-chance across languages. We then use these features to precisely modify model behavior in a machine translation task; this demonstrates both the generality and selectivity of these feature's roles in the network. Our findings suggest that even models trained predominantly on English data can develop robust, cross-lingual abstractions of morphosyntactic concepts. 

**Abstract (ZH)**: 人类双母语者在处理多种语言时常常使用相似的大脑区域，这取决于他们学习第二语言的时间以及他们的熟练程度。在大规模语言模型（LLMs）中，是如何学习和编码多种语言的？在这项研究中，我们探讨了LLMs在多大程度上共享表示如语法数、性状和时态等形态语法概念。我们对Llama-3-8B和Aya-23-8B进行了稀疏自编码器训练，并发现抽象的语法概念经常以跨多语言共享的特征方向来编码。我们通过因果干预验证了这些表示的多语言性质；具体来说，我们展示了仅消除多语言特征会导致跨语言分类器性能下降至接近随机水平。随后，我们利用这些特征精确地修改了模型在机器翻译任务中的行为；这表明这些特征在模型中的作用具有广泛性和选择性。我们的研究发现表明，即使模型主要在英语数据上训练，也能够发展出稳健的跨语言形态语法概念抽象。 

---
# Audio-CoT: Exploring Chain-of-Thought Reasoning in Large Audio Language Model 

**Title (ZH)**: Audio-CoT：探索大规模音频语言模型中的链式推理能力 

**Authors**: Ziyang Ma, Zhuo Chen, Yuping Wang, Eng Siong Chng, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.07246)  

**Abstract**: Large Audio-Language Models (LALMs) have demonstrated remarkable performance in tasks involving audio perception and understanding, such as speech recognition and audio captioning. However, their reasoning capabilities - critical for solving complex real-world problems - remain underexplored. In this work, we conduct the first exploration into integrating Chain-of-Thought (CoT) reasoning into LALMs to enhance their reasoning ability across auditory modalities. We evaluate representative CoT methods, analyzing their performance in both information extraction and reasoning tasks across sound, music, and speech domains. Our findings reveal that CoT methods significantly improve performance on easy and medium tasks but encounter challenges with hard tasks, where reasoning chains can confuse the model rather than improve accuracy. Additionally, we identify a positive correlation between reasoning path length and accuracy, demonstrating the potential of scaling inference for advanced instruction-following and reasoning. This study not only highlights the promise of CoT in enhancing LALM reasoning capabilities but also identifies key limitations and provides actionable directions for future research. 

**Abstract (ZH)**: 大型音频语言模型（LALMs）在语音识别和音频描述等涉及音频感知和理解的任务中表现出了卓越的能力。然而，它们的推理能力——对于解决复杂的现实世界问题至关重要——仍然未被充分探索。在本文中，我们首次探索将推理链（Chain-of-Thought, CoT）机制整合到LALMs中，以增强其跨听觉模态的推理能力。我们评估了代表性CoT方法，并分析了它们在声音、音乐和语音领域中信息提取和推理任务中的表现。研究结果表明，CoT方法显著提高了简单和中等难度任务的表现，但在处理复杂任务时遇到了挑战，在这些任务中，推理链可能会使模型陷入混乱，而不是提高准确性。此外，我们发现推理路径长度和准确性之间存在正相关关系，这表明扩展推理对于高级指令遵循和推理具有潜在价值。这项研究不仅突显了CoT在增强LALM推理能力方面的潜力，还指出了其关键限制，并为未来的研究所提供了可操作的方向。 

---
# EmoXpt: Analyzing Emotional Variances in Human Comments and LLM-Generated Responses 

**Title (ZH)**: EmoXpt：分析人类评论与大语言模型生成回应中的情感波动 

**Authors**: Shireesh Reddy Pyreddy, Tarannum Shaila Zaman  

**Link**: [PDF](https://arxiv.org/pdf/2501.06597)  

**Abstract**: The widespread adoption of generative AI has generated diverse opinions, with individuals expressing both support and criticism of its applications. This study investigates the emotional dynamics surrounding generative AI by analyzing human tweets referencing terms such as ChatGPT, OpenAI, Copilot, and LLMs. To further understand the emotional intelligence of ChatGPT, we examine its responses to selected tweets, highlighting differences in sentiment between human comments and LLM-generated responses. We introduce EmoXpt, a sentiment analysis framework designed to assess both human perspectives on generative AI and the sentiment embedded in ChatGPT's responses. Unlike prior studies that focus exclusively on human sentiment, EmoXpt uniquely evaluates the emotional expression of ChatGPT. Experimental results demonstrate that LLM-generated responses are notably more efficient, cohesive, and consistently positive than human responses. 

**Abstract (ZH)**: 生成式AI的广泛应用引起了社会各界的不同看法，个人既支持也有批评其应用。本文通过分析提及ChatGPT、OpenAI、Copilot和大规模语言模型（LLMs）的人类推文，研究生成式AI周围的情感动态。为进一步了解ChatGPT的情感智能，我们对其对选定推文的回应进行了研究，突显了人类评论与生成式AI回应之间情感差异。本文引入了EmoXpt情感分析框架，旨在评估人类对生成式AI的观点以及ChatGPT回应中嵌入的情感。不同于以往主要集中在人类情感的研究，EmoXpt独特地评估了ChatGPT的情感表达。实验结果表明，生成式AI生成的回应在效率、连贯性和一致性方面明显优于人类回应。 

---
# Using Pre-trained LLMs for Multivariate Time Series Forecasting 

**Title (ZH)**: 使用预训练的大语言模型进行多变量时间序列预测 

**Authors**: Malcolm L. Wolff, Shenghao Yang, Kari Torkkola, Michael W. Mahoney  

**Link**: [PDF](https://arxiv.org/pdf/2501.06386)  

**Abstract**: Pre-trained Large Language Models (LLMs) encapsulate large amounts of knowledge and take enormous amounts of compute to train. We make use of this resource, together with the observation that LLMs are able to transfer knowledge and performance from one domain or even modality to another seemingly-unrelated area, to help with multivariate demand time series forecasting. Attention in transformer-based methods requires something worth attending to -- more than just samples of a time-series. We explore different methods to map multivariate input time series into the LLM token embedding space. In particular, our novel multivariate patching strategy to embed time series features into decoder-only pre-trained Transformers produces results competitive with state-of-the-art time series forecasting models. We also use recently-developed weight-based diagnostics to validate our findings. 

**Abstract (ZH)**: 预训练大规模语言模型（LLMs）蕴含了大量知识，并需要巨大的计算资源来训练。我们利用这一资源，并结合LLMs具备在不同领域或模态之间转移知识和性能的能力，来帮助进行多变量需求时间序列预测。基于变换器的方法中的注意力机制需要关注有意义的内容——而不仅仅是时间序列样本。我们探索了不同的方法，将多变量输入时间序列映射到LLM标记嵌入空间。特别是，我们提出的新颖的多变量片断化策略，将时间序列特征嵌入仅解码器预训练变换器中，产生的结果与最新时间序列预测模型相媲美。我们还使用了最近开发的基于权重的诊断方法来验证我们的发现。 

---
# PROEMO: Prompt-Driven Text-to-Speech Synthesis Based on Emotion and Intensity Control 

**Title (ZH)**: PROEMO：基于情感和强度控制的提示驱动文本到语音合成 

**Authors**: Shaozuo Zhang, Ambuj Mehrish, Yingting Li, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2501.06276)  

**Abstract**: Speech synthesis has significantly advanced from statistical methods to deep neural network architectures, leading to various text-to-speech (TTS) models that closely mimic human speech patterns. However, capturing nuances such as emotion and style in speech synthesis is challenging. To address this challenge, we introduce an approach centered on prompt-based emotion control. The proposed architecture incorporates emotion and intensity control across multi-speakers. Furthermore, we leverage large language models (LLMs) to manipulate speech prosody while preserving linguistic content. Using embedding emotional cues, regulating intensity levels, and guiding prosodic variations with prompts, our approach infuses synthesized speech with human-like expressiveness and variability. Lastly, we demonstrate the effectiveness of our approach through a systematic exploration of the control mechanisms mentioned above. 

**Abstract (ZH)**: 语音合成从统计方法显著发展到了深层神经网络架构，产生了各种能够密切模仿人类语音模式的文本到语音（TTS）模型。然而，在语音合成中捕捉情绪和风格的细微差别是具有挑战性的。为了解决这一挑战，我们提出了基于提示的情绪控制方法。所提出的架构在多说话人中实现了情绪和强度控制。此外，我们利用大规模语言模型（LLMs）来操控言语韵律，同时保留语言内容。通过嵌入情绪线索、调节强度级别，并用提示引导韵律变化，我们的方法赋予合成语音以类似人类的表达性和变异性。最后，我们通过系统性地探索上述控制机制的有效性来验证我们的方法。 

---
