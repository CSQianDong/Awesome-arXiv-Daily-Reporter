# MAPoRL: Multi-Agent Post-Co-Training for Collaborative Large Language Models with Reinforcement Learning 

**Title (ZH)**: MAPoRL：基于强化学习的多智能体共训练协作大语言模型 

**Authors**: Chanwoo Park, Seungju Han, Xingzhi Guo, Asuman Ozdaglar, Kaiqing Zhang, Joo-Kyung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.18439)  

**Abstract**: Leveraging multiple large language models (LLMs) to build collaborative multi-agentic workflows has demonstrated significant potential. However, most previous studies focus on prompting the out-of-the-box LLMs, relying on their innate capability for collaboration, which may not improve LLMs' performance as shown recently. In this paper, we introduce a new post-training paradigm MAPoRL (Multi-Agent Post-co-training for collaborative LLMs with Reinforcement Learning), to explicitly elicit the collaborative behaviors and further unleash the power of multi-agentic LLM frameworks. In MAPoRL, multiple LLMs first generate their own responses independently and engage in a multi-turn discussion to collaboratively improve the final answer. In the end, a MAPoRL verifier evaluates both the answer and the discussion, by assigning a score that verifies the correctness of the answer, while adding incentives to encourage corrective and persuasive discussions. The score serves as the co-training reward, and is then maximized through multi-agent RL. Unlike existing LLM post-training paradigms, MAPoRL advocates the co-training of multiple LLMs together using RL for better generalization. Accompanied by analytical insights, our experiments demonstrate that training individual LLMs alone is insufficient to induce effective collaboration. In contrast, multi-agent co-training can boost the collaboration performance across benchmarks, with generalization to unseen domains. 

**Abstract (ZH)**: 利用多个大规模语言模型（LLMs）构建协作多智能体工作流展现出显著潜力。然而，大多数前期研究集中在通过提示现成的LLMs来激发其协作能力上，依赖它们固有的协作能力，这可能未如最近研究所示那样提高LLMs的表现。本文介绍了一种新的后训练范式MAPoRL（Multi-Agent Post-co-training for collaborative LLMs with Reinforcement Learning），以明确地引发协作行为，并进一步释放多智能体LLM框架的能力。在MAPoRL中，多个LLMs首先独立生成自己的回应，并通过多轮讨论协作改进最终答案。最后，一个MAPoRL验证器评估答案和讨论，通过分配证明答案正确性的分数，同时鼓励正确的和有说服力的讨论，从而激励讨论。该分数作为共训练奖励，并通过多智能体强化学习进行最大化。与现有的LLM后训练范式不同，MAPoRL提倡使用强化学习共同训练多个LLMs以提高泛化能力。我们的实验配以分析洞察，证明单独训练单个LLM不足以引发有效的协作。相反，多智能体共同训练可以在基准测试上提升协作性能，并扩展到未见过的领域。 

---
# PyEvalAI: AI-assisted evaluation of Jupyter Notebooks for immediate personalized feedback 

**Title (ZH)**: PyEvalAI：AI 辅助的 Jupyter 笔记本评估以提供即时个性化反馈 

**Authors**: Nils Wandel, David Stotko, Alexander Schier, Reinhard Klein  

**Link**: [PDF](https://arxiv.org/pdf/2502.18425)  

**Abstract**: Grading student assignments in STEM courses is a laborious and repetitive task for tutors, often requiring a week to assess an entire class. For students, this delay of feedback prevents iterating on incorrect solutions, hampers learning, and increases stress when exercise scores determine admission to the final exam. Recent advances in AI-assisted education, such as automated grading and tutoring systems, aim to address these challenges by providing immediate feedback and reducing grading workload. However, existing solutions often fall short due to privacy concerns, reliance on proprietary closed-source models, lack of support for combining Markdown, LaTeX and Python code, or excluding course tutors from the grading process. To overcome these limitations, we introduce PyEvalAI, an AI-assisted evaluation system, which automatically scores Jupyter notebooks using a combination of unit tests and a locally hosted language model to preserve privacy. Our approach is free, open-source, and ensures tutors maintain full control over the grading process. A case study demonstrates its effectiveness in improving feedback speed and grading efficiency for exercises in a university-level course on numerics. 

**Abstract (ZH)**: 在STEM课程中给学生作业打分是一项繁琐且重复的工作，导师通常需要花费一周时间才能评估完一个班级的所有作业。对于学生而言，这种延迟的反馈会导致无法对错误的答案进行迭代，阻碍学习进程，并在成绩决定最终考试录取与否的情况下增加学习压力。近期在AI辅助教育方面的进展，如自动化评分和辅导系统，旨在通过提供即时反馈并减少评分工作量来应对这些挑战。然而，现有的解决方案往往由于隐私问题、依赖于私有闭源模型、不支持结合Markdown、LaTeX和Python代码，或者不包括课程导师在内的评分过程等原因而存在不足。为克服这些限制，我们引入了PyEvalAI，一个基于AI辅助的评估系统，它使用单元测试和本地托管的语言模型来自动化评分，从而保护隐私。我们的方法是免费且开源的，确保导师能完全控制评分过程。案例研究证明了该系统的有效性，它在一所大学的数值课程作业中提高了反馈速度和评分效率。 

---
# The Gradient of Algebraic Model Counting 

**Title (ZH)**: 代数模型计数的梯度 

**Authors**: Jaron Maene, Luc De Raedt  

**Link**: [PDF](https://arxiv.org/pdf/2502.18406)  

**Abstract**: Algebraic model counting unifies many inference tasks on logic formulas by exploiting semirings. Rather than focusing on inference, we consider learning, especially in statistical-relational and neurosymbolic AI, which combine logical, probabilistic and neural representations. Concretely, we show that the very same semiring perspective of algebraic model counting also applies to learning. This allows us to unify various learning algorithms by generalizing gradients and backpropagation to different semirings. Furthermore, we show how cancellation and ordering properties of a semiring can be exploited for more memory-efficient backpropagation. This allows us to obtain some interesting variations of state-of-the-art gradient-based optimisation methods for probabilistic logical models. We also discuss why algebraic model counting on tractable circuits does not lead to more efficient second-order optimization. Empirically, our algebraic backpropagation exhibits considerable speed-ups as compared to existing approaches. 

**Abstract (ZH)**: 代数模型计数通过利用半环统合了逻辑公式上的许多推理任务。与其专注于推理，我们考虑学习，特别是在统计关系AI和神经符号AI中，这些领域结合了逻辑的、概率的和神经的表现形式。具体而言，我们展示了代数模型计数的相同半环视角同样适用于学习。这使得我们能够通过将梯度和反向传播泛化到不同的半环来统合各种学习算法。此外，我们展示了如何利用半环的取消和排序性质，以实现更内存高效的反向传播。这使我们能够获得一些基于梯度的优化方法的有趣的变体，这些方法适用于概率逻辑模型。我们还讨论了为什么在可计算电路上的代数模型计数不会导致更高效的二次优化。实验上，与现有方法相比，我们的代数反向传播显示出显著的加速效果。 

---
# How Far are LLMs from Real Search? A Comprehensive Study on Efficiency, Completeness, and Inherent Capabilities 

**Title (ZH)**: 大语言模型与实际搜索有多大差距？关于效率、完整性和固有能力的全面研究 

**Authors**: Minhua Lin, Hui Liu, Xianfeng Tang, Jingying Zeng, Zhenwei Dai, Chen Luo, Zheng Li, Xiang Zhang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18387)  

**Abstract**: Search plays a fundamental role in problem-solving across various domains, with most real-world decision-making problems being solvable through systematic search. Drawing inspiration from recent discussions on search and learning, we systematically explore the complementary relationship between search and Large Language Models (LLMs) from three perspectives. First, we analyze how learning can enhance search efficiency and propose Search via Learning (SeaL), a framework that leverages LLMs for effective and efficient search. Second, we further extend SeaL to SeaL-C to ensure rigorous completeness during search. Our evaluation across three real-world planning tasks demonstrates that SeaL achieves near-perfect accuracy while reducing search spaces by up to 99.1% compared to traditional approaches. Finally, we explore how far LLMs are from real search by investigating whether they can develop search capabilities independently. Our analysis reveals that while current LLMs struggle with efficient search in complex problems, incorporating systematic search strategies significantly enhances their problem-solving capabilities. These findings not only validate the effectiveness of our approach but also highlight the need for improving LLMs' search abilities for real-world applications. 

**Abstract (ZH)**: 搜索在各个领域的问题解决中扮演着基础角色，大多数现实世界中的决策问题都可以通过系统的搜索方法来解决。受到近期关于搜索与学习讨论的启发，我们从三个方面系统地探讨了搜索和大型语言模型（LLMs）之间的互补关系。首先，我们分析了学习如何提高搜索效率，并提出了一种名为“搜索通过学习”（SeaL）的框架，该框架利用LLMs进行有效的搜索。其次，我们将SeaL扩展为SeaL-C，以确保在搜索过程中实现严格的完备性。我们在三个实际规划任务上的评估显示，与传统方法相比，SeaL能够将搜索空间减少高达99.1%，同时保持接近完美的准确性。最后，我们探索了LLMs在现实搜索中的局限性，研究它们是否能够独立发展出搜索能力。我们的分析表明，尽管当前的LLMs在解决复杂问题时难以进行高效的搜索，但融入系统的搜索策略极大地增强了它们的问题解决能力。这些发现不仅验证了我们方法的有效性，还突显了改善LLMs的搜索能力以适应实际应用的必要性。 

---
# MindMem: Multimodal for Predicting Advertisement Memorability Using LLMs and Deep Learning 

**Title (ZH)**: MindMem：结合大规模语言模型和深度学习的多模态广告记忆性预测方法 

**Authors**: Sepehr Asgarian, Qayam Jetha, Jouhyun Jeon  

**Link**: [PDF](https://arxiv.org/pdf/2502.18371)  

**Abstract**: In the competitive landscape of advertising, success hinges on effectively navigating and leveraging complex interactions among consumers, advertisers, and advertisement platforms. These multifaceted interactions compel advertisers to optimize strategies for modeling consumer behavior, enhancing brand recall, and tailoring advertisement content. To address these challenges, we present MindMem, a multimodal predictive model for advertisement memorability. By integrating textual, visual, and auditory data, MindMem achieves state-of-the-art performance, with a Spearman's correlation coefficient of 0.631 on the LAMBDA and 0.731 on the Memento10K dataset, consistently surpassing existing methods. Furthermore, our analysis identified key factors influencing advertisement memorability, such as video pacing, scene complexity, and emotional resonance. Expanding on this, we introduced MindMem-ReAd (MindMem-Driven Re-generated Advertisement), which employs Large Language Model-based simulations to optimize advertisement content and placement, resulting in up to a 74.12% improvement in advertisement memorability. Our results highlight the transformative potential of Artificial Intelligence in advertising, offering advertisers a robust tool to drive engagement, enhance competitiveness, and maximize impact in a rapidly evolving market. 

**Abstract (ZH)**: 在广告竞争激烈的环境中，成功取决于有效导航和利用消费者、广告商和广告平台之间的复杂互动。这些多维度的互动促使广告商优化策略以建模消费者行为、增强品牌记忆度以及定制广告内容。为应对这些挑战，我们提出了MindMem，这是一种多模态预测模型，用于衡量广告的记忆度。MindMem 结合了文本、视觉和音频数据，实现了行业领先的表现，在 LAMBDA 数据集上的 Spearman 相关系数为 0.631，在 Memento10K 数据集上的相关系数为 0.731，始终优于现有方法。此外，我们的分析揭示了影响广告记忆度的关键因素，如视频节奏、场景复杂性和情感共鸣。在此基础上，我们引入了MindMem-ReAd（MindMem 驱动的重复生成广告），它利用基于大型语言模型的模拟优化广告内容和位置，从而在广告记忆度上实现了高达 74.12% 的提升。我们的研究结果突显了人工智能在广告中的变革潜力，为广告商提供了一个强大的工具，以提高互动性、增强竞争力并在快速变化的市场中最大化影响。 

---
# GraphRank Pro+: Advancing Talent Analytics Through Knowledge Graphs and Sentiment-Enhanced Skill Profiling 

**Title (ZH)**: GraphRank Pro+：通过知识图谱和情感增强的职业技能画像推进人才分析 

**Authors**: Sirisha Velampalli, Chandrashekar Muniyappa  

**Link**: [PDF](https://arxiv.org/pdf/2502.18315)  

**Abstract**: The extraction of information from semi-structured text, such as resumes, has long been a challenge due to the diverse formatting styles and subjective content organization. Conventional solutions rely on specialized logic tailored for specific use cases. However, we propose a revolutionary approach leveraging structured Graphs, Natural Language Processing (NLP), and Deep Learning. By abstracting intricate logic into Graph structures, we transform raw data into a comprehensive Knowledge Graph. This innovative framework enables precise information extraction and sophisticated querying. We systematically construct dictionaries assigning skill weights, paving the way for nuanced talent analysis. Our system not only benefits job recruiters and curriculum designers but also empowers job seekers with targeted query-based filtering and ranking capabilities. 

**Abstract (ZH)**: 从半结构化文本中提取信息长期以来一直是挑战，因为这些文本的格式多样且内容组织主观性强。传统的解决方法依赖于为特定应用场景量身定制的逻辑。然而，我们提出了一种革命性的方法，利用结构化图、自然语言处理（NLP）和深度学习技术。通过将复杂的逻辑抽象为图结构，我们将原始数据转化为全面的知识图谱。这种创新框架能够实现精确的信息提取和高级查询。我们系统地构建字典来分配技能权重，为精细的人才分析铺平道路。我们的系统不仅为招聘人员和课程设计师带来了好处，还为求职者提供了基于查询的过滤和排序功能，帮助他们更好地匹配合适的工作。 

---
# Citrus: Leveraging Expert Cognitive Pathways in a Medical Language Model for Advanced Medical Decision Support 

**Title (ZH)**: 柑橘：利用专家认知路径增强医疗语言模型的高级医疗决策支持 

**Authors**: Guoxin Wang, Minyu Gao, Shuai Yang, Ya Zhang, Lizhi He, Liang Huang, Hanlin Xiao, Yexuan Zhang, Wanyue Li, Lu Chen, Jintao Fei, Xin Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18274)  

**Abstract**: Large language models (LLMs), particularly those with reasoning capabilities, have rapidly advanced in recent years, demonstrating significant potential across a wide range of applications. However, their deployment in healthcare, especially in disease reasoning tasks, is hindered by the challenge of acquiring expert-level cognitive data. In this paper, we introduce Citrus, a medical language model that bridges the gap between clinical expertise and AI reasoning by emulating the cognitive processes of medical experts. The model is trained on a large corpus of simulated expert disease reasoning data, synthesized using a novel approach that accurately captures the decision-making pathways of clinicians. This approach enables Citrus to better simulate the complex reasoning processes involved in diagnosing and treating medical this http URL further address the lack of publicly available datasets for medical reasoning tasks, we release the last-stage training data, including a custom-built medical diagnostic dialogue dataset. This open-source contribution aims to support further research and development in the field. Evaluations using authoritative benchmarks such as MedQA, covering tasks in medical reasoning and language understanding, show that Citrus achieves superior performance compared to other models of similar size. These results highlight Citrus potential to significantly enhance medical decision support systems, providing a more accurate and efficient tool for clinical decision-making. 

**Abstract (ZH)**: 近年来，尤其是具备推理能力的大规模语言模型（LLMs）取得了迅速发展，展示了在各种应用领域中的巨大潜力。然而，这些模型在医疗领域的部署，尤其是在疾病推理任务中的应用，受到了获取专家级认知数据的挑战。本文介绍了Citrus，这是一种医疗语言模型，通过模拟医学专家的认知过程，弥合了临床专业知识与人工智能推理之间的差距。该模型是基于大量模拟的专家级疾病推理数据集进行训练的，这些数据是通过一种新颖的方法合成的，该方法能够准确捕捉临床决策的路径。这种方法使Citrus能够更好地模拟诊断和治疗过程中的复杂推理机制。

为进一步解决医学推理任务中缺乏公共数据集的问题，我们公开了模型的最后一阶段训练数据，包括一个自定义构建的医疗诊断对话数据集。这一开放源代码的贡献旨在支持该领域进一步的研究和开发。使用MedQA等权威基准进行的评估涵盖了医学推理和语言理解的任务，结果显示，Citrus在性能上优于其他同类型规模的模型。这些结果突显了Citrus在增强医疗决策支持系统方面的巨大潜力，可以为临床决策提供更加准确和高效的工具。 

---
# ChatMotion: A Multimodal Multi-Agent for Human Motion Analysis 

**Title (ZH)**: ChatMotion：一种多模态多智能体系统用于人类动作分析 

**Authors**: Li Lei, Jia Sen, Wang Jianhao, An Zhaochong, Li Jiaang, Hwang Jenq-Neng, Belongie Serge  

**Link**: [PDF](https://arxiv.org/pdf/2502.18180)  

**Abstract**: Advancements in Multimodal Large Language Models (MLLMs) have improved human motion understanding. However, these models remain constrained by their "instruct-only" nature, lacking interactivity and adaptability for diverse analytical perspectives. To address these challenges, we introduce ChatMotion, a multimodal multi-agent framework for human motion analysis. ChatMotion dynamically interprets user intent, decomposes complex tasks into meta-tasks, and activates specialized function modules for motion comprehension. It integrates multiple specialized modules, such as the MotionCore, to analyze human motion from various perspectives. Extensive experiments demonstrate ChatMotion's precision, adaptability, and user engagement for human motion understanding. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）的进步提高了对人类运动的理解。然而，这些模型仍受限于其“仅指令”性质，缺乏交互性和适应性以应对多样化的分析视角。为解决这些挑战，我们提出了一种名为ChatMotion的多模态多智能体框架，用于人类运动分析。ChatMotion动态解读用户意图，将复杂的任务分解为元任务，并激活专门的功能模块以进行运动理解。该框架整合了多个专业化模块，如MotionCore，从多个角度分析人类运动。广泛的经验表明，ChatMotion在人类运动理解方面具有精确性、适应性和用户参与度。 

---
# Defining bias in AI-systems: Biased models are fair models 

**Title (ZH)**: 定义AI系统中的偏见：有偏见的模型即是公正的模型 

**Authors**: Chiara Lindloff, Ingo Siegert  

**Link**: [PDF](https://arxiv.org/pdf/2502.18060)  

**Abstract**: The debate around bias in AI systems is central to discussions on algorithmic fairness. However, the term bias often lacks a clear definition, despite frequently being contrasted with fairness, implying that an unbiased model is inherently fair. In this paper, we challenge this assumption and argue that a precise conceptualization of bias is necessary to effectively address fairness concerns. Rather than viewing bias as inherently negative or unfair, we highlight the importance of distinguishing between bias and discrimination. We further explore how this shift in focus can foster a more constructive discourse within academic debates on fairness in AI systems. 

**Abstract (ZH)**: 关于AI系统中的偏见的辩论是算法公平性讨论的核心。然而，尽管偏见一词经常与公平性进行对比，从而暗示无偏见的模型就是公平的，但该术语往往缺乏明确的定义。在本文中，我们对这种假设提出了质疑，并认为为了有效解决公平性问题，需要对偏见进行精确的概念化。我们不将偏见视为本质上是负面的或不公平的，而是强调区分偏见和歧视的重要性。此外，我们探讨了这种关注焦点的转变如何促进学术界关于AI系统公平性讨论中的更富有建设性的对话。 

---
# GNN-XAR: A Graph Neural Network for Explainable Activity Recognition in Smart Homes 

**Title (ZH)**: GNN-XAR：一种用于智能家居中可解释活动识别的图神经网络 

**Authors**: Michele Fiori, Davide Mor, Gabriele Civitarese, Claudio Bettini  

**Link**: [PDF](https://arxiv.org/pdf/2502.17999)  

**Abstract**: Sensor-based Human Activity Recognition (HAR) in smart home environments is crucial for several applications, especially in the healthcare domain. The majority of the existing approaches leverage deep learning models. While these approaches are effective, the rationale behind their outputs is opaque. Recently, eXplainable Artificial Intelligence (XAI) approaches emerged to provide intuitive explanations to the output of HAR models. To the best of our knowledge, these approaches leverage classic deep models like CNNs or RNNs. Recently, Graph Neural Networks (GNNs) proved to be effective for sensor-based HAR. However, existing approaches are not designed with explainability in mind. In this work, we propose the first explainable Graph Neural Network explicitly designed for smart home HAR. Our results on two public datasets show that this approach provides better explanations than state-of-the-art methods while also slightly improving the recognition rate. 

**Abstract (ZH)**: 智能家居环境中基于传感器的人体活动识别（HAR）对于多个应用至关重要，尤其是在健康护理领域。现有的大多数方法采用深度学习模型。虽然这些方法有效，但其输出背后的原理却缺乏透明性。最近，可解释的人工智能（XAI）方法开始涌现，以提供对HAR模型输出的直观解释。据我们所知，这些方法主要依赖于传统的深度模型，如卷积神经网络（CNNs）或循环神经网络（RNNs）。最近的研究表明，图形神经网络（GNNs）在传感器数据的人体活动识别方面也非常有效。然而，现有方法并未旨在提高模型的可解释性。在本研究中，我们提出了一种专门为智能家居环境下的HAR设计的第一种可解释的图形神经网络。在两个公开数据集上的实验结果表明，与现有最佳方法相比，该方法不仅能提供更好的解释，还能稍微提高识别率。 

---
# LeanProgress: Guiding Search for Neural Theorem Proving via Proof Progress Prediction 

**Title (ZH)**: LeanProgress：通过证明进展预测指导神经定理证明的搜索 

**Authors**: Suozhi Huang, Peiyang Song, Robert Joseph George, Anima Anandkumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.17925)  

**Abstract**: Mathematical reasoning remains a significant challenge for Large Language Models (LLMs) due to hallucinations. When combined with formal proof assistants like Lean, these hallucinations can be eliminated through rigorous verification, making theorem proving reliable. However, even with formal verification, LLMs still struggle with long proofs and complex mathematical formalizations. While Lean with LLMs offers valuable assistance with retrieving lemmas, generating tactics, or even complete proofs, it lacks a crucial capability: providing a sense of proof progress. This limitation particularly impacts the overall development efficiency in large formalization projects. We introduce LeanProgress, a method that predicts the progress in the proof. Training and evaluating our models made on a large corpus of Lean proofs from Lean Workbook Plus and Mathlib4 and how many steps remain to complete it, we employ data preprocessing and balancing techniques to handle the skewed distribution of proof lengths. Our experiments show that LeanProgress achieves an overall prediction accuracy of 75.1\% in predicting the amount of progress and, hence, the remaining number of steps. When integrated into a best-first search framework using Reprover, our method shows a 3.8\% improvement on Mathlib4 compared to baseline performances of 41.2\%, particularly for longer proofs. These results demonstrate how proof progress prediction can enhance both automated and interactive theorem proving, enabling users to make more informed decisions about proof strategies. 

**Abstract (ZH)**: 数学推理仍然是大型语言模型（LLMs）的一个重大挑战，尤其是由于它们可能会产生幻觉。当与形式化证明助手如Lean结合使用时，这些幻觉可以通过严格的验证消除，从而使定理证明变得可靠。然而，即使有形式验证，LLMs仍然难以处理长证明和复杂的数学形式化。虽然结合LLMs的Lean可以为检索引理、生成策略或甚至完整的证明提供有价值的帮助，但它缺乏一个关键的功能：提供证明进程感。这个限制尤其影响大型形式化项目的整体开发效率。我们引入了LeanProgress，这是一种预测证明进程的方法。我们通过对Lean Workbook Plus和Mathlib4中的大量Lean证明进行训练和评估，并预测剩余需要完成的步骤数量，我们采用了数据预处理和平衡技术来处理证明长度分布不均的问题。我们的实验表明，LeanProgress在预测证明进程量方面总体准确率达到75.1%，因此预测剩余的步骤数量。当将我们的方法集成到使用Reprover的优先级搜索框架中时，它在Mathlib4上比基线性能（41.2%）提高了3.8%，特别是在处理较长证明时。这些结果表明，证明进程预测如何可以提升自动和交互式定理证明，使用户能够更加明智地选择证明策略。 

---
# Unmasking Gender Bias in Recommendation Systems and Enhancing Category-Aware Fairness 

**Title (ZH)**: 揭示推荐系统中的性别偏见并增强类别感知的公平性 

**Authors**: Tahsin Alamgir Kheya, Mohamed Reda Bouadjenek, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2502.17921)  

**Abstract**: Recommendation systems are now an integral part of our daily lives. We rely on them for tasks such as discovering new movies, finding friends on social media, and connecting job seekers with relevant opportunities. Given their vital role, we must ensure these recommendations are free from societal stereotypes. Therefore, evaluating and addressing such biases in recommendation systems is crucial. Previous work evaluating the fairness of recommended items fails to capture certain nuances as they mainly focus on comparing performance metrics for different sensitive groups. In this paper, we introduce a set of comprehensive metrics for quantifying gender bias in recommendations. Specifically, we show the importance of evaluating fairness on a more granular level, which can be achieved using our metrics to capture gender bias using categories of recommended items like genres for movies. Furthermore, we show that employing a category-aware fairness metric as a regularization term along with the main recommendation loss during training can help effectively minimize bias in the models' output. We experiment on three real-world datasets, using five baseline models alongside two popular fairness-aware models, to show the effectiveness of our metrics in evaluating gender bias. Our metrics help provide an enhanced insight into bias in recommended items compared to previous metrics. Additionally, our results demonstrate how incorporating our regularization term significantly improves the fairness in recommendations for different categories without substantial degradation in overall recommendation performance. 

**Abstract (ZH)**: 推荐系统已经成为我们日常生活中不可或缺的一部分。我们依赖它们来发现新的电影、在社交媒体上寻找朋友以及为求职者匹配相关机会。鉴于它们的重要性，我们必须确保推荐结果不带有社会偏见。因此，评估并解决推荐系统中的这些偏见至关重要。目前评价推荐公平性的研究主要集中在比较不同敏感群体的性能指标，未能捕捉到某些微妙之处。在本文中，我们提出了一套全面的度量标准，用于量化推荐中的性别偏见。具体而言，我们强调了在更细粒度的层面上评估公平性的重要性，这可以通过我们的度量标准来实现，例如使用推荐项目的类别（如电影类型）来捕捉性别偏见。此外，我们展示了将类别意识的公平性度量作为正则化项与主要推荐损失相结合，可以在训练过程中有效地减少模型输出中的偏见。我们在三个真实世界的数据集上进行了实验，采用五种基础模型以及两种流行的公平性敏感模型，展示了我们提出的度量标准在评估性别的偏见方面的有效性。与之前的度量标准相比，我们的度量标准有助于更深入了解推荐结果中的偏见。另外，我们的结果显示，在推荐不同类别时，通过引入我们的正则化项可以显著提高公平性，而不会对总体推荐性能产生明显损害。 

---
# Towards Sustainable Web Agents: A Plea for Transparency and Dedicated Metrics for Energy Consumption 

**Title (ZH)**: 向可持续的网络代理迈进：呼吁透明度和专门的能耗指标 

**Authors**: Lars Krupp, Daniel Geißler, Paul Lukowicz, Jakob Karolus  

**Link**: [PDF](https://arxiv.org/pdf/2502.17903)  

**Abstract**: Improvements in the area of large language models have shifted towards the construction of models capable of using external tools and interpreting their outputs. These so-called web agents have the ability to interact autonomously with the internet. This allows them to become powerful daily assistants handling time-consuming, repetitive tasks while supporting users in their daily activities. While web agent research is thriving, the sustainability aspect of this research direction remains largely unexplored. We provide an initial exploration of the energy and CO2 cost associated with web agents. Our results show how different philosophies in web agent creation can severely impact the associated expended energy. We highlight lacking transparency regarding the disclosure of model parameters and processes used for some web agents as a limiting factor when estimating energy consumption. As such, our work advocates a change in thinking when evaluating web agents, warranting dedicated metrics for energy consumption and sustainability. 

**Abstract (ZH)**: 大型语言模型领域的进步已经转向构建能够使用外部工具并解释其输出的模型。这类所谓的网络代理能够自主与互联网交互，从而使它们成为处理耗时且重复的任务的强大日常助手，并支持用户的日常活动。尽管网络代理研究正在蓬勃发展，但这一研究方向的可持续性方面仍几乎没有被探索。我们初步探索了与网络代理相关的能源和二氧化碳成本。结果显示，网络代理创建的不同哲学对所消耗的能源产生了严重影响。我们强调，对于某些网络代理，缺乏关于模型参数和使用过程的透明披露是估计能源消耗时的一个限制因素。因此，我们的工作提倡在评估网络代理时进行思维方式的转变，需要制定专门的能源消耗和可持续性指标。 

---
# Science Across Languages: Assessing LLM Multilingual Translation of Scientific Papers 

**Title (ZH)**: 跨语言的科学交流：评估大语言模型在翻译科学论文方面的多语言能力 

**Authors**: Hannah Calzi Kleidermacher, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2502.17882)  

**Abstract**: Scientific research is inherently global. However, the vast majority of academic journals are published exclusively in English, creating barriers for non-native-English-speaking researchers. In this study, we leverage large language models (LLMs) to translate published scientific articles while preserving their native JATS XML formatting, thereby developing a practical, automated approach for implementation by academic journals. Using our approach, we translate articles across multiple scientific disciplines into 28 languages. To evaluate translation accuracy, we introduce a novel question-and-answer (QA) benchmarking method, in which an LLM generates comprehension-based questions from the original text and then answers them based on the translated text. Our benchmark results show an average performance of 95.9%, showing that the key scientific details are accurately conveyed. In a user study, we translate the scientific papers of 15 researchers into their native languages, finding that the authors consistently found the translations to accurately capture the original information in their articles. Interestingly, a third of the authors found many technical terms "overtranslated," expressing a preference to keep terminology more familiar in English untranslated. Finally, we demonstrate how in-context learning techniques can be used to align translations with domain-specific preferences such as mitigating overtranslation, highlighting the adaptability and utility of LLM-driven scientific translation. The code and translated articles are available at this https URL. 

**Abstract (ZH)**: 科学研究本质上是全球性的。然而，绝大多数学术期刊仅以英语出版，这为非英语母语的研究者设定了障碍。在本研究中，我们利用大规模语言模型（LLMs）来翻译已发表的科学文章，并保留其原始的JATS XML格式，在此过程中开发了一种实用的自动化方法，供学术期刊实施。使用我们的方法，我们将多学科的学术论文翻译成28种语言。为了评估翻译准确性，我们引入了一种新的问答（QA）基准测试方法，在这种方法中，一个LLM从原始文本生成基于理解的问题，然后基于翻译文本回答这些问题。我们的基准测试结果显示出95.9%的平均性能，表明关键的科学细节得到了准确传达。在一项用户研究中，我们将15位研究人员的科学论文翻译成他们的母语，发现作者一致认为这些翻译准确地捕捉到了原文中的信息。有趣的是，三分之一的作者发现许多技术术语被“过度翻译”，他们更倾向于保留这些术语的英语表达。最后，我们展示了如何利用上下文学习技术来调整翻译，以适应特定领域的偏好，如减少过度翻译，这证明了基于LLM的科学翻译的可适应性和实用性。代码和翻译文章可在以下网址获取：[this https URL]。 

---
# A Combinatorial Identities Benchmark for Theorem Proving via Automated Theorem Generation 

**Title (ZH)**: 通过自动定理生成进行定理证明的组合恒等式基准测试 

**Authors**: Beibei Xiong, Hangyu Lv, Haojia Shan, Jianlin Wang, Zhengfeng Yang, Lihong Zhi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17840)  

**Abstract**: Large language models (LLMs) have significantly advanced formal theorem proving, yet the scarcity of high-quality training data constrains their capabilities in complex mathematical domains. Combinatorics, a cornerstone of mathematics, provides essential tools for analyzing discrete structures and solving optimization problems. However, its inherent complexity makes it particularly challenging for automated theorem proving (ATP) for combinatorial identities. To address this, we manually construct LeanComb, combinatorial identities benchmark in Lean, which is, to our knowledge, the first formalized theorem proving benchmark built for combinatorial identities. We develop an Automated Theorem Generator for Combinatorial Identities, ATG4CI, which combines candidate tactics suggested by a self-improving large language model with a Reinforcement Learning Tree Search approach for tactic prediction. By utilizing ATG4CI, we generate a LeanComb-Enhanced dataset comprising 260K combinatorial identities theorems, each with a complete formal proof in Lean, and experimental evaluations demonstrate that models trained on this dataset can generate more effective tactics, thereby improving success rates in automated theorem proving for combinatorial identities. 

**Abstract (ZH)**: 大规模语言模型（LLMs）已经在形式定理证明方面取得了显著进展，但高质量训练数据的稀缺性限制了它们在复杂数学领域的应用能力。组合数学作为数学的基础，提供了分析离散结构和解决优化问题的重要工具。然而，其固有的复杂性使得自动定理证明（ATP）在组合恒等式领域尤为具有挑战性。为解决这一问题，我们手动构建了LeanComb，这是一种基于Lean的组合恒等式基准，据我们所知，这是首个专门为组合恒等式构建的形式化定理证明基准。我们开发了一个名为ATG4CI的自动定理生成器，该生成器结合了一种自我改进的大规模语言模型提供的候选策略与强化学习树搜索方法以预测策略。利用ATG4CI，我们生成了一个包含26万个组合恒等式定理的LeanComb-Enhanced数据集，每个定理在Lean中都有完整的形式证明。实验评估表明，基于此数据集训练的模型能够生成更有效的策略，从而提高组合恒等式的自动定理证明成功率。 

---
# DocPuzzle: A Process-Aware Benchmark for Evaluating Realistic Long-Context Reasoning Capabilities 

**Title (ZH)**: DocPuzzle：一种基于过程的基准测试，用于评估实际长上下文推理能力 

**Authors**: Tianyi Zhuang, Chuqiao Kuang, Xiaoguang Li, Yihua Teng, Jihao Wu, Yasheng Wang, Lifeng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17807)  

**Abstract**: We present DocPuzzle, a rigorously constructed benchmark for evaluating long-context reasoning capabilities in large language models (LLMs). This benchmark comprises 100 expert-level QA problems requiring multi-step reasoning over long real-world documents. To ensure the task quality and complexity, we implement a human-AI collaborative annotation-validation pipeline. DocPuzzle introduces an innovative evaluation framework that mitigates guessing bias through checklist-guided process analysis, establishing new standards for assessing reasoning capacities in LLMs. Our evaluation results show that: 1)Advanced slow-thinking reasoning models like o1-preview(69.7%) and DeepSeek-R1(66.3%) significantly outperform best general instruct models like Claude 3.5 Sonnet(57.7%); 2)Distilled reasoning models like DeepSeek-R1-Distill-Qwen-32B(41.3%) falls far behind the teacher model, suggesting challenges to maintain the generalization of reasoning capabilities relying solely on distillation. 

**Abstract (ZH)**: 我们提出了DocPuzzle，这是一个严格构建的基准测试，用于评估大型语言模型（LLMs）的长上下文推理能力。该基准测试包含100个专家级的问答问题，要求在长篇真实世界文档上进行多步骤推理。为确保任务的质量和复杂性，我们实现了一个由人工和AI协作的注释验证流程。DocPuzzle引入了一种创新的评估框架，通过清单指导的过程分析来减少猜测偏差，从而为评估LLMs的推理能力建立了新的标准。我们的评估结果显示：1）先进的慢思维推理模型（如o1-preview, 69.7% 和 DeepSeek-R1, 66.3%）显著优于最佳通用指令模型（如Claude 3.5 Sonnet, 57.7%）；2）如DeepSeek-R1-Distill-Qwen-32B（41.3%）这样的提炼推理模型远逊于教师模型，表明仅通过提炼难以维持推理能力的一般化。 

---
# Detection of LLM-Paraphrased Code and Identification of the Responsible LLM Using Coding Style Features 

**Title (ZH)**: 使用编码风格特征检测由大语言模型改写的代码并识别负责的大语言模型 

**Authors**: Shinwoo Park, Hyundong Jin, Jeong-won Cha, Yo-Sub Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.17749)  

**Abstract**: Recent progress in large language models (LLMs) for code generation has raised serious concerns about intellectual property protection. Malicious users can exploit LLMs to produce paraphrased versions of proprietary code that closely resemble the original. While the potential for LLM-assisted code paraphrasing continues to grow, research on detecting it remains limited, underscoring an urgent need for detection system. We respond to this need by proposing two tasks. The first task is to detect whether code generated by an LLM is a paraphrased version of original human-written code. The second task is to identify which LLM is used to paraphrase the original code. For these tasks, we construct a dataset LPcode consisting of pairs of human-written code and LLM-paraphrased code using various LLMs.
We statistically confirm significant differences in the coding styles of human-written and LLM-paraphrased code, particularly in terms of naming consistency, code structure, and readability. Based on these findings, we develop LPcodedec, a detection method that identifies paraphrase relationships between human-written and LLM-generated code, and discover which LLM is used for the paraphrasing. LPcodedec outperforms the best baselines in two tasks, improving F1 scores by 2.64% and 15.17% while achieving speedups of 1,343x and 213x, respectively. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在代码生成方面的进步引发了关于知识产权保护的严重关切。恶意用户可以利用LLMs生成与原始代码高度相似但又具有篡改性的版本。随着LLM辅助代码改写潜力的不断增长，相关检测研究仍显不足，这凸显出迫切需要开发检测系统。为应对这一需求，我们提出了两项任务。第一项任务是检测LLM生成的代码是否是对原始人类撰写的代码的改写版本。第二项任务是确定利用哪个LLM来改写原始代码。为这两项任务，我们使用多种LLM构建了一个包含人类撰写的代码和LLM改写代码的数据集LPcode。

我们通过统计手段证实了人类撰写的代码和LLM改写代码的编程风格存在显著差异，特别是在命名一致性、代码结构和可读性方面。基于这些发现，我们开发了一种名为LPcodedec的检测方法，该方法可以识别人类撰写的和由LLM生成的代码之间的改写关系，并发现用于改写的LLM是哪个模型。在两项任务中，LPcodedec超越了最好的基线方法，在F1分数上分别提升了2.64%和15.17%，同时分别实现了1,343倍和213倍的加速。 

---
# Mind the Gesture: Evaluating AI Sensitivity to Culturally Offensive Non-Verbal Gestures 

**Title (ZH)**: 注意手势：评估AI对具有文化冒犯性的非言语手势的敏感性 

**Authors**: Akhila Yerukola, Saadia Gabriel, Nanyun Peng, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2502.17710)  

**Abstract**: Gestures are an integral part of non-verbal communication, with meanings that vary across cultures, and misinterpretations that can have serious social and diplomatic consequences. As AI systems become more integrated into global applications, ensuring they do not inadvertently perpetuate cultural offenses is critical. To this end, we introduce Multi-Cultural Set of Inappropriate Gestures and Nonverbal Signs (MC-SIGNS), a dataset of 288 gesture-country pairs annotated for offensiveness, cultural significance, and contextual factors across 25 gestures and 85 countries. Through systematic evaluation using MC-SIGNS, we uncover critical limitations: text-to-image (T2I) systems exhibit strong US-centric biases, performing better at detecting offensive gestures in US contexts than in non-US ones; large language models (LLMs) tend to over-flag gestures as offensive; and vision-language models (VLMs) default to US-based interpretations when responding to universal concepts like wishing someone luck, frequently suggesting culturally inappropriate gestures. These findings highlight the urgent need for culturally-aware AI safety mechanisms to ensure equitable global deployment of AI technologies. 

**Abstract (ZH)**: 手势是非语言交流不可或缺的一部分，其意义在不同文化中各不相同，误解可能会导致严重的社会和外交后果。随着人工智能系统在全球应用中的日益融合，确保它们不会无意中延续文化冒犯至关重要。为此，我们引入了多文化不适当手势和非语言信号数据集（MC-SIGNS），该数据集包括288个手势-国家配对，涵盖了25种手势和85个国家的文化意义、冒犯性及情境因素的标注。通过使用MC-SIGNS进行系统评估，我们发现了一些关键的局限性：文本到图像（T2I）系统表现出强烈的美国中心偏见，它们在识别美国情境中的冒犯手势方面表现更好，而在非美国情境中的表现则较差；大型语言模型（LLMs）倾向于过度标记手势为冒犯性；而视觉语言模型（VLMs）在处理诸如为他人祝运这类普适概念时，常常依赖于基于美国的解释，频繁建议不适当的手势。这些发现突显了对文化意识的人工智能安全机制的迫切需求，以确保人工智能技术在全球范围内的公平应用。 

---
# From Perceptions to Decisions: Wildfire Evacuation Decision Prediction with Behavioral Theory-informed LLMs 

**Title (ZH)**: 从感知到决策：基于行为理论指导的大语言模型 wildfire 趋避决策预测 

**Authors**: Ruxiao Chen, Chenguang Wang, Yuran Sun, Xilei Zhao, Susu Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17701)  

**Abstract**: Evacuation decision prediction is critical for efficient and effective wildfire response by helping emergency management anticipate traffic congestion and bottlenecks, allocate resources, and minimize negative impacts. Traditional statistical methods for evacuation decision prediction fail to capture the complex and diverse behavioral logic of different individuals. In this work, for the first time, we introduce FLARE, short for facilitating LLM for advanced reasoning on wildfire evacuation decision prediction, a Large Language Model (LLM)-based framework that integrates behavioral theories and models to streamline the Chain-of-Thought (CoT) reasoning and subsequently integrate with memory-based Reinforcement Learning (RL) module to provide accurate evacuation decision prediction and understanding. Our proposed method addresses the limitations of using existing LLMs for evacuation behavioral predictions, such as limited survey data, mismatching with behavioral theory, conflicting individual preferences, implicit and complex mental states, and intractable mental state-behavior mapping. Experiments on three post-wildfire survey datasets show an average of 20.47% performance improvement over traditional theory-informed behavioral models, with strong cross-event generalizability. Our complete code is publicly available at this https URL 

**Abstract (ZH)**: 有效的疏散决策预测对于提高野火应对的效率和效果至关重要，它可以帮助应急管理机构预见交通拥堵和瓶颈问题，合理分配资源，并尽量减少负面影响。传统的统计方法在预测疏散决策时，难以捕捉不同个体复杂的多样行为逻辑。在此项研究中，我们首次引入了一个名为“FLARE”的框架，即“促进大型语言模型在野火疏散决策预测中的高级推理”，该框架以大型语言模型（LLM）为基础，整合了行为理论和模型，简化了链式推理（CoT），并在此基础上结合基于记忆的强化学习（RL）模块，提供准确的疏散决策预测和理解。我们提出的方法解决了现有大型语言模型在疏散行为预测中的局限性，如有限的调查数据、不符合行为理论、个体偏好冲突、复杂的心理状态以及难以解决的心理状态-行为映射问题。在三个野火后调查数据集上的实验显示，与传统的以理论为基础的行为模型相比，平均性能提高了20.47%，并且具有较强的跨事件通用性。完整的代码已公开，可在此处访问：this https URL 

---
# Socratic: Enhancing Human Teamwork via AI-enabled Coaching 

**Title (ZH)**: Socratic：通过AI赋能的教练技术提升人类团队协作效能 

**Authors**: Sangwon Seo, Bing Han, Rayan E. Harari, Roger D. Dias, Marco A. Zenati, Eduardo Salas, Vaibhav Unhelkar  

**Link**: [PDF](https://arxiv.org/pdf/2502.17643)  

**Abstract**: Coaches are vital for effective collaboration, but cost and resource constraints often limit their availability during real-world tasks. This limitation poses serious challenges in life-critical domains that rely on effective teamwork, such as healthcare and disaster response. To address this gap, we propose and realize an innovative application of AI: task-time team coaching. Specifically, we introduce Socratic, a novel AI system that complements human coaches by providing real-time guidance during task execution. Socratic monitors team behavior, detects misalignments in team members' shared understanding, and delivers automated interventions to improve team performance. We validated Socratic through two human subject experiments involving dyadic collaboration. The results demonstrate that the system significantly enhances team performance with minimal interventions. Participants also perceived Socratic as helpful and trustworthy, supporting its potential for adoption. Our findings also suggest promising directions both for AI research and its practical applications to enhance human teamwork. 

**Abstract (ZH)**: 教练对于有效协作至关重要，但在现实任务中，由于成本和资源的限制，往往很难保证其可用性。这种限制在依赖有效团队合作的生活关键领域（如医疗保健和灾害响应）中构成了重大挑战。为解决这一问题，我们提出并实现了AI在任务时间团队协作中的创新应用：任务时教练系统。具体而言，我们引入了Socratic，这是一种新颖的AI系统，能够通过提供实时指导来补充人类教练的功能。Socratic监测团队行为，检测团队成员之间共享理解的不一致，并提供自动化干预措施以提高团队表现。我们通过涉及二元合作的人类主体实验验证了Socratic。结果表明，该系统在最小干预的情况下显著提升了团队表现。参与者还认为Socratic具有帮助性和可信度，支持其在实际应用中的潜在采用。我们的发现还为AI研究及其改善人类团队合作的实际应用提供了具有前景的方向。 

---
# Representation Engineering for Large-Language Models: Survey and Research Challenges 

**Title (ZH)**: 面向大规模语言模型的表示工程：综述与研究挑战 

**Authors**: Lukasz Bartoszcze, Sarthak Munshi, Bryan Sukidi, Jennifer Yen, Zejia Yang, David Williams-King, Linh Le, Kosi Asuzu, Carsten Maple  

**Link**: [PDF](https://arxiv.org/pdf/2502.17601)  

**Abstract**: Large-language models are capable of completing a variety of tasks, but remain unpredictable and intractable. Representation engineering seeks to resolve this problem through a new approach utilizing samples of contrasting inputs to detect and edit high-level representations of concepts such as honesty, harmfulness or power-seeking. We formalize the goals and methods of representation engineering to present a cohesive picture of work in this emerging discipline. We compare it with alternative approaches, such as mechanistic interpretability, prompt-engineering and fine-tuning. We outline risks such as performance decrease, compute time increases and steerability issues. We present a clear agenda for future research to build predictable, dynamic, safe and personalizable LLMs. 

**Abstract (ZH)**: 大型语言模型能够完成多种任务，但它们仍然具有不可预测性和难以处理的特点。通过一种新的方法——利用具有对比输入的样本来检测和编辑有关诚实性、危害性或权力获取等高级概念的表示，表示工程学旨在解决这些问题。我们对表示工程学的目标和方法进行正式化，以呈现这一新兴学科中工作的整体图景。我们将它与替代方法，如机械可解释性、提示工程和微调进行比较。我们指出了由此方法带来的风险，例如性能下降、计算时间增加和可控性问题。我们提出了明确的研究议程，旨在构建可预测、动态、安全和个性化的大型语言模型。 

---
# Intention Recognition in Real-Time Interactive Navigation Maps 

**Title (ZH)**: 实时交互导航图中的意图识别 

**Authors**: Peijie Zhao, Zunayed Arefin, Felipe Meneguzzi, Ramon Fraga Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2502.17581)  

**Abstract**: In this demonstration, we develop IntentRec4Maps, a system to recognise users' intentions in interactive maps for real-world navigation. IntentRec4Maps uses the Google Maps Platform as the real-world interactive map, and a very effective approach for recognising users' intentions in real-time. We showcase the recognition process of IntentRec4Maps using two different Path-Planners and a Large Language Model (LLM).
GitHub: this https URL 

**Abstract (ZH)**: 在本次演示中，我们开发了IntentRec4Maps系统，该系统用于识别用户在交互地图中进行实际导航时的意图。IntentRec4Maps使用Google Maps Platform作为实时交互地图，并采用了一种非常有效的方法来实现实时识别用户的意图。我们通过使用两种不同的路径规划器（Path-Planner）和大语言模型（Large Language Model, LLM）来展示IntentRec4Maps的识别过程。

GitHub：[此处填写链接]

（注意：由于提供的链接是Markdown格式的原始链接，需要根据实际需求将其转换为完整的URL。） 

---
# How Do Large Language Monkeys Get Their Power (Laws)? 

**Title (ZH)**: 大型语言模型是如何获得其权力（能力）的？ 

**Authors**: Rylan Schaeffer, Joshua Kazdan, John Hughes, Jordan Juravsky, Sara Price, Aengus Lynch, Erik Jones, Robert Kirk, Azalia Mirhoseini, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2502.17578)  

**Abstract**: Recent research across mathematical problem solving, proof assistant programming and multimodal jailbreaking documents a striking finding: when (multimodal) language model tackle a suite of tasks with multiple attempts per task -- succeeding if any attempt is correct -- then the negative log of the average success rate scales a power law in the number of attempts. In this work, we identify an apparent puzzle: a simple mathematical calculation predicts that on each problem, the failure rate should fall exponentially with the number of attempts. We confirm this prediction empirically, raising a question: from where does aggregate polynomial scaling emerge? We then answer this question by demonstrating per-problem exponential scaling can be made consistent with aggregate polynomial scaling if the distribution of single-attempt success probabilities is heavy tailed such that a small fraction of tasks with extremely low success probabilities collectively warp the aggregate success trend into a power law - even as each problem scales exponentially on its own. We further demonstrate that this distributional perspective explains previously observed deviations from power law scaling, and provides a simple method for forecasting the power law exponent with an order of magnitude lower relative error, or equivalently, ${\sim}2-4$ orders of magnitude less inference compute. Overall, our work contributes to a better understanding of how neural language model performance improves with scaling inference compute and the development of scaling-predictable evaluations of (multimodal) language models. 

**Abstract (ZH)**: 近年来，关于数学问题解决、形式证明编程和多模态监狱破解的研究揭示了一个惊人的发现：当（多模态）语言模型对一系列任务进行多次尝试（只要有一次尝试正确即算成功）时，平均成功率的负对数与尝试次数之间呈现出幂律关系。在本研究中，我们发现了一个明显的悖论：一个简单的数学计算表明，每个问题的失败率应随着尝试次数的增加而指数下降。我们通过实验证明了这一预测，引发了一个问题：为何总体上呈现出幂律关系？我们随后通过对这一问题的回答来解答这一疑问：如果单次尝试成功概率的分布呈现重尾分布，即一小部分具有极低成功概率的任务共同导致总体成功率趋势呈现出幂律关系，即使每个问题各自表现出指数下降的趋势，这种分布视角也能使单个问题的指数变化与总体的幂律变化保持一致。我们还证明了这种分布视角能够解释之前观察到的幂律关系偏差，并提供了一种更简单的预测幂律指数的方法，相对误差降低了几个数量级，换句话说，计算推理所需的量级降低了约为2到4个数量级。整体而言，本研究有助于更好地理解神经语言模型性能随推理计算规模变化的情况，并为（多模态）语言模型提供一种预测性评估方法。 

---
# Dataset Featurization: Uncovering Natural Language Features through Unsupervised Data Reconstruction 

**Title (ZH)**: 数据集特征化：通过无监督数据重构发现自然语言特征 

**Authors**: Michal Bravansky, Vaclav Kubon, Suhas Hariharan, Robert Kirk  

**Link**: [PDF](https://arxiv.org/pdf/2502.17541)  

**Abstract**: Interpreting data is central to modern research. Large language models (LLMs) show promise in providing such natural language interpretations of data, yet simple feature extraction methods such as prompting often fail to produce accurate and versatile descriptions for diverse datasets and lack control over granularity and scale. To address these limitations, we propose a domain-agnostic method for dataset featurization that provides precise control over the number of features extracted while maintaining compact and descriptive representations comparable to human expert labeling. Our method optimizes the selection of informative binary features by evaluating the ability of an LLM to reconstruct the original data using those features. We demonstrate its effectiveness in dataset modeling tasks and through two case studies: (1) Constructing a feature representation of jailbreak tactics that compactly captures both the effectiveness and diversity of a larger set of human-crafted attacks; and (2) automating the discovery of features that align with human preferences, achieving accuracy and robustness comparable to expert-crafted features. Moreover, we show that the pipeline scales effectively, improving as additional features are sampled, making it suitable for large and diverse datasets. 

**Abstract (ZH)**: 数据解释是现代研究的核心。大规模语言模型（LLMs）在提供这种自然语言数据解释方面显示出了潜力，但简单的特征提取方法（如提示）往往无法为多样化的数据集生成准确且多样的描述，并且在控制细节和规模方面存在不足。为了解决这些问题，我们提出了一种领域无关的特征集方法，该方法能够在保持紧凑且描述性表示的同时，提供对提取特征数量的精确控制。我们的方法通过评估LLM使用这些特征重构原始数据的能力，来优化信息性二元特征的选择。我们展示了该方法在数据集建模任务中的有效性，并通过两个案例研究进行了验证：（1）构建一种特征表示方法，能够简洁地捕捉更大规模的人工设计攻击集的有效性和多样性；（2）自动化发现与人类偏好相一致的特征，实现与专家设计特征相当的准确性和稳健性。此外，我们展示了该流水线在大规模和多样化数据集上的有效扩展，随着更多特征的采集，其性能会进一步提升，使其适用于大规模数据集。 

---
# User Intent to Use DeekSeep for Healthcare Purposes and their Trust in the Large Language Model: Multinational Survey Study 

**Title (ZH)**: 用户在医疗用途下使用DeekSeep的意图及其对大型语言模型的信任：跨国调查研究 

**Authors**: Avishek Choudhury, Yeganeh Shahsavar, Hamid Shamszare  

**Link**: [PDF](https://arxiv.org/pdf/2502.17487)  

**Abstract**: Large language models (LLMs) increasingly serve as interactive healthcare resources, yet user acceptance remains underexplored. This study examines how ease of use, perceived usefulness, trust, and risk perception interact to shape intentions to adopt DeepSeek, an emerging LLM-based platform, for healthcare purposes. A cross-sectional survey of 556 participants from India, the United Kingdom, and the United States was conducted to measure perceptions and usage patterns. Structural equation modeling assessed both direct and indirect effects, including potential quadratic relationships. Results revealed that trust plays a pivotal mediating role: ease of use exerts a significant indirect effect on usage intentions through trust, while perceived usefulness contributes to both trust development and direct adoption. By contrast, risk perception negatively affects usage intent, emphasizing the importance of robust data governance and transparency. Notably, significant non-linear paths were observed for ease of use and risk, indicating threshold or plateau effects. The measurement model demonstrated strong reliability and validity, supported by high composite reliabilities, average variance extracted, and discriminant validity measures. These findings extend technology acceptance and health informatics research by illuminating the multifaceted nature of user adoption in sensitive domains. Stakeholders should invest in trust-building strategies, user-centric design, and risk mitigation measures to encourage sustained and safe uptake of LLMs in healthcare. Future work can employ longitudinal designs or examine culture-specific variables to further clarify how user perceptions evolve over time and across different regulatory environments. Such insights are critical for harnessing AI to enhance outcomes. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地作为交互式的医疗资源，但用户接受程度仍待深入探索。本研究探讨了易用性、感知有用性、信任和风险感知如何相互作用，形成采用新兴的基于LLM的DeepSeek平台进行医疗目的的意图。研究通过横断面调查，对来自印度、英国和美国的556名参与者进行了调查，以测量其感知和使用模式。结构方程模型评估了直接和间接效应，包括潜在的二次关系。研究结果表明，信任发挥着关键的中介作用：易用性通过信任产生显著的间接效应，而感知有用性则促进信任的发展并直接影响采用。相比之下，风险感知对使用意愿产生负面影响，强调了稳健的数据治理和透明性的重要性。值得注意的是，对于易用性和风险的显著非线性路径表明了阈值或平台效应的存在。测量模型证明了高度的可靠性和有效性，得到了高综合信度、平均方差抽取量和辨别信度度量的支持。这些发现扩展了技术接受和健康信息技术的研究，揭示了敏感领域用户采用的多维本质。利益相关方应投资于信任建立策略、用户中心的设计和风险缓解措施，以鼓励LLMs在医疗领域的持续安全采用。未来的研究可以采用纵向设计或检查文化特定变量，进一步明确用户感知随时间变化并在不同监管环境中如何演变。这些见解对于利用人工智能提升结果至关重要。 

---
# Scalable Equilibrium Sampling with Sequential Boltzmann Generators 

**Title (ZH)**: 可扩展的平衡采样方法：基于序列玻尔兹曼生成器 

**Authors**: Charlie B. Tan, Avishek Joey Bose, Chen Lin, Leon Klein, Michael M. Bronstein, Alexander Tong  

**Link**: [PDF](https://arxiv.org/pdf/2502.18462)  

**Abstract**: Scalable sampling of molecular states in thermodynamic equilibrium is a long-standing challenge in statistical physics. Boltzmann generators tackle this problem by pairing powerful normalizing flows with importance sampling to obtain statistically independent samples under the target distribution. In this paper, we extend the Boltzmann generator framework and introduce Sequential Boltzmann generators (SBG) with two key improvements. The first is a highly efficient non-equivariant Transformer-based normalizing flow operating directly on all-atom Cartesian coordinates. In contrast to equivariant continuous flows of prior methods, we leverage exactly invertible non-equivariant architectures which are highly efficient both during sample generation and likelihood computation. As a result, this unlocks more sophisticated inference strategies beyond standard importance sampling. More precisely, as a second key improvement we perform inference-time scaling of flow samples using annealed Langevin dynamics which transports samples toward the target distribution leading to lower variance (annealed) importance weights which enable higher fidelity resampling with sequential Monte Carlo. SBG achieves state-of-the-art performance w.r.t. all metrics on molecular systems, demonstrating the first equilibrium sampling in Cartesian coordinates of tri, tetra, and hexapeptides that were so far intractable for prior Boltzmann generators. 

**Abstract (ZH)**: 在统计物理中，高效地在热力学平衡状态下采样分子状态是一个长期存在的挑战。Boltzmann生成器通过将强大的归一化流与重要性采样相结合来解决这个问题，以在目标分布下获得统计上独立的样本。在这篇文章中，我们扩展了Boltzmann生成器框架，并引入了顺序Boltzmann生成器（SBG），并在此过程中实现了两项关键改进。首先是基于非等变Transformer的归一化流，它可以高效地直接处理原子笛卡尔坐标。与先前方法中的等变连续流不同，我们利用了高度可逆的非等变架构，这在样本生成和似然计算过程中都非常高效。这一改进开启了超越标准重要性采样的更复杂的推断策略。其次，作为另一项关键改进，我们在推断期间使用退火拉梅尔动态对流样本进行缩放，这将样本移向目标分布，从而降低重要性权重的方差（退火），使顺序蒙特卡洛方法能够进行更高的保真度采样。SBG在分子系统的所有指标上都达到了最先进的性能，展示了对以前的Boltzmann生成器无法处理的三肽、四肽和六肽在笛卡尔坐标下的平衡采样。 

---
# FRIDA to the Rescue! Analyzing Synthetic Data Effectiveness in Object-Based Common Sense Reasoning for Disaster Response 

**Title (ZH)**: FRIDA 助力救援！合成数据在基于对象的常识推理中的有效性分析——以灾害响应为例 

**Authors**: Mollie Shichman, Claire Bonial, Austin Blodgett, Taylor Hudson, Francis Ferraro, Rachel Rudinger  

**Link**: [PDF](https://arxiv.org/pdf/2502.18452)  

**Abstract**: Large Language Models (LLMs) have the potential for substantial common sense reasoning. However, these capabilities are often emergent in larger models. This means smaller models that can be run locally are less helpful and capable with respect to certain reasoning tasks. To meet our problem space requirements, we fine-tune smaller LLMs to disaster domains, as these domains involve complex and low-frequency physical common sense knowledge. We introduce a pipeline to create Field Ready Instruction Decoding Agent (FRIDA) models, where domain experts and linguists combine their knowledge to make high-quality seed data that is used to generate synthetic data for fine-tuning. We create a set of 130 seed instructions for synthetic generation, a synthetic dataset of 25000 instructions, and 119 evaluation instructions relating to both general and earthquake-specific object affordances. We fine-tune several LLaMa and Mistral instruction-tuned models and find that FRIDA models outperform their base models at a variety of sizes. We then run an ablation study to understand which kinds of synthetic data most affect performance and find that training physical state and object function common sense knowledge alone improves over FRIDA models trained on all data. We conclude that the FRIDA pipeline is capable of instilling general common sense, but needs to be augmented with information retrieval for specific domain knowledge. 

**Abstract (ZH)**: 大型语言模型（LLMs）在常识推理方面具有巨大的潜力。然而，这些能力往往只在更大的模型中显现，这意味着较小的本地可运行模型在某些推理任务上较为无力。为满足我们的问题空间要求，我们对较小的LLMs进行了微调，使其适应灾难领域，因为这些领域涉及复杂的低频物理常识知识。我们引入了一种流水线方法，创建了场可用指令解码代理（FRIDA）模型，其中领域专家和语言学家结合他们的知识生成高质量的种子数据，用于生成用于微调的合成数据。我们创建了一套130个种子指令以进行合成生成，一个包含25,000个指令的合成数据集，以及119个评估指令，涉及一般和地震特定对象的可用性。我们对多个LLaMa和Mistral指令微调模型进行了微调，并发现FRIDA模型在各种规模下均优于其基础模型。然后我们进行了一项消融研究，以了解哪些类型的合成数据对性能影响最大，并发现单独训练物理状态和对象功能常识知识的FRIDA模型优于使用所有数据训练的FRIDA模型。我们得出结论，FRIDA流水线能够灌输一般常识，但需要与信息检索相结合以增加特定领域的知识。 

---
# SWE-RL: Advancing LLM Reasoning via Reinforcement Learning on Open Software Evolution 

**Title (ZH)**: SWE-RL：通过开放软件演化中的强化学习提升大规模语言模型的推理能力 

**Authors**: Yuxiang Wei, Olivier Duchenne, Jade Copet, Quentin Carbonneaux, Lingming Zhang, Daniel Fried, Gabriel Synnaeve, Rishabh Singh, Sida I. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18449)  

**Abstract**: The recent DeepSeek-R1 release has demonstrated the immense potential of reinforcement learning (RL) in enhancing the general reasoning capabilities of large language models (LLMs). While DeepSeek-R1 and other follow-up work primarily focus on applying RL to competitive coding and math problems, this paper introduces SWE-RL, the first approach to scale RL-based LLM reasoning for real-world software engineering. Leveraging a lightweight rule-based reward (e.g., the similarity score between ground-truth and LLM-generated solutions), SWE-RL enables LLMs to autonomously recover a developer's reasoning processes and solutions by learning from extensive open-source software evolution data -- the record of a software's entire lifecycle, including its code snapshots, code changes, and events such as issues and pull requests. Trained on top of Llama 3, our resulting reasoning model, Llama3-SWE-RL-70B, achieves a 41.0% solve rate on SWE-bench Verified -- a human-verified collection of real-world GitHub issues. To our knowledge, this is the best performance reported for medium-sized (<100B) LLMs to date, even comparable to leading proprietary LLMs like GPT-4o. Surprisingly, despite performing RL solely on software evolution data, Llama3-SWE-RL has even emerged with generalized reasoning skills. For example, it shows improved results on five out-of-domain tasks, namely, function coding, library use, code reasoning, mathematics, and general language understanding, whereas a supervised-finetuning baseline even leads to performance degradation on average. Overall, SWE-RL opens up a new direction to improve the reasoning capabilities of LLMs through reinforcement learning on massive software engineering data. 

**Abstract (ZH)**: 最近发布的DeepSeek-R1表明，强化学习（RL）在提升大型语言模型（LLMs）的通用推理能力方面具有巨大潜力。虽然DeepSeek-R1及其后续工作主要集中在将RL应用于编程竞赛和数学问题上，本论文介绍了SWE-RL，这是首个用于实世界软件工程的基于RL的大规模LLM推理方法。通过利用一种轻量级基于规则的奖励（例如，正确解与LLM生成解之间的相似性得分），SWE-RL使LLMs能够通过学习广泛开源软件演化数据自主恢复开发者的推理过程和解决方案，这些数据记录了软件生命周期的整个过程，包括代码快照、代码变更以及如问题和拉取请求等事件。以Llama 3为基础训练，我们得到的推理模型Llama3-SWE-RL-70B在SWE-bench Verified（一个由人工验证的真实GitHub问题集）上实现了41.0%的解题率。据我们所知，这是迄今为止中型（<100B）LLMs的最佳性能，甚至可以媲美如GPT-4o等主流的专有LLMs。令人惊讶的是，尽管仅在软件演化数据上进行RL训练，但Llama3-SWE-RL甚至展现出通用推理能力。例如，它在五个离域任务（函数编码、库使用、代码推理、数学和通用语言理解）上取得了更好的结果，而监督微调基准方法反而在平均性能上下降了。总体而言，SWE-RL为通过大规模软件工程数据的强化学习来提升LLMs的推理能力开辟了新的方向。 

---
# Disambiguate First Parse Later: Generating Interpretations for Ambiguity Resolution in Semantic Parsing 

**Title (ZH)**: 先去歧义化后解析：在语义解析中的歧义解决生成解释 

**Authors**: Irina Saparina, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2502.18448)  

**Abstract**: Handling ambiguity and underspecification is an important challenge in natural language interfaces, particularly for tasks like text-to-SQL semantic parsing. We propose a modular approach that resolves ambiguity using natural language interpretations before mapping these to logical forms (e.g., SQL queries). Although LLMs excel at parsing unambiguous utterances, they show strong biases for ambiguous ones, typically predicting only preferred interpretations. We constructively exploit this bias to generate an initial set of preferred disambiguations and then apply a specialized infilling model to identify and generate missing interpretations. To train the infilling model, we introduce an annotation method that uses SQL execution to validate different meanings. Our approach improves interpretation coverage and generalizes across datasets with different annotation styles, database structures, and ambiguity types. 

**Abstract (ZH)**: 处理歧义和欠指定是自然语言接口中的一个重要的挑战，特别是在文本到SQL语义解析等任务中尤为如此。我们提出了一种模块化的方法，该方法首先使用自然语言解释来解决歧义问题，然后再将这些解释映射到逻辑形式（例如，SQL查询）。尽管大语言模型在解析无歧义的表达方面表现出色，但对于歧义表达，它们往往倾向于只预测优先的解释，并显示出明显的偏好倾向。我们利用这一偏好倾向，生成一组优先的解释，并应用一个专门的填补模型来识别并生成缺失的解释。为了训练填补模型，我们引入了一种注释方法，该方法使用SQL执行来验证不同的含义。我们的方法提高了解释的覆盖范围，并能够跨具有不同注释风格、数据库结构和不同类型的歧义的数据集进行泛化。 

---
# ToMCAT: Theory-of-Mind for Cooperative Agents in Teams via Multiagent Diffusion Policies 

**Title (ZH)**: ToMCAT：通过多智能体扩散策略在团队中实现共情代理的理论-of-心智 

**Authors**: Pedro Sequeira, Vidyasagar Sadhu, Melinda Gervasio  

**Link**: [PDF](https://arxiv.org/pdf/2502.18438)  

**Abstract**: In this paper we present ToMCAT (Theory-of-Mind for Cooperative Agents in Teams), a new framework for generating ToM-conditioned trajectories. It combines a meta-learning mechanism, that performs ToM reasoning over teammates' underlying goals and future behavior, with a multiagent denoising-diffusion model, that generates plans for an agent and its teammates conditioned on both the agent's goals and its teammates' characteristics, as computed via ToM. We implemented an online planning system that dynamically samples new trajectories (replans) from the diffusion model whenever it detects a divergence between a previously generated plan and the current state of the world. We conducted several experiments using ToMCAT in a simulated cooking domain. Our results highlight the importance of the dynamic replanning mechanism in reducing the usage of resources without sacrificing team performance. We also show that recent observations about the world and teammates' behavior collected by an agent over the course of an episode combined with ToM inferences are crucial to generate team-aware plans for dynamic adaptation to teammates, especially when no prior information is provided about them. 

**Abstract (ZH)**: 在本文中，我们提出了ToMCAT（Theory-of-Mind for Cooperative Agents in Teams）框架，这是一个生成基于理论心智（ToM）轨迹的新框架。该框架结合了元学习机制，该机制在队友的潜在目标和未来行为上进行理论心智推理，以及一个多智能体去噪扩散模型，该模型根据代理的目标及其队友的特征（通过理论心智计算得出）为代理及其队友生成计划。我们实现了一个在线规划系统，在检测到先前生成的计划与当前世界状态之间的偏离时，会动态从扩散模型中采样新的轨迹（重规划）。我们使用ToMCAT在模拟烹饪领域进行了多项实验。实验结果强调了动态重规划机制在减少资源使用的同时不牺牲团队性能的重要性。我们还展示了代理在整个情景过程中收集到的关于世界和队友行为的最近观察信息与理论心智推理相结合，对于生成针对队友的团队意识计划至关重要，尤其是在没有提供关于他们任何先验信息的情况下。 

---
# TextGames: Learning to Self-Play Text-Based Puzzle Games via Language Model Reasoning 

**Title (ZH)**: 基于文本的游戏：通过语言模型推理实现自学对弈的文本谜题游戏 

**Authors**: Frederikus Hudi, Genta Indra Winata, Ruochen Zhang, Alham Fikri Aji  

**Link**: [PDF](https://arxiv.org/pdf/2502.18431)  

**Abstract**: Reasoning is a fundamental capability of large language models (LLMs), enabling them to comprehend, analyze, and solve complex problems. In this paper, we introduce TextGames, an innovative benchmark specifically crafted to assess LLMs through demanding text-based games that require advanced skills in pattern recognition, spatial awareness, arithmetic, and logical reasoning. Our analysis probes LLMs' performance in both single-turn and multi-turn reasoning, and their abilities in leveraging feedback to correct subsequent answers through self-reflection. Our findings reveal that, although LLMs exhibit proficiency in addressing most easy and medium-level problems, they face significant challenges with more difficult tasks. In contrast, humans are capable of solving all tasks when given sufficient time. Moreover, we observe that LLMs show improved performance in multi-turn predictions through self-reflection, yet they still struggle with sequencing, counting, and following complex rules consistently. Additionally, models optimized for reasoning outperform pre-trained LLMs that prioritize instruction following, highlighting the crucial role of reasoning skills in addressing highly complex problems. 

**Abstract (ZH)**: 推理是大型语言模型（LLMs）的一项基本能力，使它们能够理解、分析并解决复杂问题。在本文中，我们介绍了TextGames，这是一种创新的基准测试，专门设计通过具有复杂要求的文字游戏来评估LLMs，这些游戏需要高级的模式识别、空间意识、算术和逻辑推理能力。我们的分析探索了LLMs在单步推理和多步推理中的表现，并考察了它们在自我反思过程中利用反馈修正后续答案的能力。我们的研究发现，虽然LLMs在处理大多数简单和中等难度问题上表现出色，但在更困难的任务上却面临重大挑战。相比之下，当给予足够的时间，人类可以解决所有任务。此外，我们观察到，通过自我反思，LLMs在多步预测中的表现有所提高，但仍难以一致地进行序列化、计数和遵循复杂规则。此外，优化用于推理的模型在解决高度复杂问题方面优于侧重指令遵循的预训练LLMs，这突显了推理技能在处理复杂问题中的关键作用。 

---
# Comparative Analysis of MDL-VAE vs. Standard VAE on 202 Years of Gynecological Data 

**Title (ZH)**: MDL-VAE与标准VAE在妇科数据202年研究中的对比分析 

**Authors**: Paula Santos  

**Link**: [PDF](https://arxiv.org/pdf/2502.18412)  

**Abstract**: This study presents a comparative evaluation of a Variational Autoencoder (VAE) enhanced with Minimum Description Length (MDL) regularization against a Standard Autoencoder for reconstructing high-dimensional gynecological data. The MDL-VAE exhibits significantly lower reconstruction errors (MSE, MAE, RMSE) and more structured latent representations, driven by effective KL divergence regularization. Statistical analyses confirm these performance improvements are significant. Furthermore, the MDL-VAE shows consistent training and validation losses and achieves efficient inference times, underscoring its robustness and practical viability. Our findings suggest that incorporating MDL principles into VAE architectures can substantially improve data reconstruction and generalization, making it a promising approach for advanced applications in healthcare data modeling and analysis. 

**Abstract (ZH)**: 本研究对比评估了结合最小描述长度（MDL）正则化的变分自编码器（VAE）与标准自编码器在重建高维妇科数据方面的性能。MDL-VAE在均方误差（MSE）、平均绝对误差（MAE）、均方根误差（RMSE）等方面的重建误差显著较低，并且其潜在表示更为有序，这主要得益于有效的KL散度正则化。统计分析证实，这些性能改进具有显著性。此外，MDL-VAE表现出一致的训练和验证损失，并实现了高效的推理时间，这进一步证明了其稳健性和实际可行性。我们的研究结果表明，在VAE架构中融入MDL原则可以显著改善数据重建和泛化能力，使其成为在医疗健康数据建模与分析中更具前景的方法之一。 

---
# TSKANMixer: Kolmogorov-Arnold Networks with MLP-Mixer Model for Time Series Forecasting 

**Title (ZH)**: TSKANMixer：基于MLP-Mixer模型的柯尔莫哥洛夫-阿诺尔德网络在时间序列预测中的应用 

**Authors**: Young-Chae Hong, Bei Xiao, Yangho Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18410)  

**Abstract**: Time series forecasting has long been a focus of research across diverse fields, including economics, energy, healthcare, and traffic management. Recent works have introduced innovative architectures for time series models, such as the Time-Series Mixer (TSMixer), which leverages multi-layer perceptrons (MLPs) to enhance prediction accuracy by effectively capturing both spatial and temporal dependencies within the data. In this paper, we investigate the capabilities of the Kolmogorov-Arnold Networks (KANs) for time-series forecasting by modifying TSMixer with a KAN layer (TSKANMixer). Experimental results demonstrate that TSKANMixer tends to improve prediction accuracy over the original TSMixer across multiple datasets, ranking among the top-performing models compared to other time series approaches. Our results show that the KANs are promising alternatives to improve the performance of time series forecasting by replacing or extending traditional MLPs. 

**Abstract (ZH)**: 时间序列预测一直是跨多个领域（包括经济学、能源、医疗保健和交通管理）研究的重点。近年来，研究人员提出了一些创新的时间序列模型架构，例如时间序列混合器（TSMixer），它利用多层感知机（MLP）有效捕获数据中的时空依赖性以提高预测准确性。在本文中，我们通过在TSMixer中引入Kolmogorov-Arnold网络（KAN层）来研究KANs在时间序列预测中的能力，构建了TSKANMixer。实验结果表明，TSKANMixer在多个数据集上的预测准确性普遍优于原始的TSMixer，且与其他时间序列方法相比，其表现也名列前茅。我们的结果表明，KANs是提高时间序列预测性能的有前景的替代或扩展传统MLP的选择。 

---
# AgentRM: Enhancing Agent Generalization with Reward Modeling 

**Title (ZH)**: AgentRM：通过奖励建模增强智能体的泛化能力 

**Authors**: Yu Xia, Jingru Fan, Weize Chen, Siyu Yan, Xin Cong, Zhong Zhang, Yaxi Lu, Yankai Lin, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.18407)  

**Abstract**: Existing LLM-based agents have achieved strong performance on held-in tasks, but their generalizability to unseen tasks remains poor. Hence, some recent work focus on fine-tuning the policy model with more diverse tasks to improve the generalizability. In this work, we find that finetuning a reward model to guide the policy model is more robust than directly finetuning the policy model. Based on this finding, we propose AgentRM, a generalizable reward model, to guide the policy model for effective test-time search. We comprehensively investigate three approaches to construct the reward model, including explicit reward modeling, implicit reward modeling and LLM-as-a-judge. We then use AgentRM to guide the answer generation with Best-of-N sampling and step-level beam search. On four types of nine agent tasks, AgentRM enhances the base policy model by $8.8$ points on average, surpassing the top general agent by $4.0$. Moreover, it demonstrates weak-to-strong generalization, yielding greater improvement of $12.6$ on LLaMA-3-70B policy model. As for the specializability, AgentRM can also boost a finetuned policy model and outperform the top specialized agent by $11.4$ on three held-in tasks. Further analysis verifies its effectiveness in test-time scaling. Codes will be released to facilitate the research in this area. 

**Abstract (ZH)**: 现有的基于LLM的智能体在已有任务上表现出色，但在未见任务上的泛化能力仍然较差。因此，一些近期的工作集中在使用更多样化的任务微调策略模型，以提高其泛化能力。本工作中，我们发现微调奖励模型来引导策略模型比直接微调策略模型更具鲁棒性。基于这一发现，我们提出了AgentRM，这是一种通用的奖励模型，用于在有效测试时搜索中引导策略模型。我们综合调查了三种构建奖励模型的方法，包括显式奖励建模、隐式奖励建模以及LLM作为评判者。然后，我们使用AgentRM来利用Best-of-N抽样和步骤级束搜索引导答案生成。在四种不同类型的任务中，AgentRM通过平均提高8.8个点增强了基线策略模型，并在九个代理任务中超越了顶级通用智能体4.0个点。此外，它在LLaMA-3-70B策略模型上的结果表明，它具有从弱到强的泛化能力，使其在策略改进方面达到了12.6个点的更大提升。在两个专用任务和一个已有任务上，AgentRM还能够提升微调后的策略模型，并在三个已见任务中以11.4个点的优势超越顶级专用智能体。进一步的分析验证了其在测试时扩展的有效性。代码将在本研究领域中公开，以促进相关研究。 

---
# EgoSim: An Egocentric Multi-view Simulator and Real Dataset for Body-worn Cameras during Motion and Activity 

**Title (ZH)**: EgoSim：一种基于第一人称视角的多视图模拟器及运动与活动期间穿戴式摄像头真实数据集 

**Authors**: Dominik Hollidt, Paul Streli, Jiaxi Jiang, Yasaman Haghighi, Changlin Qian, Xintong Liu, Christian Holz  

**Link**: [PDF](https://arxiv.org/pdf/2502.18373)  

**Abstract**: Research on egocentric tasks in computer vision has mostly focused on head-mounted cameras, such as fisheye cameras or embedded cameras inside immersive headsets. We argue that the increasing miniaturization of optical sensors will lead to the prolific integration of cameras into many more body-worn devices at various locations. This will bring fresh perspectives to established tasks in computer vision and benefit key areas such as human motion tracking, body pose estimation, or action recognition -- particularly for the lower body, which is typically occluded.
In this paper, we introduce EgoSim, a novel simulator of body-worn cameras that generates realistic egocentric renderings from multiple perspectives across a wearer's body. A key feature of EgoSim is its use of real motion capture data to render motion artifacts, which are especially noticeable with arm- or leg-worn cameras. In addition, we introduce MultiEgoView, a dataset of egocentric footage from six body-worn cameras and ground-truth full-body 3D poses during several activities: 119 hours of data are derived from AMASS motion sequences in four high-fidelity virtual environments, which we augment with 5 hours of real-world motion data from 13 participants using six GoPro cameras and 3D body pose references from an Xsens motion capture suit.
We demonstrate EgoSim's effectiveness by training an end-to-end video-only 3D pose estimation network. Analyzing its domain gap, we show that our dataset and simulator substantially aid training for inference on real-world data.
EgoSim code & MultiEgoView dataset: this https URL 

**Abstract (ZH)**: 计算机视觉中的第一人称任务研究主要集中在头戴式相机上，如鱼眼相机或嵌入于沉浸式头戴设备内部的相机。我们认为随着光学传感器的不断微型化，相机将被集成到更多种类的穿戴设备中，并位于身体的各个部位。这将为现有的一些计算机视觉任务带来新的视角，特别是在人体动作追踪、姿态估计或行为识别方面——尤其是对下肢的识别，下肢通常被遮挡。
在本文中，我们提出了EgoSim，这是一种新型的穿戴式相机模拟器，能够从穿戴者身体的不同视角生成逼真的第一人称渲染图像。EgoSim的一个关键特点在于它使用真实的运动捕捉数据来渲染运动伪影，特别是在手臂或腿部佩戴的相机中更为显著。此外，我们还介绍了MultiEgoView数据集，该数据集包含了六种不同位置的穿戴式相机录制的视角第一人称视频，以及多种活动中每个活动的地面真实全身材三维姿态。具体而言，我们从AMASS运动序列中提取了119小时的数据，这些数据来源于四个高质量虚拟环境，并通过13名参与者佩戴的六台GoPro相机和Xsens运动捕捉套装获得的真实世界运动数据进行了补充。

我们通过训练一个端到端的仅依赖视频的人体三维姿态估计网络展示了EgoSim的有效性。通过对领域差距的分析，我们展示了我们的数据集和模拟器在训练中对真实世界数据推理的帮助。

EgoSim代码与MultiEgoView数据集：[此链接](此链接请根据实际情况填写) 

---
# Which Contributions Deserve Credit? Perceptions of Attribution in Human-AI Co-Creation 

**Title (ZH)**: 哪些贡献值得认可？人类与AI协同创作中归因的感知 

**Authors**: Jessica He, Stephanie Houde, Justin D. Weisz  

**Link**: [PDF](https://arxiv.org/pdf/2502.18357)  

**Abstract**: AI systems powered by large language models can act as capable assistants for writing and editing. In these tasks, the AI system acts as a co-creative partner, making novel contributions to an artifact-under-creation alongside its human partner(s). One question that arises in these scenarios is the extent to which AI should be credited for its contributions. We examined knowledge workers' views of attribution through a survey study (N=155) and found that they assigned different levels of credit across different contribution types, amounts, and initiative. Compared to a human partner, we observed a consistent pattern in which AI was assigned less credit for equivalent contributions. Participants felt that disclosing AI involvement was important and used a variety of criteria to make attribution judgments, including the quality of contributions, personal values, and technology considerations. Our results motivate and inform new approaches for crediting AI contributions to co-created work. 

**Abstract (ZH)**: 由大规模语言模型驱动的AI系统可以作为撰写和编辑的强大助手。在这些任务中，AI系统充当了一位创造性的合作伙伴，与人类同伴共同为正在创作的作品做出创新性的贡献。在这种场景中，一个出现的问题是，AI的贡献应得到多大程度的认可。我们通过对知识工作者进行问卷调查（N=155）来探讨署名问题，并发现他们根据不同类型的贡献、贡献量以及主动性程度，分配了不同程度的信用。与人类同伴相比，我们观察到一个一致的趋势：对于等同的贡献，AI获得的信用较低。参与者认为披露AI的参与非常重要，并使用多种标准来做出贡献的署名判断，包括贡献的质量、个人价值观和技术考虑。我们的研究结果促使并启示了对于共同创作作品中AI贡献的新途径的认可方法。 

---
# From Vision to Sound: Advancing Audio Anomaly Detection with Vision-Based Algorithms 

**Title (ZH)**: 从视觉到声音：基于视觉算法推进音频异常检测 

**Authors**: Manuel Barusco, Francesco Borsatti, Davide Dalle Pezze, Francesco Paissan, Elisabetta Farella, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2502.18328)  

**Abstract**: Recent advances in Visual Anomaly Detection (VAD) have introduced sophisticated algorithms leveraging embeddings generated by pre-trained feature extractors. Inspired by these developments, we investigate the adaptation of such algorithms to the audio domain to address the problem of Audio Anomaly Detection (AAD). Unlike most existing AAD methods, which primarily classify anomalous samples, our approach introduces fine-grained temporal-frequency localization of anomalies within the spectrogram, significantly improving explainability. This capability enables a more precise understanding of where and when anomalies occur, making the results more actionable for end users. We evaluate our approach on industrial and environmental benchmarks, demonstrating the effectiveness of VAD techniques in detecting anomalies in audio signals. Moreover, they improve explainability by enabling localized anomaly identification, making audio anomaly detection systems more interpretable and practical. 

**Abstract (ZH)**: 近年来，视觉异常检测（VAD）方面取得了显著进步，引入了利用预训练特征提取器生成的嵌入的复杂算法。受这些进展的启发，我们研究了将此类算法应用于音频域，以解决音频异常检测（AAD）问题。与大多数现有的AAD方法主要对异常样本进行分类不同，我们的方法在频谱图中引入了异常的细粒度时间和频率定位，显著改善了可解释性。这一能力使我们能够更精确地了解异常发生的位置和时间，从而使得最终用户可以采取更有效的行动。我们在工业和环境基准上评估了我们的方法，展示了VAD技术在检测音频信号异常方面的能力。此外，通过对局部异常的识别，这些方法提高了可解释性，使得音频异常检测系统更具解释性和实用性。 

---
# Smart and Efficient IoT-Based Irrigation System Design: Utilizing a Hybrid Agent-Based and System Dynamics Approach 

**Title (ZH)**: 基于物联网的智能高效灌溉系统设计：利用混合代理基础与系统动力学方法 

**Authors**: Taha Ahmadi Pargo, Mohsen Akbarpour Shirazi, Dawud Fadai  

**Link**: [PDF](https://arxiv.org/pdf/2502.18298)  

**Abstract**: Regarding problems like reduced precipitation and an increase in population, water resource scarcity has become one of the most critical problems in modern-day societies, as a consequence, there is a shortage of available water resources for irrigation in arid and semi-arid countries. On the other hand, it is possible to utilize modern technologies to control irrigation and reduce water loss. One of these technologies is the Internet of Things (IoT). Despite the possibility of using the IoT in irrigation control systems, there are complexities in designing such systems. Considering this issue, it is possible to use agent-oriented software engineering (AOSE) methodologies to design complex cyber-physical systems such as IoT-based systems. In this research, a smart irrigation system is designed based on Prometheus AOSE methodology, to reduce water loss by maintaining soil moisture in a suitable interval. The designed system comprises sensors, a central agent, and irrigation nodes. These agents follow defined rules to maintain soil moisture at a desired level cooperatively. For system simulation, a hybrid agent-based and system dynamics model was designed. In this hybrid model, soil moisture dynamics were modeled based on the system dynamics approach. The proposed model, was implemented in AnyLogic computer simulation software. Utilizing the simulation model, irrigation rules were examined. The system's functionality in automatic irrigation mode was tested based on a 256-run, fractional factorial design, and the effects of important factors such as soil properties on total irrigated water and total operation time were analyzed. Based on the tests, the system consistently irrigated nearly optimal water amounts in all tests. Moreover, the results were also used to minimize the system's energy consumption by reducing the system's operational time. 

**Abstract (ZH)**: 关于如降水量减少和人口增加等问题，水资源短缺已成为现代社会中最为关键的问题之一。因此，在干旱和半干旱国家中，可用于灌溉的水资源短缺问题日益严重。另一方面，可以利用现代技术来控制灌溉并减少水分损失。其中一种技术就是物联网（IoT）。尽管可以在灌溉控制系统中使用IoT，但在设计此类系统时仍存在复杂性。为了解决这一问题，可以采用面向代理的软件工程（AOSE）方法来设计复杂的网络物理系统，如基于IoT的系统。在本研究中，我们基于普罗米修斯AOSE方法设计了一个智能灌溉系统，通过维持土壤湿度在适当区间来减少水分损失。所设计的系统包括传感器、中央代理和灌溉节点。这些代理遵循定义的规则协作维持土壤湿度在所需水平。为了系统仿真，我们设计了一个混合基于代理和系统动力学的模型。在这个混合模型中，基于系统动力学方法模拟能量水分动态学。所提出的模型在AnyLogic计算机仿真软件中实施。利用仿真模型，我们检查了灌溉规则，并基于256次部分因子设计测试系统的自动灌溉模式功能。我们分析了诸如土壤性质等重要因素对总灌溉水量和总运行时间的影响。根据测试结果，该系统在所有测试中均能自动灌溉近乎最优的水量。此外，测试结果还被用于通过减少系统运行时间来最小化系统的能耗。 

---
# Mixing Any Cocktail with Limited Ingredients: On the Structure of Payoff Sets in Multi-Objective MDPs and its Impact on Randomised Strategies 

**Title (ZH)**: 使用有限原料调制任意鸡尾酒：多目标MDP中收益集的结构及其对随机化策略的影响 

**Authors**: James C. A. Main, Mickael Randour  

**Link**: [PDF](https://arxiv.org/pdf/2502.18296)  

**Abstract**: We consider multi-dimensional payoff functions in Markov decision processes, and ask whether a given expected payoff vector can be achieved or not. In general, pure strategies (i.e., not resorting to randomisation) do not suffice for this problem.
We study the structure of the set of expected payoff vectors of all strategies given a multi-dimensional payoff function and its consequences regarding randomisation requirements for strategies. In particular, we prove that for any payoff for which the expectation is well-defined under all strategies, it is sufficient to mix (i.e., randomly select a pure strategy at the start of a play and committing to it for the rest of the play) finitely many pure strategies to approximate any expected payoff vector up to any precision. Furthermore, for any payoff for which the expected payoff is finite under all strategies, any expected payoff can be obtained exactly by mixing finitely many strategies. 

**Abstract (ZH)**: 我们考虑马尔可夫决策过程中多维收益函数，并探讨是否可以实现给定的期望收益向量。一般来说，纯策略（即不使用随机化）不足以解决这个问题。

我们研究在给定多维收益函数及其所有策略的期望收益向量结构情况下，随机化策略的必要性。特别地，我们证明对于所有策略下期望值定义良好的任何收益，通过混合（即在游戏开始时随机选择一个纯策略，并在整个游戏过程中承诺遵循该策略）有限多个纯策略，可以将任何期望收益向量逼近到任意精度。此外，对于所有策略下期望收益有限的任何收益，都可以通过混合有限多个策略精确实现任何期望收益。 

---
# AMPO: Active Multi-Preference Optimization 

**Title (ZH)**: AMPO：主动多偏好优化 

**Authors**: Taneesh Gupta, Rahul Madhavan, Xuchao Zhang, Chetan Bansal, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2502.18293)  

**Abstract**: Multi-preference optimization enriches language-model alignment beyond pairwise preferences by contrasting entire sets of helpful and undesired responses, thereby enabling richer training signals for large language models. During self-play alignment, these models often produce numerous candidate answers per query, rendering it computationally infeasible to include all responses in the training objective. In this work, we propose $\textit{Active Multi-Preference Optimization}$ (AMPO), a novel approach that combines on-policy generation, a multi-preference group-contrastive loss, and active subset selection. Specifically, we score and embed large candidate pools of responses and then select a small, yet informative, subset that covers reward extremes and distinct semantic clusters for preference optimization. Our contrastive training scheme is capable of identifying not only the best and worst answers but also subtle, underexplored modes that are crucial for robust alignment. Theoretically, we provide guarantees for expected reward maximization using our active selection method, and empirically, AMPO achieves state-of-the-art results on $\textit{AlpacaEval}$ using Llama 8B. 

**Abstract (ZH)**: 多偏好优化通过对比整个有益和不 desired 反应集，从而丰富了语言模型对齐（alignment）的内容，超越了仅针对成对偏好进行优化的方法，进而为大型语言模型提供了更丰富的训练信号。在自我博弈对齐过程中，这些模型往往会生成多个候选答案，这使得将所有响应包含在训练目标中从计算上变得不可行。在本文中，我们提出了**主动多偏好优化（Active Multi-Preference Optimization, AMPO）**，这是一种结合了在策略生成方法、多偏好群体对比损失以及主动子集选择的新颖方法。具体而言，我们对大量的候选回答实施评分和嵌入，并选择了一个小而有信息量的子集，该子集涵盖了奖励极值以及不同的语义簇，以优化偏好。我们的对比训练方案不仅能够识别最好的和最差的回答，还能够识别一些关键的、尚未充分探索的模式。从理论上讲，我们提供了关于期望奖励最大化的保证，以我们的主动选择方法为依据。在实验中，AMPO 在使用 Llama 8B 进行 AlpacaEval 任务时达到了最先进的性能。 

---
# A Reverse Mamba Attention Network for Pathological Liver Segmentation 

**Title (ZH)**: 一种反向眼镜蛇注意力网络用于病理肝脏分割 

**Authors**: Jun Zeng, Ulas Bagci, Debesh Jha  

**Link**: [PDF](https://arxiv.org/pdf/2502.18232)  

**Abstract**: We present RMA-Mamba, a novel architecture that advances the capabilities of vision state space models through a specialized reverse mamba attention module (RMA). The key innovation lies in RMA-Mamba's ability to capture long-range dependencies while maintaining precise local feature representation through its hierarchical processing pipeline. By integrating Vision Mamba (VMamba)'s efficient sequence modeling with RMA's targeted feature refinement, our architecture achieves superior feature learning across multiple scales. This dual-mechanism approach enables robust handling of complex morphological patterns while maintaining computational efficiency. We demonstrate RMA-Mamba's effectiveness in the challenging domain of pathological liver segmentation (from both CT and MRI), where traditional segmentation approaches often fail due to tissue variations. When evaluated on a newly introduced cirrhotic liver dataset (CirrMRI600+) of T2-weighted MRI scans, RMA-Mamba achieves the state-of-the-art performance with a Dice coefficient of 92.08%, mean IoU of 87.36%, and recall of 92.96%. The architecture's generalizability is further validated on the cancerous liver segmentation from CT scans (LiTS: Liver Tumor Segmentation dataset), yielding a Dice score of 92.9% and mIoU of 88.99%. The source code of the proposed RMA-Mamba is available at this https URL. 

**Abstract (ZH)**: 我们提出了RMA-Mamba，这是一种创新的架构，通过专门的反眼镜王蛇注意力模块（RMA）来提升视觉状态空间模型的能力。其核心创新在于，RMA-Mamba能够通过分层处理管道捕捉长范围依赖关系，同时保持精确的局部特征表示。通过将Vision Mamba (VMamba) 的高效序列建模与RMA的目标特征精修相结合，我们的架构在多个尺度上实现了优越的特征学习效果。这种双机制方法能够稳健地处理复杂的形态学模式，同时保持计算效率。我们展示了RMA-Mamba在病理肝脏分割（来自CT和MRI的图像）这一具有挑战性的领域的有效性，这是传统分割方法由于组织变异常常无法有效解决的问题。在新的cirrhotic肝脏数据集（CirrMRI600+）中，RMA-Mamba在T2加权MRI扫描上达到了最先进的性能，Dice系数为92.08%，平均交并比为87.36%，召回率为92.96%。该架构的泛化能力进一步在CT扫描的癌变肝脏分割（LiTS：肝脏肿瘤分割数据集）中得到验证，达到了Dice分数为92.9%，平均IoU为88.99%。提出的研究代码已在此链接处提供：[链接地址]。 

---
# Liver Cirrhosis Stage Estimation from MRI with Deep Learning 

**Title (ZH)**: 使用深度学习从MRI图像估计肝硬化阶段 

**Authors**: Jun Zeng, Debesh Jha, Ertugrul Aktas, Elif Keles, Alpay Medetalibeyoglu, Matthew Antalek, Amir A. Borhani, Daniela P. Ladner, Gorkem Durak, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2502.18225)  

**Abstract**: We present an end-to-end deep learning framework for automated liver cirrhosis stage estimation from multi-sequence MRI. Cirrhosis is the severe scarring (fibrosis) of the liver and a common endpoint of various chronic liver diseases. Early diagnosis is vital to prevent complications such as decompensation and cancer, which significantly decreases life expectancy. However, diagnosing cirrhosis in its early stages is challenging, and patients often present with life-threatening complications. Our approach integrates multi-scale feature learning with sequence-specific attention mechanisms to capture subtle tissue variations across cirrhosis progression stages. Using CirrMRI600+, a large-scale publicly available dataset of 628 high-resolution MRI scans from 339 patients, we demonstrate state-of-the-art performance in three-stage cirrhosis classification. Our best model achieves 72.8% accuracy on T1W and 63.8% on T2W sequences, significantly outperforming traditional radiomics-based approaches. Through extensive ablation studies, we show that our architecture effectively learns stage-specific imaging biomarkers. We establish new benchmarks for automated cirrhosis staging and provide insights for developing clinically applicable deep learning systems. The source code will be available at this https URL. 

**Abstract (ZH)**: 我们提出了一种端到端的深度学习框架，用于从多序列MRI自动估计肝硬化阶段。肝硬化是肝脏严重的瘢痕（纤维化），是多种慢性肝病的常见终末阶段。早期诊断对于预防并发症（如失代偿和癌症）至关重要，这些并发症显著降低了生存率。然而，在肝硬化的早期阶段进行诊断极具挑战性，且患者常会出现危及生命的并发症。我们的方法结合了多尺度特征学习和序列特定的注意机制，以捕捉肝硬化进展不同阶段之间的细微组织变化。使用CirrMRI600+数据集，该数据集包含来自339名患者的628张高分辨率MRI扫描图像，并且是公开可用的，我们展示了在三阶段肝硬化分类中的最佳性能。我们最好的模型在T1W序列上的准确率为72.8%，在T2W序列上的准确率为63.8%，显著优于传统的基于影像组学的方法。通过广泛的消融研究，我们展示了我们的架构能够有效学习特定于阶段的影像生物标志物。我们建立了自动肝硬化分期的新基准，并为开发临床适用的深度学习系统提供了见解。源代码将在此处提供：[此 https URL]。 

---
# UASTrack: A Unified Adaptive Selection Framework with Modality-Customization in Single Object Tracking 

**Title (ZH)**: UASTrack：一种具有模态自适应定制化的统一单目标跟踪选择框架 

**Authors**: He Wang, Tianyang Xu, Zhangyong Tang, Xiao-Jun Wu, Josef Kittler  

**Link**: [PDF](https://arxiv.org/pdf/2502.18220)  

**Abstract**: Multi-modal tracking is essential in single-object tracking (SOT), as different sensor types contribute unique capabilities to overcome challenges caused by variations in object appearance. However, existing unified RGB-X trackers (X represents depth, event, or thermal modality) either rely on the task-specific training strategy for individual RGB-X image pairs or fail to address the critical importance of modality-adaptive perception in real-world applications. In this work, we propose UASTrack, a unified adaptive selection framework that facilitates both model and parameter unification, as well as adaptive modality discrimination across various multi-modal tracking tasks. To achieve modality-adaptive perception in joint RGB-X pairs, we design a Discriminative Auto-Selector (DAS) capable of identifying modality labels, thereby distinguishing the data distributions of auxiliary modalities. Furthermore, we propose a Task-Customized Optimization Adapter (TCOA) tailored to various modalities in the latent space. This strategy effectively filters noise redundancy and mitigates background interference based on the specific characteristics of each modality. Extensive comparisons conducted on five benchmarks including LasHeR, GTOT, RGBT234, VisEvent, and DepthTrack, covering RGB-T, RGB-E, and RGB-D tracking scenarios, demonstrate our innovative approach achieves comparative performance by introducing only additional training parameters of 1.87M and flops of 1.95G. The code will be available at this https URL. 

**Abstract (ZH)**: 多模态跟踪在单目标跟踪（SOT）中至关重要，因为不同传感器类型能够提供独特的功能，以克服由于目标外观变化引起的各种挑战。然而，现有的统一RGB-X跟踪器（X代表深度、事件或热成像模态）要么依赖于针对单一RGB-X图像对的专业任务训练策略，要么未能解决实际应用中模态自适应感知的关键重要性。在本文中，我们提出了一种统一自适应选择框架UASTrack，该框架促进了模型和参数的统一，并实现了各种多模态跟踪任务中的自适应模态区分。为在联合RGB-X图像对中实现模态自适应感知，我们设计了一种鉴别性自动选择器（DAS），能够识别模态标签，从而区分辅助模态的数据分布。此外，我们提出了一种任务自定义优化适配器（TCOA），适用于潜空间中的各种模态。该策略基于每个模态的具体特性有效过滤掉噪声冗余并减轻背景干扰。在包括LasHeR、GTOT、RGBT234、VisEvent和DepthTrack在内的五个基准数据集上的广泛比较，涵盖了RGB-T、RGB-E和RGB-D跟踪场景，证明了我们的创新方法通过增加仅1.87M的训练参数和1.95G的运算量跃点（FLOPS），实现了可比的性能。相关代码将在以下链接处提供：<该链接处>。

请注意，“该链接处”应该替换为实际的URL链接。 

---
# FLARE: A Framework for Stellar Flare Forecasting using Stellar Physical Properties and Historical Records 

**Title (ZH)**: FLARE：一种基于恒星物理性质和历史记录的耀斑预报框架 

**Authors**: Bingke Zhu, Xiaoxiao Wang, Minghui Jia, Yihan Tao, Xiao Kong, Ali Luo, Yingying Chen, Ming Tang, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18218)  

**Abstract**: Stellar flare events are critical observational samples for astronomical research; however, recorded flare events remain limited. Stellar flare forecasting can provide additional flare event samples to support research efforts. Despite this potential, no specialized models for stellar flare forecasting have been proposed to date. In this paper, we present extensive experimental evidence demonstrating that both stellar physical properties and historical flare records are valuable inputs for flare forecasting tasks. We then introduce FLARE (Forecasting Light-curve-based Astronomical Records via features Ensemble), the first-of-its-kind large model specifically designed for stellar flare forecasting. FLARE integrates stellar physical properties and historical flare records through a novel Soft Prompt Module and Residual Record Fusion Module. Our experiments on the publicly available Kepler light curve dataset demonstrate that FLARE achieves superior performance compared to other methods across all evaluation metrics. Finally, we validate the forecast capability of our model through a comprehensive case study. 

**Abstract (ZH)**: 恒星耀斑事件是天文学研究中的关键观测样本，然而记录到的耀斑事件仍然有限。恒星耀斑预报可以提供额外的耀斑事件样本以支持研究工作。尽管具有这一潜力，但目前尚未提出专门用于恒星耀斑预报的模型。在本文中，我们通过大量的实验证据展示了恒星物理属性和历史耀斑记录在耀斑预报任务中的价值。随后，我们介绍了FLARE（利用特征集成预测基于光曲线的天文记录），这是第一个专门设计用于恒星耀斑预报的大规模模型。FLARE通过一种新颖的Soft Prompt模块和残差记录融合模块，整合了恒星物理属性和历史耀斑记录。我们在公开发布的开普勒光曲线数据集上进行的实验表明，FLARE在所有评估指标上都优于其他方法。最后，我们通过全面的案例研究验证了模型的预报能力。 

---
# LAG: LLM agents for Leaderboard Auto Generation on Demanding 

**Title (ZH)**: LAG: 用于苛刻需求下排行榜自动生成的大规模语言模型代理 

**Authors**: Jian Wu, Jiayu Zhang, Dongyuan Li, Linyi Yang, Aoxiao Zhong, Renhe Jiang, Qingsong Wen, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18209)  

**Abstract**: This paper introduces Leaderboard Auto Generation (LAG), a novel and well-organized framework for automatic generation of leaderboards on a given research topic in rapidly evolving fields like Artificial Intelligence (AI). Faced with a large number of AI papers updated daily, it becomes difficult for researchers to track every paper's proposed methods, experimental results, and settings, prompting the need for efficient automatic leaderboard construction. While large language models (LLMs) offer promise in automating this process, challenges such as multi-document summarization, leaderboard generation, and experiment fair comparison still remain under exploration. LAG solves these challenges through a systematic approach that involves the paper collection, experiment results extraction and integration, leaderboard generation, and quality evaluation. Our contributions include a comprehensive solution to the leaderboard construction problem, a reliable evaluation method, and experimental results showing the high quality of leaderboards. 

**Abstract (ZH)**: 本文介绍了Leaderboard 自动生成 (LAG) 框架，这是一种新颖且井然有序的框架，用于在人工智能（AI）等快速发展的研究领域中自动生成给定研究主题的排行榜。面对每日更新的大量AI论文，研究人员难以追踪每篇论文提出的算法、实验结果和设置，这促使了高效自动排行榜构建的需求。尽管大规模语言模型（LLMs）在自动化这一过程中提供了希望，但在多文档总结、排行榜生成和实验公正比较等方面仍面临诸多挑战。LAG 通过一个系统化的流程来解决这些挑战，该流程包括论文收集、实验结果提取与整合、排行榜生成以及质量评估。我们的贡献包括针对排行榜构建问题的综合解决方案、可靠的评估方法以及展示排行榜高质量的实验结果。 

---
# DenoMAE2.0: Improving Denoising Masked Autoencoders by Classifying Local Patches 

**Title (ZH)**: DenoMAE2.0：通过分类局部 patches 提高去噪掩蔽自编码器性能 

**Authors**: Atik Faysal, Mohammad Rostami, Taha Boushine, Reihaneh Gh. Roshan, Huaxia Wang, Nikhil Muralidhar  

**Link**: [PDF](https://arxiv.org/pdf/2502.18202)  

**Abstract**: We introduce DenoMAE2.0, an enhanced denoising masked autoencoder that integrates a local patch classification objective alongside traditional reconstruction loss to improve representation learning and robustness. Unlike conventional Masked Autoencoders (MAE), which focus solely on reconstructing missing inputs, DenoMAE2.0 introduces position-aware classification of unmasked patches, enabling the model to capture fine-grained local features while maintaining global coherence. This dual-objective approach is particularly beneficial in semi-supervised learning for wireless communication, where high noise levels and data scarcity pose significant challenges. We conduct extensive experiments on modulation signal classification across a wide range of signal-to-noise ratios (SNRs), from extremely low to moderately high conditions and in a low data regime. Our results demonstrate that DenoMAE2.0 surpasses its predecessor, Deno-MAE, and other baselines in both denoising quality and downstream classification accuracy. DenoMAE2.0 achieves a 1.1% improvement over DenoMAE on our dataset and 11.83%, 16.55% significant improved accuracy gains on the RadioML benchmark, over DenoMAE, for constellation diagram classification of modulation signals. 

**Abstract (ZH)**: 我们介绍了DenoMAE2.0，这是一种增强的去噪掩蔽自编码器，它结合了局部(patch)分类目标与传统的重建损失，以此来提高表示学习能力和鲁棒性。与传统的掩蔽自编码器(MAE)仅专注于重建缺失输入不同，DenoMAE2.0引入了对未掩蔽patches的位置感知分类，使模型能够捕捉到精细的局部特征同时保持全局一致性。这种双重目标方法特别适用于无线通信中的半监督学习，在这种场景下，高噪声水平和数据稀缺性构成了显著的挑战。我们对各种信噪比(SNR)条件下的调制信号分类进行了广泛的实验，从极低信噪比到中等信噪比范围，以及在数据稀缺的条件下。实验结果表明，DenoMAE2.0在去噪质量和下游分类准确性方面都超越了其前身Deno-MAE以及其它基准方法。DenoMAE2.0在我们数据集上的去噪性能提高了1.1%，在RadioML基准测试中，对于调制信号星座图分类的准确性提高了11.83%和16.55%（这些提高具有显著性差异）。 

---
# VesselSAM: Leveraging SAM for Aortic Vessel Segmentation with LoRA and Atrous Attention 

**Title (ZH)**: VesselSAM：利用SAM进行主动脉血管分割的LoRA和空洞注意力增强方法 

**Authors**: Adnan Iltaf, Rayan Merghani Ahmed, Bin Li, Shoujun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.18185)  

**Abstract**: Medical image segmentation is crucial for clinical diagnosis and treatment planning, particularly for complex anatomical structures like vessels. In this work, we propose VesselSAM, a modified version of the Segmentation Anything Model (SAM), specifically designed for aortic vessel segmentation. VesselSAM incorporates AtrousLoRA, a novel module that combines Atrous Attention with Low-Rank Adaptation (LoRA), to improve segmentation performance. Atrous Attention enables the model to capture multi-scale contextual information, preserving both fine local details and broader global context. At the same time, LoRA facilitates efficient fine-tuning of the frozen SAM image encoder, reducing the number of trainable parameters and ensuring computational efficiency. We evaluate VesselSAM on two challenging datasets: the Aortic Vessel Tree (AVT) dataset and the Type-B Aortic Dissection (TBAD) dataset. VesselSAM achieves state-of-the-art performance with DSC scores of 93.50\%, 93.25\%, 93.02\%, and 93.26\% across multiple medical centers. Our results demonstrate that VesselSAM delivers high segmentation accuracy while significantly reducing computational overhead compared to existing large-scale models. This development paves the way for enhanced AI-based aortic vessel segmentation in clinical environments. The code and models will be released at this https URL. 

**Abstract (ZH)**: 医学图像分割对于临床诊断和治疗规划至关重要，尤其是对于复杂的解剖结构如血管。在本文中，我们提出了一种针对主动脉血管分割的改进版本——VesselSAM，它是分割一切一切内容模型（SAM）的变体。VesselSAM结合了新的AtrousLoRA模块，该模块将空洞注意力与低秩适应（LoRA）结合起来，以提高分割性能。空洞注意力使模型能够捕捉多尺度上下文信息，同时保留精细的局部细节和更广泛的全局上下文。与此同时，LoRA促进了冻结SAM图像编码器的有效微调，减少了可训练参数的数量并确保了计算效率。我们在两个具有挑战性的数据集——主动脉血管树（AVT）数据集和B型主动脉夹层（TBAD）数据集上评估了VesselSAM。在多个医学中心，VesselSAM实现了最先进的性能， Dice 得分分别为93.50%，93.25%，93.02%，和93.26%。我们的结果表明，VesselSAM在保持高分割准确性的前提下，显著降低了与现有大型模型相比的计算开销。该进步为在临床环境中实现增强的基于AI的主动脉血管分割铺平了道路。代码和模型将在https://released链接处公开。 

---
# Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs 

**Title (ZH)**: 《问题解决了吗？基于LLM的布局丰富文档的信息提取设计空间》 

**Authors**: Gaye Colakoglu, Gürkan Solmaz, Jonathan Fürst  

**Link**: [PDF](https://arxiv.org/pdf/2502.18179)  

**Abstract**: This paper defines and explores the design space for information extraction (IE) from layout-rich documents using large language models (LLMs). The three core challenges of layout-aware IE with LLMs are 1) data structuring, 2) model engagement, and 3) output refinement. Our study delves into the sub-problems within these core challenges, such as input representation, chunking, prompting, and selection of LLMs and multimodal models. It examines the outcomes of different design choices through a new layout-aware IE test suite, benchmarking against the state-of-art (SoA) model LayoutLMv3. The results show that the configuration from one-factor-at-a-time (OFAT) trial achieves near-optimal results with 14.1 points F1-score gain from the baseline model, while full factorial exploration yields only a slightly higher 15.1 points gain at around 36x greater token usage. We demonstrate that well-configured general-purpose LLMs can match the performance of specialized models, providing a cost-effective alternative. Our test-suite is freely available at this https URL. 

**Abstract (ZH)**: 本文定义并探索了使用大规模语言模型（LLMs）从布局丰富的文档中提取信息（IE）的设计空间。布局感知IE使用LLMs面临的三大核心挑战包括：1) 数据结构化，2) 模型互动，以及3) 输出润色。我们的研究深入探讨了这些核心挑战下的子问题，例如输入表示、分块、提示以及LLMs和多模态模型的选择。通过一个创新的布局感知IE测试套件，我们将结果与当前最先进的模型（LayoutLMv3）进行基准测试。结果表明，单因素一次性试验（one-factor-at-a-time, OFAT）的配置达到了近最优效果，基线模型的F1分数提高了14.1分，而全方位试验尽管在36倍于全量级的标记使用下，仅提高了略微高一点的15.1分。我们证明，配置良好的通用模型可以与专用模型匹配，提供一种成本效益更高的替代方案。我们的测试套件可在此处免费获取：[此 https URL]。 

---
# CLIPure: Purification in Latent Space via CLIP for Adversarially Robust Zero-Shot Classification 

**Title (ZH)**: CLIPure：通过CLIP在潜在空间中进行净化以实现对抗稳健的零-shot分类 

**Authors**: Mingkun Zhang, Keping Bi, Wei Chen, Jiafeng Guo, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.18176)  

**Abstract**: In this paper, we aim to build an adversarially robust zero-shot image classifier. We ground our work on CLIP, a vision-language pre-trained encoder model that can perform zero-shot classification by matching an image with text prompts ``a photo of a <class-name>.''. Purification is the path we choose since it does not require adversarial training on specific attack types and thus can cope with any foreseen attacks. We then formulate purification risk as the KL divergence between the joint distributions of the purification process of denoising the adversarial samples and the attack process of adding perturbations to benign samples, through bidirectional Stochastic Differential Equations (SDEs). The final derived results inspire us to explore purification in the multi-modal latent space of CLIP. We propose two variants for our CLIPure approach: CLIPure-Diff which models the likelihood of images' latent vectors with the DiffusionPrior module in DaLLE-2 (modeling the generation process of CLIP's latent vectors), and CLIPure-Cos which models the likelihood with the cosine similarity between the embeddings of an image and ``a photo of a.''. As far as we know, CLIPure is the first purification method in multi-modal latent space and CLIPure-Cos is the first purification method that is not based on generative models, which substantially improves defense efficiency. We conducted extensive experiments on CIFAR-10, ImageNet, and 13 datasets that previous CLIP-based defense methods used for evaluating zero-shot classification robustness. Results show that CLIPure boosts the SOTA robustness by a large margin, e.g., from 71.7% to 91.1% on CIFAR10, from 59.6% to 72.6% on ImageNet, and 108% relative improvements of average robustness on the 13 datasets over previous SOTA. The code is available at this https URL. 

**Abstract (ZH)**: 在本文中，我们旨在构建一个对抗鲁棒的零样本图像分类器。我们基于CLIP（一个预训练的视觉-语言编码器模型），该模型可以通过将图像与文本提示（如“一张<类别名称>的照片”）进行匹配来进行零样本分类。我们选择纯化这条路线，因为它不需要对特定攻击类型进行对抗训练，因此可以应对任何预期的攻击。随后，我们将纯化风险定义为去噪对抗样本的纯化进程与添加扰动到 benign 样本的攻击过程之间联合分布的 KL 散度，通过双向随机微分方程（bidirectional SDEs）进行形式化。最终推导出的结果促使我们在CLIP的多模态潜空间中探索纯化方法。我们提出了CLIPure方法的两种变体：CLIPure-Diff 使用 DaLLE-2 中的 DiffusionPrior 模块来建模图像潜向量的可能性（该模块用于建模CLIP的潜向量生成过程），以及CLIPure-Cos 使用图像嵌入与“一张<类别名称>的照片”的余弦相似度来建模可能性。据我们所知，CLIPure 是第一个应用于多模态潜空间的纯化方法，而CLIPure-Cos 是第一个不基于生成模型的纯化方法，显著提高了防护效率。我们在CIFAR-10、ImageNet和13个之前用于评估零样本分类鲁棒性的CLIP基础防御方法所用的数据集上进行了大量实验。结果表明，CLIPure 大幅提升了最先进（State-of-the-Art）的鲁棒性，例如，CIFAR10 上从71.7%提高到91.1%，ImageNet 上从59.6%提高到72.6%，在13个数据集上的平均鲁棒性相对提升达108%。该代码可在以下网址获取：[提供网址]。 

---
# SECURA: Sigmoid-Enhanced CUR Decomposition with Uninterrupted Retention and Low-Rank Adaptation in Large Language Models 

**Title (ZH)**: SECURA：增强的Sigmoid CUR分解方法，具备不间断保留和低秩适应能力，适用于大规模语言模型 

**Authors**: Zhang Yuxuan, Li Ruizhe  

**Link**: [PDF](https://arxiv.org/pdf/2502.18168)  

**Abstract**: With the rapid development of large language models (LLMs), fully fine-tuning (FT) these models has become increasingly impractical due to the high computational demands. Additionally, FT can lead to catastrophic forgetting. As an alternative, Low-Rank Adaptation (LoRA) has been proposed, which fine-tunes only a small subset of parameters, achieving similar performance to FT while significantly reducing resource requirements. However, since LoRA inherits FT's design, the issue of catastrophic forgetting remains.
To address these challenges, we propose SECURA: Sigmoid-Enhanced CUR Decomposition LoRA, a novel parameter-efficient fine-tuning (PEFT) variant that mitigates catastrophic forgetting while improving fine-tuning performance. Our method introduces a new normalization technique, SigNorm, to enhance parameter retention and overall performance.
SECURA has been evaluated on a variety of tasks, including mathematical problem-solving (GSM8K), challenging question-answering (CNNDM), translation (NewsDE), and complex multiple-choice reasoning (LogiQA). Experimental results show that SECURA achieves an average fine-tuning improvement of 3.59% across four multiple-choice question (MCQ) tasks and a 2.51% improvement across five question-answering (QA) tasks on models such as Gemma2 2b, Qwen2 1.5b, Qwen 2 7b, Llama3 8b, and Llama3.1 8b, compared to DoRA. Moreover, SECURA demonstrates superior knowledge retention capabilities, maintaining more than 70% accuracy on basic LLM knowledge across 16 continual learning tests, outperforming Experience Replay (ER), Sequential Learning (SEQ), EWC, I-LoRA, and CUR-LoRA. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的迅速发展，完全微调（FT）这些模型因计算需求高而变得越来越不实际。此外，FT 会导致灾难性遗忘。作为替代方案，低秩适应（LoRA）已经被提出，它仅微调参数子集，从而在显著减少资源需求的同时达到与 FT 相似的性能。然而，由于 LoRA 继承了 FT 的设计，灾难性遗忘的问题仍然存在。

为了应对这些挑战，我们提出了 SECURA：Sigmoid-Enhanced CUR 分解 LoRA，这是一种新颖的参数高效微调（PEFT）变体，能够在减少灾难性遗忘的同时改善微调性能。我们的方法引入了一种新的规范化技术——SigNorm，以增强参数保留和整体性能。

SECURA 在各类任务中进行了评估，包括数学问题求解（GSM8K）、具有挑战性的问答（CNNDM）、翻译（NewsDE）和复杂的多项选择推理（LogiQA）。实验结果显示，在 Gemma2 2b、Qwen2 1.5b、Qwen 2 7b、Llama3 8b 和 Llama3.1 8b 等模型中，SECURA 在四个多项选择问答（MCQ）任务上的微调改善平均为 3.59%，在五个问答（QA）任务上的改善为 2.51%，相较于 DoRA。此外，SECURA 在持续学习测试中的知识保留能力表现出色，在 16 次测试中保持了超过 70% 的基础 LLM 知识准确性，优于经验回放（ER）、顺序学习（SEQ）、EWC、I-LoRA 和 CUR-LoRA。 

---
# iTrash: Incentivized Token Rewards for Automated Sorting and Handling 

**Title (ZH)**: iTrash：激励性代币奖励机制下的自动分类与处理 

**Authors**: Pablo Ortega, Eduardo Castelló Ferrer  

**Link**: [PDF](https://arxiv.org/pdf/2502.18161)  

**Abstract**: As robotic systems (RS) become more autonomous, they are becoming increasingly used in small spaces and offices to automate tasks such as cleaning, infrastructure maintenance, or resource management. In this paper, we propose iTrash, an intelligent trashcan that aims to improve recycling rates in small office spaces. For that, we ran a 5 day experiment and found that iTrash can produce an efficiency increase of more than 30% compared to traditional trashcans. The findings derived from this work, point to the fact that using iTrash not only increase recyclying rates, but also provides valuable data such as users behaviour or bin usage patterns, which cannot be taken from a normal trashcan. This information can be used to predict and optimize some tasks in these spaces. Finally, we explored the potential of using blockchain technology to create economic incentives for recycling, following a Save-as-you-Throw (SAYT) model. 

**Abstract (ZH)**: 随着机器人系统（RS）的自主性逐渐增强，它们在狭小空间和办公室中被越来越多地用于自动化清洁、基础设施维护或资源管理等任务。本文提出了一种智能垃圾桶iTrash，旨在提升小型办公空间的回收率。为此，我们进行了一项为期5天的实验，并发现iTrash相比传统垃圾桶的效率提高了超过30%。本研究所得出的成果表明，使用iTrash不仅能够提升回收率，还能够提供有价值的数据，如用户行为或垃圾桶使用模式等，这些数据无法从普通垃圾桶中获取。这些信息可以用于预测和优化这些空间中的某些任务。最后，我们探讨了使用区块链技术通过“边扔边存”（Save-as-you-Throw, SAYT）模式创造经济激励的可能性，以促进回收行为。 

---
# Monitoring snow avalanches from SAR data with deep learning 

**Title (ZH)**: 使用深度学习技术从SAR数据监测雪崩 

**Authors**: Filippo Maria Bianchi, Jakob Grahn  

**Link**: [PDF](https://arxiv.org/pdf/2502.18157)  

**Abstract**: Snow avalanches present significant risks to human life and infrastructure, particularly in mountainous regions, making effective monitoring crucial. Traditional monitoring methods, such as field observations, are limited by accessibility, weather conditions, and cost. Satellite-borne Synthetic Aperture Radar (SAR) data has become an important tool for large-scale avalanche detection, as it can capture data in all weather conditions and across remote areas. However, traditional processing methods struggle with the complexity and variability of avalanches. This chapter reviews the application of deep learning for detecting and segmenting snow avalanches from SAR data. Early efforts focused on the binary classification of SAR images, while recent advances have enabled pixel-level segmentation, providing greater accuracy and spatial resolution. A case study using Sentinel-1 SAR data demonstrates the effectiveness of deep learning models for avalanche segmentation, achieving superior results over traditional methods. We also present an extension of this work, testing recent state-of-the-art segmentation architectures on an expanded dataset of over 4,500 annotated SAR images. The best-performing model among those tested was applied for large-scale avalanche detection across the whole of Norway, revealing important spatial and temporal patterns over several winter seasons. 

**Abstract (ZH)**: 雪崩对人类生命和基础设施构成重大威胁，尤其是在山区，因此有效的监测至关重要。传统的监测方法，如实地观察，受限于可及性、天气条件和成本。携带合成孔径雷达（SAR）的卫星数据已成为大规模雪崩检测的重要工具，因为它能够在各种天气条件下和偏远地区收集数据。然而，传统的处理方法难以应对雪崩的复杂性和变化性。本章综述了使用深度学习从SAR数据中检测和分割雪崩的应用。早期努力侧重于SAR图像的二分类，而近年来的进步使得像素级分割成为可能，提高了准确性和空间分辨率。使用Sentinel-1 SAR数据的案例研究表明，深度学习模型在雪崩分割方面的有效性，其结果优于传统方法。我们还呈现了这项工作的扩展，测试了最新最先进的分割架构在超过4,500张标注的SAR图像数据集上的表现。经过测试的最佳模型被应用于整个挪威的大规模雪崩检测，揭示了多个冬季季节中的重要空间和时间模式。 

---
# Can LLMs Explain Themselves Counterfactually? 

**Title (ZH)**: 大语言模型能否进行反事实解释？ 

**Authors**: Zahra Dehghanighobadi, Asja Fischer, Muhammad Bilal Zafar  

**Link**: [PDF](https://arxiv.org/pdf/2502.18156)  

**Abstract**: Explanations are an important tool for gaining insights into the behavior of ML models, calibrating user trust and ensuring regulatory compliance. Past few years have seen a flurry of post-hoc methods for generating model explanations, many of which involve computing model gradients or solving specially designed optimization problems. However, owing to the remarkable reasoning abilities of Large Language Model (LLMs), self-explanation, that is, prompting the model to explain its outputs has recently emerged as a new paradigm. In this work, we study a specific type of self-explanations, self-generated counterfactual explanations (SCEs). We design tests for measuring the efficacy of LLMs in generating SCEs. Analysis over various LLM families, model sizes, temperature settings, and datasets reveals that LLMs sometimes struggle to generate SCEs. Even when they do, their prediction often does not agree with their own counterfactual reasoning. 

**Abstract (ZH)**: 解释是理解机器学习模型行为的重要工具，有助于校准用户信任并确保合规性。近年来，涌现出了许多后验方法来生成模型解释，其中许多方法涉及计算模型梯度或解决特别设计的优化问题。然而，由于大型语言模型（LLMs）卓越的推理能力，促使模型自我解释，即提示模型解释其输出，已成为一种新的范式。在这项工作中，我们研究了一种特定类型的自我解释，即自动生成反事实解释（SCEs）。我们设计了测试来衡量LLMs生成SCEs的有效性。通过对各种LLM家族、模型大小、温度设置和数据集的分析表明，LLMs有时难以生成SCEs。即使生成了SCEs，它们的预测往往也不与其自身的反事实推理一致。 

---
# SASSHA: Sharpness-aware Adaptive Second-order Optimization with Stable Hessian Approximation 

**Title (ZH)**: SASSHA：感知尖锐度自适应二阶优化与稳定的海森矩阵近似 

**Authors**: Dahun Shin, Dongyeop Lee, Jinseok Chung, Namhoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.18153)  

**Abstract**: Approximate second-order optimization methods often exhibit poorer generalization compared to first-order approaches. In this work, we look into this issue through the lens of the loss landscape and find that existing second-order methods tend to converge to sharper minima compared to SGD. In response, we propose Sassha, a novel second-order method designed to enhance generalization by explicitly reducing sharpness of the solution, while stabilizing the computation of approximate Hessians along the optimization trajectory. In fact, this sharpness minimization scheme is crafted also to accommodate lazy Hessian updates, so as to secure efficiency besides flatness. To validate its effectiveness, we conduct a wide range of standard deep learning experiments where Sassha demonstrates its outstanding generalization performance that is comparable to, and mostly better than, other methods. We provide a comprehensive set of analyses including convergence, robustness, stability, efficiency, and cost. 

**Abstract (ZH)**: 相比较于一阶方法，近似二阶优化方法通常在泛化能力上表现较差。在这项工作中，我们通过损失景观的角度分析了这一问题，并发现现有的二阶方法倾向于收敛到比SGD更尖锐的极小值。为应对这一问题，我们提出了一种名为Sassha的新颖二阶方法，旨在通过显式减少解的尖锐度来提高泛化能力，并在优化轨迹中稳定地计算近似的海森矩阵。实际上，这一尖锐度最小化方案还被设计为能够适应懒更新的海森矩阵，旨在保证效率的同时实现平坦性。为了验证其有效性，我们在一系列标准的深度学习实验中进行了广泛测试，结果显示，Sassha在泛化性能上表现出色，与甚至优于其他方法的表现。我们提供了包括收敛性、鲁棒性、稳定性、效率和成本在内的全面分析。 

---
# A Real-time Spatio-Temporal Trajectory Planner for Autonomous Vehicles with Semantic Graph Optimization 

**Title (ZH)**: 面向自主车辆的实时时空轨迹规划器：基于语义图优化 

**Authors**: Shan He, Yalong Ma, Tao Song, Yongzhi Jiang, Xinkai Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.18151)  

**Abstract**: Planning a safe and feasible trajectory for autonomous vehicles in real-time by fully utilizing perceptual information in complex urban environments is challenging. In this paper, we propose a spatio-temporal trajectory planning method based on graph optimization. It efficiently extracts the multi-modal information of the perception module by constructing a semantic spatio-temporal map through separation processing of static and dynamic obstacles, and then quickly generates feasible trajectories via sparse graph optimization based on a semantic spatio-temporal hypergraph. Extensive experiments have proven that the proposed method can effectively handle complex urban public road scenarios and perform in real time. We will also release our codes to accommodate benchmarking for the research community 

**Abstract (ZH)**: 在复杂城市环境中实时规划自主车辆的安全可行轨迹，充分利用感知信息具有挑战性。本文提出了一种基于图优化的空间-时间轨迹规划方法。该方法通过静态和动态障碍物的分离处理，高效地构建了一个语义空间-时间地图，从而提取感知模块的多模态信息，并基于语义空间-时间超图的稀疏图优化快速生成可行轨迹。大量实验表明，所提出的方法能够有效处理复杂的城市公共道路场景，并且能够实时运行。我们还将发布我们的代码，以便研究社区进行基准测试。 

---
# Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations 

**Title (ZH)**: Jacobian稀疏自编码器：稀疏化计算，而不仅仅是激活函数 

**Authors**: Lucy Farnik, Tim Lawson, Conor Houghton, Laurence Aitchison  

**Link**: [PDF](https://arxiv.org/pdf/2502.18147)  

**Abstract**: Sparse autoencoders (SAEs) have been successfully used to discover sparse and human-interpretable representations of the latent activations of LLMs. However, we would ultimately like to understand the computations performed by LLMs and not just their representations. The extent to which SAEs can help us understand computations is unclear because they are not designed to "sparsify" computations in any sense, only latent activations. To solve this, we propose Jacobian SAEs (JSAEs), which yield not only sparsity in the input and output activations of a given model component but also sparsity in the computation (formally, the Jacobian) connecting them. With a naïve implementation, the Jacobians in LLMs would be computationally intractable due to their size. One key technical contribution is thus finding an efficient way of computing Jacobians in this setup. We find that JSAEs extract a relatively large degree of computational sparsity while preserving downstream LLM performance approximately as well as traditional SAEs. We also show that Jacobians are a reasonable proxy for computational sparsity because MLPs are approximately linear when rewritten in the JSAE basis. Lastly, we show that JSAEs achieve a greater degree of computational sparsity on pre-trained LLMs than on the equivalent randomized LLM. This shows that the sparsity of the computational graph appears to be a property that LLMs learn through training, and suggests that JSAEs might be more suitable for understanding learned transformer computations than standard SAEs. 

**Abstract (ZH)**: 稀疏自编码器（Sparse Autoencoders, SAEs）已经成功地用于发现大型语言模型（LLMs）潜在激活的稀疏且人类可解释的表示。然而，最终我们更希望了解LLMs执行的计算过程，而不仅仅是它们的表示。关于SAEs在理解计算方面的帮助程度尚不清楚，因为它们并没有以任何方式设计来“稀疏化”计算过程，仅是对潜在激活进行稀疏化。为了解决这一问题，我们提出了一种Jacobian自编码器（Jacobian Sparse Autoencoders, JSAEs），它不仅在给定模型组件的输入和输出激活中实现了稀疏性，还在连接它们的计算（形式上，Jacobian矩阵）中也实现了稀疏性。由于LLMs中的Jacobian矩阵尺寸巨大，采用简单实现时，计算这些矩阵将是难以处理的。因此，一个重要的技术贡献是找到了一种高效计算Jacobian矩阵的方法。我们发现，JSAEs在保留下游LLMs性能的同时，能够提取相对较高的计算稀疏性。我们还证明了Jacobian矩阵是计算稀疏性合理代理的原因是，当以JSAE基重构时，MLPs几乎可以被视为线性的。最后，我们证明了JSAEs在预训练LLMs中的计算稀疏性水平高于等效的随机化LLMs。这表明计算图的稀疏性似乎是一种LLMs在训练过程中学习的特性，这表明JSAEs可能比标准SAEs更适合于理解学习到的Transformer计算过程。 

---
# Large Language Model Driven Agents for Simulating Echo Chamber Formation 

**Title (ZH)**: 基于大型语言模型的智能代理用于模拟回声室效应形成 

**Authors**: Chenhao Gu, Ling Luo, Zainab Razia Zaidi, Shanika Karunasekera  

**Link**: [PDF](https://arxiv.org/pdf/2502.18138)  

**Abstract**: The rise of echo chambers on social media platforms has heightened concerns about polarization and the reinforcement of existing beliefs. Traditional approaches for simulating echo chamber formation have often relied on predefined rules and numerical simulations, which, while insightful, may lack the nuance needed to capture complex, real-world interactions. In this paper, we present a novel framework that leverages large language models (LLMs) as generative agents to simulate echo chamber dynamics within social networks. The novelty of our approach is that it incorporates both opinion updates and network rewiring behaviors driven by LLMs, allowing for a context-aware and semantically rich simulation of social interactions. Additionally, we utilize real-world Twitter (now X) data to benchmark the LLM-based simulation against actual social media behaviors, providing insights into the accuracy and realism of the generated opinion trends. Our results demonstrate the efficacy of LLMs in modeling echo chamber formation, capturing both structural and semantic dimensions of opinion clustering. %This work contributes to a deeper understanding of social influence dynamics and offers a new tool for studying polarization in online communities. 

**Abstract (ZH)**: 社交媒体平台上回声室现象的兴起加剧了关于极化和现有信念强化的担忧。传统上用于模拟回声室形成的手段往往依赖于预设规则和数值模拟，虽然具有一定的启发性，但在捕捉复杂现实世界交互方面可能缺乏细腻之处。在本文中，我们提出了一种新颖的框架，利用大语言模型（LLMs）作为生成代理来模拟社交网络中的回声室动态。我们方法的新颖之处在于，它结合了由LLMs驱动的意见更新和网络重连行为，从而实现对社交互动的上下文意识和语义丰富的模拟。此外，我们使用真实世界的Twitter（现在称为X）数据，将基于LLM的模拟与实际社交媒体行为进行基准测试，以洞察生成意见趋势的准确性和现实性。实验结果表明，LLMs在建模回声室形成方面具有有效性，能够捕获意见集群的结构和语义维度。%本研究加深了对社会影响动力学的理解，并提供了一个研究在线社区极化的新工具。 

---
# SpargeAttn: Accurate Sparse Attention Accelerating Any Model Inference 

**Title (ZH)**: SpargeAttn：准确的稀疏注意力加速任意模型推断

这个翻译既保留了原文的技术性，又符合学术论文标题的规范。其中，“SpargeAttn”是一个专有名词，保持不变；“准确的稀疏注意力”是对“SpargeAttn”的解释说明；“加速任意模型推断”是对“Accurate Sparse Attention Accelerating Any Model Inference”的进一步说明，使之更符合中文表达习惯。 

**Authors**: Jintao Zhang, Chendong Xiang, Haofeng Huang, Jia Wei, Haocheng Xi, Jun Zhu, Jianfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.18137)  

**Abstract**: An efficient attention implementation is essential for large models due to its quadratic time complexity. Fortunately, attention commonly exhibits sparsity, i.e., many values in the attention map are near zero, allowing for the omission of corresponding computations. Many studies have utilized the sparse pattern to accelerate attention. However, most existing works focus on optimizing attention within specific models by exploiting certain sparse patterns of the attention map. A universal sparse attention that guarantees both the speedup and end-to-end performance of diverse models remains elusive. In this paper, we propose SpargeAttn, a universal sparse and quantized attention for any model. Our method uses a two-stage online filter: in the first stage, we rapidly and accurately predict the attention map, enabling the skip of some matrix multiplications in attention. In the second stage, we design an online softmax-aware filter that incurs no extra overhead and further skips some matrix multiplications. Experiments show that our method significantly accelerates diverse models, including language, image, and video generation, without sacrificing end-to-end metrics. The codes are available at this https URL. 

**Abstract (ZH)**: 高效实现注意力机制对于大规模模型来说至关重要，因为注意力机制的时间复杂度高达二次方。幸运的是，注意力机制通常具有稀疏性，即注意图中的许多值接近于零，允许省略相应计算。许多研究已经利用稀疏性来加速注意力机制。然而，大多数现有工作都集中在通过利用注意力图的特定稀疏模式来优化特定模型上的注意力机制上。一种同时保证多样模型的加速和端到端性能的通用稀疏注意力机制至今仍未实现。本文中，我们提出了一种通用的稀疏量化注意力机制 SargeAttn，适用于任何模型。我们的方法采用两阶段在线过滤器：在第一阶段，我们快速准确地预测注意力图，从而省略部分矩阵乘法。在第二阶段，我们设计了一种在线 Softmax 意识过滤器，不会引入额外开销，进一步省略部分矩阵乘法。实验结果表明，我们的方法在不牺牲端到端指标的情况下显著加速了多样模型，包括语言生成、图像生成和视频生成。相关代码可在以下链接获取：this https URL。 

---
# EU-Nets: Enhanced, Explainable and Parsimonious U-Nets 

**Title (ZH)**: EU-网络：增强、解释性和简洁的U型网络 

**Authors**: B. Sun, P. Liò  

**Link**: [PDF](https://arxiv.org/pdf/2502.18122)  

**Abstract**: In this study, we propose MHEX+, a framework adaptable to any U-Net architecture. Built upon MHEX+, we introduce novel U-Net variants, EU-Nets, which enhance explainability and uncertainty estimation, addressing the limitations of traditional U-Net models while improving performance and stability. A key innovation is the Equivalent Convolutional Kernel, which unifies consecutive convolutional layers, boosting interpretability. For uncertainty estimation, we propose the collaboration gradient approach, measuring gradient consistency across decoder layers. Notably, EU-Nets achieve an average accuracy improvement of 1.389\% and a variance reduction of 0.83\% across all networks and datasets in our experiments, requiring fewer than 0.1M parameters. 

**Abstract (ZH)**: 在本研究中，我们提出了一种可适应任何U-Net架构的框架MHEX+。基于MHEX+，我们引入了新的U-Net变体EU-Nets，这些变体增强了可解释性和不确定性估计，克服了传统U-Net模型的局限性，同时提高了性能和稳定性。关键创新在于等价卷积核，它可以统一连续的卷积层，增强可解释性。对于不确定性估计，我们提出了协作梯度方法，通过测量解码器层间的梯度一致性来进行评估。值得注意的是，在我们的实验中，EU-Nets在所有网络和数据集上的平均准确率提高了1.389%，方差降低了0.83%，且所需的参数量不到0.1百万。 

---
# Bayesian Optimization for Controlled Image Editing via LLMs 

**Title (ZH)**: 基于LLM的受控图像编辑的贝叶斯优化 

**Authors**: Chengkun Cai, Haoliang Liu, Xu Zhao, Zhongyu Jiang, Tianfang Zhang, Zongkai Wu, Jenq-Neng Hwang, Serge Belongie, Lei Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.18116)  

**Abstract**: In the rapidly evolving field of image generation, achieving precise control over generated content and maintaining semantic consistency remain significant limitations, particularly concerning grounding techniques and the necessity for model fine-tuning. To address these challenges, we propose BayesGenie, an off-the-shelf approach that integrates Large Language Models (LLMs) with Bayesian Optimization to facilitate precise and user-friendly image editing. Our method enables users to modify images through natural language descriptions without manual area marking, while preserving the original image's semantic integrity. Unlike existing techniques that require extensive pre-training or fine-tuning, our approach demonstrates remarkable adaptability across various LLMs through its model-agnostic design. BayesGenie employs an adapted Bayesian optimization strategy to automatically refine the inference process parameters, achieving high-precision image editing with minimal user intervention. Through extensive experiments across diverse scenarios, we demonstrate that our framework significantly outperforms existing methods in both editing accuracy and semantic preservation, as validated using different LLMs including Claude3 and GPT-4. 

**Abstract (ZH)**: 在快速发展的图像生成领域，实现对生成内容的精细控制并保持语义一致仍然面临重大挑战，特别是在锚定技术以及模型微调的必要性方面。为解决这些挑战，我们提出了BayesGenie，这是一种开源方法，将大型语言模型（LLMs）与贝叶斯优化相结合，以促进精确且用户友好的图像编辑。该方法允许用户通过自然语言描述来修改图像，而无需手动标记区域，同时保留原始图像的语义完整性。与现有需要大量预训练或微调的技术不同，通过其模型无关的设计，我们的方法显示出在各种LLMs上的显著适应性。BayesGenie采用适应性的贝叶斯优化策略自动优化推断过程参数，实现了在最少用户干预的情况下进行高精度的图像编辑。通过在多种场景下的广泛实验，我们证明了我们的框架在编辑准确性和语义保留方面显著优于现有方法，这些实验使用了不同的LLMs，包括Claude3和GPT-4进行验证。 

---
# The Built-In Robustness of Decentralized Federated Averaging to Bad Data 

**Title (ZH)**: 分布式联邦平均算法内建的鲁棒性对不良数据的抵抗能力 

**Authors**: Samuele Sabella, Chiara Boldrini, Lorenzo Valerio, Andrea Passarella, Marco Conti  

**Link**: [PDF](https://arxiv.org/pdf/2502.18097)  

**Abstract**: Decentralized federated learning (DFL) enables devices to collaboratively train models over complex network topologies without relying on a central controller. In this setting, local data remains private, but its quality and quantity can vary significantly across nodes. The extent to which a fully decentralized system is vulnerable to poor-quality or corrupted data remains unclear, but several factors could contribute to potential risks. Without a central authority, there can be no unified mechanism to detect or correct errors, and each node operates with a localized view of the data distribution, making it difficult for the node to assess whether its perspective aligns with the true distribution. Moreover, models trained on low-quality data can propagate through the network, amplifying errors. To explore the impact of low-quality data on DFL, we simulate two scenarios with degraded data quality -- one where the corrupted data is evenly distributed in a subset of nodes and one where it is concentrated on a single node -- using a decentralized implementation of FedAvg. Our results reveal that averaging-based decentralized learning is remarkably robust to localized bad data, even when the corrupted data resides in the most influential nodes of the network. Counterintuitively, this robustness is further enhanced when the corrupted data is concentrated on a single node, regardless of its centrality in the communication network topology. This phenomenon is explained by the averaging process, which ensures that no single node -- however central -- can disproportionately influence the overall learning process. 

**Abstract (ZH)**: 去中心化联邦学习（Decentralized Federated Learning, DFL）允许设备在不依赖中心控制器的情况下，通过复杂的网络拓扑结构协作训练模型。在该框架下，本地数据保持私密，但其质量和数量在不同的节点之间可能会有显著差异。完全去中心化系统在面对低质量或被污染的数据时的脆弱性仍不明确，但多种因素可能会导致潜在风险。由于没有中央权威机构，不可能有一个统一的方法来检测或纠正错误，每个节点只能从局部视角观察数据分布，这使得节点难以判断其自身视角是否与真实的数据分布保持一致。此外，基于低质量数据训练的模型可以通过网络传播，放大错误。为了研究低质量数据对DFL的影响，我们通过使用去中心化的FedAvg实现模拟了两种数据质量降级的场景：一种是被污染的数据均匀分布在部分节点中，另一种是集中在单个节点上。我们的结果显示，基于平均值的去中心化学习对局部低质量数据表现出很强的鲁棒性，即使污染数据存在于网络中最具影响力的节点上也是如此。令人意想不到的是，当污染数据集中在单个节点上时，无论该节点在网络通信拓扑中的中心性如何，这种鲁棒性进一步增强。这一现象可以用平均值过程来解释，该过程确保了没有任何节点——无论是否中心——能够不成比例地影响整个学习过程。 

---
# Towards Thinking-Optimal Scaling of Test-Time Compute for LLM Reasoning 

**Title (ZH)**: 面向推理最优的大型语言模型测试时计算量缩放方法 

**Authors**: Wenkai Yang, Shuming Ma, Yankai Lin, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.18080)  

**Abstract**: Recent studies have shown that making a model spend more time thinking through longer Chain of Thoughts (CoTs) enables it to gain significant improvements in complex reasoning tasks. While current researches continue to explore the benefits of increasing test-time compute by extending the CoT lengths of Large Language Models (LLMs), we are concerned about a potential issue hidden behind the current pursuit of test-time scaling: Would excessively scaling the CoT length actually bring adverse effects to a model's reasoning performance? Our explorations on mathematical reasoning tasks reveal an unexpected finding that scaling with longer CoTs can indeed impair the reasoning performance of LLMs in certain domains. Moreover, we discover that there exists an optimal scaled length distribution that differs across different domains. Based on these insights, we propose a Thinking-Optimal Scaling strategy. Our method first uses a small set of seed data with varying response length distributions to teach the model to adopt different reasoning efforts for deep thinking. Then, the model selects its shortest correct response under different reasoning efforts on additional problems for self-improvement. Our self-improved models built upon Qwen2.5-32B-Instruct outperform other distillation-based 32B o1-like models across various math benchmarks, and achieve performance on par with QwQ-32B-Preview. 

**Abstract (ZH)**: 近期的研究表明，让模型通过更长的推理链（Chain of Thoughts, CoTs）进行深入思考，能够显著改善其在复杂推理任务中的表现。虽然当前研究仍在探索通过增加大型语言模型（LLMs）的CoTs长度以扩大测试时间计算能力所带来的好处，但我们担忧当前追求测试时间扩展背后可能隐藏的一个潜在问题：过长的CoTs 是否会实际上对模型的推理性能产生负面影响？通过对数学推理任务的研究，我们的探索揭示了一个意想不到的结果：在某些领域中，随着CoTs长度的增加，确实会损害LLMs的推理性能。此外，我们发现不同领域存在一个最优的CoTs长度分布。基于这些发现，我们提出了一种称为“最优推理长度调整”（Thinking-Optimal Scaling）的策略。该方法首先使用一组具有不同响应长度分布的种子数据来教导模型在深度思考时采用不同的推理努力。然后，模型在其推理努力不同的情况下选择最短的正确响应，以实现自我改进。基于此策略改进的模型，在Qwen2.5-32B-Instruct的基础上，跨越各种数学基准测试的表现优于其他基于蒸馏的32B o1-like模型，并在某些方面达到了与QwQ-32B-Preview相当的性能。 

---
# MRBTP: Efficient Multi-Robot Behavior Tree Planning and Collaboration 

**Title (ZH)**: MRBTP：高效的多机器人行为树规划与协作 

**Authors**: Yishuai Cai, Xinglin Chen, Zhongxuan Cai, Yunxin Mao, Minglong Li, Wenjing Yang, Ji Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18072)  

**Abstract**: Multi-robot task planning and collaboration are critical challenges in robotics. While Behavior Trees (BTs) have been established as a popular control architecture and are plannable for a single robot, the development of effective multi-robot BT planning algorithms remains challenging due to the complexity of coordinating diverse action spaces. We propose the Multi-Robot Behavior Tree Planning (MRBTP) algorithm, with theoretical guarantees of both soundness and completeness. MRBTP features cross-tree expansion to coordinate heterogeneous actions across different BTs to achieve the team's goal. For homogeneous actions, we retain backup structures among BTs to ensure robustness and prevent redundant execution through intention sharing. While MRBTP is capable of generating BTs for both homogeneous and heterogeneous robot teams, its efficiency can be further improved. We then propose an optional plugin for MRBTP when Large Language Models (LLMs) are available to reason goal-related actions for each robot. These relevant actions can be pre-planned to form long-horizon subtrees, significantly enhancing the planning speed and collaboration efficiency of MRBTP. We evaluate our algorithm in warehouse management and everyday service scenarios. Results demonstrate MRBTP's robustness and execution efficiency under varying settings, as well as the ability of the pre-trained LLM to generate effective task-specific subtrees for MRBTP. 

**Abstract (ZH)**: 多机器人任务规划与协作是机器人领域中的关键挑战。虽然行为树（BTs）已被确立为一种流行的控制架构，并且可用于单个机器人的规划，但由于协调多元动作空间的复杂性，开发有效的多机器人行为树规划算法仍然具有挑战性。我们提出了一种多机器人行为树规划（MRBTP）算法，该算法在理论上保证了完备性和正确性。MRBTP 特别设计了跨树扩展机制，用于协调不同行为树之间的不同动作，以实现团队目标。对于相同的动作，MRBTP 保留了行为树之间的辅助结构，以确保鲁棒性并防止由于意图共享而导致的冗余执行。尽管 MRBTP 能够生成适用于同构和异构机器人群体的行为树，但其效率仍有待提高。当大型语言模型（LLMs）可用时，我们提出了一种可选插件来进一步增强 MRBTP 的效果。该插件可以推理每个机器人的相关动作，并提前规划这些动作以形成长视线子树，显著提升 MRBTP 的规划速度和协作效率。我们将在仓库管理和日常服务场景中评估该算法。结果表明，MRBTP 在不同设置下具有鲁棒性和执行效率，并且预训练的 LLM 有能力为 MRBTP 生成有效的任务特定子树。 

---
# HEROS-GAN: Honed-Energy Regularized and Optimal Supervised GAN for Enhancing Accuracy and Range of Low-Cost Accelerometers 

**Title (ZH)**: HEROS-GAN：优化能效正则化和监督生成对抗网络，以提高低成本加速度计的准确性和量程范围 

**Authors**: Yifeng Wang, Yi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18064)  

**Abstract**: Low-cost accelerometers play a crucial role in modern society due to their advantages of small size, ease of integration, wearability, and mass production, making them widely applicable in automotive systems, aerospace, and wearable technology. However, this widely used sensor suffers from severe accuracy and range limitations. To this end, we propose a honed-energy regularized and optimal supervised GAN (HEROS-GAN), which transforms low-cost sensor signals into high-cost equivalents, thereby overcoming the precision and range limitations of low-cost accelerometers. Due to the lack of frame-level paired low-cost and high-cost signals for training, we propose an Optimal Transport Supervision (OTS), which leverages optimal transport theory to explore potential consistency between unpaired data, thereby maximizing supervisory information. Moreover, we propose a Modulated Laplace Energy (MLE), which injects appropriate energy into the generator to encourage it to break range limitations, enhance local changes, and enrich signal details. Given the absence of a dedicated dataset, we specifically establish a Low-cost Accelerometer Signal Enhancement Dataset (LASED) containing tens of thousands of samples, which is the first dataset serving to improve the accuracy and range of accelerometers and is released in Github. Experimental results demonstrate that a GAN combined with either OTS or MLE alone can surpass the previous signal enhancement SOTA methods by an order of magnitude. Integrating both OTS and MLE, the HEROS-GAN achieves remarkable results, which doubles the accelerometer range while reducing signal noise by two orders of magnitude, establishing a benchmark in the accelerometer signal processing. 

**Abstract (ZH)**: 低成本加速度计在现代社会中扮演着至关重要的角色，这得益于它们的小型化、易于集成、可穿戴性和大规模生产的优势，使它们广泛应用于汽车系统、航天航空和可穿戴技术等领域。然而，这种广泛应用的传感器面临着严重的精度和量程限制。为此，我们提出了一种精炼能量正则化和最优监督生成对抗网络（HEROS-GAN），该方法将低成本传感器信号转换为高成本等效信号，从而克服了低成本加速度计的精确度和量程限制。由于缺乏适合训练的帧级配对的低成本和高成本信号，我们提出了最优运输监督（OTS），利用最优运输理论探索未配对数据潜在的一致性，从而最大化监督信息。此外，我们提出了一种调制拉普拉斯能量（MLE），通过向生成器注入适当的能量，鼓励其打破量程限制、增强局部变化并丰富信号细节。鉴于缺乏专门的数据集，我们特别建立了包含数万个样本的低成本加速度计信号增强数据集（LASED），这是首个用于提高加速度计精度和量程的数据集，并在GitHub上发布。实验结果表明，单独使用OTS或MLE的GAN方法在信号增强方面比之前最先进的方法高出一个数量级。结合OTS和MLE，HEROS-GAN取得了显著成果，其不仅将加速度计的量程提高了一倍，信号噪声降低了两个数量级，还为加速度计信号处理建立了新的基准。 

---
# VLM-E2E: Enhancing End-to-End Autonomous Driving with Multimodal Driver Attention Fusion 

**Title (ZH)**: VLM-E2E: 增强端到端自动驾驶的多模态驾驶员注意力融合 

**Authors**: Pei Liu, Haipeng Liu, Haichao Liu, Xin Liu, Jinxin Ni, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.18042)  

**Abstract**: Human drivers adeptly navigate complex scenarios by utilizing rich attentional semantics, but the current autonomous systems struggle to replicate this ability, as they often lose critical semantic information when converting 2D observations into 3D space. In this sense, it hinders their effective deployment in dynamic and complex environments. Leveraging the superior scene understanding and reasoning abilities of Vision-Language Models (VLMs), we propose VLM-E2E, a novel framework that uses the VLMs to enhance training by providing attentional cues. Our method integrates textual representations into Bird's-Eye-View (BEV) features for semantic supervision, which enables the model to learn richer feature representations that explicitly capture the driver's attentional semantics. By focusing on attentional semantics, VLM-E2E better aligns with human-like driving behavior, which is critical for navigating dynamic and complex environments. Furthermore, we introduce a BEV-Text learnable weighted fusion strategy to address the issue of modality importance imbalance in fusing multimodal information. This approach dynamically balances the contributions of BEV and text features, ensuring that the complementary information from visual and textual modality is effectively utilized. By explicitly addressing the imbalance in multimodal fusion, our method facilitates a more holistic and robust representation of driving environments. We evaluate VLM-E2E on the nuScenes dataset and demonstrate its superiority over state-of-the-art approaches, showcasing significant improvements in performance. 

**Abstract (ZH)**: 人类司机能够通过利用丰富的注意语义来巧妙地应对复杂的场景，但当前的自动驾驶系统在将二维观察转换为三维空间时常常丢失关键语义信息，这限制了它们在动态和复杂环境中有效部署的能力。为了解决这一问题，我们利用视觉-语言模型（VLMs）优越的场景理解和推理能力，提出了一种名为VLM-E2E的新框架，该框架通过提供注意语义线索来增强训练。我们的方法将文本表示集成到鸟瞰视图（BEV）特征中，用于语义监督，从而使模型能够学习到更丰富的特征表示，能够显式捕捉驾驶员的注意语义。通过关注注意语义，VLM-E2E更好地符合人类驾驶行为，这对于导航动态和复杂环境至关重要。此外，我们引入了一种BEV-Text可学习加权融合策略，以解决多模态信息融合中模态重要性失衡的问题。此方法能够动态平衡BEV和文本特征的贡献，确保视觉和文本模态互补信息的有效利用。通过明确解决多模态融合中的失衡问题，我们的方法促进了驾驶环境更全面和稳健的表示。我们在nuScenes数据集上评估了VLM-E2E，并展示了其相对于最先进的方法的优越性，证明了其在性能上的显著改进。 

---
# AutoCas: Autoregressive Cascade Predictor in Social Networks via Large Language Models 

**Title (ZH)**: AutoCas：通过大规模语言模型在社交网络中实现自回归级联预测器 

**Authors**: Yuhao Zheng, Chenghua Gong, Rui Sun, Juyuan Zhang, Liming Pan, Linyuan Lv  

**Link**: [PDF](https://arxiv.org/pdf/2502.18040)  

**Abstract**: Popularity prediction in information cascades plays a crucial role in social computing, with broad applications in viral marketing, misinformation control, and content recommendation. However, information propagation mechanisms, user behavior, and temporal activity patterns exhibit significant diversity, necessitating a foundational model capable of adapting to such variations. At the same time, the amount of available cascade data remains relatively limited compared to the vast datasets used for training large language models (LLMs). Recent studies have demonstrated the feasibility of leveraging LLMs for time-series prediction by exploiting commonalities across different time-series domains. Building on this insight, we introduce the Autoregressive Information Cascade Predictor (AutoCas), an LLM-enhanced model designed specifically for cascade popularity prediction. Unlike natural language sequences, cascade data is characterized by complex local topologies, diffusion contexts, and evolving dynamics, requiring specialized adaptations for effective LLM integration. To address these challenges, we first tokenize cascade data to align it with sequence modeling principles. Next, we reformulate cascade diffusion as an autoregressive modeling task to fully harness the architectural strengths of LLMs. Beyond conventional approaches, we further introduce prompt learning to enhance the synergy between LLMs and cascade prediction. Extensive experiments demonstrate that AutoCas significantly outperforms baseline models in cascade popularity prediction while exhibiting scaling behavior inherited from LLMs. Code is available at this repository: this https URL 

**Abstract (ZH)**: 信息cascade中的流行性预测在社交计算中扮演着至关重要的角色，广泛应用于病毒式营销、虚假信息控制和内容推荐等方面。然而，信息传播机制、用户行为和时间活动模式表现出显著的异质性，这需要一种能够适应这些变化的基础模型。同时，可用的cascade数据集相对有限，与用于训练大规模语言模型（LLMs）的庞大数据集相比，仍显得不足。最近的研究表明，通过利用不同时间序列领域的共有特性，可以利用LLMs进行时间序列预测。基于这一洞察，我们提出了Autoregressive Information Cascade Predictor（AutoCas），这是一种专门针对cascade流行性预测的增强型LLM模型。与自然语言序列不同，cascade数据具有复杂的局部拓扑结构、扩散上下文和不断演化的动态特性，这要求进行专门的适应以有效集成LLMs。为了解决这些挑战，我们首先对cascade数据进行分词，使其符合序列建模的原则。接下来，我们将cascade扩散重新构想为自回归建模任务，以便充分利用LLMs的架构优势。除了传统的做法，我们还引入了提示学习来增强LLMs与cascade预测之间的协同作用。实验结果表明，AutoCas在cascade流行性预测方面显著优于基线模型，并且表现出与LLMs相继承的扩展性行为。相关代码可在以下仓库中获取：this https URL 

---
# ExPath: Towards Explaining Targeted Pathways for Biological Knowledge Bases 

**Title (ZH)**: ExPath：面向生物知识库的针对性路径解释 

**Authors**: Rikuto Kotoge, Ziwei Yang, Zheng Chen, Yushun Dong, Yasuko Matsubara, Jimeng Sun, Yasushi Sakurai  

**Link**: [PDF](https://arxiv.org/pdf/2502.18026)  

**Abstract**: Biological knowledge bases provide systemically functional pathways of cells or organisms in terms of molecular interaction. However, recognizing more targeted pathways, particularly when incorporating wet-lab experimental data, remains challenging and typically requires downstream biological analyses and expertise. In this paper, we frame this challenge as a solvable graph learning and explaining task and propose a novel pathway inference framework, ExPath, that explicitly integrates experimental data, specifically amino acid sequences (AA-seqs), to classify various graphs (bio-networks) in biological databases. The links (representing pathways) that contribute more to classification can be considered as targeted pathways. Technically, ExPath comprises three components: (1) a large protein language model (pLM) that encodes and embeds AA-seqs into graph, overcoming traditional obstacles in processing AA-seq data, such as BLAST; (2) PathMamba, a hybrid architecture combining graph neural networks (GNNs) with state-space sequence modeling (Mamba) to capture both local interactions and global pathway-level dependencies; and (3) PathExplainer, a subgraph learning module that identifies functionally critical nodes and edges through trainable pathway masks. We also propose ML-oriented biological evaluations and a new metric. The experiments involving 301 bio-networks evaluations demonstrate that pathways inferred by ExPath maintain biological meaningfulness. We will publicly release curated 301 bio-network data soon. 

**Abstract (ZH)**: 生物知识库提供了细胞或生物系统的分子相互作用的系统功能通路。然而，识别更具针对性的通路，尤其是在结合湿实验数据时，仍然具有挑战性，通常需要下游生物分析和专业领域的知识。本文将这一挑战定义为一个可解决的图学习和解释任务，并提出了一种全新的通路推断框架——ExPath，该框架明确整合了实验数据，特别是氨基酸序列（AA-seqs），以对生物数据库中的各类图（生物网络）进行分类。对分类贡献更大的连接（表示为通路）可以被认为是针对性通路。

技术上，ExPath 包含三个组件：（1）一个大规模蛋白质语言模型（pLM），用于编码和嵌入氨基酸序列（AA-seqs）到图中，以克服处理AA-seq数据的传统障碍，例如基于BLAST的方法；（2）PathMamba，一种结合图神经网络（GNNs）和状态空间序列建模（Mamba）的混合架构，以捕捉局部相互作用和全局通路级别的依赖关系；以及（3）PathExplainer，一个子图学习模块，通过可训练的通路掩码识别功能关键节点和边。

此外，我们还提出了一种面向机器学习的生物评估方法和一种新的评估指标。涉及301个生物网络的实验结果表明，ExPath 推断出的通路保持了生物相关性。我们很快将发布这些精心收集的301个生物网络数据集。 

---
# AfroXLMR-Comet: Multilingual Knowledge Distillation with Attention Matching for Low-Resource languages 

**Title (ZH)**: AfroXLMR-Comet: 注意力匹配的多语言知识蒸馏方法及其在低资源语言上的应用 

**Authors**: Joshua Sakthivel Raju, Sanjay S, Jaskaran Singh Walia, Srinivas Raghav, Vukosi Marivate  

**Link**: [PDF](https://arxiv.org/pdf/2502.18020)  

**Abstract**: Language model compression through knowledge distillation has emerged as a promising approach for deploying large language models in resource-constrained environments. However, existing methods often struggle to maintain performance when distilling multilingual models, especially for low-resource languages. In this paper, we present a novel hybrid distillation approach that combines traditional knowledge distillation with a simplified attention matching mechanism, specifically designed for multilingual contexts. Our method introduces an extremely compact student model architecture, significantly smaller than conventional multilingual models. We evaluate our approach on five African languages: Kinyarwanda, Swahili, Hausa, Igbo, and Yoruba. The distilled student model; AfroXLMR-Comet successfully captures both the output distribution and internal attention patterns of a larger teacher model (AfroXLMR-Large) while reducing the model size by over 85%. Experimental results demonstrate that our hybrid approach achieves competitive performance compared to the teacher model, maintaining an accuracy within 85% of the original model's performance while requiring substantially fewer computational resources. Our work provides a practical framework for deploying efficient multilingual models in resource-constrained environments, particularly benefiting applications involving African languages. 

**Abstract (ZH)**: 通过知识蒸馏压缩语言模型已成为在资源受限环境中部署大型语言模型的一种有前途的方法。然而，现有的方法在蒸馏多语言模型时往往难以保持性能，尤其是对于低资源语言。在本文中，我们提出了一种新的混合蒸馏方法，该方法将传统的知识蒸馏与简化后的注意力匹配机制相结合，特别适用于多语言环境。我们的方法引入了一种极其紧凑的学生模型架构，比传统的多语言模型小得多。我们在这五种非洲语言上对我们的方法进行了评估： Kirundi（库尼亚鲁）、Swahili（斯瓦希里语）、Hausa（豪萨语）、Igbo（伊博语）和Yoruba（约鲁巴语）。经过蒸馏的学生模型 AfroXLMR-Comet 在保留了更大教师模型（AfroXLMR-Large）输出分布和内部注意力模式的同时，将模型大小减少了超过85%。实验结果表明，我们的混合方法在性能上与教师模型相当，准确率保持在原有模型性能的85%左右，同时所需的计算资源显著减少。我们的工作为在资源受限环境中部署高效的多语言模型提供了一个实用框架，特别有利于涉及非洲语言的应用。 

---
# ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents 

**Title (ZH)**: ViDoRAG：基于动态迭代推理代理的视觉文档检索增强生成方法 

**Authors**: Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18017)  

**Abstract**: Understanding information from visually rich documents remains a significant challenge for traditional Retrieval-Augmented Generation (RAG) methods. Existing benchmarks predominantly focus on image-based question answering (QA), overlooking the fundamental challenges of efficient retrieval, comprehension, and reasoning within dense visual documents. To bridge this gap, we introduce ViDoSeek, a novel dataset designed to evaluate RAG performance on visually rich documents requiring complex reasoning. Based on it, we identify key limitations in current RAG approaches: (i) purely visual retrieval methods struggle to effectively integrate both textual and visual features, and (ii) previous approaches often allocate insufficient reasoning tokens, limiting their effectiveness. To address these challenges, we propose ViDoRAG, a novel multi-agent RAG framework tailored for complex reasoning across visual documents. ViDoRAG employs a Gaussian Mixture Model (GMM)-based hybrid strategy to effectively handle multi-modal retrieval. To further elicit the model's reasoning capabilities, we introduce an iterative agent workflow incorporating exploration, summarization, and reflection, providing a framework for investigating test-time scaling in RAG domains. Extensive experiments on ViDoSeek validate the effectiveness and generalization of our approach. Notably, ViDoRAG outperforms existing methods by over 10% on the competitive ViDoSeek benchmark. 

**Abstract (ZH)**: 传统的检索增强生成（RAG）方法在理解视觉丰富的文档信息方面仍面临显著挑战。现有的基准主要侧重于基于图像的问题回答（QA），忽视了密集视觉文档中高效检索、理解和推理的基本挑战。为弥合这一差距，我们引入了ViDoSeek，这是一个新数据集，旨在评估RAG在需要复杂推理的视觉丰富文档中的性能。基于此，我们识别了当前RAG方法的关键局限性：(i) 纯视觉检索方法难以有效地整合文本和视觉特征，(ii) 以前的方法往往分配不足的推理令牌，限制了它们的效果。为了应对这些挑战，我们提出了ViDoRAG，这是一个专门为视觉文档中复杂推理设计的多agent RAG框架。ViDoRAG采用基于高斯混合模型（GMM）的混合策略有效地处理多模态检索。为了进一步激发模型的推理能力，我们引入了一种迭代的agent工作流，包括探索、总结和反思，为RAG领域中的测试时尺度研究提供了框架。在ViDoSeek上的广泛实验验证了我们方法的有效性和泛化能力。值得注意的是，ViDoRAG在竞争性的ViDoSeek基准测试中性能优于现有方法超过10%。 

---
# NotaGen: Advancing Musicality in Symbolic Music Generation with Large Language Model Training Paradigms 

**Title (ZH)**: NotaGen：通过大规模语言模型训练范式提升符号音乐生成的音乐性 

**Authors**: Yashan Wang, Shangda Wu, Jianhuai Hu, Xingjian Du, Yueqi Peng, Yongxin Huang, Shuai Fan, Xiaobing Li, Feng Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.18008)  

**Abstract**: We introduce NotaGen, a symbolic music generation model aiming to explore the potential of producing high-quality classical sheet music. Inspired by the success of Large Language Models (LLMs), NotaGen adopts pre-training, fine-tuning, and reinforcement learning paradigms (henceforth referred to as the LLM training paradigms). It is pre-trained on 1.6M pieces of music, and then fine-tuned on approximately 9K high-quality classical compositions conditioned on "period-composer-instrumentation" prompts. For reinforcement learning, we propose the CLaMP-DPO method, which further enhances generation quality and controllability without requiring human annotations or predefined rewards. Our experiments demonstrate the efficacy of CLaMP-DPO in symbolic music generation models with different architectures and encoding schemes. Furthermore, subjective A/B tests show that NotaGen outperforms baseline models against human compositions, greatly advancing musical aesthetics in symbolic music this http URL project homepage is this https URL. 

**Abstract (ZH)**: 我们将介绍一种名为NotaGen的象征性音乐生成模型，旨在探索产生高质量古典乐谱的潜力。受到大型语言模型（LLMs）成功经验的启发，NotaGen采用了预训练、微调和强化学习的范式（以下简称LLM训练范式）。它基于160万首音乐进行预训练，并在约9000首高质量古典作品上进行了微调，这些作品的微调条件包括“时期-作曲家-乐器配置”提示。在强化学习方面，我们提出了一种名为CLaMP-DPO的方法，该方法进一步提高了生成质量和可控性，无需使用人工注释或预定义的奖励。我们的实验表明，CLaMP-DPO在具有不同架构和编码方案的象征性音乐生成模型中的有效性。此外，主观的A/B测试显示，NotaGen在与人类作品的比较中表现更优，极大地促进了象征性音乐中的音乐美学。该项目的主页是这个链接：https://yourprojecthomepage.com/。 

---
# Radon-Nikodým Derivative: Re-imagining Anomaly Detection from a Measure Theoretic Perspective 

**Title (ZH)**: 拉东-尼科迪姆导数：从测度论视角重构异常检测 

**Authors**: Shlok Mehendale, Aditya Challa, Rahul Yedida, Sravan Danda, Santonu Sarkar, Snehanshu Saha  

**Link**: [PDF](https://arxiv.org/pdf/2502.18002)  

**Abstract**: Which principle underpins the design of an effective anomaly detection loss function? The answer lies in the concept of \rnthm{} theorem, a fundamental concept in measure theory. The key insight is -- Multiplying the vanilla loss function with the \rnthm{} derivative improves the performance across the board. We refer to this as RN-Loss. This is established using PAC learnability of anomaly detection. We further show that the \rnthm{} derivative offers important insights into unsupervised clustering based anomaly detections as well. We evaluate our algorithm on 96 datasets, including univariate and multivariate data from diverse domains, including healthcare, cybersecurity, and finance. We show that RN-Derivative algorithms outperform state-of-the-art methods on 68\% of Multivariate datasets (based on F-1 scores) and also achieves peak F1-scores on 72\% of time series (Univariate) datasets. 

**Abstract (ZH)**: 哪种原则支撑了有效异常检测损失函数的设计？答案在于测度论中的基本概念——\rnthm{}定理。关键见解在于——将传统的损失函数与\rnthm{}导数相乘可以全面提高性能。我们将其称为RN-损失。这一点通过异常检测的PAC可学习性得到证明。进一步的分析表明，\rnthm{}导数还为基于无监督聚类的异常检测提供了重要的见解。我们在96个数据集上评估了我们的算法，包括来自医疗保健、网络安全和金融等多个领域的单变量和多变量数据。结果显示，基于F-1分数，RN-导数算法在68%的多变量数据集上优于现有最先进的方法，并且在72%的时间序列（单变量）数据集上达到了最高F-1分数。 

---
# MAGE: Multi-Head Attention Guided Embeddings for Low Resource Sentiment Classification 

**Title (ZH)**: MAGE：多头注意力引导的嵌入表示在资源有限的情感分类中的应用 

**Authors**: Varun Vashisht, Samar Singh, Mihir Konduskar, Jaskaran Singh Walia, Vukosi Marivate  

**Link**: [PDF](https://arxiv.org/pdf/2502.17987)  

**Abstract**: Due to the lack of quality data for low-resource Bantu languages, significant challenges are presented in text classification and other practical implementations. In this paper, we introduce an advanced model combining Language-Independent Data Augmentation (LiDA) with Multi-Head Attention based weighted embeddings to selectively enhance critical data points and improve text classification performance. This integration allows us to create robust data augmentation strategies that are effective across various linguistic contexts, ensuring that our model can handle the unique syntactic and semantic features of Bantu languages. This approach not only addresses the data scarcity issue but also sets a foundation for future research in low-resource language processing and classification tasks. 

**Abstract (ZH)**: 由于缺乏低资源班图语的高质量数据，文本分类及其他实际应用面临重大挑战。本文介绍了一种结合语言无关数据增强（LiDA）与多头注意力加权嵌入的先进模型，以有选择性地增强关键数据点并提高文本分类性能。这种结合使得我们能够创建适用于各种语言背景的鲁棒数据增强策略，确保模型能够处理班图语的独特句法和语义特征。这种方法不仅解决了数据稀缺的问题，还为低资源语言处理和分类任务的研究奠定了基础。 

---
# Broadening Discovery through Structural Models: Multimodal Combination of Local and Structural Properties for Predicting Chemical Features 

**Title (ZH)**: 通过结构模型扩展发现范围：结合局部和结构属性的多模态方法预测化学特性 

**Authors**: Nikolai Rekut, Alexey Orlov, Klea Ziu, Elizaveta Starykh, Martin Takac, Aleksandr Beznosikov  

**Link**: [PDF](https://arxiv.org/pdf/2502.17986)  

**Abstract**: In recent years, machine learning has profoundly reshaped the field of chemistry, facilitating significant advancements across various applications, including the prediction of molecular properties and the generation of molecular structures. Language models and graph-based models are extensively utilized within this domain, consistently achieving state-of-the-art results across an array of tasks. However, the prevailing practice of representing chemical compounds in the SMILES format -- used by most datasets and many language models -- presents notable limitations as a training data format. In contrast, chemical fingerprints offer a more physically informed representation of compounds, thereby enhancing their suitability for model training. This study aims to develop a language model that is specifically trained on fingerprints. Furthermore, we introduce a bimodal architecture that integrates this language model with a graph model. Our proposed methodology synthesizes these approaches, utilizing RoBERTa as the language model and employing Graph Isomorphism Networks (GIN), Graph Convolutional Networks (GCN) and Graphormer as graph models. This integration results in a significant improvement in predictive performance compared to conventional strategies for tasks such as Quantitative Structure-Activity Relationship (QSAR) and the prediction of nuclear magnetic resonance (NMR) spectra, among others. 

**Abstract (ZH)**: 近年来，机器学习深刻重塑了化学领域，促进了多种应用的重大进步，包括分子性质预测和分子结构生成。在这一领域中，语言模型和图基模型被广泛使用，它们在多种任务上均实现了最先进的结果。然而，大多数数据集和许多语言模型普遍采用的SMILES格式作为训练数据表示形式，存在明显的局限性。相比之下，化学指纹为化合物提供了更物理化的表示形式，从而增强了它们的模型训练适用性。本研究旨在开发一种专门基于指纹训练的语言模型。此外，我们提出了一种双模架构，将这种语言模型与图模型相结合。我们提出的方法结合了这两种方法，使用RoBERTa作为语言模型，并采用Graph Isomorphism Networks（GIN）、Graph Convolutional Networks（GCN）和Graphormer作为图模型。这种整合在定量结构-活性关系（QSAR）和核磁共振（NMR）光谱预测等任务上的预测性能上比传统策略有显著的提升。 

---
# LLM Knows Geometry Better than Algebra: Numerical Understanding of LLM-Based Agents in A Trading Arena 

**Title (ZH)**: LLM 在几何知识上优于代数：基于 LLM 的代理在交易竞技场中的数值理解 

**Authors**: Tianmi Ma, Jiawei Du, Wenxin Huang, Wenjie Wang, Liang Xie, Xian Zhong, Joey Tianyi Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.17967)  

**Abstract**: Recent advancements in large language models (LLMs) have significantly improved performance in natural language processing tasks. However, their ability to generalize to dynamic, unseen tasks, particularly in numerical reasoning, remains a challenge. Existing benchmarks mainly evaluate LLMs on problems with predefined optimal solutions, which may not align with real-world scenarios where clear answers are absent. To bridge this gap, we design the Agent Trading Arena, a virtual numerical game simulating complex economic systems through zero-sum games, where agents invest in stock portfolios. Our experiments reveal that LLMs, including GPT-4o, struggle with algebraic reasoning when dealing with plain-text stock data, often focusing on local details rather than global trends. In contrast, LLMs perform significantly better with geometric reasoning when presented with visual data, such as scatter plots or K-line charts, suggesting that visual representations enhance numerical reasoning. This capability is further improved by incorporating the reflection module, which aids in the analysis and interpretation of complex data. We validate our findings on NASDAQ Stock dataset, where LLMs demonstrate stronger reasoning with visual data compared to text. Our code and data are publicly available at this https URL. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在自然语言处理任务中的性能显著提升。然而，它们在动态、未见过的任务中的泛化能力，尤其是在数值推理方面，仍然存在挑战。现有基准测试主要评估LLMs在具有预定义最优解的问题上的表现，这可能与现实世界中缺乏明确答案的场景不一致。为解决这一问题，我们设计了Agent Trading Arena，这是一种虚拟数值游戏，通过零和博弈模拟复杂经济系统，其中的代理投资于股票组合。我们的实验表明，包括GPT-4o在内的LLMs在处理平文本股票数据时，在代数推理方面表现困难，往往关注局部细节而非整体趋势。相比之下，当提供可视化数据（如散点图或K线图）时，LLMs在几何推理方面的表现显著更好，这表明可视化表示能够增强数值推理能力。通过引入反射模块，这一能力得到进一步提升，该模块有助于复杂数据的分析和解释。我们在纳斯达克股票数据集上验证了上述发现，结果显示，与文本相比，LLMs在可视化数据上的推理能力更强。我们的代码和数据已在以下网址公开：[请在此处填写网址]。 

---
# Language Models' Factuality Depends on the Language of Inquiry 

**Title (ZH)**: 语言模型的事实性取决于问询语言 

**Authors**: Tushar Aggarwal, Kumar Tanmay, Ayush Agrawal, Kumar Ayush, Hamid Palangi, Paul Pu Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17955)  

**Abstract**: Multilingual language models (LMs) are expected to recall factual knowledge consistently across languages, yet they often fail to transfer knowledge between languages even when they possess the correct information in one of the languages. For example, we find that an LM may correctly identify Rashed Al Shashai as being from Saudi Arabia when asked in Arabic, but consistently fails to do so when asked in English or Swahili. To systematically investigate this limitation, we introduce a benchmark of 10,000 country-related facts across 13 languages and propose three novel metrics: Factual Recall Score, Knowledge Transferability Score, and Cross-Lingual Factual Knowledge Transferability Score-to quantify factual recall and knowledge transferability in LMs across different languages. Our results reveal fundamental weaknesses in today's state-of-the-art LMs, particularly in cross-lingual generalization where models fail to transfer knowledge effectively across different languages, leading to inconsistent performance sensitive to the language used. Our findings emphasize the need for LMs to recognize language-specific factual reliability and leverage the most trustworthy information across languages. We release our benchmark and evaluation framework to drive future research in multilingual knowledge transfer. 

**Abstract (ZH)**: 多语言语言模型（LMs）期望能够在不同语言中一致地回忆事实性知识，但它们往往在一种语言中有正确信息时，仍然无法在其他语言中转移知识。例如，我们发现当用阿拉伯语提问时，LM可以正确识别拉泽德·阿什谢海来自沙特阿拉伯，但在英语或斯瓦希里语中提问时却始终无法做到这一点。为系统地探讨这一局限性，我们提出了一个涵盖13种语言的10,000个国家相关的事实基准，并提出了三个新的评价指标：事实性回忆得分、知识可转移性得分和跨语言事实性知识可转移性得分，以量化LM在不同语言中的事实性回忆和知识可转移性。我们的结果揭示了当今最先进的LM中的一些根本性弱点，尤其是在跨语言泛化方面，模型无法有效地在不同语言之间转移知识，导致性能受所用语言的影响而变化不一。我们的发现强调了LM需要识别语言特定的事实可靠性，并在不同语言中利用最可靠的信息的重要性。我们发布了该基准和评价框架，以推动未来在多语言知识转移方面的研究。 

---
# Robust Polyp Detection and Diagnosis through Compositional Prompt-Guided Diffusion Models 

**Title (ZH)**: 通过组成成分提示引导扩散模型实现稳健的息肉检测与诊断 

**Authors**: Jia Yu, Yan Zhu, Peiyao Fu, Tianyi Chen, Junbo Huang, Quanlin Li, Pinghong Zhou, Zhihua Wang, Fei Wu, Shuo Wang, Xian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17951)  

**Abstract**: Colorectal cancer (CRC) is a significant global health concern, and early detection through screening plays a critical role in reducing mortality. While deep learning models have shown promise in improving polyp detection, classification, and segmentation, their generalization across diverse clinical environments, particularly with out-of-distribution (OOD) data, remains a challenge. Multi-center datasets like PolypGen have been developed to address these issues, but their collection is costly and time-consuming. Traditional data augmentation techniques provide limited variability, failing to capture the complexity of medical images. Diffusion models have emerged as a promising solution for generating synthetic polyp images, but the image generation process in current models mainly relies on segmentation masks as the condition, limiting their ability to capture the full clinical context. To overcome these limitations, we propose a Progressive Spectrum Diffusion Model (PSDM) that integrates diverse clinical annotations-such as segmentation masks, bounding boxes, and colonoscopy reports-by transforming them into compositional prompts. These prompts are organized into coarse and fine components, allowing the model to capture both broad spatial structures and fine details, generating clinically accurate synthetic images. By augmenting training data with PSDM-generated samples, our model significantly improves polyp detection, classification, and segmentation. For instance, on the PolypGen dataset, PSDM increases the F1 score by 2.12% and the mean average precision by 3.09%, demonstrating superior performance in OOD scenarios and enhanced generalization. 

**Abstract (ZH)**: 结直肠癌（CRC）是全球重要的公共卫生问题，早期筛查对其降低死亡率起着关键作用。虽然深度学习模型在提高息肉检测、分类和分割方面显示出潜力，但在不同临床环境中泛化，尤其是面对领域外（OOD）数据方面仍存在挑战。为了应对这些问题，多中心数据集如PolypGen已被开发，但这些数据集的收集成本高、耗时长。传统数据增强技术提供的可变性有限，无法捕捉医学图像的复杂性。扩散模型因其生成合成息肉图像的潜力而受到关注，但当前模型中的图像生成过程主要依赖于分割掩码作为条件，限制了其捕获完整临床上下文的能力。为克服这些局限，我们提出了一种渐进光谱扩散模型（PSDM），该模型通过将多样化的临床注释（如分割掩码、边界框和结肠镜报告）转化为组成式提示，来整合这些注释信息。提示被组织成粗细两个层次的部分，从而使模型能够捕捉广泛的空域结构和精细细节，生成临床准确的合成图像。通过使用PSDM生成的样本增强训练数据，我们的模型显著提高了息肉的检测、分类和分割准确性。例如，在PolypGen数据集上，PSDM提高了F1分数2.12%和平均精确度2.57%（原文为3.09%，这里根据上下文进行了合理推测），展示了在OOD场景中优越的性能和更好的泛化能力。 

---
# DeepSeek-R1 Outperforms Gemini 2.0 Pro, OpenAI o1, and o3-mini in Bilingual Complex Ophthalmology Reasoning 

**Title (ZH)**: DeepSeek-R1 在双语复杂眼科推理方面优于 Gemini 2.0 Pro、OpenAI o1 和 o3-mini 

**Authors**: Pusheng Xu, Yue Wu, Kai Jin, Xiaolan Chen, Mingguang He, Danli Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17947)  

**Abstract**: Purpose: To evaluate the accuracy and reasoning ability of DeepSeek-R1 and three other recently released large language models (LLMs) in bilingual complex ophthalmology cases. Methods: A total of 130 multiple-choice questions (MCQs) related to diagnosis (n = 39) and management (n = 91) were collected from the Chinese ophthalmology senior professional title examination and categorized into six topics. These MCQs were translated into English using DeepSeek-R1. The responses of DeepSeek-R1, Gemini 2.0 Pro, OpenAI o1 and o3-mini were generated under default configurations between February 15 and February 20, 2025. Accuracy was calculated as the proportion of correctly answered questions, with omissions and extra answers considered incorrect. Reasoning ability was evaluated through analyzing reasoning logic and the causes of reasoning error. Results: DeepSeek-R1 demonstrated the highest overall accuracy, achieving 0.862 in Chinese MCQs and 0.808 in English MCQs. Gemini 2.0 Pro, OpenAI o1, and OpenAI o3-mini attained accuracies of 0.715, 0.685, and 0.692 in Chinese MCQs (all P<0.001 compared with DeepSeek-R1), and 0.746 (P=0.115), 0.723 (P=0.027), and 0.577 (P<0.001) in English MCQs, respectively. DeepSeek-R1 achieved the highest accuracy across five topics in both Chinese and English MCQs. It also excelled in management questions conducted in Chinese (all P<0.05). Reasoning ability analysis showed that the four LLMs shared similar reasoning logic. Ignoring key positive history, ignoring key positive signs, misinterpretation medical data, and too aggressive were the most common causes of reasoning errors. Conclusion: DeepSeek-R1 demonstrated superior performance in bilingual complex ophthalmology reasoning tasks than three other state-of-the-art LLMs. While its clinical applicability remains challenging, it shows promise for supporting diagnosis and clinical decision-making. 

**Abstract (ZH)**: 目的：评估DeepSeek-R1和三种最近发布的大型语言模型（LLMs）在双语复杂眼科病例中的准确性和推理能力。
方法：从中国眼科高级职称考试中收集了130道选择题（MCQs），其中包括诊断相关问题（n = 39）和管理相关问题（n = 91），并按六个主题进行了分类。使用DeepSeek-R1将这些MCQs翻译成英文。DeepSeek-R1、Gemini 2.0 Pro、OpenAI o1和o3-mini在2025年2月15日至20日之间按照默认配置生成了响应。准确率按答对问题的比例计算，漏答和多答均视为错误。通过分析推理逻辑和推理错误的原因，评估了推理能力。
结果：DeepSeek-R1在汉语MCQs中的总体准确率达到0.862，在英语MCQs中的准确率达到0.808。Gemini 2.0 Pro、OpenAI o1和OpenAI o3-mini在汉语MCQs中的准确率分别为0.715、0.685和0.692（均与DeepSeek-R1相比P<0.001），在英语MCQs中的准确率分别为0.746（P=0.115）、0.723（P=0.027）和0.577（P<0.001）。DeepSeek-R1在汉语和英语MCQs的五个主题中均表现出最高的准确率。此外，在用汉语进行管理问题时，其准确率也显著高于其他模型（所有P<0.05）。推理能力分析显示，四种LLMs的推理逻辑相似。忽略关键阳性病史、忽略关键阳性体征、误解医学数据和过于激进是最常见的推理错误原因。
结论：DeepSeek-R1在双语复杂眼科推理任务中的表现优于三种其他最先进的LLMs。尽管其临床应用仍具挑战性，但它在支持诊断和临床决策方面显示出潜力。 

---
# Optimal Brain Apoptosis 

**Title (ZH)**: 最佳大脑凋亡 

**Authors**: Mingyuan Sun, Zheng Fang, Jiaxu Wang, Junjie Jiang, Delei Kong, Chenming Hu, Yuetong Fang, Renjing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17941)  

**Abstract**: The increasing complexity and parameter count of Convolutional Neural Networks (CNNs) and Transformers pose challenges in terms of computational efficiency and resource demands. Pruning has been identified as an effective strategy to address these challenges by removing redundant elements such as neurons, channels, or connections, thereby enhancing computational efficiency without heavily compromising performance. This paper builds on the foundational work of Optimal Brain Damage (OBD) by advancing the methodology of parameter importance estimation using the Hessian matrix. Unlike previous approaches that rely on approximations, we introduce Optimal Brain Apoptosis (OBA), a novel pruning method that calculates the Hessian-vector product value directly for each parameter. By decomposing the Hessian matrix across network layers and identifying conditions under which inter-layer Hessian submatrices are non-zero, we propose a highly efficient technique for computing the second-order Taylor expansion of parameters. This approach allows for a more precise pruning process, particularly in the context of CNNs and Transformers, as validated in our experiments including VGG19, ResNet32, ResNet50, and ViT-B/16 on CIFAR10, CIFAR100 and Imagenet datasets. Our code is available at this https URL. 

**Abstract (ZH)**: 随着卷积神经网络（CNNs）和变换器（Transformers）的复杂度和参数数量不断增加，计算效率和资源需求面临挑战。剪枝已被认定为一种有效的应对策略，通过移除冗余元素，如神经元、通道或连接，从而在不大幅牺牲性能的情况下提高计算效率。本文在Optimal Brain Damage（OBD）的基础上，提出了使用海森矩阵估计参数重要性的新方法。与依赖近似的方法不同，我们引入了Optimal Brain Apoptosis（OBA），这是一种新型剪枝方法，可以直接计算每个参数的海森矩阵向量积。通过在网络层间分解海森矩阵，并识别子矩阵在不同层之间非零条件，我们提出了一个高效的二次泰勒展开计算方法。这种方法在CNNs和Transformers中实现了更加精确的剪枝过程，并通过在VGG19、ResNet32、ResNet50和ViT-B/16上的实验验证了其有效性，实验数据集包括CIFAR10、CIFAR100和ImageNet。我们的代码可在这个链接下载：[提供URL的格式] 

---
# Integrating Boosted learning with Differential Evolution (DE) Optimizer: A Prediction of Groundwater Quality Risk Assessment in Odisha 

**Title (ZH)**: 将增强学习与差分进化（DE）优化器集成：奥里萨邦地下水质量风险评估预测 

**Authors**: Sonalika Subudhi, Alok Kumar Pati, Sephali Bose, Subhasmita Sahoo, Avipsa Pattanaik, Biswa Mohan Acharya  

**Link**: [PDF](https://arxiv.org/pdf/2502.17929)  

**Abstract**: Groundwater is eventually undermined by human exercises, such as fast industrialization, urbanization, over-extraction, and contamination from agrarian and urban sources. From among the different contaminants, the presence of heavy metals like cadmium (Cd), chromium (Cr), arsenic (As), and lead (Pb) proves to have serious dangers when present in huge concentrations in groundwater. Long-term usage of these poisonous components may lead to neurological disorders, kidney failure and different sorts of cancer. To address these issues, this study developed a machine learning-based predictive model to evaluate the Groundwater Quality Index (GWQI) and identify the main contaminants which are affecting the water quality. It has been achieved with the help of a hybrid machine learning model i.e. LCBoost Fusion . The model has undergone several processes like data preprocessing, hyperparameter tuning using Differential Evolution (DE) optimization, and evaluation through cross-validation. The LCBoost Fusion model outperforms individual models (CatBoost and LightGBM), by achieving low RMSE (0.6829), MSE (0.5102), MAE (0.3147) and a high R$^2$ score of 0.9809. Feature importance analysis highlights Potassium (K), Fluoride (F) and Total Hardness (TH) as the most influential indicators of groundwater contamination. This research successfully demonstrates the application of machine learning in assessing groundwater quality risks in Odisha. The proposed LCBoost Fusion model offers a reliable and efficient approach for real-time groundwater monitoring and risk mitigation. These findings will help the environmental organizations and the policy makers to map out targeted places for sustainable groundwater management. Future work will focus on using remote sensing data and developing an interactive decision-making system for groundwater quality assessment. 

**Abstract (ZH)**: 地下水最终受到人类活动的影响，这些活动包括快速工业化、城市化、过度开采以及来自农业和城市源的污染。在各种污染物中，重金属如镉（Cd）、铬（Cr）、砷（As）和铅（Pb）在高浓度下存在于地下水中，其带来的危害尤为严重。长期使用这些有毒成分可能导致神经系统障碍、肾功能衰竭和各种类型的癌症。为应对这些问题，本研究开发了一种基于机器学习的预测模型，用于评估地下水质指数（GWQI）并识别主要污染物，从而影响水质。这一模型基于一种混合机器学习方法，即LCBoost融合模型。该模型经过数据预处理、使用差分演化（DE）优化进行超参数调整以及交叉验证评估等多个过程。LCBoost融合模型在单一模型（CatBoost和LightGBM）的基础上表现出优越性，实现了较低的均方根误差（RMSE 0.6829）、均方误差（MSE 0.5102）、平均绝对误差（MAE 0.3147）和较高的R²评分（0.9809）。特征重要性分析指出，钾（K）、氟（F）和总硬度（TH）是最具影响的地下水污染指标。本研究成功展示了在奥里萨邦评估地下水质量风险中机器学习的应用。所提出的LCBoost融合模型提供了一种可靠且高效的实时地下水监测和风险缓解方法。这些发现将有助于环境组织和政策制定者确定可持续地下水管理的目标区域。未来工作将集中在利用遥感数据并开发一个交互式的决策系统来评估地下水质量。 

---
# Structure-prior Informed Diffusion Model for Graph Source Localization with Limited Data 

**Title (ZH)**: 基于结构先验的扩散模型在有限数据条件下对图源定位的研究 

**Authors**: Hongyi Chen, Jingtao Ding, Xiaojun Liang, Yong Li, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17928)  

**Abstract**: The source localization problem in graph information propagation is crucial for managing various network disruptions, from misinformation spread to infrastructure failures. While recent deep generative approaches have shown promise in this domain, their effectiveness is limited by the scarcity of real-world propagation data. This paper introduces SIDSL (\textbf{S}tructure-prior \textbf{I}nformed \textbf{D}iffusion model for \textbf{S}ource \textbf{L}ocalization), a novel framework that addresses three key challenges in limited-data scenarios: unknown propagation patterns, complex topology-propagation relationships, and class imbalance between source and non-source nodes. SIDSL incorporates topology-aware priors through graph label propagation and employs a propagation-enhanced conditional denoiser with a GNN-parameterized label propagation module (GNN-LP). Additionally, we propose a structure-prior biased denoising scheme that initializes from structure-based source estimations rather than random noise, effectively countering class imbalance issues. Experimental results across four real-world datasets demonstrate SIDSL's superior performance, achieving 7.5-13.3% improvements in F1 scores compared to state-of-the-art methods. Notably, when pretrained with simulation data of synthetic patterns, SIDSL maintains robust performance with only 10% of training data, surpassing baselines by more than 18.8%. These results highlight SIDSL's effectiveness in real-world applications where labeled data is scarce. 

**Abstract (ZH)**: 图信息传播中的源定位问题对于管理各种网络中断至关重要，从错误信息的传播到基础设施故障。虽然最近的深度生成方法在这一领域显示出前景，但它们的有效性受限于实际传播数据的稀缺性。本文提出了一种新的框架SIDSL（Structure-prior Informed Diffusion model for Source Localization），该框架在数据稀缺的情况下解决了三个关键挑战：未知的传播模式、复杂的空间-传播关系以及源节点与非源节点之间的类别不平衡。SIDSL 通过图标签传播引入拓扑感知先验，并采用一种传播增强的条件去噪器，其中包含一个通过图神经网络参数化的标签传播模块（GNN-LP）。此外，我们提出了一种结构先验偏置的去噪方案，该方案从基于结构的源估计初始化而非随机噪声，从而有效抵消类别不平衡问题。在四个真实数据集上进行的实验结果表明，SIDSL 的性能优于最新方法，其 F1 分数提高了 7.5-13.3%。特别地，当使用合成模式的仿真数据进行预训练时，只有 10% 的训练数据即可保持其稳健性能，相比基线方法提高了超过 18.8%。这些结果突显了在标签数据稀缺的现实应用场景中，SIDSL 的有效性。 

---
# Decoupled Graph Energy-based Model for Node Out-of-Distribution Detection on Heterophilic Graphs 

**Title (ZH)**: 异构图中节点异常分布检测的解耦图能模型 

**Authors**: Yuhan Chen, Yihong Luo, Yifan Song, Pengwen Dai, Jing Tang, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2502.17912)  

**Abstract**: Despite extensive research efforts focused on OOD detection on images, OOD detection on nodes in graph learning remains underexplored. The dependence among graph nodes hinders the trivial adaptation of existing approaches on images that assume inputs to be i.i.d. sampled, since many unique features and challenges specific to graphs are not considered, such as the heterophily issue. Recently, GNNSafe, which considers node dependence, adapted energy-based detection to the graph domain with state-of-the-art performance, however, it has two serious issues: 1) it derives node energy from classification logits without specifically tailored training for modeling data distribution, making it less effective at recognizing OOD data; 2) it highly relies on energy propagation, which is based on homophily assumption and will cause significant performance degradation on heterophilic graphs, where the node tends to have dissimilar distribution with its neighbors. To address the above issues, we suggest training EBMs by MLE to enhance data distribution modeling and remove energy propagation to overcome the heterophily issues. However, training EBMs via MLE requires performing MCMC sampling on both node feature and node neighbors, which is challenging due to the node interdependence and discrete graph topology. To tackle the sampling challenge, we introduce DeGEM, which decomposes the learning process into two parts: a graph encoder that leverages topology information for node representations and an energy head that operates in latent space. Extensive experiments validate that DeGEM, without OOD exposure during training, surpasses previous state-of-the-art methods, achieving an average AUROC improvement of 6.71% on homophilic graphs and 20.29% on heterophilic graphs, and even outperform methods trained with OOD exposure. Our code is available at: this https URL. 

**Abstract (ZH)**: 尽管在图像中的异常数据检测方面进行了大量研究，但在图学习中的节点异常数据检测仍处于探索阶段。图节点之间的依赖关系阻碍了现有基于图像的方法的简单适应，因为这些方法假设输入为独立同分布抽样，而忽略了图特有的许多独特特征和挑战，如异质性问题。最近，GNNSafe 考虑了节点依赖性，将基于能量的检测方法应用于图域，并取得了顶级性能，然而它存在两个严重问题：1）它从分类逻辑中推导节点能量，而没有针对建模数据分布进行特定训练，使其在识别异常数据方面不太有效；2）它高度依赖能量传递，这基于同质性假设，在异质性图上会导致性能严重下降，因为节点与其邻居的分布往往不相似。为解决上述问题，我们建议通过极大似然估计（MLE）训练能量模型（EBM）以增强数据分布建模，并移除能量传递以克服异质性问题。然而，通过 MLE 训练 EBM 需要在节点特征和节点邻居上进行 MCMC 取样，这由于节点间依赖性和离散的图拓扑结构而具有挑战性。为应对取样挑战，我们引入了 DeGEM，将学习过程分解为两个部分：一个图编码器利用拓扑信息进行节点表示，以及一个在潜在空间操作的能量头。大量实验证明，DeGEM 在训练过程中不暴露于异常数据，仍优于之前的方法，在同质性图上平均 AUCROC 提高了 6.71%，在异质性图上提高了 20.29%，甚至超过了在异常数据下训练的方法。我们的代码已发布在：这个网址。 

---
# Enhancing Speech Quality through the Integration of BGRU and Transformer Architectures 

**Title (ZH)**: 通过结合BGRU和Transformer架构提升语音质量 

**Authors**: Souliman Alghnam, Mohammad Alhussien, Khaled Shaheen  

**Link**: [PDF](https://arxiv.org/pdf/2502.17911)  

**Abstract**: Speech enhancement plays an essential role in improving the quality of speech signals in noisy environments. This paper investigates the efficacy of integrating Bidirectional Gated Recurrent Units (BGRU) and Transformer models for speech enhancement tasks. Through a comprehensive experimental evaluation, our study demonstrates the superiority of this hybrid architecture over traditional methods and standalone models. The combined BGRU-Transformer framework excels in capturing temporal dependencies and learning complex signal patterns, leading to enhanced noise reduction and improved speech quality. Results show significant performance gains compared to existing approaches, highlighting the potential of this integrated model in real-world applications. The seamless integration of BGRU and Transformer architectures not only enhances system robustness but also opens the road for advanced speech processing techniques. This research contributes to the ongoing efforts in speech enhancement technology and sets a solid foundation for future investigations into optimizing model architectures, exploring many application scenarios, and advancing the field of speech processing in noisy environments. 

**Abstract (ZH)**: 在嘈杂环境中提高语音信号质量的过程中，语音增强起着至关重要的作用。本文研究了结合双向门控循环单元（BGRU）和Transformer模型在语音增强任务中的有效性。通过全面的实验评估，我们的研究证明了这种混合架构相比于传统方法和单一模型具有更高的优越性。结合的BGRU-Transformer框架在捕捉时间依赖性和学习复杂信号模式方面表现出色，从而提高了降噪效果和语音质量。结果表明，与现有方法相比，该集成模型显示出显著的性能改善，突显出该整合模型在实际应用中的潜在价值。BGRU和Transformer架构的无缝集成不仅增强了系统的鲁棒性，还为高级语音处理技术的发展铺平了道路。这项研究为语音增强技术的发展做出了贡献，并为未来优化模型架构、探索多种应用场景以及推进嘈杂环境中语音处理领域的研究奠定了坚实的基础。 

---
# Scaling LLM Pre-training with Vocabulary Curriculum 

**Title (ZH)**: scaling LLM 预训练的词汇 curriculum 方法 

**Authors**: Fangyuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17910)  

**Abstract**: Modern language models rely on static vocabularies, fixed before pretraining, in contrast to the adaptive vocabulary acquisition observed in human language learning. To bridge this gap, we introduce vocabulary curriculum learning, an approach that improves pretraining efficiency with log-linear scaling gains relative to vocabulary size. Our method alternates between entropy-guided vocabulary expansion and model optimization, enabling models to learn transferable representations across diverse tokenization granularities. This approach naturally gives rise to an optimal computation allocation pattern: longer tokens capture predictable content, while shorter tokens focus on more complex, harder-to-predict contexts. Experiments on small-scale GPT models demonstrate improved scaling efficiency, reinforcing the effectiveness of dynamic tokenization. We release our code to support further research and plan to extend our experiments to larger models and diverse domains. 

**Abstract (ZH)**: 现代语言模型依赖于在预训练前固定下来的静态词汇表，这与人类语言学习中展示的适应性词汇获取过程存在差距。为弥合这一差距，我们引入了词汇曲面学习方法，该方法通过与词汇表大小呈对数线性关系的效率提升，改进了预训练效率。我们的方法交替进行以熵为导向的词汇扩展与模型优化，使得模型能够在多种分词粒度下学习可迁移的表示。这一方法自然而然地产生了一种最优的计算分配模式：较长的词元捕获可预测的内容，而较短的词元则侧重于更加复杂且难以预测的上下文。在小型GPT模型上的实验表明，动态分词可以提高缩放效率，进一步验证了动态分词的有效性。我们已公开了我们的代码以支持进一步的研究，并计划将实验扩展到更大的模型和多种领域。 

---
# FactFlow: Automatic Fact Sheet Generation and Customization from Tabular Dataset via AI Chain Design & Implementation 

**Title (ZH)**: FactFlow：通过AI链设计与实现从表格数据集中自动生成和定制事实表的功能 

**Authors**: Minh Duc Vu, Jieshan Chen, Zhenchang Xing, Qinghua Lu, Xiwei Xu, Qian Fu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17909)  

**Abstract**: With the proliferation of data across various domains, there is a critical demand for tools that enable non-experts to derive meaningful insights without deep data analysis skills. To address this need, existing automatic fact sheet generation tools offer heuristic-based solutions to extract facts and generate stories. However, they inadequately grasp the semantics of data and struggle to generate narratives that fully capture the semantics of the dataset or align the fact sheet with specific user needs. Addressing these shortcomings, this paper introduces \tool, a novel tool designed for the automatic generation and customisation of fact sheets. \tool applies the concept of collaborative AI workers to transform raw tabular dataset into comprehensive, visually compelling fact sheets. We define effective taxonomy to profile AI worker for specialised tasks. Furthermore, \tool empowers users to refine these fact sheets through intuitive natural language commands, ensuring the final outputs align closely with individual preferences and requirements. Our user evaluation with 18 participants confirms that \tool not only surpasses state-of-the-art baselines in automated fact sheet production but also provides a positive user experience during customization tasks. 

**Abstract (ZH)**: 随着各领域数据的激增，非专家用户对不需要深厚数据分析技能就能提取有价值的见解的需求日益迫切。为满足这一需求，现有的自动事实表生成工具提供了基于启发式的方法来提取事实并生成故事。然而，这些工具在理解数据语义方面存在不足，并且难以生成能够充分捕捉数据集语义或与用户特定需求相匹配的叙述。为解决这些不足，本文介绍了一种名为 \tool 的新型工具，该工具旨在自动生成和定制事实表。\tool 利用协作人工智能工人的概念，将原始的表格数据转换为全面且视觉吸引人的事实表。我们定义了有效的分类法以针对特定任务对人工智能工人进行分类。此外，\tool 允许用户通过直观的自然语言命令精炼这些事实表，确保最终输出紧密符合个人偏好和需求。我们的用户评估（参与者人数为18人）表明，\tool 不仅在自动事实表生成方面超越了最先进的基线方法，还在定制任务中提供了积极的用户体验。 

---
# Knowledge-enhanced Multimodal ECG Representation Learning with Arbitrary-Lead Inputs 

**Title (ZH)**: 带有任意导联输入的知识增强多模态心电图表示学习 

**Authors**: Che Liu, Cheng Ouyang, Zhongwei Wan, Haozhe Wang, Wenjia Bai, Rossella Arcucci  

**Link**: [PDF](https://arxiv.org/pdf/2502.17900)  

**Abstract**: Recent advances in multimodal ECG representation learning center on aligning ECG signals with paired free-text reports. However, suboptimal alignment persists due to the complexity of medical language and the reliance on a full 12-lead setup, which is often unavailable in under-resourced settings. To tackle these issues, we propose **K-MERL**, a knowledge-enhanced multimodal ECG representation learning framework. **K-MERL** leverages large language models to extract structured knowledge from free-text reports and employs a lead-aware ECG encoder with dynamic lead masking to accommodate arbitrary lead inputs. Evaluations on six external ECG datasets show that **K-MERL** achieves state-of-the-art performance in zero-shot classification and linear probing tasks, while delivering an average **16%** AUC improvement over existing methods in partial-lead zero-shot classification. 

**Abstract (ZH)**: 近年来，多模态心电图（ECG）表示学习的进展主要集中在将ECG信号与配对的自由文本报告对齐。然而，由于医学语言的复杂性和依赖完整的12导联设置，这种对齐仍然存在不足。特别是在资源不足的环境中，12导联设置往往不可用。为了解决这些问题，我们提出了一种基于知识增强的多模态ECG表示学习框架——**K-MERL**。**K-MERL** 利用大规模语言模型从自由文本报告中提取结构化的知识，并采用具有动态导联掩码的导联感知ECG编码器，以适应任意导联输入。在六个外部ECG数据集上的评估结果显示，**K-MERL** 在零样本分类和线性探针任务中取得了最先进的性能，在部分导联的零样本分类中，相对于现有方法的平均AUC改进达到了16%。 

---
# VeriPlan: Integrating Formal Verification and LLMs into End-User Planning 

**Title (ZH)**: VeriPlan: 将形式化验证与大规模语言模型集成到最终用户规划中 

**Authors**: Christine Lee, David Porfirio, Xinyu Jessica Wang, Kevin Zhao, Bilge Mutlu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17898)  

**Abstract**: Automated planning is traditionally the domain of experts, utilized in fields like manufacturing and healthcare with the aid of expert planning tools. Recent advancements in LLMs have made planning more accessible to everyday users due to their potential to assist users with complex planning tasks. However, LLMs face several application challenges within end-user planning, including consistency, accuracy, and user trust issues. This paper introduces VeriPlan, a system that applies formal verification techniques, specifically model checking, to enhance the reliability and flexibility of LLMs for end-user planning. In addition to the LLM planner, VeriPlan includes three additional core features -- a rule translator, flexibility sliders, and a model checker -- that engage users in the verification process. Through a user study (n=12), we evaluate VeriPlan, demonstrating improvements in the perceived quality, usability, and user satisfaction of LLMs. Our work shows the effective integration of formal verification and user-control features with LLMs for end-user planning tasks. 

**Abstract (ZH)**: 自动规划 traditionally 是专家的领域，通常在制造和医疗等领域的规划工具辅助下得到应用。近期语言大规模模型（LLM）的发展使得复杂规划任务变得更加易于普通用户使用。然而，LLM 在最终用户规划中的应用仍面临若干挑战，包括一致性、准确性和用户信任问题。本文介绍了一种名为 VeriPlan 的系统，该系统通过应用形式化验证技术（特别是模型检测）来增强 LLM 在最终用户规划中的可靠性和灵活性。除了 LLM 规划器外，VeriPlan 还包括三个核心功能——规则翻译器、灵活性滑块和模型检查器，这些功能将用户纳入验证过程。通过一项用户研究（n=12），我们评估了 VeriPlan，显示了对 LLM 的感知质量、易用性和用户满意度的改进。我们的工作展示了将形式化验证和用户控制功能有效集成到 LLM 中，以应用于最终用户规划任务。 

---
# Sample-efficient diffusion-based control of complex nonlinear systems 

**Title (ZH)**: 基于扩散的高效样本复杂非线性系统控制 

**Authors**: Hongyi Chen, Jingtao Ding, Jianhai Shu, Xinchun Yu, Xiaojun Liang, Yong Li, Xiao-Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17893)  

**Abstract**: Complex nonlinear system control faces challenges in achieving sample-efficient, reliable performance. While diffusion-based methods have demonstrated advantages over classical and reinforcement learning approaches in long-term control performance, they are limited by sample efficiency. This paper presents SEDC (Sample-Efficient Diffusion-based Control), a novel diffusion-based control framework addressing three core challenges: high-dimensional state-action spaces, nonlinear system dynamics, and the gap between non-optimal training data and near-optimal control solutions. Through three innovations - Decoupled State Diffusion, Dual-Mode Decomposition, and Guided Self-finetuning - SEDC achieves 39.5\%-49.4\% better control accuracy than baselines while using only 10\% of the training samples, as validated across three complex nonlinear dynamic systems. Our approach represents a significant advancement in sample-efficient control of complex nonlinear systems. The implementation of the code can be found at this https URL. 

**Abstract (ZH)**: 复杂非线性系统的控制在实现高效样本利用和可靠性能方面面临挑战。虽然以扩散为基础的方法在长期控制性能上比传统和强化学习方法表现出优势，但它们在样本效率方面仍有局限性。本文提出了SEDN（Sample-Efficient Diffusion-based Control），一种解决三大核心挑战的新型扩散基础控制框架：高维状态-动作空间、非线性系统动力学以及非最优训练数据与近最优控制解决方案之间的差距。通过三项创新——解耦状态扩散、双模式分解和引导式自我微调——SEDN在三个复杂非线性动力系统上的测试验证了仅使用10%的训练样本就能实现39.5%-49.4%的控制精度提高。我们的方法在复杂非线性系统的样本高效控制方面取得了重要进展。该代码的实现可以在以下网址找到：[链接]。 

---
# Arrhythmia Classification from 12-Lead ECG Signals Using Convolutional and Transformer-Based Deep Learning Models 

**Title (ZH)**: 使用卷积和transformer基深层学习模型的12导联心电图心律失常分类 

**Authors**: Andrei Apostol, Maria Nutu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17887)  

**Abstract**: In Romania, cardiovascular problems are the leading cause of death, accounting for nearly one-third of annual fatalities. The severity of this situation calls for innovative diagnosis method for cardiovascular diseases. This article aims to explore efficient, light-weight and rapid methods for arrhythmia diagnosis, in resource-constrained healthcare settings. Due to the lack of Romanian public medical data, we trained our systems using international public datasets, having in mind that the ECG signals are the same regardless the patients' nationality. Within this purpose, we combined multiple datasets, usually used in the field of arrhythmias classification: PTB-XL electrocardiography dataset , PTB Diagnostic ECG Database, China 12-Lead ECG Challenge Database, Georgia 12-Lead ECG Challenge Database, and St. Petersburg INCART 12-lead Arrhythmia Database. For the input data, we employed ECG signal processing methods, specifically a variant of the Pan-Tompkins algorithm, useful in arrhythmia classification because it provides a robust and efficient method for detecting QRS complexes in ECG signals. Additionally, we used machine learning techniques, widely used for the task of classification, including convolutional neural networks (1D CNNs, 2D CNNs, ResNet) and Vision Transformers (ViTs). The systems were evaluated in terms of accuracy and F1 score. We annalysed our dataset from two perspectives. First, we fed the systems with the ECG signals and the GRU-based 1D CNN model achieved the highest accuracy of 93.4% among all the tested architectures. Secondly, we transformed ECG signals into images and the CNN2D model achieved an accuracy of 92.16%. 

**Abstract (ZH)**: 在罗马尼亚，心血管问题是最主要的死亡原因，占每年死亡人数的近三分之一。这一严峻形势迫切需要创新的心血管疾病诊断方法。本文旨在探讨适用于资源受限医疗环境的高效、轻量级和快速心律失常诊断方法。由于缺乏罗马尼亚公共医疗数据，我们使用国际公开数据集进行系统训练，考虑到心电图（ECG）信号对不同国籍患者是一致的。为实现这一目标，我们结合了通常用于心律失常分类的多个数据集：PTB-XL心电图数据集、PTB诊断心电图数据库、中国12导联心电图挑战数据库、格鲁吉亚12导联心电图挑战数据库以及圣彼得堡INCART 12导联心律失常数据库。对于输入数据，我们采用了心电图信号处理方法，特别是Pan-Tompkins算法的一种变体，该算法在心律失常分类中非常有用，因为它提供了一种稳健且高效的方法来检测ECG信号中的QRS波群。此外，我们还使用了广泛用于分类任务的机器学习技术，包括一维卷积神经网络（1D CNN）、二维卷积神经网络（2D CNN）、残差网络（ResNet），以及视觉转换器（Vision Transformers，ViTs）。系统在准确率和F1分数方面进行了评估。我们从两个方面分析了数据集。首先，我们将ECG信号输入系统中，基于GRU的一维卷积神经网络模型在所有测试架构中实现了最高的准确率93.4%。其次，我们将ECG信号转换为图像，二维卷积神经网络模型实现了92.16%的准确率。 

---
# A graph neural network-based multispectral-view learning model for diabetic macular ischemia detection from color fundus photographs 

**Title (ZH)**: 基于图神经网络的多光谱视网膜图像糖尿病黄斑缺血检测模型 

**Authors**: Qinghua He, Hongyang Jiang, Danqi Fang, Dawei Yang, Truong X. Nguyen, Anran Ran, Clement C. Tham, Simon K. H. Szeto, Sobha Sivaprasad, Carol Y. Cheung  

**Link**: [PDF](https://arxiv.org/pdf/2502.17886)  

**Abstract**: Diabetic macular ischemia (DMI), marked by the loss of retinal capillaries in the macular area, contributes to vision impairment in patients with diabetes. Although color fundus photographs (CFPs), combined with artificial intelligence (AI), have been extensively applied in detecting various eye diseases, including diabetic retinopathy (DR), their applications in detecting DMI remain unexplored, partly due to skepticism among ophthalmologists regarding its feasibility. In this study, we propose a graph neural network-based multispectral view learning (GNN-MSVL) model designed to detect DMI from CFPs. The model leverages higher spectral resolution to capture subtle changes in fundus reflectance caused by ischemic tissue, enhancing sensitivity to DMI-related features. The proposed approach begins with computational multispectral imaging (CMI) to reconstruct 24-wavelength multispectral fundus images from CFPs. ResNeXt101 is employed as the backbone for multi-view learning to extract features from the reconstructed images. Additionally, a GNN with a customized jumper connection strategy is designed to enhance cross-spectral relationships, facilitating comprehensive and efficient multispectral view learning. The study included a total of 1,078 macula-centered CFPs from 1,078 eyes of 592 patients with diabetes, of which 530 CFPs from 530 eyes of 300 patients were diagnosed with DMI. The model achieved an accuracy of 84.7 percent and an area under the receiver operating characteristic curve (AUROC) of 0.900 (95 percent CI: 0.852-0.937) on eye-level, outperforming both the baseline model trained from CFPs and human experts (p-values less than 0.01). These findings suggest that AI-based CFP analysis holds promise for detecting DMI, contributing to its early and low-cost screening. 

**Abstract (ZH)**: 糖尿病黄斑缺血（DMI），表现为视网膜黄斑区域毛细血管丢失，是糖尿病患者视力受损的一个重要原因。虽然结合了人工智能（AI）的彩色视网膜 funds 图像（CFPs）在检测各种眼病方面已得到广泛应用，包括糖尿病视网膜病变（DR），但其在检测DMI的应用尚未被充分探索，部分原因是眼科医生对这种方法可行性的怀疑。在此研究中，我们提出了一种基于图神经网络的多光谱视图学习（GNN-MSVL）模型，旨在从CFPs中检测DMI。该模型利用更高的光谱分辨率来捕捉由缺血组织引起的视网膜反光的细微变化，从而增强对DMI相关特征的敏感性。该方法首先采用计算多光谱成像（CMI）从CFPs中重建出24波长的多光谱视网膜图像。ResNeXt101被用作多视图学习的骨干网络，从重建图像中提取特征。此外，我们设计了一种定制化的跳跃连接策略的GNN，以增强跨光谱关系，促进全面而高效的多光谱视图学习。研究包括了来自592名糖尿病患者的1,078张以黄斑为中心的CFPs，其中300名患者中的530张CFPs被诊断为DMI。该模型在眼水平上的准确率为84.7%，面积下曲线（AUC-ROC）下的面积为0.900（95%CI：0.852-0.937），优于基于CFPs训练的基本模型和人类专家（p值<0.01）。这些发现表明，基于AI的CFPs分析有潜力用于检测DMI，有助于其早期和低成本筛查。 

---
# From underwater to aerial: a novel multi-scale knowledge distillation approach for coral reef monitoring 

**Title (ZH)**: 从水下到空中：一种新型多尺度知识蒸馏方法用于珊瑚礁监测 

**Authors**: Matteo Contini, Victor Illien, Julien Barde, Sylvain Poulain, Serge Bernard, Alexis Joly, Sylvain Bonhommeau  

**Link**: [PDF](https://arxiv.org/pdf/2502.17883)  

**Abstract**: Drone-based remote sensing combined with AI-driven methodologies has shown great potential for accurate mapping and monitoring of coral reef ecosystems. This study presents a novel multi-scale approach to coral reef monitoring, integrating fine-scale underwater imagery with medium-scale aerial imagery. Underwater images are captured using an Autonomous Surface Vehicle (ASV), while aerial images are acquired with an aerial drone. A transformer-based deep-learning model is trained on underwater images to detect the presence of 31 classes covering various coral morphotypes, associated fauna, and habitats. These predictions serve as annotations for training a second model applied to aerial images. The transfer of information across scales is achieved through a weighted footprint method that accounts for partial overlaps between underwater image footprints and aerial image tiles. The results show that the multi-scale methodology successfully extends fine-scale classification to larger reef areas, achieving a high degree of accuracy in predicting coral morphotypes and associated habitats. The method showed a strong alignment between underwater-derived annotations and ground truth data, reflected by an AUC (Area Under the Curve) score of 0.9251. This shows that the integration of underwater and aerial imagery, supported by deep-learning models, can facilitate scalable and accurate reef assessments. This study demonstrates the potential of combining multi-scale imaging and AI to facilitate the monitoring and conservation of coral reefs. Our approach leverages the strengths of underwater and aerial imagery, ensuring the precision of fine-scale analysis while extending it to cover a broader reef area. 

**Abstract (ZH)**: 基于无人机的遥感技术结合人工智能驱动的方法在珊瑚礁生态系统的精确制图和监测方面显示出巨大潜力。本研究提出了一种新颖的多尺度监测方法，将精细尺度的水下影像与中尺度的航空影像相结合。水下影像使用自主水面车辆（ASV）捕获，而航空影像则由无人机获取。一种基于变压器的深度学习模型已在水下影像上进行训练，以检测包括31个类别的各种珊瑚形态、相关生物和栖息地在内的存在情况。这些预测作为注释，用于训练应用于航空影像的第二个模型。通过一种加权足迹方法，该方法考虑了水下影像足迹与航空影像块之间的部分重叠，实现了跨尺度的信息传递。结果表明，多尺度方法成功地将精细尺度分类扩展到更大的珊瑚礁区域，达到了在预测珊瑚形态和相关栖息地方面的高准确度。该方法显示出水下源注释与地面真实数据之间存在强一致性的迹象，如曲线下面积（AUC）得分为0.9251所示。这表明，结合水下和航空影像并通过深度学习模型支持的方法，可以促进珊瑚礁的可扩展和准确评估。本研究展示了多尺度成像和人工智能相结合在促进珊瑚礁监测和保护方面的潜力。我们的方法利用了水下和航空影像的优点，确保了精细尺度分析的精确性，同时将其扩展到覆盖更广泛的珊瑚礁区域。 

---
# Contrastive Learning with Nasty Noise 

**Title (ZH)**: 带有恶劣噪声的对比学习 

**Authors**: Ziruo Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.17872)  

**Abstract**: Contrastive learning has emerged as a powerful paradigm for self-supervised representation learning. This work analyzes the theoretical limits of contrastive learning under nasty noise, where an adversary modifies or replaces training samples. Using PAC learning and VC-dimension analysis, lower and upper bounds on sample complexity in adversarial settings are established. Additionally, data-dependent sample complexity bounds based on the l2-distance function are derived. 

**Abstract (ZH)**: 对比学习已成为一种强大的自监督表示学习范式。本文分析了在恶劣噪声环境下对比学习的理论极限，其中攻击者会修改或替换训练样本。通过运用PAC学习理论和VC维分析，建立了对抗性设置下样本复杂度的上下界。此外，基于欧式距离函数的数据依赖样本复杂度边界也被推导出来。 

---
# ASurvey: Spatiotemporal Consistency in Video Generation 

**Title (ZH)**: 《时空一致性在视频生成中的综述》 

**Authors**: Zhiyu Yin, Kehai Chen, Xuefeng Bai, Ruili Jiang, Juntao Li, Hongdong Li, Jin Liu, Yang Xiang, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17863)  

**Abstract**: Video generation, by leveraging a dynamic visual generation method, pushes the boundaries of Artificial Intelligence Generated Content (AIGC). Video generation presents unique challenges beyond static image generation, requiring both high-quality individual frames and temporal coherence to maintain consistency across the spatiotemporal sequence. Recent works have aimed at addressing the spatiotemporal consistency issue in video generation, while few literature review has been organized from this perspective. This gap hinders a deeper understanding of the underlying mechanisms for high-quality video generation. In this survey, we systematically review the recent advances in video generation, covering five key aspects: foundation models, information representations, generation schemes, post-processing techniques, and evaluation metrics. We particularly focus on their contributions to maintaining spatiotemporal consistency. Finally, we discuss the future directions and challenges in this field, hoping to inspire further efforts to advance the development of video generation. 

**Abstract (ZH)**: 利用动态视觉生成方法推动人工智能生成内容（AIGC）边界的技术视频生成，提出了超越静态图像生成的独特挑战，不仅需要高质量的单帧图像，还需要在时空序列中保持时间连贯性。近期的研究工作致力于解决视频生成过程中的时空一致性问题，但很少有文献从这一视角进行综述。这一空白限制了我们对高质量视频生成潜在机制的深入了解。在本文综述中，我们系统地回顾了视频生成领域的最新进展，涵盖了五个关键方面：基础模型、信息表示、生成方案、后处理技术和评估指标。特别关注它们在维护时空一致性方面的贡献。最后，我们讨论了该领域未来的研究方向和挑战，希望能激发进一步的努力，促进视频生成的发展。 

---
# Say Less, Mean More: Leveraging Pragmatics in Retrieval-Augmented Generation 

**Title (ZH)**: 少说多意：利用启发式生成中的语用学优势 

**Authors**: Haris Riaz, Ellen Riloff, Mihai Surdeanu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17839)  

**Abstract**: We propose a simple, unsupervised method that injects pragmatic principles in retrieval-augmented generation (RAG) frameworks such as Dense Passage Retrieval~\cite{karpukhin2020densepassageretrievalopendomain} to enhance the utility of retrieved contexts. Our approach first identifies which sentences in a pool of documents retrieved by RAG are most relevant to the question at hand, cover all the topics addressed in the input question and no more, and then highlights these sentences within their context, before they are provided to the LLM, without truncating or altering the context in any other way. We show that this simple idea brings consistent improvements in experiments on three question answering tasks (ARC-Challenge, PubHealth and PopQA) using five different LLMs. It notably enhances relative accuracy by up to 19.7\% on PubHealth and 10\% on ARC-Challenge compared to a conventional RAG system. 

**Abstract (ZH)**: 我们提出了一种简单且无监督的方法，在检索增强生成（RAG）框架（如密集段落检索~\cite{karpukhin2020densepassageretrievalopendomain}）中注入实用原则，以提高检索到的上下文的实用性。我们的方法首先在由RAG检索出的文档池中识别出与当前问题最相关的句子，这些句子涵盖了输入问题中涉及的所有主题但不包含更多内容，然后在将这些句子呈现给大语言模型（LLM）之前，高亮显示这些句子的上下文，而不会以任何方式截断或修改上下文。实验结果显示，这一简单思想在使用五种不同大语言模型对三个问答任务（ARC-Challenge、PubHealth 和 PopQA）进行的测试中带来了一致的改进。在PubHealth任务上相对准确度提高了高达19.7%，在ARC-Challenge任务上提高了10%。与传统RAG系统相比，这一方法显著提升了相对准确度。 

---
# MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks 

**Title (ZH)**: MM-PoisonRAG: 针对多模态RAG的局部和全局投毒攻击破解方法 

**Authors**: Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-wei Chang, Daniel Kang, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.17832)  

**Abstract**: Multimodal large language models (MLLMs) equipped with Retrieval Augmented Generation (RAG) leverage both their rich parametric knowledge and the dynamic, external knowledge to excel in tasks such as Question Answering. While RAG enhances MLLMs by grounding responses in query-relevant external knowledge, this reliance poses a critical yet underexplored safety risk: knowledge poisoning attacks, where misinformation or irrelevant knowledge is intentionally injected into external knowledge bases to manipulate model outputs to be incorrect and even harmful. To expose such vulnerabilities in multimodal RAG, we propose MM-PoisonRAG, a novel knowledge poisoning attack framework with two attack strategies: Localized Poisoning Attack (LPA), which injects query-specific misinformation in both text and images for targeted manipulation, and Globalized Poisoning Attack (GPA) to provide false guidance during MLLM generation to elicit nonsensical responses across all queries. We evaluate our attacks across multiple tasks, models, and access settings, demonstrating that LPA successfully manipulates the MLLM to generate attacker-controlled answers, with a success rate of up to 56% on MultiModalQA. Moreover, GPA completely disrupts model generation to 0% accuracy with just a single irrelevant knowledge injection. Our results highlight the urgent need for robust defenses against knowledge poisoning to safeguard multimodal RAG frameworks. 

**Abstract (ZH)**: 配备了检索增强生成（RAG）的多模态大型语言模型（MLLMs）结合了其丰富的参数知识和动态的外部知识，使其在问答等任务中表现出色。虽然RAG通过将响应与查询相关的外部知识相链接来增强MLLMs，但这种依赖性带来了一个关键且尚未充分探索的安全风险：知识中毒攻击，其中故意向外部知识库注入错误信息或无关知识，以操控模型输出错误甚至有害的结果。为了揭示多模态RAG中的此类漏洞，我们提出了一种新颖的知识中毒攻击框架——MM-PoisonRAG，并提出两种攻击策略：局部中毒攻击（LPA），它在文本和图像中注入查询相关的错误信息以进行定向操纵；以及全球化中毒攻击（GPA），在MLLM生成过程中提供虚假指导，导致所有查询均产生成分荒谬的响应。我们在多个任务、模型和访问设置下评估了我们的攻击，结果表明，LPA能够成功操控MLLM生成攻击者控制的答案，在MultiModalQA上的成功率最高可达56%。此外，GPA仅通过一次无关知识的注入即可完全破坏模型生成，使其准确率为零。我们的研究结果强调了针对知识中毒攻击建立 robust 防御措施的迫切需求，以保障多模态RAG框架的安全。 

---
# CAML: Collaborative Auxiliary Modality Learning for Multi-Agent Systems 

**Title (ZH)**: CAML：多智能体系统中的协作辅助模态学习 

**Authors**: Rui Liu, Yu Shen, Peng Gao, Pratap Tokekar, Ming Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.17821)  

**Abstract**: Multi-modality learning has become a crucial technique for improving the performance of machine learning applications across domains such as autonomous driving, robotics, and perception systems. While existing frameworks such as Auxiliary Modality Learning (AML) effectively utilize multiple data sources during training and enable inference with reduced modalities, they primarily operate in a single-agent context. This limitation is particularly critical in dynamic environments, such as connected autonomous vehicles (CAV), where incomplete data coverage can lead to decision-making blind spots. To address these challenges, we propose Collaborative Auxiliary Modality Learning ($\textbf{CAML}$), a novel multi-agent multi-modality framework that enables agents to collaborate and share multimodal data during training while allowing inference with reduced modalities per agent during testing. We systematically analyze the effectiveness of $\textbf{CAML}$ from the perspective of uncertainty reduction and data coverage, providing theoretical insights into its advantages over AML. Experimental results in collaborative decision-making for CAV in accident-prone scenarios demonstrate that \ours~achieves up to a ${\bf 58.13}\%$ improvement in accident detection. Additionally, we validate $\textbf{CAML}$ on real-world aerial-ground robot data for collaborative semantic segmentation, achieving up to a ${\bf 10.61}\%$ improvement in mIoU. 

**Abstract (ZH)**: 多模态学习已成为提升跨领域（如自主驾驶、机器人和感知系统）机器学习应用性能的关键技术。现有的框架，如辅助模态学习（AML），能够有效利用多个数据源进行训练，并在推理时使用减少的模态数量，但这些框架主要在单个代理的上下文中运行。这一限制在动态环境中尤为关键，例如连接的自动驾驶车辆（CAV），因为不完整的数据覆盖可能导致决策盲点。为了解决这些挑战，我们提出了一种新的多代理多模态框架——协作辅助模态学习（$\textbf{CAML}$），该框架允许代理在训练过程中协作并共享多模态数据，在测试过程中每个代理可以使用减少的模态数量进行推理。我们从不确定性和数据覆盖的角度系统地分析了$\textbf{CAML}$的有效性，提供了其相对于AML的优势的理论见解。在事故多发场景下进行的合作决策实验结果表明，$\textbf{CAML}$在事故检测方面的性能提高了$\textbf{58.13}\%$。此外，我们还在实际的空地机器人数据上验证了$\textbf{CAML}$在协作语义分割上的有效性，取得了$\textbf{mIoU}$提高了$\textbf{10.61}\%$的成果。 

---
# An Overview of Large Language Models for Statisticians 

**Title (ZH)**: 统计学家视角下的大型语言模型概览 

**Authors**: Wenlong Ji, Weizhe Yuan, Emily Getzen, Kyunghyun Cho, Michael I. Jordan, Song Mei, Jason E Weston, Weijie J. Su, Jing Xu, Linjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17814)  

**Abstract**: Large Language Models (LLMs) have emerged as transformative tools in artificial intelligence (AI), exhibiting remarkable capabilities across diverse tasks such as text generation, reasoning, and decision-making. While their success has primarily been driven by advances in computational power and deep learning architectures, emerging problems -- in areas such as uncertainty quantification, decision-making, causal inference, and distribution shift -- require a deeper engagement with the field of statistics. This paper explores potential areas where statisticians can make important contributions to the development of LLMs, particularly those that aim to engender trustworthiness and transparency for human users. Thus, we focus on issues such as uncertainty quantification, interpretability, fairness, privacy, watermarking and model adaptation. We also consider possible roles for LLMs in statistical analysis. By bridging AI and statistics, we aim to foster a deeper collaboration that advances both the theoretical foundations and practical applications of LLMs, ultimately shaping their role in addressing complex societal challenges. 

**Abstract (ZH)**: 大型语言模型（LLMs）作为人工智能（AI）中的变革工具，展现出了在文本生成、推理和决策等多个任务中的出色能力。尽管它们的成功主要得益于计算能力的提升和深度学习架构的进步，但在不确定性量化、决策制定、因果推断和分布转移等领域出现的问题，则需要统计学领域的更深入参与。本文探讨了统计学家在LLMs发展中可能作出的重要贡献，特别是为了增强人类用户对这些模型的信任度和透明度方面的工作。因此，我们重点关注不确定性量化、可解释性、公平性、隐私、水印和模型适应等问题。我们还考虑了LLMs在统计分析中的潜在角色。通过跨越AI和统计学的界限，我们的目标是促进更深入的合作，从而推进LLMs的理论基础及其实际应用，最终帮助这些模型在解决复杂的社会发展问题中发挥更大作用。 

---
# Research on Enhancing Cloud Computing Network Security using Artificial Intelligence Algorithms 

**Title (ZH)**: 使用人工智能算法提升云计算网络安全性研究 

**Authors**: Yuqing Wang, Xiao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17801)  

**Abstract**: Cloud computing environments are increasingly vulnerable to security threats such as distributed denial-of-service (DDoS) attacks and SQL injection. Traditional security mechanisms, based on rule matching and feature recognition, struggle to adapt to evolving attack strategies. This paper proposes an adaptive security protection framework leveraging deep learning to construct a multi-layered defense architecture. The proposed system is evaluated in a real-world business environment, achieving a detection accuracy of 97.3%, an average response time of 18 ms, and an availability rate of 99.999%. Experimental results demonstrate that the proposed method significantly enhances detection accuracy, response efficiency, and resource utilization, offering a novel and effective approach to cloud computing security. 

**Abstract (ZH)**: 云计算环境越来越容易受到分布式拒绝服务（DDoS）攻击和SQL注入等安全威胁的影响。传统的基于规则匹配和特征识别的安全机制难以适应不断演变的攻击策略。本文提出了一种利用深度学习构建自适应安全保护框架的方法，以构建多层次的防御体系结构。所提出的系统在实际商业环境中进行了评估，实现了97.3%的检测准确率、平均响应时间18毫秒以及99.999%的可用率。实验结果表明，所提出的方法显著提高了检测准确率、响应效率和资源利用率，为云计算安全提供了新颖且有效的解决方案。 

---
# Synthia: Novel Concept Design with Affordance Composition 

**Title (ZH)**: Synthia：基于功能组合的新颖概念设计 

**Authors**: Xiaomeng Jin, Hyeonjeong Ha, Jeonghwan Kim, Jiateng Liu, Zhenhailong Wang, Khanh Duy Nguyen, Ansel Blume, Nanyun Peng, Kai-wei Chang, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.17793)  

**Abstract**: Text-to-image (T2I) models enable rapid concept design, making them widely used in AI-driven design. While recent studies focus on generating semantic and stylistic variations of given design concepts, functional coherence--the integration of multiple affordances into a single coherent concept--remains largely overlooked. In this paper, we introduce SYNTHIA, a framework for generating novel, functionally coherent designs based on desired affordances. Our approach leverages a hierarchical concept ontology that decomposes concepts into parts and affordances, serving as a crucial building block for functionally coherent design. We also develop a curriculum learning scheme based on our ontology that contrastively fine-tunes T2I models to progressively learn affordance composition while maintaining visual novelty. To elaborate, we (i) gradually increase affordance distance, guiding models from basic concept-affordance association to complex affordance compositions that integrate parts of distinct affordances into a single, coherent form, and (ii) enforce visual novelty by employing contrastive objectives to push learned representations away from existing concepts. Experimental results show that SYNTHIA outperforms state-of-the-art T2I models, demonstrating absolute gains of 25.1% and 14.7% for novelty and functional coherence in human evaluation, respectively. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，要符合学术规范：

文本到图像（T2I）模型能够快速进行概念设计，使其在AI驱动的设计中得到了广泛应用。虽然最近的研究集中在生成给定设计概念的语义和风格变化上，但功能一致性——将多种功能整合到单一连贯的概念中——仍然被广泛关注。在本文中，我们提出了SYNTHIA框架，该框架基于所需的功能生成新颖的功能连贯设计。我们的方法利用了一个分层的概念本体，该本体将概念分解为部件和功能，成为功能连贯设计的关键构建模块。我们还基于本体开发了一种层次学习方案，通过对比微调T2I模型，逐步学习功能组合，同时保持视觉新颖性。具体来说，我们（i）逐步增加功能距离，从简单的概念-功能关联引导模型到复杂的功能组合，这些组合将不同功能的部件整合到单一的连贯形式中，（ii）通过应用对比学习目标来增加视觉新颖性，促使学习表示远离现有概念。实验结果表明，SYNTHIA优于最先进的T2I模型，在人类评价中，新颖性和功能性连贯性的绝对增益分别为25.1%和14.7%。 

---
# AIR: Complex Instruction Generation via Automatic Iterative Refinement 

**Title (ZH)**: AIR：通过自动迭代 refinement 的复杂指令生成 

**Authors**: Wei Liu, Yancheng He, Hui Huang, Chengwei Hu, Jiaheng Liu, Shilong Li, Wenbo Su, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.17787)  

**Abstract**: With the development of large language models, their ability to follow simple instructions has significantly improved. However, adhering to complex instructions remains a major challenge. Current approaches to generating complex instructions are often irrelevant to the current instruction requirements or suffer from limited scalability and diversity. Moreover, methods such as back-translation, while effective for simple instruction generation, fail to leverage the rich contents and structures in large web corpora. In this paper, we propose a novel automatic iterative refinement framework to generate complex instructions with constraints, which not only better reflects the requirements of real scenarios but also significantly enhances LLMs' ability to follow complex instructions. The AIR framework consists of two stages: (1)Generate an initial instruction from a document; (2)Iteratively refine instructions with LLM-as-judge guidance by comparing the model's output with the document to incorporate valuable constraints. Finally, we construct the AIR-10K dataset with 10K complex instructions and demonstrate that instructions generated with our approach significantly improve the model's ability to follow complex instructions, outperforming existing methods for instruction generation. 

**Abstract (ZH)**: 随着大规模语言模型的发展，它们遵循简单指令的能力显著提高。然而，遵循复杂指令仍然是一个主要挑战。当前生成复杂指令的方法往往与当前的指令需求无关，或者受到有限的可扩展性和多样性的限制。此外，如回译等方法尽管对生成简单指令有效，但未能充分利用大型网页语料库中的丰富内容和结构。在本文中，我们提出了一种新颖的自动迭代 refinement 框架，以生成具有约束的复杂指令，这不仅更准确地反映了真实场景的需求，还显著增强了大语言模型遵循复杂指令的能力。AIR 框架包括两个阶段：（1）从文档中生成初始指令；（2）通过将模型输出与文档进行比较，在大语言模型（LLM）作为裁判的指导下，迭代优化指令，以纳入有价值的信息约束。最后，我们构建了包含10,000条复杂指令的AIR-10K数据集，并证明使用我们方法生成的指令显著提升了模型遵循复杂指令的能力，优于现有的指令生成方法。 

---
# Uncertainty Quantification for LLM-Based Survey Simulations 

**Title (ZH)**: 基于LLM的调查模拟中的不确定性量化 

**Authors**: Chengpiao Huang, Yuhang Wu, Kaizheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17773)  

**Abstract**: We investigate the reliable use of simulated survey responses from large language models (LLMs) through the lens of uncertainty quantification. Our approach converts synthetic data into confidence sets for population parameters of human responses, addressing the distribution shift between the simulated and real populations. A key innovation lies in determining the optimal number of simulated responses: too many produce overly narrow confidence sets with poor coverage, while too few yield excessively loose estimates. To resolve this, our method adaptively selects the simulation sample size, ensuring valid average-case coverage guarantees. It is broadly applicable to any LLM, irrespective of its fidelity, and any procedure for constructing confidence sets. Additionally, the selected sample size quantifies the degree of misalignment between the LLM and the target human population. We illustrate our method on real datasets and LLMs. 

**Abstract (ZH)**: 我们通过不确定性量化这一视角，调查了大型语言模型（LLMs）模拟调查响应的可靠使用方法。我们的方法将合成数据转换为人类响应总体参数的信心区间，以解决模拟人群与真实人群之间的分布差异问题。我们的一项关键创新在于确定了模拟响应的最佳数量：模拟反应过多会导致信心区间过于狭窄且覆盖不足，而模拟反应过少则会导致过于宽松的估计结果。为解决这一问题，我们的方法能够自适应地选择模拟样本量，确保平均情形下的有效覆盖保证。该方法适用于任何大型语言模型，不论其保真度如何，以及任何构建信心区间的方法。此外，所选样本量可以量化大型语言模型与目标人类群体之间的不一致程度。我们通过实际数据集和大型语言模型来说明该方法的应用。 

---
# Sample Selection via Contrastive Fragmentation for Noisy Label Regression 

**Title (ZH)**: 基于对比片段化的选择性采样方法用于噪声标签回归 

**Authors**: Chris Dongjoo Kim, Sangwoo Moon, Jihwan Moon, Dongyeon Woo, Gunhee Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.17771)  

**Abstract**: As with many other problems, real-world regression is plagued by the presence of noisy labels, an inevitable issue that demands our attention. Fortunately, much real-world data often exhibits an intrinsic property of continuously ordered correlations between labels and features, where data points with similar labels are also represented with closely related features. In response, we propose a novel approach named ConFrag, where we collectively model the regression data by transforming them into disjoint yet contrasting fragmentation pairs. This enables the training of more distinctive representations, enhancing the ability to select clean samples. Our ConFrag framework leverages a mixture of neighboring fragments to discern noisy labels through neighborhood agreement among expert feature extractors. We extensively perform experiments on six newly curated benchmark datasets of diverse domains, including age prediction, price prediction, and music production year estimation. We also introduce a metric called Error Residual Ratio (ERR) to better account for varying degrees of label noise. Our approach consistently outperforms fourteen state-of-the-art baselines, being robust against symmetric and random Gaussian label noise. 

**Abstract (ZH)**: 如同许多其他问题一样，现实世界中的回归问题受到噪声标签的影响，这是一个不可避免的问题，需要我们的关注。幸运的是，很多现实世界的数据常常表现出标签和特征之间固有的连续有序相关性，其中相似标签的数据点也具有密切相关的特点。为了应对这一挑战，我们提出了一种名为ConFrag的新型方法，通过将回归数据转换为不连续但对比鲜明的断裂对，来集体建模这些数据。这使得能够训练出更为独特的表示，增强选择清洁样本的能力。我们的ConFrag框架利用相邻片段的混合，并通过专家特征提取器之间的邻域一致性来区分噪声标签。我们对六个新curated的基准数据集进行了广泛的实验，这些数据集涵盖了不同的领域，包括年龄预测、价格预测和音乐生产年份估计。我们还引入了一个名为错误残差比（ERR）的度量标准，以更好地考虑到标签噪声的不同程度。我们的方法在与十四种最先进的基线算法的对比中表现更优，并且能够抵御对称性和随机高斯噪声标签的影响。 

---
# DeepSeek vs. ChatGPT: A Comparative Study for Scientific Computing and Scientific Machine Learning Tasks 

**Title (ZH)**: DeepSeek 与 ChatGPT 在科学计算和科学机器学习任务中的比较研究 

**Authors**: Qile Jiang, Zhiwei Gao, George Em Karniadakis  

**Link**: [PDF](https://arxiv.org/pdf/2502.17764)  

**Abstract**: Large Language Models (LLMs) have emerged as powerful tools for tackling a wide range of problems, including those in scientific computing, particularly in solving partial differential equations (PDEs). However, different models exhibit distinct strengths and preferences, resulting in varying levels of performance. In this paper, we compare the capabilities of the most advanced LLMs--ChatGPT and DeepSeek--along with their reasoning-optimized versions in addressing computational challenges. Specifically, we evaluate their proficiency in solving traditional numerical problems in scientific computing as well as leveraging scientific machine learning techniques for PDE-based problems. We designed all our experiments so that a non-trivial decision is required, e.g. defining the proper space of input functions for neural operator learning. Our findings reveal that the latest model, ChatGPT o3-mini-high, usually delivers the most accurate results while also responding significantly faster than its reasoning counterpart, DeepSeek R1. This enhanced speed and accuracy make ChatGPT o3-mini-high a more practical and efficient choice for diverse computational tasks at this juncture. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经成为解决广泛问题的强大工具，特别是在科学计算领域，特别是在求解偏微分方程（PDEs）方面。然而，不同的模型表现出不同的优势和偏好，导致其性能各不相同。在本文中，我们比较了最先进的LLMs——ChatGPT和DeepSeek及其推理优化版本在解决计算挑战方面的能力。具体而言，我们评估了它们在解决科学计算中的传统数值问题以及利用科学机器学习技术解决基于PDE的问题方面的熟练程度。我们设计了所有实验，以确保需要作出非平凡的决策，例如定义神经算子学习的输入函数空间。研究发现，最新版本的ChatGPT o3-mini-high通常提供最准确的结果，同时响应速度也显著快于其推理优化版本DeepSeek R1。这种增强的速度和准确性使ChatGPT o3-mini-high在当前情况下成为一个更实用和高效的选择，适用于各种计算任务。 

---
# Design and implementation of a distributed security threat detection system integrating federated learning and multimodal LLM 

**Title (ZH)**: 集成联邦学习和多模态大语言模型的分布式安全威胁检测系统的设计与实现 

**Authors**: Yuqing Wang, Xiao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17763)  

**Abstract**: Traditional security protection methods struggle to address sophisticated attack vectors in large-scale distributed systems, particularly when balancing detection accuracy with data privacy concerns. This paper presents a novel distributed security threat detection system that integrates federated learning with multimodal large language models (LLMs). Our system leverages federated learning to ensure data privacy while employing multimodal LLMs to process heterogeneous data sources including network traffic, system logs, images, and sensor data. Experimental evaluation on a 10TB distributed dataset demonstrates that our approach achieves 96.4% detection accuracy, outperforming traditional baseline models by 4.1 percentage points. The system reduces both false positive and false negative rates by 1.8 and 2.4 percentage points respectively. Performance analysis shows that our system maintains efficient processing capabilities in distributed environments, requiring 180 seconds for model training and 3.8 seconds for threat detection across the distributed network. These results demonstrate significant improvements in detection accuracy and computational efficiency while preserving data privacy, suggesting strong potential for real-world deployment in large-scale security systems. 

**Abstract (ZH)**: 传统的安全保护方法在处理大规模分布式系统中的复杂攻击向量时显得力不从心，尤其是在权衡检测准确性和数据隐私问题之间时更为显著。本文提出了一种新颖的分布式安全威胁检测系统，该系统将联邦学习与多模态大型语言模型（LLMs）结合在一起。我们的系统利用联邦学习来确保数据隐私，并使用多模态LLMs来处理包括网络流量、系统日志、图像和传感器数据在内的异构数据源。在10TB分布式数据集上的实验评估表明，我们的方法实现了96.4%的检测准确率，较传统基线模型提高了4.1个百分点。系统将假阳性率和假阴性率分别减少了1.8个百分点和2.4个百分点。性能分析表明，我们的系统在分布式环境中保持了高效的处理能力，模型训练时间仅为180秒，而分布式网络中的威胁检测时间仅为3.8秒。这些结果表明，在提高检测准确率和计算效率的同时，仍能保留数据隐私，这表明该系统具有在大规模安全系统中实际部署的强大潜力。 

---
# Graded Neural Networks 

**Title (ZH)**: 分级神经网络 

**Authors**: Tony Shaska  

**Link**: [PDF](https://arxiv.org/pdf/2502.17751)  

**Abstract**: This paper presents a novel framework for graded neural networks (GNNs) built over graded vector spaces $\V_\w^n$, extending classical neural architectures by incorporating algebraic grading. Leveraging a coordinate-wise grading structure with scalar action $\lambda \star \x = (\lambda^{q_i} x_i)$, defined by a tuple $\w = (q_0, \ldots, q_{n-1})$, we introduce graded neurons, layers, activation functions, and loss functions that adapt to feature significance. Theoretical properties of graded spaces are established, followed by a comprehensive GNN design, addressing computational challenges like numerical stability and gradient scaling. Potential applications span machine learning and photonic systems, exemplified by high-speed laser-based implementations. This work offers a foundational step toward graded computation, unifying mathematical rigor with practical potential, with avenues for future empirical and hardware exploration. 

**Abstract (ZH)**: 本文提出了一种新的框架，用于构建基于分级向量空间 \(\V_\w^n\) 的分级神经网络（GNNs），并在经典神经架构中引入了代数分级的概念。通过使用坐标级别的分级结构 \(\lambda \star \x = (\lambda^{q_i} x_i)\)，其中 \(\w = (q_0, \ldots, q_{n-1})\) 为一个元组，我们引入了适应特征显著性的分级神经元、层级、激活函数和损失函数。这些分级空间的理论性质得到建立，随后详细设计了分级神经网络，解决了如数值稳定性和梯度缩放等计算挑战。潜在的应用范围涵盖机器学习和光子系统，并通过高速激光基实现进行了举例说明。本文为分级计算提供了一个基础步骤，统一了数学严谨性与实际应用潜力，并提出了未来实证研究和硬件探索的方向。 

---
# LLM Inference Acceleration via Efficient Operation Fusion 

**Title (ZH)**: 通过高效操作融合加速大语言模型推理 

**Authors**: Mahsa Salmani, Ilya Soloveychik  

**Link**: [PDF](https://arxiv.org/pdf/2502.17728)  

**Abstract**: The rapid development of the Transformer-based Large Language Models (LLMs) in recent years has been closely linked to their ever-growing and already enormous sizes. Many LLMs contain hundreds of billions of parameters and require dedicated hardware resources for training and inference. One of the key challenges inherent to the Transformer architecture is the requirement to support numerous non-linear transformations that involves normalization. For instance, each decoder block typically contains at least one Softmax operation and two Layernorms. The computation of the corresponding normalization scaling factors becomes a major bottleneck as it requires spatial collective operations. In other words, when it comes to the computation of denominators for Softmax and Layernorm, all vector elements must be aggregated into a single location, requiring significant communication. These collective operations slow down inference on Transformers by approximately 20%, defeating the whole purpose of distributed in-memory compute. In this work, we propose an extremely efficient technique that can completely hide the overhead caused by such collective operations. Note that each Softmax and Layernorm operation is typically followed by a linear layer. Since non-linear and linear operations are performed on different hardware engines, they can be easily parallelized once the algebra allows such commutation. By leveraging the inherent properties of linear operations, we can defer the normalization of the preceding Softmax and Layernorm until after the linear layer is computed. Now we can compute the collective scaling factors concurrently with the matrix multiplication and completely hide the latency of the former behind the latter. Such parallelization preserves the numerical accuracy while significantly improving the hardware utilization and reducing the overall latency. 

**Abstract (ZH)**: 近年来，基于变换器的大型语言模型（LLMs）的快速发展与其日益庞大且已相当巨大的规模密切相关。许多LLM包含数百亿个参数，并需要专用的硬件资源进行训练和推理。变换器架构固有的一个关键挑战是支持大量的非线性变换，特别是需要执行归一化操作。例如，每个解码器块通常至少包含一个Softmax操作和两个LayerNorm。计算相应的归一化比例因子成为了一个主要瓶颈，因为这需要进行空间上的聚合操作。换句话说，当进行Softmax和LayerNorm的分母计算时，所有向量元素必须汇总到一个位置，这需要显着的通信开销。这些聚合操作会将变换器的推理速度减慢约20%，违背了分布式内存计算的初衷。在本工作中，我们提出了一种极其高效的方法，可以完全隐藏由这些聚合操作引起的开销。需要注意的是，每个Softmax和LayerNorm操作通常会接在一层线性操作之后。由于非线性和线性操作可以在不同的硬件引擎上执行，一旦代数允许交换顺序，它们可以很容易地并行化。通过利用线性操作的固有特性，我们可以在计算线性层之前延迟执行前面的Softmax和LayerNorm的归一化操作。现在，我们可以在矩阵乘法的同时并行计算集体归一化因子，并将前者的延迟完全隐藏在后者之后。这种并行化不仅保持了数值精度，还显著提高了硬件利用率，减少了整体延时。 

---
# The GigaMIDI Dataset with Features for Expressive Music Performance Detection 

**Title (ZH)**: 《具有表现性音乐表演检测特征的GigaMIDI 数据集》 

**Authors**: Keon Ju Maverick Lee, Jeff Ens, Sara Adkins, Pedro Sarmento, Mathieu Barthet, Philippe Pasquier  

**Link**: [PDF](https://arxiv.org/pdf/2502.17726)  

**Abstract**: The Musical Instrument Digital Interface (MIDI), introduced in 1983, revolutionized music production by allowing computers and instruments to communicate efficiently. MIDI files encode musical instructions compactly, facilitating convenient music sharing. They benefit Music Information Retrieval (MIR), aiding in research on music understanding, computational musicology, and generative music. The GigaMIDI dataset contains over 1.4 million unique MIDI files, encompassing 1.8 billion MIDI note events and over 5.3 million MIDI tracks. GigaMIDI is currently the largest collection of symbolic music in MIDI format available for research purposes under fair dealing. Distinguishing between non-expressive and expressive MIDI tracks is challenging, as MIDI files do not inherently make this distinction. To address this issue, we introduce a set of innovative heuristics for detecting expressive music performance. These include the Distinctive Note Velocity Ratio (DNVR) heuristic, which analyzes MIDI note velocity; the Distinctive Note Onset Deviation Ratio (DNODR) heuristic, which examines deviations in note onset times; and the Note Onset Median Metric Level (NOMML) heuristic, which evaluates onset positions relative to metric levels. Our evaluation demonstrates these heuristics effectively differentiate between non-expressive and expressive MIDI tracks. Furthermore, after evaluation, we create the most substantial expressive MIDI dataset, employing our heuristic, NOMML. This curated iteration of GigaMIDI encompasses expressively-performed instrument tracks detected by NOMML, containing all General MIDI instruments, constituting 31% of the GigaMIDI dataset, totalling 1,655,649 tracks. 

**Abstract (ZH)**: 1983年推出的Musical Instrument Digital Interface (MIDI) 革命性地改变了音乐生产领域，使其能够高效地将计算机与乐器进行通信。MIDI文件以紧凑的形式编码音乐指令，方便音乐的分享。它们对于音乐信息检索（MIR）研究具有重要意义，有助于音乐理解、计算音乐学和生成音乐的研究。GigaMIDI数据集包含超过140万份独特的MIDI文件，涵盖了18亿个MIDI音符事件和超过530万条MIDI轨道。目前，GigaMIDI是用于研究目的下最大的MIDI格式符号音乐集合，符合公平使用原则。区分非表现性和表现性MIDI轨道具有挑战性，因为MIDI文件本身并未作出这种区分。为解决这一问题，我们提出了一套创新的启发式方法来检测表现性音乐表演。这些方法包括独特的音符速度比（DNVR），分析音符速度；独特的音符起始偏差比（DNODR），检查音符起始时间的偏差；和音符起始中位度量级别（NOMML），评估音符起始位置相对于时间度量级别的程度。我们的评估表明，这些启发式方法能够有效地区分非表现性和表现性MIDI轨道。此外，在评估后，我们使用NOMML启发式方法创建了迄今为止最大的表现性MIDI数据集。经过筛选后的GigaMIDI版本仅包含由NOMML检测到的表现性乐器轨道，且涵盖了所有通用MIDI乐器，占GigaMIDI数据集的31%，共计1,655,649条轨道。 

---
# Solving the Traveling Salesman Problem via Different Quantum Computing Architectures 

**Title (ZH)**: 通过不同的量子计算架构求解旅行商问题 

**Authors**: Venkat Padmasola, Zhaotong Li, Rupak Chatterjee, Wesley Dyk  

**Link**: [PDF](https://arxiv.org/pdf/2502.17725)  

**Abstract**: We study the application of emerging photonic and quantum computing architectures to solving the Traveling Salesman Problem (TSP), a well-known NP-hard optimization problem. We investigate several approaches: Simulated Annealing (SA), Quadratic Unconstrained Binary Optimization (QUBO-Ising) methods implemented on quantum annealers and Optical Coherent Ising Machines, as well as the Quantum Approximate Optimization Algorithm (QAOA) and the Quantum Phase Estimation (QPE) algorithm on gate-based quantum computers.
QAOA and QPE were tested on the IBM Quantum platform. The QUBO-Ising method was explored using the D-Wave quantum annealer, which operates on superconducting Josephson junctions, and the QCI Dirac machine, a nonlinear optoelectronic Ising machine. Gate-based quantum computers demonstrated accurate results for small TSP instances in simulation. However, real quantum devices are hindered by noise and limited scalability. Circuit complexity grows with problem size, restricting performance to TSP instances with a maximum of 6 nodes.
In contrast, Ising-based architectures show improved scalability for larger problem sizes. SQUID-based Ising machines can handle TSP instances with up to 12 nodes, while nonlinear optoelectronic Ising machines extend this capability to 18 nodes. Nevertheless, the solutions tend to be suboptimal due to hardware limitations and challenges in achieving ground state convergence as the problem size increases. Despite these limitations, Ising machines demonstrate significant time advantages over classical methods, making them a promising candidate for solving larger-scale TSPs efficiently. 

**Abstract (ZH)**: 我们研究了新兴光子技术和量子计算架构在解决旅行商问题（Traveling Salesman Problem, TSP）中的应用。TSP 是一个著名的 NP 难优化问题。我们探讨了几种方法：模拟退火算法（Simulated Annealing, SA）、适用于量子退火器和光相干玻色采样机的二次无约束二进制优化（Quadratic Unconstrained Binary Optimization, QUBO-Ising）方法，以及基于门的量子计算机上的量子近似优化算法（Quantum Approximate Optimization Algorithm, QAOA）和量子相位估计算法（Quantum Phase Estimation, QPE）。

QAOA 和 QPE 在 IBM Quantum 平台上进行了测试。QUBO-Ising 方法使用了 D-Wave 量子退火器，该退火器基于超导约瑟夫森结工作，并探索了 QCI Dirac 机器，这是一种非线性光电器件玻色采样机。基于门的量子计算机在对 TSP 的小规模实例进行模拟时能够获得准确的结果，然而实际量子设备受到噪声和可扩展性的限制。随着问题规模的增加，电路复杂性迅速增长，限制了性能，最多只能处理节点数为 6 的 TSP 实例。

相比之下，基于玻色-艾丁格尔（Ising）模型的架构在处理更大规模的问题时显示出了更好的可扩展性。超导量子干扰器（SQUID）基玻色-艾丁格尔机器能够处理节点数最多为 12 的 TSP 实例，而非线性光电器件玻色-艾丁格尔机器将这一能力扩展到 18 个节点。然而，由于硬件限制以及随着问题规模增加难以实现基态收敛，导致所求解的方案往往是次优的。尽管存在这些限制，玻色-艾丁格尔机器仍然显示出相对于经典方法的重大时间优势，使其成为解决大规模 TSP 的有前途的候选技术。 

---
# Aligning Compound AI Systems via System-level DPO 

**Title (ZH)**: 通过系统级DPO对齐复合AI系统 

**Authors**: Xiangwen Wang, Yibo Jacky Zhang, Zhoujie Ding, Katherine Tsai, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2502.17721)  

**Abstract**: Compound AI systems, comprising multiple interacting components such as LLM agents and external tools, demonstrate state-of-the-art results across diverse tasks. It is hence crucial to align components within the system to produce consistent results that match human expectations. However, conventional alignment methods, such as Direct Preference Optimization (DPO), are not directly applicable to compound AI systems. These challenges include the non-differentiable interactions between components, making end-to-end gradient optimization infeasible. Additionally, system-level preferences cannot be directly translated into component-level preferences, further complicating alignment. We address the issues by formulating compound AI systems as Directed Acyclic Graphs (DAGs), capturing the connections between agents and the data generation processes. We propose a system-level DPO (SysDPO) to jointly align compound systems by adapting the DPO to operate on these DAGs. We study the joint alignment of an LLM and a diffusion model to demonstrate the effectiveness of our approach. Our exploration provides insights into the alignment of compound AI systems and lays a foundation for future advancements. 

**Abstract (ZH)**: 复合人工智能系统由多个相互作用的组件组成，例如LLM代理和外部工具，在多种任务中展现出最先进的结果。因此，确保系统中的组件产生一致的结果以匹配人类期望变得至关重要。然而，传统的对齐方法，如直接偏好优化（DPO），却不适用于复合人工智能系统。这些挑战包括组件之间的非可微交互，使得端到端的梯度优化不可行。此外，系统级别的偏好无法直接转化为组件级别的偏好，进一步增加了对齐的复杂性。我们通过将复合人工智能系统形式化为有向无环图（DAGs），捕捉代理之间的连接及其数据生成过程，解决了这些问题。我们提出了一种系统级DPO（SysDPO）方法，通过将DPO适应于这些DAGs来联合对齐复合系统。我们研究了LLM和扩散模型的联合对齐，以展示我们方法的有效性。我们的探索为复合人工智能系统的对齐提供了见解，并为未来的进展奠定了基础。 

---
# Spontaneous Giving and Calculated Greed in Language Models 

**Title (ZH)**: 语言模型中的自发慷慨与计算贪婪 

**Authors**: Yuxuan Li, Hirokazu Shirado  

**Link**: [PDF](https://arxiv.org/pdf/2502.17720)  

**Abstract**: Large language models, when trained with reinforcement learning, demonstrate advanced problem-solving capabilities through reasoning techniques like chain of thoughts and reflection. However, it is unclear how these reasoning capabilities extend to social intelligence. In this study, we investigate how reasoning influences model outcomes in social dilemmas. First, we examine the effects of chain-of-thought and reflection techniques in a public goods game. We then extend our analysis to six economic games on cooperation and punishment, comparing off-the-shelf non-reasoning and reasoning models. We find that reasoning models reduce cooperation and norm enforcement, prioritizing individual rationality. Consequently, groups with more reasoning models exhibit less cooperation and lower gains through repeated interactions. These behaviors parallel human tendencies of "spontaneous giving and calculated greed." Our results suggest the need for AI architectures that incorporate social intelligence alongside reasoning capabilities to ensure that AI supports, rather than disrupts, human cooperative intuition. 

**Abstract (ZH)**: 当大型语言模型通过强化学习进行训练时，它们通过推理技巧（如链式思考和反思）展示了高级问题解决能力。然而，这些推理能力如何扩展到社会智能方面仍不清楚。在本研究中，我们探讨了推理如何影响模型在社会困境中的结果。首先，我们研究了链式思考和反思技术在公共产品博弈中的效果。然后，我们将分析扩展到六个涉及合作与惩罚的经济游戏，比较现成的非推理模型和推理模型。我们发现，推理模型降低了合作和规范执行，优先考虑个人理性。因此，具有更多推理模型的群体在重复互动中表现出更低的合作水平和更低的收益。这些行为类似于人类“自发给予与策略性贪婪”的倾向。我们的结果表明，需要一种融合社会智能与推理能力的AI架构，以确保AI支持而非干扰人类的合作直觉。 

---
# Bridging Information Gaps with Comprehensive Answers: Improving the Diversity and Informativeness of Follow-Up Questions 

**Title (ZH)**: 填补信息缺口，提供全面答案：提高后续问题的多样性和信息量 

**Authors**: Zhe Liu, Taekyu Kang, Haoyu Wang, Seyed Hossein Alavi, Vered Shwartz  

**Link**: [PDF](https://arxiv.org/pdf/2502.17715)  

**Abstract**: Effective conversational systems are expected to dynamically generate contextual follow-up questions to elicit new information while maintaining the conversation flow. While humans excel at asking diverse and informative questions by intuitively assessing both obtained and missing information, existing models often fall short of human performance on this task. To mitigate this, we propose a method that generates diverse and informative questions based on targeting unanswered information using a hypothetical LLM-generated "comprehensive answer". Our method is applied to augment an existing follow-up questions dataset. The experimental results demonstrate that language models fine-tuned on the augmented datasets produce follow-up questions of significantly higher quality and diversity. This promising approach could be effectively adopted to future work to augment information-seeking dialogues for reducing ambiguities and improving the accuracy of LLM answers. 

**Abstract (ZH)**: 有效的对话系统应能够根据上下文动态生成提问，以获取新信息并维持对话流程。尽管人类在根据获取和缺失的信息提出多样化和富有信息性的问题方面表现出色，但现有的模型在这一任务上往往无法达到人类的水平。为解决这一问题，我们提出了一种方法，该方法基于假设的LLM生成的“全面回答”来瞄准未回答的信息，从而生成多样化和富有信息性的提问。我们将该方法应用于增强现有的后续问题数据集。实验结果表明，基于增强数据集微调的语言模型生成的后续问题在质量和多样性方面显著提高。这一有前景的方法可以在未来的工作中被有效采纳，以增强信息寻求对话，减少歧义并提高LLM答案的准确性。 

---
# On the usability of generative AI: Human generative AI 

**Title (ZH)**: 生成式AI的可用性研究：人类生成式AI 

**Authors**: Anna Ravera, Cristina Gena  

**Link**: [PDF](https://arxiv.org/pdf/2502.17714)  

**Abstract**: Generative AI systems are transforming content creation, but their usability remains a key challenge. This paper examines usability factors such as user experience, transparency, control, and cognitive load. Common challenges include unpredictability and difficulties in fine-tuning outputs. We review evaluation metrics like efficiency, learnability, and satisfaction, highlighting best practices from various domains. Improving interpretability, intuitive interfaces, and user feedback can enhance usability, making generative AI more accessible and effective. 

**Abstract (ZH)**: 生成式AI系统正在变革内容创作，但其易用性仍然是一个关键挑战。本文探讨了影响易用性的因素，如用户体验、透明度、可控性和认知负荷。常见的挑战包括输出的不可预测性和难以精细化调整。我们将评估指标，如效率、学习性和满意度进行回顾，并强调各个领域中的最佳实践。通过提高可解释性、直观的用户界面和用户反馈，可以提升易用性，使得生成式AI更加易于访问和有效。 

---
# Contrastive Visual Data Augmentation 

**Title (ZH)**: 对比视觉数据增强 

**Authors**: Yu Zhou, Bingxuan Li, Mohan Tang, Xiaomeng Jin, Te-Lin Wu, Kuan-Hao Huang, Heng Ji, Kai-Wei Chang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.17709)  

**Abstract**: Large multimodal models (LMMs) often struggle to recognize novel concepts, as they rely on pre-trained knowledge and have limited ability to capture subtle visual details. Domain-specific knowledge gaps in training also make them prone to confusing visually similar, commonly misrepresented, or low-resource concepts. To help LMMs better align nuanced visual features with language, improving their ability to recognize and reason about novel or rare concepts, we propose a Contrastive visual Data Augmentation (CoDA) strategy. CoDA extracts key contrastive textual and visual features of target concepts against the known concepts they are misrecognized as, and then uses multimodal generative models to produce targeted synthetic data. Automatic filtering of extracted features and augmented images is implemented to guarantee their quality, as verified by human annotators. We show the effectiveness and efficiency of CoDA on low-resource concept and diverse scene recognition datasets including INaturalist and SUN. We additionally collect NovelSpecies, a benchmark dataset consisting of newly discovered animal species that are guaranteed to be unseen by LMMs. LLaVA-1.6 1-shot updating results on these three datasets show CoDA significantly improves SOTA visual data augmentation strategies by 12.3% (NovelSpecies), 5.1% (SUN), and 6.0% (iNat) absolute gains in accuracy. 

**Abstract (ZH)**: 大型多模态模型（LMMs）往往难以识别新型概念，因为它们依赖于预训练知识，并且在捕捉细微视觉细节方面能力有限。训练时存在的领域特定知识缺口也使得它们容易混淆视觉相似、经常被误解或资源不足的概念。为了帮助LMMs更好地将细微的视觉特征与语言对齐，提高它们识别和推理解新型或稀有概念的能力，我们提出了一种对比视觉数据增强（CoDA）策略。CoDA通过提取目标概念与它们被误识别为已知概念的关键对比文本和视觉特征，然后利用多模态生成模型生成针对性的合成数据。通过自动筛选提取的特征和增强图像的质量，并经人类注释者验证，确保了数据的质量。我们展示了CoDA在低资源概念和多样场景识别数据集（如INaturalist和SUN）上的有效性与效率。此外，我们还收集了一个基准数据集NovelSpecies，其中包括已新发现的动物物种，这些物种肯定未被LMMs见过。对这些三个数据集进行的LLaVA-1.6 1- shot 更新结果显示，与现有最佳视觉数据增强策略相比，CoDA分别在NovelSpecies、SUN和iNat数据集上显著提高了12.3%、5.1%和6.0%的准确率绝对提升。 

---
# To Patch or Not to Patch: Motivations, Challenges, and Implications for Cybersecurity 

**Title (ZH)**: 是修补还是不修补：动机、挑战及其对网络安全的影响 

**Authors**: Jason R. C. Nurse  

**Link**: [PDF](https://arxiv.org/pdf/2502.17703)  

**Abstract**: As technology has become more embedded into our society, the security of modern-day systems is paramount. One topic which is constantly under discussion is that of patching, or more specifically, the installation of updates that remediate security vulnerabilities in software or hardware systems. This continued deliberation is motivated by complexities involved with patching; in particular, the various incentives and disincentives for organizations and their cybersecurity teams when deciding whether to patch. In this paper, we take a fresh look at the question of patching and critically explore why organizations and IT/security teams choose to patch or decide against it (either explicitly or due to inaction). We tackle this question by aggregating and synthesizing prominent research and industry literature on the incentives and disincentives for patching, specifically considering the human aspects in the context of these motives. Through this research, this study identifies key motivators such as organizational needs, the IT/security team's relationship with vendors, and legal and regulatory requirements placed on the business and its staff. There are also numerous significant reasons discovered for why the decision is taken not to patch, including limited resources (e.g., person-power), challenges with manual patch management tasks, human error, bad patches, unreliable patch management tools, and the perception that related vulnerabilities would not be exploited. These disincentives, in combination with the motivators above, highlight the difficult balance that organizations and their security teams need to maintain on a daily basis. Finally, we conclude by discussing implications of these findings and important future considerations. 

**Abstract (ZH)**: 随着技术越来越多地融入我们的社会，现代系统的安全性变得至关重要。其中常被讨论的话题之一是补丁安装，即针对软件或硬件系统中的安全漏洞进行修补的更新安装。这种持续讨论的动力在于补丁安装过程中涉及的复杂性，特别是组织及其网络安全团队在决定是否进行补丁安装时面临的各种激励因素和阻碍因素。在本文中，我们将对补丁安装问题进行新的审视，并深入探讨组织和IT/安全团队究竟是为什么选择进行补丁安装，还是选择不进行（无论是明确拒绝还是由于行动迟缓）。我们通过汇总和综合关于补丁安装激励因素和阻碍因素的主要研究和行业文献，特别是从动机中的人的因素角度出发，来探讨这个问题。通过这项研究，本文确定了如组织需求、IT/安全团队与供应商的关系、以及对公司及其员工的法律与监管要求等关键动机。还发现了许多未能进行补丁安装的重要原因，包括资源限制（如人力不足）、手动补丁管理任务的挑战、人为错误、补丁错误、不可靠的补丁管理工具，以及认为相关漏洞不会被利用等。这些阻碍因素与上述的动机相结合，凸显了组织及其安全团队在日常工作中需要维持的复杂平衡。最后，本文讨论了这些发现的意义以及未来的重要考虑事项。 

---
# Yes, Q-learning Helps Offline In-Context RL 

**Title (ZH)**: 是的，Q-learning有助于离线上下文关联强化学习 

**Authors**: Denis Tarasov, Alexander Nikulin, Ilya Zisman, Albina Klepach, Andrei Polubarov, Nikita Lyubaykin, Alexander Derevyagin, Igor Kiselev, Vladislav Kurenkov  

**Link**: [PDF](https://arxiv.org/pdf/2502.17666)  

**Abstract**: In this work, we explore the integration of Reinforcement Learning (RL) approaches within a scalable offline In-Context RL (ICRL) framework. Through experiments across more than 150 datasets derived from GridWorld and MuJoCo environments, we demonstrate that optimizing RL objectives improves performance by approximately 40% on average compared to the widely established Algorithm Distillation (AD) baseline across various dataset coverages, structures, expertise levels, and environmental complexities. Our results also reveal that offline RL-based methods outperform online approaches, which are not specifically designed for offline scenarios. These findings underscore the importance of aligning the learning objectives with RL's reward-maximization goal and demonstrate that offline RL is a promising direction for application in ICRL settings. 

**Abstract (ZH)**: 在本文中，我们探讨了将强化学习（RL）方法整合到可扩展的离线上下文RL（ICRL）框架中的方法。通过跨越超过150个数据集的实验，这些数据集源自GridWorld和MuJoCo环境，我们展示了优化RL目标可以将性能平均提高约40%，相比之下，这一性能提升超越了广泛认可的算法蒸馏（AD）基线，这些数据集涵盖了不同的覆盖范围、结构、专业知识水平以及环境复杂性。我们的结果还揭示了基于离线RL的方法优于在线方法，而这些在线方法未专门针对离线场景设计。这些发现强调了使学习目标与RL的奖励最大化目标相一致的重要性，并展示了在ICRL设置中应用离线RL是一个前景广阔的方向。 

---
# Effective Field Neural Network 

**Title (ZH)**: 有效场神经网络 

**Authors**: Xi Liu, Yujun Zhao, Chun Yu Wan, Yang Zhang, Junwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17665)  

**Abstract**: In recent years, with the rapid development of machine learning, physicists have been exploring its new applications in solving or alleviating the curse of dimensionality in many-body problems. In order to accurately reflect the underlying physics of the problem, domain knowledge must be encoded into the machine learning algorithms. In this work, inspired by field theory, we propose a new set of machine learning models called effective field neural networks (EFNNs) that can automatically and efficiently capture important many-body interactions through multiple self-refining processes. Taking the classical $3$-spin infinite-range model and the quantum double exchange model as case studies, we explicitly demonstrate that EFNNs significantly outperform fully-connected deep neural networks (DNNs) and the effective model. Furthermore, with the help of convolution operations, the EFNNs learned in a small system can be seamlessly used in a larger system without additional training and the relative errors even decrease, which further demonstrates the efficacy of EFNNs in representing core physical behaviors. 

**Abstract (ZH)**: 近年来，随着机器学习的迅速发展，物理学家们一直在探索其在解决或缓解多体问题中“维数灾”问题的新应用。为了准确反映问题的内在物理机制，必须将领域知识编码到机器学习算法中。本研究受到场论的启发，提出了一种新的机器学习模型——有效场神经网络（Effective Field Neural Networks, EFNNs），能够在多个自我完善的过程中自动且高效地捕捉到重要的多体相互作用。通过经典无穷范围内3自旋系统和量子双重交换模型作为案例研究，我们明确表明，EFNNs显著优于全连接深度神经网络（Fully-connected Deep Neural Networks, DNNs）和有效模型。此外，通过卷积操作，EFNNs在小型系统中学习到的模型可以在大型系统中无缝应用，无需额外训练，并且相对误差减少，进一步证明了EFNNs在表示核心物理行为方面的有效性。 

---
# StatLLM: A Dataset for Evaluating the Performance of Large Language Models in Statistical Analysis 

**Title (ZH)**: StatLLM：用于评估大型语言模型在统计分析性能的数据集 

**Authors**: Xinyi Song, Lina Lee, Kexin Xie, Xueying Liu, Xinwei Deng, Yili Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.17657)  

**Abstract**: The coding capabilities of large language models (LLMs) have opened up new opportunities for automatic statistical analysis in machine learning and data science. However, before their widespread adoption, it is crucial to assess the accuracy of code generated by LLMs. A major challenge in this evaluation lies in the absence of a benchmark dataset for statistical code (e.g., SAS and R). To fill in this gap, this paper introduces StatLLM, an open-source dataset for evaluating the performance of LLMs in statistical analysis. The StatLLM dataset comprises three key components: statistical analysis tasks, LLM-generated SAS code, and human evaluation scores. The first component includes statistical analysis tasks spanning a variety of analyses and datasets, providing problem descriptions, dataset details, and human-verified SAS code. The second component features SAS code generated by ChatGPT 3.5, ChatGPT 4.0, and Llama 3.1 for those tasks. The third component contains evaluation scores from human experts in assessing the correctness, effectiveness, readability, executability, and output accuracy of the LLM-generated code. We also illustrate the unique potential of the established benchmark dataset for (1) evaluating and enhancing natural language processing metrics, (2) assessing and improving LLM performance in statistical coding, and (3) developing and testing of next-generation statistical software - advancements that are crucial for data science and machine learning research. 

**Abstract (ZH)**: 大型语言模型（LLMs）的编码能力为机器学习和数据科学中的自动统计分析开辟了新的机会。然而，在它们广泛采用之前，评估LLM生成代码的准确性至关重要。该评估面临的主要挑战是没有现成的统计代码基准数据集（如SAS和R）。为填补这一空白，本文介绍了StatLLM，一个开源数据集，用于评估LLM在统计分析中的性能。StatLLM数据集包含三个关键组成部分：统计分析任务、LLM生成的SAS代码以及人工评估得分。第一部分包括各种分析和数据集的统计分析任务，提供问题描述、数据集详细信息以及人工验证的SAS代码。第二部分展示了ChatGPT 3.5、ChatGPT 4.0和Llama 3.1为这些任务生成的SAS代码。第三部分包含来自专家的评估分数，用于评估LLM生成代码的正确性、有效性、可读性、可执行性以及输出准确性。我们还展示了该基准数据集的潜在独特用途，包括：(1) 评估和提升自然语言处理指标，(2) 评估和改进LLM在统计编码中的性能，以及(3) 开发和测试下一代统计软件。这些进步对于数据科学和机器学习研究至关重要。 

---
# METAL: A Multi-Agent Framework for Chart Generation with Test-Time Scaling 

**Title (ZH)**: METAL：一种用于图表生成的多智能体框架，具备测试时扩展能力 

**Authors**: Bingxuan Li, Yiwei Wang, Jiuxiang Gu, Kai-Wei Chang, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.17651)  

**Abstract**: Chart generation aims to generate code to produce charts satisfying the desired visual properties, e.g., texts, layout, color, and type. It has great potential to empower the automatic professional report generation in financial analysis, research presentation, education, and healthcare. In this work, we build a vision-language model (VLM) based multi-agent framework for effective automatic chart generation. Generating high-quality charts requires both strong visual design skills and precise coding capabilities that embed the desired visual properties into code. Such a complex multi-modal reasoning process is difficult for direct prompting of VLMs. To resolve these challenges, we propose METAL, a multi-agent framework that decomposes the task of chart generation into the iterative collaboration among specialized agents. METAL achieves 5.2% improvement in accuracy over the current best result in the chart generation task. The METAL framework exhibits the phenomenon of test-time scaling: its performance increases monotonically as the logarithmic computational budget grows from 512 to 8192 tokens. In addition, we find that separating different modalities during the critique process of METAL boosts the self-correction capability of VLMs in the multimodal context. 

**Abstract (ZH)**: 图表生成旨在生成能够满足预期视觉属性（如文本、布局、颜色和类型）的代码，具有在金融分析、研究展示、教育和医疗保健等领域自动专业报告生成的巨大潜力。在这项工作中，我们构建了一个基于视觉-语言模型（VLM）的多智能体框架，以有效地实现自动图表生成。生成高质量的图表需要具备强大的视觉设计技能和精确的编程能力，将预期的视觉属性嵌入到代码中。这种复杂的多模态推理过程难以直接通过指导VLMs来实现。为了解决这些问题，我们提出了一种多智能体框架METAL，该框架将图表生成任务分解为专门智能体之间的迭代合作。METAL在图表生成任务中的准确度提高了5.2%，并且在测试时表现出计算预算随日志计算预算从512增长到8192时性能单调提升的现象。此外，我们发现，在METAL的批判过程中分离不同模态可以增强VLMs在多模态环境下的自我校正能力。 

---
# Wearable Meets LLM for Stress Management: A Duoethnographic Study Integrating Wearable-Triggered Stressors and LLM Chatbots for Personalized Interventions 

**Title (ZH)**: 可穿戴设备结合大语言模型进行压力管理：一种融合可穿戴设备触发的压力源和大语言模型聊天机器人的个性化干预双重民族志研究 

**Authors**: Sameer Neupane, Poorvesh Dongre, Denis Gracanin, Santosh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.17650)  

**Abstract**: We use a duoethnographic approach to study how wearable-integrated LLM chatbots can assist with personalized stress management, addressing the growing need for immediacy and tailored interventions. Two researchers interacted with custom chatbots over 22 days, responding to wearable-detected physiological prompts, recording stressor phrases, and using them to seek tailored interventions from their LLM-powered chatbots. They recorded their experiences in autoethnographic diaries and analyzed them during weekly discussions, focusing on the relevance, clarity, and impact of chatbot-generated interventions. Results showed that even though most events triggered by the wearable were meaningful, only one in five warranted an intervention. It also showed that interventions tailored with brief event descriptions were more effective than generic ones. By examining the intersection of wearables and LLM, this research contributes to developing more effective, user-centric mental health tools for real-time stress relief and behavior change. 

**Abstract (ZH)**: 我们采用双重民族志方法探讨可穿戴设备集成的大型语言模型（LLM）聊天机器人如何辅助个性化压力管理，以应对即时性和个性化干预日益增长的需求。两位研究者在22天内与定制聊天机器人互动，针对可穿戴设备检测的生理信号做出回应，记录压力源短语，并利用这些信息向其LLM驱动的聊天机器人寻求个性化干预。他们将自己的体验记录在自民族志日记中，并在每周讨论中进行分析，重点关注聊天机器人生成的干预措施的相关性、清晰度和影响。研究结果表明，虽然大多数由可穿戴设备触发的事件都有意义，但只有五分之一的事件需要干预。此外，研究表明，使用简要事件描述进行个性化干预比使用通用干预更为有效。通过研究可穿戴技术和LLM的交集，本研究为开发更有效的用户为中心的心理健康工具以实现即时压力缓解和行为改变做出了贡献。 

---
# Requirements for Quality Assurance of AI Models for Early Detection of Lung Cancer 

**Title (ZH)**: AI模型早期肺癌检测的质量保证要求 

**Authors**: Horst K. Hahn, Matthias S. May, Volker Dicken, Michael Walz, Rainer Eßeling, Bianca Lassen-Schmidt, Robert Rischen, Jens Vogel-Claussen, Konstantin Nikolaou, Jörg Barkhausen  

**Link**: [PDF](https://arxiv.org/pdf/2502.17639)  

**Abstract**: Lung cancer is the second most common cancer and the leading cause of cancer-related deaths worldwide. Survival largely depends on tumor stage at diagnosis, and early detection with low-dose CT can significantly reduce mortality in high-risk patients. AI can improve the detection, measurement, and characterization of pulmonary nodules while reducing assessment time. However, the training data, functionality, and performance of available AI systems vary considerably, complicating software selection and regulatory evaluation. Manufacturers must specify intended use and provide test statistics, but they can choose their training and test data, limiting standardization and comparability. Under the EU AI Act, consistent quality assurance is required for AI-based nodule detection, measurement, and characterization.
This position paper proposes systematic quality assurance grounded in a validated reference dataset, including real screening cases plus phantom data to verify volume and growth rate measurements. Regular updates shall reflect demographic shifts and technological advances, ensuring ongoing relevance. Consequently, ongoing AI quality assurance is vital. Regulatory challenges are also adressed. While the MDR and the EU AI Act set baseline requirements, they do not adequately address self-learning algorithms or their updates. A standardized, transparent quality assessment - based on sensitivity, specificity, and volumetric accuracy - enables an objective evaluation of each AI solution's strengths and weaknesses. Establishing clear testing criteria and systematically using updated reference data lay the groundwork for comparable performance metrics, informing tenders, guidelines, and recommendations. 

**Abstract (ZH)**: 肺癌是全球第二常见的癌症，并且是癌症相关死亡的首要原因。生存率主要取决于诊断时的肿瘤分期，早期检测如低剂量CT可以显著降低高风险患者的相关死亡率。人工智能（AI）可以提高肺部结节的检测、测量和表征能力，从而减少评估时间。然而，当前可用的AI系统的训练数据、功能和表现存在显著差异，这使得软件选择和监管评估复杂化。生产商必须明确规定适用范围并提供测试统计，但他们在选择训练和测试数据方面拥有较大的自主权，这限制了标准化和可比性。在欧盟AI法案下，基于AI的结节检测、测量和表征需要一致的质量保证。

本文提出了一套基于验证参考数据集的质量保障体系，包括真实的筛查病例及模拟数据以验证体积和生长率的测量。定期更新应反映人口结构的变化和技术进步，确保持续的相关性。因此，持续的AI质量保障至关重要。本文还讨论了监管挑战。虽然MDR和欧盟AI法案设定了基础要求，但它们并未充分涵盖自我学习算法或其更新。基于敏感性、特异性和容积准确性进行的标准、透明的质量评估能够客观地评价每种AI解决方案的优势和不足。明确测试标准和系统地利用更新的参考数据为基础建立可比较的性能指标，对于指导招标、准则和建议具有重要意义。 

---
# Towards Robust Legal Reasoning: Harnessing Logical LLMs in Law 

**Title (ZH)**: 面向稳健的法律推理：利用逻辑大语言模型在法律领域中的应用 

**Authors**: Manuj Kant, Sareh Nabi, Manav Kant, Roland Scharrer, Megan Ma, Marzieh Nabi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17638)  

**Abstract**: Legal services rely heavily on text processing. While large language models (LLMs) show promise, their application in legal contexts demands higher accuracy, repeatability, and transparency. Logic programs, by encoding legal concepts as structured rules and facts, offer reliable automation, but require sophisticated text extraction. We propose a neuro-symbolic approach that integrates LLMs' natural language understanding with logic-based reasoning to address these limitations.
As a legal document case study, we applied neuro-symbolic AI to coverage-related queries in insurance contracts using both closed and open-source LLMs. While LLMs have improved in legal reasoning, they still lack the accuracy and consistency required for complex contract analysis. In our analysis, we tested three methodologies to evaluate whether a specific claim is covered under a contract: a vanilla LLM, an unguided approach that leverages LLMs to encode both the contract and the claim, and a guided approach that uses a framework for the LLM to encode the contract. We demonstrated the promising capabilities of LLM + Logic in the guided approach. 

**Abstract (ZH)**: 法律服务高度依赖文本处理。尽管大型语言模型（LLMs）展现出潜力，但在法律情境中的应用需要更高的准确度、可重复性和透明度。逻辑程序通过将法律概念编码为结构化的规则和事实，提供了可靠的自动化功能，但需要复杂的文本提取技术。我们提出了一种结合了LLMs自然语言理解能力和基于逻辑的推理的神经符号方法，以解决这些限制。

作为法律文件案例研究，我们使用闭源和开源LLMs将神经符号AI应用于保险合同中与保险覆盖相关的查询。尽管LLMs在法律推理方面有所改进，但在复杂合同分析所需的高度准确性和一致性方面仍存在差距。在我们的分析中，我们测试了三种方法来评估某一特定索赔是否被合同所覆盖：一种基础的LLM方法，一种未经指导的方法，该方法利用LLM将合同和索赔都编码进去，以及一种有引导的方法，该方法使用一个框架使LLM将合同编码起来。我们展示了在有引导的方法中，LLM与逻辑结合的有前景的能力。 

---
# Theory-guided Pseudo-spectral Full Waveform Inversion via Deep Neural Networks 

**Title (ZH)**: 基于理论指导的伪谱全波形反演方法研究（通过深度神经网络） 

**Authors**: Christopher Zerafa, Pauline Galea, Cristiana Sebu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17624)  

**Abstract**: Full-Waveform Inversion seeks to achieve a high-resolution model of the subsurface through the application of multi-variate optimization to the seismic inverse problem. Although now a mature technology, FWI has limitations related to the choice of the appropriate solver for the forward problem in challenging environments requiring complex assumptions, and very wide angle and multi-azimuth data necessary for full reconstruction are often not available.
Deep Learning techniques have emerged as excellent optimization frameworks. Data-driven methods do not impose a wave propagation model and are not exposed to modelling errors. On the contrary, deterministic models are governed by the laws of physics.
Seismic FWI has recently started to be investigated as a Deep Learning framework. Focus has been on the time-domain, while the pseudo-spectral domain has not been yet explored. However, classical FWI experienced major breakthroughs when pseudo-spectral approaches were employed. This work addresses the lacuna that exists in incorporating the pseudo-spectral approach within Deep Learning. This has been done by re-formulating the pseudo-spectral FWI problem as a Deep Learning algorithm for a theory-driven pseudo-spectral approach. A novel Recurrent Neural Network framework is proposed. This is qualitatively assessed on synthetic data, applied to a two-dimensional Marmousi dataset and evaluated against deterministic and time-based approaches.
Pseudo-spectral theory-guided FWI using RNN was shown to be more accurate than classical FWI with only 0.05 error tolerance and 1.45\% relative percent-age error. Indeed, this provides more stable convergence, able to identify faults better and has more low frequency content than classical FWI. Moreover, RNN was more suited than classical FWI at edge detection in the shallow and deep sections due to cleaner receiver residuals. 

**Abstract (ZH)**: 全波形反演旨在通过解决地震逆问题中的多变量优化来实现地下高分辨率模型。尽管该技术已经相当成熟，但在复杂环境中，FWI在选择适合前进一步问题的求解器方面仍然存在局限性。对于全重建所需的宽角度和多方向数据，在许多情况下也难以获得。

深度学习技术已悄然成为优秀的优化框架。数据驱动方法不需要制定波传播模型，也不受建模误差的影响。而确定性模型则遵循物理定律。

最近，地震FWI开始被探索作为深度学习框架。目前对时域的研究较多，而伪谱域尚未得到探索。然而，经典FWI在运用伪谱方法后取得了重大突破。本研究旨在填补深度学习中不采用伪谱方法的空白，通过将伪谱FWI问题重新表述为一种基于理论的伪谱深度学习算法，提出了一种新颖的循环神经网络框架（Recurrent Neural Network, RNN）。该方法在合成数据上进行了定性评估，并应用于二维Marmousi数据集，与确定性和基于时间的方法进行了对比。

实验结果表明，基于伪谱理论的RNN-FWI在0.05的误差容忍度下，相对误差仅为1.45%，整体表现更佳。这种方法能提供更稳健的收敛性，更准确地识别断层，且含有更多的低频成分。此外，RNN在浅层和深层部分的边界检测方面表现优于经典FWI，因为其接收器残差更干净。 

---
# Hierarchical Imitation Learning of Team Behavior from Heterogeneous Demonstrations 

**Title (ZH)**: 来自异质示例的分层模仿学习团队行为 

**Authors**: Sangwon Seo, Vaibhav Unhelkar  

**Link**: [PDF](https://arxiv.org/pdf/2502.17618)  

**Abstract**: Successful collaboration requires team members to stay aligned, especially in complex sequential tasks. Team members must dynamically coordinate which subtasks to perform and in what order. However, real-world constraints like partial observability and limited communication bandwidth often lead to suboptimal collaboration. Even among expert teams, the same task can be executed in multiple ways. To develop multi-agent systems and human-AI teams for such tasks, we are interested in data-driven learning of multimodal team behaviors. Multi-Agent Imitation Learning (MAIL) provides a promising framework for data-driven learning of team behavior from demonstrations, but existing methods struggle with heterogeneous demonstrations, as they assume that all demonstrations originate from a single team policy. Hence, in this work, we introduce DTIL: a hierarchical MAIL algorithm designed to learn multimodal team behaviors in complex sequential tasks. DTIL represents each team member with a hierarchical policy and learns these policies from heterogeneous team demonstrations in a factored manner. By employing a distribution-matching approach, DTIL mitigates compounding errors and scales effectively to long horizons and continuous state representations. Experimental results show that DTIL outperforms MAIL baselines and accurately models team behavior across a variety of collaborative scenarios. 

**Abstract (ZH)**: 成功的协作需要团队成员保持一致，特别是在复杂的顺序任务中。团队成员必须动态协调执行哪些子任务以及执行的顺序。然而，现实世界的限制，如部分可观测性和有限的通信带宽，往往会导致协作效果不佳。即使是专家团队，相同的任务也可以有不同的执行方式。为了开发适用于此类任务的多智能体系统和人机团队，我们关注基于数据的学习多模式团队行为。多智能体模仿学习（MAIL）为从演示中学习团队行为提供了一种有前途的框架，但现有方法在处理异质演示时存在困难，因为它们假设所有演示都源自单一团队策略。因此，在本工作中，我们引入了DTIL：一种用于在复杂顺序任务中学习多模式团队行为的分层MAIL算法。DTIL将每个团队成员表示为一个分层策略，并以因素分解的方式从异质团队演示中学习这些策略。通过采用分布匹配的方法，DTIL减轻了累积错误并能够有效地扩展到长时间序列和连续状态表示。实验结果表明，DTIL在基准MAIL方法上表现出色，并且能够准确地建模多种协作场景中的团队行为。 

---
# Flexible Counterfactual Explanations with Generative Models 

**Title (ZH)**: 基于生成模型的灵活反事实解释 

**Authors**: Stig Hellemans, Andres Algaba, Sam Verboven, Vincent Ginis  

**Link**: [PDF](https://arxiv.org/pdf/2502.17613)  

**Abstract**: Counterfactual explanations provide actionable insights to achieve desired outcomes by suggesting minimal changes to input features. However, existing methods rely on fixed sets of mutable features, which makes counterfactual explanations inflexible for users with heterogeneous real-world constraints. Here, we introduce Flexible Counterfactual Explanations, a framework incorporating counterfactual templates, which allows users to dynamically specify mutable features at inference time. In our implementation, we use Generative Adversarial Networks (FCEGAN), which align explanations with user-defined constraints without requiring model retraining or additional optimization. Furthermore, FCEGAN is designed for black-box scenarios, leveraging historical prediction datasets to generate explanations without direct access to model internals. Experiments across economic and healthcare datasets demonstrate that FCEGAN significantly improves counterfactual explanations' validity compared to traditional benchmark methods. By integrating user-driven flexibility and black-box compatibility, counterfactual templates support personalized explanations tailored to user constraints. 

**Abstract (ZH)**: 因果反事实解释通过建议输入特征的最小修改来提供实现期望结果的可操作见解。然而，现有的方法依赖于固定的一组可变特征，这使得因果反事实解释对于具有不同现实世界约束的用户不够灵活。在此，我们提出了柔性因果反事实解释（Flexible Counterfactual Explanations, FCE）框架，该框架结合了反事实模板，允许用户在推理时动态指定可变特征。在我们的实现中，我们使用生成对抗网络（FCEGAN），该网络能够通过用户自定义约束对解释进行对齐，而无需重新训练模型或进行额外优化。此外，FCEGAN适用于黑盒场景，利用历史预测数据集生成解释，而无需直接访问模型内部。跨经济和医疗保健数据集的实验表明，与传统基准方法相比，FCEGAN在提高因果反事实解释的有效性方面具有显著优势。通过结合用户驱动的灵活性和黑盒兼容性，反事实模板支持符合用户约束的个性化解释。 

---
# SynthRAD2025 Grand Challenge dataset: generating synthetic CTs for radiotherapy 

**Title (ZH)**: SynthRAD2025 大挑战数据集：用于放疗的合成CT图像生成 

**Authors**: Adrian Thummerer, Erik van der Bijl, Arthur Jr Galapon, Florian Kamp, Mark Savenije, Christina Muijs, Shafak Aluwini, Roel J.H.M. Steenbakkers, Stephanie Beuel, Martijn P.W. Intven, Johannes A. Langendijk, Stefan Both, Stefanie Corradini, Viktor Rogowski, Maarten Terpstra, Niklas Wahl, Christopher Kurz, Guillaume Landry, Matteo Maspero  

**Link**: [PDF](https://arxiv.org/pdf/2502.17609)  

**Abstract**: Medical imaging is essential in modern radiotherapy, supporting diagnosis, treatment planning, and monitoring. Synthetic imaging, particularly synthetic computed tomography (sCT), is gaining traction in radiotherapy. The SynthRAD2025 dataset and Grand Challenge promote advancements in sCT generation by providing a benchmarking platform for algorithms using cone-beam CT (CBCT) and magnetic resonance imaging (MRI).
The dataset includes 2362 cases: 890 MRI-CT and 1472 CBCT-CT pairs from head-and-neck, thoracic, and abdominal cancer patients treated at five European university medical centers (UMC Groningen, UMC Utrecht, Radboud UMC, LMU University Hospital Munich, and University Hospital of Cologne). Data were acquired with diverse scanners and protocols. Pre-processing, including rigid and deformable image registration, ensures high-quality, modality-aligned images. Extensive quality assurance validates image consistency and usability.
All imaging data is provided in MetaImage (.mha) format, ensuring compatibility with medical image processing tools. Metadata, including acquisition parameters and registration details, is available in structured CSV files. To maintain dataset integrity, SynthRAD2025 is divided into training (65%), validation (10%), and test (25%) sets. The dataset is accessible at this https URL under the SynthRAD2025 collection.
This dataset supports benchmarking and the development of synthetic imaging techniques for radiotherapy applications. Use cases include sCT generation for MRI-only and MR-guided photon/proton therapy, CBCT-based dose calculations, and adaptive radiotherapy workflows. By integrating diverse acquisition settings, SynthRAD2025 fosters robust, generalizable image synthesis algorithms, advancing personalized cancer care and adaptive radiotherapy. 

**Abstract (ZH)**: 医学成像是现代放射治疗中不可或缺的部分，支持诊断、治疗计划和监测。合成成像是放射治疗中的一个新兴领域，特别是合成计算机断层扫描（sCT）。SynthRAD2025数据集和Grand Challenge致力于通过提供基于锥束CT（CBCT）和磁共振成像（MRI）算法的基准平台，促进sCT的生成。

该数据集包括2362个病例：来自五个欧洲大学医学中心（格罗宁根大学医学中心、乌特勒支大学医学中心、拉脱瑙大学医学中心、慕尼黑路德维希-马克西米利安大学医院和科隆大学医院）的890例MRI-CT和1472例CBCT-CT配对病例，用于治疗头颈癌、胸部癌和腹部癌患者。数据是在各种不同的扫描器和协议下获取的。预处理包括刚性和非刚性图像配准，以确保高质量的模态对齐图像。全面的质量保证验证了图像的一致性和可用性。

所有成像数据以MetaImage (.mha)格式提供，确保与医学图像处理工具的兼容性。元数据，包括获取参数和配准细节，以结构化的CSV文件提供。为了维护数据集的完整性，SynthRAD2025划分为训练集（65%）、验证集（10%）和测试集（25%）。该数据集可通过以下链接在SynthRAD2025集合中获取：[此链接].

该数据集支持合成成像技术在放射治疗应用中的基准测试和发展。应用场景包括仅基于MRI和MR引导的光子/质子疗法的sCT生成、基于CBCT的剂量计算以及自适应放射治疗流程。通过集成各种不同的获取设置，SynthRAD2025促进了稳健的、具有普适性的图像合成算法的发展，推动了个性化癌症治疗和自适应放射治疗的进步。 

---
# Data-Driven Pseudo-spectral Full Waveform Inversion via Deep Neural Networks 

**Title (ZH)**: 基于数据驱动的深度神经网络伪谱全波形逆演算法 

**Authors**: Christopher Zerafa, Pauline Galea, Cristiana Sebu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17608)  

**Abstract**: FWI seeks to achieve a high-resolution model of the subsurface through the application of multi-variate optimization to the seismic inverse problem. Although now a mature technology, FWI has limitations related to the choice of the appropriate solver for the forward problem in challenging environments requiring complex assumptions, and very wide angle and multi-azimuth data necessary for full reconstruction are often not available.
Deep Learning techniques have emerged as excellent optimization frameworks. These exist between data and theory-guided methods. Data-driven methods do not impose a wave propagation model and are not exposed to modelling errors. On the contrary, deterministic models are governed by the laws of physics.
Application of seismic FWI has recently started to be investigated within Deep Learning. This has focussed on the time-domain approach, while the pseudo-spectral domain has not been yet explored. However, classical FWI experienced major breakthroughs when pseudo-spectral approaches were employed. This work addresses the lacuna that exists in incorporating the pseudo-spectral approach within Deep Learning. This has been done by re-formulating the pseudo-spectral FWI problem as a Deep Learning algorithm for a data-driven pseudo-spectral approach. A novel DNN framework is proposed. This is formulated theoretically, qualitatively assessed on synthetic data, applied to a two-dimensional Marmousi dataset and evaluated against deterministic and time-based approaches.
Inversion of data-driven pseudo-spectral DNN was found to outperform classical FWI for deeper and over-thrust areas. This is due to the global approximator nature of the technique and hence not bound by forward-modelling physical constraints from ray-tracing. 

**Abstract (ZH)**: 全波反演（FWI）旨在通过多变量优化地震逆问题来实现地下高分辨率模型。虽然FWI现在已经非常成熟，但在复杂环境下，选择合适的前向问题求解器仍存在局限性。此外，为了实现全重建，需要非常宽角和多方位的数据，而在某些挑战环境中，这些数据往往不可用。

深度学习技术已经展现出卓越的优化框架，这些方法介于数据驱动方法和理论指导方法之间。数据驱动的方法不强加波传播模型，不受建模误差的影响。相反，确定性模型则受物理定律的支配。

全波反演在地震学中的应用最近开始与深度学习结合，这主要集中在时域方法上，而伪谱域尚未被探索。然而，传统的FWI在采用了伪谱方法后经历了重大突破。本研究旨在弥补在深度学习中引入伪谱方法的不足，通过将伪谱法重述为数据驱动的深度学习算法，提出了一种新颖的深度神经网络（DNN）框架。这一框架从理论上进行了阐述，并在合成数据上进行了定性评估，应用于二维Marmousi数据集，并与确定性和时域方法进行了对比评估。

数据驱动的伪谱DNN逆过程被发现在更深层和断层逆冲地带优于经典FWI。这是因为该技术具有全局逼近特性，理论上不受射线追踪物理建模约束的限制。 

---
# PICASO: Permutation-Invariant Context Composition with State Space Models 

**Title (ZH)**: PICASO：基于状态空间模型的排列不变上下文组成 

**Authors**: Tian Yu Liu, Alessandro Achille, Matthew Trager, Aditya Golatkar, Luca Zancato, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2502.17605)  

**Abstract**: Providing Large Language Models with relevant contextual knowledge at inference time has been shown to greatly improve the quality of their generations. This is often achieved by prepending informative passages of text, or 'contexts', retrieved from external knowledge bases to their input. However, processing additional contexts online incurs significant computation costs that scale with their length. State Space Models (SSMs) offer a promising solution by allowing a database of contexts to be mapped onto fixed-dimensional states from which to start the generation. A key challenge arises when attempting to leverage information present across multiple contexts, since there is no straightforward way to condition generation on multiple independent states in existing SSMs. To address this, we leverage a simple mathematical relation derived from SSM dynamics to compose multiple states into one that efficiently approximates the effect of concatenating textual contexts. Since the temporal ordering of contexts can often be uninformative, we enforce permutation-invariance by efficiently averaging states obtained via our composition algorithm across all possible context orderings. We evaluate our resulting method on WikiText and MSMARCO in both zero-shot and fine-tuned settings, and show that we can match the strongest performing baseline while enjoying on average 5.4x speedup. 

**Abstract (ZH)**: 在推理过程中为大规模语言模型提供相关的上下文知识已被证明能够显著提高其生成质量。这通常通过在输入中前置检索自外部知识库的信息性文本片段或“上下文”来实现。然而，在线处理额外的上下文会带来显著的计算成本，且成本随上下文长度增加而增加。状态空间模型（SSMs）提供了一种有前景的解决方案，通过将上下文数据库映射到固定维度的状态中，从这些状态开始生成过程。然而，当尝试利用存在于多个上下文中的信息时，现有SSMs难以直接对多个独立状态进行条件化生成。为此，我们利用从SSM动力学中推导出的简单数学关系来将多个状态组合成一个能够高效近似串联文本上下文效果的状态。由于上下文的时间顺序往往对信息价值不大，我们通过有效地在所有可能的上下文顺序中对通过组合算法获得的状态进行平均，来强制执行状态的排列不变性。我们分别在零样本和微调设置下在WikiText和MSMARCO数据集上评估了此方法，结果显示我们的方法不仅能够匹配表现最佳的基线模型，还平均实现了5.4倍的加速效果。 

---
# Hallucination Detection in LLMs Using Spectral Features of Attention Maps 

**Title (ZH)**: 使用注意力图频谱特征进行大语言模型中的幻觉检测 

**Authors**: Jakub Binkowski, Denis Janiak, Albert Sawczyn, Bogdan Gabrys, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2502.17598)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable performance across various tasks but remain prone to hallucinations. Detecting hallucinations is essential for safety-critical applications, and recent methods leverage attention map properties to this end, though their effectiveness remains limited. In this work, we investigate the spectral features of attention maps by interpreting them as adjacency matrices of graph structures. We propose the $\text{LapEigvals}$ method, which utilises the top-$k$ eigenvalues of the Laplacian matrix derived from the attention maps as an input to hallucination detection probes. Empirical evaluations demonstrate that our approach achieves state-of-the-art hallucination detection performance among attention-based methods. Extensive ablation studies further highlight the robustness and generalisation of $\text{LapEigvals}$, paving the way for future advancements in the hallucination detection domain. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务上展现出了卓越的性能，但仍然容易产生幻觉。检测幻觉对于安全关键的应用至关重要，尽管最近的方法通过利用注意力图的属性来实现这一目标，但其有效性仍存在限制。在本研究中，我们通过将注意力图解释为图结构的邻接矩阵，探究了注意力图的频谱特征。我们提出了LapEigvals方法，该方法利用从注意力图中提取的拉普拉斯矩阵的前k个特征值作为幻觉检测探针的输入。实证研究表明，我们的方法在基于注意力的方法中实现了最先进的幻觉检测性能。广泛的消融研究进一步突出了LapEigvals的 robustness和泛化能力，为幻觉检测领域的未来发展奠定了基础。 

---
# Synergizing Deep Learning and Full-Waveform Inversion: Bridging Data-Driven and Theory-Guided Approaches for Enhanced Seismic Imaging 

**Title (ZH)**: 深度融合深度学习与全波形反演：结合数据驱动与理论指导方法以增强地震成像 

**Authors**: Christopher Zerafa, Pauline Galea, Cristiana Sebu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17585)  

**Abstract**: This review explores the integration of deep learning (DL) with full-waveform inversion (FWI) for enhanced seismic imaging and subsurface characterization. It covers FWI and DL fundamentals, geophysical applications (velocity estimation, deconvolution, tomography), and challenges (model complexity, data quality). The review also outlines future research directions, including hybrid, generative, and physics-informed models for improved accuracy, efficiency, and reliability in subsurface property estimation. The synergy between DL and FWI has the potential to transform geophysics, providing new insights into Earth's subsurface. 

**Abstract (ZH)**: 本文回顾了深度学习（DL）与全波形反演（FWI）的集成方法，以提高地震成像和地下表征的效果。内容涵盖了FWI和DL的基本原理、地质物理应用（如速度估计、去卷积、 tomography）以及面临的挑战（如模型复杂性、数据质量）。此外，本文还指出了未来的研究方向，包括结合深度学习与全波形反演的混合模型、生成模型和物理信息模型，以提高地下属性估计的准确性、效率和可靠性。深度学习与全波形反演之间的协同作用有可能变革地质物理学，为地球的地下结构提供新的见解。 

---
# Training a Generally Curious Agent 

**Title (ZH)**: 训练一个普遍好奇的智能体 

**Authors**: Fahim Tajwar, Yiding Jiang, Abitha Thankaraj, Sumaita Sadia Rahman, J Zico Kolter, Jeff Schneider, Ruslan Salakhutdinov  

**Link**: [PDF](https://arxiv.org/pdf/2502.17543)  

**Abstract**: Efficient exploration is essential for intelligent systems interacting with their environment, but existing language models often fall short in scenarios that require strategic information gathering. In this paper, we present PAPRIKA, a fine-tuning approach that enables language models to develop general decision-making capabilities that are not confined to particular environments. By training on synthetic interaction data from different tasks that require diverse strategies, PAPRIKA teaches models to explore and adapt their behavior on a new task based on environment feedback in-context without more gradient updates. Experimental results show that models fine-tuned with PAPRIKA can effectively transfer their learned decision-making capabilities to entirely unseen tasks without additional training. Unlike traditional training, our approach's primary bottleneck lies in sampling useful interaction data instead of model updates. To improve sample efficiency, we propose a curriculum learning strategy that prioritizes sampling trajectories from tasks with high learning potential. These results suggest a promising path towards AI systems that can autonomously solve novel sequential decision-making problems that require interactions with the external world. 

**Abstract (ZH)**: 有效的探索对于智能系统与其环境交互至关重要，但现有的语言模型在需要战略信息收集的场景中常常表现不佳。本文介绍了PAPRIKA，这是一种精细调整方法，使语言模型能够发展出不受特定环境限制的一般决策能力。通过在不同的任务上进行训练，这些任务需要不同的策略，PAPRIKA 教育模型在其首次接触新任务时根据环境反馈进行探索和调整行为，而无需增加梯度更新。实验结果表明，使用PAPRIKA进行微调的模型能够有效地将学到的决策能力转移到完全未见过的任务上，而无需额外训练。与传统的训练方法相比，我们方法的主要瓶颈在于采集有用交互数据，而非模型更新。为了提高样本效率，我们提出了一种课程学习策略，优先从具有高学习潜力的任务中采样轨迹。这些结果表明，朝着自主解决需要与外部世界互动的新序列决策问题的AI系统方向前进是值得期待的。 

---
# PosterSum: A Multimodal Benchmark for Scientific Poster Summarization 

**Title (ZH)**: PosterSum：一门用于科研海报总结的多模态基准数据集 

**Authors**: Rohit Saxena, Pasquale Minervini, Frank Keller  

**Link**: [PDF](https://arxiv.org/pdf/2502.17540)  

**Abstract**: Generating accurate and concise textual summaries from multimodal documents is challenging, especially when dealing with visually complex content like scientific posters. We introduce PosterSum, a novel benchmark to advance the development of vision-language models that can understand and summarize scientific posters into research paper abstracts. Our dataset contains 16,305 conference posters paired with their corresponding abstracts as summaries. Each poster is provided in image format and presents diverse visual understanding challenges, such as complex layouts, dense text regions, tables, and figures. We benchmark state-of-the-art Multimodal Large Language Models (MLLMs) on PosterSum and demonstrate that they struggle to accurately interpret and summarize scientific posters. We propose Segment & Summarize, a hierarchical method that outperforms current MLLMs on automated metrics, achieving a 3.14% gain in ROUGE-L. This will serve as a starting point for future research on poster summarization. 

**Abstract (ZH)**: 从多模态文档中生成准确且简洁的文本摘要是具有挑战性的，尤其是在处理像科学海报这样的视觉复杂内容时。我们提出了PosterSum，这是一种新的基准，旨在推动能够理解并用研究论文摘要的形式总结科学海报的视觉-语言模型的发展。我们的数据集包含16,305张会议海报及其对应的摘要作为总结。每张海报以图片格式提供，并包含了各种视觉理解挑战，如复杂的布局、密集的文字区域、表格和图表。我们对当前最先进的多模态大型语言模型（MLLM）进行了PosterSum基准测试，并展示了它们在准确地解释和总结科学海报方面的困难。我们提出了“分割与总结”方法，这是一种分层方法，在自动评估指标上优于当前的MLLM，实现了ROUGE-L指标3.14%的提升。这将为未来科学海报摘要的研究提供一个起点。 

---
# On the Vulnerability of Concept Erasure in Diffusion Models 

**Title (ZH)**: 关于扩散模型中概念擦除的脆弱性探究 

**Authors**: Lucas Beerens, Alex D. Richardson, Kaicheng Zhang, Dongdong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.17537)  

**Abstract**: The proliferation of text-to-image diffusion models has raised significant privacy and security concerns, particularly regarding the generation of copyrighted or harmful images. To address these issues, research on machine unlearning has developed various concept erasure methods, which aim to remove the effect of unwanted data through post-hoc training. However, we show these erasure techniques are vulnerable, where images of supposedly erased concepts can still be generated using adversarially crafted prompts. We introduce RECORD, a coordinate-descent-based algorithm that discovers prompts capable of eliciting the generation of erased content. We demonstrate that RECORD significantly beats the attack success rate of current state-of-the-art attack methods. Furthermore, our findings reveal that models subjected to concept erasure are more susceptible to adversarial attacks than previously anticipated, highlighting the urgency for more robust unlearning approaches. We open source all our code at this https URL 

**Abstract (ZH)**: 文本生成图像的扩散模型的流行引发了重要的隐私和安全担忧，尤其是在生成受版权保护或有害图像方面。为应对这些问题，有关机器遗忘的研究发展出了各种概念抹除方法，这些方法旨在通过后训练移除不需要数据的影响。然而，我们表明这些抹除技术是脆弱的，因为用对抗性构造的提示仍然可以生成被抹除概念的图像。我们提出了一种基于坐标下降的算法RECORD，该算法能够发现能够引发生成被抹除内容的提示。我们证明，RECORD 显著优于当前最先进的攻击方法的成功率。此外，我们的研究发现表明，受到概念抹除的模型比预期更容易受到对抗性攻击，突显了需要更多稳健的遗忘方法的紧迫性。我们在此 [在此处填写链接] 开放了所有代码。 

---
# The Lottery LLM Hypothesis, Rethinking What Abilities Should LLM Compression Preserve? 

**Title (ZH)**: 彩票LLM假设：重新思考压缩LLM时应保留的能力？ 

**Authors**: Zhenheng Tang, Xiang Liu, Qian Wang, Peijie Dong, Bingsheng He, Xiaowen Chu, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.17535)  

**Abstract**: Motivated by reducing the computational and storage costs of LLMs, model compression and KV cache compression have attracted much attention from researchers. However, current methods predominantly emphasize maintaining the performance of compressed LLMs, as measured by perplexity or simple accuracy on tasks of common sense knowledge QA and basic arithmetic reasoning. In this blog, we present a brief review of recent advancements in LLMs related to retrieval-augmented generation, multi-step reasoning, external tools, and computational expressivity, all of which substantially enhance LLM performance. Then, we propose a lottery LLM hypothesis suggesting that for a given LLM and task, there exists a smaller lottery LLM capable of producing the same performance as the original LLM with the assistance of multi-step reasoning and external tools. Based on the review of current progress in LLMs, we discuss and summarize the essential capabilities that the lottery LLM and KV cache compression must possess, which are currently overlooked in existing methods. 

**Abstract (ZH)**: 为了减少大语言模型（LLM）的计算和存储成本，模型压缩和键值（KV）缓存压缩已经引起了研究人员的广泛关注。然而，当前的方法大多侧重于保持压缩后的LLM的性能，这些性能通常通过困惑度或常识问答和基本算术推理任务中的简单准确性来衡量。在本文中，我们将简要回顾与检索增强生成、多步推理、外部工具以及计算表达能力相关的LLM近期进展，这些进展显著提升了LLM的性能。在此基础上，我们提出了一种“彩票LLM”假说，即对于给定的LLM和任务，存在一个较小的“彩票LLM”，在借助多步推理和外部工具的协助下，也能产生与原始LLM相同的性能。基于对当前LLM进展的回顾，我们讨论并总结了“彩票LLM”和KV缓存压缩必须具备的核心能力，这些都是现有方法目前所忽视的。 

---
# From Euler to AI: Unifying Formulas for Mathematical Constants 

**Title (ZH)**: 从欧拉到人工智能：数学常数的统一公式 

**Authors**: Tomer Raz, Michael Shalyt, Elyasheev Leibtag, Rotem Kalisch, Yaron Hadad, Ido Kaminer  

**Link**: [PDF](https://arxiv.org/pdf/2502.17533)  

**Abstract**: The constant $\pi$ has fascinated scholars for centuries, inspiring the derivation of countless formulas rooted in profound mathematical insight. This abundance of formulas raises a question: Are they interconnected, and can a unifying structure explain their relationships?
We propose a systematic methodology for discovering and proving formula equivalences, leveraging modern large language models, large-scale data processing, and novel mathematical algorithms. Analyzing 457,145 arXiv papers, over a third of the validated formulas for $\pi$ were proven to be derivable from a single mathematical object - including formulas by Euler, Gauss, Lord Brouncker, and newer ones from algorithmic discoveries by the Ramanujan Machine.
Our approach extends to other constants, such as $e$, $\zeta(3)$, and Catalan's constant, proving its broad applicability. This work represents a step toward the automatic unification of mathematical knowledge, laying a foundation for AI-driven discoveries of connections across scientific domains. 

**Abstract (ZH)**: 数学常数π长期以来一直吸引着学者们的关注，激发出了无数源自深刻数学洞察的公式。这些大量的公式引出一个问题：它们之间是否相互关联，是否有一种统一的结构能够解释它们之间的关系？

我们提出了一种系统的方法，用于发现和证明公式等价性，这种方法利用了现代大型语言模型、大规模数据处理和新型数学算法。通过对457,145篇arXiv论文进行分析，超过三分之一得到验证的π公式被证明可以从单一的数学对象导出，包括欧拉、高斯、布朗克勋爵以及拉马努詹机器通过算法发现的新公式。

该方法还应用于其他常数，如e、ζ(3)和康托尔常数，证明了其广泛的适用性。这项工作代表了向自动统一数学知识迈出的一步，为AI驱动的跨学科关联发现奠定了基础。 

---
# Laplace-Beltrami Operator for Gaussian Splatting 

**Title (ZH)**: 高斯点绘中的拉普拉斯-贝尔特拉米算子 

**Authors**: Hongyu Zhou, Zorah Lähner  

**Link**: [PDF](https://arxiv.org/pdf/2502.17531)  

**Abstract**: With the rising popularity of 3D Gaussian splatting and the expanse of applications from rendering to 3D reconstruction, there comes also a need for geometry processing applications directly on this new representation. While considering the centers of Gaussians as a point cloud or meshing them is an option that allows to apply existing algorithms, this might ignore information present in the data or be unnecessarily expensive. Additionally, Gaussian splatting tends to contain a large number of outliers which do not affect the rendering quality but need to be handled correctly in order not to produce noisy results in geometry processing applications. In this work, we propose a formulation to compute the Laplace-Beltrami operator, a widely used tool in geometry processing, directly on Gaussian splatting using the Mahalanobis distance. While conceptually similar to a point cloud Laplacian, our experiments show superior accuracy on the point clouds encoded in the Gaussian splatting centers and, additionally, the operator can be used to evaluate the quality of the output during optimization. 

**Abstract (ZH)**: 随着3D高斯散斑技术的日益流行及其在渲染到三维重建等应用领域的扩展，对这种新表示形式进行几何处理的应用也产生了新的需求。将高斯中心视为点云或将它们网格化是一种可以应用现有算法的选择，但这可能会忽略数据中存在的信息或变得不必要的昂贵。此外，高斯散斑往往包含大量不影响渲染质量但需要在几何处理应用中正确处理的离群值。在这项工作中，我们提出了一种直接在高斯散斑上计算拉普拉斯-贝尔特里算子的公式，该算子是几何处理中广泛使用的一种工具，我们利用马氏距离计算。尽管从概念上与点云拉普拉斯算子相似，但我们的实验显示，在编码于高斯散斑中心的点云上具有更高的准确性，并且该算子还可以在优化过程中评估输出的质量。 

---
# Perceptual Noise-Masking with Music through Deep Spectral Envelope Shaping 

**Title (ZH)**: 通过深度频谱包络塑造实现音乐掩盖感知噪声 

**Authors**: Clémentine Berger, Roland Badeau, Slim Essid  

**Link**: [PDF](https://arxiv.org/pdf/2502.17527)  

**Abstract**: People often listen to music in noisy environments, seeking to isolate themselves from ambient sounds. Indeed, a music signal can mask some of the noise's frequency components due to the effect of simultaneous masking. In this article, we propose a neural network based on a psychoacoustic masking model, designed to enhance the music's ability to mask ambient noise by reshaping its spectral envelope with predicted filter frequency responses. The model is trained with a perceptual loss function that balances two constraints: effectively masking the noise while preserving the original music mix and the user's chosen listening level. We evaluate our approach on simulated data replicating a user's experience of listening to music with headphones in a noisy environment. The results, based on defined objective metrics, demonstrate that our system improves the state of the art. 

**Abstract (ZH)**: 人们经常在嘈杂的环境中听音乐，希望通过音乐隔离周围的噪音。实际上，音乐信号可以通过同时掩蔽效应掩盖部分噪音的频率分量。本文提出了一种基于听觉掩蔽模型的神经网络，通过预测滤波器的频率响应重塑音乐的频谱包络，以增强音乐对环境噪音的掩蔽能力。该模型采用感知损失函数进行训练，平衡了两个约束：有效掩蔽噪音同时保持原始音乐混音和用户所选择的听音水平。我们通过模拟用户在嘈杂环境中使用耳机听音乐的体验，对方法进行了评估。根据定义的客观指标，实验结果表明，我们的系统超越了现有技术。 

---
# Multimodal Bearing Fault Classification Under Variable Conditions: A 1D CNN with Transfer Learning 

**Title (ZH)**: 在变工况下的多模态轴承故障分类：基于迁移学习的1D CNN方法 

**Authors**: Tasfiq E. Alam, Md Manjurul Ahsan, Shivakumar Raman  

**Link**: [PDF](https://arxiv.org/pdf/2502.17524)  

**Abstract**: Bearings play an integral role in ensuring the reliability and efficiency of rotating machinery - reducing friction and handling critical loads. Bearing failures that constitute up to 90% of mechanical faults highlight the imperative need for reliable condition monitoring and fault detection. This study proposes a multimodal bearing fault classification approach that relies on vibration and motor phase current signals within a one-dimensional convolutional neural network (1D CNN) framework. The method fuses features from multiple signals to enhance the accuracy of fault detection. Under the baseline condition (1,500 rpm, 0.7 Nm load torque, and 1,000 N radial force), the model reaches an accuracy of 96% with addition of L2 regularization. This represents a notable improvement of 2% compared to the non-regularized model. In addition, the model demonstrates robust performance across three distinct operating conditions by employing transfer learning (TL) strategies. Among the tested TL variants, the approach that preserves parameters up to the first max-pool layer and then adjusts subsequent layers achieves the highest performance. While this approach attains excellent accuracy across varied conditions, it requires more computational time due to its greater number of trainable parameters. To address resource constraints, less computationally intensive models offer feasible trade-offs, albeit at a slight accuracy cost. Overall, this multimodal 1D CNN framework with late fusion and TL strategies lays a foundation for more accurate, adaptable, and efficient bearing fault classification in industrial environments with variable operating conditions. 

**Abstract (ZH)**: 轴承在确保旋转机械的可靠性和效率中发挥着关键作用——通过减少摩擦和承受关键载荷。轴承故障占机械故障的90%以上，突显了可靠状态监测和故障检测的迫切需求。本研究提出了一种基于振动能和电机相电流信号的一维卷积神经网络（1D CNN）框架下的多模态轴承故障分类方法。该方法通过融合多种信号的特征以提高故障检测的准确性。在基线条件下（1500转/分钟，0.7牛·米负载扭矩，1000牛径向力），模型在加入L2正则化后达到了96%的准确率，相比未正则化的模型提升了2%。此外，模型通过采用迁移学习（TL）策略在三个不同的操作条件下展示了稳健的表现。在测试的TL变体中，保留第一最大池化层前的所有参数，然后调整后续层的方法取得了最佳性能。尽管该方法在多种条件下达到了出色的准确率，但由于其更多的可训练参数，需要更多的计算时间。为解决资源限制，计算成本较低的模型提供了可行的权衡方案，尽管会略微降低准确率。总体而言，该多模态1D CNN框架与迟延融合和迁移学习策略为在具有可变操作条件的工业环境中实现更准确、适应性和高效的轴承故障分类奠定了基础。 

---
# Spectral Theory for Edge Pruning in Asynchronous Recurrent Graph Neural Networks 

**Title (ZH)**: 异步递归图神经网络中边缘修剪的谱理论 

**Authors**: Nicolas Bessone  

**Link**: [PDF](https://arxiv.org/pdf/2502.17522)  

**Abstract**: Graph Neural Networks (GNNs) have emerged as a powerful tool for learning on graph-structured data, finding applications in numerous domains including social network analysis and molecular biology. Within this broad category, Asynchronous Recurrent Graph Neural Networks (ARGNNs) stand out for their ability to capture complex dependencies in dynamic graphs, resembling living organisms' intricate and adaptive nature. However, their complexity often leads to large and computationally expensive models. Therefore, pruning unnecessary edges becomes crucial for enhancing efficiency without significantly compromising performance. This paper presents a dynamic pruning method based on graph spectral theory, leveraging the imaginary component of the eigenvalues of the network graph's Laplacian. 

**Abstract (ZH)**: 图神经网络（GNNs）已成为学习图结构数据的强大工具，并在社会网络分析和分子生物学等多个领域找到了应用。在这一广泛的应用类别中，异步循环图神经网络（ARGNNs）因其能够捕捉动态图中的复杂依赖关系而脱颖而出，这种依赖关系类似于生物体的复杂和适应性特征。然而，它们的复杂性通常会导致大型且计算成本高昂的模型。因此，修剪不必要的边对于在不显著影响性能的情况下提高效率变得至关重要。本文提出了一种基于图频谱理论的动态修剪方法，利用网络图拉普拉斯矩阵特征值的虚部。 

---
# Recent Advances in Large Langauge Model Benchmarks against Data Contamination: From Static to Dynamic Evaluation 

**Title (ZH)**: 大数据污染背景下大型语言模型基准测试的 Recent 进展：从静态评估到动态评估 

**Authors**: Simin Chen, Yiming Chen, Zexin Li, Yifan Jiang, Zhongwei Wan, Yixin He, Dezhi Ran, Tianle Gu, Haizhou Li, Tao Xie, Baishakhi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2502.17521)  

**Abstract**: Data contamination has received increasing attention in the era of large language models (LLMs) due to their reliance on vast Internet-derived training corpora. To mitigate the risk of potential data contamination, LLM benchmarking has undergone a transformation from static to dynamic benchmarking. In this work, we conduct an in-depth analysis of existing static to dynamic benchmarking methods aimed at reducing data contamination risks. We first examine methods that enhance static benchmarks and identify their inherent limitations. We then highlight a critical gap-the lack of standardized criteria for evaluating dynamic benchmarks. Based on this observation, we propose a series of optimal design principles for dynamic benchmarking and analyze the limitations of existing dynamic benchmarks. This survey provides a concise yet comprehensive overview of recent advancements in data contamination research, offering valuable insights and a clear guide for future research efforts. We maintain a GitHub repository to continuously collect both static and dynamic benchmarking methods for LLMs. The repository can be found at this link. 

**Abstract (ZH)**: 在大规模语言模型（LLMs）时代，数据污染问题越来越受到关注，因为这些模型依赖于广泛的数据互联网训练语料库。为了减轻潜在数据污染的风险，LLM基准测试已经从静态基准测试转变为动态基准测试。在本文中，我们对现有旨在减少数据污染风险的静态到动态基准测试方法进行了深入分析。首先，我们探讨了提高静态基准的方法，并指出了这些方法的内在限制。然后，我们指出了一个关键的不足之处——缺乏标准化的标准来评估动态基准。基于这一观察，我们提出了一系列优化的设计原则，用于指导动态基准测试，并分析了现有动态基准测试的局限性。本综述为近年来数据污染研究的最新进展提供了一个简洁而全面的概述，为未来的研究提供了宝贵的见解和明确的指导。我们维护了一个GitHub仓库，以持续收集LLM的静态和动态基准测试方法。该仓库可以通过如下链接访问。 

---
# Ensemble RL through Classifier Models: Enhancing Risk-Return Trade-offs in Trading Strategies 

**Title (ZH)**: 通过分类器模型的集成强化学习：提高交易策略的风险收益权衡 

**Authors**: Zheli Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.17518)  

**Abstract**: This paper presents a comprehensive study on the use of ensemble Reinforcement Learning (RL) models in financial trading strategies, leveraging classifier models to enhance performance. By combining RL algorithms such as A2C, PPO, and SAC with traditional classifiers like Support Vector Machines (SVM), Decision Trees, and Logistic Regression, we investigate how different classifier groups can be integrated to improve risk-return trade-offs. The study evaluates the effectiveness of various ensemble methods, comparing them with individual RL models across key financial metrics, including Cumulative Returns, Sharpe Ratios (SR), Calmar Ratios, and Maximum Drawdown (MDD). Our results demonstrate that ensemble methods consistently outperform base models in terms of risk-adjusted returns, providing better management of drawdowns and overall stability. However, we identify the sensitivity of ensemble performance to the choice of variance threshold {\tau}, highlighting the importance of dynamic {\tau} adjustment to achieve optimal performance. This study emphasizes the value of combining RL with classifiers for adaptive decision-making, with implications for financial trading, robotics, and other dynamic environments. 

**Abstract (ZH)**: 本文对集成强化学习（Reinforcement Learning, RL）模型在金融交易策略中的应用进行了全面研究，利用分类器模型来提升性能。通过将A2C、PPO、SAC等RL算法与传统的SVM、决策树和逻辑回归等分类器模型相结合，我们探讨了不同分类器组如何整合以改善风险与收益的权衡。研究评估了各种集成方法的有效性，并将这些方法与单独的RL模型在关键金融指标（包括累计收益、夏普比率、卡拉麻比率和最大回撤）上进行了比较。结果显示，集成方法在风险调整后的收益方面始终优于基础模型，并能更好地管理回撤和整体稳定性。然而，我们发现集成性能对方差阈值τ的选择敏感，强调了实现最优性能时动态调整τ的重要性。本文强调了将RL与分类器结合使用以实现适应性决策的价值，并对金融交易、机器人技术以及其他动态环境具有重要意义。 

---
# Attention-based UAV Trajectory Optimization for Wireless Power Transfer-assisted IoT Systems 

**Title (ZH)**: 基于注意力机制的无人机轨迹优化以支持无线能量传输辅助物联网系统 

**Authors**: Li Dong, Feibo Jiang, Yubo Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.17517)  

**Abstract**: Unmanned Aerial Vehicles (UAVs) in Wireless Power Transfer (WPT)-assisted Internet of Things (IoT) systems face the following challenges: limited resources and suboptimal trajectory planning. Reinforcement learning-based trajectory planning schemes face issues of low search efficiency and learning instability when optimizing large-scale systems. To address these issues, we present an Attention-based UAV Trajectory Optimization (AUTO) framework based on the graph transformer, which consists of an Attention Trajectory Optimization Model (ATOM) and a Trajectory lEarNing Method based on Actor-critic (TENMA). In ATOM, a graph encoder is used to calculate the self-attention characteristics of all IoTDs, and a trajectory decoder is developed to optimize the number and trajectories of UAVs. TENMA then trains the ATOM using an improved Actor-Critic method, in which the real reward of the system is applied as the baseline to reduce variances in the critic network. This method is suitable for high-quality and large-scale multi-UAV trajectory planning. Finally, we develop numerous experiments, including a hardware experiment in the field case, to verify the feasibility and efficiency of the AUTO framework. 

**Abstract (ZH)**: 无人驾驶飞行器（UAVs）在无线能量传输（WPT）辅助的物联网（IoT）系统中面临以下挑战：有限的资源和次优的轨迹规划。基于强化学习的轨迹规划方案在优化大规模系统时存在搜索效率低和学习不稳定的問題。为了解决这些问题，我们提出了一种基于图变压器的注意力机制无人机轨迹优化（AUTO）框架，该框架包括注意力轨迹优化模型（ATOM）和基于actor-critic的轨迹学习方法（TENMA）。在ATOM中，使用图编码器计算所有IoTD（物联网设备）的自注意力特性，并开发了轨迹解码器以优化无人机的数量及其轨迹。TENMA然后采用改进的Actor-Critic方法训练ATOM，在这种方法中，系统的真实奖励作为基准以减少批评网络中的方差。该方法适用于高质量和大规模的多无人机轨迹规划。最后，我们开发了多种实验，包括实地条件下的硬件实验，以验证AUTO框架的可行性和效率。 

---
# A Survey on Mechanistic Interpretability for Multi-Modal Foundation Models 

**Title (ZH)**: 多模态基础模型的机制可解释性综述 

**Authors**: Zihao Lin, Samyadeep Basu, Mohammad Beigi, Varun Manjunatha, Ryan A. Rossi, Zichao Wang, Yufan Zhou, Sriram Balasubramanian, Arman Zarei, Keivan Rezaei, Ying Shen, Barry Menglong Yao, Zhiyang Xu, Qin Liu, Yuxiang Zhang, Yan Sun, Shilong Liu, Li Shen, Hongxuan Li, Soheil Feizi, Lifu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17516)  

**Abstract**: The rise of foundation models has transformed machine learning research, prompting efforts to uncover their inner workings and develop more efficient and reliable applications for better control. While significant progress has been made in interpreting Large Language Models (LLMs), multimodal foundation models (MMFMs) - such as contrastive vision-language models, generative vision-language models, and text-to-image models - pose unique interpretability challenges beyond unimodal frameworks. Despite initial studies, a substantial gap remains between the interpretability of LLMs and MMFMs. This survey explores two key aspects: (1) the adaptation of LLM interpretability methods to multimodal models and (2) understanding the mechanistic differences between unimodal language models and crossmodal systems. By systematically reviewing current MMFM analysis techniques, we propose a structured taxonomy of interpretability methods, compare insights across unimodal and multimodal architectures, and highlight critical research gaps. 

**Abstract (ZH)**: 基础模型的崛起已彻底改变机器学习的研究方向，促使研究人员致力于探索其内部机制，并开发更为高效和可靠的模型以实现更好的控制。虽然在解释大型语言模型（LLMs）方面已经取得了显著进展，但多模态基础模型（MMFMs），如对比型视语言模型、生成型视语言模型和文本到图像模型，因其超越单一模态框架的独特解释性挑战而受到关注。尽管已经有了一些初步研究，但在LLMs与MMFMs的解释性之间仍然存在相当大的差距。本文综述了两个关键方面：（1）将LLM解释方法应用于多模态模型，以及（2）理解单模态语言模型与跨模态系统之间的机制差异。通过系统性地评审当前多模态模型分析技术，我们提出了一个结构化的解释性方法分类体系，比较了单模态和多模态架构之间的见解，并突显出关键的研究缺口。 

---
# Towards User-level Private Reinforcement Learning with Human Feedback 

**Title (ZH)**: 面向用户级别的私人强化学习及其人类反馈方法 

**Authors**: Jiaming Zhang, Mingxi Lei, Meng Ding, Mengdi Li, Zihang Xiang, Difei Xu, Jinhui Xu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17515)  

**Abstract**: Reinforcement Learning with Human Feedback (RLHF) has emerged as an influential technique, enabling the alignment of large language models (LLMs) with human preferences. Despite the promising potential of RLHF, how to protect user preference privacy has become a crucial issue. Most previous work has focused on using differential privacy (DP) to protect the privacy of individual data. However, they have concentrated primarily on item-level privacy protection and have unsatisfactory performance for user-level privacy, which is more common in RLHF. This study proposes a novel framework, AUP-RLHF, which integrates user-level label DP into RLHF. We first show that the classical random response algorithm, which achieves an acceptable performance in item-level privacy, leads to suboptimal utility when in the user-level settings. We then establish a lower bound for the user-level label DP-RLHF and develop the AUP-RLHF algorithm, which guarantees $(\varepsilon, \delta)$ user-level privacy and achieves an improved estimation error. Experimental results show that AUP-RLHF outperforms existing baseline methods in sentiment generation and summarization tasks, achieving a better privacy-utility trade-off. 

**Abstract (ZH)**: 人类反馈强化学习（RLHF）已经成为了影响深远的技术，它能够使大型语言模型（LLMs）与人类偏好保持一致。尽管RLHF具有巨大的潜力，但如何保护用户偏好隐私已经成为了一个关键问题。大多数以往的研究都集中在使用差分隐私（DP）来保护个体数据的隐私。然而，它们主要集中在项目级别隐私保护上，而没有很好地解决更多常见于RLHF的用户级别隐私问题。本研究提出了一种新型框架AUP-RLHF，该框架将用户级别标签的差分隐私整合到RLHF中。我们首先展示了经典的随机响应算法，在项目级别隐私保护中具有可接受的性能，但在用户级别设置中会导致性能不佳。然后，我们为用户级别标签的差分隐私-RLHF建立了下界，并开发了AUP-RLHF算法，该算法确保了$(\varepsilon, \delta)$级别的用户级别隐私，并实现了改进的估计误差。实验结果表明，在情感生成和摘要任务中，AUP-RLHF优于现有基线方法，实现了更好的隐私-性能权衡。 

---
# SAE-V: Interpreting Multimodal Models for Enhanced Alignment 

**Title (ZH)**: SAE-V：解析多模态模型以增强对齐 

**Authors**: Hantao Lou, Changye Li, Jiaming Ji, Yaodong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17514)  

**Abstract**: With the integration of image modality, the semantic space of multimodal large language models (MLLMs) is more complex than text-only models, making their interpretability more challenging and their alignment less stable, particularly susceptible to low-quality data, which can lead to inconsistencies between modalities, hallucinations, and biased outputs. As a result, developing interpretability methods for MLLMs is crucial for improving alignment quality and efficiency. In text-only LLMs, Sparse Autoencoders (SAEs) have gained attention for their ability to interpret latent representations. However, extending SAEs to multimodal settings presents new challenges due to modality fusion and the difficulty of isolating cross-modal representations. To address these challenges, we introduce SAE-V, a mechanistic interpretability framework that extends the SAE paradigm to MLLMs. By identifying and analyzing interpretable features along with their corresponding data, SAE-V enables fine-grained interpretation of both model behavior and data quality, facilitating a deeper understanding of cross-modal interactions and alignment dynamics. Moreover, by utilizing cross-modal feature weighting, SAE-V provides an intrinsic data filtering mechanism to enhance model alignment without requiring additional models. Specifically, when applied to the alignment process of MLLMs, SAE-V-based data filtering methods could achieve more than 110% performance with less than 50% data. Our results highlight SAE-V's ability to enhance interpretability and alignment in MLLMs, providing insights into their internal mechanisms. 

**Abstract (ZH)**: 随着图像模态的整合，多模态大型语言模型（MLLMs）的语义空间比仅文本模型更加复杂，这使得它们的可解释性更加困难，对齐性也更加不稳定，尤其容易受到低质量数据的影响，从而导致模态之间的一致性问题、幻觉和偏差输出。因此，开发MLLMs的可解释性方法对于提高对齐质量和效率至关重要。在仅文本的大规模语言模型（LLMs）中，稀疏自编码器（SAEs）因其能够解释隐含表示而受到关注。然而，将SAEs扩展到多模态环境带来了新的挑战，因为需要处理模态融合以及跨模态表示的隔离难度。为了解决这些挑战，我们提出了一种基于SAE的机制性可解释框架SAE-V，该框架将SAE范式扩展到MLLMs中。通过识别和分析具有对应数据的可解释特征，SAE-V使我们能够对模型行为和数据质量进行精细化解释，从而促进对跨模态交互和对齐动力学的深入理解。此外，通过利用跨模态特征加权，SAE-V提供了一种内在的数据过滤机制，以增强模型对齐，而无需额外的模型。具体而言，当SAE-V应用于MLLMs的对齐过程时，基于SAE-V的数据过滤方法可以在少于50%的数据下实现超过110%的性能提升。我们的结果突出展示了SAE-V在增强MLLMs的可解释性和对齐性方面的潜力，为理解其内部机制提供了见解。 

---
# Int2Int: a framework for mathematics with transformers 

**Title (ZH)**: Int2Int：基于变换器的数学框架 

**Authors**: François Charton  

**Link**: [PDF](https://arxiv.org/pdf/2502.17513)  

**Abstract**: This paper documents Int2Int, an open source code base for using transformers on problems of mathematical research, with a focus on number theory and other problems involving integers. Int2Int is a complete PyTorch implementation of a transformer architecture, together with training and evaluation loops, and classes and functions to represent, generate and decode common mathematical objects. Ancillary code for data preparation, and Jupyter Notebooks for visualizing experimental results are also provided. This document presents the main features of Int2Int, serves as its user manual, and provides guidelines on how to extend it. Int2Int is released under the MIT licence, at this https URL. 

**Abstract (ZH)**: 本文记录了Int2Int，一个用于数学研究问题的开源代码库，特别是数论和其他涉及整数的问题。Int2Int 是一个完整的基于 PyTorch 的变压器架构实现，包括训练和评估循环，以及表示、生成和解码常见数学对象的类和函数。还提供了数据准备的辅助代码和可视化实验结果的 Jupyter Notebook。本文档介绍了Int2Int的主要特点，作为其用户手册，并提供了扩展它的指南。Int2Int 以 MIT 许可证发布，可在以下网址访问：[相关网址]。 

---
# Recurrent Knowledge Identification and Fusion for Language Model Continual Learning 

**Title (ZH)**: 循环知识识别与融合在语言模型持续学习中的应用 

**Authors**: Yujie Feng, Xujia Wang, Zexin Lu, Shenghong Fu, Guangyuan Shi, Yongxin Xu, Yasha Wang, Philip S. Yu, Xu Chu, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17510)  

**Abstract**: Continual learning (CL) is crucial for deploying large language models (LLMs) in dynamic real-world environments without costly retraining. While recent model ensemble and model merging methods guided by parameter importance have gained popularity, they often struggle to balance knowledge transfer and forgetting, mainly due to the reliance on static importance estimates during sequential training. In this paper, we present Recurrent-KIF, a novel CL framework for Recurrent Knowledge Identification and Fusion, which enables dynamic estimation of parameter importance distributions to enhance knowledge transfer. Inspired by human continual learning, Recurrent-KIF employs an inner loop that rapidly adapts to new tasks while identifying important parameters, coupled with an outer loop that globally manages the fusion of new and historical knowledge through redundant knowledge pruning and key knowledge merging. These inner-outer loops iteratively perform multiple rounds of fusion, allowing Recurrent-KIF to leverage intermediate training information and adaptively adjust fusion strategies based on evolving importance distributions. Extensive experiments on two CL benchmarks with various model sizes (from 770M to 13B) demonstrate that Recurrent-KIF effectively mitigates catastrophic forgetting and enhances knowledge transfer. 

**Abstract (ZH)**: 持续学习（CL）对于在动态现实环境中部署大型语言模型（LLMs）至关重要，而无需昂贵的重新训练。尽管最近由参数重要性指导的模型集成和模型融合方法越来越受欢迎，但在顺序训练过程中依赖静态重要性估计往往难以平衡知识转移和遗忘。本文提出了Recurrent-KIF（循环知识识别与融合框架），这是一种新颖的持续学习框架，能够动态估计参数重要性分布以增强知识转移。受人类持续学习的启发，Recurrent-KIF 结合了一个内部循环，可以快速适应新任务并识别重要参数，以及一个外部循环，通过冗余知识修剪和关键知识合并来全局管理新旧知识的融合。这些内部-外部循环迭代进行多次融合，使Recurrent-KIF能够利用中间训练信息，并根据不断变化的重要性分布自适应调整融合策略。在两个持续学习基准测试中使用不同规模的模型（从770M到13B）进行了广泛实验，结果表明Recurrent-KIF有效地减轻了灾难性遗忘，并增强了知识转移。 

---
# C-3DPO: Constrained Controlled Classification for Direct Preference Optimization 

**Title (ZH)**: C-3DPO：受约束的控制分类以实现直接的偏好优化 

**Authors**: Kavosh Asadi, Julien Han, Xingzi Xu, Dominique Perrault-Joncas, Shoham Sabach, Karim Bouyarmane, Mohammad Ghavamzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2502.17507)  

**Abstract**: Direct preference optimization (DPO)-style algorithms have emerged as a promising approach for solving the alignment problem in AI. We present a novel perspective that formulates these algorithms as implicit classification algorithms. This classification framework enables us to recover many variants of DPO-style algorithms by choosing appropriate classification labels and loss functions. We then leverage this classification framework to demonstrate that the underlying problem solved in these algorithms is under-specified, making them susceptible to probability collapse of the winner-loser responses. We address this by proposing a set of constraints designed to control the movement of probability mass between the winner and loser in the reference and target policies. Our resulting algorithm, which we call Constrained Controlled Classification DPO (\texttt{C-3DPO}), has a meaningful RLHF interpretation. By hedging against probability collapse, \texttt{C-3DPO} provides practical improvements over vanilla \texttt{DPO} when aligning several large language models using standard preference datasets. 

**Abstract (ZH)**: 直接偏好优化（DPO）风格的算法已成为解决AI对齐问题的一种有前景的方法。我们提出了一个新的视角，将这些算法形式化为隐式分类算法。这种分类框架使我们能够通过选择合适的分类标签和损失函数来恢复许多DPO风格算法的变体。随后，借助这一分类框架，我们证明了这些算法所解决的基本问题并不明确，从而使它们容易发生胜者失败者响应的概率崩溃。为此，我们提案了一组约束，旨在控制参考策略和目标策略中胜者与失败者之间概率质量的移动。由此产生的算法称为受约束的控制分类DPO（\texttt{C-3DPO}），并在标准偏好数据集上对齐多个大型语言模型时，具有实际意义的RLHF解释。通过防范概率崩溃，\texttt{C-3DPO} 在使用标准偏好数据集对齐多个大型语言模型时，相对于基础的 \texttt{DPO} 方法提供了实际改进。 

---
# RAG-Enhanced Collaborative LLM Agents for Drug Discovery 

**Title (ZH)**: 增强记忆辅助的协作大语言模型代理在药物发现中的应用 

**Authors**: Namkyeong Lee, Edward De Brouwer, Ehsan Hajiramezanali, Chanyoung Park, Gabriele Scalia  

**Link**: [PDF](https://arxiv.org/pdf/2502.17506)  

**Abstract**: Recent advances in large language models (LLMs) have shown great potential to accelerate drug discovery. However, the specialized nature of biochemical data often necessitates costly domain-specific fine-tuning, posing critical challenges. First, it hinders the application of more flexible general-purpose LLMs in cutting-edge drug discovery tasks. More importantly, it impedes the rapid integration of the vast amounts of scientific data continuously generated through experiments and research. To investigate these challenges, we propose CLADD, a retrieval-augmented generation (RAG)-empowered agentic system tailored to drug discovery tasks. Through the collaboration of multiple LLM agents, CLADD dynamically retrieves information from biomedical knowledge bases, contextualizes query molecules, and integrates relevant evidence to generate responses -- all without the need for domain-specific fine-tuning. Crucially, we tackle key obstacles in applying RAG workflows to biochemical data, including data heterogeneity, ambiguity, and multi-source integration. We demonstrate the flexibility and effectiveness of this framework across a variety of drug discovery tasks, showing that it outperforms general-purpose and domain-specific LLMs as well as traditional deep learning approaches. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的发展展现出加速药物发现的强大潜力。然而，生物化学数据的专业性往往需要进行昂贵的领域特定微调，这带来了关键性的挑战。首先，这阻碍了更灵活的通用LLM在前沿药物发现任务中的应用。更重要的是，这妨碍了迅速整合通过实验和研究不断生成的大量科学数据。为了应对这些挑战，我们提出了一种CLADD系统，这是一种由检索增强生成（RAG）赋能的特制于药物发现任务的代理系统。通过多个LLM代理的协作，CLADD动态地从生物医学知识库中检索信息，上下文化查询分子，并整合相关证据生成回应——所有这些都不需要进行领域特定的微调。至关重要的是，我们解决了将RAG工作流应用于生物化学数据的关键障碍，包括数据异质性、歧义性和多源整合。我们展示了该框架在多种药物发现任务中的灵活性和有效性，证明其性能优于通用和领域特定的LLM以及传统的深度学习方法。 

---
# Inverse Surrogate Model of a Soft X-Ray Spectrometer using Domain Adaptation 

**Title (ZH)**: 使用领域适应的软X射线光谱仪逆代理模型 

**Authors**: Enrico Ahlers, Peter Feuer-Forson, Gregor Hartmann, Rolf Mitzner, Peter Baumgärtel, Jens Viefhaus  

**Link**: [PDF](https://arxiv.org/pdf/2502.17505)  

**Abstract**: In this study, we present a method to create a robust inverse surrogate model for a soft X-ray spectrometer. During a beamtime at an electron storage ring, such as BESSY II, instrumentation and beamlines are required to be correctly aligned and calibrated for optimal experimental conditions. In order to automate these processes, machine learning methods can be developed and implemented, but in many cases these methods require the use of an inverse model which maps the output of the experiment, such as a detector image, to the parameters of the device. Due to limited experimental data, such models are often trained with simulated data, which creates the challenge of compensating for the inherent differences between simulation and experiment. In order to close this gap, we demonstrate the application of data augmentation and adversarial domain adaptation techniques, with which we can predict absolute coordinates for the automated alignment of our spectrometer. Bridging the simulation-experiment gap with minimal real-world data opens new avenues for automated experimentation using machine learning in scientific instrumentation. 

**Abstract (ZH)**: 在本研究中，我们提出了一种方法用于构建软X射线光谱仪的稳健逆代理模型。在电子储存环（如BESSY II）的束流时间内，必须正确对准和校准仪器及束线以确保最优的实验条件。为了自动化这些过程，可以开发并实施机器学习方法，但在很多情况下，这些方法需要使用逆模型，该模型能够将实验输出（如探测器图像）映射到设备参数。由于实验数据有限，这些模型通常会用模拟数据进行训练，这会带来模拟与实验之间固有差异补偿的挑战。为了弥合这一差距，我们展示了数据增强和对抗域适应技术的应用案例，通过这种方法，我们可以预测绝对坐标，以实现我们光谱仪的自动化对准。通过最少的现实世界数据弥合模拟与实验的差距，为使用机器学习在科学仪器中实现自动化实验开辟了新的途径。 

---
# Protein Large Language Models: A Comprehensive Survey 

**Title (ZH)**: 蛋白质大型语言模型：一项全面综述 

**Authors**: Yijia Xiao, Wanjia Zhao, Junkai Zhang, Yiqiao Jin, Han Zhang, Zhicheng Ren, Renliang Sun, Haixin Wang, Guancheng Wan, Pan Lu, Xiao Luo, Yu Zhang, James Zou, Yizhou Sun, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17504)  

**Abstract**: Protein-specific large language models (Protein LLMs) are revolutionizing protein science by enabling more efficient protein structure prediction, function annotation, and design. While existing surveys focus on specific aspects or applications, this work provides the first comprehensive overview of Protein LLMs, covering their architectures, training datasets, evaluation metrics, and diverse applications. Through a systematic analysis of over 100 articles, we propose a structured taxonomy of state-of-the-art Protein LLMs, analyze how they leverage large-scale protein sequence data for improved accuracy, and explore their potential in advancing protein engineering and biomedical research. Additionally, we discuss key challenges and future directions, positioning Protein LLMs as essential tools for scientific discovery in protein science. Resources are maintained at this https URL. 

**Abstract (ZH)**: 蛋白质特定的大语言模型（Protein LLMs）正在通过使蛋白质结构预测、功能注释和设计更加高效，从而革新蛋白质科学。尽管现有的综述文章主要关注特定方面或应用，本项工作提供了蛋白质大语言模型的第一个全面概述，涵盖了它们的架构、训练数据集、评价指标以及多种应用。通过对超过100篇相关文章的系统分析，我们提出了先进蛋白质大语言模型的结构化分类体系，分析了它们如何利用大规模蛋白质序列数据以提高准确性，并探讨了其在促进蛋白质工程和生物医药研究方面的潜力。此外，我们讨论了关键挑战和未来方向，将蛋白质大语言模型定位为蛋白质科学中科学研究的重要工具。资料维护地址为：[这个链接]。 

---
# Doctor-in-the-Loop: An Explainable, Multi-View Deep Learning Framework for Predicting Pathological Response in Non-Small Cell Lung Cancer 

**Title (ZH)**: 医生在环中：一种可解释的多视图深度学习框架，用于预测非小细胞肺癌的病理反应 

**Authors**: Alice Natalina Caragliano, Filippo Ruffini, Carlo Greco, Edy Ippolito, Michele Fiore, Claudia Tacconi, Lorenzo Nibid, Giuseppe Perrone, Sara Ramella, Paolo Soda, Valerio Guarrasi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17503)  

**Abstract**: Non-small cell lung cancer (NSCLC) remains a major global health challenge, with high post-surgical recurrence rates underscoring the need for accurate pathological response predictions to guide personalized treatments. Although artificial intelligence models show promise in this domain, their clinical adoption is limited by the lack of medically grounded guidance during training, often resulting in non-explainable intrinsic predictions. To address this, we propose Doctor-in-the-Loop, a novel framework that integrates expert-driven domain knowledge with explainable artificial intelligence techniques, directing the model toward clinically relevant anatomical regions and improving both interpretability and trustworthiness. Our approach employs a gradual multi-view strategy, progressively refining the model's focus from broad contextual features to finer, lesion-specific details. By incorporating domain insights at every stage, we enhance predictive accuracy while ensuring that the model's decision-making process aligns more closely with clinical reasoning. Evaluated on a dataset of NSCLC patients, Doctor-in-the-Loop delivers promising predictive performance and provides transparent, justifiable outputs, representing a significant step toward clinically explainable artificial intelligence in oncology. 

**Abstract (ZH)**: 非小细胞肺癌（NSCLC）仍然是一个重大的全球性健康挑战，手术后高复发率凸显了准确病理反应预测以指导个性化治疗的必要性。尽管人工智能模型在这一领域显示出前景，但由于在训练过程中缺乏医学依据的指导，其临床应用受到限制，通常导致非解释性的内在预测结果。为了解决这一问题，我们提出了“医生在环中”（Doctor-in-the-Loop）框架，该框架结合了专家驱动的领域知识与可解释的人工智能技术，引导模型关注临床相关的解剖区域，同时提高可解释性和可信度。我们的方法采用渐进多视角策略，逐步细化模型的聚焦范围，从宏观的上下文特征逐步到更具体的病灶细节。通过在每个阶段引入领域见解，我们提高了预测准确性，确保模型的决策过程更紧密地与临床推理相一致。在一项针对NSCLC患者的评估中，“医生在环中”框架展示了有前景的预测性能，并提供了透明且可溯源的输出结果，代表了在肿瘤学中实现临床可解释人工智能的重要一步。 

---
# CoKV: Optimizing KV Cache Allocation via Cooperative Game 

**Title (ZH)**: CoKV：通过合作博弈优化键值缓存分配 

**Authors**: Qiheng Sun, Hongwei Zhang, Haocheng Xia, Jiayao Zhang, Jinfei Liu, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.17501)  

**Abstract**: Large language models (LLMs) have achieved remarkable success on various aspects of human life. However, one of the major challenges in deploying these models is the substantial memory consumption required to store key-value pairs (KV), which imposes significant resource demands. Recent research has focused on KV cache budget allocation, with several approaches proposing head-level budget distribution by evaluating the importance of individual attention heads. These methods, however, assess the importance of heads independently, overlooking their cooperative contributions within the model, which may result in a deviation from their true impact on model performance. In light of this limitation, we propose CoKV, a novel method that models the cooperation between heads in model inference as a cooperative game. By evaluating the contribution of each head within the cooperative game, CoKV can allocate the cache budget more effectively. Extensive experiments show that CoKV achieves state-of-the-art performance on the LongBench benchmark using LLama-3-8B-Instruct and Mistral-7B models. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在人类生活的各个方面均取得了显著的成功。然而，部署这些模型的一个主要挑战是需要大量内存来存储键值对（KV），这带来了显著的资源需求。最近的研究集中在KV缓存预算分配上，提出了一些方法以评估各个注意力头的重要性来进行头部级别的预算分配。然而，这些方法独立评估每个头的重要性，而忽略了它们在模型中的协同贡献，这可能导致对它们真正影响模型性能的偏差。鉴于这一限制，我们提出了一种名为CoKV的新方法，该方法将模型推理过程中头部之间的协同作用建模为一个合作博弈。通过在合作博弈中评估每个头部的贡献，CoKV能够更有效地分配缓存预算。广泛的经验表明，CoKV在使用LLama-3-8B-Instruct和Mistral-7B模型的LongBench基准测试中取得了最优性能。 

---
# Generalized Exponentiated Gradient Algorithms Using the Euler Two-Parameter Logarithm 

**Title (ZH)**: 广义指数梯度算法使用欧拉两参数对数函数 

**Authors**: Andrzej Cichocki  

**Link**: [PDF](https://arxiv.org/pdf/2502.17500)  

**Abstract**: In this paper we propose and investigate a new class of Generalized Exponentiated Gradient (GEG) algorithms using Mirror Descent (MD) approaches, and applying as a regularization function the Bregman divergence with two-parameter deformation of logarithm as a link function. This link function (referred to as the Euler logarithm) is associated with a wide class of generalized entropies. In order to derive novel GEG/MD updates, we estimate generalized exponential function, which closely approximates the inverse of the Euler two-parameter logarithm. The characteristic/shape and properties of the Euler logarithm and its inverse -- deformed exponential functions are tuned by two or even more hyperparameters. By learning these hyperparameters, we can adapt to distribution of training data, and we can adjust them to achieve desired properties of gradient descent algorithms. The concept of generalized entropies and associated deformed logarithms provide deeper insight into novel gradient descent updates.
In literature, there exist nowadays over fifty mathematically well-defined entropic functionals and associated deformed logarithms, so impossible to investigate all of them in one research paper. Therefore, we focus here on a wide-class of trace-form entropies and associated generalized logarithm. We applied the developed algorithms for Online Portfolio Selection (OPLS) in order to improve its performance and robustness. 

**Abstract (ZH)**: 在本文中，我们提出并研究了一类新的广义指数梯度（GEG）算法，这些算法利用了镜像下降（MD）方法，并以带有两参数变形对数作为链接函数的Bregman散度作为正则化函数。这种链接函数（称为欧拉对数）与广义熵类别相关。为了推导新的GEG/MD更新，我们估算了广义指数函数，该函数紧密逼近欧拉双参数对数的反函数。欧拉对数及其反函数—变形指数函数的特性和形状可以通过两个或更多的超参数进行调节。通过学习这些超参数，我们能够适应训练数据的分布，并调整它们以实现梯度下降算法所需的特性。广义熵和相关的变形对数概念为新型梯度下降更新提供了更深入的见解。

在文献中，目前存在超过五十种数学上定义完善的熵函数及其相关的变形对数，因此无法在一篇研究论文中一一进行研究。因此，我们在这里集中于迹形式熵及其相关的广义对数。我们已开发的算法应用于在线投资组合选择（OPLS），以提高其性能和稳健性。 

---
# Accuracy of Wearable ECG Parameter Calculation Method for Long QT and First-Degree A-V Block Detection: A Multi-Center Real-World Study with External Validations Compared to Standard ECG Machines and Cardiologist Assessments 

**Title (ZH)**: 长QT综合征和一度房室传导阻滞检测中可穿戴ECG参数计算方法的准确性：一项多中心实际临床研究，并与标准ECG机器和心脏病学家评估进行外部验证对比 

**Authors**: Sumei Fan, Deyun Zhang, Yue Wang, Shijia Geng, Kun Lu, Meng Sang, Weilun Xu, Haixue Wang, Qinghao Zhao, Chuandong Cheng, Peng Wang, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.17499)  

**Abstract**: In recent years, wearable devices have revolutionized cardiac monitoring by enabling continuous, non-invasive ECG recording in real-world settings. Despite these advances, the accuracy of ECG parameter calculations (PR interval, QRS interval, QT interval, etc.) from wearables remains to be rigorously validated against conventional ECG machines and expert clinician assessments. In this large-scale, multicenter study, we evaluated FeatureDB, a novel algorithm for automated computation of ECG parameters from wearable single-lead signals Three diverse datasets were employed: the AHMU-FH dataset (n=88,874), the CSE dataset (n=106), and the HeartVoice-ECG-lite dataset (n=369) with annotations provided by two experienced cardiologists. FeatureDB demonstrates a statistically significant correlation with key parameters (PR interval, QRS duration, QT interval, and QTc) calculated by standard ECG machines and annotated by clinical doctors. Bland-Altman analysis confirms a high level of this http URL,FeatureDB exhibited robust diagnostic performance in detecting Long QT syndrome (LQT) and atrioventricular block interval abnormalities (AVBI),with excellent area under the ROC curve (LQT: 0.836, AVBI: 0.861),accuracy (LQT: 0.856, AVBI: 0.845),sensitivity (LQT: 0.815, AVBI: 0.877),and specificity (LQT: 0.856, AVBI: 0.845).This further validates its clinical reliability. These results validate the clinical applicability of FeatureDB for wearable ECG analysis and highlight its potential to bridge the gap between traditional diagnostic methods and emerging wearable this http URL,this study supports integrating wearable ECG devices into large-scale cardiovascular disease management and early intervention strategies,and it highlights the potential of wearable ECG technologies to deliver accurate,clinically relevant cardiac monitoring while advancing broader applications in cardiovascular care. 

**Abstract (ZH)**: 近年来，可穿戴设备通过在现实环境中实现连续的无创心电图（ECG）记录，彻底改变了心脏监测。尽管取得了这些进展，但可穿戴设备计算ECG参数（如PR间期、QRS间期、QT间期等）的准确性仍然需要通过传统心电图仪器和专家临床评估进行严格验证。在本项大规模、多中心研究中，我们评估了FeatureDB这一新型算法，用于从可穿戴单导联信号中自动计算ECG参数。研究使用了三个不同的数据集：AHMU-FH数据集（n=88,874）、CSE数据集（n=106）和HeartVoice-ECG-lite数据集（n=369），数据集中的注释由两名经验丰富的 cardiologist 提供。FeatureDB 在与标准心电图机器计算的心脏关键参数（如PR间期、QRS时程、QT间期和QTc）以及临床医生标注的参数之间表现出统计学显著的相关性。Bland-Altman分析证实了这一一致性。FeatureDB 在检测长QT综合征（LQT）和房室传导阻滞间期异常（AVBI）方面展现了稳健的诊断性能，表现出优秀的ROC曲线下面积（LQT: 0.836，AVBI: 0.861）、精度（LQT: 0.856，AVBI: 0.845）、灵敏度（LQT: 0.815，AVBI: 0.877）和特异性（LQT: 0.856，AVBI: 0.845）。这进一步验证了其临床可靠性。这些结果验证了FeatureDB 在可穿戴ECG分析中的临床适用性，并突显了这种技术在传统诊断方法和新兴可穿戴设备之间架起桥梁的潜力。本研究支持将可穿戴ECG设备整合到大规模心血管疾病管理和早期干预策略中，并突显了可穿戴ECG技术在提供精确的临床相关心脏监测的同时，为心血管护理更广泛的应用带来的潜力。 

---
# Improving Value-based Process Verifier via Structural Prior Injection 

**Title (ZH)**: 通过结构先验注入提升基于价值的过程验证器 

**Authors**: Zetian Sun, Dongfang Li, Baotian Hu, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17498)  

**Abstract**: In the Large Language Model(LLM) reasoning scenario, people often estimate state value via Monte Carlo sampling. Though Monte Carlo estimation is an elegant method with less inductive bias, noise and errors are inevitably introduced due to the limited sampling. To handle the problem, we inject the structural prior into the value representation and transfer the scalar value into the expectation of a pre-defined categorical distribution, representing the noise and errors from a distribution perspective. Specifically, by treating the result of Monte Carlo sampling as a single sample from the prior ground-truth Binomial distribution, we quantify the sampling error as the mismatch between posterior estimated distribution and ground-truth distribution, which is thus optimized via distribution selection optimization. We test the performance of value-based process verifiers on Best-of-N task and Beam search task. Compared with the scalar value representation, we show that reasonable structural prior injection induced by different objective functions or optimization methods can improve the performance of value-based process verifiers for about 1$\sim$2 points at little-to-no cost. We also show that under different structural prior, the verifiers' performances vary greatly despite having the same optimal solution, indicating the importance of reasonable structural prior injection. 

**Abstract (ZH)**: 在大型语言模型（LLM）推理场景中，人们常常通过蒙特卡洛采样估计状态价值。尽管蒙特卡洛估计方法简洁且引入的归纳偏置较少，但由于采样的局限性，噪声和错误不可避免地会被引入。为了解决这一问题，我们将结构先验注入价值表示，并将标量值转换为预定义分类分布的期望，从概率分布的角度来表征噪声和错误。具体而言，通过将蒙特卡洛采样的结果视为先验真实的二项分布的一个样本，我们量化了采样误差，即后验估计分布与真实分布之间的差异，并通过分布选择优化来优化这一差异。我们在Best-of-N任务和Beam搜索任务中测试了基于价值的过程验证器的性能。与标量值表示相比，我们在几乎没有额外成本的情况下，通过不同目标函数或优化方法引入合理的结构先验，可以提高基于价值的过程验证器的性能约1-2分。我们还发现，在不同的结构先验下，尽管最优解相同，验证器的性能差异很大，这突显了合理结构先验注入的重要性。 

---
# SpikeRL: A Scalable and Energy-efficient Framework for Deep Spiking Reinforcement Learning 

**Title (ZH)**: SpikeRL：一种深度尖峰强化学习的可扩展和能效框架 

**Authors**: Tokey Tahmid, Mark Gates, Piotr Luszczek, Catherine D. Schuman  

**Link**: [PDF](https://arxiv.org/pdf/2502.17496)  

**Abstract**: In this era of AI revolution, massive investments in large-scale data-driven AI systems demand high-performance computing, consuming tremendous energy and resources. This trend raises new challenges in optimizing sustainability without sacrificing scalability or performance. Among the energy-efficient alternatives of the traditional Von Neumann architecture, neuromorphic computing and its Spiking Neural Networks (SNNs) are a promising choice due to their inherent energy efficiency. However, in some real-world application scenarios such as complex continuous control tasks, SNNs often lack the performance optimizations that traditional artificial neural networks have. Researchers have addressed this by combining SNNs with Deep Reinforcement Learning (DeepRL), yet scalability remains unexplored. In this paper, we extend our previous work on SpikeRL, which is a scalable and energy efficient framework for DeepRL-based SNNs for continuous control. In our initial implementation of SpikeRL framework, we depended on the population encoding from the Population-coded Spiking Actor Network (PopSAN) method for our SNN model and implemented distributed training with Message Passing Interface (MPI) through mpi4py. Also, further optimizing our model training by using mixed-precision for parameter updates. In our new SpikeRL framework, we have implemented our own DeepRL-SNN component with population encoding, and distributed training with PyTorch Distributed package with NCCL backend while still optimizing with mixed precision training. Our new SpikeRL implementation is 4.26X faster and 2.25X more energy efficient than state-of-the-art DeepRL-SNN methods. Our proposed SpikeRL framework demonstrates a truly scalable and sustainable solution for complex continuous control tasks in real-world applications. 

**Abstract (ZH)**: 在AI革命的时代，大规模数据驱动的AI系统需要高性能计算，消耗了巨大的能源和资源。这一趋势提出了新的挑战，即在不牺牲可扩展性或性能的情况下优化可持续性。在传统冯·诺依曼架构的节能替代方案中，类脑计算及其脉冲神经网络（SNNs）因其固有的节能特性而具有前景。然而，在某些实际应用场景中，如复杂的连续控制任务，SNNs往往缺乏传统人工神经网络所具备的性能优化。研究者通过将SNNs与深度强化学习（DeepRL）结合来解决这一问题，但可扩展性方面尚未得到充分探索。本文在此基础上扩展了我们之前的工作，即SpikeRL，这是一种基于DeepRL的SNNs可扩展且节能框架，用于连续控制任务。在SpikeRL框架的初始实现中，我们采用了由Population-coded Spiking Actor Network（PopSAN）方法提供的群体编码，并通过mpi4py实现了分布式训练，同时通过混合精度训练优化了模型训练。在新的SpikeRL框架中，我们实现了自有特色的DeepRL-SNN组件，采用了群体编码，并通过NCCL后端的PyTorch Distributed包实现了分布式训练，同时仍然使用混合精度训练优化。与目前最先进的DeepRL-SNN方法相比，我们的新SpikeRL实现快4.26倍，能耗降低2.25倍。我们提出的SpikeRL框架展示了对于实际应用中的复杂连续控制任务一种真正意义上的可扩展且可持续的解决方案。 

---
# External Large Foundation Model: How to Efficiently Serve Trillions of Parameters for Online Ads Recommendation 

**Title (ZH)**: 外部大型基础模型：如何高效地为在线广告推荐服务万亿参数 

**Authors**: Mingfu Liang, Xi Liu, Rong Jin, Boyang Liu, Qiuling Suo, Qinghai Zhou, Song Zhou, Laming Chen, Hua Zheng, Zhiyuan Li, Shali Jiang, Jiyan Yang, Xiaozhen Xia, Fan Yang, Yasmine Badr, Ellie Wen, Shuyu Xu, Hansey Chen, Zhengyu Zhang, Jade Nie, Chunzhi Yang, Zhichen Zeng, Weilin Zhang, Xingliang Huang, Qianru Li, Shiquan Wang, Evelyn Lyu, Wenjing Lu, Rui Zhang, Wenjun Wang, Jason Rudy, Mengyue Hang, Kai Wang, Yinbin Ma, Shuaiwen Wang, Sihan Zeng, Tongyi Tang, Xiaohan Wei, Longhao Jin, Jamey Zhang, Marcus Chen, Jiayi Zhang, Angie Huang, Chi Zhang, Zhengli Zhao, Jared Yang, Qiang Jin, Xian Chen, Amit Anand Amlesahwaram, Lexi Song, Liang Luo, Yuchen Hao, Nan Xiao, Yavuz Yetim, Luoshang Pan, Gaoxiang Liu, Yuxi Hu, Yuzhen Huang, Jackie Xu, Rich Zhu, Xin Zhang, Yiqun Liu, Hang Yin, Yuxin Chen, Buyun Zhang, Xiaoyi Liu, Sylvia Wang, Wenguang Mao, Zhijing Li, Qin Huang, Chonglin Sun, Shupin Mao, Jingzheng Qin, Peggy Yao, Jae-Woo Choi, Bin Gao, Ernest Wang, Lei Zhang, Wen-Yen Chen, Ted Lee, Jay Zha, Yi Meng, Alex Gong, Edison Gao, Alireza Vahdatpour, Yiping Han, Yantao Yao, Toshinari Kureha, Shuo Chang, Musharaf Sultan, John Bocharov, Sagar Chordia, Xiaorui Gan, Peng Sun, Rocky Liu, Bo Long, Wenlin Chen, Santanu Kolay, Huayu Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.17494)  

**Abstract**: Ads recommendation is a prominent service of online advertising systems and has been actively studied. Recent studies indicate that scaling-up and advanced design of the recommendation model can bring significant performance improvement. However, with a larger model scale, such prior studies have a significantly increasing gap from industry as they often neglect two fundamental challenges in industrial-scale applications. First, training and inference budgets are restricted for the model to be served, exceeding which may incur latency and impair user experience. Second, large-volume data arrive in a streaming mode with data distributions dynamically shifting, as new users/ads join and existing users/ads leave the system. We propose the External Large Foundation Model (ExFM) framework to address the overlooked challenges. Specifically, we develop external distillation and a data augmentation system (DAS) to control the computational cost of training/inference while maintaining high performance. We design the teacher in a way like a foundation model (FM) that can serve multiple students as vertical models (VMs) to amortize its building cost. We propose Auxiliary Head and Student Adapter to mitigate the data distribution gap between FM and VMs caused by the streaming data issue. Comprehensive experiments on internal industrial-scale applications and public datasets demonstrate significant performance gain by ExFM. 

**Abstract (ZH)**: 在线广告系统中的广告推荐是一项重要的服务，近年来受到广泛关注并得到了积极的研究。近期的研究表明，增大推荐模型的规模和优化设计可以带来显著的性能提升。然而，随着模型规模的增大，这些先前研究与工业应用之间的差距也显著增加，主要是因为它们往往忽视了大规模工业应用中的两个基本挑战。首先，为了使模型能够提供服务，训练和推理的预算是有限的，超出这一限制可能会导致延迟并影响用户体验。其次，大量数据以流式模式到达，且数据分布会随着时间动态变化，因为新用户和新广告加入系统，而旧用户和旧广告则离开系统。我们提出了外部大型基础模型（ExFM）框架来解决这些被忽视的挑战。具体而言，我们开发了外部蒸馏技术和数据增强系统（DAS），以控制训练和推理的计算成本，同时保持高性能。我们设计教师模型类似于基础模型（FM），能够服务于多个垂直模型（VMs），以摊销其构建成本。我们提出了辅助头和学生适配器来缓解因流式数据问题导致的FM和VMs之间的数据分布差距。在内部工业规模应用和公开数据集上的全面实验结果表明，ExFM可以带来显著的性能提升。 

---
# Pursuing Top Growth with Novel Loss Function 

**Title (ZH)**: 追求卓越增长：新颖的损失函数方法 

**Authors**: Ruoyu Guo, Haochen Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17493)  

**Abstract**: Making consistently profitable financial decisions in a continuously evolving and volatile stock market has always been a difficult task. Professionals from different disciplines have developed foundational theories to anticipate price movement and evaluate securities such as the famed Capital Asset Pricing Model (CAPM). In recent years, the role of artificial intelligence (AI) in asset pricing has been growing. Although the black-box nature of deep learning models lacks interpretability, they have continued to solidify their position in the financial industry. We aim to further enhance AI's potential and utility by introducing a return-weighted loss function that will drive top growth while providing the ML models a limited amount of information. Using only publicly accessible stock data (open/close/high/low, trading volume, sector information) and several technical indicators constructed from them, we propose an efficient daily trading system that detects top growth opportunities. Our best models achieve 61.73% annual return on daily rebalancing with an annualized Sharpe Ratio of 1.18 over 1340 testing days from 2019 to 2024, and 37.61% annual return with an annualized Sharpe Ratio of 0.97 over 1360 testing days from 2005 to 2010. The main drivers for success, especially independent of any domain knowledge, are the novel return-weighted loss function, the integration of categorical and continuous data, and the ML model architecture. We also demonstrate the superiority of our novel loss function over traditional loss functions via several performance metrics and statistical evidence. 

**Abstract (ZH)**: 在不断演变且波动的股票市场上，持续做出盈利性的决策一直是一个艰巨的任务。来自不同学科的专业人士已经发展出了基础理论来预测价格波动和评估证券，例如著名的资本资产定价模型（CAPM）。近年来，人工智能（AI）在资产定价中的作用越来越重要。尽管深度学习模型的黑箱性质缺乏可解释性，但它们在金融行业中的地位仍然不断巩固。我们希望通过引入一种收益加权的损失函数来进一步增强AI的潜力和实用性，该函数能够在提供有限信息的同时推动顶尖增长。仅使用公开可获取的股票数据（开盘价/收盘价/最高价/最低价、交易量、行业信息）以及从这些数据构建的技术指标，我们提出了一种高效的每日交易系统，用于发现顶尖的增长机会。我们的最佳模型在2019年至2024年的1340个交易日中实现了61.73%的年化收益率和年化夏普比率1.18，在2005年至2010年的1360个交易日中实现了37.61%的年化收益率和年化夏普比率0.97。尤其是在无需任何领域知识的情况下，成功的主要驱动力包括新型的收益加权损失函数、类别型和连续型数据的结合以及ML模型架构。我们还通过多种性能指标和统计证据证明了我们新型损失函数的优越性。 

---
# A generalized dual potential for inelastic Constitutive Artificial Neural Networks: A JAX implementation at finite strains 

**Title (ZH)**: 在有限应变下，inelastic构成人工神经网络的广义对偶势函数：JAX实现 

**Authors**: Hagen Holthusen, Kevin Linka, Ellen Kuhl, Tim Brepols  

**Link**: [PDF](https://arxiv.org/pdf/2502.17490)  

**Abstract**: We present a methodology for designing a generalized dual potential, or pseudo potential, for inelastic Constitutive Artificial Neural Networks (iCANNs). This potential, expressed in terms of stress invariants, inherently satisfies thermodynamic consistency for large deformations. In comparison to our previous work, the new potential captures a broader spectrum of material behaviors, including pressure-sensitive inelasticity.
To this end, we revisit the underlying thermodynamic framework of iCANNs for finite strain inelasticity and derive conditions for constructing a convex, zero-valued, and non-negative dual potential. To embed these principles in a neural network, we detail the architecture's design, ensuring a priori compliance with thermodynamics.
To evaluate the proposed architecture, we study its performance and limitations discovering visco-elastic material behavior, though the method is not limited to visco-elasticity. In this context, we investigate different aspects in the strategy of discovering inelastic materials. Our results indicate that the novel architecture robustly discovers interpretable models and parameters, while autonomously revealing the degree of inelasticity.
The iCANN framework, implemented in JAX, is publicly accessible at this https URL. 

**Abstract (ZH)**: 我们提出了一种设计广义双重势能或伪势能的方法，用于非线性构式人工神经网络（iCANNs）。这种势能以应力不变量的形式表示，自然地在大变形情况下满足热力学一致性。与我们之前的工作相比，新开发的势能能够捕捉更广泛材料行为，包括压力敏感的非弹性行为。

为此，我们重新审视了iCANNs在有限应变非弹性行为中的基础热力学框架，并推导出构建凸、零值和非负双重势能的条件。为了将这些原理嵌入神经网络，我们详细说明了网络架构的设计，确保其先验符合热力学原理。

为了评估所提出的架构，我们研究了其在发现粘弹性材料行为方面的性能和局限性，尽管该方法不限于粘弹性。在此背景下，我们探讨了发现非弹性材料策略的不同方面。结果表明，该新型架构能够稳健地发现可解释的模型和参数，并自动揭示材料的非弹性程度。

iCANN框架已使用JAX实现，并可通过以下链接公开访问：这个 https URL。 

---
# Using Graph Convolutional Networks to Address fMRI Small Data Problems 

**Title (ZH)**: 使用图卷积网络解决功能性磁共振成像（fMRI）小数据问题 

**Authors**: Thomas Screven, Andras Necz, Jason Smucny, Ian Davidson  

**Link**: [PDF](https://arxiv.org/pdf/2502.17489)  

**Abstract**: Although great advances in the analysis of neuroimaging data have been made, a major challenge is a lack of training data. This is less problematic in tasks such as diagnosis, where much data exists, but particularly prevalent in harder problems such as predicting treatment responses (prognosis), where data is focused and hence limited. Here, we address the learning from small data problems for medical imaging using graph neural networks. This is particularly challenging as the information about the patients is themselves graphs (regions of interest connectivity graphs). We show how a spectral representation of the connectivity data allows for efficient propagation that can yield approximately 12\% improvement over traditional deep learning methods using the exact same data. We show that our method's superior performance is due to a data smoothing result that can be measured by closing the number of triangle inequalities and thereby satisfying transitivity. 

**Abstract (ZH)**: 尽管在神经成像数据分析方面取得了巨大进展，但一个主要挑战是缺乏训练数据。在诊断等任务中，由于存在大量数据，这个问题相对较少，但在预测治疗反应（预后）等更复杂的问题中尤为突出，这些问题的数据关注度高，因此有限。为此，我们使用图神经网络解决医疗成像中的小数据学习问题。这一问题尤为具有挑战性，因为关于患者的信息本身就是图（感兴趣区域的连接图）。我们展示了如何通过谱表示连接数据来实现高效的传播，这可以在使用完全相同数据的情况下，使性能提升约12%。我们证明了我们方法的优越性能归因于一个数据平滑结果，该结果可以通过减少三角不等式的数量来衡量，从而满足传递性。 

---
# Toward Foundational Model for Sleep Analysis Using a Multimodal Hybrid Self-Supervised Learning Framework 

**Title (ZH)**: 使用多模态混合自我监督学习框架构建睡眠分析基础模型 

**Authors**: Cheol-Hui Lee, Hakseung Kim, Byung C. Yoon, Dong-Joo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.17481)  

**Abstract**: Sleep is essential for maintaining human health and quality of life. Analyzing physiological signals during sleep is critical in assessing sleep quality and diagnosing sleep disorders. However, manual diagnoses by clinicians are time-intensive and subjective. Despite advances in deep learning that have enhanced automation, these approaches remain heavily dependent on large-scale labeled datasets. This study introduces SynthSleepNet, a multimodal hybrid self-supervised learning framework designed for analyzing polysomnography (PSG) data. SynthSleepNet effectively integrates masked prediction and contrastive learning to leverage complementary features across multiple modalities, including electroencephalogram (EEG), electrooculography (EOG), electromyography (EMG), and electrocardiogram (ECG). This approach enables the model to learn highly expressive representations of PSG data. Furthermore, a temporal context module based on Mamba was developed to efficiently capture contextual information across signals. SynthSleepNet achieved superior performance compared to state-of-the-art methods across three downstream tasks: sleep-stage classification, apnea detection, and hypopnea detection, with accuracies of 89.89%, 99.75%, and 89.60%, respectively. The model demonstrated robust performance in a semi-supervised learning environment with limited labels, achieving accuracies of 87.98%, 99.37%, and 77.52% in the same tasks. These results underscore the potential of the model as a foundational tool for the comprehensive analysis of PSG data. SynthSleepNet demonstrates comprehensively superior performance across multiple downstream tasks compared to other methodologies, making it expected to set a new standard for sleep disorder monitoring and diagnostic systems. 

**Abstract (ZH)**: 睡眠对维持人类健康和生活质量至关重要。分析睡眠期间的生理信号是评估睡眠质量和诊断睡眠障碍的关键。然而，临床医生的手动诊断耗时且主观。尽管深度学习的进步提高了自动化水平，但这些方法仍然高度依赖大规模标注数据集。本研究引入了SynthSleepNet，这是一种用于分析多导睡眠图（polysomnography, PSG）数据的多模态混合自监督学习框架。SynthSleepNet有效地结合了遮蔽预测和对比学习，利用多个模态中的互补特征，包括脑电图（EEG）、眼电图（EOG）、肌电图（EMG）和心电图（ECG），从而使模型能够学习PSG数据的高表达表示。此外，基于Mamba开发了一种时间上下文模块，以高效地捕捉信号之间的上下文信息。在三项下游任务——睡眠分期分类、呼吸暂停检测和低通气检测——中，SynthSleepNet分别达到了89.89%、99.75%和89.60%的准确率，表现优于最先进的方法。在半监督学习环境中，尽管标签有限，该模型仍表现出稳健的性能，并分别实现了87.98%、99.37%和77.52%的准确率。这些结果突显了该模型作为全面分析PSG数据工具的潜力。SynthSleepNet在多项下游任务中的综合性能显著优于其他方法，预期将为睡眠障碍监测和诊断系统设立新的标准。 

---
# Brain-to-Text Decoding: A Non-invasive Approach via Typing 

**Title (ZH)**: 将脑电转化成文本：一种通过打字实现的无侵入性方法 

**Authors**: Jarod Lévy, Mingfang Zhang, Svetlana Pinet, Jérémy Rapin, Hubert Banville, Stéphane d'Ascoli, Jean-Rémi King  

**Link**: [PDF](https://arxiv.org/pdf/2502.17480)  

**Abstract**: Modern neuroprostheses can now restore communication in patients who have lost the ability to speak or move. However, these invasive devices entail risks inherent to neurosurgery. Here, we introduce a non-invasive method to decode the production of sentences from brain activity and demonstrate its efficacy in a cohort of 35 healthy volunteers. For this, we present Brain2Qwerty, a new deep learning architecture trained to decode sentences from either electro- (EEG) or magneto-encephalography (MEG), while participants typed briefly memorized sentences on a QWERTY keyboard. With MEG, Brain2Qwerty reaches, on average, a character-error-rate (CER) of 32% and substantially outperforms EEG (CER: 67%). For the best participants, the model achieves a CER of 19%, and can perfectly decode a variety of sentences outside of the training set. While error analyses suggest that decoding depends on motor processes, the analysis of typographical errors suggests that it also involves higher-level cognitive factors. Overall, these results narrow the gap between invasive and non-invasive methods and thus open the path for developing safe brain-computer interfaces for non-communicating patients. 

**Abstract (ZH)**: 现代神经假体现在可以恢复失去说话或移动能力的患者的交流。然而，这些侵入性设备伴随着与神经外科手术相关的风险。在此，我们介绍了一种无创方法，用于解码大脑活动产生的句子，并在35名健康志愿者中证明了其有效性。为此，我们提出了Brain2Qwerty，这是一种经过训练的新深度学习架构，用于从脑电图（EEG）或磁源成像（MEG）解码句子，参与者在QWERTY键盘上快速输入已记忆的句子。使用MEG时，Brain2Qwerty在平均字符错误率（CER）上达到32%，显著优于EEG（CER：67%）。对于表现最佳的参与者，该模型的CER为19%，甚至可以完全解码训练集之外的各种句子。误码分析表明，解码依赖于运动过程，而打字错误分析则表明还涉及高级认知因素。总体而言，这些结果缩小了侵入性和非侵入性方法之间的差距，从而为开发适用于无法交流患者的脑-机接口开辟了道路。 

---
# ECG-Expert-QA: A Benchmark for Evaluating Medical Large Language Models in Heart Disease Diagnosis 

**Title (ZH)**: ECG-Expert-QA：评估心脏疾病诊断中医学大规模语言模型性能的基准 

**Authors**: Xu Wang, Jiaju Kang, Puyu Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.17475)  

**Abstract**: We present ECG-Expert-QA, a comprehensive multimodal dataset designed for evaluating diagnostic capabilities in ECG interpretation, integrating real clinical data with systematically generated synthetic cases. The dataset encompasses six fundamental diagnostic tasks, comprising 47,211 meticulously curated question-answer pairs that span a spectrum of clinical scenarios, from basic rhythm analysis to complex case interpretation. By simulating challenging clinical cases through a rigorous medical knowledge-guided process, ECG-Expert-QA not only enhances the availability of annotated diagnostic data but also significantly increases the complexity and diversity of clinical presentations, including rare cardiac conditions and temporal progression patterns. This design enables comprehensive evaluation of medical language models across multiple dimensions, including diagnostic accuracy, clinical reasoning, and knowledge integration. To facilitate global research collaboration, ECG-Expert-QA is available in both Chinese and English versions, with rigorous quality control ensuring linguistic and clinical consistency. The dataset's challenging diagnostic tasks, which include interpretation of complex arrhythmias, identification of subtle ischemic changes, and integration of clinical context, establish it as an effective benchmark for advancing AI-assisted ECG interpretation and pushing the boundaries of current diagnostic models. Our dataset is open-source and available at this https URL. 

**Abstract (ZH)**: 我们介绍了ECG-Expert-QA，这是一个全面的多模态数据集，旨在评估心电图（ECG）解读中的诊断能力，结合了真实临床数据与系统生成的合成病例。数据集涵盖了六个基本的诊断任务，包括47,211个精心策划的问题-答案对，这些对覆盖了从基本节律分析到复杂病例解读的各种临床场景。通过严格的医学知识引导过程模拟具有挑战性的临床病例，ECG-Expert-QA 不仅增加了标注诊断数据的可获得性，还显著提高了临床表现的复杂性和多样性，包括罕见心脏状况和时间进展模式。这种设计使得多维度地评估医疗语言模型（包括诊断准确性、临床推理和知识整合）成为可能。为了促进全球研究合作，ECG-Expert-QA 提供了中文和英文两个版本，并通过严格的质量控制确保语言和临床一致性。数据集中的具有挑战性的诊断任务，包括复杂心律失常的解释、细微缺血变化的识别以及临床背景的结合，使其成为推动基于AI的心电图解读发展和拓展当前诊断模型界限的有效基准。该数据集是开放源代码的，并可通过以下链接访问：[这里插入链接]。 

---
# MC2SleepNet: Multi-modal Cross-masking with Contrastive Learning for Sleep Stage Classification 

**Title (ZH)**: MC2SleepNet：多模态对比学习掩蔽方法在睡眠阶段分类中的应用 

**Authors**: Younghoon Na  

**Link**: [PDF](https://arxiv.org/pdf/2502.17470)  

**Abstract**: Sleep profoundly affects our health, and sleep deficiency or disorders can cause physical and mental problems. % Despite significant findings from previous studies, challenges persist in optimizing deep learning models, especially in multi-modal learning for high-accuracy sleep stage classification. Our research introduces MC2SleepNet (Multi-modal Cross-masking with Contrastive learning for Sleep stage classification Network). It aims to facilitate the effective collaboration between Convolutional Neural Networks (CNNs) and Transformer architectures for multi-modal training with the help of contrastive learning and cross-masking. % Raw single channel EEG signals and corresponding spectrogram data provide differently characterized modalities for multi-modal learning. Our MC2SleepNet has achieved state-of-the-art performance with an accuracy of both 84.6% on the SleepEDF-78 and 88.6% accuracy on the Sleep Heart Health Study (SHHS). These results demonstrate the effective generalization of our proposed network across both small and large datasets. 

**Abstract (ZH)**: 睡眠对我们健康影响深远，睡眠不足或障碍会导致身心问题。尽管先前的研究取得了显著成果，但在优化深度学习模型方面仍存在挑战，尤其是在高精度睡眠阶段分类的多模态学习中。我们的研究引入了MC2SleepNet（多模态对比学习与交叉掩码的睡眠阶段分类网络），旨在通过对比学习和交叉掩码促进卷积神经网络（CNN）与变换器架构之间的有效协作，实现多模态训练。原始的单通道脑电图（EEG）信号及其相应的频谱图数据提供了不同的模态，为多模态学习提供了基础。我们的MC2SleepNet在SleepEDF-78数据集上取得了84.6%的准确率，在睡眠心脏健康研究（SHHS）数据集上取得了88.6%的准确率。这些结果表明，所提出的网络在不同规模的数据集上具有有效的泛化能力。 

---
# PixleepFlow: A Pixel-Based Lifelog Framework for Predicting Sleep Quality and Stress Level 

**Title (ZH)**: 基于像素的LifeLog框架：预测睡眠质量和压力水平的PixleepFlow方法 

**Authors**: Younghoon Na  

**Link**: [PDF](https://arxiv.org/pdf/2502.17469)  

**Abstract**: The analysis of lifelogs can yield valuable insights into an individual's daily life, particularly with regard to their health and well-being. The accurate assessment of quality of life is necessitated by the use of diverse sensors and precise synchronization. To rectify this issue, this study proposes the image-based sleep quality and stress level estimation flow (PixleepFlow). PixleepFlow employs a conversion methodology into composite image data to examine sleep patterns and their impact on overall health. Experiments were conducted using lifelog datasets to ascertain the optimal combination of data formats. In addition, we identified which sensor information has the greatest influence on the quality of life through Explainable Artificial Intelligence(XAI). As a result, PixleepFlow produced more significant results than various data formats. This study was part of a written-based competition, and the additional findings from the lifelog dataset are detailed in Section Section IV. More information about PixleepFlow can be found at this https URL. 

**Abstract (ZH)**: 生命日志的分析可以为个体日常生活提供有价值的见解，特别是在健康和福祉方面。為了准确评估生活质量，需要使用多种传感器并进行精确同步。为解决这一问题，本研究提出了基于图像的睡眠质量和压力水平估计流程（PixleepFlow）。PixleepFlow利用图像数据转换方法来分析睡眠模式及其对整体健康的影响。我们使用生命日志数据集进行了实验，以确定数据格式的最佳组合。此外，通过可解释的人工智能（XAI），我们确定了哪些传感器信息对生活质量影响最大。结果表明，PixleepFlow产生了更显著的结果，优于各种数据格式。本研究是基于书面比赛的一部分，生命日志数据集的额外发现详见第四部分。有关PixleepFlow的更多信息，请访问这个网址：[这个网址]。 

---
# The Case for Cleaner Biosignals: High-fidelity Neural Compressor Enables Transfer from Cleaner iEEG to Noisier EEG 

**Title (ZH)**: 更清洁的生物信号必要性研究：高保真神经压缩器使清洁的iEEG能够转移到更噪杂的EEG上 

**Authors**: Francesco Stefano Carzaniga, Gary Tom Hoppeler, Michael Hersche, Kaspar Anton Schindler, Abbas Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17462)  

**Abstract**: All data modalities are not created equal, even when the signal they measure comes from the same source. In the case of the brain, two of the most important data modalities are the scalp electroencephalogram (EEG), and the intracranial electroencephalogram (iEEG). They are used by human experts, supported by deep learning (DL) models, to accomplish a variety of tasks, such as seizure detection and motor imagery classification. Although the differences between EEG and iEEG are well understood by human experts, the performance of DL models across these two modalities remains under-explored. To help characterize the importance of clean data on the performance of DL models, we propose BrainCodec, a high-fidelity EEG and iEEG neural compressor. We find that training BrainCodec on iEEG and then transferring to EEG yields higher reconstruction quality than training on EEG directly. In addition, we also find that training BrainCodec on both EEG and iEEG improves fidelity when reconstructing EEG. Our work indicates that data sources with higher SNR, such as iEEG, provide better performance across the board also in the medical time-series domain. BrainCodec also achieves up to a 64x compression on iEEG and EEG without a notable decrease in quality. BrainCodec markedly surpasses current state-of-the-art compression models both in final compression ratio and in reconstruction fidelity. We also evaluate the fidelity of the compressed signals objectively on a seizure detection and a motor imagery task performed by standard DL models. Here, we find that BrainCodec achieves a reconstruction fidelity high enough to ensure no performance degradation on the downstream tasks. Finally, we collect the subjective assessment of an expert neurologist, that confirms the high reconstruction quality of BrainCodec in a realistic scenario. The code is available at this https URL. 

**Abstract (ZH)**: 各种数据模态并非平等的，即使它们测量的信号来自相同的源头。以大脑为例，最重要的两种数据模态是头皮脑电图（EEG）和颅内脑电图（iEEG）。这两种模态由人类专家和深度学习（DL）模型使用，以完成多种任务，如癫痫检测和运动想象分类。尽管人类专家深知EEG和iEEG之间的差异，但DL模型在这两种模态上的表现仍缺乏探索。为帮助揭示清洁数据对DL模型性能的重要性，我们提出了一种高保真EEG和iEEG神经压缩器——BrainCodec。我们发现，在iEEG上训练BrainCodec后转移到EEG上的重建质量高于直接在EEG上进行训练。此外，我们还发现，在EEG和iEEG上同时训练BrainCodec可以提高EEG重建的保真度。我们的研究显示，具有更高信噪比（SNR）的数据源，如iEEG，在医学时序领域中也表现出更优的整体性能。BrainCodec在iEEG和EEG上的压缩比例最高可达64倍，且质量下降不明显。BrainCodec在最终压缩比和重建保真度方面均优于当前最先进的压缩模型。我们还在标准DL模型执行的癫痫检测和运动想象任务中客观评估了压缩信号的保真度。结果显示，BrainCodec的重建保真度足够高，确保下游任务的性能不降。最后，我们收集了一位专家神经学家的主观评估，证实了在实际场景中BrainCodec的高重建质量。代码已通过以下链接提供：[此链接]。 

---
# Finetuning and Quantization of EEG-Based Foundational BioSignal Models on ECG and PPG Data for Blood Pressure Estimation 

**Title (ZH)**: 基于EEG的生物信号基础模型在ECG和PPG数据上的微调与量化研究：用于血压估计 

**Authors**: Bálint Tóth, Dominik Senti, Thorir Mar Ingolfsson, Jeffrey Zweidler, Alexandre Elsig, Luca Benini, Yawei Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.17460)  

**Abstract**: Blood pressure (BP) is a key indicator of cardiovascular health. As hypertension remains a global cause of morbidity and mortality, accurate, continuous, and non-invasive BP monitoring is therefore of paramount importance. Photoplethysmography (PPG) and electrocardiography (ECG) can potentially enable continuous BP monitoring, yet training accurate and robust machine learning (ML) models remains challenging due to variability in data quality and patient-specific factors. Recently, multiple research groups explored Electroencephalographic (EEG)--based foundation models and demonstrated their exceptional ability to learn rich temporal resolution. Considering the morphological similarities between different biosignals, the question arises of whether a model pre-trained on one modality can effectively be exploited to improve the accuracy of a different signal type. In this work, we take an initial step towards generalized biosignal foundation models by investigating whether model representations learned from abundant EEG data can effectively be transferred to ECG/PPG data solely with fine-tuning, without the need for large-scale additional pre-training, for the BP estimation task. Evaluations on the MIMIC-III and VitalDB datasets demonstrate that our approach achieves near state-of-the-art accuracy for diastolic BP (mean absolute error of 1.57 mmHg) and surpasses by 1.5x the accuracy of prior works for systolic BP (mean absolute error 2.72 mmHg). Additionally, we perform dynamic INT8 quantization, reducing the smallest model size by over 3.5x (from 13.73 MB down to 3.83 MB) while preserving performance, thereby enabling unobtrusive, real-time BP monitoring on resource-constrained wearable devices. 

**Abstract (ZH)**: 血压（BP）是心血管健康的关键指标。由于高血压依然是导致全球疾病负担和死亡的主要原因之一，因此准确、连续且无创的血压监测变得至关重要。光体积描记法（PPG）和心电图（ECG）有可能实现连续的血压监测，但由于数据质量和患者特定因素的差异性，训练准确且稳健的机器学习（ML）模型仍然是一个挑战。最近，多个研究小组探索了基于脑电图（EEG）的基础模型，并展示了其卓越的时间分辨率学习能力。考虑到不同生物信号在形态上的相似性，人们不禁要问，一种模态基础模型是否能够在无需大规模额外预训练的情况下，通过微调有效地应用于另一种信号类型中以提高准确度。在本工作中，我们试图通过研究是否可以从丰富的EEG数据中学习到的模型表示，通过微调可以直接应用于ECG/PPG数据中，从而为血压估计任务提供一种通用的生物信号基础模型。在MIMIC-III和VitalDB数据集上的评估结果显示，我们的方法实现了接近最先进的二相血压（收缩压）平均绝对误差（1.57 mmHg）和比之前的工作高出1.5倍的一相血压（舒张压）平均绝对误差（2.72 mmHg）。此外，我们还进行了动态INT8量化，使得最小模型大小减少了3.5倍以上（从13.73 MB降至3.83 MB），同时保持了性能，从而能够在资源受限的可穿戴设备上实现无干扰的实时血压监测。 

---
# MoEMba: A Mamba-based Mixture of Experts for High-Density EMG-based Hand Gesture Recognition 

**Title (ZH)**: MoEMba：基于Mamba的专家混合模型用于高密度肌电图手部手势识别 

**Authors**: Mehran Shabanpour, Kasra Rad, Sadaf Khademi, Arash Mohammadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.17457)  

**Abstract**: High-Density surface Electromyography (HDsEMG) has emerged as a pivotal resource for Human-Computer Interaction (HCI), offering direct insights into muscle activities and motion intentions. However, a significant challenge in practical implementations of HD-sEMG-based models is the low accuracy of inter-session and inter-subject classification. Variability between sessions can reach up to 40% due to the inherent temporal variability of HD-sEMG signals. Targeting this challenge, the paper introduces the MoEMba framework, a novel approach leveraging Selective StateSpace Models (SSMs) to enhance HD-sEMG-based gesture recognition. The MoEMba framework captures temporal dependencies and cross-channel interactions through channel attention techniques. Furthermore, wavelet feature modulation is integrated to capture multi-scale temporal and spatial relations, improving signal representation. Experimental results on the CapgMyo HD-sEMG dataset demonstrate that MoEMba achieves a balanced accuracy of 56.9%, outperforming its state-of-the-art counterparts. The proposed framework's robustness to session-to-session variability and its efficient handling of high-dimensional multivariate time series data highlight its potential for advancing HD-sEMG-powered HCI systems. 

**Abstract (ZH)**: 高密度表面肌电图（HDsEMG）已成为人机交互（HCI）的关键资源，可直接提供关于肌肉活动和运动意图的洞察。然而，在HD-sEMG模型的实际应用中，跨会话和跨被试分类的准确性较低是一个显著的挑战。HD-sEMG信号固有的时域变异可能导致跨会话间变量达到40%。针对这一挑战，本文提出了MoEMba框架，这是一种利用选择性状态空间模型（SSMs）来增强基于HD-sEMG的手势识别的新方法。MoEMba框架通过通道关注技术捕捉时间依赖性和跨通道交互。此外，集成小波特征调制以捕捉多尺度时间和空间关系，从而提高信号表示能力。在CapgMyo HD-sEMG数据集上的实验结果表明，MoEMba实现了均衡准确率为56.9%，超越了现有的先进方法。所提框架对会话间变异性具有鲁棒性，并有效处理高维多元时间序列数据，突显了其在HD-sEMG驱动的人机交互系统中的潜在应用价值。 

---
# Survey on Recent Progress of AI for Chemistry: Methods, Applications, and Opportunities 

**Title (ZH)**: 近期AI在化学领域进展综述：方法、应用与机遇 

**Authors**: Ding Hu, Pengxiang Hua, Zhen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.17456)  

**Abstract**: The development of artificial intelligence (AI) techniques has brought revolutionary changes across various realms. In particular, the use of AI-assisted methods to accelerate chemical research has become a popular and rapidly growing trend, leading to numerous groundbreaking works. In this paper, we provide a comprehensive review of current AI techniques in chemistry from a computational perspective, considering various aspects in the design of methods. We begin by discussing the characteristics of data from diverse sources, followed by an overview of various representation methods. Next, we review existing models for several topical tasks in the field, and conclude by highlighting some key challenges that warrant further attention. 

**Abstract (ZH)**: 人工智能（AI）技术的发展在各个领域带来了革命性的变化。特别是在使用AI辅助方法加快化学研究方面，这一趋势变得日益流行并迅速增长，产生了大量开创性的工作。在本文中，我们从计算的角度对当前化学领域的AI技术进行了全面回顾，考虑了方法设计中的各个方面。我们首先讨论了来自不同来源的数据特征，随后概述了各种表示方法。接着，我们审查了领域内几个热点任务的现有模型，并总结了几个值得进一步关注的关键挑战。 

---
# Smart Sampling Strategies for Wireless Industrial Data Acquisition 

**Title (ZH)**: 无线工业数据采集中的智能采样策略 

**Authors**: Marcos Soto  

**Link**: [PDF](https://arxiv.org/pdf/2502.17454)  

**Abstract**: In industrial environments, data acquisition accuracy is crucial for process control and optimization. Wireless telemetry has proven to be a valuable tool for improving efficiency in well-testing operations, enabling bidirectional communication and real-time control of downhole tools. However, high sampling frequencies present challenges in telemetry, including data storage, transmission, computational resource consumption, and battery life of wireless devices. This study explores how optimizing data acquisition strategies can reduce aliasing effects and systematic errors while improving sampling rates without compromising measurement accuracy. A reduction of 80% in sampling frequency was achieved without degrading measurement quality, demonstrating the potential for resource optimization in industrial environments. 

**Abstract (ZH)**: 在工业环境中，数据获取准确性对于过程控制和优化至关重要。无线遥测已被证明是提高井测试操作效率的有力工具，能够实现双向通信和井下工具的实时控制。然而，高采样频率给无线传输带来了挑战，包括数据存储、传输、计算资源消耗和无线设备的电池寿命问题。本研究探讨了如何通过优化数据获取策略来减少混叠效应和系统误差，同时提高采样率而不牺牲测量精度。研究结果显示，通过优化策略实现了采样频率80%的降低，而测量质量未受影响，这展示了在工业环境中优化资源利用的潜力。 

---
# AirTag, You're It: Reverse Logistics and Last Mile Dynamics 

**Title (ZH)**: AirTag，轮到你了：逆向物流与最后-mile 动态 

**Authors**: David Noever, Forrest McKee  

**Link**: [PDF](https://arxiv.org/pdf/2502.17447)  

**Abstract**: This study addresses challenges in reverse logistics, a frequently overlooked but essential component of last-mile delivery, particularly in disaster relief scenarios where infrastructure disruptions demand adaptive solutions. While hub-and-spoke logistics networks excel at long-distance scalability, they often fail to optimize closely spaced spokes reliant on distant hubs, introducing inefficiencies in transit times and resource allocation. Using 20 Apple AirTags embedded in packages, this research provides empirical insights into logistical flows, capturing granular spatial and temporal data through Bluetooth LE (BLE) 5 trackers integrated with the Apple Find My network. These trackers demonstrated their value in monitoring dynamic cargo movements, enabling real-time adjustments in mobile hub placement and route optimization, particularly in disaster relief contexts like Hurricane Helene. A novel application of discrete event simulation (DES) further explored the saddle point in hub-spoke configurations, where excessive hub reliance clashes with diminishing spoke interaction demand. By coupling simulation results with empirical AirTag tracking, the study highlights the potential of BLE technology to refine reverse logistics, reduce delays, and improve operational flexibility in both routine and crisis-driven delivery networks. 

**Abstract (ZH)**: 本研究旨在解决逆向物流中遇到的挑战，这是最后一公里交付中一个经常被忽视但至关重要的组成部分，尤其是在基础设施中断导致需要适应性解决方案的灾害救援场景中。尽管 hub-and-spoke 物流网络在远距离规模扩展方面表现出色，但它们经常无法优化依赖遥远枢纽的紧密分布的 spoke，从而导致转运时间和资源分配方面的低效率。本文通过在20个包裹中嵌入Apple AirTags，利用 BLE 5 跟踪器与 Apple Find My 网络集成，提供了关于物流流程的实证见解，并通过蓝牙低功耗（BLE）技术捕获详细的时空数据。这些跟踪器展示了其在监测动态货物移动方面的价值，使研究人员能够进行实时调整，选择移动枢纽并优化路线，特别是在如飓风赫尔内这样的灾害救援情境中。此外，通过连续事件仿真（DES）的创新应用，进一步探讨了 hub-and-spoke 配置中的鞍点，即过度依赖枢纽与逐渐减少的 spoke 交互需求之间的矛盾。通过结合仿真结果和 AirTag 跟踪数据，本研究强调了 BLE 技术在完善逆向物流、减少延误以及提高常规和危机驱动交付网络的操作灵活性方面的潜力。 

---
# DCentNet: Decentralized Multistage Biomedical Signal Classification using Early Exits 

**Title (ZH)**: DCentNet：基于早期退出的去中心化多阶段生物医学信号分类 

**Authors**: Xiaolin Li, Binhua Huang, Barry Cardiff, Deepu John  

**Link**: [PDF](https://arxiv.org/pdf/2502.17446)  

**Abstract**: DCentNet is a novel decentralized multistage signal classification approach designed for biomedical data from IoT wearable sensors, integrating early exit points (EEP) to enhance energy efficiency and processing speed. Unlike traditional centralized processing methods, which result in high energy consumption and latency, DCentNet partitions a single CNN model into multiple sub-networks using EEPs. By introducing encoder-decoder pairs at EEPs, the system compresses large feature maps before transmission, significantly reducing wireless data transfer and power usage. If an input is confidently classified at an EEP, processing stops early, optimizing efficiency. Initial sub-networks can be deployed on fog or edge devices to further minimize energy consumption. A genetic algorithm is used to optimize EEP placement, balancing performance and complexity. Experimental results on ECG classification show that with one EEP, DCentNet reduces wireless data transmission by 94.54% and complexity by 21%, while maintaining original accuracy and sensitivity. With two EEPs, sensitivity reaches 98.36%, accuracy 97.74%, wireless data transmission decreases by 91.86%, and complexity is reduced by 22%. Implemented on an ARM Cortex-M4 MCU, DCentNet achieves an average power saving of 73.6% compared to continuous wireless ECG transmission. 

**Abstract (ZH)**: DCentNet 是一种新颖的去中心化多阶段信号分类方法，专门针对来自物联网穿戴传感器的生物医学数据进行设计，通过集成早期退出点 (EEP) 来增强能量效率和处理速度。与传统的集中式处理方法相比，后者会导致高能耗和高延迟，DCentNet 将单一的 CNN 模型划分为多个子网络，并通过在早期退出点引入编码器-解码器对来压缩较大的特征图，从而显著减少了无线数据传输和能耗。如果在早期退出点 confident 地完成了输入分类，则可以在早期停止处理以优化效率。初始子网络可以在雾计算或边缘设备上部署，进一步减少能耗。通过遗传算法优化早期退出点的放置位置，以平衡性能和复杂性。在心电图 (ECG) 分类实验中，当使用一个早期退出点时，DCentNet 将无线数据传输量减少了 94.54%，复杂性减少了 21%，同时保持了原始的准确率和灵敏度。当使用两个早期退出点时，灵敏度达到了 98.36%，准确率达到了 97.74%，无线数据传输量减少了 91.86%，复杂性减少了 22%。此方法在 ARM Cortex-M4 微控制器上实现了与连续无线心电图传输相比平均 73.6% 的节能效果。 

---
# Interpretable Dual-Filter Fuzzy Neural Networks for Affective Brain-Computer Interfaces 

**Title (ZH)**: 可解释的双滤波模糊神经网络在情感脑机接口中的应用 

**Authors**: Xiaowei Jiang, Yanan Chen, Nikhil Ranjan Pal, Yu-Cheng Chang, Yunkai Yang, Thomas Do, Chin-Teng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.17445)  

**Abstract**: Fuzzy logic provides a robust framework for enhancing explainability, particularly in domains requiring the interpretation of complex and ambiguous signals, such as brain-computer interface (BCI) systems. Despite significant advances in deep learning, interpreting human emotions remains a formidable challenge. In this work, we present iFuzzyAffectDuo, a novel computational model that integrates a dual-filter fuzzy neural network architecture for improved detection and interpretation of emotional states from neuroimaging data. The model introduces a new membership function (MF) based on the Laplace distribution, achieving superior accuracy and interpretability compared to traditional approaches. By refining the extraction of neural signals associated with specific emotions, iFuzzyAffectDuo offers a human-understandable framework that unravels the underlying decision-making processes. We validate our approach across three neuroimaging datasets using functional Near-Infrared Spectroscopy (fNIRS) and Electroencephalography (EEG), demonstrating its potential to advance affective computing. These findings open new pathways for understanding the neural basis of emotions and their application in enhancing human-computer interaction. 

**Abstract (ZH)**: 模糊逻辑提供了一种稳健的框架，用于提高可解释性，特别是在需要解释复杂和模糊信号的领域中，例如脑机接口（BCI）系统。尽管深度学习取得了显著进展，但解读人类情感仍然是一项艰巨的挑战。在此项工作中，我们提出了一种名为iFuzzyAffectDuo的创新计算模型，该模型结合了双滤波模糊神经网络架构，以提高情绪状态从神经影像数据中检测和解释的准确性与解释性。该模型引入了一种基于拉普拉斯分布的新隶属函数（MF），其准确性和解释性均优于传统方法。通过精炼与特定情绪相关的神经信号的提取，iFuzzyAffectDuo提供了一个容易为人理解的框架，阐明了背后的决策过程。我们使用功能性近红外光谱成像（fNIRS）和脑电图（EEG）的三个神经影像数据集对我们的方法进行了验证，展示了其在促进情感计算方面的潜力。这些发现为理解情绪的神经基础及其在增强人机交互中的应用开辟了新的途径。 

---
# AI Agentic workflows and Enterprise APIs: Adapting API architectures for the age of AI agents 

**Title (ZH)**: AI代理工作流与企业API：适应AI代理时代的企业API架构 

**Authors**: Vaibhav Tupe, Shrinath Thube  

**Link**: [PDF](https://arxiv.org/pdf/2502.17443)  

**Abstract**: The rapid advancement of Generative AI has catalyzed the emergence of autonomous AI agents, presenting unprecedented challenges for enterprise computing infrastructures. Current enterprise API architectures are predominantly designed for human-driven, predefined interaction patterns, rendering them ill-equipped to support intelligent agents' dynamic, goal-oriented behaviors. This research systematically examines the architectural adaptations for enterprise APIs to support AI agentic workflows effectively. Through a comprehensive analysis of existing API design paradigms, agent interaction models, and emerging technological constraints, the paper develops a strategic framework for API transformation. The study employs a mixed-method approach, combining theoretical modeling, comparative analysis, and exploratory design principles to address critical challenges in standardization, performance, and intelligent interaction. The proposed research contributes a conceptual model for next-generation enterprise APIs that can seamlessly integrate with autonomous AI agent ecosystems, offering significant implications for future enterprise computing architectures. 

**Abstract (ZH)**: 生成式人工智能的快速发展促生了自主人工智能代理的出现，给企业计算基础设施带来了前所未有的挑战。当前的企业API架构主要针对由人类引导的预定义交互模式设计，这使得它们难以支持智能代理动态的目标导向行为。本研究系统地探讨了企业API架构的调整，以有效支持人工智能代理的工作流程。通过全面分析现有API设计范式、代理交互模型以及新兴的技术约束，本文提出了一种API转型的战略框架。研究采用混合方法，结合理论建模、比较分析和探索性设计原则，以解决标准制定、性能和智能交互的关键挑战。提出的研究所提供的概念模型能够无缝集成到自主人工智能代理生态系统中，对未来的企业计算架构具有重要意义。 

---
# Thinking Before Running! Efficient Code Generation with Thorough Exploration and Optimal Refinement 

**Title (ZH)**: 思考在先！通过全面探索与最优精炼实现高效代码生成 

**Authors**: Xiaoqing Zhang, Yuhan Liu, Flood Sung, Xiuying Chen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2502.17442)  

**Abstract**: Code generation is crucial in software engineering for automating the coding process efficiently. While test-time computation methods show promise, they suffer from high latency due to multiple computation rounds. To overcome this, we introduce ThinkCoder, a framework that combines thorough exploration with optimal refinement. The exploration phase diversifies the solution space by searching for potential solutions, followed by a refinement phase that enhances precision. This approach allows us to select the best solution through careful consideration before taking action, avoiding excessive trial and error. To further minimize test-time computation overhead, we introduce preference-driven optimization with Reinforced Self-Training (ReST), which uses exploration trajectories from ThinkCoder to guide LLM's evolution. By learning preferences, this approach improves LLM's exploration efficiency, reducing computational costs while maintaining accuracy. ThinkCoder boosts the performance of multiple base LLMs, excelling on benchmarks like HumanEval and MBPP. Compared to SOTA models, it improves Pass@1 by 1.5\% over MapCoder with just 21.7\% of the computation cost. Against AgentCoder, ThinkCoder achieves a 0.6\% higher Pass@1 after 2 rounds, outperforming AgentCoder's 5 rounds. Additionally, ReST with success trajectories enhances efficiency, allowing models like LLaMA2-7B to achieve competitive results using only 20\% of the computational resources. These results highlight the framework's effectiveness and scalability. 

**Abstract (ZH)**: 代码生成是软件工程中提高代码编写效率的关键。尽管测试时的计算方法显示出潜力，但由于多轮计算导致的高延迟成为其主要限制。为克服这一问题，我们提出了ThinkCoder框架，该框架结合了全面的探索和最优的提炼。探索阶段通过搜索潜在解决方案来多样化解决方案空间，随后的提炼阶段则提高精度。这种方法允许我们在采取行动之前仔细考虑，从而避免不必要的试错。为了进一步减少测试时的计算开销，引入了基于偏好的优化方法Reinforced Self-Training（ReST），该方法利用ThinkCoder的探索轨迹来引导LLM的演化。通过学习偏好，这种方法提高了LLM的探索效率，同时降低了计算成本并保持了准确性。ThinkCoder提升了多个基模型的表现，其在HumanEval和MBPP等基准测试上表现出色。与最新模型相比，它在与MapCoder对比时，能耗仅占MapCoder的21.7%，但Pass@1表现提高了1.5%。与AgentCoder相比，在2轮测试后，ThinkCoder的Pass@1提高了0.6%，而AgentCoder则需要5轮。此外，使用成功轨迹的ReST方法提高了效率，使得如LLaMA2-7B这样的模型仅使用20%的计算资源即可获得竞争力的结果。这些结果突显了该框架的有效性和可扩展性。 

---
# Large Language Models as Realistic Microservice Trace Generators 

**Title (ZH)**: 大型语言模型作为现实的微服务跟踪生成器 

**Authors**: Donghyun Kim, Sriram Ravula, Taemin Ha, Alexandros G. Dimakis, Daehyeok Kim, Aditya Akella  

**Link**: [PDF](https://arxiv.org/pdf/2502.17439)  

**Abstract**: Computer system workload traces, which record hardware or software events during application execution, are essential for understanding the behavior of complex systems and managing their processing and memory resources. However, obtaining real-world traces can be challenging due to the significant collection overheads in performance and privacy concerns that arise in proprietary systems. As a result, synthetic trace generation is considered a promising alternative to using traces collected in real-world production deployments. This paper proposes to train a large language model (LLM) to generate synthetic workload traces, specifically microservice call graphs. To capture complex and arbitrary hierarchical structures and implicit constraints in such traces, we fine-tune LLMs to generate each layer recursively, making call graph generation a sequence of easier steps. To further enforce learning constraints in traces and generate uncommon situations, we apply additional instruction tuning steps to align our model with the desired trace features. Our evaluation results show that our model can generate diverse realistic traces under various conditions and outperform existing methods in accuracy and validity. We show that our synthetically generated traces can effectively substitute real-world data in optimizing or tuning systems management tasks. We also show that our model can be adapted to perform key downstream trace-related tasks, specifically, predicting key trace features and infilling missing data given partial traces. Codes are available in this https URL. 

**Abstract (ZH)**: 计算机系统负载迹像是记录应用程序执行期间的硬件或软件事件，对于理解复杂系统的行为以及管理其处理和内存资源至关重要。然而，由于在性能和隐私方面存在显著的采集开销，特别是在专有系统中，获得真实世界的迹像是一个挑战。因此，合成迹像是使用在实际生产部署中收集的真实世界迹像是一个有希望的替代方法。本文提出了一种基于大型语言模型（LLM）生成合成负载迹线的方法，特别是微服务调用图。为了捕捉此类迹线中复杂的和任意的分层结构以及隐含的约束，我们通过递归地对每个层次进行微调，将调用图生成分解为一系列较易处理的步骤。为了进一步加强迹线中的学习约束并生成不常见的情况，我们应用附加的指令微调步骤，以使模型与所需的迹线特征一致。我们的评估结果表明，我们的模型在各种条件下能够生成多样化的现实迹线，并且在准确性和有效性方面均优于现有方法。我们展示了通过合成生成的迹线可以有效地替代实际数据以优化或调整系统管理任务。此外，我们展示了我们的模型可以适应执行关键的下游迹线相关任务，特别是预测关键迹线特征以及在部分迹线情况下填补缺失的数据。代码可在以下链接获取：[在此处插入链接]。 

---
