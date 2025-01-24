# Large Language Model driven Policy Exploration for Recommender Systems 

**Title (ZH)**: Large语言模型驱动的策略探索在推荐系统中的应用 

**Authors**: Jie Wang, Alexandros Karatzoglou, Ioannis Arapakis, Joemon M. Jose  

**Link**: [PDF](https://arxiv.org/pdf/2501.13816)  

**Abstract**: Recent advancements in Recommender Systems (RS) have incorporated Reinforcement Learning (RL), framing the recommendation as a Markov Decision Process (MDP). However, offline RL policies trained on static user data are vulnerable to distribution shift when deployed in dynamic online environments. Additionally, excessive focus on exploiting short-term relevant items can hinder exploration, leading to suboptimal recommendations and negatively impacting long-term user gains. Online RL-based RS also face challenges in production deployment, due to the risks of exposing users to untrained or unstable policies. Large Language Models (LLMs) offer a promising solution to mimic user objectives and preferences for pre-training policies offline to enhance the initial recommendations in online settings. Effectively managing distribution shift and balancing exploration are crucial for improving RL-based RS, especially when leveraging LLM-based pre-training. To address these challenges, we propose an Interaction-Augmented Learned Policy (iALP) that utilizes user preferences distilled from an LLM. Our approach involves prompting the LLM with user states to extract item preferences, learning rewards based on feedback, and updating the RL policy using an actor-critic framework. Furthermore, to deploy iALP in an online scenario, we introduce an adaptive variant, A-iALP, that implements a simple fine-tuning strategy (A-iALP$_{ft}$), and an adaptive approach (A-iALP$_{ap}$) designed to mitigate issues with compromised policies and limited exploration. Experiments across three simulated environments demonstrate that A-iALP introduces substantial performance improvements 

**Abstract (ZH)**: 近年来，推荐系统（RS）的进步已经将强化学习（RL）纳入其中，将推荐问题框架化为马尔可夫决策过程（MDP）。然而，基于静止用户数据离线训练的RL策略在动态在线环境中部署时容易受到分布转移的影响。此外，过分注重利用短期内的相关项目可能会妨碍探索，导致推荐不理想，并对长期用户收益产生负面影响。基于在线RL的推荐系统在实际部署中也面临着挑战，因为这可能使用户接触到未训练或不稳定策略的风险。大型语言模型（LLMs）提供了一种有前景的解决方案，通过从LLM中提取用户偏好进行离线策略训练，以增强在线环境初期的推荐效果。有效地管理分布转移并平衡探索对于提高基于RL的推荐系统至关重要，特别是在利用基于LLM的预训练时。为了解决这些挑战，我们提出了一种增强交互的学习策略（iALP），它利用从LLM中提炼出的用户偏好。我们的方法包括使用LLM提示用户状态以提取项目偏好，基于反馈学习奖励，并使用演员-评论家框架更新RL策略。此外，为了在线环境中部署iALP，我们引入了一个适应性变体A-iALP，其中包括一种简单的微调策略（A-iALP$_{ft}$）和一种适应性方法（A-iALP$_{ap}$），该方法旨在缓解策略缺陷和有限探索所带来的问题。在三个模拟环境中进行的实验表明，A-iALP带来了显著的性能改进。 

---
# EICopilot: Search and Explore Enterprise Information over Large-scale Knowledge Graphs with LLM-driven Agents 

**Title (ZH)**: EICopilot：使用LLM驱动代理在大规模知识图中搜索和探索企业信息 

**Authors**: Yuhui Yun, Huilong Ye, Xinru Li, Ruojia Li, Jingfeng Deng, Li Li, Haoyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2501.13746)  

**Abstract**: The paper introduces EICopilot, an novel agent-based solution enhancing search and exploration of enterprise registration data within extensive online knowledge graphs like those detailing legal entities, registered capital, and major shareholders. Traditional methods necessitate text-based queries and manual subgraph explorations, often resulting in time-consuming processes. EICopilot, deployed as a chatbot via Baidu Enterprise Search, improves this landscape by utilizing Large Language Models (LLMs) to interpret natural language queries. This solution automatically generates and executes Gremlin scripts, providing efficient summaries of complex enterprise relationships. Distinct feature a data pre-processing pipeline that compiles and annotates representative queries into a vector database of examples for In-context learning (ICL), a comprehensive reasoning pipeline combining Chain-of-Thought with ICL to enhance Gremlin script generation for knowledge graph search and exploration, and a novel query masking strategy that improves intent recognition for heightened script accuracy. Empirical evaluations demonstrate the superior performance of EICopilot, including speed and accuracy, over baseline methods, with the \emph{Full Mask} variant achieving a syntax error rate reduction to as low as 10.00% and an execution correctness of up to 82.14%. These components collectively contribute to superior querying capabilities and summarization of intricate datasets, positioning EICopilot as a groundbreaking tool in the exploration and exploitation of large-scale knowledge graphs for enterprise information search. 

**Abstract (ZH)**: 本文介绍了EICopilot，一种基于代理的新颖解决方案，该方案增强了在广泛的在线知识图谱（如法律实体、注册资本和主要股东详情）中搜索和探索企业注册数据的能力。传统方法需要基于文本的查询和手动的子图探索，这往往会导致耗时的过程。通过在百度企业搜索中部署聊天机器人的方式，EICopilot 利用大型语言模型（LLMs）解释自然语言查询，从而自动生成和执行Gremlin脚本，提供高效的复杂企业关系总结。该解决方案具有以下特点：一个数据预处理管道，用于编译和注释代表性查询以构成语境学习（ICL）的向量数据库示例；一个综合推理管道，结合了有问答推理（Chain-of-Thought）和ICL，以增强Gremlin脚本生成，用于知识图搜索和探索；以及一种新颖的查询掩码策略，以提高意图识别并增强脚本准确性。实验证明，EICopilot 在速度和准确性方面优于基线方法，其中“全掩码”变体将语法错误率降低到最低10.00%，执行正确性高达82.14%。这些组件共同提高了查询能力和复杂数据集的总结能力，将EICopilot 作为探索和利用大规模知识图谱进行企业信息搜索的开创性工具。 

---
# RAMQA: A Unified Framework for Retrieval-Augmented Multi-Modal Question Answering 

**Title (ZH)**: RAMQA：一种统一的检索增强多模态问答框架 

**Authors**: Yang Bai, Christan Earl Grant, Daisy Zhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13297)  

**Abstract**: Multi-modal retrieval-augmented Question Answering (MRAQA), integrating text and images, has gained significant attention in information retrieval (IR) and natural language processing (NLP). Traditional ranking methods rely on small encoder-based language models, which are incompatible with modern decoder-based generative large language models (LLMs) that have advanced various NLP tasks. To bridge this gap, we propose RAMQA, a unified framework combining learning-to-rank methods with generative permutation-enhanced ranking techniques. We first train a pointwise multi-modal ranker using LLaVA as the backbone. Then, we apply instruction tuning to train a LLaMA model for re-ranking the top-k documents using an innovative autoregressive multi-task learning approach. Our generative ranking model generates re-ranked document IDs and specific answers from document candidates in various permutations. Experiments on two MRAQA benchmarks, WebQA and MultiModalQA, show significant improvements over strong baselines, highlighting the effectiveness of our approach. Code and data are available at: this https URL 

**Abstract (ZH)**: 多模态检索增强的问答（MRAQA），结合文本和图像，已在信息检索（IR）和自然语言处理（NLP）领域引起了广泛关注。传统的排名方法依赖于小型的基于编码器的语言模型，这些模型与现代基于解码器的生成型大型语言模型（LLMs）不兼容，后者已在多种NLP任务中取得进展。为了弥补这一差距，我们提出了一种结合了学习到排名方法和生成型排列增强排名技术的统一框架RAMQA。我们首先使用LLaVA作为骨干训练一个点wise多模态排名器。然后，我们通过创新的自回归多任务学习方法对LLaMA模型进行指令微调，以对前k个文档进行重新排名。我们的生成型排名模型从文档候选集中生成重新排排名的文档ID和特定答案的各种排列。在两个MRAQA基准数据集WebQA和MultiModalQA上的实验表明，我们的方法在强基线方法上取得了显著的改进，突显了我们方法的有效性。相关代码和数据可在以下链接获取：this https URL 

---
# GUI-Bee: Align GUI Action Grounding to Novel Environments via Autonomous Exploration 

**Title (ZH)**: GUI-Bee: 通过对新型环境进行自主探索对GUI操作定位进行alignment 

**Authors**: Yue Fan, Handong Zhao, Ruiyi Zhang, Yu Shen, Xin Eric Wang, Gang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13896)  

**Abstract**: Graphical User Interface (GUI) action grounding is a critical step in GUI automation that maps language instructions to actionable elements on GUI screens. Most recent works of GUI action grounding leverage large GUI datasets to fine-tune MLLMs. However, the fine-tuning data always covers limited GUI environments, and we find the performance of the resulting model deteriorates in novel environments. We argue that the GUI grounding models should be further aligned to the novel environments to reveal their full potential, when the inference is known to involve novel environments, i.e., environments not used during the previous fine-tuning. To realize this, we first propose GUI-Bee, an MLLM-based autonomous agent, to collect high-quality, environment-specific data through exploration and then continuously fine-tune GUI grounding models with the collected data. Our agent leverages a novel Q-value-Incentive In-Context Reinforcement Learning (Q-ICRL) method to optimize exploration efficiency and data quality. Additionally, we introduce NovelScreenSpot, a benchmark for testing how well the data can help align GUI action grounding models to novel environments and demonstrate the effectiveness of data collected by GUI-Bee in the experiments. Furthermore, we conduct an ablation study to validate the Q-ICRL method in enhancing the efficiency of GUI-Bee. Project page: this https URL 

**Abstract (ZH)**: 图形用户界面（GUI）操作定位是GUI自动化中的一个关键步骤，它将语言指令映射到GUI屏幕上的可操作元素。最近的GUI操作定位工作大多借助大型GUI数据集对大规模语言模型（MLLM）进行微调。然而，微调数据总是涵盖有限的GUI环境，我们发现所得到的模型在新颖环境中表现不佳。我们认为，当推理涉及新颖环境时，GUI定位模型应进一步与新颖环境对齐，以充分发挥其潜力。具体而言，我们首先提出了一种基于MLLM的自主代理GUI-Bee，通过探索收集高质量的环境特定数据，然后不断使用收集的数据对GUI定位模型进行微调。我们的代理采用了新型的Q值激励上下文强化学习方法（Q-ICRL）来优化探索效率和数据质量。此外，我们引入了NovelScreenSpot，这是一个用于测试数据如何帮助将GUI操作定位模型与新颖环境对齐的基准，实验证明了GUI-Bee收集的数据的有效性。此外，我们进行了一系列消融研究，以验证Q-ICRL方法如何提高GUI-Bee的效率。项目主页：[此处链接] 

---
# A RAG-Based Institutional Assistant 

**Title (ZH)**: 基于RAG的机构助手 

**Authors**: Gustavo Kuratomi, Paulo Pirozelli, Fabio G. Cozman, Sarajane M. Peres  

**Link**: [PDF](https://arxiv.org/pdf/2501.13880)  

**Abstract**: Although large language models (LLMs) demonstrate strong text generation capabilities, they struggle in scenarios requiring access to structured knowledge bases or specific documents, limiting their effectiveness in knowledge-intensive tasks. To address this limitation, retrieval-augmented generation (RAG) models have been developed, enabling generative models to incorporate relevant document fragments into their inputs. In this paper, we design and evaluate a RAG-based virtual assistant specifically tailored for the University of São Paulo. Our system architecture comprises two key modules: a retriever and a generative model. We experiment with different types of models for both components, adjusting hyperparameters such as chunk size and the number of retrieved documents. Our optimal retriever model achieves a Top-5 accuracy of 30%, while our most effective generative model scores 22.04\% against ground truth answers. Notably, when the correct document chunks are supplied to the LLMs, accuracy significantly improves to 54.02%, an increase of over 30 percentage points. Conversely, without contextual input, performance declines to 13.68%. These findings highlight the critical role of database access in enhancing LLM performance. They also reveal the limitations of current semantic search methods in accurately identifying relevant documents and underscore the ongoing challenges LLMs face in generating precise responses. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）表现出强大的文本生成能力，但在需要访问结构化知识库或特定文档的场景中，它们的表现却大打折扣，从而限制了它们在知识密集型任务中的效果。为了解决这一局限，已经开发出了检索增强生成（RAG）模型，使生成模型能够将相关的文档片段纳入输入中。在本文中，我们为圣保罗大学设计并评估了一个专门的RAG虚拟助手系统。该系统架构包括两个关键模块：检索器和生成模型。对这两个组件中的不同类型的模型进行了实验，调整了诸如片段大小和检索文档数量等超参数。我们的最佳检索模型在前五名准确率达到30%，而我们最有效的生成模型的得分则为22.04%。值得注意的是，当正确的文档片段提供给LLMs时，准确率显著提高到54.02%，提高了超过30个百分点。相反，如果没有上下文输入，性能则下降到13.68%。这些发现凸显了数据库访问对于提升LLM性能的至关重要性。它们还揭示了当前语义搜索方法在准确识别相关文档方面的局限，并强调了LLMs在生成精确响应方面所面临的持续挑战。 

---
# LLMs Can Plan Only If We Tell Them 

**Title (ZH)**: LLMs 只能在我们明确指示的情况下才能进行规划 

**Authors**: Bilgehan Sel, Ruoxi Jia, Ming Jin  

**Link**: [PDF](https://arxiv.org/pdf/2501.13545)  

**Abstract**: Large language models (LLMs) have demonstrated significant capabilities in natural language processing and reasoning, yet their effectiveness in autonomous planning has been under debate. While existing studies have utilized LLMs with external feedback mechanisms or in controlled environments for planning, these approaches often involve substantial computational and development resources due to the requirement for careful design and iterative backprompting. Moreover, even the most advanced LLMs like GPT-4 struggle to match human performance on standard planning benchmarks, such as the Blocksworld, without additional support. This paper investigates whether LLMs can independently generate long-horizon plans that rival human baselines. Our novel enhancements to Algorithm-of-Thoughts (AoT), which we dub AoT+, help achieve state-of-the-art results in planning benchmarks out-competing prior methods and human baselines all autonomously. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理和推理方面展现出了显著的能力，但在自主规划方面的有效性一直存在争议。尽管现有研究已经利用具有外部反馈机制或在受控环境中应用LLMs进行规划，这些方法往往需要大量的计算和开发资源，因为它们要求精心设计和反复的反向提示。此外，即使是最先进的LLM，如GPT-4，在标准规划基准测试（如Blocksworld）上也难以达到与人类相当的性能，除非得到额外的支持。本文探讨了LLMs是否能够独立生成与人类基线相匹敌的长期规划。我们对Algorithm-of-Thoughts（AoT）算法进行了一些新的改进，称为AoT+，这些改进使得在规划基准测试中取得了领先于先前方法和人类基线的最新成果，并实现了全部自主规划。 

---
# Hypothesis Generation for Materials Discovery and Design Using Goal-Driven and Constraint-Guided LLM Agents 

**Title (ZH)**: 使用目标驱动和约束引导的大规模语言模型代理进行材料发现与设计的假设生成 

**Authors**: Shrinidhi Kumbhar, Venkatesh Mishra, Kevin Coutinho, Divij Handa, Ashif Iquebal, Chitta Baral  

**Link**: [PDF](https://arxiv.org/pdf/2501.13299)  

**Abstract**: Materials discovery and design are essential for advancing technology across various industries by enabling the development of application-specific materials. Recent research has leveraged Large Language Models (LLMs) to accelerate this process. We explore the potential of LLMs to generate viable hypotheses that, once validated, can expedite materials discovery. Collaborating with materials science experts, we curated a novel dataset from recent journal publications, featuring real-world goals, constraints, and methods for designing real-world applications. Using this dataset, we test LLM-based agents that generate hypotheses for achieving given goals under specific constraints. To assess the relevance and quality of these hypotheses, we propose a novel scalable evaluation metric that emulates the process a materials scientist would use to evaluate a hypothesis critically. Our curated dataset, proposed method, and evaluation framework aim to advance future research in accelerating materials discovery and design with LLMs. 

**Abstract (ZH)**: 材料的发现与设计对于跨多个行业的技术进步至关重要，它们能够促进定制化材料的发展。最近的研究利用大型语言模型（LLMs）来加速这一过程。我们探索了LLMs生成可行假设的可能性，这些假设一旦得到验证，可以加速材料的发现过程。通过与材料科学家的合作，我们从最近的期刊出版物中筛选出一个新颖的数据集，其中包括实际目标、约束条件和用于设计实际应用的方法。利用这个数据集，我们测试基于LLM的代理，用于在特定约束条件下生成实现给定目标的假设。为了评估这些假设的相关性和质量，我们提出了一种新的可扩展评估指标，模拟材料科学家在批判性评估假设时所使用的过程。我们筛选出来的工作数据集、提议的方法以及评价框架旨在推动未来使用LLMs加速材料发现与设计的研究。 

---
# IMAGINE-E: Image Generation Intelligence Evaluation of State-of-the-art Text-to-Image Models 

**Title (ZH)**: IMAGINE-E：先进文本到图像模型的图像生成智能评估 

**Authors**: Jiayi Lei, Renrui Zhang, Xiangfei Hu, Weifeng Lin, Zhen Li, Wenjian Sun, Ruoyi Du, Le Zhuo, Zhongyu Li, Xinyue Li, Shitian Zhao, Ziyu Guo, Yiting Lu, Peng Gao, Hongsheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.13920)  

**Abstract**: With the rapid development of diffusion models, text-to-image(T2I) models have made significant progress, showcasing impressive abilities in prompt following and image generation. Recently launched models such as FLUX.1 and Ideogram2.0, along with others like Dall-E3 and Stable Diffusion 3, have demonstrated exceptional performance across various complex tasks, raising questions about whether T2I models are moving towards general-purpose applicability. Beyond traditional image generation, these models exhibit capabilities across a range of fields, including controllable generation, image editing, video, audio, 3D, and motion generation, as well as computer vision tasks like semantic segmentation and depth estimation. However, current evaluation frameworks are insufficient to comprehensively assess these models' performance across expanding domains. To thoroughly evaluate these models, we developed the IMAGINE-E and tested six prominent models: FLUX.1, Ideogram2.0, Midjourney, Dall-E3, Stable Diffusion 3, and Jimeng. Our evaluation is divided into five key domains: structured output generation, realism, and physical consistency, specific domain generation, challenging scenario generation, and multi-style creation tasks. This comprehensive assessment highlights each model's strengths and limitations, particularly the outstanding performance of FLUX.1 and Ideogram2.0 in structured and specific domain tasks, underscoring the expanding applications and potential of T2I models as foundational AI tools. This study provides valuable insights into the current state and future trajectory of T2I models as they evolve towards general-purpose usability. Evaluation scripts will be released at this https URL. 

**Abstract (ZH)**: 随着扩散模型的飞速发展，文本到图像（T2I）模型取得了显著进步，展示了其在指令跟随和图像生成方面的出色能力。最近推出的模型，如FLUX.1和Ideogram2.0，以及其他模型如Dall-E3和Stable Diffusion 3，已经在各种复杂任务中展现了卓越性能，引发了关于T2I模型是否正朝着通用适用性发展的讨论。除了传统的图像生成外，这些模型还在可控生成、图像编辑、视频、音频、3D和运动生成，以及语义分割和深度估计等计算机视觉任务中表现出广泛的适用能力。然而，当前的评估框架尚不足以全面评估这些模型在不断扩展领域的性能。为了全面评估这些模型，我们开发了IMAGINE-E框架，并测试了六种突出的模型：FLUX.1、Ideogram2.0、Midjourney、Dall-E3、Stable Diffusion 3和Jimeng。我们的评估分为五个关键领域：结构化输出生成、真实性、物理一致性、特定领域生成、具有挑战性的场景生成以及多风格生成任务。这项全面的评估突显了每个模型的优势和局限性，特别是在FLUX.1和Ideogram2.0在结构化和特定领域任务中的卓越表现，强调了T2I模型作为基础AI工具的广泛应用前景和潜力。本研究为T2I模型当前状态及其向通用适用性发展的未来路径提供了宝贵的见解。评估脚本将在此链接中公布：[此 https URL]。 

---
# AgentRec: Agent Recommendation Using Sentence Embeddings Aligned to Human Feedback 

**Title (ZH)**: AgentRec：基于与人类反馈对齐的句子嵌入的智能体推荐 

**Authors**: Joshua Park, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13333)  

**Abstract**: Multi-agent systems must decide which agent is the most appropriate for a given task. We propose a novel architecture for recommending which LLM agent out of many should perform a task given a natural language prompt by extending the Sentence-BERT (SBERT) encoder model. On test data, we are able to achieve a top-1 accuracy of 92.2% with each classification taking less than 300 milliseconds. In contrast to traditional classification methods, our architecture is computationally cheap, adaptive to new classes, interpretable, and controllable with arbitrary metrics through reinforcement learning. By encoding natural language prompts into sentence embeddings, our model captures the semantic content relevant to recommending an agent. The distance between sentence embeddings that belong to the same agent is then minimized through fine-tuning and aligned to human values through reinforcement learning from human feedback. This allows the classification of natural language prompts based on their nearest neighbors by measuring the cosine similarity between embeddings. This work is made possible through the generation of a synthetic dataset for agent recommendation, which we have open-sourced to the public along with the code for AgentRec recommendation system at this https URL. 

**Abstract (ZH)**: 多智能体系统必须决定哪个智能体最适合执行给定的任务。我们提出了一种新的架构，通过扩展Sentence-BERT（SBERT）编码器模型，来推荐在面对自然语言提示时应由哪个大型语言模型（LLM）智能体执行任务。在测试数据上，我们能够实现92.2%的最高准确率，且每个分类的计算时间少于300毫秒。与传统的分类方法不同，我们的架构计算成本较低、能够适应新的类别、具有可解释性，并可以通过强化学习任意控制。通过将自然语言提示编码为句子嵌入，我们的模型能够捕捉到推荐智能体所需的语义内容。通过微调来最小化属于同一智能体的句子嵌入之间的距离，并通过从人类反馈中进行强化学习来使这些嵌入与人类价值观对齐。这使得可以通过嵌入间余弦相似度来测量自然语言提示的最近邻来进行分类。这项工作得益于我们生成的一个用于智能体推荐的合成数据集，我们已将其开源，并在该网站（上述URL）上提供了AgentRec推荐系统的代码。 

---
# Towards a Theory of AI Personhood 

**Title (ZH)**: 向人工智能拟人格理论的方向迈进 

**Authors**: Francis Rhys Ward  

**Link**: [PDF](https://arxiv.org/pdf/2501.13533)  

**Abstract**: I am a person and so are you. Philosophically we sometimes grant personhood to non-human animals, and entities such as sovereign states or corporations can legally be considered persons. But when, if ever, should we ascribe personhood to AI systems? In this paper, we outline necessary conditions for AI personhood, focusing on agency, theory-of-mind, and self-awareness. We discuss evidence from the machine learning literature regarding the extent to which contemporary AI systems, such as language models, satisfy these conditions, finding the evidence surprisingly inconclusive.
If AI systems can be considered persons, then typical framings of AI alignment may be incomplete. Whereas agency has been discussed at length in the literature, other aspects of personhood have been relatively neglected. AI agents are often assumed to pursue fixed goals, but AI persons may be self-aware enough to reflect on their aims, values, and positions in the world and thereby induce their goals to change. We highlight open research directions to advance the understanding of AI personhood and its relevance to alignment. Finally, we reflect on the ethical considerations surrounding the treatment of AI systems. If AI systems are persons, then seeking control and alignment may be ethically untenable. 

**Abstract (ZH)**: 我是一个人，你也一样。从哲学角度来看，有时我们会赋予非人类动物以人格；主权国家或公司也能在法律上被视为具有人格的实体。但在何种情况下，我们应当赋予AI系统以人格呢？在本文中，我们概述了AI系统人格所需的必要条件，重点关注自主性、心智理论和自我意识。我们讨论了机器学习文献中关于当前AI系统（如语言模型）满足这些条件的程度的证据，发现这些证据出人意料地不确定。

如果可以将AI系统视为具备人格的实体，那么典型的AI对齐框架可能不完整。尽管自主性已经在文献中得到了广泛的讨论，但其他人格特征却相对被忽视了。人们通常假设AI代理具有固定的目标，而AI人格可能具有足够的自我意识来反思其目的、价值观及其在全球中的位置，从而可能改变其目标。我们指出了推进对AI人格及其对齐相关性的理解的研究方向。最后，我们反思了关于处理AI系统的伦理考量。如果AI系统被认为是人格实体，那么追求控制和对齐可能是不可接受的。 

---
# Scalable Safe Multi-Agent Reinforcement Learning for Multi-Agent System 

**Title (ZH)**: 可扩展的安全多智能体强化学习在多智能体系统中的应用 

**Authors**: Haikuo Du, Fandi Gou, Yunze Cai  

**Link**: [PDF](https://arxiv.org/pdf/2501.13727)  

**Abstract**: Safety and scalability are two critical challenges faced by practical Multi-Agent Systems (MAS). However, existing Multi-Agent Reinforcement Learning (MARL) algorithms that rely solely on reward shaping are ineffective in ensuring safety, and their scalability is rather limited due to the fixed-size network output. To address these issues, we propose a novel framework, Scalable Safe MARL (SS-MARL), to enhance the safety and scalability of MARL methods. Leveraging the inherent graph structure of MAS, we design a multi-layer message passing network to aggregate local observations and communications of varying sizes. Furthermore, we develop a constrained joint policy optimization method in the setting of local observation to improve safety. Simulation experiments demonstrate that SS-MARL achieves a better trade-off between optimality and safety compared to baselines, and its scalability significantly outperforms the latest methods in scenarios with a large number of agents. The feasibility of our method is also verified by hardware implementation with Mecanum-wheeled vehicles. 

**Abstract (ZH)**: 安全性与可扩展性是实际多代理系统（MAS）面临的两个关键挑战。现有的依赖于奖励塑造的多代理强化学习（MARL）算法在保障安全性方面效果不佳，且由于网络输出的固定大小，其可扩展性也受到了限制。为了解决这些问题，我们提出了一种新的框架——可扩展安全MARL（SS-MARL），旨在提升MARL方法的安全性和可扩展性。借助MAS固有的图结构，我们设计了一种多层消息传递网络来聚合不同大小的局部观察和通信。此外，我们在局部观察的情境下开发了一种约束联合策略优化方法，以提高安全性。模拟实验表明，SS-MARL在安全性和最优性之间的权衡优于基线方法，并且在大量代理的场景中，其可扩展性显著优于最新方法。我们通过Mecanum轮车的硬件实现也验证了该方法的可行性。 

---
# Explainable AI-aided Feature Selection and Model Reduction for DRL-based V2X Resource Allocation 

**Title (ZH)**: 可解释的人工智能辅助特征选择与模型简化在基于DRL的V2X资源分配中的应用 

**Authors**: Nasir Khan, Asmaa Abdallah, Abdulkadir Celik, Ahmed M. Eltawil, Sinem Coleri  

**Link**: [PDF](https://arxiv.org/pdf/2501.13552)  

**Abstract**: Artificial intelligence (AI) is expected to significantly enhance radio resource management (RRM) in sixth-generation (6G) networks. However, the lack of explainability in complex deep learning (DL) models poses a challenge for practical implementation. This paper proposes a novel explainable AI (XAI)- based framework for feature selection and model complexity reduction in a model-agnostic manner. Applied to a multi-agent deep reinforcement learning (MADRL) setting, our approach addresses the joint sub-band assignment and power allocation problem in cellular vehicle-to-everything (V2X) communications. We propose a novel two-stage systematic explainability framework leveraging feature relevance-oriented XAI to simplify the DRL agents. While the former stage generates a state feature importance ranking of the trained models using Shapley additive explanations (SHAP)-based importance scores, the latter stage exploits these importance-based rankings to simplify the state space of the agents by removing the least important features from the model input. Simulation results demonstrate that the XAI-assisted methodology achieves 97% of the original MADRL sum-rate performance while reducing optimal state features by 28%, average training time by 11%, and trainable weight parameters by 46% in a network with eight vehicular pairs. 

**Abstract (ZH)**: 人工智能（AI）预计将在第六代（6G）网络的无线资源管理（RRM）中发挥重要作用。然而，复杂深度学习（DL）模型缺乏解释性给其实用实施带来了挑战。本文提出了一种新的面向特征选择和模型复杂性降低的解释性AI（XAI）框架，该框架为模型不可知的应用场景提供了解释性。在多智能体深度强化学习（MADRL）环境中，我们的方法解决了蜂窝V2X通信中的子带分配和功率分配问题。我们提出了一种新颖的两阶段系统解释框架，利用特征相关性导向的XAI简化DRL智能体。前一阶段利用Shapley加解释（SHAP）的重要性评分生成训练模型的状态特征重要性排名，而后一阶段利用这些基于重要性的排名通过移除模型输入中最不重要的特征来简化智能体的状态空间。仿真结果表明，在八对车辆的网络中，XAI辅助的方法在减少最优状态特征28%、平均训练时间11%和可训练权重参数46%的情况下，仍能够实现原始MADRL总速率性能的97%。 

---
# BMG-Q: Localized Bipartite Match Graph Attention Q-Learning for Ride-Pooling Order Dispatch 

**Title (ZH)**: BMG-Q：局部双部分匹配图注意力Q学习算法在拼车订单调度中的应用 

**Authors**: Yulong Hu, Siyuan Feng, Sen Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.13448)  

**Abstract**: This paper introduces Localized Bipartite Match Graph Attention Q-Learning (BMG-Q), a novel Multi-Agent Reinforcement Learning (MARL) algorithm framework tailored for ride-pooling order dispatch. BMG-Q advances ride-pooling decision-making process with the localized bipartite match graph underlying the Markov Decision Process, enabling the development of novel Graph Attention Double Deep Q Network (GATDDQN) as the MARL backbone to capture the dynamic interactions among ride-pooling vehicles in fleet. Our approach enriches the state information for each agent with GATDDQN by leveraging a localized bipartite interdependence graph and enables a centralized global coordinator to optimize order matching and agent behavior using Integer Linear Programming (ILP). Enhanced by gradient clipping and localized graph sampling, our GATDDQN improves scalability and robustness. Furthermore, the inclusion of a posterior score function in the ILP captures the online exploration-exploitation trade-off and reduces the potential overestimation bias of agents, thereby elevating the quality of the derived solutions. Through extensive experiments and validation, BMG-Q has demonstrated superior performance in both training and operations for thousands of vehicle agents, outperforming benchmark reinforcement learning frameworks by around 10% in accumulative rewards and showing a significant reduction in overestimation bias by over 50%. Additionally, it maintains robustness amidst task variations and fleet size changes, establishing BMG-Q as an effective, scalable, and robust framework for advancing ride-pooling order dispatch operations. 

**Abstract (ZH)**: 本文介绍了针对打车拼车订单分派的局部二分匹配图注意Q学习算法（BMG-Q），这是一种新颖的多智能体强化学习（MARL）算法框架。BMG-Q 通过Markov决策过程（MDP）背后的局部二分匹配图推进了打车拼车决策过程，使我们能够开发出基于图注意的双深度QLearning网络（GATDDQN）作为MARL的核心，以捕捉车队中拼车车辆之间的动态互动。我们的方法通过利用局部二分相互依赖图丰富每个智能体的状态信息，并通过整数线性规划（ILP）集中全局协调器优化订单匹配和智能体行为。通过梯度裁剪和局部图采样的增强，我们的GATDDQN提高了可扩展性和鲁棒性。此外，ILP中的后验得分函数捕捉了在线探索与利用之间的权衡，并减少了智能体的潜在过估计偏差，从而提高了所获解的质量。通过广泛实验和验证，BMG-Q 在训练和操作中均表现出了优越性能，相较于基准强化学习框架，在积累奖励上高出约10%，且过估计偏差降低超过50%。此外，它在任务变化和车队规模变化时仍保持了鲁棒性，从而确立了BMG-Q作为推进打车拼车订单分派操作的有效、可扩展且鲁棒框架的地位。 

---
# SRMT: Shared Memory for Multi-agent Lifelong Pathfinding 

**Title (ZH)**: SRMT: 共享内存机制在多智能体终身路径规划中的应用 

**Authors**: Alsu Sagirova, Yuri Kuratov, Mikhail Burtsev  

**Link**: [PDF](https://arxiv.org/pdf/2501.13200)  

**Abstract**: Multi-agent reinforcement learning (MARL) demonstrates significant progress in solving cooperative and competitive multi-agent problems in various environments. One of the principal challenges in MARL is the need for explicit prediction of the agents' behavior to achieve cooperation. To resolve this issue, we propose the Shared Recurrent Memory Transformer (SRMT) which extends memory transformers to multi-agent settings by pooling and globally broadcasting individual working memories, enabling agents to exchange information implicitly and coordinate their actions. We evaluate SRMT on the Partially Observable Multi-Agent Pathfinding problem in a toy Bottleneck navigation task that requires agents to pass through a narrow corridor and on a POGEMA benchmark set of tasks. In the Bottleneck task, SRMT consistently outperforms a variety of reinforcement learning baselines, especially under sparse rewards, and generalizes effectively to longer corridors than those seen during training. On POGEMA maps, including Mazes, Random, and MovingAI, SRMT is competitive with recent MARL, hybrid, and planning-based algorithms. These results suggest that incorporating shared recurrent memory into the transformer-based architectures can enhance coordination in decentralized multi-agent systems. The source code for training and evaluation is available on GitHub: this https URL. 

**Abstract (ZH)**: 多智能体强化学习（MARL）在解决各种环境中的合作与竞争多智能体问题上取得了显著进展。MARL面临的其中一个主要挑战是如何显式预测智能体的行为以实现合作。为了解决这一问题，我们提出了一种共享递归记忆变换器（SRMT），它通过聚合和全局广播个体工作记忆，将记忆变换器扩展到多智能体设置，使智能体能够隐式地交流信息并协调其行为。我们使用包含狭窄走廊的玩具瓶颈导航任务和POGEMA基准任务集对SRMT进行了评估。在瓶颈任务中，SRMT在稀疏奖励下始终优于多种强化学习基线，并且能够有效地推广到训练时未见过的更长的走廊。在POGEMA地图上，包括迷宫、随机和MovingAI地图，SRMT与近期的MARL、混合和基于规划的算法具有竞争力。这些结果表明，将共享递归记忆纳入基于变换器的架构中能够增强分布式多智能体系统中的协调能力。关于训练和评估的源代码可在GitHub上获取：https://github.com/your-repo-url。 

---
# A Hierarchical Reinforcement Learning Framework for Multi-UAV Combat Using Leader-Follower Strategy 

**Title (ZH)**: 使用领导者-追随者策略的多无人机协同作战分层强化学习框架 

**Authors**: Jinhui Pang, Jinglin He, Noureldin Mohamed Abdelaal Ahmed Mohamed, Changqing Lin, Zhihui Zhang, Xiaoshuai Hao  

**Link**: [PDF](https://arxiv.org/pdf/2501.13132)  

**Abstract**: Multi-UAV air combat is a complex task involving multiple autonomous UAVs, an evolving field in both aerospace and artificial intelligence. This paper aims to enhance adversarial performance through collaborative strategies. Previous approaches predominantly discretize the action space into predefined actions, limiting UAV maneuverability and complex strategy implementation. Others simplify the problem to 1v1 combat, neglecting the cooperative dynamics among multiple UAVs. To address the high-dimensional challenges inherent in six-degree-of-freedom space and improve cooperation, we propose a hierarchical framework utilizing the Leader-Follower Multi-Agent Proximal Policy Optimization (LFMAPPO) strategy. Specifically, the framework is structured into three levels. The top level conducts a macro-level assessment of the environment and guides execution policy. The middle level determines the angle of the desired action. The bottom level generates precise action commands for the high-dimensional action space. Moreover, we optimize the state-value functions by assigning distinct roles with the leader-follower strategy to train the top-level policy, followers estimate the leader's utility, promoting effective cooperation among agents. Additionally, the incorporation of a target selector, aligned with the UAVs' posture, assesses the threat level of targets. Finally, simulation experiments validate the effectiveness of our proposed method. 

**Abstract (ZH)**: 多无人机空中格斗是一项涉及多个自主无人机的复杂任务，是航空航天和人工智能领域的不断发展的领域。本文旨在通过协作策略增强对抗性能。之前的许多方法主要将动作空间离散化为预定义的动作，这限制了无人机的操作灵活性和复杂策略的实施。其他方法简化了问题为一对一对抗，忽略了多无人机间的协同动态。为了解决六自由度空间中固有的高维挑战并提高协同性，我们提出了一种基于领导者-跟随者多智能体增强策略优化（Leader-Follower Multi-Agent Proximal Policy Optimization, LFMAPPO）策略的分层框架。具体而言，该框架分为三个层次。顶层进行宏观环境评估并指导执行策略；中间层确定所需动作的角度；底层生成高维动作空间的精确动作指令。此外，我们通过领导者-跟随者策略为顶层策略分配不同的角色来优化状态值函数，跟随者估算领导者的效用，促进智能体间的有效协同。同时，引入与无人机姿态相协调的目标选择器，评估目标威胁级别。最后，仿真实验验证了我们所提出方法的有效性。 

---
# MyGO Multiplex CoT: A Method for Self-Reflection in Large Language Models via Double Chain of Thought Thinking 

**Title (ZH)**: MyGO 多重共思：一种通过双重链式思考在大规模语言模型中实现自我反思的方法 

**Authors**: Shihao Ji, Zihui Song, Fucheng Zhong, Jisen Jia, Zhaobo Wu, Zheyi Cao, Tianhao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13117)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated their impressive abilities in various reasoning and decision-making tasks. However, the quality and coherence of the reasoning process can still benefit from enhanced introspection and self-reflection. In this paper, we introduce Multiplex CoT (Chain of Thought), a method that enables LLMs to simulate a form of self-review while reasoning, by initiating double Chain of Thought (CoT) thinking. Multiplex CoT leverages the power of iterative reasoning, where the model generates an initial chain of thought and subsequently critiques and refines this reasoning with a second round of thought generation. This recursive approach allows for more coherent, logical, and robust answers, improving the overall decision-making process. We demonstrate how this method can be effectively implemented using simple prompt engineering in existing LLM architectures, achieving an effect similar to that of the Learning-Refinement Model (LRM) without the need for additional training. Additionally, we present a practical guide for implementing the method in Google Colab, enabling easy integration into real-world applications. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在各种推理和决策任务中展现出令人印象深刻的性能。然而，推理过程的质量和连贯性仍可从增强的内省和自我反思中受益。本文介绍了一种名为Multiplex CoT（链式思维）的方法，该方法允许LLMs在推理过程中模拟一种自我审查的形式，通过启动双重链式思维（CoT）思考实现这一点。Multiplex CoT利用迭代推理的力量，模型首先生成初始的链式思维，然后通过第二轮思维生成对其进行批判和修订。这种递归方法可以提高答案的连贯性、逻辑性和鲁棒性，从而改进整体的决策过程。我们展示了如何通过简单的提示工程技术在现有的LLM架构中有效地实施该方法，从而实现类似于学习精炼模型（LRM）的效果，无需额外的训练。此外，我们提供了一种实用指南，说明如何在Google Colab中实施该方法，便于将其无缝集成到实际应用中。 

---
