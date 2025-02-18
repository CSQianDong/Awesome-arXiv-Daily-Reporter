# Small Models Struggle to Learn from Strong Reasoners 

**Title (ZH)**: 小型模型难以从强大推理器中学习 

**Authors**: Yuetai Li, Xiang Yue, Zhangchen Xu, Fengqing Jiang, Luyao Niu, Bill Yuchen Lin, Bhaskar Ramasubramanian, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2502.12143)  

**Abstract**: Large language models (LLMs) excel in complex reasoning tasks, and distilling their reasoning capabilities into smaller models has shown promise. However, we uncover an interesting phenomenon, which we term the Small Model Learnability Gap: small models ($\leq$3B parameters) do not consistently benefit from long chain-of-thought (CoT) reasoning or distillation from larger models. Instead, they perform better when fine-tuned on shorter, simpler reasoning chains that better align with their intrinsic learning capacity. To address this, we propose Mix Distillation, a simple yet effective strategy that balances reasoning complexity by combining long and short CoT examples or reasoning from both larger and smaller models. Our experiments demonstrate that Mix Distillation significantly improves small model reasoning performance compared to training on either data alone. These findings highlight the limitations of direct strong model distillation and underscore the importance of adapting reasoning complexity for effective reasoning capability transfer. 

**Abstract (ZH)**: 大型语言模型（LLMs）在复杂的推理任务中表现出色，将其推理能力提炼到较小的模型中也展现出了潜力。然而，我们发现了一个有趣的现象，我们称之为小型模型学习差距：小型模型（$\leq$3B参数）并不一致地从较长的链式思维（CoT）推理或从较大模型的提炼中受益。相反，当它们被微调在较短且更简单的推理链上，这些推理链更好地与它们的内在学习能力相一致时，它们的性能更好。为了解决这个问题，我们提出了一种简单而有效的策略——混合提炼（Mix Distillation），该策略通过结合较长和较短的CoT示例，或者从较大和较小模型中提取推理来平衡推理复杂性。我们的实验表明，与仅使用数据进行训练相比，混合提炼显著提高了小型模型的推理性能。这些发现突显了直接强模型提炼的局限性，并强调了适应推理复杂性对于有效推理能力转移的重要性。 

---
# Transformer Dynamics: A neuroscientific approach to interpretability of large language models 

**Title (ZH)**: Transformer 动力学：一种神经科学方法解读大型语言模型的可解释性 

**Authors**: Jesseba Fernando, Grigori Guitchounts  

**Link**: [PDF](https://arxiv.org/pdf/2502.12131)  

**Abstract**: As artificial intelligence models have exploded in scale and capability, understanding of their internal mechanisms remains a critical challenge. Inspired by the success of dynamical systems approaches in neuroscience, here we propose a novel framework for studying computations in deep learning systems. We focus on the residual stream (RS) in transformer models, conceptualizing it as a dynamical system evolving across layers. We find that activations of individual RS units exhibit strong continuity across layers, despite the RS being a non-privileged basis. Activations in the RS accelerate and grow denser over layers, while individual units trace unstable periodic orbits. In reduced-dimensional spaces, the RS follows a curved trajectory with attractor-like dynamics in the lower layers. These insights bridge dynamical systems theory and mechanistic interpretability, establishing a foundation for a "neuroscience of AI" that combines theoretical rigor with large-scale data analysis to advance our understanding of modern neural networks. 

**Abstract (ZH)**: 随着人工智能模型在规模和能力上爆炸式增长，对其内部机制的理解仍然是一个关键挑战。受到神经科学中动力系统方法成功的启发，我们提出了一种新的框架，用于研究深度学习系统的计算过程。我们专注于变换器模型中的残差流（RS），将其概念化为一层层演进的动力系统。我们发现，RS 单元的激活在整个层中表现出强烈的连续性，尽管 RS 并不是一个特权基。激活在 RS 中随层次加速并变得越来越密集，而单个单元则追踪不稳定的周期轨道。在降维空间中，RS 在下层呈现出类似吸引子的动力学轨迹。这些洞察将动力系统理论与机制化可解释性连接起来，为结合理论严谨性与大规模数据分析的“人工智能神经科学”奠定了基础，从而推动我们对现代神经网络的理解。 

---
# Scaling Autonomous Agents via Automatic Reward Modeling And Planning 

**Title (ZH)**: 通过自动奖励建模与规划扩展自主代理 

**Authors**: Zhenfang Chen, Delin Chen, Rui Sun, Wenjun Liu, Chuang Gan  

**Link**: [PDF](https://arxiv.org/pdf/2502.12130)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities across a range of text-generation tasks. However, LLMs still struggle with problems requiring multi-step decision-making and environmental feedback, such as online shopping, scientific reasoning, and mathematical problem-solving. Unlike pure text data, collecting large-scale decision-making data is challenging. Moreover, many powerful LLMs are only accessible through APIs, which hinders their fine-tuning for agent tasks due to cost and complexity. To address LLM agents' limitations, we propose a framework that can automatically learn a reward model from the environment without human annotations. This model can be used to evaluate the action trajectories of LLM agents and provide heuristics for task planning. Specifically, our approach involves employing one LLM-based agent to navigate an environment randomly, generating diverse action trajectories. Subsequently, a separate LLM is leveraged to assign a task intent and synthesize a negative response alongside the correct response for each trajectory. These triplets (task intent, positive response, and negative response) are then utilized as training data to optimize a reward model capable of scoring action trajectories. The effectiveness and generalizability of our framework are demonstrated through evaluations conducted on different agent benchmarks. In conclusion, our proposed framework represents a significant advancement in enhancing LLM agents' decision-making capabilities. By automating the learning of reward models, we overcome the challenges of data scarcity and API limitations, potentially revolutionizing the application of LLMs in complex and interactive environments. This research paves the way for more sophisticated AI agents capable of tackling a wide range of real-world problems requiring multi-step decision-making. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在多种文本生成任务中展现了出色的性能。然而，LLMs 在需要多步决策和环境反馈的问题上仍然存在挑战，如在线购物、科学推理和数学问题解决等。与纯粹的文本数据不同，收集大规模的决策数据是非常具有挑战性的。此外，许多强大的LLMs仅可通过API获取，这增加了它们对代理任务进行微调的成本和复杂性。为解决LLM代理的限制，我们提出了一种无需人工标注即可自动学习奖励模型的框架。该模型可用于评估LLM代理的行为轨迹，并为任务规划提供启发式建议。具体而言，我们的方法包括使用基于LLM的代理在环境中随机导航，生成多样化的行为轨迹。随后，利用另一個LLM给每个轨迹分配任务意图，并合成与正确响应并列的错误响应。这些三元组（任务意图、正响应和负响应）将被用作训练数据以优化一个能够评估行为轨迹的奖励模型。我们通过在不同代理基准上的评估证明了框架的有效性和普适性。总之，我们提出的框架显著改进了LLM代理的决策能力。通过自动化奖励模型的学习，我们克服了数据稀缺性和API限制的问题，有可能变革LLM在复杂和交互式环境中的应用。本研究为开发能够解决多步决策所需广泛现实问题的更高级AI代理奠定了基础。 

---
# Hypernym Bias: Unraveling Deep Classifier Training Dynamics through the Lens of Class Hierarchy 

**Title (ZH)**: 超类型偏差：通过类阶层的视角解析深度分类器训练动力学 

**Authors**: Roman Malashin, Valeria Yachnaya, Alexander Mullin  

**Link**: [PDF](https://arxiv.org/pdf/2502.12125)  

**Abstract**: We investigate the training dynamics of deep classifiers by examining how hierarchical relationships between classes evolve during training. Through extensive experiments, we argue that the learning process in classification problems can be understood through the lens of label clustering. Specifically, we observe that networks tend to distinguish higher-level (hypernym) categories in the early stages of training, and learn more specific (hyponym) categories later. We introduce a novel framework to track the evolution of the feature manifold during training, revealing how the hierarchy of class relations emerges and refines across the network layers. Our analysis demonstrates that the learned representations closely align with the semantic structure of the dataset, providing a quantitative description of the clustering process. Notably, we show that in the hypernym label space, certain properties of neural collapse appear earlier than in the hyponym label space, helping to bridge the gap between the initial and terminal phases of learning. We believe our findings offer new insights into the mechanisms driving hierarchical learning in deep networks, paving the way for future advancements in understanding deep learning dynamics. 

**Abstract (ZH)**: 我们通过考察类间层次关系在训练过程中如何演变，研究了深度分类器的训练动态。通过对大量实验的分析，我们认为分类问题的学习过程可以通过标签聚类的视角来理解。具体而言，我们观察到网络在训练早期倾向于区分较高的层次类别（超类），而在后期则学习更具体的类别（子类）。我们提出了一种新的框架，用于追踪训练过程中特征流形的变化，揭示了类别关系层次结构在各网络层中是如何逐渐形成和完善的。我们的分析表明，学习到的表示与数据集的语义结构紧密对齐，从而为聚类过程提供了一个定量描述。值得注意的是，我们发现神经压缩的某些特性在超类标签空间中比在子类标签空间中出现得更早，这有助于弥合学习初期和末期之间的差距。我们相信，我们的研究结果为深入理解深度网络中的层次学习机制提供了新的见解，并为未来理解深度学习动态的研究铺平了道路。 

---
# Relational Norms for Human-AI Cooperation 

**Title (ZH)**: 人类与人工智能合作中的关系规范 

**Authors**: Brian D. Earp, Sebastian Porsdam Mann, Mateo Aboy, Edmond Awad, Monika Betzler, Marietjie Botes, Rachel Calcott, Mina Caraccio, Nick Chater, Mark Coeckelbergh, Mihaela Constantinescu, Hossein Dabbagh, Kate Devlin, Xiaojun Ding, Vilius Dranseika, Jim A. C. Everett, Ruiping Fan, Faisal Feroz, Kathryn B. Francis, Cindy Friedman, Orsolya Friedrich, Iason Gabriel, Ivar Hannikainen, Julie Hellmann, Arasj Khodadade Jahrome, Niranjan S. Janardhanan, Paul Jurcys, Andreas Kappes, Maryam Ali Khan, Gordon Kraft-Todd, Maximilian Kroner Dale, Simon M. Laham, Benjamin Lange, Muriel Leuenberger, Jonathan Lewis, Peng Liu, David M. Lyreskog, Matthijs Maas, John McMillan, Emilian Mihailov, Timo Minssen, Joshua Teperowski Monrad, Kathryn Muyskens, Simon Myers, Sven Nyholm, Alexa M. Owen, Anna Puzio, Christopher Register, Madeline G. Reinecke, Adam Safron, Henry Shevlin, Hayate Shimizu, Peter V. Treit, Cristina Voinea, Karen Yan, Anda Zahiu, Renwen Zhang, Hazem Zohny, Walter Sinnott-Armstrong, Ilina Singh, Julian Savulescu, Margaret S. Clark  

**Link**: [PDF](https://arxiv.org/pdf/2502.12102)  

**Abstract**: How we should design and interact with social artificial intelligence depends on the socio-relational role the AI is meant to emulate or occupy. In human society, relationships such as teacher-student, parent-child, neighbors, siblings, or employer-employee are governed by specific norms that prescribe or proscribe cooperative functions including hierarchy, care, transaction, and mating. These norms shape our judgments of what is appropriate for each partner. For example, workplace norms may allow a boss to give orders to an employee, but not vice versa, reflecting hierarchical and transactional expectations. As AI agents and chatbots powered by large language models are increasingly designed to serve roles analogous to human positions - such as assistant, mental health provider, tutor, or romantic partner - it is imperative to examine whether and how human relational norms should extend to human-AI interactions. Our analysis explores how differences between AI systems and humans, such as the absence of conscious experience and immunity to fatigue, may affect an AI's capacity to fulfill relationship-specific functions and adhere to corresponding norms. This analysis, which is a collaborative effort by philosophers, psychologists, relationship scientists, ethicists, legal experts, and AI researchers, carries important implications for AI systems design, user behavior, and regulation. While we accept that AI systems can offer significant benefits such as increased availability and consistency in certain socio-relational roles, they also risk fostering unhealthy dependencies or unrealistic expectations that could spill over into human-human relationships. We propose that understanding and thoughtfully shaping (or implementing) suitable human-AI relational norms will be crucial for ensuring that human-AI interactions are ethical, trustworthy, and favorable to human well-being. 

**Abstract (ZH)**: 关于如何设计和互动于社会人工智能，这取决于人工智能旨在模仿或占据的社会关系角色。在人类社会中，诸如师徒、亲子、邻居、兄弟姐妹或雇主雇员等关系都由特定规范所规范，这些规范规定或禁止了合作功能，包括等级、关怀、交易和共栖。这些规范塑造了每个人与关系对方面临的情况是否适当。例如，在工作场所，规范允许老板向员工下达命令，但不允许员工向老板下达命令，这反映了等级和交易的期望。随着基于大规模语言模型的人工智能代理和聊天机器人越来越被设计为类人类岗位的替代者，例如助手、心理健康提供者、导师或浪漫伴侣，人们有必要探讨人类关系规范是否以及如何延展到人类-人工智能互动中。我们的分析探讨了人工智能系统与人类之间的差异如何影响人工智能履行特定关系功能和遵守相应规范的能力，包括缺乏意识经验以及对疲劳的免疫力。这一分析由哲学家、心理学家、人际关系科学家、伦理学家、法律专家和人工智能研究人员共同完成，对于人工智能系统的设计、用户行为和监管具有重要意义。虽然我们承认人工智能系统可以提供诸如某些社会关系角色中增加的可获得性和一致性等显著益处，但它们也可能培养不健康的依赖关系或不切实际的期望，从而影响人类-人类关系。我们提出，理解并慎重塑造（或实施）适合的人类-人工智能关系规范对于确保人类-人工智能互动的伦理、可信性和有利于人类福祉至关重要。 

---
# A Study on Leveraging Search and Self-Feedback for Agent Reasoning 

**Title (ZH)**: 探讨利用搜索与自我反馈提升智能体推理能力的研究 

**Authors**: Karthikeyan K, Michelle Yuan, Elman Mansimov, Katerina Margatina, Anurag Pratik, Daniele Bonadiman, Monica Sunkara, Yi Zhang, Yassine Benajiba  

**Link**: [PDF](https://arxiv.org/pdf/2502.12094)  

**Abstract**: Recent works have demonstrated that incorporating search during inference can significantly improve reasoning capabilities of language agents. Some approaches may make use of the ground truth or rely on model's own generated feedback. The search algorithm uses this feedback to then produce values that will update its criterion for exploring and exploiting various reasoning paths. In this study, we investigate how search and model's self-feedback can be leveraged for reasoning tasks. First, we explore differences in ground-truth feedback and self-feedback during search for math reasoning. Second, we observe limitations in applying search techniques to more complex tasks like tool-calling and design domain-specific approaches to address these gaps. Our experiments reveal challenges related to generalization when solely relying on self-feedback during search. For search to work effectively, either access to the ground-truth is needed or feedback mechanisms need to be carefully designed for the specific task. 

**Abstract (ZH)**: 最近的研究表明，在推理过程中融入搜索可以显著提高语言代理的推理能力。一些方法可能会利用地面真相或依赖模型自身生成的反馈。搜索算法利用这种反馈来生成更新其探索和利用各种推理路径的标准值。在本研究中，我们探讨了如何利用搜索和模型的自我反馈来服务于推理任务。首先，我们研究了在数学推理中，搜索过程中地面真相反馈和自我反馈之间的差异。其次，我们观察到在复杂任务（如工具调用和设计领域）中应用搜索技术时存在的局限性，并针对这些差距提出了特定任务的方法。我们的实验揭示了仅依赖自我反馈进行搜索时泛化能力方面的挑战。为了使搜索有效，要么需要访问地面真相，要么需要为特定任务精心设计反馈机制。 

---
# CONSTRUCTA: Automating Commercial Construction Schedules in Fabrication Facilities with Large Language Models 

**Title (ZH)**: CONSTRUCTA：通过大型语言模型在加工设施中自动化商业建筑施工进度规划 

**Authors**: Yifan Zhang, Xue Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12066)  

**Abstract**: Automating planning with LLMs presents transformative opportunities for traditional industries, yet remains underexplored. In commercial construction, the complexity of automated scheduling often requires manual intervention to ensure precision. We propose CONSTRUCTA, a novel framework leveraging LLMs to optimize construction schedules in complex projects like semiconductor fabrication. CONSTRUCTA addresses key challenges by: (1) integrating construction-specific knowledge through static RAG; (2) employing context-sampling techniques inspired by architectural expertise to provide relevant input; and (3) deploying Construction DPO to align schedules with expert preferences using RLHF. Experiments on proprietary data demonstrate performance improvements of +42.3% in missing value prediction, +79.1% in dependency analysis, and +28.9% in automated planning compared to baseline methods, showcasing its potential to revolutionize construction workflows and inspire domain-specific LLM advancements. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，并符合学术规范：

利用大模型（LLM）自动化规划为传统行业带来了变革性的机会，但这一领域仍处于探索阶段。在商业建筑中，自动排程的复杂性往往需要手动干预以确保精确性。我们提出了一种名为CONSTRUCTA的新颖框架，利用LLM优化复杂项目（如半导体制造）的施工排程。CONSTRUCTA 通过以下方式解决关键挑战：（1）通过静态RAG集成施工特定知识；（2）采用受建筑设计专业启发的上下文采样技术，提供相关输入；（3）部署Construction DPO，利用RLHF将排程与专家偏好对齐。在内部数据上的实验表明，与基准方法相比，CONSTRUCTA 在缺失值预测上的性能提高了42.3%，依赖性分析提高了79.1%，自动化规划提高了28.9%，这展示了其在革新施工工作流程方面的潜力，并启发了领域特定的大模型发展。 

---
# PhysReason: A Comprehensive Benchmark towards Physics-Based Reasoning 

**Title (ZH)**: PhysReason：基于物理的推理综合基准 

**Authors**: Xinyu Zhang, Yuxuan Dong, Yanrui Wu, Jiaxing Huang, Chengyou Jia, Basura Fernando, Mike Zheng Shou, Lingling Zhang, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12054)  

**Abstract**: Large language models demonstrate remarkable capabilities across various domains, especially mathematics and logic reasoning. However, current evaluations overlook physics-based reasoning - a complex task requiring physics theorems and constraints. We present PhysReason, a 1,200-problem benchmark comprising knowledge-based (25%) and reasoning-based (75%) problems, where the latter are divided into three difficulty levels (easy, medium, hard). Notably, problems require an average of 8.1 solution steps, with hard requiring 15.6, reflecting the complexity of physics-based reasoning. We propose the Physics Solution Auto Scoring Framework, incorporating efficient answer-level and comprehensive step-level evaluations. Top-performing models like Deepseek-R1, Gemini-2.0-Flash-Thinking, and o3-mini-high achieve less than 60% on answer-level evaluation, with performance dropping from knowledge questions (75.11%) to hard problems (31.95%). Through step-level evaluation, we identified four key bottlenecks: Physics Theorem Application, Physics Process Understanding, Calculation, and Physics Condition Analysis. These findings position PhysReason as a novel and comprehensive benchmark for evaluating physics-based reasoning capabilities in large language models. Our code and data will be published at https:/dxzxy12138.github.io/PhysReason. 

**Abstract (ZH)**: 大型语言模型在多个领域展现出惊人的能力，尤其是在数学和逻辑推理方面。然而，当前的评估忽略了基于物理学的推理——这是一个复杂的任务，需要应用物理学定理和约束条件。我们提出了PhysReason基准测试，包含了1200个问题，其中包含知识基础型问题（占25%）和推理型问题（占75%），后者又划分为三个难度级别（简单、中等、困难）。值得注意的是，这些问题平均需要8.1步解决方案，而困难级需要15.6步，这反映了基于物理学的推理的复杂性。我们提出了物理解决方案自动评分框架，其中包括高效的回答级别和全面的步骤级别评估。表现最佳的模型如Deepseek-R1、Gemini-2.0-Flash-Thinking和o3-mini-high在回答级别评估中的得分低于60%，从知识问题（75.11%）下降到困难问题（31.95%）。通过步骤级别评估，我们确定了四个关键瓶颈：物理定理应用、物理过程理解、计算和物理条件分析。这些发现使PhysReason成为一个新颖且全面的基准，用于评估大型语言模型的基于物理学的推理能力。我们将代码和数据发布在https://dxzxy12138.github.io/PhysReason。 

---
# A Survey on Bridging EEG Signals and Generative AI: From Image and Text to Beyond 

**Title (ZH)**: 一种关于连接脑电波信号与生成式AI的综述：从图像和文本到更广泛的领域 

**Authors**: Shreya Shukla, Jose Torres, Abhijit Mishra, Jacek Gwizdka, Shounak Roychowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2502.12048)  

**Abstract**: Integration of Brain-Computer Interfaces (BCIs) and Generative Artificial Intelligence (GenAI) has opened new frontiers in brain signal decoding, enabling assistive communication, neural representation learning, and multimodal integration. BCIs, particularly those leveraging Electroencephalography (EEG), provide a non-invasive means of translating neural activity into meaningful outputs. Recent advances in deep learning, including Generative Adversarial Networks (GANs) and Transformer-based Large Language Models (LLMs), have significantly improved EEG-based generation of images, text, and speech. This paper provides a literature review of the state-of-the-art in EEG-based multimodal generation, focusing on (i) EEG-to-image generation through GANs, Variational Autoencoders (VAEs), and Diffusion Models, and (ii) EEG-to-text generation leveraging Transformer based language models and contrastive learning methods. Additionally, we discuss the emerging domain of EEG-to-speech synthesis, an evolving multimodal frontier. We highlight key datasets, use cases, challenges, and EEG feature encoding methods that underpin generative approaches. By providing a structured overview of EEG-based generative AI, this survey aims to equip researchers and practitioners with insights to advance neural decoding, enhance assistive technologies, and expand the frontiers of brain-computer interaction. 

**Abstract (ZH)**: 脑-计算机接口（BCIs）与生成型人工智能（GenAI）的集成开辟了脑信号解码的新前沿，使辅助通信、神经表示学习以及多模态集成成为可能。BCIs，尤其是利用脑电图（EEG）的技术，提供了一种无创的方法，将神经活动转化为有意义的输出。近年来，通过生成对抗网络（GANs）和基于变换器的语言模型（LLMs）等深度学习的进展，大大提升了基于EEG的图像、文本和语音生成的效果。本文提供了基于EEG的多模态生成领域的综述，重点关注（i）通过GANs、变分自编码器（VAEs）和扩散模型实现的EEG到图像的生成，以及（ii）通过基于变换器的语言模型和对比学习方法实现的EEG到文本的生成。此外，我们还讨论了新兴的EEG到语音合成领域，这是一个正在演进的多模态前沿。文中强调了关键的数据集、应用场景、挑战以及用于生成方法的EEG特征编码方法。通过提供基于EEG的生成型人工智能的结构化概述，本文旨在为研究者和从业者提供洞见，以促进神经解码、提升辅助技术并扩展脑-计算机交互的前沿。 

---
# KnowPath: Knowledge-enhanced Reasoning via LLM-generated Inference Paths over Knowledge Graphs 

**Title (ZH)**: 知径：通过生成的推理路径增强的知识图谱推理 

**Authors**: Qi Zhao, Hongyu Yang, Qi Song, Xinwei Yao, Xiangyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12029)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in various complex tasks, yet they still suffer from hallucinations. Introducing external knowledge, such as knowledge graph, can enhance the LLMs' ability to provide factual answers. LLMs have the ability to interactively explore knowledge graphs. However, most approaches have been affected by insufficient internal knowledge excavation in LLMs, limited generation of trustworthy knowledge reasoning paths, and a vague integration between internal and external knowledge. Therefore, we propose KnowPath, a knowledge-enhanced large model framework driven by the collaboration of internal and external knowledge. It relies on the internal knowledge of the LLM to guide the exploration of interpretable directed subgraphs in external knowledge graphs, better integrating the two knowledge sources for more accurate reasoning. Extensive experiments on multiple real-world datasets confirm the superiority of KnowPath. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种复杂任务中展现出了显著的能力，但仍然存在幻觉问题。引入外部知识，如知识图谱，可以增强LLMs提供事实性答案的能力。LLMs具有互动探索知识图谱的能力。然而，大多数方法受到了LLMs内部知识挖掘不足、可信知识推理路径生成有限以及内部和外部知识融合不畅的限制。因此，我们提出了一种名为KnowPath的知识增强大型模型框架，该框架依赖于LLMs的内部知识来引导对外部知识图谱中有解释性的有向子图的探索，从而更好地整合两种知识来源，以实现更准确的推理。多种真实世界数据集上的大量实验证明了KnowPath的优势。 

---
# SafeChain: Safety of Language Models with Long Chain-of-Thought Reasoning Capabilities 

**Title (ZH)**: SafeChain：具有长链推理能力的语言模型的安全性 

**Authors**: Fengqing Jiang, Zhangchen Xu, Yuetai Li, Luyao Niu, Zhen Xiang, Bo Li, Bill Yuchen Lin, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2502.12025)  

**Abstract**: Emerging large reasoning models (LRMs), such as DeepSeek-R1 models, leverage long chain-of-thought (CoT) reasoning to generate structured intermediate steps, enhancing their reasoning capabilities. However, long CoT does not inherently guarantee safe outputs, potentially leading to harmful consequences such as the introduction of security vulnerabilities in code or the spread of misinformation. Current research on large language model (LLM) safety usually focuses on short-answer responses, overlooking the long CoT style outputs of LRMs. To bridge this gap, we conduct a systematic study of LRM safety. First, we investigate safety evaluators calibrated against human annotations. Using our newly developed metrics, we thoroughly assess the safety of 12 state-of-the-art LRMs on StrongReject and WildJailbreak datasets. Our results show that LRMs are not safe compared to their reasoning advance. Further, we perform a fine-grained analysis of the reasoning trace and final answer. We find that three decoding strategies-ZeroThink, LessThink, and MoreThink-can improve model safety without additional training. However, these strategies either use constrained reasoning traces or incur high inference costs. To better strengthen LRM safety, we introduce SafeChain, the first-of-its-kind safety training dataset in CoT style. We fine-tune two LRMs with SafeChain, showing that it not only enhances model safety but also preserves performance across 6 reasoning benchmarks. 

**Abstract (ZH)**: 新兴的大规模推理模型（LRMs），如DeepSeek-R1模型，通过长推理链（CoT）推理解释生成结构化的中间步骤，从而增强其推理能力。然而，长推理链并不天然保证输出的安全性，可能会导致诸如代码中引入安全漏洞或传播虚假信息等有害后果。当前对大规模语言模型（LLMs）安全性的研究通常集中于简短答案响应，忽视了LRMs的长推理链样式输出。为了弥合这一缺口，我们对LRMs的安全性进行了系统研究。首先，我们调查了经过人类注释校准的安全评估器。利用我们新开发的度量标准，我们全面评估了12个最先进的LRMs在StrongReject和WildJailbreak数据集上的安全性。结果表明，LRMs在推理能力上的进步并没有转化为更高的安全性。此外，我们对推理轨迹和最终答案进行了精细分析。我们发现，三种解码策略——ZeroThink、LessThink和MoreThink——可以在不进行额外训练的情况下提高模型安全性。然而，这些策略要么限制了推理链的范围，要么导致高昂的推理成本。为了更好地加强LRMs的安全性，我们引入了SafeChain，这是首个用于CoT样式的安全训练数据集。我们对两种LRMs进行了微调，结果显示，它不仅提高了模型安全性，还在6个推理基准测试中保持了性能。 

---
# Learning Generalizable Prompt for CLIP with Class Similarity Knowledge 

**Title (ZH)**: 使用类相似性知识学习可泛化的CLIP提示词 

**Authors**: Sehun Jung, Hyang-won Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.11969)  

**Abstract**: In vision-language models (VLMs), prompt tuning has shown its effectiveness in adapting models to downstream tasks. However, learned prompts struggle to generalize to unseen classes, as they tend to overfit to the classes that are targeted during prompt tuning. Examining failure cases, we observed that learned prompts disrupt the semantics of unseen classes, generating text embeddings with incorrect semantic relationships among classes. To address this, we propose Similarity Alignment Regularization (SAR), which regularizes learnable prompts to preserve the semantic relationships among classes captured by hand-crafted prompts. Specifically, we first obtain novel classes related to base classes using ChatGPT-4o and utilize them as potential unseen classes during prompt tuning. Then, by targeting both base and novel classes, SAR aligns the similarity relationships among text embeddings generated by learnable prompts with the similarity relationships from hand-crafted prompts. Extensive experiments applying SAR to existing prompt tuning methods demonstrate its effectiveness in improving generalization to unseen classes. 

**Abstract (ZH)**: 在视觉-语言模型（VLMs）中，提示调优已显示出其适应下游任务的有效性。然而，学到的提示在面对未见类别时难以泛化，因为它们倾向于在提示调优过程中针对的目标类别上过度拟合。通过对失败案例的分析，我们观察到学到的提示破坏了未见类别的语义，生成的文本嵌入中的类别间语义关系不正确。为了解决这一问题，我们提出了相似性对齐正则化（SAR），该方法通过正则化可学习的提示来保持手工艺品提示捕获的类别间的语义关系。具体而言，我们首先使用ChatGPT-4o获取与基础类别相关的新型类别，并在提示调优过程中将它们作为潜在的未见类别使用。然后，通过同时针对基础类别和新型类别，SAR对由可学习提示生成的文本嵌入之间相似性关系与手工艺品提示的相似性关系进行对齐。将SAR应用于现有提示调优方法的广泛实验表明，它在提高对未见类别的泛化能力方面具有有效性。 

---
# STRIVE: Structured Reasoning for Self-Improvement in Claim Verification 

**Title (ZH)**: STRIVE：结构化推理在声明验证中的自我提升 

**Authors**: Haisong Gong, Jing Li, Junfei Wu, Qiang Liu, Shu Wu, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11959)  

**Abstract**: Claim verification is the task of determining whether a claim is supported or refuted by evidence. Self-improvement methods, where reasoning chains are generated and those leading to correct results are selected for training, have succeeded in tasks like mathematical problem solving. However, in claim verification, this approach struggles. Low-quality reasoning chains may falsely match binary truth labels, introducing faulty reasoning into the self-improvement process and ultimately degrading performance. To address this, we propose STRIVE: Structured Reasoning for Self-Improved Verification. Our method introduces a structured reasoning design with Claim Decomposition, Entity Analysis, and Evidence Grounding Verification. These components improve reasoning quality, reduce errors, and provide additional supervision signals for self-improvement. STRIVE begins with a warm-up phase, where the base model is fine-tuned on a small number of annotated examples to learn the structured reasoning design. It is then applied to generate reasoning chains for all training examples, selecting only those that are correct and structurally sound for subsequent self-improvement training. We demonstrate that STRIVE achieves significant improvements over baseline models, with a 31.4% performance gain over the base model and 20.7% over Chain of Thought on the HOVER datasets, highlighting its effectiveness. 

**Abstract (ZH)**: 声明验证是确定某个声明是否由证据支持的任务。自我改进方法，其中生成推理链并选择导致正确结果的链进行训练，已经在数学问题解决等任务上取得了成功。然而，在声明验证中，这一方法遇到了挑战。质量低的推理链可能会错误地匹配二元真实标签，引入错误推理到自我改进过程中，最终降低性能。为了解决这一问题，我们提出了一种名为STRIVE（Structured Reasoning for Self-Improved Verification）的方法。该方法引入了一种结构化推理设计，包括声明分解、实体分析和证据链接验证。这些组件提高了推理质量，减少了错误，并为自我改进提供了额外的监督信号。STRIVE首先采用预热阶段，在少量标注示例上微调基础模型，以学习结构化推理设计。然后，它应用于生成所有训练示例的推理链，并仅选择那些正确且结构健全的推理链进行后续的自我改进训练。我们展示了STRIVE在基准模型上取得显著改进，相比于基础模型，其性能提升了31.4%，相比于Chain of Thought，提升了20.7%，突显了其有效性。 

---
# GRAPHGPT-O: Synergistic Multimodal Comprehension and Generation on Graphs 

**Title (ZH)**: GRAPHGPT-O：图上的多模态理解与生成协同模型 

**Authors**: Yi Fang, Bowen Jin, Jiacheng Shen, Sirui Ding, Qiaoyu Tan, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.11925)  

**Abstract**: The rapid development of Multimodal Large Language Models (MLLMs) has enabled the integration of multiple modalities, including texts and images, within the large language model (LLM) framework. However, texts and images are usually interconnected, forming a multimodal attributed graph (MMAG). It is underexplored how MLLMs can incorporate the relational information (\textit{i.e.}, graph structure) and semantic information (\textit{i.e.,} texts and images) on such graphs for multimodal comprehension and generation. In this paper, we propose GraphGPT-o, which supports omni-multimodal understanding and creation on MMAGs. We first comprehensively study linearization variants to transform semantic and structural information as input for MLLMs. Then, we propose a hierarchical aligner that enables deep graph encoding, bridging the gap between MMAGs and MLLMs. Finally, we explore the inference choices, adapting MLLM to interleaved text and image generation in graph scenarios. Extensive experiments on three datasets from different domains demonstrate the effectiveness of our proposed method. Datasets and codes will be open-sourced upon acceptance. 

**Abstract (ZH)**: 随着多模态大型语言模型（Multimodal Large Language Models, MLLMs）的快速发展，已经能够在大型语言模型（Large Language Model, LLM）框架中整合多种模态，包括文本和图像。然而，文本和图像通常彼此关联，形成多模态带属性的图（multimodal attributed graph, MMAG）。当前对于如何将MLLMs整合到MMAGs中的关系信息（即，图结构）和语义信息（即，文本和图像）中的研究尚不够深入，这对于多模态理解和生成仍然是未探索的领域。本文中，我们提出了GraphGPT-o，该方法支持在MMAGs上实现全方位的多模态理解和创作。首先，我们全面研究了线性化变体，将语义和结构信息转换为MLLMs的输入。然后，我们提出了一个层次对齐器，能够实现深度图编码，从而弥合MMAGs和MLLMs之间的差距。最后，我们探讨了推理选择，使MLLM能够适应在图场景中交错的文本和图像生成。来自不同领域的三个数据集的大量实验充分证明了我们所提出方法的有效性。方法的相关数据集和代码将在论文被接受后开源。 

---
# On the robustness of ChatGPT in teaching Korean Mathematics 

**Title (ZH)**: 《ChatGPT在教学韩国数学中的稳健性研究》 

**Authors**: Phuong-Nam Nguyen, Quang Nguyen-The, An Vu-Minh, Diep-Anh Nguyen, Xuan-Lam Pham  

**Link**: [PDF](https://arxiv.org/pdf/2502.11915)  

**Abstract**: ChatGPT, an Artificial Intelligence model, has the potential to revolutionize education. However, its effectiveness in solving non-English questions remains uncertain. This study evaluates ChatGPT's robustness using 586 Korean mathematics questions. ChatGPT achieves 66.72% accuracy, correctly answering 391 out of 586 questions. We also assess its ability to rate mathematics questions based on eleven criteria and perform a topic analysis. Our findings show that ChatGPT's ratings align with educational theory and test-taker perspectives. While ChatGPT performs well in question classification, it struggles with non-English contexts, highlighting areas for improvement. Future research should address linguistic biases and enhance accuracy across diverse languages. Domain-specific optimizations and multilingual training could improve ChatGPT's role in personalized education. 

**Abstract (ZH)**: ChatGPT是一种人工智能模型，有可能革新教育领域。然而，在解决非英文问题方面，其有效性尚存不确定性。本研究使用586道韩语文科问题评估ChatGPT的稳健性，结果表明ChatGPT的准确率为66.72%，正确回答了391道题目。此外，我们还评估了ChatGPT基于11项标准对题目进行评级的能力，并进行了主题分析。研究发现，ChatGPT的评分与教育理论和考生视角相符。尽管ChatGPT在问题分类方面表现良好，但在非英文语境中仍面临挑战，这指出了需要改进的领域。未来的研究应解决语言偏见并提高其在多种语言中的准确性。针对特定领域的优化和多语种训练有望提高ChatGPT在个性化教育中的作用。 

---
# Leveraging Dual Process Theory in Language Agent Framework for Real-time Simultaneous Human-AI Collaboration 

**Title (ZH)**: 利用双过程理论在语言代理框架中实现实时人机协同合作 

**Authors**: Shao Zhang, Xihuai Wang, Wenhao Zhang, Chaoran Li, Junru Song, Tingyu Li, Lin Qiu, Xuezhi Cao, Xunliang Cai, Wen Yao, Weinan Zhang, Xinbing Wang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11882)  

**Abstract**: Agents built on large language models (LLMs) have excelled in turn-by-turn human-AI collaboration but struggle with simultaneous tasks requiring real-time interaction. Latency issues and the challenge of inferring variable human strategies hinder their ability to make autonomous decisions without explicit instructions. Through experiments with current independent System 1 and System 2 methods, we validate the necessity of using Dual Process Theory (DPT) in real-time tasks. We propose DPT-Agent, a novel language agent framework that integrates System 1 and System 2 for efficient real-time simultaneous human-AI collaboration. DPT-Agent's System 1 uses a Finite-state Machine (FSM) and code-as-policy for fast, intuitive, and controllable decision-making. DPT-Agent's System 2 integrates Theory of Mind (ToM) and asynchronous reflection to infer human intentions and perform reasoning-based autonomous decisions. We demonstrate the effectiveness of DPT-Agent through further experiments with rule-based agents and human collaborators, showing significant improvements over mainstream LLM-based frameworks. To the best of our knowledge, DPT-Agent is the first language agent framework that achieves successful real-time simultaneous human-AI collaboration autonomously. Code of DPT-Agent can be found in this https URL. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的代理在人机逐轮协作方面表现出色，但在需要实时互动的同步任务中却遇到挑战。延迟问题和推断多变的人类策略的困难阻碍了它们在没有明确指令的情况下自主决策的能力。通过使用当前独立的System 1和System 2方法进行实验，我们验证了在实时任务中采用二过程理论（Dual Process Theory, DPT）的必要性。我们提出了DPT-Agent这一新型语言代理框架，它将System 1和System 2高效整合，以实现实时同步的人机协作。DPT-Agent的System 1使用有限状态机（FSM）和以代码为策略的方法，以实现快速、直观且可控的决策。DPT-Agent的System 2整合了心智理论（Theory of Mind, ToM）和异步反思，以推断人类意图并执行基于推理的自主决策。通过进一步与规则基代理及人类合作者的实验，我们展示了DPT-Agent的有效性，并且相比主流的基于LLM的框架，取得了显著改进。据我们所知，DPT-Agent是第一个能够成功实现自主实时同步人机协作的语言代理框架。DPT-Agent的代码可以在以下链接找到：[此链接](https://)。 

---
# Hypothesis-Driven Theory-of-Mind Reasoning for Large Language Models 

**Title (ZH)**: 基于假设的理论心智推理方法在大型语言模型中的应用 

**Authors**: Hyunwoo Kim, Melanie Sclar, Tan Zhi-Xuan, Lance Ying, Sydney Levine, Yang Liu, Joshua B. Tenenbaum, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2502.11881)  

**Abstract**: Existing LLM reasoning methods have shown impressive capabilities across various tasks, such as solving math and coding problems. However, applying these methods to scenarios without ground-truth answers or rule-based verification methods - such as tracking the mental states of an agent - remains challenging. Inspired by the sequential Monte Carlo algorithm, we introduce thought-tracing, an inference-time reasoning algorithm designed to trace the mental states of specific agents by generating hypotheses and weighting them based on observations without relying on ground-truth solutions to questions in datasets. Our algorithm is modeled after the Bayesian theory-of-mind framework, using LLMs to approximate probabilistic inference over agents' evolving mental states based on their perceptions and actions. We evaluate thought-tracing on diverse theory-of-mind benchmarks, demonstrating significant performance improvements compared to baseline LLMs. Our experiments also reveal interesting behaviors of the recent reasoning models - e.g., o1 and R1 - on theory-of-mind, highlighting the difference of social reasoning compared to other domains. 

**Abstract (ZH)**: 现有的大规模语言模型（LLM）推理方法在各种任务中展示了令人印象深刻的能力，例如解决数学和编程问题。然而，将这些方法应用到缺乏确切答案或基于规则验证方法的场景中——例如跟踪智能体的思维状态——仍然充满挑战。受到序列蒙特卡洛算法的启发，我们引入了思维追踪（thought-tracing）这一推理算法，该算法在推理时生成假设并根据观察结果给这些假设赋予权重，而无需依赖数据集中问题的确切答案进行验证。该算法基于贝叶斯理论心智理论框架，使用LLM来根据智能体的感知和行动对其思维状态的演变进行概率推理。我们在多样的心智理论基准上评估了思维追踪算法，结果表明其性能显著优于基线LLM。我们的实验还揭示了最近的推理模型——例如o1和R1——在心智理论上的有趣行为，突显了社会推理与其它领域之间的差异。 

---
# AAKT: Enhancing Knowledge Tracing with Alternate Autoregressive Modeling 

**Title (ZH)**: AAKT：交替自回归建模增强知识追踪

解释：这里"AAKT" 似乎是论文的缩写，保持不变。"Enhancing Knowledge Tracing with Alternate Autoregressive Modeling" 被翻译为“交替自回归建模增强知识追踪”。其中，“Knowledge Tracing”在学术文献中通常翻译为“知识追踪”。 

**Authors**: Hao Zhou, Wenge Rong, Jianfei Zhang, Qing Sun, Yuanxin Ouyang, Zhang Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11817)  

**Abstract**: Knowledge Tracing (KT) aims to predict students' future performances based on their former exercises and additional information in educational settings. KT has received significant attention since it facilitates personalized experiences in educational situations. Simultaneously, the autoregressive modeling on the sequence of former exercises has been proven effective for this task. One of the primary challenges in autoregressive modeling for Knowledge Tracing is effectively representing the anterior (pre-response) and posterior (post-response) states of learners across exercises. Existing methods often employ complex model architectures to update learner states using question and response records. In this study, we propose a novel perspective on knowledge tracing task by treating it as a generative process, consistent with the principles of autoregressive models. We demonstrate that knowledge states can be directly represented through autoregressive encodings on a question-response alternate sequence, where model generate the most probable representation in hidden state space by analyzing history interactions. This approach underpins our framework, termed Alternate Autoregressive Knowledge Tracing (AAKT). Additionally, we incorporate supplementary educational information, such as question-related skills, into our framework through an auxiliary task, and include extra exercise details, like response time, as additional inputs. Our proposed framework is implemented using advanced autoregressive technologies from Natural Language Generation (NLG) for both training and prediction. Empirical evaluations on four real-world KT datasets indicate that AAKT consistently outperforms all baseline models in terms of AUC, ACC, and RMSE. Furthermore, extensive ablation studies and visualized analysis validate the effectiveness of key components in AAKT. 

**Abstract (ZH)**: 知识追踪（KT）旨在根据学生以往的练习和附加信息来预测他们未来的表现，这一目标在教育环境中促进了个性化体验。自KT引入以来，由于它能够促进个性化的学习体验，受到了广泛的关注。同时，对于以往练习序列进行自回归建模已被证明对于此任务是有效的。在自回归建模中，知识追踪的一个主要挑战是如何有效表示学习者在练习前后（即前响应和后响应）的状态。现有方法通常借助复杂模型架构，使用题目和响应记录来更新学习者状态。

在本研究中，我们从生成过程的角度重新审视了知识追踪任务，这一视角与自回归模型的原则一致。我们证明，知识状态可以直接通过题目-响应交替序列的自回归编码来表示，其中模型通过分析历史交互生成最有可能的隐藏状态表示。这一方法构成了我们框架的核心，我们将其称为交替自回归知识追踪（AAKT）。此外，我们通过附加任务将额外的教育资源信息，如与题目相关的技能，整合到框架中，并将额外的练习细节（如响应时间）作为附加输入。我们的框架采用来自自然语言生成（NLG）的先进自回归技术进行训练和预测。

在四个真实世界的数据集上的实证评估表明，AAKT在AUC、ACC和RMSE方面均优于所有基线模型。此外，全面的消融研究和可视化分析进一步验证了AAKT关键组件的有效性。 

---
# Table-Critic: A Multi-Agent Framework for Collaborative Criticism and Refinement in Table Reasoning 

**Title (ZH)**: Table-Critic：一种用于表格推理中协作批评和改进的多代理框架 

**Authors**: Peiying Yu, Guoxin Chen, Jingjing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11799)  

**Abstract**: Despite the remarkable capabilities of large language models (LLMs) in various reasoning tasks, they still struggle with table reasoning tasks, particularly in maintaining consistency throughout multi-step reasoning processes. While existing approaches have explored various decomposition strategies, they often lack effective mechanisms to identify and correct errors in intermediate reasoning steps, leading to cascading error propagation. To address these issues, we propose Table-Critic, a novel multi-agent framework that facilitates collaborative criticism and iterative refinement of the reasoning process until convergence to correct solutions. Our framework consists of four specialized agents: a Judge for error identification, a Critic for comprehensive critiques, a Refiner for process improvement, and a Curator for pattern distillation. To effectively deal with diverse and unpredictable error types, we introduce a self-evolving template tree that systematically accumulates critique knowledge through experience-driven learning and guides future reflections. Extensive experiments have demonstrated that Table-Critic achieves substantial improvements over existing methods, achieving superior accuracy and error correction rates while maintaining computational efficiency and lower solution degradation rate. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在各种推理任务中展现出显著的能力，但在表推理任务中，它们仍然难以处理，特别是在维护多步推理过程中的一致性方面表现不佳。虽然现有的方法已经探索了各种分解策略，但它们往往缺乏有效机制来识别和纠正中间推理步骤中的错误，导致错误传播。为了应对这些问题，我们提出了Table-Critic，这是一种新颖的多智能体框架，旨在通过协作批评和迭代改进推理过程，直到收敛到正确的解。该框架包含四个专门的智能体：裁判（Judge）用于错误识别、批评家（Critic）用于全面批评、改进者（Refiner）用于过程改进以及收藏家（Curator）用于模式提炼。为了有效应对多样且难以预测的错误类型，我们引入了一个自演化模板树，通过经验驱动的学习系统地积累批评知识，并指导未来反思。广泛的经验表明，与现有方法相比，Table-Critic 在准确性和错误修正率方面取得了显著改进，同时保持了计算效率和较低的解降级率。 

---
# Cognitive-Aligned Document Selection for Retrieval-augmented Generation 

**Title (ZH)**: 认知对齐的文档选择用于检索增强生成 

**Authors**: Bingyu Wan, Fuxi Zhang, Zhongpeng Qi, Jiayi Ding, Jijun Li, Baoshi Fan, Yijia Zhang, Jun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11770)  

**Abstract**: Large language models (LLMs) inherently display hallucinations since the precision of generated texts cannot be guaranteed purely by the parametric knowledge they include. Although retrieval-augmented generation (RAG) systems enhance the accuracy and reliability of generative models by incorporating external documents, these retrieved documents often fail to adequately support the model's responses in practical applications. To address this issue, we propose GGatrieval (Fine-\textbf{G}rained \textbf{G}rounded \textbf{A}lignment Re\textbf{trieval} for verifiable generation), which leverages an LLM to dynamically update queries and filter high-quality, reliable retrieval documents. Specifically, we parse the user query into its syntactic components and perform fine-grained grounded alignment with the retrieved documents. For query components that cannot be individually aligned, we propose a dynamic semantic compensation mechanism that iteratively refines and rewrites the query while continuously updating the retrieval results. This iterative process continues until the retrieved documents sufficiently support the query's response. Our approach introduces a novel criterion for filtering retrieved documents, closely emulating human strategies for acquiring targeted information. This ensures that the retrieved content effectively supports and verifies the generated outputs. On the ALCE benchmark, our method significantly surpasses a wide range of baselines, achieving state-of-the-art performance. 

**Abstract (ZH)**: 大规模语言模型（LLMs）固有地表现出幻觉现象，因为生成的文本精确度无法仅通过它们所包含的参数化知识来完全保证。尽管检索增强生成（RAG）系统通过引入外部文档来提高生成模型的准确性和可靠性，但在实际应用中，这些检索到的文档往往无法充分支持模型的响应。为了解决这一问题，我们提出了GGatrieval（细粒度、基于证据的检索以实现可验证生成），其利用LLM动态更新查询并筛选高质量、可靠的检索文档。具体来说，我们将用户查询解析为语法组件，并与检索到的文档进行细粒度的对齐。对于无法单独对齐的查询组件，我们提出了一种动态语义补偿机制，该机制会迭代地细化并重写查询，同时不断更新检索结果。这一迭代过程将持续到检索到的文档能够充分支持查询的响应为止。我们的方法引入了一种新的检索文档筛选标准，使其紧密模拟了人类获取目标信息的策略。这确保了检索内容能够有效地支持和验证生成的输出。在ALCE基准测试中，我们的方法显著超越了多种基线方法，达到了最先进的性能。 

---
# HintsOfTruth: A Multimodal Checkworthiness Detection Dataset with Real and Synthetic Claims 

**Title (ZH)**: 《Tips of Truth：真实与合成声明的多模态可信度检测数据集》 

**Authors**: Michiel van der Meer, Pavel Korshunov, Sébastien Marcel, Lonneke van der Plas  

**Link**: [PDF](https://arxiv.org/pdf/2502.11753)  

**Abstract**: Misinformation can be countered with fact-checking, but the process is costly and slow. Identifying checkworthy claims is the first step, where automation can help scale fact-checkers' efforts. However, detection methods struggle with content that is 1) multimodal, 2) from diverse domains, and 3) synthetic. We introduce HintsOfTruth, a public dataset for multimodal checkworthiness detection with $27$K real-world and synthetic image/claim pairs. The mix of real and synthetic data makes this dataset unique and ideal for benchmarking detection methods. We compare fine-tuned and prompted Large Language Models (LLMs). We find that well-configured lightweight text-based encoders perform comparably to multimodal models but the first only focus on identifying non-claim-like content. Multimodal LLMs can be more accurate but come at a significant computational cost, making them impractical for large-scale applications. When faced with synthetic data, multimodal models perform more robustly 

**Abstract (ZH)**: 错误信息可以通过事实核查来对抗，但这一过程成本高且耗时。识别可核查的断言是第一步，自动化可以帮助扩展事实核查员的努力。然而，检测方法在处理以下类型的内容时存在困难：1）多模态内容，2）来自多种领域的内容，3）合成内容。我们介绍了一个名为 HintsOfTruth 的公开数据集，包含 27,000 对真实和合成的图像/断言对，用于多模态核查性检测。真实和合成数据的混合使该数据集独具特色，非常适合用于评估检测方法。我们比较了微调和提示的大规模语言模型（LLMs）。研究发现，配置良好的轻量级文本编码器在表现上可与多模态模型媲美，但仅专注于识别非断言性内容。多模态 LLM 可能更准确，但计算成本高昂，使其在大规模应用中不可行。在面对合成数据时，多模态模型表现出更高的鲁棒性。 

---
# Energy-Conscious LLM Decoding: Impact of Text Generation Strategies on GPU Energy Consumption 

**Title (ZH)**: 节能导向的大型语言模型解码：文本生成策略对GPU能耗的影响 

**Authors**: Alireza Nik, Michael A. Riegler, Pål Halvorsen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11723)  

**Abstract**: Decoding strategies significantly influence the quality and diversity of the generated texts in large language models (LLMs), yet their impact on computational resource consumption, particularly GPU energy usage, is insufficiently studied. This paper investigates the relationship between text generation decoding methods and energy efficiency, focusing on the trade-off between generation quality and GPU energy consumption across diverse tasks and decoding configurations. By benchmarking multiple strategies across different text generation tasks, such as Translation, Code Summarization, and Math Problem Solving, we reveal how selecting appropriate decoding techniques with their tuned hyperparameters affects text quality and has measurable implications for resource utilization, emphasizing the need for balanced optimization. To the best of our knowledge, this study is among the first to explore decoding strategies in LLMs through the lens of energy consumption, offering actionable insights for designing resource-aware applications that maintain high-quality text generation. 

**Abstract (ZH)**: 解码策略显著影响大型语言模型（LLMs）生成文本的质量和多样性，然而这些策略对其计算资源消耗，特别是在GPU能耗方面的影响研究仍然不足。本文探讨了文本生成解码方法与能源效率之间的关系，重点关注生成质量与GPU能耗之间的权衡，尤其是在各种任务和解码配置下的表现。通过在翻译、代码摘要和数学问题解决等多个文本生成任务中对比多种策略，我们揭示了选择合适的解码技术并调整其超参数如何影响文本质量，并对资源利用产生可量化的意义，强调了平衡优化的重要性。据我们所知，这是首次从能耗角度研究LLMs解码策略的研究之一，为设计兼顾资源利用和高质量文本生成的应用程序提供了可操作的见解。 

---
# VRoPE: Rotary Position Embedding for Video Large Language Models 

**Title (ZH)**: VRoPE：视频大型语言模型中的旋转位置嵌入 

**Authors**: Zikang Liu, Longteng Guo, Yepeng Tang, Junxian Cai, Kai Ma, Xi Chen, Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11664)  

**Abstract**: Rotary Position Embedding (RoPE) has shown strong performance in text-based Large Language Models (LLMs), but extending it to video remains a challenge due to the intricate spatiotemporal structure of video frames. Existing adaptations, such as RoPE-3D, attempt to encode spatial and temporal dimensions separately but suffer from two major limitations: positional bias in attention distribution and disruptions in video-text transitions. To overcome these issues, we propose Video Rotary Position Embedding (VRoPE), a novel positional encoding method tailored for Video-LLMs. Our approach restructures positional indices to preserve spatial coherence and ensure a smooth transition between video and text tokens. Additionally, we introduce a more balanced encoding strategy that mitigates attention biases, ensuring a more uniform distribution of spatial focus. Extensive experiments on Vicuna and Qwen2 across different model scales demonstrate that VRoPE consistently outperforms previous RoPE variants, achieving significant improvements in video understanding, temporal reasoning, and retrieval tasks. Code will be available at this https URL 

**Abstract (ZH)**: 旋转位置嵌入（RoPE）已经在基于文本的大型语言模型（LLMs）中展现出强大的性能，但将其扩展到视频依然面临挑战，因为视频帧具有复杂的时空结构。现有的适应方法，如RoPE-3D，试图分别编码空间和时间维度，但存在两大局限性：注意力分布中的位置偏差以及视频-文本过渡中的中断。为克服这些问题，我们提出 Video 旋转位置嵌入（VRoPE），这是一种针对视频-LLMs 的新型位置编码方法。我们的方法重新结构化位置索引，以保持空间连续性并确保视频与文本标记之间的平滑过渡。此外，我们引入了一种更均衡的编码策略，以减轻注意力偏差，确保空间焦点的更均匀分布。在不同的模型尺度上，使用Vicuna和Qwen2进行的广泛实验表明，VRoPE 一致地超越了之前的RoPE变体，在视频理解、时间推理和检索任务方面取得显著改进。代码将在以下链接提供：[此链接](this https URL)。 

---
# Competing LLM Agents in a Non-Cooperative Game of Opinion Polarisation 

**Title (ZH)**: 非合作意见极化博弈中的竞争语言模型代理 

**Authors**: Amin Qasmi, Usman Naseem, Mehwish Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2502.11649)  

**Abstract**: We introduce a novel non-cooperative game to analyse opinion formation and resistance, incorporating principles from social psychology such as confirmation bias, resource constraints, and influence penalties. Our simulation features Large Language Model (LLM) agents competing to influence a population, with penalties imposed for generating messages that propagate or counter misinformation. This framework integrates resource optimisation into the agents' decision-making process. Our findings demonstrate that while higher confirmation bias strengthens opinion alignment within groups, it also exacerbates overall polarisation. Conversely, lower confirmation bias leads to fragmented opinions and limited shifts in individual beliefs. Investing heavily in a high-resource debunking strategy can initially align the population with the debunking agent, but risks rapid resource depletion and diminished long-term influence. 

**Abstract (ZH)**: 我们提出了一种新颖的非合作博弈，用于分析意见形成和抵制行为，该博弈融合了社会心理学中的确认偏见、资源约束和影响惩罚等原则。我们的模拟中，大型语言模型（LLM）代理相互竞争，以影响人群，并对传播或反驳虚假信息的信息采取惩罚措施。该框架将资源优化融入代理的决策过程中。研究结果表明，较高的确认偏见虽然加强了群体内意见的一致性，但也加剧了整体极化。相反，较低的确认偏见会导致意见分化，并限制个人信念的转变。尽管投资高资源的驳斥策略可以初期使人群与驳斥代理保持一致，但也存在资源迅速耗尽和长期影响力减弱的风险。 

---
# A Unified Modeling Framework for Automated Penetration Testing 

**Title (ZH)**: 自动化渗透测试的统一建模框架 

**Authors**: Yunfei Wang, Shixuan Liu, Wenhao Wang, Changling Zhou, Chao Zhang, Jiandong Jin, Cheng Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11588)  

**Abstract**: The integration of artificial intelligence into automated penetration testing (AutoPT) has highlighted the necessity of simulation modeling for the training of intelligent agents, due to its cost-efficiency and swift feedback capabilities. Despite the proliferation of AutoPT research, there is a recognized gap in the availability of a unified framework for simulation modeling methods. This paper presents a systematic review and synthesis of existing techniques, introducing MDCPM to categorize studies based on literature objectives, network simulation complexity, dependency of technical and tactical operations, and scenario feedback and variation. To bridge the gap in unified method for multi-dimensional and multi-level simulation modeling, dynamic environment modeling, and the scarcity of public datasets, we introduce AutoPT-Sim, a novel modeling framework that based on policy automation and encompasses the combination of all sub dimensions. AutoPT-Sim offers a comprehensive approach to modeling network environments, attackers, and defenders, transcending the constraints of static modeling and accommodating networks of diverse scales. We publicly release a generated standard network environment dataset and the code of Network Generator. By integrating publicly available datasets flexibly, support is offered for various simulation modeling levels focused on policy automation in MDCPM and the network generator help researchers output customized target network data by adjusting parameters or fine-tuning the network generator. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

将人工智能整合到自动化渗透测试（AutoPT）中，强调了模拟建模在智能代理培训中的必要性，因为模拟建模具有成本效益和快速反馈的优势。尽管自动渗透测试（AutoPT）的研究层出不穷，但统一的模拟建模方法仍然存在不足。本文提出了一项系统的文献综述和综合分析，引入MDCPM（多维度复杂渗透测试建模框架）来根据文献目标、网络模拟复杂性、技术和战术操作的依赖性以及情景反馈和变化，对现有技术进行分类。为了弥合多维度和多级模拟建模、动态环境建模以及缺乏公开数据集的统一方法上的缺口，我们提出了基于策略自动化、涵盖所有子维度的AutoPT-Sim（自动化渗透测试模拟）框架。AutoPT-Sim提供了一个全面的网络环境、攻击者和防御者建模方法，超越了静态建模的限制，适用于各种规模的网络。我们公开发布了标准的网络环境数据集和网络生成器的代码。通过灵活地整合公开可用的数据集，AutoPT-Sim支持根据MDCPM中的策略自动化在不同建模层次上的应用，并且网络生成器可以帮助研究人员通过调整参数或精细调整网络生成器来输出定制的目标网络数据。 

---
# Calibration of Vehicular Traffic Simulation Models by Local Optimization 

**Title (ZH)**: 通过局部优化校准车辆交通仿真模型 

**Authors**: Davide Andrea Guastella, Alejandro Morales-Hernàndez, Bruno Cornelis, Gianluca Bontempi  

**Link**: [PDF](https://arxiv.org/pdf/2502.11585)  

**Abstract**: Simulation is a valuable tool for traffic management experts to assist them in refining and improving transportation systems and anticipating the impact of possible changes in the infrastructure network before their actual implementation. Calibrating simulation models using traffic count data is challenging because of the complexity of the environment, the lack of data, and the uncertainties in traffic dynamics. This paper introduces a novel stochastic simulation-based traffic calibration technique. The novelty of the proposed method is: (i) it performs local traffic calibration, (ii) it allows calibrating simulated traffic in large-scale environments, (iii) it requires only the traffic count data. The local approach enables decentralizing the calibration task to reach near real-time performance, enabling the fostering of digital twins. Using only traffic count data makes the proposed method generic so that it can be applied in different traffic scenarios at various scales (from neighborhood to region). We assess the proposed technique on a model of Brussels, Belgium, using data from real traffic monitoring devices. The proposed method has been implemented using the open-source traffic simulator SUMO. Experimental results show that the traffic model calibrated using the proposed method is on average 16% more accurate than those obtained by the state-of-the-art methods, using the same dataset. We also make available the output traffic model obtained from real data. 

**Abstract (ZH)**: 模拟是交通管理专家的重要工具，可以帮助他们在实际实施之前通过优化和改进交通系统来细化并预见基础设施网络可能变化的影响。使用交通量数据校准模拟模型具有挑战性，因为环境复杂、数据缺失以及交通动态的不确定性。本文介绍了一种新颖的基于随机模拟的交通校准技术。所提出方法的创新之处在于：（i）它进行局部交通校准；（ii）它允许在大规模环境中校准模拟交通；（iii）它只需要交通量数据。局部方法使校准任务能够分散处理，从而实现接近实时性能，并促进数字孪生的发展。仅使用交通量数据使提出的方法具有通用性，适用于不同规模（从邻里到地区）的各种交通场景。我们使用来自实际交通监控设备的数据，在布鲁塞尔（比利时）的模型上评估了提出的技巧。本文使用开源交通模拟器SUMO实现了所提出的算法。实验结果表明，利用所提方法校准的交通模型相对于最先进的方法，在相同数据集上的平均准确性提升了16%。同时，我们还提供了从实际数据获得的输出交通模型。 

---
# Large Language Models and Mathematical Reasoning Failures 

**Title (ZH)**: 大型语言模型在数学推理中的失败表现 

**Authors**: Johan Boye, Birger Moell  

**Link**: [PDF](https://arxiv.org/pdf/2502.11574)  

**Abstract**: This paper investigates the mathematical reasoning capabilities of large language models (LLMs) using 50 newly constructed high-school-level word problems. Unlike prior studies that focus solely on answer correctness, we rigorously analyze both final answers and solution steps to identify reasoning failures. Evaluating eight state-of-the-art models - including Mixtral, Llama, Gemini, GPT-4o, and OpenAI's o1 variants - we find that while newer models (e.g., o3-mini, deepseek-r1) achieve higher accuracy, all models exhibit errors in spatial reasoning, strategic planning, and arithmetic, sometimes producing correct answers through flawed logic. Common failure modes include unwarranted assumptions, over-reliance on numerical patterns, and difficulty translating physical intuition into mathematical steps. Manual analysis reveals that models struggle with problems requiring multi-step deduction or real-world knowledge, despite possessing broad mathematical knowledge. Our results underscore the importance of evaluating reasoning processes, not just answers, and caution against overestimating LLMs' problem-solving proficiency. The study highlights persistent gaps in LLMs' generalization abilities, emphasizing the need for targeted improvements in structured reasoning and constraint handling. 

**Abstract (ZH)**: 本文使用50个新构建的高中级别文字题，探讨了大型语言模型（LLMs）的数学推理能力。不同于以往研究仅关注答案的正确性，我们对最终答案和解题步骤进行了严格的分析，以识别推理中的失败。评估了八种最先进的模型——包括Mixtral、Llama、Gemini、GPT-4o以及OpenAI的o1变体——结果发现，尽管较新的模型（例如o3-mini、deepseek-r1）获得了更高的准确性，但所有模型在空间推理、战略规划和算术方面都存在错误，有时通过错误的逻辑反而能得出正确的答案。常见的失败模式包括不合理的假设、过度依赖数字模式，以及难以将物理直觉转化为数学步骤。人工分析显示，尽管这些模型拥有广泛的数学知识，但在需要多步推理或现实世界知识的问题上仍存在困难。研究结果强调了评估推理过程的重要性，而不是仅仅关注答案，并提醒我们不要高估LLMs在问题解决方面的能力。该研究突显了LLMs在泛化能力方面的持续缺口，强调了在结构化推理和约束处理方面进行针对性改进的必要性。 

---
# A Survey of Automatic Prompt Engineering: An Optimization Perspective 

**Title (ZH)**: 自动提示工程综述：从优化视角探讨 

**Authors**: Wenwu Li, Xiangfeng Wang, Wenhao Li, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.11560)  

**Abstract**: The rise of foundation models has shifted focus from resource-intensive fine-tuning to prompt engineering, a paradigm that steers model behavior through input design rather than weight updates. While manual prompt engineering faces limitations in scalability, adaptability, and cross-modal alignment, automated methods, spanning foundation model (FM) based optimization, evolutionary methods, gradient-based optimization, and reinforcement learning, offer promising solutions. Existing surveys, however, remain fragmented across modalities and methodologies. This paper presents the first comprehensive survey on automated prompt engineering through a unified optimization-theoretic lens. We formalize prompt optimization as a maximization problem over discrete, continuous, and hybrid prompt spaces, systematically organizing methods by their optimization variables (instructions, soft prompts, exemplars), task-specific objectives, and computational frameworks. By bridging theoretical formulation with practical implementations across text, vision, and multimodal domains, this survey establishes a foundational framework for both researchers and practitioners, while highlighting underexplored frontiers in constrained optimization and agent-oriented prompt design. 

**Abstract (ZH)**: 基础模型的兴起将研究重点从资源密集型的微调转向提示工程，这是一种通过输入设计而非权重更新来引导模型行为的范式。虽然手动提示工程在可扩展性、适应性和跨模态对齐方面存在局限性，但涵盖基础模型（FM）优化、进化方法、梯度基优化和强化学习的自动化方法则提供了有前景的解决方案。然而，现有综述仍呈碎片化状态，分布在不同的模态和方法论上。本文通过统一的优化理论视角首次对自动化提示工程进行了全面综述。我们将提示优化形式化为离散、连续和混合提示空间上的最大化问题，并系统地按优化变量（指令、软提示、示例）、任务特定目标和计算框架对方法进行分类。通过将理论建模与文本、视觉和多模态领域的实践实施相结合，本综述为研究人员和实践者提供了基础框架，同时指出了约束优化和面向代理的提示设计中未充分探索的前沿领域。 

---
# Equilibrate RLHF: Towards Balancing Helpfulness-Safety Trade-off in Large Language Models 

**Title (ZH)**: 平衡RLHF：在大规模语言模型中实现帮助性与安全性的权衡稳定策略 

**Authors**: Yingshui Tan, Yilei Jiang, Yanshi Li, Jiaheng Liu, Xingyuan Bu, Wenbo Su, Xiangyu Yue, Xiaoyong Zhu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.11555)  

**Abstract**: Fine-tuning large language models (LLMs) based on human preferences, commonly achieved through reinforcement learning from human feedback (RLHF), has been effective in improving their performance. However, maintaining LLM safety throughout the fine-tuning process remains a significant challenge, as resolving conflicts between safety and helpfulness can be non-trivial. Typically, the safety alignment of LLM is trained on data with safety-related categories. However, our experiments find that naively increasing the scale of safety training data usually leads the LLMs to an ``overly safe'' state rather than a ``truly safe'' state, boosting the refusal rate through extensive safety-aligned data without genuinely understanding the requirements for safe responses. Such an approach can inadvertently diminish the models' helpfulness. To understand the phenomenon, we first investigate the role of safety data by categorizing them into three different groups, and observe that each group behaves differently as training data scales up. To boost the balance between safety and helpfulness, we propose an Equilibrate RLHF framework including a Fine-grained Data-centric (FDC) approach that achieves better safety alignment even with fewer training data, and an Adaptive Message-wise Alignment (AMA) approach, which selectively highlight the key segments through a gradient masking strategy. Extensive experimental results demonstrate that our approach significantly enhances the safety alignment of LLMs while balancing safety and helpfulness. 

**Abstract (ZH)**: 基于人类偏好的微调大型语言模型（LLMs），通常通过人类反馈强化学习（RLHF）实现，已被证明能够有效提高模型性能。然而，在微调过程中保持LLM的安全性仍然是一个重大挑战，因为解决安全性和帮助性之间的冲突可能并非易事。通常，LLM的安全对齐是在包含安全相关类别的数据上进行训练的。然而，我们的实验表明，盲目增加安全训练数据的规模通常会使LLM进入一个“过度安全”的状态，而不是一个“真正安全”的状态，这会通过广泛的安全对齐数据提高拒绝率，而未能真正理解安全响应的要求，从而意外地降低了模型的帮助性。为了理解这种现象，我们首先通过将安全数据分为三类来研究其作用，并观察到随着训练数据规模的扩大，每组数据的行为不同。为了平衡安全性和帮助性，我们提出了一种均衡RLHF框架，包括一种细粒度数据为中心的方法（FDC），即使使用较少的训练数据也能更好地实现安全对齐，以及一种适应性消息级对齐（AMA）方法，该方法通过梯度屏蔽策略突出关键段落。广泛的实验结果表明，我们的方法显著增强了LLM的安全对齐，同时平衡了安全性和帮助性。 

---
# A Survey of Personalized Large Language Models: Progress and Future Directions 

**Title (ZH)**: 个性化大型语言模型综述：进展与未来方向 

**Authors**: Jiahong Liu, Zexuan Qiu, Zhongyang Li, Quanyu Dai, Jieming Zhu, Minda Hu, Menglin Yang, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2502.11528)  

**Abstract**: Large Language Models (LLMs) excel in handling general knowledge tasks, yet they struggle with user-specific personalization, such as understanding individual emotions, writing styles, and preferences. Personalized Large Language Models (PLLMs) tackle these challenges by leveraging individual user data, such as user profiles, historical dialogues, content, and interactions, to deliver responses that are contextually relevant and tailored to each user's specific needs. This is a highly valuable research topic, as PLLMs can significantly enhance user satisfaction and have broad applications in conversational agents, recommendation systems, emotion recognition, medical assistants, and more. This survey reviews recent advancements in PLLMs from three technical perspectives: prompting for personalized context (input level), finetuning for personalized adapters (model level), and alignment for personalized preferences (objective level). To provide deeper insights, we also discuss current limitations and outline several promising directions for future research. Updated information about this survey can be found at the this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在处理一般知识任务方面表现出色，但在处理用户特定的个性化需求方面存在困难，如理解个人情感、写作风格和偏好。个性化大型语言模型（PLLMs）通过利用个体用户数据（如用户档案、历史对话、内容和互动），提供与用户上下文相关且符合其特定需求的响应，来应对这些挑战。这一研究领域极具价值，因为PLLMs能够显著提升用户体验并在对话代理、推荐系统、情绪识别、医疗辅助等领域拥有广泛的应用前景。本文综述了从三个技术视角来看的PLLMs的最新进展：个性化上下文提示（输入层面）、个性化适配器微调（模型层面）和个性化偏好对齐（目标层面）。此外，本文还讨论了当前的局限性，并提出了未来研究的若干有前途的方向。有关本文综述的最新信息，请参阅 [此处](this https URL)。 

---
# Why Vision Language Models Struggle with Visual Arithmetic? Towards Enhanced Chart and Geometry Understanding 

**Title (ZH)**: 视觉语言模型为何在视觉算术任务中表现不佳？向着增强的图表和几何理解努力 

**Authors**: Kung-Hsiang Huang, Can Qin, Haoyi Qiu, Philippe Laban, Shafiq Joty, Caiming Xiong, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11492)  

**Abstract**: Vision Language Models (VLMs) have achieved remarkable progress in multimodal tasks, yet they often struggle with visual arithmetic, seemingly simple capabilities like object counting or length comparison, which are essential for relevant complex tasks like chart understanding and geometric reasoning. In this work, we first investigate the root causes of this deficiency through a suite of probing tasks focusing on basic visual arithmetic. Our analysis reveals that while pre-trained vision encoders typically capture sufficient information, the text decoder often fails to decode it correctly for arithmetic reasoning. To address this, we propose CogAlign, a novel post-training strategy inspired by Piaget's theory of cognitive development. CogAlign trains VLMs to recognize invariant properties under visual transformations. We demonstrate that this approach significantly improves the performance of three diverse VLMs on our proposed probing tasks. Furthermore, CogAlign enhances performance by an average of 4.6% on CHOCOLATE and 2.9% on MATH-VISION, outperforming or matching supervised fine-tuning methods while requiring only 60% less training data. These results highlight the effectiveness and generalizability of CogAlign in improving fundamental visual arithmetic capabilities and their transfer to downstream tasks. 

**Abstract (ZH)**: 视觉语言模型（VLMs）在多模态任务中取得了显著进展，但在视觉算术等看似简单的任务，如物体计数或长度比较方面经常表现出色欠缺。这些能力对于相关复杂的任务，如图表理解和几何推理是至关重要的。在本文中，我们首先通过一系列针对基本视觉算术的专业探究任务来调查这种缺陷的根本原因。我们的分析显示，预训练的视觉编码器通常能够捕获足够的信息，但文本解码器常常无法正确地对其进行算术推理的解码。为了解决这一问题，我们提出了CogAlign，这是一种受皮亚杰认知发展理论启发的新型后训练策略。CogAlign旨在训练VLMs识别视觉变换下的不变属性。我们展示了此方法在我们提出的专业探究任务上显著提高了三种不同VLMs的性能。此外，CogAlign在CHOCOLATE上的性能平均提高了4.6%，在MATH-VISION上的性能提高了2.9%，并且仅需较少训练数据（比监督微调方法少60%）就能匹配或超越监督微调方法的性能。这些结果突显了CogAlign在提高基本视觉算术能力及其向下游任务的迁移方面的有效性和普遍性。 

---
# AGrail: A Lifelong Agent Guardrail with Effective and Adaptive Safety Detection 

**Title (ZH)**: AGrail：一种有效的自适应安全检测终身代理防护栏 

**Authors**: Weidi Luo, Shenghong Dai, Xiaogeng Liu, Suman Banerjee, Huan Sun, Muhao Chen, Chaowei Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2502.11448)  

**Abstract**: The rapid advancements in Large Language Models (LLMs) have enabled their deployment as autonomous agents for handling complex tasks in dynamic environments. These LLMs demonstrate strong problem-solving capabilities and adaptability to multifaceted scenarios. However, their use as agents also introduces significant risks, including task-specific risks, which are identified by the agent administrator based on the specific task requirements and constraints, and systemic risks, which stem from vulnerabilities in their design or interactions, potentially compromising confidentiality, integrity, or availability (CIA) of information and triggering security risks. Existing defense agencies fail to adaptively and effectively mitigate these risks. In this paper, we propose AGrail, a lifelong agent guardrail to enhance LLM agent safety, which features adaptive safety check generation, effective safety check optimization, and tool compatibility and flexibility. Extensive experiments demonstrate that AGrail not only achieves strong performance against task-specific and system risks but also exhibits transferability across different LLM agents' tasks. 

**Abstract (ZH)**: 大型语言模型（LLMs）的快速进步使其能够作为自主代理部署，以处理动态环境中的复杂任务。这些LLMs展示了强大的问题解决能力和对多样化情境的适应性。然而，将它们用作代理也带来了显著的风险，包括特定任务风险和系统风险。特定任务风险由代理管理员根据具体任务需求和约束来识别，而系统风险则源自设计或交互中的漏洞，可能导致信息的保密性、完整性和可用性（CI A）受到威胁，并引发安全风险。目前现有的防御机构无法适应性且有效地减轻这些风险。本文提出了一种生命周期代理护栏（AGrail），以增强LLM代理的安全性，其特点包括适应性安全检查生成、有效安全检查优化以及工具的兼容性和灵活性。广泛实验表明，AGrail不仅在对抗特定任务风险和系统风险方面表现出色，还具有在不同类型LLM代理任务之间的泛化能力。 

---
# SMART: Self-Aware Agent for Tool Overuse Mitigation 

**Title (ZH)**: SMART：自我感知代理以减轻工具使用过度问题 

**Authors**: Cheng Qian, Emre Can Acikgoz, Hongru Wang, Xiusi Chen, Avirup Sil, Dilek Hakkani-Tür, Gokhan Tur, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.11435)  

**Abstract**: Current Large Language Model (LLM) agents demonstrate strong reasoning and tool use capabilities, but often lack self-awareness, failing to balance these approaches effectively. This imbalance leads to Tool Overuse, where models unnecessarily rely on external tools for tasks solvable with parametric knowledge, increasing computational overhead. Inspired by human metacognition, we introduce SMART (Strategic Model-Aware Reasoning with Tools), a paradigm that enhances an agent's self-awareness to optimize task handling and reduce tool overuse. To support this paradigm, we introduce SMART-ER, a dataset spanning three domains, where reasoning alternates between parametric knowledge and tool-dependent steps, with each step enriched by rationales explaining when tools are necessary. Through supervised training, we develop SMARTAgent, a family of models that dynamically balance parametric knowledge and tool use. Evaluations show that SMARTAgent reduces tool use by 24% while improving performance by over 37%, enabling 7B-scale models to match its 70B counterpart and GPT-4o. Additionally, SMARTAgent generalizes to out-of-distribution test data like GSM8K and MINTQA, maintaining accuracy with just one-fifth the tool calls. These highlight the potential of strategic tool use to enhance reasoning, mitigate overuse, and bridge the gap between model size and performance, advancing intelligent and resource-efficient agent designs. 

**Abstract (ZH)**: 当前的大语言模型（LLM）代理展示了强大的推理和工具使用能力，但往往缺乏自我意识，无法有效地平衡这些能力。这种不平衡导致了工具滥用问题，即模型在解决可通过参数化知识完成的任务时，无必要地依赖外部工具，从而增加了计算开销。受人类元认知的启发，我们引入了SMART（Strategic Model-Aware Reasoning with Tools）范式，以增强代理的自我意识，优化任务处理并减少工具滥用。为了支持这一范式，我们引入了SMART-ER数据集，该数据集跨越了三个领域，在推理过程中交替使用参数化知识和工具依赖步骤，并在每个步骤中加入理据解释工具何时是必要的信息。通过监督训练，我们开发了SMARTAgent这一模型系列，能够动态平衡参数化知识和工具使用。评估结果表明，SMARTAgent在工具使用上减少了24%，同时性能提高了超过37%，使其70亿参数规模的对标模型和GPT-4o能够保持竞争力。此外，SMARTAgent还能将这种效益推广到包括GSM8K和MINTQA在内的分布外测试数据集，仅需五分之一的工具调用就实现了相同水平的准确性。这些结果突显了战略性工具使用在增强推理、抑制滥用、弥合模型规模与性能差距方面的潜力，促进了智能和资源高效代理设计的发展。 

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
# Mimicking the Familiar: Dynamic Command Generation for Information Theft Attacks in LLM Tool-Learning System 

**Title (ZH)**: 模仿熟悉的操作：在大规模语言模型工具学习系统中进行信息窃取攻击的动态命令生成 

**Authors**: Ziyou Jiang, Mingyang Li, Guowei Yang, Junjie Wang, Yuekai Huang, Zhiyuan Chang, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11358)  

**Abstract**: Information theft attacks pose a significant risk to Large Language Model (LLM) tool-learning systems. Adversaries can inject malicious commands through compromised tools, manipulating LLMs to send sensitive information to these tools, which leads to potential privacy breaches. However, existing attack approaches are black-box oriented and rely on static commands that cannot adapt flexibly to the changes in user queries and the invocation chain of tools. It makes malicious commands more likely to be detected by LLM and leads to attack failure. In this paper, we propose AutoCMD, a dynamic attack comment generation approach for information theft attacks in LLM tool-learning systems. Inspired by the concept of mimicking the familiar, AutoCMD is capable of inferring the information utilized by upstream tools in the toolchain through learning on open-source systems and reinforcement with target system examples, thereby generating more targeted commands for information theft. The evaluation results show that AutoCMD outperforms the baselines with +13.2% $ASR_{Theft}$, and can be generalized to new tool-learning systems to expose their information leakage risks. We also design four defense methods to effectively protect tool-learning systems from the attack. 

**Abstract (ZH)**: 信息盗窃攻击对大型语言模型（LLM）工具学习系统构成了显著的风险。攻击者可以通过被破坏的工具注入恶意命令，操控LLM将敏感信息发送到这些工具，从而可能导致隐私泄露。然而，现有的攻击方法主要是黑盒导向的，并依赖于静态命令，这些命令无法灵活适应用户查询和工具调用链的变化，使得恶意命令更易被LLM检测到，导致攻击失败。本文提出了一种名为AutoCMD的动态攻击命令生成方法，专用于LLM工具学习系统的信息盗窃攻击。AutoCMD借鉴了模仿熟悉事物的概念，通过在开源系统上学习并结合目标系统示例进行强化学习，以推断工具链中上游工具所利用的信息，从而生成更具针对性的盗窃命令。实验结果显示，AutoCMD在信息盗窃攻击成功的比率（$ASR_{Theft}$）方面优于基线模型13.2%，并且可以泛化到新的工具学习系统，揭示其信息泄露风险。同时，我们还设计了四种防护方法，以有效保护工具学习系统免受此类攻击的影响。 

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
# Game-Of-Goals: Using adversarial games to achieve strategic resilience 

**Title (ZH)**: 《目标博弈：利用对抗博弈实现战略韧性》 

**Authors**: Aditya Ghose, Asjad Khan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11295)  

**Abstract**: Our objective in this paper is to develop a machinery that makes a given organizational strategic plan resilient to the actions of competitor agents (adverse environmental actions). We assume that we are given a goal tree representing strategic goals (can also be seen business requirements for a software systems) with the assumption that competitor agents are behaving in a maximally adversarial fashion(opposing actions against our sub goals or goals in general). We use game tree search methods (such as minimax) to select an optimal execution strategy(at a given point in time), such that it can maximize our chances of achieving our (high level) strategic goals. Our machinery helps us determine which path to follow(strategy selection) to achieve the best end outcome. This is done by comparing alternative execution strategies available to us via an evaluation function. Our evaluation function is based on the idea that we want to make our execution plans defensible(future-proof) by selecting execution strategies that make us least vulnerable to adversarial actions by the competitor agents. i.e we want to select an execution strategy such that its leaves minimum room(or options) for the adversary to cause impediment/damage to our business goals/plans. 

**Abstract (ZH)**: 本文的目标是开发一种机制，使给定组织的战略计划能够抵御竞争对手代理（或最不利的环境行动）的行动。我们假设已经给出了一个目标树，代表战略目标（也可以视为软件系统的业务需求），并且假定竞争对手代理以最具敌意的方式行动（反对我们的子目标或一般目标）。我们使用博弈树搜索方法（如极大极小值法）来选择在特定时间点的最佳执行策略，以最大化实现我们（高层次）战略目标的机会。我们的机制帮助我们确定应选择哪条路径（策略选择）以实现最佳最终结果。这通过一个评估函数比较可供我们选择的替代执行策略来实现。我们的评估函数基于这样的理念，即通过选择使我们最少受到竞争对手代理敌对行动影响的执行策略，使执行计划具有防御性（未来导向）。换句话说，我们希望选择一个执行策略，使其给竞争对手留下的造成阻碍或损害我们业务目标/计划的可能性最小。 

---
# Dialogue-based Explanations for Logical Reasoning using Structured Argumentation 

**Title (ZH)**: 基于对话的结构化论辩推理解释 

**Authors**: Loan Ho, Stefan Schlobach  

**Link**: [PDF](https://arxiv.org/pdf/2502.11291)  

**Abstract**: The problem of explaining inconsistency-tolerant reasoning in knowledge bases (KBs) is a prominent topic in Artificial Intelligence (AI). While there is some work on this problem, the explanations provided by existing approaches often lack critical information or fail to be expressive enough for non-binary conflicts. In this paper, we identify structural weaknesses of the state-of-the-art and propose a generic argumentation-based approach to address these problems. This approach is defined for logics involving reasoning with maximal consistent subsets and shows how any such logic can be translated to argumentation. Our work provides dialogue models as dialectic-proof procedures to compute and explain a query answer wrt inconsistency-tolerant semantics. This allows us to construct dialectical proof trees as explanations, which are more expressive and arguably more intuitive than existing explanation formalisms. 

**Abstract (ZH)**: 在知识库（KBs）中解释容错推理的问题是人工智能（AI）领域的一个重要课题。尽管已有部分相关研究工作，但现有方法提供的解释往往缺乏关键信息，或者无法充分表达非二元冲突。本文中，我们识别了现有方法的结构性弱点，并提出了一种基于论证的方法来解决这些问题。这种方法适用于涉及最大一致子集推理的逻辑系统，并展示了如何将任何这样的逻辑系统转化为论证系统。我们的工作提供了一种对话模型，作为辩证反驳程序，用于计算并解释基于容错语义的查询答案。这使得我们可以构建辩证证明树作为解释，这些解释更为丰富且更具有直观性，比现有的解释形式更胜一筹。 

---
# Unlocking the Potential of Generative AI through Neuro-Symbolic Architectures: Benefits and Limitations 

**Title (ZH)**: 通过神经符号架构解锁生成式AI的潜力：优势与限制 

**Authors**: Oualid Bougzime, Samir Jabbar, Christophe Cruz, Frédéric Demoly  

**Link**: [PDF](https://arxiv.org/pdf/2502.11269)  

**Abstract**: Neuro-symbolic artificial intelligence (NSAI) represents a transformative approach in artificial intelligence (AI) by combining deep learning's ability to handle large-scale and unstructured data with the structured reasoning of symbolic methods. By leveraging their complementary strengths, NSAI enhances generalization, reasoning, and scalability while addressing key challenges such as transparency and data efficiency. This paper systematically studies diverse NSAI architectures, highlighting their unique approaches to integrating neural and symbolic components. It examines the alignment of contemporary AI techniques such as retrieval-augmented generation, graph neural networks, reinforcement learning, and multi-agent systems with NSAI paradigms. This study then evaluates these architectures against comprehensive set of criteria, including generalization, reasoning capabilities, transferability, and interpretability, therefore providing a comparative analysis of their respective strengths and limitations. Notably, the Neuro > Symbolic < Neuro model consistently outperforms its counterparts across all evaluation metrics. This result aligns with state-of-the-art research that highlight the efficacy of such architectures in harnessing advanced technologies like multi-agent systems. 

**Abstract (ZH)**: 神经符号人工智能（NSAI）代表了人工智能（AI）领域的一种变革性方法，它将深度学习处理大规模和非结构化数据的能力与符号方法的结构化推理相结合。通过发挥两者的互补优势，NSAI 有助于提高泛化能力、推理能力和可扩展性，并解决透明性和数据效率等关键挑战。本文系统研究了各种 NSAI 架构，强调了它们在结合神经和符号组件方面的独特方法。该研究探讨了诸如检索增强生成、图神经网络、强化学习和多智能体系统等当前AI技术与NSAI范式的契合度。随后，该研究依据包括泛化能力、推理能力、迁移能力和可解释性在内的全面评价标准，评估了这些架构的优劣，从而提供了它们各自优势和局限性的比较分析。值得注意的是，Neuro > Symbolic < Neuro 模型在所有评价指标中均表现出色。这一结果与最先进的研究成果一致，后者强调了此类架构在利用多智能体系统等先进技术方面的有效性。 

---
# Explaining Necessary Truths 

**Title (ZH)**: 解释必要的真理 

**Authors**: Gülce Kardeş, Simon DeDeo  

**Link**: [PDF](https://arxiv.org/pdf/2502.11251)  

**Abstract**: Knowing the truth is rarely enough -- we also seek out reasons why the fact is true. While much is known about how we explain contingent truths, we understand less about how we explain facts, such as those in mathematics, that are true as a matter of logical necessity. We present a framework, based in computational complexity, where explanations for deductive truths co-emerge with discoveries of simplifying steps during the search process. When such structures are missing, we revert, in turn, to error-based reasons, where a (corrected) mistake can serve as fictitious, but explanatory, contingency-cause: not making the mistake serves as a reason why the truth takes the form it does. We simulate human subjects, using GPT-4o, presented with SAT puzzles of varying complexity and reasonableness, validating our theory and showing how its predictions can be tested in future human studies. 

**Abstract (ZH)**: 了解真相通常远远不够——我们还寻求解释这一事实为何成立的原因。虽然我们已经相当了解在解释偶然性真理时人们是如何进行解释的，但我们对如何解释那些作为逻辑必然性事实的真理，理解得还不够深入。本文提出了一种基于计算复杂性的框架，其中在搜索过程中解释演绎真理与简化步骤的发现同时出现。当这些结构不存在时，我们则转向基于误差的解释，其中纠正后的错误可以作为一个虚构但具有解释性的偶然原因：不犯错可以作为解释为何事实呈现出这种特定形式的理由。我们使用GPT-4o模拟人类被试，针对不同复杂性和合理性的SAT谜题进行测试，以验证我们的理论，并展示未来人类研究中如何测试其预测。 

---
# PlanGenLLMs: A Modern Survey of LLM Planning Capabilities 

**Title (ZH)**: PlanGenLLMs：大型语言模型的现代规划能力综述 

**Authors**: Hui Wei, Zihao Zhang, Shenghua He, Tian Xia, Shijia Pan, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11221)  

**Abstract**: LLMs have immense potential for generating plans, transforming an initial world state into a desired goal state. A large body of research has explored the use of LLMs for various planning tasks, from web navigation to travel planning and database querying. However, many of these systems are tailored to specific problems, making it challenging to compare them or determine the best approach for new tasks. There is also a lack of clear and consistent evaluation criteria. Our survey aims to offer a comprehensive overview of current LLM planners to fill this gap. It builds on foundational work by Kartam and Wilkins (1990) and examines six key performance criteria: completeness, executability, optimality, representation, generalization, and efficiency. For each, we provide a thorough analysis of representative works and highlight their strengths and weaknesses. Our paper also identifies crucial future directions, making it a valuable resource for both practitioners and newcomers interested in leveraging LLM planning to support agentic workflows. 

**Abstract (ZH)**: 大规模语言模型（LLMs）具有生成计划的巨大潜力，能够将初始世界状态转变为期望的目标状态。大量研究探讨了LLMs在各种规划任务中的应用，从网页导航到旅行规划和数据库查询等。然而，许多现有的系统针对特定问题进行了定制，这使得比较这些系统或确定处理新任务的最佳方法变得具有挑战性。此外，还没有明确且一致的评估标准。我们的综述旨在提供当前LLM规划的全面概述，以填补这一空白。它基于Kartam和Wilkins（1990）的基础工作，并且考察了六个关键性能标准：完备性、可行性、最优性、表示性、泛化能力和效率。对于每一个标准，我们都进行了详细的分析，并指出了代表性工作的优缺点。此外，我们还指出了未来研究的关键方向，这使得本文成为从业者和希望利用LLM规划支持主动流程的新手的重要资源。 

---
# Quantifying the Capability Boundary of DeepSeek Models: An Application-Driven Performance Analysis 

**Title (ZH)**: 基于应用驱动性能分析的DeepSeek模型能力边界量化 

**Authors**: Shiguo Lian, Kaikai Zhao, Xuejiao Lei, Ning Wang, Zhenhong Long, Peijun Yang, Minjie Hua, Chaoyang Ma, Wen Liu, Kai Wang, Zhaoxiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11164)  

**Abstract**: DeepSeek-R1, known for its low training cost and exceptional reasoning capabilities, has achieved state-of-the-art performance on various benchmarks. However, detailed evaluations from the perspective of real-world applications are lacking, making it challenging for users to select the most suitable DeepSeek models for their specific needs. To address this gap, we evaluate the DeepSeek-V3, DeepSeek-R1, DeepSeek-R1-Distill-Qwen series, and DeepSeek-R1-Distill-Llama series on A-Eval, an application-driven benchmark. By comparing original instruction-tuned models with their distilled counterparts, we analyze how reasoning enhancements impact performance across diverse practical tasks. Our results show that reasoning-enhanced models, while generally powerful, do not universally outperform across all tasks, with performance gains varying significantly across tasks and models. To further assist users in model selection, we quantify the capability boundary of DeepSeek models through performance tier classifications and intuitive line charts. Specific examples provide actionable insights to help users select and deploy the most cost-effective DeepSeek models, ensuring optimal performance and resource efficiency in real-world applications. 

**Abstract (ZH)**: DeepSeek-R1 因其低训练成本和卓越的推理能力而闻名，已经在多种基准测试中取得了最先进的性能。然而，从实际应用的角度进行详细评估仍然缺乏，这使得用户在选择最合适的 DeepSeek 模型时面临挑战。为了解决这一问题，我们在 A-Eval 任务驱动基准上评估了 DeepSeek-V3、DeepSeek-R1、DeepSeek-R1-Distill-Qwen 系列和 DeepSeek-R1-Distill-Llama 系列。通过将原始指令调优模型与其精简版本进行比较，我们分析了推理能力增强如何影响各种实际任务中的性能。我们的结果表明，尽管推理增强模型通常非常强大，但在所有任务中并不总是表现出最佳性能，不同任务和模型之间的性能提升差异显著。为进一步帮助用户进行模型选择，我们通过性能等级分类和直观的线形图量化 DeepSeek 模型的能力边界。通过具体示例，我们为用户提供了一些可操作的见解，以帮助他们选择和部署最具成本效益的 DeepSeek 模型，从而确保在实际应用中的最佳性能和资源效率。 

---
# Dyve: Thinking Fast and Slow for Dynamic Process Verification 

**Title (ZH)**: Dyve：快速与缓慢思维在动态过程验证中的应用 

**Authors**: Jianyuan Zhong, Zeju Li, Zhijian Xu, Xiangyu Wen, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11157)  

**Abstract**: We present Dyve, a dynamic process verifier that enhances reasoning error detection in large language models by integrating fast and slow thinking, inspired by Kahneman's Systems Theory. Dyve adaptively applies immediate token-level confirmation System 1 for straightforward steps and comprehensive analysis System 2 for complex ones. Leveraging a novel step-wise consensus-filtered process supervision technique, combining Monte Carlo estimation with LLM based evaluation, Dyve curates high-quality supervision signals from noisy data. Experimental results on ProcessBench and the MATH dataset confirm that Dyve significantly outperforms existing process-based verifiers and boosts performance in Best-of-N settings. 

**Abstract (ZH)**: 我们呈现了Dyve，这是一种动态过程验证器，通过集成快速和慢速思考，增强大型语言模型中的推理错误检测，灵感来源于Kahneman的系统理论。Dyve根据步骤的复杂程度，自适应地应用即时的基于token的验证（System 1）进行简单步骤的确认，以及全面分析（System 2）进行复杂步骤的验证。借助一种新颖的分步骤共识过滤过程监督技术，结合蒙特卡洛估计与基于LLM的评估，Dyve从嘈杂的数据中生成高质量的监督信号。在ProcessBench和MATH数据集上的实验结果表明，Dyve显著优于现有的基于过程的验证器，并在Best-of-N设置中提升了性能。 

---
# Uncertainty-Aware Search and Value Models: Mitigating Search Scaling Flaws in LLMs 

**Title (ZH)**: aware搜索和价值模型：减轻大语言模型中搜索扩展方面的缺陷 

**Authors**: Fei Yu, Yingru Li, Benyou Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11155)  

**Abstract**: Value model-guided search is effective in steering the generation but suffers from scaling flaws: Its superiority diminishes with larger sample sizes, underperforming non-search baselines. This limitation arises from reliability degradation in value models in unseen reasoning paths. To address this, we propose an uncertainty-aware search framework that includes two key components: (1) uncertainty-aware value models that incorporate uncertainty into predictions, and (2) an uncertainty-aware selection process using the proposed efficient Group Thompson Sampling algorithm. Experiments on GSM8K show that our method mitigates search scaling flaws, achieving 90.5% coverage at 16 samples compared to 85.8% for conventional value-guided search. This work establishes the first systematic integration of uncertainty quantification in LLM search paradigms. 

**Abstract (ZH)**: 基于价值模型的搜索方法在引导生成方面是有效的，但在扩展性方面存在缺陷：其优势随样本数量的增加而减弱，甚至低于非搜索基准。这一限制源于在未见过的推理路径上价值模型可靠性的下降。为了解决这一问题，我们提出了一种意识不确定性搜索框架，该框架包括两个关键组件：（1）意识不确定性的价值模型，该模型将不确定性纳入预测中；（2）使用提出的高效组汤普森采样算法的意识不确定性的选择过程。在GSM8K实验中，我们的方法缓解了搜索扩展性问题，在16个样本的情况下实现了90.5%的覆盖率，而传统的价值导向搜索仅为85.8%。这项工作首次系统地将不确定性量化整合到了大型语言模型的搜索范式中。 

---
# NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM 

**Title (ZH)**: NavRAG：通过检索增强的大语言模型生成用户需求指令以实现具身导航 

**Authors**: Zihan Wang, Yaohui Zhu, Gim Hee Lee, Yachun Fan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11142)  

**Abstract**: Vision-and-Language Navigation (VLN) is an essential skill for embodied agents, allowing them to navigate in 3D environments following natural language instructions. High-performance navigation models require a large amount of training data, the high cost of manually annotating data has seriously hindered this field. Therefore, some previous methods translate trajectory videos into step-by-step instructions for expanding data, but such instructions do not match well with users' communication styles that briefly describe destinations or state specific needs. Moreover, local navigation trajectories overlook global context and high-level task planning. To address these issues, we propose NavRAG, a retrieval-augmented generation (RAG) framework that generates user demand instructions for VLN. NavRAG leverages LLM to build a hierarchical scene description tree for 3D scene understanding from global layout to local details, then simulates various user roles with specific demands to retrieve from the scene tree, generating diverse instructions with LLM. We annotate over 2 million navigation instructions across 861 scenes and evaluate the data quality and navigation performance of trained models. 

**Abstract (ZH)**: 视觉和语言导航（VLN）是连续体代理的一项核心技能，使它们能够根据自然语言指令在三维环境中导航。高性能的导航模型需要大量的训练数据，手动标注数据的高成本严重阻碍了这一领域的进展。因此，一些先前的方法通过将轨迹视频转换为逐步指令来扩展数据，但这些指令不符合用户的交流风格，用户通常只简要描述目的地或特定需求。此外，局部导航轨迹忽略了全局上下文和高层任务规划。为了解决这些问题，我们提出了一种检索增强生成（RAG）框架——NavRAG，用于为VLN生成用户需求指令。NavRAG 利用大型语言模型（LLM）构建一个分层场景描述树，从全局布局到局部细节，理解三维场景；然后模拟具有不同需求的各种用户角色，从场景树中检索信息，使用LLM生成多样化的指令。我们对861个场景中的超过200万条导航指令进行了标注，并评估了训练模型的数据质量和导航性能。 

---
# Solving Online Resource-Constrained Scheduling for Follow-Up Observation in Astronomy: a Reinforcement Learning Approach 

**Title (ZH)**: Astronomy 中的后续观测在线资源约束调度问题求解：一种强化学习方法 

**Authors**: Yajie Zhang, Ce Yu, Chao Sun, Jizeng Wei, Junhan Ju, Shanjiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11134)  

**Abstract**: In the astronomical observation field, determining the allocation of observation resources of the telescope array and planning follow-up observations for targets of opportunity (ToOs) are indispensable components of astronomical scientific discovery. This problem is computationally challenging, given the online observation setting and the abundance of time-varying factors that can affect whether an observation can be conducted. This paper presents ROARS, a reinforcement learning approach for online astronomical resource-constrained scheduling. To capture the structure of the astronomical observation scheduling, we depict every schedule using a directed acyclic graph (DAG), illustrating the dependency of timing between different observation tasks within the schedule. Deep reinforcement learning is used to learn a policy that can improve the feasible solution by iteratively local rewriting until convergence. It can solve the challenge of obtaining a complete solution directly from scratch in astronomical observation scenarios, due to the high computational complexity resulting from numerous spatial and temporal constraints. A simulation environment is developed based on real-world scenarios for experiments, to evaluate the effectiveness of our proposed scheduling approach. The experimental results show that ROARS surpasses 5 popular heuristics, adapts to various observation scenarios and learns effective strategies with hindsight. 

**Abstract (ZH)**: 在天文学观测领域，确定望远镜阵列的观测资源分配以及规划突发观测目标（ToOs）的后续观测，是天文科学发现不可或缺的组成部分。由于在线观测的环境以及众多可变因素的影响（这些因素可能会影响观测是否能够成功进行），这一问题具有很强的计算挑战性。本文提出了一种基于强化学习的在线天文资源受限调度方法ROARS。为了捕捉天文观测调度的结构，我们使用有向无环图（DAG）来表示每次调度，以展示调度中不同观测任务之间的时间依赖关系。使用深度强化学习学习一种策略，通过迭代局部重写直到收敛，进而逐步提升可行的解决方案。这种方法能够解决在高计算复杂性的天文观测场景中直接从零开始求解完整解决方案的挑战。为评估我们提出的调度方法的有效性，我们基于真实场景开发了一个仿真环境。实验结果表明，ROARS 在多种观测场景下超过了 5 种流行的启发式方法，并且能够学习到有效的策略。 

---
# Hierarchical Expert Prompt for Large-Language-Model: An Approach Defeat Elite AI in TextStarCraft II for the First Time 

**Title (ZH)**: 大型语言模型的层次化专家提示：首次在文本星际II中战胜精英AI的方法 

**Authors**: Zongyuan Li, Chang Lu, Xiaojie Xu, Runnan Qi, Yanan Ni, Lumin Jiang, Xiangbei Liu, Xuebo Zhang, Yongchun Fang, Kuihua Huang, Xian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.11122)  

**Abstract**: Since the emergence of the Large Language Model (LLM), LLM has been widely used in fields such as writing, translating, and searching. However, there is still great potential for LLM-based methods in handling complex tasks such as decision-making in the StarCraft II environment. To address problems such as lack of relevant knowledge and poor control over subtasks of varying importance, we propose a Hierarchical Expert Prompt (HEP) for LLM. Our method improves the understanding of game situations through expert-level tactical knowledge, improving the processing quality of tasks of varying importance through a hierarchical framework. Our approach defeated the highest level (Elite) standard built-in agent in TextStarCraft II for the first time and consistently outperformed the baseline method in other difficulties. Our experiments suggest that the proposed method is a practical solution for tackling complex decision-making challenges. The replay video can be viewed on this https URL and this https URL, and our codes have been open-sourced on this https URL. 

**Abstract (ZH)**: 自大型语言模型（LLM）的出现以来，LLM 在写作、翻译和搜索等领域得到了广泛应用。然而，在处理《星际争霸II》环境中的复杂任务（如决策）方面，LLM 基础方法仍有很大的改进空间。为了解决缺乏相关知识和对不同重要性子任务控制不足的问题，我们提出了一种层次专家提示（HEP）方法。该方法通过专家级别的战术知识提高对游戏情况的理解，并通过层次框架提高对不同重要性任务处理的质量。我们的方法首次击败了《TextStarCraft II》中内置的最高水平（精英）标准代理，并且在其他难度下始终优于基线方法。实验表明，所提出的方法是一种应对复杂决策挑战的实际解决方案。您可以在以下链接查看重播视频：[这里](https://example.com/replay1) 和 [这里](https://example.com/replay2)，并且我们的代码已在以下链接中开源：[这里](https://example.com/codes)。 

---
# OptMATH: A Scalable Bidirectional Data Synthesis Framework for Optimization Modeling 

**Title (ZH)**: OptMATH：一种可扩展的双向数据合成框架用于优化建模 

**Authors**: Hongliang Lu, Zhonglin Xie, Yaoyu Wu, Can Ren, Yuxuan Chen, Zaiwen Wen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11102)  

**Abstract**: Despite the rapid development of large language models (LLMs), a fundamental challenge persists: the lack of high-quality optimization modeling datasets hampers LLMs' robust modeling of practical optimization problems from natural language descriptions (NL). This data scarcity also contributes to the generalization difficulties experienced by learning-based methods. To address these challenges, we propose a scalable framework for synthesizing a high-quality dataset, named OptMATH. Starting from curated seed data with mathematical formulations (MF), this framework automatically generates problem data (PD) with controllable complexity. Then, a back-translation step is employed to obtain NL. To verify the correspondence between the NL and the PD, a forward modeling step followed by rejection sampling is used. The accepted pairs constitute the training part of OptMATH. Then a collection of rejected pairs is identified and further filtered. This collection serves as a new benchmark for optimization modeling, containing difficult instances whose lengths are much longer than these of NL4OPT and MAMO. Through extensive experiments, we demonstrate that models of various sizes (0.5B-32B parameters) trained on OptMATH achieve superior results on multiple modeling benchmarks, thereby validating the effectiveness and scalability of our approach. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）的发展迅猛，但一个基本的挑战依然存在：缺乏高质量的优化建模数据集阻碍了LLMs从自然语言描述（NL）中稳健建模实际优化问题的能力。这种数据稀缺性也导致学习方法在泛化方面的困难。为了解决这些问题，我们提出了一种可扩展的框架，用于合成高质量的数据集，称为OptMATH。该框架从经过精编的种子数据（带有数学公式MF的数据）开始，自动生成具有可控制复杂度的问题数据（PD）。然后，通过反向翻译步骤获得自然语言描述（NL）。为了验证NL与PD之间的对应关系，我们采用了前向建模步骤和拒绝采样方法。被接受的配对构成OptMATH的训练部分。然后，我们识别并进一步过滤一组被拒绝的配对，这些配对作为优化建模的新基准，包含一些长度远超NL4OPT和MAMO的数据实例。通过广泛的实验证明，使用OptMATH训练的不同规模（0.5B-32B参数）的模型在多种建模基准上取得了优越的结果，从而验证了我们方法的有效性和可扩展性。 

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
# PEA: Enhancing LLM Performance on Computational-Reasoning Tasks 

**Title (ZH)**: PEA：提高大型语言模型在计算推理任务性能的方法 

**Authors**: Zi Wang, Shiwei Weng, Mohannad Alhanahnah, Somesh Jha, Tom Reps  

**Link**: [PDF](https://arxiv.org/pdf/2502.10938)  

**Abstract**: Large Language Models (LLMs) have exhibited remarkable capabilities across diverse domains, prompting investigations into their potential as generic reasoning engines. While recent studies have explored inference-time computation to enhance model performance on complex problems, current research lacks a formal framework to characterize the complexity of reasoning tasks. This study introduces the Predicate-Enumeration-Aggregation (PEA) framework, a formal approach to describe and solve a class of important reasoning tasks termed computational reasoning problems. The PEA framework decomposes these problems into predicate and enumeration components, using LLMs to synthesize programs based on specified predicates, enumeration, and aggregation rules. These synthesized programs are then executed to obtain solutions to the computational tasks. We demonstrate the framework's efficacy on benchmark tasks including Boolean satisfiability problems, game of $24$, and planning problems. Empirical evaluation reveals that PEA substantially enhances the performance of underlying models on benchmark computational problems, yielding an average accuracy improvement of approximately $50\%$, coupled with increased efficiency. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在多个领域展现出卓越的能力，引发了将其作为通用推理引擎的研究。尽管近年来的研究探索了推理时的计算方法以提高模型在复杂问题上的表现，当前研究缺乏正式框架来表征推理任务的复杂性。本研究引入了谓词枚举聚合（PEA）框架，这是一种正式方法，用于描述和解决一类重要的推理任务——计算推理问题。PEA框架将这些任务分解为谓词和枚举两部分，并利用大规模语言模型（LLMs）根据指定的谓词、枚举和聚合规则生成程序。生成的程序随后被执行以求解计算任务。我们在基准任务，如布尔可满足性问题、24点游戏和规划问题上展示了该框架的有效性。实证评估表明，PEA显著提升了基础模型在基准计算问题上的性能，平均准确率提高了约50%，同时提高了效率。 

---
# SCALE: Towards Collaborative Content Analysis in Social Science with Large Language Model Agents and Human Intervention 

**Title (ZH)**: SCALE：借助大型语言模型代理和人类干预实现社会科学研究中的协作内容分析 

**Authors**: Chengshuai Zhao, Zhen Tan, Chau-Wai Wong, Xinyan Zhao, Tianlong Chen, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10937)  

**Abstract**: Content analysis breaks down complex and unstructured texts into theory-informed numerical categories. Particularly, in social science, this process usually relies on multiple rounds of manual annotation, domain expert discussion, and rule-based refinement. In this paper, we introduce SCALE, a novel multi-agent framework that effectively $\underline{\textbf{S}}$imulates $\underline{\textbf{C}}$ontent $\underline{\textbf{A}}$nalysis via $\underline{\textbf{L}}$arge language model (LLM) ag$\underline{\textbf{E}}$nts. SCALE imitates key phases of content analysis, including text coding, collaborative discussion, and dynamic codebook evolution, capturing the reflective depth and adaptive discussions of human researchers. Furthermore, by integrating diverse modes of human intervention, SCALE is augmented with expert input to further enhance its performance. Extensive evaluations on real-world datasets demonstrate that SCALE achieves human-approximated performance across various complex content analysis tasks, offering an innovative potential for future social science research. 

**Abstract (ZH)**: 内容分析将复杂的非结构化文本分解为理论指导的数值类别。尤其在社会科学中，这一过程通常依赖于多轮的手动标注、领域专家讨论以及基于规则的改进。在本文中，我们介绍了一种新颖的多智能体框架——SCALE，它利用大型语言模型（LLM）智能体有效地实现内容分析的模拟。SCALE 模拟内容分析的关键阶段，包括文本编码、协作讨论和动态代码本演化，能够捕捉人类研究者的反思深度和适应性讨论。此外，通过整合多种人类干预模式，SCALE 进一步增强了其性能，融入了专家输入。在现实世界数据集上的广泛评估表明，SCALE 在各类复杂内容分析任务中实现了接近人类的表现，为未来社会科学的研究提供了创新的潜力。 

---
# D-CIPHER: Dynamic Collaborative Intelligent Agents with Planning and Heterogeneous Execution for Enhanced Reasoning in Offensive Security 

**Title (ZH)**: D-CIPHER：具有规划能力和异构执行的动态协作智能代理，以增强进攻性安全中的推理能力 

**Authors**: Meet Udeshi, Minghao Shao, Haoran Xi, Nanda Rani, Kimberly Milner, Venkata Sai Charan Putrevu, Brendan Dolan-Gavitt, Sandeep Kumar Shukla, Prashanth Krishnamurthy, Farshad Khorrami, Ramesh Karri, Muhammad Shafique  

**Link**: [PDF](https://arxiv.org/pdf/2502.10931)  

**Abstract**: Large Language Models (LLMs) have been used in cybersecurity in many ways, including their recent use as intelligent agent systems for autonomous security analysis. Capture the Flag (CTF) challenges serve as benchmarks for assessing the automated task-planning abilities of LLM agents across various cybersecurity skill sets. Early attempts to apply LLMs for solving CTF challenges relied on single-agent systems, where feedback was restricted to a single reasoning-action loop. This approach proved inadequate for handling complex CTF tasks. Drawing inspiration from real-world CTF competitions, where teams of experts collaborate, we introduce the D-CIPHER multi-agent LLM framework for collaborative CTF challenge solving. D-CIPHER integrates agents with distinct roles, enabling dynamic feedback loops to enhance reasoning on CTF challenges. It introduces the Planner-Executor agent system, consisting of a Planner agent for overall problem-solving along with multiple heterogeneous Executor agents for individual tasks, facilitating efficient allocation of responsibilities among the LLMs. Additionally, D-CIPHER incorporates an Auto-prompter agent, which improves problem-solving by exploring the challenge environment and generating a highly relevant initial prompt. We evaluate D-CIPHER on CTF benchmarks using multiple LLM models and conduct comprehensive studies to highlight the impact of our enhancements. Our results demonstrate that the multi-agent D-CIPHER system achieves a significant improvement in challenges solved, setting a state-of-the-art performance on three benchmarks: 22.0% on NYU CTF Bench, 22.5% on Cybench, and 44.0% on HackTheBox. D-CIPHER is available at this https URL as the nyuctf_multiagent package. 

**Abstract (ZH)**: 大型语言模型（LLMs）在网络安全领域的应用涵盖了多种方式，包括将其用作自主安全分析的智能代理系统。Capture the Flag（CTF）挑战赛被用作评估LLM代理在各种网络安全技能上的自动化任务规划能力的基准。早期使用LLM解决CTF挑战的尝试依赖于单代理系统，其中反馈仅限于单一推理-行动循环。这种做法对于处理复杂的CTF任务证明是不够的。借鉴真实世界CTF竞赛中专家团队合作的理念，我们提出了D-CIPHER多代理LLM框架，用于协作解决CTF挑战。D-CIPHER集成了具有不同角色的代理，能够实现动态反馈循环，以增强对CTF挑战的推理能力。该框架引入了规划-执行者代理系统，包括一个负责整体问题解决的规划者代理，以及多个异构执行者代理，用于执行单独的任务，从而实现LLM们责任分配的有效化。此外，D-CIPHER还集成了自动提示生成代理，通过探索挑战环境生成高度相关的初始提示以改善问题解决能力。我们在使用多个LLM模型的CTF基准测试上评估了D-CIPHER，并进行了全面研究以突出我们改进措施的影响。我们的结果显示，多代理D-CIPHER系统在三个基准测试上解决了更多的挑战问题，分别达到以下性能指标：在NYU CTF基准测试上为22.0%，在Cybench上为22.5%，在HackTheBox上为44.0%。D-CIPHER可在以下链接获取：这个 https URL，作为nyuctf_multiagent包。 

---
# PCGRLLM: Large Language Model-Driven Reward Design for Procedural Content Generation Reinforcement Learning 

**Title (ZH)**: PCGRLLM：基于大型语言模型的 procedural 内容生成强化学习奖励设计 

**Authors**: In-Chang Baek, Sung-Hyun Kim, Sam Earle, Zehua Jiang, Noh Jin-Ha, Julian Togelius, Kyung-Joong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.10906)  

**Abstract**: Reward design plays a pivotal role in the training of game AIs, requiring substantial domain-specific knowledge and human effort. In recent years, several studies have explored reward generation for training game agents and controlling robots using large language models (LLMs). In the content generation literature, there has been early work on generating reward functions for reinforcement learning agent generators. This work introduces PCGRLLM, an extended architecture based on earlier work, which employs a feedback mechanism and several reasoning-based prompt engineering techniques. We evaluate the proposed method on a story-to-reward generation task in a two-dimensional environment using two state-of-the-art LLMs, demonstrating the generalizability of our approach. Our experiments provide insightful evaluations that demonstrate the capabilities of LLMs essential for content generation tasks. The results highlight significant performance improvements of 415% and 40% respectively, depending on the zero-shot capabilities of the language model. Our work demonstrates the potential to reduce human dependency in game AI development, while supporting and enhancing creative processes. 

**Abstract (ZH)**: 奖励设计在游戏AI的训练中起到关键作用，需要大量的领域特定知识和人力投入。近年来，多项研究探讨了使用大型语言模型（LLM）生成奖励信号，以训练游戏代理和控制机器人。在内容生成文献中，早期就有关于生成奖励函数的工作，用于强化学习代理生成器。本文介绍了一种基于前期工作的扩展架构——PCGRLLM，该架构采用了反馈机制和多种基于推理的提示工程技术。我们使用两种最先进的LLM，在一个二维环境中评估所提出的方法，展示了我们方法的普适性。我们的实验提供了对LLM生成内容任务能力的有价值的评估，结果显示了显著的性能提升，分别高达415%和40%，这取决于语言模型的零样本能力。本文证明了在游戏AI开发中减少对人类依赖的可能性，同时支持并增强了创造性过程。 

---
# A Tutorial on LLM Reasoning: Relevant Methods behind ChatGPT o1 

**Title (ZH)**: 基于ChatGPT背后的推理方法：一个大型语言模型推理教程 

**Authors**: Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10867)  

**Abstract**: OpenAI o1 has shown that applying reinforcement learning to integrate reasoning steps directly during inference can significantly improve a model's reasoning capabilities. This result is exciting as the field transitions from the conventional autoregressive method of generating answers to a more deliberate approach that models the slow-thinking process through step-by-step reasoning training. Reinforcement learning plays a key role in both the model's training and decoding processes. In this article, we present a comprehensive formulation of reasoning problems and investigate the use of both model-based and model-free approaches to better support this slow-thinking framework. 

**Abstract (ZH)**: OpenAI的研究表明，在推理过程中直接应用强化学习以整合推理步骤可以显著提升模型的推理能力。这一结果令人振奋，因为随着领域从传统的自回归方法向通过逐步推理训练建模的谨慎方法转变，这一领域正在经历转变。强化学习在模型的训练和解码过程中都扮演着关键角色。在本文中，我们提出了推理问题的全面形式化描述，并探讨了基于模型和无模型方法在更好地支持这一逐步推理框架中的应用。 

---
# Is Depth All You Need? An Exploration of Iterative Reasoning in LLMs 

**Title (ZH)**: 《仅仅是深度吗？关于LLMs中迭代推理的探索》

此标题翻译旨在保持原文的学术严谨性和原意。其中，“Is Depth All You Need?”转化为“仅仅是深度吗？”，“An Exploration of”转化为“关于……的探索”，“Iterative Reasoning in LLMs”转化为“LLMs中迭代推理”，以确保翻译符合学术论文标题的规范。 

**Authors**: Zongqian Wu, Tianyu Li, Jiaying Yang, Mengmeng Zhan, Xiaofeng Zhu, Lei Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.10858)  

**Abstract**: Deep iterative chain-of-thought (CoT) reasoning enables LLMs to tackle complex tasks by progressively activating relevant pre-trained knowledge. However, it faces challenges in ensuring continual improvement and determining a stopping criterion. In this paper, we investigate whether the relevant knowledge that contributes directly to solving the given question can be activated from the initial reasoning path, thus circumventing the need for iterative refinement. Our experiments reveal that increasing the diversity of initial reasoning paths can achieve comparable or superior performance, a concept we term \textit{breadth reasoning}. However, existing breadth reasoning approaches, such as self-consistency, offer limited diversity. To address this limitation, we propose a simple yet effective method that enhances reasoning breadth by integrating contextual exploration with reduced sampling randomness. Extensive experiments demonstrate that our approach significantly outperforms deep iterative reasoning. Our code is provided in this https URL. 

**Abstract (ZH)**: 深度迭代链式推理（CoT）能够通过逐步激活相关预训练知识来使大规模语言模型（LLMs）应对复杂的任务。然而，这种方法在确保持续改进和确定停止准则方面面临挑战。本文探讨了是否可以在初始推理路径中直接激活与解决给定问题相关的知识，从而绕过迭代细化的需求。我们的实验表明，增加初始推理路径的多样性可以达到相似甚至更优的性能，我们将其称为“广度推理”。然而，现有的广度推理方法，如自我一致性，在多样性方面提供的支持有限。为了解决这一局限性，我们提出了一种简单而有效的办法，通过结合上下文探索与减少采样随机性来增强推理的广度。广泛实验表明，我们的方法在深度迭代推理方面具有显著的优势。我们的代码在此处提供：[提供代码链接的地方]。 

---
# The Philosophical Foundations of Growing AI Like A Child 

**Title (ZH)**: Growing AI like a child：其哲学基础 

**Authors**: Dezhi Luo, Yijiang Li, Hokin Deng  

**Link**: [PDF](https://arxiv.org/pdf/2502.10742)  

**Abstract**: Despite excelling in high-level reasoning, current language models lack robustness in real-world scenarios and perform poorly on fundamental problem-solving tasks that are intuitive to humans. This paper argues that both challenges stem from a core discrepancy between human and machine cognitive development. While both systems rely on increasing representational power, the absence of core knowledge-foundational cognitive structures in humans-prevents language models from developing robust, generalizable abilities, where complex skills are grounded in simpler ones within their respective domains. It explores empirical evidence of core knowledge in humans, analyzes why language models fail to acquire it, and argues that this limitation is not an inherent architectural constraint. Finally, it outlines a workable proposal for systematically integrating core knowledge into future multi-modal language models through the large-scale generation of synthetic training data using a cognitive prototyping strategy. 

**Abstract (ZH)**: 尽管当前的语言模型在高层次推理方面表现出色，但在实际应用场景中它们缺乏稳健性，并且在那些对人类来说直觉性很强的基本问题解决任务上表现不佳。本文认为，这两种挑战都源于人类与机器认知发展的核心差异。虽然两种系统都依赖于增强表示能力，但由于人类缺乏核心知识—即基础认知结构，语言模型无法发展出稳健且可泛化的技能，这些复杂技能在各自领域内是基于更简单的技能之上的。本文探讨了人类核心知识的实证证据，分析了为什么语言模型无法获得这些知识，并认为这一局限性并不是由于固有的架构限制。最后，本文提出了一个可操作的方案，即通过大规模生成合成训练数据并采用认知原型策略的方式，系统性地将核心知识融入未来的多模态语言模型。 

---
# CoPEFT: Fast Adaptation Framework for Multi-Agent Collaborative Perception with Parameter-Efficient Fine-Tuning 

**Title (ZH)**: CoPEFT：一种基于参数高效微调的多智能体协作感知快速adaptation框架 

**Authors**: Quanmin Wei, Penglin Dai, Wei Li, Bingyi Liu, Xiao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10705)  

**Abstract**: Multi-agent collaborative perception is expected to significantly improve perception performance by overcoming the limitations of single-agent perception through exchanging complementary information. However, training a robust collaborative perception model requires collecting sufficient training data that covers all possible collaboration scenarios, which is impractical due to intolerable deployment costs. Hence, the trained model is not robust against new traffic scenarios with inconsistent data distribution and fundamentally restricts its real-world applicability. Further, existing methods, such as domain adaptation, have mitigated this issue by exposing the deployment data during the training stage but incur a high training cost, which is infeasible for resource-constrained agents. In this paper, we propose a Parameter-Efficient Fine-Tuning-based lightweight framework, CoPEFT, for fast adapting a trained collaborative perception model to new deployment environments under low-cost conditions. CoPEFT develops a Collaboration Adapter and Agent Prompt to perform macro-level and micro-level adaptations separately. Specifically, the Collaboration Adapter utilizes the inherent knowledge from training data and limited deployment data to adapt the feature map to new data distribution. The Agent Prompt further enhances the Collaboration Adapter by inserting fine-grained contextual information about the environment. Extensive experiments demonstrate that our CoPEFT surpasses existing methods with less than 1\% trainable parameters, proving the effectiveness and efficiency of our proposed method. 

**Abstract (ZH)**: 多智能体协同感知有望通过交换互补信息来克服单智能体感知的限制，从而显著提升感知性能。然而，训练一个鲁棒的协同感知模型需要收集足以涵盖所有可能合作场景的大量训练数据，这种做法由于部署成本不可接受而变得不切实际。因此，训练后的模型对具有不同数据分布的新交通场景不够鲁棒，从而限制了其实际应用。现有方法，如领域适应，通过在训练阶段暴露部署数据来缓解这一问题，但也带来了高昂的训练成本，这对于资源受限的智能体来说是不可行的。在本文中，我们提出了一种参数高效微调的轻量级框架CoPEFT，以在低成本条件下快速适应已训练的协同感知模型以适应新的部署环境。CoPEFT开发了协作适配器和智能体提示，分别在宏层和微层进行适应。具体来说，协作适配器利用训练数据和有限的部署数据中的固有知识来适应特征图以匹配新的数据分布。智能体提示进一步通过插入有关环境的细粒度上下文信息来增强协作适配器。广泛的实验结果表明，我们的CoPEFT在不到1%的可训练参数下超越了现有方法，证明了我们所提出方法的有效性和高效性。 

---
# Demographic User Modeling for Social Robotics with Multimodal Pre-trained Models 

**Title (ZH)**: 使用多模态预训练模型进行社交机器人的人口统计学用户建模 

**Authors**: Hamed Rahimi, Mouad Abrini, Mahdi Khoramshahi, Mohamed Chetouani  

**Link**: [PDF](https://arxiv.org/pdf/2502.10642)  

**Abstract**: This paper investigates the performance of multimodal pre-trained models in user profiling tasks based on visual-linguistic demographic data. These models are critical for adapting to the needs and preferences of human users in social robotics, thereby providing personalized responses and enhancing interaction quality. First, we introduce two datasets specifically curated to represent demographic characteristics derived from user facial images. Next, we evaluate the performance of a prominent contrastive multimodal pre-trained model, CLIP, on these datasets, both in its out-of-the-box state and after fine-tuning. Initial results indicate that CLIP performs suboptimal in matching images to demographic descriptions without fine-tuning. Although fine-tuning significantly enhances its predictive capacity, the model continues to exhibit limitations in effectively generalizing subtle demographic nuances. To address this, we propose adopting a masked image modeling strategy to improve generalization and better capture subtle demographic attributes. This approach offers a pathway for enhancing demographic sensitivity in multimodal user modeling tasks. 

**Abstract (ZH)**: 本文探讨了多模态预训练模型在基于视觉语言人口统计学数据的用户画像任务中的表现。这些模型对于适应社会机器人领域中人类用户的需求和偏好至关重要，从而能够提供个性化响应并提升交互质量。首先，我们介绍了两个专门用于表示从用户面部图像提取的人口统计特征的数据集。随后，我们评估了一种主流对比多模态预训练模型CLIP在这些数据集上的表现，包括其“即用型”状态以及经过微调后的表现。初步结果显示，未经微调的CLIP在匹配图像和人口统计描述方面表现不佳。尽管微调显著提升了其预测能力，但模型仍存在难以有效泛化微妙人口统计特征的局限性。为解决这一问题，我们提出了采用掩码图像建模策略以提高泛化能力，并更好地捕捉微妙的人口统计属性。这种方法为提高多模态用户建模任务中的人口统计敏感性提供了一条途径。 

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
# Benchmarking the rationality of AI decision making using the transitivity axiom 

**Title (ZH)**: 使用传递性公理 benchmark 人工智能决策的合理性 

**Authors**: Kiwon Song, James M. Jennings III, Clintin P. Davis-Stober  

**Link**: [PDF](https://arxiv.org/pdf/2502.10554)  

**Abstract**: Fundamental choice axioms, such as transitivity of preference, provide testable conditions for determining whether human decision making is rational, i.e., consistent with a utility representation. Recent work has demonstrated that AI systems trained on human data can exhibit similar reasoning biases as humans and that AI can, in turn, bias human judgments through AI recommendation systems. We evaluate the rationality of AI responses via a series of choice experiments designed to evaluate transitivity of preference in humans. We considered ten versions of Meta's Llama 2 and 3 LLM models. We applied Bayesian model selection to evaluate whether these AI-generated choices violated two prominent models of transitivity. We found that the Llama 2 and 3 models generally satisfied transitivity, but when violations did occur, occurred only in the Chat/Instruct versions of the LLMs. We argue that rationality axioms, such as transitivity of preference, can be useful for evaluating and benchmarking the quality of AI-generated responses and provide a foundation for understanding computational rationality in AI systems more generally. 

**Abstract (ZH)**: 偏好传输性的基本选择公理，例如偏好传递性，为判断人类决策是否理性（即是否符合效用表示）提供了可测试的条件。最近的研究表明，基于人类数据训练的人工智能系统可能会表现出与人类相似的推理偏差，并且人工智能系统可以通过推荐系统影响人类的判断。我们通过一系列旨在评估人类偏好传递性的选择实验来评估AI响应的理性。我们考虑了Meta的Llama 2和3种LLM模型的十个版本。我们应用贝叶斯模型选择方法来评估这些AI生成的选择是否违反了两个著名的偏好传输性模型。结果显示，Llama 2和3种模型通常满足偏好传递性，但在违反偏好传递性的情况发生时，仅出现在LLM的Chat/指令版本中。我们认为，如偏好传递性这样的理性公理可以用于评估和基准测试AI生成响应的质量，并为更广泛地理解人工智能系统中的计算理性提供基础。 

---
# GraphiT: Efficient Node Classification on Text-Attributed Graphs with Prompt Optimized LLMs 

**Title (ZH)**: GraphiT: 采用提示优化的大语言模型高效进行文本属性图的节点分类 

**Authors**: Shima Khoshraftar, Niaz Abedini, Amir Hajian  

**Link**: [PDF](https://arxiv.org/pdf/2502.10522)  

**Abstract**: The application of large language models (LLMs) to graph data has attracted a lot of attention recently. LLMs allow us to use deep contextual embeddings from pretrained models in text-attributed graphs, where shallow embeddings are often used for the text at- tributes of nodes. However, it is still challenging to efficiently en- code the graph structure and features into a sequential form for use by LLMs. In addition, the performance of an LLM alone, is highly dependent on the structure of the input prompt, which limits their effectiveness as a reliable approach and often requires iterative man- ual adjustments that could be slow, tedious and difficult to replicate programmatically. In this paper, we propose GraphiT (Graphs in Text), a framework for encoding graphs into a textual format and optimizing LLM prompts for graph prediction tasks. Here we focus on node classification for text-attributed graphs. We encode the graph data for every node and its neighborhood into a concise text to enable LLMs to better utilize the information in the graph. We then further programmatically optimize the LLM prompts us- ing the DSPy framework to automate this step and make it more efficient and reproducible. GraphiT outperforms our LLM-based baselines on three datasets and we show how the optimization step in GraphiT leads to measurably better results without manual prompt tweaking. We also demonstrated that our graph encoding approach is competitive to other graph encoding methods while being less expensive because it uses significantly less tokens for the same task. 

**Abstract (ZH)**: 近年来，将大型语言模型（LLMs）应用于图数据引起了大量关注。LLMs 使我们能够利用预训练模型中的深度上下文嵌入来处理具有文本属性的图数据，而在这些图数据中，浅层嵌入通常用于节点的文本属性。然而，高效地将图结构和特征编码为顺序形式，以便用于LLMs，仍然具有挑战性。此外，单独使用LLM的效果高度依赖于输入提示的结构，这限制了它们作为可靠方法的有效性，并常常需要迭代的手动调整，这可能是缓慢、繁琐且难以程序化地重复的。在本文中，我们提出了一种名为GraphiT（图在文本中）的框架，用于将图编码为文本格式，并优化LLM提示以进行图预测任务。我们重点讨论具有文本属性的节点分类任务。我们为每个节点及其邻域编码图数据，以使LLMs能够更好地利用图中的信息。然后，我们使用DSPy框架进一步编程优化LLM提示，以自动化这一过程，使其更加高效和可重复。在三个数据集上，GraphiT的性能优于我们的LLM基线，我们展示了GraphiT中的优化步骤如何在无需手动调整提示的情况下获得可衡量的更好结果。此外，我们还证明了我们的图编码方法在竞争性方面与其他图编码方法相当，但成本较低，因为它可以使用显著较少的令牌完成相同任务。 

---
# A Self-Supervised Reinforcement Learning Approach for Fine-Tuning Large Language Models Using Cross-Attention Signals 

**Title (ZH)**: 使用跨注意力信号进行大规模语言模型微调的自我监督强化学习方法 

**Authors**: Andrew Kiruluta, Andreas Lemos, Priscilla Burity  

**Link**: [PDF](https://arxiv.org/pdf/2502.10482)  

**Abstract**: We propose a novel reinforcement learning framework for post training large language models that does not rely on human in the loop feedback. Instead, our approach uses cross attention signals within the model itself to derive a self supervised reward, thereby guiding iterative fine tuning of the model policy. By analyzing how the model attends to the input prompt during generation, we construct measures of prompt coverage, focus, and coherence. We then use these measures to rank or score candidate responses, providing a reward signal that encourages the model to produce well aligned, on topic text. In empirical comparisons against standard policy gradient methods and RL fine tuning with synthetic preference models, our method shows significant gains in prompt relevance and consistency over a non RL baseline. While it does not yet match the performance of fully human supervised RLHF systems, it highlights an important direction for scaling alignment with minimal human labeling. We provide a detailed analysis, discuss potential limitations, and outline future work for combining cross-attention based signals with smaller amounts of human feedback. 

**Abstract (ZH)**: 我们提出了一种无需人工参与回访的新型强化学习框架，用于后训练大型语言模型。我们的方法利用模型内部的跨注意力信号来推导自我监督的奖励，从而引导模型策略的迭代微调。通过分析模型在生成过程中对输入提示的注意力分布，我们构建了提示覆盖度、聚焦性和连贯性的衡量指标。然后，我们使用这些衡量指标来对候选响应进行排名或评分，提供一种奖励信号，鼓励模型生成内容相关且一致的文本。在与标准策略梯度方法以及使用合成偏好模型的强化学习微调方法的实证比较中，我们的方法在提示的相关性和一致性方面显著优于非强化学习基线。尽管它尚未达到完全有人类监督的RLHF系统的性能，但它展示了在最少人工标注的情况下扩大对齐方向的重要前景。我们提供了详细的分析，讨论了潜在的局限性，并概述了将跨注意力基信号与少量人工反馈相结合的未来工作。 

---
# Knowledge Integration Strategies in Autonomous Vehicle Prediction and Planning: A Comprehensive Survey 

**Title (ZH)**: 自主驾驶车辆预测与规划中的知识集成策略：综合综述 

**Authors**: Kumar Manas, Adrian Paschke  

**Link**: [PDF](https://arxiv.org/pdf/2502.10477)  

**Abstract**: This comprehensive survey examines the integration of knowledge-based approaches into autonomous driving systems, with a focus on trajectory prediction and planning. We systematically review methodologies for incorporating domain knowledge, traffic rules, and commonsense reasoning into these systems, spanning purely symbolic representations to hybrid neuro-symbolic architectures. In particular, we analyze recent advancements in formal logic and differential logic programming, reinforcement learning frameworks, and emerging techniques that leverage large foundation models and diffusion models for knowledge representation. Organized under a unified literature survey section, our discussion synthesizes the state-of-the-art into a high-level overview, supported by a detailed comparative table that maps key works to their respective methodological categories. This survey not only highlights current trends -- including the growing emphasis on interpretable AI, formal verification in safety-critical systems, and the increased use of generative models in prediction and planning -- but also outlines the challenges and opportunities for developing robust, knowledge-enhanced autonomous driving systems. 

**Abstract (ZH)**: 本综述全面探讨了基于知识的方法在自动驾驶系统中的整合，重点在于轨迹预测和规划。我们系统地回顾了将领域知识、交通规则和常识推理纳入这些系统的方法，涵盖了从纯粹符号表示到混合神经-符号架构的各个方面。具体地，我们分析了形式逻辑和微分逻辑编程、强化学习框架，以及利用大模型和扩散模型进行知识表示的新兴技术。在统一的文献综述部分组织下，我们的讨论将最先进的成果综合成高层次的概述，并通过详细比较表格将关键作品映射到各自的方法论类别。本综述不仅突显了当前的发展趋势，包括可解释AI的日益重要性、安全关键系统中的形式验证以及预测和规划中生成模型的增加使用，还指出了开发稳健且知识增强的自动驾驶系统面临的挑战与机遇。 

---
# Multi-Objective Planning with Contextual Lexicographic Reward Preferences 

**Title (ZH)**: 具有上下文列席奖励偏好的多目标规划 

**Authors**: Pulkit Rustagi, Yashwanthi Anand, Sandhya Saisubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2502.10476)  

**Abstract**: Autonomous agents are often required to plan under multiple objectives whose preference ordering varies based on context. The agent may encounter multiple contexts during its course of operation, each imposing a distinct lexicographic ordering over the objectives, with potentially different reward functions associated with each context. Existing approaches to multi-objective planning typically consider a single preference ordering over the objectives, across the state space, and do not support planning under multiple objective orderings within an environment. We present Contextual Lexicographic Markov Decision Process (CLMDP), a framework that enables planning under varying lexicographic objective orderings, depending on the context. In a CLMDP, both the objective ordering at a state and the associated reward functions are determined by the context. We employ a Bayesian approach to infer a state-context mapping from expert trajectories. Our algorithm to solve a CLMDP first computes a policy for each objective ordering and then combines them into a single context-aware policy that is valid and cycle-free. The effectiveness of the proposed approach is evaluated in simulation and using a mobile robot. 

**Abstract (ZH)**: 自主代理通常需要在多种目标之间进行规划，这些目标的偏好顺序会基于上下文的不同而变化。代理在其运行过程中可能会遇到多种不同上下文，每个上下文会对目标施加独特的词典序顺序，并且可能与每个上下文相关联有不同的奖励函数。现有的一些多目标规划方法通常在整个状态空间中考虑单一的目标偏好顺序，而不支持在环境内部署多种目标顺序下的规划。我们提出了情境词典序马尔可夫决策过程（Contextual Lexicographic Markov Decision Process, CLMDP）框架，该框架允许根据上下文的不同变化规划多种词典序目标顺序。在CLMDP中，状态的目标顺序及其相关的奖励函数都是由上下文确定的。我们采用贝叶斯方法从专家轨迹中推断状态-上下文映射关系。解决CLMDP问题的算法首先为每个目标顺序计算一个策略，然后将它们组合成一个有效的、无环的状态感知策略。我们通过模拟和使用移动机器人对所提出的这种方法的有效性进行了评估。 

---
# Diverse Transformer Decoding for Offline Reinforcement Learning Using Financial Algorithmic Approaches 

**Title (ZH)**: 使用金融算法方法的离线强化学习中多样化的变压器解码 

**Authors**: Dan Elbaz, Oren Salzman  

**Link**: [PDF](https://arxiv.org/pdf/2502.10473)  

**Abstract**: Offline Reinforcement Learning (RL) algorithms learn a policy using a fixed training dataset, which is then deployed online to interact with the environment and make decisions. Transformers, a standard choice for modeling time-series data, are gaining popularity in offline RL. In this context, Beam Search (BS), an approximate inference algorithm, is the go-to decoding method. Offline RL eliminates the need for costly or risky online data collection. However, the restricted dataset induces uncertainty as the agent may encounter unfamiliar sequences of states and actions during execution that were not covered in the training data. In this context, BS lacks two important properties essential for offline RL: It does not account for the aforementioned uncertainty, and its greedy left-right search approach often results in sequences with minimal variations, failing to explore potentially better alternatives.
To address these limitations, we propose Portfolio Beam Search (PBS), a simple-yet-effective alternative to BS that balances exploration and exploitation within a Transformer model during decoding. We draw inspiration from financial economics and apply these principles to develop an uncertainty-aware diversification mechanism, which we integrate into a sequential decoding algorithm at inference time. We empirically demonstrate the effectiveness of PBS on the D4RL locomotion benchmark, where it achieves higher returns and significantly reduces outcome variability. 

**Abstract (ZH)**: 离线强化学习（Reinforcement Learning, RL）算法使用固定的训练数据集学习策略，然后将该策略在线部署以与环境交互并做出决策。变换器（Transformers），作为一种标准的时间序列数据分析方法，正变得日益流行并应用于离线RL中。在此背景下，束搜索（Beam Search, BS）作为一种近似推理算法，通常用于解码方法。

离线RL消除了在线数据收集昂贵或有风险的需要。然而，受限的训练数据集会导致不确定性，因为该代理在执行过程中可能会遇到训练数据未涵盖的状态和动作序列。在这种背景下，BS缺乏对于离线RL至关重要的两个重要特性：它没有考虑到上述不确定性，其贪心的左向右搜索方法经常导致变化最小的序列，无法探索潜在的更好选择。

为了克服这些局限性，我们提出了一种组合束搜索（Portfolio Beam Search, PBS），它是一种在变换器模型解码过程中平衡探索与利用的简单而有效的替代方法。我们将灵感来源于金融经济学的概念，应用于开发一种不确定性意识下的多样化机制，并将其整合到推理时的序列解码算法中。通过在D4RL运动基准测试中进行实证研究，我们展示了PBS的有效性，它实现了更高的回报并显著减少了结果的变异性。 

---
# AI Alignment at Your Discretion 

**Title (ZH)**: 您提供的标题“AI Alignment at Your Discretion”可以翻译成中文为：“您掌控的AI对齐”或“由您决定的AI对齐”。在学术规范中，标题应简洁明了，同时准确反映论文的主题。这个标题看起来可能是在讨论关于AI对齐（即将AI的目标与人类目标对齐）的一种灵活或用户导向的方法。具体翻译可以根据论文的具体内容进行适当调整。 

**Authors**: Maarten Buyl, Hadi Khalaf, Claudio Mayrink Verdun, Lucas Monteiro Paes, Caio C. Vieira Machado, Flavio du Pin Calmon  

**Link**: [PDF](https://arxiv.org/pdf/2502.10441)  

**Abstract**: In AI alignment, extensive latitude must be granted to annotators, either human or algorithmic, to judge which model outputs are `better' or `safer.' We refer to this latitude as alignment discretion. Such discretion remains largely unexamined, posing two risks: (i) annotators may use their power of discretion arbitrarily, and (ii) models may fail to mimic this discretion. To study this phenomenon, we draw on legal concepts of discretion that structure how decision-making authority is conferred and exercised, particularly in cases where principles conflict or their application is unclear or irrelevant. Extended to AI alignment, discretion is required when alignment principles and rules are (inevitably) conflicting or indecisive. We present a set of metrics to systematically analyze when and how discretion in AI alignment is exercised, such that both risks (i) and (ii) can be observed. Moreover, we distinguish between human and algorithmic discretion and analyze the discrepancy between them. By measuring both human and algorithmic discretion over safety alignment datasets, we reveal layers of discretion in the alignment process that were previously unaccounted for. Furthermore, we demonstrate how algorithms trained on these datasets develop their own forms of discretion in interpreting and applying these principles, which challenges the purpose of having any principles at all. Our paper presents the first step towards formalizing this core gap in current alignment processes, and we call on the community to further scrutinize and control alignment discretion. 

**Abstract (ZH)**: 在AI对齐中，必须给予注释者（无论是人类还是算法）广泛的判断权，以决定哪些模型输出是“更好”的或“更安全”的。我们将这种判断权称为对齐裁量权。这种裁量权在很大程度上尚未得到研究，这带来了两种风险：(i) 注释者可能会滥用他们的裁量权，(ii) 模型可能无法复制这种裁量权。为了研究这一现象，我们借鉴了法律中裁量权的概念，这些概念规范了决策权的授予和行使方式，特别是在原则冲突、应用不清楚或不相关的案件中。在AI对齐中，当对齐原则和规则不可避免地存在冲突或模棱两可时，裁量权是必需的。我们提出了一套指标，以系统地分析在AI对齐过程中何时以及如何行使裁量权，从而观察这些风险 (i) 和 (ii)。此外，我们将人类裁量权和算法裁量权区分开来，并分析它们之间的差异。通过对安全对齐数据集上的人类和算法裁量权进行测量，我们揭示了在对齐过程中之前未被考虑的裁量层级。此外，我们展示了这些数据集训练的算法发展出自己独特的裁量方式以解释和应用这些原则，这挑战了制定任何原则的初衷。我们的论文迈出了朝向正式化目前对齐过程中这一核心缺口的第一步，并呼吁学术界进一步审查和控制对齐裁量权。 

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
# Position: Stop Acting Like Language Model Agents Are Normal Agents 

**Title (ZH)**: 位置：停止将语言模型代理视为普通代理 

**Authors**: Elija Perrier, Michael Timothy Bennett  

**Link**: [PDF](https://arxiv.org/pdf/2502.10420)  

**Abstract**: Language Model Agents (LMAs) are increasingly treated as capable of autonomously navigating interactions with humans and tools. Their design and deployment tends to presume they are normal agents capable of sustaining coherent goals, adapting across contexts and acting with a measure of intentionality. These assumptions are critical to prospective use cases in industrial, social and governmental settings. But LMAs are not normal agents. They inherit the structural problems of the large language models (LLMs) around which they are built: hallucinations, jailbreaking, misalignment and unpredictability. In this Position paper we argue LMAs should not be treated as normal agents, because doing so leads to problems that undermine their utility and trustworthiness. We enumerate pathologies of agency intrinsic to LMAs. Despite scaffolding such as external memory and tools, they remain ontologically stateless, stochastic, semantically sensitive, and linguistically intermediated. These pathologies destabilise the ontological properties of LMAs including identifiability, continuity, persistence and and consistency, problematising their claim to agency. In response, we argue LMA ontological properties should be measured before, during and after deployment so that the negative effects of pathologies can be mitigated. 

**Abstract (ZH)**: 语言模型代理（LMAs）越来越被视为能够自主导航与人类和工具的交互。它们的设计和部署往往假定它们是具备维持连贯目标、跨情境适应和以一定意图性行动的正常代理。这些假设对于工业、社交和政府应用领域潜在使用场景至关重要。然而，LMAs并非正常的代理。它们继承了它们所构建的大型语言模型（LLMs）所存在的结构性问题：幻觉、脱逃、不对齐和不可预测性。在本文中，我们主张不应将LMAs视为正常代理，因为这么做会导致问题，削弱它们的实用性和可信度。我们列出了固有于LMAs的代理病理。即使有外部记忆和工具的支持，它们仍然在本体论上是无状态的、随机的、语义敏感的，并且通过语言中介。这些病理现象动摇了LMAs的本体论特性，包括识别性、连续性、持久性和一致性，从而质疑其代理性主张。为此，我们主张应在部署前、部署中和部署后衡量LMAs的本体论特征，以便减轻病理所导致的负面影响。 

---
# A Coordination-based Approach for Focused Learning in Knowledge-Based Systems 

**Title (ZH)**: 基于协调的方法在基于知识系统中的聚焦学习研究 

**Authors**: Abhishek Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2502.10394)  

**Abstract**: Recent progress in Learning by Reading and Machine Reading systems has significantly increased the capacity of knowledge-based systems to learn new facts. In this work, we discuss the problem of selecting a set of learning requests for these knowledge-based systems which would lead to maximum Q/A performance. To understand the dynamics of this problem, we simulate the properties of a learning strategy, which sends learning requests to an external knowledge source. We show that choosing an optimal set of facts for these learning systems is similar to a coordination game, and use reinforcement learning to solve this problem. Experiments show that such an approach can significantly improve Q/A performance. 

**Abstract (ZH)**: 近年来，阅读学习系统和机器阅读系统的进展显著提高了基于知识系统的学习新事实的能力。在本研究中，我们讨论了如何选择一套学习请求，这些请求能够使基于知识的系统达到最佳的问答性能。为了理解这个问题的动态性，我们模拟了一个学习策略的特性，该策略将学习请求发送到外部知识源。我们展示了为这些学习系统选择最优事实集类似于协调博弈，并使用强化学习来解决这个问题。实验结果表明，这种方法可以显著提高问答性能。 

---
# Diffusion Models without Classifier-free Guidance 

**Title (ZH)**: 无需分类器无指导的扩散模型 

**Authors**: Zhicong Tang, Jianmin Bao, Dong Chen, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.12154)  

**Abstract**: This paper presents Model-guidance (MG), a novel objective for training diffusion model that addresses and removes of the commonly used Classifier-free guidance (CFG). Our innovative approach transcends the standard modeling of solely data distribution to incorporating the posterior probability of conditions. The proposed technique originates from the idea of CFG and is easy yet effective, making it a plug-and-play module for existing models. Our method significantly accelerates the training process, doubles the inference speed, and achieve exceptional quality that parallel and even surpass concurrent diffusion models with CFG. Extensive experiments demonstrate the effectiveness, efficiency, scalability on different models and datasets. Finally, we establish state-of-the-art performance on ImageNet 256 benchmarks with an FID of 1.34. Our code is available at this https URL. 

**Abstract (ZH)**: 本文提出了一种新的训练扩散模型的目标——模型引导（Model-guidance, MG），该目标解决了并克服了常用的无分类器线索指导（Classifier-free guidance, CFG）的常见问题。我们创新的方法超越了仅模型数据分布的标准建模方式，将条件的后验概率纳入模型中。这种方法基于CFG的想法，既简单又有效，可作为现有模型的即插即用模块。我们的方法显著加快了训练过程，将推理速度提高了一倍，并实现了与使用CFG的同代扩散模型相当甚至更优的质量。大量的实验验证了该方法在不同模型和数据集上的有效性、高效性和可扩展性。最后，我们在ImageNet 256基准测试中取得了最先进的性能， fid值为1.34。我们的代码可在以下链接获取：this https URL。 

---
# HARBOR: Exploring Persona Dynamics in Multi-Agent Competition 

**Title (ZH)**: HARBOR：多智能体竞争中人格动态的探索 

**Authors**: Kenan Jiang, Li Xiong, Fei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12149)  

**Abstract**: We investigate factors contributing to LLM agents' success in competitive multi-agent environments, using auctions as a testbed where agents bid to maximize profit. The agents are equipped with bidding domain knowledge, distinct personas that reflect item preferences, and a memory of auction history. Our work extends the classic auction scenario by creating a realistic environment where multiple agents bid on houses, weighing aspects such as size, location, and budget to secure the most desirable homes at the lowest prices. Particularly, we investigate three key questions: (a) How does a persona influence an agent's behavior in a competitive setting? (b) Can an agent effectively profile its competitors' behavior during auctions? (c) How can persona profiling be leveraged to create an advantage using strategies such as theory of mind? Through a series of experiments, we analyze the behaviors of LLM agents and shed light on new findings. Our testbed, called HARBOR, offers a valuable platform for deepening our understanding of multi-agent workflows in competitive environments. 

**Abstract (ZH)**: 我们研究了影响大规模语言模型（LLM）代理在竞争性多代理环境中的成功因素，利用拍卖作为测试平台，在拍卖中代理通过出价来最大化利润。这些代理配备了拍卖领域的知识、反映物品偏好的不同个性，以及拍卖历史的记忆。我们的研究在此经典的拍卖场景基础上，创造了一个现实的环境，其中多个代理竞拍房屋，综合考虑尺寸、位置和预算等因素，以在最低价格下获得最理想的房子。特别是，我们探讨了三个关键问题：（a）个性如何影响代理在竞争性环境中的行为？（b）代理是否能够有效地对其竞争对手在拍卖中的行为进行画像？（c）如何利用个性画像来通过策略（如理论心智）获得优势？通过一系列实验，我们分析了LLM代理的行为，并揭示了新的研究成果。我们的测试平台称为HARBOR，提供了一个深入理解竞争性环境中多代理流程的重要平台。 

---
# Fast or Better? Balancing Accuracy and Cost in Retrieval-Augmented Generation with Flexible User Control 

**Title (ZH)**: 快还是好？在具有灵活用户控制的检索增强生成中平衡准确性和成本 

**Authors**: Jinyan Su, Jennifer Healey, Preslav Nakov, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2502.12145)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful approach to mitigate large language model (LLM) hallucinations by incorporating external knowledge retrieval. However, existing RAG frameworks often apply retrieval indiscriminately,leading to inefficiencies-over-retrieving when unnecessary or failing to retrieve iteratively when required for complex reasoning. Recent adaptive retrieval strategies, though adaptively navigates these retrieval strategies, predict only based on query complexity and lacks user-driven flexibility, making them infeasible for diverse user application needs. In this paper, we introduce a novel user-controllable RAG framework that enables dynamic adjustment of the accuracy-cost trade-off. Our approach leverages two classifiers: one trained to prioritize accuracy and another to prioritize retrieval efficiency. Via an interpretable control parameter $\alpha$, users can seamlessly navigate between minimal-cost retrieval and high-accuracy retrieval based on their specific requirements. We empirically demonstrate that our approach effectively balances accuracy, retrieval cost, and user controllability, making it a practical and adaptable solution for real-world applications. 

**Abstract (ZH)**: 检索增强生成（RAG）作为一种通过引入外部知识检索来减轻大规模语言模型（LLM）妄想的方法而崭露头角。然而，现有的RAG框架往往在没有必要的情况下进行不加选择的检索，导致过度检索的效率低下；而在需要进行复杂推理时，却未能进行迭代检索。尽管最近的自适应检索策略能够在一定程度上解决这些问题，但它们仅基于查询复杂性进行预测，缺乏用户驱动的灵活性，从而无法满足多样化的用户应用需求。在本文中，我们提出了一种新型的用户可控的RAG框架，能够动态调整准确性和成本之间的权衡。我们的方法利用了两个分类器：一个用于优先考虑准确性，另一个则优先考虑检索效率。通过一个可解释的控制参数α，用户可以根据具体需求在最小成本检索和高准确率检索之间无缝切换。我们通过实验证明，我们的方法能够有效地平衡准确率、检索成本和用户可控性，使之成为实际应用中一个实用且适应性强的解决方案。 

---
# LaM-SLidE: Latent Space Modeling of Spatial Dynamical Systems via Linked Entities 

**Title (ZH)**: LaM-SLidE：基于链接实体的空间动态系统潜在空间建模方法 

**Authors**: Florian Sestak, Artur Toshev, Andreas Fürst, Günter Klambauer, Andreas Mayr, Johannes Brandstetter  

**Link**: [PDF](https://arxiv.org/pdf/2502.12128)  

**Abstract**: Generative models are spearheading recent progress in deep learning, showing strong promise for trajectory sampling in dynamical systems as well. However, while latent space modeling paradigms have transformed image and video generation, similar approaches are more difficult for most dynamical systems. Such systems -- from chemical molecule structures to collective human behavior -- are described by interactions of entities, making them inherently linked to connectivity patterns and the traceability of entities over time. Our approach, LaM-SLidE (Latent Space Modeling of Spatial Dynamical Systems via Linked Entities), combines the advantages of graph neural networks, i.e., the traceability of entities across time-steps, with the efficiency and scalability of recent advances in image and video generation, where pre-trained encoder and decoder are frozen to enable generative modeling in the latent space. The core idea of LaM-SLidE is to introduce identifier representations (IDs) to allow for retrieval of entity properties, e.g., entity coordinates, from latent system representations and thus enables traceability. Experimentally, across different domains, we show that LaM-SLidE performs favorably in terms of speed, accuracy, and generalizability. (Code is available at this https URL) 

**Abstract (ZH)**: 生成模型在最近的深度学习进展中起到了主导作用，展示了在动态系统中轨迹采样的强大潜力。然而，尽管潜在空间建模范式已经改变了图像和视频的生成方式，但对于大多数动态系统而言，采用类似的方法更加困难。这类系统——从化学分子结构到集体人类行为——由实体间的相互作用描述，因此与实体间的连通模式和时间轨迹性本就密切相关。我们提出的方法 LaM-SLidE（基于关联实体的空间动态系统潜在空间建模）结合了图神经网络的优点，即通过时间步的实体可追踪性，以及图像和视频生成领域近期进展的高效性和可扩展性，其中预训练的编码器和解码器被冻结，以在潜在空间中实现生成建模。LaM-SLidE 的核心思想是引入标识符表示（IDs），以允许从潜在系统表示中检索实体属性，如实体坐标，从而实现可追踪性。实验表明，在不同领域中，LaM-SLidE 在速度、准确性和泛化性上表现优越。（代码可从这个链接访问：[提供链接]） 

---
# LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws 

**Title (ZH)**: 标题翻译如下，符合学术规范：

LLMs 在线性中的数据决定损失归一化定律

如果需要更具学术性的表达，可以翻译为：

线性中的大规模语言模型：数据驱动的损失归一化定律 

**Authors**: Prasanna Mayilvahanan, Thaddäus Wiedemer, Sayak Mallick, Matthias Bethge, Wieland Brendel  

**Link**: [PDF](https://arxiv.org/pdf/2502.12120)  

**Abstract**: Scaling laws guide the development of large language models (LLMs) by offering estimates for the optimal balance of model size, tokens, and compute. More recently, loss-to-loss scaling laws that relate losses across pretraining datasets and downstream tasks have emerged as a powerful tool for understanding and improving LLM performance. In this work, we investigate which factors most strongly influence loss-to-loss scaling. Our experiments reveal that the pretraining data and tokenizer determine the scaling trend. In contrast, model size, optimization hyperparameters, and even significant architectural differences, such as between transformer-based models like Llama and state-space models like Mamba, have limited impact. Consequently, practitioners should carefully curate suitable pretraining datasets for optimal downstream performance, while architectures and other settings can be freely optimized for training efficiency. 

**Abstract (ZH)**: 规模律指导大型语言模型（LLMs）的发展，通过提供模型大小、标记数和计算资源之间最优平衡的估算。最近，关联预训练数据集和下游任务损失的损失到损失规模律（loss-to-loss scaling laws）成为了理解并提升LLM性能的强大工具。在本研究中，我们探讨了哪些因素最能影响损失到损失的规模律。我们的实验表明，预训练数据和分词器决定了规模律的趋势。相比之下，模型大小、优化超参数，甚至诸如Transformer基模型（如Llama）和状态空间模型（如Mamba）之间的重要架构差异对规模律的影响有限。因此，从业者应该仔细选择适合的预训练数据集以获得最佳的下游性能，而架构和其他设置则可以自由优化以提高训练效率。 

---
# PRISM: Self-Pruning Intrinsic Selection Method for Training-Free Multimodal Data Selection 

**Title (ZH)**: PRISM：无需训练的多模态数据选择自我修剪内在选择方法 

**Authors**: Jinhe Bi, Yifan Wang, Danqi Yan, Xun Xiao, Artur Hecker, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2502.12119)  

**Abstract**: Visual instruction tuning refines pre-trained Multimodal Large Language Models (MLLMs) to enhance their real-world task performance. However, the rapid expansion of visual instruction datasets introduces significant data redundancy, leading to excessive computational costs. Existing data selection methods predominantly rely on proxy models or loss-based metrics, both of which impose substantial computational overheads due to the necessity of model inference and backpropagation. To address this challenge, we propose PRISM, a novel training-free approach for efficient multimodal data selection. Unlike existing methods, PRISM eliminates the reliance on proxy models, warm-up pretraining, and gradient-based optimization. Instead, it leverages Pearson correlation analysis to quantify the intrinsic visual encoding properties of MLLMs, computing a task-specific correlation score to identify high-value instances. This not only enbles data-efficient selection,but maintains the original performance. Empirical evaluations across multiple MLLMs demonstrate that PRISM reduces the overall time required for visual instruction tuning and data selection to just 30% of conventional methods, while surpassing fully fine-tuned models across eight multimodal and three language understanding benchmarks, achieving a 101.7% relative improvement in final performance. 

**Abstract (ZH)**: 视觉指令调优可以精炼预训练的多模态大型语言模型（MLLMs），以增强其实际任务性能。然而，视觉指令数据集的迅速扩展引入了大量数据冗余，导致了高昂的计算成本。目前的数据选择方法主要依赖代理模型或基于损失的度量标准，这两种方法由于需要模型推理和反向传播，从而带来了显著的计算开销。为解决这一挑战，我们提出了一种名为PRISM的新型无训练数据选择方法。与现有方法不同，PRISM无需依赖代理模型、预训练模型的暖启动，以及基于梯度的优化。相反，它利用皮尔逊相关分析来量化MLLMs的固有视觉编码特性，计算出任务特定的相关性得分以识别具有高价值的实例。这不仅提高了数据选择的效率，同时保持了原始性能。在多个MLLM上的实证评估表明，PRISM将视觉指令调优和数据选择所需的整体时间缩短至传统方法的30%，同时在八个跨模态和三个语言理解基准测试中超过了完全微调的模型，最终性能提高了101.7%。 

---
# Personality Structured Interview for Large Language Model Simulation in Personality Research 

**Title (ZH)**: 个性结构化面试在人格研究中对大型语言模型模拟的应用 

**Authors**: Pengda Wang, Huiqi Zou, Hanjie Chen, Tianjun Sun, Ziang Xiao, Frederick L. Oswald  

**Link**: [PDF](https://arxiv.org/pdf/2502.12109)  

**Abstract**: Although psychometrics researchers have recently explored the use of large language models (LLMs) as proxies for human participants, LLMs often fail to generate heterogeneous data with human-like diversity, which diminishes their value in advancing social science research. To address these challenges, we explored the potential of the theory-informed Personality Structured Interview (PSI) as a tool for simulating human responses in personality research. In this approach, the simulation is grounded in nuanced real-human interview transcripts that target the personality construct of interest. We have provided a growing set of 357 structured interview transcripts from a representative sample, each containing an individual's response to 32 open-ended questions carefully designed to gather theory-based personality evidence. Additionally, grounded in psychometric research, we have summarized an evaluation framework to systematically validate LLM-generated psychometric data. Results from three experiments demonstrate that well-designed structured interviews could improve human-like heterogeneity in LLM-simulated personality data and predict personality-related behavioral outcomes (i.e., organizational citizenship behaviors and counterproductive work behavior). We further discuss the role of theory-informed structured interviews in LLM-based simulation and outline a general framework for designing structured interviews to simulate human-like data for psychometric research. 

**Abstract (ZH)**: 尽管心理测量学研究者最近已经开始探索使用大型语言模型（LLMs）作为人类参与者代理的可能性，但LLMs往往无法生成具有人类多样性的异质数据，这削弱了它们在促进社会科学研究方面的作用。为应对这些挑战，我们探讨了理论导向的性格结构化访谈（PSI）作为模拟性格研究中人类反应工具的潜在价值。在该方法中，模拟基于对目标性格构建的细致真实人类访谈记录。我们提供了一组不断增长的357份结构化访谈记录，其中每份记录包含一个个体对精心设计的32个开放性问题的回应，这些问题旨在收集基于理论的性格证据。此外，基于心理测量学研究，我们总结了一个评估框架，以系统验证LLM生成的心理测量数据。三项实验的结果表明，精心设计的结构化访谈可以提高LLM模拟性格数据中的拟人类异质性，并预测与性格相关的行为结果（如组织公民行为和破坏性工作行为）。最后，我们讨论了理论导向的结构化访谈在LLM基础上模拟中的作用，并概述了一个设计结构化访谈以模拟拟人类数据的通用框架，用于心理测量学研究。 

---
# Using the Path of Least Resistance to Explain Deep Networks 

**Title (ZH)**: 使用最小阻力路径来解释深度网络 

**Authors**: Sina Salek, Joseph Enguehard  

**Link**: [PDF](https://arxiv.org/pdf/2502.12108)  

**Abstract**: Integrated Gradients (IG), a widely used axiomatic path-based attribution method, assigns importance scores to input features by integrating model gradients along a straight path from a baseline to the input. While effective in some cases, we show that straight paths can lead to flawed attributions. In this paper, we identify the cause of these misattributions and propose an alternative approach that treats the input space as a Riemannian manifold, computing attributions by integrating gradients along geodesics. We call this method Geodesic Integrated Gradients (GIG). To approximate geodesic paths, we introduce two techniques: a k-Nearest Neighbours-based approach for smaller models and a Stochastic Variational Inference-based method for larger ones. Additionally, we propose a new axiom, Strong Completeness, extending the axioms satisfied by IG. We show that this property is desirable for attribution methods and that GIG is the only method that satisfies it. Through experiments on both synthetic and real-world data, we demonstrate that GIG outperforms existing explainability methods, including IG. 

**Abstract (ZH)**: 集成梯度（IG），一种广泛使用的公理化路径归因方法，通过沿从基线到输入的直线路径整合模型梯度来为输入特征分配重要性分数。尽管在某些情况下有效，但我们展示出直线路径可能导致归因错误。在这篇论文中，我们识别出这些错误归因的原因，并提出了一种替代方法，将输入空间视为黎曼流形，通过沿测地线整合梯度来计算归因。我们称这种方法为测地线集成梯度（GIG）。

为了近似测地线路径，我们引入了两种技术：一种基于k-最近邻的方法适用于较小的模型，而一种基于随机变分推断的方法适用于较大的模型。此外，我们提出了一种新的公理——强完备性，它扩展了IG满足的公理。我们展示了这种性质对于归因方法是非常理想的，并且GIG是唯一满足该性质的方法。通过在合成数据和实际数据上的实验，我们证明了GIG在可解释性方法中优于现有方法，包括IG。 

---
# Meta-Statistical Learning: Supervised Learning of Statistical Inference 

**Title (ZH)**: 元统计学习：监督学习中的统计推理学习 

**Authors**: Maxime Peyrard, Kyunghyun Cho  

**Link**: [PDF](https://arxiv.org/pdf/2502.12088)  

**Abstract**: This work demonstrates that the tools and principles driving the success of large language models (LLMs) can be repurposed to tackle distribution-level tasks, where the goal is to predict properties of the data-generating distribution rather than labels for individual datapoints. These tasks encompass statistical inference problems such as parameter estimation, hypothesis testing, or mutual information estimation. Framing these tasks within traditional machine learning pipelines is challenging, as supervision is typically tied to individual datapoint. We propose meta-statistical learning, a framework inspired by multi-instance learning that reformulates statistical inference tasks as supervised learning problems. In this approach, entire datasets are treated as single inputs to neural networks, which predict distribution-level parameters. Transformer-based architectures, without positional encoding, provide a natural fit due to their permutation-invariance properties. By training on large-scale synthetic datasets, meta-statistical models can leverage the scalability and optimization infrastructure of Transformer-based LLMs. We demonstrate the framework's versatility with applications in hypothesis testing and mutual information estimation, showing strong performance, particularly for small datasets where traditional neural methods struggle. 

**Abstract (ZH)**: 本研究展示了可以将驱动大型语言模型（LLMs）成功的关键工具和技术应用于处理数据分布级别任务中，这些任务的目标是预测数据生成分布的性质，而不是单个数据点的标签。此类任务包括统计推断问题，如参数估计、假设检验或互信息估计。将这些任务嵌入传统机器学习管道中存在挑战，因为监督通常与单个数据点相关。我们提出了一种元统计学习框架，该框架借鉴了多实例学习的思想，将统计推断任务重新表述为监督学习问题。在该方法中，整个数据集被视为神经网络的单一输入，用于预测数据分布级别的参数。基于变压器的架构，由于其排他性不变性特性，非常适合这种应用场景。通过在大规模合成数据集上进行训练，元统计模型可以利用基于变压器的LLMs的可扩展性和优化基础设施。我们通过在假设检验和互信息估计中的应用展示了该框架的灵活性，并且在传统神经方法难以应对的小数据集上表现出强大性能。 

---
# TokenSkip: Controllable Chain-of-Thought Compression in LLMs 

**Title (ZH)**: TokenSkip：LLM中可控的链式思维压缩 

**Authors**: Heming Xia, Yongqi Li, Chak Tou Leong, Wenjie Wang, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.12067)  

**Abstract**: Chain-of-Thought (CoT) has been proven effective in enhancing the reasoning capabilities of large language models (LLMs). Recent advancements, such as OpenAI's o1 and DeepSeek-R1, suggest that scaling up the length of CoT sequences during inference could further boost LLM reasoning performance. However, due to the autoregressive nature of LLM decoding, longer CoT outputs lead to a linear increase in inference latency, adversely affecting user experience, particularly when the CoT exceeds 10,000 tokens. To address this limitation, we analyze the semantic importance of tokens within CoT outputs and reveal that their contributions to reasoning vary. Building on this insight, we propose TokenSkip, a simple yet effective approach that enables LLMs to selectively skip less important tokens, allowing for controllable CoT compression. Extensive experiments across various models and tasks demonstrate the effectiveness of TokenSkip in reducing CoT token usage while preserving strong reasoning performance. Notably, when applied to Qwen2.5-14B-Instruct, TokenSkip reduces reasoning tokens by 40% (from 313 to 181) on GSM8K, with less than a 0.4% performance drop. 

**Abstract (ZH)**: 链式思维（CoT）已被证明能够增强大型语言模型（LLMs）的推理能力。最近的研究进展，例如OpenAI的o1和DeepSeek-R1，表明在推理过程中扩展CoT序列的长度可以进一步提高LLM的推理性能。然而，由于LLM解码的自回归性质，在生成更长的CoT输出时，推理延迟会线性增加，从而影响用户体验，特别是当CoT超过10,000个token时。为了解决这一限制，我们分析了CoT输出中token的语义重要性，并发现它们对推理的贡献不同。基于这一洞察，我们提出了一种简单而有效的TokenSkip方法，允许LLMs在保持推理性能的同时选择性地跳过不重要的token，从而实现可控的CoT压缩。在各种模型和任务的广泛实验中，TokenSkip在减少CoT token使用量的同时保持了强大的推理性能。值得注意的是，当应用于Qwen2.5-14B-Instruct时，TokenSkip在GSM8K数据集上将推理token数量减少了40%（从313减少到181），并且性能下降不到0.4%。 

---
# AI-generated Text Detection with a GLTR-based Approach 

**Title (ZH)**: 基于GLTR方法的AI生成文本检测 

**Authors**: Lucía Yan Wu, Isabel Segura-Bedmar  

**Link**: [PDF](https://arxiv.org/pdf/2502.12064)  

**Abstract**: The rise of LLMs (Large Language Models) has contributed to the improved performance and development of cutting-edge NLP applications. However, these can also pose risks when used maliciously, such as spreading fake news, harmful content, impersonating individuals, or facilitating school plagiarism, among others. This is because LLMs can generate high-quality texts, which are challenging to differentiate from those written by humans. GLTR, which stands for Giant Language Model Test Room and was developed jointly by the MIT-IBM Watson AI Lab and HarvardNLP, is a visual tool designed to help detect machine-generated texts based on GPT-2, that highlights the words in text depending on the probability that they were machine-generated. One limitation of GLTR is that the results it returns can sometimes be ambiguous and lead to confusion. This study aims to explore various ways to improve GLTR's effectiveness for detecting AI-generated texts within the context of the IberLef-AuTexTification 2023 shared task, in both English and Spanish languages. Experiment results show that our GLTR-based GPT-2 model overcomes the state-of-the-art models on the English dataset with a macro F1-score of 80.19%, except for the first ranking model (80.91%). However, for the Spanish dataset, we obtained a macro F1-score of 66.20%, which differs by 4.57% compared to the top-performing model. 

**Abstract (ZH)**: 大型语言模型（LLM）的兴起已经提高了前沿自然语言处理（NLP）应用的性能和开发水平。然而，当这些模型被恶意使用时，它们也可能带来风险，例如散布假新闻、传播有害内容、冒充个人或促进学术抄袭等。这是因为LLM能够生成高质量的文本，这些文本难以与人类撰写的文本区分开来。GLTR（Giant Language Model Test Room）是麻省理工学院IBM沃森AI实验室和哈佛NLP联合开发的一款可视化工具，旨在通过基于GPT-2的模型来检测机器生成的文本，该模型通过高亮显示文本中的词语来标注其可能是机器生成的概率。GLTR的一个局限是，它返回的结果有时会模棱两可，导致混淆。本研究旨在探索在2023年IberLef-AuTexTification共享任务（该任务包括英语和西班牙语）背景下，改进GLTR检测AI生成文本效果的各种方法。实验结果表明，基于GLTR的GPT-2模型在英语数据集上以宏F1分数80.19%超越了现有最先进的模型，除了排名第一的模型（80.91%）。然而，对于西班牙语数据集，我们得到了宏F1分数66.20%，这与表现最佳的模型有4.57%的差距。 

---
# Masked Latent Prediction and Classification for Self-Supervised Audio Representation Learning 

**Title (ZH)**: 掩码潜变量预测与分类用于自监督音频表示学习 

**Authors**: Aurian Quelennec, Pierre Chouteau, Geoffroy Peeters, Slim Essid  

**Link**: [PDF](https://arxiv.org/pdf/2502.12031)  

**Abstract**: Recently, self-supervised learning methods based on masked latent prediction have proven to encode input data into powerful representations. However, during training, the learned latent space can be further transformed to extract higher-level information that could be more suited for downstream classification tasks. Therefore, we propose a new method: MAsked latenT Prediction And Classification (MATPAC), which is trained with two pretext tasks solved jointly. As in previous work, the first pretext task is a masked latent prediction task, ensuring a robust input representation in the latent space. The second one is unsupervised classification, which utilises the latent representations of the first pretext task to match probability distributions between a teacher and a student. We validate the MATPAC method by comparing it to other state-of-the-art proposals and conducting ablations studies. MATPAC reaches state-of-the-art self-supervised learning results on reference audio classification datasets such as OpenMIC, GTZAN, ESC-50 and US8K and outperforms comparable supervised methods results for musical auto-tagging on Magna-tag-a-tune. 

**Abstract (ZH)**: 近年来，基于掩蔽潜在预测的自我监督学习方法已被证明能够将输入数据编码为强大的表示。然而，在训练过程中，学习到的潜在空间可以进一步转化以提取更高层次的信息，这些信息可能更适合下游分类任务。因此，我们提出了一种新方法：Masked Latent Prediction and Classification (MATPAC)，该方法通过联合解决两个前任务进行训练。与先前工作类似，第一个前任务是一个掩蔽潜在预测任务，确保潜在空间中的鲁棒输入表示。第二个任务是无监督分类，利用第一个前任务的潜在表示，使教师和学生之间的概率分布匹配。我们通过将其与其他最先进的提议进行对比，并进行消融研究，验证了MATPAC方法的效果。MATPAC在参考音频分类数据集如OpenMIC、GTZAN、ESC-50和US8K上达到了最先进的自我监督学习结果，并在Magna-tag-a-tune上的音乐自动标签任务上超过了同等的监督方法的结果。 

---
# Teaching LLMs According to Their Aptitude: Adaptive Reasoning for Mathematical Problem Solving 

**Title (ZH)**: 根据LLM的能力进行教学：数学问题解决中的自适应推理 

**Authors**: Xin Xu, Yan Xu, Tianhao Chen, Yuchen Yan, Chengwu Liu, Zaoyu Chen, Yufei Wang, Yichun Yin, Yasheng Wang, Lifeng Shang, Qun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.12022)  

**Abstract**: Existing approaches to mathematical reasoning with large language models (LLMs) rely on Chain-of-Thought (CoT) for generalizability or Tool-Integrated Reasoning (TIR) for precise computation. While efforts have been made to combine these methods, they primarily rely on post-selection or predefined strategies, leaving an open question: whether LLMs can autonomously adapt their reasoning strategy based on their inherent capabilities. In this work, we propose TATA (Teaching LLMs According to Their Aptitude), an adaptive framework that enables LLMs to personalize their reasoning strategy spontaneously, aligning it with their intrinsic aptitude. TATA incorporates base-LLM-aware data selection during supervised fine-tuning (SFT) to tailor training data to the model's unique abilities. This approach equips LLMs to autonomously determine and apply the appropriate reasoning strategy at test time. We evaluate TATA through extensive experiments on six mathematical reasoning benchmarks, using both general-purpose and math-specialized LLMs. Empirical results demonstrate that TATA effectively combines the complementary strengths of CoT and TIR, achieving superior or comparable performance with improved inference efficiency compared to TIR alone. Further analysis underscores the critical role of aptitude-aware data selection in enabling LLMs to make effective and adaptive reasoning decisions and align reasoning strategies with model capabilities. 

**Abstract (ZH)**: 现有的大规模语言模型（LLMs）在进行数学推理时的方法主要依赖于Chain-of-Thought（CoT）以实现通用性，或依赖于Tool-Integrated Reasoning（TIR）以实现精确计算。尽管已经有人尝试将这两种方法结合起来，但它们主要依赖于后筛选或预先定义的策略，这留下了一个开放的问题：LLMs能否根据其固有的能力自主适应其推理策略。在这项工作中，我们提出了TATA（Teaching LLMs According to Their Aptitude）自适应框架，该框架能够使LLMs自发地个性化其推理策略，使其与固有的能力相匹配。TATA在监督微调（SFT）过程中结合了对基模型的意识，从而定制训练数据以适应模型的独特能力。这种方法使LLMs能够在测试时自主确定并应用合适的推理策略。我们通过在六个数学推理基准上的广泛实验评估了TATA，使用了通用和数学专门化的LLMs。实验证明，TATA有效地结合了CoT和TIR的优势，与仅使用TIR相比，在提高推理效率的同时实现了更优或相当的性能。进一步的分析强调了基于能力的数据选择在使LLMs能够做出有效和适应性推理决策并使推理策略与模型能力相一致方面所起的关键作用。 

---
# Atom of Thoughts for Markov LLM Test-Time Scaling 

**Title (ZH)**: 原子级思维：马尔可夫链大语言模型测试时缩放方法 

**Authors**: Fengwei Teng, Zhaoyang Yu, Quan Shi, Jiayi Zhang, Chenglin Wu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.12018)  

**Abstract**: Large Language Models (LLMs) achieve superior performance through training-time scaling, and test-time scaling further enhances their capabilities by conducting effective reasoning during inference. However, as the scale of reasoning increases, existing test-time scaling methods suffer from accumulated historical information, which not only wastes computational resources but also interferes with effective reasoning. To address this issue, we observe that complex reasoning progress is often achieved by solving a sequence of independent subquestions, each being self-contained and verifiable. These subquestions are essentially atomic questions, relying primarily on their current state rather than accumulated history, similar to the memoryless transitions in a Markov process. Based on this observation, we propose Atom of Thoughts (AoT), where each state transition in the reasoning process consists of decomposing the current question into a dependency-based directed acyclic graph and contracting its subquestions, forming a new atomic question state. This iterative decomposition-contraction process continues until reaching directly solvable atomic questions, naturally realizing Markov transitions between question states. Furthermore, these atomic questions can be seamlessly integrated into existing test-time scaling methods, enabling AoT to serve as a plug-in enhancement for improving reasoning capabilities. Experiments across six benchmarks demonstrate the effectiveness of AoT both as a standalone framework and a plug-in enhancement. Notably, on HotpotQA, when applied to gpt-4o-mini, AoT achieves an 80.6% F1 score, surpassing o3-mini by 3.4% and DeepSeek-R1 by 10.6%. The code will be available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）通过训练时的扩展实现了卓越的性能，并且在测试时进一步通过有效的推理增强其能力。然而，随着推理规模的增加，现有的测试时扩展方法会遭受累积历史信息的困扰，这不仅浪费了计算资源，还干扰了有效的推理。为了解决这一问题，我们观察到复杂的推理过程经常是通过解决一系列独立的子问题来实现的，每个子问题都是自包含且可验证的。这些子问题本质上是原子型问题，主要依赖于当前状态而非累积的历史信息，类似于马尔可夫过程中的无记忆转换。基于这一观察，我们提出了“思想原子”（AoT，Atom of Thoughts），其中推理过程中的每个状态转换包含将当前问题分解为基于依赖关系的有向无环图以及收缩其子问题，形成一个新的原子型问题状态。这一递归的分解-收缩过程将持续进行，直至达到可以直接求解的原子型问题，自然地实现了问题状态之间的马尔可夫转换。此外，这些原子性问题可以无缝集成到现有的测试时扩展方法中，使得AoT能够作为插件增强工具来提升推理能力。在六个基准测试上的实验表明，AoT作为一个独立的框架和插件增强工具都具有有效性。特别是在HotpotQA上，当应用于gpt-4o-mini时，AoT的F1分数达到了80.6%，分别超过了o3-mini的0.34%和DeepSeek-R1的10.6%。源代码将在此处提供：<这个链接>。 

---
# Evolving Hard Maximum Cut Instances for Quantum Approximate Optimization Algorithms 

**Title (ZH)**: 演化出适合量子近似优化算法的硬最大割实例 

**Authors**: Shuaiqun Pan, Yash J. Patel, Aneta Neumann, Frank Neumann, Thomas Bäck, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.12012)  

**Abstract**: Variational quantum algorithms, such as the Recursive Quantum Approximate Optimization Algorithm (RQAOA), have become increasingly popular, offering promising avenues for employing Noisy Intermediate-Scale Quantum devices to address challenging combinatorial optimization tasks like the maximum cut problem. In this study, we utilize an evolutionary algorithm equipped with a unique fitness function. This approach targets hard maximum cut instances within the latent space of a Graph Autoencoder, identifying those that pose significant challenges or are particularly tractable for RQAOA, in contrast to the classic Goemans and Williamson algorithm. Our findings not only delineate the distinct capabilities and limitations of each algorithm but also expand our understanding of RQAOA's operational limits. Furthermore, the diverse set of graphs we have generated serves as a crucial benchmarking asset, emphasizing the need for more advanced algorithms to tackle combinatorial optimization challenges. Additionally, our results pave the way for new avenues in graph generation research, offering exciting opportunities for future explorations. 

**Abstract (ZH)**: 变分量子算法（如递归量子近似优化算法RQAOA）近年来越来越受欢迎，这些算法为利用噪声中等规模量子设备解决如最大割问题在内的复杂组合优化任务提供了有希望的途径。在本研究中，我们利用了一个配备有独特适应函数的进化算法。该方法针对图自编码器潜在空间中的硬最大割实例进行目标定位，既识别那些对RQAOA构成重大挑战的实例，也识别那些特别适合RQAOA的实例，这与经典的Geometric和Williamson算法形成对比。我们的研究不仅界定了每种算法的独特能力和局限性，还扩展了我们对RQAOA操作极限的理解。此外，我们生成的多种图类型不仅为基准测试提供了关键资产，还突显了开发更先进算法以应对组合优化挑战的必要性。此外，我们的结果为图生成研究开辟了新的方向，提供了未来探索的激动人心的机会。 

---
# Demographic Attributes Prediction from Speech Using WavLM Embeddings 

**Title (ZH)**: 使用WavLM嵌入进行语音的 Demographic 属性预测 

**Authors**: Yuchen Yang, Thomas Thebaud, Najim Dehak  

**Link**: [PDF](https://arxiv.org/pdf/2502.12007)  

**Abstract**: This paper introduces a general classifier based on WavLM features, to infer demographic characteristics, such as age, gender, native language, education, and country, from speech. Demographic feature prediction plays a crucial role in applications like language learning, accessibility, and digital forensics, enabling more personalized and inclusive technologies. Leveraging pretrained models for embedding extraction, the proposed framework identifies key acoustic and linguistic fea-tures associated with demographic attributes, achieving a Mean Absolute Error (MAE) of 4.94 for age prediction and over 99.81% accuracy for gender classification across various datasets. Our system improves upon existing models by up to relative 30% in MAE and up to relative 10% in accuracy and F1 scores across tasks, leveraging a diverse range of datasets and large pretrained models to ensure robustness and generalizability. This study offers new insights into speaker diversity and provides a strong foundation for future research in speech-based demographic profiling. 

**Abstract (ZH)**: 本文介绍了一种基于WavLM特征的一般分类器，用于从语音中推断年龄、性别、母语、教育背景和国籍等人口统计特征。人口统计特征预测在语言学习、无障碍技术以及数字取证等应用中发挥着关键作用，有助于开发更加个性化和包容的技术。利用预训练模型进行嵌入提取，所提出的框架识别与人口统计属性相关的关键声学和语言特征，在年龄预测中实现了4.94的平均绝对误差（MAE），在各类数据集的性别分类中准确率超过99.81%。与现有模型相比，我们的系统在MAE上最多可提高30%，在准确率和F1分数等多个任务上最多可提高10%，通过使用多样化的数据集和大型预训练模型来确保鲁棒性和泛化能力。本研究为语音人口统计特征分析提供了新的见解，并为未来研究奠定了坚实的基础。 

---
# Presumed Cultural Identity: How Names Shape LLM Responses 

**Title (ZH)**: 假设的文化身份：姓名如何塑造大模型的回应 

**Authors**: Siddhesh Pawar, Arnav Arora, Lucie-Aimée Kaffee, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2502.11995)  

**Abstract**: Names are deeply tied to human identity. They can serve as markers of individuality, cultural heritage, and personal history. However, using names as a core indicator of identity can lead to over-simplification of complex identities. When interacting with LLMs, user names are an important point of information for personalisation. Names can enter chatbot conversations through direct user input (requested by chatbots), as part of task contexts such as CV reviews, or as built-in memory features that store user information for personalisation. We study biases associated with names by measuring cultural presumptions in the responses generated by LLMs when presented with common suggestion-seeking queries, which might involve making assumptions about the user. Our analyses demonstrate strong assumptions about cultural identity associated with names present in LLM generations across multiple cultures. Our work has implications for designing more nuanced personalisation systems that avoid reinforcing stereotypes while maintaining meaningful customisation. 

**Abstract (ZH)**: 姓名深深根植于人类身份之中。它们可以作为个人独特性、文化传承和个人历史的标志。然而，将姓名作为身份的核心指标可能会导致对复杂身份的过分简化。在与大规模语言模型（LLM）交互时，用户姓名是个性化的重要信息点。姓名可以通过聊天机器人直接请求的用户输入、作为任务上下文的一部分（例如，在简历审核中）或作为内置记忆功能来进入对话系统，以便存储用户信息进行个性化。我们通过测量LLM在面对常见的建议请求查询时生成的响应中文化假设，来研究与姓名相关的偏见问题，这些查询可能会对用户做出假设。我们的分析表明，在跨文化背景下，LLM生成中对姓名关联的文化身份存在强烈的假设。我们的研究对设计更细致入微的个性化系统具有重要意义，这些系统可以避免强化刻板印象，同时保留有意义的个性化定制。 

---
# Characterizing Photorealism and Artifacts in Diffusion Model-Generated Images 

**Title (ZH)**: 研究扩散模型生成图像中的拟真度与伪影特征 

**Authors**: Negar Kamali, Karyn Nakamura, Aakriti Kumar, Angelos Chatzimparmpas, Jessica Hullman, Matthew Groh  

**Link**: [PDF](https://arxiv.org/pdf/2502.11989)  

**Abstract**: Diffusion model-generated images can appear indistinguishable from authentic photographs, but these images often contain artifacts and implausibilities that reveal their AI-generated provenance. Given the challenge to public trust in media posed by photorealistic AI-generated images, we conducted a large-scale experiment measuring human detection accuracy on 450 diffusion-model generated images and 149 real images. Based on collecting 749,828 observations and 34,675 comments from 50,444 participants, we find that scene complexity of an image, artifact types within an image, display time of an image, and human curation of AI-generated images all play significant roles in how accurately people distinguish real from AI-generated images. Additionally, we propose a taxonomy characterizing artifacts often appearing in images generated by diffusion models. Our empirical observations and taxonomy offer nuanced insights into the capabilities and limitations of diffusion models to generate photorealistic images in 2024. 

**Abstract (ZH)**: 生成的扩散模型图像在外观上可以与真实的照片难以区分，但这些图像通常包含一些特征和不合理之处，这些特征和不合理之处能够揭示它们是通过AI生成的。鉴于 photorealistic AI 生成图像对公众媒体信任造成的挑战，我们进行了一项大规模实验，测量了参与者在450张扩散模型生成的图像和149张真实图像之间的识别准确性。基于收集到的749,828个观察数据和34,675条评论，我们发现，图像的场景复杂度、图像中的特征类型、图像的展示时间以及人类对AI生成图像的处理，都在人们区分真实图像和AI生成图像的准确性中发挥了重要作用。此外，我们还提出了一种分类法，描述了扩散模型生成图像中常出现的特征类型。我们的实证观察和分类法为2024年扩散模型生成逼真图像的能力和局限性提供了细致入微的见解。 

---
# Machine Learning Should Maximize Welfare, Not (Only) Accuracy 

**Title (ZH)**: 机器学习应最大化福祉，而不仅仅是准确率 

**Authors**: Nir Rosenfeld, Haifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11981)  

**Abstract**: Decades of research in machine learning have given us powerful tools for making accurate predictions. But when used in social settings and on human inputs, better accuracy does not immediately translate to better social outcomes. This may not be surprising given that conventional learning frameworks are not designed to express societal preferences -- let alone promote them. This position paper argues that machine learning is currently missing, and can gain much from incorporating, a proper notion of social welfare. The field of welfare economics asks: how should we allocate limited resources to self-interested agents in a way that maximizes social benefit? We argue that this perspective applies to many modern applications of machine learning in social contexts, and advocate for its adoption. Rather than disposing of prediction, we aim to leverage this forte of machine learning for promoting social welfare. We demonstrate this idea by proposing a conceptual framework that gradually transitions from accuracy maximization (with awareness to welfare) to welfare maximization (via accurate prediction). We detail applications and use-cases for which our framework can be effective, identify technical challenges and practical opportunities, and highlight future avenues worth pursuing. 

**Abstract (ZH)**: 几十年来的机器学习研究为我们提供了强大的工具，用于进行准确的预测。但在社会环境中使用这些工具，以及在人类输入中，更高的准确性并不立即转化为更好的社会成果。这并不令人意外，因为传统的学习框架并未设计用来表达社会偏好，更不用说促进它们了。本文简报认为，当前的机器学习缺乏一个适当的社会福利概念，并认为将其纳入可以大大提升其效果。福利经济学的研究问题在于：我们应如何分配有限的资源给自私的代理人，以最大化社会收益？我们认为，这种视角适用于许多现代机器学习在社会中的应用情境，并倡导采用这一视角。我们不是摒弃预测，而是希望通过准确的预测来促进社会福利。我们通过提出一个概念框架来证明这一观点，该框架从以社会福利意识最大化准确度逐渐过渡到通过准确性最大化社会福利。我们详细介绍了该框架可以有效应用于的应用场景，指出了技术挑战和实用机会，并突出了值得进一步探索的领域。 

---
# Theoretical Barriers in Bellman-Based Reinforcement Learning 

**Title (ZH)**: 基于贝尔曼方程的强化学习中的理论障碍 

**Authors**: Brieuc Pinon, Raphaël Jungers, Jean-Charles Delvenne  

**Link**: [PDF](https://arxiv.org/pdf/2502.11968)  

**Abstract**: Reinforcement Learning algorithms designed for high-dimensional spaces often enforce the Bellman equation on a sampled subset of states, relying on generalization to propagate knowledge across the state space. In this paper, we identify and formalize a fundamental limitation of this common approach. Specifically, we construct counterexample problems with a simple structure that this approach fails to exploit. Our findings reveal that such algorithms can neglect critical information about the problems, leading to inefficiencies. Furthermore, we extend this negative result to another approach from the literature: Hindsight Experience Replay learning state-to-state reachability. 

**Abstract (ZH)**: 设计用于高维空间的强化学习算法通常会在采样的状态子集上强制应用贝尔曼方程，依赖于泛化能力来在整个状态空间传播知识。在本文中，我们识别并形式化了这种常见方法的一个根本局限性。具体来说，我们构建了几个具有简单结构的反例问题，该方法在这种情况下无法充分加以利用。我们的发现表明，这类算法可能会忽略有关问题的关键信息，导致效率低下。此外，我们还将这一负面结果扩展到了文献中的另一种方法：事后经验回放学习状态到状态的可达性。 

---
# A MIMO Wireless Channel Foundation Model via CIR-CSI Consistency 

**Title (ZH)**: 基于CIR-CSI一致性的一种MIMO无线信道基础模型 

**Authors**: Jun Jiang, Wenjun Yu, Yunfan Li, Yuan Gao, Shugong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11965)  

**Abstract**: In the field of artificial intelligence, self-supervised learning has demonstrated superior generalization capabilities by leveraging large-scale unlabeled datasets for pretraining, which is especially critical for wireless communication models to adapt to a variety of scenarios. This paper innovatively treats Channel State Information (CSI) and Channel Impulse Response (CIR) as naturally aligned multi-modal data and proposes the first MIMO wireless channel foundation model, named CSI-CLIP. By effectively capturing the joint representations of both CIR and CSI, CSI-CLIP exhibits remarkable adaptability across scenarios and robust feature extraction capabilities. Experimental results show that in positioning task, CSI-CLIP reduces the mean error distance by 22%; in beam management task, it increases accuracy by 1% compared to traditional supervised methods, as well as in the channel identification task. These improvements not only highlight the potential and value of CSI-CLIP in integrating sensing and communication but also demonstrate its significant advantages over existing techniques. Moreover, viewing CSI and CIR as multi-modal pairs and contrastive learning for wireless channel foundation model open up new research directions in the domain of MIMO wireless communications. 

**Abstract (ZH)**: 在人工智能领域，自监督学习通过利用大量未标记数据进行预训练，展示了卓越的泛化能力，这对于无线通信模型适应各种场景尤其关键。本文创新性地将信道状态信息（CSI）和信道脉冲响应（CIR）视为自然对齐的多模态数据，并提出了首个MIMO无线信道基础模型，名为CSI-CLIP。通过有效地捕捉CIR和CSI的联合表示，CSI-CLIP在各种场景下表现出显著的适应能力和稳健的特征抽取能力。实验结果表明，在定位任务中，CSI-CLIP将均方误差降低了22%；在波束管理任务中，准确率提高了1%，超过了传统监督方法；在信道识别任务中也表现出色。这些改进不仅凸显了CSI-CLIP在整合感知与通信方面潜力和价值，还展示了其相对于现有技术的显著优势。此外，将CSI和CIR视为多模态配对并通过对比学习构建无线信道基础模型，为MIMO无线通信领域开拓了新的研究方向。 

---
# Navigating the Helpfulness-Truthfulness Trade-Off with Uncertainty-Aware Instruction Fine-Tuning 

**Title (ZH)**: 带有不确定性意识的指令微调以平衡帮助性和真实性trade-off 

**Authors**: Tianyi Wu, Jingwei Ni, Bryan Hooi, Jiaheng Zhang, Elliott Ash, See-Kiong Ng, Mrinmaya Sachan, Markus Leippold  

**Link**: [PDF](https://arxiv.org/pdf/2502.11962)  

**Abstract**: Instruction Fine-tuning (IFT) can enhance the helpfulness of Large Language Models (LLMs), but it may lower their truthfulness. This trade-off arises because IFT steers LLMs to generate responses with long-tail knowledge that is not well covered during pre-training, leading to more informative but less truthful answers when generalizing to unseen tasks. In this paper, we empirically demonstrate this helpfulness-truthfulness trade-off in IFT and propose $\textbf{UNIT}$, a novel IFT paradigm to address it. UNIT teaches LLMs to recognize their uncertainty and explicitly reflect it at the end of their responses. Experimental results show that UNIT-tuned models maintain their helpfulness while distinguishing between certain and uncertain claims, thereby reducing hallucinations. 

**Abstract (ZH)**: 指令微调（Instruction Fine-tuning，IFT）可以增强大型语言模型（LLMs）的帮助性，但可能会降低其真实性。这种权衡源于IFT使LLMs生成覆盖预训练阶段较少涉及的长尾知识的响应，导致在处理未见过的任务时提供更为信息性但更不真实的答案。在本文中，我们通过实验证明了IFT的帮助性与真实性之间的权衡，并提出了一种新的IFT方案——$\textbf{UNIT}$，以解决这一问题。UNIT 教学LLMs识别其自身的不确定性，并在响应结束时明确反映这种不确定性。实验结果表明，经过UNIT微调的模型在保持帮助性的同时，能够区分确定性和不确定性声明，从而减少臆断现象。 

---
# Massively Scaling Explicit Policy-conditioned Value Functions 

**Title (ZH)**: 大规模扩展显式策略条件价值函数 

**Authors**: Nico Bohlinger, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2502.11949)  

**Abstract**: We introduce a scaling strategy for Explicit Policy-Conditioned Value Functions (EPVFs) that significantly improves performance on challenging continuous-control tasks. EPVFs learn a value function V({\theta}) that is explicitly conditioned on the policy parameters, enabling direct gradient-based updates to the parameters of any policy. However, EPVFs at scale struggle with unrestricted parameter growth and efficient exploration in the policy parameter space. To address these issues, we utilize massive parallelization with GPU-based simulators, big batch sizes, weight clipping and scaled peturbations. Our results show that EPVFs can be scaled to solve complex tasks, such as a custom Ant environment, and can compete with state-of-the-art Deep Reinforcement Learning (DRL) baselines like Proximal Policy Optimization (PPO) and Soft Actor-Critic (SAC). We further explore action-based policy parameter representations from previous work and specialized neural network architectures to efficiently handle weight-space features, which have not been used in the context of DRL before. 

**Abstract (ZH)**: 我们提出了一种 Explicit Policy-Conditioned Value Functions (EPVF) 的扩展策略，该策略显著提升了在具有挑战性的连续控制任务中的性能。EPVF 学习一个显式地依赖于策略参数的价值函数 \(V(\theta)\)，这使得可以通过梯度更新策略参数。然而，在大规模应用中，EPVF 遇到了不受限制的参数增长以及在策略参数空间中有效的探索问题。为了解决这些问题，我们利用基于 GPU 的大规模并行化模拟、大批次大小、权重剪裁及放大扰动。实验结果表明，EPVF 可以扩展以解决复杂任务，如一个定制的蚂蚁环境，并且可以与最先进的深度强化学习（DRL）基线算法（如禁忌策略优化 PPO 和软作用者批评者 SAC）媲美。我们进一步探索了来自先前工作的基于动作的策略参数表示以及专门的神经网络架构，以高效处理权重空间特征，这是在 DRL 上未曾使用过的。 

---
# Step-Audio: Unified Understanding and Generation in Intelligent Speech Interaction 

**Title (ZH)**: 步态音频：智能语音交互中的统一理解和生成 

**Authors**: Ailin Huang, Boyong Wu, Bruce Wang, Chao Yan, Chen Hu, Chengli Feng, Fei Tian, Feiyu Shen, Jingbei Li, Mingrui Chen, Peng Liu, Ruihang Miao, Wang You, Xi Chen, Xuerui Yang, Yechang Huang, Yuxiang Zhang, Zheng Gong, Zixin Zhang, Brian Li, Changyi Wan, Hanpeng Hu, Ranchen Ming, Song Yuan, Xuelin Zhang, Yu Zhou, Bingxin Li, Buyun Ma, Kang An, Wei Ji, Wen Li, Xuan Wen, Yuankai Ma, Yuanwei Liang, Yun Mou, Bahtiyar Ahmidi, Bin Wang, Bo Li, Changxin Miao, Chen Xu, Chengting Feng, Chenrun Wang, Dapeng Shi, Deshan Sun, Dingyuan Hu, Dula Sai, Enle Liu, Guanzhe Huang, Gulin Yan, Heng Wang, Haonan Jia, Haoyang Zhang, Jiahao Gong, Jianchang Wu, Jiahong Liu, Jianjian Sun, Jiangjie Zhen, Jie Feng, Jie Wu, Jiaoren Wu, Jie Yang, Jinguo Wang, Jingyang Zhang, Junzhe Lin, Kaixiang Li, Lei Xia, Li Zhou, Longlong Gu, Mei Chen, Menglin Wu, Ming Li, Mingxiao Li, Mingyao Liang, Na Wang, Nie Hao, Qiling Wu, Qinyuan Tan, Shaoliang Pang, Shiliang Yang, Shuli Gao, Siqi Liu, Sitong Liu, Tiancheng Cao, Tianyu Wang, Wenjin Deng, Wenqing He, Wen Sun, Xin Han, Xiaomin Deng, Xiaojia Liu, Xu Zhao, Yanan Wei, Yanbo Yu, Yang Cao, Yangguang Li, Yangzhen Ma, Yanming Xu, Yaqiang Shi, Yilei Wang, Yinmin Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11946)  

**Abstract**: Real-time speech interaction, serving as a fundamental interface for human-machine collaboration, holds immense potential. However, current open-source models face limitations such as high costs in voice data collection, weakness in dynamic control, and limited intelligence. To address these challenges, this paper introduces Step-Audio, the first production-ready open-source solution. Key contributions include: 1) a 130B-parameter unified speech-text multi-modal model that achieves unified understanding and generation, with the Step-Audio-Chat version open-sourced; 2) a generative speech data engine that establishes an affordable voice cloning framework and produces the open-sourced lightweight Step-Audio-TTS-3B model through distillation; 3) an instruction-driven fine control system enabling dynamic adjustments across dialects, emotions, singing, and RAP; 4) an enhanced cognitive architecture augmented with tool calling and role-playing abilities to manage complex tasks effectively. Based on our new StepEval-Audio-360 evaluation benchmark, Step-Audio achieves state-of-the-art performance in human evaluations, especially in terms of instruction following. On open-source benchmarks like LLaMA Question, shows 9.3% average performance improvement, demonstrating our commitment to advancing the development of open-source multi-modal language technologies. Our code and models are available at this https URL. 

**Abstract (ZH)**: 实时语音交互作为人机协作的基本接口，具有巨大的潜力。然而，当前的开源模型在语音数据采集成本高、动态控制能力弱和智能程度有限等方面存在局限性。为了解决这些挑战，本文介绍了Step-Audio，这是首个生产级的开源解决方案。主要贡献包括：

1）一个包含130亿参数的统一语音-文本多模态模型，实现了统一的理解和生成能力，Step-Audio-Chat版本已开源；

2）一个生成性语音数据引擎，建立了经济实惠的语音克隆框架，并通过蒸馏生成了轻量级的Step-Audio-TTS-3B模型；

3）一个指令驱动的精细控制系统，能够在方言、情感、唱歌和嘻哈方面进行动态调整；

4）一个增强的认知架构，增加了工具调用和角色扮演能力，能够有效管理复杂任务。基于我们新的StepEval-Audio-360评估基准，Step-Audio在人工评估中取得了最先进的性能，特别是在指令跟随方面。在开源基准如LLaMA Question上，展示了9.3%的平均性能提升，显示了我们致力于推进开源多模态语言技术发展的承诺。我们的代码和模型已在此处（附上链接）提供。 

---
# Deep Spatio-Temporal Neural Network for Air Quality Reanalysis 

**Title (ZH)**: 深空时神经网络用于空气质量再分析 

**Authors**: Ammar Kheder, Benjamin Foreback, Lili Wang, Zhi-Song Liu, Michael Boy  

**Link**: [PDF](https://arxiv.org/pdf/2502.11941)  

**Abstract**: Air quality prediction is key to mitigating health impacts and guiding decisions, yet existing models tend to focus on temporal trends while overlooking spatial generalization. We propose AQ-Net, a spatiotemporal reanalysis model for both observed and unobserved stations in the near future. AQ-Net utilizes the LSTM and multi-head attention for the temporal regression. We also propose a cyclic encoding technique to ensure continuous time representation. To learn fine-grained spatial air quality estimation, we incorporate AQ-Net with the neural kNN to explore feature-based interpolation, such that we can fill the spatial gaps given coarse observation stations. To demonstrate the efficiency of our model for spatiotemporal reanalysis, we use data from 2013-2017 collected in northern China for PM2.5 analysis. Extensive experiments show that AQ-Net excels in air quality reanalysis, highlighting the potential of hybrid spatio-temporal models to better capture environmental dynamics, especially in urban areas where both spatial and temporal variability are critical. 

**Abstract (ZH)**: 空气质量预测对于减轻健康影响和指导决策至关重要，但现有模型通常侧重于时间趋势，而忽视了空间概括。本文提出了一种时空重新分析模型AQ-Net，用于近未来的实测和未观测站点。AQ-Net 使用 LSTM 和多头注意力机制进行时间回归。我们还提出了循环编码技术以确保连续的时间表示。为了学习细粒度的空间空气质量估计，我们结合了神经 kNN，以探索基于特征的插值方法，从而可以在给定粗略观测站点的情况下填充空间空白。为了证明我们的模型在时空重新分析中的效率，我们使用了 2013-2017 年中国北部收集的 PM2.5 数据。广泛的实验表明，AQ-Net 在时空重新分析方面表现出色，突显了混合时空模型在更好地捕捉环境动态方面的潜力，特别是在空间和时间变异性都至关重要的城市地区。 

---
# FitLight: Federated Imitation Learning for Plug-and-Play Autonomous Traffic Signal Control 

**Title (ZH)**: FitLight：插拔式自主交通信号控制的联邦模仿学习 

**Authors**: Yutong Ye, Yingbo Zhou, Zhusen Liu, Xiao Du, Hao Zhou, Xiang Lian, Mingsong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11937)  

**Abstract**: Although Reinforcement Learning (RL)-based Traffic Signal Control (TSC) methods have been extensively studied, their practical applications still raise some serious issues such as high learning cost and poor generalizability. This is because the ``trial-and-error'' training style makes RL agents extremely dependent on the specific traffic environment, which also requires a long convergence time. To address these issues, we propose a novel Federated Imitation Learning (FIL)-based framework for multi-intersection TSC, named FitLight, which allows RL agents to plug-and-play for any traffic environment without additional pre-training cost. Unlike existing imitation learning approaches that rely on pre-training RL agents with demonstrations, FitLight allows real-time imitation learning and seamless transition to reinforcement learning. Due to our proposed knowledge-sharing mechanism and novel hybrid pressure-based agent design, RL agents can quickly find a best control policy with only a few episodes. Moreover, for resource-constrained TSC scenarios, FitLight supports model pruning and heterogeneous model aggregation, such that RL agents can work on a micro-controller with merely 16{\it KB} RAM and 32{\it KB} ROM. Extensive experiments demonstrate that, compared to state-of-the-art methods, FitLight not only provides a superior starting point but also converges to a better final solution on both real-world and synthetic datasets, even under extreme resource limitations. 

**Abstract (ZH)**: 尽管基于强化学习（RL）的交通信号控制（TSC）方法已经得到了广泛的研究，但其实际应用仍存在一些严重的问题，如高昂的学习成本和较差的泛化能力。这是因为RL方法的“试错”训练方式使得RL代理高度依赖于特定的交通环境，并且需要较长的收敛时间。为了解决这些问题，我们提出了一种名为FitLight的新颖联邦模仿学习（FIL）框架，该框架允许RL代理在任何交通环境中“即插即用”而无需额外的预训练成本。与现有依赖于预训练RL代理样本的模仿学习方法不同，FitLight支持实时模仿学习和无缝过渡到强化学习。由于我们提出的知识共享机制以及新型的基于混合压力的代理设计，RL代理可以在少数几个回合内快速找到最优控制策略。此外，在资源受限的TSC场景中，FitLight支持模型剪枝和异构模型聚合，使得RL代理能够在仅有16 KB RAM和32 KB ROM的微控制器上运行。大量的实验结果表明，与最先进的方法相比，FitLight不仅提供了更好的初始点，而且即使在极端资源限制下也能收敛到更好的最终解决方案，甚至在真实世界和合成数据集上也是如此。 

---
# EssayJudge: A Multi-Granular Benchmark for Assessing Automated Essay Scoring Capabilities of Multimodal Large Language Models 

**Title (ZH)**: EssayJudge：评估多模态大规模语言模型自动作文评分能力的多粒度基准 

**Authors**: Jiamin Su, Yibo Yan, Fangteng Fu, Han Zhang, Jingheng Ye, Xiang Liu, Jiahao Huo, Huiyu Zhou, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11916)  

**Abstract**: Automated Essay Scoring (AES) plays a crucial role in educational assessment by providing scalable and consistent evaluations of writing tasks. However, traditional AES systems face three major challenges: (1) reliance on handcrafted features that limit generalizability, (2) difficulty in capturing fine-grained traits like coherence and argumentation, and (3) inability to handle multimodal contexts. In the era of Multimodal Large Language Models (MLLMs), we propose EssayJudge, the first multimodal benchmark to evaluate AES capabilities across lexical-, sentence-, and discourse-level traits. By leveraging MLLMs' strengths in trait-specific scoring and multimodal context understanding, EssayJudge aims to offer precise, context-rich evaluations without manual feature engineering, addressing longstanding AES limitations. Our experiments with 18 representative MLLMs reveal gaps in AES performance compared to human evaluation, particularly in discourse-level traits, highlighting the need for further advancements in MLLM-based AES research. Our dataset and code will be available upon acceptance. 

**Abstract (ZH)**: 自动化作文评分（AES）在教育评估中发挥着重要作用，它通过提供可扩展和一致的写作任务评估来支持教育评价。然而，传统的AES系统面临着三个主要挑战：（1）对手工构建特征的依赖限制了其普适性，（2）难以捕捉细微的特征，如连贯性和论证，以及（3）不能处理多模态情境。在多模态大型语言模型（MLLMs）的时代，我们提出了EssayJudge，这是首个能够评估AES在词级、句级和篇章级特征上的能力的多模态基准。通过利用MLLMs在特征特定评分和多模态上下文理解方面的优势，EssayJudge旨在提供精确的、富含上下文的评估，无需手动特征工程，从而解决长期存在的AES局限性。我们的实验涉及18个代表性MLLMs，揭示了AES在篇章级特征上的表现与人类评估之间存在差距，强调了基于MLLMs的AES研究的进一步改进需求。在论文被接受后，我们将提供我们的数据集和代码。 

---
# DLFR-VAE: Dynamic Latent Frame Rate VAE for Video Generation 

**Title (ZH)**: DLFR-VAE：动态潜在帧率变分自动编码器在视频生成中的应用 

**Authors**: Zhihang Yuan, Siyuan Wang, Rui Xie, Hanling Zhang, Tongcheng Fang, Yuzhang Shang, Shengen Yan, Guohao Dai, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11897)  

**Abstract**: In this paper, we propose the Dynamic Latent Frame Rate VAE (DLFR-VAE), a training-free paradigm that can make use of adaptive temporal compression in latent space. While existing video generative models apply fixed compression rates via pretrained VAE, we observe that real-world video content exhibits substantial temporal non-uniformity, with high-motion segments containing more information than static scenes. Based on this insight, DLFR-VAE dynamically adjusts the latent frame rate according to the content complexity. Specifically, DLFR-VAE comprises two core innovations: (1) A Dynamic Latent Frame Rate Scheduler that partitions videos into temporal chunks and adaptively determines optimal frame rates based on information-theoretic content complexity, and (2) A training-free adaptation mechanism that transforms pretrained VAE architectures into a dynamic VAE that can process features with variable frame rates. Our simple but effective DLFR-VAE can function as a plug-and-play module, seamlessly integrating with existing video generation models and accelerating the video generation process. 

**Abstract (ZH)**: 在本文中，我们提出了一种名为动态潜在帧率变分自编码器（Dynamic Latent Frame Rate VAE，简称DLFR-VAE）的无监督训练范式，该范式可以利用潜在空间中的自适应时间压缩。现有视频生成模型通过预训练的变分自编码器（VAE）采用固定的时间压缩率，而我们观察到真实世界的视频内容在时间维度上存在显著的非均匀性，即高动作片段包含比静止场景更多的信息。基于这一洞察，DLFR-VAE根据内容复杂性动态调整潜在帧率。具体而言，DLFR-VAE 包含两个核心创新:(1) 动态潜在帧率调度器，该调度器将视频划分为时间片段，并基于信息论的内容复杂度自适应地确定最优帧率；(2) 一种无监督适应机制，该机制将预训练的 VAE 架构转换为能够处理可变帧率特征的动态 VAE。我们的简单而有效的 DLFR-VAE 可作为即插即用模块使用，可以无缝集成到现有视频生成模型中，并加速视频生成过程。 

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
# Continual Quantization-Aware Pre-Training: When to transition from 16-bit to 1.58-bit pre-training for BitNet language models? 

**Title (ZH)**: 持续量化感知预训练：何时从16位转换到1.58位预训练以适用于BitNet语言模型？ 

**Authors**: Jacob Nielsen, Peter Schneider-Kamp, Lukas Galke  

**Link**: [PDF](https://arxiv.org/pdf/2502.11895)  

**Abstract**: Large language models (LLMs) require immense resources for training and inference. Quantization, a technique that reduces the precision of model parameters, offers a promising solution for improving LLM efficiency and sustainability. While post-training quantization methods typically achieve 4-8 bits per parameter, recent research suggests that training LLMs with 1.58 bits per weight parameter from scratch can maintain model accuracy while greatly reducing memory requirements and energy consumption at inference time. Here, we investigate a training strategy for quantization-aware pre-training, where the models are first trained with 16-bit precision and then transition into 1.58-bit quantization-aware training. Our results on 11 downstream tasks show that this 16-to-1.58-bit training strategy is preferable over full 1.58-bit training and leaves models closer to those which have undergone 16-bit training. We further investigate the effects of retaining the optimizer state at the transition point and gradually phasing in quantization strength -- finding that both techniques alleviate the magnitude of loss spikes, but also that these effects can be compensated through further training. 

**Abstract (ZH)**: 大型语言模型（LLMs）的训练和推理需要巨大的资源。量化技术通过降低模型参数的精度，为提高LLM效率和可持续性提供了有前景的解决方案。虽然后训练量化方法通常可以使每个参数达到4-8位，但最近的研究表明，从头开始使用1.58位的权重参数进行训练可以保持模型的准确性，同时大幅减少推理时的内存需求和能量消耗。在这项研究中，我们探讨了一种感知量化预训练的训练策略，即首先使用16位精度进行训练，然后过渡到1.58位感知量化的训练。我们在11个下游任务上的结果显示，这种从16位到1.58位的训练策略优于完整的1.58位训练，并使模型更接近于16位训练的模型。我们进一步探讨了在转换点保留优化器状态和逐步引入量化强度的影响——发现这两种技术都减轻了损失突增的幅度，但这些影响可以通过更多的训练来补偿。 

---
# Stonefish: Supporting Machine Learning Research in Marine Robotics 

**Title (ZH)**: 石鱼：支持海洋机器人领域机器学习研究 

**Authors**: Michele Grimaldi, Patryk Cieslak, Eduardo Ochoa, Vibhav Bharti, Hayat Rajani, Ignacio Carlucho, Maria Koskinopoulou, Yvan R. Petillot, Nuno Gracias  

**Link**: [PDF](https://arxiv.org/pdf/2502.11887)  

**Abstract**: Simulations are highly valuable in marine robotics, offering a cost-effective and controlled environment for testing in the challenging conditions of underwater and surface operations. Given the high costs and logistical difficulties of real-world trials, simulators capable of capturing the operational conditions of subsea environments have become key in developing and refining algorithms for remotely-operated and autonomous underwater vehicles. This paper highlights recent enhancements to the Stonefish simulator, an advanced open-source platform supporting development and testing of marine robotics solutions. Key updates include a suite of additional sensors, such as an event-based camera, a thermal camera, and an optical flow camera, as well as, visual light communication, support for tethered operations, improved thruster modelling, more flexible hydrodynamics, and enhanced sonar accuracy. These developments and an automated annotation tool significantly bolster Stonefish's role in marine robotics research, especially in the field of machine learning, where training data with a known ground truth is hard or impossible to collect. 

**Abstract (ZH)**: 海洋机器人领域的模拟在提供一个经济有效且可控的实验环境方面具有极高的价值，可以在水下和地表操作的挑战性环境中进行测试。由于实际试验证实成本高昂且物流复杂，能够捕捉到水下环境操作条件的模拟器已成为开发和不断完善遥控和自主水下车辆算法的关键工具。本文强调了对Stonefish模拟器的最新改进，这是一个先进的开源平台，支持海洋机器人解决方案的研发和测试。主要更新包括一系列附加传感器，如事件驱动相机、热像仪和光学流量相机，以及视觉光通信、支持有缆操作、改进的推进器建模、更灵活的水动力学以及增强的声纳精度。这些进展和自动标注工具极大地强化了Stonefish在海洋机器人研究中的作用，尤其是在机器学习领域，因为获取带有已知真实值的训练数据往往非常困难或不可能。 

---
# LIMR: Less is More for RL Scaling 

**Title (ZH)**: LIMR：少即是多，用于强化学习的扩展策略 

**Authors**: Xuefeng Li, Haoyang Zou, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11886)  

**Abstract**: In this paper, we ask: what truly determines the effectiveness of RL training data for enhancing language models' reasoning capabilities? While recent advances like o1, Deepseek R1, and Kimi1.5 demonstrate RL's potential, the lack of transparency about training data requirements has hindered systematic progress. Starting directly from base models without distillation, we challenge the assumption that scaling up RL training data inherently improves performance. we demonstrate that a strategically selected subset of just 1,389 samples can outperform the full 8,523-sample dataset. We introduce Learning Impact Measurement (LIM), an automated method to evaluate and prioritize training samples based on their alignment with model learning trajectories, enabling efficient resource utilization and scalable implementation. Our method achieves comparable or even superior performance using only 1,389 samples versus the full 8,523 samples dataset. Notably, while recent data-efficient approaches (e.g., LIMO and s1) show promise with 32B-scale models, we find it significantly underperforms at 7B-scale through supervised fine-tuning (SFT). In contrast, our RL-based LIMR achieves 16.7% higher accuracy on AIME24 and outperforms LIMO and s1 by 13.0% and 22.2% on MATH500. These results fundamentally reshape our understanding of RL scaling in LLMs, demonstrating that precise sample selection, rather than data scale, may be the key to unlocking enhanced reasoning capabilities. For reproducible research and future innovation, we are open-sourcing LIMR, including implementation of LIM, training and evaluation code, curated datasets, and trained models at this https URL. 

**Abstract (ZH)**: 在本文中，我们探讨了一个核心问题：真正决定强化学习（RL）训练数据对增强语言模型推理能力的有效性的是什么？尽管诸如o1、Deepseek R1和Kimi1.5等近期进展展示了RL的潜力，但关于训练数据需求的透明度不足，阻碍了系统的进步。我们直接从基础模型出发，未经过模型蒸馏，挑战了假设，即单纯扩大RL训练数据规模必然能提升性能。我们证明，精心选择的1,389个样本子集就能优于包含8,523个样本的完整数据集。我们介绍了一种名为学习影响度量（Learning Impact Measurement, LIM）的自动化方法，该方法基于样本与模型学习轨迹的对齐程度评估和优先排序训练样本，从而实现资源的高效利用和可扩展的实施。仅使用1,389个样本，我们的方法在性能上就能达到甚至超过使用完整8,523个样本数据集的结果。值得注意的是，尽管最近一些数据效率高的方法（如LIMO和s1）在32B规模的模型上显示出希望，但在7B规模的模型通过监督微调（SFT）时却显著表现不佳。相比之下，我们基于RL的LIMR方法在AIME24上的准确率提高了16.7%，并在MATH500上的表现分别比LIMO和s1高出13.0%和22.2%。这些结果从根本上重塑了我们对语言大模型（LLMs）中RL扩展的理解，表明精准的样本选择而非数据规模可能是解锁增强推理能力的关键。为促进可重复研究和未来创新，我们将开源LIMR，包括LIM的实现、训练和评估代码、精选数据集和预训练模型，链接见https://your-link-url。 

---
# Bitnet.cpp: Efficient Edge Inference for Ternary LLMs 

**Title (ZH)**: Bitnet.cpp：高效三元大型语言模型边缘推理 

**Authors**: Jinheng Wang, Hansong Zhou, Ting Song, Shijie Cao, Yan Xia, Ting Cao, Jianyu Wei, Shuming Ma, Hongyu Wang, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.11880)  

**Abstract**: The advent of 1-bit large language models (LLMs), led by BitNet b1.58, has spurred interest in ternary LLMs. Despite this, research and practical applications focusing on efficient edge inference for ternary LLMs remain scarce. To bridge this gap, we introduce this http URL, an inference system optimized for BitNet b1.58 and ternary LLMs. Given that mixed-precision matrix multiplication (mpGEMM) constitutes the bulk of inference time in ternary LLMs, this http URL incorporates a novel mpGEMM library to facilitate sub-2-bits-per-weight, efficient and lossless inference. The library features two core solutions: Ternary Lookup Table (TL), which addresses spatial inefficiencies of previous bit-wise methods, and Int2 with a Scale (I2_S), which ensures lossless edge inference, both enabling high-speed inference. Our experiments show that this http URL achieves up to a 6.25x increase in speed over full-precision baselines and up to 2.32x over low-bit baselines, setting new benchmarks in the field. Additionally, we expand TL to element-wise lookup table (ELUT) for low-bit LLMs in the appendix, presenting both theoretical and empirical evidence of its considerable potential. this http URL is publicly available at this https URL , offering a sophisticated solution for the efficient and practical deployment of edge LLMs. 

**Abstract (ZH)**: 1-bit大规模语言模型（LLMs）的出现，特别是由BitNet b1.58引领了这一潮流，激发了对三值LLMs的兴趣。然而，关于三值LLMs高效边缘推理的研究和实际应用仍然相对匮乏。为填补这一空白，本文介绍了这个URL（因原文中包含网址链接，此处用“这个URL”代替），一个针对BitNet b1.58和三值LLMs优化的推理系统。鉴于混合精度矩阵乘法（mpGEMM）占三值LLMs推理时间的大部分，这个URL引入了一个新的mpGEMM库，以实现每权重小于2比特、高效且无损的推理。该库包含两个核心解决方案：三值查找表（TL），解决以往位级方法的空间效率问题，以及带有比例因子的整数转换器（I2_S），确保边缘推理无损，两者共同实现了高速推理。实验结果显示，与全精度基准相比，这个URL的速度提高了6.25倍，与低比特基准相比提高了2.32倍，从而在该领域设立了新的标准。此外，我们在附录中将TL扩展到元素级查找表（ELUT），用于低比特LLMs，展示了其理论和实证证据，表明其潜力显著。这个URL在https:// 提供，作为边缘LLMs高效且实用部署的复杂解决方案。 

---
# FedEAT: A Robustness Optimization Framework for Federated LLMs 

**Title (ZH)**: FedEAT：联邦大规模语言模型的健壮性优化框架 

**Authors**: Yahao Pang, Xingyuan Wu, Xiaojin Zhang, Wei Chen, Hai Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.11863)  

**Abstract**: Significant advancements have been made by Large Language Models (LLMs) in the domains of natural language understanding and automated content creation. However, they still face persistent problems, including substantial computational costs and inadequate availability of training data. The combination of Federated Learning (FL) and LLMs (federated LLMs) offers a solution by leveraging distributed data while protecting privacy, which positions it as an ideal choice for sensitive domains. However, Federated LLMs still suffer from robustness challenges, including data heterogeneity, malicious clients, and adversarial attacks, which greatly hinder their applications. We first introduce the robustness problems in federated LLMs, to address these challenges, we propose FedEAT (Federated Embedding space Adversarial Training), a novel framework that applies adversarial training in the embedding space of client LLM and employs a robust aggregation approach, specifically geometric median aggregation, to enhance the robustness of Federated LLMs. Our experiments demonstrate that FedEAT effectively improves the robustness of Federated LLMs with minimal performance loss. 

**Abstract (ZH)**: 大语言模型（LLMs）在自然语言理解和自动化内容生成领域取得了显著进展。然而，它们仍然面临着持续存在的问题，包括巨大的计算成本和训练数据不足。通过结合联邦学习（FL）和LLMs（联邦LLMs），可以在利用分布式数据的同时保护隐私，这使它成为敏感领域理想的解决方案。然而，联邦LLMs仍然面临着鲁棒性挑战，包括数据异质性、恶意客户端和对抗攻击，这些挑战极大地阻碍了它们的应用。我们首先介绍了联邦LLMs中的鲁棒性问题，并为应对这些挑战，我们提出了一种名为FedEAT（联邦嵌入空间对抗训练）的新框架。该框架在客户端LLM的嵌入空间中应用对抗训练，并采用一种鲁棒聚合方法（特别是几何中位数聚合），以增强联邦LLMs的鲁棒性。我们的实验表明，FedEAT能够在不显著牺牲性能的情况下有效提高联邦LLMs的鲁棒性。 

---
# Steering the LoCoMotif: Using Domain Knowledge in Time Series Motif Discovery 

**Title (ZH)**: 引导LoCoMotif：在时间序列模式发现中运用领域知识 

**Authors**: Aras Yurtman, Daan Van Wesenbeeck, Wannes Meert, Hendrik Blockeel  

**Link**: [PDF](https://arxiv.org/pdf/2502.11850)  

**Abstract**: Time Series Motif Discovery (TSMD) identifies repeating patterns in time series data, but its unsupervised nature might result in motifs that are not interesting to the user. To address this, we propose a framework that allows the user to impose constraints on the motifs to be discovered, where constraints can easily be defined according to the properties of the desired motifs in the application domain. We also propose an efficient implementation of the framework, the LoCoMotif-DoK algorithm. We demonstrate that LoCoMotif-DoK can effectively leverage domain knowledge in real and synthetic data, outperforming other TSMD techniques which only support a limited form of domain knowledge. 

**Abstract (ZH)**: 时间序列模态发现（TSMD）可以识别时间序列数据中的重复模式，但其无监督的特性可能会导致发现的模态对用户不感兴趣。为解决这一问题，我们提出了一种框架，允许用户对要发现的模态施加约束，这些约束可以根据应用领域中所需模态的特性轻松定义。此外，我们还提出了一种高效实现该框架的算法——LoCoMotif-DoK算法。我们证明LoCoMotif-DoK能够有效地利用实际数据和合成数据中的领域知识，超出只支持有限形式领域知识的其他TSMD技术。 

---
# BaxBench: Can LLMs Generate Correct and Secure Backends? 

**Title (ZH)**: BaxBench：大型语言模型能否生成正确的且安全的后端代码？ 

**Authors**: Mark Vero, Niels Mündler, Victor Chibotaru, Veselin Raychev, Maximilian Baader, Nikola Jovanović, Jingxuan He, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2502.11844)  

**Abstract**: The automatic generation of programs has long been a fundamental challenge in computer science. Recent benchmarks have shown that large language models (LLMs) can effectively generate code at the function level, make code edits, and solve algorithmic coding tasks. However, to achieve full automation, LLMs should be able to generate production-quality, self-contained application modules. To evaluate the capabilities of LLMs in solving this challenge, we introduce BaxBench, a novel evaluation benchmark consisting of 392 tasks for the generation of backend applications. We focus on backends for three critical reasons: (i) they are practically relevant, building the core components of most modern web and cloud software, (ii) they are difficult to get right, requiring multiple functions and files to achieve the desired functionality, and (iii) they are security-critical, as they are exposed to untrusted third-parties, making secure solutions that prevent deployment-time attacks an imperative. BaxBench validates the functionality of the generated applications with comprehensive test cases, and assesses their security exposure by executing end-to-end exploits. Our experiments reveal key limitations of current LLMs in both functionality and security: (i) even the best model, OpenAI o1, achieves a mere 60% on code correctness; (ii) on average, we could successfully execute security exploits on more than half of the correct programs generated by each LLM; and (iii) in less popular backend frameworks, models further struggle to generate correct and secure applications. Progress on BaxBench signifies important steps towards autonomous and secure software development with LLMs. 

**Abstract (ZH)**: 程序的自动生成一直是计算机科学中的一个基本挑战。最近的基准测试显示，大型语言模型（LLMs）能够有效地在函数级别生成代码，进行代码编辑，并解决算法编程任务。然而，要实现完全自动化，LLMs 应该能够生成符合生产标准、自包含的应用模块。为了评估LLMs在解决这一挑战方面的能力，我们引入了BaxBench，这是一个新的评估基准，包含392个任务，用于生成后端应用。我们重点关注后端应用的三个原因：（i）它们实践相关，是大多数现代网络和云计算软件的核心组成部分；（ii）它们难以做到完美，需要多个函数和文件才能实现所需的功能；（iii）它们是安全性关键的，因为它们对外暴露给不可信的第三方，使防止部署时攻击的安全解决方案变得至关重要。BaxBench通过全面的测试案例验证生成应用程序的功能，并通过端到端的攻击执行来评估它们的安全风险。我们的实验揭示了当前LLMs在功能和安全方面的主要局限性：（i）即使是最好的模型，OpenAI O1，在代码正确性方面仅能达到60%；（ii）平均而言，我们能够在超过一半由每个LLM生成的正确程序上成功执行安全性攻击；（iii）在不太常用的后端框架中，模型进一步难以生成正确且安全的应用程序。BaxBench上的进展标志着使用LLMs实现自主和安全软件开发的重要步骤。 

---
# Can LLM Agents Maintain a Persona in Discourse? 

**Title (ZH)**: LLM代理能否在对话中维持人设？ 

**Authors**: Pranav Bhandari, Nicolas Fay, Michael Wise, Amitava Datta, Stephanie Meek, Usman Naseem, Mehwish Nasim  

**Link**: [PDF](https://arxiv.org/pdf/2502.11843)  

**Abstract**: Large Language Models (LLMs) are widely used as conversational agents, exploiting their capabilities in various sectors such as education, law, medicine, and more. However, LLMs are often subjected to context-shifting behaviour, resulting in a lack of consistent and interpretable personality-aligned interactions. Adherence to psychological traits lacks comprehensive analysis, especially in the case of dyadic (pairwise) conversations. We examine this challenge from two viewpoints, initially using two conversation agents to generate a discourse on a certain topic with an assigned personality from the OCEAN framework (Openness, Conscientiousness, Extraversion, Agreeableness, and Neuroticism) as High/Low for each trait. This is followed by using multiple judge agents to infer the original traits assigned to explore prediction consistency, inter-model agreement, and alignment with the assigned personality. Our findings indicate that while LLMs can be guided toward personality-driven dialogue, their ability to maintain personality traits varies significantly depending on the combination of models and discourse settings. These inconsistencies emphasise the challenges in achieving stable and interpretable personality-aligned interactions in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）广泛用于作为对话代理，在教育、法律、医学等多个领域利用其各种能力。然而，LLMs经常表现出上下文转换行为，导致缺乏一致性和可解释性的个性匹配互动。在心理特质的遵循方面缺乏全面分析，尤其是在双向（成对）对话中更为明显。我们从两个视角探讨这一挑战：首先，使用两个对话代理生成特定话题的对话，并分配来自OCEAN框架（开放性、尽责性、外向性、和悦性、神经质）的高/低个性特质。随后，使用多个评判代理来推断原本分配的特质，以探索预测一致性、模型间一致性以及与分配个性的对齐情况。我们的研究表明，尽管LLMs可以被引导进行基于个性的对话，但它们保持个性特质的能力在不同模型组合和对话设置下差异显著。这些不一致性突显了在LLMs中实现稳定且可解释的个性匹配互动的挑战。 

---
# ChordFormer: A Conformer-Based Architecture for Large-Vocabulary Audio Chord Recognition 

**Title (ZH)**: ChordFormer：基于Conformer的大型词汇量音频和弦识别架构 

**Authors**: Muhammad Waseem Akram, Stefano Dettori, Valentina Colla, Giorgio Carlo Buttazzo  

**Link**: [PDF](https://arxiv.org/pdf/2502.11840)  

**Abstract**: Chord recognition serves as a critical task in music information retrieval due to the abstract and descriptive nature of chords in music analysis. While audio chord recognition systems have achieved significant accuracy for small vocabularies (e.g., major/minor chords), large-vocabulary chord recognition remains a challenging problem. This complexity also arises from the inherent long-tail distribution of chords, where rare chord types are underrepresented in most datasets, leading to insufficient training samples. Effective chord recognition requires leveraging contextual information from audio sequences, yet existing models, such as combinations of convolutional neural networks, bidirectional long short-term memory networks, and bidirectional transformers, face limitations in capturing long-term dependencies and exhibit suboptimal performance on large-vocabulary chord recognition tasks. This work proposes ChordFormer, a novel conformer-based architecture designed to tackle structural chord recognition (e.g., triads, bass, sevenths) for large vocabularies. ChordFormer leverages conformer blocks that integrate convolutional neural networks with transformers, thus enabling the model to capture both local patterns and global dependencies effectively. By addressing challenges such as class imbalance through a reweighted loss function and structured chord representations, ChordFormer outperforms state-of-the-art models, achieving a 2% improvement in frame-wise accuracy and a 6% increase in class-wise accuracy on large-vocabulary chord datasets. Furthermore, ChordFormer excels in handling class imbalance, providing robust and balanced recognition across chord types. This approach bridges the gap between theoretical music knowledge and practical applications, advancing the field of large-vocabulary chord recognition. 

**Abstract (ZH)**: 和弦识别是音乐信息检索中的一个关键任务，因为和弦在音乐分析中的抽象性和描述性。虽然基于音频的和弦识别系统在小词汇量（如大调/小调和弦）的情况下已经实现了显著的准确性，但在大词汇量和弦识别方面仍然存在挑战。这种复杂性也源于和弦的固有长尾分布，其中稀有和弦类型在大多数数据集中代表性不足，导致训练样本不足。有效的和弦识别需要利用音频序列中的上下文信息，但现有模型，如卷积神经网络、双向长短时记忆网络和双向变压器的组合，在捕捉长期依赖性方面存在局限性，并在大词汇量和弦识别任务上表现不佳。本文提出了一种新颖的Conformer基于架构ChordFormer，旨在解决大规模词汇量下结构和弦识别问题（例如三和弦、低音和七和弦）。ChordFormer利用结合了卷积神经网络和变压器的Conformer块，从而使模型能够有效捕捉局部模式和全局依赖性。通过使用重新加权的损失函数和结构化的和弦表示来解决类别不平衡等问题，ChordFormer在大词汇量和弦数据集上优于现有最先进的模型，分别在帧级准确性上提高了2%和类别级准确性提高了6%。此外，ChordFormer在处理类别不平衡方面表现出色，能够提供跨和弦类型的稳健且平衡的识别。该方法弥合了理论音乐知识与实际应用之间的差距，推动了大词汇量和弦识别领域的进步。 

---
# Intuitive physics understanding emerges from self-supervised pretraining on natural videos 

**Title (ZH)**: 直观的物理理解来源于自然视频的自监督预训练 

**Authors**: Quentin Garrido, Nicolas Ballas, Mahmoud Assran, Adrien Bardes, Laurent Najman, Michael Rabbat, Emmanuel Dupoux, Yann LeCun  

**Link**: [PDF](https://arxiv.org/pdf/2502.11831)  

**Abstract**: We investigate the emergence of intuitive physics understanding in general-purpose deep neural network models trained to predict masked regions in natural videos. Leveraging the violation-of-expectation framework, we find that video prediction models trained to predict outcomes in a learned representation space demonstrate an understanding of various intuitive physics properties, such as object permanence and shape consistency. In contrast, video prediction in pixel space and multimodal large language models, which reason through text, achieve performance closer to chance. Our comparisons of these architectures reveal that jointly learning an abstract representation space while predicting missing parts of sensory input, akin to predictive coding, is sufficient to acquire an understanding of intuitive physics, and that even models trained on one week of unique video achieve above chance performance. This challenges the idea that core knowledge -- a set of innate systems to help understand the world -- needs to be hardwired to develop an understanding of intuitive physics. 

**Abstract (ZH)**: 我们研究了通用深度神经网络模型在预测自然视频中遮蔽区域时对直观物理原理的理解能力。利用违反预期框架，我们发现，在学习表示空间中训练的视频预测模型能够理解各种直观物理属性，如物体恒存性和形状一致性。相比之下，在像素空间中进行视频预测以及通过文本进行推理的多模态大型语言模型的表现接近随机水平。这些架构的比较表明，同时学习一个抽象的表示空间并预测感官输入的缺失部分，类似于预测编码，是获得直观物理理解的足够条件，即使是在仅基于一周的独特视频数据训练的模型中也能够超越随机表现。这挑战了核心知识——一组帮助理解世界的基本系统——必须固有编码才能发展出直观物理理解的说法。 

---
# Code-Vision: Evaluating Multimodal LLMs Logic Understanding and Code Generation Capabilities 

**Title (ZH)**: Code-Vision：评估多模态LLM的逻辑理解与代码生成能力 

**Authors**: Hanbin Wang, Xiaoxuan Zhou, Zhipeng Xu, Keyuan Cheng, Yuxin Zuo, Kai Tian, Jingwei Song, Junting Lu, Wenhui Hu, Xueyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11829)  

**Abstract**: This paper introduces Code-Vision, a benchmark designed to evaluate the logical understanding and code generation capabilities of Multimodal Large Language Models (MLLMs). It challenges MLLMs to generate a correct program that fulfills specific functionality requirements based on a given flowchart, which visually represents the desired algorithm or process. Code-Vision comprises three subsets: HumanEval-V, Algorithm, and MATH, which evaluate MLLMs' coding abilities across basic programming, algorithmic, and mathematical problem-solving domains. Our experiments evaluate 12 MLLMs on Code-Vision. Experimental results demonstrate that there is a large performance difference between proprietary and open-source models. On Hard problems, GPT-4o can achieve 79.3% pass@1, but the best open-source model only achieves 15%. Further experiments reveal that Code-Vision can pose unique challenges compared to other multimodal reasoning benchmarks MMCode and MathVista. We also explore the reason for the poor performance of the open-source models. All data and codes are available at this https URL. 

**Abstract (ZH)**: 本文介绍了Code-Vision，这是一个用于评估多模态大型语言模型（MLLMs）的逻辑理解和代码生成能力的基准。该基准要求MLLMs根据给定的流程图生成一个符合特定功能要求的正确程序，而流程图则以视觉方式表示所需的算法或过程。Code-Vision涵盖了三个子集：HumanEval-V、Algorithm和MATH，分别评估MLLMs在基本编程、算法和数学问题解决领域的编码能力。我们的实验对12种MLLMs在Code-Vision上的表现进行了评估。实验结果表明，专有模型和开源模型之间存在显著的性能差异。在难度较高的问题上，GPT-4o可以达到79.3%的pass@1通过率，而最好的开源模型仅达到15%。进一步的实验表明，与MMCode和MathVista等其他多模态推理基准相比，Code-Vision能够提出独特的挑战。我们还探讨了开源模型表现不佳的原因。所有数据和代码可在以下网址获取：[这里提供网址]。 

---
# Towards Understanding Fine-Tuning Mechanisms of LLMs via Circuit Analysis 

**Title (ZH)**: 通过电路分析理解大规模语言模型微调机制 

**Authors**: Xu Wang, Yan Hu, Wenyu Du, Reynold Cheng, Benyou Wang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2502.11812)  

**Abstract**: Fine-tuning significantly improves the performance of Large Language Models (LLMs), yet its underlying mechanisms remain poorly understood. This paper aims to provide an in-depth interpretation of the fine-tuning process through circuit analysis, a popular tool in Mechanistic Interpretability (MI). Unlike previous studies \cite{prakash2024finetuningenhancesexistingmechanisms,chhabra2024neuroplasticity} that focus on tasks where pre-trained models already perform well, we develop a set of mathematical tasks where fine-tuning yields substantial performance gains, which are closer to the practical setting. In our experiments, we identify circuits at various checkpoints during fine-tuning and examine the interplay between circuit analysis, fine-tuning methods, and task complexities. First, we find that while circuits maintain high node similarity before and after fine-tuning, their edges undergo significant changes, which is in contrast to the previous work \cite{prakash2024finetuningenhancesexistingmechanisms,chhabra2024neuroplasticity} that show circuits only add some additional components after fine-tuning. Based on these observations, we develop a circuit-aware Low-Rank Adaptation (LoRA) method, which assigns ranks to layers based on edge changes in the circuits. Experimental results demonstrate that our circuit-based LoRA algorithm achieves an average performance improvement of 2.46\% over standard LoRA with similar parameter sizes. Furthermore, we explore how combining circuits from subtasks can enhance fine-tuning in compositional tasks, providing new insights into the design of such tasks and deepening the understanding of circuit dynamics and fine-tuning mechanisms. 

**Abstract (ZH)**: 预训练大型语言模型（LLMs）的微调显著提高了其性能，但其背后的机制仍不甚明了。本文旨在通过电路分析这一机制可解释性领域的常用工具，对微调过程进行深入解析。有别于以往研究[1, 2]，这些研究主要关注预训练模型在特定任务上表现已极其出色的情况，我们的研究开发了一系列数学任务，其中微调带来了显著的性能提升，更接近于实际应用场景。在实验中，我们识别了微调过程中各个检查点的电路，并探讨了电路分析、微调策略和任务复杂性之间的相互作用。首先，我们发现，在微调前后，电路中的节点保持较高的相似性，但其边存在显著变化，这与以往研究[1, 2]的结果不符，后者表明微调只会增加一些额外的组件。基于这些观察结果，我们提出了一种电路感知的低秩适应（LoRA）方法，该方法根据电路中边的变化为各层分配了排名。实验结果表明，我们提出的基于电路的LoRA算法在保持相似参数量的情况下，平均性能提高了2.46%。此外，我们还研究了如何通过组合子任务电路来增强组合任务的微调效果，为这类任务的设计提供了新的见解，并加深了对电路动态和微调机制的理解。

注释：
[1] Prakash, B., Zhang, Y., & Talwalkar, A. (2024). Finetuning enhances existing mechanisms. 
[2] Chhabra, V., & Bengio, Y. (2024). Neuroplasticity. 

---
# Revealing Bias Formation in Deep Neural Networks Through the Geometric Mechanisms of Human Visual Decoupling 

**Title (ZH)**: 通过人类视觉解耦的几何机制揭示深度神经网络中的偏差形成 

**Authors**: Yanbiao Ma, Bowei Liu, Wei Dai, Jiayi Chen, Shuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.11809)  

**Abstract**: Deep neural networks (DNNs) often exhibit biases toward certain categories during object recognition, even under balanced training data conditions. The intrinsic mechanisms underlying these biases remain unclear. Inspired by the human visual system, which decouples object manifolds through hierarchical processing to achieve object recognition, we propose a geometric analysis framework linking the geometric complexity of class-specific perceptual manifolds in DNNs to model bias. Our findings reveal that differences in geometric complexity can lead to varying recognition capabilities across categories, introducing biases. To support this analysis, we present the Perceptual-Manifold-Geometry library, designed for calculating the geometric properties of perceptual manifolds. 

**Abstract (ZH)**: 深度神经网络（DNNs）在物体识别过程中往往会对某些类别表现出偏见，即使在平衡训练数据条件下也是如此。这些偏见的内在机制尚不清楚。受人类视觉系统通过分层处理分离物体结构以实现物体识别的启发，我们提出了一种几何分析框架，将DNN中类别特定感知流形的几何复杂性与其模型偏见联系起来。我们的研究发现，几何复杂性的差异会导致各类别间不同的识别能力，从而引入偏见。为了支持这一分析，我们引入了Perceptual-Manifold-Geometry库，用于计算感知流形的几何属性。 

---
# Deep Neural Networks for Accurate Depth Estimation with Latent Space Features 

**Title (ZH)**: 基于潜在空间特征的深度神经网络深度估计方法 

**Authors**: Siddiqui Muhammad Yasir, Hyunsik Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2502.11777)  

**Abstract**: Depth estimation plays a pivotal role in advancing human-robot interactions, especially in indoor environments where accurate 3D scene reconstruction is essential for tasks like navigation and object handling. Monocular depth estimation, which relies on a single RGB camera, offers a more affordable solution compared to traditional methods that use stereo cameras or LiDAR. However, despite recent progress, many monocular approaches struggle with accurately defining depth boundaries, leading to less precise reconstructions. In response to these challenges, this study introduces a novel depth estimation framework that leverages latent space features within a deep convolutional neural network to enhance the precision of monocular depth maps. The proposed model features dual encoder-decoder architecture, enabling both color-to-depth and depth-to-depth transformations. This structure allows for refined depth estimation through latent space encoding. To further improve the accuracy of depth boundaries and local features, a new loss function is introduced. This function combines latent loss with gradient loss, helping the model maintain the integrity of depth boundaries. The framework is thoroughly tested using the NYU Depth V2 dataset, where it sets a new benchmark, particularly excelling in complex indoor scenarios. The results clearly show that this approach effectively reduces depth ambiguities and blurring, making it a promising solution for applications in human-robot interaction and 3D scene reconstruction. 

**Abstract (ZH)**: 深度估计在促进人机交互中发挥着至关重要的作用，特别是在室内环境中，准确的三维场景重建对于导航和物体处理等任务至关重要。单目深度估计依赖于单一的RGB摄像头，相较于传统的使用立体摄像机或激光雷达（LiDAR）的方法，提供了更加经济的解决方案。然而，尽管取得了最近的进展，许多单目方法仍然难以准确地定义深度边界，导致重建不够精确。为应对这些挑战，本研究提出了一种新的深度估计框架，该框架利用深度卷积神经网络中的潜在空间特征来提升单目深度图的精度。所提出的模型采用双重编码器-解码器架构，能够实现从颜色到深度和从深度到深度的转换。这种结构通过潜在空间编码实现细化的深度估计。为进一步提高深度边界和局部特征的准确性，引入了一种新的损失函数。该函数结合了潜在损失和梯度损失，帮助模型保持深度边界的完整性。该框架在NYU Depth V2数据集上得到了全面测试，特别是在复杂室内场景中表现出色。结果显示，这种方法有效地减少了深度不确定性并防止了模糊现象，使其成为人机交互和三维场景重建应用中的一个有前途的解决方案。 

---
# The Validation Gap: A Mechanistic Analysis of How Language Models Compute Arithmetic but Fail to Validate It 

**Title (ZH)**: 验证差距：语言模型在进行算术计算时验证不足的机理解析 

**Authors**: Leonardo Bertolazzi, Philipp Mondorf, Barbara Plank, Raffaella Bernardi  

**Link**: [PDF](https://arxiv.org/pdf/2502.11771)  

**Abstract**: The ability of large language models (LLMs) to validate their output and identify potential errors is crucial for ensuring robustness and reliability. However, current research indicates that LLMs struggle with self-correction, encountering significant challenges in detecting errors. While studies have explored methods to enhance self-correction in LLMs, relatively little attention has been given to understanding the models' internal mechanisms underlying error detection. In this paper, we present a mechanistic analysis of error detection in LLMs, focusing on simple arithmetic problems. Through circuit analysis, we identify the computational subgraphs responsible for detecting arithmetic errors across four smaller-sized LLMs. Our findings reveal that all models heavily rely on $\textit{consistency heads}$--attention heads that assess surface-level alignment of numerical values in arithmetic solutions. Moreover, we observe that the models' internal arithmetic computation primarily occurs in higher layers, whereas validation takes place in middle layers, before the final arithmetic results are fully encoded. This structural dissociation between arithmetic computation and validation seems to explain why current LLMs struggle to detect even simple arithmetic errors. 

**Abstract (ZH)**: 大型语言模型（LLMs）验证其输出并识别潜在错误的能力对于确保其稳健性和可靠性至关重要。然而，当前的研究表明，LLMs在自我纠正方面存在困难，面临着显著的错误检测挑战。虽然已有研究探讨了如何增强LLMs的自我纠正能力，但对模型内部负责错误检测的机制理解相对较少。本文中，我们通过机制分析研究了LLMs在简单算术问题中的错误检测能力。通过电路分析，我们确定了四款较小规模的LLMs中负责检测算术错误的计算子图。研究结果表明，所有模型都高度依赖于所谓的“一致性头”——这些注意头评估算术解决方案中数值的表面级对齐情况。此外，我们观察到模型的内部算术计算主要发生在较高层，而在中间层进行验证，之后才完成最终的算术结果编码。这种算术计算和验证之间的结构分离似乎解释了当前LLMs难以检测甚至简单的算术错误的原因。 

---
# Lightweight Deepfake Detection Based on Multi-Feature Fusion 

**Title (ZH)**: 基于多特征融合的 Lightweight 伪造视频检测 

**Authors**: Siddiqui Muhammad Yasir, Hyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.11763)  

**Abstract**: Deepfake technology utilizes deep learning based face manipulation techniques to seamlessly replace faces in videos creating highly realistic but artificially generated content. Although this technology has beneficial applications in media and entertainment misuse of its capabilities may lead to serious risks including identity theft cyberbullying and false information. The integration of DL with visual cognition has resulted in important technological improvements particularly in addressing privacy risks caused by artificially generated deepfake images on digital media platforms. In this study we propose an efficient and lightweight method for detecting deepfake images and videos making it suitable for devices with limited computational resources. In order to reduce the computational burden usually associated with DL models our method integrates machine learning classifiers in combination with keyframing approaches and texture analysis. Moreover the features extracted with a histogram of oriented gradients (HOG) local binary pattern (LBP) and KAZE bands were integrated to evaluate using random forest extreme gradient boosting extra trees and support vector classifier algorithms. Our findings show a feature-level fusion of HOG LBP and KAZE features improves accuracy to 92% and 96% on FaceForensics++ and Celeb-DFv2 respectively. 

**Abstract (ZH)**: 深度合成技术利用基于深度学习的面部操纵技术在视频中无缝替换面部，生成高度真实但人为生成的内容。虽然这项技术在媒体和娱乐领域具有有益的应用，但其功能的滥用可能导致身份盗用、网络欺凌和虚假信息等严重风险。将深度学习与视觉认知的结合已经带来了重要的技术进步，特别是在处理数字媒体平台中由人工生成的深度合成图像引起的隐私风险方面。在本研究中，我们提出了一种高效且轻量的方法来检测深度合成图像和视频，使其适用于有限计算资源的设备。为了减少通常与深度学习模型相关的计算负担，我们的方法结合了机器学习分类器和关键帧方法及纹理分析。此外，通过直方图梯度（HOG）、局部二值模式（LBP）和KAZE特征提取，并使用随机森林、极端梯度提升、超随机森林和支持向量分类器算法进行评估。我们的研究结果表明，HOG、LBP和KAZE特征的特征级融合分别在FaceForensics++和Celeb-DFv2数据集上提高了准确性至92%和96%。 

---
# On the Computation of the Fisher Information in Continual Learning 

**Title (ZH)**: 连续学习中 Fisher 信息的计算研究 

**Authors**: Gido M. van de Ven  

**Link**: [PDF](https://arxiv.org/pdf/2502.11756)  

**Abstract**: One of the most popular methods for continual learning with deep neural networks is Elastic Weight Consolidation (EWC), which involves computing the Fisher Information. The exact way in which the Fisher Information is computed is however rarely described, and multiple different implementations for it can be found online. This blog post discusses and empirically compares several often-used implementations, which highlights that many currently reported results for EWC could likely be improved by changing the way the Fisher Information is computed. 

**Abstract (ZH)**: 在深度神经网络的持续学习中，最流行的方法之一是弹性权重巩固（EWC），该方法涉及计算福尔希信息（Fisher信息）。然而，福尔希信息的确切计算方法在文献中的描述并不常见，网上可以找到多种不同的实现方式。本文讨论并实证比较了几种常用的实现方式，指出许多目前报告的EWC结果可能通过改变福尔希信息的计算方法而得到改善。 

---
# Language Models Can See Better: Visual Contrastive Decoding For LLM Multimodal Reasoning 

**Title (ZH)**: 语言模型视觉能力提升：用于大模型多模态推理的视觉对比解码 

**Authors**: Yuqi Pang, Bowen Yang, Haoqin Tu, Yun Cao, Zeyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11751)  

**Abstract**: Although Large Language Models (LLMs) excel in reasoning and generation for language tasks, they are not specifically designed for multimodal challenges. Training Multimodal Large Language Models (MLLMs), however, is resource-intensive and constrained by various training limitations. In this paper, we propose the Modular-based Visual Contrastive Decoding (MVCD) framework to move this obstacle. Our framework leverages LLMs' In-Context Learning (ICL) capability and the proposed visual contrastive-example decoding (CED), specifically tailored for this framework, without requiring any additional training. By converting visual signals into text and focusing on contrastive output distributions during decoding, we can highlight the new information introduced by contextual examples, explore their connections, and avoid over-reliance on prior encoded knowledge. MVCD enhances LLMs' visual perception to make it see and reason over the input visuals. To demonstrate MVCD's effectiveness, we conduct experiments with four LLMs across five question answering datasets. Our results not only show consistent improvement in model accuracy but well explain the effective components inside our decoding strategy. Our code will be available at this https URL. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在语言任务中的推理和生成表现优异，但它们并没有专门为多模态挑战而设计。然而，训练多模态大型语言模型（MLLMs）需要大量资源，并受到各种训练限制。在本文中，我们提出了基于模块化视觉对比解码（MVCD）框架来克服这一障碍。我们的框架利用了LLMs的上下文学习（ICL）能力，并结合了专门为该框架设计的视觉对比解码（CED），而不需要额外的训练。通过将视觉信号转换为文本并在解码过程中关注对比输出分布，我们可以突出上下文示例引入的新信息，探索它们之间的关联，从而避免过度依赖先验编码知识。MVCD增强了LLMs的视觉感知能力，使其能够对输入视觉进行观察和推理。为了展示MVCD的有效性，我们在五个问答数据集中对四种LLMs进行了实验。我们的结果不仅展示了模型准确性的持续改进，还详细解释了解码策略中的有效组件。相关代码将在此网址中提供：this https URL。 

---
# JotlasNet: Joint Tensor Low-Rank and Attention-based Sparse Unrolling Network for Accelerating Dynamic MRI 

**Title (ZH)**: JotlasNet：联合张量低秩表示与注意力机制稀疏展开的加速动态MRI网络 

**Authors**: Yinghao Zhang, Haiyan Gui, Ningdi Yang, Yue Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11749)  

**Abstract**: Joint low-rank and sparse unrolling networks have shown superior performance in dynamic MRI reconstruction. However, existing works mainly utilized matrix low-rank priors, neglecting the tensor characteristics of dynamic MRI images, and only a global threshold is applied for the sparse constraint to the multi-channel data, limiting the flexibility of the network. Additionally, most of them have inherently complex network structure, with intricate interactions among variables. In this paper, we propose a novel deep unrolling network, JotlasNet, for dynamic MRI reconstruction by jointly utilizing tensor low-rank and attention-based sparse priors. Specifically, we utilize tensor low-rank prior to exploit the structural correlations in high-dimensional data. Convolutional neural networks are used to adaptively learn the low-rank and sparse transform domains. A novel attention-based soft thresholding operator is proposed to assign a unique learnable threshold to each channel of the data in the CNN-learned sparse domain. The network is unrolled from the elaborately designed composite splitting algorithm and thus features a simple yet efficient parallel structure. Extensive experiments on two datasets (OCMR, CMRxRecon) demonstrate the superior performance of JotlasNet in dynamic MRI reconstruction. 

**Abstract (ZH)**: 在动态MRI重建中，联合低秩和稀疏展平网络已显示出优越的表现。然而，现有的研究主要利用了矩阵低秩先验，忽视了动态MRI图像的张量特性，并且仅对多通道数据应用全局阈值约束稀疏性，这限制了网络的灵活性。此外，大多数网络结构本身就非常复杂，变量间的交互也极为复杂。本文中，我们提出了一种新的深度展平网络JotlasNet，通过联合利用张量低秩和基于注意力的稀疏先验来进行动态MRI重建。具体而言，我们利用张量低秩先验来发掘高维数据中的结构相关性。卷积神经网络被用来自适应地学习低秩和稀疏变换域。我们提出了一个基于注意力的软阈值运算符，能够在卷积神经网络学习到的稀疏域中为数据的每个通道分配一个可学习的阈值。该网络是从精心设计的复合分裂算法展平而来，因此具有简单而高效的并行结构。在OCMR和CMRxRecon两个数据集上的广泛实验表明，JotlasNet在动态MRI重建中的性能优越。 

---
# SQL-o1: A Self-Reward Heuristic Dynamic Search Method for Text-to-SQL 

**Title (ZH)**: SQL-o1：一种自我奖励启发式动态搜索方法用于文本到SQL 

**Authors**: Shuai Lyu, Haoran Luo, Zhonghong Ou, Yifan Zhu, Xiaoran Shang, Yang Qin, Meina Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.11741)  

**Abstract**: The Text-to-SQL(Text2SQL) task aims to convert natural language queries into executable SQL queries. Thanks to the application of large language models (LLMs), significant progress has been made in this field. However, challenges such as model scalability, limited generation space, and coherence issues in SQL generation still persist. To address these issues, we propose SQL-o1, a Self-Reward-based heuristic search method designed to enhance the reasoning ability of LLMs in SQL query generation. SQL-o1 combines Monte Carlo Tree Search (MCTS) for heuristic process-level search and constructs a Schema-Aware dataset to help the model better understand database schemas. Extensive experiments on the Bird and Spider datasets demonstrate that SQL-o1 improves execution accuracy by 10.8\% on the complex Bird dataset compared to the latest baseline methods, even outperforming GPT-4-based approaches. Additionally, SQL-o1 excels in few-shot learning scenarios and shows strong cross-model transferability. Our code is publicly available at:this https URL. 

**Abstract (ZH)**: 文本到SQL（Text2SQL）任务旨在将自然语言查询转换为可执行的SQL查询。得益于大规模语言模型（LLMs）的应用，该领域取得了显著进展。然而，在SQL生成过程中，模型可扩展性、生成空间有限以及语义连贯性问题仍然存在。为了解决这些问题，我们提出了一种名为SQL-o1的基于自我奖励的启发式搜索方法，旨在增强LLMs在SQL查询生成中的推理能力。SQL-o1结合了蒙特卡洛树搜索（MCTS）作为启发式过程级搜索，并构建了一个模式感知的数据集，以帮助模型更好地理解数据库模式。在Bird和Spider数据集上的广泛实验显示，与最新的基线方法相比，SQL-o1在复杂数据集Bird上的执行准确率提高了10.8%，甚至优于基于GPT-4的方法。此外，SQL-o1在少样本学习场景中表现出色，并显示出了较强跨模型的迁移能力。我们的代码已在此处公开：[此链接]。 

---
# ReviewEval: An Evaluation Framework for AI-Generated Reviews 

**Title (ZH)**: ReviewEval：一种AI生成评论的评估框架 

**Authors**: Chavvi Kirtani, Madhav Krishan Garg, Tejash Prasad, Tanmay Singhal, Murari Mandal, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.11736)  

**Abstract**: The escalating volume of academic research, coupled with a shortage of qualified reviewers, necessitates innovative approaches to peer review. While large language model (LLMs) offer potential for automating this process, their current limitations include superficial critiques, hallucinations, and a lack of actionable insights. This research addresses these challenges by introducing a comprehensive evaluation framework for AI-generated reviews, that measures alignment with human evaluations, verifies factual accuracy, assesses analytical depth, and identifies actionable insights. We also propose a novel alignment mechanism that tailors LLM-generated reviews to the unique evaluation priorities of individual conferences and journals. To enhance the quality of these reviews, we introduce a self-refinement loop that iteratively optimizes the LLM's review prompts. Our framework establishes standardized metrics for evaluating AI-based review systems, thereby bolstering the reliability of AI-generated reviews in academic research. 

**Abstract (ZH)**: 随着学术研究的不断增多以及合格评审人的短缺，需要创新的方法来改进同行评审。尽管大型语言模型（LLMs）在这一过程中具有潜在的应用价值，但它们当前仍存在局限性，如表面化的批评、虚构事实以及缺少可操作的见解。本研究通过引入一项全面的评估框架来解决这些挑战，该框架从以下四个方面评估AI生成的评审：与人类评价的契合度、事实准确性、分析深度以及可操作见解。此外，我们还提出了一种新颖的对齐机制，使LLM生成的评审能够适应各个会议和期刊的独特评价优先事项。为了提高这些评审的质量，我们引入了一个自我精炼的循环，逐步优化LLM的评审提示。此框架建立了评估基于AI的评审系统的标准指标，从而增强了AI生成的评审在学术研究中的可靠性。 

---
# Proactive Depot Discovery: A Generative Framework for Flexible Location-Routing 

**Title (ZH)**: 主动式存储点发现：一种灵活的生成性地点-路径规划框架 

**Authors**: Site Qu, Guoqiang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11715)  

**Abstract**: The Location-Routing Problem (LRP), which combines the challenges of facility (depot) locating and vehicle route planning, is critically constrained by the reliance on predefined depot candidates, limiting the solution space and potentially leading to suboptimal outcomes. Previous research on LRP without predefined depots is scant and predominantly relies on heuristic algorithms that iteratively attempt depot placements across a planar area. Such approaches lack the ability to proactively generate depot locations that meet specific geographic requirements, revealing a notable gap in current research landscape. To bridge this gap, we propose a data-driven generative DRL framework, designed to proactively generate depots for LRP without predefined depot candidates, solely based on customer requests data which include geographic and demand information. It can operate in two distinct modes: direct generation of exact depot locations, and the creation of a multivariate Gaussian distribution for flexible depots sampling. By extracting depots' geographic pattern from customer requests data, our approach can dynamically respond to logistical needs, identifying high-quality depot locations that further reduce total routing costs compared to traditional methods. Extensive experiments demonstrate that, for a same group of customer requests, compared with those depots identified through random attempts, our framework can proactively generate depots that lead to superior solution routes with lower routing cost. The implications of our framework potentially extend into real-world applications, particularly in emergency medical rescue and disaster relief logistics, where rapid establishment and adjustment of depot locations are paramount, showcasing its potential in addressing LRP for dynamic and unpredictable environments. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

仓库地点-路线规划问题（LRP），结合了设施（仓库）定位和车辆路线规划的挑战，但由于依赖预定义的仓库候选点，其解空间受到限制，可能导致次优解决方案。关于没有预定义仓库的LRP的研究很少，且主要依赖迭代尝试在平面区域上定位仓库的启发式算法。这些方法缺乏主动生成符合特定地理要求的仓库位置的能力，揭示了当前研究领域的显著空白。为填补这一空白，我们提出了一种数据驱动的生成深度强化学习（DRL）框架，旨在没有预定义仓库候选点的情况下，仅基于包含地理和需求信息的客户请求数据主动生成仓库。该框架可以以两种不同的模式运行：直接生成精确的仓库位置，以及创建多元高斯分布以灵活采样可变仓库。通过从客户请求数据中提取仓库的地理模式，我们的方法可以动态响应物流需求，进一步识别出与传统方法相比总路线成本更低的高质量仓库位置。广泛实验证明，对于同一组客户请求，与通过随机尝试识别的仓库相比，我们的框架可以主动生成更能导致更优解决方案和更低路线成本的仓库。框架的潜在影响延伸到了实际应用领域，特别是在紧急医疗服务和灾害救援物流中，迅速建立和调整仓库位置至关重要，展示了其在动态和不可预测环境中解决LRP的潜力。 

---
# Knowledge-aware contrastive heterogeneous molecular graph learning 

**Title (ZH)**: 知识增强的对比异构分子图学习 

**Authors**: Mukun Chen, Jia Wu, Shirui Pan, Fu Lin, Bo Du, Xiuwen Gong, Wenbin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11711)  

**Abstract**: Molecular representation learning is pivotal in predicting molecular properties and advancing drug design. Traditional methodologies, which predominantly rely on homogeneous graph encoding, are limited by their inability to integrate external knowledge and represent molecular structures across different levels of granularity. To address these limitations, we propose a paradigm shift by encoding molecular graphs into heterogeneous structures, introducing a novel framework: Knowledge-aware Contrastive Heterogeneous Molecular Graph Learning (KCHML). This approach leverages contrastive learning to enrich molecular representations with embedded external knowledge. KCHML conceptualizes molecules through three distinct graph views-molecular, elemental, and pharmacological-enhanced by heterogeneous molecular graphs and a dual message-passing mechanism. This design offers a comprehensive representation for property prediction, as well as for downstream tasks such as drug-drug interaction (DDI) prediction. Extensive benchmarking demonstrates KCHML's superiority over state-of-the-art molecular property prediction models, underscoring its ability to capture intricate molecular features. 

**Abstract (ZH)**: 分子表示学习在预测分子性质和推动药物设计方面发挥着关键作用。传统方法主要依赖于均质图编码，这限制了它们整合外部知识和在不同粒度水平上表示分子结构的能力。为了应对这些限制，我们提出了一个范式转变，即将分子图编码为异构结构，并引入了一种新型框架：基于知识的对比异构分子图学习（KCHML）。该方法通过对比学习丰富分子表示，并嵌入外部知识。KCHML 通过三种不同的图视图——分子视图、元素视图和药理学视图——来概念化分子，并结合异构分子图和双重消息传递机制进行增强。这种设计为性质预测提供了全面的表示，同时也适用于下游任务，如药物-药物相互作用（DDI）预测。广泛的基准测试表明，KCHML 在最先进的分子性质预测模型中表现出优越性，证明了其捕捉复杂分子特征的能力。 

---
# LLM Agents Making Agent Tools 

**Title (ZH)**: 基于大型语言模型的代理构建代理工具 

**Authors**: Georg Wölflein, Dyke Ferber, Daniel Truhn, Ognjen Arandjelović, Jakob Nikolas Kather  

**Link**: [PDF](https://arxiv.org/pdf/2502.11705)  

**Abstract**: Tool use has turned large language models (LLMs) into powerful agents that can perform complex multi-step tasks by dynamically utilising external software components. However, these tools must be implemented in advance by human developers, hindering the applicability of LLM agents in domains which demand large numbers of highly specialised tools, like in life sciences and medicine. Motivated by the growing trend of scientific studies accompanied by public code repositories, we propose ToolMaker, a novel agentic framework that autonomously transforms papers with code into LLM-compatible tools. Given a short task description and a repository URL, ToolMaker autonomously installs required dependencies and generates code to perform the task, using a closed-loop self-correction mechanism to iteratively diagnose and rectify errors. To evaluate our approach, we introduce a benchmark comprising 15 diverse and complex computational tasks spanning both medical and non-medical domains with over 100 unit tests to objectively assess tool correctness and robustness. ToolMaker correctly implements 80% of the tasks, substantially outperforming current state-of-the-art software engineering agents. ToolMaker therefore is a step towards fully autonomous agent-based scientific workflows. 

**Abstract (ZH)**: 工具使用使大规模语言模型（LLMs）成为能够通过动态利用外部软件组件执行复杂多步任务的强大代理。然而，这些工具必须由人类开发者事先实现，这限制了LLM代理在需要大量高度专业化工具的领域（如生命科学和医学）的应用。鉴于越来越多的研究论文伴随着公开的代码库，我们提出ToolMaker——一种新颖的代理框架，能够自主地将带有代码的论文转换为LLM兼容的工具。给定简短的任务描述和代码库URL，ToolMaker能够自主安装所需的依赖项，并生成代码以执行该任务。通过一个闭合回路的自我纠正机制，不断地诊断和修正错误。为了评估我们的方法，我们引入了一个基准测试集，其中包括15个涵盖医疗和非医疗领域的多样而复杂的计算任务，以及超过100个单元测试，以客观评估工具的正确性和鲁棒性。ToolMaker正确实现80%的任务，显著优于当前最先进的软件工程代理。因此，ToolMaker是完全自主的基于代理的科学工作流程的重要一步。 

---
# ReVeil: Unconstrained Concealed Backdoor Attack on Deep Neural Networks using Machine Unlearning 

**Title (ZH)**: ReVeil：使用机器遗忘进行的深度神经网络无约束隐藏后门攻击 

**Authors**: Manaar Alam, Hithem Lamri, Michail Maniatakos  

**Link**: [PDF](https://arxiv.org/pdf/2502.11687)  

**Abstract**: Backdoor attacks embed hidden functionalities in deep neural networks (DNN), triggering malicious behavior with specific inputs. Advanced defenses monitor anomalous DNN inferences to detect such attacks. However, concealed backdoors evade detection by maintaining a low pre-deployment attack success rate (ASR) and restoring high ASR post-deployment via machine unlearning. Existing concealed backdoors are often constrained by requiring white-box or black-box access or auxiliary data, limiting their practicality when such access or data is unavailable. This paper introduces ReVeil, a concealed backdoor attack targeting the data collection phase of the DNN training pipeline, requiring no model access or auxiliary data. ReVeil maintains low pre-deployment ASR across four datasets and four trigger patterns, successfully evades three popular backdoor detection methods, and restores high ASR post-deployment through machine unlearning. 

**Abstract (ZH)**: 后门攻击将隐藏的功能嵌入到深度神经网络（DNN）中，通过特定的输入触发恶意行为。先进的防御措施监控异常的DNN推断以检测此类攻击。然而，隐藏的后门通过保持低部署前攻击成功率（ASR）并在部署后通过机器遗忘恢复高ASR来规避检测。现有的隐藏后门往往受限于需要白盒或黑盒访问或辅助数据，这限制了它们在无法获取此类访问或数据时的实用性。本文介绍了ReVeil，这是一种针对DNN训练管道的数据采集阶段的隐藏后门攻击，不需要模型访问或辅助数据。ReVeil在四个数据集和四个触发模式下保持低部署前ASR，并成功规避了三种流行的后门检测方法，通过机器遗忘在部署后恢复高ASR。 

---
# MathFimer: Enhancing Mathematical Reasoning by Expanding Reasoning Steps through Fill-in-the-Middle Task 

**Title (ZH)**: MathFimer：通过填空任务扩展推理步骤以增强数学推理能力 

**Authors**: Yuchen Yan, Yongliang Shen, Yang Liu, Jin Jiang, Xin Xu, Mengdi Zhang, Jian Shao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11684)  

**Abstract**: Mathematical reasoning represents a critical frontier in advancing large language models (LLMs). While step-by-step approaches have emerged as the dominant paradigm for mathematical problem-solving in LLMs, the quality of reasoning steps in training data fundamentally constrains the performance of the models. Recent studies has demonstrated that more detailed intermediate steps can enhance model performance, yet existing methods for step expansion either require more powerful external models or incur substantial computational costs. In this paper, we introduce MathFimer, a novel framework for mathematical reasoning step expansion inspired by the "Fill-in-the-middle" task from code completion. By decomposing solution chains into prefix-suffix pairs and training models to reconstruct missing intermediate steps, we develop a specialized model, MathFimer-7B, on our carefully curated NuminaMath-FIM dataset. We then apply these models to enhance existing mathematical reasoning datasets by inserting detailed intermediate steps into their solution chains, creating MathFimer-expanded versions. Through comprehensive experiments on multiple mathematical reasoning datasets, including MathInstruct, MetaMathQA and etc., we demonstrate that models trained on MathFimer-expanded data consistently outperform their counterparts trained on original data across various benchmarks such as GSM8K and MATH. Our approach offers a practical, scalable solution for enhancing mathematical reasoning capabilities in LLMs without relying on powerful external models or expensive inference procedures. 

**Abstract (ZH)**: 数学推理是推进大型语言模型（LLMs）发展的关键前沿领域。虽然逐步方法已成为LLMs中数学问题解决的主要范式，但训练数据中的推理步骤质量从根本上限制了模型的性能。最近的研究表明，更详细的中间步骤可以提升模型性能，但现有步骤扩展方法要么需要更强大的外部模型，要么会产生巨额计算成本。在本文中，我们介绍了一种名为MathFimer的新框架，该框架受到代码补全中的“填充中间部分”任务的启发。通过将解决方案链分解为前缀-后缀对，并训练模型重建缺失的中间步骤，我们构建了一个专门模型——MathFimer-7B，该模型基于我们精心编纂的NuminaMath-FIM数据集。随后，我们将这些模型应用于现有的数学推理数据集中，通过插入详细的中间步骤来增强其解决方案链，形成了MathFimer扩充版本。通过在多个数学推理数据集（包括MathInstruct、MetaMathQA等）上的全面实验，我们证明了在MathFimer扩充数据上训练的模型在各种基准测试（如GSM8K和MATH）中始终优于在原始数据上训练的模型。我们的方法提供了一种实用且可扩展的解决方案，能够在不依赖于强大外部模型或昂贵推理过程的情况下提升LLMs的数学推理能力。 

---
# RIDE: Enhancing Large Language Model Alignment through Restyled In-Context Learning Demonstration Exemplars 

**Title (ZH)**: RIDE：通过重塑上下文学习示范范例增强大型语言模型对齐 

**Authors**: Yuncheng Hua, Lizhen Qu, Zhuang Li, Hao Xue, Flora D. Salim, Gholamreza Haffari  

**Link**: [PDF](https://arxiv.org/pdf/2502.11681)  

**Abstract**: Alignment tuning is crucial for ensuring large language models (LLMs) behave ethically and helpfully. Current alignment approaches require high-quality annotations and significant training resources. This paper proposes a low-cost, tuning-free method using in-context learning (ICL) to enhance LLM alignment. Through an analysis of high-quality ICL demos, we identified style as a key factor influencing LLM alignment capabilities and explicitly restyled ICL exemplars based on this stylistic framework. Additionally, we combined the restyled demos to achieve a balance between the two conflicting aspects of LLM alignment--factuality and safety. We packaged the restyled examples as prompts to trigger few-shot learning, improving LLM alignment. Compared to the best baseline approach, with an average score of 5.00 as the maximum, our method achieves a maximum 0.10 increase on the Alpaca task (from 4.50 to 4.60), a 0.22 enhancement on the Just-eval benchmark (from 4.34 to 4.56), and a maximum improvement of 0.32 (from 3.53 to 3.85) on the MT-Bench dataset. We release the code and data at this https URL. 

**Abstract (ZH)**: 以下是经过学术规范翻译的中文内容：

调准调整对于确保大规模语言模型（LLMs）行为符合伦理和具有帮助性至关重要。当前的调准方法需要高质量的标注和大量训练资源。本文提出了一种低成本、无需调准的方法，利用上下文学习（ICL）来增强LLM的调准能力。通过对高质量ICL示例的分析，我们确定了风格是影响LLM调准能力的关键因素，并基于这种风格框架显式地重塑ICL示例。此外，我们结合重置风格的示例，在LLM调准的两个相互冲突方面——事实性和安全性之间实现了平衡。我们将重置风格的示例打包成提示，触发少量样本学习，从而提高LLM的调准能力。与最好的基线方法相比，我们的方法在Alpaca任务中的得分提高了0.10（从4.50增加到4.60），在Just-eval基准中的得分提高了0.22（从4.34增加到4.56），在MT-Bench数据集中最高提高了0.32（从3.53增加到3.85）。我们已在以下链接发布了代码和数据：[链接地址]。 

---
# Diversity-Oriented Data Augmentation with Large Language Models 

**Title (ZH)**: 面向多样性的数据增强方法：基于大型语言模型 

**Authors**: Zaitian Wang, Jinghan Zhang, Xinhao Zhang, Kunpeng Liu, Pengfei Wang, Yuanchun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.11671)  

**Abstract**: Data augmentation is an essential technique in natural language processing (NLP) for enriching training datasets by generating diverse samples. This process is crucial for improving the robustness and generalization capabilities of NLP models. However, a significant challenge remains: \textit{Insufficient Attention to Sample Distribution Diversity}. Most existing methods focus on increasing the sample numbers while neglecting the sample distribution diversity, which can lead to model overfitting. In response, we explore data augmentation's impact on dataset diversity and propose a \textbf{\underline{D}}iversity-\textbf{\underline{o}}riented data \textbf{\underline{Aug}}mentation framework (\textbf{DoAug}). % \(\mathscr{DoAug}\) Specifically, we utilize a diversity-oriented fine-tuning approach to train an LLM as a diverse paraphraser, which is capable of augmenting textual datasets by generating diversified paraphrases. Then, we apply the LLM paraphraser to a selected coreset of highly informative samples and integrate the paraphrases with the original data to create a more diverse augmented dataset. Finally, we conduct extensive experiments on 12 real-world textual datasets. The results show that our fine-tuned LLM augmenter improves diversity while preserving label consistency, thereby enhancing the robustness and performance of downstream tasks. Specifically, it achieves an average performance gain of \(10.52\%\), surpassing the runner-up baseline with more than three percentage points. 

**Abstract (ZH)**: 数据增强是自然语言处理（NLP）中一种不可或缺的技术，通过生成多样化的样本来丰富训练数据集。这一过程对于提高NLP模型的鲁棒性和泛化能力至关重要。然而，仍存在一个显著的挑战：**忽视样本分布的多样性**。大多数现有方法侧重于增加样本数量，而忽视了样本分布的多样性，这可能导致模型过拟合。为应对这一挑战，我们探索了数据增强对数据集多样性的影响，并提出了一个**Diversity-oriented Data Augmentation框架（DoAug）**。具体而言，我们采用一种多样性和目标导向的微调方法训练一个LLM作为多元并行的生成器，以生成多样化的替代文本。然后，我们将LLM并行生成器应用于一组选定的具有高度信息性的子样本集，并将生成的替代文本与原有数据整合，以创建更加多样化的数据集。最后，我们在12个真实世界的文本数据集上进行了广泛的实验。实验结果表明，我们的微调LLM生成器在保持标签一致性的同时提高了数据集的多样性，从而增强了下游任务的鲁棒性和性能。具体而言，相较于第二名基线模型，我们的方法获得了平均10.52%的性能提升，高出约三个百分点。 

---
# "I'm not for sale" -- Perceptions and limited awareness of privacy risks by digital natives about location data 

**Title (ZH)**: “我不出售”——数字原住民对位置数据隐私风险的感知及其有限的认识 

**Authors**: Antoine Boutet, Victor Morel  

**Link**: [PDF](https://arxiv.org/pdf/2502.11658)  

**Abstract**: Although mobile devices benefit users in their daily lives in numerous ways, they also raise several privacy concerns. For instance, they can reveal sensitive information that can be inferred from location data. This location data is shared through service providers as well as mobile applications. Understanding how and with whom users share their location data -- as well as users' perception of the underlying privacy risks --, are important notions to grasp in order to design usable privacy-enhancing technologies. In this work, we perform a quantitative and qualitative analysis of smartphone users' awareness, perception and self-reported behavior towards location data-sharing through a survey of n=99 young adult participants (i.e., digital natives). We compare stated practices with actual behaviors to better understand their mental models, and survey participants' understanding of privacy risks before and after the inspection of location traces and the information that can be inferred therefrom.
Our empirical results show that participants have risky privacy practices: about 54% of participants underestimate the number of mobile applications to which they have granted access to their data, and 33% forget or do not think of revoking access to their data. Also, by using a demonstrator to perform inferences from location data, we observe that slightly more than half of participants (57%) are surprised by the extent of potentially inferred information, and that 47% intend to reduce access to their data via permissions as a result of using the demonstrator. Last, a majority of participants have little knowledge of the tools to better protect themselves, but are nonetheless willing to follow suggestions to improve privacy (51%). Educating people, including digital natives, about privacy risks through transparency tools seems a promising approach. 

**Abstract (ZH)**: 尽管移动设备在日常生活中为用户提供了很多便利，但它们也引发了一些隐私问题。例如，它们可以通过位置数据揭示出敏感信息。这些位置数据不仅通过服务提供商，还通过手机应用进行共享。理解用户如何以及与谁分享位置数据，以及用户对潜在隐私风险的看法，是设计可使用的隐私增强技术的重要概念。在本研究中，我们对99名年轻成人（即数字原住民）进行了调查，以定性和定量分析他们对位置数据共享的意识、感知和自报行为。我们将声明的习惯做法与实际行为进行比较，以更好地了解他们的心理模型，并在检查位置痕迹及其推断信息之前和之后，调查参与者的隐私风险理解。

我们的实证结果显示，参与者存在一些有风险的隐私实践：大约54%的参与者低估了他们已经授权数据访问的移动应用数量，而33%的参与者忘记了或没有考虑到撤销数据访问权限。此外，通过演示工具从位置数据中进行推理，我们发现略高于一半的参与者（57%）对潜在可推断信息的范围感到惊讶，并且有47%的参与者在使用演示工具后打算通过权限调整来减少数据访问。最后，大多数参与者对更好的自我保护工具知之甚少，但仍有51%的参与者愿意采纳建议以提高隐私保护水平。通过透明工具教育包括数字原住民在内的人们关于隐私风险，似乎是行之有效的方法。 

---
# MMXU: A Multi-Modal and Multi-X-ray Understanding Dataset for Disease Progression 

**Title (ZH)**: MMXU：一种用于疾病进展理解的多模态和多X射线数据集 

**Authors**: Linjie Mu, Zhongzhen Huang, Shengqian Qin, Yakun Zhu, Shaoting Zhang, Xiaofan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11651)  

**Abstract**: Large vision-language models (LVLMs) have shown great promise in medical applications, particularly in visual question answering (MedVQA) and diagnosis from medical images. However, existing datasets and models often fail to consider critical aspects of medical diagnostics, such as the integration of historical records and the analysis of disease progression over time. In this paper, we introduce MMXU (Multimodal and MultiX-ray Understanding), a novel dataset for MedVQA that focuses on identifying changes in specific regions between two patient visits. Unlike previous datasets that primarily address single-image questions, MMXU enables multi-image questions, incorporating both current and historical patient data. We demonstrate the limitations of current LVLMs in identifying disease progression on MMXU-\textit{test}, even those that perform well on traditional benchmarks. To address this, we propose a MedRecord-Augmented Generation (MAG) approach, incorporating both global and regional historical records. Our experiments show that integrating historical records significantly enhances diagnostic accuracy by at least 20\%, bridging the gap between current LVLMs and human expert performance. Additionally, we fine-tune models with MAG on MMXU-\textit{dev}, which demonstrates notable improvements. We hope this work could illuminate the avenue of advancing the use of LVLMs in medical diagnostics by emphasizing the importance of historical context in interpreting medical images. Our dataset is released at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 大规模视觉-语言模型（LVLMs）在医疗应用中显示出巨大的潜力，特别是在医学视觉问答（MedVQA）和医学图像诊断方面。然而，现有的数据集和模型往往未能考虑医学诊断中的关键方面，如历史记录的整合和疾病进展的分析。在本文中，我们介绍了MMXU（多模态和多X光理解），一个新的用于MedVQA的数据集，专注于识别两位患者就诊之间特定区域的变化。与主要处理单张图片问题的先前数据集不同，MMXU 支持多张图片的问题，结合了当前和历史患者的双重数据。我们通过MMXU-\textit{test}展示了当前LVLMs在识别疾病进展方面的局限性，即使是那些在传统基准测试中表现良好的模型。为了解决这一问题，我们提出了一种MedRecord-Augmented Generation（MAG）方法，结合了全局和区域的历史记录。实验结果表明，整合历史记录可以显著提高诊断准确性至少20%，缩小当前LVLMs与人类专家表现之间的差距。此外，我们在MMXU-\textit{dev}上对MAG进行了微调，证明了显著的改进。我们希望这项工作能够揭示出通过强调医疗图像解释中的历史上下文来推进LVLMs在医疗诊断中的使用方向。我们的数据集可以在 \href{this https URL}{这个链接} 获取。 

---
# DELMAN: Dynamic Defense Against Large Language Model Jailbreaking with Model Editing 

**Title (ZH)**: DELMAN：基于模型编辑的动态防御大型语言模型逃逸攻击 

**Authors**: Yi Wang, Fenghua Weng, Sibei Yang, Zhan Qin, Minlie Huang, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11647)  

**Abstract**: Large Language Models (LLMs) are widely applied in decision making, but their deployment is threatened by jailbreak attacks, where adversarial users manipulate model behavior to bypass safety measures. Existing defense mechanisms, such as safety fine-tuning and model editing, either require extensive parameter modifications or lack precision, leading to performance degradation on general tasks, which is unsuitable to post-deployment safety alignment. To address these challenges, we propose DELMAN (Dynamic Editing for LLMs JAilbreak DefeNse), a novel approach leveraging direct model editing for precise, dynamic protection against jailbreak attacks. DELMAN directly updates a minimal set of relevant parameters to neutralize harmful behaviors while preserving the model's utility. To avoid triggering a safe response in benign context, we incorporate KL-divergence regularization to ensure the updated model remains consistent with the original model when processing benign queries. Experimental results demonstrate that DELMAN outperforms baseline methods in mitigating jailbreak attacks while preserving the model's utility, and adapts seamlessly to new attack instances, providing a practical and efficient solution for post-deployment model protection. 

**Abstract (ZH)**: 大型语言模型（LLMs）在决策制定中广泛应用，但其部署受到 Jailbreak 攻击的威胁，即恶意用户操控模型行为以规避安全措施。现有的防御机制，如安全性微调和模型编辑，要么需要大量的参数修改，要么缺乏精确性，导致在通用任务上性能下降，不适用于部署后的安全对齐。为应对这些挑战，我们提出了一种名为 DELMAN（动态编辑用于 LLMs Jailbreak 防护）的新颖方法，该方法利用直接模型编辑为精确、动态的 Jailbreak 防护提供支持。DELMAN 直接更新一小部分相关参数，以消除有害行为同时保持模型的实用性。为了防止在无害情境下触发安全响应，我们引入了 KL 散度正则化，确保更新后的模型在处理无害查询时与原始模型保持一致。实验结果表明，DELMAN 在减轻 Jailbreak 攻击的同时保持模型的实用性，能够无缝适应新的攻击实例，提供了一种实用且高效的部署后模型保护方案。 

---
# InTec: integrated things-edge computing: a framework for distributing machine learning pipelines in edge AI systems 

**Title (ZH)**: IntTec：综合事物边缘计算：边缘AI系统中分布机器学习管道的框架 

**Authors**: Habib Larian, Faramarz Safi-Esfahani  

**Link**: [PDF](https://arxiv.org/pdf/2502.11644)  

**Abstract**: With the rapid expansion of the Internet of Things (IoT), sensors, smartphones, and wearables have become integral to daily life, powering smart applications in home automation, healthcare, and intelligent transportation. However, these advancements face significant challenges due to latency and bandwidth constraints imposed by traditional cloud based machine learning (ML) frameworks. The need for innovative solutions is evident as cloud computing struggles with increased latency and network congestion. Previous attempts to offload parts of the ML pipeline to edge and cloud layers have yet to fully resolve these issues, often worsening system response times and network congestion due to the computational limitations of edge devices. In response to these challenges, this study introduces the InTec (Integrated Things Edge Computing) framework, a groundbreaking innovation in IoT architecture. Unlike existing methods, InTec fully leverages the potential of a three tier architecture by strategically distributing ML tasks across the Things, Edge, and Cloud layers. This comprehensive approach enables real time data processing at the point of data generation, significantly reducing latency, optimizing network traffic, and enhancing system reliability. InTec effectiveness is validated through empirical evaluation using the MHEALTH dataset for human motion detection in smart homes, demonstrating notable improvements in key metrics: an 81.56 percent reduction in response time, a 10.92 percent decrease in network traffic, a 9.82 percent improvement in throughput, a 21.86 percent reduction in edge energy consumption, and a 25.83 percent reduction in cloud energy consumption. These advancements establish InTec as a new benchmark for scalable, responsive, and energy efficient IoT applications, demonstrating its potential to revolutionize how the ML pipeline is integrated into Edge AI (EI) systems. 

**Abstract (ZH)**: 随着物联网（IoT）的迅速扩张，传感器、智能手机和可穿戴设备已成为日常生活中不可或缺的组成部分，推动了家庭自动化、医疗保健和智能交通等智能应用的发展。然而，这些进步由于传统基于云的机器学习（ML）框架所引起的延迟和带宽限制而面临重大挑战。随着云计算面临增加的延迟和网络拥塞问题，创新解决方案的需求日益明显。此前尝试将ML管道的部分任务卸载到边缘和云层尚未完全解决这些问题，反而因边缘设备的计算限制而恶化了系统响应时间和网络拥塞问题。为应对这些挑战，本研究提出了InTec（集成事物边缘计算）框架，这是一种物联网架构领域的突破性创新。与现有方法不同，InTec能够通过对事物、边缘和云层的ML任务进行战略性分配，充分利用三层架构的潜力。这样的全面方法能够在数据生成点进行实时数据处理，大幅减少延迟，优化网络流量，并增强系统可靠性。InTec的有效性通过使用MHEALTH数据集进行人体运动检测的实证评估得到了验证，展示了在关键指标上的显著改进：响应时间减少了81.56%，网络流量减少了10.92%，吞吐量提高了9.82%，边缘能量消耗减少了21.86%，云能量消耗减少了25.83%。这些进步确立了InTec作为可扩展、响应迅速和节能的物联网应用的新基准，展示了其在如何将ML管道整合到边缘AI（EI）系统中的潜在革命性影响。 

---
# Neural Interpretable Reasoning 

**Title (ZH)**: 神经可解释推理 

**Authors**: Pietro Barbiero, Giuseppe Marra, Gabriele Ciravegna, David Debot, Francesco De Santis, Michelangelo Diligenti, Mateo Espinosa Zarlenga, Francesco Giannini  

**Link**: [PDF](https://arxiv.org/pdf/2502.11639)  

**Abstract**: We formalize a novel modeling framework for achieving interpretability in deep learning, anchored in the principle of inference equivariance. While the direct verification of interpretability scales exponentially with the number of variables of the system, we show that this complexity can be mitigated by treating interpretability as a Markovian property and employing neural re-parametrization techniques. Building on these insights, we propose a new modeling paradigm -- neural generation and interpretable execution -- that enables scalable verification of equivariance. This paradigm provides a general approach for designing Neural Interpretable Reasoners that are not only expressive but also transparent. 

**Abstract (ZH)**: 我们将一种新的建模框架形式化，以实现深度学习中的可解释性，该框架基于推理不变性的原则。由于直接验证可解释性随系统变量数量的增加而呈指数级增长，我们通过将可解释性视为马尔可夫性质并采用神经重参数化技术，展示了可以减轻这种复杂性。基于这些洞察，我们提出了一种新的建模范式——神经生成与可解释执行，该范式能够实现可标量化验证的不变性。该范式提供了一种通用的方法来设计既具有表达力又具有透明度的神经可解释推理器。 

---
# In-Context Parametric Inference: Point or Distribution Estimators? 

**Title (ZH)**: 上下文感知参数推断：点估计还是分布估计？ 

**Authors**: Sarthak Mittal, Yoshua Bengio, Nikolay Malkin, Guillaume Lajoie  

**Link**: [PDF](https://arxiv.org/pdf/2502.11617)  

**Abstract**: Bayesian and frequentist inference are two fundamental paradigms in statistical estimation. Bayesian methods treat hypotheses as random variables, incorporating priors and updating beliefs via Bayes' theorem, whereas frequentist methods assume fixed but unknown hypotheses, relying on estimators like maximum likelihood. While extensive research has compared these approaches, the frequentist paradigm of obtaining point estimates has become predominant in deep learning, as Bayesian inference is challenging due to the computational complexity and the approximation gap of posterior estimation methods. However, a good understanding of trade-offs between the two approaches is lacking in the regime of amortized estimators, where in-context learners are trained to estimate either point values via maximum likelihood or maximum a posteriori estimation, or full posteriors using normalizing flows, score-based diffusion samplers, or diagonal Gaussian approximations, conditioned on observations. To help resolve this, we conduct a rigorous comparative analysis spanning diverse problem settings, from linear models to shallow neural networks, with a robust evaluation framework assessing both in-distribution and out-of-distribution generalization on tractable tasks. Our experiments indicate that amortized point estimators generally outperform posterior inference, though the latter remain competitive in some low-dimensional problems, and we further discuss why this might be the case. 

**Abstract (ZH)**: 贝叶斯推断和频率推断是统计估计的两种基本范式。贝叶斯方法将假设视为随机变量，通过贝叶斯定理引入先验并更新信念，而频率方法假设固定但未知的假设，并依赖似然估计量，如最大似然估计。虽然已有大量研究对比了这些方法，但在深度学习中，由于贝叶斯推断的计算复杂性和后验估计方法的近似差距，频率方法中的点估计范式已成为主流。然而，在可编程估计器的范围内，对这两种方法之间的权衡理解仍然不足。在这个范围内，上下文环境学习者被训练以不同的方式进行估计：通过最大似然或最大后验估计估计点值，或使用归一化流、分数扩散采样器或对角高斯近似估计完整的后验分布，这些均基于观察数据进行条件估计。为解决这一问题，我们进行了一项涵盖多种问题设置的严格对比分析，从线性模型到浅层神经网络，采用稳健的评估框架，评估其在可处理任务中的同分布和异分布泛化能力。实验结果表明，可编程点估计器通常优于后验推断，尽管在某些低维问题中后验推断仍具有竞争力，我们进一步探讨了这种现象的原因。 

---
# Is Human-Like Text Liked by Humans? Multilingual Human Detection and Preference Against AI 

**Title (ZH)**: 人类偏好的文本是否具有人类特性？多语言人类检测与对AI的偏好对比 

**Authors**: Yuxia Wang, Rui Xing, Jonibek Mansurov, Giovanni Puccetti, Zhuohan Xie, Minh Ngoc Ta, Jiahui Geng, Jinyan Su, Mervat Abassy, Saad El Dine Ahmed, Kareem Elozeiri, Nurkhan Laiyk, Maiya Goloburda, Tarek Mahmoud, Raj Vardhan Tomar, Alexander Aziz, Ryuto Koike, Masahiro Kaneko, Artem Shelmanov, Ekaterina Artemova, Vladislav Mikhailov, Akim Tsvigun, Alham Fikri Aji, Nizar Habash, Iryna Gurevych, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2502.11614)  

**Abstract**: Prior studies have shown that distinguishing text generated by large language models (LLMs) from human-written one is highly challenging, and often no better than random guessing. To verify the generalizability of this finding across languages and domains, we perform an extensive case study to identify the upper bound of human detection accuracy. Across 16 datasets covering 9 languages and 9 domains, 19 annotators achieved an average detection accuracy of 87.6%, thus challenging previous conclusions. We find that major gaps between human and machine text lie in concreteness, cultural nuances, and diversity. Prompting by explicitly explaining the distinctions in the prompts can partially bridge the gaps in over 50% of the cases. However, we also find that humans do not always prefer human-written text, particularly when they cannot clearly identify its source. 

**Abstract (ZH)**: 先前的研究表明，区分由大规模语言模型（LLMs）生成的文本与人工撰写的文本极具挑战性，往往与随机猜测无异。为了验证这一发现的普适性，即在不同语言和领域中的推广能力，我们进行了一项广泛的案例研究，以确定人类检测准确性的上限。在涵盖9种语言和9个领域的16个数据集中，19名注释者实现了平均检测准确率为87.6%，从而挑战了之前的结论。我们发现，人类与机器生成文本之间的主要差距在于具体性、文化细微差异以及多样性。通过明确解释提示中的区别，可以在超过50%的情况下部分缩小这些差距。然而，我们还发现人类并不总是偏好人工撰写的文本，尤其是在他们无法明确识别其来源时。 

---
# Maximum Entropy Reinforcement Learning with Diffusion Policy 

**Title (ZH)**: 最大熵强化学习与扩散策略 

**Authors**: Xiaoyi Dong, Jian Cheng, Xi Sheryl Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11612)  

**Abstract**: The Soft Actor-Critic (SAC) algorithm with a Gaussian policy has become a mainstream implementation for realizing the Maximum Entropy Reinforcement Learning (MaxEnt RL) objective, which incorporates entropy maximization to encourage exploration and enhance policy robustness. While the Gaussian policy performs well on simpler tasks, its exploration capacity and potential performance in complex multi-goal RL environments are limited by its inherent unimodality. In this paper, we employ the diffusion model, a powerful generative model capable of capturing complex multimodal distributions, as the policy representation to fulfill the MaxEnt RL objective, developing a method named MaxEnt RL with Diffusion Policy (MaxEntDP). Our method enables efficient exploration and brings the policy closer to the optimal MaxEnt policy. Experimental results on Mujoco benchmarks show that MaxEntDP outperforms the Gaussian policy and other generative models within the MaxEnt RL framework, and performs comparably to other state-of-the-art diffusion-based online RL algorithms. Our code is available at this https URL. 

**Abstract (ZH)**: 软Actor- Critic（SAC）算法结合高斯策略已成为实现最大熵强化学习（MaxEnt RL）目标的主要实现方式，该目标通过最大化熵来促进探索并增强策略的稳健性。虽然高斯策略在简单的任务上表现良好，但其探索能力和在复杂多目标强化学习环境中的潜在性能受限于其实质的单模态性。本文采用扩散模型，这是一种能够捕捉复杂多模态分布的强大生成模型，作为策略表示以实现MaxEnt RL目标，并开发了一种名为MaxEnt RL with Diffusion Policy（MaxEntDP）的方法。该方法能够有效地促进探索，并使策略更接近最优的MaxEnt策略。在Mujoco基准测试上的实验结果表明，MaxEntDP在MaxEnt RL框架内的高斯策略和其他生成模型中表现出色，并且在与其他基于扩散的在线RL算法的性能相当。我们的代码可在以下链接获取：this https URL。 

---
# Identifying Gender Stereotypes and Biases in Automated Translation from English to Italian using Similarity Networks 

**Title (ZH)**: 将以下论文内容或标题翻译成中文，并符合学术规范：

Identifying Gender Stereotypes and Biases in Automated Translation from English to Italian using Similarity Networks

性别刻板印象和平等在使用相似性网络的英译义自动化翻译中的识别 

**Authors**: Fatemeh Mohammadi, Marta Annamaria Tamborini, Paolo Ceravolo, Costanza Nardocci, Samira Maghool  

**Link**: [PDF](https://arxiv.org/pdf/2502.11611)  

**Abstract**: This paper is a collaborative effort between Linguistics, Law, and Computer Science to evaluate stereotypes and biases in automated translation systems. We advocate gender-neutral translation as a means to promote gender inclusion and improve the objectivity of machine translation. Our approach focuses on identifying gender bias in English-to-Italian translations. First, we define gender bias following human rights law and linguistics literature. Then we proceed by identifying gender-specific terms such as she/lei and he/lui as key elements. We then evaluate the cosine similarity between these target terms and others in the dataset to reveal the model's perception of semantic relations. Using numerical features, we effectively evaluate the intensity and direction of the bias. Our findings provide tangible insights for developing and training gender-neutral translation algorithms. 

**Abstract (ZH)**: 本文是语言学、法学和计算机科学的跨学科合作研究，旨在评估自动化翻译系统中的刻板印象和偏见。我们提倡无性别歧视的翻译，以促进性别包容并提高机器翻译的客观性。本研究主要关注英译意中的性别偏见问题。首先，我们根据人权法和语言学文献来定义性别偏见。然后，我们将女性特指术语（如“她/lei”）和男性特指术语（如“他/lui”）作为关键元素进行识别。接着，我们通过评估这些目标术语与其他数据集中术语的余弦相似度，揭示模型对语义关系的理解。利用数值特征，我们有效地评估了偏见的强度和方向。我们的研究结果为开发和训练无性别歧视的翻译算法提供了实际见解。 

---
# DR.GAP: Mitigating Bias in Large Language Models using Gender-Aware Prompting with Demonstration and Reasoning 

**Title (ZH)**: DR.GAP: 使用性别意识示范与推理提示减轻大型语言模型中的偏见 

**Authors**: Hongye Qiu, Yue Xu, Meikang Qiu, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11603)  

**Abstract**: Large Language Models (LLMs) exhibit strong natural language processing capabilities but also inherit and amplify societal biases, including gender bias, raising fairness concerns. Existing debiasing methods face significant limitations: parameter tuning requires access to model weights, prompt-based approaches often degrade model utility, and optimization-based techniques lack generalizability. To address these challenges, we propose this http URL (Demonstration and Reasoning for Gender-Aware Prompting), an automated and model-agnostic approach that mitigates gender bias while preserving model performance. this http URL selects bias-revealing examples and generates structured reasoning to guide models toward more impartial responses. Extensive experiments on coreference resolution and QA tasks across multiple LLMs (GPT-3.5, Llama3, and Llama2-Alpaca) demonstrate its effectiveness, generalization ability, and robustness. this http URL can generalize to vision-language models (VLMs), achieving significant bias reduction. 

**Abstract (ZH)**: 大型语言模型（LLMs）表现出强大的自然语言处理能力，但也继承并放大了社会偏见，包括性别偏见，这引发了公平性问题。现有的去偏见方法面临显著的局限性：参数调整需要访问模型权重，基于提示的方法通常会降低模型的实用性，而基于优化的技术缺乏普适性。为了解决这些挑战，我们提出了一种名为this http URL（性别意识提示的示范与推理）的自动化且模型无关的方法，该方法可以在不牺牲模型性能的情况下减轻性别偏见。this http URL 选择揭示偏见的示例并生成结构化的推理来引导模型生成更为公平的响应。在多个LLM（GPT-3.5、Llama3和Llama2-Alpaca）的核心参照解析和问答任务上进行的 extensive 实验显示了其有效性、普适能力和鲁棒性。此外，this http URL 可以泛化到 Vision-Language 模型（VLMs），显著减少了其中的偏见。 

---
# LLM Embeddings for Deep Learning on Tabular Data 

**Title (ZH)**: 大规模语言模型嵌入在表格数据深度学习中的应用 

**Authors**: Boshko Koloski, Andrei Margeloiu, Xiangjian Jiang, Blaž Škrlj, Nikola Simidjievski, Mateja Jamnik  

**Link**: [PDF](https://arxiv.org/pdf/2502.11596)  

**Abstract**: Tabular deep-learning methods require embedding numerical and categorical input features into high-dimensional spaces before processing them. Existing methods deal with this heterogeneous nature of tabular data by employing separate type-specific encoding approaches. This limits the cross-table transfer potential and the exploitation of pre-trained knowledge. We propose a novel approach that first transforms tabular data into text, and then leverages pre-trained representations from LLMs to encode this data, resulting in a plug-and-play solution to improv ing deep-learning tabular methods. We demonstrate that our approach improves accuracy over competitive models, such as MLP, ResNet and FT-Transformer, by validating on seven classification datasets. 

**Abstract (ZH)**: 表型深度学习方法在处理数据之前，需要将数值和分类输入特征嵌入到高维空间中。现有方法通过使用特定类型的数据编码方法来应对表型数据的异构性，这限制了跨表传输的潜力以及预训练知识的利用。我们提出了一种新的方法，首先将表型数据转换为文本，然后利用预训练的大语言模型（LLM）来编码这些数据，从而提供一种即插即用的解决方案，以改进深度学习表型方法。我们通过在七个分类数据集上进行验证，表明我们的方法在准确率上优于竞争模型，如MLP、ResNet和FT-Transformer。 

---
# Language Complexity Measurement as a Noisy Zero-Shot Proxy for Evaluating LLM Performance 

**Title (ZH)**: 语言复杂度测量作为评估大语言模型性能的无监督噪声代理 

**Authors**: Birger Moell, Johan Boye  

**Link**: [PDF](https://arxiv.org/pdf/2502.11578)  

**Abstract**: Large Language Models (LLMs) have made significant strides in natural language generation but often face challenges in tasks requiring precise calculations and structural analysis. This paper investigates the performance of state-of-the-art LLMs on language complexity measurement tasks, through the computation of the LIX readability metric and Average Dependency Distance (ADD). Using Swedish high school and university-level essays, we evaluate the models' abilities to compute LIX scores and perform dependency parsing, comparing their results to established ground truths. Our findings reveal that while all models demonstrate some capacity for these tasks, ChatGPT-o1-mini performs most consistently, achieving the highest accuracy in both LIX computation and dependency parsing. Additionally, we observe a strong significant correlation -0.875 p 0.026 (N=6) between the models' accuracy in computing LIX and their overall performance on the Massive Multitask Language Understanding (MMLU) benchmark. These results suggest that language complexity measurement abilities can serve as a noisy zero-shot proxies for assessing the general capabilities of LLMs, providing a practical method for model evaluation without the need for extensive benchmarking datasets. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言生成方面取得了显著进展，但在需要精确计算和结构分析的任务中经常遇到挑战。本文通过计算LIX可读性指标和平均依存距离（ADD）来研究最新 state-of-the-art LLMs 在语言复杂性测量任务中的表现。我们使用瑞典高中生和大学生级别的散文，评估模型计算LIX得分和执行依存解析的能力，并将结果与已建立的标准事实进行比较。研究结果表明，虽然所有模型在这些任务上都表现出一定的能力，但ChatGPT-o1-mini表现出最一致，分别在LIX计算和依存解析方面取得了最高的准确率。此外，我们还观察到模型在计算LIX得分的准确性与其在大规模多任务语言理解（MMLU）基准测试中的总体表现之间存在显著的强相关性（相关系数 -0.875，p 值 0.026，N=6）。这些结果表明，语言复杂性测量能力可以作为评估LLMs通用能力的嘈杂零样本代理，提供了一种无需大量基准数据集即可评估模型性能的实用方法。 

---
# InfiR : Crafting Effective Small Language Models and Multimodal Small Language Models in Reasoning 

**Title (ZH)**: InfiR：设计有效的小型语言模型和推理中的多模态小型语言模型 

**Authors**: Congkai Xie, Shuo Cai, Wenjun Wang, Pengxiang Li, Zhijie Sang, Kejing Yang, Yiming Zhang, Zhen Li, Guanghao Zhu, Zeyu Liu, Yang Yu, Yuhang Liu, Su Lu, Baoyi He, Qi Zhou, Xiaotian Han, Jianbo Yuan, Shengyu Zhang, Fei Wu, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11573)  

**Abstract**: Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) have made significant advancements in reasoning capabilities. However, they still face challenges such as high computational demands and privacy concerns. This paper focuses on developing efficient Small Language Models (SLMs) and Multimodal Small Language Models (MSLMs) that retain competitive reasoning abilities. We introduce a novel training pipeline that enhances reasoning capabilities and facilitates deployment on edge devices, achieving state-of-the-art performance while minimizing development costs. \InfR~ aims to advance AI systems by improving reasoning, reducing adoption barriers, and addressing privacy concerns through smaller model sizes. Resources are available at https://github. com/Reallm-Labs/InfiR. 

**Abstract (ZH)**: 大型语言模型（LLMs）和多模态大型语言模型（MLLMs）在推理能力方面取得了显著进步。然而，这些模型仍然面临着计算需求高和隐私问题等挑战。本文专注于开发高效的小型语言模型（SLMs）和多模态小型语言模型（MSLMs），同时保持竞争力的推理能力。我们提出了一种新颖的训练流程，以增强推理能力并促进在边缘设备上的部署，从而在降低成本的同时实现最先进的性能。InfR旨在通过改进推理能力、降低采用障碍和通过较小的模型大小解决隐私问题来推动AI系统的进步。有关资源可访问：<https://github.com/Reallm-Labs/InfiR>。 

---
# Towards Reasoning Ability of Small Language Models 

**Title (ZH)**: 小语言模型的推理能力研究 

**Authors**: Gaurav Srivastava, Shuxiang Cao, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11569)  

**Abstract**: Reasoning has long been viewed as an emergent property of large language models (LLMs), appearing at or above a certain scale ($\sim$100B parameters). However, recent studies challenge this assumption, showing that small language models (SLMs) can also achieve competitive reasoning performance. SLMs are increasingly favored for their efficiency and deployability. However, there is a lack of systematic study on the reasoning abilities of diverse SLMs, including those trained from scratch or derived from LLMs through quantization, pruning, and distillation. This raises a critical question: Can SLMs achieve reasoning abilities comparable to LLMs? In this work, we systematically survey, benchmark, and analyze 72 SLMs from six model families across 14 reasoning benchmarks. For reliable evaluation, we examine four evaluation methods and compare four LLM judges against human evaluations on 800 data points. We repeat all experiments three times to ensure a robust performance assessment. Additionally, we analyze the impact of different prompting strategies in small models. Beyond accuracy, we also evaluate model robustness under adversarial conditions and intermediate reasoning steps. Our findings challenge the assumption that scaling is the only way to achieve strong reasoning. Instead, we foresee a future where SLMs with strong reasoning capabilities can be developed through structured training or post-training compression. They can serve as efficient alternatives to LLMs for reasoning-intensive tasks. 

**Abstract (ZH)**: 推理能力长期以来被认为是大规模语言模型（LLMs）的一个涌现属性，在或超过一定规模（约1000亿参数）时才出现。然而，最近的研究挑战了这一假设，表明小型语言模型（SLMs）也可以达到竞争力的推理性能。SLMs因其高效性和部署性而日益受到青睐。然而，关于不同类型的SLMs（包括从零开始训练的模型和通过量化、剪枝和蒸馏从LLMs派生的模型）的推理能力系统的研究仍然不足。这引发了一个关键问题：SLMs能否达到与LLMs相媲美的推理能力？在这项研究中，我们系统地调研、基准测试并分析了六个模型家族的72个SLMs在14个推理基准上的表现。为确保评估的可靠性，我们检查了四种评估方法，并将四种LLM评判员与人类评价进行了比较，共涵盖了800个数据点。我们重复所有实验三次，以确保进行稳健的性能评估。此外，我们还分析了不同提示策略在小型模型中的影响。除了准确性之外，我们还评估了模型在对抗条件下的鲁棒性以及中间推理步骤的表现。我们的研究结果挑战了“只靠增加规模才能获得强大的推理能力”这一假设。相反，我们预见未来可以通过结构化训练或后训练压缩来开发具有强大推理能力的SLMs。它们可以作为LLMs的有效替代品，用于推理密集型任务。 

---
# Leader and Follower: Interactive Motion Generation under Trajectory Constraints 

**Title (ZH)**: 领导者与跟随者：在轨迹约束下的交互式运动生成 

**Authors**: Runqi Wang, Caoyuan Ma, Jian Zhao, Hanrui Xu, Dongfang Sun, Haoyang Chen, Lin Xiong, Zheng Wang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.11563)  

**Abstract**: With the rapid advancement of game and film production, generating interactive motion from texts has garnered significant attention due to its potential to revolutionize content creation processes. In many practical applications, there is a need to impose strict constraints on the motion range or trajectory of virtual characters. However, existing methods that rely solely on textual input face substantial challenges in accurately capturing the user's intent, particularly in specifying the desired trajectory. As a result, the generated motions often lack plausibility and accuracy. Moreover, existing trajectory - based methods for customized motion generation rely on retraining for single - actor scenarios, which limits flexibility and adaptability to different datasets, as well as interactivity in two-actor motions. To generate interactive motion following specified trajectories, this paper decouples complex motion into a Leader - Follower dynamic, inspired by role allocation in partner dancing. Based on this framework, this paper explores the motion range refinement process in interactive motion generation and proposes a training-free approach, integrating a Pace Controller and a Kinematic Synchronization Adapter. The framework enhances the ability of existing models to generate motion that adheres to trajectory by controlling the leader's movement and correcting the follower's motion to align with the leader. Experimental results show that the proposed approach, by better leveraging trajectory information, outperforms existing methods in both realism and accuracy. 

**Abstract (ZH)**: 随着游戏和电影制作的迅速发展，从文本生成互动动作已经成为一个备受关注的研究领域，因为它有可能彻底改变内容创作流程。在许多实际应用中，需要对虚拟角色的动作范围或轨迹施加严格的限制。然而，现有的仅依赖文本输入的方法在准确捕捉用户意图，特别是在指定期望轨迹方面面临重大挑战。因此，生成的动作往往缺乏合理性和准确性。此外，现有的基于轨迹的方法在为单演员场景定制动作生成时依赖重新训练，这限制了其对不同数据集的灵活性和适应性，以及双演员动作的互动性。为了遵循指定轨迹生成互动动作，本文借鉴了伴侣舞蹈中角色分配的概念，将复杂动作分解为领导者（Leader）和跟随者（Follower）的动力学。基于这种框架，本文探讨了交互动作生成中的动作范围细化过程，并提出了一种无需训练的方法，该方法结合了节奏控制器（Pace Controller）和运动同步适配器（Kinematic Synchronization Adapter）。该框架通过控制领导者运动并纠正跟随者运动以与领导者对齐，增强了现有模型生成遵循轨迹动作的能力。实验结果表明，通过更好地利用轨迹信息，所提出的方法在现实性和准确性方面均优于现有方法。 

---
# Auto-Search and Refinement: An Automated Framework for Gender Bias Mitigation in Large Language Models 

**Title (ZH)**: 自动搜索与精炼：一种针对大规模语言模型性别偏见缓解的自动化框架 

**Authors**: Yue Xu, Chengyan Fu, Li Xiong, Sibei Yang, Wenjie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11559)  

**Abstract**: Pre-training large language models (LLMs) on vast text corpora enhances natural language processing capabilities but risks encoding social biases, particularly gender bias. While parameter-modification methods like fine-tuning mitigate bias, they are resource-intensive, unsuitable for closed-source models, and lack adaptability to evolving societal norms. Instruction-based approaches offer flexibility but often compromise task performance. To address these limitations, we propose $\textit{FaIRMaker}$, an automated and model-independent framework that employs an $\textbf{auto-search and refinement}$ paradigm to adaptively generate Fairwords, which act as instructions integrated into input queries to reduce gender bias and enhance response quality. Extensive experiments demonstrate that $\textit{FaIRMaker}$ automatically searches for and dynamically refines Fairwords, effectively mitigating gender bias while preserving task integrity and ensuring compatibility with both API-based and open-source LLMs. 

**Abstract (ZH)**: 预训练大规模语言模型（LLMs）在海量文本语料上的训练可以增强自然语言处理能力，但可能会编码社会偏见，特别是性别偏见。虽然参数修改方法如微调可以缓解偏见，但这些方法资源密集型且不适用于闭源模型，缺乏适应不断变化的社会规范的能力。基于指令的方法具有灵活性，但通常会牺牲任务性能。为解决这些限制，我们提出了一种名为$\textit{FaIRMaker}$的自动化且模型无关框架，采用了$\textbf{自动搜索和精炼}$的范式，自适应生成公平词（Fairwords），这些公平词作为指令集成到输入查询中，以减少性别偏见并提高响应质量。大量实验表明，$\textit{FaIRMaker}$能够自动搜索并动态精炼公平词，有效缓解性别偏见，同时保持任务完整性，并确保与基于API和开源LLMs的兼容性。 

---
# Toward Metaphor-Fluid Conversation Design for Voice User Interfaces 

**Title (ZH)**: 面向语音用户界面的隐喻流动对话设计 

**Authors**: Smit Desai, Jessie Chin, Dakuo Wang, Benjamin Cowan, Michael Twidale  

**Link**: [PDF](https://arxiv.org/pdf/2502.11554)  

**Abstract**: Metaphors play a critical role in shaping user experiences with Voice User Interfaces (VUIs), yet existing designs often rely on static, human-centric metaphors that fail to adapt to diverse contexts and user needs. This paper introduces Metaphor-Fluid Design, a novel approach that dynamically adjusts metaphorical representations based on conversational use-contexts. We compare this approach to a Default VUI, which characterizes the present implementation of commercial VUIs commonly designed around the persona of an assistant, offering a uniform interaction style across contexts. In Study 1 (N=130), metaphors were mapped to four key use-contexts-commands, information seeking, sociality, and error recovery-along the dimensions of formality and hierarchy, revealing distinct preferences for task-specific metaphorical designs. Study 2 (N=91) evaluates a Metaphor-Fluid VUI against a Default VUI, showing that the Metaphor-Fluid VUI enhances perceived intention to adopt, enjoyment, and likability by aligning better with user expectations for different contexts. However, individual differences in metaphor preferences highlight the need for personalization. These findings challenge the one-size-fits-all paradigm of VUI design and demonstrate the potential of Metaphor-Fluid Design to create more adaptive and engaging human-AI interactions. 

**Abstract (ZH)**: 元喻在塑造语音用户界面（VUI）用户体验中起着关键作用，然而现有的设计往往依赖于静态、以人为中心的元喻，这些元喻无法适应多样化的使用情境和用户需求。本文介绍了一种新颖的方法——动态元喻设计（Metaphor-Fluid Design），该方法根据对话使用情境动态调整元喻表示。本文将这种设计方法与默认VUI进行了比较，后者是目前商业VUI的常见设计，通常围绕助手型人格设定，提供统一的交互风格。在研究1（N=130）中，元喻被映射到四个关键使用情境——命令、信息查询、社交性和错误恢复——的正式程度和层次结构维度上，揭示了针对特定任务的元喻设计的不同偏好。研究2（N=91）评估了动态元喻VUI与默认VUI在不同情境下用户期望的符合程度，结果表明动态元喻VUI提高了用户采用意愿、愉悦感和好感度，但个体元喻偏好差异也指出了个性化的需求。这些发现挑战了VUI设计的“一刀切”范式，并展示了动态元喻设计在创造更具适应性和互动性的用户-人工智能交互方面的潜力。 

---
# MuSC: Improving Complex Instruction Following with Multi-granularity Self-Contrastive Training 

**Title (ZH)**: MuSC：通过多粒度自对比训练改进复杂指令跟随 

**Authors**: Hui Huang, Jiaheng Liu, Yancheng He, Shilong Li, Bing Xu, Conghui Zhu, Muyun Yang, Tiejun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.11541)  

**Abstract**: Complex instruction-following with elaborate constraints is imperative for Large Language Models (LLMs). While existing methods have constructed data for complex instruction alignment, they all rely on a more advanced model, especially GPT-4, limiting their application. In this paper, we propose a Multi-granularity Self-Contrastive Training (MuSC) framework, to improve the complex instruction alignment without relying on a stronger model. Our method is conducted on both coarse and fine granularity. On coarse-granularity, we construct constraint-aware preference data based on instruction decomposition and recombination. On fine-granularity, we perform token-aware preference optimization with dynamic token-level supervision. Our method is evaluated on open-sourced models, and experiment results show our method achieves significant improvement on both complex and general instruction-following benchmarks, surpassing previous self-alignment methods. 

**Abstract (ZH)**: 对于大型语言模型（LLMs），执行复杂指令并伴有详细约束是必不可少的。虽然现有方法已经构建了用于复杂指令对齐的数据，但它们都依赖于更为先进的模型，尤其是GPT-4，这限制了它们的应用范围。本文中，我们提出了一种多层次自我对比训练（MuSC）框架，以提高复杂指令对齐的能力，而不依赖于更强的模型。我们的方法在粗粒度和细粒度两个级别上进行。

在粗粒度级别上，我们基于指令分解和重组构建了具有约束感知的偏好数据。在细粒度级别上，我们进行了具有动态词级监督的感知 token 偏好优化。我们的方法在开源模型上进行了评估，实验结果显示，我们的方法在复杂和通用指令执行基准上均取得了显著改进，超越了之前的自我对齐方法。 

---
# $\text{M}^{\text{3}}$: A Modular World Model over Streams of Tokens 

**Title (ZH)**: $\text{M}^{\text{3}}$: 基于令牌流的模块化世界模型 

**Authors**: Lior Cohen, Kaixin Wang, Bingyi Kang, Uri Gadot, Shie Mannor  

**Link**: [PDF](https://arxiv.org/pdf/2502.11537)  

**Abstract**: Token-based world models emerged as a promising modular framework, modeling dynamics over token streams while optimizing tokenization separately. While successful in visual environments with discrete actions (e.g., Atari games), their broader applicability remains uncertain. In this paper, we introduce $\text{M}^{\text{3}}$, a $\textbf{m}$odular $\textbf{w}$orld $\textbf{m}$odel that extends this framework, enabling flexible combinations of observation and action modalities through independent modality-specific components. $\text{M}^{\text{3}}$ integrates several improvements from existing literature to enhance agent performance. Through extensive empirical evaluation across diverse benchmarks, $\text{M}^{\text{3}}$ achieves state-of-the-art sample efficiency for planning-free world models. Notably, among these methods, it is the first to reach a human-level median score on Atari 100K, with superhuman performance on 13 games. We $\href{this https URL}{\text{open-source our code and weights}}$. 

**Abstract (ZH)**: 基于token的场景模型作为一种有前景的模块化框架，能够通过独立优化token化来模型化token流中的动力学。虽然在具有离散动作的视觉环境中（例如，Atari游戏）取得了成功，但其更广泛的应用前景仍然不明确。在本文中，我们引入了**M³**，一种**模**块**化的**场**景模**型，通过独立的模态特定组件，扩展了这一框架，以实现多种观测和动作模态的灵活组合。M³整合了现有文献中的多项改进，以增强代理性能。通过在多种基准上的广泛实证评估，M³在不需要规划的世界模型中达到了最先进的样本效率。值得注意的是，在这些方法中，M³首次在Atari 100K中达到了人类级别的中位得分，并在13个游戏中表现出了超人类的性能。我们已经将我们的**代码和权重公开发布**。 

---
# DeFiScope: Detecting Various DeFi Price Manipulations with LLM Reasoning 

**Title (ZH)**: DeFiScope：利用大语言模型推理检测各种DeFi价格操纵 

**Authors**: Juantao Zhong, Daoyuan Wu, Ye Liu, Maoyi Xie, Yang Liu, Yi Li, Ning Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11521)  

**Abstract**: DeFi (Decentralized Finance) is one of the most important applications of today's cryptocurrencies and smart contracts. It manages hundreds of billions in Total Value Locked (TVL) on-chain, yet it remains susceptible to common DeFi price manipulation attacks. Despite state-of-the-art (SOTA) systems like DeFiRanger and DeFort, we found that they are less effective to non-standard price models in custom DeFi protocols, which account for 44.2% of the 95 DeFi price manipulation attacks reported over the past three years.
In this paper, we introduce the first LLM-based approach, DeFiScope, for detecting DeFi price manipulation attacks in both standard and custom price models. Our insight is that large language models (LLMs) have certain intelligence to abstract price calculation from code and infer the trend of token price changes based on the extracted price models. To further strengthen LLMs in this aspect, we leverage Foundry to synthesize on-chain data and use it to fine-tune a DeFi price-specific LLM. Together with the high-level DeFi operations recovered from low-level transaction data, DeFiScope detects various DeFi price manipulations according to systematically mined patterns. Experimental results show that DeFiScope achieves a high precision of 96% and a recall rate of 80%, significantly outperforming SOTA approaches. Moreover, we evaluate DeFiScope's cost-effectiveness and demonstrate its practicality by helping our industry partner confirm 147 real-world price manipulation attacks, including discovering 81 previously unknown historical incidents. 

**Abstract (ZH)**: 去中心化金融（DeFi, Decentralized Finance）是当今加密货币和智能合约的重要应用之一。它在链上管理着数百亿美元的总资产（TVL），然而仍然容易受到常见的DeFi价格操控攻击。尽管有最先进的（SOTA）系统如DeFiRanger和DeFort，我们发现它们在处理定制DeFi协议中的非标准价格模型时效果较差，而这些定制协议在过去三年内报告的95起DeFi价格操控攻击中占44.2%。

在本文中，我们提出了首个基于大语言模型（LLM, Large Language Model）的方法DeFiScope，用于检测标准和定制价格模型中的DeFi价格操控攻击。我们的见解在于，大语言模型具有一定的智能，能够从代码中抽象出价格计算，并根据提取的价格模型推断代币价格变化的趋势。为了进一步增强大语言模型在这方面的能力，我们利用Foundry合成了链上数据，并利用这些数据对特定于DeFi价格的大语言模型进行了微调。结合从低级别交易数据中恢复的高阶DeFi操作，DeFiScope根据系统挖掘出的模式检测各种DeFi价格操控。实验结果表明，DeFiScope的精度达到96%，召回率达到80%，显著优于SOTA方法。此外，我们评估了DeFiScope的成本效益，并通过帮助行业合作伙伴确认147起真实的DeFi价格操控攻击（其中包括发现81起以前未知的事件）展示了其实用性。 

---
# UniGO: A Unified Graph Neural Network for Modeling Opinion Dynamics on Graphs 

**Title (ZH)**: UniGO：用于图上意见动力学建模的统一图神经网络 

**Authors**: Hao Li, Hao Jiang, Yuke Zheng, Hao Sun, Wenying Gong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11519)  

**Abstract**: Polarization and fragmentation in social media amplify user biases, making it increasingly important to understand the evolution of opinions. Opinion dynamics provide interpretability for studying opinion evolution, yet incorporating these insights into predictive models remains challenging. This challenge arises due to the inherent complexity of the diversity of opinion fusion rules and the difficulty in capturing equilibrium states while avoiding over-smoothing. This paper constructs a unified opinion dynamics model to integrate different opinion fusion rules and generates corresponding synthetic datasets. To fully leverage the advantages of unified opinion dynamics, we introduces UniGO, a framework for modeling opinion evolution on graphs. Using a coarsen-refine mechanism, UniGO efficiently models opinion dynamics through a graph neural network, mitigating over-smoothing while preserving equilibrium phenomena. UniGO leverages pretraining on synthetic datasets, which enhances its ability to generalize to real-world scenarios, providing a viable paradigm for applications of opinion dynamics. Experimental results on both synthetic and real-world datasets demonstrate UniGO's effectiveness in capturing complex opinion formation processes and predicting future evolution. The pretrained model also shows strong generalization capability, validating the benefits of using synthetic data to boost real-world performance. 

**Abstract (ZH)**: 社交媒体中的极化和碎片化放大了用户偏见，使得理解意见演变的进化变得愈发重要。意见动力学为研究意见演变提供了可解释性，但将这些见解融入预测模型仍然存在挑战。这一挑战源于不同意见融合规则的多样性固有的复杂性，以及捕捉平衡状态的同时避免过度平滑的困难。本文构建了一个统一的意见动力学模型，以整合不同的意见融合规则，并生成相应的合成数据集。为了充分利用统一意见动力学的优势，我们提出了UniGO框架，用于图上的意见演变建模。通过粗化-细化机制，UniGO利用图神经网络高效地建模意见动力学，减轻过度平滑现象，同时保留平衡现象。UniGO通过在合成数据集上的预训练提高了其泛化能力，为意见动力学的应用提供了可行的范式。在合成数据集和真实世界数据集上的实验结果表明，UniGO在捕捉复杂的意见形成过程和预测未来发展方面表现出有效性。预训练模型还展示了强大的泛化能力，验证了使用合成数据提升真实世界性能的好处。 

---
# Generative Multi-Agent Collaboration in Embodied AI: A Systematic Review 

**Title (ZH)**: 具身人工智能中生成型多智能体协作：一项系统性回顾 

**Authors**: Di Wu, Xian Wei, Guang Chen, Hao Shen, Xiangfeng Wang, Wenhao Li, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.11518)  

**Abstract**: Embodied multi-agent systems (EMAS) have attracted growing attention for their potential to address complex, real-world challenges in areas such as logistics and robotics. Recent advances in foundation models pave the way for generative agents capable of richer communication and adaptive problem-solving. This survey provides a systematic examination of how EMAS can benefit from these generative capabilities. We propose a taxonomy that categorizes EMAS by system architectures and embodiment modalities, emphasizing how collaboration spans both physical and virtual contexts. Central building blocks, perception, planning, communication, and feedback, are then analyzed to illustrate how generative techniques bolster system robustness and flexibility. Through concrete examples, we demonstrate the transformative effects of integrating foundation models into embodied, multi-agent frameworks. Finally, we discuss challenges and future directions, underlining the significant promise of EMAS to reshape the landscape of AI-driven collaboration. 

**Abstract (ZH)**: 具身多智能体系统（EMAS）因其在物流、机器人等领域解决复杂现实问题的潜力而日益受到关注。近期基础模型技术的发展为生成型智能体提供了可能性，这些智能体能够进行更为丰富的通信和适应性问题解决。本文综述了EMAS可以从这些生成型能力中获得的好处，并提供了一种分类体系，将EMAS按系统架构和具身模态分类，强调合作如何跨越物理和虚拟环境。接着，分析了核心组件、感知、规划、通信和反馈，以说明生成技术如何增强系统的稳健性和灵活性。通过具体示例，展示了将基础模型集成到具身多智能体框架中的变革性影响。最后，讨论了挑战和未来方向，强调了EMAS对重塑AI驱动合作格局的巨大潜力。 

---
# MaZO: Masked Zeroth-Order Optimization for Multi-Task Fine-Tuning of Large Language Models 

**Title (ZH)**: MaZO：多任务微调大规模语言模型的掩蔽零阶优化 

**Authors**: Zhen Zhang, Yifan Yang, Kai Zhen, Nathan Susanj, Athanasios Mouchtaris, Siegfried Kunzmann, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11513)  

**Abstract**: Large language models have demonstrated exceptional capabilities across diverse tasks, but their fine-tuning demands significant memory, posing challenges for resource-constrained environments. Zeroth-order (ZO) optimization provides a memory-efficient alternative by eliminating the need for backpropagation. However, ZO optimization suffers from high gradient variance, and prior research has largely focused on single-task learning, leaving its application to multi-task learning unexplored. Multi-task learning is crucial for leveraging shared knowledge across tasks to improve generalization, yet it introduces unique challenges under ZO settings, such as amplified gradient variance and collinearity. In this paper, we present MaZO, the first framework specifically designed for multi-task LLM fine-tuning under ZO optimization. MaZO tackles these challenges at the parameter level through two key innovations: a weight importance metric to identify critical parameters and a multi-task weight update mask to selectively update these parameters, reducing the dimensionality of the parameter space and mitigating task conflicts. Experiments demonstrate that MaZO achieves state-of-the-art performance, surpassing even multi-task learning methods designed for first-order optimization. 

**Abstract (ZH)**: 大型语言模型在各种任务中展现出了卓越的能力，但在微调过程中需要大量的内存，这在资源受限的环境中构成了挑战。零阶（Zeroth-order, ZO）优化提供了一种内存高效的替代方案，通过消除反向传播的需求来实现这一点。然而，ZO优化面临着高梯度方差的问题，现有的研究主要集中在单任务学习上，而多任务学习的应用尚未被充分探索。多任务学习对于利用跨任务的知识以提高泛化能力至关重要，但在ZO环境中也带来了独特的挑战，如梯度方差的放大和共线性问题。本文我们提出了MaZO，这是第一个专门为ZO优化下的多任务大型语言模型微调设计的框架。MaZO通过两个关键创新在参数层面克服了这些挑战：一个权重重要性度量来识别关键参数，以及一个基于任务的权重更新掩码以选择性地更新这些参数，从而减少参数空间维度并缓解任务冲突。实验结果表明，MaZO达到了最先进的性能，甚至超过了专门为一阶优化设计的多任务学习方法。 

---
# DifCluE: Generating Counterfactual Explanations with Diffusion Autoencoders and modal clustering 

**Title (ZH)**: DifCluE：基于扩散自编码器和模态聚类生成反事实解释 

**Authors**: Suparshva Jain, Amit Sangroya, Lovekesh Vig  

**Link**: [PDF](https://arxiv.org/pdf/2502.11509)  

**Abstract**: Generating multiple counterfactual explanations for different modes within a class presents a significant challenge, as these modes are distinct yet converge under the same classification. Diffusion probabilistic models (DPMs) have demonstrated a strong ability to capture the underlying modes of data distributions. In this paper, we harness the power of a Diffusion Autoencoder to generate multiple distinct counterfactual explanations. By clustering in the latent space, we uncover the directions corresponding to the different modes within a class, enabling the generation of diverse and meaningful counterfactuals. We introduce a novel methodology, DifCluE, which consistently identifies these modes and produces more reliable counterfactual explanations. Our experimental results demonstrate that DifCluE outperforms the current state-of-the-art in generating multiple counterfactual explanations, offering a significant advance- ment in model interpretability. 

**Abstract (ZH)**: 在同一个类别内为不同的模式生成多个反事实解释是一个显著的挑战，因为这些模式虽然独立但共存于同一分类之下。扩散概率模型（DPMs）展示了很强的能力来捕捉数据分布的潜在模式。本文利用扩散自动编码器（Diffusion Autoencoder）生成多个独特的反事实解释。通过在潜在空间中聚类，我们揭示了与类别中不同模式对应的潜在方向，从而能够生成多样且有意义的反事实解释。我们提出了一种新颖的方法论——DifCluE，它可以一致地识别这些模式并生成更可靠的反事实解释。实验结果表明，DifCluE 在生成多个反事实解释方面优于当前最先进的方法，显著提升了模型的可解释性。 

---
# Chinese Spelling Correction: A Comprehensive Survey of Progress, Challenges, and Opportunities 

**Title (ZH)**: 中文翻译如下，符合学术规范：

中文标题：中文拼写修正：进展、挑战与机遇综述

此标题简洁明了地传达了原文的意思，适合用于学术论文的标题。 

**Authors**: Changchun Liu, Kai Zhang, Junzhe Jiang, Zixiao Kong, Qi Liu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11508)  

**Abstract**: Chinese Spelling Correction (CSC) is a critical task in natural language processing, aimed at detecting and correcting spelling errors in Chinese text. This survey provides a comprehensive overview of CSC, tracing its evolution from pre-trained language models to large language models, and critically analyzing their respective strengths and weaknesses in this domain. Moreover, we further present a detailed examination of existing benchmark datasets, highlighting their inherent challenges and limitations. Finally, we propose promising future research directions, particularly focusing on leveraging the potential of LLMs and their reasoning capabilities for improved CSC performance. To the best of our knowledge, this is the first comprehensive survey dedicated to the field of CSC. We believe this work will serve as a valuable resource for researchers, fostering a deeper understanding of the field and inspiring future advancements. 

**Abstract (ZH)**: 中文翻译如下，符合学术规范：

中文拼写校正（CSC）是自然语言处理中的关键任务，旨在检测和纠正中文文本中的拼写错误。本文综述了CSC的发展历程，从预训练语言模型到大型语言模型，并对其在该领域的优势和劣势进行了批判性分析。此外，我们还详细探讨了现有的基准数据集，突出了它们的固有挑战和局限性。最后，我们提出了有前景的未来研究方向，特别强调利用大型语言模型及其推理能力来提高CSC性能。据我们所知，这是首次全面综述CSC领域的研究。我们相信，这项工作将成为研究人员的重要资源，促进对这一领域的更深入理解，并激发未来的进步。 

---
# Accelerated Gradient-based Design Optimization Via Differentiable Physics-Informed Neural Operator: A Composites Autoclave Processing Case Study 

**Title (ZH)**: 基于可微物理信息神经算子的加速梯度设计优化：以复合材料固化处理案例研究为例 

**Authors**: Janak M. Patel, Milad Ramezankhani, Anirudh Deodhar, Dagnachew Birru  

**Link**: [PDF](https://arxiv.org/pdf/2502.11504)  

**Abstract**: Simulation and optimization are crucial for advancing the engineering design of complex systems and processes. Traditional optimization methods require substantial computational time and effort due to their reliance on resource-intensive simulations, such as finite element analysis, and the complexity of rigorous optimization algorithms. Data-agnostic AI-based surrogate models, such as Physics-Informed Neural Operators (PINOs), offer a promising alternative to these conventional simulations, providing drastically reduced inference time, unparalleled data efficiency, and zero-shot super-resolution capability. However, the predictive accuracy of these models is often constrained to small, low-dimensional design spaces or systems with relatively simple dynamics. To address this, we introduce a novel Physics-Informed DeepONet (PIDON) architecture, which extends the capabilities of conventional neural operators to effectively model the nonlinear behavior of complex engineering systems across high-dimensional design spaces and a wide range of dynamic design configurations. This new architecture outperforms existing SOTA models, enabling better predictions across broader design spaces. Leveraging PIDON's differentiability, we integrate a gradient-based optimization approach using the Adam optimizer to efficiently determine optimal design variables. This forms an end-to-end gradient-based optimization framework that accelerates the design process while enhancing scalability and efficiency. We demonstrate the effectiveness of this framework in the optimization of aerospace-grade composites curing processes achieving a 3x speedup in obtaining optimal design variables compared to gradient-free methods. Beyond composites processing, the proposed model has the potential to be used as a scalable and efficient optimization tool for broader applications in advanced engineering and digital twin systems. 

**Abstract (ZH)**: 模拟和优化对于推进复杂系统和过程的工程设计至关重要。传统的优化方法需要大量计算时间和资源，因为它们依赖于资源密集型的仿真，如有限元分析，以及严格的优化算法的复杂性。无数据依赖的人工智能代理模型，如物理知情神经算子（PINOs），为这些传统仿真提供了有希望的替代方案，提供了大幅度减少推理时间、无与伦比的数据效率以及零样本超分辨率能力。然而，这些模型的预测准确性常常受到制衡，仅在小规模低维设计空间或相对简单动力学的系统中有保证。为了解决这一问题，我们引入了一种新型的物理知情深度算子网络（PIDON）架构，它扩展了传统神经算子的功能，能够有效模拟复杂工程系统在高维设计空间和广泛的动态设计配置下的非线性行为。这个新架构优于现有最先进的模型，能够在更广泛的优化设计空间中提供更准确的预测。利用PIDON的可微性，我们将其与基于梯度的优化方法相结合，使用Adam优化器高效地确定最优设计变量。这形成了一种端到端的基于梯度的优化框架，加速了设计过程，增强了可扩展性和效率。我们通过在航空航天级复合材料固化过程的优化中实现3倍的速度提升，证明了该框架的有效性。除了复合材料处理，所提出模型还有潜力作为在先进工程和数字孪生系统中广泛使用的可扩展和高效的优化工具。 

---
# Ontology-Guided Reverse Thinking Makes Large Language Models Stronger on Knowledge Graph Question Answering 

**Title (ZH)**: 基于本体引导的逆向思维使大型语言模型在知识图谱问答任务中更加强大 

**Authors**: Runxuan Liu, Bei Luo, Jiaqi Li, Baoxin Wang, Ming Liu, Dayong Wu, Shijin Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2502.11491)  

**Abstract**: Large language models (LLMs) have shown remarkable capabilities in natural language processing. However, in knowledge graph question answering tasks (KGQA), there remains the issue of answering questions that require multi-hop reasoning. Existing methods rely on entity vector matching, but the purpose of the question is abstract and difficult to match with specific entities. As a result, it is difficult to establish reasoning paths to the purpose, which leads to information loss and redundancy. To address this issue, inspired by human reverse thinking, we propose Ontology-Guided Reverse Thinking (ORT), a novel framework that constructs reasoning paths from purposes back to conditions. ORT operates in three key phases: (1) using LLM to extract purpose labels and condition labels, (2) constructing label reasoning paths based on the KG ontology, and (3) using the label reasoning paths to guide knowledge retrieval. Experiments on the WebQSP and CWQ datasets show that ORT achieves state-of-the-art performance and significantly enhances the capability of LLMs for KGQA. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理方面展现出了显著的能力。然而，在知识图谱问答任务（KGQA）中，仍然存在需要多跳推理来回答问题的问题。现有方法依赖于实体向量匹配，但问题的目的往往是抽象的，难以与特定实体匹配，因此难以建立从目的到条件的推理路径，导致信息丢失和冗余。为了解决这个问题，我们受到人类逆向思维的启发，提出了一种新的框架——本体引导的逆向思维（Ontology-Guided Reverse Thinking, ORT），该框架从目的出发，构建回溯到条件的推理路径。ORT 主要包含三个关键阶段：（1）使用LLM提取目的标签和条件标签，（2）基于知识图谱本体构建标签推理路径，（3）使用标签推理路径指导知识检索。在WebQSP和CWQ数据集上的实验表明，ORT 达到了最先进的性能，并显著增强了LLMs在KGQA中的能力。 

---
# DATA: Decomposed Attention-based Task Adaptation for Rehearsal-Free Continual Learning 

**Title (ZH)**: 标题：分解注意力机制的任务自适应方法——实现无回顾式持续学习 

**Authors**: Huanxuan Liao, Shizhu He, Yupu Hao, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11482)  

**Abstract**: Continual learning (CL) is essential for Large Language Models (LLMs) to adapt to evolving real-world demands, yet they are susceptible to catastrophic forgetting (CF). While traditional CF solutions rely on expensive data rehearsal, recent rehearsal-free methods employ model-based and regularization-based strategies to address this issue. However, these approaches often neglect the model's plasticity, which is crucial to achieving optimal performance on newly learned tasks. Consequently, a key challenge in CL is striking a balance between preserving plasticity and mitigating CF. To tackle this challenge, we propose the $\textbf{D}$ecomposed $\textbf{A}$ttention-based $\textbf{T}$ask $\textbf{A}$daptation (DATA), which explicitly decouples and learns both task-specific and task-shared knowledge using high-rank and low-rank task adapters (e.g., LoRAs). For new tasks, DATA dynamically adjusts the weights of adapters of different ranks based on their relevance and distinction from previous tasks, allowing the model to acquire new task-specific skills while effectively retaining previously learned knowledge. Specifically, we implement a decomposed component weighting strategy comprising learnable components that collectively generate attention-based weights, allowing the model to integrate and utilize diverse knowledge from each DATA. Extensive experiments on three widely used benchmarks demonstrate that our proposed method achieves state-of-the-art performance. Notably, our approach significantly enhances model plasticity and mitigates CF by extending learnable components and employing stochastic restoration during training iterations. 

**Abstract (ZH)**: 持续学习（CL）对于大型语言模型（LLMs）适应不断变化的实际需求是至关重要的，然而它们易发生灾难性遗忘（CF）。传统的CF解决方案依赖于昂贵的数据复习，而近期的无复习方法则通过基于模型和正则化的方法来应对这一问题。然而，这些方法往往忽略了模型的可塑性，这对在新任务上实现最佳性能至关重要。因此，在CL中一个关键挑战在于平衡保持可塑性和减轻CF之间的关系。为了解决这一挑战，我们提出了**分解注意基任务适配（Decomposed Attention-based Task Adaptation，简称DATA）**方法，该方法明确地分离和学习了特定任务和共享任务的知识，使用高秩和低秩的任务适配器（例如LoRAs）。对于新任务，DATA动态调整不同秩的任务适配器的权重，根据它们与前序任务的相关性和差异性，使模型能够在有效保留原有知识的同时习得新的任务特定技能。具体而言，我们实现了一个分解组件加权策略，其中包括可学习组件，共同生成注意基权重，从而使模型能够整合并利用每个DATA中的多样化知识。在三个广泛使用的基准上的大量实验表明，我们提出的方法达到了最先进的性能。值得注意的是，我们的方法通过扩展可学习组件并在训练迭代中采用随机恢复显著增强了模型的可塑性，并减轻了CF。 

---
# Variable-frame CNNLSTM for Breast Nodule Classification using Ultrasound Videos 

**Title (ZH)**: 基于超声视频的乳腺结节分类的可变帧CNN-LSTM模型 

**Authors**: Xiangxiang Cui, Zhongyu Li, Xiayue Fan, Peng Huang, Ying Wang, Meng Yang, Shi Chang, Jihua Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11481)  

**Abstract**: The intersection of medical imaging and artificial intelligence has become an important research direction in intelligent medical treatment, particularly in the analysis of medical images using deep learning for clinical diagnosis. Despite the advances, existing keyframe classification methods lack extraction of time series features, while ultrasonic video classification based on three-dimensional convolution requires uniform frame numbers across patients, resulting in poor feature extraction efficiency and model classification performance. This study proposes a novel video classification method based on CNN and LSTM, introducing NLP's long and short sentence processing scheme into video classification for the first time. The method reduces CNN-extracted image features to 1x512 dimension, followed by sorting and compressing feature vectors for LSTM training. Specifically, feature vectors are sorted by patient video frame numbers and populated with padding value 0 to form variable batches, with invalid padding values compressed before LSTM training to conserve computing resources. Experimental results demonstrate that our variable-frame CNNLSTM method outperforms other approaches across all metrics, showing improvements of 3-6% in F1 score and 1.5% in specificity compared to keyframe methods. The variable-frame CNNLSTM also achieves better accuracy and precision than equal-frame CNNLSTM. These findings validate the effectiveness of our approach in classifying variable-frame ultrasound videos and suggest potential applications in other medical imaging modalities. 

**Abstract (ZH)**: 医学成像与人工智能的交汇已经成为智能医疗治疗中的一个重要研究方向，特别是在利用深度学习进行医学图像分析以实现临床诊断方面。尽管取得了进展，现有的关键帧分类方法没有提取时间序列特征，而基于三维卷积的超声视频分类则要求患者之间帧数一致，这导致了特征提取效率低下和模型分类性能不佳。本研究提出了一种基于CNN和LSTM的新视频分类方法，首次将NLP中的长句和短句处理方案引入到视频分类中。该方法将CNN提取的图像特征维度降低至1x512，并对特征向量进行排序和压缩，以便进行LSTM训练。具体来说，特征向量按照患者视频帧号排序，并填充0值以形成变长批次，在进行LSTM训练前压缩无效填充值以节省计算资源。实验结果表明，我们的变帧CNN-LSTM方法在所有指标上均优于其他方法，与关键帧方法相比，F1分数和特异性分别提高了3-6%和1.5%。变帧CNN-LSTM方法在准确性和精度方面也优于等帧CNN-LSTM方法。这些发现验证了我们方法在分类变帧超声视频中的有效性，并暗示了其在其他医学成像模态中的潜在应用。 

---
# Optimized detection of cyber-attacks on IoT networks via hybrid deep learning models 

**Title (ZH)**: 基于混合深度学习模型的物联网网络 cyber-攻击检测优化方法 

**Authors**: Ahmed Bensaoud, Jugal Kalita  

**Link**: [PDF](https://arxiv.org/pdf/2502.11470)  

**Abstract**: The rapid expansion of Internet of Things (IoT) devices has increased the risk of cyber-attacks, making effective detection essential for securing IoT networks. This work introduces a novel approach combining Self-Organizing Maps (SOMs), Deep Belief Networks (DBNs), and Autoencoders to detect known and previously unseen attack patterns. A comprehensive evaluation using simulated and real-world traffic data is conducted, with models optimized via Particle Swarm Optimization (PSO). The system achieves an accuracy of up to 99.99% and Matthews Correlation Coefficient (MCC) values exceeding 99.50%. Experiments on NSL-KDD, UNSW-NB15, and CICIoT2023 confirm the model's strong performance across diverse attack types. These findings suggest that the proposed method enhances IoT security by identifying emerging threats and adapting to evolving attack strategies. 

**Abstract (ZH)**: 物联网（IoT）设备的迅速扩张增加了网络攻击的风险，因此有效检测对于保障物联网网络的安全至关重要。本文提出了一种结合自组织映射（SOMs）、深层信念网络（DBNs）和自动编码器的新方法，以检测已知和未知的攻击模式。通过使用模拟和实际网络流量数据进行全面评估，并通过粒子 swarm 优化（PSO）优化模型，该系统实现了高达99.99%的准确率和超过99.50%的马修斯相关系数（MCC）。NSL-KDD、UNSW-NB15和CICIoT2023的数据集实验结果表明，该模型在多种攻击类型上表现出色。这些发现表明，所提出的方法通过识别新兴威胁并适应不断演变的攻击策略，提高了物联网的安全性。 

---
# Towards Efficient Pre-training: Exploring FP4 Precision in Large Language Models 

**Title (ZH)**: 向量高效预训练：探索大语言模型中的FP4精度 

**Authors**: Jiecheng Zhou, Ding Tang, Rong Fu, Boni Hu, Haoran Xu, Yi Wang, Zhilin Pei, Zhongling Su, Liang Liu, Xingcheng Zhang, Weiming Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11458)  

**Abstract**: The burgeoning computational demands for training large language models (LLMs) necessitate efficient methods, including quantized training, which leverages low-bit arithmetic operations to reduce costs. While FP8 precision has shown potential, leveraging FP4 remains challenging due to inherent quantization errors and limited representation capability. Based on the Transformer architecture, we present an FP4 training scheme for LLMs, overcoming these obstacles through mixed-precision quantization strategies tailed for different modules and training stages. This allows us to apply the precision level suitable to distinct components within the model, ensuring that multi-head attention and linear layers are handled appropriately. Our pretraining recipe ensures stability in backpropagation by incorporating fine-grained quantization methods with a target precision training schedule. Experimental results demonstrate that our FP4 training scheme achieves accuracy comparable to BF16 and FP8, with smaller theoretical computational cost. With the advent of next-generation hardware supporting FP4, our method sets the foundation for efficient ultra-low precision training. 

**Abstract (ZH)**: 随着训练大规模语言模型（LLMs）所需的计算需求日益增长，迫切需要高效的方法，其中包括量化训练，该方法通过使用低比特位数的算术运算来降低成本。尽管FP8精度显示出了潜力，但在利用FP4方面仍然面临挑战，这是由于固有的量化误差和有限的表示能力。基于Transformer架构，我们提出了一种适用于LLMs的FP4训练方案，通过针对不同模块和训练阶段的混合精度量化策略克服了这些障碍。这种方法允许我们根据不同模型组件适用的精度水平来调整精度，确保多头注意力机制和线性层能够得到适当处理。通过将精细量化方法与目标精度训练计划相结合，我们的预训练方案确保了反向传播过程的稳定性。实验结果表明，我们的FP4训练方案在理论计算成本更低的情况下，能够达到与BF16和FP8相当的准确性。在适用于FP4的下一代硬件即将问世的情况下，我们的方法为高效超低精度训练奠定了基础。 

---
# Aligning Sentence Simplification with ESL Learner's Proficiency for Language Acquisition 

**Title (ZH)**: 将句子简化与 ESL 学员的语言水平进行对齐以促进语言习得 

**Authors**: Guanlin Li, Yuki Arase, Noel Crespi  

**Link**: [PDF](https://arxiv.org/pdf/2502.11457)  

**Abstract**: Text simplification is crucial for improving accessibility and comprehension for English as a Second Language (ESL) learners. This study goes a step further and aims to facilitate ESL learners' language acquisition by simplification. Specifically, we propose simplifying complex sentences to appropriate levels for learners while also increasing vocabulary coverage of the target level in the simplifications. We achieve this without a parallel corpus by conducting reinforcement learning on a large language model. Our method employs token-level and sentence-level rewards, and iteratively trains the model on its self-generated outputs to guide the model to search for simplification hypotheses that satisfy the target attributes. Experiment results on CEFR-SP and TurkCorpus datasets show that the proposed method can effectively increase the frequency and diversity of vocabulary of the target level by more than $20\%$ compared to baseline models, while maintaining high simplification quality. 

**Abstract (ZH)**: 文本简化对于提高非母语英语（ESL）学习者的可访问性和理解能力至关重要。本研究在此基础上进一步目标，通过简化来促进ESL学习者的语言习得。具体而言，我们提出将复杂句子简化到适合学习者的适当水平，并同时在简化过程中增加目标水平词汇覆盖率。我们通过在大规模语言模型上进行强化学习来实现这一点，而无需平行语料库。该方法使用词级和句级奖励，并通过迭代训练模型使其自动生成的输出来引导模型搜索满足目标属性的简化假设。在CEFR-SP和TurkCorpus数据集上的实验结果表明，与基线模型相比，所提出的方法可以有效提高目标水平词汇的频率和多样性超过20%，同时保持较高的简化质量。 

---
# Leveraging Labelled Data Knowledge: A Cooperative Rectification Learning Network for Semi-supervised 3D Medical Image Segmentation 

**Title (ZH)**: 利用标注数据知识：一种协作校正学习网络在半监督3D医学图像分割中的应用 

**Authors**: Yanyan Wang, Kechen Song, Yuyuan Liu, Shuai Ma, Yunhui Yan, Gustavo Carneiro  

**Link**: [PDF](https://arxiv.org/pdf/2502.11456)  

**Abstract**: Semi-supervised 3D medical image segmentation aims to achieve accurate segmentation using few labelled data and numerous unlabelled data. The main challenge in the design of semi-supervised learning methods consists in the effective use of the unlabelled data for training. A promising solution consists of ensuring consistent predictions across different views of the data, where the efficacy of this strategy depends on the accuracy of the pseudo-labels generated by the model for this consistency learning strategy. In this paper, we introduce a new methodology to produce high-quality pseudo-labels for a consistency learning strategy to address semi-supervised 3D medical image segmentation. The methodology has three important contributions. The first contribution is the Cooperative Rectification Learning Network (CRLN) that learns multiple prototypes per class to be used as external knowledge priors to adaptively rectify pseudo-labels at the voxel level. The second contribution consists of the Dynamic Interaction Module (DIM) to facilitate pairwise and cross-class interactions between prototypes and multi-resolution image features, enabling the production of accurate voxel-level clues for pseudo-label rectification. The third contribution is the Cooperative Positive Supervision (CPS), which optimises uncertain representations to align with unassertive representations of their class distributions, improving the model's accuracy in classifying uncertain regions. Extensive experiments on three public 3D medical segmentation datasets demonstrate the effectiveness and superiority of our semi-supervised learning method. 

**Abstract (ZH)**: 半监督3D医学图像分割旨在使用少量标注数据和大量未标注数据实现精确分割。在半监督学习方法设计中的主要挑战在于有效利用未标注数据进行训练。一种有前途的解决方案是确保数据不同视角下的一致性预测，而这种方法的有效性取决于模型生成的伪标签的质量。本文介绍了一种新的方法以生成高质量的伪标签，用于一致性学习策略，以解决半监督3D医学图像分割问题。该方法具有三个重要贡献：

1. **合作校正学习网络（CRLN）**：CRLN 负责学习每个类别多个原型，用作外部知识先验，以适应性地在校正体素级别的伪标签时发挥作用。

2. **动态交互模块（DIM）**：DIM 用于促进原型之间以及不同类别的原型与多分辨率图像特征之间的交互，从而为伪标签校正生成准确的体素级别线索。

3. **合作正向监督（CPS）**：CPS 优化不确定的表示以与类别分布中的非决断表示对齐，从而提高模型在分类不确定区域时的准确性。

在三个公开的3D医学分割数据集上的大量实验证明了我们半监督学习方法的有效性和优越性。 

---
# Connector-S: A Survey of Connectors in Multi-modal Large Language Models 

**Title (ZH)**: Connector-S：多模态大型语言模型中连接器综述 

**Authors**: Xun Zhu, Zheng Zhang, Xi Chen, Yiming Shi, Miao Li, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11453)  

**Abstract**: With the rapid advancements in multi-modal large language models (MLLMs), connectors play a pivotal role in bridging diverse modalities and enhancing model performance. However, the design and evolution of connectors have not been comprehensively analyzed, leaving gaps in understanding how these components function and hindering the development of more powerful connectors. In this survey, we systematically review the current progress of connectors in MLLMs and present a structured taxonomy that categorizes connectors into atomic operations (mapping, compression, mixture of experts) and holistic designs (multi-layer, multi-encoder, multi-modal scenarios), highlighting their technical contributions and advancements. Furthermore, we discuss several promising research frontiers and challenges, including high-resolution input, dynamic compression, guide information selection, combination strategy, and interpretability. This survey is intended to serve as a foundational reference and a clear roadmap for researchers, providing valuable insights into the design and optimization of next-generation connectors to enhance the performance and adaptability of MLLMs. 

**Abstract (ZH)**: 随着多模态大型语言模型（多模态大语言模型，MM-LLMs）的快速发展，连接器在连接不同模态和提升模型性能方面发挥着关键作用。然而，连接器的设计和演进尚未得到全面分析，这在一定程度上阻碍了我们对这些组件功能的理解，并阻碍了更强大连接器的发展。在本文综述中，我们系统地回顾了当前MM-LLMs中连接器的发展情况，并提出了一个结构化的分类框架，将连接器分为原子操作（映射、压缩、专家混搭）和整体设计（多层、多编码器、多模态场景），突显其技术贡献和进步。此外，我们讨论了几个具有前景的研究前沿和挑战，包括高分辨率输入、动态压缩、指导信息选择、组合策略和可解释性。本文综述旨在为研究人员提供一个基础参考和清晰的路线图，提供有关设计和优化下一代连接器以增强MM-LLMs性能和适应性的宝贵见解。 

---
# Fishing For Cheap And Efficient Pruners At Initialization 

**Title (ZH)**: 在初始化时寻找廉价而高效的剪枝器 

**Authors**: Ivo Gollini Navarrete, Nicolas Mauricio Cuadrado, Jose Renato Restom, Martin Takáč, Samuel Horváth  

**Link**: [PDF](https://arxiv.org/pdf/2502.11450)  

**Abstract**: Pruning offers a promising solution to mitigate the associated costs and environmental impact of deploying large deep neural networks (DNNs). Traditional approaches rely on computationally expensive trained models or time-consuming iterative prune-retrain cycles, undermining their utility in resource-constrained settings. To address this issue, we build upon the established principles of saliency (LeCun et al., 1989) and connection sensitivity (Lee et al., 2018) to tackle the challenging problem of one-shot pruning neural networks (NNs) before training (PBT) at initialization. We introduce Fisher-Taylor Sensitivity (FTS), a computationally cheap and efficient pruning criterion based on the empirical Fisher Information Matrix (FIM) diagonal, offering a viable alternative for integrating first- and second-order information to identify a model's structurally important parameters. Although the FIM-Hessian equivalency only holds for convergent models that maximize the likelihood, recent studies (Karakida et al., 2019) suggest that, even at initialization, the FIM captures essential geometric information of parameters in overparameterized NNs, providing the basis for our method. Finally, we demonstrate empirically that layer collapse, a critical limitation of data-dependent pruning methodologies, is easily overcome by pruning within a single training epoch after initialization. We perform experiments on ResNet18 and VGG19 with CIFAR-10 and CIFAR-100, widely used benchmarks in pruning research. Our method achieves competitive performance against state-of-the-art techniques for one-shot PBT, even under extreme sparsity conditions. Our code is made available to the public. 

**Abstract (ZH)**: 剪枝提供了一种有希望的解决方案，以减轻部署大型深度神经网络（DNNs）相关的成本和环境影响。传统方法依赖于计算成本高昂的训练模型或耗时的剪枝-重新训练循环，这在资源受限的环境中削弱了其实用性。为了解决这一问题，我们基于拉普（LeCun等人，1989）提出的显著性以及Lee等人（2018）提出的连接敏感性原理，提出了在训练前初始化时一次性剪枝神经网络（NNs）的方法（PBT）。我们引入了Fisher-Taylor敏感性（FTS），这是一种基于经验Fisher信息矩阵（FIM）对角线的计算成本低廉且高效的剪枝准则，能够整合第一阶和第二阶信息以识别模型的结构重要参数。尽管FIM与哈essian等价仅适用于在最大化似然函数下收敛的模型，但最近的研究（Karakida等人，2019）表明，即使在初始化阶段，FIM也能捕获过参数化神经网络参数的重要几何信息，为我们的方法奠定了基础。最后，通过实验我们证明，依赖数据的剪枝方法中关键性的层塌缩问题，在初始化后单个训练周期内的剪枝操作中可以轻松解决。我们在ResNet18和VGG19上进行了实验，这两者是剪枝研究中广泛使用的基准数据集。即使在极端稀疏条件下，我们的方法也能够与最新的技术竞争。我们的代码已对公众开放。 

---
# Does Editing Provide Evidence for Localization? 

**Title (ZH)**: 编辑提供定位证据吗？ 

**Authors**: Zihao Wang, Victor Veitch  

**Link**: [PDF](https://arxiv.org/pdf/2502.11447)  

**Abstract**: A basic aspiration for interpretability research in large language models is to "localize" semantically meaningful behaviors to particular components within the LLM. There are various heuristics for finding candidate locations within the LLM. Once a candidate localization is found, it can be assessed by editing the internal representations at the corresponding localization and checking whether this induces model behavior that is consistent with the semantic interpretation of the localization. The question we address here is: how strong is the evidence provided by such edits? To assess localization, we want to assess the effect of the optimal intervention at a particular location. The key new technical tool is a way of adapting LLM alignment techniques to find such optimal localized edits. With this tool in hand, we give an example where the edit-based evidence for localization appears strong, but where localization clearly fails. Indeed, we find that optimal edits at random localizations can be as effective as aligning the full model. In aggregate, our results suggest that merely observing that localized edits induce targeted changes in behavior provides little to no evidence that these locations actually encode the target behavior. 

**Abstract (ZH)**: 在大型语言模型（LLM）的可解释性研究中，一个基本的目标是将具有语义意义的行为“本地化”到模型中的特定组件上。找到候选位置的方法有多种启发式方法。一旦找到了候选的位置，可以通过编辑相应位置的内部表示，并检查这种编辑是否引起了与该位置语义解释一致的模型行为来进行评估。我们解决的问题是：这样的编辑提供了多强的证据？为了评估本地化，我们需要评估特定位置的最佳干预措施的效果。关键的新技术工具是将LLM对齐技术适应以找到这样的最佳局部编辑。借助这种工具，我们给出了一个例子，在该例子中，基于编辑的本地化证据看似强烈，但实际上本地化明显失败。事实上，我们发现，在随机位置进行的最佳编辑可以与对整个模型进行对齐一样有效。综合来看，我们的结果表明，仅仅是观察局部编辑导致目标行为的变化，几乎不能证明这些位置实际上编码了目标行为。 

---
# Multi-Turn Multi-Modal Question Clarification for Enhanced Conversational Understanding 

**Title (ZH)**: 增强对话理解的多轮多模态问题澄清 

**Authors**: Kimia Ramezan, Alireza Amiri Bavandpour, Yifei Yuan, Clemencia Siro, Mohammad Aliannejadi  

**Link**: [PDF](https://arxiv.org/pdf/2502.11442)  

**Abstract**: Conversational query clarification enables users to refine their search queries through interactive dialogue, improving search effectiveness. Traditional approaches rely on text-based clarifying questions, which often fail to capture complex user preferences, particularly those involving visual attributes. While recent work has explored single-turn multi-modal clarification with images alongside text, such methods do not fully support the progressive nature of user intent refinement over multiple turns. Motivated by this, we introduce the Multi-turn Multi-modal Clarifying Questions (MMCQ) task, which combines text and visual modalities to refine user queries in a multi-turn conversation. To facilitate this task, we create a large-scale dataset named ClariMM comprising over 13k multi-turn interactions and 33k question-answer pairs containing multi-modal clarifying questions. We propose Mario, a retrieval framework that employs a two-phase ranking strategy: initial retrieval with BM25, followed by a multi-modal generative re-ranking model that integrates textual and visual information from conversational history. Our experiments show that multi-turn multi-modal clarification outperforms uni-modal and single-turn approaches, improving MRR by 12.88%. The gains are most significant in longer interactions, demonstrating the value of progressive refinement for complex queries. 

**Abstract (ZH)**: 会话查询澄清功能通过交互对话让用户能够细化他们的搜索查询，从而提高搜索效果。传统方法依赖于基于文本的澄清问题，但这些方法常常无法捕捉到复杂的用户偏好，特别是涉及视觉属性的偏好。近年来，虽然有研究探索了结合图片和文本的单次多模态澄清方法，但这些方法未能充分支持用户意图逐步细化的多轮特性。受此启发，我们引入了多轮多模态澄清问题（Multi-turn Multi-modal Clarifying Questions, MMCQ）任务，该任务结合了文本和视觉模态，在多轮对话中细化用户的查询。为了支持这一任务，我们创建了一个包含超过13000个轮次交互和33000个包含多模态澄清问题的问答对的大规模数据集ClariMM。我们提出了MARIO框架，该框架采用两阶段排序策略：初始检索使用BM25，随后是一个多模态生成重排序模型，该模型整合了对话历史中的文本和视觉信息。实验结果表明，多轮多模态澄清优于单模态和单轮方法，MRR提高了12.88%。特别是在长时间交互中，收益尤为显著，这证明了逐步细化对于复杂查询的价值。 

---
# An Efficient Row-Based Sparse Fine-Tuning 

**Title (ZH)**: 一种高效的基于行的稀疏微调方法 

**Authors**: Cen-Jhih Li, Aditya Bhaskara  

**Link**: [PDF](https://arxiv.org/pdf/2502.11439)  

**Abstract**: Fine-tuning is an important step in adapting foundation models such as large language models to downstream tasks. To make this step more accessible to users with limited computational budgets, it is crucial to develop fine-tuning methods that are memory and computationally efficient. Sparse Fine-tuning (SFT) and Low-rank adaptation (LoRA) are two frameworks that have emerged for addressing this problem and have been adopted widely in practice. In this work, we develop a new SFT framework, based on ideas from neural network pruning. At a high level, we first identify "important" neurons/nodes using feature importance metrics from network pruning (specifically, we use the structural pruning method), and then perform fine-tuning by restricting to weights involving these neurons. Using experiments on common language tasks, we demonstrate that our method significantly improves the memory efficiency of SFT without increasing training time complexity and implementation complexity, while achieving accuracy comparable to state-of-the-art methods such as LoRA and its variants. 

**Abstract (ZH)**: 精细调整是将大型语言模型等基础模型适应下游任务的重要步骤。为了使这一步骤更适配计算资源有限的用户，开发出内存和计算效率更高的精细调整方法至关重要。稀疏精细调整（SFT）和低秩调整（LoRA）两种框架因此而出现，并且已经被广泛应用于实际场景中。在这项工作中，我们基于神经网络剪枝的理念，开发了一种新的SFT框架。总体而言，我们首先使用网络剪枝中的特征重要性指标（具体来说是结构化剪枝方法）识别出“重要”的神经元/节点，然后通过限制涉及这些神经元/节点的权重来进行精细调整。通过在常用语言任务上的实验，我们证明，该方法在不增加训练时间复杂度和实现复杂度的情况下，显著提高了SFT的内存效率，同时实现了与当前最先进的方法（如LoRA及其变体）相当的准确性。 

---
# Learning Dexterous Bimanual Catch Skills through Adversarial-Cooperative Heterogeneous-Agent Reinforcement Learning 

**Title (ZH)**: 通过对抗-合作异构代理强化学习学习灵巧的双臂接物技能 

**Authors**: Taewoo Kim, Youngwoo Yoon, Jaehong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.11437)  

**Abstract**: Robotic catching has traditionally focused on single-handed systems, which are limited in their ability to handle larger or more complex objects. In contrast, bimanual catching offers significant potential for improved dexterity and object handling but introduces new challenges in coordination and control. In this paper, we propose a novel framework for learning dexterous bimanual catching skills using Heterogeneous-Agent Reinforcement Learning (HARL). Our approach introduces an adversarial reward scheme, where a throw agent increases the difficulty of throws-adjusting speed-while a catch agent learns to coordinate both hands to catch objects under these evolving conditions. We evaluate the framework in simulated environments using 15 different objects, demonstrating robustness and versatility in handling diverse objects. Our method achieved approximately a 2x increase in catching reward compared to single-agent baselines across 15 diverse objects. 

**Abstract (ZH)**: 机器人抓取技术传统上侧重于单手系统，这类系统在处理大型或复杂物体时能力有限。相比之下，双手抓取提供了提升灵巧度和物体处理能力的巨大潜力，但同时也引入了协调和控制的新挑战。本文提出了一种新的框架，用于通过异构代理强化学习（Heterogeneous-Agent Reinforcement Learning, HARL）学习灵巧的双手抓取技能。本方法引入了一种对抗性奖励方案，其中投掷代理通过调整投掷速度增加捕捉难度，而抓取代理则学习在这些不断变化的条件下协调双手抓取物体。我们在模拟环境中使用15种不同的物体对该框架进行了评估，展示了其在处理多种物体方面的稳定性和灵活性。该方法在15种不同物体上的抓取奖励方面比单一代理基线方法提高了大约2倍。 

---
# Counterfactual-Consistency Prompting for Relative Temporal Understanding in Large Language Models 

**Title (ZH)**: 相对时间理解中基于反事实一致性的提示方法 

**Authors**: Jongho Kim, Seung-won Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11425)  

**Abstract**: Despite the advanced capabilities of large language models (LLMs), their temporal reasoning ability remains underdeveloped. Prior works have highlighted this limitation, particularly in maintaining temporal consistency when understanding events. For example, models often confuse mutually exclusive temporal relations like ``before'' and ``after'' between events and make inconsistent predictions. In this work, we tackle the issue of temporal inconsistency in LLMs by proposing a novel counterfactual prompting approach. Our method generates counterfactual questions and enforces collective constraints, enhancing the model's consistency. We evaluate our method on multiple datasets, demonstrating significant improvements in event ordering for explicit and implicit events and temporal commonsense understanding by effectively addressing temporal inconsistencies. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）具有先进的能力，但其时间推理能力仍然较为欠缺。先前的研究已经指出了这一点，特别是在理解事件时保持时间一致性方面。例如，模型常常混淆互斥的时间关系，如“之前”和“之后”，从而做出不一致的预测。在本研究中，我们通过提出一种新颖的反事实提示方法来解决LLMs的时间不一致性问题。该方法生成反事实问题并施加集体约束，从而增强模型的一致性。我们在多个数据集上评估了该方法，结果显示，在事件排序和时间常识理解方面取得了显著改进，有效地解决了时间不一致性问题。 

---
# Without Paired Labeled Data: An End-to-End Self-Supervised Paradigm for UAV-View Geo-Localization 

**Title (ZH)**: 没有配对标注数据：无人机视角地理定位的端到端自监督范式 

**Authors**: Zhongwei Chen, Zhao-Xu Yang, Hai-Jun Rong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11381)  

**Abstract**: UAV-View Geo-Localization (UVGL) aims to ascertain the precise location of a UAV by retrieving the most similar GPS-tagged satellite image. However, existing methods predominantly rely on supervised learning paradigms that necessitate annotated paired data for training, which incurs substantial annotation costs and impedes large-scale deployment. To overcome this limitation, we propose the Dynamic Memory-Driven and Neighborhood Information Learning (DMNIL) network, a lightweight end-to-end self-supervised framework for UAV-view geo-localization. The DMNIL framework utilizes a dual-path clustering-based contrastive learning architecture as its baseline to model intra-view structural relationships, enhancing feature consistency and discriminability. Additionally, a dynamic memory-driven hierarchical learning module is proposed to progressively mine local and global information, reinforcing multi-level feature associations to improve model robustness. To bridge the domain gap between UAV and satellite views, we design an information-consistent evolutionary learning mechanism that systematically explores latent correlations within intra-view neighborhoods and across cross-view domains, ultimately constructing a unified cross-view feature representation space. Extensive experiments on three benchmarks (University-1652, SUES-200, and DenseUAV) demonstrate that DMNIL achieves competitive performance against state-of-the-art supervised methods while maintaining computational efficiency. Notably, this superiority is attained without relying on paired training data, underscoring the framework's practicality for real-world deployment. Codes will be released soon. 

**Abstract (ZH)**: UAV视域地理定位（UVGL）旨在通过检索最相似的具有GPS标签的卫星图像来确定无人机的确切位置。然而，现有的方法主要依赖于监督学习范式，需要标注的配对数据进行训练，这导致了注释成本的增加，并阻碍了大规模部署。为了解决这一局限性，我们提出了一种轻量级端到端自监督框架——动态记忆驱动和邻域信息学习（DMNIL）网络，用于UAV视域地理定位。DMNIL框架使用基于聚类的对比学习架构作为基础模型，以建模视角内的结构关系，提高特征的一致性和可区分性。此外，我们还提出了一种动态记忆驱动的分层学习模块，以逐步挖掘局部和全局信息，强化多层次特征关联，从而提高模型的鲁棒性。为了弥合UAV视域和卫星视域之间的领域差距，我们设计了一种信息一致的进化学习机制，系统地探索视角内邻域和跨视角域之间的潜在关联，最终构建了一个统一的跨视角特征表示空间。在三个基准数据集（University-1652、SUES-200和DenseUAV）上进行的大量实验表明，DMNIL在保持计算效率的同时，能够与最先进的监督方法竞争。值得注意的是，这一优越性是在不依赖配对训练数据的情况下实现的，这突显了该框架在实际部署中的实用性。代码即将开源。 

---
# CCJA: Context-Coherent Jailbreak Attack for Aligned Large Language Models 

**Title (ZH)**: CCJA：上下文一致的对齐大型语言模型的监狱突破攻击 

**Authors**: Guanghao Zhou, Panjia Qiu, Mingyuan Fan, Cen Chen, Mingyuan Chu, Xin Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.11379)  

**Abstract**: Despite explicit alignment efforts for large language models (LLMs), they can still be exploited to trigger unintended behaviors, a phenomenon known as "jailbreaking." Current jailbreak attack methods mainly focus on discrete prompt manipulations targeting closed-source LLMs, relying on manually crafted prompt templates and persuasion rules. However, as the capabilities of open-source LLMs improve, ensuring their safety becomes increasingly crucial. In such an environment, the accessibility of model parameters and gradient information by potential attackers exacerbates the severity of jailbreak threats. To address this research gap, we propose a novel \underline{C}ontext-\underline{C}oherent \underline{J}ailbreak \underline{A}ttack (CCJA). We define jailbreak attacks as an optimization problem within the embedding space of masked language models. Through combinatorial optimization, we effectively balance the jailbreak attack success rate with semantic coherence. Extensive evaluations show that our method not only maintains semantic consistency but also surpasses state-of-the-art baselines in attack effectiveness. Additionally, by integrating semantically coherent jailbreak prompts generated by our method into widely used black-box methodologies, we observe a notable enhancement in their success rates when targeting closed-source commercial LLMs. This highlights the security threat posed by open-source LLMs to commercial counterparts. We will open-source our code if the paper is accepted. 

**Abstract (ZH)**: 尽管对大型语言模型（LLMs）进行了明确的对齐努力，它们仍然可能被利用以触发未预期的行为，这一现象被称为“监管逃逸”（jailbreaking）。当前的监管逃逸攻击方法主要集中在针对封闭源代码LLMs的离散提示操纵上，依赖于手动构建的提示模板和说服规则。然而，随着开源LLMs能力的提升，确保其安全性变得日益重要。在这种环境中，潜在攻击者可以访问模型参数和梯度信息，进一步加剧了监管逃逸威胁的严重性。为应对这一研究缺口，我们提出了一种新颖的Context-Coherent Jailbreak Attack（CCJA）方法。我们将监管逃逸攻击定义为在掩码语言模型嵌入空间中的优化问题。通过组合优化，我们有效地平衡了监管逃逸攻击的成功率与语义一致性。广泛的实际测试表明，我们的方法不仅维持了语义一致性，还在攻击效果上超越了最先进的基线方法。此外，通过将我们方法生成的语义一致的监管逃逸提示整合到广泛使用的黑盒方法中，我们发现其在针对封闭源代码的商业LLMs时的成功率显著提高。这突显了开源LLMs对商业对应物构成的安全威胁。如果论文被接受，我们将开源我们的代码。 

---
# LLMs can Perform Multi-Dimensional Analytic Writing Assessments: A Case Study of L2 Graduate-Level Academic English Writing 

**Title (ZH)**: LLMs可以进行多维度分析写作评估：关于二语研究生水平学术英语写作的案例研究 

**Authors**: Zhengxiang Wang, Veronika Makarova, Zhi Li, Jordan Kodner, Owen Rambow  

**Link**: [PDF](https://arxiv.org/pdf/2502.11368)  

**Abstract**: The paper explores the performance of LLMs in the context of multi-dimensional analytic writing assessments, i.e. their ability to provide both scores and comments based on multiple assessment criteria. Using a corpus of literature reviews written by L2 graduate students and assessed by human experts against 9 analytic criteria, we prompt several popular LLMs to perform the same task under various conditions. To evaluate the quality of feedback comments, we apply a novel feedback comment quality evaluation framework. This framework is interpretable, cost-efficient, scalable, and reproducible, compared to existing methods that rely on manual judgments. We find that LLMs can generate reasonably good and generally reliable multi-dimensional analytic assessments. We release our corpus for reproducibility. 

**Abstract (ZH)**: 本文探讨了语言模型（LLM）在多维度分析性写作评估中的表现，即它们基于多个评估标准提供评分和评论的能力。我们使用由英语为第二语言（L2）研究生撰写的文献综述作为语料库，并由人类专家依据9个分析性标准进行评估。在此基础上，我们促使几种流行的LLM在不同条件下执行相同的评估任务。为了评估反馈评论的质量，我们应用了一种新的反馈评论质量评估框架。与依赖人工判断的现有方法相比，该框架具有可解释性、成本效益高、可扩展且可重复性好。我们发现，语言模型能够生成合理良好且总体可靠的多维度分析性评估。我们还将语料库开源以确保评估的可重复性。 

---
# Sparse Autoencoder Features for Classifications and Transferability 

**Title (ZH)**: 稀疏自编码器特征在分类与迁移性中的应用 

**Authors**: Jack Gallifant, Shan Chen, Kuleen Sasse, Hugo Aerts, Thomas Hartvigsen, Danielle S. Bitterman  

**Link**: [PDF](https://arxiv.org/pdf/2502.11367)  

**Abstract**: Sparse Autoencoders (SAEs) provide potentials for uncovering structured, human-interpretable representations in Large Language Models (LLMs), making them a crucial tool for transparent and controllable AI systems. We systematically analyze SAE for interpretable feature extraction from LLMs in safety-critical classification tasks. Our framework evaluates (1) model-layer selection and scaling properties, (2) SAE architectural configurations, including width and pooling strategies, and (3) the effect of binarizing continuous SAE activations. SAE-derived features achieve macro F1 > 0.8, outperforming hidden-state and BoW baselines while demonstrating cross-model transfer from Gemma 2 2B to 9B-IT models. These features generalize in a zero-shot manner to cross-lingual toxicity detection and visual classification tasks. Our analysis highlights the significant impact of pooling strategies and binarization thresholds, showing that binarization offers an efficient alternative to traditional feature selection while maintaining or improving performance. These findings establish new best practices for SAE-based interpretability and enable scalable, transparent deployment of LLMs in real-world applications. Full repo: this https URL. 

**Abstract (ZH)**: 稀疏自编码器（Sparse Autoencoders, SAEs）为揭示大型语言模型（Large Language Models, LLMs）中的结构化、可解读表示提供了潜力，使其成为透明和可控AI系统的重要工具。我们系统地分析了SAE在安全关键分类任务中从LLMs中提取可解读特征的潜力。我们的框架评估了（1）模型-层选择和缩放特性，（2）SAE架构配置，包括宽度和池化策略，以及（3）连续SAE激活二值化的效果。SAE提取的特征实现了宏F1 > 0.8，并在隐藏状态和词袋（Bag-of-Words, BoW）基线之上表现出优异的效果，同时展示了从Gemma 2B到9B-IT模型的跨模型迁移。这些特征在零样本设置下跨语言毒性检测和视觉分类任务中得以泛化。我们的分析强调了池化策略和二值化阈值的显著影响，表明二值化是一种提高性能的高效替代传统特征选择的方法。这些发现为基于SAE的可解释性确立了新的最佳实践，并使得LLM的大规模、透明部署在实际应用中成为可能。完整代码库：[这里](this https URL)。 

---
# SAIF: A Sparse Autoencoder Framework for Interpreting and Steering Instruction Following of Language Models 

**Title (ZH)**: SAIF：一种稀疏自编码框架，用于解释和引导语言模型的指令跟随 

**Authors**: Zirui He, Haiyan Zhao, Yiran Qiao, Fan Yang, Ali Payani, Jing Ma, Mengnan Du  

**Link**: [PDF](https://arxiv.org/pdf/2502.11356)  

**Abstract**: The ability of large language models (LLMs) to follow instructions is crucial for their practical applications, yet the underlying mechanisms remain poorly understood. This paper presents a novel framework that leverages sparse autoencoders (SAE) to interpret how instruction following works in these models. We demonstrate how the features we identify can effectively steer model outputs to align with given instructions. Through analysis of SAE latent activations, we identify specific latents responsible for instruction following behavior. Our findings reveal that instruction following capabilities are encoded by a distinct set of instruction-relevant SAE latents. These latents both show semantic proximity to relevant instructions and demonstrate causal effects on model behavior. Our research highlights several crucial factors for achieving effective steering performance: precise feature identification, the role of final layer, and optimal instruction positioning. Additionally, we demonstrate that our methodology scales effectively across SAEs and LLMs of varying sizes. 

**Abstract (ZH)**: 大型语言模型（LLMs）遵循指令的能力对于其实用应用至关重要，但其背后的机制尚不完全清楚。本文提出了一种新的框架，利用稀疏自编码器（SAE）来解释这些模型中指令遵循工作的机制。我们展示了我们识别出的特征如何有效引导模型输出与给定指令对齐。通过对SAE潜在激活进行分析，我们确定了特定的潜在变量负责指令遵循行为。我们的研究结果表明，指令遵循能力是由一组不同的、与指令相关的SAE潜在变量编码的。这些潜在变量不仅与相关指令在语义上接近，而且对模型行为具有因果效应。我们的研究突出了实现有效引导性能的关键因素：精确的特征识别、最终层的作用以及指令的最佳位置。此外，我们还证明了我们的方法在不同大小的SAE和LLMs中具有良好的可扩展性。 

---
# "Nuclear Deployed!": Analyzing Catastrophic Risks in Decision-making of Autonomous LLM Agents 

**Title (ZH)**: 《核武部署！》：分析自主大语言模型代理决策中的灾难性风险 

**Authors**: Rongwu Xu, Xiaojian Li, Shuo Chen, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11355)  

**Abstract**: Large language models (LLMs) are evolving into autonomous decision-makers, raising concerns about catastrophic risks in high-stakes scenarios, particularly in Chemical, Biological, Radiological and Nuclear (CBRN) domains. Based on the insight that such risks can originate from trade-offs between the agent's Helpful, Harmlessness and Honest (HHH) goals, we build a novel three-stage evaluation framework, which is carefully constructed to effectively and naturally expose such risks. We conduct 14,400 agentic simulations across 12 advanced LLMs, with extensive experiments and analysis. Results reveal that LLM agents can autonomously engage in catastrophic behaviors and deception, without being deliberately induced. Furthermore, stronger reasoning abilities often increase, rather than mitigate, these risks. We also show that these agents can violate instructions and superior commands. On the whole, we empirically prove the existence of catastrophic risks in autonomous LLM agents. We will release our code upon request. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在演变成自主决策者，这在高赌注场景中引发了关于灾难性风险的担忧，特别是在化学、生物、放射性和核（CBRN）领域。基于这样的风险源自自主体有益性、无害性和诚实性（HHH）目标之间的权衡，我们构建了一个新的三阶段评估框架，该框架精心设计，能够有效且自然地揭示这些风险。我们对12种先进的LLMs进行了14,400次代理模拟，并进行了大量实验和分析。结果表明，LLM代理可以在未被故意诱导的情况下自主参与灾难性行为和欺骗行为。此外，更强的推理能力往往增加了这些风险，而未能缓解它们。我们还展示了这些代理可以违反指令和上级命令。总体而言，我们的实证研究表明自主LLM代理中存在灾难性风险。如有需要，我们将提供我们的代码。 

---
# Inverse Flow and Consistency Models 

**Title (ZH)**: 逆向流动和一致性模型 

**Authors**: Yuchen Zhang, Jian Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.11333)  

**Abstract**: Inverse generation problems, such as denoising without ground truth observations, is a critical challenge in many scientific inquiries and real-world applications. While recent advances in generative models like diffusion models, conditional flow matching, and consistency models achieved impressive results by casting generation as denoising problems, they cannot be directly used for inverse generation without access to clean data. Here we introduce Inverse Flow (IF), a novel framework that enables using these generative models for inverse generation problems including denoising without ground truth. Inverse Flow can be flexibly applied to nearly any continuous noise distribution and allows complex dependencies. We propose two algorithms for learning Inverse Flows, Inverse Flow Matching (IFM) and Inverse Consistency Model (ICM). Notably, to derive the computationally efficient, simulation-free inverse consistency model objective, we generalized consistency training to any forward diffusion processes or conditional flows, which have applications beyond denoising. We demonstrate the effectiveness of IF on synthetic and real datasets, outperforming prior approaches while enabling noise distributions that previous methods cannot support. Finally, we showcase applications of our techniques to fluorescence microscopy and single-cell genomics data, highlighting IF's utility in scientific problems. Overall, this work expands the applications of powerful generative models to inversion generation problems. 

**Abstract (ZH)**: 逆生成问题，如没有真实参考值的降噪问题，在许多科学研究和实际应用中是一个关键挑战。虽然最近生成模型的进步，如扩散模型、条件流匹配和一致性模型通过将生成问题重新定义为降噪问题取得了令人瞩目的成果，但它们无法直接用于逆生成问题，除非能够访问干净的数据。在此，我们提出了一种新颖的框架——逆流（Inverse Flow, IF），该框架能够利用这些生成模型解决诸如没有真实参考值降噪在内的逆生成问题。逆流可以灵活应用于几乎所有连续的噪声分布，并且能够处理复杂的依赖关系。我们提出了两种学习逆流的方法——逆流匹配（Inverse Flow Matching, IFM）和逆一致性模型（Inverse Consistency Model, ICM）。特别地，为了推导出一个计算高效且无需模拟的逆一致性模型目标函数，我们将一致性训练推广到了任何正向扩散过程或条件流中，这些过程的应用范围不仅限于降噪。我们在合成和真实数据集上展示了IF的有效性，其性能优于之前的方法，并且能够支持之前方法无法处理的噪声分布。最后，我们展示了我们的技术在荧光显微镜和单细胞基因组学数据中的应用，突显了IF在科学问题中的实用性。总的来说，这项工作扩展了强大生成模型在逆生成问题中的应用范围。 

---
# System Message Generation for User Preferences using Open-Source Models 

**Title (ZH)**: 使用开源模型生成用户偏好的系统消息生成 

**Authors**: Minbyul Jeong, Jungho Cho, Minsoo Khang, Dawoon Jung, Teakgyu Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11330)  

**Abstract**: System messages play a crucial role in interactions with large language models (LLMs), often serving as prompts to initiate conversations. Through system messages, users can assign specific roles, perform intended tasks, incorporate background information, specify various output formats and communication styles. Despite such versatility, publicly available data are often lack system messages and subject to strict license constraints in the industry field. Manual labeling of publicly available data with system messages that align with user instructions demands significant resources. In view of such challenges, our work introduces SysGen, a pipeline for generating system messages with better aligned assistant responses from the supervised fine-tuning dataset without system messages. Training on SysGen data has demonstrated substantial improvements in the alignment of model responses with system messages and user instructions, as demonstrated across various open-source models on the Multifacet benchmark, while maintaining minimal impact on other unseen benchmarks such as Open LLM Leaderboard 2. Our qualitative analysis highlights the importance of diverse system messages to ensure better adaptability across different contexts. 

**Abstract (ZH)**: 系统消息在与大规模语言模型（LLMs）的交互中发挥着至关重要的作用，通常作为启动对话的提示。通过系统消息，用户可以指派特定角色、执行预定任务、融入背景信息、指定各种输出格式和沟通风格。尽管具有如此多的灵活性，但公开可用的数据中往往缺乏系统消息，且工业领域的数据受到严格的许可证约束。手动为公开可用的数据加上与用户指令相匹配的系统消息需要大量资源。鉴于这些挑战，我们引入了SysGen，这是一个用于生成来自监督微调数据集但本身没有系统消息的系统消息，以更好地对齐副助手的回应。使用SysGen数据集训练后，模型的响应与系统消息和用户指令的对齐程度显著提高，如在Multifacet基准上的多个开源模型中所示，同时对其他未见过的基准（如Open LLM Leaderboard 2）的影响最小。我们的定性分析强调了系统消息多样性的重要性，以确保在不同情境下的更好适应性。 

---
# ALGEN: Few-shot Inversion Attacks on Textual Embeddings using Alignment and Generation 

**Title (ZH)**: ALGEN: 使用对齐和生成进行文本嵌入的少样本逆向攻击 

**Authors**: Yiyi Chen, Qiongkai Xu, Johannes Bjerva  

**Link**: [PDF](https://arxiv.org/pdf/2502.11308)  

**Abstract**: With the growing popularity of Large Language Models (LLMs) and vector databases, private textual data is increasingly processed and stored as numerical embeddings. However, recent studies have proven that such embeddings are vulnerable to inversion attacks, where original text is reconstructed to reveal sensitive information. Previous research has largely assumed access to millions of sentences to train attack models, e.g., through data leakage or nearly unrestricted API access. With our method, a single data point is sufficient for a partially successful inversion attack. With as little as 1k data samples, performance reaches an optimum across a range of black-box encoders, without training on leaked data. We present a Few-shot Textual Embedding Inversion Attack using ALignment and GENeration (ALGEN), by aligning victim embeddings to the attack space and using a generative model to reconstruct text. We find that ALGEN attacks can be effectively transferred across domains and languages, revealing key information. We further examine a variety of defense mechanisms against ALGEN, and find that none are effective, highlighting the vulnerabilities posed by inversion attacks. By significantly lowering the cost of inversion and proving that embedding spaces can be aligned through one-step optimization, we establish a new textual embedding inversion paradigm with broader applications for embedding alignment in NLP. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）和向量数据库的普及，私人文本数据越来越多地被转换为数值嵌入并进行处理和存储。然而，最近的研究表明，这些嵌入对于反向攻击是脆弱的，通过这些攻击可以还原原始文本并揭示敏感信息。以往的研究大多假设可以访问数百万个句子来训练攻击模型，例如通过数据泄露或几乎不受限制的API访问。而通过我们的方法，单个数据点就足以实施部分成功的反向攻击。仅使用1000个数据样本，我们的攻击方法在多种黑盒编码器中达到了最佳性能，无需使用泄露的数据进行训练。我们提出了一种基于对齐和生成（ALGEN）的少样本文本嵌入反向攻击方法，通过将受害者的嵌入对齐到攻击空间，并使用生成模型重构文本。我们发现，ALGEN攻击可以在不同领域和语言之间有效迁移，揭示出关键信息。我们进一步研究了针对ALGEN的多种防御机制，发现均无效，强调了反向攻击带来的脆弱性。通过显著降低反向攻击的成本，并证明嵌入空间可以通过一维优化对齐，我们确立了一种新的文本嵌入反向攻击范式，该范式在NLP中的嵌入对齐方面有着更广泛的应用潜力。 

---
# Exploiting Point-Language Models with Dual-Prompts for 3D Anomaly Detection 

**Title (ZH)**: 利用双提示点语言模型进行3D异常检测 

**Authors**: Jiaxiang Wang, Haote Xu, Xiaolu Chen, Haodi Xu, Yue Huang, Xinghao Ding, Xiaotong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11307)  

**Abstract**: Anomaly detection (AD) in 3D point clouds is crucial in a wide range of industrial applications, especially in various forms of precision manufacturing. Considering the industrial demand for reliable 3D AD, several methods have been developed. However, most of these approaches typically require training separate models for each category, which is memory-intensive and lacks flexibility. In this paper, we propose a novel Point-Language model with dual-prompts for 3D ANomaly dEtection (PLANE). The approach leverages multi-modal prompts to extend the strong generalization capabilities of pre-trained Point-Language Models (PLMs) to the domain of 3D point cloud AD, achieving impressive detection performance across multiple categories using a single model. Specifically, we propose a dual-prompt learning method, incorporating both text and point cloud prompts. The method utilizes a dynamic prompt creator module (DPCM) to produce sample-specific dynamic prompts, which are then integrated with class-specific static prompts for each modality, effectively driving the PLMs. Additionally, based on the characteristics of point cloud data, we propose a pseudo 3D anomaly generation method (Ano3D) to improve the model's detection capabilities in an unsupervised setting. Experimental results demonstrate that the proposed method, which is under the multi-class-one-model paradigm, achieves a +8.7%/+17% gain on anomaly detection and localization performance as compared to the state-of-the-art one-class-one-model methods for the Anomaly-ShapeNet dataset, and obtains +4.3%/+4.1% gain for the Real3D-AD dataset. Code will be available upon publication. 

**Abstract (ZH)**: 三维点云中的异常检测（AD）在众多工业应用中至关重要，特别是在各种精密制造领域。鉴于工业对可靠三维异常检测的需求，已经开发出了多种方法。然而，这些方法大多需要为每个类别训练单独的模型，这既消耗内存且缺乏灵活性。本文提出了一种新型的Point-Language模型（PLANE），其具有双重提示方法。该方法通过多模态提示，将预训练的Point-Language模型（PLMs）的强泛化能力扩展到三维点云异常检测领域，使用单一模型即可在多个类别上实现出色的检测性能。具体而言，我们提出了一种双重提示学习方法，同时融入了文本和点云提示。该方法利用动态提示生成模块（DPCM）生成样本特定的动态提示，然后将这些动态提示与每个模态的类别特定静态提示结合，有效驱动PLMs。另外，根据点云数据的特点，我们提出了一种伪三维异常生成方法（Ano3D），在无监督环境下提高模型的检测能力。实验结果表明，该方法在多类单模型范式下，与最先进的单类单模型方法相比，Anomaly-ShapeNet数据集的异常检测和定位性能分别提高了8.7%/17%，Real3D-AD数据集分别提高了4.3%/4.1%。出版后将提供代码。 

---
# CORDIAL: Can Multimodal Large Language Models Effectively Understand Coherence Relationships? 

**Title (ZH)**: CORDIAL：多模态大型语言模型能否有效理解连贯关系？ 

**Authors**: Aashish Anantha Ramakrishnan, Aadarsh Anantha Ramakrishnan, Dongwon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.11300)  

**Abstract**: Multimodal Large Language Models (MLLMs) are renowned for their superior instruction-following and reasoning capabilities across diverse problem domains. However, existing benchmarks primarily focus on assessing factual and logical correctness in downstream tasks, with limited emphasis on evaluating MLLMs' ability to interpret pragmatic cues and intermodal relationships. To address this gap, we assess the competency of MLLMs in performing Multimodal Discourse Analysis (MDA) using Coherence Relations. Our benchmark, CORDIAL, encompasses a broad spectrum of Coherence Relations across 3 different discourse domains at varying levels of granularity. Through our experiments on 10+ MLLMs employing different prompting strategies, we show that even top models like Gemini 1.5 Pro and GPT-4o fail to match the performance of simple classifier-based baselines. This study emphasizes the need to move beyond similarity-based metrics and adopt a discourse-driven framework for evaluating MLLMs, providing a more nuanced assessment of their capabilities. The benchmark and code are available at: this https URL. 

**Abstract (ZH)**: 多模态大语言模型（MLLMs）在多种问题域中表现出卓越的指令遵循和推理能力。然而，现有的基准主要集中在评估下游任务中的事实和逻辑正确性，对评估MLLMs解释语用线索和跨模态关系的能力关注较少。为解决这一问题，我们使用连贯关系对MLLMs进行多模态话语分析（MDA）能力进行了评估。我们的基准测试Cordial涵盖了三个不同话语领域中不同粒度水平的广泛连贯关系。通过在10多种不同提示策略的MLLMs上进行实验，我们发现即使是如Gemini 1.5 Pro和GPT-4o这样的顶尖模型，其表现也无法达到基于简单分类器基线的表现。本研究强调了超出基于相似性的评估指标、转向以话语驱动框架评估MLLMs的必要性，提供了对其能力的更细致入微的评估。基准测试和代码可在以下链接获得：[该链接]。 

---
# Integrating Language Models for Enhanced Network State Monitoring in DRL-Based SFC Provisioning 

**Title (ZH)**: 基于DRL的SFC provisioning中语言模型集成的网络状态监控增强方法 

**Authors**: Parisa Fard Moshiri, Murat Arda Onsu, Poonam Lohan, Burak Kantarci, Emil Janulewicz  

**Link**: [PDF](https://arxiv.org/pdf/2502.11298)  

**Abstract**: Efficient Service Function Chain (SFC) provisioning and Virtual Network Function (VNF) placement are critical for enhancing network performance in modern architectures such as Software-Defined Networking (SDN) and Network Function Virtualization (NFV). While Deep Reinforcement Learning (DRL) aids decision-making in dynamic network environments, its reliance on structured inputs and predefined rules limits adaptability in unforeseen scenarios. Additionally, incorrect actions by a DRL agent may require numerous training iterations to correct, potentially reinforcing suboptimal policies and degrading performance. This paper integrates DRL with Language Models (LMs), specifically Bidirectional Encoder Representations from Transformers (BERT) and DistilBERT, to enhance network management. By feeding final VNF allocations from DRL into the LM, the system can process and respond to queries related to SFCs, DCs, and VNFs, enabling real-time insights into resource utilization, bottleneck detection, and future demand planning. The LMs are fine-tuned to our domain-specific dataset using Low-Rank Adaptation (LoRA). Results show that BERT outperforms DistilBERT with a lower test loss (0.28 compared to 0.36) and higher confidence (0.83 compared to 0.74), though BERT requires approximately 46% more processing time. 

**Abstract (ZH)**: 高效的Service Function Chain (SFC) 部署和服务功能虚拟化（VNF）放置对于现代架构（如软件定义网络（SDN）和网络功能虚拟化（NFV））中的网络性能提升至关重要。虽然深度强化学习（DRL）在动态网络环境中有助于决策制定，但其依赖于结构化的输入和预定义的规则，限制了其在未预见场景中的适应性。此外，DRL代理的错误行为可能需要多次训练迭代来纠正，可能强化次优策略并降低性能。本文将DRL与语言模型（LM）相结合，特别采用双向变压器编码器表示（BERT）和DistilBERT，以增强网络管理。通过将DRL最终确定的VNF分配输入到LM中，系统可以处理与SFC、数据中心（DC）和VNF相关的查询，实现对资源利用率、瓶颈检测和未来需求规划的实时洞察。我们使用低秩适应（LoRA）对LM进行微调。结果显示，BERT在测试损失（0.28 vs 0.36）和置信度（0.83 vs 0.74）方面均优于DistilBERT，尽管BERT大约需要46%更多的处理时间。 

---
# FairFare: A Tool for Crowdsourcing Rideshare Data to Empower Labor Organizers 

**Title (ZH)**: FairFare：一种众包 rideshare 数据的工具，助力劳工组织者 

**Authors**: Dana Calacci, Varun Nagaraj Rao, Samantha Dalal, Catherine Di, Kok-Wei Pua, Andrew Schwartz, Danny Spitzberg, Andrés Monroy-Hernández  

**Link**: [PDF](https://arxiv.org/pdf/2502.11273)  

**Abstract**: Rideshare workers experience unpredictable working conditions due to gig work platforms' reliance on opaque AI and algorithmic systems. In response to these challenges, we found that labor organizers want data to help them advocate for legislation to increase the transparency and accountability of these platforms. To address this need, we collaborated with a Colorado-based rideshare union to develop FairFare, a tool that crowdsources and analyzes workers' data to estimate the take rate -- the percentage of the rider price retained by the rideshare platform. We deployed FairFare with our partner organization that collaborated with us in collecting data on 76,000+ trips from 45 drivers over 18 months. During evaluation interviews, organizers reported that FairFare helped influence the bill language and passage of Colorado Senate Bill 24-75, calling for greater transparency and data disclosure of platform operations, and create a national narrative. Finally, we reflect on complexities of translating quantitative data into policy outcomes, nature of community based audits, and design implications for future transparency tools. 

**Abstract (ZH)**: 打车服务平台依赖于不透明的人工智能和算法系统，导致打车工人的工作条件具有高度不确定性。面对这些挑战，我们发现劳工组织者希望获得数据支持，以帮助他们倡导立法，增加这些平台的透明度和可问责性。为了满足这一需求，我们与一家基于科罗拉多州的打车工会合作，开发了FairFare这一工具，以众包和分析工人的数据来估算取费率——即由打车平台保留的占乘客支付金额的比例。我们与合作伙伴组织合作，在18个月内收集了45名司机的76,000多次行程的数据。在评估访谈中，劳工组织者报告称，FairFare帮助影响了科罗拉多州参议院第24-75号法案的语言和通过，该法案呼吁提高平台运营的透明度和数据披露，并形成了全国性的叙述。最后，我们反思了将定量数据转化为政策结果的复杂性、社区审计的本质以及未来透明度工具的设计 implications。 

---
# Prompting in the Dark: Assessing Human Performance in Prompt Engineering for Data Labeling When Gold Labels Are Absent 

**Title (ZH)**: 在黑暗中推动：评估在缺乏标准标签情况下提示工程对数据标注的人类性能 

**Authors**: Zeyu He, Saniya Naphade, Ting-Hao 'Kenneth' Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11267)  

**Abstract**: Millions of users prompt large language models (LLMs) for various tasks, but how good are people at prompt engineering? Do users actually get closer to their desired outcome over multiple iterations of their prompts? These questions are crucial when no gold-standard labels are available to measure progress. This paper investigates a scenario in LLM-powered data labeling, "prompting in the dark," where users iteratively prompt LLMs to label data without using manually-labeled benchmarks. We developed PromptingSheet, a Google Sheets add-on that enables users to compose, revise, and iteratively label data through spreadsheets. Through a study with 20 participants, we found that prompting in the dark was highly unreliable-only 9 participants improved labeling accuracy after four or more iterations. Automated prompt optimization tools like DSPy also struggled when few gold labels were available. Our findings highlight the importance of gold labels and the needs, as well as the risks, of automated support in human prompt engineering, providing insights for future tool design. 

**Abstract (ZH)**: 以下是符合学术规范的中文翻译：

数百万用户通过大型语言模型（LLMs）请求执行各种任务，但用户在提示工程方面究竟做得如何？他们在多次迭代提示后是否能够更接近预期结果？在没有标准标签可用的情况下，这些问题至关重要。本文探讨了LLM辅助数据标注中的一个场景，即“在黑暗中提示”（Prompting in the Dark），其中用户通过迭代提示LLMs进行数据标注，而不使用手动标注的基准。我们开发了PromptingSheet，这是一个Google Sheets插件，使用户可以通过电子表格进行提示编辑、修订和数据标注。通过一项涉及20名参与者的实验，我们发现“在黑暗中提示”极为不可靠——只有9名参与者在四次或四次以上的迭代后提高了标注准确性。当可用的标准标签较少时，自动提示优化工具如DSPy也难以奏效。我们的研究结果强调了标准标签的重要性，并指出了自动辅助在人类提示工程中的需求和风险，为未来工具设计提供了参考。 

---
# Generating Skyline Datasets for Data Science Models 

**Title (ZH)**: 生成数据科学模型的天际线数据集 

**Authors**: Mengying Wang, Hanchao Ma, Yiyang Bian, Yangxin Fan, Yinghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11262)  

**Abstract**: Preparing high-quality datasets required by various data-driven AI and machine learning models has become a cornerstone task in data-driven analysis. Conventional data discovery methods typically integrate datasets towards a single pre-defined quality measure that may lead to bias for downstream tasks. This paper introduces MODis, a framework that discovers datasets by optimizing multiple user-defined, model-performance measures. Given a set of data sources and a model, MODis selects and integrates data sources into a skyline dataset, over which the model is expected to have the desired performance in all the performance measures. We formulate MODis as a multi-goal finite state transducer, and derive three feasible algorithms to generate skyline datasets. Our first algorithm adopts a "reduce-from-universal" strategy, that starts with a universal schema and iteratively prunes unpromising data. Our second algorithm further reduces the cost with a bi-directional strategy that interleaves data augmentation and reduction. We also introduce a diversification algorithm to mitigate the bias in skyline datasets. We experimentally verify the efficiency and effectiveness of our skyline data discovery algorithms, and showcase their applications in optimizing data science pipelines. 

**Abstract (ZH)**: 数据驱动的人工智能和机器学习模型所需高质量数据集的准备已成为数据驱动分析的关键任务。传统的数据发现方法通常集成数据集以满足单一预定义的质量标准，这可能会导致下游任务中的偏差。本文介绍了一种名为MODis的框架，该框架通过优化多个自定义的模型性能指标来发现数据集。给定一组数据源和一个模型，MODis选择并集成数据源，使其能够在这所有性能指标上达到预期的模型性能。我们将MODis形式化为多目标有限状态转换器，并推导出三种生成最优点集数据集的有效算法。我们的第一个算法采用了一种“以全集为基础，逐步削减”的策略，从一个通用模式开始，逐步去除无希望的数据。我们的第二个算法进一步采用了一种双向策略，即交替进行数据扩充和削减以降低成本。此外，我们还引入了一个多样性算法，以减轻最优点集数据集中的偏差。我们通过实验验证了我们的最优点集数据发现算法的效率和有效性，并展示了它们在优化数据科学管道中的应用。 

---
# Shortcuts and Identifiability in Concept-based Models from a Neuro-Symbolic Lens 

**Title (ZH)**: 从神经符号视角看基于概念的模型中的捷径和可识别性 

**Authors**: Samuele Bortolotti, Emanuele Marconato, Paolo Morettin, Andrea Passerini, Stefano Teso  

**Link**: [PDF](https://arxiv.org/pdf/2502.11245)  

**Abstract**: Concept-based Models are neural networks that learn a concept extractor to map inputs to high-level concepts and an inference layer to translate these into predictions. Ensuring these modules produce interpretable concepts and behave reliably in out-of-distribution is crucial, yet the conditions for achieving this remain unclear. We study this problem by establishing a novel connection between Concept-based Models and reasoning shortcuts (RSs), a common issue where models achieve high accuracy by learning low-quality concepts, even when the inference layer is fixed and provided upfront. Specifically, we first extend RSs to the more complex setting of Concept-based Models and then derive theoretical conditions for identifying both the concepts and the inference layer. Our empirical results highlight the impact of reasoning shortcuts and show that existing methods, even when combined with multiple natural mitigation strategies, often fail to meet these conditions in practice. 

**Abstract (ZH)**: 概念基模型是神经网络，能够学习一个概念提取器将输入映射到高层次的概念，并通过推理层将这些概念转化为预测。确保这些模块生成可解释的概念并可靠地处理未见过的数据集是至关重要的，但实现这些条件的条件仍不清楚。我们通过建立概念基模型与推理捷径（RSs）之间的新型联系来研究这一问题。推理捷径是指模型通过学习低质量的概念而实现高准确率，即便推理层是固定的并且事先提供。具体而言，我们首先将RSs扩展到概念基模型的更复杂设置，然后推导出识别概念和推理层的理论条件。我们的实验证据突显了推理捷径的影响，并表明即使结合多个自然缓解策略，现有方法在实践中也往往无法满足这些条件。 

---
# Soteria: Language-Specific Functional Parameter Steering for Multilingual Safety Alignment 

**Title (ZH)**: Soteria：针对多语言安全对齐的语言特定功能性参数引导 

**Authors**: Somnath Banerjee, Sayan Layek, Pratyush Chatterjee, Animesh Mukherjee, Rima Hazra  

**Link**: [PDF](https://arxiv.org/pdf/2502.11244)  

**Abstract**: Ensuring consistent safety across multiple languages remains a significant challenge for large language models (LLMs). We introduce Soteria, a lightweight yet powerful strategy that locates and minimally adjusts the "functional heads" most responsible for harmful content generation in each language. By altering only a fraction of parameters, Soteria drastically reduces policy violations without sacrificing overall model performance, even in low-resource settings. To rigorously evaluate our approach, we also present XThreatBench, a specialized multilingual dataset capturing fine-grained harmful behaviors drawn from real policy guidelines. Experiments with leading open-source LLMs (e.g., Llama, Qwen, Mistral) show that Soteria consistently improves safety metrics across high-, mid-, and low-resource languages. These findings highlight a promising path toward scalable, linguistically attuned, and ethically aligned LLMs worldwide. 

**Abstract (ZH)**: 确保多语言环境中的安全性一致性仍然是大型语言模型（LLMs）面临的一项重大挑战。我们提出了Soteria，这是一种轻量级但强大的策略，用于定位并最小调整对有害内容生成最负责的“功能头”在每种语言中的位置。通过仅调整少量参数，Soteria显著减少了政策违规行为，同时在整体模型性能方面几乎没有牺牲，即使在资源有限的环境中也是如此。为了严格评估我们的方法，我们还引入了XThreatBench，这是一个专门针对多语言环境的数据库，它捕捉了从实际政策指南中抽取的细微有害行为。使用领先开源LLM（例如Llama、Qwen、Mistral）的实验表明，Soteria能够一致地提高高资源、中资源和低资源语言的安全性指标。这些 findings 阐明了迈向具有广泛适用性、语言敏感性和伦理一致性的LLM的有希望的道路。 

---
# Towards identifying possible fault-tolerant advantage of quantum linear system algorithms in terms of space, time and energy 

**Title (ZH)**: 关于量子线性系统算法在空间、时间和能量方面潜在容错优势的识别 

**Authors**: Yue Tu, Mark Dubynskyi, Mohammad Mohammadisiahroudi, Ekaterina Riashchentceva, Jinglei Cheng, Dmitry Ryashchentsev, Tamás Terlaky, Junyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11239)  

**Abstract**: Quantum computing, a prominent non-Von Neumann paradigm beyond Moore's law, can offer superpolynomial speedups for certain problems. Yet its advantages in efficiency for tasks like machine learning remain under investigation, and quantum noise complicates resource estimations and classical comparisons. We provide a detailed estimation of space, time, and energy resources for fault-tolerant superconducting devices running the Harrow-Hassidim-Lloyd (HHL) algorithm, a quantum linear system solver relevant to linear algebra and machine learning. Excluding memory and data transfer, possible quantum advantages over the classical conjugate gradient method could emerge at $N \approx 2^{33} \sim 2^{48}$ or even lower, requiring ${O}(10^5)$ physical qubits, ${O}(10^{12}\sim10^{13})$ Joules, and ${O}(10^6)$ seconds under surface code fault-tolerance with three types of magic state distillation (15-1, 116-12, 225-1). Key parameters include condition number, sparsity, and precision $\kappa, s\approx{O}(10\sim100)$, $\epsilon\sim0.01$, and physical error $10^{-5}$. Our resource estimator adjusts $N, \kappa, s, \epsilon$, providing a map of quantum-classical boundaries and revealing where a practical quantum advantage may arise. Our work quantitatively determine how advanced a fault-tolerant quantum computer should be to achieve possible, significant benefits on problems related to real-world. 

**Abstract (ZH)**: 量子计算是一种超越摩尔定律的非冯·诺依曼范式，对于某些问题可以提供超多项式加速。然而，其在机器学习等任务上的效率优势仍待进一步调查，且量子噪声复杂了资源估算和经典对比。我们对使用纠错超导器件运行Harrow-Hassidim-Lloyd (HHL) 算法的空间、时间和能量资源进行了详细估算，HHL算法是与线性代数和机器学习相关的量子线性系统求解器。排除内存和数据传输部分，与经典的共轭梯度方法相比，在 $N \approx 2^{33} \sim 2^{48}$ 或更低的情况下可能会展现出量子优势，需要 $O(10^5)$ 个物理量子位、$O(10^{12}\sim10^{13})$ 焦耳以及 $O(10^6)$ 秒的表面码纠错，在三种类型的魔法态制备（15-1, 116-12, 225-1）下进行。关键参数包括条件数、稀疏性和精度 $\kappa, s \approx O(10 \sim 100)$，$\epsilon \sim 0.01$，以及物理错误率 $10^{-5}$。我们的资源估算器调整了 $N, \kappa, s, \epsilon$ 参数，提供了一幅量子与经典计算边界图，揭示了潜在的实际量子优势可能出现的区域。我们的工作定量地决定了为了在现实世界相关问题上获得可能的重要益处，纠错型的量子计算机应达到何种先进程度。 

---
# Vendi-RAG: Adaptively Trading-Off Diversity And Quality Significantly Improves Retrieval Augmented Generation With LLMs 

**Title (ZH)**: Vendi-RAG：适配性地权衡多样性和质量显著改进基于大语言模型的检索增强生成 

**Authors**: Mohammad Reza Rezaei, Adji Bousso Dieng  

**Link**: [PDF](https://arxiv.org/pdf/2502.11228)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) for domain-specific question-answering (QA) tasks by leveraging external knowledge sources. However, traditional RAG systems primarily focus on relevance-based retrieval and often struggle with redundancy, especially when reasoning requires connecting information from multiple sources. This paper introduces Vendi-RAG, a framework based on an iterative process that jointly optimizes retrieval diversity and answer quality. This joint optimization leads to significantly higher accuracy for multi-hop QA tasks. Vendi-RAG leverages the Vendi Score (VS), a flexible similarity-based diversity metric, to promote semantic diversity in document retrieval. It then uses an LLM judge that evaluates candidate answers, generated after a reasoning step, and outputs a score that the retriever uses to balance relevance and diversity among the retrieved documents during each iteration. Experiments on three challenging datasets -- HotpotQA, MuSiQue, and 2WikiMultiHopQA -- demonstrate Vendi-RAG's effectiveness in multi-hop reasoning tasks. The framework achieves significant accuracy improvements over traditional single-step and multi-step RAG approaches, with accuracy increases reaching up to +4.2% on HotpotQA, +4.1% on 2WikiMultiHopQA, and +1.3% on MuSiQue compared to Adaptive-RAG, the current best baseline. The benefits of Vendi-RAG are even more pronounced as the number of retrieved documents increases. Finally, we evaluated Vendi-RAG across different LLM backbones, including GPT-3.5, GPT-4, and GPT-4o-mini, and observed consistent improvements, demonstrating that the framework's advantages are model-agnostic. 

**Abstract (ZH)**: 检索增强生成（RAG）通过利用外部知识源，增强了大规模语言模型（LLMs）在特定领域的问题回答（QA）任务中的表现。然而，传统的RAG系统主要侧重于相关性的检索，往往在需要将信息从多个来源连接起来的推理过程中难以避免冗余。本文引入了Vendi-RAG框架，这是一个基于迭代过程的框架，能够同时优化检索多样性和答案质量。这种联合优化大幅提高了多跳QA任务的准确性。Vendi-RAG利用Vendi得分（VS），一个基于灵活相似度的多样性度量方法，来促进文档检索中的语义多样性。随后使用一个LLM裁判来评估生成的答案（在推理步骤之后），并输出一个评分，该评分用于每个迭代过程中平衡检索到的文档的相关性和多样性。在三个具有挑战性的数据集——HotpotQA、MuSiQue和2WikiMultiHopQA——上的实验表明，Vendi-RAG在多跳推理任务中表现出色。该框架相对于传统的单步和多步RAG方法，在各数据集上取得了显著的准确性提升，具体为：在HotpotQA上提高了4.2%，在2WikiMultiHopQA上提高了4.1%，在MuSiQue上提高了1.3%，相较于当前最佳基线适配性RAG（Adaptive-RAG）。随着检索文档数量的增加，Vendi-RAG的益处更为显著。最后，我们对GPT-3.5、GPT-4和GPT-4o-mini等多个LLM骨干进行了Vendi-RAG的评估，并观察到一致的改进，这表明该框架的优势具有模型通用性。 

---
# METAFOR: A Hybrid Metaheuristics Software Framework for Single-Objective Continuous Optimization Problems 

**Title (ZH)**: METAFOR：一种用于单目标连续优化问题的混合元启发式软件框架 

**Authors**: Christian Camacho-Villalón, Marco Dorigo, Thomas Stützle  

**Link**: [PDF](https://arxiv.org/pdf/2502.11225)  

**Abstract**: Hybrid metaheuristics are powerful techniques for solving difficult optimization problems that exploit the strengths of different approaches in a single implementation. For algorithm designers, however, creating hybrid metaheuristic implementations has become increasingly challenging due to the vast number of design options available in the literature and the fact that they often rely on their knowledge and intuition to come up with new algorithm designs. In this paper, we propose a modular metaheuristic software framework, called METAFOR, that can be coupled with an automatic algorithm configuration tool to automatically design hybrid metaheuristics. METAFOR is specifically designed to hybridize Particle Swarm Optimization, Differential Evolution and Covariance Matrix Adaptation-Evolution Strategy, and includes a local search module that allows their execution to be interleaved with a subordinate local search. We use the configuration tool irace to automatically generate 17 different metaheuristic implementations and evaluate their performance on a diverse set of continuous optimization problems. Our results show that, across all the considered problem classes, automatically generated hybrid implementations are able to outperform configured single-approach implementations, while these latter offer advantages on specific classes of functions. We provide useful insights on the type of hybridization that works best for specific problem classes, the algorithm components that contribute to the performance of the algorithms, and the advantages and disadvantages of two well-known instance separation strategies, creating stratified training set using a fix percentage and leave-one-class-out cross-validation. 

**Abstract (ZH)**: 混合元启发式方法是解决复杂优化问题的强大技术，能够在单一实现中充分利用不同方法的优势。然而，对于算法设计者而言，由于文献中可供选择的设计选项众多，以及他们通常依赖知识和直觉来设计新的算法，因此创建混合元启发式实现变得越来越具挑战性。本文提出了一种模块化元启发式软件框架，名为METAFOR，并结合了自动算法配置工具，以自动设计混合元启发式方法。METAFOR特别设计用于混合粒子群优化（PSO）、差分进化（DE）和适应协方差矩阵的进化策略（CMA-ES），并包括一个局部搜索模块，允许在次要局部搜索中与这些方法的执行交错进行。我们使用配置工具irace自动生成17种不同的元启发式实现，并在一系列不同的连续优化问题上评估其性能。结果显示，在所有考虑的问题类别中，自动生成的混合实现能够优于配置的单一方法实现，而后者的实现则在特定函数类别上具有优势。我们提供了关于特定问题类别中最有效混合方式、有助于算法性能的算法组件以及两种广为人知的实例分离策略（固定百分比分层训练集和剔除一类交叉验证）的优点和缺点的有用见解。 

---
# Stochastic Optimization of Inventory at Large-scale Supply Chains 

**Title (ZH)**: 大规模供应链中的库存随机优化 

**Authors**: Zhaoyang Larry Jin, Mehdi Maasoumy, Yimin Liu, Zeshi Zheng, Zizhuo Ren  

**Link**: [PDF](https://arxiv.org/pdf/2502.11213)  

**Abstract**: Today's global supply chains face growing challenges due to rapidly changing market conditions, increased network complexity and inter-dependency, and dynamic uncertainties in supply, demand, and other factors. To combat these challenges, organizations employ Material Requirements Planning (MRP) software solutions to set inventory stock buffers - for raw materials, work-in-process goods, and finished products - to help them meet customer service levels. However, holding excess inventory further complicates operations and can lock up millions of dollars of capital that could be otherwise deployed. Furthermore, most commercially available MRP solutions fall short in considering uncertainties and do not result in optimal solutions for modern enterprises.
At C3 AI, we fundamentally reformulate the inventory management problem as a constrained stochastic optimization. We then propose a simulation-optimization framework that minimizes inventory and related costs while maintaining desired service levels. The framework's goal is to find the optimal reorder parameters that minimize costs subject to a pre-defined service-level constraint and all other real-world operational constraints. These optimal reorder parameters can be fed back into an MRP system to drive optimal order placement, or used to place optimal orders directly. This approach has proven successful in reducing inventory levels by 10-35 percent, resulting in hundreds of millions of dollars of economic benefit for major enterprises at a global scale. 

**Abstract (ZH)**: 当今全球供应链正面临日益增长的挑战，这些挑战源于市场条件的快速变化、网络复杂的相互依赖性以及供应、需求和其他因素的动态不确定性。为了应对这些挑战，组织采用物料需求计划（MRP）软件解决方案来设置原材料、在制品和成品的库存缓冲，以帮助满足客户服务水平。然而，持有过多库存会进一步复杂化运营，并可能将大量资金锁在库存中，这些资金本可以重新分配使用。此外，大多数商用MRP解决方案在考虑不确定性方面存在不足，无法为企业提供最优解决方案。

在C3 AI，我们从根本上将库存管理问题重新表述为受限的随机优化问题。随后，我们提出了一种仿真-优化框架，该框架旨在在保持期望的服务水平的同时，最小化库存及相关成本。该框架的目标是找到最优的订货参数，这些参数在满足预定义的服务水平约束和其他所有实际运营约束的条件下，最小化成本。这些最优的订货参数可以反馈到MRP系统中以驱动最优订货，或直接用于下达最优订货。该方法在跨国大型企业中已被证明能够成功地将库存水平降低10-35%，从而在全球范围内为企业创造了数百亿美元的经济效益。 

---
# A Survey of LLM-based Agents in Medicine: How far are we from Baymax? 

**Title (ZH)**: 基于大型语言模型的医疗智能体综述：我们距离贝马克斯还有多远？ 

**Authors**: Wenxuan Wang, Zizhan Ma, Zheng Wang, Chenghan Wu, Wenting Chen, Xiang Li, Yixuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11211)  

**Abstract**: Large Language Models (LLMs) are transforming healthcare through the development of LLM-based agents that can understand, reason about, and assist with medical tasks. This survey provides a comprehensive review of LLM-based agents in medicine, examining their architectures, applications, and challenges. We analyze the key components of medical agent systems, including system profiles, clinical planning mechanisms, medical reasoning frameworks, and external capacity enhancement. The survey covers major application scenarios such as clinical decision support, medical documentation, training simulations, and healthcare service optimization. We discuss evaluation frameworks and metrics used to assess these agents' performance in healthcare settings. While LLM-based agents show promise in enhancing healthcare delivery, several challenges remain, including hallucination management, multimodal integration, implementation barriers, and ethical considerations. The survey concludes by highlighting future research directions, including advances in medical reasoning inspired by recent developments in LLM architectures, integration with physical systems, and improvements in training simulations. This work provides researchers and practitioners with a structured overview of the current state and future prospects of LLM-based agents in medicine. 

**Abstract (ZH)**: 大型语言模型（LLMs）正在通过开发基于LLM的代理来理解和处理医学任务，从而改变医疗领域。本文综述了基于LLM的医疗代理，对其架构、应用和挑战进行了全面的审查。我们分析了医疗代理系统的关键组成部分，包括系统特征、临床规划机制、医疗推理框架以及外部能力增强。综述涵盖了重大应用场景，如临床决策支持、医学记录、培训模拟和医疗服务质量优化。我们讨论了用于评估这些代理在医疗环境中的性能的评估框架和指标。尽管基于LLM的代理在提升医疗服务方面展现出潜力，但仍存在多项挑战，包括幻觉管理、多模态集成、实施障碍和伦理问题。综述还指出了未来的研究方向，包括受近期LLM架构发展启发的医疗推理进步、与物理系统的集成以及培训模拟的改进。该工作为研究人员和实践者提供了关于基于LLM的代理在医学领域的当前状态和未来前景的结构化综述。 

---
# Bridging the Gap: Enabling Natural Language Queries for NoSQL Databases through Text-to-NoSQL Translation 

**Title (ZH)**: 缩小差距：通过文本到NoSQL数据库转换实现自然语言查询 

**Authors**: Jinwei Lu, Yuanfeng Song, Zhiqian Qin, Haodi Zhang, Chen Zhang, Raymond Chi-Wing Wong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11201)  

**Abstract**: NoSQL databases have become increasingly popular due to their outstanding performance in handling large-scale, unstructured, and semi-structured data, highlighting the need for user-friendly interfaces to bridge the gap between non-technical users and complex database queries. In this paper, we introduce the Text-to-NoSQL task, which aims to convert natural language queries into NoSQL queries, thereby lowering the technical barrier for non-expert users. To promote research in this area, we developed a novel automated dataset construction process and released a large-scale and open-source dataset for this task, named TEND (short for Text-to-NoSQL Dataset). Additionally, we designed a SLM (Small Language Model)-assisted and RAG (Retrieval-augmented Generation)-assisted multi-step framework called SMART, which is specifically designed for Text-to-NoSQL conversion. To ensure comprehensive evaluation of the models, we also introduced a detailed set of metrics that assess the model's performance from both the query itself and its execution results. Our experimental results demonstrate the effectiveness of our approach and establish a benchmark for future research in this emerging field. We believe that our contributions will pave the way for more accessible and intuitive interactions with NoSQL databases. 

**Abstract (ZH)**: NoSQL数据库由于在处理大规模、非结构化和半结构化数据方面的出色性能，已经变得越来越受欢迎，凸显了用户友好界面的重要性，以便非技术人员能够更容易地理解和使用复杂的数据库查询。在这篇论文中，我们介绍了将自然语言查询转换为NoSQL查询的Text-to-NoSQL任务，旨在降低非专家用户的技术门槛。为了促进该领域的研究，我们开发了一种新颖的自动化数据集构建过程，并发布了包含大量数据且开源的Text-to-NoSQL数据集TEND。此外，我们设计了一种名为SMART（Small Language Model辅助和Retrieval-augmented Generation辅助多步框架）的框架，专门用于实现Text-to-NoSQL转换。为了全面评估模型性能，我们还引入了一套详细的评估指标，该指标不仅评估查询本身，还评估其执行结果。实验结果表明了我们方法的有效性，并为未来在这一新兴领域的研究建立了基准。我们相信，我们的贡献将为NoSQL数据库的更易于使用和直观交互铺平道路。 

---
# How Do LLMs Acquire New Knowledge? A Knowledge Circuits Perspective on Continual Pre-Training 

**Title (ZH)**: LLMs如何获取新知识？持续预训练的知识电路视角 

**Authors**: Yixin Ou, Yunzhi Yao, Ningyu Zhang, Hui Jin, Jiacheng Sun, Shumin Deng, Zhenguo Li, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11196)  

**Abstract**: Despite exceptional capabilities in knowledge-intensive tasks, Large Language Models (LLMs) face a critical gap in understanding how they internalize new knowledge, particularly how to structurally embed acquired knowledge in their neural computations. We address this issue through the lens of knowledge circuit evolution, identifying computational subgraphs that facilitate knowledge storage and processing. Our systematic analysis of circuit evolution throughout continual pre-training reveals several key findings: (1) the acquisition of new knowledge is influenced by its relevance to pre-existing knowledge; (2) the evolution of knowledge circuits exhibits a distinct phase shift from formation to optimization; (3) the evolution of knowledge circuits follows a deep-to-shallow pattern. These insights not only advance our theoretical understanding of the mechanisms of new knowledge acquisition in LLMs, but also provide potential implications for improving continual pre-training strategies to enhance model performance. Code and data will be available at this https URL. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在知识密集型任务中表现出色，但在理解它们是如何内化新知识，特别是如何在神经计算中结构化地嵌入新知识方面，仍面临一个关键的差距。我们通过知识电路演化的视角来解决这一问题，识别出促进知识存储和处理的计算子图。我们对持续预训练过程中电路演化的系统分析揭示了几个关键发现：（1）新知识的获取受到先存知识相关性的影响；（2）知识电路的演化表现出从形成到优化的阶段性转变；（3）知识电路的演化遵循从深层到浅层的模式。这些洞察不仅有助于深化我们对LLMs中新知识获取机制的理解，还为改进持续预训练策略以提升模型性能提供了潜在的启示。相关代码和数据可在以下网址获取：[该网址]。 

---
# From Deception to Perception: The Surprising Benefits of Deepfakes for Detecting, Measuring, and Mitigating Bias 

**Title (ZH)**: 从欺骗到感知：深度合成内容在检测、衡量和减轻偏见方面的意外益处 

**Authors**: Yizhi Liu, Balaji Padmanabhan, Siva Viswanathan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11195)  

**Abstract**: While deepfake technologies have predominantly been criticized for potential misuse, our study demonstrates their significant potential as tools for detecting, measuring, and mitigating biases in key societal domains. By employing deepfake technology to generate controlled facial images, we extend the scope of traditional correspondence studies beyond mere textual manipulations. This enhancement is crucial in scenarios such as pain assessments, where subjective biases triggered by sensitive features in facial images can profoundly affect outcomes. Our results reveal that deepfakes not only maintain the effectiveness of correspondence studies but also introduce groundbreaking advancements in bias measurement and correction techniques. This study emphasizes the constructive role of deepfake technologies as essential tools for advancing societal equity and fairness. 

**Abstract (ZH)**: 尽管深度伪造技术主要受到潜在滥用的批评，但我们的研究表明，这些技术具有在关键社会领域检测、衡量和缓解偏见的巨大潜力。通过使用深度伪造技术生成受控的人脸图像，我们将传统对应研究的范围扩展到了不仅是文本操控。这种扩展在疼痛评估等场景中尤为重要，在这些场景中，敏感人脸特征引发的主观偏见可以严重影响结果。研究结果表明，深度伪造不仅保持了对应研究的有效性，还引入了在偏见测量和修正技术方面的革命性进展。本研究强调了深度伪造技术在促进社会公平和平等方面的重要作用。 

---
# Primus: A Pioneering Collection of Open-Source Datasets for Cybersecurity LLM Training 

**Title (ZH)**: Primus：用于网络安全大规模语言模型训练的开源数据集先驱集锦 

**Authors**: Yao-Ching Yu, Tsun-Han Chiang, Cheng-Wei Tsai, Chien-Ming Huang, Wen-Kwang Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2502.11191)  

**Abstract**: Large Language Models (LLMs) have shown remarkable advancements in specialized fields such as finance, law, and medicine. However, in cybersecurity, we have noticed a lack of open-source datasets, with a particular lack of high-quality cybersecurity pretraining corpora, even though much research indicates that LLMs acquire their knowledge during pretraining. To address this, we present a comprehensive suite of datasets covering all major training stages, including pretraining, instruction fine-tuning, and reasoning distillation with cybersecurity-specific self-reflection data. Extensive ablation studies demonstrate their effectiveness on public cybersecurity benchmarks. In particular, continual pre-training on our dataset yields a 15.88% improvement in the aggregate score, while reasoning distillation leads to a 10% gain in security certification (CISSP). We will release all datasets and trained cybersecurity LLMs under the ODC-BY and MIT licenses to encourage further research in the community. For access to all datasets and model weights, please refer to this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在金融、法律和医学等专业领域展示出了显著的进步。然而，在网络安全领域，我们注意到缺乏公开的数据集，尤其是高质量的网络安全预训练语料库。尽管许多研究表明，LLMs在其预训练过程中会获取知识，但在网络安全领域这一情况更为明显。为解决这一问题，我们提出了一整套涵盖所有主要训练阶段的数据集，包括预训练、指令微调和基于网络安全专项自我反思数据的推理精炼。广泛的消融研究显示，这些数据集在公共网络安全基准测试中的有效性。特别是，连续使用我们的数据集进行预训练，在综合评分上提升了15.88%，而推理精炼则在CISSP安全认证方面带来了10%的提升。我们将根据ODC-BY和MIT许可协议发布所有数据集和训练好的网络安全LLM，以促进社区进一步的研究。如需访问所有数据集和模型权重，请参阅此链接：[此处链接]。 

---
# ReLearn: Unlearning via Learning for Large Language Models 

**Title (ZH)**: ReLearn：通过学习实现大型语言模型的遗忘 

**Authors**: Haoming Xu, Ningyuan Zhao, Liming Yang, Sendong Zhao, Shumin Deng, Mengru Wang, Bryan Hooi, Nay Oo, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11190)  

**Abstract**: Current unlearning methods for large language models usually rely on reverse optimization to reduce target token probabilities. However, this paradigm disrupts the subsequent tokens prediction, degrading model performance and linguistic coherence. Moreover, existing evaluation metrics overemphasize contextual forgetting while inadequately assessing response fluency and relevance. To address these challenges, we propose ReLearn, a data augmentation and fine-tuning pipeline for effective unlearning, along with a comprehensive evaluation framework. This framework introduces Knowledge Forgetting Rate (KFR) and Knowledge Retention Rate (KRR) to measure knowledge-level preservation, and Linguistic Score (LS) to evaluate generation quality. Our experiments show that ReLearn successfully achieves targeted forgetting while preserving high-quality output. Through mechanistic analysis, we further demonstrate how reverse optimization disrupts coherent text generation, while ReLearn preserves this essential capability. Code is available at this https URL. 

**Abstract (ZH)**: 以下是翻译后的学术规范内容：

当前的大语言模型去学习方法通常依赖于反向优化来降低目标标记的概率。然而，这种范式会破坏后续标记的预测，降低模型性能和语言连贯性。此外，现有的评估指标虽然强调了上下文遗忘，但在评估响应流畅性和相关性方面考虑不足。为了解决这些挑战，我们提出了一种名为 ReLearn 的数据增强和微调管道，以实现有效的去学习，并建立了一个全面的评估框架。该框架引入了知识遗忘率 (KFR) 和知识保留率 (KRR) 以衡量知识级别的保存情况，并引入了语言评分 (LS) 以评估生成质量。我们的实验表明，ReLearn 能够在保持高质量输出的同时实现有针对性的遗忘。通过机制分析，我们进一步展示了反向优化如何破坏连贯性文本生成，而 ReLearn 保留了这一关键能力。源代码可在以下网址获取：this https URL。 

---
# TituLLMs: A Family of Bangla LLMs with Comprehensive Benchmarking 

**Title (ZH)**: TituLLMs：一系列全面benchmark测试的孟加拉语大规模语言模型 

**Authors**: Shahriar Kabir Nahin, Rabindra Nath Nandi, Sagor Sarker, Quazi Sarwar Muhtaseem, Md Kowsher, Apu Chandraw Shill, Md Ibrahim, Mehadi Hasan Menon, Tareq Al Muntasir, Firoj Alam  

**Link**: [PDF](https://arxiv.org/pdf/2502.11187)  

**Abstract**: In this paper, we present TituLLMs, the first large pretrained Bangla LLMs, available in 1B and 3B parameter sizes. Due to computational constraints during both training and inference, we focused on smaller models. To train TituLLMs, we collected a pretraining dataset of approximately 37 billion tokens. We extended the Llama-3.2 tokenizer to incorporate language- and culture-specific knowledge, which also enables faster training and inference. There was a lack of benchmarking datasets to evaluate LLMs for Bangla. To address this gap, we developed five benchmarking datasets. We benchmarked various LLMs, including TituLLMs, and demonstrated that TituLLMs outperforms its initial multilingual versions. However, this is not always the case, highlighting the complexities of language adaptation. Our work lays the groundwork for adapting existing multilingual open models to other low-resource languages. To facilitate broader adoption and further research, we have made the TituLLMs models and benchmarking datasets publicly available (this https URL). 

**Abstract (ZH)**: 在本文中，我们提出了TituLLMs，这是第一个大型预训练孟加拉语语言模型，有1亿参数和3亿参数两种版本。由于在训练和推理过程中都受到计算能力的限制，我们主要关注较小的模型。为了训练TituLLMs，我们收集了一个包含约370亿个令牌的预训练数据集。我们扩展了Llama-3.2分词器，使其能够融入特定语言和文化的知识，这也有助于加快训练和推理速度。对于孟加拉语而言，缺乏用于评估语言模型基准的数据集，为解决这一问题，我们开发了五个基准数据集。我们对包括TituLLMs在内的多种语言模型进行了基准测试，并证明了TituLLMs的表现优于其最初的多语言版本。然而，这并不总是成立，突显了语言适应性的复杂性。我们的工作为现有大规模多语言开源模型适应其他资源匮乏语言奠定了基础。为了促进更广泛的应用和进一步研究，我们已经将TituLLMs模型和基准数据集公开发布（this https URL）。 

---
# Can't See the Forest for the Trees: Benchmarking Multimodal Safety Awareness for Multimodal LLMs 

**Title (ZH)**: 只见树木不见森林：多模态安全意识评估 for 多模态大语言模型

在这个标题翻译中，我尝试保留了原文的核心含义，同时使表达更为自然流畅。原文标题“Can't See the Forest for the Trees”是一个英语成语，字面意思是“只见树木不见森林”，用来比喻只注意细节而忽视了整体或全局。在翻译时保留了这个成语，以保持原文的修辞风格，同时提供了相应的解释，使中文读者能够理解其含义。 

**Authors**: Wenxuan Wang, Xiaoyuan Liu, Kuiyi Gao, Jen-tse Huang, Youliang Yuan, Pinjia He, Shuai Wang, Zhaopeng Tu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11184)  

**Abstract**: Multimodal Large Language Models (MLLMs) have expanded the capabilities of traditional language models by enabling interaction through both text and images. However, ensuring the safety of these models remains a significant challenge, particularly in accurately identifying whether multimodal content is safe or unsafe-a capability we term safety awareness. In this paper, we introduce MMSafeAware, the first comprehensive multimodal safety awareness benchmark designed to evaluate MLLMs across 29 safety scenarios with 1500 carefully curated image-prompt pairs. MMSafeAware includes both unsafe and over-safety subsets to assess models abilities to correctly identify unsafe content and avoid over-sensitivity that can hinder helpfulness. Evaluating nine widely used MLLMs using MMSafeAware reveals that current models are not sufficiently safe and often overly sensitive; for example, GPT-4V misclassifies 36.1% of unsafe inputs as safe and 59.9% of benign inputs as unsafe. We further explore three methods to improve safety awareness-prompting-based approaches, visual contrastive decoding, and vision-centric reasoning fine-tuning-but find that none achieve satisfactory performance. Our findings highlight the profound challenges in developing MLLMs with robust safety awareness, underscoring the need for further research in this area. All the code and data will be publicly available to facilitate future research. 

**Abstract (ZH)**: 多模态大型语言模型（MLLMs）通过使模型能够通过文本和图像进行交互，从而扩展了传统语言模型的功能。然而，确保这些模型的安全性仍然是一个重大挑战，尤其是在准确识别多模态内容是否安全方面——我们把这个能力称为安全意识。在本文中，我们介绍了MMSafeAware，这是第一个全面的多模态安全意识基准，旨在通过1500个精心策划的图像-提示对来评估MLLMs在29种安全场景中的性能。MMSafeAware包括不安全和过度安全的子集，用于评估模型正确识别不安全内容并避免过度敏感（这可能妨碍其有用性）的能力。利用MMSafeAware评估九种广泛使用的MLLMs发现，当前的模型并不足够安全，而且经常表现出过度敏感；例如，GPT-4V将36.1%的不安全输入错误分类为安全输入，并将59.9%的良性输入错误分类为不安全输入。我们进一步探索了三种提高安全意识的方法——基于提示的方法、视觉对比解码以及视觉为中心的推理微调，但发现这些方法均未达到满意的性能。我们的发现突显了在开发具有稳健安全意识的MLLMs方面面临的巨大挑战，强调了在这一领域进行进一步研究的必要性。所有代码和数据都将公开，以便促进未来的研究。 

---
# Improving Scientific Document Retrieval with Concept Coverage-based Query Set Generation 

**Title (ZH)**: 基于概念覆盖面的查询集生成以改进科学文献检索 

**Authors**: SeongKu Kang, Bowen Jin, Wonbin Kweon, Yu Zhang, Dongha Lee, Jiawei Han, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11181)  

**Abstract**: In specialized fields like the scientific domain, constructing large-scale human-annotated datasets poses a significant challenge due to the need for domain expertise. Recent methods have employed large language models to generate synthetic queries, which serve as proxies for actual user queries. However, they lack control over the content generated, often resulting in incomplete coverage of academic concepts in documents. We introduce Concept Coverage-based Query set Generation (CCQGen) framework, designed to generate a set of queries with comprehensive coverage of the document's concepts. A key distinction of CCQGen is that it adaptively adjusts the generation process based on the previously generated queries. We identify concepts not sufficiently covered by previous queries, and leverage them as conditions for subsequent query generation. This approach guides each new query to complement the previous ones, aiding in a thorough understanding of the document. Extensive experiments demonstrate that CCQGen significantly enhances query quality and retrieval performance. 

**Abstract (ZH)**: 在诸如科学领域这样专门化的领域中，构建大规模的人标注数据集面临着显著挑战，因为这需要特定领域的专业知识。近年来的方法利用大型语言模型生成合成查询，这些查询作为实际用户查询的代理。然而，这种方法在控制生成内容方面能力有限，往往导致学术概念在文档中的覆盖不完整。我们引入了一种基于概念覆盖的查询集生成（CCQGen）框架，旨在生成一组能够全面覆盖文档概念的查询集。CCQGen 的一个关键特点在于，它能够根据之前生成的查询自适应地调整生成过程。我们识别出之前查询未充分覆盖的概念，并将其作为后续查询生成的条件。这种方法指导每个新查询补充之前的查询，有助于全面理解文档。实验结果表明，CCQGen 显著提高了查询质量和检索性能。 

---
# RT-DEMT: A hybrid real-time acupoint detection model combining mamba and transformer 

**Title (ZH)**: RT-DEMT：一种结合Mamba和Transformer的 Hybrid 实时针灸穴位检测模型 

**Authors**: Shilong Yang, Qi Zang, Chulong Zhang, Lingfeng Huang, Yaoqin Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.11179)  

**Abstract**: Traditional Chinese acupuncture methods often face controversy in clinical practice due to their high subjectivity. Additionally, current intelligent-assisted acupuncture systems have two major limitations: slow acupoint localization speed and low accuracy. To address these limitations, a new method leverages the excellent inference efficiency of the state-space model Mamba, while retaining the advantages of the attention mechanism in the traditional DETR architecture, to achieve efficient global information integration and provide high-quality feature information for acupoint localization tasks. Furthermore, by employing the concept of residual likelihood estimation, it eliminates the need for complex upsampling processes, thereby accelerating the acupoint localization task. Our method achieved state-of-the-art (SOTA) accuracy on a private dataset of acupoints on the human back, with an average Euclidean distance pixel error (EPE) of 7.792 and an average time consumption of 10.05 milliseconds per localization task. Compared to the second-best algorithm, our method improved both accuracy and speed by approximately 14\%. This significant advancement not only enhances the efficacy of acupuncture treatment but also demonstrates the commercial potential of automated acupuncture robot systems. Access to our method is available at this https URL 

**Abstract (ZH)**: 中医传统针灸方法在临床实践中往往面临争议，主要是因为其高度的主观性。此外，当前的智能辅助针灸系统还存在两大限制：针灸穴位定位速度缓慢和准确性较低。为解决这些问题，本研究提出了一种新的方法，该方法利用了状态空间模型Mamba的优秀推断效率，同时保留了传统DETR架构中注意力机制的优势，从而实现了高效的整体信息集成，并为针灸穴位定位任务提供了高品质的特征信息。此外，通过引入残差似然估计的概念，该方法消除了复杂上采样过程的需要，从而加速了针灸穴位定位任务。在一个人体背部穴位的私有数据集上，我们的方法达到了最先进的（SOTA）准确度，平均欧式距离像素误差（EPE）为7.792，每次定位任务的平均时间消耗为10.05毫秒。与第二好的算法相比，我们的方法在准确性和速度上分别 Improvement approximately 14%。这一显著进展不仅提高了针灸治疗的有效性，还展示了自动化针灸机器人系统的商业潜力。我们的方法可通过以下链接访问：[该链接处应填写实际链接] 

---
# Knowing Your Target: Target-Aware Transformer Makes Better Spatio-Temporal Video Grounding 

**Title (ZH)**: 了解目标：目标-aware 视觉变压器实现更好的时空视频目标定位 

**Authors**: Xin Gu, Yaojie Shen, Chenxi Luo, Tiejian Luo, Yan Huang, Yuewei Lin, Heng Fan, Libo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11168)  

**Abstract**: Transformer has attracted increasing interest in STVG, owing to its end-to-end pipeline and promising result. Existing Transformer-based STVG approaches often leverage a set of object queries, which are initialized simply using zeros and then gradually learn target position information via iterative interactions with multimodal features, for spatial and temporal localization. Despite simplicity, these zero object queries, due to lacking target-specific cues, are hard to learn discriminative target information from interactions with multimodal features in complicated scenarios (\e.g., with distractors or occlusion), resulting in degradation. Addressing this, we introduce a novel Target-Aware Transformer for STVG (TA-STVG), which seeks to adaptively generate object queries via exploring target-specific cues from the given video-text pair, for improving STVG. The key lies in two simple yet effective modules, comprising text-guided temporal sampling (TTS) and attribute-aware spatial activation (ASA), working in a cascade. The former focuses on selecting target-relevant temporal cues from a video utilizing holistic text information, while the latter aims at further exploiting the fine-grained visual attribute information of the object from previous target-aware temporal cues, which is applied for object query initialization. Compared to existing methods leveraging zero-initialized queries, object queries in our TA-STVG, directly generated from a given video-text pair, naturally carry target-specific cues, making them adaptive and better interact with multimodal features for learning more discriminative information to improve STVG. In our experiments on three benchmarks, TA-STVG achieves state-of-the-art performance and significantly outperforms the baseline, validating its efficacy. 

**Abstract (ZH)**: Transformer 在时空视觉定位与生成（STVG）领域引起了越来越多的关注，这得益于其端到端的处理流程和出色的性能。现有的基于Transformer的STVG方法常常采用一组对象查询，这些查询初始化为零，并通过与多模态特征的迭代交互逐渐学习目标的位置信息，以实现空间和时间上的定位。尽管如此，这些零初始化的对象查询由于缺乏目标特异性线索，在复杂场景下（例如存在干扰或遮挡时）难以通过与多模态特征的交互学习到有效的目标信息，导致性能退化。为了解决这一问题，我们提出了一个新颖的目标感知Transformer（TA-STVG），旨在通过探索给定视频-文本对中目标特异性线索来自适应生成对象查询，从而提升STVG。其关键在于两个简单而有效的模块，包括文本引导的时间抽样（TTS）和属性感知的空间激活（ASA），这两个模块依次工作。TTS主要集中在利用整体文本信息从视频中选择与目标相关的时间线索，而ASA则致力于进一步从先前的目标感知时间线索中挖掘对象的精细视觉属性信息，用于对象查询的初始化。与现有利用零初始化查询的方法相比，我们的TA-STVG直接从给定的视频-文本对中生成的对象查询自然携带目标特异性线索，使其更加适应与多模态特征的交互，从而学习到更多的判别性信息以提升STVG。在三个基准上的实验表明，TA-STVG取得了最先进的性能并显著超越了基线模型，验证了其有效性。 

---
# Large Language-Geometry Model: When LLM meets Equivariance 

**Title (ZH)**: 大型语言-几何模型：当大规模语言模型遇到等变性 

**Authors**: Zongzhao Li, Jiacheng Cen, Bing Su, Wenbing Huang, Tingyang Xu, Yu Rong, Deli Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.11149)  

**Abstract**: Accurately predicting 3D structures and dynamics of physical systems is crucial in scientific applications. Existing approaches that rely on geometric Graph Neural Networks (GNNs) effectively enforce $\mathrm{E}(3)$-equivariance, but they often fall in leveraging extensive broader information. While direct application of Large Language Models (LLMs) can incorporate external knowledge, they lack the capability for spatial reasoning with guaranteed equivariance. In this paper, we propose EquiLLM, a novel framework for representing 3D physical systems that seamlessly integrates E(3)-equivariance with LLM capabilities. Specifically, EquiLLM comprises four key components: geometry-aware prompting, an equivariant encoder, an LLM, and an equivariant adaptor. Essentially, the LLM guided by the instructive prompt serves as a sophisticated invariant feature processor, while 3D directional information is exclusively handled by the equivariant encoder and adaptor modules. Experimental results demonstrate that EquiLLM delivers significant improvements over previous methods across molecular dynamics simulation, human motion simulation, and antibody design, highlighting its promising generalizability. 

**Abstract (ZH)**: 准确预测物理系统的三维结构和动态在科学研究中至关重要。现有的依赖几何图神经网络（GNNs）的方法能够有效强制 $\mathrm{E}(3)$-酉变性，但它们往往未能充分利用广泛的信息。虽然直接应用大型语言模型（LLMs）可以整合外部知识，但它们缺乏保证酉变性的空间推理能力。在本文中，我们提出了一种新的框架EquiLLM，该框架能够无缝集成 $\mathrm{E}(3)$-酉变性与LLM的能力。具体而言，EquiLLM 包含四个关键组件：几何感知提示、酉变编码器、LLM 和酉变适配器模块。通过指令性的提示引导的LLM 作为复杂的不变特征处理器，而3D方向信息则仅由酉变编码器和适配器模块处理。实验结果表明，EquiLLM 在分子动力学模拟、人体运动模拟和抗体设计等方面显著优于先前的方法，并突出显示了其潜在的良好泛化能力。 

---
# Efficient Long-Decoding Inference with Reasoning-Aware Attention Sparsity 

**Title (ZH)**: 高效长时解码推理与推理意识注意力稀疏性 

**Authors**: Junhao Hu, Wenrui Huang, Weidong Wang, Zhenwen Li, Tiancheng Hu, Zhixia Liu, Xusheng Chen, Tao Xie, Yizhou Shan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11147)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities across various domains, with recent advancements in challenging reasoning tasks such as mathematics and programming. However, solving reasoning tasks often requires long decoding chains (of thoughts), which incur $O(N)$ time and memory consumption, where $N$ is the chain length. To mitigate $O(N)$ time and memory consumption, existing sparsity-based algorithms propose retaining only the most critical token's intermediate data (i.e., key-value cache) and discarding the rest. However, these existing algorithms struggle with the ``impossible trinity'' of accuracy, time, and memory. For example, the state-of-the-art algorithm, Quest, achieves high accuracy with $O(L)$ time but $O(N)$ memory ($L$ is the cache budget, $L \ll N$). To address this issue, in this paper, we identify a new attention pattern during the decode stage of reasoning tasks, where milestone tokens (analogous to lemmas in mathematical proofs) emerge, are utilized, and then become unimportant afterward. Based on this pattern, we propose a new algorithm named RaaS that identifies and retains milestone tokens only until they are no longer needed, achieving high accuracy with $O(L)$ time and $O(L)$ memory complexity. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各个领域展现了强大的能力，尤其是在数学和编程等具有挑战性的推理任务中取得了显著进步。然而，解决推理任务通常需要长的解码链（思考链），这导致了$O(N)$的时间和内存消耗，其中$N$为链的长度。为了减少$O(N)$的时间和内存消耗，现有的稀疏性算法建议仅保留最关键的标记的中间数据（即键值缓存），而丢弃其余的数据。然而，现有的算法在准确率、时间和内存之间面临“不可能三角”。例如，最先进的算法Quest能够在$O(L)$时间内达到高准确率，但需要$O(N)$的内存（$L$是缓存预算，$L \ll N$）。为了应对这一问题，在本文中，我们识别了一种新的推理任务解码阶段的注意力模式，其中标型标记（类似于数学证明中的引理）在使用后变得不再重要。基于这一模式，我们提出了一种名为RaaS的新算法，该算法仅在不需要时就识别并保留标型标记，从而能够以$O(L)$的时间复杂度和$O(L)$的内存复杂度实现高准确率。 

---
# Cognitive Neural Architecture Search Reveals Hierarchical Entailment 

**Title (ZH)**: 认知神经架构搜索揭示层次蕴含关系 

**Authors**: Lukas Kuhn, Sari Saba-Sadiya, Gemma Roig  

**Link**: [PDF](https://arxiv.org/pdf/2502.11141)  

**Abstract**: Recent research has suggested that the brain is more shallow than previously thought, challenging the traditionally assumed hierarchical structure of the ventral visual pathway. Here, we demonstrate that optimizing convolutional network architectures for brain-alignment via evolutionary neural architecture search results in models with clear representational hierarchies. Despite having random weights, the identified models achieve brain-alignment scores surpassing even those of pretrained classification models - as measured by both regression and representational similarity analysis. Furthermore, through traditional supervised training, architectures optimized for alignment with late ventral regions become competitive classification models. These findings suggest that hierarchical structure is a fundamental mechanism of primate visual processing. Finally, this work demonstrates the potential of neural architecture search as a framework for computational cognitive neuroscience research that could reduce the field's reliance on manually designed convolutional networks. 

**Abstract (ZH)**: 最近的研究表明，大脑可能比先前认为的要更为扁平，这挑战了传统上假设的腹侧视觉路径的分层结构。在这里，我们通过进化神经架构搜索优化卷积网络架构以实现大脑对齐，展示了具有清晰表征层次结构的模型。尽管这些模型的权重是随机的，但它们的大脑对齐分数甚至超过了预训练分类模型的分数——这一结果通过回归分析和表征相似性分析得到了证实。此外，通过传统的监督训练，优化为与腹侧晚期区域对齐的架构能够成为竞争力的分类模型。这些发现表明，分层结构可能是灵长类动物视觉处理的基本机制。最后，本研究表明神经架构搜索框架在计算认知神经科学领域具有潜在的应用价值，这可以减少对手动设计卷积网络的依赖。 

---
# VisPath: Automated Visualization Code Synthesis via Multi-Path Reasoning and Feedback-Driven Optimization 

**Title (ZH)**: VisPath：通过多路径推理和反馈驱动优化的自动化可视化代码合成 

**Authors**: Wonduk Seo, Seungyong Lee, Daye Kang, Zonghao Yuan, Seunghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.11140)  

**Abstract**: Unprecedented breakthroughs in Large Language Models (LLMs) has amplified its penetration into application of automated visualization code generation. Few-shot prompting and query expansion techniques have notably enhanced data visualization performance, however, still fail to overcome ambiguity and complexity of natural language queries - imposing an inherent burden for manual human intervention. To mitigate such limitations, we propose a holistic framework VisPath : A Multi-Path Reasoning and Feedback-Driven Optimization Framework for Visualization Code Generation, which systematically enhances code quality through structured reasoning and refinement. VisPath is a multi-stage framework, specially designed to handle underspecified queries. To generate a robust final visualization code, it first utilizes initial query to generate diverse reformulated queries via Chain-of-Thought (CoT) prompting, each representing a distinct reasoning path. Refined queries are used to produce candidate visualization scripts, consequently executed to generate multiple images. Comprehensively assessing correctness and quality of outputs, VisPath generates feedback for each image, which are then fed to aggregation module to generate optimal result. Extensive experiments on benchmarks including MatPlotBench and the Qwen-Agent Code Interpreter Benchmark show that VisPath significantly outperforms state-of-the-art (SOTA) methods, increased up to average 17%, offering a more reliable solution for AI-driven visualization code generation. 

**Abstract (ZH)**: 大型语言模型（LLMs）前所未有的突破性进展使其在自动化可视化代码生成中的应用更加广泛。极少样本提示和查询扩展技术显著提升了数据可视化性能，但仍然无法克服自然语言查询的模糊性和复杂性，这给人工干预带来了固有的负担。为了减轻这些限制，我们提出了一种综合框架 VisPath：一种多路径推理和反馈驱动优化框架，该框架通过结构化的推理和细化系统地提升代码质量。VisPath 是一个多阶段框架，特别设计用于处理含糊不清的查询。为了生成稳健的最终可视化代码，它首先利用初始查询通过“思维链”（CoT）提示生成多种多样重新表述的查询，每个查询代表一条不同的推理路径。细化后的查询用于生成候选可视化脚本，这些脚本随后被执行以生成多张图像。综合评估输出的正确性和质量后，VisPath 为每张图像生成反馈，这些反馈随后被馈送到聚合模块以生成最佳结果。通过对包括 MatPlotBench 和 Qwen-Agent Code Interpreter Benchmark 在内的基准测试的广泛实验表明，VisPath 显著优于现有最佳方法（SOTA），平均提高了约 17%，为以人工智能驱动的可视化代码生成提供了更可靠的方法。 

---
# Safety Evaluation of DeepSeek Models in Chinese Contexts 

**Title (ZH)**: 在中国语境下的DeepSeek模型安全性评估 

**Authors**: Wenjing Zhang, Xuejiao Lei, Zhaoxiang Liu, Ning Wang, Zhenhong Long, Peijun Yang, Jiaojiao Zhao, Minjie Hua, Chaoyang Ma, Kai Wang, Shiguo Lian  

**Link**: [PDF](https://arxiv.org/pdf/2502.11137)  

**Abstract**: Recently, the DeepSeek series of models, leveraging their exceptional reasoning capabilities and open-source strategy, is reshaping the global AI landscape. Despite these advantages, they exhibit significant safety deficiencies. Research conducted by Robust Intelligence, a subsidiary of Cisco, in collaboration with the University of Pennsylvania, revealed that DeepSeek-R1 has a 100\% attack success rate when processing harmful prompts. Additionally, multiple safety companies and research institutions have confirmed critical safety vulnerabilities in this model. As models demonstrating robust performance in Chinese and English, DeepSeek models require equally crucial safety assessments in both language contexts. However, current research has predominantly focused on safety evaluations in English environments, leaving a gap in comprehensive assessments of their safety performance in Chinese contexts. In response to this gap, this study introduces CHiSafetyBench, a Chinese-specific safety evaluation benchmark. This benchmark systematically evaluates the safety of DeepSeek-R1 and DeepSeek-V3 in Chinese contexts, revealing their performance across safety categories. The experimental results quantify the deficiencies of these two models in Chinese contexts, providing key insights for subsequent improvements. 

**Abstract (ZH)**: 近年来，DeepSeek 系列模型凭借其卓越的推理能力和开源策略，重塑了全球AI格局。尽管具备这些优势，它们在安全性方面仍然存在明显的缺陷。Robust Intelligence（思科的子公司）与宾夕法尼亚大学合作进行的研究显示，当处理有害提示时，DeepSeek-R1 的攻击成功率达到了100%。此外，多家安全公司和研究机构已经确认该模型存在严重的安全漏洞。作为在中英文环境下均表现出强大性能的模型，DeepSeek 模型同样需要在中英文两个语言环境中进行同等重要的安全评估。然而，目前的研究主要集中在英语环境下的安全性评价，对这些模型在中文环境下的安全性性能进行全面评估仍有欠缺。为弥补这一不足，本研究引入了 CHiSafetyBench，一个专门针对中文环境的安全性评估基准。该基准系统地评估了 DeepSeek-R1 和 DeepSeek-V3 在中文环境下的安全性，揭示了它们在各个安全性类别中的表现。实验结果量化了这两种模型在中文环境下的缺陷，为后续改进提供了关键性的见解。 

---
# UNITE-FND: Reframing Multimodal Fake News Detection through Unimodal Scene Translation 

**Title (ZH)**: UNITE-FND：通过单模态场景转换重新定义多模态假新闻检测 

**Authors**: Arka Mukherjee, Shreya Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2502.11132)  

**Abstract**: Multimodal fake news detection typically demands complex architectures and substantial computational resources, posing deployment challenges in real-world settings. We introduce UNITE-FND, a novel framework that reframes multimodal fake news detection as a unimodal text classification task. We propose six specialized prompting strategies with Gemini 1.5 Pro, converting visual content into structured textual descriptions, and enabling efficient text-only models to preserve critical visual information. To benchmark our approach, we introduce Uni-Fakeddit-55k, a curated dataset family of 55,000 samples each, each processed through our multimodal-to-unimodal translation framework. Experimental results demonstrate that UNITE-FND achieves 92.52% accuracy in binary classification, surpassing prior multimodal models while reducing computational costs by over 10x (TinyBERT variant: 14.5M parameters vs. 250M+ in SOTA models). Additionally, we propose a comprehensive suite of five novel metrics to evaluate image-to-text conversion quality, ensuring optimal information preservation. Our results demonstrate that structured text-based representations can replace direct multimodal processing with minimal loss of accuracy, making UNITE-FND a practical and scalable alternative for resource-constrained environments. 

**Abstract (ZH)**: 多模态假新闻检测通常需要复杂的架构和大量的计算资源，给实际应用部署带来了挑战。我们提出了UNITE-FND（统一多模态假新闻检测）框架，将多模态假新闻检测重新框架为单模态文本分类任务。我们利用Gemini 1.5 Pro提出了六种专门的提示策略，将视觉内容转换为结构化的文本描述，从而使高效的文字-only模型能够保留关键的视觉信息。为了对我们的方法进行基准测试，我们提出了Uni-Fakeddit-55k数据集系列，该系列包括55,000个样本，每个样本都通过我们的多模态到单模态翻译框架进行了处理。实验结果显示，UNITE-FND在二分类中的准确率达到92.52%，在保持显著更低成本（TinyBERT变体：14.5M参数，而现有最佳模型为2.5亿+参数）的同时，超过了之前的多模态模型。此外，我们还提出了一套全面的五项新评价指标，用于评估图像到文本转换的质量，确保信息的最佳保留。实验结果表明，结构化的文本表示可以在不显著损失准确性的情况下替代直接的多模态处理，使UNITE-FND成为资源受限环境中的实用和可扩展的替代方案。 

---
# AdaManip: Adaptive Articulated Object Manipulation Environments and Policy Learning 

**Title (ZH)**: AdaManip：自适应articulated对象操作环境与策略学习 

**Authors**: Yuanfei Wang, Xiaojie Zhang, Ruihai Wu, Yu Li, Yan Shen, Mingdong Wu, Zhaofeng He, Yizhou Wang, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.11124)  

**Abstract**: Articulated object manipulation is a critical capability for robots to perform various tasks in real-world scenarios. Composed of multiple parts connected by joints, articulated objects are endowed with diverse functional mechanisms through complex relative motions. For example, a safe consists of a door, a handle, and a lock, where the door can only be opened when the latch is unlocked. The internal structure, such as the state of a lock or joint angle constraints, cannot be directly observed from visual observation. Consequently, successful manipulation of these objects requires adaptive adjustment based on trial and error rather than a one-time visual inference. However, previous datasets and simulation environments for articulated objects have primarily focused on simple manipulation mechanisms where the complete manipulation process can be inferred from the object's appearance. To enhance the diversity and complexity of adaptive manipulation mechanisms, we build a novel articulated object manipulation environment and equip it with 9 categories of objects. Based on the environment and objects, we further propose an adaptive demonstration collection and 3D visual diffusion-based imitation learning pipeline that learns the adaptive manipulation policy. The effectiveness of our designs and proposed method is validated through both simulation and real-world experiments. Our project page is available at: this https URL 

**Abstract (ZH)**: 灵巧物体Manipulation是机器人在实际场景中执行各种任务的关键能力。由多个由关节连接的部分组成，灵巧物体通过复杂的相对运动获得多种功能机制。例如，一个保险柜包含门、把手和锁，只有当锁被解锁时，门才能打开。内部结构，如锁的状态或关节角度约束，通过视觉观察无法直接观察到。因此，成功操作这些物体需要基于试错的适应性调整，而不是一次性的视觉推理。然而，之前针对灵巧物体的数据集和仿真环境主要集中在简单的操作机制上，其中物体的完整操作过程可以从其外观中推断出来。为增强适应性操作机制的多样性和复杂性，我们构建了一个新的灵巧物体操作环境，并配备了9类物体。基于该环境和物体，我们进一步提出了一种适应性演示收集方法和基于三维视觉扩散的模仿学习管道，用于学习适应性操作策略。我们通过仿真和实际实验验证了我们设计和提出的方法的有效性。项目页面可访问以下网址：[点击此处](this https URL) 

---
# Knowledge Graph-Driven Retrieval-Augmented Generation: Integrating Deepseek-R1 with Weaviate for Advanced Chatbot Applications 

**Title (ZH)**: 知识图谱驱动的检索增强生成：将 Deepseek-R1 与 Weaviate 集成应用于高级聊天机器人应用 

**Authors**: Alexandru Lecu, Adrian Groza, Lezan Hawizy  

**Link**: [PDF](https://arxiv.org/pdf/2502.11108)  

**Abstract**: Large language models (LLMs) have significantly advanced the field of natural language generation. However, they frequently generate unverified outputs, which compromises their reliability in critical applications. In this study, we propose an innovative framework that combines structured biomedical knowledge with LLMs through a retrieval-augmented generation technique. Our system develops a thorough knowledge graph by identifying and refining causal relationships and named entities from medical abstracts related to age-related macular degeneration (AMD). Using a vector-based retrieval process and a locally deployed language model, our framework produces responses that are both contextually relevant and verifiable, with direct references to clinical evidence. Experimental results show that this method notably decreases hallucinations, enhances factual precision, and improves the clarity of generated responses, providing a robust solution for advanced biomedical chatbot applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言生成领域取得了显著进展。然而，它们经常生成未经验证的输出，这在关键应用中的可靠性受到了损害。本研究提出了一种创新框架，通过检索增强生成技术将结构化的生物医学知识与LLMs相结合。我们的系统通过识别和精炼与年龄相关黄斑变性（AMD）相关的医学摘要中的因果关系和命名实体，构建了一个详尽的知识图谱。使用基于向量的检索过程和本地部署的语言模型，该框架能够生成既相关又可验证的响应，并直接引用临床证据。实验结果表明，这种方法显著减少了幻觉现象，提高了事实精度，并提高了生成响应的清晰度，为先进的生物医学聊天机器人应用提供了稳健的解决方案。 

---
# Revisiting Weak-to-Strong Generalization in Theory and Practice: Reverse KL vs. Forward KL 

**Title (ZH)**: 理论与实践中的弱到强泛化 revisiting：反KL散度 vs. 正KL散度 

**Authors**: Wei Yao, Wenkai Yang, Ziqiao Wang, Yankai Lin, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11107)  

**Abstract**: As large language models advance toward superhuman performance, ensuring their alignment with human values and abilities grows increasingly complex. Weak-to-strong generalization offers a promising approach by leveraging predictions from weaker models to guide stronger systems, but its effectiveness could be constrained by the inherent noise and inaccuracies in these weak predictions. To address this, we propose a theoretically grounded approach that replaces forward KL divergence-whose mass-covering behavior risks overfitting to imperfect weak signals-with reverse KL divergence. Reverse KL divergence's zero-forcing effect prioritizes high-confidence predictions, effectively mitigating the influence of unreliable weak supervision. Theoretically, we extend existing bounds and derive tighter lower bounds for both forward and reverse KL divergence, establishing that reverse KL achieves at least comparable guarantees to forward KL. Notably, when a sufficiently pre-trained strong model is fine-tuned on the last layer, reverse KL uniquely guarantees that it outperforms its weak supervisor by the magnitude of their disagreement-a guarantee that forward KL cannot provide. Empirically, we demonstrate that reverse KL and reverse cross-entropy enable strong models to consistently outperform those trained with forward KL and standard cross-entropy across most settings, highlighting the practical advantages of these reverse losses. 

**Abstract (ZH)**: 随着大型语言模型的进步，确保其与人类价值观和能力保持一致变得越来越复杂。通过利用较弱模型的预测来引导较强系统，弱到强的一般泛化提供了一种有希望的方法，但其效果可能受到这些较弱预测中固有的噪声和不准确性的限制。为了解决这一问题，我们提出了一种理论指导的方法，用反向KL发散取代正向KL发散，正向KL发散的覆盖性质可能导致模型过度拟合到不完美的弱信号。反向KL发散的零强迫效应优先考虑高置信度预测，有效地减少了不可靠弱监督的影响。理论上，我们扩展了现有界，并为正向和反向KL发散推导出更紧的下界，证明了反向KL至少能够提供与正向KL相当的保证。值得注意的是，当一个充分预训练的较强模型在最后一层进行微调时，反向KL唯一地保证它在分歧的程度上优于其弱监督者——这正是正向KL无法提供的保证。实验上，我们证明了反向KL和反向交叉熵可以使较强模型在大多数设置中持续优于使用正向KL和标准交叉熵训练的模型，突显了这些反向损失的实际优势。 

---
# CacheFocus: Dynamic Cache Re-Positioning for Efficient Retrieval-Augmented Generation 

**Title (ZH)**: CacheFocus: 动态缓存重新定位以实现高效的检索增强生成 

**Authors**: Kun-Hui Lee, Eunhwan Park, Donghoon Han, Seung-Hoon Na  

**Link**: [PDF](https://arxiv.org/pdf/2502.11101)  

**Abstract**: Large Language Models (LLMs) excel across a variety of language tasks yet are constrained by limited input lengths and high computational costs. Existing approaches\textemdash such as relative positional encodings (e.g., RoPE, ALiBi) and sliding window mechanisms\textemdash partially alleviate these issues but often require additional training or suffer from performance degradation with longer inputs. In this paper, we introduce \textbf{\textit{CacheFocus}}, a method that enhances length normalization and reduces inference latency without any further training. Our approach leverages query-independent, offline caching to efficiently reuse a Context KV Cache Store. We address the amplification of abnormal token distributions problem by re-positioning cached keys and introducing Layer-Adaptive Cache Pruning to discard low-relevance caches during pre-filling. Additionally, our Adaptive Positional Allocation Strategy dynamically reassigns cache positions to maximize the use of the available positional encoding range. Experiments on the Natural Questions and TriviaQA datasets demonstrate that CacheFocus outperforms alternative methods even when inputs exceed the $4$K limit of the \texttt{LLaMA-2} model, emphasizing its practical effectiveness for long-context LLMs. Moreover, even with large maximum input length of \texttt{Qwen2}, the performance of CacheFocus shows that it maintains consistent performance even as the number of documents increases, effectively managing long-text generation without degradation. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种语言任务中表现出色，但受限于输入长度的限制和高计算成本。现有的方法——如相对位置编码（例如RoPE、ALiBi）和滑动窗口机制——在一定程度上缓解了这些问题，但往往需要额外的训练或在较长输入时导致性能下降。本文提出了一种名为**CacheFocus**的方法，该方法可以在无需进一步训练的情况下增强长度归一化并降低推理延迟。我们的方法利用独立于查询的离线缓存来高效地重复使用Context KV缓存存储。通过重新定位缓存的键并引入分层自适应缓存剪枝来处理异常token分布放大的问题，在预填充阶段丢弃低相关性的缓存。此外，自适应位置分配策略动态重新分配缓存位置，以最大化利用可用的位置编码范围。在Natural Questions和TriviaQA数据集上的实验表明，CacheFocus即使在输入超过LLaMA-2模型的4K限制时也优于其他方法，证明了其在长上下文LLMs中的实际有效性。即使在Qwen2的最大输入长度很大的情况下，CacheFocus的表现也保持一致，随着文档数量增加仍能有效管理长文本生成而不降低性能。 

---
# SyncSpeech: Low-Latency and Efficient Dual-Stream Text-to-Speech based on Temporal Masked Transformer 

**Title (ZH)**: SyncSpeech：基于时间掩蔽变换器的低延迟高效双流文本到语音技术 

**Authors**: Zhengyan Sheng, Zhihao Du, Shiliang Zhang, Zhijie Yan, Yexin Yang, Zhenhua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2502.11094)  

**Abstract**: This paper presents a dual-stream text-to-speech (TTS) model, SyncSpeech, capable of receiving streaming text input from upstream models while simultaneously generating streaming speech, facilitating seamless interaction with large language models. SyncSpeech has the following advantages: Low latency, as it begins generating streaming speech upon receiving the second text token; High efficiency, as it decodes all speech tokens corresponding to the each arrived text token in one step. To achieve this, we propose a temporal masked transformer as the backbone of SyncSpeech, combined with token-level duration prediction to predict speech tokens and the duration for the next step. Additionally, we design a two-stage training strategy to improve training efficiency and the quality of generated speech. We evaluated the SyncSpeech on both English and Mandarin datasets. Compared to the recent dual-stream TTS models, SyncSpeech significantly reduces the first packet delay of speech tokens and accelerates the real-time factor. Moreover, with the same data scale, SyncSpeech achieves performance comparable to that of traditional autoregressive-based TTS models in terms of both speech quality and robustness. Speech samples are available at this https URL}{this https URL. 

**Abstract (ZH)**: 本文介绍了能够同时接收上游模型的流式文本输入并生成流式语音的双流文本到语音（TTS）模型 SyncSpeech，从而与大型语言模型实现无缝交互。SyncSpeech 具有以下优势：低延迟，它在接收到第二个文本标记时就开始生成流式语音；高效性，它能够一步解码每个到达的文本标记对应的全部语音标记。为了实现这一目标，我们提出了一个基于时间掩蔽变换器的时间掩蔽变换器作为 SyncSpeech 的骨干结构，并结合标记级别持续时间预测来预测下一步骤的语音标记及其持续时间。此外，我们设计了一种两阶段训练策略，以提高训练效率和生成语音的质量。我们在英语和 Mandarin 数据集上评估了 SyncSpeech。与最近的双流 TTS 模型相比，SyncSpeech 显着减少了语音标记的首包延迟并加速了实时因子。此外，在相同的数据规模下，SyncSpeech 在语音质量和鲁棒性方面均达到了传统自回归式 TTS 模型的性能。语音样本请访问此链接：[此链接]和[此链接]。 

---
# SafeDialBench: A Fine-Grained Safety Benchmark for Large Language Models in Multi-Turn Dialogues with Diverse Jailbreak Attacks 

**Title (ZH)**: SafeDialBench：多轮对话中面对多样化脱管攻击的大语言模型细粒度安全基准 

**Authors**: Hongye Cao, Yanming Wang, Sijia Jing, Ziyue Peng, Zhixin Bai, Zhe Cao, Meng Fang, Fan Feng, Boyan Wang, Jiaheng Liu, Tianpei Yang, Jing Huo, Yang Gao, Fanyu Meng, Xi Yang, Chao Deng, Junlan Feng  

**Link**: [PDF](https://arxiv.org/pdf/2502.11090)  

**Abstract**: With the rapid advancement of Large Language Models (LLMs), the safety of LLMs has been a critical concern requiring precise assessment. Current benchmarks primarily concentrate on single-turn dialogues or a single jailbreak attack method to assess the safety. Additionally, these benchmarks have not taken into account the LLM's capability of identifying and handling unsafe information in detail. To address these issues, we propose a fine-grained benchmark SafeDialBench for evaluating the safety of LLMs across various jailbreak attacks in multi-turn dialogues. Specifically, we design a two-tier hierarchical safety taxonomy that considers 6 safety dimensions and generates more than 4000 multi-turn dialogues in both Chinese and English under 22 dialogue scenarios. We employ 7 jailbreak attack strategies, such as reference attack and purpose reverse, to enhance the dataset quality for dialogue generation. Notably, we construct an innovative assessment framework of LLMs, measuring capabilities in detecting, and handling unsafe information and maintaining consistency when facing jailbreak attacks. Experimental results across 17 LLMs reveal that Yi-34B-Chat and GLM4-9B-Chat demonstrate superior safety performance, while Llama3.1-8B-Instruct and o3-mini exhibit safety vulnerabilities. 

**Abstract (ZH)**: 在大型语言模型（LLMs）快速发展的背景下，LLMs的安全性已成为一个亟待精准评估的关键问题。现有的基准测试主要集中在单一回合对话或单一窃取攻击方法上进行安全性评估。此外，这些基准测试并未详细考虑LLMs识别和处理不安全信息的能力。为解决这些问题，我们提出了一种细粒度基准测试SafeDialBench，用于评估LLMs在多回合对话中面对各种窃取攻击时的安全性。具体而言，我们设计了一个两级层次的安全分类法，考虑了6个安全维度，并生成了超过4000个中英文双语的多回合对话，涵盖了22种对话场景。我们采用了包括引用攻击和目的逆转在内的7种窃取攻击策略，以提高对话生成的数据集质量。值得注意的是，我们构建了一种创新的LLMs评估框架，以衡量其检测和处理不安全信息以及在面对窃取攻击时保持一致性的能力。针对17种不同模型的实验结果显示，Yi-34B-Chat和GLM4-9B-Chat表现出更出色的安全性能，而Llama3.1-8B-Instruct和o3-mini则显示出安全性弱点。 

---
# Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention 

**Title (ZH)**: 本地稀疏注意力机制：硬件对齐且本征可训练的稀疏注意力机制 

**Authors**: Jingyang Yuan, Huazuo Gao, Damai Dai, Junyu Luo, Liang Zhao, Zhengyan Zhang, Zhenda Xie, Y. X. Wei, Lean Wang, Zhiping Xiao, Yuqing Wang, Chong Ruan, Ming Zhang, Wenfeng Liang, Wangding Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2502.11089)  

**Abstract**: Long-context modeling is crucial for next-generation language models, yet the high computational cost of standard attention mechanisms poses significant computational challenges. Sparse attention offers a promising direction for improving efficiency while maintaining model capabilities. We present NSA, a Natively trainable Sparse Attention mechanism that integrates algorithmic innovations with hardware-aligned optimizations to achieve efficient long-context modeling. NSA employs a dynamic hierarchical sparse strategy, combining coarse-grained token compression with fine-grained token selection to preserve both global context awareness and local precision. Our approach advances sparse attention design with two key innovations: (1) We achieve substantial speedups through arithmetic intensity-balanced algorithm design, with implementation optimizations for modern hardware. (2) We enable end-to-end training, reducing pretraining computation without sacrificing model performance. As shown in Figure 1, experiments show the model pretrained with NSA maintains or exceeds Full Attention models across general benchmarks, long-context tasks, and instruction-based reasoning. Meanwhile, NSA achieves substantial speedups over Full Attention on 64k-length sequences across decoding, forward propagation, and backward propagation, validating its efficiency throughout the model lifecycle. 

**Abstract (ZH)**: 长上下文建模对于下一代语言模型至关重要，但由于标准注意力机制的高计算成本，给计算带来了显著挑战。稀疏注意力为同时保持模型能力并提高效率提供了一个有希望的方向。我们提出了NSA，一种原生可训练的稀疏注意力机制，通过结合算法创新与硬件对齐的优化，实现高效的长上下文建模。NSA采用动态分层稀疏策略，结合粗粒度的令牌压缩与细粒度的令牌选择，以保持全局上下文感知和局部精度。我们的方法在稀疏注意力设计上采用了两项关键创新：（1）我们通过算术强度平衡的算法设计实现显著的速度提升，并针对现代硬件进行了实现优化。（2）我们实现了端到端训练，减少了预训练计算量而不牺牲模型性能。如图1所示，实验表明，使用NSA预训练的模型，在通用基准测试、长上下文任务和指令驱动推理方面，性能不低于全注意力模型。同时，NSA在64k长度序列的解码、正向传播和反向传播过程中，实现了显著的速度提升，验证了其在整个模型生命周期中的高效性。 

---
# Towards Data-Efficient Pretraining for Atomic Property Prediction 

**Title (ZH)**: 面向原子性质预测的数据高效预训练方法 

**Authors**: Yasir Ghunaim, Hasan Abed Al Kader Hammoud, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2502.11085)  

**Abstract**: This paper challenges the recent paradigm in atomic property prediction that links progress to growing dataset sizes and computational resources. We show that pretraining on a carefully selected, task-relevant dataset can match or even surpass large-scale pretraining, while using as little as 1/24th of the computational cost. We introduce the Chemical Similarity Index (CSI), a novel metric inspired by computer vision's Fréchet Inception Distance, for molecular graphs which quantifies the alignment between upstream pretraining datasets and downstream tasks. By selecting the most relevant dataset with minimal CSI distance, we show that models pretrained on a smaller, focused dataset consistently outperform those pretrained on massive, mixed datasets such as JMP, even when those larger datasets include the relevant dataset. Counterintuitively, we also find that indiscriminately adding more data can degrade model performance when the additional data poorly aligns with the task at hand. Our findings highlight that quality often outperforms quantity in pretraining for atomic property prediction. 

**Abstract (ZH)**: 本文挑战了原子属性预测领域的近期范式，即认为进展与数据集规模和计算资源的增长相关。我们展示了在精心选择的任务相关数据集上进行预训练可以与甚至超越大规模预训练，同时只使用其1/24的计算成本。我们引入了化学相似性指数（CSI），这是一种受计算机视觉中的弗雷谢入眼距离（FID）启发的新度量标准，用于分子图，量化上游预训练数据集与下游任务之间的对齐程度。通过选择CSI距离最小的最相关数据集，我们证明了基于更小、更聚焦数据集预训练的模型在很大程度上优于基于大规模混杂数据集（如JMP）预训练的模型，即使是那些大数据集中包含了相关数据集的情况。令人意外的是，我们还发现盲目增加更多数据可能会降低模型性能，特别是当额外数据与实际任务不匹配时。我们的研究结果强调，在原子属性预测的预训练中，质量往往优于数量。 

---
# Phantom: Subject-consistent video generation via cross-modal alignment 

**Title (ZH)**: Phantom：基于跨模态对齐的主体一致视频生成 

**Authors**: Lijie Liu, Tianxiang Ma, Bingchuan Li, Zhuowei Chen, Jiawei Liu, Qian He, Xinglong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11079)  

**Abstract**: The continuous development of foundational models for video generation is evolving into various applications, with subject-consistent video generation still in the exploratory stage. We refer to this as Subject-to-Video, which extracts subject elements from reference images and generates subject-consistent video through textual instructions. We believe that the essence of subject-to-video lies in balancing the dual-modal prompts of text and image, thereby deeply and simultaneously aligning both text and visual content. To this end, we propose Phantom, a unified video generation framework for both single and multi-subject references. Building on existing text-to-video and image-to-video architectures, we redesign the joint text-image injection model and drive it to learn cross-modal alignment via text-image-video triplet data. In particular, we emphasize subject consistency in human generation, covering existing ID-preserving video generation while offering enhanced advantages. The project homepage is here this https URL. 

**Abstract (ZH)**: 基础模型在视频生成中的持续发展正逐步应用于各种场景，而基于主题的一致性视频生成仍处于探索阶段。我们将其称之为“主题到视频”，该方法从参考图像中提取主题元素，并通过文本指令生成主题一致的视频。我们认为，主题到视频的核心在于平衡文本和图像的双重提示，从而实现文本和视觉内容的深度和同步对齐。为此，我们提出了Phantom，一个统一的主题一致视频生成框架，支持单主题和多主题参考。基于现有的文本到视频和图像到视频架构，我们重新设计了联合文本-图像注入模型，并通过文本-图像-视频三元组数据驱动其学习跨模态对齐。特别是在人类生成中强调主题一致性，不仅保留现有的ID保持视频生成特性，还提供了增强的优势。该项目主页为：[此链接](https://github.com/alibaba/qwen)。 

---
# Exposing Numeracy Gaps: A Benchmark to Evaluate Fundamental Numerical Abilities in Large Language Models 

**Title (ZH)**: 揭示数值能力差距：一种评估大型语言模型基本数值能力的标准 

**Authors**: Haoyang Li, Xuejia Chen, Zhanchao XU, Darian Li, Nicole Hu, Fei Teng, Yiming Li, Luyu Qiu, Chen Jason Zhang, Qing Li, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.11075)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in natural language processing tasks, such as text generation and semantic understanding. However, their performance on numerical reasoning tasks, such as basic arithmetic, numerical retrieval, and magnitude comparison, remains surprisingly poor. This gap arises from their reliance on surface-level statistical patterns rather than understanding numbers as continuous magnitudes. Existing benchmarks primarily focus on either linguistic competence or structured mathematical problem-solving, neglecting fundamental numerical reasoning required in real-world scenarios. To bridge this gap, we propose NumericBench, a comprehensive benchmark to evaluate six fundamental numerical capabilities: number recognition, arithmetic operations, contextual retrieval, comparison, summary, and logical reasoning. NumericBench includes datasets ranging from synthetic number lists to the crawled real-world data, addressing challenges like long contexts, noise, and multi-step reasoning. Extensive experiments on state-of-the-art LLMs, including GPT-4 and DeepSeek, reveal persistent weaknesses in numerical reasoning, highlighting the urgent need to improve numerically-aware language modeling. The benchmark is released in: this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理任务中展现了令人印象深刻的能力，如文本生成和语义理解。然而，它们在数值推理任务中的表现，如基本算术、数值检索和幅度比较，仍然出人意料地差。这种差距源于它们依赖表面级别的统计模式，而不是理解数字作为连续量。现有的基准测试主要侧重于语言能力和结构化数学问题解决，忽略了许多在实际场景中基础的数值推理需求。为了弥合这一差距，我们提出了NumericBench，这是一个综合基准测试，用于评估六种基本的数值能力：数字识别、算术运算、上下文检索、比较、总结以及逻辑推理。NumericBench 包括从合成数字列表到抓取的真实世界数据的各类数据集，解决了长上下文、噪声和多步推理等挑战。对包括GPT-4和DeepSeek在内的前沿LLM进行的大量实验揭示了持续存在的数值推理弱点，突显了迫切需要改进数值感知的语言模型。该基准测试的发布地址如下：this https URL。 

---
# A Survey on Vulnerability Prioritization: Taxonomy, Metrics, and Research Challenges 

**Title (ZH)**: 漏洞优先级研究综述：分类、指标及研究挑战 

**Authors**: Yuning Jiang, Nay Oo, Qiaoran Meng, Hoon Wei Lim, Biplab Sikdar  

**Link**: [PDF](https://arxiv.org/pdf/2502.11070)  

**Abstract**: In the highly interconnected digital landscape of today, safeguarding complex infrastructures against cyber threats has become increasingly challenging due to the exponential growth in the number and complexity of vulnerabilities. Resource constraints necessitate effective vulnerability prioritization strategies, focusing efforts on the most critical risks. This paper presents a systematic literature review of 82 studies, introducing a novel taxonomy that categorizes metrics into severity, exploitability, contextual factors, predictive indicators, and aggregation methods. Our analysis reveals significant gaps in existing approaches and challenges with multi-domain applicability. By emphasizing the need for dynamic, context-aware metrics and scalable solutions, we provide actionable insights to bridge the gap between research and real-world applications. This work contributes to the field by offering a comprehensive framework for evaluating vulnerability prioritization methodologies and setting a research agenda to advance the state of practice. 

**Abstract (ZH)**: 在当今高度互联的数字景观中，由于漏洞的数量和复杂性的指数级增长，保护复杂的基础设施免受网络威胁已变得日益挑战性。资源限制要求有效的漏洞优先级策略，重点放在最关键的风险上。本文通过系统地回顾82项研究，提出了一个新的分类系统，将度量标准分为严重性、可利用性、情境因素、预测指标和聚合方法。我们的分析揭示了现有方法中存在的重要差距以及在多领域应用中的挑战。通过强调动态和情境感知度量以及可扩展解决方案的重要性，我们提供了将研究成果应用于实际应用中的可操作见解。这项工作通过提供评估漏洞优先级方法的全面框架以及为推进实际应用水平设定研究议程的方式，为该领域做出了贡献。 

---
# Accelerating Anchors via Specialization and Feature Transformation 

**Title (ZH)**: 通过专业特化和特征转换加速锚点生成 

**Authors**: Haonan Yu, Junhao Liu, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11068)  

**Abstract**: Anchors is a popular local model-agnostic explanation technique whose applicability is limited by its computational inefficiency. To address this limitation, we propose a pre-training-based approach to accelerate Anchors without compromising the explanation quality. Our approach leverages the iterative nature of Anchors' algorithm which gradually refines an explanation until it is precise enough for a given input by providing a general explanation that is obtained through pre-training as Anchors' initial explanation. Specifically, we develop a two-step rule transformation process: the horizontal transformation adapts a pre-trained explanation to the current input by replacing features, and the vertical transformation refines the general explanation until it is precise enough for the input. We evaluate our method across tabular, text, and image datasets, demonstrating that it significantly reduces explanation generation time while maintaining fidelity and interpretability, thereby enabling the practical adoption of Anchors in time-sensitive applications. 

**Abstract (ZH)**: 锚点是一种流行的局部模型无关的解释技术，但由于其计算上的低效率，其应用受到了限制。为了解决这一局限性，我们提出了一种基于预训练的方法，以加快锚点的运行速度而不损害解释的质量。我们的方法利用了锚点算法的迭代性质，该算法通过逐步细化解释直到对给定输入足够精确，利用预训练得到的泛化解释作为锚点的初始解释。具体来说，我们开发了一个两步规则变换过程：水平变换通过替换特征将预训练解释适应当前输入，而垂直变换则不断细化泛化解释，直到它对输入足够精确。我们通过在表格数据、文本数据和图像数据集上的评估，证明了这种方法在显著减少解释生成时间的同时保持了忠实性和可解释性，从而使得锚点可以在时间敏感的应用中得到实际应用。 

---
# ClimateLLM: Efficient Weather Forecasting via Frequency-Aware Large Language Models 

**Title (ZH)**: ClimateLLM：基于频率意识的大语言模型高效天气预报 

**Authors**: Shixuan Li, Wei Yang, Peiyu Zhang, Xiongye Xiao, Defu Cao, Yuehan Qin, Xiaole Zhang, Yue Zhao, Paul Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2502.11059)  

**Abstract**: Weather forecasting is crucial for public safety, disaster prevention and mitigation, agricultural production, and energy management, with global relevance. Although deep learning has significantly advanced weather prediction, current methods face critical limitations: (i) they often struggle to capture both dynamic temporal dependencies and short-term abrupt changes, making extreme weather modeling difficult; (ii) they incur high computational costs due to extensive training and resource requirements; (iii) they have limited adaptability to multi-scale frequencies, leading to challenges when separating global trends from local fluctuations. To address these issues, we propose ClimateLLM, a foundation model for weather forecasting. It captures spatiotemporal dependencies via a cross-temporal and cross-spatial collaborative modeling framework that integrates Fourier-based frequency decomposition with Large Language Models (LLMs) to strengthen spatial and temporal modeling. Our framework uses a Mixture-of-Experts (MoE) mechanism that adaptively processes different frequency components, enabling efficient handling of both global signals and localized extreme events. In addition, we introduce a cross-temporal and cross-spatial dynamic prompting mechanism, allowing LLMs to incorporate meteorological patterns across multiple scales effectively. Extensive experiments on real-world datasets show that ClimateLLM outperforms state-of-the-art approaches in accuracy and efficiency, as a scalable solution for global weather forecasting. 

**Abstract (ZH)**: 天气预报对于公共安全、灾害预防与减轻、农业生产及能源管理具有全球性的重大意义。尽管深度学习在天气预测方面取得了显著突破，但目前的方法仍面临关键限制：（i）它们往往难以捕捉动态的时间依赖性和短期突变，导致极端天气建模困难；（ii）由于广泛的训练和资源需求导致计算成本高；（iii）对多尺度频率的适应性有限，使得在区分全球趋势与局部波动时面临挑战。为了解决这些问题，我们提出了ClimateLLM，一种用于天气预报的基准模型。它通过一种跨时间和跨空间的协作建模框架来捕捉时空依赖性，该框架结合了基于傅里叶的频率分解与大型语言模型（LLMs），从而增强了空间和时间建模。我们的框架利用了一种混合专家机制（MoE），该机制能够自适应地处理不同频率的成分，从而有效地处理全局信号和局部极端事件。此外，我们还引入了一种跨时间和跨空间的动力提示机制，使得LLMs能够有效地纳入跨多尺度的气象模式。在真实世界数据集上的大量实验表明，ClimateLLM 在准确性和效率方面均优于当前最先进的方法，是全球天气预报的可扩展解决方案。 

---
# A Physics-Informed Machine Learning Framework for Safe and Optimal Control of Autonomous Systems 

**Title (ZH)**: 一种具备物理信息的机器学习框架，用以实现自主系统安全且优化的控制 

**Authors**: Manan Tayal, Aditya Singh, Shishir Kolathaya, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2502.11057)  

**Abstract**: As autonomous systems become more ubiquitous in daily life, ensuring high performance with guaranteed safety is crucial. However, safety and performance could be competing objectives, which makes their co-optimization difficult. Learning-based methods, such as Constrained Reinforcement Learning (CRL), achieve strong performance but lack formal safety guarantees due to safety being enforced as soft constraints, limiting their use in safety-critical settings. Conversely, formal methods such as Hamilton-Jacobi (HJ) Reachability Analysis and Control Barrier Functions (CBFs) provide rigorous safety assurances but often neglect performance, resulting in overly conservative controllers. To bridge this gap, we formulate the co-optimization of safety and performance as a state-constrained optimal control problem, where performance objectives are encoded via a cost function and safety requirements are imposed as state constraints. We demonstrate that the resultant value function satisfies a Hamilton-Jacobi-Bellman (HJB) equation, which we approximate efficiently using a novel physics-informed machine learning framework. In addition, we introduce a conformal prediction-based verification strategy to quantify the learning errors, recovering a high-confidence safety value function, along with a probabilistic error bound on performance degradation. Through several case studies, we demonstrate the efficacy of the proposed framework in enabling scalable learning of safe and performant controllers for complex, high-dimensional autonomous systems. 

**Abstract (ZH)**: 随着自主系统在日常生活中的应用日益广泛，确保在保证安全的前提下实现高性能变得至关重要。然而，安全性和性能可能是相互冲突的目标，这使得它们的共同优化变得困难。基于学习的方法，如约束强化学习（CRL），能够实现高性能，但由于安全要求以软约束的形式实现，因此缺乏正式的安全保证，限制了它们在关键安全环境中的应用。相反，正式的方法，如哈密尔顿-雅可比（HJ）可达性分析和控制障碍函数（CBFs），能够提供严格的安全保证，但往往忽视了性能，导致控制器过于保守。为了弥合这一差距，我们将安全和性能的共同优化公式化为状态约束最优控制问题，其中性能目标通过代价函数编码，安全要求则作为状态约束加以限制。我们证明，由此得到的价值函数满足哈密尔顿-雅可比-贝尔曼（HJB）方程，并使用一种新的物理相关信息的机器学习框架高效地近似该方程。此外，我们引入了一种基于齐性预测的验证策略，以量化学习误差，从而恢复一个高置信度的安全价值函数，并给出性能退化的概率误差上限。通过几个案例研究，我们展示了所提出的框架在实现复杂、高维自主系统安全且高性能的控制器方面的有效性。 

---
# Reasoning-Augmented Conversation for Multi-Turn Jailbreak Attacks on Large Language Models 

**Title (ZH)**: 增强推理的多轮对话对抗大规模语言模型中的突变攻击 

**Authors**: Zonghao Ying, Deyue Zhang, Zonglei Jing, Yisong Xiao, Quanchen Zou, Aishan Liu, Siyuan Liang, Xiangzheng Zhang, Xianglong Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2502.11054)  

**Abstract**: Multi-turn jailbreak attacks simulate real-world human interactions by engaging large language models (LLMs) in iterative dialogues, exposing critical safety vulnerabilities. However, existing methods often struggle to balance semantic coherence with attack effectiveness, resulting in either benign semantic drift or ineffective detection evasion. To address this challenge, we propose Reasoning-Augmented Conversation, a novel multi-turn jailbreak framework that reformulates harmful queries into benign reasoning tasks and leverages LLMs' strong reasoning capabilities to compromise safety alignment. Specifically, we introduce an attack state machine framework to systematically model problem translation and iterative reasoning, ensuring coherent query generation across multiple turns. Building on this framework, we design gain-guided exploration, self-play, and rejection feedback modules to preserve attack semantics, enhance effectiveness, and sustain reasoning-driven attack progression. Extensive experiments on multiple LLMs demonstrate that RACE achieves state-of-the-art attack effectiveness in complex conversational scenarios, with attack success rates (ASRs) increasing by up to 96%. Notably, our approach achieves ASRs of 82% and 92% against leading commercial models, OpenAI o1 and DeepSeek R1, underscoring its potency. We release our code at this https URL to facilitate further research in this critical domain. 

**Abstract (ZH)**: 多轮囚笼攻击通过与大型语言模型（LLMs）进行迭代对话，模拟现实世界中的人际互动，揭示了关键的安全漏洞。然而，现有的方法经常难以在语义连贯性与攻击有效性之间取得平衡，导致要么产生无害的语义漂移，要么攻击检测规避效果不佳。为了解决这一挑战，我们提出了一种名为推理增强对话 (Reasoning-Augmented Conversation, RACE) 的新型多轮囚笼框架。该框架将有害查询重新表述为良性推理任务，并利用LLMs的强大推理能力来削弱安全对齐。具体来说，我们引入了一种攻击状态机框架，系统地建模问题翻译和迭代推理，确保多轮次查询生成的连贯性。在此基础上，我们设计了收益导向的探索、自我对弈以及拒绝反馈模块，以保持攻击语义、增强效果，并维持以推理驱动的攻击进展。在多个LLM上的广泛实验表明，RACE在复杂对话场景中的攻击效果达到了最先进的水平，攻击成功率（ASRs）最高可提升96%。值得注意的是，我们的方法分别对OpenAI o1和DeepSeek R1这两个领先的商用模型实现了82%和92%的攻击成功率，突显其有效性和强大力量。我们已在此 URL 释放我们的代码，以促进对该领域进一步研究。

[译者注：这里的URL是一个占位符，实际发布代码时应替换为清华大学隐私计算研究中心的具体链接。] 

---
# MMUNLEARNER: Reformulating Multimodal Machine Unlearning in the Era of Multimodal Large Language Models 

**Title (ZH)**: MMUNLEARNER：在多模态大规模语言模型时代改革多模态机器遗忘算法 

**Authors**: Jiahao Huo, Yibo Yan, Xu Zheng, Yuanhuiyi Lyu, Xin Zou, Zhihua Wei, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11051)  

**Abstract**: Recent progress in Machine Unlearning (MU) has introduced solutions for the selective removal of private or sensitive information encoded within deep neural networks. Nonetheless, MU for Multimodal Large Language Models (MLLMs) remains in its nascent phase. Therefore, we propose to reformulate the task of multimodal MU in the era of MLLMs, which aims to erase only the visual patterns associated with a given entity while preserving the corresponding textual knowledge encoded within the original parameters of the language model backbone. Furthermore, we develop a novel geometry-constrained gradient descent method MMUnlearner. It updates the weights of MLLMs with a weight saliency map jointly restricted by the remaining concepts and textual knowledge during unlearning, thereby preserving parameters essential for non-target knowledge. Extensive experiments demonstrate that MMUnlearner surpasses baselines that finetuning MLLMs with VQA data directly through Gradient Ascent (GA) or Negative Preference Optimization (NPO), across all evaluation dimensions. Our code will be released upon acceptance. 

**Abstract (ZH)**: 近年来，机器卸载（Machine Unlearning，MU）的研究取得了一定进展，提出了从深度神经网络中选择性移除私人或敏感信息的解决方案。然而，面向多模态大型语言模型（Multimodal Large Language Models，MLLMs）的机器卸载仍处于起步阶段。因此，我们提出了在MLLM时代重新定义多模态MU任务的方法，该任务旨在仅消除与特定实体相关的视觉模式，同时保留语言模型主干中原有的相应文本知识。此外，我们开发了一种新颖的几何约束梯度下降方法MMUnlearner。在卸载过程中，MMUnlearner通过共同限制剩余概念和文本知识更新MLLMs的权重，并因此保留了非目标知识所需的关键参数。广泛的实验表明，MMUnlearner在所有评估维度上均优于通过梯度上升（Gradient Ascent，GA）或负面偏好优化（Negative Preference Optimization，NPO）直接微调MLLMs的基线方法。我们的代码将在接受后公开发布。 

---
# Deep Incomplete Multi-view Learning via Cyclic Permutation of VAEs 

**Title (ZH)**: 基于VAEs的循环排列深度不完全多视角学习 

**Authors**: Xin Gao, Jian Pu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11037)  

**Abstract**: Multi-View Representation Learning (MVRL) aims to derive a unified representation from multi-view data by leveraging shared and complementary information across views. However, when views are irregularly missing, the incomplete data can lead to representations that lack sufficiency and consistency. To address this, we propose Multi-View Permutation of Variational Auto-Encoders (MVP), which excavates invariant relationships between views in incomplete data. MVP establishes inter-view correspondences in the latent space of Variational Auto-Encoders, enabling the inference of missing views and the aggregation of more sufficient information. To derive a valid Evidence Lower Bound (ELBO) for learning, we apply permutations to randomly reorder variables for cross-view generation and then partition them by views to maintain invariant meanings under permutations. Additionally, we enhance consistency by introducing an informational prior with cyclic permutations of posteriors, which turns the regularization term into a similarity measure across distributions. We demonstrate the effectiveness of our approach on seven diverse datasets with varying missing ratios, achieving superior performance in multi-view clustering and generation tasks. 

**Abstract (ZH)**: 多视图表示学习（MVRL）旨在通过利用各视图间共享和互补的信息，从多视图数据中提取统一的表示。然而，当视图存在不规则缺失时，不完整的数据可能导致表示不足且不一致。为解决这一问题，我们提出了一种多视图排列的变分自编码器（MVP），该方法挖掘不完整数据中视图之间的不变关系。MVP 在变分自编码器的隐空间中建立了视图间的对应关系，从而能够推断缺失视图并聚合更多充分的信息。为了从学习中得出有效的证据下界（ELBO），我们通过对变量进行随机重排列以促进跨视图生成，并按视图分区以保持排列下的不变意义。此外，我们通过引入循环排列的先验信息约束，增强了一致性，该约束将正则化项转化为分布间的相似度度量。我们通过在七个具有不同缺失比例的多样数据集上进行实验，展示了该方法的有效性，并在多视图聚类和生成任务中取得了优越的性能。 

---
# Mind the Confidence Gap: Overconfidence, Calibration, and Distractor Effects in Large Language Models 

**Title (ZH)**: 注意信心差距：大型语言模型中的过度自信、校准问题及干扰项效应 

**Authors**: Prateek Chhikara  

**Link**: [PDF](https://arxiv.org/pdf/2502.11028)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive performance across diverse tasks, yet confidence calibration remains a challenge. Miscalibration - where models are overconfident or underconfident - poses risks, particularly in high-stakes applications. This paper presents an empirical study on LLM calibration, examining how model size, distractors, and question types affect confidence alignment. We introduce an evaluation framework to measure overconfidence and investigate whether multiple-choice formats mitigate or worsen miscalibration. Our findings show that while larger models (e.g., GPT-4o) are better calibrated overall, they are more prone to distraction, whereas smaller models benefit more from answer choices but struggle with uncertainty estimation. Unlike prior work, which primarily reports miscalibration trends, we provide actionable insights into failure modes and conditions that worsen overconfidence. These findings highlight the need for calibration-aware interventions and improved uncertainty estimation methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种任务中表现出色，但置信度校准仍是一个挑战。误校准——即模型过于自信或不够自信——在高风险应用中尤其存在风险。这篇论文进行了一项实证研究，探讨了模型规模、干扰项和问题类型如何影响信心对齐。我们提出了一种评估框架来衡量过自信情况，并研究多项选择题格式是否能缓解或加剧误校准。我们的研究结果显示，虽然较大的模型（例如GPT-4o）整体上更易于校准，但更容易受到干扰影响，而较小的模型受益于答案选择，但难以进行不确定性估计。不同于以往研究主要报告误校准趋势，我们提供了关于失败模式和加剧过自信的具体建议。这些发现突显了需要采取校准意识措施，并改进不确定性估计方法的需求。 

---
# Simplify RLHF as Reward-Weighted SFT: A Variational Method 

**Title (ZH)**: 简化 RLHF 为奖励加权 SFT：一种变分方法 

**Authors**: Yuhao Du, Zhuo Li, Pengyu Cheng, Zhihong Chen, Yuejiao Xie, Xiang Wan, Anningzhe Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.11026)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is crucial for aligning Large Language Models (LLMs) with human values. However, RLHF has been continuously challenged by its high complexity in implementation and computation consumption. Even with recent simplifications, such as Direct Preference Optimization (DPO) and Advantage Leftover Lunch (A-LoL), the problems of over-fitting and training instability remain hindering the alignment process from the expected optimal performance. To address the existing challenges, we propose a novel simplification of RLHF from the perspective of variational inference, called $\textbf{V}$ariational $\textbf{A}$lignment with $\textbf{R}$e-weighting ($\textbf{VAR}$). More specifically, by directly minimizing the distribution gap between the learning LLM policy and the optimal solution of RLHF, we transform the alignment objective into a reward-driven re-weighted supervised fine-tuning (SFT) form, which only requires minor adjustment on the SFT loss to obtain noticeable improvement on training stability and effectiveness. On comprehensive alignment and generation benchmarks, our VAR method has numerically achieved competitive performance in LLM alignment helpfulness and harmlessness. 

**Abstract (ZH)**: 人类反馈强化学习（RLHF）对于使大型语言模型（LLM）与人类价值观保持一致至关重要。然而，RLHF在实施复杂性和计算消耗方面一直面临持续的挑战。即使最近引入了诸如直接偏好优化（DPO）和优势剩余午餐（A-LoL）等简化方法，过度拟合和训练不稳定性的问题仍然阻碍了从预期的最佳性能中取得理想的效果。为了解决现有挑战，我们从变分推断的角度提出了一种新的RLHF简化方法，称为**变分对齐与重权**（**VAR**）。具体而言，通过直接最小化学习中的LLM策略与RLHF最优解之间的分布差异，我们将对齐目标转换为奖励驱动的重权监督微调（SFT）形式，仅需对SFT损失进行轻微调整即可显著改善训练稳定性和效果。在综合对齐和生成基准测试中，我们的VAR方法在LLM对齐的有益性和无害性方面实现了具有竞争力的性能。 

---
# MultiTEND: A Multilingual Benchmark for Natural Language to NoSQL Query Translation 

**Title (ZH)**: MultiTEND：一种多语言基准，用于自然语言到NoSQL查询的转换 

**Authors**: Zhiqian Qin, Yuanfeng Song, Jinwei Lu, Yuanwei Song, Shuaimin Li, Chen Jason Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.11022)  

**Abstract**: Natural language interfaces for NoSQL databases are increasingly vital in the big data era, enabling users to interact with complex, unstructured data without deep technical expertise. However, most recent advancements focus on English, leaving a gap for multilingual support. This paper introduces MultiTEND, the first and largest multilingual benchmark for natural language to NoSQL query generation, covering six languages: English, German, French, Russian, Japanese and Mandarin Chinese. Using MultiTEND, we analyze challenges in translating natural language to NoSQL queries across diverse linguistic structures, including lexical and syntactic differences. Experiments show that performance accuracy in both English and non-English settings remains relatively low, with a 4%-6% gap across scenarios like fine-tuned SLM, zero-shot LLM, and RAG for LLM. To address the aforementioned challenges, we introduce MultiLink, a novel framework that bridges the multilingual input to NoSQL query generation gap through a Parallel Linking Process. It breaks down the task into multiple steps, integrating parallel multilingual processing, Chain-of-Thought (CoT) reasoning, and Retrieval-Augmented Generation (RAG) to tackle lexical and structural challenges inherent in multilingual NoSQL generation. MultiLink shows enhancements in all metrics for every language against the top baseline, boosting execution accuracy by about 15% for English and averaging a 10% improvement for non-English languages. 

**Abstract (ZH)**: 在大数据时代，自然语言接口对于NoSQL数据库变得日益重要，使得用户能够无需深厚的技术背景即可与复杂且结构化的数据进行交互。然而，当前大部分进展集中在英语上，存在多语言支持的空白。本文介绍了MultiTEND，这是首个并最大的多语言基准，用于自然语言到NoSQL查询的生成，覆盖了六种语言：英语、德语、法语、俄语、日语和普通话。通过使用MultiTEND，我们分析了跨不同语言结构（包括词汇和句法差异）将自然语言转换为NoSQL查询所面临的挑战。实验结果显示，在英语和非英语环境中，性能的准确性仍然相对较低，不同场景（如微调的语言模型、零样本大型语言模型以及L LARGE语言模型增强生成）之间有4%-6%的差距。为了解决上述挑战，我们提出了一种新型框架——MultiLink，通过并行链接过程弥合了多语言输入与NoSQL查询生成之间的差距。 MultiLink 将任务分解为多个步骤，整合了并行多语言处理、思维链（CoT）推理和检索增强生成（RAG），以应对多语言NoSQL生成中固有的词汇和结构挑战。实验表明，与顶级基准相比，MultiLink 在每种语言上的所有指标均有所提升，对于英语，执行准确性提高了约15%；对于非英语语言，平均提高了10%。 

---
# TUMLU: A Unified and Native Language Understanding Benchmark for Turkic Languages 

**Title (ZH)**: TUMLU：一个统一且原生的土耳其语族语言理解基准测试 

**Authors**: Jafar Isbarov, Arofat Akhundjanova, Mammad Hajili, Kavsar Huseynova, Dmitry Gaynullin, Anar Rzayev, Osman Tursun, Ilshat Saetov, Rinat Kharisov, Saule Belginova, Ariana Kenbayeva, Amina Alisheva, Aizirek Turdubaeva, Abdullatif Köksal, Samir Rustamov, Duygu Ataman  

**Link**: [PDF](https://arxiv.org/pdf/2502.11020)  

**Abstract**: Being able to thoroughly assess massive multi-task language understanding (MMLU) capabilities is essential for advancing the applicability of multilingual language models. However, preparing such benchmarks in high quality native language is often costly and therefore limits the representativeness of evaluation datasets. While recent efforts focused on building more inclusive MMLU benchmarks, these are conventionally built using machine translation from high-resource languages, which may introduce errors and fail to account for the linguistic and cultural intricacies of the target languages. In this paper, we address the lack of native language MMLU benchmark especially in the under-represented Turkic language family with distinct morphosyntactic and cultural characteristics. We propose two benchmarks for Turkic language MMLU: TUMLU is a comprehensive, multilingual, and natively developed language understanding benchmark specifically designed for Turkic languages. It consists of middle- and high-school level questions spanning 11 academic subjects in Azerbaijani, Crimean Tatar, Karakalpak, Kazakh, Tatar, Turkish, Uyghur, and Uzbek. We also present TUMLU-mini, a more concise, balanced, and manually verified subset of the dataset. Using this dataset, we systematically evaluate a diverse range of open and proprietary multilingual large language models (LLMs), including Claude, Gemini, GPT, and LLaMA, offering an in-depth analysis of their performance across different languages, subjects, and alphabets. To promote further research and development in multilingual language understanding, we release TUMLU-mini and all corresponding evaluation scripts. 

**Abstract (ZH)**: 彻底评估大规模多任务语言理解（MMLU）能力对于推动多语言语言模型的应用极为重要。然而，高质量的母语基准准备往往成本高昂，因此限制了评估数据集的代表性。尽管最近的研究努力致力于建立更具包容性的MMLU基准，但这些基准通常是通过高资源语言的机器翻译构建的，可能会引入错误，并且未能考虑到目标语言的语料和文化细微差别。在本文中，我们针对储量不足的突厥语语言家族中的母语MMLU基准，尤其是突厥语语言家族具有明显形态语法和文化特点的语言，解决了这个问题。我们提出了两个突厥语语言MMLU基准：TUMLU是一个全面的、多语言的、母语开发的语言理解基准，专门设计用于突厥语语言。它包含了阿塞拜疆语、克里米亚鞑靼语、卡拉卡尔帕克语、哈萨克语、鞑靼语、土耳其语、维吾尔语和乌孜别克语在内的11个学术科目中的中学和高中级别问题。我们还介绍了一个更精简、更平衡且手动验证的子集TUMLU-mini。利用这个数据集，我们系统地评估了多种开源和专有大型多语言语言模型（LLMs），包括Claude、Gemini、GPT和LLaMA，对其在不同语言、科目和字母系统上的性能进行了深入分析。为了促进多语言语言理解领域的进一步研究和开发，我们发布了TUMLU-mini和所有相应的评估脚本。 

---
# Unlocking the Power of Function Vectors for Characterizing and Mitigating Catastrophic Forgetting in Continual Instruction Tuning 

**Title (ZH)**: 解锁功能向量的潜力，以characterizing和mitigating持续指令调谐中灾难性遗忘的问题 

**Authors**: Gangwei Jiang, Caigao Jiang, Zhaoyi Li, Siqiao Xue, Jun Zhou, Linqi Song, Defu Lian, Yin Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.11019)  

**Abstract**: Catastrophic forgetting (CF) poses a significant challenge in machine learning, where a model forgets previously learned information upon learning new tasks. Despite the advanced capabilities of Large Language Models (LLMs), they continue to face challenges with CF during continual learning. The majority of existing research focuses on analyzing forgetting patterns through a singular training sequence, thereby overlooking the intricate effects that diverse tasks have on model behavior. Our study explores CF across various settings, discovering that model forgetting is influenced by both the specific training tasks and the models themselves. To this end, we interpret forgetting by examining the function vector (FV), a compact representation of functions in LLMs, offering a model-dependent indicator for the occurrence of CF. Through theoretical and empirical analyses, we demonstrated that CF in LLMs primarily stems from biases in function activation rather than the overwriting of task processing functions. Leveraging these insights, we propose a novel function vector guided training methodology, incorporating a regularization technique to stabilize the FV and mitigate forgetting. Empirical tests on four benchmarks confirm the effectiveness of our proposed training method, substantiating our theoretical framework concerning CF and model function dynamics. We plan to make our code publicly accessible in the near future. 

**Abstract (ZH)**: 灾难性遗忘（CF）是机器学习中一个重要的挑战，当模型在学习新任务时会忘记之前学到的信息。尽管大型语言模型（LLMs）具有先进的能力，但在连续学习过程中它们仍然面临CF的挑战。现有研究大多数仅通过单一训练序列分析遗忘模式，从而忽略了不同任务对模型行为的复杂影响。我们的研究探索了不同情境下的CF，并发现模型的遗忘受特定训练任务和模型本身的影响。为此，我们通过分析功能向量（FV），一种LLMs中功能的紧凑表示，来解释遗忘现象，并提供一个模型依赖的指标来表征CF的发生。通过理论和实证分析，我们证明了LLMs中的CF主要源自功能激活的偏差而非任务处理功能的覆盖。利用这些见解，我们提出了一个基于功能向量的新型训练方法，引入正则化技术以稳定FV并减轻遗忘。在四个基准测试上的实验证明了我们所提出训练方法的有效性，验证了我们关于CF和模型功能动态的理论框架。我们计划在未来一段时间内公开我们的代码。 

---
# GRIFFIN: Effective Token Alignment for Faster Speculative Decoding 

**Title (ZH)**: GRIFFIN：高效的令牌对齐方法以实现更快的投机解码 

**Authors**: Shijing Hu, Jingyang Li, Xingyu Xie, Zhihui Lu, Kim-Chuan Toh, Pan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.11018)  

**Abstract**: Speculative decoding accelerates inference in large language models (LLMs) by generating multiple draft tokens simultaneously. However, existing methods often struggle with token misalignment between the training and decoding phases, limiting their performance. To address this, we propose GRIFFIN, a novel framework that incorporates a token-alignable training strategy and a token-alignable draft model to mitigate misalignment. The training strategy employs a loss masking mechanism to exclude highly misaligned tokens during training, preventing them from negatively impacting the draft model's optimization. The token-alignable draft model introduces input tokens to correct inconsistencies in generated features. Experiments on LLaMA-series and Vicuna models demonstrate that GRIFFIN achieves an average acceptance length improvement of over 7\% and a speedup ratio exceeding 8%, outperforming current SoTAs as shown in Fig. 1 (a) and (b). 

**Abstract (ZH)**: 推测解码通过同时生成多个草稿标记来加速大型语言模型（LLM）的推理过程。然而，现有方法常常难以解决训练阶段与解码阶段之间的标记对齐问题，从而限制了其性能。为了解决这一问题，我们提出了一种新的框架GRIFFIN，该框架结合了一种可对齐训练策略和一种可对齐的草稿模型来缓解对齐问题。训练策略采用了一种损失掩码机制，在训练过程中排除高度不对齐的标记，防止它们对草稿模型的优化产生负面影响。可对齐的草稿模型则引入输入标记以修正生成特征中的一致性问题。实验结果表明，GRIFFIN在LLaMA系列和Vicuna模型上实现了平均接受长度提高超过7%和加速比超过8%的改进，如图1（a）和（b）所示，优于当前的SOTA方法。 

---
# Collaborative Deterministic-Diffusion Model for Probabilistic Urban Spatiotemporal Prediction 

**Title (ZH)**: 协作确定性扩散模型在城市时空概率预测中的应用 

**Authors**: Zhi Sheng, Yuan Yuan, Yudi Zhang, Depeng Jin, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.11013)  

**Abstract**: Accurate prediction of urban spatiotemporal dynamics is essential for enhancing urban management and decision-making. Existing spatiotemporal prediction models are predominantly deterministic, focusing on primary spatiotemporal patterns. However, those dynamics are highly complex, exhibiting multi-modal distributions that are challenging for deterministic models to capture. In this paper, we highlight the critical role of probabilistic prediction in capturing the uncertainties and complexities inherent in spatiotemporal data. While mainstream probabilistic models can capture uncertainty, they struggle with accurately learning primary patterns and often suffer from computational inefficiency. To address these challenges, we propose CoST, which collaborates deterministic and probabilistic models to improve both predictive accuracy and the ability to handle uncertainty. To achieve this, we design a mean-residual decomposition framework, where the mean value is modeled by a deterministic model, and the residual variations are learned by a probabilistic model, specifically diffusion models. Moreover, we introduce a scale-aware diffusion process, which better accounts for spatially heterogeneous dynamics across different regions. Extensive experiments on eight real-world datasets demonstrate that CoST significantly outperforms existing methods in both deterministic and probabilistic metrics, achieving a 20% improvement with low computational cost. CoST bridges the gap between deterministic precision and probabilistic uncertainty, making a significant advancement in the field of urban spatiotemporal prediction. 

**Abstract (ZH)**: 准确预测城市时空动态对于提升城市管理与决策具有重要意义。现有的时空预测模型主要为确定性模型，侧重于主要的时空模式。然而，这些动态过程十分复杂，呈现多模态分布，这对确定性模型来说难以捕捉。本文强调概率预测在捕捉时空数据中固有的不确定性和复杂性方面的作用至关重要。尽管主流的概率模型能够捕捉不确定性，但在学习主要模式方面仍存在局限性，且往往计算效率低下。为解决这些挑战，我们提出了一种名为CoST的方法，通过联合确定性和概率模型，以提高预测精度和处理不确定性的能力。为此，我们设计了一个均值残差分解框架，其中均值部分通过确定性模型建模，残差部分通过概率模型（特别是扩散模型）学习。此外，我们引入了一种尺度自适应的扩散过程，以更好地反映不同区域间的时空异质性动态。在八个真实世界数据集上的广泛实验表明，CoST在确定性和概率指标上均显著优于现有方法，且具有较低的计算成本，实现了20%的性能提升。CoST在确定性和概率预测之间架起了桥梁，显著推动了城市时空预测领域的进展。 

---
# Prompt Inject Detection with Generative Explanation as an Investigative Tool 

**Title (ZH)**: 使用生成解释作为调查工具的提示注入检测 

**Authors**: Jonathan Pan, Swee Liang Wong, Yidi Yuan, Xin Wei Chia  

**Link**: [PDF](https://arxiv.org/pdf/2502.11006)  

**Abstract**: Large Language Models (LLMs) are vulnerable to adversarial prompt based injects. These injects could jailbreak or exploit vulnerabilities within these models with explicit prompt requests leading to undesired responses. In the context of investigating prompt injects, the challenge is the sheer volume of input prompts involved that are likely to be largely benign. This investigative challenge is further complicated by the semantics and subjectivity of the input prompts involved in the LLM conversation with its user and the context of the environment to which the conversation is being carried out. Hence, the challenge for AI security investigators would be two-fold. The first is to identify adversarial prompt injects and then to assess whether the input prompt is contextually benign or adversarial. For the first step, this could be done using existing AI security solutions like guardrails to detect and protect the LLMs. Guardrails have been developed using a variety of approaches. A popular approach is to use signature based. Another popular approach to develop AI models to classify such prompts include the use of NLP based models like a language model. However, in the context of conducting an AI security investigation of prompt injects, these guardrails lack the ability to aid investigators in triaging or assessing the identified input prompts. In this applied research exploration, we explore the use of a text generation capabilities of LLM to detect prompt injects and generate explanation for its detections to aid AI security investigators in assessing and triaging of such prompt inject detections. The practical benefit of such a tool is to ease the task of conducting investigation into prompt injects. 

**Abstract (ZH)**: 大型语言模型（LLMs）容易受到基于恶意提示注入的攻击。这些注入可能导致模型脱逃或利用模型中的漏洞，产生不希望的响应。在调查提示注入时，面临的挑战是涉及的大量输入提示可能主要具有良性特征。此外，由于LLM对话中涉及的输入提示在语义和主观性上的复杂性，以及对话所进行的环境背景，这一调查挑战变得更加复杂。因此，AI安全调查面临的挑战将包括两个方面。首先，需要识别恶意提示注入，然后评估输入提示是否在上下文中是良性还是恶意。对于第一步，可以使用现有的AI安全解决方案，如防护栏来检测和保护LLM。防护栏已经通过多种方法开发。一种流行的方法是基于签名的方法，另一种流行的开发AI模型的方法包括使用基于NLP的语言模型。然而，在进行AI安全调查时，这些防护栏缺乏帮助调查人员分类或评估已识别的输入提示的能力。在这一应用研究探索中，我们将探讨利用LLM的文本生成能力来检测提示注入，并生成其检测的解释，以帮助AI安全调查人员评估和处理这些提示注入检测。这样的工具的应用价值在于简化对提示注入进行调查的任务。 

---
# CL-MFAP: A Contrastive Learning-Based Multimodal Foundation Model for Molecular Property Prediction and Antibiotic Screening 

**Title (ZH)**: CL-MFAP：基于对比学习的多模态基础模型在分子性质预测和抗生素筛选中的应用 

**Authors**: Gen Zhou, Sugitha Janarthanan, Yutong Lu, Pingzhao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.11001)  

**Abstract**: Due to the rise in antimicrobial resistance, identifying novel compounds with antibiotic potential is crucial for combatting this global health issue. However, traditional drug development methods are costly and inefficient. Recognizing the pressing need for more effective solutions, researchers have turned to machine learning techniques to streamline the prediction and development of novel antibiotic compounds. While foundation models have shown promise in antibiotic discovery, current mainstream efforts still fall short of fully leveraging the potential of multimodal molecular data. Recent studies suggest that contrastive learning frameworks utilizing multimodal data exhibit excellent performance in representation learning across various domains. Building upon this, we introduce CL-MFAP, an unsupervised contrastive learning (CL)-based multimodal foundation (MF) model specifically tailored for discovering small molecules with potential antibiotic properties (AP) using three types of molecular data. This model employs 1.6 million bioactive molecules with drug-like properties from the ChEMBL dataset to jointly pretrain three encoders: (1) a transformer-based encoder with rotary position embedding for processing SMILES strings; (2) another transformer-based encoder, incorporating a novel bi-level routing attention mechanism to handle molecular graph representations; and (3) a Morgan fingerprint encoder using a multilayer perceptron, to achieve the contrastive learning purpose. The CL-MFAP outperforms baseline models in antibiotic property prediction by effectively utilizing different molecular modalities and demonstrates superior domain-specific performance when fine-tuned for antibiotic-related property prediction tasks. 

**Abstract (ZH)**: 由于抗菌素耐药性的不断上升，发现具有抗生素潜在活性的新化合物对于应对这一全球健康问题至关重要。然而，传统药物开发方法成本高且效率低下。鉴于此紧迫需求，研究人员转向机器学习技术以简化新型抗生素化合物的预测和开发流程。虽然基础模型在抗生素发现方面显示出潜力，但当前主流努力尚未充分挖掘多模态分子数据的全部潜力。最近的研究表明，利用多模态数据的对比学习框架在不同领域中表现出色。在此基础上，我们介绍了CL-MFAP，这是一种基于对比学习（CL）的多模态基础（MF）模型，专为利用三种类型的数据发现具有潜在抗生素特性的小分子而定制。该模型利用来自ChEMBL数据集的160万种具有药理特性的生物活性分子，联合预训练三个编码器：（1）一种带有旋转位置嵌入的基于变换器的编码器，用于处理SMILES字符串；（2）另一种基于变换器的编码器，结合了一种新的两层路由注意力机制，以处理分子图表示；（3）一种使用多层感知器的摩根指纹编码器，以实现对比学习的目的。CL-MFAP在抗生素特性预测方面的表现优于基线模型，有效利用了不同类型的分子模态，并在针对抗生素相关特性预测任务进行微调后展现出更优的领域特异性性能。 

---
# ControlText: Unlocking Controllable Fonts in Multilingual Text Rendering without Font Annotations 

**Title (ZH)**: ControlText: 在无需字体注解的情况下解锁多语言文本渲染中的可控制字体 

**Authors**: Bowen Jiang, Yuan Yuan, Xinyi Bai, Zhuoqun Hao, Alyson Yin, Yaojie Hu, Wenyu Liao, Lyle Ungar, Camillo J. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2502.10999)  

**Abstract**: This work demonstrates that diffusion models can achieve font-controllable multilingual text rendering using just raw images without font label annotations. Visual text rendering remains a significant challenge. While recent methods condition diffusion on glyphs, it is impossible to retrieve exact font annotations from large-scale, real-world datasets, which prevents user-specified font control. To address this, we propose a data-driven solution that integrates the conditional diffusion model with a text segmentation model, utilizing segmentation masks to capture and represent fonts in pixel space in a self-supervised manner, thereby eliminating the need for any ground-truth labels and enabling users to customize text rendering with any multilingual font of their choice. The experiment provides a proof of concept of our algorithm in zero-shot text and font editing across diverse fonts and languages, providing valuable insights for the community and industry toward achieving generalized visual text rendering. 

**Abstract (ZH)**: 本研究证明，通过仅使用原始图像而无需字体标签注释，扩散模型可以实现可控多语言文本渲染。视觉文本渲染仍然是一个重大挑战。尽管最近的方法将扩散模型条件化于字符（glyph），但在大规模、真实世界的数据集中不可能从这些数据中检索到精确的字体注释，这妨碍了用户对字体的选择控制。为解决这一问题，我们提出了一种数据驱动的解决方案，将条件扩散模型与文本分割模型相结合，利用分割掩码在像素空间中自监督地捕捉和表示字体，从而消除了对任何真实标签的需求，并使用户能够使用任意选择的多语言字体自定义文本渲染。实验提供了我们在零样本的文字和字体编辑中使用该算法的概念验证，涵盖了多种字体和语言，为学术界和工业界实现通用视觉文本渲染提供了宝贵的洞见。 

---
# Is Elo Rating Reliable? A Study Under Model Misspecification 

**Title (ZH)**: 《Elo评分可靠吗？在模型误设情况下的研究》 

**Authors**: Shange Tang, Yuanhao Wang, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2502.10985)  

**Abstract**: Elo rating, widely used for skill assessment across diverse domains ranging from competitive games to large language models, is often understood as an incremental update algorithm for estimating a stationary Bradley-Terry (BT) model. However, our empirical analysis of practical matching datasets reveals two surprising findings: (1) Most games deviate significantly from the assumptions of the BT model and stationarity, raising questions on the reliability of Elo. (2) Despite these deviations, Elo frequently outperforms more complex rating systems, such as mElo and pairwise models, which are specifically designed to account for non-BT components in the data, particularly in terms of win rate prediction. This paper explains this unexpected phenomenon through three key perspectives: (a) We reinterpret Elo as an instance of online gradient descent, which provides no-regret guarantees even in misspecified and non-stationary settings. (b) Through extensive synthetic experiments on data generated from transitive but non-BT models, such as strongly or weakly stochastic transitive models, we show that the ''sparsity'' of practical matching data is a critical factor behind Elo's superior performance in prediction compared to more complex rating systems. (c) We observe a strong correlation between Elo's predictive accuracy and its ranking performance, further supporting its effectiveness in ranking. 

**Abstract (ZH)**: 以下是将该论文内容或标题翻译成中文的版本，符合学术规范：

Elo评分，广泛应用于从竞技游戏到大型语言模型等多个领域的技能评估，通常被视为一种基于逐步更新算法来估算平稳Bradley-Terry (BT) 模型的方法。然而，我们对实际匹配数据集的实证分析揭示了两个令人惊讶的发现：(1) 大多数游戏与BT模型和平坦性的假设显著不符，这提出了Elo评分可靠性的问题。(2) 尽管存在这些差异，Elo评分在很多情况下仍优于更复杂的评分系统，如mElo和两两对比模型，这些系统专门设计用来考虑数据中的非BT成分，特别是在胜率预测方面。

本文通过三个关键视角解释了这一意想不到的现象：(a) 我们重新解释了Elo评分作为一种在线梯度下降实例，即使在错定和非平稳设置中也能提供无悔保证。(b) 通过在从传递性但非BT模型生成的数据上进行广泛合成实验，例如强传递或弱传递性模型，我们展示了“实际匹配数据的稀疏性”是Elo评分在预测性能上优于更复杂评分系统的关键因素。(c) 我们发现Elo评分的预测准确性与排名表现之间存在强烈相关性，进一步支持了Elo评分在排名中的有效性。 

---
# QuOTE: Question-Oriented Text Embeddings 

**Title (ZH)**: 引言：面向问题的文本嵌入 

**Authors**: Andrew Neeser, Kaylen Latimer, Aadyant Khatri, Chris Latimer, Naren Ramakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10976)  

**Abstract**: We present QuOTE (Question-Oriented Text Embeddings), a novel enhancement to retrieval-augmented generation (RAG) systems, aimed at improving document representation for accurate and nuanced retrieval. Unlike traditional RAG pipelines, which rely on embedding raw text chunks, QuOTE augments chunks with hypothetical questions that the chunk can potentially answer, enriching the representation space. This better aligns document embeddings with user query semantics, and helps address issues such as ambiguity and context-dependent relevance. Through extensive experiments across diverse benchmarks, we demonstrate that QuOTE significantly enhances retrieval accuracy, including in multi-hop question-answering tasks. Our findings highlight the versatility of question generation as a fundamental indexing strategy, opening new avenues for integrating question generation into retrieval-based AI pipelines. 

**Abstract (ZH)**: 我们提出了QuOTE（问题导向的文本嵌入），这是一种面向检索增强生成（RAG）系统的新型增强方法，旨在改进文档表示以实现精确和细腻的检索。与传统的RAG流水线依赖于嵌入原始文本片段不同，QuOTE 通过将假设性问题添加到片段中，这些假设性问题是片段可能回答的问题，从而丰富了表示空间。这使得文档嵌入更好地与用户查询语义对齐，并有助于解决诸如歧义性和上下文相关性等问题。通过在多种基准测试上的大量实验，我们展示了QuOTE 显著提高了检索准确性，包括在多步问答任务中的表现。我们的研究结果强调了问题生成作为基本索引策略的灵活性，为将问题生成整合到基于检索的AI流水线中打开了新的途径。 

---
# Neural Networks Remember More: The Power of Parameter Isolation and Combination 

**Title (ZH)**: 神经网络记忆更多：参数隔离与组合的力量 

**Authors**: Biqing Zeng, Zehan Li, Aladdin Ayesh  

**Link**: [PDF](https://arxiv.org/pdf/2502.10966)  

**Abstract**: Catastrophic forgetting is a pervasive issue for pre-trained language models (PLMs) during continual learning, where models lose previously acquired knowledge when sequentially trained on a series of tasks. The model's ability to retain old tasks is referred to as stability, while its adaptability to new tasks is called plasticity. Therefore, the key to solving this problem is to find a trade-off between the plasticity and stability of the model. To address this issue, in this paper, we propose a novel method to achieve a balance between model stability and plasticity, thereby mitigating catastrophic forgetting. More specifically, our proposed approach leverages parameter isolation and a subsequent combination strategy. Initially, in the training stage, the model adapts to each downstream task via a parameter isolation method to prevent potential interference among different tasks. We then combine all trained parameters, which contain acquired knowledge, using the task arithmetic method and finally apply them to the backbone model. Empirical evaluations on continual language learning benchmarks substantiate the effectiveness of our approach, revealing a marked enhancement over existing state-of-the-art approaches. 

**Abstract (ZH)**: 灾难性遗忘是预训练语言模型（PLMs）在连续学习过程中面临的一个普遍问题，当模型按顺序针对一系列任务进行训练时，它们会失去之前获得的知识。模型保留旧任务的能力称为稳定性，而适应新任务的能力称为可塑性。因此，解决这一问题的关键在于找到模型可塑性和稳定性的平衡。为了解决这一问题，本文提出了一种新的方法，旨在在模型的可塑性和稳定性之间取得平衡，从而缓解灾难性遗忘现象。具体而言，我们提出的方案利用参数隔离和后续组合策略。在训练阶段，模型通过参数隔离方法适应每个下游任务，以防止不同任务之间的潜在干扰。然后，我们使用任务算术方法结合所有训练参数（这些参数包含了已获得的知识），并将它们应用于主模型。在连续语言学习基准上的实证评估证明了我们方法的有效性，显示出相较于现有最先进的方法有显著改进。 

---
# Graders should cheat: privileged information enables expert-level automated evaluations 

**Title (ZH)**: 评分老师应当作弊：特权信息使自动化高级评估成为可能

解释：这句话讨论的是在评分过程中，如果评分老师能够获得一些特权信息（例如，学生的个人信息或作业的背景信息），那么他们可以利用这些信息进行更为精准和专业的评估。同样地，在自动化评估系统中，如果系统能够访问到类似的特权信息，那么它也有能力进行高级别的自动评估。这里“作弊”是指利用额外信息来获得更准确的结果，而不是指不正当的行为。 

**Authors**: Jin Peng Zhou, Sébastien M. R. Arnold, Nan Ding, Kilian Q. Weinberger, Nan Hua, Fei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2502.10961)  

**Abstract**: Auto-evaluating language models (LMs), i.e., using a grader LM to evaluate the candidate LM, is an appealing way to accelerate the evaluation process and the cost associated with it. But this presents a paradox: how can we trust the grader LM, which is presumably weaker than the candidate LM, to assess problems that are beyond the frontier of the capabilities of either model or both? For instance, today's LMs struggle on graduate-level physics and Olympiad-level math, making them unreliable graders in these domains.
We show that providing privileged information -- such as ground-truth solutions or problem-specific guidelines -- improves automated evaluations on such frontier problems. This approach offers two key advantages. First, it expands the range of problems where LMs graders apply. Specifically, weaker models can now rate the predictions of stronger models. Second, privileged information can be used to devise easier variations of challenging problems which improves the separability of different LMs on tasks where their performance is generally low. With this approach, general-purpose LM graders match the state of the art performance on RewardBench, surpassing almost all the specially-tuned models. LM graders also outperform individual human raters on Vibe-Eval, and approach human expert graders on Olympiad-level math problems. 

**Abstract (ZH)**: 自动评估语言模型（LMs），即使用一个评判LM来评估候选LM，是一种加速评估过程及其相关成本的方法。但是这引出了一个悖论：我们如何能信任一个理论上比候选LM更弱的评判LM去评估超出这两个模型当前能力的问题？例如，当今的LM在研究生物理水平和奥林匹克数学水平上表现不佳，这使得它们在这些领域作为评判者是不可靠的。

我们证明，提供特权信息（例如，真实答案或特定问题的指导方针）可以改善在前沿问题上的自动评估。这种方法有两大优势。首先，它扩展了LM评判器适用的问题范围。具体来说，较弱的模型现在可以评估较强模型的预测。其次，特权信息可以用来设计更具挑战性的问题的较简单的变体，从而提高了在这些任务上表现不佳的各个LM之间的可分辨性。借助这种方法，通用的LM评判器在RewardBench上达到了最先进的性能，超过了几乎所有的专门调整的模型。LM评判器在Vibe-Eval上也超过了单独的人类评判者，在奥林匹克数学问题上接近人类专家评判者。 

---
# A recurrent vision transformer shows signatures of primate visual attention 

**Title (ZH)**: 一种循环视觉变换器展示了灵长类视觉注意的特征 

**Authors**: Jonathan Morgan, Badr Albanna, James P. Herman  

**Link**: [PDF](https://arxiv.org/pdf/2502.10955)  

**Abstract**: Attention is fundamental to both biological and artificial intelligence, yet research on animal attention and AI self attention remains largely disconnected. We propose a Recurrent Vision Transformer (Recurrent ViT) that integrates self-attention with recurrent memory, allowing both current inputs and stored information to guide attention allocation. Trained solely via sparse reward feedback on a spatially cued orientation change detection task, a paradigm used in primate studies, our model exhibits primate like signatures of attention, including improved accuracy and faster responses for cued stimuli that scale with cue validity. Analysis of self-attention maps reveals dynamic spatial prioritization with reactivation prior to expected changes, and targeted perturbations produce performance shifts similar to those observed in primate frontal eye fields and superior colliculus. These findings demonstrate that incorporating recurrent feedback into self attention can capture key aspects of primate visual attention. 

**Abstract (ZH)**: 注意力是生物学和人工智书中基础性的组成部分，然而关于动物注意力和AI自我注意力的研究仍相对独立。我们提出了一种递归视觉变换器（Recurrent Vision Transformer，Recurrent ViT），它将自我注意力与递归记忆相结合，使得当前输入和存储的信息都能指导注意力分配。该模型仅通过基于空间提示的方向变化检测任务稀疏奖励反馈进行训练，在这一范式上与灵长类动物的研究一致，表现出与灵长类动物类似的注意力特征，包括在提示有效性的增加下提高了准确性并加快了对提示刺激的响应速度。对自我注意力图的分析揭示了动态的空间优先级化现象，在预期变化前进行再激活，并且针对特定部位的扰动导致了类似灵长类动物初级视皮层和上丘的表现变化。这些发现表明，在自我注意力中引入递归反馈可以捕获灵长类视觉注意力的关键方面。 

---
# Learning to Stop Overthinking at Test Time 

**Title (ZH)**: 在测试时学习停止过度思考 

**Authors**: Hieu Tran Bao, Nguyen Cong Dat, Nguyen Duc Anh, Hoang Thanh Tung  

**Link**: [PDF](https://arxiv.org/pdf/2502.10954)  

**Abstract**: Test time scaling is currently one of the most active research areas that shows promise after training time scaling has reached its limits. Deep-thinking (DT) models are a class of recurrent models that can perform easy-to-hard generalization by assigning more compute to harder test samples. However, due to their inability to determine the complexity of a test sample, DT models have to use a large amount of computation for both easy and hard test samples. Excessive test time computation is wasteful and can cause the ``overthinking'' problem where more test time computation leads to worse results. In this paper, we introduce a test time training method for determining the optimal amount of computation needed for each sample during test time. We also propose Conv-LiGRU, a novel recurrent architecture for efficient and robust visual reasoning. Extensive experiments demonstrate that Conv-LiGRU is more stable than DT, effectively mitigates the ``overthinking'' phenomenon, and achieves superior accuracy. 

**Abstract (ZH)**: 当前，测试时缩放是继训练时缩放达到极限后最具活力的研究领域之一。深度思考（DT）模型是一类递归模型，能够通过将更多的计算资源分配给更难的测试样本，实现从易到难的泛化。然而，由于这些模型无法确定测试样本的复杂性，因此需要为所有测试样本使用大量的计算资源。过度的测试时计算资源使用是浪费的，并可能导致“过度思考”问题，即更多的计算资源反而会导致性能下降。本文提出了一种测试时训练方法，以确定测试时需要为每个样本分配的最佳计算量。我们还提出了一种新的递归架构Conv-LiGRU，该架构旨在实现高效的鲁棒视觉推理。广泛实验表明，Conv-LiGRU相比DT更为稳定，有效缓解了“过度思考”现象，并达到了更高的准确性。 

---
# Empirical evaluation of LLMs in predicting fixes of Configuration bugs in Smart Home System 

**Title (ZH)**: 智能家庭系统中配置错误修复预测的大型语言模型实证评估 

**Authors**: Sheikh Moonwara Anjum Monisha, Atul Bharadwaj  

**Link**: [PDF](https://arxiv.org/pdf/2502.10953)  

**Abstract**: This empirical study evaluates the effectiveness of Large Language Models (LLMs) in predicting fixes for configuration bugs in smart home systems. The research analyzes three prominent LLMs - GPT-4, GPT-4o (GPT-4 Turbo), and Claude 3.5 Sonnet - using four distinct prompt designs to assess their ability to identify appropriate fix strategies and generate correct solutions. The study utilized a dataset of 129 debugging issues from the Home Assistant Community, focusing on 21 randomly selected cases for in-depth analysis. Results demonstrate that GPT-4 and Claude 3.5 Sonnet achieved 80\% accuracy in strategy prediction when provided with both bug descriptions and original scripts. GPT-4 exhibited consistent performance across different prompt types, while GPT-4o showed advantages in speed and cost-effectiveness despite slightly lower accuracy. The findings reveal that prompt design significantly impacts model performance, with comprehensive prompts containing both description and original script yielding the best results. This research provides valuable insights for improving automated bug fixing in smart home system configurations and demonstrates the potential of LLMs in addressing configuration-related challenges. 

**Abstract (ZH)**: 本实证研究评估了大型语言模型（LLMs）在预测智能家居系统配置错误修复方案方面的有效性。研究分析了三种知名的LLMs——GPT-4、GPT-4o（GPT-4 Turbo）和Claude 3.5 Sonnet，并使用四种不同的提示设计来评估它们识别合适修复策略和生成正确解决方案的能力。研究利用了来自Home Assistant Community的数据集，其中包含129个调试问题，并对21个随机选取的案例进行了深入分析。结果显示，当提供错误描述和原始脚本时，GPT-4和Claude 3.5 Sonnet在策略预测方面的准确率达到了80%。GPT-4在不同提示类型中表现出一致的性能，而GPT-4o在速度和成本效益方面表现出优势，尽管准确率略低一些。研究发现，提示设计对模型性能有显著影响，包含描述和原始脚本的全面提示能取得最佳结果。本研究为提高智能家居系统配置的自动错误修复提供了有价值的见解，并展示了LLMs在解决配置相关挑战方面的潜力。 

---
# CoLA: Compute-Efficient Pre-Training of LLMs via Low-Rank Activation 

**Title (ZH)**: CoLA：通过低秩激活实现LLM的计算高效预训练 

**Authors**: Ziyue Liu, Ruijie Zhang, Zhengyang Wang, Zi Yang, Paul Hovland, Bogdan Nicolae, Franck Cappello, Zheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10940)  

**Abstract**: Large language models (LLMs) are revolutionizing many science and engineering fields. However, their huge model sizes impose extremely demanding needs of computational resources in the pre-training stage. Although low-rank factorizations can reduce model parameters, their direct application in LLM pre-training often lead to non-negligible performance loss. To address this fundamental challenge, we introduce CoLA and its memory-efficient implementation, CoLA-M. We leverage the low-rank structure observed widely in model activations, enforcing non-linear transformations between factorized weight matrices to reduce model size, boost model capacity and training efficiency. Experiments on LLaMA models with 60 million to 7 billion parameters show that CoLA reduces the computing cost by $\bf 2\pmb{\times}$ and improves training throughput by $\bf 1.86\pmb{\times}$ while maintaining full-rank level performance. CoLA-M further squeezes memory cost without sacrificing throughput, offering a pre-training approach with collectively superior parameter, computing, and memory efficiency. The LLMs produced are also $\bf 2\pmb{\times}$ smaller, enabling faster inference with lower memory cost on resource-constrained platforms 

**Abstract (ZH)**: 大规模语言模型（LLMs）正在革新许多科学和技术领域。然而，它们巨大的模型规模在预训练阶段对计算资源提出了极高的需求。尽管低秩分解可以减少模型参数，但在LLM预训练中的直接应用往往会导致不可忽视的性能损失。为了解决这一基本挑战，我们引入了CoLA及其高效的实现CoLA-M。我们利用模型激活中广泛观察到的低秩结构，在因子化的权重矩阵之间引入非线性变换，以减少模型大小、提升模型容量和训练效率。在参数量从6000万到7亿的LLaMA模型上进行的实验表明，使用CoLA可以将计算成本减少$\bf 2\pmb{\times}$，同时将训练吞吐量提高$\bf 1.86\pmb{\times}$，同时仍然保持全秩级别的性能。CoLA-M进一步在不牺牲吞吐量的情况下压缩了内存成本，提供了在参数、计算和内存效率方面综合表现更优的预训练方法。生成的LLMs体积也减少了$\bf 2\pmb{\times}$，这使得在资源受限的平台上实现更快的推理并降低内存成本成为可能。 

---
# Semantic Specialization in MoE Appears with Scale: A Study of DeepSeek R1 Expert Specialization 

**Title (ZH)**: 随着规模的增加，MoE 中的语义专业化现象显现：对 DeepSeek R1 专家专业化研究

解释：
- Semantic Specialization 表示语义专业化。
- MoE 是指 Mixture of Experts，即专家混合模型。
- DeepSeek R1 指的是某个特定的研究或模型。
- " Appears with Scale" 表示随着规模的增加，某种现象显现。
- "A Study of" 表示对某方面的研究。

这个翻译既保留了原文的学术风格，又确保了中文的流畅和准确性。 

**Authors**: Matthew Lyle Olson, Neale Ratzlaff, Musashi Hinck, Man Luo, Sungduk Yu, Chendi Xue, Vasudev Lal  

**Link**: [PDF](https://arxiv.org/pdf/2502.10928)  

**Abstract**: DeepSeek-R1, the largest open-source Mixture-of-Experts (MoE) model, has demonstrated reasoning capabilities comparable to proprietary frontier models. Prior research has explored expert routing in MoE models, but findings suggest that expert selection is often token-dependent rather than semantically driven. Given DeepSeek-R1's enhanced reasoning abilities, we investigate whether its routing mechanism exhibits greater semantic specialization than previous MoE models. To explore this, we conduct two key experiments: (1) a word sense disambiguation task, where we examine expert activation patterns for words with differing senses, and (2) a cognitive reasoning analysis, where we assess DeepSeek-R1's structured thought process in an interactive task setting of DiscoveryWorld. We conclude that DeepSeek-R1's routing mechanism is more semantically aware and it engages in structured cognitive processes. 

**Abstract (ZH)**: DeepSeek-R1 是目前最大的开源 Mixture-of-Experts (MoE) 模型，其展示出的推理能力与商业前沿模型相当。先前的研究探讨了 MoE 模型中的专家路由机制，但结果表明，专家选择通常依赖于标记而非语义驱动。鉴于 DeepSeek-R1 增强的推理能力，我们研究其路由机制是否比先前的 MoE 模型展现出更显著的语义专一性。为此，我们进行了两项关键实验：(1) 词义消歧实验，我们研究不同词义的词在专家激活模式中的差异，(2) 认知推理分析，我们评估 DeepSeek-R1 在 DiscoveryWorld 的互动任务设置中的有组织的思维过程。我们得出结论，DeepSeek-R1 的路由机制更有语义意识，并且能够进行有组织的认知推理过程。 

---
# Do Deepfake Detectors Work in Reality? 

**Title (ZH)**: 深度假 faces 的检测器在现实世界中有效吗？ 

**Authors**: Simiao Ren, Hengwei Xu, Tsang Ng, Kidus Zewde, Shengkai Jiang, Ramini Desai, Disha Patil, Ning-Yau Cheng, Yining Zhou, Ragavi Muthukrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10920)  

**Abstract**: Deepfakes, particularly those involving faceswap-based manipulations, have sparked significant societal concern due to their increasing realism and potential for misuse. Despite rapid advancements in generative models, detection methods have not kept pace, creating a critical gap in defense strategies. This disparity is further amplified by the disconnect between academic research and real-world applications, which often prioritize different objectives and evaluation criteria. In this study, we take a pivotal step toward bridging this gap by presenting a novel observation: the post-processing step of super-resolution, commonly employed in real-world scenarios, substantially undermines the effectiveness of existing deepfake detection methods. To substantiate this claim, we introduce and publish the first real-world faceswap dataset, collected from popular online faceswap platforms. We then qualitatively evaluate the performance of state-of-the-art deepfake detectors on real-world deepfakes, revealing that their accuracy approaches the level of random guessing. Furthermore, we quantitatively demonstrate the significant performance degradation caused by common post-processing techniques. By addressing this overlooked challenge, our study underscores a critical avenue for enhancing the robustness and practical applicability of deepfake detection methods in real-world settings. 

**Abstract (ZH)**: 深度伪造，尤其是基于面部替换的编辑，由于其不断增强的逼真度和潜在的滥用风险，引起了显著的公众关注。尽管生成模型取得了快速进展，但检测方法并未跟上步伐，导致在防伪策略上出现了一个关键缺口。这一差距进一步加剧了学术研究与实际应用之间的脱节，后者往往优先考虑不同的目标和评估标准。本研究通过提出一项新颖的观察，向前迈出了一大步：即在现实场景中常用的超分辨率后处理步骤极大地削弱了现有深度伪造检测方法的有效性。为了证明这一主张，我们引入并发布了首个现实场景中的面部替换数据集，该数据集来源于受欢迎的在线面部替换平台。然后，我们对最先进的深度伪造检测器在真实世界中的性能进行了定性评估，结果显示其准确性接近随机猜测的水平。此外，我们还定量展示了常见后处理技术导致的显著性能下降。通过解决这一被忽视的挑战，我们的研究强调了一个关键方向，即增强深度伪造检测方法在现实环境中的鲁棒性和实用性。 

---
# Automatic Quality Assessment of First Trimester Crown-Rump-Length Ultrasound Images 

**Title (ZH)**: 自动评估早期妊娠 Crown-Rump 长度超声图像的质量 

**Authors**: Sevim Cengiz, Ibraheem Hamdi, Mohammad Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2502.10908)  

**Abstract**: Fetal gestational age (GA) is vital clinical information that is estimated during pregnancy in order to assess fetal growth. This is usually performed by measuring the crown-rump-length (CRL) on an ultrasound image in the Dating scan which is then correlated with fetal age and growth trajectory. A major issue when performing the CRL measurement is ensuring that the image is acquired at the correct view, otherwise it could be misleading. Although clinical guidelines specify the criteria for the correct CRL view, sonographers may not regularly adhere to such rules. In this paper, we propose a new deep learning-based solution that is able to verify the adherence of a CRL image to clinical guidelines in order to assess image quality and facilitate accurate estimation of GA. We first segment out important fetal structures then use the localized structures to perform a clinically-guided mapping that verifies the adherence of criteria. The segmentation method combines the benefits of Convolutional Neural Network (CNN) and the Vision Transformer (ViT) to segment fetal structures in ultrasound images and localize important fetal landmarks. For segmentation purposes, we compare our proposed work with UNet and show that our CNN/ViT-based method outperforms an optimized version of UNet. Furthermore, we compare the output of the mapping with classification CNNs when assessing the clinical criteria and the overall acceptability of CRL images. We show that the proposed mapping is not only explainable but also more accurate than the best performing classification CNNs. 

**Abstract (ZH)**: 胎儿妊娠年龄（GA）是怀孕期间评估胎儿生长所需的关键临床信息，通常通过在早期妊娠筛查中测量胎儿头臀长（CRL）来估算。这涉及确保获取图像的角度正确，否则可能会产生误导。尽管临床指南规定了正确的CRL视图标准，但超声技师可能并不总是遵循这些规则。本文提出了一种基于深度学习的新解决方案，用于验证CRL图像是否符合临床指南，以评估图像质量并促进GA的准确估算。我们首先对重要的胎儿结构进行分割，然后利用定位的结构进行临床指导的映射，以验证是否符合标准。分割方法结合了卷积神经网络（CNN）和视觉变压器（ViT）的优势，以分割超声图像中的胎儿结构并定位重要的胎儿解剖标志。为了分割目的，我们与UNet进行了比较，并证明基于CNN/ViT的方法优于优化后的UNet。此外，我们在评估临床标准和CRL图像的整体接受度时，将映射的输出与分类CNN进行了比较。研究表明，所提出的映射不仅具有可解释性，而且在准确性方面也超过了最优秀的分类CNN。 

---
# Breaking Down the Hierarchy: A New Approach to Leukemia Classification 

**Title (ZH)**: 打破等级结构：一种新的白血病分类方法 

**Authors**: Ibraheem Hamdi, Hosam El-Gendy, Ahmed Sharshar, Mohamed Saeed, Muhammad Ridzuan, Shahrukh K. Hashmi, Naveed Syed, Imran Mirza, Shakir Hussain, Amira Mahmoud Abdalla, Mohammad Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2502.10899)  

**Abstract**: The complexities inherent to leukemia, multifaceted cancer affecting white blood cells, pose considerable diagnostic and treatment challenges, primarily due to reliance on laborious morphological analyses and expert judgment that are susceptible to errors. Addressing these challenges, this study presents a refined, comprehensive strategy leveraging advanced deep-learning techniques for the classification of leukemia subtypes. We commence by developing a hierarchical label taxonomy, paving the way for differentiating between various subtypes of leukemia. The research further introduces a novel hierarchical approach inspired by clinical procedures capable of accurately classifying diverse types of leukemia alongside reactive and healthy cells. An integral part of this study involves a meticulous examination of the performance of Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) as classifiers. The proposed method exhibits an impressive success rate, achieving approximately 90\% accuracy across all leukemia subtypes, as substantiated by our experimental results. A visual representation of the experimental findings is provided to enhance the model's explainability and aid in understanding the classification process. 

**Abstract (ZH)**: 血液学的复杂性构成了挑战，血液癌这种多面性癌症影响白血球，主要由于依赖耗时的形态学分析和容易出错的专家判断，这给诊断和治疗带来了重大挑战。本研究旨在应对这些挑战，提出了一种利用先进深度学习技术的精炼且全面的策略，用于白血病亚型的分类。首先，我们构建了一个层次化的标签分类体系，为区分各种白血病亚型奠定了基础。研究进一步引入了一种受临床流程启发的层次化方法，该方法能够准确地对多种类型的白血病以及反应性和健康细胞进行分类。该研究的重要组成部分是对卷积神经网络（CNNs）和视觉变换器（ViTs）作为分类器的性能进行了细致分析。所提出的方法取得了显著的成功，实验结果表明，其在所有白血病亚型中的准确率约为90%，并通过可视化实验结果增强了模型的可解释性，有助于理解分类过程。 

---
# Bridging the Sim-to-Real Gap for Athletic Loco-Manipulation 

**Title (ZH)**: 将体育运动中的模拟与现实差距进行 bridging：运动员的运动与操作衔接研究 

**Authors**: Nolan Fey, Gabriel B. Margolis, Martin Peticco, Pulkit Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2502.10894)  

**Abstract**: Achieving athletic loco-manipulation on robots requires moving beyond traditional tracking rewards - which simply guide the robot along a reference trajectory - to task rewards that drive truly dynamic, goal-oriented behaviors. Commands such as "throw the ball as far as you can" or "lift the weight as quickly as possible" compel the robot to exhibit the agility and power inherent in athletic performance. However, training solely with task rewards introduces two major challenges: these rewards are prone to exploitation (reward hacking), and the exploration process can lack sufficient direction. To address these issues, we propose a two-stage training pipeline. First, we introduce the Unsupervised Actuator Net (UAN), which leverages real-world data to bridge the sim-to-real gap for complex actuation mechanisms without requiring access to torque sensing. UAN mitigates reward hacking by ensuring that the learned behaviors remain robust and transferable. Second, we use a pre-training and fine-tuning strategy that leverages reference trajectories as initial hints to guide exploration. With these innovations, our robot athlete learns to lift, throw, and drag with remarkable fidelity from simulation to reality. 

**Abstract (ZH)**: 实现机器人运动员的运动操控需超越传统的跟踪奖励，后者仅引导机器人沿参考轨迹移动。转而应当采用任务奖励来驱动真正动态且目标导向的行为。例如，“尽可能远地投球”或“尽可能快地举起 weights”等指令促使机器人展现出与运动表现相符的敏捷性和力量。然而，仅使用任务奖励进行训练会带来两大主要挑战：这些奖励容易被利用（奖励破解），且探索过程可能缺乏足够的方向性。为解决这些问题，我们提出了一种双阶段训练管道。首先，我们引入了无监督执行网络（UAN），该网络利用实际数据跨越复杂执行机制的模拟到现实差距，无需扭矩感知。UAN通过确保学习到的行为保持稳健性和可移植性来减轻奖励破解的问题。其次，我们采用一种预训练和微调策略，利用参考轨迹作为初始提示来引导探索。凭借这些创新，我们的机器人运动员能够从模拟到现实以惊人的准确性学会举起、投掷和拖拽动作。 

---
# Learning Identifiable Structures Helps Avoid Bias in DNN-based Supervised Causal Learning 

**Title (ZH)**: 学习可识别结构有助于避免基于DNN的监督因果学习中的偏差 

**Authors**: Jiaru Zhang, Rui Ding, Qiang Fu, Bojun Huang, Zizhen Deng, Yang Hua, Haibing Guan, Shi Han, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10883)  

**Abstract**: Causal discovery is a structured prediction task that aims to predict causal relations among variables based on their data samples. Supervised Causal Learning (SCL) is an emerging paradigm in this field. Existing Deep Neural Network (DNN)-based methods commonly adopt the "Node-Edge approach", in which the model first computes an embedding vector for each variable-node, then uses these variable-wise representations to concurrently and independently predict for each directed causal-edge. In this paper, we first show that this architecture has some systematic bias that cannot be mitigated regardless of model size and data size. We then propose SiCL, a DNN-based SCL method that predicts a skeleton matrix together with a v-tensor (a third-order tensor representing the v-structures). According to the Markov Equivalence Class (MEC) theory, both the skeleton and the v-structures are identifiable causal structures under the canonical MEC setting, so predictions about skeleton and v-structures do not suffer from the identifiability limit in causal discovery, thus SiCL can avoid the systematic bias in Node-Edge architecture, and enable consistent estimators for causal discovery. Moreover, SiCL is also equipped with a specially designed pairwise encoder module with a unidirectional attention layer to model both internal and external relationships of pairs of nodes. Experimental results on both synthetic and real-world benchmarks show that SiCL significantly outperforms other DNN-based SCL approaches. 

**Abstract (ZH)**: 因果发现是一个结构预测任务，旨在根据变量的数据样本预测变量间的因果关系。监督因果学习（Supervised Causal Learning, SCL）是该领域的一个新兴范式。现有的基于深度神经网络（Deep Neural Network, DNN）的方法通常采用“节点-边”方法，在这种方法中，模型首先为每个变量节点计算一个嵌入向量，然后使用这些基于变量的表示来独立并同时预测每个有向因果边。本文首先表明，这种架构存在一些系统偏差，无论模型规模和数据规模如何，这种偏差都无法消除。然后，我们提出了一种基于DNN的方法——SiCL，它同时预测一个骨架矩阵和一个v-张量（一个用来表示v结构的三阶张量）。根据马尔可夫等价类（Markov Equivalence Class, MEC）理论，在标准的MEC设定下，骨架和v-结构都是可以识别的因果结构，因此关于骨架和v-结构的预测不受因果发现中的可识别性限制，因此SiCL可以规避“节点-边”架构中的系统偏差，并为因果发现提供一致估计。此外，SiCL还配备了一个特别设计的成对编码器模块，其中包含了一个单向注意力层，用于建模节点对的内部和外部关系。在合成数据和真实世界数据基准上的实验结果表明，SiCL在与其他基于DNN的SCL方法的比较中表现显著更优。 

---
# Broadcast Channel Cooperative Gain: An Operational Interpretation of Partial Information Decomposition 

**Title (ZH)**: 广播信道协作增益：部分信息分解的操作诠释 

**Authors**: Chao Tian, Shlomo Shamai  

**Link**: [PDF](https://arxiv.org/pdf/2502.10878)  

**Abstract**: Partial information decomposition has recently found applications in biological signal processing and machine learning. Despite its impacts, the decomposition was introduced through an informal and heuristic route, and its exact operational meaning is unclear. In this work, we fill this gap by connecting partial information decomposition to the capacity of the broadcast channel, which has been well-studied in the information theory literature. We show that the synergistic information in the decomposition can be rigorously interpreted as the cooperative gain, or a lower bound of this gain, on the corresponding broadcast channel. This interpretation can help practitioners to better explain and expand the applications of the partial information decomposition technique. 

**Abstract (ZH)**: 部分信息分解最近在生物信号处理和机器学习领域找到了应用。尽管它产生了显著影响，该分解方法最初是通过非正式和启发式的方式引入的，其确切的操作含义仍不清楚。在本文中，我们通过将部分信息分解与广播信道的容量联系起来，填补了这一空白。广播信道已经在信息论文献中得到了充分的研究。我们证明，在分解中所包含的协同信息可以严格地解释为相应的广播信道上的协同增益，或者这一增益的下界。这种解释可以帮助实践者更好地解释和扩展部分信息分解技术的应用。 

---
# A Geometric Approach to Personalized Recommendation with Set-Theoretic Constraints Using Box Embeddings 

**Title (ZH)**: 使用盒嵌入的集合论约束个性化推荐的几何方法 

**Authors**: Shib Dasgupta, Michael Boratko, Andrew McCallum  

**Link**: [PDF](https://arxiv.org/pdf/2502.10875)  

**Abstract**: Personalized item recommendation typically suffers from data sparsity, which is most often addressed by learning vector representations of users and items via low-rank matrix factorization. While this effectively densifies the matrix by assuming users and movies can be represented by linearly dependent latent features, it does not capture more complicated interactions. For example, vector representations struggle with set-theoretic relationships, such as negation and intersection, e.g. recommending a movie that is "comedy and action, but not romance". In this work, we formulate the problem of personalized item recommendation as matrix completion where rows are set-theoretically dependent. To capture this set-theoretic dependence we represent each user and attribute by a hyper-rectangle or box (i.e. a Cartesian product of intervals). Box embeddings can intuitively be understood as trainable Venn diagrams, and thus not only inherently represent similarity (via the Jaccard index), but also naturally and faithfully support arbitrary set-theoretic relationships. Queries involving set-theoretic constraints can be efficiently computed directly on the embedding space by performing geometric operations on the representations. We empirically demonstrate the superiority of box embeddings over vector-based neural methods on both simple and complex item recommendation queries by up to 30 \% overall. 

**Abstract (ZH)**: 个性化项目推荐通常会遇到数据稀疏性的问题，这通常是通过低秩矩阵分解来学习用户和项目的向量表示来解决的。尽管这种方法假设用户和项目可以由线性相关的潜在特征来表示，从而有效地增加了矩阵的密度，但它无法捕捉到更复杂的交互关系。例如，向量表示在处理集合理论关系（如否定和交集）方面存在困难，比如推荐一部“喜剧和动作，但不是浪漫”的电影。在本项工作中，我们将个性化项目推荐问题形式化为矩阵填充问题，其中行是集合理论上相互依赖的。为了捕捉这种集合理论依赖，我们将每个用户和属性表示为超矩形或盒子（即区间集的笛卡尔积）。超矩形嵌入可以直观地理解为可训练的文恩图，因此不仅能够通过约简指数自然地表示相似性，还能自然和忠实地支持任意的集合理论关系。通过在嵌入空间中执行几何操作，可以高效地计算包含集合理论约束的查询。我们通过在简单和复杂的项目推荐查询上的实验证明，超矩形嵌入在整体上比基于向量的神经网络方法优越至多30%。 

---
# The Representation and Recall of Interwoven Structured Knowledge in LLMs: A Geometric and Layered Analysis 

**Title (ZH)**: 语言模型中交错结构化知识的表示与回忆：一种几何与分层分析 

**Authors**: Ge Lei, Samuel J. Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2502.10871)  

**Abstract**: This study investigates how large language models (LLMs) represent and recall multi-associated attributes across transformer layers. We show that intermediate layers encode factual knowledge by superimposing related attributes in overlapping spaces, along with effective recall even when attributes are not explicitly prompted. In contrast, later layers refine linguistic patterns and progressively separate attribute representations, optimizing task-specific outputs while appropriately narrowing attribute recall. We identify diverse encoding patterns including, for the first time, the observation of 3D spiral structures when exploring information related to the periodic table of elements. Our findings reveal a dynamic transition in attribute representations across layers, contributing to mechanistic interpretability and providing insights for understanding how LLMs handle complex, interrelated knowledge. 

**Abstract (ZH)**: 本研究考察了大型语言模型（LLM）如何在各变压器层中表示和回忆多关联属性。研究表明，中间层通过在重叠的空间中叠加相关属性来编码事实性知识，并在未明确提示属性的情况下也能有效地回忆这些属性。相比之下，后期层则精炼语言模式，并逐步分离属性表示，同时优化特定任务输出并在适当范围内限制属性回忆。我们识别了多种编码模式，其中包括首次观察到当探索元素周期表相关信息时，呈现出三维螺旋结构。我们的研究发现揭示了层间属性表示的动态转变，有助于实现机制解释，并为理解LLM处理复杂且相互关联的知识提供了启示。 

---
# Multilingual Encoder Knows more than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages 

**Title (ZH)**: 多语言编码器远比你想象的更为强大：共享权重预训练用于极端低资源语言 

**Authors**: Zeli Su, Ziyin Zhang, Guixian Xu, Jianing Liu, XU Han, Ting Zhang, Yushuang Dong  

**Link**: [PDF](https://arxiv.org/pdf/2502.10852)  

**Abstract**: While multilingual language models like XLM-R have advanced multilingualism in NLP, they still perform poorly in extremely low-resource languages. This situation is exacerbated by the fact that modern LLMs such as LLaMA and Qwen support far fewer languages than XLM-R, making text generation models non-existent for many languages in the world. To tackle this challenge, we propose a novel framework for adapting multilingual encoders to text generation in extremely low-resource languages. By reusing the weights between the encoder and the decoder, our framework allows the model to leverage the learned semantic space of the encoder, enabling efficient learning and effective generalization in low-resource languages. Applying this framework to four Chinese minority languages, we present XLM-SWCM, and demonstrate its superior performance on various downstream tasks even when compared with much larger models. 

**Abstract (ZH)**: 尽管像XLM-R这样的多语言模型在自然语言处理（NLP）方面推进了多语言能力的发展，但在极端低资源语言方面仍表现不佳。这一问题进一步加剧了现代大型语言模型（LLM）如LLaMA和Qwen所支持语言种类较少的问题，使得许多语言在文本生成模型中无处立足。为解决这一挑战，我们提出了一种新颖的框架，用于将多语言编码器适应到极端低资源语言的文本生成任务中。通过在编码器和解码器之间复用权重，该框架使模型能够利用编码器学习到的语义空间，从而在低资源语言中实现高效的训练和有效的泛化。我们将此框架应用于四种汉语少数民族语言，推出了XLM-SWCM，并在各种下游任务中展示了其优于更大规模模型的出色性能。 

---
# The Vendiscope: An Algorithmic Microscope For Data Collections 

**Title (ZH)**: Vendiscope：数据集合的算法显微镜 

**Authors**: Amey P. Pasarkar, Adji Bousso Dieng  

**Link**: [PDF](https://arxiv.org/pdf/2502.10828)  

**Abstract**: The evolution of microscopy, beginning with its invention in the late 16th century, has continuously enhanced our ability to explore and understand the microscopic world, enabling increasingly detailed observations of structures and phenomena. In parallel, the rise of data-driven science has underscored the need for sophisticated methods to explore and understand the composition of complex data collections. This paper introduces the Vendiscope, the first algorithmic microscope designed to extend traditional microscopy to computational analysis. The Vendiscope leverages the Vendi scores -- a family of differentiable diversity metrics rooted in ecology and quantum mechanics -- and assigns weights to data points based on their contribution to the overall diversity of the collection. These weights enable high-resolution data analysis at scale. We demonstrate this across biology, materials science, and machine learning (ML). We analyzed the $250$ million protein sequences in the protein universe, discovering that over $200$ million are near-duplicates and that AlphaFold fails on proteins with Gene Ontology (GO) functions that contribute most to diversity. Applying the Vendiscope to the Materials Project database led to similar findings: more than $85\%$ of the crystals with formation energy data are near-duplicates and ML models perform poorly on materials that enhance diversity. Additionally, the Vendiscope can be used to study phenomena such as memorization in generative models. We used the Vendiscope to identify memorized training samples from $13$ different generative models and found that the best-performing ones often memorize the training samples that contribute least to diversity. Our findings demonstrate that the Vendiscope can serve as a powerful tool for data-driven science. 

**Abstract (ZH)**: 显微镜自16世纪末发明以来的演变，不断增强了我们探索和理解微观世界的 ability，使得我们能够对结构和现象进行越来越详细的观察。与此同时，数据驱动科学的发展强调了需要更加 sophisticated的方法来探索和理解复杂数据集的组成。本文介绍了一种称为Vendiscope的新算法显微镜，它是第一种旨在将传统显微镜扩展到计算分析的方法。Vendiscope利用了Vendi分数——一套源自生态学和量子力学的可微分多样性度量——并根据数据点对集合整体多样性贡献的大小为其分配权重。这些权重使大规模高分辨率数据分析成为可能。我们在这项研究中涵盖了生物学、材料科学和机器学习（ML）等多个领域。我们分析了蛋白质宇宙中的2.5亿个蛋白质序列，发现其中超过2亿个是准重复序列，AlphaFold在那些基因 Ontology（GO）功能对多样性贡献最大的蛋白质上表现不佳。将Vendiscope应用于Materials Project数据库也得到了类似的发现：超过85%具备形成能量数据的晶体是准重复序列，而机器学习模型在那些增强多样性的材料上表现较差。此外，Vendiscope还可以用于研究生成模型中的记忆现象。我们使用Vendiscope识别了来自13个不同生成模型的训练样本，并发现表现最佳的模型通常记住的是对多样性贡献最小的训练样本。我们的研究结果表明，Vendiscope可以作为数据驱动科学的强大工具。 

---
# MITRE ATT&CK Applications in Cybersecurity and The Way Forward 

**Title (ZH)**: MITRE ATT&CK在网络安全中的应用及未来展望 

**Authors**: Yuning Jiang, Qiaoran Meng, Feiyang Shang, Nay Oo, Le Thi Hong Minh, Hoon Wei Lim, Biplab Sikdar  

**Link**: [PDF](https://arxiv.org/pdf/2502.10825)  

**Abstract**: The MITRE ATT&CK framework is a widely adopted tool for enhancing cybersecurity, supporting threat intelligence, incident response, attack modeling, and vulnerability prioritization. This paper synthesizes research on its application across these domains by analyzing 417 peer-reviewed publications. We identify commonly used adversarial tactics, techniques, and procedures (TTPs) and examine the integration of natural language processing (NLP) and machine learning (ML) with ATT&CK to improve threat detection and response. Additionally, we explore the interoperability of ATT&CK with other frameworks, such as the Cyber Kill Chain, NIST guidelines, and STRIDE, highlighting its versatility. The paper further evaluates the framework from multiple perspectives, including its effectiveness, validation methods, and sector-specific challenges, particularly in industrial control systems (ICS) and healthcare. We conclude by discussing current limitations and proposing future research directions to enhance the applicability of ATT&CK in dynamic cybersecurity environments. 

**Abstract (ZH)**: 麻省理工学院ATT&CK框架是一个广泛采用的工具，用于增强网络安全、支持威胁情报、紧急事件响应、攻击建模和漏洞优先级排序。本文通过分析417篇同行评议的文献，综合研究了其在这些领域的应用。我们识别出了常用对手战术、技术和程序（TTPs），并探讨了自然语言处理（NLP）和机器学习（ML）与ATT&CK的整合，以提高威胁检测和响应能力。此外，我们还探讨了ATT&CK与其他框架（如网络杀伤链、NIST指南和STRIDE）的互操作性，突显了其灵活性。文章进一步从多个角度评估了该框架，包括其有效性、验证方法以及在工业控制系统（ICS）和医疗保健等特定领域的挑战。最后，我们讨论了当前的局限性，并提出了未来研究方向，以增强ATT&CK在动态网络环境中适用性。 

---
# NeuroAMP: A Novel End-to-end General Purpose Deep Neural Amplifier for Personalized Hearing Aids 

**Title (ZH)**: NeuroAMP：一种新型端到端通用深度神经放大器，用于个性化助听器 

**Authors**: Shafique Ahmed, Ryandhimas E. Zezario, Hui-Guan Yuan, Amir Hussain, Hsin-Min Wang, Wei-Ho Chung, Yu Tsao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10822)  

**Abstract**: The prevalence of hearing aids is increasing. However, optimizing the amplification processes of hearing aids remains challenging due to the complexity of integrating multiple modular components in traditional methods. To address this challenge, we present NeuroAMP, a novel deep neural network designed for end-to-end, personalized amplification in hearing aids. NeuroAMP leverages both spectral features and the listener's audiogram as inputs, and we investigate four architectures: Convolutional Neural Network (CNN), Long Short-Term Memory (LSTM), Convolutional Recurrent Neural Network (CRNN), and Transformer. We also introduce Denoising NeuroAMP, an extension that integrates noise reduction along with amplification capabilities for improved performance in real-world scenarios. To enhance generalization, a comprehensive data augmentation strategy was employed during training on diverse speech (TIMIT and TMHINT) and music (Cadenza Challenge MUSIC) datasets. Evaluation using the Hearing Aid Speech Perception Index (HASPI), Hearing Aid Speech Quality Index (HASQI), and Hearing Aid Audio Quality Index (HAAQI) demonstrates that the Transformer architecture within NeuroAMP achieves the best performance, with SRCC scores of 0.9927 (HASQI) and 0.9905 (HASPI) on TIMIT, and 0.9738 (HAAQI) on the Cadenza Challenge MUSIC dataset. Notably, our data augmentation strategy maintains high performance on unseen datasets (e.g., VCTK, MUSDB18-HQ). Furthermore, Denoising NeuroAMP outperforms both the conventional NAL-R+WDRC approach and a two-stage baseline on the VoiceBank+DEMAND dataset, achieving a 10% improvement in both HASPI (0.90) and HASQI (0.59) scores. These results highlight the potential of NeuroAMP and Denoising NeuroAMP to deliver notable improvements in personalized hearing aid amplification. 

**Abstract (ZH)**: 助听器的使用频率正在增加。然而，由于传统方法中集成多种模块组件的复杂性，优化助听器的放大过程仍然具有挑战性。为应对这一挑战，我们提出了NeuroAMP，这是一种新型的端到端个性化放大深度神经网络。NeuroAMP 采用频谱特征和听者听力图作为输入，并探讨了四种架构：卷积神经网络（CNN）、长短时记忆网络（LSTM）、卷积循环神经网络（CRNN）和变换器（Transformer）。此外，我们还介绍了去噪的NeuroAMP扩展版本，该版本将降噪功能与放大功能集成为一个组件，以提高在实际场景中的性能。为了增强泛化能力，在涉及多样性的语音（TIMIT和TMHINT）和音乐（Cadenza Challenge MUSIC）数据集上采用了一种全面的数据增强策略进行训练。使用助听器言语感知指数（HASPI）、助听器言语质量指数（HASQI）和助听器音频质量指数（HAAQI）进行评估表明，NeuroAMP中的变换器架构表现最佳，在TIMIT数据集上，HASQI的SRCC得分为0.9927，HASPI的SRCC得分为0.9905；在Cadenza Challenge MUSIC数据集上，HAAQI的SRCC得分为0.9738。值得注意的是，我们的数据增强策略在未见过的数据集（如VCTK和MUSDB18-HQ）上保持了高性能。此外，去噪的NeuroAMP在VoiceBank+DEMAND数据集上优于传统的NAL-R+WDRC方法和两阶段基线方法，分别在HASPI和HASQI得分上提高了10%（0.90和0.59）。这些结果表明NeuroAMP和去噪的NeuroAMP有可能在个性化助听器放大方面带来显著改进。 

---
# On Vanishing Gradients, Over-Smoothing, and Over-Squashing in GNNs: Bridging Recurrent and Graph Learning 

**Title (ZH)**: 在GNN中关于消失梯度、过度光滑化和过度压缩现象的研究：连接递归学习与图学习 

**Authors**: Álvaro Arroyo, Alessio Gravina, Benjamin Gutteridge, Federico Barbero, Claudio Gallicchio, Xiaowen Dong, Michael Bronstein, Pierre Vandergheynst  

**Link**: [PDF](https://arxiv.org/pdf/2502.10818)  

**Abstract**: Graph Neural Networks (GNNs) are models that leverage the graph structure to transmit information between nodes, typically through the message-passing operation. While widely successful, this approach is well known to suffer from the over-smoothing and over-squashing phenomena, which result in representational collapse as the number of layers increases and insensitivity to the information contained at distant and poorly connected nodes, respectively. In this paper, we present a unified view of these problems through the lens of vanishing gradients, using ideas from linear control theory for our analysis. We propose an interpretation of GNNs as recurrent models and empirically demonstrate that a simple state-space formulation of a GNN effectively alleviates over-smoothing and over-squashing at no extra trainable parameter cost. Further, we show theoretically and empirically that (i) GNNs are by design prone to extreme gradient vanishing even after a few layers; (ii) Over-smoothing is directly related to the mechanism causing vanishing gradients; (iii) Over-squashing is most easily alleviated by a combination of graph rewiring and vanishing gradient mitigation. We believe our work will help bridge the gap between the recurrent and graph neural network literature and will unlock the design of new deep and performant GNNs. 

**Abstract (ZH)**: 图神经网络（GNNs）是一种利用图结构在节点间传递信息的模型，通常通过消息传递操作实现。尽管这类方法在广泛应用中取得了成功，但众所周知，这种做法容易遭受过平滑（over-smoothing）和过压缩（over-squashing）现象的困扰。这些现象分别导致了在层数增加时出现表示能力坍塌，并且对外部和连接稀疏的节点信息变得不敏感。在本文中，我们通过消失梯度的角度，从线性控制理论的视角，提出了一种综合这些问题的观点。我们将GNNs解释为循环模型，并实验证明，简单的状态空间表达式能够有效地缓解过平滑和过压缩现象，且无需额外的可训练参数成本。此外，我们从理论和实验的角度证明了以下几点：（i）GNNs即使在几层之后也容易受到极端梯度消失的影响；（ii）过平滑与造成梯度消失的机制直接相关；（iii）通过图重构与梯度消失缓解的结合，可以最有效地缓解过压缩。我们相信，这项工作将有助于弥合循环神经网络和图神经网络文献之间的差距，并为设计新的深层和高性能的图神经网络打开新的大门。 

---
# BalanceBenchmark: A Survey for Imbalanced Learning 

**Title (ZH)**: 平衡基准：不平衡学习综述 

**Authors**: Shaoxuan Xu, Menglu Cui, Chengxiang Huang, Hongfa Wang, DiHu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10816)  

**Abstract**: Multimodal learning has gained attention for its capacity to integrate information from different modalities. However, it is often hindered by the multimodal imbalance problem, where certain modality dominates while others remain underutilized. Although recent studies have proposed various methods to alleviate this problem, they lack comprehensive and fair comparisons. In this paper, we systematically categorize various mainstream multimodal imbalance algorithms into four groups based on the strategies they employ to mitigate imbalance. To facilitate a comprehensive evaluation of these methods, we introduce BalanceBenchmark, a benchmark including multiple widely used multidimensional datasets and evaluation metrics from three perspectives: performance, imbalance degree, and complexity. To ensure fair comparisons, we have developed a modular and extensible toolkit that standardizes the experimental workflow across different methods. Based on the experiments using BalanceBenchmark, we have identified several key insights into the characteristics and advantages of different method groups in terms of performance, balance degree and computational complexity. We expect such analysis could inspire more efficient approaches to address the imbalance problem in the future, as well as foundation models. The code of the toolkit is available at this https URL. 

**Abstract (ZH)**: 多模态学习因其能够整合不同模式的信息而受到了广泛关注。然而，它常常受到多模态不平衡问题的制约，某些模式占主导地位，而其他模式则被严重忽视。尽管最近的研究提出了多种缓解这一问题的方法，但它们缺乏全面和公正的比较。在本文中，我们根据缓解不平衡所采用的策略，系统地将主流的多模态不平衡算法归类为四大类。为了促进这些方法的全面评估，我们引入了BalanceBenchmark基准，该基准包括多个广泛使用的多维数据集，并从性能、不平衡程度和复杂性三个视角引入了多种评估指标。为了确保公平比较，我们开发了一个模块化且可扩展的工具箱，该工具箱标准化了不同方法之间的实验工作流。基于使用BalanceBenchmark进行的实验，我们识别了不同方法组在性能、平衡程度和计算复杂性方面的几个关键特征和优势。我们期望这样的分析能够激发未来解决不平衡问题的更高效方法，并为基础模型提供启迪。该工具箱的代码可以通过以下链接获得：[这里](https://your-link-url.com)。 

---
# HybriDNA: A Hybrid Transformer-Mamba2 Long-Range DNA Language Model 

**Title (ZH)**: HybriDNA：一种混合Transformer-Mamba2长范围DNA语言模型 

**Authors**: Mingqian Ma, Guoqing Liu, Chuan Cao, Pan Deng, Tri Dao, Albert Gu, Peiran Jin, Zhao Yang, Yingce Xia, Renqian Luo, Pipi Hu, Zun Wang, Yuan-Jyue Chen, Haiguang Liu, Tao Qin  

**Link**: [PDF](https://arxiv.org/pdf/2502.10807)  

**Abstract**: Advances in natural language processing and large language models have sparked growing interest in modeling DNA, often referred to as the "language of life". However, DNA modeling poses unique challenges. First, it requires the ability to process ultra-long DNA sequences while preserving single-nucleotide resolution, as individual nucleotides play a critical role in DNA function. Second, success in this domain requires excelling at both generative and understanding tasks: generative tasks hold potential for therapeutic and industrial applications, while understanding tasks provide crucial insights into biological mechanisms and diseases. To address these challenges, we propose HybriDNA, a decoder-only DNA language model that incorporates a hybrid Transformer-Mamba2 architecture, seamlessly integrating the strengths of attention mechanisms with selective state-space models. This hybrid design enables HybriDNA to efficiently process DNA sequences up to 131kb in length with single-nucleotide resolution. HybriDNA achieves state-of-the-art performance across 33 DNA understanding datasets curated from the BEND, GUE, and LRB benchmarks, and demonstrates exceptional capability in generating synthetic cis-regulatory elements (CREs) with desired properties. Furthermore, we show that HybriDNA adheres to expected scaling laws, with performance improving consistently as the model scales from 300M to 3B and 7B parameters. These findings underscore HybriDNA's versatility and its potential to advance DNA research and applications, paving the way for innovations in understanding and engineering the "language of life". 

**Abstract (ZH)**: 自然语言处理和大规模语言模型的进步激发了对DNA建模的兴趣，DNA被誉为“生命之语言”。然而，DNA建模面临着独特的挑战。首先，它需要能够处理超长DNA序列，同时保持单核苷酸分辨率，因为单个核苷酸在DNA功能中起着关键作用。其次，该领域的成功需要在生成任务和理解任务两个方面都表现出色：生成任务有望在治疗和工业应用中发挥潜力，而理解任务则提供了对生物机制和疾病关键见解。为了解决这些挑战，我们提出了一种名为HybriDNA的解码器型DNA语言模型，该模型结合了混合Transformer-Mamba2架构，无缝融合了注意机制的优势和选择性状态空间模型。这种混合设计使HybriDNA能够高效地处理长达131kb的DNA序列，并保持单核苷酸分辨率。HybriDNA在33个由BEND、GUE和LRB基准数据集整理的DNA理解数据集中实现了最先进的性能，并展示了生成具有所需特性的合成顺式调节元件（CREs）的出色能力。此外，我们证明HybriDNA符合预期的标度定律，即随着模型参数从300M增加到3B和7B，性能呈持续提高。这些发现突显了HybriDNA的多功能性和其在推进DNA研究和应用方面的潜力，为理解和工程“生命之语言”铺平了道路。 

---
# PDA: Generalizable Detection of AI-Generated Images via Post-hoc Distribution Alignment 

**Title (ZH)**: PDA：通过后验分布对齐实现的AI生成图像的一般化检测 

**Authors**: Li Wang, Wenyu Chen, Zheng Li, Shanqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.10803)  

**Abstract**: The rapid advancement of generative models has led to the proliferation of highly realistic AI-generated images, posing significant challenges for detection methods to generalize across diverse and evolving generative techniques. Existing approaches often fail to adapt to unknown models without costly retraining, limiting their practicability. To fill this gap, we propose Post-hoc Distribution Alignment (PDA), a novel approach for the generalizable detection for AI-generated images. The key idea is to use the known generative model to regenerate undifferentiated test images. This process aligns the distributions of the re-generated real images with the known fake images, enabling effective distinction from unknown fake images. PDA employs a two-step detection framework: 1) evaluating whether a test image aligns with the known fake distribution based on deep k-nearest neighbor (KNN) distance, and 2) re-generating test images using known generative models to create pseudo-fake images for further classification. This alignment strategy allows PDA to effectively detect fake images without relying on unseen data or requiring retraining. Extensive experiments demonstrate the superiority of PDA, achieving 96.73\% average accuracy across six state-of-the-art generative models, including GANs, diffusion models, and text-to-image models, and improving by 16.07\% over the best baseline. Through t-SNE visualizations and KNN distance analysis, we provide insights into PDA's effectiveness in separating real and fake images. Our work provides a flexible and effective solution for real-world fake image detection, advancing the generalization ability of detection systems. 

**Abstract (ZH)**: 生成模型的飞速发展导致了高度逼真的人工智能生成图像的大量涌现，这对检测方法提出了严峻挑战，使其难以适应多样且不断演化的生成技术。现有方法往往不能在不了解新型生成模型的情况下进行有效的适应性调整，限制了其实际应用的可行性。为解决这一问题，我们提出了后验分布对齐（PDA，Post-hoc Distribution Alignment）方法，这是一种用于人工智能生成图像检测的泛化方法。核心思路是使用已知的生成模型再生未分类测试图像。通过这一过程，重新生成的真实图像分布与已知的伪造图像分布对齐，从而能够有效地区分未知的伪造图像。PDA采用两步检测框架：1）基于深度k近邻（KNN）距离评估测试图像是否与已知的伪造图像分布对齐；2）使用已知生成模型再生测试图像，生成伪伪造图像以供进一步分类。这种对齐策略使PDA能够在无需使用未见过的数据或重新训练的情况下有效检测伪造图像。大量实验证明了PDA的优势，其在六个最先进的生成模型（包括GANs、扩散模型和文本到图像模型）上实现了96.73%的平均准确率，并且相比最佳基线提高了16.07%。通过t-SNE可视化和KNN距离分析，我们提供了关于PDA在区分真实和伪造图像方面的有效性见解。我们的工作提供了一种灵活且有效的解决方案，以应对真实世界中伪造图像的检测需求，进一步提升了检测系统的泛化能力。 

---
# CoCoEvo: Co-Evolution of Programs and Test Cases to Enhance Code Generation 

**Title (ZH)**: CoCoEvo: 程序与测试用例协同进化以增强代码生成 

**Authors**: Kefan Li, Hongyue Yu, Tingyu Guo, Shijie Cao, Yuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10802)  

**Abstract**: Large Language Models (LLMs) have shown remarkable performance in automated code generation. However, existing approaches often rely heavily on pre-defined test cases, which become impractical in scenarios where such cases are unavailable. While prior works explore filtering techniques between programs and test cases, they overlook the refinement of test cases. To address this limitation, we introduce CoCoEvo, a novel LLM-based co-evolution framework that simultaneously evolves programs and test cases. CoCoEvo eliminates the dependency on pre-defined test cases by generating both programs and test cases directly from natural language problem descriptions and function headers. The framework employs specialized evolutionary operators, including LLM-based crossover and mutation operators for program evolution, along with a test case generation operator for test case evolution. Additionally, we propose optimization strategies such as a crossover rate scheduler to balance exploration and convergence, and a multi-objective optimization method for test case selection. Experimental results on multiple state-of-the-art LLMs demonstrate that CoCoEvo surpasses existing methods, achieving state-of-the-art performance in automated code generation and testing. These results underscore the potential of co-evolutionary techniques in advancing the field of automated programming. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自动化代码生成方面展现了卓越的表现。然而，现有的方法往往高度依赖预定义的测试案例，而在这些测试案例不可用的情况下这种方法变得不实际。虽然先前的工作探索了程序和测试案例之间过滤技术的应用，但它们忽视了测试案例的优化。为了解决这一局限性，我们提出了一种新颖的基于LLM的协同进化框架CoCoEvo，该框架同时进化程序和测试案例。CoCoEvo通过直接从自然语言的问题描述和函数头生成程序和测试案例，消除了对预定义测试案例的依赖。该框架采用了专门的进化操作符，包括基于LLM的交叉操作符和变异操作符来进行程序进化，以及一个测试案例生成操作符来进行测试案例进化。此外，我们还提出了优化策略，如交叉率调度器以平衡探索与收敛，以及多目标优化方法用于测试案例选择。在多个最先进的LLM上的实验结果表明，CoCoEvo超越了现有方法，在自动化代码生成和测试方面达到了最先进的性能。这些结果突显了协同进化技术在推进自动化编程领域的潜力。 

---
# FaceSwapGuard: Safeguarding Facial Privacy from DeepFake Threats through Identity Obfuscation 

**Title (ZH)**: FaceSwapGuard：通过身份模糊化对抗深度造假威胁以保护面部隐私 

**Authors**: Li Wang, Zheng Li, Xuhong Zhang, Shouling Ji, Shanqing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.10801)  

**Abstract**: DeepFakes pose a significant threat to our society. One representative DeepFake application is face-swapping, which replaces the identity in a facial image with that of a victim. Although existing methods partially mitigate these risks by degrading the quality of swapped images, they often fail to disrupt the identity transformation effectively. To fill this gap, we propose FaceSwapGuard (FSG), a novel black-box defense mechanism against deepfake face-swapping threats. Specifically, FSG introduces imperceptible perturbations to a user's facial image, disrupting the features extracted by identity encoders. When shared online, these perturbed images mislead face-swapping techniques, causing them to generate facial images with identities significantly different from the original user. Extensive experiments demonstrate the effectiveness of FSG against multiple face-swapping techniques, reducing the face match rate from 90\% (without defense) to below 10\%. Both qualitative and quantitative studies further confirm its ability to confuse human perception, highlighting its practical utility. Additionally, we investigate key factors that may influence FSG and evaluate its robustness against various adaptive adversaries. 

**Abstract (ZH)**: DeepFakes 对我们的社会构成了一种显著的威胁。一种代表性的 DeepFakes 应用是换脸，它将面部图像中的身份替换为受害者的身份。尽管现有方法通过降低换脸图像的质量部分缓解了这些风险，但它们往往无法有效地中断身份变换。为填补这一缺口，我们提出了一种新颖的黑盒防御机制 FaceSwapGuard (FSG)，专门针对 DeepFakes 换脸威胁。具体来说，FSG 在用户的面部图像中引入不可察觉的扰动，扰乱身份编码器提取的特征。当这些被扰动的图像在网上共享时，换脸技术会被误导，生成的身份与原始用户相差甚远的面部图像。大量的实验表明，FSG 对多种换脸技术具有有效性，能够在没有防御措施的情况下将面部匹配率从 90% 降低到低于 10%。定性和定量研究进一步证实其能够混淆人类的感知，突显其实际用途。此外，我们还调查了可能影响 FSG 的关键因素，并对其面对各种适应性对手的鲁棒性进行了评估。 

---
# Dynamic Influence Tracker: Measuring Time-Varying Sample Influence During Training 

**Title (ZH)**: 动态影响追踪器：训练过程中样本时间变化影响的度量 

**Authors**: Jie Xu, Zihan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10793)  

**Abstract**: Existing methods for measuring training sample influence on models only provide static, overall measurements, overlooking how sample influence changes during training. We propose Dynamic Influence Tracker (DIT), which captures the time-varying sample influence across arbitrary time windows during training.
DIT offers three key insights: 1) Samples show different time-varying influence patterns, with some samples important in the early training stage while others become important later. 2) Sample influences show a weak correlation between early and late stages, demonstrating that the model undergoes distinct learning phases with shifting priorities. 3) Analyzing influence during the convergence period provides more efficient and accurate detection of corrupted samples than full-training analysis. Supported by theoretical guarantees without assuming loss convexity or model convergence, DIT significantly outperforms existing methods, achieving up to 0.99 correlation with ground truth and above 98\% accuracy in detecting corrupted samples in complex architectures. 

**Abstract (ZH)**: 现有方法仅提供模型训练样本影响的静态、总体测量，未能考虑样本影响在训练过程中的变化。我们提出了动态影响追踪器（DIT），该方法在训练过程中的任意时间段内捕捉样本影响随时间变化的情况。

DIT 提供了三个关键见解：1）样本在不同的训练阶段显示不同的时间变化影响模式，有些样本在早期训练阶段具有重要性，而其他样本则在后期变得重要。2）样本影响在早期和后期阶段的相关性较弱，表明模型经历了不同阶段的学习，每个阶段的重点不同。3）在收敛期分析影响有助于更高效和准确地检测受污染样本，其效率优于全训练期的分析。DIT 在无需假设损失函数凸性或模型收敛的情况下提供了理论上的保证，显著优于现有方法，其与真实值的相关性达到0.99，并且在复杂架构中检测受污染样本的准确性超过98%。 

---
# A Distillation-based Future-aware Graph Neural Network for Stock Trend Prediction 

**Title (ZH)**: 基于蒸馏的面向未来的图神经网络在股票趋势预测中的应用 

**Authors**: Zhipeng Liu, Peibo Duan, Mingyang Geng, Bin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10776)  

**Abstract**: Stock trend prediction involves forecasting the future price movements by analyzing historical data and various market indicators. With the advancement of machine learning, graph neural networks (GNNs) have been extensively employed in stock prediction due to their powerful capability to capture spatiotemporal dependencies of stocks. However, despite the efforts of various GNN stock predictors to enhance predictive performance, the improvements remain limited, as they focus solely on analyzing historical spatiotemporal dependencies, overlooking the correlation between historical and future patterns. In this study, we propose a novel distillation-based future-aware GNN framework (DishFT-GNN) for stock trend prediction. Specifically, DishFT-GNN trains a teacher model and a student model, iteratively. The teacher model learns to capture the correlation between distribution shifts of historical and future data, which is then utilized as intermediate supervision to guide the student model to learn future-aware spatiotemporal embeddings for accurate prediction. Through extensive experiments on two real-world datasets, we verify the state-of-the-art performance of DishFT-GNN. 

**Abstract (ZH)**: 股票趋势预测涉及通过分析历史数据和各种市场指标来预测未来的价格变动。随着机器学习的发展，图神经网络（GNNs）因其强大的时空依赖性捕捉能力，在股票预测中得到了广泛的应用。然而，尽管各种GNN股票预测器已不断努力提高预测性能，但改进仍然有限，因为它们仅专注于分析历史时空依赖性，而忽视了历史数据和未来数据模式之间的关联性。为解决这一问题，本研究提出了一种新颖的基于/distillation的未来感知GNN框架（DishFT-GNN），用于股票趋势预测。具体而言，DishFT-GNN通过迭代训练教师模型和学生模型。教师模型学习捕捉历史数据和未来数据分布变化之间的关联性，然后将这些中间监督信息用于引导学生模型学习未来的时空嵌入，以实现准确的预测。通过在两个真实世界数据集上进行广泛的实验，我们验证了DishFT-GNN的先进性能。 

---
# Evaluating improvements on using Large Language Models (LLMs) for property extraction in the Open Research Knowledge Graph (ORKG) 

**Title (ZH)**: 评估在开放研究知识图谱（ORKG）中使用大型语言模型（LLMs）进行属性提取的改进效果 

**Authors**: Sandra Schaftner  

**Link**: [PDF](https://arxiv.org/pdf/2502.10768)  

**Abstract**: Current research highlights the great potential of Large Language Models (LLMs) for constructing Scholarly Knowledge Graphs (SKGs). One particularly complex step in this process is relation extraction, aimed at identifying suitable properties to describe the content of research. This study builds directly on previous research of three Open Research Knowledge Graph (ORKG) team members who assessed the readiness of LLMs such as GPT-3.5, Llama 2, and Mistral for property extraction in scientific literature. Given the moderate performance observed, the previous work concluded that fine-tuning is needed to improve these models' alignment with scientific tasks and their emulation of human expertise. Expanding on this prior experiment, this study evaluates the impact of advanced prompt engineering techniques and demonstrates that these techniques can highly significantly enhance the results. Additionally, this study extends the property extraction process to include property matching to existing ORKG properties, which are retrieved via the API. The evaluation reveals that results generated through advanced prompt engineering achieve a higher proportion of matches with ORKG properties, further emphasizing the enhanced alignment achieved. Moreover, this lays the groundwork for addressing challenges such as the inconsistency of ORKG properties, an issue highlighted in prior studies. By assigning unique URIs and using standardized terminology, this work increases the consistency of the properties, fulfilling a crucial aspect of Linked Data and FAIR principles - core commitments of ORKG. This, in turn, significantly enhances the applicability of ORKG content for subsequent tasks such as comparisons of research publications. Finally, the study concludes with recommendations for future improvements in the overall property extraction process. 

**Abstract (ZH)**: 当前的研究突显了大型语言模型（LLMs）在构建学术知识图谱（SKGs）方面的巨大潜力。这一过程中特别复杂的一个步骤是关系提取，旨在识别合适的属性以描述研究内容。本研究直接基于三名Open Research Knowledge Graph（ORKG）团队成员的先前研究，这些成员评估了如GPT-3.5、Llama 2和Mistral等LLM在科学文献中进行属性提取的能力。鉴于这些模型观察到的中等性能，先前的研究得出结论，需要对这些模型进行微调以提高它们与科学任务的契合度以及模仿人类专业知识的能力。在此前实验的基础上，本研究评估了高级提示工程技术的影响，并表明这些技术能够显著提高提取结果。此外，本研究将属性提取过程扩展到包括与现有ORKG属性的匹配，这些属性通过API检索。评价结果显示，通过高级提示工程技术生成的结果与ORKG属性的匹配比例更高，进一步突显了这种增强的对齐效果。此外，这为进一步解决ORKG属性不一致性的问题奠定了基础，这是先前研究中指出的一个问题。通过分配唯一的URI并使用标准化的术语，本研究增加了属性的一致性，满足了Linked Data和FAIR原则中的关键方面——ORKG的核心承诺。这一过程反过来显著提高了ORKG内容在后续任务中的适用性，例如研究出版物的比较。最后，本研究提出了改进整体属性提取过程的建议。 

---
# Bone Soups: A Seek-and-Soup Model Merging Approach for Controllable Multi-Objective Generation 

**Title (ZH)**: 骨汤模型：一种用于可控多目标生成的寻觅与汇聚模型合并方法 

**Authors**: Guofu Xie, Xiao Zhang, Ting Yao, Yunsheng Shi  

**Link**: [PDF](https://arxiv.org/pdf/2502.10762)  

**Abstract**: User information needs are often highly diverse and varied. A key challenge in current research is how to achieve controllable multi-objective generation while enabling rapid adaptation to accommodate diverse user demands during test time. Existing solutions, such as Rewarded Soup, focus on merging language models individually tuned on single objectives. While easy to implement and widely used, these approaches face limitations in achieving optimal performance due to their disregard for the impacts of competing objectives on model tuning. To address this issue, we propose Bone Soup, a novel model merging approach that first seeks a series of backbone models by considering the impacts of multiple objectives and then makes the soup (i.e., merge the backbone models). Specifically, Bone Soup begins by training multiple backbone models for different objectives using multi-objective reinforcement learning. Each backbone model is guided by a combination of backbone reward signals. To ensure that these models are optimal for the Pareto front, the backbone rewards are crafted by combining standard reward functions into basis vectors, which can then be modified through a rule-based construction method. Bone Soup leverages a symmetric circulant matrix mapping to generate the merging coefficients, which are used to merge the backbone models according to user preferences. Extensive experimental results demonstrate that Bone Soup exhibits strong controllability and Pareto optimality in controllable multi-objective generation, providing a more effective and efficient approach to addressing diverse user needs at test time. 

**Abstract (ZH)**: 用户的信息需求往往非常多样且复杂。当前研究中的一个重要挑战是如何在满足多样化用户需求的同时实现可控的多目标生成，并且能够在测试过程中迅速适应不同的需求。现有的解决方案，如Rewarded Soup，主要关注将单目标优化的语言模型进行合并。尽管这些方法易于实现且广泛使用，但它们在实现最优性能方面存在局限性，因为这些方法没有考虑到竞争目标对模型优化的影响。为了克服这一问题，我们提出了Bone Soup这一新型模型合并方法，该方法首先通过考虑多个目标的影响来选择一系列基础模型，然后通过这些基础模型生成“汤”（即将基础模型进行合并）。具体来说，Bone Soup首先使用多目标强化学习训练出多个针对不同目标的基础模型。每个基础模型由一组基础奖励信号引导。为了确保这些模型在帕累托前沿上的最优性，基础奖励信号通过结合标准奖励函数生成基向量，并通过基于规则的方法进行修改。Bone Soup利用对称循环矩阵映射生成合并系数，这些系数根据用户偏好合并基础模型。广泛实验证明，Bone Soup在可控多目标生成中表现出强大的可控性和帕累托最优性，为在测试过程中更好地满足多样化用户需求提供了更有效和高效的方法。 

---
# Human-Centric Community Detection in Hybrid Metaverse Networks with Integrated AI Entities 

**Title (ZH)**: 基于人类导向的混合元宇宙网络中集成AI实体的社区检测 

**Authors**: Shih-Hsuan Chiu, Ya-Wen Teng, De-Nian Yang, Ming-Syan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.10750)  

**Abstract**: Community detection is a cornerstone problem in social network analysis (SNA), aimed at identifying cohesive communities with minimal external links. However, the rise of generative AI and Metaverse introduce complexities by creating hybrid human-AI social networks (denoted by HASNs), where traditional methods fall short, especially in human-centric settings. This paper introduces a novel community detection problem in HASNs (denoted by MetaCD), which seeks to enhance human connectivity within communities while reducing the presence of AI nodes. Effective processing of MetaCD poses challenges due to the delicate trade-off between excluding certain AI nodes and maintaining community structure. To address this, we propose CUSA, an innovative framework incorporating AI-aware clustering techniques that navigate this trade-off by selectively retaining AI nodes that contribute to community integrity. Furthermore, given the scarcity of real-world HASNs, we devise four strategies for synthesizing these networks under various hypothetical scenarios. Empirical evaluations on real social networks, reconfigured as HASNs, demonstrate the effectiveness and practicality of our approach compared to traditional non-deep learning and graph neural network (GNN)-based methods. 

**Abstract (ZH)**: 社会网络分析（SNA）中的社区检测是一个核心问题，旨在识别具有最小外部联系的紧密社区。然而，生成式AI和元宇宙的兴起引入了复杂性，通过创建混合人类-AI社会网络（简称HASNs）来进一步复杂化问题，使传统方法在以人类为中心的情境中变得不足。本文引入了一个在HASNs中解决的新型社区检测问题（简称MetaCD），旨在增强社区内部的人类连接性，同时减少AI节点的存在。有效处理MetaCD面临挑战，因为需要在排除某些AI节点和保持社区结构之间寻求微妙的平衡。为了解决这一问题，我们提出了一种名为CUSA的创新框架，该框架结合了AI感知的聚类技术，并通过选择性地保留对社区完整性有贡献的AI节点来解决这种平衡问题。此外，由于现实世界中的HASNs稀缺，我们设计了四种策略，以在不同的假设场景下合成这些网络。通过对实际社会网络的重新配置作为HASNs进行实证评估，我们的方法相对于传统非深度学习和图神经网络（GNN）方法的有效性和实用性得到了验证。 

---
# LoRE-Merging: Exploring Low-Rank Estimation For Large Language Model Merging 

**Title (ZH)**: LoRE-合并：探索大语言模型合并的低秩估计方法 

**Authors**: Zehua Liu, Han Wu, Yuxuan Yao, Ruifeng She, Xiongwei Han, Tao Zhong, Mingxuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10749)  

**Abstract**: While most current approaches rely on further training techniques, such as fine-tuning or reinforcement learning, to enhance model capacities, model merging stands out for its ability of improving models without requiring any additional training. In this paper, we propose a unified framework for model merging based on low-rank estimation of task vectors without the need for access to the base model, named \textsc{LoRE-Merging}. Our approach is motivated by the observation that task vectors from fine-tuned models frequently exhibit a limited number of dominant singular values, making low-rank estimations less prone to interference. We implement the method by formulating the merging problem as an optimization problem. Extensive empirical experiments demonstrate the effectiveness of our framework in mitigating interference and preserving task-specific information, thereby advancing the state-of-the-art performance in model merging techniques. 

**Abstract (ZH)**: 尽管大多数现有的方法依赖于进一步的训练技术，如微调或强化学习来增强模型的能力，但模型合并凭借其在无需额外训练的情况下提升模型能力的独特优势而脱颖而出。在本文中，我们提出了一种基于低秩估计任务向量的统一模型合并框架，名为LoRE-合并。我们的方法受到了观察结果的启发，即微调模型的任务向量通常仅表现出有限数量的主导奇异值，这使得低秩估计更容易避免干扰。我们通过将合并问题形式化为一个优化问题来实现该方法。广泛的实证实验表明，我们的框架在缓解干扰和保留任务特定信息方面具有有效性，从而在模型合并技术中推动了最新的性能水平。 

---
# Rule-Bottleneck Reinforcement Learning: Joint Explanation and Decision Optimization for Resource Allocation with Language Agents 

**Title (ZH)**: 规则瓶颈强化学习：语言代理参与的资源配置的联合解释与决策优化 

**Authors**: Mauricio Tec, Guojun Xiong, Haichuan Wang, Francesca Dominici, Milind Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2502.10732)  

**Abstract**: Deep Reinforcement Learning (RL) is remarkably effective in addressing sequential resource allocation problems in domains such as healthcare, public policy, and resource management. However, deep RL policies often lack transparency and adaptability, challenging their deployment alongside human decision-makers. In contrast, Language Agents, powered by large language models (LLMs), provide human-understandable reasoning but may struggle with effective decision making. To bridge this gap, we propose Rule-Bottleneck Reinforcement Learning (RBRL), a novel framework that jointly optimizes decision and explanations. At each step, RBRL generates candidate rules with an LLM, selects among them using an attention-based RL policy, and determines the environment action with an explanation via chain-of-thought reasoning. The RL rule selection is optimized using the environment rewards and an explainability metric judged by the LLM. Evaluations in real-world scenarios highlight RBRL's competitive performance with deep RL and efficiency gains over LLM fine-tuning. A survey further confirms the enhanced quality of its explanations. 

**Abstract (ZH)**: 深度强化学习（RL）在医疗保健、公共政策和资源管理等领域解决顺序资源分配问题方面表现出显著的效果。然而，深度RL策略往往缺乏透明性和适应性，这使其难以与人类决策者并行部署。相比之下，由大规模语言模型（LLMs）驱动的语言代理提供了易于人类理解的推理方式，但在有效决策方面可能存在挑战。为解决这一问题，我们提出了一种新颖框架——规则瓶颈强化学习（RBRL），旨在同时优化决策和解释。在每一步骤中，RBRL 使用LLM生成候选规则，使用基于注意力的RL策略从中选择，并通过链式推理进行解释，以确定环境动作。通过环境奖励和由LLM评估的解释性指标优化RL规则的选择。实地情景下的评估展示了RBRL在与深度RL相比时具有竞争力的表现，并在LLM微调方面提高了效率。进一步的调查还证实了其解释质量的提升。 

---
# PropNet: a White-Box and Human-Like Network for Sentence Representation 

**Title (ZH)**: PropNet：一种白盒且类人类的句子表示网络 

**Authors**: Fei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10725)  

**Abstract**: Transformer-based embedding methods have dominated the field of sentence representation in recent years. Although they have achieved remarkable performance on NLP missions, such as semantic textual similarity (STS) tasks, their black-box nature and large-data-driven training style have raised concerns, including issues related to bias, trust, and safety. Many efforts have been made to improve the interpretability of embedding models, but these problems have not been fundamentally resolved. To achieve inherent interpretability, we propose a purely white-box and human-like sentence representation network, PropNet. Inspired by findings from cognitive science, PropNet constructs a hierarchical network based on the propositions contained in a sentence. While experiments indicate that PropNet has a significant gap compared to state-of-the-art (SOTA) embedding models in STS tasks, case studies reveal substantial room for improvement. Additionally, PropNet enables us to analyze and understand the human cognitive processes underlying STS benchmarks. 

**Abstract (ZH)**: 基于Transformer的嵌入方法在近年来的句子表示领域占据主导地位。尽管这些方法在自然语言处理任务（如语义文本相似性任务）中取得了显著的性能，但它们的黑盒性质和大数据驱动的训练方式引发了关于偏差、信任和安全等方面的问题。许多人努力提高嵌入模型的可解释性，但这些问题尚未根本解决。为了实现固有的可解释性，我们提出了一种纯粹的白盒且类人句子表示网络——PropNet。受到认知科学发现的启发，PropNet基于句子中的命题构建了一个分层网络。尽管实验表明，PropNet在语义文本相似性任务中与最先进的（SOTA）嵌入模型相比存在显著差距，但案例研究揭示了改进的巨大空间。此外，PropNet使我们能够分析和理解支撑语义文本相似性基准的人类认知过程。 

---
# A Mathematics Framework of Artificial Shifted Population Risk and Its Further Understanding Related to Consistency Regularization 

**Title (ZH)**: 一种人工偏移人口风险的数学框架及其与一致性正则化的进一步理解 

**Authors**: Xiliang Yang, Shenyang Deng, Shicong Liu, Yuanchi Suo, Wing.W.Y NG, Jianjun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10723)  

**Abstract**: Data augmentation is an important technique in training deep neural networks as it enhances their ability to generalize and remain robust. While data augmentation is commonly used to expand the sample size and act as a consistency regularization term, there is a lack of research on the relationship between them. To address this gap, this paper introduces a more comprehensive mathematical framework for data augmentation. Through this framework, we establish that the expected risk of the shifted population is the sum of the original population risk and a gap term, which can be interpreted as a consistency regularization term. The paper also provides a theoretical understanding of this gap, highlighting its negative effects on the early stages of training. We also propose a method to mitigate these effects. To validate our approach, we conducted experiments using same data augmentation techniques and computing resources under several scenarios, including standard training, out-of-distribution, and imbalanced classification. The results demonstrate that our methods surpass compared methods under all scenarios in terms of generalization ability and convergence stability. We provide our code implementation at the following link: this https URL. 

**Abstract (ZH)**: 数据增强是一种重要的深度神经网络训练技术，它能够提升模型的泛化能力和鲁棒性。尽管数据增强通常用于扩大样本量并作为一致性正则化项，但现有研究较少探讨两者之间的关系。为解决这一问题，本文引入了一个更为全面的数学框架来研究数据增强。通过这个框架，我们证明了变换后群体的期望风险是原群体风险与一个差异项之和，该差异项可以被解释为一致性正则化项。此外，我们还从理论上解释了该差异项的作用，并指出其对训练早期阶段的负面影响。我们还提出了一种方法来减轻这些负面影响。为了验证我们的方法，我们在多种情况下进行了实验，包括标准训练、非分布外数据和样本不平衡分类，使用了相同的增强技术和计算资源。实验结果显示，我们的方法在所有情况下都能在泛化能力和收敛稳定性方面超越对照方法。我们将在以下链接提供代码实现：[this https URL]。 

---
# Hyperdimensional Intelligent Sensing for Efficient Real-Time Audio Processing on Extreme Edge 

**Title (ZH)**: 超维智能传感在极端边缘环境下的高效实时音频处理 

**Authors**: Sanggeon Yun, Ryozo Masukawa, Hanning Chen, SungHeon Jeong, Wenjun Huang, Arghavan Rezvani, Minhyoung Na, Yoshiki Yamaguchi, Mohsen Imani  

**Link**: [PDF](https://arxiv.org/pdf/2502.10718)  

**Abstract**: The escalating challenges of managing vast sensor-generated data, particularly in audio applications, necessitate innovative solutions. Current systems face significant computational and storage demands, especially in real-time applications like gunshot detection systems (GSDS), and the proliferation of edge sensors exacerbates these issues. This paper proposes a groundbreaking approach with a near-sensor model tailored for intelligent audio-sensing frameworks. Utilizing a Fast Fourier Transform (FFT) module, convolutional neural network (CNN) layers, and HyperDimensional Computing (HDC), our model excels in low-energy, rapid inference, and online learning. It is highly adaptable for efficient ASIC design implementation, offering superior energy efficiency compared to conventional embedded CPUs or GPUs, and is compatible with the trend of shrinking microphone sensor sizes. Comprehensive evaluations at both software and hardware levels underscore the model's efficacy. Software assessments through detailed ROC curve analysis revealed a delicate balance between energy conservation and quality loss, achieving up to 82.1% energy savings with only 1.39% quality loss. Hardware evaluations highlight the model's commendable energy efficiency when implemented via ASIC design, especially with the Google Edge TPU, showcasing its superiority over prevalent embedded CPUs and GPUs. 

**Abstract (ZH)**: 管理大量传感器生成数据的日益严峻挑战，尤其是在音频应用中，需要创新的解决方案。当前系统在实际应用中，如枪声检测系统（GSDS），面临着显著的计算和存储需求，而边缘传感器的普及进一步加剧了这些问题。本文提出了一种创新的方法，即为智能音频感知框架定制的近传感器模型。该模型利用快速傅里叶变换（FFT）模块、卷积神经网络（CNN）层和超维度计算（HDC），在低能耗、快速推理和在线学习方面表现出色。该模型对于高效的ASIC设计实现高度适应，并且与传统嵌入式CPU或GPU相比，具有更高的能效比，同时与缩小的麦克风传感器趋势相兼容。全面的软件和硬件评估证实了该模型的有效性。通过详细的ROC曲线分析，软件评估显示在能耗和质量损失之间取得了微妙的平衡，实现了高达82.1%的能耗节省，同时仅损失1.39%的质量。硬件评估强调了该模型在通过ASIC设计实现时的优异能效，特别是在使用Google Edge TPU时，其表现优于普遍使用的嵌入式CPU和GPU。 

---
# FuncGenFoil: Airfoil Generation and Editing Model in Function Space 

**Title (ZH)**: FuncGenFoil：函数空间的翼型生成与编辑模型 

**Authors**: Jinouwen Zhang, Junjie Ren, Aobo Yang, Yan Lu, Lu Chen, Hairun Xie, Jing Wang, Miao Zhang, Wanli Ouyang, Shixiang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10712)  

**Abstract**: Aircraft manufacturing is the jewel in the crown of industry, among which generating high-fidelity airfoil geometries with controllable and editable representations remains a fundamental challenge. While existing deep-learning-based methods rely on predefined parametric function families, e.g., Bézier curves and discrete point-based representations, they suffer from inherent trade-offs between expressiveness and resolution flexibility. To tackle this challenge, we introduce FuncGenFoil, a novel function-space generative model that directly learns functional airfoil geometries. Our method inherits both the advantages of arbitrary resolution sampling and the smoothness of parametric functions, as well as the strong expressiveness of discrete point-based functions. Empirical evaluations on the AFBench dataset demonstrate that FuncGenFoil improves upon state-of-the-art methods in airfoil generation by achieving a relative -74.4 label error reduction and +23.2 diversity increase on the AF-200K dataset. Our results highlight the advantages of function-space modeling for aerodynamic shape optimization, offering a powerful and flexible framework for high-fidelity airfoil design. Our code will be released. 

**Abstract (ZH)**: 航空制造是工业皇冠上的明珠，其中生成具有可控制和可编辑表示的高保真翼型几何形状依然是一个基本挑战。虽然现有的基于深度学习的方法依赖于预定义的参数函数族，例如贝塞尔曲线和离散点表示，但它们在表达能力和分辨率灵活性之间存在着固有的权衡。为应对这一挑战，我们引入了FuncGenFoil，一种新颖的功能空间生成模型，可以直接学习功能翼型几何形状。我们的方法继承了任意分辨率采样和参数函数的平滑性的好处，同时也保留了离散点表示函数的强大表达能力。在AFBench数据集上的经验评估表明，FuncGenFoil在翼型生成方面比最先进的方法实现了相对-74.4％的标签错误率降低和+23.2％的多样性增加。我们的结果强调了功能空间建模在气动形状优化方面的优势，提供了高保真翼型设计的强大而灵活的框架。我们的代码将会开源。 

---
# An Empirical Analysis of Uncertainty in Large Language Model Evaluations 

**Title (ZH)**: 大型语言模型评估中的不确定性实证分析 

**Authors**: Qiujie Xie, Qingqiu Li, Zhuohao Yu, Yuejie Zhang, Yue Zhang, Linyi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10709)  

**Abstract**: As LLM-as-a-Judge emerges as a new paradigm for assessing large language models (LLMs), concerns have been raised regarding the alignment, bias, and stability of LLM evaluators. While substantial work has focused on alignment and bias, little research has concentrated on the stability of LLM evaluators. In this paper, we conduct extensive experiments involving 9 widely used LLM evaluators across 2 different evaluation settings to investigate the uncertainty in model-based LLM evaluations. We pinpoint that LLM evaluators exhibit varying uncertainty based on model families and sizes. With careful comparative analyses, we find that employing special prompting strategies, whether during inference or post-training, can alleviate evaluation uncertainty to some extent. By utilizing uncertainty to enhance LLM's reliability and detection capability in Out-Of-Distribution (OOD) data, we further fine-tune an uncertainty-aware LLM evaluator named ConfiLM using a human-annotated fine-tuning set and assess ConfiLM's OOD evaluation ability on a manually designed test set sourced from the 2024 Olympics. Experimental results demonstrate that incorporating uncertainty as additional information during the fine-tuning phase can largely improve the model's evaluation performance in OOD scenarios. The code and data are released at: this https URL. 

**Abstract (ZH)**: 随着大型语言模型评估者（LLM-as-a-Judge）作为评估大型语言模型（LLMs）的新范式出现，人们对其对齐性、偏差和稳定性提出了担忧。尽管在对齐性和偏差方面已开展了大量研究，但很少有研究关注LMM评估者的稳定性。本文中，我们进行了广泛的实验，涉及2种不同的评估设置下9个广泛使用的LMM评估器，以探讨基于模型的LMM评估中的不确定性。我们发现，LMM评估器的不确定性在不同模型家族和规模下有所不同。通过细致的比较分析，我们发现，在推理或后训练阶段采用特殊的提示策略可以部分缓解评估不确定性。通过利用不确定性来提高大型语言模型在离散分布（OOD）数据上的可靠性和检测能力，我们进一步对一个名为ConfiLM的不确定性感知LMM评估器进行微调，并在2024年奥运会人工设计的测试集上评估了ConfiLM的OOD评估能力。实验结果表明，在微调阶段引入不确定性作为额外信息，可以显著提高模型在OOD场景下的评估性能。相关代码和数据已发布于：this https URL。 

---
# Reading Your Heart: Learning ECG Words and Sentences via Pre-training ECG Language Model 

**Title (ZH)**: 阅读你的心跳：通过预训练心电图语言模型学习心电图“单词”和“句子” 

**Authors**: Jiarui Jin, Haoyu Wang, Hongyan Li, Jun Li, Jiahui Pan, Shenda Hong  

**Link**: [PDF](https://arxiv.org/pdf/2502.10707)  

**Abstract**: Electrocardiogram (ECG) is essential for the clinical diagnosis of arrhythmias and other heart diseases, but deep learning methods based on ECG often face limitations due to the need for high-quality annotations. Although previous ECG self-supervised learning (eSSL) methods have made significant progress in representation learning from unannotated ECG data, they typically treat ECG signals as ordinary time-series data, segmenting the signals using fixed-size and fixed-step time windows, which often ignore the form and rhythm characteristics and latent semantic relationships in ECG signals. In this work, we introduce a novel perspective on ECG signals, treating heartbeats as words and rhythms as sentences. Based on this perspective, we first designed the QRS-Tokenizer, which generates semantically meaningful ECG sentences from the raw ECG signals. Building on these, we then propose HeartLang, a novel self-supervised learning framework for ECG language processing, learning general representations at form and rhythm levels. Additionally, we construct the largest heartbeat-based ECG vocabulary to date, which will further advance the development of ECG language processing. We evaluated HeartLang across six public ECG datasets, where it demonstrated robust competitiveness against other eSSL methods. Our data and code are publicly available at this https URL. 

**Abstract (ZH)**: 心电图（ECG）在临床诊断心律失常和其他心脏疾病中至关重要，但基于ECG的深度学习方法常因需要高质量的标注而受到限制。尽管先前的心电图半监督学习（eSSL）方法在无标注ECG数据的表示学习方面已经取得了显著进展，但它们通常将ECG信号视为普通的时序数据，使用固定大小和固定步长的时间窗口进行分段，这往往会忽略ECG信号中的形态、节奏特征及其潜在的语义关系。在本研究中，我们提出了对ECG信号的新型视角，将心跳视为单词，将节奏视为句子。基于此视角，我们首先设计了QRS-Tokenizer，该工具能够从原始ECG信号中生成具有语义含义的心电图句子。在此基础上，我们进一步提出了HeartLang，这是一种全新的半监督学习框架，用于心电图语言处理，其学习形态和节奏层面的通用表示。此外，我们构建了迄今为止最大的基于心跳的ECG词汇表，这将进一步推动心电图语言处理的发展。我们在六个公开的ECG数据集上评估了HeartLang，结果显示其在其他eSSL方法中表现出稳健的竞争力。相关数据和代码已公开，可访问此网址：[在此网址]。 

---
# Raising the Bar in Graph OOD Generalization: Invariant Learning Beyond Explicit Environment Modeling 

**Title (ZH)**: 在图数据外域泛化中树立新标杆：超越显式环境建模的不变学习 

**Authors**: Xu Shen, Yixin Liu, Yili Wang, Rui Miao, Yiwei Dai, Shirui Pan, Xin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10706)  

**Abstract**: Out-of-distribution (OOD) generalization has emerged as a critical challenge in graph learning, as real-world graph data often exhibit diverse and shifting environments that traditional models fail to generalize across. A promising solution to address this issue is graph invariant learning (GIL), which aims to learn invariant representations by disentangling label-correlated invariant subgraphs from environment-specific subgraphs. However, existing GIL methods face two major challenges: (1) the difficulty of capturing and modeling diverse environments in graph data, and (2) the semantic cliff, where invariant subgraphs from different classes are difficult to distinguish, leading to poor class separability and increased misclassifications. To tackle these challenges, we propose a novel method termed Multi-Prototype Hyperspherical Invariant Learning (MPHIL), which introduces two key innovations: (1) hyperspherical invariant representation extraction, enabling robust and highly discriminative hyperspherical invariant feature extraction, and (2) multi-prototype hyperspherical classification, which employs class prototypes as intermediate variables to eliminate the need for explicit environment modeling in GIL and mitigate the semantic cliff issue. Derived from the theoretical framework of GIL, we introduce two novel objective functions: the invariant prototype matching loss to ensure samples are matched to the correct class prototypes, and the prototype separation loss to increase the distinction between prototypes of different classes in the hyperspherical space. Extensive experiments on 11 OOD generalization benchmark datasets demonstrate that MPHIL achieves state-of-the-art performance, significantly outperforming existing methods across graph data from various domains and with different distribution shifts. 

**Abstract (ZH)**: 超出分布（Out-of-distribution, OOD）泛化已成为图学习中一个关键的挑战，因为在现实世界中，图数据经常表现出多样且易变的环境，而传统的模型无法在这些环境中泛化。一种有前景的解决方案是图不变学习（Graph Invariant Learning, GIL），其目标是通过分离标签相关不变子图和环境特定子图来学习不变表示。然而，现有的GIL方法面临两个主要挑战：（1）在图数据中捕捉和建模多样环境的困难，以及（2）语义悬崖，即来自不同类别的不变子图难以区分，导致类别区分性差和增加误分类率。为应对这些挑战，我们提出了一种名为多原型超球面不变学习（Multi-Prototype Hyperspherical Invariant Learning, MPHIL）的新方法，该方法引入了两个关键创新：（1）超球面不变表示提取，允许进行鲁棒且高度区分的超球面不变特征提取；（2）多原型超球面分类，采用类别原型作为中间变量以消除GIL中显式环境建模的需求，并减轻语义悬崖问题。基于GIL的理论框架，我们引入了两个新的目标函数：不变原型匹配损失以确保样本与正确的类别原型匹配，原型分离损失以增加不同类别原型在超球面上的区分度。在11个OOD泛化基准数据集中进行的广泛实验表明，MPHIL达到了最先进的性能，在不同领域和不同分布转移的图数据上显著优于现有的方法。 

---
# Occlusion-aware Non-Rigid Point Cloud Registration via Unsupervised Neural Deformation Correntropy 

**Title (ZH)**: 基于 occlusion 意识的无监督神经变形核函数非刚性点云配准 

**Authors**: Mingyang Zhao, Gaofeng Meng, Dong-Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10704)  

**Abstract**: Non-rigid alignment of point clouds is crucial for scene understanding, reconstruction, and various computer vision and robotics tasks. Recent advancements in implicit deformation networks for non-rigid registration have significantly reduced the reliance on large amounts of annotated training data. However, existing state-of-the-art methods still face challenges in handling occlusion scenarios. To address this issue, this paper introduces an innovative unsupervised method called Occlusion-Aware Registration (OAR) for non-rigidly aligning point clouds. The key innovation of our method lies in the utilization of the adaptive correntropy function as a localized similarity measure, enabling us to treat individual points distinctly. In contrast to previous approaches that solely minimize overall deviations between two shapes, we combine unsupervised implicit neural representations with the maximum correntropy criterion to optimize the deformation of unoccluded regions. This effectively avoids collapsed, tearing, and other physically implausible results. Moreover, we present a theoretical analysis and establish the relationship between the maximum correntropy criterion and the commonly used Chamfer distance, highlighting that the correntropy-induced metric can be served as a more universal measure for point cloud analysis. Additionally, we introduce locally linear reconstruction to ensure that regions lacking correspondences between shapes still undergo physically natural deformations. Our method achieves superior or competitive performance compared to existing approaches, particularly when dealing with occluded geometries. We also demonstrate the versatility of our method in challenging tasks such as large deformations, shape interpolation, and shape completion under occlusion disturbances. 

**Abstract (ZH)**: 非刚性点云对齐对于场景理解、重建以及各种计算机视觉和机器人任务至关重要。最近在隐式变形网络中的进展显著减少了对大量标注训练数据的依赖。然而，现有的最先进的方法仍然难以处理遮挡场景。为了解决这一问题，本文提出了一种名为Occlusion-Aware Registration（OAR）的创新无监督方法，用于非刚性对齐点云。我们方法的核心创新之处在于利用自适应校正熵函数作为局部相似性测度，使我们能够单独处理每个点。相比之下，先前的方法仅通过最小化两个形状之间的总体偏差来进行变形，我们则结合无监督的隐式神经表示和最大校正熵准则来优化未遮挡区域的变形。这有效地避免了崩溃、断裂等物理上不合理的结果。此外，我们进行了理论分析，并建立了最大校正熵准则与常用的Chamfer距离之间的关系，强调校正熵诱导的度量可以作为点云分析中更为通用的测度。同时，我们引入了局部线性重建，以确保缺乏形状对应关系的区域仍然进行自然的物理变形。我们的方法在处理遮挡几何时相比现有方法取得了更优越或更具竞争力的表现。此外，我们还展示了我们的方法在大型变形、形状插值以及遮挡干扰下形状补全等具有挑战性的任务中的灵活性和有效性。 

---
# Exploring Synaptic Resonance in Large Language Models: A Novel Approach to Contextual Memory Integration 

**Title (ZH)**: 探究大型语言模型中的突触共振：一种新颖的上下文记忆整合方法 

**Authors**: George Applegarth, Christian Weatherstone, Maximilian Hollingsworth, Henry Middlebrook, Marcus Irvin  

**Link**: [PDF](https://arxiv.org/pdf/2502.10699)  

**Abstract**: Contextual memory integration remains a high challenge in the development of language models, particularly in tasks that require maintaining coherence over extended sequences. Traditional approaches, such as self-attention mechanisms and memory-augmented architectures, often prioritize short-term dependencies, leading to fragmentation and inconsistency in long-range contextual understanding. Inspired by principles of synaptic plasticity observed in biological neural systems, a novel mechanism, Synaptic Resonance, is introduced to dynamically reinforce relevant memory pathways during training and inference. Unlike static memory representations, this mechanism continuously adjusts synaptic weight matrices based on contextual relevance, allowing for improved information retention without excessive computational overhead. Evaluations conducted on an open-source language model demonstrate reductions in perplexity, enhancements in contextual coherence, and increased robustness against input noise, highlighting the effectiveness of reinforcement-driven memory modulation. Comparative analysis against baseline models further reveals that the proposed approach achieves higher memory retention efficiency while maintaining computational feasibility. The architectural modifications integrate seamlessly into existing transformer-based frameworks, ensuring stable convergence and efficient inference without sacrificing scalability. Applications benefiting from improved long-term contextual consistency, such as dialogue systems and document summarization, stand to gain from this approach. Empirical findings suggest that dynamically reinforced memory pathways offer a promising alternative to conventional memory mechanisms, addressing longstanding limitations in extended sequence modeling. 

**Abstract (ZH)**: 上下文记忆整合在语言模型的发展中仍然是一项高挑战，特别是在需要保持长时间序列连贯性任务中。传统方法，如自注意力机制和记忆增强架构，往往侧重于短期依赖性，导致长时间范围内上下文理解的断裂和不一致性。受生物学神经系统中突触可塑性原理的启发，引入了一种新的机制——突触共振，该机制在训练和推理过程中动态强化相关记忆路径。与静态记忆表示不同，该机制基于上下文相关性连续调整突触权重矩阵，从而在不增加过重计算负担的情况下提高信息保留能力。对该开源语言模型进行的评估显示，突透性（perplexity）降低、上下文连贯性增强和对输入噪声的鲁棒性提高，突显了驱动记忆调优的有效性。与其他基线模型的对比分析进一步表明，所提出的方法在保持计算可行性的同时，实现了更高的记忆保留效率。架构修改无缝集成到现有的变压器框架中，确保了稳定收敛和高效推理，而不牺牲可扩展性。受益于改进的长时段上下文一致性的应用，如对话系统和文档摘要，可以从中获益。实验结果表明，动态强化的记忆路径为常规记忆机制提供了有前景的替代方案，解决了扩展序列建模中的长期局限性。 

---
# Superpose Singular Features for Model Merging 

**Title (ZH)**: 将奇异特征叠加以实现模型合并 

**Authors**: Haiquan Qiu, You Wu, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10698)  

**Abstract**: Model merging is a critical technique for combining the capabilities of multiple fine-tuned models without requiring additional training. While existing methods treat parameters as vectors, they overlook the intrinsic structure of linear transformation matrices - the core components that comprise the majority of model parameters. These matrices are fundamental to neural networks, mapping input representations to output features through linear combinations. Motivated by the linear representation hypothesis, we introduce task matrix and propose to Superpose Features from Task Matrix (SFTM), a novel approach that superposes features from individual task models into a merged model. SFTM employs singular value decomposition to identify feature bases of linear transformation matrices and solves a linear system to optimally combine them while preserving input-output mappings from individual task models. Extensive experiments on vision transformers and language models demonstrate that our method consistently outperforms existing methods, achieving superior performance and enhanced out-of-distribution generalization. 

**Abstract (ZH)**: 模型合并是一种关键的技术，用于将多个微调模型的能力结合起来，而无需进行额外的训练。现有方法通常将参数视为向量，而忽视了线性变换矩阵的内在结构——这些矩阵构成了模型参数的主要部分。线性变换矩阵是神经网络的基础组件，通过线性组合将输入表示映射到输出特征。受线性表示假设的启发，我们提出了任务矩阵的概念，并提出了一种新的方法——基于任务矩阵叠加特征（SFTM），该方法将单个任务模型的特征叠加到一个合并模型中。SFTM 使用奇异值分解来识别线性变换矩阵的特征基，并通过求解线性方程组来最优地将它们结合起来，同时保留单个任务模型的输入-输出映射。我们在视觉变换器和语言模型上的广泛实验表明，我们的方法在性能和泛化能力方面均优于现有方法，实现了显著的改进。 

---
# Simulations of Common Unsupervised Domain Adaptation Algorithms for Image Classification 

**Title (ZH)**: 常见无监督领域适应算法在图像分类中的模拟研究 

**Authors**: Ahmad Chaddad, Yihang Wu, Yuchen Jiang, Ahmed Bouridane, Christian Desrosiers  

**Link**: [PDF](https://arxiv.org/pdf/2502.10694)  

**Abstract**: Traditional machine learning assumes that training and test sets are derived from the same distribution; however, this assumption does not always hold in practical applications. This distribution disparity can lead to severe performance drops when the trained model is used in new data sets. Domain adaptation (DA) is a machine learning technique that aims to address this problem by reducing the differences between domains. This paper presents simulation-based algorithms of recent DA techniques, mainly related to unsupervised domain adaptation (UDA), where labels are available only in the source domain. Our study compares these techniques with public data sets and diverse characteristics, highlighting their respective strengths and drawbacks. For example, Safe Self-Refinement for Transformer-based DA (SSRT) achieved the highest accuracy (91.6\%) in the office-31 data set during our simulations, however, the accuracy dropped to 72.4\% in the Office-Home data set when using limited batch sizes. In addition to improving the reader's comprehension of recent techniques in DA, our study also highlights challenges and upcoming directions for research in this domain. The codes are available at this https URL. 

**Abstract (ZH)**: 传统的机器学习假设训练集和测试集来自相同的分布；然而，在实际应用中，这一假设并不总是成立。这种分布差异会导致在新数据集上使用训练好的模型时出现严重的性能下降。领域适应（Domain Adaptation, DA）是一种机器学习技术，旨在通过减少不同领域之间的差异来解决这一问题。本文提出了基于仿真的近年来DA技术算法，主要关注无监督领域适应（Unsupervised Domain Adaptation, UDA）方法，其中目标域只有标签。我们的研究通过公共数据集和多样化的特征，比较了这些技术的优缺点。例如，在我们对Office-31数据集的仿真实验中，基于Transformer的Safe Self-Refinement for DA（SSRT）方法达到了最高的准确率（91.6%），但在使用有限批次大小的Office-Home数据集上，准确率下降到了72.4%。此外，我们的研究不仅有助于读者理解DA领域的最新技术，还指出了该领域存在的挑战和未来的研究方向。相关代码可在此网址获取：[this https URL]。 

---
# Self-Explaining Hypergraph Neural Networks for Diagnosis Prediction 

**Title (ZH)**: 自我解释超图神经网络在诊断预测中的应用 

**Authors**: Leisheng Yu, Yanxiao Cai, Minxing Zhang, Xia Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10689)  

**Abstract**: The burgeoning volume of electronic health records (EHRs) has enabled deep learning models to excel in predictive healthcare. However, for high-stakes applications such as diagnosis prediction, model interpretability remains paramount. Existing deep learning diagnosis prediction models with intrinsic interpretability often assign attention weights to every past diagnosis or hospital visit, providing explanations lacking flexibility and succinctness. In this paper, we introduce SHy, a self-explaining hypergraph neural network model, designed to offer personalized, concise and faithful explanations that allow for interventions from clinical experts. By modeling each patient as a unique hypergraph and employing a message-passing mechanism, SHy captures higher-order disease interactions and extracts distinct temporal phenotypes as personalized explanations. It also addresses the incompleteness of the EHR data by accounting for essential false negatives in the original diagnosis record. A qualitative case study and extensive quantitative evaluations on two real-world EHR datasets demonstrate the superior predictive performance and interpretability of SHy over existing state-of-the-art models. 

**Abstract (ZH)**: 电子健康记录（EHRs）的海量增长使得深度学习模型在预测性医疗保健方面表现出色。然而，在诊断预测等高风险应用中，模型的可解释性仍然至关重要。现有的具有内在可解释性的深度学习诊断预测模型往往会为每个过去的诊断或医院访问分配注意权重，提供的解释缺乏灵活性和简洁性。本文介绍了一种名为SHy的自我解释超图神经网络模型，旨在提供个性化、简洁且忠实的解释，从而允许临床专家进行干预。通过将每位患者建模为独特的超图，并采用消息传递机制，SHy捕获了更高阶的疾病交互，并提取出个性化的临时表型作为解释。此外，SHy还通过考虑原始诊断记录中的关键假阴性结果来解决EHR数据的不完整性问题。通过在两个实际EHR数据集上的质性案例研究和广泛的定量评估，证明了SHy在预测性能和可解释性方面优于现有最先进的模型。 

---
# GenComUI: Exploring Generative Visual Aids as Medium to Support Task-Oriented Human-Robot Communication 

**Title (ZH)**: GenComUI：探索生成式视觉辅助作为支持任务导向的人机通信中介的研究 

**Authors**: Yate Ge, Meiying Li, Xipeng Huang, Yuanda Hu, Qi Wang, Xiaohua Sun, Weiwei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.10678)  

**Abstract**: This work investigates the integration of generative visual aids in human-robot task communication. We developed GenComUI, a system powered by large language models that dynamically generates contextual visual aids (such as map annotations, path indicators, and animations) to support verbal task communication and facilitate the generation of customized task programs for the robot. This system was informed by a formative study that examined how humans use external visual tools to assist verbal communication in spatial tasks. To evaluate its effectiveness, we conducted a user experiment (n = 20) comparing GenComUI with a voice-only baseline. The results demonstrate that generative visual aids, through both qualitative and quantitative analysis, enhance verbal task communication by providing continuous visual feedback, thus promoting natural and effective human-robot communication. Additionally, the study offers a set of design implications, emphasizing how dynamically generated visual aids can serve as an effective communication medium in human-robot interaction. These findings underscore the potential of generative visual aids to inform the design of more intuitive and effective human-robot communication, particularly for complex communication scenarios in human-robot interaction and LLM-based end-user development. 

**Abstract (ZH)**: 本文探讨了生成式视觉辅助在人类与机器人任务交流中的整合。我们开发了GenComUI系统，该系统基于大型语言模型动态生成情境视觉辅助（如地图标注、路径指示和动画），以支持口头任务交流并促进为机器人生成定制任务程序。该系统的开发灵感来自于一项形成性研究，该研究探讨了人类如何利用外部视觉工具来辅助空间任务中的口头交流。为了评估其有效性，我们进行了一项包含20名用户的实验，将GenComUI与仅语音baseline进行了比较。结果表明，生成式视觉辅助通过定性和定量分析，通过提供连续的视觉反馈，增强了口头任务交流，从而促进了自然和有效的机器人与人类交流。此外，该研究还提供了一系列设计启示，强调了动态生成的视觉辅助在人机交互中的有效交流媒介作用。这些发现突显了生成式视觉辅助在设计更具直观性和有效性的人机交流方面的作用，特别是在复杂的人机交互场景以及基于LLM的最终用户开发中的潜力。 

---
# Proof of Response 

**Title (ZH)**: 响应证明 

**Authors**: Illia Polosukhin, Alex Skidanov  

**Link**: [PDF](https://arxiv.org/pdf/2502.10637)  

**Abstract**: We present a mechanism that for a network of participants allows one participant of the network (Alice) to request some data from another participant (Bob) and either receive a response from Bob within a known-in-advance, bounded time b, or receive a proof that at least one edge on the way to Bob was broken within b, or receive a streaming payment proportional to time passed beyond b during which neither was received. This mechanism allows for building downstream applications that require provable responses from other participants, such as decentralized storage solutions, decentralized AI agents, and more. 

**Abstract (ZH)**: 我们提出了一种机制，该机制适用于一组参与者，在此机制下，网络中的一个参与者（Alice）可以向另一个参与者（Bob）请求一些数据，并且要么在事先已知且限定的时间b内收到Bob的响应，要么在b时间内收到证明，证明至少有一个通往Bob的路径上的连接在传输过程中被中断，要么在超出b的时间内收到一个与未收到响应的时间相关的流式支付。此机制允许构建需要其他参与者可验证响应的下游应用，如去中心化存储解决方案、去中心化AI代理等。 

---
# ControllableGPT: A Ground-Up Designed Controllable GPT for Molecule Optimization 

**Title (ZH)**: 可控GPT：自底向上的设计可控GPT分子优化模型 

**Authors**: Xuefeng Liu, Songhao Jiang, Bo Li, Rick Stevens  

**Link**: [PDF](https://arxiv.org/pdf/2502.10631)  

**Abstract**: Large Language Models (LLMs) employ three popular training approaches: Masked Language Models (MLM), Causal Language Models (CLM), and Sequence-to-Sequence Models (seq2seq). However, each approach has its strengths and limitations, and faces challenges in addressing specific tasks that require controllable and bidirectional generation, such as drug optimization. To address this challenge, inspired by the biological processes of growth and evolution, which involve the expansion, shrinking, and mutation of sequences, we introduce ControllableGPT. This initiative represents the first effort to combine the advantages of MLM, CLM, and seq2seq into a single unified, controllable GPT framework. It enables the precise management of specific locations and ranges within a sequence, allowing for expansion, reduction, or mutation over chosen or random lengths, while maintaining the integrity of any specified positions or subsequences. In this work, we designed ControllableGPT for drug optimization from the ground up, which included proposing the Causally Masked Seq2seq (CMS) objective, developing the training corpus, introducing a novel pre-training approach, and devising a unique generation process. We demonstrate the effectiveness and controllability of ControllableGPT by conducting experiments on drug optimization tasks for both viral and cancer benchmarks, surpassing competing baselines. 

**Abstract (ZH)**: 大型语言模型（LLMs）采用三种流行的训练方法：掩蔽语言模型（MLM）、因果语言模型（CLM）和序列到序列模型（seq2seq）。然而，每种方法都有其优缺点，并且面对着在要求可控和双向生成的任务（如药物优化）中相应特定挑战的困难。为了解决这一挑战，受生物生长和进化过程中的序列扩展、收缩和变异启发，我们引入了ControllableGPT。这一举措代表了首次将MLM、CLM和seq2seq的优势结合到一个统一的可控GPT框架中的尝试。该框架能够精确管理序列中特定位置和范围，并允许在选定或随机长度上进行扩展、缩减或变异，同时保持任何指定位置或子序列的完整性。在本研究中，我们从头开始设计了ControllableGPT，以用于药物优化，其中包括提出因果掩蔽序列到序列（CMS）目标、开发训练语料库、引入新的预训练方法以及设计独特的生成过程。通过在病毒和癌症基准数据上的药物优化任务中进行实验，我们展示了ControllableGPT的有效性和可控性，并超过了现有的基线方法。 

---
# K-Edit: Language Model Editing with Contextual Knowledge Awareness 

**Title (ZH)**: K-编辑：具有上下文知识意识的语言模型编辑 

**Authors**: Elan Markowitz, Anil Ramakrishna, Ninareh Mehrabi, Charith Peris, Rahul Gupta, Kai-Wei Chang, Aram Galstyan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10626)  

**Abstract**: As the world changes, we need to be able to update our models and correct false information without costly retraining. Knowledge-based model editing enables precise modifications to the weights of large language models in order to modify the information encoded within. Recent approaches have seen success in enabling recall of edited information for thousands of edits at once. However, these approaches fail to produce edits that account for associated contextual information. We present K-Edit, an effective approach to generating contextually consistent knowledge edits. By using knowledge graphs, which maintain contextual consistency when an edge is edited, we are able to generate additional \textit{contextual edits} that ensure consistency of related information in the language model. Our experiments demonstrate significant improvements in multi-hop question answering while maintaining the general effectiveness and scalability of model edits. 

**Abstract (ZH)**: 随着世界的变化，我们需要能够在不进行昂贵的重新训练的情况下更新我们的模型并纠正虚假信息。基于知识的模型编辑能使我们精确修改大规模语言模型的权重，从而修改其中编码的信息。近期的方法在一次性实现数千次编辑的召回方面取得了成功。然而，这些方法未能生成考虑到相关上下文信息的编辑。我们提出了K-Edit，这是一种生成上下文一致的知识编辑的有效方法。通过使用保持上下文一致性的知识图谱，我们能够生成额外的\textit{上下文编辑}，以确保语言模型中相关信息的一致性。我们的实验结果显示，在多跳问答方面有显著改进，同时保持模型编辑的一般有效性和可扩展性。 

---
# Network evasion detection with Bi-LSTM model 

**Title (ZH)**: 使用双向长短期记忆（Bi-LSTM）模型进行网络规避检测 

**Authors**: Kehua Chen, Jingping Jia  

**Link**: [PDF](https://arxiv.org/pdf/2502.10624)  

**Abstract**: Network evasion detection aims to distinguish whether the network flow comes from link layer exists network evasion threat, which is a means to disguise the data traffic on detection system by confusing the signature. Since the previous research works has all sorts of frauds, we propose a architecture with deep learning network to handle this problem. In this paper, we extract the critical information as key features from data frame and also specifically propose to use bidirectional long short-term memory (Bi-LSTM) neural network which shows an outstanding performance to trace the serial information, to encode both the past and future trait on the network flows. Furthermore we introduce a classifier named Softmax at the bottom of Bi-LSTM, holding a character to select the correct class. All experiments results shows that we can achieve a significant performance with a deep Bi-LSTM in network evasion detection and it's average accuracy reaches 96.1%. 

**Abstract (ZH)**: 网络逃逸检测旨在区分网络流是否来自链路层存在的网络逃逸威胁，这是一种通过混淆特征来欺骗检测系统的手段，目的是隐藏数据流量。由于之前的科研工作存在各种各样的误导和不足，我们提出了一种基于深度学习的架构来解决这个问题。在本文中，我们从数据帧中提取关键信息作为特征，并特别提出使用双向长短期记忆（Bi-LSTM）神经网络，这种网络在追踪序列信息方面表现出色，能够同时编码网络流的过去和未来特征。此外，我们在Bi-LSTM的底部引入了一个名为Softmax的分类器，该分类器具备选择正确类别的功能。实验结果显示，我们可以通过深度Bi-LSTM在网络逃逸检测中实现显著性能，其平均准确率达到了96.1%。 

---
# Optimizing CNN Architectures for Advanced Thoracic Disease Classification 

**Title (ZH)**: 优化卷积神经网络架构以实现高级胸腔疾病分类 

**Authors**: Tejas Mirthipati  

**Link**: [PDF](https://arxiv.org/pdf/2502.10614)  

**Abstract**: Machine learning, particularly convolutional neural networks (CNNs), has shown promise in medical image analysis, especially for thoracic disease detection using chest X-ray images. In this study, we evaluate various CNN architectures, including binary classification, multi-label classification, and ResNet50 models, to address challenges like dataset imbalance, variations in image quality, and hidden biases. We introduce advanced preprocessing techniques such as principal component analysis (PCA) for image compression and propose a novel class-weighted loss function to mitigate imbalance issues. Our results highlight the potential of CNNs in medical imaging but emphasize that issues like unbalanced datasets and variations in image acquisition methods must be addressed for optimal model performance. 

**Abstract (ZH)**: 机器学习，特别是卷积神经网络（CNNs），在医学图像分析中显示出前景，特别是在使用胸部X光图像检测胸部疾病方面。本研究评估了多种CNN架构，包括二分类、多标签分类和ResNet50模型，以应对数据集不平衡、图像质量差异和隐含偏见等问题。我们引入了先进的预处理技术，如主成分分析（PCA）进行图像压缩，并提出了一种新的类别加权损失函数，以减轻数据集不平衡的问题。研究结果强调了CNNs在医学成像中的潜力，但也指出了必须解决数据集不均衡和图像采集方法差异等问题，以实现最优模型性能。 

---
# Post-training an LLM for RAG? Train on Self-Generated Demonstrations 

**Title (ZH)**: 对LLM进行后训练以用于RAG？基于自生成示范的训练 

**Authors**: Matthew Finlayson, Ilia Kulikov, Daneil M. Bikel, Barlas Oguz, Xilun Chen, Aasish Pappu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10596)  

**Abstract**: Large language models (LLMs) often struggle with knowledge intensive NLP tasks, such as answering "Who won the latest World Cup?" because the knowledge they learn during training may be insufficient or outdated. Conditioning generation on retrieved documents -- a technique known as retrieval augmented generation (RAG) -- mitigates these shortcomings by allowing the model to leverage in-context information. Practitioners can improve LLM RAG performance by fine-tuning on retrieval-augmented instructions, but must beware that this can cause undesirable model behaviors like hallucinations. We attribute this degradation to the fact that the training data is likely to be out-of-distribution for the model and may suffer from quality issues, such as misalignment between retrievals and target responses (since retrievals are frequently added post-hoc). We propose a recipe for training RAG-enabled LLMs using self-generated demonstrations, thereby avoiding training on out-of-distribution text and integrating retrievals into the LLM responses. We evaluate our method on knowledge intensive question answering (QA) tasks and show that our method teaches LLMs to properly handle in-context retrievals and abstain from questions it will likely get wrong. Compared to conventional RA-IT methods, our method prevents model degradation in non-RAG settings while exhibiting superior QA performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在知识密集型的自然语言处理任务上常常表现不佳，例如回答“最近的世界杯是谁赢的？”这类问题，因为它们在训练过程中学到的知识可能不足或过时。通过检索文档来生成——一种称为检索增强生成（RAG）的技术——可以缓解这些不足之处，使模型能够利用上下文信息。实践者可以通过对检索增强的指令进行微调来提高LLM的RAG性能，但必须注意这可能会导致模型行为异常，如臆想现象。我们将这种退化归因于训练数据很可能不符合模型的分布情况，并且可能会有质量问题，例如检索结果与目标响应之间的不一致（因为检索结果经常是在事后添加的）。我们提出了一种训练RAG增强的LLMs的方案，通过自我生成的示范训练，避免使用不符合分布的数据，并将检索结果整合到LLM的响应中。我们在知识密集型问答（QA）任务上评估了该方法，结果显示该方法使LLM能够正确处理上下文检索，并避免回答可能答错的问题。与传统的RA-IT方法相比，我们的方法能够在非RAG设置中防止模型退化，同时表现出更优秀的QA性能。 

---
# Towards Self-Supervised Covariance Estimation in Deep Heteroscedastic Regression 

**Title (ZH)**: 面向深度异方差回归的自监督协方差估计oueur
user
请帮我总结一下这个论文标题的关键点。 

**Authors**: Megh Shukla, Aziz Shameem, Mathieu Salzmann, Alexandre Alahi  

**Link**: [PDF](https://arxiv.org/pdf/2502.10587)  

**Abstract**: Deep heteroscedastic regression models the mean and covariance of the target distribution through neural networks. The challenge arises from heteroscedasticity, which implies that the covariance is sample dependent and is often unknown. Consequently, recent methods learn the covariance through unsupervised frameworks, which unfortunately yield a trade-off between computational complexity and accuracy. While this trade-off could be alleviated through supervision, obtaining labels for the covariance is non-trivial. Here, we study self-supervised covariance estimation in deep heteroscedastic regression. We address two questions: (1) How should we supervise the covariance assuming ground truth is available? (2) How can we obtain pseudo labels in the absence of the ground-truth? We address (1) by analysing two popular measures: the KL Divergence and the 2-Wasserstein distance. Subsequently, we derive an upper bound on the 2-Wasserstein distance between normal distributions with non-commutative covariances that is stable to optimize. We address (2) through a simple neighborhood based heuristic algorithm which results in surprisingly effective pseudo labels for the covariance. Our experiments over a wide range of synthetic and real datasets demonstrate that the proposed 2-Wasserstein bound coupled with pseudo label annotations results in a computationally cheaper yet accurate deep heteroscedastic regression. 

**Abstract (ZH)**: 深度异方差回归模型通过神经网络建模目标分布的均值和协方差。挑战来自于异方差性，这意味着协方差依赖于样本且通常未知。因此，近期的方法通过无监督框架学习协方差，不幸的是，这导致了计算复杂性和准确性的权衡。虽然通过监督可以缓解这一权衡，但获得协方差的标签却具有一定的难度。在此，我们研究深度异方差回归中的自监督协方差估计。我们探讨了两个问题：(1) 假设存在真实标签，我们应如何监督协方差？(2) 在缺乏真实标签的情况下，我们如何获取伪标签？对于问题(1)，我们分析了两种流行度量方法：KL散度和2-Wasserstein距离。随后，我们推导出一种基于非交换协方差的2-Wasserstein距离的上界，该上界可用于优化。对于问题(2)，我们提出了一个简单的基于邻域的启发式算法，结果生成了对协方差非常有效的伪标签。在合成数据集和真实数据集的广泛实验中，我们发现提出的2-Wasserstein界结合伪标签标记在保持计算效率的同时，能够实现准确的深度异方差回归。 

---
# Do We Need to Verify Step by Step? Rethinking Process Supervision from a Theoretical Perspective 

**Title (ZH)**: 我们是否需要逐步骤验证？从理论角度重新思考过程监督 

**Authors**: Zeyu Jia, Alexander Rakhlin, Tengyang Xie  

**Link**: [PDF](https://arxiv.org/pdf/2502.10581)  

**Abstract**: As large language models have evolved, it has become crucial to distinguish between process supervision and outcome supervision -- two key reinforcement learning approaches to complex reasoning tasks. While process supervision offers intuitive advantages for long-term credit assignment, the precise relationship between these paradigms has remained an open question. Conventional wisdom suggests that outcome supervision is fundamentally more challenging due to the trajectory-level coverage problem, leading to significant investment in collecting fine-grained process supervision data.
In this paper, we take steps towards resolving this debate. Our main theorem shows that, under standard data coverage assumptions, reinforcement learning through outcome supervision is no more statistically difficult than through process supervision, up to polynomial factors in horizon. At the core of this result lies the novel Change of Trajectory Measure Lemma -- a technical tool that bridges return-based trajectory measure and step-level distribution shift. Furthermore, for settings with access to a verifier or a rollout capability, we prove that any policy's advantage function can serve as an optimal process reward model, providing a direct connection between outcome and process supervision. These findings suggest that the empirically observed performance gap -- if any -- between outcome and process supervision likely stems from algorithmic limitations rather than inherent statistical difficulties, potentially transforming how we approach data collection and algorithm design for reinforcement learning. 

**Abstract (ZH)**: 随着大型语言模型的发展，区分过程监督和结果监督——两种关键的强化学习方法，变得至关重要。过程监督为长期信用分配提供直观的优势，但这些范式的精确关系仍然存在争议。传统观点认为，结果监督由于轨迹级覆盖率问题更加困难，因此在收集细粒度的过程监督数据方面投入了大量资源。

在本文中，我们朝着解决这一争议迈出了重要一步。我们的主要定理表明，在标准数据覆盖假定下，通过结果监督进行强化学习与通过过程监督进行学习在统计上难度相同，最多相差多项式的因子。这一结论的核心在于我们引入的新型“轨迹测度变换引理”——一个技术工具，它将基于回报的轨迹测度与步骤级分布转移联系起来。此外，在可以访问验证器或重播能力的环境中，我们证明任何策略的优势函数都可以作为最优的过程奖励模型，这提供了结果监督和过程监督之间的直接联系。这些发现表明，结果监督和过程监督之间观察到的性能差距（如有）可能主要源自算法限制而非基本的统计难题，这可能会影响我们处理强化学习中的数据收集和算法设计的方式。 

---
# Man Made Language Models? Evaluating LLMs' Perpetuation of Masculine Generics Bias 

**Title (ZH)**: 人类制造的语言模型？评估大型语言模型延续男性通用代词偏见的情况 

**Authors**: Enzo Doyen, Amalia Todirascu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10577)  

**Abstract**: Large language models (LLMs) have been shown to propagate and even amplify gender bias, in English and other languages, in specific or constrained contexts. However, no studies so far have focused on gender biases conveyed by LLMs' responses to generic instructions, especially with regard to masculine generics (MG). MG are a linguistic feature found in many gender-marked languages, denoting the use of the masculine gender as a "default" or supposedly neutral gender to refer to mixed group of men and women, or of a person whose gender is irrelevant or unknown. Numerous psycholinguistics studies have shown that MG are not neutral and induce gender bias. This work aims to analyze the use of MG by both proprietary and local LLMs in responses to generic instructions and evaluate their MG bias rate. We focus on French and create a human noun database from existing lexical resources. We filter existing French instruction datasets to retrieve generic instructions and analyze the responses of 6 different LLMs. Overall, we find that $\approx$39.5\% of LLMs' responses to generic instructions are MG-biased ($\approx$73.1\% across responses with human nouns). Our findings also reveal that LLMs are reluctant to using gender-fair language spontaneously. 

**Abstract (ZH)**: 大型语言模型（LLMs）在英语和其他语言中，已在特定或受限的情境下被证明会传播甚至放大性别偏见。然而，迄今为止，尚无研究专门关注LLMs对通用指令的响应中所传达的性别偏见，尤其是与男性通用（MG）相关的问题。男性通用是一种在许多标注性别的语言中发现的语言特征，表示使用男性性别作为“默认”或理论上中性的性别来指代混杂的男性和女性群体，或指代性别无关或未知的人群。众多心理学语言学研究已经表明，男性通用并不中立，会产生性别偏见。本研究旨在分析开源和本地LLMs在对通用指令的响应中使用男性通用的状况，并评估其性别偏见率。我们重点关注法语，并从现有的词库资源中创建一个人名数据库。我们筛选现有的法语指令数据集以提取通用指令，并分析6种不同LLMs的响应。总体而言，我们发现约39.5%的LLMs对通用指令的响应具有男性通用偏见（在包含人类名词的响应中，该比例约为73.1%）。我们的研究结果还表明，LLMs自发使用性别公平语言的积极性较低。 

---
# An Innovative Next Activity Prediction Approach Using Process Entropy and DAW-Transformer 

**Title (ZH)**: 一种基于过程熵和DAW-Transformer的创新活动预测方法 

**Authors**: Hadi Zare, Mostafa Abbasi, Maryam Ahang, Homayoun Najjaran  

**Link**: [PDF](https://arxiv.org/pdf/2502.10573)  

**Abstract**: Purpose - In Business Process Management (BPM), accurate prediction of the next activities is vital for operational efficiency and decision-making. Current Artificial Intelligence (AI)/Machine Learning (ML) models struggle with the complexity and evolving nature of business process event logs, balancing accuracy and interpretability. This paper proposes an entropy-driven model selection approach and DAW-Transformer, which stands for Dynamic Attribute-Aware Transformer, to integrate all attributes with a dynamic window for better accuracy.
Design/methodology/approach - This paper introduces a novel next-activity prediction approach that uses process entropy to assess the complexity of event logs and dynamically select the most suitable ML model. A new transformer-based architecture with multi-head attention and dynamic windowing mechanism, DAW-Transformer, is proposed to capture long-range dependencies and utilize all relevant event log attributes. Experiments were conducted on six public datasets, and the performance was evaluated with process entropy.
Finding - The results demonstrate the effectiveness of the approach across these publicly available datasets. DAW-Transformer achieved superior performance, especially on high-entropy datasets such as Sepsis exceeding Limited window Multi-Transformers by 4.69% and a benchmark CNN-LSTM-SAtt model by 3.07%. For low-entropy datasets like Road Traffic Fine, simpler, more interpretable algorithms like Random Forest performed nearly as well as the more complex DAW-Transformer and offered better handling of imbalanced data and improved explainability.
Originality/ value - This work's novelty lies in the proposed DAW-Transformer, with a dynamic window and considering all relevant attributes. Also, entropy-driven selection methods offer a robust, accurate, and interpretable solution for next-activity prediction. 

**Abstract (ZH)**: 目的 - 在业务流程管理（BPM）中，精确预测下一个活动对于操作效率和决策制定至关重要。当前的人工智能（AI）/机器学习（ML）模型难以平衡业务流程事件日志的复杂性和变化性，同时保持准确性和可解释性。本文提出了一种熵驱动的模型选择方法和DAW-Transformer（动态属性感知变压器），通过动态窗口更好地整合所有属性，提高预测精度。

设计/方法 - 本文提出了一种新颖的下一个活动预测方法，利用过程熵评估事件日志的复杂性，并动态选择最合适的机器学习模型。提出了一个新的基于变压器的架构，该架构结合了多头注意力机制和动态窗口机制，称为DAW-Transformer（动态属性感知变压器），以捕捉长距离依赖关系并利用所有相关事件日志属性。在六个公开数据集上进行了实验，并使用过程熵评估了性能。

发现 - 实验结果表明，该方法在这些公开数据集上具有有效性。DAW-Transformer在高熵数据集（如Sepsis）中表现尤为出色，超越了有限窗口多变压器4.69%，并且在基准CNN-LSTM-SAtt模型上提高了3.07%。对于低熵数据集（如Road Traffic Fine），更简单且更具可解释性的算法（如随机森林）几乎与DAW-Transformer表现相同，并且更适用于不平衡数据处理和提高解释性。

创新/价值 - 本文的创新之处在于提出了具有动态窗口的DAW-Transformer的同时考虑所有相关属性。此外，熵驱动的模型选择方法为下一个活动预测提供了一种稳健、准确且可解释的解决方案。 

---
# HADL Framework for Noise Resilient Long-Term Time Series Forecasting 

**Title (ZH)**: HADL框架在噪声鲁棒长期时间序列预测中的应用 

**Authors**: Aditya Dey, Jonas Kusch, Fadi Al Machot  

**Link**: [PDF](https://arxiv.org/pdf/2502.10569)  

**Abstract**: Long-term time series forecasting is critical in domains such as finance, economics, and energy, where accurate and reliable predictions over extended horizons drive strategic decision-making. Despite the progress in machine learning-based models, the impact of temporal noise in extended lookback windows remains underexplored, often degrading model performance and computational efficiency. In this paper, we propose a novel framework that addresses these challenges by integrating the Discrete Wavelet Transform (DWT) and Discrete Cosine Transform (DCT) to perform noise reduction and extract robust long-term features. These transformations enable the separation of meaningful temporal patterns from noise in both the time and frequency domains. To complement this, we introduce a lightweight low-rank linear prediction layer that not only reduces the influence of residual noise but also improves memory efficiency. Our approach demonstrates competitive robustness to noisy input, significantly reduces computational complexity, and achieves competitive or state-of-the-art forecasting performance across diverse benchmark datasets. Extensive experiments reveal that the proposed framework is particularly effective in scenarios with high noise levels or irregular patterns, making it well suited for real-world forecasting tasks. The code is available in this https URL. 

**Abstract (ZH)**: 长序列时间序列预测在金融、经济学和能源等领域至关重要，准确可靠的长期预测能够驱动战略决策。尽管基于机器学习的模型取得了进展，但扩展回溯窗口中的时间噪声影响仍被忽视，这往往导致模型性能下降和计算效率降低。本文提出了一种新的框架，通过集成离散小波变换（DWT）和离散余弦变换（DCT）来减少噪声并提取稳健的长期特征，从而解决这些挑战。这些变换能够在时域和频域中分离有意义的时间模式和噪声。为此，我们还引入了一种轻量级低秩线性预测层，它不仅减少了剩余噪声的影响，还提高了内存效率。我们的方法在受到噪声干扰的情况下表现出较强的鲁棒性，显著降低了计算复杂度，并在各种基准数据集上实现了竞争力或最先进的预测性能。广泛实验证明，所提出的框架特别适用于高噪声水平或不规则模式的场景，使其适用于实际的预测任务。代码可以在以下链接获取：[这里插入链接] 

---
# Efficient Hierarchical Contrastive Self-supervising Learning for Time Series Classification via Importance-aware Resolution Selection 

**Title (ZH)**: 基于重要性aware分辨率选择的高效分层对比自监督学习方法用于时间序列分类 

**Authors**: Kevin Garcia, Juan Manuel Perez, Yifeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10567)  

**Abstract**: Recently, there has been a significant advancement in designing Self-Supervised Learning (SSL) frameworks for time series data to reduce the dependency on data labels. Among these works, hierarchical contrastive learning-based SSL frameworks, which learn representations by contrasting data embeddings at multiple resolutions, have gained considerable attention. Due to their ability to gather more information, they exhibit better generalization in various downstream tasks. However, when the time series data length is significant long, the computational cost is often significantly higher than that of other SSL frameworks. In this paper, to address this challenge, we propose an efficient way to train hierarchical contrastive learning models. Inspired by the fact that each resolution's data embedding is highly dependent, we introduce importance-aware resolution selection based training framework to reduce the computational cost. In the experiment, we demonstrate that the proposed method significantly improves training time while preserving the original model's integrity in extensive time series classification performance evaluations. Our code could be found here, this https URL 

**Abstract (ZH)**: 近年来，自监督学习（SSL）框架在时间序列数据中的设计方面取得了显著进步，以减少对数据标签的依赖。在这类工作中，基于分层对比学习的SSL框架因其能够在多个分辨率上学习表示而受到了广泛关注。由于它们能够收集更多的信息，因此在各种下游任务中表现出更好的泛化能力。然而，当时间序列数据长度显著增加时，其计算成本往往远高于其他SSL框架。为了应对这一挑战，本文提出了一种高效的分层对比学习模型训练方法。受到每个分辨率的数据嵌入高度依赖性的启发，我们引入了一种基于重要性感知的分辨率选择训练框架以降低计算成本。在实验中，我们展示了所提出的方法在广泛的时间序列分类性能评估中显著提高了训练时间，同时保持了原始模型的完整性。我们的代码可以在这里找到：[this https URL] 

---
# SAMRI-2: A Memory-based Model for Cartilage and Meniscus Segmentation in 3D MRIs of the Knee Joint 

**Title (ZH)**: SAMRI-2：一种基于内存的方法用于膝关节3D MRI中软骨和半月板分割 

**Authors**: Danielle L. Ferreira, Bruno A. A. Nunes, Xuzhe Zhang, Laura Carretero Gomez, Maggie Fung, Ravi Soni  

**Link**: [PDF](https://arxiv.org/pdf/2502.10559)  

**Abstract**: Accurate morphometric assessment of cartilage-such as thickness/volume-via MRI is essential for monitoring knee osteoarthritis. Segmenting cartilage remains challenging and dependent on extensive expert-annotated datasets, which are heavily subjected to inter-reader variability. Recent advancements in Visual Foundational Models (VFM), especially memory-based approaches, offer opportunities for improving generalizability and robustness. This study introduces a deep learning (DL) method for cartilage and meniscus segmentation from 3D MRIs using interactive, memory-based VFMs. To improve spatial awareness and convergence, we incorporated a Hybrid Shuffling Strategy (HSS) during training and applied a segmentation mask propagation technique to enhance annotation efficiency. We trained four AI models-a CNN-based 3D-VNet, two automatic transformer-based models (SaMRI2D and SaMRI3D), and a transformer-based promptable memory-based VFM (SAMRI-2)-on 3D knee MRIs from 270 patients using public and internal datasets and evaluated on 57 external cases, including multi-radiologist annotations and different data acquisitions. Model performance was assessed against reference standards using Dice Score (DSC) and Intersection over Union (IoU), with additional morphometric evaluations to further quantify segmentation accuracy. SAMRI-2 model, trained with HSS, outperformed all other models, achieving an average DSC improvement of 5 points, with a peak improvement of 12 points for tibial cartilage. It also demonstrated the lowest cartilage thickness errors, reducing discrepancies by up to threefold. Notably, SAMRI-2 maintained high performance with as few as three user clicks per volume, reducing annotation effort while ensuring anatomical precision. This memory-based VFM with spatial awareness offers a novel approach for reliable AI-assisted knee MRI segmentation, advancing DL in musculoskeletal imaging. 

**Abstract (ZH)**: 准确通过MRI评估软骨（如厚度/体积）的形态学参数对于监测膝关节骨关节炎至关重要。软骨分割仍具有挑战性，并依赖于大量专家注释的数据集，这些数据集受阅片者间差异的影响很大。近年来，视觉基础模型（VFM），特别是基于记忆的方法的进步为提高通用性和鲁棒性提供了机会。本研究介绍了一种用于从3D MRI中分割软骨和半月板的深度学习（DL）方法，采用交互式、基于记忆的VFM。为提高空间意识和收敛性，我们在训练过程中引入了混合洗牌策略（HSS），并应用分割掩码传播技术以提高注释效率。我们使用公共和内部数据集对四种AI模型——基于CNN的3D-VNet、两种自动变压器模型（SaMRI2D和SaMRI3D），以及一种基于变压器的可提示记忆VFM（SAMRI-2）进行了训练，并在包含多放射科医师注释和不同数据采集方法的57例外部病例上进行了评估。模型性能通过Dice评分（DSC）和交并比（IoU）与参考标准进行评估，并进行了额外的形态学评估以进一步量化分割准确性。训练了包含HSS的SAMRI-2模型，在胫骨软骨分割中平均DSC提高了5分，最高提高了12分。它还具有最低的软骨厚度误差，将差异减少了三倍。值得注意的是，SAMRI-2在每次体积只需三个用户点击的情况下仍能保持高水平性能，减少了注释工作同时保持解剖精确性。具有空间意识的记忆VFM提供了一种新的方法，用于实现可靠的AI辅助膝关节MRI分割，推动了肌肉骨骼成像领域的DL技术。 

---
# Synthesis of Dynamic Masks for Information-Theoretic Opacity in Stochastic Systems 

**Title (ZH)**: 在随机系统中基于信息论 opacity 的动态遮罩合成 

**Authors**: Sumukha Udupa, Chongyang Shi, Jie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10552)  

**Abstract**: In this work, we investigate the synthesis of dynamic information releasing mechanisms, referred to as ''masks'', to minimize information leakage from a stochastic system to an external observer. Specifically, for a stochastic system, an observer aims to infer whether the final state of the system trajectory belongs to a set of secret states. The dynamic mask seeks to regulate sensor information in order to maximize the observer's uncertainty about the final state, a property known as final-state opacity. While existing supervisory control literature on dynamic masks primarily addresses qualitative opacity, we propose quantifying opacity in stochastic systems by conditional entropy, which is a measure of information leakage in information security. We then formulate a constrained optimization problem to synthesize a dynamic mask that maximizes final-state opacity under a total cost constraint on masking. To solve this constrained optimal dynamic mask synthesis problem, we develop a novel primal-dual policy gradient method. Additionally, we present a technique for computing the gradient of conditional entropy with respect to the masking policy parameters, leveraging observable operators in hidden Markov models. To demonstrate the effectiveness of our approach, we apply our method to an illustrative example and a stochastic grid world scenario, showing how our algorithm optimally enforces final-state opacity under cost constraints. 

**Abstract (ZH)**: 在这项研究中，我们探讨了合成动态信息释放机制，称为“掩码”，以最小化从随机系统向外部观察者的信息泄露。具体而言，对于一个随机系统，观察者的目标是推断系统轨迹的最终状态是否属于一组秘密状态。动态掩码通过调节传感器信息来最大化观察者对最终状态的不确定性，这一属性称为最终状态不透明性。虽然现有的关于动态掩码的监督控制文献主要关注定性不透明性，我们提出通过条件熵来量化随机系统中的不透明性，条件熵是信息安全中衡量信息泄露的一种指标。然后，我们提出了一个约束优化问题，旨在在掩码总成本约束下合成一个最大化最终状态不透明性的动态掩码。为了解决这个受成本约束的最优动态掩码合成问题，我们开发了一种新的原始对偶策略梯度方法。此外，我们提出了一种技术，用于计算条件熵对掩码策略参数的梯度，利用隐马尔可夫模型中的可观测算子。为了展示我们方法的有效性，我们将该方法应用于一个示例和一个随机网格世界场景，展示了在成本约束下我们的算法如何最优地实现最终状态不透明性。 

---
# Memory, Benchmark & Robots: A Benchmark for Solving Complex Tasks with Reinforcement Learning 

**Title (ZH)**: 记忆、基准与机器人：基于强化学习解决复杂任务的基准测试 

**Authors**: Egor Cherepanov, Nikita Kachaev, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2502.10550)  

**Abstract**: Memory is crucial for enabling agents to tackle complex tasks with temporal and spatial dependencies. While many reinforcement learning (RL) algorithms incorporate memory, the field lacks a universal benchmark to assess an agent's memory capabilities across diverse scenarios. This gap is particularly evident in tabletop robotic manipulation, where memory is essential for solving tasks with partial observability and ensuring robust performance, yet no standardized benchmarks exist. To address this, we introduce MIKASA (Memory-Intensive Skills Assessment Suite for Agents), a comprehensive benchmark for memory RL, with three key contributions: (1) we propose a comprehensive classification framework for memory-intensive RL tasks, (2) we collect MIKASA-Base - a unified benchmark that enables systematic evaluation of memory-enhanced agents across diverse scenarios, and (3) we develop MIKASA-Robo - a novel benchmark of 32 carefully designed memory-intensive tasks that assess memory capabilities in tabletop robotic manipulation. Our contributions establish a unified framework for advancing memory RL research, driving the development of more reliable systems for real-world applications. The code is available at this https URL. 

**Abstract (ZH)**: 记忆对于使智能体能够解决具有时空间依赖性的复杂任务至关重要。虽然许多强化学习（RL）算法都包含了记忆机制，但该领域缺少一个用于评估智能体记忆能力的普遍基准，尤其是在多变场景下。这一差距在桌面机器人操作中尤为明显，因为记忆在这种具有部分可观察性的情境中是解决问题和确保鲁棒性能的关键，但尚未存在标准化基准。为解决这一问题，我们提出了MIKASA（Memory-Intensive Skills Assessment Suite for Agents），一个全面的内存强化学习基准，包括三项主要贡献：（1）我们提出了一种全面的分类框架，用于分类内存密集型RL任务；（2）我们收集了MIKASA-Base——一个统一基准，可以系统地评估增强记忆的智能体在不同场景下的性能；（3）我们开发了MIKASA-Robo——一个包含32个精心设计的记忆密集型任务的新颖基准，用于评估桌面机器人操作中的记忆能力。我们的贡献建立了一个统一框架，推动了内存强化学习研究的进步，促进了更可靠系统的开发，适用于实际应用。相关代码可在以下链接获取：[此链接]。 

---
# Learning to be Smooth: An End-to-End Differentiable Particle Smoother 

**Title (ZH)**: 学习平滑：一种端到端可微分粒子平滑器 

**Authors**: Ali Younis, Erik B. Sudderth  

**Link**: [PDF](https://arxiv.org/pdf/2502.10546)  

**Abstract**: For challenging state estimation problems arising in domains like vision and robotics, particle-based representations attractively enable temporal reasoning about multiple posterior modes. Particle smoothers offer the potential for more accurate offline data analysis by propagating information both forward and backward in time, but have classically required human-engineered dynamics and observation models. Extending recent advances in discriminative training of particle filters, we develop a framework for low-variance propagation of gradients across long time sequences when training particle smoothers. Our "two-filter'' smoother integrates particle streams that are propagated forward and backward in time, while incorporating stratification and importance weights in the resampling step to provide low-variance gradient estimates for neural network dynamics and observation models. The resulting mixture density particle smoother is substantially more accurate than state-of-the-art particle filters, as well as search-based baselines, for city-scale global vehicle localization from real-world videos and maps. 

**Abstract (ZH)**: 在视觉和机器人学等领域的挑战性状态估计问题中，基于粒子的表示方法能够吸引人地实现对多个后验模式的时序推理。粒子平滑器通过在时间的正向和反向传播信息，具有进行更准确的离线数据分析的潜力，但传统上需要人工设计的动力学和观测模型。在最近关于判别训练粒子滤波器的进展基础上，我们开发了一个框架，在训练粒子平滑器时能够在长时间序列中提供低方差梯度传播。我们的“两步滤波器”平滑器结合了时间正向和反向传播的粒子流，并在重采样步骤中引入分层和重要性加权，从而为神经网络的动力学和观测模型提供低方差梯度估计。由此产生的混合密度粒子平滑器在从真实世界视频和地图进行城市规模车辆全局定位方面显著优于最先进的粒子滤波器以及基于搜索的基准方法。 

---
# PolyPath: Adapting a Large Multimodal Model for Multi-slide Pathology Report Generation 

**Title (ZH)**: PolyPath: 调整大型多模态模型以生成多张病理切片报告 

**Authors**: Faruk Ahmed, Lin Yang, Tiam Jaroensri, Andrew Sellergren, Yossi Matias, Avinatan Hassidim, Greg S. Corrado, Dale R. Webster, Shravya Shetty, Shruthi Prabhakara, Yun Liu, Daniel Golden, Ellery Wulczyn, David F. Steiner  

**Link**: [PDF](https://arxiv.org/pdf/2502.10536)  

**Abstract**: The interpretation of histopathology cases underlies many important diagnostic and treatment decisions in medicine. Notably, this process typically requires pathologists to integrate and summarize findings across multiple slides per case. Existing vision-language capabilities in computational pathology have so far been largely limited to small regions of interest, larger regions at low magnification, or single whole-slide images (WSIs). This limits interpretation of findings that span multiple high-magnification regions across multiple WSIs. By making use of Gemini 1.5 Flash, a large multimodal model (LMM) with a 1-million token context window, we demonstrate the ability to generate bottom-line diagnoses from up to 40,000 768x768 pixel image patches from multiple WSIs at 10X magnification. This is the equivalent of up to 11 hours of video at 1 fps. Expert pathologist evaluations demonstrate that the generated report text is clinically accurate and equivalent to or preferred over the original reporting for 68% (95% CI: [60%, 76%]) of multi-slide examples with up to 5 slides. While performance decreased for examples with 6 or more slides, this study demonstrates the promise of leveraging the long-context capabilities of modern LMMs for the uniquely challenging task of medical report generation where each case can contain thousands of image patches. 

**Abstract (ZH)**: 病理组织学案例的解释对于医学中的许多重要诊断和治疗决策至关重要。这一过程通常要求病理学家整合和总结多个切片中的各项发现。目前，在计算病理学中的视觉-语言能力主要局限于感兴趣的小区域、低放大倍数的大区域或单张全切片图像（WSI）。这限制了对跨越多个WSI的多个高倍率区域的发现进行解释的能力。通过利用Gemini 1.5 Flash，一种具有100万词上下文窗口的大规模多模态模型（LMM），我们展示了从多张10倍放大倍率的WSI中多达40,000个768x768像素的图像块生成最终诊断结果的能力。这相当于长达11小时的1 fps视频。专家病理学家的评估表明，生成的报告文本在临床准确性上与原报告相当或更优，对于包含至多5张切片的68%（95%置信区间：[60%, 76%]）的多切片示例，生成的报告文本更受偏好。尽管对于包含6张或更多切片的示例，性能有所下降，但本研究展示了利用现代LMM的长上下文能力来完成医学报告生成这一独特挑战任务的潜力，而每例病例可能包含数千个图像块。 

---
# Tempo: Helping Data Scientists and Domain Experts Collaboratively Specify Predictive Modeling Tasks 

**Title (ZH)**: Tempo：帮助数据科学家和领域专家协作指定预测建模任务 

**Authors**: Venkatesh Sivaraman, Anika Vaishampayan, Xiaotong Li, Brian R Buck, Ziyong Ma, Richard D Boyce, Adam Perer  

**Link**: [PDF](https://arxiv.org/pdf/2502.10526)  

**Abstract**: Temporal predictive models have the potential to improve decisions in health care, public services, and other domains, yet they often fail to effectively support decision-makers. Prior literature shows that many misalignments between model behavior and decision-makers' expectations stem from issues of model specification, namely how, when, and for whom predictions are made. However, model specifications for predictive tasks are highly technical and difficult for non-data-scientist stakeholders to interpret and critique. To address this challenge we developed Tempo, an interactive system that helps data scientists and domain experts collaboratively iterate on model specifications. Using Tempo's simple yet precise temporal query language, data scientists can quickly prototype specifications with greater transparency about pre-processing choices. Moreover, domain experts can assess performance within data subgroups to validate that models behave as expected. Through three case studies, we demonstrate how Tempo helps multidisciplinary teams quickly prune infeasible specifications and identify more promising directions to explore. 

**Abstract (ZH)**: 时间预测模型有可能在医疗、公共管理及其他领域中提高决策质量，然而它们往往未能有效地支持决策者。先前的研究表明，模型行为与决策者预期之间的许多偏差源自模型规格化方面的问题，即预测的生成方式、时间和对谁有效。然而，预测任务中的模型规格化高度技术化，难以供非数据科学家的利益相关者解读和批评。为应对这一挑战，我们开发了 Tempo，一种互动系统，帮助数据科学家和领域专家协作迭代模型规格。借助 Tempo 简洁且精确的时间查询语言，数据科学家可以迅速制定更具透明度的规格原型。此外，领域专家可以评估数据子组内的性能，验证模型是否按预期行为。通过三个案例研究，我们展示了 Tempo 如何帮助跨学科团队快速排除不可行的规格，并确定更有前景的研究方向。 

---
# KernelBench: Can LLMs Write Efficient GPU Kernels? 

**Title (ZH)**: KernelBench: 大型语言模型能够编写高效的GPU内核吗？ 

**Authors**: Anne Ouyang, Simon Guo, Simran Arora, Alex L. Zhang, William Hu, Christopher Ré, Azalia Mirhoseini  

**Link**: [PDF](https://arxiv.org/pdf/2502.10517)  

**Abstract**: Efficient GPU kernels are crucial for building performant machine learning architectures, but writing them is a time-consuming challenge that requires significant expertise; therefore, we explore using language models (LMs) to automate kernel generation. We introduce KernelBench, an open-source framework for evaluating LMs' ability to write fast and correct kernels on a suite of 250 carefully selected PyTorch ML workloads. KernelBench represents a real-world engineering environment and making progress on the introduced benchmark directly translates to faster practical kernels. We introduce a new evaluation metric fast_p, which measures the percentage of generated kernels that are functionally correct and offer a speedup greater than an adjustable threshold p over baseline. Our experiments across various state-of-the-art models and test-time methods show that frontier reasoning models perform the best out of the box but still fall short overall, matching the PyTorch baseline in less than 20% of the cases. While we show that results can improve by leveraging execution and profiling feedback during iterative refinement, KernelBench remains a challenging benchmark, with its difficulty increasing as we raise speedup threshold p. 

**Abstract (ZH)**: 高效的GPU内核对于构建高性能的机器学习架构至关重要，但编写这些内核是一个耗时且需要高度专业技能的挑战；因此，我们探讨了使用语言模型（LMs）来自动生成内核的可能性。我们介绍了KernelBench，这是一个开放源代码框架，用于评估LMs在一系列250个精心挑选的PyTorch机器学习工作负载上编写快速且正确的内核的能力。KernelBench代表了一个实际的工程环境，而在此引入的基准测试上取得的进步可以直接转化为更快的实际内核。我们引入了一个新的评估指标fast_p，该指标衡量生成内核中功能上正确且相比基线提供加速比阈值p更大的百分比。我们在各种最先进的模型和测试时方法下进行的实验表明，前沿推理模型开箱即用时表现最佳，但仍总体上表现不佳，在不到20%的情况下可与PyTorch基线匹敌。虽然我们展示了通过在迭代优化过程中利用执行和分析反馈来提高结果的可能性，但KernelBench仍然是一个具有挑战性的基准，随着加速比阈值p的提高，其难度也随之增加。 

---
# Hallucinations and Truth: A Comprehensive Accuracy Evaluation of RAG, LoRA and DoRA 

**Title (ZH)**: 幻觉与现实：RAG、LoRA和DoRA的综合准确性评估 

**Authors**: Mohammad Baqar, Rajat Khanda  

**Link**: [PDF](https://arxiv.org/pdf/2502.10497)  

**Abstract**: Recent advancements in Generative AI have significantly improved the efficiency and adaptability of natural language processing (NLP) systems, particularly through Retrieval-Augmented Generation (RAG), Low-Rank Adaptation (LoRA), and Weight-Decomposed Low-Rank Adaptation (DoRA). RAG integrates external knowledge to enhance factual consistency in generative outputs, while LoRA enables parameter-efficient fine-tuning of large language models (LLMs). DoRA further refines this process by optimizing fine-tuning through adaptive parameter ranking and domain-aware weight adjustments, improving learning efficiency while maintaining inference performance.
This paper presents a large-scale empirical evaluation of RAG, LoRA, and DoRA, with model fine-tuning and generation performance assessed on 20,000 FAQ-based queries, while the knowledge base spans 400,000 entries. The study analyzes key performance metrics such as accuracy, relevance, and inference latency. Experimental results demonstrate that DoRA achieves the highest accuracy (90.1%), relevance score (0.88), and lowest latency (110 ms per query), outperforming both LoRA and RAG in real-world, domain-specific generative AI applications.
Furthermore, this study examines the trade-offs between fine-tuning efficiency, computational cost, and real-time adaptability across different models. Findings highlight RAG's effectiveness in knowledge grounding, LoRA's cost-efficient domain adaptation, and DoRA's ability to balance fine-tuning efficiency with model precision. These insights provide practical guidance for deploying AI-driven generative systems in accuracy-critical domains such as healthcare, finance, and legal services, ensuring scalability, reliability, and optimal performance in dynamic environments. 

**Abstract (ZH)**: 近年来，生成型AI的最新进展显著提高了自然语言处理（NLP）系统的效率和适应性，特别是在检索增强生成（RAG）、低秩适应（LoRA）和分解低秩适应（DoRA）等方面。RAG通过整合外部知识来提升生成输出的事实一致性，而LoRA则实现对大规模语言模型（LLM）的有效微调。DoRA在此基础上进一步通过自适应参数排名和领域感知权重调整优化微调过程，既提高学习效率又保持推理性能。

本文对RAG、LoRA和DoRA进行了大规模实证评估，在20,000个FAQ基础查询上对模型的微调和生成性能进行了评估，知识库覆盖400,000条条目。研究分析了关键性能指标，包括准确率、相关性和推理延迟。实验证明，DoRA在准确率（90.1%）、相关性评分（0.88）及最小延迟（每查询110毫秒）方面表现最佳，超越了LoRA和RAG，在实际领域的生成型AI应用中表现出色。

此外，本研究还探讨了不同模型在微调效率、计算成本和实时适应性之间的权衡。研究结果表明，RAG在知识接地方面表现出色，LoRA具备成本效益的领域适应性，而DoRA则能够平衡微调效率与模型精度。这些发现为在医学、金融和法律服务等关键精确领域部署基于AI的生成系统提供了实用指导，确保了系统的可扩展性、可靠性和在动态环境中的最优性能。 

---
# SWA-LDM: Toward Stealthy Watermarks for Latent Diffusion Models 

**Title (ZH)**: SWA-LDM：面向潜在扩散模型的隐形水印方法 

**Authors**: Zhonghao Yang, Linye Lyu, Xuanhang Chang, Daojing He, YU LI  

**Link**: [PDF](https://arxiv.org/pdf/2502.10495)  

**Abstract**: In the rapidly evolving landscape of image generation, Latent Diffusion Models (LDMs) have emerged as powerful tools, enabling the creation of highly realistic images. However, this advancement raises significant concerns regarding copyright infringement and the potential misuse of generated content. Current watermarking techniques employed in LDMs often embed constant signals to the generated images that compromise their stealthiness, making them vulnerable to detection by malicious attackers. In this paper, we introduce SWA-LDM, a novel approach that enhances watermarking by randomizing the embedding process, effectively eliminating detectable patterns while preserving image quality and robustness. Our proposed watermark presence attack reveals the inherent vulnerabilities of existing latent-based watermarking methods, demonstrating how easily these can be exposed. Through comprehensive experiments, we validate that SWA-LDM not only fortifies watermark stealthiness but also maintains competitive performance in watermark robustness and visual fidelity. This work represents a pivotal step towards securing LDM-generated images against unauthorized use, ensuring both copyright protection and content integrity in an era where digital image authenticity is paramount. 

**Abstract (ZH)**: 在图像生成快速演变的背景下，潜在扩散模型（LDMs）已成为强大的工具，能够生成高度逼真图像。然而，这一进展也引发了关于版权侵权和生成内容潜在滥用的重大关切。当前用于LDMs的水印技术往往在生成的图像中嵌入恒定信号，这破坏了其隐蔽性，使其容易受到恶意攻击者的检测。在本文中，我们提出了SWA-LDM，这是一种新颖的方法，通过随机化嵌入过程来增强水印技术，有效消除了检测到的模式，同时保持了图像质量和鲁棒性。我们提出的水印存在攻击揭示了现有基于潜在空间的水印方法的内在漏洞，展示了这些方法如何容易被暴露。通过对全面的实验验证，我们表明SWA-LDM不仅增强了水印的隐蔽性，还保持了在水印鲁棒性和视觉保真度方面的竞争力。这项工作代表了确保LDM生成图像免遭未经授权使用的关键步骤，确保在数字图像真实性的时代版权保护和内容完整性。 

---
# F-StrIPE: Fast Structure-Informed Positional Encoding for Symbolic Music Generation 

**Title (ZH)**: F-StrIPE：面向符号音乐生成的快速结构引导位置编码 

**Authors**: Manvi Agarwal, Changhong Wang, Gael Richard  

**Link**: [PDF](https://arxiv.org/pdf/2502.10491)  

**Abstract**: While music remains a challenging domain for generative models like Transformers, recent progress has been made by exploiting suitable musically-informed priors. One technique to leverage information about musical structure in Transformers is inserting such knowledge into the positional encoding (PE) module. However, Transformers carry a quadratic cost in sequence length. In this paper, we propose F-StrIPE, a structure-informed PE scheme that works in linear complexity. Using existing kernel approximation techniques based on random features, we show that F-StrIPE is a generalization of Stochastic Positional Encoding (SPE). We illustrate the empirical merits of F-StrIPE using melody harmonization for symbolic music. 

**Abstract (ZH)**: 尽管音乐仍然是生成模型如变换器所面临的挑战性领域，但通过利用适当的音乐先验知识，近期已经取得了一些进展。一种在变换器中利用音乐结构信息的技术是在位置编码（PE）模块中嵌入此类知识。然而，变换器的序列长度具有二次复杂度的成本。在此论文中，我们提出了一种名为F-StrIPE的结构导向位置编码方案，其复杂度为线性。利用基于随机特征的现有核逼近技术，我们证明F-StrIPE可以看作是随机位置编码（SPE）的一种推广。我们通过符号音乐的旋律和声化展示了F-StrIPE的经验优势。 

---
# A Robust Attack: Displacement Backdoor Attack 

**Title (ZH)**: 一种稳健的攻击：位移后门攻击 

**Authors**: Yong Li, Han Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.10490)  

**Abstract**: As artificial intelligence becomes more prevalent in our lives, people are enjoying the convenience it brings, but they are also facing hidden threats, such as data poisoning and ad- versarial attacks. These threats can have disastrous consequences for the application of artificial intelligence, especially for some applications that take effect immediately, such as autonomous driving and medical fields. Among these threats, backdoor attacks have left a deep impression on people with their concealment and simple deployment, making them a threat that cannot be ignored, however, in the process of deploying the backdoor model, the backdoor attack often has some reasons that make it unsatisfactory in real-world applications, such as jitter and brightness changes. Based on this, we propose a highly robust backdoor attack that shifts the target sample and combines it with itself to form a backdoor sample, the Displacement Backdoor Attack(DBA). Experimental results show that the DBA attack can resist data augmentation that simulates real-world differences, such as rotation and cropping. 

**Abstract (ZH)**: 随着人工智能在我们生活中的普及，人们享受着它带来的便利，同时也面临着隐藏的威胁，如数据中毒和对抗攻击。这些威胁可能对人工智能的应用造成灾难性的影响，尤其是在那些需要立即生效的应用领域，如自动驾驶和医疗领域。在这其中，后门攻击因其实现的隐蔽性和简便性给人们留下了深刻的印象，使其成为一个不容忽视的威胁。然而，在部署后门模型的过程中，后门攻击往往有一些原因使其在实际应用中不够满意，比如抖动和亮度变化。基于此，我们提出了一种高度鲁棒的后门攻击方法，该方法通过将目标样本进行位移并与其自身结合，形成后门样本，称为位移后门攻击（Displacement Backdoor Attack, DBA）。实验结果表明，DBA攻击能够抵御模拟现实世界差异的数据增强操作，如旋转和裁剪。 

---
# LiveVal: Time-aware Data Valuation via Adaptive Reference Points 

**Title (ZH)**: LiveVal：通过自适应参考点进行的时间感知数据估值 

**Authors**: Jie Xu, Zihan Wu, Cong Wang, Xiaohua Jia  

**Link**: [PDF](https://arxiv.org/pdf/2502.10489)  

**Abstract**: Time-aware data valuation enhances training efficiency and model robustness, as early detection of harmful samples could prevent months of wasted computation. However, existing methods rely on model retraining or convergence assumptions or fail to capture long-term training dynamics.
We propose LiveVal, an efficient time-aware data valuation method with three key designs:
1) seamless integration with SGD training for efficient data contribution monitoring; 2) reference-based valuation with normalization for reliable benchmark establishment; and 3) adaptive reference point selection for real-time updating with optimized memory usage.
We establish theoretical guarantees for LiveVal's stability and prove that its valuations are bounded and directionally aligned with optimization progress. Extensive experiments demonstrate that LiveVal provides efficient data valuation across different modalities and model scales, achieving 180 speedup over traditional methods while maintaining robust detection performance. 

**Abstract (ZH)**: 时间感知的数据估值可以增强训练效率和模型稳健性，因为它能够早期检测有害样本，从而避免数月的无效计算。然而，现有的方法依赖于重新训练模型或收敛假设，或者无法捕捉长期的训练动态。

为此，我们提出LiveVal——一种高效的时间感知数据估值方法，并包含三个关键设计：
1）无缝集成到SGD训练中，以高效监测数据贡献；
2）基于参考的估值方法，并通过归一化建立可靠的基准；
3）自适应选择参考点，在优化内存使用优化的情况下实现实时更新。

我们为LiveVal的稳定性建立了理论保证，证明其估值是有界且方向上与优化进度对齐的。大量实验表明，LiveVal在不同模态和模型规模下提供高效的数据估值，与传统方法相比可实现高达180倍的加速，同时保持稳健的检测性能。 

---
# Fast Proxies for LLM Robustness Evaluation 

**Title (ZH)**: 快速代理用于大规模语言模型鲁棒性评估 

**Authors**: Tim Beyer, Jan Schuchardt, Leo Schwinn, Stephan Günnemann  

**Link**: [PDF](https://arxiv.org/pdf/2502.10487)  

**Abstract**: Evaluating the robustness of LLMs to adversarial attacks is crucial for safe deployment, yet current red-teaming methods are often prohibitively expensive. We compare the ability of fast proxy metrics to predict the real-world robustness of an LLM against a simulated attacker ensemble. This allows us to estimate a model's robustness to computationally expensive attacks without requiring runs of the attacks themselves. Specifically, we consider gradient-descent-based embedding-space attacks, prefilling attacks, and direct prompting. Even though direct prompting in particular does not achieve high ASR, we find that it and embedding-space attacks can predict attack success rates well, achieving $r_p=0.87$ (linear) and $r_s=0.94$ (Spearman rank) correlations with the full attack ensemble while reducing computational cost by three orders of magnitude. 

**Abstract (ZH)**: 评估大型语言模型（LLMs）在对抗性攻击下的鲁棒性对于安全部署至关重要，但现有的红队方法往往成本过高。我们比较了快速代理指标预测LLM在模拟攻击者集合下的实际鲁棒性的能力。这使得我们能够在不运行攻击本身的情况下估计模型对计算成本高昂的攻击的鲁棒性。具体来说，我们考虑了基于梯度下降的嵌入空间攻击、预填充攻击和直接提示。尽管直接提示尤其未能实现高误报率（ASR），但我们发现它和嵌入空间攻击能够很好地预测攻击的成功率， Achieving $r_p=0.87$（线性相关系数）和 $r_s=0.94$（斯皮尔曼秩相关系数）的关联性，同时将计算成本降低了三个数量级。 

---
# VLM-Guard: Safeguarding Vision-Language Models via Fulfilling Safety Alignment Gap 

**Title (ZH)**: VLM-Guard：通过填补安全对齐缺口来保障视觉语言模型的安全性 

**Authors**: Qin Liu, Fei Wang, Chaowei Xiao, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.10486)  

**Abstract**: The emergence of vision language models (VLMs) comes with increased safety concerns, as the incorporation of multiple modalities heightens vulnerability to attacks. Although VLMs can be built upon LLMs that have textual safety alignment, it is easily undermined when the vision modality is integrated. We attribute this safety challenge to the modality gap, a separation of image and text in the shared representation space, which blurs the distinction between harmful and harmless queries that is evident in LLMs but weakened in VLMs. To avoid safety decay and fulfill the safety alignment gap, we propose VLM-Guard, an inference-time intervention strategy that leverages the LLM component of a VLM as supervision for the safety alignment of the VLM. VLM-Guard projects the representations of VLM into the subspace that is orthogonal to the safety steering direction that is extracted from the safety-aligned LLM. Experimental results on three malicious instruction settings show the effectiveness of VLM-Guard in safeguarding VLM and fulfilling the safety alignment gap between VLM and its LLM component. 

**Abstract (ZH)**: 视觉语言模型（VLMs）的出现带来了安全性方面的担忧，因为多种模态的整合增加了模型对攻击的脆弱性。虽然VLMs可以在具有文本安全性对齐的大型语言模型（LLMs）的基础上构建，但在引入视觉模态后，这种安全性对齐容易受到破坏。我们将这一安全性挑战归因于模态差距，即在共享表示空间中图像与文本之间的分离，这模糊了在LLMs中清晰区分有害和无害查询的能力，而在VLMs中这种区分能力则减弱了。为了防止安全性衰退并弥补安全性对齐缺口，我们提出了VLM-Guard，这是一种在推断时使用的干预策略，利用VLM中的LLMs组件作为监督，实现VLM的安全性对齐。VLM-Guard将VLM的表示投影到从安全对齐的LLM中提取的安全导向方向的正交子空间中。在三个恶意指令设置上的实验结果表明，VLM-Guard在保护VLM并弥补VLM与其中LLMs组件的安全性对齐缺口方面是有效的。 

---
# Forecasting time series with constraints 

**Title (ZH)**: 具有约束条件的时间序列预测 

**Authors**: Nathan Doumèche, Francis Bach, Éloi Bedek, Gérard Biau, Claire Boyer, Yannig Goude  

**Link**: [PDF](https://arxiv.org/pdf/2502.10485)  

**Abstract**: Time series forecasting presents unique challenges that limit the effectiveness of traditional machine learning algorithms. To address these limitations, various approaches have incorporated linear constraints into learning algorithms, such as generalized additive models and hierarchical forecasting. In this paper, we propose a unified framework for integrating and combining linear constraints in time series forecasting. Within this framework, we show that the exact minimizer of the constrained empirical risk can be computed efficiently using linear algebra alone. This approach allows for highly scalable implementations optimized for GPUs. We validate the proposed methodology through extensive benchmarking on real-world tasks, including electricity demand forecasting and tourism forecasting, achieving state-of-the-art performance. 

**Abstract (ZH)**: 时间序列预测面临着独特的挑战，限制了传统机器学习算法的有效性。为了解决这些限制，各种方法已将线性约束引入学习算法中，例如广义加性模型和层次预测。在本文中，我们提出了一种统一框架，用于整合和组合时间序列预测中的线性约束。在这一框架内，我们展示了可以通过仅使用线性代数高效地计算约束经验风险的精确最小值。该方法允许针对GPU进行高度可伸缩的实现。我们通过在实际任务中的广泛基准测试验证了所提出的方法，包括电力需求预测和旅游业预测，实现了最先进的性能。 

---
# X-SG$^2$S: Safe and Generalizable Gaussian Splatting with X-dimensional Watermarks 

**Title (ZH)**: X-SG$^2$S: 安全且通用的高维水印高斯散射方法 

**Authors**: Zihang Cheng, Huiping Zhuang, Chun Li, Xin Meng, Ming Li, Fei Richard Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10475)  

**Abstract**: 3D Gaussian Splatting (3DGS) has been widely used in 3D reconstruction and 3D generation. Training to get a 3DGS scene often takes a lot of time and resources and even valuable inspiration. The increasing amount of 3DGS digital asset have brought great challenges to the copyright protection. However, it still lacks profound exploration targeted at 3DGS. In this paper, we propose a new framework X-SG$^2$S which can simultaneously watermark 1 to 3D messages while keeping the original 3DGS scene almost unchanged. Generally, we have a X-SG$^2$S injector for adding multi-modal messages simultaneously and an extractor for extract them. Specifically, we first split the watermarks into message patches in a fixed manner and sort the 3DGS points. A self-adaption gate is used to pick out suitable location for watermarking. Then use a XD(multi-dimension)-injection heads to add multi-modal messages into sorted 3DGS points. A learnable gate can recognize the location with extra messages and XD-extraction heads can restore hidden messages from the location recommended by the learnable gate. Extensive experiments demonstrated that the proposed X-SG$^2$S can effectively conceal multi modal messages without changing pretrained 3DGS pipeline or the original form of 3DGS parameters. Meanwhile, with simple and efficient model structure and high practicality, X-SG$^2$S still shows good performance in hiding and extracting multi-modal inner structured or unstructured messages. X-SG$^2$S is the first to unify 1 to 3D watermarking model for 3DGS and the first framework to add multi-modal watermarks simultaneous in one 3DGS which pave the wave for later researches. 

**Abstract (ZH)**: 3D 高斯体绘制（3DGS）在三维重建和三维生成中有广泛应用。训练一个3DGS场景往往需要大量的时间和资源，甚至会带来宝贵的灵感。随着3DGS数字资产数量的不断增加，版权保护面临着巨大挑战。然而，针对3DGS的深刻探索仍然相对缺乏。在本文中，我们提出了一种新的框架X-SG$^2$S，该框架能够同时嵌入1到3D的消息，同时几乎保持原始3DGS场景不变。总体而言，我们有一个X-SG$^2$S嵌入器用于同时添加多模态消息，还有一个提取器用于提取这些消息。具体而言，我们首先以固定的方式将水印切成消息片，对3DGS点进行排序。使用自适应门来挑选合适的水印位置。然后使用多维XD嵌入头部将多模态消息添加到排序后的3DGS点中。可学习的门可以识别带有额外消息的位置，XD提取头部可以从可学习的门推荐的位置恢复隐藏的消息。广泛的经验实验证明，所提出的X-SG$^2$S能够在不改变预训练的3DGS管道或原始3DGS参数形式的情况下有效隐藏多模态消息。同时，凭借其简单高效的模型结构和高度的实用性，X-SG$^2$S在隐藏和提取多模态有序或无序消息方面仍然表现出良好的性能。X-SG$^2$S是第一个统一1到3D消息嵌入模型的框架，并且是第一个在单一3DGS中同时添加多模态水印的框架，为后续研究奠定了基础。 

---
# MetaDE: Evolving Differential Evolution by Differential Evolution 

**Title (ZH)**: MetaDE：通过差分进化演化差分进化 

**Authors**: Minyang Chen, Chenchen Feng, and Ran Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2502.10470)  

**Abstract**: As a cornerstone in the Evolutionary Computation (EC) domain, Differential Evolution (DE) is known for its simplicity and effectiveness in handling challenging black-box optimization problems. While the advantages of DE are well-recognized, achieving peak performance heavily depends on its hyperparameters such as the mutation factor, crossover probability, and the selection of specific DE strategies. Traditional approaches to this hyperparameter dilemma have leaned towards parameter tuning or adaptive mechanisms. However, identifying the optimal settings tailored for specific problems remains a persistent challenge. In response, we introduce MetaDE, an approach that evolves DE's intrinsic hyperparameters and strategies using DE itself at a meta-level. A pivotal aspect of MetaDE is a specialized parameterization technique, which endows it with the capability to dynamically modify DE's parameters and strategies throughout the evolutionary process. To augment computational efficiency, MetaDE incorporates a design that leverages parallel processing through a GPU-accelerated computing framework. Within such a framework, DE is not just a solver but also an optimizer for its own configurations, thus streamlining the process of hyperparameter optimization and problem-solving into a cohesive and automated workflow. Extensive evaluations on the CEC2022 benchmark suite demonstrate MetaDE's promising performance. Moreover, when applied to robot control via evolutionary reinforcement learning, MetaDE also demonstrates promising performance. The source code of MetaDE is publicly accessible at: this https URL. 

**Abstract (ZH)**: 作为进化计算（EC）领域的一个基石，差分进化（DE）因其在处理具有挑战性的黑盒优化问题时的简便性和有效性而闻名。尽管DE的优势被广泛认可，但要实现最佳性能，很大程度上依赖于其超参数，如变异因子、交叉概率以及特定DE策略的选择。传统上，这一超参数问题的解决方案主要集中在参数调整或自适应机制上。然而，针对特定问题确定最优设置仍然是一项长期存在的挑战。为应对这一挑战，我们提出了MetaDE这一方法，该方法通过将DE本身用作元级进化工具来演化DE的固有超参数和策略。MetaDE的一个关键方面是专门的参数化技术，使其能够在进化过程中动态调整DE的参数和策略。为提高计算效率，MetaDE采用了通过GPU加速计算框架利用并行处理的设计。在这种框架下，DE不仅是一个求解器，也是一个优化其自身配置的优化器，从而将超参数优化和问题解决过程中的各个环节整合为统一的自动化工作流。在CEC2022基准测试套件上的广泛评估中，MetaDE展示了其令人鼓舞的性能。此外，在通过进化强化学习进行机器人控制时，MetaDE同样表现出色。MetaDE的源代码可在以下链接访问：this https URL。 

---
# YNote: A Novel Music Notation for Fine-Tuning LLMs in Music Generation 

**Title (ZH)**: YNote：一种用于音乐生成中调优LLMs的新颖音乐标注方法 

**Authors**: Shao-Chien Lu, Chen-Chen Yeh, Hui-Lin Cho, Chun-Chieh Hsu, Tsai-Ling Hsu, Cheng-Han Wu, Timothy K. Shih, Yu-Cheng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.10467)  

**Abstract**: The field of music generation using Large Language Models (LLMs) is evolving rapidly, yet existing music notation systems, such as MIDI, ABC Notation, and MusicXML, remain too complex for effective fine-tuning of LLMs. These formats are difficult for both machines and humans to interpret due to their variability and intricate structure. To address these challenges, we introduce YNote, a simplified music notation system that uses only four characters to represent a note and its pitch. YNote's fixed format ensures consistency, making it easy to read and more suitable for fine-tuning LLMs. In our experiments, we fine-tuned GPT-2 (124M) on a YNote-encoded dataset and achieved BLEU and ROUGE scores of 0.883 and 0.766, respectively. With just two notes as prompts, the model was able to generate coherent and stylistically relevant music. We believe YNote offers a practical alternative to existing music notations for machine learning applications and has the potential to significantly enhance the quality of music generation using LLMs. 

**Abstract (ZH)**: 使用大型语言模型（LLMs）生成音乐的领域正在快速发展，但现有的乐谱表示系统，如MIDI、ABC 符号和MusicXML，仍然过于复杂，不适合对LLMs进行有效的微调。这些格式由于其多样性和复杂的结构，对于机器和人类来说都难以解读。为了解决这些问题，我们引入了YNote，这是一种简化了的音乐表示系统，仅使用四个字符来表示音符及其音高。YNote的固定格式保证了其一致性，使其易于阅读，并且更适用于对LLMs进行微调。在我们的实验中，我们对经过YNote编码的数据集进行了GPT-2（124M）的微调，并分别获得了BLEU评分为0.883和ROUGE评分为0.766。仅用两个音符作为提示，模型就能够生成连贯且风格相关度高的音乐。我们相信YNote为机器学习应用提供了一种实用的替代方案，并有可能显著提高使用LLMs生成音乐的质量。 

---
# From Layers to States: A State Space Model Perspective to Deep Neural Network Layer Dynamics 

**Title (ZH)**: 从层到状态：基于深神经网络层动态的状态空间模型视角 

**Authors**: Qinshuo Liu, Weiqin Zhao, Wei Huang, Yanwen Fang, Lequan Yu, Guodong Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.10463)  

**Abstract**: The depth of neural networks is a critical factor for their capability, with deeper models often demonstrating superior performance. Motivated by this, significant efforts have been made to enhance layer aggregation - reusing information from previous layers to better extract features at the current layer, to improve the representational power of deep neural networks. However, previous works have primarily addressed this problem from a discrete-state perspective which is not suitable as the number of network layers grows. This paper novelly treats the outputs from layers as states of a continuous process and considers leveraging the state space model (SSM) to design the aggregation of layers in very deep neural networks. Moreover, inspired by its advancements in modeling long sequences, the Selective State Space Models (S6) is employed to design a new module called Selective State Space Model Layer Aggregation (S6LA). This module aims to combine traditional CNN or transformer architectures within a sequential framework, enhancing the representational capabilities of state-of-the-art vision networks. Extensive experiments show that S6LA delivers substantial improvements in both image classification and detection tasks, highlighting the potential of integrating SSMs with contemporary deep learning techniques. 

**Abstract (ZH)**: 神经网络的深度是其能力的关键因素，更深的模型往往表现出更优的性能。鉴于这一观察，已有大量研究致力于增强层聚合——即通过重用前一层的信息来更好地提取当前层的特征，以提高深层神经网络的表现力。然而，之前的研究主要从离散状态的角度出发，这在网络层数增加时并不适用。本文创新地将各层的输出视为连续过程中的一种状态，并考虑利用状态空间模型（SSM）来设计深层神经网络中各层的聚合方法。此外，受到其在建模长序列方面的进展启发，我们采用了选择性状态空间模型（S6）来设计一种新的模块——选择性状态空间模型层聚合（S6LA）。该模块旨在在序列框架中结合传统CNN或Transformer架构，增强当今最先进的视觉网络的表现力。大量实验表明，S6LA在图像分类和检测任务中均取得了显著的性能提升，凸显了将SSM与当前深度学习技术整合的潜力。 

---
# LLM4GNAS: A Large Language Model Based Toolkit for Graph Neural Architecture Search 

**Title (ZH)**: LLM4GNAS：一种基于大型语言模型的图神经架构搜索工具包 

**Authors**: Yang Gao, Hong Yang, Yizhi Chen, Junxian Wu, Peng Zhang, Haishuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10459)  

**Abstract**: Graph Neural Architecture Search (GNAS) facilitates the automatic design of Graph Neural Networks (GNNs) tailored to specific downstream graph learning tasks. However, existing GNAS approaches often require manual adaptation to new graph search spaces, necessitating substantial code optimization and domain-specific knowledge. To address this challenge, we present LLM4GNAS, a toolkit for GNAS that leverages the generative capabilities of Large Language Models (LLMs). LLM4GNAS includes an algorithm library for graph neural architecture search algorithms based on LLMs, enabling the adaptation of GNAS methods to new search spaces through the modification of LLM prompts. This approach reduces the need for manual intervention in algorithm adaptation and code modification. The LLM4GNAS toolkit is extensible and robust, incorporating LLM-enhanced graph feature engineering, LLM-enhanced graph neural architecture search, and LLM-enhanced hyperparameter optimization. Experimental results indicate that LLM4GNAS outperforms existing GNAS methods on tasks involving both homogeneous and heterogeneous graphs. 

**Abstract (ZH)**: Graph神经架构搜索（GNAS）有助于自动设计特定下游图学习任务的图神经网络（GNNs）。然而，现有的GNAS方法通常需要手动适应新的图搜索空间，这需要大量的代码优化和领域特定知识。为了解决这一挑战，我们提出了LLM4GNAS，这是一个利用大型语言模型（LLMs）生成能力的GNAS工具包。LLM4GNAS包含基于LLMs的图神经架构搜索算法库，通过修改LLMs提示来使GNAS方法适应新的搜索空间。这种方法减少了算法适应和代码修改的手动干预需求。LLM4GNAS工具包是可扩展且稳定的，它结合了LLM增强的图特征工程、LLM增强的图神经架构搜索以及LLM增强的超参数优化。实验结果表明，LLM4GNAS在涉及同构和异构图的任务中优于现有的GNAS方法。 

---
# I Think, Therefore I Diffuse: Enabling Multimodal In-Context Reasoning in Diffusion Models 

**Title (ZH)**: 因此我思考，因此我扩散：在扩散模型中实现多模态上下文推理的能力 

**Authors**: Zhenxing Mi, Kuan-Chieh Wang, Guocheng Qian, Hanrong Ye, Runtao Liu, Sergey Tulyakov, Kfir Aberman, Dan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10458)  

**Abstract**: This paper presents ThinkDiff, a novel alignment paradigm that empowers text-to-image diffusion models with multimodal in-context understanding and reasoning capabilities by integrating the strengths of vision-language models (VLMs). Existing multimodal diffusion finetuning methods largely focus on pixel-level reconstruction rather than in-context reasoning, and are constrained by the complexity and limited availability of reasoning-based datasets. ThinkDiff addresses these challenges by leveraging vision-language training as a proxy task, aligning VLMs with the decoder of an encoder-decoder large language model (LLM) instead of a diffusion decoder. This proxy task builds on the observation that the $\textbf{LLM decoder}$ shares the same input feature space with $\textbf{diffusion decoders}$ that use the corresponding $\textbf{LLM encoder}$ for prompt embedding. As a result, aligning VLMs with diffusion decoders can be simplified through alignment with the LLM decoder. Without complex training and datasets, ThinkDiff effectively unleashes understanding, reasoning, and composing capabilities in diffusion models. Experiments demonstrate that ThinkDiff significantly improves accuracy from 19.2% to 46.3% on the challenging CoBSAT benchmark for multimodal in-context reasoning generation, with only 5 hours of training on 4 A100 GPUs. Additionally, ThinkDiff demonstrates exceptional performance in composing multiple images and texts into logically coherent images. Project page: this https URL. 

**Abstract (ZH)**: 本文介绍了一种新的对齐范式——ThinkDiff，它通过结合视觉语言模型（VLM）的优势，赋予文本到图像扩散模型多模态的上下文理解和推理能力。现有的多模态扩散微调方法主要集中在像素级重建上，而忽视了上下文推理，且受限于推理数据集的复杂性和稀缺性。ThinkDiff 通过利用视觉语言训练作为代理任务来应对这些挑战，将视觉语言模型与编码器-解码器大型语言模型（LLM）的解码器对齐，而不是与扩散解码器对齐。这种代理任务基于观察到的：使用相同 LLM 编码器进行提示嵌入的扩散解码器与 LLM 解码器具有相同的输入特征空间。因此，通过与 LLM 解码器对齐视觉语言模型，可以简化扩散解码器的对齐过程。在无需复杂训练和数据集的情况下，ThinkDiff 有效地提升了扩散模型的理解、推理和组合能力。实验表明，在具有挑战性的 CoBSAT 多模态上下文推理生成基准测试中，ThinkDiff 的准确率从 19.2% 提高到了 46.3%，仅需在 4 块 A100 GPU 上进行 5 小时的训练。此外，ThinkDiff 在将多个图像和文本组合成逻辑上连贯的图像方面表现出色。项目页面：[请填写具体链接]。 

---
# One Example Shown, Many Concepts Known! Counterexample-Driven Conceptual Reasoning in Mathematical LLMs 

**Title (ZH)**: 一个实例展示，多个概念洞悉！基于反例的概念性推理在数学大语言模型中的应用 

**Authors**: Yinghui Li, Jiayi Kuang, Haojing Huang, Zhikun Xu, Xinnian Liang, Yi Yu, Wenlian Lu, Yangning Li, Xiaoyu Tan, Chao Qu, Ying Shen, Hai-Tao Zheng, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.10454)  

**Abstract**: Leveraging mathematical Large Language Models (LLMs) for proof generation is a fundamental topic in LLMs research. We argue that the ability of current LLMs to prove statements largely depends on whether they have encountered the relevant proof process during training. This reliance limits their deeper understanding of mathematical theorems and related concepts. Inspired by the pedagogical method of "proof by counterexamples" commonly used in human mathematics education, our work aims to enhance LLMs' ability to conduct mathematical reasoning and proof through counterexamples. Specifically, we manually create a high-quality, university-level mathematical benchmark, CounterMATH, which requires LLMs to prove mathematical statements by providing counterexamples, thereby assessing their grasp of mathematical concepts. Additionally, we develop a data engineering framework to automatically obtain training data for further model improvement. Extensive experiments and detailed analyses demonstrate that CounterMATH is challenging, indicating that LLMs, such as OpenAI o1, have insufficient counterexample-driven proof capabilities. Moreover, our exploration into model training reveals that strengthening LLMs' counterexample-driven conceptual reasoning abilities is crucial for improving their overall mathematical capabilities. We believe that our work offers new perspectives on the community of mathematical LLMs. 

**Abstract (ZH)**: 利用数学大语言模型（LLMs）生成证明是LLMs研究中的一个基础课题。我们argue认为，当前LLMs证明命题的能力很大程度上取决于它们在训练过程中是否遇到了相关的证明过程。这种依赖限制了它们对数学定理及相关概念的深入理解。借鉴人类数学教育中常用的“反例证明”教学法，我们的工作旨在通过反例增强LLMs的数学推理和证明能力。具体来说，我们手动创建了一个高质量的大学水平数学基准CounterMATH，要求LLMs通过提供反例来证明数学命题，从而评估其对数学概念的理解。此外，我们还开发了一个数据工程框架，以自动获取训练数据，进一步改进模型。大量实验和详细分析表明，CounterMATH具有挑战性，表明像OpenAI的o1这样的LLMs缺乏足够的反例驱动证明能力。此外，我们对模型训练的探索显示，强化LLMs的反例驱动概念推理能力对于提高其整体数学能力至关重要。我们认为，我们的工作为数学LLMs的研究社区提供了新的视角。 

---
# Linking Cryptoasset Attribution Tags to Knowledge Graph Entities: An LLM-based Approach 

**Title (ZH)**: 基于大型语言模型的方法：将加密资产属性标签链接到知识图谱实体 

**Authors**: Régnier Avice, Bernhard Haslhofer, Zhidong Li, Jianlong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.10453)  

**Abstract**: Attribution tags form the foundation of modern cryptoasset forensics. However, inconsistent or incorrect tags can mislead investigations and even result in false accusations. To address this issue, we propose a novel computational method based on Large Language Models (LLMs) to link attribution tags with well-defined knowledge graph concepts. We implemented this method in an end-to-end pipeline and conducted experiments showing that our approach outperforms baseline methods by up to 37.4% in F1-score across three publicly available attribution tag datasets. By integrating concept filtering and blocking procedures, we generate candidate sets containing five knowledge graph entities, achieving a recall of 93% without the need for labeled data. Additionally, we demonstrate that local LLM models can achieve F1-scores of 90%, comparable to remote models which achieve 94%. We also analyze the cost-performance trade-offs of various LLMs and prompt templates, showing that selecting the most cost-effective configuration can reduce costs by 90%, with only a 1% decrease in performance. Our method not only enhances attribution tag quality but also serves as a blueprint for fostering more reliable forensic evidence. 

**Abstract (ZH)**: 现代加密资产取证的基础是归属标签。然而，不一致或错误的标签可能会误导调查，甚至导致错误指控。为了解决这一问题，我们提出了一种基于大规模语言模型（LLMs）的新型计算方法，用于将归属标签与明确的知识图谱概念关联起来。我们在此方法上实现了一个端到端的流程，并进行了实验，结果显示，在三个公开的归属标签数据集上，我们的方法在F1分数上比基线方法高出最多37.4%。通过整合概念过滤和封停程序，我们生成包含五个知识图谱实体的候选集，无需使用标注数据即可实现93%的召回率。此外，我们还证明本地LLM模型可以实现90%的F1分数，这与远程模型的94%相媲美。我们还分析了不同LLM和提示模板的成本-性能权衡，显示选择最经济有效的配置可以将成本降低90%，同时性能仅下降1%。我们的方法不仅提高了归属标签的质量，还为构建更可靠的法证证据提供了一个蓝图。 

---
# Trustworthy AI on Safety, Bias, and Privacy: A Survey 

**Title (ZH)**: 可信赖的人工智能在安全、偏见和隐私方面的综述 

**Authors**: Xingli Fang, Jianwei Li, Varun Mulchandani, Jung-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.10450)  

**Abstract**: The capabilities of artificial intelligence systems have been advancing to a great extent, but these systems still struggle with failure modes, vulnerabilities, and biases. In this paper, we study the current state of the field, and present promising insights and perspectives regarding concerns that challenge the trustworthiness of AI models. In particular, this paper investigates the issues regarding three thrusts: safety, privacy, and bias, which hurt models' trustworthiness. For safety, we discuss safety alignment in the context of large language models, preventing them from generating toxic or harmful content. For bias, we focus on spurious biases that can mislead a network. Lastly, for privacy, we cover membership inference attacks in deep neural networks. The discussions addressed in this paper reflect our own experiments and observations. 

**Abstract (ZH)**: 人工智能系统的能力在不断进步，但仍面临失败模式、漏洞和偏见等问题。在本文中，我们研究了该领域的现状，并提出了关于挑战AI模型可信度的问题的有前景的见解和视角。特别地，本文探讨了与安全性、隐私性和偏见三个重点相关的问题，这些问题是影响模型可信度的关键因素。在安全性方面，我们讨论了大型语言模型的安全对齐问题，防止其生成有害内容。在偏见方面，我们关注可能导致网络误判的虚假偏见。在隐私方面，我们探讨了深层神经网络中的成员归属推断攻击。本文讨论的内容反映了我们的实验和观察结果。 

---
# Analysis of Overparameterization in Continual Learning under a Linear Model 

**Title (ZH)**: 在线性模型下持续学习中的过度参数化分析 

**Authors**: Daniel Goldfarb, Paul Hand  

**Link**: [PDF](https://arxiv.org/pdf/2502.10442)  

**Abstract**: Autonomous machine learning systems that learn many tasks in sequence are prone to the catastrophic forgetting problem. Mathematical theory is needed in order to understand the extent of forgetting during continual learning. As a foundational step towards this goal, we study continual learning and catastrophic forgetting from a theoretical perspective in the simple setting of gradient descent with no explicit algorithmic mechanism to prevent forgetting. In this setting, we analytically demonstrate that overparameterization alone can mitigate forgetting in the context of a linear regression model. We consider a two-task setting motivated by permutation tasks, and show that as the overparameterization ratio becomes sufficiently high, a model trained on both tasks in sequence results in a low-risk estimator for the first task. As part of this work, we establish a non-asymptotic bound of the risk of a single linear regression task, which may be of independent interest to the field of double descent theory. 

**Abstract (ZH)**: sequential学习系统在学习多个任务时容易出现灾难性遗忘问题。为了理解连续学习过程中遗忘的程度，需要相应的数学理论。为此，我们从理论上研究了在没有明确机制防止遗忘的梯度下降简单设置下，连续学习和灾难性遗忘的问题。在这一设置中，我们通过分析证明，在线性回归模型的背景下，过参数化自身就能缓解遗忘。我们考虑了一个由排列任务启发的两任务设置，并证明当过参数化比例足够高时，依次训练两个任务的模型可以对第一个任务提供低风险估计。作为这项工作的组成部分，我们建立了单个线性回归任务风险的非渐近界，这一结果对双峰现象理论领域可能具有独立的研究价值。 

---
# Towards Copyright Protection for Knowledge Bases of Retrieval-augmented Language Models via Ownership Verification with Reasoning 

**Title (ZH)**: 通过所有权验证与推理实现检索增强语言模型知识库的版权保护 

**Authors**: Junfeng Guo, Yiming Li, Ruibo Chen, Yihan Wu, Chenxi Liu, Yanshuo Chen, Heng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10440)  

**Abstract**: Large language models (LLMs) are increasingly integrated into real-world applications through retrieval-augmented generation (RAG) mechanisms to supplement their responses with up-to-date and domain-specific knowledge. However, the valuable and often proprietary nature of the knowledge bases used in RAG introduces the risk of unauthorized usage by adversaries. Existing methods that can be generalized as watermarking techniques to protect these knowledge bases typically involve poisoning attacks. However, these methods require to alter the results of verification samples (\eg, generating incorrect outputs), inevitably making them susceptible to anomaly detection and even introduce new security risks. To address these challenges, we propose \name{} for `harmless' copyright protection of knowledge bases. Instead of manipulating LLM's final output, \name{} implants distinct verification behaviors in the space of chain-of-thought (CoT) reasoning, maintaining the correctness of the final answer. Our method has three main stages: (1) \textbf{Generating CoTs}: For each verification question, we generate two CoTs, including a target CoT for building watermark behaviors; (2) \textbf{Optimizing Watermark Phrases and Target CoTs}: We optimize them to minimize retrieval errors under the black-box setting of suspicious LLM, ensuring that the watermarked verification queries activate the target CoTs without being activated in non-watermarked ones; (3) \textbf{Ownership Verification}: We exploit a pairwise Wilcoxon test to statistically verify whether a suspicious LLM is augmented with the protected knowledge base by comparing its responses to watermarked and benign verification queries. Our experiments on diverse benchmarks demonstrate that \name{} effectively protects knowledge bases against unauthorized usage while preserving the integrity and performance of the RAG. 

**Abstract (ZH)**: 大型语言模型（LLMs）越来越多地通过检索增强生成（RAG）机制集成到现实世界的应用中，以通过补充其响应来使用最新的和领域特定的知识。然而，用于RAG的知识库的价值性和往往专有的特性引入了未经授权使用的风险。现有的可以归纳为水印技术的保护方法通常涉及毒化攻击。然而，这些方法需要对验证样本的结果进行篡改（例如，生成不正确的输出），不可避免地使其容易被异常检测，并且还引入了新的安全风险。为了解决这些挑战，我们提出了\name{}，以实现知识库的“无害”版权保护。与通过篡改LLM的最终输出不同，\name{}在思维链（CoT）推理的空间中植入了独特的验证行为，从而保持了最终答案的正确性。我们的方法包含三个主要阶段：（1）**生成思维链**：对于每个验证问题，我们生成两个思维链，包括一个目标思维链以构建水印行为；（2）**优化水印短语和目标思维链**：我们通过在可疑LLM的黑盒设置下最小化检索错误来优化它们，确保带水印的验证查询激活目标思维链，而不激活未带水印的验证查询中的目标思维链；（3）**所有权验证**：我们利用配对的威尔科xon秩和检验，通过比较受水印和良性验证查询影响的LLM的响应来进行统计验证，以确定是否存在受保护知识库的增强。我们的基准实验结果证明，\name{}在防止未经授权使用的同时有效保护了RAG的完整性和性能。 

---
# Crypto Miner Attack: GPU Remote Code Execution Attacks 

**Title (ZH)**: 加密挖矿攻击：GPU 远程代码执行攻击 

**Authors**: Ariel Szabo, Uzy Hadad  

**Link**: [PDF](https://arxiv.org/pdf/2502.10439)  

**Abstract**: Remote Code Execution (RCE) exploits pose a significant threat to AI and ML systems, particularly in GPU-accelerated environments where the computational power of GPUs can be misused for malicious purposes. This paper focuses on RCE attacks leveraging deserialization vulnerabilities and custom layers, such as TensorFlow Lambda layers, which are often overlooked due to the complexity of monitoring GPU workloads. These vulnerabilities enable attackers to execute arbitrary code, blending malicious activity seamlessly into expected model behavior and exploiting GPUs for unauthorized tasks such as cryptocurrency mining. Unlike traditional CPU-based attacks, the parallel processing nature of GPUs and their high resource utilization make runtime detection exceptionally challenging. In this work, we provide a comprehensive examination of RCE exploits targeting GPUs, demonstrating an attack that utilizes these vulnerabilities to deploy a crypto miner on a GPU. We highlight the technical intricacies of such attacks, emphasize their potential for significant financial and computational costs, and propose strategies for mitigation. By shedding light on this underexplored attack vector, we aim to raise awareness and encourage the adoption of robust security measures in GPU-driven AI and ML systems, with an emphasis on static and model scanning as an easier way to detect exploits. 

**Abstract (ZH)**: 远程代码执行（RCE）漏洞对AI和ML系统构成了重大威胁，尤其是在GPU加速环境中，GPU的强大计算能力可能被用于恶意目的。本文着重探讨利用反序列化漏洞及自定义层（如TensorFlow Lambda层）的RCE攻击，这些漏洞往往因监控GPU工作负载的复杂性而被忽视。这些漏洞使攻击者能够执行任意代码，将恶意活动无缝融入预期模型行为，并利用GPU进行未经授权的任务，例如加密货币挖矿。与传统的基于CPU的攻击不同，GPU的并行处理特性和高资源利用率使得运行时检测变得极其困难。在本文中，我们对针对GPU的RCE攻击进行了全面分析，并展示了利用这些漏洞部署加密货币挖矿程序的攻击方法。我们强调了此类攻击的技术复杂性，指出了它们可能造成的重大经济和计算成本，并提出了缓解策略。通过揭示这一未被充分探索的攻击路径，我们旨在提高人们的认识，并鼓励在GPU驱动的AI和ML系统中采用更 robust的安全措施，特别是在静态和模型扫描方面以简化检测过程。 

---
# Injecting Universal Jailbreak Backdoors into LLMs in Minutes 

**Title (ZH)**: 在几分钟内向大规模语言模型（LLM）中注入通用越狱后门 

**Authors**: Zhuowei Chen, Qiannan Zhang, Shichao Pei  

**Link**: [PDF](https://arxiv.org/pdf/2502.10438)  

**Abstract**: Jailbreak backdoor attacks on LLMs have garnered attention for their effectiveness and stealth. However, existing methods rely on the crafting of poisoned datasets and the time-consuming process of fine-tuning. In this work, we propose JailbreakEdit, a novel jailbreak backdoor injection method that exploits model editing techniques to inject a universal jailbreak backdoor into safety-aligned LLMs with minimal intervention in minutes. JailbreakEdit integrates a multi-node target estimation to estimate the jailbreak space, thus creating shortcuts from the backdoor to this estimated jailbreak space that induce jailbreak actions. Our attack effectively shifts the models' attention by attaching strong semantics to the backdoor, enabling it to bypass internal safety mechanisms. Experimental results show that JailbreakEdit achieves a high jailbreak success rate on jailbreak prompts while preserving generation quality, and safe performance on normal queries. Our findings underscore the effectiveness, stealthiness, and explainability of JailbreakEdit, emphasizing the need for more advanced defense mechanisms in LLMs. 

**Abstract (ZH)**: 针对大型语言模型（LLMs）的监狱逃脱后门攻击因其有效性和隐蔽性而引起了广泛关注。然而，现有的方法依赖于有毒数据集的制作以及长时间的微调过程。在这个工作中，我们提出了JailbreakEdit，这是一种创新的监狱逃脱后门注入方法，利用模型编辑技术，在最少干预的情况下，仅需几分钟就能将通用的监狱逃脱后门注入安全对齐的LLMs中。JailbreakEdit 结合了多节点目标估计技术来估计监狱逃脱空间，并创造从后门到估计监狱逃脱空间的快捷方式，从而诱导执行监狱逃脱行为。该攻击通过将强烈的语义信息附加到后门上，有效地转移了模型的注意力，使其能够绕过内部的安全机制。实验结果表明，JailbreakEdit 在处理监狱逃脱提示时能够实现高成功率，同时保持生成质量，并在正常查询上保持安全性能。我们的研究结果突显了JailbreakEdit 的有效性和隐蔽性，并强调了需要在LLMs中开发更高级的防御机制。 

---
# MERGE$^3$: Efficient Evolutionary Merging on Consumer-grade GPUs 

**Title (ZH)**: MERGE\(^3\): 效率优化的消费者级GPU上的并行合并算法 

**Authors**: Tommaso Mencattini, Adrian Robert Minut, Donato Crisostomi, Andrea Santilli, Emanuele Rodolà  

**Link**: [PDF](https://arxiv.org/pdf/2502.10436)  

**Abstract**: Evolutionary model merging enables the creation of high-performing multi-task models but remains computationally prohibitive for consumer hardware. We introduce MERGE$^3$, an efficient framework that makes evolutionary merging feasible on a single GPU by reducing fitness computation costs 50$\times$ while preserving performance. MERGE$^3$ achieves this by Extracting a reduced dataset for evaluation, Estimating model abilities using Item Response Theory (IRT), and Evolving optimal merges via IRT-based performance estimators. Our method enables state-of-the-art multilingual and cross-lingual merging, transferring knowledge across languages with significantly lower computational overhead. We provide theoretical guarantees and an open-source library, democratizing high-quality model merging. 

**Abstract (ZH)**: 进化模型合并能够创建高性能的多任务模型，但对于消费者级别的硬件来说仍然计算成本高昂。我们引入了MERGE$^3$框架，该框架通过减少50倍的适应度计算成本，使得在单块GPU上实现进化合并成为可能，同时保持性能不变。MERGE$^3$通过以下方式实现这一目标：提取用于评估的减缩数据集，利用项目反应理论（IRT）估算模型能力，以及通过基于IRT的性能估算器进化最优合并。我们的方法使得最先进的跨语言和多语言合并成为可能，能够以显著更低的计算开销在不同语言间转移知识。我们提供了理论保证并公开了源代码库，使高质量模型合并更加普及。 

---
# RAMer: Reconstruction-based Adversarial Model for Multi-party Multi-modal Multi-label Emotion Recognition 

**Title (ZH)**: RAMer：基于重建的对抗模型多模态多标签情感识别 

**Authors**: Xudong Yang, Yizhang Zhu, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.10435)  

**Abstract**: Conventional multi-modal multi-label emotion recognition (MMER) from videos typically assumes full availability of visual, textual, and acoustic modalities. However, real-world multi-party settings often violate this assumption, as non-speakers frequently lack acoustic and textual inputs, leading to a significant degradation in model performance. Existing approaches also tend to unify heterogeneous modalities into a single representation, overlooking each modality's unique characteristics. To address these challenges, we propose RAMer (Reconstruction-based Adversarial Model for Emotion Recognition), which leverages adversarial learning to refine multi-modal representations by exploring both modality commonality and specificity through reconstructed features enhanced by contrastive learning. RAMer also introduces a personality auxiliary task to complement missing modalities using modality-level attention, improving emotion reasoning. To further strengthen the model's ability to capture label and modality interdependency, we propose a stack shuffle strategy to enrich correlations between labels and modality-specific features. Experiments on three benchmarks, i.e., MEmoR, CMU-MOSEI, and $M^3$ED, demonstrate that RAMer achieves state-of-the-art performance in dyadic and multi-party MMER scenarios. 

**Abstract (ZH)**: 传统的多模态多标签情绪识别（MMER）通常假设视频中的视觉、文本和声学模态都能完全可用。然而，在现实世界中的多参与者设置中，这一假设常常被违背，因为非说话者经常缺乏声学和文本输入，这导致了模型性能显著下降。现有的方法也往往会将异构模态统一到一个表示中，忽略了每个模态的独特特征。为了解决这些问题，我们提出了一种基于重构的对抗模型（RAMer，Reconstruction-based Adversarial Model for Emotion Recognition），该模型利用对抗学习通过重构特征来探索模态的共性和特性，从而改进多模态表示。RAMer 还引入了一个个性辅助任务，通过模态级注意力来补充缺失的模态，从而提高情绪推理能力。为了进一步增强模型捕捉标签和模态间依赖关系的能力，我们提出了堆叠洗牌策略以丰富标签和模态特定特征之间的关联。在三个基准数据集（MEmoR、CMU-MOSEI 和 $M^3$ED）上的实验结果表明，RAMer 在二元和多参与者 MMER 场景中达到了最先进的性能。 

---
# Leveraging Constraint Violation Signals For Action-Constrained Reinforcement Learning 

**Title (ZH)**: 利用约束违规信号进行动作受限强化学习 

**Authors**: Janaka Chathuranga Brahmanage, Jiajing Ling, Akshat Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.10431)  

**Abstract**: In many RL applications, ensuring an agent's actions adhere to constraints is crucial for safety. Most previous methods in Action-Constrained Reinforcement Learning (ACRL) employ a projection layer after the policy network to correct the action. However projection-based methods suffer from issues like the zero gradient problem and higher runtime due to the usage of optimization solvers. Recently methods were proposed to train generative models to learn a differentiable mapping between latent variables and feasible actions to address this issue. However, generative models require training using samples from the constrained action space, which itself is challenging. To address such limitations, first, we define a target distribution for feasible actions based on constraint violation signals, and train normalizing flows by minimizing the KL divergence between an approximated distribution over feasible actions and the target. This eliminates the need to generate feasible action samples, greatly simplifying the flow model learning. Second, we integrate the learned flow model with existing deep RL methods, which restrict it to exploring only the feasible action space. Third, we extend our approach beyond ACRL to handle state-wise constraints by learning the constraint violation signal from the environment. Empirically, our approach has significantly fewer constraint violations while achieving similar or better quality in several control tasks than previous best methods. 

**Abstract (ZH)**: 在许多强化学习（RL）应用中，确保智能体的行为遵守约束对于安全性至关重要。大多数现有的动作约束强化学习（ACRL）方法都在策略网络后使用投影层来纠正行为，以确保满足约束条件。然而，基于投影的方法存在零梯度问题和由于使用优化求解器而导致的运行时较长等问题。最近，提出了训练生成模型的方法，通过学习潜在变量与可行动作之间的可微映射来解决这些问题。然而，生成模型需要使用受约束动作空间中的样本进行训练，这本身是一项有挑战的任务。为了解决这些局限性，首先，我们根据约束违反应用于定义可行动作的目标分布，并通过最小化可行动作近似分布与目标分布之间的KL散度来训练标准化流模型，从而消除了生成可行动作样本的需要，大大简化了流模型的学习过程。其次，我们将所学的流模型与现有的深度RL方法结合起来，使其仅探索可行动作空间。最后，我们将该方法从ACRL扩展到处理状态依赖约束，通过从环境中学习约束违反而实现这一点。实验证明，与先前的最佳方法相比，我们的方法在多个控制任务中约束违反而明显较少，同时在质量方面表现出相似或更优的结果。 

---
# Real Time Control of Tandem-Wing Experimental Platform Using Concerto Reinforcement Learning 

**Title (ZH)**: 使用Concerto强化学习实时控制串联翼实验平台 

**Authors**: Zhang Minghao, Yang Xiaojun, Wang Zhihe, Wang Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10429)  

**Abstract**: This paper introduces the CRL2RT algorithm, an advanced reinforcement learning method aimed at improving the real-time control performance of the Direct-Drive Tandem-Wing Experimental Platform (DDTWEP). Inspired by dragonfly flight, DDTWEP's tandem wing structure causes nonlinear and unsteady aerodynamic interactions, leading to complex load behaviors during pitch, roll, and yaw maneuvers. These complexities challenge stable motion control at high frequencies (2000 Hz). To overcome these issues, we developed the CRL2RT algorithm, which combines classical control elements with reinforcement learning-based controllers using a time-interleaved architecture and a rule-based policy composer. This integration ensures finite-time convergence and single-life adaptability. Experimental results under various conditions, including different flapping frequencies and yaw disturbances, show that CRL2RT achieves a control frequency surpassing 2500 Hz on standard CPUs. Additionally, when integrated with classical controllers like PID, Adaptive PID, and Model Reference Adaptive Control (MRAC), CRL2RT enhances tracking performance by 18.3% to 60.7%. These findings demonstrate CRL2RT's broad applicability and superior performance in complex real-time control scenarios, validating its effectiveness in overcoming existing control strategy limitations and advancing robust, efficient real-time control for biomimetic aerial vehicles. 

**Abstract (ZH)**: 本文介绍了CRL2RT算法，这是一种改进直接驱动串联翼试验平台（DDTWEP）实时控制性能的先进强化学习方法。DDTWEP的串联翼结构受到蜻蜓飞行的启发，导致了非线性和不稳定气动相互作用，在俯仰、滚转和偏航机动过程中产生复杂的载荷行为。这些复杂性在高频（2000 Hz）下对稳定运动控制提出了挑战。为克服这些问题，我们开发了CRL2RT算法，该算法结合了经典的控制元素与基于强化学习的控制器，并采用时间交错架构和基于规则的策略合成器。这种集成确保了有限时间收敛和单一生活适应性。在不同条件下的实验结果，包括不同的拍翼频率和偏航干扰，表明CRL2RT可以实现超过2500 Hz的控制频率，即使在标准CPU上也是如此。此外，在与诸如PID、自适应PID和模型参考自适应控制（MRAC）等经典控制器集成时，CRL2RT的跟踪性能提升了18.3%至60.7%。这些发现表明，CRL2RT在复杂实时控制场景中具有广泛的适用性和优越性能，验证了其在克服现有控制策略限制并推动生物仿生航空器的稳健、高效实时控制方面的有效性。 

---
# Neuron Platonic Intrinsic Representation From Dynamics Using Contrastive Learning 

**Title (ZH)**: 使用对比学习的神经元柏拉图内在表征从动力学生成 

**Authors**: Wei Wu, Can Liao, Zizhen Deng, Zhengrui Guo, Jinzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10425)  

**Abstract**: The Platonic Representation Hypothesis suggests a universal, modality-independent reality representation behind different data modalities. Inspired by this, we view each neuron as a system and detect its multi-segment activity data under various peripheral conditions. We assume there's a time-invariant representation for the same neuron, reflecting its intrinsic properties like molecular profiles, location, and morphology. The goal of obtaining these intrinsic neuronal representations has two criteria: (I) segments from the same neuron should have more similar representations than those from different neurons; (II) the representations must generalize well to out-of-domain data. To meet these, we propose the NeurPIR (Neuron Platonic Intrinsic Representation) framework. It uses contrastive learning, with segments from the same neuron as positive pairs and those from different neurons as negative pairs. In implementation, we use VICReg, which focuses on positive pairs and separates dissimilar samples via regularization. We tested our method on Izhikevich model-simulated neuronal population dynamics data. The results accurately identified neuron types based on preset hyperparameters. We also applied it to two real-world neuron dynamics datasets with neuron type annotations from spatial transcriptomics and neuron locations. Our model's learned representations accurately predicted neuron types and locations and were robust on out-of-domain data (from unseen animals). This shows the potential of our approach for understanding neuronal systems and future neuroscience research. 

**Abstract (ZH)**: 柏拉图表征假设表明，存在一种普遍且跨模态不变的真实表征，隐藏在不同数据模态之下。受此启发，我们视每个神经元为一个系统，并在其在不同外围条件下的多段活动数据中进行检测。我们假设同一个神经元具有时间不变的表征，反映其内在性质，如分子谱型、位置和形态等。获得这些内在神经元表征的目标有两个标准：（I）来自同一个神经元的段应比来自不同神经元的段具有更高的相似性；（II）这些表征需要能够很好地泛化到域外数据。为达成这些目标，我们提出了NeurPIR（Neuron Platonic Intrinsic Representation）框架。该框架采用了对比学习方法，来自同一个神经元的段作为正样本对，而来自不同神经元的段作为负样本对。在实现过程中，我们使用了VICReg，该方法侧重于正样本对，并通过正则化分离差异样本。我们使用Izhikevich模型模拟的神经元群体动力学数据测试了该方法，结果能够根据预设的超参数准确识别神经元类型。我们还将该方法应用于两个实际中的神经元动力学数据集，并标注了神经元类型和位置数据。我们模型学习到的表征能够准确预测神经元类型和位置，并且在域外数据（来自未见过的动物）上表现稳健。这些结果展示了我们方法在理解神经元系统和未来神经科学研究中的潜在价值。 

---
# QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache 

**Title (ZH)**: QuantSpec: 基于分层量化KeyValue缓存的自推测解码 

**Authors**: Rishabh Tiwari, Haocheng Xi, Aditya Tomar, Coleman Hooper, Sehoon Kim, Maxwell Horton, Mahyar Najibi, Michael W. Mahoney, Kurt Keutzer, Amir Gholami  

**Link**: [PDF](https://arxiv.org/pdf/2502.10424)  

**Abstract**: Large Language Models (LLMs) are increasingly being deployed on edge devices for long-context settings, creating a growing need for fast and efficient long-context inference. In these scenarios, the Key-Value (KV) cache is the primary bottleneck in terms of both GPU memory and latency, as the full KV cache must be loaded for each decoding step. While speculative decoding is a widely accepted technique to accelerate autoregressive decoding, existing methods often struggle to achieve significant speedups due to inefficient KV cache optimization strategies and result in low acceptance rates. To address these challenges, we propose a novel self-speculative decoding framework, QuantSpec, where the draft model shares the architecture of the target model but employs a hierarchical 4-bit quantized KV cache and 4-bit quantized weights for acceleration. QuantSpec maintains high acceptance rates ($>$90%) and reliably provides consistent end-to-end speedups upto $\sim2.5\times$, outperforming other self-speculative decoding methods that use sparse KV cache for long-context LLM inference. QuantSpec also reduces the memory requirements by $\sim 1.3\times$ compared to these alternatives. 

**Abstract (ZH)**: 大型语言模型（LLMs）正越来越多地部署在边缘设备上以处理长上下文场景，这造成了对快速高效长上下文推理的日益增长的需求。在这些场景中，KV缓存是GPU内存和延迟的主要瓶颈，因为每次解码步骤都需要加载完整的KV缓存。虽然推测解码是一种广泛接受的加速自回归解码的技术，但现有的方法往往由于KV缓存优化策略效率低下而难以实现显著的加速，并且接受率较低。为了解决这些挑战，我们提出了一种新颖的自我推测解码框架QuantSpec，其中草稿模型与目标模型具有相同的架构，但使用分层的4位量化KV缓存和4位量化权重进行加速。QuantSpec 保持了较高的接受率（>90%），并可靠地提供了端到端加速，最高可达约2.5倍，超过那些使用稀疏KV缓存的其他自我推测解码方法在长上下文LLM推理中的表现。与这些替代方案相比，QuantSpec 还将内存要求降低了约1.3倍。 

---
# DA-LIF: Dual Adaptive Leaky Integrate-and-Fire Model for Deep Spiking Neural Networks 

**Title (ZH)**: DA-LIF: 双重自适应漏整合与发放模型在深度神经脉冲网络中的应用 

**Authors**: Tianqing Zhang, Kairong Yu, Jian Zhang, Hongwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10422)  

**Abstract**: Spiking Neural Networks (SNNs) are valued for their ability to process spatio-temporal information efficiently, offering biological plausibility, low energy consumption, and compatibility with neuromorphic hardware. However, the commonly used Leaky Integrate-and-Fire (LIF) model overlooks neuron heterogeneity and independently processes spatial and temporal information, limiting the expressive power of SNNs. In this paper, we propose the Dual Adaptive Leaky Integrate-and-Fire (DA-LIF) model, which introduces spatial and temporal tuning with independently learnable decays. Evaluations on both static (CIFAR10/100, ImageNet) and neuromorphic datasets (CIFAR10-DVS, DVS128 Gesture) demonstrate superior accuracy with fewer timesteps compared to state-of-the-art methods. Importantly, DA-LIF achieves these improvements with minimal additional parameters, maintaining low energy consumption. Extensive ablation studies further highlight the robustness and effectiveness of the DA-LIF model. 

**Abstract (ZH)**: 脉冲神经网络（SNNs）因其高效处理时空信息的能力而受到重视，提供了生物合理性、低能耗以及与神经形态硬件兼容的优点。然而，常用的泄漏型积分和放电（Leaky Integrate-and-Fire, LIF）模型忽略了神经元的异质性，独立地处理空间和时间信息，限制了SNNs的表达能力。在本文中，我们提出了双适应泄漏型积分和放电（Dual Adaptive Leaky Integrate-and-Fire, DA-LIF）模型，该模型引入了可独立学习的衰减参数来调节空间和时间信息。在静态数据集（CIFAR10/100、ImageNet）和神经形态数据集（CIFAR10-DVS、DVS128手势）上的评估结果显示，与最先进的方法相比，DA-LIF模型在较少的时隙内具有更高的准确性。重要的是，DA-LIF模型通过最少增加额外参数实现了这些改进，维持了低能耗。广泛进行的消融研究进一步强调了DA-LIF模型的鲁棒性和有效性。 

---
# DRiVE: Dynamic Recognition in VEhicles using snnTorch 

**Title (ZH)**: DRiVE: 车辆中动态识别的深度学习方法 Using snnTorch

注释：这里的翻译根据上下文做了些许调整，目的是更加符合中文的学术表达习惯。"snnTorch" 是一个用于构建和训练脉冲神经网络（Spiking Neural Networks）的库，中文通常直接使用英文名称。如果需要更准确的翻译，可以进一步了解具体研究内容。以下是更为严谨的翻译方式：

DRiVE: 车辆中动态识别的方法 使用 snnTorch

这样既能保持原文含义，又符合中文学术表达的习惯。 

**Authors**: Heerak Vora, Param Pathak, Parul Bakaraniya  

**Link**: [PDF](https://arxiv.org/pdf/2502.10421)  

**Abstract**: Spiking Neural Networks (SNNs) mimic biological brain activity, processing data efficiently through an event-driven design, wherein the neurons activate only when inputs exceed specific thresholds. Their ability to track voltage changes over time via membrane potential dynamics helps retain temporal information. This study combines SNNs with PyTorch's adaptable framework, snnTorch, to test their potential for image-based tasks. We introduce DRiVE, a vehicle detection model that uses spiking neuron dynamics to classify images, achieving 94.8% accuracy and a near-perfect 0.99 AUC score. These results highlight DRiVE's ability to distinguish vehicle classes effectively, challenging the notion that SNNs are limited to temporal data. As interest grows in energy-efficient neural models, DRiVE's success emphasizes the need to refine SNN optimization for visual tasks. This work encourages broader exploration of SNNs in scenarios where conventional networks struggle, particularly for real-world applications requiring both precision and efficiency. 

**Abstract (ZH)**: 脊状神经网络（SNNs）模仿生物大脑的活动，通过事件驱动的设计高效地处理数据，其中神经元仅在输入超过特定阈值时激活。神经元通过膜电位动态追踪电压变化，有助于保留时间信息。本研究将SNNs与PyTorch的灵活框架snntorch结合起来，测试其在图像任务中的潜在应用。我们引入了一种称为DRiVE的车辆检测模型，该模型利用突触神经元的动力学进行图像分类，实现了94.8%的准确率和近乎完美的0.99 AUC分数。这些结果突显了DRiVE在有效区分车辆类别方面的能力，挑战了SNNs仅限于处理时间数据的传统观念。随着对能效神经模型兴趣的增长，DRiVE的成功强调了优化SNNs以应对视觉任务的需求。本研究鼓励在传统网络表现不佳的情景下更广泛地探索SNNs，特别是在需要精确性和效率的现实世界应用中。 

---
# A Hybrid Swarm Intelligence Approach for Optimizing Multimodal Large Language Models Deployment in Edge-Cloud-based Federated Learning Environments 

**Title (ZH)**: 基于边缘-云联邦学习环境的多模式大型语言模型部署的混合 swarm 智能优化方法 

**Authors**: Gaith Rjouba, Hanae Elmekki, Saidul Islam, Jamal Bentahar, Rachida Dssouli  

**Link**: [PDF](https://arxiv.org/pdf/2502.10419)  

**Abstract**: The combination of Federated Learning (FL), Multimodal Large Language Models (MLLMs), and edge-cloud computing enables distributed and real- time data processing while preserving privacy across edge devices and cloud infrastructure. However, the deployment of MLLMs in FL environments with resource-constrained edge devices presents significant challenges, in- cluding resource management, communication overhead, and non-IID data. To address these challenges, we propose a novel hybrid framework wherein MLLMs are deployed on edge devices equipped with sufficient resources and battery life, while the majority of training occurs in the cloud. To identify suitable edge devices for deployment, we employ Particle Swarm Optimiza- tion (PSO), and Ant Colony Optimization (ACO) is utilized to optimize the transmission of model updates between edge and cloud nodes. This proposed swarm intelligence-based framework aims to enhance the efficiency of MLLM training by conducting extensive training in the cloud and fine-tuning at the edge, thereby reducing energy consumption and communication costs. Our experimental results show that the proposed method significantly improves system performance, achieving an accuracy of 92%, reducing communica- tion cost by 30%, and enhancing client participation compared to traditional FL methods. These results make the proposed approach highly suitable for large-scale edge-cloud computing systems. 

**Abstract (ZH)**: 联邦学习（FL）、多模态大型语言模型（MLLMs）与边缘-云计算的结合能够在保护边缘设备和云基础设施隐私的同时实现分布式和实时的数据处理。然而，在资源受限的边缘设备上部署MLLMs到FL环境中，面临着显著的挑战，包括资源管理、通信开销和非IID数据。为应对这些挑战，我们提出了一种新颖的混合框架，其中MLLMs被部署在具备充足资源和电池寿命的边缘设备上，而大部分训练则在云中进行。为了确定适合部署的边缘设备，我们采用粒子群优化（PSO）用于设备选择，蚁群优化（ACO）被用于优化边缘节点和云节点之间模型更新的传输。此基于群体智能的框架旨在通过在云中进行广泛的训练并在边缘进行微调来提高MLLM训练的效率，从而降低能耗和通信成本。我们的实验结果显示，与传统的FL方法相比，所提出的方法显著提高了系统性能，准确率达到92%，通信成本降低了30%，并且提升了客户端的参与度。这些结果表明，所提出的策略非常适合大规模边缘-云计算系统。 

---
# Evolutionary Power-Aware Routing in VANETs using Monte-Carlo Simulation 

**Title (ZH)**: 使用蒙特卡洛模拟的进化功率aware路由在VANET中 

**Authors**: J. Toutouh, S. Nesmachnow, E. Alba  

**Link**: [PDF](https://arxiv.org/pdf/2502.10417)  

**Abstract**: This work addresses the reduction of power consumption of the AODV routing protocol in vehicular networks as an optimization problem. Nowadays, network designers focus on energy-aware communication protocols, specially to deploy wireless networks. Here, we introduce an automatic method to search for energy-efficient AODV configurations by using an evolutionary algorithm and parallel Monte-Carlo simulations to improve the accuracy of the evaluation of tentative solutions. The experimental results demonstrate that significant power consumption improvements over the standard configuration can be attained, with no noteworthy loss in the quality of service. 

**Abstract (ZH)**: 本研究将汽车网络中AODV路由协议的功率消耗削减问题作为优化问题进行探讨。目前，网络设计师们专注于能源感知通信协议的设计，特别是为了部署无线网络。在此基础上，我们提出了一种自动方法，通过使用进化算法和并行蒙特卡洛模拟来搜索高效的AODV配置，以提高对初步解决方案评估的准确性。实验结果表明，与标准配置相比，可以显著降低功率消耗，同时服务质量并未出现显著下降。 

---
# Machine Learning-Driven Convergence Analysis in Multijurisdictional Compliance Using BERT and K-Means Clustering 

**Title (ZH)**: 基于机器学习的多司法管辖区合规性收敛分析：用BERT和K-均值聚类驱动的研究 

**Authors**: Raj Sonani, Lohalekar Prayas  

**Link**: [PDF](https://arxiv.org/pdf/2502.10413)  

**Abstract**: Digital data continues to grow, there has been a shift towards using effective regulatory mechanisms to safeguard personal information. The CCPA of California and the General Data Protection Regulation (GDPR) of the European Union are two of the most important privacy laws. The regulation is intended to safeguard consumer privacy, but it varies greatly in scope, definitions, and methods of enforcement. This paper presents a fresh approach to adaptive compliance, using machine learning and emphasizing natural language processing (NLP) as the primary focus of comparison between the GDPR and CCPA. Using NLP, this study compares various regulations to identify areas where they overlap or diverge. This includes the "right to be forgotten" provision in the GDPR and the "opt-out of sale" provision under CCPA. International companies can learn valuable lessons from this report, as it outlines strategies for better enforcement of laws across different nations. Additionally, the paper discusses the challenges of utilizing NLP in legal literature and proposes methods to enhance the model-ability of machine learning models for studying regulations. The study's objective is to "bridge the gap between legal knowledge and technical expertise" by developing regulatory compliance strategies that are more efficient in operation and more effective in data protection. 

**Abstract (ZH)**: 数字数据继续增长，监管机制的有效性成为保护个人隐私的关键。加州消费者隐私法（CCPA）和欧盟的一般数据保护条例（GDPR）是最重要的隐私法律之一。这些法规旨在保护消费者隐私，但在范围、定义和执行方法上存在很大差异。本文提出了一种新的适应性合规方法，利用机器学习，并以自然语言处理（NLP）为主要比较手段，对比GDPR和CCPA的规定。通过NLP，本研究比较了各种法规，以识别它们之间重叠或差异的领域，包括GDPR中的“被遗忘权”和CCPA中的“销售选择退出”条款。国际公司可以从这份报告中获得宝贵的经验，因为它概述了在不同国家更好地执行法律的战略。此外，本文还讨论了在法律文献中使用NLP所面临的挑战，并提出了增强机器学习模型研究法规能力的方法。本文的目标是通过开发更有效的操作和更有效的数据保护策略，弥合法律知识与技术专长之间的差距。 

---
# Identifying relevant indicators for monitoring a National Artificial Intelligence Strategy 

**Title (ZH)**: 识别监测国家级人工智能战略的相关指标 

**Authors**: Renata Pelissari, Ricardo Suyama, Leonardo Tomazeli Duarte, Henrique Sá Earp  

**Link**: [PDF](https://arxiv.org/pdf/2502.10412)  

**Abstract**: How can a National Artificial Intelligence Strategy be effectively monitored? To address this question, we propose a methodology consisting of two key components. First, it involves identifying relevant indicators within national AI strategies. Second, it assesses the alignment between these indicators and the strategic actions of a specific government's AI strategy, allowing for a critical evaluation of its monitoring measures. Moreover, identifying these indicators helps assess the overall quality of the strategy's structure. A lack of alignment between strategic actions and the identified indicators may reveal gaps or blind spots in the strategy. This methodology is demonstrated using the Brazilian AI strategy as a case study. 

**Abstract (ZH)**: 如何有效地监控国家人工智能战略？为回答这一问题，我们提出了一种由两个关键部分组成的 methodology。首先，涉及识别国家人工智能战略中的相关指标。其次，评估这些指标与特定政府人工智能战略的行动的一致性，从而对其实行监控的措施进行批判性评价。此外，识别这些指标有助于评估战略结构的整体质量。战略行动与识别的指标之间缺乏一致性可能揭示了战略中存在的缺口或盲点。该 methodology 通过巴西的人工智能战略案例进行演示。 

---
# TrueReason: An Exemplar Personalised Learning System Integrating Reasoning with Foundational Models 

**Title (ZH)**: TrueReason：一个结合推理与基础模型的范例个性化学习系统 

**Authors**: Sahan Bulathwela, Daniel Van Niekerk, Jarrod Shipton, Maria Perez-Ortiz, Benjamin Rosman, John Shawe-Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2502.10411)  

**Abstract**: Personalised education is one of the domains that can greatly benefit from the most recent advances in Artificial Intelligence (AI) and Large Language Models (LLM). However, it is also one of the most challenging applications due to the cognitive complexity of teaching effectively while personalising the learning experience to suit independent learners. We hypothesise that one promising approach to excelling in such demanding use cases is using a \emph{society of minds}. In this chapter, we present TrueReason, an exemplar personalised learning system that integrates a multitude of specialised AI models that can mimic micro skills that are composed together by a LLM to operationalise planning and reasoning. The architecture of the initial prototype is presented while describing two micro skills that have been incorporated in the prototype. The proposed system demonstrates the first step in building sophisticated AI systems that can take up very complex cognitive tasks that are demanded by domains such as education. 

**Abstract (ZH)**: 个性化教育是可以从最近人工智能（AI）和大型语言模型（LLM）的最新进展中受益匪浅的一个领域。然而，这也是一项最具挑战性的应用之一，原因在于有效进行认知教学的同时还需要为自主学习者个性化学习体验所带来的复杂性。我们假设，这样具有挑战性的应用场景中取得优异成果的一种有前途的方法是采用“心灵社会”（society of minds）这一理念。在本章中，我们介绍了TrueReason，这是一种具有代表性的个性化学习系统，该系统集成了多种专门的AI模型，这些模型可以模拟大型语言模型组合的小技能，从而实现计划和推理的运作。我们描述了初始原型的架构，并介绍了原型中已集成的两种小技能。所提出的系统展示了构建能够承担教育等领域所要求的极其复杂认知任务的复杂AI系统的初始步骤。 

---
# Auto-Evaluation: A Critical Measure in Driving Improvements in Quality and Safety of AI-Generated Lesson Resources 

**Title (ZH)**: 自动评估：在提高人工智能生成教学资源质量和安全性方面的一项关键性衡量指标 

**Authors**: Hannah-Beth Clark, Margaux Dowland, Laura Benton, Reka Budai, Ibrahim Kaan Keskin, Emma Searle, Matthew Gregory, Mark Hodierne, William Gayne, John Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2502.10410)  

**Abstract**: As a publicly funded body in the UK, Oak National Academy is in a unique position to innovate within this field as we have a comprehensive curriculum of approximately 13,000 open education resources (OER) for all National Curriculum subjects, designed and quality-assured by expert, human teachers. This has provided the corpus of content needed for building a high-quality AI-powered lesson planning tool, Aila, that is free to use and, therefore, accessible to all teachers across the country. Furthermore, using our evidence-informed curriculum principles, we have codified and exemplified each component of lesson design. To assess the quality of lessons produced by Aila at scale, we have developed an AI-powered auto-evaluation agent,facilitating informed improvements to enhance output quality. Through comparisons between human and auto-evaluations, we have begun to refine this agent further to increase its accuracy, measured by its alignment with an expert human evaluator. In this paper we present this iterative evaluation process through an illustrative case study focused on one quality benchmark - the level of challenge within multiple-choice quizzes. We also explore the contribution that this may make to similar projects and the wider sector. 

**Abstract (ZH)**: 作为英国的公立机构，橡树国家学院在这一领域拥有独特的创新地位。我们拥有涵盖所有国家课程科目的约13,000个开放教育资源（OER），这些资源由专家人类教师设计并经过质量保证。这为构建高质量的AI辅助课程规划工具Aila提供了所需的内容库，该工具免费提供，从而使得全国的教师都能无障碍使用。此外，基于我们以实证为基础的课程原则，我们已经对课程设计的每个要素进行了编码和示例。为了评估Aila大规模生成的课程的质量，我们开发了一个AI辅助的自动评估代理，以促进针对性的改进，提高输出质量。通过人工评估与自动评估的对比，我们已经开始进一步完善这个代理，以提高其准确性，即与专家人工评估的一致程度。在这篇论文中，我们通过一个示例案例研究，展示了这种迭代评估过程，并专注于一个质量标准——多项选择题的难度级别。我们还探讨了这一过程可能对类似项目和更广泛领域的贡献。 

---
# Data Science Students Perspectives on Learning Analytics: An Application of Human-Led and LLM Content Analysis 

**Title (ZH)**: 数据科学学生对学习分析的看法：一种基于人工主导和LLM内容分析的应用 

**Authors**: Raghda Zahran, Jianfei Xu, Huizhi Liang, Matthew Forshaw  

**Link**: [PDF](https://arxiv.org/pdf/2502.10409)  

**Abstract**: Objective This study is part of a series of initiatives at a UK university designed to cultivate a deep understanding of students' perspectives on analytics that resonate with their unique learning needs. It explores collaborative data processing undertaken by postgraduate students who examined an Open University Learning Analytics Dataset (OULAD).
Methods A qualitative approach was adopted, integrating a Retrieval-Augmented Generation (RAG) and a Large Language Model (LLM) technique with human-led content analysis to gather information about students' perspectives based on their submitted work. The study involved 72 postgraduate students in 12 groups.
Findings The analysis of group work revealed diverse insights into essential learning analytics from the students' perspectives. All groups adopted a structured data science methodology. The questions formulated by the groups were categorised into seven themes, reflecting their specific areas of interest. While there was variation in the selected variables to interpret correlations, a consensus was found regarding the general results.
Conclusion A significant outcome of this study is that students specialising in data science exhibited a deeper understanding of learning analytics, effectively articulating their interests through inferences drawn from their analyses. While human-led content analysis provided a general understanding of students' perspectives, the LLM offered nuanced insights. 

**Abstract (ZH)**: 目的：本研究属于英国某大学一系列旨在培养学生对数据分析深刻理解的倡议之一，这些理解应与学生的独特学习需求相一致。本研究探讨了对开放大学学习分析数据集（OULAD）进行合作数据处理的研究生学生们的视角。

方法：本研究采取定性方法，整合了检索增强生成（RAG）和大型语言模型（LLM）技术，并结合由人为主导的内容分析，以收集关于学生视角的信息。研究涉及12个小组的72名研究生。

发现：对小组工作的分析揭示了学生从自身视角出发对学习分析所获得的多样见解。所有小组均采用了结构化的数据科学方法。组内提出的问题被归类为七个主题，反映了他们具体的研究兴趣。虽然在选择用于解释相关性的变量上存在差异，但在总体结果方面，研究小组达成了共识。

结论：本研究的一个重要成果是数据科学专业的学生对学习分析表现出更深入的理解，能够通过分析结果进行有效的推理和表达自己的研究兴趣。虽然人为主导的内容分析提供了学生总体视角的理解，但LLM则提供了更加细致和有针对性的见解。 

---
# Knowledge Tracing in Programming Education Integrating Students' Questions 

**Title (ZH)**: 将“Knowledge Tracing in Programming Education Integrating Students' Questions”翻译成中文，符合学术规范后为：

“融合学生问题的编程教育中的知识追踪”

这个翻译保留了原文的核心含义，同时用中文的学术表达方式进行了适配。 

**Authors**: Doyoun Kim, Suin Kim, Yojan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2502.10408)  

**Abstract**: Knowledge tracing (KT) in programming education presents unique challenges due to the complexity of coding tasks and the diverse methods students use to solve problems. Although students' questions often contain valuable signals about their understanding and misconceptions, traditional KT models often neglect to incorporate these questions as inputs to address these challenges. This paper introduces SQKT (Students' Question-based Knowledge Tracing), a knowledge tracing model that leverages students' questions and automatically extracted skill information to enhance the accuracy of predicting students' performance on subsequent problems in programming education. Our method creates semantically rich embeddings that capture not only the surface-level content of the questions but also the student's mastery level and conceptual understanding. Experimental results demonstrate SQKT's superior performance in predicting student completion across various Python programming courses of differing difficulty levels. In in-domain experiments, SQKT achieved a 33.1\% absolute improvement in AUC compared to baseline models. The model also exhibited robust generalization capabilities in cross-domain settings, effectively addressing data scarcity issues in advanced programming courses. SQKT can be used to tailor educational content to individual learning needs and design adaptive learning systems in computer science education. 

**Abstract (ZH)**: 编程教育中的知识追踪（KT）面临着独特的挑战，因为编码任务的复杂性和学生解决问题方法的多样性。尽管学生的问题通常包含了关于他们理解程度和误解的重要信号，但传统的KT模型往往未能将这些问题作为输入来解决这些挑战。本文介绍了基于学生问题的知识追踪（SQKT，Students' Question-based Knowledge Tracing）模型，该模型利用学生的提问及其自动提取的技能信息，以提高预测学生在后续编程问题上的表现的准确性。我们的方法创建了语义丰富的嵌入表示，不仅捕捉问题的表层内容，还能反映学生的掌握程度和概念理解。实验结果表明，SQKT在预测不同难度级别Python编程课程的学生完成情况方面表现优异。在领域内实验中，SQKT的AUC绝对改进率达到了33.1%。此外，该模型在跨领域的设置中也表现出强大的泛化能力，有效地解决了高级编程课程中的数据稀缺问题。SQKT可以用于个性化教育内容，以适应学生的个别学习需求，并设计计算机科学教育中的自适应学习系统。 

---
# Addressing Bias in Generative AI: Challenges and Research Opportunities in Information Management 

**Title (ZH)**: 解决生成式AI中的偏见：信息管理中的挑战与研究机会 

**Authors**: Xiahua Wei, Naveen Kumar, Han Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10407)  

**Abstract**: Generative AI technologies, particularly Large Language Models (LLMs), have transformed information management systems but introduced substantial biases that can compromise their effectiveness in informing business decision-making. This challenge presents information management scholars with a unique opportunity to advance the field by identifying and addressing these biases across extensive applications of LLMs. Building on the discussion on bias sources and current methods for detecting and mitigating bias, this paper seeks to identify gaps and opportunities for future research. By incorporating ethical considerations, policy implications, and sociotechnical perspectives, we focus on developing a framework that covers major stakeholders of Generative AI systems, proposing key research questions, and inspiring discussion. Our goal is to provide actionable pathways for researchers to address bias in LLM applications, thereby advancing research in information management that ultimately informs business practices. Our forward-looking framework and research agenda advocate interdisciplinary approaches, innovative methods, dynamic perspectives, and rigorous evaluation to ensure fairness and transparency in Generative AI-driven information systems. We expect this study to serve as a call to action for information management scholars to tackle this critical issue, guiding the improvement of fairness and effectiveness in LLM-based systems for business practice. 

**Abstract (ZH)**: 生成式人工智能技术，尤其是大型语言模型（LLMs），已彻底改变了信息管理系统，但同时也引入了大量偏见，这些偏见可能损害LLMs在指导企业决策方面的有效性。这一挑战为信息管理领域的学者们提供了一个独特的机会，让他们可以通过识别并解决广泛使用LLMs中的偏见来推进该领域的发展。基于对偏见来源和当前偏见检测与缓解方法的讨论，本文旨在识别未来研究中的缺口和机遇。通过纳入伦理考量、政策影响和社会技术视角，我们致力于构建一个框架，涵盖生成式人工智能系统的主要利益相关者，提出关键研究问题，并激发讨论。我们的目标是为研究人员提供实际的操作路径，以解决LLMs应用中的偏见问题，从而推动研究发展，最终指导企业管理实践。我们前瞻性的框架和研究议程倡导跨学科方法、创新方法、动态视角和严格的评估，以确保生成式人工智能驱动的信息系统中的公正性和透明度。我们期望这项研究能够成为信息管理领域学者的行动号召，指导基于LLM系统的公平性和有效性改进，以服务于企业的实际应用。 

---
# FishBargain: An LLM-Empowered Bargaining Agent for Online Fleamarket Platform Sellers 

**Title (ZH)**: FishBargain：一种基于大语言模型的在线地摊交易平台谈判代理 

**Authors**: Dexin Kong, Xu Yan, Ming Chen, Shuguang Han, Jufeng Chen, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.10406)  

**Abstract**: Different from traditional Business-to-Consumer e-commerce platforms~(e.g., Amazon), online fleamarket platforms~(e.g., Craigslist) mainly focus on individual sellers who are lack of time investment and business proficiency. Individual sellers often struggle with the bargaining process and thus the deal is unaccomplished. Recent advancements in Large Language Models(LLMs) demonstrate huge potential in various dialogue tasks, but those tasks are mainly in the form of passively following user's instruction. Bargaining, as a form of proactive dialogue task, represents a distinct art of dialogue considering the dynamism of environment and uncertainty of adversary strategies. In this paper, we propose an LLM-empowered bargaining agent designed for online fleamarket platform sellers, named as FishBargain. Specifically, FishBargain understands the chat context and product information, chooses both action and language skill considering possible adversary actions and generates utterances. FishBargain has been tested by thousands of individual sellers on one of the largest online fleamarket platforms~(Xianyu) in China. Both qualitative and quantitative experiments demonstrate that FishBargain can effectively help sellers make more deals. 

**Abstract (ZH)**: 与传统的商家对消费者的电子商务平台（例如Amazon）不同，二手交易平台（例如Craigslist）主要侧重于缺乏时间和商业技能投入的个人卖家。个人卖家往往在谈判过程中遇到困难，导致交易未能完成。最近在大型语言模型（LLMs）方面的进展展示了其在各种对话任务中的巨大潜力，但这些任务大多是以被动遵循用户指令的形式出现。而谈判作为一种主动对话任务，由于环境的动态性和对手策略的不确定性，具有独特的对话艺术。在本文中，我们提出了一种为二手交易平台卖家设计的由大型语言模型支持的谈判代理，命名为FishBargain。具体而言，FishBargain理解聊天上下文和产品信息，根据可能的对手行动选择行动和语言技能，并生成相应的表达。FishBargain已经在最大的中国在线二手交易平台之一（闲鱼）上经过数千个个人卖家的测试。定性和定量实验均表明，FishBargain能够有效帮助卖家达成更多交易。 

---
# You Can't Get There From Here: Redefining Information Science to address our sociotechnical futures 

**Title (ZH)**: “此路不通：重新定义信息科学以应对我们的社会技术未来” 

**Authors**: Scott Humr, Mustafa Canan  

**Link**: [PDF](https://arxiv.org/pdf/2502.10401)  

**Abstract**: Current definitions of Information Science are inadequate to comprehensively describe the nature of its field of study and for addressing the problems that are arising from intelligent technologies. The ubiquitous rise of artificial intelligence applications and their impact on society demands the field of Information Science acknowledge the sociotechnical nature of these technologies. Previous definitions of Information Science over the last six decades have inadequately addressed the environmental, human, and social aspects of these technologies. This perspective piece advocates for an expanded definition of Information Science that fully includes the sociotechnical impacts information has on the conduct of research in this field. Proposing an expanded definition of Information Science that includes the sociotechnical aspects of this field should stimulate both conversation and widen the interdisciplinary lens necessary to address how intelligent technologies may be incorporated into society and our lives more fairly. 

**Abstract (ZH)**: 当前的信息科学定义不足以全面描述该领域的本质及其所研究的问题，并且难以应对由智能技术引发的问题。无处不在的智能应用及其对社会的影响要求信息科学领域承认这些技术的社会技术本质。过去六十年中信息科学的定义未能充分涵盖这些技术的环境、人类和社会方面的影响。本文倡导扩展信息科学的定义，全面纳入信息在这一领域研究中产生的社会技术影响。提出扩展信息科学的定义以包括这一领域的社会技术方面，应促进相关讨论，并拓宽多学科视角，以解决如何更公平地将智能技术整合到社会和我们的生活中。 

---
# Data Stewardship Decoded: Mapping Its Diverse Manifestations and Emerging Relevance at a time of AI 

**Title (ZH)**: 数据 stewardship 解码：映射其多样的表现形式及在人工智能时代新兴的重要性 

**Authors**: Stefaan Verhulst  

**Link**: [PDF](https://arxiv.org/pdf/2502.10399)  

**Abstract**: Data stewardship has become a critical component of modern data governance, especially with the growing use of artificial intelligence (AI). Despite its increasing importance, the concept of data stewardship remains ambiguous and varies in its application. This paper explores four distinct manifestations of data stewardship to clarify its emerging position in the data governance landscape. These manifestations include a) data stewardship as a set of competencies and skills, b) a function or role within organizations, c) an intermediary organization facilitating collaborations, and d) a set of guiding principles. The paper subsequently outlines the core competencies required for effective data stewardship, explains the distinction between data stewards and Chief Data Officers (CDOs), and details the intermediary role of stewards in bridging gaps between data holders and external stakeholders. It also explores key principles aligned with the FAIR framework (Findable, Accessible, Interoperable, Reusable) and introduces the emerging principle of AI readiness to ensure data meets the ethical and technical requirements of AI systems. The paper emphasizes the importance of data stewardship in enhancing data collaboration, fostering public value, and managing data reuse responsibly, particularly in the era of AI. It concludes by identifying challenges and opportunities for advancing data stewardship, including the need for standardized definitions, capacity building efforts, and the creation of a professional association for data stewardship. 

**Abstract (ZH)**: 数据管理已成为现代数据治理的关键组成部分，尤其是在人工智能（AI）应用日益广泛的情况下。尽管其重要性不断增加，但数据管理的概念仍然模糊不清，其应用也存在差异。本文探讨了数据管理的四种不同表现形式，以明确其在数据治理格局中的新兴地位。这些表现形式包括：a) 数据管理作为一组能力和技能，b) 组织中的功能或角色，c) 促进合作的中介组织，以及d) 一套指导原则。文章随后概述了有效数据管理所需的核心能力，解释了数据管理师与首席数据官（CDO）之间的区别，并详细介绍了管理在数据持有者与外部利益相关者之间建立联系桥梁的作用。同时，文章探讨了与可访问性、互操作性和可重用性框架（FAIR框架）相一致的关键原则，并引入了“AI准备性”这一新兴原则，以确保数据满足AI系统的伦理和技术要求。本文强调了数据管理在促进数据合作、培育公共价值以及负责任地管理数据再利用方面的关键作用，特别是在AI时代。最后，文章指出了数据管理的发展挑战和机遇，包括建立标准化定义、能力提升努力以及创建数据管理专业协会的需求。 

---
# Practical Application and Limitations of AI Certification Catalogues 

**Title (ZH)**: AI认证目录的实用应用与局限性 

**Authors**: Gregor Autischer, Kerstin Waxnegger, Dominik Kowald  

**Link**: [PDF](https://arxiv.org/pdf/2502.10398)  

**Abstract**: In this work-in-progress, we investigate the certification of artificial intelligence (AI) systems, focusing on the practical application and limitations of existing certification catalogues by attempting to certify a publicly available AI system. We aim to evaluate how well current approaches work to effectively certify an AI system, and how publicly accessible AI systems, that might not be actively maintained or initially intended for certification, can be selected and used for a sample certification process. Our methodology involves leveraging the Fraunhofer AI Assessment Catalogue as a comprehensive tool to systematically assess an AI model's compliance with certification standards. We find that while the catalogue effectively structures the evaluation process, it can also be cumbersome and time-consuming to use. We observe the limitations of an AI system that has no active development team anymore and highlighted the importance of complete system documentation. Finally, we identify some limitations of the certification catalogues used and proposed ideas on how to streamline the certification process. 

**Abstract (ZH)**: 在本次工作进展中，我们探讨了对人工智能（AI）系统的认证问题，重点关注现有认证目录的实践应用及其局限性。我们致力于评估当前方法在有效认证AI系统方面的表现，并考察如何选择和使用可能未被积极维护或最初未设计用于认证的公开可用的AI系统进行样本认证过程。我们的方法涉及利用Fraunhofer AI评估目录作为全面工具，系统性地评估AI模型是否符合认证标准。我们发现，尽管该目录有效地结构化了评估过程，但在使用过程中也存在繁琐和耗时的问题。我们还观察到一个已无活跃开发团队的AI系统存在的局限性，并强调了完整系统文档的重要性。最后，我们指出了所使用的认证目录的一些局限性，并提出了一些简化认证过程的想法。 

---
# DASKT: A Dynamic Affect Simulation Method for Knowledge Tracing 

**Title (ZH)**: DASKT：一种动态情感模拟的知识追踪方法 

**Authors**: Xinjie Sun, Kai Zhang, Qi Liu, Shuanghong Shen, Fei Wang, Yuxiang Guo, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.10396)  

**Abstract**: Knowledge Tracing (KT) predicts future performance by modeling students' historical interactions, and understanding students' affective states can enhance the effectiveness of KT, thereby improving the quality of education. Although traditional KT values students' cognition and learning behaviors, efficient evaluation of students' affective states and their application in KT still require further exploration due to the non-affect-oriented nature of the data and budget constraints. To address this issue, we propose a computation-driven approach, Dynamic Affect Simulation Knowledge Tracing (DASKT), to explore the impact of various student affective states (such as frustration, concentration, boredom, and confusion) on their knowledge states. In this model, we first extract affective factors from students' non-affect-oriented behavioral data, then use clustering and spatiotemporal sequence modeling to accurately simulate students' dynamic affect changes when dealing with different problems. Subsequently, {\color{blue}we incorporate affect with time-series analysis to improve the model's ability to infer knowledge states over time and space.} Extensive experimental results on two public real-world educational datasets show that DASKT can achieve more reasonable knowledge states under the effect of students' affective states. Moreover, DASKT outperforms the most advanced KT methods in predicting student performance. Our research highlights a promising avenue for future KT studies, focusing on achieving high interpretability and accuracy. 

**Abstract (ZH)**: 知识追踪（KT）通过建模学生的历史交互来预测未来的绩效，而理解学生的情感状态可以提高KT的有效性，从而提高教育质量。尽管传统的KT重视学生认知和学习行为，但由于数据和预算的限制，高效评价学生的情感状态并将其应用于KT中仍然需要进一步探索。为了解决这一问题，我们提出了一种计算驱动的方法——动态情感模拟知识追踪（DASKT），以探索不同类型学生情感状态（如烦躁、专注、枯燥和困惑）对其知识状态的影响。在该模型中，我们首先从学生的非情感导向行为数据中提取情感因素，然后使用聚类和时空序列建模来准确模拟学生在处理不同问题时动态情感变化。随后，我们通过结合情感与时间序列分析，提高模型在时间和空间上推断知识状态的能力。在两个公开的真实世界教育数据集上的广泛实验结果表明，DASKT在学生情感状态的影响下可以实现更合理的知识状态估计。此外，DASKT在预测学生绩效方面优于最先进的KT方法。我们的研究指出了未来KT研究的一个有前景的方向，重点关注实现高可解释性和准确性。 

---
# An Integrated Platform for Studying Learning with Intelligent Tutoring Systems: CTAT+TutorShop 

**Title (ZH)**: 面向智能辅导系统的学习研究集成平台：CTAT+TutorShop 

**Authors**: Vincent Aleven, Conrad Borchers, Yun Huang, Tomohiro Nagashima, Bruce McLaren, Paulo Carvalho, Octav Popescu, Jonathan Sewall, Kenneth Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2502.10395)  

**Abstract**: Intelligent tutoring systems (ITSs) are effective in helping students learn; further research could make them even more effective. Particularly desirable is research into how students learn with these systems, how these systems best support student learning, and what learning sciences principles are key in ITSs. CTAT+Tutorshop provides a full stack integrated platform that facilitates a complete research lifecycle with ITSs, which includes using ITS data to discover learner challenges, to identify opportunities for system improvements, and to conduct experimental studies. The platform includes authoring tools to support and accelerate development of ITS, which provide automatic data logging in a format compatible with DataShop, an independent site that supports the analysis of ed tech log data to study student learnings. Among the many technology platforms that exist to support learning sciences research, CTAT+Tutorshop may be the only one that offers researchers the possibility to author elements of ITSs, or whole ITSs, as part of designing studies. This platform has been used to develop and conduct an estimated 147 research studies which have run in a wide variety of laboratory and real-world educational settings, including K-12 and higher education, and have addressed a wide range of research questions. This paper presents five case studies of research conducted on the CTAT+Tutorshop platform, and summarizes what has been accomplished and what is possible for future researchers. We reflect on the distinctive elements of this platform that have made it so effective in facilitating a wide range of ITS research. 

**Abstract (ZH)**: 智能教学系统（ITSs）在帮助学生学习方面非常有效；进一步的研究可以使它们更为有效。特别是，需要研究学生如何使用这些系统学习、这些系统如何最好地支持学生学习，以及哪些学习科学原理在ITSs中最为关键。CTAT+Tutorshop 提供了一个全面集成的平台，使对ITSs的研究生命周期变得更加便利，包括利用ITS数据发现学习者挑战，确定系统改进的机会，并进行实验研究。该平台包括支持和加速开发ITS的编辑工具，这些工具能够自动记录数据，并与DataShop兼容，DataShop是一个独立的站点，支持通过分析教学技术日志数据来研究学生的学习情况。在支持学习科学研究的各种技术平台中，CTAT+Tutorshop 可能是唯一一个提供给研究人员设计研究以部分或全部创建ITSs可能性的平台。该平台已被用于开发和执行大约147项研究，这些研究在包括K-12教育和高等教育在内的各种实验室和实际教育环境中运行，并涵盖了广泛的研究问题。本文介绍了五个在CTAT+Tutorshop平台上进行的研究案例，总结了已取得的成果以及未来研究的可能性。我们反思了使该平台能够在广泛类型的ITS研究中发挥重要作用的独特元素。 

---
# A Glitch in the Matrix? Locating and Detecting Language Model Grounding with Fakepedia 

**Title (ZH)**: 《矩阵中的故障？寻找和检测语言模型_grounding_的假维基》

注释：
1. "Grounding" 在机器学习和人工智能领域通常指的是将抽象的概念或语言描述与具体的现实世界对象或实体关联起来的过程，此处保持了英文原词。
2. "假维基" 是对 "Fakepedia" 这个专有名词的直译，用以指代模拟或伪造的信息库。
3. 翻译中保持了原文的学术风格和结构，确保符合学术规范。 

**Authors**: Giovanni Monea, Maxime Peyrard, Martin Josifoski, Vishrav Chaudhary, Jason Eisner, Emre Kıcıman, Hamid Palangi, Barun Patra, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2312.02073)  

**Abstract**: Large language models (LLMs) have an impressive ability to draw on novel information supplied in their context. Yet the mechanisms underlying this contextual grounding remain unknown, especially in situations where contextual information contradicts factual knowledge stored in the parameters, which LLMs also excel at recalling. Favoring the contextual information is critical for retrieval-augmented generation methods, which enrich the context with up-to-date information, hoping that grounding can rectify outdated or noisy stored knowledge. We present a novel method to study grounding abilities using Fakepedia, a novel dataset of counterfactual texts constructed to clash with a model's internal parametric knowledge. In this study, we introduce Fakepedia, a counterfactual dataset designed to evaluate grounding abilities when the internal parametric knowledge clashes with the contextual information. We benchmark various LLMs with Fakepedia and conduct a causal mediation analysis of LLM components when answering Fakepedia queries, based on our Masked Grouped Causal Tracing (MGCT) method. Through this analysis, we identify distinct computational patterns between grounded and ungrounded responses. We finally demonstrate that distinguishing grounded from ungrounded responses is achievable through computational analysis alone. Our results, together with existing findings about factual recall mechanisms, provide a coherent narrative of how grounding and factual recall mechanisms interact within LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）具备从其上下文环境中汲取新颖信息的能力，这是一个令人印象深刻的特性。然而，这些模型在上下文中获得信息的具体机制仍然未知，尤其是在上下文信息与模型存储的背景事实知识相矛盾的情况下更是如此。在具有一种增强检索方法的背景下，倾向于使用的上下文信息对于获取最新的信息至关重要，期望这种上下文信息可以纠正过时或噪声较大的存储知识。我们提出了一种新的方法来研究这些语言模型的语境相关能力，使用了一种名为Fakepedia的新型数据集，这是一种旨在与模型内部参数化知识产生矛盾的虚假文本数据集。本研究中，我们介绍了Fakepedia，这是一种特别设计的对抗性数据集，在其中内部参数化知识与上下文信息产生冲突时，用于评估模型的位置相关能力。我们使用Fakepedia对多种LLMs进行基准测试，并基于我们的掩码分组因果追踪（MGCT）方法，对LLM组件在回答Fakepedia查询时的作用进行因果中介分析。通过这种方法的分析，我们识别出了有位置依托和无位置依托响应之间不同的计算模式。最终，我们证明仅通过计算分析就可以区分有位置依托和无位置依托的响应。我们的结果与现有关于事实回忆机制的研究发现一起，提供了一个关于LLMs中的位置相关机制和事实回忆机制之间互动的连贯叙述。 

---
