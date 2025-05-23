# Preference Leakage: A Contamination Problem in LLM-as-a-judge 

**Title (ZH)**: 偏好泄露：LLM 作为裁判时的污染问题 

**Authors**: Dawei Li, Renliang Sun, Yue Huang, Ming Zhong, Bohan Jiang, Jiawei Han, Xiangliang Zhang, Wei Wang, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.01534)  

**Abstract**: Large Language Models (LLMs) as judges and LLM-based data synthesis have emerged as two fundamental LLM-driven data annotation methods in model development. While their combination significantly enhances the efficiency of model training and evaluation, little attention has been given to the potential contamination brought by this new model development paradigm. In this work, we expose preference leakage, a contamination problem in LLM-as-a-judge caused by the relatedness between the synthetic data generators and LLM-based evaluators. To study this issue, we first define three common relatednesses between data generator LLM and judge LLM: being the same model, having an inheritance relationship, and belonging to the same model family. Through extensive experiments, we empirically confirm the bias of judges towards their related student models caused by preference leakage across multiple LLM baselines and benchmarks. Further analysis suggests that preference leakage is a pervasive issue that is harder to detect compared to previously identified biases in LLM-as-a-judge scenarios. All of these findings imply that preference leakage is a widespread and challenging problem in the area of LLM-as-a-judge. We release all codes and data at: this https URL. 

**Abstract (ZH)**: 作为法官的大规模语言模型（LLMs）和基于LLM的数据合成已经成为了两种基本的LLM驱动的数据注释方法，广泛应用于模型开发中。这两种方法的结合显著提升了模型训练和评估的效率，但尚未充分关注这种新的模型开发范式带来的潜在污染问题。本研究旨在揭示LLM作为法官时由于数据生成器与基于LLM的评估器的相关性所导致的偏好泄露（Preference Leakage）问题。为研究这一问题，我们首先定义了数据生成器LLM与法官LLM之间的三种常见相关性：同为一个模型、有继承关系以及同属一个模型家族。通过广泛的实验，我们实证证实了偏好泄露导致法官倾向于与其相关的“学生”模型的偏差问题，并且这种偏差在多个LLM基线和基准测试中普遍存在。进一步的分析表明，偏好泄露是一个比以前识别的法官偏好偏差更普遍且更难检测的问题。所有这些发现表明，偏好泄露在法官LLM领域是一个普遍存在的、具有挑战性的问题。我们已经将所有代码和数据公开至以下链接：this https URL。 

---
# Towards Safer Chatbots: A Framework for Policy Compliance Evaluation of Custom GPTs 

**Title (ZH)**: 向着更安全的聊天机器人：自定义GPT政策合规性评估框架 

**Authors**: David Rodriguez, William Seymour, Jose M. Del Alamo, Jose Such  

**Link**: [PDF](https://arxiv.org/pdf/2502.01436)  

**Abstract**: Large Language Models (LLMs) have gained unprecedented prominence, achieving widespread adoption across diverse domains and integrating deeply into society. The capability to fine-tune general-purpose LLMs, such as Generative Pre-trained Transformers (GPT), for specific tasks has facilitated the emergence of numerous Custom GPTs. These tailored models are increasingly made available through dedicated marketplaces, such as OpenAI's GPT Store. However, their black-box nature introduces significant safety and compliance risks. In this work, we present a scalable framework for the automated evaluation of Custom GPTs against OpenAI's usage policies, which define the permissible behaviors of these systems. Our framework integrates three core components: (1) automated discovery and data collection of models from the GPT store, (2) a red-teaming prompt generator tailored to specific policy categories and the characteristics of each target GPT, and (3) an LLM-as-a-judge technique to analyze each prompt-response pair for potential policy violations.
We validate our framework with a manually annotated ground truth, and evaluate it through a large-scale study with 782 Custom GPTs across three categories: Romantic, Cybersecurity, and Academic GPTs. Our manual annotation process achieved an F1 score of 0.975 in identifying policy violations, confirming the reliability of the framework's assessments. The results reveal that 58.7% of the analyzed models exhibit indications of non-compliance, exposing weaknesses in the GPT store's review and approval processes. Furthermore, our findings indicate that a model's popularity does not correlate with compliance, and non-compliance issues largely stem from behaviors inherited from base models rather than user-driven customizations. We believe this approach is extendable to other chatbot platforms and policy domains, improving LLM-based systems safety. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经获得了前所未有的重视，广泛应用于各个领域，并深深融入社会之中。对通用语言模型，如生成预训练Transformer（GPT）等进行微调以适应特定任务的能力，促进了大量自定义GPT（Custom GPTs）的出现。这些定制模型越来越通过专门的市场平台（如OpenAI的GPT Store）提供，但其黑箱特征引入了重大的安全和合规风险。在本研究中，我们提出了一种可扩展的框架，用于自动化评估自定义GPTs是否符合OpenAI的使用政策，这些政策规定了这些系统的可接受行为。我们的框架整合了三个核心组件：（1）自动发现和收集GPT Store中的模型数据；（2）针对特定政策类别和每个目标GPT的特征定制的红队提示生成器；（3）将LLM作为法官的技术，分析每对提示-响应对以识别潜在的政策违规行为。

我们通过手动标注的验证数据集验证了该框架，并通过一项大规模研究对782个自定义GPTs进行评估，分为三类：浪漫型GPTs、网络安全型GPTs和学术型GPTs。我们的手动标注过程在识别政策违规方面达到了0.975的F1分数，证明了该框架评估的可靠性。研究结果表明，58.7%的分析模型表现出违规迹象，揭示了GPT Store审核和批准流程中的弱点。此外，我们的研究还发现，模型的流行度与合规性之间没有关联，违规问题主要源自基础模型继承的行为，而不是用户驱动的定制化。我们认为这种方法可以扩展到其他聊天机器人平台和政策领域，从而提高基于LLM系统的安全性。 

---
# RPGBENCH: Evaluating Large Language Models as Role-Playing Game Engines 

**Title (ZH)**: RPGBENCH：评估大型语言模型作为角色扮演游戏引擎的能力 

**Authors**: Pengfei Yu, Dongming Shen, Silin Meng, Jaewon Lee, Weisu Yin, Andrea Yaoyun Cui, Zhenlin Xu, Yi Zhu, Xingjian Shi, Mu Li, Alex Smola  

**Link**: [PDF](https://arxiv.org/pdf/2502.00595)  

**Abstract**: We present RPGBench, the first benchmark designed to evaluate large language models (LLMs) as text-based role-playing game (RPG) engines. RPGBench comprises two core tasks: Game Creation (GC) and Game Simulation (GS). In GC, an LLM must craft a valid and playable RPG world using a structured event-state representation, ensuring logical coherence and proper termination conditions. In GS, the LLM simulates interactive gameplay across multiple rounds while consistently updating states and enforcing game rules. To comprehensively assess performance, RPGBench integrates objective and subjective evaluation methodologies. Objective measures verify adherence to event mechanics and check variable updates without requiring human intervention. Subjective measures, such as content interestingness, action quality, and role-playing capability, are evaluated via an LLM-as-a-judge framework, where a strong LLM grades each candidate's outputs. Empirical results demonstrate that state-of-the-art LLMs can produce engaging stories but often struggle to implement consistent, verifiable game mechanics, particularly in long or complex scenarios. By combining structured, rule-based assessments with LLM-based judgments, RPGBench provides a new standard for evaluating how well LLMs can balance creativity, coherence, and complexity in text-based RPGs, opening avenues for more immersive and controllable interactive storytelling. 

**Abstract (ZH)**: 我们提出了RPGBench，这是首个用于评估大型语言模型（LLMs）作为文本角色扮演游戏（RPG）引擎性能的基准测试。RPGBench 包含两个核心任务：游戏创建（GC）和游戏模拟（GS）。在游戏创建任务（GC）中，LLM 必须使用结构化事件状态表示法创作一个有效的、可玩的RPG世界，确保逻辑连贯性和适当的结束条件。在游戏模拟任务（GS）中，LLM 需要在多轮交互中模拟游戏玩法，同时持续更新状态并遵守游戏规则。为了全面评估性能，RPGBench 整合了客观和主观评价方法。客观指标验证了事件机制的遵循情况，并检查变量更新，无需人工干预。主观指标，如内容趣味性、行为质量以及角色扮演能力，则通过LLM作为评委的框架进行评估，其中强大的LLM会对每个候选者的输出进行评分。实验结果表明，最先进的LLM能够生成引人入胜的故事，但在实现一致性和可验证的游戏机制方面常常遇到困难，特别是在长时间或复杂的场景中。通过结合结构化、规则导向的评估与基于LLM的判断，RPGBench 提供了一个新的标准，用于评估LLM在文本RPG中平衡创造力、连贯性和复杂性的能力，从而为更具沉浸感和可控性的交互叙事打开了新的途径。 

---
# Can AI Solve the Peer Review Crisis? A Large Scale Experiment on LLM's Performance and Biases in Evaluating Economics Papers 

**Title (ZH)**: 人工智能能解决同行评审危机吗？一项大型实验探究LLM在评估经济论文时的表现与偏见 

**Authors**: Pat Pataranutaporn, Nattavudh Powdthavee, Pattie Maes  

**Link**: [PDF](https://arxiv.org/pdf/2502.00070)  

**Abstract**: We investigate whether artificial intelligence can address the peer review crisis in economics by analyzing 27,090 evaluations of 9,030 unique submissions using a large language model (LLM). The experiment systematically varies author characteristics (e.g., affiliation, reputation, gender) and publication quality (e.g., top-tier, mid-tier, low-tier, AI generated papers). The results indicate that LLMs effectively distinguish paper quality but exhibit biases favoring prominent institutions, male authors, and renowned economists. Additionally, LLMs struggle to differentiate high-quality AI-generated papers from genuine top-tier submissions. While LLMs offer efficiency gains, their susceptibility to bias necessitates cautious integration and hybrid peer review models to balance equity and accuracy. 

**Abstract (ZH)**: 我们通过分析9030篇独特提交论文的27,090份评估结果，使用大型语言模型（LLM）来探讨人工智能是否能解决经济学同行评审危机。实验系统地变化了作者特征（如：隶属关系、声誉、性别）以及出版质量（如：顶级、中等级、低等级、AI生成的文章）。结果表明，LLM能够有效区分论文质量，但表现出对知名机构、男性作者和著名经济学家的偏见。此外，LLM难以区分高质量的AI生成论文和真正的顶级论文。虽然LLM提供了效率上的改进，但其易受偏见的影响需要谨慎整合，并采用混合同行评审模式以平衡公平性和准确性。 

---
