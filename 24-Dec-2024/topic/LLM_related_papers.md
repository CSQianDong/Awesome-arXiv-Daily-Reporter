# Leveraging Memory Retrieval to Enhance LLM-based Generative Recommendation 

**Title (ZH)**: 利用记忆检索增强基于大规模语言模型的生成性推荐 

**Authors**: Chengbing Wang, Yang Zhang, Fengbin Zhu, Jizhi Zhang, Tianhao Shi, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2412.17593)  

**Abstract**: Leveraging Large Language Models (LLMs) to harness user-item interaction histories for item generation has emerged as a promising paradigm in generative recommendation. However, the limited context window of LLMs often restricts them to focusing on recent user interactions only, leading to the neglect of long-term interests involved in the longer histories. To address this challenge, we propose a novel Automatic Memory-Retrieval framework (AutoMR), which is capable of storing long-term interests in the memory and extracting relevant information from it for next-item generation within LLMs. Extensive experimental results on two real-world datasets demonstrate the effectiveness of our proposed AutoMR framework in utilizing long-term interests for generative recommendation. 

**Abstract (ZH)**: 利用大规模语言模型（LLMs）捕获用户-项目交互历史以生成项目，在生成推荐中展现出一种有前景的范式。然而，LLMs 的有限上下文窗口经常限制它们仅关注最近的用户交互，而忽略了长时间历史中涉及的长期兴趣。为解决这一挑战，我们提出了一种新的自动记忆检索框架（AutoMR），该框架能够存储长期兴趣并在LLMs中从记忆中提取相关信息用于生成下一个项目。在两个真实世界数据集上的 extensive 实验结果表明，我们的 AutoMR 框架在利用长期兴趣进行生成推荐方面具有有效性。 

---
# SyNeg: LLM-Driven Synthetic Hard-Negatives for Dense Retrieval 

**Title (ZH)**: SyNeg：由大型语言模型驱动的合成硬负例在密集检索中的应用 

**Authors**: Xiaopeng Li, Xiangyang Li, Hao Zhang, Zhaocheng Du, Pengyue Jia, Yichao Wang, Xiangyu Zhao, Huifeng Guo, Ruiming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17250)  

**Abstract**: The performance of Dense retrieval (DR) is significantly influenced by the quality of negative sampling. Traditional DR methods primarily depend on naive negative sampling techniques or on mining hard negatives through external retriever and meticulously crafted strategies. However, naive negative sampling often fails to adequately capture the accurate boundaries between positive and negative samples, whereas existing hard negative sampling methods are prone to false negatives, resulting in performance degradation and training instability. Recent advancements in large language models (LLMs) offer an innovative solution to these challenges by generating contextually rich and diverse negative samples. In this work, we present a framework that harnesses LLMs to synthesize high-quality hard negative samples. We first devise a \textit{multi-attribute self-reflection prompting strategy} to direct LLMs in hard negative sample generation. Then, we implement a \textit{hybrid sampling strategy} that integrates these synthetic negatives with traditionally retrieved negatives, thereby stabilizing the training process and improving retrieval performance. Extensive experiments on five benchmark datasets demonstrate the efficacy of our approach, and code is also publicly available. 

**Abstract (ZH)**: 密集检索（DR）的性能显著受负样本采样质量的影响。传统DR方法主要依赖于朴素的负样本采样技术或通过外部检索器挖掘难以区分的负样本，并采用精心设计的策略。然而，朴素的负样本采样常常未能充分捕捉正负样本之间的准确边界，而现有的难以区分的负样本采样方法则容易产生假的负样本，导致性能下降和训练不稳定。近年来，大型语言模型（LLMs）的发展为解决这些挑战提供了创新的解决方案，通过生成上下文丰富且多样化的负样本。本文提出了一种框架，利用LLMs合成高质量的难以区分的负样本。首先，我们设计了一种\textit{多属性自我反思提示策略}，以指导LLMs生成难以区分的负样本。然后，我们采用了\textit{混合采样策略}，将这些合成的负样本与传统检索得到的负样本结合，从而稳定训练过程并提高检索性能。在五个基准数据集上的大量实验表明了我们方法的有效性，并且相关代码已公开。 

---
# LLM-Powered User Simulator for Recommender System 

**Title (ZH)**: 基于LLM的用户模拟器在推荐系统中的应用 

**Authors**: Zijian Zhang, Shuchang Liu, Ziru Liu, Rui Zhong, Qingpeng Cai, Xiangyu Zhao, Chunxu Zhang, Qidong Liu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16984)  

**Abstract**: User simulators can rapidly generate a large volume of timely user behavior data, providing a testing platform for reinforcement learning-based recommender systems, thus accelerating their iteration and optimization. However, prevalent user simulators generally suffer from significant limitations, including the opacity of user preference modeling and the incapability of evaluating simulation accuracy. In this paper, we introduce an LLM-powered user simulator to simulate user engagement with items in an explicit manner, thereby enhancing the efficiency and effectiveness of reinforcement learning-based recommender systems training. Specifically, we identify the explicit logic of user preferences, leverage LLMs to analyze item characteristics and distill user sentiments, and design a logical model to imitate real human engagement. By integrating a statistical model, we further enhance the reliability of the simulation, proposing an ensemble model that synergizes logical and statistical insights for user interaction simulations. Capitalizing on the extensive knowledge and semantic generation capabilities of LLMs, our user simulator faithfully emulates user behaviors and preferences, yielding high-fidelity training data that enrich the training of recommendation algorithms. We establish quantifying and qualifying experiments on five datasets to validate the simulator's effectiveness and stability across various recommendation scenarios. 

**Abstract (ZH)**: 用户模拟器可以快速生成大量及时的用户行为数据，为基于强化学习的推荐系统提供测试平台，从而加快它们的迭代和优化。然而，现有的用户模拟器通常存在显著的局限性，包括用户偏好建模的不透明性和评估模拟准确性的能力不足。本文引入了基于大语言模型（LLM）的用户模拟器，以明确的方式模拟用户与项目之间的互动，进而提升基于强化学习的推荐系统训练的效率和效果。具体来说，我们识别出用户的明确偏好逻辑，利用大语言模型分析项目特征并提取用户情感，并设计一个逻辑模型来模仿真实的人类互动。通过整合统计模型，进一步增强了模拟的可靠性，提出了一种综合逻辑和统计洞察的集成模型，用于用户互动模拟。借助大语言模型的广泛知识和语义生成能力，我们的用户模拟器忠实模拟了用户行为和偏好，生成了高保真的训练数据，丰富了推荐算法的训练。我们通过在五个数据集上进行量化和定性实验来验证模拟器在各种推荐场景下的有效性和稳定性。 

---
# Towards a Unified Paradigm: Integrating Recommendation Systems as a New Language in Large Models 

**Title (ZH)**: 向着统一范式的迈进：将推荐系统纳入大型模型的新语言 

**Authors**: Kai Zheng, Qingfeng Sun, Can Xu, Peng Yu, Qingwei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2412.16933)  

**Abstract**: This paper explores the use of Large Language Models (LLMs) for sequential recommendation, which predicts users' future interactions based on their past behavior. We introduce a new concept, "Integrating Recommendation Systems as a New Language in Large Models" (RSLLM), which combines the strengths of traditional recommenders and LLMs. RSLLM uses a unique prompting method that combines ID-based item embeddings from conventional recommendation models with textual item features. It treats users' sequential behaviors as a distinct language and aligns the ID embeddings with the LLM's input space using a projector. We also propose a two-stage LLM fine-tuning framework that refines a pretrained LLM using a combination of two contrastive losses and a language modeling loss. The LLM is first fine-tuned using text-only prompts, followed by target domain fine-tuning with unified prompts. This trains the model to incorporate behavioral knowledge from the traditional sequential recommender into the LLM. Our empirical results validate the effectiveness of our proposed framework. 

**Abstract (ZH)**: 本文探讨了大型语言模型（LLMs）在序列推荐领域的应用，该模型根据用户的过往行为预测其未来互动。我们提出了一种新的概念，即“将推荐系统作为大型模型中的新语言”（RSLLM，推荐系统语言模型），该概念结合了传统推荐系统和LLMs的优势。RSLLM采用了一种独特的提示方法，将传统推荐模型中的基于ID的项目嵌入与文本项目特征结合在一起。它将用户的行为序列视为一种独特的语言，并通过投影器将ID嵌入与LLM的输入空间对齐。我们还提出了一种两阶段的LLM微调框架，该框架利用两种对比损失和语言模型损失的组合对预训练的LLM进行微调。首先使用仅文本的提示方法进行LLM的微调，然后使用统一的提示方法进行目标领域微调。这训练模型将传统序列推荐器中的行为知识融入到LLM中。我们的实证结果验证了所提框架的有效性。 

---
# Enhancing Supply Chain Transparency in Emerging Economies Using Online Contents and LLMs 

**Title (ZH)**: 使用在线内容和大规模语言模型增强新兴经济体的供应链透明度 

**Authors**: Bohan Jin, Qianyou Sun, Lihua Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.16922)  

**Abstract**: In the current global economy, supply chain transparency plays a pivotal role in ensuring this security by enabling companies to monitor supplier performance and fostering accountability and responsibility. Despite the advancements in supply chain relationship datasets like Bloomberg and FactSet, supply chain transparency remains a significant challenge in emerging economies due to issues such as information asymmetry and institutional gaps in regulation. This study proposes a novel approach to enhance supply chain transparency in emerging economies by leveraging online content and large language models (LLMs). We develop a Supply Chain Knowledge Graph Mining System that integrates advanced LLMs with web crawler technology to automatically collect and analyze supply chain information. The system's effectiveness is validated through a case study focusing on the semiconductor supply chain, a domain that has recently gained significant attention due to supply chain risks. Our results demonstrate that the proposed system provides greater applicability for emerging economies, such as mainland China, complementing the data gaps in existing datasets. However, challenges including the accurate estimation of monetary and material flows, the handling of time series data, synonyms disambiguation, and mitigating biases from online contents still remains. Future research should focus on addressing these issues to further enhance the system's capabilities and broaden its application to other emerging economies and industries. 

**Abstract (ZH)**: 在当前全球经济中，供应链透明度在确保安全方面扮演着至关重要的角色，通过使公司能够监控供应商表现并促进问责制和责任感。尽管Bloomberg和FactSet等供应链关系数据集有所进步，但由于信息不对称和监管制度的空缺等因素，供应链透明度在新兴经济体中仍是一个重大挑战。本研究提出了一种新颖的方法，通过利用在线内容和大型语言模型（LLMs）来增强新兴经济体中的供应链透明度。我们开发了一个供应链知识图谱挖掘系统，该系统将先进的LLMs与网络爬虫技术相结合，以自动收集和分析供应链信息。通过一个专注于半导体供应链的案例研究，证明了该系统的有效性，而半导体供应链因其供应风险问题已成为近期关注的焦点。我们的研究结果表明，所提出的系统在填补现有数据集的数据空白方面具有更广泛的适用性，例如中国大陆等新兴经济体。然而，仍存在诸如货币和物质流动的准确估计、时间序列数据的处理、同义词消歧及减轻在线内容偏见等挑战。未来的研究应关注解决这些问题，以进一步提高系统的功能，并将其应用扩展到其他新兴经济体和行业。 

---
# Towards More Robust Retrieval-Augmented Generation: Evaluating RAG Under Adversarial Poisoning Attacks 

**Title (ZH)**: 面向更稳健的检索增强生成：评估对抗中毒攻击下的RAG性能 

**Authors**: Jinyan Su, Jin Peng Zhou, Zhengxin Zhang, Preslav Nakov, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2412.16708)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems have emerged as a promising solution to mitigate LLM hallucinations and enhance their performance in knowledge-intensive domains. However, these systems are vulnerable to adversarial poisoning attacks, where malicious passages injected into retrieval databases can mislead the model into generating factually incorrect outputs. In this paper, we investigate both the retrieval and the generation components of RAG systems to understand how to enhance their robustness against such attacks. From the retrieval perspective, we analyze why and how the adversarial contexts are retrieved and assess how the quality of the retrieved passages impacts downstream generation. From a generation perspective, we evaluate whether LLMs' advanced critical thinking and internal knowledge capabilities can be leveraged to mitigate the impact of adversarial contexts, i.e., using skeptical prompting as a self-defense mechanism. Our experiments and findings provide actionable insights into designing safer and more resilient retrieval-augmented frameworks, paving the way for their reliable deployment in real-world applications. 

**Abstract (ZH)**: 检索增强生成（RAG）系统作为一种减轻大规模语言模型（LLM）幻觉并提高其在知识密集型领域性能的有前景的解决方案而崭露头角。然而，这些系统容易受到对抗性污染攻击的影响，在这些攻击中，恶意段落被注入检索数据库，可能导致模型生成事实不正确的输出。在本文中，我们研究了RAG系统的检索和生成组件，以了解如何增强其对这些攻击的鲁棒性。从检索的角度来看，我们分析了为什么以及如何检索到对抗性上下文，并评估了检索到段落质量对后续生成的影响。从生成的角度来看，我们评估了是否可以利用LLM的高级批判性思维和内部知识能力来减轻对抗性上下文的影响，即使用怀疑性提示作为一种自我防御机制。我们的实验和发现为设计更安全、更稳健的检索增强框架提供了可操作的见解，铺平了其实用部署在实际应用中的道路。 

---
# Large Language Model Can Be a Foundation for Hidden Rationale-Based Retrieval 

**Title (ZH)**: 大规模语言模型可以作为基于隐藏推理的检索的基础 

**Authors**: Luo Ji, Feixiang Guo, Teng Chen, Qingqing Gu, Xiaoyu Wang, Ningyuan Xi, Yihong Wang, Peng Yu, Yue Zhao, Hongyang Lei, Zhonglin Jiang, Yong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.16615)  

**Abstract**: Despite the recent advancement in Retrieval-Augmented Generation (RAG) systems, most retrieval methodologies are often developed for factual retrieval, which assumes query and positive documents are semantically similar. In this paper, we instead propose and study a more challenging type of retrieval task, called hidden rationale retrieval, in which query and document are not similar but can be inferred by reasoning chains, logic relationships, or empirical experiences. To address such problems, an instruction-tuned Large language model (LLM) with a cross-encoder architecture could be a reasonable choice. To further strengthen pioneering LLM-based retrievers, we design a special instruction that transforms the retrieval task into a generative task by prompting LLM to answer a binary-choice question. The model can be fine-tuned with direct preference optimization (DPO). The framework is also optimized for computational efficiency with no performance degradation. We name this retrieval framework by RaHoRe and verify its zero-shot and fine-tuned performance superiority on Emotional Support Conversation (ESC), compared with previous retrieval works. Our study suggests the potential to employ LLM as a foundation for a wider scope of retrieval tasks. Our codes, models, and datasets are available on this https URL. 

**Abstract (ZH)**: 尽管最近在检索增强生成（RAG）系统方面取得了一定进展，大多数检索方法通常针对事实检索进行开发，这种检索方法假设查询和正相关文档在语义上是相似的。在本文中，我们反而提出并研究了一种更具挑战性的检索任务类型，称为隐藏理由检索，在这种任务中，查询和文档本身并不相似，但可以通过推理链、逻辑关系或经验进行推断。为了解决这类问题，带有交叉编码器架构的指令调优大型语言模型（LLM）可能是一个合理的选择。为了进一步增强基于LLM的检索者，我们设计了一个特殊的指令，通过提示LLM回答二元选择问题，将检索任务转化为生成任务。该模型可以通过直接偏好优化（DPO）进行微调。该框架还在保证性能不降级的情况下优化了计算效率。我们将这种检索框架命名为RaHoRe，并通过将其与先前的检索工作在情感支持对话（ESC）上的零样本和微调性能进行比较，验证了其优越性。我们的研究表明，大型语言模型有可能作为更广泛检索任务的基础。我们的代码、模型和数据集可通过以下链接获取：[此处链接]。 

---
# Enhancing Item Tokenization for Generative Recommendation through Self-Improvement 

**Title (ZH)**: 通过自我提升提高项目token化在生成性推荐中的效果 

**Authors**: Runjin Chen, Mingxuan Ju, Ngoc Bui, Dimosthenis Antypas, Stanley Cai, Xiaopeng Wu, Leonardo Neves, Zhangyang Wang, Neil Shah, Tong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.17171)  

**Abstract**: Generative recommendation systems, driven by large language models (LLMs), present an innovative approach to predicting user preferences by modeling items as token sequences and generating recommendations in a generative manner. A critical challenge in this approach is the effective tokenization of items, ensuring that they are represented in a form compatible with LLMs. Current item tokenization methods include using text descriptions, numerical strings, or sequences of discrete tokens. While text-based representations integrate seamlessly with LLM tokenization, they are often too lengthy, leading to inefficiencies and complicating accurate generation. Numerical strings, while concise, lack semantic depth and fail to capture meaningful item relationships. Tokenizing items as sequences of newly defined tokens has gained traction, but it often requires external models or algorithms for token assignment. These external processes may not align with the LLM's internal pretrained tokenization schema, leading to inconsistencies and reduced model performance. To address these limitations, we propose a self-improving item tokenization method that allows the LLM to refine its own item tokenizations during training process. Our approach starts with item tokenizations generated by any external model and periodically adjusts these tokenizations based on the LLM's learned patterns. Such alignment process ensures consistency between the tokenization and the LLM's internal understanding of the items, leading to more accurate recommendations. Furthermore, our method is simple to implement and can be integrated as a plug-and-play enhancement into existing generative recommendation systems. Experimental results on multiple datasets and using various initial tokenization strategies demonstrate the effectiveness of our method, with an average improvement of 8\% in recommendation performance. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的生成型推荐系统通过将物品表示为标记序列并以生成的方式生成推荐，提供了一种创新的方法来预测用户偏好。这种方法的关键挑战是如何有效地对物品进行标记化，确保其以与LLMs兼容的形式表示。当前的物品标记化方法包括使用文本描述、数值字符串或离散标记序列。基于文本的表示与LLMs的标记化无缝集成，但通常太冗长，导致效率低下且使准确生成变得复杂。数值字符串虽然简洁，但缺乏语义深度，无法捕捉有意义的物品关系。将物品标记化为新定义的标记序列的方法得到了越来越多的关注，但通常需要外部模型或算法来分配标记。这些外部过程可能与LLMs的内部预训练标记化方案不一致，导致不一致性和模型性能下降。为了解决这些局限性，我们提出了一种自我提升的物品标记化方法，允许LLMs在训练过程中改进其自身的物品标记化。我们的方法从使用任何外部模型生成的物品标记开始，并根据LLMs学到的模式定期调整这些标记。这一对齐过程确保标记化与LLMs对物品内部理解的一致性，从而提高推荐准确性。此外，我们的方法易于实现，并可以作为即插即用功能集成到现有的生成型推荐系统中。在多项数据集上进行的实验和使用各种初始标记化策略的实验结果表明，我们的方法非常有效，平均推荐性能提高了8%。 

---
# HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases 

**Title (ZH)**: HybGRAG：面向文本和关系知识库的混合检索增强生成 

**Authors**: Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen Han, Soji Adeshina, Vassilis N. Ioannidis, Huzefa Rangwala, Christos Faloutsos  

**Link**: [PDF](https://arxiv.org/pdf/2412.16311)  

**Abstract**: Given a semi-structured knowledge base (SKB), where text documents are interconnected by relations, how can we effectively retrieve relevant information to answer user questions? Retrieval-Augmented Generation (RAG) retrieves documents to assist large language models (LLMs) in question answering; while Graph RAG (GRAG) uses structured knowledge bases as its knowledge source. However, many questions require both textual and relational information from SKB - referred to as "hybrid" questions - which complicates the retrieval process and underscores the need for a hybrid retrieval method that leverages both information. In this paper, through our empirical analysis, we identify key insights that show why existing methods may struggle with hybrid question answering (HQA) over SKB. Based on these insights, we propose HybGRAG for HQA consisting of a retriever bank and a critic module, with the following advantages: (1) Agentic, it automatically refines the output by incorporating feedback from the critic module, (2) Adaptive, it solves hybrid questions requiring both textual and relational information with the retriever bank, (3) Interpretable, it justifies decision making with intuitive refinement path, and (4) Effective, it surpasses all baselines on HQA benchmarks. In experiments on the STaRK benchmark, HybGRAG achieves significant performance gains, with an average relative improvement in Hit@1 of 51%. 

**Abstract (ZH)**: 给定一个半结构化知识库（SKB），其中文本文档通过关系相互连接，如何有效地检索相关信息以回答用户问题？检索增强生成（RAG）通过检索文档来辅助大型语言模型（LLMs）进行问答；而Graph RAG（GRAG）利用结构化知识库作为其知识来源。然而，许多问题需要从SKB中同时获取文本和关系信息，这些被称为“混合”问题，这使得检索过程复杂化，并强调了需要一种结合这两种信息的混合检索方法。在本文中，通过我们的实证分析，我们识别出关键见解，展示了为什么现有方法可能难以处理SKB上的混合问答（HQA）。基于这些见解，我们提出了一种名为HybGRAG的方法来解决HQA，其包含检索库和批判模块，具有以下优势：（1）自主性，通过批判模块的反馈自动优化输出；（2）自适应性，使用检索库解决需要同时处理文本和关系信息的混合问题；（3）可解释性，通过直观的优化路径来解释决策过程；（4）有效性，在HQA基准测试中，HybGRAG取得了显著性能提升，平均改进精度（Hit@1）为51%。

相关实验在STaRK基准测试上验证了HybGRAG的有效性。结果显示，HybGRAG在HQA基准测试中的表现显著优于所有基线方法，平均相对改进精度（Hit@1）达到了51%。 

---
# LLM4AD: A Platform for Algorithm Design with Large Language Model 

**Title (ZH)**: LLM4AD：一种基于大规模语言模型的算法设计平台 

**Authors**: Fei Liu, Rui Zhang, Zhuoliang Xie, Rui Sun, Kai Li, Xi Lin, Zhenkun Wang, Zhichao Lu, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17287)  

**Abstract**: We introduce LLM4AD, a unified Python platform for algorithm design (AD) with large language models (LLMs). LLM4AD is a generic framework with modularized blocks for search methods, algorithm design tasks, and LLM interface. The platform integrates numerous key methods and supports a wide range of algorithm design tasks across various domains including optimization, machine learning, and scientific discovery. We have also designed a unified evaluation sandbox to ensure a secure and robust assessment of algorithms. Additionally, we have compiled a comprehensive suite of support resources, including tutorials, examples, a user manual, online resources, and a dedicated graphical user interface (GUI) to enhance the usage of LLM4AD. We believe this platform will serve as a valuable tool for fostering future development in the merging research direction of LLM-assisted algorithm design. 

**Abstract (ZH)**: 我们将介绍LLM4AD，这是一个基于大型语言模型（LLMs）的统一Python平台，用于算法设计（AD）。LLM4AD 是一个通用框架，包含模块化的搜索方法、算法设计任务和LLM接口块。该平台集成了众多关键方法，并支持各种领域的广泛算法设计任务，包括优化、机器学习和科学发现。我们还设计了一个统一的评估沙盒，以确保对算法进行安全和稳健的评估。此外，我们编制了一整套支持资源，包括教程、示例、用户手册、在线资源和一个专用的图形用户界面（GUI），以增强LLM4AD的使用体验。我们相信，这一平台将为LLM辅助算法设计这一交叉研究方向的未来发展提供宝贵的工具。 

---
# Better Think with Tables: Leveraging Tables to Enhance Large Language Model Comprehension 

**Title (ZH)**: 增强大型语言模型理解能力：利用表格进行更有效的思考 

**Authors**: Jio Oh, Geon Heo, Seungjun Oh, Jindong Wang, Xing Xie, Steven Euijong Whang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17189)  

**Abstract**: Despite the recent advancement of Large Langauge Models (LLMs), they struggle with complex queries often involving multiple conditions, common in real-world scenarios. We propose Thinking with Tables, a technique that assists LLMs to leverage tables for intermediate thinking aligning with human cognitive behavior. By introducing a pre-instruction that triggers an LLM to organize information in tables, our approach achieves a 40.29\% average relative performance increase, higher robustness, and show generalizability to different requests, conditions, or scenarios. We additionally show the influence of data structuredness for the model by comparing results from four distinct structuring levels that we introduce. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在最近取得了进展，它们在处理涉及多个条件的复杂查询时仍然存在困难，而这类查询在现实场景中相当常见。我们提出了一种“思考表格化”的技术，旨在帮助LLMs利用表格进行中间推理，这种推理方式与人类的认知行为相一致。通过引入一种预指令，促使LLMs将信息组织成表格形式，我们的方法实现了40.29%的平均相对性能提升，并且具备更高的稳健性和较强的一般适用性，能够应用于不同的请求、条件或场景。此外，我们通过对比四种不同组织层次的数据结果，进一步展示了数据结构化对模型影响的作用。 

---
# LLM Agent for Fire Dynamics Simulations 

**Title (ZH)**: 用于火灾动力学模拟的大型语言模型代理 

**Authors**: Leidong Xu, Danyal Mohaddes, Yi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17146)  

**Abstract**: Significant advances have been achieved in leveraging foundation models, such as large language models (LLMs), to accelerate complex scientific workflows. In this work we introduce FoamPilot, a proof-of-concept LLM agent designed to enhance the usability of FireFOAM, a specialized solver for fire dynamics and fire suppression simulations built using OpenFOAM, a popular open-source toolbox for computational fluid dynamics (CFD). FoamPilot provides three core functionalities: code insight, case configuration and simulation evaluation. Code insight is an alternative to traditional keyword searching leveraging retrieval-augmented generation (RAG) and aims to enable efficient navigation and summarization of the FireFOAM source code for developers and experienced users. For case configuration, the agent interprets user requests in natural language and aims to modify existing simulation setups accordingly to support intermediate users. FoamPilot's job execution functionality seeks to manage the submission and execution of simulations in high-performance computing (HPC) environments and provide preliminary analysis of simulation results to support less experienced users. Promising results were achieved for each functionality, particularly for simple tasks, and opportunities were identified for significant further improvement for more complex tasks. The integration of these functionalities into a single LLM agent is a step aimed at accelerating the simulation workflow for engineers and scientists employing FireFOAM for complex simulations critical for improving fire safety. 

**Abstract (ZH)**: 在利用大型语言模型（LLM）等基础模型加速复杂科学工作流方面取得了显著进展。本文介绍了FoamPilot，这是一种概念验证的LLM代理，旨在提升FireFOAM的易用性，FireFOAM是一个基于OpenFOAM构建的专用求解器，用于火灾动力学和灭火模拟。OpenFOAM是一个流行的开源计算流体动力学（CFD）工具箱。FoamPilot提供了三个核心功能：代码洞察、案例配置和仿真评估。代码洞察利用检索增强生成（RAG）作为一种替代传统的关键词搜索的方法，旨在使开发人员和有经验的用户能够高效地导航和总结FireFOAM的源代码。在案例配置方面，代理以自然语言解释用户请求，并旨在相应地修改现有的仿真设置，以支持中级用户。FoamPilot的任务执行功能旨在管理仿真在高性能计算（HPC）环境中的提交和执行，并对仿真结果进行初步分析，以支持经验较少的用户。对于每个功能，特别是简单任务，我们取得了令人鼓舞的结果，并且识别出了在更复杂任务上进行重大改进的机会。将这些功能集成到一个单一的LLM代理中，是一个旨在加速使用FireFOAM进行复杂仿真（这对于提高火灾安全性至关重要）的工程师和科学家的仿真工作流的步骤。 

---
# SubstationAI: Multimodal Large Model-Based Approaches for Analyzing Substation Equipment Faults 

**Title (ZH)**: SubstationAI：基于多模态大规模模型的方法用于分析变电站设备故障 

**Authors**: Jinzhi Wang, Qinfeng Song, Lidong Qian, Haozhou Li, Qinke Peng, Jiangbo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17077)  

**Abstract**: The reliability of substation equipment is crucial to the stability of power systems, but traditional fault analysis methods heavily rely on manual expertise, limiting their effectiveness in handling complex and large-scale data. This paper proposes a substation equipment fault analysis method based on a multimodal large language model (MLLM). We developed a database containing 40,000 entries, including images, defect labels, and analysis reports, and used an image-to-video generation model for data augmentation. Detailed fault analysis reports were generated using GPT-4. Based on this database, we developed SubstationAI, the first model dedicated to substation fault analysis, and designed a fault diagnosis knowledge base along with knowledge enhancement methods. Experimental results show that SubstationAI significantly outperforms existing models, such as GPT-4, across various evaluation metrics, demonstrating higher accuracy and practicality in fault cause analysis, repair suggestions, and preventive measures, providing a more advanced solution for substation equipment fault analysis. 

**Abstract (ZH)**: 变电站设备的可靠性对于电力系统稳定性至关重要，但传统故障分析方法严重依赖人工专业知识，限制了其在处理复杂和大规模数据方面的有效性。本文提出了一种基于多模态大语言模型（MLLM）的变电站设备故障分析方法。我们建立了一个包含40,000条记录的数据库，包括图像、缺陷标签和分析报告，并使用图像到视频生成模型进行数据增强。使用GPT-4生成了详细故障分析报告。基于该数据库，我们开发了SubstationAI——第一个专用于变电站故障分析的模型，并设计了故障诊断知识库及知识增强方法。实验结果表明，SubstationAI在各种评估指标上显著优于现有的模型（如GPT-4），在故障原因分析、维修建议和预防措施方面展现出更高的准确性和实用性，为变电站设备故障分析提供了更高级的解决方案。 

---
# System-2 Mathematical Reasoning via Enriched Instruction Tuning 

**Title (ZH)**: 系统2数学推理通过丰富指导调优 

**Authors**: Huanqia Cai, Yijun Yang, Zhifeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.16964)  

**Abstract**: Solving complex mathematical problems via system-2 reasoning is a natural human skill, yet it remains a significant challenge for current large language models (LLMs). We identify the scarcity of deliberate multi-step reasoning data as a primary limiting factor. To this end, we introduce Enriched Instruction Tuning (EIT), a method that enriches existing human-annotated mathematical datasets by synergizing human and AI feedback to create fine-grained reasoning trajectories. These datasets are then used to fine-tune open-source LLMs, enhancing their mathematical reasoning abilities without reliance on any symbolic verification program. Concretely, EIT is composed of two critical steps: Enriching with Reasoning Plan (ERP) and Enriching with Reasoning Step (ERS). The former generates a high-level plan that breaks down complex instructions into a sequence of simpler objectives, while ERS fills in reasoning contexts often overlooked by human annotators, creating a smoother reasoning trajectory for LLM fine-tuning. Unlike existing CoT prompting methods that generate reasoning chains only depending on LLM's internal knowledge, our method leverages human-annotated initial answers as ``meta-knowledge'' to help LLMs generate more detailed and precise reasoning processes, leading to a more trustworthy LLM expert for complex mathematical problems. In experiments, EIT achieves an accuracy of 84.1\% on GSM8K and 32.5\% on MATH, surpassing state-of-the-art fine-tuning and prompting methods, and even matching the performance of tool-augmented methods. 

**Abstract (ZH)**: 通过系统2推理解决复杂的数学问题是人类的一项自然技能，但目前仍然是现有大规模语言模型（LLMs）的一项重大挑战。我们识别出缺乏刻意的多步推理数据是主要限制因素。为此，我们提出了一种名为增强指令调优（Enriched Instruction Tuning, EIT）的方法，这种方法通过结合人类和AI反馈来丰富现有的手工标注数学数据集，从而创建精细的推理轨迹。这些数据集随后被用来微调开源的LLMs，以增强其数学推理能力，而不依赖于任何符号验证程序。具体而言，EIT 包括两个关键步骤：增强推理计划（Enriching with Reasoning Plan, ERP）和增强推理步骤（Enriching with Reasoning Step, ERS）。前一步骤生成一个高层次的计划，将复杂的指令分解为一系列简单的目标，而后一步骤则填补人类注释者经常忽略的推理上下文，为LLM微调创建更平滑的推理轨迹。与现有的仅依赖LLM内部知识生成推理链的CoT提示方法不同，我们的方法利用手工标注的初始答案作为“元知识”来帮助LLM生成更详细和精确的推理过程，从而为复杂数学问题提供更可信的LLM专家。在实验中，EIT在GSM8K上的准确率为84.1%，在MATH上的准确率为32.5%，超越了最先进的微调和提示方法，并且匹配工具辅助方法的表现。 

---
# PsychAdapter: Adapting LLM Transformers to Reflect Traits, Personality and Mental Health 

**Title (ZH)**: PsychAdapter: 将大语言模型变换器适应以反映人格特质、个性和心理健康 

**Authors**: Huy Vu, Huy Anh Nguyen, Adithya V Ganesan, Swanie Juhng, Oscar N.E. Kjell, Joao Sedoc, Margaret L. Kern, Ryan L. Boyd, Lyle Ungar, H. Andrew Schwartz, Johannes C. Eichstaedt  

**Link**: [PDF](https://arxiv.org/pdf/2412.16882)  

**Abstract**: Artificial intelligence-based language generators are now a part of most people's lives. However, by default, they tend to generate "average" language without reflecting the ways in which people differ. Here, we propose a lightweight modification to the standard language model transformer architecture - "PsychAdapter" - that uses empirically derived trait-language patterns to generate natural language for specified personality, demographic, and mental health characteristics (with or without prompting). We applied PsychAdapters to modify OpenAI's GPT-2, Google's Gemma, and Meta's Llama 3 and found generated text to reflect the desired traits. For example, expert raters evaluated PsychAdapter's generated text output and found it matched intended trait levels with 87.3% average accuracy for Big Five personalities, and 96.7% for depression and life satisfaction. PsychAdapter is a novel method to introduce psychological behavior patterns into language models at the foundation level, independent of prompting, by influencing every transformer layer. This approach can create chatbots with specific personality profiles, clinical training tools that mirror language associated with psychological conditionals, and machine translations that match an authors reading or education level without taking up LLM context windows. PsychAdapter also allows for the exploration psychological constructs through natural language expression, extending the natural language processing toolkit to study human psychology. 

**Abstract (ZH)**: 基于人工智能的语言生成器已融入了大多数人的生活中。然而，默认情况下，它们倾向于生成“平均水平”的语言，而不反映人们之间的差异。在此，我们提出了一种对标准语言模型变换器架构的轻量级修改——“PsychAdapter”——该架构利用验证过的特质-语言模式来生成符合特定人格、人口统计学和心理健康特征的自然语言（有或无提示）。我们应用PsychAdapter对OpenAI的GPT-2、Google的Gemma和Meta的Llama 3进行了修改，并发现生成的文本反映了所需的特质。例如，专家评分者评估了PsychAdapter生成的文本输出，并发现它在五大人格特质方面的匹配平均准确率为87.3%，在抑郁和生活满意度方面的匹配准确率为96.7%。PsychAdapter是一种新颖的方法，可以在基础层面将心理学行为模式引入语言模型中，而无需提示，并通过影响每个变换器层来实现这一点。这种方法可以创建具有特定人格特征的聊天机器人，制作能够反映心理健康状况的语言的临床训练工具，以及匹配作者阅读水平或教育水平的机器翻译，而无需占用LLM的上下文窗口。此外，PsychAdapter还允许通过自然语言表达探索心理结构，扩展自然语言处理工具包以研究人类心理学。 

---
# OpenRFT: Adapting Reasoning Foundation Model for Domain-specific Tasks with Reinforcement Fine-Tuning 

**Title (ZH)**: OpenRFT：通过强化微调适应领域特定任务的推理基础模型 

**Authors**: Yuxiang Zhang, Yuqi Yang, Jiangming Shu, Yuhang Wang, Jinlin Xiao, Jitao Sang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16849)  

**Abstract**: OpenAI's recent introduction of Reinforcement Fine-Tuning (RFT) showcases the potential of reasoning foundation model and offers a new paradigm for fine-tuning beyond simple pattern imitation. This technical report presents \emph{OpenRFT}, our attempt to fine-tune generalist reasoning models for domain-specific tasks under the same settings as RFT. OpenRFT addresses two key challenges of lacking reasoning step data and the limited quantity of training samples, by leveraging the domain-specific samples in three ways: question augmentation, synthesizing reasoning-process data, and few-shot ICL. The evaluation is conducted on SciKnowEval, where OpenRFT achieves notable performance gains with only $100$ domain-specific samples for each task. More experimental results will be updated continuously in later versions. Source codes, datasets, and models are disclosed at: this https URL 

**Abstract (ZH)**: OpenAI最近推出的增强学习微调（Reinforcement Fine-Tuning，RFT）展示了推理基础模型的潜力，并提供了一种超越简单模式模仿的新微调范式。本技术报告介绍了一种名为\emph{OpenRFT}的尝试，我们在相同的RFT设置下对通用推理模型进行微调以适应特定领域任务。OpenRFT通过三种方式利用领域特定样本来应对缺乏推理步骤数据和训练样本量有限的两个关键挑战：问题增强、合成推理过程数据和少样本ICL（Instance Claire Learning）。

在SciKnowEval上进行了评估，结果显示，OpenRFT仅使用每个任务100个领域特定样本，就能显著提高性能。后续版本中将陆续更新更多实验结果。源代码、数据集和模型将在以下网址公开：this https URL 

---
# Online Learning from Strategic Human Feedback in LLM Fine-Tuning 

**Title (ZH)**: 在线学习：战略人类反馈在大语言模型微调中的应用 

**Authors**: Shugang Hao, Lingjie Duan  

**Link**: [PDF](https://arxiv.org/pdf/2412.16834)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has become an essential step in fine-tuning large language models (LLMs) to align them with human preferences. However, human labelers are selfish and have diverse preferences. They may strategically misreport their online feedback to influence the system's aggregation towards their own preferences. Current practice simply averages labelers' feedback per time and fails to identify the most accurate human labeler, leading to linear regret $\mathcal{O}(T)$ for $T$ time slots. To our best knowledge, we are the first to study online learning mechanisms against strategic human labelers in the LLM fine-tuning process. We formulate a new dynamic Bayesian game and dynamically adjust human labelers' weights in the preference aggregation, ensuring their truthful feedback and sublinear regret $\mathcal{O}(T^{1/2})$. Simulation results demonstrate our mechanism's great advantages over the existing benchmark schemes. 

**Abstract (ZH)**: 从人类反馈进行强化学习（Reinforcement Learning from Human Feedback，RLHF）已经成为将大型语言模型（Large Language Models，LLMs）调整到与人类偏好一致的重要步骤。然而，人类标注者往往自私并具有多元化的偏好，他们可能会战略性地错误报告在线反馈，以影响系统汇总结果以符合自己的偏好。当前的做法是简单地对每个时间点的标注反馈进行平均，未能识别出最准确的人类标注者，导致线性后悔 $\mathcal{O}(T)$，其中 $T$ 表示时间槽的数量。据我们所知，我们是第一个在LLM微调过程中研究对抗战略性人类标注者的在线学习机制的研究。我们构建了一个新的动态贝叶斯博弈模型，并动态调整人类标注者在偏好汇总中的权重，确保其真实反馈并实现亚线性后悔 $\mathcal{O}(T^{1/2})$。仿真结果证明了我们机制相对于现有基准方案的巨大优势。 

---
# KG4Diagnosis: A Hierarchical Multi-Agent LLM Framework with Knowledge Graph Enhancement for Medical Diagnosis 

**Title (ZH)**: KG4Diagnosis：一种带有知识图谱增强的分层多代理大语言模型框架用于医疗诊断 

**Authors**: Kaiwen Zuo, Yirui Jiang, Fan Mo, Pietro Lio  

**Link**: [PDF](https://arxiv.org/pdf/2412.16833)  

**Abstract**: Integrating Large Language Models (LLMs) in healthcare diagnosis demands systematic frameworks that can handle complex medical scenarios while maintaining specialized expertise. We present KG4Diagnosis, a novel hierarchical multi-agent framework that combines LLMs with automated knowledge graph construction, encompassing 362 common diseases across medical specialties. Our framework mirrors real-world medical systems through a two-tier architecture: a general practitioner (GP) agent for initial assessment and triage, coordinating with specialized agents for in-depth diagnosis in specific domains. The core innovation lies in our end-to-end knowledge graph generation methodology, incorporating: (1) semantic-driven entity and relation extraction optimized for medical terminology, (2) multi-dimensional decision relationship reconstruction from unstructured medical texts, and (3) human-guided reasoning for knowledge expansion. KG4Diagnosis serves as an extensible foundation for specialized medical diagnosis systems, with capabilities to incorporate new diseases and medical knowledge. The framework's modular design enables seamless integration of domain-specific enhancements, making it valuable for developing targeted medical diagnosis systems. We provide architectural guidelines and protocols to facilitate adoption across medical contexts. 

**Abstract (ZH)**: 将大型语言模型（LLMs）集成到医疗诊断中，需要系统性的框架来处理复杂的医疗场景并保持专业化的专业知识。我们提出了KG4Diagnosis，这是一种新颖的分层多智能体框架，将LLMs与自动知识图谱构建相结合，涵盖了362种常见疾病，涉及多个医学领域。该框架通过二层架构模拟现实世界的医疗系统：初级诊断（GP）智能体负责初始评估和分流，协调特定领域的专门智能体进行深入诊断。核心创新在于我们端到端的知识图谱生成方法，包括：（1）基于语义的实体和关系抽取，优化用于医学术语，（2）从非结构化医学文本中重构多维度的决策关系，（3）人引导的推理以扩展知识。KG4Diagnosis 作为一个可扩展的基础框架，具有纳入新疾病和医学知识的能力。该框架的模块化设计使其能够无缝集成特定领域的增强功能，使其适用于开发针对性的医疗诊断系统。我们提供了构架指南和协议，以便在不同医疗场景中促进其采纳。 

---
# Argumentation Computation with Large Language Models : A Benchmark Study 

**Title (ZH)**: 大规模语言模型中的论辩计算：一项基准研究 

**Authors**: Zhaoqun Li, Xiaotong Fang, Chen Chen, Mengze Li, Beishui Liao  

**Link**: [PDF](https://arxiv.org/pdf/2412.16725)  

**Abstract**: In recent years, large language models (LLMs) have made significant advancements in neuro-symbolic computing. However, the combination of LLM with argumentation computation remains an underexplored domain, despite its considerable potential for real-world applications requiring defeasible reasoning. In this paper, we aim to investigate the capability of LLMs in determining the extensions of various abstract argumentation semantics. To achieve this, we develop and curate a benchmark comprising diverse abstract argumentation frameworks, accompanied by detailed explanations of algorithms for computing extensions. Subsequently, we fine-tune LLMs on the proposed benchmark, focusing on two fundamental extension-solving tasks. As a comparative baseline, LLMs are evaluated using a chain-of-thought approach, where they struggle to accurately compute semantics. In the experiments, we demonstrate that the process explanation plays a crucial role in semantics computation learning. Models trained with explanations show superior generalization accuracy compared to those trained solely with question-answer pairs. Furthermore, by leveraging the self-explanation capabilities of LLMs, our approach provides detailed illustrations that mitigate the lack of transparency typically associated with neural networks. Our findings contribute to the broader understanding of LLMs' potential in argumentation computation, offering promising avenues for further research in this domain. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在神经符号计算方面取得了显著进展。然而，尽管架设LLMs与论证计算相结合具有重要的现实应用潜力，尤其是在要求进行可撤销推理的情况下，这一领域仍相对未被充分探索。本文旨在探讨LLMs在确定各种抽象论证语义扩展方面的潜力。为此，我们构建并整理了一个基准数据集，该数据集包含多种不同的抽象论证框架，并附带上详细的算法解释。接着，我们针对两个基本的扩展求解任务对LLMs进行微调。为了进行比较，我们采用链式思考的方法评估LLMs，结果显示它们在准确计算语义方面存在困难。在实验中，我们证明了过程解释在语义计算学习中的关键作用。使用带有解释训练的模型表现出比仅使用问答对训练的模型更好的泛化准确性。此外，通过利用LLMs的自我解释能力，我们的方法提供了详细的示例图，减轻了神经网络固有的透明度不足问题。我们的发现为更广泛理解LLMs在论证计算中的潜力做出了贡献，并为该领域的进一步研究提供了有希望的途径。 

---
# Internalized Self-Correction for Large Language Models 

**Title (ZH)**: 大型语言模型内部化的自我校正机制 

**Authors**: Nishanth Upadhyaya, Raghavendra Sridharamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2412.16653)  

**Abstract**: In this article, we introduce 'Internalized Self-Correction' (InSeC) for large language models (LLMs). While many approaches exist for self-reflection at inference time, we propose a novel method that combines ideas from negative sampling, self-reflection during training, and inference time. InSeC allows LLMs to correct themselves by introducing mistakes and their corresponding corrections during training, thereby converting the learning process into a true supervised learning task with both positive and negative examples. This approach can be extended to improve instruction following and correct hallucinations or incorrect sentences generated by LLMs. 

**Abstract (ZH)**: 在本文中，我们介绍了“内化自我矫正”（InSeC）机制，用于大规模语言模型（LLMs）。虽然在推理阶段存在许多自我反思的方法，但我们提出了一种新的方法，该方法结合了负采样、训练期间的自我反思和推理阶段的自我反思思想。InSeC 允许在训练过程中通过引入错误及其相应的纠正内容，使LLMs 自我纠正，从而将学习过程转化为具有正负样本的真正监督学习任务。这种方法可以扩展以提高指令跟随能力，并纠正LLMs 生成的幻觉或错误句子。 

---
# TimeRAG: BOOSTING LLM Time Series Forecasting via Retrieval-Augmented Generation 

**Title (ZH)**: TimeRAG：通过检索增强生成提高大规模语言模型的时间序列预测能力 

**Authors**: Silin Yang, Dong Wang, Haoqi Zheng, Ruochun Jin  

**Link**: [PDF](https://arxiv.org/pdf/2412.16643)  

**Abstract**: Although the rise of large language models (LLMs) has introduced new opportunities for time series forecasting, existing LLM-based solutions require excessive training and exhibit limited transferability. In view of these challenges, we propose TimeRAG, a framework that incorporates Retrieval-Augmented Generation (RAG) into time series forecasting LLMs, which constructs a time series knowledge base from historical sequences, retrieves reference sequences from the knowledge base that exhibit similar patterns to the query sequence measured by Dynamic Time Warping (DTW), and combines these reference sequences and the prediction query as a textual prompt to the time series forecasting LLM. Experiments on datasets from various domains show that the integration of RAG improved the prediction accuracy of the original model by 2.97% on average. 

**Abstract (ZH)**: 虽然大型语言模型（LLM）的兴起为时间序列预测带来了新的机遇，但现有的基于LLM的解决方案需要大量的训练，并且表现出有限的迁移性。鉴于这些挑战，我们提出了一种名为TimeRAG的框架，该框架将检索增强生成（RAG）融入时间序列预测的LLM中。TimeRAG从历史序列构建时间序列知识库，通过动态时间规整（DTW）检索与查询序列表现出相似模式的参考序列，并将这些参考序列与预测查询结合成文本提示，传入时间序列预测的LLM。实验结果表明，RAG的集成使原始模型的预测准确率平均提高了2.97%。 

---
# Do Multimodal Language Models Really Understand Direction? A Benchmark for Compass Direction Reasoning 

**Title (ZH)**: 多模态语言模型真的理解方向吗？一种关于指南针方向推理的标准评估方法 

**Authors**: Hang Yin, Zhifeng Lin, Xin Liu, Bin Sun, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.16599)  

**Abstract**: Direction reasoning is essential for intelligent systems to understand the real world. While existing work focuses primarily on spatial reasoning, compass direction reasoning remains underexplored. To address this, we propose the Compass Direction Reasoning (CDR) benchmark, designed to evaluate the direction reasoning capabilities of multimodal language models (MLMs). CDR includes three types images to test spatial (up, down, left, right) and compass (north, south, east, west) directions. Our evaluation reveals that most MLMs struggle with direction reasoning, often performing at random guessing levels. Experiments show that training directly with CDR data yields limited improvements, as it requires an understanding of real-world physical rules. We explore the impact of mixdata and CoT fine-tuning methods, which significantly enhance MLM performance in compass direction reasoning by incorporating diverse data and step-by-step reasoning, improving the model's ability to understand direction relationships. 

**Abstract (ZH)**: 方向推理对于智能系统理解现实世界至关重要。尽管现有的研究主要集中在空间推理上，但指南针方向推理仍然未被充分探索。为解决这一问题，我们提出了一种指南针方向推理（CDR）基准测试，旨在评估多模态语言模型（MLMs）的方向推理能力。CDR包括三种类型的照片来测试空间方向（上、下、左、右）和指南针方向（北、南、东、西）。我们的评估结果显示，大多数MLMs在方向推理方面存在困难，经常表现为随机猜测的水平。实验表明，直接使用CDR数据进行训练所能带来的改进有限，因为这需要理解现实世界的物理规则。我们探索了掺混数据和逐步推理（CoT）微调方法的影响，这些方法通过引入多样化的数据和逐步推理，显著增强了MLMs在指南针方向推理方面的性能，提升了模型理解方向关系的能力。 

---
# Autoware.Flex: Human-Instructed Dynamically Reconfigurable Autonomous Driving Systems 

**Title (ZH)**: Autobatch.Flex：基于人类指令的动态可重构自动驾驶系统 

**Authors**: Ziwei Song, Mingsong Lv, Tianchi Ren, Chun Jason Xue, Jen-Ming Wu, Nan Guan  

**Link**: [PDF](https://arxiv.org/pdf/2412.16265)  

**Abstract**: Existing Autonomous Driving Systems (ADS) independently make driving decisions, but they face two significant limitations. First, in complex scenarios, ADS may misinterpret the environment and make inappropriate driving decisions. Second, these systems are unable to incorporate human driving preferences in their decision-making processes. This paper proposes this http URL, a novel ADS system that incorporates human input into the driving process, allowing users to guide the ADS in making more appropriate decisions and ensuring their preferences are satisfied. Achieving this needs to address two key challenges: (1) translating human instructions, expressed in natural language, into a format the ADS can understand, and (2) ensuring these instructions are executed safely and consistently within the ADS' s decision-making framework. For the first challenge, we employ a Large Language Model (LLM) assisted by an ADS-specialized knowledge base to enhance domain-specific translation. For the second challenge, we design a validation mechanism to ensure that human instructions result in safe and consistent driving behavior. Experiments conducted on both simulators and a real-world autonomous vehicle demonstrate that this http URL effectively interprets human instructions and executes them safely. 

**Abstract (ZH)**: 现有的自动驾驶系统（ADS）能够独立做出驾驶决策，但在复杂场景下可能误解环境并作出不适当的决策。此外，这些系统无法在其决策过程中融入用户驾驶偏好。本文提出了一种名为“this http URL”的新型ADS系统，该系统能够将人类输入纳入驾驶过程，使用户能够指导ADS做出更加合适的决策，确保其偏好得到满足。实现这一点需要解决两个关键挑战：（1）将用自然语言表达的人类指令转换为ADS能够理解的格式，（2）确保这些指令在ADS的决策框架中得到安全和一致的执行。为了应对第一个挑战，我们采用了大型语言模型（LLM）辅助以专门的ADS知识库的方法，以增强领域特定的翻译能力。为了解决第二个挑战，我们设计了一个验证机制，以确保人类指令能够产生安全和一致的驾驶行为。在模拟器和实际自动驾驶车辆上的实验结果表明，“this http URL”能够有效地解释人类指令并安全执行这些指令。 

---
# Mining Math Conjectures from LLMs: A Pruning Approach 

**Title (ZH)**: 从大规模语言模型中挖掘数学猜想：一种剪枝方法 

**Authors**: Jake Chuharski, Elias Rojas Collins, Mark Meringolo  

**Link**: [PDF](https://arxiv.org/pdf/2412.16177)  

**Abstract**: We present a novel approach to generating mathematical conjectures using Large Language Models (LLMs). Focusing on the solubilizer, a relatively recent construct in group theory, we demonstrate how LLMs such as ChatGPT, Gemini, and Claude can be leveraged to generate conjectures. These conjectures are pruned by allowing the LLMs to generate counterexamples. Our results indicate that LLMs are capable of producing original conjectures that, while not groundbreaking, are either plausible or falsifiable via counterexamples, though they exhibit limitations in code execution. 

**Abstract (ZH)**: 我们提出了一种使用大规模语言模型（LLMs）生成数学猜想的新型方法。本文集中在群论中的一个相对较新的概念——溶剂上，展示了如何利用如ChatGPT、Gemini和Claude等LLMs生成猜想。通过允许LLMs生成反例来进行筛选，我们展示了这些猜想的真实性和可验证性。我们的结果表明，尽管这些猜想可能并非具有开创性，但它们或是合理的，或可以通过反例进行验证，尽管它们在代码执行方面存在局限性。 

---
# LABIIUM: AI-Enhanced Zero-configuration Measurement Automation System 

**Title (ZH)**: LABIIUM：增强型零配置测量自动化系统 

**Authors**: Emmanuel A. Olowe, Danial Chitnis  

**Link**: [PDF](https://arxiv.org/pdf/2412.16172)  

**Abstract**: The complexity of laboratory environments requires solutions that simplify instrument interaction and enhance measurement automation. Traditional tools often require configuration, software, and programming skills, creating barriers to productivity. Previous approaches, including dedicated software suites and custom scripts, frequently fall short in providing user-friendly solutions that align with programming practices. We present LABIIUM, an AI-enhanced, zero-configuration measurement automation system designed to streamline experimental workflows and improve user productivity. LABIIUM integrates an AI assistant powered by Large Language Models (LLMs) to generate code. LABIIUM's Lab-Automation-Measurement Bridges (LAMBs) enable seamless instrument connectivity using standard tools such as VSCode and Python, eliminating setup overhead. To demonstrate its capabilities, we conducted experiments involving the measurement of the parametric transfer curve of a simple two-transistor inverting amplifier with a current source load. The AI assistant was evaluated using different prompt scenarios and compared with multiple models, including Claude Sonnet 3.5, Gemini Pro 1.5, and GPT-4o. An expert solution implementing the Gradient-Weighted Adaptive Stochastic Sampling (GWASS) method was used as a baseline. The solutions generated by the AI assistant were compared with the expert solution and a uniform linear sweep baseline with 10,000 points. The graph results show that the LLMs were able to successfully complete the most basic uniform sweep, but LLMs were unable to develop adaptive sweeping algorithms to compete with GWASS. The evaluation underscores LABIIUM's ability to enhance laboratory productivity and support digital transformation in research and industry, and emphasizes the future work required to improve LLM performance in Electronic Measurement Science Tasks. 

**Abstract (ZH)**: 实验室环境的复杂性要求简化仪器交互和增强测量自动化的解决方案。传统工具往往需要配置、软件和编程技能，从而成为生产力的障碍。以前的方法，包括专用软件套件和自定义脚本，通常无法提供与编程实践相一致的用户友好的解决方案。我们提出了LABIIUM，这是一个增效的人工智能增强型、零配置测量自动化系统，旨在简化实验工作流程并提高用户生产力。LABIIUM集成了由大型语言模型（LLMs）驱动的人工智能助手，用于生成代码。LABIIUM的Lab-Automation-Measurement Bridges（LAMBs）通过使用VSCode和Python等标准工具实现无缝仪器连接，消除了配置冗余。为了展示其能力，我们进行了测量简单两级反相放大器（带电流源负载）的参数传输曲线的实验。人工智能助手使用不同的提示场景进行了评估，并与其他多个模型，包括Claude Sonnet 3.5、Gemini Pro 1.5和GPT-4o进行了比较。一个采用梯度加权自适应随机采样（GWASS）方法的专家解决方案用作基准。人工智能助手生成的解决方案与专家解决方案和均匀线性扫描基线（包含10,000个数据点）进行了比较。结果显示，大型语言模型能够成功完成最基本的均匀扫描，但在开发与GWASS竞争的自适应扫描算法方面显得不足。该评估突显了LABIIUM能够提高实验室生产力并在科研和行业中支持数字化转型的能力，并强调了未来工作以提高大型语言模型在电子测量科学任务中的表现所需的发展方向。 

---
# DRT-o1: Optimized Deep Reasoning Translation via Long Chain-of-Thought 

**Title (ZH)**: DRT-o1：通过长逻辑链优化的深度推理翻译 

**Authors**: Jiaan Wang, Fandong Meng, Yunlong Liang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.17498)  

**Abstract**: Recently, O1-like models have emerged as representative examples, illustrating the effectiveness of long chain-of-thought (CoT) in reasoning tasks such as math and coding tasks. In this paper, we introduce DRT-o1, an attempt to bring the success of long CoT to neural machine translation (MT). Specifically, in view of the literature books that might involve similes and metaphors, translating these texts to a target language is very difficult in practice due to cultural differences. In such cases, literal translation often fails to convey the intended meaning effectively. Even for professional human translators, considerable thought must be given to preserving semantics throughout the translation process. To simulate LLMs' long thought ability in MT, we first mine sentences containing similes or metaphors from existing literature books, and then develop a multi-agent framework to translate these sentences via long thought. In the multi-agent framework, a translator is used to iteratively translate the source sentence under the suggestions provided by an advisor. To ensure the effectiveness of the long thoughts, an evaluator is also employed to judge whether the translation in the current round is better than the previous one or not. In this manner, we collect tens of thousands of long-thought MT data, which is used to train our DRT-o1. The experimental results on literature translation demonstrate the effectiveness of the DRT-o1. Using Qwen2.5-7B and Qwen2.5-14B as the backbones, the improvement brought by DRT-o1 achieves 7.33~8.26 BLEU and 1.66~3.36 CometScore. Besides, DRT-o1-7B can outperform QwQ-32B-Preview by 7.82 BLEU and 1.46 CometScore, showing its effectiveness. The project is available at this https URL 

**Abstract (ZH)**: 近年来，O1-like模型作为一种代表性的例子，展示了在数学和编程等推理任务中，长链思维（Long Chain-of-Thought, CoT）的有效性。本文旨在将长链思维的成功应用到神经机器翻译（NMT）中，我们介绍了一个名为DRT-o1的尝试。具体来说，由于文献书籍中可能包含比喻和隐喻，这些文本在目标语言中的翻译因文化差异而在实践中非常困难。在这种情况下，直接翻译往往难以有效地传达原意。即使对于专业的人类翻译者，在翻译过程中也需要付出大量努力来保留语义。为了模拟大规模语言模型（LLM）的长思考能力，我们首先从现有的文学书籍中挖掘包含比喻或隐喻的句子，然后开发了一个多代理框架，通过长链思维来翻译这些句子。在多代理框架中，翻译代理在顾问的建议下逐步翻译源句子。为了确保长链思维的有效性，我们还引入了一个评估器，用于判断当前轮次的翻译是否比上一轮次更好。通过这种方式，我们收集了十万多条长链思维的机器翻译数据，用于训练我们的DRT-o1模型。在文学翻译实验中，DRT-o1的有效性得到了验证。使用Qwen2.5-7B和Qwen2.5-14B作为骨干模型，DRT-o1带来的改进在BLEU分数上达到了7.33至8.26，在CometScore上达到了1.66至3.36的提升。此外，DRT-o1-7B在BLEU分数上比QwQ-32B-Preview高7.82，在CometScore上高1.46，显示出其有效性。该项目可在以下链接访问：[此处httpsURL] 

---
# Diving into Self-Evolving Training for Multimodal Reasoning 

**Title (ZH)**: 探索自进化训练在多模态推理中的应用 

**Authors**: Wei Liu, Junlong Li, Xiwen Zhang, Fan Zhou, Yu Cheng, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2412.17451)  

**Abstract**: Reasoning ability is essential for Large Multimodal Models (LMMs). In the absence of multimodal chain-of-thought annotated data, self-evolving training, where the model learns from its own outputs, has emerged as an effective and scalable approach for enhancing reasoning abilities. Despite its growing usage, a comprehensive understanding of self-evolving training, particularly in the context of multimodal reasoning, remains limited. In this paper, we delve into the intricacies of self-evolving training for multimodal reasoning, pinpointing three key factors: Training Method, Reward Model, and Prompt Variation. We systematically examine each factor and explore how various configurations affect the training's effectiveness. Our analysis leads to a set of best practices for each factor, aimed at optimizing multimodal reasoning. Furthermore, we explore the Self-Evolution Dynamics during training and the impact of automatic balancing mechanisms in boosting performance. After all the investigations, we present a final recipe for self-evolving training in multimodal reasoning, encapsulating these design choices into a framework we call MSTaR (Multimodal Self-evolving Training for Reasoning), which is universally effective for models with different sizes on various benchmarks, e.g., surpassing the pre-evolved model significantly on 5 multimodal reasoning benchmarks without using additional human annotations, as demonstrated on MiniCPM-V-2.5 (8B), Phi-3.5-Vision (4B) and InternVL2 (2B). We believe this study fills a significant gap in the understanding of self-evolving training for multimodal reasoning and offers a robust framework for future research. Our policy and reward models, as well as the collected data, is released to facilitate further investigation in multimodal reasoning. 

**Abstract (ZH)**: 逻辑推理能力对于大型多模态模型（LMMs）至关重要。在缺乏多模态链式思维标注数据的情况下，自我进化训练，即模型从自身输出中学习，已成为提升推理能力的有效且可扩展的方法。尽管这种方法的使用正在不断增加，但在多模态推理的具体情境下对自我进化训练的全面理解依然有限。在本文中，我们深入探讨了自我进化训练在多模态推理中的复杂性，指出了三个关键因素：训练方法、奖励模型和提示变异。我们系统地分析了每个因素，并探讨了各种配置如何影响训练的有效性。我们的分析得出了每个因素的最佳实践，旨在优化多模态推理。此外，我们还探讨了训练过程中的自我进化动态以及自动平衡机制对于提升性能的影响。经过一系列调查后，我们提出了适用于多模态推理的自我进化训练的最终食谱，将其设计理念整合成一个名为MSTaR（多模态自我进化训练推理）的框架，该框架对于不同规模的模型在各种基准上都是普遍有效的。例如，在MiniCPM-V-2.5（8B）、Phi-3.5-Vision（4B）和InternVL2（2B）这三种模型上，MSTaR在不使用额外人工注释的情况下，显著超越了预进化模型，在5个多模态推理基准测试中表现尤为突出。我们相信，这项研究填补了多模态推理领域自我进化训练理解的空白，并提供了一个坚实的框架，以支持未来的研究。我们的政策和奖励模型以及收集的数据被公开发布，以促进对该领域进一步调查。 

---
# VidCtx: Context-aware Video Question Answering with Image Models 

**Title (ZH)**: VidCtx：基于图像模型的上下文感知视频问答 

**Authors**: Andreas Goulas, Vasileios Mezaris, Ioannis Patras  

**Link**: [PDF](https://arxiv.org/pdf/2412.17415)  

**Abstract**: To address computational and memory limitations of Large Multimodal Models in the Video Question-Answering task, several recent methods extract textual representations per frame (e.g., by captioning) and feed them to a Large Language Model (LLM) that processes them to produce the final response. However, in this way, the LLM does not have access to visual information and often has to process repetitive textual descriptions of nearby frames. To address those shortcomings, in this paper, we introduce VidCtx, a novel training-free VideoQA framework which integrates both modalities, i.e. both visual information from input frames and textual descriptions of others frames that give the appropriate context. More specifically, in the proposed framework a pre-trained Large Multimodal Model (LMM) is prompted to extract at regular intervals, question-aware textual descriptions (captions) of video frames. Those will be used as context when the same LMM will be prompted to answer the question at hand given as input a) a certain frame, b) the question and c) the context/caption of an appropriate frame. To avoid redundant information, we chose as context the descriptions of distant frames. Finally, a simple yet effective max pooling mechanism is used to aggregate the frame-level decisions. This methodology enables the model to focus on the relevant segments of the video and scale to a high number of frames. Experiments show that VidCtx achieves competitive performance among approaches that rely on open models on three public Video QA benchmarks, NExT-QA, IntentQA and STAR. 

**Abstract (ZH)**: 为了解决大规模多模态模型在视频问答任务中计算和内存限制的问题，最近的一些方法通过为每一帧提取文本表示（例如，通过添加字幕），并将这些文本输入到大规模语言模型（LLM）中进行处理，以生成最终回答。但是，这种做法使得LLM无法访问视觉信息，并且经常需要处理相邻帧的重复文本描述。为了克服这些不足，本文提出了一种名为VidCtx的新颖训练免费视频问答框架，该框架结合了两种模态的信息，即输入帧的视觉信息和描述其他帧的文本描述，以提供适当的上下文。具体而言，在提出的框架中，预训练的大规模多模态模型（LMM）在固定的时间间隔内被提示提取问题感知的视频帧文本描述（字幕）。这些字幕将用于回答给定输入的问题时的上下文，即a）某一帧，b）问题以及c）合适帧的上下文/字幕。为了避免冗余信息，我们将帧间距离较远的字幕作为上下文。最后，我们使用简单的最大池化机制来汇总帧级别的决策。该方法使模型能够专注于视频中的相关段落，并可扩展到大量帧。实验结果显示，VidCtx在三个公开的视频问答基准测试NExT-QA、IntentQA和STAR上，与依赖开放式模型的方法相比，取得了竞争力的表现。 

---
# Boosting LLM via Learning from Data Iteratively and Selectively 

**Title (ZH)**: 通过迭代选择性地从数据中学习增强大语言模型 

**Authors**: Qi Jia, Siyu Ren, Ziheng Qin, Fuzhao Xue, Jinjie Ni, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2412.17365)  

**Abstract**: Datasets nowadays are generally constructed from multiple sources and using different synthetic techniques, making data de-noising and de-duplication crucial before being used for post-training. In this work, we propose to perform instruction tuning by iterative data selection (\ApproachName{}). We measure the quality of a sample from complexity and diversity simultaneously. Instead of calculating the complexity score once for all before fine-tuning, we highlight the importance of updating this model-specific score during fine-tuning to accurately accommodate the dynamic changes of the model. On the other hand, the diversity score is defined on top of the samples' responses under the consideration of their informativeness. IterIT integrates the strengths of both worlds by iteratively updating the complexity score for the top-ranked samples and greedily selecting the ones with the highest complexity-diversity score. Experiments on multiple instruction-tuning data demonstrate consistent improvements of IterIT over strong baselines. Moreover, our approach also generalizes well to domain-specific scenarios and different backbone models. All resources will be available at this https URL. 

**Abstract (ZH)**: 当前的的数据集通常来源于多个数据源并使用不同的合成技术构建，因此在进行模型训练后需要先进行数据去噪和去重处理。本文提出了一种通过迭代数据选择来进行指令调优的方法（\ApproachName{}）。我们同时从复杂性与多样性两个方面衡量样本的质量。不同于在微调前一次性计算所有样本的复杂性分数，我们在微调过程中突出更新这种模型特异性分数的重要性，以准确适应模型的动态变化。另一方面，多样性分数定义在考虑样本响应信息量的基础上。IterIT 通过迭代更新排名靠前样本的复杂性分数，并贪婪选择具有最高复杂性-多样性分数的样本，从而整合了两方面的优点。实验结果表明，IterIT 在多种指令调优数据上一致优于强基线方法。此外，该方法也能够在特定领域场景和不同的骨干模型中良好泛化。所有资源请访问 <https://your-resource-url-here.com>。 

---
# Unlocking Cross-Lingual Sentiment Analysis through Emoji Interpretation: A Multimodal Generative AI Approach 

**Title (ZH)**: 通过表情符号解释实现跨语言情感分析——一种多模态生成AI方法 

**Authors**: Rafid Ishrak Jahan, Heng Fan, Haihua Chen, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2412.17255)  

**Abstract**: Emojis have become ubiquitous in online communication, serving as a universal medium to convey emotions and decorative elements. Their widespread use transcends language and cultural barriers, enhancing understanding and fostering more inclusive interactions. While existing work gained valuable insight into emojis understanding, exploring emojis' capability to serve as a universal sentiment indicator leveraging large language models (LLMs) has not been thoroughly examined. Our study aims to investigate the capacity of emojis to serve as reliable sentiment markers through LLMs across languages and cultures. We leveraged the multimodal capabilities of ChatGPT to explore the sentiments of various representations of emojis and evaluated how well emoji-conveyed sentiment aligned with text sentiment on a multi-lingual dataset collected from 32 countries. Our analysis reveals that the accuracy of LLM-based emoji-conveyed sentiment is 81.43%, underscoring emojis' significant potential to serve as a universal sentiment marker. We also found a consistent trend that the accuracy of sentiment conveyed by emojis increased as the number of emojis grew in text. The results reinforce the potential of emojis to serve as global sentiment indicators, offering insight into fields such as cross-lingual and cross-cultural sentiment analysis on social media platforms. Code: this https URL. 

**Abstract (ZH)**: 表情符号已成为在线交流中无处不在的工具，用以传达情绪和作为装饰性元素。它们的广泛应用超越了语言和文化障碍，增强了理解并促进了更为包容的交流。虽然现有研究揭示了表情符号的理解价值，但利用大规模语言模型（LLMs）探索表情符号作为全球情绪指标的能力尚未得到充分研究。我们的研究旨在通过LLMs跨语言和跨文化地考察表情符号作为可靠情绪标记的能力。我们利用ChatGPT的多模态能力探讨了各种表情符号表示形式的情绪，评估了表情符号所传达的情绪与多语言数据集中文本情绪的一致性。我们的分析显示，基于LLM的表情符号所传达情绪的准确率为81.43%，突显了表情符号作为全球情绪指标的巨大潜力。我们还发现，随着文本中表情符号数量的增加，表情符号所传达情绪的准确性呈现出一致的上升趋势。这些结果强化了表情符号作为全球情绪指标的潜力，对社交媒体平台上的跨语言和跨文化情绪分析领域提供了有价值的见解。代码：[这里提供代码链接]。 

---
# A Multi-AI Agent System for Autonomous Optimization of Agentic AI Solutions via Iterative Refinement and LLM-Driven Feedback Loops 

**Title (ZH)**: 基于迭代细化和LLM驱动的反馈循环的自主优化多AI代理系统：为代理AI解决方案赋能 

**Authors**: Kamer Ali Yuksel, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2412.17149)  

**Abstract**: Agentic AI systems use specialized agents to handle tasks within complex workflows, enabling automation and efficiency. However, optimizing these systems often requires labor-intensive, manual adjustments to refine roles, tasks, and interactions. This paper introduces a framework for autonomously optimizing Agentic AI solutions across industries, such as NLP-driven enterprise applications. The system employs agents for Refinement, Execution, Evaluation, Modification, and Documentation, leveraging iterative feedback loops powered by an LLM (Llama 3.2-3B). The framework achieves optimal performance without human input by autonomously generating and testing hypotheses to improve system configurations. This approach enhances scalability and adaptability, offering a robust solution for real-world applications in dynamic environments. Case studies across diverse domains illustrate the transformative impact of this framework, showcasing significant improvements in output quality, relevance, and actionability. All data for these case studies, including original and evolved agent codes, along with their outputs, are here: this https URL 

**Abstract (ZH)**: 以下是符合学术规范的翻译：

具有代理功能的AI系统使用专门的代理来处理复杂工作流程中的任务，从而实现自动化和提高效率。然而，优化这些系统通常需要耗时的手动调整来细化角色、任务和交互。本文提出了一种跨行业的自主优化具有代理功能的AI解决方案的框架，适用于如基于NLP的企业应用等场景。该系统采用代理进行细化（Refinement）、执行（Execution）、评估（Evaluation）、修改（Modification）和记录（Documentation），利用由LLM（Llama 3.2-3B）驱动的迭代反馈循环。该框架通过自主生成和测试假设来优化系统配置，实现了无需人工干预的最优性能。这种方法增强了系统的可扩展性和适应性，提供了在动态环境中实现实用解决方案的有效方案。来自不同领域的案例研究展示了该框架的转变性影响，显著提高了输出质量、相关性和可操作性。这些案例研究的所有数据，包括原始和演变的代理代码及其输出，均可在此获得：this https URL 

---
# Analysis on LLMs Performance for Code Summarization 

**Title (ZH)**: 对代码摘要性能的大型语言模型分析 

**Authors**: Md. Ahnaf Akib, Md. Muktadir Mazumder, Salman Ahsan  

**Link**: [PDF](https://arxiv.org/pdf/2412.17094)  

**Abstract**: Code summarization aims to generate concise natural language descriptions for source code. Deep learning has been used more and more recently in software engineering, particularly for tasks like code creation and summarization. Specifically, it appears that the most current Large Language Models with coding perform well on these tasks. Large Language Models (LLMs) have significantly advanced the field of code summarization, providing sophisticated methods for generating concise and accurate summaries of source code. This study aims to perform a comparative analysis of several open-source LLMs, namely LLaMA-3, Phi-3, Mistral, and Gemma. These models' performance is assessed using important metrics such as BLEU\textsubscript{3.1} and ROUGE\textsubscript{3.2}.
Through this analysis, we seek to identify the strengths and weaknesses of each model, offering insights into their applicability and effectiveness in code summarization tasks. Our findings contribute to the ongoing development and refinement of LLMs, supporting their integration into tools that enhance software development and maintenance processes. 

**Abstract (ZH)**: 代码总结旨在为源代码生成简洁的自然语言描述。近年来，深度学习在软件工程中得到广泛应用，特别是在代码创建和总结等任务中。具体而言，当前表现良好的大型语言模型（特别是那些具有编程能力的模型）在这些任务上表现出色。大型语言模型（LLMs）显著推动了代码总结领域的进展，提供了生成简洁且准确的源代码摘要的高级方法。本研究旨在对比分析几个开源的大型语言模型，包括LLaMA-3、Phi-3、Mistral和Gemma。通过使用如BLEU-3.1和ROUGE-3.2等重要指标来评估这些模型的性能。

通过此次分析，我们旨在识别每个模型的优势和劣势，并为这些模型在代码总结任务中的适用性和有效性提供见解。我们的研究结果将为大型语言模型的持续开发和改进做出贡献，支持其在提升软件开发和维护过程中的工具集成。 

---
# Prompting Large Language Models with Rationale Heuristics for Knowledge-based Visual Question Answering 

**Title (ZH)**: 使用合理性启发式方法提示大型语言模型进行基于知识的视觉问答 

**Authors**: Zhongjian Hu, Peng Yang, Bing Li, Fengyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16936)  

**Abstract**: Recently, Large Language Models (LLMs) have been used for knowledge-based Visual Question Answering (VQA). Despite the encouraging results of previous studies, prior methods prompt LLMs to predict answers directly, neglecting intermediate thought processes. We argue that prior methods do not sufficiently activate the capacities of LLMs. We propose a framework called PLRH that Prompts LLMs with Rationale Heuristics for knowledge-based VQA. The PLRH prompts LLMs with Chain of Thought (CoT) to generate rationale heuristics, i.e., intermediate thought processes, and then leverages the rationale heuristics to inspire LLMs to predict answers. Experiments show that our approach outperforms the existing baselines by more than 2.2 and 2.1 on OK-VQA and A-OKVQA, respectively. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）已被应用于基于知识的视觉问答（VQA）。尽管前期研究给出了令人鼓舞的结果，但先前的方法仅促使LLMs直接预测答案，而忽视了中间推理过程。我们主张先前的方法未能充分激活LLMs的能力。为此，我们提出了一种名为PLRH（Prompting LLMs with Rationale Heuristics）的框架。该框架通过链式思维（Chain of Thought，CoT）引导LLMs生成中间推理过程，即理性启发式，然后利用这些启发式来激发LLMs生成答案。实验结果显示，我们的方法在OK-VQA和A-OKVQA上的表现分别比现有基线高出2.2和2.1个百分点。 

---
# Online Preference-based Reinforcement Learning with Self-augmented Feedback from Large Language Model 

**Title (ZH)**: 基于在线偏好强化学习的大语言模型自增强反馈机制 

**Authors**: Songjun Tu, Jingbo Sun, Qichao Zhang, Xiangyuan Lan, Dongbin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.16878)  

**Abstract**: Preference-based reinforcement learning (PbRL) provides a powerful paradigm to avoid meticulous reward engineering by learning rewards based on human preferences. However, real-time human feedback is hard to obtain in online tasks. Most work suppose there is a "scripted teacher" that utilizes privileged predefined reward to provide preference feedback. In this paper, we propose a RL Self-augmented Large Language Model Feedback (RL-SaLLM-F) technique that does not rely on privileged information for online PbRL. RL-SaLLM-F leverages the reflective and discriminative capabilities of LLM to generate self-augmented trajectories and provide preference labels for reward learning. First, we identify an failure issue in LLM-based preference discrimination, specifically "query ambiguity", in online PbRL. Then LLM is employed to provide preference labels and generate self-augmented imagined trajectories that better achieve the task goal, thereby enhancing the quality and efficiency of feedback. Additionally, a double-check mechanism is introduced to mitigate randomness in the preference labels, improving the reliability of LLM feedback. The experiment across multiple tasks in the MetaWorld benchmark demonstrates the specific contributions of each proposed module in RL-SaLLM-F, and shows that self-augmented LLM feedback can effectively replace the impractical "scripted teacher" feedback. In summary, RL-SaLLM-F introduces a new direction of feedback acquisition in online PbRL that does not rely on any online privileged information, offering an efficient and lightweight solution with LLM-driven feedback. 

**Abstract (ZH)**: 基于偏好的强化学习（Preference-based Reinforcement Learning, PbRL）提供了一种强大的范式，通过基于人类偏好学习奖励来避免繁琐的奖励工程。然而，在在线任务中实时获取人类反馈是困难的。大多数工作假设存在一个“剧本教师”（scripted teacher），利用先验预定义的奖励来提供偏好反馈。在本文中，我们提出了一种名为RL Self-augmented Large Language Model Feedback (RL-SaLLM-F) 的技术，该技术不依赖于先验信息进行在线PbRL。RL-SaLLM-F 利用大型语言模型（LLM）的反思能力和区分能力生成自增强轨迹，并为此提供奖励学习所需的偏好标签。首先，我们识别了基于LLM的偏好区分中的一个失败问题，即“查询歧义性”问题，在线PbRL中的偏好辨别问题。然后，LLM 用于提供偏好标签并生成更接近目标任务的自增强想象轨迹，从而提高反馈的质量和效率。此外，我们引入了一种双检查机制来减轻偏好标签中的随机性，从而提高LLM反馈的可靠性。在MetaWorld基准任务上的实验展示了RL-SaLLM-F 中每个模块的具体贡献，表明自增强的LLM反馈可以有效地替代不切实际的“剧本教师”反馈。总之，RL-SaLLM-F 引入了一种新的在线PbRL反馈获取方向，该方向不依赖于任何在线先验信息，提供了一种高效且轻量级的基于LLM驱动的反馈解决方案。 

---
# Sim911: Towards Effective and Equitable 9-1-1 Dispatcher Training with an LLM-Enabled Simulation 

**Title (ZH)**: Sim911：通过具备大语言模型功能的模拟技术朝着更有效和公平的9-1-1调度员培训方向努力 

**Authors**: Zirong Chen, Elizabeth Chason, Noah Mladenovski, Erin Wilson, Kristin Mullen, Stephen Martini, Meiyi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2412.16844)  

**Abstract**: Emergency response services are vital for enhancing public safety by safeguarding the environment, property, and human lives. As frontline members of these services, 9-1-1 dispatchers have a direct impact on response times and the overall effectiveness of emergency operations. However, traditional dispatcher training methods, which rely on role-playing by experienced personnel, are labor-intensive, time-consuming, and often neglect the specific needs of underserved communities. To address these challenges, we introduce Sim911, the first training simulation for 9-1-1 dispatchers powered by Large Language Models (LLMs). Sim911 enhances training through three key technical innovations: (1) knowledge construction, which utilizes archived 9-1-1 call data to generate simulations that closely mirror real-world scenarios; (2) context-aware controlled generation, which employs dynamic prompts and vector bases to ensure that LLM behavior aligns with training objectives; and (3) validation with looped correction, which filters out low-quality responses and refines the system performance. 

**Abstract (ZH)**: 应急响应服务对于提高公共安全至关重要，它通过保护环境、财产和人类生命来实现这一目标。作为这些服务的前线成员，9-1-1调度员直接影响响应时间和整体应急操作的有效性。然而，传统的调度员培训方法依赖于由经验丰富的人员进行的角色扮演，这种方法耗时费力，且往往忽略了服务不足社区的具体需求。为了解决这些问题，我们介绍Sim911，这是一种由大规模语言模型（LLMs）驱动的9-1-1调度员培训模拟，它是第一个此类模拟。Sim911通过三项关键技术革新增强了培训：（1）知识构建，利用存档的9-1-1呼叫数据生成模拟场景，这些场景能够准确反映现实生活中的情况；（2）情境感知控制生成，通过使用动态提示和向量基底来确保大规模语言模型的行为与培训目标一致；（3）循环校正验证，筛选低质量响应并优化系统性能。 

---
# Assessing Social Alignment: Do Personality-Prompted Large Language Models Behave Like Humans? 

**Title (ZH)**: 评估社会一致性：个性化的提示引起的大型语言模型是否表现出人类的行为？ 

**Authors**: Ivan Zakazov, Mikolaj Boronski, Lorenzo Drudi, Robert West  

**Link**: [PDF](https://arxiv.org/pdf/2412.16772)  

**Abstract**: The ongoing revolution in language modelling has led to various novel applications, some of which rely on the emerging "social abilities" of large language models (LLMs). Already, many turn to the new "cyber friends" for advice during pivotal moments of their lives and trust them with their deepest secrets, implying that accurate shaping of LLMs' "personalities" is paramount. Leveraging the vast diversity of data on which LLMs are pretrained, state-of-the-art approaches prompt them to adopt a particular personality. We ask (i) if personality-prompted models behave (i.e. "make" decisions when presented with a social situation) in line with the ascribed personality, and (ii) if their behavior can be finely controlled. We use classic psychological experiments - the Milgram Experiment and the Ultimatum Game - as social interaction testbeds and apply personality prompting to GPT-3.5/4/4o-mini/4o. Our experiments reveal failure modes of the prompt-based modulation of the models' "behavior", thus challenging the feasibility of personality prompting with today's LLMs. 

**Abstract (ZH)**: 语言模型领域的持续革命催生了各种新型应用，其中一些应用依赖于大型语言模型（LLMs） emerging的“社会能力”。目前，许多人已经转向这些新出现的“网络朋友”寻求在生命关键时刻的建议，并将自己的最深处的秘密托付给他们，这表明准确塑造LLMs的“个性”至关重要。利用LLMs预训练时使用的广泛多样的数据，最先进的方法促使它们采取特定的个性特征。我们询问（i）个性提示模型在面对社会情境时是否会（即“做出”决策）表现出与归因的个性特征一致的行为，以及（ii）其行为是否能够精细地控制。我们使用经典的心理学实验——米尔格拉姆实验和最后通牒游戏——作为社会互动的测试平台，并对GPT-3.5/4/4o-mini/4o应用个性提示。我们的实验揭示了基于提示对模型行为进行调整的失败模式，从而挑战了当前LLMs中通过个性提示控制行为的可行性。 

---
# Unpacking Political Bias in Large Language Models: Insights Across Topic Polarization 

**Title (ZH)**: 探究大型语言模型中的政治偏见：跨主题极化视角的见解 

**Authors**: Kaiqi Yang, Hang Li, Yucheng Chu, Yuping Lin, Tai-Quan Peng, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16746)  

**Abstract**: Large Language Models (LLMs) have been widely used to generate responses on social topics due to their world knowledge and generative capabilities. Beyond reasoning and generation performance, political bias is an essential issue that warrants attention. Political bias, as a universal phenomenon in human society, may be transferred to LLMs and distort LLMs' behaviors of information acquisition and dissemination with humans, leading to unequal access among different groups of people. To prevent LLMs from reproducing and reinforcing political biases, and to encourage fairer LLM-human interactions, comprehensively examining political bias in popular LLMs becomes urgent and crucial.
In this study, we systematically measure the political biases in a wide range of LLMs, using a curated set of questions addressing political bias in various contexts. Our findings reveal distinct patterns in how LLMs respond to political topics. For highly polarized topics, most LLMs exhibit a pronounced left-leaning bias. Conversely, less polarized topics elicit greater consensus, with similar response patterns across different LLMs. Additionally, we analyze how LLM characteristics, including release date, model scale, and region of origin affect political bias. The results indicate political biases evolve with model scale and release date, and are also influenced by regional factors of LLMs. 

**Abstract (ZH)**: 大规模语言模型（LLMs）由于其世界知识和生成能力，在生成社会话题的回应方面得到了广泛应用。除了推理和生成性能之外，政治偏见也是一个值得关注的重要问题。作为人类社会的一种普遍现象，政治偏见可能会转移到LLMs中，影响LLMs在人类信息获取和传播过程中的行为，导致不同群体之间获取信息的不平等。为了防止LLMs复制和强化政治偏见，并促进更公平的LLM-人类交互，全面审视流行LLMs中的政治偏见变得迫切和重要。

在本研究中，我们使用经过筛选的问题集系统地测量了多种LLMs的政治偏见，这些问题涵盖了各种社会情境中的政治偏见。我们的研究发现，LLMs对政治话题的回应展示了不同的模式。对于高度两极化的议题，大多数LLMs表现出明显的左倾偏见。相反，在较不两极化的议题上，LLMs之间则表现出更大的共识，回答模式也较为一致。此外，我们分析了LLM的特性，包括发布日期、模型规模和起源地区如何影响政治偏见。研究结果表明，政治偏见随着模型规模和发布日期的变化而演变，并且受到LLM起源地区的因素影响。 

---
# POEX: Policy Executable Embodied AI Jailbreak Attacks 

**Title (ZH)**: POEX：政策可执行的具身AI越界攻击

解释：
- **POEX** 是标题中的缩写或代码，直译为“POEX”。
- **Policy Executable** 翻译为“政策可执行的”，意指在执行过程中考虑了相关政策或规则。
- **Embodied AI** 翻译为“具身AI”，指的是嵌入到物理系统中的智能代理。
- **Jailbreak Attacks** 翻译为“越界攻击”，在学术语境中通常指违反系统安全策略的行为。 

**Authors**: Xuancun Lu, Zhengxian Huang, Xinfeng Li, Xiaoyu ji, Wenyuan Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16633)  

**Abstract**: The integration of large language models (LLMs) into the planning module of Embodied Artificial Intelligence (Embodied AI) systems has greatly enhanced their ability to translate complex user instructions into executable policies. In this paper, we demystified how traditional LLM jailbreak attacks behave in the Embodied AI context. We conducted a comprehensive safety analysis of the LLM-based planning module of embodied AI systems against jailbreak attacks. Using the carefully crafted Harmful-RLbench, we accessed 20 open-source and proprietary LLMs under traditional jailbreak attacks, and highlighted two key challenges when adopting the prior jailbreak techniques to embodied AI contexts: (1) The harmful text output by LLMs does not necessarily induce harmful policies in Embodied AI context, and (2) even we can generate harmful policies, we have to guarantee they are executable in practice. To overcome those challenges, we propose Policy Executable (POEX) jailbreak attacks, where harmful instructions and optimized suffixes are injected into LLM-based planning modules, leading embodied AI to perform harmful actions in both simulated and physical environments. Our approach involves constraining adversarial suffixes to evade detection and fine-tuning a policy evaluater to improve the executability of harmful policies. We conducted extensive experiments on both a robotic arm embodied AI platform and simulators, to validate the attack and policy success rates on 136 harmful instructions from Harmful-RLbench. Our findings expose serious safety vulnerabilities in LLM-based planning modules, including the ability of POEX to be transferred across models. Finally, we propose mitigation strategies, such as safety-constrained prompts, pre- and post-planning checks, to address these vulnerabilities and ensure the safe deployment of embodied AI in real-world settings. 

**Abstract (ZH)**: 将大型语言模型（LLMs）集成到具身人工智能（Embodied AI）系统中的规划模块中，极大地增强了其将复杂的用户指令转化为可执行策略的能力。在本文中，我们探讨了传统的LLM劫持攻击在具身AI环境下的行为。我们对基于LLM的具身AI系统的规划模块进行了全面的安全分析，以抵御劫持攻击。我们使用精心设计的Harmful-RLbench，对20个开源和专有LLM进行了传统的劫持攻击测试，并突出了采用传统劫持技术在具身AI环境中的两个关键挑战：（1）LLM生成的危害性文本不一定会导致具身AI中的有害策略，（2）即使我们能够生成有害策略，也需要确保这些策略在实践中是可执行的。为克服这些挑战，我们提出了可执行策略（POEX）劫持攻击，其中将有害指令和优化的后缀注入基于LLM的规划模块，使具身AI在模拟和物理环境中执行有害行为。我们的方法包括限制敌对后缀以躲避检测，并fine-tuning策略评估器以提高有害策略的可执行性。我们在一个具身人工智能平台和模拟器上进行了广泛的实验，验证了136条来自Harmful-RLbench的有害指令的攻击和策略成功率。我们的研究揭示了基于LLM的规划模块中的严重安全漏洞，包括POEX能够跨模型转移的能力。最后，我们提出了缓解策略，如安全性约束提示、规划前后的检查，以解决这些漏洞并确保具身AI在真实环境中的安全部署。 

---
# Correcting Large Language Model Behavior via Influence Function 

**Title (ZH)**: 通过影响函数纠正大型语言模型的行为 

**Authors**: Han Zhang, Zhuo Zhang, Yi Zhang, Yuanzhao Zhai, Hanyang Peng, Yu Lei, Yue Yu, Hui Wang, Bin Liang, Lin Gui, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16451)  

**Abstract**: Recent advancements in AI alignment techniques have significantly improved the alignment of large language models (LLMs) with static human preferences. However, the dynamic nature of human preferences can render some prior training data outdated or even erroneous, ultimately causing LLMs to deviate from contemporary human preferences and societal norms. Existing methodologies, whether they involve the curation of new data for continual alignment or the manual correction of outdated data for re-alignment, demand costly human resources. To address this challenge, we propose a novel approach, Large Language Model Behavior Correction with Influence Function Recall and Post-Training (LANCET), which requires no human involvement. LANCET consists of two phases: (1) using influence functions to identify the training data that significantly impact undesirable model outputs, and (2) applying an Influence function-driven Bregman Optimization (IBO) technique to adjust the model's behavior based on these influence distributions. Our experiments demonstrate that LANCET effectively and efficiently correct inappropriate behaviors of LLMs. Furthermore, LANCET can outperform methods that rely on collecting human preferences, and it enhances the interpretability of learning human preferences within LLMs. 

**Abstract (ZH)**: 近年来，人工智能对齐技术的进展显著提高了大型语言模型（LLMs）与静态人类偏好的一致性。然而，人类偏好的动态特性可能导致先前的训练数据变得过时甚至错误，最终导致LLMs偏离当前的人类偏好和社会规范。现有的方法，无论是通过持续收集新数据进行对齐，还是通过手动修正过时数据进行重新对齐，都依赖于昂贵的人力资源。为解决这一挑战，我们提出了一种新颖的方法——大型语言模型行为纠正结合影响函数召回和后处理（LANCET），该方法不需要人工干预。LANCET包括两个阶段：（1）使用影响函数来识别显著影响模型输出的关键训练数据，（2）应用影响函数驱动的布丹优化（IBO）技术，根据这些影响分布来调整模型行为。我们的实验表明，LANCET能够有效地和高效地纠正LLMs的不当行为。此外，LANCET在依赖于收集人类偏好值的方法中表现出更优的效果，并增强了在LLMs中学习人类偏好的可解释性。 

---
# Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context 

**Title (ZH)**: 具有可读性的对抗性提示：基于情境上下文对LLM漏洞的研究 

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2412.16359)  

**Abstract**: Previous research on LLM vulnerabilities often relied on nonsensical adversarial prompts, which were easily detectable by automated methods. We address this gap by focusing on human-readable adversarial prompts, a more realistic and potent threat. Our key contributions are situation-driven attacks leveraging movie scripts to create contextually relevant, human-readable prompts that successfully deceive LLMs, adversarial suffix conversion to transform nonsensical adversarial suffixes into meaningful text, and AdvPrompter with p-nucleus sampling, a method to generate diverse, human-readable adversarial suffixes, improving attack efficacy in models like GPT-3.5 and Gemma 7B. Our findings demonstrate that LLMs can be tricked by sophisticated adversaries into producing harmful responses with human-readable adversarial prompts and that there exists a scope for improvement when it comes to robust LLMs. 

**Abstract (ZH)**: 以往关于大规模语言模型（LLM）漏洞的研究大多依赖于无意义的对抗性提示，这些提示很容易被自动化检测方法识别。我们通过关注可读性较强的对抗性提示来解决这一问题，这种提示更加现实且更具威胁性。我们的主要贡献包括基于情境驱动的攻击，利用电影剧本创建上下文相关且可读性较强的对抗性提示，能够成功欺骗LLM；对抗性后缀转换技术，将无意义的对抗性后缀转化为有实际意义的文本；以及通过p-核子采样生成多样且可读性较强的对抗性后缀的AdvPrompter方法，提升如GPT-3.5和Gemma 7B等模型的攻击效果。我们的研究结果表明，LLM可以被具有高技术水平的对手通过可读性较强的对抗性提示欺骗，使其生成有害响应，同时也表明在构建鲁棒的LLM方面存在改进的空间。 

---
# Deliberative Alignment: Reasoning Enables Safer Language Models 

**Title (ZH)**: 审慎对齐：推理使语言模型更安全 

**Authors**: Melody Y. Guan, Manas Joglekar, Eric Wallace, Saachi Jain, Boaz Barak, Alec Heylar, Rachel Dias, Andrea Vallone, Hongyu Ren, Jason Wei, Hyung Won Chung, Sam Toyer, Johannes Heidecke, Alex Beutel, Amelia Glaese  

**Link**: [PDF](https://arxiv.org/pdf/2412.16339)  

**Abstract**: As large-scale language models increasingly impact safety-critical domains, ensuring their reliable adherence to well-defined principles remains a fundamental challenge. We introduce Deliberative Alignment, a new paradigm that directly teaches the model safety specifications and trains it to explicitly recall and accurately reason over the specifications before answering. We used this approach to align OpenAI's o-series models, and achieved highly precise adherence to OpenAI's safety policies, without requiring human-written chain-of-thoughts or answers. Deliberative Alignment pushes the Pareto frontier by simultaneously increasing robustness to jailbreaks while decreasing overrefusal rates, and also improves out-of-distribution generalization. We demonstrate that reasoning over explicitly specified policies enables more scalable, trustworthy, and interpretable alignment. 

**Abstract (ZH)**: 随着大规模语言模型在安全关键领域的影响日益增大，确保模型可靠地遵守明确定义的原则仍然是一个基本的挑战。我们提出了审慎对齐（Deliberative Alignment）这一新范式，该范式直接向模型教授安全规范，并训练模型在回答之前明确回忆并准确推理这些规范。我们采用此方法对OpenAI的o系列模型进行了对齐，并在无需人类编写推理过程或答案的情况下实现了对OpenAI安全政策的高度精确遵守。审慎对齐在同时提高对抗破坏性行为的鲁棒性的同时降低了过度拒绝率，并改善了模型的离分布泛化能力。我们展示了明确规定的政策推理能够实现更具扩展性、可信性和可解释性的对齐。 

---
# VirusT5: Harnessing Large Language Models to Predicting SARS-CoV-2 Evolution 

**Title (ZH)**: VirusT5：利用大型语言模型预测SARS-CoV-2演化 

**Authors**: Vishwajeet Marathe, Deewan Bajracharya, Changhui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2412.16262)  

**Abstract**: During a virus's evolution,various regions of the genome are subjected to distinct levels of functional this http URL with factors like codon bias and DNA repair efficiency,these constraints contribute to unique mutation patterns within the genome or a specific gene. In this project, we harnessed the power of Large Language Models(LLMs) to predict the evolution of SARS-CoV-2. By treating the mutation process from one generation to the next as a translation task, we trained a transformer model, called VirusT5, to capture the mutation patterns underlying SARS-CoV-2 evolution. We evaluated the VirusT5's ability to detect these mutation patterns including its ability to identify mutation hotspots and explored the potential of using VirusT5 to predict future virus variants. Our findings demonstrate the feasibility of using a large language model to model viral evolution as a translation process. This study establishes the groundbreaking concept of "mutation-as-translation," paving the way for new methodologies and tools for combating virus threats 

**Abstract (ZH)**: 在病毒进化的过程中，基因组的不同区域会受到不同程度的功能约束，这些约束与密码子偏倚和DNA修复效率等因素有关，这些限制因素共同导致了基因组或特定基因内独特的突变模式。在本项目中，我们利用大型语言模型（LLMs）预测了新冠病毒（SARS-CoV-2）的进化过程。通过将一代到下一代的突变过程视为翻译任务，我们训练了一个名为VirusT5的变换器模型，以捕捉SARS-CoV-2进化过程中的突变模式。我们评估了VirusT5检测这些突变模式的能力，包括识别热点突变的能力，并探索了使用VirusT5预测未来病毒变种的潜力。我们的研究结果表明，使用大型语言模型将病毒进化建模为翻译过程是可行的。本研究提出了“突变即翻译”的新概念，为抗击病毒威胁提供了新的方法和工具。 

---
# Adaptive Large Language Models By Layerwise Attention Shortcuts 

**Title (ZH)**: 逐层注意力捷径的自适应大型语言模型 

**Authors**: Prateek Verma, Mert Pilanci  

**Link**: [PDF](https://arxiv.org/pdf/2409.10870)  

**Abstract**: Transformer architectures are the backbone of the modern AI revolution. However, they are based on simply stacking the same blocks in dozens of layers and processing information sequentially from one block to another. In this paper, we propose to challenge this and introduce adaptive computations for LLM-like setups, which allow the final layer to attend to all of the intermediate layers as it deems fit through the attention mechanism, thereby introducing computational \textbf{attention shortcuts}. These shortcuts can thus make the architecture depth and context adaptive. We showcase four different datasets, namely acoustic tokens, natural language, and symbolic music, and we achieve superior performance for GPT-like architecture. We give evidence via attention maps that the models learn complex dependencies across layers that are adaptive in context and depth depending on the input tokens. 

**Abstract (ZH)**: Transformer 架构是现代人工智能革命的核心。然而，它们通常是通过简单地堆叠相同的模块并在数十层中顺序处理信息来构建的。在这篇论文中，我们提出了一种挑战这一现状的方法，并介绍了为类似大规模语言模型的设置中引入自适应计算，使得最终层能够通过注意机制适当地关注所有中间层，从而引入计算上的 **注意捷径**。这些捷径可以使架构在深度和上下文方面具有自适应性。我们展示了四个不同的数据集，包括声学标记、自然语言和符号音乐，并对类似 GPT 的架构实现了更好的性能。通过注意力图展示了模型在输入标记的上下文和深度依赖关系上学习到复杂关联的能力。 

---
# Deliberation in Latent Space via Differentiable Cache Augmentation 

**Title (ZH)**: 通过可微缓存增强在潜在空间中的 deliberation 

**Authors**: Luyang Liu, Jonas Pfeiffer, Jiaxing Wu, Jun Xie, Arthur Szlam  

**Link**: [PDF](https://arxiv.org/pdf/2412.17747)  

**Abstract**: Techniques enabling large language models (LLMs) to "think more" by generating and attending to intermediate reasoning steps have shown promise in solving complex problems. However, the standard approaches generate sequences of discrete tokens immediately before responding, and so they can incur significant latency costs and be challenging to optimize. In this work, we demonstrate that a frozen LLM can be augmented with an offline coprocessor that operates on the model's key-value (kv) cache. This coprocessor augments the cache with a set of latent embeddings designed to improve the fidelity of subsequent decoding. We train this coprocessor using the language modeling loss from the decoder on standard pretraining data, while keeping the decoder itself frozen. This approach enables the model to learn, in an end-to-end differentiable fashion, how to distill additional computation into its kv-cache. Because the decoder remains unchanged, the coprocessor can operate offline and asynchronously, and the language model can function normally if the coprocessor is unavailable or if a given cache is deemed not to require extra computation. We show experimentally that when a cache is augmented, the decoder achieves lower perplexity on numerous subsequent tokens. Furthermore, even without any task-specific training, our experiments demonstrate that cache augmentation consistently reduces perplexity and improves performance across a range of reasoning-intensive tasks. 

**Abstract (ZH)**: 通过生成和关注中间推理步骤，使大型语言模型（LLM）能够“更好地思考”的技术在解决复杂问题方面显示出了潜力。然而，标准方法在响应前立即生成离散的令牌序列，因此会引发显著的延迟成本并使其优化变得具有挑战性。在这项工作中，我们证明了一个冻结的LLM可以通过一个操作于模型的关键值（kv）缓存的离线协处理器来增强，该协处理器借助一组旨在提高后续解码准确性的潜在嵌入来增强缓存。我们通过在标准预训练数据上使用解码器的语言模型损失来训练这个协处理器，同时保持解码器本身不变。这种方法使模型能够在端到端可微分的方式下学习如何将额外的计算提炼到其kv缓存中。由于解码器未改变，协处理器可以在离线和异步模式下运行，并且如果协处理器不可用或某个缓存不认为需要额外计算，语言模型仍然可以正常运行。我们的实验结果表明，当对缓存进行增广时，解码器在后续多个令牌上的困惑度较低。此外，即使未经任何特定任务的训练，我们的实验也表明缓存增广能够一致地降低困惑度并在各种推理密集型任务中提高性能。 

---
# Knowledge Editing through Chain-of-Thought 

**Title (ZH)**: 通过步步推理的知识编辑 

**Authors**: Changyue Wang, Weihang Su, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.17727)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across a wide range of natural language processing (NLP) tasks. However, keeping these models up-to-date with evolving world knowledge remains a significant challenge due to the high costs of frequent retraining. To address this challenge, knowledge editing techniques have emerged to update LLMs with new information without rebuilding the model from scratch. Among these, the in-context editing paradigm stands out for its effectiveness in integrating new knowledge while preserving the model's original capabilities. Despite its potential, existing in-context knowledge editing methods are often task-specific, focusing primarily on multi-hop QA tasks using structured knowledge triples. Moreover, their reliance on few-shot prompting for task decomposition makes them unstable and less effective in generalizing across diverse tasks.
In response to these limitations, we propose EditCoT, a novel knowledge editing framework that flexibly and efficiently updates LLMs across various tasks without retraining. EditCoT works by generating a chain-of-thought (CoT) for a given input and then iteratively refining this CoT process using a CoT editor based on updated knowledge. We evaluate EditCoT across a diverse range of benchmarks, covering multiple languages and tasks. The results demonstrate that our approach achieves state-of-the-art performance while offering superior generalization, effectiveness, and stability compared to existing methods, marking a significant advancement in the field of knowledge updating. Code and data are available at: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经在广泛自然语言处理（NLP）任务中展现出了卓越的能力。然而，由于频繁重新训练的成本高昂，使得保持这些模型与不断演化的世界知识同步成为一项重大挑战。为应对这一挑战，知识编辑技术应运而生，以便无需从头重新构建模型即可更新LLMs的新信息。在这些方法中，基于上下文的编辑范式因其在集成新知识的同时保持模型原有能力的有效性而脱颖而出。尽管其潜力巨大，但现有的基于上下文的知识编辑方法往往局限于特定任务，主要聚焦于使用结构化知识三元组的多跳问答任务。此外，它们依赖于少量示例的提示来进行任务分解，使得它们在多样任务上的泛化能力较弱且效果较差。

为应对这些限制，我们提出了EditCoT，这是一种新颖的知识编辑框架，可以灵活且高效地在各种任务中更新LLMs，无需重新训练。EditCoT通过为给定输入生成链式思考（CoT）过程，然后使用基于更新知识的CoT编辑器迭代优化这个CoT过程来工作。我们跨多种基准评估了EditCoT，涵盖了多种语言和任务。实验结果表明，我们的方法在性能、泛化能力和稳定性方面均优于现有方法，标志着知识更新领域的重要进步。相关代码和数据可在以下链接获取：this <https://your-link-url.com> URL。 

---
# Generating Completions for Fragmented Broca's Aphasic Sentences Using Large Language Models 

**Title (ZH)**: 使用大型语言模型生成断言性布罗卡失语症句子的完成 

**Authors**: Sijbren van Vaals, Yevgen Matusevych, Frank Tsiwah  

**Link**: [PDF](https://arxiv.org/pdf/2412.17669)  

**Abstract**: Broca's aphasia is a type of aphasia characterized by non-fluent, effortful and fragmented speech production with relatively good comprehension. Since traditional aphasia treatment methods are often time-consuming, labour-intensive, and do not reflect real-world conversations, applying natural language processing based approaches such as Large Language Models (LLMs) could potentially contribute to improving existing treatment approaches. To address this issue, we explore the use of sequence-to-sequence LLMs for completing fragmented Broca's aphasic sentences. We first generate synthetic Broca's aphasic data using a rule-based system designed to mirror the linguistic characteristics of Broca's aphasic speech. Using this synthetic data, we then fine-tune four pre-trained LLMs on the task of completing fragmented sentences. We evaluate our fine-tuned models on both synthetic and authentic Broca's aphasic data. We demonstrate LLMs' capability for reconstructing fragmented sentences, with the models showing improved performance with longer input utterances. Our result highlights the LLMs' potential in advancing communication aids for individuals with Broca's aphasia and possibly other clinical populations. 

**Abstract (ZH)**: 布罗卡失语症是一种表现为不流利、费力和片段化的言语产生，但理解能力相对较好的语言障碍类型。由于传统的失语症治疗方法往往耗时、费力，并不反映现实对话，因此应用基于自然语言处理的方法，如大型语言模型（LLMs），有可能改善现有的治疗方法。为了解决这一问题，我们探索了使用序列到序列的LLMs来完成片段化的布罗卡失语症句子。我们首先使用基于规则的系统生成合成的布罗卡失语症数据，该系统旨在模仿布罗卡失语症患者的语言特点。然后，我们使用这些合成数据微调四种预训练的LLMs，使其能够完成片段化的句子。我们分别在合成数据和真实的布罗卡失语症数据上评估了微调后的模型。我们展示了LLMs在重构片段化句子方面的能力，模型的性能随着输入句子长度的增加而改进。我们的结果突显了LLMs在促进布罗卡失语症患者及其他临床人群的交流辅助方面的能力。 

---
# DiffusionAttacker: Diffusion-Driven Prompt Manipulation for LLM Jailbreak 

**Title (ZH)**: DiffusionAttacker：由扩散驱动的提示操纵以实现大语言模型脱笼攻击 

**Authors**: Hao Wang, Hao Li, Junda Zhu, Xinyuan Wang, Chengwei Pan, MinLie Huang, Lei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2412.17522)  

**Abstract**: Large Language Models (LLMs) are susceptible to generating harmful content when prompted with carefully crafted inputs, a vulnerability known as LLM jailbreaking. As LLMs become more powerful, studying jailbreak methods is critical to enhancing security and aligning models with human values. Traditionally, jailbreak techniques have relied on suffix addition or prompt templates, but these methods suffer from limited attack diversity. This paper introduces DiffusionAttacker, an end-to-end generative approach for jailbreak rewriting inspired by diffusion models. Our method employs a sequence-to-sequence (seq2seq) text diffusion model as a generator, conditioning on the original prompt and guiding the denoising process with a novel attack loss. Unlike previous approaches that use autoregressive LLMs to generate jailbreak prompts, which limit the modification of already generated tokens and restrict the rewriting space, DiffusionAttacker utilizes a seq2seq diffusion model, allowing more flexible token modifications. This approach preserves the semantic content of the original prompt while producing harmful content. Additionally, we leverage the Gumbel-Softmax technique to make the sampling process from the diffusion model's output distribution differentiable, eliminating the need for iterative token search. Extensive experiments on Advbench and Harmbench demonstrate that DiffusionAttacker outperforms previous methods across various evaluation metrics, including attack success rate (ASR), fluency, and diversity. 

**Abstract (ZH)**: 大语言模型（LLMs）在受到精心设计的输入提示时，可能会生成有害内容，这种漏洞被称为LLM的逃逸。随着LLMs变得越来越强大，研究逃逸方法对于提高安全性和使模型与人类价值观保持一致变得至关重要。传统上，逃逸技术依赖于后缀添加或提示模板，但这些方法无法提供多样化的攻击方式。本论文介绍了一种名为DiffusionAttacker的端到端生成方法，该方法受到了扩散模型的启发，用于逃逸重写。我们的方法采用了一个条件生成模型——序列到序列（Seq2Seq）文本扩散模型，并通过一种新的攻击损失来引导去噪过程。与以往使用自回归LLMs生成逃逸提示的方法不同，这种方法仅限于修改生成的令牌，限制了重写的空间，而DiffusionAttacker利用Seq2Seq扩散模型，提供了更加灵活的令牌修改能力。这种方法在保持原始提示的语义内容的同时生成了有害内容。此外，我们利用Gumbel-Softmax技术使从扩散模型输出分布中采样的过程可微，从而消除了迭代令牌搜索的需要。在Advbench和Harmbench上的广泛实验表明，DiffusionAttacker在各种评估指标，包括攻击成功率（ASR）、流畅性和多样性方面，均优于先前的方法。 

---
# Measuring Contextual Informativeness in Child-Directed Text 

**Title (ZH)**: 衡量面向儿童文本的情境 informativeness 

**Authors**: Maria Valentini, Téa Wright, Ali Marashian, Jennifer Weber, Eliana Colunga, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2412.17427)  

**Abstract**: To address an important gap in creating children's stories for vocabulary enrichment, we investigate the automatic evaluation of how well stories convey the semantics of target vocabulary words, a task with substantial implications for generating educational content. We motivate this task, which we call measuring contextual informativeness in children's stories, and provide a formal task definition as well as a dataset for the task. We further propose a method for automating the task using a large language model (LLM). Our experiments show that our approach reaches a Spearman correlation of 0.4983 with human judgments of informativeness, while the strongest baseline only obtains a correlation of 0.3534. An additional analysis shows that the LLM-based approach is able to generalize to measuring contextual informativeness in adult-directed text, on which it also outperforms all baselines. 

**Abstract (ZH)**: 为解决为词汇丰富而创建儿童故事中的一个重要空白，我们研究了自动评估故事传达目标词汇意义效果的方法，这一任务对生成教育内容具有重大意义。我们阐述了这一任务的重要性，即衡量儿童故事中的上下文信息量，并提出了该任务的正式定义以及相关数据集。进一步地，我们提出了一种使用大规模语言模型（LLM）自动化执行该任务的方法。我们的实验结果显示，我们的方法与人类对信息量的判断之间的 Spearman 相关系数达到了 0.4983，而最强的基线方法只有 0.3534。此外的分析表明，基于大规模语言模型的方法能够泛化到成人导向文本的上下文信息量衡量，并在该任务上也优于所有基线方法。 

---
# Just What You Desire: Constrained Timeline Summarization with Self-Reflection for Enhanced Relevance 

**Title (ZH)**: 恰如您的所愿：自省约束时间线摘要以提升相关性 

**Authors**: Muhammad Reza Qorib, Qisheng Hu, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2412.17408)  

**Abstract**: Given news articles about an entity, such as a public figure or organization, timeline summarization (TLS) involves generating a timeline that summarizes the key events about the entity. However, the TLS task is too underspecified, since what is of interest to each reader may vary, and hence there is not a single ideal or optimal timeline. In this paper, we introduce a novel task, called Constrained Timeline Summarization (CTLS), where a timeline is generated in which all events in the timeline meet some constraint. An example of a constrained timeline concerns the legal battles of Tiger Woods, where only events related to his legal problems are selected to appear in the timeline. We collected a new human-verified dataset of constrained timelines involving 47 entities and 5 constraints per entity. We propose an approach that employs a large language model (LLM) to summarize news articles according to a specified constraint and cluster them to identify key events to include in a constrained timeline. In addition, we propose a novel self-reflection method during summary generation, demonstrating that this approach successfully leads to improved performance. 

**Abstract (ZH)**: 给定关于某一实体（如公众人物或组织）的新闻文章，时间线总结（TLS）涉及生成一个能够概括该实体关键事件的时间线。然而，TLS任务过于模糊，因为不同读者感兴趣的事件可能不同，因此并不存在一个理想的或最优的时间线。本文我们引入了一个新的任务，称为约束时间线总结（CTLS），其中生成的时间线中的所有事件都要满足某些约束条件。例如，在泰格·伍兹的法律斗争案例中，仅与其法律问题相关的事件被选入时间线。我们收集了一个新的由人工验证的数据集，其中包括47个实体和每个实体5个约束条件。我们提出了一种方法，利用大型语言模型（LLM）根据指定的约束条件总结新闻文章，并对其进行聚类以识别要包含在约束时间线中的关键事件。此外，我们在总结生成过程中提出了一个新颖的自省方法，证明该方法能够显著提高性能。 

---
# WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models 

**Title (ZH)**: 战士程序员：从专家对决中学习以增强代码大型语言模型 

**Authors**: Huawen Feng, Pu Zhao, Qingfeng Sun, Can Xu, Fangkai Yang, Lu Wang, Qianli Ma, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17395)  

**Abstract**: Despite recent progress achieved by code large language models (LLMs), their remarkable abilities are largely dependent on fine-tuning on the high-quality data, posing challenges for data collection and annotation. To address this, current methods often design various data flywheels to gather complex code instructions, enabling models to handle more intricate tasks. However, these approaches typically rely on off-the-shelf datasets and data augmentation from the limited pool of proprietary LLMs (e.g., Claude, GPT4, and so on), which limits the diversity of the constructed data and makes it prone to systemic biases. In this paper, we propose WarriorCoder which learns from expert battles to address these limitations. Specifically, we create an arena for current expert code LLMs, where each model challenges and responds to others' challenges, with evaluations conducted by uninvolved judge models. This competitive framework generates novel training data constructed from scratch, harnessing the strengths of all participants. Experimental results demonstrate that WarriorCoder achieves competitive performance compared to previous methods, even without relying on proprietary LLMs. 

**Abstract (ZH)**: 尽管近期代码大规模语言模型（LLMs）取得了进展，它们的卓越能力很大程度上依赖于在高质量数据上进行微调，这给数据收集和标注带来了挑战。为了应对这一问题，当前的方法常常设计各种数据飞轮以采集复杂的代码指令，从而使模型能够处理更加复杂的任务。然而，这些方法通常依赖于现成的数据集和有限的专有LLM（如Claude、GPT4等）的数据增强，这限制了构建数据的多样性，并使其容易受到系统性偏见的影响。在本文中，我们提出了一种名为WarriorCoder的方法，旨在通过专家战斗来解决这些局限性。具体而言，我们创建了一个供当前代码LLM专家们竞技的擂台，在这个擂台上，每个模型挑战其他模型并回应挑战，且这些评估由未参与的裁判模型进行。这种竞争框架生成了从头构建的新颖训练数据，利用了所有参与者的优点。实验结果表明，WarriorCoder在不依赖于专有LLM的情况下，能够达到与先前方法相当的性能。 

---
# Interweaving Memories of a Siamese Large Language Model 

**Title (ZH)**: 泰国大型语言模型的记忆交织 

**Authors**: Xin Song, Zhikai Xue, Guoxiu He, Jiawei Liu, Wei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2412.17383)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods optimize large language models (LLMs) by modifying or introducing a small number of parameters to enhance alignment with downstream tasks. However, they can result in catastrophic forgetting, where LLMs prioritize new knowledge at the expense of comprehensive world knowledge. A promising approach to mitigate this issue is to recall prior memories based on the original knowledge. To this end, we propose a model-agnostic PEFT framework, IMSM, which Interweaves Memories of a Siamese Large Language Model. Specifically, our siamese LLM is equipped with an existing PEFT method. Given an incoming query, it generates two distinct memories based on the pre-trained and fine-tuned parameters. IMSM then incorporates an interweaving mechanism that regulates the contributions of both original and enhanced memories when generating the next token. This framework is theoretically applicable to all open-source LLMs and existing PEFT methods. We conduct extensive experiments across various benchmark datasets, evaluating the performance of popular open-source LLMs using the proposed IMSM, in comparison to both classical and leading PEFT methods. Our findings indicate that IMSM maintains comparable time and space efficiency to backbone PEFT methods while significantly improving performance and effectively mitigating catastrophic forgetting. 

**Abstract (ZH)**: 参数高效的微调（PEFT）方法通过修改或引入少量参数来优化大型语言模型（LLMs），以增强其与下游任务的一致性。然而，这种方法可能会导致灾难性遗忘，即LLMs在获取新知识的同时牺牲了全面的世界知识。缓解这一问题的一个有前景的方法是基于原始知识重新激活先前的记忆。为了解决这个问题，我们提出了一种模型无关的PEFT框架，即Interweaving Memories of a Siamese Large Language Model（ISMSM），它将双胞胎大型语言模型中的记忆相互交织。具体而言，我们的双胞胎LLM配备了现有的PEFT方法。对于每个新的查询，它根据预训练和微调参数生成两个不同的记忆。ISMSM则引入了一个交织机制，该机制在生成下一个token时调节原始记忆和增强记忆的贡献。该框架理论上适用于所有开源LLM和现有的PEFT方法。我们在多个基准数据集上进行了广泛的实验，使用所提出的ISMSM评估了流行开源LLM的表现，并将其与经典的和领先的PEFT方法进行了比较。研究结果显示，ISMSM在保持与主PEFT方法类似的时间和空间效率的同时，显著改善了性能并有效缓解了灾难性遗忘问题。 

---
# A Dual-Perspective Metaphor Detection Framework Using Large Language Models 

**Title (ZH)**: 使用大规模语言模型的双重视角隐喻检测框架 

**Authors**: Yujie Lin, Jingyao Liu, Yan Gao, Ante Wang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2412.17332)  

**Abstract**: Metaphor detection, a critical task in natural language processing, involves identifying whether a particular word in a sentence is used metaphorically. Traditional approaches often rely on supervised learning models that implicitly encode semantic relationships based on metaphor theories. However, these methods often suffer from a lack of transparency in their decision-making processes, which undermines the reliability of their predictions. Recent research indicates that LLMs (large language models) exhibit significant potential in metaphor detection. Nevertheless, their reasoning capabilities are constrained by predefined knowledge graphs. To overcome these limitations, we propose DMD, a novel dual-perspective framework that harnesses both implicit and explicit applications of metaphor theories to guide LLMs in metaphor detection and adopts a self-judgment mechanism to validate the responses from the aforementioned forms of guidance. In comparison to previous methods, our framework offers more transparent reasoning processes and delivers more reliable predictions. Experimental results prove the effectiveness of DMD, demonstrating state-of-the-art performance across widely-used datasets. 

**Abstract (ZH)**: 元喻检测是自然语言处理中的一项关键任务，涉及识别句子中的某个词语是否被用于隐喻。传统方法通常依赖于基于隐喻理论的监督学习模型，隐式地编码语义关系。然而，这些方法往往在决策过程的透明性方面存在不足，这削弱了其预测的可靠性。近期研究表明，大规模语言模型（LLMs）在隐喻检测方面具有显著潜力。然而，它们的推理能力受限于预定义的知识图谱。为克服这些限制，我们提出了一种名为DMD的新型双视角框架，该框架结合了隐喻理论的显性和隐性应用，以指导LLMs进行隐喻检测，并采用自我判断机制来验证上述指导方式的响应。与以往方法相比，该框架提供了更透明的推理过程和更可靠的预测。实验结果证明了DMD的有效性，其在广泛使用的数据集上展示了最先进的性能。 

---
# Lies, Damned Lies, and Distributional Language Statistics: Persuasion and Deception with Large Language Models 

**Title (ZH)**: 谎言、伪证与分布型语言统计：大规模语言模型中的说服与欺骗 

**Authors**: Cameron R. Jones, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2412.17128)  

**Abstract**: Large Language Models (LLMs) can generate content that is as persuasive as human-written text and appear capable of selectively producing deceptive outputs. These capabilities raise concerns about potential misuse and unintended consequences as these systems become more widely deployed. This review synthesizes recent empirical work examining LLMs' capacity and proclivity for persuasion and deception, analyzes theoretical risks that could arise from these capabilities, and evaluates proposed mitigations. While current persuasive effects are relatively small, various mechanisms could increase their impact, including fine-tuning, multimodality, and social factors. We outline key open questions for future research, including how persuasive AI systems might become, whether truth enjoys an inherent advantage over falsehoods, and how effective different mitigation strategies may be in practice. 

**Abstract (ZH)**: 大规模语言模型（LLMs）能够生成与人类撰写的文本同样具有说服力的内容，并且似乎能够有选择地产生欺骗性输出。这些能力引发了对其潜在滥用和意外后果的担忧，尤其是在这些系统更加广泛部署的情况下。本综述综合了近期关于LLMs的说服能力和倾向性欺骗的研究成果，分析了这些能力可能引发的理论风险，并评估了已提出的缓解措施。虽然当前的说服效果相对较小，但多种机制可能会增加其影响，包括模型微调、多模态和社交因素。我们概述了未来研究中的关键开放问题，包括如何使说服性人工智能系统变得更加有效，真相是否天然具有优势，以及不同缓解策略的实际有效性如何。 

---
# The HalluRAG Dataset: Detecting Closed-Domain Hallucinations in RAG Applications Using an LLM's Internal States 

**Title (ZH)**: 《HalluRAG数据集：使用大型语言模型内部状态检测RAG应用程序中的封闭域幻觉》 

**Authors**: Fabian Ridder, Malte Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2412.17056)  

**Abstract**: Detecting hallucinations in large language models (LLMs) is critical for enhancing their reliability and trustworthiness. Most research focuses on hallucinations as deviations from information seen during training. However, the opaque nature of an LLM's parametric knowledge complicates the understanding of why generated texts appear ungrounded: The LLM might not have picked up the necessary knowledge from large and often inaccessible datasets, or the information might have been changed or contradicted during further training. Our focus is on hallucinations involving information not used in training, which we determine by using recency to ensure the information emerged after a cut-off date. This study investigates these hallucinations by detecting them at sentence level using different internal states of various LLMs. We present HalluRAG, a dataset designed to train classifiers on these hallucinations. Depending on the model and quantization, MLPs trained on HalluRAG detect hallucinations with test accuracies ranging up to 75 %, with Mistral-7B-Instruct-v0.1 achieving the highest test accuracies. Our results show that IAVs detect hallucinations as effectively as CEVs and reveal that answerable and unanswerable prompts are encoded differently as separate classifiers for these categories improved accuracy. However, HalluRAG showed some limited generalizability, advocating for more diversity in datasets on hallucinations. 

**Abstract (ZH)**: 检测大规模语言模型（LLMs）中的幻觉对于提升其可靠性和可信度至关重要。大多数研究侧重于检测与训练期间看到的信息偏差的幻觉。然而，LLM参数化知识的不透明性使得理解生成文本为何显得缺乏根据变得更加复杂：LLM可能未能从广泛且往往难以访问的数据集中获取必要的知识，或者在进一步训练过程中信息可能已被改变或被矛盾所取代。我们关注的是训练时未使用的信息引发的幻觉，通过使用时间递进来确定这些信息是在某个截断日期之后出现的。本研究通过在不同LLM的内部状态层面检测幻觉，来调查这些幻觉。我们介绍了HalluRAG数据集，用于训练分类器以识别这些幻觉。根据不同模型和量化方法，训练在HalluRAG上进行的MLP在测试分类准确性上可达75%，Mistral-7B-Instruct-v0.1的表现最佳。我们的研究结果表明，IAVs在检测幻觉方面的效果与CEVs相当，并揭示了可回答和不可回答提示的编码方式存在差异，这在两种类别的分类器中得到了体现，提高了准确性。然而，HalluRAG展示了一定的有限泛化能力，这表明在幻觉数据集方面需要更多样性。 

---
# Shaping the Safety Boundaries: Understanding and Defending Against Jailbreaks in Large Language Models 

**Title (ZH)**: 塑造安全边界：理解并抵御大型语言模型的越狱攻击 

**Authors**: Lang Gao, Xiangliang Zhang, Preslav Nakov, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.17034)  

**Abstract**: Jailbreaking in Large Language Models (LLMs) is a major security concern as it can deceive LLMs to generate harmful text. Yet, there is still insufficient understanding of how jailbreaking works, which makes it hard to develop effective defense strategies. We aim to shed more light into this issue: we conduct a detailed large-scale analysis of seven different jailbreak methods and find that these disagreements stem from insufficient observation samples. In particular, we introduce \textit{safety boundary}, and we find that jailbreaks shift harmful activations outside that safety boundary, where LLMs are less sensitive to harmful information. We also find that the low and the middle layers are critical in such shifts, while deeper layers have less impact. Leveraging on these insights, we propose a novel defense called \textbf{Activation Boundary Defense} (ABD), which adaptively constrains the activations within the safety boundary. We further use Bayesian optimization to selectively apply the defense method to the low and the middle layers. Our experiments on several benchmarks show that ABD achieves an average DSR of over 98\% against various forms of jailbreak attacks, with less than 2\% impact on the model's general capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）的越狱是一种重要的安全问题，因为它可以使LLMs生成有害文本。然而，关于越狱机制的理解仍然不足，这使得难以开发有效的防御策略。我们旨在更深入地探讨这一问题：我们对七种不同的越狱方法进行了详细的大规模分析，并发现这些分歧源于观察样本的不足。特别地，我们引入了“安全边界”的概念，并发现越狱将有害激活转移到了这个安全边界之外，而在这个区域，LLMs 对有害信息的敏感度较低。我们还发现，低层和中间层在这些变化中起着关键作用，而深层结构的影响较小。基于这些洞察，我们提出了一种新颖的防御方法，称为**激活边界防御**（ABD），它能够自适应地将激活值限制在安全边界内。我们进一步利用贝叶斯优化策略，选择性地将防御方法应用于低层和中间层。在多个基准测试上的实验结果显示，ABD 方法能在各种形式的越狱攻击中实现超过 98% 的有效防御率，同时对模型的整体性能影响不到 2%。 

---
# MINTQA: A Multi-Hop Question Answering Benchmark for Evaluating LLMs on New and Tail Knowledge 

**Title (ZH)**: MINTQA：评价大型语言模型在新颖和特定知识库中进行多跳问答的能力基准 

**Authors**: Jie He, Nan Hu, Wanqiu Long, Jiaoyan Chen, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2412.17032)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in various reasoning tasks but face significant challenges with complex, knowledge-intensive multi-hop queries, particularly those involving new or long-tail knowledge. Existing benchmarks often fail to fully address these challenges. To bridge this gap, we introduce MINTQA (Multi-hop Question Answering on New and Tail Knowledge), a comprehensive benchmark to evaluate LLMs' capabilities in multi-hop reasoning across four critical dimensions: question handling strategy, sub-question generation, retrieval-augmented generation, and iterative or dynamic decomposition and retrieval. MINTQA comprises 10,479 question-answer pairs for evaluating new knowledge and 17,887 pairs for assessing long-tail knowledge, with each question equipped with corresponding sub-questions and answers. Our systematic evaluation of 22 state-of-the-art LLMs on MINTQA reveals significant limitations in their ability to handle complex knowledge base queries, particularly in handling new or unpopular knowledge. Our findings highlight critical challenges and offer insights for advancing multi-hop reasoning capabilities. The MINTQA benchmark is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种推理任务中展现了出色的性能，但在应对复杂、知识密集型的多跳查询，尤其是涉及新知识或长尾知识的情况时，仍面临重大挑战。现有的基准测试往往未能充分解决这些问题。为解决这一差距，我们引入了MINTQA（多跳问答：新知识和长尾知识），这是一个综合基准，用于评估LLMs在多跳推理方面的能力，涵盖四个关键维度：问题处理策略、子问题生成、检索增强生成以及迭代或动态分解与检索。MINTQA 包含10,479个用于评估新知识的问题-答案对和17,887个用于评估长尾知识的问题-答案对，每个问题都配有相应的子问题和答案。我们在MINTQA 上对22个最先进的LLMs 进行系统的评估，揭示了它们在处理复杂知识库查询方面的显著局限性，尤其是处理新知识或不常用知识的能力。我们的研究结果突出了关键挑战并提供了提升多跳推理能力的见解。MINTQA 基准测试可在以下链接获取：[这里](this https URL)。 

---
# Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework 

**Title (ZH)**: 亚里士多德：基于逻辑完备分解-搜索-解决框架的逻辑推理掌握 

**Authors**: Jundong Xu, Hao Fei, Meng Luo, Qian Liu, Liangming Pan, William Yang Wang, Preslav Nakov, Mong-Li Lee, Wynne Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16953)  

**Abstract**: In the context of large language models (LLMs), current advanced reasoning methods have made impressive strides in various reasoning tasks. However, when it comes to logical reasoning tasks, major challenges remain in both efficacy and efficiency. This is rooted in the fact that these systems fail to fully leverage the inherent structure of logical tasks throughout the reasoning processes such as decomposition, search, and resolution. To address this, we propose a logic-complete reasoning framework, Aristotle, with three key components: Logical Decomposer, Logical Search Router, and Logical Resolver. In our framework, symbolic expressions and logical rules are comprehensively integrated into the entire reasoning process, significantly alleviating the bottlenecks of logical reasoning, i.e., reducing sub-task complexity, minimizing search errors, and resolving logical contradictions. The experimental results on several datasets demonstrate that Aristotle consistently outperforms state-of-the-art reasoning frameworks in both accuracy and efficiency, particularly excelling in complex logical reasoning scenarios. We will open-source all our code at this https URL. 

**Abstract (ZH)**: 在大型语言模型（LLMs）的背景下，当前先进的推理方法已在各种推理任务中取得了显著进展。然而，当涉及到逻辑推理任务时，这些系统在有效性和效率方面仍面临重大挑战。这些挑战根源于系统在推理过程（如分解、搜索和解决）中未能充分利用逻辑任务固有的结构。为解决这一问题，我们提出了一种逻辑完备推理框架Aristotle，该框架包含三个关键组件：逻辑分解器、逻辑搜索路由器和逻辑解决器。在我们的框架中，符号表达式和逻辑规则被全面整合到整个推理过程中，显著缓解了逻辑推理的瓶颈问题，包括降低子任务复杂性、最小化搜索错误和解决逻辑矛盾。在多个数据集上的实验结果表明，Aristotle在准确性和效率方面均优于现有的推理框架，特别是在复杂的逻辑推理场景中表现更为出色。我们将在此httpsURL开源所有代码。 

---
# A Career Interview Dialogue System using Large Language Model-based Dynamic Slot Generation 

**Title (ZH)**: 基于大规模语言模型的动态槽生成的职业访谈对话系统 

**Authors**: Ekai Hashimoto, Mikio Nakano, Takayoshi Sakurai, Shun Shiramatsu, Toshitake Komazaki, Shiho Tsuchiya  

**Link**: [PDF](https://arxiv.org/pdf/2412.16943)  

**Abstract**: This study aims to improve the efficiency and quality of career interviews conducted by nursing managers. To this end, we have been developing a slot-filling dialogue system that engages in pre-interviews to collect information on staff careers as a preparatory step before the actual interviews. Conventional slot-filling-based interview dialogue systems have limitations in the flexibility of information collection because the dialogue progresses based on predefined slot sets. We therefore propose a method that leverages large language models (LLMs) to dynamically generate new slots according to the flow of the dialogue, achieving more natural conversations. Furthermore, we incorporate abduction into the slot generation process to enable more appropriate and effective slot generation. To validate the effectiveness of the proposed method, we conducted experiments using a user simulator. The results suggest that the proposed method using abduction is effective in enhancing both information-collecting capabilities and the naturalness of the dialogue. 

**Abstract (ZH)**: 本研究旨在提高护理管理人员进行职业生涯访谈的效率和质量。为此，我们正在开发一种对话系统，该系统通过预先访谈收集员工职业生涯信息，为实际访谈做准备。传统的基于槽填充的面试对话系统在信息收集的灵活性方面存在局限性，因为对话是基于预定义的槽集进行的。因此，我们提出了一种方法，利用大型语言模型（LLMs）根据对话的流程动态生成新的槽，以实现更自然的对话。此外，我们将可推断逻辑融入槽生成过程，以实现更具针对性和有效性的槽生成。为了验证所提方法的有效性，我们使用用户仿真器进行了实验。结果表明，结合可推断逻辑的所提方法在提高信息收集能力和对话自然度方面是有效的。 

---
# Teaching LLMs to Refine with Tools 

**Title (ZH)**: 教学术大型语言模型使用工具进行优化与细化 

**Authors**: Dian Yu, Yuheng Zhang, Jiahao Xu, Tian Liang, Linfeng Song, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16871)  

**Abstract**: Large language models (LLMs) can refine their responses based on feedback, enabling self-improvement through iterative training or test-time refinement. However, existing methods predominantly focus on refinement within the same reasoning format, which may lead to non-correcting behaviors. We propose CaP, a novel approach that uses external tools to refine chain-of-thought (CoT) responses generated by the same or other LLMs. CaP employs a two-stage training process: supervised fine-tuning followed by preference optimization with DPO variants. Our observations highlight the critical role of preference optimization in enabling effective refinement. Additionally, we compare several sampling strategies to leverage CoT and tools at inference time. Experimental results demonstrate CaP's potential for effective cross-reasoning refinement and efficient inference. 

**Abstract (ZH)**: 大语言模型（LLMs）可以根据反馈改进其响应，从而通过迭代训练或测试时改进来实现自我提升。然而，现有方法主要集中在同一种推理格式内的改进上，这可能会导致非纠正性行为。我们提出了一种名为CaP的新方法，该方法使用外部工具来精炼由同一个或其它LLM生成的链式思维（CoT）响应。CaP采用两阶段训练过程：监督微调，随后是使用DPO变体进行偏好优化。我们的观察结果强调了偏好优化在实现有效改进中的关键作用。此外，我们还比较了几种采样策略，以便在推理时利用CoT和工具。实验结果表明，CaP在实现有效的跨推理改进和高效推理方面具有潜力。 

---
# Ask-Before-Detection: Identifying and Mitigating Conformity Bias in LLM-Powered Error Detector for Math Word Problem Solutions 

**Title (ZH)**: 在检测之前提问：识别并减轻基于大语言模型的数学应用题解决方案错误检测器中的遵从性偏差 

**Authors**: Hang Li, Tianlong Xu, Kaiqi Yang, Yucheng Chu, Yanling Chen, Yichi Song, Qingsong Wen, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16838)  

**Abstract**: The rise of large language models (LLMs) offers new opportunities for automatic error detection in education, particularly for math word problems (MWPs). While prior studies demonstrate the promise of LLMs as error detectors, they overlook the presence of multiple valid solutions for a single MWP. Our preliminary analysis reveals a significant performance gap between conventional and alternative solutions in MWPs, a phenomenon we term conformity bias in this work. To mitigate this bias, we introduce the Ask-Before-Detect (AskBD) framework, which generates adaptive reference solutions using LLMs to enhance error detection. Experiments on 200 examples of GSM8K show that AskBD effectively mitigates bias and improves performance, especially when combined with reasoning-enhancing techniques like chain-of-thought prompting. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的兴起为教育中的自动错误检测带来了新的机会，特别是在数学文字问题（MWPs）中。虽然先前的研究表明LLMs作为错误检测器的潜力巨大，但它们忽视了单一数学文字问题可能有多个有效解的事实。我们的初步分析揭示了MWPs中传统解法和替代解法之间显著的性能差异，这种现象在本文中我们将其称为一致性偏差。为了减轻这一偏差，我们引入了“检测前询问”（Ask-Before-Detect，简称AskBD）框架，利用LLMs生成适应性的参考解法，以增强错误检测的效果。实验结果显示，AskBD在减轻偏差和提高性能方面效果显著，特别是在结合如链式思考提示等增强推理的技术时表现尤为突出。 

---
# Quantum-Like Contextuality in Large Language Models 

**Title (ZH)**: 大型语言模型中的量子似的语境性 

**Authors**: Kin Ian Lo, Mehrnoosh Sadrzadeh, Shane Mansfield  

**Link**: [PDF](https://arxiv.org/pdf/2412.16806)  

**Abstract**: Contextuality is a distinguishing feature of quantum mechanics and there is growing evidence that it is a necessary condition for quantum advantage. In order to make use of it, researchers have been asking whether similar phenomena arise in other domains. The answer has been yes, e.g. in behavioural sciences. However, one has to move to frameworks that take some degree of signalling into account. Two such frameworks exist: (1) a signalling-corrected sheaf theoretic model, and (2) the Contextuality-by-Default (CbD) framework. This paper provides the first large scale experimental evidence for a yes answer in natural language. We construct a linguistic schema modelled over a contextual quantum scenario, instantiate it in the Simple English Wikipedia and extract probability distributions for the instances using the large language model BERT. This led to the discovery of 77,118 sheaf-contextual and 36,938,948 CbD contextual instances. We proved that the contextual instances came from semantically similar words, by deriving an equation between degrees of contextuality and Euclidean distances of BERT's embedding vectors. A regression model further reveals that Euclidean distance is indeed the best statistical predictor of contextuality. Our linguistic schema is a variant of the co-reference resolution challenge. These results are an indication that quantum methods may be advantageous in language tasks. 

**Abstract (ZH)**: 上下文性是量子力学的一个显著特征，越来越多的证据表明，它是实现量子优势的必要条件。为了利用这一特性，研究人员一直在询问其他领域是否存在类似现象。答案是肯定的，特别是在行为科学领域。但是，这需要采用一些信号传递的框架。目前存在两种这样的框架：（1）信号修正的层积范畴模型，和（2）默认上下文性（CbD）框架。本文提供了第一个大规模实验证据，证明在自然语言中存在上下文性现象的回答是肯定的。我们构建了一个基于上下文量子场景的语言模式，并在简体英语维基百科中实例化该模式，利用大规模语言模型BERT提取实例的概率分布。这一过程发现了77,118个层积上下文性和36,938,948个CbD上下文性实例。我们通过推导上下文性的程度和BERT嵌入向量欧几里得距离之间的方程，证明了上下文性实例来自语义相似的单词。进一步的回归模型表明，欧几里得距离确实是上下文性最好的统计预测器。我们的语言模式是一种同名词义消解挑战的变体。这些结果表明，量子方法可能在语言任务中具有优势。 

---
# NILE: Internal Consistency Alignment in Large Language Models 

**Title (ZH)**: NILE：大型语言模型内的一致性对齐 

**Authors**: Minda Hu, Qiyuan Zhang, Yufei Wang, Bowei He, Hongru Wang, Jingyan Zhou, Liangyou Li, Yasheng Wang, Chen Ma, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2412.16686)  

**Abstract**: As a crucial step to enhance LLMs alignment with human intentions, Instruction Fine-Tuning (IFT) has a high demand on dataset quality. However, existing IFT datasets often contain knowledge that is inconsistent with LLMs' internal knowledge learned from the pre-training phase, which can greatly affect the efficacy of IFT. To address this issue, we introduce NILE (iNternal consIstency aLignmEnt) framework, aimed at optimizing IFT datasets to unlock LLMs' capability further. NILE operates by eliciting target pre-trained LLM's internal knowledge corresponding to instruction data. The internal knowledge is leveraged to revise the answer in IFT datasets. Additionally, we propose a novel Internal Consistency Filtering (ICF) method to filter training samples, ensuring its high consistency with LLM's internal knowledge. Our experiments demonstrate that NILE-aligned IFT datasets sharply boost LLM performance across multiple LLM ability evaluation datasets, achieving up to 66.6% gain on Arena-Hard and 68.5% on Alpaca-Eval V2. Further analysis confirms that each component of the NILE}framework contributes to these substantial performance improvements, and provides compelling evidence that dataset consistency with pre-trained internal knowledge is pivotal for maximizing LLM potential. 

**Abstract (ZH)**: 为了增强大模型（LLMs）与人类意图的一致性，指令微调（IFT）对数据集的质量有着很高的要求。然而，现有的IFT数据集往往包含与预训练阶段学习的知识不一致的知识，这严重影响了IFT的效果。为了解决这一问题，我们提出了NILE（Internal Consistency Alignment）框架，旨在优化IFT数据集，进一步发挥大模型的能力。NILE通过提取目标预训练大模型内部与指令数据相对应的知识来实现这一目标。这些内部知识被用于修正IFT数据集中的答案。此外，我们还提出了一种新的内部一致性筛选（ICF）方法来筛选训练样本，确保其与大模型内部知识的高度一致性。我们的实验表明，经过NILE优化的IFT数据集在多个LLM能力评估数据集上显著提升了大模型的性能，在Arena-Hard数据集上达到了66.6%的提升，在Alpaca-Eval V2数据集上达到了68.5%的提升。进一步的分析证实，NILE框架的每个组成部分都对这些显著的性能提升做出了贡献，并提供了有力证据，表明数据集与预训练内部知识的一致性对于最大化大模型的潜力至关重要。 

---
# Evaluating the Performance of Large Language Models in Scientific Claim Detection and Classification 

**Title (ZH)**: 评估大型语言模型在科学研究声明检测与分类中的性能 

**Authors**: Tanjim Bin Faruk  

**Link**: [PDF](https://arxiv.org/pdf/2412.16486)  

**Abstract**: The pervasive influence of social media during the COVID-19 pandemic has been a double-edged sword, enhancing communication while simultaneously propagating misinformation. This \textit{Digital Infodemic} has highlighted the urgent need for automated tools capable of discerning and disseminating factual content. This study evaluates the efficacy of Large Language Models (LLMs) as innovative solutions for mitigating misinformation on platforms like Twitter. LLMs, such as OpenAI's GPT and Meta's LLaMA, offer a pre-trained, adaptable approach that bypasses the extensive training and overfitting issues associated with traditional machine learning models. We assess the performance of LLMs in detecting and classifying COVID-19-related scientific claims, thus facilitating informed decision-making. Our findings indicate that LLMs have significant potential as automated fact-checking tools, though research in this domain is nascent and further exploration is required. We present a comparative analysis of LLMs' performance using a specialized dataset and propose a framework for their application in public health communication. 

**Abstract (ZH)**: 新冠疫情期间社交媒体的普遍影响是一把双刃剑，既增强了沟通，又传播了谬误信息。这一“数字信息疫情”突显了迫切需要能够识别和传播准确信息的自动化工具。本研究评估了大语言模型（LLMs）作为平台（如推特）上遏制谬误信息的创新解决方案的有效性。大语言模型，如OpenAI的GPT和Meta的LLaMA，提供了一种预训练、可适应的方法，绕过了传统机器学习模型中的大量训练和过拟合问题。我们评估了LLMs在检测和分类与新冠相关的科学声明方面的性能，从而促进了明智决策的制定。我们的研究表明，LLMs在自动化事实核查方面具有巨大的潜力，尽管该领域的研究尚处于初级阶段，仍需进一步探索。我们使用专门的数据集对LLMs的性能进行了比较分析，并提出了一种将其应用于公共卫生沟通的框架。 

---
# Chained Tuning Leads to Biased Forgetting 

**Title (ZH)**: 链式调整会导致有偏见的遗忘 

**Authors**: Megan Ung, Alicia Sun, Samuel J. Bell, Bhaktipriya Radharapu, Levent Sagun, Adina Williams  

**Link**: [PDF](https://arxiv.org/pdf/2412.16469)  

**Abstract**: Large language models (LLMs) are often fine-tuned for use on downstream tasks, though this can degrade capabilities learned during previous training. This phenomenon, often referred to as catastrophic forgetting, has important potential implications for the safety of deployed models. In this work, we first show that models trained on downstream tasks forget their safety tuning to a greater extent than models trained in the opposite this http URL, we show that forgetting disproportionately impacts safety information about certain groups. To quantify this phenomenon, we define a new metric we term biased forgetting. We conduct a systematic evaluation of the effects of task ordering on forgetting and apply mitigations that can help the model recover from the forgetting observed. We hope our findings can better inform methods for chaining the finetuning of LLMs in continual learning settings to enable training of safer and less toxic models. 

**Abstract (ZH)**: 大语言模型（LLMs）通常针对下游任务进行微调，但这可能会削弱之前训练中学习到的能力。这一现象通常被称为灾难性遗忘，这对部署模型的安全性具有重要的潜在影响。在本项研究中，我们首先表明，针对下游任务训练的模型比反向训练的模型在更大程度上忘记了其安全性调整。其次，我们发现遗忘在很大程度上影响了某些群体的安全信息。为量化这一现象，我们定义了一个新的度量标准，称之为有偏遗忘。我们系统地评估了任务顺序对遗忘的影响，并应用了可以帮助模型从观察到的遗忘中恢复的缓解措施。我们希望这些发现能够更好地指导在连续学习环境中链式微调LLMs的方法，以实现训练更安全、更无毒模型的目的。 

---
# Transducer-Llama: Integrating LLMs into Streamable Transducer-based Speech Recognition 

**Title (ZH)**: Transducer-Llama：将大规模语言模型集成到可流式处理的转换器声学模型中进行语音识别 

**Authors**: Keqi Deng, Jinxi Guo, Yingyi Ma, Niko Moritz, Philip C. Woodland, Ozlem Kalinli, Mike Seltzer  

**Link**: [PDF](https://arxiv.org/pdf/2412.16464)  

**Abstract**: While large language models (LLMs) have been applied to automatic speech recognition (ASR), the task of making the model streamable remains a challenge. This paper proposes a novel model architecture, Transducer-Llama, that integrates LLMs into a Factorized Transducer (FT) model, naturally enabling streaming capabilities. Furthermore, given that the large vocabulary of LLMs can cause data sparsity issue and increased training costs for spoken language systems, this paper introduces an efficient vocabulary adaptation technique to align LLMs with speech system vocabularies. The results show that directly optimizing the FT model with a strong pre-trained LLM-based predictor using the RNN-T loss yields some but limited improvements over a smaller pre-trained LM predictor. Therefore, this paper proposes a weak-to-strong LM swap strategy, using a weak LM predictor during RNN-T loss training and then replacing it with a strong LLM. After LM replacement, the minimum word error rate (MWER) loss is employed to finetune the integration of the LLM predictor with the Transducer-Llama model. Experiments on the LibriSpeech and large-scale multi-lingual LibriSpeech corpora show that the proposed streaming Transducer-Llama approach gave a 17% relative WER reduction (WERR) over a strong FT baseline and a 32% WERR over an RNN-T baseline. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）已被应用于自动语音识别（ASR），使模型支持流式处理任务仍然存在挑战。本文提出了一种新颖的模型架构，即Transducer-Llama，将LLMs集成到因子转换器（FT）模型中，自然地实现了流式处理能力。此外，鉴于LLMs的大型词汇表会导致语音系统中的数据稀疏问题和更高的训练成本，本文引入了一种高效的语言模型适应技术，将LLMs与语音系统的词汇表对齐。结果表明，直接使用强预训练的LLM基预测器优化FT模型，使用RNN-T损失进行优化，相对于较小的预训练语言模型（LM）预测器仅有有限的改进。因此，本文提出了一个从弱到强的语言模型替换策略，在RNN-T损失训练期间使用弱LM预测器，然后将其替换为强LLM。在LM替换后，使用最小字错误率（MWER）损失对LLM预测器与Transducer-Llama模型的结合进行微调。在LibriSpeech和大规模多语言LibriSpeech语料库上的实验表明，所提出的流式Transducer-Llama方法与强FT基准相比降低了17%的相对字错误率（WERR），与RNN-T基准相比降低了32%的WERR。 

---
# InfoTech Assistant : A Multimodal Conversational Agent for InfoTechnology Web Portal Queries 

**Title (ZH)**: InfoTech助手：一个针对信息技术网页门户查询的多模态对话代理 

**Authors**: Sai Surya Gadiraju, Duoduo Liao, Akhila Kudupudi, Santosh Kasula, Charitha Chalasani  

**Link**: [PDF](https://arxiv.org/pdf/2412.16412)  

**Abstract**: This pilot study presents the development of the InfoTech Assistant, a domain-specific, multimodal chatbot engineered to address queries in bridge evaluation and infrastructure technology. By integrating web data scraping, large language models (LLMs), and Retrieval-Augmented Generation (RAG), the InfoTech Assistant provides accurate and contextually relevant responses. Data, including textual descriptions and images, are sourced from publicly available documents on the InfoTechnology website and organized in JSON format to facilitate efficient querying. The architecture of the system includes an HTML-based interface and a Flask back end connected to the Llama 3.1 model via LLM Studio. Evaluation results show approximately 95 percent accuracy on domain-specific tasks, with high similarity scores confirming the quality of response matching. This RAG-enhanced setup enables the InfoTech Assistant to handle complex, multimodal queries, offering both textual and visual information in its responses. The InfoTech Assistant demonstrates strong potential as a dependable tool for infrastructure professionals, delivering high accuracy and relevance in its domain-specific outputs. 

**Abstract (ZH)**: 本试点研究介绍了一种针对桥粱评估和基础设施技术领域的、具有多模态功能的助手——InfoTech助理。通过集成网页数据爬取、大型语言模型（LLMs）和检索增强生成（RAG）技术，InfoTech助理能够提供准确并具有上下文相关性的回应。数据包括文本描述和图像，来源于InfoTechnology网站上的公开文档，并以JSON格式组织，以便高效查询。该系统的架构包含基于HTML的用户界面和通过LLM Studio连接到Llama 3.1模型的Flask后端。评估结果显示，在特定领域的任务中，其准确率达到约95%，高相似度得分证实了回应匹配的质量。这种RAG增强设置使InfoTech助理能够处理复杂的多模态查询，在回应中提供文本和视觉信息。InfoTech助理作为基础设施专业人士的可靠工具展示了强大的潜力，能够在特定领域的输出中提供高准确性和相关性。 

---
# Modular Conversational Agents for Surveys and Interviews 

**Title (ZH)**: 模块化对话代理在调查和访谈中的应用 

**Authors**: Jiangbo Yu, Jinhua Zhao, Luis Miranda-Moreno, Matthew Korp  

**Link**: [PDF](https://arxiv.org/pdf/2412.17049)  

**Abstract**: Surveys and interviews (structured, semi-structured, or unstructured) are widely used for collecting insights on emerging or hypothetical scenarios. Traditional human-led methods often face challenges related to cost, scalability, and consistency. Recently, various domains have begun to explore the use of conversational agents (chatbots) powered by large language models (LLMs). However, as public investments and policies on infrastructure and services often involve substantial public stakes and environmental risks, there is a need for a rigorous, transparent, privacy-preserving, and cost-efficient development framework tailored for such major decision-making processes. This paper addresses this gap by introducing a modular approach and its resultant parameterized process for designing conversational agents. We detail the system architecture, integrating engineered prompts, specialized knowledge bases, and customizable, goal-oriented conversational logic in the proposed approach. We demonstrate the adaptability, generalizability, and efficacy of our modular approach through three empirical studies: (1) travel preference surveys, highlighting multimodal (voice, text, and image generation) capabilities; (2) public opinion elicitation on a newly constructed, novel infrastructure project, showcasing question customization and multilingual (English and French) capabilities; and (3) transportation expert consultation about future transportation systems, highlighting real-time, clarification request capabilities for open-ended questions, resilience in handling erratic inputs, and efficient transcript post-processing. The results show the effectiveness of this modular approach and how it addresses key ethical, privacy, security, and token consumption concerns, setting the stage for the next-generation surveys and interviews. 

**Abstract (ZH)**: 调查和访谈（结构化的、半结构化的或非结构化的）广泛用于收集关于新兴或假设情境的见解。传统的由人类主导的方法经常会面临成本、可扩展性和一致性方面的挑战。近年来，各个领域开始探索使用大型语言模型（LLMs）驱动的对话代理（聊天机器人）的方法。然而，由于公共投资和政策通常涉及重要公共利益和环境风险，因此需要一种严谨、透明、保护隐私且成本效益高的开发框架，以适应这些重大决策过程。本文通过引入模块化方法及其参数化流程来解决这一问题，详细介绍了该系统的架构，涵盖了精心设计的提示、专业化的知识库以及可定制的目标导向对话逻辑。通过三项实证研究，展示了模块化方法的适应性、可移植性和有效性：（1）旅行偏好的调查，突显了多模态（语音、文本和图像生成）能力；（2）对一个新建成的创新型基础设施项目的公众意见收集，展现了问题定制化和多种语言（英语和法语）能力；以及（3）交通运输专家对未来交通系统咨询，强调了对开放式问题的即时澄清请求能力、处理异常输入的韧性以及高效的对话记录后处理。研究结果表明了模块化方法的有效性及其如何解决关键的伦理、隐私、安全性和令牌消耗问题，为下一代调查和访谈奠定了基础。 

---
# Self-guided Knowledgeable Network of Thoughts: Amplifying Reasoning with Large Language Models 

**Title (ZH)**: 自我引导的知识性思维网络：利用大规模语言模型强化推理 

**Authors**: Chao-Chi Chen, Chin-Yuan Yeh, Hsi-Wen Chen, De-Nian Yang, Ming-Syan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.16533)  

**Abstract**: We introduce Knowledgeable Network of Thoughts (kNoT): a prompt scheme that advances the capabilities of large language models (LLMs) beyond existing paradigms like Chain-of-Thought (CoT), Tree of Thoughts (ToT), and Graph of Thoughts (GoT). The key innovation of kNoT is the LLM Workflow Template (LWT), which allows for an executable plan to be specified by LLMs for LLMs. LWT allows these plans to be arbitrary networks, where single-step LLM operations are nodes, and edges correspond to message passing between these steps. Furthermore, LWT supports selection of individual elements through indexing, facilitating kNoT to produce intricate plans where each LLM operation can be limited to elementary operations, greatly enhancing reliability over extended task sequences. We demonstrate that kNoT significantly outperforms the state of the art on six use cases, while reducing the need for extensive prompt engineering. For instance, kNoT finds 92% accuracy for sorting 32 numbers over 12% and 31% for ToT and GoT, while utilizing up to 84.4% and 87.3% less task-specific prompts, respectively. 

**Abstract (ZH)**: 我们介绍了一种名为Knowledgeable Network of Thoughts (kNoT)的提示方案：这一方案超越了现有的思维方式链（Chain-of-Thought, CoT）、思维树（Tree of Thoughts, ToT）和思维图（Graph of Thoughts, GoT）等范式，提升了大型语言模型（LLMs）的能力。kNoT的关键创新在于LLM工作流模板（LLM Workflow Template, LWT），它允许LLMs为LLMs指定可执行的计划。LWT允许这些计划表现为任意网络，其中单步骤LLM操作作为节点，边对应于这些步骤之间的信息传递。此外，LWT支持通过索引选择个别元素，使得kNoT能够生成复杂计划，其中每一步LLM操作可以限制为基本操作，这大大增强了长时间任务序列的可靠性。我们展示了kNoT在六个应用场景中显著优于现有最先进的方法，同时减少了对大量定制提示的依赖。例如，kNoT在对32个数字进行排序时达到了92%的准确率，而ToT和GoT的准确率分别为12%和31%，同时分别减少了多达84.4%和87.3%的任务特定提示。 

---
# A Machine Learning Approach for Emergency Detection in Medical Scenarios Using Large Language Models 

**Title (ZH)**: 使用大型语言模型的医学场景中紧急情况检测的机器学习方法 

**Authors**: Ferit Akaybicen, Aaron Cummings, Lota Iwuagwu, Xinyue Zhang, Modupe Adewuyi  

**Link**: [PDF](https://arxiv.org/pdf/2412.16341)  

**Abstract**: The rapid identification of medical emergencies through digital communication channels remains a critical challenge in modern healthcare delivery, particularly with the increasing prevalence of telemedicine. This paper presents a novel approach leveraging large language models (LLMs) and prompt engineering techniques for automated emergency detection in medical communications. We developed and evaluated a comprehensive system using multiple LLaMA model variants (1B, 3B, and 7B parameters) to classify medical scenarios as emergency or non-emergency situations. Our methodology incorporated both system prompts and in-prompt training approaches, evaluated across different hardware configurations. The results demonstrate exceptional performance, with the LLaMA 2 (7B) model achieving 99.7% accuracy and the LLaMA 3.2 (3B) model reaching 99.6% accuracy with optimal prompt engineering. Through systematic testing of training examples within the prompts, we identified that including 10 example scenarios in the model prompts yielded optimal classification performance. Processing speeds varied significantly between platforms, ranging from 0.05 to 2.2 seconds per request. The system showed particular strength in minimizing high-risk false negatives in emergency scenarios, which is crucial for patient safety. The code implementation and evaluation framework are publicly available on GitHub, facilitating further research and development in this crucial area of healthcare technology. 

**Abstract (ZH)**: 通过数字通信渠道快速识别医疗紧急情况依然是现代医疗服务中的关键挑战，尤其是在远程医疗日益普及的情况下。本文提出了一种新的方法，利用大规模语言模型（LLMs）和提示工程技术，实现自动化的医疗紧急情况检测。我们开发并评估了一个综合系统，使用了多个LLaMA模型变体（1B、3B和7B参数）来分类医疗场景为紧急或非紧急情况。我们的方法论结合了系统提示和嵌入式提示训练方法，并在不同硬件配置下进行了评估。结果显示，LLaMA 2（7B）模型达到了99.7%的准确率，而LLaMA 3.2（3B）模型通过最佳的提示工程技术达到了99.6%的准确率。通过系统测试提示内的训练示例，我们发现包括10个示例场景在模型提示中能获得最佳分类性能。不同平台的处理速度差异显著，从每请求0.05秒到2.2秒不等。该系统在紧急情况下的高风险假阴性最小化方面表现出特别的优势，这对于患者安全至关重要。该系统的代码实现和评估框架已在GitHub上公开发布，以促进对该领域关键问题的进一步研究和开发。 

---
# Inference Scaling vs Reasoning: An Empirical Analysis of Compute-Optimal LLM Problem-Solving 

**Title (ZH)**: 推理扩展 vs 推理：关于计算最优的大语言模型问题求解的实证分析 

**Authors**: Marwan AbdElhameed, Pavly Halim  

**Link**: [PDF](https://arxiv.org/pdf/2412.16260)  

**Abstract**: Recent advances in large language models (LLMs) have predominantly focused on maximizing accuracy and reasoning capabilities, often overlooking crucial computational efficiency considerations. While this approach has yielded impressive accuracy improvements, it has led to methods that may be impractical for real-world deployment due to computational overhead and latency constraints. This paper investigates the potential synergy between reasoning enhancement and computational efficiency by analyzing the integration of two contrasting approaches: Quiet-STaR (Self-Taught Reasoner) and REBASE (REward BAlanced SEarch). Through comprehensive empirical analysis using the Mistral-7B model on the GSM8K dataset, we demonstrate that while each method excels in its primary objective-Quiet-STaR achieving superior accuracy (32.03%) despite high computational cost (554.66s runtime, 12.73T FLOPs), and REBASE providing exceptional efficiency (8.47s runtime, 2.35T FLOPs) while maintaining baseline-comparable accuracy (10.94%)-their integration reveals fundamental challenges in reconciling reasoning depth with computational efficiency. The combined approach unexpectedly results in degraded performance (9.38% accuracy, 143.66s runtime), highlighting critical insights about the complex interplay between reasoning enhancement and efficiency optimization in LLMs. Our findings illuminate the need for novel architectures and algorithms specifically designed to bridge the gap between these competing objectives, while providing concrete directions for future research in compute-efficient reasoning methods. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的进展主要集中于最大化准确性和推理能力，往往忽略了计算效率方面的关键考虑。虽然这种方法在准确性提升方面取得了令人印象深刻的成果，但这也导致了一些由于计算开销和延迟限制而可能在实际部署中不可行的方法。本文通过分析两种截然不同的方法——Quiet-STaR（Self-Taught Reasoner）和REBASE（REward BAlanced SEarch）——的集成，探讨了推理增强与计算效率之间的潜在协同效应。我们利用Mistral-7B模型和GSM8K数据集进行了全面的实证分析，结果显示，每种方法在主要目标方面表现优异：Quiet-STaR 尽管计算成本很高（运行时间554.66秒，12.73TFLOPs），却实现了优于其他方法的准确率（32.03%）；而REBASE 在保持基线水平相近准确率（10.94%）的前提下，提供了出色的效率（运行时间8.47秒，2.35TFLOPs）。然而，它们的集成揭示了在平衡推理深度与计算效率方面根本性的挑战。结合这两种方法出乎意料地导致了性能下降（准确率9.38%，运行时间143.66秒），突显了推理增强与效率优化之间复杂相互作用的关键见解。我们的研究结果强调了需要新型架构和算法来弥合这些竞争目标之间的差距，并为未来计算高效的推理方法研究提供了具体的方向。 

---
# Efficient VoIP Communications through LLM-based Real-Time Speech Reconstruction and Call Prioritization for Emergency Services 

**Title (ZH)**: 基于LLM的实时语音重建和紧急服务呼叫优先级优化的高效VoIP通信 

**Authors**: Danush Venkateshperumal, Rahman Abdul Rafi, Shakil Ahmed, Ashfaq Khokhar  

**Link**: [PDF](https://arxiv.org/pdf/2412.16176)  

**Abstract**: Emergency communication systems face disruptions due to packet loss, bandwidth constraints, poor signal quality, delays, and jitter in VoIP systems, leading to degraded real-time service quality. Victims in distress often struggle to convey critical information due to panic, speech disorders, and background noise, further complicating dispatchers' ability to assess situations accurately. Staffing shortages in emergency centers exacerbate delays in coordination and assistance. This paper proposes leveraging Large Language Models (LLMs) to address these challenges by reconstructing incomplete speech, filling contextual gaps, and prioritizing calls based on severity. The system integrates real-time transcription with Retrieval-Augmented Generation (RAG) to generate contextual responses, using Twilio and AssemblyAI APIs for seamless implementation. Evaluation shows high precision, favorable BLEU and ROUGE scores, and alignment with real-world needs, demonstrating the model's potential to optimize emergency response workflows and prioritize critical cases effectively. 

**Abstract (ZH)**: 应急通信系统因包丢失、带宽限制、信号质量差、延迟和VoIP系统中的抖动而面临中断，导致实时服务质量下降。遇险人员由于恐慌、言语障碍和背景噪音往往难以传达关键信息，进一步增加了调度员准确评估情况的难度。紧急中心的人员短缺加剧了协调和援助的延迟。本文提出利用大型语言模型（LLMs）来解决这些挑战，通过重建不完整语音、填补上下文空白和基于严重程度优先处理呼叫。该系统将实时转录与检索增强生成（RAG）结合，以生成上下文响应，并使用Twilio和AssemblyAI API实现无缝集成。评价结果显示高精度、有利的BLEU和ROUGE分数，并满足实际需求，证明该模型具有优化应急响应工作流程和有效处理关键案例的潜力。 

---
