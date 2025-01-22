# Optimizing Leaky Private Information Retrieval Codes to Achieve ${O}(\log K)$ Leakage Ratio Exponent 

**Title (ZH)**: 优化泄露型私有信息检索码，以实现${O}(\log K)$泄露率指数 

**Authors**: Wenyuan Zhao, Yu-Shin Huang, Chao Tian, Alex Sprintson  

**Link**: [PDF](https://arxiv.org/pdf/2501.12310)  

**Abstract**: We study the problem of leaky private information retrieval (L-PIR), where the amount of privacy leakage is measured by the pure differential privacy parameter, referred to as the leakage ratio exponent. Unlike the previous L-PIR scheme proposed by Samy et al., which only adjusted the probability allocation to the clean (low-cost) retrieval pattern, we optimize the probabilities assigned to all the retrieval patterns jointly. It is demonstrated that the optimal retrieval pattern probability distribution is quite sophisticated and has a layered structure: the retrieval patterns associated with the random key values of lower Hamming weights should be assigned higher probabilities. This new scheme provides a significant improvement, leading to an ${O}(\log K)$ leakage ratio exponent with fixed download cost $D$ and number of servers $N$, in contrast to the previous art that only achieves a $\Theta(K)$ exponent, where $K$ is the number of messages. 

**Abstract (ZH)**: 我们研究了泄漏型私有信息检索（L-PIR）问题，其中隐私泄漏的程度通过纯差异隐私参数来度量，称为泄漏比率指数。与Samy等人的先前提出的L-PIR方案不同，该方案仅调整了清洁（低成本）检索模式的概率分配，我们对所有检索模式的概率进行了联合优化。研究表明，最优的检索模式概率分布结构非常复杂，具有分层结构：与较低汉明重量的随机密钥值相应的检索模式应分配更高的概率。这种新方案提供了显著改进，固定下载成本为$D$和服务器数量为$N$的情况下，可以达到$O(\log K)$的泄漏比率指数，而以往方法只能达到$\Theta(K)$的指数级别，其中$K$为消息的数量。 

---
# DataPro -- A Standardized Data Understanding and Processing Procedure: A Case Study of an Eco-Driving Project 

**Title (ZH)**: DataPro — 一种标准化的数据理解和处理流程：一项生态驾驶项目案例研究 

**Authors**: Zhipeng Ma, Bo Nørregaard Jørgensen, Zheng Grace Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.12176)  

**Abstract**: A systematic pipeline for data processing and knowledge discovery is essential to extracting knowledge from big data and making recommendations for operational decision-making. The CRISP-DM model is the de-facto standard for developing data-mining projects in practice. However, advancements in data processing technologies require enhancements to this framework. This paper presents the DataPro (a standardized data understanding and processing procedure) model, which extends CRISP-DM and emphasizes the link between data scientists and stakeholders by adding the "technical understanding" and "implementation" phases. Firstly, the "technical understanding" phase aligns business demands with technical requirements, ensuring the technical team's accurate comprehension of business goals. Next, the "implementation" phase focuses on the practical application of developed data science models, ensuring theoretical models are effectively applied in business contexts. Furthermore, clearly defining roles and responsibilities in each phase enhances management and communication among all participants. Afterward, a case study on an eco-driving data science project for fuel efficiency analysis in the Danish public transportation sector illustrates the application of the DataPro model. By following the proposed framework, the project identified key business objectives, translated them into technical requirements, and developed models that provided actionable insights for reducing fuel consumption. Finally, the model is evaluated qualitatively, demonstrating its superiority over other data science procedures. 

**Abstract (ZH)**: 从大数据中提取知识并对运营决策进行推荐的过程需要一套系统化的工作流程和知识发现机制。在实践中，CRISP-DM模型是开发数据挖掘项目的事实标准。然而，数据处理技术的进步要求对该框架进行改进。本文提出了一种标准化的数据理解与处理流程（DataPro），该流程在CRISP-DM的基础上增加了“技术理解”和“实施”阶段，从而加强了数据科学家与利益相关者之间的联系。首先，“技术理解”阶段将业务需求与技术要求相匹配，确保技术团队准确理解业务目标。其次，“实施”阶段重点关注已开发的数据科学模型的实际应用，确保理论模型能在业务环境中有效应用。此外，明确界定每个阶段的角色和责任可以提高所有参与者的管理和沟通效率。随后，以丹麦公共交通运输领域的生态驾驶数据科学项目为例，说明了DataPro模型的应用。该项目遵循提出的框架，确定了关键业务目标，将其转化为技术需求，并开发了提供了降低燃料消耗行动建议的模型。最后，通过定性的评价，证明了DataPro模型优于其他数据科学流程。 

---
# Less is More: Information Bottleneck Denoised Multimedia Recommendation 

**Title (ZH)**: 更少也是更多：信息瓶颈去噪多模态推荐算法 

**Authors**: Yonghui Yang, Le Wu, Zhuangzhuang He, Zhengwei Wu, Richang Hong, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.12175)  

**Abstract**: Empowered by semantic-rich content information, multimedia recommendation has emerged as a potent personalized technique. Current endeavors center around harnessing multimedia content to refine item representation or uncovering latent item-item structures based on modality similarity. Despite the effectiveness, we posit that these methods are usually suboptimal due to the introduction of irrelevant multimedia features into recommendation tasks. This stems from the fact that generic multimedia feature extractors, while well-designed for domain-specific tasks, can inadvertently introduce task-irrelevant features, leading to potential misguidance of recommenders. In this work, we propose a denoised multimedia recommendation paradigm via the Information Bottleneck principle (IB). Specifically, we propose a novel Information Bottleneck denoised Multimedia Recommendation (IBMRec) model to tackle the irrelevant feature issue. IBMRec removes task-irrelevant features from both feature and item-item structure perspectives, which are implemented by two-level IB learning modules: feature-level (FIB) and graph-level (GIB). In particular, FIB focuses on learning the minimal yet sufficient multimedia features. This is achieved by maximizing the mutual information between multimedia representation and recommendation tasks, while concurrently minimizing it between multimedia representation and pre-trained multimedia features. Furthermore, GIB is designed to learn the robust item-item graph structure, it refines the item-item graph based on preference affinity, then minimizes the mutual information between the original graph and the refined one. Extensive experiments across three benchmarks validate the effectiveness of our proposed model, showcasing high performance, and applicability to various multimedia recommenders. 

**Abstract (ZH)**: 得益于丰富的语义内容信息，多媒体推荐已经成为一种强大的个性化技术。当前的研究主要集中在利用多媒体内容来优化项目表示或基于模态相似性揭露潜在的项-项结构。尽管这些方法在实践中证明了其有效性，但我们认为这些方法通常可能是不理想的，因为它们在推荐任务中引入了无关的多媒体特征。这源于通用的多媒体特征提取器虽然被精心设计用于特定领域任务，但可能会不自觉地引入与任务无关的特征，从而误导推荐系统。在本研究中，我们通过信息瓶颈原理（Information Bottleneck, IB）提出了一种去噪的多媒体推荐范式。具体来说，我们提出了一种新颖的基于信息瓶颈原理的去噪多媒体推荐模型（Information Bottleneck Denoised Multimedia Recommendation, IBMRec），以解决无关特征问题。IBMRec从特征层面和项-项结构层面去除无关特征，分别通过两个层次的信息瓶颈学习模块实现：特征层面的信息瓶颈学习模块（Feature-level Information Bottleneck, FIB）和图层面的信息瓶颈学习模块（Graph-level Information Bottleneck, GIB）。具体而言，FIB专注于学习最小但足够的多媒体特征。这一目标通过最大化多媒体表示与推荐任务之间的互信息，同时最小化多媒体表示与预训练的多媒体特征之间的互信息，从而实现。此外，GIB设计用于学习稳健的项-项图结构，它基于偏好亲和性调整项-项图，然后最小化原始图与优化后的图之间的互信息。在三个基准测试中的广泛实验验证了我们提出模型的有效性，展示了高性能，并且适用于各种多媒体推荐系统。 

---
# A Contrastive Framework with User, Item and Review Alignment for Recommendation 

**Title (ZH)**: 一种基于用户、物品和评价对齐的对比框架推荐方法 

**Authors**: Hoang V. Dong, Yuan Fang, Hady W. Lauw  

**Link**: [PDF](https://arxiv.org/pdf/2501.11963)  

**Abstract**: Learning effective latent representations for users and items is the cornerstone of recommender systems. Traditional approaches rely on user-item interaction data to map users and items into a shared latent space, but the sparsity of interactions often poses challenges. While leveraging user reviews could mitigate this sparsity, existing review-aware recommendation models often exhibit two key limitations. First, they typically rely on reviews as additional features, but reviews are not universal, with many users and items lacking them. Second, such approaches do not integrate reviews into the user-item space, leading to potential divergence or inconsistency among user, item, and review representations. To overcome these limitations, our work introduces a Review-centric Contrastive Alignment Framework for Recommendation (ReCAFR), which incorporates reviews into the core learning process, ensuring alignment among user, item, and review representations within a unified space. Specifically, we leverage two self-supervised contrastive strategies that not only exploit review-based augmentation to alleviate sparsity, but also align the tripartite representations to enhance robustness. Empirical studies on public benchmark datasets demonstrate the effectiveness and robustness of ReCAFR. 

**Abstract (ZH)**: 学习有效的潜在表示是推荐系统的核心。传统方法依赖用户-项交互数据将用户和项映射到共享的潜在空间，但交互数据的稀疏性常常带来挑战。尽管利用用户评论可以缓解这种稀疏性，现有的评论知情推荐模型仍存在两个关键局限。首先，这些模型通常将评论作为附加特征使用，但评论并不是普遍存在的，许多用户和项没有评论。其次，此类方法没有将评论整合到用户-项空间中，这可能导致用户、项和评论表示之间存在潜在的分歧或不一致。为克服这些局限，我们提出了一个以评论为中心的对比对齐推荐框架（ReCAFR），该框架将评论整合到核心学习过程中，确保在统一起点内用户、项和评论表示之间的一致性。具体而言，我们采用了两种自监督的对比策略，不仅利用基于评论的数据增强以缓解稀疏性，还对三重表示进行对齐以增强鲁棒性。在公共基准数据集上的实证研究证实了ReCAFR的有效性和鲁棒性。 

---
# Generating with Fairness: A Modality-Diffused Counterfactual Framework for Incomplete Multimodal Recommendations 

**Title (ZH)**: 生成公正的推荐：一种跨模态生成对抗框架用于不完整多模态推荐 

**Authors**: Jin Li, Shoujin Wang, Qi Zhang, Shui Yu, Fang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.11916)  

**Abstract**: Incomplete scenario is a prevalent, practical, yet challenging setting in Multimodal Recommendations (MMRec), where some item modalities are missing due to various factors. Recently, a few efforts have sought to improve the recommendation accuracy by exploring generic structures from incomplete data. However, two significant gaps persist: 1) the difficulty in accurately generating missing data due to the limited ability to capture modality distributions; and 2) the critical but overlooked visibility bias, where items with missing modalities are more likely to be disregarded due to the prioritization of items' multimodal data over user preference alignment. This bias raises serious concerns about the fair treatment of items. To bridge these two gaps, we propose a novel Modality-Diffused Counterfactual (MoDiCF) framework for incomplete multimodal recommendations. MoDiCF features two key modules: a novel modality-diffused data completion module and a new counterfactual multimodal recommendation module. The former, equipped with a particularly designed multimodal generative framework, accurately generates and iteratively refines missing data from learned modality-specific distribution spaces. The latter, grounded in the causal perspective, effectively mitigates the negative causal effects of visibility bias and thus assures fairness in recommendations. Both modules work collaboratively to address the two aforementioned significant gaps for generating more accurate and fair results. Extensive experiments on three real-world datasets demonstrate the superior performance of MoDiCF in terms of both recommendation accuracy and fairness 

**Abstract (ZH)**: 不完整场景是多模态推荐（MMRec）中一个普遍存在且具有挑战性的设置，在这种场景中，由于多种因素，某些项模态数据缺失。最近，一些努力试图通过探索不完整数据中的通用结构来提高推荐精度。然而，仍然存在两个重要的间隙：1）由于难以准确捕捉模态分布而导致的生成缺失数据的难度；2）忽视的但至关重要的可见性偏见，其中由于优先考虑项的多模态数据而非用户偏好匹配，具有缺失模态数据的项更容易被忽略。这种偏见引发了对公平对待项的严重关切。为了弥合这两个差距，我们提出了一种新颖的模态扩散反事实（MoDiCF）框架，用于不完整多模态推荐。MoDiCF 包含两个关键模块：一个新颖的模态扩散数据填充模块和一个新的反事实多模态推荐模块。前者配备了特别设计的多模态生成框架，能够从学习到的模态特定分布空间中准确生成并迭代细化缺失数据。后者以因果视角为基础，有效减轻了可见性偏见的负面影响，从而确保推荐的公平性。两个模块协作以解决上述两个关键差距，以生成更准确和公平的结果。在三个真实世界的数据集上进行的广泛实验表明，MoDiCF 在推荐准确性和公平性方面均表现出更优的性能。 

---
# Integrate Temporal Graph Learning into LLM-based Temporal Knowledge Graph Model 

**Title (ZH)**: 将时间图学习集成到基于大规模语言模型的时间知识图模型中 

**Authors**: He Chang, Jie Wu, Zhulin Tao, Yunshan Ma, Xianglin Huang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2501.11911)  

**Abstract**: Temporal Knowledge Graph Forecasting (TKGF) aims to predict future events based on the observed events in history. Recently, Large Language Models (LLMs) have exhibited remarkable capabilities, generating significant research interest in their application for reasoning over temporal knowledge graphs (TKGs). Existing LLM-based methods have integrated retrieved historical facts or static graph representations into LLMs. Despite the notable performance of LLM-based methods, they are limited by the insufficient modeling of temporal patterns and ineffective cross-modal alignment between graph and language, hindering the ability of LLMs to fully grasp the temporal and structural information in TKGs. To tackle these issues, we propose a novel framework TGL-LLM to integrate temporal graph learning into LLM-based temporal knowledge graph model. Specifically, we introduce temporal graph learning to capture the temporal and relational patterns and obtain the historical graph embedding. Furthermore, we design a hybrid graph tokenization to sufficiently model the temporal patterns within LLMs. To achieve better alignment between graph and language, we employ a two-stage training paradigm to finetune LLMs on high-quality and diverse data, thereby resulting in better performance. Extensive experiments on three real-world datasets show that our approach outperforms a range of state-of-the-art (SOTA) methods. 

**Abstract (ZH)**: 时间知识图谱预测（TKGF）旨在基于历史观察事件来预测未来事件。近年来，大型语言模型（LLMs）表现出显著的能力，并引起了将其应用于时间知识图谱（TKGs）推理的研究兴趣。现有的基于LLM的方法将检索到的历史事实或静态图表示整合到LLM中。尽管基于LLM的方法取得了显著的性能，但它们受限于对时间模式建模不足和图与语言之间的无效跨模态对齐，这阻碍了LLM完全理解和把握TKGs中的时间和结构信息的能力。为了解决这些问题，我们提出了一种新的框架TGL-LLM，将时间图学习整合到基于LLM的时间知识图谱模型中。具体而言，我们引入了时间图学习来捕捉时间和关系模式，并获取历史图嵌入。此外，我们设计了一种混合图标记化方法，以充分在LLM中建模时间模式。为了实现更好的图与语言之间的对齐，我们采用两阶段训练范式在高质量和多样性的数据上微调LLM，从而提高性能。在三个真实世界的数据集上的广泛实验表明，我们的方法在多种先进方法中表现更优。 

---
# Coarse-to-Fine Lightweight Meta-Embedding for ID-Based Recommendation 

**Title (ZH)**: 从粗到细的轻量级元嵌入方法用于基于身份的推荐 

**Authors**: Yang Wang, Haipeng Liu, Zeqian Yi, Biao Qian, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11870)  

**Abstract**: The state-of-the-art recommendation systems have shifted the attention to efficient recommendation, e.g., on-device recommendation, under memory constraints. To this end, the existing methods either focused on the lightweight embeddings for both users and items, or involved on-device systems enjoying the compact embeddings to enhance reusability and reduces space complexity. However, they focus solely on the coarse granularity of embedding, while overlook the fine-grained semantic nuances, to adversarially downgrade the efficacy of meta-embeddings in capturing the intricate relationship over both user and item, consequently resulting into the suboptimal recommendations. In this paper, we aim to study how the meta-embedding can efficiently learn varied grained semantics, together with how the fine-grained meta-embedding can strengthen the representation of coarse-grained meta-embedding. To answer these questions, we develop a novel graph neural networks (GNNs) based recommender where each user and item serves as the node, linked directly to coarse-grained virtual nodes and indirectly to fine-grained virtual nodes, ensuring different grained semantic learning, while disclosing: 1) In contrast to coarse-grained semantics, fine-grained semantics are well captured through sparse meta-embeddings, which adaptively 2) balance the embedding uniqueness and memory constraint. Additionally, the initialization method come up upon SparsePCA, along with a soft thresholding activation function to render the sparseness of the meta-embeddings. We propose a weight bridging update strategy that focuses on matching each coarse-grained meta-embedding with several fine-grained meta-embeddings based on the users/items' semantics. Extensive experiments substantiate our method's superiority over existing baselines. Our code is available at this https URL. 

**Abstract (ZH)**: 当今最先进的推荐系统已经将注意力转向高效的推荐，例如在设备端进行推荐，同时在内存限制下实现高效推荐。为此，现有方法要么专注于用户和项目轻量级嵌入，要么利用紧凑的嵌入在设备端系统中增强可重用性和降低空间复杂度。然而，这些方法仅关注嵌入的一般粒度，而忽略了细微的语义差异，从而导致元嵌入在捕捉用户和项目之间复杂关系时效果不佳，最终导致推荐效果不佳。本文旨在研究元嵌如何高效学习不同粒度的语义，以及精细粒度的元嵌如何增强粗粒度元嵌的表现。为了解答这些问题，我们提出了一种基于图神经网络（GNNs）的推荐模型，其中每个用户和项目作为节点，直接与粗粒度的虚拟节点相连，间接与精细粒度的虚拟节点相连，确保不同粒度的语义学习。具体来说：1）与粗粒度语义相比，细致入微的语义可以通过稀疏元嵌入有效地捕捉，这些嵌入能够适当地平衡嵌入的独特性和内存限制。2）此外，初始化方法借鉴SparsePCA，并结合软阈值激活函数以使元嵌入具有稀疏性。我们提出了一种权重桥梁更新策略，该策略基于用户/项目的语义，将每个粗粒度的元嵌入与多个精细粒度的元嵌入进行匹配。广泛的实验表明，我们的方法在现有基线方法中表现更优。我们的代码托管在以下链接：this https URL。 

---
# Poison-RAG: Adversarial Data Poisoning Attacks on Retrieval-Augmented Generation in Recommender Systems 

**Title (ZH)**: Poison-RAG：面向推荐系统检索增强生成的对抗性数据污染攻击 

**Authors**: Fatemeh Nazary, Yashar Deldjoo, Tommaso di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2501.11759)  

**Abstract**: This study presents Poison-RAG, a framework for adversarial data poisoning attacks targeting retrieval-augmented generation (RAG)-based recommender systems. Poison-RAG manipulates item metadata, such as tags and descriptions, to influence recommendation outcomes. Using item metadata generated through a large language model (LLM) and embeddings derived via the OpenAI API, we explore the impact of adversarial poisoning attacks on provider-side, where attacks are designed to promote long-tail items and demote popular ones. Two attack strategies are proposed: local modifications, which personalize tags for each item using BERT embeddings, and global modifications, applying uniform tags across the dataset. Experiments conducted on the MovieLens dataset in a black-box setting reveal that local strategies improve manipulation effectiveness by up to 50\%, while global strategies risk boosting already popular items. Results indicate that popular items are more susceptible to attacks, whereas long-tail items are harder to manipulate. Approximately 70\% of items lack tags, presenting a cold-start challenge; data augmentation and synthesis are proposed as potential defense mechanisms to enhance RAG-based systems' resilience. The findings emphasize the need for robust metadata management to safeguard recommendation frameworks. Code and data are available at this https URL. 

**Abstract (ZH)**: 本文提出了一种名为Poison-RAG的框架，该框架针对基于检索增强生成（RAG）的推荐系统进行了对抗性数据投毒攻击研究。Poison-RAG通过操控项目元数据（如标签和描述）来影响推荐结果。利用大型语言模型（LLM）生成的项目元数据和通过OpenAI API获取的嵌入信息，我们探索了对抗性投毒攻击在提供者侧的影响，此类攻击旨在促进长尾项目，并降低热门项目的推荐。两种攻击策略被提出：局部修改，使用BERT嵌入个性化每个项目的标签；全局修改，对整个数据集应用统一的标签。在黑盒设置下，使用MovieLens数据集进行的实验表明，局部策略可以提高操纵效果高达50%，而全局策略则存在提升已热门项目的风险。结果表明，热门项目更易受到攻击，而长尾项目更难被操纵。大约70%的项目缺乏标签，这构成了冷启动挑战；数据增强和合成被提议作为潜在的防御机制，以增强基于RAG的系统的抗攻击能力。研究结果强调了稳健的元数据管理对于保护推荐框架的重要性。相关代码和数据可在此网页访问：[提供链接]。 

---
# Exploring Preference-Guided Diffusion Model for Cross-Domain Recommendation 

**Title (ZH)**: 探索基于偏好引导的扩散模型在跨域推荐中的应用 

**Authors**: Xiaodong Li, Hengzhu Tang, Jiawei Sheng, Xinghua Zhang, Li Gao, Suqi Cheng, Dawei Yin, Tingwen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.11671)  

**Abstract**: Cross-domain recommendation (CDR) has been proven as a promising way to alleviate the cold-start issue, in which the most critical problem is how to draw an informative user representation in the target domain via the transfer of user preference existing in the source domain. Prior efforts mostly follow the embedding-and-mapping paradigm, which first integrate the preference into user representation in the source domain, and then perform a mapping function on this representation to the target domain. However, they focus on mapping features across domains, neglecting to explicitly model the preference integration process, which may lead to learning coarse user representation. Diffusion models (DMs), which contribute to more accurate user/item representations due to their explicit information injection capability, have achieved promising performance in recommendation systems. Nevertheless, these DMs-based methods cannot directly account for valuable user preference in other domains, leading to challenges in adapting to the transfer of preference for cold-start users. Consequently, the feasibility of DMs for CDR remains underexplored. To this end, we explore to utilize the explicit information injection capability of DMs for user preference integration and propose a Preference-Guided Diffusion Model for CDR to cold-start users, termed as DMCDR. Specifically, we leverage a preference encoder to establish the preference guidance signal with the user's interaction history in the source domain. Then, we explicitly inject the preference guidance signal into the user representation step by step to guide the reverse process, and ultimately generate the personalized user representation in the target domain, thus achieving the transfer of user preference across domains. Furthermore, we comprehensively explore the impact of six DMs-based variants on CDR. 

**Abstract (ZH)**: 跨领域推荐（CDR）已被证明是一种缓解冷启动问题的有效途径，其中最关键的问题是如何通过源域中存在的用户偏好来构建在目标域中的具有信息量的用户表示。以前的努力主要遵循嵌入和映射的范式，首先在源域中将偏好整合到用户表示中，然后在该表示上执行映射函数到目标域。然而，它们主要关注特征在域间的映射，忽视了明确建模偏好整合过程的重要性，这可能导致学习到粗糙的用户表示。扩散模型（DMs），由于其显式信息注入的能力，能够生成更准确的用户/项目表示，在推荐系统中取得了优异的性能。然而，基于DMs的方法不能直接处理其他域中的有价值用户偏好，这为偏好转移适应冷启动用户带来了挑战。因此，DMs在CDR中的可行性尚未得到充分探索。为了应对这一挑战，我们探索了利用DMs的显式信息注入能力来整合用户偏好，并提出了一种基于偏好引导的扩散模型（Preference-Guided Diffusion Model for CDR），简称DMCDR。具体而言，我们利用偏好编码器建立偏好引导信号，并结合用户在源域中的交互历史。然后，我们逐步显式地将偏好引导信号注入用户的表示中，以指导逆过程，最终生成个性化的用户表示，从而实现在目标域中偏好在域间的迁移。此外，我们综合探讨了六种基于DMs的变体在CDR中的影响。 

---
# Investigating the Scalability of Approximate Sparse Retrieval Algorithms to Massive Datasets 

**Title (ZH)**: 探索近似稀疏检索算法在大规模数据集上的可扩展性 

**Authors**: Sebastian Bruch, Franco Maria Nardini, Cosimo Rulli, Rossano Venturini, Leonardo Venuta  

**Link**: [PDF](https://arxiv.org/pdf/2501.11628)  

**Abstract**: Learned sparse text embeddings have gained popularity due to their effectiveness in top-k retrieval and inherent interpretability. Their distributional idiosyncrasies, however, have long hindered their use in real-world retrieval systems. That changed with the recent development of approximate algorithms that leverage the distributional properties of sparse embeddings to speed up retrieval. Nonetheless, in much of the existing literature, evaluation has been limited to datasets with only a few million documents such as MSMARCO. It remains unclear how these systems behave on much larger datasets and what challenges lurk in larger scales. To bridge that gap, we investigate the behavior of state-of-the-art retrieval algorithms on massive datasets. We compare and contrast the recently-proposed Seismic and graph-based solutions adapted from dense retrieval. We extensively evaluate Splade embeddings of 138M passages from MsMarco-v2 and report indexing time and other efficiency and effectiveness metrics. 

**Abstract (ZH)**: 由于在 top-k 查找和固有的可解释性方面表现出色，学习到的稀疏文本嵌入已经变得非常流行。然而，它们的分布特性长期以来阻碍了它们在实际检索系统中的应用。这一情况随着近来利用稀疏嵌入分布特性的近似算法的发展得以改变，这些算法能够加速检索过程。尽管如此，在现有文献中，大多数评价都是基于只有数百万份文档的数据集（如 MSMARCO）进行的。尚不清楚这些系统在更大规模的数据集上表现如何，以及在更大规模下会面临哪些挑战。为了解决这个问题，我们研究了最先进的检索算法在大规模数据集上的行为。我们比较了近期提出的 Seismic 和基于图的方法与密集检索方法的异同，并对来自 MsMarco-v2 的 1380 万段落的 Splade 嵌入进行了广泛评估，报告了索引时间以及其他效率和效果指标。 

---
# KEIR @ ECIR 2025: The Second Workshop on Knowledge-Enhanced Information Retrieval 

**Title (ZH)**: KEIR @ ECIR 2025：第二届知识增强信息检索研讨会 

**Authors**: Zihan Wang, Jinyuan Fang, Giacomo Frisoni, Zhuyun Dai, Zaiqiao Meng, Gianluca Moro, Emine Yilmaz  

**Link**: [PDF](https://arxiv.org/pdf/2501.11499)  

**Abstract**: Pretrained language models (PLMs) like BERT and GPT-4 have become the foundation for modern information retrieval (IR) systems. However, existing PLM-based IR models primarily rely on the knowledge learned during training for prediction, limiting their ability to access and incorporate external, up-to-date, or domain-specific information. Therefore, current information retrieval systems struggle with semantic nuances, context relevance, and domain-specific issues. To address these challenges, we propose the second Knowledge-Enhanced Information Retrieval workshop (KEIR @ ECIR 2025) as a platform to discuss innovative approaches that integrate external knowledge, aiming to enhance the effectiveness of information retrieval in a rapidly evolving technological landscape. The goal of this workshop is to bring together researchers from academia and industry to discuss various aspects of knowledge-enhanced information retrieval. 

**Abstract (ZH)**: 预训练语言模型（PLMs）如BERT和GPT-4已成为现代信息检索（IR）系统的基石。然而，现有的基于PLM的信息检索模型主要依赖于训练期间学到的知识进行预测，这限制了它们访问和整合外部、最新或领域特定信息的能力。因此，当前的信息检索系统在处理语义细微差别、上下文相关性和领域特定问题方面存在困难。为应对这些挑战，我们提议举办第二届知识增强型信息检索研讨会（KEIR @ ECIR 2025），作为讨论将外部知识整合到信息检索中的创新方法的平台，旨在增强在快速发展的技术环境中信息检索的有效性。该研讨会的目标是将学术界和产业界的研究人员聚集在一起，讨论知识增强型信息检索的各个方面。 

---
# Ontology Matching with Large Language Models and Prioritized Depth-First Search 

**Title (ZH)**: 使用大型语言模型和优先级深度优先搜索的知识库对齐方法 

**Authors**: Maria Taboada, Diego Martinez, Mohammed Arideh, Rosa Mosquera  

**Link**: [PDF](https://arxiv.org/pdf/2501.11441)  

**Abstract**: Ontology matching (OM) plays a key role in enabling data interoperability and knowledge sharing, but it remains challenging due to the need for large training datasets and limited vocabulary processing in machine learning approaches. Recently, methods based on Large Language Model (LLMs) have shown great promise in OM, particularly through the use of a retrieve-then-prompt pipeline. In this approach, relevant target entities are first retrieved and then used to prompt the LLM to predict the final matches. Despite their potential, these systems still present limited performance and high computational overhead. To address these issues, we introduce MILA, a novel approach that embeds a retrieve-identify-prompt pipeline within a prioritized depth-first search (PDFS) strategy. This approach efficiently identifies a large number of semantic correspondences with high accuracy, limiting LLM requests to only the most borderline cases. We evaluated MILA using the biomedical challenge proposed in the 2023 and 2024 editions of the Ontology Alignment Evaluation Initiative. Our method achieved the highest F-Measure in four of the five unsupervised tasks, outperforming state-of-the-art OM systems by up to 17%. It also performed better than or comparable to the leading supervised OM systems. MILA further exhibited task-agnostic performance, remaining stable across all tasks and settings, while significantly reducing LLM requests. These findings highlight that high-performance LLM-based OM can be achieved through a combination of programmed (PDFS), learned (embedding vectors), and prompting-based heuristics, without the need of domain-specific heuristics or fine-tuning. 

**Abstract (ZH)**: 本研究内容或标题可翻译成中文如下：

本研究内容摘要：
本研究内容强调了本体匹配（OM）在促进数据互操作性和知识共享方面的重要作用，但由于机器学习方法在构建大型训练数据集和词汇处理方面的限制，使其仍具有挑战性。近年来，基于大型语言模型（LLMs）的方法在本体匹配方面显示出巨大潜力，尤其是在“检索-提示”流水线的应用上。这种方法首先检索相关的目标实体，然后使用这些实体来提示LLMs进行最终匹配预测。尽管这些系统具有潜力，但它们仍然存在性能有限和计算成本高的问题。为了解决这些问题，我们引入了一种名为MILA的新颖方法，该方法将“检索-识别-提示”流水线嵌入优先深度优先搜索（PDFS）策略中。这种方法能够高效识别大量高精度的语义对应关系，仅在边缘情况时请求LLMs。我们使用2023年和2024年生物医学挑战赛中提出的本体对齐评估倡议来评估MILA。我们的方法在四个未监督任务中的F-度量值最高，比最先进的本体匹配系统高出最多17%，并在多个监督本体匹配系统中表现出优越性或可比性。MILA 进一步展示了任务无关的表现，在所有任务和设置中保持稳定，同时显著减少了对LLMs的请求。这些发现表明，高精度的基于LLMs的本体匹配可以通过结合编程（PDFS）、学习（嵌入向量）和提示启发式方法实现，而无需特定领域的启发式方法或微调。 

---
# Revisiting Language Models in Neural News Recommender Systems 

**Title (ZH)**: 重新审视神经新闻推荐系统中的语言模型 

**Authors**: Yuyue Zhao, Jin Huang, David Vos, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2501.11391)  

**Abstract**: Neural news recommender systems (RSs) have integrated language models (LMs) to encode news articles with rich textual information into representations, thereby improving the recommendation process. Most studies suggest that (i) news RSs achieve better performance with larger pre-trained language models (PLMs) than shallow language models (SLMs), and (ii) that large language models (LLMs) outperform PLMs. However, other studies indicate that PLMs sometimes lead to worse performance than SLMs. Thus, it remains unclear whether using larger LMs consistently improves the performance of news RSs. In this paper, we revisit, unify, and extend these comparisons of the effectiveness of LMs in news RSs using the real-world MIND dataset. We find that (i) larger LMs do not necessarily translate to better performance in news RSs, and (ii) they require stricter fine-tuning hyperparameter selection and greater computational resources to achieve optimal recommendation performance than smaller LMs. On the positive side, our experiments show that larger LMs lead to better recommendation performance for cold-start users: they alleviate dependency on extensive user interaction history and make recommendations more reliant on the news content. 

**Abstract (ZH)**: 神经新闻推荐系统（RS）已经将语言模型（LMs）集成进来，用于将包含丰富文本信息的新闻文章编码为表示形式，从而改进了推荐过程。大多数研究指出，(i) 与浅层语言模型（SLMs）相比，使用更大规模的预训练语言模型（PLMs）能获得更好的性能，以及(ii) 大规模语言模型（LLMs）优于PLMs。然而，其他研究显示PLMs有时会导致性能不如SLMs。因此，使用更大规模的LMs是否能始终改善新闻RSs的表现还不得而知。在本文中，我们使用真实世界的MIND数据集重新审视、统一并扩展了LMs在新闻RSs有效性方面的比较。我们发现：(i) 更大规模的LMs不一定能在新闻RSs中表现得更好；(ii) 与较小的LMs相比，它们需要更严格的微调超参数选择，并且需要更多的计算资源才能实现最佳推荐性能。另一方面，我们的实验表明，较大规模的LMs在冷启动用户上的推荐效果更好：它们减少了对用户交互历史的依赖，使得推荐更加依赖于新闻内容。 

---
# Disentangled Modeling of Preferences and Social Influence for Group Recommendation 

**Title (ZH)**: 群体推荐中偏好与社会影响力的解耦建模 

**Authors**: Guangze Ye, Wen Wu, Guoqing Wang, Xi Chen, Hong Zheng, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2501.11342)  

**Abstract**: The group recommendation (GR) aims to suggest items for a group of users in social networks. Existing work typically considers individual preferences as the sole factor in aggregating group preferences. Actually, social influence is also an important factor in modeling users' contributions to the final group decision. However, existing methods either neglect the social influence of individual members or bundle preferences and social influence together as a unified representation. As a result, these models emphasize the preferences of the majority within the group rather than the actual interaction items, which we refer to as the preference bias issue in GR. Moreover, the self-supervised learning (SSL) strategies they designed to address the issue of group data sparsity fail to account for users' contextual social weights when regulating group representations, leading to suboptimal results. To tackle these issues, we propose a novel model based on Disentangled Modeling of Preferences and Social Influence for Group Recommendation (DisRec). Concretely, we first design a user-level disentangling network to disentangle the preferences and social influence of group members with separate embedding propagation schemes based on (hyper)graph convolution networks. We then introduce a socialbased contrastive learning strategy, selectively excluding user nodes based on their social importance to enhance group representations and alleviate the group-level data sparsity issue. The experimental results demonstrate that our model significantly outperforms state-of-the-art methods on two realworld datasets. 

**Abstract (ZH)**: 群体推荐（Group Recommendation, GR）旨在为社交网络中的用户群体提供项目建议。现有工作的典型做法是将个体偏好视为聚合群体偏好时的唯一因素。实际上，在建模用户对最终群体决策的贡献时，社交影响力也是一个重要因素。然而，现有方法要么忽略了个体成员的社交影响力，要么将偏好和社交影响力打包成一个统一的表示形式。因此，这些模型更多地强调群体中多数人的偏好，而不是实际的交互项，我们称之为群体推荐中的偏好偏差问题。此外，他们为了解决群体数据稀疏性问题而设计的自监督学习（Self-Supervised Learning, SSL）策略在调节群体表示时未能考虑用户的上下文社交权重，导致结果不佳。为解决这些问题，我们提出了一种基于偏好和社交影响力解耦模型的群体推荐新型模型（DisRec）。具体而言，我们首先设计了一个用户级别的解耦网络，通过基于（超）图卷积网络的独立嵌入传播方案，以解耦群体成员的偏好和社交影响力。然后，我们引入了一种基于社交的对比学习策略，根据不同用户在社交中的重要程度，有选择地排除用户节点，以增强群体表示并缓解群体层面的数据稀疏性问题。实验结果表明，我们的模型在两个现实世界数据集上显著优于现有最先进的方法。 

---
# PlotEdit: Natural Language-Driven Accessible Chart Editing in PDFs via Multimodal LLM Agents 

**Title (ZH)**: PlotEdit：通过多模态大语言模型代理实现的基于自然语言的PDF图表编辑 

**Authors**: Kanika Goswami, Puneet Mathur, Ryan Rossi, Franck Dernoncourt  

**Link**: [PDF](https://arxiv.org/pdf/2501.11233)  

**Abstract**: Chart visualizations, while essential for data interpretation and communication, are predominantly accessible only as images in PDFs, lacking source data tables and stylistic information. To enable effective editing of charts in PDFs or digital scans, we present PlotEdit, a novel multi-agent framework for natural language-driven end-to-end chart image editing via self-reflective LLM agents. PlotEdit orchestrates five LLM agents: (1) Chart2Table for data table extraction, (2) Chart2Vision for style attribute identification, (3) Chart2Code for retrieving rendering code, (4) Instruction Decomposition Agent for parsing user requests into executable steps, and (5) Multimodal Editing Agent for implementing nuanced chart component modifications - all coordinated through multimodal feedback to maintain visual fidelity. PlotEdit outperforms existing baselines on the ChartCraft dataset across style, layout, format, and data-centric edits, enhancing accessibility for visually challenged users and improving novice productivity. 

**Abstract (ZH)**: 虽然图表可视化对于数据解释和传播至关重要，但它们主要以PDF中的图像形式存在，缺乏源数据表和风格信息，这限制了其编辑能力。为了实现对PDF或数字扫描中的图表的有效编辑，我们提出了一种名为PlotEdit的新型多智能体框架，该框架通过自我反思的LLM代理实现了基于自然语言的从头到尾的图表图像编辑。PlotEdit集成了五个LLM代理：（1）Chart2Table用于提取数据表，（2）Chart2Vision用于识别样式属性，（3）Chart2Code用于检索渲染代码，（4）指令分解代理用于将用户请求解析为可执行步骤，以及（5）多模态编辑代理用于实施精确的图表组件修改——所有这些都通过多模态反馈协调，以保持视觉保真度。在ChartCraft数据集上，PlotEdit在风格、布局、格式和数据中心编辑方面都优于现有基线，提升了视觉受损用户的访问性和新手的生产效率。 

---
# Generative Retrieval for Book search 

**Title (ZH)**: 书籍检索的生成性检索 

**Authors**: Yubao Tang, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Shihao Liu, Shuaiqing Wang, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.11034)  

**Abstract**: In book search, relevant book information should be returned in response to a query. Books contain complex, multi-faceted information such as metadata, outlines, and main text, where the outline provides hierarchical information between chapters and sections. Generative retrieval (GR) is a new retrieval paradigm that consolidates corpus information into a single model to generate identifiers of documents that are relevant to a given query. How can GR be applied to book search? Directly applying GR to book search is a challenge due to the unique characteristics of book search: The model needs to retain the complex, multi-faceted information of the book, which increases the demand for labeled data. Splitting book information and treating it as a collection of separate segments for learning might result in a loss of hierarchical information. We propose an effective Generative retrieval framework for Book Search (GBS) that features two main components: data augmentation and outline-oriented book encoding. For data augmentation, GBS constructs multiple query-book pairs for training; it constructs multiple book identifiers based on the outline, various forms of book contents, and simulates real book retrieval scenarios with varied pseudo-queries. This includes coverage-promoting book identifier augmentation, allowing the model to learn to index effectively, and diversity-enhanced query augmentation, allowing the model to learn to retrieve effectively. Outline-oriented book encoding improves length extrapolation through bi-level positional encoding and retentive attention mechanisms to maintain context over long sequences. Experiments on a proprietary Baidu dataset demonstrate that GBS outperforms strong baselines, achieving a 9.8\% improvement in terms of MRR@20, over the state-of-the-art RIPOR method... 

**Abstract (ZH)**: 在图书检索中，应针对查询返回相关的图书信息。图书包含复杂且多方面的信息，如元数据、提纲和主体文本，其中提纲提供了章节与节之间的层级信息。生成式检索（GR）是一种新的检索范式，即将语料库信息整合到单一模型中生成与给定查询相关的文档标识符。GR 如何应用于图书检索？直接将 GR 应用于图书检索存在挑战，因为图书检索具有独特的特点：模型需要保留图书的复杂、多方面的信息，这增加了对标记数据的需求。将图书信息拆分并将其视为多个独立片段进行学习可能会导致层级信息的丢失。我们提出了一种有效的图书检索生成式检索框架（GBS），其主要特点包括数据增强和基于提纲的图书编码。在数据增强方面，GBS 构建了多个查询-图书对用于训练；基于提纲、不同形式的图书内容并模拟具有多种伪查询的现实图书检索场景来构建多个图书标识符。这包括覆盖增强的图书标识符增强，允许模型学习有效索引，以及多样性的增强查询增强，允许模型学习有效检索。基于提纲的图书编码通过两层位置编码和保持注意力机制来改善长度外推效果，从而在长序列中维持上下文。在百度内部数据集上的实验表明，GBS 在 MRR@20 方面优于强大的基线方法，比最先进的 RIPOR 方法高出 9.8%。 

---
# Enhancing User Intent for Recommendation Systems via Large Language Models 

**Title (ZH)**: 通过大型语言模型增强推荐系统的用户意图 

**Authors**: Xiaochuan Xu, Zeqiu Xu, Peiyang Yu, Jiani Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10871)  

**Abstract**: Recommendation systems play a critical role in enhancing user experience and engagement in various online platforms. Traditional methods, such as Collaborative Filtering (CF) and Content-Based Filtering (CBF), rely heavily on past user interactions or item features. However, these models often fail to capture the dynamic and evolving nature of user preferences. To address these limitations, we propose DUIP (Dynamic User Intent Prediction), a novel framework that combines LSTM networks with Large Language Models (LLMs) to dynamically capture user intent and generate personalized item recommendations. The LSTM component models the sequential and temporal dependencies of user behavior, while the LLM utilizes the LSTM-generated prompts to predict the next item of interest. Experimental results on three diverse datasets ML-1M, Games, and Bundle show that DUIP outperforms a wide range of baseline models, demonstrating its ability to handle the cold-start problem and real-time intent adaptation. The integration of dynamic prompts based on recent user interactions allows DUIP to provide more accurate, context-aware, and personalized recommendations. Our findings suggest that DUIP is a promising approach for next-generation recommendation systems, with potential for further improvements in cross-modal recommendations and scalability. 

**Abstract (ZH)**: 推荐系统在各类在线平台中对于提升用户体验和参与度方面发挥着关键作用。传统的推荐方法，如协同过滤（Collaborative Filtering, CF）和基于内容的过滤（Content-Based Filtering, CBF），主要依赖于过往的用户交互或项目特征。然而，这些模型往往难以捕捉用户偏好动态变化的特性。为解决这些问题，我们提出了一种名为DUIP（Dynamic User Intent Prediction）的新颖框架，它结合了长短期记忆网络（Long Short-Term Memory, LSTM）和大型语言模型（Large Language Models, LLMs），以动态捕捉用户意图并生成个性化推荐。LSTM组件模型用户行为的序列性和时间依赖性，而LLM则利用LSTM生成的提示来预测用户感兴趣的下一个项目。在三个不同的数据集ML-1M、Games和Bundle上的实验结果表明，DUIP在广泛的基本模型中表现出优越性，证明了其处理冷启动问题和实时意图调整的能力。基于最近用户交互动态提示的集成使得DUIP能够提供更准确、上下文相关且个性化的推荐。我们的研究结果表明，DUIP是一种有前景的下一代推荐系统方法，未来有望在跨模态推荐和扩展性方面进一步改进。 

---
# Diffusion Models in Recommendation Systems: A Survey 

**Title (ZH)**: 推荐系统中的扩散模型：综述 

**Authors**: Ting-Ruen Wei, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10548)  

**Abstract**: Recommender systems remain an essential topic due to its wide application in various domains and the business potential behind them. With the rise of deep learning, common solutions have leveraged neural networks to facilitate collaborative filtering, and some have turned to generative adversarial networks to augment the dataset and tackle the data sparsity issue. However, they are limited in learning the complex user and item distribution and still suffer from model collapse. Given the great generation capability exhibited by diffusion models in computer vision recently, many recommender systems have adopted diffusion models and found improvements in performance for various tasks. Diffusion models in recommender systems excel in managing complex user and item distributions and do not suffer from mode collapse. With these advantages, the amount of research in this domain have been growing rapidly and calling for a systematic survey. In this survey paper, we present and propose a taxonomy on past research papers in recommender systems that utilize diffusion models. Distinct from a prior survey paper that categorizes based on the role of the diffusion model, we categorize based on the recommendation task at hand. The decision originates from the rationale that after all, the adoption of diffusion models is to enhance the recommendation performance, not vice versa: adapting the recommendation task to enable diffusion models. Nonetheless, we offer a unique perspective for diffusion models in recommender systems complementary to existing surveys. We present the foundation algorithms in diffusion models and their applications in recommender systems to summarize the rapid development in this field. Finally, we discuss open research directions to prepare and encourage further efforts to advance the field. We compile the relevant papers in a public GitHub repository. 

**Abstract (ZH)**: 推荐系统仍然是一个重要的研究主题，这得益于其在各个领域中的广泛应用以及其背后的商业潜力。随着深度学习的兴起，常见的解决方案利用神经网络来增强合作过滤的效果，一些研究将生成对抗网络引入数据集扩展中以解决数据稀疏问题。然而，这些方法在学习复杂的用户和项目分布方面受到限制，并且仍然受到模式崩溃的问题困扰。近期，扩散模型在计算机视觉领域的强大生成能力引起了广泛关注，许多推荐系统采用了扩散模型，并在各种任务中取得了性能改进。扩散模型在推荐系统中擅长管理复杂的用户和项目分布，且不会出现模式崩溃的问题。凭借这些优势，针对这一领域的研究正在迅速增长，需要一个系统性的综述。在本文综述中，我们呈现并提出了一种关于利用扩散模型的推荐系统研究论文的分类框架。不同于以往基于扩散模型作用进行分类的综述文章，我们根据推荐任务进行分类。这种分发源于这样一个观点：扩散模型的应用旨在增强推荐性能，而不是相反——将推荐任务适应于使扩散模型发挥作用。尽管如此，我们提供了一个补充现有综述的独特视角，介绍了扩散模型的基础算法及其在推荐系统中的应用，以总结该领域迅速发展的进程。最后，我们探讨了开放的研究方向，以准备并鼓励进一步努力推动该领域的发展。我们将在公共GitHub仓库中整理相关论文。 

---
# Biomedical Knowledge Graph: A Survey of Domains, Tasks, and Real-World Applications 

**Title (ZH)**: 生物医学知识图谱：领域、任务及实际应用综述 

**Authors**: Yuxing Lu, Sin Yee Goi, Xukai Zhao, Jinzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11632)  

**Abstract**: Biomedical knowledge graphs (BKGs) have emerged as powerful tools for organizing and leveraging the vast and complex data found across the biomedical field. Yet, current reviews of BKGs often limit their scope to specific domains or methods, overlooking the broader landscape and the rapid technological progress reshaping it. In this survey, we address this gap by offering a systematic review of BKGs from three core perspectives: domains, tasks, and applications. We begin by examining how BKGs are constructed from diverse data sources, including molecular interactions, pharmacological datasets, and clinical records. Next, we discuss the essential tasks enabled by BKGs, focusing on knowledge management, retrieval, reasoning, and interpretation. Finally, we highlight real-world applications in precision medicine, drug discovery, and scientific research, illustrating the translational impact of BKGs across multiple sectors. By synthesizing these perspectives into a unified framework, this survey not only clarifies the current state of BKG research but also establishes a foundation for future exploration, enabling both innovative methodological advances and practical implementations. 

**Abstract (ZH)**: 生物医学知识图谱（BKGs）已经发展成为组织和利用生物医学领域大量复杂数据的强大工具。然而，目前对BKGs的综述往往局限于特定领域或方法，忽视了更广泛的图谱景观及其快速的技术进步。在此综述中，我们通过从三个核心视角对BKGs进行系统性的回顾来填补这一缺口：领域、任务和应用。我们首先探讨BKGs是如何从包括分子相互作用、药理学数据集和临床记录在内的多种数据源中构建起来的。接着，我们讨论BKGs所支持的核心任务，重点关注知识管理、检索、推理和解释。最后，我们突出展示了BKGs在精准医学、药物发现和科学研究中的实际应用，展示了BKGs在多个领域的转译影响。通过将这些视角综合为一个统一的框架，本综述不仅明确了当前BKG研究的状态，还为未来的探索奠定了基础，促进了创新方法学的进步和实际应用的实施。 

---
# Uncertainty Estimation in the Real World: A Study on Music Emotion Recognition 

**Title (ZH)**: 现实世界中的不确定性估计：音乐情绪识别研究 

**Authors**: Karn N. Watcharasupat, Yiwei Ding, T. Aleksandra Ma, Pavan Seshadri, Alexander Lerch  

**Link**: [PDF](https://arxiv.org/pdf/2501.11570)  

**Abstract**: Any data annotation for subjective tasks shows potential variations between individuals. This is particularly true for annotations of emotional responses to musical stimuli. While older approaches to music emotion recognition systems frequently addressed this uncertainty problem through probabilistic modeling, modern systems based on neural networks tend to ignore the variability and focus only on predicting central tendencies of human subjective responses. In this work, we explore several methods for estimating not only the central tendencies of the subjective responses to a musical stimulus, but also for estimating the uncertainty associated with these responses. In particular, we investigate probabilistic loss functions and inference-time random sampling. Experimental results indicate that while the modeling of the central tendencies is achievable, modeling of the uncertainty in subjective responses proves significantly more challenging with currently available approaches even when empirical estimates of variations in the responses are available. 

**Abstract (ZH)**: 任何针对主观任务的数据标注都可能存在个体间的潜在差异。特别是在对音乐刺激引发的情绪反应进行标注时，这一现象尤为明显。早期针对音乐情绪识别系统的方法通常通过概率建模来解决这种不确定性问题，而基于神经网络的现代系统倾向于忽略这种差异，只专注于预测人类主观响应的中心趋势。在本研究中，我们探索了几种方法，不仅用于估计音乐刺激引发的主观响应的中心趋势，也用于估计这些响应的不确定性。特别地，我们研究了概率损失函数和推理阶段的随机抽样。实验结果表明，虽然可以实现对中心趋势的建模，但在目前可用的方法中，对主观响应的不确定性进行建模证明要困难得多，即使可以获得响应变化的经验估计。 

---
# Verifying Cross-modal Entity Consistency in News using Vision-language Models 

**Title (ZH)**: 使用视觉语言模型在新闻中验证跨模态实体一致性 

**Authors**: Sahar Tahmasebi, Eric Müller-Budack, Ralph Ewerth  

**Link**: [PDF](https://arxiv.org/pdf/2501.11403)  

**Abstract**: The web has become a crucial source of information, but it is also used to spread disinformation, often conveyed through multiple modalities like images and text. The identification of inconsistent cross-modal information, in particular entities such as persons, locations, and events, is critical to detect disinformation. Previous works either identify out-of-context disinformation by assessing the consistency of images to the whole document, neglecting relations of individual entities, or focus on generic entities that are not relevant to news. So far, only few approaches have addressed the task of validating entity consistency between images and text in news. However, the potential of large vision-language models (LVLMs) has not been explored yet. In this paper, we propose an LVLM-based framework for verifying Cross-modal Entity Consistency~(LVLM4CEC), to assess whether persons, locations and events in news articles are consistent across both modalities. We suggest effective prompting strategies for LVLMs for entity verification that leverage reference images crawled from web. Moreover, we extend three existing datasets for the task of entity verification in news providing manual ground-truth data. Our results show the potential of LVLMs for automating cross-modal entity verification, showing improved accuracy in identifying persons and events when using evidence images. Moreover, our method outperforms a baseline for location and event verification in documents. The datasets and source code are available on GitHub at \url{this https URL}. 

**Abstract (ZH)**: 互联网已经成为重要信息来源，但也被用于传播不良信息，这些信息常通过多种模态（如图像和文本）进行传播。特别是在识别不一致的跨模态信息（特别是人物、地点和事件等实体）方面，这对于检测不良信息至关重要。过往的研究要么通过评估图像与整个文档的一致性来识别脱靶的不良信息，忽视了各实体之间的关系；要么专注于与新闻无关的通用实体。迄今为止，只有少数方法尝试验证新闻中图像与文本之间的实体一致性。然而，大型视觉-语言模型（LVLM）的潜力尚未得到充分开发。本文提出了一种基于LVLM的框架（LVLM4CEC），以评估新闻文章中的人物、地点和事件在两种模态下是否具有一致性。我们为LVLM提出了有效提示策略，利用从互联网上爬取的参考图像进行实体验证。此外，我们扩展了三个现有数据集，为新闻中的实体验证任务提供手动标注的数据。我们的结果表明，LVLM在自动化的跨模态实体验证中具有潜力，使用证据图像时在识别人物和事件方面表现出更高的准确性。此外，我们的方法在地点和事件验证方面优于基线方法。数据集和源代码可在GitHub上获取，网址为 \url{this https URL}。 

---
# Counteracting temporal attacks in Video Copy Detection 

**Title (ZH)**: 视频复制检测中对抗时间攻击的方法 

**Authors**: Katarzyna Fojcik, Piotr Syga  

**Link**: [PDF](https://arxiv.org/pdf/2501.11171)  

**Abstract**: Video Copy Detection (VCD) plays a crucial role in copyright protection and content verification by identifying duplicates and near-duplicates in large-scale video databases. The META AI Challenge on video copy detection provided a benchmark for evaluating state-of-the-art methods, with the Dual-level detection approach emerging as a winning solution. This method integrates Video Editing Detection and Frame Scene Detection to handle adversarial transformations and large datasets efficiently. However, our analysis reveals significant limitations in the VED component, particularly in its ability to handle exact copies. Moreover, Dual-level detection shows vulnerability to temporal attacks. To address it, we propose an improved frame selection strategy based on local maxima of interframe differences, which enhances robustness against adversarial temporal modifications while significantly reducing computational overhead. Our method achieves an increase of 1.4 to 5.8 times in efficiency over the standard 1 FPS approach. Compared to Dual-level detection method, our approach maintains comparable micro-average precision ($\mu$AP) while also demonstrating improved robustness against temporal attacks. Given 56\% reduced representation size and the inference time of more than 2 times faster, our approach is more suitable to real-world resource restriction. 

**Abstract (ZH)**: 视频复制检测（VCD）在版权保护和内容验证中起着关键作用，通过识别大规模视频数据库中的重复和近似重复内容。META AI挑战赛中的视频复制检测提供了评估最新方法的标准基准，其中双层检测方法脱颖而出，成为获胜解决方案。该方法结合了视频编辑检测和帧场景检测，以有效地处理敌对变换和大规模数据集。然而，我们的分析揭示了VCD组件（视频编辑检测）在处理精确副本方面存在显著局限性。此外，双层检测方法在应对时间攻击方面也显示出脆弱性。为此，我们提出了一种基于帧间差异局部最大值的改进帧选择策略，该策略不仅增强了对抗时间修改的鲁棒性，还显著降低了计算开销。我们的方法相对于标准每秒1帧（1 FPS）的方法，在效率上提升了1.4至5.8倍。与双层检测方法相比，我们提出的方法在保持微平均精度（$\mu$AP）相似的同时，也表现出对时间攻击更强的鲁棒性。鉴于减少了56%的表示大小和超过2倍更快的推理时间，我们的方法更适用于实际资源限制。 

---
# LegalGuardian: A Privacy-Preserving Framework for Secure Integration of Large Language Models in Legal Practice 

**Title (ZH)**: LegalGuardian：一种在法律实践中安全集成大型语言模型的隐私保护框架 

**Authors**: M. Mikail Demir, Hakan T. Otal, M. Abdullah Canbaz  

**Link**: [PDF](https://arxiv.org/pdf/2501.10915)  

**Abstract**: Large Language Models (LLMs) hold promise for advancing legal practice by automating complex tasks and improving access to justice. However, their adoption is limited by concerns over client confidentiality, especially when lawyers include sensitive Personally Identifiable Information (PII) in prompts, risking unauthorized data exposure. To mitigate this, we introduce LegalGuardian, a lightweight, privacy-preserving framework tailored for lawyers using LLM-based tools. LegalGuardian employs Named Entity Recognition (NER) techniques and local LLMs to mask and unmask confidential PII within prompts, safeguarding sensitive data before any external interaction. We detail its development and assess its effectiveness using a synthetic prompt library in immigration law scenarios. Comparing traditional NER models with one-shot prompted local LLM, we find that LegalGuardian achieves a F1-score of 93% with GLiNER and 97% with Qwen2.5-14B in PII detection. Semantic similarity analysis confirms that the framework maintains high fidelity in outputs, ensuring robust utility of LLM-based tools. Our findings indicate that legal professionals can harness advanced AI technologies without compromising client confidentiality or the quality of legal documents. 

**Abstract (ZH)**: 大型语言模型（LLMs）有潜力通过自动化复杂任务和改善司法 доступ来促进法律实践。然而，它们的应用受到客户保密性担忧的限制，特别是在律师在提示中包含敏感的个人信息（PII）时，这可能会导致未经授权的数据暴露。为了缓解这一问题，我们提出了LegalGuardian，这是一种轻量级、保护隐私的框架，专为使用基于LLM工具的律师设计。LegalGuardian利用命名实体识别（NER）技术及本地LLM，在提示中对敏感PII进行遮蔽与解遮蔽，从而在任何外部交互之前保护敏感数据。我们详细介绍了该框架的开发过程，并通过在移民法场景中使用合成提示库进行了效果评估。我们将传统的NER模型与一次提示的本地LLM进行了比较，发现LegalGuardian在PII检测方面使用GLiNER时达到了93%的F1评分，使用Qwen2.5-14B时达到了97%。语义相似性分析证实，该框架在保持输出高保真度方面表现优异，确保了基于LLM工具的工具的稳健实用性。我们的研究结果表明，法律专业人士可以在不牺牲客户保密性或法律文件质量的情况下利用先进的人工智能技术。 

---
# LD-DETR: Loop Decoder DEtection TRansformer for Video Moment Retrieval and Highlight Detection 

**Title (ZH)**: LD-DETR：循环解码器检测变换器，用于视频关键 moment检索和高光检测 

**Authors**: Pengcheng Zhao, Zhixian He, Fuwei Zhang, Shujin Lin, Fan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.10787)  

**Abstract**: Video Moment Retrieval and Highlight Detection aim to find corresponding content in the video based on a text query. Existing models usually first use contrastive learning methods to align video and text features, then fuse and extract multimodal information, and finally use a Transformer Decoder to decode multimodal information. However, existing methods face several issues: (1) Overlapping semantic information between different samples in the dataset hinders the model's multimodal aligning performance; (2) Existing models are not able to efficiently extract local features of the video; (3) The Transformer Decoder used by the existing model cannot adequately decode multimodal features. To address the above issues, we proposed the LD-DETR model for Video Moment Retrieval and Highlight Detection tasks. Specifically, we first distilled the similarity matrix into the identity matrix to mitigate the impact of overlapping semantic information. Then, we designed a method that enables convolutional layers to extract multimodal local features more efficiently. Finally, we fed the output of the Transformer Decoder back into itself to adequately decode multimodal information. We evaluated LD-DETR on four public benchmarks and conducted extensive experiments to demonstrate the superiority and effectiveness of our approach. Our model outperforms the State-Of-The-Art models on QVHighlight, Charades-STA and TACoS datasets. Our code is available at this https URL. 

**Abstract (ZH)**: 视频片段检索和亮点检测旨在基于文本查询在视频中找到相应的內容。现有模型通常首先使用对比学习方法对视频和文本特征进行对齐，然后融合和提取多模态信息，最后使用Transformer Decoder解码多模态信息。然而，现有方法面临几个问题：（1）数据集中不同样本之间的重叠语义信息阻碍了模型的多模态对齐性能；（2）现有模型无法高效提取视频的局部特征；（3）现有模型中使用的Transformer Decoder无法充分解码多模态特征。为了解决上述问题，我们提出了LD-DETR模型，用于视频片段检索和亮点检测任务。具体而言，我们首先将相似度矩阵提炼成单位矩阵，以减轻不同样本之间的重叠语义信息的影响。然后，我们设计了一种方法，使卷积层能够更有效地提取多模态局部特征。最后，我们将Transformer Decoder的输出重新反馈到其自身，以充分解码多模态信息。我们将LD-DETR在四个公开基准上进行了评估，并进行了广泛的实验以展示我们方法的优越性和有效性。我们的模型在QVHighlight、Charades-STA和TACoS数据集上优于当前最佳模型。我们的代码可以在以下链接获取：[代码链接]。 

---
# A Resource-Efficient Training Framework for Remote Sensing Text--Image Retrieval 

**Title (ZH)**: 一种资源高效训练框架——应用于遥感文本-图像检索 

**Authors**: Weihang Zhang, Jihao Li, Shuoke Li, Ziqing Niu, Jialiang Chen, Wenkai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.10638)  

**Abstract**: Remote sensing text--image retrieval (RSTIR) aims to retrieve the matched remote sensing (RS) images from the database according to the descriptive text. Recently, the rapid development of large visual-language pre-training models provides new insights for RSTIR. Nevertheless, as the complexity of models grows in RSTIR, the previous studies suffer from suboptimal resource efficiency during transfer learning. To address this issue, we propose a computation and memory-efficient retrieval (CMER) framework for RSTIR. To reduce the training memory consumption, we propose the Focus-Adapter module, which adopts a side branch structure. Its focus layer suppresses the interference of background pixels for small targets. Simultaneously, to enhance data efficacy, we regard the RS scene category as the metadata and design a concise augmentation technique. The scene label augmentation leverages the prior knowledge from land cover categories and shrinks the search space. We propose the negative sample recycling strategy to make the negative sample pool decoupled from the mini-batch size. It improves the generalization performance without introducing additional encoders. We have conducted quantitative and qualitative experiments on public datasets and expanded the benchmark with some advanced approaches, which demonstrates the competitiveness of the proposed CMER. Compared with the recent advanced methods, the overall retrieval performance of CMER is 2%--5% higher on RSITMD. Moreover, our proposed method reduces memory consumption by 49% and has a 1.4x data throughput during training. The code of the CMER and the dataset will be released at this https URL. 

**Abstract (ZH)**: 遥感文本-图像检索（RSTIR）旨在根据描述性文本从数据库中检索匹配的遥感（RS）图像。近年来，大规模视觉-语言预训练模型的快速发展为RSTIR提供了新的见解。然而，随着RSTIR中模型复杂性的增加，之前的研究所面临的迁移学习中的资源效率低下问题变得更加显著。为解决这一问题，我们提出了一种计算和内存高效检索（CMER）框架，以应对RSTIR的需求。为了减少训练过程中的内存消耗，我们提出了一种Focus-Adapter模块，该模块采用了分支结构。其聚焦层能够抑制小目标背景像素的干扰。同时，为增强数据的有效性，我们将遥感场景类别作为元数据，并设计了一种简洁的数据增强技术。场景标签增强利用了土地覆盖类别中的先验知识，从而缩小了搜索空间。我们提出了负样本回收策略，使负样本池与小批量大小解耦，从而在不引入额外编码器的情况下提高泛化性能。我们在公开数据集上进行了定量和定性的实验，并扩展了基准，展示了所提出的CMER的竞争力。与最近的先进方法相比，CMER在RSITMD上的整体检索性能提高了2%至5%。此外，我们提出的方法将内存消耗降低了49%，在训练过程中还提高了1.4倍的数据吞吐量。我们将发布CMER的代码和数据集，链接如下：[此 https URL] 

---
# Lossless Compression of Vector IDs for Approximate Nearest Neighbor Search 

**Title (ZH)**: 无损压缩向量ID以实现近似最近邻搜索 

**Authors**: Daniel Severo, Giuseppe Ottaviano, Matthew Muckley, Karen Ullrich, Matthijs Douze  

**Link**: [PDF](https://arxiv.org/pdf/2501.10479)  

**Abstract**: Approximate nearest neighbor search for vectors relies on indexes that are most often accessed from RAM. Therefore, storage is the factor limiting the size of the database that can be served from a machine. Lossy vector compression, i.e., embedding quantization, has been applied extensively to reduce the size of indexes. However, for inverted file and graph-based indices, auxiliary data such as vector ids and links (edges) can represent most of the storage cost. We introduce and evaluate lossless compression schemes for these cases. These approaches are based on asymmetric numeral systems or wavelet trees that exploit the fact that the ordering of ids is irrelevant within the data structures. In some settings, we are able to compress the vector ids by a factor 7, with no impact on accuracy or search runtime. On billion-scale datasets, this results in a reduction of 30% of the index size. Furthermore, we show that for some datasets, these methods can also compress the quantized vector codes losslessly, by exploiting sub-optimalities in the original quantization algorithm. The source code for our approach available at this https URL. 

**Abstract (ZH)**: 矢量近邻搜索通常依赖于从RAM中频繁访问的索引。因此，存储成为决定一台机器可以服务的数据库大小的因素。有损矢量压缩，即嵌入量化，已被广泛应用于减少索引的大小。然而，对于倒排文件和图基索引，辅助数据如矢量ID和链接（边）通常占据了大部分存储成本。我们介绍了并评估了在这些情况下的无损压缩方案。这些方法基于不对称数系统或小波树，利用数据结构中ID排序无相关性的事实。在某些设置下，我们能够将矢量ID压缩7倍，同时不对准确度或搜索时间产生影响。在针对十亿规模的数据集时，这导致索引大小减少了30%。此外，我们表明对于某些数据集，这些方法还可以对原始量化算法中的次优化部分进行无损压缩，从而进一步压缩量化后的矢量代码。我们的方法的源代码可在以下链接获得：[此处链接]。

请注意，您的原文中提到的“[此处链接]”应该替换为实际的链接地址，以便读者能够访问源代码。 

---
# Off-policy Evaluation for Payments at Adyen 

**Title (ZH)**: Adyen支付情况的离策评估 

**Authors**: Alex Egg  

**Link**: [PDF](https://arxiv.org/pdf/2501.10470)  

**Abstract**: This paper demonstrates the successful application of Off-Policy Evaluation (OPE) to accelerate recommender system development and optimization at Adyen, a global leader in financial payment processing. Facing the limitations of traditional A/B testing, which proved slow, costly, and often inconclusive, we integrated OPE to enable rapid evaluation of new recommender system variants using historical data. Our analysis, conducted on a billion-scale dataset of transactions, reveals a strong correlation between OPE estimates and online A/B test results, projecting an incremental 9--54 million transactions over a six-month period. We explore the practical challenges and trade-offs associated with deploying OPE in a high-volume production environment, including leveraging exploration traffic for data collection, mitigating variance in importance sampling, and ensuring scalability through the use of Apache Spark. By benchmarking various OPE estimators, we provide guidance on their effectiveness and integration into the decision-making systems for large-scale industrial payment systems. 

**Abstract (ZH)**: 本文展示了Off-Policy Evaluation (OPE)在加速Adyen（一家全球领先的金融服务支付处理公司）推荐系统开发和优化中的成功应用。面对传统A/B测试的局限性，如速度慢、成本高且往往不具结论性，我们整合了OPE，利用历史数据快速评估新的推荐系统变体。在亿级交易数据集上进行的分析显示，OPE估算值与在线A/B测试结果之间存在显著的相关性，在六个月的时间段内预计可增加900万至5400万次交易。我们探讨了在高流量生产环境中部署OPE所面临的实际挑战与权衡，包括利用探索流量收集数据、减轻重要性抽样的方差以及通过使用Apache Spark确保可扩展性。通过对各种OPE估计器进行基准测试，我们为大规模工业支付系统中的决策系统提供了关于其有效性和集成的指导。 

---
# Making Software FAIR: A machine-assisted workflow for the research software lifecycle 

**Title (ZH)**: 使软件开放可获得且可互操作：一种辅助机器工作流用于研究软件生命周期 

**Authors**: Petr Knoth, Laurent Romary, Patrice Lopez, Roberto Di Cosmo, Pavel Smrz, Tomasz Umerle, Melissa Harrison, Alain Monteil, Matteo Cancellieri, David Pride  

**Link**: [PDF](https://arxiv.org/pdf/2501.10415)  

**Abstract**: A key issue hindering discoverability, attribution and reusability of open research software is that its existence often remains hidden within the manuscript of research papers. For these resources to become first-class bibliographic records, they first need to be identified and subsequently registered with persistent identifiers (PIDs) to be made FAIR (Findable, Accessible, Interoperable and Reusable). To this day, much open research software fails to meet FAIR principles and software resources are mostly not explicitly linked from the manuscripts that introduced them or used them. SoFAIR is a 2-year international project (2024-2025) which proposes a solution to the above problem realised over the content available through the global network of open repositories. SoFAIR will extend the capabilities of widely used open scholarly infrastructures (CORE, Software Heritage, HAL) and tools (GROBID) operated by the consortium partners, delivering and deploying an effective solution for the management of the research software lifecycle, including: 1) ML-assisted identification of research software assets from within the manuscripts of scholarly papers, 2) validation of the identified assets by authors, 3) registration of software assets with PIDs and their archival. 

**Abstract (ZH)**: 阻碍开放研究软件的可发现性、归属权和重复使用的关键问题之一是其存在往往局限于研究论文的手稿中，未能被外界发现。为了使这些资源成为一级文献记录，首先需要对其加以识别，并后续通过持久标识符（PIDs）进行注册，从而使其具备可查找性、可访问性、互操作性和可重用性（FAIR原则）。目前，许多开放研究软件仍未达到FAIR原则，大部分软件资源在引入或使用它们的研究论文中也没有被明确链接。SoFAIR是一个为期两年的国际项目（2024-2025），旨在解决上述问题，该项目基于通过全球开放存取仓储网络获得的内容实现解决方案。SoFAIR将扩展由合作伙伴运营的广泛使用的开放学术基础设施（如CORE、软件遗产、HAL）和工具（如GROBID）的能力，提供并部署一个有效的解决方案来管理研究软件生命周期，包括：1）利用机器学习辅助识别研究软件资产，从学术论文的手稿中提取这些资产；2）由作者验证识别出的资产；3）将软件资产注册并归档到持久标识符中。 

---
