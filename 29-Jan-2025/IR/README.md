# Enhanced Retrieval of Long Documents: Leveraging Fine-Grained Block Representations with Large Language Models 

**Title (ZH)**: 增强长文档检索：利用大型语言模型细粒度块表示技术 

**Authors**: Minghan Li, Eric Gaussier, Guodong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2501.17039)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated exceptional power in various domains, including information retrieval. Most of the previous practices involve leveraging these models to create a single embedding for each query, each passage, or each document individually, a strategy exemplified and used by the Retrieval-Augmented Generation (RAG) framework. While this method has proven effective, we argue that it falls short in fully capturing the nuanced intricacies of document-level texts due to its reliance on a relatively coarse-grained representation. To address this limitation, we introduce a novel, fine-grained approach aimed at enhancing the accuracy of relevance scoring for long documents. Our methodology firstly segments a long document into blocks, each of which is embedded using an LLM, for matching with the query representation. When calculating the relevance score, we aggregate the query-block relevance scores through a weighted sum method, yielding a comprehensive score for the query with the entire document. Despite its apparent simplicity, our experimental findings reveal that this approach outperforms standard representation methods and achieves a significant reduction in embedding generation latency. Moreover, by carefully optimizing pairwise loss functions, superior performances have been achieved. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在各种领域中展现出 exceptional 的能力，特别是在信息检索方面。大多数以往的做法都通过利用这些模型为每个查询、每段文本或每个文档单独生成一个嵌入向量，这一策略由检索增强生成（RAG）框架所体现。尽管这种方法已被证明是有效的，但我们认为它在捕捉文档级文本的细微差别时依然存在不足，这是因为其依赖于相对粗粒度的表示。为了解决这一问题，我们提出了一种新的细粒度方法，旨在提高长文档相关性评分的准确性。我们的方法首先将长文档分割成块，每个块都使用 LLM 进行嵌入，然后与查询表示进行匹配。在计算相关性评分时，我们通过加权求和的方法聚合查询-块的相关性评分，从而得到查询与整个文档的综合评分。尽管这种方法看似简单，但我们的实验结果表明，这种方法在准确性和嵌入生成延迟方面均优于标准表示方法。此外，通过精心优化成对损失函数，我们实现了显著的性能提升。 

---
# Document Screenshot Retrievers are Vulnerable to Pixel Poisoning Attacks 

**Title (ZH)**: 文档截图检索器易受像素中毒攻击的影响 

**Authors**: Shengyao Zhuang, Ekaterina Khramtsova, Xueguang Ma, Bevan Koopman, Jimmy Lin, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2501.16902)  

**Abstract**: Recent advancements in dense retrieval have introduced vision-language model (VLM)-based retrievers, such as DSE and ColPali, which leverage document screenshots embedded as vectors to enable effective search and offer a simplified pipeline over traditional text-only methods. In this study, we propose three pixel poisoning attack methods designed to compromise VLM-based retrievers and evaluate their effectiveness under various attack settings and parameter configurations. Our empirical results demonstrate that injecting even a single adversarial screenshot into the retrieval corpus can significantly disrupt search results, poisoning the top-10 retrieved documents for 41.9% of queries in the case of DSE and 26.4% for ColPali. These vulnerability rates notably exceed those observed with equivalent attacks on text-only retrievers. Moreover, when targeting a small set of known queries, the attack success rate raises, achieving complete success in certain cases. By exposing the vulnerabilities inherent in vision-language models, this work highlights the potential risks associated with their deployment. 

**Abstract (ZH)**: 近年来，密集检索领域的最新进展引入了基于视觉-语言模型（VLM）的检索器，如DSE和ColPali，这些模型通过将文档截图嵌入向量中，实现了有效的搜索，并提供了一种比传统纯文本方法更简化的流水线。在本研究中，我们提出了三种像素篡改攻击方法，旨在破坏基于VLM的检索器，并在不同的攻击设置和参数配置下评估其有效性。我们的实验结果表明，即使向检索语料库中注入单个 adversarial screenshot 也能显著破坏搜索结果，在DSE中，有41.9%的查询导致检索到的前10篇文档被污染，在ColPali中，这一比例为26.4%。这些漏洞率明显高于对纯文本检索器进行等效攻击所观察到的漏洞率。此外，当针对一组已知查询时，攻击成功率提高，某些情况下可以完全成功。通过揭示视觉-语言模型固有的漏洞，本研究强调了它们部署时可能面临的风险。 

---
# Secure Federated Graph-Filtering for Recommender Systems 

**Title (ZH)**: 安全联邦图过滤推荐系统 

**Authors**: Julien Nicolas, César Sabater, Mohamed Maouche, Sonia Ben Mokhtar, Mark Coates  

**Link**: [PDF](https://arxiv.org/pdf/2501.16888)  

**Abstract**: Recommender systems often rely on graph-based filters, such as normalized item-item adjacency matrices and low-pass filters. While effective, the centralized computation of these components raises concerns about privacy, security, and the ethical use of user data. This work proposes two decentralized frameworks for securely computing these critical graph components without centralizing sensitive information. The first approach leverages lightweight Multi-Party Computation and distributed singular vector computations to privately compute key graph filters. The second extends this framework by incorporating low-rank approximations, enabling a trade-off between communication efficiency and predictive performance. Empirical evaluations on benchmark datasets demonstrate that the proposed methods achieve comparable accuracy to centralized state-of-the-art systems while ensuring data confidentiality and maintaining low communication costs. Our results highlight the potential for privacy-preserving decentralized architectures to bridge the gap between utility and user data protection in modern recommender systems. 

**Abstract (ZH)**: 推荐系统通常依赖于基于图的滤波器，例如归一化的物品-物品邻接矩阵和低通滤波器。虽然这些方法非常有效，但这些组件的集中式计算引发了关于隐私、安全以及用户数据伦理使用的担忧。本文提出了一种去中心化的框架，可以在不集中敏感信息的情况下安全地计算这些关键的图组件。第一种方法利用轻量级多方计算和分布式奇异向量计算来私密地计算关键图滤波器。第二种方法在此基础上引入了低秩近似，以在通信效率与预测性能之间寻求平衡。实验评价表明，所提出的方法在基准数据集上的准确度与集中式最先进的系统相当，同时确保数据保密性并保持较低的通信成本。我们的结果强调了隐私保护的去中心化架构在现代推荐系统中实现实用性和用户数据保护之间平衡的潜力。 

---
# Hypergraph Diffusion for High-Order Recommender Systems 

**Title (ZH)**: 高阶荐系统中的超图扩散技术 

**Authors**: Darnbi Sakong, Thanh Trung Huynh, Jun Jo  

**Link**: [PDF](https://arxiv.org/pdf/2501.16722)  

**Abstract**: Recommender systems rely on Collaborative Filtering (CF) to predict user preferences by leveraging patterns in historical user-item interactions. While traditional CF methods primarily focus on learning compact vector embeddings for users and items, graph neural network (GNN)-based approaches have emerged as a powerful alternative, utilizing the structure of user-item interaction graphs to enhance recommendation accuracy. However, existing GNN-based models, such as LightGCN and UltraGCN, often struggle with two major limitations: an inability to fully account for heterophilic interactions, where users engage with diverse item categories, and the over-smoothing problem in multi-layer GNNs, which hinders their ability to model complex, high-order relationships. To address these gaps, we introduce WaveHDNN, an innovative wavelet-enhanced hypergraph diffusion framework. WaveHDNN integrates a Heterophily-aware Collaborative Encoder, designed to capture user-item interactions across diverse categories, with a Multi-scale Group-wise Structure Encoder, which leverages wavelet transforms to effectively model localized graph structures. Additionally, cross-view contrastive learning is employed to maintain robust and consistent representations. Experiments on benchmark datasets validate the efficacy of WaveHDNN, demonstrating its superior ability to capture both heterophilic and localized structural information, leading to improved recommendation performance. 

**Abstract (ZH)**: 推荐系统依赖于协作过滤（Collaborative Filtering, CF）来通过挖掘用户-项目历史交互模式来预测用户偏好。传统的CF方法主要集中在学习用户和项目的紧凑向量嵌入上，而基于图神经网络（Graph Neural Network, GNN）的方法则因其能够利用用户-项目交互图的结构来提升推荐准确性而成为强有力的竞争者。然而，现有的GNN基模型，如LightGCN和UltraGCN，在处理两类主要限制方面经常表现出不足：一是无法充分考虑异构交互，即用户与多种不同类型的项目进行交互；二是多层GNN中的过平滑问题，这妨碍了其建模复杂、高阶关系的能力。为了解决这些问题，我们提出了WaveHDNN，这是一种创新的波动增强超图扩散框架。WaveHDNN集成了一个异构性感知的合作编码器，旨在跨多种类别捕捉用户-项目的交互，以及一个多尺度分组结构编码器，利用小波变换有效地建模局部图结构。此外，采用跨视图对比学习以保持鲁棒且一致的表示。基准数据集上的实验验证了WaveHDNN的有效性，展示了其在捕捉异构性和局部结构信息方面的优越能力，从而提升了推荐性能。 

---
# 360Brew: A Decoder-only Foundation Model for Personalized Ranking and Recommendation 

**Title (ZH)**: 360Brew：一种用于个性化排名和推荐的解码器-only 基础模型 

**Authors**: Hamed Firooz, Maziar Sanjabi, Adrian Englhardt, Aman Gupta, Ben Levine, Dre Olgiati, Gungor Polatkan, Iuliia Melnychuk, Karthik Ramgopal, Kirill Talanine, Kutta Srinivasan, Luke Simon, Natesh Sivasubramoniapillai, Necip Fazil Ayan, Qingquan Song, Samira Sriram, Souvik Ghosh, Tao Song, Vignesh Kothapalli, Xiaoling Zhai, Ya Xu, Yu Wang, Yun Dai  

**Link**: [PDF](https://arxiv.org/pdf/2501.16450)  

**Abstract**: Ranking and recommendation systems are the foundation for numerous online experiences, ranging from search results to personalized content delivery. These systems have evolved into complex, multilayered architectures that leverage vast datasets and often incorporate thousands of predictive models. The maintenance and enhancement of these models is a labor intensive process that requires extensive feature engineering. This approach not only exacerbates technical debt but also hampers innovation in extending these systems to emerging problem domains. In this report, we present our research to address these challenges by utilizing a large foundation model with a textual interface for ranking and recommendation tasks. We illustrate several key advantages of our approach: (1) a single model can manage multiple predictive tasks involved in ranking and recommendation, (2) decoder models with textual interface due to their comprehension of reasoning capabilities, can generalize to new recommendation surfaces and out-of-domain problems, and (3) by employing natural language interfaces for task definitions and verbalizing member behaviors and their social connections, we eliminate the need for feature engineering and the maintenance of complex directed acyclic graphs of model dependencies. We introduce our research pre-production model, 360Brew V1.0, a 150B parameter, decoder-only model that has been trained and fine-tuned on LinkedIn's data and tasks. This model is capable of solving over 30 predictive tasks across various segments of the LinkedIn platform, achieving performance levels comparable to or exceeding those of current production systems based on offline metrics, without task-specific fine-tuning. Notably, each of these tasks is conventionally addressed by dedicated models that have been developed and maintained over multiple years by teams of a similar or larger size than our own. 

**Abstract (ZH)**: 排名和推荐系统是众多在线体验的基础，从搜索结果到个性化内容配送。这些系统已经发展成为复杂的多层架构，利用大规模数据集，并经常整合数千个预测模型。这些模型的维护和优化是一个劳动密集型的过程，需要大量特征工程的支持。这种方法不仅加重了技术债务，还阻碍了将这些系统扩展至新兴问题领域的创新。本报告中，我们通过利用具有文本接口的基础模型，提出了我们的研究来应对这些挑战。我们展示了我们方法的几个关键优势：（1）一个模型可以管理排名和推荐任务中涉及的多项预测任务；（2）由于具备理解和应用推理能力的解码器模型可以通过文本接口进行推广，从而适用于新的推荐界面和跨领域问题；（3）通过使用自然语言界面定义任务，并用自然语言描述成员行为及其社会联系，我们可消除特征工程的需求，并避免维护复杂的模型依赖的有向无环图。我们介绍我们的研究预生产模型360Brew V1.0，一个包含150B参数、仅解码器的模型，已在领英的数据和任务上进行训练和微调。该模型能够在领英平台的不同领域中解决超过30项预测任务，且在离线指标下达到当前生产系统的性能水平或超越，而无需针对特定任务进行微调。值得注意的是，每个任务通常由类似或更大规模的团队开发并维护多年的专门模型来处理。 

---
# VeriFact: Verifying Facts in LLM-Generated Clinical Text with Electronic Health Records 

**Title (ZH)**: VeriFact：通过电子健康记录验证大规模语言模型生成的临床文本中的事实 

**Authors**: Philip Chung, Akshay Swaminathan, Alex J. Goodell, Yeasul Kim, S. Momsen Reincke, Lichy Han, Ben Deverett, Mohammad Amin Sadeghi, Abdel-Badih Ariss, Marc Ghanem, David Seong, Andrew A. Lee, Caitlin E. Coombes, Brad Bradshaw, Mahir A. Sufian, Hyo Jung Hong, Teresa P. Nguyen, Mohammad R. Rasouli, Komal Kamra, Mark A. Burbridge, James C. McAvoy, Roya Saffary, Stephen P. Ma, Dev Dash, James Xie, Ellen Y. Wang, Clifford A. Schmiesing, Nigam Shah, Nima Aghaeepour  

**Link**: [PDF](https://arxiv.org/pdf/2501.16672)  

**Abstract**: Methods to ensure factual accuracy of text generated by large language models (LLM) in clinical medicine are lacking. VeriFact is an artificial intelligence system that combines retrieval-augmented generation and LLM-as-a-Judge to verify whether LLM-generated text is factually supported by a patient's medical history based on their electronic health record (EHR). To evaluate this system, we introduce VeriFact-BHC, a new dataset that decomposes Brief Hospital Course narratives from discharge summaries into a set of simple statements with clinician annotations for whether each statement is supported by the patient's EHR clinical notes. Whereas highest agreement between clinicians was 88.5%, VeriFact achieves up to 92.7% agreement when compared to a denoised and adjudicated average human clinican ground truth, suggesting that VeriFact exceeds the average clinician's ability to fact-check text against a patient's medical record. VeriFact may accelerate the development of LLM-based EHR applications by removing current evaluation bottlenecks. 

**Abstract (ZH)**: 大型语言模型（LLM）生成的文本在临床医学中确保事实准确性的方法存在不足。VeriFact 是一种人工智能系统，结合了检索增强生成和“LLM作为法官”的技术，以验证生成的文本是否得到了患者医疗历史记录（基于电子健康记录，EHR）的支持。为了评估该系统，我们引入了VeriFact-BHC这一新数据集，将出院总结中的简要住院病程报告分解为一系列简单陈述，并由临床医生标注每个陈述是否得到患者EHR临床笔记的支持。虽然临床医生之间的最高一致性为88.5%，但VeriFact与去噪和仲裁后的平均临床医生真实地面真相相比，一致率达到92.7%，表明VeriFact超越了平均临床医生核查文本与患者医疗记录之间一致性的能力。VeriFact 可能通过消除当前评估瓶颈来加速LLM驱动的EHR应用的开发。 

---
# On Storage Neural Network Augmented Approximate Nearest Neighbor Search 

**Title (ZH)**: 基于存储神经网络增强的近似最近邻搜索 

**Authors**: Taiga Ikeda, Daisuke Miyashita, Jun Deguchi  

**Link**: [PDF](https://arxiv.org/pdf/2501.16375)  

**Abstract**: Large-scale approximate nearest neighbor search (ANN) has been gaining attention along with the latest machine learning researches employing ANNs. If the data is too large to fit in memory, it is necessary to search for the most similar vectors to a given query vector from the data stored in storage devices, not from that in memory. The storage device such as NAND flash memory has larger capacity than the memory device such as DRAM, but they also have larger latency to read data. Therefore, ANN methods for storage require completely different approaches from conventional in-memory ANN methods. Since the approximation that the time required for search is determined only by the amount of data fetched from storage holds under reasonable assumptions, our goal is to minimize it while maximizing recall. For partitioning-based ANNs, vectors are partitioned into clusters in the index building phase. In the search phase, some of the clusters are chosen, the vectors in the chosen clusters are fetched from storage, and the nearest vector is retrieved from the fetched vectors. Thus, the key point is to accurately select the clusters containing the ground truth nearest neighbor vectors. We accomplish this by proposing a method to predict the correct clusters by means of a neural network that is gradually refined by alternating supervised learning and duplicated cluster assignment. Compared to state-of-the-art SPANN and an exhaustive method using k-means clustering and linear search, the proposed method achieves 90% recall on SIFT1M with 80% and 58% less data fetched from storage, respectively. 

**Abstract (ZH)**: 大规模近似最近邻搜索（Approximate Nearest Neighbor Search, ANN）随着最新的机器学习研究中对ANN的应用而逐渐得到关注。如果数据量过大无法全部存储在内存中，则需要在存储设备中搜索与给定查询向量最相似的向量，而不是在内存中进行搜索。例如，与DRAM这类内存设备相比，NAND闪存这类存储设备的容量更大，但读取数据的延迟也更大。因此，针对存储设备的ANN方法与传统的内存中ANN方法需要完全不同的方法。在合理的假设下，如果搜索所需时间仅由从存储中获取的数据量决定，我们的目标是在最小化数据获取量的同时最大化召回率。对于基于分区的ANN方法，索引构建阶段将向量划分为多个簇。在搜索阶段，根据某些策略选择一些簇中的向量，从存储中获取这些向量，然后从中检索最近的向量。因此，关键点是准确地选择包含真实最近邻向量的簇。我们通过交替进行监督学习和重复簇分配，并利用神经网络预测正确簇的方法来实现这一点。与最先进的SPANN方法以及使用k-means聚类和线性搜索的穷举方法相比，我们提出的方法在SIFT1M数据集上分别减少了80%和58%的数据获取量，同时达到90%的召回率。 

---
# RAPID: Retrieval-Augmented Parallel Inference Drafting for Text-Based Video Event Retrieval 

**Title (ZH)**: RAPID：基于检索增强并行推理的文字驱动视频事件检索草稿方法 

**Authors**: Long Nguyen, Huy Nguyen, Bao Khuu, Huy Luu, Huy Le, Tuan Nguyen, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2501.16303)  

**Abstract**: Retrieving events from videos using text queries has become increasingly challenging due to the rapid growth of multimedia content. Existing methods for text-based video event retrieval often focus heavily on object-level descriptions, overlooking the crucial role of contextual information. This limitation is especially apparent when queries lack sufficient context, such as missing location details or ambiguous background elements. To address these challenges, we propose a novel system called RAPID (Retrieval-Augmented Parallel Inference Drafting), which leverages advancements in Large Language Models (LLMs) and prompt-based learning to semantically correct and enrich user queries with relevant contextual information. These enriched queries are then processed through parallel retrieval, followed by an evaluation step to select the most relevant results based on their alignment with the original query. Through extensive experiments on our custom-developed dataset, we demonstrate that RAPID significantly outperforms traditional retrieval methods, particularly for contextually incomplete queries. Our system was validated for both speed and accuracy through participation in the Ho Chi Minh City AI Challenge 2024, where it successfully retrieved events from over 300 hours of video. Further evaluation comparing RAPID with the baseline proposed by the competition organizers demonstrated its superior effectiveness, highlighting the strength and robustness of our approach. 

**Abstract (ZH)**: 由于多媒体内容的快速增长，使用文本查询从视频中检索事件变得日益具有挑战性。现有基于文本的视频事件检索方法往往侧重于对物体级别的描述，忽视了上下文信息的关键作用。尤其在查询缺乏足够上下文时，这一局限性尤为明显，例如缺少地点细节或背景元素模糊不清。为了解决这些挑战，我们提出了一种新的系统，称为RAPID（Retrieval-Augmented Parallel Inference Drafting），该系统利用了大型语言模型（LLMs）和基于提示的学习技术，通过相关上下文信息改进用户的查询。改进后的查询随后通过并行检索进行处理，并通过评估步骤根据与原始查询的匹配度选择最相关的结果。通过在我们自开发的数据集上进行广泛实验，我们证明了RAPID在几乎所有情况下都显著优于传统检索方法，特别是在上下文不完整查询方面。通过参加2024胡志明市AI挑战赛，我们的系统在速 度和准确性方面得到了验证，并成功从超过300小时的视频中检索到事件。进一步的评估将RAPID与竞赛组织方提供的基线方法进行比较，显示了其优越的效果，突显了我们方法的强劲与稳健性。 

---
# URAG: Implementing a Unified Hybrid RAG for Precise Answers in University Admission Chatbots -- A Case Study at HCMUT 

**Title (ZH)**: URAG：在胡志明市科技大学大学入学聊天机器人中实现统一混合检索生成模型以获取精确答案——一个案例研究 

**Authors**: Long Nguyen, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2501.16276)  

**Abstract**: With the rapid advancement of Artificial Intelligence, particularly in Natural Language Processing, Large Language Models (LLMs) have become pivotal in educational question-answering systems, especially university admission chatbots. Concepts such as Retrieval-Augmented Generation (RAG) and other advanced techniques have been developed to enhance these systems by integrating specific university data, enabling LLMs to provide informed responses on admissions and academic counseling. However, these enhanced RAG techniques often involve high operational costs and require the training of complex, specialized modules, which poses challenges for practical deployment. Additionally, in the educational context, it is crucial to provide accurate answers to prevent misinformation, a task that LLM-based systems find challenging without appropriate strategies and methods. In this paper, we introduce the Unified RAG (URAG) Framework, a hybrid approach that significantly improves the accuracy of responses, particularly for critical queries. Experimental results demonstrate that URAG enhances our in-house, lightweight model to perform comparably to state-of-the-art commercial models. Moreover, to validate its practical applicability, we conducted a case study at our educational institution, which received positive feedback and acclaim. This study not only proves the effectiveness of URAG but also highlights its feasibility for real-world implementation in educational settings. 

**Abstract (ZH)**: 随着人工智能的迅速发展，特别是在自然语言处理领域的进步，大型语言模型（LLMs）在教育问答系统中扮演着关键角色，尤其是在大学入学聊天机器人方面。检索增强生成（RAG）等先进技术和其他相关技术的发展，通过融合特定的大学数据，增强这些系统的能力，使LLMs能够在录取和学术咨询方面提供更加精准的回答。然而，这些增强的RAG技术通常涉及较高的运营成本，并需要培训复杂的专用模块，这给实际部署带来了挑战。此外，在教育领域，提供准确的答案以防止信息误导至关重要，这是一项LLM基于系统难以完成的任务，除非具备适当的战略和方法。在这项论文中，我们提出了统一的RAG（URAG）框架，这是一种混合方法，显著提高了关键查询的准确性。实验结果表明，URAG提升了我们内部的轻量级模型，使其与最先进的商用模型性能相当。此外，为了验证其实际应用可能性，我们在教育机构进行了案例研究，获得了积极的反馈和认可。这项研究不仅证明了URAG的有效性，还突显了其在教育场景中实际实施的可行性。 

---
