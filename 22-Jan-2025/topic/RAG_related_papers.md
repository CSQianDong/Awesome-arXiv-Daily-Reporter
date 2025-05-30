# Poison-RAG: Adversarial Data Poisoning Attacks on Retrieval-Augmented Generation in Recommender Systems 

**Title (ZH)**: Poison-RAG：面向推荐系统检索增强生成的对抗性数据污染攻击 

**Authors**: Fatemeh Nazary, Yashar Deldjoo, Tommaso di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2501.11759)  

**Abstract**: This study presents Poison-RAG, a framework for adversarial data poisoning attacks targeting retrieval-augmented generation (RAG)-based recommender systems. Poison-RAG manipulates item metadata, such as tags and descriptions, to influence recommendation outcomes. Using item metadata generated through a large language model (LLM) and embeddings derived via the OpenAI API, we explore the impact of adversarial poisoning attacks on provider-side, where attacks are designed to promote long-tail items and demote popular ones. Two attack strategies are proposed: local modifications, which personalize tags for each item using BERT embeddings, and global modifications, applying uniform tags across the dataset. Experiments conducted on the MovieLens dataset in a black-box setting reveal that local strategies improve manipulation effectiveness by up to 50\%, while global strategies risk boosting already popular items. Results indicate that popular items are more susceptible to attacks, whereas long-tail items are harder to manipulate. Approximately 70\% of items lack tags, presenting a cold-start challenge; data augmentation and synthesis are proposed as potential defense mechanisms to enhance RAG-based systems' resilience. The findings emphasize the need for robust metadata management to safeguard recommendation frameworks. Code and data are available at this https URL. 

**Abstract (ZH)**: 本文提出了一种名为Poison-RAG的框架，该框架针对基于检索增强生成（RAG）的推荐系统进行了对抗性数据投毒攻击研究。Poison-RAG通过操控项目元数据（如标签和描述）来影响推荐结果。利用大型语言模型（LLM）生成的项目元数据和通过OpenAI API获取的嵌入信息，我们探索了对抗性投毒攻击在提供者侧的影响，此类攻击旨在促进长尾项目，并降低热门项目的推荐。两种攻击策略被提出：局部修改，使用BERT嵌入个性化每个项目的标签；全局修改，对整个数据集应用统一的标签。在黑盒设置下，使用MovieLens数据集进行的实验表明，局部策略可以提高操纵效果高达50%，而全局策略则存在提升已热门项目的风险。结果表明，热门项目更易受到攻击，而长尾项目更难被操纵。大约70%的项目缺乏标签，这构成了冷启动挑战；数据增强和合成被提议作为潜在的防御机制，以增强基于RAG的系统的抗攻击能力。研究结果强调了稳健的元数据管理对于保护推荐框架的重要性。相关代码和数据可在此网页访问：[提供链接]。 

---
# Generative Retrieval for Book search 

**Title (ZH)**: 书籍检索的生成性检索 

**Authors**: Yubao Tang, Ruqing Zhang, Jiafeng Guo, Maarten de Rijke, Shihao Liu, Shuaiqing Wang, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2501.11034)  

**Abstract**: In book search, relevant book information should be returned in response to a query. Books contain complex, multi-faceted information such as metadata, outlines, and main text, where the outline provides hierarchical information between chapters and sections. Generative retrieval (GR) is a new retrieval paradigm that consolidates corpus information into a single model to generate identifiers of documents that are relevant to a given query. How can GR be applied to book search? Directly applying GR to book search is a challenge due to the unique characteristics of book search: The model needs to retain the complex, multi-faceted information of the book, which increases the demand for labeled data. Splitting book information and treating it as a collection of separate segments for learning might result in a loss of hierarchical information. We propose an effective Generative retrieval framework for Book Search (GBS) that features two main components: data augmentation and outline-oriented book encoding. For data augmentation, GBS constructs multiple query-book pairs for training; it constructs multiple book identifiers based on the outline, various forms of book contents, and simulates real book retrieval scenarios with varied pseudo-queries. This includes coverage-promoting book identifier augmentation, allowing the model to learn to index effectively, and diversity-enhanced query augmentation, allowing the model to learn to retrieve effectively. Outline-oriented book encoding improves length extrapolation through bi-level positional encoding and retentive attention mechanisms to maintain context over long sequences. Experiments on a proprietary Baidu dataset demonstrate that GBS outperforms strong baselines, achieving a 9.8\% improvement in terms of MRR@20, over the state-of-the-art RIPOR method... 

**Abstract (ZH)**: 在图书检索中，应针对查询返回相关的图书信息。图书包含复杂且多方面的信息，如元数据、提纲和主体文本，其中提纲提供了章节与节之间的层级信息。生成式检索（GR）是一种新的检索范式，即将语料库信息整合到单一模型中生成与给定查询相关的文档标识符。GR 如何应用于图书检索？直接将 GR 应用于图书检索存在挑战，因为图书检索具有独特的特点：模型需要保留图书的复杂、多方面的信息，这增加了对标记数据的需求。将图书信息拆分并将其视为多个独立片段进行学习可能会导致层级信息的丢失。我们提出了一种有效的图书检索生成式检索框架（GBS），其主要特点包括数据增强和基于提纲的图书编码。在数据增强方面，GBS 构建了多个查询-图书对用于训练；基于提纲、不同形式的图书内容并模拟具有多种伪查询的现实图书检索场景来构建多个图书标识符。这包括覆盖增强的图书标识符增强，允许模型学习有效索引，以及多样性的增强查询增强，允许模型学习有效检索。基于提纲的图书编码通过两层位置编码和保持注意力机制来改善长度外推效果，从而在长序列中维持上下文。在百度内部数据集上的实验表明，GBS 在 MRR@20 方面优于强大的基线方法，比最先进的 RIPOR 方法高出 9.8%。 

---
# Med-R$^2$: Crafting Trustworthy LLM Physicians through Retrieval and Reasoning of Evidence-Based Medicine 

**Title (ZH)**: Med-R²：通过基于证据的医学检索与推理打造可信赖的LLM医生 

**Authors**: Keer Lu, Zheng Liang, Da Pan, Shusen Zhang, Xin Wu, Weipeng Chen, Zenan Zhou, Guosheng Dong, Bin Cui, Wentao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.11885)  

**Abstract**: In recent years, Large Language Models (LLMs) have exhibited remarkable capabilities in clinical scenarios. However, despite their potential, existing works face challenges when applying LLMs to medical settings. Strategies relying on training with medical datasets are highly cost-intensive and may suffer from outdated training data. Leveraging external knowledge bases is a suitable alternative, yet it faces obstacles such as limited retrieval precision and poor effectiveness in answer extraction. These issues collectively prevent LLMs from demonstrating the expected level of proficiency in mastering medical expertise. To address these challenges, we introduce Med-R^2, a novel LLM physician framework that adheres to the Evidence-Based Medicine (EBM) process, efficiently integrating retrieval mechanisms as well as the selection and reasoning processes of evidence, thereby enhancing the problem-solving capabilities of LLMs in healthcare scenarios and fostering a trustworthy LLM physician. Our comprehensive experiments indicate that Med-R^2 achieves a 14.87\% improvement over vanilla RAG methods and even a 3.59\% enhancement compared to fine-tuning strategies, without incurring additional training costs. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在临床场景中展现出了非凡的能力。然而，尽管这些模型具有潜力，在将它们应用于医疗领域时仍面临诸多挑战。依赖医学数据集进行训练的方法成本高昂，且可能受到过时训练数据的影响。利用外部知识库是一个可行的替代方案，但此类方法也面临着诸如检索精度较低和难以有效抽取答案等障碍。这些问题共同阻碍了LLMs在掌握医学专业知识方面达到预期水平。为解决这些问题，我们提出了Med-R^2，这是一种新颖的LLM医生框架，遵循循证医学（EBM）过程，高效地整合了检索机制以及证据的选择和推理过程，从而增强LLMs在医疗场景中的问题解决能力，并培养一种值得信赖的LLM医生。我们全面的实验表明，Med-R^2 在与基础检索聚合（RAG）方法相比时展示了14.87% 的改进，并且相较于微调策略还展现了3.59% 的提升，而无需额外的训练成本。 

---
# Network-informed Prompt Engineering against Organized Astroturf Campaigns under Extreme Class Imbalance 

**Title (ZH)**: 网络导向的提示工程以对抗极端类别不平衡下的组织化水军运动 

**Authors**: Nikos Kanakaris, Heng Ping, Xiongye Xiao, Nesreen K. Ahmed, Luca Luceri, Emilio Ferrara, Paul Bogdan  

**Link**: [PDF](https://arxiv.org/pdf/2501.11849)  

**Abstract**: Detecting organized political campaigns is of paramount importance in fighting against disinformation on social media. Existing approaches for the identification of such organized actions employ techniques mostly from network science, graph machine learning and natural language processing. Their ultimate goal is to analyze the relationships and interactions (e.g. re-posting) among users and the textual similarities of their posts. Despite their effectiveness in recognizing astroturf campaigns, these methods face significant challenges, notably the class imbalance in available training datasets. To mitigate this issue, recent methods usually resort to data augmentation or increasing the number of positive samples, which may not always be feasible or sufficient in real-world settings. Following a different path, in this paper, we propose a novel framework for identifying astroturf campaigns based solely on large language models (LLMs), introducing a Balanced Retrieval-Augmented Generation (Balanced RAG) component. Our approach first gives both textual information concerning the posts (in our case tweets) and the user interactions of the social network as input to a language model. Then, through prompt engineering and the proposed Balanced RAG method, it effectively detects coordinated disinformation campaigns on X (Twitter). The proposed framework does not require any training or fine-tuning of the language model. Instead, by strategically harnessing the strengths of prompt engineering and Balanced RAG, it facilitates LLMs to overcome the effects of class imbalance and effectively identify coordinated political campaigns. The experimental results demonstrate that by incorporating the proposed prompt engineering and Balanced RAG methods, our framework outperforms the traditional graph-based baselines, achieving 2x-3x improvements in terms of precision, recall and F1 scores. 

**Abstract (ZH)**: 检测有组织的政治活动对于打击社交 media 上的虚假信息至关重要。现有的识别此类有组织行动的方法主要采用了网络科学、图机器学习和自然语言处理的技术。这些方法的最终目标是分析用户之间的关系和互动（例如转发）以及其帖子的文本相似性。尽管这些方法在识别假流量活动方面表现出有效性，但它们在应对可用训练数据集中的类别不平衡问题时面临显著挑战。为缓解这一问题，近年来的方法通常依赖于数据增强或增加正样本数量，但在实际应用场景中这可能并不总是可行或足够。

与这种方法不同，本文提出了一种基于大型语言模型（LLMs）的新框架，引入了平衡检索增强生成（Balanced RAG）组件。我们的方法首先将有关帖子（例如推特）的文本信息以及社交网络用户的互动信息输入到语言模型。然后，通过提示工程和所提出的平衡检索增强生成方法，有效检测 X（推特）上的协调性虚假信息活动。该提出的框架不需要对语言模型进行任何训练或微调。相反，通过战略性地利用提示工程和平衡检索增强生成的优势，它使大型语言模型能够克服类别不平衡的影响，并有效识别有组织的政治活动。实验结果表明，通过集成所提出的提示工程和平衡检索增强生成方法，我们的框架在精准度、召回率和 F1 分数上超过了传统的基于图的方法，取得了 2 至 3 倍的提升。 

---
# PIKE-RAG: sPecIalized KnowledgE and Rationale Augmented Generation 

**Title (ZH)**: PIKE-RAG：专门知识和推理增强的生成 

**Authors**: Jinyu Wang, Jingjing Fu, Lei Song, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2501.11551)  

**Abstract**: Despite notable advancements in Retrieval-Augmented Generation (RAG) systems that expand large language model (LLM) capabilities through external retrieval, these systems often struggle to meet the complex and diverse needs of real-world industrial applications. The reliance on retrieval alone proves insufficient for extracting deep, domain-specific knowledge performing in logical reasoning from specialized corpora. To address this, we introduce sPecIalized KnowledgE and Rationale Augmentation Generation (PIKE-RAG), focusing on extracting, understanding, and applying specialized knowledge, while constructing coherent rationale to incrementally steer LLMs toward accurate responses. Recognizing the diverse challenges of industrial tasks, we introduce a new paradigm that classifies tasks based on their complexity in knowledge extraction and application, allowing for a systematic evaluation of RAG systems' problem-solving capabilities. This strategic approach offers a roadmap for the phased development and enhancement of RAG systems, tailored to meet the evolving demands of industrial applications. Furthermore, we propose knowledge atomizing and knowledge-aware task decomposition to effectively extract multifaceted knowledge from the data chunks and iteratively construct the rationale based on original query and the accumulated knowledge, respectively, showcasing exceptional performance across various benchmarks. 

**Abstract (ZH)**: 尽管在检索增强生成（RAG）系统方面取得了显著进展，通过外部检索扩展大语言模型（LLM）的能力，但这些系统在满足现实工业应用中复杂多变的需求时往往表现不佳。单纯依赖检索提取深入的专业领域知识以进行逻辑推理的能力仍然不足。为解决这一问题，我们提出了专门知识与推理增强生成（PIKE-RAG），专注于提取、理解和应用专门知识，并构建连贯的推理以逐步引导LLM生成准确的回答。鉴于工业任务的多样性，我们引入了一种新的范式，根据知识提取和应用的复杂性对任务进行分类，从而系统评估RAG系统的解决问题能力。这种策略为分阶段开发和改进RAG系统提供了蓝图，以满足工业应用不断变化的需求。此外，我们提出知识原子化和知识驱动的任务分解，以有效地从数据块中提取多层次的知识，并基于原始查询和累积的知识逐步构建推理，从而在多个基准测试中展示了出色的表现。 

---
# RACCOON: A Retrieval-Augmented Generation Approach for Location Coordinate Capture from News Articles 

**Title (ZH)**: RACCOON：一种用于新闻文章中地理位置坐标抓取的检索增强生成方法 

**Authors**: Jonathan Lin, Aditya Joshi, Hye-young Paik, Tri Dung Doung, Deepti Gurdasani  

**Link**: [PDF](https://arxiv.org/pdf/2501.11440)  

**Abstract**: Geocoding involves automatic extraction of location coordinates of incidents reported in news articles, and can be used for epidemic intelligence or disaster management. This paper introduces Retrieval-Augmented Coordinate Capture Of Online News articles (RACCOON), an open-source geocoding approach that extracts geolocations from news articles. RACCOON uses a retrieval-augmented generation (RAG) approach where candidate locations and associated information are retrieved in the form of context from a location database, and a prompt containing the retrieved context, location mentions and news articles is fed to an LLM to generate the location coordinates. Our evaluation on three datasets, two underlying LLMs, three baselines and several ablation tests based on the components of RACCOON demonstrate the utility of RACCOON. To the best of our knowledge, RACCOON is the first RAG-based approach for geocoding using pre-trained LLMs. 

**Abstract (ZH)**: 地理编码涉及从新闻文章中自动提取事件发生地点的经度和纬度坐标，并可用于疾病监控或灾害管理。本文介绍了“在线新闻文章地理坐标的检索增强捕捉方法”（RACCOON），这是一种开源的地理编码方法，用于从新闻文章中提取地理位置信息。RACCOON采用了检索增强生成（RAG）方法，其中候选地点及其相关信息从地理位置数据库中检索出来，形成上下文，并将包含检索到的上下文、地点提及和新闻文章的提示输入到语言模型（LLM）中生成位置坐标。我们在三个数据集、两种基础语言模型、三种基线方法以及针对RACCOON组件的多个消融测试上对RACCOON进行了评估，显示了其实用性。据我们所知，RACCOON是第一个使用预训练语言模型的基于检索增强生成的地理编码方法。 

---
# Explainable Lane Change Prediction for Near-Crash Scenarios Using Knowledge Graph Embeddings and Retrieval Augmented Generation 

**Title (ZH)**: 使用知识图嵌入和检索增强生成的可解释变道预测方法及其在近碰撞场景中的应用 

**Authors**: M. Manzour, A. Ballardini, R. Izquierdo, M. Á. Sotelo  

**Link**: [PDF](https://arxiv.org/pdf/2501.11560)  

**Abstract**: Lane-changing maneuvers, particularly those executed abruptly or in risky situations, are a significant cause of road traffic accidents. However, current research mainly focuses on predicting safe lane changes. Furthermore, existing accident datasets are often based on images only and lack comprehensive sensory data. In this work, we focus on predicting risky lane changes using the CRASH dataset (our own collected dataset specifically for risky lane changes), and safe lane changes (using the HighD dataset). Then, we leverage KG and Bayesian inference to predict these maneuvers using linguistic contextual information, enhancing the model's interpretability and transparency. The model achieved a 91.5% f1-score with anticipation time extending to four seconds for risky lane changes, and a 90.0% f1-score for predicting safe lane changes with the same anticipation time. We validate our model by integrating it into a vehicle within the CARLA simulator in scenarios that involve risky lane changes. The model managed to anticipate sudden lane changes, thus providing automated vehicles with further time to plan and execute appropriate safe reactions. Finally, to enhance the explainability of our model, we utilize RAG to provide clear and natural language explanations for the given prediction. 

**Abstract (ZH)**: 车道变换行为，尤其是那些在突然或高风险情况下执行的变换，是道路交通事故的重要原因。然而，当前的研究主要集中在预测安全的车道变换。此外，现有的事故数据集通常仅基于图像数据，缺乏全面的感官数据。在本研究中，我们专注于使用CRASH数据集（一个专门为研究高风险车道变换收集的数据集）和HighD数据集来预测高风险和安全的车道变换。我们利用知识图谱（KG）和贝叶斯推理结合语义上下文信息来预测这些行为，以增强模型的可解释性和透明度。该模型在预警时间延长至四秒的情况下，在预测高风险车道变换时达到了91.5%的F1分数，在预测相同预警时间的安全车道变换时达到了90.0%的F1分数。

我们通过将模型集成到CARLA模拟器中的车辆中，在包含高风险车道变换的场景中验证了该模型，成功预见了突然的车道变换，从而为自动驾驶汽车提供了更多时间来规划和执行适当的避险措施。最后，为了进一步提高模型的可解释性，我们利用RAG（Reading-Aware Generation）技术，为给定的预测提供清晰且自然的解释。 

---
# GEC-RAG: Improving Generative Error Correction via Retrieval-Augmented Generation for Automatic Speech Recognition Systems 

**Title (ZH)**: GEC-RAG：通过检索增强生成提高生成式错误修正技术在自动语音识别系统中的性能 

**Authors**: Amin Robatian, Mohammad Hajipour, Mohammad Reza Peyghan, Fatemeh Rajabi, Sajjad Amini, Shahrokh Ghaemmaghami, Iman Gholampour  

**Link**: [PDF](https://arxiv.org/pdf/2501.10734)  

**Abstract**: Automatic Speech Recognition (ASR) systems have demonstrated remarkable performance across various applications. However, limited data and the unique language features of specific domains, such as low-resource languages, significantly degrade their performance and lead to higher Word Error Rates (WER). In this study, we propose Generative Error Correction via Retrieval-Augmented Generation (GEC-RAG), a novel approach designed to improve ASR accuracy for low-resource domains, like Persian. Our approach treats the ASR system as a black-box, a common practice in cloud-based services, and proposes a Retrieval-Augmented Generation (RAG) approach within the In-Context Learning (ICL) scheme to enhance the quality of ASR predictions. By constructing a knowledge base that pairs ASR predictions (1-best and 5-best hypotheses) with their corresponding ground truths, GEC-RAG retrieves lexically similar examples to the ASR transcription using the Term Frequency-Inverse Document Frequency (TF-IDF) measure. This process provides relevant error patterns of the system alongside the ASR transcription to the Generative Large Language Model (LLM), enabling targeted corrections. Our results demonstrate that this strategy significantly reduces WER in Persian and highlights a potential for domain adaptation and low-resource scenarios. This research underscores the effectiveness of using RAG in enhancing ASR systems without requiring direct model modification or fine-tuning, making it adaptable to any domain by simply updating the transcription knowledge base with domain-specific data. 

**Abstract (ZH)**: 自动语音识别（ASR）系统已在多种应用中展示了卓越的性能。然而，有限的数据以及特定领域如低资源语言的独特语言特征，显著降低了其性能，导致更高的词错误率（WER）。在本研究中，我们提出了生成纠错超检索增强生成（GEC-RAG）方法，这是一种旨在提高低资源领域ASR准确性的新方法，例如波斯语。我们的方法将ASR系统视为一个黑盒模型，这是云服务中的常见做法，并在此基础上提出了一种在背景学习（In-Context Learning, ICL）框架内的检索增强生成（RAG）方法，以提高ASR预测质量。通过构建一个知识库，该知识库将ASR预测（1-best和5-best假设）与其对应的地面真实值配对，GEC-RAG使用.Term频率-逆文档频率（TF-IDF）测量检索出与ASR转录词义相似的示例。这个过程向生成大规模语言模型（Generative Large Language Model, GLLM）提供了与ASR转录相关的系统错误模式，使其能够进行目标纠错。我们的结果显示，这种方法显著减少了波斯语的WER，并指出了领域适应性和低资源场景的潜在价值。这项研究强调了在无需直接修改或微调模型的情况下，使用RAG增强ASR系统的有效性，并使其可以通过更新与特定领域相关的转录知识库来适应任何领域。 

---
# 4bit-Quantization in Vector-Embedding for RAG 

**Title (ZH)**: 4位量化在 vectors 嵌入中的应用：面向Retriever-Augmented Generation（检索增强生成）的场景 

**Authors**: Taehee Jeong  

**Link**: [PDF](https://arxiv.org/pdf/2501.10534)  

**Abstract**: Retrieval-augmented generation (RAG) is a promising technique that has shown great potential in addressing some of the limitations of large language models (LLMs). LLMs have two major limitations: they can contain outdated information due to their training data, and they can generate factually inaccurate responses, a phenomenon known as hallucinations. RAG aims to mitigate these issues by leveraging a database of relevant documents, which are stored as embedding vectors in a high-dimensional space. However, one of the challenges of using high-dimensional embeddings is that they require a significant amount of memory to store. This can be a major issue, especially when dealing with large databases of documents. To alleviate this problem, we propose the use of 4-bit quantization to store the embedding vectors. This involves reducing the precision of the vectors from 32-bit floating-point numbers to 4-bit integers, which can significantly reduce the memory requirements. Our approach has several benefits. Firstly, it significantly reduces the memory storage requirements of the high-dimensional vector database, making it more feasible to deploy RAG systems in resource-constrained environments. Secondly, it speeds up the searching process, as the reduced precision of the vectors allows for faster computation. Our code is available at this https URL 

**Abstract (ZH)**: 检索增强生成（RAG）是一种有前景的技术，已在解决大型语言模型（LLMs）的一些局限性方面显示出巨大的潜力。大型语言模型（LLMs）存在两大局限性：由于训练数据可能导致包含过时信息，以及生成事实不准确的响应，这种现象被称为幻觉。RAG 旨在通过利用相关文档数据库来缓解这些问题，这些文档作为嵌入向量存储在高维空间中。然而，使用高维嵌入的一个挑战是需要大量内存来存储它们。特别是在处理大量文档数据库时，这是一个重大问题。为了解决这个问题，我们提议使用4位量化来存储嵌入向量。这涉及到将向量的精度从32位浮点数减少到4位整数，这可以显著减少内存需求。我们的方法具有多个优势。首先，它显著减少了高维向量数据库的内存存储需求，使得在资源受限的环境中部署RAG系统更为可行。其次，它加速了搜索过程，因为向量的精度降低使得计算变快。我们的代码可在以下网址获取：[此处提供网址] 

---
