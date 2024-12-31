# Unsupervised dense retrieval with conterfactual contrastive learning 

**Title (ZH)**: 无监督密集检索的反事实对比学习 

**Authors**: Haitian Chen, Qingyao Ai, Xiao Wang, Yiqun Liu, Fen Lin, Qin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.20756)  

**Abstract**: Efficiently retrieving a concise set of candidates from a large document corpus remains a pivotal challenge in Information Retrieval (IR). Neural retrieval models, particularly dense retrieval models built with transformers and pretrained language models, have been popular due to their superior performance. However, criticisms have also been raised on their lack of explainability and vulnerability to adversarial attacks. In response to these challenges, we propose to improve the robustness of dense retrieval models by enhancing their sensitivity of fine-graned relevance signals. A model achieving sensitivity in this context should exhibit high variances when documents' key passages determining their relevance to queries have been modified, while maintaining low variances for other changes in irrelevant passages. This sensitivity allows a dense retrieval model to produce robust results with respect to attacks that try to promote documents without actually increasing their relevance. It also makes it possible to analyze which part of a document is actually relevant to a query, and thus improve the explainability of the retrieval model. Motivated by causality and counterfactual analysis, we propose a series of counterfactual regularization methods based on game theory and unsupervised learning with counterfactual passages. Experiments show that, our method can extract key passages without reliance on the passage-level relevance annotations. Moreover, the regularized dense retrieval models exhibit heightened robustness against adversarial attacks, surpassing the state-of-the-art anti-attack methods. 

**Abstract (ZH)**: 从大型文档语料库中高效检索一组简洁的候选文档仍然是信息检索（IR）中的关键挑战。基于变压器和预训练语言模型构建的密集检索模型因其卓越的性能而广受欢迎。然而，人们也对其缺乏可解释性和对抗攻击的脆弱性提出了批评。针对这些挑战，我们提出通过增强密集检索模型对细粒度相关性信号的敏感性来提高其鲁棒性。在这种情况下，一个具有高度敏感性的模型在文档的关键段落被修改以决定其与查询的相关性时应表现出高变异性，而在其他无关段落的修改时保持低变异性。这种敏感性使得密集检索模型可以在不实际增加相关性的前提下，对试图促进文档的攻击产生稳健的结果。同时，它也有助于分析文档中哪些部分实际上与查询相关，从而提高检索模型的可解释性。受因果性和反事实分析的启发，我们提出了一种基于博弈论和无监督学习的反事实正则化方法，使用反事实段落。实验结果显示，我们的方法可以在不依赖段落级相关性注释的情况下提取关键段落。此外，正则化的密集检索模型对对抗攻击的鲁棒性也有显著提升，超过了最先进的抗攻击方法。 

---
# AmalREC: A Dataset for Relation Extraction and Classification Leveraging Amalgamation of Large Language Models 

**Title (ZH)**: AmalREC：一种基于大规模语言模型融合的关系提取与分类数据集 

**Authors**: Mansi, Pranshu Pandya, Mahek Bhavesh Vora, Soumya Bharadwaj, Ashish Anand  

**Link**: [PDF](https://arxiv.org/pdf/2412.20427)  

**Abstract**: Existing datasets for relation classification and extraction often exhibit limitations such as restricted relation types and domain-specific biases. This work presents a generic framework to generate well-structured sentences from given tuples with the help of Large Language Models (LLMs). This study has focused on the following major questions: (i) how to generate sentences from relation tuples, (ii) how to compare and rank them, (iii) can we combine strengths of individual methods and amalgamate them to generate an even bette quality of sentences, and (iv) how to evaluate the final dataset? For the first question, we employ a multifaceted 5-stage pipeline approach, leveraging LLMs in conjunction with template-guided generation. We introduce Sentence Evaluation Index(SEI) that prioritizes factors like grammatical correctness, fluency, human-aligned sentiment, accuracy, and complexity to answer the first part of the second question. To answer the second part of the second question, this work introduces a SEI-Ranker module that leverages SEI to select top candidate generations. The top sentences are then strategically amalgamated to produce the final, high-quality sentence. Finally, we evaluate our dataset on LLM-based and SOTA baselines for relation classification. The proposed dataset features 255 relation types, with 15K sentences in the test set and around 150k in the train set organized in, significantly enhancing relational diversity and complexity. This work not only presents a new comprehensive benchmark dataset for RE/RC task, but also compare different LLMs for generation of quality sentences from relational tuples. 

**Abstract (ZH)**: 现有的用于关系分类和提取的数据集往往存在一些限制，如关系类型受限和领域偏见。本文提出了一种通用框架，借助大型语言模型（LLMs）从给定的元组中生成结构良好的句子。本研究主要关注以下问题：(i) 如何从关系元组生成句子，(ii) 如何比较和排名这些句子，(iii) 是否可以结合不同方法的优势，将它们整合以生成质量更高的句子，以及(iv) 如何评估最终数据集？对于第一个问题，我们采用一个多层次的5阶段管道方法，并结合模板指导生成和LLMs。我们介绍了句子评估指数（SEI），该指数优先考虑语法正确性、流畅性、与人类一致的情感、准确性以及复杂性等因素，以回答第二部分的评估问题。为了解决第二部分的评估问题，本研究引入了一个SEI-Ranker模块，该模块利用SEI来选择最佳候选生成。然后，将这些顶级句子战略性地合并，以生成最终高质量的句子。最后，我们在基于LLMs和当前最优基线（SOTA）的模型上对我们的数据集进行了关系分类评估。所提出的数据集包含255种关系类型，测试集中有15,000个句子，训练集中约有150,000个句子，显著增强了关系的多样性和复杂性。本文不仅提供了一个用于关系提取/关系分类任务的新型综合性基准数据集，还比较了不同LLMs在生成关系元组高质量句子方面的表现。 

---
# Introducing Semantic Capability in LinkedIn's Content Search Engine 

**Title (ZH)**: 将LinkedIn的内容搜索引擎中引入语义能力 

**Authors**: Xin Yang, Rachel Zheng, Madhumitha Mohan, Sonali Bhadra, Lingyu Zhang, Rupesh Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2412.20366)  

**Abstract**: In the past, most search queries issued to a search engine were short and simple. A keyword based search engine was able to answer such queries quite well. However, members are now developing the habit of issuing long and complex natural language queries. Answering such queries requires evolution of a search engine to have semantic capability. In this paper we present the design of LinkedIn's new content search engine with semantic capability, and its impact on metrics. 

**Abstract (ZH)**: 过去，提交给搜索引擎的大多数查询都是简短且简单的。基于关键词的搜索引擎能够很好地回答这类查询。然而，用户现在逐渐养成了提交长度较长且内容复杂的自然语言查询的习惯。回答这类查询需要搜索引擎具备语义能力的进化。在本文中，我们介绍了LinkedIn的新内容搜索引擎的设计及其在指标方面的影响。 

---
# Topic-Aware Knowledge Graph with Large Language Models for Interoperability in Recommender Systems 

**Title (ZH)**: 面向主题的知识图谱：通过大型语言模型实现推荐系统中的互操作性 

**Authors**: Minhye Jeon, Seokho Ahn, Young-Duk Seo  

**Link**: [PDF](https://arxiv.org/pdf/2412.20163)  

**Abstract**: The use of knowledge graphs in recommender systems has become one of the common approaches to addressing data sparsity and cold start problems. Recent advances in large language models (LLMs) offer new possibilities for processing side and context information within knowledge graphs. However, consistent integration across various systems remains challenging due to the need for domain expert intervention and differences in system characteristics. To address these issues, we propose a consistent approach that extracts both general and specific topics from both side and context information using LLMs. First, general topics are iteratively extracted and updated from side information. Then, specific topics are extracted using context information. Finally, to address synonymous topics generated during the specific topic extraction process, a refining algorithm processes and resolves these issues effectively. This approach allows general topics to capture broad knowledge across diverse item characteristics, while specific topics emphasize detailed attributes, providing a more comprehensive understanding of the semantic features of items and the preferences of users. Experimental results demonstrate significant improvements in recommendation performance across diverse knowledge graphs. 

**Abstract (ZH)**: 知识图谱在推荐系统中的应用已成为解决数据稀疏性和冷启动问题的一种常见方法。大规模语言模型（LLMs）的最新进展为处理知识图谱内的辅助信息和上下文信息提供了新的可能性。然而，跨各种系统的一致集成仍面临挑战，这主要归因于需要领域专家的干预以及系统特性的差异。为了解决这些问题，我们提出了一种一致的方法，通过LLMs从辅助信息和上下文信息中提取通用和特定主题。首先，从辅助信息中迭代地提取和更新通用主题。然后，使用上下文信息提取特定主题。最后，针对特定主题提取过程中产生的同义主题，采用优化算法有效地处理和解决这些问题。这种方法使通用主题能够捕捉到各类物品特性的广泛知识，而特定主题则强调详细的属性，从而更全面地理解物品的语义特征和用户偏好。实验结果表明，该方法在不同知识图谱中显著提高了推荐性能。 

---
# A Contrastive Pretrain Model with Prompt Tuning for Multi-center Medication Recommendation 

**Title (ZH)**: 带有提示调整的对比预训练模型在多中心药物推荐中的应用 

**Authors**: Qidong Liu, Zhaopeng Qiu, Xiangyu Zhao, Xian Wu, Zijian Zhang, Tong Xu, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2412.20040)  

**Abstract**: Medication recommendation is one of the most critical health-related applications, which has attracted extensive research interest recently. Most existing works focus on a single hospital with abundant medical data. However, many small hospitals only have a few records, which hinders applying existing medication recommendation works to the real world. Thus, we seek to explore a more practical setting, i.e., multi-center medication recommendation. In this setting, most hospitals have few records, but the total number of records is large. Though small hospitals may benefit from total affluent records, it is also faced with the challenge that the data distributions between various hospitals are much different. In this work, we introduce a novel conTrastive prEtrain Model with Prompt Tuning (TEMPT) for multi-center medication recommendation, which includes two stages of pretraining and finetuning. We first design two self-supervised tasks for the pretraining stage to learn general medical knowledge. They are mask prediction and contrastive tasks, which extract the intra- and inter-relationships of input diagnosis and procedures. Furthermore, we devise a novel prompt tuning method to capture the specific information of each hospital rather than adopting the common finetuning. On the one hand, the proposed prompt tuning can better learn the heterogeneity of each hospital to fit various distributions. On the other hand, it can also relieve the catastrophic forgetting problem of finetuning. To validate the proposed model, we conduct extensive experiments on the public eICU, a multi-center medical dataset. The experimental results illustrate the effectiveness of our model. The implementation code is available to ease the reproducibility this https URL. 

**Abstract (ZH)**: 药物推荐是与健康相关的重要应用之一，近年来引起了广泛的研究兴趣。目前大多数现有工作集中在单一医院，拥有丰富的医疗数据。然而，许多小型医院仅有少量记录，这阻碍了现有药物推荐工作的应用。因此，我们寻求探索一个更为实际的设置，即跨中心药物推荐。在这种设置中，大多数医院的记录数量较少，但总记录数量较大。虽然小型医院可以从丰富的总记录中受益，但它们也面临着不同医院之间数据分布差异较大的挑战。在本文中，我们介绍了一种新颖的对抗预训练模型（TEtempt）与提示调优方法，该方法包括预训练和微调两个阶段。我们首先在预训练阶段设计了两个自监督任务，以学习通用医疗知识。它们是遮蔽预测和对抗任务，分别提取输入诊断和程序之间的内部关系和外部关系。此外，我们设计了一种新颖的提示调优方法，以捕捉每个医院的特定信息，而不是采用常见的微调方法。一方面，所提出的提示调优方法可以更好地学习每个医院的异质性，以适应不同的分布；另一方面，它还可以缓解微调中的灾难性遗忘问题。为了验证所提出模型的有效性，我们在公开的eICU数据集上进行了广泛的实验，这是一个跨中心的医疗数据集。实验结果证明了我们模型的有效性。我们已开源实现代码，以简化重复实验的实现。更多细节请参见此链接：[https://example.com/implementation_code]。 

---
# Invariant debiasing learning for recommendation via biased imputation 

**Title (ZH)**: 基于偏差填补的不变偏差消除推荐学习 

**Authors**: Ting Bai, Weijie Chen, Cheng Yang, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2412.20036)  

**Abstract**: Previous debiasing studies utilize unbiased data to make supervision of model training. They suffer from the high trial risks and experimental costs to obtain unbiased data. Recent research attempts to use invariant learning to detach the invariant preference of users for unbiased recommendations in an unsupervised way. However, it faces the drawbacks of low model accuracy and unstable prediction performance due to the losing cooperation with variant preference. In this paper, we experimentally demonstrate that invariant learning causes information loss by directly discarding the variant information, which reduces the generalization ability and results in the degradation of model performance in unbiased recommendations. Based on this consideration, we propose a novel lightweight knowledge distillation framework (KDDebias) to automatically learn the unbiased preference of users from both invariant and variant information. Specifically, the variant information is imputed to the invariant user preference in the distance-aware knowledge distillation process. Extensive experiments on three public datasets, i.e., Yahoo!R3, Coat, and MIND, show that with the biased imputation from the variant preference of users, our proposed method achieves significant improvements with less than 50% learning parameters compared to the SOTA unsupervised debiasing model in recommender systems. Our code are publicly available at this https URL. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的中文内容：

先前的去偏研究利用未偏数据进行模型训练的监督。然而，获取未偏数据面临着高试验风险和实验成本问题。最近的研究试图通过不变学习来提取用户的不变偏好，以无监督的方式实现未偏推荐。然而，这种方法由于舍弃了可变偏好而表现出较低的模型准确性和不稳定预测性能。在本文中，我们实验性地展示了直接舍弃可变信息导致信息丢失，这降低了模型的泛化能力，并导致未偏推荐中的模型性能退化。基于这一考虑，我们提出了一种新的轻量化知识蒸馏框架（KDDebias），能够自动从不变信息和可变信息中学习用户的未偏偏好。具体而言，在距离感知的知识蒸馏过程中，将可变信息补全到不变用户偏好中。我们在三个公开数据集，即Yahoo!R3、Coat和MIND上进行的广泛实验表明，通过用户可变偏好中的偏置补全，所提出的方法在与最优无监督去偏模型相当的学习参数量（少于50%）下，实现了显著的性能改进。我们的代码已在此链接公开：[公开代码链接]。 

---
# Towards Identity-Aware Cross-Modal Retrieval: a Dataset and a Baseline 

**Title (ZH)**: 面向身份意识的跨模态检索：一个数据集和基准模型 

**Authors**: Nicola Messina, Lucia Vadicamo, Leo Maltese, Claudio Gennaro  

**Link**: [PDF](https://arxiv.org/pdf/2412.21009)  

**Abstract**: Recent advancements in deep learning have significantly enhanced content-based retrieval methods, notably through models like CLIP that map images and texts into a shared embedding space. However, these methods often struggle with domain-specific entities and long-tail concepts absent from their training data, particularly in identifying specific individuals. In this paper, we explore the task of identity-aware cross-modal retrieval, which aims to retrieve images of persons in specific contexts based on natural language queries. This task is critical in various scenarios, such as for searching and browsing personalized video collections or large audio-visual archives maintained by national broadcasters. We introduce a novel dataset, COCO Person FaceSwap (COCO-PFS), derived from the widely used COCO dataset and enriched with deepfake-generated faces from VGGFace2. This dataset addresses the lack of large-scale datasets needed for training and evaluating models for this task. Our experiments assess the performance of different CLIP variations repurposed for this task, including our architecture, Identity-aware CLIP (Id-CLIP), which achieves competitive retrieval performance through targeted fine-tuning. Our contributions lay the groundwork for more robust cross-modal retrieval systems capable of recognizing long-tail identities and contextual nuances. Data and code are available at this https URL. 

**Abstract (ZH)**: 近年来，深度学习的进步显著提升了基于内容的检索方法，特别是通过CLIP等模型将图像和文本映射到共享嵌入空间中。然而，这些方法在处理训练数据中缺乏的专业领域实体和长尾概念时往往表现不佳，特别是在识别特定个体方面。本文探讨了身份感知多模态检索的任务，该任务旨在根据自然语言查询基于特定上下文检索特定个体的图像。这一任务在多种场景中至关重要，例如在搜索和浏览个性化视频集合或由国家级广播机构维护的大规模音频-视觉档案时。我们引入了一个全新的数据集，COCO Person FaceSwap（COCO-PFS），该数据集基于广为使用的COCO数据集，并通过VGGFace2生成的深度伪造面孔进行丰富。这一数据集解决了训练和评估此类任务所需的大型数据集缺乏的问题。我们的实验评估了不同CLIP变体在这项任务中的性能，包括我们的架构，身份感知CLIP（Id-CLIP），该架构通过针对性的微调实现了竞争力的检索性能。我们的贡献为构建更加稳健的多模态检索系统奠定了基础，这些系统能够识别长尾身份和上下文微 辽。数据和代码可在以下链接获取：[这里提供链接]。 

---
# Rise of Generative Artificial Intelligence in Science 

**Title (ZH)**: 生成式人工智能在科学中的崛起 

**Authors**: Liangping Ding, Cornelia Lawson, Philip Shapira  

**Link**: [PDF](https://arxiv.org/pdf/2412.20960)  

**Abstract**: Generative Artificial Intelligence (GenAI, generative AI) has rapidly become available as a tool in scientific research. To explore the use of generative AI in science, we conduct an empirical analysis using OpenAlex. Analyzing GenAI publications and other AI publications from 2017 to 2023, we profile growth patterns, the diffusion of GenAI publications across fields of study, and the geographical spread of scientific research on generative AI. We also investigate team size and international collaborations to explore whether GenAI, as an emerging scientific research area, shows different collaboration patterns compared to other AI technologies. The results indicate that generative AI has experienced rapid growth and increasing presence in scientific publications. The use of GenAI now extends beyond computer science to other scientific research domains. Over the study period, U.S. researchers contributed nearly two-fifths of global GenAI publications. The U.S. is followed by China, with several small and medium-sized advanced economies demonstrating relatively high levels of GenAI deployment in their research publications. Although scientific research overall is becoming increasingly specialized and collaborative, our results suggest that GenAI research groups tend to have slightly smaller team sizes than found in other AI fields. Furthermore, notwithstanding recent geopolitical tensions, GenAI research continues to exhibit levels of international collaboration comparable to other AI technologies. 

**Abstract (ZH)**: 生成式人工智能（GenAI，生成人工智能）已迅速成为科学研究中的一种工具。为了探索生成式人工智能在科学中的应用，我们利用OpenAlex进行实证分析。分析2017年至2023年间发布生成式人工智能论文及其他人工智能论文，我们概要了其增长模式、生成式人工智能论文在不同研究领域的扩散情况以及科学界关于生成式人工智能的研究地理分布。我们还调查了团队规模和国际合作，探讨生成式人工智能作为新兴科学研究领域，其合作模式是否与其他人工智能技术有所不同。结果表明，生成式人工智能在科学研究中的使用经历了快速扩张，并且其存在感在科学文献中的逐年增加。生成式人工智能的应用现已超出计算机科学领域，扩展到了其他科学研究领域。研究期间，美国研究人员贡献了全球生成式人工智能论文的近四分之三。美国之后是中国，一些中等规模的先进经济体在其研究论文中也展示了较高的生成式人工智能部署水平。尽管整体科学界的研究变得越来越专业化和合作化，我们的结果表明，生成式人工智能研究团队的规模略小于其他人工智能领域。此外，尽管近期存在地缘政治紧张局势，生成式人工智能研究在国际合作水平上仍与其它人工智能技术相当。 

---
# Ontology-grounded Automatic Knowledge Graph Construction by LLM under Wikidata schema 

**Title (ZH)**: 基于Wikidata模式的本体导向自动知识图谱构建方法（由大语言模型实现） 

**Authors**: Xiaohan Feng, Xixin Wu, Helen Meng  

**Link**: [PDF](https://arxiv.org/pdf/2412.20942)  

**Abstract**: We propose an ontology-grounded approach to Knowledge Graph (KG) construction using Large Language Models (LLMs) on a knowledge base. An ontology is authored by generating Competency Questions (CQ) on knowledge base to discover knowledge scope, extracting relations from CQs, and attempt to replace equivalent relations by their counterpart in Wikidata. To ensure consistency and interpretability in the resulting KG, we ground generation of KG with the authored ontology based on extracted relations. Evaluation on benchmark datasets demonstrates competitive performance in knowledge graph construction task. Our work presents a promising direction for scalable KG construction pipeline with minimal human intervention, that yields high quality and human-interpretable KGs, which are interoperable with Wikidata semantics for potential knowledge base expansion. 

**Abstract (ZH)**: 我们提出一种基于本体的方法，利用大型语言模型（LLMs）在知识库上构建知识图谱（KG）。该方法包括生成 Competency Questions（CQ），以发现知识范围、从 CQ 中抽取关系，并尝试用维基数据中的对应关系替换等价关系。为了确保结果 KG 的一致性和可解释性，我们基于抽取到的关系使用所生成的本体对 KG 的生成进行约束。在基准数据集上的评估显示，我们的方法在知识图谱构建任务中表现出竞争力。我们的工作为通过最小的人工干预实现可扩展的 KG 构建管道提供了一个有前景的方向，能够生成高质量且可由人类解释的 KG，并且这些 KG 可与 Wikidata 的语义兼容，以促进知识库的扩展。 

---
# Comparative Performance of Advanced NLP Models and LLMs in Multilingual Geo-Entity Detection 

**Title (ZH)**: 多语言地理实体检测中高级NLP模型和LLMs的性能比较 

**Authors**: Kalin Kopanov  

**Link**: [PDF](https://arxiv.org/pdf/2412.20414)  

**Abstract**: The integration of advanced Natural Language Processing (NLP) methodologies and Large Language Models (LLMs) has significantly enhanced the extraction and analysis of geospatial data from multilingual texts, impacting sectors such as national and international security. This paper presents a comprehensive evaluation of leading NLP models -- SpaCy, XLM-RoBERTa, mLUKE, GeoLM -- and LLMs, specifically OpenAI's GPT 3.5 and GPT 4, within the context of multilingual geo-entity detection. Utilizing datasets from Telegram channels in English, Russian, and Arabic, we examine the performance of these models through metrics such as accuracy, precision, recall, and F1 scores, to assess their effectiveness in accurately identifying geospatial references. The analysis exposes each model's distinct advantages and challenges, underscoring the complexities involved in achieving precise geo-entity identification across varied linguistic landscapes. The conclusions drawn from this experiment aim to direct the enhancement and creation of more advanced and inclusive NLP tools, thus advancing the field of geospatial analysis and its application to global security. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

先进自然语言处理（NLP）方法与大规模语言模型（LLMs）的集成显著提升了从多语言文本中提取和分析地理空间数据的能力，影响了国家和国际安全等领域。本文在多语言地理实体检测的背景下，对该领域的领先NLP模型（包括SpaCy、XLM-RoBERTa、mLUKE和GeoLM）以及LLMs（尤其是OpenAI的GPT 3.5和GPT 4）进行了全面评估。利用来自Telegram的英语、俄语和阿拉伯语数据集，我们通过准确率、精度、召回率和F1分数等指标，考察了这些模型在准确识别地理空间引用方面的性能。分析揭示了每种模型的独特优势和挑战，突显了在不同语言环境中实现精确地理实体识别的复杂性。实验得出的结论旨在指导更先进和包容性NLP工具的发展与创新，从而推动地理空间分析及其在全球安全领域的应用。

这样翻译可以确保内容准确、语言规范，同时也符合学术写作的要求。 

---
# Left-handed representation in top 100 male professional tennis players: Multi-disciplinary perspectives 

**Title (ZH)**: 位居前100名的职业男 tennis 球员中的左利手表现：跨学科视角 

**Authors**: Boris Bačić, Ali Ghazala  

**Link**: [PDF](https://arxiv.org/pdf/2412.20360)  

**Abstract**: A commonly held opinion is that left-handed tennis players are overrepresented compared to the percentage of left-handers within the general population. This study provides the domain insights supported by data analysis that could help inform the decision of parents and coaches considering whether a child should start playing tennis as left- or right-handed when there is no strong arm-handed dominance. Compared to the commonly cited figure of about 10% of left-handed male population, data analysis from the official ATP web site for the top 100 ranked tennis players over the past decades (1985-2016) shows evidence of overrepresentation of left-handed elite tennis players (about 15%). The insights and data analysis can inform the handedness decision, advance coaching and strategic game concepts, enhance media coverage/analytics, left-handed facts and statistics, and inform tennis equipment manufacturing. 

**Abstract (ZH)**: 一种常见的观点认为，左撇子网球运动员在顶尖选手中的比例高于普通人群左撇子的比例。本研究通过数据分析提供了相关领域的见解，这些见解可以帮助家长和教练在孩子是否应该从小开始以左撇子或右撇子的身份打网球时进行决策，尤其是在没有明确的手部主导性的情况下。与常被引用的男性左撇子人口比例约为10%相比，过去几十年（1985-2016年）ATP官方网站上排名前100位的网球选手的数据分析表明，顶尖网球选手中左撇子的比例（约15%）确实存在过量现象。这些洞察和数据分析可以为选择手部主导性提供信息，促进教练理念和战略战术的发展，增强媒体报道和数据分析，丰富左撇子的相关事实和统计数据，以及指导网球相关装备的制造。 

---
# OneKE: A Dockerized Schema-Guided LLM Agent-based Knowledge Extraction System 

**Title (ZH)**: OneKE：一种基于模式引导的LLM代理知识提取系统（Docker化实现） 

**Authors**: Yujie Luo, Xiangyuan Ru, Kangwei Liu, Lin Yuan, Mengshu Sun, Ningyu Zhang, Lei Liang, Zhiqiang Zhang, Jun Zhou, Lanning Wei, Da Zheng, Haofen Wang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.20005)  

**Abstract**: We introduce OneKE, a dockerized schema-guided knowledge extraction system, which can extract knowledge from the Web and raw PDF Books, and support various domains (science, news, etc.). Specifically, we design OneKE with multiple agents and a configure knowledge base. Different agents perform their respective roles, enabling support for various extraction scenarios. The configure knowledge base facilitates schema configuration, error case debugging and correction, further improving the performance. Empirical evaluations on benchmark datasets demonstrate OneKE's efficacy, while case studies further elucidate its adaptability to diverse tasks across multiple domains, highlighting its potential for broad applications. We have open-sourced the Code at this https URL and released a Video at this http URL. 

**Abstract (ZH)**: 我们介绍了OneKE，这是一个容器化的基于模式的知识提取系统，能够从互联网和原始PDF书籍中提取知识，并支持各种领域（如科学、新闻等）。具体而言，我们设计了OneKE，采用了多个代理和一个可配置的知识库。不同的代理各自执行特定任务，从而支持多种提取场景。可配置的知识库便于模式配置、错误案例调试和修正，进一步提高系统的性能。基准数据集上的实证评估显示了OneKE的有效性，而案例研究进一步说明了它在多个领域的多种任务中的适应性，突显了其在广泛应用场景中的潜力。我们已将源代码开源并在以下链接发布了视频：[此链接]和[此链接]。 

---
# ERPA: Efficient RPA Model Integrating OCR and LLMs for Intelligent Document Processing 

**Title (ZH)**: ERPA：结合OCR和LLMs的高效RPA模型以实现智能文档处理 

**Authors**: Osama Abdellaif, Abdelrahman Nader, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2412.19840)  

**Abstract**: This paper presents ERPA, an innovative Robotic Process Automation (RPA) model designed to enhance ID data extraction and optimize Optical Character Recognition (OCR) tasks within immigration workflows. Traditional RPA solutions often face performance limitations when processing large volumes of documents, leading to inefficiencies. ERPA addresses these challenges by incorporating Large Language Models (LLMs) to improve the accuracy and clarity of extracted text, effectively handling ambiguous characters and complex structures. Benchmark comparisons with leading platforms like UiPath and Automation Anywhere demonstrate that ERPA significantly reduces processing times by up to 94 percent, completing ID data extraction in just 9.94 seconds. These findings highlight ERPA's potential to revolutionize document automation, offering a faster and more reliable alternative to current RPA solutions. 

**Abstract (ZH)**: 本文介绍了ERPA，这是一种创新的机器人流程自动化（RPA）模型，旨在增强身份数据提取并优化移民工作流程中的光学字符识别（OCR）任务。传统的RPA解决方案在处理大量文档时常常面临性能限制，导致效率低下。ERPA通过集成大型语言模型（LLMs），提高了提取文本的准确性和清晰度，有效地处理了模糊字符和复杂结构。与UiPath和Automation Anywhere等领先平台的基准比较显示，ERPA将处理时间显著降低了高达94%，仅需9.94秒即可完成身份数据提取。这些发现突显了ERPA在文档自动化方面的潜在革命性影响，提供了一种比当前RPA解决方案更快且更可靠的选择。 

---
