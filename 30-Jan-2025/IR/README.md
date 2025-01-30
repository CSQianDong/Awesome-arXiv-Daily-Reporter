# WARP: An Efficient Engine for Multi-Vector Retrieval 

**Title (ZH)**: WARP：一种高效的多向量检索引擎 

**Authors**: Jan Luca Scheerer, Matei Zaharia, Christopher Potts, Gustavo Alonso, Omar Khattab  

**Link**: [PDF](https://arxiv.org/pdf/2501.17788)  

**Abstract**: We study the efficiency of multi-vector retrieval methods like ColBERT and its recent variant XTR. We introduce WARP, a retrieval engine that drastically improves the efficiency of XTR-based ColBERT retrievers through three key innovations: (1) WARP$_\text{SELECT}$ for dynamic similarity imputation, (2) implicit decompression to bypass costly vector reconstruction, and (3) a two-stage reduction process for efficient scoring. Combined with optimized C++ kernels and specialized inference runtimes, WARP reduces end-to-end latency by 41x compared to XTR's reference implementation and thereby achieves a 3x speedup over PLAID from the the official ColBERT implementation.
We study the efficiency of multi-vector retrieval methods like ColBERT and its recent variant XTR. We introduce WARP, a retrieval engine that drastically improves the efficiency of XTR-based ColBERT retrievers through three key innovations: (1) WARP$_\text{SELECT}$ for dynamic similarity imputation, (2) implicit decompression during retrieval, and (3) a two-stage reduction process for efficient scoring. Thanks also to highly-optimized C++ kernels and to the adoption of specialized inference runtimes, WARP can reduce end-to-end query latency relative to XTR's reference implementation by 41x. And it thereby achieves a 3x speedup over the official ColBERTv2 PLAID engine, while preserving retrieval quality. 

**Abstract (ZH)**: 我们研究了像ColBERT和其近期变体XTR这样的多向量检索方法的效率。我们引入了WARP，这是一种通过三项关键创新大大提高了基于XTR的ColBERT检索器效率的检索引擎：(1) WARP$_\text{SELECT}$ 动态相似度填充，(2) 隐式解压缩以绕过昂贵的向量重建，以及(3) 一种两级归约过程以实现高效的评分。得益于高度优化的C++内核和专门的推理运行时环境，WARP将端到端查询延迟与XTR的参考实现相比减少了41倍，并因此相对官方的ColBERTv2 PLAID引擎实现了3倍的速度提升，同时保持了检索质量。 

---
# Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation 

**Title (ZH)**: 基于差分序列推荐的卓越量化指导方法 

**Authors**: Wenyu Mao, Shuchang Liu, Haoyang Liu, Haozhe Liu, Xiang Li, Lanatao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.17670)  

**Abstract**: Diffusion models (DMs) have emerged as promising approaches for sequential recommendation due to their strong ability to model data distributions and generate high-quality items. Existing work typically adds noise to the next item and progressively denoises it guided by the user's interaction sequence, generating items that closely align with user interests. However, we identify two key issues in this paradigm. First, the sequences are often heterogeneous in length and content, exhibiting noise due to stochastic user behaviors. Using such sequences as guidance may hinder DMs from accurately understanding user interests. Second, DMs are prone to data bias and tend to generate only the popular items that dominate the training dataset, thus failing to meet the personalized needs of different users. To address these issues, we propose Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation (DiQDiff), which aims to extract robust guidance to understand user interests and generate distinguished items for personalized user interests within DMs. To extract robust guidance, DiQDiff introduces Semantic Vector Quantization (SVQ) to quantize sequences into semantic vectors (e.g., collaborative signals and category interests) using a codebook, which can enrich the guidance to better understand user interests. To generate distinguished items, DiQDiff personalizes the generation through Contrastive Discrepancy Maximization (CDM), which maximizes the distance between denoising trajectories using contrastive loss to prevent biased generation for different users. Extensive experiments are conducted to compare DiQDiff with multiple baseline models across four widely-used datasets. The superior recommendation performance of DiQDiff against leading approaches demonstrates its effectiveness in sequential recommendation tasks. 

**Abstract (ZH)**: 扩散模型（Diffusion Models，DMs）因其强大的数据分布建模能力和生成高质量项目的能力，已成为序列推荐领域的有前途的方法。现有工作通常通过对下一个项目添加噪声，并在用户的交互序列引导下逐步去噪，生成与用户兴趣高度一致的项目。然而，我们发现这种范式存在两个关键问题。首先，序列往往在长度和内容上异质性很大，由于用户行为的随机性而表现出噪声。使用这种序列作为指导可能阻碍DMs准确理解用户兴趣。其次，DMs容易产生数据偏差，倾向于只生成那些在训练数据集中占主导地位的热门项目，从而无法满足不同用户个性化的需求。为解决这些问题，我们提出了基于扩散模型的序列推荐的识别量化指引方法（Distinguished Quantized Guidance for Diffusion-based Sequence Recommendation，DiQDiff），旨在从DMs中提取稳健的指引，以便更好地理解用户兴趣并为个性化用户兴趣生成独具特色的项目。为了提取稳健的指引，DiQDiff引入了语义向量量化（Semantic Vector Quantization，SVQ），通过代码簿将序列量化为语义向量（如协作信号和类别兴趣），这可以丰富指引，更好地理解用户兴趣。为了生成独具特色的项目，DiQDiff通过对比差异最大化（Contrastive Discrepancy Maximization，CDM）进行个性化生成，通过对比损失最大化去噪轨迹之间的距离，以防止不同用户生成有所偏差的项目。通过在四个广泛使用的数据集上与多个基线模型进行广泛的实验比较，DiQDiff的推荐性能优于现有方法，证明了其在序列推荐任务中的有效性。 

---
# Uncertainty Quantification and Decomposition for LLM-based Recommendation 

**Title (ZH)**: 基于大语言模型的推荐系统的不确定性量化与分解 

**Authors**: Wonbin Kweon, Sanghwan Jang, SeongKu Kang, Hwanjo Yu  

**Link**: [PDF](https://arxiv.org/pdf/2501.17630)  

**Abstract**: Despite the widespread adoption of large language models (LLMs) for recommendation, we demonstrate that LLMs often exhibit uncertainty in their recommendations. To ensure the trustworthy use of LLMs in generating recommendations, we emphasize the importance of assessing the reliability of recommendations generated by LLMs. We start by introducing a novel framework for estimating the predictive uncertainty to quantitatively measure the reliability of LLM-based recommendations. We further propose to decompose the predictive uncertainty into recommendation uncertainty and prompt uncertainty, enabling in-depth analyses of the primary source of uncertainty. Through extensive experiments, we (1) demonstrate predictive uncertainty effectively indicates the reliability of LLM-based recommendations, (2) investigate the origins of uncertainty with decomposed uncertainty measures, and (3) propose uncertainty-aware prompting for a lower predictive uncertainty and enhanced recommendation. Our source code and model weights are available at this https URL 

**Abstract (ZH)**: 尽管大规模语言模型（LLMs）在推荐系统中得到了广泛应用，但我们证明了LLMs在推荐方面往往表现出不确定性。为了确保LLMs在生成推荐时的可信使用，我们强调了评估LLM生成推荐可靠性的必要性。我们首先引入了一个新的框架，用于估计预测不确定性，以定量衡量LLM推荐的可靠性。我们进一步提出将预测不确定性分解为推荐不确定性和提示不确定性，从而能够深入了解不确定性的主要来源。通过广泛的实验，我们（1）证明预测不确定性有效地指示了LLM推荐的可靠性，（2）通过分解的不确定性度量来探究不确定性的根源，并（3）提出了具备不确定性感知的提示方法，以降低预测不确定性并提高推荐质量。我们的源代码和模型权重可在以下链接获得：[此处链接] 

---
# Value Function Decomposition in Markov Recommendation Process 

**Title (ZH)**: 马尔可夫推荐过程中的价值函数分解 

**Authors**: Xiaobei Wang, Shuchang Liu, Qingpeng Cai, Xiang Li, Lantao Hu, Han li, Guangming Xie  

**Link**: [PDF](https://arxiv.org/pdf/2501.17409)  

**Abstract**: Recent advances in recommender systems have shown that user-system interaction essentially formulates long-term optimization problems, and online reinforcement learning can be adopted to improve recommendation performance. The general solution framework incorporates a value function that estimates the user's expected cumulative rewards in the future and guides the training of the recommendation policy. To avoid local maxima, the policy may explore potential high-quality actions during inference to increase the chance of finding better future rewards. To accommodate the stepwise recommendation process, one widely adopted approach to learning the value function is learning from the difference between the values of two consecutive states of a user. However, we argue that this paradigm involves an incorrect approximation in the stochastic process. Specifically, between the current state and the next state in each training sample, there exist two separate random factors from the stochastic policy and the uncertain user environment. Original temporal difference (TD) learning under these mixed random factors may result in a suboptimal estimation of the long-term rewards. As a solution, we show that these two factors can be separately approximated by decomposing the original temporal difference loss. The disentangled learning framework can achieve a more accurate estimation with faster learning and improved robustness against action exploration. As empirical verification of our proposed method, we conduct offline experiments with online simulated environments built based on public datasets. 

**Abstract (ZH)**: 近年来，推荐系统的发展表明，用户-系统交互本质上形成了长期优化问题，而在线强化学习能够用于提高推荐性能。该一般解决方案框架包含一个价值函数，该函数估计用户在未来累积奖励的期望，并指导推荐策略的训练。为了避免陷入局部最大值，策略可能在推理过程中探索潜在的高质量行动，以增加找到更好未来奖励的机会。为了适应逐步推荐过程，为学习价值函数广泛采用的一种方法是通过学习用户连续两个状态之间值的差异来进行。然而，我们认为这种方法在随机过程中的推理存在不正确的近似。具体来说，每个训练样本中的当前状态和下一个状态之间存在两个单独的随机因素：来自随机策略的随机性以及不确定的用户环境。在这些混合随机因素下进行原生的临时差异（TD）学习可能会导致长期奖励的次优估计。作为解决方案，我们展示了可以通过将原始的临时差异损失分解来分别近似这两种因素。这种分离学习框架能够实现更准确的估计，并且具有更快的学习速度和更强的抗探索动作的鲁棒性。为了实证验证我们提出的方法，我们在基于公开数据集构建的在线模拟环境中进行了离线实验。 

---
# Aggregation Schemes for Single-Vector WSI Representation Learning in Digital Pathology 

**Title (ZH)**: 数字病理学中单向量WSI表示学习的聚合方案 

**Authors**: Sobhan Hemati, Ghazal Alabtah, Saghir Alfasly, H.R. Tizhoosh  

**Link**: [PDF](https://arxiv.org/pdf/2501.17822)  

**Abstract**: A crucial step to efficiently integrate Whole Slide Images (WSIs) in computational pathology is assigning a single high-quality feature vector, i.e., one embedding, to each WSI. With the existence of many pre-trained deep neural networks and the emergence of foundation models, extracting embeddings for sub-images (i.e., tiles or patches) is straightforward. However, for WSIs, given their high resolution and gigapixel nature, inputting them into existing GPUs as a single image is not feasible. As a result, WSIs are usually split into many patches. Feeding each patch to a pre-trained model, each WSI can then be represented by a set of patches, hence, a set of embeddings. Hence, in such a setup, WSI representation learning reduces to set representation learning where for each WSI we have access to a set of patch embeddings. To obtain a single embedding from a set of patch embeddings for each WSI, multiple set-based learning schemes have been proposed in the literature. In this paper, we evaluate the WSI search performance of multiple recently developed aggregation techniques (mainly set representation learning techniques) including simple average or max pooling operations, Deep Sets, Memory networks, Focal attention, Gaussian Mixture Model (GMM) Fisher Vector, and deep sparse and binary Fisher Vector on four different primary sites including bladder, breast, kidney, and Colon from TCGA. Further, we benchmark the search performance of these methods against the median of minimum distances of patch embeddings, a non-aggregating approach used for WSI retrieval. 

**Abstract (ZH)**: 高效整合Whole Slide Images（WSIs）在计算病理学中的关键步骤之一是为每个WSI分配一个高质量的特征向量，即一个嵌入（embedding）。借助众多预训练的深度神经网络以及基础模型的出现，提取子图像（即瓷砖或补丁）的嵌入变得相对容易。然而，对于具有高分辨率和 gigapixel 特性的 WSIs，将其作为一个完整的图像输入到现有 GPU 中是不可行的。因此，WSIs 通常会被分割成许多补丁。将每个补丁输入到预训练模型中，每个 WSI 可以用一组补丁来表示，因此也表示为一组嵌入。在这种设置中，WSI 表示学习简化为集合表示学习，对于每个 WSI，我们都有权访问一组补丁嵌入。为了从每组补丁嵌入中获得一个单一的嵌入，文献中提出了多种基于集合的学习方案。在本文中，我们评估了多个近期开发的聚合技术（主要是集合表示学习技术）的 WSI 检索性能，包括简单的平均或最大池化操作、Deep Sets、Memory Networks、Focal Attention、Gaussian Mixture Model（GMM）Fisher Vectors 和深度稀疏二元 Fisher Vectors 在膀胱、乳腺、肾脏和结肠（TCGA 数据库四个不同主要部位）中的表现。此外，我们将这些方法的检索性能与使用非聚合方法（即补丁嵌入最小距离的中位数）进行基准测试，该方法用于 WSI 提取。 

---
# Leveraging Multimodal LLM for Inspirational User Interface Search 

**Title (ZH)**: 利用多模态大语言模型进行启发式用户界面搜索 

**Authors**: Seokhyeon Park, Yumin Song, Soohyun Lee, Jaeyoung Kim, Jinwook Seo  

**Link**: [PDF](https://arxiv.org/pdf/2501.17799)  

**Abstract**: Inspirational search, the process of exploring designs to inform and inspire new creative work, is pivotal in mobile user interface (UI) design. However, exploring the vast space of UI references remains a challenge. Existing AI-based UI search methods often miss crucial semantics like target users or the mood of apps. Additionally, these models typically require metadata like view hierarchies, limiting their practical use. We used a multimodal large language model (MLLM) to extract and interpret semantics from mobile UI images. We identified key UI semantics through a formative study and developed a semantic-based UI search system. Through computational and human evaluations, we demonstrate that our approach significantly outperforms existing UI retrieval methods, offering UI designers a more enriched and contextually relevant search experience. We enhance the understanding of mobile UI design semantics and highlight MLLMs' potential in inspirational search, providing a rich dataset of UI semantics for future studies. 

**Abstract (ZH)**: 启发式搜索是指探索设计方案以启发新创造力的过程，在移动用户界面（UI）设计中至关重要。然而，探索大量的UI参考空间仍然是一个挑战。现有的基于AI的UI搜索方法往往忽略了目标用户或应用程序的情绪等关键语义。此外，这些模型通常需要视图层次结构等元数据，限制了其实际应用。我们采用了多模态大型语言模型（MLLM）来从移动UI图像中提取和解释语义。通过形成性研究，我们识别了关键的UI语义，并开发了一个基于语义的UI搜索系统。通过计算和人工评估，我们证明了我们的方法显著优于现有的UI检索方法，为UI设计师提供了更为丰富和相关的搜索体验。我们加深了对移动UI设计语义的理解，并突显了MLLM在启发式搜索中的潜力，提供了丰富的UI语义数据集，以供未来研究使用。 

---
# Cross-Language Approach for Quranic QA 

**Title (ZH)**: 跨语言方法在可兰经问答中的应用 

**Authors**: Islam Oshallah, Mohamed Basem, Ali Hamdi, Ammar Mohammed  

**Link**: [PDF](https://arxiv.org/pdf/2501.17449)  

**Abstract**: Question answering systems face critical limitations in languages with limited resources and scarce data, making the development of robust models especially challenging. The Quranic QA system holds significant importance as it facilitates a deeper understanding of the Quran, a Holy text for over a billion people worldwide. However, these systems face unique challenges, including the linguistic disparity between questions written in Modern Standard Arabic and answers found in Quranic verses written in Classical Arabic, and the small size of existing datasets, which further restricts model performance. To address these challenges, we adopt a cross-language approach by (1) Dataset Augmentation: expanding and enriching the dataset through machine translation to convert Arabic questions into English, paraphrasing questions to create linguistic diversity, and retrieving answers from an English translation of the Quran to align with multilingual training requirements; and (2) Language Model Fine-Tuning: utilizing pre-trained models such as BERT-Medium, RoBERTa-Base, DeBERTa-v3-Base, ELECTRA-Large, Flan-T5, Bloom, and Falcon to address the specific requirements of Quranic QA. Experimental results demonstrate that this cross-language approach significantly improves model performance, with RoBERTa-Base achieving the highest MAP@10 (0.34) and MRR (0.52), while DeBERTa-v3-Base excels in Recall@10 (0.50) and Precision@10 (0.24). These findings underscore the effectiveness of cross-language strategies in overcoming linguistic barriers and advancing Quranic QA systems 

**Abstract (ZH)**: 面向资源有限和数据稀缺的语言的问答系统面临关键限制，这使构建稳健的模型尤为重要具有挑战性。清真文问答系统尤为重要，因为它有助于加深对全世界超过十亿信徒的《古兰经》的理解。然而，这些系统面临着独特挑战，包括现代标准阿拉伯语撰写的问答之间在语言上的差距，以及经文出自古典阿拉伯语的《古兰经》文本，同时现有数据集规模较小，进一步限制了模型性能。为应对这些挑战，我们采用跨语言方法，具体做法如下：(1) 数据集扩充：通过机器翻译扩展和丰富数据集，将阿拉伯语问题转换为英语，对问题进行改写以增加语言多样性，并从英语译本中检索答案以满足多语言训练需求；(2) 语言模型微调：利用预训练的模型，如BERT-Medium、RoBERTa-Base、DeBERTa-v3-Base、ELECTRA-Large、Flan-T5、Bloom和Falcon，针对清真文问答的具体需求进行优化。实验结果表明，这种方法显著提高了模型性能，RoBERTa-Base在MAP@10（0.34）和MRR（0.52）方面表现出最佳表现，而DeBERTa-v3-Base在Recall@10（0.50）和Precision@10（0.24）方面表现出色。这些发现强调了跨语言策略在克服语言障碍和推进清真文问答系统方面的有效性。 

---
# Aspect-Aware Decomposition for Opinion Summarization 

**Title (ZH)**: Opinion 汇总中的方面感知分解 

**Authors**: Miao Li, Jey Han Lau, Eduard Hovy, Mirella Lapata  

**Link**: [PDF](https://arxiv.org/pdf/2501.17191)  

**Abstract**: Opinion summarization plays a key role in deriving meaningful insights from large-scale online reviews. To make this process more explainable and grounded, we propose a modular approach guided by review aspects which separates the tasks of aspect identification, opinion consolidation, and meta-review synthesis, enabling greater transparency and ease of inspection. We conduct extensive experiments across datasets representing scientific research, business, and product domains. Results show that our method generates more grounded summaries compared to strong baseline models, as verified through automated and human evaluations. Additionally, our modular approach, which incorporates reasoning based on review aspects, produces more informative intermediate outputs than knowledge-agnostic decomposed prompting. These intermediate outputs can also effectively support humans in summarizing opinions from large volumes of reviews. 

**Abstract (ZH)**: 意见总结在从大规模在线评论中提取有意义的洞察中起着关键作用。为了使这一过程更具可解释性和现实性，我们提出了一种模块化的方法，该方法依据评论方面来分离方面识别、意见整合和元评论合成的任务，从而增强透明度和检查的便捷性。我们在涵盖科学研究、商业和产品领域数据集的广泛实验中进行了实验。结果表明，与强基准模型相比，我们的方法生成了更具现实性的摘要，并通过自动化和人工评估得到了验证。此外，我们的模块化方法结合了基于评论方面的推理，相较于知识无关的分解提示，生成了更具信息性的中间输出。这些中间输出也可以有效地支持人类从大量评论中总结意见。 

---
# An AI-Driven Live Systematic Reviews in the Brain-Heart Interconnectome: Minimizing Research Waste and Advancing Evidence Synthesis 

**Title (ZH)**: 基于AI驱动的脑-心互联组直播系统性评价：减少研究浪费与促进证据综合 

**Authors**: Arya Rahgozar, Pouria Mortezaagha, Jodi Edwards, Douglas Manuel, Jessie McGowen, Merrick Zwarenstein, Dean Fergusson, Andrea Tricco, Kelly Cobey, Margaret Sampson, Malcolm King, Dawn Richards, Alexandra Bodnaruc, David Moher  

**Link**: [PDF](https://arxiv.org/pdf/2501.17181)  

**Abstract**: The Brain-Heart Interconnectome (BHI) combines neurology and cardiology but is hindered by inefficiencies in evidence synthesis, poor adherence to quality standards, and research waste. To address these challenges, we developed an AI-driven system to enhance systematic reviews in the BHI domain. The system integrates automated detection of Population, Intervention, Comparator, Outcome, and Study design (PICOS), semantic search using vector embeddings, graph-based querying, and topic modeling to identify redundancies and underexplored areas. Core components include a Bi-LSTM model achieving 87% accuracy for PICOS compliance, a study design classifier with 95.7% accuracy, and Retrieval-Augmented Generation (RAG) with GPT-3.5, which outperformed GPT-4 for graph-based and topic-driven queries. The system provides real-time updates, reducing research waste through a living database and offering an interactive interface with dashboards and conversational AI. While initially developed for BHI, the system's adaptable architecture enables its application across various biomedical fields, supporting rigorous evidence synthesis, efficient resource allocation, and informed clinical decision-making. 

**Abstract (ZH)**: 脑-心互联组学（Brain-Heart Interconnectome, BHI）结合了神经学和心脏病学的研究，但受到证据综合效率低下、质量标准执行不力和研究浪费的困扰。为解决这些挑战，我们开发了一个基于人工智能的系统，旨在增强BHI领域的系统性文献回顾。该系统集成了自动检测人群（Population）、干预措施（Intervention）、对照（Comparator）、结果（Outcome）和研究设计（Study Design, PICOS）的功能，利用向量嵌入进行语义搜索，基于图的查询和主题建模，以识别冗余和未充分探索的领域。核心组件包括一个双层长短时记忆网络（Bi-LSTM）模型，其PICOS合规性达到87%的准确率，一个具有95.7%准确率的研究设计分类器，以及与GPT-3.5版本结合的检索增强生成（Retrieval-Augmented Generation, RAG），该系统在基于图和主题驱动的查询方面表现优于GPT-4。该系统提供实时更新，通过活数据库减少了研究浪费，并提供了一个交互式界面，包括仪表板和对话式人工智能。尽管最初是为BHI开发的，但该系统的可适应架构使其能够在各种生物医学领域中应用，支持严格的证据综合、高效的资源分配和基于证据的临床决策。 

---
# Document-Level Sentiment Analysis of Urdu Text Using Deep Learning Techniques 

**Title (ZH)**: 使用深度学习技术的乌尔都语文档级情感分析 

**Authors**: Ammarah Irum, M. Ali Tahir  

**Link**: [PDF](https://arxiv.org/pdf/2501.17175)  

**Abstract**: Document level Urdu Sentiment Analysis (SA) is a challenging Natural Language Processing (NLP) task as it deals with large documents in a resource-poor language. In large documents, there are ample amounts of words that exhibit different viewpoints. Deep learning (DL) models comprise of complex neural network architectures that have the ability to learn diverse features of the data to classify various sentiments. Besides audio, image and video classification; DL algorithms are now extensively used in text-based classification problems. To explore the powerful DL techniques for Urdu SA, we have applied five different DL architectures namely, Bidirectional Long Short Term Memory (BiLSTM), Convolutional Neural Network (CNN), Convolutional Neural Network with Bidirectional Long Short Term Memory (CNN-BiLSTM), Bidirectional Encoder Representation from Transformer (BERT). In this paper, we have proposed a DL hybrid model that integrates BiLSTM with Single Layer Multi Filter Convolutional Neural Network (BiLSTM-SLMFCNN). The proposed and baseline techniques are applied on Urdu Customer Support data set and IMDB Urdu movie review data set by using pretrained Urdu word embeddings that are suitable for (SA) at the document level. Results of these techniques are evaluated and our proposed model outperforms all other DL techniques for Urdu SA. BiLSTM-SLMFCNN outperformed the baseline DL models and achieved 83{\%}, 79{\%}, 83{\%} and 94{\%} accuracy on small, medium and large sized IMDB Urdu movie review data set and Urdu Customer Support data set respectively. 

**Abstract (ZH)**: Urdu 长文档情感分析（SA）是一项具有挑战性的自然语言处理（NLP）任务，因为它涉及资源贫乏的语言中的大量文档。在长文档中，存在着大量表现出不同观点的词汇。深度学习（DL）模型包含复杂的神经网络架构，能够学习多样化的数据特征以分类各种情感。除了音频、图像和视频分类外，DL算法现在也被广泛应用于基于文本的分类问题中。为了探索强大的DL技术在Urdu情感分析中的应用，我们应用了五种不同的DL架构，包括双向长短期记忆网络（BiLSTM）、卷积神经网络（CNN）、带有双向长短期记忆网络的卷积神经网络（CNN-BiLSTM）、来自变换器的双向编码表示（BERT）。在本文中，我们提出了一种DL混合模型，将双向长短期记忆网络与单层多滤波卷积神经网络（BiLSTM-SLMFCNN）相结合。所提出的方法和技术应用于乌尔都语客户支持数据集和IMDB乌尔都语电影评论数据集，并使用预训练的乌尔都语词嵌入，这些词嵌入适合用于文档级的情感分析（SA）。这些技术的结果进行了评估，我们的提出模型在所有其他DL技术中表现最佳。BiLSTM-SLMFCNN 在 IMDB 乌尔都语电影评论数据集和乌尔都语客户支持数据集的小规模、中规模和大规模数据集上的准确率分别为83%，79%，83% 和 94%。 

---
