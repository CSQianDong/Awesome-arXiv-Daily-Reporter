# Mitigating Bias in RAG: Controlling the Embedder 

**Title (ZH)**: 缓解RAG中的偏差：控制嵌入器 

**Authors**: Taeyoun Kim, Jacob Springer, Aditi Raghunathan, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2502.17390)  

**Abstract**: In retrieval augmented generation (RAG) systems, each individual component -- the LLM, embedder, and corpus -- could introduce biases in the form of skews towards outputting certain perspectives or identities. In this work, we study the conflict between biases of each component and their relationship to the overall bias of the RAG system, which we call bias conflict. Examining both gender and political biases as case studies, we show that bias conflict can be characterized through a linear relationship among components despite its complexity in 6 different LLMs. Through comprehensive fine-tuning experiments creating 120 differently biased embedders, we demonstrate how to control bias while maintaining utility and reveal the importance of reverse-biasing the embedder to mitigate bias in the overall system. Additionally, we find that LLMs and tasks exhibit varying sensitivities to the embedder bias, a crucial factor to consider for debiasing. Our results underscore that a fair RAG system can be better achieved by carefully controlling the bias of the embedder rather than increasing its fairness. 

**Abstract (ZH)**: 在检索增强生成（RAG）系统中，每一部分构件——大语言模型（LLM）、嵌入器和语料库——都有可能以偏见的形式偏向输出某些特定视角或身份。在本研究中，我们探讨了每个构件偏见与其对RAG系统整体偏见关系之间的冲突，称之为偏见冲突。通过性别和政治偏见作为案例研究，我们展示了尽管在6种不同的LLM中复杂性不同，但可以通过这些构件之间的线性关系来表征偏见冲突。通过全面的微调实验创建了120种不同程度的偏见嵌入器，并展示了如何在保持实用性的前提下控制偏见，揭示了逆向偏置嵌入器以减轻整体系统偏见的重要性。此外，我们发现大语言模型和任务对嵌入器偏见的敏感性存在差异，这是去偏过程中需要考虑的关键因素。我们的研究表明，通过仔细控制嵌入器的偏见而非单纯提高其公平性，可以更好地实现公平的RAG系统。 

---
# MEMERAG: A Multilingual End-to-End Meta-Evaluation Benchmark for Retrieval Augmented Generation 

**Title (ZH)**: MEMERAG：一种多语言端到端元评估基准，用于检索增强生成 

**Authors**: María Andrea Cruz Blandón, Jayasimha Talur, Bruno Charron, Dong Liu, Saab Mansour, Marcello Federico  

**Link**: [PDF](https://arxiv.org/pdf/2502.17163)  

**Abstract**: Automatic evaluation of retrieval augmented generation (RAG) systems relies on fine-grained dimensions like faithfulness and relevance, as judged by expert human annotators. Meta-evaluation benchmarks support the development of automatic evaluators that correlate well with human judgement. However, existing benchmarks predominantly focus on English or use translated data, which fails to capture cultural nuances. A native approach provides a better representation of the end user experience.
In this work, we develop a Multilingual End-to-end Meta-Evaluation RAG benchmark (MEMERAG). Our benchmark builds on the popular MIRACL dataset, using native-language questions and generating responses with diverse large language models (LLMs), which are then assessed by expert annotators for faithfulness and relevance. We describe our annotation process and show that it achieves high inter-annotator agreement. We then analyse the performance of the answer-generating LLMs across languages as per the human evaluators. Finally we apply the dataset to our main use-case which is to benchmark multilingual automatic evaluators (LLM-as-a-judge). We show that our benchmark can reliably identify improvements offered by advanced prompting techniques and LLMs. We release our benchmark to support the community developing accurate evaluation methods for multilingual RAG systems. 

**Abstract (ZH)**: 自动评估检索增强生成（RAG）系统依赖于专家人工注释者评判的精细维度，如忠实度和相关性。元评估基准支持开发与人类判断高度相关的自动评估器。然而，现有的基准主要集中在英语上或使用翻译数据，这未能捕捉到文化细微差别。本地化的方法为最终用户体验提供了更好的代表。

在本文中，我们开发了一个多语言端到端元评估RAG基准（MEMERAG）。该基准基于流行的MIRACL数据集，使用本地语言问题，生成多种大型语言模型（LLM）的响应，然后由专家注释者对忠实度和相关性进行评估。我们描述了我们的注释过程，并展示了该过程在注释者之间达到了高一致性。随后，我们分析了回答生成的LLM在不同语言上的表现，根据人类评估者的评判。最后，我们将数据集应用于我们的主要用途案例，即基准测试多语言自动评估器（LLM作为裁判）。我们展示了我们的基准可以可靠地识别由高级提示技术和LLM提供的改进。我们发布了该基准，以支持开发准确评估方法的多语言RAG系统社区。 

---
# A Hybrid Approach to Information Retrieval and Answer Generation for Regulatory Texts 

**Title (ZH)**: 一种综合方法用于监管文本的信息检索与答案生成 

**Authors**: Jhon Rayo, Raul de la Rosa, Mario Garrido  

**Link**: [PDF](https://arxiv.org/pdf/2502.16767)  

**Abstract**: Regulatory texts are inherently long and complex, presenting significant challenges for information retrieval systems in supporting regulatory officers with compliance tasks. This paper introduces a hybrid information retrieval system that combines lexical and semantic search techniques to extract relevant information from large regulatory corpora. The system integrates a fine-tuned sentence transformer model with the traditional BM25 algorithm to achieve both semantic precision and lexical coverage. To generate accurate and comprehensive responses, retrieved passages are synthesized using Large Language Models (LLMs) within a Retrieval Augmented Generation (RAG) framework. Experimental results demonstrate that the hybrid system significantly outperforms standalone lexical and semantic approaches, with notable improvements in Recall@10 and MAP@10. By openly sharing our fine-tuned model and methodology, we aim to advance the development of robust natural language processing tools for compliance-driven applications in regulatory domains. 

**Abstract (ZH)**: 监管文本天生具有较长和复杂的特点，这为信息检索系统在支持监管官员合规任务时带来了巨大的挑战。本文介绍了一种结合词法和语义搜索技术的混合信息检索系统，该系统可以从大型监管档案库中提取相关的信息。该系统将微调的句子变换模型与传统的BM25算法相结合，以实现语义精度和词法覆盖的双重目标。为了生成准确和全面的响应，检索到的段落通过检索增强生成（RAG）框架内的大型语言模型（LLMs）进行综合处理。实验结果表明，该混合系统显著优于单独的词法和语义方法，在Recall@10和MAP@10方面有显著改进。通过公开分享我们的微调模型和方法，我们旨在推动监管领域基于合规的应用中稳健自然语言处理工具的发展。 

---
# Visual-RAG: Benchmarking Text-to-Image Retrieval Augmented Generation for Visual Knowledge Intensive Queries 

**Title (ZH)**: 视觉-RAG：增强生成文本到图像检索的视觉知识密集型查询基准测试 

**Authors**: Yin Wu, Quanyu Long, Jing Li, Jianfei Yu, Wenya Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16636)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a popular approach for enhancing Large Language Models (LLMs) by addressing their limitations in verifying facts and answering knowledge-intensive questions. As the research in LLM extends their capability to handle input modality other than text, e.g. image, several multimodal RAG benchmarks are proposed. Nonetheless, they mainly use textual knowledge bases as the primary source of evidences for augmentation. There still lack benchmarks designed to evaluate images as augmentation in RAG systems and how they leverage visual knowledge. We propose Visual-RAG, a novel Question Answering benchmark that emphasizes visual knowledge intensive questions. Unlike prior works relying on text-based evidence, Visual-RAG necessitates text-to-image retrieval and integration of relevant clue images to extract visual knowledge as evidence. With Visual-RAG, we evaluate 5 open-sourced and 3 proprietary Multimodal LLMs (MLLMs), revealing that images can serve as good evidence in RAG; however, even the SoTA models struggle with effectively extracting and utilizing visual knowledge 

**Abstract (ZH)**: 检索增强生成（RAG）是一种通过解决大型语言模型（LLMs）在事实验证和回答知识密集型问题方面的局限性来增强LLMs的流行方法。随着LLMs研究扩展其处理输入模态的能力，例如图像，已经提出了几种多模态RAG基准。然而，它们主要依赖于文本知识库作为增强的主要证据来源。尚未设计出专门评估图像作为RAG系统增强手段以及如何利用视觉知识的基准。我们提出了Visual-RAG，这是一种新的问答基准，侧重于视觉知识密集型的问题。与以往依赖于文本证据的工作不同，Visual-RAG 要求进行文本到图像的检索和将相关线索图像集成起来以提取视觉知识作为证据。通过Visual-RAG，我们评估了5个开源和3个专有的多模态大型语言模型（MLLMs），结果显示图像可以作为RAG的良好证据；然而，即使是当前最好的模型，在有效提取和利用视觉知识方面也存在困难。 

---
# MMRAG: Multi-Mode Retrieval-Augmented Generation with Large Language Models for Biomedical In-Context Learning 

**Title (ZH)**: MMRAG：基于大型语言模型的多模式检索增强生成方法在生物医学领域内的上下文学习 

**Authors**: Zaifu Zhan, Jun Wang, Shuang Zhou, Jiawen Deng, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.15954)  

**Abstract**: Objective: To optimize in-context learning in biomedical natural language processing by improving example selection. Methods: We introduce a novel multi-mode retrieval-augmented generation (MMRAG) framework, which integrates four retrieval strategies: (1) Random Mode, selecting examples arbitrarily; (2) Top Mode, retrieving the most relevant examples based on similarity; (3) Diversity Mode, ensuring variation in selected examples; and (4) Class Mode, selecting category-representative examples. This study evaluates MMRAG on three core biomedical NLP tasks: Named Entity Recognition (NER), Relation Extraction (RE), and Text Classification (TC). The datasets used include BC2GM for gene and protein mention recognition (NER), DDI for drug-drug interaction extraction (RE), GIT for general biomedical information extraction (RE), and HealthAdvice for health-related text classification (TC). The framework is tested with two large language models (Llama2-7B, Llama3-8B) and three retrievers (Contriever, MedCPT, BGE-Large) to assess performance across different retrieval strategies. Results: The results from the Random mode indicate that providing more examples in the prompt improves the model's generation performance. Meanwhile, Top mode and Diversity mode significantly outperform Random mode on the RE (DDI) task, achieving an F1 score of 0.9669, a 26.4% improvement. Among the three retrievers tested, Contriever outperformed the other two in a greater number of experiments. Additionally, Llama 2 and Llama 3 demonstrated varying capabilities across different tasks, with Llama 3 showing a clear advantage in handling NER tasks. Conclusion: MMRAG effectively enhances biomedical in-context learning by refining example selection, mitigating data scarcity issues, and demonstrating superior adaptability for NLP-driven healthcare applications. 

**Abstract (ZH)**: 目的：通过改进示例选择来优化生物医学自然语言处理中的上下文学习。方法：我们提出了一种新型的多模式检索增强生成（MMRAG）框架，该框架结合了四种检索策略：（1）随机模式，随机选择示例；（2）顶级模式，根据相似性检索最相关示例；（3）多样性模式，确保所选示例的多样性；（4）类别模式，选择代表性示例。本研究将MMRAG应用于三个核心的生物医学自然语言处理任务：命名实体识别（NER）、关系提取（RE）和文本分类（TC）。使用的数据集包括：BC2GM用于基因和蛋白质提及识别（NER）、DDI用于药物-药物相互作用提取（RE）、GIT用于一般生物医学信息提取（RE）、HealthAdvice用于与健康有关的文本分类（TC）。该框架使用两个大型语言模型（Llama2-7B，Llama3-8B）和三种检索器（Contriever，MedCPT，BGE-Large）进行测试，以评估不同检索策略下的性能。结果：随机模式的结果表明，在提示中提供更多的示例可以提高模型的生成性能。同时，顶级模式和多样性模式在RE（DDI）任务上显著优于随机模式，取得了0.9669的F1分数，提高了26.4%。在三个测试的检索器中，Contriever在大多数实验中表现优于其他两个检索器。此外，Llama 2和Llama 3在不同任务中显示出了不同的能力，Llama 3在处理NER任务时显示出明显的优势。结论：MMRAG通过改进示例选择有效地提升了生物医学上下文学习，缓解了数据稀缺问题，并展示了在NLP驱动的健康医疗应用中优越的适应性。 

---
# Retrieval-Augmented Visual Question Answering via Built-in Autoregressive Search Engines 

**Title (ZH)**: 内置自回归搜索引擎辅助的检索增强视觉问答 

**Authors**: Xinwei Long, Zhiyuan Ma, Ermo Hua, Kaiyan Zhang, Biqing Qi, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.16641)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged to address the knowledge-intensive visual question answering (VQA) task. Current methods mainly employ separate retrieval and generation modules to acquire external knowledge and generate answers, respectively. We propose ReAuSE, an alternative to the previous RAG model for the knowledge-based VQA task, which seamlessly integrates knowledge retriever into the generative multi-modal large language model, serving as a built-in search engine. Specifically, our model functions both as a generative retriever and an accurate answer generator. It not only helps retrieve documents from the knowledge base by producing identifiers for each document, but it also answers visual questions based on the retrieved documents. Furthermore, we propose a reinforced retrieval calibration module from relevance feedback to improve retrieval performance and align with the preferences for accurate answer generation. Extensive experiments on two representative OKVQA and A-OKVQA datasets demonstrate significant improvements ranging from 2.9\% to 9.6\% across all evaluation metrics when compared to strong baselines. 

**Abstract (ZH)**: 检索增强生成（RAG）方法已 emergence 用于解决知识密集型视觉问答（VQA）任务。当前的方法主要通过独立的检索和生成模块来获取外部知识并生成答案。我们提出了一种名为 ReAuSE 的替代性 RAG 模型，专门用于基于知识的 VQA 任务，能够在生成多模态大型语言模型中无缝集成知识检索器，充当内置搜索引擎。具体来说，我们的模型既可以作为生成检索器，又可以作为准确的答案生成器。它不仅通过为每个文档生成标识符来帮助从知识库检索文档，还可以基于检索到的文档来回答视觉问题。此外，我们还提出了一种强化检索校准模块，利用相关反馈来提高检索性能，并与准确答案生成的需求保持一致。在两个典型的 OKVQA 和 A-OKVQA 数据集上的广泛实验表明，与强基线相比，所有评估指标均取得了从 2.9% 到 9.6% 的显著改进。 

---
# Enhancing Domain-Specific Retrieval-Augmented Generation: Synthetic Data Generation and Evaluation using Reasoning Models 

**Title (ZH)**: 增强领域特定的检索增强生成：基于推理模型的合成数据生成与评估 

**Authors**: Aryan Jadon, Avinash Patil, Shashank Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2502.15854)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems face significant performance gaps when applied to technical domains requiring precise information extraction from complex documents. Current evaluation methodologies relying on document-level metrics inadequately capture token-resolution retrieval accuracy that is critical for domain-related documents. We propose a framework combining granular evaluation metrics with synthetic data generation to optimize domain-specific RAG performance. First, we introduce token-aware metrics Precision $\Omega$ and Intersection-over-Union (IoU) that quantify context preservation versus information density trade-offs inherent in technical texts. Second, we develop a reasoning model-driven pipeline using instruction-tuned LLMs (DeepSeek-R1, DeepSeek-R1 distilled variants, and Phi-4) to generate context-anchored QA pairs with discontinuous reference spans across three specialized corpora: SEC 10-K filings (finance), biomedical abstracts (PubMed), and APT threat reports (cybersecurity).
Our empirical analysis reveals critical insights: smaller chunks (less than 10 tokens) improve precision by 31-42% (IoU = 0.071 vs. baseline 0.053) at recall costs (-18%), while domain-specific embedding strategies yield 22% variance in optimal chunk sizing (5-20 tokens). The DeepSeek-R1-Distill-Qwen-32B model demonstrates superior concept alignment (+14% mean IoU over alternatives), though no configuration universally dominates. Financial texts favor larger chunks for risk factor coverage (Recall = 0.81 at size = 20), whereas cybersecurity content benefits from atomic segmentation, Precision $\Omega = 0.28$ at size = 5.
Our code is available on this https URL 

**Abstract (ZH)**: Retrieval-Augmented Generation (RAG)系统在应用于需要从复杂文档中精确提取信息的技术领域时，面临着显著的性能差距。当前依赖文档级指标的评估方法未能充分捕捉到关键的标记级检索准确性，这对于领域的相关文档尤为重要。我们提出了一种结合细粒度评价指标和合成数据生成框架，以优化特定领域RAG性能。首先，我们引入了标记感知度量Precision $\Omega$和交并比(IoU)，这些度量量化了技术文本中内容保留与信息密度之间的权衡。其次，我们开发了基于推理模型的生成管道，该管道使用指令调优的大语言模型（DeepSeek-R1、DeepSeek-R1精简变体和Phi-4），以生成基于上下文的问答对，在三个专业语料库中使用跨段引用：SEC 10-K表单（金融）、PubMed生物医学摘要（生物医学）和APT威胁报告（网络安全）。

我们的实证分析揭示了关键见解：更小的片段（小于10个标记）在召回率为-18%的情况下，通过度量IoU = 0.071比基线0.053提高了31-42%的精确度，而特定领域嵌入策略在最佳片段大小（5-20个标记）方面的方差达到22%。DeepSeek-R1-Distill-Qwen-32B模型展示了概念一致性上的优势（相对于替代方案的平均IoU提高14%），但没有一种配置能够普遍占据优势。金融文本更倾向于使用更大的片段来覆盖风险因素（在大小为20时召回率为0.81），而网络安全内容则受益于原子化分割，在片段大小为5时Precision $\Omega$ = 0.28。

我们的代码可以在以下网址获取：[链接] 

---
# Political Events using RAG with LLMs 

**Title (ZH)**: 使用LLM的RAG进行政治事件处理 

**Authors**: Muhammad Arslan, Saba Munawar, Christophe Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2502.15701)  

**Abstract**: In the contemporary digital landscape, media content stands as the foundation for political news analysis, offering invaluable insights sourced from various channels like news articles, social media updates, speeches, and reports. Natural Language Processing (NLP) has revolutionized Political Information Extraction (IE), automating tasks such as Event Extraction (EE) from these diverse media outlets. While traditional NLP methods often necessitate specialized expertise to build rule-based systems or train machine learning models with domain-specific datasets, the emergence of Large Language Models (LLMs) driven by Generative Artificial Intelligence (GenAI) presents a promising alternative. These models offer accessibility, alleviating challenges associated with model construction from scratch and reducing the dependency on extensive datasets during the training phase, thus facilitating rapid implementation. However, challenges persist in handling domain-specific tasks, leading to the development of the Retrieval-Augmented Generation (RAG) framework. RAG enhances LLMs by integrating external data retrieval, enriching their contextual understanding, and expanding their knowledge base beyond pre-existing training data. To illustrate RAG's efficacy, we introduce the Political EE system, specifically tailored to extract political event information from news articles. Understanding these political insights is essential for remaining informed about the latest political advancements, whether on a national or global scale. 

**Abstract (ZH)**: 在当今的数字景观中，媒体内容构成了政治新闻分析的基础，提供了来自各种渠道（如新闻文章、社交媒体更新、演讲和报告）的宝贵见解。自然语言处理（NLP）已彻底改变了政治信息抽取（IE），实现了从这些多元媒体渠道自动提取事件（EE）等任务的自动化。虽然传统的NLP方法通常需要专门的专家构建基于规则的系统或使用领域特定数据集训练机器学习模型，但生成人工智能（GenAI）驱动的大规模语言模型（LLMs）的出现为这一领域的自动化提供了有前景的替代方案。这些模型提高了可访问性，减轻了从头构建模型的挑战，并在训练阶段减少了对大量数据集的依赖，从而促进了快速实现。然而，处理领域特定任务仍然存在挑战，这促进了检索增强生成（RAG）框架的发展。RAG通过整合外部数据检索，增强了大规模语言模型的上下文理解能力，并扩展了它们的知识库，使其超越了预先存在的训练数据。为了展示RAG的有效性，我们介绍了一种专门用于从新闻文章中抽取政治事件信息的政治事件提取系统。理解这些政治见解对于了解最新的政治进展至关重要，无论是国家级的还是国际性的。 

---
# Benchmarking Retrieval-Augmented Generation in Multi-Modal Contexts 

**Title (ZH)**: 在多模态上下文中的检索增强生成基准研究 

**Authors**: Zhenghao Liu, Xingsheng Zhu, Tianshuo Zhou, Xinyi Zhang, Xiaoyuan Yi, Yukun Yan, Yu Gu, Ge Yu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.17297)  

**Abstract**: This paper introduces Multi-Modal Retrieval-Augmented Generation (M^2RAG), a benchmark designed to evaluate the effectiveness of Multi-modal Large Language Models (MLLMs) in leveraging knowledge from multi-modal retrieval documents. The benchmark comprises four tasks: image captioning, multi-modal question answering, multi-modal fact verification, and image reranking. All tasks are set in an open-domain setting, requiring RAG models to retrieve query-relevant information from a multi-modal document collection and use it as input context for RAG modeling. To enhance the context utilization capabilities of MLLMs, we also introduce Multi-Modal Retrieval-Augmented Instruction Tuning (MM-RAIT), an instruction tuning method that optimizes MLLMs within multi-modal contexts. Our experiments show that MM-RAIT improves the performance of RAG systems by enabling them to effectively learn from multi-modal contexts. All data and code are available at this https URL. 

**Abstract (ZH)**: 本文介绍了一种名为 Multi-Modal Retrieval-Augmented Generation (M^2RAG) 的基准测试，用于评估多模态大型语言模型（Multi-modal Large Language Models, MLLMs）在利用多模态检索文档中的知识方面的有效性。该基准测试包括四个任务：图像字幕生成、多模态问答、多模态事实验证和图像重新排序。所有任务均在开放领域环境中进行，要求 RAG 模型从多模态文档集合中检索与查询相关的信息，并将其用作 RAG 模型的输入上下文。为了增强 MLLMs 对上下文的利用能力，我们还介绍了一种名为 Multi-Modal Retrieval-Augmented Instruction Tuning (MM-RAIT) 的指令调优方法，该方法旨在优化 MLLMs 在多模态上下文中的表现。实验结果表明，MM-RAIT 通过使 RAG 系统能够有效地从多模态上下文中学习，从而提升了其性能。所有数据和代码均在此处提供：[提供的链接]。 

---
# LawPal : A Retrieval Augmented Generation Based System for Enhanced Legal Accessibility in India 

**Title (ZH)**: LawPal：一种增强印度法律可访问性的检索增强生成系统 

**Authors**: Dnyanesh Panchal, Aaryan Gole, Vaibhav Narute, Raunak Joshi  

**Link**: [PDF](https://arxiv.org/pdf/2502.16573)  

**Abstract**: Access to legal knowledge in India is often hindered by a lack of awareness, misinformation and limited accessibility to judicial resources. Many individuals struggle to navigate complex legal frameworks, leading to the frequent misuse of laws and inadequate legal protection. To address these issues, we propose a Retrieval-Augmented Generation (RAG)-based legal chatbot powered by vectorstore oriented FAISS for efficient and accurate legal information retrieval. Unlike traditional chatbots, our model is trained using an extensive dataset comprising legal books, official documentation and the Indian Constitution, ensuring accurate responses to even the most complex or misleading legal queries. The chatbot leverages FAISS for rapid vector-based search, significantly improving retrieval speed and accuracy. It is also prompt-engineered to handle twisted or ambiguous legal questions, reducing the chances of incorrect interpretations. Apart from its core functionality of answering legal queries, the platform includes additional features such as real-time legal news updates, legal blogs, and access to law-related books, making it a comprehensive resource for users. By integrating advanced AI techniques with an optimized retrieval system, our chatbot aims to democratize legal knowledge, enhance legal literacy, and prevent the spread of misinformation. The study demonstrates that our approach effectively improves legal accessibility while maintaining high accuracy and efficiency, thereby contributing to a more informed and empowered society. 

**Abstract (ZH)**: 在印度，法律知识的获取往往受到缺乏意识、错误信息以及司法资源有限性的阻碍。许多个人难以驾驭复杂的法律框架，导致法律的误用和不充分的法律保护。为解决这些问题，我们提出了一种基于检索增强生成（RAG）的法律聊天机器人，该机器人由面向向量存储的FAISS驱动，以实现高效和准确的法律信息检索。与传统的聊天机器人不同，我们的模型是通过包含法律书籍、官方文件和印度宪法在内的大规模数据集进行训练的，确保能够准确回答最复杂或误导性的法律查询。该聊天机器人利用FAISS进行快速向量搜索，显著提高检索速度和准确性。此外，还通过对提示进行了精心设计，使其能够处理扭曲或模糊的法律问题，从而减少错误解释的可能性。除了其核心功能——回答法律查询外，该平台还包括实时法律新闻更新、法律博客以及法律书籍的访问权限，使其成为一个综合性的资源库供用户使用。通过整合先进的AI技术与优化的检索系统，我们的聊天机器人旨在使法律知识普及化，提高法律素养，并防止错误信息的传播。研究表明，我们的方法有效提高了法律信息的可访问性，同时保持了高度的准确性和效率，从而有助于一个更加知情和自主的社会。 

---
# Worse than Zero-shot? A Fact-Checking Dataset for Evaluating the Robustness of RAG Against Misleading Retrievals 

**Title (ZH)**: 比零样本更糟糕？一种用于评估RAG抵御误导性检索 robustness 的事实核查数据集 

**Authors**: Linda Zeng, Rithwik Gupta, Divij Motwani, Diji Yang, Yi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.16101)  

**Abstract**: Retrieval-augmented generation (RAG) has shown impressive capabilities in mitigating hallucinations in large language models (LLMs). However, LLMs struggle to handle misleading retrievals and often fail to maintain their own reasoning when exposed to conflicting or selectively-framed evidence, making them vulnerable to real-world misinformation. In such real-world retrieval scenarios, misleading and conflicting information is rampant, particularly in the political domain, where evidence is often selectively framed, incomplete, or polarized. However, existing RAG benchmarks largely assume a clean retrieval setting, where models succeed by accurately retrieving and generating answers from gold-standard documents. This assumption fails to align with real-world conditions, leading to an overestimation of RAG system performance. To bridge this gap, we introduce RAGuard, a fact-checking dataset designed to evaluate the robustness of RAG systems against misleading retrievals. Unlike prior benchmarks that rely on synthetic noise, our dataset constructs its retrieval corpus from Reddit discussions, capturing naturally occurring misinformation. It categorizes retrieved evidence into three types: supporting, misleading, and irrelevant, providing a realistic and challenging testbed for assessing how well RAG systems navigate different retrieval information. Our benchmark experiments reveal that when exposed to misleading retrievals, all tested LLM-powered RAG systems perform worse than their zero-shot baselines (i.e., no retrieval at all), highlighting their susceptibility to noisy environments. To the best of our knowledge, RAGuard is the first benchmark to systematically assess RAG robustness against misleading evidence. We expect this benchmark will drive future research toward improving RAG systems beyond idealized datasets, making them more reliable for real-world applications. 

**Abstract (ZH)**: 检索增强生成（RAG）在减轻大型语言模型（LLMs）幻觉方面表现出令人印象深刻的 capability。然而，当接触到误导性或部分呈现的证据时，LLMs 往往难以处理误导性的检索结果，常常在摇摆或有偏向的信息面前无法保持自己的推理能力，从而容易受到现实世界中的误导信息的影响。在这样的现实检索场景中，误导性和矛盾的信息非常普遍，尤其是在政治领域，证据往往是部分的、有偏向的或极化的。然而，现有的 RAG 基准测试大多假设一个干净的检索设置，在这种设置下，模型可以通过准确检索和生成答案来从金标准文档中取得成功。这种假设未能与现实世界的条件相吻合，导致对 RAG 系统性能的高估。为了弥合这一差距，我们介绍了一种名为 RAGuard 的事实核查数据集，用于评估 RAG 系统在面对误导性检索结果时的稳健性。与依赖合成噪声的前基准测试不同，我们的数据集从 Reddit 讨论中构建了其检索语料库，从而捕捉到真实发生的误导信息。该数据集将检索到的证据分为三类：支持性、误导性和不相关性，提供了一个现实和具有挑战性的测试平台，用于评估 RAG 系统如何处理不同的检索信息。我们的基准测试实验显示，当面对误导性检索结果时，所有测试的 LLM 动力 RAG 系统的表现都劣于零样本基准（即完全没有检索），突显了它们在嘈杂环境下的脆弱性。据我们所知，RAGuard 是第一个系统性评估 RAG 系统对误导性证据稳健性的基准测试。我们期望这一基准测试能够推动未来的研究，以改进 RAG 系统并超越理想化的数据集，使它们在实际应用中更可靠。 

---
# A novel approach to the relationships between data features -- based on comprehensive examination of mathematical, technological, and causal methodology 

**Title (ZH)**: 一种新的数据特征关系研究方法——基于数学、技术与因果关系方法的全面考察 

**Authors**: JaeHong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.15838)  

**Abstract**: The expansion of artificial intelligence (AI) has raised concerns about transparency, accountability, and interpretability, with counterfactual reasoning emerging as a key approach to addressing these issues. However, current mathematical, technological, and causal methodologies rely on externalization techniques that normalize feature relationships within a single coordinate space, often distorting intrinsic interactions. This study proposes the Convergent Fusion Paradigm (CFP) theory, a framework integrating mathematical, technological, and causal perspectives to provide a more precise and comprehensive analysis of feature relationships. CFP theory introduces Hilbert space and backward causation to reinterpret the feature relationships as emergent structures, offering a potential solution to the common cause problem -- a fundamental challenge in causal modeling. From a mathematical -- technical perspective, it utilizes a Riemannian manifold-based framework, thereby improving the structural representation of high- and low-dimensional data interactions. From a causal inference perspective, CFP theory adopts abduction as a methodological foundation, employing Hilbert space for a dynamic causal reasoning approach, where causal relationships are inferred abductively, and feature relationships evolve as emergent properties. Ultimately, CFP theory introduces a novel AI modeling methodology that integrates Hilbert space, backward causation, and Riemannian geometry, strengthening AI governance and transparency in counterfactual reasoning. 

**Abstract (ZH)**: 人工智能（AI）的扩展引发了关于透明度、问责制和可解释性的问题，反事实推理作为一种关键方法被提出以应对这些问题。然而，现有的数学、技术和因果推理方法依赖于将特征关系外部化的技术，这通常会导致内在交互的失真。本研究提出了统一融合范式（Convergent Fusion Paradigm，简称CFP）理论，这是一种整合数学、技术和因果推理视角的框架，旨在提供更精确和全面的特征关系分析。CFP理论通过引入希尔伯特空间和反向因果关系，重新解释特征关系为涌现结构，为因果模型中的共同原因问题提供了潜在的解决方案——这一问题是因果建模的基本挑战之一。从数学技术的角度来看，它利用了基于黎曼流形的框架，从而提高了高维和低维数据交互结构的表示。从因果推理的角度来看，CFP理论采用类比推理作为方法论基础，利用希尔伯特空间进行动态因果推理，其中因果关系通过类比推理推断，特征关系演化为涌现特性。最终，CFP理论引入了一种新型的AI建模方法，该方法结合了希尔伯特空间、反向因果关系和黎曼几何，从而增强了反事实推理中的AI治理和透明度。 

---
# Code Summarization Beyond Function Level 

**Title (ZH)**: 超越函数级的代码总结 

**Authors**: Vladimir Makharev, Vladimir Ivanov  

**Link**: [PDF](https://arxiv.org/pdf/2502.16704)  

**Abstract**: Code summarization is a critical task in natural language processing and software engineering, which aims to generate concise descriptions of source code. Recent advancements have improved the quality of these summaries, enhancing code readability and maintainability. However, the content of a repository or a class has not been considered in function code summarization. This study investigated the effectiveness of code summarization models beyond the function level, exploring the impact of class and repository contexts on the summary quality. The study involved revising benchmarks for evaluating models at class and repository levels, assessing baseline models, and evaluating LLMs with in-context learning to determine the enhancement of summary quality with additional context. The findings revealed that the fine-tuned state-of-the-art CodeT5+ base model excelled in code summarization, while incorporating few-shot learning and retrieved code chunks from RAG significantly enhanced the performance of LLMs in this task. Notably, the Deepseek Coder 1.3B and Starcoder2 15B models demonstrated substantial improvements in metrics such as BLEURT, METEOR, and BLEU-4 at both class and repository levels. Repository-level summarization exhibited promising potential but necessitates significant computational resources and gains from the inclusion of structured context. Lastly, we employed the recent SIDE code summarization metric in our evaluation. This study contributes to refining strategies for prompt engineering, few-shot learning, and RAG, addressing gaps in benchmarks for code summarization at various levels. Finally, we publish all study details, code, datasets, and results of evaluation in the GitHub repository available at this https URL. 

**Abstract (ZH)**: 代码总结是自然语言处理和软件工程中的一个关键任务，旨在生成源代码的简洁描述。近年来，代码总结的质量有所提升，从而增强了代码的可读性和可维护性。然而，在函数代码总结中，尚未考虑存储库或类的内容。本研究着眼于超越函数层面的代码总结模型的有效性，探索类和存储库上下文对总结质量的影响。研究包括修订用于评估模型的类和存储库级别基准，评估基线模型，并利用带上下文学习的LLM评估模型，以确定额外上下文对总结质量的提升。研究结果显示，微调的最新CodeT5+基模型在代码总结中表现出色，而少量示例学习以及从RAG检索代码片段显著提高了LLM在该任务中的性能。值得注意的是，Deepseek Coder 1.3B和Starcoder2 15B模型在BLEURT、METEOR和BLEU-4等指标方面在类和存储库级别上显示出显著改进。存储库级别的总结显示出巨大的潜力，但需要大量的计算资源，并能够从结构化上下文中获益。最后，我们在评估中采用了最新的SIDE代码总结度量标准。这项研究为优化提示工程、少量示例学习和RAG策略作出了贡献，填补了在不同级别上代码总结基准的空白。最后，我们在GitHub仓库（此 https URL）中发布了所有研究细节、代码、数据集和评估结果。 

---
# Retrieval Augmented Generation Based LLM Evaluation For Protocol State Machine Inference With Chain-of-Thought Reasoning 

**Title (ZH)**: 基于检索增强生成的大型语言模型评估：带有链式思考推理的协议状态机推理 

**Authors**: Youssef Maklad, Fares Wael, Wael Elsersy, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2502.15727)  

**Abstract**: This paper presents a novel approach to evaluate the efficiency of a RAG-based agentic Large Language Model (LLM) architecture in network packet seed generation for network protocol fuzzing. Enhanced by chain-of-thought (COT) prompting techniques, the proposed approach focuses on the improvement of the seeds structural quality in order to guide protocol fuzzing frameworks through a wide exploration of the protocol state space. Our method leverages RAG and text embeddings in a two-stages. In the first stage, the agent dynamically refers to the Request For Comments (RFC) documents knowledge base for answering queries regarding the protocol Finite State Machine (FSM), then it iteratively reasons through the retrieved knowledge, for output refinement and proper seed placement. In the second stage, we evaluate the response structure quality of the agent's output, based on metrics as BLEU, ROUGE, and Word Error Rate (WER) by comparing the generated packets against the ground truth packets. Our experiments demonstrate significant improvements of up to 18.19%, 14.81%, and 23.45% in BLEU, ROUGE, and WER, respectively, over baseline models. These results confirm the potential of such approach, improving LLM-based protocol fuzzing frameworks for the identification of hidden vulnerabilities. 

**Abstract (ZH)**: 本文提出了一种评估基于 Retrieval-Augmented Generation (RAG) 的代理型大型语言模型（LLM）架构在网络协议 fuzzing 中网络数据包种子生成效率的新方法。通过链式思考（Chain-of-Thought, COT）提示技术的增强，该方法专注于提高种子的结构质量，以指导 protocol fuzzing 框架在同一协议状态空间中进行广泛的探索。我们的方法在两个阶段中利用了 RAG 和文本嵌入。在第一阶段，代理动态地参考 RFC 文档知识库以回答有关协议有限状态机（FSM）的问题，然后通过检索到的知识进行迭代推理，以优化输出并正确放置种子。在第二阶段，我们根据 BLEU、ROUGE 和单词错误率（Word Error Rate, WER）等指标评估代理输出的响应结构质量，通过将生成的数据包与真实数据包进行比较。实验结果显示，与基线模型相比，我们的方法在 BLEU、ROUGE 和 WER 上分别取得了高达 18.19%、14.81% 和 23.45% 的改进。这些结果证实了该方法的潜力，可以改进基于 LLM 的协议 fuzzing 框架，以识别隐藏的漏洞。 

---
# Balancing Content Size in RAG-Text2SQL System 

**Title (ZH)**: 平衡RAG-Text2SQL系统中的内容大小 

**Authors**: Prakhar Gurawa, Anjali Dharmik  

**Link**: [PDF](https://arxiv.org/pdf/2502.15723)  

**Abstract**: Large Language Models (LLMs) have emerged as a promising solution for converting natural language queries into SQL commands, enabling seamless database interaction. However, these Text-to-SQL (Text2SQL) systems face inherent limitations, hallucinations, outdated knowledge, and untraceable reasoning. To address these challenges, the integration of retrieval-augmented generation (RAG) with Text2SQL models has gained traction. RAG serves as a retrieval mechanism, providing essential contextual information, such as table schemas and metadata, to enhance the query generation process. Despite their potential, RAG + Text2SQL systems are susceptible to the quality and size of retrieved documents. While richer document content can improve schema relevance and retrieval accuracy, it also introduces noise, increasing the risk of hallucinations and reducing query fidelity as the prompt size of the Text2SQL model increases. This research investigates the nuanced trade-off between document size and quality, aiming to strike a balance that optimizes system performance. Key thresholds are identified where performance degradation occurs, along with actionable strategies to mitigate these challenges. Additionally, we explore the phenomenon of hallucinations in Text2SQL models, emphasizing the critical role of curated document presentation in minimizing errors. Our findings provide a roadmap for enhancing the robustness of RAG + Text2SQL systems, offering practical insights for real-world applications. 

**Abstract (ZH)**: 大型语言模型（LLMs）已展现出将自然语言查询转化为SQL命令的潜在解决方案，从而实现与数据库的无缝交互。然而，这些文本到SQL（Text2SQL）系统存在固有的局限性，如虚构事实、过时知识以及不可追溯的推理。为应对这些挑战，将检索增强生成（RAG）与Text2SQL模型结合的应用逐渐受到关注。RAG充当一种检索机制，提供如表结构和元数据等关键的上下文信息，以提高查询生成的质量。尽管RAG + Text2SQL系统具有潜在优势，但它们对检索到文档的质量和数量非常敏感。丰富的文档内容有助于提高模式的相关性和检索准确性，但也可能引入噪音，增加虚构事实的风险，并随着Text2SQL模型提示长度的增加而降低查询精度。本研究探讨了文档大小与质量之间的微妙权衡，旨在寻找优化系统性能的最佳平衡点。我们识别出导致性能下降的关键阈值，并提出了一系列可操作的策略来应对这些挑战。此外，我们还探讨了Text2SQL模型中虚构事实的现象，强调精心呈现的文档在减少错误方面的作用。本研究的发现为增强RAG + Text2SQL系统的稳健性提供了蓝图，提供了实用的见解，适用于实际应用。 

---
# TutorLLM: Customizing Learning Recommendations with Knowledge Tracing and Retrieval-Augmented Generation 

**Title (ZH)**: TutorLLM：基于知识追踪和检索增强生成的个性化学习推荐 

**Authors**: Zhaoxing Li, Vahid Yazdanpanah, Jindi Wang, Wen Gu, Lei Shi, Alexandra I. Cristea, Sarah Kiden, Sebastian Stein  

**Link**: [PDF](https://arxiv.org/pdf/2502.15709)  

**Abstract**: The integration of AI in education offers significant potential to enhance learning efficiency. Large Language Models (LLMs), such as ChatGPT, Gemini, and Llama, allow students to query a wide range of topics, providing unprecedented flexibility. However, LLMs face challenges, such as handling varying content relevance and lack of personalization. To address these challenges, we propose TutorLLM, a personalized learning recommender LLM system based on Knowledge Tracing (KT) and Retrieval-Augmented Generation (RAG). The novelty of TutorLLM lies in its unique combination of KT and RAG techniques with LLMs, which enables dynamic retrieval of context-specific knowledge and provides personalized learning recommendations based on the student's personal learning state. Specifically, this integration allows TutorLLM to tailor responses based on individual learning states predicted by the Multi-Features with Latent Relations BERT-based KT (MLFBK) model and to enhance response accuracy with a Scraper model. The evaluation includes user assessment questionnaires and performance metrics, demonstrating a 10\% improvement in user satisfaction and a 5\% increase in quiz scores compared to using general LLMs alone. 

**Abstract (ZH)**: 将人工智能（AI）集成到教育中具有显著潜力，可提升学习效率。大型语言模型（LLMs），如ChatGPT、Gemini和Llama，使学生能够查询广泛的主题，提供了前所未有的灵活性。然而，LLMs 面临挑战，例如处理内容相关性的变化以及缺乏个性化。为了解决这些挑战，我们提出了一种基于知识追踪（KT）和检索增强生成（RAG）技术的个性化学习推荐LLMs系统——TutorLLM。TutorLLM 的创新之处在于将KT和RAG技术与LLMs相结合，使其能够动态检索上下文相关的知识，并基于学生的个人学习状态提供个性化学习建议。具体而言，这种集成允许TutorLLM 根据由基于多特征潜在关系BERT的知识追踪（MLFBK模型）预测的个别学习状态定制响应，并通过Scraper模型增强响应准确性。评估包括用户评估问卷和性能指标，结果显示，与单独使用通用的LLMs相比，用户满意度提高了10%，测验成绩提高了5%。 

---
# Sustainable Digitalization of Business with Multi-Agent RAG and LLM 

**Title (ZH)**: 基于多代理检索增强和大语言模型的可持续数字化商业 

**Authors**: Muhammad Arslan, Saba Munawar, Christophe Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2502.15700)  

**Abstract**: Businesses heavily rely on data sourced from various channels like news articles, financial reports, and consumer reviews to drive their operations, enabling informed decision-making and identifying opportunities. However, traditional manual methods for data extraction are often time-consuming and resource-intensive, prompting the adoption of digital transformation initiatives to enhance efficiency. Yet, concerns persist regarding the sustainability of such initiatives and their alignment with the United Nations (UN)'s Sustainable Development Goals (SDGs). This research aims to explore the integration of Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) as a sustainable solution for Information Extraction (IE) and processing. The research methodology involves reviewing existing solutions for business decision-making, noting that many systems require training new machine learning models, which are resource-intensive and have significant environmental impacts. Instead, we propose a sustainable business solution using pre-existing LLMs that can work with diverse datasets. We link domain-specific datasets to tailor LLMs to company needs and employ a Multi-Agent architecture to divide tasks such as information retrieval, enrichment, and classification among specialized agents. This approach optimizes the extraction process and improves overall efficiency. Through the utilization of these technologies, businesses can optimize resource utilization, improve decision-making processes, and contribute to sustainable development goals, thereby fostering environmental responsibility within the corporate sector. 

**Abstract (ZH)**: 企业高度依赖来自各种渠道的数据，如新闻报道、财务报告和消费者评论，以驱动其运营活动，从而实现明智的决策并识别机遇。然而，传统的手动数据提取方法通常耗时且资源密集，因此企业需采取数字化转型举措以提高效率。尽管如此，仍有关于这些举措可持续性和其与联合国可持续发展目标（SDGs）一致性方面的担忧。本研究旨在探讨将大型语言模型（LLMs）与检索增强生成（RAG）相结合作为信息提取（IE）和处理的可持续解决方案。研究方法包括审查现有企业决策支持解决方案，指出许多系统需要训练新的机器学习模型，这在资源和环境影响方面都具有显著的负担。相反，我们提出了一个可持续的商业解决方案，使用现有的LLMs并使其能够与多样化的数据集合作。通过将领域特定的数据集与公司需求相结合，优化LLMs，并采用多代理架构来分配信息检索、丰富和分类等任务，以此提高整体效率。通过利用这些技术，企业可以优化资源利用率、改进决策过程，并为可持续发展目标贡献力量，从而在企业界培养环境责任意识。 

---
# Open-Source Retrieval Augmented Generation Framework for Retrieving Accurate Medication Insights from Formularies for African Healthcare Workers 

**Title (ZH)**: 面向非洲医疗工作者从药典中检索准确药物洞察的开源检索增强生成框架 

**Authors**: Axum AI, J. Owoyemi, S. Abubakar, A. Owoyemi, T.O. Togunwa, F.C. Madubuko, S. Oyatoye, Z. Oyetolu, K. Akyea, A.O. Mohammed, A. Adebakin  

**Link**: [PDF](https://arxiv.org/pdf/2502.15722)  

**Abstract**: Accessing accurate medication insights is vital for enhancing patient safety, minimizing errors, and supporting clinical decision-making. However, healthcare professionals in Africa often rely on manual and time-consuming processes to retrieve drug information, exacerbated by limited access to pharmacists due to brain drain and healthcare disparities. This paper presents "Drug Insights," an open-source Retrieval-Augmented Generation (RAG) chatbot designed to streamline medication lookup for healthcare workers in Africa. By leveraging a corpus of Nigerian pharmaceutical data and advanced AI technologies, including Pinecone databases and GPT models, the system delivers accurate, context-specific responses with minimal hallucination. The chatbot integrates prompt engineering and S-BERT evaluation to optimize retrieval and response generation. Preliminary tests, including pharmacist feedback, affirm the tool's potential to improve drug information access while highlighting areas for enhancement, such as UI/UX refinement and extended corpus integration. 

**Abstract (ZH)**: 获取准确的药物见解对于提高患者安全、减少错误并支持临床决策至关重要。然而，由于人才流失和医疗资源分配不均，非洲的医疗卫生专业人员往往依赖耗时的手动过程来检索药物信息。本文介绍了“Drug Insights”，一个基于开源检索增强生成（RAG）聊天机器人的系统，旨在简化非洲医疗卫生工作者的药物检索流程。通过利用尼日利亚药学数据语料库和先进的AI技术，包括Pinecone数据库和GPT模型，该系统能够提供高精度、情境相关的响应，且极少出现幻觉。该聊天机器人集成了提示工程和S-BERT评估，以优化检索和响应生成。初步测试，包括药剂师的反馈，证实了该工具在提高药物信息访问方面的潜力，并指出了需要改进的领域，如用户界面/用户体验的优化和语料库扩展。 

---
# Developing an Artificial Intelligence Tool for Personalized Breast Cancer Treatment Plans based on the NCCN Guidelines 

**Title (ZH)**: 基于NCCN指南的个性化乳腺癌治疗方案人工智能工具的开发 

**Authors**: Abdul M. Mohammed, Iqtidar Mansoor, Sarah Blythe, Dennis Trujillo  

**Link**: [PDF](https://arxiv.org/pdf/2502.15698)  

**Abstract**: Cancer treatments require personalized approaches based on a patient's clinical condition, medical history, and evidence-based guidelines. The National Comprehensive Cancer Network (NCCN) provides frequently updated, complex guidelines through visuals like flowcharts and diagrams, which can be time consuming for oncologists to stay current with treatment protocols. This study presents an AI (Artificial Intelligence)-driven methodology to accurately automate treatment regimens following NCCN guidelines for breast cancer patients.
We proposed two AI-driven methods: Agentic-RAG (Retrieval-Augmented Generation) and Graph-RAG. Agentic-RAG used a three-step Large Language Model (LLM) process to select clinical titles from NCCN guidelines, retrieve matching JSON content, and iteratively refine recommendations based on insufficiency checks. Graph-RAG followed a Microsoft-developed framework with proprietary prompts, where JSON data was converted to text via an LLM, summarized, and mapped into graph structures representing key treatment relationships. Final recommendations were generated by querying relevant graph summaries. Both were evaluated using a set of patient descriptions, each with four associated questions.
As shown in Table 1, Agentic RAG achieved a 100% adherence (24/24) with no hallucinations or incorrect treatments. Graph-RAG had 95.8% adherence (23/24) with one incorrect treatment and no hallucinations. Chat GPT-4 showed 91.6% adherence (22/24) with two wrong treatments and no hallucinations. Both Agentic RAG and Graph-RAG provided detailed treatment recommendations with accurate references to relevant NCCN document page numbers. 

**Abstract (ZH)**: 癌症治疗需要根据患者的临床状况、医疗史和基于证据的指南采用个性化的方法。美国国家综合癌症网络（NCCN）提供了经常更新且复杂的指南，通过流程图和图表等形式呈现，这些对于肿瘤学家来说跟上治疗规范可能需要花费较多时间。本研究提出了一种人工智能（AI）驱动的方法，以准确地根据NCCN指南自动制定乳腺癌患者的治疗方案。

我们提出了两种AI驱动的方法：Agentic-RAG（检索增强生成）和Graph-RAG。Agentic-RAG采用了一种包含三个步骤的大规模语言模型（LLM）过程，从NCCN指南中选择临床标题，检索匹配的JSON内容，并基于不足检查逐步细化推荐。Graph-RAG遵循了由微软开发的框架，并使用专有提示，通过LLM将JSON数据转换为文本，进行总结并映射到表示关键治疗关系的图结构。最终的推荐通过查询相关图摘要生成。这两种方法都使用了一组患者的描述进行评估，每组描述包含四个关联的问题。

如表1所示，Agentic RAG达到了100%的合规性（24/24），没有出现幻觉或错误治疗。Graph-RAG的合规性为95.8%（23/24），有一个错误的治疗并且没有出现幻觉。Chat GPT-4的合规性为91.6%（22/24），有两个错误的治疗但没有出现幻觉。Agentic RAG和Graph-RAG两种方法都提供了详细的治疗建议，并准确引用了相关的NCCN文档页码。 

---
