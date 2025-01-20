# A Simple but Effective Closed-form Solution for Extreme Multi-label Learning 

**Title (ZH)**: 一种简单而有效的极端多标签学习的闭形式解决方案 

**Authors**: Kazuma Onishi, Katsuhiko Hayashi  

**Link**: [PDF](https://arxiv.org/pdf/2501.10179)  

**Abstract**: Extreme multi-label learning (XML) is a task of assigning multiple labels from an extremely large set of labels to each data instance. Many current high-performance XML models are composed of a lot of hyperparameters, which complicates the tuning process. Additionally, the models themselves are adapted specifically to XML, which complicates their reimplementation. To remedy this problem, we propose a simple method based on ridge regression for XML. The proposed method not only has a closed-form solution but also is composed of a single hyperparameter. Since there are no precedents on applying ridge regression to XML, this paper verified the performance of the method by using various XML benchmark datasets. Furthermore, we enhanced the prediction of low-frequency labels in XML, which hold informative content. This prediction is essential yet challenging because of the limited amount of data. Here, we employed a simple frequency-based weighting. This approach greatly simplifies the process compared with existing techniques. Experimental results revealed that it can achieve levels of performance comparable to, or even exceeding, those of models with numerous hyperparameters. Additionally, we found that the frequency-based weighting significantly improved the predictive performance for low-frequency labels, while requiring almost no changes in implementation. The source code for the proposed method is available on github at this https URL. 

**Abstract (ZH)**: 极多标签学习（Extreme Multi-Label Learning，XML）是一项任务，即从一个极其庞大的标签集合中为每个数据实例分配多个标签。当前许多高性能的XML模型包含大量的超参数，这使得调优过程变得复杂。此外，这些模型本身是针对XML专门设计的，增加了重新实现的复杂性。为解决这一问题，我们提出了一种基于岭回归的简单方法来处理XML。该方法不仅具有闭式解，还仅包含一个超参数。由于尚未见有将岭回归应用于XML的先例，本文通过使用多种XML基准数据集验证了方法的性能。此外，我们还增强了XML中低频标签的预测，这些标签包含有价值的信息。由于数据量有限，这一预测既是必要的又是极具挑战性的。在此过程中，我们采用了简单的基于频率的加权方法。与现有技术相比，这种方法大大简化了过程。实验结果表明，该方法可以达到与大量超参数模型相当，甚至更好的性能水平。此外，我们发现基于频率的加权方法显著提高了低频标签的预测性能，同时几乎不需要修改实现过程。所提出方法的源代码可以在GitHub上通过此链接获得：[GitHub链接]。 

---
# MechIR: A Mechanistic Interpretability Framework for Information Retrieval 

**Title (ZH)**: MechIR：信息检索的机制可解释性框架 

**Authors**: Andrew Parry, Catherine Chen, Carsten Eickhoff, Sean MacAvaney  

**Link**: [PDF](https://arxiv.org/pdf/2501.10165)  

**Abstract**: Mechanistic interpretability is an emerging diagnostic approach for neural models that has gained traction in broader natural language processing domains. This paradigm aims to provide attribution to components of neural systems where causal relationships between hidden layers and output were previously uninterpretable. As the use of neural models in IR for retrieval and evaluation becomes ubiquitous, we need to ensure that we can interpret why a model produces a given output for both transparency and the betterment of systems. This work comprises a flexible framework for diagnostic analysis and intervention within these highly parametric neural systems specifically tailored for IR tasks and architectures. In providing such a framework, we look to facilitate further research in interpretable IR with a broader scope for practical interventions derived from mechanistic interpretability. We provide preliminary analysis and look to demonstrate our framework through an axiomatic lens to show its applications and ease of use for those IR practitioners inexperienced in this emerging paradigm. 

**Abstract (ZH)**: 机制可解释性是一种新兴的诊断方法，已在更广泛的自然语言处理领域获得关注。该范式旨在为神经系统的组件提供归因，这些组件之间的因果关系在以前是不可解释的。随着神经模型在信息检索（IR）中的应用（用于检索和评估）变得普遍，我们需要确保能够解释模型生成特定输出的原因，以确保透明度并改善系统性能。本文构架了一个灵活的框架，用于在这些高度参数化的神经系统中进行诊断分析和干预，特别针对IR任务和架构。通过提供这样一个构架，我们旨在促进对可解释IR的研究，并扩大从机制可解释性中获得实际干预的应用范围。我们提供了初步分析，并通过公理化的方法来展示该框架的应用和便捷性，以便为不熟悉这一新兴范式的IR实践者展示其效用。 

---
# A Worrying Reproducibility Study of Intent-Aware Recommendation Models 

**Title (ZH)**: 令人担忧的意图感知推荐模型可重复性研究 

**Authors**: Faisal Shehzad, Maurizio Ferrari Dacrema, Dietmar Jannach  

**Link**: [PDF](https://arxiv.org/pdf/2501.10143)  

**Abstract**: Lately, we have observed a growing interest in intent-aware recommender systems (IARS). The promise of such systems is that they are capable of generating better recommendations by predicting and considering the underlying motivations and short-term goals of consumers. From a technical perspective, various sophisticated neural models were recently proposed in this emerging and promising area. In the broader context of complex neural recommendation models, a growing number of research works unfortunately indicates that (i) reproducing such works is often difficult and (ii) that the true benefits of such models may be limited in reality, e.g., because the reported improvements were obtained through comparisons with untuned or weak baselines. In this work, we investigate if recent research in IARS is similarly affected by such problems. Specifically, we tried to reproduce five contemporary IARS models that were published in top-level outlets, and we benchmarked them against a number of traditional non-neural recommendation models. In two of the cases, running the provided code with the optimal hyperparameters reported in the paper did not yield the results reported in the paper. Worryingly, we find that all examined IARS approaches are consistently outperformed by at least one traditional model. These findings point to sustained methodological issues and to a pressing need for more rigorous scholarly practices. 

**Abstract (ZH)**: 近年来，我们观察到在意图感知推荐系统（IARS）方面出现了越来越大的兴趣。这类系统的核心承诺在于，它们能够通过预测和考虑消费者的潜在动机和短期目标来生成更好的推荐。从技术角度来看，这一新兴且充满潜力的领域中最近提出了一些复杂的神经模型。在更广泛的复杂神经推荐模型的背景下，越来越多的研究工作表明，（i）重现这些工作往往非常困难，（ii）并且这类模型的实际效益可能有限，例如，因为所报告的改进是通过与未调参或较弱的基础模型进行比较得出的。在本文中，我们研究近期的IARS研究是否也受到了类似问题的影响。具体来说，我们尝试重现了五种发表在顶级出版物上的当代IARS模型，并将它们与多种传统的非神经推荐模型进行了基准测试。在两种情况下，使用论文中报告的最优超参数运行提供的代码并没有得到论文中报告的结果。令人担忧的是，我们发现所有研究的IARS方法至少都被一种传统模型超过了。这些发现指向了持续存在的方法论问题，并强调了更严格的学术实践的紧迫需求。 

---
# PaSa: An LLM Agent for Comprehensive Academic Paper Search 

**Title (ZH)**: PaSa：一种全面学术论文搜索的大型语言模型代理 

**Authors**: Yichen He, Guanhua Huang, Peiyuan Feng, Yuan Lin, Yuchen Zhang, Hang Li, Weinan E  

**Link**: [PDF](https://arxiv.org/pdf/2501.10120)  

**Abstract**: We introduce PaSa, an advanced Paper Search agent powered by large language models. PaSa can autonomously make a series of decisions, including invoking search tools, reading papers, and selecting relevant references, to ultimately obtain comprehensive and accurate results for complex scholarly queries. We optimize PaSa using reinforcement learning with a synthetic dataset, AutoScholarQuery, which includes 35k fine-grained academic queries and corresponding papers sourced from top-tier AI conference publications. Additionally, we develop RealScholarQuery, a benchmark collecting real-world academic queries to assess PaSa performance in more realistic scenarios. Despite being trained on synthetic data, PaSa significantly outperforms existing baselines on RealScholarQuery, including Google, Google Scholar, Google with GPT-4 for paraphrased queries, chatGPT (search-enabled GPT-4o), GPT-o1, and PaSa-GPT-4o (PaSa implemented by prompting GPT-4o). Notably, PaSa-7B surpasses the best Google-based baseline, Google with GPT-4o, by 37.78% in recall@20 and 39.90% in recall@50. It also exceeds PaSa-GPT-4o by 30.36% in recall and 4.25% in precision. Model, datasets, and code are available at this https URL. 

**Abstract (ZH)**: 我们介绍了PaSa，这是一种由大规模语言模型驱动的高级论文搜索代理。PaSa可以自主作出一系列决策，包括调用搜索工具、阅读论文和选择相关参考文献，最终为复杂的学术查询提供全面而准确的结果。我们使用强化学习并利用一个合成数据集AutoScholarQuery来优化PaSa，该数据集包含了35,000个精细的学术查询及其相应的论文，这些论文来源于顶级人工智能会议的出版物。此外，我们还开发了RealScholarQuery基准数据集，收集了真实世界的学术查询，以更现实的场景评估PaSa的表现。尽管PaSa是基于合成数据进行训练的，但它在RealScholarQuery上的表现显著优于现有基线，包括Google、Google Scholar、使用GPT-4进行改写的Google、chatGPT（搜索功能增强的GPT-4o）、GPT-o1和PaSa-GPT-4o（基于GPT-4o实现的PaSa）。值得注意的是，PaSa-7B在召回率@20上比基于Google的最佳基线（Google与GPT-4结合）高出37.78%，在召回率@50上高出39.90%。此外，它的召回率超过了PaSa-GPT-4o 30.36%，而精确率高出4.25%。相关模型、数据集和代码可在以下网址获取。 

---
# Empirical Evaluation of Embedding Models in the Context of Text Classification in Document Review in Construction Delay Disputes 

**Title (ZH)**: 在建筑延期纠纷文件审查中基于文本分类的嵌入模型实证评价 

**Authors**: Fusheng Wei, Robert Neary, Han Qin, Qiang Mao, Jianping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.09859)  

**Abstract**: Text embeddings are numerical representations of text data, where words, phrases, or entire documents are converted into vectors of real numbers. These embeddings capture semantic meanings and relationships between text elements in a continuous vector space. The primary goal of text embeddings is to enable the processing of text data by machine learning models, which require numerical input. Numerous embedding models have been developed for various applications. This paper presents our work in evaluating different embeddings through a comprehensive comparative analysis of four distinct models, focusing on their text classification efficacy. We employ both K-Nearest Neighbors (KNN) and Logistic Regression (LR) to perform binary classification tasks, specifically determining whether a text snippet is associated with 'delay' or 'not delay' within a labeled dataset. Our research explores the use of text snippet embeddings for training supervised text classification models to identify delay-related statements during the document review process of construction delay disputes. The results of this study highlight the potential of embedding models to enhance the efficiency and accuracy of document analysis in legal contexts, paving the way for more informed decision-making in complex investigative scenarios. 

**Abstract (ZH)**: 文本嵌入是文本数据的数值表示，其中单词、短语或整个文档被转换成实数值向量。这些嵌入在连续向量空间中捕捉文本元素的语义意义及其相互关系。文本嵌入的主要目标是通过需要数值输入的机器学习模型来处理文本数据。已经为各种应用开发了多种嵌入模型。本文呈现了我们对不同嵌入模型的全面比较分析，重点研究它们在文本分类中的有效性。我们采用K-近邻（KNN）和逻辑回归（LR）来执行二分类任务，具体是确定文本片段是否与“延误”或“不延误”相关，特别是在标记数据集中。我们的研究探讨了使用文本片段嵌入来训练监督文本分类模型，以识别建筑延误争议文件审查过程中相关的延误声明。该研究的结果突显了嵌入模型在提升法律背景下文档分析效率和准确性方面的潜力，为复杂调查场景中的更明智决策铺平了道路。 

---
# Conversational Text Extraction with Large Language Models Using Retrieval-Augmented Systems 

**Title (ZH)**: 使用检索增强系统的大型语言模型进行会话文本提取 

**Authors**: Soham Roy, Mitul Goswami, Nisharg Nargund, Suneeta Mohanty, Prasant Kumar Pattnaik  

**Link**: [PDF](https://arxiv.org/pdf/2501.09801)  

**Abstract**: This study introduces a system leveraging Large Language Models (LLMs) to extract text and enhance user interaction with PDF documents via a conversational interface. Utilizing Retrieval-Augmented Generation (RAG), the system provides informative responses to user inquiries while highlighting relevant passages within the PDF. Upon user upload, the system processes the PDF, employing sentence embeddings to create a document-specific vector store. This vector store enables efficient retrieval of pertinent sections in response to user queries. The LLM then engages in a conversational exchange, using the retrieved information to extract text and generate comprehensive, contextually aware answers. While our approach demonstrates competitive ROUGE values compared to existing state-of-the-art techniques for text extraction and summarization, we acknowledge that further qualitative evaluation is necessary to fully assess its effectiveness in real-world applications. The proposed system gives competitive ROUGE values as compared to existing state-of-the-art techniques for text extraction and summarization, thus offering a valuable tool for researchers, students, and anyone seeking to efficiently extract knowledge and gain insights from documents through an intuitive question-answering interface. 

**Abstract (ZH)**: 本研究介绍了一种利用大型语言模型（LLMs）的系统，通过对话界面提取文本并增强用户与PDF文档的交互。该系统利用检索增强生成（RAG）技术，在用户提问时提供信息性的响应，并突出显示相关段落。在用户上传PDF文档后，系统对其进行处理，使用句子嵌入创建文档特定的向量存储。该向量存储可高效地检索与用户查询相关的内容。随后，LLM参与对话交流，利用检索到的信息提取文本并生成全面且上下文相关的问题回答。尽管我们的方法在文本提取和总结方面的ROUGE值与现有最先进的技术相比具有竞争力，但我们承认还需要进一步的定性评估，以全面评估其在实际应用中的有效性。所提出系统在文本提取和总结方面的ROUGE值上与现有最先进的技术具有竞争力，因此为研究人员、学生以及任何希望通过直观的问答界面高效提取知识和获得见解的人提供了一个有价值的工具。 

---
# Passage Segmentation of Documents for Extractive Question Answering 

**Title (ZH)**: 文档提取式问答中的段落分割技术 

**Authors**: Zuhong Liu, Charles-Elie Simon, Fabien Caspani  

**Link**: [PDF](https://arxiv.org/pdf/2501.09940)  

**Abstract**: Retrieval-Augmented Generation (RAG) has proven effective in open-domain question answering. However, the chunking process, which is essential to this pipeline, often receives insufficient attention relative to retrieval and synthesis components. This study emphasizes the critical role of chunking in improving the performance of both dense passage retrieval and the end-to-end RAG pipeline. We then introduce the Logits-Guided Multi-Granular Chunker (LGMGC), a novel framework that splits long documents into contextualized, self-contained chunks of varied granularity. Our experimental results, evaluated on two benchmark datasets, demonstrate that LGMGC not only improves the retrieval step but also outperforms existing chunking methods when integrated into a RAG pipeline. 

**Abstract (ZH)**: 检索增强生成（RAG）已被证明在开放域问答中非常有效。然而，在这一过程中至关重要的切块步骤往往没有受到与检索和合成组件相同程度的关注。本研究强调了切块在提高密集段落检索性能以及端到端RAG管道性能中的关键作用。随后，我们介绍了Logits引导的多粒度切块器（LGMGC），这是一种新颖的框架，可以将长文档划分为不同粒度的自包含上下文块。我们的实验结果，在两个基准数据集上进行了评估，证明LGMGC不仅能改进检索步骤，而且在集成到RAG管道中时还能优于现有的切块方法。 

---
# Semi-Supervised Image-Based Narrative Extraction: A Case Study with Historical Photographic Records 

**Title (ZH)**: 半监督图像基础叙事提取：以历史摄影记录为例的研究 

**Authors**: Fausto German, Brian Keith, Mauricio Matus, Diego Urrutia, Claudio Meneses  

**Link**: [PDF](https://arxiv.org/pdf/2501.09884)  

**Abstract**: This paper presents a semi-supervised approach to extracting narratives from historical photographic records using an adaptation of the narrative maps algorithm. We extend the original unsupervised text-based method to work with image data, leveraging deep learning techniques for visual feature extraction and similarity computation. Our method is applied to the ROGER dataset, a collection of photographs from the 1928 Sacambaya Expedition in Bolivia captured by Robert Gerstmann. We compare our algorithmically extracted visual narratives with expert-curated timelines of varying lengths (5 to 30 images) to evaluate the effectiveness of our approach. In particular, we use the Dynamic Time Warping (DTW) algorithm to match the extracted narratives with the expert-curated baseline. In addition, we asked an expert on the topic to qualitatively evaluate a representative example of the resulting narratives. Our findings show that the narrative maps approach generally outperforms random sampling for longer timelines (10+ images, p < 0.05), with expert evaluation confirming the historical accuracy and coherence of the extracted narratives. This research contributes to the field of computational analysis of visual cultural heritage, offering new tools for historians, archivists, and digital humanities scholars to explore and understand large-scale image collections. The method's ability to generate meaningful narratives from visual data opens up new possibilities for the study and interpretation of historical events through photographic evidence. 

**Abstract (ZH)**: 本文提出了一种半监督方法，用于从历史照片记录中提取叙述，该方法基于对Narrative Maps算法的改编。我们将最初的无监督基于文本的方法扩展到处理图像数据，并利用深度学习技术进行视觉特征提取和相似性计算。该方法应用于ROGER数据集，该数据集包含1928年玻利维亚Sacambaya探险期间由Robert Gerstmann拍摄的照片集合。我们将算法提取的视觉叙述与不同长度（5至30张图像）的专家编目时间线进行比较，以评估我们方法的有效性。特别是，我们使用动态时间规整（DTW）算法将提取的叙述与专家编目的基线进行匹配。此外，我们请该领域的专家对结果中典型示例的叙述进行定性评估。研究结果表明，对于较长的时间线（10张及以上图像，p < 0.05），叙事地图方法通常优于随机抽样，在专家评估中确认了提取叙述的历史准确性和一致性。这项研究为视觉文化遗产的计算分析领域做出了贡献，为历史学家、档案工作者和人文学者提供了新的工具，以探索和理解大规模图像集合。该方法从视觉数据中生成有意义的叙述的能力为通过照片证据研究和解释历史事件开辟了新的可能。 

---
# OmniThink: Expanding Knowledge Boundaries in Machine Writing through Thinking 

**Title (ZH)**: OmniThink：通过思考扩展机器写作的知识边界 

**Authors**: Zekun Xi, Wenbiao Yin, Jizhan Fang, Jialong Wu, Runnan Fang, Ningyu Zhang, Jiang Yong, Pengjun Xie, Fei Huang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.09751)  

**Abstract**: Machine writing with large language models often relies on retrieval-augmented generation. However, these approaches remain confined within the boundaries of the model's predefined scope, limiting the generation of content with rich information. Specifically, vanilla-retrieved information tends to lack depth, utility, and suffers from redundancy, which negatively impacts the quality of generated articles, leading to shallow, repetitive, and unoriginal outputs. To address these issues, we propose OmniThink, a machine writing framework that emulates the human-like process of iterative expansion and reflection. The core idea behind OmniThink is to simulate the cognitive behavior of learners as they progressively deepen their knowledge of the topics. Experimental results demonstrate that OmniThink improves the knowledge density of generated articles without compromising metrics such as coherence and depth. Human evaluations and expert feedback further highlight the potential of OmniThink to address real-world challenges in the generation of long-form articles. 

**Abstract (ZH)**: 大语言模型进行机器写作时，常常依赖检索增强生成。然而，这些方法仍然局限于模型预设的范畴内，限制了丰富信息生成内容的能力。具体来说，单纯通过检索获得的信息往往缺乏深度、实用性，并且容易出现冗余，这降低了生成文章的质量，导致浅薄、重复且缺乏原创性的输出。为了解决这些问题，我们提出了OmniThink，这是一种模仿人类逐步深化知识学习过程的认知行为的机器写作框架。OmniThink的核心思想是模拟学习者在逐步加深对主题理解时的认知行为。实验结果表明，OmniThink在提高生成文章的知识密度的同时，不会牺牲连贯性和深度等指标。人工评估和专家反馈进一步突显了OmniThink在应对长篇文章生成中的现实挑战方面的潜力。 

---
