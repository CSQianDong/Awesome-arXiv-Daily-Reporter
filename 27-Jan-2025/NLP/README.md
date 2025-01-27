# Comparable Corpora: Opportunities for New Research Directions 

**Title (ZH)**: 可比语料库：新的研究方向机遇

这个标题是学术论文常见的翻译方法，既保留了原意，又符合中文的表达习惯。如有更多具体内容或需要进一步的帮助，请告诉我！ 

**Authors**: Kenneth Church  

**Link**: [PDF](https://arxiv.org/pdf/2501.14721)  

**Abstract**: Most conference papers present new results, but this paper will focus more on opportunities for the audience to make their own contributions. This paper is intended to challenge the community to think more broadly about what we can do with comparable corpora. We will start with a review of the history, and then suggest new directions for future research. This was a keynote at BUCC-2025, a workshop associated with Coling-2025. 

**Abstract (ZH)**: 以下是这段内容的中文翻译，符合学术规范：

大多数会议论文侧重于展示新的研究成果，但本文更侧重于为读者提供自己做出贡献的机会。本文旨在推动学术界从更广泛的角度思考如何利用可比语料库进行研究。我们将从历史回顾开始，然后提出未来研究的新方向。本文是BUCC-2025（与Coling-2025相关的研讨会）的特邀演讲。 

---
# Do LLMs Provide Consistent Answers to Health-Related Questions across Languages? 

**Title (ZH)**: 大型语言模型在不同语言中对健康相关问题的回答是否具有一致性？ 

**Authors**: Ipek Baris Schlicht, Zhixue Zhao, Burcu Sayin, Lucie Flek, Paolo Rosso  

**Link**: [PDF](https://arxiv.org/pdf/2501.14719)  

**Abstract**: Equitable access to reliable health information is vital for public health, but the quality of online health resources varies by language, raising concerns about inconsistencies in Large Language Models (LLMs) for healthcare. In this study, we examine the consistency of responses provided by LLMs to health-related questions across English, German, Turkish, and Chinese. We largely expand the HealthFC dataset by categorizing health-related questions by disease type and broadening its multilingual scope with Turkish and Chinese translations. We reveal significant inconsistencies in responses that could spread healthcare misinformation. Our main contributions are 1) a multilingual health-related inquiry dataset with meta-information on disease categories, and 2) a novel prompt-based evaluation workflow that enables sub-dimensional comparisons between two languages through parsing. Our findings highlight key challenges in deploying LLM-based tools in multilingual contexts and emphasize the need for improved cross-lingual alignment to ensure accurate and equitable healthcare information. 

**Abstract (ZH)**: 公平获取可靠的健康信息对于公共卫生至关重要，但在线健康资源的质量因语言而异，这引发了对大规模语言模型（LLM）在健康领域中一致性问题的担忧。在本文中，我们考察了LLM对英语、德语、土耳其语和汉语健康相关问题的回答一致性。我们通过对疾病类型进行分类，并使用土耳其语和汉语翻译扩展HealthFC数据集，大大扩大了其多语言范围。我们揭示了回答中存在显著的一致性问题，可能会传播医疗健康误导信息。我们的主要贡献包括：1）一个包含疾病类别元数据的多语言健康相关查询数据集；2）一种新的基于提示的评估工作流程，可以通过解析实现两种语言在子维度上的比较。我们的研究结果突出了在多语言环境中部署基于LLM的工具所面临的关键挑战，并强调了加强跨语言对齐以确保准确和公平的健康信息的重要性。 

---
# Towards Better Understanding Table Instruction Tuning: Decoupling the Effects from Data versus Models 

**Title (ZH)**: 朝更好地理解表指令调优进展：分离数据与模型效果的影响 

**Authors**: Naihao Deng, Sheng Zhang, Henghui Zhu, Shuaichen Chang, Jiani Zhang, Alexander Hanbo Li, Chung-Wei Hang, Hideo Kobayashi, Yiqun Hu, Patrick Ng  

**Link**: [PDF](https://arxiv.org/pdf/2501.14717)  

**Abstract**: Recent advances in natural language processing have leveraged instruction tuning to enhance Large Language Models (LLMs) for table-related tasks. However, previous works train different base models with different training data, lacking an apples-to-apples comparison across the result table LLMs. To address this, we fine-tune base models from the Mistral, OLMo, and Phi families on existing public training datasets. Our replication achieves performance on par with or surpassing existing table LLMs, establishing new state-of-the-art performance on Hitab, a table question-answering dataset. More importantly, through systematic out-of-domain evaluation, we decouple the contributions of training data and the base model, providing insight into their individual impacts. In addition, we assess the effects of table-specific instruction tuning on general-purpose benchmarks, revealing trade-offs between specialization and generalization. 

**Abstract (ZH)**: 近年来，自然语言处理领域的最新进展利用指令调优来增强大型语言模型（LLMs）在表格相关任务上的性能。然而，以往的工作使用了不同基础模型和不同训练数据集进行训练，缺乏对结果表格LLMs的直接对比。为了解决这一问题，我们对来自Mistral、OLMo和Phi系列的基础模型，在现有的公共训练数据集上进行了微调。我们的复现结果达到了或超过了现有表格LLMs的性能，建立了在Hitab数据集上的新的最先进性能，Hitab是一个表格问答数据集。更重要的是，通过系统的领域外评估，我们分离了训练数据和基础模型对性能的影响，提供了它们各自影响的洞察。此外，我们还评估了针对表格的指令调优对通用基准的影响，揭示了专业化与泛化之间的权衡。 

---
# FlexiGPT: Pruning and Extending Large Language Models with Low-Rank Weight Sharing 

**Title (ZH)**: FlexiGPT：通过低秩权重共享进行大规模语言模型的剪枝与扩展 

**Authors**: James Seale Smith, Chi-Heng Lin, Shikhar Tuli, Haris Jeelani, Shangqian Gao, Yilin Shen, Hongxia Jin, Yen-Chang Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14713)  

**Abstract**: The rapid proliferation of large language models (LLMs) in natural language processing (NLP) has created a critical need for techniques that enable efficient deployment on memory-constrained devices without compromising performance. We present a method to prune LLMs that selectively prunes model blocks based on an importance score and replaces them with a low-parameter replacement strategy. Specifically, we propose a principled metric to replace each pruned block using a weight-sharing mechanism that leverages unpruned counterparts from the model and block-specific low-rank adapters. Furthermore, we facilitate the learning of these replacement blocks with output feature normalization and an adapter initialization scheme built on low-rank SVD reconstructions. Empirical evaluations demonstrate substantial performance gains over existing methods, achieving state-of-the-art performance on 5/6 benchmarks for a compression rate of 30% and 6/6 benchmarks for a compression rate of 40%. We also demonstrate that our approach can extend smaller models, boosting performance on 6/6 benchmarks using only ~0.3% tokens of extended training with minimal additional parameter costs. 

**Abstract (ZH)**: 自然语言处理（NLP）领域中大型语言模型（LLMs）的迅速发展，对能够在内存受限设备上高效部署的技术提出了迫切需求，同时不牺牲性能。本文提出了一种方法，通过基于重要性评分选择性地剪枝模型模块，并用低参数替换策略来替换这些模块。具体来说，我们提出了一种原则性的度量方法，通过利用模型中未剪枝的对应模块和特定模块的低秩适配器，使用权重共享机制为每个剪枝模块寻找替换模块。此外，我们通过输出特征归一化和基于低秩SVD重构的适配器初始化方案，促进了这些替换模块的学习。实证评估表明，相较于现有方法，本方法取得了显著的性能提升，在压缩率为30%时，取得了5/6基准测试的最优表现；在压缩率为40%时，则在6/6基准测试中取得了最优表现。我们还展示了这种方法可以扩展较小的模型，在仅进行约0.3%额外训练序列的扩展训练下，在所有6/6基准测试中提升了性能，且几乎不增加额外的参数成本。 

---
# NLP-based assessment of prescription appropriateness from Italian referrals 

**Title (ZH)**: 基于NLP的意大利转诊单处方适宜性评估 

**Authors**: Vittorio Torri, Annamaria Bottelli, Michele Ercolanoni, Olivia Leoni, Francesca Ieva  

**Link**: [PDF](https://arxiv.org/pdf/2501.14701)  

**Abstract**: Objective: This study proposes a Natural Language Processing pipeline to evaluate prescription appropriateness in Italian referrals, where reasons for prescriptions are recorded only as free text, complicating automated comparisons with guidelines. The pipeline aims to derive, for the first time, a comprehensive summary of the reasons behind these referrals and a quantification of their appropriateness. While demonstrated in a specific case study, the approach is designed to generalize to other types of examinations.
Methods: Leveraging embeddings from a transformer-based model, the proposed approach clusters referral texts, maps clusters to labels, and aligns these labels with existing guidelines. We present a case study on a dataset of 496,971 referrals, consisting of all referrals for venous echocolordopplers of the lower limbs between 2019 and 2021 in the Lombardy Region. A sample of 1,000 referrals was manually annotated to validate the results.
Results: The pipeline exhibited high performance for referrals' reasons (Prec=92.43%, Rec=83.28%) and excellent results for referrals' appropriateness (Prec=93.58%, Rec=91.52%) on the annotated subset. Analysis of the entire dataset identified clusters matching guideline-defined reasons - both appropriate and inappropriate - as well as clusters not addressed in the guidelines. Overall, 34.32% of referrals were marked as appropriate, 34.07% inappropriate, 14.37% likely inappropriate, and 17.24% could not be mapped to guidelines.
Conclusions: The proposed pipeline effectively assessed prescription appropriateness across a large dataset, serving as a valuable tool for health authorities. Findings have informed the Lombardy Region's efforts to strengthen recommendations and reduce the burden of inappropriate referrals. 

**Abstract (ZH)**: 目标：本文提出了一种自然语言处理（NLP）管道，用于评估意大利转诊中的处方适宜性，其中处方原因仅记录为自由文本，这给与指南进行自动化比较带来了挑战。该管道旨在首次提取这些转诊背后的多个理由，并量化其适宜性。虽然该方法在特定案例研究中得到验证，但其设计目的在于推广到其他类型的检查。

方法：利用基于变换器的模型生成的词嵌入，提出的方法将转诊文本聚类，将聚类映射到标签，并将这些标签与现有的指导方针对齐。我们采用了一组496,971份转诊记录的数据集进行案例研究，这些数据集包括2019年至2021年伦巴第大区下肢静脉超声彩色多普勒检查的所有转诊记录。从中随机选出1,000份转诊记录进行人工标注，以验证结果。

结果：该管道在带有标注的数据子集上对转诊原因表现出高水平的性能（精确率P=92.43%，召回率R=83.28%），对转诊适宜性的量化评估也表现出色（精确率P=93.58%，召回率R=91.52%）。通过对整个数据集的分析，识别出了与指导方针定义的理由（包括恰当的和不恰当的）相对应的聚类，以及未在指导方针中提及的聚类。总体而言，34.32%的转诊被标记为适当，34.07%的转诊被标记为不适当，14.37%的转诊被标记为很可能不适当，17.24%的转诊无法与指导方针对齐。

结论：提出的管道有效地评估了大规模数据集中处方的适宜性，成为卫生部门的一项有价值的工具。研究结果为伦巴第大区加强建议和减少不适当转诊的负担提供了依据。 

---
# Rethinking Table Instruction Tuning 

**Title (ZH)**: 重新思考表格指令调优 

**Authors**: Naihao Deng, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2501.14693)  

**Abstract**: Recent advances in table understanding have focused on instruction-tuning large language models (LLMs) for table-related tasks. However, existing research has overlooked the impact of hyperparameter choices and lacks a comprehensive evaluation of the out-of-domain table understanding ability and the general capabilities of these table LLMs. In this paper, we evaluate these abilities in existing table LLMs, and reveal significant declines in both out-of-domain table understanding and general capabilities compared to their base models. Through systematic analysis, we show that hyperparameters, such as learning rate, can significantly influence both table-specific and general capabilities. Contrary to the existing table instruction-tuning works, we demonstrate that smaller learning rates and fewer training instances can enhance table understanding while preserving general capabilities. Based on our findings, we introduce TAMA, a TAble LLM instruction-tuned from LLaMA 3.1 8B Instruct, which achieves performance on par with, or surpassing GPT-3.5 and GPT-4 on table tasks, while maintaining strong out-of-domain generalization and general capabilities. Our findings highlight the potential for reduced data annotation costs and more efficient model development through careful hyperparameter selection. 

**Abstract (ZH)**: 近年来，表格理解领域的进展主要集中在使用指令调优大型语言模型（LLMs）进行表格相关任务。然而，现有研究忽略了超参数选择的影响，并且缺乏对这些表格LLMs跨域表格理解能力和一般能力的全面评估。本文评估了现有表格LLMs的这些能力，发现这些模型在跨域表格理解和一般能力上相比基模型有显著下降。通过系统的分析，我们揭示了学习率等超参数对表格特定能力和一般能力有显著影响。与现有表格指令调优工作相反，我们的研究表明较小的学习率和较少的训练实例可以提升表格理解能力，同时保持一般能力。基于我们的发现，我们引入了TAMA，这是一个从LLaMA 3.1 8B Instruct调优的表格LLM，能够在表格任务上达到或超越GPT-3.5和GPT-4的性能，同时保持强大的跨域泛化能力和一般能力。我们的发现强调了通过仔细选择超参数减少数据注释成本和提高模型开发效率的潜力。 

---
# State Space Models for Extractive Summarization in Low Resource Scenarios 

**Title (ZH)**: 低资源场景下基于状态空间模型的抽取式总结方法 

**Authors**: Nisrine Ait Khayi  

**Link**: [PDF](https://arxiv.org/pdf/2501.14673)  

**Abstract**: Extractive summarization involves selecting the most relevant sentences from a text. Recently, researchers have focused on advancing methods to improve state-of-the-art results in low-resource settings. Motivated by these advancements, we propose the MPoincareSum method. This method applies the Mamba state space model to generate the semantics of reviews and sentences, which are then concatenated. A Poincare compression is used to select the most meaningful features, followed by the application of a linear layer to predict sentence relevance based on the corresponding review. Finally, we paraphrase the relevant sentences to create the final summary. To evaluate the effectiveness of MPoincareSum, we conducted extensive experiments using the Amazon review dataset. The performance of the method was assessed using ROUGE scores. The experimental results demonstrate that MPoincareSum outperforms several existing approaches in the literature 

**Abstract (ZH)**: 提取式总结涉及从文本中选择最相关的句子。近日，研究人员聚焦于在资源匮乏的情况下改进方法以提升最先进的结果。受到这些进展的启发，我们提出了MPoincareSum方法。该方法利用Mamba状态空间模型生成评论和句子的语义，并将这些语义进行拼接。随后使用Poincare压缩来选择最有意义的特征，并应用线性层基于相应的评论预测句子的相关性。最后，我们对相关的句子进行改写以生成最终的摘要。为了评估MPoincareSum的有效性，我们使用Amazon评论数据集进行了广泛的实验，并使用ROUGE分数评估了方法的性能。实验结果表明，MPoincareSum在文献中现有的多种方法中表现更优。 

---
# Investigating the (De)Composition Capabilities of Large Language Models in Natural-to-Formal Language Conversion 

**Title (ZH)**: 探究大型语言模型在自然语言到形式语言转换中的（分解与）组合能力 

**Authors**: Ziyao Xu, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14649)  

**Abstract**: To achieve generalized and robust natural-to-formal language conversion (N2F), large language models (LLMs) need to have strong capabilities of decomposition and composition in N2F when faced with an unfamiliar formal language and be able to cope with compositional gaps and counter-intuitive symbolic names. To investigate whether LLMs have this set of basic capabilities in N2F, we propose the DEDC framework. This framework semi-automatically performs sample and task construction, allowing decoupled evaluation of the set of decomposition and composition capabilities of LLMs in N2F. Based on this framework, we evaluate and analyze the most advanced LLMs, and the main findings include that: (1) the LLMs are deficient in both decomposition and composition; (2) the LLMs show a wide coverage of error types that can be attributed to deficiencies in natural language understanding and the learning and use of symbolic systems; (3) compositional gaps and counter-intuitive symbolic names both affect the decomposition and composition of the LLMs. Our work provides a new perspective for investigating the basic capabilities of decomposition and composition of LLMs in N2F. The detailed analysis of deficiencies and attributions can help subsequent improvements of LLMs. 

**Abstract (ZH)**: 为了实现通用且鲁棒的自然语言到形式语言转换（N2F），大型语言模型（LLMs）在面对不熟悉的正式语言时，需要具备强大的分解和组合能力，并能够应对组合空白和反直觉的符号名称。为了探究LLMs是否具备这种基本的N2F能力，我们提出了DEDC框架。该框架半自动地执行样本和任务的构造，使得能够从解构和组合两个方面独立评估LLMs在N2F中的能力。基于此框架，我们评估和分析了最先进的LLMs，主要发现包括：（1）LLMs在解构和组合方面都存在缺陷；（2）LLMs显示出广泛覆盖的错误类型，这些错误可以归因于自然语言理解和符号系统学习与使用能力的不足；（3）组合空白和反直觉的符号名称都影响着LLMs的解构和组合能力。我们的工作为探究LLMs在N2F中的基本解构和组合能力提供了新的视角。详细的缺陷分析和归因可以帮助后续改进LLMs。 

---
# Funzac at CoMeDi Shared Task: Modeling Annotator Disagreement from Word-In-Context Perspectives 

**Title (ZH)**: Funzac在CoMeDi共享任务中的研究：基于词上下文视角建模注释者分歧 

**Authors**: Olufunke O. Sarumi, Charles Welch, Lucie Flek, Jörg Schlötterer  

**Link**: [PDF](https://arxiv.org/pdf/2501.14617)  

**Abstract**: In this work, we evaluate annotator disagreement in Word-in-Context (WiC) tasks exploring the relationship between contextual meaning and disagreement as part of the CoMeDi shared task competition. While prior studies have modeled disagreement by analyzing annotator attributes with single-sentence inputs, this shared task incorporates WiC to bridge the gap between sentence-level semantic representation and annotator judgment variability. We describe three different methods that we developed for the shared task, including a feature enrichment approach that combines concatenation, element-wise differences, products, and cosine similarity, Euclidean and Manhattan distances to extend contextual embedding representations, a transformation by Adapter blocks to obtain task-specific representations of contextual embeddings, and classifiers of varying complexities, including ensembles. The comparison of our methods demonstrates improved performance for methods that include enriched and task-specfic features. While the performance of our method falls short in comparison to the best system in subtask 1 (OGWiC), it is competitive to the official evaluation results in subtask 2 (DisWiC). 

**Abstract (ZH)**: 在本研究中，我们评估了Word-in-Context (WiC) 任务中的注释者分歧，并探讨了上下文意义与分歧之间的关系，这作为CoMeDi共享任务竞赛的一部分。尽管以前的研究通过分析注释者属性并对单句输入进行建模来估计分歧，但本次共享任务通过引入WiC，跨越了句级语义表示与注释者判断变异之间的差距。我们描述了为共享任务开发的三种不同方法，包括结合串联、逐元素差异、乘积和余弦相似度以及欧几里得和曼哈顿距离进行特征丰富的方法，以扩展上下文嵌入表示；通过Adapter块进行变换以获得特定任务的上下文嵌入表示；以及包括集成在内的不同复杂度的分类器。我们的方法之间的比较表明，包含丰富和特定任务特征的方法在性能上有所提高。尽管我们的方法在子任务1（OGWiC）中的表现不如最佳系统，但在子任务2（DisWiC）中，我们的方法与官方评估结果相当。 

---
# Idiom Detection in Sorani Kurdish Texts 

**Title (ZH)**: 索拉尼库尔德语中的成语识别 

**Authors**: Skala Kamaran Omer, Hossein Hassani  

**Link**: [PDF](https://arxiv.org/pdf/2501.14528)  

**Abstract**: Idiom detection using Natural Language Processing (NLP) is the computerized process of recognizing figurative expressions within a text that convey meanings beyond the literal interpretation of the words. While idiom detection has seen significant progress across various languages, the Kurdish language faces a considerable research gap in this area despite the importance of idioms in tasks like machine translation and sentiment analysis. This study addresses idiom detection in Sorani Kurdish by approaching it as a text classification task using deep learning techniques. To tackle this, we developed a dataset containing 10,580 sentences embedding 101 Sorani Kurdish idioms across diverse contexts. Using this dataset, we developed and evaluated three deep learning models: KuBERT-based transformer sequence classification, a Recurrent Convolutional Neural Network (RCNN), and a BiLSTM model with an attention mechanism. The evaluations revealed that the transformer model, the fine-tuned BERT, consistently outperformed the others, achieving nearly 99% accuracy while the RCNN achieved 96.5% and the BiLSTM 80%. These results highlight the effectiveness of Transformer-based architectures in low-resource languages like Kurdish. This research provides a dataset, three optimized models, and insights into idiom detection, laying a foundation for advancing Kurdish NLP. 

**Abstract (ZH)**: 使用自然语言处理（NLP）进行成语检测是计算机化地识别文本中具有字面意义之外含义的比喻表达的过程。尽管在多种语言中成语检测取得了显著进展，但在库尔德语这一重要领域，仍存在较大的研究缺口，这在机器翻译和情感分析等任务中具有重要意义。本研究旨在通过使用深度学习技术将库尔德语成语检测作为文本分类任务来填补这一空白。为此，我们开发了一个包含10,580个句子的数据集，这些句子涵盖了101个不同上下文中出现的索尔尼库尔德语成语。使用该数据集，我们构建并评估了三种深度学习模型：基于KuBERT的转换器序列分类模型、循环卷积神经网络（RCNN）以及带有注意力机制的双向长短期记忆（BiLSTM）模型。评估结果显示，转换器模型（即微调后的BERT）始终表现最佳，准确率达近99%，而RCNN的准确率为96.5%，BiLSTM为80%。这些结果突显了转换器架构在资源较少的语言（如库尔德语）中的有效性。本研究提供了数据集、三种优化模型以及关于成语检测的见解，为进一步推进库尔德语NLP奠定了基础。 

---
# WanJuanSiLu: A High-Quality Open-Source Webtext Dataset for Low-Resource Languages 

**Title (ZH)**: WanJuanSiLu：一种面向低资源语言的高质量开源网络文本数据集 

**Authors**: Jia Yu, Fei Yuan, Rui Min, Jing Yu, Pei Chu, Jiayang Li, Wei Li, Ruijie Zhang, Zhenxiang Li, Zhifei Ren, Dong Zheng, Wenjian Zhang, Yan Teng, Lingyu Meng, ZhenJiang Jin, Jiantao Qiu, ShaSha Wang, Zhongying Tu, Dahua Lin, Yu Wang, Yu Qiao, Yanfeng Wang, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2501.14506)  

**Abstract**: This paper introduces the open-source dataset WanJuanSiLu, designed to provide high-quality training corpora for low-resource languages, thereby advancing the research and development of multilingual models. To achieve this, we have developed a systematic data processing framework tailored for low-resource languages. This framework encompasses key stages such as data extraction, corpus cleaning, content deduplication, security filtering, quality evaluation, and theme classification. Through the implementation of this framework, we have significantly improved both the quality and security of the dataset, while maintaining its linguistic diversity. As of now, data for all five languages have been fully open-sourced. The dataset can be accessed at this https URL, and GitHub repository is available at this https URL 

**Abstract (ZH)**: 本文介绍了开源数据集“万卷四录”（WanJuanSiLu），旨在为低资源语言提供高质量的训练语料，从而推动多语言模型的研究与发展。为此，我们开发了一套专门针对低资源语言的数据处理框架，该框架涵盖了数据提取、语料清洗、内容去重、安全过滤、质量评估和主题分类等关键步骤。通过实施这一框架，我们显著提高了数据集的质量和安全性，同时保持了其语言多样性。截至目前，五种语言的数据均已完全开源。数据集可通过以下链接访问：[这个网址]，GitHub仓库地址为：[这个网址] 

---
# Evaluating and Improving Graph to Text Generation with Large Language Models 

**Title (ZH)**: 评估并改进基于大型语言模型的图到文本生成 

**Authors**: Jie He, Yijun Yang, Wanqiu Long, Deyi Xiong, Victor Gutierrez Basulto, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2501.14497)  

**Abstract**: Large language models (LLMs) have demonstrated immense potential across various tasks. However, research for exploring and improving the capabilities of LLMs in interpreting graph structures remains limited. To address this gap, we conduct a comprehensive evaluation of prompting current open-source LLMs on graph-to-text generation tasks. Although we explored the optimal prompting strategies and proposed a novel and effective diversity-difficulty-based few-shot sample selection method, we found that the improvements from tuning-free approaches were incremental, as LLMs struggle with planning on complex graphs, particularly those with a larger number of triplets. To further improve LLMs in planning with graph sequences and grounding in truth, we introduce a new graph-to-text dataset, PlanGTG, annotated with two sub-tasks: reordering and attribution. Through extensive automatic and human evaluations, we demonstrate significant improvements in the quality of generated text from both few-shot learning and fine-tuning perspectives using the PlanGTG dataset. Our study paves the way for new research directions in graph-to-text generation. PlanGTG datasets can be found in this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在各种任务中展现出了巨大的潜力。然而，对于探索和提升LLMs在解释图形结构方面的能力的研究仍然有限。为了弥补这一空白，我们对当前开源的LLMs在图形到文本生成任务上的提示进行了全面评估。尽管我们探索了最优的提示策略，并提出了一种基于多样性和难度的少样本示例选择的新颖且有效的方法，但我们发现，无微调的方法带来的改进是增量的，主要是因为LLMs在处理复杂图形时存在困难，特别是当图形包含大量三元组时。为了进一步提高LLMs在图形序列上的规划能力和与事实的对接能力，我们引入了一个新的图形到文本数据集PlanGTG，并为此数据集标注了两个子任务：重排序和归因。通过广泛的自动和人工评估，我们证明了在使用PlanGTG数据集从少样本学习和微调的角度来看，生成文本的质量得到了显著提升。本研究为图形到文本生成领域的未来研究方向铺平了道路。PlanGTG数据集可以在以下链接找到：[请在这里插入链接]。 

---
# RealCritic: Towards Effectiveness-Driven Evaluation of Language Model Critiques 

**Title (ZH)**: RealCritic: 面向有效性评估的语言模型批评标准 

**Authors**: Zhengyang Tang, Ziniu Li, Zhenyang Xiao, Tian Ding, Ruoyu Sun, Benyou Wang, Dayiheng Liu, Fei Huang, Tianyu Liu, Bowen Yu, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2501.14492)  

**Abstract**: Critiques are important for enhancing the performance of Large Language Models (LLMs), enabling both self-improvement and constructive feedback for others by identifying flaws and suggesting improvements. However, evaluating the critique capabilities of LLMs presents a significant challenge due to the open-ended nature of the task. In this work, we introduce a new benchmark designed to assess the critique capabilities of LLMs. Unlike existing benchmarks, which typically function in an open-loop fashion, our approach employs a closed-loop methodology that evaluates the quality of corrections generated from critiques. Moreover, the benchmark incorporates features such as self-critique, cross-critique, and iterative critique, which are crucial for distinguishing the abilities of advanced reasoning models from more classical ones. We implement this benchmark using eight challenging reasoning tasks. We have several interesting findings. First, despite demonstrating comparable performance in direct chain-of-thought generation, classical LLMs significantly lag behind the advanced reasoning-based model o1-mini across all critique scenarios. Second, in self-critique and iterative critique settings, classical LLMs may even underperform relative to their baseline capabilities. We hope that this benchmark will serve as a valuable resource to guide future advancements. The code and data are available at \url{this https URL}. 

**Abstract (ZH)**: 批判对于提升大规模语言模型（LLMs）的性能至关重要，通过识别错误和建议改进措施，模型可以实现自我提升并与他人提供建设性的反馈。然而，评估LLMs的批判能力是一个重大挑战，原因在于该任务的开放性。在本文中，我们提出了一种新的基准测试方法，用于评估LLMs的批判能力。不同于现有的通常以开环方式运作的基准测试，我们的方法采用了闭环方法，评估从批判中生成的修正的质量。此外，该基准测试还包含自我批判、交叉批判和迭代批判等特征，这对于区分高级推理模型与经典模型的能力至关重要。我们利用八个具有挑战性的推理任务来实施这一基准测试。我们的研究有几个有趣的发现。首先，尽管在直接链式思考生成方面表现出相似的性能，但经典LLMs在所有批判场景中都显著落后于基于高级推理的模型o1-mini。其次，在自我批判和迭代批判设置中，经典LLMs甚至可能不如其基线性能。我们希望这个基准测试能够作为未来研究的一个宝贵资源。相关的代码和数据可在以下链接获得：\url{this https URL}。 

---
# Analyzing the Effect of Linguistic Similarity on Cross-Lingual Transfer: Tasks and Experimental Setups Matter 

**Title (ZH)**: 分析语言相似性对跨语言迁移效果的影响：任务类型和实验设置至关重要 

**Authors**: Verena Blaschke, Masha Fedzechkina, Maartje ter Hoeve  

**Link**: [PDF](https://arxiv.org/pdf/2501.14491)  

**Abstract**: Cross-lingual transfer is a popular approach to increase the amount of training data for NLP tasks in a low-resource context. However, the best strategy to decide which cross-lingual data to include is unclear. Prior research often focuses on a small set of languages from a few language families and/or a single task. It is still an open question how these findings extend to a wider variety of languages and tasks. In this work, we analyze cross-lingual transfer for 266 languages from a wide variety of language families. Moreover, we include three popular NLP tasks: POS tagging, dependency parsing, and topic classification. Our findings indicate that the effect of linguistic similarity on transfer performance depends on a range of factors: the NLP task, the (mono- or multilingual) input representations, and the definition of linguistic similarity. 

**Abstract (ZH)**: 跨语种迁移是一种在资源有限环境下增加NLP任务训练数据的方法。然而，决定纳入哪些跨语种数据的最佳策略尚不明确。先前的研究通常仅关注少数几种语言家族中的少量语言和/或单一任务。这些发现是否能推广到更广泛的语言和任务仍是一个悬而未决的问题。本研究分析了266种来自不同语言家族的跨语种迁移。此外，还纳入了三种流行的NLP任务：词性标注、依存句法解析和主题分类。我们的研究结果表明，语言相似性对迁移性能的影响取决于多种因素：NLP任务、单语或多语输入表示，以及语言相似性的定义。 

---
# Understanding and Mitigating Gender Bias in LLMs via Interpretable Neuron Editing 

**Title (ZH)**: 通过可解释的神经元编辑理解并减轻大语言模型中的性别偏见 

**Authors**: Zeping Yu, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2501.14457)  

**Abstract**: Large language models (LLMs) often exhibit gender bias, posing challenges for their safe deployment. Existing methods to mitigate bias lack a comprehensive understanding of its mechanisms or compromise the model's core capabilities. To address these issues, we propose the CommonWords dataset, to systematically evaluate gender bias in LLMs. Our analysis reveals pervasive bias across models and identifies specific neuron circuits, including gender neurons and general neurons, responsible for this behavior. Notably, editing even a small number of general neurons can disrupt the model's overall capabilities due to hierarchical neuron interactions. Based on these insights, we propose an interpretable neuron editing method that combines logit-based and causal-based strategies to selectively target biased neurons. Experiments on five LLMs demonstrate that our method effectively reduces gender bias while preserving the model's original capabilities, outperforming existing fine-tuning and editing approaches. Our findings contribute a novel dataset, a detailed analysis of bias mechanisms, and a practical solution for mitigating gender bias in LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）经常会表现出性别偏向，这为它们的安全部署带来了挑战。现有减少偏见的方法缺乏对其机制的全面理解，也可能损害模型的核心能力。为应对这些问题，我们提出了CommonWords数据集，以系统地评估LLMs中的性别偏向。我们的分析揭示了这些偏向在不同模型中的普遍存在性，并确定了特定的神经元电路，包括性别神经元和通用神经元，这些电路负责这种行为。值得注意的是，即使编辑少量的通用神经元，也可能会因为层次神经元间的相互作用而严重影响模型的整体能力。基于这些洞见，我们提出了一种可解释的神经元编辑方法，它结合了基于logit和因果策略来针对性地靶向偏见神经元。在五个LLMs上的实验表明，我们的方法不仅有效减少了性别偏向，而且保持了模型原有的能力，优于现有的微调和编辑方法。我们的研究成果贡献了一个新型数据集、对偏见机制的详细分析，以及在LLMs中减少性别偏见的实际解决方案。 

---
# Domaino1s: Guiding LLM Reasoning for Explainable Answers in High-Stakes Domains 

**Title (ZH)**: Domaino1s：引导LLM推理以在高 stakes 领域生成可解释的答案 

**Authors**: Xu Chu, Zhijie Tan, Hanlin Xue, Guanyu Wang, Tong Mo, Weiping Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.14431)  

**Abstract**: Large Language Models (LLMs) are widely applied to downstream domains. However, current LLMs for high-stakes domain tasks, such as financial investment and legal QA, typically generate brief answers without reasoning processes and explanations. This limits users' confidence in making decisions based on their responses. While original CoT shows promise, it lacks self-correction mechanisms during reasoning. This work introduces Domain$o1$s, which enhances LLMs' reasoning capabilities on domain tasks through supervised fine-tuning and tree search. We construct CoT-stock-2k and CoT-legal-2k datasets for fine-tuning models that activate domain-specific reasoning steps based on their judgment. Additionally, we propose Selective Tree Exploration to spontaneously explore solution spaces and sample optimal reasoning paths to improve performance. We also introduce PROOF-Score, a new metric for evaluating domain models' explainability, complementing traditional accuracy metrics with richer assessment dimensions. Extensive experiments on stock investment recommendation and legal reasoning QA tasks demonstrate Domaino1s's leading performance and explainability. Our code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）广泛应用于下游领域。然而，当前应用于高风险任务领域的LLMs，如金融投资和法律问答，通常仅生成简短的答案而缺乏推理过程和解释。这限制了用户根据这些答案做出决策的信心。虽然原始CoT显示出潜力，但其推理过程中缺乏自我修正机制。本研究引入了Domain\$o1\$, 通过监督微调和树搜索增强LLMs在领域任务上的推理能力。我们构建了CoT-stock-2k和CoT-legal-2k数据集，用于微调能够基于自身判断激活领域特异性推理步骤的模型。此外，我们提出了选择性树探索方法，自动探索解决方案空间，并采样最优推理路径以改进表现。我们还引入了PROOF-Score这一新的评估指标，通过丰富的评估维度补充了传统的准确性指标，以评估领域模型的解释性。大量实验在股票投资推荐和法律推理问答任务上证明了Domain\$o1\$的优越性能和解释性。我们的代码可在以下链接获取：https://your-link-url.com 

---
# DRESSing Up LLM: Efficient Stylized Question-Answering via Style Subspace Editing 

**Title (ZH)**: 强化LLM的着装：通过风格子空间编辑实现高效的样式化问答 

**Authors**: Xinyu Ma, Yifeng Xu, Yang Lin, Tianlong Wang, Xu Chu, Xin Gao, Junfeng Zhao, Yasha Wang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14371)  

**Abstract**: We introduce DRESS, a novel approach for generating stylized large language model (LLM) responses through representation editing. Existing methods like prompting and fine-tuning are either insufficient for complex style adaptation or computationally expensive, particularly in tasks like NPC creation or character role-playing. Our approach leverages the over-parameterized nature of LLMs to disentangle a style-relevant subspace within the model's representation space to conduct representation editing, ensuring a minimal impact on the original semantics. By applying adaptive editing strengths, we dynamically adjust the steering vectors in the style subspace to maintain both stylistic fidelity and semantic integrity. We develop two stylized QA benchmark datasets to validate the effectiveness of DRESS, and the results demonstrate significant improvements compared to baseline methods such as prompting and ITI. In short, DRESS is a lightweight, train-free solution for enhancing LLMs with flexible and effective style control, making it particularly useful for developing stylized conversational agents. Codes and benchmark datasets are available at this https URL. 

**Abstract (ZH)**: 我们介绍了DRESS，这是一种通过表示编辑生成风格化大型语言模型（LLM）响应的新型方法。现有方法如提示和微调要么在复杂风格适应方面不够充分，要么在诸如NPC创作或角色扮演等任务中计算成本高昂。我们的方法利用大型语言模型的过参数化特性，在模型表示空间中解耦一个与风格相关的子空间，以进行表示编辑，确保对原始语义的影响最小。通过应用自适应编辑强度，我们动态调整风格子空间中的控制向量，以保持风格忠诚度和语义完整性。我们开发了两个风格化问答基准数据集来验证DRESS的有效性，结果显示与提示和交互式训练（ITI）等基线方法相比，有显著改进。简而言之，DRESS是一种轻量级、无需训练的解决方案，可通过灵活有效的风格控制增强LLMs，使其特别适用于开发风格化对话代理。代码和基准数据集可在以下链接访问：this https URL。 

---
# Clear Minds Think Alike: What Makes LLM Fine-tuning Robust? A Study of Token Perplexity 

**Title (ZH)**: 清朗之心思相似：.what 使预训练语言模型微调鲁棒？基于标记困惑度的研究 

**Authors**: Chao-Chung Wu, Zhi Rui Tam, Chieh-Yen Lin, Hung-yi Lee, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2501.14315)  

**Abstract**: Maintaining consistent model performance across domains is a fundamental challenge in machine learning. While recent work has explored using LLM-generated data for fine-tuning, its impact on cross-domain generalization remains poorly understood. In this paper, we present a systematic analysis revealing that fine-tuning with LLM-generated data not only improves target task performance but also reduces out-of-domain (OOD) degradation compared to fine-tuning with ground truth data. Through analyzing the data sequence in tasks of various domains, we demonstrate that this enhanced OOD robustness stems from a reduced prevalence of high perplexity tokens in LLM-generated sequences. Following this hypothesis we showed that masking high perplexity tokens in ground truth training data also achieves similar OOD preservation comparable to using LLM-generated data. Extensive experiments across diverse model architectures and scales, including Gemma2-2B, Mistral-7B and Llama3-8B, corroborate the consistency of our findings. To the best of our knowledge, this work provides the first mechanistic explanation for the superior OOD robustness conferred by LLM-generated training data, offering valuable insights for developing more robust fine-tuning strategies. 

**Abstract (ZH)**: 在不同领域保持一致的模型性能是机器学习中的一个基本挑战。尽管近期研究探索了使用大语言模型（LLM）生成的数据进行微调的方法，但其对跨领域泛化的具体影响尚未充分了解。本文我们进行了一项系统性分析，揭示了使用LLM生成的数据进行微调不仅能提高目标任务的性能，还能减少在未见过领域（OOD）上的性能下降，相较于使用真实数据进行微调的情况。通过对不同领域任务数据序列的分析，我们证明了这种增强的OOD鲁棒性来源于LLM生成序列中高困惑度词元概率的降低。基于这一假设，我们展示了屏蔽真实数据训练中的高困惑度词元也能实现类似水平的OOD保护，效果与使用LLM生成数据相当。广泛的实验涵盖了多种不同的模型架构和规模，包括Gemma2-2B、Mistral-7B和Llama3-8B，验证了我们研究结果的一致性。据我们所知，这项工作首次对LLM生成训练数据所赋予的更强OOD鲁棒性提供了机制性解释，为开发更鲁棒的微调策略提供了宝贵的见解。 

---
# Examining Alignment of Large Language Models through Representative Heuristics: The Case of Political Stereotypes 

**Title (ZH)**: 通过代表性启发式方法考察大型语言模型的对齐情况：以政治刻板印象为例 

**Authors**: Sullam Jeoung, Yubin Ge, Haohan Wang, Jana Diesner  

**Link**: [PDF](https://arxiv.org/pdf/2501.14294)  

**Abstract**: Examining the alignment of large language models (LLMs) has become increasingly important, particularly when these systems fail to operate as intended. This study explores the challenge of aligning LLMs with human intentions and values, with specific focus on their political inclinations. Previous research has highlighted LLMs' propensity to display political leanings, and their ability to mimic certain political parties' stances on various issues. However, the extent and conditions under which LLMs deviate from empirical positions have not been thoroughly examined. To address this gap, our study systematically investigates the factors contributing to LLMs' deviations from empirical positions on political issues, aiming to quantify these deviations and identify the conditions that cause them.
Drawing on cognitive science findings related to representativeness heuristics -- where individuals readily recall the representative attribute of a target group in a way that leads to exaggerated beliefs -- we scrutinize LLM responses through this heuristics lens. We conduct experiments to determine how LLMs exhibit stereotypes by inflating judgments in favor of specific political parties. Our results indicate that while LLMs can mimic certain political parties' positions, they often exaggerate these positions more than human respondents do. Notably, LLMs tend to overemphasize representativeness to a greater extent than humans. This study highlights the susceptibility of LLMs to representativeness heuristics, suggeseting potential vulnerabilities to political stereotypes. We propose prompt-based mitigation strategies that demonstrate effectiveness in reducing the influence of representativeness in LLM responses. 

**Abstract (ZH)**: 研究大型语言模型（LLMs）与人类意图和价值观的对齐变得越来越重要，特别是当这些系统未能按预期运行时。本研究探讨了如何使LLMs与人类意图和价值观保持一致，特别关注其政治倾向。先前的研究已指出，LLMs倾向于表现出政治倾向，并且能够模仿不同政治党派在诸多问题上的立场。然而，LLMs在多大程度上以及在何种条件下偏离实际立场尚未得到充分研究。为了弥补这一不足，本研究系统考察了导致LLMs在政治问题上偏离实际立场的各种因素，旨在量化这些偏差并识别导致它们的原因。

本研究借鉴了认知科学中代表性启发法的发现——即个体倾向于容易回忆目标群体的代表性特征，从而产生夸大的信念——通过启发法的视角审查LLMs的响应。我们通过实验来确定LLMs如何通过夸大人对特定政治党派的判断来表现出刻板印象。研究结果表明，尽管LLMs能够模仿某些政治党派的立场，但它们往往将这些立场夸大的程度超过了人类被试。值得注意的是，与人类相比，LLMs更容易过分强调代表性。本研究强调了LLMs对代表性启发法的敏感性，暗示它们可能存在政治刻板印象的潜在脆弱性。我们提出了基于提示的缓解策略，这些策略在减少LLMs响应中代表性影响的有效性方面表现出色。 

---
# A Comprehensive Framework for Semantic Similarity Detection Using Transformer Architectures and Enhanced Ensemble Techniques 

**Title (ZH)**: 使用Transformer架构和增强集成技术的语义相似度检测综合框架 

**Authors**: Lifu Gao, Qi Zhang, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14288)  

**Abstract**: Detecting AI-generated text, especially in short-context documents, is difficult because there is not enough context for accurate classification. This paper presents a new teacher-student model that uses domain adaptation and data augmentation to solve these problems. The teacher model, which combines DeBERTa-v3-large and Mamba-790m, learns semantic knowledge through domain-specific fine-tuning. The student model handles short-context text more efficiently. The system uses a Mean Squared Error (MSE) loss function to guide the student's learning, improving both accuracy and efficiency. Also, data augmentation methods like spelling correction and error injection make the model more robust. Experimental results show that this approach works better than baseline methods, proving its usefulness for real-time AI-generated text detection and other text classification tasks. 

**Abstract (ZH)**: 检测AI生成的文本，尤其是在短文档中，因其缺乏足够的上下文信息而难以进行准确分类。本文提出了一种新的教师-学生模型，该模型利用领域适应和数据增强技术解决这些问题。教师模型结合了DeBERTa-v3-large和Mamba-790m，通过领域特定微调学习语义知识。学生模型能够更有效地处理短文档中的文本。该系统采用均方误差（MSE）损失函数来指导学生的学习，提高了准确性和效率。此外，通过拼写校正和错误注入等数据增强方法，使模型更具健壮性。实验结果表明，该方法比基准方法更为有效，证明了其在实时检测AI生成文本和其他文本分类任务中的实用价值。 

---
# Leveraging Online Olympiad-Level Math Problems for LLMs Training and Contamination-Resistant Evaluation 

**Title (ZH)**: 利用在线奥林匹克级别数学问题进行大型语言模型训练和抗污染评估 

**Authors**: Sadegh Mahdavi, Muchen Li, Kaiwen Liu, Christos Thrampoulidis, Leonid Sigal, Renjie Liao  

**Link**: [PDF](https://arxiv.org/pdf/2501.14275)  

**Abstract**: Advances in Large Language Models (LLMs) have sparked interest in their ability to solve Olympiad-level math problems. However, the training and evaluation of these models are constrained by the limited size and quality of available datasets, as creating large-scale data for such advanced problems requires extensive effort from human experts. In addition, current benchmarks are prone to contamination, leading to unreliable evaluations. In this paper, we present an automated pipeline that leverages the rich resources of the Art of Problem Solving (AoPS) forum, which predominantly features Olympiad-level problems and community-driven solutions. Using open-source LLMs, we develop a method to extract question-answer pairs from the forum, resulting in AoPS-Instruct, a dataset of more than 600,000 high-quality QA pairs. Our experiments demonstrate that fine-tuning LLMs on AoPS-Instruct improves their reasoning abilities across various benchmarks. Moreover, we build an automatic pipeline that introduces LiveAoPSBench, an evolving evaluation set with timestamps, derived from the latest forum data, providing a contamination-resistant benchmark for assessing LLM performance. Notably, we observe a significant decline in LLM performance over time, suggesting their success on older examples may stem from pre-training exposure rather than true reasoning ability. Our work presents a scalable approach to creating and maintaining large-scale, high-quality datasets for advanced math reasoning, offering valuable insights into the capabilities and limitations of LLMs in this domain. Our benchmark and code is available at this https URL 

**Abstract (ZH)**: 大型语言模型（LLMs）的进步引发了对其解决奥林匹克级别数学问题能力的兴趣。然而，这些模型的训练和评估受到可用数据集规模有限和质量不高的限制，因为创造用于如此高等级问题的大规模数据需要大量的人工专家努力。此外，当前的基准易受污染，导致评估结果不可靠。在本文中，我们提出了一种自动化管道，利用艺术与解题论坛（AoPS）丰富的资源，该论坛主要涉及奥林匹克级别问题及其社区驱动的解决方案。利用开源LLM，我们开发了一种方法从论坛中提取问题-答案对，从而形成了包含超过60万高质量问题-答案对的AoPS-Instruct数据集。我们的实验表明，在AoPS-Instruct上微调LLM可以提高它们在各种基准测试中的推理能力。此外，我们构建了一种自动化管道，通过引入LiveAoPSBench（一个带有时间戳的不断演化的评估集），从最新的论坛数据中获得了一个污染抵抗基准，用于评估LLM的表现。值得注意的是，我们观察到LLM在时间上的表现显著下降，表明它们在较古老示例上的成功可能源于预训练曝光而非真正的推理能力。我们的工作提供了一种可扩展的方法，用于创建和维护大型高质量数据集，这些数据集适用于高级数学推理，并提供了关于LLM在这种领域的能力和局限性的宝贵见解。我们的基准测试和代码可以在以下链接访问：[此链接] 

---
# Siren: A Learning-Based Multi-Turn Attack Framework for Simulating Real-World Human Jailbreak Behaviors 

**Title (ZH)**: Siren：一种基于学习的多轮攻击框架，用于模拟真实世界的人类 Jailbreak 行为 

**Authors**: Yi Zhao, Youzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2501.14250)  

**Abstract**: Large language models (LLMs) are widely used in real-world applications, raising concerns about their safety and trustworthiness. While red-teaming with jailbreak prompts exposes the vulnerabilities of LLMs, current efforts focus primarily on single-turn attacks, overlooking the multi-turn strategies used by real-world adversaries. Existing multi-turn methods rely on static patterns or predefined logical chains, failing to account for the dynamic strategies during attacks. We propose Siren, a learning-based multi-turn attack framework designed to simulate real-world human jailbreak behaviors. Siren consists of three stages: (1) training set construction utilizing Turn-Level LLM feedback (Turn-MF), (2) post-training attackers with supervised fine-tuning (SFT) and direct preference optimization (DPO), and (3) interactions between the attacking and target LLMs. Experiments demonstrate that Siren achieves an attack success rate (ASR) of 90% with LLaMA-3-8B as the attacker against Gemini-1.5-Pro as the target model, and 70% with Mistral-7B against GPT-4o, significantly outperforming single-turn baselines. Moreover, Siren with a 7B-scale model achieves performance comparable to a multi-turn baseline that leverages GPT-4o as the attacker, while requiring fewer turns and employing decomposition strategies that are better semantically aligned with attack goals. We hope Siren inspires the development of stronger defenses against advanced multi-turn jailbreak attacks under realistic scenarios. Code is available at this https URL. Warning: This paper contains potentially harmful text. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在实际应用中广泛应用，引发了对其安全性和可信度的担忧。虽然通过牢笼破解提示进行红队测试可以揭示LLMs的漏洞，但当前的努力主要集中在单轮攻击上，未能充分考虑现实世界对手采用的多轮策略。现有的多轮方法依赖于静态模式或预定义逻辑链，无法充分考虑到攻击过程中的动态策略。我们提出了一种名为Siren的学习型多轮攻击框架，旨在模拟现实世界中人类进行牢笼破解的行为。Siren包含三个阶段：（1）利用基于轮次的语言模型反馈（Turn-Level LLM Feedback, Turn-MF）构建训练集；（2）通过监督微调（Supervised Fine-Tuning, SFT）和直接偏好优化（Direct Policy Optimization, DPO）进行后训练攻击者；（3）攻击者与目标LLM之间的交互。实验结果显示，使用Siren攻击Gemini-1.5-Pro（作为目标模型）时，针对LLaMA-3-8B的攻击成功率（Attack Success Rate, ASR）为90%，而攻击GPT-4o（ Mistral-7B 作为目标模型）时，ASR为70%，明显优于单轮基线。此外，使用Siren的7B规模模型在性能上与利用GPT-4o进行攻击的多轮基线相当，但所需的轮次更少，并采用了更具语义上与攻击目标相符的分解策略。我们希望Siren能够激发对实际场景中更强大的多轮牢笼破解攻击防御方案的研究与开发。更多信息请参阅此处该链接。警告：本文可能包含潜在有害的文字。 

---
# Multi-agent KTO: Reinforcing Strategic Interactions of Large Language Model in Language Game 

**Title (ZH)**: 多智能体KTO：增强大型语言模型在语言博弈中的战略互动 

**Authors**: Rong Ye, Yongxin Zhang, Yikai Zhang, Haoyu Kuang, Zhongyu Wei, Peng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.14225)  

**Abstract**: Achieving Artificial General Intelligence (AGI) requires AI agents that can not only make stratigic decisions but also engage in flexible and meaningful communication. Inspired by Wittgenstein's language game theory in Philosophical Investigations, we propose that language agents can learn through in-context interaction rather than traditional multi-stage frameworks that separate decision-making from language expression. Using Werewolf, a social deduction game that tests language understanding, strategic interaction, and adaptability, we develop the Multi-agent Kahneman & Tversky's Optimization (MaKTO). MaKTO engages diverse models in extensive gameplay to generate unpaired desirable and unacceptable responses, then employs KTO to refine the model's decision-making process. In 9-player Werewolf games, MaKTO achieves a 61% average win rate across various models, outperforming GPT-4o and two-stage RL agents by relative improvements of 23.0% and 10.9%, respectively. Notably, MaKTO also demonstrates human-like performance, winning 60% against expert players and showing only 49% detectability in Turing-style blind tests. These results showcase MaKTO's superior decision-making, strategic adaptation, and natural language generation in complex social deduction games. 

**Abstract (ZH)**: 实现人工通用智能（AGI）需要能够做出战略决策并且能够进行灵活和有意义交流的AI代理。受到《哲学探究》中维特根斯坦的语言游戏理论的启发，我们认为语言代理可以通过上下文交互学习，而不是传统的分阶段框架，这些框架将决策过程与语言表达分离。我们通过社会推理游戏狼人杀（Werewolf）——一种测试语言理解、战略互动和适应性的游戏——开发了多代理坎布纳姆与特维斯基优化（Multi-agent Kahneman & Tversky's Optimization, MaKTO）。MaKTO使多种模型在广泛的游戏互动中进行交流，生成不匹配的可取和不可取回应，然后利用KTO改进模型的决策过程。在9人的狼人杀游戏中，MaKTO在各种模型中平均获得61%的胜率，分别比GPT-4o和两阶段的强化学习代理高出23.0%和10.9%。特别值得注意的是，MaKTO还表现出类似人类的表现，在与专家玩家的较量中赢得了60%的比赛，并且在模仿图灵测试中仅被检测出49%。这些结果展示了MaKTO在复杂社会推理游戏中的卓越决策能力、战略适应性和自然语言生成能力。 

---
# Test-Time Code-Switching for Cross-lingual Aspect Sentiment Triplet Extraction 

**Title (ZH)**: 跨语言方面情感三元组提取中的测试时代码转换 

**Authors**: Dongming Sheng, Kexin Han, Hao Li, Yan Zhang, Yucheng Huang, Jun Lang, Wenqiang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14144)  

**Abstract**: Aspect Sentiment Triplet Extraction (ASTE) is a thriving research area with impressive outcomes being achieved on high-resource languages. However, the application of cross-lingual transfer to the ASTE task has been relatively unexplored, and current code-switching methods still suffer from term boundary detection issues and out-of-dictionary problems. In this study, we introduce a novel Test-Time Code-SWitching (TT-CSW) framework, which bridges the gap between the bilingual training phase and the monolingual test-time prediction. During training, a generative model is developed based on bilingual code-switched training data and can produce bilingual ASTE triplets for bilingual inputs. In the testing stage, we employ an alignment-based code-switching technique for test-time augmentation. Extensive experiments on cross-lingual ASTE datasets validate the effectiveness of our proposed method. We achieve an average improvement of 3.7% in terms of weighted-averaged F1 in four datasets with different languages. Additionally, we set a benchmark using ChatGPT and GPT-4, and demonstrate that even smaller generative models fine-tuned with our proposed TT-CSW framework surpass ChatGPT and GPT-4 by 14.2% and 5.0% respectively. 

**Abstract (ZH)**: 情感方面 triplet 提取 (ASTE) 是一个蓬勃发展的研究领域，已经在资源丰富语言上取得了显著成果。然而，跨语言迁移在 ASTE 任务中的应用相对较少被探索，且当前的代码混用方法仍然存在词边界检测问题和词表之外的问题。在本研究中，我们引入了一个新颖的测试时代码混用 (TT-CSW) 框架，该框架在双语训练阶段和单语测试时预测之间建立了桥梁。在训练阶段，基于双语代码混用训练数据开发了一个生成模型，能够为双语输入生成双语ASTE三元组。在测试阶段，我们采用基于对齐的代码混用技术进行测试时增强。在跨语言 ASTE 数据集上的广泛实验验证了我们提出方法的有效性。我们在四个不同语言的数据集上实现了平均 3.7% 的加权 F1 值改进。此外，我们在 ChatGPT 和 GPT-4 上设立了一个基准测试，并且展示了即使使用我们提出的 TT-CSW 框架微调的较小生成模型也分别在 ChatGPT 和 GPT-4 上取得了 14.2% 和 5.0% 的显著性能提升。 

---
# Autonomous Structural Memory Manipulation for Large Language Models Using Hierarchical Embedding Augmentation 

**Title (ZH)**: 使用分层嵌入增强进行自主结构内存操作的大语言模型 

**Authors**: Derek Yotheringhay, Alistair Kirkland, Humphrey Kirkbride, Josiah Whitesteeple  

**Link**: [PDF](https://arxiv.org/pdf/2501.14119)  

**Abstract**: Transformative innovations in model architectures have introduced hierarchical embedding augmentation as a means to redefine the representation of tokens through multi-level semantic structures, offering enhanced adaptability to complex linguistic inputs. Autonomous structural memory manipulation further advances this paradigm through dynamic memory reallocation mechanisms that prioritize critical contextual features while suppressing less relevant information, enabling scalable and efficient performance across diverse tasks. Experimental results reveal substantial improvements in computational efficiency, with marked reductions in processing overhead for longer input sequences, achieved through memory reorganization strategies that adapt to evolving contextual requirements. Hierarchical embeddings not only improved contextual alignment but also facilitated task generalization by capturing relationships at varying semantic granularities, ensuring coherence across layers without introducing significant computational redundancies. Comparative analysis against baseline models demonstrated unique advantages in accuracy, efficiency, and interpretability, particularly in tasks requiring complex contextual understanding or domain-specific adaptability. The ability to dynamically adjust token representations and memory configurations contributed to the model's robustness under varied and unpredictable input conditions. Applications benefiting from these advancements include multi-domain generalization, interactive systems, and scenarios involving real-time decision-making, where traditional static memory architectures often face limitations. The proposed methodology combines advanced embedding and memory management strategies into a cohesive framework that addresses scalability challenges while preserving task-specific relevance. 

**Abstract (ZH)**: 模型架构的变革性创新引入了层次嵌入增强机制，通过多层次语义结构重新定义了标记的表示，从而增强了对复杂语言输入的适应性。自主结构记忆操控在此基础上进一步推进了这一范式，通过动态的内存重分配机制优先处理关键上下文特征，同时抑制无关信息，从而实现跨多种任务的可扩展和高效性能。实验结果表明，通过适应性地重新组织内存结构，计算效率有显著提升，尤其是在较长输入序列的处理中，处理开销显著减少。层次嵌入不仅提高了语境对齐性，还通过捕捉不同语义粒度的关系，促进了任务泛化能力，同时确保了跨层的一致性且没有引入显著的计算冗余。与基准模型的对比分析显示，该方法在准确度、效率和可解释性方面具有独特优势，特别是在需要复杂上下文理解或特定域适应的任务中表现尤为突出。动态调整标记表示和内存配置的能力，使该模型在各种不确定输入条件下具有更高的鲁棒性。这些进展的应用范围包括多域泛化、交互系统和实时决策场景，而在这些应用场景中，传统的静态内存架构常常面临局限性。所提出的方法将先进的嵌入技术和内存管理策略整合到一个一致的框架中，解决了可扩展性挑战，同时保留了任务特定的相关性。 

---
# LeCoPCR: Legal Concept-guided Prior Case Retrieval for European Court of Human Rights cases 

**Title (ZH)**: LeCoPCR：指导性法律概念引导的先行案例检索方法——应用于欧洲人权法院案件 

**Authors**: T.Y.S.S. Santosh, Isaac Misael Olguín Nolasco, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2501.14114)  

**Abstract**: Prior case retrieval (PCR) is crucial for legal practitioners to find relevant precedent cases given the facts of a query case. Existing approaches often overlook the underlying semantic intent in determining relevance with respect to the query case. In this work, we propose LeCoPCR, a novel approach that explicitly generate intents in the form of legal concepts from a given query case facts and then augments the query with these concepts to enhance models understanding of semantic intent that dictates relavance. To overcome the unavailability of annotated legal concepts, we employ a weak supervision approach to extract key legal concepts from the reasoning section using Determinantal Point Process (DPP) to balance quality and diversity. Experimental results on the ECtHR-PCR dataset demonstrate the effectiveness of leveraging legal concepts and DPP-based key concept extraction. 

**Abstract (ZH)**: 优先案例检索（Prior Case Retrieval, PCR）对于法律从业者根据查询案件的事实找到相关的先例案件至关重要。现有方法往往会忽视在确定相关性时查询案件背后的意义意图。在本文中，我们提出了一种名为LeCoPCR的新颖方法，该方法旨在从给定的查询案件事实中显式地生成法律概念，并将这些概念增强到查询中，以提高模型对意义意图的理解，该意图决定了相关性。为克服注释法律概念的缺失，我们采用了一种弱监督方法，使用行列式点过程（Determinantal Point Process, DPP）从推理部分中提取关键法律概念，以平衡质量和多样性。在ECtHR-PCR数据集上的实验结果表明，利用法律概念和基于DPP的关键概念提取方法的有效性。 

---
# RELexED: Retrieval-Enhanced Legal Summarization with Exemplar Diversity 

**Title (ZH)**: RELexED：基于范例多样性增强的检索增强法律摘要生成 

**Authors**: T.Y.S.S. Santosh, Chen Jia, Patrick Goroncy, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2501.14113)  

**Abstract**: This paper addresses the task of legal summarization, which involves distilling complex legal documents into concise, coherent summaries. Current approaches often struggle with content theme deviation and inconsistent writing styles due to their reliance solely on source documents. We propose RELexED, a retrieval-augmented framework that utilizes exemplar summaries along with the source document to guide the model. RELexED employs a two-stage exemplar selection strategy, leveraging a determinantal point process to balance the trade-off between similarity of exemplars to the query and diversity among exemplars, with scores computed via influence functions. Experimental results on two legal summarization datasets demonstrate that RELexED significantly outperforms models that do not utilize exemplars and those that rely solely on similarity-based exemplar selection. 

**Abstract (ZH)**: 本文探讨了法律总结的任务，即将复杂的法律文件提炼为简洁且连贯的摘要。现有的方法往往因为仅依赖原始文档而面临内容主题偏移和不一致写作风格的问题。我们提出了一种名为RELexED的检索增强框架，该框架利用示例摘要和原始文档来指导模型。RELexED采用两阶段的示例选择策略，利用确定性点过程在示例与查询之间的相似性和示例之间的多样性之间进行权衡，并通过影响函数计算得分。在两个法律总结数据集上的实验结果表明，RELexED在不使用示例和仅依赖基于相似性的示例选择的模型中表现显著更好。 

---
# CoPERLex: Content Planning with Event-based Representations for Legal Case Summarization 

**Title (ZH)**: CoPERLex：基于事件表示的内容规划法在法律案例摘要生成中的应用 

**Authors**: T.Y.S.S. Santosh, Youssef Farag, Matthias Grabmair  

**Link**: [PDF](https://arxiv.org/pdf/2501.14112)  

**Abstract**: Legal professionals often struggle with lengthy judgments and require efficient summarization for quick comprehension. To address this challenge, we investigate the need for structured planning in legal case summarization, particularly through event-centric representations that reflect the narrative nature of legal case documents. We propose our framework, CoPERLex, which operates in three stages: first, it performs content selection to identify crucial information from the judgment; second, the selected content is utilized to generate intermediate plans through event-centric representations modeled as Subject-Verb-Object tuples; and finally, it generates coherent summaries based on both the content and the structured plan. Our experiments on four legal summarization datasets demonstrate the effectiveness of integrating content selection and planning components, highlighting the advantages of event-centric plans over traditional entity-centric approaches in the context of legal judgements. 

**Abstract (ZH)**: 法律专业人士经常面临着冗长判决的挑战，需要高效的摘要以便快速理解。为应对这一挑战，我们探讨了结构化规划在法律案例摘要中的需求，特别关注基于事件的表示，这些表示反映了法律案例文档的叙述性质。我们提出了一个框架CoPERLex，该框架分为三个阶段：首先，它执行内容选择，以从判决中识别关键信息；其次，选择的内容用于生成通过事件为中心的表示生成中间计划，这些表示被建模为主语-谓语-宾语三元组；最后，基于内容和结构化计划生成连贯的摘要。我们在四个法律摘要数据集上的实验证明了整合内容选择和规划组件的有效性，突显了在法律判决中事件为中心的计划相较于传统实体为中心的方法的优势。 

---
# MedSlice: Fine-Tuned Large Language Models for Secure Clinical Note Sectioning 

**Title (ZH)**: MedSlice：用于安全临床笔记分段的大型语言模型微调 

**Authors**: Joshua Davis, Thomas Sounack, Kate Sciacca, Jessie M Brain, Brigitte N Durieux, Nicole D Agaronnik, Charlotta Lindvall  

**Link**: [PDF](https://arxiv.org/pdf/2501.14105)  

**Abstract**: Extracting sections from clinical notes is crucial for downstream analysis but is challenging due to variability in formatting and labor-intensive nature of manual sectioning. While proprietary large language models (LLMs) have shown promise, privacy concerns limit their accessibility. This study develops a pipeline for automated note sectioning using open-source LLMs, focusing on three sections: History of Present Illness, Interval History, and Assessment and Plan. We fine-tuned three open-source LLMs to extract sections using a curated dataset of 487 progress notes, comparing results relative to proprietary models (GPT-4o, GPT-4o mini). Internal and external validity were assessed via precision, recall and F1 score. Fine-tuned Llama 3.1 8B outperformed GPT-4o (F1=0.92). On the external validity test set, performance remained high (F1= 0.85). Fine-tuned open-source LLMs can surpass proprietary models in clinical note sectioning, offering advantages in cost, performance, and accessibility. 

**Abstract (ZH)**: 从临床笔记中提取段落对于下游分析至关重要，但由于格式的多样性以及手动分段的劳动密集性，这一任务具有挑战性。虽然专有的大型语言模型（LLMs）显示出潜力，但隐私问题限制了它们的可访问性。本研究开发了一种使用开源LLMs的自动化笔记分段管道，重点关注三个段落：现病史、间诊史和评估与计划。我们使用一个包含487份病程记录的精标数据集，对三种开源LLMs进行了微调，以提取这些段落，并将结果与专有模型（GPT-4o、GPT-4o mini）进行了比较。内部和外部有效性通过精确度、召回率和F1分数进行了评估。微调后的Llama 3.1 8B在F1分数方面优于GPT-4o（F1=0.92）。在外在有效性测试集中，性能依然很高（F1=0.85）。微调后的开源LLMs在临床笔记分段方面可以超越专有模型，提供了成本、性能和可访问性方面的优势。 

---
# Communicating Activations Between Language Model Agents 

**Title (ZH)**: 语言模型代理之间的激活通信 

**Authors**: Vignav Ramesh, Kenneth Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.14082)  

**Abstract**: Communication between multiple language model (LM) agents has been shown to scale up the reasoning ability of LMs. While natural language has been the dominant medium for inter-LM communication, it is not obvious this should be the standard: not only does natural language communication incur high inference costs that scale quickly with the number of both agents and messages, but also the decoding process abstracts away too much rich information that could be otherwise accessed from the internal activations. In this work, we propose a simple technique whereby LMs communicate via activations; concretely, we pause an LM $\textit{B}$'s computation at an intermediate layer, combine its current activation with another LM $\textit{A}$'s intermediate activation via some function $\textit{f}$, then pass $\textit{f}$'s output into the next layer of $\textit{B}$ and continue the forward pass till decoding is complete. This approach scales up LMs on new tasks with zero additional parameters and data, and saves a substantial amount of compute over natural language communication. We test our method with various functional forms $\textit{f}$ on two experimental setups--multi-player coordination games and reasoning benchmarks--and find that it achieves up to $27.0\%$ improvement over natural language communication across datasets with $<$$1/4$ the compute, illustrating the superiority and robustness of activations as an alternative "language" for communication between LMs. 

**Abstract (ZH)**: 多语言模型（LM）代理之间的交流已被证明可以提升语言模型的推理能力。虽然自然语言一直是主要的跨语言模型交流媒介，但这并不意味着必须如此：自然语言交流会带来随着代理和信息量增加而迅速增长的高昂推理成本，此外，解码过程会丢失过多原本可以从内部激活中获取的丰富信息。在本文中，我们提出了一种简单的技术，即通过激活进行语言模型之间的交流；具体来说，我们暂停语言模型B在中间层的计算，将它的当前激活与另一个语言模型A的中间激活通过某种函数f结合在一起，然后将f的输出传递给B的下一隐藏层，并继续前向传播直到完成解码。这种方法可以在无需额外参数和数据的情况下提升语言模型在新任务上的性能，并显著减少与自然语言交流相比所需的计算量。我们通过使用不同形式的函数f，在多个实验设置——多玩家协调博弈和推理基准测试中测试了这种方法，并发现在少于四分之一的计算量下，其在多个数据集上达到了高达27.0%的改进，证明了激活作为一种交流语言的优越性和鲁棒性。 

---
# Enhancing Biomedical Relation Extraction with Directionality 

**Title (ZH)**: 增强医学关系抽取的方向性方法 

**Authors**: Po-Ting Lai, Chih-Hsuan Wei, Shubo Tian, Robert Leaman, Zhiyong Lu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14079)  

**Abstract**: Biological relation networks contain rich information for understanding the biological mechanisms behind the relationship of entities such as genes, proteins, diseases, and chemicals. The vast growth of biomedical literature poses significant challenges updating the network knowledge. The recent Biomedical Relation Extraction Dataset (BioRED) provides valuable manual annotations, facilitating the develop-ment of machine-learning and pre-trained language model approaches for automatically identifying novel document-level (inter-sentence context) relationships. Nonetheless, its annotations lack directionality (subject/object) for the entity roles, essential for studying complex biological networks. Herein we annotate the entity roles of the relationships in the BioRED corpus and subsequently propose a novel multi-task language model with soft-prompt learning to jointly identify the relationship, novel findings, and entity roles. Our results in-clude an enriched BioRED corpus with 10,864 directionality annotations. Moreover, our proposed method outperforms existing large language models such as the state-of-the-art GPT-4 and Llama-3 on two benchmarking tasks. Our source code and dataset are available at this https URL. 

**Abstract (ZH)**: 生物关系网络包含了理解基因、蛋白质、疾病和化学物质之间关系背后的生物机制的丰富信息。生物医学文献的快速增长对更新网络知识构成了重大挑战。近期发布的Biomedical Relation Extraction Dataset (BioRED) 提供了宝贵的手动标注数据，促进了基于机器学习和预训练语言模型的方法发展，以自动识别文档级别（跨句子上下文）的新颖关系。然而，其标注缺乏实体角色的方向性信息，这对于研究复杂的生物网络至关重要。在此，我们对BioRED语料库中的实体角色进行了标注，并随后提出了一种新的基于软提示学习的多任务语言模型，以联合识别关系、新颖发现和实体角色。我们的结果包括了一个包含10,864个方向性标注的丰富BioRED语料库。此外，我们提出的方法在两个基准任务上的表现优于现有的大型语言模型，如最先进的GPT-4和Llama-3。我们的源代码和数据集可在以下链接获取：[请提供具体链接]。 

---
# LLMs are Vulnerable to Malicious Prompts Disguised as Scientific Language 

**Title (ZH)**: 大型语言模型容易受到伪装成科学语言的恶意提示攻击 

**Authors**: Yubin Ge, Neeraja Kirtane, Hao Peng, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2501.14073)  

**Abstract**: As large language models (LLMs) have been deployed in various real-world settings, concerns about the harm they may propagate have grown. Various jailbreaking techniques have been developed to expose the vulnerabilities of these models and improve their safety. This work reveals that many state-of-the-art proprietary and open-source LLMs are vulnerable to malicious requests hidden behind scientific language. Specifically, our experiments with GPT4o, GPT4o-mini, GPT-4, LLama3-405B-Instruct, Llama3-70B-Instruct, Cohere, Gemini models on the StereoSet data demonstrate that, the models' biases and toxicity substantially increase when prompted with requests that deliberately misinterpret social science and psychological studies as evidence supporting the benefits of stereotypical biases. Alarmingly, these models can also be manipulated to generate fabricated scientific arguments claiming that biases are beneficial, which can be used by ill-intended actors to systematically jailbreak even the strongest models like GPT. Our analysis studies various factors that contribute to the models' vulnerabilities to malicious requests in academic language. Mentioning author names and venues enhances the persuasiveness of some models, and the bias scores can increase as dialogues progress. Our findings call for a more careful investigation on the use of scientific data in the training of LLMs. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在各种实际应用场景中的部署，人们对它们可能传播的危害的担忧日益增加。各种“监狱突破”技术已被开发出来，以揭示这些模型的漏洞并提高其安全性。本研究发现，许多最先进的专有和开源LLMs在遇到故意误用社会科学研究和心理学研究作为支持刻板偏见益处证据的恶意请求时，其偏见和毒性显著增加。令人警觉的是，这些模型还可以被操纵生成虚假的科学论点，声称偏见是有益的，这可能被恶意行为者用来系统地突破即使是最强的模型如GPT。我们的分析研究了在学术语言中导致模型对恶意请求脆弱的各种因素。提及作者姓名和会议信息可以增强某些模型的说服力，而偏见得分可能会随着对话的进展而增加。我们的研究结果呼吁对LLMs在训练中使用科学数据的方式进行更仔细的研究。 

---
# Leveraging Large Language Models to Analyze Emotional and Contextual Drivers of Teen Substance Use in Online Discussions 

**Title (ZH)**: 利用大型语言模型分析青少年在线讨论中物质使用的情感和情境驱动因素 

**Authors**: Jianfeng Zhu, Ruoming Jin, Hailong Jiang, Yulan Wang, Xinyu Zhang, Karin G. Coifman  

**Link**: [PDF](https://arxiv.org/pdf/2501.14037)  

**Abstract**: Adolescence is a critical stage often linked to risky behaviors, including substance use, with significant developmental and public health implications. Social media provides a lens into adolescent self-expression, but interpreting emotional and contextual signals remains complex. This study applies Large Language Models (LLMs) to analyze adolescents' social media posts, uncovering emotional patterns (e.g., sadness, guilt, fear, joy) and contextual factors (e.g., family, peers, school) related to substance use. Heatmap and machine learning analyses identified key predictors of substance use-related posts. Negative emotions like sadness and guilt were significantly more frequent in substance use contexts, with guilt acting as a protective factor, while shame and peer influence heightened substance use risk. Joy was more common in non-substance use discussions. Peer influence correlated strongly with sadness, fear, and disgust, while family and school environments aligned with non-substance use. Findings underscore the importance of addressing emotional vulnerabilities and contextual influences, suggesting that collaborative interventions involving families, schools, and communities can reduce risk factors and foster healthier adolescent development. 

**Abstract (ZH)**: 青春期是一个关键的发展阶段，常常与风险行为相关，包括物质滥用，这对其个体发展和公共卫生具有重大影响。社交媒体为了解青少年的自我表达提供了窗口，但解读情感和情境信号仍然复杂。本研究应用大规模语言模型（LLMs）分析青少年的社交媒体帖子，揭示与物质滥用相关的情感模式（例如，悲伤、内疚、恐惧、快乐）和情境因素（例如，家庭、同伴、学校）。通过热图和机器学习分析，确定了与物质滥用相关的帖子的关键预测因素。负向情感如悲伤和内疚在物质滥用情境中更为常见，其中内疚起到了保护性作用，而羞愧和同伴影响则提高了物质滥用的风险。快乐在非物质滥用讨论中更为常见。同伴影响与悲伤、恐惧和厌恶相关，而家庭和学校环境则与非物质滥用相符。研究结果强调了关注情感脆弱性和情境因素的重要性，表明家庭、学校和社区之间的合作干预措施可以降低风险因素，促进更健康的青少年发展。 

---
# Advancing Math Reasoning in Language Models: The Impact of Problem-Solving Data, Data Synthesis Methods, and Training Stages 

**Title (ZH)**: 提升语言模型的数学推理能力：问题解决数据、数据合成方法及训练阶段的影响研究 

**Authors**: Zui Chen, Tianqiao Liu, Mi Tian, Qing Tong, Weiqi Luo, Zitao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14002)  

**Abstract**: Advancements in LLMs have significantly expanded their capabilities across various domains. However, mathematical reasoning remains a challenging area, prompting the development of math-specific LLMs. These models typically follow a two-stage training paradigm: pre-training with math-related corpora and post-training with problem datasets for SFT. Despite these efforts, the improvements in mathematical reasoning achieved through continued pre-training (CPT) are often less significant compared to those obtained via SFT. This study addresses this discrepancy by exploring alternative strategies during the pre-training phase, focusing on the use of problem-solving data over general mathematical corpora. We investigate three primary research questions: (1) Can problem-solving data enhance the model's mathematical reasoning capabilities more effectively than general mathematical corpora during CPT? (2) Are synthetic data from the same source equally effective, and which synthesis methods are most efficient? (3) How do the capabilities developed from the same problem-solving data differ between the CPT and SFT stages, and what factors contribute to these differences? Our findings indicate that problem-solving data significantly enhances the model's mathematical capabilities compared to general mathematical corpora. We also identify effective data synthesis methods, demonstrating that the tutorship amplification synthesis method achieves the best performance. Furthermore, while SFT facilitates instruction-following abilities, it underperforms compared to CPT with the same data, which can be partially attributed to its poor learning capacity for hard multi-step problem-solving data. These insights provide valuable guidance for optimizing the mathematical reasoning capabilities of LLMs, culminating in our development of a powerful mathematical base model called JiuZhang-8B. 

**Abstract (ZH)**: 大规模语言模型（LLM）的进步显著扩展了它们在各个领域的功能。然而，数学推理仍然是一个具有挑战性的领域，推动了专门的数学LLM的发展。这些模型通常采用两阶段训练范式：使用与数学相关的语料库进行预训练，然后使用问题数据集进行监督 fine-tuning（SFT）。尽管如此，通过持续预训练（CPT）获得的数学推理改进往往不如通过SFT获得的改进显著。本研究通过探索预训练阶段的替代策略，重点使用问题解决数据而非一般数学语料库，旨在解决这一差异。我们调查了三个主要的研究问题：（1）在CPT过程中，问题解决数据是否比一般数学语料库更有效地提升模型的数学推理能力？（2）相同来源的合成数据是否同样有效，哪些合成方法最有效？（3）相同问题解决数据在CPT阶段与SFT阶段产生的能力有何不同，是什么因素导致了这些差异？研究结果表明，问题解决数据显著提升了模型的数学能力，优于一般数学语料库。我们还确定了有效的数据合成方法，指出教师扩展示方法在性能上表现最佳。此外，虽然SFT有助于提升执行指令的能力，但在使用相同数据时，其表现不如CPT阶段，部分原因在于其在解决复杂多步问题数据的较低学习能力。这些洞察为优化LLM的数学推理能力提供了宝贵指导，最终促使我们开发了强大的数学基础模型——九章-8B。 

---
# Framework for Progressive Knowledge Fusion in Large Language Models Through Structured Conceptual Redundancy Analysis 

**Title (ZH)**: 通过结构化概念冗余分析在大型语言模型中实现渐进式知识融合的框架 

**Authors**: Joseph Sakau, Evander Kozlowski, Roderick Thistledown, Basil Steinberger  

**Link**: [PDF](https://arxiv.org/pdf/2501.13999)  

**Abstract**: The organization of latent knowledge within large-scale models poses unique challenges when addressing overlapping representations and optimizing contextual accuracy. Conceptual redundancies embedded across layers often result in inefficiencies that affect both computational demands and task-specific outcomes. A framework was proposed to restructure these redundancies through advanced clustering techniques and dynamic thresholding, ensuring that critical semantic relationships are preserved while removing unnecessary overlaps. Evaluations revealed improved memory efficiency and faster inference times, alongside better alignment in latent knowledge clusters that enhanced interpretability. Improvements in error rates and adversarial robustness suggest that restructuring redundancies has broader implications for increasing model reliability across diverse applications. Comparative analyses highlighted reductions in resource consumption and notable gains in performance, particularly in translation and summarization tasks. Energy metrics demonstrated significant savings during training phases, further validating the practicality of the approach for real-world deployments. Representational fidelity was also enhanced, with latent space evaluations indicating better cluster alignment and higher semantic consistency. The methodology bridges a key gap in model optimization through directly addressing redundancies at the structural level. Its application opens avenues for scalable, efficient, and contextually aware systems that can adapt to complex, domain-specific tasks without compromising on performance. 

**Abstract (ZH)**: 大规模模型中潜在知识的组织在处理重叠表示和优化上下文准确性时提出了独特的挑战。嵌入多层的概念冗余往往导致效率低下，影响计算需求和特定任务的结果。提出了一种框架，通过先进的聚类技术和动态阈值来重新组织这些冗余，确保保留关键的语义关系的同时移除不必要的重叠。评估结果显示，这种框架提高了内存效率，加快了推理时间，并增强了潜在知识集群的一致性，从而提升了解释性。错误率和对抗鲁棒性的改善表明，重新组织冗余对于提高模型在多种应用中的可靠性具有更广泛的影响。对比分析显示，资源消耗减少，特别是在翻译和摘要任务中获得了显著性能提升。能耗指标表明，在训练阶段具有显著的节能效果，进一步验证了该方法在实际部署中的实用性。潜在空间评估表明，表征保真度也得到了增强，表现为更好的簇对齐和更高的语义一致性。该方法通过直接在结构层面解决冗余问题，填补了模型优化中的关键空白。其应用为开发可扩展、高效且上下文感知的系统开启了途径，这些系统可以在不牺牲性能的情况下适应复杂的领域特定任务。 

---
# CAPRAG: A Large Language Model Solution for Customer Service and Automatic Reporting using Vector and Graph Retrieval-Augmented Generation 

**Title (ZH)**: CAPRAG：一种基于向量和图检索增强生成的大规模语言模型解决方案，用于客户服务和自动报告生成 

**Authors**: Hamza Landolsi, Kais Letaief, Nizar Taghouti, Ines Abdeljaoued-Tej  

**Link**: [PDF](https://arxiv.org/pdf/2501.13993)  

**Abstract**: The introduction of new features and services in the banking sector often overwhelms customers, creating an opportunity for banks to enhance user experience through financial chatbots powered by large language models (LLMs). We initiated an AI agent designed to provide customers with relevant information about banking services and insights from annual reports. We proposed a hybrid Customer Analysis Pipeline Retrieval-Augmented Generation (CAPRAG) that effectively addresses both relationship-based and contextual queries, thereby improving customer engagement in the digital banking landscape. To implement this, we developed a processing pipeline to refine text data, which we utilized in two main frameworks: Vector RAG and Graph RAG. This dual approach enables us to populate both vector and graph databases with processed data for efficient retrieval. The Cypher query component is employed to effectively query the graph database. When a user submits a query, it is first expanded by a query expansion module before being routed to construct a final query from the hybrid Knowledge Base (KB). This final query is then sent to an open-source LLM for response generation. Overall, our innovative, designed to international banks, serves bank's customers in an increasingly complex digital environment, enhancing clarity and accessibility of information. 

**Abstract (ZH)**: 银行部门引入新功能和服务往往会让客户不知所措，这为银行通过大语言模型（LLMs）驱动的金融聊天机器人改善用户体验提供了机会。我们设计了一个AI代理，旨在为客户提供与银行服务相关的信息以及年度报告中的见解。我们提出了一种混合客户分析管道检索增强生成（CAPRAG）方法，该方法能够有效应对基于关系和上下文的查询，从而在数字银行领域提升客户参与度。为了实现这一目标，我们开发了一个数据处理管道，以精炼文本数据，并利用两个主要框架：向量RAG和图RAG。这种双重方法使我们能够高效地将处理后的数据填充到向量和图数据库中。我们使用Cypher查询组件来高效查询图数据库。当用户提交查询时，查询首先通过查询扩展模块进行扩展，然后被路由构建最终查询，该查询来自混合知识库（KB）。最终查询随后发送给开源大语言模型以生成响应。总体而言，我们的创新方法旨在帮助国际银行增强其客户在日益复杂数字环境中的体验，提高信息的清晰度和可访问性。 

---
# Comprehensive Modeling and Question Answering of Cancer Clinical Practice Guidelines using LLMs 

**Title (ZH)**: 使用大规模语言模型对癌症临床实践指南进行全面建模与问答研究 

**Authors**: Bhumika Gupta, Pralaypati Ta, Keerthi Ram, Mohanasankar Sivaprakasam  

**Link**: [PDF](https://arxiv.org/pdf/2501.13984)  

**Abstract**: The updated recommendations on diagnostic procedures and treatment pathways for a medical condition are documented as graphical flows in Clinical Practice Guidelines (CPGs). For effective use of the CPGs in helping medical professionals in the treatment decision process, it is necessary to fully capture the guideline knowledge, particularly the contexts and their relationships in the graph. While several existing works have utilized these guidelines to create rule bases for Clinical Decision Support Systems, limited work has been done toward directly capturing the full medical knowledge contained in CPGs. This work proposes an approach to create a contextually enriched, faithful digital representation of National Comprehensive Cancer Network (NCCN) Cancer CPGs in the form of graphs using automated extraction and node & relationship classification. We also implement semantic enrichment of the model by using Large Language Models (LLMs) for node classification, achieving an accuracy of 80.86% and 88.47% with zero-shot learning and few-shot learning, respectively. Additionally, we introduce a methodology for answering natural language questions with constraints to guideline text by leveraging LLMs to extract the relevant subgraph from the guideline knowledge base. By generating natural language answers based on subgraph paths and semantic information, we mitigate the risk of incorrect answers and hallucination associated with LLMs, ensuring factual accuracy in medical domain Question Answering. 

**Abstract (ZH)**: 医学条件诊断程序和治疗路径的更新建议在临床实践指南（CPGs）中以图形流程的形式记录。为了有效利用CPGs帮助医疗专业人员进行治疗决策过程，有必要全面捕捉指导原则的知识，尤其是图中的上下文及其关系。虽然已有不少研究利用这些指南为临床决策支持系统创建规则库，但直接从CPGs中提取完整医学知识的工作相对较少。本研究提出了一种利用自动化提取和节点与关系分类，创建富含上下文并忠实于原始内容的NCCN癌症CPGs的图形表示的方法。我们还通过使用大规模语言模型（LLMs）对模型节点进行语义丰富化，在零样本学习和少样本学习中分别实现了80.86%和88.47%的分类精度。此外，我们通过对指导原则文本的利用LLMs提取相关子图的方法，提出了一个基于约束的利用自然语言问题解答的方法。通过基于子图路径和语义信息生成自然语言答案，我们减轻了语言模型可能产生的不准确答案和幻觉风险，确保了医学领域问答的准确性。 

---
# AdEval: Alignment-based Dynamic Evaluation to Mitigate Data Contamination in Large Language Models 

**Title (ZH)**: AdEval：基于对齐的动态评估方法以减轻大型语言模型中的数据污染问题 

**Authors**: Yang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2501.13983)  

**Abstract**: As Large Language Models (LLMs) are pretrained on massive-scale corpora, the issue of data contamination has become increasingly severe, leading to potential overestimation of model performance during evaluation. To address this, we propose AdEval (Alignment-based Dynamic Evaluation), a dynamic data evaluation method aimed at mitigating the impact of data contamination on evaluation reliability. AdEval extracts key knowledge points and main ideas to align dynamically generated questions with static data's core concepts. It also leverages online search to provide detailed explanations of related knowledge points, thereby creating high-quality evaluation samples with robust knowledge support. Furthermore, AdEval incorporates mechanisms to control the number and complexity of questions, enabling dynamic alignment and flexible adjustment. This ensures that the generated questions align with the complexity of static data while supporting varied complexity levels. Based on Bloom's taxonomy, AdEval conducts a multi-dimensional evaluation of LLMs across six cognitive levels: remembering, understanding, applying, analyzing, evaluating, and creating. Experimental results on multiple datasets demonstrate that AdEval effectively reduces the impact of data contamination on evaluation outcomes, enhancing both the fairness and reliability of the evaluation process. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在大规模语料库中进行预训练，数据污染问题变得越来越严重，可能导致模型评估时性能被过度高估。为了解决这一问题，我们提出了一种名为AdEval（基于对齐的动态评估）的方法，这是一种动态数据评估方法，旨在减轻数据污染对评估可靠性的负面影响。AdEval通过提取关键知识点和主要思想，动态生成的问题与静态数据的核心概念相一致。它还利用在线搜索为相关知识点提供详尽的解释，从而创建高质量、知识支持强劲的评估样本。此外，AdEval整合了控制问题数量和复杂性的机制，能够实现动态对齐和灵活调整。这确保了生成的问题与静态数据的复杂性相一致，同时支持不同的复杂性水平。基于布卢姆分类法，AdEval在记忆、理解、应用、分析、评价和创造六个认知层面对LLMs进行多维度评估。在多个数据集上的实验结果表明，AdEval有效地减少了数据污染对评估结果的影响，提高了评估过程的公正性和可靠性。 

---
# Chain of Grounded Objectives: Bridging Process and Goal-oriented Prompting for Code Generation 

**Title (ZH)**: 基于 grounded 目标的链式推理：过程导向与目标导向提示生成代码的研究 

**Authors**: Sangyeop Yeo, Seung-won Hwang, Yu-Seung Ma  

**Link**: [PDF](https://arxiv.org/pdf/2501.13978)  

**Abstract**: The use of Large Language Models (LLMs) for code generation has gained significant attention in recent years. Existing methods often aim to improve the quality of generated code by incorporating additional contextual information or guidance into input prompts. Many of these approaches adopt sequential reasoning strategies, mimicking human-like step-by-step thinking. However, such strategies may constrain flexibility, as they do not always align with the structured characteristics of programming languages. This paper introduces the Chain of Grounded Objectives (CGO), a method that embeds functional objectives into input prompts to enhance code generation. By leveraging appropriately structured objectives as input and avoiding explicit sequential procedures, CGO adapts effectively to the structured nature of programming tasks. Empirical evaluations demonstrate that CGO effectively enhances code generation, addressing limitations of existing approaches. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在代码生成中的应用引起了广泛关注。现有方法通常通过在输入提示中融入额外的上下文信息或指导来提高生成代码的质量。这些方法中许多采用了顺序推理策略，模仿了一步步的人类思维过程。然而，这样的策略可能限制了灵活性，因为它们不一定能够适应编程语言的结构化特性。本文介绍了一种名为“基于目标链条”（Chain of Grounded Objectives，简称CGO）的方法，该方法将功能性目标嵌入输入提示中，以增强代码生成。通过利用适当结构化的目标作为输入，并避免明确的顺序程序，CGO 能够有效地适应编程任务的结构化特性。实证评估表明，CGO 能够有效提高代码生成质量，弥补现有方法的不足。 

---
# Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms 

**Title (ZH)**: 使用大型语言模型重新排序以减轻社交媒体平台上有害内容的暴露风险 

**Authors**: Rajvardhan Oak, Muhammad Haroon, Claire Jo, Magdalena Wojcieszak, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2501.13977)  

**Abstract**: Social media platforms utilize Machine Learning (ML) and Artificial Intelligence (AI) powered recommendation algorithms to maximize user engagement, which can result in inadvertent exposure to harmful content. Current moderation efforts, reliant on classifiers trained with extensive human-annotated data, struggle with scalability and adapting to new forms of harm. To address these challenges, we propose a novel re-ranking approach using Large Language Models (LLMs) in zero-shot and few-shot settings. Our method dynamically assesses and re-ranks content sequences, effectively mitigating harmful content exposure without requiring extensive labeled data. Alongside traditional ranking metrics, we also introduce two new metrics to evaluate the effectiveness of re-ranking in reducing exposure to harmful content. Through experiments on three datasets, three models and across three configurations, we demonstrate that our LLM-based approach significantly outperforms existing proprietary moderation approaches, offering a scalable and adaptable solution for harm mitigation. 

**Abstract (ZH)**: 社交媒体平台利用机器学习（ML）和人工智能（AI）驱动的推荐算法以最大化用户参与度，这可能导致用户无意中接触有害内容。当前的管理努力依赖于使用大量人工标注数据训练的分类器，这在可扩展性和适应新形式的伤害方面存在困难。为了解决这些挑战，我们提出了一种使用大规模语言模型（LLMs）进行零样本和少样本重排的新方法。我们的方法动态评估和重新排序内容序列，有效减轻了有害内容的暴露，而无需大量标注数据。除了传统的排名指标，我们还引入了两种新的指标来评估重排在减少有害内容暴露方面的有效性。通过对三个数据集、三种模型和三种配置进行实验，我们证明了基于LLM的方法显著优于现有的专有管理方法，为有害内容的减轻提供了可扩展且适应性强的解决方案。 

---
# Towards Safer Social Media Platforms: Scalable and Performant Few-Shot Harmful Content Moderation Using Large Language Models 

**Title (ZH)**: 向着更安全的社会媒体平台：使用大型语言模型的可扩展高效少量样本有害内容审核 

**Authors**: Akash Bonagiri, Lucen Li, Rajvardhan Oak, Zeerak Babar, Magdalena Wojcieszak, Anshuman Chhabra  

**Link**: [PDF](https://arxiv.org/pdf/2501.13976)  

**Abstract**: The prevalence of harmful content on social media platforms poses significant risks to users and society, necessitating more effective and scalable content moderation strategies. Current approaches rely on human moderators, supervised classifiers, and large volumes of training data, and often struggle with scalability, subjectivity, and the dynamic nature of harmful content (e.g., violent content, dangerous challenge trends, etc.). To bridge these gaps, we utilize Large Language Models (LLMs) to undertake few-shot dynamic content moderation via in-context learning. Through extensive experiments on multiple LLMs, we demonstrate that our few-shot approaches can outperform existing proprietary baselines (Perspective and OpenAI Moderation) as well as prior state-of-the-art few-shot learning methods, in identifying harm. We also incorporate visual information (video thumbnails) and assess if different multimodal techniques improve model performance. Our results underscore the significant benefits of employing LLM based methods for scalable and dynamic harmful content moderation online. 

**Abstract (ZH)**: 社交媒体平台上有害内容的普遍存在对用户和社会构成了严重风险，因此需要更加有效且可扩展的内容审核策略。当前的方法依赖于人类审核员、监督分类器和大量的训练数据，常常难以应对可扩展性、主观性和有害内容的动态性（例如暴力内容、危险挑战趋势等）所带来的挑战。为解决这些问题，我们利用大规模语言模型（LLMs）通过情境学习来开展少量示例的动态内容审核。通过在多种LLMs上进行广泛的实验，我们证明我们的少量示例方法可以在识别危害方面超越现有私有基准（Perspective和OpenAI Moderation）以及先前的最优少量示例学习方法。我们还整合了视觉信息（视频缩略图），并评估了不同多模态技术如何提升模型性能。我们的研究结果强调了利用基于LLM的方法在线开展可扩展和动态的内容审核的巨大优势。 

---
# Assisting Mathematical Formalization with A Learning-based Premise Retriever 

**Title (ZH)**: 使用基于学习的前提检索助手辅助数学形式化 

**Authors**: Yicheng Tao, Haotian Liu, Shanwen Wang, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13959)  

**Abstract**: Premise selection is a crucial yet challenging step in mathematical formalization, especially for users with limited experience. Due to the lack of available formalization projects, existing approaches that leverage language models often suffer from data scarcity. In this work, we introduce an innovative method for training a premise retriever to support the formalization of mathematics. Our approach employs a BERT model to embed proof states and premises into a shared latent space. The retrieval model is trained within a contrastive learning framework and incorporates a domain-specific tokenizer along with a fine-grained similarity computation method. Experimental results show that our model is highly competitive compared to existing baselines, achieving strong performance while requiring fewer computational resources. Performance is further enhanced through the integration of a re-ranking module. To streamline the formalization process, we will release a search engine that enables users to query Mathlib theorems directly using proof states, significantly improving accessibility and efficiency. Codes are available at this https URL. 

**Abstract (ZH)**: 前提选择是数学形式化过程中至关重要但具有挑战性的步骤，尤其对于经验有限的用户而言。由于缺乏可用的形式化项目，现有的依赖语言模型的方法往往受到数据稀少的限制。在本研究中，我们介绍了一种创新的方法来训练一个前提检索器，以支持数学形式化过程。我们的方法使用BERT模型将证明状态和前提嵌入到共享的潜在空间中。检索模型在对比学习框架下进行训练，并结合了领域特定的分词器和细粒度的相似性计算方法。实验结果表明，与现有基线相比，我们的模型具有很强的竞争性，能够在计算资源较少的情况下实现优异的性能。通过引入重排名模块，性能进一步提升。为了简化形式化过程，我们将发布一个搜索引擎，使用户能够直接使用证明状态查询Mathlib定理，从而大大提高访问性和效率。代码可在以下链接获取：this https URL。 

---
# A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models 

**Title (ZH)**: 图检索增强生成在定制大型语言模型中的综述 

**Authors**: Qinggang Zhang, Shengyuan Chen, Yuanchen Bei, Zheng Yuan, Huachi Zhou, Zijin Hong, Junnan Dong, Hao Chen, Yi Chang, Xiao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13958)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities in a wide range of tasks, yet their application to specialized domains remains challenging due to the need for deep expertise. Retrieval-augmented generation (RAG) has emerged as a promising solution to customize LLMs for professional fields by seamlessly integrating external knowledge bases, enabling real-time access to domain-specific expertise during inference. Despite its potential, traditional RAG systems, based on flat text retrieval, face three critical challenges: (i) complex query understanding in professional contexts, (ii) difficulties in knowledge integration across distributed sources, and (iii) system efficiency bottlenecks at scale. This survey presents a systematic analysis of Graph-based Retrieval-Augmented Generation (GraphRAG), a new paradigm that revolutionizes domain-specific LLM applications. GraphRAG addresses traditional RAG limitations through three key innovations: (i) graph-structured knowledge representation that explicitly captures entity relationships and domain hierarchies, (ii) efficient graph-based retrieval techniques that enable context-preserving knowledge retrieval with multihop reasoning ability, and (iii) structure-aware knowledge integration algorithms that leverage retrieved knowledge for accurate and logical coherent generation of LLMs. In this survey, we systematically analyze the technical foundations of GraphRAG and examine current implementations across various professional domains, identifying key technical challenges and promising research directions. All the related resources of GraphRAG, including research papers, open-source data, and projects, are collected for the community in \textcolor{blue}{\url{this https URL}}. 

**Abstract (ZH)**: 以下是论文内容或标题的中文翻译，符合学术规范：

大型语言模型（LLMs）在广泛的任务中展现了卓越的能力，但在专业领域的应用仍面临挑战，因为需要深厚的专业知识。检索增强生成（RAG）技术已作为一种有望的解决方案，通过无缝集成外部知识库来定制LLMs用于专业领域，从而在推理过程中实现实时访问特定领域的专业知识。尽管具有潜力，传统的RAG系统，基于平面文本检索，面临三个关键挑战：（i）在专业语境中复杂的查询理解；（ii）跨分布式源的知识整合困难；（iii）大规模应用下的系统效率瓶颈。本文综述展示了基于图的检索增强生成（GraphRAG）这一新范式，通过三个关键创新革新了特定领域LLMs的应用：（i）结构化的图状知识表示，明确捕获实体关系和领域层次结构；（ii）高效的基于图的检索技术，能够进行多跳推理的上下文保留知识检索；（iii）结构感知的知识整合算法，利用检索到的知识生成LLMs的准确和逻辑连贯的内容。在本文综述中，我们系统分析了GraphRAG的技术基础，并探讨了其在各种专业领域的现状实施，指出了关键的技术挑战和有前景的研究方向。所有关于GraphRAG的相关资源，包括研究论文、开源数据和项目，均可在 \textcolor{blue}{\url{this https URL}} 供社区使用。 

---
# Benchmarking Generative AI for Scoring Medical Student Interviews in Objective Structured Clinical Examinations (OSCEs) 

**Title (ZH)**: 用于客观结构化临床考试（OSCEs）中评分医学学生面试的生成式AI基准测试 

**Authors**: Jadon Geathers, Yann Hicke, Colleen Chan, Niroop Rajashekar, Justin Sewell, Susannah Cornes, Rene Kizilcec, Dennis Shung  

**Link**: [PDF](https://arxiv.org/pdf/2501.13957)  

**Abstract**: Introduction. Objective Structured Clinical Examinations (OSCEs) are widely used to assess medical students' communication skills, but scoring interview-based assessments is time-consuming and potentially subject to human bias. This study explored the potential of large language models (LLMs) to automate OSCE evaluations using the Master Interview Rating Scale (MIRS).
Methods. We compared the performance of four state-of-the-art LLMs (GPT-4o, Claude 3.5, Llama 3.1, and Gemini 1.5 Pro) in evaluating OSCE transcripts across all 28 items of the MIRS under the conditions of zero-shot, chain-of-thought (CoT), few-shot, and multi-step prompting. The models were benchmarked against a dataset of 10 OSCE cases with 174 expert consensus scores available. Model performance was measured using three accuracy metrics (exact, off-by-one, thresholded).
Results. Averaging across all MIRS items and OSCE cases, LLMs performed with low exact accuracy (0.27 to 0.44), and moderate to high off-by-one accuracy (0.67 to 0.87) and thresholded accuracy (0.75 to 0.88). A zero temperature parameter ensured high intra-rater reliability ($\alpha = 0.98$ for GPT-4o). CoT, few-shot, and multi-step techniques proved valuable when tailored to specific assessment items. The performance was consistent across MIRS items independent of encounter phases and communication domains.
Conclusion. We demonstrated the feasibility of AI-assisted OSCE evaluation and provided benchmarking of multiple LLMs across multiple prompt techniques. Our work provides a baseline performance assessment for LLMs that lays a foundation for future research in automated assessment of clinical communication skills. 

**Abstract (ZH)**: 介绍。结构化临床考试（OSCEs）广泛用于评估医学生沟通技能，但基于面试的评估打分耗时且可能受到人为偏见的影响。本研究探讨了大型语言模型（LLMs）在使用主面试评分量表（MIRS）自动进行OSCE评价方面的潜力。

方法。我们比较了四种最新的LLM（GPT-4o、Claude 3.5、Llama 3.1和Gemini 1.5 Pro）在零样本、链式思维（CoT）、少样本和多步骤提示条件下，对OSCE转录记录中所有28项MIRS项目进行评估的性能。模型使用包含10个OSCE案例和174个专家共识评分的数据集进行了基准测试。模型性能通过三种精度指标（精确度、相差一个点的精确度、阈值精确度）进行衡量。

结果。在所有MIRS项目和OSCE案例的平均值上，LLM的精确度较低（0.27至0.44），相差一个点的精确度（0.67至0.87）和阈值精确度（0.75至0.88）中等至较高。零温度参数确保了GPT-4o的高评分者内部一致性信度（$\alpha = 0.98$）。链式思维、少样本和多步骤提示技术在针对特定评估项目进行调整时具有价值。性能在MIRS项目中保持一致，不随接诊阶段和沟通领域变化。

结论。本研究展示了AI辅助OSCE评价的可能性，并为多种LLM在多种提示技术下的基准进行比较。我们的工作为基于未来的临床沟通技能自动化评估研究提供了LLM的基本性能评估，奠定了基础。 

---
# Zep: A Temporal Knowledge Graph Architecture for Agent Memory 

**Title (ZH)**: Zep：一种用于代理记忆的时间知识图谱架构

这个标题翻译成中文时，保持了原文的结构和意义，确保了学术规范。其中，“Zep”被保留为原名，如果它有特定的含义或缩写，最好保持不变并提供必要的解释。 

**Authors**: Preston Rasmussen, Pavlo Paliychuk, Travis Beauvais, Jack Ryan, Daniel Chalef  

**Link**: [PDF](https://arxiv.org/pdf/2501.13956)  

**Abstract**: We introduce Zep, a novel memory layer service for AI agents that outperforms the current state-of-the-art system, MemGPT, in the Deep Memory Retrieval (DMR) benchmark. Additionally, Zep excels in more comprehensive and challenging evaluations than DMR that better reflect real-world enterprise use cases. While existing retrieval-augmented generation (RAG) frameworks for large language model (LLM)-based agents are limited to static document retrieval, enterprise applications demand dynamic knowledge integration from diverse sources including ongoing conversations and business data. Zep addresses this fundamental limitation through its core component Graphiti -- a temporally-aware knowledge graph engine that dynamically synthesizes both unstructured conversational data and structured business data while maintaining historical relationships. In the DMR benchmark, which the MemGPT team established as their primary evaluation metric, Zep demonstrates superior performance (94.8% vs 93.4%). Beyond DMR, Zep's capabilities are further validated through the more challenging LongMemEval benchmark, which better reflects enterprise use cases through complex temporal reasoning tasks. In this evaluation, Zep achieves substantial results with accuracy improvements of up to 18.5% while simultaneously reducing response latency by 90% compared to baseline implementations. These results are particularly pronounced in enterprise-critical tasks such as cross-session information synthesis and long-term context maintenance, demonstrating Zep's effectiveness for deployment in real-world applications. 

**Abstract (ZH)**: 我们介绍了Zep，这是一种新型的记忆层服务，它在深度记忆检索（DMR）基准中超越了现有的最佳系统MemGPT。此外，Zep在比DMR更为全面和挑战性的评估中表现出色，这些评估更贴近现实世界的企业应用场景。现有的基于大型语言模型（LLM）的检索增强生成（RAG）框架仅限于静态文档检索，而企业应用则需要从包括实时对话和业务数据在内的多种来源动态集成知识。Zep通过其核心组件Graphiti——一个具有时间感知能力的知识图谱引擎，解决了这一基本局限。Graphiti动态综合了未结构化的对话数据和结构化的业务数据，并保持了历史关系。在MemGPT团队将其作为主要评估标准的DMR基准中，Zep展现了更优的性能（94.8% vs 93.4%）。除了DMR基准，Zep的能力还通过LongMemEval基准得到了进一步验证，LongMemEval基准通过复杂的时序推理任务更好地反映了企业应用场景。在这一评估中，Zep取得了显著的结果，准确率提高了最高达18.5%，同时将响应延迟减少了90%相比基线实现。这些结果在企业关键任务中尤为突出，如跨会话信息综合和长期上下文维护，证明了Zep在实际应用部署中的有效性。 

---
# Guided Persona-based AI Surveys: Can we replicate personal mobility preferences at scale using LLMs? 

**Title (ZH)**: 基于引导式人格模型的AI调查：我们能否使用大型语言模型（LLMs）在大规模范围内复制个人的出行偏好？ 

**Authors**: Ioannis Tzachristas, Santhanakrishnan Narayanan, Constantinos Antoniou  

**Link**: [PDF](https://arxiv.org/pdf/2501.13955)  

**Abstract**: This study explores the potential of Large Language Models (LLMs) to generate artificial surveys, with a focus on personal mobility preferences in Germany. By leveraging LLMs for synthetic data creation, we aim to address the limitations of traditional survey methods, such as high costs, inefficiency and scalability challenges. A novel approach incorporating "Personas" - combinations of demographic and behavioural attributes - is introduced and compared to five other synthetic survey methods, which vary in their use of real-world data and methodological complexity. The MiD 2017 dataset, a comprehensive mobility survey in Germany, serves as a benchmark to assess the alignment of synthetic data with real-world patterns. The results demonstrate that LLMs can effectively capture complex dependencies between demographic attributes and preferences while offering flexibility to explore hypothetical scenarios. This approach presents valuable opportunities for transportation planning and social science research, enabling scalable, cost-efficient and privacy-preserving data generation. 

**Abstract (ZH)**: 本研究探讨了大型语言模型（LLMs）在生成人工调查方面的潜力，重点是德国人的出行偏好。通过利用LLMs生成合成数据，我们旨在解决传统调查方法中存在的高成本、不高效和可扩展性挑战。我们提出了一种新颖的方法，即“人设”——结合了人口统计和行为特征的组合，并将其与五种其他合成调查方法进行了比较，这些方法在使用真实世界数据和方法复杂性方面存在差异。德国MiD 2017数据集作为基准数据集，用于评估合成数据与真实世界模式的一致性。研究结果表明，LLMs能够有效地捕捉人口统计学特征与偏好之间的复杂依赖关系，同时提供探索假设情景的灵活性。这一方法为交通规划和社会科学研究提供了宝贵的机会，能够实现大规模、低成本和隐私保护的数据生成。 

---
# Chat3GPP: An Open-Source Retrieval-Augmented Generation Framework for 3GPP Documents 

**Title (ZH)**: Chat3GPP: 一个用于3GPP文档的开源检索增强生成框架 

**Authors**: Long Huang, Ming Zhao, Limin Xiao, Xiujun Zhang, Jungang Hu  

**Link**: [PDF](https://arxiv.org/pdf/2501.13954)  

**Abstract**: The 3rd Generation Partnership Project (3GPP) documents is key standards in global telecommunications, while posing significant challenges for engineers and researchers in the telecommunications field due to the large volume and complexity of their contents as well as the frequent updates. Large language models (LLMs) have shown promise in natural language processing tasks, but their general-purpose nature limits their effectiveness in specific domains like telecommunications. To address this, we propose Chat3GPP, an open-source retrieval-augmented generation (RAG) framework tailored for 3GPP specifications. By combining chunking strategies, hybrid retrieval and efficient indexing methods, Chat3GPP can efficiently retrieve relevant information and generate accurate responses to user queries without requiring domain-specific fine-tuning, which is both flexible and scalable, offering significant potential for adapting to other technical standards beyond 3GPP. We evaluate Chat3GPP on two telecom-specific datasets and demonstrate its superior performance compared to existing methods, showcasing its potential for downstream tasks like protocol generation and code automation. 

**Abstract (ZH)**: 3GPP文档是全球电信领域的重要标准，但由于其内容庞大且复杂，以及频繁更新，这给电信领域的工程师和研究人员带来了重大挑战。大规模语言模型（LLMs）在自然语言处理任务中显示出潜力，但由于其通用性，它们在特定领域如电信中的有效性受到限制。为了解决这个问题，我们提出了一种名为Chat3GPP的开源检索增强生成（RAG）框架，专门针对3GPP规范。通过结合切片策略、混合检索和高效索引方法，Chat3GPP可以高效地检索相关信息，并生成准确的用户查询响应，而无需进行领域特定的微调，从而提供灵活且可扩展的解决方案，具备适应其他技术标准的巨大潜力。我们在两个电信特定数据集上评估了Chat3GPP，并展示了其在现有方法中的优越性能，彰显了其在协议生成和代码自动化等下游任务中的潜在应用价值。 

---
# Redundancy Principles for MLLMs Benchmarks 

**Title (ZH)**: MLLMs基准中的冗余原则 

**Authors**: Zicheng Zhang, Xiangyu Zhao, Xinyu Fang, Chunyi Li, Xiaohong Liu, Xiongkuo Min, Haodong Duan, Kai Chen, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2501.13953)  

**Abstract**: With the rapid iteration of Multi-modality Large Language Models (MLLMs) and the evolving demands of the field, the number of benchmarks produced annually has surged into the hundreds. The rapid growth has inevitably led to significant redundancy among benchmarks. Therefore, it is crucial to take a step back and critically assess the current state of redundancy and propose targeted principles for constructing effective MLLM benchmarks. In this paper, we focus on redundancy from three key perspectives: 1) Redundancy of benchmark capability dimensions, 2) Redundancy in the number of test questions, and 3) Cross-benchmark redundancy within specific domains. Through the comprehensive analysis over hundreds of MLLMs' performance across more than 20 benchmarks, we aim to quantitatively measure the level of redundancy lies in existing MLLM evaluations, provide valuable insights to guide the future development of MLLM benchmarks, and offer strategies to refine and address redundancy issues effectively. 

**Abstract (ZH)**: 随着多模态大型语言模型（MLLMs）的快速迭代以及该领域需求的不断演变，每年发布的评估基准数量已激增至数百个。这一快速增长不可避免地导致了评估基准之间的显著冗余性。因此，亟需退一步，对当前的冗余性进行批判性评估，并提出构建有效MLLM基准的目标原则。在本文中，我们从三个关键视角出发探讨冗余性问题：1）基准能力维度的冗余性；2）测试问题数量的冗余性；3）特定领域内的跨基准冗余性。通过对20多个基准上数百个MLLM性能的全面分析，我们旨在定量衡量现有MLLM评估中的冗余程度，提供有价值的指导以引导未来MLLM基准的开发，并提出有效解决冗余问题的策略。 

---
# The Dual-use Dilemma in LLMs: Do Empowering Ethical Capacities Make a Degraded Utility? 

**Title (ZH)**: LLM的双重用途困境：增强道德能力是否会削弱其实用价值？ 

**Authors**: Yiyi Zhang, Xingyu Chen, Kexin Chen, Yuyang Du, Xilin Dang, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2501.13952)  

**Abstract**: Recent years have witnessed extensive efforts to enhance Large Language Models (LLMs) across various domains, alongside growing attention to their ethical implications. However, a critical challenge remains largely overlooked: LLMs must balance between rejecting harmful requests for safety and accommodating legitimate ones for utility. This paper presents a Direct Preference Optimization (DPO) based alignment framework that achieves better overall performance by addressing this ethical-utility trade-off, using chemical domain applications as a proof-of-concept. Our alignment pipeline starts with a GPT-assisted three-phase data generation scheme, in which we create LibraChemQA, a chemical question-answering dataset comprising 31.6k triplet instances. By incorporating an innovative balanced seed in the data generation process, our framework systematically considers both legitimate and illegitimate requests. The framework also introduces a rephrasing mechanism for efficient data augmentation that enhances the model's chemical comprehension. We further develop a novel hybrid evaluation scheme with LLM judges for precise assessment of both safety and utility. Experimental results demonstrate our model's substantial improvements in overall performance where both safety and utility are considered - our resulting model, LibraChem, outperforms leading LLMs including Claude-3, GPT-4o, and LLaMA-3 by margins of 13.44%, 7.16%, and 7.10% respectively on our released benchmark. 

**Abstract (ZH)**: 近年来，各界对增强大型语言模型（LLMs）的研究取得了广泛进展，同时对其伦理影响的关注也在不断增长。然而，一个关键挑战依然被很大程度上忽视：LLMs 必须在保障安全拒绝有害请求与满足实用请求之间找到平衡。本文提出了一种基于直接偏好优化（DPO）的对齐框架，通过解决这一伦理-实用性权衡问题，实现了更好的整体性能，并以化学领域的应用作为概念验证。我们的对齐管道始于由 GPT 辅助的三阶段数据生成方案，在此过程中，我们构建了包含31,600个三元组实例的 LibraChemQA 化学问答数据集。通过在数据生成过程中引入一种创新的平衡种子，我们的框架系统地考虑了合法和不合法请求。此外，该框架引入了一种重述机制，用于高效的数据增强，以提高模型的化学理解能力。我们还开发了一种新的混合评估方案，使用 LLMS 判官进行精确评估，以评估安全性和实用性。实验结果表明，当考虑安全性和实用性时，我们的模型在整体性能上有了显著改进：我们的 LibraChem 模型在发布的基准测试中分别比 Claude-3、GPT-4o 和 LLaMA-3 获得了13.44%、7.16% 和 7.10% 的性能优势。 

---
# A Layered Multi-Expert Framework for Long-Context Mental Health Assessments 

**Title (ZH)**: 一种分层多专家框架，用于长上下文心理健康评估 

**Authors**: Jinwen Tang, Qiming Guo, Wenbo Sun, Yi Shang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13951)  

**Abstract**: Long-form mental health assessments pose unique challenges for large language models (LLMs), which often exhibit hallucinations or inconsistent reasoning when handling extended, domain-specific contexts. We introduce Stacked Multi-Model Reasoning (SMMR), a layered framework that leverages multiple LLMs and specialized smaller models as coequal 'experts'. Early layers isolate short, discrete subtasks, while later layers integrate and refine these partial outputs through more advanced long-context models. We evaluate SMMR on the DAIC-WOZ depression-screening dataset and 48 curated case studies with psychiatric diagnoses, demonstrating consistent improvements over single-model baselines in terms of accuracy, F1-score, and PHQ-8 error reduction. By harnessing diverse 'second opinions', SMMR mitigates hallucinations, captures subtle clinical nuances, and enhances reliability in high-stakes mental health assessments. Our findings underscore the value of multi-expert frameworks for more trustworthy AI-driven screening. 

**Abstract (ZH)**: 长形式心理健康评估为大规模语言模型（LLMs）带来了独特的挑战，这些模型在处理扩展的、特定领域的内容时常常表现出幻觉或不一致的推理。我们提出了一种分层框架——堆叠多模型推理（Stacked Multi-Model Reasoning, SMMR），该框架利用多个LLMs和专门的小型模型作为平等的“专家”。早期层隔离短期的离散子任务，而后期层通过更复杂的长上下文模型整合和精炼这些部分输出。我们使用DAIC-WOZ抑郁筛查数据集和48个经过精心选择的精神疾病病例研究评估了SMMR，结果显示在准确率、F1分数和PHQ-8错误减少方面，SMMR相较于单模型基线模型具有一致的改进。通过利用多样化的“第二意见”，SMMR减轻了幻觉现象，捕捉到了临床细微差别，并在高风险的心理健康评估中提升了可靠性。我们的研究结果强调了多专家框架在更可信的人工智能筛查中的价值。 

---
# Can OpenAI o1 Reason Well in Ophthalmology? A 6,990-Question Head-to-Head Evaluation Study 

**Title (ZH)**: OpenAI在眼科领域能进行有效的推理吗？一项基于6,990个问题的头对头评估研究 

**Authors**: Sahana Srinivasan, Xuguang Ai, Minjie Zou, Ke Zou, Hyunjae Kim, Thaddaeus Wai Soon Lo, Krithi Pushpanathan, Yiming Kong, Anran Li, Maxwell Singer, Kai Jin, Fares Antaki, David Ziyou Chen, Dianbo Liu, Ron A. Adelman, Qingyu Chen, Yih Chung Tham  

**Link**: [PDF](https://arxiv.org/pdf/2501.13949)  

**Abstract**: Question: What is the performance and reasoning ability of OpenAI o1 compared to other large language models in addressing ophthalmology-specific questions?
Findings: This study evaluated OpenAI o1 and five LLMs using 6,990 ophthalmological questions from MedMCQA. O1 achieved the highest accuracy (0.88) and macro-F1 score but ranked third in reasoning capabilities based on text-generation metrics. Across subtopics, o1 ranked first in ``Lens'' and ``Glaucoma'' but second to GPT-4o in ``Corneal and External Diseases'', ``Vitreous and Retina'' and ``Oculoplastic and Orbital Diseases''. Subgroup analyses showed o1 performed better on queries with longer ground truth explanations.
Meaning: O1's reasoning enhancements may not fully extend to ophthalmology, underscoring the need for domain-specific refinements to optimize performance in specialized fields like ophthalmology. 

**Abstract (ZH)**: 问题：OpenAI o1 在应对眼科特定问题时的表现和推理能力与其他大型语言模型相比如何？

发现：本研究使用来自 MedMCQA 的 6,990 个眼科问题，评估了 OpenAI o1 和五个其他大型语言模型（LLMs）。OpenAI o1 在准确性（0.88）和宏-F1 分数方面表现最佳，但在基于文本生成的指标中排名第三，在推理能力方面位居第三。就子主题而言，o1 在“晶状体”和“青光眼”方面排名第一，但在“角膜和外眼疾病”、“玻璃体和视网膜疾病”以及“眼整形和眼眶疾病”方面的表现仅次于 GPT-4o。子组分析表明，o1 在具有较长正确解释的答案查询上表现更好。

意义：OpenAI o1 的推理增强可能并未完全扩展到眼科领域，强调了在优化特定领域如眼科的性能时，需要进行领域特定的改进。 

---
# Longitudinal Abuse and Sentiment Analysis of Hollywood Movie Dialogues using LLMs 

**Title (ZH)**: 使用大语言模型对好莱坞电影对话的纵向虐待及情感分析 

**Authors**: Rohitash Chandra, Guoxiang Ren, Group-H  

**Link**: [PDF](https://arxiv.org/pdf/2501.13948)  

**Abstract**: Over the past decades, there has been an increasing concern about the prevalence of abusive and violent content in Hollywood movies. This study uses Large Language Models (LLMs) to explore the longitudinal abuse and sentiment analysis of Hollywood Oscar and blockbuster movie dialogues from 1950 to 2024. By employing fine-tuned LLMs, we analyze subtitles for over a thousand movies categorised into four genres to examine the trends and shifts in emotional and abusive content over the past seven decades. Our findings reveal significant temporal changes in movie dialogues, which reflect broader social and cultural influences. Overall, the emotional tendencies in the films are diverse, and the detection of abusive content also exhibits significant fluctuations. The results show a gradual rise in abusive content in recent decades, reflecting social norms and regulatory policy changes. Genres such as thrillers still present a higher frequency of abusive content that emphasises the ongoing narrative role of violence and conflict. At the same time, underlying positive emotions such as humour and optimism remain prevalent in most of the movies. Furthermore, the gradual increase of abusive content in movie dialogues has been significant over the last two decades, where Oscar-nominated movies overtook the top ten blockbusters. 

**Abstract (ZH)**: 在过去的几十年中，好莱坞电影中虐待和暴力内容的盛行引起了越来越多的关注。本研究利用大规模语言模型（LLMs）探索从1950年至2024年奥斯卡获奖和大片对白的纵向虐待和情绪分析。通过使用细调后的LLMs，我们对四大类别的上千部电影的字幕进行了分析，以考察过去七十年间情绪性和虐待性内容的趋势和变化。研究发现，电影对白在时间维度上显示出显著的变化，这些变化反映了更广泛的社会和文化影响。总体而言，电影中的情感倾向是多样化的，虐待内容的检测也表现出显著的波动。结果显示，在过去几十年中，虐待内容逐渐增多，这反映了社会规范和监管政策的变化。例如，惊悚片仍频繁出现虐待内容，这突显了暴力和冲突在叙事中的持续作用。同时，大多数电影中仍保持着积极情绪，如幽默和乐观的普遍存在。此外，电影对话中虐待内容的逐渐增加在最近二十多年尤为显著，其中奥斯卡提名电影超过了前十部大片。 

---
# A Comprehensive Survey on Integrating Large Language Models with Knowledge-Based Methods 

**Title (ZH)**: 大型语言模型与知识基础方法集成的综合调研 

**Authors**: Lilian Some, Wenli Yang, Michael Bain, Byeong Kang  

**Link**: [PDF](https://arxiv.org/pdf/2501.13947)  

**Abstract**: The rapid development of artificial intelligence has brought about substantial advancements in the field. One promising direction is the integration of Large Language Models (LLMs) with structured knowledge-based systems. This approach aims to enhance AI capabilities by combining the generative language understanding of LLMs with the precise knowledge representation of structured systems. This survey explores the synergy between LLMs and knowledge bases, focusing on real-world applications and addressing associated technical, operational, and ethical challenges. Through a comprehensive literature review, the study identifies critical issues and evaluates existing solutions. The paper highlights the benefits of integrating generative AI with knowledge bases, including improved data contextualization, enhanced model accuracy, and better utilization of knowledge resources. The findings provide a detailed overview of the current state of research, identify key gaps, and offer actionable recommendations. These insights contribute to advancing AI technologies and support their practical deployment across various sectors. 

**Abstract (ZH)**: 随着人工智能的迅速发展，该领域取得了显著的进步。一个有前途的方向是将大型语言模型（LLMs）与结构化知识系统集成。这种方法旨在通过结合LLMs的生成语言理解和结构化系统精确的知识表示来增强人工智能的能力。本文综述了LLMs与知识库之间的协同作用，重点关注实际应用，并解决相关的技术和操作性挑战以及伦理问题。通过全面的文献综述，研究识别出关键问题并评估现有解决方案。文章强调了将生成式AI与知识库集成的好处，包括改善数据上下文、提高模型准确性以及更好地利用知识资源。研究结果提供了当前研究状态的详细概述，指出现有研究的关键缺口，并提出可操作建议。这些见解有助于推动人工智能技术的进步，并支持其在各种领域的实际部署。 

---
# Hallucination Mitigation using Agentic AI Natural Language-Based Frameworks 

**Title (ZH)**: 使用自主人工智能基于自然语言的框架减轻幻觉现象 

**Authors**: Diego Gosmar, Deborah A. Dahl  

**Link**: [PDF](https://arxiv.org/pdf/2501.13946)  

**Abstract**: Hallucinations remain a significant challenge in current Generative AI models, undermining trust in AI systems and their reliability. This study investigates how orchestrating multiple specialized Artificial Intelligent Agents can help mitigate such hallucinations, with a focus on systems leveraging Natural Language Processing (NLP) to facilitate seamless agent interactions. To achieve this, we design a pipeline that introduces over three hundred prompts, purposefully crafted to induce hallucinations, into a front-end agent. The outputs are then systematically reviewed and refined by second- and third-level agents, each employing distinct large language models and tailored strategies to detect unverified claims, incorporate explicit disclaimers, and clarify speculative content. Additionally, we introduce a set of novel Key Performance Indicators (KPIs) specifically designed to evaluate hallucination score levels. A dedicated fourth-level AI agent is employed to evaluate these KPIs, providing detailed assessments and ensuring accurate quantification of shifts in hallucination-related behaviors. A core component of this investigation is the use of the OVON (Open Voice Network) framework, which relies on universal NLP-based interfaces to transfer contextual information among agents. Through structured JSON messages, each agent communicates its assessment of the hallucination likelihood and the reasons underlying questionable content, thereby enabling the subsequent stage to refine the text without losing context. The results demonstrate that employing multiple specialized agents capable of interoperating with each other through NLP-based agentic frameworks can yield promising outcomes in hallucination mitigation, ultimately bolstering trust within the AI community. 

**Abstract (ZH)**: 当前生成式人工智能模型中的幻觉依然构成了一项重大挑战，削弱了人们对AI系统的信任和可靠性。本研究探讨了如何通过协调多个专业化的人工智能代理来减轻这些幻觉，重点关注利用自然语言处理（NLP）促进代理间无缝交互的系统。为了实现这一目标，我们设计了一种管道，引入了超过三百个旨在引发幻觉的提示，这些提示被故意设计以检验系统表现。输出结果随后由第二级和第三级代理人系统地审查和优化。这些第二级和第三级代理人各自采用不同的大型语言模型和定制策略，以检测未经验证的声明，纳入明确的免责声明，并澄清推测性内容。此外，我们引入了一组新的关键绩效指标（KPIs），专门用于评估幻觉得分水平。我们采用一个专门的第四级AI代理来评估这些KPIs，提供详细评估，并确保准确量化幻觉相关行为的转变。本研究的核心组成部分是使用OVON（开放语音网络）框架，该框架依赖于基于NLP的通用接口在代理之间传递上下文信息。通过结构化的JSON消息，每个代理传达其对幻觉可能性的评估及其判断可疑内容的原因，从而允许下个阶段在不丢失上下文的情况下优化文本。结果表明，通过利用多个人工智能代理并利用基于NLP的代理框架实现互操作，可以在减轻幻觉方面取得有前景的成果，最终增强人工智能社区内的信任。 

---
# Self-Explanation in Social AI Agents 

**Title (ZH)**: 社交人工智能代理的自我解释研究 

**Authors**: Rhea Basappa, Mustafa Tekman, Hong Lu, Benjamin Faught, Sandeep Kakar, Ashok K. Goel  

**Link**: [PDF](https://arxiv.org/pdf/2501.13945)  

**Abstract**: Social AI agents interact with members of a community, thereby changing the behavior of the community. For example, in online learning, an AI social assistant may connect learners and thereby enhance social interaction. These social AI assistants too need to explain themselves in order to enhance transparency and trust with the learners. We present a method of self-explanation that uses introspection over a self-model of an AI social assistant. The self-model is captured as a functional model that specifies how the methods of the agent use knowledge to achieve its tasks. The process of generating self-explanations uses Chain of Thought to reflect on the self-model and ChatGPT to provide explanations about its functioning. We evaluate the self-explanation of the AI social assistant for completeness and correctness. We also report on its deployment in a live class. 

**Abstract (ZH)**: 社会AI代理与社区成员进行交互，从而改变社区的行为。例如，在在线学习中，一个AI社交助手可能会连接学习者，从而增强社交互动。这些社会AI助手也需要进行自我解释，以提高透明度和学习者的信任度。我们提出了一种自我解释的方法，该方法利用AI社交助手的自我模型进行内省。自我模型被描述为一个功能模型，该模型指定了代理的方法如何使用知识来完成其任务。生成自我解释的过程使用了“思维链”来反思自我模型，并使用ChatGPT来解释其运作方式。我们评估了AI社交助手的自我解释的完整性和正确性。我们还报告了该系统在实际课堂中的部署情况。 

---
# Fanar: An Arabic-Centric Multimodal Generative AI Platform 

**Title (ZH)**: Fanar：一个以阿拉伯语为中心的多模态生成人工智能平台 

**Authors**: Fanar Team, Ummar Abbas, Mohammad Shahmeer Ahmad, Firoj Alam, Enes Altinisik, Ehsannedin Asgari, Yazan Boshmaf, Sabri Boughorbel, Sanjay Chawla, Shammur Chowdhury, Fahim Dalvi, Kareem Darwish, Nadir Durrani, Mohamed Elfeky, Ahmed Elmagarmid, Mohamed Eltabakh, Masoomali Fatehkia, Anastasios Fragkopoulos, Maram Hasanain, Majd Hawasly, Mus'ab Husaini, Soon-Gyo Jung, Ji Kim Lucas, Walid Magdy, Safa Messaoud, Abubakr Mohamed, Tasnim Mohiuddin, Basel Mousi, Hamdy Mubarak, Ahmad Musleh, Zan Naeem, Mourad Ouzzani, Dorde Popovic, Amin Sadeghi, Husrev Taha Sencar, Mohammed Shinoy, Omar Sinan, Yifan Zhang, Ahmed Ali, Yassine El Kheir, Xiaosong Ma, Chaoyi Ruan  

**Link**: [PDF](https://arxiv.org/pdf/2501.13944)  

**Abstract**: We present Fanar, a platform for Arabic-centric multimodal generative AI systems, that supports language, speech and image generation tasks. At the heart of Fanar are Fanar Star and Fanar Prime, two highly capable Arabic Large Language Models (LLMs) that are best in the class on well established benchmarks for similar sized models. Fanar Star is a 7B (billion) parameter model that was trained from scratch on nearly 1 trillion clean and deduplicated Arabic, English and Code tokens. Fanar Prime is a 9B parameter model continually trained on the Gemma-2 9B base model on the same 1 trillion token set. Both models are concurrently deployed and designed to address different types of prompts transparently routed through a custom-built orchestrator. The Fanar platform provides many other capabilities including a customized Islamic Retrieval Augmented Generation (RAG) system for handling religious prompts, a Recency RAG for summarizing information about current or recent events that have occurred after the pre-training data cut-off date. The platform provides additional cognitive capabilities including in-house bilingual speech recognition that supports multiple Arabic dialects, voice and image generation that is fine-tuned to better reflect regional characteristics. Finally, Fanar provides an attribution service that can be used to verify the authenticity of fact based generated content.
The design, development, and implementation of Fanar was entirely undertaken at Hamad Bin Khalifa University's Qatar Computing Research Institute (QCRI) and was sponsored by Qatar's Ministry of Communications and Information Technology to enable sovereign AI technology development. 

**Abstract (ZH)**: 我们介绍了Fanar平台，这是一个针对阿拉伯语的多模态生成AI系统平台，支持语言、语音和图像生成任务。Fanar的核心是Fanar Star和Fanar Prime，这两个阿拉伯大型语言模型（LLMs）在同类模型的标准基准测试中表现优异。Fanar Star是一个70亿参数的模型，从近1万亿个干净且去重的阿拉伯语、英语和代码词汇中从头开始训练。Fanar Prime是一个90亿参数的模型，持续从Gemma-2 90亿参数的基础模型中进行训练，使用相同的1万亿词汇集。这两个模型同时部署，并设计为通过自定义构建的协调器透明地处理不同类型的提示。Fanar平台还提供了其他许多功能，包括针对宗教提示的定制伊斯兰检索增强生成（RAG）系统，用于总结预训练数据截止日期之后发生的信息或近期事件的最新RAG系统。该平台还提供了认知功能，包括支持多种阿拉伯方言的内部双语语音识别、细调以更好地反映地区特性的语音和图像生成。最后，Fanar提供了一种归属服务，可用于验证基于事实生成内容的真实性。

Fanar的设计、开发和实现全部由卡塔尔霍姆德·宾·哈利法大学的卡塔尔计算机研究学院（QCRI）完成，并得到了卡塔尔通信与信息技术部的赞助，旨在促进主权AI技术的发展。 

---
# Language Representation Favored Zero-Shot Cross-Domain Cognitive Diagnosis 

**Title (ZH)**: 语言表示在零样本跨领域认知诊断中占优 

**Authors**: Shuo Liu, Zihan Zhou, Yuanhao Liu, Jing Zhang, Hong Qian  

**Link**: [PDF](https://arxiv.org/pdf/2501.13943)  

**Abstract**: Cognitive diagnosis aims to infer students' mastery levels based on their historical response logs. However, existing cognitive diagnosis models (CDMs), which rely on ID embeddings, often have to train specific models on specific domains. This limitation may hinder their directly practical application in various target domains, such as different subjects (e.g., Math, English and Physics) or different education platforms (e.g., ASSISTments, Junyi Academy and Khan Academy). To address this issue, this paper proposes the language representation favored zero-shot cross-domain cognitive diagnosis (LRCD). Specifically, LRCD first analyzes the behavior patterns of students, exercises and concepts in different domains, and then describes the profiles of students, exercises and concepts using textual descriptions. Via recent advanced text-embedding modules, these profiles can be transformed to vectors in the unified language space. Moreover, to address the discrepancy between the language space and the cognitive diagnosis space, we propose language-cognitive mappers in LRCD to learn the mapping from the former to the latter. Then, these profiles can be easily and efficiently integrated and trained with existing CDMs. Extensive experiments show that training LRCD on real-world datasets can achieve commendable zero-shot performance across different target domains, and in some cases, it can even achieve competitive performance with some classic CDMs trained on the full response data on target domains. Notably, we surprisingly find that LRCD can also provide interesting insights into the differences between various subjects (such as humanities and sciences) and sources (such as primary and secondary education). 

**Abstract (ZH)**: 认知诊断旨在通过学生的历史作答记录推断其掌握水平。然而，现有的认知诊断模型（CDMs），依赖于ID嵌入，往往需要针对特定领域进行特定模型的训练。这一限制可能会阻碍它们在各种目标领域的直接应用，例如不同的学科（如数学、英语和物理）或不同的教育平台（如ASSISTments、Junyi Academy和Khan Academy）。为解决这一问题，本文提出了一种受语言表示青睐的零样本跨域认知诊断方法（LRCD）。具体而言，LRCD 首先分析不同领域中学生、习题和概念的行为模式，然后使用文本描述来描绘学生、习题和概念的特征。通过近期先进的文本嵌入模块，这些特征可以被转换到统一的语言空间的向量中。此外，为了消除语言空间与认知诊断空间之间的差异，LRCD 中提出了语言-认知映射器来学习从前者到后者的映射。然后，这些特征可以轻松且高效地与现有的 CDMs 整合和训练。广泛实验表明，LRCD 在实际数据集上的训练可以实现跨不同目标领域的出色零样本性能，并且在某些情况下，其性能甚至可以与针对目标领域完整作答数据训练的经典CDMs媲美。值得注意的是，我们意外地发现，LRCD 还可以提供有关不同学科（如人文学科和自然科学）和不同来源（如初等教育和中等教育）之间差异的有趣见解。 

---
# Mitigating GenAI-powered Evidence Pollution for Out-of-Context Multimodal Misinformation Detection 

**Title (ZH)**: 利用基于生成式人工智能的方法减轻脱离上下文的跨模态虚假信息检测中的证据污染问题 

**Authors**: Zehong Yan, Peng Qi, Wynne Hsu, Mong Li Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.14728)  

**Abstract**: While large generative artificial intelligence (GenAI) models have achieved significant success, they also raise growing concerns about online information security due to their potential misuse for generating deceptive content. Out-of-context (OOC) multimodal misinformation detection, which often retrieves Web evidence to identify the repurposing of images in false contexts, faces the issue of reasoning over GenAI-polluted evidence to derive accurate predictions. Existing works simulate GenAI-powered pollution at the claim level with stylistic rewriting to conceal linguistic cues, and ignore evidence-level pollution for such information-seeking applications. In this work, we investigate how polluted evidence affects the performance of existing OOC detectors, revealing a performance degradation of more than 9 percentage points. We propose two strategies, cross-modal evidence reranking and cross-modal claim-evidence reasoning, to address the challenges posed by polluted evidence. Extensive experiments on two benchmark datasets show that these strategies can effectively enhance the robustness of existing out-of-context detectors amidst polluted evidence. 

**Abstract (ZH)**: 尽管大型生成人工智能（GenAI）模型取得了显著的成果，但它们也因潜在的误导性内容生成而引发了日益增长的在线信息安全 concerns。脱嵌（Out-of-Context, OOC）多模态虚假信息检测通常通过检索网络证据来识别错误上下文中的图像用途，但在处理 GenAI 污染的证据以获得准确预测时面临挑战。现有研究在声明层面通过风格性重写模拟 GenAI 动力污染，以掩盖语言线索，但在此类信息检索应用中忽略了证据层面的污染。在此项工作中，我们探讨了污染证据对现有 OOC 检测器性能的影响，发现性能下降超过 9 个百分点。我们提出了两种策略——跨模态证据重排和跨模态声明-证据推理——以应对污染证据带来的挑战。在两个基准数据集上的广泛实验表明，这些策略能够有效提升在污染证据环境下现有 OOC 检测器的鲁棒性。 

---
# The Karp Dataset 

**Title (ZH)**: 卡普数据集 

**Authors**: Mason DiCicco, Eamon Worden, Conner Olsen, Nikhil Gangaram, Daniel Reichman, Neil Heffernan  

**Link**: [PDF](https://arxiv.org/pdf/2501.14705)  

**Abstract**: Understanding the mathematical reasoning capabilities of Large Language Models (LLMs) is a central topic in the study of artificial intelligence. This new domain necessitates the creation of datasets of reasoning tasks for both training and benchmarking the performance of LLMs. To this end, we introduce the Karp dataset: The first dataset composed of detailed proofs of NP-completeness reductions. The reductions vary in difficulty, ranging from simple exercises of undergraduate courses to more challenging reductions from academic papers. We compare the performance of state-of-the-art models on this task and demonstrate the effect of fine-tuning with the Karp dataset on reasoning capacity. 

**Abstract (ZH)**: 了解大规模语言模型（LLMs）的数学推理能力是人工智能研究中的一个核心主题。这一新领域要求创建用于训练和评估LLMs性能的推理任务数据集。为此，我们引入了Karp数据集：这是第一个由详细的NP完全性归约证明组成的数据集。这些归约在难度上有所不同，从本科课程中的简单练习到学术论文中的更具挑战性的归约。我们比较了当前最先进的模型在该任务上的表现，并展示了使用Karp数据集进行微调对推理能力的影响。 

---
# Chain-of-Retrieval Augmented Generation 

**Title (ZH)**: 链式检索增强生成 

**Authors**: Liang Wang, Haonan Chen, Nan Yang, Xiaolong Huang, Zhicheng Dou, Furu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2501.14342)  

**Abstract**: This paper introduces an approach for training o1-like RAG models that retrieve and reason over relevant information step by step before generating the final answer. Conventional RAG methods usually perform a single retrieval step before the generation process, which limits their effectiveness in addressing complex queries due to imperfect retrieval results. In contrast, our proposed method, CoRAG (Chain-of-Retrieval Augmented Generation), allows the model to dynamically reformulate the query based on the evolving state. To train CoRAG effectively, we utilize rejection sampling to automatically generate intermediate retrieval chains, thereby augmenting existing RAG datasets that only provide the correct final answer. At test time, we propose various decoding strategies to scale the model's test-time compute by controlling the length and number of sampled retrieval chains. Experimental results across multiple benchmarks validate the efficacy of CoRAG, particularly in multi-hop question answering tasks, where we observe more than 10 points improvement in EM score compared to strong baselines. On the KILT benchmark, CoRAG establishes a new state-of-the-art performance across a diverse range of knowledge-intensive tasks. Furthermore, we offer comprehensive analyses to understand the scaling behavior of CoRAG, laying the groundwork for future research aimed at developing factual and grounded foundation models. 

**Abstract (ZH)**: 本文介绍了一种训练O1-like RAG模型的方法，该方法在生成最终答案之前，逐步检索和推理相关的信息。传统的RAG方法通常在生成过程之前只进行一次检索步骤，这限制了它们在处理复杂查询时的有效性，尤其是在检索结果不够完善的情况下。相比之下，我们提出的方法CoRAG（Chain-of-Retrieval Augmented Generation）允许模型根据检索状态的演变动态重新构建查询。为了有效地训练CoRAG，我们利用拒绝采样自动生成中间检索链，从而增强现有仅提供正确最终答案的RAG数据集。在测试阶段，我们提出了多种解码策略，通过控制检索链的长度和数量来扩展模型的测试计算能力。在多个基准测试中的实验结果验证了CoRAG的有效性，特别是在多跳问答任务中，观察到EM分数相对于强基线模型有超过10点的改进。在KILT基准测试中，CoRAG在各种知识密集型任务中取得了新的最佳性能。此外，我们还提供了全面的分析，以理解CoRAG的扩展行为，为未来旨在开发事实性和实际 grounding 的基础模型的研究奠定了基础。 

---
# Fast Think-on-Graph: Wider, Deeper and Faster Reasoning of Large Language Model on Knowledge Graph 

**Title (ZH)**: 快速图思考：大型语言模型在知识图谱中更宽、更深、更快的推理方法 

**Authors**: Xujian Liang, Zhaoquan Gu  

**Link**: [PDF](https://arxiv.org/pdf/2501.14300)  

**Abstract**: Graph Retrieval Augmented Generation (GRAG) is a novel paradigm that takes the naive RAG system a step further by integrating graph information, such as knowledge graph (KGs), into large-scale language models (LLMs) to mitigate hallucination. However, existing GRAG still encounter limitations: 1) simple paradigms usually fail with the complex problems due to the narrow and shallow correlations capture from KGs 2) methods of strong coupling with KGs tend to be high computation cost and time consuming if the graph is dense. In this paper, we propose the Fast Think-on-Graph (FastToG), an innovative paradigm for enabling LLMs to think ``community by community" within KGs. To do this, FastToG employs community detection for deeper correlation capture and two stages community pruning - coarse and fine pruning for faster retrieval. Furthermore, we also develop two Community-to-Text methods to convert the graph structure of communities into textual form for better understanding by LLMs. Experimental results demonstrate the effectiveness of FastToG, showcasing higher accuracy, faster reasoning, and better explainability compared to the previous works. 

**Abstract (ZH)**: Graph Retrieval Augmented Generation (GRAG) 是一种新颖的范式，通过将知识图谱（KGs）等图信息整合到大规模语言模型（LLMs）中，进一步缓解了幻觉问题。然而，现有的 GRAG 仍然存在一些局限性：1）简单的范式难以应对复杂的任务，因为 KGs 中捕获的关联较为狭窄和浅显；2）与 KGs 强耦合的方法往往在图稠密时计算成本高且耗时。在本文中，我们提出了一种名为 Fast Think-on-Graph（FastToG）的创新范式，旨在使 LLMs 在 KGs 中能够“社区逐个社区”地进行思考。为了实现这一目标，FastToG 使用社区检测来捕获更深层次的关联，并采用两阶段社区剪枝——粗修剪和细修剪，以加快检索速度。此外，我们还开发了两种社区到文本的方法，将社区的图结构转换为文本形式，以便更好地被 LLMs 理解。实验结果表明，FastToG 的有效性，体现在更高的准确性、更快的推理速度和更好的可解释性方面，优于以前的工作。 

---
# Humanity's Last Exam 

**Title (ZH)**: 人类的最后一试 

**Authors**: Long Phan, Alice Gatti, Ziwen Han, Nathaniel Li, Josephina Hu, Hugh Zhang, Sean Shi, Michael Choi, Anish Agrawal, Arnav Chopra, Adam Khoja, Ryan Kim, Jason Hausenloy, Oliver Zhang, Mantas Mazeika, Daron Anderson, Tung Nguyen, Mobeen Mahmood, Fiona Feng, Steven Y. Feng, Haoran Zhao, Michael Yu, Varun Gangal, Chelsea Zou, Zihan Wang, Jessica P. Wang, Pawan Kumar, Oleksandr Pokutnyi, Robert Gerbicz, Serguei Popov, John-Clark Levin, Mstyslav Kazakov, Johannes Schmitt, Geoff Galgon, Alvaro Sanchez, Yongki Lee, Will Yeadon, Scott Sauers, Marc Roth, Chidozie Agu, Søren Riis, Fabian Giska, Saiteja Utpala, Zachary Giboney, Gashaw M. Goshu, Joan of Arc Xavier, Sarah-Jane Crowson, Mohinder Maheshbhai Naiya, Noah Burns, Lennart Finke, Zerui Cheng, Hyunwoo Park, Francesco Fournier-Facio, John Wydallis, Mark Nandor, Ankit Singh, Tim Gehrunger, Jiaqi Cai, Ben McCarty, Darling Duclosel, Jungbae Nam, Jennifer Zampese, Ryan G. Hoerr, Aras Bacho, Gautier Abou Loume, Abdallah Galal, Hangrui Cao, Alexis C Garretson, Damien Sileo, Qiuyu Ren, Doru Cojoc, Pavel Arkhipov, Usman Qazi, Lianghui Li, Sumeet Motwani, Christian Schroeder de Witt, Edwin Taylor, Johannes Veith, Eric Singer, Taylor D. Hartman, Paolo Rissone, Jaehyeok Jin, Jack Wei Lun Shi, Chris G. Willcocks, Joshua Robinson, Aleksandar Mikov, Ameya Prabhu, Longke Tang, Xavier Alapont, Justine Leon Uro, Kevin Zhou, Emily de Oliveira Santos, Andrey Pupasov Maksimov, Edward Vendrow, Kengo Zenitani, Julien Guillod, Yuqi Li, Joshua Vendrow, Vladyslav Kuchkin, Ng Ze-An  

**Link**: [PDF](https://arxiv.org/pdf/2501.14249)  

**Abstract**: Benchmarks are important tools for tracking the rapid advancements in large language model (LLM) capabilities. However, benchmarks are not keeping pace in difficulty: LLMs now achieve over 90\% accuracy on popular benchmarks like MMLU, limiting informed measurement of state-of-the-art LLM capabilities. In response, we introduce Humanity's Last Exam (HLE), a multi-modal benchmark at the frontier of human knowledge, designed to be the final closed-ended academic benchmark of its kind with broad subject coverage. HLE consists of 3,000 questions across dozens of subjects, including mathematics, humanities, and the natural sciences. HLE is developed globally by subject-matter experts and consists of multiple-choice and short-answer questions suitable for automated grading. Each question has a known solution that is unambiguous and easily verifiable, but cannot be quickly answered via internet retrieval. State-of-the-art LLMs demonstrate low accuracy and calibration on HLE, highlighting a significant gap between current LLM capabilities and the expert human frontier on closed-ended academic questions. To inform research and policymaking upon a clear understanding of model capabilities, we publicly release HLE at this https URL. 

**Abstract (ZH)**: 基准是跟踪大规模语言模型（LLM）能力快速进展的重要工具。然而，基准的难度并未跟上：LLM 现在在流行基准如 MMLU 上的准确率超过 90%，这限制了对最新 LLM 能力的深入了解。为应对这一问题，我们引入了人类最后考试（HLE），这是一个前沿的多模态基准，旨在成为此类基准中的最后一项闭卷学术基准，涵盖广泛的学科领域。HLE 包含 3,000 道题目，涵盖数学、人文学科和自然科学等多个学科。HLE 由各学科领域的专家全球开发，其中包括选择题和简答题，适合自动化评分。每个问题都有已知且明确的正确答案，可以方便验证，但不能通过互联网检索迅速得出答案。最先进的 LLM 在 HLE 上的准确率和校准度较低，突显当前 LLM 能力与闭卷学术问题上的专家人类前沿之间存在显著差距。为明确了解模型的能力，以指导研究和政策制定，我们在本处公开发布了 HLE（https://example.com/HLE）。 

---
# QuanTaxo: A Quantum Approach to Self-Supervised Taxonomy Expansion 

**Title (ZH)**: QuanTaxo：一种量子自监督税务分类扩展方法 

**Authors**: Sahil Mishra, Avi Patni, Niladri Chatterjee, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2501.14011)  

**Abstract**: A taxonomy is a hierarchical graph containing knowledge to provide valuable insights for various web applications. Online retail organizations like Microsoft and Amazon utilize taxonomies to improve product recommendations and optimize advertisement by enhancing query interpretation. However, the manual construction of taxonomies requires significant human effort. As web content continues to expand at an unprecedented pace, existing taxonomies risk becoming outdated, struggling to incorporate new and emerging information effectively. As a consequence, there is a growing need for dynamic taxonomy expansion to keep them relevant and up-to-date. Existing taxonomy expansion methods often rely on classical word embeddings to represent entities. However, these embeddings fall short in capturing hierarchical polysemy, where an entity's meaning can vary based on its position in the hierarchy and its surrounding context. To address this challenge, we introduce QuanTaxo, an innovative quantum-inspired framework for taxonomy expansion. QuanTaxo encodes entity representations in quantum space, effectively modeling hierarchical polysemy by leveraging the principles of Hilbert space to capture interference effects between entities, yielding richer and more nuanced representations. Comprehensive experiments on four real-world benchmark datasets show that QuanTaxo significantly outperforms classical embedding models, achieving substantial improvements of 18.45% in accuracy, 20.5% in Mean Reciprocal Rank, and 17.87% in Wu & Palmer metrics across eight classical embedding-based baselines. We further highlight the superiority of QuanTaxo through extensive ablation and case studies. 

**Abstract (ZH)**: -taxonomy 是一种包含知识的层次图，为各种网络应用提供有价值的洞察。像微软和亚马逊这样的在线零售组织利用 taxonomy 来通过改进查询解释来提升产品推荐和优化广告。然而，手工构建 taxonomy 要求大量的人力投入。随着 web 内容以前所未有的速度扩展，现有的 taxonomy 也面临着过时的风险，难以有效纳入新的和出现的资讯。因此，为了使 taxonomy 保持相关性和时效性，动态 taxonomy 扩展的需求日益增加。现有的 taxonomy 扩展方法通常依赖于经典的词嵌入来表示实体，但这些嵌入无法充分捕捉层次上的多义性，即实体的意义可以根据其在层次中的位置及其周围上下文而变化。为了解决这一挑战，我们引入了 QuanTaxo，一种受量子启发的 taxonomy 扩展框架。QuanTaxo 将实体表示编码到量子空间中，通过利用希尔伯特空间的原理来捕捉实体之间的干涉效应，从而更有效地建模层次上的多义性，提供更丰富和细腻的表示。在四个真实世界基准数据集上的全面实验表明，QuanTaxo 在准确性、平均互倒数排名和 Wu & Palmer 测量指标方面，相较于基于经典词嵌入的八大基线模型，分别取得了 18.45%、20.5% 和 17.87% 的显著提高。我们还通过广泛的消融实验和案例研究进一步突显了 QuanTaxo 的优越性。 

---
# GaussMark: A Practical Approach for Structural Watermarking of Language Models 

**Title (ZH)**: GaussMark：一种实用的语言模型结构水印方法 

**Authors**: Adam Block, Ayush Sekhari, Alexander Rakhlin  

**Link**: [PDF](https://arxiv.org/pdf/2501.13941)  

**Abstract**: Recent advances in Large Language Models (LLMs) have led to significant improvements in natural language processing tasks, but their ability to generate human-quality text raises significant ethical and operational concerns in settings where it is important to recognize whether or not a given text was generated by a human. Thus, recent work has focused on developing techniques for watermarking LLM-generated text, i.e., introducing an almost imperceptible signal that allows a provider equipped with a secret key to determine if given text was generated by their model. Current watermarking techniques are often not practical due to concerns with generation latency, detection time, degradation in text quality, or robustness. Many of these drawbacks come from the focus on token-level watermarking, which ignores the inherent structure of text. In this work, we introduce a new scheme, GaussMark, that is simple and efficient to implement, has formal statistical guarantees on its efficacy, comes at no cost in generation latency, and embeds the watermark into the weights of the model itself, providing a structural watermark. Our approach is based on Gaussian independence testing and is motivated by recent empirical observations that minor additive corruptions to LLM weights can result in models of identical (or even improved) quality. We show that by adding a small amount of Gaussian noise to the weights of a given LLM, we can watermark the model in a way that is statistically detectable by a provider who retains the secret key. We provide formal statistical bounds on the validity and power of our procedure. Through an extensive suite of experiments, we demonstrate that GaussMark is reliable, efficient, and relatively robust to corruptions such as insertions, deletions, substitutions, and roundtrip translations and can be instantiated with essentially no loss in model quality. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的进展显著提升了自然语言处理任务的效果，但它们生成人类质量文本的能力在需要区分文本是否由人类生成的环境中引发了重大的伦理和操作担忧。因此，最近的研究工作集中于开发用于水印LLM生成文本的技术，即引入几乎不可察觉的信号，使得配备秘密密钥的提供者能够确定给定的文本是否由他们的模型生成。当前的水印技术由于生成延迟、检测时间、文本质量下降或鲁棒性等方面的担忧，往往不太实用。这些缺点大多源自对令牌级别水印的关注，而忽略了文本固有的结构。在此研究中，我们提出了一种新的方案——GaussMark，该方案简单且易于实现，具有形式化的统计保证，在生成延迟上没有任何成本，并将水印嵌入模型的权重中，提供了一种结构性水印。我们的方法基于高斯独立性检验，并受到最近的经验观察的启发，即对LLM权重进行轻微的加性破坏可以导致模型质量相同（甚至更好）。我们证明通过向给定的LLM权重中添加少量高斯噪声，可以以一种统计上可检测的方式水印模型，提供给保留秘密密钥的提供者。我们提供了关于此程序有效性和功效的正式统计界线。通过一系列广泛的实验，我们展示了GaussMark的可靠性、效率以及对其它诸如插入、删除、替代和往返翻译等破坏具有相对的鲁棒性，且几乎不会损失模型质量。 

---
# Evaluating Computational Accuracy of Large Language Models in Numerical Reasoning Tasks for Healthcare Applications 

**Title (ZH)**: 评估大型语言模型在医疗应用中数值推理任务计算准确性 

**Authors**: Arjun R. Malghan  

**Link**: [PDF](https://arxiv.org/pdf/2501.13936)  

**Abstract**: Large Language Models (LLMs) have emerged as transformative tools in the healthcare sector, demonstrating remarkable capabilities in natural language understanding and generation. However, their proficiency in numerical reasoning, particularly in high-stakes domains like in clinical applications, remains underexplored. Numerical reasoning is critical in healthcare applications, influencing patient outcomes, treatment planning, and resource allocation. This study investigates the computational accuracy of LLMs in numerical reasoning tasks within healthcare contexts. Using a curated dataset of 1,000 numerical problems, encompassing real-world scenarios such as dosage calculations and lab result interpretations, the performance of a refined LLM based on the GPT-3 architecture was evaluated. The methodology includes prompt engineering, integration of fact-checking pipelines, and application of regularization techniques to enhance model accuracy and generalization. Key metrics such as precision, recall, and F1-score were utilized to assess the model's efficacy. The results indicate an overall accuracy of 84.10%, with improved performance in straightforward numerical tasks and challenges in multi-step reasoning. The integration of a fact-checking pipeline improved accuracy by 11%, underscoring the importance of validation mechanisms. This research highlights the potential of LLMs in healthcare numerical reasoning and identifies avenues for further refinement to support critical decision-making in clinical environments. The findings aim to contribute to the development of reliable, interpretable, and contextually relevant AI tools for healthcare. 

**Abstract (ZH)**: 大型语言模型（LLMs）在医疗保健领域中崭露头角，展示了在自然语言理解和生成方面的非凡能力。然而，它们在数值推理方面的 proficiency，尤其是在临床等高风险领域中，仍待进一步探索。数值推理在医疗保健应用中至关重要，影响患者结果、治疗规划和资源分配。本研究旨在探讨LLMs在医疗保健背景下数值推理任务中的计算准确度。研究使用了一个由1000个数值问题组成的精心策划的数据集，涵盖了诸如剂量计算和实验室结果解读等实际场景，评估了一种基于GPT-3架构的优化后的LLM的性能。研究方法包括提示工程、事实核查流程的整合以及正则化技术的应用，以提高模型的准确度和泛化能力。采用诸如精确度、召回率和F1分数等关键指标来评估模型的有效性。结果表明总体准确率为84.10%，在简单的数值任务中表现良好，但在多步推理问题上面临挑战。事实核查流程的整合提高了11%的准确度，凸显了验证机制的重要性。本研究揭示了LLMs在医疗保健数值推理中的潜力，并指出了支持临床环境中的关键决策的进一步优化途径。研究结果旨在促进开发可靠、可解释且情境相关的人工智能工具，以支持医疗保健。 

---
