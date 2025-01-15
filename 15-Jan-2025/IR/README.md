# TriMod Fusion for Multimodal Named Entity Recognition in Social Media 

**Title (ZH)**: 社交媒体多模态命名实体识别的TriMod融合方法 

**Authors**: Mosab Alfaqeeh  

**Link**: [PDF](https://arxiv.org/pdf/2501.08267)  

**Abstract**: Social media platforms serve as invaluable sources of user-generated content, offering insights into various aspects of human behavior. Named Entity Recognition (NER) plays a crucial role in analyzing such content by identifying and categorizing named entities into predefined classes. However, traditional NER models often struggle with the informal, contextually sparse, and ambiguous nature of social media language. To address these challenges, recent research has focused on multimodal approaches that leverage both textual and visual cues for enhanced entity recognition. Despite advances, existing methods face limitations in capturing nuanced mappings between visual objects and textual entities and addressing distributional disparities between modalities. In this paper, we propose a novel approach that integrates textual, visual, and hashtag features (TriMod), utilizing Transformer-attention for effective modality fusion. The improvements exhibited by our model suggest that named entities can greatly benefit from the auxiliary context provided by multiple modalities, enabling more accurate recognition. Through the experiments on a multimodal social media dataset, we demonstrate the superiority of our approach over existing state-of-the-art methods, achieving significant improvements in precision, recall, and F1 score. 

**Abstract (ZH)**: 社交媒体平台是用户生成内容的重要来源，为人类行为的各个层面提供了见解。命名实体识别（NER）在分析此类内容时起着至关重要的作用，通过识别和将命名实体分类到预定义的类别中。然而，传统NER模型往往难以处理社交媒体语言的非正式性、上下文稀疏性和模糊性。为应对这些挑战，最近的研究集中于利用文本和视觉线索的多模态方法以增强实体识别效果。尽管取得了进展，但现有方法在捕捉视觉对象和文本实体之间的细微映射关系以及解决模态分布差异方面仍存在局限性。在本文中，我们提出了一种新颖的方法，即结合文本、视觉和标签特征的TriMod模型，利用Transformer-attention实现有效的模态融合。我们的模型表现改进表明，命名实体可以从多种模态提供的辅助上下文中显著受益，从而实现更准确的识别。通过在多模态社交媒体数据集上的实验，我们展示了该方法优于现有最先进的方法，实现了在精确度、召回率和F1分数上的显著提升。 

---
# Unsupervised Query Routing for Retrieval Augmented Generation 

**Title (ZH)**: 无监督查询路由以增强检索生成 

**Authors**: Feiteng Mu, Liwen Zhang, Yong Jiang, Wenjie Li, Zhen Zhang, Pengjun Xie, Fei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2501.07793)  

**Abstract**: Query routing for retrieval-augmented generation aims to assign an input query to the most suitable search engine. Existing works rely heavily on supervised datasets that require extensive manual annotation, resulting in high costs and limited scalability, as well as poor generalization to out-of-distribution scenarios. To address these challenges, we introduce a novel unsupervised method that constructs the "upper-bound" response to evaluate the quality of retrieval-augmented responses. This evaluation enables the decision of the most suitable search engine for a given query. By eliminating manual annotations, our approach can automatically process large-scale real user queries and create training data. We conduct extensive experiments across five datasets, demonstrating that our method significantly enhances scalability and generalization capabilities. 

**Abstract (ZH)**: 检索增强生成中的查询路由旨在将输入查询分配给最适合的搜索引擎。现有工作主要依赖于需要大量人工标注的监督数据集，导致成本高且扩展有限，并且在处理分布外场景时表现不佳。为了应对这些挑战，我们提出了一种新的无监督方法，该方法构建了“上限”响应以评估检索增强响应的质量。通过这种评估，可以为给定查询决定最适合的搜索引擎。通过消除人工标注，我们的方法可以自动处理大规模的实际用户查询并生成训练数据。我们通过五个数据集进行了广泛的实验，表明我们的方法在可扩展性和泛化能力方面显著增强。 

---
# Constructing Set-Compositional and Negated Representations for First-Stage Ranking 

**Title (ZH)**: 构建集组合性和否定性表示以用于第一阶段排序 

**Authors**: Antonios Minas Krasakis, Andrew Yates, Evangelos Kanoulas  

**Link**: [PDF](https://arxiv.org/pdf/2501.07679)  

**Abstract**: Set compositional and negated queries are crucial for expressing complex information needs and enable the discovery of niche items like Books about non-European monarchs. Despite the recent advances in LLMs, first-stage ranking remains challenging due to the requirement of encoding documents and queries independently from each other. This limitation calls for constructing compositional query representations that encapsulate logical operations or negations, and can be used to match relevant documents effectively. In the first part of this work, we explore constructing such representations in a zero-shot setting using vector operations between lexically grounded Learned Sparse Retrieval (LSR) representations. Specifically, we introduce Disentangled Negation that penalizes only the negated parts of a query, and a Combined Pseudo-Term approach that enhances LSRs ability to handle intersections. We find that our zero-shot approach is competitive and often outperforms retrievers fine-tuned on compositional data, highlighting certain limitations of LSR and Dense Retrievers. Finally, we address some of these limitations and improve LSRs representation power for negation, by allowing them to attribute negative term scores and effectively penalize documents containing the negated terms. 

**Abstract (ZH)**: 复合查询和否定查询对于表达复杂的信息需求至关重要，并能帮助发现诸如关于非欧洲王公的书籍之类的特定物品。尽管近年来大语言模型（LLMs）取得了进展，但在第一阶段排名任务上仍面临挑战，主要是因为在独立编码文档和查询方面的要求。这一限制促使我们构建能够封装逻辑操作或否定的复合查询表示，并利用这些表示有效地匹配相关文档。在本文的第一部分，我们探索在零样本设置下通过词汇扎根的Learned Sparse Retrieval（LSR）表示之间的向量操作来构建此类表示。具体而言，我们引入了分离的否定方法，该方法仅惩罚查询中的否定部分，并提出了一种组合伪术语方法，以增强LSR处理交集的能力。我们发现我们的零样本方法具有竞争力，并且通常优于在复合数据上进行微调的检索器，这突显了某些LSR和密集检索器的局限性。最后，我们解决了一些这些局限性，并通过允许LSR为否定术语分配分数并有效地惩罚包含否定术语的文档来提高其表示能力。 

---
# Eliciting In-context Retrieval and Reasoning for Long-context Large Language Models 

**Title (ZH)**: 激发上下文检索与推理以适应长上下文的大语言模型 

**Authors**: Yifu Qiu, Varun Embar, Yizhe Zhang, Navdeep Jaitly, Shay B. Cohen, Benjamin Han  

**Link**: [PDF](https://arxiv.org/pdf/2501.08248)  

**Abstract**: Recent advancements in long-context language models (LCLMs) promise to transform Retrieval-Augmented Generation (RAG) by simplifying pipelines. With their expanded context windows, LCLMs can process entire knowledge bases and perform retrieval and reasoning directly -- a capability we define as In-Context Retrieval and Reasoning (ICR^2). However, existing benchmarks like LOFT often overestimate LCLM performance by providing overly simplified contexts. To address this, we introduce ICR^2, a benchmark that evaluates LCLMs in more realistic scenarios by including confounding passages retrieved with strong retrievers. We then propose three methods to enhance LCLM performance: (1) retrieve-then-generate fine-tuning, (2) retrieval-attention-probing, which uses attention heads to filter and de-noise long contexts during decoding, and (3) joint retrieval head training alongside the generation head. Our evaluation of five well-known LCLMs on LOFT and ICR^2 demonstrates significant gains with our best approach applied to Mistral-7B: +17 and +15 points by Exact Match on LOFT, and +13 and +2 points on ICR^2, compared to vanilla RAG and supervised fine-tuning, respectively. It even outperforms GPT-4-Turbo on most tasks despite being a much smaller model. 

**Abstract (ZH)**: 近期在长上下文语言模型（LCLMs）方面的进展有望通过简化管道来改变检索增强生成（RAG）的技术。随着它们扩大了上下文窗口，LCLMs能够处理整个知识库，并直接进行检索和推理——我们将其定义为上下文检索与推理（ICR^2）。然而，现有的基准测试如LOFT通常通过提供过于简化的上下文来高估LCLM的表现。为了解决这个问题，我们提出了ICR^2这一基准测试，它通过包含使用强大检索器检索出的混杂段落来评估LCLMs在更现实场景中的表现。然后，我们提出了三种方法以提升LCLM的表现：（1）检索后再生成的微调，（2）检索注意力探查，该方法使用注意力头在解码过程中过滤和去噪长上下文，以及（3）联合训练检索头部和生成头部。我们在LOFT和ICR^2上对五种知名的LCLMs进行的评估表明，通过将我们的最佳方法应用于Mistral-7B，可以实现显著的性能提升：与传统的RAG和监督微调相比，在LOFT中Exact Match得分分别提高了17和15分，在ICR^2中分别提高了13和2分。尽管Mistral-7B是一个较小的模型，但在大多数任务中，它甚至超越了GPT-4-Turbo。 

---
# Optimize Incompatible Parameters through Compatibility-aware Knowledge Integration 

**Title (ZH)**: 通过兼容性意识的知识整合优化不兼容参数 

**Authors**: Zheqi Lv, Keming Ye, Zishu Wei, Qi Tian, Shengyu Zhang, Wenqiao Zhang, Wenjie Wang, Kun Kuang, Tat-Seng Chua, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2501.07596)  

**Abstract**: Deep neural networks have become foundational to advancements in multiple domains, including recommendation systems, natural language processing, and so on. Despite their successes, these models often contain incompatible parameters that can be underutilized or detrimental to model performance, particularly when faced with specific, varying data distributions. Existing research excels in removing such parameters or merging the outputs of multiple different pretrained models. However, the former focuses on efficiency rather than performance, while the latter requires several times more computing and storage resources to support inference. In this paper, we set the goal to explicitly improve these incompatible parameters by leveraging the complementary strengths of different models, thereby directly enhancing the models without any additional parameters. Specifically, we propose Compatibility-aware Knowledge Integration (CKI), which consists of Parameter Compatibility Assessment and Parameter Splicing, which are used to evaluate the knowledge content of multiple models and integrate the knowledge into one model, respectively. The integrated model can be used directly for inference or for further fine-tuning. We conduct extensive experiments on various datasets for recommendation and language tasks, and the results show that Compatibility-aware Knowledge Integration can effectively optimize incompatible parameters under multiple tasks and settings to break through the training limit of the original model without increasing the inference cost. 

**Abstract (ZH)**: 深度神经网络已成为多个领域发展的基础，包括推荐系统、自然语言处理等。尽管取得了一定的成功，但这些模型往往包含不兼容的参数，这些参数可能导致模型性能的浪费或下降，特别是在面对特定和变化的数据分布时更为明显。现有研究在移除这些不兼容参数或合并多个预训练模型的输出方面表现出色。然而，前者侧重于效率而非性能，而后者则需要更多的计算和存储资源来支持推理。在本文中，我们的目标是通过利用不同模型互补的优势，显式地改进这些不兼容的参数，从而直接提升模型的性能而不增加额外参数。具体而言，我们提出了兼容性感知知识集成（CKI），它由参数兼容性评估和参数拼接组成，分别用于评估多个模型的知识内容并将知识整合到一个模型中。整合后的模型可以直接用于推理或进一步微调。我们在推荐和语言任务的各种数据集上进行了广泛的实验，结果表明，兼容性感知知识集成可以有效优化多个任务和场景下不兼容的参数，从而突破原始模型的训练限制，而不增加推理成本。 

---
