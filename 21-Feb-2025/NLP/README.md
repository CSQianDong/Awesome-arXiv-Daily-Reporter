# LServe: Efficient Long-sequence LLM Serving with Unified Sparse Attention 

**Title (ZH)**: LServe：高效统一稀疏注意机制的长序列LLM服务 

**Authors**: Shang Yang, Junxian Guo, Haotian Tang, Qinghao Hu, Guangxuan Xiao, Jiaming Tang, Yujun Lin, Zhijian Liu, Yao Lu, Song Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.14866)  

**Abstract**: Large language models (LLMs) have shown remarkable potential in processing long sequences, yet efficiently serving these long-context models remains challenging due to the quadratic computational complexity of attention in the prefilling stage and the large memory footprint of the KV cache in the decoding stage. To address these issues, we introduce LServe, an efficient system that accelerates long-sequence LLM serving via hybrid sparse attention. This method unifies different hardware-friendly, structured sparsity patterns for both prefilling and decoding attention into a single framework, where computations on less important tokens are skipped block-wise. LServe demonstrates the compatibility of static and dynamic sparsity in long-context LLM attention. This design enables multiplicative speedups by combining these optimizations. Specifically, we convert half of the attention heads to nearly free streaming heads in both the prefilling and decoding stages. Additionally, we find that only a constant number of KV pages is required to preserve long-context capabilities, irrespective of context length. We then design a hierarchical KV page selection policy that dynamically prunes KV pages based on query-centric similarity. On average, LServe accelerates LLM prefilling by up to 2.9x and decoding by 1.3-2.1x over vLLM, maintaining long-context accuracy. Code is released at this https URL. 

**Abstract (ZH)**: 大语言模型（LLMs）在处理长序列方面展现了显著潜力，但在预填充阶段的注意力机制计算复杂度为二次方，以及解码阶段的KV缓存占用大量内存，使得高效服务这些长上下文模型仍然具有挑战性。为了解决这些问题，我们引入了LServe，这是一种通过混合稀疏注意力加速长序列LLM服务的高效系统。该方法将预填充和解码注意力的不同硬件友好型、结构化稀疏模式统一到一个框架中，在该框架中，对于不重要的令牌，在块级别跳过计算。LServe展示了静态和动态稀疏性在长上下文LLM注意力中的兼容性。这一设计通过结合这些优化实现了乘法加速。具体来说，我们将在预填充和解码两个阶段将一半的注意力头转换为接近免费的流式头。此外，我们发现，为了保持长上下文能力，仅需固定数量的KV页面，而与上下文长度无关。我们随后设计了一个分层的KV页面选择策略，基于查询相关性动态剪枝KV页面。平均而言，LServe在LLM预填充方面比vLLM快2.9倍，在解码方面快1.3到2.1倍，同时保持了长上下文准确性。相关代码已发布，详见[此链接](this https URL)。 

---
# Interpretable Text Embeddings and Text Similarity Explanation: A Primer 

**Title (ZH)**: 可 interpretability 文本嵌入与文本相似性解释：入门指南 

**Authors**: Juri Opitz, Lucas Möller, Andrianos Michail, Simon Clematide  

**Link**: [PDF](https://arxiv.org/pdf/2502.14862)  

**Abstract**: Text embeddings and text embedding models are a backbone of many AI and NLP systems, particularly those involving search. However, interpretability challenges persist, especially in explaining obtained similarity scores, which is crucial for applications requiring transparency. In this paper, we give a structured overview of interpretability methods specializing in explaining those similarity scores, an emerging research area. We study the methods' individual ideas and techniques, evaluating their potential for improving interpretability of text embeddings and explaining predicted similarities. 

**Abstract (ZH)**: 文本嵌入和文本嵌入模型是许多AI和NLP系统的核心，特别是在涉及搜索的应用中。然而，解释性挑战仍然存在，尤其是在解释获得的相似度分数方面，这对于需要透明度的应用至关重要。在本文中，我们提供了一个结构化的综述，专注于解释这些相似度分数的解释性方法，这是一个新兴的研究领域。我们研究了这些方法的独特理念和技术，并评估了它们提高文本嵌入的解释性和解释预测相似度的潜力。 

---
# Aligning LLMs to Ask Good Questions A Case Study in Clinical Reasoning 

**Title (ZH)**: 将大语言模型导向提出 good 问题：临床推理案例研究 

**Authors**: Shuyue Stella Li, Jimin Mun, Faeze Brahman, Jonathan S. Ilgen, Yulia Tsvetkov, Maarten Sap  

**Link**: [PDF](https://arxiv.org/pdf/2502.14860)  

**Abstract**: Large language models (LLMs) often fail to ask effective questions under uncertainty, making them unreliable in domains where proactive information-gathering is essential for decisionmaking. We present ALFA, a framework that improves LLM question-asking by (i) decomposing the notion of a "good" question into a set of theory-grounded attributes (e.g., clarity, relevance), (ii) controllably synthesizing attribute-specific question variations, and (iii) aligning models via preference-based optimization to explicitly learn to ask better questions along these fine-grained attributes. Focusing on clinical reasoning as a case study, we introduce the MediQ-AskDocs dataset, composed of 17k real-world clinical interactions augmented with 80k attribute-specific preference pairs of follow-up questions, as well as a novel expert-annotated interactive healthcare QA task to evaluate question-asking abilities. Models aligned with ALFA reduce diagnostic errors by 56.6% on MediQ-AskDocs compared to SOTA instruction-tuned LLMs, with a question-level win-rate of 64.4% and strong generalizability. Our findings suggest that explicitly guiding question-asking with structured, fine-grained attributes offers a scalable path to improve LLMs, especially in expert application domains. 

**Abstract (ZH)**: 大型语言模型（LLMs）在不确定性情况下往往不能提出有效的问题，这使得它们在需要主动信息收集以做出决策的领域中不可靠。我们提出了一种名为ALFA的框架，通过（i）将“好问题”的概念分解为一套理论支持的特征（例如，清晰度、相关性），（ii）有控制地生成特定特征的问题变体，以及（iii）通过基于偏好的优化对模型进行对齐，以明确学习在这些细粒度特征上提出更好问题。以临床推理为例，我们介绍了MediQ-AskDocs数据集，该数据集包含17,000个真实的临床互动，并附加了80,000个针对随访问题的具体特征偏好对，以及一项新的专家注释的互动医疗问答任务，用于评估提问能力。与最先进的指令调优的LLM相比，使用ALFA对齐的模型在MediQ-AskDocs上将诊断错误减少了56.6%，问题级别的胜出率为64.4%，且具有较强的泛化能力。我们的研究结果表明，明确地用结构化的细粒度特征引导提问为提高LLM提供了可扩展的路径，特别是在专家应用领域。 

---
# FR-Spec: Accelerating Large-Vocabulary Language Models via Frequency-Ranked Speculative Sampling 

**Title (ZH)**: FR-Spec: 通过频率排名推测性采样加速大规模词汇量语言模型 

**Authors**: Weilin Zhao, Tengyu Pan, Xu Han, Yudi Zhang, Ao Sun, Yuxiang Huang, Kaihuo Zhang, Weilun Zhao, Yuxuan Li, Jianyong Wang, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.14856)  

**Abstract**: Speculative sampling has emerged as an important technique for accelerating the auto-regressive generation process of large language models (LLMs) by utilizing a draft-then-verify mechanism to produce multiple tokens per forward pass. While state-of-the-art speculative sampling methods use only a single layer and a language modeling (LM) head as the draft model to achieve impressive layer compression, their efficiency gains are substantially reduced for large-vocabulary LLMs, such as Llama-3-8B with a vocabulary of 128k tokens. To address this, we present FR-Spec, a frequency-ranked speculative sampling framework that optimizes draft candidate selection through vocabulary space compression. By constraining the draft search to a frequency-prioritized token subset, our method reduces LM Head computation overhead by 75% while ensuring the equivalence of the final output distribution. Experiments across multiple datasets demonstrate an average of 1.12$\times$ speedup over the state-of-the-art speculative sampling method EAGLE-2. 

**Abstract (ZH)**: 推测采样作为一种重要技术，通过利用先草拟后验证的机制，在每次前向传递中生成多个令牌，从而加速了大型语言模型（LLMs）的自回归生成过程。尽管当前领先的推测采样方法仅使用单一层级及语言模型（LM）头部作为草拟模型，实现了显著的层压缩，但对于具有大词汇量的LLMs（如词汇量为128k的Llama-3-8B），其效率增益显著降低。为解决这一问题，我们提出了FR-Spec，这是一种基于频率排序的推测采样框架，通过词汇空间压缩优化草拟候选人的选择。通过将草拟搜索限制在一个以频率优先的令牌子集内，该方法将LM头部的计算开销减少了75%，同时确保最终输出分布的等效性。实验结果显示，与当前领先的推测采样方法EAGLE-2相比，平均加速了1.12倍。 

---
# CLIPPER: Compression enables long-context synthetic data generation 

**Title (ZH)**: CLIPPER：压缩 enables 长上下文合成数据生成

注：在翻译过程中，我们尝试保持原文的缩写和专有名词不变。但是“enables”在这里可能需要进一步明确，从英文原句来看，“enables”在这里可能意为“促进”或“使可能”，因此，更符合中文表达习惯的翻译可以是：

CLIPPER：压缩促进长上下文合成数据生成

这样更符合中文的学术表达习惯。 

**Authors**: Chau Minh Pham, Yapei Chang, Mohit Iyyer  

**Link**: [PDF](https://arxiv.org/pdf/2502.14854)  

**Abstract**: LLM developers are increasingly reliant on synthetic data, but generating high-quality data for complex long-context reasoning tasks remains challenging. We introduce CLIPPER, a compression-based approach for generating synthetic data tailored to narrative claim verification - a task that requires reasoning over a book to verify a given claim. Instead of generating claims directly from the raw text of the book, which results in artifact-riddled claims, CLIPPER first compresses the book into chapter outlines and book summaries and then uses these intermediate representations to generate complex claims and corresponding chain-of-thoughts. Compared to naive approaches, CLIPPER produces claims that are more valid, grounded, and complex. Using CLIPPER, we construct a dataset of 19K synthetic book claims paired with their source texts and chain-of-thought reasoning, and use it to fine-tune three open-weight models. Our best model achieves breakthrough results on narrative claim verification (from 28% to 76% accuracy on our test set) and sets a new state-of-the-art for sub-10B models on the NoCha leaderboard. Further analysis shows that our models generate more detailed and grounded chain-of-thought reasoning while also improving performance on other narrative understanding tasks (e.g., NarrativeQA). 

**Abstract (ZH)**: 大语言模型开发者越来越多地依赖合成数据，但在生成适用于复杂长语境推理任务的高质量数据方面仍然颇具挑战。我们提出了CLIPPER，这是一种基于压缩的方法，用于生成针对叙述性声明验证的定制合成数据——一个需要在书籍范围内进行推理以验证给定声明的任务。CLIPPER 不是从书的原始文本直接生成声明，这会导致声明中出现伪影，而是首先将书籍压缩为章节大纲和书籍摘要，然后利用这些中间表示生成复杂的声明及其对应的推理链。与简单的策略相比，CLIPPER 生成的声明更加有效、具有一致性且复杂度更高。通过使用CLIPPER，我们构建了一个包含19,000个合成书籍声明及其来源文本和推理链的数据集，并用其对三种开放权重模型进行了微调。我们最好的模型在叙述性声明验证上取得了突破性成果（测试集准确率从28%提高到76%），并在NoCha排行榜上成为少于10B参数模型的新最佳表现。进一步的分析表明，我们的模型生成了更详细和具有一致性的推理链，并且在其他叙述性理解任务（如NarrativeQA）上也提高了性能。 

---
# GATE: Graph-based Adaptive Tool Evolution Across Diverse Tasks 

**Title (ZH)**: GATE：基于图的适应性工具演化以应对多样化任务 

**Authors**: Jianwen Luo, Yiming Huang, Jinxiang Meng, Fangyu Lei, Shizhu He, Xiao Liu, Shanshan Jiang, Bin Dong, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14848)  

**Abstract**: Large Language Models (LLMs) have shown great promise in tool-making, yet existing frameworks often struggle to efficiently construct reliable toolsets and are limited to single-task settings. To address these challenges, we propose GATE (Graph-based Adaptive Tool Evolution), an adaptive framework that dynamically constructs and evolves a hierarchical graph of reusable tools across multiple scenarios. We evaluate GATE on open-ended tasks (Minecraft), agent-based tasks (TextCraft, DABench), and code generation tasks (MATH, Date, TabMWP). Our results show that GATE achieves up to 4.3x faster milestone completion in Minecraft compared to the previous SOTA, and provides an average improvement of 9.23% over existing tool-making methods in code generation tasks and 10.03% in agent tasks. GATE demonstrates the power of adaptive evolution, balancing tool quantity, complexity, and functionality while maintaining high efficiency. Code and data are available at \url{this https URL}. 

**Abstract (ZH)**: 大型语言模型（LLMs）在工具开发方面展现出了巨大潜力，但现有的框架在高效构建可靠工具集方面常常遇到挑战，并且主要局限于单任务设置。为了解决这些问题，我们提出了一种基于图的自适应工具演化框架（GATE，Graph-based Adaptive Tool Evolution），该框架能够动态构建和演化跨多个场景的可复用工具层次图。我们分别在开放式任务（Minecraft）、基于代理的任务（TextCraft、DABench）和代码生成任务（MATH、Date、TabMWP）上评估了GATE。实验结果显示，与前一个最佳方法相比，GATE在Minecraft中的里程碑完成速度提高了4.3倍，并在代码生成任务和代理任务中分别提供了平均9.23%和10.03%的改进。GATE展示了自适应演化的力量，既平衡了工具的数量、复杂性和功能，又保持了高效率。有关代码和数据，可访问以下链接：\url{this https URL}。 

---
# Revealing and Mitigating Over-Attention in Knowledge Editing 

**Title (ZH)**: 揭示和减轻知识编辑中的过度关注问题 

**Authors**: Pinzheng Wang, Zecheng Tang, Keyan Zhou, Juntao Li, Qiaoming Zhu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14838)  

**Abstract**: Large Language Models have demonstrated superior performance across a wide range of tasks, but they still exhibit undesirable errors due to incorrect knowledge learned from the training data. To avoid this, knowledge editing methods emerged to precisely edit the specific model knowledge via efficiently modifying a very small percentage of parameters. % However, those methods can lead to the problem of Specificity Failure: when the content related to the edited knowledge occurs in the context, it can inadvertently corrupt other pre-existing knowledge. However, those methods can lead to the problem of Specificity Failure, where the existing knowledge and capabilities are severely degraded due to editing. Our preliminary indicates that Specificity Failure primarily stems from the model's attention heads assigning excessive attention scores to entities related to the edited knowledge, thereby unduly focusing on specific snippets within the context, which we denote as the Attention Drift phenomenon. To mitigate such Attention Drift issue, we introduce a simple yet effective method Selective Attention Drift Restriction}(SADR), which introduces an additional regularization term during the knowledge editing process to restrict changes in the attention weight distribution, thereby preventing undue focus on the edited entity. Experiments on five frequently used strong LLMs demonstrate the effectiveness of our method, where SADR can significantly mitigate Specificity Failure in the predominant knowledge editing tasks. 

**Abstract (ZH)**: 大规模语言模型在各种任务中表现出色，但在训练数据中学习到的错误知识仍然会导致一些不希望出现的错误。为了解决这一问题，出现了知识编辑方法，通过高效地修改少量参数来精确编辑模型的知识。然而，这些方法可能会导致特异性失败（Specificity Failure）的问题：当与编辑知识相关的上下文内容出现时，可能会意外地破坏其他已存在的知识。我们的初步研究表明，特异性失败主要源于模型的注意力头过度分配注意力分数给与编辑知识相关的实体，从而不当关注上下文中的特定片段，我们将其称为注意力漂移（Attention Drift）现象。为缓解这种注意力漂移问题，我们提出了一种简单而有效的方法——选择性注意力漂移限制（Selective Attention Drift Restriction, SADR），该方法在知识编辑过程中引入了一个额外的正则化项，以限制注意权重分布的变化，从而防止过度关注编辑的知识实体。在五个常用的强大语言模型上的实验表明，我们的方法能够显著降低主流知识编辑任务中的特异性失败问题。 

---
# Towards Economical Inference: Enabling DeepSeek's Multi-Head Latent Attention in Any Transformer-based LLMs 

**Title (ZH)**: 向经济高效的推理迈进：在任意基于Transformer的大型语言模型中启用DeepSeek的多头潜在注意力机制 

**Authors**: Tao Ji, Bin Guo, Yuanbin Wu, Qipeng Guo, Lixing Shen, Zhan Chen, Xipeng Qiu, Qi Zhang, Tao Gui  

**Link**: [PDF](https://arxiv.org/pdf/2502.14837)  

**Abstract**: Multi-head Latent Attention (MLA) is an innovative architecture proposed by DeepSeek, designed to ensure efficient and economical inference by significantly compressing the Key-Value (KV) cache into a latent vector. Compared to MLA, standard LLMs employing Multi-Head Attention (MHA) and its variants such as Grouped-Query Attention (GQA) exhibit significant cost disadvantages. Enabling well-trained LLMs (e.g., Llama) to rapidly adapt to MLA without pre-training from scratch is both meaningful and challenging. This paper proposes the first data-efficient fine-tuning method for transitioning from MHA to MLA (MHA2MLA), which includes two key components: for partial-RoPE, we remove RoPE from dimensions of queries and keys that contribute less to the attention scores, for low-rank approximation, we introduce joint SVD approximations based on the pre-trained parameters of keys and values. These carefully designed strategies enable MHA2MLA to recover performance using only a small fraction (0.3% to 0.6%) of the data, significantly reducing inference costs while seamlessly integrating with compression techniques such as KV cache quantization. For example, the KV cache size of Llama2-7B is reduced by 92.19%, with only a 0.5% drop in LongBench performance. 

**Abstract (ZH)**: 多头潜在注意（MLA）是一种由DeepSeek提出的创新架构，旨在通过显著压缩关键值（KV）缓存为潜在向量来确保高效和经济的推理。与使用多头注意（MHA）及其变体（如分组查询注意（GQA））的标准大型语言模型（LLM）相比，MLA在成本方面具有明显的优势。使得经过充分训练的LLMs（例如，Llama）能够快速适应MLA而无需从头开始进行预训练，这既具有重要意义也具有挑战性。本文提出了一种新的数据高效微调方法（MHA2MLA），以实现从MHA到MLA的过渡，其中包括两个关键组件：对于部分RoPE，我们从对注意分数贡献较小的查询和关键维度中移除RoPE；对于低秩近似，我们引入基于预先训练的键和值参数的联合SVD近似。这些精心设计的策略使MHA2MLA仅使用少量数据（0.3%到0.6%）即可恢复性能，显著降低了推理成本，并能够无缝集成诸如KV缓存量化等压缩技术。例如，Llama2-7B的KV缓存大小减少了92.19%，而LongBench性能仅下降了0.5%。 

---
# Middle-Layer Representation Alignment for Cross-Lingual Transfer in Fine-Tuned LLMs 

**Title (ZH)**: 面向细调大规模语言模型的跨语言迁移中层表示对齐方法 

**Authors**: Danni Liu, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2502.14830)  

**Abstract**: While large language models demonstrate remarkable capabilities at task-specific applications through fine-tuning, extending these benefits across diverse languages is essential for broad accessibility. However, effective cross-lingual transfer is hindered by LLM performance gaps across languages and the scarcity of fine-tuning data in many languages. Through analysis of LLM internal representations from over 1,000+ language pairs, we discover that middle layers exhibit the strongest potential for cross-lingual alignment. Building on this finding, we propose a middle-layer alignment objective integrated into task-specific training. Our experiments on slot filling, machine translation, and structured text generation show consistent improvements in cross-lingual transfer, especially to lower-resource languages. The method is robust to the choice of alignment languages and generalizes to languages unseen during alignment. Furthermore, we show that separately trained alignment modules can be merged with existing task-specific modules, improving cross-lingual capabilities without full re-training. Our code is publicly available (this https URL). 

**Abstract (ZH)**: 尽管通过微调，大型语言模型在特定任务应用中展现了非凡的能力，但在多种语言之间推广应用这些优势对于实现广泛的适用性至关重要。然而，有效的跨语言迁移受到语言之间大型语言模型表现差异以及许多语言缺乏微调数据的阻碍。通过对超过1000多个语言对的大型语言模型内部表示进行分析，我们发现中间层具有最强的跨语言对齐潜力。基于这一发现，我们提出了一种将中间层对齐目标整合到特定任务训练过程中的方法。我们的实验在槽填充、机器翻译和结构化文本生成任务上显示了一致的跨语言迁移改进，特别是在低资源语言方面。该方法对于对齐语言的选择具有鲁棒性，并可以应用于在对齐过程中未见过的语言。此外，我们展示了独立训练的对齐模块可以与现有的特定任务模块合并，从而在无需完全重新训练的情况下改进跨语言能力。我们的代码已公开（请参见链接）。 

---
# Measuring Faithfulness of Chains of Thought by Unlearning Reasoning Steps 

**Title (ZH)**: 通过反学习推理步骤来衡量思维链的忠实性 

**Authors**: Martin Tutek, Fateme Hashemi Chaleshtori, Ana Marasović, Yonatan Belinkov  

**Link**: [PDF](https://arxiv.org/pdf/2502.14829)  

**Abstract**: When prompted to think step-by-step, language models (LMs) produce a chain of thought (CoT), a sequence of reasoning steps that the model supposedly used to produce its prediction. However, despite much work on CoT prompting, it is unclear if CoT reasoning is faithful to the models' parameteric beliefs. We introduce a framework for measuring parametric faithfulness of generated reasoning, and propose Faithfulness by Unlearning Reasoning steps (FUR), an instance of this framework. FUR erases information contained in reasoning steps from model parameters. We perform experiments unlearning CoTs of four LMs prompted on four multi-choice question answering (MCQA) datasets. Our experiments show that FUR is frequently able to change the underlying models' prediction by unlearning key steps, indicating when a CoT is parametrically faithful. Further analysis shows that CoTs generated by models post-unlearning support different answers, hinting at a deeper effect of unlearning. Importantly, CoT steps identified as important by FUR do not align well with human notions of plausbility, emphasizing the need for specialized alignment 

**Abstract (ZH)**: 在被提示进行逐步思考时，语言模型（LMs）会产生一个推理链（CoT），即模型据称用于生成其预测的一系列推理步骤。然而，尽管在CoT提示方面做了许多工作，但CoT推理是否忠实于模型的参数化信念尚不明确。我们提出了一种衡量生成推理参数化忠实性的框架，并提出了基于消除推理步骤（FUR）的方法，这是一种该框架的具体应用。FUR方法会从模型参数中删除包含在推理步骤中的信息。我们对四个LMs在四个多项选择题解答（MCQA）数据集上被提示生成的CoT进行了消除学习实验。实验结果显示，FUR经常能够通过消除关键步骤来改变模型的底层预测，这表明CoT可能是参数化的忠实性表现。进一步的分析表明，模型在消除学习后生成的CoT支持不同的答案，暗示了消除学习的更深层次影响。重要的是，FUR识别出的重要CoT步骤与人类对合理性的真实感知并不一致，这强调了需要专门的对齐方法。 

---
# eC-Tab2Text: Aspect-Based Text Generation from e-Commerce Product Tables 

**Title (ZH)**: eC-Tab2Text: 从电子商务产品表格中生成方面的文本 

**Authors**: Luis Antonio Gutiérrez Guanilo, Mir Tafseer Nayeem, Cristian López, Davood Rafiei  

**Link**: [PDF](https://arxiv.org/pdf/2502.14820)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional versatility across diverse domains, yet their application in e-commerce remains underexplored due to a lack of domain-specific datasets. To address this gap, we introduce eC-Tab2Text, a novel dataset designed to capture the intricacies of e-commerce, including detailed product attributes and user-specific queries. Leveraging eC-Tab2Text, we focus on text generation from product tables, enabling LLMs to produce high-quality, attribute-specific product reviews from structured tabular data. Fine-tuned models were rigorously evaluated using standard Table2Text metrics, alongside correctness, faithfulness, and fluency assessments. Our results demonstrate substantial improvements in generating contextually accurate reviews, highlighting the transformative potential of tailored datasets and fine-tuning methodologies in optimizing e-commerce workflows. This work highlights the potential of LLMs in e-commerce workflows and the essential role of domain-specific datasets in tailoring them to industry-specific challenges. 

**Abstract (ZH)**: 大型语言模型（LLMs）在多个领域展现了卓越的灵活性，但在电子商务领域的应用仍然相对未能充分探索，主要原因是缺乏特定领域的数据集。为解决这一问题，我们介绍了eC-Tab2Text，这是一个新型数据集，旨在捕捉电子商务的复杂性，包括详细的商品属性和用户特定的查询。利用eC-Tab2Text，我们专注于从产品表生成文本，使LLMs能够从结构化表格数据中生成高质量、属性特定的商品评论。经过微调的模型使用标准的Table2Text度量标准进行严格评估，并进行了准确性、忠实性和流畅性评估。实验结果表明，在生成上下文相关性更高的评论方面实现了显著改进，这突显了定制数据集和微调方法在优化电子商务工作流方面的潜力。这项工作展示了LLMs在电子商务工作流中的潜力，并强调了特定领域数据集在针对行业特定挑战进行定制方面的重要作用。 

---
# From RAG to Memory: Non-Parametric Continual Learning for Large Language Models 

**Title (ZH)**: 从RAG到记忆：面向大规模语言模型的非参数持续学习 

**Authors**: Bernal Jiménez Gutiérrez, Yiheng Shu, Weijian Qi, Sizhe Zhou, Yu Su  

**Link**: [PDF](https://arxiv.org/pdf/2502.14802)  

**Abstract**: Our ability to continuously acquire, organize, and leverage knowledge is a key feature of human intelligence that AI systems must approximate to unlock their full potential. Given the challenges in continual learning with large language models (LLMs), retrieval-augmented generation (RAG) has become the dominant way to introduce new information. However, its reliance on vector retrieval hinders its ability to mimic the dynamic and interconnected nature of human long-term memory. Recent RAG approaches augment vector embeddings with various structures like knowledge graphs to address some of these gaps, namely sense-making and associativity. However, their performance on more basic factual memory tasks drops considerably below standard RAG. We address this unintended deterioration and propose HippoRAG 2, a framework that outperforms standard RAG comprehensively on factual, sense-making, and associative memory tasks. HippoRAG 2 builds upon the Personalized PageRank algorithm used in HippoRAG and enhances it with deeper passage integration and more effective online use of an LLM. This combination pushes this RAG system closer to the effectiveness of human long-term memory, achieving a 7% improvement in associative memory tasks over the state-of-the-art embedding model while also exhibiting superior factual knowledge and sense-making memory capabilities. This work paves the way for non-parametric continual learning for LLMs. Our code and data will be released at this https URL. 

**Abstract (ZH)**: 我们持续获取、组织和利用知识的能力是人类智能的关键特征，这一能力是AI系统必须近似的，以便充分发挥其潜力。鉴于在大规模语言模型（LLMs）中持续学习的挑战，检索增强生成（RAG）已成为引入新信息的主要方式。然而，其对向量检索的依赖性阻碍了它模仿人类长期记忆的动态和相互关联的特性。最近的RAG方法通过引入如知识图谱等不同结构来弥补这些差距，以应对意义构建和关联性方面的问题。然而，它们在更基本的 factual 记忆任务上的表现明显低于标准RAG。本文针对这种意外的退化，提出了一种名为HippoRAG 2的框架，该框架在关于事实性、意义构建和关联性记忆任务上全面优于标准RAG。HippoRAG 2在HippoRAG所使用的个性化PageRank算法基础上进行了改进，并增强了更深入的段落整合和更有效的LLM在线使用。这种结合使这个RAG系统更接近人类长期记忆的效果，在关联性记忆任务上比最先进的嵌入模型实现了7%的性能提升，同时展示了优越的事实性和意义构建记忆能力。这项工作为大规模语言模型的非参数化持续学习铺平了道路。我们的代码和数据将在以下网址发布：<此链接>。 

---
# Rapid Word Learning Through Meta In-Context Learning 

**Title (ZH)**: 通过元上下文学习实现快速词汇学习 

**Authors**: Wentao Wang, Guangyuan Jiang, Tal Linzen, Brenden M. Lake  

**Link**: [PDF](https://arxiv.org/pdf/2502.14791)  

**Abstract**: Humans can quickly learn a new word from a few illustrative examples, and then systematically and flexibly use it in novel contexts. Yet the abilities of current language models for few-shot word learning, and methods for improving these abilities, are underexplored. In this study, we introduce a novel method, Meta-training for IN-context learNing Of Words (Minnow). This method trains language models to generate new examples of a word's usage given a few in-context examples, using a special placeholder token to represent the new word. This training is repeated on many new words to develop a general word-learning ability. We find that training models from scratch with Minnow on human-scale child-directed language enables strong few-shot word learning, comparable to a large language model (LLM) pre-trained on orders of magnitude more data. Furthermore, through discriminative and generative evaluations, we demonstrate that finetuning pre-trained LLMs with Minnow improves their ability to discriminate between new words, identify syntactic categories of new words, and generate reasonable new usages and definitions for new words, based on one or a few in-context examples. These findings highlight the data efficiency of Minnow and its potential to improve language model performance in word learning tasks. 

**Abstract (ZH)**: 人类能够从几个示例中快速学习新单词，并在新的语境中系统地、灵活地使用它。然而，当前语言模型在少样本词学习方面的能力及其提升方法还远未被充分探索。本研究介绍了一种新的方法——Meta-training for IN-context learnNing Of Words (Minnow)，旨在训练语言模型根据几个上下文示例生成新词的用法实例，使用一个特殊的占位符标记来表示新词。这种训练方法通过大量新词的反复进行，开发出一种通用的词汇学习能力。我们发现，从零开始使用Minnow训练针对人类规模的儿童语言，能够实现强大的少样本词学习，与大量数据前训练的大语言模型（LLM）相比具有可比性。此外，通过辨别性和生成性评估发现，使用Minnow微调预训练的LLM能够提高其区分新词、识别新词的句法类别以及根据一两个上下文示例生成合理的新用法和定义的能力。这些发现突显了Minnow的数据效率，并展示了其在词汇学习任务中提升语言模型性能的潜力。 

---
# ReVision: A Dataset and Baseline VLM for Privacy-Preserving Task-Oriented Visual Instruction Rewriting 

**Title (ZH)**: ReVision：一个用于隐私保护任务导向视觉指令重写的数据集和基准多模态语言模型 

**Authors**: Abhijit Mishra, Richard Noh, Hsiang Fu, Mingda Li, Minji Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.14780)  

**Abstract**: Efficient and privacy-preserving multimodal interaction is essential as AR, VR, and modern smartphones with powerful cameras become primary interfaces for human-computer communication. Existing powerful large vision-language models (VLMs) enabling multimodal interaction often rely on cloud-based processing, raising significant concerns about (1) visual privacy by transmitting sensitive vision data to servers, and (2) their limited real-time, on-device usability. This paper explores Visual Instruction Rewriting, a novel approach that transforms multimodal instructions into text-only commands, allowing seamless integration of lightweight on-device instruction rewriter VLMs (250M parameters) with existing conversational AI systems, enhancing vision data privacy. To achieve this, we present a dataset of over 39,000 examples across 14 domains and develop a compact VLM, pretrained on image captioning datasets and fine-tuned for instruction rewriting. Experimental results, evaluated through NLG metrics such as BLEU, METEOR, and ROUGE, along with semantic parsing analysis, demonstrate that even a quantized version of the model (<500MB storage footprint) can achieve effective instruction rewriting, thus enabling privacy-focused, multimodal AI applications. 

**Abstract (ZH)**: 高效的隐私保护多模态交互对于AR、VR以及配备强大摄像头的现代智能手机成为人机通信主要接口至关重要。现有的强大多模态视觉-语言模型（VLMs）虽然可以实现多模态交互，但往往依赖于基于云的处理，引发了以下两个重大问题：（1）视觉隐私，因为敏感的视觉数据被传输到服务器；（2）实时性和离线使用能力受限。本文探讨了视觉指令重写（Visual Instruction Rewriting）这一前沿方法，其将多模态指令转换为纯文本命令，使轻量级的离设备指令重写VLM（参数量仅为250M）能够无缝集成到现有的对话AI系统中，提升了视觉数据的隐私保护水平。为实现这一目标，我们构建了一个包含超过39,000个示例的数据集，涵盖14个领域，并开发了一个紧凑型VLM，该VLM预训练于图像描述数据集，并针对指令重写进行了微调。实验结果通过NLG指标（如BLEU、METEOR和ROUGE）以及语义解析分析表明，即使量化后的模型（存储占用小于500MB）也能有效地进行指令重写，从而为注重隐私的多模态AI应用提供支持。 

---
# Harnessing PDF Data for Improving Japanese Large Multimodal Models 

**Title (ZH)**: 利用PDF数据提升日语大规模跨模态模型performance 

**Authors**: Jeonghun Baek, Akiko Aizawa, Kiyoharu Aizawa  

**Link**: [PDF](https://arxiv.org/pdf/2502.14778)  

**Abstract**: Large Multimodal Models (LMMs) have demonstrated strong performance in English, but their effectiveness in Japanese remains limited due to the lack of high-quality training data. Current Japanese LMMs often rely on translated English datasets, restricting their ability to capture Japan-specific cultural knowledge. To address this, we explore the potential of Japanese PDF data as a training resource, an area that remains largely underutilized. We introduce a fully automated pipeline that leverages pretrained models to extract image-text pairs from PDFs through layout analysis, OCR, and vision-language pairing, removing the need for manual annotation. Additionally, we construct instruction data from extracted image-text pairs to enrich the training data. To evaluate the effectiveness of PDF-derived data, we train Japanese LMMs and assess their performance on the Japanese LMM Benchmark. Our results demonstrate substantial improvements, with performance gains ranging from 3.9% to 13.8% on Heron-Bench. Further analysis highlights the impact of PDF-derived data on various factors, such as model size and language models, reinforcing its value as a multimodal resource for Japanese LMMs. We plan to make the source code and data publicly available upon acceptance. 

**Abstract (ZH)**: 大规模多模态模型（LMMs）在英语上已经展示了出色的性能，但在日语上仍表现出局限性，主要是由于缺乏高质量的训练数据。当前的日语LMMs往往依赖于翻译自英语的数据集，这限制了它们捕捉日本特定文化知识的能力。为解决这一问题，我们探讨了利用日语PDF数据作为训练资源的潜力，这一领域目前仍被严重忽视。我们介绍了一种完全自动化的管道，该管道利用预训练模型通过布局分析、OCR和视觉语言配对从PDF中提取图像-文本对，从而无需手动标注。此外，我们从提取的图像-文本对中构建指令数据以丰富训练数据。为了评估PDF衍生数据的有效性，我们对日语LMMs进行了训练，并在日语LMM基准测试上评估了其性能。结果表明，LMMs在Heron-Bench上的性能有所提高，提高幅度从3.9%到13.8%不等。进一步的分析强调了PDF衍生数据对各种因素（如模型规模和语言模型）的影响，这进一步证明了其作为日语LMMs多模态资源的价值。我们计划在论文被接受后公开源代码和数据。 

---
# SurveyX: Academic Survey Automation via Large Language Models 

**Title (ZH)**: SurveyX：通过大语言模型实现的学术调查自动化 

**Authors**: Xun Liang, Jiawei Yang, Yezhaohui Wang, Chen Tang, Zifan Zheng, Simin Niu, Shichao Song, Hanyu Wang, Bo Tang, Feiyu Xiong, Keming Mao, Zhiyu li  

**Link**: [PDF](https://arxiv.org/pdf/2502.14776)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional comprehension capabilities and a vast knowledge base, suggesting that LLMs can serve as efficient tools for automated survey generation. However, recent research related to automated survey generation remains constrained by some critical limitations like finite context window, lack of in-depth content discussion, and absence of systematic evaluation frameworks. Inspired by human writing processes, we propose SurveyX, an efficient and organized system for automated survey generation that decomposes the survey composing process into two phases: the Preparation and Generation phases. By innovatively introducing online reference retrieval, a pre-processing method called AttributeTree, and a re-polishing process, SurveyX significantly enhances the efficacy of survey composition. Experimental evaluation results show that SurveyX outperforms existing automated survey generation systems in content quality (0.259 improvement) and citation quality (1.76 enhancement), approaching human expert performance across multiple evaluation dimensions. Examples of surveys generated by SurveyX are available on this http URL 

**Abstract (ZH)**: 大型语言模型（LLMs）展现了出色的理解能力和广泛的知识基础，表明LLMs可以作为高效自动化问卷生成工具。然而，近期关于自动化问卷生成的研究仍受到一些关键限制的制约，如有限的上下文窗口、缺乏深入内容讨论以及缺乏系统的评估框架。受人类写作过程的启发，我们提出了SurveyX，这是一种高效且有组织的自动化问卷生成系统，将问卷构建过程分解为两个阶段：准备阶段和生成阶段。通过创新地引入在线参考检索、一种预处理方法——属性树（AttributeTree）以及后续润色过程，SurveyX显著提高了问卷构建的效率。实验评估结果表明，与现有的自动化问卷生成系统相比，SurveyX在内容质量（提高0.259）和引文质量（增强1.76）方面表现更优，多个评估维度接近人类专家的水平。SurveyX生成的问卷示例可在以下网址查看：[请填写具体网址] 

---
# Logic-RL: Unleashing LLM Reasoning with Rule-Based Reinforcement Learning 

**Title (ZH)**: Logic-RL：基于规则的强化学习释放大语言模型的推理能力 

**Authors**: Tian Xie, Zitian Gao, Qingnan Ren, Haoming Luo, Yuqian Hong, Bryan Dai, Joey Zhou, Kai Qiu, Zhirong Wu, Chong Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.14768)  

**Abstract**: Inspired by the success of DeepSeek-R1, we explore the potential of rule-based reinforcement learning (RL) in large reasoning models. To analyze reasoning dynamics, we use synthetic logic puzzles as training data due to their controllable complexity and straightforward answer verification. We make some key technical contributions that lead to effective and stable RL training: a system prompt that emphasizes the thinking and answering process, a stringent format reward function that penalizes outputs for taking shortcuts, and a straightforward training recipe that achieves stable convergence. Our 7B model develops advanced reasoning skills-such as reflection, verification, and summarization-that are absent from the logic corpus. Remarkably, after training on just 5K logic problems, it demonstrates generalization abilities to the challenging math benchmarks AIME and AMC. 

**Abstract (ZH)**: 受DeepSeek-R1成功的启发，我们探索了基于规则的增强学习（RL）在大型推理模型中的潜力。为了分析推理动态，我们使用合成逻辑谜题作为训练数据，因为这些数据的复杂性可控且答案验证简单明了。我们做出了一些关键性的技术贡献，这些贡献导致了有效的稳定RL训练：一个强调思考和回答过程的系统提示，一个严格的格式奖励函数，惩罚跳过的输出，以及一个简单的训练食谱，实现了稳定的收敛。我们的7B模型发展出了一些高级的推理技能，如反思、验证和总结，这些技能在逻辑语料库中是不存在的。令人惊讶的是，在仅仅训练了5,000个逻辑问题之后，它在具有挑战性的数学基准测试AIME和AMC中展示了泛化的性能。 

---
# Tree-of-Debate: Multi-Persona Debate Trees Elicit Critical Thinking for Scientific Comparative Analysis 

**Title (ZH)**: 树状辩论：多角色辩论树促进科学比较分析中的批判性思维 

**Authors**: Priyanka Kargupta, Ishika Agarwal, Tal August, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2502.14767)  

**Abstract**: With the exponential growth of research facilitated by modern technology and improved accessibility, scientific discoveries have become increasingly fragmented within and across fields. This makes it challenging to assess the significance, novelty, incremental findings, and equivalent ideas between related works, particularly those from different research communities. Large language models (LLMs) have recently demonstrated strong quantitative and qualitative reasoning abilities, and multi-agent LLM debates have shown promise in handling complex reasoning tasks by exploring diverse perspectives and reasoning paths. Inspired by this, we introduce Tree-of-Debate (ToD), a framework which converts scientific papers into LLM personas that debate their respective novelties. To emphasize structured, critical reasoning rather than focusing solely on outcomes, ToD dynamically constructs a debate tree, enabling fine-grained analysis of independent novelty arguments within scholarly articles. Through experiments on scientific literature across various domains, evaluated by expert researchers, we demonstrate that ToD generates informative arguments, effectively contrasts papers, and supports researchers in their literature review. 

**Abstract (ZH)**: 随着现代技术的发展和科研可访问性的提升，科学研究成果在领域内及跨领域之间呈现出指数级增长并日益碎片化。这使得评估相关研究成果的意义、新颖性、增量发现以及等效思想变得愈发困难，特别是当这些成果源于不同的研究社群时。大语言模型（LLMs）最近展示了强大的定量和定性推理能力，多智能体LLM辩论在处理复杂推理任务方面也展现出了潜力，通过探索不同的视角和推理路径来应对这些挑战。基于此，我们提出了一种名为“辩论树”（Tree-of-Debate, ToD）的框架，该框架将科学论文转换为LLM人物，进行各自的新颖性辩论。ToD 强调结构化的批判性推理，而不是仅关注结果，通过动态构建辩论树，能够对学术文章中的独立新颖性论点进行精细分析。通过对涵盖不同领域的科学文献进行实验，并由专家研究人员进行评估，我们展示了ToD生成了富有信息量的论点，有效地对比了论文，并支持研究人员进行文献综述。 

---
# Step-by-Step Fact Verification System for Medical Claims with Explainable Reasoning 

**Title (ZH)**: 基于可解释推理的逐步医疗声明验证系统 

**Authors**: Juraj Vladika, Ivana Hacajová, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2502.14765)  

**Abstract**: Fact verification (FV) aims to assess the veracity of a claim based on relevant evidence. The traditional approach for automated FV includes a three-part pipeline relying on short evidence snippets and encoder-only inference models. More recent approaches leverage the multi-turn nature of LLMs to address FV as a step-by-step problem where questions inquiring additional context are generated and answered until there is enough information to make a decision. This iterative method makes the verification process rational and explainable. While these methods have been tested for encyclopedic claims, exploration on domain-specific and realistic claims is missing. In this work, we apply an iterative FV system on three medical fact-checking datasets and evaluate it with multiple settings, including different LLMs, external web search, and structured reasoning using logic predicates. We demonstrate improvements in the final performance over traditional approaches and the high potential of step-by-step FV systems for domain-specific claims. 

**Abstract (ZH)**: 事实核查（Fact Verification, FV）旨在基于相关证据评估一个断言的真实性。传统的自动化事实核查方法依赖于简短的证据片段和仅编码模型的三阶段流水线。近期的方法则利用大语言模型（LLM）的多轮特性，将事实核查视为一个逐步解决问题的过程，在此过程中生成并回答涉及额外背景信息的问题，直到有足够的信息来做决定。这种迭代方法使得核查过程更加合理且具可解释性。虽然这些方法已在百科式断言上进行了测试，但在特定领域和现实世界断言上的探索仍然不足。在本研究中，我们在三个医学事实核查数据集中应用了一个迭代的事实核查系统，并通过不同的大语言模型、外部网络搜索和使用逻辑谓词进行结构化推理等多种设置对其进行评估。我们展示了此迭代系统在最终性能上的改进，并展示了逐步事实核查系统对特定领域断言的巨大潜力。 

---
# On the Influence of Context Size and Model Choice in Retrieval-Augmented Generation Systems 

**Title (ZH)**: 关于检索增强生成系统中上下文大小和模型选择的影响研究 

**Authors**: Juraj Vladika, Florian Matthes  

**Link**: [PDF](https://arxiv.org/pdf/2502.14759)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as an approach to augment large language models (LLMs) by reducing their reliance on static knowledge and improving answer factuality. RAG retrieves relevant context snippets and generates an answer based on them. Despite its increasing industrial adoption, systematic exploration of RAG components is lacking, particularly regarding the ideal size of provided context, and the choice of base LLM and retrieval method. To help guide development of robust RAG systems, we evaluate various context sizes, BM25 and semantic search as retrievers, and eight base LLMs. Moving away from the usual RAG evaluation with short answers, we explore the more challenging long-form question answering in two domains, where a good answer has to utilize the entire context. Our findings indicate that final QA performance improves steadily with up to 15 snippets but stagnates or declines beyond that. Finally, we show that different general-purpose LLMs excel in the biomedical domain than the encyclopedic one, and that open-domain evidence retrieval in large corpora is challenging. 

**Abstract (ZH)**: 检索增强生成（RAG）作为一种通过减少对静态知识的依赖并提高答案的真实性来增强大规模语言模型（LLMs）的方法而逐渐崭露头角。RAG通过检索相关的上下文片段，并基于这些片段生成答案。尽管RAG在工业界的应用越来越广泛，但对其组件的系统性探索仍然不足，尤其是在提供的上下文大小的理想尺寸以及基础LLM和检索方法的选择方面。为帮助指导稳健的RAG系统的开发，我们评估了不同大小的上下文、BM25和语义搜索作为检索方法，以及八种基础LLM。我们从常见的使用短答案的RAG评估转向探索更具挑战性的长形式问答，分别在两个领域进行了研究，在这两个领域中，好的答案需要利用整个上下文。我们的研究结果表明，直到15个片段的总数量，最终的问答性能都会逐步提高，但超过这一数量后则会停滞不前或下降。最后，我们展示了不同通用语言模型在生物医学领域与百科领域中的表现存在差异，并且在大规模语料库中进行开放领域证据检索具有挑战性。 

---
# TritonBench: Benchmarking Large Language Model Capabilities for Generating Triton Operators 

**Title (ZH)**: TritonBench: 评估生成 Triton 运算能力的大规模语言模型性能 

**Authors**: Jianling Li, Shangzhan Li, Zhenye Gao, Qi Shi, Yuxuan Li, Zefan Wang, Jiacheng Huang, Haojie Wang, Jianrong Wang, Xu Han, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.14752)  

**Abstract**: Triton, a high-level Python-like language designed for building efficient GPU kernels, is widely adopted in deep learning frameworks due to its portability, flexibility, and accessibility. However, programming and parallel optimization still require considerable trial and error from Triton developers. Despite advances in large language models (LLMs) for conventional code generation, these models struggle to generate accurate, performance-optimized Triton code, as they lack awareness of its specifications and the complexities of GPU programming. More critically, there is an urgent need for systematic evaluations tailored to Triton. In this work, we introduce TritonBench, the first comprehensive benchmark for Triton operator generation. TritonBench features two evaluation channels: a curated set of 184 real-world operators from GitHub and a collection of operators aligned with PyTorch interfaces. Unlike conventional code benchmarks prioritizing functional correctness, TritonBench also profiles efficiency performance on widely deployed GPUs aligned with industry applications. Our study reveals that current state-of-the-art code LLMs struggle to generate efficient Triton operators, highlighting a significant gap in high-performance code generation. TritonBench will be available at this https URL. 

**Abstract (ZH)**: Triton 是一种高级 Python 风格的编程语言，专门设计用于构建高效的 GPU 内核，因此在深度学习框架中广泛采用，因为它具有良好的移植性、灵活性和易用性。然而，编程和并行优化仍然需要 Triton 开发者进行大量的尝试和错误。尽管大型语言模型（LLMs）在常规代码生成方面取得了进展，但这些模型仍然难以生成准确且性能优化的 Triton 代码，因为它们缺乏对 Triton 规范和 GPU 编程复杂性的了解。更为关键的是，对 Triton 的系统性评估非常迫切。在本工作中，我们推出了 TritonBench，这是首个全面评估 Triton 运算符生成的基准测试。TritonBench 包含两个评估通道：从 GitHub 中精心挑选的 184 个实际运算符以及与 PyTorch 接口对齐的运算符集合。不同于优先考虑功能正确性的常规代码基准，TritonBench 还针对广泛部署的 GPU 和工业应用进行了效率性能的评估。我们的研究发现，当前最先进的代码 LLM 难以生成高效的 Triton 运算符，揭示了高性能代码生成方面的显著差距。TritonBench 可通过以下链接访问：此httpsURL。 

---
# Large Language Models Struggle to Describe the Haystack without Human Help: Human-in-the-loop Evaluation of LLMs 

**Title (ZH)**: 大型语言模型在没有人类帮助的情况下难以描述文档的核心内容：关于大型语言模型的人类在环评估 

**Authors**: Zongxia Li, Lorena Calvo-Bartolomé, Alexander Hoyle, Paiheng Xu, Alden Dima, Juan Francisco Fung, Jordan Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2502.14748)  

**Abstract**: A common use of NLP is to facilitate the understanding of large document collections, with a shift from using traditional topic models to Large Language Models. Yet the effectiveness of using LLM for large corpus understanding in real-world applications remains under-explored. This study measures the knowledge users acquire with unsupervised, supervised LLM-based exploratory approaches or traditional topic models on two datasets. While LLM-based methods generate more human-readable topics and show higher average win probabilities than traditional models for data exploration, they produce overly generic topics for domain-specific datasets that do not easily allow users to learn much about the documents. Adding human supervision to the LLM generation process improves data exploration by mitigating hallucination and over-genericity but requires greater human effort. In contrast, traditional. models like Latent Dirichlet Allocation (LDA) remain effective for exploration but are less user-friendly. We show that LLMs struggle to describe the haystack of large corpora without human help, particularly domain-specific data, and face scaling and hallucination limitations due to context length constraints. Dataset available at https://huggingface. co/datasets/zli12321/Bills. 

**Abstract (ZH)**: 自然语言处理（NLP）的一个常见用途是帮助理解大规模文档集合，从使用传统的话题模型转向使用大型语言模型（LLM）。然而，LLM 在实际应用中对大规模语料库的理解效果仍缺乏充分探索。本研究通过在两个数据集上使用无监督和监督的LLM探索方法或传统话题模型，测量用户获取的知识。虽然基于LLM的方法生成了更易于理解的主题，并在数据探索中显示出更高的平均获胜概率，但它们对特定领域的数据集生成的主题过于通用，使用户难以学到很多关于文档的知识。通过在LLM生成过程中添加人工监督，可以减轻幻觉和通用性过强的问题，但需要更多的手工努力。相比之下，传统的模型如潜在狄利克雷分配（LDA）仍然适用于探索，但用户友好性较差。我们表明，没有人工帮助，LLM 在描述大规模语料库（尤其是特定领域的数据）时显得力不从心，并且由于上下文长度限制，面临扩展性和幻觉的挑战。数据集可从 <https://huggingface.co/datasets/zli12321/Bills> 获取。 

---
# HiddenDetect: Detecting Jailbreak Attacks against Large Vision-Language Models via Monitoring Hidden States 

**Title (ZH)**: HiddenDetect：通过监控隐藏状态检测针对大规模视觉-语言模型的越权攻击 

**Authors**: Yilei Jiang, Xinyan Gao, Tianshuo Peng, Yingshui Tan, Xiaoyong Zhu, Bo Zheng, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2502.14744)  

**Abstract**: The integration of additional modalities increases the susceptibility of large vision-language models (LVLMs) to safety risks, such as jailbreak attacks, compared to their language-only counterparts. While existing research primarily focuses on post-hoc alignment techniques, the underlying safety mechanisms within LVLMs remain largely unexplored. In this work , we investigate whether LVLMs inherently encode safety-relevant signals within their internal activations during inference. Our findings reveal that LVLMs exhibit distinct activation patterns when processing unsafe prompts, which can be leveraged to detect and mitigate adversarial inputs without requiring extensive fine-tuning. Building on this insight, we introduce HiddenDetect, a novel tuning-free framework that harnesses internal model activations to enhance safety. Experimental results show that {HiddenDetect} surpasses state-of-the-art methods in detecting jailbreak attacks against LVLMs. By utilizing intrinsic safety-aware patterns, our method provides an efficient and scalable solution for strengthening LVLM robustness against multimodal threats. Our code will be released publicly at this https URL. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，要符合学术规范：

将额外模态整合到大型视觉语言模型（LVLMs）中会增加其对安全风险的敏感性，如脱困攻击（jailbreak attacks），相较于仅语言模型的版本。目前的研究主要集中在事后对齐技术上，但LVLMs内的内在安全机制尚未得到充分探索。在本研究中，我们探讨了LVLMs在推理过程中是否固有地在内部激活中编码了与安全性相关的信息。我们发现，当处理不安全提示时，LVLMs表现出不同的激活模式，这些模式可以被利用来检测和缓解恶意输入，而无需进行大量的微调。基于这一发现，我们提出了一种名为HiddenDetect的新颖无微调框架，该框架利用模型内部激活来增强安全性。实验结果显示，HiddenDetect在检测LVLM对抗脱困攻击方面超过了当前最先进的方法。通过利用内在的安全感知模式，我们的方法提供了一种高效且可扩展的解决方案，以增强LVLM在多模态威胁下的鲁棒性。我们的代码将在此网址公开发布：https://... 

---
# SuperGPQA: Scaling LLM Evaluation across 285 Graduate Disciplines 

**Title (ZH)**: SuperGPQA：跨285个研究生学科领域扩展大语言模型评估 

**Authors**: M-A-P Team, Xinrun Du, Yifan Yao, Kaijing Ma, Bingli Wang, Tianyu Zheng, Kang Zhu, Minghao Liu, Yiming Liang, Xiaolong Jin, Zhenlin Wei, Chujie Zheng, Kaixing Deng, Shuyue Guo, Shian Jia, Sichao Jiang, Yiyan Liao, Rui Li, Qinrui Li, Sirun Li, Yizhi Li, Yunwen Li, Dehua Ma, Yuansheng Ni, Haoran Que, Qiyao Wang, Zhoufutu Wen, Siwei Wu, Tianshun Xing, Ming Xu, Zhenzhu Yang, Zekun Moore Wang, Junting Zhou, Yuelin Bai, Xingyuan Bu, Chenglin Cai, Liang Chen, Yifan Chen, Chengtuo Cheng, Tianhao Cheng, Keyi Ding, Siming Huang, Yun Huang, Yaoru Li, Yizhe Li, Zhaoqun Li, Tianhao Liang, Chengdong Lin, Hongquan Lin, Yinghao Ma, Zhongyuan Peng, Zifan Peng, Qige Qi, Shi Qiu, Xingwei Qu, Yizhou Tan, Zili Wang, Chenqing Wang, Hao Wang, Yiya Wang, Yubo Wang, Jiajun Xu, Kexin Yang, Ruibin Yuan, Yuanhao Yue, Tianyang Zhan, Chun Zhang, Jingyang Zhang, Xiyue Zhang, Xingjian Zhang, Yue Zhang, Yongchi Zhao, Xiangyu Zheng, Chenghua Zhong, Yang Gao, Zhoujun Li, Dayiheng Liu, Qian Liu, Tianyu Liu, Shiwen Ni, Junran Peng, Yujia Qin, Wenbo Su, Guoyin Wang, Shi Wang, Jian Yang, Min Yang, Meng Cao, Xiang Yue, Zhaoxiang Zhang, Wangchunshu Zhou, Jiaheng Liu, Qunshu Lin, Wenhao Huang, Ge Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14739)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable proficiency in mainstream academic disciplines such as mathematics, physics, and computer science. However, human knowledge encompasses over 200 specialized disciplines, far exceeding the scope of existing benchmarks. The capabilities of LLMs in many of these specialized fields-particularly in light industry, agriculture, and service-oriented disciplines-remain inadequately evaluated. To address this gap, we present SuperGPQA, a comprehensive benchmark that evaluates graduate-level knowledge and reasoning capabilities across 285 disciplines. Our benchmark employs a novel Human-LLM collaborative filtering mechanism to eliminate trivial or ambiguous questions through iterative refinement based on both LLM responses and expert feedback. Our experimental results reveal significant room for improvement in the performance of current state-of-the-art LLMs across diverse knowledge domains (e.g., the reasoning-focused model DeepSeek-R1 achieved the highest accuracy of 61.82% on SuperGPQA), highlighting the considerable gap between current model capabilities and artificial general intelligence. Additionally, we present comprehensive insights from our management of a large-scale annotation process, involving over 80 expert annotators and an interactive Human-LLM collaborative system, offering valuable methodological guidance for future research initiatives of comparable scope. 

**Abstract (ZH)**: 大型语言模型（LLMs）在数学、物理学和计算机科学等主流学术领域中展现出了卓越的能力。然而，人类知识涵盖了超过200个专门学科，远超现有基准的涵盖范围。在许多专门领域中，特别是轻工业、农业以及以服务为导向的学科中，LLMs的能力仍缺乏充分的评估。为了解决这一缺口，我们提出了SuperGPQA这一全面基准，用于评估涵盖285个学科的研究生级知识和推理能力。我们的基准利用了新的人类-LLM协作过滤机制，在基于LLM响应和专家反馈的迭代完善过程中，排除了简单或模棱两可的问题。我们的实验结果揭示了当前最先进LLMs在不同知识领域中的显著改进空间（例如，专注于推理的模型DeepSeek-R1在其SuperGPQA上的最高准确率为61.82%），突显了当前模型能力与人工通用智能之间的巨大差距。此外，我们还提供了大规模注释流程管理的全面见解，涉及超过80位专家注释员和一个交互式的LLM辅助系统，为未来同类规模的研究项目提供了宝贵的指导方法。 

---
# Sentence Smith: Formally Controllable Text Transformation and its Application to Evaluation of Text Embedding Models 

**Title (ZH)**: 句子 smith：形式可控的文本转换及其在文本嵌入模型评估中的应用 

**Authors**: Hongji Li, Andrianos Michail, Reto Gubelmann, Simon Clematide, Juri Opitz  

**Link**: [PDF](https://arxiv.org/pdf/2502.14734)  

**Abstract**: We propose the Sentence Smith framework that enables controlled and specified manipulation of text meaning. It consists of three main steps: 1. Parsing a sentence into a semantic graph, 2. Applying human-designed semantic manipulation rules, and 3. Generating text from the manipulated graph. A final filtering step (4.) ensures the validity of the applied transformation. To demonstrate the utility of Sentence Smith in an application study, we use it to generate hard negative pairs that challenge text embedding models. Since the controllable generation makes it possible to clearly isolate different types of semantic shifts, we can gain deeper insights into the specific strengths and weaknesses of widely used text embedding models, also addressing an issue in current benchmarking where linguistic phenomena remain opaque. Human validation confirms that the generations produced by Sentence Smith are highly accurate. 

**Abstract (ZH)**: 我们提出了一种Sentence Smith框架，该框架能够实现对文本含义的可控和指定修改。该框架主要包含三个步骤：1. 将句子解析为语义图；2. 应用人类设计的语义修改规则；3. 从修改后的图生成文本。最后一步（第4步）确保了所应用变换的有效性。为了展示Sentence Smith在实际应用中的实用性，我们使用它生成对抗对（hard negative pairs），以挑战文本嵌入模型。由于可控生成使得能够清晰地隔离不同类型的语义转移，我们能够更深入地了解广泛使用中的文本嵌入模型的具体优势和弱点，也可解决当前基准测试中的一个问题，即语言现象仍不透明。人工验证确认，Sentence Smith生成的内容具有高度的准确性。 

---
# Entity Framing and Role Portrayal in the News 

**Title (ZH)**: 新闻中的实体框架与角色呈现研究 

**Authors**: Tarek Mahmoud, Zhuohan Xie, Dimitar Dimitrov, Nikolaos Nikolaidis, Purificação Silvano, Roman Yangarber, Shivam Sharma, Elisa Sartori, Nicolas Stefanovitch, Giovanni Da San Martino, Jakub Piskorski, Preslav Nakov  

**Link**: [PDF](https://arxiv.org/pdf/2502.14718)  

**Abstract**: We introduce a novel multilingual hierarchical corpus annotated for entity framing and role portrayal in news articles. The dataset uses a unique taxonomy inspired by storytelling elements, comprising 22 fine-grained roles, or archetypes, nested within three main categories: protagonist, antagonist, and innocent. Each archetype is carefully defined, capturing nuanced portrayals of entities such as guardian, martyr, and underdog for protagonists; tyrant, deceiver, and bigot for antagonists; and victim, scapegoat, and exploited for innocents. The dataset includes 1,378 recent news articles in five languages (Bulgarian, English, Hindi, European Portuguese, and Russian) focusing on two critical domains of global significance: the Ukraine-Russia War and Climate Change. Over 5,800 entity mentions have been annotated with role labels. This dataset serves as a valuable resource for research into role portrayal and has broader implications for news analysis. We describe the characteristics of the dataset and the annotation process, and we report evaluation results on fine-tuned state-of-the-art multilingual transformers and hierarchical zero-shot learning using LLMs at the level of a document, a paragraph, and a sentence. 

**Abstract (ZH)**: 我们介绍了一个新颖的多语言层次结构语料库，该语料库针对新闻文章中的实体框架和角色呈现进行了标注。该数据集采用了一种独特的分类体系，该体系受到叙事元素的启发，包括22个精细分类的角色（或原型），这些角色分为三大类别： protagonist（主人公）、antagonist（反派）和innocent（无辜者）。每个原型都进行了仔细定义，捕捉到了如guardian（守护者）、martyr（先驱者）、underdog（弱者）等主人公的细微表现；tyrant（暴君）、deceiver（骗子）、bigot（偏执者）等反派的细微表现；以及victim（受害者）、scapegoat（替罪羊）、exploited（被剥削者）等无辜者的细微表现。该数据集包含了五种语言（保加利亚语、英语、印地语、欧洲葡萄牙语和俄语）的1,378篇最近的新闻文章，重点关注全球两个关键领域：乌克兰-俄罗斯战争和气候变化。超过5,800个实体提及已被标记为角色标签。该数据集为角色呈现的研究提供了一个宝贵的资源，并对新闻分析具有更广泛的含义。我们描述了数据集的特点和标注过程，并报告了在微调的最先进的多语言变换器和层次零样本学习方面对文档、段落和句子级别的评估结果。 

---
# Data-Efficient Pretraining with Group-Level Data Influence Modeling 

**Title (ZH)**: 基于组级数据影响建模的数据高效预训练 

**Authors**: Zichun Yu, Fei Peng, Jie Lei, Arnold Overwijk, Wen-tau Yih, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.14709)  

**Abstract**: Data-efficient pretraining has shown tremendous potential to elevate scaling laws. This paper argues that effective pretraining data should be curated at the group level, treating a set of data points as a whole rather than as independent contributors. To achieve that, we propose Group-Level Data Influence Modeling (Group-MATES), a novel data-efficient pretraining method that captures and optimizes group-level data utility. Specifically, Group-MATES collects oracle group-level influences by locally probing the pretraining model with data sets. It then fine-tunes a relational data influence model to approximate oracles as relationship-weighted aggregations of individual influences. The fine-tuned model selects the data subset by maximizing its group-level influence prediction, with influence-aware clustering to enable efficient inference. Experiments on the DCLM benchmark demonstrate that Group-MATES achieves a 10% relative core score improvement on 22 downstream tasks over DCLM-Baseline and 5% over individual-influence-based methods, establishing a new state-of-the-art. Further analyses highlight the effectiveness of relational data influence models in capturing intricate interactions between data points. 

**Abstract (ZH)**: 数据效率的预训练显示出了极大的潜力，能够提升模型的扩展规律。本文认为，有效的预训练数据应从组层面进行筛选与管理，将数据点作为一个整体而非独立个体进行处理。为此，我们提出了组级数据影响模型（Group-MATES）这一新颖的数据效率预训练方法，旨在捕捉和优化组层面的数据效用。具体而言，Group-MATES 通过局部探查预训练模型的数据集，收集oracle组级影响，并通过 fine-tune 一种关系数据影响模型，将其与个体影响进行加权聚合来近似或acles。fine-tuned 模型根据组级影响预测选择数据子集，并使用影响感知的聚类以实现高效推理。在 DCLM 基准上的实验表明，Group-MATES 在 22 个下游任务上的相对核心分数提高了 10%，优于 DCLM 基准和基于个体影响的方法，创下了新的最佳水平。进一步的分析强调了关系数据影响模型在捕捉数据点之间复杂相互作用方面的有效性。 

---
# I-MCTS: Enhancing Agentic AutoML via Introspective Monte Carlo Tree Search 

**Title (ZH)**: I-MCTS：基于反省蒙特卡罗树搜索的自主智能自动化机器学习增强方法 

**Authors**: Zujie Liang, Feng Wei, Wujiang Xu, Lin Chen, Yuxi Qian, Xinhui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14693)  

**Abstract**: Recent advancements in large language models (LLMs) have shown remarkable potential in automating machine learning tasks. However, existing LLM-based agents often struggle with low-diversity and suboptimal code generation. While recent work has introduced Monte Carlo Tree Search (MCTS) to address these issues, limitations persist in the quality and diversity of thoughts generated, as well as in the scalar value feedback mechanisms used for node selection. In this study, we introduce Introspective Monte Carlo Tree Search (I-MCTS), a novel approach that iteratively expands tree nodes through an introspective process that meticulously analyzes solutions and results from parent and sibling nodes. This facilitates a continuous refinement of the node in the search tree, thereby enhancing the overall decision-making this http URL, we integrate a Large Language Model (LLM)-based value model to facilitate direct evaluation of each node's solution prior to conducting comprehensive computational rollouts. A hybrid rewarding mechanism is implemented to seamlessly transition the Q-value from LLM-estimated scores to actual performance scores. This allows higher-quality nodes to be traversed this http URL to the various ML tasks, our approach demonstrates a6\% absolute improvement in performance compared to the strong open-source AutoML agents, showcasing its effectiveness in enhancing agentic AutoML systems. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）在自动化机器学习任务方面展现出了显著的潜力。然而，现有的基于LLM的代理程序常常面临低多样性和生成次优代码的问题。尽管最近的研究引入了蒙特卡洛树搜索（MCTS）来解决这些问题，但在生成思维的质量和多样性以及节点选择的标量反馈机制方面仍然存在局限。在本研究中，我们提出了反省蒙特卡洛树搜索（I-MCTS），这是一种新颖的方法，通过一个反省的过程，详细分析父节点和同族节点的解决方案和结果，逐步扩展树节点。这有助于在搜索树中不断优化节点，从而提高整体决策质量。为了实现这一目标，我们整合了一个基于大型语言模型（LLM）的价值模型，可以在进行全面计算展开之前直接评估每个节点的解决方案。我们实现了一个混合奖励机制，使Q值能够平滑过渡从LLM估计的分数到实际性能分数。这使得更高质量的节点能够被更频繁地访问。对于各种机器学习任务，我们的方法在性能上相对于现有的强大开源自动机器学习代理取得了绝对6%的提升，这表明了其在增强代理型自动机器学习系统方面的有效性。 

---
# Bridging the Gap: Transforming Natural Language Questions into SQL Queries via Abstract Query Pattern and Contextual Schema Markup 

**Title (ZH)**: 填补空白：通过抽象查询模式和上下文模式标记将自然语言问题转化为SQL查询 

**Authors**: Yonghui Kong, Hongbing Hu, Dan Zhang, Siyuan Chai, Fan Zhang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14682)  

**Abstract**: Large language models have demonstrated excellent performance in many tasks, including Text-to-SQL, due to their powerful in-context learning capabilities. They are becoming the mainstream approach for Text-to-SQL. However, these methods still have a significant gap compared to human performance, especially on complex questions. As the complexity of questions increases, the gap between questions and SQLs increases. We identify two important gaps: the structural mapping gap and the lexical mapping gap. To tackle these two gaps, we propose PAS-SQL, an efficient SQL generation pipeline based on LLMs, which alleviates gaps through Abstract Query Pattern (AQP) and Contextual Schema Markup (CSM). AQP aims to obtain the structural pattern of the question by removing database-related information, which enables us to find structurally similar demonstrations. CSM aims to associate database-related text span in the question with specific tables or columns in the database, which alleviates the lexical mapping gap. Experimental results on the Spider and BIRD datasets demonstrate the effectiveness of our proposed method. Specifically, PAS-SQL + GPT-4o sets a new state-of-the-art on the Spider benchmark with an execution accuracy of 87.9\%, and achieves leading results on the BIRD dataset with an execution accuracy of 64.67\%. 

**Abstract (ZH)**: 大型语言模型在许多任务中展示了出色的表现，包括Text-to-SQL，这得益于它们强大的上下文学习能力。它们已成为Text-to-SQL的主流方法。然而，这些方法与人类的表现之间仍然存在显著差距，特别是在复杂问题上。随着问题复杂性的增加，问题与SQL之间的差距也增大。我们识别出两个重要的差距：结构映射差距和词汇映射差距。为了解决这两个差距，我们提出了PAS-SQL，这是一种基于LLM的有效SQL生成流水线，通过抽象查询模式（AQP）和上下文模式标记（CSM）来缓解这些差距。AQP旨在通过去除数据库相关的信息来获取问题的结构模式，使我们能够找到结构上相似的示例。CSM旨在将问题中的数据库相关文本片段与数据库中的特定表或列相关联，从而缓解了词汇映射差距。在Spider和BIRD数据集上的实验结果表明了我们提出方法的有效性。具体来说，PAS-SQL + GPT-4o在Spider基准测试上设定了新的最佳性能，执行准确率为87.9%，在BIRD数据集上也取得了领先结果，执行准确率为64.67%。 

---
# How to Get Your LLM to Generate Challenging Problems for Evaluation 

**Title (ZH)**: 如何让大语言模型生成具有挑战性的评估问题 

**Authors**: Arkil Patel, Siva Reddy, Dzmitry Bahdanau  

**Link**: [PDF](https://arxiv.org/pdf/2502.14678)  

**Abstract**: The pace of evolution of Large Language Models (LLMs) necessitates new approaches for rigorous and comprehensive evaluation. Traditional human annotation is increasingly impracticable due to the complexities and costs involved in generating high-quality, challenging problems. In this work, we introduce CHASE, a unified framework to synthetically generate challenging problems using LLMs without human involvement. For a given task, our approach builds a hard problem in a bottom-up manner from simpler components. Moreover, our framework decomposes the generation process into independently verifiable sub-tasks, thereby ensuring a high level of quality and correctness. We implement CHASE to create evaluation benchmarks across three diverse domains: (1) document-based question answering, (2) repository-level code completion, and (3) math reasoning. The performance of state-of-the-art LLMs on these synthetic benchmarks lies in the range of 40-60% accuracy, thereby demonstrating the effectiveness of our framework at generating challenging problems. We publicly release our benchmarks and code. 

**Abstract (ZH)**: 大型语言模型（LLMs）的发展速度 necessitates 新的方法来进行严谨和全面的评估。传统的手工注释越来越不切实际，因为生成高质量、具有挑战性的问题涉及复杂性和高昂的成本。在本文中，我们引入了CHASE，这是一种无需人工干预即可使用LLMs合成生成具有挑战性问题的统一框架。对于给定的任务，我们的方法从简单的组件自底向上构建一个困难的问题。此外，我们的框架将生成过程分解为可独立验证的子任务，从而确保较高的质量和正确性。我们实现了CHASE，在三个不同的领域创建了评估基准：（1）基于文档的问答，（2）代码仓库级别的代码完成，以及（3）数学推理。最先进的LLMs在这三个合成基准上的性能准确率范围为40-60%，这表明我们的框架在生成具有挑战性的问题方面具有有效性。我们还将我们的基准和代码公开发布。 

---
# Data-Constrained Synthesis of Training Data for De-Identification 

**Title (ZH)**: 基于数据约束的脱敏训练数据合成 

**Authors**: Thomas Vakili, Aron Henriksson, Hercules Dalianis  

**Link**: [PDF](https://arxiv.org/pdf/2502.14677)  

**Abstract**: Many sensitive domains -- such as the clinical domain -- lack widely available datasets due to privacy risks. The increasing generative capabilities of large language models (LLMs) have made synthetic datasets a viable path forward. In this study, we domain-adapt LLMs to the clinical domain and generate synthetic clinical texts that are machine-annotated with tags for personally identifiable information using capable encoder-based NER models. The synthetic corpora are then used to train synthetic NER models. The results show that training NER models using synthetic corpora incurs only a small drop in predictive performance. The limits of this process are investigated in a systematic ablation study -- using both Swedish and Spanish data. Our analysis shows that smaller datasets can be sufficient for domain-adapting LLMs for data synthesis. Instead, the effectiveness of this process is almost entirely contingent on the performance of the machine-annotating NER models trained using the original data. 

**Abstract (ZH)**: 许多敏感领域，如临床领域，由于隐私风险而缺乏广泛可用的数据集。大型语言模型（LLMs）不断增强的生成能力使其成为生成合成数据集的可行途径。在本研究中，我们将LLMs适应临床领域，并使用具备强大编码器的实体识别（NER）模型对生成的临床文本进行机器标注，标注个人可识别信息标签。随后，使用这些合成语料库训练合成NER模型。结果显示，使用合成语料库训练NER模型仅会导致预测性能轻微下降。通过系统性的消除研究，我们使用瑞典和西班牙数据探索了这一过程的限制。我们的分析表明，较小的数据集可能足以适应LLMs进行数据合成。相反，这一过程的有效性几乎完全取决于使用原始数据训练的机器标注NER模型的性能。 

---
# Explanations of Deep Language Models Explain Language Representations in the Brain 

**Title (ZH)**: 深度语言模型的解释揭示了大脑中的语言表示 

**Authors**: Maryam Rahimi, Yadollah Yaghoobzadeh, Mohammad Reza Daliri  

**Link**: [PDF](https://arxiv.org/pdf/2502.14671)  

**Abstract**: Recent advances in artificial intelligence have given rise to large language models (LLMs) that not only achieve human-like performance but also share computational principles with the brain's language processing mechanisms. While previous research has primarily focused on aligning LLMs' internal representations with neural activity, we introduce a novel approach that leverages explainable AI (XAI) methods to forge deeper connections between the two domains. Using attribution methods, we quantified how preceding words contribute to an LLM's next-word predictions and employed these explanations to predict fMRI recordings from participants listening to the same narratives. Our findings demonstrate that attribution methods robustly predict brain activity across the language network, surpassing traditional internal representations in early language areas. This alignment is hierarchical: early-layer explanations correspond to the initial stages of language processing in the brain, while later layers align with more advanced stages. Moreover, the layers more influential on LLM next-word prediction$\unicode{x2014}$those with higher attribution scores$\unicode{x2014}$exhibited stronger alignment with neural activity. This work establishes a bidirectional bridge between AI and neuroscience. First, we demonstrate that attribution methods offer a powerful lens for investigating the neural mechanisms of language comprehension, revealing how meaning emerges from preceding context. Second, we propose using brain alignment as a metric to evaluate the validity of attribution methods, providing a framework for assessing their biological plausibility. 

**Abstract (ZH)**: 近年来，人工智能的最新进展催生了大规模语言模型（LLMs），这些模型不仅在性能上达到了类人的水平，还在计算原理上与大脑的语言处理机制相似。虽然之前的研究所主要集中在使LLMs的内部表示与神经活动对齐，但我们引入了一种新的方法，利用可解释的人工智能（XAI）方法，建立了两个领域之间的更深层次联系。通过归因方法，我们量化了前一个词语对LLMs下一个词语预测的贡献，并利用这些解释预测了参与者在听相同叙事时的fMRI记录。我们的发现表明，归因方法能够稳健地预测语言网络中的脑活动，在早期语言区域的表现超过了传统的内部表示。这种对齐具有层次性：早期层次的解释对应于大脑语言处理的初始阶段，而较晚的层次则与更高级的阶段相对应。此外，对LLMs下一个词语预测有较大影响的层次（即具有较高归因分数的层次）与神经活动的对齐更密切。这项工作建立了AI与神经科学之间的双向桥梁。首先，我们展示了归因方法作为一种强有力的探针，用于研究语言理解的神经机制，揭示了意义如何从背景中涌现。其次，我们提出使用脑活动对齐作为评估归因方法有效性的度量指标，从而提供了一个评估其生物合理性框架。 

---
# AlphaMaze: Enhancing Large Language Models' Spatial Intelligence via GRPO 

**Title (ZH)**: AlphaMaze：通过GRPO增强大型语言模型的空间智能 

**Authors**: Alan Dao, Dinh Bach Vu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14669)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive capabilities in language processing, yet they often struggle with tasks requiring genuine visual spatial reasoning. In this paper, we introduce a novel two-stage training framework designed to equip standard LLMs with visual reasoning abilities for maze navigation. First, we leverage Supervised Fine Tuning (SFT) on a curated dataset of tokenized maze representations to teach the model to predict step-by-step movement commands. Next, we apply Group Relative Policy Optimization (GRPO)-a technique used in DeepSeekR1-with a carefully crafted reward function to refine the model's sequential decision-making and encourage emergent chain-of-thought behaviors. Experimental results on synthetically generated mazes show that while a baseline model fails to navigate the maze, the SFT-trained model achieves 86% accuracy, and further GRPO fine-tuning boosts accuracy to 93%. Qualitative analyses reveal that GRPO fosters more robust and self-corrective reasoning, highlighting the potential of our approach to bridge the gap between language models and visual spatial tasks. These findings offer promising implications for applications in robotics, autonomous navigation, and other domains that require integrated visual and sequential reasoning. 

**Abstract (ZH)**: 大型语言模型（LLMs）在语言处理方面展现了令人印象深刻的性能，但它们在需要真实视觉空间推理的任务上往往表现不佳。本文介绍了一种新颖的两阶段训练框架，旨在使标准LLMs具备迷宫导航所需的视觉推理能力。首先，我们通过一个精心策划的迷宫表示数据集进行监督微调（SFT），让模型学会预测每一步的移动指令。接着，我们应用了DeepSeekR1中使用的组相对策略优化（GRPO）技术，并结合一个巧妙设计的奖励函数，以改进模型的顺序决策，并促进涌现的推理链行为。在合成生成的迷宫上的实验结果表明，基准模型无法导航迷宫，而经过SFT训练的模型达到了86%的准确率，进一步的GRPO微调将其准确率提升至93%。定性分析表明，GRPO促进了更为稳健和自我纠错的推理，突显了我们方法在弥补语言模型与视觉空间任务之间差距方面的潜力。这些发现为在机器人技术、自主导航以及其他需要整合视觉和序列推理的应用领域提供了有前景的启示。 

---
# InstructAgent: Building User Controllable Recommender via LLM Agent 

**Title (ZH)**: InstructAgent: 构建可由用户控制的推荐系统以利用语言模型代理 

**Authors**: Wujiang Xu, Yunxiao Shi, Zujie Liang, Xuying Ning, Kai Mei, Kun Wang, Xi Zhu, Min Xu, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14662)  

**Abstract**: Traditional recommender systems usually take the user-platform paradigm, where users are directly exposed under the control of the platform's recommendation algorithms. However, the defect of recommendation algorithms may put users in very vulnerable positions under this paradigm. First, many sophisticated models are often designed with commercial objectives in mind, focusing on the platform's benefits, which may hinder their ability to protect and capture users' true interests. Second, these models are typically optimized using data from all users, which may overlook individual user's preferences. Due to these shortcomings, users may experience several disadvantages under the traditional user-platform direct exposure paradigm, such as lack of control over the recommender system, potential manipulation by the platform, echo chamber effects, or lack of personalization for less active users due to the dominance of active users during collaborative learning. Therefore, there is an urgent need to develop a new paradigm to protect user interests and alleviate these issues. Recently, some researchers have introduced LLM agents to simulate user behaviors, these approaches primarily aim to optimize platform-side performance, leaving core issues in recommender systems unresolved. To address these limitations, we propose a new user-agent-platform paradigm, where agent serves as the protective shield between user and recommender system that enables indirect exposure. To this end, we first construct four recommendation datasets, denoted as $\dataset$, along with user instructions for each record. 

**Abstract (ZH)**: 传统的推荐系统通常采用用户-平台范式，用户在平台的推荐算法控制下直接暴露。然而，在这种范式下，推荐算法的缺陷可能使用户处于非常脆弱的位置。首先，许多复杂的模型往往是出于商业目的设计的，重点在于平台的利益，这可能妨碍它们保护和捕捉用户的真实兴趣。其次，这些模型通常通过所有用户的数据进行优化，可能会忽视个别用户的偏好。由于这些不足，用户在传统的用户-平台直接暴露范式下可能会体验到多种劣势，如对推荐系统的控制不足、平台可能存在的操纵行为、回声室效应，或者在协作学习中活跃用户主导导致不活跃用户的个性化不足。因此，迫切需要开发一种新的范式来保护用户利益并解决这些问题。最近，一些研究人员引入了大模型代理（LLM agents）来模拟用户行为，这些方法主要旨在优化平台端的表现，但未能解决推荐系统中的核心问题。为解决这些局限性，我们提出了用户-代理-平台范式，其中代理作为用户和推荐系统之间的保护屏障，实现间接暴露。为此，我们首先构建了四个推荐数据集，分别表示为$\dataset$，并为每个记录提供了用户指令。 

---
# Edit Once, Update Everywhere: A Simple Framework for Cross-Lingual Knowledge Synchronization in LLMs 

**Title (ZH)**: 一次编辑，全面更新：LLM中跨语言知识同步的简单框架 

**Authors**: Yuchen Wu, Liang Ding, Li Shen, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14645)  

**Abstract**: Knowledge editing allows for efficient adaptation of large language models (LLMs) to new information or corrections without requiring full retraining. However, prior methods typically focus on either single-language editing or basic multilingual editing, failing to achieve true cross-linguistic knowledge synchronization. To address this, we present a simple and practical state-of-the-art (SOTA) recipe Cross-Lingual Knowledge Democracy Edit (X-KDE), designed to propagate knowledge from a dominant language to other languages effectively. Our X-KDE comprises two stages: (i) Cross-lingual Edition Instruction Tuning (XE-IT), which fine-tunes the model on a curated parallel dataset to modify in-scope knowledge while preserving unrelated information, and (ii) Target-language Preference Optimization (TL-PO), which applies advanced optimization techniques to ensure consistency across languages, fostering the transfer of updates. Additionally, we contribute a high-quality, cross-lingual dataset, specifically designed to enhance knowledge transfer across languages. Extensive experiments on the Bi-ZsRE and MzsRE benchmarks show that X-KDE significantly enhances cross-lingual performance, achieving an average improvement of +8.19%, while maintaining high accuracy in monolingual settings. 

**Abstract (ZH)**: 知识编辑允许高效地调整大型语言模型（LLMs）以适应新信息或修正，而无需进行全面重训练。然而，先前的方法通常仅侧重于单语言编辑或基本的多语言编辑，未能实现真正的跨语言知识同步。为解决这一问题，我们提出了一个简单且实用的最先进的（SOTA）方法——跨语言知识民主编辑（X-KDE），该方法旨在有效传播主导语言的知识到其他语言。X-KDE 包含两个阶段：(i) 跨语言编辑指令调优 (XE-IT)，该阶段在精心策划的平行数据集上微调模型以修改相关知识并保留不相关的信息，以及 (ii) 目标语言偏好优化 (TL-PO)，该阶段应用高级优化技术以确保语言之间的一致性，促进更新的传递。此外，我们还贡献了一个高质量的跨语言数据集，专门设计用于增强跨语言知识的传递。在 Bi-ZsRE 和 MzsRE 基准测试上的广泛实验表明，X-KDE 显著提升了跨语言性能，平均提升了 +8.19%，同时在单一语言设置中保持了高准确性。 

---
# LIFT: Improving Long Context Understanding of Large Language Models through Long Input Fine-Tuning 

**Title (ZH)**: LIFT：通过长输入微调提高大型语言模型的长上下文理解能力 

**Authors**: Yansheng Mao, Yufei Xu, Jiaqi Li, Fanxu Meng, Haotong Yang, Zilong Zheng, Xiyuan Wang, Muhan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14644)  

**Abstract**: Long context understanding remains challenging for large language models due to their limited context windows. This paper presents Long Input Fine-Tuning (LIFT), a novel framework for long-context modeling that can improve the long-context performance of arbitrary (short-context) LLMs by dynamically adapting model parameters based on the long input. Importantly, LIFT, rather than endlessly extending the context window size to accommodate increasingly longer inputs in context, chooses to store and absorb the long input in parameter. By fine-tuning the long input into model parameters, LIFT allows short-context LLMs to answer questions even when the required information is not provided in the context during inference. Furthermore, to enhance LIFT performance while maintaining the original in-context learning (ICL) capabilities, we introduce Gated Memory, a specialized attention adapter that automatically balances long input memorization and ICL. We provide a comprehensive analysis of the strengths and limitations of LIFT on long context understanding, offering valuable directions for future research. 

**Abstract (ZH)**: 长上下文理解仍然是大规模语言模型面临的挑战，因其受限于有限的上下文窗口。本文提出了Long Input Fine-Tuning（LIFT），这是一种新型框架，可以通过动态调整模型参数以适应长输入，从而提升任意短上下文语言模型（LLM）的长上下文性能。重要的是，LIFT 并没有无止境地扩大上下文窗口大小以适应越来越长的输入，而是选择将长输入存储在参数中。通过将长输入微调到模型参数中，LIFT 使短上下文语言模型在推断时即使所需的上下文信息未提供也能回答问题。此外，为了在保持原有上下文学习（ICL）能力的同时提升 LIFT 性能，我们引入了门控记忆（Gated Memory），这是一种专门的注意力适配器，能够自动平衡长输入记忆和 ICL。我们从多个方面对 LIFT 在长上下文理解中的优缺点进行了全面分析，为未来的研究提供了宝贵的方向。 

---
# Length-Controlled Margin-Based Preference Optimization without Reference Model 

**Title (ZH)**: 基于边际的偏好优化控制长度无需参考模型 

**Authors**: Gengxu Li, Tingyu Xia, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14643)  

**Abstract**: Direct Preference Optimization (DPO) is a widely adopted offline algorithm for preference-based reinforcement learning from human feedback (RLHF), designed to improve training simplicity and stability by redefining reward functions. However, DPO is hindered by several limitations, including length bias, memory inefficiency, and probability degradation. To address these challenges, we propose Length-Controlled Margin-Based Preference Optimization (LMPO), a more efficient and robust alternative. LMPO introduces a uniform reference model as an upper bound for the DPO loss, enabling a more accurate approximation of the original optimization objective. Additionally, an average log-probability optimization strategy is employed to minimize discrepancies between training and inference phases. A key innovation of LMPO lies in its Length-Controlled Margin-Based loss function, integrated within the Bradley-Terry framework. This loss function regulates response length while simultaneously widening the margin between preferred and rejected outputs. By doing so, it mitigates probability degradation for both accepted and discarded responses, addressing a significant limitation of existing methods. We evaluate LMPO against state-of-the-art preference optimization techniques on two open-ended large language models, Mistral and LLaMA3, across six conditional benchmarks. Our experimental results demonstrate that LMPO effectively controls response length, reduces probability degradation, and outperforms existing approaches. The code is available at \url{this https URL}. 

**Abstract (ZH)**: 直接偏好优化（DPO）是一种广泛采用的离线算法，用于基于人类反馈的强化学习（RLHF），旨在通过重新定义奖励函数来提高训练的简单性和稳定性。然而，DPO面临着几个限制，包括长度偏见、内存使用效率低下以及概率衰减。为了解决这些挑战，我们提出了一种更高效和鲁棒的替代方案——长度可控的基于边际的偏好优化（LMPO）。LMPO引入了一个统一的参考模型作为DPO损失的上限，从而更准确地逼近原本的优化目标。此外，还采用了一种平均对数概率优化策略，以减少训练和推理阶段之间的差异。LMPO的一个关键创新在于其长度可控的基于边际的损失函数，该函数整合了Bradley-Terry框架。这种损失函数在调节响应长度的同时，还扩大了被偏好和被拒绝的输出之间的差距。通过这种方式，它可以减轻被接受和被拒绝响应的概率衰减，从而解决现有方法的一个显著限制。我们在两个开放性大型语言模型（Mistral和LLaMA3）的六个条件基准测试中，将LMPO与最先进的偏好优化技术进行了评估。实验结果表明，LMPO能够有效控制响应长度、减少概率衰减，并优于现有方法。代码可在 \url{此链接} 获取。 

---
# How Far are LLMs from Being Our Digital Twins? A Benchmark for Persona-Based Behavior Chain Simulation 

**Title (ZH)**: 大语言模型与数字孪生之间的差距：基于人设的行为链模拟基准研究 

**Authors**: Rui Li, Heming Xia, Xinfeng Yuan, Qingxiu Dong, Lei Sha, Wenjie Li, Zhifang Sui  

**Link**: [PDF](https://arxiv.org/pdf/2502.14642)  

**Abstract**: Recently, LLMs have garnered increasing attention across academic disciplines for their potential as human digital twins, virtual proxies designed to replicate individuals and autonomously perform tasks such as decision-making, problem-solving, and reasoning on their behalf. However, current evaluations of LLMs primarily emphasize dialogue simulation while overlooking human behavior simulation, which is crucial for digital twins. To address this gap, we introduce BehaviorChain, the first benchmark for evaluating LLMs' ability to simulate continuous human behavior. BehaviorChain comprises diverse, high-quality, persona-based behavior chains, totaling 15,846 distinct behaviors across 1,001 unique personas, each with detailed history and profile metadata. For evaluation, we integrate persona metadata into LLMs and employ them to iteratively infer contextually appropriate behaviors within dynamic scenarios provided by BehaviorChain. Comprehensive evaluation results demonstrated that even state-of-the-art models struggle with accurately simulating continuous human behavior. 

**Abstract (ZH)**: 近年来，大语言模型（LLMs）在学术界获得了广泛关注，因其作为人类数字双胞胎的潜力而受到重视。这些数字双胞胎是设计用于模仿个体，并自主完成诸如决策、问题解决和推理等任务的虚拟代理。然而，目前对LLMs的评估主要集中在对话模拟上，而忽视了对人类行为的模拟，这对于数字双胞胎至关重要。为弥补这一空白，我们提出了BehaviorChain，这是一个用于评估LLMs模拟连续人类行为能力的第一个基准。BehaviorChain包含多种多样的高质量行为链，共计涵盖1,001个独特的人格，这些人格之间的行为总数达到15,846种，每个性格都有详细的背景信息和个性资料。在评估过程中，我们将人格资料整合到LLMs中，并运用它们在由BehaviorChain提供的动态场景中逐步推断出上下文合适的行为。全面的评估结果表明，即使是当前最先进的模型，在准确模拟连续人类行为方面也存在困难。 

---
# NAVIG: Natural Language-guided Analysis with Vision Language Models for Image Geo-localization 

**Title (ZH)**: NAVIG：基于自然语言指导的视觉语言模型在图像地理定位中的分析方法 

**Authors**: Zheyuan Zhang, Runze Li, Tasnim Kabir, Jordan Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2502.14638)  

**Abstract**: Image geo-localization is the task of predicting the specific location of an image and requires complex reasoning across visual, geographical, and cultural contexts. While prior Vision Language Models (VLMs) have the best accuracy at this task, there is a dearth of high-quality datasets and models for analytical reasoning. We first create NaviClues, a high-quality dataset derived from GeoGuessr, a popular geography game, to supply examples of expert reasoning from language. Using this dataset, we present Navig, a comprehensive image geo-localization framework integrating global and fine-grained image information. By reasoning with language, Navig reduces the average distance error by 14% compared to previous state-of-the-art models while requiring fewer than 1000 training samples. Our dataset and code are available at this https URL. 

**Abstract (ZH)**: 图像地理定位任务是指预测图像的具体位置，这需要在视觉、地理和文化上下文中进行复杂的推理。虽然先前的视觉语言模型（VLMs）在这一任务上具有最高的准确性，但用于分析推理的高质量数据集和模型仍然稀缺。我们首先创建了一个名为NaviClues的数据集，该数据集源自流行的地缘猜谜游戏GeoGuessr，提供了语言专家推理的示例。使用此数据集，我们提出了Navig，一个整合全局和细粒度图像信息的完整图像地理定位框架。通过语言推理，Navig相比之前的顶级模型将平均距离误差降低了14%，同时只需要不到1000个训练样本。我们的数据集和代码已发布，可以通过以下链接访问：[此处替换为具体的URL链接]。 

---
# Multi-Record Web Page Information Extraction From News Websites 

**Title (ZH)**: 从新闻网站中多记录网页信息提取 

**Authors**: Alexander Kustenkov, Maksim Varlamov, Alexander Yatskov  

**Link**: [PDF](https://arxiv.org/pdf/2502.14625)  

**Abstract**: In this paper, we focused on the problem of extracting information from web pages containing many records, a task of growing importance in the era of massive web data. Recently, the development of neural network methods has improved the quality of information extraction from web pages. Nevertheless, most of the research and datasets are aimed at studying detailed pages. This has left multi-record "list pages" relatively understudied, despite their widespread presence and practical significance.
To address this gap, we created a large-scale, open-access dataset specifically designed for list pages. This is the first dataset for this task in the Russian language. Our dataset contains 13,120 web pages with news lists, significantly exceeding existing datasets in both scale and complexity. Our dataset contains attributes of various types, including optional and multi-valued, providing a realistic representation of real-world list pages. These features make our dataset a valuable resource for studying information extraction from pages containing many records.
Furthermore, we proposed our own multi-stage information extraction methods. In this work, we explore and demonstrate several strategies for applying MarkupLM to the specific challenges of multi-record web pages. Our experiments validate the advantages of our methods.
By releasing our dataset to the public, we aim to advance the field of information extraction from multi-record pages. 

**Abstract (ZH)**: 在本文中，我们专注于从包含大量记录的网页中提取信息的问题，这是一个在大规模网络数据时代日益重要的任务。近年来，神经网络方法的发展提高了从网页中提取信息的质量。然而，大多数的研究和数据集主要侧重于研究详细网页，而相对忽略了占据广泛存在和实际重要性的多记录“列表页”。鉴于此，我们创建了一个大规模且开放获取的数据集，专门用于处理列表页。这是首个以俄语编写的针对此类任务的数据集。我们的数据集包含13,120个包含新闻列表的网页，不仅在规模上，而且在复杂性上都远超现有数据集。数据集包含各种类型的属性，包括可选属性和多值属性，这为真实世界的列表页提供了现实的表示。这些特性使我们的数据集成为研究包含多个记录的页面信息提取领域的宝贵资源。

此外，我们还提出了自己的多阶段信息提取方法。在本文中，我们探讨并展示了将 MarkupLM 应用于处理多记录网页特定挑战的各种策略。我们的实验验证了我们方法的优点。

通过向公众发布我们的数据集，我们旨在推动多记录页面信息提取领域的研究进展。 

---
# Exploring RWKV for Sentence Embeddings: Layer-wise Analysis and Baseline Comparison for Semantic Similarity 

**Title (ZH)**: 探索RWKV在句向量表示中的应用：基于层的分析与语义相似性基线比较 

**Authors**: Xinghan Pan  

**Link**: [PDF](https://arxiv.org/pdf/2502.14620)  

**Abstract**: This paper investigates the efficacy of RWKV, a novel language model architecture known for its linear attention mechanism, for generating sentence embeddings in a zero-shot setting. I conduct a layer-wise analysis to evaluate the semantic similarity captured by embeddings from different hidden layers of a pre-trained RWKV model. The performance is assessed on the Microsoft Research Paraphrase Corpus (MRPC) dataset using Spearman correlation and compared against a GloVe-based baseline. My results indicate that while RWKV embeddings capture some semantic relatedness, they underperform compared to the GloVe baseline in terms of Spearman correlation. I also analyze the inference time and GPU memory usage, highlighting the computational trade-offs associated with RWKV embeddings. The findings suggest that while RWKV offers potential advantages in terms of linear scaling, its zero-shot sentence embedding quality for semantic similarity tasks requires further investigation and potential task-specific fine-tuning to match or exceed simpler baselines. 

**Abstract (ZH)**: 本文探讨了RWKV这一新型语言模型架构在零样本设置下生成句子嵌入的有效性，RWKV以线性注意力机制著称。通过逐层分析，评估了预训练RWKV模型不同隐藏层生成的嵌入所捕获的语义相似性。性能评估使用了Microsoft Research Paraphrase Corpus (MRPC)数据集，并采用Spearman相关性与基于GloVe的基线进行对比。实验结果表明，虽然RWKV嵌入能够捕获一定的语义相关信息，但在Spearman相关性上仍不如GloVe基线。本文还分析了推理时间和GPU内存使用情况，突出了RWKV嵌入相关的计算权衡。研究发现，尽管RWKV在层线性扩展方面具有潜在优势，但在语义相似性任务中的零样本句子嵌入质量仍需进一步研究，并可能需要针对特定任务的微调，以匹配或超越更简单的基线。

这段翻译符合学术规范，准确地传达了原论文内容，并保持了严谨性和专业性。 

---
# FIND: Fine-grained Information Density Guided Adaptive Retrieval-Augmented Generation for Disease Diagnosis 

**Title (ZH)**: FIND：细粒度信息密度引导的自适应检索增强生成方法在疾病诊断中的应用 

**Authors**: Mingyi Jia, Junwen Duan, Yan Song, Jianxin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14614)  

**Abstract**: Retrieval-Augmented Large Language Models (LLMs), which integrate external knowledge into LLMs, have shown remarkable performance in various medical domains, including clinical diagnosis. However, existing RAG methods struggle to effectively assess task difficulty to make retrieval decisions, thereby failing to meet the clinical requirements for balancing efficiency and accuracy. So in this paper, we propose FIND (\textbf{F}ine-grained \textbf{In}formation \textbf{D}ensity Guided Adaptive RAG), a novel framework that improves the reliability of RAG in disease diagnosis scenarios. FIND incorporates a fine-grained adaptive control module to determine whether retrieval is necessary based on the information density of the input. By optimizing the retrieval process and implementing a knowledge filtering module, FIND ensures that the retrieval is better suited to clinical scenarios. Experiments on three Chinese electronic medical record datasets demonstrate that FIND significantly outperforms various baseline methods, highlighting its effectiveness in clinical diagnosis tasks. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

检索增强的大语言模型（RAG，Retrieval-Augmented Large Language Models）通过将外部知识整合到大语言模型中，在各种医学领域，包括临床诊断中已经展现了显著的效果。然而，现有的RAG方法难以有效地评估任务难度以做出检索决策，从而未能满足临床对效率和准确性的平衡要求。因此，在本文中，我们提出了一种名为FIND（细粒度信息密度指导的自适应RAG）的新框架，该框架旨在提高RAG在疾病诊断场景中的可靠性。FIND集成了一个细粒度的自适应控制模块，该模块根据输入的信息密度来确定是否需要进行检索。通过优化检索过程并实现一个知识过滤模块，FIND确保了检索更适合于临床场景。在三个中文电子病历数据集上的实验表明，FIND显著优于各种基准方法，突显了其在临床诊断任务中的有效性。 

---
# Behavioral Analysis of Information Salience in Large Language Models 

**Title (ZH)**: 大型语言模型中信息显著性的行为分析 

**Authors**: Jan Trienes, Jörg Schlötterer, Junyi Jessy Li, Christin Seifert  

**Link**: [PDF](https://arxiv.org/pdf/2502.14613)  

**Abstract**: Large Language Models (LLMs) excel at text summarization, a task that requires models to select content based on its importance. However, the exact notion of salience that LLMs have internalized remains unclear. To bridge this gap, we introduce an explainable framework to systematically derive and investigate information salience in LLMs through their summarization behavior. Using length-controlled summarization as a behavioral probe into the content selection process, and tracing the answerability of Questions Under Discussion throughout, we derive a proxy for how models prioritize information. Our experiments on 13 models across four datasets reveal that LLMs have a nuanced, hierarchical notion of salience, generally consistent across model families and sizes. While models show highly consistent behavior and hence salience patterns, this notion of salience cannot be accessed through introspection, and only weakly correlates with human perceptions of information salience. 

**Abstract (ZH)**: 大型语言模型（LLMs）在文本摘要方面表现出色，这一任务要求模型根据内容的重要性进行选择。然而，LLMs内部化的具体显著性概念仍然不清楚。为了弥合这一差距，我们提出了一种可解释框架，系统地通过LLMs的摘要行为推导和研究信息的重要性。我们利用长度控制的摘要作为内容选择过程的行为探针，并在整个过程中追踪讨论问题的可回答性，从而推导出模型对信息优先级的代理指标。在四个数据集上对13个模型的实验表明，LLMs具有复杂的、分层级的重要性概念，这种概念在不同模型家族和规模中通常是保持一致的。虽然模型表现出高度一致的行为和显著性模式，但这种重要性的概念却无法通过自我反思获取，并且与人类对信息显著性的感知仅有弱相关。 

---
# Can LLMs Predict Citation Intent? An Experimental Analysis of In-context Learning and Fine-tuning on Open LLMs 

**Title (ZH)**: 大型语言模型能否预测引用意图？关于开放大型语言模型的上下文学习和微调的实验分析 

**Authors**: Paris Koloveas, Serafeim Chatzopoulos, Thanasis Vergoulis, Christos Tryfonopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2502.14561)  

**Abstract**: This work investigates the ability of open Large Language Models (LLMs) to predict citation intent through in-context learning and fine-tuning. Unlike traditional approaches that rely on pre-trained models like SciBERT, which require extensive domain-specific pretraining and specialized architectures, we demonstrate that general-purpose LLMs can be adapted to this task with minimal task-specific data. We evaluate twelve model variations across five prominent open LLM families using zero, one, few, and many-shot prompting to assess performance across scenarios. Our experimental study identifies the top-performing model through extensive experimentation of in-context learning-related parameters, which we fine-tune to further enhance task performance. The results highlight the strengths and limitations of LLMs in recognizing citation intents, providing valuable insights for model selection and prompt engineering. Additionally, we make our end-to-end evaluation framework and models openly available for future use. 

**Abstract (ZH)**: 本研究探讨了开放式大规模语言模型（LLMs）通过上下文学习和微调来预测引文意图的能力。与依赖于像SciBERT这样的预训练模型的传统方法不同，后者需要大量的领域特定预训练和专门的架构，我们展示了通用语言模型可以借助极少的任务特定数据来适应这一任务。我们使用零样本、单样本、少量样本和多样本提示，评估了五个主要的开放式LLM家族中的十二种模型变体，以评估其在不同场景下的性能。通过广泛实验相关上下文学习参数，我们的研究确定了表现最佳的模型，并进一步对其进行微调以提高任务性能。研究结果突显了LLMs在识别引文意图方面的优势和局限性，为模型选择和提示工程提供了宝贵见解。此外，我们还公开提供了完整的评估框架和模型，以供未来使用。 

---
# Multiscale Byte Language Models -- A Hierarchical Architecture for Causal Million-Length Sequence Modeling 

**Title (ZH)**: 多尺度字语言模型——一种用于因果百万长度序列建模的分层架构 

**Authors**: Eric Egli, Matteo Manica, Jannis Born  

**Link**: [PDF](https://arxiv.org/pdf/2502.14553)  

**Abstract**: Bytes form the basis of the digital world and thus are a promising building block for multimodal foundation models. Recently, Byte Language Models (BLMs) have emerged to overcome tokenization, yet the excessive length of bytestreams requires new architectural paradigms. Therefore, we present the Multiscale Byte Language Model (MBLM), a model-agnostic hierarchical decoder stack that allows training with context windows of $5$M bytes on single GPU in full model precision. We thoroughly examine MBLM's performance with Transformer and Mamba blocks on both unimodal and multimodal tasks. Our experiments demonstrate that hybrid architectures are efficient in handling extremely long byte sequences during training while achieving near-linear generational efficiency. To the best of our knowledge, we present the first evaluation of BLMs on visual Q\&A tasks and find that, despite serializing images and the absence of an encoder, a MBLM with pure next token prediction can match custom CNN-LSTM architectures with designated classification heads. We show that MBLMs exhibit strong adaptability in integrating diverse data representations, including pixel and image filestream bytes, underlining their potential toward omnimodal foundation models. Source code is publicly available at: this https URL 

**Abstract (ZH)**: 以下是翻译后的论文内容或标题，符合学术规范：

字节构成了数字世界的基石，因此成为多模态基础模型的有前景构建块。近期，字节语言模型（BLMs）已经出现，以克服分词的问题。然而，字节流的过长需要新的架构范式。因此，我们提出了多尺度字节语言模型（MBLM），这是一种模型无关的分层解码堆栈，允许在单个GPU上使用5M字节的上下文窗口进行全精度模型训练。我们全面考察了MBLM在Transformer和Mamba块下在单模态和多模态任务中的性能。实验证明，在训练过程中，混合架构能够高效地处理极其长的字节序列，同时实现接近线性的生成效率。据我们所知，这是首次对BLM在视觉问答任务上的评估，结果显示，在未使用图像编码器的情况下，一种仅以下一个词预测字节的MBLM可以与针对特定分类头设计的CNN-LSTM架构相媲美。我们展示了MBLM在整合各种数据表示（包括像素和图像文件流字节）方面的强大适应性，这突出了其向全模态基础模型发展的潜力。相关代码已公开发布：this https URL 

---
# LLM-based User Profile Management for Recommender System 

**Title (ZH)**: 基于LLM的用户画像管理在推荐系统中的应用 

**Authors**: Seunghwan Bang, Hwanjun Song  

**Link**: [PDF](https://arxiv.org/pdf/2502.14541)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened new opportunities in recommender systems by enabling zero-shot recommendation without conventional training. Despite their potential, most existing works rely solely on users' purchase histories, leaving significant room for improvement by incorporating user-generated textual data, such as reviews and product descriptions. Addressing this gap, we propose PURE, a novel LLM-based recommendation framework that builds and maintains evolving user profiles by systematically extracting and summarizing key information from user reviews. PURE consists of three core components: a Review Extractor for identifying user preferences and key product features, a Profile Updater for refining and updating user profiles, and a Recommender for generating personalized recommendations using the most current profile. To evaluate PURE, we introduce a continuous sequential recommendation task that reflects real-world scenarios by adding reviews over time and updating predictions incrementally. Our experimental results on Amazon datasets demonstrate that PURE outperforms existing LLM-based methods, effectively leveraging long-term user information while managing token limitations. 

**Abstract (ZH)**: 大型语言模型（LLMs）的迅速发展为推荐系统开辟了新的机会，使其能够在不依赖传统训练的情况下实现零样本推荐。尽管具有潜力，现有的大多数研究依旧主要依赖用户的购买历史，而未能充分利用用户生成的文本数据，如评论和产品描述，这为改进提供了很大的空间。为弥补这一空白，我们提出了PURE，一种新颖的基于LLM的推荐框架，通过系统地从用户评论中提取和总结关键信息来构建和发展演变中的用户画像。PURE由三个核心组件组成：评论提取器用于识别用户偏好和关键产品特征；用户画像更新器用于完善和更新用户画像；以及推荐器使用最新的画像生成个性化推荐。为了评估PURE，我们引入了一项连续的序贯推荐任务，通过随时间添加评论并在预测中逐步更新，以更好地反映现实场景。在亚马逊数据集上的实验结果表明，PURE 在利用长期用户信息的同时管理词汇量限制方面优于现有的基于LLM的方法。 

---
# LoRA-GGPO: Mitigating Double Descent in LoRA Fine-Tuning via Gradient-Guided Perturbation Optimization 

**Title (ZH)**: LoRA-GGPO：通过梯度引导扰动优化缓解LoRA微调中的双重下降问题 

**Authors**: Yupeng Chang, Chenlu Guo, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14538)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success in natural language processing, but their full fine-tuning remains resource-intensive. Parameter-Efficient Fine-Tuning (PEFT) methods, such as Low-Rank Adaptation (LoRA), have emerged as a practical solution by approximating parameter updates with low-rank matrices. However, LoRA often exhibits a "double descent" phenomenon during fine-tuning, where model performance degrades due to overfitting and limited expressiveness caused by low-rank constraints. To address this issue, we propose LoRA-GGPO (Gradient-Guided Perturbation Optimization), a novel method that leverages gradient and weight norms to generate targeted perturbations. By optimizing the sharpness of the loss landscape, LoRA-GGPO guides the model toward flatter minima, mitigating the double descent problem and improving generalization. Extensive experiments on natural language understanding (NLU) and generation (NLG) tasks demonstrate that LoRA-GGPO outperforms LoRA and its state-of-the-art variants. Furthermore, extended experiments specifically designed to analyze the double descent phenomenon confirm that LoRA-GGPO effectively alleviates this issue, producing more robust and generalizable models. Our work provides a robust and efficient solution for fine-tuning LLMs, with broad applicability in real-world scenarios. The code is available at this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在自然语言处理领域已经取得了显著的成功，但其完全微调仍然消耗大量资源。参数高效微调（PEFT）方法，如低秩适应（LoRA），通过使用低秩矩阵近似参数更新，已成为一种实际的解决方案。然而，在微调过程中，LoRA 经常表现出“双重下降”现象，即因过拟合和由低秩约束引起的表达能力限制，导致模型性能下降。为了解决这一问题，我们提出了 LoRA-GGPO（梯度引导的扰动优化）方法，这是一种结合梯度和权重范数生成目标扰动的新型方法。通过优化损失景观的锋利度，LoRA-GGPO 引导模型朝着更平坦的极小值发展，缓解了双重下降问题，提高了泛化能力。在自然语言理解（NLU）和生成（NLG）任务上的广泛实验表明，LoRA-GGPO 在性能上优于 LoRA 及其最先进的变体。此外，特别设计的延伸实验专门用于分析双重下降现象，证实了 LoRA-GGPO 能够有效缓解这一问题，生成更稳健和可泛化的模型。我们的工作为大规模语言模型的微调提供了一种稳健且高效的解决方案，并具有广泛的实际应用场景。代码可用 [点击此处](this https URL) 获取。 

---
# CORBA: Contagious Recursive Blocking Attacks on Multi-Agent Systems Based on Large Language Models 

**Title (ZH)**: 基于大型语言模型的多智能体系统中的传染性递归阻断攻击：CORBA 

**Authors**: Zhenhong Zhou, Zherui Li, Jie Zhang, Yuanhe Zhang, Kun Wang, Yang Liu, Qing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.14529)  

**Abstract**: Large Language Model-based Multi-Agent Systems (LLM-MASs) have demonstrated remarkable real-world capabilities, effectively collaborating to complete complex tasks. While these systems are designed with safety mechanisms, such as rejecting harmful instructions through alignment, their security remains largely unexplored. This gap leaves LLM-MASs vulnerable to targeted disruptions. In this paper, we introduce Contagious Recursive Blocking Attacks (Corba), a novel and simple yet highly effective attack that disrupts interactions between agents within an LLM-MAS. Corba leverages two key properties: its contagious nature allows it to propagate across arbitrary network topologies, while its recursive property enables sustained depletion of computational resources. Notably, these blocking attacks often involve seemingly benign instructions, making them particularly challenging to mitigate using conventional alignment methods. We evaluate Corba on two widely-used LLM-MASs, namely, AutoGen and Camel across various topologies and commercial models. Additionally, we conduct more extensive experiments in open-ended interactive LLM-MASs, demonstrating the effectiveness of Corba in complex topology structures and open-source models. Our code is available at: this https URL. 

**Abstract (ZH)**: 基于大型语言模型的多代理系统（LLM-MASs）在现实世界中展现出了显著的能力，能够有效协作以完成复杂的任务。尽管这些系统在设计时包含了安全机制，例如通过对齐方式拒绝有害指令，但其安全性仍然很大程度上未受到探索，这使得LLM-MASs容易遭受针对性的破坏。在本文中，我们介绍了一种新颖且简单但高度有效的攻击方法——传染性递归阻断攻击（Corba），这种攻击能够破坏LLM-MAS中代理之间的交互。Corba 利用两种关键特性：其传染性使其能够传播到任意网络拓扑结构中，而其递归性则使其实现持续的计算资源耗竭。值得注意的是，这些阻断攻击往往涉及看似无害的指令，这使其难以通过传统的对齐方法来缓解。我们在两个广泛使用的LLM-MASs，即AutoGen和Camel上，以及各种网络拓扑和商业模型上评估了Corba的有效性。此外，我们还在开放式的交互式LLM-MASs中进行了更广泛的实验，证明了Corba在复杂拓扑结构和开源模型中的有效性。我们的代码可在以下链接获取：this https URL。 

---
# MultiSlav: Using Cross-Lingual Knowledge Transfer to Combat the Curse of Multilinguality 

**Title (ZH)**: 多语种知识迁移的应用：克服多语种化带来的挑战 

**Authors**: Artur Kot, Mikołaj Koszowski, Wojciech Chojnowski, Mieszko Rutkowski, Artur Nowakowski, Kamil Guttmann, Mikołaj Pokrywka  

**Link**: [PDF](https://arxiv.org/pdf/2502.14509)  

**Abstract**: Does multilingual Neural Machine Translation (NMT) lead to The Curse of the Multlinguality or provides the Cross-lingual Knowledge Transfer within a language family? In this study, we explore multiple approaches for extending the available data-regime in NMT and we prove cross-lingual benefits even in 0-shot translation regime for low-resource languages. With this paper, we provide state-of-the-art open-source NMT models for translating between selected Slavic languages. We released our models on the HuggingFace Hub (this https URL) under the CC BY 4.0 license. Slavic language family comprises morphologically rich Central and Eastern European languages. Although counting hundreds of millions of native speakers, Slavic Neural Machine Translation is under-studied in our opinion. Recently, most NMT research focuses either on: high-resource languages like English, Spanish, and German - in WMT23 General Translation Task 7 out of 8 task directions are from or to English; massively multilingual models covering multiple language groups; or evaluation techniques. 

**Abstract (ZH)**: 多语言神经机器翻译（NMT）是带来语言多样性诅咒还是提供语言家族间的跨语言知识迁移？在本研究中，我们探索了多种方法以扩展NMT的数据范围，并证明即使在零样本翻译情况下，低资源语言也能从中受益。通过本文，我们为翻译选定的斯拉夫语言提供了当前最先进的开源NMT模型。我们已将这些模型发布在HuggingFace Hub上 (https://...)，并采用CC BY 4.0许可协议。斯拉夫语族包含形态丰富的中欧和东欧语言。尽管拥有数亿母语使用者，我们认为斯拉夫语神经机器翻译的研究仍显不足。近年来，NMT研究主要集中在高资源语言（如英语、西班牙语和德语）及其相关的任务中，在WMT23通用翻译任务中，有7个方向的任务是关于英语的；或是在涵盖多个语言组的大型多语言模型上；或者是在评估技术上。 

---
# Can LLMs Simulate L2-English Dialogue? An Information-Theoretic Analysis of L1-Dependent Biases 

**Title (ZH)**: LLMs能否模拟二外英语对话？一种基于信息论的母语依赖偏差分析 

**Authors**: Rena Gao, Xuetong Wu, Tatsuki Kuribayashi, Mingrui Ye, Siya Qi, Carsten Roever, Yuanxing Liu, Zheng Yuan, Jey Han Lau  

**Link**: [PDF](https://arxiv.org/pdf/2502.14507)  

**Abstract**: This study evaluates Large Language Models' (LLMs) ability to simulate non-native-like English use observed in human second language (L2) learners interfered with by their native first language (L1). In dialogue-based interviews, we prompt LLMs to mimic L2 English learners with specific L1s (e.g., Japanese, Thai, Urdu) across seven languages, comparing their outputs to real L2 learner data. Our analysis examines L1-driven linguistic biases, such as reference word usage and avoidance behaviors, using information-theoretic and distributional density measures. Results show that modern LLMs (e.g., Qwen2.5, LLAMA3.3, DeepseekV3, GPT-4o) replicate L1-dependent patterns observed in human L2 data, with distinct influences from various languages (e.g., Japanese, Korean, and Mandarin significantly affect tense agreement, and Urdu influences noun-verb collocations). Our results reveal the potential of LLMs for L2 dialogue generation and evaluation for future educational applications. 

**Abstract (ZH)**: 本研究评估了大型语言模型（LLMs）模拟受母语（L1）干扰的人类第二语言（L2）学习者所使用的非母语似的英语能力。通过基于对话的访谈，我们促使LLMs模拟具有特定L1（如日语、泰语、乌尔都语）的学习者的L2英语使用情况，并将它们的输出与真实的L2学习者数据进行比较。分析采用信息论和分布密度量度来考察由L1驱动的语言偏见，例如参考词的使用和回避行为。结果表明，现代LLMs（如Qwen2.5、LLAMA3.3、DeepSeekV3、GPT-4o）能够复制人类L2数据中观察到的L1依赖模式，不同语言（如日语、韩语和普通话显著影响时态一致性，而乌尔都语影响名词与动词的搭配）对这些模式有不同的影响。我们的研究结果揭示了LLMs在未来的教育应用中生成和评估L2对话的潜在价值。 

---
# How Much Knowledge Can You Pack into a LoRA Adapter without Harming LLM? 

**Title (ZH)**: 你可以在LoRA适配器中打包多少知识而不损害大语言模型的性能？ 

**Authors**: Sergey Pletenev, Maria Marina, Daniil Moskovskiy, Vasily Konovalov, Pavel Braslavski, Alexander Panchenko, Mikhail Salnikov  

**Link**: [PDF](https://arxiv.org/pdf/2502.14502)  

**Abstract**: The performance of Large Language Models (LLMs) on many tasks is greatly limited by the knowledge learned during pre-training and stored in the model's parameters. Low-rank adaptation (LoRA) is a popular and efficient training technique for updating or domain-specific adaptation of LLMs. In this study, we investigate how new facts can be incorporated into the LLM using LoRA without compromising the previously learned knowledge. We fine-tuned Llama-3.1-8B-instruct using LoRA with varying amounts of new knowledge. Our experiments have shown that the best results are obtained when the training data contains a mixture of known and new facts. However, this approach is still potentially harmful because the model's performance on external question-answering benchmarks declines after such fine-tuning. When the training data is biased towards certain entities, the model tends to regress to few overrepresented answers. In addition, we found that the model becomes more confident and refuses to provide an answer in only few cases. These findings highlight the potential pitfalls of LoRA-based LLM updates and underscore the importance of training data composition and tuning parameters to balance new knowledge integration and general model capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）在许多任务上的表现受到预训练期间学习的知识以及存储在模型参数中的知识的限制。低秩适应（LoRA）是一种流行的高效训练技术，用于更新或特定领域适应LLMs。在本研究中，我们探讨了如何使用LoRA将新事实融入LLMs，而不损害先前学习的知识。我们使用不同的新知识量对Llama-3.1-8B-instruct进行微调。我们的实验表明，当训练数据包含已知和新知识的混合时，可以获得最佳结果。然而，这种做法仍然可能带来危害，因为在这种微调之后，模型在外部问答基准测试中的性能会下降。当训练数据偏向某些实体时，模型往往会回归到少数过代表的答案。此外，我们发现模型变得更有信心，并且只在少数情况下拒绝提供答案。这些发现强调了基于LoRA的LLM更新可能带来的潜在风险，并突显了训练数据组成和微调参数的重要性，以平衡新知识的整合和通用模型能力。 

---
# Towards a Perspectivist Turn in Argument Quality Assessment 

**Title (ZH)**: 朝向观点主义转向在论证质量评估中的应用 

**Authors**: Julia Romberg, Maximilian Maurer, Henning Wachsmuth, Gabriella Lapesa  

**Link**: [PDF](https://arxiv.org/pdf/2502.14501)  

**Abstract**: The assessment of argument quality depends on well-established logical, rhetorical, and dialectical properties that are unavoidably subjective: multiple valid assessments may exist, there is no unequivocal ground truth. This aligns with recent paths in machine learning, which embrace the co-existence of different perspectives. However, this potential remains largely unexplored in NLP research on argument quality. One crucial reason seems to be the yet unexplored availability of suitable datasets. We fill this gap by conducting a systematic review of argument quality datasets. We assign them to a multi-layered categorization targeting two aspects: (a) What has been annotated: we collect the quality dimensions covered in datasets and consolidate them in an overarching taxonomy, increasing dataset comparability and interoperability. (b) Who annotated: we survey what information is given about annotators, enabling perspectivist research and grounding our recommendations for future actions. To this end, we discuss datasets suitable for developing perspectivist models (i.e., those containing individual, non-aggregated annotations), and we showcase the importance of a controlled selection of annotators in a pilot study. 

**Abstract (ZH)**: 评估论据质量依赖于逻辑、修辞和辩证性等已经确立的属性，这些属性不可避免地具有主观性：可能存在多种有效的评估方法，不存在确切的绝对真理。这一点与最近的机器学习路径一致，这些路径接受不同视角的共存。然而，在自然语言处理（NLP）研究中，论据质量的研究中这一潜在可能性仍然未被充分探索。一个关键原因似乎是尚未探索到适合的数据集。我们通过系统地审查论据质量数据集来填补这一空白。我们对这些数据集进行了多层面的分类，重点关注两个方面：（a）已标注的内容：我们收集数据集中涵盖的质量维度，并将它们汇总为一个综合分类体系，从而提高数据集的可比性和兼容性；（b）谁进行了标注：我们调查了关于标注者的相关信息，以支持视角主义研究，并为未来行为提供参考建议。为此，我们讨论了适合构建视角主义模型的数据集（即包含个体非汇总标注的数据集），并在试点研究中展示了选择标注者时控制其选择的重要性。 

---
# MLGym: A New Framework and Benchmark for Advancing AI Research Agents 

**Title (ZH)**: MLGym：一个促进人工智能研究代理发展的新框架与基准 

**Authors**: Deepak Nathani, Lovish Madaan, Nicholas Roberts, Nikolay Bashlykov, Ajay Menon, Vincent Moens, Amar Budhiraja, Despoina Magka, Vladislav Vorotilov, Gaurav Chaurasia, Dieuwke Hupkes, Ricardo Silveira Cabral, Tatiana Shavrina, Jakob Foerster, Yoram Bachrach, William Yang Wang, Roberta Raileanu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14499)  

**Abstract**: We introduce Meta MLGym and MLGym-Bench, a new framework and benchmark for evaluating and developing LLM agents on AI research tasks. This is the first Gym environment for machine learning (ML) tasks, enabling research on reinforcement learning (RL) algorithms for training such agents. MLGym-bench consists of 13 diverse and open-ended AI research tasks from diverse domains such as computer vision, natural language processing, reinforcement learning, and game theory. Solving these tasks requires real-world AI research skills such as generating new ideas and hypotheses, creating and processing data, implementing ML methods, training models, running experiments, analyzing the results, and iterating through this process to improve on a given task. We evaluate a number of frontier large language models (LLMs) on our benchmarks such as Claude-3.5-Sonnet, Llama-3.1 405B, GPT-4o, o1-preview, and Gemini-1.5 Pro. Our MLGym framework makes it easy to add new tasks, integrate and evaluate models or agents, generate synthetic data at scale, as well as develop new learning algorithms for training agents on AI research tasks. We find that current frontier models can improve on the given baselines, usually by finding better hyperparameters, but do not generate novel hypotheses, algorithms, architectures, or substantial improvements. We open-source our framework and benchmark to facilitate future research in advancing the AI research capabilities of LLM agents. 

**Abstract (ZH)**: 我们介绍了Meta MLGym和MLGym-Bench，这是一个新的框架和基准，用于评估和开发在AI研究任务中工作的LLM代理。这是首个适用于机器学习（ML）任务的Gym环境，允许研究人员对训练此类代理的强化学习（RL）算法进行研究。MLGym-Bench包含来自计算机视觉、自然语言处理、强化学习和博弈论等多个领域的13项多样性和开放性AI研究任务。解决这些问题需要真实世界的AI研究技能，如产生新想法和假设、创建和处理数据、实现ML方法、训练模型、运行实验、分析结果以及在这一过程中的迭代以提高任务表现。我们在基准测试中评估了多个前沿大型语言模型（LLMs），例如Claude-3.5-Sonnet、Llama-3.1 405B、GPT-4o、o1-preview 和 Gemini-1.5 Pro。我们的MLGym框架使新增任务、集成和评估模型或代理、大规模生成合成数据以及为训练AI研究任务中的代理开发新的学习算法变得容易。我们发现，当前的前沿模型通常可以通过找到更好的超参数来改进给定的基本方法，但它们未能生成新的假设、算法、架构或实质性改进。我们开源了我们的框架和基准，以促进未来对提升LLM代理AI研究能力的研究。 

---
# Stories that (are) Move(d by) Markets: A Causal Exploration of Market Shocks and Semantic Shifts across Different Partisan Groups 

**Title (ZH)**: 市场波动与其引发的故事转变：不同政治倾向群体中市场冲击与语义变化的因果探究 

**Authors**: Felix Drinkall, Stefan Zohren, Michael McMahon, Janet B. Pierrehumbert  

**Link**: [PDF](https://arxiv.org/pdf/2502.14497)  

**Abstract**: Macroeconomic fluctuations and the narratives that shape them form a mutually reinforcing cycle: public discourse can spur behavioural changes leading to economic shifts, which then result in changes in the stories that propagate. We show that shifts in semantic embedding space can be causally linked to financial market shocks -- deviations from the expected market behaviour. Furthermore, we show how partisanship can influence the predictive power of text for market fluctuations and shape reactions to those same shocks. We also provide some evidence that text-based signals are particularly salient during unexpected events such as COVID-19, highlighting the value of language data as an exogenous variable in economic forecasting. Our findings underscore the bidirectional relationship between news outlets and market shocks, offering a novel empirical approach to studying their effect on each other. 

**Abstract (ZH)**: 宏观经济波动及其塑造这些波动的叙事构成一种相互强化的循环：公共话语可以激发行为变化，进而导致经济变化，这些变化又会导致传播的故事发生变化。我们展示了语义嵌入空间的变化可以与金融市场冲击（即市场行为的偏差）建立因果联系。此外，我们还表明，政治倾向可以影响文本在预测市场波动方面的预测能力，并塑造对这些冲击的反应。同时，我们也提供了一些证据表明，在如COVID-19这样的意外事件中，基于文本的信号尤为突出，突显了语言数据作为经济预测外生变量的价值。我们的研究强调了新闻媒体与市场冲击之间的双向关系，提供了研究它们彼此影响的一种新颖实证方法。 

---
# Enhancing Language Multi-Agent Learning with Multi-Agent Credit Re-Assignment for Interactive Environment Generalization 

**Title (ZH)**: 通过多agent信用重新分配增强语言多agent学习以实现交互环境泛化 

**Authors**: Zhitao He, Zijun Liu, Peng Li, May Fung, Ming Yan, Ji Zhang, Fei Huang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14496)  

**Abstract**: LLM-based agents have made significant advancements in interactive environments, such as mobile operations and web browsing, and other domains beyond computer using. Current multi-agent systems universally excel in performance, compared to single agents, but struggle with generalization across environments due to predefined roles and inadequate strategies for generalizing language agents. The challenge of achieving both strong performance and good generalization has hindered the progress of multi-agent systems for interactive environments. To address these issues, we propose CollabUIAgents, a multi-agent reinforcement learning framework with a novel multi-agent credit re-assignment (CR) strategy, assigning process rewards with LLMs rather than environment-specific rewards and learning with synthesized preference data, in order to foster generalizable, collaborative behaviors among the role-free agents' policies. Empirical results show that our framework improves both performance and cross-environment generalizability of multi-agent systems. Moreover, our 7B-parameter system achieves results on par with or exceed strong closed-source models, and the LLM that guides the CR. We also provide insights in using granular CR rewards effectively for environment generalization, and accommodating trained LLMs in multi-agent systems. 

**Abstract (ZH)**: 基于大语言模型（LLM）的智能体在交互环境中取得了显著进展，包括移动操作和网页浏览等领域，以及其他不依赖计算机的领域。目前，多智能体系统在与单智能体相比时普遍表现出更优秀的性能，但因预设的角色限制和语言智能体泛化策略不足，在跨环境的泛化上存在困难。实现强性能和良好泛化之间的挑战阻碍了交互环境中多智能体系统的进步。为了解决这些问题，我们提出了CollabUIAgents，这是一种具有新颖的多智能体信用重分配（CR）策略的多智能体增强学习框架。该框架通过使用LLM分配过程奖励，而非环境特定的奖励，并利用合成偏好数据进行学习，来促进无角色限制智能体政策之间的协作行为泛化。实验结果表明，我们的框架提高了多智能体系统的性能和跨环境泛化能力。此外，我们的7亿参数系统在性能上达到了或超过了强大的封闭源模型，并且引导CR的LLM也取得了优异成绩。我们还提供了如何有效使用粒度化的CR奖励以促进环境泛化以及如何在多智能体系统中纳入训练好的LLM的见解。 

---
# StructFlowBench: A Structured Flow Benchmark for Multi-turn Instruction Following 

**Title (ZH)**: StructFlowBench：一种用于多轮指令跟随的结构化流程基准测试 

**Authors**: Jinnan Li, Jinzhe Li, Yue Wang, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14494)  

**Abstract**: Multi-turn instruction following capability constitutes a core competency of large language models (LLMs) in real-world applications. Existing evaluation benchmarks predominantly focus on fine-grained constraint satisfaction and domain-specific capability assessment, yet overlook the crucial structural dependency between dialogue turns that distinguishes multi-turn from single-turn interactions. This structural dependency not only reflects user intent but also establishes a second dimension for instruction following evaluation beyond constraint satisfaction. To address this gap, we propose StructFlowBench, a multi-turn instruction following benchmark with structural flow modeling. The benchmark innovatively defines a structural flow framework comprising six fundamental inter-turn relationships, which not only introduces novel structural constraints for model evaluation but also serves as generation parameters for creating customized dialogue flows tailored to specific scenarios. Adopting established LLM-based automatic evaluation methodologies, we conduct systematic evaluations of 13 leading open-source and closed-source LLMs. Experimental results reveal significant deficiencies in current models' comprehension of multi-turn dialogue structures. The code is available at \url{this https URL}. 

**Abstract (ZH)**: 多轮指令跟随能力是大型语言模型（LLMs）在实际应用中的核心竞争力。现有的评价基准主要关注细粒度的约束满足和领域特定的能力评估，而较少注意到区分多轮交互与单轮交互的关键结构依赖性。这种结构依赖性不仅反映了用户意图，还为指令跟随评价引入了除约束满足之外的第二维度。为弥补这一缺陷，我们提出了StructFlowBench，这是一种具有结构流建模的多轮指令跟随基准。该基准创新性地定义了一个由六种基本的跨轮关系组成的结构流框架，不仅引入了新的结构约束用于模型评估，还能生成定制化的对话流程，以适应特定场景。采用现有的LLM基自动评价方法，我们系统性地评估了13个领先的开源和闭源LLM。实验结果揭示了当前模型在理解多轮对话结构方面存在的显著不足。代码可在 \url{此链接} 获取。 

---
# NLoRA: Nyström-Initiated Low-Rank Adaptation for Large Language Models 

**Title (ZH)**: NLoRA: Nyström 初始化低秩适应性方法用于大型语言模型 

**Authors**: Chenlu Guo, Yuan Wu, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14482)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) is essential for adapting large language models (LLMs), with low-rank adaptation (LoRA) being the most popular approach. However, LoRA suffers from slow convergence, and some recent LoRA variants, such as PiSSA, primarily rely on Singular Value Decomposition (SVD) for initialization, leading to expensive computation. To mitigate these problems, we use the Nyström method, which follows a three-matrix manipulation. We first introduce StructuredLoRA (SLoRA), which investigates adding a small intermediate matrix between the low-rank matrices A and B. Secondly, we propose NyströmLoRA (NLoRA), which leverages Nyström-based initialization for SLoRA to improve its effectiveness and efficiency. Finally, we propose IntermediateTune (IntTune), which explores fine-tuning exclusively on the intermediate matrix of NLoRA to further boost LLM efficiency. We evaluate our methods on five natural language generation (NLG) tasks and eight natural language understanding (NLU) tasks. On GSM8K, SLoRA and NLoRA achieve accuracies of 56.48% and 57.70%, surpassing LoRA by 33.52% and 36.41%, with only 3.67 million additional trainable parameters. IntTune improves average NLG performance over LoRA by 7.45% while using only 1.25% of its parameters. These results demonstrate the efficiency and effectiveness of our approach in enhancing model performance with minimal parameter overhead. 

**Abstract (ZH)**: 参数高效微调（PEFT）对于适应大型语言模型（LLMs）至关重要，其中低秩适应（LoRA）是最受欢迎的方法。然而，LoRA在收敛速度方面存在问题，一些最近的LoRA变体，如PiSSA，主要依赖奇异值分解（SVD）进行初始化，导致计算成本高昂。为了解决这些问题，我们采用Nyström方法，该方法遵循三矩阵操作的模式。首先，我们引入了结构化LoRA（SLoRA），研究了在低秩矩阵A和B之间添加一个小的中间矩阵的可能性。其次，我们提出了NyströmLoRA（NLoRA），利用基于Nyström的初始化来提升SLoRA的有效性和效率。最后，我们提出了中间调优（IntTune），专注于对NLoRA的中间矩阵进行微调，以进一步提高LLM的效率。我们在五个自然语言生成（NLG）任务和八个自然语言理解（NLU）任务上对我们的方法进行了评估。在GSM8K数据集上，SLoRA和NLoRA分别获得了56.48%和57.70%的准确率，分别优于LoRA 33.52%和36.41%，并且只增加了367万个可训练参数。IntTune在NLG性能上比LoRA提高了7.45%，但仅使用了LoRA参数的1.25%。这些结果表明，我们的方法在最小参数开销的情况下能够有效提升模型性能。 

---
# Unshackling Context Length: An Efficient Selective Attention Approach through Query-Key Compression 

**Title (ZH)**: 解除上下文长度限制：一种通过查询-关键子压缩实现的高效选择性注意力方法 

**Authors**: Haoyu Wang, Tong Teng, Tianyu Guo, An Xiao, Duyu Tang, Hanting Chen, Yunhe Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14477)  

**Abstract**: Handling long-context sequences efficiently remains a significant challenge in large language models (LLMs). Existing methods for token selection in sequence extrapolation either employ a permanent eviction strategy or select tokens by chunk, which may lead to the loss of critical information. We propose Efficient Selective Attention (ESA), a novel approach that extends context length by efficiently selecting the most critical tokens at the token level to compute attention. ESA reduces the computational complexity of token selection by compressing query and key vectors into lower-dimensional representations. We evaluate ESA on long sequence benchmarks with maximum lengths up to 256k using open-source LLMs with context lengths of 8k and 32k. ESA outperforms other selective attention methods, especially in tasks requiring the retrieval of multiple pieces of information, achieving comparable performance to full-attention extrapolation methods across various tasks, with superior results in certain tasks. 

**Abstract (ZH)**: 高效处理长上下文序列仍然是大型语言模型（LLMs）中的一个重要挑战。现有序列外推的 token 选择方法要么采用永久驱逐策略，要么按块选择 token，这可能导致关键信息的丢失。我们提出了一种名为高效选择性注意（Efficient Selective Attention, ESA）的新方法，该方法在 token 级别通过高效选择最重要的 token 来计算注意机制，从而扩展上下文长度。ESA 通过压缩查询向量和键向量到低维度表示来减少 token 选择的计算复杂度。我们使用具有 8k 和 32k 上下文长度的开源 LLM，在最长长度达 256k 的长序列基准上评估了 ESA。在多种任务中，ESA 在检索多个信息片段的任务中表现优于其他选择性注意方法，且在各种任务中能达到全注意机制外推方法的可比性能，在某些任务中表现更优。 

---
# Argument-Based Comparative Question Answering Evaluation Benchmark 

**Title (ZH)**: 基于论证的比较型问答评估基准 

**Authors**: Irina Nikishina, Saba Anwar, Nikolay Dolgov, Maria Manina, Daria Ignatenko, Viktor Moskvoretskii, Artem Shelmanov, Tim Baldwin, Chris Biemann  

**Link**: [PDF](https://arxiv.org/pdf/2502.14476)  

**Abstract**: In this paper, we aim to solve the problems standing in the way of automatic comparative question answering. To this end, we propose an evaluation framework to assess the quality of comparative question answering summaries. We formulate 15 criteria for assessing comparative answers created using manual annotation and annotation from 6 large language models and two comparative question asnwering datasets. We perform our tests using several LLMs and manual annotation under different settings and demonstrate the constituency of both evaluations. Our results demonstrate that the Llama-3 70B Instruct model demonstrates the best results for summary evaluation, while GPT-4 is the best for answering comparative questions. All used data, code, and evaluation results are publicly available\footnote{\url{this https URL}}. 

**Abstract (ZH)**: 在本文中，我们旨在解决自动比较问答中遇到的问题。为此，我们提出了一种评估框架，用以评估比较问答总结的质量。我们制定了15项标准来评估使用手工标注和6种大型语言模型以及两个比较问答数据集创建的比较答案的质量。我们在不同设置下使用了几种大型语言模型和手工标注进行了测试，并展示了两种评估方法的有效性。我们的结果显示，Llama-3 70B Instruct 模型在总结评估中表现最优，而GPT-4在回答比较问题方面表现最佳。所有使用的数据、代码和评估结果均已公开\footnote{\url{这个链接}}。 

---
# Enhancing Smart Environments with Context-Aware Chatbots using Large Language Models 

**Title (ZH)**: 使用大规模语言模型增强基于上下文的聊天机器人的智能环境 

**Authors**: Aurora Polo-Rodríguez, Laura Fiorini, Erika Rovini, Filippo Cavallo, Javier Medina-Quero  

**Link**: [PDF](https://arxiv.org/pdf/2502.14469)  

**Abstract**: This work presents a novel architecture for context-aware interactions within smart environments, leveraging Large Language Models (LLMs) to enhance user experiences. Our system integrates user location data obtained through UWB tags and sensor-equipped smart homes with real-time human activity recognition (HAR) to provide a comprehensive understanding of user context. This contextual information is then fed to an LLM-powered chatbot, enabling it to generate personalised interactions and recommendations based on the user's current activity and environment. This approach moves beyond traditional static chatbot interactions by dynamically adapting to the user's real-time situation. A case study conducted from a real-world dataset demonstrates the feasibility and effectiveness of our proposed architecture, showcasing its potential to create more intuitive and helpful interactions within smart homes. The results highlight the significant benefits of integrating LLM with real-time activity and location data to deliver personalised and contextually relevant user experiences. 

**Abstract (ZH)**: 本文提出了一种针对智能环境中的上下文感知交互的新型架构，利用大型语言模型（LLMs）以提升用户体验。我们的系统将通过超宽带标签（UWB tags）和传感器装备的智能家庭获取的用户位置数据与实时人体活动识别（HAR）相结合，以提供全面的用户上下文理解。随后，将此类上下文信息输入到以LLM为动力的聊天机器人中，使其能够根据用户的当前活动和环境生成个性化的交互和建议。这种方法超越了传统的静态聊天机器人交互，能够动态适应用户的实时情况。从实际数据集进行的案例研究证明了我们所提出架构的可行性和有效性，展示了其在智能家庭中创造更加直观和有用的交互的潜力。结果突显了将LLM与实时活动和位置数据结合使用以提供个性化和上下文相关的用户体验的显著优势。 

---
# Optimal word order for non-causal text generation with Large Language Models: the Spanish case 

**Title (ZH)**: 使用大型语言模型进行非因果文本生成的最佳词序：以西班牙语为例 

**Authors**: Andrea Busto-Castiñeira, Silvia García-Méndez, Francisco de Arriba-Pérez, Francisco J. González-Castaño  

**Link**: [PDF](https://arxiv.org/pdf/2502.14451)  

**Abstract**: Natural Language Generation (NLG) popularity has increased owing to the progress in Large Language Models (LLMs), with zero-shot inference capabilities. However, most neural systems utilize decoder-only causal (unidirectional) transformer models, which are effective for English but may reduce the richness of languages with less strict word order, subject omission, or different relative clause attachment preferences. This is the first work that analytically addresses optimal text generation order for non-causal language models. We present a novel Viterbi algorithm-based methodology for maximum likelihood word order estimation. We analyze the non-causal most-likelihood order probability for NLG in Spanish and, then, the probability of generating the same phrases with Spanish causal NLG. This comparative analysis reveals that causal NLG prefers English-like SVO structures. We also analyze the relationship between optimal generation order and causal left-to-right generation order using Spearman's rank correlation. Our results demonstrate that the ideal order predicted by the maximum likelihood estimator is not closely related to the causal order and may be influenced by the syntactic structure of the target sentence. 

**Abstract (ZH)**: 自然语言生成（NLG）由于大型语言模型（LLMs）的进步而日益流行，这些模型具有零样本推理能力。然而，大多数神经系统利用的是解码器仅因果（单向）变换器模型，这种模型对于英语非常有效，但可能会在语序不那么严格、有主语省略或不同的名词性从句附加偏好的语言中降低语言的丰富性。这是首次通过分析性方法解决非因果语言模型的最佳文本生成顺序的工作。我们提出了一种基于维特比算法的新颖方法，用于估计最大似然词序。我们分析了西班牙语中非因果最大似然词序的概率，并随后分析了使用西班牙语因果NLG生成相同短语的概率。这种比较分析揭示了因果NLG倾向于偏好SVO结构。我们还使用斯皮尔曼秩相关分析了最佳生成顺序与因果自左向右生成顺序之间的关系。我们的结果表明，最大似然估计器预测的理想顺序与因果顺序关系不大，且可能受目标句子语法结构的影响。 

---
# PredictaBoard: Benchmarking LLM Score Predictability 

**Title (ZH)**: PredictaBoard：评估大型语言模型得分可预测性benchmark 

**Authors**: Lorenzo Pacchiardi, Konstantinos Voudouris, Ben Slater, Fernando Martínez-Plumed, José Hernández-Orallo, Lexin Zhou, Wout Schellaert  

**Link**: [PDF](https://arxiv.org/pdf/2502.14445)  

**Abstract**: Despite possessing impressive skills, Large Language Models (LLMs) often fail unpredictably, demonstrating inconsistent success in even basic common sense reasoning tasks. This unpredictability poses a significant challenge to ensuring their safe deployment, as identifying and operating within a reliable "safe zone" is essential for mitigating risks. To address this, we present PredictaBoard, a novel collaborative benchmarking framework designed to evaluate the ability of score predictors (referred to as assessors) to anticipate LLM errors on specific task instances (i.e., prompts) from existing datasets. PredictaBoard evaluates pairs of LLMs and assessors by considering the rejection rate at different tolerance errors. As such, PredictaBoard stimulates research into developing better assessors and making LLMs more predictable, not only with a higher average performance. We conduct illustrative experiments using baseline assessors and state-of-the-art LLMs. PredictaBoard highlights the critical need to evaluate predictability alongside performance, paving the way for safer AI systems where errors are not only minimised but also anticipated and effectively mitigated. Code for our benchmark can be found at this https URL 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）掌握了令人印象深刻的能力，但它们在基本常识推理任务中往往表现出不可预测的失败，未能表现出一致的成功率。这种不可预测性对确保其安全部署构成了重大挑战，因为在操作中识别并运行在可靠的“安全区”内对减轻风险至关重要。为解决这一问题，我们提出了PredictaBoard，这是一种新颖的协作基准框架，旨在评估评估器（即评分预测器）在现有数据集上的能力，以预测LLM在特定任务实例（即提示）中的错误。PredictaBoard通过考虑在不同容差误差下的拒绝率来评估LLM对评估器配对的表现。因此，PredictaBoard激发了研究灵感，推动发展更好的评估器并使LLM更具可预测性，不仅仅是在平均性能上有更高的表现。我们使用基线评估器和最新的LLM进行直观实验。PredictaBoard强调在评估性能的同时也需要评估可预测性的关键需求，铺平了更安全AI系统的道路，使得错误不仅得以最小化，还能够被预见并有效应对。我们的基准代码可以在此链接中找到：[该链接] 

---
# An Enhancement of Jiang, Z., et al.s Compression-Based Classification Algorithm Applied to News Article Categorization 

**Title (ZH)**: 江南等人的基于压缩的分类算法在新闻文章分类中的 enhancements 

**Authors**: Sean Lester C. Benavides, Cid Antonio F. Masapol, Jonathan C. Morano, Dan Michael A. Cortez  

**Link**: [PDF](https://arxiv.org/pdf/2502.14444)  

**Abstract**: This study enhances Jiang et al.'s compression-based classification algorithm by addressing its limitations in detecting semantic similarities between text documents. The proposed improvements focus on unigram extraction and optimized concatenation, eliminating reliance on entire document compression. By compressing extracted unigrams, the algorithm mitigates sliding window limitations inherent to gzip, improving compression efficiency and similarity detection. The optimized concatenation strategy replaces direct concatenation with the union of unigrams, reducing redundancy and enhancing the accuracy of Normalized Compression Distance (NCD) calculations. Experimental results across datasets of varying sizes and complexities demonstrate an average accuracy improvement of 5.73%, with gains of up to 11% on datasets containing longer documents. Notably, these improvements are more pronounced in datasets with high-label diversity and complex text structures. The methodology achieves these results while maintaining computational efficiency, making it suitable for resource-constrained environments. This study provides a robust, scalable solution for text classification, emphasizing lightweight preprocessing techniques to achieve efficient compression, which in turn enables more accurate classification. 

**Abstract (ZH)**: 本研究通过解决江等人提出的基于压缩的分类算法在检测文本文档间语义相似性方面的局限性，增强了该算法。提出的改进集中在一元词提取和优化组合上，减少了对整个文档压缩的依赖。通过压缩提取的一元词，该算法减轻了gzip固有的滑动窗口限制，提高了压缩效率和相似性检测的准确性。优化的组合策略用一元词的并集替换直接组合，减少了冗余，提高了归一化压缩距离（NCD）计算的准确性。在不同规模和复杂程度的数据集上的实验结果显示，平均准确率提高了5.73%，在包含较长文档的数据集中，增幅可达11%。值得注意的是，这些改进在标签多样性高和文本结构复杂的数据集中更为显著。该方法在保持计算效率的同时达到了这一效果，使其适合资源受限的环境。本研究提供了一种稳健且可扩展的文本分类解决方案，强调轻量级预处理技术，从而实现有效的压缩，进而提高分类的准确性。 

---
# Natural Language Generation 

**Title (ZH)**: 自然语言生成 

**Authors**: Ehud Reiter  

**Link**: [PDF](https://arxiv.org/pdf/2502.14437)  

**Abstract**: This book provides a broad overview of Natural Language Generation (NLG), including technology, user requirements, evaluation, and real-world applications. The focus is on concepts and insights which hopefully will remain relevant for many years, not on the latest LLM innovations. It draws on decades of work by the author and others on NLG.
The book has the following chapters: Introduction to NLG; Rule-Based NLG; Machine Learning and Neural NLG; Requirements; Evaluation; Safety, Maintenance, and Testing; and Applications. All chapters include examples and anecdotes from the author's personal experiences, and end with a Further Reading section.
The book should be especially useful to people working on applied NLG, including NLG researchers, people in other fields who want to use NLG, and commercial developers. It will not however be useful to people who want to understand the latest LLM technology.
There is a companion site with more information at this https URL 

**Abstract (ZH)**: 这本书提供了自然语言生成（NLG）的广泛概述，包括技术、用户需求、评价和实际应用场景。重点关注的是一些有望长期保持相关性的概念和见解，而非最新的大语言模型（LLM）创新。本书借鉴了作者及其合作者数十年来在NLG领域的研究成果。

本书包含以下章节：自然语言生成简介；基于规则的NLG；机器学习和神经网络驱动的NLG；需求分析；评价方法；安全、维护与测试；以及应用。每一章都包含了作者个人经验中的实例和趣闻，并在章末附有进一步阅读的参考资料。

本书特别适合从事应用NLG研究的人士，包括NLG研究人员、希望应用NLG的其他领域人士以及商业开发者。然而，本书可能并不适合那些希望了解最新LLM技术的读者。

配有更多信息的相关网站：[此处填写网址] 

---
# Early-Exit and Instant Confidence Translation Quality Estimation 

**Title (ZH)**: 早期退出和即时信心翻译质量估计 

**Authors**: Vilém Zouhar, Maike Züfle, Beni Egressy, Julius Cheng, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2502.14429)  

**Abstract**: Quality estimation is omnipresent in machine translation, for both evaluation and generation. Unfortunately, quality estimation models are often opaque and computationally expensive, making them impractical to be part of large-scale pipelines. In this work, we tackle two connected challenges: (1) reducing the cost of quality estimation at scale, and (2) developing an inexpensive uncertainty estimation method for quality estimation. To address the latter, we introduce Instant Confidence COMET, an uncertainty-aware quality estimation model that matches the performance of previous approaches at a fraction of their costs. We extend this to Early-Exit COMET, a quality estimation model that can compute quality scores and associated confidences already at early model layers, allowing us to early-exit computations and reduce evaluation costs. We also apply our model to machine translation reranking. We combine Early-Exit COMET with an upper confidence bound bandit algorithm to find the best candidate from a large pool without having to run the full evaluation model on all candidates. In both cases (evaluation and reranking) our methods reduce the required compute by 50% with very little degradation in performance. 

**Abstract (ZH)**: 质量评估在机器翻译中无处不在，无论是评价还是生成。不幸的是，质量评估模型往往是不透明的，并且计算上较为昂贵，这使得它们不适合大规模管线。本文我们解决了两个相关的挑战：（1）大规模下减少质量评估的成本，以及（2）开发一种低成本的不确定性评估方法用于质量评估。为了解决后者，我们引入了Instant Confidence COMET，这是一种具有不确定性感知的质量评估模型，其性能仅需传统方法的一小部分成本即可达到。我们进一步将其扩展为Early-Exit COMET，这是一种可以在早期模型层就计算出质量分数及其相关置信度的质量评估模型，从而使我们可以提前退出计算，并降低评估成本。我们还将我们的模型应用于机器翻译重排序。我们结合使用Early-Exit COMET和上置信边界（Upper Confidence Bound, UCB）多臂老虎机算法，可以在不需要对所有候选进行完整评估模型运行的情况下，从大量候选中找出最优候选。在两种情况下（评估和重排序），我们的方法将所需的计算量减少50%，并且性能降耗非常小。 

---
# Token-Level Density-Based Uncertainty Quantification Methods for Eliciting Truthfulness of Large Language Models 

**Title (ZH)**: 面向大型语言模型真实性验证的词级密度基不确定性量化方法 

**Authors**: Artem Vazhentsev, Lyudmila Rvanova, Ivan Lazichny, Alexander Panchenko, Maxim Panov, Timothy Baldwin, Artem Shelmanov  

**Link**: [PDF](https://arxiv.org/pdf/2502.14427)  

**Abstract**: Uncertainty quantification (UQ) is a prominent approach for eliciting truthful answers from large language models (LLMs). To date, information-based and consistency-based UQ have been the dominant UQ methods for text generation via LLMs. Density-based methods, despite being very effective for UQ in text classification with encoder-based models, have not been very successful with generative LLMs. In this work, we adapt Mahalanobis Distance (MD) - a well-established UQ technique in classification tasks - for text generation and introduce a new supervised UQ method. Our method extracts token embeddings from multiple layers of LLMs, computes MD scores for each token, and uses linear regression trained on these features to provide robust uncertainty scores. Through extensive experiments on eleven datasets, we demonstrate that our approach substantially improves over existing UQ methods, providing accurate and computationally efficient uncertainty scores for both sequence-level selective generation and claim-level fact-checking tasks. Our method also exhibits strong generalization to out-of-domain data, making it suitable for a wide range of LLM-based applications. 

**Abstract (ZH)**: 不确定性量化（UQ）是一种从大语言模型（LLM）中获取真实答案的重要方法。到目前为止，基于信息的方法和基于一致性的方法是主要用于通过LLM生成文本的UQ方法中的主导方法。尽管基于密度的方法在使用编码器模型的文本分类任务中表现非常有效，但在生成性LLM中并不非常成功。在本研究中，我们借鉴了分类任务中广泛应用于不确定性量化中的马哈拉诺比斯距离（MD），将其应用于文本生成，并引入了一种新的监督不确定性量化方法。该方法从LLM的多个层中提取标记嵌入，为每个标记计算MD得分，并使用基于这些特征训练的线性回归模型提供稳健的不确定性评分。通过在十一个数据集上的广泛实验，我们证明了我们的方法在现有UQ方法的基础上有了显著的提升，能够为序列级选择性生成和断言级事实检查任务提供准确且计算高效的不确定性评分。此外，我们的方法还表现出对领域外数据的强大泛化能力，使其适用于多种基于LLM的应用。 

---
# A Survey on Data Contamination for Large Language Models 

**Title (ZH)**: 大型语言模型中的数据污染综述 

**Authors**: Yuxing Cheng, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14425)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have demonstrated significant progress in various areas, such as text generation and code synthesis. However, the reliability of performance evaluation has come under scrutiny due to data contamination-the unintended overlap between training and test datasets. This overlap has the potential to artificially inflate model performance, as LLMs are typically trained on extensive datasets scraped from publicly available sources. These datasets often inadvertently overlap with the benchmarks used for evaluation, leading to an overestimation of the models' true generalization capabilities. In this paper, we first examine the definition and impacts of data contamination. Secondly, we review methods for contamination-free evaluation, focusing on three strategies: data updating-based methods, data rewriting-based methods, and prevention-based methods. Specifically, we highlight dynamic benchmarks and LLM-driven evaluation methods. Finally, we categorize contamination detecting methods based on model information dependency: white-Box, gray-Box, and black-Box detection approaches. Our survey highlights the requirements for more rigorous evaluation protocols and proposes future directions for addressing data contamination challenges. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在文本生成和代码合成等多个领域取得了显著进展。然而，性能评估的可靠性由于数据污染（即训练集和测试集之间的意外重叠）问题而受到质疑。这种重叠有可能人为地夸大了模型的性能，因为LLMs通常是从广泛公开可用的数据源中大量抓取的训练数据进行训练。这些数据集常常无意中与用于评估的基准数据集重叠，从而高估了模型的真实外推能力。本文首先探讨数据污染的定义及其影响，然后回顾无污染评估的方法，重点讨论三种策略：基于数据更新的方法、基于数据重写的方法和预防性方法。特别地，我们强调动态基准和LLM驱动的评估方法。最后，我们根据模型信息依赖性将污染检测方法分类为白盒、灰盒和黑盒检测方法。本文的综述强调了更严格评估协议的必要性，并提出了应对数据污染挑战的未来方向。 

---
# Unstructured Evidence Attribution for Long Context Query Focused Summarization 

**Title (ZH)**: 长上下文查询导向的无结构证据归因摘要 

**Authors**: Dustin Wright, Zain Muhammad Mujahid, Lu Wang, Isabelle Augenstein, David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2502.14409)  

**Abstract**: Large language models (LLMs) are capable of generating coherent summaries from very long contexts given a user query. Extracting and properly citing evidence spans could help improve the transparency and reliability of these summaries. At the same time, LLMs suffer from positional biases in terms of which information they understand and attend to, which could affect evidence citation. Whereas previous work has focused on evidence citation with predefined levels of granularity (e.g. sentence, paragraph, document, etc.), we propose the task of long-context query focused summarization with unstructured evidence citation. We show how existing systems struggle to generate and properly cite unstructured evidence from their context, and that evidence tends to be "lost-in-the-middle". To help mitigate this, we create the Summaries with Unstructured Evidence Text dataset (SUnsET), a synthetic dataset generated using a novel domain-agnostic pipeline which can be used as supervision to adapt LLMs to this task. We demonstrate across 5 LLMs of different sizes and 4 datasets with varying document types and lengths that LLMs adapted with SUnsET data generate more relevant and factually consistent evidence than their base models, extract evidence from more diverse locations in their context, and can generate more relevant and consistent summaries. 

**Abstract (ZH)**: 大型语言模型（LLMs）能够在接收到用户查询后，从非常长的上下文中生成连贯的摘要。从上下文中提取和恰当地引用证据片段有助于提高这些摘要的透明度和可靠性。同时，LLMs在理解并关注信息时存在位置偏见，这可能会影响到证据引用。虽然以前的工作主要集中在使用预定义粒度（如句子、段落、文档等）的证据引用上，但本文提出了一个基于未结构化证据引用的长时间上下文查询重点摘要任务。我们展示了现有系统在生成和恰当地引用未结构化证据方面的困难，表明证据往往会“在中间丢失”。为了缓解这一问题，我们构建了Summaries with Unstructured Evidence Text数据集（SUnsET），这是一个使用新颖且领域无关的生成管道创建的合成数据集，可以用作监督，以使LLMs适应这一任务。我们通过5种不同规模的LLMs和4个不同文档类型和长度的数据集，展示了使用SUnsET数据改编后的LLMs生成的证据更具相关性和事实一致性，从上下文中提取的证据来源更多样，并能生成更具相关性和一致性的摘要。 

---
# Enhancing Portuguese Variety Identification with Cross-Domain Approaches 

**Title (ZH)**: 使用跨域方法增强葡萄牙语变体识别 

**Authors**: Hugo Sousa, Rúben Almeida, Purificação Silvano, Inês Cantante, Ricardo Campos, Alípio Jorge  

**Link**: [PDF](https://arxiv.org/pdf/2502.14394)  

**Abstract**: Recent advances in natural language processing have raised expectations for generative models to produce coherent text across diverse language varieties. In the particular case of the Portuguese language, the predominance of Brazilian Portuguese corpora online introduces linguistic biases in these models, limiting their applicability outside of Brazil. To address this gap and promote the creation of European Portuguese resources, we developed a cross-domain language variety identifier (LVI) to discriminate between European and Brazilian Portuguese. Motivated by the findings of our literature review, we compiled the PtBrVarId corpus, a cross-domain LVI dataset, and study the effectiveness of transformer-based LVI classifiers for cross-domain scenarios. Although this research focuses on two Portuguese varieties, our contribution can be extended to other varieties and languages. We open source the code, corpus, and models to foster further research in this task. 

**Abstract (ZH)**: 近年来自然语言处理的进展提高了生成模型跨多种语言变体生成连贯文本的期望。在葡萄牙语特别情况下，大量在线的巴西葡萄牙语语料库引入了语言偏差，限制了这些模型的应用范围仅限于巴西。为了解决这一差距并促进创建欧洲葡萄牙语资源，我们开发了一个跨领域语言变体识别器（LVI）来区分欧洲葡萄牙语和巴西葡萄牙语。基于我们文献回顾的研究发现，我们编译了PtBrVarId语料库——一个跨领域LVI数据集，并研究了基于变压器的LVI分类器在跨领域情景中的有效性。尽管本研究重点关注两种葡萄牙语变体，但我们的贡献可以扩展到其他变体和语言。我们开源了代码、语料库和模型，以促进对该任务的进一步研究。 

---
# Leveraging Small LLMs for Argument Mining in Education: Argument Component Identification, Classification, and Assessment 

**Title (ZH)**: 利用小型语言模型在教育领域进行论据挖掘：论据组件识别、分类与评估 

**Authors**: Lucile Favero, Juan Antonio Pérez-Ortiz, Tanja Käser, Nuria Oliver  

**Link**: [PDF](https://arxiv.org/pdf/2502.14389)  

**Abstract**: Argument mining algorithms analyze the argumentative structure of essays, making them a valuable tool for enhancing education by providing targeted feedback on the students' argumentation skills. While current methods often use encoder or encoder-decoder deep learning architectures, decoder-only models remain largely unexplored, offering a promising research direction.
This paper proposes leveraging open-source, small Large Language Models (LLMs) for argument mining through few-shot prompting and fine-tuning. These models' small size and open-source nature ensure accessibility, privacy, and computational efficiency, enabling schools and educators to adopt and deploy them locally. Specifically, we perform three tasks: segmentation of student essays into arguments, classification of the arguments by type, and assessment of their quality. We empirically evaluate the models on the Feedback Prize - Predicting Effective Arguments dataset of grade 6-12 students essays and demonstrate how fine-tuned small LLMs outperform baseline methods in segmenting the essays and determining the argument types while few-shot prompting yields comparable performance to that of the baselines in assessing quality. This work highlights the educational potential of small, open-source LLMs to provide real-time, personalized feedback, enhancing independent learning and writing skills while ensuring low computational cost and privacy. 

**Abstract (ZH)**: 以下是经过学术规范翻译后的中文内容：

论文字义提取算法通过分析作文的论据结构，成为提高教育质量的重要工具，通过针对性地反馈学生的论据能力为学生提供帮助。当前的方法往往采用编码器或编码器-解码器深度学习架构，而完全解码模型尚未得到充分探索，为未来的研究提供了广阔的前景。

本文提出了利用开源的小型大语言模型（LLMs）进行论文字义提取的方法，通过少量示例的提示及微调实现。小型LLM的体积小、开源的特性确保了其可访问性、隐私保护和计算效率，使得学校和教育者能够将其部署在本地环境中。具体而言，本文执行了三项任务：将学生的作文拆分为论据、按类型对论据进行分类，并评估其质量。通过在6-12年级学生的作文数据集（Feedback Prize - Predicting Effective Arguments）上进行实证评估，我们发现微调的小型LLM在拆分作文和确定论据类型方面优于基线方法，而少量示例的提示在评估质量方面与基线方法具有可比性。本研究表明，小型开源LLM能够在实时、个性化反馈方面发挥教育潜力，促进自主学习和写作能力的发展，同时确保低计算成本和隐私保护。

---

这种翻译方式保持了原文的学术风格和结构，并将其转换成了中文，符合学术论文的写作规范。 

---
# Tradutor: Building a Variety Specific Translation Model 

**Title (ZH)**: 翻译如下，符合学术规范：

构建特定领域的翻译模型 

**Authors**: Hugo Sousa, Satya Almasian, Ricardo Campos, Alípio Jorge  

**Link**: [PDF](https://arxiv.org/pdf/2502.14385)  

**Abstract**: Language models have become foundational to many widely used systems. However, these seemingly advantageous models are double-edged swords. While they excel in tasks related to resource-rich languages like English, they often lose the fine nuances of language forms, dialects, and varieties that are inherent to languages spoken in multiple regions of the world. Languages like European Portuguese are neglected in favor of their more popular counterpart, Brazilian Portuguese, leading to suboptimal performance in various linguistic tasks. To address this gap, we introduce the first open-source translation model specifically tailored for European Portuguese, along with a novel dataset specifically designed for this task. Results from automatic evaluations on two benchmark datasets demonstrate that our best model surpasses existing open-source translation systems for Portuguese and approaches the performance of industry-leading closed-source systems for European Portuguese. By making our dataset, models, and code publicly available, we aim to support and encourage further research, fostering advancements in the representation of underrepresented language varieties. 

**Abstract (ZH)**: 语言模型已经成为许多广泛使用的系统的基础。然而，这些看似具有优势的模型实际上是双刃剑。尽管它们在资源丰富的语言（如英语）相关的任务中表现出色，但它们往往在反映不同地区使用的语言中的细微差别、方言和变体方面表现较差。欧洲葡萄牙语在这种情况下被忽视，而其更为流行的巴西葡萄牙语则受到了更多关注，导致在各种语言任务中的表现不理想。为解决这一问题，我们介绍了第一个专门为欧洲葡萄牙语设计的开源翻译模型，以及一个专为此任务设计的新数据集。来自两个基准数据集的自动评估结果显示，我们的最佳模型超过了现有的开源葡萄牙语翻译系统，并接近于行业领先闭源系统在欧洲葡萄牙语上的表现。通过公开我们的数据集、模型和代码，我们旨在支持和鼓励进一步研究，促进对未充分代表的语言变体的表示方面的进展。 

---
# Rumor Detection by Multi-task Suffix Learning based on Time-series Dual Sentiments 

**Title (ZH)**: 基于时间序列双情绪的多任务后缀学习谣言检测 

**Authors**: Zhiwei Liu, Kailai Yang, Eduard Hovy, Sophia Ananiadou  

**Link**: [PDF](https://arxiv.org/pdf/2502.14383)  

**Abstract**: The widespread dissemination of rumors on social media has a significant impact on people's lives, potentially leading to public panic and fear. Rumors often evoke specific sentiments, resonating with readers and prompting sharing. To effectively detect and track rumors, it is essential to observe the fine-grained sentiments of both source and response message pairs as the rumor evolves over time. However, current rumor detection methods fail to account for this aspect. In this paper, we propose MSuf, the first multi-task suffix learning framework for rumor detection and tracking using time series dual (coupled) sentiments. MSuf includes three modules: (1) an LLM to extract sentiment intensity features and sort them chronologically; (2) a module that fuses the sorted sentiment features with their source text word embeddings to obtain an aligned embedding; (3) two hard prompts are combined with the aligned vector to perform rumor detection and sentiment analysis using one frozen LLM. MSuf effectively enhances the performance of LLMs for rumor detection with only minimal parameter fine-tuning. Evaluating MSuf on four rumor detection benchmarks, we find significant improvements compared to other emotion-based methods. 

**Abstract (ZH)**: 社交媒体上谣言的广泛传播对人们的生活产生了显著影响，可能导致公众恐慌和恐惧。谣言往往引发特定的情感，与读者产生共鸣，并促使他们分享。为了有效地检测和追踪谣言，在谣言演变过程中观察来源和响应消息对的情感细微变化至关重要。然而，当前的谣言检测方法未能考虑这一点。在这篇文章中，我们提出了MSuf，这是一种首次利用时间序列双（耦合）情感进行谣言检测和追踪的多任务后缀学习框架。MSuf包括三个模块：(1) 一个大型语言模型用于提取情感强度特征并按时间顺序排序；(2) 一个模块将排序后的情感特征与它们的源文本词嵌入结合，以获得对齐的嵌入；(3) 将两个硬提示与对齐的向量结合，使用一个冻结的大型语言模型进行谣言检测和情感分析。MSuf仅通过少量参数微调便有效提升了LLM在谣言检测中的性能。在四个谣言检测基准上评估MSuf，我们发现与基于情绪的方法相比取得了显著改进。 

---
# Affinity and Diversity: A Unified Metric for Demonstration Selection via Internal Representations 

**Title (ZH)**: 亲和度与多样性：一种通过内部表示统一衡量示范选择的指标 

**Authors**: Mariko Kato, Hakaze Cho, Yoshihiro Sakai, Naoya Inoue  

**Link**: [PDF](https://arxiv.org/pdf/2502.14380)  

**Abstract**: The performance of In-Context Learning (ICL) is highly sensitive to the selected demonstrations. Existing approaches to demonstration selection optimize different objectives, yielding inconsistent results. To address this, we propose a unified metric--affinity and diversity--that leverages ICL model's internal representations. Our experiments show that both affinity and diversity strongly correlate with test accuracies, indicating their effectiveness for demonstration selection. Moreover, we show that our proposed metrics align well with various previous works to unify the inconsistency. 

**Abstract (ZH)**: נה fattica 翻译如下，符合学术规范：

基于上下文学习（ICL）的表现高度依赖于所选示例。现有的示例选择方法优化不同的目标，导致结果不一致。为了解决这一问题，我们提出了一种统一的度量标准——亲和力和多样性，该标准利用了ICL模型的内部表示。我们的实验表明，亲和力和多样性与测试准确性有很强的相关性，表明它们在示例选择中的有效性。此外，我们表明，我们提出的度量标准与各种先前工作很好地对齐，以统一不一致的情况。 

---
# A Similarity Paradigm Through Textual Regularization Without Forgetting 

**Title (ZH)**: 一种基于文本正则化的相似性范式不失忆性 

**Authors**: Fangming Cui, Jan Fong, Rongfei Zeng, Xinmei Tian, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14376)  

**Abstract**: Prompt learning has emerged as a promising method for adapting pre-trained visual-language models (VLMs) to a range of downstream tasks. While optimizing the context can be effective for improving performance on specific tasks, it can often lead to poor generalization performance on unseen classes or datasets sampled from different distributions. It may be attributed to the fact that textual prompts tend to overfit downstream data distributions, leading to the forgetting of generalized knowledge derived from hand-crafted prompts. In this paper, we propose a novel method called Similarity Paradigm with Textual Regularization (SPTR) for prompt learning without forgetting. SPTR is a two-pronged design based on hand-crafted prompts that is an inseparable framework. 1) To avoid forgetting general textual knowledge, we introduce the optimal transport as a textual regularization to finely ensure approximation with hand-crafted features and tuning textual features. 2) In order to continuously unleash the general ability of multiple hand-crafted prompts, we propose a similarity paradigm for natural alignment score and adversarial alignment score to improve model robustness for generalization. Both modules share a common objective in addressing generalization issues, aiming to maximize the generalization capability derived from multiple hand-crafted prompts. Four representative tasks (i.e., non-generalization few-shot learning, base-to-novel generalization, cross-dataset generalization, domain generalization) across 11 datasets demonstrate that SPTR outperforms existing prompt learning methods. 

**Abstract (ZH)**: 提示学习作为一种有前景的方法，已经被广泛应用到使预训练的视觉-语言模型（VLMs）适应各种下游任务中。虽然优化上下文可以在特定任务上取得良好的效果，但这也可能会导致在未见过的类或不同分布的数据样本中泛化性能较差。这可能是因为文本提示倾向于过度拟合下游数据分布，从而导致从手工设计提示中获得的泛化知识的遗忘。在本文中，我们提出了一种名为文本正则化相似性范式（SPTR，Similarity Paradigm with Textual Regularization）的新方法，用于在不遗忘的情况下进行提示学习。SPTR 是基于手工设计提示的一种双重设计，是一个不可分割的框架。1) 为避免遗忘通用的文本知识，我们引入最优传输作为文本正则化手段，以精细地确保对手工设计特征的逼近，并调整文本特征。2) 为了持续释放多种手工设计提示的通用能力，我们提出了一种相似性范式，用于自然对齐得分和对抗对齐得分，以提高模型在泛化方面的鲁棒性。这两个模块在解决泛化问题的目标上共享一个共同点，旨在最大化从多种手工设计提示中获得的泛化能力。在涵盖11个数据集的四种代表性任务（即泛化不足的少样本学习、基本到新颖类别的泛化、跨数据集泛化、领域泛化）中，SPTR 优于现有的提示学习方法。 

---
# Entropy-UID: A Method for Optimizing Information Density 

**Title (ZH)**: 熵-UID：一种优化信息密度的方法 

**Authors**: Xinpeng Shou  

**Link**: [PDF](https://arxiv.org/pdf/2502.14366)  

**Abstract**: Balanced and efficient information flow is essential for optimizing language generation models. In this work, we propose Entropy-UID, a new token selection method that balances entropy and Uniform Information Density (UID) principles for enhanced efficiency of text generation. Our approach adaptively adjusts token selection by jointly minimizing entropy and surprisal, promoting more even information distribution across generated sequences. Theoretical validation demonstrates that Entropy-UID optimally reduces information spikes while maintaining fluency and coherence. The method has been evulated using information-theoretic metrics on multiple benchmark datasets, including WikiText-2, OpenWebText, and WMT. Experimental results show that Entropy-UID achieves lower surprisal and entropy variance compared to standard GPT-2 and alternative heuristics, leading to more balanced and human-like text generation. Our findings point towards the potential of leveraging information-theoretic constraints to refine token selection strategies in autoregressive language models. 

**Abstract (ZH)**: 平衡且高效的信道信息流通是优化语言生成模型的关键。本文提出了一种新的标记选择方法——Entropy-UID，它结合了熵和均匀信息密度（UID）原则，以提高文本生成的效率。我们的方法通过同时最小化熵和意外程度，适应性调整标记选择，从而促进生成序列中信息分布更加均匀。理论验证表明，Entropy-UID 最优地降低了信息尖峰现象，同时保持了流畅性和连贯性。该方法在多个基准数据集（包括 WikiText-2、OpenWebText 和 WMT）上使用信息论度量进行了评估。实验结果表明，与标准的 GPT-2 和其他启发式方法相比，Entropy-UID 产生更低的意外程度和熵变异性，从而生成了更加平衡和类人的文本。我们的研究结果表明，利用信息论约束来优化自回归语言模型中的标记选择策略具有潜力。 

---
# Triangulating LLM Progress through Benchmarks, Games, and Cognitive Tests 

**Title (ZH)**: 通过基准测试、游戏和认知测试 triangulate 大规模语言模型的进步 

**Authors**: Filippo Momentè, Alessandro Suglia, Mario Giulianelli, Ambra Ferrari, Alexander Koller, Oliver Lemon, David Schlangen, Raquel Fernández, Raffaella Bernardi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14359)  

**Abstract**: We examine three evaluation paradigms: large question-answering benchmarks (e.g., MMLU and BBH), interactive games (e.g., Signalling Games or Taboo), and cognitive tests (e.g., for working memory or theory of mind). First, we investigate which of the former two-benchmarks or games-is most effective at discriminating LLMs of varying quality. Then, inspired by human cognitive assessments, we compile a suite of targeted tests that measure cognitive abilities deemed essential for effective language use, and we investigate their correlation with model performance in benchmarks and games. Our analyses reveal that interactive games are superior to standard benchmarks in discriminating models. Causal and logical reasoning correlate with both static and interactive tests, while differences emerge regarding core executive functions and social/emotional skills, which correlate more with games. We advocate the development of new interactive benchmarks and targeted cognitive tasks inspired by assessing human abilities but designed specifically for LLMs. 

**Abstract (ZH)**: 我们考察了三种评估范式：大规模问答基准（例如MMLU和BBH）、互动游戏（例如信号博弈或禁忌词游戏）以及认知测试（例如工作记忆或理论思维测试）。首先，我们研究在区分不同质量的语言模型方面，哪种基准或游戏更为有效。接着，在受人类认知评估启发的基础上，我们编译了一系列针对性强的测验，这些测验衡量了有效语言使用所必需的认知能力，并探讨了这些测验与基准和游戏中模型性能的相关性。我们的分析表明，互动游戏在区分模型方面优于标准基准。因果推理和逻辑推理与静态和互动测验均相关，但在核心执行功能和社交/情感技能方面则呈现出不同，后者与游戏表现的相关性更强。我们建议开发新的互动基准和特定设计用于语言模型的认知任务，这些任务灵感源于评估人类能力。 

---
# Full-Step-DPO: Self-Supervised Preference Optimization with Step-wise Rewards for Mathematical Reasoning 

**Title (ZH)**: 全程步进偏好优化：基于分步骤奖励的自我监督偏好优化方法在数学推理中的应用 

**Authors**: Huimin Xu, Xin Mao, Feng-Lin Li, Xiaobao Wu, Wang Chen, Wei Zhang, Anh Tuan Luu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14356)  

**Abstract**: Direct Preference Optimization (DPO) often struggles with long-chain mathematical reasoning. Existing approaches, such as Step-DPO, typically improve this by focusing on the first erroneous step in the reasoning chain. However, they overlook all other steps and rely heavily on humans or GPT-4 to identify erroneous steps. To address these issues, we propose Full-Step-DPO, a novel DPO framework tailored for mathematical reasoning. Instead of optimizing only the first erroneous step, it leverages step-wise rewards from the entire reasoning chain. This is achieved by training a self-supervised process reward model, which automatically scores each step, providing rewards while avoiding reliance on external signals. Furthermore, we introduce a novel step-wise DPO loss, which dynamically updates gradients based on these step-wise rewards. This endows stronger reasoning capabilities to language models. Extensive evaluations on both in-domain and out-of-domain mathematical reasoning benchmarks across various base language models, demonstrate that Full-Step-DPO achieves superior performance compared to state-of-the-art baselines. 

**Abstract (ZH)**: 直接偏好优化（DPO）在处理长链条数学推理时常常面临困难。现有方法，如Step-DPO，通常通过集中优化推理链中的第一个错误步骤来改进这一问题。然而，这些方法忽略了其他所有步骤，并且很大程度上依赖于人工或GPT-4来识别错误步骤。为了解决这些问题，我们提出了一种名为Full-Step-DPO的新型DPO框架，专门用于数学推理。该框架不仅优化第一个错误步骤，还利用了整个推理链中的逐步奖励。这通过训练一个自我监督的过程奖励模型实现，该模型可以自动为每个步骤评分，并提供奖励而不依赖于外部信号。此外，我们引入了一种新颖的逐步DPO损失函数，该函数根据逐步奖励动态更新梯度。这一方法赋予了语言模型更强的推理能力。广泛的内部领域和外部领域数学推理基准测试结果表明，Full-Step-DPO相比最先进的基线方法具有优越的性能。 

---
# SR-LLM: Rethinking the Structured Representation in Large Language Model 

**Title (ZH)**: SR-LLM: 重新审视大型语言模型中的结构化表示 

**Authors**: Jiahuan Zhang, Tianheng Wang, Hanqing Wu, Ziyi Huang, Yulong Wu, Dongbai Chen, Linfeng Song, Yue Zhang, Guozheng Rao, Kaicheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14352)  

**Abstract**: Structured representations, exemplified by Abstract Meaning Representation (AMR), have long been pivotal in computational linguistics. However, their role remains ambiguous in the Large Language Models (LLMs) era. Initial attempts to integrate structured representation into LLMs via a zero-shot setting yielded inferior performance. We hypothesize that such a decline stems from the structure information being passed into LLMs in a code format unfamiliar to LLMs' training corpora. Consequently, we propose SR-LLM, an innovative framework with two settings to explore a superior way of integrating structured representation with LLMs from training-free and training-dependent perspectives. The former integrates structural information through natural language descriptions in LLM prompts, whereas its counterpart augments the model's inference capability through fine-tuning on linguistically described structured representations. Performance improvements were observed in widely downstream datasets, with particularly notable gains of 3.17% and 12.38% in PAWS. To the best of our knowledge, this work represents the pioneering demonstration that leveraging structural representations can substantially enhance LLMs' inference capability. We hope that our work sheds light and encourages future research to enhance the reasoning and interoperability of LLMs by structure data. 

**Abstract (ZH)**: 结构化表示，以抽象语义表示（AMR）为例，在计算语言学中一直起着关键作用。然而，在大型语言模型（LLMs）时代，它们的作用仍不明确。早期尝试通过零样本设置将结构化表示整合到LLMs中，性能表现较差。我们假设这种下降是由结构信息以LLMs的训练语料库不熟悉的编码格式传递引起的。因此，我们提出SR-LLM，这是一种创新框架，包含两种设置，旨在从无训练和依赖训练的角度探索将结构化表示与LLMs整合的更优方式。前者通过LLMs提示中的自然语言描述来整合结构信息，而后者则通过在语言描述的结构化表示上进行微调来增强模型的推理能力。在广泛的应用下游数据集上观察到了性能改进，特别是在PAWS数据集上的改进尤为显著，达到3.17%和12.38%。据我们所知，这项工作展示了利用结构化表示可以显著提升LLMs的推理能力的首次示例。我们希望我们的工作能为未来研究提供启示，通过结构化数据提高LLMs的推理能力和互操性。 

---
# Earlier Tokens Contribute More: Learning Direct Preference Optimization From Temporal Decay Perspective 

**Title (ZH)**: 先前的令牌贡献更大：从时间衰减视角学习直接偏好优化 

**Authors**: Ruichen Shao, Bei Li, Gangao Liu, Yang Chen, Xiang Zhou, Jingang Wang, Xunliang Cai, Peng Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.14340)  

**Abstract**: Direct Preference Optimization (DPO) has gained attention as an efficient alternative to reinforcement learning from human feedback (RLHF) for aligning large language models (LLMs) with human preferences. Despite its advantages, DPO suffers from a length bias, generating responses longer than those from the reference model. Existing solutions like SimPO and SamPO address this issue but uniformly treat the contribution of rewards across sequences, overlooking temporal dynamics. To this end, we propose an enhanced preference optimization method that incorporates a temporal decay factor controlled by a gamma parameter. This dynamic weighting mechanism adjusts the influence of each reward based on its position in the sequence, prioritizing earlier tokens that are more critical for alignment. By adaptively focusing on more relevant feedback, our approach mitigates overfitting to less pertinent data and remains responsive to evolving human preferences. Experimental results on several benchmarks show that our approach consistently outperforms vanilla DPO by 5.9-8.8 points on AlpacaEval 2 and 3.3-9.7 points on Arena-Hard across different model architectures and sizes. Furthermore, additional experiments on mathematical and reasoning benchmarks (MMLU, GSM8K, and MATH) confirm that our method enhances performance without compromising general capabilities. Our codebase would be available at \url{this https URL}. 

**Abstract (ZH)**: 直接偏好优化（DPO）已经成为一种高效替代方法，用于利用人类反馈（RLHF）对大规模语言模型（LLMs）进行人机对齐。尽管DPO具有优势，但它存在长度偏差的问题，生成的响应比参考模型的响应更长。现有的解决方法如SimPO和SamPO解决了这个问题，但它们以统一的方式处理序列中奖励的贡献，忽略了时间动态性。为了解决这一问题，我们提出了一种增强的偏好优化方法，该方法引入了一个由伽马参数控制的时间衰减因子。这种动态加权机制根据奖励在序列中的位置调整其影响力，优先考虑对对齐更为关键的早期令牌。通过有选择地关注更为相关的反馈，这种方法减轻了对不相关数据的过度拟合，并且能够对不断变化的人类偏好做出反应。在多个基准测试上进行的实验结果显示，与传统的DPO相比，我们的方法在AlpacaEval 2上提高了5.9-8.8个点，Arena-Hard上提高了3.3-9.7个点，这种改进适用于不同的模型架构和规模。此外，在数学和推理基准测试（MMLU、GSM8K和MATH）上的进一步实验也证实了，我们的方法在提高性能的同时并未牺牲一般能力。我们的代码库将可以在 \url{this https URL} 获取。 

---
# English Please: Evaluating Machine Translation for Multilingual Bug Reports 

**Title (ZH)**: 请用英文：评估机器翻译在多语言bug报告中的效果 

**Authors**: Avinash Patil, Aryan Jadon  

**Link**: [PDF](https://arxiv.org/pdf/2502.14338)  

**Abstract**: Accurate translation of bug reports is critical for efficient collaboration in global software development. In this study, we conduct the first comprehensive evaluation of machine translation (MT) performance on bug reports, analyzing the capabilities of DeepL, AWS Translate, and ChatGPT using data from the Visual Studio Code GitHub repository, specifically focusing on reports labeled with the english-please tag. To thoroughly assess the accuracy and effectiveness of each system, we employ multiple machine translation metrics, including BLEU, BERTScore, COMET, METEOR, and ROUGE. Our findings indicate that DeepL consistently outperforms the other systems across most automatic metrics, demonstrating strong lexical and semantic alignment. AWS Translate performs competitively, particularly in METEOR, while ChatGPT lags in key metrics. This study underscores the importance of domain adaptation for translating technical texts and offers guidance for integrating automated translation into bug-triaging workflows. Moreover, our results establish a foundation for future research to refine machine translation solutions for specialized engineering contexts. The code and dataset for this paper are available at GitHub: this https URL. 

**Abstract (ZH)**: 准确地翻译缺陷报告对于全球软件开发中的高效协作至关重要。在本研究中，我们首次全面评估了机器翻译系统的性能，利用 Visual Studio Code 在 GitHub 仓库中的数据，特别分析了带有 english-please 标签的缺陷报告，使用 DeepL、AWS Translate 和 ChatGPT 这三种系统。为了全面评估各系统准确性与有效性，我们采用了多个机器翻译指标，包括 BLEU、BERTScore、COMET、METEOR 和 ROUGE。研究发现，DeepL 在大多数自动评估指标中表现出色，显示出较强的词法和语义一致性。AWS Translate 在 METEOR 指标上表现出竞争力，而 ChatGPT 在关键指标上表现落后。本研究强调了在翻译技术文本时进行领域适应的重要性，并为将自动化翻译集成到缺陷报告流程中提供了指导。此外，我们的结果为未来研究如何专门为工程领域优化机器翻译解决方案奠定了基础。本文的代码和数据集可在 GitHub 上获取：this https URL。 

---
# Information Types in Product Reviews 

**Title (ZH)**: 产品评论中的信息类型 

**Authors**: Ori Shapira, Yuval Piniter  

**Link**: [PDF](https://arxiv.org/pdf/2502.14335)  

**Abstract**: Information in text is communicated in a way that supports a goal for its reader. Product reviews, for example, contain opinions, tips, product descriptions, and many other types of information that provide both direct insights, as well as unexpected signals for downstream applications. We devise a typology of 24 communicative goals in sentences from the product review domain, and employ a zero-shot multi-label classifier that facilitates large-scale analyses of review data. In our experiments, we find that the combination of classes in the typology forecasts helpfulness and sentiment of reviews, while supplying explanations for these decisions. In addition, our typology enables analysis of review intent, effectiveness and rhetorical structure. Characterizing the types of information in reviews unlocks many opportunities for more effective consumption of this genre. 

**Abstract (ZH)**: 文本中的信息是以支持其读者目标的方式传达的。例如，产品评价包含观点、建议、产品描述等多种类型的信息，既提供了直接见解，也提供了下游应用中意想不到的信号。我们在这篇文献中设计了一个产品评价领域中句子的24种沟通目标类型，并采用了一种零样本多标签分类器，以促进大规模的产品评价数据分析。在我们的实验中，我们发现类型组合能够预测评论的有用性和情感，同时为这些决策提供解释。此外，我们的类型分类使我们能够分析评论的意图、效果和修辞结构。识别评价中信息的类型为更有效地消费这一文体提供了许多机会。 

---
# A Survey on Feedback-based Multi-step Reasoning for Large Language Models on Mathematics 

**Title (ZH)**: 基于反馈的多步推理综述：大规模语言模型在数学领域中的应用 

**Authors**: Ting-Ruen Wei, Haowei Liu, Xuyang Wu, Yi Fang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14333)  

**Abstract**: Recent progress in large language models (LLM) found chain-of-thought prompting strategies to improve the reasoning ability of LLMs by encouraging problem solving through multiple steps. Therefore, subsequent research aimed to integrate the multi-step reasoning process into the LLM itself through process rewards as feedback and achieved improvements over prompting strategies. Due to the cost of step-level annotation, some turn to outcome rewards as feedback. Aside from these training-based approaches, training-free techniques leverage frozen LLMs or external tools for feedback at each step to enhance the reasoning process. With the abundance of work in mathematics due to its logical nature, we present a survey of strategies utilizing feedback at the step and outcome levels to enhance multi-step math reasoning for LLMs. As multi-step reasoning emerges a crucial component in scaling LLMs, we hope to establish its foundation for easier understanding and empower further research. 

**Abstract (ZH)**: 近年来，大型语言模型（LLM）的进步发现，通过多步骤的思维链提示策略可以提高LLM的推理能力，这种方法通过引导多步问题解决来提升模型的表现。因此，后续研究试图将多步骤推理过程整合到LLM本身中，通过过程奖励作为反馈来实现对提示策略的改进。由于步骤级别注解的成本较高，一些研究转向使用结果奖励作为反馈。除了基于训练的方法之外，无需训练的技术利用冻结的LLM或外部工具在每个步骤上提供反馈，以增强推理过程。由于数学问题处理具有逻辑性，因此问题众多，我们对利用步骤级别和结果级别反馈策略提升LLM多步骤数学推理进行了综述。鉴于多步骤推理已成为扩大LLM的关键组成部分，我们希望为更易于理解奠定基础，并促进进一步的研究。 

---
# Line Goes Up? Inherent Limitations of Benchmarks for Evaluating Large Language Models 

**Title (ZH)**: 《线在上升吗？大型语言模型评估基准的固有局限性》

这个标题翻译成中文既保持了原文的意思，又符合学术规范。如果你有更多具体的段落或内容需要翻译，请提供文本，我会给出更专业的翻译。 

**Authors**: James Fodor  

**Link**: [PDF](https://arxiv.org/pdf/2502.14318)  

**Abstract**: Large language models (LLMs) regularly demonstrate new and impressive performance on a wide range of language, knowledge, and reasoning benchmarks. Such rapid progress has led many commentators to argue that LLM general cognitive capabilities have likewise rapidly improved, with the implication that such models are becoming progressively more capable on various real-world tasks. Here I summarise theoretical and empirical considerations to challenge this narrative. I argue that inherent limitations with the benchmarking paradigm, along with specific limitations of existing benchmarks, render benchmark performance highly unsuitable as a metric for generalisable competence over cognitive tasks. I also contend that alternative methods for assessing LLM capabilities, including adversarial stimuli and interpretability techniques, have shown that LLMs do not have robust competence in many language and reasoning tasks, and often fail to learn representations which facilitate generalisable inferences. I conclude that benchmark performance should not be used as a reliable indicator of general LLM cognitive capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）在语言、知识和推理等多个基准测试上的表现经常展现出新的令人印象深刻的性能。这种快速的进步已使许多评论者认为，LLMs的通用认知能力也得到了快速提升，并暗示这些模型在众多实际任务中的能力正在逐步增强。在此，我总结了一些理论与实证考虑，以反驳这一叙事。我认为，基准测试范式的固有局限性，以及现有基准的具体局限性，使得基于基准性能的通用认知能力衡量标准极为不合适。此外，我还主张使用对抗性刺激和可解释性技术等替代方法评估LLMs的能力，这些方法表明LLMs在许多语言和推理任务上并没有稳健的能力，并且往往无法学习有助于通用化推理的表示。最终，我认为不应将基准性能用作可靠的标准来衡量LLMs的一般认知能力。 

---
# ParallelComp: Parallel Long-Context Compressor for Length Extrapolation 

**Title (ZH)**: 并行压缩器ParallelComp：用于长度外推的并行长上下文压缩器 

**Authors**: Jing Xiong, Jianghan Shen, Chuanyang Zheng, Zhongwei Wan, Chenyang Zhao, Chiwun Yang, Fanghua Ye, Hongxia Yang, Lingpeng Kong, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2502.14317)  

**Abstract**: Efficiently handling long contexts is crucial for large language models (LLMs). While rotary position embeddings (RoPEs) enhance length generalization, effective length extrapolation remains challenging and often requires costly fine-tuning. In contrast, recent training-free approaches suffer from the attention sink phenomenon, leading to severe performance degradation. In this paper, we introduce ParallelComp, a novel training-free method for long-context extrapolation that extends LLMs' context length from 4K to 128K while maintaining high throughput and preserving perplexity, and integrates seamlessly with Flash Attention. Our analysis offers new insights into attention biases in parallel attention mechanisms and provides practical solutions to tackle these challenges. To mitigate the attention sink issue, we propose an attention calibration strategy that reduces biases, ensuring more stable long-range attention. Additionally, we introduce a chunk eviction strategy to efficiently manage ultra-long contexts on a single A100 80GB GPU. To further enhance efficiency, we propose a parallel KV cache eviction technique, which improves chunk throughput by 1.76x, thereby achieving a 23.50x acceleration in the prefilling stage with negligible performance loss due to attention calibration. Furthermore, ParallelComp achieves 91.17% of GPT-4's performance on long-context tasks using an 8B model trained on 8K-length context, outperforming powerful closed-source models such as Claude-2 and Kimi-Chat. 

**Abstract (ZH)**: 高效处理长上下文对大型语言模型（LLMs）至关重要。虽然旋转位置嵌入（RoPEs）提高了长度泛化能力，但有效的长度外推仍然充满挑战，并且通常需要成本高昂的微调。相比之下，最近的无训练方法受到注意力陷阱现象的影响，导致性能严重下降。在本文中，我们提出了一种名为ParallelComp的无训练方法，该方法可以将LLMs的上下文长度从4K扩大到128K，同时保持高吞吐量和困惑度，并与Flash Attention无缝集成。我们的分析提供了并行注意力机制中的注意力偏差的新见解，并提供了实际解决方案以应对这些挑战。为缓解注意力陷阱问题，我们提出了一种注意力校准策略，以减少偏差，确保更稳定的远程注意力。此外，我们引入了一种分块淘汰策略，以高效管理单个A100 80GB GPU上的超长上下文。为了进一步提高效率，我们提出了一种并行KV缓存淘汰技术，该技术通过1.76倍的分块吞吐量改进，从而在预填充阶段实现了23.50倍的加速，同时由于注意力校准造成的性能损失可忽略不计。此外，使用训练上下文为8K的8B模型，ParallelComp在长上下文任务上的性能达到了GPT-4的91.17%，超过了诸如Claude-2和Kimi-Chat等强大的封闭源模型。 

---
# Unveiling Cultural Blind Spots: Analyzing the Limitations of mLLMs in Procedural Text Comprehension 

**Title (ZH)**: 揭示文化盲区：分析mLLMs在程序性文本理解中的局限性 

**Authors**: Amir Hossein Yari, Fajri Koto  

**Link**: [PDF](https://arxiv.org/pdf/2502.14315)  

**Abstract**: Despite the impressive performance of multilingual large language models (mLLMs) in various natural language processing tasks, their ability to understand procedural texts, particularly those with culture-specific content, remains largely unexplored. Texts describing cultural procedures, including rituals, traditional craftsmanship, and social etiquette, require an inherent understanding of cultural context, presenting a significant challenge for mLLMs. In this work, we introduce CAPTex, a benchmark designed to evaluate mLLMs' ability to process and reason about culturally diverse procedural texts across multiple languages using various methodologies to assess their performance. Our findings indicate that (1) mLLMs face difficulties with culturally contextualized procedural texts, showing notable performance declines in low-resource languages, (2) model performance fluctuates across cultural domains, with some areas presenting greater difficulties, and (3) language models exhibit better performance on multiple-choice tasks within conversational frameworks compared to direct questioning. These results underscore the current limitations of mLLMs in handling culturally nuanced procedural texts and highlight the need for culturally aware benchmarks like CAPTex to enhance their adaptability and comprehension across diverse linguistic and cultural landscapes. 

**Abstract (ZH)**: 尽管多语言大型语言模型（mLLMs）在多种自然语言处理任务中表现出色，但它们对程序性文本的理解能力，尤其是具有文化特定内容的文本，仍鲜有研究。描述文化程序的文本，包括仪式、传统手工艺和社交礼仪，需要对文化背景有内在的理解，这为mLLMs带来了显著的挑战。在这项工作中，我们引入了CAPTex基准，旨在评估mLLMs在多种语言中处理和推理跨文化程序性文本的能力，通过多种方法评估其性能。我们的研究结果表明：（1）mLLMs在处理具有文化背景的程序性文本时面临困难，在低资源语言中的表现显著下降；（2）模型在不同文化领域的表现波动不定，某些领域表现出更大的困难；（3）语言模型在对话框架内的多项选择任务中表现优于直接提问。这些结果强调了mLLMs在处理文化细微差异的程序性文本方面的现有局限性，并突显了需要像CAPTex这样的文化意识基准来增强其在多元语言和文化环境中的适应性和理解能力。 

---
# MedHallu: A Comprehensive Benchmark for Detecting Medical Hallucinations in Large Language Models 

**Title (ZH)**: MedHallu：大规模语言模型中医学幻觉检测的综合基准 

**Authors**: Shrey Pandit, Jiawei Xu, Junyuan Hong, Zhangyang Wang, Tianlong Chen, Kaidi Xu, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2502.14302)  

**Abstract**: Advancements in Large Language Models (LLMs) and their increasing use in medical question-answering necessitate rigorous evaluation of their reliability. A critical challenge lies in hallucination, where models generate plausible yet factually incorrect outputs. In the medical domain, this poses serious risks to patient safety and clinical decision-making. To address this, we introduce MedHallu, the first benchmark specifically designed for medical hallucination detection. MedHallu comprises 10,000 high-quality question-answer pairs derived from PubMedQA, with hallucinated answers systematically generated through a controlled pipeline. Our experiments show that state-of-the-art LLMs, including GPT-4o, Llama-3.1, and the medically fine-tuned UltraMedical, struggle with this binary hallucination detection task, with the best model achieving an F1 score as low as 0.625 for detecting "hard" category hallucinations. Using bidirectional entailment clustering, we show that harder-to-detect hallucinations are semantically closer to ground truth. Through experiments, we also show incorporating domain-specific knowledge and introducing a "not sure" category as one of the answer categories improves the precision and F1 scores by up to 38% relative to baselines. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的进步及其在医疗问答领域的应用增长，亟需对其可靠性进行严格的评估。一个关键挑战在于幻觉现象，即模型生成虽然听起来合理但实际上不正确的输出。在医疗领域，这种情况会给患者安全和临床决策带来严重风险。为应对这一挑战，我们推出了MedHallu，这是首个专门用于医疗幻觉检测的基准测试。MedHallu包含来自PubMedQA的10,000对高质量的问题-答案对，通过一个控制化的管道系统性地生成了幻觉答案。我们的实验表明，最先进的LLMs，包括GPT-4o、Llama-3.1和经过医学微调的UltraMedical，在二元幻觉检测任务中并不擅长，最佳模型在检测“困难”类别幻觉时的F1分数仅为0.625。利用双向蕴含聚类，我们发现更难检测的幻觉与真实信息在语义上更加接近。通过实验我们还发现，将领域特定知识纳入考量，并引入“不确定”类别作为答案选项之一，相比基线模型可将准确率和F1分数提高高达38%。 

---
# SEA-HELM: Southeast Asian Holistic Evaluation of Language Models 

**Title (ZH)**: SEA-HELM：东南亚全方位语言模型评估 

**Authors**: Yosephine Susanto, Adithya Venkatadri Hulagadri, Jann Railey Montalan, Jian Gang Ngui, Xian Bin Yong, Weiqi Leong, Hamsawardhini Rengarajan, Peerat Limkonchotiwat, Yifan Mai, William Chandra Tjhi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14301)  

**Abstract**: With the rapid emergence of novel capabilities in Large Language Models (LLMs), the need for rigorous multilingual and multicultural benchmarks that are integrated has become more pronounced. Though existing LLM benchmarks are capable of evaluating specific capabilities of LLMs in English as well as in various mid- to low-resource languages, including those in the Southeast Asian (SEA) region, a comprehensive and authentic evaluation suite for the SEA languages has not been developed thus far. Here, we present SEA-HELM, a holistic linguistic and cultural LLM evaluation suite that emphasizes SEA languages, comprising five core pillars: (1) NLP Classics, (2) LLM-specifics, (3) SEA Linguistics, (4) SEA Culture, (5) Safety. SEA-HELM currently supports Filipino, Indonesian, Tamil, Thai, and Vietnamese. We also introduce the SEA-HELM leaderboard, which allows users to understand models' multilingual and multicultural performance in a systematic and user-friendly manner. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）新兴能力的迅速发展，需要更加严格且多语言、多文化集成的基准测试的需求愈发凸显。尽管现有的LLM基准测试能够评估LLMs在英语以及各种中低资源语言（包括东南亚地区语言）中的特定能力，但迄今为止尚未开发出一套综合且真实反映东南亚语言的评估体系。在此背景下，我们提出了SEA-HELM，这是一种全面的语言和文化LLM评估套件，强调东南亚语言，并包含五个核心支柱：（1）NLP经典任务，（2）LLM特定任务，（3）东南亚语言学，（4）东南亚文化，（5）安全性。目前，SEA-HELM支持菲律宾语、印尼语、泰米尔语、泰语和越南语。我们还介绍了SEA-HELM排行榜，该排行榜允许用户以系统且用户友好的方式了解模型在多语言和多文化方面的表现。 

---
# Drift: Decoding-time Personalized Alignments with Implicit User Preferences 

**Title (ZH)**: 漂移：基于隐含用户偏好的解码时个性化对齐 

**Authors**: Minbeom Kim, Kang-il Lee, Seongho Joo, Hwaran Lee, Minbeom Kim  

**Link**: [PDF](https://arxiv.org/pdf/2502.14289)  

**Abstract**: Personalized alignments for individual users have been a long-standing goal in large language models (LLMs). We introduce Drift, a novel framework that personalizes LLMs at decoding time with implicit user preferences. Traditional Reinforcement Learning from Human Feedback (RLHF) requires thousands of annotated examples and expensive gradient updates. In contrast, Drift personalizes LLMs in a training-free manner, using only a few dozen examples to steer a frozen model through efficient preference modeling. Our approach models user preferences as a composition of predefined, interpretable attributes and aligns them at decoding time to enable personalized generation. Experiments on both a synthetic persona dataset (Perspective) and a real human-annotated dataset (PRISM) demonstrate that Drift significantly outperforms RLHF baselines while using only 50-100 examples. Our results and analysis show that Drift is both computationally efficient and interpretable. 

**Abstract (ZH)**: 在大型语言模型（LLMs）中，个性化对齐用户个别的需求一直是一个长期目标。我们引入了Drift框架，该框架在解码时通过隐含的用户偏好个性化LLMs。传统的基于人类反馈强化学习（Reinforcement Learning from Human Feedback, RLHF）需要成千上万的标注示例和昂贵的梯度更新。相比之下，Drift以培训无需的方式个性化LLMs，仅使用几十个示例通过有效的偏好建模引导一个冻结的模型。我们的方法将用户偏好建模为预定义的可解释属性的组合，并在解码时对齐这些偏好以实现个性化生成。我们在一个合成的人设数据集（Perspective）和一个真实的人类标注数据集（PRISM）上的实验表明，Drift明显优于RLHF基线，并且仅使用了50-100个示例。我们的结果和分析表明，Drift在计算效率和可解释性方面都是高效的。 

---
# Vulnerability of Text-to-Image Models to Prompt Template Stealing: A Differential Evolution Approach 

**Title (ZH)**: 文本到图像模型对提示模板窃取的脆弱性：一种差分进化方法 

**Authors**: Yurong Wu, Fangwen Mu, Qiuhong Zhang, Jinjing Zhao, Xinrun Xu, Lingrui Mei, Yang Wu, Lin Shi, Junjie Wang, Zhiming Ding, Yiwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14285)  

**Abstract**: Prompt trading has emerged as a significant intellectual property concern in recent years, where vendors entice users by showcasing sample images before selling prompt templates that can generate similar images. This work investigates a critical security vulnerability: attackers can steal prompt templates using only a limited number of sample images. To investigate this threat, we introduce Prism, a prompt-stealing benchmark consisting of 50 templates and 450 images, organized into Easy and Hard difficulty levels. To identify the vulnerabity of VLMs to prompt stealing, we propose EvoStealer, a novel template stealing method that operates without model fine-tuning by leveraging differential evolution algorithms. The system first initializes population sets using multimodal large language models (MLLMs) based on predefined patterns, then iteratively generates enhanced offspring through MLLMs. During evolution, EvoStealer identifies common features across offspring to derive generalized templates. Our comprehensive evaluation conducted across open-source (INTERNVL2-26B) and closed-source models (GPT-4o and GPT-4o-mini) demonstrates that EvoStealer's stolen templates can reproduce images highly similar to originals and effectively generalize to other subjects, significantly outperforming baseline methods with an average improvement of over 10%. Moreover, our cost analysis reveals that EvoStealer achieves template stealing with negligible computational expenses. Our code and dataset are available at this https URL. 

**Abstract (ZH)**: 提示交易近年来已成为一个重要的人工智能知识产权问题，供应商通过展示样本图像来吸引用户，从而在销售生成类似图像的提示模板后获利。本文探讨了一个关键的安全漏洞：攻击者仅使用有限数量的样本图像就能窃取提示模板。为了调查这一威胁，我们引入了Prism，一个包含50个模板和450张图像的提示窃取基准，分为简易和困难两个难度级别。为了识别视觉语言模型（VLMs）对提示窃取的脆弱性，我们提出了EvoStealer，这是一种新颖的不需要微调模型的模板窃取方法，通过利用差异进化算法来实现。系统首先使用多模态大型语言模型（MLLMs）基于预定义的模式初始化种群集合，然后通过MLLMs迭代生成增强的后代。在进化过程中，EvoStealer识别后代的共同特征，从而推导出通用的提示模板。我们在开源（INTERNVL2-26B）和封闭源代码模型（GPT-4o和GPT-4o-mini）上的全面评估表明，EvoStealer窃取的模板能够生成与原始图像高度相似的图像，并且能够有效地泛化到其他主题，平均性能提高超过10%，显著优于基准方法。此外，我们的成本分析表明，EvoStealer的模板窃取具有极小的计算成本。我们的代码和数据集可在以下链接获取：[此处链接]。 

---
# EpMAN: Episodic Memory AttentioN for Generalizing to Longer Contexts 

**Title (ZH)**: EpMAN： episodic 记忆注意力机制 以适应更长的上下文 

**Authors**: Subhajit Chaudhury, Payel Das, Sarathkrishna Swaminathan, Georgios Kollias, Elliot Nelson, Khushbu Pahwa, Tejaswini Pedapati, Igor Melnyk, Matthew Riemer  

**Link**: [PDF](https://arxiv.org/pdf/2502.14280)  

**Abstract**: Recent advances in Large Language Models (LLMs) have yielded impressive successes on many language tasks. However, efficient processing of long contexts using LLMs remains a significant challenge. We introduce \textbf{EpMAN} -- a method for processing long contexts in an \textit{episodic memory} module while \textit{holistically attending to} semantically relevant context chunks. The output of \textit{episodic attention} is then used to reweigh the decoder's self-attention to the stored KV cache of the context during training and generation. When an LLM decoder is trained using \textbf{EpMAN}, its performance on multiple challenging single-hop long-context recall and question-answering benchmarks is found to be stronger and more robust across the range from 16k to 256k tokens than baseline decoders trained with self-attention, and popular retrieval-augmented generation frameworks. 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在许多自然语言处理任务中取得了令人印象深刻的成功。然而，高效处理长上下文仍然是一个重大挑战。我们介绍了一种名为**EpMAN**的方法，即在**情景记忆**模块中处理长上下文，同时**整体关注**语义相关的上下文片段。在训练和生成过程中，情景注意力的输出用于重新加权解码器的自注意力，使之指向上下文的存储KV缓存。当使用**EpMAN**训练LLM解码器时，其在多个具有挑战性的单一跳长上下文回忆和问答基准测试中的表现，在从16k到256k词元的整个范围内，均比使用自注意力和流行检索增强生成框架训练的基本解码器更为出色和稳定。 

---
# Fact or Guesswork? Evaluating Large Language Model's Medical Knowledge with Structured One-Hop Judgment 

**Title (ZH)**: 事实还是推测？基于结构化单跳判断评估大型语言模型的医学知识 

**Authors**: Jiaxi Li, Yiwei Wang, Kai Zhang, Yujun Cai, Bryan Hooi, Nanyun Peng, Kai-Wei Chang, Jin Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14275)  

**Abstract**: Large language models (LLMs) have been widely adopted in various downstream task domains. However, their ability to directly recall and apply factual medical knowledge remains under-explored. Most existing medical QA benchmarks assess complex reasoning or multi-hop inference, making it difficult to isolate LLMs' inherent medical knowledge from their reasoning capabilities. Given the high-stakes nature of medical applications, where incorrect information can have critical consequences, it is essential to evaluate how well LLMs encode, retain, and recall fundamental medical facts.
To bridge this gap, we introduce the Medical Knowledge Judgment, a dataset specifically designed to measure LLMs' one-hop factual medical knowledge. MKJ is constructed from the Unified Medical Language System (UMLS), a large-scale repository of standardized biomedical vocabularies and knowledge graphs. We frame knowledge assessment as a binary judgment task, requiring LLMs to verify the correctness of medical statements extracted from reliable and structured knowledge sources.
Our experiments reveal that LLMs struggle with factual medical knowledge retention, exhibiting significant performance variance across different semantic categories, particularly for rare medical conditions. Furthermore, LLMs show poor calibration, often being overconfident in incorrect answers. To mitigate these issues, we explore retrieval-augmented generation, demonstrating its effectiveness in improving factual accuracy and reducing uncertainty in medical decision-making. 

**Abstract (ZH)**: 大型语言模型（LLMs）已在各种下游任务领域得到广泛应用。然而，它们直接回忆和应用医学事实知识的能力仍较少被探索。现有的大多数医学问答基准评估的是复杂推理或多跳推理能力，这使得难以将LLMs固有的医学知识与其推理能力区分开来。鉴于医学应用的高度风险性，错误信息可能会导致严重后果，因此评估LLMs编码、存储和回忆基本医学事实的能力至关重要。

为解决这一问题，我们引入了《医学知识判断》（Medical Knowledge Judgment，MKJ）数据集，专门用于测量LLMs的一跳式医学事实知识。MKJ数据集是从统一医学语言系统（UMLS）构建的，UMLS是一个大规模的标准化生物医学词汇库和知识图谱的仓库。我们将知识评估任务设定为二分类判断任务，要求LLMs验证从可靠的结构化知识源中提取的医学声明的正确性。

我们的实验表明，LLMs在医学事实知识的存储方面存在困难，不同语义类别间的性能差异显著，尤其是对于罕见医学状况的表现尤为突出。此外，LLMs的校准效果较差，常常对自己的错误答案过于自信。为了缓解这些问题，我们探索了检索增强生成方法，并证明了这种方法在提高事实准确性、减少医学决策中的不确定性方面的有效性。 

---
# Capturing Nuanced Preferences: Preference-Aligned Distillation for Small Language Models 

**Title (ZH)**: 捕捉细腻的偏好：面向偏好的精简蒸馏方法用于小型语言模型 

**Authors**: Yanggan Gu, Junzhuo Li, Sirui Huang, Xin Zou, Zhenghua Li, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14272)  

**Abstract**: Aligning small language models (SLMs) with human values typically involves distilling preference knowledge from large language models (LLMs). However, existing distillation methods model preference knowledge in teacher LLMs by comparing pairwise responses, overlooking the extent of difference between responses. This limitation hinders student SLMs from capturing the nuanced preferences for multiple responses. In this paper, we propose a Preference-Aligned Distillation (PAD) framework, which models teacher's preference knowledge as a probability distribution over all potential preferences, thereby providing more nuanced supervisory signals. Our insight in developing PAD is rooted in the demonstration that language models can serve as reward functions, reflecting their intrinsic preferences. Based on this, PAD comprises three key steps: (1) sampling diverse responses using high-temperature; (2) computing rewards for both teacher and student to construct their intrinsic preference; and (3) training the student's intrinsic preference distribution to align with the teacher's. Experiments on four mainstream alignment benchmarks demonstrate that PAD consistently and significantly outperforms existing approaches, achieving over 20\% improvement on AlpacaEval 2 and Arena-Hard, indicating superior alignment with human preferences. Notably, on MT-Bench, using the \textsc{Gemma} model family, the student trained by PAD surpasses its teacher, further validating the effectiveness of our PAD. 

**Abstract (ZH)**: 将以下论文内容或标题翻译成中文，要符合学术规范：

将小型语言模型（SLMs）与人类价值观对齐通常涉及从大型语言模型（LLMs）中提取偏好知识。然而，现有的distillation方法通过成对响应比较来建模教师LLMs的偏好知识，忽略了响应之间的差异程度。这一局限性阻碍了学生SLMs捕捉多个响应的细微偏好。本文中，我们提出了一种偏好对齐蒸馏（Preference-Aligned Distillation, PAD）框架，该框架将教师的偏好知识建模为所有潜在偏好概率分布，从而提供更细致的监督信号。我们开发PAD的见解植根于语言模型可以作为奖励函数的演示，反映了它们的内在偏好。基于这一点，PAD包括三个关键步骤：（1）使用高温度抽样多样化的响应；（2）计算教师和学生各自的奖励来构建其内在偏好；（3）训练学生的内在偏好分布以与教师的偏好对齐。在四个主流对齐基准上的实验表明，PAD始终且显著优于现有的方法，在AlpacaEval 2和Arena-Hard上的改进超过20%，表明了更高的与人类偏好对齐程度。值得注意的是，在MT-Bench上，使用Gemma模型家族，通过PAD训练的学生超越了它的教师，进一步验证了我们PAD的有效性。 

---
# PaperHelper: Knowledge-Based LLM QA Paper Reading Assistant 

**Title (ZH)**: PaperHelper：基于知识的LLM问答式论文阅读辅助系统 

**Authors**: Congrui Yin, Evan Wei, Zhongxing Zhang, Zaifu Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2502.14271)  

**Abstract**: In the paper, we introduce a paper reading assistant, PaperHelper, a potent tool designed to enhance the capabilities of researchers in efficiently browsing and understanding scientific literature. Utilizing the Retrieval-Augmented Generation (RAG) framework, PaperHelper effectively minimizes hallucinations commonly encountered in large language models (LLMs), optimizing the extraction of accurate, high-quality knowledge. The implementation of advanced technologies such as RAFT and RAG Fusion significantly boosts the performance, accuracy, and reliability of the LLMs-based literature review process. Additionally, PaperHelper features a user-friendly interface that facilitates the batch downloading of documents and uses the Mermaid format to illustrate structural relationships between documents. Experimental results demonstrate that PaperHelper, based on a fine-tuned GPT-4 API, achieves an F1 Score of 60.04, with a latency of only 5.8 seconds, outperforming the basic RAG model by 7\% in F1 Score. 

**Abstract (ZH)**: 在本文中，我们引入了一种论文阅读助手——PaperHelper，这是一种强大的工具，旨在增强研究人员高效浏览和理解科学文献的能力。通过利用检索增强生成（RAG）框架，PaperHelper有效减少了大型语言模型（LLMs）中常见的幻觉现象，优化了高质量、高准确性的知识提取。先进的技术如RAFT和RAG融合的实施显著提升了基于LLMs的文献回顾过程的性能、准确性和可靠性。此外，PaperHelper还具备用户友好的界面，便于批量下载文档，并使用Mermaid格式展示文档之间的结构关系。实验结果表明，基于微调的GPT-4 API，PaperHelper实现了F1分数60.04，响应延迟仅为5.8秒，相比基本的RAG模型在F1分数上提高了7%。 

---
# MCQA-Eval: Efficient Confidence Evaluation in NLG with Gold-Standard Correctness Labels 

**Title (ZH)**: MCQA-Eval：基于标准正确性标签的NLG中高效自信心评估 

**Authors**: Xiaoou Liu, Zhen Lin, Longchao Da, Chacha Chen, Shubhendu Trivedi, Hua Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.14268)  

**Abstract**: Large Language Models (LLMs) require robust confidence estimation, particularly in critical domains like healthcare and law where unreliable outputs can lead to significant consequences. Despite much recent work in confidence estimation, current evaluation frameworks rely on correctness functions -- various heuristics that are often noisy, expensive, and possibly introduce systematic biases. These methodological weaknesses tend to distort evaluation metrics and thus the comparative ranking of confidence measures. We introduce MCQA-Eval, an evaluation framework for assessing confidence measures in Natural Language Generation (NLG) that eliminates dependence on an explicit correctness function by leveraging gold-standard correctness labels from multiple-choice datasets. MCQA-Eval enables systematic comparison of both internal state-based white-box (e.g. logit-based) and consistency-based black-box confidence measures, providing a unified evaluation methodology across different approaches. Through extensive experiments on multiple LLMs and widely used QA datasets, we report that MCQA-Eval provides efficient and more reliable assessments of confidence estimation methods than existing approaches. 

**Abstract (ZH)**: 大语言模型（LLMs）需要稳健的置信度估计，特别是在医疗和法律等关键领域，不准确的输出可能导致严重后果。尽管在置信度估计方面已经进行了大量的研究工作，但当前的评估框架仍然依赖于纠正函数——这通常是噪声较大、成本较高且可能引入系统性偏差的各种启发式方法。这些方法论上的弱点往往扭曲了评估指标并影响置信度度量的比较排名。我们提出了MCQA-Eval，这是一种用于自然语言生成（NLG）中评估置信度度量的评估框架，通过使用来自多项选择数据集的金标准正确性标签，消除了对显式正确性函数的依赖。MCQA-Eval 使内部状态为基础的白盒置信度度量（例如，基于logit的）和基于一致性的黑盒置信度度量的系统比较成为可能，提供了统一的评估方法论，适用于不同方法。通过在多种LLM和广泛使用的问答数据集上进行广泛的实验，我们报告说，MCQA-Eval 提供了比现有方法更高效且更可靠的置信度估计方法评估。 

---
# Does Time Have Its Place? Temporal Heads: Where Language Models Recall Time-specific Information 

**Title (ZH)**: 时间有其位置吗？时间头部：语言模型如何回忆特定时间段的信息 

**Authors**: Yein Park, Chanwoong Yoon, Jungwoo Park, Minbyul Jeong, Jaewoo Kang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14258)  

**Abstract**: While the ability of language models to elicit facts has been widely investigated, how they handle temporally changing facts remains underexplored. We discover Temporal Heads, specific attention heads primarily responsible for processing temporal knowledge through circuit analysis. We confirm that these heads are present across multiple models, though their specific locations may vary, and their responses differ depending on the type of knowledge and its corresponding years. Disabling these heads degrades the model's ability to recall time-specific knowledge while maintaining its general capabilities without compromising time-invariant and question-answering performances. Moreover, the heads are activated not only numeric conditions ("In 2004") but also textual aliases ("In the year ..."), indicating that they encode a temporal dimension beyond simple numerical representation. Furthermore, we expand the potential of our findings by demonstrating how temporal knowledge can be edited by adjusting the values of these heads. 

**Abstract (ZH)**: 尽管语言模型提取事实的能力已经被广泛研究，但它们如何处理随着时间变化的事实仍然未被充分探索。我们发现了时间头部，这是一种特定的注意力头部，主要通过电路分析处理时间知识。我们证实，这些头部存在于多个模型中，尽管它们的具体位置可能不同，且其响应会根据知识类型和相应的年份而变化。禁用这些头部会降低模型回忆特定时间知识的能力，同时其一般能力保持不变，而不会影响其不变时间和问答性能。此外，这些头部不仅被数值条件（“在2004年”）激活，还被文本别名（“在...年”）激活，这表明它们编码了超越简单数值表示的时间维度。此外，我们通过演示如何通过调整这些头部的值来编辑时间知识，扩展了我们发现的潜在应用。 

---
# Effects of Prompt Length on Domain-specific Tasks for Large Language Models 

**Title (ZH)**: 大型语言模型在特定领域任务中提示长度的影响研究 

**Authors**: Qibang Liu, Wenzhe Wang, Jeffrey Willard  

**Link**: [PDF](https://arxiv.org/pdf/2502.14255)  

**Abstract**: In recent years, Large Language Models have garnered significant attention for their strong performance in various natural language tasks, such as machine translation and question answering. These models demonstrate an impressive ability to generalize across diverse tasks. However, their effectiveness in tackling domain-specific tasks, such as financial sentiment analysis and monetary policy understanding, remains a topic of debate, as these tasks often require specialized knowledge and precise reasoning. To address such challenges, researchers design various prompts to unlock the models' abilities. By carefully crafting input prompts, researchers can guide these models to produce more accurate responses. Consequently, prompt engineering has become a key focus of study. Despite the advancements in both models and prompt engineering, the relationship between the two-specifically, how prompt design impacts models' ability to perform domain-specific tasks-remains underexplored. This paper aims to bridge this research gap. 

**Abstract (ZH)**: 近年来，大规模语言模型因其在各种自然语言任务中的出色表现（如机器翻译和问答）而引起了广泛关注。这些模型展示了跨多种任务的强大泛化能力。然而，它们在处理特定领域任务（如金融情绪分析和货币政策理解）方面的有效性仍是一个有争议的话题，因为这些任务通常需要专门的知识和精确的推理。为应对这些挑战，研究人员设计了各种提示来充分发挥模型的能力。通过精心设计输入提示，研究人员可以引导这些模型生成更准确的响应。因此，提示工程已经成为了研究的重点之一。尽管在模型和提示工程方面取得了进展，但模型和提示设计之间关系的具体机制，特别是提示设计如何影响模型在特定领域任务上的表现，仍需进一步探索。本文旨在弥补这一研究空白。 

---
# Mitigating Lost-in-Retrieval Problems in Retrieval Augmented Multi-Hop Question Answering 

**Title (ZH)**: 缓解检索增强多跳问答中的检索丢失问题 

**Authors**: Rongzhi Zhu, Xiangyu Liu, Zequn Sun, Yiwei Wang, Wei Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14245)  

**Abstract**: In this paper, we identify a critical problem, "lost-in-retrieval", in retrieval-augmented multi-hop question answering (QA): the key entities are missed in LLMs' sub-question decomposition. "Lost-in-retrieval" significantly degrades the retrieval performance, which disrupts the reasoning chain and leads to the incorrect answers. To resolve this problem, we propose a progressive retrieval and rewriting method, namely ChainRAG, which sequentially handles each sub-question by completing missing key entities and retrieving relevant sentences from a sentence graph for answer generation. Each step in our retrieval and rewriting process builds upon the previous one, creating a seamless chain that leads to accurate retrieval and answers. Finally, all retrieved sentences and sub-question answers are integrated to generate a comprehensive answer to the original question. We evaluate ChainRAG on three multi-hop QA datasets$\unicode{x2013}$MuSiQue, 2Wiki, and HotpotQA$\unicode{x2013}$using three large language models: GPT4o-mini, Qwen2.5-72B, and GLM-4-Plus. Empirical results demonstrate that ChainRAG consistently outperforms baselines in both effectiveness and efficiency. 

**Abstract (ZH)**: 在本文中，我们识别出检索增强多跳问答（QA）中的一个关键问题：“检索丢失”（lost-in-retrieval）：在大语言模型（LLMs）的子问题分解过程中，关键实体被遗漏了。“检索丢失”严重影响了检索性能，打断了推理链，导致错误的答案。为了解决这一问题，我们提出了一种渐进式的检索和重写方法，称为ChainRAG，该方法逐个处理每个子问题，通过补充缺失的关键实体并从句子图中检索相关句子来生成答案。在我们的检索和重写过程中，每一步都依赖于上一步，形成一个无缝的链，从而实现准确的检索和答案生成。最后，所有检索到的句子和子问题答案被整合起来生成原始问题的全面答案。我们使用三个多跳问答数据集（MuSiQue、2Wiki 和 HotpotQA）以及三个大型语言模型（GPT4o-mini、Qwen2.5-72B 和 GLM-4-Plus）对ChainRAG 进行了评估。实证结果表明，ChainRAG 在有效性和效率上都优于基线方法。 

---
# Transfer-Prompting: Enhancing Cross-Task Adaptation in Large Language Models via Dual-Stage Prompts Optimization 

**Title (ZH)**: 迁移提示：通过双重阶段提示优化增强大型语言模型的跨任务适应性 

**Authors**: Yupeng Chang, Yi Chang, Yuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14211)  

**Abstract**: Large language models (LLMs) face significant challenges when balancing multiple high-level objectives, such as generating coherent, relevant, and high-quality responses while maintaining efficient task adaptation across diverse tasks. To address these challenges, we introduce Transfer-Prompting, a novel two-stage framework designed to enhance cross-task adaptation in prompt generation. The framework comprises two key components: (1) source prompt construction, which refines the original prompts on source task datasets to generate source prompts with enhanced generalization ability, and (2) target prompt generation, which enhances cross-task adaptation of target prompts by fine-tuning a set of high-scored source prompts on task-specific datasets. In each optimization cycle, a reference LLM generates candidate prompts based on historical prompt-score pairs and task descriptions in our designed reference prompt. These candidate prompts are refined iteratively, while a scorer LLM evaluates their effectiveness using the multi-dimensional metrics designed in the objective prompts evaluator-a novel contribution in this work that provides a holistic evaluation of prompt quality and task performance. This feedback loop facilitates continuous refinement, optimizing both prompt quality and task-specific outcomes. We validate Transfer-Prompting through extensive experiments across 25 LLMs, including 7 foundational models and 18 specialized models, evaluated on 9 diverse datasets. The results demonstrate that Transfer-Prompting significantly improves task-specific performance, highlighting its potential for enhancing cross-task adaptation in LLMs. The code is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在平衡多个高级目标时面临重大挑战，例如生成连贯、相关且高质量的响应，同时保持高效的任务适应性，应用于多样化任务。为应对这些挑战，我们提出了转移提示（Transfer-Prompting），这是一种新颖的两阶段框架，旨在增强提示生成中的跨任务适应性。该框架包含两个关键组件：（1）源提示构建，即将原始提示在源任务数据集上进行细化，生成具有增强泛化能力的源提示；（2）目标提示生成，通过在任务特定数据集上微调得分较高的源提示，增强目标提示的跨任务适应性。在每次优化循环中，参考LLM基于我们设计的参考提示中的历史提示-分数对和任务描述生成候选提示。这些候选提示在迭代过程中不断改进，而评分LLM则使用目标提示评估者（本工作中的一项新贡献）设计的多维度指标评估其有效性，该评估者提供了对提示质量及任务性能的全面评价。这种反馈循环促进了持续的优化，既优化了提示质量又增强了任务特定效果。我们通过涵盖25个LLM（包括7个基础模型和18个专门模型）的广泛实验，评估了Transfer-Prompting在9个不同数据集上的表现。结果表明，Transfer-Prompting显著提高了任务特定性能，突显了其在提升LLM的跨任务适应性方面的潜力。代码可在以下网址获取：[在这里插入网址]。 

---
# On-the-fly Preference Alignment via Principle-Guided Decoding 

**Title (ZH)**: 基于原则引导解码的即时偏好对齐 

**Authors**: Mingye Zhu, Yi Liu, Lei Zhang, Junbo Guo, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14204)  

**Abstract**: With the rapidly expanding landscape of large language models, aligning model generations with human values and preferences is becoming increasingly important. Popular alignment methods, such as Reinforcement Learning from Human Feedback, have shown significant success in guiding models with greater control. However, these methods require considerable computational resources, which is inefficient, and substantial collection of training data to accommodate the diverse and pluralistic nature of human preferences, which is impractical. These limitations significantly constrain the scope and efficacy of both task-specific and general preference alignment methods. In this work, we introduce On-the-fly Preference Alignment via Principle-Guided Decoding (OPAD) to directly align model outputs with human preferences during inference, eliminating the need for fine-tuning. Our approach involves first curating a surrogate solution to an otherwise infeasible optimization problem and then designing a principle-guided reward function based on this surrogate. The final aligned policy is derived by maximizing this customized reward, which exploits the discrepancy between the constrained policy and its unconstrained counterpart. OPAD directly modifies the model's predictions during inference, ensuring principle adherence without incurring the computational overhead of retraining or fine-tuning. Experiments show that OPAD achieves competitive or superior performance in both general and personalized alignment tasks, demonstrating its efficiency and effectiveness compared to state-of-the-art baselines. 

**Abstract (ZH)**: 随着大规模语言模型应用场景的迅速扩展，将模型生成与人类价值观和偏好进行对齐变得越来越重要。诸如基于人类反馈强化学习（Reinforcement Learning from Human Feedback, RLHF）等流行的方法在引导具有更高控制能力的模型方面已经取得了显著的成功。然而，这些方法需要大量的计算资源，这是低效的，同时还需收集大量的训练数据以适应人类偏好在多样性和包容性方面的特点，这是不切实际的。这些限制对任务特定和一般偏好对齐方法的适用范围和有效性产生了显著的制约。在本项研究中，我们引入了一种名为基于原则指导解码的实时偏好对齐（On-the-fly Preference Alignment via Principle-Guided Decoding, OPAD）的方法，在推理过程中直接将模型输出与人类偏好对齐，从而省去了微调的需要。我们的方法首先提供了一个可行的替代解来解决一个原本身不可行的优化问题，然后基于这一替代解设计了一个原则指导的奖励函数。最终对齐的策略通过最大化这一定制化的奖励获得，该奖励利用了约束策略与其无约束对应策略之间的差异。OPAD 在推理过程中直接修改模型的预测，确保了原则的遵守，同时没有重新训练或微调带来的计算开销。实验结果表明，OPAD 在通用和个性化偏好对齐任务中均表现出竞争性的或优越的性能，证明了其在效率和效果方面的优越性，相比于最先进的基线方法具有明显优势。 

---
# NLP-AKG: Few-Shot Construction of NLP Academic Knowledge Graph Based on LLM 

**Title (ZH)**: NLP-AKG：基于大语言模型的少量示例构建NLP学术知识图谱 

**Authors**: Jiayin Lan, Jiaqi Li, Baoxin Wang, Ming Liu, Dayong Wu, Shijin Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2502.14192)  

**Abstract**: Large language models (LLMs) have been widely applied in question answering over scientific research papers. To enhance the professionalism and accuracy of responses, many studies employ external knowledge augmentation. However, existing structures of external knowledge in scientific literature often focus solely on either paper entities or domain concepts, neglecting the intrinsic connections between papers through shared domain concepts. This results in less comprehensive and specific answers when addressing questions that combine papers and concepts. To address this, we propose a novel knowledge graph framework that captures deep conceptual relations between academic papers, constructing a relational network via intra-paper semantic elements and inter-paper citation relations. Using a few-shot knowledge graph construction method based on LLM, we develop NLP-AKG, an academic knowledge graph for the NLP domain, by extracting 620,353 entities and 2,271,584 relations from 60,826 papers in ACL Anthology. Based on this, we propose a 'sub-graph community summary' method and validate its effectiveness on three NLP scientific literature question answering datasets. 

**Abstract (ZH)**: 大型语言模型（LLMs）在科研论文的问答任务中得到了广泛应用。为了提升回应的专业性和准确性，许多研究采用了外部知识增强的方法。然而，现有科学文献中的外部知识结构往往仅专注于论文实体或领域概念，忽略了通过共享领域概念论文之间的内在联系。这种做法导致在回答结合论文和概念的问题时，答案不够全面和具体。为解决这一问题，我们提出了一种新的知识图谱框架，能够捕获学术论文之间深层次的概念关系，通过构建基于论文内语义元素和论文间引用关系的交互网络来进行这一过程。我们基于大模型（LLM）采用少量示例的知识图谱构建方法，构建了NLP-AKG——一个用于NLP领域的学术知识图谱，从中提取了来自ACL Anthology的60,826篇论文中的620,353个实体和2,271,584个关系。在此基础上，我们提出了一种“子图社区摘要”方法，并在三个NLP科学文献问答数据集上验证了该方法的有效性。 

---
# QUAD-LLM-MLTC: Large Language Models Ensemble Learning for Healthcare Text Multi-Label Classification 

**Title (ZH)**: QUAD-LLM-MLTC：医疗文本多标签分类的大语言模型集成学习 

**Authors**: Hajar Sakai, Sarah S. Lam  

**Link**: [PDF](https://arxiv.org/pdf/2502.14189)  

**Abstract**: The escalating volume of collected healthcare textual data presents a unique challenge for automated Multi-Label Text Classification (MLTC), which is primarily due to the scarcity of annotated texts for training and their nuanced nature. Traditional machine learning models often fail to fully capture the array of expressed topics. However, Large Language Models (LLMs) have demonstrated remarkable effectiveness across numerous Natural Language Processing (NLP) tasks in various domains, which show impressive computational efficiency and suitability for unsupervised learning through prompt engineering. Consequently, these LLMs promise an effective MLTC of medical narratives. However, when dealing with various labels, different prompts can be relevant depending on the topic. To address these challenges, the proposed approach, QUAD-LLM-MLTC, leverages the strengths of four LLMs: GPT-4o, BERT, PEGASUS, and BART. QUAD-LLM-MLTC operates in a sequential pipeline in which BERT extracts key tokens, PEGASUS augments textual data, GPT-4o classifies, and BART provides topics' assignment probabilities, which results in four classifications, all in a 0-shot setting. The outputs are then combined using ensemble learning and processed through a meta-classifier to produce the final MLTC result. The approach is evaluated using three samples of annotated texts, which contrast it with traditional and single-model methods. The results show significant improvements across the majority of the topics in the classification's F1 score and consistency (F1 and Micro-F1 scores of 78.17% and 80.16% with standard deviations of 0.025 and 0.011, respectively). This research advances MLTC using LLMs and provides an efficient and scalable solution to rapidly categorize healthcare-related text data without further training. 

**Abstract (ZH)**: 不断增长的医疗文本数据量为自动多标签文本分类（MLTC）带来了独特挑战，主要原因是对训练数据的注释稀少及其复杂的性质。传统机器学习模型往往难以充分捕捉到表达的各种主题。然而，大型语言模型（LLMs）在多个自然语言处理（NLP）任务中显示出巨大的效果，尤其是在不同领域，它们展示了出色的计算效率和通过提示工程进行无监督学习的适用性。因此，这些LLMs有望有效地实现医疗叙事的MLTC。然而，在处理各种标签时，不同的话题可能需要不同的提示。为应对这些挑战，我们提出了一种名为QUAD-LLM-MLTC的方法，充分利用了四个LLM的优势：GPT-4o、BERT、PEGASUS和BART。QUAD-LLM-MLTC在顺序处理管道中运作，首先由BERT提取关键标记，然后由PEGASUS扩充文本数据，GPT-4o进行分类，BART提供主题的概率分配，从而产生四个分类结果，均在零样本设置下完成。最终的MLTC结果通过集成学习和元分类器生成。该方法使用三个标注数据样本进行评估，与传统方法和单模型方法进行对比。结果显示，在分类F1分数和一致性方面，多数话题均有显著改进（F1和微观F1分数分别为78.17%和80.16%，标准差分别为0.025和0.011）。这项研究通过LLMs推进了MLTC的方法，并提供了一种高效可扩展的解决方案，可以在不进一步训练的情况下快速对相关医疗文本数据进行分类。 

---
# Enhancing Conversational Agents with Theory of Mind: Aligning Beliefs, Desires, and Intentions for Human-Like Interaction 

**Title (ZH)**: 增强对话代理的心智理论：使信念、欲望和意图对齐以实现类似人类的互动 

**Authors**: Mohammadmahdi Jafari, Devin Yuncheng Hua, Hao Xue, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2502.14171)  

**Abstract**: Natural language interaction with agentic Artificial Intelligence (AI), driven by Large Language Models (LLMs), is expected to remain a dominant paradigm in the near future. While humans instinctively align their communication with mental states -- an ability known as Theory of Mind (ToM), current LLM powered systems exhibit significant limitations in this regard. This study examines the extent to which open source language models (LLaMA) can capture and preserve ToM related information and how effectively it contributes to consistent ToM reasoning in generated responses. We further investigate whether explicit manipulation of ToM related components, such as beliefs, desires, and intentions, can enhance response alignment. Experiments on two LLaMA 3 variants demonstrate that incorporating ToM informed alignment improves response quality, achieving win rates of 67 and 63 percent for the 3B and 8B models, respectively. These findings highlight the potential of ToM driven strategies to improve alignment in LLM based conversational agents. 

**Abstract (ZH)**: 由大规模语言模型（LLMs）驱动的代理人工智能（AGI）自然语言交互，预计在未来一段时间内仍将是主导范式。尽管人类本能地根据心理状态调整自己的交流方式（这种能力被称为心智理论，即Theory of Mind, ToM），但当前的LLM驱动系统在这方面表现出明显的局限性。本研究考察了开源语言模型（例如LaMA）能否捕捉和保留与ToM相关的信息，以及这些信息如何有效促进生成响应中的持续性ToM推理。此外，我们还研究了显式操纵与ToM相关的成分（如信念、欲望和意图）是否能够提升响应的对齐程度。在两种LaMA 3变体的实验中，将ToM导向的对齐机制纳入模型显著提高了响应质量，3B模型和8B模型的赢率分别为67%和63%。这些发现突显了ToM驱动策略在提高基于LLM的对话代理对齐程度方面的潜力。 

---
# LLM-Enhanced Dialogue Management for Full-Duplex Spoken Dialogue Systems 

**Title (ZH)**: 基于LLM增强的全双工语音对话系统对话管理 

**Authors**: Hao Zhang, Weiwei Li, Rilin Chen, Vinay Kothapally, Meng Yu, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14145)  

**Abstract**: Achieving full-duplex communication in spoken dialogue systems (SDS) requires real-time coordination between listening, speaking, and thinking. This paper proposes a semantic voice activity detection (VAD) module as a dialogue manager (DM) to efficiently manage turn-taking in full-duplex SDS. Implemented as a lightweight (0.5B) LLM fine-tuned on full-duplex conversation data, the semantic VAD predicts four control tokens to regulate turn-switching and turn-keeping, distinguishing between intentional and unintentional barge-ins while detecting query completion for handling user pauses and hesitations. By processing input speech in short intervals, the semantic VAD enables real-time decision-making, while the core dialogue engine (CDE) is only activated for response generation, reducing computational overhead. This design allows independent DM optimization without retraining the CDE, balancing interaction accuracy and inference efficiency for scalable, next-generation full-duplex SDS. 

**Abstract (ZH)**: 在语音对话系统（Spoken Dialogue Systems, SDS）中实现全双工通信需要实时协调倾听、说话和思考之间的关系。本文提出了一种语义语音活动检测（VAD）模块，作为对话管理器（Dialogue Manager, DM），以高效管理全双工SDS中的轮换。该语义VAD模块采用轻量级（约0.5B）语言模型（LLM），并针对全双工对话数据进行了微调，能够预测四个控制标记以调节轮换和保持发言，区分有意和无意的插话，同时检测查询完成，以应对用户的暂停和犹豫。通过以短时间片处理输入语音，语义VAD能够实现实时决策，而核心对话引擎（Core Dialogue Engine, CDE）仅在生成响应时被激活，从而减少了计算负担。这种设计允许DM独立优化，而无需重新训练CDE，同时平衡交互准确性和推理效率，以实现可扩展的下一代全双工SDS。 

---
# UM_FHS at TREC 2024 PLABA: Exploration of Fine-tuning and AI agent approach for plain language adaptations of biomedical text 

**Title (ZH)**: UM_FHS在TREC 2024 PLABA任务中的探索：生物医学文本 plain language 调整的微调与AI代理方法研究

注释：
1. TREC指的是文本检索评估会议（Text REtrieval Conference）。
2. PLABA是2024年TREC的一个专题任务，全称可能是Plain Language Adaptation of Biomedical Abstracts。
3. "plain language 调整" 旨在翻译 "plain language adaptations"，指的是将复杂的医学文本转化为更易于普通读者理解的语言。
4. 该翻译力求准确传达原始标题的意思，同时符合中文的学术表达习惯。 

**Authors**: Primoz Kocbek, Leon Kopitar, Zhihong Zhang, Emirhan Aydin, Maxim Topaz, Gregor Stiglic  

**Link**: [PDF](https://arxiv.org/pdf/2502.14144)  

**Abstract**: This paper describes our submissions to the TREC 2024 PLABA track with the aim to simplify biomedical abstracts for a K8-level audience (13-14 years old students). We tested three approaches using OpenAI's gpt-4o and gpt-4o-mini models: baseline prompt engineering, a two-AI agent approach, and fine-tuning. Adaptations were evaluated using qualitative metrics (5-point Likert scales for simplicity, accuracy, completeness, and brevity) and quantitative readability scores (Flesch-Kincaid grade level, SMOG Index). Results indicated that the two-agent approach and baseline prompt engineering with gpt-4o-mini models show superior qualitative performance, while fine-tuned models excelled in accuracy and completeness but were less simple. The evaluation results demonstrated that prompt engineering with gpt-4o-mini outperforms iterative improvement strategies via two-agent approach as well as fine-tuning with gpt-4o. We intend to expand our investigation of the results and explore advanced evaluations. 

**Abstract (ZH)**: 本文描述了我们参加TREC 2024 PLABA赛道的提交，旨在简化医学摘要以便K8级别（13-14岁学生）的受众理解。我们使用了OpenAI的gpt-4o和gpt-4o-mini模型，共测试了三种方法：基线提示工程、双人工智能代理方法和微调。这些方法的适应性通过定性指标（针对简单性、准确性、完整性和简洁性的5点李克特量表）和定量的可读性评分（Flesch-Kincaid年级水平、SMOG指数）进行了评估。结果表明，双人工智能代理方法和使用gpt-4o-mini模型的基线提示工程在定性性能上表现更优，而微调模型在准确性和完整性方面表现更出色，但在简单性方面则较差。评估结果显示，使用gpt-4o-mini模型的提示工程策略优于通过双人工智能代理方法进行的迭代改进策略，也优于使用gpt-4o模型的微调方法。我们计划进一步研究这些结果，并探索更先进的评估方法。 

---
# Self-Regularization with Latent Space Explanations for Controllable LLM-based Classification 

**Title (ZH)**: 基于潜在空间解释的自我正则化控制性大型语言模型分类 

**Authors**: Xuansheng Wu, Wenhao Yu, Xiaoming Zhai, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14133)  

**Abstract**: Modern text classification methods heavily rely on contextual embeddings from large language models (LLMs). Compared to human-engineered features, these embeddings provide automatic and effective representations for classification model training. However, they also introduce a challenge: we lose the ability to manually remove unintended features, such as sensitive or task-irrelevant features, to guarantee regulatory compliance or improve the generalizability of classification models. This limitation arises because LLM embeddings are opaque and difficult to interpret. In this paper, we propose a novel framework to identify and regularize unintended features in the LLM latent space. Specifically, we first pre-train a sparse autoencoder (SAE) to extract interpretable features from LLM latent spaces. To ensure the SAE can capture task-specific features, we further fine-tune it on task-specific datasets. In training the classification model, we propose a simple and effective regularizer, by minimizing the similarity between the classifier weights and the identified unintended feature, to remove the impacts of these unintended features toward classification. We evaluate the proposed framework on three real-world tasks, including toxic chat detection, reward modeling, and disease diagnosis. Results show that the proposed framework can significantly improve the classifier's generalizability by regularizing those features that are not semantically correlated to each task. This work pioneers controllable text classification on LLM latent spaces by leveraging interpreted features to address generalizability, fairness, and privacy challenges. We will release our code and data once accepted. 

**Abstract (ZH)**: 现代文本分类方法强烈依赖于大语言模型（LLM）的上下文嵌入。与人工设计的特征相比，这些嵌入能够自动且有效地为分类模型的训练提供表示形式。然而，这同时也带来了一个挑战：我们失去了手动移除未预期特征的能力，例如敏感信息或与任务无关的特征，以确保符合监管要求或提高分类模型的一般性。这种局限性源于LLM嵌入的不透明性和难以解释性。在本文中，我们提出了一种新的框架来识别和规范LLM潜在空间中的未预期特征。具体而言，我们首先预训练了一个稀疏自编码器（SAE），从LLM潜在空间中提取可解释的特征。为了确保SAE能够捕捉到任务相关的特征，我们进一步针对特定任务的数据集对其进行微调。在训练分类模型时，我们提出了一种简单而有效的正则化方法，通过最小化分类器权重与识别出的未预期特征之间的相似性，来移除这些未预期特征对分类的影响。我们使用三个实际任务对提出的框架进行了评估，包括有毒聊天检测、奖励建模和疾病诊断。结果表明，该框架通过规范化与任务语义无关的特征，能显著提高分类器的一般性。这项工作通过利用可解释的特征来解决LLM潜在空间中的可控制文本分类问题中的泛化能力、公平性和隐私问题，取得了先驱性进展。论文被接受后，我们将发布我们的代码和数据。 

---
# Can Community Notes Replace Professional Fact-Checkers? 

**Title (ZH)**: 社区笔记能否取代专业事实核查员？ 

**Authors**: Nadav Borenstein, Greta Warren, Desmond Elliott, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2502.14132)  

**Abstract**: Two commonly-employed strategies to combat the rise of misinformation on social media are (i) fact-checking by professional organisations and (ii) community moderation by platform users. Policy changes by Twitter/X and, more recently, Meta, signal a shift away from partnerships with fact-checking organisations and towards an increased reliance on crowdsourced community notes. However, the extent and nature of dependencies between fact-checking and helpful community notes remain unclear. To address these questions, we use language models to annotate a large corpus of Twitter/X community notes with attributes such as topic, cited sources, and whether they refute claims tied to broader misinformation narratives. Our analysis reveals that community notes cite fact-checking sources up to five times more than previously reported. Fact-checking is especially crucial for notes on posts linked to broader narratives, which are twice as likely to reference fact-checking sources compared to other sources. In conclusion, our results show that successful community moderation heavily relies on professional fact-checking. 

**Abstract (ZH)**: 对抗社交媒体上信息误导问题的两种常见策略是：（i）专业组织进行事实核查，以及（ii）平台用户进行社区管理。Twitter/X和最近的Meta所采取的政策变化，表明他们正在从与事实核查组织的合作转向更依赖于众包社区注释。然而，事实核查与有益的社区注释之间的依赖关系的程度和性质仍然不清楚。为了解决这些问题，我们使用语言模型对大量的Twitter/X社区注释进行了标注，包括主题、引用的来源以及其是否反驳了更广泛的信息误导叙事中的相关声明。我们的分析显示，社区注释引用的事实核查来源的数量是之前报告的五倍之多。特别需要注意的是，对于链接到更广泛叙事的帖子上的注释，这些注释比其他来源更有可能引用事实核查来源，几率是后者的两倍。最终，我们的结果表明，有效的社区管理高度依赖于专业事实核查。 

---
# Which of These Best Describes Multiple Choice Evaluation with LLMs? A) Forced B) Flawed C) Fixable D) All of the Above 

**Title (ZH)**: 以下哪个选项最能描述使用大规模语言模型（LLM）进行的选择题评估？A）被迫的 B）有缺陷的 C）可修复的 D）以上全部 

**Authors**: Nishant Balepur, Rachel Rudinger, Jordan Lee Boyd-Graber  

**Link**: [PDF](https://arxiv.org/pdf/2502.14127)  

**Abstract**: Multiple choice question answering (MCQA) is popular for LLM evaluation due to its simplicity and human-like testing, but we argue for its reform. We first reveal flaws in MCQA's format, as it struggles to: 1) test generation/subjectivity; 2) match LLM use cases; and 3) fully test knowledge. We instead advocate for generative formats based on human testing-where LLMs construct and explain answers-better capturing user needs and knowledge while remaining easy to score. We then show even when MCQA is a useful format, its datasets suffer from: leakage; unanswerability; shortcuts; and saturation. In each issue, we give fixes from education, like rubrics to guide MCQ writing; scoring methods to bridle guessing; and Item Response Theory to build harder MCQs. Lastly, we discuss LLM errors in MCQA-robustness, biases, and unfaithful explanations-showing how our prior solutions better measure or address these issues. While we do not need to desert MCQA, we encourage more efforts in refining the task based on educational testing, advancing evaluations. 

**Abstract (ZH)**: 多项选择题回答（MCQA）因其简洁性和接近人类的测试方式而常被用于大语言模型（LLM）评估，但我们认为需要改革这一形式。我们首先揭示了MCQA格式中的缺陷，因为它难以：1) 测试生成能力和主观性；2) 符合LLM的应用场景；3) 完整测试知识。因此，我们主张采用基于人工测试的生成性格式，其中LLM构建并解释答案，更能捕捉用户的需求和知识，并且易于评分。接着，我们展示了即使在MCQA是一种有用的格式时，其数据集也存在泄露、不可回答性、捷径和饱和等问题。在每个问题上，我们提供了源自教育领域的解决方案，比如评估量表来指导MCQ写作；评分方法来限制猜测；以及项目反应理论来构建更难的MCQ。最后，我们讨论了MCQA中LLM的错误，包括鲁棒性、偏见和不忠实的解释，展示了我们之前提出的方法如何更好地衡量或解决这些问题。虽然我们不必放弃MCQA，但我们鼓励在基于教育测试的原则下进一步完善这一任务，推动评估的进步。 

---
# Benchmarking LLMs for Political Science: A United Nations Perspective 

**Title (ZH)**: 从联合国视角 benchmark 政治科学中的大规模语言模型 

**Authors**: Yueqing Liang, Liangwei Yang, Chen Wang, Congying Xia, Rui Meng, Xiongxiao Xu, Haoran Wang, Ali Payani, Kai Shu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14122)  

**Abstract**: Large Language Models (LLMs) have achieved significant advances in natural language processing, yet their potential for high-stake political decision-making remains largely unexplored. This paper addresses the gap by focusing on the application of LLMs to the United Nations (UN) decision-making process, where the stakes are particularly high and political decisions can have far-reaching consequences. We introduce a novel dataset comprising publicly available UN Security Council (UNSC) records from 1994 to 2024, including draft resolutions, voting records, and diplomatic speeches. Using this dataset, we propose the United Nations Benchmark (UNBench), the first comprehensive benchmark designed to evaluate LLMs across four interconnected political science tasks: co-penholder judgment, representative voting simulation, draft adoption prediction, and representative statement generation. These tasks span the three stages of the UN decision-making process--drafting, voting, and discussing--and aim to assess LLMs' ability to understand and simulate political dynamics. Our experimental analysis demonstrates the potential and challenges of applying LLMs in this domain, providing insights into their strengths and limitations in political science. This work contributes to the growing intersection of AI and political science, opening new avenues for research and practical applications in global governance. The UNBench Repository can be accessed at: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在自然语言处理方面取得了显著的进步，但它们在高风险政治决策中的潜力尚未得到充分探索。本文通过聚焦于将其应用于联合国（UN）的决策过程来填补这一空白，其中的风险尤为高企，政治决策往往具有深远的后果。我们介绍了一个新的数据集，该数据集包含从1994年到2024年公开发布的联合国安全理事会（UNSC）记录，包括提案决议、投票记录和外交演讲。利用这一数据集，我们提出了联合国基准测试（UNBench），这是首个旨在全面评估LLMs在四个相互关联的政治理论任务上的基准：共同提案人判断、代表投票模拟、提案采纳预测和代表发言生成。这些任务涵盖了联合国决策过程的三个阶段——起草、投票和讨论——旨在评估LLMs理解和模拟政治动态的能力。我们的实验分析展示了在这一领域应用LLMs的潜力和挑战，并提供了其在政治科学中的优势和局限性方面的见解。本文为人工智能与政治科学日益增长的交叉领域做出了贡献，并为全球治理的研究和实际应用开辟了新的途径。联合国基准测试库（UNBench Repository）可以在以下网址访问：this https URL。 

---
# Meaning Beyond Truth Conditions: Evaluating Discourse Level Understanding via Anaphora Accessibility 

**Title (ZH)**: 超越真值条件的意义：通过消解指称可及性评估话语层面理解 

**Authors**: Xiaomeng Zhu, Zhenghao Zhou, Simon Charlow, Robert Frank  

**Link**: [PDF](https://arxiv.org/pdf/2502.14119)  

**Abstract**: We present a hierarchy of natural language understanding abilities and argue for the importance of moving beyond assessments of understanding at the lexical and sentence levels to the discourse level. We propose the task of anaphora accessibility as a diagnostic for assessing discourse understanding, and to this end, present an evaluation dataset inspired by theoretical research in dynamic semantics. We evaluate human and LLM performance on our dataset and find that LLMs and humans align on some tasks and diverge on others. Such divergence can be explained by LLMs' reliance on specific lexical items during language comprehension, in contrast to human sensitivity to structural abstractions. 

**Abstract (ZH)**: 我们提出了一种自然语言理解能力的层次结构，并强调超越词级和句级的理解评估，转向对话语层面的理解评估的重要性。我们提议使用消解可访问性任务作为评估话语理解的诊断工具，并为此目的提出了一个受到动态语义理论研究启发的评估数据集。我们评估了人类和大语言模型（LLM）在这数据集上的表现，并发现两者在某些任务上表现出一致性，而在其他任务上则存在分歧。这种分歧可以由大语言模型在语言理解过程中对特定词汇项目的依赖性来解释，与人类对结构抽象的敏感性形成对比。 

---
# Towards Context-Robust LLMs: A Gated Representation Fine-tuning Approach 

**Title (ZH)**: 面向上下文稳健的大型语言模型：一种门控表示微调方法 

**Authors**: Shenglai Zeng, Pengfei He, Kai Guo, Tianqi Zheng, Hanqing Lu, Yue Xing, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14100)  

**Abstract**: Large Language Models (LLMs) enhanced with external contexts, such as through retrieval-augmented generation (RAG), often face challenges in handling imperfect evidence. They tend to over-rely on external knowledge, making them vulnerable to misleading and unhelpful contexts. To address this, we propose the concept of context-robust LLMs, which can effectively balance internal knowledge with external context, similar to human cognitive processes. Specifically, context-robust LLMs should rely on external context only when lacking internal knowledge, identify contradictions between internal and external knowledge, and disregard unhelpful contexts. To achieve this goal, we introduce Grft, a lightweight and plug-and-play gated representation fine-tuning approach. Grft consists of two key components: a gating mechanism to detect and filter problematic inputs, and low-rank representation adapters to adjust hidden representations. By training a lightweight intervention function with only 0.0004\% of model size on fewer than 200 examples, Grft can effectively adapt LLMs towards context-robust behaviors. 

**Abstract (ZH)**: 增强外部上下文的大型语言模型（LLMs），如通过检索增强生成（RAG）进行增强，往往在处理不完善证据时面临挑战。它们倾向于过度依赖外部知识，使其容易受到误导性和无用上下文的影响。为了解决这一问题，我们提出了上下文稳健型LLMs的概念，这种模型能够有效地平衡内部知识与外部上下文，类似于人类的认知过程。具体而言，上下文稳健型LLMs应在缺乏内部知识时依赖外部上下文，识别内部知识与外部知识之间的矛盾，并忽视无用的上下文。

为了实现这一目标，我们引入了一种轻量级且即插即用的门控表示微调方法Grft。Grft包含两个关键组件：一个门控机制用于检测并过滤有问题的输入，以及低秩表示适配器以调整隐藏表示。通过在少于200个样本上训练一个仅占模型大小0.0004%的轻量级干预函数，Grft可以有效地使LLMs朝着上下文稳健型行为进行适应。 

---
# Retrieving Versus Understanding Extractive Evidence in Few-Shot Learning 

**Title (ZH)**: 在少样本学习中检索 versus 理解提取证据 

**Authors**: Karl Elbakian, Samuel Carton  

**Link**: [PDF](https://arxiv.org/pdf/2502.14095)  

**Abstract**: A key aspect of alignment is the proper use of within-document evidence to construct document-level decisions. We analyze the relationship between the retrieval and interpretation of within-document evidence for large language model in a few-shot setting. Specifically, we measure the extent to which model prediction errors are associated with evidence retrieval errors with respect to gold-standard human-annotated extractive evidence for five datasets, using two popular closed proprietary models. We perform two ablation studies to investigate when both label prediction and evidence retrieval errors can be attributed to qualities of the relevant evidence. We find that there is a strong empirical relationship between model prediction and evidence retrieval error, but that evidence retrieval error is mostly not associated with evidence interpretation error--a hopeful sign for downstream applications built on this mechanism. 

**Abstract (ZH)**: 对齐的一个关键方面是正确使用文档内的证据来构建文档级决策。我们在少量示例的设置下分析大型语言模型在检索和解释文档内证据之间的关系。具体而言，我们使用两个流行的商业闭源模型，针对五个数据集，测量模型预测错误与黄金标准的人工标注提取证据之间的关联程度，从而评估证据检索错误的程度。我们进行了两项消融研究，以探讨在哪些情况下标签预测和证据检索错误都可以归因于相关证据的质量。我们发现，模型预测错误与证据检索错误之间存在明显的经验关联，但证据检索错误与证据解释错误之间关联不大——这是一个对未来基于这一机制的应用充满希望的信号。 

---
# Navigating Semantic Relations: Challenges for Language Models in Abstract Common-Sense Reasoning 

**Title (ZH)**: 导航语义关系：语言模型在抽象常识推理中的挑战 

**Authors**: Cole Gawin, Yidan Sun, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2502.14086)  

**Abstract**: Large language models (LLMs) have achieved remarkable performance in generating human-like text and solving reasoning tasks of moderate complexity, such as question-answering and mathematical problem-solving. However, their capabilities in tasks requiring deeper cognitive skills, such as common-sense understanding and abstract reasoning, remain under-explored. In this paper, we systematically evaluate abstract common-sense reasoning in LLMs using the ConceptNet knowledge graph. We propose two prompting approaches: instruct prompting, where models predict plausible semantic relationships based on provided definitions, and few-shot prompting, where models identify relations using examples as guidance. Our experiments with the gpt-4o-mini model show that in instruct prompting, consistent performance is obtained when ranking multiple relations but with substantial decline when the model is restricted to predicting only one relation. In few-shot prompting, the model's accuracy improves significantly when selecting from five relations rather than the full set, although with notable bias toward certain relations. These results suggest significant gaps still, even in commercially used LLMs' abstract common-sense reasoning abilities, compared to human-level understanding. However, the findings also highlight the promise of careful prompt engineering, based on selective retrieval, for obtaining better performance. 

**Abstract (ZH)**: 大型语言模型（LLMs）在生成类人类文本和解决中等复杂度的推理任务（如问答和数学问题解决）方面取得了显著成果。然而，它们在需要更深层次认知技能的任务（如常识理解和抽象推理）中的能力仍然未得到充分探索。本文系统性地使用ConceptNet知识图谱评估LLMs的抽象常识推理能力。我们提出了两种提示方法：指令提示（instruct prompting），在这种提示中模型根据提供的定义预测可能的语义关系；少量样本提示（few-shot prompting），在这种提示中模型根据示例指导识别关系。实验结果显示，使用gpt-4o-mini模型，在指令提示中，当对多种关系进行排名时，能够获得一致的表现，但在仅预测单一关系时，表现有显著下降。在少量样本提示中，当从五个关系中选择时模型的准确性有了显著提高，尽管存在对某些关系的明显偏好。这些结果表明，即使是商用LLMs在抽象常识推理能力方面仍然与人类水平的理解存在显著差距。然而，实验结果也表明，基于选择性检索的精细提示工程在获得更好性能方面具有巨大的潜力。 

---
# Are Rules Meant to be Broken? Understanding Multilingual Moral Reasoning as a Computational Pipeline with UniMoral 

**Title (ZH)**: 《规则是不是用来被打破的？通过UniMoral理解多语言道德推理的计算管道》

这个翻译既保持了原文的学术风格，又符合中文的表达习惯。"UniMoral"被保留为专有名词，未进行翻译。如果您需要进一步的调整或其他帮助，请随时告知。 

**Authors**: Shivani Kumar, David Jurgens  

**Link**: [PDF](https://arxiv.org/pdf/2502.14083)  

**Abstract**: Moral reasoning is a complex cognitive process shaped by individual experiences and cultural contexts and presents unique challenges for computational analysis. While natural language processing (NLP) offers promising tools for studying this phenomenon, current research lacks cohesion, employing discordant datasets and tasks that examine isolated aspects of moral reasoning. We bridge this gap with UniMoral, a unified dataset integrating psychologically grounded and social-media-derived moral dilemmas annotated with labels for action choices, ethical principles, contributing factors, and consequences, alongside annotators' moral and cultural profiles. Recognizing the cultural relativity of moral reasoning, UniMoral spans six languages, Arabic, Chinese, English, Hindi, Russian, and Spanish, capturing diverse socio-cultural contexts. We demonstrate UniMoral's utility through a benchmark evaluations of three large language models (LLMs) across four tasks: action prediction, moral typology classification, factor attribution analysis, and consequence generation. Key findings reveal that while implicitly embedded moral contexts enhance the moral reasoning capability of LLMs, there remains a critical need for increasingly specialized approaches to further advance moral reasoning in these models. 

**Abstract (ZH)**: 道德推理是一种受个体经验和文化背景影响的复杂认知过程，为计算分析带来了独特的挑战。尽管自然语言处理（NLP）提供了研究这一现象的有前景的工具，但当前的研究缺乏连贯性，使用了不一致的数据集和任务，这些任务仅考察了道德推理的孤立方面。为此，我们通过建立UniMoral这一统一数据集来填补这一空白。UniMoral涵盖了基于心理研究和社交媒体衍生的道德难题，并对行动选择、伦理原则、影响因素、后果进行了标注，同时还记录了标注人员的道德和文化背景。我们认识到道德推理具有文化相对性，因此UniMoral覆盖了六种语言——阿拉伯语、中文、英语、印地语、俄语和西班牙语，以捕捉多样的社会文化背景。我们通过三项大型语言模型（LLMs）在四项任务中的基准评估，展示了UniMoral的应用价值：行动预测、道德类型分类、影响因素分析和后果生成。关键发现表明，虽然隐含嵌入的道德背景能够增强LLMs的道德推理能力，但仍需进一步发展专门的方法来进一步推动这些模型中的道德推理能力。 

---
# RocketKV: Accelerating Long-Context LLM Inference via Two-Stage KV Cache Compression 

**Title (ZH)**: RocketKV：通过两阶段键值缓存压缩加速长上下文LLM推断 

**Authors**: Payman Behnam, Yaosheng Fu, Ritchie Zhao, Po-An Tsai, Zhiding Yu, Alexey Tumanov  

**Link**: [PDF](https://arxiv.org/pdf/2502.14051)  

**Abstract**: Transformer-based Large Language Models rely critically on KV cache to efficiently handle extended contexts during the decode phase. Yet, the size of the KV cache grows proportionally with the input length, burdening both memory bandwidth and capacity as decoding progresses. To address this challenge, we present RocketKV, a training-free KV cache compression strategy designed specifically to reduce both memory bandwidth and capacity demand of KV cache during the decode phase. RocketKV contains two consecutive stages. In the first stage, it performs coarse-grain KV cache eviction on the input sequence tokens with SnapKV++, a method improved upon SnapKV by introducing adaptive pooling size and full compatibility with grouped-query attention. In the second stage, it adopts a hybrid attention method to conduct fine-grain top-k sparse attention, approximating the attention scores by leveraging both head and sequence dimensional reductions. Combining these two stages, RocketKV achieves significant KV cache fetching bandwidth and storage savings while maintaining comparable accuracy to full KV cache attention. We show that RocketKV provides end-to-end speedup by up to 3$\times$ as well as peak memory reduction by up to 31% in the decode phase on an NVIDIA H100 GPU compared to the full KV cache baseline, while achieving negligible accuracy loss on a variety of long-context tasks. 

**Abstract (ZH)**: 基于Transformer的大语言模型在解码阶段高效处理扩展上下文时高度依赖于键值（KV）缓存。然而，随着输入长度的增加，KV缓存的大小也会相应增长，对内存带宽和容量造成负担。为解决这一挑战，我们提出了RocketKV，这是一种无需训练的KV缓存压缩策略，旨在减轻解码过程中KV缓存对内存带宽和容量的需求。RocketKV包含两个连续的阶段。在第一个阶段，它使用SnapKV++对输入序列令牌进行粗粒度的KV缓存淘汰，SnapKV++是一种改进的SnapKV方法，引入了自适应池化大小，并且与分组查询注意力完全兼容。在第二个阶段，它采用一种混合注意力方法来执行细粒度的top-k稀疏注意力，利用头部和序列维度减少来近似注意力分数。结合这两个阶段，RocketKV实现了显著的KV缓存获取带宽和存储节省，同时保持与完整KV缓存注意力相当的准确性。实验结果显示，与全KV缓存基线相比，借助于NVIDIA H100 GPU，RocketKV在解码阶段提供了高达3倍的端到端加速，并且峰值内存减少可达31%，同时在多种长上下文任务中实现了可忽略的准确性损失。 

---
# Diversity-driven Data Selection for Language Model Tuning through Sparse Autoencoder 

**Title (ZH)**: 通过稀疏自编码器驱动的多样性数据选择方法用于语言模型调优 

**Authors**: Xianjun Yang, Shaoliang Nie, Lijuan Liu, Suchin Gururangan, Ujjwal Karn, Rui Hou, Madian Khabsa, Yuning Mao  

**Link**: [PDF](https://arxiv.org/pdf/2502.14050)  

**Abstract**: Current pre-trained large language models typically need instruction tuning to align with human preferences. However, instruction tuning data is often quantity-saturated due to the large volume of data collection and fast model iteration, leaving coreset data selection important but underexplored. On the other hand, existing quality-driven data selection methods such as LIMA (NeurIPS 2023 (Zhou et al., 2024)) and AlpaGasus (ICLR 2024 (Chen et al.)) generally ignore the equal importance of data diversity and complexity. In this work, we aim to design a diversity-aware data selection strategy and creatively propose using sparse autoencoders to tackle the challenge of data diversity measure. In addition, sparse autoencoders can also provide more interpretability of model behavior and explain, e.g., the surprising effectiveness of selecting the longest response (ICML 2024 (Zhao et al.)). Using effective data selection, we experimentally prove that models trained on our selected data can outperform other methods in terms of model capabilities, reduce training cost, and potentially gain more control over model behaviors. 

**Abstract (ZH)**: 当前的预训练大型语言模型通常需要指令调优以与人类偏好对齐。然而，由于数据采集量大和模型迭代速度快，指令调优数据往往存在数量饱和的问题，使得核心集数据选择变得重要但未被充分探索。另一方面，现有的质量驱动的数据选择方法，如LIMA（NeurIPS 2023，Zhou et al., 2024）和AlpaGasus（ICLR 2024，Chen et al.），通常忽略了数据多样性和复杂性的同等重要性。在本文中，我们旨在设计一种意识多样性数据选择策略，并创造性地提出使用稀疏自编码器来应对数据多样性度量的挑战。此外，稀疏自编码器还能提供模型行为的更多可解释性，解释例如选择最长响应的惊人效果（ICML 2024，Zhao et al.）。通过有效数据选择，我们实验证明，使用我们选择的数据训练的模型在模型能力上优于其他方法，减少训练成本，并可能获得对模型行为的更多控制。 

---
# Semantic Decomposition and Selective Context Filtering -- Text Processing Techniques for Context-Aware NLP-Based Systems 

**Title (ZH)**: 语义分解与选择性背景过滤——基于上下文感知的自然语言处理系统中的文本处理技术 

**Authors**: Karl John Villardar  

**Link**: [PDF](https://arxiv.org/pdf/2502.14048)  

**Abstract**: In this paper, we present two techniques for use in context-aware systems: Semantic Decomposition, which sequentially decomposes input prompts into a structured and hierarchal information schema in which systems can parse and process easily, and Selective Context Filtering, which enables systems to systematically filter out specific irrelevant sections of contextual information that is fed through a system's NLP-based pipeline. We will explore how context-aware systems and applications can utilize these two techniques in order to implement dynamic LLM-to-system interfaces, improve an LLM's ability to generate more contextually cohesive user-facing responses, and optimize complex automated workflows and pipelines. 

**Abstract (ZH)**: 在本文中，我们提出了两种用于上下文感知系统的技术：语义分解和选择性上下文过滤。语义分解是一种序列化分解输入提示的方法，将其分解为一个结构化和分层的信息模式，使系统能够轻松解析和处理这些信息。选择性上下文过滤则允许系统有系统地筛选出特定的无关上下文信息，这些信息通过系统的基于自然语言处理（NLP）的管道输入。我们将探讨如何利用这两种技术，以实现动态的大规模语言模型（LLM）到系统接口，提高LLM生成更具上下文一致性的用户响应的能力，并优化复杂的自动化工作流和管道。 

---
# DiffSampling: Enhancing Diversity and Accuracy in Neural Text Generation 

**Title (ZH)**: DiffSampling：增强神经文本生成中的多样性和准确性 

**Authors**: Giorgio Franceschelli, Mirco Musolesi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14037)  

**Abstract**: Despite their increasing performance, large language models still tend to reproduce training data, generate several repetitions, and focus on the most common grammatical structures and words. A possible cause is the decoding strategy adopted: the most common ones either consider only the most probable tokens, reducing output diversity, or increase the likelihood of unlikely tokens at the cost of output accuracy and correctness. In this paper, we propose a family of three new decoding methods by leveraging a mathematical analysis of the token probability distribution. In particular, the difference between consecutive, sorted probabilities can be used to avoid incorrect tokens and increase the chance of low-probable but accurate words. Experiments concerning math problem solving, extreme summarization, and the divergent association task show that our approach consistently performs at least as well as current alternatives in terms of quality and diversity. 

**Abstract (ZH)**: 尽管大型语言模型的性能不断提高，但仍倾向于复制训练数据、生成多次重复，并且倾向于使用最常用的语法结构和词汇。可能的原因之一是所采用的解码策略：最常见的策略要么只考虑最可能的令牌，从而减少了输出的多样性，要么增加了不太可能的令牌的可能性，最终牺牲了输出的准确性和正确性。在这篇论文中，我们提出了一种新的解码方法，该方法通过利用令牌概率分布的数学分析而实现。具体而言，连续排序概率之间的差异可用于避免错误的令牌并增加低概率但准确的单词的出现机会。关于数学问题求解、极端总结和发散联想任务的实验表明，我们的方法在质量和多样性方面至少与当前的替代方法一样好。 

---
# Dehumanizing Machines: Mitigating Anthropomorphic Behaviors in Text Generation Systems 

**Title (ZH)**: 机器去人性化：减轻文本生成系统中的拟人化行为 

**Authors**: Myra Cheng, Su Lin Blodgett, Alicia DeVrio, Lisa Egede, Alexandra Olteanu  

**Link**: [PDF](https://arxiv.org/pdf/2502.14019)  

**Abstract**: As text generation systems' outputs are increasingly anthropomorphic -- perceived as human-like -- scholars have also raised increasing concerns about how such outputs can lead to harmful outcomes, such as users over-relying or developing emotional dependence on these systems. How to intervene on such system outputs to mitigate anthropomorphic behaviors and their attendant harmful outcomes, however, remains understudied. With this work, we aim to provide empirical and theoretical grounding for developing such interventions. To do so, we compile an inventory of interventions grounded both in prior literature and a crowdsourced study where participants edited system outputs to make them less human-like. Drawing on this inventory, we also develop a conceptual framework to help characterize the landscape of possible interventions, articulate distinctions between different types of interventions, and provide a theoretical basis for evaluating the effectiveness of different interventions. 

**Abstract (ZH)**: 随着文本生成系统输出的内容越来越拟人化——即被感知为类似人类的——学者们也对这些输出可能导致的负面影响提出了越来越多的担忧，例如用户过度依赖或对这些系统产生情感依赖。然而，如何干预这些系统输出，以减少拟人化行为及其相关负面影响，仍缺乏研究。通过本研究，我们旨在为开发此类干预措施提供实证和理论支持。为此，我们编制了一份基于先前文献和一项众包研究的干预措施清单。在众包研究中，参与者编辑了系统输出，使其更加非拟人化。基于这份清单，我们还开发了一个概念框架，以帮助描绘可能的干预措施的景观、区分不同类型的干预措施，并为评估不同干预措施的有效性提供理论基础。 

---
# MaskPrune: Mask-based LLM Pruning for Layer-wise Uniform Structures 

**Title (ZH)**: MaskPrune：基于掩码的层内均匀结构的大语言模型剪枝 

**Authors**: Jiayu Qin, Jianchao Tan, Kefeng Zhang, Xunliang Cai, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.14008)  

**Abstract**: The remarkable performance of large language models (LLMs) in various language tasks has attracted considerable attention. However, the ever-increasing size of these models presents growing challenges for deployment and inference. Structured pruning, an effective model compression technique, is gaining increasing attention due to its ability to enhance inference efficiency. Nevertheless, most previous optimization-based structured pruning methods sacrifice the uniform structure across layers for greater flexibility to maintain performance. The heterogeneous structure hinders the effective utilization of off-the-shelf inference acceleration techniques and impedes efficient configuration for continued training. To address this issue, we propose a novel masking learning paradigm based on minimax optimization to obtain the uniform pruned structure by optimizing the masks under sparsity regularization. Extensive experimental results demonstrate that our method can maintain high performance while ensuring the uniformity of the pruned model structure, thereby outperforming existing SOTA methods. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种语言任务中的出色表现引起了广泛关注。然而，这些模型的不断增大使其实现和推理部署面临越来越大的挑战。结构化剪枝作为一种有效的模型压缩技术，因其能提高推理效率而逐渐受到关注。然而，大多数基于优化的结构化剪枝方法为了获得更大的灵活性而在各层之间牺牲了一致性结构，这阻碍了现成的推理加速技术的有效应用，并妨碍了持续训练的高效配置。为了解决这个问题，我们提出了一种基于极小极大优化的新型掩码学习范式，通过在稀疏正则化下优化掩码来获得一致的剪枝结构。广泛的经验分析结果表明，我们的方法能够在保持高性能的同时确保剪枝模型结构的一致性，从而优于现有最优方法。 

---
# Prompt-to-Leaderboard 

**Title (ZH)**: "Prompt-to-Leaderboard" 可以翻译为学术规范的中文标题为：“从提示到排行榜”。这个标题通常用于描述一种方法或系统，能够根据给定的提示生成或评估模型的表现，并将其结果提交到排行榜上进行比较和展示。 

**Authors**: Evan Frick, Connor Chen, Joseph Tennyson, Tianle Li, Wei-Lin Chiang, Anastasios N. Angelopoulos, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2502.14855)  

**Abstract**: Large language model (LLM) evaluations typically rely on aggregated metrics like accuracy or human preference, averaging across users and prompts. This averaging obscures user- and prompt-specific variations in model performance. To address this, we propose Prompt-to-Leaderboard (P2L), a method that produces leaderboards specific to a prompt. The core idea is to train an LLM taking natural language prompts as input to output a vector of Bradley-Terry coefficients which are then used to predict the human preference vote. The resulting prompt-dependent leaderboards allow for unsupervised task-specific evaluation, optimal routing of queries to models, personalization, and automated evaluation of model strengths and weaknesses. Data from Chatbot Arena suggest that P2L better captures the nuanced landscape of language model performance than the averaged leaderboard. Furthermore, our findings suggest that P2L's ability to produce prompt-specific evaluations follows a power law scaling similar to that observed in LLMs themselves. In January 2025, the router we trained based on this methodology achieved the \#1 spot in the Chatbot Arena leaderboard. Our code is available at this GitHub link: this https URL. 

**Abstract (ZH)**: 以下是论文内容或标题的中文翻译，符合学术规范：

大型语言模型（LLM）的评估通常依赖于聚合指标，如准确率或人类偏好，这些指标是通过对多个用户和提示进行平均计算得出的。这种平均计算掩盖了不同用户和提示下模型性能的具体差异。为了解决这一问题，我们提出了Prompt-to-Leaderboard（P2L）方法，该方法能够生成针对特定提示的排行榜。核心思路是训练一个LLM，使其以自然语言提示作为输入，输出布拉德利-特里系数向量，然后使用这些系数来预测人类偏好投票。由此产生的提示依赖式排行榜使得无需监督即可进行任务特定评估、优化模型查询路由、个性化以及自动化评估模型的优势和劣势。Chatbot Arena的数据表明，P2L更能捕捉语言模型性能的细微差异，其排行榜表现优于平均排行榜。此外，我们的研究发现表明，P2L生成提示特定评估的能力与LLM本身的幂律缩放类似。在2025年1月，我们基于该方法训练的路由器在Chatbot Arena排行榜上取得了第一名的成绩。我们的代码可在以下GitHub链接访问：this https URL。 

---
# Scaling Text-Rich Image Understanding via Code-Guided Synthetic Multimodal Data Generation 

**Title (ZH)**: 通过代码引导的合成多模态数据生成实现丰富的文本图像理解扩展示lastname: 

**Authors**: Yue Yang, Ajay Patel, Matt Deitke, Tanmay Gupta, Luca Weihs, Andrew Head, Mark Yatskar, Chris Callison-Burch, Ranjay Krishna, Aniruddha Kembhavi, Christopher Clark  

**Link**: [PDF](https://arxiv.org/pdf/2502.14846)  

**Abstract**: Reasoning about images with rich text, such as charts and documents, is a critical application of vision-language models (VLMs). However, VLMs often struggle in these domains due to the scarcity of diverse text-rich vision-language data. To address this challenge, we present CoSyn, a framework that leverages the coding capabilities of text-only large language models (LLMs) to automatically create synthetic text-rich multimodal data. Given input text describing a target domain (e.g., "nutrition fact labels"), CoSyn prompts an LLM to generate code (Python, HTML, LaTeX, etc.) for rendering synthetic images. With the underlying code as textual representations of the synthetic images, CoSyn can generate high-quality instruction-tuning data, again relying on a text-only LLM. Using CoSyn, we constructed a dataset comprising 400K images and 2.7M rows of vision-language instruction-tuning data. Comprehensive experiments on seven benchmarks demonstrate that models trained on our synthetic data achieve state-of-the-art performance among competitive open-source models, including Llama 3.2, and surpass proprietary models such as GPT-4V and Gemini 1.5 Flash. Furthermore, CoSyn can produce synthetic pointing data, enabling VLMs to ground information within input images, showcasing its potential for developing multimodal agents capable of acting in real-world environments. 

**Abstract (ZH)**: 利用丰富的文本（如图表和文档）对图像进行推理是视觉-语言模型（VLMs）的一个关键应用。然而，由于这类领域中多样化文本丰富视觉数据的稀缺性，VLMs 往往难以应对这些挑战。为了解决这一问题，我们提出了一种名为 CoSyn 的框架，该框架利用仅文本大型语言模型（LLMs）的编码能力，自动创建合成的文本丰富的多模态数据。给定描述目标领域的输入文本（例如，“营养成分标签”），CoSyn 可以促使 LLM 生成用于渲染合成图像的代码（如 Python、HTML、LaTeX 等）。通过将底层代码作为合成图像的文本表示，CoSyn 可以生成高质量的指令调优数据，依赖于仅文本的 LLM。使用 CoSyn，我们构建了一个包含 40 万张图像和 270 万行视觉-语言指令调优数据的数据集。在七个基准上的全面实验表明，在我们的合成数据上训练的模型在竞争性的开源模型（包括 Llama 3.2）中达到了最先进的性能，甚至超过了专有模型 GPT-4V 和 Gemini 1.5 Flash。此外，CoSyn 还可以生成合成的指示数据，使 VLMs 能够在输入图像中定位信息，展示了其在开发能够实现在真实世界环境中行动的多模态代理方面的潜力。 

---
# LongWriter-V: Enabling Ultra-Long and High-Fidelity Generation in Vision-Language Models 

**Title (ZH)**: 长写作家-V：.enable 在视觉语言模型中实现超长高保真生成 

**Authors**: Shangqing Tu, Yucheng Wang, Daniel Zhang-Li, Yushi Bai, Jifan Yu, Yuhao Wu, Lei Hou, Huiqin Liu, Zhiyuan Liu, Bin Xu, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.14834)  

**Abstract**: Existing Large Vision-Language Models (LVLMs) can process inputs with context lengths up to 128k visual and text tokens, yet they struggle to generate coherent outputs beyond 1,000 words. We find that the primary limitation is the absence of long output examples during supervised fine-tuning (SFT). To tackle this issue, we introduce LongWriter-V-22k, a SFT dataset comprising 22,158 examples, each with multiple input images, an instruction, and corresponding outputs ranging from 0 to 10,000 words. Moreover, to achieve long outputs that maintain high-fidelity to the input images, we employ Direct Preference Optimization (DPO) to the SFT model. Given the high cost of collecting human feedback for lengthy outputs (e.g., 3,000 words), we propose IterDPO, which breaks long outputs into segments and uses iterative corrections to form preference pairs with the original outputs. Additionally, we develop MMLongBench-Write, a benchmark featuring six tasks to evaluate the long-generation capabilities of VLMs. Our 7B parameter model, trained with LongWriter-V-22k and IterDPO, achieves impressive performance on this benchmark, outperforming larger proprietary models like GPT-4o. Code and data: this https URL 

**Abstract (ZH)**: 现有的大规模视觉-语言模型（LVLMs）能够处理多达128k视觉和文本令牌的输入，但在生成超过1,000个单词的连贯输出方面存在困难。我们发现主要的限制在于监督微调（SFT）期间缺乏长输出示例。为了解决这一问题，我们引入了LongWriter-V-22k，这是一个SFT数据集，包含22,158个例子，每个例子包含多个输入图像、指令以及长度从0到10,000个单词的相应输出。此外，为了生成与输入图像保真度高的长输出，我们在SFT模型中采用了直接偏好优化（DPO）。鉴于收集长输出（如3,000个单词）的人类反馈成本高昂，我们提出了一种名为IterDPO的方法，将长输出分成段落，并通过迭代修正形成与原始输出的偏好对。此外，我们开发了MMLongBench-Write基准测试，其中包括六个任务以评估VLMs的长生成能力。我们的7B参数模型，在使用LongWriter-V-22k和IterDPO训练后，在这一基准测试中的表现令人印象深刻，超越了诸如GPT-4o等更大的专用模型。代码和数据：[此链接] 

---
# Optimizing Model Selection for Compound AI Systems 

**Title (ZH)**: 优化复合人工智能系统中的模型选择 

**Authors**: Lingjiao Chen, Jared Quincy Davis, Boris Hanin, Peter Bailis, Matei Zaharia, James Zou, Ion Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2502.14815)  

**Abstract**: Compound AI systems that combine multiple LLM calls, such as self-refine and multi-agent-debate, achieve strong performance on many AI tasks. We address a core question in optimizing compound systems: for each LLM call or module in the system, how should one decide which LLM to use? We show that these LLM choices have a large effect on quality, but the search space is exponential. We propose LLMSelector, an efficient framework for model selection in compound systems, which leverages two key empirical insights: (i) end-to-end performance is often monotonic in how well each module performs, with all other modules held fixed, and (ii) per-module performance can be estimated accurately by an LLM. Building upon these insights, LLMSelector iteratively selects one module and allocates to it the model with the highest module-wise performance, as estimated by an LLM, until no further gain is possible. LLMSelector is applicable to any compound system with a bounded number of modules, and its number of API calls scales linearly with the number of modules, achieving high-quality model allocation both empirically and theoretically. Experiments with popular compound systems such as multi-agent debate and self-refine using LLMs such as GPT-4o, Claude 3.5 Sonnet and Gemini 1.5 show that LLMSelector confers 5%-70% accuracy gains compared to using the same LLM for all modules. 

**Abstract (ZH)**: 结合多个大语言模型（LLM）调用的复合AI系统，如自我完善和多代理辩论，已经在许多AI任务中表现出色。本文解决了一个核心问题：在系统中的每一个LLM调用或模块应该如何决定使用哪个LLM？我们展示了这些LLM选择对质量有重大影响，但是搜索空间是指数级的。我们提出了LLMSelector，这是一个用于复合系统中模型选择的有效框架，该框架利用了两个关键的实证洞察：（i）端到端性能通常随着每个模块表现的提高而单调增加，其他模块保持不变；（ii）每个模块的表现可以通过一个LLM准确估计。基于这些洞察，LLMSelector 会迭代地选择一个模块，并为其分配根据LLM估计性能最佳的模型，直到无法进一步提高为止。LLMSelector 可应用于模块数量有上限的任何复合系统，并且其API调用的数量与模块数量线性相关，在实践和理论上都能实现高质量的模型分配。使用流行的复合系统如多代理辩论和自我完善，以及LLM如GPT-4o、Claude 3.5 Sonnet和Gemini 1.5进行的实验表明，与所有模块都使用同一个LLM相比，LLMSelector 可以获得5%到70%的准确性提升。 

---
# From Knowledge Generation to Knowledge Verification: Examining the BioMedical Generative Capabilities of ChatGPT 

**Title (ZH)**: 从知识生成到知识验证：探究ChatGPT的生物医学生成能力 

**Authors**: Ahmed Abdeen Hamed, Byung Suk Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.14714)  

**Abstract**: The generative capabilities of LLM models present opportunities in accelerating tasks and concerns with the authenticity of the knowledge it produces. To address the concerns, we present a computational approach that systematically evaluates the factual accuracy of biomedical knowledge that an LLM model has been prompted to generate. Our approach encompasses two processes: the generation of disease-centric associations and the verification of them using the semantic knowledge of the biomedical ontologies. Using ChatGPT as the select LLM model, we designed a set of prompt-engineering processes to generate linkages between diseases, drugs, symptoms, and genes to establish grounds for assessments. Experimental results demonstrate high accuracy in identifying disease terms (88%-97%), drug names (90%-91%), and genetic information (88%-98%). The symptom term identification accuracy was notably lower (49%-61%), as verified against the DOID, ChEBI, SYMPTOM, and GO ontologies accordingly. The verification of associations reveals literature coverage rates of (89%-91%) among disease-drug and disease-gene associations. The low identification accuracy for symptom terms also contributed to the verification of symptom-related associations (49%-62%). 

**Abstract (ZH)**: 大型语言模型（LLM）的生成能力为加速任务提供了机会，同时也引发了对其生成知识真实性方面的担忧。为应对这些担忧，我们提出了一种计算方法，系统地评估LLM模型生成的生物医学知识的准确性。该方法包含两个过程：疾病中心关联的生成和通过生物医学本体的语义知识验证这些关联。我们选用ChatGPT作为LLM模型，设计了一套提示工程过程，生成疾病、药物、症状和基因之间的联系，为评估奠定了基础。实验结果表明，在识别疾病术语（88%-97%）、药物名称（90%-91%）和遗传信息（88%-98%）方面具有较高的准确性。症状术语的识别准确率较低（49%-61%），经过DOID、ChEBI、SYMPTOM和GO本体的验证。关联的验证显示，疾病-药物和疾病-基因关联的文献覆盖率为（89%-91%）。症状术语识别准确率的较低也影响了与其相关的关联验证结果（49%-62%）。 

---
# PEARL: Towards Permutation-Resilient LLMs 

**Title (ZH)**: PEARL：朝着抗置换鲁棒的大型语言模型方向发展 

**Authors**: Liang Chen, Li Shen, Yang Deng, Xiaoyan Zhao, Bin Liang, Kam-Fai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2502.14628)  

**Abstract**: The in-context learning (ICL) capability of large language models (LLMs) enables them to perform challenging tasks using provided demonstrations. However, ICL is highly sensitive to the ordering of demonstrations, leading to instability in predictions. This paper shows that this vulnerability can be exploited to design a natural attack - difficult for model providers to detect - that achieves nearly 80% success rate on LLaMA-3 by simply permuting the demonstrations. Existing mitigation methods primarily rely on post-processing and fail to enhance the model's inherent robustness to input permutations, raising concerns about safety and reliability of LLMs. To address this issue, we propose Permutation-resilient learning (PEARL), a novel framework based on distributionally robust optimization (DRO), which optimizes model performance against the worst-case input permutation. Specifically, PEARL consists of a permutation-proposal network (P-Net) and the LLM. The P-Net generates the most challenging permutations by treating it as an optimal transport problem, which is solved using an entropy-constrained Sinkhorn algorithm. Through minimax optimization, the P-Net and the LLM iteratively optimize against each other, progressively improving the LLM's robustness. Experiments on synthetic pre-training and real-world instruction tuning tasks demonstrate that PEARL effectively mitigates permutation attacks and enhances performance. Notably, despite being trained on fewer shots and shorter contexts, PEARL achieves performance gains of up to 40% when scaled to many-shot and long-context scenarios, highlighting its efficiency and generalization capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）的上下文内学习（ICL）能力使它们能够利用提供的示范来完成具有挑战性的任务。然而，ICL 对示范的顺序极为敏感，导致预测结果不稳定。本文表明，这一脆弱性可用于设计一种自然攻击——对于模型供应商难以检测，且能够在通过简单地重新排列示范使得 LLaMA-3 达到接近 80% 的成功率。现有的缓解方法主要依赖于后处理，并未能增强模型对输入排列的固有鲁棒性，这引发了对LLMs 安全性和可靠性的担忧。为解决这一问题，我们提出了一种基于分布鲁棒优化（DRO）的新型框架——Permutation-resilient learning (PEARL)，该框架旨在优化模型在最坏输入排列情况下的性能。具体而言，PEARL 包括一个排列建议网络（P-Net）和 LLM。P-Net 将排列问题视为一个最优传输问题，并通过熵约束的 Sinkhorn 算法进行求解。通过最小极大优化，P-Net 和 LLM 相互优化，逐步提高LLM 的鲁棒性。在合成预训练和现实世界的指令调优任务上的实验表明，PEARL 有效缓解了排列攻击并提高了性能。值得注意的是，尽管在少量示例和较短上下文中进行训练，当扩展到多示例和长上下文场景时，PEARL 能够实现高达 40% 的性能提升，这突显了其高效性和泛化能力。 

---
# Reward Models Identify Consistency, Not Causality 

**Title (ZH)**: 奖励模型识别一致性，而非因果关系 

**Authors**: Yuhui Xu, Hanze Dong, Lei Wang, Caiming Xiong, Junnan Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.14619)  

**Abstract**: Reward models (RMs) play a crucial role in aligning large language models (LLMs) with human preferences and enhancing reasoning quality. Traditionally, RMs are trained to rank candidate outputs based on their correctness and coherence. However, in this work, we present several surprising findings that challenge common assumptions about RM behavior. Our analysis reveals that state-of-the-art reward models prioritize structural consistency over causal correctness. Specifically, removing the problem statement has minimal impact on reward scores, whereas altering numerical values or disrupting the reasoning flow significantly affects RM outputs. Furthermore, RMs exhibit a strong dependence on complete reasoning trajectories truncated or incomplete steps lead to significant variations in reward assignments, indicating that RMs primarily rely on learned reasoning patterns rather than explicit problem comprehension. These findings hold across multiple architectures, datasets, and tasks, leading to three key insights: (1) RMs primarily assess coherence rather than true reasoning quality; (2) The role of explicit problem comprehension in reward assignment is overstated; (3) Current RMs may be more effective at ranking responses than verifying logical validity. Our results suggest a fundamental limitation in existing reward modeling approaches, emphasizing the need for a shift toward causality-aware reward models that go beyond consistency-driven evaluation. 

**Abstract (ZH)**: 奖励模型（RMs）在将大型语言模型（LLMs）与人类偏好对齐以及提升推理质量方面发挥着关键作用。传统上，RMs 被训练以基于正确性和连贯性来排列候选输出。然而，在本工作中，我们提出了几个令人惊讶的发现，这些发现挑战了对 RMs 行为的常见假设。我们的分析揭示，最先进的奖励模型更注重结构性一致性，而不是因果正确性。具体而言，移除问题陈述对奖励分数的影响甚微，而改变数值或破坏推理流程则显著影响 RMs 的输出。此外，RMs 对完整的推理轨迹显示出强烈的依赖性，截断或不完整的推理步骤会导致奖励分配的重大变化，这表明 RMs 主要依赖于学习到的推理模式，而不是明确的问题理解。这些发现贯穿于多种架构、数据集和任务中，导致以下三个关键洞见：（1）RMs 主要评估连贯性而非真正的推理质量；（2）明确的问题理解在奖励分配中的作用被夸大了；（3）当前的 RMs 可能在响应排名上更为有效，而对逻辑有效性验证的效果较差。我们的研究结果表明，现有奖励建模方法存在根本局限，强调了向因果性感知的奖励模型的转变需求，这种模型超越了基于一致性的评估。 

---
# A Statistical Case Against Empirical Human-AI Alignment 

**Title (ZH)**: 统计学上对经验性人机齐心现象的反驳 

**Authors**: Julian Rodemann, Esteban Garces Arias, Christoph Luther, Christoph Jansen, Thomas Augustin  

**Link**: [PDF](https://arxiv.org/pdf/2502.14581)  

**Abstract**: Empirical human-AI alignment aims to make AI systems act in line with observed human behavior. While noble in its goals, we argue that empirical alignment can inadvertently introduce statistical biases that warrant caution. This position paper thus advocates against naive empirical alignment, offering prescriptive alignment and a posteriori empirical alignment as alternatives. We substantiate our principled argument by tangible examples like human-centric decoding of language models. 

**Abstract (ZH)**: 经验性的人机一致性旨在使AI系统的行为与观察到的人类行为相符。虽然其目标高尚，但我们认为经验性一致性可能会不小心引入统计偏差，这值得谨慎对待。因此，本文建议反对简单经验性一致性，提倡先导性一致性和事后经验性一致性的替代方案。我们通过诸如以人类为中心的语言模型解码等具体实例来论证我们基于原则的论点。 

---
# ReVISE: Learning to Refine at Test-Time via Intrinsic Self-Verification 

**Title (ZH)**: ReVISE：通过内在自我验证在测试时学习 refinement 

**Authors**: Hyunseok Lee, Seunghyuk Oh, Jaehyung Kim, Jinwoo Shin, Jihoon Tack  

**Link**: [PDF](https://arxiv.org/pdf/2502.14565)  

**Abstract**: Self-awareness, i.e., the ability to assess and correct one's own generation, is a fundamental aspect of human intelligence, making its replication in large language models (LLMs) an important yet challenging task. Previous works tackle this by employing extensive reinforcement learning or rather relying on large external verifiers. In this work, we propose Refine via Intrinsic Self-Verification (ReVISE), an efficient and effective framework that enables LLMs to self-correct their outputs through self-verification. The core idea of ReVISE is to enable LLMs to verify their reasoning processes and continually rethink reasoning trajectories based on its verification. We introduce a structured curriculum based upon online preference learning to implement this efficiently. Specifically, as ReVISE involves two challenging tasks (i.e., self-verification and reasoning correction), we tackle each task sequentially using curriculum learning, collecting both failed and successful reasoning paths to construct preference pairs for efficient training. During inference, our approach enjoys natural test-time scaling by integrating self-verification and correction capabilities, further enhanced by our proposed confidence-aware decoding mechanism. Our experiments on various reasoning tasks demonstrate that ReVISE achieves efficient self-correction and significantly improves reasoning performance. 

**Abstract (ZH)**: 自我意识，即评估和纠正自己生成内容的能力，是人类智能的一个基本方面，因此在大规模语言模型（LLMs）中复制这一能力是一项既重要又具有挑战性的工作。先前的研究通过广泛使用强化学习或依赖大型外部验证者来解决这一问题。在本研究中，我们提出了一种名为内在自我验证的精炼方法（ReVISE）的有效框架，使LLMs能够通过自我验证来纠正其输出。ReVISE的核心思想是使LLMs能够验证其推理过程，并基于验证结果持续重新思考推理路径。为此，我们引入了一种基于在线偏好学习的结构化课程，以高效地实施这一过程。具体来说，由于ReVISE涉及两个具有挑战性的任务（自我验证和推理纠正），我们通过逐个解决每个任务并收集成功和失败的推理路径来构建偏好对，以实现高效的训练。在推理过程中，我们的方法通过整合自我验证和纠正能力，自然地实现了测试时的规模扩展，并通过我们提出的一种基于信心的解码机制进一步增强。在各种推理任务上的实验表明，ReVISE能够实现高效的自我纠正，并显著提高推理性能。 

---
# Less is More: Improving LLM Alignment via Preference Data Selection 

**Title (ZH)**: 少即是多：通过偏好数据选择提高大模型一致性 

**Authors**: Xun Deng, Han Zhong, Rui Ai, Fuli Feng, Zheng Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2502.14560)  

**Abstract**: Direct Preference Optimization (DPO) has emerged as a promising approach for aligning large language models with human preferences. While prior work mainly extends DPO from the aspect of the objective function, we instead improve DPO from the largely overlooked but critical aspect of data selection. Specifically, we address the issue of parameter shrinkage caused by noisy data by proposing a novel margin-maximization principle for dataset curation in DPO training. To accurately estimate margins for data selection, we propose a dual-margin guided approach that considers both external reward margins and implicit DPO reward margins. Extensive experiments demonstrate that our method reduces computational cost dramatically while improving performance. Remarkably, by using just 10\% of the Ultrafeedback dataset, our approach achieves 3\% to 8\% improvements across various Llama and Mistral series models on the AlpacaEval 2.0 benchmark. Furthermore, our approach seamlessly extends to iterative DPO, yielding a roughly 3\% improvement with 25\% online data, while further reducing training time. These results highlight the potential of data selection strategies for advancing preference optimization. 

**Abstract (ZH)**: Direct偏好优化（DPO）已成为一种有前景的方法，用于使大型语言模型与人类偏好对齐。尽管先前的研究主要在目标函数方面扩展了DPO，我们则从被大大忽视但至关重要的数据选择方面改进了DPO。具体而言，我们通过提出一种新颖的边距最大化原则来解决由于噪声数据导致的参数收缩问题，从而在DPO训练中改进数据集筛选。为了准确估计用于数据选择的边距，我们提出了一种双重边距引导的方法，该方法考虑了外部奖励边距和隐式DPO奖励边距。大量实验表明，我们的方法在大幅降低计算成本的同时提高了性能。值得注意的是，仅使用Ultrafeedback数据集的10%，我们的方法在AlpacaEval 2.0基准测试中针对各种Llama和Mistral系列模型实现了3%到8%的性能提升。此外，我们的方法无缝地扩展到了迭代DPO，并在离线数据占比25%的情况下实现了约3%的性能提升，同时进一步缩短了训练时间。这些结果突显了数据选择策略在促进偏好优化方面的潜力。 

---
# Generative adversarial networks vs large language models: a comparative study on synthetic tabular data generation 

**Title (ZH)**: 生成对抗网络与大规模语言模型：合成表格数据生成的比较研究 

**Authors**: Austin A. Barr, Robert Rozman, Eddie Guo  

**Link**: [PDF](https://arxiv.org/pdf/2502.14523)  

**Abstract**: We propose a new framework for zero-shot generation of synthetic tabular data. Using the large language model (LLM) GPT-4o and plain-language prompting, we demonstrate the ability to generate high-fidelity tabular data without task-specific fine-tuning or access to real-world data (RWD) for pre-training. To benchmark GPT-4o, we compared the fidelity and privacy of LLM-generated synthetic data against data generated with the conditional tabular generative adversarial network (CTGAN), across three open-access datasets: Iris, Fish Measurements, and Real Estate Valuation. Despite the zero-shot approach, GPT-4o outperformed CTGAN in preserving means, 95% confidence intervals, bivariate correlations, and data privacy of RWD, even at amplified sample sizes. Notably, correlations between parameters were consistently preserved with appropriate direction and strength. However, refinement is necessary to better retain distributional characteristics. These findings highlight the potential of LLMs in tabular data synthesis, offering an accessible alternative to generative adversarial networks and variational autoencoders. 

**Abstract (ZH)**: 我们提出了一种新的框架，用于零样本生成合成表格数据。通过使用大型语言模型（LLM）GPT-4o 和简单的语言提示，我们展示了在无需特定任务的微调或访问真实世界数据（RWD）进行预训练的情况下，生成高保真度表格数据的能力。为了评估 GPT-4o，我们将其生成的合成数据的保真度和隐私性与使用条件性生成对抗网络（CTGAN）生成的合成数据进行了比较，测试了三个开源数据集：Iris、Fish Measurements 和 Real Estate Valuation。尽管采用零样本方法，GPT-4o 在保留均值、95% 置信区间、双变量相关性和 RWD 的数据隐私方面仍然优于 CTGAN，即使在放大样本量的情况下也是如此。值得注意的是，与参数之间的相关性保持了适当的方向和强度。然而，仍需改进以更好地保留数据分布特性。这些发现突显了LLMs在表格数据合成中的潜力，为生成对抗网络和变分自编码器提供了可访问的替代方案。 

---
# How Jailbreak Defenses Work and Ensemble? A Mechanistic Investigation 

**Title (ZH)**: 《 jailbreak 防御机制及其集成研究 —— 一种机制性调查》

在这个翻译中，我遵循了学术翻译的一般规范，尽量保持原意的同时使用了更为学术化和符合中文表达习惯的词汇。"How Jailbreak Defenses Work and Ensemble?" 这个部分翻译成了 "jailbreak 防御机制及其集成"，而 "A Mechanistic Investigation" 则翻译成 "一种机制性调查"。 

**Authors**: Zhuohang Long, Siyuan Wang, Shujun Liu, Yuhang Lai, Xuanjing Huang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2502.14486)  

**Abstract**: Jailbreak attacks, where harmful prompts bypass generative models' built-in safety, raise serious concerns about model vulnerability. While many defense methods have been proposed, the trade-offs between safety and helpfulness, and their application to Large Vision-Language Models (LVLMs), are not well understood. This paper systematically examines jailbreak defenses by reframing the standard generation task as a binary classification problem to assess model refusal tendencies for both harmful and benign queries. We identify two key defense mechanisms: safety shift, which increases refusal rates across all queries, and harmfulness discrimination, which improves the model's ability to distinguish between harmful and benign inputs. Using these mechanisms, we develop two ensemble defense strategies-inter-mechanism ensembles and intra-mechanism ensembles-to balance safety and helpfulness. Experiments on the MM-SafetyBench and MOSSBench datasets with LLaVA-1.5 models show that these strategies effectively improve model safety or optimize the trade-off between safety and helpfulness. 

**Abstract (ZH)**: 在生成模型内置安全机制被有害提示绕过的狱突攻击中，对模型脆弱性的担忧尤为严重。尽管已经提出了许多防御方法，但安全性和帮助性之间的权衡以及这些方法在大规模视觉-语言模型（LVLMs）上的应用尚不明确。本文系统地探讨了狱突防御措施，通过将标准生成任务重新构建成二元分类问题来评估模型对有害查询和良性查询的拒绝倾向。我们识别出两种关键防御机制：安全性转移，它会在所有查询中增加拒绝率；有害性区分，它能提升模型区分有害和良性输入的能力。利用这些机制，我们开发了两种综合防御策略：跨机制综合和同机制综合，以平衡安全性和帮助性。对LLaVA-1.5模型在MM-SafetyBench和MOSSBench数据集上的实验结果表明，这些策略能够有效提升模型的安全性或优化安全性和帮助性的权衡。 

---
# A Macro- and Micro-Hierarchical Transfer Learning Framework for Cross-Domain Fake News Detection 

**Title (ZH)**: 跨域假新闻检测的宏观和微观层次迁移学习框架 

**Authors**: Xuankai Yang, Yan Wang, Xiuzhen Zhang, Shoujin Wang, Huaxiong Wang, Kwok Yan Lam  

**Link**: [PDF](https://arxiv.org/pdf/2502.14403)  

**Abstract**: Cross-domain fake news detection aims to mitigate domain shift and improve detection performance by transferring knowledge across domains. Existing approaches transfer knowledge based on news content and user engagements from a source domain to a target domain. However, these approaches face two main limitations, hindering effective knowledge transfer and optimal fake news detection performance. Firstly, from a micro perspective, they neglect the negative impact of veracity-irrelevant features in news content when transferring domain-shared features across domains. Secondly, from a macro perspective, existing approaches ignore the relationship between user engagement and news content, which reveals shared behaviors of common users across domains and can facilitate more effective knowledge transfer. To address these limitations, we propose a novel macro- and micro- hierarchical transfer learning framework (MMHT) for cross-domain fake news detection. Firstly, we propose a micro-hierarchical disentangling module to disentangle veracity-relevant and veracity-irrelevant features from news content in the source domain for improving fake news detection performance in the target domain. Secondly, we propose a macro-hierarchical transfer learning module to generate engagement features based on common users' shared behaviors in different domains for improving effectiveness of knowledge transfer. Extensive experiments on real-world datasets demonstrate that our framework significantly outperforms the state-of-the-art baselines. 

**Abstract (ZH)**: 跨领域假新闻检测旨在通过领域间的知识转移来减轻领域偏移问题，从而提高检测性能。现有的方法通过从源领域向目标领域传输新闻内容和用户互动的知识来进行知识转移。然而，这些方法存在两个主要局限性，阻碍了有效知识转移和最佳假新闻检测性能的实现。首先，从微观角度来看，它们在跨领域转移共享特征时忽视了新闻内容中与真实性无关的特征的负面影响。其次，从宏观角度来看，现有方法忽略了用户互动和新闻内容之间的关系，这揭示了不同领域中共同用户的共享行为，可以促进更有效的知识转移。为了解决这些局限性，我们提出了一种新的宏-微观层次转移学习框架（MMHT）来实现跨领域假新闻检测。首先，我们提出了一种微观层次解耦模块，以从源领域中解耦真实性相关和无关的特征，从而提高目标领域中的假新闻检测性能。其次，我们提出了一种宏观层次转移学习模块，根据不同领域中共同用户共享的行为来生成互动特征，从而提高知识转移的有效性。在真实数据集上的广泛实验表明，我们的框架显著优于现有的基线方法。 

---
# Self-Improvement Towards Pareto Optimality: Mitigating Preference Conflicts in Multi-Objective Alignment 

**Title (ZH)**: 向帕累托最优自我提升：缓解多目标对齐中的偏好冲突 

**Authors**: Moxin Li, Yuantao Zhang, Wenjie Wang, Wentao Shi, Zhuo Liu, Fuli Feng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2502.14354)  

**Abstract**: Multi-Objective Alignment (MOA) aims to align LLMs' responses with multiple human preference objectives, with Direct Preference Optimization (DPO) emerging as a prominent approach. However, we find that DPO-based MOA approaches suffer from widespread preference conflicts in the data, where different objectives favor different responses. This results in conflicting optimization directions, hindering the optimization on the Pareto Front. To address this, we propose to construct Pareto-optimal responses to resolve preference conflicts. To efficiently obtain and utilize such responses, we propose a self-improving DPO framework that enables LLMs to self-generate and select Pareto-optimal responses for self-supervised preference alignment. Extensive experiments on two datasets demonstrate the superior Pareto Front achieved by our framework compared to various baselines. Code is available at \url{this https URL}. 

**Abstract (ZH)**: 多目标对齐（MOA）旨在使大规模语言模型（LLM）的响应与多个人类偏好目标保持一致，直接偏好优化（DPO）已成为一种突出的方法。然而，我们发现基于DPO的MOA方法在数据中普遍存在偏好冲突，即不同的目标倾向于不同的响应。这导致了互相矛盾的优化方向，阻碍了帕累托前沿的优化。为解决这一问题，我们提出构建帕累托最优响应以解决偏好冲突。为了高效地获取和利用这些响应，我们提出了一种自我提升的DPO框架，使LLM能够自我生成和选择帕累托最优响应，以实现自我监督的偏好对齐。在两个数据集上的广泛实验表明，与各种基线相比，我们的框架实现了更优的帕累托前沿。源代码可在以下链接获取：\url{this https URL}。 

---
# Beyond Self-Talk: A Communication-Centric Survey of LLM-Based Multi-Agent Systems 

**Title (ZH)**: 超越自我对话：基于通信的LLM驱动多Agent系统综述 

**Authors**: Bingyu Yan, Xiaoming Zhang, Litian Zhang, Lian Zhang, Ziyi Zhou, Dezhuang Miao, Chaozhuo Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.14321)  

**Abstract**: Large Language Models (LLMs) have recently demonstrated remarkable capabilities in reasoning, planning, and decision-making. Building upon these strengths, researchers have begun incorporating LLMs into multi-agent systems (MAS), where agents collaborate or compete through natural language interactions to tackle tasks beyond the scope of single-agent setups. In this survey, we present a communication-centric perspective on LLM-based multi-agent systems, examining key system-level features such as architecture design and communication goals, as well as internal mechanisms like communication strategies, paradigms, objects and content. We illustrate how these communication elements interplay to enable collective intelligence and flexible collaboration. Furthermore, we discuss prominent challenges, including scalability, security, and multimodal integration, and propose directions for future work to advance research in this emerging domain. Ultimately, this survey serves as a catalyst for further innovation, fostering more robust, scalable, and intelligent multi-agent systems across diverse application domains. 

**Abstract (ZH)**: 大型语言模型（LLMs）最近在推理、规划和决策方面展现了卓越的能力。在此基础上，研究人员开始将LLMs融入多Agent系统（MAS），其中的Agent通过自然语言交互合作或竞争，以应对单Agent系统无法胜任的任务。本文综述了以通信为中心的LLM基多Agent系统视角，探讨了系统层面的关键特征，如架构设计和通信目标，以及内部机制，如通信策略、范式、对象和内容。我们展示了这些通信元素如何相互作用，以实现集体智能和灵活的合作。此外，我们讨论了包括扩展性、安全性和多模态集成在内的主要挑战，并提出了未来工作的方向，以推动这一新兴领域的研究。最终，本文综述旨在激发进一步的创新，促进在各种应用领域更加稳健、可扩展和智能的多Agent系统的发展。 

---
# The Impact and Feasibility of Self-Confidence Shaping for AI-Assisted Decision-Making 

**Title (ZH)**: 自我信心塑造对AI辅助决策影响及其可行性研究 

**Authors**: Takehiro Takayanagi, Ryuji Hashimoto, Chung-Chi Chen, Kiyoshi Izumi  

**Link**: [PDF](https://arxiv.org/pdf/2502.14311)  

**Abstract**: In AI-assisted decision-making, it is crucial but challenging for humans to appropriately rely on AI, especially in high-stakes domains such as finance and healthcare. This paper addresses this problem from a human-centered perspective by presenting an intervention for self-confidence shaping, designed to calibrate self-confidence at a targeted level. We first demonstrate the impact of self-confidence shaping by quantifying the upper-bound improvement in human-AI team performance. Our behavioral experiments with 121 participants show that self-confidence shaping can improve human-AI team performance by nearly 50% by mitigating both over- and under-reliance on AI. We then introduce a self-confidence prediction task to identify when our intervention is needed. Our results show that simple machine-learning models achieve 67% accuracy in predicting self-confidence. We further illustrate the feasibility of such interventions. The observed relationship between sentiment and self-confidence suggests that modifying sentiment could be a viable strategy for shaping self-confidence. Finally, we outline future research directions to support the deployment of self-confidence shaping in a real-world scenario for effective human-AI collaboration. 

**Abstract (ZH)**: 在人工智能辅助决策中，人类适当地依赖AI尤其在金融和医疗等高风险领域中至关重要，但这一过程充满挑战。本文从以人为本的角度出发，提出了一个自信心塑造干预措施，旨在将自信心调整到目标水平。首先，通过量化人类-AI团队性能的最大改进，我们展示了自信心塑造的影响。我们的行为实验（121名参与者参加）表明，自信心塑造能够通过减轻对AI的过度依赖和不足依赖，将人类-AI团队的性能提高近50%。随后，我们介绍了一个自信心预测任务，以确定何时需要实施干预措施。结果显示，简单的机器学习模型在预测自信心方面达到了67%的准确性。我们进一步阐述了此类干预措施的可行性。观察到的情绪与自信心之间的关系表明，修改情绪可能是塑造自信心的一种可行策略。最后，我们概述了未来的研究方向，以支持在实际场景中部署自信心塑造措施，以促进高效的人机协作。 

---
# STeCa: Step-level Trajectory Calibration for LLM Agent Learning 

**Title (ZH)**: STeCa：LLM代理学习中的步骤级轨迹校准 

**Authors**: Hanlin Wang, Jian Wang, Chak Tou Leong, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.14276)  

**Abstract**: Large language model (LLM)-based agents have shown promise in tackling complex tasks by interacting dynamically with the environment. Existing work primarily focuses on behavior cloning from expert demonstrations and preference learning through exploratory trajectory sampling. However, these methods often struggle in long-horizon tasks, where suboptimal actions accumulate step by step, causing agents to deviate from correct task trajectories. To address this, we highlight the importance of timely calibration and the need to automatically construct calibration trajectories for training agents. We propose Step-Level Trajectory Calibration (STeCa), a novel framework for LLM agent learning. Specifically, STeCa identifies suboptimal actions through a step-level reward comparison during exploration. It constructs calibrated trajectories using LLM-driven reflection, enabling agents to learn from improved decision-making processes. These calibrated trajectories, together with successful trajectory data, are utilized for reinforced training. Extensive experiments demonstrate that STeCa significantly outperforms existing methods. Further analysis highlights that step-level calibration enables agents to complete tasks with greater robustness. Our code and data are available at this https URL. 

**Abstract (ZH)**: 基于大规模语言模型（LLM）的代理在通过与环境动态互动来应对复杂任务方面显示出潜力。现有工作主要集中在从专家演示中学习行为克隆和通过探索轨迹采样学习偏好。然而，这些方法在长时间任务方面往往表现不佳，特别是当不可最优动作逐步累积时，会导致代理偏离正确的任务轨迹。为解决这一问题，我们强调及时校准的重要性，并指出需要自动构建校准轨迹以供代理训练。我们提出了Step-Level Trajectory Calibration（STeCa），这是一种新的LLM代理学习框架。具体而言，STeCa通过探索过程中的步骤级奖励比较来识别不可最优动作，并利用LLM驱动的反思构建校准轨迹，使代理能够从改进的决策过程中学习。这些校准轨迹与成功的轨迹数据一起用于强化训练。广泛的实验结果表明，STeCa显著优于现有方法。进一步分析表明，步骤级校准使代理能够以更高的鲁棒性完成任务。我们的代码和数据可在以下链接获取：[此处链接]。 

---
# Do LLMs Consider Security? An Empirical Study on Responses to Programming Questions 

**Title (ZH)**: 大型语言模型考虑安全性吗？关于应对程序问题的回答的实证研究 

**Authors**: Amirali Sajadi, Binh Le, Anh Nguyen, Kostadin Damevski, Preetha Chatterjee  

**Link**: [PDF](https://arxiv.org/pdf/2502.14202)  

**Abstract**: The widespread adoption of conversational LLMs for software development has raised new security concerns regarding the safety of LLM-generated content. Our motivational study outlines ChatGPT's potential in volunteering context-specific information to the developers, promoting safe coding practices. Motivated by this finding, we conduct a study to evaluate the degree of security awareness exhibited by three prominent LLMs: Claude 3, GPT-4, and Llama 3. We prompt these LLMs with Stack Overflow questions that contain vulnerable code to evaluate whether they merely provide answers to the questions or if they also warn users about the insecure code, thereby demonstrating a degree of security awareness. Further, we assess whether LLM responses provide information about the causes, exploits, and the potential fixes of the vulnerability, to help raise users' awareness. Our findings show that all three models struggle to accurately detect and warn users about vulnerabilities, achieving a detection rate of only 12.6% to 40% across our datasets. We also observe that the LLMs tend to identify certain types of vulnerabilities related to sensitive information exposure and improper input neutralization much more frequently than other types, such as those involving external control of file names or paths. Furthermore, when LLMs do issue security warnings, they often provide more information on the causes, exploits, and fixes of vulnerabilities compared to Stack Overflow responses. Finally, we provide an in-depth discussion on the implications of our findings and present a CLI-based prompting tool that can be used to generate significantly more secure LLM responses. 

**Abstract (ZH)**: 软件开发中广泛采用对话型大语言模型（Conversational LLMs）引起了对大语言模型生成内容安全性的新担忧。本动机研究概述了ChatGPT在上下文特定信息提供中的潜力，以促进安全编码实践。受此发现的启发，我们对Claude 3、GPT-4和Llama 3这三种主要的LLM进行了研究，评估其在安全意识方面的程度。我们向这些LLM提供了包含漏洞代码的Stack Overflow问题，评估它们是否仅仅提供问题的答案，还是也会警告用户关于不安全代码的内容，从而表现出一定程度的安全意识。此外，我们进一步评估了LLM的回答是否提供了有关漏洞的原因、利用方法及其潜在修复措施的信息，以帮助提高用户的安全意识。研究结果显示，这三种模型在准确检测和警告用户关于漏洞方面存在困难，在我们的数据集中，检测率仅为12.6%至40%。我们还发现，这些LLM更频繁地识别与敏感信息暴露和不适当输入中和相关的某些类型漏洞，而不是其他类型的漏洞，如与外部控制文件名或路径相关的漏洞。此外，当LLM发布安全警告时，它们提供的关于漏洞原因、利用方法及其修复措施的信息比Stack Overflow的回答要多。最后，我们深入讨论了这些发现的意义，并提出了一个基于命令行界面的提示工具，该工具可用于生成显著更安全的LLM回答。 

---
# Federated Fine-Tuning of Large Language Models: Kahneman-Tversky vs. Direct Preference Optimization 

**Title (ZH)**: 大规模语言模型的联邦微调：卡尼曼-特维斯基对比直接偏好优化 

**Authors**: Fernando Spadea, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2502.14187)  

**Abstract**: We evaluate Kahneman-Tversky Optimization (KTO) as a fine-tuning method for large language models (LLMs) in federated learning (FL) settings, comparing it against Direct Preference Optimization (DPO). Using Alpaca-7B as the base model, we fine-tune on a realistic dataset under both methods and evaluate performance using MT-Bench-1, Vicuna, and AdvBench benchmarks. Additionally, we introduce a redistributed dataset setup, where only KTO is applicable due to its ability to handle single-response feedback, unlike DPO's reliance on paired responses. Our results demonstrate that KTO, in both its original (KTOO) and redistributed (KTOR) configurations, consistently outperforms DPO across all benchmarks. In the redistributed setup, KTO further validates its flexibility and resilience by maintaining superior performance in scenarios where DPO cannot be applied. These findings establish KTO as a robust and scalable fine-tuning method for FL, motivating its adoption for privacy-preserving, decentralized, and heterogeneous environments. 

**Abstract (ZH)**: 我们将Kahneman-Tversky优化（KTO）评估为联邦学习（FL）环境中大型语言模型（LLMs）的微调方法，并将其与直接偏好优化（DPO）进行了对比。以Alpaca-7B为基模型，在两种方法下对其进行微调，并使用MT-Bench-1、Vicuna和AdvBench基准测试其性能。此外，我们引入了一种重新分布的数据集设置，在这种设置中，只有KTO适用，因为KTO能够处理单响应反馈，而DPO依赖于成对的反馈。结果显示，在所有基准测试中，无论是在原始配置（KTOO）还是重新分布配置（KTOR）下，KTO都持续优于DPO。在重新分布设置中，KTO进一步验证了其灵活性和鲁棒性，在DPO无法应用的场景中仍能保持更优的性能。这些发现证明了KTO作为一种适用于FL的稳健且可扩展的微调方法的有效性，并促使其在保护隐私、去中心化和异构环境中得到采用。 

---
# On the logical skills of large language models: evaluations using arbitrarily complex first-order logic problems 

**Title (ZH)**: 大型语言模型的逻辑能力评估：使用任意复杂的一阶逻辑问题 

**Authors**: Shokhrukh Ibragimov, Arnulf Jentzen, Benno Kuckuck  

**Link**: [PDF](https://arxiv.org/pdf/2502.14180)  

**Abstract**: We present a method of generating first-order logic statements whose complexity can be controlled along multiple dimensions. We use this method to automatically create several datasets consisting of questions asking for the truth or falsity of first-order logic statements in Zermelo-Fraenkel set theory. While the resolution of these questions does not require any knowledge beyond basic notation of first-order logic and set theory, it does require a degree of planning and logical reasoning, which can be controlled up to arbitrarily high difficulty by the complexity of the generated statements. Furthermore, we do extensive evaluations of the performance of various large language models, including recent models such as DeepSeek-R1 and OpenAI's o3-mini, on these datasets. All of the datasets along with the code used for generating them, as well as all data from the evaluations is publicly available at this https URL. 

**Abstract (ZH)**: 我们提出了一种生成一阶逻辑命题的方法，其复杂性可以在多个维度上进行控制。我们使用这种方法自动生成了多个数据集，这些数据集包含的问题是询问一阶逻辑命题在策梅洛-弗兰克尔集合理论中的真假。解答这些问题不需要任何超出基本一阶逻辑和集合理论符号的知识，但需要一定程度的计划和逻辑推理能力，而这种能力可以通过生成命题的复杂性来任意调整到很高的难度。此外，我们对包括最近发布的模型DeepSeek-R1和OpenAI的o3-mini在内的多种大型语言模型在这些数据集上的性能进行了广泛评估。所有数据集、生成这些数据集所使用的代码以及评估中的所有数据均可通过以下网址公开访问：[此处插入网址]。 

---
# Giving AI Personalities Leads to More Human-Like Reasoning 

**Title (ZH)**: 赋予AI个性有助于实现更加人性化的推理 

**Authors**: Animesh Nighojkar, Bekhzodbek Moydinboyev, My Duong, John Licato  

**Link**: [PDF](https://arxiv.org/pdf/2502.14155)  

**Abstract**: In computational cognitive modeling, capturing the full spectrum of human judgment and decision-making processes, beyond just optimal behaviors, is a significant challenge. This study explores whether Large Language Models (LLMs) can emulate the breadth of human reasoning by predicting both intuitive, fast System 1 and deliberate, slow System 2 processes. We investigate the potential of AI to mimic diverse reasoning behaviors across a human population, addressing what we call the {\em full reasoning spectrum problem}. We designed reasoning tasks using a novel generalization of the Natural Language Inference (NLI) format to evaluate LLMs' ability to replicate human reasoning. The questions were crafted to elicit both System 1 and System 2 responses. Human responses were collected through crowd-sourcing and the entire distribution was modeled, rather than just the majority of the answers. We used personality-based prompting inspired by the Big Five personality model to elicit AI responses reflecting specific personality traits, capturing the diversity of human reasoning, and exploring how personality traits influence LLM outputs. Combined with genetic algorithms to optimize the weighting of these prompts, this method was tested alongside traditional machine learning models. The results show that LLMs can mimic human response distributions, with open-source models like Llama and Mistral outperforming proprietary GPT models. Personality-based prompting, especially when optimized with genetic algorithms, significantly enhanced LLMs' ability to predict human response distributions, suggesting that capturing suboptimal, naturalistic reasoning may require modeling techniques incorporating diverse reasoning styles and psychological profiles. The study concludes that personality-based prompting combined with genetic algorithms is promising for enhancing AI's \textit{human-ness} in reasoning. 

**Abstract (ZH)**: 在计算认知建模中，超越最优化行为，捕捉人类判断和决策的完整谱系是一个重大挑战。本研究探讨大型语言模型（LLMs）是否能够模拟人类推理的广泛性，不仅预测直觉快速的System 1，还预测慎思慢速的System 2过程。我们研究了人工智能在人类群体中模拟多样推理行为的潜力，即我们称之为“完整推理谱系问题”。我们使用了一种自然语言推理（NLI）的新颖泛化形式来设计推理任务，以评估LLMs重现人类推理的能力。问题设计旨在触发System 1和System 2的反应。人类反应通过众包收集，并且整个分布被建模，而不仅仅是大多数答案。我们使用基于性格的提示，灵感来源于大五人格模型，激发反映特定人格特质的AI响应，捕捉人类推理的多样性，探索人格特质如何影响LLMs的输出。结合使用遗传算法优化这些提示的权重，该方法被与传统机器学习模型一起测试。结果表明，LLMs能够模拟人类反应的分布，开源模型如Llama和Mistral的表现优于专有的GPT模型。基于性格的提示，特别是与遗传算法优化结合时，显著增强了LLMs预测人类反应分布的能力，这表明捕捉非优化的、自然主义的推理可能需要包含多元推理风格和心理特征的建模技术。研究结论认为，结合遗传算法的基于性格的提示是提高AI在推理中“人性”方面的有前景的方法。 

---
# Investigating Non-Transitivity in LLM-as-a-Judge 

**Title (ZH)**: 探究基于语言模型的法官（LLM-as-a-Judge）中非传递性的现象 

**Authors**: Yi Xu, Laura Ruis, Tim Rocktäschel, Robert Kirk  

**Link**: [PDF](https://arxiv.org/pdf/2502.14074)  

**Abstract**: Automatic evaluation methods based on large language models (LLMs) are emerging as the standard tool for assessing the instruction-following abilities of LLM-based agents. The most common method in this paradigm, pairwise comparisons with a baseline model, critically depends on the assumption of transitive preferences. However, the validity of this assumption remains largely unexplored. In this study, we investigate the presence of non-transitivity within the AlpacaEval framework and analyze its effects on model rankings. We find that LLM judges exhibit non-transitive preferences, leading to rankings that are sensitive to the choice of the baseline model. To mitigate this issue, we show that round-robin tournaments combined with Bradley-Terry models of preference can produce more reliable rankings. Notably, our method increases both the Spearman correlation and the Kendall correlation with Chatbot Arena (95.0% -> 96.4% and 82.1% -> 86.3% respectively). To address the computational cost of round-robin tournaments, we propose Swiss-Wise Iterative Matchmaking (Swim) tournaments, using a dynamic matching strategy to capture the benefits of round-robin tournaments while maintaining computational efficiency. 

**Abstract (ZH)**: 基于大型语言模型（LLMs）的自动评估方法正在成为评估LLM代理执行指令能力的标准工具。在这一范式中最常见的方法是使用基线模型进行成对比较，这种方法严格依赖于传递偏好假设的有效性。然而，这一假设的有效性尚未得到充分探讨。在本研究中，我们考察了AlpacaEval框架中的非传递性现象，并分析了这种现象对模型排名的影响。我们发现，LLM评判者表现出非传递性偏好，这使得排名对基线模型的选择非常敏感。为了缓解这一问题，我们展示了采用轮换锦标赛结合布雷德利-特里模型的偏好方法可以产生更可靠的排名。值得注意的是，我们的方法分别将Spearman相关性和肯德尔相关性与Chatbot Arena的匹配提高到了96.4%（从95.0%）和86.3%（从82.1%）。为解决轮换锦标赛的计算成本问题，我们提出了智慧迭代匹配（Swim）锦标赛，通过动态匹配策略同时保留轮换锦标赛的优势并保持计算效率。 

---
# Which Attention Heads Matter for In-Context Learning? 

**Title (ZH)**: 上下文学习中哪些注意力头更重要？ 

**Authors**: Kayo Yin, Jacob Steinhardt  

**Link**: [PDF](https://arxiv.org/pdf/2502.14010)  

**Abstract**: Large language models (LLMs) exhibit impressive in-context learning (ICL) capability, enabling them to perform new tasks using only a few demonstrations in the prompt. Two different mechanisms have been proposed to explain ICL: induction heads that find and copy relevant tokens, and function vector (FV) heads whose activations compute a latent encoding of the ICL task. To better understand which of the two distinct mechanisms drives ICL, we study and compare induction heads and FV heads in 12 language models.
Through detailed ablations, we discover that few-shot ICL performance depends primarily on FV heads, especially in larger models. In addition, we uncover that FV and induction heads are connected: many FV heads start as induction heads during training before transitioning to the FV mechanism. This leads us to speculate that induction facilitates learning the more complex FV mechanism that ultimately drives ICL. 

**Abstract (ZH)**: 大型语言模型（LLMs）表现出显著的上下文内学习（ICL）能力，使它们能够在仅使用少量示范的情况下执行新任务。提出了两种不同的机制来解释ICL：诱导头（induction heads），它寻找并复制相关令牌；以及功能向量（FV）头，其激活值计算ICL任务的潜在编码。为了更好地理解哪种机制驱动ICL，我们研究并比较了12种语言模型中的诱导头和FV头。

通过详细的消融实验，我们发现，少量示例的ICL性能主要依赖于FV头，特别是在更大规模的模型中。此外，我们发现FV头和诱导头之间存在联系：许多FV头在训练期间最初作为诱导头出现，随后过渡到FV机制。这使我们推测，诱导机制有助于学习更为复杂的FV机制，最终驱动ICL。 

---
