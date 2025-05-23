# Speak Easy: Eliciting Harmful Jailbreaks from LLMs with Simple Interactions 

**Title (ZH)**: 轻松启奏：通过简单交互激发来自LLM的有害越界行为 

**Authors**: Yik Siu Chan, Narutatsu Ri, Yuxin Xiao, Marzyeh Ghassemi  

**Link**: [PDF](https://arxiv.org/pdf/2502.04322)  

**Abstract**: Despite extensive safety alignment efforts, large language models (LLMs) remain vulnerable to jailbreak attacks that elicit harmful behavior. While existing studies predominantly focus on attack methods that require technical expertise, two critical questions remain underexplored: (1) Are jailbroken responses truly useful in enabling average users to carry out harmful actions? (2) Do safety vulnerabilities exist in more common, simple human-LLM interactions? In this paper, we demonstrate that LLM responses most effectively facilitate harmful actions when they are both actionable and informative--two attributes easily elicited in multi-step, multilingual interactions. Using this insight, we propose HarmScore, a jailbreak metric that measures how effectively an LLM response enables harmful actions, and Speak Easy, a simple multi-step, multilingual attack framework. Notably, by incorporating Speak Easy into direct request and jailbreak baselines, we see an average absolute increase of 0.319 in Attack Success Rate and 0.426 in HarmScore in both open-source and proprietary LLMs across four safety benchmarks. Our work reveals a critical yet often overlooked vulnerability: Malicious users can easily exploit common interaction patterns for harmful intentions. 

**Abstract (ZH)**: 尽管在安全对齐方面做出了大量努力，大规模语言模型（LLMs）仍然容易受到触发有害行为的“逃逸攻击”的威胁。现有研究主要集中在需要技术专业知识的攻击方法上，但仍有两个关键问题尚未得到充分探索：（1）被劫持的响应是否真的能够帮助普通用户实施有害行为？（2）在更常见、更简单的交互中，是否存在安全漏洞？在本文中，我们证明，当LLM的响应兼具可行性和信息性时，它们最有效地促进了有害行为的实施——这两种属性在多步骤、多语言交互中很容易被激发。基于这一见解，我们提出了HarmScore，一种衡量LLM响应促进有害行为效果的劫持指标，以及Speak Easy，一种简单的多步骤、多语言攻击框架。值得注意的是，通过将Speak Easy整合到直接请求和劫持基线中，我们在四个安全基准中的开源和专有LLM上均观察到攻击成功率平均绝对值提高了0.319，HarmScore平均绝对值提高了0.426。我们的研究揭示了一个关键但往往被忽视的漏洞：恶意用户可以轻松利用常见的交互模式来实现其有害意图。 

---
# ChamaleonLLM: Batch-Aware Dynamic Low-Rank Adaptation via Inference-Time Clusters 

**Title (ZH)**: ChamaleonLLM：推理时聚类驱动的批次感知低秩自适应方法 

**Authors**: Kamer Ali Yuksel, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2502.04315)  

**Abstract**: Recent advances in large language models (LLMs) have shown remarkable performance across diverse tasks. However, these models are typically deployed with fixed weights, which limits their ability to adapt dynamically to the variability inherent in real-world data during inference. This paper introduces ChamaleonLLM, a novel framework that enables inference-time adaptation of LLMs by leveraging batch-aware clustering and on-the-fly generation of low-rank updates. Unlike traditional fine-tuning approaches such as Low-Rank Adaptation (LoRA) or methods that rely on a fixed set of pre-learned uniforms (changeable masks), our method dynamically generates adaptive modifications to the decoder weights based on the aggregated statistics of clustered batches. By intelligently grouping similar inputs and computing context-aware low-rank updates via a hyper-network, ChamaleonLLM achieves significant performance gains, outperforming conventional LoRA methods while eliminating the overhead of maintaining multiple expert models. Our experiments highlight the potential of our approach to serve as a versatile and highly adaptive solution for language model inference. ChamaleonLLM is open-sourced to ensure the reproducibility of our experiments: this https URL 

**Abstract (ZH)**: 近年来，大规模语言模型（LLMs）在多种任务中的表现突出。然而，这些模型通常以固定权值的形式部署，这限制了它们在推理过程中动态适应真实世界数据固有变异性的能力。本文介绍了ChamaleonLLM，这是一种新型框架，通过利用批处理意识聚类和随机制低秩更新，实现了LLMs的推理时适应。与传统的细调方法如低秩适应（LoRA）或其他依赖于固定预学习动态掩码的方法不同，我们的方法能够根据聚类批次的聚合统计自适应地修改解码器权重。通过智能地分组相似输入并通过超网络计算上下文感知的低秩更新，ChamaleonLLM 达到了显著的性能提升，优于传统的 LoRA 方法，并且消除了维护多个专家模型的开销。实验结果展示了我们方法的潜力，作为一种既灵活又高度适应的解决方案，用于语言模型推理。ChamaleonLLM 已开源以确保实验的可重复性：[此链接] 

---
# The Best Instruction-Tuning Data are Those That Fit 

**Title (ZH)**: 最佳的指令调优数据是那些符合需求的数据 

**Authors**: Dylan Zhang, Qirun Dai, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.04194)  

**Abstract**: High-quality supervised fine-tuning (SFT) data are crucial for eliciting strong capabilities from pretrained large language models (LLMs). Typically, instructions are paired with multiple responses sampled from other LLMs, which are often out of the distribution of the target model to be fine-tuned. This, at scale, can lead to diminishing returns and even hurt the models' performance and robustness. We propose **GRAPE**, a novel SFT framework that accounts for the unique characteristics of the target model. For each instruction, it gathers responses from various LLMs and selects the one with the highest probability measured by the target model, indicating that it aligns most closely with the target model's pretrained distribution; it then proceeds with standard SFT training.
We first evaluate GRAPE with a controlled experiment, where we sample various solutions for each question in UltraInteract from multiple models and fine-tune commonly used LMs like LLaMA3.1-8B, Mistral-7B, and Qwen2.5-7B on GRAPE-selected data. GRAPE significantly outperforms strong baselines, including distilling from the strongest model with an absolute gain of up to 13.8%, averaged across benchmarks, and training on 3x more data with a maximum performance improvement of 17.3%. GRAPE's strong performance generalizes to realistic settings. We experiment with the post-training data used for Tulu3 and Olmo-2. GRAPE outperforms strong baselines trained on 4.5 times more data by 6.1% and a state-of-the-art data selection approach by 3% on average performance. Remarkably, using 1/3 of the data and half the number of epochs, GRAPE enables LLaMA3.1-8B to surpass the performance of Tulu3-SFT by 3.5%. 

**Abstract (ZH)**: 高质量的监督微调（SFT）数据对于从预训练的大语言模型（LLMs）中激发强大的能力至关重要。通常，指令会与来自其他LLMs的多个响应配对，这些响应往往不符合目标模型的分布。在大规模应用中，这可能导致边际效益递减，甚至损害模型的性能和稳健性。我们提出了**GRAPE**，这是一种新颖的SFT框架，能够考虑到目标模型的独特特性。对于每个指令，它从各种LLMs中收集响应，并选择目标模型测量概率最高的响应，表明该响应最符合目标模型的预训练分布；然后进行标准的SFT训练。

我们首先通过控制实验评估了GRAPE，从多个模型中抽取UltraInteract中的各种解决方案，并使用GRAPE选择的数据对LLaMA3.1-8B、Mistral-7B和Qwen2.5-7B等常用LMs进行微调。GRAPE显著优于强基准模型，包括从最强模型蒸馏，绝对增益最高可达13.8%，覆盖不同基准中的平均增益为13.8%，以及使用3倍多的数据训练，达到最大性能提升17.3%。GRAPE的强大性能能够推广到实际场景中。我们在Tulu3和Olmo-2的后训练数据中进行了实验。与训练在4.5倍更多数据上的强基准模型相比，GRAPE在平均性能上高出6.1%；与最先进的数据选择方法相比，平均性能提高3%。值得注意的是，使用数据量的1/3和一半的训练周期，GRAPE使LLaMA3.1-8B的性能超过了Tulu3-SFT 3.5%。 

---
# LLMs to Support a Domain Specific Knowledge Assistant 

**Title (ZH)**: 支持特定领域知识助手的大规模语言模型 

**Authors**: Maria-Flavia Lovin  

**Link**: [PDF](https://arxiv.org/pdf/2502.04095)  

**Abstract**: This work presents a custom approach to developing a domain specific knowledge assistant for sustainability reporting using the International Financial Reporting Standards (IFRS). In this domain, there is no publicly available question-answer dataset, which has impeded the development of a high-quality chatbot to support companies with IFRS reporting. The two key contributions of this project therefore are:
(1) A high-quality synthetic question-answer (QA) dataset based on IFRS sustainability standards, created using a novel generation and evaluation pipeline leveraging Large Language Models (LLMs). This comprises 1,063 diverse QA pairs that address a wide spectrum of potential user queries in sustainability reporting. Various LLM-based techniques are employed to create the dataset, including chain-of-thought reasoning and few-shot prompting. A custom evaluation framework is developed to assess question and answer quality across multiple dimensions, including faithfulness, relevance, and domain specificity. The dataset averages a score range of 8.16 out of 10 on these metrics.
(2) Two architectures for question-answering in the sustainability reporting domain - a RAG pipeline and a fully LLM-based pipeline. The architectures are developed by experimenting, fine-tuning, and training on the QA dataset. The final pipelines feature an LLM fine-tuned on domain specific data and an industry classification component to improve the handling of complex queries. The RAG architecture achieves an accuracy of 85.32% on single-industry and 72.15% on cross-industry multiple-choice questions, outperforming the baseline approach by 4.67 and 19.21 percentage points, respectively. The LLM-based pipeline achieves an accuracy of 93.45% on single-industry and 80.30% on cross-industry multiple-choice questions, an improvement of 12.80 and 27.36 percentage points over the baseline, respectively. 

**Abstract (ZH)**: 本文提出了一种定制方法，旨在利用国际财务报告准则（IFRS）开发一个专注于可持续报告的专用知识助手。在该领域中，没有公开可用的问答数据集，这阻碍了开发高质量的聊天机器人以支持使用IFRS进行报告的公司。因此，该项目的两大主要贡献是：
(1) 基于IFRS可持续性标准的一个高质量合成问答（QA）数据集，该数据集利用大型语言模型（LLMs）的新型生成和评估管道创建。该数据集包含1,063个多样化的问答对，涵盖了可持续报告中潜在用户查询的广泛范围。使用多种基于LLM的技术来创建数据集，包括链式推理和少样本提示。开发了一个自定义评估框架来从多个维度评估问题和答案的质量，包括忠实性、相关性和领域特定性。这些指标下该数据集平均得分为8.16/10。
(2) 在可持续报告领域中的两套问答架构——即检索与生成（Retrieval-Augmented Generation, RAG）流程和全基于LLM的流程。这些架构通过实验、微调和基于问答数据集进行训练而开发。最终的流程包括一个针对特定领域数据进行微调的LLM和一个行业分类组件，以提高处理复杂查询的能力。RAG架构在单一行业单选题中实现了85.32%的准确率，在跨行业多选题中实现了72.15%的准确率，分别超出基线方法4.67和19.21个百分点。基于LLM的流程在单一行业单选题中的准确率为93.45%，在跨行业多选题中的准确率为80.30%，分别提高了12.80和27.36个百分点。 

---
# Predicting Large Language Model Capabilities on Closed-Book QA Tasks Using Only Information Available Prior to Training 

**Title (ZH)**: 在仅使用训练前可用信息的情况下，预测大型语言模型在闭卷问答任务中的能力 

**Authors**: Changhao Jiang, Ming Zhang, Junjie Ye, Xiaoran Fan, Yifei Cao, Jiajun Sun, Zhiheng Xi, Shihan Dou, Yi Dong, Yujiong Shen, Jingqi Tong, Zhen Wang, Tao Liang, Zhihui Fei, Mingyang Wan, Guojun Ma, Qi Zhang, Tao Gui, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2502.04066)  

**Abstract**: The GPT-4 technical report from OpenAI suggests that model performance on specific tasks can be predicted prior to training, though methodologies remain unspecified. This approach is crucial for optimizing resource allocation and ensuring data alignment with target tasks. To achieve this vision, we focus on predicting performance on Closed-book Question Answering (CBQA) tasks, which are closely tied to pre-training data and knowledge retention. We address three major challenges: 1) mastering the entire pre-training process, especially data construction; 2) evaluating a model's knowledge retention; and 3) predicting task-specific knowledge retention using only information available prior to training. To tackle these challenges, we pre-train three large language models (i.e., 1.6B, 7B, and 13B) using 560k dollars and 520k GPU hours. We analyze the pre-training data with knowledge triples and assess knowledge retention using established methods. Additionally, we introduce the SMI metric, an information-theoretic measure that quantifies the relationship between pre-training data, model size, and task-specific knowledge retention. Our experiments reveal a strong linear correlation ($\text{R}^2 > 0.84$) between the SMI metric and the model's accuracy on CBQA tasks across models of varying sizes (i.e., 1.1B, 1.6B, 7B, and 13B). The dataset, model, and code are available at this https URL. 

**Abstract (ZH)**: OpenAI发布的GPT-4技术报告表明，在训练之前可以预测模型在特定任务上的表现，尽管具体方法尚未明确。这种预测方法对于优化资源配置和确保数据与目标任务的一致性至关重要。为了实现这一愿景，我们关注预测封闭书本问答（Closed-book Question Answering, CBQA）任务的表现，这类任务与预训练数据和知识保留密切相关。我们主要应对三个主要挑战：1）掌握整个预训练过程，特别是数据构建；2）评估模型的知识保留情况；3）仅使用训练前的信息预测任务特定的知识保留情况。为解决这些挑战，我们使用560,000美元和520,000个GPU小时对三种大语言模型（即1.6B、7B和13B）进行了预训练。我们使用知识三元组分析预训练数据，并采用已建立的方法评估知识保留情况。此外，我们引入了SMI度量，这是一种信息论测量方法，用于量化预训练数据、模型大小与任务特定知识保留之间的关系。我们的实验显示，SMI度量与不同规模模型（即1.1B、1.6B、7B和13B）在CBQA任务上的准确性之间存在显著的线性相关性（$R^2 > 0.84$）。数据集、模型和代码可从以下网址获得：[此链接处]。

请注意，这里的链接URL未给出，您需要将实际的链接地址填充到相应位置。 

---
# Ontology-Guided, Hybrid Prompt Learning for Generalization in Knowledge Graph Question Answering 

**Title (ZH)**: 基于本体引导的混合提示学习方法在知识图谱问答中的泛化能力提升 

**Authors**: Longquan Jiang, Junbo Huang, Cedric Möller, Ricardo Usbeck  

**Link**: [PDF](https://arxiv.org/pdf/2502.03992)  

**Abstract**: Most existing Knowledge Graph Question Answering (KGQA) approaches are designed for a specific KG, such as Wikidata, DBpedia or Freebase. Due to the heterogeneity of the underlying graph schema, topology and assertions, most KGQA systems cannot be transferred to unseen Knowledge Graphs (KGs) without resource-intensive training data. We present OntoSCPrompt, a novel Large Language Model (LLM)-based KGQA approach with a two-stage architecture that separates semantic parsing from KG-dependent interactions. OntoSCPrompt first generates a SPARQL query structure (including SPARQL keywords such as SELECT, ASK, WHERE and placeholders for missing tokens) and then fills them with KG-specific information. To enhance the understanding of the underlying KG, we present an ontology-guided, hybrid prompt learning strategy that integrates KG ontology into the learning process of hybrid prompts (e.g., discrete and continuous vectors). We also present several task-specific decoding strategies to ensure the correctness and executability of generated SPARQL queries in both stages. Experimental results demonstrate that OntoSCPrompt performs as well as SOTA approaches without retraining on a number of KGQA datasets such as CWQ, WebQSP and LC-QuAD 1.0 in a resource-efficient manner and can generalize well to unseen domain-specific KGs like DBLP-QuAD and CoyPu KG Code: \href{this https URL}{this https URL} 

**Abstract (ZH)**: 以下是经过学术规范翻译后的文本：

大多数现有的知识图谱问答（KGQA）方法都是为特定的知识图谱设计的，例如维基数据（Wikidata）、DBpedia或Freebase。由于底层图模式、拓扑结构和断言的异构性，大多数KGQA系统在没有密集训练数据的情况下无法被移植到未见过的知识图谱（KGs）中。我们提出了一种新的基于大型语言模型（LLM）的两阶段KGQA方法——OntoSCPrompt，该方法将语义解析与KG相关的交互过程分离。OntoSCPrompt首先生成SPARQL查询结构（包括SELECT、ASK、WHERE关键字及其缺失参数的占位符），然后填充KG特定的信息。为增强对底层KG的理解，我们提出了一种基于本体的混合提示学习策略，将KG本体整合到混合提示（例如离散向量和连续向量）的训练过程中。此外，我们还提出了一些具体任务的解码策略，以确保生成的SPARQL查询在两个阶段中正确且可执行。实验结果表明，OntoSCPrompt在诸如CWQ、WebQSP和LC-QuAD 1.0等KGQA数据集上，无需重新训练，便能高效地达到当前顶尖方法（SOTA）的性能，并能很好地泛化到未见过的特定领域KG，例如DBLP-QuAD和CoyPu KG。代码：\href{此处提供链接}{此处提供链接} 

---
# MAQInstruct: Instruction-based Unified Event Relation Extraction 

**Title (ZH)**: MAQInstruct：基于指令的统一事件关系抽取 

**Authors**: Jun Xu, Mengshu Sun, Zhiqiang Zhang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.03954)  

**Abstract**: Extracting event relations that deviate from known schemas has proven challenging for previous methods based on multi-class classification, MASK prediction, or prototype matching. Recent advancements in large language models have shown impressive performance through instruction tuning. Nevertheless, in the task of event relation extraction, instruction-based methods face several challenges: there are a vast number of inference samples, and the relations between events are non-sequential. To tackle these challenges, we present an improved instruction-based event relation extraction framework named MAQInstruct. Firstly, we transform the task from extracting event relations using given event-event instructions to selecting events using given event-relation instructions, which reduces the number of samples required for inference. Then, by incorporating a bipartite matching loss, we reduce the dependency of the instruction-based method on the generation sequence. Our experimental results demonstrate that MAQInstruct significantly improves the performance of event relation extraction across multiple LLMs. 

**Abstract (ZH)**: 基于多类分类、MASK预测或原型匹配的方法，抽取与已知模式相偏差的事件关系 proving 挑战重重。最近大语言模型的发展通过指令微调展现了令人印象深刻的性能。然而，在事件关系抽取任务中，基于指令的方法面临多个挑战：需要大量的推理样本，且事件之间的关系是非序贯的。为了应对这些挑战，我们提出了一种改进的基于指令的事件关系抽取框架，名为 MAQInstruct。首先，我们把任务从使用给定的事件-事件指令抽取事件关系，转变为使用给定的事件-关系指令选择事件，从而减少了所需的推理样本数量。然后，通过引入二分图匹配损失，我们减少了基于指令方法对生成序列的依赖性。我们的实验结果表明，MAQInstruct 显著提升了多种大语言模型在事件关系抽取任务中的性能。 

---
# Experiments with Large Language Models on Retrieval-Augmented Generation for Closed-Source Simulation Software 

**Title (ZH)**: 大型语言模型在闭源仿真软件检索增强生成实验中的应用 

**Authors**: Andreas Baumann, Peter Eberhard  

**Link**: [PDF](https://arxiv.org/pdf/2502.03916)  

**Abstract**: Large Language Models (LLMs) are increasingly helpful in text generation, even writing code in programming languages based on user prompts written in natural language. They are even applied to generate simulation models for multibody systems from natural language. Research results suggest that LLMs surpass the mere replication of existing code examples, where some LLMs have been trained on an open-source multibody simulation code. However, for closed-source simulation software, such results are not to be expected as their ideas and concepts might differ from other publicly available ones. LLMs can hallucinate for knowledge-intensive tasks, such as model creation, which can lead to wrong responses. This is especially the case for the LLM unknown closed-source simulation software. The same applies to other internal knowledge kept private to protect intellectual property or data privacy. The Retrieval-Augmented Generation (RAG) approach might yield a solution for these knowledge-intensive tasks. This paper explores the application of RAG to closed-source simulation software and presents first experiments. After a brief introduction to LLMs, the RAG approach, and the simulation method applied by the close-source simulation software, several examples are provided to test LLMs' knowledge of the simulation software and the creation of simulation models using two RAG systems. The examples show promising results indicating the benefits of applying RAG systems to closed-source simulation software, helping to access their knowledge. Nevertheless, they also reveal gaps in the applied information and open questions for further research. 

**Abstract (ZH)**: 大型语言模型（LLMs）在文本生成方面越来越有帮助，甚至可以根据自然语言撰写的用户提示编写编程语言代码。它们甚至被应用于根据自然语言生成多体系统模拟模型。研究结果表明，LLMs不仅能够复制现有的代码示例，有些LLMs还被训练在开放源代码多体系统模拟代码上。然而，对于封闭源代码的模拟软件，这样的结果并不常见，因为它们的理念和概念可能与其他公开可用的软件不同。对于知识密集型任务，如模型创建，LLMs可能会产生错误的回答，尤其是在未知的封闭源代码模拟软件情况下。同样，对于其他内部知识，这些知识被保留在私有信息中以保护知识产权或数据隐私，也存在这种情况。检索增强生成（RAG）方法可能为这些知识密集型任务提供解决方案。本文探讨了RAG在封闭源代码模拟软件中的应用，并报告了初步实验。文章首先简要介绍了大型语言模型、RAG方法以及封闭源代码模拟软件所采用的模拟方法，然后提供了几个实验示例，以测试大型语言模型对模拟软件的知识以及利用两个RAG系统创建模拟模型的情况。这些示例表明了应用RAG系统于封闭源代码模拟软件的潜在益处，有助于获取其知识。然而，这些示例也揭示了所应用信息的不足之处，并提出了进一步研究中的开放问题。 

---
# Improving Natural Language Understanding for LLMs via Large-Scale Instruction Synthesis 

**Title (ZH)**: 通过大规模指令合成提高大规模语言模型的自然语言理解能力 

**Authors**: Lin Yuan, Jun Xu, Honghao Gui, Mengshu Sun, Zhiqiang Zhang, Lei Liang, Jun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.03843)  

**Abstract**: High-quality, large-scale instructions are crucial for aligning large language models (LLMs), however, there is a severe shortage of instruction in the field of natural language understanding (NLU). Previous works on constructing NLU instructions mainly focus on information extraction (IE), neglecting tasks such as machine reading comprehension, question answering, and text classification. Furthermore, the lack of diversity in the data has led to a decreased generalization ability of trained LLMs in other NLU tasks and a noticeable decline in the fundamental model's general capabilities. To address this issue, we propose Hum, a large-scale, high-quality synthetic instruction corpus for NLU tasks, designed to enhance the NLU capabilities of LLMs. Specifically, Hum includes IE (either close IE or open IE), machine reading comprehension, text classification, and instruction generalist tasks, thereby enriching task diversity. Additionally, we introduce a human-LLMs collaborative mechanism to synthesize instructions, which enriches instruction diversity by incorporating guidelines, preference rules, and format variants. We conduct extensive experiments on 5 NLU tasks and 28 general capability evaluation datasets for LLMs. Experimental results show that Hum enhances the NLU capabilities of six LLMs by an average of 3.1\%, with no significant decline observed in other general capabilities. 

**Abstract (ZH)**: 高质量的大规模指令对于对齐大规模语言模型（LLMs）至关重要，然而，在自然语言理解（NLU）领域，高质量的大规模指令极度稀缺。此前有关构建NLU指令的工作主要关注于信息提取（IE），忽视了机器阅读理解、问答和文本分类等任务。此外，数据多样性的缺乏导致训练好的LLMs在其他NLU任务中的泛化能力降低，并且基本模型的基础能力也有所下降。为解决这一问题，我们提出了Hum，这是一个大规模、高质量的合成指令语料库，旨在增强LLMs的NLU能力。具体而言，Hum包括信息提取（无论是封闭的IE还是开放的IE）、机器阅读理解、文本分类和指令通用任务，从而丰富了任务多样性。此外，我们引入了一种人-LLMs协作机制来合成指令，通过融合准则、偏好规则和格式变化，进一步丰富指令多样性。我们在5个NLU任务和28个基础能力评估数据集上进行了广泛的实验。实验结果表明，Hum可以平均提高6个LLMs的NLU能力3.1%，并且在其他基础能力方面未观察到显著下降。 

---
# Syntriever: How to Train Your Retriever with Synthetic Data from LLMs 

**Title (ZH)**: Syntriever：如何使用大型语言模型生成的合成数据训练检索模型 

**Authors**: Minsang Kim, Seungjun Baek  

**Link**: [PDF](https://arxiv.org/pdf/2502.03824)  

**Abstract**: LLMs have boosted progress in many AI applications. Recently, there were attempts to distill the vast knowledge of LLMs into information retrieval systems. Those distillation methods mostly use output probabilities of LLMs which are unavailable in the latest black-box LLMs. We propose Syntriever, a training framework for retrievers using synthetic data from black-box LLMs. Syntriever consists of two stages. Firstly in the distillation stage, we synthesize relevant and plausibly irrelevant passages and augmented queries using chain-of-thoughts for the given queries. LLM is asked to self-verify the synthetic data for possible hallucinations, after which retrievers are trained with a loss designed to cluster the embeddings of relevant passages. Secondly in the alignment stage, we align the retriever with the preferences of LLMs. We propose a preference modeling called partial Plackett-Luce ranking to learn LLM preferences with regularization which prevents the model from deviating excessively from that trained in the distillation stage. Experiments show that Syntriever achieves state-of-the-art performances on benchmark datasets from various domains in nDCG@$K$. The code is available at \href{this https URL}{this https URL}. 

**Abstract (ZH)**: 大语言模型（LLMs）在许多AI应用中推动了进展。最近，有人尝试将LLMs的大量知识提炼到信息检索系统中。这些提炼方法大多使用了LLMs的输出概率，而在最新的黑盒LLMs中，这些输出概率是不可用的。我们提出了一种名为Syntriever的训练框架，用于使用黑盒LLMs生成的合成数据训练检索器。Syntriever包含两个阶段。首先，在提炼阶段，我们使用链式思考生成相关和可能不相关的段落以及增强查询。然后要求LLM对其生成的数据进行自我验证，以检查可能的幻觉，之后使用设计用于聚类相关段落嵌入的损失来训练检索器。其次，在对齐阶段，我们将检索器与LLMs的偏好对齐。我们提出了一种偏好的建模方法，即部分Plackett-Luce排名，以通过正则化学习LLMs的偏好，从而防止模型在提炼阶段训练时出现过度偏离。实验表明，Syntriever在不同领域的基准数据集上实现了最先进的nDCG@K性能。代码可以在\href{这个链接}{这个链接}获取。 

---
# Understanding and Supporting Formal Email Exchange by Answering AI-Generated Questions 

**Title (ZH)**: 理解并支持由AI生成的问题所驱动的形式化电子邮件交流 

**Authors**: Yusuke Miura, Chi-Lan Yang, Masaki Kuribayashi, Keigo Matsumoto, Hideaki Kuzuoka, Shigeo Morishima  

**Link**: [PDF](https://arxiv.org/pdf/2502.03804)  

**Abstract**: Replying to formal emails is time-consuming and cognitively demanding, as it requires polite phrasing and ensuring an adequate response to the sender's demands. Although systems with Large Language Models (LLM) were designed to simplify the email replying process, users still needed to provide detailed prompts to obtain the expected output. Therefore, we proposed and evaluated an LLM-powered question-and-answer (QA)-based approach for users to reply to emails by answering a set of simple and short questions generated from the incoming email. We developed a prototype system, ResQ, and conducted controlled and field experiments with 12 and 8 participants. Our results demonstrated that QA-based approach improves the efficiency of replying to emails and reduces workload while maintaining email quality compared to a conventional prompt-based approach that requires users to craft appropriate prompts to obtain email drafts. We discuss how QA-based approach influences the email reply process and interpersonal relationship dynamics, as well as the opportunities and challenges associated with using a QA-based approach in AI-mediated communication. 

**Abstract (ZH)**: 回复正式电子邮件既耗时又认知消耗大，因为它要求使用者采用礼貌的表达方式，并确保对发送者的请求做出充分回应。尽管大型语言模型（LLM）驱动的系统被设计来简化电子邮件回复过程，用户仍然需要提供详细的提示才能获得期望的输出。因此，我们提出并评估了使用基于问题和答案（QA）的方法，让使用者通过回答从收到的电子邮件生成的一系列简单且简短的问题来回复电子邮件。我们开发了一个原型系统ResQ，并在12名和8名参与者中分别进行了受控实验和实地实验。结果显示，与传统的基于提示的方法相比，基于QA的方法提高了电子邮件回复的效率，减轻了工作负担，同时保持了电子邮件的质量。我们讨论了基于QA的方法如何影响电子邮件回复过程及人际关系动态，并探讨了在人工智能中介的沟通中使用基于QA的方法所带来的机会和挑战。 

---
# It's All in The [MASK]: Simple Instruction-Tuning Enables BERT-like Masked Language Models As Generative Classifiers 

**Title (ZH)**: 《尽在掩码之中：简单的指令调优 enables 类似 BERT 的掩码语言模型作为生成分类器》

注意：这里的[MASK]在原文中被替换为中文中常见的掩码词“掩码”，同时保持了英文原文中的“enables”和“as”的翻译准确性，以符合学术规范。原文中的“[MASK]”似乎是填补模型名称的地方，实际模型名称应该根据上下文填补完整。如果有具体模型名称应替换，可以进一步提供信息以便更加准确地翻译。 

**Authors**: Benjamin Clavié, Nathan Cooper, Benjamin Warner  

**Link**: [PDF](https://arxiv.org/pdf/2502.03793)  

**Abstract**: While encoder-only models such as BERT and ModernBERT are ubiquitous in real-world NLP applications, their conventional reliance on task-specific classification heads can limit their applicability compared to decoder-based large language models (LLMs). In this work, we introduce ModernBERT-Large-Instruct, a 0.4B-parameter encoder model that leverages its masked language modelling (MLM) head for generative classification. Our approach employs an intentionally simple training loop and inference mechanism that requires no heavy pre-processing, heavily engineered prompting, or architectural modifications. ModernBERT-Large-Instruct exhibits strong zero-shot performance on both classification and knowledge-based tasks, outperforming similarly sized LLMs on MMLU and achieving 93% of Llama3-1B's MMLU performance with 60% less parameters. We also demonstrate that, when fine-tuned, the generative approach using the MLM head matches or even surpasses traditional classification-head methods across diverse NLU this http URL capability emerges specifically in models trained on contemporary, diverse data mixes, with models trained on lower volume, less-diverse data yielding considerably weaker performance. Although preliminary, these results demonstrate the potential of using the original generative masked language modelling head over traditional task-specific heads for downstream tasks. Our work suggests that further exploration into this area is warranted, highlighting many avenues for future improvements. 

**Abstract (ZH)**: 尽管BERT和ModernBERT这类编码器-only模型在实际的自然语言处理(NLP)应用中无处不在，但它们依赖于特定任务的分类头部，这在适用性上不如基于解码器的大规模语言模型(LLMs)。本文中，我们提出了ModernBERT-Large-Instruct，这是一种包含0.4B参数的编码器模型，它利用其掩码语言模型(MLM)头部进行生成性分类。我们的方法采用了一个简单且故意设计的训练循环和推理机制，无需进行重大的预处理、高度工程化的提示或架构修改。ModernBERT-Large-Instruct 在零样本性能上表现出色，不仅在分类任务上，也在基于知识的任务上同样表现突出。在MMLU基准测试上，它与相同规模的LLMs相比表现出色，并且使用60%较少的参数达到了Llama3-1B 93%的性能。我们还展示了，当进行微调时，使用MLM头部的生成方法在不同的NLU任务上与传统分类头部方法相比表现相当甚至更优。这一能力尤其在训练数据多样且丰富的模型中表现显著，而在训练数据较少且不那么多样化的模型中则表现较差。尽管这些结果是初步的，但它们表明，使用原始的生成性掩码语言模型头部替代传统的特定任务头部可能在下游任务中有很大的潜力。我们的研究表明，进一步探索这一领域是值得的，并为未来改进提供了多个方向。 

---
# Efficiently Generating Expressive Quadruped Behaviors via Language-Guided Preference Learning 

**Title (ZH)**: 通过语言导向的偏好学习高效生成富有表现力的四足机器人行为 

**Authors**: Jaden Clark, Joey Hejna, Dorsa Sadigh  

**Link**: [PDF](https://arxiv.org/pdf/2502.03717)  

**Abstract**: Expressive robotic behavior is essential for the widespread acceptance of robots in social environments. Recent advancements in learned legged locomotion controllers have enabled more dynamic and versatile robot behaviors. However, determining the optimal behavior for interactions with different users across varied scenarios remains a challenge. Current methods either rely on natural language input, which is efficient but low-resolution, or learn from human preferences, which, although high-resolution, is sample inefficient. This paper introduces a novel approach that leverages priors generated by pre-trained LLMs alongside the precision of preference learning. Our method, termed Language-Guided Preference Learning (LGPL), uses LLMs to generate initial behavior samples, which are then refined through preference-based feedback to learn behaviors that closely align with human expectations. Our core insight is that LLMs can guide the sampling process for preference learning, leading to a substantial improvement in sample efficiency. We demonstrate that LGPL can quickly learn accurate and expressive behaviors with as few as four queries, outperforming both purely language-parameterized models and traditional preference learning approaches. Website with videos: this https URL 

**Abstract (ZH)**: 情感化的机器人行为对于机器人在社交环境中被广泛接受至关重要。近年来，学习型腿足运动控制器的发展使机器人行为更加动态和多样化。然而，如何确定不同用户在不同场景下的最佳互动行为仍然是一个挑战。当前的方法要么依赖自然语言输入，这虽然高效但缺乏细节；要么通过学习人类的偏好，虽具有高细节但样本效率低下。本文提出了一种新的方法，该方法利用预训练的语言模型（LLM）生成的先验知识，并结合偏好学习的精确性。我们的方法称为语言引导的偏好学习（LGPL），使用语言模型生成初始行为样本，然后通过基于偏好的反馈进行细化，学习出与人类期望相符的行为。我们的核心洞察是，语言模型可以引导偏好学习的过程的采样，从而显著提高样本效率。我们证明，与纯语言参数模型和传统偏好学习方法相比，LGPL仅需四次查询即可快速学习出准确且富有表现力的行为。网站包含视频演示：[这个链接](这个链接请具体化为可访问的URL) 

---
# Boosting Knowledge Graph-based Recommendations through Confidence-Aware Augmentation with Large Language Models 

**Title (ZH)**: 通过大型语言模型aware置信度增强的知识图谱推荐提升 

**Authors**: Rui Cai, Chao Wang, Qianyi Cai, Dazhong Shen, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.03715)  

**Abstract**: Knowledge Graph-based recommendations have gained significant attention due to their ability to leverage rich semantic relationships. However, constructing and maintaining Knowledge Graphs (KGs) is resource-intensive, and the accuracy of KGs can suffer from noisy, outdated, or irrelevant triplets. Recent advancements in Large Language Models (LLMs) offer a promising way to improve the quality and relevance of KGs for recommendation tasks. Despite this, integrating LLMs into KG-based systems presents challenges, such as efficiently augmenting KGs, addressing hallucinations, and developing effective joint learning methods. In this paper, we propose the Confidence-aware KG-based Recommendation Framework with LLM Augmentation (CKG-LLMA), a novel framework that combines KGs and LLMs for recommendation task. The framework includes: (1) an LLM-based subgraph augmenter for enriching KGs with high-quality information, (2) a confidence-aware message propagation mechanism to filter noisy triplets, and (3) a dual-view contrastive learning method to integrate user-item interactions and KG data. Additionally, we employ a confidence-aware explanation generation process to guide LLMs in producing realistic explanations for recommendations. Finally, extensive experiments demonstrate the effectiveness of CKG-LLMA across multiple public datasets. 

**Abstract (ZH)**: 基于知识图谱的推荐方法由于能够利用丰富的语义关系而引起了广泛关注。然而，构建和维护知识图谱（KGs）耗费资源，且KGs的准确性可能会受到嘈杂、过时或无关的三元组的影响。近年来，大型语言模型（LLMs）的进步为提高KGs的质量和相关性提供了有前景的方式。尽管如此，将LLMs整合到基于KG的系统中仍面临一些挑战，例如高效地扩充KGs、解决幻觉问题以及开发有效的联合学习方法。在本文中，我们提出了基于知识图谱的认知增强推荐框架与LLM增强（CKG-LLMA），这是一种结合KGs和LLMs的创新框架，用于推荐任务。该框架包括：（1）基于LLM的子图扩充器，用于丰富KGs中的高质量信息；（2）认知增强的消息传播机制，用于过滤噪音三元组；（3）双视角对比学习方法，用于整合用户-物品交互和KG数据。此外，我们还采用认知增强的解释生成过程来引导LLMs生成与推荐相关的现实解释。最后，广泛的实验表明，CKG-LLMA在多个公开数据集上的有效性。 

---
# MultiQ&A: An Analysis in Measuring Robustness via Automated Crowdsourcing of Question Perturbations and Answers 

**Title (ZH)**: 多问答系统：通过自动众包问题扰动和答案生成来衡量鲁棒性的分析 

**Authors**: Nicole Cho, William Watson  

**Link**: [PDF](https://arxiv.org/pdf/2502.03711)  

**Abstract**: One critical challenge in the institutional adoption journey of Large Language Models (LLMs) stems from their propensity to hallucinate in generated responses. To address this, we propose MultiQ&A, a systematic approach for evaluating the robustness and consistency of LLM-generated answers. We demonstrate MultiQ&A's ability to crowdsource question perturbations and their respective answers through independent LLM agents at scale. Our experiments culminated in the examination of 1.9 million question perturbations and 2.3 million answers. Furthermore, MultiQ&A shows that ensembled LLMs, such as gpt-3.5-turbo, remain relatively robust and consistent under perturbations. MultiQ&A provides clarity in the response generation space, offering an effective method for inspecting disagreements and variability. Therefore, our system offers a potential framework for institutional LLM adoption with the ability to measure confidence, consistency, and the quantification of hallucinations. 

**Abstract (ZH)**: 大型语言模型（LLMs）在机构采用过程中面临的一个关键挑战在于它们在生成响应时容易出现幻觉。为了解决这一问题，我们提出了一种系统性的方法——MultiQ&A，用于评估LLM生成答案的稳健性和一致性。我们展示了MultiQ&A能够通过大规模独立的LLM代理来众包问题扰动及其相应的答案。我们的实验最终检查了190万种问题扰动和230万份答案。此外，MultiQ&A表明，如gpt-3.5-turbo等集成的LLM在扰动下仍保持相对的稳健性和一致性。MultiQ&A为响应生成空间提供了清晰度，提供了一种有效的方法来检查分歧和变化性。因此，我们的系统为机构LLM采用提供了一种潜在的框架，能够衡量信心、一致性和幻觉的量化。 

---
# Aggregate and conquer: detecting and steering LLM concepts by combining nonlinear predictors over multiple layers 

**Title (ZH)**: 聚合与征服：通过结合多层上的非线性预测器来检测和引导大模型概念 

**Authors**: Daniel Beaglehole, Adityanarayanan Radhakrishnan, Enric Boix-Adserà, Mikhail Belkin  

**Link**: [PDF](https://arxiv.org/pdf/2502.03708)  

**Abstract**: A trained Large Language Model (LLM) contains much of human knowledge. Yet, it is difficult to gauge the extent or accuracy of that knowledge, as LLMs do not always ``know what they know'' and may even be actively misleading. In this work, we give a general method for detecting semantic concepts in the internal activations of LLMs. Furthermore, we show that our methodology can be easily adapted to steer LLMs toward desirable outputs. Our innovations are the following: (1) we use a nonlinear feature learning method to identify important linear directions for predicting concepts from each layer; (2) we aggregate features across layers to build powerful concept detectors and steering mechanisms. We showcase the power of our approach by attaining state-of-the-art results for detecting hallucinations, harmfulness, toxicity, and untruthful content on seven benchmarks. We highlight the generality of our approach by steering LLMs towards new concepts that, to the best of our knowledge, have not been previously considered in the literature, including: semantic disambiguation, human languages, programming languages, hallucinated responses, science subjects, poetic/Shakespearean English, and even multiple concepts simultaneously. Moreover, our method can steer concepts with numerical attributes such as product reviews. We provide our code (including a simple API for our methods) at this https URL . 

**Abstract (ZH)**: 经过训练的大语言模型（LLM）包含了大量的人类知识。然而，很难评估这些知识的范围和准确性，因为LLM们并不总是知道自己知道什么，有时甚至会主动误导。在这项工作中，我们提出了一种通用方法来检测LLM内部激活中的语义概念。此外，我们展示了我们的方法可以轻松地调整，引导LLM生成期望的输出。我们的创新之处在于：（1）我们使用非线性特征学习方法来识别从每一层预测概念的重要线性方向；（2）我们跨层聚合特征，构建强大的概念检测器和引导机制。我们通过在七个基准测试中达到最先进的检测幻觉、危害性、毒性及不实内容的结果，展示了我们方法的能力。我们通过引导LLM生成以前文献中尚未考虑的新概念，突显了我们方法的通用性，这些新概念包括：语义消歧、人类语言、编程语言、幻觉回答、科学学科、诗体/莎士比亚英语，甚至同时处理多个概念。此外，我们的方法可以引导具有数值属性的概念，例如产品评论。我们已将代码（包括我们方法的简单API）发布在如下网址：[此链接]。 

---
# LLM Alignment as Retriever Optimization: An Information Retrieval Perspective 

**Title (ZH)**: LLM对齐作为检索器优化：从信息检索视角看待 

**Authors**: Bowen Jin, Jinsung Yoon, Zhen Qin, Ziqi Wang, Wei Xiong, Yu Meng, Jiawei Han, Sercan O. Arik  

**Link**: [PDF](https://arxiv.org/pdf/2502.03699)  

**Abstract**: Large Language Models (LLMs) have revolutionized artificial intelligence with capabilities in reasoning, coding, and communication, driving innovation across industries. Their true potential depends on effective alignment to ensure correct, trustworthy and ethical behavior, addressing challenges like misinformation, hallucinations, bias and misuse. While existing Reinforcement Learning (RL)-based alignment methods are notoriously complex, direct optimization approaches offer a simpler alternative. In this work, we introduce a novel direct optimization approach for LLM alignment by drawing on established Information Retrieval (IR) principles. We present a systematic framework that bridges LLM alignment and IR methodologies, mapping LLM generation and reward models to IR's retriever-reranker paradigm. Building on this foundation, we propose LLM Alignment as Retriever Preference Optimization (LarPO), a new alignment method that enhances overall alignment quality. Extensive experiments validate LarPO's effectiveness with 38.9 % and 13.7 % averaged improvement on AlpacaEval2 and MixEval-Hard respectively. Our work opens new avenues for advancing LLM alignment by integrating IR foundations, offering a promising direction for future research. 

**Abstract (ZH)**: 大规模语言模型（LLMs）通过其推理、编程和通信能力革新了人工智能领域，并正在各行各业中推动创新。其真正潜力取决于有效的对齐，以确保正确的、可信的和符合伦理的行为，解决诸如假信息、幻觉、偏见和滥用等问题。虽然现有的基于强化学习（RL）的对齐方法非常复杂，直接优化的方法提供了一种更简单的选择。在本文中，我们通过借鉴成熟的检索原理（IR），介绍了一种新的直接优化方法，用于LLM对齐。我们提出了一种系统框架，将LLM对齐和检索方法论联系起来，将LLM生成和奖励模型映射到检索器-重排序器范式。基于这一基础，我们提出了LLM对齐作为一种检索偏好优化（LarPO）的新方法，以提高整体对齐质量。广泛的实验验证了LarPO的有效性，分别在AlpacaEval2和MixEval-Hard上平均提高了38.9%和13.7%。我们的工作通过整合检索基础，为LLM对齐开辟了新的途径，并为未来的研究提供了富有前景的方向。 

---
# Reflection-Window Decoding: Text Generation with Selective Refinement 

**Title (ZH)**: 反射窗口解码：选择性细化的文本生成 

**Authors**: Zeyu Tang, Zhenhao Chen, Loka Li, Xiangchen Song, Yunlong Deng, Yifan Shen, Guangyi Chen, Peter Spirtes, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.03678)  

**Abstract**: The autoregressive decoding for text generation in large language models (LLMs), while widely used, is inherently suboptimal due to the lack of a built-in mechanism to perform refinement and/or correction of the generated content. In this paper, we consider optimality in terms of the joint probability over the generated response, when jointly considering all tokens at the same time. We theoretically characterize the potential deviation of the autoregressively generated response from its globally optimal counterpart that is of the same length. Our analysis suggests that we need to be cautious when noticeable uncertainty arises during text generation, which may signal the sub-optimality of the generation history. To address the pitfall of autoregressive decoding for text generation, we propose an approach that incorporates a sliding reflection window and a pausing criterion, such that refinement and generation can be carried out interchangeably as the decoding proceeds. Our selective refinement framework strikes a balance between efficiency and optimality, and our extensive experimental results demonstrate the effectiveness of our approach. 

**Abstract (ZH)**: 在大规模语言模型（LLMs）中，自回归解码方法因其生成文本时缺乏内置的修正机制而在文本生成中广泛使用，但本质上是次优的。本文从同时联合考虑生成响应中所有标记的联合概率的角度，探讨最优性。我们理论分析了自回归生成的响应与同长度的全局最优响应之间的潜在偏差。我们的分析表明，在文本生成过程中出现明显的不确定性时，我们需要注意生成历史的次优性。为解决自回归解码方法在文本生成中的缺陷，我们提出了一种结合滑动反射窗口和暂停准则的方法，使得在解码过程中可以交替进行修正和生成。我们的选择性修正框架在效率和最优性之间取得了平衡，广泛实验结果证明了该方法的有效性。 

---
# Advancing Reasoning in Large Language Models: Promising Methods and Approaches 

**Title (ZH)**: 大语言模型中推理能力的提升：前景方法与途径 

**Authors**: Avinash Patil  

**Link**: [PDF](https://arxiv.org/pdf/2502.03671)  

**Abstract**: Large Language Models (LLMs) have succeeded remarkably in various natural language processing (NLP) tasks, yet their reasoning capabilities remain a fundamental challenge. While LLMs exhibit impressive fluency and factual recall, their ability to perform complex reasoning-spanning logical deduction, mathematical problem-solving, commonsense inference, and multi-step reasoning-often falls short of human expectations. This survey provides a comprehensive review of emerging techniques enhancing reasoning in LLMs. We categorize existing methods into key approaches, including prompting strategies (e.g., Chain-of-Thought reasoning, Self-Consistency, and Tree-of-Thought reasoning), architectural innovations (e.g., retrieval-augmented models, modular reasoning networks, and neuro-symbolic integration), and learning paradigms (e.g., fine-tuning with reasoning-specific datasets, reinforcement learning, and self-supervised reasoning objectives). Additionally, we explore evaluation frameworks used to assess reasoning in LLMs and highlight open challenges, such as hallucinations, robustness, and reasoning generalization across diverse tasks. By synthesizing recent advancements, this survey aims to provide insights into promising directions for future research and practical applications of reasoning-augmented LLMs. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种自然语言处理（NLP）任务中取得了显著的成功，但其推理能力仍然是一个根本性的挑战。虽然LLMs在流畅性和事实回忆方面表现出色，但在进行复杂的推理（如逻辑演绎、数学问题解决、常识推理和多步推理）时，往往未能达到人类的期望。本文综述了增强LLMs推理能力的新兴技术。我们根据方法的关键思路将现有技术分为几类，包括提示策略（例如，链条式推理、自我一致性推理和思维树推理）、架构创新（例如，检索增强模型、模块化推理网络和神经符号融合）以及学习范式（例如，以推理特定数据集为基础的微调、强化学习和自监督的推理目标）。此外，我们还探讨了用于评估LLMs推理能力的评估框架，并指出了开放性挑战，如涌现、鲁棒性和跨多元任务的推理泛化。通过综合最新的进展，本文旨在为未来增强推理能力的LLMs的研究和应用提供有价值的见解。 

---
# AdaPhish: AI-Powered Adaptive Defense and Education Resource Against Deceptive Emails 

**Title (ZH)**: AdaPhish：面向欺诈邮件的AI驱动自适应防御与教育资源 

**Authors**: Rei Meguro, Ng S. T. Chong  

**Link**: [PDF](https://arxiv.org/pdf/2502.03622)  

**Abstract**: Phishing attacks remain a significant threat in the digital age, yet organizations lack effective methods to tackle phishing attacks without leaking sensitive information. Phish bowl initiatives are a vital part of cybersecurity efforts against these attacks. However, traditional phish bowls require manual anonymization and are often limited to internal use. To overcome these limitations, we introduce AdaPhish, an AI-powered phish bowl platform that automatically anonymizes and analyzes phishing emails using large language models (LLMs) and vector databases. AdaPhish achieves real-time detection and adaptation to new phishing tactics while enabling long-term tracking of phishing trends. Through automated reporting, adaptive analysis, and real-time alerts, AdaPhish presents a scalable, collaborative solution for phishing detection and cybersecurity education. 

**Abstract (ZH)**: 在数字时代，网络钓鱼攻击仍然构成重大威胁，但组织缺乏有效的方法来应对网络钓鱼攻击而不泄露敏感信息。反网络钓鱼举措是应对这些攻击的网络安全努力的重要组成部分。然而，传统的反网络钓鱼举措通常需要手动匿名化处理，并且往往仅限于内部使用。为克服这些限制，我们提出了AdaPhish，这是一个基于人工智能的反网络钓鱼平台，利用大规模语言模型（LLMs）和向量数据库自动匿名化和分析网络钓鱼邮件。AdaPhish 实现了对新网络钓鱼手法的实时检测和适应，并能长期跟踪网络钓鱼趋势。通过自动化报告、适应性分析和实时警报，AdaPhish 提供了一个可扩展、协作的解决方案，用于网络钓鱼检测和网络安全教育。 

---
# Code Simulation as a Proxy for High-order Tasks in Large Language Models 

**Title (ZH)**: 将代码模拟作为大型语言模型执行高阶任务的代理 

**Authors**: Emanuele La Malfa, Christoph Weinhuber, Orazio Torre, Fangru Lin, X. Angelo Huang, Samuele Marro, Anthony Cohn, Nigel Shadbolt, Michael Wooldridge  

**Link**: [PDF](https://arxiv.org/pdf/2502.03568)  

**Abstract**: Many reasoning, planning, and problem-solving tasks share an intrinsic algorithmic nature: correctly simulating each step is a sufficient condition to solve them correctly. We collect pairs of naturalistic and synthetic reasoning tasks to assess the capabilities of Large Language Models (LLM). While naturalistic tasks often require careful human handcrafting, we show that synthetic data is, in many cases, a good proxy that is much easier to collect at scale. We leverage common constructs in programming as the counterpart of the building blocks of naturalistic reasoning tasks, such as straight-line programs, code that contains critical paths, and approximate and redundant instructions. We further assess the capabilities of LLMs on sorting problems and repeated operations via sorting algorithms and nested loops. Our synthetic datasets further reveal that while the most powerful LLMs exhibit relatively strong execution capabilities, the process is fragile: it is negatively affected by memorisation and seems to rely heavily on pattern recognition. Our contribution builds upon synthetically testing the reasoning capabilities of LLMs as a scalable complement to handcrafted human-annotated problems. 

**Abstract (ZH)**: 许多推理、规划和问题解决任务具有固有的算法性质：正确模拟每个步骤是正确解决它们的充分条件。我们收集了真实世界和合成的推理任务以评估大型语言模型（LLM）的能力。尽管真实世界的任务经常需要精心的人工设计，但我们发现，在许多情况下，合成数据是一个很好的替代品，可以更容易大规模采集。我们利用编程中的常见结构作为真实世界推理任务的构建块，如直线程序、包含关键路径的代码和近似的冗余指令。我们还通过排序问题和嵌套循环中的排序算法进一步评估了LLM的能力。我们的合成数据集进一步揭示出，尽管最强大的LLM表现出相对较强的执行能力，但该过程是脆弱的：其受记忆的影响较大，并且似乎主要依赖于模式识别。我们的贡献在于合成测试LLM的推理能力，作为手工设计的人类标注问题的可扩展补充。 

---
# An Empirical Exploration of ChatGPT's Ability to Support Problem Formulation Tasks for Mission Engineering and a Documentation of its Performance Variability 

**Title (ZH)**: 对ChatGPT在任务工程中支持问题表述任务能力的实证探索及其性能变异性文档 

**Authors**: Max Ofsa, Taylan G. Topcu  

**Link**: [PDF](https://arxiv.org/pdf/2502.03511)  

**Abstract**: Systems engineering (SE) is evolving with the availability of generative artificial intelligence (AI) and the demand for a systems-of-systems perspective, formalized under the purview of mission engineering (ME) in the US Department of Defense. Formulating ME problems is challenging because they are open-ended exercises that involve translation of ill-defined problems into well-defined ones that are amenable for engineering development. It remains to be seen to which extent AI could assist problem formulation objectives. To that end, this paper explores the quality and consistency of multi-purpose Large Language Models (LLM) in supporting ME problem formulation tasks, specifically focusing on stakeholder identification. We identify a relevant reference problem, a NASA space mission design challenge, and document ChatGPT-3.5's ability to perform stakeholder identification tasks. We execute multiple parallel attempts and qualitatively evaluate LLM outputs, focusing on both their quality and variability. Our findings portray a nuanced picture. We find that the LLM performs well in identifying human-focused stakeholders but poorly in recognizing external systems and environmental factors, despite explicit efforts to account for these. Additionally, LLMs struggle with preserving the desired level of abstraction and exhibit a tendency to produce solution specific outputs that are inappropriate for problem formulation. More importantly, we document great variability among parallel threads, highlighting that LLM outputs should be used with caution, ideally by adopting a stochastic view of their abilities. Overall, our findings suggest that, while ChatGPT could reduce some expert workload, its lack of consistency and domain understanding may limit its reliability for problem formulation tasks. 

**Abstract (ZH)**: 系统工程（SE）随着生成式人工智能（AI）的可用性以及对系统族视图（SoS）的需求而不断发展，这一需求在美国国防部的事务工程（ME）框架下被正式化。制定ME问题极具挑战性，因为这些问题是开放式的；涉及将不明确的问题转换为明确的问题，以便于工程开发。目前尚不清楚AI在多大程度上能帮助解决这一问题。为此，本文探讨了多用途大型语言模型（LLM）在支持ME问题制定任务中的质量和一致性，特别是在利益相关者识别方面的作用。我们确定了一个相关的参考问题，即NASA的空间任务设计挑战，并记录了ChatGPT-3.5在执行利益相关者识别任务方面的表现。我们执行了多次平行尝试，并对LLM输出进行了定性的评估，重点关注其质量和变异性。我们的研究结果呈现了一个复杂的图景。我们发现，LLM 在识别人类导向的利益相关者方面表现良好，但在识别外部系统和环境因素方面表现不佳，尽管尝试将这些因素纳入考量。此外，LLM 在保持合适的抽象水平方面存在问题，并倾向于生成特定于解决方案的输出，这在问题制定过程中是不合适的。更重要的是，我们记录了平行线程之间的显著变异性，这表明LLM输出应谨慎使用，理想情况下应采用其能力的随机视角。总体而言，我们的研究结果表明，虽然ChatGPT 可以减少一些专家的工作量，但由于缺乏一致性和领域的理解，它可能不适合用于问题制定任务。 

---
# Teaching Language Models to Critique via Reinforcement Learning 

**Title (ZH)**: 通过强化学习教学语言模型进行评论 

**Authors**: Zhihui Xie, Jie chen, Liyu Chen, Weichao Mao, Jingjing Xu, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2502.03492)  

**Abstract**: Teaching large language models (LLMs) to critique and refine their outputs is crucial for building systems that can iteratively improve, yet it is fundamentally limited by the ability to provide accurate judgments and actionable suggestions. In this work, we study LLM critics for code generation and propose $\texttt{CTRL}$, a framework for $\texttt{C}$ritic $\texttt{T}$raining via $\texttt{R}$einforcement $\texttt{L}$earning, which trains a critic model to generate feedback that maximizes correction performance for a fixed generator model without human supervision. Our results demonstrate that critics trained with $\texttt{CTRL}$ significantly enhance pass rates and mitigate compounding errors across both base and stronger generator models. Furthermore, we show that these critic models act as accurate generative reward models and enable test-time scaling through iterative critique-revision, achieving up to 106.1% relative improvements across challenging code generation benchmarks. 

**Abstract (ZH)**: 训练大型语言模型（LLMs）批判和改进其输出对于构建能够迭代改进的系统至关重要，但这从根本上受到了提供准确判断和可行建议能力的限制。在本研究中，我们探讨了代码生成领域的LLM批判方法，并提出了一个基于强化学习的框架$\texttt{CTRL}$（Critic Training via Reinforcement Learning），该框架用于训练一个批判模型，使其能够生成反馈，以最大化特定固定生成器模型的纠正性能，无需人类监督。我们的结果显示，使用$\texttt{CTRL}$训练的批判模型显著提升了通过率，并减轻了不同基础和更强生成器模型中的累积错误。此外，我们还展示了这些批判模型作为准确生成奖励模型的作用，并通过迭代批判与修订实现测试时的扩展，从而在挑战性的代码生成基准测试中实现了高达106.1%的相对性能提升。 

---
# ScoreFlow: Mastering LLM Agent Workflows via Score-based Preference Optimization 

**Title (ZH)**: ScoreFlow：通过基于分数的偏好优化掌握LLM代理工作流 

**Authors**: Yinjie Wang, Ling Yang, Guohao Li, Mengdi Wang, Bryon Aragam  

**Link**: [PDF](https://arxiv.org/pdf/2502.04306)  

**Abstract**: Recent research has leveraged large language model multi-agent systems for complex problem-solving while trying to reduce the manual effort required to build them, driving the development of automated agent workflow optimization methods. However, existing methods remain inflexible due to representational limitations, a lack of adaptability, and poor scalability when relying on discrete optimization techniques. We address these challenges with ScoreFlow, a simple yet high-performance framework that leverages efficient gradient-based optimization in a continuous space. ScoreFlow incorporates Score-DPO, a novel variant of the direct preference optimization method that accounts for quantitative feedback. Across six benchmarks spanning question answering, coding, and mathematical reasoning, ScoreFlow achieves an 8.2% improvement over existing baselines. Moreover, it empowers smaller models to outperform larger ones with lower inference costs. Project: this https URL 

**Abstract (ZH)**: 近年来，研究人员利用大型语言模型多智能体系统来解决复杂问题，并试图减少构建这些系统的手工努力，从而推动了自动智能体工作流优化方法的发展。然而，现有的方法由于表示限制、缺乏适应性和依赖离散优化技术时的不佳可扩展性，仍然不够灵活。我们通过ScoreFlow框架解决了这些问题，该框架使用高效的基于梯度的优化方法在连续空间中运行，简单且高性能。ScoreFlow整合了Score-DPO，这是一种直接偏好优化的新变体，能够考虑定量反馈。在涵盖问答、编程和数学推理的六个基准测试中，ScoreFlow在现有基线方法上实现了8.2%的改进。此外，它还使较小的模型在较低的推理成本下能够超越较大的模型。项目链接：[该项目链接] 

---
# Beyond Prompt Content: Enhancing LLM Performance via Content-Format Integrated Prompt Optimization 

**Title (ZH)**: 超越提示内容：通过内容-格式集成提示优化提升大型语言模型性能 

**Authors**: Yuanye Liu, Jiahang Xu, Li Lyna Zhang, Qi Chen, Xuan Feng, Yang Chen, Zhongxin Guo, Yuqing Yang, Cheng Peng  

**Link**: [PDF](https://arxiv.org/pdf/2502.04295)  

**Abstract**: Large Language Models (LLMs) have shown significant capability across various tasks, with their real-world effectiveness often driven by prompt design. While recent research has focused on optimizing prompt content, the role of prompt formatting, a critical but often overlooked dimension, has received limited systematic investigation. In this paper, we introduce Content-Format Integrated Prompt Optimization (CFPO), an innovative methodology that jointly optimizes both prompt content and formatting through an iterative refinement process. CFPO leverages natural language mutations to explore content variations and employs a dynamic format exploration strategy that systematically evaluates diverse format options. Our extensive evaluations across multiple tasks and open-source LLMs demonstrate that CFPO demonstrates measurable performance improvements compared to content-only optimization methods. This highlights the importance of integrated content-format optimization and offers a practical, model-agnostic approach to enhancing LLM performance. Code will be available at this https URL. 

**Abstract (ZH)**: 以下是将该论文内容或标题翻译成中文，并符合学术规范的版本：

大型语言模型（LLMs）在各种任务中的表现显示出显著的能力，其实际应用效果往往由提示设计驱动。尽管近期研究主要集中在优化提示内容上，但提示格式——这一关键但常被忽视的维度——尚未得到系统性的研究。本文提出了一种创新方法，即内容-格式综合优化（Content-Format Integrated Prompt Optimization, CFPO），该方法通过迭代优化过程同时优化提示内容和格式。CFPO 利用自然语言变异来探索内容变化，并采用一种动态格式探索策略，系统性地评估多种格式选项。在多个任务和开源语言模型上的广泛评估表明，与仅优化内容的方法相比，CFPO 在性能上具有可测量的提升。这强调了综合内容和格式优化的重要性，并提供了一种鲁棒的、模型无关的方法来提升语言模型的性能。相关代码将发布在以下链接：[此处替换为实际链接]。

请注意，上述翻译保留了原文的专业术语和学术风格。对于“this https URL”中的实际链接部分，需要根据实际情况进行替换。 

---
# The Order Effect: Investigating Prompt Sensitivity in Closed-Source LLMs 

**Title (ZH)**: 有序效应：探究封闭源代码大语言模型的提示敏感性 

**Authors**: Bryan Guan, Tanya Roosta, Peyman Passban, Mehdi Rezagholizadeh  

**Link**: [PDF](https://arxiv.org/pdf/2502.04134)  

**Abstract**: As large language models (LLMs) become integral to diverse applications, ensuring their reliability under varying input conditions is crucial. One key issue affecting this reliability is order sensitivity, wherein slight variations in input arrangement can lead to inconsistent or biased outputs. Although recent advances have reduced this sensitivity, the problem remains unresolved. This paper investigates the extent of order sensitivity in closed-source LLMs by conducting experiments across multiple tasks, including paraphrasing, relevance judgment, and multiple-choice questions. Our results show that input order significantly affects performance across tasks, with shuffled inputs leading to measurable declines in output accuracy. Few-shot prompting demonstrates mixed effectiveness and offers partial mitigation, however, fails to fully resolve the problem. These findings highlight persistent risks, particularly in high-stakes applications, and point to the need for more robust LLMs or improved input-handling techniques in future development. 

**Abstract (ZH)**: 随着大规模语言模型（LLMs）在各种应用中发挥越来越重要的作用，确保它们在不同输入条件下的可靠性变得至关重要。影响这一可靠性的关键问题是顺序敏感性，即输入排列的微小变动会导致输出结果不一致或存在偏差。尽管近期的研究已经降低了一些敏感性，但这一问题仍未得到彻底解决。本文通过在多个任务（包括重述、相关性判断和多项选择题）上进行实验，探讨了闭源LLMs中的顺序敏感性程度。实验结果显示，输入顺序对任务性能产生了显著影响，打乱输入会导致输出准确性下降。少量的提示技巧显示其效果不一，并能在一定程度上缓解该问题，但并未完全解决。这些发现强调了高风险应用场景下的持续风险，并指出了未来开发中需要更稳健的LLMs或改进的输入处理技术。 

---
# AttentionPredictor: Temporal Pattern Matters for Efficient LLM Inference 

**Title (ZH)**: 注意预测器：时间模式对于高效大型语言模型推理至关重要 

**Authors**: Qingyue Yang, Jie Wang, Xing Li, Zhihai Wang, Chen Chen, Lei Chen, Xianzhi Yu, Wulong Liu, Jianye Hao, Mingxuan Yuan, Bin Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.04077)  

**Abstract**: With the development of large language models (LLMs), efficient inference through Key-Value (KV) cache compression has attracted considerable attention, especially for long-context generation. To compress the KV cache, recent methods identify critical KV tokens through heuristic ranking with attention scores. However, these methods often struggle to accurately determine critical tokens as they neglect the \textit{temporal patterns} in attention scores, resulting in a noticeable degradation in LLM performance. To address this challenge, we propose AttentionPredictor, which is the first learning-based critical token identification approach. Specifically, AttentionPredictor learns a lightweight convolution model to capture spatiotemporal patterns and predict the next-token attention score. An appealing feature of AttentionPredictor is that it accurately predicts the attention score while consuming negligible memory. Moreover, we propose a cross-token critical cache prefetching framework that hides the token estimation time overhead to accelerate the decoding stage. By retaining most of the attention information, AttentionPredictor achieves 16$\times$ KV cache compression with comparable LLM performance, significantly outperforming the state-of-the-art. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）的发展，通过键值（KV）缓存压缩进行高效推理吸引了大量关注，尤其是在长上下文生成方面。为了压缩KV缓存，最近的方法通过带注意分数启发式排名来识别关键的KV令牌。然而，这些方法往往难以准确确定关键令牌，因为它们忽略了注意分数中的时间模式，导致LLM性能显著下降。为了解决这一挑战，我们提出了AttentionPredictor，这是首个基于学习的关键令牌识别方法。具体来说，AttentionPredictor 学习了一个轻量级卷积模型以捕获时空模式并预测下一个令牌的注意分数。AttentionPredictor 的一个吸引人的特点是，它在几乎不消耗额外内存的情况下准确预测注意力分数。此外，我们提出了一个跨令牌关键缓存预取框架，以隐藏令牌估计时间的开销来加速解码阶段。通过保留大量注意信息，AttentionPredictor 实现了16倍的KV缓存压缩，且保持了可比的LLM性能，显著优于现有最佳方法。 

---
# Quantification of Biodiversity from Historical Survey Text with LLM-based Best-Worst Scaling 

**Title (ZH)**: 基于LLM的最优最劣标度法在历史调查文本中生物多样性量化中的应用 

**Authors**: Thomas Haider, Tobias Perschl, Malte Rehbein  

**Link**: [PDF](https://arxiv.org/pdf/2502.04022)  

**Abstract**: In this study, we evaluate methods to determine the frequency of species via quantity estimation from historical survey text. To that end, we formulate classification tasks and finally show that this problem can be adequately framed as a regression task using Best-Worst Scaling (BWS) with Large Language Models (LLMs). We test Ministral-8B, DeepSeek-V3, and GPT-4, finding that the latter two have reasonable agreement with humans and each other. We conclude that this approach is more cost-effective and similarly robust compared to a fine-grained multi-class approach, allowing automated quantity estimation across species. 

**Abstract (ZH)**: 在本研究中，我们评估了通过历史调查文本的数量估计来确定物种频率的方法。为此，我们制定了分类任务，并最终展示了可以使用最佳最差标度法（BWS）与大规模语言模型（LLMs）将该问题适当地框定为回归任务。我们测试了Ministral-8B、DeepSeek-V3和GPT-4，发现后两者与人类以及其他模型之间具有合理的共识。我们得出结论，这种approach比细粒度的多类别方法更具成本效益且同样稳健，能够实现跨物种的自动化数量估计。 

---
# BOLT: Bootstrap Long Chain-of-Thought in Language Models without Distillation 

**Title (ZH)**: BOLT：无需知识蒸馏的语言模型链式思考强化抽样 

**Authors**: Bo Pang, Hanze Dong, Jiacheng Xu, Silvio Savarese, Yingbo Zhou, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.03860)  

**Abstract**: Large language models (LLMs), such as o1 from OpenAI, have demonstrated remarkable reasoning capabilities. o1 generates a long chain-of-thought (LongCoT) before answering a question. LongCoT allows LLMs to analyze problems, devise plans, reflect, and backtrack effectively. These actions empower LLM to solve complex problems. After the release of o1, many teams have attempted to replicate its LongCoT and reasoning capabilities. In terms of methods, they primarily rely on knowledge distillation with data from existing models with LongCoT capacities (e.g., OpenAI-o1, Qwen-QwQ, DeepSeek-R1-Preview), leaving significant uncertainties on systematically developing such reasoning abilities. In terms of data domains, these works focus narrowly on math while a few others include coding, limiting their generalizability. This paper introduces a novel approach to enable LLM's LongCoT capacity without distillation from o1-like models or expensive human annotations, where we bootstrap LongCoT (BOLT) from a standard instruct model. BOLT involves three stages: 1) LongCoT data bootstrapping with in-context learning on a standard instruct model; 2) LongCoT supervised finetuning; 3) online training to further refine LongCoT capacities. In BOLT, only a few in-context examples need to be constructed during the bootstrapping stage; in our experiments, we created 10 examples, demonstrating the feasibility of this approach. We use Llama-3.1-70B-Instruct to bootstrap LongCoT and apply our method to various model scales (7B, 8B, 70B). We achieve impressive performance on a variety of benchmarks, Arena-Hard, MT-Bench, WildBench, ZebraLogic, MATH500, which evaluate diverse task-solving and reasoning capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs），如来自OpenAI的o1，展示了显著的推理能力。o1在回答问题之前会生成一个长链推理（LongCoT）。LongCoT使LLMs能够分析问题、制定计划、反思和回溯，这些操作赋予LLMs解决复杂问题的能力。在o1发布之后，许多研究团队尝试模仿其LongCoT和推理能力。在方法方面，他们主要依赖知识蒸馏，并使用具有LongCoT能力的现有模型的数据（例如OpenAI-o1、Qwen-QwQ、DeepSeek-R1-Preview），但系统发展这些推理能力仍然存在显著的不确定性和挑战。在数据领域方面，这些研究主要集中在数学问题上，少数研究包括编程领域，限制了其泛化能力。本文介绍了一种新的方法，能够在无需从类似o1的模型进行蒸馏或昂贵的人工注释的情况下，使LLMs具备LongCoT能力，我们通过标准指令模型自举长链推理（BOLT）实现这一目标。BOLT包括三个阶段：1）使用标准指令模型的上下文内学习自举LongCoT数据；2）监督微调LongCoT；3）在线训练进一步细化LongCoT能力。在BOLT中，只需要在自举阶段构建少量的上下文内示例；在我们的实验中，我们构建了10个示例，证明了该方法的可行性。我们使用Llama-3.1-70B-Instruct自举LongCoT，并将该方法应用于不同规模的模型（7B、8B、70B）。我们在Arena-Hard、MT-Bench、WildBench、ZebraLogic、MATH500等多种基准测试中实现了出色的表现，这些基准测试评估了多元任务解决能力和推理能力。 

---
# PsyPlay: Personality-Infused Role-Playing Conversational Agents 

**Title (ZH)**: PsyPlay：融入人格特征的角色扮演对话代理 

**Authors**: Tao Yang, Yuhua Zhu, Xiaojun Quan, Cong Liu, Qifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.03821)  

**Abstract**: The current research on Role-Playing Conversational Agents (RPCAs) with Large Language Models (LLMs) primarily focuses on imitating specific speaking styles and utilizing character backgrounds, neglecting the depiction of deeper personality traits.~In this study, we introduce personality-infused role-playing for LLM agents, which encourages agents to accurately portray their designated personality traits during dialogues. We then propose PsyPlay, a dialogue generation framework that facilitates the expression of rich personalities among multiple LLM agents. Specifically, PsyPlay enables agents to assume roles with distinct personality traits and engage in discussions centered around specific topics, consistently exhibiting their designated personality traits throughout the interactions. Validation on generated dialogue data demonstrates that PsyPlay can accurately portray the intended personality traits, achieving an overall success rate of 80.31% on GPT-3.5. Notably, we observe that LLMs aligned with positive values are more successful in portraying positive personality roles compared to negative ones. Moreover, we construct a dialogue corpus for personality-infused role-playing, called PsyPlay-Bench. The corpus, which consists of 4745 instances of correctly portrayed dialogues using PsyPlay, aims to further facilitate research in personalized role-playing and dialogue personality detection. 

**Abstract (ZH)**: 当前对基于大型语言模型（LLMs）的角色扮演对话代理（RPCAs）的研究主要集中在模仿特定的说话风格和利用角色背景，而忽视了深度个性特征的描绘。本研究引入了融入个性的角色扮演方法，鼓励代理在对话中准确地展现其指定的个性特征。随后，我们提出了PsyPlay对话生成框架，该框架促进了多个LLM代理之间丰富个性的表现。具体而言，PsyPlay允许代理承担具有不同个性特征的角色，并围绕特定主题展开讨论，整个互动过程中一贯展现其指定的个性特征。生成的对话数据验证结果显示，PsyPlay能够准确地展现预期的个性特征，对GPT-3.5的总体成功率达到了80.31%。此外，我们观察到，与正面价值观对齐的LLM在表现正面个性角色方面比负面角色更成功。此外，我们构建了一个用于个性融入的角色扮演的数据集，称为PsyPlay-Bench。该数据集包含4745个使用PsyPlay正确表现的对话实例，旨在进一步促进个性化角色扮演和对话个性识别的研究。 

---
# Identify Critical KV Cache in LLM Inference from an Output Perturbation Perspective 

**Title (ZH)**: 从输出扰动视角识别LLM推理中的关键KV缓存 

**Authors**: Yuan Feng, Junlin Lv, Yukun Cao, Xike Xie, S Kevin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.03805)  

**Abstract**: Large language models have revolutionized natural language processing but face significant challenges of high storage and runtime costs, due to the transformer architecture's reliance on self-attention, particularly the large Key-Value (KV) cache for long-sequence inference. Recent efforts to reduce KV cache size by pruning less critical entries based on attention weights remain empirical and lack formal grounding. This paper presents a formal study on identifying critical KV cache entries by analyzing attention output perturbation. Our analysis reveals that, beyond attention weights, the value states within KV entries and pretrained parameter matrices are also crucial. Based on this, we propose a perturbation-constrained selection algorithm that optimizes the worst-case output perturbation to identify critical entries. Evaluations on the Needle-in-a-Haystack test and Longbench benchmark show our algorithm enhances state-of-the-art cache eviction methods. Further empirical analysis confirms that our algorithm achieves lower output perturbations in over 92% attention heads in Llama model, thereby providing a significant improvement over existing methods. 

**Abstract (ZH)**: 大规模语言模型已经彻底改变了自然语言处理，但它们由于自注意力机制，尤其是长序列推理时依赖的大型键值（KV）缓存而面临着显著的存储和运行成本挑战。通过剪枝基于注意力权重的非关键条目来缩减KV缓存大小的近期努力仍处于经验阶段，缺乏正式的理论依据。本文对通过分析注意力输出扰动来识别关键KV缓存条目的关键性进行了正式研究。我们的分析表明，除了注意力权重外，KV条目内的值状态以及预训练参数矩阵也至关重要。基于此，我们提出了一种受限扰动选择算法，以优化最坏情况下的输出扰动来识别关键条目。在Needle-in-a-Haystack测试和LongBench基准上的评估表明，我们的算法增强了最先进的缓存淘汰方法。进一步的经验分析证实，我们的算法在Llama模型的超过92%的注意力头中实现了更低的输出扰动，从而显著优于现有方法。 

---
# Enhancing Hallucination Detection through Noise Injection 

**Title (ZH)**: 通过噪声注入提高幻觉检测效果 

**Authors**: Litian Liu, Reza Pourreza, Sunny Panchal, Apratim Bhattacharyya, Yao Qin, Roland Memisevic  

**Link**: [PDF](https://arxiv.org/pdf/2502.03799)  

**Abstract**: Large Language Models (LLMs) are prone to generating plausible yet incorrect responses, known as hallucinations. Effectively detecting hallucinations is therefore crucial for the safe deployment of LLMs. Recent research has linked hallucinations to model uncertainty, suggesting that hallucinations can be detected by measuring dispersion over answer distributions obtained from a set of samples drawn from a model. While drawing from the distribution over tokens defined by the model is a natural way to obtain samples, in this work, we argue that it is sub-optimal for the purpose of detecting hallucinations. We show that detection can be improved significantly by taking into account model uncertainty in the Bayesian sense. To this end, we propose a very simple and efficient approach that perturbs an appropriate subset of model parameters, or equivalently hidden unit activations, during sampling. We demonstrate its effectiveness across a wide range of datasets and model architectures. 

**Abstract (ZH)**: 大型语言模型（LLMs）可能会生成虽然听起来合理但实际上是错误的响应，这种现象被称为幻觉。因此，有效地检测幻觉对于大型语言模型的安全部署至关重要。近期的研究将幻觉与模型不确定性联系起来，表明可以通过测量模型生成的一组样本的答案分布来检测幻觉。虽然从模型定义的令牌分布中抽取样本是一种自然的方法，但在这项工作中，我们提出这种做法在检测幻觉方面并不理想。我们证明，通过在贝叶斯意义上考虑模型不确定性，可以显著提高检测效果。为此，我们提出了一种简单且高效的抽样方法，即在抽样过程中对模型参数的适当子集，或等价地隐藏单元激活进行扰动。我们展示了该方法在多种数据集和模型架构上的有效性。 

---
# Controlled LLM Decoding via Discrete Auto-regressive Biasing 

**Title (ZH)**: 通过离散自回归偏置进行的受控大语言模型解码 

**Authors**: Patrick Pynadath, Ruqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.03685)  

**Abstract**: Controlled text generation allows for enforcing user-defined constraints on large language model outputs, an increasingly important field as LLMs become more prevalent in everyday life. One common approach uses energy-based decoding, which defines a target distribution through an energy function that combines multiple constraints into a weighted average. However, these methods often struggle to balance fluency with constraint satisfaction, even with extensive tuning of the energy function's coefficients. In this paper, we identify that this suboptimal balance arises from sampling in continuous space rather than the natural discrete space of text tokens. To address this, we propose Discrete Auto-regressive Biasing, a controlled decoding algorithm that leverages gradients while operating entirely in the discrete text domain. Specifically, we introduce a new formulation for controlled text generation by defining a joint distribution over the generated sequence and an auxiliary bias sequence. To efficiently sample from this joint distribution, we propose a Langevin-within-Gibbs sampling algorithm using gradient-based discrete MCMC. Our method significantly improves constraint satisfaction while maintaining comparable or better fluency, all with even lower computational costs. We demonstrate the advantages of our controlled decoding method on sentiment control, language detoxification, and keyword-guided generation. 

**Abstract (ZH)**: 受控文本生成允许用户在大型语言模型输出上施加自定义约束，这是一个随着大型语言模型在日常生活中的普及而变得越来越重要的领域。一种常见的方法是使用基于能量的解码，该方法通过一个能量函数来定义目标分布，该能量函数将多种约束加权平均结合在一起。然而，这些方法往往难以平衡流畅性和约束满足，即便对能量函数的系数进行了广泛的调整。在本论文中，我们发现这种次优平衡源于在连续空间中进行采样，而不是在文本令牌的自然离散空间中进行采样。为了解决这一问题，我们提出了一种受控自回归偏差算法——离散自回归偏差，该算法利用梯度同时完全在离散文本域中运行。具体而言，我们通过定义生成序列和辅助偏差序列的联合分布，引入了一种新的受控文本生成公式。为了高效地从该联合分布中采样，我们提出了一种基于梯度的离散MCMC内的拉angevin采样算法。我们的方法在保持或提高流畅性的同时，显著提高了约束满足度，并且计算成本更低。我们通过情感控制、语言去污和关键词导向生成展示了我们受控解码方法的优势。 

---
# Context-Preserving Gradient Modulation for Large Language Models: A Novel Approach to Semantic Consistency in Long-Form Text Generation 

**Title (ZH)**: 保留上下文的梯度调制：长文本生成中语义一致性的新方法 

**Authors**: Nirola Kobanov, Edmund Weatherstone, Zachary Vanderpoel, Orlando Wetherby  

**Link**: [PDF](https://arxiv.org/pdf/2502.03643)  

**Abstract**: Maintaining semantic consistency over extended text sequences remains a fundamental challenge in long-form text generation, where conventional training methodologies often struggle to prevent contextual drift and coherence degradation. A novel gradient modulation approach is introduced, designed to adjust parameter updates dynamically in response to contextual relevance, ensuring that generated text remains aligned with prior discourse. By integrating a modulation function that selectively amplifies or attenuates gradients based on learned contextual dependencies, the proposed method enhances the stability of model-generated narratives without imposing significant computational overhead. Comparative evaluations against baseline models reveal improvements in coherence, contextual retention, and long-range dependency tracking, demonstrating the effectiveness of modifying the learning process at the gradient level. The results indicate that sentence structure variability and lexical diversity benefit from this approach, mitigating repetitive phrasing and improving adaptability across diverse linguistic contexts. Statistical validation of coherence metrics further substantiates the observed enhancements, with a significant reduction in inconsistencies emerging as a direct consequence of the modulation mechanism. Computational efficiency assessments confirm that the framework achieves these gains without requiring substantial modifications to the underlying architecture, ensuring compatibility with existing optimization workflows. 

**Abstract (ZH)**: 在长篇文本生成中保持语义一致性仍然是一个基本挑战，传统训练方法往往难以防止语境漂移和连贯性的下降。提出了一种新颖的梯度调节方法，旨在动态调整参数更新以响应语境的相关性，确保生成的文本与先前的讨论保持一致。通过结合一个调节函数，在根据学习到的语境依赖性有选择地放大或抑制梯度的基础上，所提出的方法在不增加显著计算开销的情况下增强了模型生成叙述的稳定性。与基线模型的比较评估显示了连贯性、语境保留和长距离依赖跟踪方面的改进，这证明了在梯度层面修改学习过程的有效性。结果表明，这种方法有助于提高句子结构的多样性和词汇多样性，减少重复表述，并提高在不同语言环境中适应性。统计验证还进一步证明了连贯性指标的增强效果，信度机制直接导致了不一致性显著减少。性能评估表明，该框架在不显著修改基础架构的情况下实现了这些收益，确保了与现有优化工作流的兼容性。 

---
# Adaptive Semantic Prompt Caching with VectorQ 

**Title (ZH)**: 自适应语义提示缓存技术：VectorQ 

**Authors**: Luis Gaspar Schroeder, Shu Liu, Alejandro Cuadron, Mark Zhao, Stephan Krusche, Alfons Kemper, Matei Zaharia, Joseph E. Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2502.03771)  

**Abstract**: Semantic prompt caches reduce the latency and cost of large language model (LLM) inference by reusing cached LLM-generated responses for semantically similar prompts. Vector similarity metrics assign a numerical score to quantify the similarity between an embedded prompt and its nearest neighbor in the cache. Existing systems rely on a static threshold to classify whether the similarity score is sufficiently high to result in a cache hit. We show that this one-size-fits-all threshold is insufficient across different prompts. We propose VectorQ, a framework to learn embedding-specific threshold regions that adapt to the complexity and uncertainty of an embedding. Through evaluations on a combination of four diverse datasets, we show that VectorQ consistently outperforms state-of-the-art systems across all static thresholds, achieving up to 12x increases in cache hit rate and error rate reductions up to 92%. 

**Abstract (ZH)**: 语义提示缓存通过重用缓存中的大语言模型（LLM）生成的响应来提高同义提示推理的延迟和成本。向量相似度度量通过给定一个数值评分来量化嵌入提示与其缓存中最近邻的相似度。现有系统依赖一个静态阈值来判断相似度评分是否足够高以产生缓存命中。我们表明，这个一刀切的阈值在不同提示下是不够合适的。我们提出了一种VectorQ框架，该框架能够学习特定于嵌入的阈值区间，以适应嵌入的复杂性和不确定性。通过在四个不同数据集上的综合评估，我们展示了VectorQ在所有静态阈值下都优于最先进的系统，缓存命中率提高了高达12倍，错误率降低了高达92%。 

---
# From Informal to Formal -- Incorporating and Evaluating LLMs on Natural Language Requirements to Verifiable Formal Proofs 

**Title (ZH)**: 从非形式化到形式化：将大语言模型集成到可验证的形式证明中及其评估 

**Authors**: Jialun Cao, Yaojie Lu, Meiziniu Li, Haoyang Ma, Haokun Li, Mengda He, Cheng Wen, Le Sun, Hongyu Zhang, Shengchao Qin, Shing-Chi Cheung, Cong Tian  

**Link**: [PDF](https://arxiv.org/pdf/2501.16207)  

**Abstract**: The research in AI-based formal mathematical reasoning has shown an unstoppable growth trend. These studies have excelled in mathematical competitions like IMO, showing significant progress. However, these studies intertwined multiple skills simultaneously, i.e., problem-solving, reasoning, and writing formal specifications, making it hard to precisely identify the LLMs' strengths and weaknesses in each task. This paper focuses on formal verification, an immediate application scenario of formal reasoning, and decomposes it into six sub-tasks. We constructed 18k high-quality instruction-response pairs across five mainstream formal specification languages (Coq, Lean4, Dafny, ACSL, and TLA+) in six formal-verification-related tasks by distilling GPT-4o. They are split into a 14k+ fine-tuning dataset FM-alpaca and a 4k benchmark FM-Bench. We found that LLMs are good at writing proof segments when given either the code, or the detailed description of proof steps. Also, the fine-tuning brought about a nearly threefold improvement at most. Interestingly, we observed that fine-tuning with formal data also enhances mathematics, reasoning, and coding abilities. We hope our findings inspire further research. Fine-tuned models are released to facilitate subsequent studies 

**Abstract (ZH)**: 基于AI的正式数学推理研究显示出了不可阻挡的增长趋势。这些研究在像国际数学奥林匹克（IMO）这样的数学竞赛中表现出色，显示出显著的进步。然而，这些研究同时融合了多种技能，如问题解决、推理和编写正式规范，使得难以精确辨别LLM在每项任务中的优势和不足。本文聚焦于正式验证，这是正式推理的一个直接应用场景，并将其分解为六个子任务。我们通过提炼GPT-4o，构建了涵盖五种主流正式规范语言（Coq、Lean4、Dafny、ACSL和TLA+）的18,000个高质量指令-响应对，分布在六个正式验证相关任务中。这些数据集被划分为一个包含14,000多个用于微调的数据集FM-alpaca和一个包含4,000个基准测试数据集FM-Bench。我们发现，当给定代码或详细的证明步骤描述时，LLM在编写证明片段方面表现良好。此外，微调带来了大约三倍的改进。有趣的是，我们观察到，使用正式数据进行微调还可以增强数学、推理和编程能力。我们希望我们的研究发现能激发进一步的研究，并发布微调后的模型以促进后续研究。 

---
