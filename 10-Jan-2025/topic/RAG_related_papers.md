# Search-o1: Agentic Search-Enhanced Large Reasoning Models 

**Title (ZH)**: Search-o1: 代理增强的大推理模型搜索 

**Authors**: Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2501.05366)  

**Abstract**: Large reasoning models (LRMs) like OpenAI-o1 have demonstrated impressive long stepwise reasoning capabilities through large-scale reinforcement learning. However, their extended reasoning processes often suffer from knowledge insufficiency, leading to frequent uncertainties and potential errors. To address this limitation, we introduce \textbf{Search-o1}, a framework that enhances LRMs with an agentic retrieval-augmented generation (RAG) mechanism and a Reason-in-Documents module for refining retrieved documents. Search-o1 integrates an agentic search workflow into the reasoning process, enabling dynamic retrieval of external knowledge when LRMs encounter uncertain knowledge points. Additionally, due to the verbose nature of retrieved documents, we design a separate Reason-in-Documents module to deeply analyze the retrieved information before injecting it into the reasoning chain, minimizing noise and preserving coherent reasoning flow. Extensive experiments on complex reasoning tasks in science, mathematics, and coding, as well as six open-domain QA benchmarks, demonstrate the strong performance of Search-o1. This approach enhances the trustworthiness and applicability of LRMs in complex reasoning tasks, paving the way for more reliable and versatile intelligent systems. The code is available at \url{this https URL}. 

**Abstract (ZH)**: 大型推理模型（LRMs）如OpenAI-o1通过大规模强化学习展示了令人印象深刻的长步骤推理能力。然而，它们的延长推理过程往往受到知识不足的影响，导致频繁的不确定性甚至潜在错误。为了解决这一限制，我们引入了**Search-o1**，一个通过添加自主检索增强生成（RAG）机制和文档内推理模块来增强LRMs的框架。Search-o1将自主搜索工作流程整合到推理过程中，使LRMs在遇到不确定的知识点时能够动态检索外部知识。此外，由于检索到的文档通常内容丰富，我们设计了一个单独的文档内推理模块，在将这些信息注入推理链之前对其进行深入分析，从而减少噪声并保持连贯的推理流程。通过在科学、数学和编程等复杂推理任务以及六个开放领域问答基准测试中的广泛实验，证明了Search-o1的出色性能。该方法增强了LRMs在复杂推理任务中的可信度和适用性，为更可靠和多功能的智能系统铺平了道路。代码可在[此处](this https URL)获得。 

---
# A General Retrieval-Augmented Generation Framework for Multimodal Case-Based Reasoning Applications 

**Title (ZH)**: 一种用于多模态案例推理应用的通用检索增强生成框架 

**Authors**: Ofir Marom  

**Link**: [PDF](https://arxiv.org/pdf/2501.05030)  

**Abstract**: Case-based reasoning (CBR) is an experience-based approach to problem solving, where a repository of solved cases is adapted to solve new cases. Recent research shows that Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) can support the Retrieve and Reuse stages of the CBR pipeline by retrieving similar cases and using them as additional context to an LLM query. Most studies have focused on text-only applications, however, in many real-world problems the components of a case are multimodal. In this paper we present MCBR-RAG, a general RAG framework for multimodal CBR applications. The MCBR-RAG framework converts non-text case components into text-based representations, allowing it to: 1) learn application-specific latent representations that can be indexed for retrieval, and 2) enrich the query provided to the LLM by incorporating all case components for better context. We demonstrate MCBR-RAG's effectiveness through experiments conducted on a simplified Math-24 application and a more complex Backgammon application. Our empirical results show that MCBR-RAG improves generation quality compared to a baseline LLM with no contextual information provided. 

**Abstract (ZH)**: 案例基于推理（CBR）是一种基于经验的解决问题方法，在这种方法中，通过调整已解决的案例库来解决新问题。最近的研究表明，附有检索增强生成（RAG）的大语言模型（LLMs）可以支持CBR工作流中的检索和重用阶段，通过检索相似的案例，并将其作为附加上下文提供给LLM查询。大多数研究集中在纯文本应用上，然而，在许多实际问题中，案例的组件是多模态的。在本文中，我们提出了MCBR-RAG，这是一种适用于多模态CBR应用的一般RAG框架。MCBR-RAG框架将非文本案例组件转换为文本表示，使其能够：1）学习特定应用的潜在表示，这些表示可以进行索引以供检索，2）通过结合所有案例组件来丰富对LLM的查询，提供更好的上下文。我们通过在简化版的Math-24应用和更复杂的背投棋应用中进行的实验，证明了MCBR-RAG的有效性。我们的实证结果表明，MCBR-RAG在提供上下文信息的情况下相比没有提供上下文信息的基线LLM，提高了生成质量。 

---
# RAG-WM: An Efficient Black-Box Watermarking Approach for Retrieval-Augmented Generation of Large Language Models 

**Title (ZH)**: RAG-WM：一种高效的黑盒水印方法，用于大型语言模型的检索增强生成 

**Authors**: Peizhuo Lv, Mengjie Sun, Hao Wang, Xiaofeng Wang, Shengzhi Zhang, Yuxuan Chen, Kai Chen, Limin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2501.05249)  

**Abstract**: In recent years, tremendous success has been witnessed in Retrieval-Augmented Generation (RAG), widely used to enhance Large Language Models (LLMs) in domain-specific, knowledge-intensive, and privacy-sensitive tasks. However, attackers may steal those valuable RAGs and deploy or commercialize them, making it essential to detect Intellectual Property (IP) infringement. Most existing ownership protection solutions, such as watermarks, are designed for relational databases and texts. They cannot be directly applied to RAGs because relational database watermarks require white-box access to detect IP infringement, which is unrealistic for the knowledge base in RAGs. Meanwhile, post-processing by the adversary's deployed LLMs typically destructs text watermark information. To address those problems, we propose a novel black-box "knowledge watermark" approach, named RAG-WM, to detect IP infringement of RAGs. RAG-WM uses a multi-LLM interaction framework, comprising a Watermark Generator, Shadow LLM & RAG, and Watermark Discriminator, to create watermark texts based on watermark entity-relationship tuples and inject them into the target RAG. We evaluate RAG-WM across three domain-specific and two privacy-sensitive tasks on four benchmark LLMs. Experimental results show that RAG-WM effectively detects the stolen RAGs in various deployed LLMs. Furthermore, RAG-WM is robust against paraphrasing, unrelated content removal, knowledge insertion, and knowledge expansion attacks. Lastly, RAG-WM can also evade watermark detection approaches, highlighting its promising application in detecting IP infringement of RAG systems. 

**Abstract (ZH)**: 近年来，检索增强生成（RAG）技术在特定领域、知识密集和隐私敏感任务中大幅提升了大型语言模型（LLMs）的表现，取得了巨大成功。然而，攻击者可能盗取这些有价值的RAG并进行部署或商业化，因此对知识产权（IP）侵权的检测变得至关重要。目前大多数现有的所有权保护解决方案，如水印技术，主要适用于关系型数据库和文本，无法直接应用于RAG。因为关系型数据库水印需要白盒访问以检测IP侵权，而在RAG的知识库中实现这一点并不现实。同时，攻击者部署的LLM的后处理通常会破坏文本水印信息。为解决这些问题，我们提出了一种新的黑盒“知识水印”方法，称为RAG-WM，用于检测RAG的IP侵权。RAG-WM采用多LLM交互框架，包括水印生成器、影子LLM和RAG以及水印鉴别器，基于水印实体-关系元组生成水印文本，并将其注入目标RAG。我们分别在四个基准LLM上进行了三种特定领域和两种隐私敏感任务的评估。实验结果表明，RAG-WM能够有效检测各种部署中被盗的RAG。此外，RAG-WM对重述、无关内容删除、知识插入和知识扩展攻击具有鲁棒性。最后，RAG-WM还可以避免其他水印检测方法的检测，凸显了它在检测RAG系统IP侵权方面的潜在应用价值。 

---
# Biomedical Relation Extraction via Adaptive Document-Relation Cross-Mapping and Concept Unique Identifier 

**Title (ZH)**: 通过自适应文档-关系交叉映射和概念唯一标识符进行生物医学关系提取 

**Authors**: Yufei Shang, Yanrong Guo, Shijie Hao, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2501.05155)  

**Abstract**: Document-Level Biomedical Relation Extraction (Bio-RE) aims to identify relations between biomedical entities within extensive texts, serving as a crucial subfield of biomedical text mining. Existing Bio-RE methods struggle with cross-sentence inference, which is essential for capturing relations spanning multiple sentences. Moreover, previous methods often overlook the incompleteness of documents and lack the integration of external knowledge, limiting contextual richness. Besides, the scarcity of annotated data further hampers model training. Recent advancements in large language models (LLMs) have inspired us to explore all the above issues for document-level Bio-RE. Specifically, we propose a document-level Bio-RE framework via LLM Adaptive Document-Relation Cross-Mapping (ADRCM) Fine-Tuning and Concept Unique Identifier (CUI) Retrieval-Augmented Generation (RAG). First, we introduce the Iteration-of-REsummary (IoRs) prompt for solving the data scarcity issue. In this way, Bio-RE task-specific synthetic data can be generated by guiding ChatGPT to focus on entity relations and iteratively refining synthetic data. Next, we propose ADRCM fine-tuning, a novel fine-tuning recipe that establishes mappings across different documents and relations, enhancing the model's contextual understanding and cross-sentence inference capabilities. Finally, during the inference, a biomedical-specific RAG approach, named CUI RAG, is designed to leverage CUIs as indexes for entities, narrowing the retrieval scope and enriching the relevant document contexts. Experiments conducted on three Bio-RE datasets (GDA, CDR, and BioRED) demonstrate the state-of-the-art performance of our proposed method by comparing it with other related works. 

**Abstract (ZH)**: 生物医学文档级关系提取（Bio-RE）旨在识别文本中生物医学实体之间的关系，是生物医学文本挖掘中的关键子领域。现有Bio-RE方法在句间推理方面存在困难，句间关系的捕获至关重要。此外，先前的方法往往忽略了文档的不完整性，缺乏外部知识的整合，限制了上下文的丰富性。此外，标注数据稀缺进一步阻碍了模型的训练。近期在大规模语言模型（LLMs）方面的进展启发我们探索这些问题以改进文档级Bio-RE方法。具体而言，我们提出了一种通过LLMs自适应文档-关系跨映射（ADRCM）微调和概念唯一标识符（CUI）检索增强生成（RAG）的文档级Bio-RE框架。首先，我们介绍了Iterative Relation Extraction Summary（IoRs）提示，以解决数据稀缺问题。通过这种方式，可以引导ChatGPT关注实体关系并迭代优化合成数据，生成任务特定的合成数据。接着，我们提出了ADRCM微调，这是一种新颖的微调方法，能够建立不同文档和关系之间的映射，增强模型的上下文理解和句间推理能力。最后，在推理过程中，我们设计了一种特定于生物医学的RAG方法——CUI RAG，利用CUI作为实体的索引，缩小检索范围并丰富相关文档上下文。在GDA、CDR和BioRED三个Bio-RE数据集上的实验表明，我们的方法在与其他相关工作的比较中表现出最先进的性能。 

---
# SUGAR: Leveraging Contextual Confidence for Smarter Retrieval 

**Title (ZH)**: SUGAR：利用上下文置信度进行更智能的检索 

**Authors**: Hanna Zubkova, Ji-Hoon Park, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2501.04899)  

**Abstract**: Bearing in mind the limited parametric knowledge of Large Language Models (LLMs), retrieval-augmented generation (RAG) which supplies them with the relevant external knowledge has served as an approach to mitigate the issue of hallucinations to a certain extent. However, uniformly retrieving supporting context makes response generation source-inefficient, as triggering the retriever is not always necessary, or even inaccurate, when a model gets distracted by noisy retrieved content and produces an unhelpful answer. Motivated by these issues, we introduce Semantic Uncertainty Guided Adaptive Retrieval (SUGAR), where we leverage context-based entropy to actively decide whether to retrieve and to further determine between single-step and multi-step retrieval. Our empirical results show that selective retrieval guided by semantic uncertainty estimation improves the performance across diverse question answering tasks, as well as achieves a more efficient inference. 

**Abstract (ZH)**: 考虑到大型语言模型（LLMs）的参数知识有限，通过检索增强生成（RAG）方法，为它们提供相关外部知识，已经在一定程度上缓解了幻觉问题。然而，均匀检索支持上下文使得响应生成缺乏效率，因为当模型受到噪声检索内容的干扰并产生无用的答案时，触发检索器并不总是必要的，甚至可能是不准确的。为了解决这些问题，我们提出了语义不确定性引导自适应检索（SUGAR），在此方法中，我们利用上下文熵来主动决定是否进行检索，并进一步确定是进行单步检索还是多步检索。我们的实验证明，基于语义不确定性估计的选择性检索提高了各种问答任务的性能，并实现了更加高效的推理。 

---
# Advancing Retrieval-Augmented Generation for Persian: Development of Language Models, Comprehensive Benchmarks, and Best Practices for Optimization 

**Title (ZH)**: 提高波斯语检索增强生成的能力：语言模型的发展、全面基准测试及优化最佳实践 

**Authors**: Sara Bourbour Hosseinbeigi, Sina Asghari, Mohammad Ali Seif Kashani, Mohammad Hossein Shalchian, Mohammad Amin Abbasi  

**Link**: [PDF](https://arxiv.org/pdf/2501.04858)  

**Abstract**: This paper examines the specific obstacles of constructing Retrieval-Augmented Generation(RAG) systems in low-resource languages, with a focus on Persian's complicated morphology and versatile syntax. The research aims to improve retrieval and generation accuracy by introducing Persian-specific models, namely MatinaRoberta(a masked language model) and MatinaSRoberta(a fine-tuned Sentence-BERT), along with a comprehensive benchmarking framework. Three datasets-general knowledge(PQuad), scientifically specialized texts, and organizational reports, were used to assess these models after they were trained on a varied corpus of 73.11 billion Persian tokens. The methodology involved extensive pretraining, fine-tuning with tailored loss functions, and systematic evaluations using both traditional metrics and the Retrieval-Augmented Generation Assessment framework. The results show that MatinaSRoberta outperformed previous embeddings, achieving superior contextual relevance and retrieval accuracy across datasets. Temperature tweaking, chunk size modifications, and document summary indexing were explored to enhance RAG setups. Larger models like Llama-3.1 (70B) consistently demonstrated the highest generation accuracy, while smaller models faced challenges with domain-specific and formal contexts. The findings underscore the potential for developing RAG systems in Persian through customized embeddings and retrieval-generation settings and highlight the enhancement of NLP applications such as search engines and legal document analysis in low-resource languages. 

**Abstract (ZH)**: 本文探讨了在低资源语言中构建检索增强生成（RAG）系统的特定障碍，着重于波斯语复杂形态和多变的句法结构。研究旨在通过引入波斯语特定模型——MatinaRoberta（一种掩码语言模型）和MatinaSRoberta（微调后的Sentence-BERT）以及一个全面的基准测试框架来提高检索和生成的准确性。经过731.1亿个波斯语令牌组成的多样化语料库训练后，使用了三个数据集：通用知识数据集（PQuad）、科学专著和组织报告，以评估这些模型。研究方法包括广泛的预训练、针对特定损失函数的微调以及使用传统度量和RAG评估框架进行系统的评估。结果显示，MatinaSRoberta超过了先前的嵌入，实现了跨数据集的更高语境相关性和检索准确率。通过调整温度、分块大小和文档摘要索引来增强RAG设置进行了探索。较大的模型如Llama-3.1（70B）在生成准确性方面表现最佳，而较小的模型在领域特定和正式语言环境中面临挑战。研究结果强调了通过定制嵌入和检索生成设置开发波斯语RAG系统的潜力，并突显了在低资源语言中增强自然语言处理（NLP）应用，如搜索引擎和法律文件分析的可能性。 

---
