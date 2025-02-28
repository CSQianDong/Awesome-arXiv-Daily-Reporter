# Granite Embedding Models 

**Title (ZH)**: 花岗岩嵌入模型 

**Authors**: Parul Awasthy, Aashka Trivedi, Yulong Li, Mihaela Bornea, David Cox, Abraham Daniels, Martin Franz, Gabe Goodhart, Bhavani Iyer, Vishwajeet Kumar, Luis Lastras, Scott McCarley, Rudra Murthy, Vignesh P, Sara Rosenthal, Salim Roukos, Jaydeep Sen, Sukriti Sharma, Avirup Sil, Kate Soule, Arafat Sultan, Radu Florian  

**Link**: [PDF](https://arxiv.org/pdf/2502.20204)  

**Abstract**: We introduce the Granite Embedding models, a family of encoder-based embedding models designed for retrieval tasks, spanning dense-retrieval and sparse retrieval architectures, with both English and Multilingual capabilities. This report provides the technical details of training these highly effective 12 layer embedding models, along with their efficient 6 layer distilled counterparts. Extensive evaluations show that the models, developed with techniques like retrieval oriented pretraining, contrastive finetuning, knowledge distillation, and model merging significantly outperform publicly available models of similar sizes on both internal IBM retrieval and search tasks, and have equivalent performance on widely used information retrieval benchmarks, while being trained on high-quality data suitable for enterprise use. We publicly release all our Granite Embedding models under the Apache 2.0 license, allowing both research and commercial use at this https URL. 

**Abstract (ZH)**: 我们介绍了Granite Embedding模型，这是一种基于编码器的嵌入模型，设计用于检索任务，支持从密集检索到稀疏检索的各种架构，并具备英语和多语言的能力。本报告提供了训练这些高度有效的12层嵌入模型的技术细节，以及其高效的6层蒸馏版本。广泛的评估显示，该模型通过诸如检索定向预训练、对比微调、知识蒸馏和模型合并等技术，在IBM内部检索和搜索任务中明显优于公开可用的同类大小模型，并在广泛使用的信息检索基准测试中表现出相当的性能，而这些模型是在适合企业使用的高质量数据上进行训练的。我们将在Apache 2.0许可证下公开发布所有Granite Embedding模型，允许在此https网址处进行研究和商业使用。 

---
# Bisecting K-Means in RAG for Enhancing Question-Answering Tasks Performance in Telecommunications 

**Title (ZH)**: 将RAG中的Bisecting K-Means用于提升电信领域问答任务性能 

**Authors**: Pedro Sousa, Cláudio Klautau Mello, Frank B. Morte, Luis F. Solis Navarro  

**Link**: [PDF](https://arxiv.org/pdf/2502.20188)  

**Abstract**: Question-answering tasks in the telecom domain are still reasonably unexplored in the literature, primarily due to the field's rapid changes and evolving standards. This work presents a novel Retrieval-Augmented Generation framework explicitly designed for the telecommunication domain, focusing on datasets composed of 3GPP documents. The framework introduces the use of the Bisecting K-Means clustering technique to organize the embedding vectors by contents, facilitating more efficient information retrieval. By leveraging this clustering technique, the system pre-selects a subset of clusters that are most similar to the user's query, enhancing the relevance of the retrieved information. Aiming for models with lower computational cost for inference, the framework was tested using Small Language Models, demonstrating improved performance with an accuracy of 66.12% on phi-2 and 72.13% on phi-3 fine-tuned models, and reduced training time. 

**Abstract (ZH)**: 电信领域的问答任务在文献中仍存在较大的未开发空间，主要原因是该领域的快速变化和不断演变的标准。本文提出了一种新颖的检索增强生成框架，专门针对电信领域，重点关注由3GPP文档构成的数据集。该框架引入了二分K均值聚类技术来组织按内容嵌入的向量，从而提高了信息检索的效率。通过利用这种聚类技术，系统预先选择与用户查询最相似的集群，增强了检索信息的相关性。为了降低推理过程中的计算成本，该框架使用了小型语言模型进行测试，结果显示，对于经过微调的phi-2和phi-3模型，准确率分别达到了66.12%和72.13%，并且缩短了训练时间。 

---
# Teaching Dense Retrieval Models to Specialize with Listwise Distillation and LLM Data Augmentation 

**Title (ZH)**: 将以下论文内容或标题翻译成中文，符合学术规范：

"通过列表级蒸馏和大语言模型数据增强使密集检索模型专业化"

这个翻译保持了原文的意思，并符合学术论文标题的规范。 

**Authors**: Manveer Singh Tamber, Suleman Kazi, Vivek Sourabh, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2502.19712)  

**Abstract**: While the current state-of-the-art dense retrieval models exhibit strong out-of-domain generalization, they might fail to capture nuanced domain-specific knowledge. In principle, fine-tuning these models for specialized retrieval tasks should yield higher effectiveness than relying on a one-size-fits-all model, but in practice, results can disappoint. We show that standard fine-tuning methods using an InfoNCE loss can unexpectedly degrade effectiveness rather than improve it, even for domain-specific scenarios. This holds true even when applying widely adopted techniques such as hard-negative mining and negative de-noising. To address this, we explore a training strategy that uses listwise distillation from a teacher cross-encoder, leveraging rich relevance signals to fine-tune the retriever. We further explore synthetic query generation using large language models. Through listwise distillation and training with a diverse set of queries ranging from natural user searches and factual claims to keyword-based queries, we achieve consistent effectiveness gains across multiple datasets. Our results also reveal that synthetic queries can rival human-written queries in training utility. However, we also identify limitations, particularly in the effectiveness of cross-encoder teachers as a bottleneck. We release our code and scripts to encourage further research. 

**Abstract (ZH)**: 尽管当前最先进的密集检索模型在跨领域任务上表现出色，但它们可能无法捕捉到细微的领域特定知识。原则上，为特定检索任务微调这些模型应该比依赖通用模型更有效，但在实践中，结果可能不尽如人意。我们发现，使用InfoNCE损失的标准微调方法在领域特定情况下不仅没有提高效果，反而可能会降低效果。即使在广泛采用的技术，如难负例挖掘和负例去噪的情况下，也是如此。为了解决这一问题，我们探索了一种采用教师交叉编码器进行列表级蒸馏的训练策略，利用丰富的相关性信号微调检索器。我们还探索了使用大型语言模型生成合成查询。通过列表级蒸馏和使用从自然用户查询、事实性断言到基于关键词的查询等多种类型的查询进行训练，我们在多个数据集上实现了一致的效果提升。我们的研究表明，合成查询在训练效用方面可以与人工编写查询相媲美。然而，我们也指出了一定的局限性，特别是在交叉编码器教师的有效性方面，这是限制因素之一。我们发布了代码和脚本来促进进一步的研究。 

---
# PCL: Prompt-based Continual Learning for User Modeling in Recommender Systems 

**Title (ZH)**: PCL：基于提示的持续学习在推荐系统中用户建模 

**Authors**: Mingdai Yang, Fan Yang, Yanhui Guo, Shaoyuan Xu, Tianchen Zhou, Yetian Chen, Simone Shao, Jia Liu, Yan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2502.19628)  

**Abstract**: User modeling in large e-commerce platforms aims to optimize user experiences by incorporating various customer activities. Traditional models targeting a single task often focus on specific business metrics, neglecting the comprehensive user behavior, and thus limiting their effectiveness. To develop more generalized user representations, some existing work adopts Multi-task Learning (MTL)approaches. But they all face the challenges of optimization imbalance and inefficiency in adapting to new tasks. Continual Learning (CL), which allows models to learn new tasks incrementally and independently, has emerged as a solution to MTL's limitations. However, CL faces the challenge of catastrophic forgetting, where previously learned knowledge is lost when the model is learning the new task. Inspired by the success of prompt tuning in Pretrained Language Models (PLMs), we propose PCL, a Prompt-based Continual Learning framework for user modeling, which utilizes position-wise prompts as external memory for each task, preserving knowledge and mitigating catastrophic forgetting. Additionally, we design contextual prompts to capture and leverage inter-task relationships during prompt tuning. We conduct extensive experiments on real-world datasets to demonstrate PCL's effectiveness. 

**Abstract (ZH)**: 在线大规模电子商务平台中的用户建模旨在通过整合各种客户活动来优化用户体验。传统的单任务模型通常专注于特定的业务指标，而忽视了全面的用户行为，从而限制了其有效性。为了开发更通用的用户表示，现有的一些工作采用了多任务学习（MTL）的方法。但这些方法都面临着优化不平衡和新任务适应效率低下的挑战。连续学习（CL），该方法允许模型逐步独立地学习新任务，已成为解决MTL局限性的解决方案。然而，CL 面临着灾难性遗忘的挑战，即当模型学习新任务时，之前的已学知识会丢失。借鉴预训练语言模型（PLMs）中的提示调优的成功经验，我们提出了一种基于提示的连续学习框架PCL（Prompt-based Continual Learning），该框架利用位置感知提示作为每个任务的外部记忆，以保留知识并减轻灾难性遗忘。此外，我们设计了上下文提示，在提示调优过程中捕捉和利用任务间关系。我们在实际数据集上进行了广泛实验，以证明PCL的有效性。 

---
# Mixture of Structural-and-Textual Retrieval over Text-rich Graph Knowledge Bases 

**Title (ZH)**: 文本丰富图知识库中的结构性和文本性检索混合模型 

**Authors**: Yongjia Lei, Haoyu Han, Ryan A. Rossi, Franck Dernoncourt, Nedim Lipka, Mahantesh M Halappanavar, Jiliang Tang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20317)  

**Abstract**: Text-rich Graph Knowledge Bases (TG-KBs) have become increasingly crucial for answering queries by providing textual and structural knowledge. However, current retrieval methods often retrieve these two types of knowledge in isolation without considering their mutual reinforcement and some hybrid methods even bypass structural retrieval entirely after neighboring aggregation. To fill in this gap, we propose a Mixture of Structural-and-Textual Retrieval (MoR) to retrieve these two types of knowledge via a Planning-Reasoning-Organizing framework. In the Planning stage, MoR generates textual planning graphs delineating the logic for answering queries. Following planning graphs, in the Reasoning stage, MoR interweaves structural traversal and textual matching to obtain candidates from TG-KBs. In the Organizing stage, MoR further reranks fetched candidates based on their structural trajectory. Extensive experiments demonstrate the superiority of MoR in harmonizing structural and textual retrieval with insights, including uneven retrieving performance across different query logics and the benefits of integrating structural trajectories for candidate reranking. Our code is available at this https URL. 

**Abstract (ZH)**: 文本丰富的图知识库（TG-KBs）对于通过提供文本和结构化知识来回答查询变得越来越关键。然而，现有的检索方法通常将这两种类型的知识孤立地进行检索，而不考虑它们之间的相互强化。此外，一些混合方法甚至在邻接聚合后完全跳过了结构化检索。为了解决这一问题，我们提出了一种结构化和文本化检索的混合方法（MoR），并通过一种规划-推理-组织框架来检索这两种类型的知识。在规划阶段，MoR 生成文本规划图，明确回答查询的逻辑。随后，在推理阶段，MoR 将结构化遍历与文本匹配交织起来，从TG-KBs中获取候选对象。在组织阶段，MoR 进一步基于候选对象的结构轨迹来重新排序。广泛的实验表明，MoR 在协调结构化和文本化检索方面具有优势，包括不同查询逻辑下的检索性能不均衡以及通过整合结构轨迹对候选对象重新排序的好处。我们的代码可从以下链接获取：this https URL。 

---
# LangProBe: a Language Programs Benchmark 

**Title (ZH)**: LangProBe：语言程序基准测试 

**Authors**: Shangyin Tan, Lakshya A Agrawal, Arnav Singhvi, Liheng Lai, Michael J Ryan, Dan Klein, Omar Khattab, Koushik Sen, Matei Zaharia  

**Link**: [PDF](https://arxiv.org/pdf/2502.20315)  

**Abstract**: Composing language models (LMs) into multi-step language programs and automatically optimizing their modular prompts is now a mainstream paradigm for building AI systems, but the tradeoffs in this space have only scarcely been studied before. We introduce LangProBe, the first large-scale benchmark for evaluating the architectures and optimization strategies for language programs, with over 2000 combinations of tasks, architectures, optimizers, and choices of LMs. Using LangProBe, we are the first to study the impact of program architectures and optimizers (and their compositions together and with different models) on tradeoffs of quality and cost. We find that optimized language programs offer strong cost--quality Pareto improvement over raw calls to models, but simultaneously demonstrate that human judgment (or empirical decisions) about which compositions to pursue is still necessary for best performance. We will open source the code and evaluation data for LangProBe. 

**Abstract (ZH)**: 将多步语言程序（即语言模型LM）的组成以及自动优化其模块化提示作为构建AI系统的主要范式已经变得日益主流，但此领域的权衡研究却较为鲜少。我们介绍了LangProBe，这是首个用于评估语言程序架构和优化策略的大规模基准，包括超过2000种任务、架构、优化器和LM的选择组合。通过LangProBe，我们首次研究了程序架构和优化器（及其组合使用和与不同模型的结合）对质量和成本权衡的影响。我们发现优化的语言程序在质量和成本方面提供了显著的帕累托改进，但也表明，为了达到最佳性能，仍然需要人工判断（或经验决策）来选择哪些组合。我们将开源LangProBe的代码和评估数据。 

---
# ReCon: Enhancing True Correspondence Discrimination through Relation Consistency for Robust Noisy Correspondence Learning 

**Title (ZH)**: ReCon：通过关系一致性提高真实对应关系区分能力，以增强鲁棒的噪声对应学习 

**Authors**: Quanxing Zha, Xin Liu, Shu-Juan Peng, Yiu-ming Cheung, Xing Xu, Nannan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2502.19962)  

**Abstract**: Can we accurately identify the true correspondences from multimodal datasets containing mismatched data pairs? Existing methods primarily emphasize the similarity matching between the representations of objects across modalities, potentially neglecting the crucial relation consistency within modalities that are particularly important for distinguishing the true and false correspondences. Such an omission often runs the risk of misidentifying negatives as positives, thus leading to unanticipated performance degradation. To address this problem, we propose a general Relation Consistency learning framework, namely ReCon, to accurately discriminate the true correspondences among the multimodal data and thus effectively mitigate the adverse impact caused by mismatches. Specifically, ReCon leverages a novel relation consistency learning to ensure the dual-alignment, respectively of, the cross-modal relation consistency between different modalities and the intra-modal relation consistency within modalities. Thanks to such dual constrains on relations, ReCon significantly enhances its effectiveness for true correspondence discrimination and therefore reliably filters out the mismatched pairs to mitigate the risks of wrong supervisions. Extensive experiments on three widely-used benchmark datasets, including Flickr30K, MS-COCO, and Conceptual Captions, are conducted to demonstrate the effectiveness and superiority of ReCon compared with other SOTAs. The code is available at: this https URL. 

**Abstract (ZH)**: 我们能否从包含不匹配数据对的多模态数据集中准确地识别出真实的对应关系？现有方法主要关注不同模态中对象表示的相似性匹配，可能忽略了模态内部的关联一致性，这是区分真实和错误对应关系特别重要的因素。这种忽略往往会增加将负样本误识别为正样本的风险，从而导致性能下降。为解决这一问题，我们提出了一种通用的关系一致性学习框架——ReCon，以准确地在多模态数据中区分真实的对应关系，并有效地减轻由不匹配引起的不利影响。具体来说，ReCon 利用一种新颖的关系一致性学习方法，确保不同模态之间跨模态关系一致性和同一模态内部关系一致性的双重对齐。得益于这种双重关系约束，ReCon 在真实对应关系的区分上表现出明显增强的效果，从而可靠地筛选出不匹配的数据对，降低错误监督的风险。在 Flickr30K、MS-COCO 和 Conceptual Captions 这三个广泛使用的基准数据集上进行了广泛的实验，以验证 ReCon 的有效性和优越性。相关代码已发布于：this https URL。 

---
# Few-Shot Multilingual Open-Domain QA from 5 Examples 

**Title (ZH)**: 从5个示例实现多语言开放域问答（Few-Shot Multilingual Open-Domain QA from 5 Examples） 

**Authors**: Fan Jiang, Tom Drummond, Trevor Cohn  

**Link**: [PDF](https://arxiv.org/pdf/2502.19722)  

**Abstract**: Recent approaches to multilingual open-domain question answering (MLODQA) have achieved promising results given abundant language-specific training data. However, the considerable annotation cost limits the application of these methods for underrepresented languages. We introduce a \emph{few-shot learning} approach to synthesise large-scale multilingual data from large language models (LLMs). Our method begins with large-scale self-supervised pre-training using WikiData, followed by training on high-quality synthetic multilingual data generated by prompting LLMs with few-shot supervision. The final model, \textsc{FsModQA}, significantly outperforms existing few-shot and supervised baselines in MLODQA and cross-lingual and monolingual retrieval. We further show our method can be extended for effective zero-shot adaptation to new languages through a \emph{cross-lingual prompting} strategy with only English-supervised data, making it a general and applicable solution for MLODQA tasks without costly large-scale annotation. 

**Abstract (ZH)**: 近年来，针对多语言开放域问答（MLODQA）的研究在大量语言特定训练数据的支持下取得了显著的成果。然而，高昂的标注成本限制了这些方法在未充分代表的语言中的应用。我们提出了一种小样本学习（Few-Shot Learning）方法，通过大型语言模型（LLMs）合成大规模多语言数据。该方法首先使用WikiData进行大规模自我监督预训练，随后利用提示LLMs生成少量监督下的高质量合成多语言数据进行训练。最终模型FsModQA在MLODQA和跨语言及单语言检索任务中显著优于现有的小样本和监督基线。我们进一步展示了通过仅使用英语监督数据的跨语言提示策略，该方法可以有效实现对新语言的零样本适应，使其成为一种低成本、广泛适用的MLODQA任务解决方案。 

---
# Trustworthy Answers, Messier Data: Bridging the Gap in Low-Resource Retrieval-Augmented Generation for Domain Expert Systems 

**Title (ZH)**: 值得信赖的答案，复杂的数据：缩小领域专家系统中低资源检索增强生成的差距 

**Authors**: Nayoung Choi, Grace Byun, Andrew Chung, Ellie S. Paek, Shinsun Lee, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2502.19596)  

**Abstract**: RAG has become a key technique for enhancing LLMs by reducing hallucinations, especially in domain expert systems where LLMs may lack sufficient inherent knowledge. However, developing these systems in low-resource settings introduces several challenges: (1) handling heterogeneous data sources, (2) optimizing retrieval phase for trustworthy answers, and (3) evaluating generated answers across diverse aspects. To address these, we introduce a data generation pipeline that transforms raw multi-modal data into structured corpus and Q&A pairs, an advanced re-ranking phase improving retrieval precision, and a reference matching algorithm enhancing answer traceability. Applied to the automotive engineering domain, our system improves factual correctness (+1.94), informativeness (+1.16), and helpfulness (+1.67) over a non-RAG baseline, based on a 1-5 scale by an LLM judge. These results highlight the effectiveness of our approach across distinct aspects, with strong answer grounding and transparency. 

**Abstract (ZH)**: RAG已成为通过减少幻觉来增强大型语言模型（LLM）的关键技术，特别是在LLM可能缺乏充分内在知识的领域专家系统中。然而，在资源匮乏的环境中开发这些系统带来了许多挑战：（1）处理异构数据源，（2）优化检索阶段以获得可信的答案，以及（3）从多个方面对生成的答案进行评估。为了应对这些挑战，我们提出了一种数据生成管道，该管道将原始多模态数据转化为结构化语料库和问答对；引入了先进的重新排名阶段，以提高检索精度；并开发了一种参考匹配算法，以增强答案的可追溯性。我们将该系统应用于汽车工程领域，在一个5分制的评分体系中，经过LLM评估者的评审，与无RAG基线相比，该系统在事实准确性（+1.94）、信息量（+1.16）和有用性（+1.67）方面均有显著改进。这些结果突显了我们方法在不同方面上的有效性，尤其是答案的深厚根基和透明度。 

---
