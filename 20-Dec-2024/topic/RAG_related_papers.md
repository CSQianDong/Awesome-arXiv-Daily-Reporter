# VISA: Retrieval Augmented Generation with Visual Source Attribution 

**Title (ZH)**: VISA：带有视觉来源归因的检索增强生成 

**Authors**: Xueguang Ma, Shengyao Zhuang, Bevan Koopman, Guido Zuccon, Wenhu Chen, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.14457)  

**Abstract**: Generation with source attribution is important for enhancing the verifiability of retrieval-augmented generation (RAG) systems. However, existing approaches in RAG primarily link generated content to document-level references, making it challenging for users to locate evidence among multiple content-rich retrieved documents. To address this challenge, we propose Retrieval-Augmented Generation with Visual Source Attribution (VISA), a novel approach that combines answer generation with visual source attribution. Leveraging large vision-language models (VLMs), VISA identifies the evidence and highlights the exact regions that support the generated answers with bounding boxes in the retrieved document screenshots. To evaluate its effectiveness, we curated two datasets: Wiki-VISA, based on crawled Wikipedia webpage screenshots, and Paper-VISA, derived from PubLayNet and tailored to the medical domain. Experimental results demonstrate the effectiveness of VISA for visual source attribution on documents' original look, as well as highlighting the challenges for improvement. Code, data, and model checkpoints will be released. 

**Abstract (ZH)**: 来源归属性推理增强生成对于提高检索增强生成（RAG）系统的可验证性至关重要。然而，现有的RAG方法主要将生成的内容与文档级别的引用相关联，使得用户在多个内容丰富的检索文档中查找证据变得困难。为了解决这一挑战，我们提出了视觉来源归属性检索增强生成（Visual Source Attribution-VISA），这是一种结合了答案生成和视觉来源归属性的方法。利用大规模的视觉-语言模型（VLMs），VISA能够识别支持生成答案的证据，并在检索到的文档截图中通过边界框突出显示具体的地区。为了评估其有效性，我们构建了两个数据集：基于爬取的维基百科网页截图的Wiki-VISA，以及基于PubLayNet并针对医学领域的Paper-VISA。实验结果表明，VISA在文档原始外观的视觉来源归属性推理方面具有有效性，并指出了改进面临的挑战。代码、数据和模型检查点将被公开。 

---
# Dehallucinating Parallel Context Extension for Retrieval-Augmented Generation 

**Title (ZH)**: 去 hallucination 的并行上下文扩展以增强检索生成 

**Authors**: Zexiong Ma, Shengnan An, Zeqi Lin, Yanzhen Zou, Jian-Guang Lou, Bing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2412.14905)  

**Abstract**: Large language models (LLMs) are susceptible to generating hallucinated information, despite the integration of retrieval-augmented generation (RAG). Parallel context extension (PCE) is a line of research attempting to effectively integrating parallel (unordered) contexts, while it still suffers from hallucinations when adapted to RAG scenarios. In this paper, we propose DePaC (Dehallucinating Parallel Context Extension), which alleviates the hallucination problem with context-aware negative training and information-calibrated aggregation. DePaC is designed to alleviate two types of in-context hallucination: fact fabrication (i.e., LLMs present claims that are not supported by the contexts) and fact omission (i.e., LLMs fail to present claims that can be supported by the contexts). Specifically, (1) for fact fabrication, we apply the context-aware negative training that fine-tunes the LLMs with negative supervisions, thus explicitly guiding the LLMs to refuse to answer when contexts are not related to questions; (2) for fact omission, we propose the information-calibrated aggregation which prioritizes context windows with higher information increment from their contexts. The experimental results on nine RAG tasks demonstrate that DePaC significantly alleviates the two types of hallucination and consistently achieves better performances on these tasks. 

**Abstract (ZH)**: 大规模语言模型（LLMs）在整合检索增强生成（RAG）之后，仍然容易生成幻觉信息。平行上下文扩展（PCE）是一条研究路线，旨在有效整合无序的平行上下文，但在适应RAG场景时仍然会遭受幻觉问题。本文中，我们提出了一种名为DePaC（Dehallucinating Parallel Context Extension）的方案，通过上下文感知的负样本训练和信息校准聚合来缓解幻觉问题。DePaC 设计用于减轻两种类型的上下文幻觉：事实虚构（即LLMs提出缺乏上下文支持的断言）和事实遗漏（即LLMs未能提供能够由上下文支持的断言）。具体而言，（1）对于事实虚构，我们应用上下文感知的负样本训练，通过微调LLMs进行负监督，从而明确引导LLMs在上下文与问题无关时拒绝作答；（2）对于事实遗漏，我们提出信息校准聚合，优先考虑具有更高信息增量的上下文窗口。在九个RAG任务上的实验结果表明，DePaC 显著缓解了这两种类型的幻觉，并且在这些任务中始终实现了更好的表现。 

---
# PA-RAG: RAG Alignment via Multi-Perspective Preference Optimization 

**Title (ZH)**: PA-RAG：多角度偏好优化下的RAG对齐 

**Authors**: Jiayi Wu, Hengyi Cai, Lingyong Yan, Hao Sun, Xiang Li, Shuaiqiang Wang, Dawei Yin, Ming Gao  

**Link**: [PDF](https://arxiv.org/pdf/2412.14510)  

**Abstract**: The emergence of Retrieval-augmented generation (RAG) has alleviated the issues of outdated and hallucinatory content in the generation of large language models (LLMs), yet it still reveals numerous limitations. When a general-purpose LLM serves as the RAG generator, it often suffers from inadequate response informativeness, response robustness, and citation quality. Past approaches to tackle these limitations, either by incorporating additional steps beyond generating responses or optimizing the generator through supervised fine-tuning (SFT), still failed to align with the RAG requirement thoroughly. Consequently, optimizing the RAG generator from multiple preference perspectives while maintaining its end-to-end LLM form remains a challenge. To bridge this gap, we propose Multiple Perspective Preference Alignment for Retrieval-Augmented Generation (PA-RAG), a method for optimizing the generator of RAG systems to align with RAG requirements comprehensively. Specifically, we construct high-quality instruction fine-tuning data and multi-perspective preference data by sampling varied quality responses from the generator across different prompt documents quality scenarios. Subsequently, we optimize the generator using SFT and Direct Preference Optimization (DPO). Extensive experiments conducted on four question-answer datasets across three LLMs demonstrate that PA-RAG can significantly enhance the performance of RAG generators. Our code and datasets are available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）的出现缓解了大语言模型（LLMs）生成内容时过时和幻觉的问题，但仍然揭示出许多局限性。当通用大语言模型作为RAG生成器时，它往往存在响应信息量不足、响应稳健性和引文质量差的问题。过去的解决方法要么通过添加生成响应之外的步骤，要么通过监督微调（SFT）来优化生成器，但这些方法仍未彻底满足RAG的要求。因此，在保持其端到端大语言模型形式的同时，从多个偏好角度优化RAG生成器仍是一个挑战。为解决这一问题，我们提出了一种多视角偏好对齐方法（PA-RAG），用于全面优化RAG系统的生成器。具体来说，我们通过从不同场景的提示文档质量中采样多种质量的响应来构建高质量的指令微调数据和多视角偏好数据。随后，我们使用SFT和直接偏好优化（DPO）来优化生成器。我们针对三个大语言模型的四个问答数据集进行了广泛实验，结果表明PA-RAG可以显著提高RAG生成器的性能。我们的代码和数据集可在以下链接获取：[提供的URL]。 

---
# Multi-OphthaLingua: A Multilingual Benchmark for Assessing and Debiasing LLM Ophthalmological QA in LMICs 

**Title (ZH)**: 多语眼医：评估和消除低收入和中等收入国家（LMICs）中LLM眼科问答偏差的多语言基准 

**Authors**: David Restrepo, Chenwei Wu, Zhengxu Tang, Zitao Shuai, Thao Nguyen Minh Phan, Jun-En Ding, Cong-Tinh Dao, Jack Gallifant, Robyn Gayle Dychiao, Jose Carlo Artiaga, André Hiroshi Bando, Carolina Pelegrini Barbosa Gracitelli, Vincenz Ferrer, Leo Anthony Celi, Danielle Bitterman, Michael G Morley, Luis Filipe Nakayama  

**Link**: [PDF](https://arxiv.org/pdf/2412.14304)  

**Abstract**: Current ophthalmology clinical workflows are plagued by over-referrals, long waits, and complex and heterogeneous medical records. Large language models (LLMs) present a promising solution to automate various procedures such as triaging, preliminary tests like visual acuity assessment, and report summaries. However, LLMs have demonstrated significantly varied performance across different languages in natural language question-answering tasks, potentially exacerbating healthcare disparities in Low and Middle-Income Countries (LMICs). This study introduces the first multilingual ophthalmological question-answering benchmark with manually curated questions parallel across languages, allowing for direct cross-lingual comparisons. Our evaluation of 6 popular LLMs across 7 different languages reveals substantial bias across different languages, highlighting risks for clinical deployment of LLMs in LMICs. Existing debiasing methods such as Translation Chain-of-Thought or Retrieval-augmented generation (RAG) by themselves fall short of closing this performance gap, often failing to improve performance across all languages and lacking specificity for the medical domain. To address this issue, We propose CLARA (Cross-Lingual Reflective Agentic system), a novel inference time de-biasing method leveraging retrieval augmented generation and self-verification. Our approach not only improves performance across all languages but also significantly reduces the multilingual bias gap, facilitating equitable LLM application across the globe. 

**Abstract (ZH)**: 当前的眼科临床工作流程受到过度转诊、长时间等待以及复杂且异质性医疗记录的困扰。大型语言模型（LLMs）为自动化各种程序（如分诊、初步测试如视力评估以及报告总结）提供了有希望的解决方案。然而，在自然语言问答任务中，LLMs在不同语言上的表现差异显著，这可能导致在低收入和中等收入国家（LMICs）中医疗保健不平等现象的加剧。本研究介绍了首个经过人工精标的问题多语言眼科问答基准，这使得可以在不同语言之间进行直接的语言对比。我们在7种不同语言上对6种流行的LLMs进行了评估，揭示了不同语言之间的显著偏差，这强调了在LMICs中临床应用LLMs的风险。现有的去偏见方法如翻译链式思考或检索增强生成（RAG）单靠自己并不能弥补这一性能差距，往往无法在所有语言中提升性能，并且缺乏对医学领域的特异性。为了解决这一问题，我们提出了CLARA（跨语言反思代理系统），一种利用检索增强生成和自我验证的新颖的推理时间去偏见方法。我们的方法不仅提高了所有语言的性能，还显著缩小了多语言偏差差距，促进了全球范围内LLMs的公平应用。 

---
# Ontology-Aware RAG for Improved Question-Answering in Cybersecurity Education 

**Title (ZH)**: 面向本体的知识元增强型 Cybersecurity 教育问答系统 

**Authors**: Chengshuai Zhao, Garima Agrawal, Tharindu Kumarage, Zhen Tan, Yuli Deng, Ying-Chih Chen, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.14191)  

**Abstract**: Integrating AI into education has the potential to transform the teaching of science and technology courses, particularly in the field of cybersecurity. AI-driven question-answering (QA) systems can actively manage uncertainty in cybersecurity problem-solving, offering interactive, inquiry-based learning experiences. Large language models (LLMs) have gained prominence in AI-driven QA systems, offering advanced language understanding and user engagement. However, they face challenges like hallucinations and limited domain-specific knowledge, which reduce their reliability in educational settings. To address these challenges, we propose CyberRAG, an ontology-aware retrieval-augmented generation (RAG) approach for developing a reliable and safe QA system in cybersecurity education. CyberRAG employs a two-step approach: first, it augments the domain-specific knowledge by retrieving validated cybersecurity documents from a knowledge base to enhance the relevance and accuracy of the response. Second, it mitigates hallucinations and misuse by integrating a knowledge graph ontology to validate the final answer. Experiments on publicly available cybersecurity datasets show that CyberRAG delivers accurate, reliable responses aligned with domain knowledge, demonstrating the potential of AI tools to enhance education. 

**Abstract (ZH)**: 将下面的论文内容或标题翻译成中文，要符合学术规范：

将人工智能（AI）融入教育有可能变革科学和技术课程的教学，特别是在网络安全领域。基于AI的问答（QA）系统可以在解决网络安全问题时积极管理不确定性，提供互动式、探究式的学习体验。大型语言模型（LLMs）在AI驱动的QA系统中占据重要地位，提供了高级语言理解和用户参与。然而，这些系统面临着如幻觉和限制造成的领域知识有限等问题，这降低了它们在教育环境中的可靠性。为了应对这些挑战，我们提出了一种名为CyberRAG的概念，这是一种基于本体的检索增强生成（RAG）方法，旨在开发网络安全教育中的可靠且安全的QA系统。CyberRAG采用两步方法：首先，通过从知识库中检索验证过的网络安全文档来增强领域特定知识，提高响应的相关性和准确性；其次，通过集成知识图谱本体来验证最终的答案，以减少幻觉和误用。利用公开的网络安全数据集进行的实验结果显示，CyberRAG能够提供符合领域知识的准确且可靠的回答，这表明AI工具在教育中的增益潜力。 

---
# Face the Facts! Evaluating RAG-based Fact-checking Pipelines in Realistic Settings 

**Title (ZH)**: 面对事实！在实际场景中评估基于RAG的事实核查流水线 

**Authors**: Daniel Russo, Stefano Menini, Jacopo Staiano, Marco Guerini  

**Link**: [PDF](https://arxiv.org/pdf/2412.15189)  

**Abstract**: Natural Language Processing and Generation systems have recently shown the potential to complement and streamline the costly and time-consuming job of professional fact-checkers. In this work, we lift several constraints of current state-of-the-art pipelines for automated fact-checking based on the Retrieval-Augmented Generation (RAG) paradigm. Our goal is to benchmark, under more realistic scenarios, RAG-based methods for the generation of verdicts - i.e., short texts discussing the veracity of a claim - evaluating them on stylistically complex claims and heterogeneous, yet reliable, knowledge bases. Our findings show a complex landscape, where, for example, LLM-based retrievers outperform other retrieval techniques, though they still struggle with heterogeneous knowledge bases; larger models excel in verdict faithfulness, while smaller models provide better context adherence, with human evaluations favouring zero-shot and one-shot approaches for informativeness, and fine-tuned models for emotional alignment. 

**Abstract (ZH)**: 自然语言处理和生成系统近年来展示了补充和简化专业事实核查人员昂贵且耗时工作的潜力。在本文中，我们根据检索增强生成（RAG）范式，松解了当前自动事实核查管道中的若干限制。我们的目标是在更加现实的场景中对标，对基于RAG的方法进行验证，评估它们在生成裁决（即，讨论声明真实性的简短文本）时的表现，特别是在复杂风格的声明和异构但可靠的知识库上的表现。我们的研究发现了一个复杂的情况，例如，基于大规模语言模型的检索器在性能上优于其他检索技术，尽管它们在处理异构知识库方面仍存在问题；较大的模型在裁决的真实度方面表现优异，而较小的模型在上下文一致性方面表现更好，人类评价更倾向于零样本和一样本方法的信息性，而微调模型则更符合情感一致性。 

---
# Query pipeline optimization for cancer patient question answering systems 

**Title (ZH)**: 癌症患者问答系统中的查询管道优化 

**Authors**: Maolin He, Rena Gao, Mike Conway, Brian E. Chapman  

**Link**: [PDF](https://arxiv.org/pdf/2412.14751)  

**Abstract**: Retrieval-augmented generation (RAG) mitigates hallucination in Large Language Models (LLMs) by using query pipelines to retrieve relevant external information and grounding responses in retrieved knowledge. However, query pipeline optimization for cancer patient question-answering (CPQA) systems requires separately optimizing multiple components with domain-specific considerations. We propose a novel three-aspect optimization approach for the RAG query pipeline in CPQA systems, utilizing public biomedical databases like PubMed and PubMed Central. Our optimization includes: (1) document retrieval, utilizing a comparative analysis of NCBI resources and introducing Hybrid Semantic Real-time Document Retrieval (HSRDR); (2) passage retrieval, identifying optimal pairings of dense retrievers and rerankers; and (3) semantic representation, introducing Semantic Enhanced Overlap Segmentation (SEOS) for improved contextual understanding. On a custom-developed dataset tailored for cancer-related inquiries, our optimized RAG approach improved the answer accuracy of Claude-3-haiku by 5.24% over chain-of-thought prompting and about 3% over a naive RAG setup. This study highlights the importance of domain-specific query optimization in realizing the full potential of RAG and provides a robust framework for building more accurate and reliable CPQA systems, advancing the development of RAG-based biomedical systems. 

**Abstract (ZH)**: 检索增强生成（RAG）通过使用查询管道检索相关外部信息并使响应基于检索到的知识，从而减轻了大型语言模型（LLMs）中的幻觉问题。然而，针对癌症患者问答（CPQA）系统的查询管道优化需要分别优化多个具有特定领域考虑的组件。我们提出了一种针对CPQA系统中RAG查询管道的新型三方面优化方法，利用如PubMed和PubMed Central等公共生物医学数据库。我们的优化包括：(1) 文档检索，通过比较NCBI资源并引入混合语义实时文档检索（HSRDR）进行比较分析；(2) 段落检索，识别密集检索器和排序器的最佳配对；以及(3) 语义表示，引入语义增强重叠分段（SEOS）以提高上下文理解能力。在为癌症相关询问定制开发的数据集上，我们优化的RAG方法在Claude-3-haiku上的答案准确性相比思维链提示提高了5.24%，相比简单的RAG设置提高了约3%。本研究强调了在实现RAG最大潜力时领域特定查询优化的重要性，并提供了一个构建更准确和可靠CPQA系统的稳健框架，促进了基于RAG的生物医学系统的发展。 

---
# CORD: Balancing COnsistency and Rank Distillation for Robust Retrieval-Augmented Generation 

**Title (ZH)**: CORD: 平衡一致性和排名提炼以实现稳健的检索增强生成 

**Authors**: Youngwon Lee, Seung-won Hwang, Daniel Campos, Filip Graliński, Zhewei Yao, Yuxiong He  

**Link**: [PDF](https://arxiv.org/pdf/2412.14581)  

**Abstract**: With the adoption of retrieval-augmented generation (RAG), large language models (LLMs) are expected to ground their generation to the retrieved contexts. Yet, this is hindered by position bias of LLMs, failing to evenly attend to all contexts. Previous work has addressed this by synthesizing contexts with perturbed positions of gold segment, creating a position-diversified train set. We extend this intuition to propose consistency regularization with augmentation and distillation. First, we augment each training instance with its position perturbation to encourage consistent predictions, regardless of ordering. We also distill behaviors of this pair, although it can be counterproductive in certain RAG scenarios where the given order from the retriever is crucial for generation quality. We thus propose CORD, balancing COnsistency and Rank Distillation. CORD adaptively samples noise-controlled perturbations from an interpolation space, ensuring both consistency and respect for the rank prior. Empirical results show this balance enables CORD to outperform consistently in diverse RAG benchmarks. 

**Abstract (ZH)**: 随着检索增强生成（RAG）的应用，大规模语言模型（LLMs）有望将其生成基于检索到的上下文。然而，这受到了LLMs的位置偏见的阻碍，导致其不能平等地关注所有上下文。之前的研究通过合成带有扰动位置的正确片段，创建一个位置多样化的训练集，来解决这一问题。我们扩展了这一思路，提出了一种带有增强和蒸馏的一致性正则化方法。首先，我们通过为每个训练实例添加位置扰动来增强训练实例，从而鼓励一致的预测，而与顺序无关。我们还对该对进行了知识蒸馏，尽管在某些RAG场景中，检索器提供的顺序对于生成质量至关重要，强调知识蒸馏可能会适得其反。因此，我们提出了CORD（Consistency and Rank Distillation），它在保持一致性的基础上平衡排名蒸馏。CORD 从插值空间中自适应地采样噪声控制的扰动，确保一致性的同时尊重排名先验。实验结果表明，这种平衡使CORD在多种RAG基准测试中都表现出色。 

---
