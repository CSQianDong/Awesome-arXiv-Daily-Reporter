# Causal Graphs Meet Thoughts: Enhancing Complex Reasoning in Graph-Augmented LLMs 

**Title (ZH)**: 因果图结合思维：增强图增强型大语言模型的复杂推理能力 

**Authors**: Hang Luo, Jian Zhang, Chujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.14892)  

**Abstract**: In knowledge-intensive tasks, especially in high-stakes domains like medicine and law, it is critical not only to retrieve relevant information but also to provide causal reasoning and explainability. Large language models (LLMs) have achieved remarkable performance in natural language understanding and generation tasks. However, they often suffer from limitations such as difficulty in incorporating new knowledge, generating hallucinations, and explaining their reasoning process. To address these challenges, integrating knowledge graphs with Graph Retrieval-Augmented Generation (Graph RAG) has emerged as an effective solution. Traditional Graph RAG methods often rely on simple graph traversal or semantic similarity, which do not capture causal relationships or align well with the model's internal reasoning steps. This paper proposes a novel pipeline that filters large knowledge graphs to emphasize cause-effect edges, aligns the retrieval process with the model's chain-of-thought (CoT), and enhances reasoning through multi-stage path improvements. Experiments on medical question-answering tasks show consistent gains, with up to a 10\% absolute improvement across multiple large language models (LLMs). This approach demonstrates the value of combining causal reasoning with stepwise retrieval, leading to more interpretable and logically grounded solutions for complex queries. 

**Abstract (ZH)**: 在知识密集型任务中，特别是在医学和法律等高风险领域，不仅需要检索相关的信息，还需要提供因果推理和可解释性。大型语言模型（LLMs）在自然语言理解与生成任务中取得了显著的成果。然而，它们往往存在难以融入新知识、生成虚假信息以及解释推理过程等局限性。为了解决这些挑战，将知识图谱与图检索增强生成（Graph RAG）相结合已成为有效的方法。传统的Graph RAG方法通常依赖于简单的图遍历或语义相似度，无法捕捉因果关系或与模型的内部推理步骤很好地对齐。本文提出了一种新的管道，该管道通过对大规模知识图谱进行过滤以强调因果关系边，将检索过程与模型的链式思维（CoT）对齐，并通过多阶段路径改进来增强推理。在医学问答任务上的实验结果表明，这种方法在多个大型语言模型（LLMs）上表现出了一致的改进，绝对改进幅度最高可达10%。该方法证明了将因果推理与逐步检索相结合的价值，从而使复杂查询的结果更易解释且逻辑基础更牢固。 

---
# Advanced Real-Time Fraud Detection Using RAG-Based LLMs 

**Title (ZH)**: 基于RAG的LLM的高级实时欺诈检测 

**Authors**: Gurjot Singh, Prabhjot Singh, Maninder Singh  

**Link**: [PDF](https://arxiv.org/pdf/2501.15290)  

**Abstract**: Artificial Intelligence has become a double edged sword in modern society being both a boon and a bane. While it empowers individuals it also enables malicious actors to perpetrate scams such as fraudulent phone calls and user impersonations. This growing threat necessitates a robust system to protect individuals In this paper we introduce a novel real time fraud detection mechanism using Retrieval Augmented Generation technology to address this challenge on two fronts. First our system incorporates a continuously updating policy checking feature that transcribes phone calls in real time and uses RAG based models to verify that the caller is not soliciting private information thus ensuring transparency and the authenticity of the conversation. Second we implement a real time user impersonation check with a two step verification process to confirm the callers identity ensuring accountability. A key innovation of our system is the ability to update policies without retraining the entire model enhancing its adaptability. We validated our RAG based approach using synthetic call recordings achieving an accuracy of 97.98 percent and an F1score of 97.44 percent with 100 calls outperforming state of the art methods. This robust and flexible fraud detection system is well suited for real world deployment. 

**Abstract (ZH)**: 人工智能已成为现代社会一把双刃剑，既是福又是祸。它不仅赋能个体，同时也让恶意行为者能进行诸如欺诈电话和冒充用户等行为。这一不断增长的威胁需要一个强大的系统来保护个体。在本文中，我们引入了一种基于检索增强生成技术的新型实时欺诈检测机制，从两个方面应对这一挑战。首先，我们的系统包含一个持续更新的策略检查功能，能够实时转录电话内容，并利用基于检索增强生成（RAG）的模型验证呼叫者是否在索取私人信息，从而确保对话的透明性和真实身份。其次，我们实施了一种实时用户冒充检查机制，通过两步验证过程确认呼叫者的身份，确保其可追溯性。系统的一个关键创新是能够无需重新训练整个模型即可更新策略，从而增强其适应性。我们使用合成电话录音验证了基于RAG的方法，在100次电话中实现了97.98％的准确率和97.44％的F1分数，超越了现有最先进的方法。这一强大且灵活的欺诈检测系统非常适合实际部署。 

---
# CFT-RAG: An Entity Tree Based Retrieval Augmented Generation Algorithm With Cuckoo Filter 

**Title (ZH)**: CFT-RAG：基于实体树的检索增强生成算法，采用 cuckoo 过滤器 

**Authors**: Zihang Li, Yangdong Ruan, Wenjun Liu, Zhengyang Wang, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15098)  

**Abstract**: Although retrieval-augmented generation(RAG) significantly improves generation quality by retrieving external knowledge bases and integrating generated content, it faces computational efficiency bottlenecks, particularly in knowledge retrieval tasks involving hierarchical structures for Tree-RAG. This paper proposes a Tree-RAG acceleration method based on the improved Cuckoo Filter, which optimizes entity localization during the retrieval process to achieve significant performance improvements. Tree-RAG effectively organizes entities through the introduction of a hierarchical tree structure, while the Cuckoo Filter serves as an efficient data structure that supports rapid membership queries and dynamic updates. The experiment results demonstrate that our method is much faster than naive Tree-RAG while maintaining high levels of generative quality. When the number of trees is large, our method is hundreds of times faster than naive Tree-RAG. Our work is available at this https URL. 

**Abstract (ZH)**: 尽管检索增强生成（RAG）通过检索外部知识库并整合生成内容显著提高了生成质量，但在涉及层次结构的知识检索任务（如Tree-RAG）中，它面临计算效率瓶颈。本文提出了一种基于改进的Cuckoo Filter的Tree-RAG加速方法，该方法在检索过程中优化实体定位，以实现显著的性能提升。通过引入层次树结构，Tree-RAG有效组织了实体，而Cuckoo Filter作为一种高效的数据结构，支持快速的成员查询和动态更新。实验结果表明，与朴素的Tree-RAG相比，我们的方法在保持高度生成质量的同时具有更快的速度。当树的数量较大时，我们的方法比朴素的Tree-RAG快数百倍。我们的工作可以在以下链接访问：[此链接]。 

---
# LLM as HPC Expert: Extending RAG Architecture for HPC Data 

**Title (ZH)**: 将以下论文内容或标题翻译成中文，确保符合学术规范：

LLM作为HPC专家：扩展基于RAG的架构以处理HPC数据

解释：
- LLM：Large Language Model（大规模语言模型）
- HPC：High-Performance Computing（高性能计算）
- RAG：Retrieval-Augmented Generation（检索增强生成）

这个标题翻译已经尽可能保持了原有的专业术语，并且符合中文的表达习惯。 

**Authors**: Yusuke Miyashita, Patrick Kin Man Tung, Johan Barthélemy  

**Link**: [PDF](https://arxiv.org/pdf/2501.14733)  

**Abstract**: High-Performance Computing (HPC) is crucial for performing advanced computational tasks, yet their complexity often challenges users, particularly those unfamiliar with HPC-specific commands and workflows. This paper introduces Hypothetical Command Embeddings (HyCE), a novel method that extends Retrieval-Augmented Generation (RAG) by integrating real-time, user-specific HPC data, enhancing accessibility to these systems. HyCE enriches large language models (LLM) with real-time, user-specific HPC information, addressing the limitations of fine-tuned models on such data. We evaluate HyCE using an automated RAG evaluation framework, where the LLM itself creates synthetic questions from the HPC data and serves as a judge, assessing the efficacy of the extended RAG with the evaluation metrics relevant for HPC tasks. Additionally, we tackle essential security concerns, including data privacy and command execution risks, associated with deploying LLMs in HPC environments. This solution provides a scalable and adaptable approach for HPC clusters to leverage LLMs as HPC expert, bridging the gap between users and the complex systems of HPC. 

**Abstract (ZH)**: 高性能计算（HPC）对于执行高级计算任务至关重要，但其复杂性往往挑战用户，尤其是那些不熟悉HPC特定命令和工作流的用户。本文介绍了一种名为假设命令嵌入（HyCE）的新方法，该方法通过集成实时的用户特定HPC数据扩展了检索增强生成（RAG），从而提高HPC系统的易用性。HyCE 通过将实时的用户特定HPC信息融入大规模语言模型（LLM），解决了调优模型在处理此类数据时的局限性。我们使用自动化RAG评估框架来评估HyCE，在该框架中，LLM本身从HPC数据中生成合成问题并充当裁判，评估扩展的RAG的有效性，以符合HPC任务的相关评估指标。此外，我们还解决了在HPC环境中部署LLM时的关键安全问题，包括数据隐私和命令执行风险。该解决方案提供了一种可扩展和适应性强的方法，使HPC集群能够利用LLM作为HPC专家，从而弥合用户与HPC复杂系统的差距。 

---
# CG-RAG: Research Question Answering by Citation Graph Retrieval-Augmented LLMs 

**Title (ZH)**: CG-RAG：引用图检索增强的LLM研究问题回答 

**Authors**: Yuntong Hu, Zhihan Lei, Zhongjie Dai, Allen Zhang, Abhinav Angirekula, Zheng Zhang, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15067)  

**Abstract**: Research question answering requires accurate retrieval and contextual understanding of scientific literature. However, current Retrieval-Augmented Generation (RAG) methods often struggle to balance complex document relationships with precise information retrieval. In this paper, we introduce Contextualized Graph Retrieval-Augmented Generation (CG-RAG), a novel framework that integrates sparse and dense retrieval signals within graph structures to enhance retrieval efficiency and subsequently improve generation quality for research question answering. First, we propose a contextual graph representation for citation graphs, effectively capturing both explicit and implicit connections within and across documents. Next, we introduce Lexical-Semantic Graph Retrieval (LeSeGR), which seamlessly integrates sparse and dense retrieval signals with graph encoding. It bridges the gap between lexical precision and semantic understanding in citation graph retrieval, demonstrating generalizability to existing graph retrieval and hybrid retrieval methods. Finally, we present a context-aware generation strategy that utilizes the retrieved graph-structured information to generate precise and contextually enriched responses using large language models (LLMs). Extensive experiments on research question answering benchmarks across multiple domains demonstrate that our CG-RAG framework significantly outperforms RAG methods combined with various state-of-the-art retrieval approaches, delivering superior retrieval accuracy and generation quality. 

**Abstract (ZH)**: 以下是根据学术规范翻译的内容：

研究问题回答需要精确检索和理解科学文献的上下文。然而，当前的检索增强生成（RAG）方法往往难以平衡复杂文档关系与精确信息检索之间的关系。本文介绍了一种名为情境化图检索增强生成（CG-RAG）的新框架，该框架通过在图结构中整合稀疏和密集的检索信号来提高检索效率，从而改善研究问题回答的生成质量。首先，我们提出了一种文献引用图的情境化图表示方法，有效地捕获文档内部及跨文档的显式和隐式联系。其次，我们引入了基于词性和语义图检索（LeSeGR）的方法，该方法无缝地将稀疏和密集的检索信号与图编码相结合。LeSeGR在文献引用图检索中填补了词性和语义理解之间的差距，展示了对该领域现有图检索和混合检索方法的通用性。最后，我们提出了一个上下文感知的生成策略，该策略利用检索到的图结构信息和大规模语言模型（LLM）生成精确且上下文丰富的响应。在多个领域的研究问题回答基准测试中的广泛实验表明，我们的CG-RAG框架显著优于结合了各种先进检索方法的RAG方法，提供了更高的检索准确性和生成质量。 

---
# Parametric Retrieval Augmented Generation 

**Title (ZH)**: 参数化检索增强生成 

**Authors**: Weihang Su, Yichen Tang, Qingyao Ai, Junxi Yan, Changyue Wang, Hongning Wang, Ziyi Ye, Yujia Zhou, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2501.15915)  

**Abstract**: Retrieval-augmented generation (RAG) techniques have emerged as a promising solution to enhance the reliability of large language models (LLMs) by addressing issues like hallucinations, outdated knowledge, and domain adaptation. In particular, existing RAG methods append relevant documents retrieved from external corpus or databases to the input of LLMs to guide their generation process, which we refer to as the in-context knowledge injection method. While this approach is simple and often effective, it has inherent limitations. Firstly, increasing the context length and number of relevant documents can lead to higher computational overhead and degraded performance, especially in complex reasoning tasks. More importantly, in-context knowledge injection operates primarily at the input level, but LLMs store their internal knowledge in their parameters. This gap fundamentally limits the capacity of in-context methods. To this end, we introduce Parametric retrieval-augmented generation (Parametric RAG), a new RAG paradigm that integrates external knowledge directly into the parameters of feed-forward networks (FFN) of an LLM through document parameterization. This approach not only saves online computational costs by eliminating the need to inject multiple documents into the LLMs' input context, but also deepens the integration of external knowledge into the parametric knowledge space of the LLM. Experimental results demonstrate that Parametric RAG substantially enhances both the effectiveness and efficiency of knowledge augmentation in LLMs. Also, it can be combined with in-context RAG methods to achieve even better performance.
We have open-sourced all the code, data, and models in the following anonymized GitHub link: this https URL 

**Abstract (ZH)**: 检索增强生成（RAG）技术已经成为了提升大型语言模型（LLMs）可靠性的有前途的解决方案，尤其是在解决幻觉、过时的知识和领域适应性等问题方面表现出色。具体而言，现有的RAG方法通过将从外部语料库或数据库中检索到的相关文档附加到LLM的输入中，以指导生成过程，这种方法我们称之为上下文内知识注入方法。尽管这种方法简单且通常有效，但它具有内在局限性。首先，增加上下文长度和相关文档的数量会带来更高的计算开销和性能下降，尤其是在复杂的推理任务中。更重要的是，上下文内知识注入主要在输入级别进行，但LLM将其内部知识存储在参数中。这一差距从根本上限制了上下文内方法的能力。为了解决这些问题，我们引入了参数化检索增强生成（Parametric RAG），这是一种新的RAG范式，通过文档参数化将外部知识直接整合到LLM前馈网络（FFN）的参数中。这种方法不仅可以通过消除向LLM输入上下文注入多个文档的需求来节省在线计算成本，还能更深层次地将外部知识整合到LLM的参数化知识空间中。实验结果表明，Parametric RAG显著提升了LLM中知识增强的有效性和效率。此外，它还可以与上下文内RAG方法结合使用，实现更佳的性能。

我们已在去标识化的GitHub链接中开源了所有代码、数据和模型：[这个链接](this https URL) 

---
# LemmaHead: RAG Assisted Proof Generation Using Large Language Models 

**Title (ZH)**: LemmaHead：使用大语言模型的RAG辅助定理证明 

**Authors**: Tianbo Yang, Mingqi Yang, Hongyi Zhao, Tianshuo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2501.15797)  

**Abstract**: Developing the logic necessary to solve mathematical problems or write mathematical proofs is one of the more difficult objectives for large language models (LLMS). Currently, the most popular methods in literature consists of fine-tuning the model on written mathematical content such as academic publications and textbooks, so that the model can learn to emulate the style of mathematical writing. In this project, we explore the effectiveness of using retrieval augmented generation (RAG) to address gaps in the mathematical reasoning of LLMs. We develop LemmaHead, a RAG knowledge base that supplements queries to the model with relevant mathematical context, with particular focus on context from published textbooks. To measure our model's performance in mathematical reasoning, our testing paradigm focuses on the task of automated theorem proving via generating proofs to a given mathematical claim in the Lean formal language. 

**Abstract (ZH)**: 开发解决数学问题或撰写数学证明所需的逻辑是大型语言模型（LLMs）面临的更加困难的目标之一。目前，文献中最流行的 方法是通过针对学术出版物和教科书中的数学内容进行微调，使模型能够学习模仿数学写作的风格。在这个项目中，我们探索使用检索增强生成（RAG）来弥补LLMs在数学推理方面的不足。我们开发了LemmaHead，这是一种RAG知识库，通过向模型提供相关数学上下文（特别是来自已出版教科书的上下文）来补充查询。为了衡量模型在数学推理方面的表现，我们测试范式关注的是通过生成给定数学命题在Lean形式语言中的证明来进行自动定理证明的任务。 

---
# Improving Retrieval-Augmented Generation through Multi-Agent Reinforcement Learning 

**Title (ZH)**: 通过多智能体强化学习提升检索增强生成 

**Authors**: Yiqun Chen, Lingyong Yan, Weiwei Sun, Xinyu Ma, Yi Zhang, Shuaiqiang Wang, Dawei Yin, Yiming Yang, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2501.15228)  

**Abstract**: Retrieval-augmented generation (RAG) is extensively utilized to incorporate external, current knowledge into large language models, thereby minimizing hallucinations. A standard RAG pipeline may comprise several components, such as query rewriting, document retrieval, document filtering, and answer generation. However, these components are typically optimized separately through supervised fine-tuning, which can lead to misalignments between the objectives of individual modules and the overarching aim of generating accurate answers in question-answering (QA) tasks. Although recent efforts have explored reinforcement learning (RL) to optimize specific RAG components, these approaches often focus on overly simplistic pipelines with only two components or do not adequately address the complex interdependencies and collaborative interactions among the modules. To overcome these challenges, we propose treating the RAG pipeline as a multi-agent cooperative task, with each component regarded as an RL agent. Specifically, we present MMOA-RAG, a Multi-Module joint Optimization Algorithm for RAG, which employs multi-agent reinforcement learning to harmonize all agents' goals towards a unified reward, such as the F1 score of the final answer. Experiments conducted on various QA datasets demonstrate that MMOA-RAG improves the overall pipeline performance and outperforms existing baselines. Furthermore, comprehensive ablation studies validate the contributions of individual components and the adaptability of MMOA-RAG across different RAG components and datasets. The code of MMOA-RAG is on this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）被广泛应用于将外部和最新的知识整合到大型语言模型中，从而减少幻觉的产生。标准的RAG管道可能包括多个组件，如查询重写、文档检索、文档过滤和答案生成。然而，这些组件通常通过有监督微调分别优化，这可能导致各个模块的目标与生成准确答案的整体目标之间的对齐问题。尽管最近的研究尝试使用强化学习（RL）来优化特定的RAG组件，但这些方法往往集中在过于简单的管道设计上，只包含两个组件，或者未能充分解决模块之间复杂相互依赖性和协作交互的问题。为了克服这些挑战，我们提出将RAG管道视为多智能体协作任务，并将每个组件视为一个RL智能体。具体而言，我们提出了MMOA-RAG，这是一种用于RAG的多模块联合优化算法，利用多智能体强化学习将所有智能体的目标协调至统一的奖励，例如最终答案的F1分数。在多个问答数据集上的实验表明，MMOA-RAG提高了整个管道的性能，并优于现有基线。进一步的消融研究验证了各组件的贡献以及MMOA-RAG在不同RAG组件和数据集上的适应性。MMOA-RAG的代码请参见此链接：[此处的链接]。 

---
# URAG: Implementing a Unified Hybrid RAG for Precise Answers in University Admission Chatbots -- A Case Study at HCMUT 

**Title (ZH)**: URAG: 实现统一混合RAG以在大学录取聊天机器人中提供精确答案——河内科技大学案例研究 

**Authors**: Long Nguyen, Tho Quan  

**Link**: [PDF](https://arxiv.org/pdf/2501.16276)  

**Abstract**: With the rapid advancement of Artificial Intelligence, particularly in Natural Language Processing, Large Language Models (LLMs) have become pivotal in educational question-answering systems, especially university admission chatbots. Concepts such as Retrieval-Augmented Generation (RAG) and other advanced techniques have been developed to enhance these systems by integrating specific university data, enabling LLMs to provide informed responses on admissions and academic counseling. However, these enhanced RAG techniques often involve high operational costs and require the training of complex, specialized modules, which poses challenges for practical deployment. Additionally, in the educational context, it is crucial to provide accurate answers to prevent misinformation, a task that LLM-based systems find challenging without appropriate strategies and methods. In this paper, we introduce the Unified RAG (URAG) Framework, a hybrid approach that significantly improves the accuracy of responses, particularly for critical queries. Experimental results demonstrate that URAG enhances our in-house, lightweight model to perform comparably to state-of-the-art commercial models. Moreover, to validate its practical applicability, we conducted a case study at our educational institution, which received positive feedback and acclaim. This study not only proves the effectiveness of URAG but also highlights its feasibility for real-world implementation in educational settings. 

**Abstract (ZH)**: 随着人工智能的迅速发展，特别是在自然语言处理领域的进步，大型语言模型（LLMs）已成为教育问答系统中的重要组成部分，尤其是在大学招生聊天机器人的应用中。检索增强生成（RAG）等先进技术的概念和其他高级技术已经发展起来，通过整合特定的大学数据来增强这些系统，使LLMs能够提供涉及入学和学术指导的知情回答。然而，这些增强的RAG技术通常涉及较高的运营成本，并需要训练复杂的专门模块，这对实际部署构成了挑战。此外，在教育背景下，提供准确的答案以防止 misinformation至关重要，这是一项LLM基础系统在缺乏适当策略和方法的情况下难以完成的任务。在本文中，我们介绍了统一RAG（URAG）框架，这是一种混合方法，显著提高了对关键查询的响应准确性。实验结果显示，URAG能够使我们内部的轻量级模型表现得与最先进的商业模型相当。此外，为了验证其实用性，我们在教育机构进行了一项案例研究，收到了积极的反馈和赞誉。这项研究不仅证明了URAG的有效性，还突显了它在教育环境中实际应用的可行性。 

---
# Provence: efficient and robust context pruning for retrieval-augmented generation 

**Title (ZH)**: Provence：高效的鲁棒上下文裁剪方法以提升检索增强生成模型的性能 

**Authors**: Nadezhda Chirkova, Thibault Formal, Vassilina Nikoulina, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2501.16214)  

**Abstract**: Retrieval-augmented generation improves various aspects of large language models (LLMs) generation, but suffers from computational overhead caused by long contexts as well as the propagation of irrelevant retrieved information into generated responses. Context pruning deals with both aspects, by removing irrelevant parts of retrieved contexts before LLM generation. Existing context pruning approaches are however limited, and do not provide a universal model that would be both efficient and robust in a wide range of scenarios, e.g., when contexts contain a variable amount of relevant information or vary in length, or when evaluated on various domains. In this work, we close this gap and introduce Provence (Pruning and Reranking Of retrieVEd relevaNt ContExts), an efficient and robust context pruner for Question Answering, which dynamically detects the needed amount of pruning for a given context and can be used out-of-the-box for various domains. The three key ingredients of Provence are formulating the context pruning task as sequence labeling, unifying context pruning capabilities with context reranking, and training on diverse data. Our experimental results show that Provence enables context pruning with negligible to no drop in performance, in various domains and settings, at almost no cost in a standard RAG pipeline. We also conduct a deeper analysis alongside various ablations to provide insights into training context pruners for future work. 

**Abstract (ZH)**: 检索增强生成可以提高大型语言模型（LLMs）生成的各个方面，但会长语境和检索信息的传播导致的相关性和无关联信息的传播带来计算开销。上下文剪枝通过在LLM生成前去除检索上下文中的无关部分，来处理这两个方面的问题。然而，现有的上下文剪枝方法存在一定局限性，无法在多种场景下提供高效且稳定的模型，例如，当上下文包含不同量的相关信息或长度不同，或在不同领域接受评估时。本研究填补了这一空白，推出了“Provence（剪枝和排序检索相关上下文）”，这是一种高效且稳健的问题解答上下文剪枝工具，能够根据给定的上下文动态检测所需的剪枝量，并且可以通用应用于多种领域。Provence的三大关键成分为：将上下文剪枝任务表述为序列标注、将上下文剪枝能力与上下文排序统一，以及在多样数据上进行训练。实验结果显示，Provence能够在多种领域和设置下实现上下文剪枝，几乎没有性能下降，并且在标准检索增强生成（RAG）管道中的成本几乎可以忽略不计。我们还进行了一系列的深入分析和多方面消融研究，以提供对未来培训上下文剪枝器的见解。 

---
# ASRank: Zero-Shot Re-Ranking with Answer Scent for Document Retrieval 

**Title (ZH)**: ASRank：基于答案香气的零样本重排序方法在文档检索中的应用 

**Authors**: Abdelrahman Abdallah, Jamshid Mozafari, Bhawna Piryani, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2501.15245)  

**Abstract**: Retrieval-Augmented Generation (RAG) models have drawn considerable attention in modern open-domain question answering. The effectiveness of RAG depends on the quality of the top retrieved documents. However, conventional retrieval methods sometimes fail to rank the most relevant documents at the top. In this paper, we introduce ASRank, a new re-ranking method based on scoring retrieved documents using zero-shot answer scent which relies on a pre-trained large language model to compute the likelihood of the document-derived answers aligning with the answer scent. Our approach demonstrates marked improvements across several datasets, including NQ, TriviaQA, WebQA, ArchivalQA, HotpotQA, and Entity Questions. Notably, ASRank increases Top-1 retrieval accuracy on NQ from $19.2\%$ to $46.5\%$ for MSS and $22.1\%$ to $47.3\%$ for BM25. It also shows strong retrieval performance on several datasets compared to state-of-the-art methods (47.3 Top-1 by ASRank vs 35.4 by UPR by BM25). 

**Abstract (ZH)**: 以下是翻译的内容，符合学术规范：

检索增强生成（RAG）模型在现代开放域问答中引起了广泛关注。RAG的有效性取决于检索出的顶级文档质量。然而，传统的检索方法有时无法将最相关文档排在最前。本文介绍了ASRank，这是一种新的重排序方法，它通过使用预训练的大语言模型计算文档衍生答案与问题答案线索一致性的可能性来进行评分，从而对检索出的文档进行重新排序。我们的方法在多个数据集（包括NQ、TriviaQA、WebQA、ArchivalQA、HotpotQA和Entity Questions）上均表现出显著的改进。值得注意的是，ASRank将MSS下NQ数据集的Top-1检索准确率从19.2%提高到46.5%，将BM25下的Top-1检索准确率从22.1%提高到47.3%。此外，ASRank在多个数据集上的检索性能明显优于最先进的方法（ASRank的47.3 Top-1 vs BM25的UPR方法的35.4）。 

---
# Federated Retrieval Augmented Generation for Multi-Product Question Answering 

**Title (ZH)**: 联邦检索增强生成在多产品问答中的应用 

**Authors**: Parshin Shojaee, Sai Sree Harsha, Dan Luo, Akash Maharaj, Tong Yu, Yunyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2501.14998)  

**Abstract**: Recent advancements in Large Language Models and Retrieval-Augmented Generation have boosted interest in domain-specific question-answering for enterprise products. However, AI Assistants often face challenges in multi-product QA settings, requiring accurate responses across diverse domains. Existing multi-domain RAG-QA approaches either query all domains indiscriminately, increasing computational costs and LLM hallucinations, or rely on rigid resource selection, which can limit search results. We introduce MKP-QA, a novel multi-product knowledge-augmented QA framework with probabilistic federated search across domains and relevant knowledge. This method enhances multi-domain search quality by aggregating query-domain and query-passage probabilistic relevance. To address the lack of suitable benchmarks for multi-product QAs, we also present new datasets focused on three Adobe products: Adobe Experience Platform, Target, and Customer Journey Analytics. Our experiments show that MKP-QA significantly boosts multi-product RAG-QA performance in terms of both retrieval accuracy and response quality. 

**Abstract (ZH)**: 近年来，大型语言模型和检索增强生成技术的最新进展激发了对特定领域的企业产品问答的兴趣。然而，在多产品问答设置中，AI助手常常面临挑战，需要在多样化的领域中提供精确的响应。现有的多领域RAG-QA方法要么不分青红皂白地查询所有领域，增加计算成本并导致LLM幻觉，要么依赖刚性资源选择，这可能限制搜索结果。我们提出了MKP-QA，这是一种新颖的多产品知识增强问答框架，包括跨领域和相关知识的概率联邦搜索。该方法通过聚合查询领域和查询段落的概率相关性来提高多领域的搜索质量。为了应对多产品问答缺乏合适的基准测试，我们还介绍了新的针对Adobe三项产品的数据集：Adobe Experience Platform、Target和Customer Journey Analytics。实验结果显示，MKP-QA在检索准确性和响应质量方面显著提升了多产品RAG-QA的性能。 

---
