# ResearchTown: Simulator of Human Research Community 

**Title (ZH)**: ResearchTown：人类研究社区模拟器 

**Authors**: Haofei Yu, Zhaochen Hong, Zirui Cheng, Kunlun Zhu, Keyang Xuan, Jinwei Yao, Tao Feng, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2412.17767)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable potential in scientific domains, yet a fundamental question remains unanswered: Can we simulate human research communities with LLMs? Addressing this question can deepen our understanding of the processes behind idea brainstorming and inspire the automatic discovery of novel scientific insights. In this work, we propose ResearchTown, a multi-agent framework for research community simulation. Within this framework, the human research community is simplified and modeled as an agent-data graph, where researchers and papers are represented as agent-type and data-type nodes, respectively, and connected based on their collaboration relationships. We also introduce TextGNN, a text-based inference framework that models various research activities (e.g., paper reading, paper writing, and review writing) as special forms of a unified message-passing process on the agent-data graph. To evaluate the quality of the research simulation, we present ResearchBench, a benchmark that uses a node-masking prediction task for scalable and objective assessment based on similarity. Our experiments reveal three key findings: (1) ResearchTown can provide a realistic simulation of collaborative research activities, including paper writing and review writing; (2) ResearchTown can maintain robust simulation with multiple researchers and diverse papers; (3) ResearchTown can generate interdisciplinary research ideas that potentially inspire novel research directions. 

**Abstract (ZH)**: 大型语言模型（LLMs）在科学领域展现出了巨大的潜力，但一个基本的问题仍未得到解答：我们能否使用LLMs模拟人类研究社区？回答这个问题有助于深化我们对创意生成过程的理解，并激发对新颖科学洞察的自动发现。在此项工作中，我们提出了ResearchTown，这是一种用于研究社区模拟的多智能体框架。在此框架中，人类研究社区被简化并建模为智能体-数据图，其中研究人员和论文分别用智能体类型节点和数据类型节点表示，并基于他们的合作关系进行连接。我们还引入了TextGNN，这是一种基于文本的推理框架，将各种研究活动（例如论文阅读、论文写作和审稿写作）建模为智能体-数据图上的统一消息传递过程的特殊形式。为了评估研究模拟的质量，我们提出了ResearchBench，这是一个基准测试，利用节点掩码预测任务进行基于相似性的可扩展和客观评估。我们的实验揭示了三个关键发现：（1）ResearchTown能够提供合作研究活动的现实模拟，包括论文写作和审稿写作；（2）ResearchTown能够使用多个研究人员和多样化论文保持稳健的模拟；（3）ResearchTown能够生成跨学科的研究理念，可能激发新的研究方向。 

---
# In Case You Missed It: ARC 'Challenge' Is Not That Challenging 

**Title (ZH)**: 如果您没有错过：ARC“挑战”并没有想象中那么具有挑战性 

**Authors**: Łukasz Borchmann  

**Link**: [PDF](https://arxiv.org/pdf/2412.17758)  

**Abstract**: ARC Challenge appears more difficult than ARC Easy for modern LLMs primarily due to an evaluation setup that prevents direct comparison of answer choices rather than inherent complexity. Although some researchers have quietly shifted to a more appropriate scheme over the last year, the implications of this change have yet to be widely acknowledged. We highlight this overlooked shift, show how similar evaluation practices falsely imply reasoning deficits in other benchmarks, and demonstrate that fairer methods dramatically reduce performance gaps (e.g. on SIQA) and even yield superhuman results (OpenBookQA). In doing so, we reveal how evaluation shapes perceived difficulty and offer guidelines to ensure that multiple-choice evaluations accurately reflect actual model capabilities. 

**Abstract (ZH)**: 与ARC Easy相比，现代大型语言模型（LLMs）在ARC挑战中的表现更加困难，主要原因在于评估设置限制了直接比较答案选项，而不是固有的复杂性。尽管在过去一年中，一些研究人员已经悄悄转向了更合适的方案，但这种变化的影响尚未得到广泛认可。我们强调了这一被忽视的转变，展示了类似的评估实践如何错误地暗示其他基准中的推理缺陷，并证明公平的方法可以显著缩小性能差距（例如在SIQA上）甚至达到超人类水平（如OpenBookQA）。通过这一点，我们揭示了评估如何塑造感知难度，并提出了确保多项选择评估能准确反映模型实际能力的指南。 

---
# Deliberation in Latent Space via Differentiable Cache Augmentation 

**Title (ZH)**: 通过可微缓存增强在潜在空间中的 deliberation 

**Authors**: Luyang Liu, Jonas Pfeiffer, Jiaxing Wu, Jun Xie, Arthur Szlam  

**Link**: [PDF](https://arxiv.org/pdf/2412.17747)  

**Abstract**: Techniques enabling large language models (LLMs) to "think more" by generating and attending to intermediate reasoning steps have shown promise in solving complex problems. However, the standard approaches generate sequences of discrete tokens immediately before responding, and so they can incur significant latency costs and be challenging to optimize. In this work, we demonstrate that a frozen LLM can be augmented with an offline coprocessor that operates on the model's key-value (kv) cache. This coprocessor augments the cache with a set of latent embeddings designed to improve the fidelity of subsequent decoding. We train this coprocessor using the language modeling loss from the decoder on standard pretraining data, while keeping the decoder itself frozen. This approach enables the model to learn, in an end-to-end differentiable fashion, how to distill additional computation into its kv-cache. Because the decoder remains unchanged, the coprocessor can operate offline and asynchronously, and the language model can function normally if the coprocessor is unavailable or if a given cache is deemed not to require extra computation. We show experimentally that when a cache is augmented, the decoder achieves lower perplexity on numerous subsequent tokens. Furthermore, even without any task-specific training, our experiments demonstrate that cache augmentation consistently reduces perplexity and improves performance across a range of reasoning-intensive tasks. 

**Abstract (ZH)**: 通过生成和关注中间推理步骤，使大型语言模型（LLM）能够“更好地思考”的技术在解决复杂问题方面显示出了潜力。然而，标准方法在响应前立即生成离散的令牌序列，因此会引发显著的延迟成本并使其优化变得具有挑战性。在这项工作中，我们证明了一个冻结的LLM可以通过一个操作于模型的关键值（kv）缓存的离线协处理器来增强，该协处理器借助一组旨在提高后续解码准确性的潜在嵌入来增强缓存。我们通过在标准预训练数据上使用解码器的语言模型损失来训练这个协处理器，同时保持解码器本身不变。这种方法使模型能够在端到端可微分的方式下学习如何将额外的计算提炼到其kv缓存中。由于解码器未改变，协处理器可以在离线和异步模式下运行，并且如果协处理器不可用或某个缓存不认为需要额外计算，语言模型仍然可以正常运行。我们的实验结果表明，当对缓存进行增广时，解码器在后续多个令牌上的困惑度较低。此外，即使未经任何特定任务的训练，我们的实验也表明缓存增广能够一致地降低困惑度并在各种推理密集型任务中提高性能。 

---
# YuLan-Mini: An Open Data-efficient Language Model 

**Title (ZH)**: YuLan-Mini：一个开放的数据高效语言模型 

**Authors**: Yiwen Hu, Huatong Song, Jia Deng, Jiapeng Wang, Jie Chen, Kun Zhou, Yutao Zhu, Jinhao Jiang, Zican Dong, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2412.17743)  

**Abstract**: Effective pre-training of large language models (LLMs) has been challenging due to the immense resource demands and the complexity of the technical processes involved. This paper presents a detailed technical report on YuLan-Mini, a highly capable base model with 2.42B parameters that achieves top-tier performance among models of similar parameter scale. Our pre-training approach focuses on enhancing training efficacy through three key technical contributions: an elaborate data pipeline combines data cleaning with data schedule strategies, a robust optimization method to mitigate training instability, and an effective annealing approach that incorporates targeted data selection and long context training. Remarkably, YuLan-Mini, trained on 1.08T tokens, achieves performance comparable to industry-leading models that require significantly more data. To facilitate reproduction, we release the full details of the data composition for each training phase. Project details can be accessed at the following link: this https URL. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的有效预训练由于资源需求巨大和技术过程的复杂性而极具挑战性。本文详细介绍了YuLan-Mini，这是一种具有24.2亿参数的高性能基础模型，在同类参数规模的模型中表现出顶级性能。我们的预训练方法侧重于通过三大关键技术改进训练效率：一个详尽的数据管道结合了数据清理和数据调度策略，一个稳健的优化方法以减轻训练不稳定性，以及一个有效的退火方法，该方法结合了目标数据选择和长上下文训练。特别值得一提的是，YuLan-Mini在训练了1080亿个令牌后，其性能可与需要大量更多数据的行业顶尖模型相媲美。为了便于复现，我们发布了每个训练阶段的完整数据组成细节。项目详情可通过以下链接访问：this https URL。 

---
# Chumor 2.0: Towards Benchmarking Chinese Humor Understanding 

**Title (ZH)**: Chumor 2.0：面向中文幽默理解的基准测试 

**Authors**: Ruiqi He, Yushu He, Longju Bai, Jiarui Liu, Zhenjie Sun, Zenghao Tang, He Wang, Hanchen Xia, Rada Mihalcea, Naihao Deng  

**Link**: [PDF](https://arxiv.org/pdf/2412.17729)  

**Abstract**: Existing humor datasets and evaluations predominantly focus on English, leaving limited resources for culturally nuanced humor in non-English languages like Chinese. To address this gap, we construct Chumor, the first Chinese humor explanation dataset that exceeds the size of existing humor datasets. Chumor is sourced from Ruo Zhi Ba, a Chinese Reddit-like platform known for sharing intellectually challenging and culturally specific jokes. We test ten LLMs through direct and chain-of-thought prompting, revealing that Chumor poses significant challenges to existing LLMs, with their accuracy slightly above random and far below human. In addition, our analysis highlights that human-annotated humor explanations are significantly better than those generated by GPT-4o and ERNIE-4-turbo. We release Chumor at this https URL, our project page is at this https URL, our leaderboard is at this https URL, and our codebase is at this https URL. 

**Abstract (ZH)**: 现有的幽默数据集和评估主要集中在英语上，对于中文等非英语语言中的文化特异性幽默资源极为有限。为填补这一空白，我们构建了Chumor，这是第一个也是规模超过现有幽默数据集的中文幽默解释数据集。Chumor源自Ruozhi Ba，这是一个类似于Reddit的中文平台，广泛传播具有智力挑战性和文化特异性的笑话。我们通过直接和链式思考提示法测试了十种大规模语言模型（LLM），结果表明Chumor对现有LLM构成了重大挑战，它们的准确率略高于随机猜测，但远低于人类水平。此外，我们的分析还显示，手工标注的幽默解释显著优于GPT-4o和ERNIE-4-turbo生成的解释。我们在此处发布Chumor：[链接]，项目页面位于此处：[链接]，排行榜则位于此处：[链接]，而我们的代码库可以在此处找到：[链接]。 

---
# Knowledge Editing through Chain-of-Thought 

**Title (ZH)**: 通过步步推理的知识编辑 

**Authors**: Changyue Wang, Weihang Su, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.17727)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional capabilities across a wide range of natural language processing (NLP) tasks. However, keeping these models up-to-date with evolving world knowledge remains a significant challenge due to the high costs of frequent retraining. To address this challenge, knowledge editing techniques have emerged to update LLMs with new information without rebuilding the model from scratch. Among these, the in-context editing paradigm stands out for its effectiveness in integrating new knowledge while preserving the model's original capabilities. Despite its potential, existing in-context knowledge editing methods are often task-specific, focusing primarily on multi-hop QA tasks using structured knowledge triples. Moreover, their reliance on few-shot prompting for task decomposition makes them unstable and less effective in generalizing across diverse tasks.
In response to these limitations, we propose EditCoT, a novel knowledge editing framework that flexibly and efficiently updates LLMs across various tasks without retraining. EditCoT works by generating a chain-of-thought (CoT) for a given input and then iteratively refining this CoT process using a CoT editor based on updated knowledge. We evaluate EditCoT across a diverse range of benchmarks, covering multiple languages and tasks. The results demonstrate that our approach achieves state-of-the-art performance while offering superior generalization, effectiveness, and stability compared to existing methods, marking a significant advancement in the field of knowledge updating. Code and data are available at: this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）已经在广泛自然语言处理（NLP）任务中展现出了卓越的能力。然而，由于频繁重新训练的成本高昂，使得保持这些模型与不断演化的世界知识同步成为一项重大挑战。为应对这一挑战，知识编辑技术应运而生，以便无需从头重新构建模型即可更新LLMs的新信息。在这些方法中，基于上下文的编辑范式因其在集成新知识的同时保持模型原有能力的有效性而脱颖而出。尽管其潜力巨大，但现有的基于上下文的知识编辑方法往往局限于特定任务，主要聚焦于使用结构化知识三元组的多跳问答任务。此外，它们依赖于少量示例的提示来进行任务分解，使得它们在多样任务上的泛化能力较弱且效果较差。

为应对这些限制，我们提出了EditCoT，这是一种新颖的知识编辑框架，可以灵活且高效地在各种任务中更新LLMs，无需重新训练。EditCoT通过为给定输入生成链式思考（CoT）过程，然后使用基于更新知识的CoT编辑器迭代优化这个CoT过程来工作。我们跨多种基准评估了EditCoT，涵盖了多种语言和任务。实验结果表明，我们的方法在性能、泛化能力和稳定性方面均优于现有方法，标志着知识更新领域的重要进步。相关代码和数据可在以下链接获取：this <https://your-link-url.com> URL。 

---
# From Models to Microtheories: Distilling a Model's Topical Knowledge for Grounded Question Answering 

**Title (ZH)**: 从模型到微观理论：提炼模型的主题知识以实现基于语境的问答 

**Authors**: Nathaniel Weir, Bhavana Dalvi Mishra, Orion Weller, Oyvind Tafjord, Sam Hornstein, Alexander Sabol, Peter Jansen, Benjamin Van Durme, Peter Clark  

**Link**: [PDF](https://arxiv.org/pdf/2412.17701)  

**Abstract**: Recent reasoning methods (e.g., chain-of-thought, entailment reasoning) help users understand how language models (LMs) answer a single question, but they do little to reveal the LM's overall understanding, or "theory," about the question's $\textit{topic}$, making it still hard to trust the model. Our goal is to materialize such theories - here called $\textit{microtheories}$ (a linguistic analog of logical microtheories) - as a set of sentences encapsulating an LM's core knowledge about a topic. These statements systematically work together to entail answers to a $\textit{set}$ of questions to both engender trust and improve performance. Our approach is to first populate a knowledge store with (model-generated) sentences that entail answers to training questions and then distill those down to a core microtheory that is concise, general, and non-redundant. We show that, when added to a general corpus (e.g., Wikipedia), microtheories can supply critical, topical information not necessarily present in the corpus, improving both a model's ability to ground its answers to verifiable knowledge (i.e., show how answers are systematically entailed by documents in the corpus, fully grounding up to +8% more answers), and the accuracy of those grounded answers (up to +8% absolute). We also show that, in a human evaluation in the medical domain, our distilled microtheories contain a significantly higher concentration of topically critical facts than the non-distilled knowledge store. Finally, we show we can quantify the coverage of a microtheory for a topic (characterized by a dataset) using a notion of $p$-relevance. Together, these suggest that microtheories are an efficient distillation of an LM's topic-relevant knowledge, that they can usefully augment existing corpora, and can provide both performance gains and an interpretable, verifiable window into the model's knowledge of a topic. 

**Abstract (ZH)**: 近年来的推理方法（例如：链式思考、蕴含推理）有助于用户理解语言模型（LMs）如何回答单个问题，但它们对揭示模型对该问题主题的整体理解（或“理论”）帮助甚微，这使得人们仍然难以信任这些模型。我们的目标是将这些理论具象化——这里称为“微观理论”（与逻辑微观理论相对应的语义模拟），让它们可以以一组句子的形式封装模型对该主题的核心知识。这些陈述能够系统地协同工作，提供一组问题的答案，从而增强信任并提升性能。我们的方法是首先在一个知识库中填充能够蕴含训练问题答案的句子（这些句子由模型生成），然后将其提炼为一个简洁、通用且无冗余的核心微观理论。我们展示了，当将这些微观理论添加到通用语料库（例如：维基百科）中时，它们可以提供关键的、与主题相关的信息，这些信息在语料库中不一定存在。这不仅提升了模型将答案与可验证知识（即，展示答案如何系统性地由语料库中的文档推导得出，从而更完全地定位多达8%更多的答案）关联的能力，而且还提高了这些定位答案的准确性（最高可达8%的绝对值提升）。我们还展示了，在医学领域的实验中，提炼后的微观理论中包含的与主题相关的关键事实的浓度明显高于未提炼的知识库。最后，我们展示了可以使用$p$-相关性的概念来量化一个主题（由数据集表征）下的微观理论的覆盖范围。这些一并表明：微观理论是模型主题相关知识的一种高效提炼，它们可以有效增强现有的语料库，并提供性能改进和可解释、可验证的窗口，以展示模型对主题的知识。 

---
# Understanding the Logic of Direct Preference Alignment through Logic 

**Title (ZH)**: 通过逻辑理解直接偏好对齐的逻辑 

**Authors**: Kyle Richardson, Vivek Srikumar, Ashish Sabharwal  

**Link**: [PDF](https://arxiv.org/pdf/2412.17696)  

**Abstract**: Recent direct preference alignment algorithms (DPA), such as DPO, have shown great promise in aligning large language models to human preferences. While this has motivated the development of many new variants of the original DPO loss, understanding the differences between these recent proposals, as well as developing new DPA loss functions, remains difficult given the lack of a technical and conceptual framework for reasoning about the underlying semantics of these algorithms. In this paper, we attempt to remedy this by formalizing DPA losses in terms of discrete reasoning problems. Specifically, we ask: Given an existing DPA loss, can we systematically derive a symbolic expression that characterizes its semantics? How do the semantics of two losses relate to each other? We propose a novel formalism for characterizing preference losses for single model and reference model based approaches, and identify symbolic forms for a number of commonly used DPA variants. Further, we show how this formal view of preference learning sheds new light on both the size and structure of the DPA loss landscape, making it possible to not only rigorously characterize the relationships between recent loss proposals but also to systematically explore the landscape and derive new loss functions from first principles. We hope our framework and findings will help provide useful guidance to those working on human AI alignment. 

**Abstract (ZH)**: 近年来，直接偏好对齐算法（DPA），如DPO，展现出了将大型语言模型与人类偏好对齐的巨大潜力。尽管这激发了对原始DPO损失函数的新变种开发，但在缺乏用于推理这些算法内在语义的技术和概念框架的情况下，理解和区分这些最新提议仍然具有挑战性，也难以开发新的DPA损失函数。在本文中，我们尝试通过将DPA损失形式化为离散推理问题来弥补这一不足。具体而言，我们提出如下问题：给定一个现有的DPA损失，我们能否系统地推导出一个符号表达式来描述其语义？两种损失的语义之间如何相互关联？我们提出了一种新型的表征单模型和参照模型基方法偏好损失的形式化方法，并识别出了许多常用DPA变种的符号形式。此外，我们展示了这种关于偏好学习的观点如何为DPA损失景观的规模和结构带来新的见解，从而不仅能够从严格的数学角度剖析近期损失提议之间的关系，还能系统地探索景观并从第一原理出发推导新的损失函数。我们希望我们的框架和发现能够为从事人类AI对齐的人提供有用的指导。 

---
# RAGONITE: Iterative Retrieval on Induced Databases and Verbalized RDF for Conversational QA over KGs with RAG 

**Title (ZH)**: RAGONITE：基于诱导数据库和自然语言化RDF进行KG上RAG驱动的迭代检索与对话式问答

注释：该翻译旨在保持原论文标题的学术规范和准确性。其中，“RAGONITE”是专有名词，保持不变。“RAG”在此代表“Robustly Augmented Generator”，是亚马逊研究实验室开发的一种对话式问答系统架构。 

**Authors**: Rishiraj Saha Roy, Chris Hinze, Joel Schlotthauer, Farzad Naderi, Viktor Hangya, Andreas Foltyn, Luzian Hahn, Fabian Kuech  

**Link**: [PDF](https://arxiv.org/pdf/2412.17690)  

**Abstract**: Conversational question answering (ConvQA) is a convenient means of searching over RDF knowledge graphs (KGs), where a prevalent approach is to translate natural language questions to SPARQL queries. However, SPARQL has certain shortcomings: (i) it is brittle for complex intents and conversational questions, and (ii) it is not suitable for more abstract needs. Instead, we propose a novel two-pronged system where we fuse: (i) SQL-query results over a database automatically derived from the KG, and (ii) text-search results over verbalizations of KG facts. Our pipeline supports iterative retrieval: when the results of any branch are found to be unsatisfactory, the system can automatically opt for further rounds. We put everything together in a retrieval augmented generation (RAG) setup, where an LLM generates a coherent response from accumulated search results. We demonstrate the superiority of our proposed system over several baselines on a knowledge graph of BMW automobiles. 

**Abstract (ZH)**: 对话式问答（ConvQA）是搜索RDF知识图谱（KGs）的一种便捷方法，其中一种常见的方法是将自然语言问题转换为SPARQL查询。然而，SPARQL也存在一些局限性：（i）它在处理复杂意图和对话式问题时较为脆弱，（ii）它不适用于更抽象的需求。相反，我们提出了一种新的两阶段系统，该系统融合了以下两部分：（i）从知识图谱自动推导出的数据库中的SQL查询结果，以及（ii）对知识图谱事实的文本搜索结果。我们的管道支持迭代检索：当任何分支的结果不满意时，系统可以自动进行进一步的检索。我们将这一切整合到一个检索增强生成（RAG）框架中，在该框架中，大型语言模型（LLM）生成从积累的搜索结果中形成的连贯响应。我们通过对宝马汽车知识图谱的几个基准系统的演示，展示了我们提出的系统的优势。 

---
# Generating Completions for Fragmented Broca's Aphasic Sentences Using Large Language Models 

**Title (ZH)**: 使用大型语言模型生成断言性布罗卡失语症句子的完成 

**Authors**: Sijbren van Vaals, Yevgen Matusevych, Frank Tsiwah  

**Link**: [PDF](https://arxiv.org/pdf/2412.17669)  

**Abstract**: Broca's aphasia is a type of aphasia characterized by non-fluent, effortful and fragmented speech production with relatively good comprehension. Since traditional aphasia treatment methods are often time-consuming, labour-intensive, and do not reflect real-world conversations, applying natural language processing based approaches such as Large Language Models (LLMs) could potentially contribute to improving existing treatment approaches. To address this issue, we explore the use of sequence-to-sequence LLMs for completing fragmented Broca's aphasic sentences. We first generate synthetic Broca's aphasic data using a rule-based system designed to mirror the linguistic characteristics of Broca's aphasic speech. Using this synthetic data, we then fine-tune four pre-trained LLMs on the task of completing fragmented sentences. We evaluate our fine-tuned models on both synthetic and authentic Broca's aphasic data. We demonstrate LLMs' capability for reconstructing fragmented sentences, with the models showing improved performance with longer input utterances. Our result highlights the LLMs' potential in advancing communication aids for individuals with Broca's aphasia and possibly other clinical populations. 

**Abstract (ZH)**: 布罗卡失语症是一种表现为不流利、费力和片段化的言语产生，但理解能力相对较好的语言障碍类型。由于传统的失语症治疗方法往往耗时、费力，并不反映现实对话，因此应用基于自然语言处理的方法，如大型语言模型（LLMs），有可能改善现有的治疗方法。为了解决这一问题，我们探索了使用序列到序列的LLMs来完成片段化的布罗卡失语症句子。我们首先使用基于规则的系统生成合成的布罗卡失语症数据，该系统旨在模仿布罗卡失语症患者的语言特点。然后，我们使用这些合成数据微调四种预训练的LLMs，使其能够完成片段化的句子。我们分别在合成数据和真实的布罗卡失语症数据上评估了微调后的模型。我们展示了LLMs在重构片段化句子方面的能力，模型的性能随着输入句子长度的增加而改进。我们的结果突显了LLMs在促进布罗卡失语症患者及其他临床人群的交流辅助方面的能力。 

---
# LiveIdeaBench: Evaluating LLMs' Scientific Creativity and Idea Generation with Minimal Context 

**Title (ZH)**: LiveIdeaBench：使用最小背景信息评估LLM的科学创造力和概念生成 

**Authors**: Kai Ruan, Xuan Wang, Jixiang Hong, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2412.17596)  

**Abstract**: While Large Language Models (LLMs) have demonstrated remarkable capabilities in scientific tasks, existing evaluation frameworks primarily assess their performance using rich contextual inputs, overlooking their ability to generate novel ideas from minimal information. We introduce LiveIdeaBench, a comprehensive benchmark that evaluates LLMs' scientific creativity and divergent thinking capabilities using single-keyword prompts. Drawing from Guilford's creativity theory, our framework employs a dynamic panel of state-of-the-art LLMs to assess generated ideas across four key dimensions: originality, feasibility, fluency, and flexibility. Through extensive experimentation with 20 leading models across 1,180 keywords spanning 18 scientific domains, we reveal that scientific creative ability shows distinct patterns from general intelligence metrics. Notably, our results demonstrate that models like QwQ-32B-preview achieve comparable creative performance to top-tier models like o1-preview, despite significant gaps in their general intelligence scores. These findings highlight the importance of specialized evaluation frameworks for scientific creativity and suggest that the development of creative capabilities in LLMs may follow different trajectories than traditional problem-solving abilities. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）在科学任务中展示了卓越的能力，现有的评估框架主要使用丰富的上下文输入来评估它们的性能，而忽视了它们从少量信息中产生新想法的能力。我们提出了LiveIdeaBench，这是一个全面的基准，使用单关键词提示来评估LLMs的科学创造力和发散思维能力。我们的框架基于Guilford的创造力理论，利用当前最先进的LLM的动态面板，从原创性、可行性、流畅性和灵活性四个方面评估生成的想法。通过对1,180个关键词覆盖18个科学领域的20种领先模型进行广泛的实验，我们揭示了科学创造力与一般智能指标显示出不同的模式。值得注意的是，我们的结果显示，像QwQ-32B-preview这样的模型在创造力方面与顶尖模型o1-preview表现出相似的性能，尽管在一般智能评分上存在显著差距。这些发现强调了为科学创造力制定专门评估框架的重要性，并表明语言模型的创造力发展可能与传统的解决问题能力遵循不同的轨迹。 

---
# Investigating Length Issues in Document-level Machine Translation 

**Title (ZH)**: 研究文档级机器翻译中的长度问题 

**Authors**: Ziqian Peng, Rachel Bawden, François Yvon  

**Link**: [PDF](https://arxiv.org/pdf/2412.17592)  

**Abstract**: Transformer architectures are increasingly effective at processing and generating very long chunks of texts, opening new perspectives for document-level machine translation (MT). In this work, we challenge the ability of MT systems to handle texts comprising up to several thousands of tokens. We design and implement a new approach designed to precisely measure the effect of length increments on MT outputs. Our experiments with two representative architectures unambiguously show that (a)~translation performance decreases with the length of the input text; (b)~the position of sentences within the document matters and translation quality is higher for sentences occurring earlier in a document. We further show that manipulating the distribution of document lengths and of positional embeddings only marginally mitigates such problems. Our results suggest that even though document-level MT is computationally feasible, it does not yet match the performance of sentence-based MT. 

**Abstract (ZH)**: 基于Transformer的架构在处理和生成非常长的文本片段方面越来越有效，这为文档级机器翻译（MT）打开了新的前景。在这项工作中，我们挑战了MT系统处理包含数千个语令牌的文本的能力。我们设计并实现了新的方法，旨在精确测量长度增量对MT输出的影响。我们的实验使用两个代表性的架构清楚地表明：（a）随着输入文本长度的增加，翻译性能下降；（b）文档中句子的位置也很重要，文档早期出现的句子的翻译质量更高。我们进一步证明，通过调整文档长度的分布和位置嵌入的分布，只能部分缓解这些问题。我们的结果表明，虽然文档级的MT从计算角度来看是可行的，但在性能上仍未达到基于句子的MT的水平。 

---
# ERUPD -- English to Roman Urdu Parallel Dataset 

**Title (ZH)**: ERUPD -- 英语到罗马化乌尔都语平行数据集 

**Authors**: Mohammed Furqan, Raahid Bin Khaja, Rayyan Habeeb  

**Link**: [PDF](https://arxiv.org/pdf/2412.17562)  

**Abstract**: Bridging linguistic gaps fosters global growth and cultural exchange. This study addresses the challenges of Roman Urdu -- a Latin-script adaptation of Urdu widely used in digital communication -- by creating a novel parallel dataset comprising 75,146 sentence pairs. Roman Urdu's lack of standardization, phonetic variability, and code-switching with English complicates language processing. We tackled this by employing a hybrid approach that combines synthetic data generated via advanced prompt engineering with real-world conversational data from personal messaging groups. We further refined the dataset through a human evaluation phase, addressing linguistic inconsistencies and ensuring accuracy in code-switching, phonetic representations, and synonym variability. The resulting dataset captures Roman Urdu's diverse linguistic features and serves as a critical resource for machine translation, sentiment analysis, and multilingual education. 

**Abstract (ZH)**: 弥合语言差异促进全球增长和文化交流。本研究通过构建包含75,146句对的新型平行数据集，应对罗马乌尔都语（一种广泛用于数字通信的拉丁字母拼写的乌尔都语）的挑战。罗马乌尔都语缺乏标准化、发音的多变性以及与英语的代码转换，这些都给语言处理带来了复杂性。为应对这一挑战，我们采用了一种结合了通过高级提示工程生成的合成数据与来自个人聊天群组的真实对话数据的混合方法。进一步通过人类评估阶段，我们解决了语言不一致性问题，并确保代码转换、发音表示和同义词变体的准确性。最终数据集捕捉到了罗马乌尔都语的多种语言特征，并成为机器翻译、情感分析和多语教育的重要资源。 

---
# A Survey of Query Optimization in Large Language Models 

**Title (ZH)**: 大型语言模型中的查询优化概述 

**Authors**: Mingyang Song, Mao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2412.17558)  

**Abstract**: \textit{Query Optimization} (QO) refers to techniques aimed at enhancing the efficiency and quality of Large Language Models (LLMs) in understanding and answering queries, especially complex ones in scenarios like Retrieval-Augmented Generation (RAG). Specifically, RAG mitigates the limitations of LLMs by dynamically retrieving and leveraging up-to-date relevant information, which provides a cost-effective solution to the challenge of LLMs producing plausible but potentially inaccurate responses. Recently, as RAG evolves and incorporates multiple components that influence its performance, QO has emerged as a critical element, playing a pivotal role in determining the effectiveness of RAG's retrieval stage in accurately sourcing the necessary multiple pieces of evidence to answer queries correctly. In this paper, we trace the evolution of QO techniques by summarizing and analyzing significant studies. Through an organized framework and categorization, we aim to consolidate existing QO techniques in RAG, elucidate their technological foundations, and highlight their potential to enhance the versatility and applications of LLMs. 

**Abstract (ZH)**: 查询优化（Query Optimization, QO）是指旨在提升大型语言模型（Large Language Models, LLMs）理解和回答查询（尤其是复杂的查询，如检索增强生成Retrieval-Augmented Generation, RAG）效率和质量的技术。具体而言，RAG通过动态检索和利用最新的相关信息来减轻LLMs的限制，从而提供了一个成本效益高的解决方案，以应对LLMs生产看似合理但可能存在误差的回应的挑战。近年来，随着RAG的演进和多组件的引入，影响其性能的因素不断增加，QO已成为一个关键要素，对确定RAG检索阶段的有效性起着至关重要的作用，该阶段负责准确地收集必要的多个证据来正确回答查询。本文通过总结和分析相关研究，追踪QO技术的发展。借助系统化的框架和分类，我们旨在汇总现有的RAG中QO技术，阐述其技术基础，并突出这些技术在增强LLMs的灵活性和应用方面的重要潜力。 

---
# Comparative Analysis of Document-Level Embedding Methods for Similarity Scoring on Shakespeare Sonnets and Taylor Swift Lyrics 

**Title (ZH)**: 莎士比亚十四行诗与泰勒·斯威夫特歌词层面嵌入方法相似性评分的比较分析 

**Authors**: Klara Kramer  

**Link**: [PDF](https://arxiv.org/pdf/2412.17552)  

**Abstract**: This study evaluates the performance of TF-IDF weighting, averaged Word2Vec embeddings, and BERT embeddings for document similarity scoring across two contrasting textual domains. By analysing cosine similarity scores, the methods' strengths and limitations are highlighted. The findings underscore TF-IDF's reliance on lexical overlap and Word2Vec's superior semantic generalisation, particularly in cross-domain comparisons. BERT demonstrates lower performance in challenging domains, likely due to insufficient domainspecific fine-tuning. 

**Abstract (ZH)**: 本研究评估了TF-IDF加权、平均Word2Vec嵌入和BERT嵌入在两个对比文本领域中文档相似性评分的表现。通过分析余弦相似度分数，突显了这些方法的优势和局限性。研究结果强调了TF-IDF对词汇重叠的依赖性以及Word2Vec在跨领域比较中的优越语义泛化能力。相比之下，BERT在具有挑战性的领域中的表现较差，这可能归因于其领域特定微调不足。 

---
# Resource-Aware Arabic LLM Creation: Model Adaptation, Integration, and Multi-Domain Testing 

**Title (ZH)**: 资源意识驱动的阿拉伯语大规模语言模型创建：模型适配、集成与多领域测试 

**Authors**: Prakash Aryan  

**Link**: [PDF](https://arxiv.org/pdf/2412.17548)  

**Abstract**: This paper presents a novel approach to fine-tuning the Qwen2-1.5B model for Arabic language processing using Quantized Low-Rank Adaptation (QLoRA) on a system with only 4GB VRAM. We detail the process of adapting this large language model to the Arabic domain, using diverse datasets including Bactrian, OpenAssistant, and Wikipedia Arabic corpora. Our methodology involves custom data preprocessing, model configuration, and training optimization techniques such as gradient accumulation and mixed-precision training. We address specific challenges in Arabic NLP, including morphological complexity, dialectal variations, and diacritical mark handling. Experimental results over 10,000 training steps show significant performance improvements, with the final loss converging to 0.1083. We provide comprehensive analysis of GPU memory usage, training dynamics, and model evaluation across various Arabic language tasks, including text classification, question answering, and dialect identification. The fine-tuned model demonstrates robustness to input perturbations and improved handling of Arabic-specific linguistic phenomena. This research contributes to multilingual AI by demonstrating a resource-efficient approach for creating specialized language models, potentially democratizing access to advanced NLP technologies for diverse linguistic communities. Our work paves the way for future research in low-resource language adaptation and efficient fine-tuning of large language models. 

**Abstract (ZH)**: 本文提出了一个新颖的方法，利用量化低秩适应（QLoRA）在仅有4GB VRAM的系统上对Qwen2-1.5B模型进行微调，以实现阿拉伯语处理的新颖方法。我们详细介绍了将这个大规模语言模型适应阿拉伯语领域的过程，使用了包括Bactrian、OpenAssistant和阿拉伯维基百科语料库在内的多样数据集。我们的方法包括定制的数据预处理、模型配置以及梯度累积和混合精度训练等训练优化技术。我们针对阿拉伯语自然语言处理（NLP）中的特定挑战，包括形态复杂性、方言差异以及标点符号处理。实验结果显示，在超过10,000个训练步骤后，性能有显著提升，最终损失收敛于0.1083。我们详细分析了GPU内存使用情况、训练动态以及模型在不同阿拉伯语言任务中的评估，包括文本分类、问答和方言识别等。微调后的模型对输入扰动具有鲁棒性，并且更好地处理了阿拉伯语特有的语言现象。这项研究在多语言AI领域通过展示一种资源高效的特定语言模型创建方法，促进了高级NLP技术在多种语言群体中的普及。我们的研究为进一步低资源语言适应和大规模语言模型高效微调的研究奠定了基础。 

---
# Domain adapted machine translation: What does catastrophic forgetting forget and why? 

**Title (ZH)**: 领域适应机器翻译：灾难性遗忘究竟忘记了什么，又为什么会出现这种情况？ 

**Authors**: Danielle Saunders, Steve DeNeefe  

**Link**: [PDF](https://arxiv.org/pdf/2412.17537)  

**Abstract**: Neural Machine Translation (NMT) models can be specialized by domain adaptation, often involving fine-tuning on a dataset of interest. This process risks catastrophic forgetting: rapid loss of generic translation quality. Forgetting has been widely observed, with many mitigation methods proposed. However, the causes of forgetting and the relationship between forgetting and adaptation data are under-explored.
This paper takes a novel approach to understanding catastrophic forgetting during NMT adaptation by investigating the impact of the data. We provide a first investigation of what is forgotten, and why. We examine the relationship between forgetting and the in-domain data, and show that the amount and type of forgetting is linked to that data's target vocabulary coverage. Our findings pave the way toward better informed NMT domain adaptation. 

**Abstract (ZH)**: 神经机器翻译（NMT）模型可以通过领域适应进行专业化，通常涉及在感兴趣的数据集上进行微调。这一过程存在灾难性遗忘的风险：即快速丧失通用翻译质量。遗忘现象已被广泛观察到，提出了许多缓解方法。然而，遗忘的原因及其与适应数据之间的关系仍缺乏深入探讨。

本文采用一种新颖的方法来理解NMT适应过程中的灾难性遗忘，重点关注数据的影响。我们首次探讨了遗忘的内容及其原因。我们还分析了遗忘与领域内数据之间的关系，并表明遗忘的程度和类型与其目标词汇表覆盖率密切相关。我们的 findings 为更好地进行NMT领域适应提供了新的途径。 

---
# Behind Closed Words: Creating and Investigating the forePLay Annotated Dataset for Polish Erotic Discourse 

**Title (ZH)**: 《Behind Closed Words：创建和探究 forePLay 注释数据集以研究波兰语色情话语》 

**Authors**: Anna Kołos, Katarzyna Lorenc, Emilia Wiśnios, Agnieszka Karlińska  

**Link**: [PDF](https://arxiv.org/pdf/2412.17533)  

**Abstract**: The surge in online content has created an urgent demand for robust detection systems, especially in non-English contexts where current tools demonstrate significant limitations. We present forePLay, a novel Polish language dataset for erotic content detection, featuring over 24k annotated sentences with a multidimensional taxonomy encompassing ambiguity, violence, and social unacceptability dimensions. Our comprehensive evaluation demonstrates that specialized Polish language models achieve superior performance compared to multilingual alternatives, with transformer-based architectures showing particular strength in handling imbalanced categories. The dataset and accompanying analysis establish essential frameworks for developing linguistically-aware content moderation systems, while highlighting critical considerations for extending such capabilities to morphologically complex languages. 

**Abstract (ZH)**: 在线内容的激增迫切需要 robust 的检测系统，尤其是在非英语语境中，现有的工具显示出显著的局限性。我们提出了 forePLay，这是一个用于色情内容检测的新颖波兰语数据集，包含超过 24,000 个标注句子，并涵盖模糊性、暴力性和社会不接受性等多个维度的分类体系。全面的评估结果表明，专门为波兰语设计的语言模型在性能上优于多种语言模型，而基于变换器的架构在处理不平衡类别方面特别强大。该数据集及其分析为开发语言意识强的内容审核系统奠定了基础，同时指出了将此类功能扩展到形态复杂的语言时的关键考虑因素。 

---
# DiffusionAttacker: Diffusion-Driven Prompt Manipulation for LLM Jailbreak 

**Title (ZH)**: DiffusionAttacker：由扩散驱动的提示操纵以实现大语言模型脱笼攻击 

**Authors**: Hao Wang, Hao Li, Junda Zhu, Xinyuan Wang, Chengwei Pan, MinLie Huang, Lei Sha  

**Link**: [PDF](https://arxiv.org/pdf/2412.17522)  

**Abstract**: Large Language Models (LLMs) are susceptible to generating harmful content when prompted with carefully crafted inputs, a vulnerability known as LLM jailbreaking. As LLMs become more powerful, studying jailbreak methods is critical to enhancing security and aligning models with human values. Traditionally, jailbreak techniques have relied on suffix addition or prompt templates, but these methods suffer from limited attack diversity. This paper introduces DiffusionAttacker, an end-to-end generative approach for jailbreak rewriting inspired by diffusion models. Our method employs a sequence-to-sequence (seq2seq) text diffusion model as a generator, conditioning on the original prompt and guiding the denoising process with a novel attack loss. Unlike previous approaches that use autoregressive LLMs to generate jailbreak prompts, which limit the modification of already generated tokens and restrict the rewriting space, DiffusionAttacker utilizes a seq2seq diffusion model, allowing more flexible token modifications. This approach preserves the semantic content of the original prompt while producing harmful content. Additionally, we leverage the Gumbel-Softmax technique to make the sampling process from the diffusion model's output distribution differentiable, eliminating the need for iterative token search. Extensive experiments on Advbench and Harmbench demonstrate that DiffusionAttacker outperforms previous methods across various evaluation metrics, including attack success rate (ASR), fluency, and diversity. 

**Abstract (ZH)**: 大语言模型（LLMs）在受到精心设计的输入提示时，可能会生成有害内容，这种漏洞被称为LLM的逃逸。随着LLMs变得越来越强大，研究逃逸方法对于提高安全性和使模型与人类价值观保持一致变得至关重要。传统上，逃逸技术依赖于后缀添加或提示模板，但这些方法无法提供多样化的攻击方式。本论文介绍了一种名为DiffusionAttacker的端到端生成方法，该方法受到了扩散模型的启发，用于逃逸重写。我们的方法采用了一个条件生成模型——序列到序列（Seq2Seq）文本扩散模型，并通过一种新的攻击损失来引导去噪过程。与以往使用自回归LLMs生成逃逸提示的方法不同，这种方法仅限于修改生成的令牌，限制了重写的空间，而DiffusionAttacker利用Seq2Seq扩散模型，提供了更加灵活的令牌修改能力。这种方法在保持原始提示的语义内容的同时生成了有害内容。此外，我们利用Gumbel-Softmax技术使从扩散模型输出分布中采样的过程可微，从而消除了迭代令牌搜索的需要。在Advbench和Harmbench上的广泛实验表明，DiffusionAttacker在各种评估指标，包括攻击成功率（ASR）、流畅性和多样性方面，均优于先前的方法。 

---
# DRT-o1: Optimized Deep Reasoning Translation via Long Chain-of-Thought 

**Title (ZH)**: DRT-o1：通过长逻辑链优化的深度推理翻译 

**Authors**: Jiaan Wang, Fandong Meng, Yunlong Liang, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.17498)  

**Abstract**: Recently, O1-like models have emerged as representative examples, illustrating the effectiveness of long chain-of-thought (CoT) in reasoning tasks such as math and coding tasks. In this paper, we introduce DRT-o1, an attempt to bring the success of long CoT to neural machine translation (MT). Specifically, in view of the literature books that might involve similes and metaphors, translating these texts to a target language is very difficult in practice due to cultural differences. In such cases, literal translation often fails to convey the intended meaning effectively. Even for professional human translators, considerable thought must be given to preserving semantics throughout the translation process. To simulate LLMs' long thought ability in MT, we first mine sentences containing similes or metaphors from existing literature books, and then develop a multi-agent framework to translate these sentences via long thought. In the multi-agent framework, a translator is used to iteratively translate the source sentence under the suggestions provided by an advisor. To ensure the effectiveness of the long thoughts, an evaluator is also employed to judge whether the translation in the current round is better than the previous one or not. In this manner, we collect tens of thousands of long-thought MT data, which is used to train our DRT-o1. The experimental results on literature translation demonstrate the effectiveness of the DRT-o1. Using Qwen2.5-7B and Qwen2.5-14B as the backbones, the improvement brought by DRT-o1 achieves 7.33~8.26 BLEU and 1.66~3.36 CometScore. Besides, DRT-o1-7B can outperform QwQ-32B-Preview by 7.82 BLEU and 1.46 CometScore, showing its effectiveness. The project is available at this https URL 

**Abstract (ZH)**: 近年来，O1-like模型作为一种代表性的例子，展示了在数学和编程等推理任务中，长链思维（Long Chain-of-Thought, CoT）的有效性。本文旨在将长链思维的成功应用到神经机器翻译（NMT）中，我们介绍了一个名为DRT-o1的尝试。具体来说，由于文献书籍中可能包含比喻和隐喻，这些文本在目标语言中的翻译因文化差异而在实践中非常困难。在这种情况下，直接翻译往往难以有效地传达原意。即使对于专业的人类翻译者，在翻译过程中也需要付出大量努力来保留语义。为了模拟大规模语言模型（LLM）的长思考能力，我们首先从现有的文学书籍中挖掘包含比喻或隐喻的句子，然后开发了一个多代理框架，通过长链思维来翻译这些句子。在多代理框架中，翻译代理在顾问的建议下逐步翻译源句子。为了确保长链思维的有效性，我们还引入了一个评估器，用于判断当前轮次的翻译是否比上一轮次更好。通过这种方式，我们收集了十万多条长链思维的机器翻译数据，用于训练我们的DRT-o1模型。在文学翻译实验中，DRT-o1的有效性得到了验证。使用Qwen2.5-7B和Qwen2.5-14B作为骨干模型，DRT-o1带来的改进在BLEU分数上达到了7.33至8.26，在CometScore上达到了1.66至3.36的提升。此外，DRT-o1-7B在BLEU分数上比QwQ-32B-Preview高7.82，在CometScore上高1.46，显示出其有效性。该项目可在以下链接访问：[此处httpsURL] 

---
# A Silver Bullet or a Compromise for Full Attention? A Comprehensive Study of Gist Token-based Context Compression 

**Title (ZH)**: 全注意的银弹还是折中方案？基于主旨词的上下文压缩的综合研究 

**Authors**: Chenlong Deng, Zhisong Zhang, Kelong Mao, Shuaiyi Li, Xinting Huang, Dong Yu, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2412.17483)  

**Abstract**: In this work, we provide a thorough investigation of gist-based context compression methods to improve long-context processing in large language models. We focus on two key questions: (1) How well can these methods replace full attention models? and (2) What potential failure patterns arise due to compression? Through extensive experiments, we show that while gist-based compression can achieve near-lossless performance on tasks like retrieval-augmented generation and long-document QA, it faces challenges in tasks like synthetic recall. Furthermore, we identify three key failure patterns: lost by the boundary, lost if surprise, and lost along the way. To mitigate these issues, we propose two effective strategies: fine-grained autoencoding, which enhances the reconstruction of original token information, and segment-wise token importance estimation, which adjusts optimization based on token dependencies. Our work provides valuable insights into the understanding of gist token-based context compression and offers practical strategies for improving compression capabilities. 

**Abstract (ZH)**: 在本文中，我们对基于主旨的上下文压缩方法进行了全面研究，以提高大语言模型中的长上下文处理能力。我们重点关注两个关键问题：（1）这些方法能否替代全注意力模型？（2）由于压缩可能会产生何种潜在的失败模式？通过广泛的实验，我们表明，基于主旨的压缩在诸如检索增强生成和长文档问答等任务上可以实现近乎无损的性能，但在合成回忆等任务中面临挑战。此外，我们识别出三种关键失败模式：在边界处丢失信息、在意外情况下丢失信息以及在整个过程中逐步丢失信息。为了缓解这些问题，我们提出了两种有效的策略：细粒度自编码，该方法增强对原始标记信息的重构，并且基于标记间的依赖关系调整优化。我们的研究为理解基于主旨的上下文压缩提供了宝贵的见解，并提供了提高压缩能力的实际策略。 

---
# A Survey on Multi-Generative Agent System: Recent Advances and New Frontiers 

**Title (ZH)**: 多生成性智能体系统综述：近期进展与新前沿 

**Authors**: Shuaihang Chen, Yuanxing Liu, Wei Han, Weinan Zhang, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.17481)  

**Abstract**: Multi-generative agent systems (MGASs) have become a research hotspot since the rise of large language models (LLMs). However, with the continuous influx of new related works, the existing reviews struggle to capture them comprehensively. This paper presents a comprehensive survey of these studies. We first discuss the definition of MGAS, a framework encompassing much of previous work. We provide an overview of the various applications of MGAS in (i) solving complex tasks, (ii) simulating specific scenarios, and (iii) evaluating generative agents. Building on previous studies, we also highlight several challenges and propose future directions for research in this field. 

**Abstract (ZH)**: 多生成代理系统（MGAS）自大型语言模型（LLMs）的兴起以来已成为研究热点。然而，随着新相关工作的不断涌现，现有综述难以全面涵盖它们。本文对这些研究进行了全面综述。我们首先讨论了MGAS的定义，并提出一个框架，涵盖了许多之前的工作。我们概述了MGAS在以下几方面的各种应用：（i）解决复杂任务，（ii）模拟特定场景，（iii）评估生成代理。基于先前的研究，我们还指出了几个挑战，并提出了未来研究方向的建议。 

---
# Diving into Self-Evolving Training for Multimodal Reasoning 

**Title (ZH)**: 深入探究自进化训练在多模态推理中的应用 

**Authors**: Wei Liu, Junlong Li, Xiwen Zhang, Fan Zhou, Yu Cheng, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2412.17451)  

**Abstract**: Reasoning ability is essential for Large Multimodal Models (LMMs). In the absence of multimodal chain-of-thought annotated data, self-evolving training, where the model learns from its own outputs, has emerged as an effective and scalable approach for enhancing reasoning abilities. Despite its growing usage, a comprehensive understanding of self-evolving training, particularly in the context of multimodal reasoning, remains limited. In this paper, we delve into the intricacies of self-evolving training for multimodal reasoning, pinpointing three key factors: Training Method, Reward Model, and Prompt Variation. We systematically examine each factor and explore how various configurations affect the training's effectiveness. Our analysis leads to a set of best practices for each factor, aimed at optimizing multimodal reasoning. Furthermore, we explore the Self-Evolution Dynamics during training and the impact of automatic balancing mechanisms in boosting performance. After all the investigations, we present a final recipe for self-evolving training in multimodal reasoning, encapsulating these design choices into a framework we call MSTaR (Multimodal Self-evolving Training for Reasoning), which is universally effective for models with different sizes on various benchmarks, e.g., surpassing the pre-evolved model significantly on 5 multimodal reasoning benchmarks without using additional human annotations, as demonstrated on MiniCPM-V-2.5 (8B), Phi-3.5-Vision (4B) and InternVL2 (2B). We believe this study fills a significant gap in the understanding of self-evolving training for multimodal reasoning and offers a robust framework for future research. Our policy and reward models, as well as the collected data, is released to facilitate further investigation in multimodal reasoning. 

**Abstract (ZH)**: 逻辑推理能力对于大型多模态模型（LMMs）至关重要。在缺乏多模态链式思维标注数据的情况下，自我演进训练，即模型从自身输出中学习，已成为提升逻辑推理能力的有效且可扩展的方法。尽管该方法的使用在不断增加，但在多模态推理的具体背景下，对其全面的理解仍然有限。本文深入探讨了自我演进训练在多模态推理中的复杂性，指出了三个关键因素：训练方法、奖励模型和提示变化。我们系统地分析了每个因素，并探讨了不同配置如何影响训练的有效性。我们的分析得出了一套针对每个因素的最佳实践，旨在优化多模态推理。此外，我们还研究了自我演进训练中的动态过程以及自动平衡机制对提升性能的影响。经过所有调查后，我们提出了多模态推理中自我演进训练的最终配方，将这些设计选择汇总成一个名为MSTaR（多模态自我演进训练推理）的框架，该框架适用于不同规模的模型在多种基准测试中的表现，例如，在MiniCPM-V-2.5（8B）、Phi-3.5-Vision（4B）和InternVL2（2B）等5个多模态推理基准测试中，显著超过了预演进的模型，而无需额外的人工标注。我们认为，本研究填补了多模态推理中自我演进训练理解的空白，并为未来研究提供了稳健的框架。我们的政策和奖励模型以及收集的数据已发布，以促进进一步在多模态推理方面的研究。 

---
# Measuring Contextual Informativeness in Child-Directed Text 

**Title (ZH)**: 衡量面向儿童文本的情境 informativeness 

**Authors**: Maria Valentini, Téa Wright, Ali Marashian, Jennifer Weber, Eliana Colunga, Katharina von der Wense  

**Link**: [PDF](https://arxiv.org/pdf/2412.17427)  

**Abstract**: To address an important gap in creating children's stories for vocabulary enrichment, we investigate the automatic evaluation of how well stories convey the semantics of target vocabulary words, a task with substantial implications for generating educational content. We motivate this task, which we call measuring contextual informativeness in children's stories, and provide a formal task definition as well as a dataset for the task. We further propose a method for automating the task using a large language model (LLM). Our experiments show that our approach reaches a Spearman correlation of 0.4983 with human judgments of informativeness, while the strongest baseline only obtains a correlation of 0.3534. An additional analysis shows that the LLM-based approach is able to generalize to measuring contextual informativeness in adult-directed text, on which it also outperforms all baselines. 

**Abstract (ZH)**: 为解决为词汇丰富而创建儿童故事中的一个重要空白，我们研究了自动评估故事传达目标词汇意义效果的方法，这一任务对生成教育内容具有重大意义。我们阐述了这一任务的重要性，即衡量儿童故事中的上下文信息量，并提出了该任务的正式定义以及相关数据集。进一步地，我们提出了一种使用大规模语言模型（LLM）自动化执行该任务的方法。我们的实验结果显示，我们的方法与人类对信息量的判断之间的 Spearman 相关系数达到了 0.4983，而最强的基线方法只有 0.3534。此外的分析表明，基于大规模语言模型的方法能够泛化到成人导向文本的上下文信息量衡量，并在该任务上也优于所有基线方法。 

---
# Just What You Desire: Constrained Timeline Summarization with Self-Reflection for Enhanced Relevance 

**Title (ZH)**: 恰如您的所愿：自省约束时间线摘要以提升相关性 

**Authors**: Muhammad Reza Qorib, Qisheng Hu, Hwee Tou Ng  

**Link**: [PDF](https://arxiv.org/pdf/2412.17408)  

**Abstract**: Given news articles about an entity, such as a public figure or organization, timeline summarization (TLS) involves generating a timeline that summarizes the key events about the entity. However, the TLS task is too underspecified, since what is of interest to each reader may vary, and hence there is not a single ideal or optimal timeline. In this paper, we introduce a novel task, called Constrained Timeline Summarization (CTLS), where a timeline is generated in which all events in the timeline meet some constraint. An example of a constrained timeline concerns the legal battles of Tiger Woods, where only events related to his legal problems are selected to appear in the timeline. We collected a new human-verified dataset of constrained timelines involving 47 entities and 5 constraints per entity. We propose an approach that employs a large language model (LLM) to summarize news articles according to a specified constraint and cluster them to identify key events to include in a constrained timeline. In addition, we propose a novel self-reflection method during summary generation, demonstrating that this approach successfully leads to improved performance. 

**Abstract (ZH)**: 给定关于某一实体（如公众人物或组织）的新闻文章，时间线总结（TLS）涉及生成一个能够概括该实体关键事件的时间线。然而，TLS任务过于模糊，因为不同读者感兴趣的事件可能不同，因此并不存在一个理想的或最优的时间线。本文我们引入了一个新的任务，称为约束时间线总结（CTLS），其中生成的时间线中的所有事件都要满足某些约束条件。例如，在泰格·伍兹的法律斗争案例中，仅与其法律问题相关的事件被选入时间线。我们收集了一个新的由人工验证的数据集，其中包括47个实体和每个实体5个约束条件。我们提出了一种方法，利用大型语言模型（LLM）根据指定的约束条件总结新闻文章，并对其进行聚类以识别要包含在约束时间线中的关键事件。此外，我们在总结生成过程中提出了一个新颖的自省方法，证明该方法能够显著提高性能。 

---
# WarriorCoder: Learning from Expert Battles to Augment Code Large Language Models 

**Title (ZH)**: 战士程序员：从专家对决中学习以增强代码大型语言模型 

**Authors**: Huawen Feng, Pu Zhao, Qingfeng Sun, Can Xu, Fangkai Yang, Lu Wang, Qianli Ma, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17395)  

**Abstract**: Despite recent progress achieved by code large language models (LLMs), their remarkable abilities are largely dependent on fine-tuning on the high-quality data, posing challenges for data collection and annotation. To address this, current methods often design various data flywheels to gather complex code instructions, enabling models to handle more intricate tasks. However, these approaches typically rely on off-the-shelf datasets and data augmentation from the limited pool of proprietary LLMs (e.g., Claude, GPT4, and so on), which limits the diversity of the constructed data and makes it prone to systemic biases. In this paper, we propose WarriorCoder which learns from expert battles to address these limitations. Specifically, we create an arena for current expert code LLMs, where each model challenges and responds to others' challenges, with evaluations conducted by uninvolved judge models. This competitive framework generates novel training data constructed from scratch, harnessing the strengths of all participants. Experimental results demonstrate that WarriorCoder achieves competitive performance compared to previous methods, even without relying on proprietary LLMs. 

**Abstract (ZH)**: 尽管近期代码大规模语言模型（LLMs）取得了进展，它们的卓越能力很大程度上依赖于在高质量数据上进行微调，这给数据收集和标注带来了挑战。为了应对这一问题，当前的方法常常设计各种数据飞轮以采集复杂的代码指令，从而使模型能够处理更加复杂的任务。然而，这些方法通常依赖于现成的数据集和有限的专有LLM（如Claude、GPT4等）的数据增强，这限制了构建数据的多样性，并使其容易受到系统性偏见的影响。在本文中，我们提出了一种名为WarriorCoder的方法，旨在通过专家战斗来解决这些局限性。具体而言，我们创建了一个供当前代码LLM专家们竞技的擂台，在这个擂台上，每个模型挑战其他模型并回应挑战，且这些评估由未参与的裁判模型进行。这种竞争框架生成了从头构建的新颖训练数据，利用了所有参与者的优点。实验结果表明，WarriorCoder在不依赖于专有LLM的情况下，能够达到与先前方法相当的性能。 

---
# Interweaving Memories of a Siamese Large Language Model 

**Title (ZH)**: 泰国大型语言模型的记忆交织 

**Authors**: Xin Song, Zhikai Xue, Guoxiu He, Jiawei Liu, Wei Lu  

**Link**: [PDF](https://arxiv.org/pdf/2412.17383)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) methods optimize large language models (LLMs) by modifying or introducing a small number of parameters to enhance alignment with downstream tasks. However, they can result in catastrophic forgetting, where LLMs prioritize new knowledge at the expense of comprehensive world knowledge. A promising approach to mitigate this issue is to recall prior memories based on the original knowledge. To this end, we propose a model-agnostic PEFT framework, IMSM, which Interweaves Memories of a Siamese Large Language Model. Specifically, our siamese LLM is equipped with an existing PEFT method. Given an incoming query, it generates two distinct memories based on the pre-trained and fine-tuned parameters. IMSM then incorporates an interweaving mechanism that regulates the contributions of both original and enhanced memories when generating the next token. This framework is theoretically applicable to all open-source LLMs and existing PEFT methods. We conduct extensive experiments across various benchmark datasets, evaluating the performance of popular open-source LLMs using the proposed IMSM, in comparison to both classical and leading PEFT methods. Our findings indicate that IMSM maintains comparable time and space efficiency to backbone PEFT methods while significantly improving performance and effectively mitigating catastrophic forgetting. 

**Abstract (ZH)**: 参数高效的微调（PEFT）方法通过修改或引入少量参数来优化大型语言模型（LLMs），以增强其与下游任务的一致性。然而，这种方法可能会导致灾难性遗忘，即LLMs在获取新知识的同时牺牲了全面的世界知识。缓解这一问题的一个有前景的方法是基于原始知识重新激活先前的记忆。为了解决这个问题，我们提出了一种模型无关的PEFT框架，即Interweaving Memories of a Siamese Large Language Model（ISMSM），它将双胞胎大型语言模型中的记忆相互交织。具体而言，我们的双胞胎LLM配备了现有的PEFT方法。对于每个新的查询，它根据预训练和微调参数生成两个不同的记忆。ISMSM则引入了一个交织机制，该机制在生成下一个token时调节原始记忆和增强记忆的贡献。该框架理论上适用于所有开源LLM和现有的PEFT方法。我们在多个基准数据集上进行了广泛的实验，使用所提出的ISMSM评估了流行开源LLM的表现，并将其与经典的和领先的PEFT方法进行了比较。研究结果显示，ISMSM在保持与主PEFT方法类似的时间和空间效率的同时，显著改善了性能并有效缓解了灾难性遗忘问题。 

---
# Boosting LLM via Learning from Data Iteratively and Selectively 

**Title (ZH)**: 通过迭代选择性地从数据中学习增强大语言模型 

**Authors**: Qi Jia, Siyu Ren, Ziheng Qin, Fuzhao Xue, Jinjie Ni, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2412.17365)  

**Abstract**: Datasets nowadays are generally constructed from multiple sources and using different synthetic techniques, making data de-noising and de-duplication crucial before being used for post-training. In this work, we propose to perform instruction tuning by iterative data selection (\ApproachName{}). We measure the quality of a sample from complexity and diversity simultaneously. Instead of calculating the complexity score once for all before fine-tuning, we highlight the importance of updating this model-specific score during fine-tuning to accurately accommodate the dynamic changes of the model. On the other hand, the diversity score is defined on top of the samples' responses under the consideration of their informativeness. IterIT integrates the strengths of both worlds by iteratively updating the complexity score for the top-ranked samples and greedily selecting the ones with the highest complexity-diversity score. Experiments on multiple instruction-tuning data demonstrate consistent improvements of IterIT over strong baselines. Moreover, our approach also generalizes well to domain-specific scenarios and different backbone models. All resources will be available at this https URL. 

**Abstract (ZH)**: 当前的的数据集通常来源于多个数据源并使用不同的合成技术构建，因此在进行模型训练后需要先进行数据去噪和去重处理。本文提出了一种通过迭代数据选择来进行指令调优的方法（\ApproachName{}）。我们同时从复杂性与多样性两个方面衡量样本的质量。不同于在微调前一次性计算所有样本的复杂性分数，我们在微调过程中突出更新这种模型特异性分数的重要性，以准确适应模型的动态变化。另一方面，多样性分数定义在考虑样本响应信息量的基础上。IterIT 通过迭代更新排名靠前样本的复杂性分数，并贪婪选择具有最高复杂性-多样性分数的样本，从而整合了两方面的优点。实验结果表明，IterIT 在多种指令调优数据上一致优于强基线方法。此外，该方法也能够在特定领域场景和不同的骨干模型中良好泛化。所有资源请访问 <https://your-resource-url-here.com>。 

---
# An Experimental Evaluation of Japanese Tokenizers for Sentiment-Based Text Classification 

**Title (ZH)**: 基于情感分类的日语分词器实验评估 

**Authors**: Andre Rusli, Makoto Shishido  

**Link**: [PDF](https://arxiv.org/pdf/2412.17361)  

**Abstract**: This study investigates the performance of three popular tokenization tools: MeCab, Sudachi, and SentencePiece, when applied as a preprocessing step for sentiment-based text classification of Japanese texts. Using Term Frequency-Inverse Document Frequency (TF-IDF) vectorization, we evaluate two traditional machine learning classifiers: Multinomial Naive Bayes and Logistic Regression. The results reveal that Sudachi produces tokens closely aligned with dictionary definitions, while MeCab and SentencePiece demonstrate faster processing speeds. The combination of SentencePiece, TF-IDF, and Logistic Regression outperforms the other alternatives in terms of classification performance. 

**Abstract (ZH)**: 本研究探讨了三种流行的分词工具（MeCab、Sudachi和SentencePiece）在情绪分类文本分类中的预处理表现，特别是应用于日文文本的情绪分类。通过使用词频-逆文档频率（TF-IDF）向量化方法，我们评估了两种传统机器学习分类器：多项式朴素贝叶斯 classifier 和逻辑回归 classifier。研究结果表明，Sudachi 生成的标记与词典定义更为接近，而 MeCab 和 SentencePiece 则显示出了更快的处理速度。SentencePiece、TF-IDF 和逻辑回归的结合在分类性能上优于其他选项。 

---
# Three-Class Text Sentiment Analysis Based on LSTM 

**Title (ZH)**: 基于LSTM的三分类文本情感分析 

**Authors**: Yin Qixuan  

**Link**: [PDF](https://arxiv.org/pdf/2412.17347)  

**Abstract**: Sentiment analysis is a crucial task in natural language processing (NLP) with applications in public opinion monitoring, market research, and beyond. This paper introduces a three-class sentiment classification method for Weibo comments using Long Short-Term Memory (LSTM) networks to discern positive, neutral, and negative sentiments. LSTM, as a deep learning model, excels at capturing long-distance dependencies in text data, providing significant advantages over traditional machine learning approaches. Through preprocessing and feature extraction from Weibo comment texts, our LSTM model achieves precise sentiment prediction. Experimental results demonstrate superior performance, achieving an accuracy of 98.31% and an F1 score of 98.28%, notably outperforming conventional models and other deep learning methods. This underscores the effectiveness of LSTM in capturing nuanced sentiment information within text, thereby enhancing classification accuracy. Despite its strengths, the LSTM model faces challenges such as high computational complexity and slower processing times for lengthy texts. Moreover, complex emotional expressions like sarcasm and humor pose additional difficulties. Future work could explore combining pre-trained models or advancing feature engineering techniques to further improve both accuracy and practicality. Overall, this study provides an effective solution for sentiment analysis on Weibo comments. 

**Abstract (ZH)**: 情感分析是自然语言处理（NLP）中的重要任务，广泛应用于公众意见监测、市场研究等领域。本文介绍了一种基于长短期记忆（LSTM）网络的三分类情感分类方法，用于微博评论，以区分正面、中性和负面情感。LSTM 作为一种深度学习模型，擅长捕捉文本数据中的长距离依赖关系，相对于传统机器学习方法具有显著优势。通过对微博评论文本进行预处理和特征提取，我们的LSTM模型实现了精准的情感预测。实验结果表明，该模型性能优越，准确率达到98.31%，F1分数为98.28%，明显优于传统模型和其他深度学习方法。这凸显了LSTM在捕捉文本中的细微情感信息方面的有效性，从而提高了分类准确性。尽管LSTM具有许多优势，但也面临着诸如高计算复杂度和较长文本处理时间等挑战。此外，复杂的语气表达方式如反讽和幽默也为模型带来了额外的困难。未来的研究可以探索结合预训练模型或改进特征工程技术，以进一步提高准确性和实用性。总体而言，本研究为微博评论的情感分析提供了一种有效的方法。 

---
# A Dual-Perspective Metaphor Detection Framework Using Large Language Models 

**Title (ZH)**: 使用大规模语言模型的双重视角隐喻检测框架 

**Authors**: Yujie Lin, Jingyao Liu, Yan Gao, Ante Wang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2412.17332)  

**Abstract**: Metaphor detection, a critical task in natural language processing, involves identifying whether a particular word in a sentence is used metaphorically. Traditional approaches often rely on supervised learning models that implicitly encode semantic relationships based on metaphor theories. However, these methods often suffer from a lack of transparency in their decision-making processes, which undermines the reliability of their predictions. Recent research indicates that LLMs (large language models) exhibit significant potential in metaphor detection. Nevertheless, their reasoning capabilities are constrained by predefined knowledge graphs. To overcome these limitations, we propose DMD, a novel dual-perspective framework that harnesses both implicit and explicit applications of metaphor theories to guide LLMs in metaphor detection and adopts a self-judgment mechanism to validate the responses from the aforementioned forms of guidance. In comparison to previous methods, our framework offers more transparent reasoning processes and delivers more reliable predictions. Experimental results prove the effectiveness of DMD, demonstrating state-of-the-art performance across widely-used datasets. 

**Abstract (ZH)**: 元喻检测是自然语言处理中的一项关键任务，涉及识别句子中的某个词语是否被用于隐喻。传统方法通常依赖于基于隐喻理论的监督学习模型，隐式地编码语义关系。然而，这些方法往往在决策过程的透明性方面存在不足，这削弱了其预测的可靠性。近期研究表明，大规模语言模型（LLMs）在隐喻检测方面具有显著潜力。然而，它们的推理能力受限于预定义的知识图谱。为克服这些限制，我们提出了一种名为DMD的新型双视角框架，该框架结合了隐喻理论的显性和隐性应用，以指导LLMs进行隐喻检测，并采用自我判断机制来验证上述指导方式的响应。与以往方法相比，该框架提供了更透明的推理过程和更可靠的预测。实验结果证明了DMD的有效性，其在广泛使用的数据集上展示了最先进的性能。 

---
# Assessing Human Editing Effort on LLM-Generated Texts via Compression-Based Edit Distance 

**Title (ZH)**: 基于压缩距离的评估人类对LLM生成文本的编辑努力程度 

**Authors**: Nicolas Devatine, Louis Abraham  

**Link**: [PDF](https://arxiv.org/pdf/2412.17321)  

**Abstract**: Assessing the extent of human edits on texts generated by Large Language Models (LLMs) is crucial to understanding the human-AI interactions and improving the quality of automated text generation systems. Existing edit distance metrics, such as Levenshtein, BLEU, ROUGE, and TER, often fail to accurately measure the effort required for post-editing, especially when edits involve substantial modifications, such as block operations. In this paper, we introduce a novel compression-based edit distance metric grounded in the Lempel-Ziv-77 algorithm, designed to quantify the amount of post-editing applied to LLM-generated texts. Our method leverages the properties of text compression to measure the informational difference between the original and edited texts. Through experiments on real-world human edits datasets, we demonstrate that our proposed metric is highly correlated with actual edit time and effort. We also show that LLMs exhibit an implicit understanding of editing speed, that aligns well with our metric. Furthermore, we compare our metric with existing ones, highlighting its advantages in capturing complex edits with linear computational efficiency. Our code and data are available at: this https URL 

**Abstract (ZH)**: 评估大型语言模型（LLMs）生成文本中人类编辑的程度对于理解人机交互和提高自动化文本生成系统的质量至关重要。现有的编辑距离度量方法，如Levenshtein、BLEU、ROUGE和TER，通常无法准确测量后续编辑所需的工作量，尤其是在涉及大量修改，例如块操作时。本文介绍了一种基于Lempel-Ziv-77算法的新型压缩编辑距离度量方法，旨在量化对LLM生成文本进行编辑的数量。我们的方法利用文本压缩的特性来衡量原始文本和编辑文本之间的信息差异。通过对实际人类编辑数据集的实验，我们证明了我们提出的度量方法与实际编辑时间和工作量高度相关。我们还表明，LLMs表现出了与我们的度量方法一致的隐含编辑速度理解能力。此外，我们将我们的度量方法与其他方法进行比较，突显了其在捕捉复杂编辑时的线性计算效率优势。我们的代码和数据可在以下链接获得：[这里填入具体的URL] 

---
# Friends-MMC: A Dataset for Multi-modal Multi-party Conversation Understanding 

**Title (ZH)**: Friends-MMC：一个用于多模态多人群体对话理解的数据集 

**Authors**: Yueqian Wang, Xiaojun Meng, Yuxuan Wang, Jianxin Liang, Qun Liu, Dongyan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2412.17295)  

**Abstract**: Multi-modal multi-party conversation (MMC) is a less studied yet important topic of research due to that it well fits real-world scenarios and thus potentially has more widely-used applications. Compared with the traditional multi-modal conversations, MMC requires stronger character-centered understanding abilities as there are many interlocutors appearing in both the visual and textual context. To facilitate the study of this problem, we present Friends-MMC in this paper, an MMC dataset that contains 24,000+ unique utterances paired with video context. To explore the character-centered understanding of the dialogue, we also annotate the speaker of each utterance, the names and bounding bboxes of faces that appear in the video. Based on this Friends-MMC dataset, we further study two fundamental MMC tasks: conversation speaker identification and conversation response prediction, both of which have the multi-party nature with the video or image as visual context. For conversation speaker identification, we demonstrate the inefficiencies of existing methods such as pre-trained models, and propose a simple yet effective baseline method that leverages an optimization solver to utilize the context of two modalities to achieve better performance. For conversation response prediction, we fine-tune generative dialogue models on Friend-MMC, and analyze the benefits of speaker information. The code and dataset is publicly available at this https URL and thus we call for more attention on modeling speaker information when understanding conversations. 

**Abstract (ZH)**: 多模态多党派对话（MMC）是一个较少研究但极具重要性的研究课题，因为它能够很好地适应现实世界的情景，从而具有更为广泛的应用前景。与传统的多模态对话相比，MMC 需要更强的以角色为中心的理解能力，因为在视觉和文本上下文中出现了多个对话者。为了促进对该问题的研究，本文提出了一个包含超过 24,000 个独特话语的 Friends-MMC 数据集，这些话语与视频背景相匹配。为了探讨对话中的角色中心理解，我们还标注了每条话语的发言者、视频中出现的人物姓名及其面部边界框。基于这个 Friends-MMC 数据集，我们进一步研究了两个基本的 MMC 任务：对话发言者识别和对话响应预测，这两个任务都具有多当事人特性和以视频或图像为背景的视觉上下文。对于对话发言者识别，我们展示了现有方法（如预训练模型）的不足之处，并提出了一种简单但有效的基线方法，该方法利用优化求解器结合两种模态的上下文以提升性能。对于对话响应预测，我们对 Friends-MMC 数据集进行了生成对话模型的微调，并分析了发言者信息的好处。代码和数据集已公开发布，因此我们呼吁在理解对话时更多关注发言者信息的建模。 

---
# Learning from Mistakes: Self-correct Adversarial Training for Chinese Unnatural Text Correction 

**Title (ZH)**: 从错误中学习：中文 unnatural 文本自我纠正对抗训练 

**Authors**: Xuan Feng, Tianlong Gu, Xiaoli Liu, Liang Chang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17279)  

**Abstract**: Unnatural text correction aims to automatically detect and correct spelling errors or adversarial perturbation errors in sentences. Existing methods typically rely on fine-tuning or adversarial training to correct errors, which have achieved significant success. However, these methods exhibit poor generalization performance due to the difference in data distribution between training data and real-world scenarios, known as the exposure bias problem. In this paper, we propose a self-correct adversarial training framework for \textbf{L}earn\textbf{I}ng from \textbf{MI}s\textbf{T}akes (\textbf{LIMIT}), which is a task- and model-independent framework to correct unnatural errors or mistakes. Specifically, we fully utilize errors generated by the model that are actively exposed during the inference phase, i.e., predictions that are inconsistent with the target. This training method not only simulates potential errors in real application scenarios, but also mitigates the exposure bias of the traditional training process. Meanwhile, we design a novel decoding intervention strategy to maintain semantic consistency. Extensive experimental results on Chinese unnatural text error correction datasets show that our proposed method can correct multiple forms of errors and outperforms the state-of-the-art text correction methods. In addition, extensive results on Chinese and English datasets validate that LIMIT can serve as a plug-and-play defense module and can extend to new models and datasets without further training. 

**Abstract (ZH)**: 非自然文本修正旨在自动检测和修正句子中的拼写错误或对抗性扰动错误。现有方法通常依赖于微调或对抗性训练来进行错误修正，这些方法已经取得了显著的成功。然而，这些方法由于训练数据与实际场景之间的数据分布差异而导致泛化性能较差，这种现象被称为暴露偏差问题。在本文中，我们提出了一种名为**L**earn**I**ng from**MI**stakes**LIMIT**的自我修正对抗训练框架，该框架是一个独立于任务和模型的框架，用于修正非自然错误或错误。具体而言，我们充分利用模型在推理阶段主动暴露的错误，即与目标不一致的预测。这种训练方法不仅模拟了实际应用场景中的潜在错误，还减轻了传统训练过程中的暴露偏差问题。同时，我们设计了一种新的解码干预策略以保持语义一致性。在多个中文非自然文本错误修正数据集上的广泛实验结果显示，我们提出的方法能够修正多种类型的错误，并且优于现有的先进文本修正方法。此外，在中文和英文数据集上的广泛实验结果验证了LIMIT可以作为即插即用的防御模块，并且无需进一步训练即可应用于新的模型和数据集。 

---
# LegalAgentBench: Evaluating LLM Agents in Legal Domain 

**Title (ZH)**: LegalAgentBench：评估法律领域中的语言模型代理 

**Authors**: Haitao Li, Junjie Chen, Jingli Yang, Qingyao Ai, Wei Jia, Youfeng Liu, Kai Lin, Yueyue Wu, Guozhi Yuan, Yiran Hu, Wuyue Wang, Yiqun Liu, Minlie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17259)  

**Abstract**: With the increasing intelligence and autonomy of LLM agents, their potential applications in the legal domain are becoming increasingly apparent. However, existing general-domain benchmarks cannot fully capture the complexity and subtle nuances of real-world judicial cognition and decision-making. Therefore, we propose LegalAgentBench, a comprehensive benchmark specifically designed to evaluate LLM Agents in the Chinese legal domain. LegalAgentBench includes 17 corpora from real-world legal scenarios and provides 37 tools for interacting with external knowledge. We designed a scalable task construction framework and carefully annotated 300 tasks. These tasks span various types, including multi-hop reasoning and writing, and range across different difficulty levels, effectively reflecting the complexity of real-world legal scenarios. Moreover, beyond evaluating final success, LegalAgentBench incorporates keyword analysis during intermediate processes to calculate progress rates, enabling more fine-grained evaluation. We evaluated eight popular LLMs, highlighting the strengths, limitations, and potential areas for improvement of existing models and methods. LegalAgentBench sets a new benchmark for the practical application of LLMs in the legal domain, with its code and data available at \url{this https URL}. 

**Abstract (ZH)**: 随着大型语言模型（LLM）代理的智能和自主性的增强，它们在法律领域的潜在应用变得愈发明显。然而，现有的通用领域基准无法全面捕捉到现实司法认知和决策中的复杂性和细微差别。因此，我们提出了一个专门针对中国法律领域的基准测试，即LegalAgentBench。LegalAgentBench 包括来自真实法律情境的17个语料库，并提供了37种与外部知识交互的工具。我们设计了一个可扩展的任务构建框架，并仔细标注了300个任务。这些任务涵盖了多种类型，包括多步推理和写作，并涵盖了不同的难度级别，有效地反映了现实法律情境的复杂性。此外，与仅仅评价最终的成功不同，LegalAgentBench 在中间过程中也进行了关键词分析以计算进度率，从而实现更为精细的评价。我们评估了八个流行的LLM，突显了现有模型和方法的优点、局限性和改进潜力。LegalAgentBench 为LLM在法律领域的实践应用设立了新的基准，其代码和数据可在 \url{this https URL} 获取。 

---
# Unlocking Cross-Lingual Sentiment Analysis through Emoji Interpretation: A Multimodal Generative AI Approach 

**Title (ZH)**: 通过表情符号解释实现跨语言情感分析——一种多模态生成AI方法 

**Authors**: Rafid Ishrak Jahan, Heng Fan, Haihua Chen, Yunhe Feng  

**Link**: [PDF](https://arxiv.org/pdf/2412.17255)  

**Abstract**: Emojis have become ubiquitous in online communication, serving as a universal medium to convey emotions and decorative elements. Their widespread use transcends language and cultural barriers, enhancing understanding and fostering more inclusive interactions. While existing work gained valuable insight into emojis understanding, exploring emojis' capability to serve as a universal sentiment indicator leveraging large language models (LLMs) has not been thoroughly examined. Our study aims to investigate the capacity of emojis to serve as reliable sentiment markers through LLMs across languages and cultures. We leveraged the multimodal capabilities of ChatGPT to explore the sentiments of various representations of emojis and evaluated how well emoji-conveyed sentiment aligned with text sentiment on a multi-lingual dataset collected from 32 countries. Our analysis reveals that the accuracy of LLM-based emoji-conveyed sentiment is 81.43%, underscoring emojis' significant potential to serve as a universal sentiment marker. We also found a consistent trend that the accuracy of sentiment conveyed by emojis increased as the number of emojis grew in text. The results reinforce the potential of emojis to serve as global sentiment indicators, offering insight into fields such as cross-lingual and cross-cultural sentiment analysis on social media platforms. Code: this https URL. 

**Abstract (ZH)**: 表情符号已成为在线交流中无处不在的工具，用以传达情绪和作为装饰性元素。它们的广泛应用超越了语言和文化障碍，增强了理解并促进了更为包容的交流。虽然现有研究揭示了表情符号的理解价值，但利用大规模语言模型（LLMs）探索表情符号作为全球情绪指标的能力尚未得到充分研究。我们的研究旨在通过LLMs跨语言和跨文化地考察表情符号作为可靠情绪标记的能力。我们利用ChatGPT的多模态能力探讨了各种表情符号表示形式的情绪，评估了表情符号所传达的情绪与多语言数据集中文本情绪的一致性。我们的分析显示，基于LLM的表情符号所传达情绪的准确率为81.43%，突显了表情符号作为全球情绪指标的巨大潜力。我们还发现，随着文本中表情符号数量的增加，表情符号所传达情绪的准确性呈现出一致的上升趋势。这些结果强化了表情符号作为全球情绪指标的潜力，对社交媒体平台上的跨语言和跨文化情绪分析领域提供了有价值的见解。代码：[这里提供代码链接]。 

---
# Brain-to-Text Benchmark '24: Lessons Learned 

**Title (ZH)**: Brain-to-Text基准测试'24：经验教训 

**Authors**: Francis R. Willett, Jingyuan Li, Trung Le, Chaofei Fan, Mingfei Chen, Eli Shlizerman, Yue Chen, Xin Zheng, Tatsuo S. Okubo, Tyler Benster, Hyun Dong Lee, Maxwell Kounga, E. Kelly Buchanan, David Zoltowski, Scott W. Linderman, Jaimie M. Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2412.17227)  

**Abstract**: Speech brain-computer interfaces aim to decipher what a person is trying to say from neural activity alone, restoring communication to people with paralysis who have lost the ability to speak intelligibly. The Brain-to-Text Benchmark '24 and associated competition was created to foster the advancement of decoding algorithms that convert neural activity to text. Here, we summarize the lessons learned from the competition ending on June 1, 2024 (the top 4 entrants also presented their experiences in a recorded webinar). The largest improvements in accuracy were achieved using an ensembling approach, where the output of multiple independent decoders was merged using a fine-tuned large language model (an approach used by all 3 top entrants). Performance gains were also found by improving how the baseline recurrent neural network (RNN) model was trained, including by optimizing learning rate scheduling and by using a diphone training objective. Improving upon the model architecture itself proved more difficult, however, with attempts to use deep state space models or transformers not yet appearing to offer a benefit over the RNN baseline. The benchmark will remain open indefinitely to support further work towards increasing the accuracy of brain-to-text algorithms. 

**Abstract (ZH)**: 基于语音的脑-计算机接口旨在仅从神经活动解码一个人想要表达的内容，恢复那些因失去流利说话能力而无法沟通的瘫痪患者的交流能力。为了促进将神经活动转换为文本的解码算法的发展，《脑-文本基准24》及其相关的竞赛应运而生。在此，我们总结了截至2024年6月1日结束的竞赛中所学到的经验教训（前四名参赛者还记录并分享了他们的参赛体验）。在准确性方面取得的最大改进是通过集成方法实现的，即使用微调的大语言模型合并多个独立解码器的输出（这种方法被所有前三名参赛者所采用）。通过改进基础递归神经网络（RNN）模型的训练方式，也取得了性能提升，包括优化学习率调度和使用双音节训练目标。然而，改进模型架构本身更具挑战性，尽管尝试使用深层状态空间模型或变换器尚未显示出比RNN基线更好的效果。该基准竞赛将长期开放，以继续支持提高脑-文本算法准确性的相关工作。 

---
# A Multi-AI Agent System for Autonomous Optimization of Agentic AI Solutions via Iterative Refinement and LLM-Driven Feedback Loops 

**Title (ZH)**: 基于迭代细化和LLM驱动的反馈循环的自主优化多AI代理系统：为代理AI解决方案赋能 

**Authors**: Kamer Ali Yuksel, Hassan Sawaf  

**Link**: [PDF](https://arxiv.org/pdf/2412.17149)  

**Abstract**: Agentic AI systems use specialized agents to handle tasks within complex workflows, enabling automation and efficiency. However, optimizing these systems often requires labor-intensive, manual adjustments to refine roles, tasks, and interactions. This paper introduces a framework for autonomously optimizing Agentic AI solutions across industries, such as NLP-driven enterprise applications. The system employs agents for Refinement, Execution, Evaluation, Modification, and Documentation, leveraging iterative feedback loops powered by an LLM (Llama 3.2-3B). The framework achieves optimal performance without human input by autonomously generating and testing hypotheses to improve system configurations. This approach enhances scalability and adaptability, offering a robust solution for real-world applications in dynamic environments. Case studies across diverse domains illustrate the transformative impact of this framework, showcasing significant improvements in output quality, relevance, and actionability. All data for these case studies, including original and evolved agent codes, along with their outputs, are here: this https URL 

**Abstract (ZH)**: 以下是符合学术规范的翻译：

具有代理功能的AI系统使用专门的代理来处理复杂工作流程中的任务，从而实现自动化和提高效率。然而，优化这些系统通常需要耗时的手动调整来细化角色、任务和交互。本文提出了一种跨行业的自主优化具有代理功能的AI解决方案的框架，适用于如基于NLP的企业应用等场景。该系统采用代理进行细化（Refinement）、执行（Execution）、评估（Evaluation）、修改（Modification）和记录（Documentation），利用由LLM（Llama 3.2-3B）驱动的迭代反馈循环。该框架通过自主生成和测试假设来优化系统配置，实现了无需人工干预的最优性能。这种方法增强了系统的可扩展性和适应性，提供了在动态环境中实现实用解决方案的有效方案。来自不同领域的案例研究展示了该框架的转变性影响，显著提高了输出质量、相关性和可操作性。这些案例研究的所有数据，包括原始和演变的代理代码及其输出，均可在此获得：this https URL 

---
# Hate Speech Detection and Target Identification in Devanagari Languages via Parameter Efficient Fine-Tuning of LLMs 

**Title (ZH)**: 通过参数高效微调大语言模型来检测和识别梵文语言中的仇恨言论及目标对象 

**Authors**: Rushendra Sidibomma, Pransh Patwa, Parth Patwa, Aman Chadha, Vinija Jain, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2412.17131)  

**Abstract**: The detection of hate speech has become increasingly important in combating online hostility and its real-world consequences. Despite recent advancements, there is limited research addressing hate speech detection in Devanagari-scripted languages, where resources and tools are scarce. While large language models (LLMs) have shown promise in language-related tasks, traditional fine-tuning approaches are often infeasible given the size of the models. In this paper, we propose a Parameter Efficient Fine tuning (PEFT) based solution for hate speech detection and target identification. We evaluate multiple LLMs on the Devanagari dataset provided by (Thapa et al., 2025), which contains annotated instances in 2 languages - Hindi and Nepali. The results demonstrate the efficacy of our approach in handling Devanagari-scripted content. 

**Abstract (ZH)**: 仇恨言论的检测在应对网络恶意行为及其现实后果方面变得越来越重要。尽管最近取得了进展，但在僧迦罗（DevaNagari）脚本语言中进行仇恨言论检测的研究仍然有限，资源和工具相对稀缺。虽然大型语言模型（LLMs）在语言相关任务中表现出一定的潜力，但由于模型规模庞大，传统的微调方法往往不可行。在本文中，我们提出了一种参数高效微调（PEFT）方法来解决僧迦罗脚本语言中的仇恨言论检测和目标识别问题。我们使用Thapa等人（2025）提供的僧迦罗语数据集对多种LLM进行评估，该数据集包含用两种语言（ Hindi 和 Nepali）标注的实例。实验结果展示了我们方法在处理僧迦罗脚本内容方面的有效性。 

---
# Lies, Damned Lies, and Distributional Language Statistics: Persuasion and Deception with Large Language Models 

**Title (ZH)**: 谎言、伪证与分布型语言统计：大规模语言模型中的说服与欺骗 

**Authors**: Cameron R. Jones, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2412.17128)  

**Abstract**: Large Language Models (LLMs) can generate content that is as persuasive as human-written text and appear capable of selectively producing deceptive outputs. These capabilities raise concerns about potential misuse and unintended consequences as these systems become more widely deployed. This review synthesizes recent empirical work examining LLMs' capacity and proclivity for persuasion and deception, analyzes theoretical risks that could arise from these capabilities, and evaluates proposed mitigations. While current persuasive effects are relatively small, various mechanisms could increase their impact, including fine-tuning, multimodality, and social factors. We outline key open questions for future research, including how persuasive AI systems might become, whether truth enjoys an inherent advantage over falsehoods, and how effective different mitigation strategies may be in practice. 

**Abstract (ZH)**: 大规模语言模型（LLMs）能够生成与人类撰写的文本同样具有说服力的内容，并且似乎能够有选择地产生欺骗性输出。这些能力引发了对其潜在滥用和意外后果的担忧，尤其是在这些系统更加广泛部署的情况下。本综述综合了近期关于LLMs的说服能力和倾向性欺骗的研究成果，分析了这些能力可能引发的理论风险，并评估了已提出的缓解措施。虽然当前的说服效果相对较小，但多种机制可能会增加其影响，包括模型微调、多模态和社交因素。我们概述了未来研究中的关键开放问题，包括如何使说服性人工智能系统变得更加有效，真相是否天然具有优势，以及不同缓解策略的实际有效性如何。 

---
# Learning to Adapt to Low-Resource Paraphrase Generation 

**Title (ZH)**: 学习适应低资源同义句生成 

**Authors**: Zhigen Li, Yanmeng Wang, Rizhao Fan, Ye Wang, Jianfeng Li, Shaojun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17111)  

**Abstract**: Paraphrase generation is a longstanding NLP task and achieves great success with the aid of large corpora. However, transferring a paraphrasing model to another domain encounters the problem of domain shifting especially when the data is sparse. At the same time, widely using large pre-trained language models (PLMs) faces the overfitting problem when training on scarce labeled data. To mitigate these two issues, we propose, LAPA, an effective adapter for PLMs optimized by meta-learning. LAPA has three-stage training on three types of related resources to solve this problem: 1. pre-training PLMs on unsupervised corpora, 2. inserting an adapter layer and meta-training on source domain labeled data, and 3. fine-tuning adapters on a small amount of target domain labeled data. This method enables paraphrase generation models to learn basic language knowledge first, then learn the paraphrasing task itself later, and finally adapt to the target task. Our experimental results demonstrate that LAPA achieves state-of-the-art in supervised, unsupervised, and low-resource settings on three benchmark datasets. With only 2\% of trainable parameters and 1\% labeled data of the target task, our approach can achieve a competitive performance with previous work. 

**Abstract (ZH)**: 摘要：短语生成是长期存在的自然语言处理任务，通过大规模语料库的帮助取得了巨大成功。然而，将短语生成模型迁移到另一个领域会遇到领域偏移问题，尤其是在数据稀少的情况下。同时，在稀缺标注数据上使用大规模预训练语言模型（PLM）面临过拟合问题。为缓解这两个问题，我们提出了LAPA，一种通过元学习优化的有效PLM适配器。LAPA 通过三个阶段在三种相关资源上进行训练来解决这个问题：1. 在无监督语料库上预训练PLM，2. 插入适配器层并在源领域标注数据上进行元训练，3. 在少量目标领域标注数据上微调适配器。这种方法允许模型首先学习基本语言知识，然后学习实际的短语生成任务，最后适应目标任务。我们的实验结果表明，LAPA 在三个基准数据集的监督、无监督和少资源设置下均达到了最佳性能。即使仅使用目标任务可训练参数的2%和标注数据的1%，我们的方法也能与现有工作取得竞争力的表现。 

---
# SAIL: Sample-Centric In-Context Learning for Document Information Extraction 

**Title (ZH)**: SAIL：基于样本的 CONTEXT 学习方法在文档信息提取中的应用 

**Authors**: Jinyu Zhang, Zhiyuan You, Jize Wang, Xinyi Le  

**Link**: [PDF](https://arxiv.org/pdf/2412.17092)  

**Abstract**: Document Information Extraction (DIE) aims to extract structured information from Visually Rich Documents (VRDs). Previous full-training approaches have demonstrated strong performance but may struggle with generalization to unseen data. In contrast, training-free methods leverage powerful pre-trained models like Large Language Models (LLMs) to address various downstream tasks with only a few examples. Nonetheless, training-free methods for DIE encounter two primary challenges: (1) understanding the complex relationship between layout and textual elements in VRDs, and (2) providing accurate guidance to pre-trained models. To address these challenges, we propose Sample-centric In-context Learning (SAIL) for DIE. SAIL introduces a fine-grained entity-level textual similarity to facilitate in-depth text analysis by LLMs and incorporates layout similarity to enhance the analysis of layouts in VRDs. Additionally, SAIL formulates a unified In-Context Learning (ICL) prompt template for various sample-centric examples, enabling tailored prompts that deliver precise guidance to pre-trained models for each sample. Extensive experiments on FUNSD, CORD, and SROIE benchmarks with various base models (e.g., LLMs) indicate that our method outperforms training-free baselines, even closer to the full-training methods. The results show the superiority and generalization of our method. 

**Abstract (ZH)**: 文档信息提取（Document Information Extraction, DIE）旨在从视觉丰富文档（Visually Rich Documents, VRDs）中提取结构化信息。之前的端到端训练方法虽然性能强大，但在处理未见过的数据时可能会出现泛化困难的问题。相比之下，无需训练的方法利用如大型语言模型（Large Language Models, LLMs）等强大的预训练模型，仅通过少量实例即可完成各种下游任务。然而，DIE中的无需训练方法面临两大主要挑战：（1）理解VRDs中布局和文本元素之间复杂的相互关系；（2）为预训练模型提供准确的指导。为应对这些挑战，我们提出了面向样本的上下文学习（Sample-centric In-context Learning, SAIL）方法。SAIL通过引入细粒度的实体级文本相似性，促进大型语言模型进行深入的文本分析；通过引入布局相似性，增强对VRDs中布局的分析。此外，SAIL制定了一个统一的面向样本的上下文学习（In-Context Learning, ICL）提示模板，能够为各种样本提供定制化的提示，以精确指导预训练模型。在FUNSD、CORD和SROIE等基准测试中，使用各种基础模型（例如，大型语言模型）进行的广泛实验表明，我们的方法在性能上超越了无需训练的基线方法，甚至接近了端到端训练的方法。实验结果展示了我们方法的优越性和泛化能力。 

---
# Computational Analysis of Character Development in Holocaust Testimonies 

**Title (ZH)**: Holocaust见证中人物发展计算分析 

**Authors**: Esther Shizgal, Eitan Wagner, Renana Keydar, Omri Abend  

**Link**: [PDF](https://arxiv.org/pdf/2412.17063)  

**Abstract**: This work presents a computational approach to analyze character development along the narrative timeline. The analysis characterizes the inner and outer changes the protagonist undergoes within a narrative, and the interplay between them. We consider transcripts of Holocaust survivor testimonies as a test case, each telling the story of an individual in first-person terms. We focus on the survivor's religious trajectory, examining the evolution of their disposition toward religious belief and practice along the testimony. Clustering the resulting trajectories in the dataset, we identify common sequences in the data. Our findings highlight multiple common structures of religiosity across the narratives: in terms of belief, most present a constant disposition, while for practice, most present an oscillating structure, serving as valuable material for historical and sociological research. This work demonstrates the potential of natural language processing techniques for analyzing character evolution through thematic trajectories in narratives. 

**Abstract (ZH)**: 本研究提出了一种计算方法来分析人物在叙事时间线上的发展。该分析刻画了叙事中主人公内在外在变化及其相互作用。我们以犹太大屠杀幸存者证词为测试案例，每份证词都是从第一人称讲述个体的故事。我们专注于幸存者的宗教轨迹，考察他们在证词中对宗教信仰和实践的态度演变。通过对数据集中的轨迹进行聚类，我们识别出数据中的常见模式。我们的研究结果突显了叙事中宗教信仰结构的多种共性：在信仰方面，大多数呈稳定态度；而在实践方面，大多数呈波动结构，这些发现对于历史学和社会学研究具有宝贵的资料价值。本研究展示了自然语言处理技术在通过主题轨迹分析叙事中人物演变方面的潜在价值。 

---
# Multi-Agent Sampling: Scaling Inference Compute for Data Synthesis with Tree Search-Based Agentic Collaboration 

**Title (ZH)**: 多代理采样：基于树搜索的代理协作扩展数据合成的推理计算能力 

**Authors**: Hai Ye, Mingbao Lin, Hwee Tou Ng, Shuicheng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2412.17061)  

**Abstract**: Scaling laws for inference compute in multi-agent systems remain under-explored compared to single-agent scenarios. This work aims to bridge this gap by investigating the problem of data synthesis through multi-agent sampling, where synthetic responses are generated by sampling from multiple distinct language models. Effective model coordination is crucial for successful multi-agent collaboration. Unlike previous approaches that rely on fixed workflows, we treat model coordination as a multi-step decision-making process, optimizing generation structures dynamically for each input question. We introduce Tree Search-based Orchestrated Agents~(TOA), where the workflow evolves iteratively during the sequential sampling process. To achieve this, we leverage Monte Carlo Tree Search (MCTS), integrating a reward model to provide real-time feedback and accelerate exploration. Our experiments on alignment, machine translation, and mathematical reasoning demonstrate that multi-agent sampling significantly outperforms single-agent sampling as inference compute scales. TOA is the most compute-efficient approach, achieving SOTA performance on WMT and a 71.8\% LC win rate on AlpacaEval. Moreover, fine-tuning with our synthesized alignment data surpasses strong preference learning methods on challenging benchmarks such as Arena-Hard and AlpacaEval. 

**Abstract (ZH)**: 与单智能体场景相比，多智能体系统中的推理计算规模律仍鲜有探索。本工作旨在通过多智能体采样问题的数据合成，填补这一空白。在多智能体采样中，合成响应通过从多个不同的语言模型中采样生成。有效的模型协调对于成功的多智能体协作至关重要。不同于之前依赖固定工作流程的方法，我们把模型协调视为一个多步骤的决策过程，动态优化每个输入问题的生成结构。我们引入了一种基于树搜索的协调智能体（Tree Search-based Orchestrated Agents, TOA），在顺序采样过程中工作流程迭代地演进。为此，我们利用蒙特卡洛树搜索（Monte Carlo Tree Search, MCTS），结合奖励模型提供实时反馈，加速探索。我们在对齐、机器翻译和数学推理方面的实验表明，随着推理计算规模的扩大，多智能体采样显著优于单智能体采样。TOA是最高效的计算方法，在WMT上实现了SOTA性能，并在AlpacaEval上获得了71.8%的LC胜率。此外，使用我们合成的对齐数据进行微调，在难度较大的基准Arena-Hard和AlpacaEval上超过了强大的偏好学习方法。 

---
# The HalluRAG Dataset: Detecting Closed-Domain Hallucinations in RAG Applications Using an LLM's Internal States 

**Title (ZH)**: 《HalluRAG数据集：使用大型语言模型内部状态检测RAG应用程序中的封闭域幻觉》 

**Authors**: Fabian Ridder, Malte Schilling  

**Link**: [PDF](https://arxiv.org/pdf/2412.17056)  

**Abstract**: Detecting hallucinations in large language models (LLMs) is critical for enhancing their reliability and trustworthiness. Most research focuses on hallucinations as deviations from information seen during training. However, the opaque nature of an LLM's parametric knowledge complicates the understanding of why generated texts appear ungrounded: The LLM might not have picked up the necessary knowledge from large and often inaccessible datasets, or the information might have been changed or contradicted during further training. Our focus is on hallucinations involving information not used in training, which we determine by using recency to ensure the information emerged after a cut-off date. This study investigates these hallucinations by detecting them at sentence level using different internal states of various LLMs. We present HalluRAG, a dataset designed to train classifiers on these hallucinations. Depending on the model and quantization, MLPs trained on HalluRAG detect hallucinations with test accuracies ranging up to 75 %, with Mistral-7B-Instruct-v0.1 achieving the highest test accuracies. Our results show that IAVs detect hallucinations as effectively as CEVs and reveal that answerable and unanswerable prompts are encoded differently as separate classifiers for these categories improved accuracy. However, HalluRAG showed some limited generalizability, advocating for more diversity in datasets on hallucinations. 

**Abstract (ZH)**: 检测大规模语言模型（LLMs）中的幻觉对于提升其可靠性和可信度至关重要。大多数研究侧重于检测与训练期间看到的信息偏差的幻觉。然而，LLM参数化知识的不透明性使得理解生成文本为何显得缺乏根据变得更加复杂：LLM可能未能从广泛且往往难以访问的数据集中获取必要的知识，或者在进一步训练过程中信息可能已被改变或被矛盾所取代。我们关注的是训练时未使用的信息引发的幻觉，通过使用时间递进来确定这些信息是在某个截断日期之后出现的。本研究通过在不同LLM的内部状态层面检测幻觉，来调查这些幻觉。我们介绍了HalluRAG数据集，用于训练分类器以识别这些幻觉。根据不同模型和量化方法，训练在HalluRAG上进行的MLP在测试分类准确性上可达75%，Mistral-7B-Instruct-v0.1的表现最佳。我们的研究结果表明，IAVs在检测幻觉方面的效果与CEVs相当，并揭示了可回答和不可回答提示的编码方式存在差异，这在两种类别的分类器中得到了体现，提高了准确性。然而，HalluRAG展示了一定的有限泛化能力，这表明在幻觉数据集方面需要更多样性。 

---
# Shaping the Safety Boundaries: Understanding and Defending Against Jailbreaks in Large Language Models 

**Title (ZH)**: 塑造安全边界：理解并抵御大型语言模型的越狱攻击 

**Authors**: Lang Gao, Xiangliang Zhang, Preslav Nakov, Xiuying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.17034)  

**Abstract**: Jailbreaking in Large Language Models (LLMs) is a major security concern as it can deceive LLMs to generate harmful text. Yet, there is still insufficient understanding of how jailbreaking works, which makes it hard to develop effective defense strategies. We aim to shed more light into this issue: we conduct a detailed large-scale analysis of seven different jailbreak methods and find that these disagreements stem from insufficient observation samples. In particular, we introduce \textit{safety boundary}, and we find that jailbreaks shift harmful activations outside that safety boundary, where LLMs are less sensitive to harmful information. We also find that the low and the middle layers are critical in such shifts, while deeper layers have less impact. Leveraging on these insights, we propose a novel defense called \textbf{Activation Boundary Defense} (ABD), which adaptively constrains the activations within the safety boundary. We further use Bayesian optimization to selectively apply the defense method to the low and the middle layers. Our experiments on several benchmarks show that ABD achieves an average DSR of over 98\% against various forms of jailbreak attacks, with less than 2\% impact on the model's general capabilities. 

**Abstract (ZH)**: 大型语言模型（LLMs）的越狱是一种重要的安全问题，因为它可以使LLMs生成有害文本。然而，关于越狱机制的理解仍然不足，这使得难以开发有效的防御策略。我们旨在更深入地探讨这一问题：我们对七种不同的越狱方法进行了详细的大规模分析，并发现这些分歧源于观察样本的不足。特别地，我们引入了“安全边界”的概念，并发现越狱将有害激活转移到了这个安全边界之外，而在这个区域，LLMs 对有害信息的敏感度较低。我们还发现，低层和中间层在这些变化中起着关键作用，而深层结构的影响较小。基于这些洞察，我们提出了一种新颖的防御方法，称为**激活边界防御**（ABD），它能够自适应地将激活值限制在安全边界内。我们进一步利用贝叶斯优化策略，选择性地将防御方法应用于低层和中间层。在多个基准测试上的实验结果显示，ABD 方法能在各种形式的越狱攻击中实现超过 98% 的有效防御率，同时对模型的整体性能影响不到 2%。 

---
# MINTQA: A Multi-Hop Question Answering Benchmark for Evaluating LLMs on New and Tail Knowledge 

**Title (ZH)**: MINTQA：评价大型语言模型在新颖和特定知识库中进行多跳问答的能力基准 

**Authors**: Jie He, Nan Hu, Wanqiu Long, Jiaoyan Chen, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2412.17032)  

**Abstract**: Large language models (LLMs) have demonstrated impressive capabilities in various reasoning tasks but face significant challenges with complex, knowledge-intensive multi-hop queries, particularly those involving new or long-tail knowledge. Existing benchmarks often fail to fully address these challenges. To bridge this gap, we introduce MINTQA (Multi-hop Question Answering on New and Tail Knowledge), a comprehensive benchmark to evaluate LLMs' capabilities in multi-hop reasoning across four critical dimensions: question handling strategy, sub-question generation, retrieval-augmented generation, and iterative or dynamic decomposition and retrieval. MINTQA comprises 10,479 question-answer pairs for evaluating new knowledge and 17,887 pairs for assessing long-tail knowledge, with each question equipped with corresponding sub-questions and answers. Our systematic evaluation of 22 state-of-the-art LLMs on MINTQA reveals significant limitations in their ability to handle complex knowledge base queries, particularly in handling new or unpopular knowledge. Our findings highlight critical challenges and offer insights for advancing multi-hop reasoning capabilities. The MINTQA benchmark is available at this https URL. 

**Abstract (ZH)**: 大型语言模型（LLMs）在各种推理任务中展现了出色的性能，但在应对复杂、知识密集型的多跳查询，尤其是涉及新知识或长尾知识的情况时，仍面临重大挑战。现有的基准测试往往未能充分解决这些问题。为解决这一差距，我们引入了MINTQA（多跳问答：新知识和长尾知识），这是一个综合基准，用于评估LLMs在多跳推理方面的能力，涵盖四个关键维度：问题处理策略、子问题生成、检索增强生成以及迭代或动态分解与检索。MINTQA 包含10,479个用于评估新知识的问题-答案对和17,887个用于评估长尾知识的问题-答案对，每个问题都配有相应的子问题和答案。我们在MINTQA 上对22个最先进的LLMs 进行系统的评估，揭示了它们在处理复杂知识库查询方面的显著局限性，尤其是处理新知识或不常用知识的能力。我们的研究结果突出了关键挑战并提供了提升多跳推理能力的见解。MINTQA 基准测试可在以下链接获取：[这里](this https URL)。 

---
# A Reality Check on Context Utilisation for Retrieval-Augmented Generation 

**Title (ZH)**: 对检索增强生成中背景信息利用情况的一种现实检视 

**Authors**: Lovisa Hagström, Sara Vera Marjanović, Haeun Yu, Arnav Arora, Christina Lioma, Maria Maistro, Pepa Atanasova, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2412.17031)  

**Abstract**: Retrieval-augmented generation (RAG) helps address the limitations of the parametric knowledge embedded within a language model (LM). However, investigations of how LMs utilise retrieved information of varying complexity in real-world scenarios have been limited to synthetic contexts. We introduce DRUID (Dataset of Retrieved Unreliable, Insufficient and Difficult-to-understand contexts) with real-world queries and contexts manually annotated for stance. The dataset is based on the prototypical task of automated claim verification, for which automated retrieval of real-world evidence is crucial. We compare DRUID to synthetic datasets (CounterFact, ConflictQA) and find that artificial datasets often fail to represent the complex and diverse real-world context settings. We show that synthetic datasets exaggerate context characteristics rare in real retrieved data, which leads to inflated context utilisation results, as measured by our novel ACU score. Moreover, while previous work has mainly focused on singleton context characteristics to explain context utilisation, correlations between singleton context properties and ACU on DRUID are surprisingly small compared to other properties related to context source. Overall, our work underscores the need for real-world aligned context utilisation studies to represent and improve performance in real-world RAG settings. 

**Abstract (ZH)**: 恢复增强生成（RAG）有助于解决参数知识嵌入在语言模型（LM）中的局限性。然而，关于语言模型在实际场景中如何利用不同复杂度的检索信息的相关研究仅限于合成情境。我们引入了DRUID（Dataset of Retrieved Unreliable, Insufficient and Difficult-to-understand contexts），该数据集包含手动标注立场的现实查询和上下文。该数据集基于自动化声明验证这一典型任务，对于自动化检索现实证据至关重要。我们将DRUID与合成数据集（CounterFact, ConflictQA）进行比较，发现合成数据集往往无法代表复杂的和多变的现实环境背景设置。我们展示了合成数据集夸大了现实中罕见的上下文特征，这导致测量到的上下文利用结果虚高，我们通过新的ACU得分来衡量这一点。此外，虽然之前的研究所主要集中在单个上下文特征的解释上，但DRUID中的单个上下文属性与ACU之间的相关性与其他与上下文来源相关的属性相比出乎意料地小。总体而言，我们的研究强调了在现实世界RAG设置中进行上下文利用研究的必要性，以代表和改进性能。 

---
# Reversed Attention: On The Gradient Descent Of Attention Layers In GPT 

**Title (ZH)**: 逆向注意机制：关于GPT中的注意层在梯度下降中的研究 

**Authors**: Shahar Katz, Lior Wolf  

**Link**: [PDF](https://arxiv.org/pdf/2412.17019)  

**Abstract**: The success of Transformer-based Language Models (LMs) stems from their attention mechanism. While this mechanism has been extensively studied in explainability research, particularly through the attention values obtained during the forward pass of LMs, the backward pass of attention has been largely overlooked. In this work, we study the mathematics of the backward pass of attention, revealing that it implicitly calculates an attention matrix we refer to as "Reversed Attention". We examine the properties of Reversed Attention and demonstrate its ability to elucidate the models' behavior and edit dynamics. In an experimental setup, we showcase the ability of Reversed Attention to directly alter the forward pass of attention, without modifying the model's weights, using a novel method called "attention patching". In addition to enhancing the comprehension of how LM configure attention layers during backpropagation, Reversed Attention maps contribute to a more interpretable backward pass. 

**Abstract (ZH)**: 基于Transformer的语言模型（LMs）的成功归功于其注意力机制。尽管这一机制在解释性研究中得到了广泛研究，特别是在LM前向传播过程中获得的注意力值，但注意力的反向传播却很少受到关注。在本研究中，我们探讨了注意力反向传播的数学原理，揭示了它隐含计算了一个我们称之为“反向注意力”的注意力矩阵。我们分析了反向注意力的性质，并展示了它在阐明模型行为和编辑动态方面的能力。在实验设置中，我们展示了如何通过一种名为“注意力补丁”（attention patching）的新型方法直接修改注意力的前向传播，而不需更改模型权重，来利用反向注意力的能力。此外，反向注意力地图有助于使反向传播过程更具可解释性，增强了对在反向传播过程中LM如何配置注意力层的理解。 

---
# Robustness of Large Language Models Against Adversarial Attacks 

**Title (ZH)**: 大型语言模型对抗性攻击的 robustness 分析 

**Authors**: Yiyi Tao, Yixian Shen, Hang Zhang, Yanxin Shen, Lun Wang, Chuanqi Shi, Shaoshuai Du  

**Link**: [PDF](https://arxiv.org/pdf/2412.17011)  

**Abstract**: The increasing deployment of Large Language Models (LLMs) in various applications necessitates a rigorous evaluation of their robustness against adversarial attacks. In this paper, we present a comprehensive study on the robustness of GPT LLM family. We employ two distinct evaluation methods to assess their resilience. The first method introduce character-level text attack in input prompts, testing the models on three sentiment classification datasets: StanfordNLP/IMDB, Yelp Reviews, and SST-2. The second method involves using jailbreak prompts to challenge the safety mechanisms of the LLMs. Our experiments reveal significant variations in the robustness of these models, demonstrating their varying degrees of vulnerability to both character-level and semantic-level adversarial attacks. These findings underscore the necessity for improved adversarial training and enhanced safety mechanisms to bolster the robustness of LLMs. 

**Abstract (ZH)**: 随着大型语言模型（LLMs）在各种应用中的部署日益增加，对其对抗攻击鲁棒性的严格评估变得至关重要。在本文中，我们对GPT LLM家族的鲁棒性进行了全面研究。我们采用了两种不同的评估方法来评估模型的抗攻击能力。第一个方法在输入提示中引入字符级别文本攻击，对三个情感分类数据集（StanfordNLP/IMDB、Yelp评论和SST-2）进行了测试。第二个方法则是使用“监狱突破”（jailbreak）提示来挑战LLMs的安全机制。我们的实验结果显示，这些模型在鲁棒性方面存在显著差异，表明它们在字符级别和语义级别对抗攻击面前具有不同程度的脆弱性。这些发现强调了改进对抗训练及增强安全机制以提高LLM鲁棒性的必要性。 

---
# On Fusing ChatGPT and Ensemble Learning in Discon-tinuous Named Entity Recognition in Health Corpora 

**Title (ZH)**: 将以下论文的内容或标题翻译成中文，并符合学术规范：

"Fusing ChatGPT and Ensemble Learning in Discontinuous Named Entity Recognition in Health Corpora"

翻译为：

"将 ChatGPT 与集成学习融合应用于医疗语料中的不连续命名实体识别" 

**Authors**: Tzu-Chieh Chen, Wen-Yang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2412.16976)  

**Abstract**: Named Entity Recognition has traditionally been a key task in natural language processing, aiming to identify and extract important terms from unstructured text data. However, a notable challenge for contemporary deep-learning NER models has been identifying discontinuous entities, which are often fragmented within the text. To date, methods to address Discontinuous Named Entity Recognition have not been explored using ensemble learning to the best of our knowledge. Furthermore, the rise of large language models, such as ChatGPT in recent years, has shown significant effectiveness across many NLP tasks. Most existing approaches, however, have primarily utilized ChatGPT as a problem-solving tool rather than exploring its potential as an integrative element within ensemble learning algorithms. In this study, we investigated the integration of ChatGPT as an arbitrator within an ensemble method, aiming to enhance performance on DNER tasks. Our method combines five state-of-the-art NER models with ChatGPT using custom prompt engineering to assess the robustness and generalization capabilities of the ensemble algorithm. We conducted experiments on three benchmark medical datasets, comparing our method against the five SOTA models, individual applications of GPT-3.5 and GPT-4, and a voting ensemble method. The results indicate that our proposed fusion of ChatGPT with the ensemble learning algorithm outperforms the SOTA results in the CADEC, ShARe13, and ShARe14 datasets, showcasing its potential to enhance NLP applications in the healthcare domain. 

**Abstract (ZH)**: 命名实体识别一直是自然语言处理中的关键任务，旨在从未结构化的文本数据中识别和提取重要项。然而，现代深度学习命名实体识别模型在识别断续实体时面临着显著挑战，这些断续实体在文本中通常被分割。迄今为止，我们所知范围内还没有使用集成学习方法来解决断续命名实体识别的问题。此外，近年来大语言模型，如ChatGPT的出现，在许多自然语言处理任务中显示出了显著的有效性。然而，现有的大多数方法主要将ChatGPT作为问题解决工具，而没有将其潜力作为集成学习算法中的整合元素进行探索。本研究中，我们探索了将ChatGPT作为集成方法中的仲裁者进行整合的方法，旨在提高断续命名实体识别任务的性能。我们使用自定义提示工程技术将五种最先进的命名实体识别模型与ChatGPT结合，评估了集成算法的鲁棒性和泛化能力。我们在三个基准医学数据集上进行了实验，将我们的方法与五种最先进模型、GPT-3.5和GPT-4的独立应用以及投票集成方法进行了比较。结果表明，我们将ChatGPT与集成学习算法的融合在CADEC、ShARe13和ShARe14数据集上优于最先进的结果，展示了其在医疗保健领域自然语言处理应用中的潜在优势。 

---
# Part-Of-Speech Sensitivity of Routers in Mixture of Experts Models 

**Title (ZH)**: 混合专家模型中路由器的词性敏感性 

**Authors**: Elie Antoine, Frédéric Béchet, Philippe Langlais  

**Link**: [PDF](https://arxiv.org/pdf/2412.16971)  

**Abstract**: This study investigates the behavior of model-integrated routers in Mixture of Experts (MoE) models, focusing on how tokens are routed based on their linguistic features, specifically Part-of-Speech (POS) tags. The goal is to explore across different MoE architectures whether experts specialize in processing tokens with similar linguistic traits. By analyzing token trajectories across experts and layers, we aim to uncover how MoE models handle linguistic information. Findings from six popular MoE models reveal expert specialization for specific POS categories, with routing paths showing high predictive accuracy for POS, highlighting the value of routing paths in characterizing tokens. 

**Abstract (ZH)**: 本研究探讨了混合专家模型（MoE）中模型集成路由器的行为，重点关注基于语言特性（特别是词性标注POS标记）如何路由标记。研究的主要目标是探讨在不同的MoE架构中，专家是否专门处理具有相似语言特征的标记。通过分析标记在专家和各层之间的路径，旨在揭示MoE模型如何处理语言信息。对六种流行的MoE模型的研究发现，专家在特定的词性类别上表现出专业化，路由路径在预测词性方面表现出较高的准确性，突显了路由路径在表征标记方面的价值。 

---
# LH-Mix: Local Hierarchy Correlation Guided Mixup over Hierarchical Prompt Tuning 

**Title (ZH)**: LH-Mix: 局部层次相关性指导的层次提示调优混合策略 

**Authors**: Fanshuang Kong, Richong Zhang, Ziqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16963)  

**Abstract**: Hierarchical text classification (HTC) aims to assign one or more labels in the hierarchy for each text. Many methods represent this structure as a global hierarchy, leading to redundant graph structures. To address this, incorporating a text-specific local hierarchy is essential. However, existing approaches often model this local hierarchy as a sequence, focusing on explicit parent-child relationships while ignoring implicit correlations among sibling/peer relationships. In this paper, we first integrate local hierarchies into a manual depth-level prompt to capture parent-child relationships. We then apply Mixup to this hierarchical prompt tuning scheme to improve the latent correlation within sibling/peer relationships. Notably, we propose a novel Mixup ratio guided by local hierarchy correlation to effectively capture intrinsic correlations. This Local Hierarchy Mixup (LH-Mix) model demonstrates remarkable performance across three widely-used datasets. 

**Abstract (ZH)**: 层次文本分类（Hierarchical Text Classification, HTC）旨在为每一项文本分配一个或多个层次结构中的标签。许多方法将这种结构表示为全局层次结构，导致冗余的图结构。为了解决这一问题，将特定文本的局部层次结构纳入考虑是必不可少的。然而，现有的方法通常将局部层次结构建模为序列，重点关注显式的父节点和子节点关系，而忽略了兄弟/同级关系之间的隐含关联。在此论文中，我们首先将局部层次结构整合到手动深度层级提示中，以捕获父节点和子节点关系。然后，我们在此层次结构提示调整方案中应用Mixup方法，以提高兄弟/同级关系中的潜在关联。值得注意的是，我们提出了一种由局部层次结构关联指导的新型Mixup比率，以有效地捕获内在关联。Local Hierarchy Mixup（LH-Mix）模型在三个广泛使用数据集上展示了卓越的性能。 

---
# Aristotle: Mastering Logical Reasoning with A Logic-Complete Decompose-Search-Resolve Framework 

**Title (ZH)**: 亚里士多德：基于逻辑完备分解-搜索-解决框架的逻辑推理掌握 

**Authors**: Jundong Xu, Hao Fei, Meng Luo, Qian Liu, Liangming Pan, William Yang Wang, Preslav Nakov, Mong-Li Lee, Wynne Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16953)  

**Abstract**: In the context of large language models (LLMs), current advanced reasoning methods have made impressive strides in various reasoning tasks. However, when it comes to logical reasoning tasks, major challenges remain in both efficacy and efficiency. This is rooted in the fact that these systems fail to fully leverage the inherent structure of logical tasks throughout the reasoning processes such as decomposition, search, and resolution. To address this, we propose a logic-complete reasoning framework, Aristotle, with three key components: Logical Decomposer, Logical Search Router, and Logical Resolver. In our framework, symbolic expressions and logical rules are comprehensively integrated into the entire reasoning process, significantly alleviating the bottlenecks of logical reasoning, i.e., reducing sub-task complexity, minimizing search errors, and resolving logical contradictions. The experimental results on several datasets demonstrate that Aristotle consistently outperforms state-of-the-art reasoning frameworks in both accuracy and efficiency, particularly excelling in complex logical reasoning scenarios. We will open-source all our code at this https URL. 

**Abstract (ZH)**: 在大型语言模型（LLMs）的背景下，当前先进的推理方法已在各种推理任务中取得了显著进展。然而，当涉及到逻辑推理任务时，这些系统在有效性和效率方面仍面临重大挑战。这些挑战根源于系统在推理过程（如分解、搜索和解决）中未能充分利用逻辑任务固有的结构。为解决这一问题，我们提出了一种逻辑完备推理框架Aristotle，该框架包含三个关键组件：逻辑分解器、逻辑搜索路由器和逻辑解决器。在我们的框架中，符号表达式和逻辑规则被全面整合到整个推理过程中，显著缓解了逻辑推理的瓶颈问题，包括降低子任务复杂性、最小化搜索错误和解决逻辑矛盾。在多个数据集上的实验结果表明，Aristotle在准确性和效率方面均优于现有的推理框架，特别是在复杂的逻辑推理场景中表现更为出色。我们将在此httpsURL开源所有代码。 

---
# A Career Interview Dialogue System using Large Language Model-based Dynamic Slot Generation 

**Title (ZH)**: 基于大规模语言模型的动态槽生成的职业访谈对话系统 

**Authors**: Ekai Hashimoto, Mikio Nakano, Takayoshi Sakurai, Shun Shiramatsu, Toshitake Komazaki, Shiho Tsuchiya  

**Link**: [PDF](https://arxiv.org/pdf/2412.16943)  

**Abstract**: This study aims to improve the efficiency and quality of career interviews conducted by nursing managers. To this end, we have been developing a slot-filling dialogue system that engages in pre-interviews to collect information on staff careers as a preparatory step before the actual interviews. Conventional slot-filling-based interview dialogue systems have limitations in the flexibility of information collection because the dialogue progresses based on predefined slot sets. We therefore propose a method that leverages large language models (LLMs) to dynamically generate new slots according to the flow of the dialogue, achieving more natural conversations. Furthermore, we incorporate abduction into the slot generation process to enable more appropriate and effective slot generation. To validate the effectiveness of the proposed method, we conducted experiments using a user simulator. The results suggest that the proposed method using abduction is effective in enhancing both information-collecting capabilities and the naturalness of the dialogue. 

**Abstract (ZH)**: 本研究旨在提高护理管理人员进行职业生涯访谈的效率和质量。为此，我们正在开发一种对话系统，该系统通过预先访谈收集员工职业生涯信息，为实际访谈做准备。传统的基于槽填充的面试对话系统在信息收集的灵活性方面存在局限性，因为对话是基于预定义的槽集进行的。因此，我们提出了一种方法，利用大型语言模型（LLMs）根据对话的流程动态生成新的槽，以实现更自然的对话。此外，我们将可推断逻辑融入槽生成过程，以实现更具针对性和有效性的槽生成。为了验证所提方法的有效性，我们使用用户仿真器进行了实验。结果表明，结合可推断逻辑的所提方法在提高信息收集能力和对话自然度方面是有效的。 

---
# Prompting Large Language Models with Rationale Heuristics for Knowledge-based Visual Question Answering 

**Title (ZH)**: 使用合理性启发式方法提示大型语言模型进行基于知识的视觉问答 

**Authors**: Zhongjian Hu, Peng Yang, Bing Li, Fengyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16936)  

**Abstract**: Recently, Large Language Models (LLMs) have been used for knowledge-based Visual Question Answering (VQA). Despite the encouraging results of previous studies, prior methods prompt LLMs to predict answers directly, neglecting intermediate thought processes. We argue that prior methods do not sufficiently activate the capacities of LLMs. We propose a framework called PLRH that Prompts LLMs with Rationale Heuristics for knowledge-based VQA. The PLRH prompts LLMs with Chain of Thought (CoT) to generate rationale heuristics, i.e., intermediate thought processes, and then leverages the rationale heuristics to inspire LLMs to predict answers. Experiments show that our approach outperforms the existing baselines by more than 2.2 and 2.1 on OK-VQA and A-OKVQA, respectively. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）已被应用于基于知识的视觉问答（VQA）。尽管前期研究给出了令人鼓舞的结果，但先前的方法仅促使LLMs直接预测答案，而忽视了中间推理过程。我们主张先前的方法未能充分激活LLMs的能力。为此，我们提出了一种名为PLRH（Prompting LLMs with Rationale Heuristics）的框架。该框架通过链式思维（Chain of Thought，CoT）引导LLMs生成中间推理过程，即理性启发式，然后利用这些启发式来激发LLMs生成答案。实验结果显示，我们的方法在OK-VQA和A-OKVQA上的表现分别比现有基线高出2.2和2.1个百分点。 

---
# Revisiting In-Context Learning with Long Context Language Models 

**Title (ZH)**: 重新审视长上下文语言模型的在上下文学习 

**Authors**: Jinheon Baek, Sun Jae Lee, Prakhar Gupta, Geunseob, Siddharth Dalmia, Prateek Kolhar  

**Link**: [PDF](https://arxiv.org/pdf/2412.16926)  

**Abstract**: In-Context Learning (ICL) is a technique by which language models make predictions based on examples provided in their input context. Previously, their context window size imposed a limit on the number of examples that can be shown, making example selection techniques crucial for identifying the maximally effective set of examples. However, the recent advent of Long Context Language Models (LCLMs) has significantly increased the number of examples that can be included in context, raising an important question of whether ICL performance in a many-shot regime is still sensitive to the method of sample selection. To answer this, we revisit these approaches in the context of LCLMs through extensive experiments on 18 datasets spanning 4 tasks. Surprisingly, we observe that sophisticated example selection techniques do not yield significant improvements over a simple random sample selection method. Instead, we find that the advent of LCLMs has fundamentally shifted the challenge of ICL from that of selecting the most effective examples to that of collecting sufficient examples to fill the context window. Specifically, in certain datasets, including all available examples does not fully utilize the context window; however, by augmenting the examples in context with a simple data augmentation approach, we substantially improve ICL performance by 5%. 

**Abstract (ZH)**: 上下文学习（In-Context Learning，ICL）是一种语言模型根据输入上下文中的示例进行预测的技术。之前，语言模型的上下文窗口大小对其能呈现的示例数量设定了限制，因此选择有效示例的技术变得至关重要。然而，近期长上下文语言模型（Long Context Language Models，LCLMs）的出现大大增加了可以包含在上下文中的示例数量，这引发了关于在多示例情况下ICL性能是否仍然受到样本选择方法影响的重要问题。为回答这一问题，我们通过在18个数据集上进行广泛的实验（这些数据集涵盖了4个任务），重新审视了在LCLMs环境下的这些方法。令人惊讶的是，我们发现复杂的示例选择技术并没有比简单的随机示例选择方法显著提升ICL性能。相反，我们发现LCLMs的出现从根本上改变了ICL的挑战，从选择最有效的示例转变为收集足够的示例以填满上下文窗口。具体而言，在某些数据集中，包含所有可用的示例并未充分利用上下文窗口；但是，通过使用简单的数据增强方法补充上下文中的示例，我们能够显著提高ICL性能，提高幅度达到5%。 

---
# Unsupervised Bilingual Lexicon Induction for Low Resource Languages 

**Title (ZH)**: 低资源语言的无监督双语词典诱导 

**Authors**: Charitha Rathnayake, P.R.S. Thilakarathna, Uthpala Nethmini, Rishemjith Kaur, Surangika Ranathunga  

**Link**: [PDF](https://arxiv.org/pdf/2412.16894)  

**Abstract**: Bilingual lexicons play a crucial role in various Natural Language Processing tasks. However, many low-resource languages (LRLs) do not have such lexicons, and due to the same reason, cannot benefit from the supervised Bilingual Lexicon Induction (BLI) techniques. To address this, unsupervised BLI (UBLI) techniques were introduced. A prominent technique in this line is structure-based UBLI. It is an iterative method, where a seed lexicon, which is initially learned from monolingual embeddings is iteratively improved. There have been numerous improvements to this core idea, however they have been experimented with independently of each other. In this paper, we investigate whether using these techniques simultaneously would lead to equal gains. We use the unsupervised version of VecMap, a commonly used structure-based UBLI framework, and carry out a comprehensive set of experiments using the LRL pairs, English-Sinhala, English-Tamil, and English-Punjabi. These experiments helped us to identify the best combination of the extensions. We also release bilingual dictionaries for English-Sinhala and English-Punjabi. 

**Abstract (ZH)**: 双语词典在各种自然语言处理任务中扮演着关键角色。然而，许多低资源语言（LRLs）缺乏这样的词典，因此无法从监督双语词典诱导（BLI）技术中受益。为了解决这一问题，提出了无监督BLI（UBLI）技术。这条线中的一个突出技术是基于结构的无监督BLI。这是一种迭代方法，在这种方法中，初始从单语嵌入中学习到的一个种子词典会被逐步改进。尽管对这一核心理念进行了许多改进，但这些改进是独立进行的。在本文中，我们研究同时使用这些技术是否能带来同等的增益。我们使用了VecMap的无监督版本，这是一个常用的基于结构的UBLI框架，通过使用英语-僧伽罗语、英语-泰米尔语和英语-旁遮普语的双语词对进行了一系列全面的实验。这些实验帮助我们确定了最佳的技术组合。我们还发布了英语-僧伽罗语和英语-旁遮普语的双语词典。 

---
# Reconsidering SMT Over NMT for Closely Related Languages: A Case Study of Persian-Hindi Pair 

**Title (ZH)**: 重新评估紧密相关语言上的统计机器翻译与神经机器翻译：波斯语-印地语语对的案例研究 

**Authors**: Waisullah Yousofi, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2412.16877)  

**Abstract**: This paper demonstrates that Phrase-Based Statistical Machine Translation (PBSMT) can outperform Transformer-based Neural Machine Translation (NMT) in moderate-resource scenarios, specifically for structurally similar languages, like the Persian-Hindi pair. Despite the Transformer architecture's typical preference for large parallel corpora, our results show that PBSMT achieves a BLEU score of 66.32, significantly exceeding the Transformer-NMT score of 53.7 on the same dataset. Additionally, we explore variations of the SMT architecture, including training on Romanized text and modifying the word order of Persian sentences to match the left-to-right (LTR) structure of Hindi. Our findings highlight the importance of choosing the right architecture based on language pair characteristics and advocate for SMT as a high-performing alternative, even in contexts commonly dominated by NMT. 

**Abstract (ZH)**: 本文展示了在中等资源场景下，基于短语的统计机器翻译（PBSMT）可以在结构相似的语言（如波斯语-印地语对）中超越基于 Transformer 的神经机器翻译（NMT）。尽管 Transformer 架构通常偏好大规模并行语料库，但我们的结果表明，PBSMT 在相同数据集上的 BLEU 得分为 66.32，显著高于 Transformer-NMT 的 53.7。此外，我们还探索了统计机器翻译架构的变体，包括在罗马化文本上进行训练以及调整波斯语句子的词序以匹配印地语的左至右（LTR）结构。我们的研究结果突显了根据语言对的特点选择合适架构的重要性，并提倡在常用由 NMT 占主导的背景下，SMT 作为一种高性能的替代方案。 

---
# Teaching LLMs to Refine with Tools 

**Title (ZH)**: 教学术大型语言模型使用工具进行优化与细化 

**Authors**: Dian Yu, Yuheng Zhang, Jiahao Xu, Tian Liang, Linfeng Song, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16871)  

**Abstract**: Large language models (LLMs) can refine their responses based on feedback, enabling self-improvement through iterative training or test-time refinement. However, existing methods predominantly focus on refinement within the same reasoning format, which may lead to non-correcting behaviors. We propose CaP, a novel approach that uses external tools to refine chain-of-thought (CoT) responses generated by the same or other LLMs. CaP employs a two-stage training process: supervised fine-tuning followed by preference optimization with DPO variants. Our observations highlight the critical role of preference optimization in enabling effective refinement. Additionally, we compare several sampling strategies to leverage CoT and tools at inference time. Experimental results demonstrate CaP's potential for effective cross-reasoning refinement and efficient inference. 

**Abstract (ZH)**: 大语言模型（LLMs）可以根据反馈改进其响应，从而通过迭代训练或测试时改进来实现自我提升。然而，现有方法主要集中在同一种推理格式内的改进上，这可能会导致非纠正性行为。我们提出了一种名为CaP的新方法，该方法使用外部工具来精炼由同一个或其它LLM生成的链式思维（CoT）响应。CaP采用两阶段训练过程：监督微调，随后是使用DPO变体进行偏好优化。我们的观察结果强调了偏好优化在实现有效改进中的关键作用。此外，我们还比较了几种采样策略，以便在推理时利用CoT和工具。实验结果表明，CaP在实现有效的跨推理改进和高效推理方面具有潜力。 

---
# GME: Improving Universal Multimodal Retrieval by Multimodal LLMs 

**Title (ZH)**: GME：通过多模态LLM提高通用多模态检索性能 

**Authors**: Xin Zhang, Yanzhao Zhang, Wen Xie, Mingxin Li, Ziqi Dai, Dingkun Long, Pengjun Xie, Meishan Zhang, Wenjie Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16855)  

**Abstract**: Universal Multimodal Retrieval (UMR) aims to enable search across various modalities using a unified model, where queries and candidates can consist of pure text, images, or a combination of both. Previous work has attempted to adopt multimodal large language models (MLLMs) to realize UMR using only text data. However, our preliminary experiments demonstrate that more diverse multimodal training data can further unlock the potential of MLLMs. Despite its effectiveness, the existing multimodal training data is highly imbalanced in terms of modality, which motivates us to develop a training data synthesis pipeline and construct a large-scale, high-quality fused-modal training dataset. Based on the synthetic training data, we develop the General Multimodal Embedder (GME), an MLLM-based dense retriever designed for UMR. Furthermore, we construct a comprehensive UMR Benchmark (UMRB) to evaluate the effectiveness of our approach. Experimental results show that our method achieves state-of-the-art performance among existing UMR methods. Last, we provide in-depth analyses of model scaling, training strategies, and perform ablation studies on both the model and synthetic data. 

**Abstract (ZH)**: 通用多模态检索（UMR）旨在使用统一模型实现跨多种模态的搜索，其中查询和候选内容可以是纯文本、图像，或者二者的组合。先前的工作尝试使用仅基于文本的数据来实现这种通用多模态检索，采用多模态大型语言模型（MLLMs）。然而，初步实验表明，更多样化的多模态训练数据可以进一步挖掘MLLMs的潜力。尽管该方法在一定程度上有效，但现有的多模态训练数据在模态方面存在高度不平衡的问题，这促使我们开发一种训练数据合成管道，并构建了一个大规模、高质量的融合模态训练数据集。基于合成训练数据，我们开发了一种基于MLLM的密集检索器——通用多模态嵌入器（GME），专门用于UMR。此外，我们构建了一个全面的UMR基准（UMRB）来评估我们方法的有效性。实验结果显示，我们的方法在现有的UMR方法中达到了最先进的性能。最后，我们对模型缩放、训练策略进行了深入分析，并在模型和合成数据上进行了消融研究。 

---
# Sim911: Towards Effective and Equitable 9-1-1 Dispatcher Training with an LLM-Enabled Simulation 

**Title (ZH)**: Sim911：通过具备大语言模型功能的模拟技术朝着更有效和公平的9-1-1调度员培训方向努力 

**Authors**: Zirong Chen, Elizabeth Chason, Noah Mladenovski, Erin Wilson, Kristin Mullen, Stephen Martini, Meiyi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2412.16844)  

**Abstract**: Emergency response services are vital for enhancing public safety by safeguarding the environment, property, and human lives. As frontline members of these services, 9-1-1 dispatchers have a direct impact on response times and the overall effectiveness of emergency operations. However, traditional dispatcher training methods, which rely on role-playing by experienced personnel, are labor-intensive, time-consuming, and often neglect the specific needs of underserved communities. To address these challenges, we introduce Sim911, the first training simulation for 9-1-1 dispatchers powered by Large Language Models (LLMs). Sim911 enhances training through three key technical innovations: (1) knowledge construction, which utilizes archived 9-1-1 call data to generate simulations that closely mirror real-world scenarios; (2) context-aware controlled generation, which employs dynamic prompts and vector bases to ensure that LLM behavior aligns with training objectives; and (3) validation with looped correction, which filters out low-quality responses and refines the system performance. 

**Abstract (ZH)**: 应急响应服务对于提高公共安全至关重要，它通过保护环境、财产和人类生命来实现这一目标。作为这些服务的前线成员，9-1-1调度员直接影响响应时间和整体应急操作的有效性。然而，传统的调度员培训方法依赖于由经验丰富的人员进行的角色扮演，这种方法耗时费力，且往往忽略了服务不足社区的具体需求。为了解决这些问题，我们介绍Sim911，这是一种由大规模语言模型（LLMs）驱动的9-1-1调度员培训模拟，它是第一个此类模拟。Sim911通过三项关键技术革新增强了培训：（1）知识构建，利用存档的9-1-1呼叫数据生成模拟场景，这些场景能够准确反映现实生活中的情况；（2）情境感知控制生成，通过使用动态提示和向量基底来确保大规模语言模型的行为与培训目标一致；（3）循环校正验证，筛选低质量响应并优化系统性能。 

---
# Ask-Before-Detection: Identifying and Mitigating Conformity Bias in LLM-Powered Error Detector for Math Word Problem Solutions 

**Title (ZH)**: 在检测之前提问：识别并减轻基于大语言模型的数学应用题解决方案错误检测器中的遵从性偏差 

**Authors**: Hang Li, Tianlong Xu, Kaiqi Yang, Yucheng Chu, Yanling Chen, Yichi Song, Qingsong Wen, Hui Liu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16838)  

**Abstract**: The rise of large language models (LLMs) offers new opportunities for automatic error detection in education, particularly for math word problems (MWPs). While prior studies demonstrate the promise of LLMs as error detectors, they overlook the presence of multiple valid solutions for a single MWP. Our preliminary analysis reveals a significant performance gap between conventional and alternative solutions in MWPs, a phenomenon we term conformity bias in this work. To mitigate this bias, we introduce the Ask-Before-Detect (AskBD) framework, which generates adaptive reference solutions using LLMs to enhance error detection. Experiments on 200 examples of GSM8K show that AskBD effectively mitigates bias and improves performance, especially when combined with reasoning-enhancing techniques like chain-of-thought prompting. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的兴起为教育中的自动错误检测带来了新的机会，特别是在数学文字问题（MWPs）中。虽然先前的研究表明LLMs作为错误检测器的潜力巨大，但它们忽视了单一数学文字问题可能有多个有效解的事实。我们的初步分析揭示了MWPs中传统解法和替代解法之间显著的性能差异，这种现象在本文中我们将其称为一致性偏差。为了减轻这一偏差，我们引入了“检测前询问”（Ask-Before-Detect，简称AskBD）框架，利用LLMs生成适应性的参考解法，以增强错误检测的效果。实验结果显示，AskBD在减轻偏差和提高性能方面效果显著，特别是在结合如链式思考提示等增强推理的技术时表现尤为突出。 

---
# Quantum-Like Contextuality in Large Language Models 

**Title (ZH)**: 大型语言模型中的量子似的语境性 

**Authors**: Kin Ian Lo, Mehrnoosh Sadrzadeh, Shane Mansfield  

**Link**: [PDF](https://arxiv.org/pdf/2412.16806)  

**Abstract**: Contextuality is a distinguishing feature of quantum mechanics and there is growing evidence that it is a necessary condition for quantum advantage. In order to make use of it, researchers have been asking whether similar phenomena arise in other domains. The answer has been yes, e.g. in behavioural sciences. However, one has to move to frameworks that take some degree of signalling into account. Two such frameworks exist: (1) a signalling-corrected sheaf theoretic model, and (2) the Contextuality-by-Default (CbD) framework. This paper provides the first large scale experimental evidence for a yes answer in natural language. We construct a linguistic schema modelled over a contextual quantum scenario, instantiate it in the Simple English Wikipedia and extract probability distributions for the instances using the large language model BERT. This led to the discovery of 77,118 sheaf-contextual and 36,938,948 CbD contextual instances. We proved that the contextual instances came from semantically similar words, by deriving an equation between degrees of contextuality and Euclidean distances of BERT's embedding vectors. A regression model further reveals that Euclidean distance is indeed the best statistical predictor of contextuality. Our linguistic schema is a variant of the co-reference resolution challenge. These results are an indication that quantum methods may be advantageous in language tasks. 

**Abstract (ZH)**: 上下文性是量子力学的一个显著特征，越来越多的证据表明，它是实现量子优势的必要条件。为了利用这一特性，研究人员一直在询问其他领域是否存在类似现象。答案是肯定的，特别是在行为科学领域。但是，这需要采用一些信号传递的框架。目前存在两种这样的框架：（1）信号修正的层积范畴模型，和（2）默认上下文性（CbD）框架。本文提供了第一个大规模实验证据，证明在自然语言中存在上下文性现象的回答是肯定的。我们构建了一个基于上下文量子场景的语言模式，并在简体英语维基百科中实例化该模式，利用大规模语言模型BERT提取实例的概率分布。这一过程发现了77,118个层积上下文性和36,938,948个CbD上下文性实例。我们通过推导上下文性的程度和BERT嵌入向量欧几里得距离之间的方程，证明了上下文性实例来自语义相似的单词。进一步的回归模型表明，欧几里得距离确实是上下文性最好的统计预测器。我们的语言模式是一种同名词义消解挑战的变体。这些结果表明，量子方法可能在语言任务中具有优势。 

---
# SubData: A Python Library to Collect and Combine Datasets for Evaluating LLM Alignment on Downstream Tasks 

**Title (ZH)**: SubData：一个用于评估大型语言模型在下游任务中一致性调优的数据收集和组合Python库 

**Authors**: Leon Fröhling, Pietro Bernardelle, Gianluca Demartini  

**Link**: [PDF](https://arxiv.org/pdf/2412.16783)  

**Abstract**: With the release of ever more capable large language models (LLMs), researchers in NLP and related disciplines have started to explore the usability of LLMs for a wide variety of different annotation tasks. Very recently, a lot of this attention has shifted to tasks that are subjective in nature. Given that the latest generations of LLMs have digested and encoded extensive knowledge about different human subpopulations and individuals, the hope is that these models can be trained, tuned or prompted to align with a wide range of different human perspectives. While researchers already evaluate the success of this alignment via surveys and tests, there is a lack of resources to evaluate the alignment on what oftentimes matters the most in NLP; the actual downstream tasks. To fill this gap we present SubData, a Python library that offers researchers working on topics related to subjectivity in annotation tasks a convenient way of collecting, combining and using a range of suitable datasets. 

**Abstract (ZH)**: 随着功能日益强大的大规模语言模型（LLMs）的发布，自然语言处理（NLP）及相关学科的研究人员已经开始探索在各种不同的标注任务中使用LLMs的可行性。最近，越来越多的研究关注主观性较强的任务。鉴于最新一代的LLMs已经消化了大量的关于不同人类亚群体和个人的知识，希望这些模型能够被训练、调优或提示，以匹配广泛的人类视角。尽管研究人员已经通过调查和测试来评估这种对齐的成功程度，但在实际的下游任务中，这种对齐的重要性往往更为关键。为了弥补这一不足，我们提出了SubData，这是一个Python库，为专注于标注任务中主观性相关课题的研究人员提供了一种方便的方法来收集、组合和使用一系列合适的数据集。 

---
# DragonVerseQA: Open-Domain Long-Form Context-Aware Question-Answering 

**Title (ZH)**: DragonVerseQA：开放领域长文本上下文感知问答 

**Authors**: Aritra Kumar Lahiri, Qinmin Vivian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16694)  

**Abstract**: This paper proposes a novel approach to develop an open-domain and long-form Over-The-Top (OTT) Question-Answering (QA) dataset, DragonVerseQA, specifically oriented to the fantasy universe of "House of the Dragon" and "Game Of Thrones" TV series. Most existing QA datasets focus on short, fact-based answers sourced almost solely from Wikipedia articles, devoid of depth and contextual richness for sophisticated narrative understanding. We curate a dataset that combines full episode summaries sourced from HBO and fandom wiki websites, user reviews from sources like IMDb and Rotten Tomatoes, and high-quality, open-domain, legally admissible sources, and structured data from repositories like WikiData into one dataset. The dataset provides a multi-dimensional context, reflecting complex character dynamics and plot developments from these varied sources. That means, on equal footing, only after heavy data preprocessing and filtering methods will meaningful, non-spam unbiased reviews be available in this enriched dataset. The comprehensive insights are given through the long-form answers generated from this enriched context. This is what makes this valuable dataset for improving conversational AI, narrative analysis, sentiment analysis, summarization techniques, and relation extraction.
A comparative analysis with state-of-the-art QA datasets such as SQuAD 2.0, TriviaQA, and Natural Questions brings to light the unique advantages of our dataset in terms of contextual complexity and answer length. Detailed reviews add layers to audience sentiment and narrative interpretation, raising the bar for domain-specific QA with a new quality benchmark. Our work also allows a deeper understanding of entertainment-industry content and opens the door to more knowledgeable and creative AI-driven interactions within digital media environments. 

**Abstract (ZH)**: 本文提出了一种新颖的方法，用于开发面向“权力的游戏”和“龙之堡”电视剧集幻想宇宙的开放领域和长格式Over-The-Top (OTT)问答数据集——DragonVerseQA。现存大多数问答数据集侧重于来自维基百科文章的简短事实性答案，缺乏深度和上下文丰富性，无法支持复杂叙事的理解。我们构建了一个结合了HBO和粉丝维基网站的全集摘要、来自IMDb和烂番茄等网站的用户评论、高质量的开放领域合法来源，以及来自WikiData等存储库的结构化数据的数据集。该数据集提供多维度上下文，反映了从各种来源中得出的复杂人物动态和剧情发展。这意味着，在经过重度数据预处理和筛选方法之后，这个丰富数据集才能提供有意义、非垃圾评论且无偏见的答案。通过这种丰富背景生成的长格式答案，提供了全面的洞察。这使得这个有价值的数据集能够提升对话式AI、叙事分析、情感分析、总结技术以及关系提取。

将现有最先进的问答数据集如SQuAD 2.0、TriviaQA和自然问题等进行比较分析，表明我们的数据集在上下文复杂性和答案长度方面具有独特优势。详细的评论增加了观众情感和叙事解释的多层性，为特定领域的问答设置了一个新的质量基准。我们的工作还允许更深入地理解娱乐产业内容，并为数字媒体环境中更加丰富和创造性的AI驱动交互打开了大门。 

---
# NILE: Internal Consistency Alignment in Large Language Models 

**Title (ZH)**: NILE：大型语言模型内的一致性对齐 

**Authors**: Minda Hu, Qiyuan Zhang, Yufei Wang, Bowei He, Hongru Wang, Jingyan Zhou, Liangyou Li, Yasheng Wang, Chen Ma, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2412.16686)  

**Abstract**: As a crucial step to enhance LLMs alignment with human intentions, Instruction Fine-Tuning (IFT) has a high demand on dataset quality. However, existing IFT datasets often contain knowledge that is inconsistent with LLMs' internal knowledge learned from the pre-training phase, which can greatly affect the efficacy of IFT. To address this issue, we introduce NILE (iNternal consIstency aLignmEnt) framework, aimed at optimizing IFT datasets to unlock LLMs' capability further. NILE operates by eliciting target pre-trained LLM's internal knowledge corresponding to instruction data. The internal knowledge is leveraged to revise the answer in IFT datasets. Additionally, we propose a novel Internal Consistency Filtering (ICF) method to filter training samples, ensuring its high consistency with LLM's internal knowledge. Our experiments demonstrate that NILE-aligned IFT datasets sharply boost LLM performance across multiple LLM ability evaluation datasets, achieving up to 66.6% gain on Arena-Hard and 68.5% on Alpaca-Eval V2. Further analysis confirms that each component of the NILE}framework contributes to these substantial performance improvements, and provides compelling evidence that dataset consistency with pre-trained internal knowledge is pivotal for maximizing LLM potential. 

**Abstract (ZH)**: 为了增强大模型（LLMs）与人类意图的一致性，指令微调（IFT）对数据集的质量有着很高的要求。然而，现有的IFT数据集往往包含与预训练阶段学习的知识不一致的知识，这严重影响了IFT的效果。为了解决这一问题，我们提出了NILE（Internal Consistency Alignment）框架，旨在优化IFT数据集，进一步发挥大模型的能力。NILE通过提取目标预训练大模型内部与指令数据相对应的知识来实现这一目标。这些内部知识被用于修正IFT数据集中的答案。此外，我们还提出了一种新的内部一致性筛选（ICF）方法来筛选训练样本，确保其与大模型内部知识的高度一致性。我们的实验表明，经过NILE优化的IFT数据集在多个LLM能力评估数据集上显著提升了大模型的性能，在Arena-Hard数据集上达到了66.6%的提升，在Alpaca-Eval V2数据集上达到了68.5%的提升。进一步的分析证实，NILE框架的每个组成部分都对这些显著的性能提升做出了贡献，并提供了有力证据，表明数据集与预训练内部知识的一致性对于最大化大模型的潜力至关重要。 

---
# L3TC: Leveraging RWKV for Learned Lossless Low-Complexity Text Compression 

**Title (ZH)**: L3TC：利用RWKV进行学习驱动的低复杂度无损文本压缩

解释：
- L3TC 是论文标题的简写形式，保持不变。
- "Leveraging" 翻译为“利用”，更符合学术表达。
- "RWKV" 是特定模型的名称，保持不变。
- "for Learned Lossless Low-Complexity Text Compression" 翻译为“进行学习驱动的低复杂度无损文本压缩”，其中“无损”指的是压缩和解压缩后文本不丢失信息，“低复杂度”指的是算法计算量较小。 

**Authors**: Junxuan Zhang, Zhengxue Cheng, Yan Zhao, Shihao Wang, Dajiang Zhou, Guo Lu, Li Song  

**Link**: [PDF](https://arxiv.org/pdf/2412.16642)  

**Abstract**: Learning-based probabilistic models can be combined with an entropy coder for data compression. However, due to the high complexity of learning-based models, their practical application as text compressors has been largely overlooked. To address this issue, our work focuses on a low-complexity design while maintaining compression performance. We introduce a novel Learned Lossless Low-complexity Text Compression method (L3TC). Specifically, we conduct extensive experiments demonstrating that RWKV models achieve the fastest decoding speed with a moderate compression ratio, making it the most suitable backbone for our method. Second, we propose an outlier-aware tokenizer that uses a limited vocabulary to cover frequent tokens while allowing outliers to bypass the prediction and encoding. Third, we propose a novel high-rank reparameterization strategy that enhances the learning capability during training without increasing complexity during inference. Experimental results validate that our method achieves 48\% bit saving compared to gzip compressor. Besides, \emph{L3TC} offers compression performance comparable to other learned compressors, with a $50\times$ reduction in model parameters. More importantly, \emph{L3TC} is the fastest among all learned compressors, providing real-time decoding speeds up to megabytes per second. 

**Abstract (ZH)**: 基于学习的概率模型可以与熵编码器结合进行数据压缩。然而，由于基于学习的模型具有高度复杂性，它们在作为文本压缩器的实际应用中一直被忽视。为解决这一问题，我们的研究重点关注低复杂度设计的同时保持压缩性能。我们提出了一种新颖的低复杂度无损文本压缩方法（L3TC）。具体来说，我们在广泛的实验中证明，RWKV模型在维持适度压缩比的同时实现了最快的解码速度，使其成为我们方法的最佳骨干模型。其次，我们提出了一种带有异常值感知的分词器，该分词器使用有限的词汇表覆盖常见词汇，同时允许异常值绕过预测和编码。第三，我们提出了一种新颖的高秩重参数化策略，在训练期间增强学习能力，同时在推理期间不增加复杂性。实验结果表明，我们的方法在与gzip压缩器相比时，可实现48%的位数节省。此外，L3TC在压缩性能方面与其它学习压缩器相当，但其模型参数量减少了50倍。更为重要的是，L3TC在所有学习压缩器中速度最快，提供了每秒兆字节级别的实时解码速度。 

---
# Acquisition of Recursive Possessives and Recursive Locatives in Mandarin 

**Title (ZH)**: Mandarin 中递归共有关系和递归方位关系的获得研究 

**Authors**: Chenxi Fu, Xiaoyi Wang, Zaijiang Man, Caimei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16556)  

**Abstract**: As recursion has been underlying any linguistic work for the last 60 years, the acquisition of recursive structures by children during language learning has become a focal point of inquiry. This study delves into the developmental trajectory of Mandarin-speaking children's acquisition of recursive possessives and locatives, assessing the impact of structural diversity on language acquisition. The research contrasts the comprehension of two-level recursive structures among children aged 3 to 7 years, employing answering question while seeing a picture task to elicit responses. The findings indicate that children do not attain adult-like proficiency in two-level recursion until the age of 6, and there exists a notable asymmetry in the acquisition of recursive possessives versus locatives. These results underscore the primacy of structural complexity and cognitive factors in the acquisition process, enhancing our comprehension of the cognitive foundations of language development and the pivotal role of recursion in child language acquisition. 

**Abstract (ZH)**: 在过去60年中，递归已经成为任何语言研究的基础。因此，儿童在语言学习过程中获得递归结构的能力成为了研究的重点。本研究探讨了说普通话的儿童在获得递归所有格和位置结构方面的发育轨迹，并评估了结构多样性对语言习得的影响。研究通过比较3至7岁儿童对两层递归结构的理解，使用看图回答问题的任务来获取他们的反应。研究结果表明，儿童通常要在6岁时才能达到成人的水平，并且在获得递归所有格和位置结构方面存在明显的不对称性。这些结果强调了结构复杂性和认知因素在习得过程中的重要性，加深了我们对语言发展认知基础及递归在儿童语言习得中的关键作用的理解。 

---
# Divide and Conquer: A Hybrid Strategy Defeats Multimodal Large Language Models 

**Title (ZH)**: 分而治之：一种混合策略战胜多模态大型语言模型 

**Authors**: Yanxu Mao, Peipei Liu, Tiehan Cui, Congying Liu, Datao You  

**Link**: [PDF](https://arxiv.org/pdf/2412.16555)  

**Abstract**: Large language models (LLMs) are widely applied in various fields of society due to their powerful reasoning, understanding, and generation capabilities. However, the security issues associated with these models are becoming increasingly severe. Jailbreaking attacks, as an important method for detecting vulnerabilities in LLMs, have been explored by researchers who attempt to induce these models to generate harmful content through various attack methods. Nevertheless, existing jailbreaking methods face numerous limitations, such as excessive query counts, limited coverage of jailbreak modalities, low attack success rates, and simplistic evaluation methods. To overcome these constraints, this paper proposes a multimodal jailbreaking method: JMLLM. This method integrates multiple strategies to perform comprehensive jailbreak attacks across text, visual, and auditory modalities. Additionally, we contribute a new and comprehensive dataset for multimodal jailbreaking research: TriJail, which includes jailbreak prompts for all three modalities. Experiments on the TriJail dataset and the benchmark dataset AdvBench, conducted on 13 popular LLMs, demonstrate advanced attack success rates and significant reduction in time overhead. 

**Abstract (ZH)**: 大规模语言模型（LLMs）因其强大的推理、理解和生成能力，在社会的各个领域得到了广泛应用。然而，与这些模型相关的安全问题日益严重。作为检测LLMs漏洞的重要方法，破戒攻击（Jailbreaking attacks）已经引起了研究人员的关注。研究人员通过各种攻击方法试图诱导这些模型生成有害内容。尽管如此，现有的破戒方法仍然存在诸多限制，例如查询次数过多、破戒模态覆盖不足、攻击成功率低以及简单的评估方法。为了克服这些限制，本文提出了一个多模态破戒方法：JMLLM。该方法结合了多种策略，在文本、视觉和听觉等多种模态下进行全面的破戒攻击。此外，我们还贡献了一个全新的、全面的多模态破戒数据集：TriJail，该数据集包含了所有三种模态的破戒提示。在TriJail数据集和基准数据集AdvBench上针对13个流行的LLM进行的实验显示，该方法具有较高的攻击成功率，并显著减少了时间开销。 

---
# Attention Entropy is a Key Factor: An Analysis of Parallel Context Encoding with Full-attention-based Pre-trained Language Models 

**Title (ZH)**: 注意力熵是关键因素：基于全注意机制的预训练语言模型并行上下文编码分析 

**Authors**: Zhisong Zhang, Yan Wang, Xinting Huang, Tianqing Fang, Hongming Zhang, Chenlong Deng, Shuaiyi Li, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16545)  

**Abstract**: Large language models have shown remarkable performance across a wide range of language tasks, owing to their exceptional capabilities in context modeling. The most commonly used method of context modeling is full self-attention, as seen in standard decoder-only Transformers. Although powerful, this method can be inefficient for long sequences and may overlook inherent input structures. To address these problems, an alternative approach is parallel context encoding, which splits the context into sub-pieces and encodes them parallelly. Because parallel patterns are not encountered during training, naively applying parallel encoding leads to performance degradation. However, the underlying reasons and potential mitigations are unclear. In this work, we provide a detailed analysis of this issue and identify that unusually high attention entropy can be a key factor. Furthermore, we adopt two straightforward methods to reduce attention entropy by incorporating attention sinks and selective mechanisms. Experiments on various tasks reveal that these methods effectively lower irregular attention entropy and narrow performance gaps. We hope this study can illuminate ways to enhance context modeling mechanisms. 

**Abstract (ZH)**: 大型语言模型在广泛的语言任务上展现出卓越的性能，这归功于它们在上下文建模方面的出色能力。最常用的上下文建模方法是全自注意力机制，这在标准的解码器架构Transformer中得到广泛应用。尽管这种方法非常强大，但它对于长序列来说可能不够高效，并且可能会忽略输入的固有结构。为了解决这些问题，一种替代的方法是并行上下文编码，它将上下文划分为子片段并并行编码。由于训练过程中不会遇到并行模式，直接应用并行编码会导致性能下降。然而，背后的原因和潜在的缓解措施尚不清楚。在本文中，我们对这一问题进行了详细的分析，并发现异常高的注意力熵可能是关键因素。此外，我们采用了两种简单的方法来降低注意力熵，这些方法通过引入注意力吸收机制和选择性机制来实现。在各种任务上的实验表明，这些方法能够有效降低不规则的注意力熵并缩小性能差距。我们希望这项研究能为改进上下文建模机制提供新的启示。 

---
# HammerBench: Fine-Grained Function-Calling Evaluation in Real Mobile Device Scenarios 

**Title (ZH)**: HammerBench：在实际移动设备场景中的细粒度函数调用评估 

**Authors**: Jun Wang, Jiamu Zhou, Muning Wen, Xiaoyun Mo, Haoyu Zhang, Qiqiang Lin, Cheng Jin, Xihuai Wang, Weinan Zhang, Qiuying Peng, Jun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16516)  

**Abstract**: Evaluating the capabilities of large language models (LLMs) in human-LLM interactions remains challenging due to the inherent complexity and openness of dialogue processes. This paper introduces HammerBench, a novel benchmarking framework designed to assess the function-calling ability of LLMs more effectively in such interactions. We model a wide range of real-world user scenarios on mobile devices, encompassing imperfect instructions, diverse question-answer trajectories, intent/argument shifts, and the use of external individual information through pronouns. To construct the corresponding datasets, we propose a comprehensive pipeline that involves LLM-generated data and multiple rounds of human validation, ensuring high data quality. Additionally, we decompose the conversations into function-calling snapshots, enabling a fine-grained evaluation of each turn. We evaluate several popular LLMs using HammerBench and highlight different performance aspects. Our empirical findings reveal that errors in parameter naming constitute the primary factor behind conversation failures across different data types. 

**Abstract (ZH)**: 在人类与大规模语言模型（LLMs）交互过程中评估LLMs的能力仍然具有挑战性，这主要归因于对话过程的内在复杂性和开放性。本文介绍了一种名为HammerBench的新颖基准评测框架，旨在更有效地评估LLMs在交互中的函数调用能力。我们针对移动设备上的一系列实际用户场景进行了建模，涵盖了不完善的指令、多样的问答路径、意图或论点的转变以及通过代词使用外部个体信息。为了构建相应的数据集，我们提出了一种全面的工作流程，该流程包括LLM生成的数据以及多轮的人工验证，以确保数据的质量。此外，我们将对话分解为函数调用快照，从而实现对每个回合的精细化评估。我们使用HammerBench评估了几种流行的LLMs，并突显了不同的性能方面。我们的实证研究发现，参数命名错误是不同类型数据中对话失败的主要原因。 

---
# Adapting Whisper for Code-Switching through Encoding Refining and Language-Aware Decoding 

**Title (ZH)**: 通过编码优化和语言感知解码适应 Whisper 于代码切换场景 

**Authors**: Jiahui Zhao, Hao Shi, Chenrui Cui, Tianrui Wang, Hexin Liu, Zhaoheng Ni, Lingxuan Ye, Longbiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16507)  

**Abstract**: Code-switching (CS) automatic speech recognition (ASR) faces challenges due to the language confusion resulting from accents, auditory similarity, and seamless language switches. Adaptation on the pre-trained multi-lingual model has shown promising performance for CS-ASR. In this paper, we adapt Whisper, which is a large-scale multilingual pre-trained speech recognition model, to CS from both encoder and decoder parts. First, we propose an encoder refiner to enhance the encoder's capacity of intra-sentence swithching. Second, we propose using two sets of language-aware adapters with different language prompt embeddings to achieve language-specific decoding information in each decoder layer. Then, a fusion module is added to fuse the language-aware decoding. The experimental results using the SEAME dataset show that, compared with the baseline model, the proposed approach achieves a relative MER reduction of 4.1% and 7.2% on the dev_man and dev_sge test sets, respectively, surpassing state-of-the-art methods. Through experiments, we found that the proposed method significantly improves the performance on non-native language in CS speech, indicating that our approach enables Whisper to better distinguish between the two languages. 

**Abstract (ZH)**: 由于口音、听觉相似性和无缝语言转换导致的语言混淆，代码混用（CS）自动语音识别（ASR）面临着挑战。针对这些挑战，预训练多语言模型的适应性调整已经显示出在CS-ASR中的良好性能。本文在Whisper（一种大规模多语言预训练语音识别模型）的编码器和解码器部分对其进行了适应性调整。具体而言，首先我们提出了一种编码器精炼方法，以增强编码器的句内转换能力。其次，我们提出使用两组语言感知适配器，并采用不同的语言提示嵌入，在每个解码器层中实现特定语言的解码信息。然后增加了融合模块，以融合语言感知的解码信息。使用SEAME数据集进行的实验结果显示，与基线模型相比，所提出的方法分别在dev_man和dev_sge测试集上实现了MER降低4.1%和7.2%，超过了现有最好的方法。通过实验，我们发现所提出的方法显著提高了Whisper在CS语音中的非母语语言性能，表明我们的方法使Whisper能够更好地区分两种语言。 

---
# Real-time Bangla Sign Language Translator 

**Title (ZH)**: 实时孟加拉手语翻译系统 

**Authors**: Rotan Hawlader Pranto, Shahnewaz Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2412.16497)  

**Abstract**: The human body communicates through various meaningful gestures, with sign language using hands being a prominent example. Bangla Sign Language Translation (BSLT) aims to bridge communication gaps for the deaf and mute community. Our approach involves using Mediapipe Holistic to gather key points, LSTM architecture for data training, and Computer Vision for realtime sign language detection with an accuracy of 94%. Keywords=Recurrent Neural Network, LSTM, Computer Vision, Bangla font. 

**Abstract (ZH)**: 人类通过各种有意义的手势进行沟通，而手势语言正是其中的突出代表。邦拉手语翻译（BSLT）旨在弥合聋哑人群体的沟通障碍。我们的方法包括使用Mediapipe Holistic收集关键点、使用LSTM架构进行数据训练，并利用计算机视觉实现94%准确率的实时手语识别。关键词：循环神经网络（RNN）、LSTM、计算机视觉、邦拉字体。 

---
# Evaluating the Performance of Large Language Models in Scientific Claim Detection and Classification 

**Title (ZH)**: 评估大型语言模型在科学研究声明检测与分类中的性能 

**Authors**: Tanjim Bin Faruk  

**Link**: [PDF](https://arxiv.org/pdf/2412.16486)  

**Abstract**: The pervasive influence of social media during the COVID-19 pandemic has been a double-edged sword, enhancing communication while simultaneously propagating misinformation. This \textit{Digital Infodemic} has highlighted the urgent need for automated tools capable of discerning and disseminating factual content. This study evaluates the efficacy of Large Language Models (LLMs) as innovative solutions for mitigating misinformation on platforms like Twitter. LLMs, such as OpenAI's GPT and Meta's LLaMA, offer a pre-trained, adaptable approach that bypasses the extensive training and overfitting issues associated with traditional machine learning models. We assess the performance of LLMs in detecting and classifying COVID-19-related scientific claims, thus facilitating informed decision-making. Our findings indicate that LLMs have significant potential as automated fact-checking tools, though research in this domain is nascent and further exploration is required. We present a comparative analysis of LLMs' performance using a specialized dataset and propose a framework for their application in public health communication. 

**Abstract (ZH)**: 新冠疫情期间社交媒体的普遍影响是一把双刃剑，既增强了沟通，又传播了谬误信息。这一“数字信息疫情”突显了迫切需要能够识别和传播准确信息的自动化工具。本研究评估了大语言模型（LLMs）作为平台（如推特）上遏制谬误信息的创新解决方案的有效性。大语言模型，如OpenAI的GPT和Meta的LLaMA，提供了一种预训练、可适应的方法，绕过了传统机器学习模型中的大量训练和过拟合问题。我们评估了LLMs在检测和分类与新冠相关的科学声明方面的性能，从而促进了明智决策的制定。我们的研究表明，LLMs在自动化事实核查方面具有巨大的潜力，尽管该领域的研究尚处于初级阶段，仍需进一步探索。我们使用专门的数据集对LLMs的性能进行了比较分析，并提出了一种将其应用于公共卫生沟通的框架。 

---
# Chained Tuning Leads to Biased Forgetting 

**Title (ZH)**: 链式调整会导致有偏见的遗忘 

**Authors**: Megan Ung, Alicia Sun, Samuel J. Bell, Bhaktipriya Radharapu, Levent Sagun, Adina Williams  

**Link**: [PDF](https://arxiv.org/pdf/2412.16469)  

**Abstract**: Large language models (LLMs) are often fine-tuned for use on downstream tasks, though this can degrade capabilities learned during previous training. This phenomenon, often referred to as catastrophic forgetting, has important potential implications for the safety of deployed models. In this work, we first show that models trained on downstream tasks forget their safety tuning to a greater extent than models trained in the opposite this http URL, we show that forgetting disproportionately impacts safety information about certain groups. To quantify this phenomenon, we define a new metric we term biased forgetting. We conduct a systematic evaluation of the effects of task ordering on forgetting and apply mitigations that can help the model recover from the forgetting observed. We hope our findings can better inform methods for chaining the finetuning of LLMs in continual learning settings to enable training of safer and less toxic models. 

**Abstract (ZH)**: 大语言模型（LLMs）通常针对下游任务进行微调，但这可能会削弱之前训练中学习到的能力。这一现象通常被称为灾难性遗忘，这对部署模型的安全性具有重要的潜在影响。在本项研究中，我们首先表明，针对下游任务训练的模型比反向训练的模型在更大程度上忘记了其安全性调整。其次，我们发现遗忘在很大程度上影响了某些群体的安全信息。为量化这一现象，我们定义了一个新的度量标准，称之为有偏遗忘。我们系统地评估了任务顺序对遗忘的影响，并应用了可以帮助模型从观察到的遗忘中恢复的缓解措施。我们希望这些发现能够更好地指导在连续学习环境中链式微调LLMs的方法，以实现训练更安全、更无毒模型的目的。 

---
# Transducer-Llama: Integrating LLMs into Streamable Transducer-based Speech Recognition 

**Title (ZH)**: Transducer-Llama：将大规模语言模型集成到可流式处理的转换器声学模型中进行语音识别 

**Authors**: Keqi Deng, Jinxi Guo, Yingyi Ma, Niko Moritz, Philip C. Woodland, Ozlem Kalinli, Mike Seltzer  

**Link**: [PDF](https://arxiv.org/pdf/2412.16464)  

**Abstract**: While large language models (LLMs) have been applied to automatic speech recognition (ASR), the task of making the model streamable remains a challenge. This paper proposes a novel model architecture, Transducer-Llama, that integrates LLMs into a Factorized Transducer (FT) model, naturally enabling streaming capabilities. Furthermore, given that the large vocabulary of LLMs can cause data sparsity issue and increased training costs for spoken language systems, this paper introduces an efficient vocabulary adaptation technique to align LLMs with speech system vocabularies. The results show that directly optimizing the FT model with a strong pre-trained LLM-based predictor using the RNN-T loss yields some but limited improvements over a smaller pre-trained LM predictor. Therefore, this paper proposes a weak-to-strong LM swap strategy, using a weak LM predictor during RNN-T loss training and then replacing it with a strong LLM. After LM replacement, the minimum word error rate (MWER) loss is employed to finetune the integration of the LLM predictor with the Transducer-Llama model. Experiments on the LibriSpeech and large-scale multi-lingual LibriSpeech corpora show that the proposed streaming Transducer-Llama approach gave a 17% relative WER reduction (WERR) over a strong FT baseline and a 32% WERR over an RNN-T baseline. 

**Abstract (ZH)**: 尽管大型语言模型（LLMs）已被应用于自动语音识别（ASR），使模型支持流式处理任务仍然存在挑战。本文提出了一种新颖的模型架构，即Transducer-Llama，将LLMs集成到因子转换器（FT）模型中，自然地实现了流式处理能力。此外，鉴于LLMs的大型词汇表会导致语音系统中的数据稀疏问题和更高的训练成本，本文引入了一种高效的语言模型适应技术，将LLMs与语音系统的词汇表对齐。结果表明，直接使用强预训练的LLM基预测器优化FT模型，使用RNN-T损失进行优化，相对于较小的预训练语言模型（LM）预测器仅有有限的改进。因此，本文提出了一个从弱到强的语言模型替换策略，在RNN-T损失训练期间使用弱LM预测器，然后将其替换为强LLM。在LM替换后，使用最小字错误率（MWER）损失对LLM预测器与Transducer-Llama模型的结合进行微调。在LibriSpeech和大规模多语言LibriSpeech语料库上的实验表明，所提出的流式Transducer-Llama方法与强FT基准相比降低了17%的相对字错误率（WERR），与RNN-T基准相比降低了32%的WERR。 

---
# Research on Violent Text Detection System Based on BERT-fasttext Model 

**Title (ZH)**: 基于BERT-fastText模型的暴力文本检测系统研究 

**Authors**: Yongsheng Yang, Xiaoying Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16455)  

**Abstract**: In the digital age of today, the internet has become an indispensable platform for people's lives, work, and information exchange. However, the problem of violent text proliferation in the network environment has arisen, which has brought about many negative effects. In view of this situation, it is particularly important to build an effective system for cutting off violent text. The study of violent text cutting off based on the BERT-fasttext model has significant meaning. BERT is a pre-trained language model with strong natural language understanding ability, which can deeply mine and analyze text semantic information; Fasttext itself is an efficient text classification tool with low complexity and good effect, which can quickly provide basic judgments for text processing. By combining the two and applying them to the system for cutting off violent text, on the one hand, it can accurately identify violent text, and on the other hand, it can efficiently and reasonably cut off the content, preventing harmful information from spreading freely on the network. Compared with the single BERT model and fasttext, the accuracy was improved by 0.7% and 0.8%, respectively. Through this model, it is helpful to purify the network environment, maintain the health of network information, and create a positive, civilized, and harmonious online communication space for netizens, driving the development of social networking, information dissemination, and other aspects in a more benign direction. 

**Abstract (ZH)**: 在当今数字化时代，互联网已成为人们生活、工作和信息交流不可或缺的平台。然而，在网络环境中，暴力文本的泛滥问题日益凸显，给社会带来了许多负面影响。鉴于此，构建有效的暴力文本过滤系统尤为重要。基于BERT和FastText模型进行暴力文本过滤的研究具有重要意义。BERT是一种预训练的语言模型，具有强大的自然语言理解能力，能够深入挖掘和分析文本语义信息；FastText本身是一种高效且效果良好的文本分类工具，具有较低的复杂度，能够快速提供文本处理的基本判断。通过将两者结合，并应用于暴力文本过滤系统，一方面能够准确识别暴力文本，另一方面能够高效且合理地过滤内容，防止有害信息在网络上自由传播。与单独使用BERT模型和FastText相比，其准确率分别提高了0.7%和0.8%。通过这种方式，有助于净化网络环境，维护网络信息健康，为网民创造一个积极、文明、和谐的网络交流空间，从而推动社交网络、信息传播等方面的良性发展。 

---
# Effective Context Modeling Framework for Emotion Recognition in Conversations 

**Title (ZH)**: 有效的对话情感识别情境建模框架 

**Authors**: Cuong Tran Van, Thanh V. T. Tran, Van Nguyen, Truong Son Hy  

**Link**: [PDF](https://arxiv.org/pdf/2412.16444)  

**Abstract**: Emotion Recognition in Conversations (ERC) facilitates a deeper understanding of the emotions conveyed by speakers in each utterance within a conversation. Recently, Graph Neural Networks (GNNs) have demonstrated their strengths in capturing data relationships, particularly in contextual information modeling and multimodal fusion. However, existing methods often struggle to fully capture the complex interactions between multiple modalities and conversational context, limiting their expressiveness. To overcome these limitations, we propose ConxGNN, a novel GNN-based framework designed to capture contextual information in conversations. ConxGNN features two key parallel modules: a multi-scale heterogeneous graph that captures the diverse effects of utterances on emotional changes, and a hypergraph that models the multivariate relationships among modalities and utterances. The outputs from these modules are integrated into a fusion layer, where a cross-modal attention mechanism is applied to produce a contextually enriched representation. Additionally, ConxGNN tackles the challenge of recognizing minority or semantically similar emotion classes by incorporating a re-weighting scheme into the loss functions. Experimental results on the IEMOCAP and MELD benchmark datasets demonstrate the effectiveness of our method, achieving state-of-the-art performance compared to previous baselines. 

**Abstract (ZH)**: 对话中情绪识别（Emotion Recognition in Conversations, ERC）有助于更深入地理解对话中每个发言所传达的情感。近年来，图神经网络（Graph Neural Networks, GNNs）已经在捕捉数据关系方面显示出了强大力量，特别是在上下文信息建模和多模态融合方面。然而，现有方法往往难以完全捕捉多模态之间及对话上下文中的复杂交互，限制了其表现力。为克服这些限制，我们提出了一种名为ConxGNN的新型GNN基框架，旨在捕捉对话中的上下文信息。ConxGNN包含两个关键的并行模块：一个多尺度异构图，用于捕捉不同发言对情感变化的多样影响；以及一个超图，用于建模模态之间和发言之间的多元关系。来自这些模块的输出被整合到一个融合层中，在该层中应用了一种跨模态注意力机制，产生一种丰富上下文的信息表示。此外，ConxGNN通过将加权方案引入损失函数中，解决了识别少数类或语义相似的情感类别这一挑战。在IEMOCAP和MELD基准数据集上的实验结果表明，我们的方法有效且表现出色，与之前的基线方法相比达到了最先进的性能。 

---
# Technical Report: Small Language Model for Japanese Clinical and Medicine 

**Title (ZH)**: 技术报告：用于日语临床和医学的小型语言模型 

**Authors**: Shogo Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2412.16423)  

**Abstract**: This report presents a small language model (SLM) for Japanese clinical and medicine, named NCVC-slm-1. This 1B parameters model was trained using Japanese text classified to be of high-quality. Moreover, NCVC-slm-1 was augmented with respect to clinical and medicine content that includes the variety of diseases, drugs, and examinations. Using a carefully designed pre-processing, a specialized morphological analyzer and tokenizer, this small and light-weight model performed not only to generate text but also indicated the feasibility of understanding clinical and medicine text. In comparison to other large language models, a fine-tuning NCVC-slm-1 demonstrated the highest scores on 6 tasks of total 8 on JMED-LLM. According to this result, SLM indicated the feasibility of performing several downstream tasks in the field of clinical and medicine. Hopefully, NCVC-slm-1 will be contributed to develop and accelerate the field of clinical and medicine for a bright future. 

**Abstract (ZH)**: 本报告介绍了一种针对日语临床和医学领域的较小语言模型（SLM），名为NCVC-slm-1。该模型包含1亿参数，并且是通过高质量的日语文本训练得到的。此外，NCVC-slm-1 在临床和医学内容方面得到了增强，涵盖了各种疾病、药物和检查项目。通过精心设计的预处理，以及专门的形态分析器和分词器，这款小型轻量级模型不仅能够生成文本，还显示了理解临床和医学文本的可行性。与现有的其他大型语言模型相比，对NCVC-slm-1进行微调后，在JMED-LLM的8项任务中，它在6项任务上的表现最为出色。根据这一结果，SLM 表明在临床医学领域执行多项下游任务的可行性。希望NCVC-slm-1 能够为临床医学领域的发展和加速进程做出贡献，并为未来带来美好的前景。 

---
# InfoTech Assistant : A Multimodal Conversational Agent for InfoTechnology Web Portal Queries 

**Title (ZH)**: InfoTech助手：一个针对信息技术网页门户查询的多模态对话代理 

**Authors**: Sai Surya Gadiraju, Duoduo Liao, Akhila Kudupudi, Santosh Kasula, Charitha Chalasani  

**Link**: [PDF](https://arxiv.org/pdf/2412.16412)  

**Abstract**: This pilot study presents the development of the InfoTech Assistant, a domain-specific, multimodal chatbot engineered to address queries in bridge evaluation and infrastructure technology. By integrating web data scraping, large language models (LLMs), and Retrieval-Augmented Generation (RAG), the InfoTech Assistant provides accurate and contextually relevant responses. Data, including textual descriptions and images, are sourced from publicly available documents on the InfoTechnology website and organized in JSON format to facilitate efficient querying. The architecture of the system includes an HTML-based interface and a Flask back end connected to the Llama 3.1 model via LLM Studio. Evaluation results show approximately 95 percent accuracy on domain-specific tasks, with high similarity scores confirming the quality of response matching. This RAG-enhanced setup enables the InfoTech Assistant to handle complex, multimodal queries, offering both textual and visual information in its responses. The InfoTech Assistant demonstrates strong potential as a dependable tool for infrastructure professionals, delivering high accuracy and relevance in its domain-specific outputs. 

**Abstract (ZH)**: 本试点研究介绍了一种针对桥粱评估和基础设施技术领域的、具有多模态功能的助手——InfoTech助理。通过集成网页数据爬取、大型语言模型（LLMs）和检索增强生成（RAG）技术，InfoTech助理能够提供准确并具有上下文相关性的回应。数据包括文本描述和图像，来源于InfoTechnology网站上的公开文档，并以JSON格式组织，以便高效查询。该系统的架构包含基于HTML的用户界面和通过LLM Studio连接到Llama 3.1模型的Flask后端。评估结果显示，在特定领域的任务中，其准确率达到约95%，高相似度得分证实了回应匹配的质量。这种RAG增强设置使InfoTech助理能够处理复杂的多模态查询，在回应中提供文本和视觉信息。InfoTech助理作为基础设施专业人士的可靠工具展示了强大的潜力，能够在特定领域的输出中提供高准确性和相关性。 

---
# Application of Multimodal Large Language Models in Autonomous Driving 

**Title (ZH)**: 多模态大规模语言模型在自动驾驶中的应用 

**Authors**: Md Robiul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2412.16410)  

**Abstract**: In this era of technological advancements, several cutting-edge techniques are being implemented to enhance Autonomous Driving (AD) systems, focusing on improving safety, efficiency, and adaptability in complex driving environments. However, AD still faces some problems including performance limitations. To address this problem, we conducted an in-depth study on implementing the Multi-modal Large Language Model. We constructed a Virtual Question Answering (VQA) dataset to fine-tune the model and address problems with the poor performance of MLLM on AD. We then break down the AD decision-making process by scene understanding, prediction, and decision-making. Chain of Thought has been used to make the decision more perfectly. Our experiments and detailed analysis of Autonomous Driving give an idea of how important MLLM is for AD. 

**Abstract (ZH)**: 在这个技术进步的时代，多种前沿技术正被应用到自动驾驶（AD）系统中，以提高其在复杂驾驶环境中的安全性、效率和适应性。然而，自动驾驶仍然面临一些问题，包括性能限制。为了解决这些问题，我们对多模态大语言模型（Multi-modal Large Language Model, MLLM）的实施进行了深入研究。我们构建了一个虚拟问答（Virtual Question Answering, VQA）数据集，以微调模型并解决MLLM在自动驾驶中表现不佳的问题。随后，我们将自动驾驶决策过程分解为场景理解、预测和决策三个阶段，并使用逆向思维（Chain of Thought）使决策更加完善。我们的实验及对自动驾驶的详细分析表明，MLLM对自动驾驶的重要性。 

---
# Overview of the First Workshop on Language Models for Low-Resource Languages (LoResLM 2025) 

**Title (ZH)**: 首届低资源语言语言模型研讨会（LoResLM 2025）概述 

**Authors**: Hansi Hettiarachchi, Tharindu Ranasinghe, Paul Rayson, Ruslan Mitkov, Mohamed Gaber, Damith Premasiri, Fiona Anting Tan, Lasitha Uyangodage  

**Link**: [PDF](https://arxiv.org/pdf/2412.16365)  

**Abstract**: The first Workshop on Language Models for Low-Resource Languages (LoResLM 2025) was held in conjunction with the 31st International Conference on Computational Linguistics (COLING 2025) in Abu Dhabi, United Arab Emirates. This workshop mainly aimed to provide a forum for researchers to share and discuss their ongoing work on language models (LMs) focusing on low-resource languages, following the recent advancements in neural language models and their linguistic biases towards high-resource languages. LoResLM 2025 attracted notable interest from the natural language processing (NLP) community, resulting in 35 accepted papers from 52 submissions. These contributions cover a broad range of low-resource languages from eight language families and 13 diverse research areas, paving the way for future possibilities and promoting linguistic inclusivity in NLP. 

**Abstract (ZH)**: 第一届低资源语言语言模型研讨会（LoResLM 2025）与第31届计算语言学国际会议（COLING 2025）在阿联酋阿布扎比联合举办。此次研讨会主要旨在为研究人员提供一个交流和讨论有关低资源语言语言模型（LMs）工作的论坛，紧跟神经语言模型的最新进展以及它们对高资源语言的偏见。LoResLM 2025 吸引了自然语言处理（NLP）社区的广泛关注，共接收了52篇提交论文中的35篇。这些贡献涵盖了来自八个语言家族和13个不同研究领域的广泛低资源语言，为未来提供了可能性，并促进了NLP中的语言包容性。 

---
# Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context 

**Title (ZH)**: 具有可读性的对抗性提示：基于情境上下文对LLM漏洞的研究 

**Authors**: Nilanjana Das, Edward Raff, Manas Gaur  

**Link**: [PDF](https://arxiv.org/pdf/2412.16359)  

**Abstract**: Previous research on LLM vulnerabilities often relied on nonsensical adversarial prompts, which were easily detectable by automated methods. We address this gap by focusing on human-readable adversarial prompts, a more realistic and potent threat. Our key contributions are situation-driven attacks leveraging movie scripts to create contextually relevant, human-readable prompts that successfully deceive LLMs, adversarial suffix conversion to transform nonsensical adversarial suffixes into meaningful text, and AdvPrompter with p-nucleus sampling, a method to generate diverse, human-readable adversarial suffixes, improving attack efficacy in models like GPT-3.5 and Gemma 7B. Our findings demonstrate that LLMs can be tricked by sophisticated adversaries into producing harmful responses with human-readable adversarial prompts and that there exists a scope for improvement when it comes to robust LLMs. 

**Abstract (ZH)**: 以往关于大规模语言模型（LLM）漏洞的研究大多依赖于无意义的对抗性提示，这些提示很容易被自动化检测方法识别。我们通过关注可读性较强的对抗性提示来解决这一问题，这种提示更加现实且更具威胁性。我们的主要贡献包括基于情境驱动的攻击，利用电影剧本创建上下文相关且可读性较强的对抗性提示，能够成功欺骗LLM；对抗性后缀转换技术，将无意义的对抗性后缀转化为有实际意义的文本；以及通过p-核子采样生成多样且可读性较强的对抗性后缀的AdvPrompter方法，提升如GPT-3.5和Gemma 7B等模型的攻击效果。我们的研究结果表明，LLM可以被具有高技术水平的对手通过可读性较强的对抗性提示欺骗，使其生成有害响应，同时也表明在构建鲁棒的LLM方面存在改进的空间。 

---
# Deliberative Alignment: Reasoning Enables Safer Language Models 

**Title (ZH)**: 审慎对齐：推理使语言模型更安全 

**Authors**: Melody Y. Guan, Manas Joglekar, Eric Wallace, Saachi Jain, Boaz Barak, Alec Heylar, Rachel Dias, Andrea Vallone, Hongyu Ren, Jason Wei, Hyung Won Chung, Sam Toyer, Johannes Heidecke, Alex Beutel, Amelia Glaese  

**Link**: [PDF](https://arxiv.org/pdf/2412.16339)  

**Abstract**: As large-scale language models increasingly impact safety-critical domains, ensuring their reliable adherence to well-defined principles remains a fundamental challenge. We introduce Deliberative Alignment, a new paradigm that directly teaches the model safety specifications and trains it to explicitly recall and accurately reason over the specifications before answering. We used this approach to align OpenAI's o-series models, and achieved highly precise adherence to OpenAI's safety policies, without requiring human-written chain-of-thoughts or answers. Deliberative Alignment pushes the Pareto frontier by simultaneously increasing robustness to jailbreaks while decreasing overrefusal rates, and also improves out-of-distribution generalization. We demonstrate that reasoning over explicitly specified policies enables more scalable, trustworthy, and interpretable alignment. 

**Abstract (ZH)**: 随着大规模语言模型在安全关键领域的影响日益增大，确保模型可靠地遵守明确定义的原则仍然是一个基本的挑战。我们提出了审慎对齐（Deliberative Alignment）这一新范式，该范式直接向模型教授安全规范，并训练模型在回答之前明确回忆并准确推理这些规范。我们采用此方法对OpenAI的o系列模型进行了对齐，并在无需人类编写推理过程或答案的情况下实现了对OpenAI安全政策的高度精确遵守。审慎对齐在同时提高对抗破坏性行为的鲁棒性的同时降低了过度拒绝率，并改善了模型的离分布泛化能力。我们展示了明确规定的政策推理能够实现更具扩展性、可信性和可解释性的对齐。 

---
# Decoding Linguistic Nuances in Mental Health Text Classification Using Expressive Narrative Stories 

**Title (ZH)**: 使用富有表现力的故事叙述解码心理健康文本分类中的语言细微差别 

**Authors**: Jinwen Tang, Qiming Guo, Yunxin Zhao, Yi Shang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16302)  

**Abstract**: Recent advancements in NLP have spurred significant interest in analyzing social media text data for identifying linguistic features indicative of mental health issues. However, the domain of Expressive Narrative Stories (ENS)-deeply personal and emotionally charged narratives that offer rich psychological insights-remains underexplored. This study bridges this gap by utilizing a dataset sourced from Reddit, focusing on ENS from individuals with and without self-declared depression. Our research evaluates the utility of advanced language models, BERT and MentalBERT, against traditional models. We find that traditional models are sensitive to the absence of explicit topic-related words, which could risk their potential to extend applications to ENS that lack clear mental health terminology. Despite MentalBERT is design to better handle psychiatric contexts, it demonstrated a dependency on specific topic words for classification accuracy, raising concerns about its application when explicit mental health terms are sparse (P-value<0.05). In contrast, BERT exhibited minimal sensitivity to the absence of topic words in ENS, suggesting its superior capability to understand deeper linguistic features, making it more effective for real-world applications. Both BERT and MentalBERT excel at recognizing linguistic nuances and maintaining classification accuracy even when narrative order is disrupted. This resilience is statistically significant, with sentence shuffling showing substantial impacts on model performance (P-value<0.05), especially evident in ENS comparisons between individuals with and without mental health declarations. These findings underscore the importance of exploring ENS for deeper insights into mental health-related narratives, advocating for a nuanced approach to mental health text analysis that moves beyond mere keyword detection. 

**Abstract (ZH)**: 近年来，自然语言处理（NLP）的进展激发了对社交媒体文本数据进行分析以识别与心理健康问题相关语言特征的显著兴趣。然而，表达性叙述故事（Expressive Narrative Stories，ENS）——这些故事深刻个人化且情感丰富，提供了丰富的心理洞察——这一领域尚未得到充分探索。本研究通过利用从Reddit收集的数据集填补了这一空白，重点关注有和没有自我报告抑郁情绪的个体的ENS。我们的研究评估了先进语言模型BERT和MentalBERT与传统模型的实用性。我们发现，传统模型对与主题无关的词汇的缺失特别敏感，这可能限制了它们在缺乏清晰心理健康术语的ENS中的应用潜力。尽管MentalBERT旨在更好地处理精神卫生背景，但它在分类准确性上仍然依赖于特定的主题词汇（P值<0.05），这引起了人们对在显性心理健康术语稀缺时应用它的担忧。相比之下，BERT在ENS中对主题词汇的缺失显得不那么敏感，表明其在理解更深层的语言特征方面具有更优越的能力，使其更适合实际应用。无论是BERT还是MentalBERT，在识别语言细微差别和保持分类准确性方面表现出色，尤其是在叙事顺序被打乱的情况下。这种稳健性具有统计学意义，句子重新排列显著影响了模型的性能（P值<0.05），特别是在有和无心理健康声明的个体之间的ENS比较中尤为明显。这些发现强调了探索ENS对于深入理解心理健康相关叙事的重要性，倡导一种超越单纯关键词检测的细腻心理健康文本分析方法。 

---
# Multi-head attention debiasing and contrastive learning for mitigating Dataset Artifacts in Natural Language Inference 

**Title (ZH)**: 多头注意力去偏见化和对比学习在减轻自然语言推理数据集中偏差方面的应用 

**Authors**: Karthik Sivakoti  

**Link**: [PDF](https://arxiv.org/pdf/2412.16194)  

**Abstract**: While Natural Language Inference (NLI) models have achieved high performances on benchmark datasets, there are still concerns whether they truly capture the intended task, or largely exploit dataset artifacts. Through detailed analysis of the Stanford Natural Language Inference (SNLI) dataset, we have uncovered complex patterns of various types of artifacts and their interactions, leading to the development of our novel structural debiasing approach. Our fine-grained analysis of 9,782 validation examples reveals four major categories of artifacts: length-based patterns, lexical overlap, subset relationships, and negation patterns. Our multi-head debiasing architecture achieves substantial improvements across all bias categories: length bias accuracy improved from 86.03% to 90.06%, overlap bias from 91.88% to 93.13%, subset bias from 95.43% to 96.49%, and negation bias from 88.69% to 94.64%. Overall, our approach reduces the error rate from 14.19% to 10.42% while maintaining high performance on unbiased examples. Analysis of 1,026 error cases shows significant improvement in handling neutral relationships, traditionally one of the most challenging areas for NLI systems. 

**Abstract (ZH)**: 尽管自然语言推理（NLI）模型在基准数据集上取得了高水平的表现，但仍有人担忧它们是否真正捕捉到了所需的任务，或只是依赖于数据集的固有特性。通过对斯坦福自然语言推理（SNLI）数据集的详细分析，我们揭示了各种类型数据集漏洞的复杂模式及其相互作用，并因此开发出了我们新的结构去偏方法。我们对9,782个验证样本进行了精细分析，发现四大类数据集漏洞：长度相关模式、词汇重叠、子集关系以及否定模式。我们的多头去偏架构在所有偏见类别中均取得了显著改进：长度偏倚准确性从86.03%提升到90.06%，重叠偏倚从91.88%提升到93.13%，子集偏倚从95.43%提升到96.49%，否定偏倚从88.69%提升到94.64%。总体而言，我们的方法将错误率从14.19%降至10.42%，同时在无偏数据样本上保持了高水平的表现。通过对1,026个错误案例的分析，我们发现显著改善了自然语言推理系统传统上最难处理的中性关系问题。 

---
# Experimenting with Multi-modal Information to Predict Success of Indian IPOs 

**Title (ZH)**: 使用多模态信息预测印度首次公开发行(IPO)成功率的实验研究 

**Authors**: Sohom Ghosh, Arnab Maji, N Harsha Vardhan, Sudip Kumar Naskar  

**Link**: [PDF](https://arxiv.org/pdf/2412.16174)  

**Abstract**: With consistent growth in Indian Economy, Initial Public Offerings (IPOs) have become a popular avenue for investment. With the modern technology simplifying investments, more investors are interested in making data driven decisions while subscribing for IPOs. In this paper, we describe a machine learning and natural language processing based approach for estimating if an IPO will be successful. We have extensively studied the impact of various facts mentioned in IPO filing prospectus, macroeconomic factors, market conditions, Grey Market Price, etc. on the success of an IPO. We created two new datasets relating to the IPOs of Indian companies. Finally, we investigated how information from multiple modalities (texts, images, numbers, and categorical features) can be used for estimating the direction and underpricing with respect to opening, high and closing prices of stocks on the IPO listing day. 

**Abstract (ZH)**: 随着印度经济的持续增长，首次公开募股（IPOs）已成为一种流行的投資渠道。现代技术简化了投资流程，越来越多的投资者倾向于在投资IPO时基于数据分析做出决策。本文介绍了基于机器学习和自然语言处理的方法，用于预测IPO的成功率。我们广泛研究了IPO招股书中的各种事实、宏观经济学因素、市场条件、灰市价格等因素对IPO成功的影响。我们还创建了两个与印度公司IPO相关的新型数据集。最后，我们探讨了来自多种模态（文本、图像、数字和分类特征）的信息如何用于估计股票在IPO上市日的开盘价、最高价和收盘价的定价偏差方向及程度。 

---
# Cross-Lingual Text-Rich Visual Comprehension: An Information Theory Perspective 

**Title (ZH)**: 跨语言文本丰富视觉理解：信息理论视角 

**Authors**: Xinmiao Yu, Xiaocheng Feng, Yun Li, Minghui Liao, Ya-Qi Yu, Xiachong Feng, Weihong Zhong, Ruihan Chen, Mengkang Hu, Jihao Wu, Dandan Tu, Duyu Tang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2412.17787)  

**Abstract**: Recent Large Vision-Language Models (LVLMs) have shown promising reasoning capabilities on text-rich images from charts, tables, and documents. However, the abundant text within such images may increase the model's sensitivity to language. This raises the need to evaluate LVLM performance on cross-lingual text-rich visual inputs, where the language in the image differs from the language of the instructions. To address this, we introduce XT-VQA (Cross-Lingual Text-Rich Visual Question Answering), a benchmark designed to assess how LVLMs handle language inconsistency between image text and questions. XT-VQA integrates five existing text-rich VQA datasets and a newly collected dataset, XPaperQA, covering diverse scenarios that require faithful recognition and comprehension of visual information despite language inconsistency. Our evaluation of prominent LVLMs on XT-VQA reveals a significant drop in performance for cross-lingual scenarios, even for models with multilingual capabilities. A mutual information analysis suggests that this performance gap stems from cross-lingual questions failing to adequately activate relevant visual information. To mitigate this issue, we propose MVCL-MI (Maximization of Vision-Language Cross-Lingual Mutual Information), where a visual-text cross-lingual alignment is built by maximizing mutual information between the model's outputs and visual information. This is achieved by distilling knowledge from monolingual to cross-lingual settings through KL divergence minimization, where monolingual output logits serve as a teacher. Experimental results on the XT-VQA demonstrate that MVCL-MI effectively reduces the visual-text cross-lingual performance disparity while preserving the inherent capabilities of LVLMs, shedding new light on the potential practice for improving LVLMs. Codes are available at: this https URL 

**Abstract (ZH)**: 近年来，大规模视觉-语言模型（LVLMs）在图表、表格和文档等富含文本的图像上的推理能力方面表现出了令人鼓舞的前景。然而，这些图像中的丰富文本可能会增加模型对语言的敏感性。这引发了对LVLM在跨语言视觉输入上的评估需求，其中图像中的语言与指令的语言不同。为了解决这一问题，我们提出了XT-VQA（跨语言富含文本的视觉问答），这是一个用于评估LVLM在图像文本与问题之间语言不一致情况下的处理能力的基准。XT-VQA将五个现有的富含文本的视觉问答数据集与一个新收集的数据集XPaperQA集成在一起，涵盖了尽管存在语言不一致，仍需忠实识别和理解视觉信息的各种场景。我们在XT-VQA上对主要的LVLM进行评估，发现即使对于具有多语言能力的模型，其在跨语言场景的表现也显著下降。我们通过互信息分析表明，这种性能差距源于跨语言问题未能充分激活相关的视觉信息。为了缓解这一问题，我们提出了MVCL-MI（视觉-语言跨语言互信息最大化），通过最大化模型输出与视觉信息之间的互信息，建立视觉-文本跨语言对齐。这通过最小化Kullback-Leibler（KL）发散从单一语言环境向跨语言环境传递知识实现，其中单一语言输出logits作为教师。在XT-VQA上的实验结果表明，MVCL-MI有效地缩小了视觉-文本跨语言性能差距，同时保持了LVLM的固有能力，为改进LVLM提供了新的启示。代码可在以下链接获取：[请插入链接] 

---
# RepoTransBench: A Real-World Benchmark for Repository-Level Code Translation 

**Title (ZH)**: RepoTransBench：面向 Repository 级别代码翻译的现实世界基准 

**Authors**: Yanli Wang, Yanlin Wang, Suiquan Wang, Daya Guo, Jiachi Chen, John Grundy, Xilin Liu, Yuchi Ma, Mingzhi Mao, Hongyu Zhang, Zibin Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2412.17744)  

**Abstract**: Repository-level code translation refers to translating an entire code repository from one programming language to another while preserving the functionality of the source repository. Many benchmarks have been proposed to evaluate the performance of such code translators. However, previous benchmarks mostly provide fine-grained samples, focusing at either code snippet, function, or file-level code translation. Such benchmarks do not accurately reflect real-world demands, where entire repositories often need to be translated, involving longer code length and more complex functionalities. To address this gap, we propose a new benchmark, named RepoTransBench, which is a real-world repository-level code translation benchmark with an automatically executable test suite. We conduct experiments on RepoTransBench to evaluate the translation performance of 11 advanced LLMs. We find that the Success@1 score (test success in one attempt) of the best-performing LLM is only 7.33%. To further explore the potential of LLMs for repository-level code translation, we provide LLMs with error-related feedback to perform iterative debugging and observe an average 7.09% improvement on Success@1. However, even with this improvement, the Success@1 score of the best-performing LLM is only 21%, which may not meet the need for reliable automatic repository-level code translation. Finally, we conduct a detailed error analysis and highlight current LLMs' deficiencies in repository-level code translation, which could provide a reference for further improvements. 

**Abstract (ZH)**: 代码仓库级别的代码翻译指的是将整个代码仓库从一种编程语言翻译成另一种语言，同时保持源代码仓库的功能。已经提出了许多基准测试来评估此类代码翻译器的性能。然而，以往的基准测试主要提供细腻化的样本，集中在代码片段、函数或文件级别的代码翻译上。这些基准测试没有准确反映现实世界的需求，因为通常需要翻译整个代码仓库，涉及更长的代码长度和更复杂的功能。为了解决这一差距，我们提出了一个新的基准测试，名为RepoTransBench，这是一个具有自动可执行测试套件的真实世界代码仓库级别的代码翻译基准测试。我们在RepoTransBench上进行实验，以评估11个先进大语言模型（LLM）的翻译性能。我们发现，最佳性能的LLM的Success@1得分（一次尝试的成功测试）仅为7.33%。为进一步探索LLM在代码仓库级别代码翻译方面的潜力，我们为LLM提供了与错误相关的反馈，使其进行迭代调试，并观察到平均7.09%的成功率提升。然而，即使有这个提高，最佳性能的LLM的成功率也仅为21%，可能无法满足可靠自动代码仓库级别代码翻译的需求。最后，我们进行了详细错误分析，并指出了当前LLM在代码仓库级别代码翻译方面的缺陷，这对进一步改进提供了参考。 

---
# Fourier Position Embedding: Enhancing Attention's Periodic Extension for Length Generalization 

**Title (ZH)**: 傅里叶位置嵌入：增强注意力周期扩展以提高长度泛化能力 

**Authors**: Ermo Hua, Che Jiang, Xingtai Lv, Kaiyan Zhang, Ning Ding, Youbang Sun, Biqing Qi, Yuchen Fan, Xue Kai Zhu, Bowen Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2412.17739)  

**Abstract**: Extending the context length of Language Models (LMs) by improving Rotary Position Embedding (RoPE) has become a trend. While existing works mainly address RoPE's limitations within attention mechanism, this paper provides an analysis across nearly all parts of LMs, uncovering their adverse effects on length generalization for RoPE-based attention. Using Discrete Signal Processing theory, we show that RoPE enables periodic attention by implicitly achieving Non-Uniform Discrete Fourier Transform. However, this periodicity is undermined by the spectral damage caused by: 1) linear layers and activation functions outside of attention; 2) insufficiently trained frequency components brought by time-domain truncation. Building on our observations, we propose Fourier Position Embedding (FoPE), which enhances attention's frequency-domain properties to improve both its periodic extension and length generalization. FoPE constructs Fourier Series and zero-outs the destructive frequency components, increasing model robustness against the spectrum damage. Experiments across various model scales show that, within varying context windows, FoPE can maintain a more stable perplexity and a more consistent accuracy in a needle-in-haystack task compared to RoPE and ALiBi. Several analyses and ablations bring further support to our method and theoretical modeling. 

**Abstract (ZH)**: 通过对语言模型（LMs）上下文长度的扩展，改进旋转位置嵌入（RoPE）已成为一种趋势。虽然现有工作主要关注RoPE在注意机制中的局限性，本文则从LMs的几乎所有部分出发，分析发现RoPE对基于RoPE的注意机制在长度泛化方面的影响。利用离散信号处理理论，我们表明RoPE通过隐式实现非均匀离散傅里叶变换来实现周期性注意。然而，这种周期性因以下原因受到损害：1） attention机制之外的线性层和激活函数；2）时间域截断带来的训练不足的频率分量。基于上述观察，我们提出了一种傅里叶位置嵌入（FoPE），通过增强注意机制在频域上的性质，改善其周期性扩展和长度泛化。FoPE通过构建傅里叶级数并消除破坏性的频率分量，增加了模型对频谱损害的鲁棒性。来自不同模型规模的实验表明，与RoPE和ALiBi相比，在不同上下文窗口内，FoPE可以保持更稳定的困惑度和更加一致的准确度，在针叶林任务中表现更佳。多项分析和消融实验进一步支持了我们的方法和理论建模。 

---
# Tracking the Feature Dynamics in LLM Training: A Mechanistic Study 

**Title (ZH)**: 对大规模语言模型训练中特征动态的机理研究 

**Authors**: Yang Xu, Yi Wang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17626)  

**Abstract**: Understanding training dynamics and feature evolution is crucial for the mechanistic interpretability of large language models (LLMs). Although sparse autoencoders (SAEs) have been used to identify features within LLMs, a clear picture of how these features evolve during training remains elusive. In this study, we: (1) introduce SAE-Track, a method to efficiently obtain a continual series of SAEs; (2) formulate the process of feature formation and conduct a mechanistic analysis; and (3) analyze and visualize feature drift during training. Our work provides new insights into the dynamics of features in LLMs, enhancing our understanding of training mechanisms and feature evolution. 

**Abstract (ZH)**: 理解训练动力学和特征演化对于大型语言模型（LLMs）的机理可解释性至关重要。尽管已经使用稀疏自动编码器（SAEs）来识别LLMs中的特征，但在训练过程中这些特征如何演变的具体图景仍然不清楚。在本研究中，我们：（1）引入了一种名为SAE-Track的方法，以高效地获得连续的SAE系列；（2）定义了特征形成的过程，并进行了机理分析；（3）分析并可视化了训练过程中特征漂移的情况。我们的工作为理解LLMs中特征的动力学提供了新的见解，增强了我们对训练机制和特征演化理解。 

---
# CiteBART: Learning to Generate Citations for Local Citation Recommendation 

**Title (ZH)**: CiteBART：学习为局部引文推荐生成引用 

**Authors**: Ege Yiğit Çelik, Selma Tekir  

**Link**: [PDF](https://arxiv.org/pdf/2412.17534)  

**Abstract**: Citations are essential building blocks in scientific writing. The scientific community is longing for support in their generation. Citation generation involves two complementary subtasks: Determining the citation worthiness of a context and, if it's worth it, proposing the best candidate papers for the citation placeholder. The latter subtask is called local citation recommendation (LCR). This paper proposes CiteBART, a custom BART pre-training based on citation token masking to generate citations to achieve LCR. In the base scheme, we mask the citation token in the local citation context to make the citation prediction. In the global one, we concatenate the citing paper's title and abstract to the local citation context to learn to reconstruct the citation token. CiteBART outperforms state-of-the-art approaches on the citation recommendation benchmarks except for the smallest FullTextPeerRead dataset. The effect is significant in the larger benchmarks, e.g., Refseer and ArXiv. We present a qualitative analysis and an ablation study to provide insights into the workings of CiteBART. Our analyses confirm that its generative nature brings about a zero-shot capability. 

**Abstract (ZH)**: 参考引文是科学写作中的重要构建块。科学界渴望能够支持引文的生成。引文生成包含两个互补的子任务：确定某个上下文的引文价值，并在其具有引文价值时，提出最适合的引用候选论文。后者被称为局部引荐推荐（Local Citation Recommendation, LCR）。本文提出了一种名为CiteBART的方法，该方法基于引用标记掩蔽的自定义BART预训练，以实现LCR。在基本方案中，我们通过掩蔽局部引文上下文中的引文标记来进行引文预测。在全局方案中，我们将引用论文的标题和摘要与局部引文上下文进行拼接，以学习重建引文标记。CiteBART在引用推荐基准测试中表现出色，除了在最小的FullTextPeerRead数据集中表现稍逊之外，在更大的基准测试中，如Refseer和ArXiv，效果显著。我们进行了定性的分析和消融研究，以揭示CiteBART的工作机制。我们的分析证实，其生成特性赋予了其零样本学习的能力。 

---
# Developmental Predictive Coding Model for Early Infancy Mono and Bilingual Vocal Continual Learning 

**Title (ZH)**: 婴儿早期单双语连续语音发展的预测编码模型 

**Authors**: Xiaodan Chen, Alexandre Pitti, Mathias Quoy, Nancy F Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.17456)  

**Abstract**: Understanding how infants perceive speech sounds and language structures is still an open problem. Previous research in artificial neural networks has mainly focused on large dataset-dependent generative models, aiming to replicate language-related phenomena such as ''perceptual narrowing''. In this paper, we propose a novel approach using a small-sized generative neural network equipped with a continual learning mechanism based on predictive coding for mono-and bilingual speech sound learning (referred to as language sound acquisition during ''critical period'') and a compositional optimization mechanism for generation where no learning is involved (later infancy sound imitation). Our model prioritizes interpretability and demonstrates the advantages of online learning: Unlike deep networks requiring substantial offline training, our model continuously updates with new data, making it adaptable and responsive to changing inputs. Through experiments, we demonstrate that if second language acquisition occurs during later infancy, the challenges associated with learning a foreign language after the critical period amplify, replicating the perceptual narrowing effect. 

**Abstract (ZH)**: 理解婴儿如何感知语音声音和语言结构仍然是一个开放性问题。以前在人工神经网络领域的研究主要侧重于依赖大量数据的生成模型，旨在复制诸如“感知收缩”等语言相关现象。本文提出了一种新型方法，使用一个小型生成神经网络，该网络结合了基于预测编码的持续学习机制，用于单语和双语语音声音学习（称为“关键期”的语言声音获取），以及一种组合优化机制用于生成（后期婴儿语音模仿，不涉及学习）。我们的模型优先考虑可解释性，并展示了在线学习的优势：不同于需要大量离线训练的深层网络，我们的模型能够持续更新新的数据，使其具有适应性和对变化输入的响应能力。通过实验，我们证明了如果第二语言习得发生在后期婴儿期，那么在关键期后学习一门外语所带来的挑战将加剧，从而复制感知收缩效应。 

---
# MineAgent: Towards Remote-Sensing Mineral Exploration with Multimodal Large Language Models 

**Title (ZH)**: MineAgent：面向多模态大规模语言模型的遥感矿产勘探研究 

**Authors**: Beibei Yu, Tao Shen, Hongbin Na, Ling Chen, Denqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.17339)  

**Abstract**: Remote-sensing mineral exploration is critical for identifying economically viable mineral deposits, yet it poses significant challenges for multimodal large language models (MLLMs). These include limitations in domain-specific geological knowledge and difficulties in reasoning across multiple remote-sensing images, further exacerbating long-context issues. To address these, we present MineAgent, a modular framework leveraging hierarchical judging and decision-making modules to improve multi-image reasoning and spatial-spectral integration. Complementing this, we propose MineBench, a benchmark specific for evaluating MLLMs in domain-specific mineral exploration tasks using geological and hyperspectral data. Extensive experiments demonstrate the effectiveness of MineAgent, highlighting its potential to advance MLLMs in remote-sensing mineral exploration. 

**Abstract (ZH)**: 遥感矿物勘探对于识别具有经济开采价值的矿床至关重要，但对多模态大型语言模型（MLLMs）提出了重大挑战。这些挑战包括领域特定地质知识的局限性以及在多张遥感图像间进行推理的难度，进一步加剧了长上下文问题。为解决这些问题，我们提出了MineAgent，这是一种模块化框架，利用层次化的判断和决策模块以提高多图像推理和谱-空域集成的效果。为配合这一框架，我们还提出了MineBench，这是一个特定于地质和高光谱数据的基准，用于评估MLLMs在特定领域的矿物勘探任务中的性能。广泛的实验表明，MineAgent的有效性，并突显了其在遥感矿物勘探中推进MLLMs的潜力。 

---
# Fast Gradient Computation for RoPE Attention in Almost Linear Time 

**Title (ZH)**: 几乎线性时间的RoPE注意力梯度快速计算 

**Authors**: Yifang Chen, Jiayan Huo, Xiaoyu Li, Yingyu Liang, Zhenmei Shi, Zhao Song  

**Link**: [PDF](https://arxiv.org/pdf/2412.17316)  

**Abstract**: The Rotary Position Embedding (RoPE) mechanism has become a powerful enhancement to the Transformer architecture, which enables models to capture token relationships when encoding positional information. However, the RoPE mechanisms make the computations of attention mechanisms more complicated, which makes efficient algorithms challenging. Earlier research introduced almost linear time, i.e., $n^{1+o(1)}$ where $n$ is the number of input tokens, algorithms for the forward computation under specific parameter settings. However, achieving a subquadratic time algorithm for other parameter regimes remains impossible unless the widely accepted Strong Exponential Time Hypothesis (SETH) is disproven. In this work, we develop the first almost linear time algorithm for backward computations in the RoPE-based attention under bounded entries. Our approach builds on recent advancements in fast RoPE attention computations, utilizing a novel combination of the polynomial method and the Fast Fourier Transform. Furthermore, we show that with lower bounds derived from the SETH, the bounded entry condition is necessary for subquadratic performance. 

**Abstract (ZH)**: 旋转位置嵌入（RoPE）机制已经成为Transformer架构的强大增强，它使模型能够在编码位置信息时捕获令牌之间的关系。然而，RoPE机制使得注意力机制的计算变得更加复杂，从而使得高效的算法变得更具挑战性。早期的研究在特定参数设置下引入了接近线性时间的算法，即当n（输入令牌的数量）增大时，时间复杂度接近于$n^{1+o(1)}$。然而，除非广泛接受的强指数时间假设（SETH）被证伪，否则在其他参数范围内实现次二次时间算法仍然是不可能的。在本文中，我们开发了RoPE基注意力下有界条目项后向计算的第一个接近线性时间算法。我们的方法建立在快速RoPE注意力计算的 recent 进展之上，利用了多项式方法和快速傅里叶变换的新型结合方式。此外，我们证明了从SETH推导出的下限表明，为了获得次二次性能，有界条目条件是必要的。 

---
# CodeV: Issue Resolving with Visual Data 

**Title (ZH)**: CodeV：基于视觉数据的问题解决 

**Authors**: Linhao Zhang, Daoguang Zan, Quanshun Yang, Zhirong Huang, Dong Chen, Bo Shen, Tianyu Liu, Yongshun Gong, Pengjie Huang, Xudong Lu, Guangtai Liang, Lizhen Cui, Qianxiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.17315)  

**Abstract**: Large Language Models (LLMs) have advanced rapidly in recent years, with their applications in software engineering expanding to more complex repository-level tasks. GitHub issue resolving is a key challenge among these tasks. While recent approaches have made progress on this task, they focus on textual data within issues, neglecting visual data. However, this visual data is crucial for resolving issues as it conveys additional knowledge that text alone cannot. We propose CodeV, the first approach to leveraging visual data to enhance the issue-resolving capabilities of LLMs. CodeV resolves each issue by following a two-phase process: data processing and patch generation. To evaluate CodeV, we construct a benchmark for visual issue resolving, namely Visual SWE-bench. Through extensive experiments, we demonstrate the effectiveness of CodeV, as well as provide valuable insights into leveraging visual data to resolve GitHub issues. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）取得了 rapid 的进展，其在软件工程中的应用扩展到了更加复杂的存储库级别任务。GitHub问题解决是一个其中的关键挑战。虽然最近的方法在这项任务上取得了进展，但它们主要专注于问题中的文本数据，忽视了视觉数据。然而，这些视觉数据对于解决问题是至关重要的，因为它提供了文本单独无法传达的知识。我们提出了 CodeV，这是第一个利用视觉数据增强 LLM 问题解决能力的方法。CodeV 通过一个两阶段的过程来解决每个问题：数据处理和补丁生成。

为了评估 CodeV，我们构建了一个视觉问题解决基准，即 Visual SWE-bench。通过大量的实验，我们展示了 CodeV 的有效性，并提供了一些有关如何利用视觉数据解决 GitHub 问题的重要见解。 

---
# B-STaR: Monitoring and Balancing Exploration and Exploitation in Self-Taught Reasoners 

**Title (ZH)**: B-STaR：自我教学推理者中探索与利用的监控和平衡 

**Authors**: Weihao Zeng, Yuzhen Huang, Lulu Zhao, Yijun Wang, Zifei Shan, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2412.17256)  

**Abstract**: In the absence of extensive human-annotated data for complex reasoning tasks, self-improvement -- where models are trained on their own outputs -- has emerged as a primary method for enhancing performance. However, the critical factors underlying the mechanism of these iterative self-improving methods remain poorly understood, such as under what conditions self-improvement is effective, and what are the bottlenecks in the current iterations. In this work, we identify and propose methods to monitor two pivotal factors in this iterative process: (1) the model's ability to generate sufficiently diverse responses (exploration); and (2) the effectiveness of external rewards in distinguishing high-quality candidates from lower-quality ones (exploitation). Using mathematical reasoning as a case study, we begin with a quantitative analysis to track the dynamics of exploration and exploitation, discovering that a model's exploratory capabilities rapidly deteriorate over iterations, and the effectiveness of exploiting external rewards diminishes as well. Motivated by these findings, we introduce B-STaR, a Self-Taught Reasoning framework that autonomously adjusts configurations across iterations to Balance exploration and exploitation, thereby optimizing the self-improving effectiveness based on the current policy model and available rewards. Our experiments on mathematical reasoning, coding, and commonsense reasoning demonstrate that B-STaR not only enhances the model's exploratory capabilities throughout training but also achieves a more effective balance between exploration and exploitation, leading to superior performance. 

**Abstract (ZH)**: 在缺乏大量人工标注数据的情况下，为了复杂推理任务，模型通过自我提升（即模型在其自身输出上进行训练）已成为提高性能的主要方法。然而，这些迭代自我提升机制的核心因素仍然知之甚少，例如在什么条件下自我提升有效，当前迭代中的瓶颈又是什么。在这项工作中，我们识别并提出了监测这一迭代过程中的两个关键因素的方法：（1）模型生成足够多样化响应的能力（探索）；（2）外部奖励在区分高质量候选对象与低质量候选对象方面的有效性（利用）。用数学推理作为案例研究，我们首先进行定量分析以跟踪探索和利用的动力学，发现模型的探索能力在迭代中迅速下降，利用外部奖励的有效性也在减弱。基于这些发现，我们引入了B-STaR（自学习推理）框架，该框架在迭代过程中自主调整配置，平衡探索与利用，从而根据当前策略模型和可用奖励优化自我提升的有效性。我们在数学推理、编程和常识推理的实验中表明，B-STaR不仅能增强模型在整个训练过程中的探索能力，还能更有效地平衡探索与利用，从而提高性能。 

---
# On the Generalization Ability of Machine-Generated Text Detectors 

**Title (ZH)**: 机器生成文本检测器的泛化能力研究 

**Authors**: Yule Liu, Zhiyuan Zhong, Yifan Liao, Zhen Sun, Jingyi Zheng, Jiaheng Wei, Qingyuan Gong, Fenghua Tong, Yang Chen, Yang Zhang, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2412.17242)  

**Abstract**: The rise of large language models (LLMs) has raised concerns about machine-generated text (MGT), including ethical and practical issues like plagiarism and misinformation. Building a robust and highly generalizable MGT detection system has become increasingly important. This work investigates the generalization capabilities of MGT detectors in three aspects: First, we construct MGTAcademic, a large-scale dataset focused on academic writing, featuring human-written texts (HWTs) and MGTs across STEM, Humanities, and Social Sciences, paired with an extensible code framework for efficient benchmarking. Second, we investigate the transferability of detectors across domains and LLMs, leveraging fine-grained datasets to reveal insights into domain transferring and implementing few-shot techniques to improve the performance by roughly 13.2%. Third, we introduce a novel attribution task where models must adapt to new classes over time without (or with very limited) access to prior training data and benchmark detectors. We implement several adapting techniques to improve the performance by roughly 10% and highlight the inherent complexity of the task. Our findings provide insights into the generalization ability of MGT detectors across diverse scenarios and lay the foundation for building robust, adaptive detection systems. 

**Abstract (ZH)**: 大规模语言模型（LLMs）的兴起引发了人们对机器生成文本（MGT）的伦理和实践问题的关注，包括剽窃和虚假信息等问题。构建一个强大且高度通用的MGT检测系统变得越来越重要。本研究从三个方面探讨了MGT检测器的泛化能力：首先，我们构建了一个名为MGTAcademic的大规模数据集，专注于学术写作，包括自然科学、人文和社会科学领域的由人类撰写的文本（HWTs）和MGT，并且提供了一个可扩展的代码框架，用于高效的基准测试。其次，我们探讨了检测器在不同领域和不同LLMs之间的可迁移性，利用精细粒度的数据集揭示了领域迁移的洞察，并通过少量提示技术将性能提高了约13.2%。第三，我们引入了一项新颖的归因任务，要求模型在（或仅有限访问）先前训练数据的情况下适应新的类别，并进行基准测试。我们实施了多种适应技术，将性能提高了约10%，并突出了该任务的内在复杂性。我们的发现为跨不同场景的MGT检测器的泛化能力提供了新的见解，并为构建鲁棒且适应性强的检测系统奠定了基础。 

---
# COVID-19 on YouTube: A Data-Driven Analysis of Sentiment, Toxicity, and Content Recommendations 

**Title (ZH)**: COVID-19在YouTube上的影响：基于数据的 sentiment、毒性及内容推荐分析 

**Authors**: Vanessa Su, Nirmalya Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2412.17180)  

**Abstract**: This study presents a data-driven analysis of COVID-19 discourse on YouTube, examining the sentiment, toxicity, and thematic patterns of video content published between January 2023 and October 2024. The analysis involved applying advanced natural language processing (NLP) techniques: sentiment analysis with VADER, toxicity detection with Detoxify, and topic modeling using Latent Dirichlet Allocation (LDA). The sentiment analysis revealed that 49.32% of video descriptions were positive, 36.63% were neutral, and 14.05% were negative, indicating a generally informative and supportive tone in pandemic-related content. Toxicity analysis identified only 0.91% of content as toxic, suggesting minimal exposure to toxic content. Topic modeling revealed two main themes, with 66.74% of the videos covering general health information and pandemic-related impacts and 33.26% focused on news and real-time updates, highlighting the dual informational role of YouTube. A recommendation system was also developed using TF-IDF vectorization and cosine similarity, refined by sentiment, toxicity, and topic filters to ensure relevant and context-aligned video recommendations. This system achieved 69% aggregate coverage, with monthly coverage rates consistently above 85%, demonstrating robust performance and adaptability over time. Evaluation across recommendation sizes showed coverage reaching 69% for five video recommendations and 79% for ten video recommendations per video. In summary, this work presents a framework for understanding COVID-19 discourse on YouTube and a recommendation system that supports user engagement while promoting responsible and relevant content related to COVID-19. 

**Abstract (ZH)**: 本文通过对2023年1月至2024年10月期间YouTube上COVID-19相关视频内容的情感、毒性与主题模式进行数据驱动分析，探讨了该时期视频内容的情感倾向、毒性和主题模式。分析采用了高级自然语言处理（NLP）技术：使用VADER进行情感分析，使用Detoxify进行毒性检测，使用潜在狄利克雷分配（LDA）进行主题建模。情感分析结果显示，49.32%的视频描述为正面情感，36.63%为中性情感，14.05%为负面情感，表明相关内容具有普遍的信息性和支持性。毒性分析发现仅有0.91%的内容包含毒性强的内容，表明毒性强的内容暴露度较低。主题建模结果显示，有66.74%的视频涵盖一般健康信息和疫情相关影响，33.26%的视频聚焦于新闻和实时更新，突显了YouTube的双重信息作用。同时，还开发了一个基于TF-IDF向量化和余弦相似度的推荐系统，并通过情感、毒性及主题筛选进行优化，以确保推荐的相关性和上下文一致性。该系统整体覆盖率达到了69%，每月覆盖率持续保持在85%以上，显示出随着时间的推移，该系统表现出了稳定性和适应性。在不同推荐数量的评估中，发现五视频推荐覆盖了69%，十视频推荐覆盖了79%。总之，本研究提出了一种了解YouTube上COVID-19讨论的框架，并开发了一个支持用户参与、促进与COVID-19相关负责任和相关内容的推荐系统。 

---
# Analysis of Speech Temporal Dynamics in the Context of Speaker Verification and Voice Anonymization 

**Title (ZH)**: 在说话人验证和语音匿名化背景下的声音时域动态分析 

**Authors**: Natalia Tomashenko, Emmanuel Vincent, Marc Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2412.17164)  

**Abstract**: In this paper, we investigate the impact of speech temporal dynamics in application to automatic speaker verification and speaker voice anonymization tasks. We propose several metrics to perform automatic speaker verification based only on phoneme durations. Experimental results demonstrate that phoneme durations leak some speaker information and can reveal speaker identity from both original and anonymized speech. Thus, this work emphasizes the importance of taking into account the speaker's speech rate and, more importantly, the speaker's phonetic duration characteristics, as well as the need to modify them in order to develop anonymization systems with strong privacy protection capacity. 

**Abstract (ZH)**: 在本文中，我们探讨了语音时序动态在自动说话人验证和说话人语音匿名化任务中的影响。我们提出了一些基于音素时长的自动说话人验证度量方法。实验结果表明，音素时长会泄露一些说话人信息，并且可以从原始语音和匿名语音中揭示说话人的身份。因此，本文强调了在考虑说话人语速的同时，更加重要的是要考虑到说话人的音素时长特征，并且需要对这些特征进行修改，以开发具有强大隐私保护能力的匿名化系统。 

---
# Iterative NLP Query Refinement for Enhancing Domain-Specific Information Retrieval: A Case Study in Career Services 

**Title (ZH)**: 迭代自然语言处理查询优化以增强领域特定信息检索：职业服务案例研究 

**Authors**: Elham Peimani, Gurpreet Singh, Nisarg Mahyavanshi, Aman Arora, Awais Shaikh  

**Link**: [PDF](https://arxiv.org/pdf/2412.17075)  

**Abstract**: Retrieving semantically relevant documents in niche domains poses significant challenges for traditional TF-IDF-based systems, often resulting in low similarity scores and suboptimal retrieval performance. This paper addresses these challenges by introducing an iterative and semi-automated query refinement methodology tailored to Humber College's career services webpages. Initially, generic queries related to interview preparation yield low top-document similarities (approximately 0.2--0.3). To enhance retrieval effectiveness, we implement a two-fold approach: first, domain-aware query refinement by incorporating specialized terms such as resources-online-learning, student-online-services, and career-advising; second, the integration of structured educational descriptors like "online resume and interview improvement tools." Additionally, we automate the extraction of domain-specific keywords from top-ranked documents to suggest relevant terms for query expansion. Through experiments conducted on five baseline queries, our semi-automated iterative refinement process elevates the average top similarity score from approximately 0.18 to 0.42, marking a substantial improvement in retrieval performance. The implementation details, including reproducible code and experimental setups, are made available in our GitHub repositories \url{this https URL} and \url{this https URL}. We also discuss the limitations of our approach and propose future directions, including the integration of advanced neural retrieval models. 

**Abstract (ZH)**: 传统的基于TF-IDF的系统在检索特定领域的语义相关文档时面临着显著挑战，往往导致相似度评分较低和检索性能不佳。本文通过引入一种针对海本田学院职业服务网页的迭代和半自动化查询优化方法来应对这些挑战。最初，与面试准备相关的通用查询导致顶级文档相似度较低（约0.2到0.3）。为提高检索效果，我们采用了两方面的策略：首先，通过引入专业术语（如“在线学习资源”、“学生在线服务”和“职业咨询”）进行领域意识下的查询优化；其次，整合结构化的教育描述符（如“在线简历和面试改进工具”）。此外，我们还自动提取顶级文档中的领域特定关键词，以建议用于查询扩展的相关术语。在五个基线查询上进行的实验结果显示，半自动迭代优化过程将平均顶级相似度评分从约0.18提高到0.42，显著提升了检索性能。我们已在GitHub仓库中提供了实现细节，包括可复现的代码和实验设置，具体链接为[this](https://github.com/example/example_repo1)和[this](https://github.com/example/example_repo2)。我们还讨论了该方法的局限性，并提出了未来的研究方向，包括集成高级神经检索模型。 

---
# Modular Conversational Agents for Surveys and Interviews 

**Title (ZH)**: 模块化对话代理在调查和访谈中的应用 

**Authors**: Jiangbo Yu, Jinhua Zhao, Luis Miranda-Moreno, Matthew Korp  

**Link**: [PDF](https://arxiv.org/pdf/2412.17049)  

**Abstract**: Surveys and interviews (structured, semi-structured, or unstructured) are widely used for collecting insights on emerging or hypothetical scenarios. Traditional human-led methods often face challenges related to cost, scalability, and consistency. Recently, various domains have begun to explore the use of conversational agents (chatbots) powered by large language models (LLMs). However, as public investments and policies on infrastructure and services often involve substantial public stakes and environmental risks, there is a need for a rigorous, transparent, privacy-preserving, and cost-efficient development framework tailored for such major decision-making processes. This paper addresses this gap by introducing a modular approach and its resultant parameterized process for designing conversational agents. We detail the system architecture, integrating engineered prompts, specialized knowledge bases, and customizable, goal-oriented conversational logic in the proposed approach. We demonstrate the adaptability, generalizability, and efficacy of our modular approach through three empirical studies: (1) travel preference surveys, highlighting multimodal (voice, text, and image generation) capabilities; (2) public opinion elicitation on a newly constructed, novel infrastructure project, showcasing question customization and multilingual (English and French) capabilities; and (3) transportation expert consultation about future transportation systems, highlighting real-time, clarification request capabilities for open-ended questions, resilience in handling erratic inputs, and efficient transcript post-processing. The results show the effectiveness of this modular approach and how it addresses key ethical, privacy, security, and token consumption concerns, setting the stage for the next-generation surveys and interviews. 

**Abstract (ZH)**: 调查和访谈（结构化的、半结构化的或非结构化的）广泛用于收集关于新兴或假设情境的见解。传统的由人类主导的方法经常会面临成本、可扩展性和一致性方面的挑战。近年来，各个领域开始探索使用大型语言模型（LLMs）驱动的对话代理（聊天机器人）的方法。然而，由于公共投资和政策通常涉及重要公共利益和环境风险，因此需要一种严谨、透明、保护隐私且成本效益高的开发框架，以适应这些重大决策过程。本文通过引入模块化方法及其参数化流程来解决这一问题，详细介绍了该系统的架构，涵盖了精心设计的提示、专业化的知识库以及可定制的目标导向对话逻辑。通过三项实证研究，展示了模块化方法的适应性、可移植性和有效性：（1）旅行偏好的调查，突显了多模态（语音、文本和图像生成）能力；（2）对一个新建成的创新型基础设施项目的公众意见收集，展现了问题定制化和多种语言（英语和法语）能力；以及（3）交通运输专家对未来交通系统咨询，强调了对开放式问题的即时澄清请求能力、处理异常输入的韧性以及高效的对话记录后处理。研究结果表明了模块化方法的有效性及其如何解决关键的伦理、隐私、安全性和令牌消耗问题，为下一代调查和访谈奠定了基础。 

---
# Why Do Speech Language Models Fail to Generate Semantically Coherent Outputs? A Modality Evolving Perspective 

**Title (ZH)**: 为什么语音语言模型无法生成语义连贯的输出？从模态演化视角进行探讨 

**Authors**: Hankun Wang, Haoran Wang, Yiwei Guo, Zhihan Li, Chenpeng Du, Xie Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2412.17048)  

**Abstract**: Although text-based large language models exhibit human-level writing ability and remarkable intelligence, speech language models (SLMs) still struggle to generate semantically coherent outputs. There are several potential reasons for this performance degradation: (A) speech tokens mainly provide phonetic information rather than semantic information, (B) the length of speech sequences is much longer than that of text sequences, and (C) paralinguistic information, such as prosody, introduces additional complexity and variability. In this paper, we explore the influence of three key factors separately by transiting the modality from text to speech in an evolving manner. Our findings reveal that the impact of the three factors varies. Factor A has a relatively minor impact, factor B influences syntactical and semantic modeling more obviously, and factor C exerts the most significant impact, particularly in the basic lexical modeling. Based on these findings, we provide insights into the unique challenges of training SLMs and highlight pathways to develop more effective end-to-end SLMs. 

**Abstract (ZH)**: 尽管基于文本的大语言模型展示了与人类相当的写作能力和显著的智能，但在生成语义连贯的输出方面，语音语言模型（SLMs）仍然面临挑战。造成这一性能下降的原因可能有以下几点：（A）语音标记主要提供音位信息而非语义信息，（B）语音序列的长度远超过文本序列的长度，以及（C）副语言信息，如语调，增加了额外的复杂性和变异性。在本文中，我们通过逐级从文本模态转换到语音模态，分别探讨了这三大因素的影响。我们的研究结果表明，这三种因素的影响程度不一。因素A的影响相对较小，因素B对句法和语义建模的影响更为明显，而因素C则产生最大的影响，尤其是在基本词汇建模方面。基于这些发现，我们提出了训练SLMs所面临的独特挑战，并强调了开发更有效的端到端SLMs的途径。 

---
# Cannot or Should Not? Automatic Analysis of Refusal Composition in IFT/RLHF Datasets and Refusal Behavior of Black-Box LLMs 

**Title (ZH)**: 当然，以下是翻译的内容，符合学术规范：

"无法自动分析还是不应分析？IFT/RLHF 数据集中的拒绝样本组成分析及其对黑盒大模型拒绝行为的影响"

对于学术论文的标题，通常会尽量简洁且准确地传达研究内容。这里的翻译保持了原文的严谨性和专业性，同时也确保了中文表达的自然流畅。如果需要进一步调整以适应具体的学术风格或其他需求，请告诉我。 

**Authors**: Alexander von Recum, Christoph Schnabl, Gabor Hollbeck, Silas Alberti, Philip Blinde, Marvin von Hagen  

**Link**: [PDF](https://arxiv.org/pdf/2412.16974)  

**Abstract**: Refusals - instances where large language models (LLMs) decline or fail to fully execute user instructions - are crucial for both AI safety and AI capabilities and the reduction of hallucinations in particular. These behaviors are learned during post-training, especially in instruction fine-tuning (IFT) and reinforcement learning from human feedback (RLHF). However, existing taxonomies and evaluation datasets for refusals are inadequate, often focusing solely on should-not-related (instead of cannot-related) categories, and lacking tools for auditing refusal content in black-box LLM outputs.
We present a comprehensive framework for classifying LLM refusals: (a) a taxonomy of 16 refusal categories, (b) a human-annotated dataset of over 8,600 instances from publicly available IFT and RLHF datasets, (c) a synthetic dataset with 8,000 examples for each refusal category, and (d) classifiers trained for refusal classification.
Our work enables precise auditing of refusal behaviors in black-box LLMs and automatic analyses of refusal patterns in large IFT and RLHF datasets. This facilitates the strategic adjustment of LLM refusals, contributing to the development of more safe and reliable LLMs. 

**Abstract (ZH)**: 拒绝行为——大型语言模型（LLM）在执行用户指令时的部分拒绝或未能完全执行的行为——对于人工智能安全和能力至关重要，特别是对于减少幻觉现象尤为重要。这些行为主要在后训练阶段学习，尤其是在指令微调（IFT）和人类反馈强化学习（RLHF）中。然而，现有的拒绝行为分类和评估数据集是不完善的，通常仅关注“不应做”的类别（而非“不能做”的类别），并且缺乏对黑盒LLM输出中拒绝内容进行审计的工具。

我们提出了一种全面的LLM拒绝行为分类框架：(a) 包含16个拒绝类别的分类体系；(b) 一个由超过8,600个实例组成的人类标注数据集，这些实例来自公开的指令微调（IFT）和人类反馈强化学习（RLHF）数据集；(c) 每个拒绝类别含有8,000个合成实例的数据集；以及(d) 用于拒绝行为分类的训练分类器。

我们的工作使得能够对黑盒LLM中的拒绝行为进行精确审计，并自动分析大型IFT和RLHF数据集中的拒绝模式。这有助于制定策略性调整LLM的拒绝行为，从而推动更加安全可靠的LLM的发展。 

---
# System-2 Mathematical Reasoning via Enriched Instruction Tuning 

**Title (ZH)**: 系统2数学推理通过丰富指导调优 

**Authors**: Huanqia Cai, Yijun Yang, Zhifeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2412.16964)  

**Abstract**: Solving complex mathematical problems via system-2 reasoning is a natural human skill, yet it remains a significant challenge for current large language models (LLMs). We identify the scarcity of deliberate multi-step reasoning data as a primary limiting factor. To this end, we introduce Enriched Instruction Tuning (EIT), a method that enriches existing human-annotated mathematical datasets by synergizing human and AI feedback to create fine-grained reasoning trajectories. These datasets are then used to fine-tune open-source LLMs, enhancing their mathematical reasoning abilities without reliance on any symbolic verification program. Concretely, EIT is composed of two critical steps: Enriching with Reasoning Plan (ERP) and Enriching with Reasoning Step (ERS). The former generates a high-level plan that breaks down complex instructions into a sequence of simpler objectives, while ERS fills in reasoning contexts often overlooked by human annotators, creating a smoother reasoning trajectory for LLM fine-tuning. Unlike existing CoT prompting methods that generate reasoning chains only depending on LLM's internal knowledge, our method leverages human-annotated initial answers as ``meta-knowledge'' to help LLMs generate more detailed and precise reasoning processes, leading to a more trustworthy LLM expert for complex mathematical problems. In experiments, EIT achieves an accuracy of 84.1\% on GSM8K and 32.5\% on MATH, surpassing state-of-the-art fine-tuning and prompting methods, and even matching the performance of tool-augmented methods. 

**Abstract (ZH)**: 通过系统2推理解决复杂的数学问题是人类的一项自然技能，但目前仍然是现有大规模语言模型（LLMs）的一项重大挑战。我们识别出缺乏刻意的多步推理数据是主要限制因素。为此，我们提出了一种名为增强指令调优（Enriched Instruction Tuning, EIT）的方法，这种方法通过结合人类和AI反馈来丰富现有的手工标注数学数据集，从而创建精细的推理轨迹。这些数据集随后被用来微调开源的LLMs，以增强其数学推理能力，而不依赖于任何符号验证程序。具体而言，EIT 包括两个关键步骤：增强推理计划（Enriching with Reasoning Plan, ERP）和增强推理步骤（Enriching with Reasoning Step, ERS）。前一步骤生成一个高层次的计划，将复杂的指令分解为一系列简单的目标，而后一步骤则填补人类注释者经常忽略的推理上下文，为LLM微调创建更平滑的推理轨迹。与现有的仅依赖LLM内部知识生成推理链的CoT提示方法不同，我们的方法利用手工标注的初始答案作为“元知识”来帮助LLM生成更详细和精确的推理过程，从而为复杂数学问题提供更可信的LLM专家。在实验中，EIT在GSM8K上的准确率为84.1%，在MATH上的准确率为32.5%，超越了最先进的微调和提示方法，并且匹配工具辅助方法的表现。 

---
# Towards a Unified Paradigm: Integrating Recommendation Systems as a New Language in Large Models 

**Title (ZH)**: 向着统一范式的迈进：将推荐系统纳入大型模型的新语言 

**Authors**: Kai Zheng, Qingfeng Sun, Can Xu, Peng Yu, Qingwei Guo  

**Link**: [PDF](https://arxiv.org/pdf/2412.16933)  

**Abstract**: This paper explores the use of Large Language Models (LLMs) for sequential recommendation, which predicts users' future interactions based on their past behavior. We introduce a new concept, "Integrating Recommendation Systems as a New Language in Large Models" (RSLLM), which combines the strengths of traditional recommenders and LLMs. RSLLM uses a unique prompting method that combines ID-based item embeddings from conventional recommendation models with textual item features. It treats users' sequential behaviors as a distinct language and aligns the ID embeddings with the LLM's input space using a projector. We also propose a two-stage LLM fine-tuning framework that refines a pretrained LLM using a combination of two contrastive losses and a language modeling loss. The LLM is first fine-tuned using text-only prompts, followed by target domain fine-tuning with unified prompts. This trains the model to incorporate behavioral knowledge from the traditional sequential recommender into the LLM. Our empirical results validate the effectiveness of our proposed framework. 

**Abstract (ZH)**: 本文探讨了大型语言模型（LLMs）在序列推荐领域的应用，该模型根据用户的过往行为预测其未来互动。我们提出了一种新的概念，即“将推荐系统作为大型模型中的新语言”（RSLLM，推荐系统语言模型），该概念结合了传统推荐系统和LLMs的优势。RSLLM采用了一种独特的提示方法，将传统推荐模型中的基于ID的项目嵌入与文本项目特征结合在一起。它将用户的行为序列视为一种独特的语言，并通过投影器将ID嵌入与LLM的输入空间对齐。我们还提出了一种两阶段的LLM微调框架，该框架利用两种对比损失和语言模型损失的组合对预训练的LLM进行微调。首先使用仅文本的提示方法进行LLM的微调，然后使用统一的提示方法进行目标领域微调。这训练模型将传统序列推荐器中的行为知识融入到LLM中。我们的实证结果验证了所提框架的有效性。 

---
# Quantifying Public Response to COVID-19 Events: Introducing the Community Sentiment and Engagement Index 

**Title (ZH)**: 量化公众对COVID-19事件的反应：介绍社区情绪与参与指数 

**Authors**: Nirmalya Thakur, Kesha A. Patel, Audrey Poon, Shuqi Cui, Nazif Azizi, Rishika Shah, Riyan Shah  

**Link**: [PDF](https://arxiv.org/pdf/2412.16925)  

**Abstract**: This study introduces the Community Sentiment and Engagement Index (CSEI), developed to capture nuanced public sentiment and engagement variations on social media, particularly in response to major events related to COVID-19. Constructed with diverse sentiment indicators, CSEI integrates features like engagement, daily post count, compound sentiment, fine-grain sentiments (fear, surprise, joy, sadness, anger, disgust, and neutral), readability, offensiveness, and domain diversity. Each component is systematically weighted through a multi-step Principal Component Analysis (PCA)-based framework, prioritizing features according to their variance contributions across temporal sentiment shifts. This approach dynamically adjusts component importance, enabling CSEI to precisely capture high-sensitivity shifts in public sentiment. The development of CSEI showed statistically significant correlations with its constituent features, underscoring internal consistency and sensitivity to specific sentiment dimensions. CSEI's responsiveness was validated using a dataset of 4,510,178 Reddit posts about COVID-19. The analysis focused on 15 major events, including the WHO's declaration of COVID-19 as a pandemic, the first reported cases of COVID-19 across different countries, national lockdowns, vaccine developments, and crucial public health measures. Cumulative changes in CSEI revealed prominent peaks and valleys aligned with these events, indicating significant patterns in public sentiment across different phases of the pandemic. Pearson correlation analysis further confirmed a statistically significant relationship between CSEI daily fluctuations and these events (p = 0.0428), highlighting the capacity of CSEI to infer and interpret shifts in public sentiment and engagement in response to major events related to COVID-19. 

**Abstract (ZH)**: 本文介绍了社区情绪和参与指数（CSEI），该指数旨在捕捉社交媒体上针对与COVID-19相关重大事件的公众情绪和参与度的细微变化。CSEI利用多样化的情绪指标，整合了参与度、每日发帖数量、复合情绪、细粒度情绪（恐惧、惊讶、喜悦、悲伤、愤怒、厌恶和中立）、可读性、不适当性以及领域多样性等特征。每个组成部分通过基于主成分分析（PCA）的多层次框架系统地赋予权重，根据其在时间情绪变化中的方差贡献优先考虑特征。这种方法动态调整各个组成部分的重要性，使CSEI能够精确捕捉公众情绪的高灵敏度变化。CSEI的发展显示了其组成部分之间的统计显著相关性，这一点表明内部一致性并对其特定情绪维度的敏感性。CSEI的响应性通过使用包含4,510,178篇关于COVID-19的Reddit帖子的数据集进行了验证。分析集中在15个重要事件上，包括世界卫生组织宣布COVID-19为大流行病、不同国家首次报告COVID-19病例、全国封锁、疫苗开发以及关键的公共卫生措施。CSEI累计变化量显示出与这些事件相一致的显著峰值和低谷，表明了不同疫情阶段公众情绪模式中的显著变化。皮尔森相关分析进一步证实了CSEI每日波动与这些事件之间存在统计显著相关关系（p = 0.0428），突显了CSEI在解释和推断公众情绪和参与度对与COVID-19相关重大事件的响应方面的能力。 

---
# Speech-Based Depression Prediction Using Encoder-Weight-Only Transfer Learning and a Large Corpus 

**Title (ZH)**: 基于编码器权重迁移学习和大规模语料库的语音抑郁症预测 

**Authors**: Amir Harati, Elizabeth Shriberg, Tomasz Rutowski, Piotr Chlebek, Yang Lu, Ricardo Oliveira  

**Link**: [PDF](https://arxiv.org/pdf/2412.16900)  

**Abstract**: Speech-based algorithms have gained interest for the management of behavioral health conditions such as depression. We explore a speech-based transfer learning approach that uses a lightweight encoder and that transfers only the encoder weights, enabling a simplified run-time model. Our study uses a large data set containing roughly two orders of magnitude more speakers and sessions than used in prior work. The large data set enables reliable estimation of improvement from transfer learning. Results for the prediction of PHQ-8 labels show up to 27% relative performance gains for binary classification; these gains are statistically significant with a p-value close to zero. Improvements were also found for regression. Additionally, the gain from transfer learning does not appear to require strong source task performance. Results suggest that this approach is flexible and offers promise for efficient implementation. 

**Abstract (ZH)**: 基于语音的算法在管理抑郁等行为健康状况方面引起了关注。我们探索了一种轻量级编码器为基础的迁移学习方法，该方法仅转移编码器权重，从而简化了运行时模型。本研究使用了一个包含约比以往工作多两个数量级的说话者和会话的大数据集。该大数据集使得能够可靠地估计迁移学习的效果。预测PHQ-8标签的结果表明，在二分类中可获得高达27%的相对性能提升；这些提升在p值接近零的情况下具有统计学意义。此外，迁移学习带来的改进也不必依赖于源任务的强大性能。结果表明，该方法具有灵活性，并且有可能实现高效的部署。 

---
# PsychAdapter: Adapting LLM Transformers to Reflect Traits, Personality and Mental Health 

**Title (ZH)**: PsychAdapter: 将大语言模型变换器适应以反映人格特质、个性和心理健康 

**Authors**: Huy Vu, Huy Anh Nguyen, Adithya V Ganesan, Swanie Juhng, Oscar N.E. Kjell, Joao Sedoc, Margaret L. Kern, Ryan L. Boyd, Lyle Ungar, H. Andrew Schwartz, Johannes C. Eichstaedt  

**Link**: [PDF](https://arxiv.org/pdf/2412.16882)  

**Abstract**: Artificial intelligence-based language generators are now a part of most people's lives. However, by default, they tend to generate "average" language without reflecting the ways in which people differ. Here, we propose a lightweight modification to the standard language model transformer architecture - "PsychAdapter" - that uses empirically derived trait-language patterns to generate natural language for specified personality, demographic, and mental health characteristics (with or without prompting). We applied PsychAdapters to modify OpenAI's GPT-2, Google's Gemma, and Meta's Llama 3 and found generated text to reflect the desired traits. For example, expert raters evaluated PsychAdapter's generated text output and found it matched intended trait levels with 87.3% average accuracy for Big Five personalities, and 96.7% for depression and life satisfaction. PsychAdapter is a novel method to introduce psychological behavior patterns into language models at the foundation level, independent of prompting, by influencing every transformer layer. This approach can create chatbots with specific personality profiles, clinical training tools that mirror language associated with psychological conditionals, and machine translations that match an authors reading or education level without taking up LLM context windows. PsychAdapter also allows for the exploration psychological constructs through natural language expression, extending the natural language processing toolkit to study human psychology. 

**Abstract (ZH)**: 基于人工智能的语言生成器已融入了大多数人的生活中。然而，默认情况下，它们倾向于生成“平均水平”的语言，而不反映人们之间的差异。在此，我们提出了一种对标准语言模型变换器架构的轻量级修改——“PsychAdapter”——该架构利用验证过的特质-语言模式来生成符合特定人格、人口统计学和心理健康特征的自然语言（有或无提示）。我们应用PsychAdapter对OpenAI的GPT-2、Google的Gemma和Meta的Llama 3进行了修改，并发现生成的文本反映了所需的特质。例如，专家评分者评估了PsychAdapter生成的文本输出，并发现它在五大人格特质方面的匹配平均准确率为87.3%，在抑郁和生活满意度方面的匹配准确率为96.7%。PsychAdapter是一种新颖的方法，可以在基础层面将心理学行为模式引入语言模型中，而无需提示，并通过影响每个变换器层来实现这一点。这种方法可以创建具有特定人格特征的聊天机器人，制作能够反映心理健康状况的语言的临床训练工具，以及匹配作者阅读水平或教育水平的机器翻译，而无需占用LLM的上下文窗口。此外，PsychAdapter还允许通过自然语言表达探索心理结构，扩展自然语言处理工具包以研究人类心理学。 

---
# Autoregressive Speech Synthesis with Next-Distribution Prediction 

**Title (ZH)**: 基于后续分布预测的自回归语音合成 

**Authors**: Xinfa Zhu, Wenjie Tian, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2412.16846)  

**Abstract**: We introduce KALL-E, a novel autoregressive (AR) language modeling approach with next-distribution prediction for text-to-speech (TTS) synthesis. Unlike existing methods, KALL-E directly models and predicts the continuous speech distribution conditioned on text without relying on VAE- or diffusion-based components. Specifically, we use WaveVAE to extract continuous speech distributions from waveforms instead of using discrete speech tokens. A single AR language model predicts these continuous speech distributions from text, with a Kullback-Leibler divergence loss as the constraint. Experimental results show that KALL-E outperforms open-source implementations of YourTTS, VALL-E, NaturalSpeech 2, and CosyVoice in terms of naturalness and speaker similarity in zero-shot TTS scenarios. Moreover, KALL-E demonstrates exceptional zero-shot capabilities in emotion and accent cloning. Importantly, KALL-E presents a more straightforward and effective paradigm for using continuous speech representations in TTS. Audio samples are available at: \url{this https URL}. 

**Abstract (ZH)**: 我们介绍了一种新颖的自回归（AR）语言建模方法KALL-E，该方法通过下一个分布预测（Next-distribution Prediction）用于文本到语音（TTS）合成。与现有方法不同，KALL-E 直接建模和预测文本条件下的连续语音分布，而不依赖于基于 VAE 或扩散模型的组件。具体而言，我们使用 WaveVAE 从波形中提取连续的语音分布，而不是使用离散的语音令牌。一个单一的AR语言模型从文本中预测这些连续的语音分布，并以Kullback-Leibler（KL）散度损失作为约束条件。实验结果显示，在零样本TTS场景中，KALL-E 在自然度和说话人相似性方面优于开源实现的YourTTS、VALL-E、NaturalSpeech 2 和 CosyVoice。此外，KALL-E 在情感和方言克隆方面展示了出色的零样本能力。重要的是，KALL-E 展示了一种更为直接和有效的使用连续语音表示的TTS范式。音频样本可在以下链接获取：\url{this https URL}。 

---
# AlzheimerRAG: Multimodal Retrieval Augmented Generation for PubMed articles 

**Title (ZH)**: 阿尔茨海默病RAG：面向PubMed文章的多模态检索增强生成 

**Authors**: Aritra Kumar Lahiri, Qinmin Vivian Hu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16701)  

**Abstract**: Recent advancements in generative AI have flourished the development of highly adept Large Language Models (LLMs) that integrate diverse data types to empower decision-making. Among these, Multimodal Retrieval-Augmented Generation (RAG) applications are promising for their capability to combine the strengths of information retrieval and generative models, enhancing their utility across various domains, including biomedical research. This paper introduces AlzheimerRAG, a Multimodal RAG pipeline tool for biomedical research use cases, primarily focusing on Alzheimer's disease from PubMed articles. Our pipeline incorporates multimodal fusion techniques to integrate textual and visual data processing by efficiently indexing and accessing vast amounts of biomedical literature. Preliminary experimental results against benchmarks, such as BioASQ and PubMedQA, have returned improved results in information retrieval and synthesis of domain-specific information. We also demonstrate a case study with our RAG pipeline across different Alzheimer's clinical scenarios. We infer that AlzheimerRAG can generate responses with accuracy non-inferior to humans and with low rates of hallucination. Overall, a reduction in cognitive task load is observed, which allows researchers to gain multimodal insights, improving understanding and treatment of Alzheimer's disease. 

**Abstract (ZH)**: 近年来，生成型人工智能的最新进展促进了高度擅长的大规模语言模型（Large Language Models, LLMs）的发展，这些模型能够综合各种数据类型，以增强决策支持能力。其中，多模态检索增强生成（Multimodal Retrieval-Augmented Generation, RAG）应用因其结合信息检索和生成模型的优势而充满潜力，从而在包括生物医学研究等各个领域中提高了其应用价值。本文介绍了一款名为AlzheimerRAG的多模态RAG流水线工具，主要应用于PubMed文献中的阿尔茨海默病研究场景。我们的流水线采用了多模态融合技术，通过高效地索引和访问大量生物医学文献，实现了文本和视觉数据的综合处理。初步实验结果表明，我们的流水线在生物医学领域的信息检索和相关信息综合方面均优于BioASQ和PubMedQA等基准。我们还展示了在不同阿尔茨海默病临床场景中使用我们RAG流水线的案例研究。我们推断，AlzheimerRAG可以生成准确性不低于人类的响应，并且产生幻觉的概率较低。总体而言，认知任务负担的减少使得研究人员能够获得多模态洞察，从而提高对阿尔茨海默病的理解和治疗。 

---
# The Task Shield: Enforcing Task Alignment to Defend Against Indirect Prompt Injection in LLM Agents 

**Title (ZH)**: 任务保护罩：强制任务对齐以防御间接提示注入攻击在LLM代理中的应用 

**Authors**: Feiran Jia, Tong Wu, Xin Qin, Anna Squicciarini  

**Link**: [PDF](https://arxiv.org/pdf/2412.16682)  

**Abstract**: Large Language Model (LLM) agents are increasingly being deployed as conversational assistants capable of performing complex real-world tasks through tool integration. This enhanced ability to interact with external systems and process various data sources, while powerful, introduces significant security vulnerabilities. In particular, indirect prompt injection attacks pose a critical threat, where malicious instructions embedded within external data sources can manipulate agents to deviate from user intentions. While existing defenses based on rule constraints, source spotlighting, and authentication protocols show promise, they struggle to maintain robust security while preserving task functionality. We propose a novel and orthogonal perspective that reframes agent security from preventing harmful actions to ensuring task alignment, requiring every agent action to serve user objectives. Based on this insight, we develop Task Shield, a test-time defense mechanism that systematically verifies whether each instruction and tool call contributes to user-specified goals. Through experiments on the AgentDojo benchmark, we demonstrate that Task Shield reduces attack success rates (2.07\%) while maintaining high task utility (69.79\%) on GPT-4o. 

**Abstract (ZH)**: 大型语言模型（LLM）代理越来越多地被部署为能够通过工具集成执行复杂现实任务的对话助手。这种增强的与外部系统交互和处理多种数据源的能力虽然强大，但也引入了显著的安全漏洞。特别是，间接提示注入攻击构成了一个关键威胁，其中嵌入在外部数据源中的恶意指令可以使代理偏离用户的意图。尽管基于规则约束、来源高亮和身份验证协议的现有防御措施显示出前景，但它们在保持安全性和保护任务功能方面存在困难。我们提出了一种新颖且独立的视角，重新定义了代理安全的焦点，从防止有害行为转移到确保任务对齐。根据这一认识，我们开发了Task Shield，这是一种运行时防御机制，系统地验证每个指令和工具调用是否有助于用户指定的目标。通过在AgentDojo基准测试上的实验，我们证明Task Shield在减少攻击成功率（2.07%）的同时，仍然保持了高任务实用性（69.79%）在GPT-4o上。 

---
# Large Language Model Can Be a Foundation for Hidden Rationale-Based Retrieval 

**Title (ZH)**: 大规模语言模型可以作为基于隐藏推理的检索的基础 

**Authors**: Luo Ji, Feixiang Guo, Teng Chen, Qingqing Gu, Xiaoyu Wang, Ningyuan Xi, Yihong Wang, Peng Yu, Yue Zhao, Hongyang Lei, Zhonglin Jiang, Yong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.16615)  

**Abstract**: Despite the recent advancement in Retrieval-Augmented Generation (RAG) systems, most retrieval methodologies are often developed for factual retrieval, which assumes query and positive documents are semantically similar. In this paper, we instead propose and study a more challenging type of retrieval task, called hidden rationale retrieval, in which query and document are not similar but can be inferred by reasoning chains, logic relationships, or empirical experiences. To address such problems, an instruction-tuned Large language model (LLM) with a cross-encoder architecture could be a reasonable choice. To further strengthen pioneering LLM-based retrievers, we design a special instruction that transforms the retrieval task into a generative task by prompting LLM to answer a binary-choice question. The model can be fine-tuned with direct preference optimization (DPO). The framework is also optimized for computational efficiency with no performance degradation. We name this retrieval framework by RaHoRe and verify its zero-shot and fine-tuned performance superiority on Emotional Support Conversation (ESC), compared with previous retrieval works. Our study suggests the potential to employ LLM as a foundation for a wider scope of retrieval tasks. Our codes, models, and datasets are available on this https URL. 

**Abstract (ZH)**: 尽管最近在检索增强生成（RAG）系统方面取得了一定进展，大多数检索方法通常针对事实检索进行开发，这种检索方法假设查询和正相关文档在语义上是相似的。在本文中，我们反而提出并研究了一种更具挑战性的检索任务类型，称为隐藏理由检索，在这种任务中，查询和文档本身并不相似，但可以通过推理链、逻辑关系或经验进行推断。为了解决这类问题，带有交叉编码器架构的指令调优大型语言模型（LLM）可能是一个合理的选择。为了进一步增强基于LLM的检索者，我们设计了一个特殊的指令，通过提示LLM回答二元选择问题，将检索任务转化为生成任务。该模型可以通过直接偏好优化（DPO）进行微调。该框架还在保证性能不降级的情况下优化了计算效率。我们将这种检索框架命名为RaHoRe，并通过将其与先前的检索工作在情感支持对话（ESC）上的零样本和微调性能进行比较，验证了其优越性。我们的研究表明，大型语言模型有可能作为更广泛检索任务的基础。我们的代码、模型和数据集可通过以下链接获取：[此处链接]。 

---
# Open-Vocabulary Mobile Manipulation Based on Double Relaxed Contrastive Learning with Dense Labeling 

**Title (ZH)**: 基于双松弛对比学习和密集标签的开放式词汇移动操作-Manipulation 

**Authors**: Daichi Yashima, Ryosuke Korekata, Komei Sugiura  

**Link**: [PDF](https://arxiv.org/pdf/2412.16576)  

**Abstract**: Growing labor shortages are increasing the demand for domestic service robots (DSRs) to assist in various settings. In this study, we develop a DSR that transports everyday objects to specified pieces of furniture based on open-vocabulary instructions. Our approach focuses on retrieving images of target objects and receptacles from pre-collected images of indoor environments. For example, given an instruction "Please get the right red towel hanging on the metal towel rack and put it in the white washing machine on the left," the DSR is expected to carry the red towel to the washing machine based on the retrieved images. This is challenging because the correct images should be retrieved from thousands of collected images, which may include many images of similar towels and appliances. To address this, we propose RelaX-Former, which learns diverse and robust representations from among positive, unlabeled positive, and negative samples. We evaluated RelaX-Former on a dataset containing real-world indoor images and human annotated instructions including complex referring expressions. The experimental results demonstrate that RelaX-Former outperformed existing baseline models across standard image retrieval metrics. Moreover, we performed physical experiments using a DSR to evaluate the performance of our approach in a zero-shot transfer setting. The experiments involved the DSR to carry objects to specific receptacles based on open-vocabulary instructions, achieving an overall success rate of 75%. 

**Abstract (ZH)**: 随着劳动力短缺日益加剧，对用于各种环境的家用服务机器人（DSRs）的需求正在增加。本研究中，我们开发了一个DSR，它可以基于开放词汇的指令将日常生活中的物品运输到指定的家具位置。我们的方法专注于从预先收集的室内环境图像中检索目标物品和容器的图像。例如，给定指令“请取挂在金属毛巾架上的红色毛巾，并将其放在左边的白色洗衣机里”，DSR 需要根据检索到的图像将红色毛巾搬运到洗衣机位置。这一过程极具挑战性，因为需要从数千张收集图像中准确检索出正确的图像，其中可能包含许多相似毛巾和家用电器的图像。为此，我们提出了RelaX-Former，该模型能够从正样本、未标注的正样本和负样本中学习多样且稳健的表示。我们在包含真实世界室内图像和人类标注指令（包括复杂的指示表达）的数据集上评估了RelaX-Former。实验结果表明，RelaX-Former在标准图像检索指标方面表现优于现有的基线模型。此外，我们在一个零样本迁移设置中使用DSR执行了物理实验，评估我们方法的表现。实验涉及DSR根据开放词汇的指令将物品搬运至特定容器，最终取得了整体成功率75%的结果。 

---
# Self-guided Knowledgeable Network of Thoughts: Amplifying Reasoning with Large Language Models 

**Title (ZH)**: 自我引导的知识性思维网络：利用大规模语言模型强化推理 

**Authors**: Chao-Chi Chen, Chin-Yuan Yeh, Hsi-Wen Chen, De-Nian Yang, Ming-Syan Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.16533)  

**Abstract**: We introduce Knowledgeable Network of Thoughts (kNoT): a prompt scheme that advances the capabilities of large language models (LLMs) beyond existing paradigms like Chain-of-Thought (CoT), Tree of Thoughts (ToT), and Graph of Thoughts (GoT). The key innovation of kNoT is the LLM Workflow Template (LWT), which allows for an executable plan to be specified by LLMs for LLMs. LWT allows these plans to be arbitrary networks, where single-step LLM operations are nodes, and edges correspond to message passing between these steps. Furthermore, LWT supports selection of individual elements through indexing, facilitating kNoT to produce intricate plans where each LLM operation can be limited to elementary operations, greatly enhancing reliability over extended task sequences. We demonstrate that kNoT significantly outperforms the state of the art on six use cases, while reducing the need for extensive prompt engineering. For instance, kNoT finds 92% accuracy for sorting 32 numbers over 12% and 31% for ToT and GoT, while utilizing up to 84.4% and 87.3% less task-specific prompts, respectively. 

**Abstract (ZH)**: 我们介绍了一种名为Knowledgeable Network of Thoughts (kNoT)的提示方案：这一方案超越了现有的思维方式链（Chain-of-Thought, CoT）、思维树（Tree of Thoughts, ToT）和思维图（Graph of Thoughts, GoT）等范式，提升了大型语言模型（LLMs）的能力。kNoT的关键创新在于LLM工作流模板（LLM Workflow Template, LWT），它允许LLMs为LLMs指定可执行的计划。LWT允许这些计划表现为任意网络，其中单步骤LLM操作作为节点，边对应于这些步骤之间的信息传递。此外，LWT支持通过索引选择个别元素，使得kNoT能够生成复杂计划，其中每一步LLM操作可以限制为基本操作，这大大增强了长时间任务序列的可靠性。我们展示了kNoT在六个应用场景中显著优于现有最先进的方法，同时减少了对大量定制提示的依赖。例如，kNoT在对32个数字进行排序时达到了92%的准确率，而ToT和GoT的准确率分别为12%和31%，同时分别减少了多达84.4%和87.3%的任务特定提示。 

---
# Improving Lip-synchrony in Direct Audio-Visual Speech-to-Speech Translation 

**Title (ZH)**: 提高直接音视频同步的语音转语音翻译中的唇同步效果 

**Authors**: Lucas Goncalves, Prashant Mathur, Xing Niu, Brady Houston, Chandrashekhar Lavania, Srikanth Vishnubhotla, Lijia Sun, Anthony Ferritto  

**Link**: [PDF](https://arxiv.org/pdf/2412.16530)  

**Abstract**: Audio-Visual Speech-to-Speech Translation typically prioritizes improving translation quality and naturalness. However, an equally critical aspect in audio-visual content is lip-synchrony-ensuring that the movements of the lips match the spoken content-essential for maintaining realism in dubbed videos. Despite its importance, the inclusion of lip-synchrony constraints in AVS2S models has been largely overlooked. This study addresses this gap by integrating a lip-synchrony loss into the training process of AVS2S models. Our proposed method significantly enhances lip-synchrony in direct audio-visual speech-to-speech translation, achieving an average LSE-D score of 10.67, representing a 9.2% reduction in LSE-D over a strong baseline across four language pairs. Additionally, it maintains the naturalness and high quality of the translated speech when overlaid onto the original video, without any degradation in translation quality. 

**Abstract (ZH)**: 视听语音翻译通常侧重于提高翻译质量和自然度。然而，在视听内容中同样重要的是唇同步，即确保嘴唇的运动与所说内容一致，这对于保持配音视频的真实性至关重要。尽管唇同步极为重要，但在视听语音翻译（Audio-Visual Speech-to-Speech Translation, AVS2S）模型中纳入唇同步约束的研究却相对较少。本研究通过将唇同步损失纳入AVS2S模型的训练过程来弥补这一不足。我们提出的方法显著提高了直接视听语音翻译中的唇同步效果，平均LSE-D得分为10.67，与四个语言对的强大基线相比，LSE-D降低了9.2%。此外，当将翻译后的语音叠加到原始视频上时，该方法能够保持翻译语音的自然度和高质量，而不牺牲翻译质量。 

---
# Text2midi: Generating Symbolic Music from Captions 

**Title (ZH)**: Text2MIDI：从描述生成符号音乐 

**Authors**: Keshav Bhandari, Abhinaba Roy, Kyra Wang, Geeta Puri, Simon Colton, Dorien Herremans  

**Link**: [PDF](https://arxiv.org/pdf/2412.16526)  

**Abstract**: This paper introduces text2midi, an end-to-end model to generate MIDI files from textual descriptions. Leveraging the growing popularity of multimodal generative approaches, text2midi capitalizes on the extensive availability of textual data and the success of large language models (LLMs). Our end-to-end system harnesses the power of LLMs to generate symbolic music in the form of MIDI files. Specifically, we utilize a pretrained LLM encoder to process captions, which then condition an autoregressive transformer decoder to produce MIDI sequences that accurately reflect the provided descriptions. This intuitive and user-friendly method significantly streamlines the music creation process by allowing users to generate music pieces using text prompts. We conduct comprehensive empirical evaluations, incorporating both automated and human studies, that show our model generates MIDI files of high quality that are indeed controllable by text captions that may include music theory terms such as chords, keys, and tempo. We release the code and music samples on our demo page (this https URL) for users to interact with text2midi. 

**Abstract (ZH)**: 本文介绍了text2midi，这是一种端到端模型，能够从文本描述生成MIDI文件。利用多模态生成方法日益增长的流行趋势，text2midi利用了大量的文本数据和大型语言模型（LLMs）的成功经验。我们的端到端系统利用LLMs生成符号化的音乐，以MIDI文件的形式呈现。具体来说，我们使用预训练的LLM编码器处理描述文本，然后通过自回归变压器解码器生成MIDI序列，这些序列准确反映了所提供的描述。这一直观且用户友好的方法显著简化了音乐创作过程，使用户能够通过文本提示生成音乐作品。我们进行了全面的经验实证评价，结合了自动和人工研究，结果显示，我们的模型能够生成高质量的MIDI文件，这些文件确实可以通过包括和弦、调性和节拍在内的文本描述进行控制。我们已在演示页面上发布了该模型的代码和音乐样本（请点击此处：this https URL），供用户与text2midi互动。 

---
# Speech Retrieval-Augmented Generation without Automatic Speech Recognition 

**Title (ZH)**: 无需自动语音识别的演讲检索增强生成 

**Authors**: Do June Min, Karel Mundnich, Andy Lapastora, Erfan Soltanmohammadi, Srikanth Ronanki, Kyu Han  

**Link**: [PDF](https://arxiv.org/pdf/2412.16500)  

**Abstract**: One common approach for question answering over speech data is to first transcribe speech using automatic speech recognition (ASR) and then employ text-based retrieval-augmented generation (RAG) on the transcriptions. While this cascaded pipeline has proven effective in many practical settings, ASR errors can propagate to the retrieval and generation steps. To overcome this limitation, we introduce SpeechRAG, a novel framework designed for open-question answering over spoken data. Our proposed approach fine-tunes a pre-trained speech encoder into a speech adapter fed into a frozen large language model (LLM)--based retrieval model. By aligning the embedding spaces of text and speech, our speech retriever directly retrieves audio passages from text-based queries, leveraging the retrieval capacity of the frozen text retriever. Our retrieval experiments on spoken question answering datasets show that direct speech retrieval does not degrade over the text-based baseline, and outperforms the cascaded systems using ASR. For generation, we use a speech language model (SLM) as a generator, conditioned on audio passages rather than transcripts. Without fine-tuning of the SLM, this approach outperforms cascaded text-based models when there is high WER in the transcripts. 

**Abstract (ZH)**: 对于语音数据上的问答任务，一种常见的方法是首先使用自动语音识别（ASR）进行语音转录，然后在转录内容上使用文本检索增强生成（RAG）方法。尽管这种级联管道在许多实际应用中已被证明是有效的，但ASR错误可能会传播到检索和生成步骤。为了克服这一局限，我们提出了一种名为SpeechRAG的新框架，专门用于处理口头数据的开放性问答任务。我们的方法是对预训练的语音编码器进行微调，使其成为嵌入语音适配器并输入冻结的大规模语言模型（LLM）为基础的检索模型。通过对齐文本和语音的嵌入空间，我们的语音检索器可以直接从基于文本查询中检索音频片段，利用冻结文本检索器的检索能力。我们在口语问答数据集上的检索实验表明，直接语音检索不会劣于基于文本的基线方法，并且在使用ASR时，超越了级联系统。在生成阶段，我们使用语音语言模型（SLM）作为生成器，基于音频片段而非转录内容。在不微调SLM的情况下，该方法在转录的错误率较高时，优于基于文本的级联模型。 

---
# Automated CVE Analysis: Harnessing Machine Learning In Designing Question-Answering Models For Cybersecurity Information Extraction 

**Title (ZH)**: 自动化CVE分析：利用机器学习设计网络安全信息提取的问答模型 

**Authors**: Tanjim Bin Faruk  

**Link**: [PDF](https://arxiv.org/pdf/2412.16484)  

**Abstract**: The vast majority of cybersecurity information is unstructured text, including critical data within databases such as CVE, NVD, CWE, CAPEC, and the MITRE ATT&CK Framework. These databases are invaluable for analyzing attack patterns and understanding attacker behaviors. Creating a knowledge graph by integrating this information could unlock significant insights. However, processing this large amount of data requires advanced deep-learning techniques. A crucial step towards building such a knowledge graph is developing a robust mechanism for automating the extraction of answers to specific questions from the unstructured text. Question Answering (QA) systems play a pivotal role in this process by pinpointing and extracting precise information, facilitating the mapping of relationships between various data points. In the cybersecurity context, QA systems encounter unique challenges due to the need to interpret and answer questions based on a wide array of domain-specific information. To tackle these challenges, it is necessary to develop a cybersecurity-specific dataset and train a machine learning model on it, aimed at enhancing the understanding and retrieval of domain-specific information. This paper presents a novel dataset and describes a machine learning model trained on this dataset for the QA task. It also discusses the model's performance and key findings in a manner that maintains a balance between formality and accessibility. 

**Abstract (ZH)**: 大多数网络安全信息是以未结构化文本形式存在的，包括诸如CVE、NVD、CWE、CAPEC及MITRE ATT&CK框架等数据库中的关键数据。这些数据库对于分析攻击模式和理解攻击者行为具有极大的价值。通过整合这些信息构建知识图谱，可以揭示许多重要的洞察。然而，处理如此大量的数据需要先进的深度学习技术。构建这样一种知识图谱的关键步骤之一是开发一种强大的机制，用于自动化从未结构化文本中提取特定问题的答案。基于问答（Question Answering, QA）系统的角色至关重要，它能够精准地定位并提取信息，促进不同数据点之间的关系映射。在网络安全领域，QA系统面临独特的挑战，因为它们需要基于广泛的专业领域信息来理解和回答问题。为了应对这些挑战，需要开发特定于网络安全的数据集，并在其中训练机器学习模型，以增强对领域特异性信息的理解和检索。本文介绍了一个新的数据集，并描述了一个在该数据集上训练的机器学习模型用于问答任务的情况。同时，本文讨论了模型的性能和关键发现，保持了形式性和易读性的平衡。 

---
# Enhancing Multilingual ASR for Unseen Languages via Language Embedding Modeling 

**Title (ZH)**: 通过语言嵌入建模增强对未见语言的多语言ASR 

**Authors**: Shao-Syuan Huang, Kuan-Po Huang, Andy T. Liu, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2412.16474)  

**Abstract**: Multilingual Automatic Speech Recognition (ASR) aims to recognize and transcribe speech from multiple languages within a single system. Whisper, one of the most advanced ASR models, excels in this domain by handling 99 languages effectively, leveraging a vast amount of data and incorporating language tags as prefixes to guide the recognition process. However, despite its success, Whisper struggles with unseen languages, those not included in its pre-training. Motivated by the observation that many languages share linguistic characteristics, we propose methods that exploit these relationships to enhance ASR performance on unseen languages. Specifically, we introduce a weighted sum method, which computes a weighted sum of the embeddings of language tags, using Whisper's predicted language probabilities. In addition, we develop a predictor-based approach that refines the weighted sum embedding to more closely approximate the true embedding for unseen languages. Experimental results demonstrate substantial improvements in ASR performance, both in zero-shot and fine-tuning settings. Our proposed methods outperform baseline approaches, providing an effective solution for addressing unseen languages in multilingual ASR. 

**Abstract (ZH)**: 多语言自动语音识别（ASR）旨在在一个系统中识别和转录多种语言的语音。Whisper 是最先进的 ASR 模型之一，它通过有效处理 99 种语言，并利用大量数据以及在语言标签前缀的帮助下指导识别过程，在这一领域表现出色。然而，尽管取得了成功，但 Whisper 在面对未见过的语言时仍然存在困难，即在预训练中未包含的语言。鉴于观察到许多语言具有语言特征上的共性，我们提出了一种方法，利用这些关系来提高对未见过语言的 ASR 性能。具体来说，我们引入了一种加权求和方法，该方法基于 Whisper 预测的语言概率计算语言标签嵌入的加权求和。此外，我们还开发了一种基于预测器的方法，该方法进一步改进加权求和嵌入，使其更接近未见过语言的真实嵌入。实验结果表明，在零样本和微调设置下，ASR 性能显著提高。我们提出的方法超越了基线方法，提供了一个有效的解决方案，以解决多语言 ASR 中未见过语言的问题。 

---
# Correcting Large Language Model Behavior via Influence Function 

**Title (ZH)**: 通过影响函数纠正大型语言模型的行为 

**Authors**: Han Zhang, Zhuo Zhang, Yi Zhang, Yuanzhao Zhai, Hanyang Peng, Yu Lei, Yue Yu, Hui Wang, Bin Liang, Lin Gui, Ruifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2412.16451)  

**Abstract**: Recent advancements in AI alignment techniques have significantly improved the alignment of large language models (LLMs) with static human preferences. However, the dynamic nature of human preferences can render some prior training data outdated or even erroneous, ultimately causing LLMs to deviate from contemporary human preferences and societal norms. Existing methodologies, whether they involve the curation of new data for continual alignment or the manual correction of outdated data for re-alignment, demand costly human resources. To address this challenge, we propose a novel approach, Large Language Model Behavior Correction with Influence Function Recall and Post-Training (LANCET), which requires no human involvement. LANCET consists of two phases: (1) using influence functions to identify the training data that significantly impact undesirable model outputs, and (2) applying an Influence function-driven Bregman Optimization (IBO) technique to adjust the model's behavior based on these influence distributions. Our experiments demonstrate that LANCET effectively and efficiently correct inappropriate behaviors of LLMs. Furthermore, LANCET can outperform methods that rely on collecting human preferences, and it enhances the interpretability of learning human preferences within LLMs. 

**Abstract (ZH)**: 近年来，人工智能对齐技术的进展显著提高了大型语言模型（LLMs）与静态人类偏好的一致性。然而，人类偏好的动态特性可能导致先前的训练数据变得过时甚至错误，最终导致LLMs偏离当前的人类偏好和社会规范。现有的方法，无论是通过持续收集新数据进行对齐，还是通过手动修正过时数据进行重新对齐，都依赖于昂贵的人力资源。为解决这一挑战，我们提出了一种新颖的方法——大型语言模型行为纠正结合影响函数召回和后处理（LANCET），该方法不需要人工干预。LANCET包括两个阶段：（1）使用影响函数来识别显著影响模型输出的关键训练数据，（2）应用影响函数驱动的布丹优化（IBO）技术，根据这些影响分布来调整模型行为。我们的实验表明，LANCET能够有效地和高效地纠正LLMs的不当行为。此外，LANCET在依赖于收集人类偏好值的方法中表现出更优的效果，并增强了在LLMs中学习人类偏好的可解释性。 

---
# Beyond End-to-End VLMs: Leveraging Intermediate Text Representations for Superior Flowchart Understanding 

**Title (ZH)**: 超越端到端的VLM：利用中间文本表示以实现更优秀的流程图理解 

**Authors**: Junyi Ye, Ankan Dash, Wenpeng Yin, Guiling Wang  

**Link**: [PDF](https://arxiv.org/pdf/2412.16420)  

**Abstract**: Flowcharts are typically presented as images, driving the trend of using vision-language models (VLMs) for end-to-end flowchart understanding. However, two key challenges arise: (i) Limited controllability--users have minimal influence over the downstream task, as they can only modify input images, while the training of VLMs is often out of reach for most researchers. (ii) Lack of explainability--it is difficult to trace VLM errors to specific causes, such as failures in visual encoding or reasoning. We propose TextFlow, addressing aforementioned issues with two stages: (i) Vision Textualizer--which generates textual representations from flowchart images; and (ii) Textual Reasoner--which performs question-answering based on the text representations. TextFlow offers three key advantages: (i) users can select the type of text representations (e.g., Graphviz, Mermaid, PlantUML), or further convert them into executable graph object to call tools, enhancing performance and controllability; (ii) it improves explainability by helping to attribute errors more clearly to visual or textual processing components; and (iii) it promotes the modularization of the solution, such as allowing advanced LLMs to be used in the Reasoner stage when VLMs underperform in end-to-end fashion. Experiments on the FlowVQA and FlowLearn benchmarks demonstrate TextFlow's state-of-the-art performance as well as its robustness. All code is publicly available. 

**Abstract (ZH)**: 流程图通常以图像形式呈现，这推动了使用视觉-语言模型（Vision-Language Models, VLMs）进行端到端流程图理解的趋势。然而，两个关键挑战随之而来：（i）控制能力有限——用户对下游任务的影响较小，他们只能修改输入图像，而大多数研究人员难以触及VLMs的训练过程。（ii）缺乏可解释性——难以追溯VLM错误的具体原因，例如视觉编码或推理的失败。为此，我们提出了TextFlow，通过两个阶段解决上述问题：（i）视觉文本化——从流程图图像生成文本表示；（ii）文本推理器——基于文本表示进行问答。TextFlow 提供了三大优势：（i）用户可以根据需要选择文本表示的类型（例如 Graphviz，Mermaid，PlantUML），甚至进一步转换为可执行的图对象调用工具，从而提升性能和可控性；（ii）它通过帮助更清晰地归因错误至视觉处理或文本处理组件，提高了可解释性；（iii）它促进了解决方案的模块化，例如当VLMs在端到端情况下表现不佳时，可以使用高级语言模型（LLMs）替代推理阶段。基准测试中的FlowVQA和FlowLearn实验展示了TextFlow的领先性能及其鲁棒性。所有代码均已公开。 

---
# Identifying Cyberbullying Roles in Social Media 

**Title (ZH)**: 识别社交媒体中的网络欺凌角色 

**Authors**: Manuel Sandoval, Mohammed Abuhamad, Patrick Furman, Mujtaba Nazari, Deborah L. Hall, Yasin N. Silva  

**Link**: [PDF](https://arxiv.org/pdf/2412.16417)  

**Abstract**: Social media has revolutionized communication, allowing people worldwide to connect and interact instantly. However, it has also led to increases in cyberbullying, which poses a significant threat to children and adolescents globally, affecting their mental health and well-being. It is critical to accurately detect the roles of individuals involved in cyberbullying incidents to effectively address the issue on a large scale. This study explores the use of machine learning models to detect the roles involved in cyberbullying interactions. After examining the AMiCA dataset and addressing class imbalance issues, we evaluate the performance of various models built with four underlying LLMs (i.e., BERT, RoBERTa, T5, and GPT-2) for role detection. Our analysis shows that oversampling techniques help improve model performance. The best model, a fine-tuned RoBERTa using oversampled data, achieved an overall F1 score of 83.5%, increasing to 89.3% after applying a prediction threshold. The top-2 F1 score without thresholding was 95.7%. Our method outperforms previously proposed models. After investigating the per-class model performance and confidence scores, we show that the models perform well in classes with more samples and less contextual confusion (e.g., Bystander Other), but struggle with classes with fewer samples (e.g., Bystander Assistant) and more contextual ambiguity (e.g., Harasser and Victim). This work highlights current strengths and limitations in the development of accurate models with limited data and complex scenarios. 

**Abstract (ZH)**: 社交媒体已经彻底改变了沟通方式，使世界各地的人们能够即时连接和互动。然而，它也导致了网络欺凌的增加，这在全球范围内对儿童和青少年构成了重大威胁，影响他们的心理健康和福祉。准确检测参与网络欺凌事件的个人角色至关重要，以便有效地大规模应对这一问题。本研究探讨了使用机器学习模型来检测网络欺凌互动中涉及的角色。在检查AMiCA数据集并解决类别不平衡问题后，我们评估了使用四款不同基础语言模型（即BERT、RoBERTa、T5和GPT-2）构建的各种模型的性能。我们的分析表明，过采样技术有助于提高模型性能。最佳模型是使用过采样数据微调的RoBERTa，在应用预测阈值后，其整体F1分数为89.3%，在未应用阈值的情况下，顶级F1分数为95.7%。我们的方法优于先前提出的方法。在探讨各分类模型性能和置信分数后，我们表明，模型在有更多样本且较少上下文混淆的类别（例如旁观者—其他）中表现良好，但在有较少样本且更多上下文含糊的类别（例如旁观者—助手）中表现较差。此外，在具有更多欺凌者和被害者等更复杂上下文的类别中也表现不佳。本研究突显了在数据有限和复杂场景条件下开发准确模型的当前优势和局限性。 

---
# REFA: Reference Free Alignment for multi-preference optimization 

**Title (ZH)**: REFA：无参考的多偏好优化对齐方法 

**Authors**: Taneesh Gupta, Rahul Madhavan, Xuchao Zhang, Chetan Bansal, Saravan Rajmohan  

**Link**: [PDF](https://arxiv.org/pdf/2412.16378)  

**Abstract**: We introduce REFA, a family of reference-free alignment methods that optimize over multiple user preferences while enforcing fine-grained length control. Our approach integrates deviation-based weighting to emphasize high-quality responses more strongly, length normalization to prevent trivial short-response solutions, and an EOS-probability regularizer to mitigate dataset-induced brevity biases. Theoretically, we show that under the Uncertainty Reduction with Sequence Length Assertion (URSLA), naive length normalization can still incentivize length-based shortcuts. By contrast, REFA corrects these subtle incentives, guiding models toward genuinely more informative and higher-quality outputs. Empirically, REFA sets a new state-of-the-art among reference-free alignment methods, producing richer responses aligned more closely with human preferences. Compared to a base supervised fine-tuned (SFT) mistral-7b model that achieves 8.4% length-controlled win rate (LC-WR) and 6.2% win rate (WR), our best REFA configuration attains 21.62% LC-WR and 19.87% WR on the AlpacaEval v2 benchmark. This represents a substantial improvement over both the strongest multi-preference baseline, InfoNCA (16.82% LC-WR, 10.44% WR), and the strongest reference-free baseline, SimPO (20.01% LC-WR, 17.65% WR) 

**Abstract (ZH)**: 我们引入了REFA，这是一种参考自由对齐方法的家族，能够在优化多个用户偏好的同时实施精细的长度控制。我们的方法集成了基于偏差的权重分配，以更加强调高质量的响应；长度归一化，以防止产生简单的短响应解决方案；以及EOS概率正则化器，以减轻由数据集引起的简短偏差。理论上，我们证明了在序列长度断言减少不确定性（URSLA）的框架下，朴素的长度归一化仍然可以激励基于长度的捷径。相比之下，REFA纠正了这些微妙的激励，引导模型产生更多真正有用且高质量的输出。实验上，REFA在参考自由对齐方法中设定了新的最先进水平，生成了与人类偏好更加一致且更加丰富的响应。与基线监督微调（SFT）的mistral-7b模型相比，该模型在长度控制胜率（LC-WR）和胜率（WR）方面的成绩分别为8.4%和6.2%，我们最好的REFA配置在AlpacaEval v2基准测试中分别取得了21.62%的LC-WR和19.87%的WR。这不仅超过了最强的多偏好基线InfoNCA（16.82%的LC-WR和10.44%的WR），也超过了最强的参考自由基线SimPO（20.01%的LC-WR和17.65%的WR）。 

---
# A High-Quality Text-Rich Image Instruction Tuning Dataset via Hybrid Instruction Generation 

**Title (ZH)**: 一种通过混合指令生成获取的高质量图文数据集 

**Authors**: Shijie Zhou, Ruiyi Zhang, Yufan Zhou, Changyou Chen  

**Link**: [PDF](https://arxiv.org/pdf/2412.16364)  

**Abstract**: Large multimodal models still struggle with text-rich images because of inadequate training data. Self-Instruct provides an annotation-free way for generating instruction data, but its quality is poor, as multimodal alignment remains a hurdle even for the largest models. In this work, we propose LLaVAR-2, to enhance multimodal alignment for text-rich images through hybrid instruction generation between human annotators and large language models. Specifically, it involves detailed image captions from human annotators, followed by the use of these annotations in tailored text prompts for GPT-4o to curate a dataset. It also implements several mechanisms to filter out low-quality data, and the resulting dataset comprises 424k high-quality pairs of instructions. Empirical results show that models fine-tuned on this dataset exhibit impressive enhancements over those trained with self-instruct data. 

**Abstract (ZH)**: 大型多模态模型在处理富文本图像方面仍然存在困难，原因在于训练数据不足。Self-Instruct提供了一种无需标注的方式来生成指令数据，但其质量较差，即使对于最大的模型，多模态对齐仍然是一个挑战。本工作中，我们提出了一种名为LLaVAR-2的方法，通过人类标注者和大型语言模型之间的混合指令生成来增强富文本图像的多模态对齐。具体来说，该方法首先利用人类标注者提供的详细图像描述，然后利用这些注释定制文本提示供GPT-4o用于构建数据集。此外，该方法还实现了多种机制来过滤低质量数据，最终构建了一个包含42.4万对高质量指令的数据集。实验结果表明，使用该数据集微调的模型在性能上显著优于使用Self-Instruct数据集训练的模型。 

---
# A Machine Learning Approach for Emergency Detection in Medical Scenarios Using Large Language Models 

**Title (ZH)**: 使用大型语言模型的医学场景中紧急情况检测的机器学习方法 

**Authors**: Ferit Akaybicen, Aaron Cummings, Lota Iwuagwu, Xinyue Zhang, Modupe Adewuyi  

**Link**: [PDF](https://arxiv.org/pdf/2412.16341)  

**Abstract**: The rapid identification of medical emergencies through digital communication channels remains a critical challenge in modern healthcare delivery, particularly with the increasing prevalence of telemedicine. This paper presents a novel approach leveraging large language models (LLMs) and prompt engineering techniques for automated emergency detection in medical communications. We developed and evaluated a comprehensive system using multiple LLaMA model variants (1B, 3B, and 7B parameters) to classify medical scenarios as emergency or non-emergency situations. Our methodology incorporated both system prompts and in-prompt training approaches, evaluated across different hardware configurations. The results demonstrate exceptional performance, with the LLaMA 2 (7B) model achieving 99.7% accuracy and the LLaMA 3.2 (3B) model reaching 99.6% accuracy with optimal prompt engineering. Through systematic testing of training examples within the prompts, we identified that including 10 example scenarios in the model prompts yielded optimal classification performance. Processing speeds varied significantly between platforms, ranging from 0.05 to 2.2 seconds per request. The system showed particular strength in minimizing high-risk false negatives in emergency scenarios, which is crucial for patient safety. The code implementation and evaluation framework are publicly available on GitHub, facilitating further research and development in this crucial area of healthcare technology. 

**Abstract (ZH)**: 通过数字通信渠道快速识别医疗紧急情况依然是现代医疗服务中的关键挑战，尤其是在远程医疗日益普及的情况下。本文提出了一种新的方法，利用大规模语言模型（LLMs）和提示工程技术，实现自动化的医疗紧急情况检测。我们开发并评估了一个综合系统，使用了多个LLaMA模型变体（1B、3B和7B参数）来分类医疗场景为紧急或非紧急情况。我们的方法论结合了系统提示和嵌入式提示训练方法，并在不同硬件配置下进行了评估。结果显示，LLaMA 2（7B）模型达到了99.7%的准确率，而LLaMA 3.2（3B）模型通过最佳的提示工程技术达到了99.6%的准确率。通过系统测试提示内的训练示例，我们发现包括10个示例场景在模型提示中能获得最佳分类性能。不同平台的处理速度差异显著，从每请求0.05秒到2.2秒不等。该系统在紧急情况下的高风险假阴性最小化方面表现出特别的优势，这对于患者安全至关重要。该系统的代码实现和评估框架已在GitHub上公开发布，以促进对该领域关键问题的进一步研究和开发。 

---
# Benchmarking LLMs and SLMs for patient reported outcomes 

**Title (ZH)**: 将下面的论文内容或标题翻译成中文，符合学术规范：

Benchmarking LLMs and SLMs for Patient-Reported Outcomes

中文翻译：

用于患者报告结果的LLM和SLM基准测试 

**Authors**: Matteo Marengo, Jarod Lévy, Jean-Emmanuel Bibault  

**Link**: [PDF](https://arxiv.org/pdf/2412.16291)  

**Abstract**: LLMs have transformed the execution of numerous tasks, including those in the medical domain. Among these, summarizing patient-reported outcomes (PROs) into concise natural language reports is of particular interest to clinicians, as it enables them to focus on critical patient concerns and spend more time in meaningful discussions. While existing work with LLMs like GPT-4 has shown impressive results, real breakthroughs could arise from leveraging SLMs as they offer the advantage of being deployable locally, ensuring patient data privacy and compliance with healthcare regulations. This study benchmarks several SLMs against LLMs for summarizing patient-reported Q\&A forms in the context of radiotherapy. Using various metrics, we evaluate their precision and reliability. The findings highlight both the promise and limitations of SLMs for high-stakes medical tasks, fostering more efficient and privacy-preserving AI-driven healthcare solutions. 

**Abstract (ZH)**: 以下是对给定内容的翻译，以符合学术规范：

大规模语言模型（LLMs）已彻底改变了众多任务的执行，包括医学领域的任务。在这之中，将患者报告的结果（PROs）总结为简洁自然语言报告尤为受到临床医生的关注，因为它使医生能够专注于患者的紧迫问题，并投入更多时间进行有意义的讨论。尽管使用像GPT-4这样的LLMs已有令人印象深刻的成果，但通过利用软件定义的语言模型（SLMs），真正的突破有望出现，因为SLMs的优势在于可以本地部署，从而确保患者数据的隐私并符合医疗保健法规。本研究基于放射治疗背景下的患者报告问答表（Q&A forms）总结任务，对比评估了多个SLMs与LLMs的性能。通过使用多种指标，我们评估了它们的精确性和可靠性。研究结果既指出了SLMs在高风险医疗任务中的潜力，也突显了其局限性，从而促进更高效且隐私保护的AI驱动医疗解决方案的发展。 

---
# Inference Scaling vs Reasoning: An Empirical Analysis of Compute-Optimal LLM Problem-Solving 

**Title (ZH)**: 推理扩展 vs 推理：关于计算最优的大语言模型问题求解的实证分析 

**Authors**: Marwan AbdElhameed, Pavly Halim  

**Link**: [PDF](https://arxiv.org/pdf/2412.16260)  

**Abstract**: Recent advances in large language models (LLMs) have predominantly focused on maximizing accuracy and reasoning capabilities, often overlooking crucial computational efficiency considerations. While this approach has yielded impressive accuracy improvements, it has led to methods that may be impractical for real-world deployment due to computational overhead and latency constraints. This paper investigates the potential synergy between reasoning enhancement and computational efficiency by analyzing the integration of two contrasting approaches: Quiet-STaR (Self-Taught Reasoner) and REBASE (REward BAlanced SEarch). Through comprehensive empirical analysis using the Mistral-7B model on the GSM8K dataset, we demonstrate that while each method excels in its primary objective-Quiet-STaR achieving superior accuracy (32.03%) despite high computational cost (554.66s runtime, 12.73T FLOPs), and REBASE providing exceptional efficiency (8.47s runtime, 2.35T FLOPs) while maintaining baseline-comparable accuracy (10.94%)-their integration reveals fundamental challenges in reconciling reasoning depth with computational efficiency. The combined approach unexpectedly results in degraded performance (9.38% accuracy, 143.66s runtime), highlighting critical insights about the complex interplay between reasoning enhancement and efficiency optimization in LLMs. Our findings illuminate the need for novel architectures and algorithms specifically designed to bridge the gap between these competing objectives, while providing concrete directions for future research in compute-efficient reasoning methods. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的进展主要集中于最大化准确性和推理能力，往往忽略了计算效率方面的关键考虑。虽然这种方法在准确性提升方面取得了令人印象深刻的成果，但这也导致了一些由于计算开销和延迟限制而可能在实际部署中不可行的方法。本文通过分析两种截然不同的方法——Quiet-STaR（Self-Taught Reasoner）和REBASE（REward BAlanced SEarch）——的集成，探讨了推理增强与计算效率之间的潜在协同效应。我们利用Mistral-7B模型和GSM8K数据集进行了全面的实证分析，结果显示，每种方法在主要目标方面表现优异：Quiet-STaR 尽管计算成本很高（运行时间554.66秒，12.73TFLOPs），却实现了优于其他方法的准确率（32.03%）；而REBASE 在保持基线水平相近准确率（10.94%）的前提下，提供了出色的效率（运行时间8.47秒，2.35TFLOPs）。然而，它们的集成揭示了在平衡推理深度与计算效率方面根本性的挑战。结合这两种方法出乎意料地导致了性能下降（准确率9.38%，运行时间143.66秒），突显了推理增强与效率优化之间复杂相互作用的关键见解。我们的研究结果强调了需要新型架构和算法来弥合这些竞争目标之间的差距，并为未来计算高效的推理方法研究提供了具体的方向。 

---
# Adversarial Robustness through Dynamic Ensemble Learning 

**Title (ZH)**: 通过动态集成学习提高对抗鲁棒性 

**Authors**: Hetvi Waghela, Jaydip Sen, Sneha Rakshit  

**Link**: [PDF](https://arxiv.org/pdf/2412.16254)  

**Abstract**: Adversarial attacks pose a significant threat to the reliability of pre-trained language models (PLMs) such as GPT, BERT, RoBERTa, and T5. This paper presents Adversarial Robustness through Dynamic Ensemble Learning (ARDEL), a novel scheme designed to enhance the robustness of PLMs against such attacks. ARDEL leverages the diversity of multiple PLMs and dynamically adjusts the ensemble configuration based on input characteristics and detected adversarial patterns. Key components of ARDEL include a meta-model for dynamic weighting, an adversarial pattern detection module, and adversarial training with regularization techniques. Comprehensive evaluations using standardized datasets and various adversarial attack scenarios demonstrate that ARDEL significantly improves robustness compared to existing methods. By dynamically reconfiguring the ensemble to prioritize the most robust models for each input, ARDEL effectively reduces attack success rates and maintains higher accuracy under adversarial conditions. This work contributes to the broader goal of developing more secure and trustworthy AI systems for real-world NLP applications, offering a practical and scalable solution to enhance adversarial resilience in PLMs. 

**Abstract (ZH)**: 对抗攻击对大型语言模型（PLMs）如GPT、BERT、RoBERTa和T5的可靠性构成了显著威胁。本文提出了通过动态集成学习增强鲁棒性的对抗鲁棒性方案（ARDEL），旨在提高PLMs在面对此类攻击时的鲁棒性。ARDEL利用了多个PLMs的多样性，并根据输入特征和检测到的对抗模式动态调整集成配置。ARDEL的关键组件包括动态加权的元模型、对抗模式检测模块以及结合正则化技术的对抗训练。使用标准化数据集和各种对抗攻击场景的全面评估表明，ARDEL在鲁棒性方面显著优于现有方法。通过动态重新配置集成，优先选择针对每个输入最鲁棒的模型，ARDEL有效地降低了攻击成功率，并在对抗条件下保持了更高的准确性。本文为开发更安全、更可信赖的AI系统以应用于实际的NLP场景做出了贡献，提供了一种实用且可扩展的解决方案，以增强PLMs的对抗鲁棒性。 

---
# GraphLoRA: Empowering LLMs Fine-Tuning via Graph Collaboration of MoE 

**Title (ZH)**: GraphLoRA：通过MoE的图协作增强大模型微调 

**Authors**: Ting Bai, Yue Yu, Le Huang, Zenan Xu, Zhe Zhao, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2412.16216)  

**Abstract**: Low-Rank Adaptation (LoRA) is a parameter-efficient fine-tuning method that has been widely adopted in various downstream applications of LLMs. Together with the Mixture-of-Expert (MoE) technique, fine-tuning approaches have shown remarkable improvements in model capability. However, the coordination of multiple experts in existing studies solely relies on the weights assigned by the simple router function. Lack of communication and collaboration among experts exacerbate the instability of LLMs due to the imbalance load problem of MoE. To address this issue, we propose a novel MoE graph-based LLM fine-tuning framework GraphLoRA, in which a graph router function is designed to capture the collaboration signals among experts by graph neural networks (GNNs). GraphLoRA enables all experts to understand input knowledge and share information from neighbor experts by aggregating operations. Besides, to enhance each expert's capability and their collaborations, we design two novel coordination strategies: the Poisson distribution-based distinction strategy and the Normal distribution-based load balance strategy. Extensive experiments on four real-world datasets demonstrate the effectiveness of our GraphLoRA in parameter-efficient fine-tuning of LLMs, showing the benefits of facilitating collaborations of multiple experts in the graph router of GraphLoRA. 

**Abstract (ZH)**: 低秩调整（Low-Rank Adaptation, LoRA）是一种参数高效的微调方法，在各种大规模语言模型（LLMs）的下游应用中得到了广泛应用。结合专家集合（Mixture-of-Experts, MoE）技术，微调方法已显示出显著提高模型能力的效果。然而，现有研究中所采用的多个专家之间的协调仅依赖于简单的路由器函数分配的权重，导致专家之间缺乏沟通与协作。这加剧了由于专家负载不平衡而导致的语言模型（LLMs）的不稳定性。为解决这一问题，我们提出了一种新的基于图的LLMs微调框架GraphLoRA，其中设计了一个图路由器函数，通过图神经网络（GNNs）捕获专家之间的协作信号。GraphLoRA允许所有专家理解输入知识，并通过聚合操作与邻近专家共享信息。此外，为提高每个专家的能力及其协作性，我们设计了两种新的协调策略：基于泊松分布的区别策略和基于正态分布的负载平衡策略。在四个真实世界的数据集上的广泛实验表明，GraphLoRA在参数高效的LLMs微调中表现出显著效果，证明了在GraphLoRA的图路由器中促进多个专家的协作所带来的优势。 

---
# Is Your World Simulator a Good Story Presenter? A Consecutive Events-Based Benchmark for Future Long Video Generation 

**Title (ZH)**: 你的世界模拟器能成为一个优秀的叙事呈现者吗？一种基于连续事件的未来长视频生成基准 

**Authors**: Yiping Wang, Xuehai He, Kuan Wang, Luyao Ma, Jianwei Yang, Shuohang Wang, Simon Shaolei Du, Yelong Shen  

**Link**: [PDF](https://arxiv.org/pdf/2412.16211)  

**Abstract**: The current state-of-the-art video generative models can produce commercial-grade videos with highly realistic details. However, they still struggle to coherently present multiple sequential events in the stories specified by the prompts, which is foreseeable an essential capability for future long video generation scenarios. For example, top T2V generative models still fail to generate a video of the short simple story 'how to put an elephant into a refrigerator.' While existing detail-oriented benchmarks primarily focus on fine-grained metrics like aesthetic quality and spatial-temporal consistency, they fall short of evaluating models' abilities to handle event-level story presentation. To address this gap, we introduce StoryEval, a story-oriented benchmark specifically designed to assess text-to-video (T2V) models' story-completion capabilities. StoryEval features 423 prompts spanning 7 classes, each representing short stories composed of 2-4 consecutive events. We employ advanced vision-language models, such as GPT-4V and LLaVA-OV-Chat-72B, to verify the completion of each event in the generated videos, applying a unanimous voting method to enhance reliability. Our methods ensure high alignment with human evaluations, and the evaluation of 11 models reveals its challenge, with none exceeding an average story-completion rate of 50%. StoryEval provides a new benchmark for advancing T2V models and highlights the challenges and opportunities in developing next-generation solutions for coherent story-driven video generation. 

**Abstract (ZH)**: 当前最先进的视频生成模型能够生成具有高度现实细节的商业级视频。然而，它们仍然难以在由提示指定的故事中连贯地呈现多个序列事件，这是未来长视频生成场景中一个必然需要的重要能力。例如，目前领先的文本到视频（T2V）生成模型仍然无法生成关于“如何将大象放进冰箱”的简短故事视频。现有的细节导向基准主要集中在美学质量和时空一致性等细粒度指标上，但这些指标未能评估模型在事件级别上叙述故事的能力。为填补这一空白，我们提出了StoryEval，这是一个以故事为导向的基准，专门用于评估T2V模型的故事完成能力。StoryEval 包含423个提示，涵盖7个类别，每个类别的故事由2至4个连续事件组成。我们采用高级的图文模型，如GPT-4V和LLaVA-OV-Chat-72B，来验证生成视频中每个事件的完成情况，并采用一致投票法以增加可靠性。我们的方法确保与人工评估的高度一致性，对11个模型的评估显示了其挑战性，所有模型的平均故事完成率均未超过50%。StoryEval 为推进T2V模型提供了一个新的基准，并突显了在开发下一代连贯故事驱动视频生成解决方案中面临的挑战和机遇。 

---
# HashEvict: A Pre-Attention KV Cache Eviction Strategy using Locality-Sensitive Hashing 

**Title (ZH)**: HashEvict：一种基于局部敏感哈希的预注意力键值缓存淘汰策略 

**Authors**: Minghui Liu, Tahseen Rabbani, Tony O'Halloran, Ananth Sankaralingam, Mary-Anne Hartley, Brian Gravelle, Furong Huang, Cornelia Fermüller, Yiannis Aloimonos  

**Link**: [PDF](https://arxiv.org/pdf/2412.16187)  

**Abstract**: Transformer-based large language models (LLMs) use the key-value (KV) cache to significantly accelerate inference by storing the key and value embeddings of past tokens. However, this cache consumes significant GPU memory. In this work, we introduce LSH-E, an algorithm that uses locality-sensitive hashing (LSH) to compress the KV cache. LSH-E quickly locates tokens in the cache that are cosine dissimilar to the current query token. This is achieved by computing the Hamming distance between binarized Gaussian projections of the current token query and cached token keys, with a projection length much smaller than the embedding dimension. We maintain a lightweight binary structure in GPU memory to facilitate these calculations. Unlike existing compression strategies that compute attention to determine token retention, LSH-E makes these decisions pre-attention, thereby reducing computational costs. Additionally, LSH-E is dynamic - at every decoding step, the key and value of the current token replace the embeddings of a token expected to produce the lowest attention score. We demonstrate that LSH-E can compress the KV cache by 30%-70% while maintaining high performance across reasoning, multiple-choice, long-context retrieval and summarization tasks. 

**Abstract (ZH)**: 基于Transformer的大语言模型（LLMs）通过存储过去令牌的键值（KV）嵌入来显著加速推理过程，从而利用KV缓存加速推理。然而，这种方法会消耗大量的GPU内存。本次工作中，我们引入了一种名为LSH-E的算法，该算法利用局部敏感哈希（LSH）对KV缓存进行压缩。LSH-E通过计算当前查询令牌和缓存中令牌键的二进制高斯投影之间的汉明距离来快速定位与当前查询令牌余弦差异较大的令牌。这一过程通过对当前令牌查询和缓存键进行二值化高斯投影，使用显著短于嵌入维度的投影长度来完成。我们维护了一种轻量级的二进制结构在GPU内存中，以支持这些计算。与现有的通过计算注意力来决定保留哪些令牌的压缩策略不同，LSH-E在注意机制前就做出了这些决策，从而降低了计算成本。此外，LSH-E是动态的，在每次解码步骤中，当前令牌的键值将取代预期能产生最低注意力分数的令牌的嵌入。实验结果表明，LSH-E在保留高性能的前提下，能够将KV缓存压缩30%至70%，适用于推理、多项选择、长上下文检索和摘要等多种任务。 

---
# Decoding Poultry Vocalizations -- Natural Language Processing and Transformer Models for Semantic and Emotional Analysis 

**Title (ZH)**: 解码家禽 vocalizations ——基于自然语言处理和变换器模型的语义与情感分析 

**Authors**: Venkatraman Manikandan, Suresh Neethirajan  

**Link**: [PDF](https://arxiv.org/pdf/2412.16182)  

**Abstract**: Deciphering the acoustic language of chickens offers new opportunities in animal welfare and ecological informatics. Their subtle vocal signals encode health conditions, emotional states, and dynamic interactions within ecosystems. Understanding the semantics of these calls provides a valuable tool for interpreting their functional vocabulary and clarifying how each sound serves a specific purpose in social and environmental contexts. We apply advanced Natural Language Processing and transformer based models to translate bioacoustic data into meaningful insights. Our method integrates Wave2Vec 2.0 for raw audio feature extraction with a fine tuned Bidirectional Encoder Representations from Transformers model, pretrained on a broad corpus of animal sounds and adapted to poultry tasks. This pipeline decodes poultry vocalizations into interpretable categories including distress calls, feeding signals, and mating vocalizations, revealing emotional nuances often overlooked by conventional analyses. Achieving 92 percent accuracy in classifying key vocalization types, our approach demonstrates the feasibility of real time automated monitoring of flock health and stress. By tracking this functional vocabulary, farmers can respond proactively to environmental or behavioral changes, improving poultry welfare, reducing stress related productivity losses, and supporting more sustainable farm management. Beyond agriculture, this research enhances our understanding of computational ecology. Accessing the semantic foundation of animal calls may indicate biodiversity, environmental stressors, and species interactions, informing integrative ecosystem level decision making. 

**Abstract (ZH)**: 解析鸡的声学语言为动物福利和生态信息学提供了新的机遇。它们微妙的鸣叫信号编码了健康状况、情绪状态以及生态系统内的动态互动。理解这些叫声的含义提供了一种理解它们功能词汇并阐明每种声音在社会和环境背景中特定作用的宝贵工具。我们应用先进的自然语言处理技术和基于变压器的模型将生物声学数据转化为有意义的见解。该方法结合了Wave2Vec 2.0进行原始音频特征提取，并使用在广泛动物声音语料库上微调的双向编码器表示从变压器模型，适应禽类任务。此流水线将禽类鸣叫解码为可解释的类别，包括求救叫声、喂食信号和求偶鸣叫，揭示了常规分析中经常被忽略的情绪细微差别。在分类关键鸣叫类型时，我们的方法实现了92%的准确率，证明了实时自动监测鸡群健康和应激的可行性。通过跟踪这一功能词汇，农场主可以及时响应环境或行为变化，提高禽类福利、减少与应激相关的生产力损失，并支持更可持续的农场管理。除了农业，这项研究还增强了我们对计算生态学的理解。访问动物叫声的语义基础可能揭示生物多样性、环境压力因素以及物种间的相互作用，从而提供综合生态系统级别的决策依据。 

---
# Efficient VoIP Communications through LLM-based Real-Time Speech Reconstruction and Call Prioritization for Emergency Services 

**Title (ZH)**: 基于LLM的实时语音重建和紧急服务呼叫优先级优化的高效VoIP通信 

**Authors**: Danush Venkateshperumal, Rahman Abdul Rafi, Shakil Ahmed, Ashfaq Khokhar  

**Link**: [PDF](https://arxiv.org/pdf/2412.16176)  

**Abstract**: Emergency communication systems face disruptions due to packet loss, bandwidth constraints, poor signal quality, delays, and jitter in VoIP systems, leading to degraded real-time service quality. Victims in distress often struggle to convey critical information due to panic, speech disorders, and background noise, further complicating dispatchers' ability to assess situations accurately. Staffing shortages in emergency centers exacerbate delays in coordination and assistance. This paper proposes leveraging Large Language Models (LLMs) to address these challenges by reconstructing incomplete speech, filling contextual gaps, and prioritizing calls based on severity. The system integrates real-time transcription with Retrieval-Augmented Generation (RAG) to generate contextual responses, using Twilio and AssemblyAI APIs for seamless implementation. Evaluation shows high precision, favorable BLEU and ROUGE scores, and alignment with real-world needs, demonstrating the model's potential to optimize emergency response workflows and prioritize critical cases effectively. 

**Abstract (ZH)**: 应急通信系统因包丢失、带宽限制、信号质量差、延迟和VoIP系统中的抖动而面临中断，导致实时服务质量下降。遇险人员由于恐慌、言语障碍和背景噪音往往难以传达关键信息，进一步增加了调度员准确评估情况的难度。紧急中心的人员短缺加剧了协调和援助的延迟。本文提出利用大型语言模型（LLMs）来解决这些挑战，通过重建不完整语音、填补上下文空白和基于严重程度优先处理呼叫。该系统将实时转录与检索增强生成（RAG）结合，以生成上下文响应，并使用Twilio和AssemblyAI API实现无缝集成。评价结果显示高精度、有利的BLEU和ROUGE分数，并满足实际需求，证明该模型具有优化应急响应工作流程和有效处理关键案例的潜力。 

---
# LABIIUM: AI-Enhanced Zero-configuration Measurement Automation System 

**Title (ZH)**: LABIIUM：增强型零配置测量自动化系统 

**Authors**: Emmanuel A. Olowe, Danial Chitnis  

**Link**: [PDF](https://arxiv.org/pdf/2412.16172)  

**Abstract**: The complexity of laboratory environments requires solutions that simplify instrument interaction and enhance measurement automation. Traditional tools often require configuration, software, and programming skills, creating barriers to productivity. Previous approaches, including dedicated software suites and custom scripts, frequently fall short in providing user-friendly solutions that align with programming practices. We present LABIIUM, an AI-enhanced, zero-configuration measurement automation system designed to streamline experimental workflows and improve user productivity. LABIIUM integrates an AI assistant powered by Large Language Models (LLMs) to generate code. LABIIUM's Lab-Automation-Measurement Bridges (LAMBs) enable seamless instrument connectivity using standard tools such as VSCode and Python, eliminating setup overhead. To demonstrate its capabilities, we conducted experiments involving the measurement of the parametric transfer curve of a simple two-transistor inverting amplifier with a current source load. The AI assistant was evaluated using different prompt scenarios and compared with multiple models, including Claude Sonnet 3.5, Gemini Pro 1.5, and GPT-4o. An expert solution implementing the Gradient-Weighted Adaptive Stochastic Sampling (GWASS) method was used as a baseline. The solutions generated by the AI assistant were compared with the expert solution and a uniform linear sweep baseline with 10,000 points. The graph results show that the LLMs were able to successfully complete the most basic uniform sweep, but LLMs were unable to develop adaptive sweeping algorithms to compete with GWASS. The evaluation underscores LABIIUM's ability to enhance laboratory productivity and support digital transformation in research and industry, and emphasizes the future work required to improve LLM performance in Electronic Measurement Science Tasks. 

**Abstract (ZH)**: 实验室环境的复杂性要求简化仪器交互和增强测量自动化的解决方案。传统工具往往需要配置、软件和编程技能，从而成为生产力的障碍。以前的方法，包括专用软件套件和自定义脚本，通常无法提供与编程实践相一致的用户友好的解决方案。我们提出了LABIIUM，这是一个增效的人工智能增强型、零配置测量自动化系统，旨在简化实验工作流程并提高用户生产力。LABIIUM集成了由大型语言模型（LLMs）驱动的人工智能助手，用于生成代码。LABIIUM的Lab-Automation-Measurement Bridges（LAMBs）通过使用VSCode和Python等标准工具实现无缝仪器连接，消除了配置冗余。为了展示其能力，我们进行了测量简单两级反相放大器（带电流源负载）的参数传输曲线的实验。人工智能助手使用不同的提示场景进行了评估，并与其他多个模型，包括Claude Sonnet 3.5、Gemini Pro 1.5和GPT-4o进行了比较。一个采用梯度加权自适应随机采样（GWASS）方法的专家解决方案用作基准。人工智能助手生成的解决方案与专家解决方案和均匀线性扫描基线（包含10,000个数据点）进行了比较。结果显示，大型语言模型能够成功完成最基本的均匀扫描，但在开发与GWASS竞争的自适应扫描算法方面显得不足。该评估突显了LABIIUM能够提高实验室生产力并在科研和行业中支持数字化转型的能力，并强调了未来工作以提高大型语言模型在电子测量科学任务中的表现所需的发展方向。 

---
