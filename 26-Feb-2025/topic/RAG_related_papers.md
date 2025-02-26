# LevelRAG: Enhancing Retrieval-Augmented Generation with Multi-hop Logic Planning over Rewriting Augmented Searchers 

**Title (ZH)**: LevelRAG：通过重写增强的检索增强生成中多跳逻辑规划的应用 

**Authors**: Zhuocheng Zhang, Yang Feng, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.18139)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a crucial method for mitigating hallucinations in Large Language Models (LLMs) and integrating external knowledge into their responses. Existing RAG methods typically employ query rewriting to clarify the user intent and manage multi-hop logic, while using hybrid retrieval to expand search scope. However, the tight coupling of query rewriting to the dense retriever limits its compatibility with hybrid retrieval, impeding further RAG performance improvements. To address this challenge, we introduce a high-level searcher that decomposes complex queries into atomic queries, independent of any retriever-specific optimizations. Additionally, to harness the strengths of sparse retrievers for precise keyword retrieval, we have developed a new sparse searcher that employs Lucene syntax to enhance retrieval this http URL web and dense searchers, these components seamlessly collaborate within our proposed method, \textbf{LevelRAG}. In LevelRAG, the high-level searcher orchestrates the retrieval logic, while the low-level searchers (sparse, web, and dense) refine the queries for optimal retrieval. This approach enhances both the completeness and accuracy of the retrieval process, overcoming challenges associated with current query rewriting techniques in hybrid retrieval scenarios. Empirical experiments conducted on five datasets, encompassing both single-hop and multi-hop question answering tasks, demonstrate the superior performance of LevelRAG compared to existing RAG methods. Notably, LevelRAG outperforms the state-of-the-art proprietary model, GPT4o, underscoring its effectiveness and potential impact on the RAG field. 

**Abstract (ZH)**: 检索增强生成（RAG）是减轻大型语言模型（LLMs）幻觉现象和将外部知识整合到其回应中的关键方法。现有RAG方法通常通过查询重新写入来澄清用户意图并管理多跳逻辑，同时使用混合检索来扩大搜索范围。然而，查询重新写入与密集检索器的紧密耦合限制了其与混合检索的兼容性，阻碍了RAG性能的进一步提升。为了解决这一挑战，我们引入了一个高级搜索器，将复杂查询分解为原子查询，不依赖于任何特定检索器的优化。此外，为了利用稀疏检索器的优势进行精准关键词检索，我们开发了一种新的稀疏搜索器，利用Lucene语法增强检索。这些组件在我们提出的方法——**LevelRAG** 中无缝协作。在LevelRAG中，高级搜索器统筹检索逻辑，而低级搜索器（稀疏、网页和密集）则对查询进行优化以实现最佳检索。这种方法提高了检索过程的完整性和准确性，克服了当前混合检索场景中查询重新写入技术面临的挑战。我们在五个数据集上进行了实证实验，涵盖了单跳和多跳问答任务，结果表明LevelRAG在与现有RAG方法的比较中表现出更优的性能。特别地，LevelRAG在性能上超越了当前最先进的商业化模型GPT4o，这进一步证明了其有效性和对RAG领域的潜在影响。 

---
# RankCoT: Refining Knowledge for Retrieval-Augmented Generation through Ranking Chain-of-Thoughts 

**Title (ZH)**: RankCoT：通过排序链思考为检索增强生成精炼知识 

**Authors**: Mingyan Wu, Zhenghao Liu, Yukun Yan, Xinze Li, Shi Yu, Zheni Zeng, Yu Gu, Ge Yu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17888)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the performance of Large Language Models (LLMs) by incorporating external knowledge. However, LLMs still encounter challenges in effectively utilizing the knowledge from retrieved documents, often being misled by irrelevant or noisy information. To address this issue, we introduce RankCoT, a knowledge refinement method that incorporates reranking signals in generating CoT-based summarization for knowledge refinement based on given query and all retrieval documents. During training, RankCoT prompts the LLM to generate Chain-of-Thought (CoT) candidates based on the query and individual documents. It then fine-tunes the LLM to directly reproduce the best CoT from these candidate outputs based on all retrieved documents, which requires LLM to filter out irrelevant documents during generating CoT-style summarization. Additionally, RankCoT incorporates a self-reflection mechanism that further refines the CoT outputs, resulting in higher-quality training data. Our experiments demonstrate the effectiveness of RankCoT, showing its superior performance over other knowledge refinement models. Further analysis reveals that RankCoT can provide shorter but effective refinement results, enabling the generator to produce more accurate answers. All code and data are available at this https URL. 

**Abstract (ZH)**: 检索增强生成（RAG）通过引入外部知识来提高大型语言模型（LLMs）的性能。然而，LLMs 在有效地利用检索文档中的知识时仍面临挑战，常常受到无关或噪声信息的误导。为了解决这一问题，我们引入了 RankCoT，这是一种知识精炼方法，基于给定查询和所有检索文档，在生成基于CoT的总结时结合使用再排序信号进行知识精炼。

在训练过程中，RankCoT 促使LLM 基于查询和每个文档生成CoT候选答案。随后，它对LLM 进行微调，使其直接根据所有检索文档再现最佳CoT，这要求LLM 在生成CoT风格的总结时过滤掉不必要的文档信息。此外，RankCoT 还引入了一种自我反思机制，进一步精炼CoT 输出，从而产生更高质量的训练数据。我们的实验证明了 RankCoT 的有效性，并显示其性能优于其他知识精炼模型。进一步的分析表明，RankCoT 可以提供更短但更有效的精炼结果，使生成器能够生成更准确的答案。所有代码和数据均可在以下链接获取：这个 https URL。 

---
# Say Less, Mean More: Leveraging Pragmatics in Retrieval-Augmented Generation 

**Title (ZH)**: 少说多意：利用启发式生成中的语用学优势 

**Authors**: Haris Riaz, Ellen Riloff, Mihai Surdeanu  

**Link**: [PDF](https://arxiv.org/pdf/2502.17839)  

**Abstract**: We propose a simple, unsupervised method that injects pragmatic principles in retrieval-augmented generation (RAG) frameworks such as Dense Passage Retrieval~\cite{karpukhin2020densepassageretrievalopendomain} to enhance the utility of retrieved contexts. Our approach first identifies which sentences in a pool of documents retrieved by RAG are most relevant to the question at hand, cover all the topics addressed in the input question and no more, and then highlights these sentences within their context, before they are provided to the LLM, without truncating or altering the context in any other way. We show that this simple idea brings consistent improvements in experiments on three question answering tasks (ARC-Challenge, PubHealth and PopQA) using five different LLMs. It notably enhances relative accuracy by up to 19.7\% on PubHealth and 10\% on ARC-Challenge compared to a conventional RAG system. 

**Abstract (ZH)**: 我们提出了一种简单且无监督的方法，在检索增强生成（RAG）框架（如密集段落检索~\cite{karpukhin2020densepassageretrievalopendomain}）中注入实用原则，以提高检索到的上下文的实用性。我们的方法首先在由RAG检索出的文档池中识别出与当前问题最相关的句子，这些句子涵盖了输入问题中涉及的所有主题但不包含更多内容，然后在将这些句子呈现给大语言模型（LLM）之前，高亮显示这些句子的上下文，而不会以任何方式截断或修改上下文。实验结果显示，这一简单思想在使用五种不同大语言模型对三个问答任务（ARC-Challenge、PubHealth 和 PopQA）进行的测试中带来了一致的改进。在PubHealth任务上相对准确度提高了高达19.7%，在ARC-Challenge任务上提高了10%。与传统RAG系统相比，这一方法显著提升了相对准确度。 

---
# ViDoRAG: Visual Document Retrieval-Augmented Generation via Dynamic Iterative Reasoning Agents 

**Title (ZH)**: ViDoRAG：基于动态迭代推理代理的视觉文档检索增强生成方法 

**Authors**: Qiuchen Wang, Ruixue Ding, Zehui Chen, Weiqi Wu, Shihang Wang, Pengjun Xie, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.18017)  

**Abstract**: Understanding information from visually rich documents remains a significant challenge for traditional Retrieval-Augmented Generation (RAG) methods. Existing benchmarks predominantly focus on image-based question answering (QA), overlooking the fundamental challenges of efficient retrieval, comprehension, and reasoning within dense visual documents. To bridge this gap, we introduce ViDoSeek, a novel dataset designed to evaluate RAG performance on visually rich documents requiring complex reasoning. Based on it, we identify key limitations in current RAG approaches: (i) purely visual retrieval methods struggle to effectively integrate both textual and visual features, and (ii) previous approaches often allocate insufficient reasoning tokens, limiting their effectiveness. To address these challenges, we propose ViDoRAG, a novel multi-agent RAG framework tailored for complex reasoning across visual documents. ViDoRAG employs a Gaussian Mixture Model (GMM)-based hybrid strategy to effectively handle multi-modal retrieval. To further elicit the model's reasoning capabilities, we introduce an iterative agent workflow incorporating exploration, summarization, and reflection, providing a framework for investigating test-time scaling in RAG domains. Extensive experiments on ViDoSeek validate the effectiveness and generalization of our approach. Notably, ViDoRAG outperforms existing methods by over 10% on the competitive ViDoSeek benchmark. 

**Abstract (ZH)**: 传统的检索增强生成（RAG）方法在理解视觉丰富的文档信息方面仍面临显著挑战。现有的基准主要侧重于基于图像的问题回答（QA），忽视了密集视觉文档中高效检索、理解和推理的基本挑战。为弥合这一差距，我们引入了ViDoSeek，这是一个新数据集，旨在评估RAG在需要复杂推理的视觉丰富文档中的性能。基于此，我们识别了当前RAG方法的关键局限性：(i) 纯视觉检索方法难以有效地整合文本和视觉特征，(ii) 以前的方法往往分配不足的推理令牌，限制了它们的效果。为了应对这些挑战，我们提出了ViDoRAG，这是一个专门为视觉文档中复杂推理设计的多agent RAG框架。ViDoRAG采用基于高斯混合模型（GMM）的混合策略有效地处理多模态检索。为了进一步激发模型的推理能力，我们引入了一种迭代的agent工作流，包括探索、总结和反思，为RAG领域中的测试时尺度研究提供了框架。在ViDoSeek上的广泛实验验证了我们方法的有效性和泛化能力。值得注意的是，ViDoRAG在竞争性的ViDoSeek基准测试中性能优于现有方法超过10%。 

---
# MM-PoisonRAG: Disrupting Multimodal RAG with Local and Global Poisoning Attacks 

**Title (ZH)**: MM-PoisonRAG: 针对多模态RAG的局部和全局投毒攻击破解方法 

**Authors**: Hyeonjeong Ha, Qiusi Zhan, Jeonghwan Kim, Dimitrios Bralios, Saikrishna Sanniboina, Nanyun Peng, Kai-wei Chang, Daniel Kang, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2502.17832)  

**Abstract**: Multimodal large language models (MLLMs) equipped with Retrieval Augmented Generation (RAG) leverage both their rich parametric knowledge and the dynamic, external knowledge to excel in tasks such as Question Answering. While RAG enhances MLLMs by grounding responses in query-relevant external knowledge, this reliance poses a critical yet underexplored safety risk: knowledge poisoning attacks, where misinformation or irrelevant knowledge is intentionally injected into external knowledge bases to manipulate model outputs to be incorrect and even harmful. To expose such vulnerabilities in multimodal RAG, we propose MM-PoisonRAG, a novel knowledge poisoning attack framework with two attack strategies: Localized Poisoning Attack (LPA), which injects query-specific misinformation in both text and images for targeted manipulation, and Globalized Poisoning Attack (GPA) to provide false guidance during MLLM generation to elicit nonsensical responses across all queries. We evaluate our attacks across multiple tasks, models, and access settings, demonstrating that LPA successfully manipulates the MLLM to generate attacker-controlled answers, with a success rate of up to 56% on MultiModalQA. Moreover, GPA completely disrupts model generation to 0% accuracy with just a single irrelevant knowledge injection. Our results highlight the urgent need for robust defenses against knowledge poisoning to safeguard multimodal RAG frameworks. 

**Abstract (ZH)**: 配备了检索增强生成（RAG）的多模态大型语言模型（MLLMs）结合了其丰富的参数知识和动态的外部知识，使其在问答等任务中表现出色。虽然RAG通过将响应与查询相关的外部知识相链接来增强MLLMs，但这种依赖性带来了一个关键且尚未充分探索的安全风险：知识中毒攻击，其中故意向外部知识库注入错误信息或无关知识，以操控模型输出错误甚至有害的结果。为了揭示多模态RAG中的此类漏洞，我们提出了一种新颖的知识中毒攻击框架——MM-PoisonRAG，并提出两种攻击策略：局部中毒攻击（LPA），它在文本和图像中注入查询相关的错误信息以进行定向操纵；以及全球化中毒攻击（GPA），在MLLM生成过程中提供虚假指导，导致所有查询均产生成分荒谬的响应。我们在多个任务、模型和访问设置下评估了我们的攻击，结果表明，LPA能够成功操控MLLM生成攻击者控制的答案，在MultiModalQA上的成功率最高可达56%。此外，GPA仅通过一次无关知识的注入即可完全破坏模型生成，使其准确率为零。我们的研究结果强调了针对知识中毒攻击建立 robust 防御措施的迫切需求，以保障多模态RAG框架的安全。 

---
# RAG-Enhanced Collaborative LLM Agents for Drug Discovery 

**Title (ZH)**: 增强记忆辅助的协作大语言模型代理在药物发现中的应用 

**Authors**: Namkyeong Lee, Edward De Brouwer, Ehsan Hajiramezanali, Chanyoung Park, Gabriele Scalia  

**Link**: [PDF](https://arxiv.org/pdf/2502.17506)  

**Abstract**: Recent advances in large language models (LLMs) have shown great potential to accelerate drug discovery. However, the specialized nature of biochemical data often necessitates costly domain-specific fine-tuning, posing critical challenges. First, it hinders the application of more flexible general-purpose LLMs in cutting-edge drug discovery tasks. More importantly, it impedes the rapid integration of the vast amounts of scientific data continuously generated through experiments and research. To investigate these challenges, we propose CLADD, a retrieval-augmented generation (RAG)-empowered agentic system tailored to drug discovery tasks. Through the collaboration of multiple LLM agents, CLADD dynamically retrieves information from biomedical knowledge bases, contextualizes query molecules, and integrates relevant evidence to generate responses -- all without the need for domain-specific fine-tuning. Crucially, we tackle key obstacles in applying RAG workflows to biochemical data, including data heterogeneity, ambiguity, and multi-source integration. We demonstrate the flexibility and effectiveness of this framework across a variety of drug discovery tasks, showing that it outperforms general-purpose and domain-specific LLMs as well as traditional deep learning approaches. 

**Abstract (ZH)**: 近年来，大型语言模型（LLMs）的发展展现出加速药物发现的强大潜力。然而，生物化学数据的专业性往往需要进行昂贵的领域特定微调，这带来了关键性的挑战。首先，这阻碍了更灵活的通用LLM在前沿药物发现任务中的应用。更重要的是，这妨碍了迅速整合通过实验和研究不断生成的大量科学数据。为了应对这些挑战，我们提出了一种CLADD系统，这是一种由检索增强生成（RAG）赋能的特制于药物发现任务的代理系统。通过多个LLM代理的协作，CLADD动态地从生物医学知识库中检索信息，上下文化查询分子，并整合相关证据生成回应——所有这些都不需要进行领域特定的微调。至关重要的是，我们解决了将RAG工作流应用于生物化学数据的关键障碍，包括数据异质性、歧义性和多源整合。我们展示了该框架在多种药物发现任务中的灵活性和有效性，证明其性能优于通用和领域特定的LLM以及传统的深度学习方法。 

---
