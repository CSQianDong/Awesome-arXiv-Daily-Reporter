# VideoRAG: Retrieval-Augmented Generation over Video Corpus 

**Title (ZH)**: VideoRAG：基于视频语料的检索增强生成 

**Authors**: Soyeong Jeong, Kangsan Kim, Jinheon Baek, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2501.05874)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a powerful strategy to address the issue of generating factually incorrect outputs in foundation models by retrieving external knowledge relevant to queries and incorporating it into their generation process. However, existing RAG approaches have primarily focused on textual information, with some recent advancements beginning to consider images, and they largely overlook videos, a rich source of multimodal knowledge capable of representing events, processes, and contextual details more effectively than any other modality. While a few recent studies explore the integration of videos in the response generation process, they either predefine query-associated videos without retrieving them according to queries, or convert videos into the textual descriptions without harnessing their multimodal richness. To tackle these, we introduce VideoRAG, a novel framework that not only dynamically retrieves relevant videos based on their relevance with queries but also utilizes both visual and textual information of videos in the output generation. Further, to operationalize this, our method revolves around the recent advance of Large Video Language Models (LVLMs), which enable the direct processing of video content to represent it for retrieval and seamless integration of the retrieved videos jointly with queries. We experimentally validate the effectiveness of VideoRAG, showcasing that it is superior to relevant baselines. 

**Abstract (ZH)**: 检索增强生成（RAG）是一种强大的策略，用于解决基础模型生成事实性错误输出的问题。通过检索与查询相关的外部知识并将其纳入生成过程，RAG能够克服这一问题。然而，现有的RAG方法主要集中在文本信息上，近年来的一些进展开始考虑图像，但它们很大程度上忽略了视频这一丰富的多模态知识来源，视频能够比任何其他模态更有效地表示事件、过程和上下文细节。虽然有一些最近的研究探讨了在响应生成过程中整合视频的方法，但它们要么预先定义与查询相关的视频而不根据查询检索它们，要么将视频转换为文本描述而不充分利用其多模态丰富性。为了解决这些问题，我们引入了VideoRAG，这是一种新型框架，不仅能够根据查询的相关性动态检索相关视频，还能够在生成输出时利用视频的视觉和文本信息。此外，为了实现这一目标，我们的方法围绕最近的大型视频语言模型（LVLMs）的进展展开，这使得可以直接处理视频内容以用于检索，并无缝地将检索到的视频与查询联合集成。通过实验验证了VideoRAG的有效性，展示了它在与基线方法相比的优越性。 

---
# Retrieval-Augmented Generation by Evidence Retroactivity in LLMs 

**Title (ZH)**: 基于证据回溯的LLM中检索增强生成 

**Authors**: Liang Xiao, Wen Dai, Shuai Chen, Bin Qin, Chongyang Shi, Haopeng Jing, Tianyu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2501.05475)  

**Abstract**: Retrieval-augmented generation has gained significant attention due to its ability to integrate relevant external knowledge, enhancing the accuracy and reliability of the LLMs' responses. Most of the existing methods apply a dynamic multiple retrieval-generating process, to address multi-hop complex questions by decomposing them into sub-problems. However, these methods rely on an unidirectional forward reasoning paradigm, where errors from insufficient reasoning steps or inherent flaws in current retrieval systems are irreversible, potentially derailing the entire reasoning chain. For the first time, this work introduces Retroactive Retrieval-Augmented Generation (RetroRAG), a novel framework to build a retroactive reasoning paradigm. RetroRAG revises and updates the evidence, redirecting the reasoning chain to the correct direction. RetroRAG constructs an evidence-collation-discovery framework to search, generate, and refine credible evidence. It synthesizes inferential evidence related to the key entities in the question from the existing source knowledge and formulates search queries to uncover additional information. As new evidence is found, RetroRAG continually updates and organizes this information, enhancing its ability to locate further necessary evidence. Paired with an Answerer to generate and evaluate outputs, RetroRAG is capable of refining its reasoning process iteratively until a reliable answer is obtained. Empirical evaluations show that RetroRAG significantly outperforms existing methods. 

**Abstract (ZH)**: 检索增强生成由于其整合相关外部知识的能力，在提高大语言模型（LLM）响应的准确性和可靠性方面获得了广泛关注。目前大多数现有方法采用动态多轮检索-生成过程，通过将多跳复杂问题分解为子问题来解决问题。然而，这些方法依赖于单向前向推理模式，其中由于推理步骤不足或当前检索系统中的内在缺陷导致的错误是不可逆的，可能会导致整个推理链的偏离。首次提出，本工作引入了回溯检索增强生成（RetroRAG），一种新颖的框架以构建回溯推理范式。RetroRAG 能够修正和更新证据，重新引导推理链的方向。RetroRAG 构建了一个证据收集发现框架，用于搜索、生成和完善可信证据。它从现有来源知识中合成与问题关键实体相关的推断性证据，并制定搜索查询以发现额外信息。随着新证据的发现，RetroRAG 不断更新和组织这些信息，增强其定位进一步所需的证据的能力。结合一个回答器生成和评估输出，RetroRAG 能够迭代地改进其推理过程，直到获得可靠的答案。实证评估表明，RetroRAG 显著优于现有方法。 

---
# LLMQuoter: Enhancing RAG Capabilities Through Efficient Quote Extraction From Large Contexts 

**Title (ZH)**: LLMQuoter：通过高效提取大规模语境中的引用来增强RAG能力 

**Authors**: Yuri Facanha Bezerra, Li Weigang  

**Link**: [PDF](https://arxiv.org/pdf/2501.05554)  

**Abstract**: We introduce LLMQuoter, a lightweight, distillation-based model designed to enhance Retrieval Augmented Generation (RAG) by extracting the most relevant textual evidence for downstream reasoning tasks. Built on the LLaMA-3B architecture and fine-tuned with Low-Rank Adaptation (LoRA) on a 15,000-sample subset of HotpotQA, LLMQuoter adopts a "quote-first-then-answer" strategy, efficiently identifying key quotes before passing curated snippets to reasoning models. This workflow reduces cognitive overhead and outperforms full-context approaches like Retrieval-Augmented Fine-Tuning (RAFT), achieving over 20-point accuracy gains across both small and large language models. By leveraging knowledge distillation from a high-performing teacher model, LLMQuoter achieves competitive results in a resource-efficient fine-tuning setup. It democratizes advanced RAG capabilities, delivering significant performance improvements without requiring extensive model retraining. Our results highlight the potential of distilled quote-based reasoning to streamline complex workflows, offering a scalable and practical solution for researchers and practitioners alike. 

**Abstract (ZH)**: 我们介绍了LLMQuoter，这是一个轻量级的蒸馏模型，旨在通过提取与下游推理由证任务最相关的文本证据来增强检索增强生成（RAG）。该模型基于LLaMA-3B架构，并在HotpotQA的一个包含15,000个样本的子集中使用低秩适应（LoRA）进行了微调。LLMQuoter采用“先引述再回答”的策略，在确定关键引述后，将精炼片段传递给推理模型。这种工作流程降低了认知负担，并在各个方面（无论是小型还是大型语言模型）都超越了全文上下文方法，如检索增强微调（RAFT），实现了超过20个百分点的准确性提升。通过利用高性能教师模型的知识蒸馏，LLMQuoter在资源高效微调设置中取得了竞争力的结果。它使高级RAG能力更加普及，能够在不进行大量模型重新训练的情况下实现显著的性能提升。我们的研究结果突显了知识蒸馏引述推理的潜力，可以通过简化复杂的工作流程来提供一种可扩展且实用的解决方案，对于研究者和实践者都具有重要意义。 

---
