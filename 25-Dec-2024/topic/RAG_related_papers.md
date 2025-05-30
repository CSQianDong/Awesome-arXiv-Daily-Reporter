# Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases 

**Title (ZH)**: 《RAG中的海盗：适应性攻击LLMs以泄露知识库》

这个标题翻译成中文时，为了使其更符合中文的表达习惯和学术规范，可以稍微调整一下：

《RAG中的适应性攻击：从LLMs泄露知识库》 

**Authors**: Christian Di Maio, Cristian Cosci, Marco Maggini, Valentina Poggioni, Stefano Melacci  

**Link**: [PDF](https://arxiv.org/pdf/2412.18295)  

**Abstract**: The growing ubiquity of Retrieval-Augmented Generation (RAG) systems in several real-world services triggers severe concerns about their security. A RAG system improves the generative capabilities of a Large Language Models (LLM) by a retrieval mechanism which operates on a private knowledge base, whose unintended exposure could lead to severe consequences, including breaches of private and sensitive information. This paper presents a black-box attack to force a RAG system to leak its private knowledge base which, differently from existing approaches, is adaptive and automatic. A relevance-based mechanism and an attacker-side open-source LLM favor the generation of effective queries to leak most of the (hidden) knowledge base. Extensive experimentation proves the quality of the proposed algorithm in different RAG pipelines and domains, comparing to very recent related approaches, which turn out to be either not fully black-box, not adaptive, or not based on open-source models. The findings from our study remark the urgent need for more robust privacy safeguards in the design and deployment of RAG systems. 

**Abstract (ZH)**: 随着检索增强生成（RAG）系统在多个实际服务中的广泛应用，其安全性问题引发了严重关注。RAG系统通过检索机制增强了大型语言模型（LLM）的生成能力，而检索机制依赖于一个私有的知识库，若该知识库意外泄露，可能导致严重的后果，包括个人和敏感信息的泄露。本文提出了一种黑盒攻击方法，迫使RAG系统泄露其私有知识库。与现有方法不同，本方法具有适应性和自动化。基于相关性机制和攻击者侧的开源LLM，能够生成有效的查询以泄露大部分（隐藏的）知识库。广泛的实验表明，在不同RAG流水线和领域中，所提出算法的质量优于非常近期的相关方法，这些方法要么不完全符合黑盒攻击条件，要么不具有适应性，要么不基于开源模型。我们的研究发现突显了在设计和部署RAG系统时加强更稳健的隐私保护措施的紧迫性。 

---
# Dynamic Multi-Agent Orchestration and Retrieval for Multi-Source Question-Answer Systems using Large Language Models 

**Title (ZH)**: 使用大型语言模型的多源问答系统中动态多agent编排与检索方法 

**Authors**: Antony Seabra, Claudio Cavalcante, Joao Nepomuceno, Lucas Lago, Nicolaas Ruberg, Sergio Lifschitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.17964)  

**Abstract**: We propose a methodology that combines several advanced techniques in Large Language Model (LLM) retrieval to support the development of robust, multi-source question-answer systems. This methodology is designed to integrate information from diverse data sources, including unstructured documents (PDFs) and structured databases, through a coordinated multi-agent orchestration and dynamic retrieval approach. Our methodology leverages specialized agents-such as SQL agents, Retrieval-Augmented Generation (RAG) agents, and router agents - that dynamically select the most appropriate retrieval strategy based on the nature of each query. To further improve accuracy and contextual relevance, we employ dynamic prompt engineering, which adapts in real time to query-specific contexts. The methodology's effectiveness is demonstrated within the domain of Contract Management, where complex queries often require seamless interaction between unstructured and structured data. Our results indicate that this approach enhances response accuracy and relevance, offering a versatile and scalable framework for developing question-answer systems that can operate across various domains and data sources. 

**Abstract (ZH)**: 我们提出了一种方法论，结合了大型语言模型（LLM）检索中的多项高级技术，以支持稳健的多源问答系统的开发。该方法论旨在通过协调多代理编排和动态检索方法，整合来自多种数据源的信息，包括未结构化的文档（如PDF）和结构化的数据库。此方法论利用了专用于SQL代理、检索增强生成（RAG）代理和路由器代理等特定任务的智能代理，它们能够根据每个查询的性质动态选择最合适的检索策略。为了进一步提高准确性和上下文相关性，我们采用了动态提示工程，它能够根据查询的具体上下文实时调整。该方法论在合同管理领域得到了验证，在该领域复杂的查询往往需要无缝地处理未结构化和结构化数据之间的交互。实验结果表明，该方法增强了响应的准确性和相关性，提供了一个在不同领域和数据源上操作的灵活和可扩展框架，用于开发问答系统。 

---
# Contrato360 2.0: A Document and Database-Driven Question-Answer System using Large Language Models and Agents 

**Title (ZH)**: Contrato360 2.0：一种基于文档和数据库的大型语言模型及智能代理驱动的问答系统 

**Authors**: Antony Seabra, Claudio Cavalcante, Joao Nepomuceno, Lucas Lago, Nicolaas Ruberg, Sergio Lifschitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.17942)  

**Abstract**: We present a question-and-answer (Q\&A) application designed to support the contract management process by leveraging combined information from contract documents (PDFs) and data retrieved from contract management systems (database). This data is processed by a large language model (LLM) to provide precise and relevant answers. The accuracy of these responses is further enhanced through the use of Retrieval-Augmented Generation (RAG), text-to-SQL techniques, and agents that dynamically orchestrate the workflow. These techniques eliminate the need to retrain the language model. Additionally, we employed Prompt Engineering to fine-tune the focus of responses. Our findings demonstrate that this multi-agent orchestration and combination of techniques significantly improve the relevance and accuracy of the answers, offering a promising direction for future information systems. 

**Abstract (ZH)**: 我们提出了一种基于问题回答（Q&A）的应用程序，旨在通过利用合同文件（PDFs）和从合同管理系统中检索的数据（数据库）的综合信息来支持合同管理流程。这些数据由大型语言模型（LLM）处理，以提供精准和相关的答案。通过使用检索增强生成（RAG）、文本到SQL技术以及能够动态调度工作流程的代理，这些答案的准确性得以进一步提高。这些技术消除了重新训练语言模型的必要性。此外，我们采用了提示工程技术来精细调整答案的焦点。我们的研究结果表明，这种多代理调度及其技术组合显著提高了答案的相关性和准确性，并为未来的信息系统提供了令人鼓舞的方向。 

---
# GeAR: Graph-enhanced Agent for Retrieval-augmented Generation 

**Title (ZH)**: GeAR：基于图的代理增强检索增强生成

这个翻译符合学术规范，同时保持了原文的意思和结构。在这里，“Graph-enhanced”被翻译为“基于图的”，“agent”翻译为“代理”，“retrieval-augmented generation”翻译为“检索增强生成”，以确保术语的专业性和准确性。 

**Authors**: Zhili Shen, Chenxin Diao, Pavlos Vougiouklis, Pascual Merita, Shriram Piramanayagam, Damien Graux, Dandan Tu, Zeren Jiang, Ruofei Lai, Yang Ren, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18431)  

**Abstract**: Retrieval-augmented generation systems rely on effective document retrieval capabilities. By design, conventional sparse or dense retrievers face challenges in multi-hop retrieval scenarios. In this paper, we present GeAR, which advances RAG performance through two key innovations: (i) graph expansion, which enhances any conventional base retriever, such as BM25, and (ii) an agent framework that incorporates graph expansion. Our evaluation demonstrates GeAR's superior retrieval performance on three multi-hop question answering datasets. Additionally, our system achieves state-of-the-art results with improvements exceeding 10% on the challenging MuSiQue dataset, while requiring fewer tokens and iterations compared to other multi-step retrieval systems. 

**Abstract (ZH)**: 检索增强生成系统依赖于有效的文档检索能力。从设计上讲，传统的稀疏或密集检索器在多跳检索场景中面临挑战。本文中，我们提出了GeAR，通过两项关键创新来提升RAG（检索增强生成）的表现：(i) 图扩展，该方法可以增强任何传统的基线检索器，例如BM25；(ii) 一个代理框架，该框架结合了图扩展。我们的评估结果显示，GeAR在三个多跳问答数据集上的检索性能优于其他方法。此外，在具有挑战性的MuSiQue数据集上，我们的系统取得了当前最佳结果，相比其他多步检索系统，所需token数量和迭代次数更少，性能提升超过10%。 

---
# Correctness is not Faithfulness in RAG Attributions 

**Title (ZH)**: 正确性不等同于忠実性在RAG归因中 

**Authors**: Jonas Wallat, Maria Heuss, Maarten de Rijke, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2412.18004)  

**Abstract**: Retrieving relevant context is a common approach to reduce hallucinations and enhance answer reliability. Explicitly citing source documents allows users to verify generated responses and increases trust. Prior work largely evaluates citation correctness - whether cited documents support the corresponding statements. But citation correctness alone is insufficient. To establish trust in attributed answers, we must examine both citation correctness and citation faithfulness. In this work, we first disentangle the notions of citation correctness and faithfulness, which have been applied inconsistently in previous studies. Faithfulness ensures that the model's reliance on cited documents is genuine, reflecting actual reference use rather than superficial alignment with prior beliefs, which we call post-rationalization. We design an experiment that reveals the prevalent issue of post-rationalization, which undermines reliable attribution and may result in misplaced trust. Our findings suggest that current attributed answers often lack citation faithfulness (up to 57 percent of the citations), highlighting the need to evaluate correctness and faithfulness for trustworthy attribution in language models. 

**Abstract (ZH)**: 检索相关上下文是减少幻觉和提高答案可靠性的一种常见方法。明确引用源文档可以让用户验证生成的答案并增加信任度。以往的工作主要评估引用的正确性——即所引用的文档是否支持相应的陈述。然而，引用的正确性本身是不够的。为了建立对归因答案的信任，我们必须同时评估引用的正确性和忠实性。在本研究中，我们首先区分了引用正确性和忠实性这两个概念，这两个概念在以往的研究中应用不一致。忠实性确保模型依赖引用的文档是真实的，反映了实际的参考使用，而不是表面化的与先验信念的一致性，这被称为后理性化。我们设计了一个实验，揭示了后理性化这一普遍存在的问题，它削弱了可靠的归因，并可能导致不适当的信任。我们的研究发现表明，当前的归因答案常常缺乏引用忠实性（高达57%的引用），突显了在语言模型中评估正确性和忠实性以实现可靠归因的必要性。 

---
