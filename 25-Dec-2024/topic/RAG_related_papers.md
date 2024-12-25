# Pirates of the RAG: Adaptively Attacking LLMs to Leak Knowledge Bases 

**Title (ZH)**: 基于检索增强的海盗：适应性攻击LLMs以泄露知识库 

**Authors**: Christian Di Maio, Cristian Cosci, Marco Maggini, Valentina Poggioni, Stefano Melacci  

**Link**: [PDF](https://arxiv.org/pdf/2412.18295)  

**Abstract**: The growing ubiquity of Retrieval-Augmented Generation (RAG) systems in several real-world services triggers severe concerns about their security. A RAG system improves the generative capabilities of a Large Language Models (LLM) by a retrieval mechanism which operates on a private knowledge base, whose unintended exposure could lead to severe consequences, including breaches of private and sensitive information. This paper presents a black-box attack to force a RAG system to leak its private knowledge base which, differently from existing approaches, is adaptive and automatic. A relevance-based mechanism and an attacker-side open-source LLM favor the generation of effective queries to leak most of the (hidden) knowledge base. Extensive experimentation proves the quality of the proposed algorithm in different RAG pipelines and domains, comparing to very recent related approaches, which turn out to be either not fully black-box, not adaptive, or not based on open-source models. The findings from our study remark the urgent need for more robust privacy safeguards in the design and deployment of RAG systems. 

**Abstract (ZH)**: 随着检索增强生成（RAG）系统在多个实际服务中的广泛应用，其安全性问题引起了严重关切。RAG系统通过基于私有知识库的检索机制提高了大型语言模型（LLM）的生成能力，而该知识库的意外暴露可能会导致严重的后果，包括隐私和敏感信息的泄露。本文提出了一种黑盒攻击方法，以迫使RAG系统泄露其私有知识库。与其他现有方法相比，该方法具有自适应性和自动化特点。基于相关性的机制和攻击方的开源大型语言模型有助于生成有效的查询，从而泄露大部分（隐藏的）知识库。广泛的实验结果证明，所提出的算法在不同RAG管道和领域中具有较高的质量，并优于非常最近的相关方法，这些方法要么不完全黑盒，要么不可适应，要么不基于开源模型。我们的研究发现强调了在设计和部署RAG系统时迫切需要更 robust 的隐私保护措施。 

---
# Dynamic Multi-Agent Orchestration and Retrieval for Multi-Source Question-Answer Systems using Large Language Models 

**Title (ZH)**: 使用大型语言模型的动态多Agent协同与检索以构建多源问答系统 

**Authors**: Antony Seabra, Claudio Cavalcante, Joao Nepomuceno, Lucas Lago, Nicolaas Ruberg, Sergio Lifschitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.17964)  

**Abstract**: We propose a methodology that combines several advanced techniques in Large Language Model (LLM) retrieval to support the development of robust, multi-source question-answer systems. This methodology is designed to integrate information from diverse data sources, including unstructured documents (PDFs) and structured databases, through a coordinated multi-agent orchestration and dynamic retrieval approach. Our methodology leverages specialized agents-such as SQL agents, Retrieval-Augmented Generation (RAG) agents, and router agents - that dynamically select the most appropriate retrieval strategy based on the nature of each query. To further improve accuracy and contextual relevance, we employ dynamic prompt engineering, which adapts in real time to query-specific contexts. The methodology's effectiveness is demonstrated within the domain of Contract Management, where complex queries often require seamless interaction between unstructured and structured data. Our results indicate that this approach enhances response accuracy and relevance, offering a versatile and scalable framework for developing question-answer systems that can operate across various domains and data sources. 

**Abstract (ZH)**: 我们提出了一种方法论，该方法论结合了大型语言模型（LLM）检索领域的多项先进技术，以支持稳健的多源问答系统的发展。该方法论旨在通过协调多智能体编排和动态检索方式，整合来自多种数据源的信息，包括未结构化的文档（PDF）和结构化数据库。该方法论利用了专门的智能体——如SQL智能体、检索增强生成（RAG）智能体和路由智能体——这些智能体能够根据每个查询的性质动态选择最合适的检索策略。为进一步提高准确性和上下文相关性，我们采用了动态提示工程，这种技术能够实时适应查询特定的上下文。该方法论的有效性在合同管理领域得到验证，该领域中复杂的查询往往需要无缝地交互未结构化和结构化数据。我们的结果表明，这种方法提高了响应的准确性和相关性，提供了一个适用于多种领域和数据源的灵活且可扩展的框架，用于开发问答系统。 

---
# Contrato360 2.0: A Document and Database-Driven Question-Answer System using Large Language Models and Agents 

**Title (ZH)**: Contrato360 2.0：一种基于文档和数据库的大语言模型及智能体驱动的问答系统 

**Authors**: Antony Seabra, Claudio Cavalcante, Joao Nepomuceno, Lucas Lago, Nicolaas Ruberg, Sergio Lifschitz  

**Link**: [PDF](https://arxiv.org/pdf/2412.17942)  

**Abstract**: We present a question-and-answer (Q\&A) application designed to support the contract management process by leveraging combined information from contract documents (PDFs) and data retrieved from contract management systems (database). This data is processed by a large language model (LLM) to provide precise and relevant answers. The accuracy of these responses is further enhanced through the use of Retrieval-Augmented Generation (RAG), text-to-SQL techniques, and agents that dynamically orchestrate the workflow. These techniques eliminate the need to retrain the language model. Additionally, we employed Prompt Engineering to fine-tune the focus of responses. Our findings demonstrate that this multi-agent orchestration and combination of techniques significantly improve the relevance and accuracy of the answers, offering a promising direction for future information systems. 

**Abstract (ZH)**: 我们提出了一种问答（Q&A）应用，该应用通过结合合同文件（PDF）和从合同管理系统（数据库）中检索的数据来支持合同管理流程。这些数据经过大规模语言模型（LLM）处理，提供精准且相关的答案。通过使用检索增强生成（RAG）、文本到SQL技术以及能够动态协调工作流的代理，这些方法进一步提高了答案的准确性。这些技术消除了重新训练语言模型的需要。此外，我们采用了提示工程（Prompt Engineering）来细化答案的焦点。我们的研究发现，这种多代理协调及技术组合显著提高了答案的相关性和准确性，为未来信息系统的发展提供了有前景的方向。 

---
# GeAR: Graph-enhanced Agent for Retrieval-augmented Generation 

**Title (ZH)**: GeAR：图增强代理用于检索增强生成 

**Authors**: Zhili Shen, Chenxin Diao, Pavlos Vougiouklis, Pascual Merita, Shriram Piramanayagam, Damien Graux, Dandan Tu, Zeren Jiang, Ruofei Lai, Yang Ren, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2412.18431)  

**Abstract**: Retrieval-augmented generation systems rely on effective document retrieval capabilities. By design, conventional sparse or dense retrievers face challenges in multi-hop retrieval scenarios. In this paper, we present GeAR, which advances RAG performance through two key innovations: (i) graph expansion, which enhances any conventional base retriever, such as BM25, and (ii) an agent framework that incorporates graph expansion. Our evaluation demonstrates GeAR's superior retrieval performance on three multi-hop question answering datasets. Additionally, our system achieves state-of-the-art results with improvements exceeding 10% on the challenging MuSiQue dataset, while requiring fewer tokens and iterations compared to other multi-step retrieval systems. 

**Abstract (ZH)**: 检索增强生成系统依赖于有效的文档检索能力。从设计上看，传统的稀疏或密集检索器在多跳检索场景中面临挑战。本文中，我们提出了一种名为GeAR的新系统，通过两项关键创新来提升RAG（Retrieval-Augmented Generation）的效果：（i）图扩展技术，可以增强任何传统的基线检索器，如BM25；（ii）包含图扩展技术的代理框架。我们的评估结果显示，GeAR在三个多跳问答数据集上表现出优越的检索性能。此外，我们的系统在具有挑战性的MuSiQue数据集上达到了最先进的性能，改进幅度超过10%，所需的token数量和迭代次数也更少，优于其他多步检索系统。 

---
# Leveraging Memory Retrieval to Enhance LLM-based Generative Recommendation 

**Title (ZH)**: 利用记忆检索增强基于大规模语言模型的生成性推荐 

**Authors**: Chengbing Wang, Yang Zhang, Fengbin Zhu, Jizhi Zhang, Tianhao Shi, Fuli Feng  

**Link**: [PDF](https://arxiv.org/pdf/2412.17593)  

**Abstract**: Leveraging Large Language Models (LLMs) to harness user-item interaction histories for item generation has emerged as a promising paradigm in generative recommendation. However, the limited context window of LLMs often restricts them to focusing on recent user interactions only, leading to the neglect of long-term interests involved in the longer histories. To address this challenge, we propose a novel Automatic Memory-Retrieval framework (AutoMR), which is capable of storing long-term interests in the memory and extracting relevant information from it for next-item generation within LLMs. Extensive experimental results on two real-world datasets demonstrate the effectiveness of our proposed AutoMR framework in utilizing long-term interests for generative recommendation. 

**Abstract (ZH)**: 利用大型语言模型（LLMs）挖掘用户-项目互动历史以生成项目，在生成型推荐领域被认为是一种有前途的范式。然而，LLMs 较小的上下文窗口限制了它们仅关注最近的用户互动，而忽视了较长历史中涉及的长期兴趣。为了解决这一挑战，我们提出了一种新的自动记忆检索框架（AutoMR），该框架能在记忆中存储长期兴趣，并从记忆中提取相关信息用于后续项目的生成。在两个实际数据集上的广泛实验结果证明了我们所提出的 AutoMR 框架在利用长期兴趣进行生成型推荐方面的有效性。 

---
