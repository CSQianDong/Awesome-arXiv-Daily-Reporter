# CUE-RAG: Towards Accurate and Cost-Efficient Graph-Based RAG via Multi-Partite Graph and Query-Driven Iterative Retrieval 

**Authors**: Yaodong Su, Yixiang Fang, Yingli Zhou, Quanqing Xu, Chuanhui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.08445)  

**Abstract**: Despite the remarkable progress of Large Language Models (LLMs), their performance in question answering (QA) remains limited by the lack of domain-specific and up-to-date knowledge. Retrieval-Augmented Generation (RAG) addresses this limitation by incorporating external information, often from graph-structured data. However, existing graph-based RAG methods suffer from poor graph quality due to incomplete extraction and insufficient utilization of query information during retrieval. To overcome these limitations, we propose CUE-RAG, a novel approach that introduces (1) a multi-partite graph index incorporates text Chunks, knowledge Units, and Entities to capture semantic content at multiple levels of granularity, (2) a hybrid extraction strategy that reduces LLM token usage while still producing accurate and disambiguated knowledge units, and (3) Q-Iter, a query-driven iterative retrieval strategy that enhances relevance through semantic search and constrained graph traversal. Experiments on three QA benchmarks show that CUE-RAG significantly outperforms state-of-the-art baselines, achieving up to 99.33% higher Accuracy and 113.51% higher F1 score while reducing indexing costs by 72.58%. Remarkably, CUE-RAG matches or outperforms baselines even without using an LLM for indexing. These results demonstrate the effectiveness and cost-efficiency of CUE-RAG in advancing graph-based RAG systems. 

---
# Using Large Language Models for Legal Decision-Making in Austrian Value-Added Tax Law: An Experimental Study 

**Authors**: Marina Luketina, Andrea Benkel, Christoph G. Schuetz  

**Link**: [PDF](https://arxiv.org/pdf/2507.08468)  

**Abstract**: This paper provides an experimental evaluation of the capability of large language models (LLMs) to assist in legal decision-making within the framework of Austrian and European Union value-added tax (VAT) law. In tax consulting practice, clients often describe cases in natural language, making LLMs a prime candidate for supporting automated decision-making and reducing the workload of tax professionals. Given the requirement for legally grounded and well-justified analyses, the propensity of LLMs to hallucinate presents a considerable challenge. The experiments focus on two common methods for enhancing LLM performance: fine-tuning and retrieval-augmented generation (RAG). In this study, these methods are applied on both textbook cases and real-world cases from a tax consulting firm to systematically determine the best configurations of LLM-based systems and assess the legal-reasoning capabilities of LLMs. The findings highlight the potential of using LLMs to support tax consultants by automating routine tasks and providing initial analyses, although current prototypes are not ready for full automation due to the sensitivity of the legal domain. The findings indicate that LLMs, when properly configured, can effectively support tax professionals in VAT tasks and provide legally grounded justifications for decisions. However, limitations remain regarding the handling of implicit client knowledge and context-specific documentation, underscoring the need for future integration of structured background information. 

---
