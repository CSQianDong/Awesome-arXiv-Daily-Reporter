# Multilingual Retrieval-Augmented Generation for Knowledge-Intensive Task 

**Authors**: Leonardo Ranaldi, Barry Haddow, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2504.03616)  

**Abstract**: Retrieval-augmented generation (RAG) has become a cornerstone of contemporary NLP, enhancing large language models (LLMs) by allowing them to access richer factual contexts through in-context retrieval. While effective in monolingual settings, especially in English, its use in multilingual tasks remains unexplored. This paper investigates the effectiveness of RAG across multiple languages by proposing novel approaches for multilingual open-domain question-answering. We evaluate the performance of various multilingual RAG strategies, including question-translation (tRAG), which translates questions into English before retrieval, and Multilingual RAG (MultiRAG), where retrieval occurs directly across multiple languages. Our findings reveal that tRAG, while useful, suffers from limited coverage. In contrast, MultiRAG improves efficiency by enabling multilingual retrieval but introduces inconsistencies due to cross-lingual variations in the retrieved content. To address these issues, we propose Crosslingual RAG (CrossRAG), a method that translates retrieved documents into a common language (e.g., English) before generating the response. Our experiments show that CrossRAG significantly enhances performance on knowledge-intensive tasks, benefiting both high-resource and low-resource languages. 

---
# HyperRAG: Enhancing Quality-Efficiency Tradeoffs in Retrieval-Augmented Generation with Reranker KV-Cache Reuse 

**Authors**: Yuwei An, Yihua Cheng, Seo Jin Park, Junchen Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.02921)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for enhancing the performance of large language models (LLMs) by integrating external knowledge into the generation process. A key component of RAG pipelines is the reranker, which selects the most relevant documents from a pool of retrieved candidates and significantly improves the quality of the generated responses. While rerankers refine the selection of retrieved documents in RAG pipelines, they introduce computational challenges that hinder high throughput and low latency. To address this problem, we propose HyperRAG, a system that optimizes the trade-off between quality and efficiency in RAG pipelines by leveraging KV-cache reuse for efficient reranker inference. By reusing document-side KV-cache, HyperRAG achieves both high-quality generation and system-level efficiency. To fully realize the benefits of KV-cache reuse, HyperRAG incorporates a range of system-level optimizations designed to enhance efficiency and scalability. Experiments show that HyperRAG achieves a 2 - 3 throughput improvement with decoder-only rerankers while also delivering higher downstream performance compared with traditional RAG service. 

---
# OnRL-RAG: Real-Time Personalized Mental Health Dialogue System 

**Authors**: Ahsan Bilal, Beiyu Lin, Mehdi Zaeifi  

**Link**: [PDF](https://arxiv.org/pdf/2504.02894)  

**Abstract**: Large language models (LLMs) have been widely used for various tasks and applications. However, LLMs and fine-tuning are limited to the pre-trained data. For example, ChatGPT's world knowledge until 2021 can be outdated or inaccurate. To enhance the capabilities of LLMs, Retrieval-Augmented Generation (RAG), is proposed to augment LLMs with additional, new, latest details and information to LLMs. While RAG offers the correct information, it may not best present it, especially to different population groups with personalizations. Reinforcement Learning from Human Feedback (RLHF) adapts to user needs by aligning model responses with human preference through feedback loops. In real-life applications, such as mental health problems, a dynamic and feedback-based model would continuously adapt to new information and offer personalized assistance due to complex factors fluctuating in a daily environment. Thus, we propose an Online Reinforcement Learning-based Retrieval-Augmented Generation (OnRL-RAG) system to detect and personalize the responding systems to mental health problems, such as stress, anxiety, and depression. We use an open-source dataset collected from 2028 College Students with 28 survey questions for each student to demonstrate the performance of our proposed system with the existing systems. Our system achieves superior performance compared to standard RAG and simple LLM via GPT-4o, GPT-4o-mini, Gemini-1.5, and GPT-3.5. This work would open up the possibilities of real-life applications of LLMs for personalized services in the everyday environment. The results will also help researchers in the fields of sociology, psychology, and neuroscience to align their theories more closely with the actual human daily environment. 

---
# AI Hiring with LLMs: A Context-Aware and Explainable Multi-Agent Framework for Resume Screening 

**Authors**: Frank P.-W. Lo, Jianing Qiu, Zeyu Wang, Haibao Yu, Yeming Chen, Gao Zhang, Benny Lo  

**Link**: [PDF](https://arxiv.org/pdf/2504.02870)  

**Abstract**: Resume screening is a critical yet time-intensive process in talent acquisition, requiring recruiters to analyze vast volume of job applications while remaining objective, accurate, and fair. With the advancements in Large Language Models (LLMs), their reasoning capabilities and extensive knowledge bases demonstrate new opportunities to streamline and automate recruitment workflows. In this work, we propose a multi-agent framework for resume screening using LLMs to systematically process and evaluate resumes. The framework consists of four core agents, including a resume extractor, an evaluator, a summarizer, and a score formatter. To enhance the contextual relevance of candidate assessments, we integrate Retrieval-Augmented Generation (RAG) within the resume evaluator, allowing incorporation of external knowledge sources, such as industry-specific expertise, professional certifications, university rankings, and company-specific hiring criteria. This dynamic adaptation enables personalized recruitment, bridging the gap between AI automation and talent acquisition. We assess the effectiveness of our approach by comparing AI-generated scores with ratings provided by HR professionals on a dataset of anonymized online resumes. The findings highlight the potential of multi-agent RAG-LLM systems in automating resume screening, enabling more efficient and scalable hiring workflows. 

---
# Talk2X -- An Open-Source Toolkit Facilitating Deployment of LLM-Powered Chatbots on the Web 

**Authors**: Lars Krupp, Daniel Geißler, Peter Hevesi, Marco Hirsch, Paul Lukowicz, Jakob Karolus  

**Link**: [PDF](https://arxiv.org/pdf/2504.03343)  

**Abstract**: Integrated into websites, LLM-powered chatbots offer alternative means of navigation and information retrieval, leading to a shift in how users access information on the web. Yet, predominantly closed-sourced solutions limit proliferation among web hosts and suffer from a lack of transparency with regard to implementation details and energy efficiency. In this work, we propose our openly available agent Talk2X leveraging an adapted retrieval-augmented generation approach (RAG) combined with an automatically generated vector database, benefiting energy efficiency. Talk2X's architecture is generalizable to arbitrary websites offering developers a ready to use tool for integration. Using a mixed-methods approach, we evaluated Talk2X's usability by tasking users to acquire specific assets from an open science repository. Talk2X significantly improved task completion time, correctness, and user experience supporting users in quickly pinpointing specific information as compared to standard user-website interaction. Our findings contribute technical advancements to an ongoing paradigm shift of how we access information on the web. 

---
