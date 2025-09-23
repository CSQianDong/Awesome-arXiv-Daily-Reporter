# A Multimodal Conversational Assistant for the Characterization of Agricultural Plots from Geospatial Open Data 

**Authors**: Juan Cañada, Raúl Alonso, Julio Molleda, Fidel Díez  

**Link**: [PDF](https://arxiv.org/pdf/2509.17544)  

**Abstract**: The increasing availability of open Earth Observation (EO) and agricultural datasets holds great potential for supporting sustainable land management. However, their high technical entry barrier limits accessibility for non-expert users. This study presents an open-source conversational assistant that integrates multimodal retrieval and large language models (LLMs) to enable natural language interaction with heterogeneous agricultural and geospatial data. The proposed architecture combines orthophotos, Sentinel-2 vegetation indices, and user-provided documents through retrieval-augmented generation (RAG), allowing the system to flexibly determine whether to rely on multimodal evidence, textual knowledge, or both in formulating an answer. To assess response quality, we adopt an LLM-as-a-judge methodology using Qwen3-32B in a zero-shot, unsupervised setting, applying direct scoring in a multi-dimensional quantitative evaluation framework. Preliminary results show that the system is capable of generating clear, relevant, and context-aware responses to agricultural queries, while remaining reproducible and scalable across geographic regions. The primary contributions of this work include an architecture for fusing multimodal EO and textual knowledge sources, a demonstration of lowering the barrier to access specialized agricultural information through natural language interaction, and an open and reproducible design. 

---
# A Knowledge Graph-based Retrieval-Augmented Generation Framework for Algorithm Selection in the Facility Layout Problem 

**Authors**: Nikhil N S, Amol Dilip Joshi, Bilal Muhammed, Soban Babu  

**Link**: [PDF](https://arxiv.org/pdf/2509.18054)  

**Abstract**: Selecting a solution algorithm for the Facility Layout Problem (FLP), an NP-hard optimization problem with a multiobjective trade-off, is a complex task that requires deep expert knowledge. The performance of a given algorithm depends on specific problem characteristics such as its scale, objectives, and constraints. This creates a need for a data-driven recommendation method to guide algorithm selection in automated design systems. This paper introduces a new recommendation method to make such expertise accessible, based on a Knowledge Graph-based Retrieval-Augmented Generation (KG RAG) framework. To address this, a domain-specific knowledge graph is constructed from published literature. The method then employs a multi-faceted retrieval mechanism to gather relevant evidence from this knowledge graph using three distinct approaches, which include a precise graph-based search, flexible vector-based search, and high-level cluster-based search. The retrieved evidence is utilized by a Large Language Model (LLM) to generate algorithm recommendations with data-driven reasoning. The proposed KG-RAG method is compared against a commercial LLM chatbot with access to the knowledge base as a table, across a series of diverse, real-world FLP test cases. Based on recommendation accuracy and reasoning capability, the proposed method performed significantly better than the commercial LLM chatbot. 

---
# One Agent to Serve All: a Lite-Adaptive Stylized AI Assistant for Millions of Multi-Style Official Accounts 

**Authors**: Xingyu Fan, Feifei Li, Wenhui Que, Hailong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.17788)  

**Abstract**: Conversational agents deployed in industrial-scale official account platforms must generate responses that are both contextually grounded and stylistically aligned-requirements that existing methods struggle to meet. Chain-of-thought (CoT) prompting induces significant latency due to multi-turn reasoning; per-account fine-tuning is computationally prohibitive; and long prompt-based methods degrade the model's ability to grasp injected context and style. In this paper, we propose WeStar, a lite-adaptive framework for stylized contextual question answering that scales to millions of official accounts. WeStar combines context-grounded generation via RAG with style-aware generation using Parametric RAG (PRAG), where LoRA modules are dynamically activated per style cluster. Our contributions are fourfold: (1) We introduce WeStar, a unified framework capable of serving large volumes of official accounts with minimal overhead. (2) We propose a multi-dimensional, cluster-based parameter sharing scheme that enables compact style representation while preserving stylistic diversity. (3) We develop a style-enhanced Direct Preference Optimization (SeDPO) method to optimize each style cluster's parameters for improved generation quality. (4) Experiments on a large-scale industrial dataset validate the effectiveness and efficiency of WeStar, underscoring its pracitical value in real-world deployment. 

---
# Turk-LettuceDetect: A Hallucination Detection Models for Turkish RAG Applications 

**Authors**: Selva Taş, Mahmut El Huseyni, Özay Ezerceli, Reyhan Bayraktar, Fatma Betül Terzioğlu  

**Link**: [PDF](https://arxiv.org/pdf/2509.17671)  

**Abstract**: The widespread adoption of Large Language Models (LLMs) has been hindered by their tendency to hallucinate, generating plausible but factually incorrect information. While Retrieval-Augmented Generation (RAG) systems attempt to address this issue by grounding responses in external knowledge, hallucination remains a persistent challenge, particularly for morphologically complex, low-resource languages like Turkish. This paper introduces Turk-LettuceDetect, the first suite of hallucination detection models specifically designed for Turkish RAG applications. Building on the LettuceDetect framework, we formulate hallucination detection as a token-level classification task and fine-tune three distinct encoder architectures: a Turkish-specific ModernBERT, TurkEmbed4STS, and multilingual EuroBERT. These models were trained on a machine-translated version of the RAGTruth benchmark dataset containing 17,790 instances across question answering, data-to-text generation, and summarization tasks. Our experimental results show that the ModernBERT-based model achieves an F1-score of 0.7266 on the complete test set, with particularly strong performance on structured tasks. The models maintain computational efficiency while supporting long contexts up to 8,192 tokens, making them suitable for real-time deployment. Comparative analysis reveals that while state-of-the-art LLMs demonstrate high recall, they suffer from low precision due to over-generation of hallucinated content, underscoring the necessity of specialized detection mechanisms. By releasing our models and translated dataset, this work addresses a critical gap in multilingual NLP and establishes a foundation for developing more reliable and trustworthy AI applications for Turkish and other languages. 

---
# Comparing RAG and GraphRAG for Page-Level Retrieval Question Answering on Math Textbook 

**Authors**: Eason Chen, Chuangji Li, Shizhuo Li, Conrad Borchers, Zimo Xiao, Chloe Qianhui Zhao, Jionghao Lin, Kenneth R. Koedinger  

**Link**: [PDF](https://arxiv.org/pdf/2509.16780)  

**Abstract**: Technology-enhanced learning environments often help students retrieve relevant learning content for questions arising during self-paced study. Large language models (LLMs) have emerged as novel aids for information retrieval during learning. While LLMs are effective for general-purpose question-answering, they typically lack alignment with the domain knowledge of specific course materials such as textbooks and slides. We investigate Retrieval-Augmented Generation (RAG) and GraphRAG, a knowledge graph-enhanced RAG approach, for page-level question answering in an undergraduate mathematics textbook. While RAG has been effective for retrieving discrete, contextually relevant passages, GraphRAG may excel in modeling interconnected concepts and hierarchical knowledge structures. We curate a dataset of 477 question-answer pairs, each tied to a distinct textbook page. We then compare the standard embedding-based RAG methods to GraphRAG for evaluating both retrieval accuracy-whether the correct page is retrieved-and generated answer quality via F1 scores. Our findings show that embedding-based RAG achieves higher retrieval accuracy and better F1 scores compared to GraphRAG, which tends to retrieve excessive and sometimes irrelevant content due to its entity-based structure. We also explored re-ranking the retrieved pages with LLM and observed mixed results, including performance drop and hallucinations when dealing with larger context windows. Overall, this study highlights both the promises and challenges of page-level retrieval systems in educational contexts, emphasizing the need for more refined retrieval methods to build reliable AI tutoring solutions in providing reference page numbers. 

---
# Enhancing Financial RAG with Agentic AI and Multi-HyDE: A Novel Approach to Knowledge Retrieval and Hallucination Reduction 

**Authors**: Akshay Govind Srinivasan, Ryan Jacob George, Jayden Koshy Joe, Hrushikesh Kant, Harshith M R, Sachin Sundar, Sudharshan Suresh, Rahul Vimalkanth, Vijayavallabh  

**Link**: [PDF](https://arxiv.org/pdf/2509.16369)  

**Abstract**: Accurate and reliable knowledge retrieval is vital for financial question-answering, where continually updated data sources and complex, high-stakes contexts demand precision. Traditional retrieval systems rely on a single database and retriever, but financial applications require more sophisticated approaches to handle intricate regulatory filings, market analyses, and extensive multi-year reports. We introduce a framework for financial Retrieval Augmented Generation (RAG) that leverages agentic AI and the Multi-HyDE system, an approach that generates multiple, nonequivalent queries to boost the effectiveness and coverage of retrieval from large, structured financial corpora. Our pipeline is optimized for token efficiency and multi-step financial reasoning, and we demonstrate that their combination improves accuracy by 11.2% and reduces hallucinations by 15%. Our method is evaluated on standard financial QA benchmarks, showing that integrating domain-specific retrieval mechanisms such as Multi-HyDE with robust toolsets, including keyword and table-based retrieval, significantly enhances both the accuracy and reliability of answers. This research not only delivers a modular, adaptable retrieval framework for finance but also highlights the importance of structured agent workflows and multi-perspective retrieval for trustworthy deployment of AI in high-stakes financial applications. 

---
# FinDebate: Multi-Agent Collaborative Intelligence for Financial Analysis 

**Authors**: Tianshi Cai, Guanxu Li, Nijia Han, Ce Huang, Zimu Wang, Changyu Zeng, Yuqi Wang, Jingshi Zhou, Haiyang Zhang, Qi Chen, Yushan Pan, Shuihua Wang, Wei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.17395)  

**Abstract**: We introduce FinDebate, a multi-agent framework for financial analysis, integrating collaborative debate with domain-specific Retrieval-Augmented Generation (RAG). Five specialized agents, covering earnings, market, sentiment, valuation, and risk, run in parallel to synthesize evidence into multi-dimensional insights. To mitigate overconfidence and improve reliability, we introduce a safe debate protocol that enables agents to challenge and refine initial conclusions while preserving coherent recommendations. Experimental results, based on both LLM-based and human evaluations, demonstrate the framework's efficacy in producing high-quality analysis with calibrated confidence levels and actionable investment strategies across multiple time horizons. 

---
