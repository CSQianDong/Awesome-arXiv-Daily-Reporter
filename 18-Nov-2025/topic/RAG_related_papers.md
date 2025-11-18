# TAdaRAG: Task Adaptive Retrieval-Augmented Generation via On-the-Fly Knowledge Graph Construction 

**Authors**: Jie Zhang, Bo Tang, Wanzi Shao, Wenqiang Wei, Jihao Zhao, Jianqing Zhu, Zhiyu li, Wen Xi, Zehao Lin, Feiyu Xiong, Yanchao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2511.12520)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves large language models by retrieving external knowledge, often truncated into smaller chunks due to the input context window, which leads to information loss, resulting in response hallucinations and broken reasoning chains. Moreover, traditional RAG retrieves unstructured knowledge, introducing irrelevant details that hinder accurate reasoning. To address these issues, we propose TAdaRAG, a novel RAG framework for on-the-fly task-adaptive knowledge graph construction from external sources. Specifically, we design an intent-driven routing mechanism to a domain-specific extraction template, followed by supervised fine-tuning and a reinforcement learning-based implicit extraction mechanism, ensuring concise, coherent, and non-redundant knowledge integration. Evaluations on six public benchmarks and a real-world business benchmark (NowNewsQA) across three backbone models demonstrate that TAdaRAG outperforms existing methods across diverse domains and long-text tasks, highlighting its strong generalization and practical effectiveness. 

---
# MME-RAG: Multi-Manager-Expert Retrieval-Augmented Generation for Fine-Grained Entity Recognition in Task-Oriented Dialogues 

**Authors**: Liang Xue, Haoyu Liu, Yajun Tian, Xinyu Zhong, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.12213)  

**Abstract**: Fine-grained entity recognition is crucial for reasoning and decision-making in task-oriented dialogues, yet current large language models (LLMs) continue to face challenges in domain adaptation and retrieval controllability. We introduce MME-RAG, a Multi-Manager-Expert Retrieval-Augmented Generation framework that decomposes entity recognition into two coordinated stages: type-level judgment by lightweight managers and span-level extraction by specialized experts. Each expert is supported by a KeyInfo retriever that injects semantically aligned, few-shot exemplars during inference, enabling precise and domain-adaptive extraction without additional training. Experiments on CrossNER, MIT-Movie, MIT-Restaurant, and our newly constructed multi-domain customer-service dataset demonstrate that MME-RAG performs better than recent baselines in most domains. Ablation studies further show that both the hierarchical decomposition and KeyInfo-guided retrieval are key drivers of robustness and cross-domain generalization, establishing MME-RAG as a scalable and interpretable solution for adaptive dialogue understanding. 

---
# Automated Construction of Medical Indicator Knowledge Graphs Using Retrieval Augmented Large Language Models 

**Authors**: Zhengda Wang, Daqian Shi, Jingyi Zhao, Xiaolei Diao, Xiongfeng Tang, Yanguo Qin  

**Link**: [PDF](https://arxiv.org/pdf/2511.13526)  

**Abstract**: Artificial intelligence (AI) is reshaping modern healthcare by advancing disease diagnosis, treatment decision-making, and biomedical research. Among AI technologies, large language models (LLMs) have become especially impactful, enabling deep knowledge extraction and semantic reasoning from complex medical texts. However, effective clinical decision support requires knowledge in structured, interoperable formats. Knowledge graphs serve this role by integrating heterogeneous medical information into semantically consistent networks. Yet, current clinical knowledge graphs still depend heavily on manual curation and rule-based extraction, which is limited by the complexity and contextual ambiguity of medical guidelines and literature. To overcome these challenges, we propose an automated framework that combines retrieval-augmented generation (RAG) with LLMs to construct medical indicator knowledge graphs. The framework incorporates guideline-driven data acquisition, ontology-based schema design, and expert-in-the-loop validation to ensure scalability, accuracy, and clinical reliability. The resulting knowledge graphs can be integrated into intelligent diagnosis and question-answering systems, accelerating the development of AI-driven healthcare solutions. 

---
# Grounded by Experience: Generative Healthcare Prediction Augmented with Hierarchical Agentic Retrieval 

**Authors**: Chuang Zhao, Hui Tang, Hongke Zhao, Xiaofang Zhou, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.13293)  

**Abstract**: Accurate healthcare prediction is critical for improving patient outcomes and reducing operational costs. Bolstered by growing reasoning capabilities, large language models (LLMs) offer a promising path to enhance healthcare predictions by drawing on their rich parametric knowledge. However, LLMs are prone to factual inaccuracies due to limitations in the reliability and coverage of their embedded knowledge. While retrieval-augmented generation (RAG) frameworks, such as GraphRAG and its variants, have been proposed to mitigate these issues by incorporating external knowledge, they face two key challenges in the healthcare scenario: (1) identifying the clinical necessity to activate the retrieval mechanism, and (2) achieving synergy between the retriever and the generator to craft contextually appropriate retrievals. To address these challenges, we propose GHAR, a \underline{g}enerative \underline{h}ierarchical \underline{a}gentic \underline{R}AG framework that simultaneously resolves when to retrieve and how to optimize the collaboration between submodules in healthcare. Specifically, for the first challenge, we design a dual-agent architecture comprising Agent-Top and Agent-Low. Agent-Top acts as the primary physician, iteratively deciding whether to rely on parametric knowledge or to initiate retrieval, while Agent-Low acts as the consulting service, summarising all task-relevant knowledge once retrieval was triggered. To tackle the second challenge, we innovatively unify the optimization of both agents within a formal Markov Decision Process, designing diverse rewards to align their shared goal of accurate prediction while preserving their distinct roles. Extensive experiments on three benchmark datasets across three popular tasks demonstrate our superiority over state-of-the-art baselines, highlighting the potential of hierarchical agentic RAG in advancing healthcare systems. 

---
# A Multimodal Manufacturing Safety Chatbot: Knowledge Base Design, Benchmark Development, and Evaluation of Multiple RAG Approaches 

**Authors**: Ryan Singh, Austin Hamilton, Amanda White, Michael Wise, Ibrahim Yousif, Arthur Carvalho, Zhe Shan, Reza Abrisham Baf, Mohammad Mayyas, Lora A. Cavuoto, Fadel M. Megahed  

**Link**: [PDF](https://arxiv.org/pdf/2511.11847)  

**Abstract**: Ensuring worker safety remains a critical challenge in modern manufacturing environments. Industry 5.0 reorients the prevailing manufacturing paradigm toward more human-centric operations. Using a design science research methodology, we identify three essential requirements for next-generation safety training systems: high accuracy, low latency, and low cost. We introduce a multimodal chatbot powered by large language models that meets these design requirements. The chatbot uses retrieval-augmented generation to ground its responses in curated regulatory and technical documentation. To evaluate our solution, we developed a domain-specific benchmark of expert-validated question and answer pairs for three representative machines: a Bridgeport manual mill, a Haas TL-1 CNC lathe, and a Universal Robots UR5e collaborative robot. We tested 24 RAG configurations using a full-factorial design and assessed them with automated evaluations of correctness, latency, and cost. Our top 2 configurations were then evaluated by ten industry experts and academic researchers. Our results show that retrieval strategy and model configuration have a significant impact on performance. The top configuration (selected for chatbot deployment) achieved an accuracy of 86.66%, an average latency of 10.04 seconds, and an average cost of $0.005 per query. Overall, our work provides three contributions: an open-source, domain-grounded safety training chatbot; a validated benchmark for evaluating AI-assisted safety instruction; and a systematic methodology for designing and assessing AI-enabled instructional and immersive safety training systems for Industry 5.0 environments. 

---
# Cog-RAG: Cognitive-Inspired Dual-Hypergraph with Theme Alignment Retrieval-Augmented Generation 

**Authors**: Hao Hu, Yifan Feng, Ruoxue Li, Rundong Xue, Xingliang Hou, Zhiqiang Tian, Yue Gao, Shaoyi Du  

**Link**: [PDF](https://arxiv.org/pdf/2511.13201)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the response quality and domain-specific performance of large language models (LLMs) by incorporating external knowledge to combat hallucinations. In recent research, graph structures have been integrated into RAG to enhance the capture of semantic relations between entities. However, it primarily focuses on low-order pairwise entity relations, limiting the high-order associations among multiple entities. Hypergraph-enhanced approaches address this limitation by modeling multi-entity interactions via hyperedges, but they are typically constrained to inter-chunk entity-level representations, overlooking the global thematic organization and alignment across chunks. Drawing inspiration from the top-down cognitive process of human reasoning, we propose a theme-aligned dual-hypergraph RAG framework (Cog-RAG) that uses a theme hypergraph to capture inter-chunk thematic structure and an entity hypergraph to model high-order semantic relations. Furthermore, we design a cognitive-inspired two-stage retrieval strategy that first activates query-relevant thematic content from the theme hypergraph, and then guides fine-grained recall and diffusion in the entity hypergraph, achieving semantic alignment and consistent generation from global themes to local details. Our extensive experiments demonstrate that Cog-RAG significantly outperforms existing state-of-the-art baseline approaches. 

---
# PolicyBot - Reliable Question Answering over Policy Documents 

**Authors**: Gautam Nagarajan, Omir Kumar, Sudarsun Santhiappan  

**Link**: [PDF](https://arxiv.org/pdf/2511.13489)  

**Abstract**: All citizens of a country are affected by the laws and policies introduced by their government. These laws and policies serve essential functions for citizens. Such as granting them certain rights or imposing specific obligations. However, these documents are often lengthy, complex, and difficult to navigate, making it challenging for citizens to locate and understand relevant information. This work presents PolicyBot, a retrieval-augmented generation (RAG) system designed to answer user queries over policy documents with a focus on transparency and reproducibility. The system combines domain-specific semantic chunking, multilingual dense embeddings, multi-stage retrieval with reranking, and source-aware generation to provide responses grounded in the original documents. We implemented citation tracing to reduce hallucinations and improve user trust, and evaluated alternative retrieval and generation configurations to identify effective design choices. The end-to-end pipeline is built entirely with open-source tools, enabling easy adaptation to other domains requiring document-grounded question answering. This work highlights design considerations, practical challenges, and lessons learned in deploying trustworthy RAG systems for governance-related contexts. 

---
