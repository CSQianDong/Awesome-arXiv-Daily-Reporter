# Optimizing Retrieval-Augmented Generation for Electrical Engineering: A Case Study on ABB Circuit Breakers 

**Authors**: Salahuddin Alawadhi, Noorhan Abbas  

**Link**: [PDF](https://arxiv.org/pdf/2505.17520)  

**Abstract**: Integrating Retrieval Augmented Generation (RAG) with Large Language Models (LLMs) has shown the potential to provide precise, contextually relevant responses in knowledge intensive domains. This study investigates the ap-plication of RAG for ABB circuit breakers, focusing on accuracy, reliability, and contextual relevance in high-stakes engineering environments. By leveraging tailored datasets, advanced embedding models, and optimized chunking strategies, the research addresses challenges in data retrieval and contextual alignment unique to engineering documentation. Key contributions include the development of a domain-specific dataset for ABB circuit breakers and the evaluation of three RAG pipelines: OpenAI GPT4o, Cohere, and Anthropic Claude. Advanced chunking methods, such as paragraph-based and title-aware segmentation, are assessed for their impact on retrieval accuracy and response generation. Results demonstrate that while certain configurations achieve high precision and relevancy, limitations persist in ensuring factual faithfulness and completeness, critical in engineering contexts. This work underscores the need for iterative improvements in RAG systems to meet the stringent demands of electrical engineering tasks, including design, troubleshooting, and operational decision-making. The findings in this paper help advance research of AI in highly technical domains such as electrical engineering. 

---
# Search Wisely: Mitigating Sub-optimal Agentic Searches By Reducing Uncertainty 

**Authors**: Peilin Wu, Mian Zhang, Xinlu Zhang, Xinya Du, Zhiyu Zoey Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.17281)  

**Abstract**: Agentic Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by enabling dynamic, multi-step reasoning and information retrieval. However, these systems often exhibit sub-optimal search behaviors like over-search (retrieving redundant information) and under-search (failing to retrieve necessary information), which hinder efficiency and reliability. This work formally defines and quantifies these behaviors, revealing their prevalence across multiple QA datasets and agentic RAG systems (e.g., one model could have avoided searching in 27.7% of its search steps). Furthermore, we demonstrate a crucial link between these inefficiencies and the models' uncertainty regarding their own knowledge boundaries, where response accuracy correlates with model's uncertainty in its search decisions. To address this, we propose $\beta$-GRPO, a reinforcement learning-based training method that incorporates confidence threshold to reward high-certainty search decisions. Experiments on seven QA benchmarks show that $\beta$-GRPO enable a 3B model with better agentic RAG ability, outperforming other strong baselines with a 4% higher average exact match score. 

---
# FB-RAG: Improving RAG with Forward and Backward Lookup 

**Authors**: Kushal Chawla, Alfy Samuel, Anoop Kumar, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.17206)  

**Abstract**: The performance of Retrieval Augmented Generation (RAG) systems relies heavily on the retriever quality and the size of the retrieved context. A large enough context ensures that the relevant information is present in the input context for the LLM, but also incorporates irrelevant content that has been shown to confuse the models. On the other hand, a smaller context reduces the irrelevant information, but it often comes at the risk of losing important information necessary to answer the input question. This duality is especially challenging to manage for complex queries that contain little information to retrieve the relevant chunks from the full context. To address this, we present a novel framework, called FB-RAG, which enhances the RAG pipeline by relying on a combination of backward lookup (overlap with the query) and forward lookup (overlap with candidate reasons and answers) to retrieve specific context chunks that are the most relevant for answering the input query. Our evaluations on 9 datasets from two leading benchmarks show that FB-RAG consistently outperforms RAG and Long Context baselines developed recently for these benchmarks. We further show that FB-RAG can improve performance while reducing latency. We perform qualitative analysis of the strengths and shortcomings of our approach, providing specific insights to guide future work. 

---
# DailyQA: A Benchmark to Evaluate Web Retrieval Augmented LLMs Based on Capturing Real-World Changes 

**Authors**: Jiehan Cheng, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2505.17162)  

**Abstract**: We propose DailyQA, an automatically updated dynamic dataset that updates questions weekly and contains answers to questions on any given date. DailyQA utilizes daily updates from Wikipedia revision logs to implement a fully automated pipeline of data filtering, query generation synthesis, quality checking, answer extraction, and query classification. The benchmark requires large language models (LLMs) to process and answer questions involving fast-changing factual data and covering multiple domains. We evaluate several open-source and closed-source LLMs using different RAG pipelines with web search augmentation. We compare the ability of different models to process time-sensitive web information and find that rerank of web retrieval results is critical. Our results indicate that LLMs still face significant challenges in handling frequently updated information, suggesting that DailyQA benchmarking provides valuable insights into the direction of progress for LLMs and RAG systems. 

---
# DO-RAG: A Domain-Specific QA Framework Using Knowledge Graph-Enhanced Retrieval-Augmented Generation 

**Authors**: David Osei Opoku, Ming Sheng, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.17058)  

**Abstract**: Domain-specific QA systems require not just generative fluency but high factual accuracy grounded in structured expert knowledge. While recent Retrieval-Augmented Generation (RAG) frameworks improve context recall, they struggle with integrating heterogeneous data and maintaining reasoning consistency. To address these challenges, we propose DO-RAG, a scalable and customizable hybrid QA framework that integrates multi-level knowledge graph construction with semantic vector retrieval. Our system employs a novel agentic chain-of-thought architecture to extract structured relationships from unstructured, multimodal documents, constructing dynamic knowledge graphs that enhance retrieval precision. At query time, DO-RAG fuses graph and vector retrieval results to generate context-aware responses, followed by hallucination mitigation via grounded refinement. Experimental evaluations in the database and electrical domains show near-perfect recall and over 94% answer relevancy, with DO-RAG outperforming baseline frameworks by up to 33.38%. By combining traceability, adaptability, and performance efficiency, DO-RAG offers a reliable foundation for multi-domain, high-precision QA at scale. 

---
# Resolving Conflicting Evidence in Automated Fact-Checking: A Study on Retrieval-Augmented LLMs 

**Authors**: Ziyu Ge, Yuhao Wu, Daniel Wai Kit Chin, Roy Ka-Wei Lee, Rui Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17762)  

**Abstract**: Large Language Models (LLMs) augmented with retrieval mechanisms have demonstrated significant potential in fact-checking tasks by integrating external knowledge. However, their reliability decreases when confronted with conflicting evidence from sources of varying credibility. This paper presents the first systematic evaluation of Retrieval-Augmented Generation (RAG) models for fact-checking in the presence of conflicting evidence. To support this study, we introduce \textbf{CONFACT} (\textbf{Con}flicting Evidence for \textbf{Fact}-Checking) (Dataset available at this https URL), a novel dataset comprising questions paired with conflicting information from various sources. Extensive experiments reveal critical vulnerabilities in state-of-the-art RAG methods, particularly in resolving conflicts stemming from differences in media source credibility. To address these challenges, we investigate strategies to integrate media background information into both the retrieval and generation stages. Our results show that effectively incorporating source credibility significantly enhances the ability of RAG models to resolve conflicting evidence and improve fact-checking performance. 

---
# FinRAGBench-V: A Benchmark for Multimodal RAG with Visual Citation in the Financial Domain 

**Authors**: Suifeng Zhao, Zhuoran Jin, Sujian Li, Jun Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.17471)  

**Abstract**: Retrieval-Augmented Generation (RAG) plays a vital role in the financial domain, powering applications such as real-time market analysis, trend forecasting, and interest rate computation. However, most existing RAG research in finance focuses predominantly on textual data, overlooking the rich visual content in financial documents, resulting in the loss of key analytical insights. To bridge this gap, we present FinRAGBench-V, a comprehensive visual RAG benchmark tailored for finance which effectively integrates multimodal data and provides visual citation to ensure traceability. It includes a bilingual retrieval corpus with 60,780 Chinese and 51,219 English pages, along with a high-quality, human-annotated question-answering (QA) dataset spanning heterogeneous data types and seven question categories. Moreover, we introduce RGenCite, an RAG baseline that seamlessly integrates visual citation with generation. Furthermore, we propose an automatic citation evaluation method to systematically assess the visual citation capabilities of Multimodal Large Language Models (MLLMs). Extensive experiments on RGenCite underscore the challenging nature of FinRAGBench-V, providing valuable insights for the development of multimodal RAG systems in finance. 

---
# Curriculum Guided Reinforcement Learning for Efficient Multi Hop Retrieval Augmented Generation 

**Authors**: Yuelyu Ji, Rui Meng, Zhuochun Li, Daqing He  

**Link**: [PDF](https://arxiv.org/pdf/2505.17391)  

**Abstract**: Retrieval-augmented generation (RAG) grounds large language models (LLMs) in up-to-date external evidence, yet existing multi-hop RAG pipelines still issue redundant subqueries, explore too shallowly, or wander through overly long search chains. We introduce EVO-RAG, a curriculum-guided reinforcement learning framework that evolves a query-rewriting agent from broad early-stage exploration to concise late-stage refinement. EVO-RAG couples a seven-factor, step-level reward vector (covering relevance, redundancy, efficiency, and answer correctness) with a time-varying scheduler that reweights these signals as the episode unfolds. The agent is trained with Direct Preference Optimization over a multi-head reward model, enabling it to learn when to search, backtrack, answer, or refuse. Across four multi-hop QA benchmarks (HotpotQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle), EVO-RAG boosts Exact Match by up to 4.6 points over strong RAG baselines while trimming average retrieval depth by 15 %. Ablation studies confirm the complementary roles of curriculum staging and dynamic reward scheduling. EVO-RAG thus offers a general recipe for building reliable, cost-effective multi-hop RAG systems. 

---
# Personalizing Student-Agent Interactions Using Log-Contextualized Retrieval Augmented Generation (RAG) 

**Authors**: Clayton Cohn, Surya Rayala, Caitlin Snyder, Joyce Fonteles, Shruti Jain, Naveeduddin Mohammed, Umesh Timalsina, Sarah K. Burriss, Ashwin T S, Namrata Srivastava, Menton Deweese, Angela Eeds, Gautam Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2505.17238)  

**Abstract**: Collaborative dialogue offers rich insights into students' learning and critical thinking. This is essential for adapting pedagogical agents to students' learning and problem-solving skills in STEM+C settings. While large language models (LLMs) facilitate dynamic pedagogical interactions, potential hallucinations can undermine confidence, trust, and instructional value. Retrieval-augmented generation (RAG) grounds LLM outputs in curated knowledge, but its effectiveness depends on clear semantic links between user input and a knowledge base, which are often weak in student dialogue. We propose log-contextualized RAG (LC-RAG), which enhances RAG retrieval by incorporating environment logs to contextualize collaborative discourse. Our findings show that LC-RAG improves retrieval over a discourse-only baseline and allows our collaborative peer agent, Copa, to deliver relevant, personalized guidance that supports students' critical thinking and epistemic decision-making in a collaborative computational modeling environment, XYZ. 

---
# PersonaBOT: Bringing Customer Personas to Life with LLMs and RAG 

**Authors**: Muhammed Rizwan, Lars Carlsson, Mohammad Loni  

**Link**: [PDF](https://arxiv.org/pdf/2505.17156)  

**Abstract**: The introduction of Large Language Models (LLMs) has significantly transformed Natural Language Processing (NLP) applications by enabling more advanced analysis of customer personas. At Volvo Construction Equipment (VCE), customer personas have traditionally been developed through qualitative methods, which are time-consuming and lack scalability. The main objective of this paper is to generate synthetic customer personas and integrate them into a Retrieval-Augmented Generation (RAG) chatbot to support decision-making in business processes. To this end, we first focus on developing a persona-based RAG chatbot integrated with verified personas. Next, synthetic personas are generated using Few-Shot and Chain-of-Thought (CoT) prompting techniques and evaluated based on completeness, relevance, and consistency using McNemar's test. In the final step, the chatbot's knowledge base is augmented with synthetic personas and additional segment information to assess improvements in response accuracy and practical utility. Key findings indicate that Few-Shot prompting outperformed CoT in generating more complete personas, while CoT demonstrated greater efficiency in terms of response time and token usage. After augmenting the knowledge base, the average accuracy rating of the chatbot increased from 5.88 to 6.42 on a 10-point scale, and 81.82% of participants found the updated system useful in business contexts. 

---
# After Retrieval, Before Generation: Enhancing the Trustworthiness of Large Language Models in RAG 

**Authors**: Xinbang Dai, Huikang Hu, Yuncheng Hua, Jiaqi Li, Yongrui Chen, Rihui Jin, Nan Hu, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.17118)  

**Abstract**: Retrieval-augmented generation (RAG) systems face critical challenges in balancing internal (parametric) and external (retrieved) knowledge, especially when these sources conflict or are unreliable. To analyze these scenarios comprehensively, we construct the Trustworthiness Response Dataset (TRD) with 36,266 questions spanning four RAG settings. We reveal that existing approaches address isolated scenarios-prioritizing one knowledge source, naively merging both, or refusing answers-but lack a unified framework to handle different real-world conditions simultaneously. Therefore, we propose the BRIDGE framework, which dynamically determines a comprehensive response strategy of large language models (LLMs). BRIDGE leverages an adaptive weighting mechanism named soft bias to guide knowledge collection, followed by a Maximum Soft-bias Decision Tree to evaluate knowledge and select optimal response strategies (trust internal/external knowledge, or refuse). Experiments show BRIDGE outperforms baselines by 5-15% in accuracy while maintaining balanced performance across all scenarios. Our work provides an effective solution for LLMs' trustworthy responses in real-world RAG applications. 

---
# VoxRAG: A Step Toward Transcription-Free RAG Systems in Spoken Question Answering 

**Authors**: Zackary Rackauckas, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.17326)  

**Abstract**: We introduce VoxRAG, a modular speech-to-speech retrieval-augmented generation system that bypasses transcription to retrieve semantically relevant audio segments directly from spoken queries. VoxRAG employs silence-aware segmentation, speaker diarization, CLAP audio embeddings, and FAISS retrieval using L2-normalized cosine similarity. We construct a 50-query test set recorded as spoken input by a native English speaker. Retrieval quality was evaluated using LLM-as-a-judge annotations. For very relevant segments, cosine similarity achieved a Recall@10 of 0.34. For somewhat relevant segments, Recall@10 rose to 0.60 and nDCG@10 to 0.27, highlighting strong topical alignment. Answer quality was judged on a 0--2 scale across relevance, accuracy, completeness, and precision, with mean scores of 0.84, 0.58, 0.56, and 0.46 respectively. While precision and retrieval quality remain key limitations, VoxRAG shows that transcription-free speech-to-speech retrieval is feasible in RAG systems. 

---
# VIBE: Vector Index Benchmark for Embeddings 

**Authors**: Elias Jääsaari, Ville Hyvönen, Matteo Ceccarello, Teemu Roos, Martin Aumüller  

**Link**: [PDF](https://arxiv.org/pdf/2505.17810)  

**Abstract**: Approximate nearest neighbor (ANN) search is a performance-critical component of many machine learning pipelines. Rigorous benchmarking is essential for evaluating the performance of vector indexes for ANN search. However, the datasets of the existing benchmarks are no longer representative of the current applications of ANN search. Hence, there is an urgent need for an up-to-date set of benchmarks. To this end, we introduce Vector Index Benchmark for Embeddings (VIBE), an open source project for benchmarking ANN algorithms. VIBE contains a pipeline for creating benchmark datasets using dense embedding models characteristic of modern applications, such as retrieval-augmented generation (RAG). To replicate real-world workloads, we also include out-of-distribution (OOD) datasets where the queries and the corpus are drawn from different distributions. We use VIBE to conduct a comprehensive evaluation of SOTA vector indexes, benchmarking 21 implementations on 12 in-distribution and 6 out-of-distribution datasets. 

---
