# AccurateRAG: A Framework for Building Accurate Retrieval-Augmented Question-Answering Applications 

**Authors**: Linh The Nguyen, Chi Tran, Dung Ngoc Nguyen, Van-Cuong Pham, Hoang Ngo, Dat Quoc Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.02243)  

**Abstract**: We introduce AccurateRAG -- a novel framework for constructing high-performance question-answering applications based on retrieval-augmented generation (RAG). Our framework offers a pipeline for development efficiency with tools for raw dataset processing, fine-tuning data generation, text embedding & LLM fine-tuning, output evaluation, and building RAG systems locally. Experimental results show that our framework outperforms previous strong baselines and obtains new state-of-the-art question-answering performance on benchmark datasets. 

---
# Stream RAG: Instant and Accurate Spoken Dialogue Systems with Streaming Tool Usage 

**Authors**: Siddhant Arora, Haidar Khan, Kai Sun, Xin Luna Dong, Sajal Choudhary, Seungwhan Moon, Xinyuan Zhang, Adithya Sagar, Surya Teja Appini, Kaushik Patnaik, Sanat Sharma, Shinji Watanabe, Anuj Kumar, Ahmed Aly, Yue Liu, Florian Metze, Zhaojiang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.02044)  

**Abstract**: End-to-end speech-in speech-out dialogue systems are emerging as a powerful alternative to traditional ASR-LLM-TTS pipelines, generating more natural, expressive responses with significantly lower latency. However, these systems remain prone to hallucinations due to limited factual grounding. While text-based dialogue systems address this challenge by integrating tools such as web search and knowledge graph APIs, we introduce the first approach to extend tool use directly into speech-in speech-out systems. A key challenge is that tool integration substantially increases response latency, disrupting conversational flow. To mitigate this, we propose Streaming Retrieval-Augmented Generation (Streaming RAG), a novel framework that reduces user-perceived latency by predicting tool queries in parallel with user speech, even before the user finishes speaking. Specifically, we develop a post-training pipeline that teaches the model when to issue tool calls during ongoing speech and how to generate spoken summaries that fuse audio queries with retrieved text results, thereby improving both accuracy and responsiveness. To evaluate our approach, we construct AudioCRAG, a benchmark created by converting queries from the publicly available CRAG dataset into speech form. Experimental results demonstrate that our streaming RAG approach increases QA accuracy by up to 200% relative (from 11.1% to 34.2% absolute) and further enhances user experience by reducing tool use latency by 20%. Importantly, our streaming RAG approach is modality-agnostic and can be applied equally to typed input, paving the way for more agentic, real-time AI assistants. 

---
# RAG-BioQA Retrieval-Augmented Generation for Long-Form Biomedical Question Answering 

**Authors**: Lovely Yeswanth Panchumarthi, Sai Prasad Gudari, Atharva Negi, Praveen Raj Budime, Harsit Upadhya  

**Link**: [PDF](https://arxiv.org/pdf/2510.01612)  

**Abstract**: The exponential growth of biomedical literature creates significant challenges for accessing precise medical information. Current biomedical question-answering systems primarily focus on short-form answers, failing to provide the comprehensive explanations necessary for clinical decision-making. We present RAG-BioQA, a novel framework combining retrieval-augmented generation with domain-specific fine-tuning to produce evidence-based, long-form biomedical answers. Our approach integrates BioBERT embeddings with FAISS indexing and compares various re-ranking strategies (BM25, ColBERT, MonoT5) to optimize context selection before synthesizing evidence through a fine-tuned T5 model. Experimental results on the PubMedQA dataset show significant improvements over baselines, with our best model achieving substantial gains across BLEU, ROUGE, and METEOR metrics, advancing the state of accessible, evidence-based biomedical knowledge retrieval. 

---
# Learning to Reason for Hallucination Span Detection 

**Authors**: Hsuan Su, Ting-Yao Hu, Hema Swetha Koppula, Kundan Krishna, Hadi Pouransari, Cheng-Yu Hsieh, Cem Koc, Joseph Yitan Cheng, Oncel Tuzel, Raviteja Vemulapalli  

**Link**: [PDF](https://arxiv.org/pdf/2510.02173)  

**Abstract**: Large language models (LLMs) often generate hallucinations -- unsupported content that undermines reliability. While most prior works frame hallucination detection as a binary task, many real-world applications require identifying hallucinated spans, which is a multi-step decision making process. This naturally raises the question of whether explicit reasoning can help the complex task of detecting hallucination spans. To answer this question, we first evaluate pretrained models with and without Chain-of-Thought (CoT) reasoning, and show that CoT reasoning has the potential to generate at least one correct answer when sampled multiple times. Motivated by this, we propose RL4HS, a reinforcement learning framework that incentivizes reasoning with a span-level reward function. RL4HS builds on Group Relative Policy Optimization and introduces Class-Aware Policy Optimization to mitigate reward imbalance issue. Experiments on the RAGTruth benchmark (summarization, question answering, data-to-text) show that RL4HS surpasses pretrained reasoning models and supervised fine-tuning, demonstrating the necessity of reinforcement learning with span-level rewards for detecting hallucination spans. 

---
# Retrieval-Augmented Framework for LLM-Based Clinical Decision Support 

**Authors**: Leon Garza, Anantaa Kotal, Michael A. Grasso, Emre Umucu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01363)  

**Abstract**: The increasing complexity of clinical decision-making, alongside the rapid expansion of electronic health records (EHR), presents both opportunities and challenges for delivering data-informed care. This paper proposes a clinical decision support system powered by Large Language Models (LLMs) to assist prescribing clinicians. The system generates therapeutic suggestions by analyzing historical EHR data, including patient demographics, presenting complaints, clinical symptoms, diagnostic information, and treatment histories. The framework integrates natural language processing with structured clinical inputs to produce contextually relevant recommendations. Rather than replacing clinician judgment, it is designed to augment decision-making by retrieving and synthesizing precedent cases with comparable characteristics, drawing on local datasets or federated sources where applicable. At its core, the system employs a retrieval-augmented generation (RAG) pipeline that harmonizes unstructured narratives and codified data to support LLM-based inference. We outline the system's technical components, including representation representation alignment and generation strategies. Preliminary evaluations, conducted with de-identified and synthetic clinical datasets, examine the clinical plausibility and consistency of the model's outputs. Early findings suggest that LLM-based tools may provide valuable decision support in prescribing workflows when appropriately constrained and rigorously validated. This work represents an initial step toward integration of generative AI into real-world clinical decision-making with an emphasis on transparency, safety, and alignment with established practices. 

---
# OntoLogX: Ontology-Guided Knowledge Graph Extraction from Cybersecurity Logs with Large Language Models 

**Authors**: Luca Cotti, Idilio Drago, Anisa Rula, Devis Bianchini, Federico Cerutti  

**Link**: [PDF](https://arxiv.org/pdf/2510.01409)  

**Abstract**: System logs represent a valuable source of Cyber Threat Intelligence (CTI), capturing attacker behaviors, exploited vulnerabilities, and traces of malicious activity. Yet their utility is often limited by lack of structure, semantic inconsistency, and fragmentation across devices and sessions. Extracting actionable CTI from logs therefore requires approaches that can reconcile noisy, heterogeneous data into coherent and interoperable representations. We introduce OntoLogX, an autonomous Artificial Intelligence (AI) agent that leverages Large Language Models (LLMs) to transform raw logs into ontology-grounded Knowledge Graphs (KGs). OntoLogX integrates a lightweight log ontology with Retrieval Augmented Generation (RAG) and iterative correction steps, ensuring that generated KGs are syntactically and semantically valid. Beyond event-level analysis, the system aggregates KGs into sessions and employs a LLM to predict MITRE ATT&CK tactics, linking low-level log evidence to higher-level adversarial objectives. We evaluate OntoLogX on both logs from a public benchmark and a real-world honeypot dataset, demonstrating robust KG generation across multiple KGs backends and accurate mapping of adversarial activity to ATT&CK tactics. Results highlight the benefits of retrieval and correction for precision and recall, the effectiveness of code-oriented models in structured log analysis, and the value of ontology-grounded representations for actionable CTI extraction. 

---
# REBot: From RAG to CatRAG with Semantic Enrichment and Graph Routing 

**Authors**: Thanh Ma, Tri-Tam La, Lam-Thu Le Huu, Minh-Nghi Nguyen, Khanh-Van Pham Luu, Huu-Hoa Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.01800)  

**Abstract**: Academic regulation advising is essential for helping students interpret and comply with institutional policies, yet building effective systems requires domain specific regulatory resources. To address this challenge, we propose REBot, an LLM enhanced advisory chatbot powered by CatRAG, a hybrid retrieval reasoning framework that integrates retrieval augmented generation with graph based reasoning. CatRAG unifies dense retrieval and graph reasoning, supported by a hierarchical, category labeled knowledge graph enriched with semantic features for domain alignment. A lightweight intent classifier routes queries to the appropriate retrieval modules, ensuring both factual accuracy and contextual depth. We construct a regulation specific dataset and evaluate REBot on classification and question answering tasks, achieving state of the art performance with an F1 score of 98.89%. Finally, we implement a web application that demonstrates the practical value of REBot in real world academic advising scenarios. 

---
# Are LLMs Better GNN Helpers? Rethinking Robust Graph Learning under Deficiencies with Iterative Refinement 

**Authors**: Zhaoyan Wang, Zheng Gao, Arogya Kharel, In-Young Ko  

**Link**: [PDF](https://arxiv.org/pdf/2510.01910)  

**Abstract**: Graph Neural Networks (GNNs) are widely adopted in Web-related applications, serving as a core technique for learning from graph-structured data, such as text-attributed graphs. Yet in real-world scenarios, such graphs exhibit deficiencies that substantially undermine GNN performance. While prior GNN-based augmentation studies have explored robustness against individual imperfections, a systematic understanding of how graph-native and Large Language Models (LLMs) enhanced methods behave under compound deficiencies is still missing. Specifically, there has been no comprehensive investigation comparing conventional approaches and recent LLM-on-graph frameworks, leaving their merits unclear. To fill this gap, we conduct the first empirical study that benchmarks these two lines of methods across diverse graph deficiencies, revealing overlooked vulnerabilities and challenging the assumption that LLM augmentation is consistently superior. Building on empirical findings, we propose Robust Graph Learning via Retrieval-Augmented Contrastive Refinement (RoGRAD) framework. Unlike prior one-shot LLM-as-Enhancer designs, RoGRAD is the first iterative paradigm that leverages Retrieval-Augmented Generation (RAG) to inject retrieval-grounded augmentations by supplying class-consistent, diverse augmentations and enforcing discriminative representations through iterative graph contrastive learning. It transforms LLM augmentation for graphs from static signal injection into dynamic refinement. Extensive experiments demonstrate RoGRAD's superiority over both conventional GNN- and LLM-enhanced baselines, achieving up to 82.43% average improvement. 

---
# Clarifying Semantics of In-Context Examples for Unit Test Generation 

**Authors**: Chen Yang, Lin Yang, Ziqi Wang, Dong Wang, Jianyi Zhou, Junjie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.01994)  

**Abstract**: Recent advances in large language models (LLMs) have enabled promising performance in unit test generation through in-context learning (ICL). However, the quality of in-context examples significantly influences the effectiveness of generated tests-poorly structured or semantically unclear test examples often lead to suboptimal outputs. In this paper, we propose CLAST, a novel technique that systematically refines unit tests to improve their semantic clarity, thereby enhancing their utility as in-context examples. The approach decomposes complex tests into logically clearer ones and improves semantic clarity through a combination of program analysis and LLM-based rewriting. We evaluated CLAST on four open-source and three industrial projects. The results demonstrate that CLAST largely outperforms UTgen, the state-of-the-art refinement technique, in both preserving test effectiveness and enhancing semantic clarity. Specifically, CLAST fully retains the original effectiveness of unit tests, while UTgen reduces compilation success rate (CSR), pass rate (PR), test coverage (Cov), and mutation score (MS) by an average of 12.90%, 35.82%, 4.65%, and 5.07%, respectively. Over 85.33% of participants in our user study preferred the semantic clarity of CLAST-refined tests. Notably, incorporating CLAST-refined tests as examples effectively improves ICL-based unit test generation approaches such as RAGGen and TELPA, resulting in an average increase of 25.97% in CSR, 28.22% in PR, and 45.99% in Cov for generated tests, compared to incorporating UTgen-refined tests. The insights from the follow-up user study not only reinforce CLAST's potential impact in software testing practice but also illuminate avenues for future research. 

---
