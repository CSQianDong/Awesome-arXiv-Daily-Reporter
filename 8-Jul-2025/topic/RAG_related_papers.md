# SIGIR 2025 -- LiveRAG Challenge Report 

**Authors**: David Carmel, Simone Filice, Guy Horowitz, Yoelle Maarek, Oren Somekh, Ran Tavory  

**Link**: [PDF](https://arxiv.org/pdf/2507.04942)  

**Abstract**: The LiveRAG Challenge at SIGIR 2025, held between March and May 2025, provided a competitive platform for advancing Retrieval-Augmented Generation (RAG) technologies. Participants from academia and industry were invited to develop a RAG-based question-answering system using a fixed corpus (Fineweb-10BT) and a common open-source LLM (Falcon3-10B-Instruct). The goal was to facilitate challenging comparisons of retrieval and prompting strategies. During the Live Challenge Day, 70 teams from 27 different countries provided answers and supportive information to 500 unseen questions within a strict two-hour time window. Evaluation was conducted in two stages: first an automated LLM-as-a-judge approach was used to compute correctness and faithfulness score, then a manual review of top ranked submissions was conducted. The finalists were announced on June 12, 2025, with prizes awarded during the LiveRAG Workshop at SIGIR 2025 in Padua, Italy. 

---
# $\textit{Grahak-Nyay:}$ Consumer Grievance Redressal through Large Language Models 

**Authors**: Shrey Ganatra, Swapnil Bhattacharyya, Harshvivek Kashid, Spandan Anaokar, Shruti Nair, Reshma Sekhar, Siddharth Manohar, Rahul Hemrajani, Pushpak Bhattacharyya  

**Link**: [PDF](https://arxiv.org/pdf/2507.04854)  

**Abstract**: Access to consumer grievance redressal in India is often hindered by procedural complexity, legal jargon, and jurisdictional challenges. To address this, we present $\textbf{Grahak-Nyay}$ (Justice-to-Consumers), a chatbot that streamlines the process using open-source Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG). Grahak-Nyay simplifies legal complexities through a concise and up-to-date knowledge base. We introduce three novel datasets: $\textit{GeneralQA}$ (general consumer law), $\textit{SectoralQA}$ (sector-specific knowledge) and $\textit{SyntheticQA}$ (for RAG evaluation), along with $\textit{NyayChat}$, a dataset of 300 annotated chatbot conversations. We also introduce $\textit{Judgments}$ data sourced from Indian Consumer Courts to aid the chatbot in decision making and to enhance user trust. We also propose $\textbf{HAB}$ metrics ($\textbf{Helpfulness, Accuracy, Brevity}$) to evaluate chatbot performance. Legal domain experts validated Grahak-Nyay's effectiveness. Code and datasets will be released. 

---
# BYOKG-RAG: Multi-Strategy Graph Retrieval for Knowledge Graph Question Answering 

**Authors**: Costas Mavromatis, Soji Adeshina, Vassilis N. Ioannidis, Zhen Han, Qi Zhu, Ian Robinson, Bryan Thompson, Huzefa Rangwala, George Karypis  

**Link**: [PDF](https://arxiv.org/pdf/2507.04127)  

**Abstract**: Knowledge graph question answering (KGQA) presents significant challenges due to the structural and semantic variations across input graphs. Existing works rely on Large Language Model (LLM) agents for graph traversal and retrieval; an approach that is sensitive to traversal initialization, as it is prone to entity linking errors and may not generalize well to custom ("bring-your-own") KGs. We introduce BYOKG-RAG, a framework that enhances KGQA by synergistically combining LLMs with specialized graph retrieval tools. In BYOKG-RAG, LLMs generate critical graph artifacts (question entities, candidate answers, reasoning paths, and OpenCypher queries), and graph tools link these artifacts to the KG and retrieve relevant graph context. The retrieved context enables the LLM to iteratively refine its graph linking and retrieval, before final answer generation. By retrieving context from different graph tools, BYOKG-RAG offers a more general and robust solution for QA over custom KGs. Through experiments on five benchmarks spanning diverse KG types, we demonstrate that BYOKG-RAG outperforms the second-best graph retrieval method by 4.5% points while showing better generalization to custom KGs. BYOKG-RAG framework is open-sourced at this https URL. 

---
# Beyond Independent Passages: Adaptive Passage Combination Retrieval for Retrieval Augmented Open-Domain Question Answering 

**Authors**: Ting-Wen Ko, Jyun-Yu Jiang, Pu-Jen Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.04069)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external documents at inference time, enabling up-to-date knowledge access without costly retraining. However, conventional RAG methods retrieve passages independently, often leading to redundant, noisy, or insufficiently diverse context-particularly problematic - particularly problematic in noisy corpora and for multi-hop questions. To address this, we propose Adaptive Passage Combination Retrieval (AdaPCR), a novel framework for open-domain question answering with black-box LMs. AdaPCR explicitly models dependencies between passages by considering passage combinations as units for retrieval and reranking. It consists of a context-aware query reformulation using concatenated passages, and a reranking step trained with a predictive objective aligned with downstream answer likelihood. Crucially, AdaPCR adaptively selects the number of retrieved passages without additional stopping modules. Experiments across several QA benchmarks show that AdaPCR outperforms baselines, particularly in multi-hop reasoning, demonstrating the effectiveness of modeling inter-passage dependencies for improved retrieval. 

---
# SpiritRAG: A Q&A System for Religion and Spirituality in the United Nations Archive 

**Authors**: Yingqiang Gao, Fabian Winiger, Patrick Montjourides, Anastassia Shaitarova, Nianlong Gu, Simon Peng-Keller, Gerold Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2507.04395)  

**Abstract**: Religion and spirituality (R/S) are complex and highly domain-dependent concepts which have long confounded researchers and policymakers. Due to their context-specificity, R/S are difficult to operationalize in conventional archival search strategies, particularly when datasets are very large, poorly accessible, and marked by information noise. As a result, considerable time investments and specialist knowledge is often needed to extract actionable insights related to R/S from general archival sources, increasing reliance on published literature and manual desk reviews. To address this challenge, we present SpiritRAG, an interactive Question Answering (Q&A) system based on Retrieval-Augmented Generation (RAG). Built using 7,500 United Nations (UN) resolution documents related to R/S in the domains of health and education, SpiritRAG allows researchers and policymakers to conduct complex, context-sensitive database searches of very large datasets using an easily accessible, chat-based web interface. SpiritRAG is lightweight to deploy and leverages both UN documents and user provided documents as source material. A pilot test and evaluation with domain experts on 100 manually composed questions demonstrates the practical value and usefulness of SpiritRAG. 

---
# Patient-Centered RAG for Oncology Visit Aid Following the Ottawa Decision Guide 

**Authors**: Siyang Liu, Lawrence Chin-I An, Rada Mihalcea  

**Link**: [PDF](https://arxiv.org/pdf/2507.04026)  

**Abstract**: Effective communication is essential in cancer care, yet patients often face challenges in preparing for complex medical visits. We present an interactive, Retrieval-augmented Generation-assisted system that helps patients progress from uninformed to visit-ready. Our system adapts the Ottawa Personal Decision Guide into a dynamic retrieval-augmented generation workflow, helping users bridge knowledge gaps, clarify personal values and generate useful questions for their upcoming visits. Focusing on localized prostate cancer, we conduct a user study with patients and a clinical expert. Results show high system usability (UMUX Mean = 6.0 out of 7), strong relevance of generated content (Mean = 6.7 out of 7), minimal need for edits, and high clinical faithfulness (Mean = 6.82 out of 7). This work demonstrates the potential of combining patient-centered design with language models to enhance clinical preparation in oncology care. 

---
# AI-VaxGuide: An Agentic RAG-Based LLM for Vaccination Decisions 

**Authors**: Abdellah Zeggai, Ilyes Traikia, Abdelhak Lakehal, Abdennour Boulesnane  

**Link**: [PDF](https://arxiv.org/pdf/2507.03493)  

**Abstract**: Vaccination plays a vital role in global public health, yet healthcare professionals often struggle to access immunization guidelines quickly and efficiently. National protocols and WHO recommendations are typically extensive and complex, making it difficult to extract precise information, especially during urgent situations. This project tackles that issue by developing a multilingual, intelligent question-answering system that transforms static vaccination guidelines into an interactive and user-friendly knowledge base. Built on a Retrieval-Augmented Generation (RAG) framework and enhanced with agent-based reasoning (Agentic RAG), the system provides accurate, context-sensitive answers to complex medical queries. Evaluation shows that Agentic RAG outperforms traditional methods, particularly in addressing multi-step or ambiguous questions. To support clinical use, the system is integrated into a mobile application designed for real-time, point-of-care access to essential vaccine information. AI-VaxGuide model is publicly available on this https URL 

---
# KinyaColBERT: A Lexically Grounded Retrieval Model for Low-Resource Retrieval-Augmented Generation 

**Authors**: Antoine Nzeyimana, Andre Niyongabo Rubungo  

**Link**: [PDF](https://arxiv.org/pdf/2507.03241)  

**Abstract**: The recent mainstream adoption of large language model (LLM) technology is enabling novel applications in the form of chatbots and virtual assistants across many domains. With the aim of grounding LLMs in trusted domains and avoiding the problem of hallucinations, retrieval-augmented generation (RAG) has emerged as a viable solution. In order to deploy sustainable RAG systems in low-resource settings, achieving high retrieval accuracy is not only a usability requirement but also a cost-saving strategy. Through empirical evaluations on a Kinyarwanda-language dataset, we find that the most limiting factors in achieving high retrieval accuracy are limited language coverage and inadequate sub-word tokenization in pre-trained language models. We propose a new retriever model, KinyaColBERT, which integrates two key concepts: late word-level interactions between queries and documents, and a morphology-based tokenization coupled with two-tier transformer encoding. This methodology results in lexically grounded contextual embeddings that are both fine-grained and self-contained. Our evaluation results indicate that KinyaColBERT outperforms strong baselines and leading commercial text embedding APIs on a Kinyarwanda agricultural retrieval benchmark. By adopting this retrieval strategy, we believe that practitioners in other low-resource settings can not only achieve reliable RAG systems but also deploy solutions that are more cost-effective. 

---
# RAG-R1 : Incentivize the Search and Reasoning Capabilities of LLMs through Multi-query Parallelism 

**Authors**: Zhiwen Tan, Jiaming Huang, Qintong Wu, Hongxuan Zhang, Chenyi Zhuang, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2507.02962)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities across various tasks, while they remain prone to generating hallucinated or outdated responses due to their static internal knowledge. Recent advancements in Retrieval-Augmented Generation (RAG) methods have explored enhancing models' search and reasoning capabilities through reinforcement learning (RL). Although these methods demonstrate promising results, they face challenges in training stability and encounter issues such as substantial inference time and restricted capabilities due to the single-query mode. In this paper, we propose RAG-R1, a novel training framework designed to enable LLMs to adaptively leverage internal and external knowledge during the reasoning process. We further expand the generation and retrieval processes within the framework from single-query mode to multi-query parallelism, aimed at reducing inference time and enhancing the model's capabilities. Extensive experiments on seven question-answering benchmarks demonstrate that our method outperforms the strongest baseline by up to 13.2% and decreases inference time by 11.1%. 

---
# RADIANT: Retrieval AugmenteD entIty-context AligNmenT -- Introducing RAG-ability and Entity-Context Divergence 

**Authors**: Vipula Rawte, Rajarshi Roy, Gurpreet Singh, Danush Khanna, Yaswanth Narsupalli, Basab Ghosh, Abhay Gupta, Argha Kamal Samanta, Aditya Shingote, Aadi Krishna Vikram, Vinija Jain, Aman Chadha, Amit Sheth, Amitava Das  

**Link**: [PDF](https://arxiv.org/pdf/2507.02949)  

**Abstract**: As Large Language Models (LLMs) continue to advance, Retrieval-Augmented Generation (RAG) has emerged as a vital technique to enhance factual accuracy by integrating external knowledge into the generation process. However, LLMs often fail to faithfully integrate retrieved evidence into their generated responses, leading to factual inconsistencies. To quantify this gap, we introduce Entity-Context Divergence (ECD), a metric that measures the extent to which retrieved information is accurately reflected in model outputs. We systematically evaluate contemporary LLMs on their ability to preserve factual consistency in retrieval-augmented settings, a capability we define as RAG-ability. Our empirical analysis reveals that RAG-ability remains low across most LLMs, highlighting significant challenges in entity retention and context fidelity. This paper introduces Radiant (Retrieval AugmenteD entIty-context AligNmenT), a novel framework that merges RAG with alignment designed to optimize the interplay between retrieved evidence and generated content. Radiant extends Direct Preference Optimization (DPO) to teach LLMs how to integrate provided additional information into subsequent generations. As a behavior correction mechanism, Radiant boosts RAG performance across varied retrieval scenarios, such as noisy web contexts, knowledge conflicts, and hallucination reduction. This enables more reliable, contextually grounded, and factually coherent content generation. 

---
# Multi-Modal Semantic Parsing for the Interpretation of Tombstone Inscriptions 

**Authors**: Xiao Zhang, Johan Bos  

**Link**: [PDF](https://arxiv.org/pdf/2507.04377)  

**Abstract**: Tombstones are historically and culturally rich artifacts, encapsulating individual lives, community memory, historical narratives and artistic expression. Yet, many tombstones today face significant preservation challenges, including physical erosion, vandalism, environmental degradation, and political shifts. In this paper, we introduce a novel multi-modal framework for tombstones digitization, aiming to improve the interpretation, organization and retrieval of tombstone content. Our approach leverages vision-language models (VLMs) to translate tombstone images into structured Tombstone Meaning Representations (TMRs), capturing both image and text information. To further enrich semantic parsing, we incorporate retrieval-augmented generation (RAG) for integrate externally dependent elements such as toponyms, occupation codes, and ontological concepts. Compared to traditional OCR-based pipelines, our method improves parsing accuracy from an F1 score of 36.1 to 89.5. We additionally evaluate the model's robustness across diverse linguistic and cultural inscriptions, and simulate physical degradation through image fusion to assess performance under noisy or damaged conditions. Our work represents the first attempt to formalize tombstone understanding using large vision-language models, presenting implications for heritage preservation. 

---
# Benchmarking Vector, Graph and Hybrid Retrieval Augmented Generation (RAG) Pipelines for Open Radio Access Networks (ORAN) 

**Authors**: Sarat Ahmad, Zeinab Nezami, Maryam Hafeez, Syed Ali Raza Zaidi  

**Link**: [PDF](https://arxiv.org/pdf/2507.03608)  

**Abstract**: Generative AI (GenAI) is expected to play a pivotal role in enabling autonomous optimization in future wireless networks. Within the ORAN architecture, Large Language Models (LLMs) can be specialized to generate xApps and rApps by leveraging specifications and API definitions from the RAN Intelligent Controller (RIC) platform. However, fine-tuning base LLMs for telecom-specific tasks remains expensive and resource-intensive. Retrieval-Augmented Generation (RAG) offers a practical alternative through in-context learning, enabling domain adaptation without full retraining. While traditional RAG systems rely on vector-based retrieval, emerging variants such as GraphRAG and Hybrid GraphRAG incorporate knowledge graphs or dual retrieval strategies to support multi-hop reasoning and improve factual grounding. Despite their promise, these methods lack systematic, metric-driven evaluations, particularly in high-stakes domains such as ORAN. In this study, we conduct a comparative evaluation of Vector RAG, GraphRAG, and Hybrid GraphRAG using ORAN specifications. We assess performance across varying question complexities using established generation metrics: faithfulness, answer relevance, context relevance, and factual correctness. Results show that both GraphRAG and Hybrid GraphRAG outperform traditional RAG. Hybrid GraphRAG improves factual correctness by 8%, while GraphRAG improves context relevance by 7%. 

---
# The Hidden Threat in Plain Text: Attacking RAG Data Loaders 

**Authors**: Alberto Castagnaro, Umberto Salviati, Mauro Conti, Luca Pajola, Simeone Pizzi  

**Link**: [PDF](https://arxiv.org/pdf/2507.05093)  

**Abstract**: Large Language Models (LLMs) have transformed human-machine interaction since ChatGPT's 2022 debut, with Retrieval-Augmented Generation (RAG) emerging as a key framework that enhances LLM outputs by integrating external knowledge. However, RAG's reliance on ingesting external documents introduces new vulnerabilities. This paper exposes a critical security gap at the data loading stage, where malicious actors can stealthily corrupt RAG pipelines by exploiting document ingestion.
We propose a taxonomy of 9 knowledge-based poisoning attacks and introduce two novel threat vectors -- Content Obfuscation and Content Injection -- targeting common formats (DOCX, HTML, PDF). Using an automated toolkit implementing 19 stealthy injection techniques, we test five popular data loaders, finding a 74.4% attack success rate across 357 scenarios. We further validate these threats on six end-to-end RAG systems -- including white-box pipelines and black-box services like NotebookLM and OpenAI Assistants -- demonstrating high success rates and critical vulnerabilities that bypass filters and silently compromise output integrity. Our results emphasize the urgent need to secure the document ingestion process in RAG systems against covert content manipulations. 

---
# UrbanMind: Towards Urban General Intelligence via Tool-Enhanced Retrieval-Augmented Generation and Multilevel Optimization 

**Authors**: Kai Yang, Zelin Zhu, Chengtao Jian, Hui Ma, Shengjie Zhao, Xiaozhou Ye, Ye Ouyang  

**Link**: [PDF](https://arxiv.org/pdf/2507.04706)  

**Abstract**: Urban general intelligence (UGI) refers to the capacity of AI systems to autonomously perceive, reason, and act within dynamic and complex urban environments. In this paper, we introduce UrbanMind, a tool-enhanced retrieval-augmented generation (RAG) framework designed to facilitate UGI. Central to UrbanMind is a novel architecture based on Continual Retrieval-Augmented MoE-based LLM (C-RAG-LLM), which dynamically incorporates domain-specific knowledge and evolving urban data to support long-term adaptability. The architecture of C-RAG-LLM aligns naturally with a multilevel optimization framework, where different layers are treated as interdependent sub-problems. Each layer has distinct objectives and can be optimized either independently or jointly through a hierarchical learning process. The framework is highly flexible, supporting both end-to-end training and partial layer-wise optimization based on resource or deployment constraints. To remain adaptive under data drift, it is further integrated with an incremental corpus updating mechanism. Evaluations on real-world urban tasks of a variety of complexity verify the effectiveness of the proposed framework. This work presents a promising step toward the realization of general-purpose LLM agents in future urban environments. 

---
# Source Attribution in Retrieval-Augmented Generation 

**Authors**: Ikhtiyor Nematov, Tarik Kalai, Elizaveta Kuzmenko, Gabriele Fugagnoli, Dimitris Sacharidis, Katja Hose, Tomer Sagi  

**Link**: [PDF](https://arxiv.org/pdf/2507.04480)  

**Abstract**: While attribution methods, such as Shapley values, are widely used to explain the importance of features or training data in traditional machine learning, their application to Large Language Models (LLMs), particularly within Retrieval-Augmented Generation (RAG) systems, is nascent and challenging. The primary obstacle is the substantial computational cost, where each utility function evaluation involves an expensive LLM call, resulting in direct monetary and time expenses. This paper investigates the feasibility and effectiveness of adapting Shapley-based attribution to identify influential retrieved documents in RAG. We compare Shapley with more computationally tractable approximations and some existing attribution methods for LLM. Our work aims to: (1) systematically apply established attribution principles to the RAG document-level setting; (2) quantify how well SHAP approximations can mirror exact attributions while minimizing costly LLM interactions; and (3) evaluate their practical explainability in identifying critical documents, especially under complex inter-document relationships such as redundancy, complementarity, and synergy. This study seeks to bridge the gap between powerful attribution techniques and the practical constraints of LLM-based RAG systems, offering insights into achieving reliable and affordable RAG explainability. 

---
# Rethinking and Exploring String-Based Malware Family Classification in the Era of LLMs and RAG 

**Authors**: Yufan Chen, Daoyuan Wu, Juantao Zhong, Zicheng Zhang, Debin Gao, Shuai Wang, Yingjiu Li, Ning Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.04055)  

**Abstract**: Malware Family Classification (MFC) aims to identify the fine-grained family (e.g., GuLoader or BitRAT) to which a potential malware sample belongs, in contrast to malware detection or sample classification that predicts only an Yes/No. Accurate family identification can greatly facilitate automated sample labeling and understanding on crowdsourced malware analysis platforms such as VirusTotal and MalwareBazaar, which generate vast amounts of data daily. In this paper, we explore and assess the feasibility of using traditional binary string features for MFC in the new era of large language models (LLMs) and Retrieval-Augmented Generation (RAG). Specifically, we investigate how Family-Specific String (FSS) features could be utilized in a manner similar to RAG to facilitate MFC. To this end, we develop a curated evaluation framework covering 4,347 samples from 67 malware families, extract and analyze over 25 million strings, and conduct detailed ablation studies to assess the impact of different design choices in four major modules. 

---
# Introducing Answered with Evidence -- a framework for evaluating whether LLM responses to biomedical questions are founded in evidence 

**Authors**: Julian D Baldwin, Christina Dinh, Arjun Mukerji, Neil Sanghavi, Saurabh Gombar  

**Link**: [PDF](https://arxiv.org/pdf/2507.02975)  

**Abstract**: The growing use of large language models (LLMs) for biomedical question answering raises concerns about the accuracy and evidentiary support of their responses. To address this, we present Answered with Evidence, a framework for evaluating whether LLM-generated answers are grounded in scientific literature. We analyzed thousands of physician-submitted questions using a comparative pipeline that included: (1) Alexandria, fka the Atropos Evidence Library, a retrieval-augmented generation (RAG) system based on novel observational studies, and (2) two PubMed-based retrieval-augmented systems (System and Perplexity). We found that PubMed-based systems provided evidence-supported answers for approximately 44% of questions, while the novel evidence source did so for about 50%. Combined, these sources enabled reliable answers to over 70% of biomedical queries. As LLMs become increasingly capable of summarizing scientific content, maximizing their value will require systems that can accurately retrieve both published and custom-generated evidence or generate such evidence in real time. 

---
