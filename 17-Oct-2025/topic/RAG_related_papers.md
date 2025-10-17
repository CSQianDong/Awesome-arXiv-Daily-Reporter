# MR.Rec: Synergizing Memory and Reasoning for Personalized Recommendation Assistant with LLMs 

**Authors**: Jiani Huang, Xingchen Zou, Lianghao Xia, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.14629)  

**Abstract**: The application of Large Language Models (LLMs) in recommender systems faces key challenges in delivering deep personalization and intelligent reasoning, especially for interactive scenarios. Current methods are often constrained by limited context windows and single-turn reasoning, hindering their ability to capture dynamic user preferences and proactively reason over recommendation contexts. To address these limitations, we propose this http URL, a novel framework that synergizes memory and reasoning for LLM-based recommendations. To achieve personalization, we develop a comprehensive Retrieval-Augmented Generation (RAG) system that efficiently indexes and retrieves relevant external memory to enhance LLM personalization capabilities. Furthermore, to enable the synergy between memory and reasoning, our RAG system goes beyond conventional query-based retrieval by integrating reasoning enhanced memory retrieval. Finally, we design a reinforcement learning framework that trains the LLM to autonomously learn effective strategies for both memory utilization and reasoning refinement. By combining dynamic memory retrieval with adaptive reasoning, this approach ensures more accurate, context-aware, and highly personalized recommendations. Extensive experiments demonstrate that this http URL significantly outperforms state-of-the-art baselines across multiple metrics, validating its efficacy in delivering intelligent and personalized recommendations. We will release code and data upon paper notification. 

---
# Multimodal RAG for Unstructured Data:Leveraging Modality-Aware Knowledge Graphs with Hybrid Retrieval 

**Authors**: Rashmi R, Vidyadhar Upadhya  

**Link**: [PDF](https://arxiv.org/pdf/2510.14592)  

**Abstract**: Current Retrieval-Augmented Generation (RAG) systems primarily operate on unimodal textual data, limiting their effectiveness on unstructured multimodal documents. Such documents often combine text, images, tables, equations, and graphs, each contributing unique information. In this work, we present a Modality-Aware Hybrid retrieval Architecture (MAHA), designed specifically for multimodal question answering with reasoning through a modality-aware knowledge graph. MAHA integrates dense vector retrieval with structured graph traversal, where the knowledge graph encodes cross-modal semantics and relationships. This design enables both semantically rich and context-aware retrieval across diverse modalities. Evaluations on multiple benchmark datasets demonstrate that MAHA substantially outperforms baseline methods, achieving a ROUGE-L score of 0.486, providing complete modality coverage. These results highlight MAHA's ability to combine embeddings with explicit document structure, enabling effective multimodal retrieval. Our work establishes a scalable and interpretable retrieval framework that advances RAG systems by enabling modality-aware reasoning over unstructured multimodal data. 

---
# MedTrust-RAG: Evidence Verification and Trust Alignment for Biomedical Question Answering 

**Authors**: Yingpeng Ning, Yuanyuan Sun, Ling Luo, Yanhua Wang, Yuchen Pan, Hongfei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.14400)  

**Abstract**: Biomedical question answering (QA) requires accurate interpretation of complex medical knowledge. Large language models (LLMs) have shown promising capabilities in this domain, with retrieval-augmented generation (RAG) systems enhancing performance by incorporating external medical literature. However, RAG-based approaches in biomedical QA suffer from hallucinations due to post-retrieval noise and insufficient verification of retrieved evidence, undermining response reliability. We propose MedTrust-Guided Iterative RAG, a framework designed to enhance factual consistency and mitigate hallucinations in medical QA. Our method introduces three key innovations. First, it enforces citation-aware reasoning by requiring all generated content to be explicitly grounded in retrieved medical documents, with structured Negative Knowledge Assertions used when evidence is insufficient. Second, it employs an iterative retrieval-verification process, where a verification agent assesses evidence adequacy and refines queries through Medical Gap Analysis until reliable information is obtained. Third, it integrates the MedTrust-Align Module (MTAM) that combines verified positive examples with hallucination-aware negative samples, leveraging Direct Preference Optimization to reinforce citation-grounded reasoning while penalizing hallucination-prone response patterns. Experiments on MedMCQA, MedQA, and MMLU-Med demonstrate that our approach consistently outperforms competitive baselines across multiple model architectures, achieving the best average accuracy with gains of 2.7% for LLaMA3.1-8B-Instruct and 2.4% for Qwen3-8B. 

---
# PluriHop: Exhaustive, Recall-Sensitive QA over Distractor-Rich Corpora 

**Authors**: Mykolas Sveistrys, Richard Kunert  

**Link**: [PDF](https://arxiv.org/pdf/2510.14377)  

**Abstract**: Recent advances in large language models (LLMs) and retrieval-augmented generation (RAG) have enabled progress on question answering (QA) when relevant evidence is in one (single-hop) or multiple (multi-hop) passages. Yet many realistic questions about recurring report data - medical records, compliance filings, maintenance logs - require aggregation across all documents, with no clear stopping point for retrieval and high sensitivity to even one missed passage. We term these pluri-hop questions and formalize them by three criteria: recall sensitivity, exhaustiveness, and exactness. To study this setting, we introduce PluriHopWIND, a diagnostic multilingual dataset of 48 pluri-hop questions built from 191 real-world wind industry reports in German and English. We show that PluriHopWIND is 8-40% more repetitive than other common datasets and thus has higher density of distractor documents, better reflecting practical challenges of recurring report corpora. We test a traditional RAG pipeline as well as graph-based and multimodal variants, and find that none of the tested approaches exceed 40% in statement-wise F1 score. Motivated by this, we propose PluriHopRAG, a RAG architecture that follows a "check all documents individually, filter cheaply" approach: it (i) decomposes queries into document-level subquestions and (ii) uses a cross-encoder filter to discard irrelevant documents before costly LLM reasoning. We find that PluriHopRAG achieves relative F1 score improvements of 18-52% depending on base LLM. Despite its modest size, PluriHopWIND exposes the limitations of current QA systems on repetitive, distractor-rich corpora. PluriHopRAG's performance highlights the value of exhaustive retrieval and early filtering as a powerful alternative to top-k methods. 

---
# Harmonizing Diverse Models: A Layer-wise Merging Strategy for Consistent Generation 

**Authors**: Xujun Peng, Anoop Kumar, Jingyu Wu, Parker Glenn, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.14915)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems leverage Large Language Models (LLMs) to generate accurate and reliable responses that are grounded in retrieved context. However, LLMs often generate inconsistent outputs for semantically equivalent inputs, a problem compounded by the scarcity of consistency-focused training data and the limitations of current fine-tuning techniques in enhancing output consistency. We propose a new approach combining systematic synthetic data generation, triplet loss for better embeddings, and a novel layer-wise model merging approach. Using consistency-aware weights derived from intermediate layer activations, our method effectively integrates knowledge from specialized models. Experimental results how that our merged model significantly enhances output consistency, achieving a ~47.5\% improvement in response similarity over the baseline, thus offering a practical solution for increasing the reliability of an industrial RAG system. 

---
# Less is More: Denoising Knowledge Graphs For Retrieval Augmented Generation 

**Authors**: Yilun Zheng, Dan Yang, Jie Li, Lin Shang, Lihui Chen, Jiahao Xu, Sitao Luan  

**Link**: [PDF](https://arxiv.org/pdf/2510.14271)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enable large language models (LLMs) instant access to relevant information for the generative process, demonstrating their superior performance in addressing common LLM challenges such as hallucination, factual inaccuracy, and the knowledge cutoff. Graph-based RAG further extends this paradigm by incorporating knowledge graphs (KGs) to leverage rich, structured connections for more precise and inferential responses. A critical challenge, however, is that most Graph-based RAG systems rely on LLMs for automated KG construction, often yielding noisy KGs with redundant entities and unreliable relationships. This noise degrades retrieval and generation performance while also increasing computational cost. Crucially, current research does not comprehensively address the denoising problem for LLM-generated KGs. In this paper, we introduce DEnoised knowledge Graphs for Retrieval Augmented Generation (DEG-RAG), a framework that addresses these challenges through: (1) entity resolution, which eliminates redundant entities, and (2) triple reflection, which removes erroneous relations. Together, these techniques yield more compact, higher-quality KGs that significantly outperform their unprocessed counterparts. Beyond the methods, we conduct a systematic evaluation of entity resolution for LLM-generated KGs, examining different blocking strategies, embedding choices, similarity metrics, and entity merging techniques. To the best of our knowledge, this is the first comprehensive exploration of entity resolution in LLM-generated KGs. Our experiments demonstrate that this straightforward approach not only drastically reduces graph size but also consistently improves question answering performance across diverse popular Graph-based RAG variants. 

---
# RAGCap-Bench: Benchmarking Capabilities of LLMs in Agentic Retrieval Augmented Generation Systems 

**Authors**: Jingru Lin, Chen Zhang, Stephen Y. Liu, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.13910)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates key limitations of Large Language Models (LLMs)-such as factual errors, outdated knowledge, and hallucinations-by dynamically retrieving external information. Recent work extends this paradigm through agentic RAG systems, where LLMs act as agents to iteratively plan, retrieve, and reason over complex queries. However, these systems still struggle with challenging multi-hop questions, and their intermediate reasoning capabilities remain underexplored. To address this, we propose RAGCap-Bench, a capability-oriented benchmark for fine-grained evaluation of intermediate tasks in agentic RAG workflows. We analyze outputs from state-of-the-art systems to identify common tasks and the core capabilities required for their execution, then construct a taxonomy of typical LLM errors to design targeted evaluation questions. Experiments show that "slow-thinking" models with stronger RAGCap performance achieve better end-to-end results, underscoring the benchmark's validity and the importance of enhancing these intermediate capabilities. 

---
# Multimodal Retrieval-Augmented Generation with Large Language Models for Medical VQA 

**Authors**: A H M Rezaul Karim, Ozlem Uzuner  

**Link**: [PDF](https://arxiv.org/pdf/2510.13856)  

**Abstract**: Medical Visual Question Answering (MedVQA) enables natural language queries over medical images to support clinical decision-making and patient care. The MEDIQA-WV 2025 shared task addressed wound-care VQA, requiring systems to generate free-text responses and structured wound attributes from images and patient queries. We present the MasonNLP system, which employs a general-domain, instruction-tuned large language model with a retrieval-augmented generation (RAG) framework that incorporates textual and visual examples from in-domain data. This approach grounds outputs in clinically relevant exemplars, improving reasoning, schema adherence, and response quality across dBLEU, ROUGE, BERTScore, and LLM-based metrics. Our best-performing system ranked 3rd among 19 teams and 51 submissions with an average score of 41.37%, demonstrating that lightweight RAG with general-purpose LLMs -- a minimal inference-time layer that adds a few relevant exemplars via simple indexing and fusion, with no extra training or complex re-ranking -- provides a simple and effective baseline for multimodal clinical NLP tasks. 

---
# BenchPress: A Human-in-the-Loop Annotation System for Rapid Text-to-SQL Benchmark Curation 

**Authors**: Fabian Wenz, Omar Bouattour, Devin Yang, Justin Choi, Cecil Gregg, Nesime Tatbul, Çağatay Demiralp  

**Link**: [PDF](https://arxiv.org/pdf/2510.13853)  

**Abstract**: Large language models (LLMs) have been successfully applied to many tasks, including text-to-SQL generation. However, much of this work has focused on publicly available datasets, such as Fiben, Spider, and Bird. Our earlier work showed that LLMs are much less effective in querying large private enterprise data warehouses and released Beaver, the first private enterprise text-to-SQL benchmark. To create Beaver, we leveraged SQL logs, which are often readily available. However, manually annotating these logs to identify which natural language questions they answer is a daunting task. Asking database administrators, who are highly trained experts, to take on additional work to construct and validate corresponding natural language utterances is not only challenging but also quite costly. To address this challenge, we introduce BenchPress, a human-in-the-loop system designed to accelerate the creation of domain-specific text-to-SQL benchmarks. Given a SQL query, BenchPress uses retrieval-augmented generation (RAG) and LLMs to propose multiple natural language descriptions. Human experts then select, rank, or edit these drafts to ensure accuracy and domain alignment. We evaluated BenchPress on annotated enterprise SQL logs, demonstrating that LLM-assisted annotation drastically reduces the time and effort required to create high-quality benchmarks. Our results show that combining human verification with LLM-generated suggestions enhances annotation accuracy, benchmark reliability, and model evaluation robustness. By streamlining the creation of custom benchmarks, BenchPress offers researchers and practitioners a mechanism for assessing text-to-SQL models on a given domain-specific workload. BenchPress is freely available via our public GitHub repository at this https URL and is also accessible on our website at this http URL. 

---
# ADMIT: Few-shot Knowledge Poisoning Attacks on RAG-based Fact Checking 

**Authors**: Yutao Wu, Xiao Liu, Yinghui Li, Yifeng Gao, Yifan Ding, Jiale Ding, Xiang Zheng, Xingjun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.13842)  

**Abstract**: Knowledge poisoning poses a critical threat to Retrieval-Augmented Generation (RAG) systems by injecting adversarial content into knowledge bases, tricking Large Language Models (LLMs) into producing attacker-controlled outputs grounded in manipulated context. Prior work highlights LLMs' susceptibility to misleading or malicious retrieved content. However, real-world fact-checking scenarios are more challenging, as credible evidence typically dominates the retrieval pool. To investigate this problem, we extend knowledge poisoning to the fact-checking setting, where retrieved context includes authentic supporting or refuting evidence. We propose \textbf{ADMIT} (\textbf{AD}versarial \textbf{M}ulti-\textbf{I}njection \textbf{T}echnique), a few-shot, semantically aligned poisoning attack that flips fact-checking decisions and induces deceptive justifications, all without access to the target LLMs, retrievers, or token-level control. Extensive experiments show that ADMIT transfers effectively across 4 retrievers, 11 LLMs, and 4 cross-domain benchmarks, achieving an average attack success rate (ASR) of 86\% at an extremely low poisoning rate of $0.93 \times 10^{-6}$, and remaining robust even in the presence of strong counter-evidence. Compared with prior state-of-the-art attacks, ADMIT improves ASR by 11.2\% across all settings, exposing significant vulnerabilities in real-world RAG-based fact-checking systems. 

---
# From Explainability to Action: A Generative Operational Framework for Integrating XAI in Clinical Mental Health Screening 

**Authors**: Ratna Kandala, Akshata Kishore Moharir, Divya Arvinda Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2510.13828)  

**Abstract**: Explainable Artificial Intelligence (XAI) has been presented as the critical component for unlocking the potential of machine learning in mental health screening (MHS). However, a persistent lab-to-clinic gap remains. Current XAI techniques, such as SHAP and LIME, excel at producing technically faithful outputs such as feature importance scores, but fail to deliver clinically relevant, actionable insights that can be used by clinicians or understood by patients. This disconnect between technical transparency and human utility is the primary barrier to real-world adoption. This paper argues that this gap is a translation problem and proposes the Generative Operational Framework, a novel system architecture that leverages Large Language Models (LLMs) as a central translation engine. This framework is designed to ingest the raw, technical outputs from diverse XAI tools and synthesize them with clinical guidelines (via RAG) to automatically generate human-readable, evidence-backed clinical narratives. To justify our solution, we provide a systematic analysis of the components it integrates, tracing the evolution from intrinsic models to generative XAI. We demonstrate how this framework directly addresses key operational barriers, including workflow integration, bias mitigation, and stakeholder-specific communication. This paper also provides a strategic roadmap for moving the field beyond the generation of isolated data points toward the delivery of integrated, actionable, and trustworthy AI in clinical practice. 

---
# Stop-RAG: Value-Based Retrieval Control for Iterative RAG 

**Authors**: Jaewan Park, Solbee Cho, Jay-Yoon Lee  

**Link**: [PDF](https://arxiv.org/pdf/2510.14337)  

**Abstract**: Iterative retrieval-augmented generation (RAG) enables large language models to answer complex multi-hop questions, but each additional loop increases latency, costs, and the risk of introducing distracting evidence, motivating the need for an efficient stopping strategy. Existing methods either use a predetermined number of iterations or rely on confidence proxies that poorly reflect whether more retrieval will actually help. We cast iterative RAG as a finite-horizon Markov decision process and introduce Stop-RAG, a value-based controller that adaptively decides when to stop retrieving. Trained with full-width forward-view Q($\lambda$) targets from complete trajectories, Stop-RAG learns effective stopping policies while remaining compatible with black-box APIs and existing pipelines. On multi-hop question-answering benchmarks, Stop-RAG consistently outperforms both fixed-iteration baselines and prompting-based stopping with LLMs. These results highlight adaptive stopping as a key missing component in current agentic systems, and demonstrate that value-based control can improve the accuracy of RAG systems. 

---
