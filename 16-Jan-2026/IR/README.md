# RoutIR: Fast Serving of Retrieval Pipelines for Retrieval-Augmented Generation 

**Authors**: Eugene Yang, Andrew Yates, Dawn Lawrie, James Mayfield, Trevor Adriaanse  

**Link**: [PDF](https://arxiv.org/pdf/2601.10644)  

**Abstract**: Retrieval models are key components of Retrieval-Augmented Generation (RAG) systems, which generate search queries, process the documents returned, and generate a response. RAG systems are often dynamic and may involve multiple rounds of retrieval. While many state-of-the-art retrieval methods are available through academic IR platforms, these platforms are typically designed for the Cranfield paradigm in which all queries are known up front and can be batch processed offline. This simplification accelerates research but leaves state-of-the-art retrieval models unable to support downstream applications that require online services, such as arbitrary dynamic RAG pipelines that involve looping, feedback, or even self-organizing agents. In this work, we introduce RoutIR, a Python package that provides a simple and efficient HTTP API that wraps arbitrary retrieval methods, including first stage retrieval, reranking, query expansion, and result fusion. By providing a minimal JSON configuration file specifying the retrieval models to serve, RoutIR can be used to construct and query retrieval pipelines on-the-fly using any permutation of available models (e.g., fusing the results of several first-stage retrieval methods followed by reranking). The API automatically performs asynchronous query batching and caches results by default. While many state-of-the-art retrieval methods are already supported by the package, RoutIR is also easily expandable by implementing the Engine abstract class. The package is open-sourced and publicly available on GitHub: this http URL. 

---
# iTIMO: An LLM-empowered Synthesis Dataset for Travel Itinerary Modification 

**Authors**: Zhuoxuan Huang, Yunshan Ma, Hongyu Zhang, Hua Ma, Zhu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2601.10609)  

**Abstract**: Addressing itinerary modification is crucial for enhancing the travel experience as it is a frequent requirement during traveling. However, existing research mainly focuses on fixed itinerary planning, leaving modification underexplored. To bridge this gap, we formally define the itinerary modification task and introduce iTIMO, a dataset specifically tailored for this purpose. We identify the lack of {\itshape need-to-modify} itinerary data as the critical bottleneck hindering research on this task and propose a general pipeline to overcome it. This pipeline frames the generation of such data as an intent-driven perturbation task. It instructs large language models to perturb real world itineraries using three atomic editing operations: REPLACE, ADD, and DELETE. Each perturbation is grounded in three intents, including disruptions of popularity, spatial distance, and category diversity. Furthermore, a hybrid evaluation metric is designed to ensure perturbation effectiveness. We conduct comprehensive experiments on iTIMO, revealing the limitations of current LLMs and lead to several valuable directions for future research. Dataset and corresponding code are available at this https URL. 

---
# Development of Ontological Knowledge Bases by Leveraging Large Language Models 

**Authors**: Le Ngoc Luyen, Marie-Hélène Abel, Philippe Gouspillou  

**Link**: [PDF](https://arxiv.org/pdf/2601.10436)  

**Abstract**: Ontological Knowledge Bases (OKBs) play a vital role in structuring domain-specific knowledge and serve as a foundation for effective knowledge management systems. However, their traditional manual development poses significant challenges related to scalability, consistency, and adaptability. Recent advancements in Generative AI, particularly Large Language Models (LLMs), offer promising solutions for automating and enhancing OKB development. This paper introduces a structured, iterative methodology leveraging LLMs to optimize knowledge acquisition, automate ontology artifact generation, and enable continuous refinement cycles. We demonstrate this approach through a detailed case study focused on developing a user context profile ontology within the vehicle sales domain. Key contributions include significantly accelerated ontology construction processes, improved ontological consistency, effective bias mitigation, and enhanced transparency in the ontology engineering process. Our findings highlight the transformative potential of integrating LLMs into ontology development, notably improving scalability, integration capabilities, and overall efficiency in knowledge management systems. 

---
# STCRank: Spatio-temporal Collaborative Ranking for Interactive Recommender System at Kuaishou E-shop 

**Authors**: Boyang Xia, Ruilin Bao, Hanjun Jiang, Jun Wang, Wenwu Ou  

**Link**: [PDF](https://arxiv.org/pdf/2601.10027)  

**Abstract**: As a popular e-commerce platform, Kuaishou E-shop provides precise personalized product recommendations to tens of millions of users every day. To better respond real-time user feedback, we have deployed an interactive recommender system (IRS) alongside our core homepage recommender system. This IRS is triggered by user click on homepage, and generates a series of highly relevant recommendations based on the clicked item to meet focused browsing demands. Different from traditional e-commerce RecSys, the full-screen UI and immersive swiping down functionality present two distinct challenges for regular ranking system. First, there exists explicit interference (overlap or conflicts) between ranking objectives, i.e., conversion, view and swipe down. This is because there are intrinsic behavioral co-occurrences under the premise of immersive browsing and swiping down functionality. Second, the ranking system is prone to temporal greedy traps in sequential recommendation slot transitions, which is caused by full-screen UI design. To alleviate these challenges, we propose a novel Spatio-temporal collaborative ranking (STCRank) framework to achieve collaboration between multi-objectives within one slot (spatial) and between multiple sequential recommondation slots. In multi-objective collaboration (MOC) module, we push Pareto frontier by mitigating the objective overlaps and conflicts. In multi-slot collaboration (MSC) module, we achieve global optima on overall sequential slots by dual-stage look-ahead ranking mechanism. Extensive experiments demonstrate our proposed method brings about purchase and DAU co-growth. The proposed system has been already deployed at Kuaishou E-shop since 2025.6. 

---
# Grounding Agent Memory in Contextual Intent 

**Authors**: Ruozhen Yang, Yucheng Jiang, Yueqi Jiang, Priyanka Kargupta, Yunyi Zhang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2601.10702)  

**Abstract**: Deploying large language models in long-horizon, goal-oriented interactions remains challenging because similar entities and facts recur under different latent goals and constraints, causing memory systems to retrieve context-mismatched evidence. We propose STITCH (Structured Intent Tracking in Contextual History), an agentic memory system that indexes each trajectory step with a structured retrieval cue, contextual intent, and retrieves history by matching the current step's intent. Contextual intent provides compact signals that disambiguate repeated mentions and reduce interference: (1) the current latent goal defining a thematic segment, (2) the action type, and (3) the salient entity types anchoring which attributes matter. During inference, STITCH filters and prioritizes memory snippets by intent compatibility, suppressing semantically similar but context-incompatible history.
For evaluation, we introduce CAME-Bench, a benchmark for context-aware retrieval in realistic, dynamic, goal-oriented trajectories. Across CAME-Bench and LongMemEval, STITCH achieves state-of-the-art performance, outperforming the strongest baseline by 35.6%, with the largest gains as trajectory length increases. Our analysis shows that intent indexing substantially reduces retrieval noise, supporting intent-aware memory for robust long-horizon reasoning. 

---
# From Single to Multi-Agent Reasoning: Advancing GeneGPT for Genomics QA 

**Authors**: Kimia Abedini, Farzad Shami, Gianmaria Silvello  

**Link**: [PDF](https://arxiv.org/pdf/2601.10581)  

**Abstract**: Comprehending genomic information is essential for biomedical research, yet extracting data from complex distributed databases remains challenging. Large language models (LLMs) offer potential for genomic Question Answering (QA) but face limitations due to restricted access to domain-specific databases. GeneGPT is the current state-of-the-art system that enhances LLMs by utilizing specialized API calls, though it is constrained by rigid API dependencies and limited adaptability. We replicate GeneGPT and propose GenomAgent, a multi-agent framework that efficiently coordinates specialized agents for complex genomics queries. Evaluated on nine tasks from the GeneTuring benchmark, GenomAgent outperforms GeneGPT by 12% on average, and its flexible architecture extends beyond genomics to various scientific domains needing expert knowledge extraction. 

---
# An Efficient Long-Context Ranking Architecture With Calibrated LLM Distillation: Application to Person-Job Fit 

**Authors**: Warren Jouanneau, Emma Jouffroy, Marc Palyart  

**Link**: [PDF](https://arxiv.org/pdf/2601.10321)  

**Abstract**: Finding the most relevant person for a job proposal in real time is challenging, especially when resumes are long, structured, and multilingual. In this paper, we propose a re-ranking model based on a new generation of late cross-attention architecture, that decomposes both resumes and project briefs to efficiently handle long-context inputs with minimal computational overhead. To mitigate historical data biases, we use a generative large language model (LLM) as a teacher, generating fine-grained, semantically grounded supervision. This signal is distilled into our student model via an enriched distillation loss function. The resulting model produces skill-fit scores that enable consistent and interpretable person-job matching. Experiments on relevance, ranking, and calibration metrics demonstrate that our approach outperforms state-of-the-art baselines. 

---
# AWED-FiNER: Agents, Web applications, and Expert Detectors for Fine-grained Named Entity Recognition across 36 Languages for 6.6 Billion Speakers 

**Authors**: Prachuryya Kaushik, Ashish Anand  

**Link**: [PDF](https://arxiv.org/pdf/2601.10161)  

**Abstract**: We introduce AWED-FiNER, an open-source ecosystem designed to bridge the gap in Fine-grained Named Entity Recognition (FgNER) for 36 global languages spoken by more than 6.6 billion people. While Large Language Models (LLMs) dominate general Natural Language Processing (NLP) tasks, they often struggle with low-resource languages and fine-grained NLP tasks. AWED-FiNER provides a collection of agentic toolkits, web applications, and several state-of-the-art expert models that provides FgNER solutions across 36 languages. The agentic tools enable to route multilingual text to specialized expert models and fetch FgNER annotations within seconds. The web-based platforms provide ready-to-use FgNER annotation service for non-technical users. Moreover, the collection of language specific extremely small sized open-source state-of-the-art expert models facilitate offline deployment in resource contraint scenerios including edge devices. AWED-FiNER covers languages spoken by over 6.6 billion people, including a specific focus on vulnerable languages such as Bodo, Manipuri, Bishnupriya, and Mizo. The resources can be accessed here: Agentic Tool (this https URL), Web Application (this https URL), and 49 Expert Detector Models (this https URL). 

---
# Efficient Content-based Recommendation Model Training via Noise-aware Coreset Selection 

**Authors**: Hung Vinh Tran, Tong Chen, Hechuan Wen, Quoc Viet Hung Nguyen, Bin Cui, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2601.10067)  

**Abstract**: Content-based recommendation systems (CRSs) utilize content features to predict user-item interactions, serving as essential tools for helping users navigate information-rich web services. However, ensuring the effectiveness of CRSs requires large-scale and even continuous model training to accommodate diverse user preferences, resulting in significant computational costs and resource demands. A promising approach to this challenge is coreset selection, which identifies a small but representative subset of data samples that preserves model quality while reducing training overhead. Yet, the selected coreset is vulnerable to the pervasive noise in user-item interactions, particularly when it is minimally sized. To this end, we propose Noise-aware Coreset Selection (NaCS), a specialized framework for CRSs. NaCS constructs coresets through submodular optimization based on training gradients, while simultaneously correcting noisy labels using a progressively trained model. Meanwhile, we refine the selected coreset by filtering out low-confidence samples through uncertainty quantification, thereby avoid training with unreliable interactions. Through extensive experiments, we show that NaCS produces higher-quality coresets for CRSs while achieving better efficiency than existing coreset selection techniques. Notably, NaCS recovers 93-95\% of full-dataset training performance using merely 1\% of the training data. The source code is available at \href{this https URL}{this https URL}. 

---
# FaTRQ: Tiered Residual Quantization for LLM Vector Search in Far-Memory-Aware ANNS Systems 

**Authors**: Tianqi Zhang, Flavio Ponzina, Tajana Rosing  

**Link**: [PDF](https://arxiv.org/pdf/2601.09985)  

**Abstract**: Approximate Nearest-Neighbor Search (ANNS) is a key technique in retrieval-augmented generation (RAG), enabling rapid identification of the most relevant high-dimensional embeddings from massive vector databases. Modern ANNS engines accelerate this process using prebuilt indexes and store compressed vector-quantized representations in fast memory. However, they still rely on a costly second-pass refinement stage that reads full-precision vectors from slower storage like SSDs. For modern text and multimodal embeddings, these reads now dominate the latency of the entire query. We propose FaTRQ, a far-memory-aware refinement system using tiered memory that eliminates the need to fetch full vectors from storage. It introduces a progressive distance estimator that refines coarse scores using compact residuals streamed from far memory. Refinement stops early once a candidate is provably outside the top-k. To support this, we propose tiered residual quantization, which encodes residuals as ternary values stored efficiently in far memory. A custom accelerator is deployed in a CXL Type-2 device to perform low-latency refinement locally. Together, FaTRQ improves the storage efficiency by 2.4$\times$ and improves the throughput by up to 9$ \times$ than SOTA GPU ANNS system. 

---
# From SERPs to Agents: A Platform for Comparative Studies of Information Interaction 

**Authors**: Saber Zerhoudi, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2601.09937)  

**Abstract**: The diversification of information access systems, from RAG to autonomous agents, creates a critical need for comparative user studies. However, the technical overhead to deploy and manage these distinct systems is a major barrier. We present UXLab, an open-source system for web-based user studies that addresses this challenge. Its core is a web-based dashboard enabling the complete, no-code configuration of complex experimental designs. Researchers can visually manage the full study, from recruitment to comparing backends like traditional search, vector databases, and LLMs. We demonstrate UXLab's value via a micro case study comparing user behavior with RAG versus an autonomous agent. UXLab allows researchers to focus on experimental design and analysis, supporting future multi-modal interaction research. 

---
# In-Browser Agents for Search Assistance 

**Authors**: Saber Zerhoudi, Michael Granitzer  

**Link**: [PDF](https://arxiv.org/pdf/2601.09928)  

**Abstract**: A fundamental tension exists between the demand for sophisticated AI assistance in web search and the need for user data privacy. Current centralized models require users to transmit sensitive browsing data to external services, which limits user control. In this paper, we present a browser extension that provides a viable in-browser alternative. We introduce a hybrid architecture that functions entirely on the client side, combining two components: (1) an adaptive probabilistic model that learns a user's behavioral policy from direct feedback, and (2) a Small Language Model (SLM), running in the browser, which is grounded by the probabilistic model to generate context-aware suggestions. To evaluate this approach, we conducted a three-week longitudinal user study with 18 participants. Our results show that this privacy-preserving approach is highly effective at adapting to individual user behavior, leading to measurably improved search efficiency. This work demonstrates that sophisticated AI assistance is achievable without compromising user privacy or data control. 

---
# Continuum Memory Architectures for Long-Horizon LLM Agents 

**Authors**: Joe Logan  

**Link**: [PDF](https://arxiv.org/pdf/2601.09913)  

**Abstract**: Retrieval-augmented generation (RAG) has become the default strategy for providing large language model (LLM) agents with contextual knowledge. Yet RAG treats memory as a stateless lookup table: information persists indefinitely, retrieval is read-only, and temporal continuity is absent. We define the \textit{Continuum Memory Architecture} (CMA), a class of systems that maintain and update internal state across interactions through persistent storage, selective retention, associative routing, temporal chaining, and consolidation into higher-order abstractions. Rather than disclosing implementation specifics, we specify the architectural requirements CMA imposes and show consistent behavioral advantages on tasks that expose RAG's structural inability to accumulate, mutate, or disambiguate memory. The empirical probes (knowledge updates, temporal association, associative recall, contextual disambiguation) demonstrate that CMA is a necessary architectural primitive for long-horizon agents while highlighting open challenges around latency, drift, and interpretability. 

---
# Clinical Document Metadata Extraction: A Scoping Review 

**Authors**: Kurt Miller, Qiuhao Lu, William Hersh, Kirk Roberts, Steven Bedrick, Andrew Wen, Hongfang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.09730)  

**Abstract**: Clinical document metadata, such as document type, structure, author role, medical specialty, and encounter setting, is essential for accurate interpretation of information captured in clinical documents. However, vast documentation heterogeneity and drift over time challenge harmonization of document metadata. Automated extraction methods have emerged to coalesce metadata from disparate practices into target schema. This scoping review aims to catalog research on clinical document metadata extraction, identify methodological trends and applications, and highlight gaps. We followed the PRISMA-ScR (Preferred Reporting Items for Systematic Reviews and Meta-Analyses Extension for Scoping Reviews) guidelines to identify articles that perform clinical document metadata extraction. We initially found and screened 266 articles published between January 2011 and August 2025, then comprehensively reviewed 67 we deemed relevant to our study. Among the articles included, 45 were methodological, 17 used document metadata as features in a downstream application, and 5 analyzed document metadata composition. We observe myriad purposes for methodological study and application types. Available labelled public data remains sparse except for structural section datasets. Methods for extracting document metadata have progressed from largely rule-based and traditional machine learning with ample feature engineering to transformer-based architectures with minimal feature engineering. The emergence of large language models has enabled broader exploration of generalizability across tasks and datasets, allowing the possibility of advanced clinical text processing systems. We anticipate that research will continue to expand into richer document metadata representations and integrate further into clinical applications and workflows. 

---
# SciNets: Graph-Constrained Multi-Hop Reasoning for Scientific Literature Synthesis 

**Authors**: Sauhard Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2601.09727)  

**Abstract**: Cross-domain scientific synthesis requires connecting mechanistic explanations across fragmented literature, a capability that remains challenging for both retrieval-based systems and unconstrained language models. While recent work has applied large language models to scientific summarization and question answering, these approaches provide limited control over reasoning depth and structural grounding. We frame mechanistic synthesis as a graph-constrained multi-hop reasoning problem over literature-derived concept graphs. Given a scientific query and a compact, query-local corpus, SciNets constructs a directed concept graph and synthesizes mechanistic explanations by identifying multi-hop reasoning paths that connect concepts that rarely co-occur within individual papers. We systematically compare shortest-path reasoning, k-shortest paths with diversity constraints, stochastic random walks, and a retrieval-augmented language model baseline. Rather than evaluating correctness, which is often indeterminate when synthesizing connections across distributed sources, we introduce a behavioral framework that measures symbolic reasoning depth, mechanistic diversity, and grounding stability. Across machine learning, biology, and climate science tasks, explicit graph constraints enable controllable multi-hop reasoning while revealing a consistent trade-off: deeper and more diverse symbolic reasoning increases grounding instability, whereas shortest-path reasoning remains highly stable but structurally conservative. These findings provide a systematic behavioral characterization of the limits and capabilities of current graph-LLM integration for scientific synthesis. 

---
# Introducing Axlerod: An LLM-based Chatbot for Assisting Independent Insurance Agents 

**Authors**: Adam Bradley, John Hastings, Khandaker Mamun Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2601.09715)  

**Abstract**: The insurance industry is undergoing a paradigm shift through the adoption of artificial intelligence (AI) technologies, particularly in the realm of intelligent conversational agents. Chatbots have evolved into sophisticated AI-driven systems capable of automating complex workflows, including policy recommendation and claims triage, while simultaneously enabling dynamic, context-aware user engagement. This paper presents the design, implementation, and empirical evaluation of Axlerod, an AI-powered conversational interface designed to improve the operational efficiency of independent insurance agents. Leveraging natural language processing (NLP), retrieval-augmented generation (RAG), and domain-specific knowledge integration, Axlerod demonstrates robust capabilities in parsing user intent, accessing structured policy databases, and delivering real-time, contextually relevant responses. Experimental results underscore Axlerod's effectiveness, achieving an overall accuracy of 93.18% in policy retrieval tasks while reducing the average search time by 2.42 seconds. This work contributes to the growing body of research on enterprise-grade AI applications in insurtech, with a particular focus on agent-assistive rather than consumer-facing architectures. 

---
