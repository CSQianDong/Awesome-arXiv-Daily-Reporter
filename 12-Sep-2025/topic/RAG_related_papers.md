# Retrieval-Augmented Generation for Reliable Interpretation of Radio Regulations 

**Authors**: Zakaria El Kassimi, Fares Fourati, Mohamed-Slim Alouini  

**Link**: [PDF](https://arxiv.org/pdf/2509.09651)  

**Abstract**: We study question answering in the domain of radio regulations, a legally sensitive and high-stakes area. We propose a telecom-specific Retrieval-Augmented Generation (RAG) pipeline and introduce, to our knowledge, the first multiple-choice evaluation set for this domain, constructed from authoritative sources using automated filtering and human validation. To assess retrieval quality, we define a domain-specific retrieval metric, under which our retriever achieves approximately 97% accuracy. Beyond retrieval, our approach consistently improves generation accuracy across all tested models. In particular, while naively inserting documents without structured retrieval yields only marginal gains for GPT-4o (less than 1%), applying our pipeline results in nearly a 12% relative improvement. These findings demonstrate that carefully targeted grounding provides a simple yet strong baseline and an effective domain-specific solution for regulatory question answering. All code and evaluation scripts, along with our derived question-answer dataset, are available at this https URL. 

---
# MetaRAG: Metamorphic Testing for Hallucination Detection in RAG Systems 

**Authors**: Channdeth Sok, David Luz, Yacine Haddam  

**Link**: [PDF](https://arxiv.org/pdf/2509.09360)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in enterprise applications, yet their reliability remains limited by hallucinations, i.e., confident but factually incorrect information. Existing detection approaches, such as SelfCheckGPT and MetaQA, primarily target standalone LLMs and do not address the unique challenges of Retrieval-Augmented Generation (RAG) systems, where responses must be consistent with retrieved evidence. We therefore present MetaRAG, a metamorphic testing framework for hallucination detection in Retrieval-Augmented Generation (RAG) systems. MetaRAG operates in a real-time, unsupervised, black-box setting, requiring neither ground-truth references nor access to model internals, making it suitable for proprietary and high-stakes domains. The framework proceeds in four stages: (1) decompose answers into atomic factoids, (2) generate controlled mutations of each factoid using synonym and antonym substitutions, (3) verify each variant against the retrieved context (synonyms are expected to be entailed and antonyms contradicted), and (4) aggregate penalties for inconsistencies into a response-level hallucination score. Crucially for identity-aware AI, MetaRAG localizes unsupported claims at the factoid span where they occur (e.g., pregnancy-specific precautions, LGBTQ+ refugee rights, or labor eligibility), allowing users to see flagged spans and enabling system designers to configure thresholds and guardrails for identity-sensitive queries. Experiments on a proprietary enterprise dataset illustrate the effectiveness of MetaRAG for detecting hallucinations and enabling trustworthy deployment of RAG-based conversational agents. We also outline a topic-based deployment design that translates MetaRAG's span-level scores into identity-aware safeguards; this design is discussed but not evaluated in our experiments. 

---
# Automated Evidence Extraction and Scoring for Corporate Climate Policy Engagement: A Multilingual RAG Approach 

**Authors**: Imene Kolli, Ario Saeid Vaghefi, Chiara Colesanti Senni, Shantam Raj, Markus Leippold  

**Link**: [PDF](https://arxiv.org/pdf/2509.08907)  

**Abstract**: InfluenceMap's LobbyMap Platform monitors the climate policy engagement of over 500 companies and 250 industry associations, assessing each entity's support or opposition to science-based policy pathways for achieving the Paris Agreement's goal of limiting global warming to 1.5°C. Although InfluenceMap has made progress with automating key elements of the analytical workflow, a significant portion of the assessment remains manual, making it time- and labor-intensive and susceptible to human error. We propose an AI-assisted framework to accelerate the monitoring of corporate climate policy engagement by leveraging Retrieval-Augmented Generation to automate the most time-intensive extraction of relevant evidence from large-scale textual data. Our evaluation shows that a combination of layout-aware parsing, the Nomic embedding model, and few-shot prompting strategies yields the best performance in extracting and classifying evidence from multilingual corporate documents. We conclude that while the automated RAG system effectively accelerates evidence extraction, the nuanced nature of the analysis necessitates a human-in-the-loop approach where the technology augments, rather than replaces, expert judgment to ensure accuracy. 

---
# Recurrence Meets Transformers for Universal Multimodal Retrieval 

**Authors**: Davide Caffagni, Sara Sarto, Marcella Cornia, Lorenzo Baraldi, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2509.08897)  

**Abstract**: With the rapid advancement of multimodal retrieval and its application in LLMs and multimodal LLMs, increasingly complex retrieval tasks have emerged. Existing methods predominantly rely on task-specific fine-tuning of vision-language models and are limited to single-modality queries or documents. In this paper, we propose ReT-2, a unified retrieval model that supports multimodal queries, composed of both images and text, and searches across multimodal document collections where text and images coexist. ReT-2 leverages multi-layer representations and a recurrent Transformer architecture with LSTM-inspired gating mechanisms to dynamically integrate information across layers and modalities, capturing fine-grained visual and textual details. We evaluate ReT-2 on the challenging M2KR and M-BEIR benchmarks across different retrieval configurations. Results demonstrate that ReT-2 consistently achieves state-of-the-art performance across diverse settings, while offering faster inference and reduced memory usage compared to prior approaches. When integrated into retrieval-augmented generation pipelines, ReT-2 also improves downstream performance on Encyclopedic-VQA and InfoSeek datasets. Our source code and trained models are publicly available at: this https URL 

---
