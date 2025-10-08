# How public datasets constrain the development of diversity-aware news recommender systems, and what law could do about it 

**Authors**: Max van Drunen, Sanne Vrijenhoek  

**Link**: [PDF](https://arxiv.org/pdf/2510.05952)  

**Abstract**: News recommender systems increasingly determine what news individuals see online. Over the past decade, researchers have extensively critiqued recommender systems that prioritise news based on user engagement. To offer an alternative, researchers have analysed how recommender systems could support the media's ability to fulfil its role in democratic society by recommending news based on editorial values, particularly diversity. However, there continues to be a large gap between normative theory on how news recommender systems should incorporate diversity, and technical literature that designs such systems. We argue that to realise diversity-aware recommender systems in practice, it is crucial to pay attention to the datasets that are needed to train modern news recommenders. We aim to make two main contributions. First, we identify the information a dataset must include to enable the development of the diversity-aware news recommender systems proposed in normative literature. Based on this analysis, we assess the limitations of currently available public datasets, and show what potential they do have to expand research into diversity-aware recommender systems. Second, we analyse why and how European law and policy can be used to provide researchers with structural access to the data they need to develop diversity-aware news recommender systems. 

---
# Limitations of Current Evaluation Practices for Conversational Recommender Systems and the Potential of User Simulation 

**Authors**: Nolwenn Bernard, Krisztian Balog  

**Link**: [PDF](https://arxiv.org/pdf/2510.05624)  

**Abstract**: Research and development on conversational recommender systems (CRSs) critically depends on sound and reliable evaluation methodologies. However, the interactive nature of these systems poses significant challenges for automatic evaluation. This paper critically examines current evaluation practices and identifies two key limitations: the over-reliance on static test collections and the inadequacy of existing evaluation metrics. To substantiate this critique, we analyze real user interactions with nine existing CRSs and demonstrate a striking disconnect between self-reported user satisfaction and performance scores reported in prior literature. To address these limitations, this work explores the potential of user simulation to generate dynamic interaction data, offering a departure from static datasets. Furthermore, we propose novel evaluation metrics, based on a general reward/cost framework, designed to better align with real user satisfaction. Our analysis of different simulation approaches provides valuable insights into their effectiveness and reveals promising initial results, showing improved correlation with system rankings compared to human evaluation. While these findings indicate a significant step forward in CRS evaluation, we also identify areas for future research and refinement in both simulation techniques and evaluation metrics. 

---
# AgentDR Dynamic Recommendation with Implicit Item-Item Relations via LLM-based Agents 

**Authors**: Mingdai Yang, Nurendra Choudhary, Jiangshu Du, Edward W.Huang, Philip S.Yu, Karthik Subbian, Danai Kourta  

**Link**: [PDF](https://arxiv.org/pdf/2510.05598)  

**Abstract**: Recent agent-based recommendation frameworks aim to simulate user behaviors by incorporating memory mechanisms and prompting strategies, but they struggle with hallucinating non-existent items and full-catalog ranking. Besides, a largely underexplored opportunity lies in leveraging LLMs'commonsense reasoning to capture user intent through substitute and complement relationships between items, which are usually implicit in datasets and difficult for traditional ID-based recommenders to capture. In this work, we propose a novel LLM-agent framework, AgenDR, which bridges LLM reasoning with scalable recommendation tools. Our approach delegates full-ranking tasks to traditional models while utilizing LLMs to (i) integrate multiple recommendation outputs based on personalized tool suitability and (ii) reason over substitute and complement relationships grounded in user history. This design mitigates hallucination, scales to large catalogs, and enhances recommendation relevance through relational reasoning. Through extensive experiments on three public grocery datasets, we show that our framework achieves superior full-ranking performance, yielding on average a twofold improvement over its underlying tools. We also introduce a new LLM-based evaluation metric that jointly measures semantic alignment and ranking correctness. 

---
# Automated Research Article Classification and Recommendation Using NLP and ML 

**Authors**: Shadikur Rahman, Hasibul Karim Shanto, Umme Ayman Koana, Syed Muhammad Danish  

**Link**: [PDF](https://arxiv.org/pdf/2510.05495)  

**Abstract**: In the digital era, the exponential growth of scientific publications has made it increasingly difficult for researchers to efficiently identify and access relevant work. This paper presents an automated framework for research article classification and recommendation that leverages Natural Language Processing (NLP) techniques and machine learning. Using a large-scale arXiv.org dataset spanning more than three decades, we evaluate multiple feature extraction approaches (TF--IDF, Count Vectorizer, Sentence-BERT, USE, Mirror-BERT) in combination with diverse machine learning classifiers (Logistic Regression, SVM, Naïve Bayes, Random Forest, Gradient Boosted Trees, and k-Nearest Neighbour). Our experiments show that Logistic Regression with TF--IDF consistently yields the best classification performance, achieving an accuracy of 69\%. To complement classification, we incorporate a recommendation module based on the cosine similarity of vectorized articles, enabling efficient retrieval of related research papers. The proposed system directly addresses the challenge of information overload in digital libraries and demonstrates a scalable, data-driven solution to support literature discovery. 

---
# Scalable In-context Ranking with Generative Models 

**Authors**: Nilesh Gupta, Chong You, Srinadh Bhojanapalli, Sanjiv Kumar, Inderjit Dhillon, Felix Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.05396)  

**Abstract**: In-context Ranking (ICR) is an emerging paradigm for Information Retrieval (IR), which leverages contextual understanding of LLMs by directly incorporating the task description, candidate documents, and the query into the model's input prompt and tasking the LLM to identify relevant document(s). While it is effective, efficiency is a significant challenge in this paradigm, especially as the candidate list grows due to quadratic/super-linear scaling of attention operation with context length. To this end, this paper first identifies inherent and exploitable structures in the attention of LLMs finetuned for ICR: (1) inter-document block sparsity: attention is dense within each document block but sparse across different documents in the context; and (2) query-document block relevance: the attention scores from certain query tokens to a document block in middle layers strongly correlate with that document's actual relevance. Motivated by these observations, we introduce BlockRank (Blockwise In-context Ranking), a novel method that adapts the attention operation in an LLM by (a) architecturally enforcing the observed inter-document block sparsity, reducing attention complexity from quadratic to linear without loss in performance, and (b) optimizing query-document block relevance for true relevant documents during fine-tuning using an auxiliary contrastive training objective, improving retrieval in attention. Experiments on BEIR, MSMarco and NQ with Mistral-7B demonstrate that FLARE Mistral matches or outperforms existing SOTA listwise rankers and controlled fine-tuned baseline while being significantly more efficient at inference (4.7x for 100 MSMarco documents in context) and scaling gracefully to long-context shortlists, around 500 documents in-context (approximately 100K context length) within a second, presenting a scalable and effective solution for ICR. 

---
# Peeking inside the Black-Box: Reinforcement Learning for Explainable and Accurate Relation Extraction 

**Authors**: Xinyu Guo, Zhengliang Shi, Minglai Yang, Mahdi Rahimi, Mihai Surdeanu  

**Link**: [PDF](https://arxiv.org/pdf/2510.06198)  

**Abstract**: This paper introduces a framework for relation extraction (RE) that enhances both accuracy and explainability. The framework has two key components: (i) a reasoning mechanism that formulates relation extraction as a series of text-processing steps inspired by cognitive science, and (ii) an optimization process driven by reinforcement learning (RL) with a novel reward function designed to improve both task accuracy and explanation quality. We call our approach CogRE. Our framework addresses the lack of supervision for language-based explanations in traditional RE by promoting outputs that include important relation keywords. These keywords are drawn from a high-quality dictionary that is automatically constructed using an LLM. We evaluate our approach for the task of one-shot RE using two LLMs and two RE datasets. Our experiments show that CogRE improves explanation quality by addressing two common failure patterns in one-shot RE: poor attention focus and limited one-shot learning capability. For example, our cognitive-structured reasoning with Qwen2.5-15B-Instruct on One-shot NYT29 achieves 24.65% F1, surpassing prior reasoning-based designs. Optimizing this approach with RL using our reward further improves performance by +23.46% (absolute). Finally, human evaluation shows that our best model generates relational keywords closely aligned with gold labels, increasing human explanation quality ratings by 54% (relative). 

---
# Deterministic Legal Retrieval: An Action API for Querying the SAT-Graph RAG 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2510.06002)  

**Abstract**: The Structure-Aware Temporal Graph RAG (SAT-Graph RAG) addresses core limitations of standard Retrieval-Augmented Generation in the legal domain by providing a verifiable knowledge graph that models hierarchical structure, temporal evolution, and causal events of legal norms. However, a critical gap remains: how to reliably query this structured knowledge without sacrificing its deterministic properties. This paper introduces the SAT-Graph API, a formal query execution layer centered on canonical actions-atomic, composable, and auditable primitives that isolate probabilistic discovery from deterministic retrieval. These actions enable: (i) high-precision hybrid search; (ii) robust reference resolution; (iii) point-in-time version retrieval; and (iv) auditable causal tracing. We demonstrate how planner-guided agents can decompose complex queries into Directed Acyclic Graphs (DAGs) of these actions. This two-layer architecture transforms retrieval from an opaque black box to a transparent, auditable process, directly addressing Explainable AI (XAI) requirements for high-stakes domains. 

---
# KEO: Knowledge Extraction on OMIn via Knowledge Graphs and RAG for Safety-Critical Aviation Maintenance 

**Authors**: Kuangshi Ai, Jonathan A. Karr Jr, Meng Jiang, Nitesh V. Chawla, Chaoli Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.05524)  

**Abstract**: We present Knowledge Extraction on OMIn (KEO), a domain-specific knowledge extraction and reasoning framework with large language models (LLMs) in safety-critical contexts. Using the Operations and Maintenance Intelligence (OMIn) dataset, we construct a QA benchmark spanning global sensemaking and actionable maintenance tasks. KEO builds a structured Knowledge Graph (KG) and integrates it into a retrieval-augmented generation (RAG) pipeline, enabling more coherent, dataset-wide reasoning than traditional text-chunk RAG. We evaluate locally deployable LLMs (Gemma-3, Phi-4, Mistral-Nemo) and employ stronger models (GPT-4o, Llama-3.3) as judges. Experiments show that KEO markedly improves global sensemaking by revealing patterns and system-level insights, while text-chunk RAG remains effective for fine-grained procedural tasks requiring localized retrieval. These findings underscore the promise of KG-augmented LLMs for secure, domain-specific QA and their potential in high-stakes reasoning. 

---
# Towards Structured Knowledge: Advancing Triple Extraction from Regional Trade Agreements using Large Language Models 

**Authors**: Durgesh Nandini, Rebekka Koch, Mirco Schoenfeld  

**Link**: [PDF](https://arxiv.org/pdf/2510.05121)  

**Abstract**: This study investigates the effectiveness of Large Language Models (LLMs) for the extraction of structured knowledge in the form of Subject-Predicate-Object triples. We apply the setup for the domain of Economics application. The findings can be applied to a wide range of scenarios, including the creation of economic trade knowledge graphs from natural language legal trade agreement texts. As a use case, we apply the model to regional trade agreement texts to extract trade-related information triples. In particular, we explore the zero-shot, one-shot and few-shot prompting techniques, incorporating positive and negative examples, and evaluate their performance based on quantitative and qualitative metrics. Specifically, we used Llama 3.1 model to process the unstructured regional trade agreement texts and extract triples. We discuss key insights, challenges, and potential future directions, emphasizing the significance of language models in economic applications. 

---
