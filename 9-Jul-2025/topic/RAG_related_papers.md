# SARA: Selective and Adaptive Retrieval-augmented Generation with Context Compression 

**Authors**: Yiqiao Jin, Kartik Sharma, Vineeth Rakesh, Yingtong Dou, Menghai Pan, Mahashweta Das, Srijan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.05633)  

**Abstract**: Retrieval-augmented Generation (RAG) extends large language models (LLMs) with external knowledge but faces key challenges: restricted effective context length and redundancy in retrieved documents. Pure compression-based approaches reduce input size but often discard fine-grained details essential for factual accuracy. We propose SARA, a unified RAG framework that balances local precision and global knowledge coverage under tight context budgets. SARA combines natural-language text snippets with semantic compression vectors to jointly enhance context efficiency and answer correctness. It represents contexts at two complementary levels: 1) fine-grained natural-language spans that preserve critical entities and numerical values, and 2) compact, interpretable vectors that summarize high-level semantics. An iterative evidence-selection module employs the compression vectors for dynamic reranking of contexts. Across 9 datasets and 5 open-source LLMs spanning 3 model families (Mistral, Llama, and Gemma), SARA consistently improves answer relevance (+17.71), answer correctness (+13.72), and semantic similarity (+15.53), demonstrating the importance of integrating textual and compressed representations for robust, context-efficient RAG. 

---
# Beyond classical and contemporary models: a transformative ai framework for student dropout prediction in distance learning using rag, prompt engineering, and cross-modal fusion 

**Authors**: Miloud Mihoubi, Meriem Zerkouk, Belkacem Chikhaoui  

**Link**: [PDF](https://arxiv.org/pdf/2507.05285)  

**Abstract**: Student dropout in distance learning remains a critical challenge, with profound societal and economic consequences. While classical machine learning models leverage structured socio-demographic and behavioral data, they often fail to capture the nuanced emotional and contextual factors embedded in unstructured student interactions. This paper introduces a transformative AI framework that redefines dropout prediction through three synergistic innovations: Retrieval-Augmented Generation (RAG) for domain-specific sentiment analysis, prompt engineering to decode academic stressors, and cross-modal attention fusion to dynamically align textual, behavioral, and socio-demographic insights. By grounding sentiment analysis in a curated knowledge base of pedagogical content, our RAG-enhanced BERT model interprets student comments with unprecedented contextual relevance, while optimized prompts isolate indicators of academic distress (e.g., "isolation," "workload anxiety"). A cross-modal attention layer then fuses these insights with temporal engagement patterns, creating holistic risk profiles. Evaluated on a longitudinal dataset of 4 423 students, the framework achieves 89% accuracy and an F1-score of 0.88, outperforming conventional models by 7% and reducing false negatives by 21%. Beyond prediction, the system generates interpretable interventions by retrieving contextually aligned strategies (e.g., mentorship programs for isolated learners). This work bridges the gap between predictive analytics and actionable pedagogy, offering a scalable solution to mitigate dropout risks in global education systems 

---
# KERAG_R: Knowledge-Enhanced Retrieval-Augmented Generation for Recommendation 

**Authors**: Zeyuan Meng, Zixuan Yi, Iadh Ounis  

**Link**: [PDF](https://arxiv.org/pdf/2507.05863)  

**Abstract**: Large Language Models (LLMs) have shown strong potential in recommender systems due to their contextual learning and generalisation capabilities. Existing LLM-based recommendation approaches typically formulate the recommendation task using specialised prompts designed to leverage their contextual abilities, and aligning their outputs closely with human preferences to yield an improved recommendation performance. However, the use of LLMs for recommendation tasks is limited by the absence of domain-specific knowledge. This lack of relevant relational knowledge about the items to be recommended in the LLM's pre-training corpus can lead to inaccuracies or hallucinations, resulting in incorrect or misleading recommendations. Moreover, directly using information from the knowledge graph introduces redundant and noisy information, which can affect the LLM's reasoning process or exceed its input context length, thereby reducing the performance of LLM-based recommendations. To address the lack of domain-specific knowledge, we propose a novel model called Knowledge-Enhanced Retrieval-Augmented Generation for Recommendation (KERAG_R). Specifically, we leverage a graph retrieval-augmented generation (GraphRAG) component to integrate additional information from a knowledge graph (KG) into instructions, enabling the LLM to collaboratively exploit recommendation signals from both text-based user interactions and the knowledge graph to better estimate the users' preferences in a recommendation context. In particular, we perform graph RAG by pre-training a graph attention network (GAT) to select the most relevant triple for the target users for the used LLM, thereby enhancing the LLM while reducing redundant and noisy information. Our extensive experiments on three public datasets show that our proposed KERAG_R model significantly outperforms ten existing state-of-the-art recommendation methods. 

---
# DRAGON: Dynamic RAG Benchmark On News 

**Authors**: Fedor Chernogorskii, Sergei Averkiev, Liliya Kudraleeva, Zaven Martirosian, Maria Tikhonova, Valentin Malykh, Alena Fenogenova  

**Link**: [PDF](https://arxiv.org/pdf/2507.05713)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a widely adopted approach for improving the factuality of large language models (LLMs) by incorporating external knowledge at inference time. Although there exist multiple RAG benchmarks for English, evaluation resources for other languages, including Russian, remain scarce and static, failing to capture the dynamic nature of real-world deployments.
In this work, we present DRAGON (Dynamic RAG Benchmark On News), the first dynamic benchmark for evaluating RAG systems in Russian on a changing news corpora. DRAGON is built upon a regularly updated corpus of Russian news and public documents and supports comprehensive evaluation of both the retriever and generator components. Question generation is performed automatically with the use of Knowledge Graph constructed from the corpus and enables the extraction of four core question types aligned with distinct subgraph patterns. We release a complete evaluation framework comprising the pipeline for automatic question generation, evaluation scripts, which are potentially reusable for other languages and multilingual settings, and benchmark data. We also launch a public leaderboard to encourage community participation and comparison. 

---
# HIRAG: Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation 

**Authors**: YiHan Jiao, ZheHao Tan, Dan Yang, DuoLin Sun, Jie Feng, Jian Wang, Peng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2507.05714)  

**Abstract**: Retrieval-augmented generation (RAG) has become a fundamental paradigm for addressing the challenges faced by large language models in handling real-time information and domain-specific problems. Traditional RAG systems primarily rely on the in-context learning (ICL) capabilities of the large language model itself. Still, in-depth research on the specific capabilities needed by the RAG generation model is lacking, leading to challenges with inconsistent document quality and retrieval system imperfections. Even the limited studies that fine-tune RAG generative models often \textit{lack a granular focus on RAG task} or \textit{a deeper utilization of chain-of-thought processes}. To address this, we propose that RAG models should possess three progressively hierarchical abilities (1) Filtering: the ability to select relevant information; (2) Combination: the ability to combine semantic information across paragraphs; and (3) RAG-specific reasoning: the ability to further process external knowledge using internal knowledge. Thus, we introduce our new RAG instruction fine-tuning method, Hierarchical-Thought Instruction-Tuning Retrieval-Augmented Generation (HIRAG) incorporates a "think before answering" strategy. This method enhances the model's open-book examination capability by utilizing multi-level progressive chain-of-thought. Experiments show that the HIRAG training strategy significantly improves the model's performance on datasets such as RGB, PopQA, MuSiQue, HotpotQA, and PubmedQA. 

---
# ReservoirChat: Interactive Documentation Enhanced with LLM and Knowledge Graph for ReservoirPy 

**Authors**: Virgile Boraud, Yannis Bendi-Ouis, Paul Bernard, Xavier Hinaut  

**Link**: [PDF](https://arxiv.org/pdf/2507.05279)  

**Abstract**: We introduce a tool designed to improve the capabilities of Large Language Models (LLMs) in assisting with code development using the ReservoirPy library, as well as in answering complex questions in the field of Reservoir Computing. By incorporating external knowledge through Retrieval-Augmented Generation (RAG) and knowledge graphs, our approach aims to reduce hallucinations and increase the factual accuracy of generated responses. The system provides an interactive experience similar to ChatGPT, tailored specifically for ReservoirPy, enabling users to write, debug, and understand Python code while accessing reliable domain-specific insights. In our evaluation, while proprietary models such as ChatGPT-4o and NotebookLM performed slightly better on general knowledge questions, our model outperformed them on coding tasks and showed a significant improvement over its base model, Codestral-22B. 

---
# Flippi: End To End GenAI Assistant for E-Commerce 

**Authors**: Anand A. Rajasekar, Praveen Tangarajan, Anjali Nainani, Amogh Batwal, Vinay Rao Dandin, Anusua Trivedi, Ozan Ersoy  

**Link**: [PDF](https://arxiv.org/pdf/2507.05788)  

**Abstract**: The emergence of conversational assistants has fundamentally reshaped user interactions with digital platforms. This paper introduces Flippi-a cutting-edge, end-to-end conversational assistant powered by large language models (LLMs) and tailored for the e-commerce sector. Flippi addresses the challenges posed by the vast and often overwhelming product landscape, enabling customers to discover products more efficiently through natural language dialogue. By accommodating both objective and subjective user requirements, Flippi delivers a personalized shopping experience that surpasses traditional search methods. This paper details how Flippi interprets customer queries to provide precise product information, leveraging advanced NLP techniques such as Query Reformulation, Intent Detection, Retrieval-Augmented Generation (RAG), Named Entity Recognition (NER), and Context Reduction. Flippi's unique capability to identify and present the most attractive offers on an e-commerce site is also explored, demonstrating how it empowers users to make cost-effective decisions. Additionally, the paper discusses Flippi's comparative analysis features, which help users make informed choices by contrasting product features, prices, and other relevant attributes. The system's robust architecture is outlined, emphasizing its adaptability for integration across various e-commerce platforms and the technological choices underpinning its performance and accuracy. Finally, a comprehensive evaluation framework is presented, covering performance metrics, user satisfaction, and the impact on customer engagement and conversion rates. By bridging the convenience of online shopping with the personalized assistance traditionally found in physical stores, Flippi sets a new standard for customer satisfaction and engagement in the digital marketplace. 

---
# Beyond Retrieval: Ensembling Cross-Encoders and GPT Rerankers with LLMs for Biomedical QA 

**Authors**: Shashank Verma, Fengyi Jiang, Xiangning Xue  

**Link**: [PDF](https://arxiv.org/pdf/2507.05577)  

**Abstract**: Biomedical semantic question answering rooted in information retrieval can play a crucial role in keeping up to date with vast, rapidly evolving and ever-growing biomedical literature. A robust system can help researchers, healthcare professionals and even layman users access relevant knowledge grounded in evidence. The BioASQ 2025 Task13b Challenge serves as an important benchmark, offering a competitive platform for advancement of this space. This paper presents the methodologies and results from our participation in this challenge where we built a Retrieval-Augmented Generation (RAG) system that can answer biomedical questions by retrieving relevant PubMed documents and snippets to generate answers. For the retrieval task, we generated dense embeddings from biomedical articles for initial retrieval, and applied an ensemble of finetuned cross-encoders and large language models (LLMs) for re-ranking to identify top relevant documents. Our solution achieved an MAP@10 of 0.1581, placing 10th on the leaderboard for the retrieval task. For answer generation, we employed few-shot prompting of instruction-tuned LLMs. Our system achieved macro-F1 score of 0.95 for yes/no questions (rank 12), Mean Reciprocal Rank (MRR) of 0.64 for factoid questions (rank 1), mean-F1 score of 0.63 for list questions (rank 5), and ROUGE-SU4 F1 score of 0.29 for ideal answers (rank 11). 

---
