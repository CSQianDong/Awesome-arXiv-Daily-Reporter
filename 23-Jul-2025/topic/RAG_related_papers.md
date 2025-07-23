# Agentic RAG with Knowledge Graphs for Complex Multi-Hop Reasoning in Real-World Applications 

**Authors**: Jean Lelong, Adnane Errazine, Annabelle Blangero  

**Link**: [PDF](https://arxiv.org/pdf/2507.16507)  

**Abstract**: Conventional Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) but often fall short on complex queries, delivering limited, extractive answers and struggling with multiple targeted retrievals or navigating intricate entity relationships. This is a critical gap in knowledge-intensive domains. We introduce INRAExplorer, an agentic RAG system for exploring the scientific data of INRAE (France's National Research Institute for Agriculture, Food and Environment). INRAExplorer employs an LLM-based agent with a multi-tool architecture to dynamically engage a rich knowledge base, through a comprehensive knowledge graph derived from open access INRAE publications. This design empowers INRAExplorer to conduct iterative, targeted queries, retrieve exhaustive datasets (e.g., all publications by an author), perform multi-hop reasoning, and deliver structured, comprehensive answers. INRAExplorer serves as a concrete illustration of enhancing knowledge interaction in specialized fields. 

---
# Never Come Up Empty: Adaptive HyDE Retrieval for Improving LLM Developer Support 

**Authors**: Fangjian Lei, Mariam El Mezouar, Shayan Noei, Ying Zou  

**Link**: [PDF](https://arxiv.org/pdf/2507.16754)  

**Abstract**: Large Language Models (LLMs) have shown promise in assisting developers with code-related questions; however, LLMs carry the risk of generating unreliable answers. To address this, Retrieval-Augmented Generation (RAG) has been proposed to reduce the unreliability (i.e., hallucinations) of LLMs. However, designing effective pipelines remains challenging due to numerous design choices. In this paper, we construct a retrieval corpus of over 3 million Java and Python related Stack Overflow posts with accepted answers, and explore various RAG pipeline designs to answer developer questions, evaluating their effectiveness in generating accurate and reliable responses. More specifically, we (1) design and evaluate 7 different RAG pipelines and 63 pipeline variants to answer questions that have historically similar matches, and (2) address new questions without any close prior matches by automatically lowering the similarity threshold during retrieval, thereby increasing the chance of finding partially relevant context and improving coverage for unseen cases. We find that implementing a RAG pipeline combining hypothetical-documentation-embedding (HyDE) with the full-answer context performs best in retrieving and answering similarcontent for Stack Overflow questions. Finally, we apply our optimal RAG pipeline to 4 open-source LLMs and compare the results to their zero-shot performance. Our findings show that RAG with our optimal RAG pipeline consistently outperforms zero-shot baselines across models, achieving higher scores for helpfulness, correctness, and detail with LLM-as-a-judge. These findings demonstrate that our optimal RAG pipelines robustly enhance answer quality for a wide range of developer queries including both previously seen and novel questions across different LLMs 

---
# Advancing Risk and Quality Assurance: A RAG Chatbot for Improved Regulatory Compliance 

**Authors**: Lars Hillebrand, Armin Berger, Daniel Uedelhoven, David Berghaus, Ulrich Warning, Tim Dilmaghani, Bernd Kliem, Thomas Schmid, Rüdiger Loitz, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2507.16711)  

**Abstract**: Risk and Quality (R&Q) assurance in highly regulated industries requires constant navigation of complex regulatory frameworks, with employees handling numerous daily queries demanding accurate policy interpretation. Traditional methods relying on specialized experts create operational bottlenecks and limit scalability. We present a novel Retrieval Augmented Generation (RAG) system leveraging Large Language Models (LLMs), hybrid search and relevance boosting to enhance R&Q query processing. Evaluated on 124 expert-annotated real-world queries, our actively deployed system demonstrates substantial improvements over traditional RAG approaches. Additionally, we perform an extensive hyperparameter analysis to compare and evaluate multiple configuration setups, delivering valuable insights to practitioners. 

---
# eSapiens's DEREK Module: Deep Extraction & Reasoning Engine for Knowledge with LLMs 

**Authors**: Isaac Shi, Zeyuan Li, Fan Liu, Wenli Wang, Lewei He, Yang Yang, Tianyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.15863)  

**Abstract**: We present the DEREK (Deep Extraction & Reasoning Engine for Knowledge) Module, a secure and scalable Retrieval-Augmented Generation pipeline designed specifically for enterprise document question answering. Designed and implemented by eSapiens, the system ingests heterogeneous content (PDF, Office, web), splits it into 1,000-token overlapping chunks, and indexes them in a hybrid HNSW+BM25 store. User queries are refined by GPT-4o, retrieved via combined vector+BM25 search, reranked with Cohere, and answered by an LLM using CO-STAR prompt engineering. A LangGraph verifier enforces citation overlap, regenerating answers until every claim is grounded. On four LegalBench subsets, 1000-token chunks improve Recall@50 by approximately 1 pp and hybrid+rerank boosts Precision@10 by approximately 7 pp; the verifier raises TRACe Utilization above 0.50 and limits unsupported statements to less than 3%. All components run in containers, enforce end-to-end TLS 1.3 and AES-256. These results demonstrate that the DEREK module delivers accurate, traceable, and production-ready document QA with minimal operational overhead. The module is designed to meet enterprise demands for secure, auditable, and context-faithful retrieval, providing a reliable baseline for high-stakes domains such as legal and finance. 

---
# mRAKL: Multilingual Retrieval-Augmented Knowledge Graph Construction for Low-Resourced Languages 

**Authors**: Hellina Hailu Nigatu, Min Li, Maartje ter Hoeve, Saloni Potdar, Sarah Chasins  

**Link**: [PDF](https://arxiv.org/pdf/2507.16011)  

**Abstract**: Knowledge Graphs represent real-world entities and the relationships between them. Multilingual Knowledge Graph Construction (mKGC) refers to the task of automatically constructing or predicting missing entities and links for knowledge graphs in a multilingual setting. In this work, we reformulate the mKGC task as a Question Answering (QA) task and introduce mRAKL: a Retrieval-Augmented Generation (RAG) based system to perform mKGC. We achieve this by using the head entity and linking relation in a question, and having our model predict the tail entity as an answer. Our experiments focus primarily on two low-resourced languages: Tigrinya and Amharic. We experiment with using higher-resourced languages Arabic and English for cross-lingual transfer. With a BM25 retriever, we find that the RAG-based approach improves performance over a no-context setting. Further, our ablation studies show that with an idealized retrieval system, mRAKL improves accuracy by 4.92 and 8.79 percentage points for Tigrinya and Amharic, respectively. 

---
# Step-Audio 2 Technical Report 

**Authors**: Boyong Wu, Chao Yan, Chen Hu, Cheng Yi, Chengli Feng, Fei Tian, Feiyu Shen, Gang Yu, Haoyang Zhang, Jingbei Li, Mingrui Chen, Peng Liu, Wang You, Xiangyu Tony Zhang, Xingyuan Li, Xuerui Yang, Yayue Deng, Yechang Huang, Yuxin Li, Yuxin Zhang, Zhao You, Brian Li, Changyi Wan, Hanpeng Hu, Jiangjie Zhen, Siyu Chen, Song Yuan, Xuelin Zhang, Yimin Jiang, Yu Zhou, Yuxiang Yang, Bingxin Li, Buyun Ma, Changhe Song, Dongqing Pang, Guoqiang Hu, Haiyang Sun, Kang An, Na Wang, Shuli Gao, Wei Ji, Wen Li, Wen Sun, Xuan Wen, Yong Ren, Yuankai Ma, Yufan Lu, Bin Wang, Bo Li, Changxin Miao, Che Liu, Chen Xu, Dapeng Shi, Dingyuan Hu, Donghang Wu, Enle Liu, Guanzhe Huang, Gulin Yan, Han Zhang, Hao Nie, Haonan Jia, Hongyu Zhou, Jianjian Sun, Jiaoren Wu, Jie Wu, Jie Yang, Jin Yang, Junzhe Lin, Kaixiang Li, Lei Yang, Liying Shi, Li Zhou, Longlong Gu, Ming Li, Mingliang Li, Mingxiao Li, Nan Wu, Qi Han, Qinyuan Tan, Shaoliang Pang, Shengjie Fan, Siqi Liu, Tiancheng Cao, Wanying Lu, Wenqing He, Wuxun Xie, Xu Zhao, Xueqi Li, Yanbo Yu, Yang Yang, Yi Liu, Yifan Lu, Yilei Wang, Yuanhao Ding, Yuanwei Liang, Yuanwei Lu, Yuchu Luo, Yuhe Yin, Yumeng Zhan, Yuxiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.16632)  

**Abstract**: This paper presents Step-Audio~2, an end-to-end multi-modal large language model designed for industry-strength audio understanding and speech conversation. By integrating a latent audio encoder and reasoning-centric reinforcement learning (RL), Step-Audio 2 achieves promising performance in automatic speech recognition (ASR) and audio understanding. To facilitate genuine end-to-end speech conversation, Step-Audio 2 incorporates the generation of discrete audio tokens into language modeling, significantly enhancing its responsiveness to paralinguistic information such as speaking styles and emotions. To effectively leverage the rich textual and acoustic knowledge in real-world data, Step-Audio 2 integrates retrieval-augmented generation (RAG) and is able to call external tools such as web search to mitigate hallucination and audio search to switch timbres. Trained on millions of hours of speech and audio data, Step-Audio 2 delivers intelligence and expressiveness across diverse conversational scenarios. Evaluation results demonstrate that Step-Audio 2 achieves state-of-the-art performance on various audio understanding and conversational benchmarks compared to other open-source and commercial solutions. Please visit this https URL for more information. 

---
