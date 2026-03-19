# Adaptive Guidance for Retrieval-Augmented Masked Diffusion Models 

**Authors**: Jaemin Kim, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.17677)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves factual grounding by incorporating external knowledge into language model generation. However, when retrieved context is noisy, unreliable, or inconsistent with the model's parametric knowledge, it introduces retrieval-prior conflicts that can degrade generation quality. While this problem has been studied in autoregressive language models, it remains largely unexplored in diffusion-based language models, where the iterative denoising process introduces unique challenges for integrating retrieved context. In this work, we propose Adaptive Retrieval-Augmented Masked Diffusion (ARAM), a training-free adaptive guidance framework for Masked Diffusion Models (MDMs) in RAG settings. ARAM dynamically calibrates the guidance scale during denoising according to the Signal-to-Noise Ratio (SNR) of the distributional shift induced by retrieved context. Intuitively, the model strengthens guidance when the retrieved context provides reliable corrective evidence and suppresses it when the contextual signal is noisy or non-supportive. Extensive experiments on multiple knowledge-intensive QA benchmarks show that ARAM improves overall QA performance over competitive RAG baselines. 

---
# PACE-RAG: Patient-Aware Contextual and Evidence-based Policy RAG for Clinical Drug Recommendation 

**Authors**: Chaeyoung Huh, Hyunmin Hwang, Jung Hwan Shin, Jinse Park, Jong Chul Ye  

**Link**: [PDF](https://arxiv.org/pdf/2603.17356)  

**Abstract**: Drug recommendation requires a deep understanding of individual patient context, especially for complex conditions like Parkinson's disease. While LLMs possess broad medical knowledge, they fail to capture the subtle nuances of actual prescribing patterns. Existing RAG methods also struggle with these complexities because guideline-based retrieval remains too generic and similar-patient retrieval often replicates majority patterns without accounting for the unique clinical nuances of individual patients. To bridge this gap, we propose PACE-RAG (Patient-Aware Contextual and Evidence-based Policy RAG), a novel framework designed to synthesize individual patient context with the prescribing tendencies of similar cases. By analyzing treatment patterns tailored to specific clinical signals, PACE-RAG identifies optimal prescriptions and generates an explainable clinical summary. Evaluated on a Parkinson's cohort and the MIMIC-IV benchmark using Llama-3.1-8B and Qwen3-8B, PACE-RAG achieved state-of-the-art performance, reaching F1 scores of 80.84% and 47.22%, respectively. These results validate PACE-RAG as a robust, clinically grounded solution for personalized decision support. Our code is available at: this https URL. 

---
# Enhancing Financial Report Question-Answering: A Retrieval-Augmented Generation System with Reranking Analysis 

**Authors**: Zhiyuan Cheng, Longying Lai, Yue Liu, Kai Cheng, Xiaoxi Qi  

**Link**: [PDF](https://arxiv.org/pdf/2603.16877)  

**Abstract**: Financial analysts face significant challenges extracting information from lengthy 10-K reports, which often exceed 100 pages. This paper presents a Retrieval-Augmented Generation (RAG) system designed to answer questions about S&P 500 financial reports and evaluates the impact of neural reranking on system performance. Our pipeline employs hybrid search combining full-text and semantic retrieval, followed by an optional reranking stage using a cross-encoder model. We conduct systematic evaluation using the FinDER benchmark dataset, comprising 1,500 queries across five experimental groups. Results demonstrate that reranking significantly improves answer quality, achieving 49.0 percent correctness for scores of 8 or above compared to 33.5 percent without reranking, representing a 15.5 percentage point improvement. Additionally, the error rate for completely incorrect answers decreases from 35.3 percent to 22.5 percent. Our findings emphasize the critical role of reranking in financial RAG systems and demonstrate performance improvements over baseline methods through modern language models and refined retrieval strategies. 

---
