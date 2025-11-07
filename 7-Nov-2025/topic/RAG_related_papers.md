# Caption Injection for Optimization in Generative Search Engine 

**Authors**: Xiaolu Chen, Yong Liao  

**Link**: [PDF](https://arxiv.org/pdf/2511.04080)  

**Abstract**: Generative Search Engines (GSEs) leverage Retrieval-Augmented Generation (RAG) techniques and Large Language Models (LLMs) to integrate multi-source information and provide users with accurate and comprehensive responses. Unlike traditional search engines that present results in ranked lists, GSEs shift users' attention from sequential browsing to content-driven subjective perception, driving a paradigm shift in information retrieval. In this context, enhancing the subjective visibility of content through Generative Search Engine Optimization (G-SEO) methods has emerged as a new research focus. With the rapid advancement of Multimodal Retrieval-Augmented Generation (MRAG) techniques, GSEs can now efficiently integrate text, images, audio, and video, producing richer responses that better satisfy complex information needs. Existing G-SEO methods, however, remain limited to text-based optimization and fail to fully exploit multimodal data. To address this gap, we propose Caption Injection, the first multimodal G-SEO approach, which extracts captions from images and injects them into textual content, integrating visual semantics to enhance the subjective visibility of content in generative search scenarios. We systematically evaluate Caption Injection on MRAMG, a benchmark for MRAG, under both unimodal and multimodal settings. Experimental results show that Caption Injection significantly outperforms text-only G-SEO baselines under the G-Eval metric, demonstrating the necessity and effectiveness of multimodal integration in G-SEO to improve user-perceived content visibility. 

---
# BanglaMedQA and BanglaMMedBench: Evaluating Retrieval-Augmented Generation Strategies for Bangla Biomedical Question Answering 

**Authors**: Sadia Sultana, Saiyma Sittul Muna, Mosammat Zannatul Samarukh, Ajwad Abrar, Tareque Mohmud Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2511.04560)  

**Abstract**: Developing accurate biomedical Question Answering (QA) systems in low-resource languages remains a major challenge, limiting equitable access to reliable medical knowledge. This paper introduces BanglaMedQA and BanglaMMedBench, the first large-scale Bangla biomedical Multiple Choice Question (MCQ) datasets designed to evaluate reasoning and retrieval in medical artificial intelligence (AI). The study applies and benchmarks several Retrieval-Augmented Generation (RAG) strategies, including Traditional, Zero-Shot Fallback, Agentic, Iterative Feedback, and Aggregate RAG, combining textbook-based and web retrieval with generative reasoning to improve factual accuracy. A key novelty lies in integrating a Bangla medical textbook corpus through Optical Character Recognition (OCR) and implementing an Agentic RAG pipeline that dynamically selects between retrieval and reasoning strategies. Experimental results show that the Agentic RAG achieved the highest accuracy 89.54% with openai/gpt-oss-120b, outperforming other configurations and demonstrating superior rationale quality. These findings highlight the potential of RAG-based methods to enhance the reliability and accessibility of Bangla medical QA, establishing a foundation for future research in multilingual medical artificial intelligence. 

---
# RAGalyst: Automated Human-Aligned Agentic Evaluation for Domain-Specific RAG 

**Authors**: Joshua Gao, Quoc Huy Pham, Subin Varghese, Silwal Saurav, Vedhus Hoskere  

**Link**: [PDF](https://arxiv.org/pdf/2511.04502)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a critical technique for grounding Large Language Models (LLMs) in factual evidence, yet evaluating RAG systems in specialized, safety-critical domains remains a significant challenge. Existing evaluation frameworks often rely on heuristic-based metrics that fail to capture domain-specific nuances and other works utilize LLM-as-a-Judge approaches that lack validated alignment with human judgment. This paper introduces RAGalyst, an automated, human-aligned agentic framework designed for the rigorous evaluation of domain-specific RAG systems. RAGalyst features an agentic pipeline that generates high-quality, synthetic question-answering (QA) datasets from source documents, incorporating an agentic filtering step to ensure data fidelity. The framework refines two key LLM-as-a-Judge metrics-Answer Correctness and Answerability-using prompt optimization to achieve a strong correlation with human annotations. Applying this framework to evaluate various RAG components across three distinct domains (military operations, cybersecurity, and bridge engineering), we find that performance is highly context-dependent. No single embedding model, LLM, or hyperparameter configuration proves universally optimal. Additionally, we provide an analysis on the most common low Answer Correctness reasons in RAG. These findings highlight the necessity of a systematic evaluation framework like RAGalyst, which empowers practitioners to uncover domain-specific trade-offs and make informed design choices for building reliable and effective RAG systems. RAGalyst is available on our Github. 

---
# Abductive Inference in Retrieval-Augmented Language Models: Generating and Validating Missing Premises 

**Authors**: Shiyin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2511.04020)  

**Abstract**: Large Language Models (LLMs) enhanced with retrieval -- commonly referred to as Retrieval-Augmented Generation (RAG) -- have demonstrated strong performance in knowledge-intensive tasks. However, RAG pipelines often fail when retrieved evidence is incomplete, leaving gaps in the reasoning process. In such cases, \emph{abductive inference} -- the process of generating plausible missing premises to explain observations -- offers a principled approach to bridge these gaps. In this paper, we propose a framework that integrates abductive inference into retrieval-augmented LLMs. Our method detects insufficient evidence, generates candidate missing premises, and validates them through consistency and plausibility checks. Experimental results on abductive reasoning and multi-hop QA benchmarks show that our approach improves both answer accuracy and reasoning faithfulness. This work highlights abductive inference as a promising direction for enhancing the robustness and explainability of RAG systems. 

---
