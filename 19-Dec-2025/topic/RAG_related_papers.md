# Leveraging Spreading Activation for Improved Document Retrieval in Knowledge-Graph-Based RAG Systems 

**Authors**: Jovan Pavlović, Miklós Krész, László Hajdu  

**Link**: [PDF](https://arxiv.org/pdf/2512.15922)  

**Abstract**: Despite initial successes and a variety of architectures, retrieval-augmented generation (RAG) systems still struggle to reliably retrieve and connect the multi-step evidence required for complicated reasoning tasks. Most of the standard RAG frameworks regard all retrieved information as equally reliable, overlooking the varying credibility and interconnected nature of large textual corpora. GraphRAG approaches offer potential improvement to RAG systems by integrating knowledge graphs, which structure information into nodes and edges, capture entity relationships, and enable multi-step logical traversal. However, GraphRAG is not always an ideal solution as it depends on high-quality graph representations of the corpus, which requires either pre-existing knowledge graphs that are expensive to build and update, or automated graph construction pipelines that are often unreliable. Moreover, systems following this paradigm typically use large language models to guide graph traversal and evidence retrieval, leading to challenges similar to those encountered with standard RAG. In this paper, we propose a novel RAG framework that employs the spreading activation algorithm to retrieve information from a corpus of documents interconnected by automatically constructed knowledge graphs, thereby enhancing the performance of large language models on complex tasks such as multi-hop question answering. Experiments show that our method achieves better or comparable performance to iterative RAG methodologies, while also being easily integrable as a plug-and-play module with a wide range of RAG-based approaches. Combining our method with chain-of-thought iterative retrieval yields up to a 39\% absolute gain in answer correctness compared to naive RAG, achieving these results with small open-weight language models and highlighting its effectiveness in resource-constrained settings. 

---
# From Facts to Conclusions : Integrating Deductive Reasoning in Retrieval-Augmented LLMs 

**Authors**: Shubham Mishra, Samyek Jain, Gorang Mehrishi, Shiv Tiwari, Harsh Sharma, Pratik Narang, Dhruv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2512.16795)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds large language models (LLMs) in external evidence, but fails when retrieved sources conflict or contain outdated or subjective information. Prior work address these issues independently but lack unified reasoning supervision. We propose a reasoning-trace-augmented RAG framework that adds structured, interpretable reasoning across three stages : (1) document-level adjudication, (2) conflict analysis, and (3) grounded synthesis, producing citation-linked answers or justified refusals. A Conflict-Aware Trust-Score (CATS) pipeline is introduced which evaluates groundedness, factual correctness, refusal accuracy, and conflict-behavior alignment using an LLM-as-a-Judge. Our 539-query reasoning dataset and evaluation pipeline establish a foundation for conflict-aware, interpretable RAG systems. Experimental results demonstrate substantial gains over baselines, most notably with Qwen, where Supervised Fine-Tuning improved End-to-End answer correctness from 0.069 to 0.883 and behavioral adherence from 0.074 to 0.722. 

---
# Introducing ORKG ASK: an AI-driven Scholarly Literature Search and Exploration System Taking a Neuro-Symbolic Approach 

**Authors**: Allard Oelen, Mohamad Yaser Jaradeh, Sören Auer  

**Link**: [PDF](https://arxiv.org/pdf/2512.16425)  

**Abstract**: As the volume of published scholarly literature continues to grow, finding relevant literature becomes increasingly difficult. With the rise of generative Artificial Intelligence (AI), and particularly Large Language Models (LLMs), new possibilities emerge to find and explore literature. We introduce ASK (Assistant for Scientific Knowledge), an AI-driven scholarly literature search and exploration system that follows a neuro-symbolic approach. ASK aims to provide active support to researchers in finding relevant scholarly literature by leveraging vector search, LLMs, and knowledge graphs. The system allows users to input research questions in natural language and retrieve relevant articles. ASK automatically extracts key information and generates answers to research questions using a Retrieval-Augmented Generation (RAG) approach. We present an evaluation of ASK, assessing the system's usability and usefulness. Findings indicate that the system is user-friendly and users are generally satisfied while using the system. 

---
# The Evolution of Reranking Models in Information Retrieval: From Heuristic Methods to Large Language Models 

**Authors**: Tejul Pandit, Sakshi Mahendru, Meet Raval, Dhvani Upadhyay  

**Link**: [PDF](https://arxiv.org/pdf/2512.16236)  

**Abstract**: Reranking is a critical stage in contemporary information retrieval (IR) systems, improving the relevance of the user-presented final results by honing initial candidate sets. This paper is a thorough guide to examine the changing reranker landscape and offer a clear view of the advancements made in reranking methods. We present a comprehensive survey of reranking models employed in IR, particularly within modern Retrieval Augmented Generation (RAG) pipelines, where retrieved documents notably influence output quality.
We embark on a chronological journey through the historical trajectory of reranking techniques, starting with foundational approaches, before exploring the wide range of sophisticated neural network architectures such as cross-encoders, sequence-generation models like T5, and Graph Neural Networks (GNNs) utilized for structural information. Recognizing the computational cost of advancing neural rerankers, we analyze techniques for enhancing efficiency, notably knowledge distillation for creating competitive, lighter alternatives. Furthermore, we map the emerging territory of integrating Large Language Models (LLMs) in reranking, examining novel prompting strategies and fine-tuning tactics. This survey seeks to elucidate the fundamental ideas, relative effectiveness, computational features, and real-world trade-offs of various reranking strategies. The survey provides a structured synthesis of the diverse reranking paradigms, highlighting their underlying principles and comparative strengths and weaknesses. 

---
# Exploration of Augmentation Strategies in Multi-modal Retrieval-Augmented Generation for the Biomedical Domain: A Case Study Evaluating Question Answering in Glycobiology 

**Authors**: Primož Kocbek, Azra Frkatović-Hodžić, Dora Lalić, Vivian Hui, Gordan Lauc, Gregor Štiglic  

**Link**: [PDF](https://arxiv.org/pdf/2512.16802)  

**Abstract**: Multi-modal retrieval-augmented generation (MM-RAG) promises grounded biomedical QA, but it is unclear when to (i) convert figures/tables into text versus (ii) use optical character recognition (OCR)-free visual retrieval that returns page images and leaves interpretation to the generator. We study this trade-off in glycobiology, a visually dense domain. We built a benchmark of 120 multiple-choice questions (MCQs) from 25 papers, stratified by retrieval difficulty (easy text, medium figures/tables, hard cross-evidence). We implemented four augmentations-None, Text RAG, Multi-modal conversion, and late-interaction visual retrieval (ColPali)-using Docling parsing and Qdrant indexing. We evaluated mid-size open-source and frontier proprietary models (e.g., Gemma-3-27B-IT, GPT-4o family). Additional testing used the GPT-5 family and multiple visual retrievers (ColPali/ColQwen/ColFlor). Accuracy with Agresti-Coull 95% confidence intervals (CIs) was computed over 5 runs per configuration. With Gemma-3-27B-IT, Text and Multi-modal augmentation outperformed OCR-free retrieval (0.722-0.740 vs. 0.510 average accuracy). With GPT-4o, Multi-modal achieved 0.808, with Text 0.782 and ColPali 0.745 close behind; within-model differences were small. In follow-on experiments with the GPT-5 family, the best results with ColPali and ColFlor improved by ~2% to 0.828 in both cases. In general, across the GPT-5 family, ColPali, ColQwen, and ColFlor were statistically indistinguishable. GPT-5-nano trailed larger GPT-5 variants by roughly 8-10%. Pipeline choice is capacity-dependent: converting visuals to text lowers the reader burden and is more reliable for mid-size models, whereas OCR-free visual retrieval becomes competitive under frontier models. Among retrievers, ColFlor offers parity with heavier options at a smaller footprint, making it an efficient default when strong generators are available. 

---
