# Evaluating Retrieval-Augmented Generation vs. Long-Context Input for Clinical Reasoning over EHRs 

**Authors**: Skatje Myers, Dmitriy Dligach, Timothy A. Miller, Samantha Barr, Yanjun Gao, Matthew Churpek, Anoop Mayampurath, Majid Afshar  

**Link**: [PDF](https://arxiv.org/pdf/2508.14817)  

**Abstract**: Electronic health records (EHRs) are long, noisy, and often redundant, posing a major challenge for the clinicians who must navigate them. Large language models (LLMs) offer a promising solution for extracting and reasoning over this unstructured text, but the length of clinical notes often exceeds even state-of-the-art models' extended context windows. Retrieval-augmented generation (RAG) offers an alternative by retrieving task-relevant passages from across the entire EHR, potentially reducing the amount of required input tokens. In this work, we propose three clinical tasks designed to be replicable across health systems with minimal effort: 1) extracting imaging procedures, 2) generating timelines of antibiotic use, and 3) identifying key diagnoses. Using EHRs from actual hospitalized patients, we test three state-of-the-art LLMs with varying amounts of provided context, using either targeted text retrieval or the most recent clinical notes. We find that RAG closely matches or exceeds the performance of using recent notes, and approaches the performance of using the models' full context while requiring drastically fewer input tokens. Our results suggest that RAG remains a competitive and efficient approach even as newer models become capable of handling increasingly longer amounts of text. 

---
# Retrieval-Augmented Generation in Industry: An Interview Study on Use Cases, Requirements, Challenges, and Evaluation 

**Authors**: Lorenz Brehme, Benedikt Dornauer, Thomas Ströhle, Maximilian Ehrhart, Ruth Breu  

**Link**: [PDF](https://arxiv.org/pdf/2508.14066)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a well-established and rapidly evolving field within AI that enhances the outputs of large language models by integrating relevant information retrieved from external knowledge sources. While industry adoption of RAG is now beginning, there is a significant lack of research on its practical application in industrial contexts. To address this gap, we conducted a semistructured interview study with 13 industry practitioners to explore the current state of RAG adoption in real-world settings. Our study investigates how companies apply RAG in practice, providing (1) an overview of industry use cases, (2) a consolidated list of system requirements, (3) key challenges and lessons learned from practical experiences, and (4) an analysis of current industry evaluation methods. Our main findings show that current RAG applications are mostly limited to domain-specific QA tasks, with systems still in prototype stages; industry requirements focus primarily on data protection, security, and quality, while issues such as ethics, bias, and scalability receive less attention; data preprocessing remains a key challenge, and system evaluation is predominantly conducted by humans rather than automated methods. 

---
# An automatic patent literature retrieval system based on LLM-RAG 

**Authors**: Yao Ding, Yuqing Wu, Ziyang Ding  

**Link**: [PDF](https://arxiv.org/pdf/2508.14064)  

**Abstract**: With the acceleration of technological innovation efficient retrieval and classification of patent literature have become essential for intellectual property management and enterprise RD Traditional keyword and rulebased retrieval methods often fail to address complex query intents or capture semantic associations across technical domains resulting in incomplete and lowrelevance results This study presents an automated patent retrieval framework integrating Large Language Models LLMs with RetrievalAugmented Generation RAG technology The system comprises three components: 1) a preprocessing module for patent data standardization, 2) a highefficiency vector retrieval engine leveraging LLMgenerated embeddings, and 3) a RAGenhanced query module that combines external document retrieval with contextaware response generation Evaluations were conducted on the Google Patents dataset 20062024 containing millions of global patent records with metadata such as filing date domain and status The proposed gpt35turbo0125RAG configuration achieved 805 semantic matching accuracy and 92.1% recall surpassing baseline LLM methods by 28 percentage points The framework also demonstrated strong generalization in crossdomain classification and semantic clustering tasks These results validate the effectiveness of LLMRAG integration for intelligent patent retrieval providing a foundation for nextgeneration AIdriven intellectual property analysis platforms 

---
# RAG-Boost: Retrieval-Augmented Generation Enhanced LLM-based Speech Recognition 

**Authors**: Pengcheng Wang, Sheng Li, Takahiro Shinozaki  

**Link**: [PDF](https://arxiv.org/pdf/2508.14048)  

**Abstract**: In this paper, we propose RAG-Boost (ST-ShinozakiLab Task I system), which enhances the baseline LLM-based ASR system of the MLC-SLM Challenge (task I) with a retrieval-augmented generation (RAG) module on the fly. Each partial ASR hypothesis queries a vector store of audio-text pairs and domain terms, and the retrieved results are fused with the live ASR hypotheses to fix recognition errors. The fused hypotheses are passed to the LLM, yielding improved responses. 

---
