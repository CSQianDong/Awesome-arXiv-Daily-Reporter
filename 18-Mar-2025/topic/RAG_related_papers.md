# LLM-Match: An Open-Sourced Patient Matching Model Based on Large Language Models and Retrieval-Augmented Generation 

**Authors**: Xiaodi Li, Shaika Chowdhury, Chung Il Wi, Maria Vassilaki, Ken Liu, Terence T Sio, Owen Garrick, Young J Juhn, James R Cerhan, Cui Tao, Nansu Zong  

**Link**: [PDF](https://arxiv.org/pdf/2503.13281)  

**Abstract**: Patient matching is the process of linking patients to appropriate clinical trials by accurately identifying and matching their medical records with trial eligibility criteria. We propose LLM-Match, a novel framework for patient matching leveraging fine-tuned open-source large language models. Our approach consists of four key components. First, a retrieval-augmented generation (RAG) module extracts relevant patient context from a vast pool of electronic health records (EHRs). Second, a prompt generation module constructs input prompts by integrating trial eligibility criteria (both inclusion and exclusion criteria), patient context, and system instructions. Third, a fine-tuning module with a classification head optimizes the model parameters using structured prompts and ground-truth labels. Fourth, an evaluation module assesses the fine-tuned model's performance on the testing datasets. We evaluated LLM-Match on four open datasets, n2c2, SIGIR, TREC 2021, and TREC 2022, using open-source models, comparing it against TrialGPT, Zero-Shot, and GPT-4-based closed models. LLM-Match outperformed all baselines. 

---
# A Survey on Transformer Context Extension: Approaches and Evaluation 

**Authors**: Yijun Liu, Jinzheng Yu, Yang Xu, Zhongyang Li, Qingfu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13299)  

**Abstract**: Large language models (LLMs) based on Transformer have been widely applied in the filed of natural language processing (NLP), demonstrating strong performance, particularly in handling short text tasks. However, when it comes to long context scenarios, the performance of LLMs degrades due to some challenges. To alleviate this phenomenon, there is a number of work proposed recently. In this survey, we first list the challenges of applying pre-trained LLMs to process long contexts. Then systematically review the approaches related to long context and propose our taxonomy categorizing them into four main types: positional encoding, context compression, retrieval augmented, and attention pattern. In addition to the approaches, we focus on the evaluation of long context, organizing relevant data, tasks, and metrics based on existing long context benchmarks. Finally, we summarize unresolved issues in the long context domain and put forward our views on future developments. 

---
# RAG-RL: Advancing Retrieval-Augmented Generation via RL and Curriculum Learning 

**Authors**: Jerry Huang, Siddarth Madala, Risham Sidhu, Cheng Niu, Julia Hockenmaier, Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12759)  

**Abstract**: Recent research highlights the challenges retrieval models face in retrieving useful contexts and the limitations of generation models in effectively utilizing those contexts in retrieval-augmented generation (RAG) settings. To address these challenges, we introduce RAG-RL, the first reasoning language model (RLM) specifically trained for RAG. RAG-RL demonstrates that stronger answer generation models can identify relevant contexts within larger sets of retrieved information -- thereby alleviating the burden on retrievers -- while also being able to utilize those contexts more effectively. Moreover, we show that curriculum design in the reinforcement learning (RL) post-training process is a powerful approach to enhancing model performance. We benchmark our method on two open-domain question-answering datasets and achieve state-of-the-art results, surpassing previous SOTA generative reader models. In addition, we offers empirical insights into various curriculum learning strategies, providing a deeper understanding of their impact on model performance. 

---
# Integrating Chain-of-Thought and Retrieval Augmented Generation Enhances Rare Disease Diagnosis from Clinical Notes 

**Authors**: Da Wu, Zhanliang Wang, Quan Nguyen, Kai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.12286)  

**Abstract**: Background: Several studies show that large language models (LLMs) struggle with phenotype-driven gene prioritization for rare diseases. These studies typically use Human Phenotype Ontology (HPO) terms to prompt foundation models like GPT and LLaMA to predict candidate genes. However, in real-world settings, foundation models are not optimized for domain-specific tasks like clinical diagnosis, yet inputs are unstructured clinical notes rather than standardized terms. How LLMs can be instructed to predict candidate genes or disease diagnosis from unstructured clinical notes remains a major challenge. Methods: We introduce RAG-driven CoT and CoT-driven RAG, two methods that combine Chain-of-Thought (CoT) and Retrieval Augmented Generation (RAG) to analyze clinical notes. A five-question CoT protocol mimics expert reasoning, while RAG retrieves data from sources like HPO and OMIM (Online Mendelian Inheritance in Man). We evaluated these approaches on rare disease datasets, including 5,980 Phenopacket-derived notes, 255 literature-based narratives, and 220 in-house clinical notes from Childrens Hospital of Philadelphia. Results: We found that recent foundations models, including Llama 3.3-70B-Instruct and DeepSeek-R1-Distill-Llama-70B, outperformed earlier versions such as Llama 2 and GPT-3.5. We also showed that RAG-driven CoT and CoT-driven RAG both outperform foundation models in candidate gene prioritization from clinical notes; in particular, both methods with DeepSeek backbone resulted in a top-10 gene accuracy of over 40% on Phenopacket-derived clinical notes. RAG-driven CoT works better for high-quality notes, where early retrieval can anchor the subsequent reasoning steps in domain-specific evidence, while CoT-driven RAG has advantage when processing lengthy and noisy notes. 

---
# Logic-RAG: Augmenting Large Multimodal Models with Visual-Spatial Knowledge for Road Scene Understanding 

**Authors**: Imran Kabir, Md Alimoor Reza, Syed Billah  

**Link**: [PDF](https://arxiv.org/pdf/2503.12663)  

**Abstract**: Large multimodal models (LMMs) are increasingly integrated into autonomous driving systems for user interaction. However, their limitations in fine-grained spatial reasoning pose challenges for system interpretability and user trust. We introduce Logic-RAG, a novel Retrieval-Augmented Generation (RAG) framework that improves LMMs' spatial understanding in driving scenarios. Logic-RAG constructs a dynamic knowledge base (KB) about object-object relationships in first-order logic (FOL) using a perception module, a query-to-logic embedder, and a logical inference engine. We evaluated Logic-RAG on visual-spatial queries using both synthetic and real-world driving videos. When using popular LMMs (GPT-4V, Claude 3.5) as proxies for an autonomous driving system, these models achieved only 55% accuracy on synthetic driving scenes and under 75% on real-world driving scenes. Augmenting them with Logic-RAG increased their accuracies to over 80% and 90%, respectively. An ablation study showed that even without logical inference, the fact-based context constructed by Logic-RAG alone improved accuracy by 15%. Logic-RAG is extensible: it allows seamless replacement of individual components with improved versions and enables domain experts to compose new knowledge in both FOL and natural language. In sum, Logic-RAG addresses critical spatial reasoning deficiencies in LMMs for autonomous driving applications. Code and data are available at this https URL. 

---
# Generative AI for Software Architecture. Applications, Trends, Challenges, and Future Directions 

**Authors**: Matteo Esposito, Xiaozhou Li, Sergio Moreschini, Noman Ahmad, Tomas Cerny, Karthik Vaidhyanathan, Valentina Lenarduzzi, Davide Taibi  

**Link**: [PDF](https://arxiv.org/pdf/2503.13310)  

**Abstract**: Context: Generative Artificial Intelligence (GenAI) is transforming much of software development, yet its application in software architecture is still in its infancy, and no prior study has systematically addressed the topic. Aim: We aim to systematically synthesize the use, rationale, contexts, usability, and future challenges of GenAI in software architecture. Method: We performed a multivocal literature review (MLR), analyzing peer-reviewed and gray literature, identifying current practices, models, adoption contexts, and reported challenges, extracting themes via open coding. Results: Our review identified significant adoption of GenAI for architectural decision support and architectural reconstruction. OpenAI GPT models are predominantly applied, and there is consistent use of techniques such as few-shot prompting and retrieved-augmented generation (RAG). GenAI has been applied mostly to initial stages of the Software Development Life Cycle (SDLC), such as Requirements-to-Architecture and Architecture-to-Code. Monolithic and microservice architectures were the dominant targets. However, rigorous testing of GenAI outputs was typically missing from the studies. Among the most frequent challenges are model precision, hallucinations, ethical aspects, privacy issues, lack of architecture-specific datasets, and the absence of sound evaluation frameworks. Conclusions: GenAI shows significant potential in software design, but several challenges remain on its path to greater adoption. Research efforts should target designing general evaluation methodologies, handling ethics and precision, increasing transparency and explainability, and promoting architecture-specific datasets and benchmarks to bridge the gap between theoretical possibilities and practical use. 

---
# Agentic Search Engine for Real-Time IoT Data 

**Authors**: Abdelrahman Elewah, Khalid Elgazzar  

**Link**: [PDF](https://arxiv.org/pdf/2503.12255)  

**Abstract**: The Internet of Things (IoT) has enabled diverse devices to communicate over the Internet, yet the fragmentation of IoT systems limits seamless data sharing and coordinated management. We have recently introduced SensorsConnect, a unified framework to enable seamless content and sensor data sharing in collaborative IoT systems, inspired by how the World Wide Web (WWW) enabled a shared and accessible space for information among humans. This paper presents the IoT Agentic Search Engine (IoT-ASE), a real-time search engine tailored for IoT environments. IoT-ASE leverages Large Language Models (LLMs) and Retrieval Augmented Generation (RAG) techniques to address the challenge of searching vast, real-time IoT data, enabling it to handle complex queries and deliver accurate, contextually relevant results. We implemented a use-case scenario in Toronto to demonstrate how IoT-ASE can improve service quality recommendations by leveraging real-time IoT data. Our evaluation shows that IoT-ASE achieves a 92\% accuracy in retrieving intent-based services and produces responses that are concise, relevant, and context-aware, outperforming generalized responses from systems like Gemini. These findings highlight the potential IoT-ASE to make real-time IoT data accessible and support effective, real-time decision-making. 

---
