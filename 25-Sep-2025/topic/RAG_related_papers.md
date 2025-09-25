# RAG Security and Privacy: Formalizing the Threat Model and Attack Surface 

**Authors**: Atousa Arzanipour, Rouzbeh Behnia, Reza Ebrahimi, Kaushik Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2509.20324)  

**Abstract**: Retrieval-Augmented Generation (RAG) is an emerging approach in natural language processing that combines large language models (LLMs) with external document retrieval to produce more accurate and grounded responses. While RAG has shown strong potential in reducing hallucinations and improving factual consistency, it also introduces new privacy and security challenges that differ from those faced by traditional LLMs. Existing research has demonstrated that LLMs can leak sensitive information through training data memorization or adversarial prompts, and RAG systems inherit many of these vulnerabilities. At the same time, reliance of RAG on an external knowledge base opens new attack surfaces, including the potential for leaking information about the presence or content of retrieved documents, or for injecting malicious content to manipulate model behavior. Despite these risks, there is currently no formal framework that defines the threat landscape for RAG systems. In this paper, we address a critical gap in the literature by proposing, to the best of our knowledge, the first formal threat model for retrieval-RAG systems. We introduce a structured taxonomy of adversary types based on their access to model components and data, and we formally define key threat vectors such as document-level membership inference and data poisoning, which pose serious privacy and integrity risks in real-world deployments. By establishing formal definitions and attack models, our work lays the foundation for a more rigorous and principled understanding of privacy and security in RAG systems. 

---
# STAF: Leveraging LLMs for Automated Attack Tree-Based Security Test Generation 

**Authors**: Tanmay Khule, Stefan Marksteiner, Jose Alguindigue, Hannes Fuchs, Sebastian Fischmeister, Apurva Narayan  

**Link**: [PDF](https://arxiv.org/pdf/2509.20190)  

**Abstract**: In modern automotive development, security testing is critical for safeguarding systems against increasingly advanced threats. Attack trees are widely used to systematically represent potential attack vectors, but generating comprehensive test cases from these trees remains a labor-intensive, error-prone task that has seen limited automation in the context of testing vehicular systems. This paper introduces STAF (Security Test Automation Framework), a novel approach to automating security test case generation. Leveraging Large Language Models (LLMs) and a four-step self-corrective Retrieval-Augmented Generation (RAG) framework, STAF automates the generation of executable security test cases from attack trees, providing an end-to-end solution that encompasses the entire attack surface. We particularly show the elements and processes needed to provide an LLM to actually produce sensible and executable automotive security test suites, along with the integration with an automated testing framework. We further compare our tailored approach with general purpose (vanilla) LLMs and the performance of different LLMs (namely GPT-4.1 and DeepSeek) using our approach. We also demonstrate the method of our operation step-by-step in a concrete case study. Our results show significant improvements in efficiency, accuracy, scalability, and easy integration in any workflow, marking a substantial advancement in automating automotive security testing methodologies. Using TARAs as an input for verfication tests, we create synergies by connecting two vital elements of a secure automotive development process. 

---
# Table Detection with Active Learning 

**Authors**: Somraj Gautam, Nachiketa Purohit, Gaurav Harit  

**Link**: [PDF](https://arxiv.org/pdf/2509.20003)  

**Abstract**: Efficient data annotation remains a critical challenge in machine learning, particularly for object detection tasks requiring extensive labeled data. Active learning (AL) has emerged as a promising solution to minimize annotation costs by selecting the most informative samples. While traditional AL approaches primarily rely on uncertainty-based selection, recent advances suggest that incorporating diversity-based strategies can enhance sampling efficiency in object detection tasks. Our approach ensures the selection of representative examples that improve model generalization. We evaluate our method on two benchmark datasets (TableBank-LaTeX, TableBank-Word) using state-of-the-art table detection architectures, CascadeTabNet and YOLOv9. Our results demonstrate that AL-based example selection significantly outperforms random sampling, reducing annotation effort given a limited budget while maintaining comparable performance to fully supervised models. Our method achieves higher mAP scores within the same annotation budget. 

---
# Solving Freshness in RAG: A Simple Recency Prior and the Limits of Heuristic Trend Detection 

**Authors**: Matthew Grofsky  

**Link**: [PDF](https://arxiv.org/pdf/2509.19376)  

**Abstract**: We address temporal failures in RAG systems using two methods on cybersecurity data. A simple recency prior achieved an accuracy of 1.00 on freshness tasks. In contrast, a clustering heuristic for topic evolution failed (0.08 F1-score), showing trend detection requires methods beyond simple heuristics. 

---
