# HANRAG: Heuristic Accurate Noise-resistant Retrieval-Augmented Generation for Multi-hop Question Answering 

**Authors**: Duolin Sun, Dan Yang, Yue Shen, Yihan Jiao, Zhehao Tan, Jie Feng, Lianzhen Zhong, Jian Wang, Peng Wei, Jinjie Gu  

**Link**: [PDF](https://arxiv.org/pdf/2509.09713)  

**Abstract**: The Retrieval-Augmented Generation (RAG) approach enhances question-answering systems and dialogue generation tasks by integrating information retrieval (IR) technologies with large language models (LLMs). This strategy, which retrieves information from external knowledge bases to bolster the response capabilities of generative models, has achieved certain successes. However, current RAG methods still face numerous challenges when dealing with multi-hop queries. For instance, some approaches overly rely on iterative retrieval, wasting too many retrieval steps on compound queries. Additionally, using the original complex query for retrieval may fail to capture content relevant to specific sub-queries, resulting in noisy retrieved content. If the noise is not managed, it can lead to the problem of noise accumulation. To address these issues, we introduce HANRAG, a novel heuristic-based framework designed to efficiently tackle problems of varying complexity. Driven by a powerful revelator, HANRAG routes queries, decomposes them into sub-queries, and filters noise from retrieved documents. This enhances the system's adaptability and noise resistance, making it highly capable of handling diverse queries. We compare the proposed framework against other leading industry methods across various benchmarks. The results demonstrate that our framework obtains superior performance in both single-hop and multi-hop question-answering tasks. 

---
# A Role-Aware Multi-Agent Framework for Financial Education Question Answering with LLMs 

**Authors**: Andy Zhu, Yingjun Du  

**Link**: [PDF](https://arxiv.org/pdf/2509.09727)  

**Abstract**: Question answering (QA) plays a central role in financial education, yet existing large language model (LLM) approaches often fail to capture the nuanced and specialized reasoning required for financial problem-solving. The financial domain demands multistep quantitative reasoning, familiarity with domain-specific terminology, and comprehension of real-world scenarios. We present a multi-agent framework that leverages role-based prompting to enhance performance on domain-specific QA. Our framework comprises a Base Generator, an Evidence Retriever, and an Expert Reviewer agent that work in a single-pass iteration to produce a refined answer. We evaluated our framework on a set of 3,532 expert-designed finance education questions from this http URL, an online learning platform. We leverage retrieval-augmented generation (RAG) for contextual evidence from 6 finance textbooks and prompting strategies for a domain-expert reviewer. Our experiments indicate that critique-based refinement improves answer accuracy by 6.6-8.3% over zero-shot Chain-of-Thought baselines, with the highest performance from Gemini-2.0-Flash. Furthermore, our method enables GPT-4o-mini to achieve performance comparable to the finance-tuned FinGPT-mt_Llama3-8B_LoRA. Our results show a cost-effective approach to enhancing financial QA and offer insights for further research in multi-agent financial LLM systems. 

---
# Querying Climate Knowledge: Semantic Retrieval for Scientific Discovery 

**Authors**: Mustapha Adamu, Qi Zhang, Huitong Pan, Longin Jan Latecki, Eduard C. Dragut  

**Link**: [PDF](https://arxiv.org/pdf/2509.10087)  

**Abstract**: The growing complexity and volume of climate science literature make it increasingly difficult for researchers to find relevant information across models, datasets, regions, and variables. This paper introduces a domain-specific Knowledge Graph (KG) built from climate publications and broader scientific texts, aimed at improving how climate knowledge is accessed and used. Unlike keyword based search, our KG supports structured, semantic queries that help researchers discover precise connections such as which models have been validated in specific regions or which datasets are commonly used with certain teleconnection patterns. We demonstrate how the KG answers such questions using Cypher queries, and outline its integration with large language models in RAG systems to improve transparency and reliability in climate-related question answering. This work moves beyond KG construction to show its real world value for climate researchers, model developers, and others who rely on accurate, contextual scientific information. 

---
# AI-Powered Assistant for Long-Term Access to RHIC Knowledge 

**Authors**: Mohammad Atif, Vincent Garonne, Eric Lancon, Jerome Lauret, Alexandr Prozorov, Michal Vranovsky  

**Link**: [PDF](https://arxiv.org/pdf/2509.09688)  

**Abstract**: As the Relativistic Heavy Ion Collider (RHIC) at Brookhaven National Laboratory concludes 25 years of operation, preserving not only its vast data holdings ($\sim$1 ExaByte) but also the embedded scientific knowledge becomes a critical priority. The RHIC Data and Analysis Preservation Plan (DAPP) introduces an AI-powered assistant system that provides natural language access to documentation, workflows, and software, with the aim of supporting reproducibility, education, and future discovery. Built upon Large Language Models using Retrieval-Augmented Generation and the Model Context Protocol, this assistant indexes structured and unstructured content from RHIC experiments and enables domain-adapted interaction. We report on the deployment, computational performance, ongoing multi-experiment integration, and architectural features designed for a sustainable and explainable long-term AI access. Our experience illustrates how modern AI/ML tools can transform the usability and discoverability of scientific legacy data. 

---
# Towards an AI-based knowledge assistant for goat farmers based on Retrieval-Augmented Generation 

**Authors**: Nana Han, Dong Liu, Tomas Norton  

**Link**: [PDF](https://arxiv.org/pdf/2509.09848)  

**Abstract**: Large language models (LLMs) are increasingly being recognised as valuable knowledge communication tools in many industries. However, their application in livestock farming remains limited, being constrained by several factors not least the availability, diversity and complexity of knowledge sources. This study introduces an intelligent knowledge assistant system designed to support health management in farmed goats. Leveraging the Retrieval-Augmented Generation (RAG), two structured knowledge processing methods, table textualization and decision-tree textualization, were proposed to enhance large language models' (LLMs) understanding of heterogeneous data formats. Based on these methods, a domain-specific goat farming knowledge base was established to improve LLM's capacity for cross-scenario generalization. The knowledge base spans five key domains: Disease Prevention and Treatment, Nutrition Management, Rearing Management, Goat Milk Management, and Basic Farming Knowledge. Additionally, an online search module is integrated to enable real-time retrieval of up-to-date information. To evaluate system performance, six ablation experiments were conducted to examine the contribution of each component. The results demonstrated that heterogeneous knowledge fusion method achieved the best results, with mean accuracies of 87.90% on the validation set and 84.22% on the test set. Across the text-based, table-based, decision-tree based Q&A tasks, accuracy consistently exceeded 85%, validating the effectiveness of structured knowledge fusion within a modular design. Error analysis identified omission as the predominant error category, highlighting opportunities to further improve retrieval coverage and context integration. In conclusion, the results highlight the robustness and reliability of the proposed system for practical applications in goat farming. 

---
# A Multimodal RAG Framework for Housing Damage Assessment: Collaborative Optimization of Image Encoding and Policy Vector Retrieval 

**Authors**: Jiayi Miao, Dingxin Lu, Zhuqi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.09721)  

**Abstract**: After natural disasters, accurate evaluations of damage to housing are important for insurance claims response and planning of resources. In this work, we introduce a novel multimodal retrieval-augmented generation (MM-RAG) framework. On top of classical RAG architecture, we further the framework to devise a two-branch multimodal encoder structure that the image branch employs a visual encoder composed of ResNet and Transformer to extract the characteristic of building damage after disaster, and the text branch harnesses a BERT retriever for the text vectorization of posts as well as insurance policies and for the construction of a retrievable restoration index. To impose cross-modal semantic alignment, the model integrates a cross-modal interaction module to bridge the semantic representation between image and text via multi-head attention. Meanwhile, in the generation module, the introduced modal attention gating mechanism dynamically controls the role of visual evidence and text prior information during generation. The entire framework takes end-to-end training, and combines the comparison loss, the retrieval loss and the generation loss to form multi-task optimization objectives, and achieves image understanding and policy matching in collaborative learning. The results demonstrate superior performance in retrieval accuracy and classification index on damage severity, where the Top-1 retrieval accuracy has been improved by 9.6%. 

---
# GeoGPT.RAG Technical Report 

**Authors**: Fei Huang, Fan Wu, Zeqing Zhang, Qihao Wang, Long Zhang, Grant Michael Boquet, Hongyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.09686)  

**Abstract**: GeoGPT is an open large language model system built to advance research in the geosciences. To enhance its domain-specific capabilities, we integrated Retrieval Augmented Generation(RAG), which augments model outputs with relevant information retrieved from an external knowledge source. GeoGPT uses RAG to draw from the GeoGPT Library, a specialized corpus curated for geoscientific content, enabling it to generate accurate, context-specific answers. Users can also create personalized knowledge bases by uploading their own publication lists, allowing GeoGPT to retrieve and respond using user-provided materials. To further improve retrieval quality and domain alignment, we fine-tuned both the embedding model and a ranking model that scores retrieved passages by relevance to the query. These enhancements optimize RAG for geoscience applications and significantly improve the system's ability to deliver precise and trustworthy outputs. GeoGPT reflects a strong commitment to open science through its emphasis on collaboration, transparency, and community driven development. As part of this commitment, we have open-sourced two core RAG components-GeoEmbedding and GeoReranker-to support geoscientists, researchers, and professionals worldwide with powerful, accessible AI tools. 

---
