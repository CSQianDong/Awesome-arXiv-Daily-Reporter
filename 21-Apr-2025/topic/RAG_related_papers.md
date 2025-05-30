# CoT-RAG: Integrating Chain of Thought and Retrieval-Augmented Generation to Enhance Reasoning in Large Language Models 

**Authors**: Feiyang Li, Peng Fang, Zhan Shi, Arijit Khan, Fang Wang, Dan Feng, Weihao Wang, Xin Zhang, Yongjian Cui  

**Link**: [PDF](https://arxiv.org/pdf/2504.13534)  

**Abstract**: While chain-of-thought (CoT) reasoning improves the performance of large language models (LLMs) in complex tasks, it still has two main challenges: the low reliability of relying solely on LLMs to generate reasoning chains and the interference of natural language reasoning chains on the inference logic of LLMs. To address these issues, we propose CoT-RAG, a novel reasoning framework with three key designs: (i) Knowledge Graph-driven CoT Generation, featuring knowledge graphs to modulate reasoning chain generation of LLMs, thereby enhancing reasoning credibility; (ii) Learnable Knowledge Case-aware RAG, which incorporates retrieval-augmented generation (RAG) into knowledge graphs to retrieve relevant sub-cases and sub-descriptions, providing LLMs with learnable information; (iii) Pseudo-Program Prompting Execution, which encourages LLMs to execute reasoning tasks in pseudo-programs with greater logical rigor. We conduct a comprehensive evaluation on nine public datasets, covering three reasoning problems. Compared with the-state-of-the-art methods, CoT-RAG exhibits a significant accuracy improvement, ranging from 4.0% to 23.0%. Furthermore, testing on four domain-specific datasets, CoT-RAG shows remarkable accuracy and efficient execution, highlighting its strong practical applicability and scalability. 

---
# Secure Multifaceted-RAG for Enterprise: Hybrid Knowledge Retrieval with Security Filtering 

**Authors**: Grace Byun, Shinsun Lee, Nayoung Choi, Jinho Choi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13425)  

**Abstract**: Existing Retrieval-Augmented Generation (RAG) systems face challenges in enterprise settings due to limited retrieval scope and data security risks. When relevant internal documents are unavailable, the system struggles to generate accurate and complete responses. Additionally, using closed-source Large Language Models (LLMs) raises concerns about exposing proprietary information. To address these issues, we propose the Secure Multifaceted-RAG (SecMulti-RAG) framework, which retrieves not only from internal documents but also from two supplementary sources: pre-generated expert knowledge for anticipated queries and on-demand external LLM-generated knowledge. To mitigate security risks, we adopt a local open-source generator and selectively utilize external LLMs only when prompts are deemed safe by a filtering mechanism. This approach enhances completeness, prevents data leakage, and reduces costs. In our evaluation on a report generation task in the automotive industry, SecMulti-RAG significantly outperforms traditional RAG - achieving 79.3 to 91.9 percent win rates across correctness, richness, and helpfulness in LLM-based evaluation, and 56.3 to 70.4 percent in human evaluation. This highlights SecMulti-RAG as a practical and secure solution for enterprise RAG. 

---
# RAG Without the Lag: Interactive Debugging for Retrieval-Augmented Generation Pipelines 

**Authors**: Quentin Romero Lauro, Shreya Shankar, Sepanta Zeighami, Aditya Parameswaran  

**Link**: [PDF](https://arxiv.org/pdf/2504.13587)  

**Abstract**: Retrieval-augmented generation (RAG) pipelines have become the de-facto approach for building AI assistants with access to external, domain-specific knowledge. Given a user query, RAG pipelines typically first retrieve (R) relevant information from external sources, before invoking a Large Language Model (LLM), augmented (A) with this information, to generate (G) responses. Modern RAG pipelines frequently chain multiple retrieval and generation components, in any order. However, developing effective RAG pipelines is challenging because retrieval and generation components are intertwined, making it hard to identify which component(s) cause errors in the eventual output. The parameters with the greatest impact on output quality often require hours of pre-processing after each change, creating prohibitively slow feedback cycles. To address these challenges, we present RAGGY, a developer tool that combines a Python library of composable RAG primitives with an interactive interface for real-time debugging. We contribute the design and implementation of RAGGY, insights into expert debugging patterns through a qualitative study with 12 engineers, and design implications for future RAG tools that better align with developers' natural workflows. 

---
# On the Feasibility of Using MultiModal LLMs to Execute AR Social Engineering Attacks 

**Authors**: Ting Bi, Chenghang Ye, Zheyu Yang, Ziyi Zhou, Cui Tang, Jun Zhang, Zui Tao, Kailong Wang, Liting Zhou, Yang Yang, Tianlong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.13209)  

**Abstract**: Augmented Reality (AR) and Multimodal Large Language Models (LLMs) are rapidly evolving, providing unprecedented capabilities for human-computer interaction. However, their integration introduces a new attack surface for social engineering. In this paper, we systematically investigate the feasibility of orchestrating AR-driven Social Engineering attacks using Multimodal LLM for the first time, via our proposed SEAR framework, which operates through three key phases: (1) AR-based social context synthesis, which fuses Multimodal inputs (visual, auditory and environmental cues); (2) role-based Multimodal RAG (Retrieval-Augmented Generation), which dynamically retrieves and integrates contextual data while preserving character differentiation; and (3) ReInteract social engineering agents, which execute adaptive multiphase attack strategies through inference interaction loops. To verify SEAR, we conducted an IRB-approved study with 60 participants in three experimental configurations (unassisted, AR+LLM, and full SEAR pipeline) compiling a new dataset of 180 annotated conversations in simulated social scenarios. Our results show that SEAR is highly effective at eliciting high-risk behaviors (e.g., 93.3% of participants susceptible to email phishing). The framework was particularly effective in building trust, with 85% of targets willing to accept an attacker's call after an interaction. Also, we identified notable limitations such as ``occasionally artificial'' due to perceived authenticity gaps. This work provides proof-of-concept for AR-LLM driven social engineering attacks and insights for developing defensive countermeasures against next-generation augmented reality threats. 

---
