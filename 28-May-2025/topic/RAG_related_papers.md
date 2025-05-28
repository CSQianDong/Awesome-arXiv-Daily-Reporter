# Cold-Start Recommendation with Knowledge-Guided Retrieval-Augmented Generation 

**Authors**: Wooseong Yang, Weizhi Zhang, Yuqing Liu, Yuwei Han, Yu Wang, Junhyun Lee, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20773)  

**Abstract**: Cold-start items remain a persistent challenge in recommender systems due to their lack of historical user interactions, which collaborative models rely on. While recent zero-shot methods leverage large language models (LLMs) to address this, they often struggle with sparse metadata and hallucinated or incomplete knowledge. We propose ColdRAG, a retrieval-augmented generation approach that builds a domain-specific knowledge graph dynamically to enhance LLM-based recommendation in cold-start scenarios, without requiring task-specific fine-tuning. ColdRAG begins by converting structured item attributes into rich natural-language profiles, from which it extracts entities and relationships to construct a unified knowledge graph capturing item semantics. Given a user's interaction history, it scores edges in the graph using an LLM, retrieves candidate items with supporting evidence, and prompts the LLM to rank them. By enabling multi-hop reasoning over this graph, ColdRAG grounds recommendations in verifiable evidence, reducing hallucinations and strengthening semantic connections. Experiments on three public benchmarks demonstrate that ColdRAG surpasses existing zero-shot baselines in both Recall and NDCG. This framework offers a practical solution to cold-start recommendation by combining knowledge-graph reasoning with retrieval-augmented LLM generation. 

---
# TeroSeek: An AI-Powered Knowledge Base and Retrieval Generation Platform for Terpenoid Research 

**Authors**: Xu Kang, Siqi Jiang, Kangwei Xu, Jiahao Li, Ruibo Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20663)  

**Abstract**: Terpenoids are a crucial class of natural products that have been studied for over 150 years, but their interdisciplinary nature (spanning chemistry, pharmacology, and biology) complicates knowledge integration. To address this, the authors developed TeroSeek, a curated knowledge base (KB) built from two decades of terpenoid literature, coupled with an AI-powered question-answering chatbot and web service. Leveraging a retrieval-augmented generation (RAG) framework, TeroSeek provides structured, high-quality information and outperforms general-purpose large language models (LLMs) in terpenoid-related queries. It serves as a domain-specific expert tool for multidisciplinary research and is publicly available at this http URL. 

---
# What LLMs Miss in Recommendations: Bridging the Gap with Retrieval-Augmented Collaborative Signals 

**Authors**: Shahrooz Pouryousef  

**Link**: [PDF](https://arxiv.org/pdf/2505.20730)  

**Abstract**: User-item interactions contain rich collaborative signals that form the backbone of many successful recommender systems. While recent work has explored the use of large language models (LLMs) for recommendation, it remains unclear whether LLMs can effectively reason over this type of collaborative information. In this paper, we conduct a systematic comparison between LLMs and classical matrix factorization (MF) models to assess LLMs' ability to leverage user-item interaction data. We further introduce a simple retrieval-augmented generation (RAG) method that enhances LLMs by grounding their predictions in structured interaction data. Our experiments reveal that current LLMs often fall short in capturing collaborative patterns inherent to MF models, but that our RAG-based approach substantially improves recommendation quality-highlighting a promising direction for future LLM-based recommenders. 

---
# Hierarchical Retrieval with Evidence Curation for Open-Domain Financial Question Answering on Standardized Documents 

**Authors**: Jaeyoung Choe, Jihoon Kim, Woohwan Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.20368)  

**Abstract**: Retrieval-augmented generation (RAG) based large language models (LLMs) are widely used in finance for their excellent performance on knowledge-intensive tasks. However, standardized documents (e.g., SEC filing) share similar formats such as repetitive boilerplate texts, and similar table structures. This similarity forces traditional RAG methods to misidentify near-duplicate text, leading to duplicate retrieval that undermines accuracy and completeness. To address these issues, we propose the Hierarchical Retrieval with Evidence Curation (HiREC) framework. Our approach first performs hierarchical retrieval to reduce confusion among similar texts. It first retrieve related documents and then selects the most relevant passages from the documents. The evidence curation process removes irrelevant passages. When necessary, it automatically generates complementary queries to collect missing information. To evaluate our approach, we construct and release a Large-scale Open-domain Financial (LOFin) question answering benchmark that includes 145,897 SEC documents and 1,595 question-answer pairs. Our code and data are available at this https URL. 

---
# Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval Augmented Generation 

**Authors**: Ekaterina Fadeeva, Aleksandr Rubashevskii, Roman Vashurin, Shehzaad Dhuliawala, Artem Shelmanov, Timothy Baldwin, Preslav Nakov, Mrinmaya Sachan, Maxim Panov  

**Link**: [PDF](https://arxiv.org/pdf/2505.21072)  

**Abstract**: Large Language Models (LLMs) enhanced with external knowledge retrieval, an approach known as Retrieval-Augmented Generation (RAG), have shown strong performance in open-domain question answering. However, RAG systems remain susceptible to hallucinations: factually incorrect outputs that may arise either from inconsistencies in the model's internal knowledge or incorrect use of the retrieved context. Existing approaches often conflate factuality with faithfulness to the retrieved context, misclassifying factually correct statements as hallucinations if they are not directly supported by the retrieval. In this paper, we introduce FRANQ (Faithfulness-based Retrieval Augmented UNcertainty Quantification), a novel method for hallucination detection in RAG outputs. FRANQ applies different Uncertainty Quantification (UQ) techniques to estimate factuality based on whether a statement is faithful to the retrieved context or not. To evaluate FRANQ and other UQ techniques for RAG, we present a new long-form Question Answering (QA) dataset annotated for both factuality and faithfulness, combining automated labeling with manual validation of challenging examples. Extensive experiments on long- and short-form QA across multiple datasets and LLMs show that FRANQ achieves more accurate detection of factual errors in RAG-generated responses compared to existing methods. 

---
# Reinforced Informativeness Optimization for Long-Form Retrieval-Augmented Generation 

**Authors**: Yuhao Wang, Ruiyang Ren, Yucheng Wang, Wayne Xin Zhao, Jing Liu, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20825)  

**Abstract**: Long-form question answering (LFQA) presents unique challenges for large language models, requiring the synthesis of coherent, paragraph-length answers. While retrieval-augmented generation (RAG) systems have emerged as a promising solution, existing research struggles with key limitations: the scarcity of high-quality training data for long-form generation, the compounding risk of hallucination in extended outputs, and the absence of reliable evaluation metrics for factual completeness. In this paper, we propose RioRAG, a novel reinforcement learning (RL) framework that advances long-form RAG through reinforced informativeness optimization. Our approach introduces two fundamental innovations to address the core challenges. First, we develop an RL training paradigm of reinforced informativeness optimization that directly optimizes informativeness and effectively addresses the slow-thinking deficit in conventional RAG systems, bypassing the need for expensive supervised data. Second, we propose a nugget-centric hierarchical reward modeling approach that enables precise assessment of long-form answers through a three-stage process: extracting the nugget from every source webpage, constructing a nugget claim checklist, and computing rewards based on factual alignment. Extensive experiments on two LFQA benchmarks LongFact and RAGChecker demonstrate the effectiveness of the proposed method. Our codes are available at this https URL. 

---
# Divide-Then-Align: Honest Alignment based on the Knowledge Boundary of RAG 

**Authors**: Xin Sun, Jianan Xie, Zhongqi Chen, Qiang Liu, Shu Wu, Yuehe Chen, Bowen Song, Weiqiang Wang, Zilei Wang, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20871)  

**Abstract**: Large language models (LLMs) augmented with retrieval systems have significantly advanced natural language processing tasks by integrating external knowledge sources, enabling more accurate and contextually rich responses. To improve the robustness of such systems against noisy retrievals, Retrieval-Augmented Fine-Tuning (RAFT) has emerged as a widely adopted method. However, RAFT conditions models to generate answers even in the absence of reliable knowledge. This behavior undermines their reliability in high-stakes domains, where acknowledging uncertainty is critical. To address this issue, we propose Divide-Then-Align (DTA), a post-training approach designed to endow RAG systems with the ability to respond with "I don't know" when the query is out of the knowledge boundary of both the retrieved passages and the model's internal knowledge. DTA divides data samples into four knowledge quadrants and constructs tailored preference data for each quadrant, resulting in a curated dataset for Direct Preference Optimization (DPO). Experimental results on three benchmark datasets demonstrate that DTA effectively balances accuracy with appropriate abstention, enhancing the reliability and trustworthiness of retrieval-augmented systems. 

---
# Less Context, Same Performance: A RAG Framework for Resource-Efficient LLM-Based Clinical NLP 

**Authors**: Satya Narayana Cheetirala, Ganesh Raut, Dhavalkumar Patel, Fabio Sanatana, Robert Freeman, Matthew A Levin, Girish N. Nadkarni, Omar Dawkins, Reba Miller, Randolph M. Steinhagen, Eyal Klang, Prem Timsina  

**Link**: [PDF](https://arxiv.org/pdf/2505.20320)  

**Abstract**: Long text classification is challenging for Large Language Models (LLMs) due to token limits and high computational costs. This study explores whether a Retrieval Augmented Generation (RAG) approach using only the most relevant text segments can match the performance of processing entire clinical notes with large context LLMs. We begin by splitting clinical documents into smaller chunks, converting them into vector embeddings, and storing these in a FAISS index. We then retrieve the top 4,000 words most pertinent to the classification query and feed these consolidated segments into an LLM. We evaluated three LLMs (GPT4o, LLaMA, and Mistral) on a surgical complication identification task. Metrics such as AUC ROC, precision, recall, and F1 showed no statistically significant differences between the RAG based approach and whole-text processing (p > 0.05p > 0.05). These findings indicate that RAG can significantly reduce token usage without sacrificing classification accuracy, providing a scalable and cost effective solution for analyzing lengthy clinical documents. 

---
# Project Riley: Multimodal Multi-Agent LLM Collaboration with Emotional Reasoning and Voting 

**Authors**: Ana Rita Ortigoso, Gabriel Vieira, Daniel Fuentes, Luis Frazão, Nuno Costa, António Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2505.20521)  

**Abstract**: This paper presents Project Riley, a novel multimodal and multi-model conversational AI architecture oriented towards the simulation of reasoning influenced by emotional states. Drawing inspiration from Pixar's Inside Out, the system comprises five distinct emotional agents - Joy, Sadness, Fear, Anger, and Disgust - that engage in structured multi-round dialogues to generate, criticise, and iteratively refine responses. A final reasoning mechanism synthesises the contributions of these agents into a coherent output that either reflects the dominant emotion or integrates multiple perspectives. The architecture incorporates both textual and visual large language models (LLMs), alongside advanced reasoning and self-refinement processes. A functional prototype was deployed locally in an offline environment, optimised for emotional expressiveness and computational efficiency. From this initial prototype, another one emerged, called Armando, which was developed for use in emergency contexts, delivering emotionally calibrated and factually accurate information through the integration of Retrieval-Augmented Generation (RAG) and cumulative context tracking. The Project Riley prototype was evaluated through user testing, in which participants interacted with the chatbot and completed a structured questionnaire assessing three dimensions: Emotional Appropriateness, Clarity and Utility, and Naturalness and Human-likeness. The results indicate strong performance in structured scenarios, particularly with respect to emotional alignment and communicative clarity. 

---
# Towards Emotionally Consistent Text-Based Speech Editing: Introducing EmoCorrector and The ECD-TSE Dataset 

**Authors**: Rui Liu, Pu Gao, Jiatian Xi, Berrak Sisman, Carlos Busso, Haizhou Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20341)  

**Abstract**: Text-based speech editing (TSE) modifies speech using only text, eliminating re-recording. However, existing TSE methods, mainly focus on the content accuracy and acoustic consistency of synthetic speech segments, and often overlook the emotional shifts or inconsistency issues introduced by text changes. To address this issue, we propose EmoCorrector, a novel post-correction scheme for TSE. EmoCorrector leverages Retrieval-Augmented Generation (RAG) by extracting the edited text's emotional features, retrieving speech samples with matching emotions, and synthesizing speech that aligns with the desired emotion while preserving the speaker's identity and quality. To support the training and evaluation of emotional consistency modeling in TSE, we pioneer the benchmarking Emotion Correction Dataset for TSE (ECD-TSE). The prominent aspect of ECD-TSE is its inclusion of $<$text, speech$>$ paired data featuring diverse text variations and a range of emotional expressions. Subjective and objective experiments and comprehensive analysis on ECD-TSE confirm that EmoCorrector significantly enhances the expression of intended emotion while addressing emotion inconsistency limitations in current TSE methods. Code and audio examples are available at this https URL. 

---
# Complex System Diagnostics Using a Knowledge Graph-Informed and Large Language Model-Enhanced Framework 

**Authors**: Saman Marandi, Yu-Shu Hu, Mohammad Modarres  

**Link**: [PDF](https://arxiv.org/pdf/2505.21291)  

**Abstract**: In this paper, we present a novel diagnostic framework that integrates Knowledge Graphs (KGs) and Large Language Models (LLMs) to support system diagnostics in high-reliability systems such as nuclear power plants. Traditional diagnostic modeling struggles when systems become too complex, making functional modeling a more attractive approach. Our approach introduces a diagnostic framework grounded in the functional modeling principles of the Dynamic Master Logic (DML) model. It incorporates two coordinated LLM components, including an LLM-based workflow for automated construction of DML logic from system documentation and an LLM agent that facilitates interactive diagnostics. The generated logic is encoded into a structured KG, referred to as KG-DML, which supports hierarchical fault reasoning. Expert knowledge or operational data can also be incorporated to refine the model's precision and diagnostic depth. In the interaction phase, users submit natural language queries, which are interpreted by the LLM agent. The agent selects appropriate tools for structured reasoning, including upward and downward propagation across the KG-DML. Rather than embedding KG content into every prompt, the LLM agent distinguishes between diagnostic and interpretive tasks. For diagnostics, the agent selects and executes external tools that perform structured KG reasoning. For general queries, a Graph-based Retrieval-Augmented Generation (Graph-RAG) approach is used, retrieving relevant KG segments and embedding them into the prompt to generate natural explanations. A case study on an auxiliary feedwater system demonstrated the framework's effectiveness, with over 90% accuracy in key elements and consistent tool and argument extraction, supporting its use in safety-critical diagnostics. 

---
# Graph RAG for Legal Norms: A Hierarchical and Temporal Approach 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00039)  

**Abstract**: This article proposes an adaptation of Graph Retrieval Augmented Generation (Graph RAG) specifically designed for the analysis and comprehension of legal norms, which are characterized by their predefined hierarchical structure, extensive network of internal and external references and multiple temporal versions. By combining structured knowledge graphs with contextually enriched text segments, Graph RAG offers a promising solution to address the inherent complexity and vast volume of legal data. The integration of hierarchical structure and temporal evolution into knowledge graphs - along with the concept of comprehensive Text Units - facilitates the construction of richer, interconnected representations of legal knowledge. Through a detailed analysis of Graph RAG and its application to legal norm datasets, this article aims to advance the field of Artificial Intelligence applied to Law, creating opportunities for more effective systems in legal research, legislative analysis, and decision support. 

---
