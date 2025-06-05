# QQSUM: A Novel Task and Model of Quantitative Query-Focused Summarization for Review-based Product Question Answering 

**Authors**: An Quang Tang, Xiuzhen Zhang, Minh Ngoc Dinh, Zhuang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.04020)  

**Abstract**: Review-based Product Question Answering (PQA) allows e-commerce platforms to automatically address customer queries by leveraging insights from user reviews. However, existing PQA systems generate answers with only a single perspective, failing to capture the diversity of customer opinions. In this paper we introduce a novel task Quantitative Query-Focused Summarization (QQSUM), which aims to summarize diverse customer opinions into representative Key Points (KPs) and quantify their prevalence to effectively answer user queries. While Retrieval-Augmented Generation (RAG) shows promise for PQA, its generated answers still fall short of capturing the full diversity of viewpoints. To tackle this challenge, our model QQSUM-RAG, which extends RAG, employs few-shot learning to jointly train a KP-oriented retriever and a KP summary generator, enabling KP-based summaries that capture diverse and representative opinions. Experimental results demonstrate that QQSUM-RAG achieves superior performance compared to state-of-the-art RAG baselines in both textual quality and quantification accuracy of opinions. Our source code is available at: this https URL 

---
# Magic Mushroom: A Customizable Benchmark for Fine-grained Analysis of Retrieval Noise Erosion in RAG Systems 

**Authors**: Yuxin Zhang, Yan Wang, Yongrui Chen, Shenyu Zhang, Xinbang Dai, Sheng Bi, Guilin Qi  

**Link**: [PDF](https://arxiv.org/pdf/2506.03901)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems enhance Large Language Models (LLMs) by incorporating external retrieved information, mitigating issues such as hallucination and outdated knowledge.
However, RAG systems are highly sensitive to retrieval noise prevalent in real-world scenarios.
Existing benchmarks fail to emulate the complex and heterogeneous noise distributions encountered in real-world retrieval environments, undermining reliable robustness assessment.
In this paper, we define four categories of retrieval noise based on linguistic properties and noise characteristics, aiming to reflect the heterogeneity of noise in real-world scenarios.
Building on this, we introduce Magic Mushroom, a benchmark for replicating "magic mushroom" noise: contexts that appear relevant on the surface but covertly mislead RAG systems.
Magic Mushroom comprises 7,468 single-hop and 3,925 multi-hop question-answer pairs.
More importantly, Magic Mushroom enables researchers to flexibly configure combinations of retrieval noise according to specific research objectives or application scenarios, allowing for highly controlled evaluation setups.
We evaluate LLM generators of varying parameter scales and classic RAG denoising strategies under diverse noise distributions to investigate their performance dynamics during progressive noise encroachment.
Our analysis reveals that both generators and denoising strategies have significant room for improvement and exhibit extreme sensitivity to noise distributions.
Magic Mushroom emerges as a promising tool for evaluating and advancing noise-robust RAG systems, accelerating their widespread deployment in real-world applications.
The Magic Mushroom benchmark is available at the this https URL. 

---
# ScoreRAG: A Retrieval-Augmented Generation Framework with Consistency-Relevance Scoring and Structured Summarization for News Generation 

**Authors**: Pei-Yun Lin, Yen-lung Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2506.03704)  

**Abstract**: This research introduces ScoreRAG, an approach to enhance the quality of automated news generation. Despite advancements in Natural Language Processing and large language models, current news generation methods often struggle with hallucinations, factual inconsistencies, and lack of domain-specific expertise when producing news articles. ScoreRAG addresses these challenges through a multi-stage framework combining retrieval-augmented generation, consistency relevance evaluation, and structured summarization. The system first retrieves relevant news documents from a vector database, maps them to complete news items, and assigns consistency relevance scores based on large language model evaluations. These documents are then reranked according to relevance, with low-quality items filtered out. The framework proceeds to generate graded summaries based on relevance scores, which guide the large language model in producing complete news articles following professional journalistic standards. Through this methodical approach, ScoreRAG aims to significantly improve the accuracy, coherence, informativeness, and professionalism of generated news articles while maintaining stability and consistency throughout the generation process. The code and demo are available at: this https URL. 

---
# Graph Counselor: Adaptive Graph Exploration via Multi-Agent Synergy to Enhance LLM Reasoning 

**Authors**: Junqi Gao, Xiang Zou, YIng Ai, Dong Li, Yichen Niu, Biqing Qi, Jianxing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.03939)  

**Abstract**: Graph Retrieval Augmented Generation (GraphRAG) effectively enhances external knowledge integration capabilities by explicitly modeling knowledge relationships, thereby improving the factual accuracy and generation quality of Large Language Models (LLMs) in specialized domains. However, existing methods suffer from two inherent limitations: 1) Inefficient Information Aggregation: They rely on a single agent and fixed iterative patterns, making it difficult to adaptively capture multi-level textual, structural, and degree information within graph data. 2) Rigid Reasoning Mechanism: They employ preset reasoning schemes, which cannot dynamically adjust reasoning depth nor achieve precise semantic correction. To overcome these limitations, we propose Graph Counselor, an GraphRAG method based on multi-agent collaboration. This method uses the Adaptive Graph Information Extraction Module (AGIEM), where Planning, Thought, and Execution Agents work together to precisely model complex graph structures and dynamically adjust information extraction strategies, addressing the challenges of multi-level dependency modeling and adaptive reasoning depth. Additionally, the Self-Reflection with Multiple Perspectives (SR) module improves the accuracy and semantic consistency of reasoning results through self-reflection and backward reasoning mechanisms. Experiments demonstrate that Graph Counselor outperforms existing methods in multiple graph reasoning tasks, exhibiting higher reasoning accuracy and generalization ability. Our code is available at this https URL. 

---
# mRAG: Elucidating the Design Space of Multi-modal Retrieval-Augmented Generation 

**Authors**: Chan-Wei Hu, Yueqi Wang, Shuo Xing, Chia-Ju Chen, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.24073)  

**Abstract**: Large Vision-Language Models (LVLMs) have made remarkable strides in multimodal tasks such as visual question answering, visual grounding, and complex reasoning. However, they remain limited by static training data, susceptibility to hallucinations, and inability to verify claims against up-to-date, external evidence, compromising their performance in dynamic real-world applications. Retrieval-Augmented Generation (RAG) offers a practical solution to mitigate these challenges by allowing the LVLMs to access large-scale knowledge databases via retrieval mechanisms, thereby grounding model outputs in factual, contextually relevant information. Here in this paper, we conduct the first systematic dissection of the multimodal RAG pipeline for LVLMs, explicitly investigating (1) the retrieval phase: on the modality configurations and retrieval strategies, (2) the re-ranking stage: on strategies to mitigate positional biases and improve the relevance of retrieved evidence, and (3) the generation phase: we further investigate how to best integrate retrieved candidates into the final generation process. Finally, we extend to explore a unified agentic framework that integrates re-ranking and generation through self-reflection, enabling LVLMs to select relevant evidence and suppress irrelevant context dynamically. Our full-stack exploration of RAG for LVLMs yields substantial insights, resulting in an average performance boost of 5% without any fine-tuning. 

---
