# Benchmarking Retrieval-Augmented Generation for Chemistry 

**Authors**: Xianrui Zhong, Bowen Jin, Siru Ouyang, Yanzhen Shen, Qiao Jin, Yin Fang, Zhiyong Lu, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07671)  

**Abstract**: Retrieval-augmented generation (RAG) has emerged as a powerful framework for enhancing large language models (LLMs) with external knowledge, particularly in scientific domains that demand specialized and dynamic information. Despite its promise, the application of RAG in the chemistry domain remains underexplored, primarily due to the lack of high-quality, domain-specific corpora and well-curated evaluation benchmarks. In this work, we introduce ChemRAG-Bench, a comprehensive benchmark designed to systematically assess the effectiveness of RAG across a diverse set of chemistry-related tasks. The accompanying chemistry corpus integrates heterogeneous knowledge sources, including scientific literature, the PubChem database, PubMed abstracts, textbooks, and Wikipedia entries. In addition, we present ChemRAG-Toolkit, a modular and extensible RAG toolkit that supports five retrieval algorithms and eight LLMs. Using ChemRAG-Toolkit, we demonstrate that RAG yields a substantial performance gain -- achieving an average relative improvement of 17.4% over direct inference methods. We further conduct in-depth analyses on retriever architectures, corpus selection, and the number of retrieved passages, culminating in practical recommendations to guide future research and deployment of RAG systems in the chemistry domain. The code and data is available at this https URL. 

---
# Reinforced Internal-External Knowledge Synergistic Reasoning for Efficient Adaptive Search Agent 

**Authors**: Ziyang Huang, Xiaowei Yuan, Yiming Ju, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07596)  

**Abstract**: Retrieval-augmented generation (RAG) is a common strategy to reduce hallucinations in Large Language Models (LLMs). While reinforcement learning (RL) can enable LLMs to act as search agents by activating retrieval capabilities, existing ones often underutilize their internal knowledge. This can lead to redundant retrievals, potential harmful knowledge conflicts, and increased inference latency. To address these limitations, an efficient and adaptive search agent capable of discerning optimal retrieval timing and synergistically integrating parametric (internal) and retrieved (external) knowledge is in urgent need. This paper introduces the Reinforced Internal-External Knowledge Synergistic Reasoning Agent (IKEA), which could indentify its own knowledge boundary and prioritize the utilization of internal knowledge, resorting to external search only when internal knowledge is deemed insufficient. This is achieved using a novel knowledge-boundary aware reward function and a knowledge-boundary aware training dataset. These are designed for internal-external knowledge synergy oriented RL, incentivizing the model to deliver accurate answers, minimize unnecessary retrievals, and encourage appropriate external searches when its own knowledge is lacking. Evaluations across multiple knowledge reasoning tasks demonstrate that IKEA significantly outperforms baseline methods, reduces retrieval frequency significantly, and exhibits robust generalization capabilities. 

---
# DynamicRAG: Leveraging Outputs of Large Language Model as Feedback for Dynamic Reranking in Retrieval-Augmented Generation 

**Authors**: Jiashuo Sun, Xianrui Zhong, Sizhe Zhou, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.07233)  

**Abstract**: Retrieval-augmented generation (RAG) systems combine large language models (LLMs) with external knowledge retrieval, making them highly effective for knowledge-intensive tasks. A crucial but often under-explored component of these systems is the reranker, which refines retrieved documents to enhance generation quality and explainability. The challenge of selecting the optimal number of documents (k) remains unsolved: too few may omit critical information, while too many introduce noise and inefficiencies. Although recent studies have explored LLM-based rerankers, they primarily leverage internal model knowledge and overlook the rich supervisory signals that LLMs can provide, such as using response quality as feedback for optimizing reranking decisions. In this paper, we propose DynamicRAG, a novel RAG framework where the reranker dynamically adjusts both the order and number of retrieved documents based on the query. We model the reranker as an agent optimized through reinforcement learning (RL), using rewards derived from LLM output quality. Across seven knowledge-intensive datasets, DynamicRAG demonstrates superior performance, achieving state-of-the-art results. The model, data and code are available at this https URL 

---
# ThreatLens: LLM-guided Threat Modeling and Test Plan Generation for Hardware Security Verification 

**Authors**: Dipayan Saha, Hasan Al Shaikh, Shams Tarek, Farimah Farahmandi  

**Link**: [PDF](https://arxiv.org/pdf/2505.06821)  

**Abstract**: Current hardware security verification processes predominantly rely on manual threat modeling and test plan generation, which are labor-intensive, error-prone, and struggle to scale with increasing design complexity and evolving attack methodologies. To address these challenges, we propose ThreatLens, an LLM-driven multi-agent framework that automates security threat modeling and test plan generation for hardware security verification. ThreatLens integrates retrieval-augmented generation (RAG) to extract relevant security knowledge, LLM-powered reasoning for threat assessment, and interactive user feedback to ensure the generation of practical test plans. By automating these processes, the framework reduces the manual verification effort, enhances coverage, and ensures a structured, adaptable approach to security verification. We evaluated our framework on the NEORV32 SoC, demonstrating its capability to automate security verification through structured test plans and validating its effectiveness in real-world scenarios. 

---
# MacRAG: Compress, Slice, and Scale-up for Multi-Scale Adaptive Context RAG 

**Authors**: Woosang Lim, Zekun Li, Gyuwan Kim, Sungyoung Ji, HyeonJung Kim, Kyuri Choi, Jin Hyuk Lim, Kyungpyo Park, William Yang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.06569)  

**Abstract**: Long-context (LC) Large Language Models (LLMs) combined with Retrieval-Augmented Generation (RAG) hold strong potential for complex multi-hop and large-document tasks. However, existing RAG systems often suffer from imprecise retrieval, incomplete context coverage under constrained context windows, and fragmented information caused by suboptimal context construction. We introduce Multi-scale Adaptive Context RAG (MacRAG), a hierarchical retrieval framework that compresses and partitions documents into coarse-to-fine granularities, then adaptively merges relevant contexts through chunk- and document-level expansions in real time. By starting from the finest-level retrieval and progressively incorporating higher-level and broader context, MacRAG constructs effective query-specific long contexts, optimizing both precision and coverage. Evaluations on the challenging LongBench expansions of HotpotQA, 2WikiMultihopQA, and Musique confirm that MacRAG consistently surpasses baseline RAG pipelines on single- and multi-step generation with Llama-3.1-8B, Gemini-1.5-pro, and GPT-4o. Our results establish MacRAG as an efficient, scalable solution for real-world long-context, multi-hop reasoning. Our code is available at this https URL. 

---
# AI Approaches to Qualitative and Quantitative News Analytics on NATO Unity 

**Authors**: Bohdan M. Pavlyshenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.06313)  

**Abstract**: The paper considers the use of GPT models with retrieval-augmented generation (RAG) for qualitative and quantitative analytics on NATO sentiments, NATO unity and NATO Article 5 trust opinion scores in different web sources: news sites found via Google Search API, Youtube videos with comments, and Reddit discussions. A RAG approach using GPT-4.1 model was applied to analyse news where NATO related topics were discussed. Two levels of RAG analytics were used: on the first level, the GPT model generates qualitative news summaries and quantitative opinion scores using zero-shot prompts; on the second level, the GPT model generates the summary of news summaries. Quantitative news opinion scores generated by the GPT model were analysed using Bayesian regression to get trend lines. The distributions found for the regression parameters make it possible to analyse an uncertainty in specified news opinion score trends. Obtained results show a downward trend for analysed scores of opinion related to NATO unity.
This approach does not aim to conduct real political analysis; rather, it consider AI based approaches which can be used for further analytics
as a part of a complex analytical approach. The obtained results demonstrate that the use of GPT models for news analysis can give informative qualitative and quantitative analytics, providing important insights.
The dynamic model based on neural ordinary differential equations was considered for modelling public opinions. This approach makes it possible to analyse different scenarios for evolving public opinions. 

---
# The Distracting Effect: Understanding Irrelevant Passages in RAG 

**Authors**: Chen Amiraz, Florin Cuconasu, Simone Filice, Zohar Karnin  

**Link**: [PDF](https://arxiv.org/pdf/2505.06914)  

**Abstract**: A well-known issue with Retrieval Augmented Generation (RAG) is that retrieved passages that are irrelevant to the query sometimes distract the answer-generating LLM, causing it to provide an incorrect response. In this paper, we shed light on this core issue and formulate the distracting effect of a passage w.r.t. a query (and an LLM). We provide a quantifiable measure of the distracting effect of a passage and demonstrate its robustness across LLMs.
Our research introduces novel methods for identifying and using hard distracting passages to improve RAG systems. By fine-tuning LLMs with these carefully selected distracting passages, we achieve up to a 7.5% increase in answering accuracy compared to counterparts fine-tuned on conventional RAG datasets. Our contribution is two-fold: first, we move beyond the simple binary classification of irrelevant passages as either completely unrelated vs. distracting, and second, we develop and analyze multiple methods for finding hard distracting passages. To our knowledge, no other research has provided such a comprehensive framework for identifying and utilizing hard distracting passages. 

---
# KAQG: A Knowledge-Graph-Enhanced RAG for Difficulty-Controlled Question Generation 

**Authors**: Ching Han Chen, Ming Fang Shiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07618)  

**Abstract**: KAQG introduces a decisive breakthrough for Retrieval-Augmented Generation (RAG) by explicitly tackling the two chronic weaknesses of current pipelines: transparent multi-step reasoning and fine-grained cognitive difficulty control. This transforms RAG from a passive retriever into an accountable generator of calibrated exam items. Technically, the framework fuses knowledge graphs, RAG retrieval, and educational assessment theory into a single pipeline. Domain passages are parsed into a structured graph; graph-aware retrieval feeds fact chains to an LLM; and an assessment layer governed by Bloom's Taxonomy levels and Item Response Theory (IRT) transforms those chains into psychometrically sound questions. This cross-disciplinary marriage yields two scholarly contributions: it shows how semantic graph contexts guide LLM reasoning paths, and it operationalizes difficulty metrics within the generation process, producing items whose IRT parameters match expert benchmarks. Every module, from KG construction scripts to the multi-agent reasoning scheduler and the automatic IRT validator, is openly released on GitHub. This enables peer laboratories to replicate experiments, benchmark against baselines, and extend individual components without licensing barriers. Its reproducible design paves the way for rigorous ablation studies, cross-domain transfer experiments, and shared leaderboards on multi-step reasoning benchmarks. 

---
# Why Uncertainty Estimation Methods Fall Short in RAG: An Axiomatic Analysis 

**Authors**: Heydar Soudani, Evangelos Kanoulas, Faegheh Hasibi  

**Link**: [PDF](https://arxiv.org/pdf/2505.07459)  

**Abstract**: Large Language Models (LLMs) are valued for their strong performance across various tasks, but they also produce inaccurate or misleading outputs. Uncertainty Estimation (UE) quantifies the model's confidence and helps users assess response reliability. However, existing UE methods have not been thoroughly examined in scenarios like Retrieval-Augmented Generation (RAG), where the input prompt includes non-parametric knowledge. This paper shows that current UE methods cannot reliably assess correctness in the RAG setting. We further propose an axiomatic framework to identify deficiencies in existing methods and guide the development of improved approaches. Our framework introduces five constraints that an effective UE method should meet after incorporating retrieved documents into the LLM's prompt. Experimental results reveal that no existing UE method fully satisfies all the axioms, explaining their suboptimal performance in RAG. We further introduce a simple yet effective calibration function based on our framework, which not only satisfies more axioms than baseline methods but also improves the correlation between uncertainty estimates and correctness. 

---
