# InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning 

**Authors**: Zheng Wang, Shu Xian Teo, Jun Jie Chew, Wei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13032)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled their use as agents for planning complex tasks. Existing methods typically rely on a thought-action-observation (TAO) process to enhance LLM performance, but these approaches are often constrained by the LLMs' limited knowledge of complex tasks. Retrieval-augmented generation (RAG) offers new opportunities by leveraging external databases to ground generation in retrieved information. In this paper, we identify two key challenges (enlargability and transferability) in applying RAG to task planning. We propose InstructRAG, a novel solution within a multi-agent meta-reinforcement learning framework, to address these challenges. InstructRAG includes a graph to organize past instruction paths (sequences of correct actions), an RL-Agent with Reinforcement Learning to expand graph coverage for enlargability, and an ML-Agent with Meta-Learning to improve task generalization for transferability. The two agents are trained end-to-end to optimize overall planning performance. Our experiments on four widely used task planning datasets demonstrate that InstructRAG significantly enhances performance and adapts efficiently to new tasks, achieving up to a 19.2% improvement over the best existing approach. 

---
# Towards Conversational AI for Human-Machine Collaborative MLOps 

**Authors**: George Fatouros, Georgios Makridis, George Kousiouris, John Soldatos, Anargyros Tsadimas, Dimosthenis Kyriazis  

**Link**: [PDF](https://arxiv.org/pdf/2504.12477)  

**Abstract**: This paper presents a Large Language Model (LLM) based conversational agent system designed to enhance human-machine collaboration in Machine Learning Operations (MLOps). We introduce the Swarm Agent, an extensible architecture that integrates specialized agents to create and manage ML workflows through natural language interactions. The system leverages a hierarchical, modular design incorporating a KubeFlow Pipelines (KFP) Agent for ML pipeline orchestration, a MinIO Agent for data management, and a Retrieval-Augmented Generation (RAG) Agent for domain-specific knowledge integration. Through iterative reasoning loops and context-aware processing, the system enables users with varying technical backgrounds to discover, execute, and monitor ML pipelines; manage datasets and artifacts; and access relevant documentation, all via intuitive conversational interfaces. Our approach addresses the accessibility gap in complex MLOps platforms like Kubeflow, making advanced ML tools broadly accessible while maintaining the flexibility to extend to other platforms. The paper describes the architecture, implementation details, and demonstrates how this conversational MLOps assistant reduces complexity and lowers barriers to entry for users across diverse technical skill levels. 

---
# Accommodate Knowledge Conflicts in Retrieval-augmented LLMs: Towards Reliable Response Generation in the Wild 

**Authors**: Jiatai Wang, Zhiwei Xu, Di Jin, Xuewen Yang, Tao Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.12982)  

**Abstract**: The proliferation of large language models (LLMs) has significantly advanced information retrieval systems, particularly in response generation (RG). Unfortunately, LLMs often face knowledge conflicts between internal memory and retrievaled external information, arising from misinformation, biases, or outdated knowledge. These conflicts undermine response reliability and introduce uncertainty in decision-making. In this work, we analyze how LLMs navigate knowledge conflicts from an information-theoretic perspective and reveal that when conflicting and supplementary information exhibit significant differences, LLMs confidently resolve their preferences. However, when the distinction is ambiguous, LLMs experience heightened uncertainty. Based on this insight, we propose Swin-VIB, a novel framework that integrates a pipeline of variational information bottleneck models into adaptive augmentation of retrieved information and guiding LLM preference in response generation. Extensive experiments on single-choice, open-ended question-answering (QA), and retrieval augmented generation (RAG) validate our theoretical findings and demonstrate the efficacy of Swin-VIB. Notably, our method improves single-choice task accuracy by at least 7.54\% over competitive baselines. 

---
# Retrieval-Augmented Generation with Conflicting Evidence 

**Authors**: Han Wang, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2504.13079)  

**Abstract**: Large language model (LLM) agents are increasingly employing retrieval-augmented generation (RAG) to improve the factuality of their responses. However, in practice, these systems often need to handle ambiguous user queries and potentially conflicting information from multiple sources while also suppressing inaccurate information from noisy or irrelevant documents. Prior work has generally studied and addressed these challenges in isolation, considering only one aspect at a time, such as handling ambiguity or robustness to noise and misinformation. We instead consider multiple factors simultaneously, proposing (i) RAMDocs (Retrieval with Ambiguity and Misinformation in Documents), a new dataset that simulates complex and realistic scenarios for conflicting evidence for a user query, including ambiguity, misinformation, and noise; and (ii) MADAM-RAG, a multi-agent approach in which LLM agents debate over the merits of an answer over multiple rounds, allowing an aggregator to collate responses corresponding to disambiguated entities while discarding misinformation and noise, thereby handling diverse sources of conflict jointly. We demonstrate the effectiveness of MADAM-RAG using both closed and open-source models on AmbigDocs -- which requires presenting all valid answers for ambiguous queries -- improving over strong RAG baselines by up to 11.40% and on FaithEval -- which requires suppressing misinformation -- where we improve by up to 15.80% (absolute) with Llama3.3-70B-Instruct. Furthermore, we find that RAMDocs poses a challenge for existing RAG baselines (Llama3.3-70B-Instruct only obtains 32.60 exact match score). While MADAM-RAG begins to address these conflicting factors, our analysis indicates that a substantial gap remains especially when increasing the level of imbalance in supporting evidence and misinformation. 

---
# ACoRN: Noise-Robust Abstractive Compression in Retrieval-Augmented Language Models 

**Authors**: Singon Kim, Gunho Jung, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2504.12673)  

**Abstract**: Abstractive compression utilizes smaller langauge models to condense query-relevant context, reducing computational costs in retrieval-augmented generation (RAG). However,retrieved documents often include information that is either irrelevant to answering the query or misleading due to factual incorrect content, despite having high relevance scores. This behavior indicates that abstractive compressors are more likely to omit important information essential for the correct answer, especially in long contexts where attention dispersion occurs. To address this issue, we categorize retrieved documents in a more fine-grained manner and propose Abstractive Compression Robust against Noise (ACoRN), which introduces two novel training steps. First, we use offline data augmentation on the training dataset to enhance compressor robustness against two distinct types of retrieval noise. Second, since the language modelbased compressor cannot fully utilize information from multiple retrieved documents and exhibits positional bias, we perform finetuning to generate summaries centered around key information that directly supports the correct answer. Our experiments demonstrate that T5-large, trained with ACoRN as a compressor, improves EM and F1 scores while preserving the answer string, which could serve as direct evidence. ACoRN excels on datasets with many accuracy-reducing documents, making it highly useful in real-world scenarios. 

---
# HM-RAG: Hierarchical Multi-Agent Multimodal Retrieval Augmented Generation 

**Authors**: Pei Liu, Xin Liu, Ruoyu Yao, Junming Liu, Siyuan Meng, Ding Wang, Jun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2504.12330)  

**Abstract**: While Retrieval-Augmented Generation (RAG) augments Large Language Models (LLMs) with external knowledge, conventional single-agent RAG remains fundamentally limited in resolving complex queries demanding coordinated reasoning across heterogeneous data ecosystems. We present HM-RAG, a novel Hierarchical Multi-agent Multimodal RAG framework that pioneers collaborative intelligence for dynamic knowledge synthesis across structured, unstructured, and graph-based data. The framework is composed of three-tiered architecture with specialized agents: a Decomposition Agent that dissects complex queries into contextually coherent sub-tasks via semantic-aware query rewriting and schema-guided context augmentation; Multi-source Retrieval Agents that carry out parallel, modality-specific retrieval using plug-and-play modules designed for vector, graph, and web-based databases; and a Decision Agent that uses consistency voting to integrate multi-source answers and resolve discrepancies in retrieval results through Expert Model Refinement. This architecture attains comprehensive query understanding by combining textual, graph-relational, and web-derived evidence, resulting in a remarkable 12.95% improvement in answer accuracy and a 3.56% boost in question classification accuracy over baseline RAG systems on the ScienceQA and CrisisMMD benchmarks. Notably, HM-RAG establishes state-of-the-art results in zero-shot settings on both datasets. Its modular architecture ensures seamless integration of new data modalities while maintaining strict data governance, marking a significant advancement in addressing the critical challenges of multimodal reasoning and knowledge synthesis in RAG systems. Code is available at this https URL. 

---
# The Other Side of the Coin: Exploring Fairness in Retrieval-Augmented Generation 

**Authors**: Zheng Zhang, Ning Li, Qi Liu, Rui Li, Weibo Gao, Qingyang Mao, Zhenya Huang, Baosheng Yu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12323)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by retrieving relevant document from external knowledge sources. By referencing this external knowledge, RAG effectively reduces the generation of factually incorrect content and addresses hallucination issues within LLMs. Recently, there has been growing attention to improving the performance and efficiency of RAG systems from various perspectives. While these advancements have yielded significant results, the application of RAG in domains with considerable societal implications raises a critical question about fairness: What impact does the introduction of the RAG paradigm have on the fairness of LLMs? To address this question, we conduct extensive experiments by varying the LLMs, retrievers, and retrieval sources. Our experimental analysis reveals that the scale of the LLMs plays a significant role in influencing fairness outcomes within the RAG framework. When the model scale is smaller than 8B, the integration of retrieval mechanisms often exacerbates unfairness in small-scale LLMs (e.g., LLaMA3.2-1B, Mistral-7B, and LLaMA3-8B). To mitigate the fairness issues introduced by RAG for small-scale LLMs, we propose two approaches, FairFT and FairFilter. Specifically, in FairFT, we align the retriever with the LLM in terms of fairness, enabling it to retrieve documents that facilitate fairer model outputs. In FairFilter, we propose a fairness filtering mechanism to filter out biased content after retrieval. Finally, we validate our proposed approaches on real-world datasets, demonstrating their effectiveness in improving fairness while maintaining performance. 

---
# Estimating Optimal Context Length for Hybrid Retrieval-augmented Multi-document Summarization 

**Authors**: Adithya Pratapa, Teruko Mitamura  

**Link**: [PDF](https://arxiv.org/pdf/2504.12972)  

**Abstract**: Recent advances in long-context reasoning abilities of language models led to interesting applications in large-scale multi-document summarization. However, prior work has shown that these long-context models are not effective at their claimed context windows. To this end, retrieval-augmented systems provide an efficient and effective alternative. However, their performance can be highly sensitive to the choice of retrieval context length. In this work, we present a hybrid method that combines retrieval-augmented systems with long-context windows supported by recent language models. Our method first estimates the optimal retrieval length as a function of the retriever, summarizer, and dataset. On a randomly sampled subset of the dataset, we use a panel of LLMs to generate a pool of silver references. We use these silver references to estimate the optimal context length for a given RAG system configuration. Our results on the multi-document summarization task showcase the effectiveness of our method across model classes and sizes. We compare against length estimates from strong long-context benchmarks such as RULER and HELMET. Our analysis also highlights the effectiveness of our estimation method for very long-context LMs and its generalization to new classes of LMs. 

---
# CDF-RAG: Causal Dynamic Feedback for Adaptive Retrieval-Augmented Generation 

**Authors**: Elahe Khatibi, Ziyu Wang, Amir M. Rahmani  

**Link**: [PDF](https://arxiv.org/pdf/2504.12560)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced large language models (LLMs) in knowledge-intensive tasks by incorporating external knowledge retrieval. However, existing RAG frameworks primarily rely on semantic similarity and correlation-driven retrieval, limiting their ability to distinguish true causal relationships from spurious associations. This results in responses that may be factually grounded but fail to establish cause-and-effect mechanisms, leading to incomplete or misleading insights. To address this issue, we introduce Causal Dynamic Feedback for Adaptive Retrieval-Augmented Generation (CDF-RAG), a framework designed to improve causal consistency, factual accuracy, and explainability in generative reasoning. CDF-RAG iteratively refines queries, retrieves structured causal graphs, and enables multi-hop causal reasoning across interconnected knowledge sources. Additionally, it validates responses against causal pathways, ensuring logically coherent and factually grounded outputs. We evaluate CDF-RAG on four diverse datasets, demonstrating its ability to improve response accuracy and causal correctness over existing RAG-based methods. Our code is publicly available at this https URL elakhatibi/CDF-RAG. 

---
# Benchmarking Biopharmaceuticals Retrieval-Augmented Generation Evaluation 

**Authors**: Hanmeng Zhong, Linqing Chen, Weilei Wang, Wentao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.12342)  

**Abstract**: Recently, the application of the retrieval-augmented Large Language Models (LLMs) in specific domains has gained significant attention, especially in biopharmaceuticals. However, in this context, there is no benchmark specifically designed for biopharmaceuticals to evaluate LLMs. In this paper, we introduce the Biopharmaceuticals Retrieval-Augmented Generation Evaluation (BRAGE) , the first benchmark tailored for evaluating LLMs' Query and Reference Understanding Capability (QRUC) in the biopharmaceutical domain, available in English, French, German and Chinese. In addition, Traditional Question-Answering (QA) metrics like accuracy and exact match fall short in the open-ended retrieval-augmented QA scenarios. To address this, we propose a citation-based classification method to evaluate the QRUC of LLMs to understand the relationship between queries and references. We apply this method to evaluate the mainstream LLMs on BRAGE. Experimental results show that there is a significant gap in the biopharmaceutical QRUC of mainstream LLMs, and their QRUC needs to be improved. 

---
