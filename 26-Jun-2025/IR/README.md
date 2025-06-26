# Unidentified and Confounded? Understanding Two-Tower Models for Unbiased Learning to Rank 

**Authors**: Philipp Hager, Onno Zoeter, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2506.20501)  

**Abstract**: Additive two-tower models are popular learning-to-rank methods for handling biased user feedback in industry settings. Recent studies, however, report a concerning phenomenon: training two-tower models on clicks collected by well-performing production systems leads to decreased ranking performance. This paper investigates two recent explanations for this observation: confounding effects from logging policies and model identifiability issues. We theoretically analyze the identifiability conditions of two-tower models, showing that either document swaps across positions or overlapping feature distributions are required to recover model parameters from clicks. We also investigate the effect of logging policies on two-tower models, finding that they introduce no bias when models perfectly capture user behavior. However, logging policies can amplify biases when models imperfectly capture user behavior, particularly when prediction errors correlate with document placement across positions. We propose a sample weighting technique to mitigate these effects and provide actionable insights for researchers and practitioners using two-tower models. 

---
# Semantic-enhanced Modality-asymmetric Retrieval for Online E-commerce Search 

**Authors**: Zhigong Zhou, Ning Ding, Xiaochuan Fan, Yue Shang, Yiming Qiu, Jingwei Zhuo, Zhiwei Ge, Songlin Wang, Lin Liu, Sulong Xu, Han Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20330)  

**Abstract**: Semantic retrieval, which retrieves semantically matched items given a textual query, has been an essential component to enhance system effectiveness in e-commerce search. In this paper, we study the multimodal retrieval problem, where the visual information (e.g, image) of item is leveraged as supplementary of textual information to enrich item representation and further improve retrieval performance. Though learning from cross-modality data has been studied extensively in tasks such as visual question answering or media summarization, multimodal retrieval remains a non-trivial and unsolved problem especially in the asymmetric scenario where the query is unimodal while the item is multimodal. In this paper, we propose a novel model named SMAR, which stands for Semantic-enhanced Modality-Asymmetric Retrieval, to tackle the problem of modality fusion and alignment in this kind of asymmetric scenario. Extensive experimental results on an industrial dataset show that the proposed model outperforms baseline models significantly in retrieval accuracy. We have open sourced our industrial dataset for the sake of reproducibility and future research works. 

---
# Multimodal Information Retrieval for Open World with Edit Distance Weak Supervision 

**Authors**: KMA Solaiman, Bharat Bhargava  

**Link**: [PDF](https://arxiv.org/pdf/2506.20070)  

**Abstract**: Existing multi-media retrieval models either rely on creating a common subspace with modality-specific representation models or require schema mapping among modalities to measure similarities among multi-media data. Our goal is to avoid the annotation overhead incurred from considering retrieval as a supervised classification task and re-use the pretrained encoders in large language models and vision tasks. We propose "FemmIR", a framework to retrieve multimodal results relevant to information needs expressed with multimodal queries by example without any similarity label. Such identification is necessary for real-world applications where data annotations are scarce and satisfactory performance is required without fine-tuning with a common framework across applications. We curate a new dataset called MuQNOL for benchmarking progress on this task. Our technique is based on weak supervision introduced through edit distance between samples: graph edit distance can be modified to consider the cost of replacing a data sample in terms of its properties, and relevance can be measured through the implicit signal from the amount of edit cost among the objects. Unlike metric learning or encoding networks, FemmIR re-uses the high-level properties and maintains the property value and relationship constraints with a multi-level interaction score between data samples and the query example provided by the user. We empirically evaluate FemmIR on a missing person use case with MuQNOL. FemmIR performs comparably to similar retrieval systems in delivering on-demand retrieval results with exact and approximate similarities while using the existing property identifiers in the system. 

---
# Controlled Retrieval-augmented Context Evaluation for Long-form RAG 

**Authors**: Jia-Huei Ju, Suzan Verberne, Maarten de Rijke, Andrew Yates  

**Link**: [PDF](https://arxiv.org/pdf/2506.20051)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models by incorporating context retrieved from external knowledge sources. While the effectiveness of the retrieval module is typically evaluated with relevance-based ranking metrics, such metrics may be insufficient to reflect the retrieval's impact on the final RAG result, especially in long-form generation scenarios. We argue that providing a comprehensive retrieval-augmented context is important for long-form RAG tasks like report generation and propose metrics for assessing the context independent of generation. We introduce CRUX, a \textbf{C}ontrolled \textbf{R}etrieval-a\textbf{U}gmented conte\textbf{X}t evaluation framework designed to directly assess retrieval-augmented contexts. This framework uses human-written summaries to control the information scope of knowledge, enabling us to measure how well the context covers information essential for long-form generation. CRUX uses question-based evaluation to assess RAG's retrieval in a fine-grained manner. Empirical results show that CRUX offers more reflective and diagnostic evaluation. Our findings also reveal substantial room for improvement in current retrieval methods, pointing to promising directions for advancing RAG's retrieval. Our data and code are publicly available to support and advance future research on retrieval. 

---
# CoVE: Compressed Vocabulary Expansion Makes Better LLM-based Recommender Systems 

**Authors**: Haochen Zhang, Tianyi Zhang, Junze Yin, Oren Gal, Anshumali Shrivastava, Vladimir Braverman  

**Link**: [PDF](https://arxiv.org/pdf/2506.19993)  

**Abstract**: Recommender systems play a pivotal role in providing relevant content to users. With the rapid development of large language models (LLMs), researchers have begun utilizing LLMs to build more powerful recommender systems. However, existing approaches that focus on aligning LLMs with recommendation tasks do not fully leverage their sequential information processing capabilities, leading to suboptimal performance.
In this paper, we propose a novel system called compressed vocabulary expansion (CoVE). In CoVE, each item is assigned a unique ID within the expanded vocabulary. Our framework effectively capitalizes on sequence understanding abilities of LLMs, significantly enhancing their performance on recommendation tasks. Additionally, we compress the embedding layer, making CoVE practical for large-scale industrial applications. The effectiveness and performance of CoVE are demonstrated through comprehensive experiments on multiple recommendation datasets and comparisons with prior works. Our code can be found at this https URL. 

---
# ReCode: Updating Code API Knowledge with Reinforcement Learning 

**Authors**: Haoze Wu, Yunzhi Yao, Wenhao Yu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.20495)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable code generation capabilities but falter when adapting to frequent updates in external library APIs. This critical limitation, stemming from reliance on outdated API knowledge from their training data, even with access to current documentation, impedes reliable code generation in dynamic environments. To tackle this issue, we propose ReCode (rule-based Reinforcement learning for Code Update), a novel framework that mimics human programmer adaptation to API changes. Specifically, we construct a dataset of approximately 2,000 data entries to train the LLMs to perform version migration based on updated information. Then, we introduce a modified string similarity metric for code evaluation as the reward for reinforcement learning. Our experiments demonstrate that ReCode substantially boosts LLMs' code generation performance in dynamic API scenarios, especially on the unseen CodeUpdateArena task. Crucially, compared to supervised fine-tuning, ReCode has less impact on LLMs' general code generation abilities. We apply ReCode on various LLMs and reinforcement learning algorithms (GRPO and DAPO), all achieving consistent improvements. Notably, after training, Qwen2.5-Coder-7B outperforms that of the 32B parameter code instruction-tuned model and the reasoning model with the same architecture. Code is available at this https URL. 

---
# Knowledge-Aware Diverse Reranking for Cross-Source Question Answering 

**Authors**: Tong Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.20476)  

**Abstract**: This paper presents Team Marikarp's solution for the SIGIR 2025 LiveRAG competition. The competition's evaluation set, automatically generated by DataMorgana from internet corpora, encompassed a wide range of target topics, question types, question formulations, audience types, and knowledge organization methods. It offered a fair evaluation of retrieving question-relevant supporting documents from a 15M documents subset of the FineWeb corpus. Our proposed knowledge-aware diverse reranking RAG pipeline achieved first place in the competition. 

---
# A Literature Review on Simulation in Conversational Recommender Systems 

**Authors**: Haoran Zhang, Xin Zhao, Jinze Chen, Junpeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.20291)  

**Abstract**: Conversational Recommender Systems (CRSs) have garnered attention as a novel approach to delivering personalized recommendations through multi-turn dialogues. This review developed a taxonomy framework to systematically categorize relevant publications into four groups: dataset construction, algorithm design, system evaluation, and empirical studies, providing a comprehensive analysis of simulation methods in CRSs research. Our analysis reveals that simulation methods play a key role in tackling CRSs' main challenges. For example, LLM-based simulation methods have been used to create conversational recommendation data, enhance CRSs algorithms, and evaluate CRSs. Despite several challenges, such as dataset bias, the limited output flexibility of LLM-based simulations, and the gap between text semantic space and behavioral semantics, persist due to the complexity in Human-Computer Interaction (HCI) of CRSs, simulation methods hold significant potential for advancing CRS research. This review offers a thorough summary of the current research landscape in this domain and identifies promising directions for future inquiry. 

---
# Irec: A Metacognitive Scaffolding for Self-Regulated Learning through Just-in-Time Insight Recall: A Conceptual Framework and System Prototype 

**Authors**: Xuefei Hou, Xizhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2506.20156)  

**Abstract**: The core challenge in learning has shifted from knowledge acquisition to effective Self-Regulated Learning (SRL): planning, monitoring, and reflecting on one's learning. Existing digital tools, however, inadequately support metacognitive reflection. Spaced Repetition Systems (SRS) use de-contextualized review, overlooking the role of context, while Personal Knowledge Management (PKM) tools require high manual maintenance.
To address these challenges, this paper introduces "Insight Recall," a novel paradigm that conceptualizes the context-triggered retrieval of personal past insights as a metacognitive scaffold to promote SRL. We formalize this paradigm using the Just-in-Time Adaptive Intervention (JITAI) framework and implement a prototype system, Irec, to demonstrate its feasibility. At its core, Irec uses a dynamic knowledge graph of the user's learning history. When a user faces a new problem, a hybrid retrieval engine recalls relevant personal "insights." Subsequently, a large language model (LLM) performs a deep similarity assessment to filter and present the most relevant scaffold in a just-in-time manner. To reduce cognitive load, Irec features a human-in-the-loop pipeline for LLM-based knowledge graph construction. We also propose an optional "Guided Inquiry" module, where users can engage in a Socratic dialogue with an expert LLM, using the current problem and recalled insights as context. The contribution of this paper is a solid theoretical framework and a usable system platform for designing next-generation intelligent learning systems that enhance metacognition and self-regulation. 

---
# LSH-DynED: A Dynamic Ensemble Framework with LSH-Based Undersampling for Evolving Multi-Class Imbalanced Classification 

**Authors**: Soheil Abadifard, Fazli Can  

**Link**: [PDF](https://arxiv.org/pdf/2506.20041)  

**Abstract**: The classification of imbalanced data streams, which have unequal class distributions, is a key difficulty in machine learning, especially when dealing with multiple classes. While binary imbalanced data stream classification tasks have received considerable attention, only a few studies have focused on multi-class imbalanced data streams. Effectively managing the dynamic imbalance ratio is a key challenge in this domain. This study introduces a novel, robust, and resilient approach to address these challenges by integrating Locality Sensitive Hashing with Random Hyperplane Projections (LSH-RHP) into the Dynamic Ensemble Diversification (DynED) framework. To the best of our knowledge, we present the first application of LSH-RHP for undersampling in the context of imbalanced non-stationary data streams. The proposed method undersamples the majority classes by utilizing LSH-RHP, provides a balanced training set, and improves the ensemble's prediction performance. We conduct comprehensive experiments on 23 real-world and ten semi-synthetic datasets and compare LSH-DynED with 15 state-of-the-art methods. The results reveal that LSH-DynED outperforms other approaches in terms of both Kappa and mG-Mean effectiveness measures, demonstrating its capability in dealing with multi-class imbalanced non-stationary data streams. Notably, LSH-DynED performs well in large-scale, high-dimensional datasets with considerable class imbalances and demonstrates adaptation and robustness in real-world circumstances. To motivate our design, we review existing methods for imbalanced data streams, outline key challenges, and offer guidance for future work. For the reproducibility of our results, we have made our implementation available on GitHub. 

---
