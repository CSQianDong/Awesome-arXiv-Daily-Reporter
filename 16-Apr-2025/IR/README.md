# Evaluation Report on MCP Servers 

**Authors**: Zhiling Luo, Xiaorong Shi, Xuanrui Lin, Jinyang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.11094)  

**Abstract**: With the rise of LLMs, a large number of Model Context Protocol (MCP) services have emerged since the end of 2024. However, the effectiveness and efficiency of MCP servers have not been well studied. To study these questions, we propose an evaluation framework, called MCPBench. We selected several widely used MCP server and conducted an experimental evaluation on their accuracy, time, and token usage. Our experiments showed that the most effective MCP, Bing Web Search, achieved an accuracy of 64%. Importantly, we found that the accuracy of MCP servers can be substantially enhanced by involving declarative interface. This research paves the way for further investigations into optimized MCP implementations, ultimately leading to better AI-driven applications and data retrieval solutions. 

---
# Document Quality Scoring for Web Crawling 

**Authors**: Francesca Pezzuti, Ariane Mueller, Sean MacAvaney, Nicola Tonellotto  

**Link**: [PDF](https://arxiv.org/pdf/2504.11011)  

**Abstract**: The internet contains large amounts of low-quality content, yet users expect web search engines to deliver high-quality, relevant results. The abundant presence of low-quality pages can negatively impact retrieval and crawling processes by wasting resources on these documents. Therefore, search engines can greatly benefit from techniques that leverage efficient quality estimation methods to mitigate these negative impacts. Quality scoring methods for web pages are useful for many processes typical for web search systems, including static index pruning, index tiering, and crawling. Building on work by Chang et al.~\cite{chang2024neural}, who proposed using neural estimators of semantic quality for static index pruning, we extend their approach and apply their neural quality scorers to assess the semantic quality of web pages in crawling prioritisation tasks. In our experimental analysis, we found that prioritising semantically high-quality pages over low-quality ones can improve downstream search effectiveness. Our software contribution consists of a Docker container that computes an effective quality score for a given web page, allowing the quality scorer to be easily included and used in other components of web search systems. 

---
# Why am I seeing this? Towards recognizing social media recommender systems with missing recommendations 

**Authors**: Sabrina Guidotti, Sabrina Patania, Giuseppe Vizzari, Dimitri Ognibene, Gregor Donabauer, Udo Kruschwitz, Davide Taibi  

**Link**: [PDF](https://arxiv.org/pdf/2504.11000)  

**Abstract**: Social media plays a crucial role in shaping society, often amplifying polarization and spreading misinformation. These effects stem from complex dynamics involving user interactions, individual traits, and recommender algorithms driving content selection. Recommender systems, which significantly shape the content users see and decisions they make, offer an opportunity for intervention and regulation. However, assessing their impact is challenging due to algorithmic opacity and limited data availability. To effectively model user decision-making, it is crucial to recognize the recommender system adopted by the platform.
This work introduces a method for Automatic Recommender Recognition using Graph Neural Networks (GNNs), based solely on network structure and observed behavior. To infer the hidden recommender, we first train a Recommender Neutral User model (RNU) using a GNN and an adapted hindsight academic network recommender, aiming to reduce reliance on the actual recommender in the data. We then generate several Recommender Hypothesis-specific Synthetic Datasets (RHSD) by combining the RNU with different known recommenders, producing ground truths for testing. Finally, we train Recommender Hypothesis-specific User models (RHU) under various hypotheses and compare each candidate with the original used to generate the RHSD.
Our approach enables accurate detection of hidden recommenders and their influence on user behavior. Unlike audit-based methods, it captures system behavior directly, without ad hoc experiments that often fail to reflect real platforms. This study provides insights into how recommenders shape behavior, aiding efforts to reduce polarization and misinformation. 

---
# MSCRS: Multi-modal Semantic Graph Prompt Learning Framework for Conversational Recommender Systems 

**Authors**: Yibiao Wei, Jie Zou, Weikang Guo, Guoqing Wang, Xing Xu, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10921)  

**Abstract**: Conversational Recommender Systems (CRSs) aim to provide personalized recommendations by interacting with users through conversations. Most existing studies of CRS focus on extracting user preferences from conversational contexts. However, due to the short and sparse nature of conversational contexts, it is difficult to fully capture user preferences by conversational contexts only. We argue that multi-modal semantic information can enrich user preference expressions from diverse dimensions (e.g., a user preference for a certain movie may stem from its magnificent visual effects and compelling storyline). In this paper, we propose a multi-modal semantic graph prompt learning framework for CRS, named MSCRS. First, we extract textual and image features of items mentioned in the conversational contexts. Second, we capture higher-order semantic associations within different semantic modalities (collaborative, textual, and image) by constructing modality-specific graph structures. Finally, we propose an innovative integration of multi-modal semantic graphs with prompt learning, harnessing the power of large language models to comprehensively explore high-dimensional semantic relationships. Experimental results demonstrate that our proposed method significantly improves accuracy in item recommendation, as well as generates more natural and contextually relevant content in response generation. We have released the code and the expanded multi-modal CRS datasets to facilitate further exploration in related research\footnote{this https URL}. 

---
# CSPLADE: Learned Sparse Retrieval with Causal Language Models 

**Authors**: Zhichao Xu, Aosong Feng, Yijun Tian, Haibo Ding, Lin Leee Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2504.10816)  

**Abstract**: In recent years, dense retrieval has been the focus of information retrieval (IR) research. While effective, dense retrieval produces uninterpretable dense vectors, and suffers from the drawback of large index size. Learned sparse retrieval (LSR) has emerged as promising alternative, achieving competitive retrieval performance while also being able to leverage the classical inverted index data structure for efficient retrieval. However, limited works have explored scaling LSR beyond BERT scale. In this work, we identify two challenges in training large language models (LLM) for LSR: (1) training instability during the early stage of contrastive training; (2) suboptimal performance due to pre-trained LLM's unidirectional attention. To address these challenges, we propose two corresponding techniques: (1) a lightweight adaptation training phase to eliminate training instability; (2) two model variants to enable bidirectional information. With these techniques, we are able to train LSR models with 8B scale LLM, and achieve competitive retrieval performance with reduced index size. Furthermore, we are among the first to analyze the performance-efficiency tradeoff of LLM-based LSR model through the lens of model quantization. Our findings provide insights into adapting LLMs for efficient retrieval modeling. 

---
# Epistemic Uncertainty-aware Recommendation Systems via Bayesian Deep Ensemble Learning 

**Authors**: Radin Cheraghi, Amir Mohammad Mahfoozi, Sepehr Zolfaghari, Mohammadshayan Shabani, Maryam Ramezani, Hamid R. Rabiee  

**Link**: [PDF](https://arxiv.org/pdf/2504.10753)  

**Abstract**: Recommending items to users has long been a fundamental task, and studies have tried to improve it ever since. Most well-known models commonly employ representation learning to map users and items into a unified embedding space for matching assessment. These approaches have primary limitations, especially when dealing with explicit feedback and sparse data contexts. Two primary limitations are their proneness to overfitting and failure to incorporate epistemic uncertainty in predictions. To address these problems, we propose a novel Bayesian Deep Ensemble Collaborative Filtering method named BDECF. To improve model generalization and quality, we utilize Bayesian Neural Networks, which incorporate uncertainty within their weight parameters. In addition, we introduce a new interpretable non-linear matching approach for the user and item embeddings, leveraging the advantages of the attention mechanism. Furthermore, we endorse the implementation of an ensemble-based supermodel to generate more robust and reliable predictions, resulting in a more complete model. Empirical evaluation through extensive experiments and ablation studies across a range of publicly accessible real-world datasets with differing sparsity characteristics confirms our proposed method's effectiveness and the importance of its components. 

---
# Enhancing Document Retrieval for Curating N-ary Relations in Knowledge Bases 

**Authors**: Xing David Wang, Ulf Leser  

**Link**: [PDF](https://arxiv.org/pdf/2504.10613)  

**Abstract**: Curation of biomedical knowledge bases (KBs) relies on extracting accurate multi-entity relational facts from the literature - a process that remains largely manual and expert-driven. An essential step in this workflow is retrieving documents that can support or complete partially observed n-ary relations. We present a neural retrieval model designed to assist KB curation by identifying documents that help fill in missing relation arguments and provide relevant contextual evidence.
To reduce dependence on scarce gold-standard training data, we exploit existing KB records to construct weakly supervised training sets. Our approach introduces two key technical contributions: (i) a layered contrastive loss that enables learning from noisy and incomplete relational structures, and (ii) a balanced sampling strategy that generates high-quality negatives from diverse KB records. On two biomedical retrieval benchmarks, our approach achieves state-of-the-art performance, outperforming strong baselines in NDCG@10 by 5.7 and 3.7 percentage points, respectively. 

---
# Integrating Textual Embeddings from Contrastive Learning with Generative Recommender for Enhanced Personalization 

**Authors**: Yijun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10545)  

**Abstract**: Recent advances in recommender systems have highlighted the complementary strengths of generative modeling and pretrained language models. We propose a hybrid framework that augments the Hierarchical Sequential Transduction Unit (HSTU) generative recommender with BLaIR -- a contrastive text embedding model. This integration enriches item representations with semantic signals from textual metadata while preserving HSTU's powerful sequence modeling capabilities.
We evaluate our method on two domains from the Amazon Reviews 2023 dataset, comparing it against the original HSTU and a variant that incorporates embeddings from OpenAI's state-of-the-art text-embedding-3-large model. While the OpenAI embedding model is likely trained on a substantially larger corpus with significantly more parameters, our lightweight BLaIR-enhanced approach -- pretrained on domain-specific data -- consistently achieves better performance, highlighting the effectiveness of contrastive text embeddings in compute-efficient settings. 

---
# Multi-Modal Hypergraph Enhanced LLM Learning for Recommendation 

**Authors**: Xu Guo, Tong Zhang, Yuanzhi Wang, Chenxu Wang, Fuyun Wang, Xudong Wang, Xiaoya Zhang, Xin Liu, Zhen Cui  

**Link**: [PDF](https://arxiv.org/pdf/2504.10541)  

**Abstract**: The burgeoning presence of Large Language Models (LLM) is propelling the development of personalized recommender systems. Most existing LLM-based methods fail to sufficiently explore the multi-view graph structure correlations inherent in recommendation scenarios. To this end, we propose a novel framework, Hypergraph Enhanced LLM Learning for multimodal Recommendation (HeLLM), designed to equip LLMs with the capability to capture intricate higher-order semantic correlations by fusing graph-level contextual signals with sequence-level behavioral patterns. In the recommender pre-training phase, we design a user hypergraph to uncover shared interest preferences among users and an item hypergraph to capture correlations within multimodal similarities among items. The hypergraph convolution and synergistic contrastive learning mechanism are introduced to enhance the distinguishability of learned representations. In the LLM fine-tuning phase, we inject the learned graph-structured embeddings directly into the LLM's architecture and integrate sequential features capturing each user's chronological behavior. This process enables hypergraphs to leverage graph-structured information as global context, enhancing the LLM's ability to perceive complex relational patterns and integrate multimodal information, while also modeling local temporal dynamics. Extensive experiments demonstrate the superiority of our proposed method over state-of-the-art baselines, confirming the advantages of fusing hypergraph-based context with sequential user behavior in LLMs for recommendation. 

---
# Distilling Transitional Pattern to Large Language Models for Multimodal Session-based Recommendation 

**Authors**: Jiajie Su, Qiyong Zhong, Yunshan Ma, Weiming Liu, Chaochao Chen, Xiaolin Zheng, Jianwei Yin, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.10538)  

**Abstract**: Session-based recommendation (SBR) predicts the next item based on anonymous sessions. Traditional SBR explores user intents based on ID collaborations or auxiliary content. To further alleviate data sparsity and cold-start issues, recent Multimodal SBR (MSBR) methods utilize simplistic pre-trained models for modality learning but have limitations in semantic richness. Considering semantic reasoning abilities of Large Language Models (LLM), we focus on the LLM-enhanced MSBR scenario in this paper, which leverages LLM cognition for comprehensive multimodal representation generation, to enhance downstream MSBR. Tackling this problem faces two challenges: i) how to obtain LLM cognition on both transitional patterns and inherent multimodal knowledge, ii) how to align both features into one unified LLM, minimize discrepancy while maximizing representation utility. To this end, we propose a multimodal LLM-enhanced framework TPAD, which extends a distillation paradigm to decouple and align transitional patterns for promoting MSBR. TPAD establishes parallel Knowledge-MLLM and Transfer-MLLM, where the former interprets item knowledge-reflected features and the latter extracts transition-aware features underneath sessions. A transitional pattern alignment module harnessing mutual information estimation theory unites two MLLMs, alleviating distribution discrepancy and distilling transitional patterns into modal representations. Extensive experiments on real-world datasets demonstrate the effectiveness of our framework. 

---
# HeteRAG: A Heterogeneous Retrieval-augmented Generation Framework with Decoupled Knowledge Representations 

**Authors**: Peiru Yang, Xintian Li, Zhiyang Hu, Jiapeng Wang, Jinhua Yin, Huili Wang, Lizhi He, Shuai Yang, Shangguang Wang, Yongfeng Huang, Tao Qi  

**Link**: [PDF](https://arxiv.org/pdf/2504.10529)  

**Abstract**: Retrieval-augmented generation (RAG) methods can enhance the performance of LLMs by incorporating retrieved knowledge chunks into the generation process. In general, the retrieval and generation steps usually have different requirements for these knowledge chunks. The retrieval step benefits from comprehensive information to improve retrieval accuracy, whereas excessively long chunks may introduce redundant contextual information, thereby diminishing both the effectiveness and efficiency of the generation process. However, existing RAG methods typically employ identical representations of knowledge chunks for both retrieval and generation, resulting in suboptimal performance. In this paper, we propose a heterogeneous RAG framework (\myname) that decouples the representations of knowledge chunks for retrieval and generation, thereby enhancing the LLMs in both effectiveness and efficiency. Specifically, we utilize short chunks to represent knowledge to adapt the generation step and utilize the corresponding chunk with its contextual information from multi-granular views to enhance retrieval accuracy. We further introduce an adaptive prompt tuning method for the retrieval model to adapt the heterogeneous retrieval augmented generation process. Extensive experiments demonstrate that \myname achieves significant improvements compared to baselines. 

---
# JEPA4Rec: Learning Effective Language Representations for Sequential Recommendation via Joint Embedding Predictive Architecture 

**Authors**: Minh-Anh Nguyen, Dung D.Le  

**Link**: [PDF](https://arxiv.org/pdf/2504.10512)  

**Abstract**: Language representation learning has emerged as a promising approach for sequential recommendation, thanks to its ability to learn generalizable representations. However, despite its advantages, this approach still struggles with data sparsity and a limited understanding of common-sense user preferences. To address these limitations, we propose $\textbf{JEPA4Rec}$, a framework that combines $\textbf{J}$oint $\textbf{E}$mbedding $\textbf{P}$redictive $\textbf{A}$rchitecture with language modeling of item textual descriptions. JEPA4Rec captures semantically rich and transferable representations, improving recommendation performance and reducing reliance on large-scale pre-training data. Specifically, JEPA4Rec represents items as text sentences by flattening descriptive information such as $\textit{title, category}$, and other attributes. To encode these sentences, we employ a bidirectional Transformer encoder with modified embedding layers tailored for capturing item information in recommendation datasets. We apply masking to text sentences and use them to predict the representations of the unmasked sentences, helping the model learn generalizable item embeddings. To further improve recommendation performance and language understanding, we employ a two-stage training strategy incorporating self-supervised learning losses. Experiments on six real-world datasets demonstrate that JEPA4Rec consistently outperforms state-of-the-art methods, particularly in cross-domain, cross-platform, and low-resource scenarios. 

---
# Beyond Reproducibility: Advancing Zero-shot LLM Reranking Efficiency with Setwise Insertion 

**Authors**: Jakub Podolak, Leon Peric, Mina Janicijevic, Roxana Petcu  

**Link**: [PDF](https://arxiv.org/pdf/2504.10509)  

**Abstract**: This study presents a comprehensive reproducibility and extension analysis of the Setwise prompting methodology for zero-shot ranking with Large Language Models (LLMs), as proposed by Zhuang et al. We evaluate its effectiveness and efficiency compared to traditional Pointwise, Pairwise, and Listwise approaches in document ranking tasks. Our reproduction confirms the findings of Zhuang et al., highlighting the trade-offs between computational efficiency and ranking effectiveness in Setwise methods. Building on these insights, we introduce Setwise Insertion, a novel approach that leverages the initial document ranking as prior knowledge, reducing unnecessary comparisons and uncertainty by focusing on candidates more likely to improve the ranking results. Experimental results across multiple LLM architectures (Flan-T5, Vicuna, and Llama) show that Setwise Insertion yields a 31% reduction in query time, a 23% reduction in model inferences, and a slight improvement in reranking effectiveness compared to the original Setwise method. These findings highlight the practical advantage of incorporating prior ranking knowledge into Setwise prompting for efficient and accurate zero-shot document reranking. 

---
# Poly-Vector Retrieval: Reference and Content Embeddings for Legal Documents 

**Authors**: João Alberto de Oliveira Lima  

**Link**: [PDF](https://arxiv.org/pdf/2504.10508)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as an effective paradigm for generating contextually accurate answers by integrating Large Language Models (LLMs) with retrieval mechanisms. However, in legal contexts, users frequently reference norms by their labels or nicknames (e.g., Article 5 of the Constitution or Consumer Defense Code (CDC)), rather than by their content, posing challenges for traditional RAG approaches that rely solely on semantic embeddings of text. Furthermore, legal texts themselves heavily rely on explicit cross-references (e.g., "pursuant to Article 34") that function as pointers. Both scenarios pose challenges for traditional RAG approaches that rely solely on semantic embeddings of text, often failing to retrieve the necessary referenced content. This paper introduces Poly-Vector Retrieval, a method assigning multiple distinct embeddings to each legal provision: one embedding captures the content (the full text), another captures the label (the identifier or proper name), and optionally additional embeddings capture alternative denominations. Inspired by Frege's distinction between Sense and Reference, this poly-vector retrieval approach treats labels, identifiers and reference markers as rigid designators and content embeddings as carriers of semantic substance. Experiments on the Brazilian Federal Constitution demonstrate that Poly-Vector Retrieval significantly improves retrieval accuracy for label-centric queries and potential to resolve internal and external cross-references, without compromising performance on purely semantic queries. The study discusses philosophical and practical implications of explicitly separating reference from content in vector embeddings and proposes future research directions for applying this approach to broader legal datasets and other domains characterized by explicit reference identifiers. 

---
# PinRec: Outcome-Conditioned, Multi-Token Generative Retrieval for Industry-Scale Recommendation Systems 

**Authors**: Anirudhan Badrinath, Prabhat Agarwal, Laksh Bhasin, Jaewon Yang, Jiajing Xu, Charles Rosenberg  

**Link**: [PDF](https://arxiv.org/pdf/2504.10507)  

**Abstract**: Generative retrieval methods utilize generative sequential modeling techniques, such as transformers, to generate candidate items for recommender systems. These methods have demonstrated promising results in academic benchmarks, surpassing traditional retrieval models like two-tower architectures. However, current generative retrieval methods lack the scalability required for industrial recommender systems, and they are insufficiently flexible to satisfy the multiple metric requirements of modern systems.
This paper introduces PinRec, a novel generative retrieval model developed for applications at Pinterest. PinRec utilizes outcome-conditioned generation, enabling modelers to specify how to balance various outcome metrics, such as the number of saves and clicks, to effectively align with business goals and user exploration. Additionally, PinRec incorporates multi-token generation to enhance output diversity while optimizing generation. Our experiments demonstrate that PinRec can successfully balance performance, diversity, and efficiency, delivering a significant positive impact to users using generative models. This paper marks a significant milestone in generative retrieval, as it presents, to our knowledge, the first rigorous study on implementing generative retrieval at the scale of Pinterest. 

---
# Human-Oriented Image Retrieval System (HORSE): A Neuro-Symbolic Approach to Optimizing Retrieval of Previewed Images 

**Authors**: Abraham Itzhak Weinberg  

**Link**: [PDF](https://arxiv.org/pdf/2504.10502)  

**Abstract**: Image retrieval remains a challenging task due to the complex interaction between human visual perception, memory, and computational processes. Current image search engines often struggle to efficiently retrieve images based on natural language descriptions, as they rely on time-consuming preprocessing, tagging, and machine learning pipelines. This paper introduces the Human-Oriented Retrieval Search Engine for Images (HORSE), a novel approach that leverages neuro-symbolic indexing to improve image retrieval by focusing on human-oriented indexing. By integrating cognitive science insights with advanced computational techniques, HORSE enhances the retrieval process, making it more aligned with how humans perceive, store, and recall visual information. The neuro-symbolic framework combines the strengths of neural networks and symbolic reasoning, mitigating their individual limitations. The proposed system optimizes image retrieval, offering a more intuitive and efficient solution for users. We discuss the design and implementation of HORSE, highlight its potential applications in fields such as design error detection and knowledge management, and suggest future directions for research to further refine the system's metrics and capabilities. 

---
# Leveraging Auto-Distillation and Generative Self-Supervised Learning in Residual Graph Transformers for Enhanced Recommender Systems 

**Authors**: Eya Mhedhbi, Youssef Mourchid, Alice Othmani  

**Link**: [PDF](https://arxiv.org/pdf/2504.10500)  

**Abstract**: This paper introduces a cutting-edge method for enhancing recommender systems through the integration of generative self-supervised learning (SSL) with a Residual Graph Transformer. Our approach emphasizes the importance of superior data enhancement through the use of pertinent pretext tasks, automated through rationale-aware SSL to distill clear ways of how users and items interact. The Residual Graph Transformer incorporates a topology-aware transformer for global context and employs residual connections to improve graph representation learning. Additionally, an auto-distillation process refines self-supervised signals to uncover consistent collaborative rationales. Experimental evaluations on multiple datasets demonstrate that our approach consistently outperforms baseline methods. 

---
# Graph-based Approaches and Functionalities in Retrieval-Augmented Generation: A Comprehensive Survey 

**Authors**: Zulun Zhu, Tiancheng Huang, Kai Wang, Junda Ye, Xinghe Chen, Siqiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2504.10499)  

**Abstract**: Large language models (LLMs) struggle with the factual error during inference due to the lack of sufficient training data and the most updated knowledge, leading to the hallucination problem. Retrieval-Augmented Generation (RAG) has gained attention as a promising solution to address the limitation of LLMs, by retrieving relevant information from external source to generate more accurate answers to the questions. Given the pervasive presence of structured knowledge in the external source, considerable strides in RAG have been made to employ the techniques related to graphs and achieve more complex reasoning based on the topological information between knowledge entities. However, there is currently neither unified review examining the diverse roles of graphs in RAG, nor a comprehensive resource to help researchers navigate and contribute to this evolving field. This survey offers a novel perspective on the functionality of graphs within RAG and their impact on enhancing performance across a wide range of graph-structured data. It provides a detailed breakdown of the roles that graphs play in RAG, covering database construction, algorithms, pipelines, and tasks. Finally, it identifies current challenges and outline future research directions, aiming to inspire further developments in this field. Our graph-centered analysis highlights the commonalities and differences in existing methods, setting the stage for future researchers in areas such as graph learning, database systems, and natural language processing. 

---
# CCSK:Cognitive Convection of Self-Knowledge Based Retrieval Augmentation for Large Language Models 

**Authors**: Jianling Lu, Mingqi Lv  

**Link**: [PDF](https://arxiv.org/pdf/2504.10498)  

**Abstract**: The performance of large language models (LLMs) in Q&A task increased substantially through Retrieval-Augmented Generation (RAG) which brings in external knowledge. However, the main difficulty lies in balancing the inherent self-knowledge of LLMs with external information retrieval (IR). The current threshold-based methods apply one-dimensional static mechanisms with single criterion. As a result, their IR decisions might be irrelevant to the LLMs' response under difficult queries. To alleviate this problem, we propose Cognitive Convection of Self-Knowledge (CCSK). Different from traditional methods that maintain single fixed IR activation criteria, CCSK implements a dynamic joint decision process via a Siamese Network module and a Response Quality Model. The Siamese Network calculates the cosine similarity between the current query and the historical queries. The Response Quality Model evaluates the responses of LLMs through LightGBM. The final decision of the CCSK is derived from the outputs of the two modules, as well as text features fused using a multi-head attention mechanism. Extensive experiments on real-world datasets show that CCSK significantly enhances the model's effectiveness in information retrieval. 

---
# Exploring Generative AI Techniques in Government: A Case Study 

**Authors**: Sunyi Liu, Mengzhe Geng, Rebecca Hart  

**Link**: [PDF](https://arxiv.org/pdf/2504.10497)  

**Abstract**: The swift progress of Generative Artificial intelligence (GenAI), notably Large Language Models (LLMs), is reshaping the digital landscape. Recognizing this transformative potential, the National Research Council of Canada (NRC) launched a pilot initiative to explore the integration of GenAI techniques into its daily operation for performance excellence, where 22 projects were launched in May 2024. Within these projects, this paper presents the development of the intelligent agent Pubbie as a case study, targeting the automation of performance measurement, data management and insight reporting at the NRC. Cutting-edge techniques are explored, including LLM orchestration and semantic embedding via RoBERTa, while strategic fine-tuning and few-shot learning approaches are incorporated to infuse domain knowledge at an affordable cost. The user-friendly interface of Pubbie allows general government users to input queries in natural language and easily upload or download files with a simple button click, greatly reducing manual efforts and accessibility barriers. 

---
# ArxivBench: Can LLMs Assist Researchers in Conducting Research? 

**Authors**: Ning Li, Jingran Zhang, Justin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2504.10496)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable effectiveness in completing various tasks such as reasoning, translation, and question answering. However the issue of factual incorrect content in LLM-generated responses remains a persistent challenge. In this study, we evaluate both proprietary and open-source LLMs on their ability to respond with relevant research papers and accurate links to articles hosted on the arXiv platform, based on high level prompts. To facilitate this evaluation, we introduce arXivBench, a benchmark specifically designed to assess LLM performance across eight major subject categories on arXiv and five subfields within computer science, one of the most popular categories among them. Our findings reveal a concerning accuracy of LLM-generated responses depending on the subject, with some subjects experiencing significantly lower accuracy than others. Notably, Claude-3.5-Sonnet exhibits a substantial advantage in generating both relevant and accurate responses. And interestingly, most LLMs achieve a much higher accuracy in the Artificial Intelligence sub-field than other sub-fields. This benchmark provides a standardized tool for evaluating the reliability of LLM-generated scientific responses, promoting more dependable use of LLMs in academic and research environments. Our code is open-sourced at this https URL and our dataset is available on huggingface at this https URL. 

---
# Bipartite Ranking From Multiple Labels: On Loss Versus Label Aggregation 

**Authors**: Michal Lukasik, Lin Chen, Harikrishna Narasimhan, Aditya Krishna Menon, Wittawat Jitkrittum, Felix X. Yu, Sashank J. Reddi, Gang Fu, Mohammadhossein Bateni, Sanjiv Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2504.11284)  

**Abstract**: Bipartite ranking is a fundamental supervised learning problem, with the goal of learning a ranking over instances with maximal area under the ROC curve (AUC) against a single binary target label. However, one may often observe multiple binary target labels, e.g., from distinct human annotators. How can one synthesize such labels into a single coherent ranking? In this work, we formally analyze two approaches to this problem -- loss aggregation and label aggregation -- by characterizing their Bayes-optimal solutions. Based on this, we show that while both methods can yield Pareto-optimal solutions, loss aggregation can exhibit label dictatorship: one can inadvertently (and undesirably) favor one label over others. This suggests that label aggregation can be preferable to loss aggregation, which we empirically verify. 

---
# Efficient Distributed Retrieval-Augmented Generation for Enhancing Language Model Performance 

**Authors**: Shangyu Liu, Zhenzhe Zheng, Xiaoyao Huang, Fan Wu, Jie Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.11197)  

**Abstract**: Small language models (SLMs) support efficient deployments on resource-constrained edge devices, but their limited capacity compromises inference performance. Retrieval-augmented generation (RAG) is a promising solution to enhance model performance by integrating external databases, without requiring intensive on-device model retraining. However, large-scale public databases and user-specific private contextual documents are typically located on the cloud and the device separately, while existing RAG implementations are primarily centralized. To bridge this gap, we propose DRAGON, a distributed RAG framework to enhance on-device SLMs through both general and personal knowledge without the risk of leaking document privacy. Specifically, DRAGON decomposes multi-document RAG into multiple parallel token generation processes performed independently and locally on the cloud and the device, and employs a newly designed Speculative Aggregation, a dual-side speculative algorithm to avoid frequent output synchronization between the cloud and device. A new scheduling algorithm is further introduced to identify the optimal aggregation side based on real-time network conditions. Evaluations on real-world hardware testbed demonstrate a significant performance improvement of DRAGON-up to 1.9x greater gains over standalone SLM compared to the centralized RAG, substantial reduction in per-token latency, and negligible Time to First Token (TTFT) overhead. 

---
