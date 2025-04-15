# Invariance Matters: Empowering Social Recommendation via Graph Invariant Learning 

**Authors**: Yonghui Yang, Le Wu, Yuxin Liao, Zhuangzhuang He, Pengyang Shao, Richang Hong, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10432)  

**Abstract**: Graph-based social recommendation systems have shown significant promise in enhancing recommendation performance, particularly in addressing the issue of data sparsity in user behaviors. Typically, these systems leverage Graph Neural Networks (GNNs) to capture user preferences by incorporating high-order social influences from observed social networks. However, existing graph-based social recommendations often overlook the fact that social networks are inherently noisy, containing task-irrelevant relationships that can hinder accurate user preference learning. The removal of these redundant social relations is crucial, yet it remains challenging due to the lack of ground truth. In this paper, we approach the social denoising problem from the perspective of graph invariant learning and propose a novel method, Social Graph Invariant Learning(SGIL). Specifically,SGIL aims to uncover stable user preferences within the input social graph, thereby enhancing the robustness of graph-based social recommendation systems. To achieve this goal, SGIL first simulates multiple noisy social environments through graph generators. It then seeks to learn environment-invariant user preferences by minimizing invariant risk across these environments. To further promote diversity in the generated social environments, we employ an adversarial training strategy to simulate more potential social noisy distributions. Extensive experimental results demonstrate the effectiveness of the proposed SGIL. The code is available at this https URL. 

---
# Brain-Machine Interfaces & Information Retrieval Challenges and Opportunities 

**Authors**: Yashar Moshfeghi, Niall McGuire  

**Link**: [PDF](https://arxiv.org/pdf/2504.10371)  

**Abstract**: The fundamental goal of Information Retrieval (IR) systems lies in their capacity to effectively satisfy human information needs - a challenge that encompasses not just the technical delivery of information, but the nuanced understanding of human cognition during information seeking. Contemporary IR platforms rely primarily on observable interaction signals, creating a fundamental gap between system capabilities and users' cognitive processes. Brain-Machine Interface (BMI) technologies now offer unprecedented potential to bridge this gap through direct measurement of previously inaccessible aspects of information-seeking behaviour. This perspective paper offers a broad examination of the IR landscape, providing a comprehensive analysis of how BMI technology could transform IR systems, drawing from advances at the intersection of both neuroscience and IR research. We present our analysis through three identified fundamental vertices: (1) understanding the neural correlates of core IR concepts to advance theoretical models of search behaviour, (2) enhancing existing IR systems through contextual integration of neurophysiological signals, and (3) developing proactive IR capabilities through direct neurophysiological measurement. For each vertex, we identify specific research opportunities and propose concrete directions for developing BMI-enhanced IR systems. We conclude by examining critical technical and ethical challenges in implementing these advances, providing a structured roadmap for future research at the intersection of neuroscience and IR. 

---
# CROSSAN: Towards Efficient and Effective Adaptation of Multiple Multimodal Foundation Models for Sequential Recommendation 

**Authors**: Junchen Fu, Yongxin Ni, Joemon M. Jose, Ioannis Arapakis, Kaiwen Zheng, Youhua Li, Xuri Ge  

**Link**: [PDF](https://arxiv.org/pdf/2504.10307)  

**Abstract**: Multimodal Foundation Models (MFMs) excel at representing diverse raw modalities (e.g., text, images, audio, videos, etc.). As recommender systems increasingly incorporate these modalities, leveraging MFMs to generate better representations has great potential. However, their application in sequential recommendation remains largely unexplored. This is primarily because mainstream adaptation methods, such as Fine-Tuning and even Parameter-Efficient Fine-Tuning (PEFT) techniques (e.g., Adapter and LoRA), incur high computational costs, especially when integrating multiple modality encoders, thus hindering research progress. As a result, it remains unclear whether we can efficiently and effectively adapt multiple (>2) MFMs for the sequential recommendation task.
To address this, we propose a plug-and-play Cross-modal Side Adapter Network (CROSSAN). Leveraging the fully decoupled side adapter-based paradigm, CROSSAN achieves high efficiency while enabling cross-modal learning across diverse modalities. To optimize the final stage of multimodal fusion across diverse modalities, we adopt the Mixture of Modality Expert Fusion (MOMEF) mechanism. CROSSAN achieves superior performance on the public datasets for adapting four foundation models with raw modalities. Performance consistently improves as more MFMs are adapted. We will release our code and datasets to facilitate future research. 

---
# MURR: Model Updating with Regularized Replay for Searching a Document Stream 

**Authors**: Eugene Yang, Nicola Tonellotto, Dawn Lawrie, Sean MacAvaney, James Mayfield, Douglas W. Oard, Scott Miller  

**Link**: [PDF](https://arxiv.org/pdf/2504.10250)  

**Abstract**: The Internet produces a continuous stream of new documents and user-generated queries. These naturally change over time based on events in the world and the evolution of language. Neural retrieval models that were trained once on a fixed set of query-document pairs will quickly start misrepresenting newly-created content and queries, leading to less effective retrieval. Traditional statistical sparse retrieval can update collection statistics to reflect these changes in the use of language in documents and queries. In contrast, continued fine-tuning of the language model underlying neural retrieval approaches such as DPR and ColBERT creates incompatibility with previously-encoded documents. Re-encoding and re-indexing all previously-processed documents can be costly. In this work, we explore updating a neural dual encoder retrieval model without reprocessing past documents in the stream. We propose MURR, a model updating strategy with regularized replay, to ensure the model can still faithfully search existing documents without reprocessing, while continuing to update the model for the latest topics. In our simulated streaming environments, we show that fine-tuning models using MURR leads to more effective and more consistent retrieval results than other strategies as the stream of documents and queries progresses. 

---
# From Prompting to Alignment: A Generative Framework for Query Recommendation 

**Authors**: Erxue Min, Hsiu-Yuan Huang, Min Yang, Xihong Yang, Xin Jia, Yunfang Wu, Hengyi Cai, Shuaiqiang Wang, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2504.10208)  

**Abstract**: In modern search systems, search engines often suggest relevant queries to users through various panels or components, helping refine their information needs. Traditionally, these recommendations heavily rely on historical search logs to build models, which suffer from cold-start or long-tail issues. Furthermore, tasks such as query suggestion, completion or clarification are studied separately by specific design, which lacks generalizability and hinders adaptation to novel applications. Despite recent attempts to explore the use of LLMs for query recommendation, these methods mainly rely on the inherent knowledge of LLMs or external sources like few-shot examples, retrieved documents, or knowledge bases, neglecting the importance of the calibration and alignment with user feedback, thus limiting their practical utility. To address these challenges, we first propose a general Generative Query Recommendation (GQR) framework that aligns LLM-based query generation with user preference. Specifically, we unify diverse query recommendation tasks by a universal prompt framework, leveraging the instruct-following capability of LLMs for effective generation. Secondly, we align LLMs with user feedback via presenting a CTR-alignment framework, which involves training a query-wise CTR predictor as a process reward model and employing list-wise preference alignment to maximize the click probability of the generated query list. Furthermore, recognizing the inconsistency between LLM knowledge and proactive search intents arising from the separation of user-initiated queries from models, we align LLMs with user initiative via retrieving co-occurrence queries as side information when historical logs are available. 

---
# HistLLM: A Unified Framework for LLM-Based Multimodal Recommendation with User History Encoding and Compression 

**Authors**: Chen Zhang, Bo Hu, Weidong Chen, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.10150)  

**Abstract**: While large language models (LLMs) have proven effective in leveraging textual data for recommendations, their application to multimodal recommendation tasks remains relatively underexplored. Although LLMs can process multimodal information through projection functions that map visual features into their semantic space, recommendation tasks often require representing users' history interactions through lengthy prompts combining text and visual elements, which not only hampers training and inference efficiency but also makes it difficult for the model to accurately capture user preferences from complex and extended prompts, leading to reduced recommendation performance. To address this challenge, we introduce HistLLM, an innovative multimodal recommendation framework that integrates textual and visual features through a User History Encoding Module (UHEM), compressing multimodal user history interactions into a single token representation, effectively facilitating LLMs in processing user preferences. Extensive experiments demonstrate the effectiveness and efficiency of our proposed mechanism. 

---
# A Survey of Personalization: From RAG to Agent 

**Authors**: Xiaopeng Li, Pengyue Jia, Derong Xu, Yi Wen, Yingyi Zhang, Wenlin Zhang, Wanyu Wang, Yichao Wang, Zhaocheng Du, Xiangyang Li, Yong Liu, Huifeng Guo, Ruiming Tang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2504.10147)  

**Abstract**: Personalization has become an essential capability in modern AI systems, enabling customized interactions that align with individual user preferences, contexts, and goals. Recent research has increasingly concentrated on Retrieval-Augmented Generation (RAG) frameworks and their evolution into more advanced agent-based architectures within personalized settings to enhance user satisfaction. Building on this foundation, this survey systematically examines personalization across the three core stages of RAG: pre-retrieval, retrieval, and generation. Beyond RAG, we further extend its capabilities into the realm of Personalized LLM-based Agents, which enhance traditional RAG systems with agentic functionalities, including user understanding, personalized planning and execution, and dynamic generation. For both personalization in RAG and agent-based personalization, we provide formal definitions, conduct a comprehensive review of recent literature, and summarize key datasets and evaluation metrics. Additionally, we discuss fundamental challenges, limitations, and promising research directions in this evolving field. Relevant papers and resources are continuously updated at this https URL. 

---
# Unveiling Contrastive Learning's Capability of Neighborhood Aggregation for Collaborative Filtering 

**Authors**: Yu Zhang, Yiwen Zhang, Yi Zhang, Lei Sang, Yun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10113)  

**Abstract**: Personalized recommendation is widely used in the web applications, and graph contrastive learning (GCL) has gradually become a dominant approach in recommender systems, primarily due to its ability to extract self-supervised signals from raw interaction data, effectively alleviating the problem of data sparsity. A classic GCL-based method typically uses data augmentation during graph convolution to generates more contrastive views, and performs contrast on these new views to obtain rich self-supervised signals. Despite this paradigm is effective, the reasons behind the performance gains remain a mystery. In this paper, we first reveal via theoretical derivation that the gradient descent process of the CL objective is formally equivalent to graph convolution, which implies that CL objective inherently supports neighborhood aggregation on interaction graphs. We further substantiate this capability through experimental validation and identify common misconceptions in the selection of positive samples in previous methods, which limit the potential of CL objective. Based on this discovery, we propose the Light Contrastive Collaborative Filtering (LightCCF) method, which introduces a novel neighborhood aggregation objective to bring users closer to all interacted items while pushing them away from other positive pairs, thus achieving high-quality neighborhood aggregation with very low time complexity. On three highly sparse public datasets, the proposed method effectively aggregate neighborhood information while preventing graph over-smoothing, demonstrating significant improvements over existing GCL-based counterparts in both training efficiency and recommendation accuracy. Our implementations are publicly accessible. 

---
# Enhancing LLM-based Recommendation through Semantic-Aligned Collaborative Knowledge 

**Authors**: Zihan Wang, Jinghao Lin, Xiaocui Yang, Yongkang Liu, Shi Feng, Daling Wang, Yifei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10107)  

**Abstract**: Large Language Models (LLMs) demonstrate remarkable capabilities in leveraging comprehensive world knowledge and sophisticated reasoning mechanisms for recommendation tasks. However, a notable limitation lies in their inability to effectively model sparse identifiers (e.g., user and item IDs), unlike conventional collaborative filtering models (Collabs.), thus hindering LLM to learn distinctive user-item representations and creating a performance bottleneck. Prior studies indicate that integrating collaborative knowledge from Collabs. into LLMs can mitigate the above limitations and enhance their recommendation performance. Nevertheless, the significant discrepancy in knowledge distribution and semantic space between LLMs and Collab. presents substantial challenges for effective knowledge transfer. To tackle these challenges, we propose a novel framework, SeLLa-Rec, which focuses on achieving alignment between the semantic spaces of Collabs. and LLMs. This alignment fosters effective knowledge fusion, mitigating the influence of discriminative noise and facilitating the deep integration of knowledge from diverse models. Specifically, three special tokens with collaborative knowledge are embedded into the LLM's semantic space through a hybrid projection layer and integrated into task-specific prompts to guide the recommendation process. Experiments conducted on two public benchmark datasets (MovieLens-1M and Amazon Book) demonstrate that SeLLa-Rec achieves state-of-the-art performance. 

---
# On Precomputation and Caching in Information Retrieval Experiments with Pipeline Architectures 

**Authors**: Sean MacAvaney, Craig Macdonald  

**Link**: [PDF](https://arxiv.org/pdf/2504.09984)  

**Abstract**: Modern information retrieval systems often rely on multiple components executed in a pipeline. In a research setting, this can lead to substantial redundant computations (e.g., retrieving the same query multiple times for evaluating different downstream rerankers). To overcome this, researchers take cached "result" files as inputs, which represent the output of another pipeline. However, these result files can be brittle and can cause a disconnect between the conceptual design of the pipeline and its logical implementation. To overcome both the redundancy problem (when executing complete pipelines) and the disconnect problem (when relying on intermediate result files), we describe our recent efforts to improve the caching capabilities in the open-source PyTerrier IR platform. We focus on two main directions: (1) automatic implicit caching of common pipeline prefixes when comparing systems and (2) explicit caching of operations through a new extension package, pyterrier-caching. These approaches allow for the best of both worlds: pipelines can be fully expressed end-to-end, while also avoiding redundant computations between pipelines. 

---
# Constrained Auto-Regressive Decoding Constrains Generative Retrieval 

**Authors**: Shiguang Wu, Zhaochun Ren, Xin Xin, Jiyuan Yang, Mengqi Zhang, Zhumin Chen, Maarten de Rijke, Pengjie Ren  

**Link**: [PDF](https://arxiv.org/pdf/2504.09935)  

**Abstract**: Generative retrieval seeks to replace traditional search index data structures with a single large-scale neural network, offering the potential for improved efficiency and seamless integration with generative large language models. As an end-to-end paradigm, generative retrieval adopts a learned differentiable search index to conduct retrieval by directly generating document identifiers through corpus-specific constrained decoding. The generalization capabilities of generative retrieval on out-of-distribution corpora have gathered significant attention.
In this paper, we examine the inherent limitations of constrained auto-regressive generation from two essential perspectives: constraints and beam search. We begin with the Bayes-optimal setting where the generative retrieval model exactly captures the underlying relevance distribution of all possible documents. Then we apply the model to specific corpora by simply adding corpus-specific constraints. Our main findings are two-fold: (i) For the effect of constraints, we derive a lower bound of the error, in terms of the KL divergence between the ground-truth and the model-predicted step-wise marginal distributions. (ii) For the beam search algorithm used during generation, we reveal that the usage of marginal distributions may not be an ideal approach. This paper aims to improve our theoretical understanding of the generalization capabilities of the auto-regressive decoding retrieval paradigm, laying a foundation for its limitations and inspiring future advancements toward more robust and generalizable generative retrieval. 

---
# StePO-Rec: Towards Personalized Outfit Styling Assistant via Knowledge-Guided Multi-Step Reasoning 

**Authors**: Yuxi Bi, Yunfan Gao, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09915)  

**Abstract**: Advancements in Generative AI offers new opportunities for FashionAI, surpassing traditional recommendation systems that often lack transparency and struggle to integrate expert knowledge, leaving the potential for personalized fashion styling remain untapped. To address these challenges, we present PAFA (Principle-Aware Fashion), a multi-granular knowledge base that organizes professional styling expertise into three levels of metadata, domain principles, and semantic relationships. Using PAFA, we develop StePO-Rec, a knowledge-guided method for multi-step outfit recommendation. StePO-Rec provides structured suggestions using a scenario-dimension-attribute framework, employing recursive tree construction to align recommendations with both professional principles and individual preferences. A preference-trend re-ranking system further adapts to fashion trends while maintaining the consistency of the user's original style. Experiments on the widely used personalized outfit dataset IQON show a 28% increase in Recall@1 and 32.8% in MAP. Furthermore, case studies highlight improved explainability, traceability, result reliability, and the seamless integration of expertise and personalization. 

---
# Constructing Micro Knowledge Graphs from Technical Support Documents 

**Authors**: Atul Kumar, Nisha Gupta, Saswati Dana  

**Link**: [PDF](https://arxiv.org/pdf/2504.09877)  

**Abstract**: Short technical support pages such as IBM Technotes are quite common in technical support domain. These pages can be very useful as the knowledge sources for technical support applications such as chatbots, search engines and question-answering (QA) systems. Information extracted from documents to drive technical support applications is often stored in the form of Knowledge Graph (KG). Building KGs from a large corpus of documents poses a challenge of granularity because a large number of entities and actions are present in each page. The KG becomes virtually unusable if all entities and actions from these pages are stored in the KG. Therefore, only key entities and actions from each page are extracted and stored in the KG. This approach however leads to loss of knowledge represented by entities and actions left out of the KG as they are no longer available to graph search and reasoning functions. We propose a set of techniques to create micro knowledge graph (micrograph) for each of such web pages. The micrograph stores all the entities and actions in a page and also takes advantage of the structure of the page to represent exactly in which part of that page these entities and actions appeared, and also how they relate to each other. These micrographs can be used as additional knowledge sources by technical support applications. We define schemas for representing semi-structured and plain text knowledge present in the technical support web pages. Solutions in technical support domain include procedures made of steps. We also propose a technique to extract procedures from these webpages and the schemas to represent them in the micrographs. We also discuss how technical support applications can take advantage of the micrographs. 

---
# RAKG:Document-level Retrieval Augmented Knowledge Graph Construction 

**Authors**: Hairong Zhang, Jiaheng Si, Guohang Yan, Boyuan Qi, Pinlong Cai, Song Mao, Ding Wang, Botian Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.09823)  

**Abstract**: With the rise of knowledge graph based retrieval-augmented generation (RAG) techniques such as GraphRAG and Pike-RAG, the role of knowledge graphs in enhancing the reasoning capabilities of large language models (LLMs) has become increasingly prominent. However, traditional Knowledge Graph Construction (KGC) methods face challenges like complex entity disambiguation, rigid schema definition, and insufficient cross-document knowledge integration. This paper focuses on the task of automatic document-level knowledge graph construction. It proposes the Document-level Retrieval Augmented Knowledge Graph Construction (RAKG) framework. RAKG extracts pre-entities from text chunks and utilizes these pre-entities as queries for RAG, effectively addressing the issue of long-context forgetting in LLMs and reducing the complexity of Coreference Resolution. In contrast to conventional KGC methods, RAKG more effectively captures global information and the interconnections among disparate nodes, thereby enhancing the overall performance of the model. Additionally, we transfer the RAG evaluation framework to the KGC field and filter and evaluate the generated knowledge graphs, thereby avoiding incorrectly generated entities and relationships caused by hallucinations in LLMs. We further developed the MINE dataset by constructing standard knowledge graphs for each article and experimentally validated the performance of RAKG. The results show that RAKG achieves an accuracy of 95.91 % on the MINE dataset, a 6.2 % point improvement over the current best baseline, GraphRAG (89.71 %). The code is available at this https URL. 

---
# Augmented Relevance Datasets with Fine-Tuned Small LLMs 

**Authors**: Quentin Fitte-Rey, Matyas Amrouche, Romain Deveaud  

**Link**: [PDF](https://arxiv.org/pdf/2504.09816)  

**Abstract**: Building high-quality datasets and labeling query-document relevance are essential yet resource-intensive tasks, requiring detailed guidelines and substantial effort from human annotators. This paper explores the use of small, fine-tuned large language models (LLMs) to automate relevance assessment, with a focus on improving ranking models' performance by augmenting their training dataset. We fine-tuned small LLMs to enhance relevance assessments, thereby improving dataset creation quality for downstream ranking model training. Our experiments demonstrate that these fine-tuned small LLMs not only outperform certain closed source models on our dataset but also lead to substantial improvements in ranking model performance. These results highlight the potential of leveraging small LLMs for efficient and scalable dataset augmentation, providing a practical solution for search engine optimization. 

---
# Outage Probability Analysis for OTFS with Finite Blocklength 

**Authors**: Xin Zhang, Wensheng Lin, Lixin Li, Zhu Han, Tad Matsumoto  

**Link**: [PDF](https://arxiv.org/pdf/2504.09628)  

**Abstract**: Orthogonal time frequency space (OTFS) modulation is widely acknowledged as a prospective waveform for future wireless communication this http URL provide insights for the practical system design, this paper analyzes the outage probability of OTFS modulation with finite this http URL begin with, we present the system model and formulate the analysis of outage probability for OTFS with finite blocklength as an equivalent problem of calculating the outage probability with finite blocklength over parallel additive white Gaussian noise (AWGN) this http URL, we apply the equivalent noise approach to derive a lower bound on the outage probability of OTFS with finite blocklength under both average power allocation and water-filling power allocation strategies, this http URL, the lower bounds of the outage probability are determined using the Monte-Carlo method for the two power allocation this http URL impact of the number of resolvable paths and coding rates on the outage probability is analyzed, and the simulation results are compared with the theoretical lower bounds. 

---
# Slow Thinking for Sequential Recommendation 

**Authors**: Junjie Zhang, Beichen Zhang, Wenqi Sun, Hongyu Lu, Wayne Xin Zhao, Yu Chen, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2504.09627)  

**Abstract**: To develop effective sequential recommender systems, numerous methods have been proposed to model historical user behaviors. Despite the effectiveness, these methods share the same fast thinking paradigm. That is, for making recommendations, these methods typically encodes user historical interactions to obtain user representations and directly match these representations with candidate item representations. However, due to the limited capacity of traditional lightweight recommendation models, this one-step inference paradigm often leads to suboptimal performance. To tackle this issue, we present a novel slow thinking recommendation model, named STREAM-Rec. Our approach is capable of analyzing historical user behavior, generating a multi-step, deliberative reasoning process, and ultimately delivering personalized recommendations. In particular, we focus on two key challenges: (1) identifying the suitable reasoning patterns in recommender systems, and (2) exploring how to effectively stimulate the reasoning capabilities of traditional recommenders. To this end, we introduce a three-stage training framework. In the first stage, the model is pretrained on large-scale user behavior data to learn behavior patterns and capture long-range dependencies. In the second stage, we design an iterative inference algorithm to annotate suitable reasoning traces by progressively refining the model predictions. This annotated data is then used to fine-tune the model. Finally, in the third stage, we apply reinforcement learning to further enhance the model generalization ability. Extensive experiments validate the effectiveness of our proposed method. 

---
# Revisiting Self-Attentive Sequential Recommendation 

**Authors**: Zan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.09596)  

**Abstract**: Recommender systems are ubiquitous in on-line services to drive businesses. And many sequential recommender models were deployed in these systems to enhance personalization. The approach of using the transformer decoder as the sequential recommender was proposed years ago and is still a strong baseline in recent works. But this kind of sequential recommender model did not scale up well, compared to language models. Quite some details in the classical self-attentive sequential recommender model could be revisited, and some new experiments may lead to new findings, without changing the general model structure which was the focus of many previous works. In this paper, we show the details and propose new experiment methodologies for future research on sequential recommendation, in hope to motivate further exploration to new findings in this area. 

---
# HD-RAG: Retrieval-Augmented Generation for Hybrid Documents Containing Text and Hierarchical Tables 

**Authors**: Chi Zhang, Qiyang Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.09554)  

**Abstract**: With the rapid advancement of large language models (LLMs), Retrieval-Augmented Generation (RAG) effectively combines LLMs generative capabilities with external retrieval-based information. The Hybrid Document RAG task aims to integrate textual and hierarchical tabular data for more comprehensive retrieval and generation in complex scenarios. However, there is no existing dataset specifically designed for this task that includes both text and tabular data. Additionally, existing methods struggle to retrieve relevant tabular data and integrate it with text. Semantic similarity-based retrieval lacks accuracy, while table-specific methods fail to handle complex hierarchical structures effectively. Furthermore, the QA task requires complex reasoning and calculations, further complicating the challenge. In this paper, we propose a new large-scale dataset, DocRAGLib, specifically designed for the question answering (QA) task scenario under Hybrid Document RAG. To tackle these challenges, we introduce HD-RAG, a novel framework that incorporates a row-and-column level (RCL) table representation, employs a two-stage process combining ensemble and LLM-based retrieval, and integrates RECAP, which is designed for multi-step reasoning and complex calculations in Document-QA tasks. We conduct comprehensive experiments with DocRAGLib, showing that HD-RAG outperforms existing baselines in both retrieval accuracy and QA performance, demonstrating its effectiveness. 

---
# Breaking the Lens of the Telescope: Online Relevance Estimation over Large Retrieval Sets 

**Authors**: Mandeep Rathee, Venktesh V, Sean MacAvaney, Avishek Anand  

**Link**: [PDF](https://arxiv.org/pdf/2504.09353)  

**Abstract**: Advanced relevance models, such as those that use large language models (LLMs), provide highly accurate relevance estimations. However, their computational costs make them infeasible for processing large document corpora. To address this, retrieval systems often employ a telescoping approach, where computationally efficient but less precise lexical and semantic retrievers filter potential candidates for further ranking. However, this approach heavily depends on the quality of early-stage retrieval, which can potentially exclude relevant documents early in the process. In this work, we propose a novel paradigm for re-ranking called online relevance estimation that continuously updates relevance estimates for a query throughout the ranking process. Instead of re-ranking a fixed set of top-k documents in a single step, online relevance estimation iteratively re-scores smaller subsets of the most promising documents while adjusting relevance scores for the remaining pool based on the estimations from the final model using an online bandit-based algorithm. This dynamic process mitigates the recall limitations of telescoping systems by re-prioritizing documents initially deemed less relevant by earlier stages -- including those completely excluded by earlier-stage retrievers. We validate our approach on TREC benchmarks under two scenarios: hybrid retrieval and adaptive retrieval. Experimental results demonstrate that our method is sample-efficient and significantly improves recall, highlighting the effectiveness of our online relevance estimation framework for modern search systems. 

---
# SynthTRIPs: A Knowledge-Grounded Framework for Benchmark Query Generation for Personalized Tourism Recommenders 

**Authors**: Ashmi Banerjee, Adithi Satish, Fitri Nur Aisyah, Wolfgang Wörndl, Yashar Deldjoo  

**Link**: [PDF](https://arxiv.org/pdf/2504.09277)  

**Abstract**: Tourism Recommender Systems (TRS) are crucial in personalizing travel experiences by tailoring recommendations to users' preferences, constraints, and contextual factors. However, publicly available travel datasets often lack sufficient breadth and depth, limiting their ability to support advanced personalization strategies -- particularly for sustainable travel and off-peak tourism. In this work, we explore using Large Language Models (LLMs) to generate synthetic travel queries that emulate diverse user personas and incorporate structured filters such as budget constraints and sustainability preferences.
This paper introduces a novel SynthTRIPs framework for generating synthetic travel queries using LLMs grounded in a curated knowledge base (KB). Our approach combines persona-based preferences (e.g., budget, travel style) with explicit sustainability filters (e.g., walkability, air quality) to produce realistic and diverse queries. We mitigate hallucination and ensure factual correctness by grounding the LLM responses in the KB. We formalize the query generation process and introduce evaluation metrics for assessing realism and alignment. Both human expert evaluations and automatic LLM-based assessments demonstrate the effectiveness of our synthetic dataset in capturing complex personalization aspects underrepresented in existing datasets. While our framework was developed and tested for personalized city trip recommendations, the methodology applies to other recommender system domains.
Code and dataset are made public at this https URL 

---
# Large Language Model Empowered Recommendation Meets All-domain Continual Pre-Training 

**Authors**: Haokai Ma, Yunshan Ma, Ruobing Xie, Lei Meng, Jialie Shen, Xingwu Sun, Zhanhui Kang, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.08949)  

**Abstract**: Recent research efforts have investigated how to integrate Large Language Models (LLMs) into recommendation, capitalizing on their semantic comprehension and open-world knowledge for user behavior understanding. These approaches predominantly employ supervised fine-tuning on single-domain user interactions to adapt LLMs for specific recommendation tasks. However, they typically encounter dual challenges: the mismatch between general language representations and domain-specific preference patterns, as well as the limited adaptability to multi-domain recommendation scenarios. To bridge these gaps, we introduce CPRec -- an All-domain Continual Pre-Training framework for Recommendation -- designed to holistically align LLMs with universal user behaviors through the continual pre-training paradigm. Specifically, we first design a unified prompt template and organize users' multi-domain behaviors into domain-specific behavioral sequences and all-domain mixed behavioral sequences that emulate real-world user decision logic. To optimize behavioral knowledge infusion, we devise a Warmup-Stable-Annealing learning rate schedule tailored for the continual pre-training paradigm in recommendation to progressively enhance the LLM's capability in knowledge adaptation from open-world knowledge to universal recommendation tasks. To evaluate the effectiveness of our CPRec, we implement it on a large-scale dataset covering seven domains and conduct extensive experiments on five real-world datasets from two distinct platforms. Experimental results confirm that our continual pre-training paradigm significantly mitigates the semantic-behavioral discrepancy and achieves state-of-the-art performance in all recommendation scenarios. The source code will be released upon acceptance. 

---
# AdaptRec: A Self-Adaptive Framework for Sequential Recommendations with Large Language Models 

**Authors**: Tong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08786)  

**Abstract**: The recent advancements in Large Language Models (LLMs) have generated considerable interest in their utilization for sequential recommendation tasks. While collaborative signals from similar users are central to recommendation modeling, effectively transforming these signals into a format that LLMs can understand and utilize remains challenging. The critical challenges include selecting relevant demonstrations from large-scale user interactions and ensuring their alignment with LLMs' reasoning process. To address these challenges, we introduce AdaptRec, a self-adaptive fram-ework that leverages LLMs for sequential recommendations by incorporating explicit collaborative signals. AdaptRec employs a two-phase user selection mechanism -- User Similarity Retrieval and Self-Adaptive User Selection -- to efficiently identify relevant user sequences in large-scale datasets from multi-metric evaluation. We also develop a User-Based Similarity Retrieval Prompt, enabling the model to actively select similar users and continuously refine its selection criteria during training. Using the collaborative signals from similar users, we construct a User-Contextualized Recommendation Prompt that translates their behavior sequences into natural language, explicitly integrating this information into the recommendation process. Experiments demonstrate AdaptRec's superior performance, with significant improvements in HitRatio@1 scores of 7.13\%, 18.16\%, and 10.41\% across real-world datasets with full fine-tuning, and even higher gains of 23.00\%, 15.97\%, and 17.98\% in few-shot scenarios. 

---
# How Relevance Emerges: Interpreting LoRA Fine-Tuning in Reranking LLMs 

**Authors**: Atharva Nijasure, Tanya Chowdhury, James Allan  

**Link**: [PDF](https://arxiv.org/pdf/2504.08780)  

**Abstract**: We conduct a behavioral exploration of LoRA fine-tuned LLMs for Passage Reranking to understand how relevance signals are learned and deployed by Large Language Models. By fine-tuning Mistral-7B, LLaMA3.1-8B, and Pythia-6.9B on MS MARCO under diverse LoRA configurations, we investigate how relevance modeling evolves across checkpoints, the impact of LoRA rank (1, 2, 8, 32), and the relative importance of updated MHA vs. MLP components. Our ablations reveal which layers and projections within LoRA transformations are most critical for reranking accuracy. These findings offer fresh explanations into LoRA's adaptation mechanisms, setting the stage for deeper mechanistic studies in Information Retrieval. All models used in this study have been shared. 

---
# Counterfactual Inference under Thompson Sampling 

**Authors**: Olivier Jeunen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08773)  

**Abstract**: Recommender systems exemplify sequential decision-making under uncertainty, strategically deciding what content to serve to users, to optimise a range of potential objectives. To balance the explore-exploit trade-off successfully, Thompson sampling provides a natural and widespread paradigm to probabilistically select which action to take. Questions of causal and counterfactual inference, which underpin use-cases like offline evaluation, are not straightforward to answer in these contexts. Specifically, whilst most existing estimators rely on action propensities, these are not readily available under Thompson sampling procedures.
We derive exact and efficiently computable expressions for action propensities under a variety of parameter and outcome distributions, enabling the use of off-policy estimators in Thompson sampling scenarios. This opens up a range of practical use-cases where counterfactual inference is crucial, including unbiased offline evaluation of recommender systems, as well as general applications of causal inference in online advertising, personalisation, and beyond. 

---
# Generate the browsing process for short-video recommendation 

**Authors**: Chao Feng, Yanze Zhang, Chenghao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08771)  

**Abstract**: This paper introduces a new model to generate the browsing process for short-video recommendation and proposes a novel Segment Content Aware Model via User Engagement Feedback (SCAM) for watch time prediction in video recommendation. Unlike existing methods that rely on multimodal features for video content understanding, SCAM implicitly models video content through users' historical watching behavior, enabling segment-level understanding without complex multimodal data. By dividing videos into segments based on duration and employing a Transformer-like architecture, SCAM captures the sequential dependence between segments while mitigating duration bias. Extensive experiments on industrial-scale and public datasets demonstrate SCAM's state-of-the-art performance in watch time prediction. The proposed approach offers a scalable and effective solution for video recommendation by leveraging segment-level modeling and users' engagement feedback. 

---
# Accelerating Causal Network Discovery of Alzheimer Disease Biomarkers via Scientific Literature-based Retrieval Augmented Generation 

**Authors**: Xiaofan Zhou, Liangjie Huang, Pinyang Cheng, Wenpen Yin, Rui Zhang, Wenrui Hao, Lu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.08768)  

**Abstract**: The causal relationships between biomarkers are essential for disease diagnosis and medical treatment planning. One notable application is Alzheimer's disease (AD) diagnosis, where certain biomarkers may influence the presence of others, enabling early detection, precise disease staging, targeted treatments, and improved monitoring of disease progression. However, understanding these causal relationships is complex and requires extensive research. Constructing a comprehensive causal network of biomarkers demands significant effort from human experts, who must analyze a vast number of research papers, and have bias in understanding diseases' biomarkers and their relation. This raises an important question: Can advanced large language models (LLMs), such as those utilizing retrieval-augmented generation (RAG), assist in building causal networks of biomarkers for further medical analysis? To explore this, we collected 200 AD-related research papers published over the past 25 years and then integrated scientific literature with RAG to extract AD biomarkers and generate causal relations among them. Given the high-risk nature of the medical diagnosis, we applied uncertainty estimation to assess the reliability of the generated causal edges and examined the faithfulness and scientificness of LLM reasoning using both automatic and human evaluation. We find that RAG enhances the ability of LLMs to generate more accurate causal networks from scientific papers. However, the overall performance of LLMs in identifying causal relations of AD biomarkers is still limited. We hope this study will inspire further foundational research on AI-driven analysis of AD biomarkers causal network discovery. 

---
# A Proposed Hybrid Recommender System for Tourism Industry in Iraq Using Evolutionary Apriori and K-means Algorithms 

**Authors**: Bryar A. Hassan, Alla A. Hassan, Joan Lu, Aram M. Ahmed, Tarik A. Rashid  

**Link**: [PDF](https://arxiv.org/pdf/2504.08767)  

**Abstract**: The rapid proliferation of tourism data across sectors, including accommodations, cultural sites, and events, has made it increasingly challenging for travelers to identify relevant and personalized recommendations. While traditional recommender systems such as collaborative, content-based, and context-aware systems offer partial solutions, they often struggle with issues like data sparsity and overspecialization. This study proposes a novel hybrid recommender system that combines evolutionary Apriori and K-means clustering algorithms to improve recommendation accuracy and efficiency in the tourism domain. Designed specifically to address the diverse and dynamic tourism landscape in Iraq, the system provides personalized recommendations and clusters of tourist destinations tailored to user preferences and contextual information. To evaluate the systems performance, experiments were conducted on an augmented dataset representative of Iraqs tourism activity, comparing the proposed system with existing methods. Results indicate that the proposed hybrid system significantly reduces execution time by 27-56% and space consumption by 24-31%, while achieving consistently lower Root Mean Square Error (RMSE) and Mean Absolute Error (MAE) values, thereby enhancing prediction accuracy. This approach offers a scalable, context-aware framework that is well-suited for application in regions where tourism data is limited, such as Iraq, ultimately advancing tourism recommender systems by addressing their limitations in complex and data-scarce environments. 

---
# Evaluation of the phi-3-mini SLM for identification of texts related to medicine, health, and sports injuries 

**Authors**: Chris Brogly, Saif Rjaibi, Charlotte Liang, Erica Lam, Edward Wang, Adam Levitan, Sarah Paleczny, Michael Cusimano  

**Link**: [PDF](https://arxiv.org/pdf/2504.08764)  

**Abstract**: Small Language Models (SLMs) have potential to be used for automatically labelling and identifying aspects of text data for medicine/health-related purposes from documents and the web. As their resource requirements are significantly lower than Large Language Models (LLMs), these can be deployed potentially on more types of devices. SLMs often are benchmarked on health/medicine-related tasks, such as MedQA, although performance on these can vary especially depending on the size of the model in terms of number of parameters. Furthermore, these test results may not necessarily reflect real-world performance regarding the automatic labelling or identification of texts in documents and the web. As a result, we compared topic-relatedness scores from Microsofts phi-3-mini-4k-instruct SLM to the topic-relatedness scores from 7 human evaluators on 1144 samples of medical/health-related texts and 1117 samples of sports injury-related texts. These texts were from a larger dataset of about 9 million news headlines, each of which were processed and assigned scores by phi-3-mini-4k-instruct. Our sample was selected (filtered) based on 1 (low filtering) or more (high filtering) Boolean conditions on the phi-3 SLM scores. We found low-moderate significant correlations between the scores from the SLM and human evaluators for sports injury texts with low filtering (\r{ho} = 0.3413, p < 0.001) and medicine/health texts with high filtering (\r{ho} = 0.3854, p < 0.001), and low significant correlation for medicine/health texts with low filtering (\r{ho} = 0.2255, p < 0.001). There was negligible, insignificant correlation for sports injury-related texts with high filtering (\r{ho} = 0.0318, p = 0.4466). 

---
# WebMap -- Large Language Model-assisted Semantic Link Induction in the Web 

**Authors**: Shiraj Pokharel, Georg P. Roßrucker, Mario M. Kubek  

**Link**: [PDF](https://arxiv.org/pdf/2504.08763)  

**Abstract**: Carrying out research tasks is only inadequately supported, if not hindered, by current web search engines. This paper therefore proposes functional extensions of WebMap, a semantically induced overlay linking structure on the web to inherently facilitate research activities. These add-ons support the dynamic determination and regrouping of document clusters, the creation of a semantic signpost in the web, and the interactive tracing of topics back to their origins. 

---
# InteractiveSurvey: An LLM-based Personalized and Interactive Survey Paper Generation System 

**Authors**: Zhiyuan Wen, Jiannong Cao, Zian Wang, Beichen Guo, Ruosong Yang, Shuaiqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08762)  

**Abstract**: The exponential growth of academic literature creates urgent demands for comprehensive survey papers, yet manual writing remains time-consuming and labor-intensive. Recent advances in large language models (LLMs) and retrieval-augmented generation (RAG) facilitate studies in synthesizing survey papers from multiple references, but most existing works restrict users to title-only inputs and fixed outputs, neglecting the personalized process of survey paper writing. In this paper, we introduce InteractiveSurvey - an LLM-based personalized and interactive survey paper generation system. InteractiveSurvey can generate structured, multi-modal survey papers with reference categorizations from multiple reference papers through both online retrieval and user uploads. More importantly, users can customize and refine intermediate components continuously during generation, including reference categorization, outline, and survey content through an intuitive interface. Evaluations of content quality, time efficiency, and user studies show that InteractiveSurvey is an easy-to-use survey generation system that outperforms most LLMs and existing methods in output content quality while remaining highly time-efficient. 

---
# UltraRAG: A Modular and Automated Toolkit for Adaptive Retrieval-Augmented Generation 

**Authors**: Yuxuan Chen, Dewen Guo, Sen Mei, Xinze Li, Hao Chen, Yishan Li, Yixuan Wang, Chaoyue Tang, Ruobing Wang, Dingjun Wu, Yukun Yan, Zhenghao Liu, Shi Yu, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.08761)  

**Abstract**: Retrieval-Augmented Generation (RAG) significantly enhances the performance of large language models (LLMs) in downstream tasks by integrating external knowledge. To facilitate researchers in deploying RAG systems, various RAG toolkits have been introduced. However, many existing RAG toolkits lack support for knowledge adaptation tailored to specific application scenarios. To address this limitation, we propose UltraRAG, a RAG toolkit that automates knowledge adaptation throughout the entire workflow, from data construction and training to evaluation, while ensuring ease of use. UltraRAG features a user-friendly WebUI that streamlines the RAG process, allowing users to build and optimize systems without coding expertise. It supports multimodal input and provides comprehensive tools for managing the knowledge base. With its highly modular architecture, UltraRAG delivers an end-to-end development solution, enabling seamless knowledge adaptation across diverse user scenarios. The code, demonstration videos, and installable package for UltraRAG are publicly available at this https URL. 

---
# Hyper-RAG: Combating LLM Hallucinations using Hypergraph-Driven Retrieval-Augmented Generation 

**Authors**: Yifan Feng, Hao Hu, Xingliang Hou, Shiquan Liu, Shihui Ying, Shaoyi Du, Han Hu, Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2504.08758)  

**Abstract**: Large language models (LLMs) have transformed various sectors, including education, finance, and medicine, by enhancing content generation and decision-making processes. However, their integration into the medical field is cautious due to hallucinations, instances where generated content deviates from factual accuracy, potentially leading to adverse outcomes. To address this, we introduce Hyper-RAG, a hypergraph-driven Retrieval-Augmented Generation method that comprehensively captures both pairwise and beyond-pairwise correlations in domain-specific knowledge, thereby mitigating hallucinations. Experiments on the NeurologyCrop dataset with six prominent LLMs demonstrated that Hyper-RAG improves accuracy by an average of 12.3% over direct LLM use and outperforms Graph RAG and Light RAG by 6.3% and 6.0%, respectively. Additionally, Hyper-RAG maintained stable performance with increasing query complexity, unlike existing methods which declined. Further validation across nine diverse datasets showed a 35.5% performance improvement over Light RAG using a selection-based assessment. The lightweight variant, Hyper-RAG-Lite, achieved twice the retrieval speed and a 3.3% performance boost compared with Light RAG. These results confirm Hyper-RAG's effectiveness in enhancing LLM reliability and reducing hallucinations, making it a robust solution for high-stakes applications like medical diagnostics. 

---
# A Framework for Lightweight Responsible Prompting Recommendation 

**Authors**: Tiago Machado, Sara E. Berger, Cassia Sanctos, Vagner Figueiredo de Santana, Lemara Williams, Zhaoqing Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08757)  

**Abstract**: Computer Science and Design practitioners have been researching and proposing alternatives for a dearth of recommendations, standards, or best practices in user interfaces for decades. Now, with the advent of generative Artificial Intelligence (GenAI), we have yet again an emerging, powerful technology that lacks sufficient guidance in terms of possible interactions, inputs, and outcomes. In this context, this work proposes a lightweight framework for responsible prompting recommendation to be added before the prompt is sent to GenAI. The framework is comprised of (1) a human-curated dataset for recommendations, (2) a red team dataset for assessing recommendations, (3) a sentence transformer for semantics mapping, (4) a similarity metric to map input prompt to recommendations, (5) a set of similarity thresholds, (6) quantized sentence embeddings, (7) a recommendation engine, and (8) an evaluation step to use the red team dataset. With the proposed framework and open-source system, the contributions presented can be applied in multiple contexts where end-users can benefit from guidance for interacting with GenAI in a more responsible way, recommending positive values to be added and harmful sentences to be removed. 

---
# MHTS: Multi-Hop Tree Structure Framework for Generating Difficulty-Controllable QA Datasets for RAG Evaluation 

**Authors**: Jeongsoo Lee, Daeyong Kwon, Kyohoon Jin, Junnyeong Jeong, Minwoo Sim, Minwoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.08756)  

**Abstract**: Existing RAG benchmarks often overlook query difficulty, leading to inflated performance on simpler questions and unreliable evaluations. A robust benchmark dataset must satisfy three key criteria: quality, diversity, and difficulty, which capturing the complexity of reasoning based on hops and the distribution of supporting evidence. In this paper, we propose MHTS (Multi-Hop Tree Structure), a novel dataset synthesis framework that systematically controls multi-hop reasoning complexity by leveraging a multi-hop tree structure to generate logically connected, multi-chunk queries. Our fine-grained difficulty estimation formula exhibits a strong correlation with the overall performance metrics of a RAG system, validating its effectiveness in assessing both retrieval and answer generation capabilities. By ensuring high-quality, diverse, and difficulty-controlled queries, our approach enhances RAG evaluation and benchmarking capabilities. 

---
# Delving into: the quantification of Ai-generated content on the internet (synthetic data) 

**Authors**: Dirk HR Spennemann  

**Link**: [PDF](https://arxiv.org/pdf/2504.08755)  

**Abstract**: While it is increasingly evident that the internet is becoming saturated with content created by generated Ai large language models, accurately measuring the scale of this phenomenon has proven challenging. By analyzing the frequency of specific keywords commonly used by ChatGPT, this paper demonstrates that such linguistic markers can effectively be used to esti-mate the presence of generative AI content online. The findings suggest that at least 30% of text on active web pages originates from AI-generated sources, with the actual proportion likely ap-proaching 40%. Given the implications of autophagous loops, this is a sobering realization. 

---
# Towards Personalized Conversational Sales Agents with Contextual User Profiling for Strategic Action 

**Authors**: Tongyoung Kim, Jeongeun Lee, Soojin Yoon, Seonghwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.08754)  

**Abstract**: Conversational Recommender Systems (CRSs) aim to engage users in dialogue to provide tailored recommendations. While traditional CRSs focus on eliciting preferences and retrieving items, real-world e-commerce interactions involve more complex decision-making, where users consider multiple factors beyond simple attributes. To bridge this gap, we introduce Conversational Sales (CSales), a novel task that unifies preference elicitation, recommendation, and persuasion to better support user decision-making. For a realistic evaluation of CSales, we present CSUser, an LLM-based user simulator constructed from real-world data, modeling diverse user profiles with needs and personalities. Additionally, we propose CSI, a conversational sales agent that proactively infers contextual profiles through dialogue for personalized action planning. Extensive experiments demonstrate that CSUser effectively replicates real-world users and emphasize the importance of contextual profiling for strategic action selection, ultimately driving successful purchases in e-commerce. 

---
# Domain Specific Question to SQL Conversion with Embedded Data Balancing Technique 

**Authors**: Jyothi, T. Satyanarayana Murthy  

**Link**: [PDF](https://arxiv.org/pdf/2504.08753)  

**Abstract**: The rise of deep learning in natural language processing has fostered the creation of text to structured query language models composed of an encoder and a decoder. Researchers have experimented with various intermediate processing like schema linking, table type aware, value extract. To generate accurate SQL results for the user question. However error analysis performed on the failed cases on these systems shows, 29 percentage of the errors would be because the system was unable to understand the values expressed by the user in their question. This challenge affects the generation of accurate SQL queries, especially when dealing with domain-specific terms and specific value conditions, where traditional methods struggle to maintain consistency and precision. To overcome these obstacles, proposed two intermediations like implementing data balancing technique and over sampling domain-specific queries which would refine the model architecture to enhance value recognition and fine tuning the model for domain-specific questions. This proposed solution achieved 10.98 percentage improvement in accuracy of the model performance compared to the state of the art model tested on WikiSQL dataset. to convert the user question accurately to SQL queries. Applying oversampling technique on the domain-specific questions shown a significant improvement as compared with traditional approaches. 

---
# Patience is all you need! An agentic system for performing scientific literature review 

**Authors**: David Brett, Anniek Myatt  

**Link**: [PDF](https://arxiv.org/pdf/2504.08752)  

**Abstract**: Large language models (LLMs) have grown in their usage to provide support for question answering across numerous disciplines. The models on their own have already shown promise for answering basic questions, however fail quickly where expert domain knowledge is required or the question is nuanced. Scientific research often involves searching for relevant literature, distilling pertinent information from that literature and analysing how the findings support or contradict one another. The information is often encapsulated in the full text body of research articles, rather than just in the abstracts. Statements within these articles frequently require the wider article context to be fully understood. We have built an LLM-based system that performs such search and distillation of information encapsulated in scientific literature, and we evaluate our keyword based search and information distillation system against a set of biology related questions from previously released literature benchmarks. We demonstrate sparse retrieval methods exhibit results close to state of the art without the need for dense retrieval, with its associated infrastructure and complexity overhead. We also show how to increase the coverage of relevant documents for literature review generation. 

---
# Research on the Design of a Short Video Recommendation System Based on Multimodal Information and Differential Privacy 

**Authors**: Haowei Yang, Lei Fu, Qingyi Lu, Yue Fan, Tianle Zhang, Ruohan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08751)  

**Abstract**: With the rapid development of short video platforms, recommendation systems have become key technologies for improving user experience and enhancing platform engagement. However, while short video recommendation systems leverage multimodal information (such as images, text, and audio) to improve recommendation effectiveness, they also face the severe challenge of user privacy leakage. This paper proposes a short video recommendation system based on multimodal information and differential privacy protection. First, deep learning models are used for feature extraction and fusion of multimodal data, effectively improving recommendation accuracy. Then, a differential privacy protection mechanism suitable for recommendation scenarios is designed to ensure user data privacy while maintaining system performance. Experimental results show that the proposed method outperforms existing mainstream approaches in terms of recommendation accuracy, multimodal fusion effectiveness, and privacy protection performance, providing important insights for the design of recommendation systems for short video platforms. 

---
# A Quantitative Approach to Evaluating Open-Source EHR Systems for Indian Healthcare 

**Authors**: Biswanath Dutta, Debanjali Bain  

**Link**: [PDF](https://arxiv.org/pdf/2504.08750)  

**Abstract**: The increasing use of Electronic Health Records (EHR) has emphasized the need for standardization and interoperability in healthcare data management. The Ministry of Health and Family Welfare, Government of India, has introduced the Electronic Health Record Minimum Data Set (EHRMDS) to facilitate uniformity in clinical documentation. However, the compatibility of Open-Source Electronic Health Record Systems (OS-EHRS) with EHRMDS remains largely unexplored. This study conducts a systematic assessment of the alignment between EHRMDS and commonly utilized OS-EHRS to determine the most appropriate system for healthcare environments in India. A quantitative closeness analysis was performed by comparing the metadata elements of EHRMDS with those of 10 selected OS-EHRS. Using crosswalk methodologies based on syntactic and semantic similarity, the study measured the extent of metadata alignment. Results indicate that OpenEMR exhibits the highest compatibility with EHRMDS, covering 73.81% of its metadata elements, while OpenClinic shows the least alignment at 33.33%. Additionally, the analysis identified 47 metadata elements present in OS-EHRS but absent in EHRMDS, suggesting the need for an extended metadata schema. By bridging gaps in clinical metadata, this study contributes to enhancing the interoperability of EHR systems in India. The findings provide valuable insights for healthcare policymakers and organizations seeking to adopt OS-EHRS aligned with national standards. Keywords. EHR metadata, electronic health record systems, EHRMDS, meta data, structured vocabularies, metadata crosswalk, methodologies and tools, SNOMED-CT, UMLS terms. 

---
# A Survey of Multimodal Retrieval-Augmented Generation 

**Authors**: Lang Mei, Siyu Mo, Zhihan Yang, Chong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08748)  

**Abstract**: Multimodal Retrieval-Augmented Generation (MRAG) enhances large language models (LLMs) by integrating multimodal data (text, images, videos) into retrieval and generation processes, overcoming the limitations of text-only Retrieval-Augmented Generation (RAG). While RAG improves response accuracy by incorporating external textual knowledge, MRAG extends this framework to include multimodal retrieval and generation, leveraging contextual information from diverse data types. This approach reduces hallucinations and enhances question-answering systems by grounding responses in factual, multimodal knowledge. Recent studies show MRAG outperforms traditional RAG, especially in scenarios requiring both visual and textual understanding. This survey reviews MRAG's essential components, datasets, evaluation methods, and limitations, providing insights into its construction and improvement. It also identifies challenges and future research directions, highlighting MRAG's potential to revolutionize multimodal information retrieval and generation. By offering a comprehensive perspective, this work encourages further exploration into this promising paradigm. 

---
# Enhancing Recommender Systems Using Textual Embeddings from Pre-trained Language Models 

**Authors**: Ngoc Luyen Le, Marie-Hélène Abel  

**Link**: [PDF](https://arxiv.org/pdf/2504.08746)  

**Abstract**: Recent advancements in language models and pre-trained language models like BERT and RoBERTa have revolutionized natural language processing, enabling a deeper understanding of human-like language. In this paper, we explore enhancing recommender systems using textual embeddings from pre-trained language models to address the limitations of traditional recommender systems that rely solely on explicit features from users, items, and user-item interactions. By transforming structured data into natural language representations, we generate high-dimensional embeddings that capture deeper semantic relationships between users, items, and contexts. Our experiments demonstrate that this approach significantly improves recommendation accuracy and relevance, resulting in more personalized and context-aware recommendations. The findings underscore the potential of PLMs to enhance the effectiveness of recommender systems. 

---
# Improving RAG for Personalization with Author Features and Contrastive Examples 

**Authors**: Mert Yazan, Suzan Verberne, Frederik Situmeang  

**Link**: [PDF](https://arxiv.org/pdf/2504.08745)  

**Abstract**: Personalization with retrieval-augmented generation (RAG) often fails to capture fine-grained features of authors, making it hard to identify their unique traits. To enrich the RAG context, we propose providing Large Language Models (LLMs) with author-specific features, such as average sentiment polarity and frequently used words, in addition to past samples from the author's profile. We introduce a new feature called Contrastive Examples: documents from other authors are retrieved to help LLM identify what makes an author's style unique in comparison to others. Our experiments show that adding a couple of sentences about the named entities, dependency patterns, and words a person uses frequently significantly improves personalized text generation. Combining features with contrastive examples boosts the performance further, achieving a relative 15% improvement over baseline RAG while outperforming the benchmarks. Our results show the value of fine-grained features for better personalization, while opening a new research dimension for including contrastive examples as a complement with RAG. We release our code publicly. 

---
# ExpertRAG: Efficient RAG with Mixture of Experts -- Optimizing Context Retrieval for Adaptive LLM Responses 

**Authors**: Esmail Gumaan  

**Link**: [PDF](https://arxiv.org/pdf/2504.08744)  

**Abstract**: ExpertRAG is a novel theoretical framework that integrates Mixture-of-Experts (MoE) architectures with Retrieval Augmented Generation (RAG) to advance the efficiency and accuracy of knowledge-intensive language modeling. We propose a dynamic retrieval gating mechanism coupled with expert routing, enabling the model to selectively consult an external knowledge store or rely on specialized internal experts based on the query's needs. The paper lays out the theoretical foundations of ExpertRAG, including a probabilistic formulation that treats retrieval and expert selection as latent decisions, and mathematical justifications for its efficiency in both computation and knowledge utilization. We derive formulae to quantify the expected computational cost savings from selective retrieval and the capacity gains from sparse expert utilization. A comparative analysis positions ExpertRAG against standard RAG (with always-on retrieval) and pure MoE models (e.g., Switch Transformer, Mixtral) to highlight its unique balance between parametric knowledge and non-parametric retrieval. We also outline an experimental validation strategy, proposing benchmarks and evaluation protocols to test ExpertRAG's performance on factual recall, generalization, and inference efficiency. The proposed framework, although presented theoretically, is supported by insights from prior work in RAG and MoE, and is poised to provide more factual, efficient, and adaptive generation by leveraging the best of both paradigms. In summary, ExpertRAG contributes a new perspective on scaling and augmenting language models, backed by a thorough analysis and a roadmap for empirical validation. 

---
# Dynamic Topic Analysis in Academic Journals using Convex Non-negative Matrix Factorization Method 

**Authors**: Yang Yang, Tong Zhang, Jian Wu, Lijie Su  

**Link**: [PDF](https://arxiv.org/pdf/2504.08743)  

**Abstract**: With the rapid advancement of large language models, academic topic identification and topic evolution analysis are crucial for enhancing AI's understanding capabilities. Dynamic topic analysis provides a powerful approach to capturing and understanding the temporal evolution of topics in large-scale datasets. This paper presents a two-stage dynamic topic analysis framework that incorporates convex optimization to improve topic consistency, sparsity, and interpretability. In Stage 1, a two-layer non-negative matrix factorization (NMF) model is employed to extract annual topics and identify key terms. In Stage 2, a convex optimization algorithm refines the dynamic topic structure using the convex NMF (cNMF) model, further enhancing topic integration and stability. Applying the proposed method to IEEE journal abstracts from 2004 to 2022 effectively identifies and quantifies emerging research topics, such as COVID-19 and digital twins. By optimizing sparsity differences in the clustering feature space between traditional and emerging research topics, the framework provides deeper insights into topic evolution and ranking analysis. Moreover, the NMF-cNMF model demonstrates superior stability in topic consistency. At sparsity levels of 0.4, 0.6, and 0.9, the proposed approach improves topic ranking stability by 24.51%, 56.60%, and 36.93%, respectively. The source code (to be open after publication) is available at this https URL. 

---
# Simulating Filter Bubble on Short-video Recommender System with Large Language Model Agents 

**Authors**: Nicholas Sukiennik, Haoyu Wang, Zailin Zeng, Chen Gao, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.08742)  

**Abstract**: An increasing reliance on recommender systems has led to concerns about the creation of filter bubbles on social media, especially on short video platforms like TikTok. However, their formation is still not entirely understood due to the complex dynamics between recommendation algorithms and user feedback. In this paper, we aim to shed light on these dynamics using a large language model-based simulation framework. Our work employs real-world short-video data containing rich video content information and detailed user-agents to realistically simulate the recommendation-feedback cycle. Through large-scale simulations, we demonstrate that LLMs can replicate real-world user-recommender interactions, uncovering key mechanisms driving filter bubble formation. We identify critical factors, such as demographic features and category attraction that exacerbate content homogenization. To mitigate this, we design and test interventions including various cold-start and feedback weighting strategies, showing measurable reductions in filter bubble effects. Our framework enables rapid prototyping of recommendation strategies, offering actionable solutions to enhance content diversity in real-world systems. Furthermore, we analyze how LLM-inherent biases may propagate through recommendations, proposing safeguards to promote equity for vulnerable groups, such as women and low-income populations. By examining the interplay between recommendation and LLM agents, this work advances a deeper understanding of algorithmic bias and provides practical tools to promote inclusive digital spaces. 

---
# Recommendation System in Advertising and Streaming Media: Unsupervised Data Enhancement Sequence Suggestions 

**Authors**: Kowei Shih, Yi Han, Li Tan  

**Link**: [PDF](https://arxiv.org/pdf/2504.08740)  

**Abstract**: Sequential recommendation is an extensively explored approach to capturing users' evolving preferences based on past interactions, aimed at predicting their next likely choice. Despite significant advancements in this domain, including methods based on RNNs and self-attention, challenges like limited supervised signals and noisy data caused by unintentional clicks persist. To address these challenges, some studies have incorporated unsupervised learning by leveraging local item contexts within individual sequences. However, these methods often overlook the intricate associations between items across multiple sequences and are susceptible to noise in item co-occurrence patterns. In this context, we introduce a novel framework, Global Unsupervised Data-Augmentation (UDA4SR), which adopts a graph contrastive learning perspective to generate more robust item embeddings for sequential recommendation. Our approach begins by integrating Generative Adversarial Networks (GANs) for data augmentation, which serves as the first step to enhance the diversity and richness of the training data. Then, we build a Global Item Relationship Graph (GIG) based on all user interaction sequences. Subsequently, we employ graph contrastive learning on the refined graph to enhance item embeddings by capturing complex global associations. To model users' dynamic and diverse interests more effectively, we enhance the CapsNet module with a novel target-attention mechanism. Extensive experiments show that UDA4SR significantly outperforms state-of-the-art approaches. 

---
# Enhancing Product Search Interfaces with Sketch-Guided Diffusion and Language Agents 

**Authors**: Edward Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.08739)  

**Abstract**: The rapid progress in diffusion models, transformers, and language agents has unlocked new possibilities, yet their potential in user interfaces and commercial applications remains underexplored. We present Sketch-Search Agent, a novel framework that transforms the image search experience by integrating a multimodal language agent with freehand sketches as control signals for diffusion models. Using the T2I-Adapter, Sketch-Search Agent combines sketches and text prompts to generate high-quality query images, encoded via a CLIP image encoder for efficient matching against an image corpus. Unlike existing methods, Sketch-Search Agent requires minimal setup, no additional training, and excels in sketch-based image retrieval and natural language interactions. The multimodal agent enhances user experience by dynamically retaining preferences, ranking results, and refining queries for personalized recommendations. This interactive design empowers users to create sketches and receive tailored product suggestions, showcasing the potential of diffusion models in user-centric image retrieval. Experiments confirm Sketch-Search Agent's high accuracy in delivering relevant product search results. 

---
# AI-Driven Sentiment Analytics: Unlocking Business Value in the E-Commerce Landscape_v1 

**Authors**: Qianye Wu, Chengxuan Xia, Sixuan Tian  

**Link**: [PDF](https://arxiv.org/pdf/2504.08738)  

**Abstract**: The rapid growth of e-commerce has led to an overwhelming volume of customer feedback, from product reviews to service interactions. Extracting meaningful insights from this data is crucial for businesses aiming to improve customer satisfaction and optimize decision-making. This paper presents an AI-driven sentiment analysis system designed specifically for e-commerce applications, balancing accuracy with interpretability. Our approach integrates traditional machine learning techniques with modern deep learning models, allowing for a more nuanced understanding of customer sentiment while ensuring transparency in decision-making. Experimental results show that our system outperforms standard sentiment analysis methods, achieving an accuracy of 89.7% on diverse, large-scale datasets. Beyond technical performance, real-world implementation across multiple e-commerce platforms demonstrates tangible improvements in customer engagement and operational efficiency. This study highlights both the potential and the challenges of applying AI to sentiment analysis in a commercial setting, offering insights into practical deployment strategies and areas for future refinement. 

---
# AlayaDB: The Data Foundation for Efficient and Effective Long-context LLM Inference 

**Authors**: Yangshen Deng, Zhengxin You, Long Xiang, Qilong Li, Peiqi Yuan, Zhaoyang Hong, Yitao Zheng, Wanting Li, Runzhong Li, Haotian Liu, Kyriakos Mouratidis, Man Lung Yiu, Huan Li, Qiaomu Shen, Rui Mao, Bo Tang  

**Link**: [PDF](https://arxiv.org/pdf/2504.10326)  

**Abstract**: AlayaDB is a cutting-edge vector database system natively architected for efficient and effective long-context inference for Large Language Models (LLMs) at AlayaDB AI. Specifically, it decouples the KV cache and attention computation from the LLM inference systems, and encapsulates them into a novel vector database system. For the Model as a Service providers (MaaS), AlayaDB consumes fewer hardware resources and offers higher generation quality for various workloads with different kinds of Service Level Objectives (SLOs), when comparing with the existing alternative solutions (e.g., KV cache disaggregation, retrieval-based sparse attention). The crux of AlayaDB is that it abstracts the attention computation and cache management for LLM inference into a query processing procedure, and optimizes the performance via a native query optimizer. In this work, we demonstrate the effectiveness of AlayaDB via (i) three use cases from our industry partners, and (ii) extensive experimental results on LLM inference benchmarks. 

---
# Session-based Recommender Systems: User Interest as a Stochastic Process in the Latent Space 

**Authors**: Klaudia Balcer, Piotr Lipinski  

**Link**: [PDF](https://arxiv.org/pdf/2504.10005)  

**Abstract**: This paper jointly addresses the problem of data uncertainty, popularity bias, and exposure bias in session-based recommender systems. We study the symptoms of this bias both in item embeddings and in recommendations. We propose treating user interest as a stochastic process in the latent space and providing a model-agnostic implementation of this mathematical concept. The proposed stochastic component consists of elements: debiasing item embeddings with regularization for embedding uniformity, modeling dense user interest from session prefixes, and introducing fake targets in the data to simulate extended exposure. We conducted computational experiments on two popular benchmark datasets, Diginetica and YooChoose 1/64, as well as several modifications of the YooChoose dataset with different ratios of popular items. The results show that the proposed approach allows us to mitigate the challenges mentioned. 

---
# VDocRAG: Retrieval-Augmented Generation over Visually-Rich Documents 

**Authors**: Ryota Tanaka, Taichi Iki, Taku Hasegawa, Kyosuke Nishida, Kuniko Saito, Jun Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2504.09795)  

**Abstract**: We aim to develop a retrieval-augmented generation (RAG) framework that answers questions over a corpus of visually-rich documents presented in mixed modalities (e.g., charts, tables) and diverse formats (e.g., PDF, PPTX). In this paper, we introduce a new RAG framework, VDocRAG, which can directly understand varied documents and modalities in a unified image format to prevent missing information that occurs by parsing documents to obtain text. To improve the performance, we propose novel self-supervised pre-training tasks that adapt large vision-language models for retrieval by compressing visual information into dense token representations while aligning them with textual content in documents. Furthermore, we introduce OpenDocVQA, the first unified collection of open-domain document visual question answering datasets, encompassing diverse document types and formats. OpenDocVQA provides a comprehensive resource for training and evaluating retrieval and question answering models on visually-rich documents in an open-domain setting. Experiments show that VDocRAG substantially outperforms conventional text-based RAG and has strong generalization capability, highlighting the potential of an effective RAG paradigm for real-world documents. 

---
# Iterative Self-Training for Code Generation via Reinforced Re-Ranking 

**Authors**: Nikita Sorokin, Ivan Sedykh, Valentin Malykh  

**Link**: [PDF](https://arxiv.org/pdf/2504.09643)  

**Abstract**: Generating high-quality code that solves complex programming tasks is challenging, especially with current decoder-based models that produce highly stochastic outputs. In code generation, even minor errors can easily break the entire solution. Leveraging multiple sampled solutions can significantly improve the overall output quality.
One effective way to enhance code generation is by pairing a code generation model with a reranker model, which selects the best solution from the generated samples. We propose a novel iterative self-training approach for self-training reranker models using Proximal Policy Optimization (PPO), aimed at improving both reranking accuracy and the overall code generation process. Unlike traditional PPO approaches, where the focus is on optimizing a generative model with a reward model, our approach emphasizes the development of a robust reward/reranking model. This model improves the quality of generated code through reranking and addresses problems and errors that the reward model might overlook during PPO alignment with the reranker. Our method iteratively refines the training dataset by re-evaluating outputs, identifying high-scoring negative examples, and incorporating them into the training loop, that boosting model performance.
Our evaluation on the MultiPL-E dataset demonstrates that our 13.4B parameter model outperforms a 33B model in code generation quality while being three times faster. Moreover, it achieves performance comparable to GPT-4 and surpasses it in one programming language. 

---
# FROG: Effective Friend Recommendation in Online Games via Modality-aware User Preferences 

**Authors**: Qiwei Wang, Dandan Lin, Wenqing Lin, Ziming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.09428)  

**Abstract**: Due to the convenience of mobile devices, the online games have become an important part for user entertainments in reality, creating a demand for friend recommendation in online games. However, none of existing approaches can effectively incorporate the multi-modal user features (\emph{e.g.}, images and texts) with the structural information in the friendship graph, due to the following limitations: (1) some of them ignore the high-order structural proximity between users, (2) some fail to learn the pairwise relevance between users at modality-specific level, and (3) some cannot capture both the local and global user preferences on different modalities. By addressing these issues, in this paper, we propose an end-to-end model \textsc{FROG} that better models the user preferences on potential friends. Comprehensive experiments on both offline evaluation and online deployment at \kw{Tencent} have demonstrated the superiority of \textsc{FROG} over existing approaches. 

---
# NoTeS-Bank: Benchmarking Neural Transcription and Search for Scientific Notes Understanding 

**Authors**: Aniket Pal, Sanket Biswas, Alloy Das, Ayush Lodh, Priyanka Banerjee, Soumitri Chattopadhyay, Dimosthenis Karatzas, Josep Llados, C.V. Jawahar  

**Link**: [PDF](https://arxiv.org/pdf/2504.09249)  

**Abstract**: Understanding and reasoning over academic handwritten notes remains a challenge in document AI, particularly for mathematical equations, diagrams, and scientific notations. Existing visual question answering (VQA) benchmarks focus on printed or structured handwritten text, limiting generalization to real-world note-taking. To address this, we introduce NoTeS-Bank, an evaluation benchmark for Neural Transcription and Search in note-based question answering. NoTeS-Bank comprises complex notes across multiple domains, requiring models to process unstructured and multimodal content. The benchmark defines two tasks: (1) Evidence-Based VQA, where models retrieve localized answers with bounding-box evidence, and (2) Open-Domain VQA, where models classify the domain before retrieving relevant documents and answers. Unlike classical Document VQA datasets relying on optical character recognition (OCR) and structured data, NoTeS-BANK demands vision-language fusion, retrieval, and multimodal reasoning. We benchmark state-of-the-art Vision-Language Models (VLMs) and retrieval frameworks, exposing structured transcription and reasoning limitations. NoTeS-Bank provides a rigorous evaluation with NDCG@5, MRR, Recall@K, IoU, and ANLS, establishing a new standard for visual document understanding and reasoning. 

---
# Pneuma: Leveraging LLMs for Tabular Data Representation and Retrieval in an End-to-End System 

**Authors**: Muhammad Imam Luthfi Balaka, David Alexander, Qiming Wang, Yue Gong, Adila Krisnadhi, Raul Castro Fernandez  

**Link**: [PDF](https://arxiv.org/pdf/2504.09207)  

**Abstract**: Finding relevant tables among databases, lakes, and repositories is the first step in extracting value from data. Such a task remains difficult because assessing whether a table is relevant to a problem does not always depend only on its content but also on the context, which is usually tribal knowledge known to the individual or team. While tools like data catalogs and academic data discovery systems target this problem, they rely on keyword search or more complex interfaces, limiting non-technical users' ability to find relevant data. The advent of large language models (LLMs) offers a unique opportunity for users to ask questions directly in natural language, making dataset discovery more intuitive, accessible, and efficient.
In this paper, we introduce Pneuma, a retrieval-augmented generation (RAG) system designed to efficiently and effectively discover tabular data. Pneuma leverages large language models (LLMs) for both table representation and table retrieval. For table representation, Pneuma preserves schema and row-level information to ensure comprehensive data understanding. For table retrieval, Pneuma augments LLMs with traditional information retrieval techniques, such as full-text and vector search, harnessing the strengths of both to improve retrieval performance. To evaluate Pneuma, we generate comprehensive benchmarks that simulate table discovery workload on six real-world datasets including enterprise data, scientific databases, warehousing data, and open data. Our results demonstrate that Pneuma outperforms widely used table search systems (such as full-text search and state-of-the-art RAG systems) in accuracy and resource efficiency. 

---
# RouterKT: Mixture-of-Experts for Knowledge Tracing 

**Authors**: Han Liao, Shuaishuai Zu  

**Link**: [PDF](https://arxiv.org/pdf/2504.08989)  

**Abstract**: Knowledge Tracing (KT) is a fundamental task in Intelligent Tutoring Systems (ITS), which aims to model the dynamic knowledge states of students based on their interaction histories. However, existing KT models often rely on a global forgetting decay mechanism for capturing learning patterns, assuming that students' performance is predominantly influenced by their most recent interactions. Such approaches fail to account for the diverse and complex learning patterns arising from individual differences and varying learning stages. To address this limitation, we propose RouterKT, a novel Mixture-of-Experts (MoE) architecture designed to capture heterogeneous learning patterns by enabling experts to specialize in different patterns without any handcrafted learning pattern bias such as forgetting decay. Specifically, RouterKT introduces a \textbf{person-wise routing mechanism} to effectively model individual-specific learning behaviors and employs \textbf{multi-heads as experts} to enhance the modeling of complex and diverse patterns. Comprehensive experiments on ten benchmark datasets demonstrate that RouterKT exhibits significant flexibility and improves the performance of various KT backbone models, with a maximum average AUC improvement of 3.29\% across different backbones and datasets, outperforming other state-of-the-art models. Moreover, RouterKT demonstrates consistently superior inference efficiency compared to existing approaches based on handcrafted learning pattern bias, highlighting its usability for real-world educational applications. The source code is available at this https URL. 

---
# Code-Craft: Hierarchical Graph-Based Code Summarization for Enhanced Context Retrieval 

**Authors**: David Sounthiraraj, Jared Hancock, Yassin Kortam, Ashok Javvaji, Prabhat Singh, Shaila Shankar  

**Link**: [PDF](https://arxiv.org/pdf/2504.08975)  

**Abstract**: Understanding and navigating large-scale codebases remains a significant challenge in software engineering. Existing methods often treat code as flat text or focus primarily on local structural relationships, limiting their ability to provide holistic, context-aware information retrieval. We present Hierarchical Code Graph Summarization (HCGS), a novel approach that constructs a multi-layered representation of a codebase by generating structured summaries in a bottom-up fashion from a code graph. HCGS leverages the Language Server Protocol for language-agnostic code analysis and employs a parallel level-based algorithm for efficient summary generation. Through extensive evaluation on five diverse codebases totaling 7,531 functions, HCGS demonstrates significant improvements in code retrieval accuracy, achieving up to 82 percentage relative improvement in top-1 retrieval precision for large codebases like libsignal (27.15 percentage points), and perfect Pass@3 scores for smaller repositories. The system's hierarchical approach consistently outperforms traditional code-only retrieval across all metrics, with particularly substantial gains in larger, more complex codebases where understanding function relationships is crucial. 

---
# Enhancing NER Performance in Low-Resource Pakistani Languages using Cross-Lingual Data Augmentation 

**Authors**: Toqeer Ehsan, Thamar Solorio  

**Link**: [PDF](https://arxiv.org/pdf/2504.08792)  

**Abstract**: Named Entity Recognition (NER), a fundamental task in Natural Language Processing (NLP), has shown significant advancements for high-resource languages. However, due to a lack of annotated datasets and limited representation in Pre-trained Language Models (PLMs), it remains understudied and challenging for low-resource languages. To address these challenges, we propose a data augmentation technique that generates culturally plausible sentences and experiments on four low-resource Pakistani languages; Urdu, Shahmukhi, Sindhi, and Pashto. By fine-tuning multilingual masked Large Language Models (LLMs), our approach demonstrates significant improvements in NER performance for Shahmukhi and Pashto. We further explore the capability of generative LLMs for NER and data augmentation using few-shot learning. 

---
# Efficient Evaluation of Large Language Models via Collaborative Filtering 

**Authors**: Xu-Xiang Zhong, Chao Yi, Han-Jia Ye  

**Link**: [PDF](https://arxiv.org/pdf/2504.08781)  

**Abstract**: With the development of Large Language Models (LLMs), numerous benchmarks have been proposed to measure and compare the capabilities of different LLMs. However, evaluating LLMs is costly due to the large number of test instances and their slow inference speed. In this paper, we aim to explore how to efficiently estimate a model's real performance on a given benchmark based on its evaluation results on a small number of instances sampled from the benchmark. Inspired by Collaborative Filtering (CF) in Recommendation Systems (RS), we treat LLMs as users and test instances as items and propose a two-stage method. In the first stage, we treat instance selection as recommending products to users to choose instances that can easily distinguish model performance. In the second stage, we see performance prediction as rating prediction problem in RS to predict the target LLM's behavior on unselected instances. Experiments on multiple LLMs and datasets imply that our method can accurately estimate the target model's performance while largely reducing its inference overhead. 

---
# GridMind: A Multi-Agent NLP Framework for Unified, Cross-Modal NFL Data Insights 

**Authors**: Jordan Chipka, Chris Moyer, Clay Troyer, Tyler Fuelling, Jeremy Hochstedler  

**Link**: [PDF](https://arxiv.org/pdf/2504.08747)  

**Abstract**: The rapid growth of big data and advancements in computational techniques have significantly transformed sports analytics. However, the diverse range of data sources -- including structured statistics, semi-structured formats like sensor data, and unstructured media such as written articles, audio, and video -- creates substantial challenges in extracting actionable insights. These various formats, often referred to as multimodal data, require integration to fully leverage their potential. Conventional systems, which typically prioritize structured data, face limitations when processing and combining these diverse content types, reducing their effectiveness in real-time sports analysis.
To address these challenges, recent research highlights the importance of multimodal data integration for capturing the complexity of real-world sports environments. Building on this foundation, this paper introduces GridMind, a multi-agent framework that unifies structured, semi-structured, and unstructured data through Retrieval-Augmented Generation (RAG) and large language models (LLMs) to facilitate natural language querying of NFL data. This approach aligns with the evolving field of multimodal representation learning, where unified models are increasingly essential for real-time, cross-modal interactions.
GridMind's distributed architecture includes specialized agents that autonomously manage each stage of a prompt -- from interpretation and data retrieval to response synthesis. This modular design enables flexible, scalable handling of multimodal data, allowing users to pose complex, context-rich questions and receive comprehensive, intuitive responses via a conversational interface. 

---
