# SePer: Measure Retrieval Utility Through The Lens Of Semantic Perplexity Reduction 

**Authors**: Lu Dai, Yijie Xu, Jinhui Ye, Hao Liu, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2503.01478)  

**Abstract**: Large Language Models (LLMs) have demonstrated improved generation performance by incorporating externally retrieved knowledge, a process known as retrieval-augmented generation (RAG). Despite the potential of this approach, existing studies evaluate RAG effectiveness by 1) assessing retrieval and generation components jointly, which obscures retrieval's distinct contribution, or 2) examining retrievers using traditional metrics such as NDCG, which creates a gap in understanding retrieval's true utility in the overall generation process. To address the above limitations, in this work, we introduce an automatic evaluation method that measures retrieval quality through the lens of information gain within the RAG framework. Specifically, we propose Semantic Perplexity (SePer), a metric that captures the LLM's internal belief about the correctness of the retrieved information. We quantify the utility of retrieval by the extent to which it reduces semantic perplexity post-retrieval. Extensive experiments demonstrate that SePer not only aligns closely with human preferences but also offers a more precise and efficient evaluation of retrieval utility across diverse RAG scenarios. 

---
# SRAG: Structured Retrieval-Augmented Generation for Multi-Entity Question Answering over Wikipedia Graph 

**Authors**: Teng Lin, Yizhang Zhu, Yuyu Luo, Nan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.01346)  

**Abstract**: Multi-entity question answering (MEQA) poses significant challenges for large language models (LLMs), which often struggle to consolidate scattered information across multiple documents. An example question might be "What is the distribution of IEEE Fellows among various fields of study?", which requires retrieving information from diverse sources e.g., Wikipedia pages. The effectiveness of current retrieval-augmented generation (RAG) methods is limited by the LLMs' capacity to aggregate insights from numerous pages. To address this gap, this paper introduces a structured RAG (SRAG) framework that systematically organizes extracted entities into relational tables (e.g., tabulating entities with schema columns like "name" and "field of study") and then apply table-based reasoning techniques. Our approach decouples retrieval and reasoning, enabling LLMs to focus on structured data analysis rather than raw text aggregation. Extensive experiments on Wikipedia-based multi-entity QA tasks demonstrate that SRAG significantly outperforms state-of-the-art long-context LLMs and RAG solutions, achieving a 29.6% improvement in accuracy. The results underscore the efficacy of structuring unstructured data to enhance LLMs' reasoning capabilities. 

---
# Explainable Depression Detection in Clinical Interviews with Personalized Retrieval-Augmented Generation 

**Authors**: Linhai Zhang, Ziyang Gao, Deyu Zhou, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2503.01315)  

**Abstract**: Depression is a widespread mental health disorder, and clinical interviews are the gold standard for assessment. However, their reliance on scarce professionals highlights the need for automated detection. Current systems mainly employ black-box neural networks, which lack interpretability, which is crucial in mental health contexts. Some attempts to improve interpretability use post-hoc LLM generation but suffer from hallucination. To address these limitations, we propose RED, a Retrieval-augmented generation framework for Explainable depression Detection. RED retrieves evidence from clinical interview transcripts, providing explanations for predictions. Traditional query-based retrieval systems use a one-size-fits-all approach, which may not be optimal for depression detection, as user backgrounds and situations vary. We introduce a personalized query generation module that combines standard queries with user-specific background inferred by LLMs, tailoring retrieval to individual contexts. Additionally, to enhance LLM performance in social intelligence, we augment LLMs by retrieving relevant knowledge from a social intelligence datastore using an event-centric retriever. Experimental results on the real-world benchmark demonstrate RED's effectiveness compared to neural networks and LLM-based baselines. 

---
# U-NIAH: Unified RAG and LLM Evaluation for Long Context Needle-In-A-Haystack 

**Authors**: Yunfan Gao, Yun Xiong, Wenlong Wu, Zijing Huang, Bohan Li, Haofen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2503.00353)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have expanded their context windows to unprecedented lengths, sparking debates about the necessity of Retrieval-Augmented Generation (RAG). To address the fragmented evaluation paradigms and limited cases in existing Needle-in-a-Haystack (NIAH), this paper introduces U-NIAH, a unified framework that systematically compares LLMs and RAG methods in controlled long context settings. Our framework extends beyond traditional NIAH by incorporating multi-needle, long-needle, and needle-in-needle configurations, along with different retrieval settings, while leveraging the synthetic Starlight Academy dataset-a fictional magical universe-to eliminate biases from pre-trained knowledge. Through extensive experiments, we investigate three research questions: (1) performance trade-offs between LLMs and RAG, (2) error patterns in RAG, and (3) RAG's limitations in complex settings. Our findings show that RAG significantly enhances smaller LLMs by mitigating the "lost-in-the-middle" effect and improving robustness, achieving an 82.58% win-rate over LLMs. However, we observe that retrieval noise and reverse chunk ordering degrade performance, while surprisingly, advanced reasoning LLMs exhibit reduced RAG compatibility due to sensitivity to semantic distractors. We identify typical error patterns including omission due to noise, hallucination under high noise critical condition, and self-doubt behaviors. Our work not only highlights the complementary roles of RAG and LLMs, but also provides actionable insights for optimizing deployments. Code: this https URL. 

---
# SolBench: A Dataset and Benchmark for Evaluating Functional Correctness in Solidity Code Completion and Repair 

**Authors**: Zaoyu Chen, Haoran Qin, Nuo Chen, Xiangyu Zhao, Lei Xue, Xiapu Luo, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.01098)  

**Abstract**: Smart contracts are crucial programs on blockchains, and their immutability post-deployment makes functional correctness vital. Despite progress in code completion models, benchmarks for Solidity, the primary smart contract language, are lacking. Existing metrics like BLEU do not adequately assess the functional correctness of generated smart contracts. To fill this gap, we introduce SolBench, a benchmark for evaluating the functional correctness of Solidity smart contracts generated by code completion models. SolBench includes 4,178 functions from 1,155 Ethereum-deployed contracts. Testing advanced models revealed challenges in generating correct code without context, as Solidity functions rely on context-defined variables and interfaces. To address this, we propose a Retrieval-Augmented Code Repair framework. In this framework, an executor verifies functional correctness, and if necessary, an LLM repairs the code using retrieved snippets informed by executor traces. We conduct a comprehensive evaluation of both closed-source and open-source LLMs across various model sizes and series to assess their performance in smart contract completion. The results show that code repair and retrieval techniques effectively enhance the correctness of smart contract completion while reducing computational costs. 

---
# Retrieval-Augmented Perception: High-Resolution Image Perception Meets Visual RAG 

**Authors**: Wenbin Wang, Yongcheng Jing, Liang Ding, Yingjie Wang, Li Shen, Yong Luo, Bo Du, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2503.01222)  

**Abstract**: High-resolution (HR) image perception remains a key challenge in multimodal large language models (MLLMs). To overcome the limitations of existing methods, this paper shifts away from prior dedicated heuristic approaches and revisits the most fundamental idea to HR perception by enhancing the long-context capability of MLLMs, driven by recent advances in long-context techniques like retrieval-augmented generation (RAG) for general LLMs. Towards this end, this paper presents the first study exploring the use of RAG to address HR perception challenges. Specifically, we propose Retrieval-Augmented Perception (RAP), a training-free framework that retrieves and fuses relevant image crops while preserving spatial context using the proposed Spatial-Awareness Layout. To accommodate different tasks, the proposed Retrieved-Exploration Search (RE-Search) dynamically selects the optimal number of crops based on model confidence and retrieval scores. Experimental results on HR benchmarks demonstrate the significant effectiveness of RAP, with LLaVA-v1.5-13B achieving a 43% improvement on $V^*$ Bench and 19% on HR-Bench. 

---
# Qilin: A Multimodal Information Retrieval Dataset with APP-level User Sessions 

**Authors**: Jia Chen, Qian Dong, Haitao Li, Xiaohui He, Yan Gao, Shaosheng Cao, Yi Wu, Ping Yang, Chen Xu, Yao Hu, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.00501)  

**Abstract**: User-generated content (UGC) communities, especially those featuring multimodal content, improve user experiences by integrating visual and textual information into results (or items). The challenge of improving user experiences in complex systems with search and recommendation (S\&R) services has drawn significant attention from both academia and industry these years. However, the lack of high-quality datasets has limited the research progress on multimodal S\&R. To address the growing need for developing better S\&R services, we present a novel multimodal information retrieval dataset in this paper, namely Qilin. The dataset is collected from Xiaohongshu, a popular social platform with over 300 million monthly active users and an average search penetration rate of over 70\%. In contrast to existing datasets, \textsf{Qilin} offers a comprehensive collection of user sessions with heterogeneous results like image-text notes, video notes, commercial notes, and direct answers, facilitating the development of advanced multimodal neural retrieval models across diverse task settings. To better model user satisfaction and support the analysis of heterogeneous user behaviors, we also collect extensive APP-level contextual signals and genuine user feedback. Notably, Qilin contains user-favored answers and their referred results for search requests triggering the Deep Query Answering (DQA) module. This allows not only the training \& evaluation of a Retrieval-augmented Generation (RAG) pipeline, but also the exploration of how such a module would affect users' search behavior. Through comprehensive analysis and experiments, we provide interesting findings and insights for further improving S\&R systems. We hope that \textsf{Qilin} will significantly contribute to the advancement of multimodal content platforms with S\&R services in the future. 

---
# DeepSolution: Boosting Complex Engineering Solution Design via Tree-based Exploration and Bi-point Thinking 

**Authors**: Zhuoqun Li, Haiyang Yu, Xuanang Chen, Hongyu Lin, Yaojie Lu, Fei Huang, Xianpei Han, Yongbin Li, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2502.20730)  

**Abstract**: Designing solutions for complex engineering challenges is crucial in human production activities. However, previous research in the retrieval-augmented generation (RAG) field has not sufficiently addressed tasks related to the design of complex engineering solutions. To fill this gap, we introduce a new benchmark, SolutionBench, to evaluate a system's ability to generate complete and feasible solutions for engineering problems with multiple complex constraints. To further advance the design of complex engineering solutions, we propose a novel system, SolutionRAG, that leverages the tree-based exploration and bi-point thinking mechanism to generate reliable solutions. Extensive experimental results demonstrate that SolutionRAG achieves state-of-the-art (SOTA) performance on the SolutionBench, highlighting its potential to enhance the automation and reliability of complex engineering solution design in real-world applications. 

---
# Towards Efficient Educational Chatbots: Benchmarking RAG Frameworks 

**Authors**: Umar Ali Khan, Ekram Khan, Fiza Khan, Athar Ali Moinuddin  

**Link**: [PDF](https://arxiv.org/pdf/2503.00781)  

**Abstract**: Large Language Models (LLMs) have proven immensely beneficial in education by capturing vast amounts of literature-based information, allowing them to generate context without relying on external sources. In this paper, we propose a generative AI-powered GATE question-answering framework (GATE stands for Graduate Aptitude Test in Engineering) that leverages LLMs to explain GATE solutions and support students in their exam preparation. We conducted extensive benchmarking to select the optimal embedding model and LLM, evaluating our framework based on criteria such as latency, faithfulness, and relevance, with additional validation through human evaluation. Our chatbot integrates state-of-the-art embedding models and LLMs to deliver accurate, context-aware responses. Through rigorous experimentation, we identified configurations that balance performance and computational efficiency, ensuring a reliable chatbot to serve students' needs. Additionally, we discuss the challenges faced in data processing and modeling and implemented solutions. Our work explores the application of Retrieval-Augmented Generation (RAG) for GATE Q/A explanation tasks, and our findings demonstrate significant improvements in retrieval accuracy and response quality. This research offers practical insights for developing effective AI-driven educational tools while highlighting areas for future enhancement in usability and scalability. 

---
# Pseudo-Knowledge Graph: Meta-Path Guided Retrieval and In-Graph Text for RAG-Equipped LLM 

**Authors**: Yuxin Yang, Haoyang Wu, Tao Wang, Jia Yang, Hao Ma, Guojie Luo  

**Link**: [PDF](https://arxiv.org/pdf/2503.00309)  

**Abstract**: The advent of Large Language Models (LLMs) has revolutionized natural language processing. However, these models face challenges in retrieving precise information from vast datasets. Retrieval-Augmented Generation (RAG) was developed to combining LLMs with external information retrieval systems to enhance the accuracy and context of responses. Despite improvements, RAG still struggles with comprehensive retrieval in high-volume, low-information-density databases and lacks relational awareness, leading to fragmented answers.
To address this, this paper introduces the Pseudo-Knowledge Graph (PKG) framework, designed to overcome these limitations by integrating Meta-path Retrieval, In-graph Text and Vector Retrieval into LLMs. By preserving natural language text and leveraging various retrieval techniques, the PKG offers a richer knowledge representation and improves accuracy in information retrieval. Extensive evaluations using Open Compass and MultiHop-RAG benchmarks demonstrate the framework's effectiveness in managing large volumes of data and complex relationships. 

---
# SAGE: A Framework of Precise Retrieval for RAG 

**Authors**: Jintao Zhang, Guoliang Li, Jinyang Su  

**Link**: [PDF](https://arxiv.org/pdf/2503.01713)  

**Abstract**: Retrieval-augmented generation (RAG) has demonstrated significant proficiency in conducting question-answering (QA) tasks within a specified corpus. Nonetheless, numerous failure instances of RAG in QA still exist. These failures are not solely attributable to the limitations of Large Language Models (LLMs); instead, they predominantly arise from the retrieval of inaccurate information for LLMs due to two limitations: (1) Current RAG methods segment the corpus without considering semantics, making it difficult to find relevant context due to impaired correlation between questions and the segments. (2) There is a trade-off between missing essential context with fewer context retrieved and getting irrelevant context with more context retrieved.
In this paper, we introduce a RAG framework (SAGE), to overcome these limitations. First, to address the segmentation issue without considering semantics, we propose to train a semantic segmentation model. This model is trained to segment the corpus into semantically complete chunks. Second, to ensure that only the most relevant chunks are retrieved while the irrelevant ones are ignored, we design a chunk selection algorithm to dynamically select chunks based on the decreasing speed of the relevance score, leading to a more relevant selection. Third, to further ensure the precision of the retrieved chunks, we propose letting LLMs assess whether retrieved chunks are excessive or lacking and then adjust the amount of context accordingly. Experiments show that SAGE outperforms baselines by 61.25% in the quality of QA on average. Moreover, by avoiding retrieving noisy context, SAGE lowers the cost of the tokens consumed in LLM inference and achieves a 49.41% enhancement in cost efficiency on average. Additionally, our work offers valuable insights for boosting RAG. 

---
# Simple Is Effective: The Roles of Graphs and Large Language Models in Knowledge-Graph-Based Retrieval-Augmented Generation 

**Authors**: Mufei Li, Siqi Miao, Pan Li  

**Link**: [PDF](https://arxiv.org/pdf/2410.20724)  

**Abstract**: Large Language Models (LLMs) demonstrate strong reasoning abilities but face limitations such as hallucinations and outdated knowledge. Knowledge Graph (KG)-based Retrieval-Augmented Generation (RAG) addresses these issues by grounding LLM outputs in structured external knowledge from KGs. However, current KG-based RAG frameworks still struggle to optimize the trade-off between retrieval effectiveness and efficiency in identifying a suitable amount of relevant graph information for the LLM to digest. We introduce SubgraphRAG, extending the KG-based RAG framework that retrieves subgraphs and leverages LLMs for reasoning and answer prediction. Our approach innovatively integrates a lightweight multilayer perceptron with a parallel triple-scoring mechanism for efficient and flexible subgraph retrieval while encoding directional structural distances to enhance retrieval effectiveness. The size of retrieved subgraphs can be flexibly adjusted to match the query's need and the downstream LLM's capabilities. This design strikes a balance between model complexity and reasoning power, enabling scalable and generalizable retrieval processes. Notably, based on our retrieved subgraphs, smaller LLMs like Llama3.1-8B-Instruct deliver competitive results with explainable reasoning, while larger models like GPT-4o achieve state-of-the-art accuracy compared with previous baselines -- all without fine-tuning. Extensive evaluations on the WebQSP and CWQ benchmarks highlight SubgraphRAG's strengths in efficiency, accuracy, and reliability by reducing hallucinations and improving response grounding. 

---
# SafeAuto: Knowledge-Enhanced Safe Autonomous Driving with Multimodal Foundation Models 

**Authors**: Jiawei Zhang, Xuan Yang, Taiqi Wang, Yu Yao, Aleksandr Petiushko, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.00211)  

**Abstract**: Traditional autonomous driving systems often struggle to integrate high-level reasoning with low-level control, resulting in suboptimal and sometimes unsafe driving behaviors. The emergence of Multimodal Large Language Models (MLLMs), which can process both visual and textual data, presents an opportunity to unify perception and reasoning tasks within a single framework. However, effectively embedding precise safety knowledge into MLLMs for autonomous driving remains a significant challenge. To address this, we propose SafeAuto, a novel framework that enhances MLLM-based autonomous driving systems by incorporating both unstructured and structured knowledge. Specifically, we first introduce the Position-Dependent Cross-Entropy (PDCE) loss function, designed to improve the accuracy of low-level control signal predictions when numerical values are represented as text. Second, to ensure safe autonomous driving by explicitly integrating precise safety knowledge into the MLLM, we develop a reasoning component for SafeAuto. This component translates driving safety regulations into first-order logic rules (e.g., "red light => stop") and incorporates these rules into a probabilistic graphical model, such as a Markov Logic Network (MLN). The MLN is trained to verify the predicted next actions using environmental attributes identified by attribute recognition models (e.g., detecting a red light) to form the predicates. Additionally, we construct a Multimodal RAG model that leverages video data, control signals, and environmental attributes to learn more effectively from past similar driving experiences. By integrating PDCE, MLN, and Multimodal RAG, SafeAuto significantly outperforms existing baselines across multiple datasets. This advancement enables more accurate, reliable, and safer autonomous driving systems that learn from experience, obey traffic laws, and perform precise control actions. 

---
