# KnowTrace: Bootstrapping Iterative Retrieval-Augmented Generation with Structured Knowledge Tracing 

**Authors**: Rui Li, Quanyu Dai, Zeyu Zhang, Xu Chen, Zhenhua Dong, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.20245)  

**Abstract**: Recent advances in retrieval-augmented generation (RAG) furnish large language models (LLMs) with iterative retrievals of relevant information to handle complex multi-hop questions. These methods typically alternate between LLM reasoning and retrieval to accumulate external information into the LLM's context. However, the ever-growing context inherently imposes an increasing burden on the LLM to perceive connections among critical information pieces, with futile reasoning steps further exacerbating this overload issue. In this paper, we present KnowTrace, an elegant RAG framework to (1) mitigate the context overload and (2) bootstrap higher-quality multi-step reasoning. Instead of simply piling the retrieved contents, KnowTrace autonomously traces out desired knowledge triplets to organize a specific knowledge graph relevant to the input question. Such a structured workflow not only empowers the LLM with an intelligible context for inference, but also naturally inspires a reflective mechanism of knowledge backtracing to identify contributive LLM generations as process supervision data for self-bootstrapping. Extensive experiments show that KnowTrace consistently surpasses existing methods across three multi-hop question answering benchmarks, and the bootstrapped version further amplifies the gains. 

---
# MA-RAG: Multi-Agent Retrieval-Augmented Generation via Collaborative Chain-of-Thought Reasoning 

**Authors**: Thang Nguyen, Peter Chin, Yu-Wing Tai  

**Link**: [PDF](https://arxiv.org/pdf/2505.20096)  

**Abstract**: We present MA-RAG, a Multi-Agent framework for Retrieval-Augmented Generation (RAG) that addresses the inherent ambiguities and reasoning challenges in complex information-seeking tasks. Unlike conventional RAG methods that rely on either end-to-end fine-tuning or isolated component enhancements, MA-RAG orchestrates a collaborative set of specialized AI agents: Planner, Step Definer, Extractor, and QA Agents, to tackle each stage of the RAG pipeline with task-aware reasoning. Ambiguities may arise from underspecified queries, sparse or indirect evidence in retrieved documents, or the need to integrate information scattered across multiple sources. MA-RAG mitigates these challenges by decomposing the problem into subtasks, such as query disambiguation, evidence extraction, and answer synthesis, and dispatching them to dedicated agents equipped with chain-of-thought prompting. These agents communicate intermediate reasoning and progressively refine the retrieval and synthesis process. Our design allows fine-grained control over information flow without any model fine-tuning. Crucially, agents are invoked on demand, enabling a dynamic and efficient workflow that avoids unnecessary computation. This modular and reasoning-driven architecture enables MA-RAG to deliver robust, interpretable results. Experiments on multi-hop and ambiguous QA benchmarks demonstrate that MA-RAG outperforms state-of-the-art training-free baselines and rivals fine-tuned systems, validating the effectiveness of collaborative agent-based reasoning in RAG. 

---
# NeuSym-RAG: Hybrid Neural Symbolic Retrieval with Multiview Structuring for PDF Question Answering 

**Authors**: Ruisheng Cao, Hanchong Zhang, Tiancheng Huang, Zhangyi Kang, Yuxin Zhang, Liangtai Sun, Hanqi Li, Yuxun Miao, Shuai Fan, Lu Chen, Kai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.19754)  

**Abstract**: The increasing number of academic papers poses significant challenges for researchers to efficiently acquire key details. While retrieval augmented generation (RAG) shows great promise in large language model (LLM) based automated question answering, previous works often isolate neural and symbolic retrieval despite their complementary strengths. Moreover, conventional single-view chunking neglects the rich structure and layout of PDFs, e.g., sections and tables. In this work, we propose NeuSym-RAG, a hybrid neural symbolic retrieval framework which combines both paradigms in an interactive process. By leveraging multi-view chunking and schema-based parsing, NeuSym-RAG organizes semi-structured PDF content into both the relational database and vectorstore, enabling LLM agents to iteratively gather context until sufficient to generate answers. Experiments on three full PDF-based QA datasets, including a self-annotated one AIRQA-REAL, show that NeuSym-RAG stably defeats both the vector-based RAG and various structured baselines, highlighting its capacity to unify both retrieval schemes and utilize multiple views. Code and data are publicly available at this https URL. 

---
# DoctorRAG: Medical RAG Fusing Knowledge with Patient Analogy through Textual Gradients 

**Authors**: Yuxing Lu, Gecheng Fu, Wei Wu, Xukai Zhao, Sin Yee Goi, Jinzhuo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19538)  

**Abstract**: Existing medical RAG systems mainly leverage knowledge from medical knowledge bases, neglecting the crucial role of experiential knowledge derived from similar patient cases -- a key component of human clinical reasoning. To bridge this gap, we propose DoctorRAG, a RAG framework that emulates doctor-like reasoning by integrating both explicit clinical knowledge and implicit case-based experience. DoctorRAG enhances retrieval precision by first allocating conceptual tags for queries and knowledge sources, together with a hybrid retrieval mechanism from both relevant knowledge and patient. In addition, a Med-TextGrad module using multi-agent textual gradients is integrated to ensure that the final output adheres to the retrieved knowledge and patient query. Comprehensive experiments on multilingual, multitask datasets demonstrate that DoctorRAG significantly outperforms strong baseline RAG models and gains improvements from iterative refinements. Our approach generates more accurate, relevant, and comprehensive responses, taking a step towards more doctor-like medical reasoning systems. 

---
# Federated Retrieval-Augmented Generation: A Systematic Mapping Study 

**Authors**: Abhijit Chakraborty, Chahana Dahal, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.18906)  

**Abstract**: Federated Retrieval-Augmented Generation (Federated RAG) combines Federated Learning (FL), which enables distributed model training without exposing raw data, with Retrieval-Augmented Generation (RAG), which improves the factual accuracy of language models by grounding outputs in external knowledge. As large language models are increasingly deployed in privacy-sensitive domains such as healthcare, finance, and personalized assistance, Federated RAG offers a promising framework for secure, knowledge-intensive natural language processing (NLP). To the best of our knowledge, this paper presents the first systematic mapping study of Federated RAG, covering literature published between 2020 and 2025. Following Kitchenham's guidelines for evidence-based software engineering, we develop a structured classification of research focuses, contribution types, and application domains. We analyze architectural patterns, temporal trends, and key challenges, including privacy-preserving retrieval, cross-client heterogeneity, and evaluation limitations. Our findings synthesize a rapidly evolving body of research, identify recurring design patterns, and surface open questions, providing a foundation for future work at the intersection of RAG and federated systems. 

---
# Removal of Hallucination on Hallucination: Debate-Augmented RAG 

**Authors**: Wentao Hu, Wengyu Zhang, Yiyang Jiang, Chen Jason Zhang, Xiaoyong Wei, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.18581)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances factual accuracy by integrating external knowledge, yet it introduces a critical issue: erroneous or biased retrieval can mislead generation, compounding hallucinations, a phenomenon we term Hallucination on Hallucination. To address this, we propose Debate-Augmented RAG (DRAG), a training-free framework that integrates Multi-Agent Debate (MAD) mechanisms into both retrieval and generation stages. In retrieval, DRAG employs structured debates among proponents, opponents, and judges to refine retrieval quality and ensure factual reliability. In generation, DRAG introduces asymmetric information roles and adversarial debates, enhancing reasoning robustness and mitigating factual inconsistencies. Evaluations across multiple tasks demonstrate that DRAG improves retrieval reliability, reduces RAG-induced hallucinations, and significantly enhances overall factual accuracy. Our code is available at this https URL. 

---
# Retrieval Augmented Generation-based Large Language Models for Bridging Transportation Cybersecurity Legal Knowledge Gaps 

**Authors**: Khandakar Ashrafi Akbar, Md Nahiyan Uddin, Latifur Khan, Trayce Hockstad, Mizanur Rahman, Mashrur Chowdhury, Bhavani Thuraisingham  

**Link**: [PDF](https://arxiv.org/pdf/2505.18426)  

**Abstract**: As connected and automated transportation systems evolve, there is a growing need for federal and state authorities to revise existing laws and develop new statutes to address emerging cybersecurity and data privacy challenges. This study introduces a Retrieval-Augmented Generation (RAG) based Large Language Model (LLM) framework designed to support policymakers by extracting relevant legal content and generating accurate, inquiry-specific responses. The framework focuses on reducing hallucinations in LLMs by using a curated set of domain-specific questions to guide response generation. By incorporating retrieval mechanisms, the system enhances the factual grounding and specificity of its outputs. Our analysis shows that the proposed RAG-based LLM outperforms leading commercial LLMs across four evaluation metrics: AlignScore, ParaScore, BERTScore, and ROUGE, demonstrating its effectiveness in producing reliable and context-aware legal insights. This approach offers a scalable, AI-driven method for legislative analysis, supporting efforts to update legal frameworks in line with advancements in transportation technologies. 

---
# BRIT: Bidirectional Retrieval over Unified Image-Text Graph 

**Authors**: Ainulla Khan, Yamada Moyuru, Srinidhi Akella  

**Link**: [PDF](https://arxiv.org/pdf/2505.18450)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising technique to enhance the quality and relevance of responses generated by large language models. While recent advancements have mainly focused on improving RAG for text-based queries, RAG on multi-modal documents containing both texts and images has not been fully explored. Especially when fine-tuning does not work. This paper proposes BRIT, a novel multi-modal RAG framework that effectively unifies various text-image connections in the document into a multi-modal graph and retrieves the texts and images as a query-specific sub-graph. By traversing both image-to-text and text-to-image paths in the graph, BRIT retrieve not only directly query-relevant images and texts but also further relevant contents to answering complex cross-modal multi-hop questions. To evaluate the effectiveness of BRIT, we introduce MM-RAG test set specifically designed for multi-modal question answering tasks that require to understand the text-image relations. Our comprehensive experiments demonstrate the superiority of BRIT, highlighting its ability to handle cross-modal questions on the multi-modal documents. 

---
# MetaGen Blended RAG: Higher Accuracy for Domain-Specific Q&A Without Fine-Tuning 

**Authors**: Kunal Sawarkar, Shivam R. Solanki, Abhilasha Mangal  

**Link**: [PDF](https://arxiv.org/pdf/2505.18247)  

**Abstract**: Despite the widespread exploration of Retrieval-Augmented Generation (RAG), its deployment in enterprises for domain-specific datasets remains limited due to poor answer accuracy. These corpora, often shielded behind firewalls in private enterprise knowledge bases, having complex, domain-specific terminology, rarely seen by LLMs during pre-training; exhibit significant semantic variability across domains (like networking, military, or legal, etc.), or even within a single domain like medicine, and thus result in poor context precision for RAG systems. Currently, in such situations, fine-tuning or RAG with fine-tuning is attempted, but these approaches are slow, expensive, and lack generalization for accuracy as the new domain-specific data emerges. We propose an approach for Enterprise Search that focuses on enhancing the retriever for a domain-specific corpus through hybrid query indexes and metadata enrichment. This 'MetaGen Blended RAG' method constructs a metadata generation pipeline using key concepts, topics, and acronyms, and then creates a metadata-enriched hybrid index with boosted search queries. This approach avoids overfitting and generalizes effectively across domains. On the PubMedQA benchmark for the biomedical domain, the proposed method achieves 82% retrieval accuracy and 77% RAG accuracy, surpassing all previous RAG accuracy results without fine-tuning and sets a new benchmark for zero-shot results while outperforming much larger models like GPT3.5. The results are even comparable to the best fine-tuned models on this dataset, and we further demonstrate the robustness and scalability of the approach by evaluating it on other Q&A datasets like SQuAD, NQ etc. 

---
# Walk&Retrieve: Simple Yet Effective Zero-shot Retrieval-Augmented Generation via Knowledge Graph Walks 

**Authors**: Martin Böckling, Heiko Paulheim, Andreea Iana  

**Link**: [PDF](https://arxiv.org/pdf/2505.16849)  

**Abstract**: Large Language Models (LLMs) have showcased impressive reasoning abilities, but often suffer from hallucinations or outdated knowledge. Knowledge Graph (KG)-based Retrieval-Augmented Generation (RAG) remedies these shortcomings by grounding LLM responses in structured external information from a knowledge base. However, many KG-based RAG approaches struggle with (i) aligning KG and textual representations, (ii) balancing retrieval accuracy and efficiency, and (iii) adapting to dynamically updated KGs. In this work, we introduce Walk&Retrieve, a simple yet effective KG-based framework that leverages walk-based graph traversal and knowledge verbalization for corpus generation for zero-shot RAG. Built around efficient KG walks, our method does not require fine-tuning on domain-specific data, enabling seamless adaptation to KG updates, reducing computational overhead, and allowing integration with any off-the-shelf backbone LLM. Despite its simplicity, Walk&Retrieve performs competitively, often outperforming existing RAG systems in response accuracy and hallucination reduction. Moreover, it demonstrates lower query latency and robust scalability to large KGs, highlighting the potential of lightweight retrieval strategies as strong baselines for future RAG research. 

---
# A Survey of LLM $\times$ DATA 

**Authors**: Xuanhe Zhou, Junxuan He, Wei Zhou, Haodong Chen, Zirui Tang, Haoyu Zhao, Xin Tong, Guoliang Li, Youmin Chen, Jun Zhou, Zhaojun Sun, Binyuan Hui, Shuo Wang, Conghui He, Zhiyuan Liu, Jingren Zhou, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.18458)  

**Abstract**: The integration of large language model (LLM) and data management (DATA) is rapidly redefining both domains. In this survey, we comprehensively review the bidirectional relationships. On the one hand, DATA4LLM, spanning large-scale data processing, storage, and serving, feeds LLMs with high quality, diversity, and timeliness of data required for stages like pre-training, post-training, retrieval-augmented generation, and agentic workflows: (i) Data processing for LLMs includes scalable acquisition, deduplication, filtering, selection, domain mixing, and synthetic augmentation; (ii) Data Storage for LLMs focuses on efficient data and model formats, distributed and heterogeneous storage hierarchies, KV-cache management, and fault-tolerant checkpointing; (iii) Data serving for LLMs tackles challenges in RAG (e.g., knowledge post-processing), LLM inference (e.g., prompt compression, data provenance), and training strategies (e.g., data packing and shuffling). On the other hand, in LLM4DATA, LLMs are emerging as general-purpose engines for data management. We review recent advances in (i) data manipulation, including automatic data cleaning, integration, discovery; (ii) data analysis, covering reasoning over structured, semi-structured, and unstructured data, and (iii) system optimization (e.g., configuration tuning, query rewriting, anomaly diagnosis), powered by LLM techniques like retrieval-augmented prompting, task-specialized fine-tuning, and multi-agent collaboration. 

---
# POQD: Performance-Oriented Query Decomposer for Multi-vector retrieval 

**Authors**: Yaoyang Liu, Junlin Li, Yinjun Wu, Zhen Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.19189)  

**Abstract**: Although Multi-Vector Retrieval (MVR) has achieved the state of the art on many information retrieval (IR) tasks, its performance highly depends on how to decompose queries into smaller pieces, say phrases or tokens. However, optimizing query decomposition for MVR performance is not end-to-end differentiable. Even worse, jointly solving this problem and training the downstream retrieval-based systems, say RAG systems could be highly inefficient. To overcome these challenges, we propose Performance-Oriented Query Decomposer (POQD), a novel query decomposition framework for MVR. POQD leverages one LLM for query decomposition and searches the optimal prompt with an LLM-based optimizer. We further propose an end-to-end training algorithm to alternatively optimize the prompt for query decomposition and the downstream models. This algorithm can achieve superior MVR performance at a reasonable training cost as our theoretical analysis suggests. POQD can be integrated seamlessly into arbitrary retrieval-based systems such as Retrieval-Augmented Generation (RAG) systems. Extensive empirical studies on representative RAG-based QA tasks show that POQD outperforms existing query decomposition strategies in both retrieval performance and end-to-end QA accuracy. POQD is available at this https URL. 

---
# GainRAG: Preference Alignment in Retrieval-Augmented Generation through Gain Signal Synthesis 

**Authors**: Yi Jiang, Sendong Zhao, Jianbo Li, Haochun Wang, Bing Qin  

**Link**: [PDF](https://arxiv.org/pdf/2505.18710)  

**Abstract**: The Retrieval-Augmented Generation (RAG) framework introduces a retrieval module to dynamically inject retrieved information into the input context of large language models (LLMs), and has demonstrated significant success in various NLP tasks. However, the current study points out that there is a preference gap between retrievers and LLMs in the RAG framework, which limit the further improvement of system performance. Some highly relevant passages may interfere with LLM reasoning because they contain complex or contradictory information; while some indirectly related or even inaccurate content may help LLM generate more accurate answers by providing suggestive information or logical clues. To solve this, we propose GainRAG, a novel approach that aligns the retriever's and LLM's preferences by defining a new metric, "gain", which measure how well an input passage contributes to correct outputs. Specifically, we propose a method to estimate these gain signals and train a middleware that aligns the preferences of the retriever and the LLM using only limited data. In addition, we introduce a pseudo-passage strategy to mitigate degradation. The experimental results on 6 datasets verify the effectiveness of GainRAG. 

---
# The Silent Saboteur: Imperceptible Adversarial Attacks against Black-Box Retrieval-Augmented Generation Systems 

**Authors**: Hongru Song, Yu-an Liu, Ruqing Zhang, Jiafeng Guo, Jianming Lv, Maarten de Rijke, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.18583)  

**Abstract**: We explore adversarial attacks against retrieval-augmented generation (RAG) systems to identify their vulnerabilities. We focus on generating human-imperceptible adversarial examples and introduce a novel imperceptible retrieve-to-generate attack against RAG. This task aims to find imperceptible perturbations that retrieve a target document, originally excluded from the initial top-$k$ candidate set, in order to influence the final answer generation. To address this task, we propose ReGENT, a reinforcement learning-based framework that tracks interactions between the attacker and the target RAG and continuously refines attack strategies based on relevance-generation-naturalness rewards. Experiments on newly constructed factual and non-factual question-answering benchmarks demonstrate that ReGENT significantly outperforms existing attack methods in misleading RAG systems with small imperceptible text perturbations. 

---
# syftr: Pareto-Optimal Generative AI 

**Authors**: Alexander Conway, Debadeepta Dey, Stefan Hackmann, Matthew Hausknecht, Michael Schmidt, Mark Steadman, Nick Volynets  

**Link**: [PDF](https://arxiv.org/pdf/2505.20266)  

**Abstract**: Retrieval-Augmented Generation (RAG) pipelines are central to applying large language models (LLMs) to proprietary or dynamic data. However, building effective RAG flows is complex, requiring careful selection among vector databases, embedding models, text splitters, retrievers, and synthesizing LLMs. The challenge deepens with the rise of agentic paradigms. Modules like verifiers, rewriters, and rerankers-each with intricate hyperparameter dependencies have to be carefully tuned. Balancing tradeoffs between latency, accuracy, and cost becomes increasingly difficult in performance-sensitive applications.
We introduce syftr, a framework that performs efficient multi-objective search over a broad space of agentic and non-agentic RAG configurations. Using Bayesian Optimization, syftr discovers Pareto-optimal flows that jointly optimize task accuracy and cost. A novel early-stopping mechanism further improves efficiency by pruning clearly suboptimal candidates. Across multiple RAG benchmarks, syftr finds flows which are on average approximately 9 times cheaper while preserving most of the accuracy of the most accurate flows on the Pareto-frontier. Furthermore, syftr's ability to design and optimize allows integrating new modules, making it even easier and faster to realize high-performing generative AI pipelines. 

---
# DGRAG: Distributed Graph-based Retrieval-Augmented Generation in Edge-Cloud Systems 

**Authors**: Wenqing Zhou, Yuxuan Yan, Qianqian Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.19847)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising approach to enhance the capabilities of language models by integrating external knowledge. Due to the diversity of data sources and the constraints of memory and computing resources, real-world data is often scattered in multiple devices. Conventional RAGs that store massive amounts of scattered data centrally face increasing privacy concerns and high computational costs. Additionally, RAG in a central node raises latency issues when searching over a large-scale knowledge base. To address these challenges, we propose a distributed Knowledge Graph-based RAG approach, referred to as DGRAG, in an edge-cloud system, where each edge device maintains a local knowledge base without the need to share it with the cloud, instead sharing only summaries of its knowledge. Specifically, DGRAG has two main phases. In the Distributed Knowledge Construction phase, DGRAG organizes local knowledge using knowledge graphs, generating subgraph summaries and storing them in a summary database in the cloud as information sharing. In the Collaborative Retrieval and Generation phase, DGRAG first performs knowledge retrieval and answer generation locally, and a gate mechanism determines whether the query is beyond the scope of local knowledge or processing capabilities. For queries that exceed the local knowledge scope, the cloud retrieves knowledge from the most relevant edges based on the summaries and generates a more precise answer. Experimental results demonstrate the effectiveness of the proposed DGRAG approach in significantly improving the quality of question-answering tasks over baseline approaches. 

---
# LLM-Agent-Controller: A Universal Multi-Agent Large Language Model System as a Control Engineer 

**Authors**: Rasoul Zahedifar, Sayyed Ali Mirghasemi, Mahdieh Soleymani Baghshah, Alireza Taheri  

**Link**: [PDF](https://arxiv.org/pdf/2505.19567)  

**Abstract**: This study presents the LLM-Agent-Controller, a multi-agent large language model (LLM) system developed to address a wide range of problems in control engineering (Control Theory). The system integrates a central controller agent with multiple specialized auxiliary agents, responsible for tasks such as controller design, model representation, control analysis, time-domain response, and simulation. A supervisor oversees high-level decision-making and workflow coordination, enhancing the system's reliability and efficiency. The LLM-Agent-Controller incorporates advanced capabilities, including Retrieval-Augmented Generation (RAG), Chain-of-Thought reasoning, self-criticism and correction, efficient memory handling, and user-friendly natural language communication. It is designed to function without requiring users to have prior knowledge of Control Theory, enabling them to input problems in plain language and receive complete, real-time solutions. To evaluate the system, we propose new performance metrics assessing both individual agents and the system as a whole. We test five categories of Control Theory problems and benchmark performance across three advanced LLMs. Additionally, we conduct a comprehensive qualitative conversational analysis covering all key services. Results show that the LLM-Agent-Controller successfully solved 83% of general tasks, with individual agents achieving an average success rate of 87%. Performance improved with more advanced LLMs. This research demonstrates the potential of multi-agent LLM architectures to solve complex, domain-specific problems. By integrating specialized agents, supervisory control, and advanced reasoning, the LLM-Agent-Controller offers a scalable, robust, and accessible solution framework that can be extended to various technical domains. 

---
# Investigating Pedagogical Teacher and Student LLM Agents: Genetic Adaptation Meets Retrieval Augmented Generation Across Learning Style 

**Authors**: Debdeep Sanyal, Agniva Maiti, Umakanta Maharana, Dhruv Kumar, Ankur Mali, C. Lee Giles, Murari Mandal  

**Link**: [PDF](https://arxiv.org/pdf/2505.19173)  

**Abstract**: Effective teaching requires adapting instructional strategies to accommodate the diverse cognitive and behavioral profiles of students, a persistent challenge in education and teacher training. While Large Language Models (LLMs) offer promise as tools to simulate such complex pedagogical environments, current simulation frameworks are limited in two key respects: (1) they often reduce students to static knowledge profiles, and (2) they lack adaptive mechanisms for modeling teachers who evolve their strategies in response to student feedback. To address these gaps, \textbf{we introduce a novel simulation framework that integrates LLM-based heterogeneous student agents with a self-optimizing teacher agent}. The teacher agent's pedagogical policy is dynamically evolved using a genetic algorithm, allowing it to discover and refine effective teaching strategies based on the aggregate performance of diverse learners. In addition, \textbf{we propose Persona-RAG}, a Retrieval Augmented Generation module that enables student agents to retrieve knowledge tailored to their individual learning styles. Persona-RAG preserves the retrieval accuracy of standard RAG baselines while enhancing personalization, an essential factor in modeling realistic educational scenarios. Through extensive experiments, we demonstrate how our framework supports the emergence of distinct and interpretable teaching patterns when interacting with varied student populations. Our results highlight the potential of LLM-driven simulations to inform adaptive teaching practices and provide a testbed for training human educators in controlled, data-driven environments. 

---
# LLMs for Supply Chain Management 

**Authors**: Haojie Wang, Jiuyun Jiang, L. Jeff Hong, Guangxin Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.18597)  

**Abstract**: The development of large language models (LLMs) has provided new tools for research in supply chain management (SCM). In this paper, we introduce a retrieval-augmented generation (RAG) framework that dynamically integrates external knowledge into the inference process, and develop a domain-specialized SCM LLM, which demonstrates expert-level competence by passing standardized SCM examinations and beer game tests. We further employ the use of LLMs to conduct horizontal and vertical supply chain games, in order to analyze competition and cooperation within supply chains. Our experiments show that RAG significantly improves performance on SCM tasks. Moreover, game-theoretic analysis reveals that the LLM can reproduce insights from the classical SCM literature, while also uncovering novel behaviors and offering fresh perspectives on phenomena such as the bullwhip effect. This paper opens the door for exploring cooperation and competition for complex supply chain network through the lens of LLMs. 

---
# RoleRAG: Enhancing LLM Role-Playing via Graph Guided Retrieval 

**Authors**: Yongjie Wang, Jonathan Leung, Zhiqi Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.18541)  

**Abstract**: Large Language Models (LLMs) have shown promise in character imitation, enabling immersive and engaging conversations. However, they often generate content that is irrelevant or inconsistent with a character's background. We attribute these failures to: (1) the inability to accurately recall character-specific knowledge due to entity ambiguity, and (2) a lack of awareness of the character's cognitive boundaries. To address these issues, we propose RoleRAG, a retrieval-based framework that integrates efficient entity disambiguation for knowledge indexing with a boundary-aware retriever for extracting contextually appropriate information from a structured knowledge graph. Experiments on role-playing benchmarks show that RoleRAG's calibrated retrieval helps both general-purpose and role-specific LLMs better align with character knowledge and reduce hallucinated responses. 

---
# Retrieval-Augmented Generation for Service Discovery: Chunking Strategies and Benchmarking 

**Authors**: Robin D. Pesl, Jerin G. Mathew, Massimo Mecella, Marco Aiello  

**Link**: [PDF](https://arxiv.org/pdf/2505.19310)  

**Abstract**: Integrating multiple (sub-)systems is essential to create advanced Information Systems. Difficulties mainly arise when integrating dynamic environments, e.g., the integration at design time of not yet existing services. This has been traditionally addressed using a registry that provides the API documentation of the endpoints. Large Language Models have shown to be capable of automatically creating system integrations (e.g., as service composition) based on this documentation but require concise input due to input oken limitations, especially regarding comprehensive API descriptions. Currently, it is unknown how best to preprocess these API descriptions. In the present work, we (i) analyze the usage of Retrieval Augmented Generation for endpoint discovery and the chunking, i.e., preprocessing, of state-of-practice OpenAPIs to reduce the input oken length while preserving the most relevant information. To further reduce the input token length for the composition prompt and improve endpoint retrieval, we propose (ii) a Discovery Agent that only receives a summary of the most relevant endpoints nd retrieves specification details on demand. We evaluate RAG for endpoint discovery using (iii) a proposed novel service discovery benchmark SOCBench-D representing a general setting across numerous domains and the real-world RestBench enchmark, first, for the different chunking possibilities and parameters measuring the endpoint retrieval accuracy. Then, we assess the Discovery Agent using the same test data set. The prototype shows how to successfully employ RAG for endpoint discovery to reduce the token count. Our experiments show that endpoint-based approaches outperform naive chunking methods for preprocessing. Relying on an agent significantly improves precision while being prone to decrease recall, disclosing the need for further reasoning capabilities. 

---
