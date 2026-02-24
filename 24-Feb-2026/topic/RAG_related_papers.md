# How Retrieved Context Shapes Internal Representations in RAG 

**Authors**: Samuel Yeh, Sharon Li  

**Link**: [PDF](https://arxiv.org/pdf/2602.20091)  

**Abstract**: Retrieval-augmented generation (RAG) enhances large language models (LLMs) by conditioning generation on retrieved external documents, but the effect of retrieved context is often non-trivial. In realistic retrieval settings, the retrieved document set often contains a mixture of documents that vary in relevance and usefulness. While prior work has largely examined these phenomena through output behavior, little is known about how retrieved context shapes the internal representations that mediate information integration in RAG. In this work, we study RAG through the lens of latent representations. We systematically analyze how different types of retrieved documents affect the hidden states of LLMs, and how these internal representation shifts relate to downstream generation behavior. Across four question-answering datasets and three LLMs, we analyze internal representations under controlled single- and multi-document settings. Our results reveal how context relevancy and layer-wise processing influence internal representations, providing explanations on LLMs output behaviors and insights for RAG system design. 

---
# Unlocking Multimodal Document Intelligence: From Current Triumphs to Future Frontiers of Visual Document Retrieval 

**Authors**: Yibo Yan, Jiahao Huo, Guanbo Feng, Mingdong Ou, Yi Cao, Xin Zou, Shuliang Liu, Yuanhuiyi Lyu, Yu Huang, Jungang Li, Kening Zheng, Xu Zheng, Philip S. Yu, James Kwok, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2602.19961)  

**Abstract**: With the rapid proliferation of multimodal information, Visual Document Retrieval (VDR) has emerged as a critical frontier in bridging the gap between unstructured visually rich data and precise information acquisition. Unlike traditional natural image retrieval, visual documents exhibit unique characteristics defined by dense textual content, intricate layouts, and fine-grained semantic dependencies. This paper presents the first comprehensive survey of the VDR landscape, specifically through the lens of the Multimodal Large Language Model (MLLM) era. We begin by examining the benchmark landscape, and subsequently dive into the methodological evolution, categorizing approaches into three primary aspects: multimodal embedding models, multimodal reranker models, and the integration of Retrieval-Augmented Generation (RAG) and Agentic systems for complex document intelligence. Finally, we identify persistent challenges and outline promising future directions, aiming to provide a clear roadmap for future multimodal document intelligence. 

---
# Learning to Reason for Multi-Step Retrieval of Personal Context in Personalized Question Answering 

**Authors**: Maryam Amirizaniani, Alireza Salemi, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2602.19317)  

**Abstract**: Personalization in Question Answering (QA) requires answers that are both accurate and aligned with users' background, preferences, and historical context. Existing state-of-the-art methods primarily rely on retrieval-augmented generation (RAG) solutions that construct personal context by retrieving relevant items from the user's profile. Existing methods use the user's query directly to retrieve personal documents, and such strategies often lead to surface-level personalization. We propose PR2 (Personalized Retrieval-Augmented Reasoning), a reinforcement learning framework that integrates reasoning and retrieval from personal context for personalization. PR2 learns adaptive retrieval-reasoning policies, determining when to retrieve, what evidence to retrieve from user profiles, and how to incorporate it into intermediate reasoning steps. By optimizing multi-turn reasoning trajectories under a personalized reward function, the framework reinforces reasoning paths that better align with user-specific preferences and contextual signals reflected by the reward model. Extensive experiments on the LaMP-QA benchmark using three LLMs show that PR2 consistently outperforms strong baselines, achieving an average relative improvement of 8.8%-12% in personalized QA. 

---
# Facet-Level Persona Control by Trait-Activated Routing with Contrastive SAE for Role-Playing LLMs 

**Authors**: Wenqiu Tang, Zhen Wan, Takahiro Komamizu, Ichiro Ide  

**Link**: [PDF](https://arxiv.org/pdf/2602.19157)  

**Abstract**: Personality control in Role-Playing Agents (RPAs) is commonly achieved via training-free methods that inject persona descriptions and memory through prompts or retrieval-augmented generation, or via supervised fine-tuning (SFT) on persona-specific corpora. While SFT can be effective, it requires persona-labeled data and retraining for new roles, limiting flexibility. In contrast, prompt- and RAG-based signals are easy to apply but can be diluted in long dialogues, leading to drifting and sometimes inconsistent persona behavior. To address this, we propose a contrastive Sparse AutoEncoder (SAE) framework that learns facet-level personality control vectors aligned with the Big Five 30-facet model. A new 15,000-sample leakage-controlled corpus is constructed to provide balanced supervision for each facet. The learned vectors are integrated into the model's residual space and dynamically selected by a trait-activated routing module, enabling precise and interpretable personality steering. Experiments on Large Language Models (LLMs) show that the proposed method maintains stable character fidelity and output quality across contextualized settings, outperforming Contrastive Activation Addition (CAA) and prompt-only baselines. The combined SAE+Prompt configuration achieves the best overall performance, confirming that contrastively trained latent vectors can enhance persona control while preserving dialogue coherence. 

---
# Rethinking Retrieval-Augmented Generation as a Cooperative Decision-Making Problem 

**Authors**: Lichang Song, Ting Long, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2602.18734)  

**Abstract**: Retrieval-Augmented Generation (RAG) has demonstrated strong effectiveness in knowledge-intensive tasks by grounding language generation in external evidence. Despite its success, many existing RAG systems are built based on a ranking-centric, asymmetric dependency paradigm, where the generation quality of the generator is highly dependent on reranking results of the reranker. To overcome this limitation, we reformulate RAG as a cooperative multi-agent decision-making problem and propose Cooperative Retrieval-Augmented Generation (CoRAG), a framework in which the reranker and the generator act as peer decision-makers rather than being connected through an asymmetric dependency pipeline. By jointly optimizing their behaviors toward a shared task objective, the reranker and generator are encouraged to cooperate, ensuring that document reranking and generation work in concert to improve the final response. Experimental results demonstrate good generalization and improved generation stability of CoRAG, even when the model is trained on only around 10K PopQA samples. Our model released in this https URL 

---
# Topology of Reasoning: Retrieved Cell Complex-Augmented Generation for Textual Graph Question Answering 

**Authors**: Sen Zhao, Lincheng Zhou, Yue Chen, Ding Zou  

**Link**: [PDF](https://arxiv.org/pdf/2602.19240)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances the reasoning ability of Large Language Models (LLMs) by dynamically integrating external knowledge, thereby mitigating hallucinations and strengthening contextual grounding for structured data such as graphs. Nevertheless, most existing RAG variants for textual graphs concentrate on low-dimensional structures -- treating nodes as entities (0-dimensional) and edges or paths as pairwise or sequential relations (1-dimensional), but overlook cycles, which are crucial for reasoning over relational loops. Such cycles often arise in questions requiring closed-loop inference about similar objects or relative positions. This limitation often results in incomplete contextual grounding and restricted reasoning capability. In this work, we propose Topology-enhanced Retrieval-Augmented Generation (TopoRAG), a novel framework for textual graph question answering that effectively captures higher-dimensional topological and relational dependencies. Specifically, TopoRAG first lifts textual graphs into cellular complexes to model multi-dimensional topological structures. Leveraging these lifted representations, a topology-aware subcomplex retrieval mechanism is proposed to extract cellular complexes relevant to the input query, providing compact and informative topological context. Finally, a multi-dimensional topological reasoning mechanism operates over these complexes to propagate relational information and guide LLMs in performing structured, logic-aware inference. Empirical evaluations demonstrate that our method consistently surpasses existing baselines across diverse textual graph tasks. 

---
# Large Language Model-Assisted UAV Operations and Communications: A Multifaceted Survey and Tutorial 

**Authors**: Yousef Emami, Hao Zhou, Radha Reddy, Atefeh Hajijamali Arani, Biliang Wang, Kai Li, Luis Almeida, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2602.19534)  

**Abstract**: Uncrewed Aerial Vehicles (UAVs) are widely deployed across diverse applications due to their mobility and agility. Recent advances in Large Language Models (LLMs) offer a transformative opportunity to enhance UAV intelligence beyond conventional optimization-based and learning-based approaches. By integrating LLMs into UAV systems, advanced environmental understanding, swarm coordination, mobility optimization, and high-level task reasoning can be achieved, thereby allowing more adaptive and context-aware aerial operations. This survey systematically explores the intersection of LLMs and UAV technologies and proposes a unified framework that consolidates existing architectures, methodologies, and applications for UAVs. We first present a structured taxonomy of LLM adaptation techniques for UAVs, including pretraining, fine-tuning, Retrieval-Augmented Generation (RAG), and prompt engineering, along with key reasoning capabilities such as Chain-of-Thought (CoT) and In-Context Learning (ICL). We then examine LLM-assisted UAV communications and operations, covering navigation, mission planning, swarm control, safety, autonomy, and network management. After that, the survey further discusses Multimodal LLMs (MLLMs) for human-swarm interaction, perception-driven navigation, and collaborative control. Finally, we address ethical considerations, including bias, transparency, accountability, and Human-in-the-Loop (HITL) strategies, and outline future research directions. Overall, this work positions LLM-assisted UAVs as a foundation for intelligent and adaptive aerial systems. 

---
