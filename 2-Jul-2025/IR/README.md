# WebArXiv: Evaluating Multimodal Agents on Time-Invariant arXiv Tasks 

**Authors**: Zihao Sun, Meng Fang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.00938)  

**Abstract**: Recent progress in large language models (LLMs) has enabled the development of autonomous web agents capable of navigating and interacting with real websites. However, evaluating such agents remains challenging due to the instability and inconsistency of existing benchmarks, which often rely on dynamic content or oversimplified simulations. In this work, we introduce WebArXiv, a static and time-invariant benchmark comprising 275 web-based tasks grounded in the arXiv platform. WebArXiv ensures reproducible and reliable evaluation by anchoring tasks in fixed web snapshots with deterministic ground truths and standardized action trajectories. Through behavioral analysis, we identify a common failure mode, Rigid History Reflection, where agents over-rely on fixed interaction histories. To address this, we propose a lightweight dynamic reflection mechanism that allows agents to selectively retrieve relevant past steps during decision-making. We evaluate ten state-of-the-art web agents on WebArXiv. Results demonstrate clear performance differences across agents and validate the effectiveness of our proposed reflection strategy. 

---
# EARN: Efficient Inference Acceleration for LLM-based Generative Recommendation by Register Tokens 

**Authors**: Chaoqun Yang, Xinyu Lin, Wenjie Wang, Yongqi Li, Teng Sun, Xianjing Han, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2507.00715)  

**Abstract**: Large Language Model-based generative recommendation (LLMRec) has achieved notable success, but it suffers from high inference latency due to massive computational overhead and memory pressure of KV Cache. Existing KV Cache reduction methods face critical limitations: cache compression offers marginal acceleration given recommendation tasks' short decoding steps, while prompt compression risks discarding vital interaction history. Through systematic analysis of attention patterns in LLMRec, we uncover two pivotal insights: 1) layer-wise attention sparsity inversion where early layers retain dense informative patterns while later layers exhibit high redundancy, and 2) dual attention sinks phenomenon where attention scores concentrate on both head and tail tokens of input sequences. Motivated by these insights, we propose EARN, an efficient inference framework that leverages the early layers to compress information into register tokens placed at the input sequence boundaries, then focuses solely on these tokens in the subsequent layers. Extensive experiments on three datasets, two LLMRec methods and two LLM architectures demonstrate EARN's superiority, achieving up to 3.79x speedup and 80.8% KV Cache reduction with better accuracy than the general finetuning approach. Our work bridges the efficiency-effectiveness gap in LLMRec, offering practical deployment advantages for industrial scenarios. 

---
# Reliable Annotations with Less Effort: Evaluating LLM-Human Collaboration in Search Clarifications 

**Authors**: Leila Tavakoli, Hamed Zamani  

**Link**: [PDF](https://arxiv.org/pdf/2507.00543)  

**Abstract**: Despite growing interest in using large language models (LLMs) to automate annotation, their effectiveness in complex, nuanced, and multi-dimensional labelling tasks remains relatively underexplored. This study focuses on annotation for the search clarification task, leveraging a high-quality, multi-dimensional dataset that includes five distinct fine-grained annotation subtasks. Although LLMs have shown impressive capabilities in general settings, our study reveals that even state-of-the-art models struggle to replicate human-level performance in subjective or fine-grained evaluation tasks. Through a systematic assessment, we demonstrate that LLM predictions are often inconsistent, poorly calibrated, and highly sensitive to prompt variations. To address these limitations, we propose a simple yet effective human-in-the-loop (HITL) workflow that uses confidence thresholds and inter-model disagreement to selectively involve human review. Our findings show that this lightweight intervention significantly improves annotation reliability while reducing human effort by up to 45%, offering a relatively scalable and cost-effective yet accurate path forward for deploying LLMs in real-world evaluation settings. 

---
# Rethinking Group Recommender Systems in the Era of Generative AI: From One-Shot Recommendations to Agentic Group Decision Support 

**Authors**: Dietmar Jannach, Amra Delić, Francesco Ricci, Markus Zanker  

**Link**: [PDF](https://arxiv.org/pdf/2507.00535)  

**Abstract**: More than twenty-five years ago, first ideas were developed on how to design a system that can provide recommendations to groups of users instead of individual users. Since then, a rich variety of algorithmic proposals were published, e.g., on how to acquire individual preferences, how to aggregate them, and how to generate recommendations for groups of users. However, despite the rich literature on the topic, barely any examples of real-world group recommender systems can be found. This lets us question common assumptions in academic research, in particular regarding communication processes in a group and how recommendation-supported decisions are made. In this essay, we argue that these common assumptions and corresponding system designs often may not match the needs or expectations of users. We thus call for a reorientation in this research area, leveraging the capabilities of modern Generative AI assistants like ChatGPT. Specifically, as one promising future direction, we envision group recommender systems to be systems where human group members interact in a chat and an AI-based group recommendation agent assists the decision-making process in an agentic way. Ultimately, this shall lead to a more natural group decision-making environment and finally to wider adoption of group recommendation systems in practice. 

---
# \texttt{WebANNS}: Fast and Efficient Approximate Nearest Neighbor Search in Web Browsers 

**Authors**: Mugeng Liu, Siqi Zhong, Qi Yang, Yudong Han, Xuanzhe Liu, Yun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2507.00521)  

**Abstract**: Approximate nearest neighbor search (ANNS) has become vital to modern AI infrastructure, particularly in retrieval-augmented generation (RAG) applications. Numerous in-browser ANNS engines have emerged to seamlessly integrate with popular LLM-based web applications, while addressing privacy protection and challenges of heterogeneous device deployments. However, web browsers present unique challenges for ANNS, including computational limitations, external storage access issues, and memory utilization constraints, which state-of-the-art (SOTA) solutions fail to address comprehensively.
We propose \texttt{WebANNS}, a novel ANNS engine specifically designed for web browsers. \texttt{WebANNS} leverages WebAssembly to overcome computational bottlenecks, designs a lazy loading strategy to optimize data retrieval from external storage, and applies a heuristic approach to reduce memory usage. Experiments show that \texttt{WebANNS} is fast and memory efficient, achieving up to $743.8\times$ improvement in 99th percentile query latency over the SOTA engine, while reducing memory usage by up to 39\%. Note that \texttt{WebANNS} decreases query time from 10 seconds to the 10-millisecond range in browsers, making in-browser ANNS practical with user-acceptable latency. 

---
# MassTool: A Multi-Task Search-Based Tool Retrieval Framework for Large Language Models 

**Authors**: Jianghao Lin, Xinyuan Wang, Xinyi Dai, Menghui Zhu, Bo Chen, Ruiming Tang, Yong Yu, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.00487)  

**Abstract**: Tool retrieval is a critical component in enabling large language models (LLMs) to interact effectively with external tools. It aims to precisely filter the massive tools into a small set of candidates for the downstream tool-augmented LLMs. However, most existing approaches primarily focus on optimizing tool representations, often neglecting the importance of precise query comprehension. To address this gap, we introduce MassTool, a multi-task search-based framework designed to enhance both query representation and tool retrieval accuracy. MassTool employs a two-tower architecture: a tool usage detection tower that predicts the need for function calls, and a tool retrieval tower that leverages a query-centric graph convolution network (QC-GCN) for effective query-tool matching. It also incorporates search-based user intent modeling (SUIM) to handle diverse and out-of-distribution queries, alongside an adaptive knowledge transfer (AdaKT) module for efficient multi-task learning. By jointly optimizing tool usage detection loss, list-wise retrieval loss, and contrastive regularization loss, MassTool establishes a robust dual-step sequential decision-making pipeline for precise query understanding. Extensive experiments demonstrate its effectiveness in improving retrieval accuracy. Our code is available at this https URL. 

---
# On Mitigating Data Sparsity in Conversational Recommender Systems 

**Authors**: Sixiao Zhang, Mingrui Liu, Cheng Long, Wei Yuan, Hongxu Chen, Xiangyu Zhao, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2507.00479)  

**Abstract**: Conversational recommender systems (CRSs) capture user preference through textual information in dialogues. However, they suffer from data sparsity on two fronts: the dialogue space is vast and linguistically diverse, while the item space exhibits long-tail and sparse distributions. Existing methods struggle with (1) generalizing to varied dialogue expressions due to underutilization of rich textual cues, and (2) learning informative item representations under severe sparsity. To address these problems, we propose a CRS model named DACRS. It consists of three modules, namely Dialogue Augmentation, Knowledge-Guided Entity Modeling, and Dialogue-Entity Matching. In the Dialogue Augmentation module, we apply a two-stage augmentation pipeline to augment the dialogue context to enrich the data and improve generalizability. In the Knowledge-Guided Entity Modeling, we propose a knowledge graph (KG) based entity substitution and an entity similarity constraint to enhance the expressiveness of entity embeddings. In the Dialogue-Entity Matching module, we fuse the dialogue embedding with the mentioned entity embeddings through a dialogue-guided attention aggregation to acquire user embeddings that contain both the explicit and implicit user preferences. Extensive experiments on two public datasets demonstrate the state-of-the-art performance of DACRS. 

---
# Read the Docs Before Rewriting: Equip Rewriter with Domain Knowledge via Continual Pre-training 

**Authors**: Qi Wang, Yixuan Cao, Yifan Liu, Jiangtao Zhao, Ping Luo  

**Link**: [PDF](https://arxiv.org/pdf/2507.00477)  

**Abstract**: A Retrieval-Augmented Generation (RAG)-based question-answering (QA) system enhances a large language model's knowledge by retrieving relevant documents based on user queries. Discrepancies between user queries and document phrasings often necessitate query rewriting. However, in specialized domains, the rewriter model may struggle due to limited domain-specific knowledge. To resolve this, we propose the R\&R (Read the doc before Rewriting) rewriter, which involves continual pre-training on professional documents, akin to how students prepare for open-book exams by reviewing textbooks. Additionally, it can be combined with supervised fine-tuning for improved results. Experiments on multiple datasets demonstrate that R\&R excels in professional QA across multiple domains, effectively bridging the query-document gap, while maintaining good performance in general scenarios, thus advancing the application of RAG-based QA systems in specialized fields. 

---
# Digital Collections Explorer: An Open-Source, Multimodal Viewer for Searching Digital Collections 

**Authors**: Ying-Hsiang Huang, Benjamin Charles Germain Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.00961)  

**Abstract**: We present Digital Collections Explorer, a web-based, open-source exploratory search platform that leverages CLIP (Contrastive Language-Image Pre-training) for enhanced visual discovery of digital collections. Our Digital Collections Explorer can be installed locally and configured to run on a visual collection of interest on disk in just a few steps. Building upon recent advances in multimodal search techniques, our interface enables natural language queries and reverse image searches over digital collections with visual features. This paper describes the system's architecture, implementation, and application to various cultural heritage collections, demonstrating its potential for democratizing access to digital archives, especially those with impoverished metadata. We present case studies with maps, photographs, and PDFs extracted from web archives in order to demonstrate the flexibility of the Digital Collections Explorer, as well as its ease of use. We demonstrate that the Digital Collections Explorer scales to hundreds of thousands of images on a MacBook Pro with an M4 chip. Lastly, we host a public demo of Digital Collections Explorer. 

---
# Exploring Large Action Sets with Hyperspherical Embeddings using von Mises-Fisher Sampling 

**Authors**: Walid Bendada, Guillaume Salha-Galvan, Romain Hennequin, Théo Bontempelli, Thomas Bouabça, Tristan Cazenave  

**Link**: [PDF](https://arxiv.org/pdf/2507.00518)  

**Abstract**: This paper introduces von Mises-Fisher exploration (vMF-exp), a scalable method for exploring large action sets in reinforcement learning problems where hyperspherical embedding vectors represent these actions. vMF-exp involves initially sampling a state embedding representation using a von Mises-Fisher distribution, then exploring this representation's nearest neighbors, which scales to virtually unlimited numbers of candidate actions. We show that, under theoretical assumptions, vMF-exp asymptotically maintains the same probability of exploring each action as Boltzmann Exploration (B-exp), a popular alternative that, nonetheless, suffers from scalability issues as it requires computing softmax values for each action. Consequently, vMF-exp serves as a scalable alternative to B-exp for exploring large action sets with hyperspherical embeddings. Experiments on simulated data, real-world public data, and the successful large-scale deployment of vMF-exp on the recommender system of a global music streaming service empirically validate the key properties of the proposed method. 

---
# Modeling Data Diversity for Joint Instance and Verbalizer Selection in Cold-Start Scenarios 

**Authors**: Mohna Chakraborty, Adithya Kulkarni, Qi Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.00330)  

**Abstract**: Prompt-based methods leverage the knowledge of pre-trained language models (PLMs) trained with a masked language modeling (MLM) objective; however, these methods are sensitive to template, verbalizer, and few-shot instance selection, particularly in cold-start settings with no labeled data. Existing studies overlook the dependency between instances and verbalizers, where instance-label probabilities depend on verbalizer token proximity in the embedding space. To address this, we propose COLDSELECT, a joint verbalizer and instance selection approach that models data diversity. COLDSELECT maps PLM vocabulary and $h_{[MASK]}$ embeddings into a shared space, applying dimensionality reduction and clustering to ensure efficient and diverse selection. By optimizing for minimal uncertainty and maximal diversity, COLDSELECT captures data relationships effectively. Experiments on eight benchmarks demonstrate COLDSELECT's superiority in reducing uncertainty and enhancing generalization, outperforming baselines in verbalizer and few-shot instance selection for cold-start scenarios. 

---
# TalentMine: LLM-Based Extraction and Question-Answering from Multimodal Talent Tables 

**Authors**: Varun Mannam, Fang Wang, Chaochun Liu, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.00041)  

**Abstract**: In talent management systems, critical information often resides in complex tabular formats, presenting significant retrieval challenges for conventional language models. These challenges are pronounced when processing Talent documentation that requires precise interpretation of tabular relationships for accurate information retrieval and downstream decision-making. Current table extraction methods struggle with semantic understanding, resulting in poor performance when integrated into retrieval-augmented chat applications. This paper identifies a key bottleneck - while structural table information can be extracted, the semantic relationships between tabular elements are lost, causing downstream query failures. To address this, we introduce TalentMine, a novel LLM-enhanced framework that transforms extracted tables into semantically enriched representations. Unlike conventional approaches relying on CSV or text linearization, our method employs specialized multimodal reasoning to preserve both structural and semantic dimensions of tabular data. Experimental evaluation across employee benefits document collections demonstrates TalentMine's superior performance, achieving 100% accuracy in query answering tasks compared to 0% for standard AWS Textract extraction and 40% for AWS Textract Visual Q&A capabilities. Our comparative analysis also reveals that the Claude v3 Haiku model achieves optimal performance for talent management applications. The key contributions of this work include (1) a systematic analysis of semantic information loss in current table extraction pipelines, (2) a novel LLM-based method for semantically enriched table representation, (3) an efficient integration framework for retrieval-augmented systems as end-to-end systems, and (4) comprehensive benchmarks on talent analytics tasks showing substantial improvements across multiple categories. 

---
