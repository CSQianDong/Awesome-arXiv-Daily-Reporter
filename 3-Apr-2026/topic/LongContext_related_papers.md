# ByteRover: Agent-Native Memory Through LLM-Curated Hierarchical Context 

**Authors**: Andy Nguyen, Danh Doan, Hoang Pham, Bao Ha, Dat Pham, Linh Nguyen, Hieu Nguyen, Thien Nguyen, Cuong Do, Phat Nguyen, Toan Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2604.01599)  

**Abstract**: Memory-Augmented Generation (MAG) extends large language models with external memory to support long-context reasoning, but existing approaches universally treat memory as an external service that agents call into, delegating storage to separate pipelines of chunking, embedding, and graph extraction. This architectural separation means the system that stores knowledge does not understand it, leading to semantic drift between what the agent intended to remember and what the pipeline actually captured, loss of coordination context across agents, and fragile recovery after failures. In this paper, we propose ByteRover, an agent-native memory architecture that inverts the memory pipeline: the same LLM that reasons about a task also curates, structures, and retrieves knowledge. ByteRover represents knowledge in a hierarchical Context Tree, a file-based knowledge graph organized as Domain, Topic, Subtopic, and Entry, where each entry carries explicit relations, provenance, and an Adaptive Knowledge Lifecycle (AKL) with importance scoring, maturity tiers, and recency decay. Retrieval uses a 5-tier progressive strategy that resolves most queries at sub-100 ms latency without LLM calls, escalating to agentic reasoning only for novel questions. Experiments on LoCoMo and LongMemEval demonstrate that ByteRover achieves state-of-the-art accuracy on LoCoMo and competitive results on LongMemEval while requiring zero external infrastructure, no vector database, no graph database, no embedding service, with all knowledge stored as human-readable markdown files on the local filesystem. 

---
# Retrieval-Augmented Question Answering over Scientific Literature for the Electron-Ion Collider 

**Authors**: Tina. J. Jat, T. Ghosh, Karthik Suresh  

**Link**: [PDF](https://arxiv.org/pdf/2604.02259)  

**Abstract**: To harness the power of Language Models in answering domain specific specialized technical questions, Retrieval Augmented Generation (RAG) is been used widely. In this work, we have developed a Q\&A application inspired by the Retrieval Augmented Generation (RAG), which is comprised of an in-house database indexed on the arXiv articles related to the Electron-Ion Collider (EIC) experiment - one of the largest international scientific collaboration and incorporated an open-source LLaMA model for answer generation. This is an extension to it's proceeding application built on proprietary model and Cloud-hosted external knowledge-base for the EIC experiment. This locally-deployed RAG-system offers a cost-effective, resource-constraint alternative solution to build a RAG-assisted Q\&A application on answering domain-specific queries in the field of experimental nuclear physics. This set-up facilitates data-privacy, avoids sending any pre-publication scientific data and information to public domain. Future improvement will expand the knowledge base to encompass heterogeneous EIC-related publications and reports and upgrade the application pipeline orchestration to the LangGraph framework. 

---
# Lifting Unlabeled Internet-level Data for 3D Scene Understanding 

**Authors**: Yixin Chen, Yaowei Zhang, Huangyue Yu, Junchao He, Yan Wang, Jiangyong Huang, Hongyu Shen, Junfeng Ni, Shaofei Wang, Baoxiong Jia, Song-Chun Zhu, Siyuan Huang  

**Link**: [PDF](https://arxiv.org/pdf/2604.01907)  

**Abstract**: Annotated 3D scene data is scarce and expensive to acquire, while abundant unlabeled videos are readily available on the internet. In this paper, we demonstrate that carefully designed data engines can leverage web-curated, unlabeled videos to automatically generate training data, to facilitate end-to-end models in 3D scene understanding alongside human-annotated datasets. We identify and analyze bottlenecks in automated data generation, revealing critical factors that determine the efficiency and effectiveness of learning from unlabeled data. To validate our approach across different perception granularities, we evaluate on three tasks spanning low-level perception, i.e., 3D object detection and instance segmentation, to high-evel reasoning, i.e., 3D spatial Visual Question Answering (VQA) and Vision-Lanugage Navigation (VLN). Models trained on our generated data demonstrate strong zero-shot performance and show further improvement after finetuning. This demonstrates the viability of leveraging readily available web data as a path toward more capable scene understanding systems. 

---
# Towards Position-Robust Talent Recommendation via Large Language Models 

**Authors**: Silin Du, Hongyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2604.02200)  

**Abstract**: Talent recruitment is a critical, yet costly process for many industries, with high recruitment costs and long hiring cycles. Existing talent recommendation systems increasingly adopt large language models (LLMs) due to their remarkable language understanding capabilities. However, most prior approaches follow a pointwise paradigm, which requires LLMs to repeatedly process some text and fails to capture the relationships among candidates in the list, resulting in higher token consumption and suboptimal recommendations. Besides, LLMs exhibit position bias and the lost-in-the-middle issue when answering multiple-choice questions and processing multiple long documents. To address these issues, we introduce an implicit strategy to utilize LLM's potential output for the recommendation task and propose L3TR, a novel framework for listwise talent recommendation with LLMs. In this framework, we propose a block attention mechanism and a local positional encoding method to enhance inter-document processing and mitigate the position bias and concurrent token bias issue. We also introduce an ID sampling method for resolving the inconsistency between candidate set sizes in the training phase and the inference phase. We design evaluation methods to detect position bias and token bias and training-free debiasing methods. Extensive experiments on two real-world datasets validated the effectiveness of L3TR, showing consistent improvements over existing baselines. 

---
