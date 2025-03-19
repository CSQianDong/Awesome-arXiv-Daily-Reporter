# KG-IRAG: A Knowledge Graph-Based Iterative Retrieval-Augmented Generation Framework for Temporal Reasoning 

**Authors**: Ruiyi Yang, Hao Xue, Imran Razzak, Hakim Hacid, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2503.14234)  

**Abstract**: Graph Retrieval-Augmented Generation (GraphRAG) has proven highly effective in enhancing the performance of Large Language Models (LLMs) on tasks that require external knowledge. By leveraging Knowledge Graphs (KGs), GraphRAG improves information retrieval for complex reasoning tasks, providing more precise and comprehensive retrieval and generating more accurate responses to QAs. However, most RAG methods fall short in addressing multi-step reasoning, particularly when both information extraction and inference are necessary. To address this limitation, this paper presents Knowledge Graph-Based Iterative Retrieval-Augmented Generation (KG-IRAG), a novel framework that integrates KGs with iterative reasoning to improve LLMs' ability to handle queries involving temporal and logical dependencies. Through iterative retrieval steps, KG-IRAG incrementally gathers relevant data from external KGs, enabling step-by-step reasoning. The proposed approach is particularly suited for scenarios where reasoning is required alongside dynamic temporal data extraction, such as determining optimal travel times based on weather conditions or traffic patterns. Experimental results show that KG-IRAG improves accuracy in complex reasoning tasks by effectively integrating external knowledge with iterative, logic-based retrieval. Additionally, three new datasets: weatherQA-Irish, weatherQA-Sydney, and trafficQA-TFNSW, are formed to evaluate KG-IRAG's performance, demonstrating its potential beyond traditional RAG applications. 

---
# Empowering GraphRAG with Knowledge Filtering and Integration 

**Authors**: Kai Guo, Harry Shomer, Shenglai Zeng, Haoyu Han, Yu Wang, Jiliang Tang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13804)  

**Abstract**: In recent years, large language models (LLMs) have revolutionized the field of natural language processing. However, they often suffer from knowledge gaps and hallucinations. Graph retrieval-augmented generation (GraphRAG) enhances LLM reasoning by integrating structured knowledge from external graphs. However, we identify two key challenges that plague GraphRAG:(1) Retrieving noisy and irrelevant information can degrade performance and (2)Excessive reliance on external knowledge suppresses the model's intrinsic reasoning. To address these issues, we propose GraphRAG-FI (Filtering and Integration), consisting of GraphRAG-Filtering and GraphRAG-Integration. GraphRAG-Filtering employs a two-stage filtering mechanism to refine retrieved information. GraphRAG-Integration employs a logits-based selection strategy to balance external knowledge from GraphRAG with the LLM's intrinsic reasoning,reducing over-reliance on retrievals. Experiments on knowledge graph QA tasks demonstrate that GraphRAG-FI significantly improves reasoning performance across multiple backbone models, establishing a more reliable and effective GraphRAG framework. 

---
# JuDGE: Benchmarking Judgment Document Generation for Chinese Legal System 

**Authors**: Weihang Su, Baoqing Yue, Qingyao Ai, Yiran Hu, Jiaqi Li, Changyue Wang, Kaiyuan Zhang, Yueyue Wu, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.14258)  

**Abstract**: This paper introduces JuDGE (Judgment Document Generation Evaluation), a novel benchmark for evaluating the performance of judgment document generation in the Chinese legal system. We define the task as generating a complete legal judgment document from the given factual description of the case. To facilitate this benchmark, we construct a comprehensive dataset consisting of factual descriptions from real legal cases, paired with their corresponding full judgment documents, which serve as the ground truth for evaluating the quality of generated documents. This dataset is further augmented by two external legal corpora that provide additional legal knowledge for the task: one comprising statutes and regulations, and the other consisting of a large collection of past judgment documents. In collaboration with legal professionals, we establish a comprehensive automated evaluation framework to assess the quality of generated judgment documents across various dimensions. We evaluate various baseline approaches, including few-shot in-context learning, fine-tuning, and a multi-source retrieval-augmented generation (RAG) approach, using both general and legal-domain LLMs. The experimental results demonstrate that, while RAG approaches can effectively improve performance in this task, there is still substantial room for further improvement. All the codes and datasets are available at: this https URL. 

---
# MoK-RAG: Mixture of Knowledge Paths Enhanced Retrieval-Augmented Generation for Embodied AI Environments 

**Authors**: Zhengsheng Guo, Linwei Zheng, Xinyang Chen, Xuefeng Bai, Kehai Chen, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.13882)  

**Abstract**: While human cognition inherently retrieves information from diverse and specialized knowledge sources during decision-making processes, current Retrieval-Augmented Generation (RAG) systems typically operate through single-source knowledge retrieval, leading to a cognitive-algorithmic discrepancy. To bridge this gap, we introduce MoK-RAG, a novel multi-source RAG framework that implements a mixture of knowledge paths enhanced retrieval mechanism through functional partitioning of a large language model (LLM) corpus into distinct sections, enabling retrieval from multiple specialized knowledge paths. Applied to the generation of 3D simulated environments, our proposed MoK-RAG3D enhances this paradigm by partitioning 3D assets into distinct sections and organizing them based on a hierarchical knowledge tree structure. Different from previous methods that only use manual evaluation, we pioneered the introduction of automated evaluation methods for 3D scenes. Both automatic and human evaluations in our experiments demonstrate that MoK-RAG3D can assist Embodied AI agents in generating diverse scenes. 

---
# MES-RAG: Bringing Multi-modal, Entity-Storage, and Secure Enhancements to RAG 

**Authors**: Pingyu Wu, Daiheng Gao, Jing Tang, Huimin Chen, Wenbo Zhou, Weiming Zhang, Nenghai Yu  

**Link**: [PDF](https://arxiv.org/pdf/2503.13563)  

**Abstract**: Retrieval-Augmented Generation (RAG) improves Large Language Models (LLMs) by using external knowledge, but it struggles with precise entity information retrieval. In this paper, we proposed MES-RAG framework, which enhances entity-specific query handling and provides accurate, secure, and consistent responses. MES-RAG introduces proactive security measures that ensure system integrity by applying protections prior to data access. Additionally, the system supports real-time multi-modal outputs, including text, images, audio, and video, seamlessly integrating into existing RAG architectures. Experimental results demonstrate that MES-RAG significantly improves both accuracy and recall, highlighting its effectiveness in advancing the security and utility of question-answering, increasing accuracy to 0.83 (+0.25) on targeted task. Our code and data are available at this https URL. 

---
# RAG-KG-IL: A Multi-Agent Hybrid Framework for Reducing Hallucinations and Enhancing LLM Reasoning through RAG and Incremental Knowledge Graph Learning Integration 

**Authors**: Hong Qing Yu, Frank McQuade  

**Link**: [PDF](https://arxiv.org/pdf/2503.13514)  

**Abstract**: This paper presents RAG-KG-IL, a novel multi-agent hybrid framework designed to enhance the reasoning capabilities of Large Language Models (LLMs) by integrating Retrieval-Augmented Generation (RAG) and Knowledge Graphs (KGs) with an Incremental Learning (IL) approach. Despite recent advancements, LLMs still face significant challenges in reasoning with structured data, handling dynamic knowledge evolution, and mitigating hallucinations, particularly in mission-critical domains. Our proposed RAG-KG-IL framework addresses these limitations by employing a multi-agent architecture that enables continuous knowledge updates, integrates structured knowledge, and incorporates autonomous agents for enhanced explainability and reasoning. The framework utilizes RAG to ensure the generated responses are grounded in verifiable information, while KGs provide structured domain knowledge for improved consistency and depth of understanding. The Incremental Learning approach allows for dynamic updates to the knowledge base without full retraining, significantly reducing computational overhead and improving the model's adaptability. We evaluate the framework using real-world case studies involving health-related queries, comparing it to state-of-the-art models like GPT-4o and a RAG-only baseline. Experimental results demonstrate that our approach significantly reduces hallucination rates and improves answer completeness and reasoning accuracy. The results underscore the potential of combining RAG, KGs, and multi-agent systems to create intelligent, adaptable systems capable of real-time knowledge integration and reasoning in complex domains. 

---
# Good/Evil Reputation Judgment of Celebrities by LLMs via Retrieval Augmented Generation 

**Authors**: Rikuto Tsuchida, Hibiki Yokoyama, Takehito Utsuro  

**Link**: [PDF](https://arxiv.org/pdf/2503.14382)  

**Abstract**: The purpose of this paper is to examine whether large language models (LLMs) can understand what is good and evil with respect to judging good/evil reputation of celebrities. Specifically, we first apply a large language model (namely, ChatGPT) to the task of collecting sentences that mention the target celebrity from articles about celebrities on Web pages. Next, the collected sentences are categorized based on their contents by ChatGPT, where ChatGPT assigns a category name to each of those categories. Those assigned category names are referred to as "aspects" of each celebrity. Then, by applying the framework of retrieval augmented generation (RAG), we show that the large language model is quite effective in the task of judging good/evil reputation of aspects and descriptions of each celebrity. Finally, also in terms of proving the advantages of the proposed method over existing services incorporating RAG functions, we show that the proposed method of judging good/evil of aspects/descriptions of each celebrity significantly outperform an existing service incorporating RAG functions. 

---
# Agent-Enhanced Large Language Models for Researching Political Institutions 

**Authors**: Joseph R. Loffredo, Suyeol Yun  

**Link**: [PDF](https://arxiv.org/pdf/2503.13524)  

**Abstract**: The applications of Large Language Models (LLMs) in political science are rapidly expanding. This paper demonstrates how LLMs, when augmented with predefined functions and specialized tools, can serve as dynamic agents capable of streamlining tasks such as data collection, preprocessing, and analysis. Central to this approach is agentic retrieval-augmented generation (Agentic RAG), which equips LLMs with action-calling capabilities for interaction with external knowledge bases. Beyond information retrieval, LLM agents may incorporate modular tools for tasks like document summarization, transcript coding, qualitative variable classification, and statistical modeling. To demonstrate the potential of this approach, we introduce CongressRA, an LLM agent designed to support scholars studying the U.S. Congress. Through this example, we highlight how LLM agents can reduce the costs of replicating, testing, and extending empirical research using the domain-specific data that drives the study of political institutions. 

---
