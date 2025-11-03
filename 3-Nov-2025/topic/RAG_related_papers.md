# Adapting Large Language Models to Emerging Cybersecurity using Retrieval Augmented Generation 

**Authors**: Arnabh Borah, Md Tanvirul Alam, Nidhi Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2510.27080)  

**Abstract**: Security applications are increasingly relying on large language models (LLMs) for cyber threat detection; however, their opaque reasoning often limits trust, particularly in decisions that require domain-specific cybersecurity knowledge. Because security threats evolve rapidly, LLMs must not only recall historical incidents but also adapt to emerging vulnerabilities and attack patterns. Retrieval-Augmented Generation (RAG) has demonstrated effectiveness in general LLM applications, but its potential for cybersecurity remains underexplored. In this work, we introduce a RAG-based framework designed to contextualize cybersecurity data and enhance LLM accuracy in knowledge retention and temporal reasoning. Using external datasets and the Llama-3-8B-Instruct model, we evaluate baseline RAG, an optimized hybrid retrieval approach, and conduct a comparative analysis across multiple performance metrics. Our findings highlight the promise of hybrid retrieval in strengthening the adaptability and reliability of LLMs for cybersecurity tasks. 

---
# LLM-based Multi-class Attack Analysis and Mitigation Framework in IoT/IIoT Networks 

**Authors**: Seif Ikbarieh, Maanak Gupta, Elmahedi Mahalal  

**Link**: [PDF](https://arxiv.org/pdf/2510.26941)  

**Abstract**: The Internet of Things has expanded rapidly, transforming communication and operations across industries but also increasing the attack surface and security breaches. Artificial Intelligence plays a key role in securing IoT, enabling attack detection, attack behavior analysis, and mitigation suggestion. Despite advancements, evaluations remain purely qualitative, and the lack of a standardized, objective benchmark for quantitatively measuring AI-based attack analysis and mitigation hinders consistent assessment of model effectiveness. In this work, we propose a hybrid framework combining Machine Learning (ML) for multi-class attack detection with Large Language Models (LLMs) for attack behavior analysis and mitigation suggestion. After benchmarking several ML and Deep Learning (DL) classifiers on the Edge-IIoTset and CICIoT2023 datasets, we applied structured role-play prompt engineering with Retrieval-Augmented Generation (RAG) to guide ChatGPT-o3 and DeepSeek-R1 in producing detailed, context-aware responses. We introduce novel evaluation metrics for quantitative assessment to guide us and an ensemble of judge LLMs, namely ChatGPT-4o, DeepSeek-V3, Mixtral 8x7B Instruct, Gemini 2.5 Flash, Meta Llama 4, TII Falcon H1 34B Instruct, xAI Grok 3, and Claude 4 Sonnet, to independently evaluate the responses. Results show that Random Forest has the best detection model, and ChatGPT-o3 outperformed DeepSeek-R1 in attack analysis and mitigation. 

---
# MARAG-R1: Beyond Single Retriever via Reinforcement-Learned Multi-Tool Agentic Retrieval 

**Authors**: Qi Luo, Xiaonan Li, Yuxin Wang, Tingshuo Fan, Yuan Li, Xinchi Chen, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2510.27569)  

**Abstract**: Large Language Models (LLMs) excel at reasoning and generation but are inherently limited by static pretraining data, resulting in factual inaccuracies and weak adaptability to new information. Retrieval-Augmented Generation (RAG) addresses this issue by grounding LLMs in external knowledge; However, the effectiveness of RAG critically depends on whether the model can adequately access relevant information. Existing RAG systems rely on a single retriever with fixed top-k selection, restricting access to a narrow and static subset of the corpus. As a result, this single-retriever paradigm has become the primary bottleneck for comprehensive external information acquisition, especially in tasks requiring corpus-level reasoning. To overcome this limitation, we propose MARAG-R1, a reinforcement-learned multi-tool RAG framework that enables LLMs to dynamically coordinate multiple retrieval mechanisms for broader and more precise information access. MARAG-R1 equips the model with four retrieval tools -- semantic search, keyword search, filtering, and aggregation -- and learns both how and when to use them through a two-stage training process: supervised fine-tuning followed by reinforcement learning. This design allows the model to interleave reasoning and retrieval, progressively gathering sufficient evidence for corpus-level synthesis. Experiments on GlobalQA, HotpotQA, and 2WikiMultiHopQA demonstrate that MARAG-R1 substantially outperforms strong baselines and achieves new state-of-the-art results in corpus-level reasoning tasks. 

---
# LLM-Centric RAG with Multi-Granular Indexing and Confidence Constraints 

**Authors**: Xiaofan Guo, Yaxuan Luan, Yue Kang, Xiangchen Song, Jinxu Guo  

**Link**: [PDF](https://arxiv.org/pdf/2510.27054)  

**Abstract**: This paper addresses the issues of insufficient coverage, unstable results, and limited reliability in retrieval-augmented generation under complex knowledge environments, and proposes a confidence control method that integrates multi-granularity memory indexing with uncertainty estimation. The method builds a hierarchical memory structure that divides knowledge representations into different levels of granularity, enabling dynamic indexing and retrieval from local details to global context, and thus establishing closer semantic connections between retrieval and generation. On this basis, an uncertainty estimation mechanism is introduced to explicitly constrain and filter low-confidence paths during the generation process, allowing the model to maintain information coverage while effectively suppressing noise and false content. The overall optimization objective consists of generation loss, entropy constraints, and variance regularization, forming a unified confidence control framework. In the experiments, comprehensive sensitivity tests and comparative analyses were designed, covering hyperparameters, environmental conditions, and data structures, to verify the stability and robustness of the proposed method across different scenarios. The results show that the method achieves superior performance over existing models in QA accuracy, retrieval recall, ranking quality, and factual consistency, demonstrating the effectiveness of combining multi-granularity indexing with confidence control. This study not only provides a new technical pathway for retrieval-augmented generation but also offers practical evidence for improving the reliability and controllability of large models in complex contexts. 

---
# Interact-RAG: Reason and Interact with the Corpus, Beyond Black-Box Retrieval 

**Authors**: Yulong Hui, Chao Chen, Zhihang Fu, Yihao Liu, Jieping Ye, Huanchen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.27566)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly enhanced LLMs by incorporating external information. However, prevailing agentic RAG approaches are constrained by a critical limitation: they treat the retrieval process as a black-box querying operation. This confines agents' actions to query issuing, hindering its ability to tackle complex information-seeking tasks. To address this, we introduce Interact-RAG, a new paradigm that elevates the LLM agent from a passive query issuer into an active manipulator of the retrieval process. We dismantle the black-box with a Corpus Interaction Engine, equipping the agent with a set of action primitives for fine-grained control over information retrieval. To further empower the agent on the entire RAG pipeline, we first develop a reasoning-enhanced workflow, which enables both zero-shot execution and the synthesis of interaction trajectories. We then leverage this synthetic data to train a fully autonomous end-to-end agent via Supervised Fine-Tuning (SFT), followed by refinement with Reinforcement Learning (RL). Extensive experiments across six benchmarks demonstrate that Interact-RAG significantly outperforms other advanced methods, validating the efficacy of our reasoning-interaction strategy. 

---
