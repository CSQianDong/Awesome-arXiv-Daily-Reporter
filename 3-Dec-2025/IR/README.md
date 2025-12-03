# LORE: A Large Generative Model for Search Relevance 

**Authors**: Chenji Lu, Zhuo Chen, Hui Zhao, Zhiyuan Zeng, Gang Zhao, Junjie Ren, Ruicong Xu, Haoran Li, Songyan Liu, Pengjie Wang, Jian Xu, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2512.03025)  

**Abstract**: Achievement. We introduce LORE, a systematic framework for Large Generative Model-based relevance in e-commerce search. Deployed and iterated over three years, LORE achieves a cumulative +27\% improvement in online GoodRate metrics. This report shares the valuable experience gained throughout its development lifecycle, spanning data, features, training, evaluation, and deployment. Insight. While existing works apply Chain-of-Thought (CoT) to enhance relevance, they often hit a performance ceiling. We argue this stems from treating relevance as a monolithic task, lacking principled deconstruction. Our key insight is that relevance comprises distinct capabilities: knowledge and reasoning, multi-modal matching, and rule adherence. We contend that a qualitative-driven decomposition is essential for breaking through current performance bottlenecks. Contributions. LORE provides a complete blueprint for the LLM relevance lifecycle. Key contributions include: (1) A two-stage training paradigm combining progressive CoT synthesis via SFT with human preference alignment via RL. (2) A comprehensive benchmark, RAIR, designed to evaluate these core capabilities. (3) A query frequency-stratified deployment strategy that efficiently transfers offline LLM capabilities to the online system. LORE serves as both a practical solution and a methodological reference for other vertical domains. 

---
# AskNearby: An LLM-Based Application for Neighborhood Information Retrieval and Personalized Cognitive-Map Recommendations 

**Authors**: Luyao Niu, Zhicheng Deng, Boyang Li, Nuoxian Huang, Ruiqi Liu, Wenjia Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02502)  

**Abstract**: The "15-minute city" envisions neighborhoods where residents can meet daily needs via a short walk or bike ride. Realizing this vision requires not only physical proximity but also efficient and reliable access to information about nearby places, services, and events. Existing location-based systems, however, focus mainly on city-level tasks and neglect the spatial, temporal, and cognitive factors that shape localized decision-making. We conceptualize this gap as the Local Life Information Accessibility (LLIA) problem and introduce AskNearby, an AI-driven community application that unifies retrieval and recommendation within the 15-minute life circle. AskNearby integrates (i) a three-layer Retrieval-Augmented Generation (RAG) pipeline that synergizes graph-based, semantic-vector, and geographic retrieval with (ii) a cognitive-map model that encodes each user's neighborhood familiarity and preferences. Experiments on real-world community datasets demonstrate that AskNearby significantly outperforms LLM-based and map-based baselines in retrieval accuracy and recommendation quality, achieving robust performance in spatiotemporal grounding and cognitive-aware ranking. Real-world deployments further validate its effectiveness. By addressing the LLIA challenge, AskNearby empowers residents to more effectively discover local resources, plan daily activities, and engage in community life. 

---
# Q-BERT4Rec: Quantized Semantic-ID Representation Learning for Multimodal Recommendation 

**Authors**: Haofeng Huang, Ling Gai  

**Link**: [PDF](https://arxiv.org/pdf/2512.02474)  

**Abstract**: Sequential recommendation plays a critical role in modern online platforms such as e-commerce, advertising, and content streaming, where accurately predicting users' next interactions is essential for personalization. Recent Transformer-based methods like BERT4Rec have shown strong modeling capability, yet they still rely on discrete item IDs that lack semantic meaning and ignore rich multimodal information (e.g., text and image). This leads to weak generalization and limited interpretability. To address these challenges, we propose Q-Bert4Rec, a multimodal sequential recommendation framework that unifies semantic representation and quantized modeling. Specifically, Q-Bert4Rec consists of three stages: (1) cross-modal semantic injection, which enriches randomly initialized ID embeddings through a dynamic transformer that fuses textual, visual, and structural features; (2) semantic quantization, which discretizes fused representations into meaningful tokens via residual vector quantization; and (3) multi-mask pretraining and fine-tuning, which leverage diverse masking strategies -- span, tail, and multi-region -- to improve sequential understanding. We validate our model on public Amazon benchmarks and demonstrate that Q-Bert4Rec significantly outperforms many strong existing methods, confirming the effectiveness of semantic tokenization for multimodal sequential recommendation. Our source code will be publicly available on GitHub after publishing. 

---
# GraphMatch: Fusing Language and Graph Representations in a Dynamic Two-Sided Work Marketplace 

**Authors**: Mikołaj Sacha, Hammad Jafri, Mattie Terzolo, Ayan Sinha, Andrew Rabinovich  

**Link**: [PDF](https://arxiv.org/pdf/2512.02849)  

**Abstract**: Recommending matches in a text-rich, dynamic two-sided marketplace presents unique challenges due to evolving content and interaction graphs. We introduce GraphMatch, a new large-scale recommendation framework that fuses pre-trained language models with graph neural networks to overcome these challenges. Unlike prior approaches centered on standalone models, GraphMatch is a comprehensive recipe built on powerful text encoders and GNNs working in tandem. It employs adversarial negative sampling alongside point-in-time subgraph training to learn representations that capture both the fine-grained semantics of evolving text and the time-sensitive structure of the graph. We evaluated extensively on interaction data from Upwork, a leading labor marketplace, at large scale, and discuss our approach towards low-latency inference suitable for real-time use. In our experiments, GraphMatch outperforms language-only and graph-only baselines on matching tasks while being efficient at runtime. These results demonstrate that unifying language and graph representations yields a highly effective solution to text-rich, dynamic two-sided recommendations, bridging the gap between powerful pretrained LMs and large-scale graphs in practice. 

---
# Towards Unification of Hallucination Detection and Fact Verification for Large Language Models 

**Authors**: Weihang Su, Jianming Long, Changyue Wang, Shiyu Lin, Jingyan Xu, Ziyi Ye, Qingyao Ai, Yiqun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2512.02772)  

**Abstract**: Large Language Models (LLMs) frequently exhibit hallucinations, generating content that appears fluent and coherent but is factually incorrect. Such errors undermine trust and hinder their adoption in real-world applications. To address this challenge, two distinct research paradigms have emerged: model-centric Hallucination Detection (HD) and text-centric Fact Verification (FV). Despite sharing the same goal, these paradigms have evolved in isolation, using distinct assumptions, datasets, and evaluation protocols. This separation has created a research schism that hinders their collective progress. In this work, we take a decisive step toward bridging this divide. We introduce UniFact, a unified evaluation framework that enables direct, instance-level comparison between FV and HD by dynamically generating model outputs and corresponding factuality labels. Through large-scale experiments across multiple LLM families and detection methods, we reveal three key findings: (1) No paradigm is universally superior; (2) HD and FV capture complementary facets of factual errors; and (3) hybrid approaches that integrate both methods consistently achieve state-of-the-art performance. Beyond benchmarking, we provide the first in-depth analysis of why FV and HD diverged, as well as empirical evidence supporting the need for their unification. The comprehensive experimental results call for a new, integrated research agenda toward unifying Hallucination Detection and Fact Verification in LLMs.
We have open-sourced all the code, data, and baseline implementation at: this https URL 

---
# FGC-Comp: Adaptive Neighbor-Grouped Attribute Completion for Graph-based Anomaly Detection 

**Authors**: Junpeng Wu, Pinheng Zong  

**Link**: [PDF](https://arxiv.org/pdf/2512.02705)  

**Abstract**: Graph-based Anomaly Detection models have gained widespread adoption in recent years, identifying suspicious nodes by aggregating neighborhood information. However, most existing studies overlook the pervasive issues of missing and adversarially obscured node attributes, which can undermine aggregation stability and prediction reliability. To mitigate this, we propose FGC-Comp, a lightweight, classifier-agnostic, and deployment-friendly attribute completion module-designed to enhance neighborhood aggregation under incomplete attributes. We partition each node's neighbors into three label-based groups, apply group-specific transforms to the labeled groups while a node-conditioned gate handles unknowns, fuse messages via residual connections, and train end-to-end with a binary classification objective to improve aggregation stability and prediction reliability under missing attributes. Experiments on two real-world fraud datasets validate the effectiveness of the approach with negligible computational overhead. 

---
# Spatially-Grounded Document Retrieval via Patch-to-Region Relevance Propagation 

**Authors**: Agathoklis Georgiou  

**Link**: [PDF](https://arxiv.org/pdf/2512.02660)  

**Abstract**: Vision-language models (VLMs) like ColPali achieve state-of-the-art document retrieval by embedding pages as images and computing fine-grained similarity between query tokens and visual patches. However, they return entire pages rather than specific regions, limiting utility for retrieval-augmented generation (RAG) where precise context is paramount. Conversely, OCR-based systems extract structured text with bounding box coordinates but lack semantic grounding for relevance assessment. We propose a hybrid architecture that unifies these paradigms: using ColPali's patch-level similarity scores as spatial relevance filters over OCR-extracted regions. We formalize the coordinate mapping between vision transformer patch grids and OCR bounding boxes, introduce intersection metrics for relevance propagation, and establish theoretical bounds on retrieval precision. Our approach operates at inference time without additional training. We release Snappy, an open-source implementation demonstrating practical applicability, with empirical evaluation ongoing. 

---
# ADORE: Autonomous Domain-Oriented Relevance Engine for E-commerce 

**Authors**: Zheng Fang, Donghao Xie, Ming Pang, Chunyuan Yuan, Xue Jiang, Changping Peng, Zhangang Lin, Zheng Luo  

**Link**: [PDF](https://arxiv.org/pdf/2512.02555)  

**Abstract**: Relevance modeling in e-commerce search remains challenged by semantic gaps in term-matching methods (e.g., BM25) and neural models' reliance on the scarcity of domain-specific hard samples. We propose ADORE, a self-sustaining framework that synergizes three innovations: (1) A Rule-aware Relevance Discrimination module, where a Chain-of-Thought LLM generates intent-aligned training data, refined via Kahneman-Tversky Optimization (KTO) to align with user behavior; (2) An Error-type-aware Data Synthesis module that auto-generates adversarial examples to harden robustness; and (3) A Key-attribute-enhanced Knowledge Distillation module that injects domain-specific attribute hierarchies into a deployable student model. ADORE automates annotation, adversarial generation, and distillation, overcoming data scarcity while enhancing reasoning. Large-scale experiments and online A/B testing verify the effectiveness of ADORE. The framework establishes a new paradigm for resource-efficient, cognitively aligned relevance modeling in industrial applications. 

---
# WorldMM: Dynamic Multimodal Memory Agent for Long Video Reasoning 

**Authors**: Woongyeong Yeo, Kangsan Kim, Jaehong Yoon, Sung Ju Hwang  

**Link**: [PDF](https://arxiv.org/pdf/2512.02425)  

**Abstract**: Recent advances in video large language models have demonstrated strong capabilities in understanding short clips. However, scaling them to hours- or days-long videos remains highly challenging due to limited context capacity and the loss of critical visual details during abstraction. Existing memory-augmented methods mitigate this by leveraging textual summaries of video segments, yet they heavily rely on text and fail to utilize visual evidence when reasoning over complex scenes. Moreover, retrieving from fixed temporal scales further limits their flexibility in capturing events that span variable durations. To address this, we introduce WorldMM, a novel multimodal memory agent that constructs and retrieves from multiple complementary memories, encompassing both textual and visual representations. WorldMM comprises three types of memory: episodic memory indexes factual events across multiple temporal scales, semantic memory continuously updates high-level conceptual knowledge, and visual memory preserves detailed information about scenes. During inference, an adaptive retrieval agent iteratively selects the most relevant memory source and leverages multiple temporal granularities based on the query, continuing until it determines that sufficient information has been gathered. WorldMM significantly outperforms existing baselines across five long video question-answering benchmarks, achieving an average 8.4% performance gain over previous state-of-the-art methods, showing its effectiveness on long video reasoning. 

---
# Deep Research: A Systematic Survey 

**Authors**: Zhengliang Shi, Yiqun Chen, Haitao Li, Weiwei Sun, Shiyu Ni, Yougang Lyu, Run-Ze Fan, Bowen Jin, Yixuan Weng, Minjun Zhu, Qiujie Xie, Xinyu Guo, Qu Yang, Jiayi Wu, Jujia Zhao, Xiaqiang Tang, Xinbei Ma, Cunxiang Wang, Jiaxin Mao, Qingyao Ai, Jen-Tse Huang, Wenxuan Wang, Yue Zhang, Yiming Yang, Zhaopeng Tu, Zhaochun Ren  

**Link**: [PDF](https://arxiv.org/pdf/2512.02038)  

**Abstract**: Large language models (LLMs) have rapidly evolved from text generators into powerful problem solvers. Yet, many open tasks demand critical thinking, multi-source, and verifiable outputs, which are beyond single-shot prompting or standard retrieval-augmented generation. Recently, numerous studies have explored Deep Research (DR), which aims to combine the reasoning capabilities of LLMs with external tools, such as search engines, thereby empowering LLMs to act as research agents capable of completing complex, open-ended tasks. This survey presents a comprehensive and systematic overview of deep research systems, including a clear roadmap, foundational components, practical implementation techniques, important challenges, and future directions. Specifically, our main contributions are as follows: (i) we formalize a three-stage roadmap and distinguish deep research from related paradigms; (ii) we introduce four key components: query planning, information acquisition, memory management, and answer generation, each paired with fine-grained sub-taxonomies; (iii) we summarize optimization techniques, including prompting, supervised fine-tuning, and agentic reinforcement learning; and (iv) we consolidate evaluation criteria and open challenges, aiming to guide and facilitate future development. As the field of deep research continues to evolve rapidly, we are committed to continuously updating this survey to reflect the latest progress in this area. 

---
