# Process vs. Outcome Reward: Which is Better for Agentic RAG Reinforcement Learning 

**Authors**: Wenlin Zhang, Xiangyang Li, Kuicai Dong, Yichao Wang, Pengyue Jia, Xiaopeng Li, Yingyi Zhang, Derong Xu, Zhaocheng Du, Huifeng Guo, Ruiming Tang, Xiangyu Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.14069)  

**Abstract**: Retrieval-augmented generation (RAG) enhances the text generation capabilities of large language models (LLMs) by integrating external knowledge and up-to-date information. However, traditional RAG systems are limited by static workflows and lack the adaptability required for multistep reasoning and complex task management. To address these limitations, agentic RAG systems (e.g., DeepResearch) have been proposed, enabling dynamic retrieval strategies, iterative context refinement, and adaptive workflows for handling complex search queries beyond the capabilities of conventional RAG. Recent advances, such as Search-R1, have demonstrated promising gains using outcome-based reinforcement learning, where the correctness of the final answer serves as the reward signal. Nevertheless, such outcome-supervised agentic RAG methods face challenges including low exploration efficiency, gradient conflict, and sparse reward signals. To overcome these challenges, we propose to utilize fine-grained, process-level rewards to improve training stability, reduce computational costs, and enhance efficiency. Specifically, we introduce a novel method ReasonRAG that automatically constructs RAG-ProGuide, a high-quality dataset providing process-level rewards for (i) query generation, (ii) evidence extraction, and (iii) answer generation, thereby enhancing model inherent capabilities via process-supervised reinforcement learning. With the process-level policy optimization, the proposed framework empowers LLMs to autonomously invoke search, generate queries, extract relevant evidence, and produce final answers. Compared to existing approaches such as Search-R1 and traditional RAG systems, ReasonRAG, leveraging RAG-ProGuide, achieves superior performance on five benchmark datasets using only 5k training instances, significantly fewer than the 90k training instances required by Search-R1. 

---
# RAR: Setting Knowledge Tripwires for Retrieval Augmented Rejection 

**Authors**: Tommaso Mario Buonocore, Enea Parimbelli  

**Link**: [PDF](https://arxiv.org/pdf/2505.13581)  

**Abstract**: Content moderation for large language models (LLMs) remains a significant challenge, requiring flexible and adaptable solutions that can quickly respond to emerging threats. This paper introduces Retrieval Augmented Rejection (RAR), a novel approach that leverages a retrieval-augmented generation (RAG) architecture to dynamically reject unsafe user queries without model retraining. By strategically inserting and marking malicious documents into the vector database, the system can identify and reject harmful requests when these documents are retrieved. Our preliminary results show that RAR achieves comparable performance to embedded moderation in LLMs like Claude 3.5 Sonnet, while offering superior flexibility and real-time customization capabilities, a fundamental feature to timely address critical vulnerabilities. This approach introduces no architectural changes to existing RAG systems, requiring only the addition of specially crafted documents and a simple rejection mechanism based on retrieval results. 

---
# RAGXplain: From Explainable Evaluation to Actionable Guidance of RAG Pipelines 

**Authors**: Dvir Cohen, Lin Burg, Gilad Barkan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13538)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems show promise by coupling large language models with external knowledge, yet traditional RAG evaluation methods primarily report quantitative scores while offering limited actionable guidance for refining these complex pipelines. In this paper, we introduce RAGXplain, an evaluation framework that quantifies RAG performance and translates these assessments into clear insights that clarify the workings of its complex, multi-stage pipeline and offer actionable recommendations. Using LLM reasoning, RAGXplain converts raw scores into coherent narratives identifying performance gaps and suggesting targeted improvements. By providing transparent explanations for AI decision-making, our framework fosters user trust-a key challenge in AI adoption. Our LLM-based metric assessments show strong alignment with human judgments, and experiments on public question-answering datasets confirm that applying RAGXplain's actionable recommendations measurably improves system performance. RAGXplain thus bridges quantitative evaluation and practical optimization, empowering users to understand, trust, and enhance their AI systems. 

---
# AMAQA: A Metadata-based QA Dataset for RAG Systems 

**Authors**: Davide Bruni, Marco Avvenuti, Nicola Tonellotto, Maurizio Tesconi  

**Link**: [PDF](https://arxiv.org/pdf/2505.13557)  

**Abstract**: Retrieval-augmented generation (RAG) systems are widely used in question-answering (QA) tasks, but current benchmarks lack metadata integration, hindering evaluation in scenarios requiring both textual data and external information. To address this, we present AMAQA, a new open-access QA dataset designed to evaluate tasks combining text and metadata. The integration of metadata is especially important in fields that require rapid analysis of large volumes of data, such as cybersecurity and intelligence, where timely access to relevant information is critical. AMAQA includes about 1.1 million English messages collected from 26 public Telegram groups, enriched with metadata such as timestamps, topics, emotional tones, and toxicity indicators, which enable precise and contextualized queries by filtering documents based on specific criteria. It also includes 450 high-quality QA pairs, making it a valuable resource for advancing research on metadata-driven QA and RAG systems. To the best of our knowledge, AMAQA is the first single-hop QA benchmark to incorporate metadata and labels such as topics covered in the messages. We conduct extensive tests on the benchmark, establishing a new standard for future research. We show that leveraging metadata boosts accuracy from 0.12 to 0.61, highlighting the value of structured context. Building on this, we explore several strategies to refine the LLM input by iterating over provided context and enriching it with noisy documents, achieving a further 3-point gain over the best baseline and a 14-point improvement over simple metadata filtering. The dataset is available at this https URL 

---
# Divide by Question, Conquer by Agent: SPLIT-RAG with Question-Driven Graph Partitioning 

**Authors**: Ruiyi Yang, Hao Xue, Imran Razzak, Hakim Hacid, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2505.13994)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems empower large language models (LLMs) with external knowledge, yet struggle with efficiency-accuracy trade-offs when scaling to large knowledge graphs. Existing approaches often rely on monolithic graph retrieval, incurring unnecessary latency for simple queries and fragmented reasoning for complex multi-hop questions. To address these challenges, this paper propose SPLIT-RAG, a multi-agent RAG framework that addresses these limitations with question-driven semantic graph partitioning and collaborative subgraph retrieval. The innovative framework first create Semantic Partitioning of Linked Information, then use the Type-Specialized knowledge base to achieve Multi-Agent RAG. The attribute-aware graph segmentation manages to divide knowledge graphs into semantically coherent subgraphs, ensuring subgraphs align with different query types, while lightweight LLM agents are assigned to partitioned subgraphs, and only relevant partitions are activated during retrieval, thus reduce search space while enhancing efficiency. Finally, a hierarchical merging module resolves inconsistencies across subgraph-derived answers through logical verifications. Extensive experimental validation demonstrates considerable improvements compared to existing approaches. 

---
# EcoSafeRAG: Efficient Security through Context Analysis in Retrieval-Augmented Generation 

**Authors**: Ruobing Yao, Yifei Zhang, Shuang Song, Neng Gao, Chenyang Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13506)  

**Abstract**: Retrieval-Augmented Generation (RAG) compensates for the static knowledge limitations of Large Language Models (LLMs) by integrating external knowledge, producing responses with enhanced factual correctness and query-specific contextualization. However, it also introduces new attack surfaces such as corpus poisoning at the same time. Most of the existing defense methods rely on the internal knowledge of the model, which conflicts with the design concept of RAG. To bridge the gap, EcoSafeRAG uses sentence-level processing and bait-guided context diversity detection to identify malicious content by analyzing the context diversity of candidate documents without relying on LLM internal knowledge. Experiments show EcoSafeRAG delivers state-of-the-art security with plug-and-play deployment, simultaneously improving clean-scenario RAG performance while maintaining practical operational costs (relatively 1.2$\times$ latency, 48\%-80\% token reduction versus Vanilla RAG). 

---
# RAVENEA: A Benchmark for Multimodal Retrieval-Augmented Visual Culture Understanding 

**Authors**: Jiaang Li, Yifei Yuan, Wenyan Li, Mohammad Aliannejadi, Daniel Hershcovich, Anders Søgaard, Ivan Vulić, Wenxuan Zhang, Paul Pu Liang, Yang Deng, Serge Belongie  

**Link**: [PDF](https://arxiv.org/pdf/2505.14462)  

**Abstract**: As vision-language models (VLMs) become increasingly integrated into daily life, the need for accurate visual culture understanding is becoming critical. Yet, these models frequently fall short in interpreting cultural nuances effectively. Prior work has demonstrated the effectiveness of retrieval-augmented generation (RAG) in enhancing cultural understanding in text-only settings, while its application in multimodal scenarios remains underexplored. To bridge this gap, we introduce RAVENEA (Retrieval-Augmented Visual culturE uNdErstAnding), a new benchmark designed to advance visual culture understanding through retrieval, focusing on two tasks: culture-focused visual question answering (cVQA) and culture-informed image captioning (cIC). RAVENEA extends existing datasets by integrating over 10,000 Wikipedia documents curated and ranked by human annotators. With RAVENEA, we train and evaluate seven multimodal retrievers for each image query, and measure the downstream impact of retrieval-augmented inputs across fourteen state-of-the-art VLMs. Our results show that lightweight VLMs, when augmented with culture-aware retrieval, outperform their non-augmented counterparts (by at least 3.2% absolute on cVQA and 6.2% absolute on cIC). This highlights the value of retrieval-augmented methods and culturally inclusive benchmarks for multimodal understanding. 

---
# Causal Cartographer: From Mapping to Reasoning Over Counterfactual Worlds 

**Authors**: Gaël Gendron, Jože M. Rožanec, Michael Witbrock, Gillian Dobbie  

**Link**: [PDF](https://arxiv.org/pdf/2505.14396)  

**Abstract**: Causal world models are systems that can answer counterfactual questions about an environment of interest, i.e. predict how it would have evolved if an arbitrary subset of events had been realized differently. It requires understanding the underlying causes behind chains of events and conducting causal inference for arbitrary unseen distributions. So far, this task eludes foundation models, notably large language models (LLMs), which do not have demonstrated causal reasoning capabilities beyond the memorization of existing causal relationships. Furthermore, evaluating counterfactuals in real-world applications is challenging since only the factual world is observed, limiting evaluation to synthetic datasets. We address these problems by explicitly extracting and modeling causal relationships and propose the Causal Cartographer framework. First, we introduce a graph retrieval-augmented generation agent tasked to retrieve causal relationships from data. This approach allows us to construct a large network of real-world causal relationships that can serve as a repository of causal knowledge and build real-world counterfactuals. In addition, we create a counterfactual reasoning agent constrained by causal relationships to perform reliable step-by-step causal inference. We show that our approach can extract causal knowledge and improve the robustness of LLMs for causal reasoning tasks while reducing inference costs and spurious correlations. 

---
# SCAN: Semantic Document Layout Analysis for Textual and Visual Retrieval-Augmented Generation 

**Authors**: Yuyang Dong, Nobuhiro Ueda, Krisztián Boros, Daiki Ito, Takuya Sera, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2505.14381)  

**Abstract**: With the increasing adoption of Large Language Models (LLMs) and Vision-Language Models (VLMs), rich document analysis technologies for applications like Retrieval-Augmented Generation (RAG) and visual RAG are gaining significant attention. Recent research indicates that using VLMs can achieve better RAG performance, but processing rich documents still remains a challenge since a single page contains large amounts of information. In this paper, we present SCAN (\textbf{S}emanti\textbf{C} Document Layout \textbf{AN}alysis), a novel approach enhancing both textual and visual Retrieval-Augmented Generation (RAG) systems working with visually rich documents. It is a VLM-friendly approach that identifies document components with appropriate semantic granularity, balancing context preservation with processing efficiency. SCAN uses a coarse-grained semantic approach that divides documents into coherent regions covering continuous components. We trained the SCAN model by fine-tuning object detection models with sophisticated annotation datasets. Our experimental results across English and Japanese datasets demonstrate that applying SCAN improves end-to-end textual RAG performance by up to 9.0\% and visual RAG performance by up to 6.4\%, outperforming conventional approaches and even commercial document processing solutions. 

---
# Multimodal RAG-driven Anomaly Detection and Classification in Laser Powder Bed Fusion using Large Language Models 

**Authors**: Kiarash Naghavi Khanghah, Zhiling Chen, Lela Romeo, Qian Yang, Rajiv Malhotra, Farhad Imani, Hongyi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13828)  

**Abstract**: Additive manufacturing enables the fabrication of complex designs while minimizing waste, but faces challenges related to defects and process anomalies. This study presents a novel multimodal Retrieval-Augmented Generation-based framework that automates anomaly detection across various Additive Manufacturing processes leveraging retrieved information from literature, including images and descriptive text, rather than training datasets. This framework integrates text and image retrieval from scientific literature and multimodal generation models to perform zero-shot anomaly identification, classification, and explanation generation in a Laser Powder Bed Fusion setting. The proposed framework is evaluated on four L-PBF manufacturing datasets from Oak Ridge National Laboratory, featuring various printer makes, models, and materials. This evaluation demonstrates the framework's adaptability and generalizability across diverse images without requiring additional training. Comparative analysis using Qwen2-VL-2B and GPT-4o-mini as MLLM within the proposed framework highlights that GPT-4o-mini outperforms Qwen2-VL-2B and proportional random baseline in manufacturing anomalies classification. Additionally, the evaluation of the RAG system confirms that incorporating retrieval mechanisms improves average accuracy by 12% by reducing the risk of hallucination and providing additional information. The proposed framework can be continuously updated by integrating emerging research, allowing seamless adaptation to the evolving landscape of AM technologies. This scalable, automated, and zero-shot-capable framework streamlines AM anomaly analysis, enhancing efficiency and accuracy. 

---
