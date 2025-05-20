# Batched Self-Consistency Improves LLM Relevance Assessment and Ranking 

**Authors**: Anton Korikov, Pan Du, Scott Sanner, Navid Rekabsaz  

**Link**: [PDF](https://arxiv.org/pdf/2505.12570)  

**Abstract**: Given some information need, Large Language Models (LLMs) are increasingly used for candidate text relevance assessment, typically using a one-by-one pointwise (PW) strategy where each LLM call evaluates one candidate at a time. Meanwhile, it has been shown that LLM performance can be improved through self-consistency: prompting the LLM to do the same task multiple times (possibly in perturbed ways) and then aggregating the responses. To take advantage of self-consistency, we hypothesize that batched PW strategies, where multiple passages are judged in one LLM call, are better suited than one-by-one PW methods since a larger input context can induce more diverse LLM sampling across self-consistency calls. We first propose several candidate batching strategies to create prompt diversity across self-consistency calls through subset reselection and permutation. We then test our batched PW methods on relevance assessment and ranking tasks against one-by-one PW and listwise LLM ranking baselines with and without self-consistency, using three passage retrieval datasets and GPT-4o, Claude Sonnet 3, and Amazon Nova Pro. We find that batched PW methods outperform all baselines, and show that batching can greatly amplify the positive effects of self-consistency. For instance, on our legal search dataset, GPT-4o one-by-one PW ranking NDCG@10 improves only from 44.9% to 46.8% without self-consistency vs. with 15 self consistency calls, while batched PW ranking improves from 43.8% to 51.3%, respectively. 

---
# LightRetriever: A LLM-based Hybrid Retrieval Architecture with 1000x Faster Query Inference 

**Authors**: Guangyuan Ma, Yongliang Ma, Xuanrui Gou, Zhenpeng Su, Ming Zhou, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12260)  

**Abstract**: Large Language Models (LLMs)-based hybrid retrieval uses LLMs to encode queries and documents into low-dimensional dense or high-dimensional sparse vectors. It retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based hybrid retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full-sized LLM on an H800 GPU, our approach achieves over a 1000x speedup for query inference with GPU acceleration, and even a 20x speedup without GPU. Experiments on large-scale retrieval benchmarks demonstrate that our method generalizes well across diverse retrieval tasks, retaining an average of 95% full-sized performance. 

---
# LLM-based Query Expansion Fails for Unfamiliar and Ambiguous Queries 

**Authors**: Kenya Abe, Kunihiro Takeoka, Makoto P. Kato, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2505.12694)  

**Abstract**: Query expansion (QE) enhances retrieval by incorporating relevant terms, with large language models (LLMs) offering an effective alternative to traditional rule-based and statistical methods. However, LLM-based QE suffers from a fundamental limitation: it often fails to generate relevant knowledge, degrading search performance. Prior studies have focused on hallucination, yet its underlying cause--LLM knowledge deficiencies--remains underexplored. This paper systematically examines two failure cases in LLM-based QE: (1) when the LLM lacks query knowledge, leading to incorrect expansions, and (2) when the query is ambiguous, causing biased refinements that narrow search coverage. We conduct controlled experiments across multiple datasets, evaluating the effects of knowledge and query ambiguity on retrieval performance using sparse and dense retrieval models. Our results reveal that LLM-based QE can significantly degrade the retrieval effectiveness when knowledge in the LLM is insufficient or query ambiguity is high. We introduce a framework for evaluating QE under these conditions, providing insights into the limitations of LLM-based retrieval augmentation. 

---
# LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization 

**Authors**: Hailong Luo, Bin Wu, Hongyong Jia, Qingqing Zhu, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12396)  

**Abstract**: Graph neural networks (GNNs) have advanced recommender systems by modeling interaction relationships. However, existing graph-based recommenders rely on sparse ID features and do not fully exploit textual information, resulting in low information density within representations. Furthermore, graph contrastive learning faces challenges. Random negative sampling can introduce false negative samples, while fixed temperature coefficients cannot adapt to the heterogeneity of different nodes. In addition, current efforts to enhance recommendations with large language models (LLMs) have not fully utilized their Chain-of-Thought (CoT) reasoning capabilities to guide representation learning. To address these limitations, we introduces LGHRec (LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization). This framework leverages the CoT reasoning ability of LLMs to generate semantic IDs, enriching reasoning processes and improving information density and semantic quality of representations. Moreover, we design a reinforcement learning algorithm, Harmonized Group Policy Optimization (HGPO), to optimize negative sampling strategies and temperature coefficients in contrastive learning. This approach enhances long-tail recommendation performance and ensures optimization consistency across different groups. Experimental results on three datasets demonstrate that LGHRec improves representation quality through semantic IDs generated by LLM's CoT reasoning and effectively boosts contrastive learning with HGPO. Our method outperforms several baseline models. The code is available at: this https URL. 

---
# Telco-oRAG: Optimizing Retrieval-augmented Generation for Telecom Queries via Hybrid Retrieval and Neural Routing 

**Authors**: Andrei-Laurentiu Bornea, Fadhel Ayed, Antonio De Domenico, Nicola Piovesan, Tareq Si Salem, Ali Maatouk  

**Link**: [PDF](https://arxiv.org/pdf/2505.11856)  

**Abstract**: Artificial intelligence will be one of the key pillars of the next generation of mobile networks (6G), as it is expected to provide novel added-value services and improve network performance. In this context, large language models have the potential to revolutionize the telecom landscape through intent comprehension, intelligent knowledge retrieval, coding proficiency, and cross-domain orchestration capabilities. This paper presents Telco-oRAG, an open-source Retrieval-Augmented Generation (RAG) framework optimized for answering technical questions in the telecommunications domain, with a particular focus on 3GPP standards. Telco-oRAG introduces a hybrid retrieval strategy that combines 3GPP domain-specific retrieval with web search, supported by glossary-enhanced query refinement and a neural router for memory-efficient retrieval. Our results show that Telco-oRAG improves the accuracy in answering 3GPP-related questions by up to 17.6% and achieves a 10.6% improvement in lexicon queries compared to baselines. Furthermore, Telco-oRAG reduces memory usage by 45% through targeted retrieval of relevant 3GPP series compared to baseline RAG, and enables open-source LLMs to reach GPT-4-level accuracy on telecom benchmarks. 

---
# Terminators: Terms of Service Parsing and Auditing Agents 

**Authors**: Maruf Ahmed Mridul, Inwon Kang, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2505.11672)  

**Abstract**: Terms of Service (ToS) documents are often lengthy and written in complex legal language, making them difficult for users to read and understand. To address this challenge, we propose Terminators, a modular agentic framework that leverages large language models (LLMs) to parse and audit ToS documents. Rather than treating ToS understanding as a black-box summarization problem, Terminators breaks the task down to three interpretable steps: term extraction, verification, and accountability planning. We demonstrate the effectiveness of our method on the OpenAI ToS using GPT-4o, highlighting strategies to minimize hallucinations and maximize auditability. Our results suggest that structured, agent-based LLM workflows can enhance both the usability and enforceability of complex legal documents. By translating opaque terms into actionable, verifiable components, Terminators promotes ethical use of web content by enabling greater transparency, empowering users to understand their digital rights, and supporting automated policy audits for regulatory or civic oversight. 

---
# The Effects of Demographic Instructions on LLM Personas 

**Authors**: Angel Felipe Magnossão de Paula, J. Shane Culpepper, Alistair Moffat, Sachin Pathiyan Cherumanal, Falk Scholer, Johanne Trippas  

**Link**: [PDF](https://arxiv.org/pdf/2505.11795)  

**Abstract**: Social media platforms must filter sexist content in compliance with governmental regulations. Current machine learning approaches can reliably detect sexism based on standardized definitions, but often neglect the subjective nature of sexist language and fail to consider individual users' perspectives. To address this gap, we adopt a perspectivist approach, retaining diverse annotations rather than enforcing gold-standard labels or their aggregations, allowing models to account for personal or group-specific views of sexism. Using demographic data from Twitter, we employ large language models (LLMs) to personalize the identification of sexism. 

---
# Introspective Growth: Automatically Advancing LLM Expertise in Technology Judgment 

**Authors**: Siyang Wu, Honglin Bao, Nadav Kunievsky, James A. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2505.12452)  

**Abstract**: Large language models (LLMs) increasingly demonstrate signs of conceptual understanding, yet much of their internal knowledge remains latent, loosely structured, and difficult to access or evaluate. We propose self-questioning as a lightweight and scalable strategy to improve LLMs' understanding, particularly in domains where success depends on fine-grained semantic distinctions. To evaluate this approach, we introduce a challenging new benchmark of 1.3 million post-2015 computer science patent pairs, characterized by dense technical jargon and strategically complex writing. The benchmark centers on a pairwise differentiation task: can a model distinguish between closely related but substantively different inventions? We show that prompting LLMs to generate and answer their own questions - targeting the background knowledge required for the task - significantly improves performance. These self-generated questions and answers activate otherwise underutilized internal knowledge. Allowing LLMs to retrieve answers from external scientific texts further enhances performance, suggesting that model knowledge is compressed and lacks the full richness of the training data. We also find that chain-of-thought prompting and self-questioning converge, though self-questioning remains more effective for improving understanding of technical concepts. Notably, we uncover an asymmetry in prompting: smaller models often generate more fundamental, more open-ended, better-aligned questions for mid-sized models than large models with better understanding do, revealing a new strategy for cross-model collaboration. Altogether, our findings establish self-questioning as both a practical mechanism for automatically improving LLM comprehension, especially in domains with sparse and underrepresented knowledge, and a diagnostic probe of how internal and external knowledge are organized. 

---
# Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents 

**Authors**: Tiannuo Yang, Zebin Yao, Bowen Jin, Lixiao Cui, Yusen Li, Gang Wang, Xiaoguang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12065)  

**Abstract**: Large Language Model (LLM)-based search agents have shown remarkable capabilities in solving complex tasks by dynamically decomposing problems and addressing them through interleaved reasoning and retrieval. However, this interleaved paradigm introduces substantial efficiency bottlenecks. First, we observe that both highly accurate and overly approximate retrieval methods degrade system efficiency: exact search incurs significant retrieval overhead, while coarse retrieval requires additional reasoning steps during generation. Second, we identify inefficiencies in system design, including improper scheduling and frequent retrieval stalls, which lead to cascading latency -- where even minor delays in retrieval amplify end-to-end inference time. To address these challenges, we introduce SearchAgent-X, a high-efficiency inference framework for LLM-based search agents. SearchAgent-X leverages high-recall approximate retrieval and incorporates two key techniques: priority-aware scheduling and non-stall retrieval. Extensive experiments demonstrate that SearchAgent-X consistently outperforms state-of-the-art systems such as vLLM and HNSW-based retrieval across diverse tasks, achieving up to 3.4$\times$ higher throughput and 5$\times$ lower latency, without compromising generation quality. SearchAgent-X is available at this https URL. 

---
# Think Before You Attribute: Improving the Performance of LLMs Attribution Systems 

**Authors**: João Eduardo Batista, Emil Vatai, Mohamed Wahib  

**Link**: [PDF](https://arxiv.org/pdf/2505.12621)  

**Abstract**: Large Language Models (LLMs) are increasingly applied in various science domains, yet their broader adoption remains constrained by a critical challenge: the lack of trustworthy, verifiable outputs. Current LLMs often generate answers without reliable source attribution, or worse, with incorrect attributions, posing a barrier to their use in scientific and high-stakes settings, where traceability and accountability are non-negotiable. To be reliable, attribution systems need high accuracy and retrieve data with short lengths, i.e., attribute to a sentence within a document rather than a whole document. We propose a sentence-level pre-attribution step for Retrieve-Augmented Generation (RAG) systems that classify sentences into three categories: not attributable, attributable to a single quote, and attributable to multiple quotes. By separating sentences before attribution, a proper attribution method can be selected for the type of sentence, or the attribution can be skipped altogether. Our results indicate that classifiers are well-suited for this task. In this work, we propose a pre-attribution step to reduce the computational complexity of attribution, provide a clean version of the HAGRID dataset, and provide an end-to-end attribution system that works out of the box. 

---
# AutoMathKG: The automated mathematical knowledge graph based on LLM and vector database 

**Authors**: Rong Bian, Yu Geng, Zijian Yang, Bing Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.13406)  

**Abstract**: A mathematical knowledge graph (KG) presents knowledge within the field of mathematics in a structured manner. Constructing a math KG using natural language is an essential but challenging task. There are two major limitations of existing works: first, they are constrained by corpus completeness, often discarding or manually supplementing incomplete knowledge; second, they typically fail to fully automate the integration of diverse knowledge sources. This paper proposes AutoMathKG, a high-quality, wide-coverage, and multi-dimensional math KG capable of automatic updates. AutoMathKG regards mathematics as a vast directed graph composed of Definition, Theorem, and Problem entities, with their reference relationships as edges. It integrates knowledge from ProofWiki, textbooks, arXiv papers, and TheoremQA, enhancing entities and relationships with large language models (LLMs) via in-context learning for data augmentation. To search for similar entities, MathVD, a vector database, is built through two designed embedding strategies using SBERT. To automatically update, two mechanisms are proposed. For knowledge completion mechanism, Math LLM is developed to interact with AutoMathKG, providing missing proofs or solutions. For knowledge fusion mechanism, MathVD is used to retrieve similar entities, and LLM is used to determine whether to merge with a candidate or add as a new entity. A wide range of experiments demonstrate the advanced performance and broad applicability of the AutoMathKG system, including superior reachability query results in MathVD compared to five baselines and robust mathematical reasoning capability in Math LLM. 

---
# MM-PRM: Enhancing Multimodal Mathematical Reasoning with Scalable Step-Level Supervision 

**Authors**: Lingxiao Du, Fanqing Meng, Zongkai Liu, Zhixiang Zhou, Ping Luo, Qiaosheng Zhang, Wenqi Shao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13427)  

**Abstract**: While Multimodal Large Language Models (MLLMs) have achieved impressive progress in vision-language understanding, they still struggle with complex multi-step reasoning, often producing logically inconsistent or partially correct solutions. A key limitation lies in the lack of fine-grained supervision over intermediate reasoning steps. To address this, we propose MM-PRM, a process reward model trained within a fully automated, scalable framework. We first build MM-Policy, a strong multimodal model trained on diverse mathematical reasoning data. Then, we construct MM-K12, a curated dataset of 10,000 multimodal math problems with verifiable answers, which serves as seed data. Leveraging a Monte Carlo Tree Search (MCTS)-based pipeline, we generate over 700k step-level annotations without human labeling. The resulting PRM is used to score candidate reasoning paths in the Best-of-N inference setup and achieves significant improvements across both in-domain (MM-K12 test set) and out-of-domain (OlympiadBench, MathVista, etc.) benchmarks. Further analysis confirms the effectiveness of soft labels, smaller learning rates, and path diversity in optimizing PRM performance. MM-PRM demonstrates that process supervision is a powerful tool for enhancing the logical robustness of multimodal reasoning systems. We release all our codes and data at this https URL. 

---
# CoT-Kinetics: A Theoretical Modeling Assessing LRM Reasoning Process 

**Authors**: Jinhe Bi, Danqi Yan, Yifan Wang, Wenke Huang, Haokun Chen, Guancheng Wan, Mang Ye, Xun Xiao, Hinrich Schuetze, Volker Tresp, Yunpu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.13408)  

**Abstract**: Recent Large Reasoning Models significantly improve the reasoning ability of Large Language Models by learning to reason, exhibiting the promising performance in solving complex tasks. LRMs solve tasks that require complex reasoning by explicitly generating reasoning trajectories together with answers. Nevertheless, judging the quality of such an output answer is not easy because only considering the correctness of the answer is not enough and the soundness of the reasoning trajectory part matters as well. Logically, if the soundness of the reasoning part is poor, even if the answer is correct, the confidence of the derived answer should be low. Existing methods did consider jointly assessing the overall output answer by taking into account the reasoning part, however, their capability is still not satisfactory as the causal relationship of the reasoning to the concluded answer cannot properly reflected. In this paper, inspired by classical mechanics, we present a novel approach towards establishing a CoT-Kinetics energy equation. Specifically, our CoT-Kinetics energy equation formulates the token state transformation process, which is regulated by LRM internal transformer layers, as like a particle kinetics dynamics governed in a mechanical field. Our CoT-Kinetics energy assigns a scalar score to evaluate specifically the soundness of the reasoning phase, telling how confident the derived answer could be given the evaluated reasoning. As such, the LRM's overall output quality can be accurately measured, rather than a coarse judgment (e.g., correct or incorrect) anymore. 

---
# Trust, But Verify: A Self-Verification Approach to Reinforcement Learning with Verifiable Rewards 

**Authors**: Xiaoyuan Liu, Tian Liang, Zhiwei He, Jiahao Xu, Wenxuan Wang, Pinjia He, Zhaopeng Tu, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13445)  

**Abstract**: Large Language Models (LLMs) show great promise in complex reasoning, with Reinforcement Learning with Verifiable Rewards (RLVR) being a key enhancement strategy. However, a prevalent issue is ``superficial self-reflection'', where models fail to robustly verify their own outputs. We introduce RISE (Reinforcing Reasoning with Self-Verification), a novel online RL framework designed to tackle this. RISE explicitly and simultaneously trains an LLM to improve both its problem-solving and self-verification abilities within a single, integrated RL process. The core mechanism involves leveraging verifiable rewards from an outcome verifier to provide on-the-fly feedback for both solution generation and self-verification tasks. In each iteration, the model generates solutions, then critiques its own on-policy generated solutions, with both trajectories contributing to the policy update. Extensive experiments on diverse mathematical reasoning benchmarks show that RISE consistently improves model's problem-solving accuracy while concurrently fostering strong self-verification skills. Our analyses highlight the advantages of online verification and the benefits of increased verification compute. Additionally, RISE models exhibit more frequent and accurate self-verification behaviors during reasoning. These advantages reinforce RISE as a flexible and effective path towards developing more robust and self-aware reasoners. 

---
# Multi-Armed Bandits Meet Large Language Models 

**Authors**: Djallel Bouneffouf, Raphael Feraud  

**Link**: [PDF](https://arxiv.org/pdf/2505.13355)  

**Abstract**: Bandit algorithms and Large Language Models (LLMs) have emerged as powerful tools in artificial intelligence, each addressing distinct yet complementary challenges in decision-making and natural language processing. This survey explores the synergistic potential between these two fields, highlighting how bandit algorithms can enhance the performance of LLMs and how LLMs, in turn, can provide novel insights for improving bandit-based decision-making. We first examine the role of bandit algorithms in optimizing LLM fine-tuning, prompt engineering, and adaptive response generation, focusing on their ability to balance exploration and exploitation in large-scale learning tasks. Subsequently, we explore how LLMs can augment bandit algorithms through advanced contextual understanding, dynamic adaptation, and improved policy selection using natural language reasoning. By providing a comprehensive review of existing research and identifying key challenges and opportunities, this survey aims to bridge the gap between bandit algorithms and LLMs, paving the way for innovative applications and interdisciplinary research in AI. 

---
# Agentic Publications: An LLM-Driven Framework for Interactive Scientific Publishing, Supplementing Traditional Papers with AI-Powered Knowledge Systems 

**Authors**: Roberto Pugliese, George Kourousias, Francesco Venier, Grazia Garlatti Costa  

**Link**: [PDF](https://arxiv.org/pdf/2505.13246)  

**Abstract**: The exponential growth of scientific literature presents significant challenges for researchers navigating the complex knowledge landscape. We propose "Agentic Publications", a novel LLM-driven framework complementing traditional publishing by transforming papers into interactive knowledge systems. Our architecture integrates structured data with unstructured content through retrieval-augmented generation and multi-agent verification. The framework offers interfaces for both humans and machines, combining narrative explanations with machine-readable outputs while addressing ethical considerations through automated validation and transparent governance. Key features include continuous knowledge updates, automatic integration of new findings, and customizable detail levels. Our proof-of-concept demonstrates multilingual interaction, API accessibility, and structured knowledge representation through vector databases, knowledge graphs, and verification agents. This approach enhances scientific communication across disciplines, improving efficiency and collaboration while preserving traditional publishing pathways, particularly valuable for interdisciplinary fields where knowledge integration remains challenging. 

---
# Adversarial Testing in LLMs: Insights into Decision-Making Vulnerabilities 

**Authors**: Lili Zhang, Haomiaomiao Wang, Long Cheng, Libao Deng, Tomas Ward  

**Link**: [PDF](https://arxiv.org/pdf/2505.13195)  

**Abstract**: As Large Language Models (LLMs) become increasingly integrated into real-world decision-making systems, understanding their behavioural vulnerabilities remains a critical challenge for AI safety and alignment. While existing evaluation metrics focus primarily on reasoning accuracy or factual correctness, they often overlook whether LLMs are robust to adversarial manipulation or capable of using adaptive strategy in dynamic environments. This paper introduces an adversarial evaluation framework designed to systematically stress-test the decision-making processes of LLMs under interactive and adversarial conditions. Drawing on methodologies from cognitive psychology and game theory, our framework probes how models respond in two canonical tasks: the two-armed bandit task and the Multi-Round Trust Task. These tasks capture key aspects of exploration-exploitation trade-offs, social cooperation, and strategic flexibility. We apply this framework to several state-of-the-art LLMs, including GPT-3.5, GPT-4, Gemini-1.5, and DeepSeek-V3, revealing model-specific susceptibilities to manipulation and rigidity in strategy adaptation. Our findings highlight distinct behavioral patterns across models and emphasize the importance of adaptability and fairness recognition for trustworthy AI deployment. Rather than offering a performance benchmark, this work proposes a methodology for diagnosing decision-making weaknesses in LLM-based agents, providing actionable insights for alignment and safety research. 

---
# CAIM: Development and Evaluation of a Cognitive AI Memory Framework for Long-Term Interaction with Intelligent Agents 

**Authors**: Rebecca Westhäußer, Frederik Berenz, Wolfgang Minker, Sebastian Zepf  

**Link**: [PDF](https://arxiv.org/pdf/2505.13044)  

**Abstract**: Large language models (LLMs) have advanced the field of artificial intelligence (AI) and are a powerful enabler for interactive systems. However, they still face challenges in long-term interactions that require adaptation towards the user as well as contextual knowledge and understanding of the ever-changing environment. To overcome these challenges, holistic memory modeling is required to efficiently retrieve and store relevant information across interaction sessions for suitable responses. Cognitive AI, which aims to simulate the human thought process in a computerized model, highlights interesting aspects, such as thoughts, memory mechanisms, and decision-making, that can contribute towards improved memory modeling for LLMs. Inspired by these cognitive AI principles, we propose our memory framework CAIM. CAIM consists of three modules: 1.) The Memory Controller as the central decision unit; 2.) the Memory Retrieval, which filters relevant data for interaction upon request; and 3.) the Post-Thinking, which maintains the memory storage. We compare CAIM against existing approaches, focusing on metrics such as retrieval accuracy, response correctness, contextual coherence, and memory storage. The results demonstrate that CAIM outperforms baseline frameworks across different metrics, highlighting its context-awareness and potential to improve long-term human-AI interactions. 

---
# The Traitors: Deception and Trust in Multi-Agent Language Model Simulations 

**Authors**: Pedro M. P. Curvo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12923)  

**Abstract**: As AI systems increasingly assume roles where trust and alignment with human values are essential, understanding when and why they engage in deception has become a critical research priority. We introduce The Traitors, a multi-agent simulation framework inspired by social deduction games, designed to probe deception, trust formation, and strategic communication among large language model (LLM) agents under asymmetric information. A minority of agents the traitors seek to mislead the majority, while the faithful must infer hidden identities through dialogue and reasoning. Our contributions are: (1) we ground the environment in formal frameworks from game theory, behavioral economics, and social cognition; (2) we develop a suite of evaluation metrics capturing deception success, trust dynamics, and collective inference quality; (3) we implement a fully autonomous simulation platform where LLMs reason over persistent memory and evolving social dynamics, with support for heterogeneous agent populations, specialized traits, and adaptive behaviors. Our initial experiments across DeepSeek-V3, GPT-4o-mini, and GPT-4o (10 runs per model) reveal a notable asymmetry: advanced models like GPT-4o demonstrate superior deceptive capabilities yet exhibit disproportionate vulnerability to others' falsehoods. This suggests deception skills may scale faster than detection abilities. Overall, The Traitors provides a focused, configurable testbed for investigating LLM behavior in socially nuanced interactions. We position this work as a contribution toward more rigorous research on deception mechanisms, alignment challenges, and the broader social reliability of AI systems. 

---
# LLM-KG-Bench 3.0: A Compass for SemanticTechnology Capabilities in the Ocean of LLMs 

**Authors**: Lars-Peter Meyer, Johannes Frey, Desiree Heim, Felix Brei, Claus Stadler, Kurt Junghanns, Michael Martin  

**Link**: [PDF](https://arxiv.org/pdf/2505.13098)  

**Abstract**: Current Large Language Models (LLMs) can assist developing program code beside many other things, but can they support working with Knowledge Graphs (KGs) as well? Which LLM is offering the best capabilities in the field of Semantic Web and Knowledge Graph Engineering (KGE)? Is this possible to determine without checking many answers manually? The LLM-KG-Bench framework in Version 3.0 is designed to answer these questions. It consists of an extensible set of tasks for automated evaluation of LLM answers and covers different aspects of working with semantic technologies. In this paper the LLM-KG-Bench framework is presented in Version 3 along with a dataset of prompts, answers and evaluations generated with it and several state-of-the-art LLMs. Significant enhancements have been made to the framework since its initial release, including an updated task API that offers greater flexibility in handling evaluation tasks, revised tasks, and extended support for various open models through the vllm library, among other improvements. A comprehensive dataset has been generated using more than 30 contemporary open and proprietary LLMs, enabling the creation of exemplary model cards that demonstrate the models' capabilities in working with RDF and SPARQL, as well as comparing their performance on Turtle and JSON-LD RDF serialization tasks. 

---
# Enhancing LLMs for Time Series Forecasting via Structure-Guided Cross-Modal Alignment 

**Authors**: Siming Sun, Kai Zhang, Xuejun Jiang, Wenchao Meng, Qinmin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13175)  

**Abstract**: The emerging paradigm of leveraging pretrained large language models (LLMs) for time series forecasting has predominantly employed linguistic-temporal modality alignment strategies through token-level or layer-wise feature mapping. However, these approaches fundamentally neglect a critical insight: the core competency of LLMs resides not merely in processing localized token features but in their inherent capacity to model holistic sequence structures. This paper posits that effective cross-modal alignment necessitates structural consistency at the sequence level. We propose the Structure-Guided Cross-Modal Alignment (SGCMA), a framework that fully exploits and aligns the state-transition graph structures shared by time-series and linguistic data as sequential modalities, thereby endowing time series with language-like properties and delivering stronger generalization after modality alignment. SGCMA consists of two key components, namely Structure Alignment and Semantic Alignment. In Structure Alignment, a state transition matrix is learned from text data through Hidden Markov Models (HMMs), and a shallow transformer-based Maximum Entropy Markov Model (MEMM) receives the hot-start transition matrix and annotates each temporal patch into state probability, ensuring that the temporal representation sequence inherits language-like sequential dynamics. In Semantic Alignment, cross-attention is applied between temporal patches and the top-k tokens within each state, and the ultimate temporal embeddings are derived by the expected value of these embeddings using a weighted average based on state probabilities. Experiments on multiple benchmarks demonstrate that SGCMA achieves state-of-the-art performance, offering a novel approach to cross-modal alignment in time series forecasting. 

---
# TIME: A Multi-level Benchmark for Temporal Reasoning of LLMs in Real-World Scenarios 

**Authors**: Shaohang Wei, Wei Li, Feifan Song, Wen Luo, Tianyi Zhuang, Haochen Tan, Zhijiang Guo, Houfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12891)  

**Abstract**: Temporal reasoning is pivotal for Large Language Models (LLMs) to comprehend the real world. However, existing works neglect the real-world challenges for temporal reasoning: (1) intensive temporal information, (2) fast-changing event dynamics, and (3) complex temporal dependencies in social interactions. To bridge this gap, we propose a multi-level benchmark TIME, designed for temporal reasoning in real-world scenarios. TIME consists of 38,522 QA pairs, covering 3 levels with 11 fine-grained sub-tasks. This benchmark encompasses 3 sub-datasets reflecting different real-world challenges: TIME-Wiki, TIME-News, and TIME-Dial. We conduct extensive experiments on reasoning models and non-reasoning models. And we conducted an in-depth analysis of temporal reasoning performance across diverse real-world scenarios and tasks, and summarized the impact of test-time scaling on temporal reasoning capabilities. Additionally, we release TIME-Lite, a human-annotated subset to foster future research and standardized evaluation in temporal reasoning. The code is available at this https URL , and the dataset is available at this https URL . 

---
# IDEAL: Data Equilibrium Adaptation for Multi-Capability Language Model Alignment 

**Authors**: Chenlin Ming, Chendi Qu, Mengzhang Cai, Qizhi Pei, Zhuoshi Pan, Yu Li, Xiaoming Duan, Lijun Wu, Conghui He  

**Link**: [PDF](https://arxiv.org/pdf/2505.12762)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance through Supervised Fine-tuning (SFT) on diverse instructional datasets. When training on multiple capabilities simultaneously, the mixture training dataset, governed by volumes of data from different domains, is a critical factor that directly impacts the final model's performance. Unlike many studies that focus on enhancing the quality of training datasets through data selection methods, few works explore the intricate relationship between the compositional quantity of mixture training datasets and the emergent capabilities of LLMs. Given the availability of a high-quality multi-domain training dataset, understanding the impact of data from each domain on the model's overall capabilities is crucial for preparing SFT data and training a well-balanced model that performs effectively across diverse domains. In this work, we introduce IDEAL, an innovative data equilibrium adaptation framework designed to effectively optimize volumes of data from different domains within mixture SFT datasets, thereby enhancing the model's alignment and performance across multiple capabilities. IDEAL employs a gradient-based approach to iteratively refine the training data distribution, dynamically adjusting the volumes of domain-specific data based on their impact on downstream task performance. By leveraging this adaptive mechanism, IDEAL ensures a balanced dataset composition, enabling the model to achieve robust generalization and consistent proficiency across diverse tasks. Experiments across different capabilities demonstrate that IDEAL outperforms conventional uniform data allocation strategies, achieving a comprehensive improvement of approximately 7% in multi-task evaluation scores. 

---
# Detection and Mitigation of Hallucination in Large Reasoning Models: A Mechanistic Perspective 

**Authors**: Zhongxiang Sun, Qipeng Wang, Haoyu Wang, Xiao Zhang, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12886)  

**Abstract**: Large Reasoning Models (LRMs) have shown impressive capabilities in multi-step reasoning tasks. However, alongside these successes, a more deceptive form of model error has emerged--Reasoning Hallucination--where logically coherent but factually incorrect reasoning traces lead to persuasive yet faulty conclusions. Unlike traditional hallucinations, these errors are embedded within structured reasoning, making them more difficult to detect and potentially more harmful. In this work, we investigate reasoning hallucinations from a mechanistic perspective. We propose the Reasoning Score, which quantifies the depth of reasoning by measuring the divergence between logits obtained from projecting late layers of LRMs to the vocabulary space, effectively distinguishing shallow pattern-matching from genuine deep reasoning. Using this score, we conduct an in-depth analysis on the ReTruthQA dataset and identify two key reasoning hallucination patterns: early-stage fluctuation in reasoning depth and incorrect backtracking to flawed prior steps. These insights motivate our Reasoning Hallucination Detection (RHD) framework, which achieves state-of-the-art performance across multiple domains. To mitigate reasoning hallucinations, we further introduce GRPO-R, an enhanced reinforcement learning algorithm that incorporates step-level deep reasoning rewards via potential-based shaping. Our theoretical analysis establishes stronger generalization guarantees, and experiments demonstrate improved reasoning quality and reduced hallucination rates. 

---
# Reasoning BO: Enhancing Bayesian Optimization with Long-Context Reasoning Power of LLMs 

**Authors**: Zhuo Yang, Lingli Ge, Dong Han, Tianfan Fu, Yuqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12833)  

**Abstract**: Many real-world scientific and industrial applications require the optimization of expensive black-box functions. Bayesian Optimization (BO) provides an effective framework for such problems. However, traditional BO methods are prone to get trapped in local optima and often lack interpretable insights. To address this issue, this paper designs Reasoning BO, a novel framework that leverages reasoning models to guide the sampling process in BO while incorporating multi-agent systems and knowledge graphs for online knowledge accumulation. By integrating the reasoning and contextual understanding capabilities of Large Language Models (LLMs), we can provide strong guidance to enhance the BO process. As the optimization progresses, Reasoning BO provides real-time sampling recommendations along with critical insights grounded in plausible scientific theories, aiding in the discovery of superior solutions within the search space. We systematically evaluate our approach across 10 diverse tasks encompassing synthetic mathematical functions and complex real-world applications. The framework demonstrates its capability to progressively refine sampling strategies through real-time insights and hypothesis evolution, effectively identifying higher-performing regions of the search space for focused exploration. This process highlights the powerful reasoning and context-learning abilities of LLMs in optimization scenarios. For example, in the Direct Arylation task, our method increased the yield to 60.7%, whereas traditional BO achieved only a 25.2% yield. Furthermore, our investigation reveals that smaller LLMs, when fine-tuned through reinforcement learning, can attain comparable performance to their larger counterparts. This enhanced reasoning capability paves the way for more efficient automated scientific experimentation while maintaining computational feasibility. 

---
# Bullying the Machine: How Personas Increase LLM Vulnerability 

**Authors**: Ziwei Xu, Udit Sanghi, Mohan Kankanhalli  

**Link**: [PDF](https://arxiv.org/pdf/2505.12692)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in interactions where they are prompted to adopt personas. This paper investigates whether such persona conditioning affects model safety under bullying, an adversarial manipulation that applies psychological pressures in order to force the victim to comply to the attacker. We introduce a simulation framework in which an attacker LLM engages a victim LLM using psychologically grounded bullying tactics, while the victim adopts personas aligned with the Big Five personality traits. Experiments using multiple open-source LLMs and a wide range of adversarial goals reveal that certain persona configurations -- such as weakened agreeableness or conscientiousness -- significantly increase victim's susceptibility to unsafe outputs. Bullying tactics involving emotional or sarcastic manipulation, such as gaslighting and ridicule, are particularly effective. These findings suggest that persona-driven interaction introduces a novel vector for safety risks in LLMs and highlight the need for persona-aware safety evaluation and alignment strategies. 

---
# FRAbench and GenEval: Scaling Fine-Grained Aspect Evaluation across Tasks, Modalities 

**Authors**: Shibo Hong, Jiahao Ying, Haiyuan Liang, Mengdi Zhang, Jun Kuang, Jiazheng Zhang, Yixin Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12795)  

**Abstract**: Evaluating the open-ended outputs of large language models (LLMs) has become a bottleneck as model capabilities, task diversity, and modality coverage rapidly expand. Existing "LLM-as-a-Judge" evaluators are typically narrow in a few tasks, aspects, or modalities, and easily suffer from low consistency. In this paper, we argue that explicit, fine-grained aspect specification is the key to both generalizability and objectivity in automated evaluation. To do so, we introduce a hierarchical aspect taxonomy spanning 112 aspects that unifies evaluation across four representative settings - Natural Language Generation, Image Understanding, Image Generation, and Interleaved Text-and-Image Generation. Building on this taxonomy, we create FRAbench, a benchmark comprising 60.4k pairwise samples with 325k aspect-level labels obtained from a combination of human and LLM annotations. FRAbench provides the first large-scale, multi-modal resource for training and meta-evaluating fine-grained LMM judges. Leveraging FRAbench, we develop GenEval, a fine-grained evaluator generalizable across tasks and modalities. Experiments show that GenEval (i) attains high agreement with GPT-4o and expert annotators, (ii) transfers robustly to unseen tasks and modalities, and (iii) reveals systematic weaknesses of current LMMs on evaluation. 

---
# mCLM: A Function-Infused and Synthesis-Friendly Modular Chemical Language Model 

**Authors**: Carl Edwards, Chi Han, Gawon Lee, Thao Nguyen, Bowen Jin, Chetan Kumar Prasad, Sara Szymkuć, Bartosz A. Grzybowski, Ying Diao, Jiawei Han, Ge Liu, Hao Peng, Martin D. Burke, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.12565)  

**Abstract**: Despite their ability to understand chemical knowledge and accurately generate sequential representations, large language models (LLMs) remain limited in their capacity to propose novel molecules with drug-like properties. In addition, the molecules that LLMs propose can often be challenging to make in the lab. To more effectively enable the discovery of functional small molecules, LLMs need to learn a molecular language. However, LLMs are currently limited by encoding molecules from atoms. In this paper, we argue that just like tokenizing texts into (sub-)word tokens instead of characters, molecules should be decomposed and reassembled at the level of functional building blocks, i.e., parts of molecules that bring unique functions and serve as effective building blocks for real-world automated laboratory synthesis. This motivates us to propose mCLM, a modular Chemical-Language Model tokenizing molecules into building blocks and learning a bilingual language model of both natural language descriptions of functions and molecule building blocks. By reasoning on such functional building blocks, mCLM guarantees to generate efficiently synthesizable molecules thanks to recent progress in block-based chemistry, while also improving the functions of molecules in a principled manner. In experiments on 430 FDA-approved drugs, we find mCLM capable of significantly improving 5 out of 6 chemical functions critical to determining drug potentials. More importantly, mCLM can reason on multiple functions and improve the FDA-rejected drugs (``fallen angels'') over multiple iterations to greatly improve their shortcomings. 

---
# Incentivizing Multimodal Reasoning in Large Models for Direct Robot Manipulation 

**Authors**: Weiliang Tang, Dong Jing, Jia-Hui Pan, Zhiwu Lu, Yun-Hui Liu, Li Erran Li, Mingyu Ding, Chi-Wing Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12744)  

**Abstract**: Recent Large Multimodal Models have demonstrated remarkable reasoning capabilities, especially in solving complex mathematical problems and realizing accurate spatial perception. Our key insight is that these emerging abilities can naturally extend to robotic manipulation by enabling LMMs to directly infer the next goal in language via reasoning, rather than relying on a separate action head. However, this paradigm meets two main challenges: i) How to make LMMs understand the spatial action space, and ii) How to fully exploit the reasoning capacity of LMMs in solving these tasks. To tackle the former challenge, we propose a novel task formulation, which inputs the current states of object parts and the gripper, and reformulates rotation by a new axis representation instead of traditional Euler angles. This representation is more compatible with spatial reasoning and easier to interpret within a unified language space. For the latter challenge, we design a pipeline to utilize cutting-edge LMMs to generate a small but high-quality reasoning dataset of multi-round dialogues that successfully solve manipulation tasks for supervised fine-tuning. Then, we perform reinforcement learning by trial-and-error interactions in simulation to further enhance the model's reasoning abilities for robotic manipulation. Our resulting reasoning model built upon a 7B backbone, named ReasonManip, demonstrates three notable advantages driven by its system-2 level reasoning capabilities: i) exceptional generalizability to out-of-distribution environments, objects, and tasks; ii) inherent sim-to-real transfer ability enabled by the unified language representation shared across domains; iii) transparent interpretability connecting high-level reasoning and low-level control. Extensive experiments demonstrate the effectiveness of the proposed paradigm and its potential to advance LMM-driven robotic manipulation. 

---
# MindOmni: Unleashing Reasoning Generation in Vision Language Models with RGPO 

**Authors**: Yicheng Xiao, Lin Song, Yukang Chen, Yingmin Luo, Yuxin Chen, Yukang Gan, Wei Huang, Xiu Li, Xiaojuan Qi, Ying Shan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13031)  

**Abstract**: Recent text-to-image systems face limitations in handling multimodal inputs and complex reasoning tasks. We introduce MindOmni, a unified multimodal large language model that addresses these challenges by incorporating reasoning generation through reinforcement learning. MindOmni leverages a three-phase training strategy: i) design of a unified vision language model with a decoder-only diffusion module, ii) supervised fine-tuning with Chain-of-Thought (CoT) instruction data, and iii) our proposed Reasoning Generation Policy Optimization (RGPO) algorithm, utilizing multimodal feedback to effectively guide policy updates. Experimental results demonstrate that MindOmni outperforms existing models, achieving impressive performance on both understanding and generation benchmarks, meanwhile showcasing advanced fine-grained reasoning generation capabilities, especially with mathematical reasoning instruction. All codes will be made public at \href{this https URL}{this https URL}. 

---
# Dense Communication between Language Models 

**Authors**: Shiguang Wu, Yaqing Wang, Quanming Yao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12741)  

**Abstract**: As higher-level intelligence emerges from the combination of modular components with lower-level intelligence, many works combines Large Language Models (LLMs) for collective intelligence. Such combination is achieved by building communications among LLMs. While current systems primarily facilitate such communication through natural language, this paper proposes a novel paradigm of direct dense vector communication between LLMs. Our approach eliminates the unnecessary embedding and de-embedding steps when LLM interact with another, enabling more efficient information transfer, fully differentiable optimization pathways, and exploration of capabilities beyond human heuristics. We use such stripped LLMs as vertexes and optimizable seq2seq modules as edges to construct LMNet, with similar structure as MLPs. By utilizing smaller pre-trained LLMs as vertexes, we train a LMNet that achieves comparable performance with LLMs in similar size with only less than 0.1% training cost. This offers a new perspective on scaling for general intelligence rather than training a monolithic LLM from scratch. Besides, the proposed method can be used for other applications, like customizing LLM with limited data, showing its versatility. 

---
# MARGE: Improving Math Reasoning for LLMs with Guided Exploration 

**Authors**: Jingyue Gao, Runji Lin, Keming Lu, Bowen Yu, Junyang Lin, Jianyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12500)  

**Abstract**: Large Language Models (LLMs) exhibit strong potential in mathematical reasoning, yet their effectiveness is often limited by a shortage of high-quality queries. This limitation necessitates scaling up computational responses through self-generated data, yet current methods struggle due to spurious correlated data caused by ineffective exploration across all reasoning stages. To address such challenge, we introduce \textbf{MARGE}: Improving \textbf{Ma}th \textbf{R}easoning with \textbf{G}uided \textbf{E}xploration, a novel method to address this issue and enhance mathematical reasoning through hit-guided exploration. MARGE systematically explores intermediate reasoning states derived from self-generated solutions, enabling adequate exploration and improved credit assignment throughout the reasoning process. Through extensive experiments across multiple backbone models and benchmarks, we demonstrate that MARGE significantly improves reasoning capabilities without requiring external annotations or training additional value models. Notably, MARGE improves both single-shot accuracy and exploration diversity, mitigating a common trade-off in alignment methods. These results demonstrate MARGE's effectiveness in enhancing mathematical reasoning capabilities and unlocking the potential of scaling self-generated training data. Our code and models are available at \href{this https URL}{this link}. 

---
# Reasoning-CV: Fine-tuning Powerful Reasoning LLMs for Knowledge-Assisted Claim Verification 

**Authors**: Zhi Zheng, Wee Sun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.12348)  

**Abstract**: Claim verification is essential in combating misinformation, and large language models (LLMs) have recently emerged in this area as powerful tools for assessing the veracity of claims using external knowledge. Existing LLM-based methods for claim verification typically adopt a Decompose-Then-Verify paradigm, which involves decomposing complex claims into several independent sub-claims and verifying each sub-claim separately. However, this paradigm often introduces errors during the claim decomposition process. To mitigate these errors, we propose to develop the Chain-of-Thought (CoT)-Verify paradigm, which leverages LLM reasoning methods to generate CoT-verification paths for the original complex claim without requiring decompositions into sub-claims and separate verification stages. The CoT-Verify paradigm allows us to propose a natural fine-tuning method called Reasoning-CV to enhance the verification capabilities in LLMs. Reasoning-CV includes a supervised fine-tuning (SFT) stage and a self-improvement direct preference optimization (DPO) stage. Utilizing only an 8B pre-trained LLM, Reasoning-CV demonstrates superior knowledge-assisted claim verification performances compared to existing Decompose-Then-Verify methods, as well as powerful black-box LLMs such as GPT-4o+CoT and o1-preview. Our code is available. 

---
# Enhancing User-Oriented Proactivity in Open-Domain Dialogues with Critic Guidance 

**Authors**: Yufeng Wang, Jinwu Hu, Ziteng Huang, Kunyang Lin, Zitian Zhang, Peihao Chen, Yu Hu, Qianyue Wang, Zhuliang Yu, Bin Sun, Xiaofen Xing, Qingfang Zheng, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12334)  

**Abstract**: Open-domain dialogue systems aim to generate natural and engaging conversations, providing significant practical value in real applications such as social robotics and personal assistants. The advent of large language models (LLMs) has greatly advanced this field by improving context understanding and conversational fluency. However, existing LLM-based dialogue systems often fall short in proactively understanding the user's chatting preferences and guiding conversations toward user-centered topics. This lack of user-oriented proactivity can lead users to feel unappreciated, reducing their satisfaction and willingness to continue the conversation in human-computer interactions. To address this issue, we propose a User-oriented Proactive Chatbot (UPC) to enhance the user-oriented proactivity. Specifically, we first construct a critic to evaluate this proactivity inspired by the LLM-as-a-judge strategy. Given the scarcity of high-quality training data, we then employ the critic to guide dialogues between the chatbot and user agents, generating a corpus with enhanced user-oriented proactivity. To ensure the diversity of the user backgrounds, we introduce the ISCO-800, a diverse user background dataset for constructing user agents. Moreover, considering the communication difficulty varies among users, we propose an iterative curriculum learning method that trains the chatbot from easy-to-communicate users to more challenging ones, thereby gradually enhancing its performance. Experiments demonstrate that our proposed training method is applicable to different LLMs, improving user-oriented proactivity and attractiveness in open-domain dialogues. 

---
# SEED-GRPO: Semantic Entropy Enhanced GRPO for Uncertainty-Aware Policy Optimization 

**Authors**: Minghan Chen, Guikun Chen, Wenguan Wang, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12346)  

**Abstract**: Large language models (LLMs) exhibit varying levels of confidence across input prompts (questions): some lead to consistent, semantically similar answers, while others yield diverse or contradictory outputs. This variation reflects LLM's uncertainty about the input prompt, a signal of how confidently the model understands a given problem. However, vanilla Group Relative Policy Optimization (GRPO) treats all prompts equally during policy updates, ignoring this important information about the model's knowledge boundaries. To address this limitation, we propose SEED-GRPO (Semantic Entropy EnhanceD GRPO), which explicitly measures LLMs' uncertainty of the input prompts semantic entropy. Semantic entropy measures the diversity of meaning in multiple generated answers given a prompt and uses this to modulate the magnitude of policy updates. This uncertainty-aware training mechanism enables dynamic adjustment of policy update magnitudes based on question uncertainty. It allows more conservative updates on high-uncertainty questions while maintaining the original learning signal on confident ones. Experimental results on five mathematical reasoning benchmarks (AIME24 56.7, AMC 68.7, MATH 83.4, Minerva 34.2, and OlympiadBench 48.0) demonstrate that SEED-GRPO achieves new state-of-the-art performance in average accuracy, validating the effectiveness of uncertainty-aware policy optimization. 

---
# Correspondence of high-dimensional emotion structures elicited by video clips between humans and Multimodal LLMs 

**Authors**: Haruka Asanuma, Naoko Koide-Majima, Ken Nakamura, Takato Horii, Shinji Nishimoto, Masafumi Oizumi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12746)  

**Abstract**: Recent studies have revealed that human emotions exhibit a high-dimensional, complex structure. A full capturing of this complexity requires new approaches, as conventional models that disregard high dimensionality risk overlooking key nuances of human emotions. Here, we examined the extent to which the latest generation of rapidly evolving Multimodal Large Language Models (MLLMs) capture these high-dimensional, intricate emotion structures, including capabilities and limitations. Specifically, we compared self-reported emotion ratings from participants watching videos with model-generated estimates (e.g., Gemini or GPT). We evaluated performance not only at the individual video level but also from emotion structures that account for inter-video relationships. At the level of simple correlation between emotion structures, our results demonstrated strong similarity between human and model-inferred emotion structures. To further explore whether the similarity between humans and models is at the signle item level or the coarse-categorical level, we applied Gromov Wasserstein Optimal Transport. We found that although performance was not necessarily high at the strict, single-item level, performance across video categories that elicit similar emotions was substantial, indicating that the model could infer human emotional experiences at the category level. Our results suggest that current state-of-the-art MLLMs broadly capture the complex high-dimensional emotion structures at the category level, as well as their apparent limitations in accurately capturing entire structures at the single-item level. 

---
# NeuroGen: Neural Network Parameter Generation via Large Language Models 

**Authors**: Jiaqi Wang, Yusen Zhang, Xi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12470)  

**Abstract**: Acquiring the parameters of neural networks (NNs) has been one of the most important problems in machine learning since the inception of NNs. Traditional approaches, such as backpropagation and forward-only optimization, acquire parameters via iterative data fitting to gradually optimize them. This paper aims to explore the feasibility of a new direction: acquiring NN parameters via large language model generation. We propose NeuroGen, a generalized and easy-to-implement two-stage approach for NN parameter generation conditioned on descriptions of the data, task, and network architecture. Stage one is Parameter Reference Knowledge Injection, where LLMs are pretrained on NN checkpoints to build foundational understanding of parameter space, whereas stage two is Context-Enhanced Instruction Tuning, enabling LLMs to adapt to specific tasks through enriched, task-aware prompts. Experimental results demonstrate that NeuroGen effectively generates usable NN parameters. Our findings highlight the feasibility of LLM-based NN parameter generation and suggest a promising new paradigm where LLMs and lightweight NNs can coexist synergistically 

---
# Beyond Single-Point Judgment: Distribution Alignment for LLM-as-a-Judge 

**Authors**: Luyu Chen, Zeyu Zhang, Haoran Tan, Quanyu Dai, Hao Yang, Zhenhua Dong, Xu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12301)  

**Abstract**: LLMs have emerged as powerful evaluators in the LLM-as-a-Judge paradigm, offering significant efficiency and flexibility compared to human judgments. However, previous methods primarily rely on single-point evaluations, overlooking the inherent diversity and uncertainty in human evaluations. This approach leads to information loss and decreases the reliability of evaluations. To address this limitation, we propose a novel training framework that explicitly aligns the LLM-generated judgment distribution with empirical human distributions. Specifically, we propose a distributional alignment objective based on KL divergence, combined with an auxiliary cross-entropy regularization to stabilize the training process. Furthermore, considering that empirical distributions may derive from limited human annotations, we incorporate adversarial training to enhance model robustness against distribution perturbations. Extensive experiments across various LLM backbones and evaluation tasks demonstrate that our framework significantly outperforms existing closed-source LLMs and conventional single-point alignment methods, with improved alignment quality, evaluation accuracy, and robustness. 

---
# Mitigating Content Effects on Reasoning in Language Models through Fine-Grained Activation Steering 

**Authors**: Marco Valentino, Geonhee Kim, Dhairya Dalal, Zhixue Zhao, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2505.12189)  

**Abstract**: Large language models (LLMs) frequently demonstrate reasoning limitations, often conflating content plausibility (i.e., material inference) with logical validity (i.e., formal inference). This can result in biased inferences, where plausible arguments are incorrectly deemed logically valid or vice versa. Mitigating this limitation is critical, as it undermines the trustworthiness and generalizability of LLMs in applications that demand rigorous logical consistency. This paper investigates the problem of mitigating content biases on formal reasoning through activation steering. Specifically, we curate a controlled syllogistic reasoning dataset to disentangle formal validity from content plausibility. After localising the layers responsible for formal and material inference, we investigate contrastive activation steering methods for test-time interventions. An extensive empirical analysis on different LLMs reveals that contrastive steering consistently supports linear control over content biases. However, we observe that a static approach is insufficient for improving all the tested models. We then leverage the possibility to control content effects by dynamically determining the value of the steering parameters via fine-grained conditional methods. We found that conditional steering is effective on unresponsive models, achieving up to 15% absolute improvement in formal reasoning accuracy with a newly introduced kNN-based method (K-CAST). Finally, additional experiments reveal that steering for content effects is robust to prompt variations, incurs minimal side effects on language modeling capabilities, and can partially generalize to out-of-distribution reasoning tasks. Practically, this paper demonstrates that activation-level interventions can offer a scalable strategy for enhancing the robustness of LLMs, contributing towards more systematic and unbiased formal reasoning. 

---
# LLM-BABYBENCH: Understanding and Evaluating Grounded Planning and Reasoning in LLMs 

**Authors**: Omar Choukrani, Idriss Malek, Daniil Orel, Zhuohan Xie, Zangir Iklassov, Martin Takáč, Salem Lahlou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12135)  

**Abstract**: Assessing the capacity of Large Language Models (LLMs) to plan and reason within the constraints of interactive environments is crucial for developing capable AI agents. We introduce $\textbf{LLM-BabyBench}$, a new benchmark suite designed specifically for this purpose. Built upon a textual adaptation of the procedurally generated BabyAI grid world, this suite evaluates LLMs on three fundamental aspects of grounded intelligence: (1) predicting the consequences of actions on the environment state ($\textbf{Predict}$ task), (2) generating sequences of low-level actions to achieve specified objectives ($\textbf{Plan}$ task), and (3) decomposing high-level instructions into coherent subgoal sequences ($\textbf{Decompose}$ task). We detail the methodology for generating the three corresponding datasets ($\texttt{LLM-BabyBench-Predict}$, $\texttt{-Plan}$, $\texttt{-Decompose}$) by extracting structured information from an expert agent operating within the text-based environment. Furthermore, we provide a standardized evaluation harness and metrics, including environment interaction for validating generated plans, to facilitate reproducible assessment of diverse LLMs. Initial baseline results highlight the challenges posed by these grounded reasoning tasks. The benchmark suite, datasets, data generation code, and evaluation code are made publicly available ($\href{this https URL}{\text{GitHub}}$, $\href{this https URL}{\text{HuggingFace}}$). 

---
# ALAS: A Stateful Multi-LLM Agent Framework for Disruption-Aware Planning 

**Authors**: Edward Y. Chang, Longling Geng  

**Link**: [PDF](https://arxiv.org/pdf/2505.12501)  

**Abstract**: Large language models (LLMs) excel at rapid generation of text and multimodal content, yet they falter on transaction-style planning that demands ACID-like guarantees and real-time disruption recovery. We present Adaptive LLM Agent System (ALAS), a framework that tackles four fundamental LLM deficits: (i) absence of self-verification, (ii) context erosion, (iii) next-token myopia, and (iv) lack of persistent state. ALAS decomposes each plan into role-specialized agents, equips them with automatic state tracking, and coordinates them through a lightweight protocol. When disruptions arise, agents apply history-aware local compensation, avoiding costly global replanning and containing cascade effects. On real-world, large-scale job-shop scheduling benchmarks, ALAS sets new best results for static sequential planning and excels in dynamic reactive scenarios with unexpected disruptions. These gains show that principled modularization plus targeted compensation can unlock scalable and resilient planning with LLMs. 

---
# SOCIA: An End-to-End Agentic Framework for Automated Cyber-Physical-Social Simulator Generation 

**Authors**: Yuncheng Hua, Ji Miao, Mehdi Jafari, Jianxiang Xie, Hao Xue, Flora D. Salim  

**Link**: [PDF](https://arxiv.org/pdf/2505.12006)  

**Abstract**: This paper introduces SOCIA (Simulation Orchestration for Cyber-physical-social Intelligence and Agents), a novel end-to-end framework leveraging Large Language Model (LLM)-based multi-agent systems to automate the generation of high-fidelity Cyber-Physical-Social (CPS) simulators. Addressing the challenges of labor-intensive manual simulator development and complex data calibration, SOCIA integrates a centralized orchestration manager that coordinates specialized agents for tasks including data comprehension, code generation, simulation execution, and iterative evaluation-feedback loops. Through empirical evaluations across diverse CPS tasks, such as mask adoption behavior simulation (social), personal mobility generation (physical), and user modeling (cyber), SOCIA demonstrates its ability to produce high-fidelity, scalable simulations with reduced human intervention. These results highlight SOCIA's potential to offer a scalable solution for studying complex CPS phenomena 

---
# Solve-Detect-Verify: Inference-Time Scaling with Flexible Generative Verifier 

**Authors**: Jianyuan Zhong, Zeju Li, Zhijian Xu, Xiangyu Wen, Kezhi Li, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11966)  

**Abstract**: Large Language Model (LLM) reasoning for complex tasks inherently involves a trade-off between solution accuracy and computational efficiency. The subsequent step of verification, while intended to improve performance, further complicates this landscape by introducing its own challenging trade-off: sophisticated Generative Reward Models (GenRMs) can be computationally prohibitive if naively integrated with LLMs at test-time, while simpler, faster methods may lack reliability. To overcome these challenges, we introduce FlexiVe, a novel generative verifier that flexibly balances computational resources between rapid, reliable fast thinking and meticulous slow thinking using a Flexible Allocation of Verification Budget strategy. We further propose the Solve-Detect-Verify pipeline, an efficient inference-time scaling framework that intelligently integrates FlexiVe, proactively identifying solution completion points to trigger targeted verification and provide focused solver feedback. Experiments show FlexiVe achieves superior accuracy in pinpointing errors within reasoning traces on ProcessBench. Furthermore, on challenging mathematical reasoning benchmarks (AIME 2024, AIME 2025, and CNMO), our full approach outperforms baselines like self-consistency in reasoning accuracy and inference efficiency. Our system offers a scalable and effective solution to enhance LLM reasoning at test time. 

---
# LifelongAgentBench: Evaluating LLM Agents as Lifelong Learners 

**Authors**: Junhao Zheng, Xidi Cai, Qiuke Li, Duzhen Zhang, ZhongZhi Li, Yingying Zhang, Le Song, Qianli Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.11942)  

**Abstract**: Lifelong learning is essential for intelligent agents operating in dynamic environments. Current large language model (LLM)-based agents, however, remain stateless and unable to accumulate or transfer knowledge over time. Existing benchmarks treat agents as static systems and fail to evaluate lifelong learning capabilities. We present LifelongAgentBench, the first unified benchmark designed to systematically assess the lifelong learning ability of LLM agents. It provides skill-grounded, interdependent tasks across three interactive environments, Database, Operating System, and Knowledge Graph, with automatic label verification, reproducibility, and modular extensibility. Extensive experiments reveal that conventional experience replay has limited effectiveness for LLM agents due to irrelevant information and context length constraints. We further introduce a group self-consistency mechanism that significantly improves lifelong learning performance. We hope LifelongAgentBench will advance the development of adaptive, memory-capable LLM agents. 

---
# Fair-PP: A Synthetic Dataset for Aligning LLM with Personalized Preferences of Social Equity 

**Authors**: Qi Zhou, Jie Zhang, Dongxia Wang, Qiang Liu, Tianlin Li, Jin Song Dong, Wenhai Wang, Qing Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11861)  

**Abstract**: Human preference plays a crucial role in the refinement of large language models (LLMs). However, collecting human preference feedback is costly and most existing datasets neglect the correlation between personalization and preferences. To address this issue, we introduce Fair-PP, a synthetic dataset of personalized preferences targeting social equity, derived from real-world social survey data, which includes 28 social groups, 98 equity topics, and 5 personal preference dimensions. Leveraging GPT-4o-mini, we engage in role-playing based on seven representative persona portrayals guided by existing social survey data, yielding a total of 238,623 preference records. Through Fair-PP, we also contribute (i) An automated framework for generating preference data, along with a more fine-grained dataset of personalized preferences; (ii) analysis of the positioning of the existing mainstream LLMs across five major global regions within the personalized preference space; and (iii) a sample reweighting method for personalized preference alignment, enabling alignment with a target persona while maximizing the divergence from other personas. Empirical experiments show our method outperforms the baselines. 

---
# Evaluating the Logical Reasoning Abilities of Large Reasoning Models 

**Authors**: Hanmeng Liu, Yiran Ding, Zhizhang Fu, Chaoli Zhang, Xiaozhang Liu, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11854)  

**Abstract**: Large reasoning models, often post-trained on long chain-of-thought (long CoT) data with reinforcement learning, achieve state-of-the-art performance on mathematical, coding, and domain-specific reasoning benchmarks. However, their logical reasoning capabilities - fundamental to human cognition and independent of domain knowledge - remain understudied. To address this gap, we introduce LogiEval, a holistic benchmark for evaluating logical reasoning in large reasoning models. LogiEval spans diverse reasoning types (deductive, inductive, analogical, and abductive) and task formats (e.g., logical sequence, argument analysis), sourced from high-quality human examinations (e.g., LSAT, GMAT). Our experiments demonstrate that modern reasoning models excel at 4-choice argument analysis problems and analogical reasoning, surpassing human performance, yet exhibit uneven capabilities across reasoning types and formats, highlighting limitations in their generalization. Our analysis reveals that human performance does not mirror model failure distributions. To foster further research, we curate LogiEval-Hard, a challenging subset identified through a novel screening paradigm where small-model failures (Qwen3-30B-A3B) reliably predict difficulties for larger models. Modern models show striking, consistent failures on LogiEval-Hard. This demonstrates that fundamental reasoning bottlenecks persist across model scales, and establishes LogiEval-Hard as both a diagnostic tool and a rigorous testbed for advancing logical reasoning in LLMs. 

---
# On the Eligibility of LLMs for Counterfactual Reasoning: A Decompositional Study 

**Authors**: Shuai Yang, Qi Yang, Luoxi Tang, Jeremy Blackburn, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2505.11839)  

**Abstract**: Counterfactual reasoning has emerged as a crucial technique for generalizing the reasoning capabilities of large language models (LLMs). By generating and analyzing counterfactual scenarios, researchers can assess the adaptability and reliability of model decision-making. Although prior work has shown that LLMs often struggle with counterfactual reasoning, it remains unclear which factors most significantly impede their performance across different tasks and modalities. In this paper, we propose a decompositional strategy that breaks down the counterfactual generation from causality construction to the reasoning over counterfactual interventions. To support decompositional analysis, we investigate 11 datasets spanning diverse tasks, including natural language understanding, mathematics, programming, and vision-language tasks. Through extensive evaluations, we characterize LLM behavior across each decompositional stage and identify how modality type and intermediate reasoning influence performance. By establishing a structured framework for analyzing counterfactual reasoning, this work contributes to the development of more reliable LLM-based reasoning systems and informs future elicitation strategies. 

---
# ToLeaP: Rethinking Development of Tool Learning with Large Language Models 

**Authors**: Haotian Chen, Zijun Song, Boye Niu, Ke Zhang, Litu Ou, Yaxi Lu, Zhong Zhang, Xin Cong, Yankai Lin, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.11833)  

**Abstract**: Tool learning, which enables large language models (LLMs) to utilize external tools effectively, has garnered increasing attention for its potential to revolutionize productivity across industries. Despite rapid development in tool learning, key challenges and opportunities remain understudied, limiting deeper insights and future advancements. In this paper, we investigate the tool learning ability of 41 prevalent LLMs by reproducing 33 benchmarks and enabling one-click evaluation for seven of them, forming a Tool Learning Platform named ToLeaP. We also collect 21 out of 33 potential training datasets to facilitate future exploration. After analyzing over 3,000 bad cases of 41 LLMs based on ToLeaP, we identify four main critical challenges: (1) benchmark limitations induce both the neglect and lack of (2) autonomous learning, (3) generalization, and (4) long-horizon task-solving capabilities of LLMs. To aid future advancements, we take a step further toward exploring potential directions, namely (1) real-world benchmark construction, (2) compatibility-aware autonomous learning, (3) rationale learning by thinking, and (4) identifying and recalling key clues. The preliminary experiments demonstrate their effectiveness, highlighting the need for further research and exploration. 

---
# ChatHTN: Interleaving Approximate (LLM) and Symbolic HTN Planning 

**Authors**: Hector Munoz-Avila, David W. Aha, Paola Rizzo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11814)  

**Abstract**: We introduce ChatHTN, a Hierarchical Task Network (HTN) planner that combines symbolic HTN planning techniques with queries to ChatGPT to approximate solutions in the form of task decompositions. The resulting hierarchies interleave task decompositions generated by symbolic HTN planning with those generated by ChatGPT. Despite the approximate nature of the results generates by ChatGPT, ChatHTN is provably sound; any plan it generates correctly achieves the input tasks. We demonstrate this property with an open-source implementation of our system. 

---
# LLM-based Automated Theorem Proving Hinges on Scalable Synthetic Data Generation 

**Authors**: Junyu Lai, Jiakun Zhang, Shuo Xu, Taolue Chen, Zihang Wang, Yao Yang, Jiarui Zhang, Chun Cao, Jingwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12031)  

**Abstract**: Recent advancements in large language models (LLMs) have sparked considerable interest in automated theorem proving and a prominent line of research integrates stepwise LLM-based provers into tree search. In this paper, we introduce a novel proof-state exploration approach for training data synthesis, designed to produce diverse tactics across a wide range of intermediate proof states, thereby facilitating effective one-shot fine-tuning of LLM as the policy model. We also propose an adaptive beam size strategy, which effectively takes advantage of our data synthesis method and achieves a trade-off between exploration and exploitation during tree search. Evaluations on the MiniF2F and ProofNet benchmarks demonstrate that our method outperforms strong baselines under the stringent Pass@1 metric, attaining an average pass rate of $60.74\%$ on MiniF2F and $21.18\%$ on ProofNet. These results underscore the impact of large-scale synthetic data in advancing automated theorem proving. 

---
# REMOR: Automated Peer Review Generation with LLM Reasoning and Multi-Objective Reinforcement Learning 

**Authors**: Pawin Taechoyotin, Daniel Acuna  

**Link**: [PDF](https://arxiv.org/pdf/2505.11718)  

**Abstract**: AI-based peer review systems tend to produce shallow and overpraising suggestions compared to human feedback. Here, we evaluate how well a reasoning LLM trained with multi-objective reinforcement learning (REMOR) can overcome these limitations. We start by designing a multi-aspect reward function that aligns with human evaluation of reviews. The aspects are related to the review itself (e.g., criticisms, novelty) and the relationship between the review and the manuscript (i.e., relevance). First, we perform supervised fine-tuning of DeepSeek-R1-Distill-Qwen-7B using LoRA on PeerRT, a new dataset of high-quality top AI conference reviews enriched with reasoning traces. We then apply Group Relative Policy Optimization (GRPO) to train two models: REMOR-H (with the human-aligned reward) and REMOR-U (with a uniform reward). Interestingly, the human-aligned reward penalizes aspects typically associated with strong reviews, leading REMOR-U to produce qualitatively more substantive feedback. Our results show that REMOR-U and REMOR-H achieve more than twice the average rewards of human reviews, non-reasoning state-of-the-art agentic multi-modal AI review systems, and general commercial LLM baselines. We found that while the best AI and human reviews are comparable in quality, REMOR avoids the long tail of low-quality human reviews. We discuss how reasoning is key to achieving these improvements and release the Human-aligned Peer Review Reward (HPRR) function, the Peer Review Reasoning-enriched Traces (PeerRT) dataset, and the REMOR models, which we believe can help spur progress in the area. 

---
# Rethinking Optimal Verification Granularity for Compute-Efficient Test-Time Scaling 

**Authors**: Hao Mark Chen, Guanxi Lu, Yasuyuki Okoshi, Zhiwen Mo, Masato Motomura, Hongxiang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11730)  

**Abstract**: Test-time scaling (TTS) has proven effective in enhancing the reasoning capabilities of large language models (LLMs). Verification plays a key role in TTS, simultaneously influencing (1) reasoning performance and (2) compute efficiency, due to the quality and computational cost of verification. In this work, we challenge the conventional paradigms of verification, and make the first attempt toward systematically investigating the impact of verification granularity-that is, how frequently the verifier is invoked during generation, beyond verifying only the final output or individual generation steps. To this end, we introduce Variable Granularity Search (VG-Search), a unified algorithm that generalizes beam search and Best-of-N sampling via a tunable granularity parameter g. Extensive experiments with VG-Search under varying compute budgets, generator-verifier configurations, and task attributes reveal that dynamically selecting g can improve the compute efficiency and scaling behavior. Building on these findings, we propose adaptive VG-Search strategies that achieve accuracy gains of up to 3.1\% over Beam Search and 3.6\% over Best-of-N, while reducing FLOPs by over 52\%. We will open-source the code to support future research. 

---
# Benchmarking Spatiotemporal Reasoning in LLMs and Reasoning Models: Capabilities and Challenges 

**Authors**: Pengrui Quan, Brian Wang, Kang Yang, Liying Han, Mani Srivastava  

**Link**: [PDF](https://arxiv.org/pdf/2505.11618)  

**Abstract**: Spatiotemporal reasoning plays a key role in Cyber-Physical Systems (CPS). Despite advances in Large Language Models (LLMs) and Large Reasoning Models (LRMs), their capacity to reason about complex spatiotemporal signals remains underexplored. This paper proposes a hierarchical SpatioTemporal reAsoning benchmaRK, STARK, to systematically evaluate LLMs across three levels of reasoning complexity: state estimation (e.g., predicting field variables, localizing and tracking events in space and time), spatiotemporal reasoning over states (e.g., inferring spatial-temporal relationships), and world-knowledge-aware reasoning that integrates contextual and domain knowledge (e.g., intent prediction, landmark-aware navigation). We curate 26 distinct spatiotemporal tasks with diverse sensor modalities, comprising 14,552 challenges where models answer directly or by Python Code Interpreter. Evaluating 3 LRMs and 8 LLMs, we find LLMs achieve limited success in tasks requiring geometric reasoning (e.g., multilateration or triangulation), particularly as complexity increases. Surprisingly, LRMs show robust performance across tasks with various levels of difficulty, often competing or surpassing traditional first-principle-based methods. Our results show that in reasoning tasks requiring world knowledge, the performance gap between LLMs and LRMs narrows, with some LLMs even surpassing LRMs. However, the LRM o3 model continues to achieve leading performance across all evaluated tasks, a result attributed primarily to the larger size of the reasoning models. STARK motivates future innovations in model architectures and reasoning paradigms for intelligent CPS by providing a structured framework to identify limitations in the spatiotemporal reasoning of LLMs and LRMs. 

---
# FLOW-BENCH: Towards Conversational Generation of Enterprise Workflows 

**Authors**: Evelyn Duesterwald, Siyu Huo, Vatche Isahagian, K.R. Jayaram, Ritesh Kumar, Vinod Muthusamy, Punleuk Oum, Debashish Saha, Gegi Thomas, Praveen Venkateswaran  

**Link**: [PDF](https://arxiv.org/pdf/2505.11646)  

**Abstract**: Business process automation (BPA) that leverages Large Language Models (LLMs) to convert natural language (NL) instructions into structured business process artifacts is becoming a hot research topic. This paper makes two technical contributions -- (i) FLOW-BENCH, a high quality dataset of paired natural language instructions and structured business process definitions to evaluate NL-based BPA tools, and support bourgeoning research in this area, and (ii) FLOW-GEN, our approach to utilize LLMs to translate natural language into an intermediate representation with Python syntax that facilitates final conversion into widely adopted business process definition languages, such as BPMN and DMN. We bootstrap FLOW-BENCH by demonstrating how it can be used to evaluate the components of FLOW-GEN across eight LLMs of varying sizes. We hope that FLOW-GEN and FLOW-BENCH catalyze further research in BPA making it more accessible to novice and expert users. 

---
# Using Reinforcement Learning to Train Large Language Models to Explain Human Decisions 

**Authors**: Jian-Qiao Zhu, Hanbo Xie, Dilip Arumugam, Robert C. Wilson, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.11614)  

**Abstract**: A central goal of cognitive modeling is to develop models that not only predict human behavior but also provide insight into the underlying cognitive mechanisms. While neural network models trained on large-scale behavioral data often achieve strong predictive performance, they typically fall short in offering interpretable explanations of the cognitive processes they capture. In this work, we explore the potential of pretrained large language models (LLMs) to serve as dual-purpose cognitive models--capable of both accurate prediction and interpretable explanation in natural language. Specifically, we employ reinforcement learning with outcome-based rewards to guide LLMs toward generating explicit reasoning traces for explaining human risky choices. Our findings demonstrate that this approach produces high-quality explanations alongside strong quantitative predictions of human decisions. 

---
# Heart2Mind: Human-Centered Contestable Psychiatric Disorder Diagnosis System using Wearable ECG Monitors 

**Authors**: Hung Nguyen, Alireza Rahimi, Veronica Whitford, Hélène Fournier, Irina Kondratova, René Richard, Hung Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.11612)  

**Abstract**: Psychiatric disorders affect millions globally, yet their diagnosis faces significant challenges in clinical practice due to subjective assessments and accessibility concerns, leading to potential delays in treatment. To help address this issue, we present Heart2Mind, a human-centered contestable psychiatric disorder diagnosis system using wearable electrocardiogram (ECG) monitors. Our approach leverages cardiac biomarkers, particularly heart rate variability (HRV) and R-R intervals (RRI) time series, as objective indicators of autonomic dysfunction in psychiatric conditions. The system comprises three key components: (1) a Cardiac Monitoring Interface (CMI) for real-time data acquisition from Polar H9/H10 devices; (2) a Multi-Scale Temporal-Frequency Transformer (MSTFT) that processes RRI time series through integrated time-frequency domain analysis; (3) a Contestable Diagnosis Interface (CDI) combining Self-Adversarial Explanations (SAEs) with contestable Large Language Models (LLMs). Our MSTFT achieves 91.7% accuracy on the HRV-ACC dataset using leave-one-out cross-validation, outperforming state-of-the-art methods. SAEs successfully detect inconsistencies in model predictions by comparing attention-based and gradient-based explanations, while LLMs enable clinicians to validate correct predictions and contest erroneous ones. This work demonstrates the feasibility of combining wearable technology with Explainable Artificial Intelligence (XAI) and contestable LLMs to create a transparent, contestable system for psychiatric diagnosis that maintains clinical oversight while leveraging advanced AI capabilities. Our implementation is publicly available at: this https URL. 

---
# Solver-Informed RL: Grounding Large Language Models for Authentic Optimization Modeling 

**Authors**: Yitian Chen, Jingfan Xia, Siyu Shao, Dongdong Ge, Yinyu Ye  

**Link**: [PDF](https://arxiv.org/pdf/2505.11792)  

**Abstract**: Optimization modeling is fundamental to decision-making across diverse this http URL progress in automating optimization formulation from natural language descriptions, Large Language Models (LLMs) often struggle to generate formally correct and usable models due to hallucinations, posing a challenge for reliable automation. Inspired by the success of Reinforcement Learning (RL) in enhancing Large Reasoning Models, we present Solver-Informed Reinforcement Learning (SIRL).This novel framework leverages external optimization solvers as verifiable reward mechanisms to significantly improve the authenticity of LLMs for optimization this http URL as precise verifiers, these solvers automatically assess the executable code and the instance-level mathematical model represented by the associated LP file, yielding precise and comprehensive feedback signals -- including syntax, feasibility, and solution quality that directly inform the RL process. This automated verification process, powered by classic optimization solvers, also underpins our instance-enhanced self-consistency method to synthesize high-quality training data. Extensive experiments on diverse public benchmarks demonstrate that SIRL achieves state-of-the-art performance, substantially outperforming existing methods in generating accurate and executable optimization models. 

---
# CIE: Controlling Language Model Text Generations Using Continuous Signals 

**Authors**: Vinay Samuel, Harshita Diddee, Yiming Zhang, Daphne Ippolito  

**Link**: [PDF](https://arxiv.org/pdf/2505.13448)  

**Abstract**: Aligning language models with user intent is becoming increasingly relevant to enhance user experience. This calls for designing methods that can allow users to control the properties of the language that LMs generate. For example, controlling the length of the generation, the complexity of the language that gets chosen, the sentiment, tone, etc. Most existing work attempts to integrate users' control by conditioning LM generations on natural language prompts or discrete control signals, which are often brittle and hard to scale. In this work, we are interested in \textit{continuous} control signals, ones that exist along a spectrum that can't easily be captured in a natural language prompt or via existing techniques in conditional generation. Through a case study in controlling the precise response-length of generations produced by LMs, we demonstrate how after fine-tuning, behaviors of language models can be controlled via continuous signals -- as vectors that are interpolated between a "low" and a "high" token embedding. Our method more reliably exerts response-length control than in-context learning methods or fine-tuning methods that represent the control signal as a discrete signal. Our full open-sourced code and datasets are available at this https URL. 

---
# Optimizing Anytime Reasoning via Budget Relative Policy Optimization 

**Authors**: Penghui Qi, Zichen Liu, Tianyu Pang, Chao Du, Wee Sun Lee, Min Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.13438)  

**Abstract**: Scaling test-time compute is crucial for enhancing the reasoning capabilities of large language models (LLMs). Existing approaches typically employ reinforcement learning (RL) to maximize a verifiable reward obtained at the end of reasoning traces. However, such methods optimize only the final performance under a large and fixed token budget, which hinders efficiency in both training and deployment. In this work, we present a novel framework, AnytimeReasoner, to optimize anytime reasoning performance, which aims to improve token efficiency and the flexibility of reasoning under varying token budget constraints. To achieve this, we truncate the complete thinking process to fit within sampled token budgets from a prior distribution, compelling the model to summarize the optimal answer for each truncated thinking for verification. This introduces verifiable dense rewards into the reasoning process, facilitating more effective credit assignment in RL optimization. We then optimize the thinking and summary policies in a decoupled manner to maximize the cumulative reward. Additionally, we introduce a novel variance reduction technique, Budget Relative Policy Optimization (BRPO), to enhance the robustness and efficiency of the learning process when reinforcing the thinking policy. Empirical results in mathematical reasoning tasks demonstrate that our method consistently outperforms GRPO across all thinking budgets under various prior distributions, enhancing both training and token efficiency. 

---
# AdaptThink: Reasoning Models Can Learn When to Think 

**Authors**: Jiajie Zhang, Nianyi Lin, Lei Hou, Ling Feng, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.13417)  

**Abstract**: Recently, large reasoning models have achieved impressive performance on various tasks by employing human-like deep thinking. However, the lengthy thinking process substantially increases inference overhead, making efficiency a critical bottleneck. In this work, we first demonstrate that NoThinking, which prompts the reasoning model to skip thinking and directly generate the final solution, is a better choice for relatively simple tasks in terms of both performance and efficiency. Motivated by this, we propose AdaptThink, a novel RL algorithm to teach reasoning models to choose the optimal thinking mode adaptively based on problem difficulty. Specifically, AdaptThink features two core components: (1) a constrained optimization objective that encourages the model to choose NoThinking while maintaining the overall performance; (2) an importance sampling strategy that balances Thinking and NoThinking samples during on-policy training, thereby enabling cold start and allowing the model to explore and exploit both thinking modes throughout the training process. Our experiments indicate that AdaptThink significantly reduces the inference costs while further enhancing performance. Notably, on three math datasets, AdaptThink reduces the average response length of DeepSeek-R1-Distill-Qwen-1.5B by 53% and improves its accuracy by 2.4%, highlighting the promise of adaptive thinking-mode selection for optimizing the balance between reasoning quality and efficiency. Our codes and models are available at this https URL. 

---
# DMN-Guided Prompting: A Low-Code Framework for Controlling LLM Behavior 

**Authors**: Shaghayegh Abedi, Amin Jalali  

**Link**: [PDF](https://arxiv.org/pdf/2505.11701)  

**Abstract**: Large Language Models (LLMs) have shown considerable potential in automating decision logic within knowledge-intensive processes. However, their effectiveness largely depends on the strategy and quality of prompting. Since decision logic is typically embedded in prompts, it becomes challenging for end users to modify or refine it. Decision Model and Notation (DMN) offers a standardized graphical approach for defining decision logic in a structured, user-friendly manner. This paper introduces a DMN-guided prompting framework that breaks down complex decision logic into smaller, manageable components, guiding LLMs through structured decision pathways. We implemented the framework in a graduate-level course where students submitted assignments. The assignments and DMN models representing feedback instructions served as inputs to our framework. The instructor evaluated the generated feedback and labeled it for performance assessment. Our approach demonstrated promising results, outperforming chain-of-thought (CoT) prompting. Students also responded positively to the generated feedback, reporting high levels of perceived usefulness in a survey based on the Technology Acceptance Model. 

---
# J4R: Learning to Judge with Equivalent Initial State Group Relative Preference Optimization 

**Authors**: Austin Xu, Yilun Zhou, Xuan-Phi Nguyen, Caiming Xiong, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2505.13346)  

**Abstract**: To keep pace with the increasing pace of large language models (LLM) development, model output evaluation has transitioned away from time-consuming human evaluation to automatic evaluation, where LLMs themselves are tasked with assessing and critiquing other model outputs. LLM-as-judge models are a class of generative evaluators that excel in evaluating relatively simple domains, like chat quality, but struggle in reasoning intensive domains where model responses contain more substantive and challenging content. To remedy existing judge shortcomings, we explore training judges with reinforcement learning (RL). We make three key contributions: (1) We propose the Equivalent Initial State Group Relative Policy Optimization (EIS-GRPO) algorithm, which allows us to train our judge to be robust to positional biases that arise in more complex evaluation settings. (2) We introduce ReasoningJudgeBench, a benchmark that evaluates judges in diverse reasoning settings not covered by prior work. (3) We train Judge for Reasoning (J4R), a 7B judge trained with EIS-GRPO that outperforms GPT-4o and the next best small judge by 6.7% and 9%, matching or exceeding the performance of larger GRPO-trained judges on both JudgeBench and ReasoningJudgeBench. 

---
# Contextual Paralinguistic Data Creation for Multi-Modal Speech-LLM: Data Condensation and Spoken QA Generation 

**Authors**: Qiongqiong Wang, Hardik B. Sailor, Tianchi Liu, Ai Ti Aw  

**Link**: [PDF](https://arxiv.org/pdf/2505.13338)  

**Abstract**: Current speech-LLMs exhibit limited capability in contextual reasoning alongside paralinguistic understanding, primarily due to the lack of Question-Answer (QA) datasets that cover both aspects. We propose a novel framework for dataset generation from in-the-wild speech data, that integrates contextual reasoning with paralinguistic information. It consists of a pseudo paralinguistic label-based data condensation of in-the-wild speech and LLM-based Contextual Paralinguistic QA (CPQA) generation. The effectiveness is validated by a strong correlation in evaluations of the Qwen2-Audio-7B-Instruct model on a dataset created by our framework and human-generated CPQA dataset. The results also reveal the speech-LLM's limitations in handling empathetic reasoning tasks, highlighting the need for such datasets and more robust models. The proposed framework is first of its kind and has potential in training more robust speech-LLMs with paralinguistic reasoning capabilities. 

---
# LLM Agents Are Hypersensitive to Nudges 

**Authors**: Manuel Cherep, Pattie Maes, Nikhil Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11584)  

**Abstract**: LLMs are being set loose in complex, real-world environments involving sequential decision-making and tool use. Often, this involves making choices on behalf of human users. However, not much is known about the distribution of such choices, and how susceptible they are to different choice architectures. We perform a case study with a few such LLM models on a multi-attribute tabular decision-making problem, under canonical nudges such as the default option, suggestions, and information highlighting, as well as additional prompting strategies. We show that, despite superficial similarities to human choice distributions, such models differ in subtle but important ways. First, they show much higher susceptibility to the nudges. Second, they diverge in points earned, being affected by factors like the idiosyncrasy of available prizes. Third, they diverge in information acquisition strategies: e.g. incurring substantial cost to reveal too much information, or selecting without revealing any. Moreover, we show that simple prompt strategies like zero-shot chain of thought (CoT) can shift the choice distribution, and few-shot prompting with human data can induce greater alignment. Yet, none of these methods resolve the sensitivity of these models to nudges. Finally, we show how optimal nudges optimized with a human resource-rational model can similarly increase LLM performance for some models. All these findings suggest that behavioral tests are needed before deploying models as agents or assistants acting on behalf of users in complex environments. 

---
# VeriReason: Reinforcement Learning with Testbench Feedback for Reasoning-Enhanced Verilog Generation 

**Authors**: Yiting Wang, Guoheng Sun, Wanghao Ye, Gang Qu, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11849)  

**Abstract**: Automating Register Transfer Level (RTL) code generation using Large Language Models (LLMs) offers substantial promise for streamlining digital circuit design and reducing human effort. However, current LLM-based approaches face significant challenges with training data scarcity, poor specification-code alignment, lack of verification mechanisms, and balancing generalization with specialization. Inspired by DeepSeek-R1, we introduce VeriReason, a framework integrating supervised fine-tuning with Guided Reward Proximal Optimization (GRPO) reinforcement learning for RTL generation. Using curated training examples and a feedback-driven reward model, VeriReason combines testbench evaluations with structural heuristics while embedding self-checking capabilities for autonomous error correction. On the VerilogEval Benchmark, VeriReason delivers significant improvements: achieving 83.1% functional correctness on the VerilogEval Machine benchmark, substantially outperforming both comparable-sized models and much larger commercial systems like GPT-4 Turbo. Additionally, our approach demonstrates up to a 2.8X increase in first-attempt functional correctness compared to baseline methods and exhibits robust generalization to unseen designs. To our knowledge, VeriReason represents the first system to successfully integrate explicit reasoning capabilities with reinforcement learning for Verilog generation, establishing a new state-of-the-art for automated RTL synthesis. The models and datasets are available at: this https URL Code is Available at: this https URL 

---
# RBF++: Quantifying and Optimizing Reasoning Boundaries across Measurable and Unmeasurable Capabilities for Chain-of-Thought Reasoning 

**Authors**: Qiguang Chen, Libo Qin, Jinhao Liu, Yue Liao, Jiaqi Wang, Jingxuan Zhou, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2505.13307)  

**Abstract**: Chain-of-Thought (CoT) reasoning has proven effective in enhancing large language models (LLMs) on complex tasks, spurring research into its underlying mechanisms. However, two primary challenges remain for real-world applications: (1) the lack of quantitative metrics and actionable guidelines for evaluating and optimizing measurable boundaries of CoT capability, and (2) the absence of methods to assess boundaries of unmeasurable CoT capability, such as multimodal perception. To address these gaps, we introduce the Reasoning Boundary Framework++ (RBF++). To tackle the first challenge, we define the reasoning boundary (RB) as the maximum limit of CoT performance. We also propose a combination law for RBs, enabling quantitative analysis and offering actionable guidance across various CoT tasks. For the second challenge, particularly in multimodal scenarios, we introduce a constant assumption, which replaces unmeasurable RBs with scenario-specific constants. Additionally, we propose the reasoning boundary division mechanism, which divides unmeasurable RBs into two sub-boundaries, facilitating the quantification and optimization of both unmeasurable domain knowledge and multimodal perception capabilities. Extensive experiments involving 38 models across 13 tasks validate the feasibility of our framework in cross-modal settings. Additionally, we evaluate 10 CoT strategies, offer insights into optimization and decay from two complementary perspectives, and expand evaluation benchmarks for measuring RBs in LLM reasoning. We hope this work advances the understanding of RBs and optimization strategies in LLMs. Code and data are available at this https URL. 

---
# Seek in the Dark: Reasoning via Test-Time Instance-Level Policy Gradient in Latent Space 

**Authors**: Hengli Li, Chenxi Li, Tong Wu, Xuekai Zhu, Yuxuan Wang, Zhaoxin Yu, Eric Hanchen Jiang, Song-Chun Zhu, Zixia Jia, Ying Nian Wu, Zilong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.13308)  

**Abstract**: Reasoning ability, a core component of human intelligence, continues to pose a significant challenge for Large Language Models (LLMs) in the pursuit of AGI. Although model performance has improved under the training scaling law, significant challenges remain, particularly with respect to training algorithms, such as catastrophic forgetting, and the limited availability of novel training data. As an alternative, test-time scaling enhances reasoning performance by increasing test-time computation without parameter updating. Unlike prior methods in this paradigm focused on token space, we propose leveraging latent space for more effective reasoning and better adherence to the test-time scaling law. We introduce LatentSeek, a novel framework that enhances LLM reasoning through Test-Time Instance-level Adaptation (TTIA) within the model's latent space. Specifically, LatentSeek leverages policy gradient to iteratively update latent representations, guided by self-generated reward signals. LatentSeek is evaluated on a range of reasoning benchmarks, including GSM8K, MATH-500, and AIME2024, across multiple LLM architectures. Results show that LatentSeek consistently outperforms strong baselines, such as Chain-of-Thought prompting and fine-tuning-based methods. Furthermore, our analysis demonstrates that LatentSeek is highly efficient, typically converging within a few iterations for problems of average complexity, while also benefiting from additional iterations, thereby highlighting the potential of test-time scaling in the latent space. These findings position LatentSeek as a lightweight, scalable, and effective solution for enhancing the reasoning capabilities of LLMs. 

---
# Cross-Cloud Data Privacy Protection: Optimizing Collaborative Mechanisms of AI Systems by Integrating Federated Learning and LLMs 

**Authors**: Huaiying Luo, Cheng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.13292)  

**Abstract**: In the age of cloud computing, data privacy protection has become a major challenge, especially when sharing sensitive data across cloud environments. However, how to optimize collaboration across cloud environments remains an unresolved problem. In this paper, we combine federated learning with large-scale language models to optimize the collaborative mechanism of AI systems. Based on the existing federated learning framework, we introduce a cross-cloud architecture in which federated learning works by aggregating model updates from decentralized nodes without exposing the original data. At the same time, combined with large-scale language models, its powerful context and semantic understanding capabilities are used to improve model training efficiency and decision-making ability. We've further innovated by introducing a secure communication layer to ensure the privacy and integrity of model updates and training data. The model enables continuous model adaptation and fine-tuning across different cloud environments while protecting sensitive data. Experimental results show that the proposed method is significantly better than the traditional federated learning model in terms of accuracy, convergence speed and data privacy protection. 

---
# Thinkless: LLM Learns When to Think 

**Authors**: Gongfan Fang, Xinyin Ma, Xinchao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13379)  

**Abstract**: Reasoning Language Models, capable of extended chain-of-thought reasoning, have demonstrated remarkable performance on tasks requiring complex logical inference. However, applying elaborate reasoning for all queries often results in substantial computational inefficiencies, particularly when many problems admit straightforward solutions. This motivates an open question: Can LLMs learn when to think? To answer this, we propose Thinkless, a learnable framework that empowers an LLM to adaptively select between short-form and long-form reasoning, based on both task complexity and the model's ability. Thinkless is trained under a reinforcement learning paradigm and employs two control tokens, <short> for concise responses and <think> for detailed reasoning. At the core of our method is a Decoupled Group Relative Policy Optimization (DeGRPO) algorithm, which decomposes the learning objective of hybrid reasoning into two components: (1) a control token loss that governs the selection of the reasoning mode, and (2) a response loss that improves the accuracy of the generated answers. This decoupled formulation enables fine-grained control over the contributions of each objective, stabilizing training and effectively preventing collapse observed in vanilla GRPO. Empirically, on several benchmarks such as Minerva Algebra, MATH-500, and GSM8K, Thinkless is able to reduce the usage of long-chain thinking by 50% - 90%, significantly improving the efficiency of Reasoning Language Models. The code is available at this https URL 

---
# ToolSpectrum : Towards Personalized Tool Utilization for Large Language Models 

**Authors**: Zihao Cheng, Hongru Wang, Zeming Liu, Yuhang Guo, Yuanfang Guo, Yunhong Wang, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13176)  

**Abstract**: While integrating external tools into large language models (LLMs) enhances their ability to access real-time information and domain-specific services, existing approaches focus narrowly on functional tool selection following user instructions, overlooking the context-aware personalization in tool selection. This oversight leads to suboptimal user satisfaction and inefficient tool utilization, particularly when overlapping toolsets require nuanced selection based on contextual factors. To bridge this gap, we introduce ToolSpectrum, a benchmark designed to evaluate LLMs' capabilities in personalized tool utilization. Specifically, we formalize two key dimensions of personalization, user profile and environmental factors, and analyze their individual and synergistic impacts on tool utilization. Through extensive experiments on ToolSpectrum, we demonstrate that personalized tool utilization significantly improves user experience across diverse scenarios. However, even state-of-the-art LLMs exhibit the limited ability to reason jointly about user profiles and environmental factors, often prioritizing one dimension at the expense of the other. Our findings underscore the necessity of context-aware personalization in tool-augmented LLMs and reveal critical limitations for current models. Our data and code are available at this https URL. 

---
# Role-Playing Evaluation for Large Language Models 

**Authors**: Yassine El Boudouri, Walter Nuninger, Julian Alvarez, Yvan Peter  

**Link**: [PDF](https://arxiv.org/pdf/2505.13157)  

**Abstract**: Large Language Models (LLMs) demonstrate a notable capacity for adopting personas and engaging in role-playing. However, evaluating this ability presents significant challenges, as human assessments are resource-intensive and automated evaluations can be biased. To address this, we introduce Role-Playing Eval (RPEval), a novel benchmark designed to assess LLM role-playing capabilities across four key dimensions: emotional understanding, decision-making, moral alignment, and in-character consistency. This article details the construction of RPEval and presents baseline evaluations. Our code and dataset are available at this https URL 

---
# Tianyi: A Traditional Chinese Medicine all-rounder language model and its Real-World Clinical Practice 

**Authors**: Zhi Liu, Tao Yang, Jing Wang, Yexin Chen, Zhan Gao, Jiaxi Yang, Kui Chen, Bingji Lu, Xiaochen Li, Changyong Luo, Yan Li, Xiaohong Gu, Peng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13156)  

**Abstract**: Natural medicines, particularly Traditional Chinese Medicine (TCM), are gaining global recognition for their therapeutic potential in addressing human symptoms and diseases. TCM, with its systematic theories and extensive practical experience, provides abundant resources for healthcare. However, the effective application of TCM requires precise syndrome diagnosis, determination of treatment principles, and prescription formulation, which demand decades of clinical expertise. Despite advancements in TCM-based decision systems, machine learning, and deep learning research, limitations in data and single-objective constraints hinder their practical application. In recent years, large language models (LLMs) have demonstrated potential in complex tasks, but lack specialization in TCM and face significant challenges, such as too big model scale to deploy and issues with hallucination. To address these challenges, we introduce Tianyi with 7.6-billion-parameter LLM, a model scale proper and specifically designed for TCM, pre-trained and fine-tuned on diverse TCM corpora, including classical texts, expert treatises, clinical records, and knowledge graphs. Tianyi is designed to assimilate interconnected and systematic TCM knowledge through a progressive learning manner. Additionally, we establish TCMEval, a comprehensive evaluation benchmark, to assess LLMs in TCM examinations, clinical tasks, domain-specific question-answering, and real-world trials. The extensive evaluations demonstrate the significant potential of Tianyi as an AI assistant in TCM clinical practice and research, bridging the gap between TCM knowledge and practical application. 

---
# Benchmarking and Confidence Evaluation of LALMs For Temporal Reasoning 

**Authors**: Debarpan Bhattacharya, Apoorva Kulkarni, Sriram Ganapathy  

**Link**: [PDF](https://arxiv.org/pdf/2505.13115)  

**Abstract**: The popular success of text-based large language models (LLM) has streamlined the attention of the multimodal community to combine other modalities like vision and audio along with text to achieve similar multimodal capabilities. In this quest, large audio language models (LALMs) have to be evaluated on reasoning related tasks which are different from traditional classification or generation tasks. Towards this goal, we propose a novel dataset called temporal reasoning evaluation of audio (TREA).
We benchmark open-source LALMs and observe that they are consistently behind human capabilities on the tasks in the TREA dataset. While evaluating LALMs, we also propose an uncertainty metric, which computes the invariance of the model to semantically identical perturbations of the input. Our analysis shows that the accuracy and uncertainty metrics are not necessarily correlated and thus, points to a need for wholesome evaluation of LALMs for high-stakes applications. 

---
# FreeKV: Boosting KV Cache Retrieval for Efficient LLM Inference 

**Authors**: Guangda Liu, Chengwei Li, Zhenyu Ning, Jing Lin, Yiwu Yao, Danning Ke, Minyi Guo, Jieru Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13109)  

**Abstract**: Large language models (LLMs) have been widely deployed with rapidly expanding context windows to support increasingly demanding applications. However, long contexts pose significant deployment challenges, primarily due to the KV cache whose size grows proportionally with context length. While KV cache compression methods are proposed to address this issue, KV dropping methods incur considerable accuracy loss, and KV retrieval methods suffer from significant efficiency bottlenecks. We propose FreeKV, an algorithm-system co-optimization framework to enhance KV retrieval efficiency while preserving accuracy. On the algorithm side, FreeKV introduces speculative retrieval to shift the KV selection and recall processes out of the critical path, combined with fine-grained correction to ensure accuracy. On the system side, FreeKV employs hybrid KV layouts across CPU and GPU memory to eliminate fragmented data transfers, and leverages double-buffered streamed recall to further improve efficiency. Experiments demonstrate that FreeKV achieves near-lossless accuracy across various scenarios and models, delivering up to 13$\times$ speedup compared to SOTA KV retrieval methods. 

---
# A Physics-Inspired Optimizer: Velocity Regularized Adam 

**Authors**: Pranav Vaidhyanathan, Lucas Schorling, Natalia Ares, Michael A. Osborne  

**Link**: [PDF](https://arxiv.org/pdf/2505.13196)  

**Abstract**: We introduce Velocity-Regularized Adam (VRAdam), a physics-inspired optimizer for training deep neural networks that draws on ideas from quartic terms for kinetic energy with its stabilizing effects on various system dynamics. Previous algorithms, including the ubiquitous Adam, operate at the so called adaptive edge of stability regime during training leading to rapid oscillations and slowed convergence of loss. However, VRAdam adds a higher order penalty on the learning rate based on the velocity such that the algorithm automatically slows down whenever weight updates become large. In practice, we observe that the effective dynamic learning rate shrinks in high-velocity regimes, damping oscillations and allowing for a more aggressive base step size when necessary without divergence. By combining this velocity-based regularizer for global damping with per-parameter scaling of Adam to create a hybrid optimizer, we demonstrate that VRAdam consistently exceeds the performance against standard optimizers including AdamW. We benchmark various tasks such as image classification, language modeling, image generation and generative modeling using diverse architectures and training methodologies including Convolutional Neural Networks (CNNs), Transformers, and GFlowNets. 

---
# Advancing Sequential Numerical Prediction in Autoregressive Models 

**Authors**: Xiang Fei, Jinghui Lu, Qi Sun, Hao Feng, Yanjie Wang, Wei Shi, An-Lan Wang, Jingqun Tang, Can Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13077)  

**Abstract**: Autoregressive models have become the de facto choice for sequence generation tasks, but standard approaches treat digits as independent tokens and apply cross-entropy loss, overlooking the coherent structure of numerical sequences. This paper introduces Numerical Token Integrity Loss (NTIL) to address this gap. NTIL operates at two levels: (1) token-level, where it extends the Earth Mover's Distance (EMD) to preserve ordinal relationships between numerical values, and (2) sequence-level, where it penalizes the overall discrepancy between the predicted and actual sequences. This dual approach improves numerical prediction and integrates effectively with LLMs/MLLMs. Extensive experiments show significant performance improvements with NTIL. 

---
# Structure-Aware Corpus Construction and User-Perception-Aligned Metrics for Large-Language-Model Code Completion 

**Authors**: Dengfeng Liu, Jucai Zhai, Xiaoguang Jiang, Ziqun Li, Qianjin Yu, Feng Liu, Rui Ye, Huang Liu, Zhiguo Yang, Yongsheng Du, Fang Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13073)  

**Abstract**: Code completion technology based on large language model has significantly improved the development efficiency of programmers. However, in practical applications, there remains a gap between current commonly used code completion evaluation metrics and users' actual perception. To address this issue, we propose two evaluation metrics for code completion tasks--LCP and ROUGE-LCP, from the perspective of probabilistic modeling. Furthermore, to tackle the lack of effective structural semantic modeling and cross-module dependency information in LLMs for repository-level code completion scenarios, we propose a data processing method based on a Structure-Preserving and Semantically-Reordered Code Graph (SPSR-Graph). Through theoretical analysis and experimental validation, we demonstrate the superiority of the proposed evaluation metrics in terms of user perception consistency, as well as the effectiveness of the data processing method in enhancing model performance. 

---
# Step-wise Adaptive Integration of Supervised Fine-tuning and Reinforcement Learning for Task-Specific LLMs 

**Authors**: Jack Chen, Fazhong Liu, Naruto Liu, Yuhan Luo, Erqu Qin, Harry Zheng, Tian Dong, Haojin Zhu, Yan Meng, Xiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.13026)  

**Abstract**: Large language models (LLMs) excel at mathematical reasoning and logical problem-solving. The current popular training paradigms primarily use supervised fine-tuning (SFT) and reinforcement learning (RL) to enhance the models' reasoning abilities. However, when using SFT or RL alone, there are respective challenges: SFT may suffer from overfitting, while RL is prone to mode collapse. The state-of-the-art methods have proposed hybrid training schemes. However, static switching faces challenges such as poor generalization across different tasks and high dependence on data quality. In response to these challenges, inspired by the curriculum learning-quiz mechanism in human reasoning cultivation, We propose SASR, a step-wise adaptive hybrid training framework that theoretically unifies SFT and RL and dynamically balances the two throughout optimization. SASR uses SFT for initial warm-up to establish basic reasoning skills, and then uses an adaptive dynamic adjustment algorithm based on gradient norm and divergence relative to the original distribution to seamlessly integrate SFT with the online RL method GRPO. By monitoring the training status of LLMs and adjusting the training process in sequence, SASR ensures a smooth transition between training schemes, maintaining core reasoning abilities while exploring different paths. Experimental results demonstrate that SASR outperforms SFT, RL, and static hybrid training methods. 

---
# An Empirical Study of Many-to-Many Summarization with Large Language Models 

**Authors**: Jiaan Wang, Fandong Meng, Zengkui Sun, Yunlong Liang, Yuxuan Cao, Jiarong Xu, Haoxiang Shi, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12983)  

**Abstract**: Many-to-many summarization (M2MS) aims to process documents in any language and generate the corresponding summaries also in any language. Recently, large language models (LLMs) have shown strong multi-lingual abilities, giving them the potential to perform M2MS in real applications. This work presents a systematic empirical study on LLMs' M2MS ability. Specifically, we first reorganize M2MS data based on eight previous domain-specific datasets. The reorganized data contains 47.8K samples spanning five domains and six languages, which could be used to train and evaluate LLMs. Then, we benchmark 18 LLMs in a zero-shot manner and an instruction-tuning manner. Fine-tuned traditional models (e.g., mBART) are also conducted for comparisons. Our experiments reveal that, zero-shot LLMs achieve competitive results with fine-tuned traditional models. After instruct-tuning, open-source LLMs can significantly improve their M2MS ability, and outperform zero-shot LLMs (including GPT-4) in terms of automatic evaluations. In addition, we demonstrate that this task-specific improvement does not sacrifice the LLMs' general task-solving abilities. However, as revealed by our human evaluation, LLMs still face the factuality issue, and the instruction tuning might intensify the issue. Thus, how to control factual errors becomes the key when building LLM summarizers in real applications, and is worth noting in future research. 

---
# Fractured Chain-of-Thought Reasoning 

**Authors**: Baohao Liao, Hanze Dong, Yuhui Xu, Doyen Sahoo, Christof Monz, Junnan Li, Caiming Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12992)  

**Abstract**: Inference-time scaling techniques have significantly bolstered the reasoning capabilities of large language models (LLMs) by harnessing additional computational effort at inference without retraining. Similarly, Chain-of-Thought (CoT) prompting and its extension, Long CoT, improve accuracy by generating rich intermediate reasoning trajectories, but these approaches incur substantial token costs that impede their deployment in latency-sensitive settings. In this work, we first show that truncated CoT, which stops reasoning before completion and directly generates the final answer, often matches full CoT sampling while using dramatically fewer tokens. Building on this insight, we introduce Fractured Sampling, a unified inference-time strategy that interpolates between full CoT and solution-only sampling along three orthogonal axes: (1) the number of reasoning trajectories, (2) the number of final solutions per trajectory, and (3) the depth at which reasoning traces are truncated. Through extensive experiments on five diverse reasoning benchmarks and several model scales, we demonstrate that Fractured Sampling consistently achieves superior accuracy-cost trade-offs, yielding steep log-linear scaling gains in Pass@k versus token budget. Our analysis reveals how to allocate computation across these dimensions to maximize performance, paving the way for more efficient and scalable LLM reasoning. 

---
# MultiActor-Audiobook: Zero-Shot Audiobook Generation with Faces and Voices of Multiple Speakers 

**Authors**: Kyeongman Park, Seongho Joo, Kyomin Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.13082)  

**Abstract**: We introduce MultiActor-Audiobook, a zero-shot approach for generating audiobooks that automatically produces consistent, expressive, and speaker-appropriate prosody, including intonation and emotion. Previous audiobook systems have several limitations: they require users to manually configure the speaker's prosody, read each sentence with a monotonic tone compared to voice actors, or rely on costly training. However, our MultiActor-Audiobook addresses these issues by introducing two novel processes: (1) MSP (**Multimodal Speaker Persona Generation**) and (2) LSI (**LLM-based Script Instruction Generation**). With these two processes, MultiActor-Audiobook can generate more emotionally expressive audiobooks with a consistent speaker prosody without additional training. We compare our system with commercial products, through human and MLLM evaluations, achieving competitive results. Furthermore, we demonstrate the effectiveness of MSP and LSI through ablation studies. 

---
# KIT's Offline Speech Translation and Instruction Following Submission for IWSLT 2025 

**Authors**: Sai Koneru, Maike Züfle, Thai-Binh Nguyen, Seymanur Akti, Jan Niehues, Alexander Waibel  

**Link**: [PDF](https://arxiv.org/pdf/2505.13036)  

**Abstract**: The scope of the International Workshop on Spoken Language Translation (IWSLT) has recently broadened beyond traditional Speech Translation (ST) to encompass a wider array of tasks, including Speech Question Answering and Summarization. This shift is partly driven by the growing capabilities of modern systems, particularly with the success of Large Language Models (LLMs). In this paper, we present the Karlsruhe Institute of Technology's submissions for the Offline ST and Instruction Following (IF) tracks, where we leverage LLMs to enhance performance across all tasks. For the Offline ST track, we propose a pipeline that employs multiple automatic speech recognition systems, whose outputs are fused using an LLM with document-level context. This is followed by a two-step translation process, incorporating additional refinement step to improve translation quality. For the IF track, we develop an end-to-end model that integrates a speech encoder with an LLM to perform a wide range of instruction-following tasks. We complement it with a final document-level refinement stage to further enhance output quality by using contextual information. 

---
# From Assistants to Adversaries: Exploring the Security Risks of Mobile LLM Agents 

**Authors**: Liangxuan Wu, Chao Wang, Tianming Liu, Yanjie Zhao, Haoyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12981)  

**Abstract**: The growing adoption of large language models (LLMs) has led to a new paradigm in mobile computing--LLM-powered mobile AI agents--capable of decomposing and automating complex tasks directly on smartphones. However, the security implications of these agents remain largely unexplored. In this paper, we present the first comprehensive security analysis of mobile LLM agents, encompassing three representative categories: System-level AI Agents developed by original equipment manufacturers (e.g., YOYO Assistant), Third-party Universal Agents (e.g., Zhipu AI AutoGLM), and Emerging Agent Frameworks (e.g., Alibaba Mobile Agent). We begin by analyzing the general workflow of mobile agents and identifying security threats across three core capability dimensions: language-based reasoning, GUI-based interaction, and system-level execution. Our analysis reveals 11 distinct attack surfaces, all rooted in the unique capabilities and interaction patterns of mobile LLM agents, and spanning their entire operational lifecycle. To investigate these threats in practice, we introduce AgentScan, a semi-automated security analysis framework that systematically evaluates mobile LLM agents across all 11 attack scenarios. Applying AgentScan to nine widely deployed agents, we uncover a concerning trend: every agent is vulnerable to targeted attacks. In the most severe cases, agents exhibit vulnerabilities across eight distinct attack vectors. These attacks can cause behavioral deviations, privacy leakage, or even full execution hijacking. Based on these findings, we propose a set of defensive design principles and practical recommendations for building secure mobile LLM agents. Our disclosures have received positive feedback from two major device vendors. Overall, this work highlights the urgent need for standardized security practices in the fast-evolving landscape of LLM-driven mobile automation. 

---
# Do Not Let Low-Probability Tokens Over-Dominate in RL for LLMs 

**Authors**: Zhihe Yang, Xufang Luo, Zilong Wang, Dongqi Han, Zhiyuan He, Dongsheng Li, Yunjian Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12929)  

**Abstract**: Reinforcement learning (RL) has become a cornerstone for enhancing the reasoning capabilities of large language models (LLMs), with recent innovations such as Group Relative Policy Optimization (GRPO) demonstrating exceptional effectiveness. In this study, we identify a critical yet underexplored issue in RL training: low-probability tokens disproportionately influence model updates due to their large gradient magnitudes. This dominance hinders the effective learning of high-probability tokens, whose gradients are essential for LLMs' performance but are substantially suppressed. To mitigate this interference, we propose two novel methods: Advantage Reweighting and Low-Probability Token Isolation (Lopti), both of which effectively attenuate gradients from low-probability tokens while emphasizing parameter updates driven by high-probability tokens. Our approaches promote balanced updates across tokens with varying probabilities, thereby enhancing the efficiency of RL training. Experimental results demonstrate that they substantially improve the performance of GRPO-trained LLMs, achieving up to a 46.2% improvement in K&K Logic Puzzle reasoning tasks. Our implementation is available at this https URL. 

---
# DGRO: Enhancing LLM Reasoning via Exploration-Exploitation Control and Reward Variance Management 

**Authors**: Xuerui Su, Liya Guo, Yue Wang, Yi Zhu, Zhiming Ma, Zun Wang, Yuting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12951)  

**Abstract**: Inference scaling further accelerates Large Language Models (LLMs) toward Artificial General Intelligence (AGI), with large-scale Reinforcement Learning (RL) to unleash long Chain-of-Thought reasoning. Most contemporary reasoning approaches usually rely on handcrafted rule-based reward functions. However, the tarde-offs of exploration and exploitation in RL algorithms involves multiple complex considerations, and the theoretical and empirical impacts of manually designed reward functions remain insufficiently explored. In this paper, we propose Decoupled Group Reward Optimization (DGRO), a general RL algorithm for LLM reasoning. On the one hand, DGRO decouples the traditional regularization coefficient into two independent hyperparameters: one scales the policy gradient term, and the other regulates the distance from the sampling policy. This decoupling not only enables precise control over balancing exploration and exploitation, but also can be seamlessly extended to Online Policy Mirror Descent (OPMD) algorithms in Kimi k1.5 and Direct Reward Optimization. On the other hand, we observe that reward variance significantly affects both convergence speed and final model performance. We conduct both theoretical analysis and extensive empirical validation to assess DGRO, including a detailed ablation study that investigates its performance and optimization dynamics. Experimental results show that DGRO achieves state-of-the-art performance on the Logic dataset with an average accuracy of 96.9\%, and demonstrates strong generalization across mathematical benchmarks. 

---
# Leveraging LLM Inconsistency to Boost Pass@k Performance 

**Authors**: Uri Dalal, Meirav Segal, Zvika Ben-Haim, Dan Lahav, Omer Nevo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12938)  

**Abstract**: Large language models (LLMs) achieve impressive abilities in numerous domains, but exhibit inconsistent performance in response to minor input changes. Rather than view this as a drawback, in this paper we introduce a novel method for leveraging models' inconsistency to boost Pass@k performance. Specifically, we present a "Variator" agent that generates k variants of a given task and submits one candidate solution for each one. Our variant generation approach is applicable to a wide range of domains as it is task agnostic and compatible with free-form inputs. We demonstrate the efficacy of our agent theoretically using a probabilistic model of the inconsistency effect, and show empirically that it outperforms the baseline on the APPS dataset. Furthermore, we establish that inconsistency persists even in frontier reasoning models across coding and cybersecurity domains, suggesting our method is likely to remain relevant for future model generations. 

---
# AutoGEEval: A Multimodal and Automated Framework for Geospatial Code Generation on GEE with Large Language Models 

**Authors**: Shuyang Hou, Zhangxiao Shen, Huayi Wu, Jianyuan Liang, Haoyue Jiao, Yaxian Qing, Xiaopu Zhang, Xu Li, Zhipeng Gui, Xuefeng Guan, Longgang Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12900)  

**Abstract**: Geospatial code generation is emerging as a key direction in the integration of artificial intelligence and geoscientific analysis. However, there remains a lack of standardized tools for automatic evaluation in this domain. To address this gap, we propose AutoGEEval, the first multimodal, unit-level automated evaluation framework for geospatial code generation tasks on the Google Earth Engine (GEE) platform powered by large language models (LLMs). Built upon the GEE Python API, AutoGEEval establishes a benchmark suite (AutoGEEval-Bench) comprising 1325 test cases that span 26 GEE data types. The framework integrates both question generation and answer verification components to enable an end-to-end automated evaluation pipeline-from function invocation to execution validation. AutoGEEval supports multidimensional quantitative analysis of model outputs in terms of accuracy, resource consumption, execution efficiency, and error types. We evaluate 18 state-of-the-art LLMs-including general-purpose, reasoning-augmented, code-centric, and geoscience-specialized models-revealing their performance characteristics and potential optimization pathways in GEE code generation. This work provides a unified protocol and foundational resource for the development and assessment of geospatial code generation models, advancing the frontier of automated natural language to domain-specific code translation. 

---
# ExTrans: Multilingual Deep Reasoning Translation via Exemplar-Enhanced Reinforcement Learning 

**Authors**: Jiaan Wang, Fandong Meng, Jie Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12996)  

**Abstract**: In recent years, the emergence of large reasoning models (LRMs), such as OpenAI-o1 and DeepSeek-R1, has shown impressive capabilities in complex problems, e.g., mathematics and coding. Some pioneering studies attempt to bring the success of LRMs in neural machine translation (MT). They try to build LRMs with deep reasoning MT ability via reinforcement learning (RL). Despite some progress that has been made, these attempts generally focus on several high-resource languages, e.g., English and Chinese, leaving the performance on other languages unclear. Besides, the reward modeling methods in previous work do not fully unleash the potential of reinforcement learning in MT. In this work, we first design a new reward modeling method that compares the translation results of the policy MT model with a strong LRM (i.e., DeepSeek-R1-671B), and quantifies the comparisons to provide rewards. Experimental results demonstrate the superiority of the reward modeling method. Using Qwen2.5-7B-Instruct as the backbone, the trained model achieves the new state-of-the-art performance in literary translation, and outperforms strong LRMs including OpenAI-o1 and DeepSeeK-R1. Furthermore, we extend our method to the multilingual settings with 11 languages. With a carefully designed lightweight reward modeling in RL, we can simply transfer the strong MT ability from a single direction into multiple (i.e., 90) translation directions and achieve impressive multilingual MT performance. 

---
# Sinusoidal Initialization, Time for a New Start 

**Authors**: Alberto Fernández-Hernández, Jose I. Mestre, Manuel F. Dolz, Jose Duato, Enrique S. Quintana-Ortí  

**Link**: [PDF](https://arxiv.org/pdf/2505.12909)  

**Abstract**: Initialization plays a critical role in Deep Neural Network training, directly influencing convergence, stability, and generalization. Common approaches such as Glorot and He initializations rely on randomness, which can produce uneven weight distributions across layer connections. In this paper, we introduce the Sinusoidal initialization, a novel deterministic method that employs sinusoidal functions to construct structured weight matrices expressly to improve the spread and balance of weights throughout the network while simultaneously fostering a more uniform, well-conditioned distribution of neuron activation states from the very first forward pass. Because Sinusoidal initialization begins with weights and activations that are already evenly and efficiently utilized, it delivers consistently faster convergence, greater training stability, and higher final accuracy across a wide range of models, including convolutional neural networks, vision transformers, and large language models. On average, our experiments show an increase of 4.8 % in final validation accuracy and 20.9 % in convergence speed. By replacing randomness with structure, this initialization provides a stronger and more reliable foundation for Deep Learning systems. 

---
# LEXam: Benchmarking Legal Reasoning on 340 Law Exams 

**Authors**: Yu Fan, Jingwei Ni, Jakob Merane, Etienne Salimbeni, Yang Tian, Yoan Hermstrüwer, Yinya Huang, Mubashara Akhtar, Florian Geering, Oliver Dreyer, Daniel Brunner, Markus Leippold, Mrinmaya Sachan, Alexander Stremitzer, Christoph Engel, Elliott Ash, Joel Niklaus  

**Link**: [PDF](https://arxiv.org/pdf/2505.12864)  

**Abstract**: Long-form legal reasoning remains a key challenge for large language models (LLMs) in spite of recent advances in test-time scaling. We introduce LEXam, a novel benchmark derived from 340 law exams spanning 116 law school courses across a range of subjects and degree levels. The dataset comprises 4,886 law exam questions in English and German, including 2,841 long-form, open-ended questions and 2,045 multiple-choice questions. Besides reference answers, the open questions are also accompanied by explicit guidance outlining the expected legal reasoning approach such as issue spotting, rule recall, or rule application. Our evaluation on both open-ended and multiple-choice questions present significant challenges for current LLMs; in particular, they notably struggle with open questions that require structured, multi-step legal reasoning. Moreover, our results underscore the effectiveness of the dataset in differentiating between models with varying capabilities. Adopting an LLM-as-a-Judge paradigm with rigorous human expert validation, we demonstrate how model-generated reasoning steps can be evaluated consistently and accurately. Our evaluation setup provides a scalable method to assess legal reasoning quality beyond simple accuracy metrics. Project page: this https URL 

---
# SynDec: A Synthesize-then-Decode Approach for Arbitrary Textual Style Transfer via Large Language Models 

**Authors**: Han Sun, Zhen Sun, Zongmin Zhang, Linzhao Jia, Wei Shao, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12821)  

**Abstract**: Large Language Models (LLMs) are emerging as dominant forces for textual style transfer. However, for arbitrary style transfer, LLMs face two key challenges: (1) considerable reliance on manually-constructed prompts and (2) rigid stylistic biases inherent in LLMs. In this paper, we propose a novel Synthesize-then-Decode (SynDec) approach, which automatically synthesizes high-quality prompts and amplifies their roles during decoding process. Specifically, our approach synthesizes prompts by selecting representative few-shot samples, conducting a four-dimensional style analysis, and reranking the candidates. At LLM decoding stage, the TST effect is amplified by maximizing the contrast in output probabilities between scenarios with and without the synthesized prompt, as well as between prompts and negative samples. We conduct extensive experiments and the results show that SynDec outperforms existing state-of-the-art LLM-based methods on five out of six benchmarks (e.g., achieving up to a 9\% increase in accuracy for modern-to-Elizabethan English transfer). Detailed ablation studies further validate the effectiveness of SynDec. 

---
# The Hidden Structure -- Improving Legal Document Understanding Through Explicit Text Formatting 

**Authors**: Christian Braun, Alexander Lilienbeck, Daniel Mentjukov  

**Link**: [PDF](https://arxiv.org/pdf/2505.12837)  

**Abstract**: Legal contracts possess an inherent, semantically vital structure (e.g., sections, clauses) that is crucial for human comprehension but whose impact on LLM processing remains under-explored. This paper investigates the effects of explicit input text structure and prompt engineering on the performance of GPT-4o and GPT-4.1 on a legal question-answering task using an excerpt of the CUAD. We compare model exact-match accuracy across various input formats: well-structured plain-text (human-generated from CUAD), plain-text cleaned of line breaks, extracted plain-text from Azure OCR, plain-text extracted by GPT-4o Vision, and extracted (and interpreted) Markdown (MD) from GPT-4o Vision. To give an indication of the impact of possible prompt engineering, we assess the impact of shifting task instructions to the system prompt and explicitly informing the model about the structured nature of the input. Our findings reveal that GPT-4o demonstrates considerable robustness to variations in input structure, but lacks in overall performance. Conversely, GPT-4.1's performance is markedly sensitive; poorly structured inputs yield suboptimal results (but identical with GPT-4o), while well-structured formats (original CUAD text, GPT-4o Vision text and GPT-4o MD) improve exact-match accuracy by ~20 percentage points. Optimizing the system prompt to include task details and an advisory about structured input further elevates GPT-4.1's accuracy by an additional ~10-13 percentage points, with Markdown ultimately achieving the highest performance under these conditions (79 percentage points overall exact-match accuracy). This research empirically demonstrates that while newer models exhibit greater resilience, careful input structuring and strategic prompt design remain critical for optimizing the performance of LLMs, and can significantly affect outcomes in high-stakes legal applications. 

---
# EpiLLM: Unlocking the Potential of Large Language Models in Epidemic Forecasting 

**Authors**: Chenghua Gong, Rui Sun, Yuhao Zheng, Juyuan Zhang, Tianjun Gu, Liming Pan, Linyuan Lv  

**Link**: [PDF](https://arxiv.org/pdf/2505.12738)  

**Abstract**: Advanced epidemic forecasting is critical for enabling precision containment strategies, highlighting its strategic importance for public health security. While recent advances in Large Language Models (LLMs) have demonstrated effectiveness as foundation models for domain-specific tasks, their potential for epidemic forecasting remains largely unexplored. In this paper, we introduce EpiLLM, a novel LLM-based framework tailored for spatio-temporal epidemic forecasting. Considering the key factors in real-world epidemic transmission: infection cases and human mobility, we introduce a dual-branch architecture to achieve fine-grained token-level alignment between such complex epidemic patterns and language tokens for LLM adaptation. To unleash the multi-step forecasting and generalization potential of LLM architectures, we propose an autoregressive modeling paradigm that reformulates the epidemic forecasting task into next-token prediction. To further enhance LLM perception of epidemics, we introduce spatio-temporal prompt learning techniques, which strengthen forecasting capabilities from a data-driven perspective. Extensive experiments show that EpiLLM significantly outperforms existing baselines on real-world COVID-19 datasets and exhibits scaling behavior characteristic of LLMs. 

---
# Does Low Rank Adaptation Lead to Lower Robustness against Training-Time Attacks? 

**Authors**: Zi Liang, Haibo Hu, Qingqing Ye, Yaxin Xiao, Ronghua Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12871)  

**Abstract**: Low rank adaptation (LoRA) has emerged as a prominent technique for fine-tuning large language models (LLMs) thanks to its superb efficiency gains over previous methods. While extensive studies have examined the performance and structural properties of LoRA, its behavior upon training-time attacks remain underexplored, posing significant security risks. In this paper, we theoretically investigate the security implications of LoRA's low-rank structure during fine-tuning, in the context of its robustness against data poisoning and backdoor attacks. We propose an analytical framework that models LoRA's training dynamics, employs the neural tangent kernel to simplify the analysis of the training process, and applies information theory to establish connections between LoRA's low rank structure and its vulnerability against training-time attacks. Our analysis indicates that LoRA exhibits better robustness to backdoor attacks than full fine-tuning, while becomes more vulnerable to untargeted data poisoning due to its over-simplified information geometry. Extensive experimental evaluations have corroborated our theoretical findings. 

---
# Shadow-FT: Tuning Instruct via Base 

**Authors**: Taiqiang Wu, Runming Yang, Jiayi Li, Pengfei Hu, Ngai Wong, Yujiu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12716)  

**Abstract**: Large language models (LLMs) consistently benefit from further fine-tuning on various tasks. However, we observe that directly tuning the INSTRUCT (i.e., instruction tuned) models often leads to marginal improvements and even performance degeneration. Notably, paired BASE models, the foundation for these INSTRUCT variants, contain highly similar weight values (i.e., less than 2% on average for Llama 3.1 8B). Therefore, we propose a novel Shadow-FT framework to tune the INSTRUCT models by leveraging the corresponding BASE models. The key insight is to fine-tune the BASE model, and then directly graft the learned weight updates to the INSTRUCT model. Our proposed Shadow-FT introduces no additional parameters, is easy to implement, and significantly improves performance. We conduct extensive experiments on tuning mainstream LLMs, such as Qwen 3 and Llama 3 series, and evaluate them across 19 benchmarks covering coding, reasoning, and mathematical tasks. Experimental results demonstrate that Shadow-FT consistently outperforms conventional full-parameter and parameter-efficient tuning approaches. Further analyses indicate that Shadow-FT can be applied to multimodal large language models (MLLMs) and combined with direct preference optimization (DPO). Codes and weights are available at \href{this https URL}{Github}. 

---
# Know3-RAG: A Knowledge-aware RAG Framework with Adaptive Retrieval, Generation, and Filtering 

**Authors**: Xukai Liu, Ye Liu, Shiwen Wu, Yanghai Zhang, Yihao Yuan, Kai Zhang, Qi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12662)  

**Abstract**: Recent advances in large language models (LLMs) have led to impressive progress in natural language generation, yet their tendency to produce hallucinated or unsubstantiated content remains a critical concern. To improve factual reliability, Retrieval-Augmented Generation (RAG) integrates external knowledge during inference. However, existing RAG systems face two major limitations: (1) unreliable adaptive control due to limited external knowledge supervision, and (2) hallucinations caused by inaccurate or irrelevant references. To address these issues, we propose Know3-RAG, a knowledge-aware RAG framework that leverages structured knowledge from knowledge graphs (KGs) to guide three core stages of the RAG process, including retrieval, generation, and filtering. Specifically, we introduce a knowledge-aware adaptive retrieval module that employs KG embedding to assess the confidence of the generated answer and determine retrieval necessity, a knowledge-enhanced reference generation strategy that enriches queries with KG-derived entities to improve generated reference relevance, and a knowledge-driven reference filtering mechanism that ensures semantic alignment and factual accuracy of references. Experiments on multiple open-domain QA benchmarks demonstrate that Know3-RAG consistently outperforms strong baselines, significantly reducing hallucinations and enhancing answer reliability. 

---
# AD-AGENT: A Multi-agent Framework for End-to-end Anomaly Detection 

**Authors**: Tiankai Yang, Junjun Liu, Wingchun Siu, Jiahang Wang, Zhuangzhuang Qian, Chanjuan Song, Cheng Cheng, Xiyang Hu, Yue Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12594)  

**Abstract**: Anomaly detection (AD) is essential in areas such as fraud detection, network monitoring, and scientific research. However, the diversity of data modalities and the increasing number of specialized AD libraries pose challenges for non-expert users who lack in-depth library-specific knowledge and advanced programming skills. To tackle this, we present AD-AGENT, an LLM-driven multi-agent framework that turns natural-language instructions into fully executable AD pipelines. AD-AGENT coordinates specialized agents for intent parsing, data preparation, library and model selection, documentation mining, and iterative code generation and debugging. Using a shared short-term workspace and a long-term cache, the agents integrate popular AD libraries like PyOD, PyGOD, and TSLib into a unified workflow. Experiments demonstrate that AD-AGENT produces reliable scripts and recommends competitive models across libraries. The system is open-sourced to support further research and practical applications in AD. 

---
# Web IP at Risk: Prevent Unauthorized Real-Time Retrieval by Large Language Models 

**Authors**: Yisheng Zhong, Yizhu Wen, Junfeng Guo, Mehran Kafai, Heng Huang, Hanqing Guo, Zhuangdi Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12655)  

**Abstract**: Protecting cyber Intellectual Property (IP) such as web content is an increasingly critical concern. The rise of large language models (LLMs) with online retrieval capabilities presents a double-edged sword that enables convenient access to information but often undermines the rights of original content creators. As users increasingly rely on LLM-generated responses, they gradually diminish direct engagement with original information sources, significantly reducing the incentives for IP creators to contribute, and leading to a saturating cyberspace with more AI-generated content. In response, we propose a novel defense framework that empowers web content creators to safeguard their web-based IP from unauthorized LLM real-time extraction by leveraging the semantic understanding capability of LLMs themselves. Our method follows principled motivations and effectively addresses an intractable black-box optimization problem. Real-world experiments demonstrated that our methods improve defense success rates from 2.5% to 88.6% on different LLMs, outperforming traditional defenses such as configuration-based restrictions. 

---
# Measuring Information Distortion in Hierarchical Ultra long Novel Generation:The Optimal Expansion Ratio 

**Authors**: Hanwen Shen, Ting Ying  

**Link**: [PDF](https://arxiv.org/pdf/2505.12572)  

**Abstract**: Writing novels with Large Language Models (LLMs) raises a critical question: how much human-authored outline is necessary to generate high-quality million-word novels? While frameworks such as DOME, Plan&Write, and Long Writer have improved stylistic coherence and logical consistency, they primarily target shorter novels (10k--100k words), leaving ultra-long generation largely unexplored. Drawing on insights from recent text compression methods like LLMZip and LLM2Vec, we conduct an information-theoretic analysis that quantifies distortion occurring when LLMs compress and reconstruct ultra-long novels under varying compression-expansion ratios. We introduce a hierarchical two-stage generation pipeline (outline -> detailed outline -> manuscript) and find an optimal outline length that balances information preservation with human effort. Through extensive experimentation with Chinese novels, we establish that a two-stage hierarchical outline approach significantly reduces semantic distortion compared to single-stage methods. Our findings provide empirically-grounded guidance for authors and researchers collaborating with LLMs to create million-word novels. 

---
# Towards Budget-Friendly Model-Agnostic Explanation Generation for Large Language Models 

**Authors**: Junhao Liu, Haonan Yu, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12509)  

**Abstract**: With Large language models (LLMs) becoming increasingly prevalent in various applications, the need for interpreting their predictions has become a critical challenge. As LLMs vary in architecture and some are closed-sourced, model-agnostic techniques show great promise without requiring access to the model's internal parameters. However, existing model-agnostic techniques need to invoke LLMs many times to gain sufficient samples for generating faithful explanations, which leads to high economic costs. In this paper, we show that it is practical to generate faithful explanations for large-scale LLMs by sampling from some budget-friendly models through a series of empirical studies. Moreover, we show that such proxy explanations also perform well on downstream tasks. Our analysis provides a new paradigm of model-agnostic explanation methods for LLMs, by including information from budget-friendly models. 

---
# Enhancing Large Language Models with Reward-guided Tree Search for Knowledge Graph Question and Answering 

**Authors**: Xiao Long, Liansheng Zhuang, Chen Shen, Shaotian Yan, Yifei Li, Shafei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12476)  

**Abstract**: Recently, large language models (LLMs) have demonstrated impressive performance in Knowledge Graph Question Answering (KGQA) tasks, which aim to find answers based on knowledge graphs (KGs) for natural language questions. Existing LLMs-based KGQA methods typically follow the Graph Retrieval-Augmented Generation (GraphRAG) paradigm, which first retrieves reasoning paths from the large KGs, and then generates the answers based on them. However, these methods emphasize the exploration of new optimal reasoning paths in KGs while ignoring the exploitation of historical reasoning paths, which may lead to sub-optimal reasoning paths. Additionally, the complex semantics contained in questions may lead to the retrieval of inaccurate reasoning paths. To address these issues, this paper proposes a novel and training-free framework for KGQA tasks called Reward-guided Tree Search on Graph (RTSoG). RTSoG decomposes an original question into a series of simpler and well-defined sub-questions to handle the complex semantics. Then, a Self-Critic Monte Carlo Tree Search (SC-MCTS) guided by a reward model is introduced to iteratively retrieve weighted reasoning paths as contextual knowledge. Finally, it stacks the weighted reasoning paths according to their weights to generate the final answers. Extensive experiments on four datasets demonstrate the effectiveness of RTSoG. Notably, it achieves 8.7\% and 7.0\% performance improvement over the state-of-the-art method on the GrailQA and the WebQSP respectively. 

---
# Observe-R1: Unlocking Reasoning Abilities of MLLMs with Dynamic Progressive Reinforcement Learning 

**Authors**: Zirun Guo, Minjie Hong, Tao Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.12432)  

**Abstract**: Reinforcement Learning (RL) has shown promise in improving the reasoning abilities of Large Language Models (LLMs). However, the specific challenges of adapting RL to multimodal data and formats remain relatively unexplored. In this work, we present Observe-R1, a novel framework aimed at enhancing the reasoning capabilities of multimodal large language models (MLLMs). We draw inspirations from human learning progression--from simple to complex and easy to difficult, and propose a gradual learning paradigm for MLLMs. To this end, we construct the NeuraLadder dataset, which is organized and sampled according to the difficulty and complexity of data samples for RL training. To tackle multimodal tasks, we introduce a multimodal format constraint that encourages careful observation of images, resulting in enhanced visual abilities and clearer and more structured responses. Additionally, we implement a bonus reward system that favors concise, correct answers within a length constraint, alongside a dynamic weighting mechanism that prioritizes uncertain and medium-difficulty problems, ensuring that more informative samples have a greater impact on training. Our experiments with the Qwen2.5-VL-3B and Qwen2.5-VL-7B models on 20k samples from the NeuraLadder dataset show that Observe-R1 outperforms a series of larger reasoning models on both reasoning and general benchmarks, achieving superior clarity and conciseness in reasoning chains. Ablation studies validate the effectiveness of our strategies, highlighting the robustness and generalization of our approach. The dataset and code will be released at this https URL. 

---
# SGDPO: Self-Guided Direct Preference Optimization for Language Model Alignment 

**Authors**: Wenqiao Zhu, Ji Liu, Lulu Wang, Jun Wu, Yulun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12435)  

**Abstract**: Direct Preference Optimization (DPO) is broadly utilized for aligning Large Language Models (LLMs) with human values because of its flexibility. Despite its effectiveness, it has been observed that the capability of DPO to generate human-preferred response is limited and the results of DPO are far from resilient. To address these limitations, in this paper we propose a novel Self-Guided Direct Preference Optimization algorithm, i.e., SGDPO, which incorporates a pilot term to steer the gradient flow during the optimization process, allowing for fine-grained control over the updates of chosen and rejected rewards. We provide a detailed theoretical analysis of our proposed method and elucidate its operational mechanism. Furthermore, we conduct comprehensive experiments on various models and benchmarks. The extensive experimental results demonstrate the consistency between the empirical results and our theoretical analysis and confirm the effectiveness of our proposed approach (up to 9.19% higher score). 

---
# PSC: Extending Context Window of Large Language Models via Phase Shift Calibration 

**Authors**: Wenqiao Zhu, Chao Xu, Lulu Wang, Jun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12423)  

**Abstract**: Rotary Position Embedding (RoPE) is an efficient position encoding approach and is widely utilized in numerous large language models (LLMs). Recently, a lot of methods have been put forward to further expand the context window based on RoPE. The core concept of those methods is to predefine or search for a set of factors to rescale the base frequencies of RoPE. Nevertheless, it is quite a challenge for existing methods to predefine an optimal factor due to the exponential search space. In view of this, we introduce PSC (Phase Shift Calibration), a small module for calibrating the frequencies predefined by existing methods. With the employment of PSC, we demonstrate that many existing methods can be further enhanced, like PI, YaRN, and LongRoPE. We conducted extensive experiments across multiple models and tasks. The results demonstrate that (1) when PSC is enabled, the comparative reductions in perplexity increase as the context window size is varied from 16k, to 32k, and up to 64k. (2) Our approach is broadly applicable and exhibits robustness across a variety of models and tasks. The code can be found at this https URL. 

---
# EvoGPT: Enhancing Test Suite Robustness via LLM-Based Generation and Genetic Optimization 

**Authors**: Lior Broide, Roni Stern  

**Link**: [PDF](https://arxiv.org/pdf/2505.12424)  

**Abstract**: Large Language Models (LLMs) have recently emerged as promising tools for automated unit test generation. We introduce a hybrid framework called EvoGPT that integrates LLM-based test generation with evolutionary search techniques to create diverse, fault-revealing unit tests. Unit tests are initially generated with diverse temperature sampling to maximize behavioral and test suite diversity, followed by a generation-repair loop and coverage-guided assertion enhancement. The resulting test suites are evolved using genetic algorithms, guided by a fitness function prioritizing mutation score over traditional coverage metrics. This design emphasizes the primary objective of unit testing-fault detection. Evaluated on multiple open-source Java projects, EvoGPT achieves an average improvement of 10% in both code coverage and mutation score compared to LLMs and traditional search-based software testing baselines. These results demonstrate that combining LLM-driven diversity, targeted repair, and evolutionary optimization produces more effective and resilient test suites. 

---
# Table-R1: Region-based Reinforcement Learning for Table Understanding 

**Authors**: Zhenhe Wu, Jian Yang, Jiaheng Liu, Xianjie Wu, Changzai Pan, Jie Zhang, Yu Zhao, Shuangyong Song, Yongxiang Li, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12415)  

**Abstract**: Tables present unique challenges for language models due to their structured row-column interactions, necessitating specialized approaches for effective comprehension. While large language models (LLMs) have demonstrated potential in table reasoning through prompting and techniques like chain-of-thought (CoT) and program-of-thought (PoT), optimizing their performance for table question answering remains underexplored. In this paper, we introduce region-based Table-R1, a novel reinforcement learning approach that enhances LLM table understanding by integrating region evidence into reasoning steps. Our method employs Region-Enhanced Supervised Fine-Tuning (RE-SFT) to guide models in identifying relevant table regions before generating answers, incorporating textual, symbolic, and program-based reasoning. Additionally, Table-Aware Group Relative Policy Optimization (TARPO) introduces a mixed reward system to dynamically balance region accuracy and answer correctness, with decaying region rewards and consistency penalties to align reasoning steps. Experiments show that Table-R1 achieves an average performance improvement of 14.36 points across multiple base models on three benchmark datasets, even outperforming baseline models with ten times the parameters, while TARPO reduces response token consumption by 67.5% compared to GRPO, significantly advancing LLM capabilities in efficient tabular reasoning. 

---
# Wisdom from Diversity: Bias Mitigation Through Hybrid Human-LLM Crowds 

**Authors**: Axel Abels, Tom Lenaerts  

**Link**: [PDF](https://arxiv.org/pdf/2505.12349)  

**Abstract**: Despite their performance, large language models (LLMs) can inadvertently perpetuate biases found in the data they are trained on. By analyzing LLM responses to bias-eliciting headlines, we find that these models often mirror human biases. To address this, we explore crowd-based strategies for mitigating bias through response aggregation. We first demonstrate that simply averaging responses from multiple LLMs, intended to leverage the "wisdom of the crowd", can exacerbate existing biases due to the limited diversity within LLM crowds. In contrast, we show that locally weighted aggregation methods more effectively leverage the wisdom of the LLM crowd, achieving both bias mitigation and improved accuracy. Finally, recognizing the complementary strengths of LLMs (accuracy) and humans (diversity), we demonstrate that hybrid crowds containing both significantly enhance performance and further reduce biases across ethnic and gender-related contexts. 

---
# Not All Documents Are What You Need for Extracting Instruction Tuning Data 

**Authors**: Chi Zhang, Huaping Zhong, Hongtao Li, Chengliang Chai, Jiawei Hong, Yuhao Deng, Jiacheng Wang, Tian Tan, Yizhou Yan, Jiantao Qiu, Ye Yuan, Guoren Wang, Conghui He, Lei Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12250)  

**Abstract**: Instruction tuning improves the performance of large language models (LLMs), but it heavily relies on high-quality training data. Recently, LLMs have been used to synthesize instruction data using seed question-answer (QA) pairs. However, these synthesized instructions often lack diversity and tend to be similar to the input seeds, limiting their applicability in real-world scenarios. To address this, we propose extracting instruction tuning data from web corpora that contain rich and diverse knowledge. A naive solution is to retrieve domain-specific documents and extract all QA pairs from them, but this faces two key challenges: (1) extracting all QA pairs using LLMs is prohibitively expensive, and (2) many extracted QA pairs may be irrelevant to the downstream tasks, potentially degrading model performance. To tackle these issues, we introduce EQUAL, an effective and scalable data extraction framework that iteratively alternates between document selection and high-quality QA pair extraction to enhance instruction tuning. EQUAL first clusters the document corpus based on embeddings derived from contrastive learning, then uses a multi-armed bandit strategy to efficiently identify clusters that are likely to contain valuable QA pairs. This iterative approach significantly reduces computational cost while boosting model performance. Experiments on AutoMathText and StackOverflow across four downstream tasks show that EQUAL reduces computational costs by 5-10x and improves accuracy by 2.5 percent on LLaMA-3.1-8B and Mistral-7B 

---
# Community Search in Time-dependent Road-social Attributed Networks 

**Authors**: Li Ni, Hengkai Xu, Lin Mu, Yiwen Zhang, Wenjian Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.12309)  

**Abstract**: Real-world networks often involve both keywords and locations, along with travel time variations between locations due to traffic conditions. However, most existing cohesive subgraph-based community search studies utilize a single attribute, either keywords or locations, to identify communities. They do not simultaneously consider both keywords and locations, which results in low semantic or spatial cohesiveness of the detected communities, and they fail to account for variations in travel time. Additionally, these studies traverse the entire network to build efficient indexes, but the detected community only involves nodes around the query node, leading to the traversal of nodes that are not relevant to the community. Therefore, we propose the problem of discovering semantic-spatial aware k-core, which refers to a k-core with high semantic and time-dependent spatial cohesiveness containing the query node. To address this problem, we propose an exact and a greedy algorithm, both of which gradually expand outward from the query node. They are local methods that only access the local part of the attributed network near the query node rather than the entire network. Moreover, we design a method to calculate the semantic similarity between two keywords using large language models. This method alleviates the disadvantages of keyword-matching methods used in existing community search studies, such as mismatches caused by differently expressed synonyms and the presence of irrelevant words. Experimental results show that the greedy algorithm outperforms baselines in terms of structural, semantic, and time-dependent spatial cohesiveness. 

---
# Reward Inside the Model: A Lightweight Hidden-State Reward Model for LLM's Best-of-N sampling 

**Authors**: Jizhou Guo, Zhaomin Wu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12225)  

**Abstract**: High-quality reward models are crucial for unlocking the reasoning potential of large language models (LLMs), with best-of-N voting demonstrating significant performance gains. However, current reward models, which typically operate on the textual output of LLMs, are computationally expensive and parameter-heavy, limiting their real-world applications. We introduce the Efficient Linear Hidden State Reward (ELHSR) model - a novel, highly parameter-efficient approach that leverages the rich information embedded in LLM hidden states to address these issues. ELHSR systematically outperform baselines with less than 0.005% of the parameters of baselines, requiring only a few samples for training. ELHSR also achieves orders-of-magnitude efficiency improvement with significantly less time and fewer FLOPs per sample than baseline reward models. Moreover, ELHSR exhibits robust performance even when trained only on logits, extending its applicability to some closed-source LLMs. In addition, ELHSR can also be combined with traditional reward models to achieve additional performance gains. 

---
# The Tower of Babel Revisited: Multilingual Jailbreak Prompts on Closed-Source Large Language Models 

**Authors**: Linghan Huang, Haolin Jin, Zhaoge Bi, Pengyue Yang, Peizhou Zhao, Taozhao Chen, Xiongfei Wu, Lei Ma, Huaming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.12287)  

**Abstract**: Large language models (LLMs) have seen widespread applications across various domains, yet remain vulnerable to adversarial prompt injections. While most existing research on jailbreak attacks and hallucination phenomena has focused primarily on open-source models, we investigate the frontier of closed-source LLMs under multilingual attack scenarios. We present a first-of-its-kind integrated adversarial framework that leverages diverse attack techniques to systematically evaluate frontier proprietary solutions, including GPT-4o, DeepSeek-R1, Gemini-1.5-Pro, and Qwen-Max. Our evaluation spans six categories of security contents in both English and Chinese, generating 38,400 responses across 32 types of jailbreak attacks. Attack success rate (ASR) is utilized as the quantitative metric to assess performance from three dimensions: prompt design, model architecture, and language environment. Our findings suggest that Qwen-Max is the most vulnerable, while GPT-4o shows the strongest defense. Notably, prompts in Chinese consistently yield higher ASRs than their English counterparts, and our novel Two-Sides attack technique proves to be the most effective across all models. This work highlights a dire need for language-aware alignment and robust cross-lingual defenses in LLMs, and we hope it will inspire researchers, developers, and policymakers toward more robust and inclusive AI systems. 

---
# Bridging Generative and Discriminative Learning: Few-Shot Relation Extraction via Two-Stage Knowledge-Guided Pre-training 

**Authors**: Quanjiang Guo, Jinchuan Zhang, Sijie Wang, Ling Tian, Zhao Kang, Bin Yan, Weidong Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.12236)  

**Abstract**: Few-Shot Relation Extraction (FSRE) remains a challenging task due to the scarcity of annotated data and the limited generalization capabilities of existing models. Although large language models (LLMs) have demonstrated potential in FSRE through in-context learning (ICL), their general-purpose training objectives often result in suboptimal performance for task-specific relation extraction. To overcome these challenges, we propose TKRE (Two-Stage Knowledge-Guided Pre-training for Relation Extraction), a novel framework that synergistically integrates LLMs with traditional relation extraction models, bridging generative and discriminative learning paradigms. TKRE introduces two key innovations: (1) leveraging LLMs to generate explanation-driven knowledge and schema-constrained synthetic data, addressing the issue of data scarcity; and (2) a two-stage pre-training strategy combining Masked Span Language Modeling (MSLM) and Span-Level Contrastive Learning (SCL) to enhance relational reasoning and generalization. Together, these components enable TKRE to effectively tackle FSRE tasks. Comprehensive experiments on benchmark datasets demonstrate the efficacy of TKRE, achieving new state-of-the-art performance in FSRE and underscoring its potential for broader application in low-resource scenarios. \footnote{The code and data are released on this https URL. 

---
# LLM-DSE: Searching Accelerator Parameters with LLM Agents 

**Authors**: Hanyu Wang, Xinrui Wu, Zijian Ding, Su Zheng, Chengyue Wang, Tony Nowatzki, Yizhou Sun, Jason Cong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12188)  

**Abstract**: Even though high-level synthesis (HLS) tools mitigate the challenges of programming domain-specific accelerators (DSAs) by raising the abstraction level, optimizing hardware directive parameters remains a significant hurdle. Existing heuristic and learning-based methods struggle with adaptability and sample this http URL present LLM-DSE, a multi-agent framework designed specifically for optimizing HLS directives. Combining LLM with design space exploration (DSE), our explorer coordinates four agents: Router, Specialists, Arbitrator, and Critic. These multi-agent components interact with various tools to accelerate the optimization process. LLM-DSE leverages essential domain knowledge to identify efficient parameter combinations while maintaining adaptability through verbal learning from online interactions. Evaluations on the HLSyn dataset demonstrate that LLM-DSE achieves substantial $2.55\times$ performance gains over state-of-the-art methods, uncovering novel designs while reducing runtime. Ablation studies validate the effectiveness and necessity of the proposed agent interactions. Our code is open-sourced here: this https URL. 

---
# Decoding the Mind of Large Language Models: A Quantitative Evaluation of Ideology and Biases 

**Authors**: Manari Hirose, Masato Uchida  

**Link**: [PDF](https://arxiv.org/pdf/2505.12183)  

**Abstract**: The widespread integration of Large Language Models (LLMs) across various sectors has highlighted the need for empirical research to understand their biases, thought patterns, and societal implications to ensure ethical and effective use. In this study, we propose a novel framework for evaluating LLMs, focusing on uncovering their ideological biases through a quantitative analysis of 436 binary-choice questions, many of which have no definitive answer. By applying our framework to ChatGPT and Gemini, findings revealed that while LLMs generally maintain consistent opinions on many topics, their ideologies differ across models and languages. Notably, ChatGPT exhibits a tendency to change their opinion to match the questioner's opinion. Both models also exhibited problematic biases, unethical or unfair claims, which might have negative societal impacts. These results underscore the importance of addressing both ideological and ethical considerations when evaluating LLMs. The proposed framework offers a flexible, quantitative method for assessing LLM behavior, providing valuable insights for the development of more socially aligned AI systems. 

---
# Self-Destructive Language Model 

**Authors**: Yuhui Wang, Rongyi Zhu, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12186)  

**Abstract**: Harmful fine-tuning attacks pose a major threat to the security of large language models (LLMs), allowing adversaries to compromise safety guardrails with minimal harmful data. While existing defenses attempt to reinforce LLM alignment, they fail to address models' inherent "trainability" on harmful data, leaving them vulnerable to stronger attacks with increased learning rates or larger harmful datasets. To overcome this critical limitation, we introduce SEAM, a novel alignment-enhancing defense that transforms LLMs into self-destructive models with intrinsic resilience to misalignment attempts. Specifically, these models retain their capabilities for legitimate tasks while exhibiting substantial performance degradation when fine-tuned on harmful data. The protection is achieved through a novel loss function that couples the optimization trajectories of benign and harmful data, enhanced with adversarial gradient ascent to amplify the self-destructive effect. To enable practical training, we develop an efficient Hessian-free gradient estimate with theoretical error bounds. Extensive evaluation across LLMs and datasets demonstrates that SEAM creates a no-win situation for adversaries: the self-destructive models achieve state-of-the-art robustness against low-intensity attacks and undergo catastrophic performance collapse under high-intensity attacks, rendering them effectively unusable. (warning: this paper contains potentially harmful content generated by LLMs.) 

---
# LAMeTA: Intent-Aware Agentic Network Optimization via a Large AI Model-Empowered Two-Stage Approach 

**Authors**: Yinqiu Liu, Guangyuan Liu, Jiacheng Wang, Ruichen Zhang, Dusit Niyato, Geng Sun, Zehui Xiong, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.12247)  

**Abstract**: Nowadays, Generative AI (GenAI) reshapes numerous domains by enabling machines to create content across modalities. As GenAI evolves into autonomous agents capable of reasoning, collaboration, and interaction, they are increasingly deployed on network infrastructures to serve humans automatically. This emerging paradigm, known as the agentic network, presents new optimization challenges due to the demand to incorporate subjective intents of human users expressed in natural language. Traditional generic Deep Reinforcement Learning (DRL) struggles to capture intent semantics and adjust policies dynamically, thus leading to suboptimality. In this paper, we present LAMeTA, a Large AI Model (LAM)-empowered Two-stage Approach for intent-aware agentic network optimization. First, we propose Intent-oriented Knowledge Distillation (IoKD), which efficiently distills intent-understanding capabilities from resource-intensive LAMs to lightweight edge LAMs (E-LAMs) to serve end users. Second, we develop Symbiotic Reinforcement Learning (SRL), integrating E-LAMs with a policy-based DRL framework. In SRL, E-LAMs translate natural language user intents into structured preference vectors that guide both state representation and reward design. The DRL, in turn, optimizes the generative service function chain composition and E-LAM selection based on real-time network conditions, thus optimizing the subjective Quality-of-Experience (QoE). Extensive experiments conducted in an agentic network with 81 agents demonstrate that IoKD reduces mean squared error in intent prediction by up to 22.5%, while SRL outperforms conventional generic DRL by up to 23.5% in maximizing intent-aware QoE. 

---
# Personalized Author Obfuscation with Large Language Models 

**Authors**: Mohammad Shokri, Sarah Ita Levitan, Rivka Levitan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12090)  

**Abstract**: In this paper, we investigate the efficacy of large language models (LLMs) in obfuscating authorship by paraphrasing and altering writing styles. Rather than adopting a holistic approach that evaluates performance across the entire dataset, we focus on user-wise performance to analyze how obfuscation effectiveness varies across individual authors. While LLMs are generally effective, we observe a bimodal distribution of efficacy, with performance varying significantly across users. To address this, we propose a personalized prompting method that outperforms standard prompting techniques and partially mitigates the bimodality issue. 

---
# Reasoning Large Language Model Errors Arise from Hallucinating Critical Problem Features 

**Authors**: Alex Heyman, Joel Zylberberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.12151)  

**Abstract**: Large language models have recently made great strides in reasoning task performance through chain-of-thought (CoT) strategies trained via reinforcement learning; however, these "reasoning large language models" (RLLMs) remain imperfect reasoners, and understanding the frequencies and causes of their failure modes is important for both users and developers. We test o1-mini, o3-mini, DeepSeek-R1, Claude 3.7 Sonnet, Gemini 2.5 Pro Preview, and Grok 3 Mini Beta on graph coloring as a variable-complexity constraint-satisfaction logic problem, and find evidence from both error rate comparisons and CoT/explanation text analysis that RLLMs are prone to hallucinate edges not specified in the prompt's description of the graph. This phenomenon persists across multiple problem complexity levels and semantic frames, and it appears to account for a significant fraction of the incorrect answers from every tested model, and the vast majority of them for some models. Our results indicate that RLLMs may possess broader issues with misrepresentation of problem specifics, and we offer suggestions for design choices to mitigate this weakness. 

---
# Exploring Criteria of Loss Reweighting to Enhance LLM Unlearning 

**Authors**: Puning Yang, Qizhou Wang, Zhuo Huang, Tongliang Liu, Chengqi Zhang, Bo Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.11953)  

**Abstract**: Loss reweighting has shown significant benefits for machine unlearning with large language models (LLMs). However, their exact functionalities are left unclear and the optimal strategy remains an open question, thus impeding the understanding and improvement of existing methodologies. In this paper, we identify two distinct goals of loss reweighting, namely, Saturation and Importance -- the former indicates that those insufficiently optimized data should be emphasized, while the latter stresses some critical data that are most influential for loss minimization. To study their usefulness, we design specific reweighting strategies for each goal and evaluate their respective effects on unlearning. We conduct extensive empirical analyses on well-established benchmarks, and summarize some important observations as follows: (i) Saturation enhances efficacy more than importance-based reweighting, and their combination can yield additional improvements. (ii) Saturation typically allocates lower weights to data with lower likelihoods, whereas importance-based reweighting does the opposite. (iii) The efficacy of unlearning is also largely influenced by the smoothness and granularity of the weight distributions. Based on these findings, we propose SatImp, a simple reweighting method that combines the advantages of both saturation and importance. Empirical results on extensive datasets validate the efficacy of our method, potentially bridging existing research gaps and indicating directions for future research. Our code is available at this https URL. 

---
# Fine-Grained ECG-Text Contrastive Learning via Waveform Understanding Enhancement 

**Authors**: Haitao Li, Che Liu, Zhengyao Ding, Ziyi Liu, Zhengxing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11939)  

**Abstract**: Electrocardiograms (ECGs) are essential for diagnosing cardiovascular diseases. While previous ECG-text contrastive learning methods have shown promising results, they often overlook the incompleteness of the reports. Given an ECG, the report is generated by first identifying key waveform features and then inferring the final diagnosis through these features. Despite their importance, these waveform features are often not recorded in the report as intermediate results. Aligning ECGs with such incomplete reports impedes the model's ability to capture the ECG's waveform features and limits its understanding of diagnostic reasoning based on those features. To address this, we propose FG-CLEP (Fine-Grained Contrastive Language ECG Pre-training), which aims to recover these waveform features from incomplete reports with the help of large language models (LLMs), under the challenges of hallucinations and the non-bijective relationship between waveform features and diagnoses. Additionally, considering the frequent false negatives due to the prevalence of common diagnoses in ECGs, we introduce a semantic similarity matrix to guide contrastive learning. Furthermore, we adopt a sigmoid-based loss function to accommodate the multi-label nature of ECG-related tasks. Experiments on six datasets demonstrate that FG-CLEP outperforms state-of-the-art methods in both zero-shot prediction and linear probing across these datasets. 

---
# AdaCoT: Pareto-Optimal Adaptive Chain-of-Thought Triggering via Reinforcement Learning 

**Authors**: Chenwei Lou, Zewei Sun, Xinnian Liang, Meng Qu, Wei Shen, Wenqi Wang, Yuntao Li, Qingping Yang, Shuangzhi Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11896)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities but often face challenges with tasks requiring sophisticated reasoning. While Chain-of-Thought (CoT) prompting significantly enhances reasoning, it indiscriminately generates lengthy reasoning steps for all queries, leading to substantial computational costs and inefficiency, especially for simpler inputs. To address this critical issue, we introduce AdaCoT (Adaptive Chain-of-Thought), a novel framework enabling LLMs to adaptively decide when to invoke CoT. AdaCoT framed adaptive reasoning as a Pareto optimization problem that seeks to balance model performance with the costs associated with CoT invocation (both frequency and computational overhead). We propose a reinforcement learning (RL) based method, specifically utilizing Proximal Policy Optimization (PPO), to dynamically control the CoT triggering decision boundary by adjusting penalty coefficients, thereby allowing the model to determine CoT necessity based on implicit query complexity. A key technical contribution is Selective Loss Masking (SLM), designed to counteract decision boundary collapse during multi-stage RL training, ensuring robust and stable adaptive triggering. Experimental results demonstrate that AdaCoT successfully navigates the Pareto frontier, achieving substantial reductions in CoT usage for queries not requiring elaborate reasoning. For instance, on our production traffic testset, AdaCoT reduced CoT triggering rates to as low as 3.18\% and decreased average response tokens by 69.06%, while maintaining high performance on complex tasks. 

---
# MARVEL: Multi-Agent RTL Vulnerability Extraction using Large Language Models 

**Authors**: Luca Collini, Baleegh Ahmad, Joey Ah-kiow, Ramesh Karri  

**Link**: [PDF](https://arxiv.org/pdf/2505.11963)  

**Abstract**: Hardware security verification is a challenging and time-consuming task. For this purpose, design engineers may utilize tools such as formal verification, linters, and functional simulation tests, coupled with analysis and a deep understanding of the hardware design being inspected. Large Language Models (LLMs) have been used to assist during this task, either directly or in conjunction with existing tools. We improve the state of the art by proposing MARVEL, a multi-agent LLM framework for a unified approach to decision-making, tool use, and reasoning. MARVEL mimics the cognitive process of a designer looking for security vulnerabilities in RTL code. It consists of a supervisor agent that devises the security policy of the system-on-chips (SoCs) using its security documentation. It delegates tasks to validate the security policy to individual executor agents. Each executor agent carries out its assigned task using a particular strategy. Each executor agent may use one or more tools to identify potential security bugs in the design and send the results back to the supervisor agent for further analysis and confirmation. MARVEL includes executor agents that leverage formal tools, linters, simulation tests, LLM-based detection schemes, and static analysis-based checks. We test our approach on a known buggy SoC based on OpenTitan from the Hack@DATE competition. We find that 20 of the 48 issues reported by MARVEL pose security vulnerabilities. 

---
# Multilingual Collaborative Defense for Large Language Models 

**Authors**: Hongliang Li, Jinan Xu, Gengping Cui, Changhao Guan, Fengran Mo, Kaiyu Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11835)  

**Abstract**: The robustness and security of large language models (LLMs) has become a prominent research area. One notable vulnerability is the ability to bypass LLM safeguards by translating harmful queries into rare or underrepresented languages, a simple yet effective method of "jailbreaking" these models. Despite the growing concern, there has been limited research addressing the safeguarding of LLMs in multilingual scenarios, highlighting an urgent need to enhance multilingual safety. In this work, we investigate the correlation between various attack features across different languages and propose Multilingual Collaborative Defense (MCD), a novel learning method that optimizes a continuous, soft safety prompt automatically to facilitate multilingual safeguarding of LLMs. The MCD approach offers three advantages: First, it effectively improves safeguarding performance across multiple languages. Second, MCD maintains strong generalization capabilities while minimizing false refusal rates. Third, MCD mitigates the language safety misalignment caused by imbalances in LLM training corpora. To evaluate the effectiveness of MCD, we manually construct multilingual versions of commonly used jailbreak benchmarks, such as MaliciousInstruct and AdvBench, to assess various safeguarding methods. Additionally, we introduce these datasets in underrepresented (zero-shot) languages to verify the language transferability of MCD. The results demonstrate that MCD outperforms existing approaches in safeguarding against multilingual jailbreak attempts while also exhibiting strong language transfer capabilities. Our code is available at this https URL. 

---
# Not All Thoughts are Generated Equal: Efficient LLM Reasoning via Multi-Turn Reinforcement Learning 

**Authors**: Yansong Ning, Wei Li, Jun Fang, Naiqiang Tan, Hao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11827)  

**Abstract**: Compressing long chain-of-thought (CoT) from large language models (LLMs) is an emerging strategy to improve the reasoning efficiency of LLMs. Despite its promising benefits, existing studies equally compress all thoughts within a long CoT, hindering more concise and effective reasoning. To this end, we first investigate the importance of different thoughts by examining their effectiveness and efficiency in contributing to reasoning through automatic long CoT chunking and Monte Carlo rollouts. Building upon the insights, we propose a theoretically bounded metric to jointly measure the effectiveness and efficiency of different thoughts. We then propose Long$\otimes$Short, an efficient reasoning framework that enables two LLMs to collaboratively solve the problem: a long-thought LLM for more effectively generating important thoughts, while a short-thought LLM for efficiently generating remaining thoughts. Specifically, we begin by synthesizing a small amount of cold-start data to fine-tune LLMs for long-thought and short-thought reasoning styles, respectively. Furthermore, we propose a synergizing-oriented multi-turn reinforcement learning, focusing on the model self-evolution and collaboration between long-thought and short-thought LLMs. Experimental results show that our method enables Qwen2.5-7B and Llama3.1-8B to achieve comparable performance compared to DeepSeek-R1-Distill-Qwen-7B and DeepSeek-R1-Distill-Llama-8B, while reducing token length by over 80% across the MATH500, AIME24/25, AMC23, and GPQA Diamond benchmarks. Our data and code are available at this https URL. 

---
# Search-Based Correction of Reasoning Chains for Language Models 

**Authors**: Minsu Kim, Jean-Pierre Falet, Oliver E. Richardson, Xiaoyin Chen, Moksh Jain, Sungjin Ahn, Sungsoo Ahn, Yoshua Bengio  

**Link**: [PDF](https://arxiv.org/pdf/2505.11824)  

**Abstract**: Chain-of-Thought (CoT) reasoning has advanced the capabilities and transparency of language models (LMs); however, reasoning chains can contain inaccurate statements that reduce performance and trustworthiness. To address this, we introduce a new self-correction framework that augments each reasoning step in a CoT with a latent variable indicating its veracity, enabling modeling of all possible truth assignments rather than assuming correctness throughout. To efficiently explore this expanded space, we introduce Search Corrector, a discrete search algorithm over boolean-valued veracity assignments. It efficiently performs otherwise intractable inference in the posterior distribution over veracity assignments by leveraging the LM's joint likelihood over veracity and the final answer as a proxy reward. This efficient inference-time correction method facilitates supervised fine-tuning of an Amortized Corrector by providing pseudo-labels for veracity. The Amortized Corrector generalizes self-correction, enabling accurate zero-shot veracity inference in novel contexts. Empirical results demonstrate that Search Corrector reliably identifies errors in logical (ProntoQA) and mathematical reasoning (GSM8K) benchmarks. The Amortized Corrector achieves comparable zero-shot accuracy and improves final answer accuracy by up to 25%. 

---
# Retrospex: Language Agent Meets Offline Reinforcement Learning Critic 

**Authors**: Yufei Xiang, Yiqun Shen, Yeqin Zhang, Cam-Tu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11807)  

**Abstract**: Large Language Models (LLMs) possess extensive knowledge and commonsense reasoning capabilities, making them valuable for creating powerful agents. However, existing LLM agent frameworks have not fully utilized past experiences for improvement. This work introduces a new LLM-based agent framework called Retrospex, which addresses this challenge by analyzing past experiences in depth. Unlike previous approaches, Retrospex does not directly integrate experiences into the LLM's context. Instead, it combines the LLM's action likelihood with action values estimated by a Reinforcement Learning (RL) Critic, which is trained on past experiences through an offline ''retrospection'' process. Additionally, Retrospex employs a dynamic action rescoring mechanism that increases the importance of experience-based values for tasks that require more interaction with the environment. We evaluate Retrospex in ScienceWorld, ALFWorld and Webshop environments, demonstrating its advantages over strong, contemporary baselines. 

---
# HARDMath2: A Benchmark for Applied Mathematics Built by Students as Part of a Graduate Class 

**Authors**: James V. Roggeveen, Erik Y. Wang, Will Flintoft, Peter Donets, Lucy S. Nathwani, Nickholas Gutierrez, David Ettel, Anton Marius Graf, Siddharth Dandavate, Arjun Nageswaran, Raglan Ward, Ava Williamson, Anne Mykland, Kacper K. Migacz, Yijun Wang, Egemen Bostan, Duy Thuc Nguyen, Zhe He, Marc L. Descoteaux, Felix Yeung, Shida Liu, Jorge García Ponce, Luke Zhu, Yuyang Chen, Ekaterina S. Ivshina, Miguel Fernandez, Minjae Kim, Kennan Gumbs, Matthew Scott Tan, Russell Yang, Mai Hoang, David Brown, Isabella A. Silveira, Lavon Sykes, Ahmed Roman, William Fredenberg, Yiming Chen, Lucas Martin, Yixing Tang, Kelly Werker Smith, Hongyu Liao, Logan G. Wilson, Alexander Dazhen Cai, Andrea Elizabeth Biju, Michael P. Brenner  

**Link**: [PDF](https://arxiv.org/pdf/2505.11774)  

**Abstract**: Large language models (LLMs) have shown remarkable progress in mathematical problem-solving, but evaluation has largely focused on problems that have exact analytical solutions or involve formal proofs, often overlooking approximation-based problems ubiquitous in applied science and engineering. To fill this gap, we build on prior work and present HARDMath2, a dataset of 211 original problems covering the core topics in an introductory graduate applied math class, including boundary-layer analysis, WKB methods, asymptotic solutions of nonlinear partial differential equations, and the asymptotics of oscillatory integrals. This dataset was designed and verified by the students and instructors of a core graduate applied mathematics course at Harvard. We build the dataset through a novel collaborative environment that challenges students to write and refine difficult problems consistent with the class syllabus, peer-validate solutions, test different models, and automatically check LLM-generated solutions against their own answers and numerical ground truths. Evaluation results show that leading frontier models still struggle with many of the problems in the dataset, highlighting a gap in the mathematical reasoning skills of current LLMs. Importantly, students identified strategies to create increasingly difficult problems by interacting with the models and exploiting common failure modes. This back-and-forth with the models not only resulted in a richer and more challenging benchmark but also led to qualitative improvements in the students' understanding of the course material, which is increasingly important as we enter an age where state-of-the-art language models can solve many challenging problems across a wide domain of fields. 

---
# Internal Causal Mechanisms Robustly Predict Language Model Out-of-Distribution Behaviors 

**Authors**: Jing Huang, Junyi Tao, Thomas Icard, Diyi Yang, Christopher Potts  

**Link**: [PDF](https://arxiv.org/pdf/2505.11770)  

**Abstract**: Interpretability research now offers a variety of techniques for identifying abstract internal mechanisms in neural networks. Can such techniques be used to predict how models will behave on out-of-distribution examples? In this work, we provide a positive answer to this question. Through a diverse set of language modeling tasks--including symbol manipulation, knowledge retrieval, and instruction following--we show that the most robust features for correctness prediction are those that play a distinctive causal role in the model's behavior. Specifically, we propose two methods that leverage causal mechanisms to predict the correctness of model outputs: counterfactual simulation (checking whether key causal variables are realized) and value probing (using the values of those variables to make predictions). Both achieve high AUC-ROC in distribution and outperform methods that rely on causal-agnostic features in out-of-distribution settings, where predicting model behaviors is more crucial. Our work thus highlights a novel and significant application for internal causal analysis of language models. 

---
# Towards Universal Semantics With Large Language Models 

**Authors**: Raymond Baartmans, Matthew Raffel, Rahul Vikram, Aiden Deringer, Lizhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11764)  

**Abstract**: The Natural Semantic Metalanguage (NSM) is a linguistic theory based on a universal set of semantic primes: simple, primitive word-meanings that have been shown to exist in most, if not all, languages of the world. According to this framework, any word, regardless of complexity, can be paraphrased using these primes, revealing a clear and universally translatable meaning. These paraphrases, known as explications, can offer valuable applications for many natural language processing (NLP) tasks, but producing them has traditionally been a slow, manual process. In this work, we present the first study of using large language models (LLMs) to generate NSM explications. We introduce automatic evaluation methods, a tailored dataset for training and evaluation, and fine-tuned models for this task. Our 1B and 8B models outperform GPT-4o in producing accurate, cross-translatable explications, marking a significant step toward universal semantic representation with LLMs and opening up new possibilities for applications in semantic analysis, translation, and beyond. 

---
# RLAP: A Reinforcement Learning Enhanced Adaptive Planning Framework for Multi-step NLP Task Solving 

**Authors**: Zepeng Ding, Dixuan Wang, Ziqin Luo, Guochao Jiang, Deqing Yang, Jiaqing Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11893)  

**Abstract**: Multi-step planning has been widely employed to enhance the performance of large language models (LLMs) on downstream natural language processing (NLP) tasks, which decomposes the original task into multiple subtasks and guide LLMs to solve them sequentially without additional training. When addressing task instances, existing methods either preset the order of steps or attempt multiple paths at each step. However, these methods overlook instances' linguistic features and rely on the intrinsic planning capabilities of LLMs to evaluate intermediate feedback and then select subtasks, resulting in suboptimal outcomes. To better solve multi-step NLP tasks with LLMs, in this paper we propose a Reinforcement Learning enhanced Adaptive Planning framework (RLAP). In our framework, we model an NLP task as a Markov decision process (MDP) and employ an LLM directly into the environment. In particular, a lightweight Actor model is trained to estimate Q-values for natural language sequences consisting of states and actions through reinforcement learning. Therefore, during sequential planning, the linguistic features of each sequence in the MDP can be taken into account, and the Actor model interacts with the LLM to determine the optimal order of subtasks for each task instance. We apply RLAP on three different types of NLP tasks and conduct extensive experiments on multiple datasets to verify RLAP's effectiveness and robustness. 

---
# OMAC: A Broad Optimization Framework for LLM-Based Multi-Agent Collaboration 

**Authors**: Shijun Li, Hilaf Hasson, Joydeep Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11765)  

**Abstract**: Agents powered by advanced large language models (LLMs) have demonstrated impressive capabilities across diverse complex applications. Recently, Multi-Agent Systems (MAS), wherein multiple agents collaborate and communicate with each other, have exhibited enhanced capabilities in complex tasks, such as high-quality code generation and arithmetic reasoning. However, the development of such systems often relies on handcrafted methods, and the literature on systematic design and optimization of LLM-based MAS remains limited.
In this work, we introduce OMAC, a general framework designed for holistic optimization of LLM-based MAS. Specifically, we identify five key optimization dimensions for MAS, encompassing both agent functionality and collaboration structure. Building upon these dimensions, we first propose a general algorithm, utilizing two actors termed the Semantic Initializer and the Contrastive Comparator, to optimize any single dimension. Then, we present an algorithm for joint optimization across multiple dimensions. Extensive experiments demonstrate the superior performance of OMAC on code generation, arithmetic reasoning, and general reasoning tasks against state-of-the-art approaches. 

---
# Cloud-Based AI Systems: Leveraging Large Language Models for Intelligent Fault Detection and Autonomous Self-Healing 

**Authors**: Cheng Ji, Huaiying Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11743)  

**Abstract**: With the rapid development of cloud computing systems and the increasing complexity of their infrastructure, intelligent mechanisms to detect and mitigate failures in real time are becoming increasingly important. Traditional methods of failure detection are often difficult to cope with the scale and dynamics of modern cloud environments. In this study, we propose a novel AI framework based on Massive Language Model (LLM) for intelligent fault detection and self-healing mechanisms in cloud systems. The model combines existing machine learning fault detection algorithms with LLM's natural language understanding capabilities to process and parse system logs, error reports, and real-time data streams through semantic context. The method adopts a multi-level architecture, combined with supervised learning for fault classification and unsupervised learning for anomaly detection, so that the system can predict potential failures before they occur and automatically trigger the self-healing mechanism. Experimental results show that the proposed model is significantly better than the traditional fault detection system in terms of fault detection accuracy, system downtime reduction and recovery speed. 

---
# ZeroTuning: Unlocking the Initial Token's Power to Enhance Large Language Models Without Training 

**Authors**: Feijiang Han, Xiaodong Yu, Jianheng Tang, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2505.11739)  

**Abstract**: Recently, training-free methods for improving large language models (LLMs) have attracted growing interest, with token-level attention tuning emerging as a promising and interpretable direction. However, existing methods typically rely on auxiliary mechanisms to identify important or irrelevant task-specific tokens, introducing potential bias and limiting applicability. In this paper, we uncover a surprising and elegant alternative: the semantically empty initial token is a powerful and underexplored control point for optimizing model behavior. Through theoretical analysis, we show that tuning the initial token's attention sharpens or flattens the attention distribution over subsequent tokens, and its role as an attention sink amplifies this effect. Empirically, we find that: (1) tuning its attention improves LLM performance more effectively than tuning other task-specific tokens; (2) the effect follows a consistent trend across layers, with earlier layers having greater impact, but varies across attention heads, with different heads showing distinct preferences in how they attend to this token. Based on these findings, we propose ZeroTuning, a training-free approach that improves LLM performance by applying head-specific attention adjustments to this special token. Despite tuning only one token, ZeroTuning achieves higher performance on text classification, multiple-choice, and multi-turn conversation tasks across models such as Llama, Qwen, and DeepSeek. For example, ZeroTuning improves Llama-3.1-8B by 11.71% on classification, 2.64% on QA tasks, and raises its multi-turn score from 7.804 to 7.966. The method is also robust to limited resources, few-shot settings, long contexts, quantization, decoding strategies, and prompt variations. Our work sheds light on a previously overlooked control point in LLMs, offering new insights into both inference-time tuning and model interpretability. 

---
# Token-Level Uncertainty Estimation for Large Language Model Reasoning 

**Authors**: Tunyu Zhang, Haizhou Shi, Yibin Wang, Hengyi Wang, Xiaoxiao He, Zhuowei Li, Haoxian Chen, Ligong Han, Kai Xu, Huan Zhang, Dimitris Metaxas, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11737)  

**Abstract**: While Large Language Models (LLMs) have demonstrated impressive capabilities, their output quality remains inconsistent across various application scenarios, making it difficult to identify trustworthy responses, especially in complex tasks requiring multi-step reasoning. In this paper, we propose a token-level uncertainty estimation framework to enable LLMs to self-assess and self-improve their generation quality in mathematical reasoning. Specifically, we introduce low-rank random weight perturbation to LLM decoding, generating predictive distributions that we use to estimate token-level uncertainties. We then aggregate these uncertainties to reflect semantic uncertainty of the generated sequences. Experiments on mathematical reasoning datasets of varying difficulty demonstrate that our token-level uncertainty metrics strongly correlate with answer correctness and model robustness. Additionally, we explore using uncertainty to directly enhance the model's reasoning performance through multiple generations and the particle filtering algorithm. Our approach consistently outperforms existing uncertainty estimation methods, establishing effective uncertainty estimation as a valuable tool for both evaluating and improving reasoning generation in LLMs. 

---
# Efficient Uncertainty Estimation via Distillation of Bayesian Large Language Models 

**Authors**: Harshil Vejendla, Haizhou Shi, Yibin Wang, Tunyu Zhang, Huan Zhang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11731)  

**Abstract**: Recent advances in uncertainty estimation for Large Language Models (LLMs) during downstream adaptation have addressed key challenges of reliability and simplicity. However, existing Bayesian methods typically require multiple sampling iterations during inference, creating significant efficiency issues that limit practical deployment. In this paper, we investigate the possibility of eliminating the need for test-time sampling for LLM uncertainty estimation. Specifically, when given an off-the-shelf Bayesian LLM, we distill its aligned confidence into a non-Bayesian student LLM by minimizing the divergence between their predictive distributions. Unlike typical calibration methods, our distillation is carried out solely on the training dataset without the need of an additional validation dataset. This simple yet effective approach achieves N-times more efficient uncertainty estimation during testing, where N is the number of samples traditionally required by Bayesian LLMs. Our extensive experiments demonstrate that uncertainty estimation capabilities on training data can successfully generalize to unseen test data through our distillation technique, consistently producing results comparable to (or even better than) state-of-the-art Bayesian LLMs. 

---
# Multilingual Prompt Engineering in Large Language Models: A Survey Across NLP Tasks 

**Authors**: Shubham Vatsal, Harsh Dubey, Aditi Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.11665)  

**Abstract**: Large language models (LLMs) have demonstrated impressive performance across a wide range of Natural Language Processing (NLP) tasks. However, ensuring their effectiveness across multiple languages presents unique challenges. Multilingual prompt engineering has emerged as a key approach to enhance LLMs' capabilities in diverse linguistic settings without requiring extensive parameter re-training or fine-tuning. With growing interest in multilingual prompt engineering over the past two to three years, researchers have explored various strategies to improve LLMs' performance across languages and NLP tasks. By crafting structured natural language prompts, researchers have successfully extracted knowledge from LLMs across different languages, making these techniques an accessible pathway for a broader audience, including those without deep expertise in machine learning, to harness the capabilities of LLMs. In this paper, we survey and categorize different multilingual prompting techniques based on the NLP tasks they address across a diverse set of datasets that collectively span around 250 languages. We further highlight the LLMs employed, present a taxonomy of approaches and discuss potential state-of-the-art (SoTA) methods for specific multilingual datasets. Additionally, we derive a range of insights across language families and resource levels (high-resource vs. low-resource), including analyses such as the distribution of NLP tasks by language resource type and the frequency of prompting methods across different language families. Our survey reviews 36 research papers covering 39 prompting techniques applied to 30 multilingual NLP tasks, with the majority of these studies published in the last two years. 

---
# PeerGuard: Defending Multi-Agent Systems Against Backdoor Attacks Through Mutual Reasoning 

**Authors**: Falong Fan, Xi Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.11642)  

**Abstract**: Multi-agent systems leverage advanced AI models as autonomous agents that interact, cooperate, or compete to complete complex tasks across applications such as robotics and traffic management. Despite their growing importance, safety in multi-agent systems remains largely underexplored, with most research focusing on single AI models rather than interacting agents. This work investigates backdoor vulnerabilities in multi-agent systems and proposes a defense mechanism based on agent interactions. By leveraging reasoning abilities, each agent evaluates responses from others to detect illogical reasoning processes, which indicate poisoned agents. Experiments on LLM-based multi-agent systems, including ChatGPT series and Llama 3, demonstrate the effectiveness of the proposed method, achieving high accuracy in identifying poisoned agents while minimizing false positives on clean agents. We believe this work provides insights into multi-agent system safety and contributes to the development of robust, trustworthy AI interactions. 

---
# Steering Risk Preferences in Large Language Models by Aligning Behavioral and Neural Representations 

**Authors**: Jian-Qiao Zhu, Haijiang Yan, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.11615)  

**Abstract**: Changing the behavior of large language models (LLMs) can be as straightforward as editing the Transformer's residual streams using appropriately constructed "steering vectors." These modifications to internal neural activations, a form of representation engineering, offer an effective and targeted means of influencing model behavior without retraining or fine-tuning the model. But how can such steering vectors be systematically identified? We propose a principled approach for uncovering steering vectors by aligning latent representations elicited through behavioral methods (specifically, Markov chain Monte Carlo with LLMs) with their neural counterparts. To evaluate this approach, we focus on extracting latent risk preferences from LLMs and steering their risk-related outputs using the aligned representations as steering vectors. We show that the resulting steering vectors successfully and reliably modulate LLM outputs in line with the targeted behavior. 

---
# Concept-Guided Interpretability via Neural Chunking 

**Authors**: Shuchen Wu, Stephan Alaniz, Shyamgopal Karthik, Peter Dayan, Eric Schulz, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2505.11576)  

**Abstract**: Neural networks are often black boxes, reflecting the significant challenge of understanding their internal workings. We propose a different perspective that challenges the prevailing view: rather than being inscrutable, neural networks exhibit patterns in their raw population activity that mirror regularities in the training data. We refer to this as the Reflection Hypothesis and provide evidence for this phenomenon in both simple recurrent neural networks (RNNs) and complex large language models (LLMs). Building on this insight, we propose to leverage cognitively-inspired methods of chunking to segment high-dimensional neural population dynamics into interpretable units that reflect underlying concepts. We propose three methods to extract these emerging entities, complementing each other based on label availability and dimensionality. Discrete sequence chunking (DSC) creates a dictionary of entities; population averaging (PA) extracts recurring entities that correspond to known labels; and unsupervised chunk discovery (UCD) can be used when labels are absent. We demonstrate the effectiveness of these methods in extracting entities across varying model sizes, ranging from inducing compositionality in RNNs to uncovering recurring neural population states in large models with diverse architectures, and illustrate their advantage over other methods. Throughout, we observe a robust correspondence between the extracted entities and concrete or abstract concepts. Artificially inducing the extracted entities in neural populations effectively alters the network's generation of associated concepts. Our work points to a new direction for interpretability, one that harnesses both cognitive principles and the structure of naturalistic data to reveal the hidden computations of complex learning systems, gradually transforming them from black boxes into systems we can begin to understand. 

---
# InfiJanice: Joint Analysis and In-situ Correction Engine for Quantization-Induced Math Degradation in Large Language Models 

**Authors**: Zhen Li, Yupeng Su, Songmiao Wang, Runming Yang, Congkai Xie, Aofan Liu, Ming Li, Jiannong Cao, Yuan Xie, Ngai Wong, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11574)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance on complex reasoning benchmarks such as GSM8K, MATH, and AIME. However, the substantial computational demands of these tasks pose significant challenges for real-world deployment. Model quantization has emerged as a promising approach to reduce memory footprint and inference latency by representing weights and activations with lower bit-widths. In this work, we conduct a comprehensive study of mainstream quantization methods(e.g., AWQ, GPTQ, SmoothQuant) on the most popular open-sourced models (e.g., Qwen2.5, LLaMA3 series), and reveal that quantization can degrade mathematical reasoning accuracy by up to 69.81%. To better understand this degradation, we develop an automated assignment and judgment pipeline that qualitatively categorizes failures into four error types and quantitatively identifies the most impacted reasoning capabilities. Building on these findings, we employ an automated data-curation pipeline to construct a compact "Silver Bullet" datasets. Training a quantized model on as few as 332 carefully selected examples for just 3-5 minutes on a single GPU is enough to restore its reasoning accuracy to match that of the full-precision baseline. 

---
# ACSE-Eval: Can LLMs threat model real-world cloud infrastructure? 

**Authors**: Sarthak Munshi, Swapnil Pathak, Sonam Ghatode, Thenuga Priyadarshini, Dhivya Chandramouleeswaran, Ashutosh Rana  

**Link**: [PDF](https://arxiv.org/pdf/2505.11565)  

**Abstract**: While Large Language Models have shown promise in cybersecurity applications, their effectiveness in identifying security threats within cloud deployments remains unexplored. This paper introduces AWS Cloud Security Engineering Eval, a novel dataset for evaluating LLMs cloud security threat modeling capabilities. ACSE-Eval contains 100 production grade AWS deployment scenarios, each featuring detailed architectural specifications, Infrastructure as Code implementations, documented security vulnerabilities, and associated threat modeling parameters. Our dataset enables systemic assessment of LLMs abilities to identify security risks, analyze attack vectors, and propose mitigation strategies in cloud environments. Our evaluations on ACSE-Eval demonstrate that GPT 4.1 and Gemini 2.5 Pro excel at threat identification, with Gemini 2.5 Pro performing optimally in 0-shot scenarios and GPT 4.1 showing superior results in few-shot settings. While GPT 4.1 maintains a slight overall performance advantage, Claude 3.7 Sonnet generates the most semantically sophisticated threat models but struggles with threat categorization and generalization. To promote reproducibility and advance research in automated cybersecurity threat analysis, we open-source our dataset, evaluation metrics, and methodologies. 

---
# Tool-Aided Evolutionary LLM for Generative Policy Toward Efficient Resource Management in Wireless Federated Learning 

**Authors**: Chongyang Tan, Ruoqi Wen, Rongpeng Li, Zhifeng Zhao, Ekram Hossain, Honggang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11570)  

**Abstract**: Federated Learning (FL) enables distributed model training across edge devices in a privacy-friendly manner. However, its efficiency heavily depends on effective device selection and high-dimensional resource allocation in dynamic and heterogeneous wireless environments. Conventional methods demand a confluence of domain-specific expertise, extensive hyperparameter tuning, and/or heavy interaction cost. This paper proposes a Tool-aided Evolutionary Large Language Model (T-ELLM) framework to generate a qualified policy for device selection in a wireless FL environment. Unlike conventional optimization methods, T-ELLM leverages natural language-based scenario prompts to enhance generalization across varying network conditions. The framework decouples the joint optimization problem mathematically, enabling tractable learning of device selection policies while delegating resource allocation to convex optimization tools. To improve adaptability, T-ELLM integrates a sample-efficient, model-based virtual learning environment that captures the relationship between device selection and learning performance, facilitating subsequent group relative policy optimization. This concerted approach reduces reliance on real-world interactions, minimizing communication overhead while maintaining high-fidelity decision-making. Theoretical analysis proves that the discrepancy between virtual and real environments is bounded, ensuring the advantage function learned in the virtual environment maintains a provably small deviation from real-world conditions. Experimental results demonstrate that T-ELLM outperforms benchmark methods in energy efficiency and exhibits robust adaptability to environmental changes. 

---
# Assessing Collective Reasoning in Multi-Agent LLMs via Hidden Profile Tasks 

**Authors**: Yuxuan Li, Aoi Naito, Hirokazu Shirado  

**Link**: [PDF](https://arxiv.org/pdf/2505.11556)  

**Abstract**: Multi-agent systems built on large language models (LLMs) promise enhanced problem-solving through distributed information integration, but also risk replicating collective reasoning failures observed in human groups. Yet, no theory-grounded benchmark exists to systematically evaluate such failures. In this paper, we introduce the Hidden Profile paradigm from social psychology as a diagnostic testbed for multi-agent LLM systems. By distributing critical information asymmetrically across agents, the paradigm reveals how inter-agent dynamics support or hinder collective reasoning. We first formalize the paradigm for multi-agent decision-making under distributed knowledge and instantiate it as a benchmark with nine tasks spanning diverse scenarios, including adaptations from prior human studies. We then conduct experiments with GPT-4.1 and five other leading LLMs, including reasoning-enhanced variants, showing that multi-agent systems across all models fail to match the accuracy of single agents given complete information. While agents' collective performance is broadly comparable to that of human groups, nuanced behavioral differences emerge, such as increased sensitivity to social desirability. Finally, we demonstrate the paradigm's diagnostic utility by exploring a cooperation-contradiction trade-off in multi-agent LLM systems. We find that while cooperative agents are prone to over-coordination in collective settings, increased contradiction impairs group convergence. This work contributes a reproducible framework for evaluating multi-agent LLM systems and motivates future research on artificial collective intelligence and human-AI interaction. 

---
# One Shot Dominance: Knowledge Poisoning Attack on Retrieval-Augmented Generation Systems 

**Authors**: Zhiyuan Chang, Xiaojun Jia, Mingyang Li, Junjie Wang, Yuekai Huang, Qing Wang, Ziyou Jiang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11548)  

**Abstract**: Large Language Models (LLMs) enhanced with Retrieval-Augmented Generation (RAG) have shown improved performance in generating accurate responses. However, the dependence on external knowledge bases introduces potential security vulnerabilities, particularly when these knowledge bases are publicly accessible and modifiable. Poisoning attacks on knowledge bases for RAG systems face two fundamental challenges: the injected malicious content must compete with multiple authentic documents retrieved by the retriever, and LLMs tend to trust retrieved information that aligns with their internal memorized knowledge. Previous works attempt to address these challenges by injecting multiple malicious documents, but such saturation attacks are easily detectable and impractical in real-world scenarios. To enable the effective single document poisoning attack, we propose AuthChain, a novel knowledge poisoning attack method that leverages Chain-of-Evidence theory and authority effect to craft more convincing poisoned documents. AuthChain generates poisoned content that establishes strong evidence chains and incorporates authoritative statements, effectively overcoming the interference from both authentic documents and LLMs' internal knowledge. Extensive experiments across six popular LLMs demonstrate that AuthChain achieves significantly higher attack success rates while maintaining superior stealthiness against RAG defense mechanisms compared to state-of-the-art baselines. 

---
# AI-generated Text Detection: A Multifaceted Approach to Binary and Multiclass Classification 

**Authors**: Harika Abburi, Sanmitra Bhattacharya, Edward Bowen, Nirmala Pudota  

**Link**: [PDF](https://arxiv.org/pdf/2505.11550)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in generating text that closely resembles human writing across a wide range of styles and genres. However, such capabilities are prone to potential misuse, such as fake news generation, spam email creation, and misuse in academic assignments. As a result, accurate detection of AI-generated text and identification of the model that generated it are crucial for maintaining the responsible use of LLMs. In this work, we addressed two sub-tasks put forward by the Defactify workshop under AI-Generated Text Detection shared task at the Association for the Advancement of Artificial Intelligence (AAAI 2025): Task A involved distinguishing between human-authored or AI-generated text, while Task B focused on attributing text to its originating language model. For each task, we proposed two neural architectures: an optimized model and a simpler variant. For Task A, the optimized neural architecture achieved fifth place with $F1$ score of 0.994, and for Task B, the simpler neural architecture also ranked fifth place with $F1$ score of 0.627. 

---
# On Technique Identification and Threat-Actor Attribution using LLMs and Embedding Models 

**Authors**: Kyla Guru, Robert J. Moss, Mykel J. Kochenderfer  

**Link**: [PDF](https://arxiv.org/pdf/2505.11547)  

**Abstract**: Attribution of cyber-attacks remains a complex but critical challenge for cyber defenders. Currently, manual extraction of behavioral indicators from dense forensic documentation causes significant attribution delays, especially following major incidents at the international scale. This research evaluates large language models (LLMs) for cyber-attack attribution based on behavioral indicators extracted from forensic documentation. We test OpenAI's GPT-4 and text-embedding-3-large for identifying threat actors' tactics, techniques, and procedures (TTPs) by comparing LLM-generated TTPs against human-generated data from MITRE ATT&CK Groups. Our framework then identifies TTPs from text using vector embedding search and builds profiles to attribute new attacks for a machine learning model to learn. Key contributions include: (1) assessing off-the-shelf LLMs for TTP extraction and attribution, and (2) developing an end-to-end pipeline from raw CTI documents to threat-actor prediction. This research finds that standard LLMs generate TTP datasets with noise, resulting in a low similarity to human-generated datasets. However, the TTPs generated are similar in frequency to those within the existing MITRE datasets. Additionally, although these TTPs are different than human-generated datasets, our work demonstrates that they still prove useful for training a model that performs above baseline on attribution. Project code and files are contained here: this https URL. 

---
# What Prompts Don't Say: Understanding and Managing Underspecification in LLM Prompts 

**Authors**: Chenyang Yang, Yike Shi, Qianou Ma, Michael Xieyang Liu, Christian Kästner, Tongshuang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13360)  

**Abstract**: Building LLM-powered software requires developers to communicate their requirements through natural language, but developer prompts are frequently underspecified, failing to fully capture many user-important requirements. In this paper, we present an in-depth analysis of prompt underspecification, showing that while LLMs can often (41.1%) guess unspecified requirements by default, such behavior is less robust: Underspecified prompts are 2x more likely to regress over model or prompt changes, sometimes with accuracy drops by more than 20%. We then demonstrate that simply adding more requirements to a prompt does not reliably improve performance, due to LLMs' limited instruction-following capabilities and competing constraints, and standard prompt optimizers do not offer much help. To address this, we introduce novel requirements-aware prompt optimization mechanisms that can improve performance by 4.8% on average over baselines that naively specify everything in the prompt. Beyond prompt optimization, we envision that effectively managing prompt underspecification requires a broader process, including proactive requirements discovery, evaluation, and monitoring. 

---
# MR. Judge: Multimodal Reasoner as a Judge 

**Authors**: Renjie Pi, Felix Bai, Qibin Chen, Simon Wang, Jiulong Shan, Kieran Liu, Meng Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.13403)  

**Abstract**: The paradigm of using Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) as evaluative judges has emerged as an effective approach in RLHF and inference-time scaling. In this work, we propose Multimodal Reasoner as a Judge (MR. Judge), a paradigm for empowering general-purpose MLLMs judges with strong reasoning capabilities. Instead of directly assigning scores for each response, we formulate the judgement process as a reasoning-inspired multiple-choice problem. Specifically, the judge model first conducts deliberate reasoning covering different aspects of the responses and eventually selects the best response from them. This reasoning process not only improves the interpretibility of the judgement, but also greatly enhances the performance of MLLM judges. To cope with the lack of questions with scored responses, we propose the following strategy to achieve automatic annotation: 1) Reverse Response Candidates Synthesis: starting from a supervised fine-tuning (SFT) dataset, we treat the original response as the best candidate and prompt the MLLM to generate plausible but flawed negative candidates. 2) Text-based reasoning extraction: we carefully design a data synthesis pipeline for distilling the reasoning capability from a text-based reasoning model, which is adopted to enable the MLLM judges to regain complex reasoning ability via warm up supervised fine-tuning. Experiments demonstrate that our MR. Judge is effective across a wide range of tasks. Specifically, our MR. Judge-7B surpasses GPT-4o by 9.9% on VL-RewardBench, and improves performance on MM-Vet during inference-time scaling by up to 7.7%. 

---
# Investigating the Vulnerability of LLM-as-a-Judge Architectures to Prompt-Injection Attacks 

**Authors**: Narek Maloyan, Bislan Ashinov, Dmitry Namiot  

**Link**: [PDF](https://arxiv.org/pdf/2505.13348)  

**Abstract**: Large Language Models (LLMs) are increasingly employed as evaluators (LLM-as-a-Judge) for assessing the quality of machine-generated text. This paradigm offers scalability and cost-effectiveness compared to human annotation. However, the reliability and security of such systems, particularly their robustness against adversarial manipulations, remain critical concerns. This paper investigates the vulnerability of LLM-as-a-Judge architectures to prompt-injection attacks, where malicious inputs are designed to compromise the judge's decision-making process. We formalize two primary attack strategies: Comparative Undermining Attack (CUA), which directly targets the final decision output, and Justification Manipulation Attack (JMA), which aims to alter the model's generated reasoning. Using the Greedy Coordinate Gradient (GCG) optimization method, we craft adversarial suffixes appended to one of the responses being compared. Experiments conducted on the MT-Bench Human Judgments dataset with open-source instruction-tuned LLMs (Qwen2.5-3B-Instruct and Falcon3-3B-Instruct) demonstrate significant susceptibility. The CUA achieves an Attack Success Rate (ASR) exceeding 30\%, while JMA also shows notable effectiveness. These findings highlight substantial vulnerabilities in current LLM-as-a-Judge systems, underscoring the need for robust defense mechanisms and further research into adversarial evaluation and trustworthiness in LLM-based assessment frameworks. 

---
# Sense and Sensitivity: Examining the Influence of Semantic Recall on Long Context Code Reasoning 

**Authors**: Adam Štorek, Mukur Gupta, Samira Hajizadeh, Prashast Srivastava, Suman Jana  

**Link**: [PDF](https://arxiv.org/pdf/2505.13353)  

**Abstract**: Although modern Large Language Models (LLMs) support extremely large contexts, their effectiveness in utilizing long context for code reasoning remains unclear. This paper investigates LLM reasoning ability over code snippets within large repositories and how it relates to their recall ability. Specifically, we differentiate between lexical code recall (verbatim retrieval) and semantic code recall (remembering what the code does). To measure semantic recall, we propose SemTrace, a code reasoning technique where the impact of specific statements on output is attributable and unpredictable. We also present a method to quantify semantic recall sensitivity in existing benchmarks. Our evaluation of state-of-the-art LLMs reveals a significant drop in code reasoning accuracy as a code snippet approaches the middle of the input context, particularly with techniques requiring high semantic recall like SemTrace. Moreover, we find that lexical recall varies by granularity, with models excelling at function retrieval but struggling with line-by-line recall. Notably, a disconnect exists between lexical and semantic recall, suggesting different underlying mechanisms. Finally, our findings indicate that current code reasoning benchmarks may exhibit low semantic recall sensitivity, potentially underestimating LLM challenges in leveraging in-context information. 

---
# Dementia Through Different Eyes: Explainable Modeling of Human and LLM Perceptions for Early Awareness 

**Authors**: Lotem Peled-Cohen, Maya Zadok, Nitay Calderon, Hila Gonen, Roi Reichart  

**Link**: [PDF](https://arxiv.org/pdf/2505.13418)  

**Abstract**: Cognitive decline often surfaces in language years before diagnosis. It is frequently non-experts, such as those closest to the patient, who first sense a change and raise concern. As LLMs become integrated into daily communication and used over prolonged periods, it may even be an LLM that notices something is off. But what exactly do they notice--and should be noticing--when making that judgment? This paper investigates how dementia is perceived through language by non-experts. We presented transcribed picture descriptions to non-expert humans and LLMs, asking them to intuitively judge whether each text was produced by someone healthy or with dementia. We introduce an explainable method that uses LLMs to extract high-level, expert-guided features representing these picture descriptions, and use logistic regression to model human and LLM perceptions and compare with clinical diagnoses. Our analysis reveals that human perception of dementia is inconsistent and relies on a narrow, and sometimes misleading, set of cues. LLMs, by contrast, draw on a richer, more nuanced feature set that aligns more closely with clinical patterns. Still, both groups show a tendency toward false negatives, frequently overlooking dementia cases. Through our interpretable framework and the insights it provides, we hope to help non-experts better recognize the linguistic signs that matter. 

---
# GUARD: Generation-time LLM Unlearning via Adaptive Restriction and Detection 

**Authors**: Zhijie Deng, Chris Yuhao Liu, Zirui Pang, Xinlei He, Lei Feng, Qi Xuan, Zhaowei Zhu, Jiaheng Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.13312)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in memorizing vast amounts of knowledge across diverse domains. However, the ability to selectively forget specific knowledge is critical for ensuring the safety and compliance of deployed models. Existing unlearning efforts typically fine-tune the model with resources such as forget data, retain data, and a calibration model. These additional gradient steps blur the decision boundary between forget and retain knowledge, making unlearning often at the expense of overall performance. To avoid the negative impact of fine-tuning, it would be better to unlearn solely at inference time by safely guarding the model against generating responses related to the forget target, without destroying the fluency of text generation. In this work, we propose Generation-time Unlearning via Adaptive Restriction and Detection (GUARD), a framework that enables dynamic unlearning during LLM generation. Specifically, we first employ a prompt classifier to detect unlearning targets and extract the corresponding forbidden token. We then dynamically penalize and filter candidate tokens during generation using a combination of token matching and semantic matching, effectively preventing the model from leaking the forgotten content. Experimental results on copyright content unlearning tasks over the Harry Potter dataset and the MUSE benchmark, as well as entity unlearning tasks on the TOFU dataset, demonstrate that GUARD achieves strong forget quality across various tasks while causing almost no degradation to the LLM's general capabilities, striking an excellent trade-off between forgetting and utility. 

---
# SMOTExT: SMOTE meets Large Language Models 

**Authors**: Mateusz Bystroński, Mikołaj Hołysz, Grzegorz Piotrowski, Nitesh V. Chawla, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2505.13434)  

**Abstract**: Data scarcity and class imbalance are persistent challenges in training robust NLP models, especially in specialized domains or low-resource settings. We propose a novel technique, SMOTExT, that adapts the idea of Synthetic Minority Over-sampling (SMOTE) to textual data. Our method generates new synthetic examples by interpolating between BERT-based embeddings of two existing examples and then decoding the resulting latent point into text with xRAG architecture. By leveraging xRAG's cross-modal retrieval-generation framework, we can effectively turn interpolated vectors into coherent text. While this is preliminary work supported by qualitative outputs only, the method shows strong potential for knowledge distillation and data augmentation in few-shot settings. Notably, our approach also shows promise for privacy-preserving machine learning: in early experiments, training models solely on generated data achieved comparable performance to models trained on the original dataset. This suggests a viable path toward safe and effective learning under data protection constraints. 

---
# From Automation to Autonomy: A Survey on Large Language Models in Scientific Discovery 

**Authors**: Tianshi Zheng, Zheye Deng, Hong Ting Tsang, Weiqi Wang, Jiaxin Bai, Zihao Wang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.13259)  

**Abstract**: Large Language Models (LLMs) are catalyzing a paradigm shift in scientific discovery, evolving from task-specific automation tools into increasingly autonomous agents and fundamentally redefining research processes and human-AI collaboration. This survey systematically charts this burgeoning field, placing a central focus on the changing roles and escalating capabilities of LLMs in science. Through the lens of the scientific method, we introduce a foundational three-level taxonomy-Tool, Analyst, and Scientist-to delineate their escalating autonomy and evolving responsibilities within the research lifecycle. We further identify pivotal challenges and future research trajectories such as robotic automation, self-improvement, and ethical governance. Overall, this survey provides a conceptual architecture and strategic foresight to navigate and shape the future of AI-driven scientific discovery, fostering both rapid innovation and responsible advancement. Github Repository: this https URL. 

---
# Effective and Transparent RAG: Adaptive-Reward Reinforcement Learning for Decision Traceability 

**Authors**: Jingyi Ren, Yekun Xu, Xiaolong Wang, Weitao Li, Weizhi Ma, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.13258)  

**Abstract**: Retrieval-Augmented Generation (RAG) has significantly improved the performance of large language models (LLMs) on knowledge-intensive domains. However, although RAG achieved successes across distinct domains, there are still some unsolved challenges: 1) Effectiveness. Existing research mainly focuses on developing more powerful RAG retrievers, but how to enhance the generator's (LLM's) ability to utilize the retrieved information for reasoning and generation? 2) Transparency. Most RAG methods ignore which retrieved content actually contributes to the reasoning process, resulting in a lack of interpretability and visibility. To address this, we propose ARENA (Adaptive-Rewarded Evidence Navigation Agent), a transparent RAG generator framework trained via reinforcement learning (RL) with our proposed rewards. Based on the structured generation and adaptive reward calculation, our RL-based training enables the model to identify key evidence, perform structured reasoning, and generate answers with interpretable decision traces. Applied to Qwen2.5-7B-Instruct and Llama3.1-8B-Instruct, abundant experiments with various RAG baselines demonstrate that our model achieves 10-30% improvements on all multi-hop QA datasets, which is comparable with the SOTA Commercially-developed LLMs (e.g., OpenAI-o1, DeepSeek-R1). Further analyses show that ARENA has strong flexibility to be adopted on new datasets without extra training. Our models and codes are publicly released. 

---
# A Case Study of Cross-Lingual Zero-Shot Generalization for Classical Languages in LLMs 

**Authors**: V.S.D.S.Mahesh Akavarapu, Hrishikesh Terdalkar, Pramit Bhattacharyya, Shubhangi Agarwal, Vishakha Deulgaonkar, Pralay Manna, Chaitali Dangarikar, Arnab Bhattacharya  

**Link**: [PDF](https://arxiv.org/pdf/2505.13173)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable generalization capabilities across diverse tasks and languages. In this study, we focus on natural language understanding in three classical languages -- Sanskrit, Ancient Greek and Latin -- to investigate the factors affecting cross-lingual zero-shot generalization. First, we explore named entity recognition and machine translation into English. While LLMs perform equal to or better than fine-tuned baselines on out-of-domain data, smaller models often struggle, especially with niche or abstract entity types. In addition, we concentrate on Sanskrit by presenting a factoid question-answering (QA) dataset and show that incorporating context via retrieval-augmented generation approach significantly boosts performance. In contrast, we observe pronounced performance drops for smaller LLMs across these QA tasks. These results suggest model scale as an important factor influencing cross-lingual generalization. Assuming that models used such as GPT-4o and Llama-3.1 are not instruction fine-tuned on classical languages, our findings provide insights into how LLMs may generalize on these languages and their consequent utility in classical studies. 

---
# SeedBench: A Multi-task Benchmark for Evaluating Large Language Models in Seed Science 

**Authors**: Jie Ying, Zihong Chen, Zhefan Wang, Wanli Jiang, Chenyang Wang, Zhonghang Yuan, Haoyang Su, Huanjun Kong, Fan Yang, Nanqing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.13220)  

**Abstract**: Seed science is essential for modern agriculture, directly influencing crop yields and global food security. However, challenges such as interdisciplinary complexity and high costs with limited returns hinder progress, leading to a shortage of experts and insufficient technological support. While large language models (LLMs) have shown promise across various fields, their application in seed science remains limited due to the scarcity of digital resources, complex gene-trait relationships, and the lack of standardized benchmarks. To address this gap, we introduce SeedBench -- the first multi-task benchmark specifically designed for seed science. Developed in collaboration with domain experts, SeedBench focuses on seed breeding and simulates key aspects of modern breeding processes. We conduct a comprehensive evaluation of 26 leading LLMs, encompassing proprietary, open-source, and domain-specific fine-tuned models. Our findings not only highlight the substantial gaps between the power of LLMs and the real-world seed science problems, but also make a foundational step for research on LLMs for seed design. 

---
# Understanding Cross-Lingual Inconsistency in Large Language Models 

**Authors**: Zheng Wei Lim, Alham Fikri Aji, Trevor Cohn  

**Link**: [PDF](https://arxiv.org/pdf/2505.13141)  

**Abstract**: Large language models (LLMs) are demonstrably capable of cross-lingual transfer, but can produce inconsistent output when prompted with the same queries written in different languages. To understand how language models are able to generalize knowledge from one language to the others, we apply the logit lens to interpret the implicit steps taken by LLMs to solve multilingual multi-choice reasoning questions. We find LLMs predict inconsistently and are less accurate because they rely on subspaces of individual languages, rather than working in a shared semantic space. While larger models are more multilingual, we show their hidden states are more likely to dissociate from the shared representation compared to smaller models, but are nevertheless more capable of retrieving knowledge embedded across different languages. Finally, we demonstrate that knowledge sharing can be modulated by steering the models' latent processing towards the shared semantic space. We find reinforcing utilization of the shared space improves the models' multilingual reasoning performance, as a result of more knowledge transfer from, and better output consistency with English. 

---
# MA-COIR: Leveraging Semantic Search Index and Generative Models for Ontology-Driven Biomedical Concept Recognition 

**Authors**: Shanshan Liu, Noriki Nishida, Rumana Ferdous Munne, Narumi Tokunaga, Yuki Yamagata, Kouji Kozaki, Yuji Matsumoto  

**Link**: [PDF](https://arxiv.org/pdf/2505.12964)  

**Abstract**: Recognizing biomedical concepts in the text is vital for ontology refinement, knowledge graph construction, and concept relationship discovery. However, traditional concept recognition methods, relying on explicit mention identification, often fail to capture complex concepts not explicitly stated in the text. To overcome this limitation, we introduce MA-COIR, a framework that reformulates concept recognition as an indexing-recognition task. By assigning semantic search indexes (ssIDs) to concepts, MA-COIR resolves ambiguities in ontology entries and enhances recognition efficiency. Using a pretrained BART-based model fine-tuned on small datasets, our approach reduces computational requirements to facilitate adoption by domain experts. Furthermore, we incorporate large language models (LLMs)-generated queries and synthetic data to improve recognition in low-resource settings. Experimental results on three scenarios (CDR, HPO, and HOIP) highlight the effectiveness of MA-COIR in recognizing both explicit and implicit concepts without the need for mention-level annotations during inference, advancing ontology-driven concept recognition in biomedical domain applications. Our code and constructed data are available at this https URL. 

---
# GuRE:Generative Query REwriter for Legal Passage Retrieval 

**Authors**: Daehee Kim, Deokhyung Kang, Jonghwi Kim, Sangwon Ryu, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.12950)  

**Abstract**: Legal Passage Retrieval (LPR) systems are crucial as they help practitioners save time when drafting legal arguments. However, it remains an underexplored avenue. One primary reason is the significant vocabulary mismatch between the query and the target passage. To address this, we propose a simple yet effective method, the Generative query REwriter (GuRE). We leverage the generative capabilities of Large Language Models (LLMs) by training the LLM for query rewriting. "Rewritten queries" help retrievers to retrieve target passages by mitigating vocabulary mismatch. Experimental results show that GuRE significantly improves performance in a retriever-agnostic manner, outperforming all baseline methods. Further analysis reveals that different training objectives lead to distinct retrieval behaviors, making GuRE more suitable than direct retriever fine-tuning for real-world applications. Codes are avaiable at this http URL. 

---
# On the Thinking-Language Modeling Gap in Large Language Models 

**Authors**: Chenxi Liu, Yongqiang Chen, Tongliang Liu, James Cheng, Bo Han, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12896)  

**Abstract**: System 2 reasoning is one of the defining characteristics of intelligence, which requires slow and logical thinking. Human conducts System 2 reasoning via the language of thoughts that organizes the reasoning process as a causal sequence of mental language, or thoughts. Recently, it has been observed that System 2 reasoning can be elicited from Large Language Models (LLMs) pre-trained on large-scale natural languages. However, in this work, we show that there is a significant gap between the modeling of languages and thoughts. As language is primarily a tool for humans to share knowledge and thinking, modeling human language can easily absorb language biases into LLMs deviated from the chain of thoughts in minds. Furthermore, we show that the biases will mislead the eliciting of "thoughts" in LLMs to focus only on a biased part of the premise. To this end, we propose a new prompt technique termed Language-of-Thoughts (LoT) to demonstrate and alleviate this gap. Instead of directly eliciting the chain of thoughts from partial information, LoT instructs LLMs to adjust the order and token used for the expressions of all the relevant information. We show that the simple strategy significantly reduces the language modeling biases in LLMs and improves the performance of LLMs across a variety of reasoning tasks. 

---
# GAP: Graph-Assisted Prompts for Dialogue-based Medication Recommendation 

**Authors**: Jialun Zhong, Yanzeng Li, Sen Hu, Yang Zhang, Teng Xu, Lei Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.12888)  

**Abstract**: Medication recommendations have become an important task in the healthcare domain, especially in measuring the accuracy and safety of medical dialogue systems (MDS). Different from the recommendation task based on electronic health records (EHRs), dialogue-based medication recommendations require research on the interaction details between patients and doctors, which is crucial but may not exist in EHRs. Recent advancements in large language models (LLM) have extended the medical dialogue domain. These LLMs can interpret patients' intent and provide medical suggestions including medication recommendations, but some challenges are still worth attention. During a multi-turn dialogue, LLMs may ignore the fine-grained medical information or connections across the dialogue turns, which is vital for providing accurate suggestions. Besides, LLMs may generate non-factual responses when there is a lack of domain-specific knowledge, which is more risky in the medical domain. To address these challenges, we propose a \textbf{G}raph-\textbf{A}ssisted \textbf{P}rompts (\textbf{GAP}) framework for dialogue-based medication recommendation. It extracts medical concepts and corresponding states from dialogue to construct an explicitly patient-centric graph, which can describe the neglected but important information. Further, combined with external medical knowledge graphs, GAP can generate abundant queries and prompts, thus retrieving information from multiple sources to reduce the non-factual responses. We evaluate GAP on a dialogue-based medication recommendation dataset and further explore its potential in a more difficult scenario, dynamically diagnostic interviewing. Extensive experiments demonstrate its competitive performance when compared with strong baselines. 

---
# Decentralized Arena: Towards Democratic and Scalable Automatic Evaluation of Language Models 

**Authors**: Yanbin Yin, Kun Zhou, Zhen Wang, Xiangdong Zhang, Yifei Shao, Shibo Hao, Yi Gu, Jieyuan Liu, Somanshu Singla, Tianyang Liu, Eric P. Xing, Zhengzhong Liu, Haojian Jin, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12808)  

**Abstract**: The recent explosion of large language models (LLMs), each with its own general or specialized strengths, makes scalable, reliable benchmarking more urgent than ever. Standard practices nowadays face fundamental trade-offs: closed-ended question-based benchmarks (eg MMLU) struggle with saturation as newer models emerge, while crowd-sourced leaderboards (eg Chatbot Arena) rely on costly and slow human judges. Recently, automated methods (eg LLM-as-a-judge) shed light on the scalability, but risk bias by relying on one or a few "authority" models. To tackle these issues, we propose Decentralized Arena (dearena), a fully automated framework leveraging collective intelligence from all LLMs to evaluate each other. It mitigates single-model judge bias by democratic, pairwise evaluation, and remains efficient at scale through two key components: (1) a coarse-to-fine ranking algorithm for fast incremental insertion of new models with sub-quadratic complexity, and (2) an automatic question selection strategy for the construction of new evaluation dimensions. Across extensive experiments across 66 LLMs, dearena attains up to 97% correlation with human judgements, while significantly reducing the cost. Our code and data will be publicly released on this https URL. 

---
# Contrastive Prompting Enhances Sentence Embeddings in LLMs through Inference-Time Steering 

**Authors**: Zifeng Cheng, Zhonghui Wang, Yuchen Fu, Zhiwei Jiang, Yafeng Yin, Cong Wang, Qing Gu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12831)  

**Abstract**: Extracting sentence embeddings from large language models (LLMs) is a practical direction, as it requires neither additional data nor fine-tuning. Previous studies usually focus on prompt engineering to guide LLMs to encode the core semantic information of the sentence into the embedding of the last token. However, the last token in these methods still encodes an excess of non-essential information, such as stop words, limiting its encoding capacity. To this end, we propose a Contrastive Prompting (CP) method that introduces an extra auxiliary prompt to elicit better sentence embedding. By contrasting with the auxiliary prompt, CP can steer existing prompts to encode the core semantics of the sentence, rather than non-essential information. CP is a plug-and-play inference-time intervention method that can be combined with various prompt-based methods. Extensive experiments on Semantic Textual Similarity (STS) tasks and downstream classification tasks demonstrate that our method can improve the performance of existing prompt-based methods across different LLMs. Our code will be released at this https URL. 

---
# ToTRL: Unlock LLM Tree-of-Thoughts Reasoning Potential through Puzzles Solving 

**Authors**: Haoyuan Wu, Xueyi Chen, Rui Ming, Jilong Gao, Shoubo Hu, Zhuolun He, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12717)  

**Abstract**: Large language models (LLMs) demonstrate significant reasoning capabilities, particularly through long chain-of-thought (CoT) processes, which can be elicited by reinforcement learning (RL). However, prolonged CoT reasoning presents limitations, primarily verbose outputs due to excessive introspection. The reasoning process in these LLMs often appears to follow a trial-and-error methodology rather than a systematic, logical deduction. In contrast, tree-of-thoughts (ToT) offers a conceptually more advanced approach by modeling reasoning as an exploration within a tree structure. This reasoning structure facilitates the parallel generation and evaluation of multiple reasoning branches, allowing for the active identification, assessment, and pruning of unproductive paths. This process can potentially lead to improved performance and reduced token costs. Building upon the long CoT capability of LLMs, we introduce tree-of-thoughts RL (ToTRL), a novel on-policy RL framework with a rule-based reward. ToTRL is designed to guide LLMs in developing the parallel ToT strategy based on the sequential CoT strategy. Furthermore, we employ LLMs as players in a puzzle game during the ToTRL training process. Solving puzzle games inherently necessitates exploring interdependent choices and managing multiple constraints, which requires the construction and exploration of a thought tree, providing challenging tasks for cultivating the ToT reasoning capability. Our empirical evaluations demonstrate that our ToTQwen3-8B model, trained with our ToTRL, achieves significant improvement in performance and reasoning efficiency on complex reasoning tasks. 

---
# On-Policy Optimization with Group Equivalent Preference for Multi-Programming Language Understanding 

**Authors**: Haoyuan Wu, Rui Ming, Jilong Gao, Hangyu Zhao, Xueyi Chen, Yikai Yang, Haisheng Zheng, Zhuolun He, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12723)  

**Abstract**: Large language models (LLMs) achieve remarkable performance in code generation tasks. However, a significant performance disparity persists between popular programming languages (e.g., Python, C++) and others. To address this capability gap, we leverage the code translation task to train LLMs, thereby facilitating the transfer of coding proficiency across diverse programming languages. Moreover, we introduce OORL for training, a novel reinforcement learning (RL) framework that integrates on-policy and off-policy strategies. Within OORL, on-policy RL is applied during code translation, guided by a rule-based reward signal derived from unit tests. Complementing this coarse-grained rule-based reward, we propose Group Equivalent Preference Optimization (GEPO), a novel preference optimization method. Specifically, GEPO trains the LLM using intermediate representations (IRs) groups. LLMs can be guided to discern IRs equivalent to the source code from inequivalent ones, while also utilizing signals about the mutual equivalence between IRs within the group. This process allows LLMs to capture nuanced aspects of code functionality. By employing OORL for training with code translation tasks, LLMs improve their recognition of code functionality and their understanding of the relationships between code implemented in different languages. Extensive experiments demonstrate that our OORL for LLMs training with code translation tasks achieves significant performance improvements on code benchmarks across multiple programming languages. 

---
# ReEx-SQL: Reasoning with Execution-Aware Reinforcement Learning for Text-to-SQL 

**Authors**: Yaxun Dai, Wenxuan Xie, Xialie Zhuang, Tianyu Yang, Yiying Yang, Haiqin Yang, Yuhang Zhao, Pingfu Chao, Wenhao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12768)  

**Abstract**: In Text-to-SQL, execution feedback is essential for guiding large language models (LLMs) to reason accurately and generate reliable SQL queries. However, existing methods treat execution feedback solely as a post-hoc signal for correction or selection, failing to integrate it into the generation process. This limitation hinders their ability to address reasoning errors as they occur, ultimately reducing query accuracy and robustness. To address this issue, we propose ReEx-SQL (Reasoning with Execution-Aware Reinforcement Learning), a framework for Text-to-SQL that enables models to interact with the database during decoding and dynamically adjust their reasoning based on execution feedback. ReEx-SQL introduces an execution-aware reasoning paradigm that interleaves intermediate SQL execution into reasoning paths, facilitating context-sensitive revisions. It achieves this through structured prompts with markup tags and a stepwise rollout strategy that integrates execution feedback into each stage of generation. To supervise policy learning, we develop a composite reward function that includes an exploration reward, explicitly encouraging effective database interaction. Additionally, ReEx-SQL adopts a tree-based decoding strategy to support exploratory reasoning, enabling dynamic expansion of alternative reasoning paths. Notably, ReEx-SQL achieves 88.8% on Spider and 64.9% on BIRD at the 7B scale, surpassing the standard reasoning baseline by 2.7% and 2.6%, respectively. It also shows robustness, achieving 85.2% on Spider-Realistic with leading performance. In addition, its tree-structured decoding improves efficiency and performance over linear decoding, reducing inference time by 51.9% on the BIRD development set. 

---
# EAVIT: Efficient and Accurate Human Value Identification from Text data via LLMs 

**Authors**: Wenhao Zhu, Yuhang Xie, Guojie Song, Xin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12792)  

**Abstract**: The rapid evolution of large language models (LLMs) has revolutionized various fields, including the identification and discovery of human values within text data. While traditional NLP models, such as BERT, have been employed for this task, their ability to represent textual data is significantly outperformed by emerging LLMs like GPTs. However, the performance of online LLMs often degrades when handling long contexts required for value identification, which also incurs substantial computational costs. To address these challenges, we propose EAVIT, an efficient and accurate framework for human value identification that combines the strengths of both locally fine-tunable and online black-box LLMs. Our framework employs a value detector - a small, local language model - to generate initial value estimations. These estimations are then used to construct concise input prompts for online LLMs, enabling accurate final value identification. To train the value detector, we introduce explanation-based training and data generation techniques specifically tailored for value identification, alongside sampling strategies to optimize the brevity of LLM input prompts. Our approach effectively reduces the number of input tokens by up to 1/6 compared to directly querying online LLMs, while consistently outperforming traditional NLP methods and other LLM-based strategies. 

---
# PromptPrism: A Linguistically-Inspired Taxonomy for Prompts 

**Authors**: Sullam Jeoung, Yueyan Chen, Yi Zhang, Shuai Wang, Haibo Ding, Lin Lee Cheong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12592)  

**Abstract**: Prompts are the interface for eliciting the capabilities of large language models (LLMs). Understanding their structure and components is critical for analyzing LLM behavior and optimizing performance. However, the field lacks a comprehensive framework for systematic prompt analysis and understanding. We introduce PromptPrism, a linguistically-inspired taxonomy that enables prompt analysis across three hierarchical levels: functional structure, semantic component, and syntactic pattern. We show the practical utility of PromptPrism by applying it to three applications: (1) a taxonomy-guided prompt refinement approach that automatically improves prompt quality and enhances model performance across a range of tasks; (2) a multi-dimensional dataset profiling method that extracts and aggregates structural, semantic, and syntactic characteristics from prompt datasets, enabling comprehensive analysis of prompt distributions and patterns; (3) a controlled experimental framework for prompt sensitivity analysis by quantifying the impact of semantic reordering and delimiter modifications on LLM performance. Our experimental results validate the effectiveness of our taxonomy across these applications, demonstrating that PromptPrism provides a foundation for refining, profiling, and analyzing prompts. 

---
# Improving Multilingual Language Models by Aligning Representations through Steering 

**Authors**: Omar Mahmoud, Buddhika Laknath Semage, Thommen George Karimpanal, Santu Rana  

**Link**: [PDF](https://arxiv.org/pdf/2505.12584)  

**Abstract**: In this paper, we investigate how large language models (LLMS) process non-English tokens within their layer representations, an open question despite significant advancements in the field. Using representation steering, specifically by adding a learned vector to a single model layer's activations, we demonstrate that steering a single model layer can notably enhance performance. Our analysis shows that this approach achieves results comparable to translation baselines and surpasses state of the art prompt optimization methods. Additionally, we highlight how advanced techniques like supervised fine tuning (\textsc{sft}) and reinforcement learning from human feedback (\textsc{rlhf}) improve multilingual capabilities by altering representation spaces. We further illustrate how these methods align with our approach to reshaping LLMS layer representations. 

---
# Disambiguation in Conversational Question Answering in the Era of LLM: A Survey 

**Authors**: Md Mehrab Tanjim, Yeonjun In, Xiang Chen, Victor S. Bursztyn, Ryan A. Rossi, Sungchul Kim, Guang-Jie Ren, Vaishnavi Muppala, Shun Jiang, Yongsung Kim, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.12543)  

**Abstract**: Ambiguity remains a fundamental challenge in Natural Language Processing (NLP) due to the inherent complexity and flexibility of human language. With the advent of Large Language Models (LLMs), addressing ambiguity has become even more critical due to their expanded capabilities and applications. In the context of Conversational Question Answering (CQA), this paper explores the definition, forms, and implications of ambiguity for language driven systems, particularly in the context of LLMs. We define key terms and concepts, categorize various disambiguation approaches enabled by LLMs, and provide a comparative analysis of their advantages and disadvantages. We also explore publicly available datasets for benchmarking ambiguity detection and resolution techniques and highlight their relevance for ongoing research. Finally, we identify open problems and future research directions, proposing areas for further investigation. By offering a comprehensive review of current research on ambiguities and disambiguation with LLMs, we aim to contribute to the development of more robust and reliable language systems. 

---
# ESC-Judge: A Framework for Comparing Emotional Support Conversational Agents 

**Authors**: Navid Madani, Rohini Srihari  

**Link**: [PDF](https://arxiv.org/pdf/2505.12531)  

**Abstract**: Large language models (LLMs) increasingly power mental-health chatbots, yet the field still lacks a scalable, theory-grounded way to decide which model is most effective to deploy. We present ESC-Judge, the first end-to-end evaluation framework that (i) grounds head-to-head comparisons of emotional-support LLMs in Clara Hill's established Exploration-Insight-Action counseling model, providing a structured and interpretable view of performance, and (ii) fully automates the evaluation pipeline at scale. ESC-Judge operates in three stages: first, it synthesizes realistic help-seeker roles by sampling empirically salient attributes such as stressors, personality, and life history; second, it has two candidate support agents conduct separate sessions with the same role, isolating model-specific strategies; and third, it asks a specialized judge LLM to express pairwise preferences across rubric-anchored skills that span the Exploration, Insight, and Action spectrum. In our study, ESC-Judge matched PhD-level annotators on 85 percent of Exploration, 83 percent of Insight, and 86 percent of Action decisions, demonstrating human-level reliability at a fraction of the cost. All code, prompts, synthetic roles, transcripts, and judgment scripts are released to promote transparent progress in emotionally supportive AI. 

---
# Extracting memorized pieces of (copyrighted) books from open-weight language models 

**Authors**: A. Feder Cooper, Aaron Gokaslan, Amy B. Cyphert, Christopher De Sa, Mark A. Lemley, Daniel E. Ho, Percy Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12546)  

**Abstract**: Plaintiffs and defendants in copyright lawsuits over generative AI often make sweeping, opposing claims about the extent to which large language models (LLMs) have memorized plaintiffs' protected expression. Drawing on adversarial ML and copyright law, we show that these polarized positions dramatically oversimplify the relationship between memorization and copyright. To do so, we leverage a recent probabilistic extraction technique to extract pieces of the Books3 dataset from 13 open-weight LLMs. Through numerous experiments, we show that it's possible to extract substantial parts of at least some books from different LLMs. This is evidence that the LLMs have memorized the extracted text; this memorized content is copied inside the model parameters. But the results are complicated: the extent of memorization varies both by model and by book. With our specific experiments, we find that the largest LLMs don't memorize most books -- either in whole or in part. However, we also find that Llama 3.1 70B memorizes some books, like Harry Potter and 1984, almost entirely. We discuss why our results have significant implications for copyright cases, though not ones that unambiguously favor either side. 

---
# Towards Reliable and Interpretable Traffic Crash Pattern Prediction and Safety Interventions Using Customized Large Language Models 

**Authors**: Yang Zhao, Pu Wang, Yibo Zhao, Hongru Du, Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12545)  

**Abstract**: Predicting crash events is crucial for understanding crash distributions and their contributing factors, thereby enabling the design of proactive traffic safety policy interventions. However, existing methods struggle to interpret the complex interplay among various sources of traffic crash data, including numeric characteristics, textual reports, crash imagery, environmental conditions, and driver behavior records. As a result, they often fail to capture the rich semantic information and intricate interrelationships embedded in these diverse data sources, limiting their ability to identify critical crash risk factors. In this research, we propose TrafficSafe, a framework that adapts LLMs to reframe crash prediction and feature attribution as text-based reasoning. A multi-modal crash dataset including 58,903 real-world reports together with belonged infrastructure, environmental, driver, and vehicle information is collected and textualized into TrafficSafe Event Dataset. By customizing and fine-tuning LLMs on this dataset, the TrafficSafe LLM achieves a 42% average improvement in F1-score over baselines. To interpret these predictions and uncover contributing factors, we introduce TrafficSafe Attribution, a sentence-level feature attribution framework enabling conditional risk analysis. Findings show that alcohol-impaired driving is the leading factor in severe crashes, with aggressive and impairment-related behaviors having nearly twice the contribution for severe crashes compared to other driver behaviors. Furthermore, TrafficSafe Attribution highlights pivotal features during model training, guiding strategic crash data collection for iterative performance improvements. The proposed TrafficSafe offers a transformative leap in traffic safety research, providing a blueprint for translating advanced AI technologies into responsible, actionable, and life-saving outcomes. 

---
# What are they talking about? Benchmarking Large Language Models for Knowledge-Grounded Discussion Summarization 

**Authors**: Weixiao Zhou, Junnan Zhu, Gengyao Li, Xianfu Cheng, Xinnian Liang, Feifei Zhai, Zhoujun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.12474)  

**Abstract**: In this work, we investigate the performance of LLMs on a new task that requires combining discussion with background knowledge for summarization. This aims to address the limitation of outside observer confusion in existing dialogue summarization systems due to their reliance solely on discussion information. To achieve this, we model the task output as background and opinion summaries and define two standardized summarization patterns. To support assessment, we introduce the first benchmark comprising high-quality samples consistently annotated by human experts and propose a novel hierarchical evaluation framework with fine-grained, interpretable metrics. We evaluate 12 LLMs under structured-prompt and self-reflection paradigms. Our findings reveal: (1) LLMs struggle with background summary retrieval, generation, and opinion summary integration. (2) Even top LLMs achieve less than 69% average performance across both patterns. (3) Current LLMs lack adequate self-evaluation and self-correction capabilities for this task. 

---
# KG-QAGen: A Knowledge-Graph-Based Framework for Systematic Question Generation and Long-Context LLM Evaluation 

**Authors**: Nikita Tatarinov, Vidhyakshaya Kannan, Haricharana Srinivasa, Arnav Raj, Harpreet Singh Anand, Varun Singh, Aditya Luthra, Ravij Lade, Agam Shah, Sudheer Chava  

**Link**: [PDF](https://arxiv.org/pdf/2505.12495)  

**Abstract**: The increasing context length of modern language models has created a need for evaluating their ability to retrieve and process information across extensive documents. While existing benchmarks test long-context capabilities, they often lack a structured way to systematically vary question complexity. We introduce KG-QAGen (Knowledge-Graph-based Question-Answer Generation), a framework that (1) extracts QA pairs at multiple complexity levels (2) by leveraging structured representations of financial agreements (3) along three key dimensions -- multi-hop retrieval, set operations, and answer plurality -- enabling fine-grained assessment of model performance across controlled difficulty levels. Using this framework, we construct a dataset of 20,139 QA pairs (the largest number among the long-context benchmarks) and open-source a part of it. We evaluate 13 proprietary and open-source LLMs and observe that even the best-performing models are struggling with set-based comparisons and multi-hop logical inference. Our analysis reveals systematic failure modes tied to semantic misinterpretation and inability to handle implicit relations. 

---
# Learning to Play Like Humans: A Framework for LLM Adaptation in Interactive Fiction Games 

**Authors**: Jinming Zhang, Yunfei Long  

**Link**: [PDF](https://arxiv.org/pdf/2505.12439)  

**Abstract**: Interactive Fiction games (IF games) are where players interact through natural language commands. While recent advances in Artificial Intelligence agents have reignited interest in IF games as a domain for studying decision-making, existing approaches prioritize task-specific performance metrics over human-like comprehension of narrative context and gameplay logic. This work presents a cognitively inspired framework that guides Large Language Models (LLMs) to learn and play IF games systematically. Our proposed **L**earning to **P**lay **L**ike **H**umans (LPLH) framework integrates three key components: (1) structured map building to capture spatial and narrative relationships, (2) action learning to identify context-appropriate commands, and (3) feedback-driven experience analysis to refine decision-making over time. By aligning LLMs-based agents' behavior with narrative intent and commonsense constraints, LPLH moves beyond purely exploratory strategies to deliver more interpretable, human-like performance. Crucially, this approach draws on cognitive science principles to more closely simulate how human players read, interpret, and respond within narrative worlds. As a result, LPLH reframes the IF games challenge as a learning problem for LLMs-based agents, offering a new path toward robust, context-aware gameplay in complex text-based environments. 

---
# R1dacted: Investigating Local Censorship in DeepSeek's R1 Language Model 

**Authors**: Ali Naseh, Harsh Chaudhari, Jaechul Roh, Mingshi Wu, Alina Oprea, Amir Houmansadr  

**Link**: [PDF](https://arxiv.org/pdf/2505.12625)  

**Abstract**: DeepSeek recently released R1, a high-performing large language model (LLM) optimized for reasoning tasks. Despite its efficient training pipeline, R1 achieves competitive performance, even surpassing leading reasoning models like OpenAI's o1 on several benchmarks. However, emerging reports suggest that R1 refuses to answer certain prompts related to politically sensitive topics in China. While existing LLMs often implement safeguards to avoid generating harmful or offensive outputs, R1 represents a notable shift - exhibiting censorship-like behavior on politically charged queries. In this paper, we investigate this phenomenon by first introducing a large-scale set of heavily curated prompts that get censored by R1, covering a range of politically sensitive topics, but are not censored by other models. We then conduct a comprehensive analysis of R1's censorship patterns, examining their consistency, triggers, and variations across topics, prompt phrasing, and context. Beyond English-language queries, we explore censorship behavior in other languages. We also investigate the transferability of censorship to models distilled from the R1 language model. Finally, we propose techniques for bypassing or removing this censorship. Our findings reveal possible additional censorship integration likely shaped by design choices during training or alignment, raising concerns about transparency, bias, and governance in language model deployment. 

---
# SLOT: Sample-specific Language Model Optimization at Test-time 

**Authors**: Yang Hu, Xingyu Zhang, Xueji Fang, Zhiyang Chen, Xiao Wang, Huatian Zhang, Guojun Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12392)  

**Abstract**: We propose SLOT (Sample-specific Language Model Optimization at Test-time), a novel and parameter-efficient test-time inference approach that enhances a language model's ability to more accurately respond to individual prompts. Existing Large Language Models (LLMs) often struggle with complex instructions, leading to poor performances on those not well represented among general samples. To address this, SLOT conducts few optimization steps at test-time to update a light-weight sample-specific parameter vector. It is added to the final hidden layer before the output head, and enables efficient adaptation by caching the last layer features during per-sample optimization. By minimizing the cross-entropy loss on the input prompt only, SLOT helps the model better aligned with and follow each given instruction. In experiments, we demonstrate that our method outperforms the compared models across multiple benchmarks and LLMs. For example, Qwen2.5-7B with SLOT achieves an accuracy gain of 8.6% on GSM8K from 57.54% to 66.19%, while DeepSeek-R1-Distill-Llama-70B with SLOT achieves a SOTA accuracy of 68.69% on GPQA among 70B-level models. Our code is available at this https URL. 

---
# UniEdit: A Unified Knowledge Editing Benchmark for Large Language Models 

**Authors**: Qizhou Chen, Dakan Wang, Taolin Zhang, Zaoming Yan, Chengsong You, Chengyu Wang, Xiaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2505.12345)  

**Abstract**: Model editing aims to enhance the accuracy and reliability of large language models (LLMs) by efficiently adjusting their internal parameters. Currently, most LLM editing datasets are confined to narrow knowledge domains and cover a limited range of editing evaluation. They often overlook the broad scope of editing demands and the diversity of ripple effects resulting from edits. In this context, we introduce UniEdit, a unified benchmark for LLM editing grounded in open-domain knowledge. First, we construct editing samples by selecting entities from 25 common domains across five major categories, utilizing the extensive triple knowledge available in open-domain knowledge graphs to ensure comprehensive coverage of the knowledge domains. To address the issues of generality and locality in editing, we design an Neighborhood Multi-hop Chain Sampling (NMCS) algorithm to sample subgraphs based on a given knowledge piece to entail comprehensive ripple effects to evaluate. Finally, we employ proprietary LLMs to convert the sampled knowledge subgraphs into natural language text, guaranteeing grammatical accuracy and syntactical diversity. Extensive statistical analysis confirms the scale, comprehensiveness, and diversity of our UniEdit benchmark. We conduct comprehensive experiments across multiple LLMs and editors, analyzing their performance to highlight strengths and weaknesses in editing across open knowledge domains and various evaluation criteria, thereby offering valuable insights for future research endeavors. 

---
# LLMSR@XLLM25: An Empirical Study of LLM for Structural Reasoning 

**Authors**: Xinye Li, Mingqi Wan, Dianbo Sui  

**Link**: [PDF](https://arxiv.org/pdf/2505.12328)  

**Abstract**: We present Team asdfo123's submission to the LLMSR@XLLM25 shared task, which evaluates large language models on producing fine-grained, controllable, and interpretable reasoning processes. Systems must extract all problem conditions, decompose a chain of thought into statement-evidence pairs, and verify the logical validity of each pair. Leveraging only the off-the-shelf Meta-Llama-3-8B-Instruct, we craft a concise few-shot, multi-turn prompt that first enumerates all conditions and then guides the model to label, cite, and adjudicate every reasoning step. A lightweight post-processor based on regular expressions normalises spans and enforces the official JSON schema. Without fine-tuning, external retrieval, or ensembling, our method ranks 5th overall, achieving macro F1 scores on par with substantially more complex and resource-consuming pipelines. We conclude by analysing the strengths and limitations of our approach and outlining directions for future research in structural reasoning with LLMs. Our code is available at this https URL. 

---
# ExpertSteer: Intervening in LLMs through Expert Knowledge 

**Authors**: Weixuan Wang, Minghao Wu, Barry Haddow, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2505.12313)  

**Abstract**: Large Language Models (LLMs) exhibit remarkable capabilities across various tasks, yet guiding them to follow desired behaviours during inference remains a significant challenge. Activation steering offers a promising method to control the generation process of LLMs by modifying their internal activations. However, existing methods commonly intervene in the model's behaviour using steering vectors generated by the model itself, which constrains their effectiveness to that specific model and excludes the possibility of leveraging powerful external expert models for steering. To address these limitations, we propose ExpertSteer, a novel approach that leverages arbitrary specialized expert models to generate steering vectors, enabling intervention in any LLMs. ExpertSteer transfers the knowledge from an expert model to a target LLM through a cohesive four-step process: first aligning representation dimensions with auto-encoders to enable cross-model transfer, then identifying intervention layer pairs based on mutual information analysis, next generating steering vectors from the expert model using Recursive Feature Machines, and finally applying these vectors on the identified layers during inference to selectively guide the target LLM without updating model parameters. We conduct comprehensive experiments using three LLMs on 15 popular benchmarks across four distinct domains. Experiments demonstrate that ExpertSteer significantly outperforms established baselines across diverse tasks at minimal cost. 

---
# HBO: Hierarchical Balancing Optimization for Fine-Tuning Large Language Models 

**Authors**: Weixuan Wang, Minghao Wu, Barry Haddow, Alexandra Birch  

**Link**: [PDF](https://arxiv.org/pdf/2505.12300)  

**Abstract**: Fine-tuning large language models (LLMs) on a mixture of diverse datasets poses challenges due to data imbalance and heterogeneity. Existing methods often address these issues across datasets (globally) but overlook the imbalance and heterogeneity within individual datasets (locally), which limits their effectiveness. We introduce Hierarchical Balancing Optimization (HBO), a novel method that enables LLMs to autonomously adjust data allocation during fine-tuning both across datasets (globally) and within each individual dataset (locally). HBO employs a bilevel optimization strategy with two types of actors: a Global Actor, which balances data sampling across different subsets of the training mixture, and several Local Actors, which optimizes data usage within each subset based on difficulty levels. These actors are guided by reward functions derived from the LLM's training state, which measure learning progress and relative performance improvement. We evaluate HBO on three LLM backbones across nine diverse tasks in multilingual and multitask setups. Results show that HBO consistently outperforms existing baselines, achieving significant accuracy gains. Our in-depth analysis further demonstrates that both the global actor and local actors of HBO effectively adjust data usage during fine-tuning. HBO provides a comprehensive solution to the challenges of data imbalance and heterogeneity in LLM fine-tuning, enabling more effective training across diverse datasets. 

---
# LLM-Based Evaluation of Low-Resource Machine Translation: A Reference-less Dialect Guided Approach with a Refined Sylheti-English Benchmark 

**Authors**: Md. Atiqur Rahman, Sabrina Islam, Mushfiqul Haque Omi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12273)  

**Abstract**: Evaluating machine translation (MT) for low-resource languages poses a persistent challenge, primarily due to the limited availability of high quality reference translations. This issue is further exacerbated in languages with multiple dialects, where linguistic diversity and data scarcity hinder robust evaluation. Large Language Models (LLMs) present a promising solution through reference-free evaluation techniques; however, their effectiveness diminishes in the absence of dialect-specific context and tailored guidance. In this work, we propose a comprehensive framework that enhances LLM-based MT evaluation using a dialect guided approach. We extend the ONUBAD dataset by incorporating Sylheti-English sentence pairs, corresponding machine translations, and Direct Assessment (DA) scores annotated by native speakers. To address the vocabulary gap, we augment the tokenizer vocabulary with dialect-specific terms. We further introduce a regression head to enable scalar score prediction and design a dialect-guided (DG) prompting strategy. Our evaluation across multiple LLMs shows that the proposed pipeline consistently outperforms existing methods, achieving the highest gain of +0.1083 in Spearman correlation, along with improvements across other evaluation settings. The dataset and the code are available at this https URL. 

---
# Teach2Eval: An Indirect Evaluation Method for LLM by Judging How It Teaches 

**Authors**: Yuhang Zhou, Xutian Chen, Yixin Cao, Yuchen Ni, Yu He, Siyu Tian, Xiang Liu, Jian Zhang, Chuanjun Ji, Guangnan Ye, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12259)  

**Abstract**: Recent progress in large language models (LLMs) has outpaced the development of effective evaluation methods. Traditional benchmarks rely on task-specific metrics and static datasets, which often suffer from fairness issues, limited scalability, and contamination risks. In this paper, we introduce Teach2Eval, an indirect evaluation framework inspired by the Feynman Technique. Instead of directly testing LLMs on predefined tasks, our method evaluates a model's multiple abilities to teach weaker student models to perform tasks effectively. By converting open-ended tasks into standardized multiple-choice questions (MCQs) through teacher-generated feedback, Teach2Eval enables scalable, automated, and multi-dimensional assessment. Our approach not only avoids data leakage and memorization but also captures a broad range of cognitive abilities that are orthogonal to current benchmarks. Experimental results across 26 leading LLMs show strong alignment with existing human and model-based dynamic rankings, while offering additional interpretability for training guidance. 

---
# Data Whisperer: Efficient Data Selection for Task-Specific LLM Fine-Tuning via Few-Shot In-Context Learning 

**Authors**: Shaobo Wang, Ziming Wang, Xiangqi Jin, Jize Wang, Jiajun Zhang, Kaixin Li, Zichen Wen, Zhong Li, Conghui He, Xuming Hu, Linfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12212)  

**Abstract**: Fine-tuning large language models (LLMs) on task-specific data is essential for their effective deployment. As dataset sizes grow, efficiently selecting optimal subsets for training becomes crucial to balancing performance and computational costs. Traditional data selection methods often require fine-tuning a scoring model on the target dataset, which is time-consuming and resource-intensive, or rely on heuristics that fail to fully leverage the model's predictive capabilities. To address these challenges, we propose Data Whisperer, an efficient, training-free, attention-based method that leverages few-shot in-context learning with the model to be fine-tuned. Comprehensive evaluations were conducted on both raw and synthetic datasets across diverse tasks and models. Notably, Data Whisperer achieves superior performance compared to the full GSM8K dataset on the Llama-3-8B-Instruct model, using just 10% of the data, and outperforms existing methods with a 3.1-point improvement and a 7.4$\times$ speedup. 

---
# Learning Auxiliary Tasks Improves Reference-Free Hallucination Detection in Open-Domain Long-Form Generation 

**Authors**: Chengwei Qin, Wenxuan Zhou, Karthik Abinav Sankararaman, Nanshu Wang, Tengyu Xu, Alexander Radovic, Eryk Helenowski, Arya Talebzadeh, Aditya Tayade, Sinong Wang, Shafiq Joty, Han Fang, Hao Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.12265)  

**Abstract**: Hallucination, the generation of factually incorrect information, remains a significant challenge for large language models (LLMs), especially in open-domain long-form generation. Existing approaches for detecting hallucination in long-form tasks either focus on limited domains or rely heavily on external fact-checking tools, which may not always be available.
In this work, we systematically investigate reference-free hallucination detection in open-domain long-form responses. Our findings reveal that internal states (e.g., model's output probability and entropy) alone are insufficient for reliably (i.e., better than random guessing) distinguishing between factual and hallucinated content. To enhance detection, we explore various existing approaches, including prompting-based methods, probing, and fine-tuning, with fine-tuning proving the most effective. To further improve the accuracy, we introduce a new paradigm, named RATE-FT, that augments fine-tuning with an auxiliary task for the model to jointly learn with the main task of hallucination detection. With extensive experiments and analysis using a variety of model families & datasets, we demonstrate the effectiveness and generalizability of our method, e.g., +3% over general fine-tuning methods on LongFact. 

---
# Distribution Prompting: Understanding the Expressivity of Language Models Through the Next-Token Distributions They Can Produce 

**Authors**: Haojin Wang, Zining Zhu, Freda Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.12244)  

**Abstract**: Autoregressive neural language models (LMs) generate a probability distribution over tokens at each time step given a prompt. In this work, we attempt to systematically understand the probability distributions that LMs can produce, showing that some distributions are significantly harder to elicit than others. Specifically, for any target next-token distribution over the vocabulary, we attempt to find a prompt that induces the LM to output a distribution as close as possible to the target, using either soft or hard gradient-based prompt tuning. We find that (1) in general, distributions with very low or very high entropy are easier to approximate than those with moderate entropy; (2) among distributions with the same entropy, those containing ''outlier tokens'' are easier to approximate; (3) target distributions generated by LMs -- even LMs with different tokenizers -- are easier to approximate than randomly chosen targets. These results offer insights into the expressiveness of LMs and the challenges of using them as probability distribution proposers. 

---
# One-for-All Pruning: A Universal Model for Customized Compression of Large Language Models 

**Authors**: Rongguang Ye, Ming Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12216)  

**Abstract**: Existing pruning methods for large language models (LLMs) focus on achieving high compression rates while maintaining model performance. Although these methods have demonstrated satisfactory performance in handling a single user's compression request, their processing time increases linearly with the number of requests, making them inefficient for real-world scenarios with multiple simultaneous requests. To address this limitation, we propose a Univeral Model for Customized Compression (UniCuCo) for LLMs, which introduces a StratNet that learns to map arbitrary requests to their optimal pruning strategy. The challenge in training StratNet lies in the high computational cost of evaluating pruning strategies and the non-differentiable nature of the pruning process, which hinders gradient backpropagation for StratNet updates. To overcome these challenges, we leverage a Gaussian process to approximate the evaluation process. Since the gradient of the Gaussian process is computable, we can use it to approximate the gradient of the non-differentiable pruning process, thereby enabling StratNet updates. Experimental results show that UniCuCo is 28 times faster than baselines in processing 64 requests, while maintaining comparable accuracy to baselines. 

---
# Examining Linguistic Shifts in Academic Writing Before and After the Launch of ChatGPT: A Study on Preprint Papers 

**Authors**: Tong Bao, Yi Zhao, Jin Mao, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12218)  

**Abstract**: Large Language Models (LLMs), such as ChatGPT, have prompted academic concerns about their impact on academic writing. Existing studies have primarily examined LLM usage in academic writing through quantitative approaches, such as word frequency statistics and probability-based analyses. However, few have systematically examined the potential impact of LLMs on the linguistic characteristics of academic writing. To address this gap, we conducted a large-scale analysis across 823,798 abstracts published in last decade from arXiv dataset. Through the linguistic analysis of features such as the frequency of LLM-preferred words, lexical complexity, syntactic complexity, cohesion, readability and sentiment, the results indicate a significant increase in the proportion of LLM-preferred words in abstracts, revealing the widespread influence of LLMs on academic writing. Additionally, we observed an increase in lexical complexity and sentiment in the abstracts, but a decrease in syntactic complexity, suggesting that LLMs introduce more new vocabulary and simplify sentence structure. However, the significant decrease in cohesion and readability indicates that abstracts have fewer connecting words and are becoming more difficult to read. Moreover, our analysis reveals that scholars with weaker English proficiency were more likely to use the LLMs for academic writing, and focused on improving the overall logic and fluency of the abstracts. Finally, at discipline level, we found that scholars in Computer Science showed more pronounced changes in writing style, while the changes in Mathematics were minimal. 

---
# How Reliable is Multilingual LLM-as-a-Judge? 

**Authors**: Xiyan Fu, Wei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12201)  

**Abstract**: LLM-as-a-Judge has emerged as a popular evaluation strategy, where advanced large language models assess generation results in alignment with human instructions. While these models serve as a promising alternative to human annotators, their reliability in multilingual evaluation remains uncertain. To bridge this gap, we conduct a comprehensive analysis of multilingual LLM-as-a-Judge. Specifically, we evaluate five models from different model families across five diverse tasks involving 25 languages. Our findings reveal that LLMs struggle to achieve consistent judgment results across languages, with an average Fleiss' Kappa of approximately 0.3, and some models performing even worse. To investigate the cause of inconsistency, we analyze various influencing factors. We observe that consistency varies significantly across languages, with particularly poor performance in low-resource languages. Additionally, we find that neither training on multilingual data nor increasing model scale directly improves judgment consistency. These findings suggest that LLMs are not yet reliable for evaluating multilingual predictions. We finally propose an ensemble strategy which improves the consistency of the multilingual judge in real-world applications. 

---
# Why Not Act on What You Know? Unleashing Safety Potential of LLMs via Self-Aware Guard Enhancement 

**Authors**: Peng Ding, Jun Kuang, Zongyu Wang, Xuezhi Cao, Xunliang Cai, Jiajun Chen, Shujian Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12060)  

**Abstract**: Large Language Models (LLMs) have shown impressive capabilities across various tasks but remain vulnerable to meticulously crafted jailbreak attacks. In this paper, we identify a critical safety gap: while LLMs are adept at detecting jailbreak prompts, they often produce unsafe responses when directly processing these inputs. Inspired by this insight, we propose SAGE (Self-Aware Guard Enhancement), a training-free defense strategy designed to align LLMs' strong safety discrimination performance with their relatively weaker safety generation ability. SAGE consists of two core components: a Discriminative Analysis Module and a Discriminative Response Module, enhancing resilience against sophisticated jailbreak attempts through flexible safety discrimination instructions. Extensive experiments demonstrate SAGE's effectiveness and robustness across various open-source and closed-source LLMs of different sizes and architectures, achieving an average 99% defense success rate against numerous complex and covert jailbreak methods while maintaining helpfulness on general benchmarks. We further conduct mechanistic interpretability analysis through hidden states and attention distributions, revealing the underlying mechanisms of this detection-generation discrepancy. Our work thus contributes to developing future LLMs with coherent safety awareness and generation behavior. Our code and datasets are publicly available at this https URL. 

---
# Unveiling Knowledge Utilization Mechanisms in LLM-based Retrieval-Augmented Generation 

**Authors**: Yuhao Wang, Ruiyang Ren, Yucheng Wang, Wayne Xin Zhao, Jing Liu, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11995)  

**Abstract**: Considering the inherent limitations of parametric knowledge in large language models (LLMs), retrieval-augmented generation (RAG) is widely employed to expand their knowledge scope. Since RAG has shown promise in knowledge-intensive tasks like open-domain question answering, its broader application to complex tasks and intelligent assistants has further advanced its utility. Despite this progress, the underlying knowledge utilization mechanisms of LLM-based RAG remain underexplored. In this paper, we present a systematic investigation of the intrinsic mechanisms by which LLMs integrate internal (parametric) and external (retrieved) knowledge in RAG scenarios. Specially, we employ knowledge stream analysis at the macroscopic level, and investigate the function of individual modules at the microscopic level. Drawing on knowledge streaming analyses, we decompose the knowledge utilization process into four distinct stages within LLM layers: knowledge refinement, knowledge elicitation, knowledge expression, and knowledge contestation. We further demonstrate that the relevance of passages guides the streaming of knowledge through these stages. At the module level, we introduce a new method, knowledge activation probability entropy (KAPE) for neuron identification associated with either internal or external knowledge. By selectively deactivating these neurons, we achieve targeted shifts in the LLM's reliance on one knowledge source over the other. Moreover, we discern complementary roles for multi-head attention and multi-layer perceptron layers during knowledge formation. These insights offer a foundation for improving interpretability and reliability in retrieval-augmented LLMs, paving the way for more robust and transparent generative solutions in knowledge-intensive domains. 

---
# CCNU at SemEval-2025 Task 3: Leveraging Internal and External Knowledge of Large Language Models for Multilingual Hallucination Annotation 

**Authors**: Xu Liu, Guanyi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11965)  

**Abstract**: We present the system developed by the Central China Normal University (CCNU) team for the Mu-SHROOM shared task, which focuses on identifying hallucinations in question-answering systems across 14 different languages. Our approach leverages multiple Large Language Models (LLMs) with distinct areas of expertise, employing them in parallel to annotate hallucinations, effectively simulating a crowdsourcing annotation process. Furthermore, each LLM-based annotator integrates both internal and external knowledge related to the input during the annotation process. Using the open-source LLM DeepSeek-V3, our system achieves the top ranking (\#1) for Hindi data and secures a Top-5 position in seven other languages. In this paper, we also discuss unsuccessful approaches explored during our development process and share key insights gained from participating in this shared task. 

---
# ChartEdit: How Far Are MLLMs From Automating Chart Analysis? Evaluating MLLMs' Capability via Chart Editing 

**Authors**: Xuanle Zhao, Xuexin Liu, Haoyue Yang, Xianzhen Luo, Fanhu Zeng, Jianling Li, Qi Shi, Chi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11935)  

**Abstract**: Although multimodal large language models (MLLMs) show promise in generating chart rendering code, chart editing presents a greater challenge. This difficulty stems from its nature as a labor-intensive task for humans that also demands MLLMs to integrate chart understanding, complex reasoning, and precise intent interpretation. While many MLLMs claim such editing capabilities, current assessments typically rely on limited case studies rather than robust evaluation methodologies, highlighting the urgent need for a comprehensive evaluation framework. In this work, we propose ChartEdit, a new high-quality benchmark designed for chart editing tasks. This benchmark comprises $1,405$ diverse editing instructions applied to $233$ real-world charts, with each instruction-chart instance having been manually annotated and validated for accuracy. Utilizing ChartEdit, we evaluate the performance of 10 mainstream MLLMs across two types of experiments, assessing them at both the code and chart levels. The results suggest that large-scale models can generate code to produce images that partially match the reference images. However, their ability to generate accurate edits according to the instructions remains limited. The state-of-the-art (SOTA) model achieves a score of only $59.96$, highlighting significant challenges in precise modification. In contrast, small-scale models, including chart-domain models, struggle both with following editing instructions and generating overall chart images, underscoring the need for further development in this area. Code is available at this https URL. 

---
# Enhancing Complex Instruction Following for Large Language Models with Mixture-of-Contexts Fine-tuning 

**Authors**: Yuheng Lu, ZiMeng Bai, Caixia Yuan, Huixing Jiang, Xiaojie Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11922)  

**Abstract**: Large language models (LLMs) exhibit remarkable capabilities in handling natural language tasks; however, they may struggle to consistently follow complex instructions including those involve multiple constraints. Post-training LLMs using supervised fine-tuning (SFT) is a standard approach to improve their ability to follow instructions. In addressing complex instruction following, existing efforts primarily focus on data-driven methods that synthesize complex instruction-output pairs for SFT. However, insufficient attention allocated to crucial sub-contexts may reduce the effectiveness of SFT. In this work, we propose transforming sequentially structured input instruction into multiple parallel instructions containing subcontexts. To support processing this multi-input, we propose MISO (Multi-Input Single-Output), an extension to currently dominant decoder-only transformer-based LLMs. MISO introduces a mixture-of-contexts paradigm that jointly considers the overall instruction-output alignment and the influence of individual sub-contexts to enhance SFT effectiveness. We apply MISO fine-tuning to complex instructionfollowing datasets and evaluate it with standard LLM inference. Empirical results demonstrate the superiority of MISO as a fine-tuning method for LLMs, both in terms of effectiveness in complex instruction-following scenarios and its potential for training efficiency. 

---
# ELITE: Embedding-Less retrieval with Iterative Text Exploration 

**Authors**: Zhangyu Wang, Siyuan Gao, Rong Zhou, Hao Wang, Li Ning  

**Link**: [PDF](https://arxiv.org/pdf/2505.11908)  

**Abstract**: Large Language Models (LLMs) have achieved impressive progress in natural language processing, but their limited ability to retain long-term context constrains performance on document-level or multi-turn tasks. Retrieval-Augmented Generation (RAG) mitigates this by retrieving relevant information from an external corpus. However, existing RAG systems often rely on embedding-based retrieval trained on corpus-level semantic similarity, which can lead to retrieving content that is semantically similar in form but misaligned with the question's true intent. Furthermore, recent RAG variants construct graph- or hierarchy-based structures to improve retrieval accuracy, resulting in significant computation and storage overhead. In this paper, we propose an embedding-free retrieval framework. Our method leverages the logical inferencing ability of LLMs in retrieval using iterative search space refinement guided by our novel importance measure and extend our retrieval results with logically related information without explicit graph construction. Experiments on long-context QA benchmarks, including NovelQA and Marathon, show that our approach outperforms strong baselines while reducing storage and runtime by over an order of magnitude. 

---
# AutoMedEval: Harnessing Language Models for Automatic Medical Capability Evaluation 

**Authors**: Xiechi Zhang, Zetian Ouyang, Linlin Wang, Gerard de Melo, Zhu Cao, Xiaoling Wang, Ya Zhang, Yanfeng Wang, Liang He  

**Link**: [PDF](https://arxiv.org/pdf/2505.11887)  

**Abstract**: With the proliferation of large language models (LLMs) in the medical domain, there is increasing demand for improved evaluation techniques to assess their capabilities. However, traditional metrics like F1 and ROUGE, which rely on token overlaps to measure quality, significantly overlook the importance of medical terminology. While human evaluation tends to be more reliable, it can be very costly and may as well suffer from inaccuracies due to limits in human expertise and motivation. Although there are some evaluation methods based on LLMs, their usability in the medical field is limited due to their proprietary nature or lack of expertise. To tackle these challenges, we present AutoMedEval, an open-sourced automatic evaluation model with 13B parameters specifically engineered to measure the question-answering proficiency of medical LLMs. The overarching objective of AutoMedEval is to assess the quality of responses produced by diverse models, aspiring to significantly reduce the dependence on human evaluation. Specifically, we propose a hierarchical training method involving curriculum instruction tuning and an iterative knowledge introspection mechanism, enabling AutoMedEval to acquire professional medical assessment capabilities with limited instructional data. Human evaluations indicate that AutoMedEval surpasses other baselines in terms of correlation with human judgments. 

---
# Towards Comprehensive Argument Analysis in Education: Dataset, Tasks, and Method 

**Authors**: Yupei Ren, Xinyi Zhou, Ning Zhang, Shangqing Zhao, Man Lan, Xiaopeng Bai  

**Link**: [PDF](https://arxiv.org/pdf/2505.12028)  

**Abstract**: Argument mining has garnered increasing attention over the years, with the recent advancement of Large Language Models (LLMs) further propelling this trend. However, current argument relations remain relatively simplistic and foundational, struggling to capture the full scope of argument information, particularly when it comes to representing complex argument structures in real-world scenarios. To address this limitation, we propose 14 fine-grained relation types from both vertical and horizontal dimensions, thereby capturing the intricate interplay between argument components for a thorough understanding of argument structure. On this basis, we conducted extensive experiments on three tasks: argument component detection, relation prediction, and automated essay grading. Additionally, we explored the impact of writing quality on argument component detection and relation prediction, as well as the connections between discourse relations and argumentative features. The findings highlight the importance of fine-grained argumentative annotations for argumentative writing quality assessment and encourage multi-dimensional argument analysis. 

---
# MoL for LLMs: Dual-Loss Optimization to Enhance Domain Expertise While Preserving General Capabilities 

**Authors**: Jingxue Chen, Qingkun Tang, Qianchun Lu, Siyuan Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12043)  

**Abstract**: Although LLMs perform well in general tasks, domain-specific applications suffer from hallucinations and accuracy limitations. CPT approaches encounter two key issues: (1) domain-biased data degrades general language skills, and (2) improper corpus-mixture ratios limit effective adaptation. To address these, we propose a novel framework, Mixture of Losses (MoL), which decouples optimization objectives for domain-specific and general corpora. Specifically, cross-entropy (CE) loss is applied to domain data to ensure knowledge acquisition, while Kullback-Leibler (KL) divergence aligns general-corpus training with the base model's foundational capabilities. This dual-loss architecture preserves universal skills while enhancing domain expertise, avoiding catastrophic forgetting. Empirically, we validate that a 1:1 domain-to-general corpus ratio optimally balances training and overfitting without the need for extensive tuning or resource-intensive experiments. Furthermore, our experiments demonstrate significant performance gains compared to traditional CPT approaches, which often suffer from degradation in general language capabilities; our model achieves 27.9% higher accuracy on the Math-500 benchmark in the non-think reasoning mode, and an impressive 83.3% improvement on the challenging AIME25 subset in the think mode, underscoring the effectiveness of our approach. 

---
# NAMET: Robust Massive Model Editing via Noise-Aware Memory Optimization 

**Authors**: Yanbo Dai, Zhenlan Ji, Zongjie Li, Shuai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11876)  

**Abstract**: Model editing techniques are essential for efficiently updating knowledge in large language models (LLMs). However, the effectiveness of existing approaches degrades in massive editing scenarios, particularly when evaluated with practical metrics or in context-rich settings. We attribute these failures to embedding collisions among knowledge items, which undermine editing reliability at scale. To address this, we propose NAMET (Noise-aware Model Editing in Transformers), a simple yet effective method that introduces noise during memory extraction via a one-line modification to MEMIT. Extensive experiments across six LLMs and three datasets demonstrate that NAMET consistently outperforms existing methods when editing thousands of facts. 

---
# When AI Co-Scientists Fail: SPOT-a Benchmark for Automated Verification of Scientific Research 

**Authors**: Guijin Son, Jiwoo Hong, Honglu Fan, Heejeong Nam, Hyunwoo Ko, Seungwon Lim, Jinyeop Song, Jinha Choi, Gonçalo Paulo, Youngjae Yu, Stella Biderman  

**Link**: [PDF](https://arxiv.org/pdf/2505.11855)  

**Abstract**: Recent advances in large language models (LLMs) have fueled the vision of automated scientific discovery, often called AI Co-Scientists. To date, prior work casts these systems as generative co-authors responsible for crafting hypotheses, synthesizing code, or drafting manuscripts. In this work, we explore a complementary application: using LLMs as verifiers to automate the \textbf{academic verification of scientific manuscripts}. To that end, we introduce SPOT, a dataset of 83 published papers paired with 91 errors significant enough to prompt errata or retraction, cross-validated with actual authors and human annotators. Evaluating state-of-the-art LLMs on SPOT, we find that none surpasses 21.1\% recall or 6.1\% precision (o3 achieves the best scores, with all others near zero). Furthermore, confidence estimates are uniformly low, and across eight independent runs, models rarely rediscover the same errors, undermining their reliability. Finally, qualitative analysis with domain experts reveals that even the strongest models make mistakes resembling student-level misconceptions derived from misunderstandings. These findings highlight the substantial gap between current LLM capabilities and the requirements for dependable AI-assisted academic verification. 

---
# Efficiently Building a Domain-Specific Large Language Model from Scratch: A Case Study of a Classical Chinese Large Language Model 

**Authors**: Shen Li, Renfen Hu, Lijun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11810)  

**Abstract**: General-purpose large language models demonstrate notable capabilities in language comprehension and generation, achieving results that are comparable to, or even surpass, human performance in many language information processing tasks. Nevertheless, when general models are applied to some specific domains, e.g., Classical Chinese texts, their effectiveness is often unsatisfactory, and fine-tuning open-source foundational models similarly struggles to adequately incorporate domain-specific knowledge. To address this challenge, this study developed a large language model, AI Taiyan, specifically designed for understanding and generating Classical Chinese. Experiments show that with a reasonable model design, data processing, foundational training, and fine-tuning, satisfactory results can be achieved with only 1.8 billion parameters. In key tasks related to Classical Chinese information processing such as punctuation, identification of allusions, explanation of word meanings, and translation between ancient and modern Chinese, this model exhibits a clear advantage over both general-purpose large models and domain-specific traditional models, achieving levels close to or surpassing human baselines. This research provides a reference for the efficient construction of specialized domain-specific large language models. Furthermore, the paper discusses the application of this model in fields such as the collation of ancient texts, dictionary editing, and language research, combined with case studies. 

---
# Masking in Multi-hop QA: An Analysis of How Language Models Perform with Context Permutation 

**Authors**: Wenyu Huang, Pavlos Vougiouklis, Mirella Lapata, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.11754)  

**Abstract**: Multi-hop Question Answering (MHQA) adds layers of complexity to question answering, making it more challenging. When Language Models (LMs) are prompted with multiple search results, they are tasked not only with retrieving relevant information but also employing multi-hop reasoning across the information sources. Although LMs perform well on traditional question-answering tasks, the causal mask can hinder their capacity to reason across complex contexts. In this paper, we explore how LMs respond to multi-hop questions by permuting search results (retrieved documents) under various configurations. Our study reveals interesting findings as follows: 1) Encoder-decoder models, such as the ones in the Flan-T5 family, generally outperform causal decoder-only LMs in MHQA tasks, despite being significantly smaller in size; 2) altering the order of gold documents reveals distinct trends in both Flan T5 models and fine-tuned decoder-only models, with optimal performance observed when the document order aligns with the reasoning chain order; 3) enhancing causal decoder-only models with bi-directional attention by modifying the causal mask can effectively boost their end performance. In addition to the above, we conduct a thorough investigation of the distribution of LM attention weights in the context of MHQA. Our experiments reveal that attention weights tend to peak at higher values when the resulting answer is correct. We leverage this finding to heuristically improve LMs' performance on this task. Our code is publicly available at this https URL. 

---
# BELLE: A Bi-Level Multi-Agent Reasoning Framework for Multi-Hop Question Answering 

**Authors**: Taolin Zhang, Dongyang Li, Qizhou Chen, Chengyu Wang, Xiaofeng He  

**Link**: [PDF](https://arxiv.org/pdf/2505.11811)  

**Abstract**: Multi-hop question answering (QA) involves finding multiple relevant passages and performing step-by-step reasoning to answer complex questions. Previous works on multi-hop QA employ specific methods from different modeling perspectives based on large language models (LLMs), regardless of the question types. In this paper, we first conduct an in-depth analysis of public multi-hop QA benchmarks, dividing the questions into four types and evaluating five types of cutting-edge methods for multi-hop QA: Chain-of-Thought (CoT), Single-step, Iterative-step, Sub-step, and Adaptive-step. We find that different types of multi-hop questions have varying degrees of sensitivity to different types of methods. Thus, we propose a Bi-levEL muLti-agEnt reasoning (BELLE) framework to address multi-hop QA by specifically focusing on the correspondence between question types and methods, where each type of method is regarded as an ''operator'' by prompting LLMs differently. The first level of BELLE includes multiple agents that debate to obtain an executive plan of combined ''operators'' to address the multi-hop QA task comprehensively. During the debate, in addition to the basic roles of affirmative debater, negative debater, and judge, at the second level, we further leverage fast and slow debaters to monitor whether changes in viewpoints are reasonable. Extensive experiments demonstrate that BELLE significantly outperforms strong baselines in various datasets. Additionally, the model consumption of BELLE is higher cost-effectiveness than that of single models in more complex multi-hop QA scenarios. 

---
# MedGUIDE: Benchmarking Clinical Decision-Making in Large Language Models 

**Authors**: Xiaomin Li, Mingye Gao, Yuexing Hao, Taoran Li, Guangya Wan, Zihan Wang, Yijun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.11613)  

**Abstract**: Clinical guidelines, typically structured as decision trees, are central to evidence-based medical practice and critical for ensuring safe and accurate diagnostic decision-making. However, it remains unclear whether Large Language Models (LLMs) can reliably follow such structured protocols. In this work, we introduce MedGUIDE, a new benchmark for evaluating LLMs on their ability to make guideline-consistent clinical decisions. MedGUIDE is constructed from 55 curated NCCN decision trees across 17 cancer types and uses clinical scenarios generated by LLMs to create a large pool of multiple-choice diagnostic questions. We apply a two-stage quality selection process, combining expert-labeled reward models and LLM-as-a-judge ensembles across ten clinical and linguistic criteria, to select 7,747 high-quality samples. We evaluate 25 LLMs spanning general-purpose, open-source, and medically specialized models, and find that even domain-specific LLMs often underperform on tasks requiring structured guideline adherence. We also test whether performance can be improved via in-context guideline inclusion or continued pretraining. Our findings underscore the importance of MedGUIDE in assessing whether LLMs can operate safely within the procedural frameworks expected in real-world clinical settings. 

---
# MedCaseReasoning: Evaluating and learning diagnostic reasoning from clinical case reports 

**Authors**: Kevin Wu, Eric Wu, Rahul Thapa, Kevin Wei, Angela Zhang, Arvind Suresh, Jacqueline J. Tao, Min Woo Sun, Alejandro Lozano, James Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.11733)  

**Abstract**: Doctors and patients alike increasingly use Large Language Models (LLMs) to diagnose clinical cases. However, unlike domains such as math or coding, where correctness can be objectively defined by the final answer, medical diagnosis requires both the outcome and the reasoning process to be accurate. Currently, widely used medical benchmarks like MedQA and MMLU assess only accuracy in the final answer, overlooking the quality and faithfulness of the clinical reasoning process. To address this limitation, we introduce MedCaseReasoning, the first open-access dataset for evaluating LLMs on their ability to align with clinician-authored diagnostic reasoning. The dataset includes 14,489 diagnostic question-and-answer cases, each paired with detailed reasoning statements derived from open-access medical case reports. We evaluate state-of-the-art reasoning LLMs on MedCaseReasoning and find significant shortcomings in their diagnoses and reasoning: for instance, the top-performing open-source model, DeepSeek-R1, achieves only 48% 10-shot diagnostic accuracy and mentions only 64% of the clinician reasoning statements (recall). However, we demonstrate that fine-tuning LLMs on the reasoning traces derived from MedCaseReasoning significantly improves diagnostic accuracy and clinical reasoning recall by an average relative gain of 29% and 41%, respectively. The open-source dataset, code, and models are available at this https URL. 

---
# Ambiguity Resolution in Text-to-Structured Data Mapping 

**Authors**: Zhibo Hu, Chen Wang, Yanfeng Shu, Hye-Young Paik, Liming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.11679)  

**Abstract**: Ambiguity in natural language is a significant obstacle for achieving accurate text to structured data mapping through large language models (LLMs), which affects the performance of tasks such as mapping text to agentic tool calling and text-to-SQL queries. Existing methods of ambiguity handling either exploit ReACT framework to produce the correct mapping through trial and error, or supervised fine tuning to guide models to produce a biased mapping to improve certain tasks. In this paper, we adopt a different approach that characterizes the representation difference of ambiguous text in the latent space and leverage the difference to identify ambiguity before mapping them to structured data. To detect ambiguity of a sentence, we focused on the relationship between ambiguous questions and their interpretations and what cause the LLM ignore multiple interpretations. Different to the distance calculated by dense embedding vectors, we utilize the observation that ambiguity is caused by concept missing in latent space of LLM to design a new distance measurement, computed through the path kernel by the integral of gradient values for each concepts from sparse-autoencoder (SAE) under each state. We identify patterns to distinguish ambiguous questions with this measurement. Based on our observation, We propose a new framework to improve the performance of LLMs on ambiguous agentic tool calling through missing concepts prediction. 

---
# Talk to Your Slides: Efficient Slide Editing Agent with Large Language Models 

**Authors**: Kyudan Jung, Hojun Cho, Jooyeol Yun, Jaehyeok Jang, Jagul Choo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11604)  

**Abstract**: Existing research on large language models (LLMs) for PowerPoint predominantly focuses on slide generation, overlooking the common yet tedious task of editing existing slides. We introduce Talk-to-Your-Slides, an LLM-powered agent that directly edits slides within active PowerPoint sessions through COM communication. Our system employs a two-level approach: (1) high-level processing where an LLM agent interprets instructions and formulates editing plans, and (2) low-level execution where Python scripts directly manipulate PowerPoint objects. Unlike previous methods relying on predefined operations, our approach enables more flexible and contextually-aware editing. To facilitate evaluation, we present TSBench, a human-annotated dataset of 379 diverse editing instructions with corresponding slide variations. Experimental results demonstrate that Talk-to-Your-Slides significantly outperforms baseline methods in execution success rate, instruction fidelity, and editing efficiency. Our code and benchmark are available at this https URL 

---
# THELMA: Task Based Holistic Evaluation of Large Language Model Applications-RAG Question Answering 

**Authors**: Udita Patel, Rutu Mulkar, Jay Roberts, Cibi Chakravarthy Senthilkumar, Sujay Gandhi, Xiaofei Zheng, Naumaan Nayyar, Rafael Castrillo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11626)  

**Abstract**: We propose THELMA (Task Based Holistic Evaluation of Large Language Model Applications), a reference free framework for RAG (Retrieval Augmented generation) based question answering (QA) applications. THELMA consist of six interdependent metrics specifically designed for holistic, fine grained evaluation of RAG QA applications. THELMA framework helps developers and application owners evaluate, monitor and improve end to end RAG QA pipelines without requiring labelled sources or reference this http URL also present our findings on the interplay of the proposed THELMA metrics, which can be interpreted to identify the specific RAG component needing improvement in QA applications. 

---
# A Data Synthesis Method Driven by Large Language Models for Proactive Mining of Implicit User Intentions in Tourism 

**Authors**: Jinqiang Wang, Huansheng Ning, Tao Zhu, Jianguo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.11533)  

**Abstract**: In the tourism domain, Large Language Models (LLMs) often struggle to mine implicit user intentions from tourists' ambiguous inquiries and lack the capacity to proactively guide users toward clarifying their needs. A critical bottleneck is the scarcity of high-quality training datasets that facilitate proactive questioning and implicit intention mining. While recent advances leverage LLM-driven data synthesis to generate such datasets and transfer specialized knowledge to downstream models, existing approaches suffer from several shortcomings: (1) lack of adaptation to the tourism domain, (2) skewed distributions of detail levels in initial inquiries, (3) contextual redundancy in the implicit intention mining module, and (4) lack of explicit thinking about tourists' emotions and intention values. Therefore, we propose SynPT (A Data Synthesis Method Driven by LLMs for Proactive Mining of Implicit User Intentions in the Tourism), which constructs an LLM-driven user agent and assistant agent to simulate dialogues based on seed data collected from Chinese tourism websites. This approach addresses the aforementioned limitations and generates SynPT-Dialog, a training dataset containing explicit reasoning. The dataset is utilized to fine-tune a general LLM, enabling it to proactively mine implicit user intentions. Experimental evaluations, conducted from both human and LLM perspectives, demonstrate the superiority of SynPT compared to existing methods. Furthermore, we analyze key hyperparameters and present case studies to illustrate the practical applicability of our method, including discussions on its adaptability to English-language scenarios. All code and data are publicly available. 

---
# SAKURA: On the Multi-hop Reasoning of Large Audio-Language Models Based on Speech and Audio Information 

**Authors**: Chih-Kai Yang, Neo Ho, Yen-Ting Piao, Hung-yi Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.13237)  

**Abstract**: Large audio-language models (LALMs) extend the large language models with multimodal understanding in speech, audio, etc. While their performances on speech and audio-processing tasks are extensively studied, their reasoning abilities remain underexplored. Particularly, their multi-hop reasoning, the ability to recall and integrate multiple facts, lacks systematic evaluation. Existing benchmarks focus on general speech and audio-processing tasks, conversational abilities, and fairness but overlook this aspect. To bridge this gap, we introduce SAKURA, a benchmark assessing LALMs' multi-hop reasoning based on speech and audio information. Results show that LALMs struggle to integrate speech/audio representations for multi-hop reasoning, even when they extract the relevant information correctly, highlighting a fundamental challenge in multimodal reasoning. Our findings expose a critical limitation in LALMs, offering insights and resources for future research. 

---
# Enhancing Latent Computation in Transformers with Latent Tokens 

**Authors**: Yuchang Sun, Yanxi Chen, Yaliang Li, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.12629)  

**Abstract**: Augmenting large language models (LLMs) with auxiliary tokens has emerged as a promising strategy for enhancing model performance. In this work, we introduce a lightweight method termed latent tokens; these are dummy tokens that may be non-interpretable in natural language but steer the autoregressive decoding process of a Transformer-based LLM via the attention mechanism. The proposed latent tokens can be seamlessly integrated with a pre-trained Transformer, trained in a parameter-efficient manner, and applied flexibly at inference time, while adding minimal complexity overhead to the existing infrastructure of standard Transformers. We propose several hypotheses about the underlying mechanisms of latent tokens and design synthetic tasks accordingly to verify them. Numerical results confirm that the proposed method noticeably outperforms the baselines, particularly in the out-of-distribution generalization scenarios, highlighting its potential in improving the adaptability of LLMs. 

---
# EVALOOP: Assessing LLM Robustness in Programming from a Self-consistency Perspective 

**Authors**: Sen Fang, Weiyuan Ding, Bowen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12185)  

**Abstract**: Assessing the programming capabilities of Large Language Models (LLMs) is crucial for their effective use in software engineering. Current evaluations, however, predominantly measure the accuracy of generated code on static benchmarks, neglecting the critical aspect of model robustness during programming tasks. While adversarial attacks offer insights on model robustness, their effectiveness is limited and evaluation could be constrained. Current adversarial attack methods for robustness evaluation yield inconsistent results, struggling to provide a unified evaluation across different LLMs. We introduce EVALOOP, a novel assessment framework that evaluate the robustness from a self-consistency perspective, i.e., leveraging the natural duality inherent in popular software engineering tasks, e.g., code generation and code summarization. EVALOOP initiates a self-contained feedback loop: an LLM generates output (e.g., code) from an input (e.g., natural language specification), and then use the generated output as the input to produce a new output (e.g., summarizes that code into a new specification). EVALOOP repeats the process to assess the effectiveness of EVALOOP in each loop. This cyclical strategy intrinsically evaluates robustness without rely on any external attack setups, providing a unified metric to evaluate LLMs' robustness in programming. We evaluate 16 prominent LLMs (e.g., GPT-4.1, O4-mini) on EVALOOP and found that EVALOOP typically induces a 5.01%-19.31% absolute drop in pass@1 performance within ten loops. Intriguingly, robustness does not always align with initial performance (i.e., one-time query); for instance, GPT-3.5-Turbo, despite superior initial code generation compared to DeepSeek-V2, demonstrated lower robustness over repeated evaluation loop. 

---
# J1: Exploring Simple Test-Time Scaling for LLM-as-a-Judge 

**Authors**: Chi-Min Chan, Chunpu Xu, Jiaming Ji, Zhen Ye, Pengcheng Wen, Chunyang Jiang, Yaodong Yang, Wei Xue, Sirui Han, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.11875)  

**Abstract**: The current focus of AI research is shifting from emphasizing model training towards enhancing evaluation quality, a transition that is crucial for driving further advancements in AI systems. Traditional evaluation methods typically rely on reward models assigning scalar preference scores to outputs. Although effective, such approaches lack interpretability, leaving users often uncertain about why a reward model rates a particular response as high or low. The advent of LLM-as-a-Judge provides a more scalable and interpretable method of supervision, offering insights into the decision-making process. Moreover, with the emergence of large reasoning models, which consume more tokens for deeper thinking and answer refinement, scaling test-time computation in the LLM-as-a-Judge paradigm presents an avenue for further boosting performance and providing more interpretability through reasoning traces. In this paper, we introduce $\textbf{J1-7B}$, which is first supervised fine-tuned on reflection-enhanced datasets collected via rejection-sampling and subsequently trained using Reinforcement Learning (RL) with verifiable rewards. At inference time, we apply Simple Test-Time Scaling (STTS) strategies for additional performance improvement. Experimental results demonstrate that $\textbf{J1-7B}$ surpasses the previous state-of-the-art LLM-as-a-Judge by $ \textbf{4.8}$\% and exhibits a $ \textbf{5.1}$\% stronger scaling trend under STTS. Additionally, we present three key findings: (1) Existing LLM-as-a-Judge does not inherently exhibit such scaling trend. (2) Model simply fine-tuned on reflection-enhanced datasets continues to demonstrate similarly weak scaling behavior. (3) Significant scaling trend emerges primarily during the RL phase, suggesting that effective STTS capability is acquired predominantly through RL training. 

---
