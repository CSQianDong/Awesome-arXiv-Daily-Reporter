# Optimizing Retrieval Augmented Generation for Object Constraint Language 

**Authors**: Kevin Chenhao Li, Vahid Zolfaghari, Nenad Petrovic, Fengjunjie Pan, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2505.13129)  

**Abstract**: The Object Constraint Language (OCL) is essential for defining precise constraints within Model-Based Systems Engineering (MBSE). However, manually writing OCL rules is complex and time-consuming. This study explores the optimization of Retrieval-Augmented Generation (RAG) for automating OCL rule generation, focusing on the impact of different retrieval strategies. We evaluate three retrieval approaches $\unicode{x2013}$ BM25 (lexical-based), BERT-based (semantic retrieval), and SPLADE (sparse-vector retrieval) $\unicode{x2013}$ analyzing their effectiveness in providing relevant context for a large language model.
To further assess our approach, we compare and benchmark our retrieval-optimized generation results against PathOCL, a state-of-the-art graph-based method. We directly compare BM25, BERT, and SPLADE retrieval methods with PathOCL to understand how different retrieval methods perform for a unified evaluation framework. Our experimental results, focusing on retrieval-augmented generation, indicate that while retrieval can enhance generation accuracy, its effectiveness depends on the retrieval method and the number of retrieved chunks (k). BM25 underperforms the baseline, whereas semantic approaches (BERT and SPLADE) achieve better results, with SPLADE performing best at lower k values. However, excessive retrieval with high k parameter can lead to retrieving irrelevant chunks which degrades model performance. Our findings highlight the importance of optimizing retrieval configurations to balance context relevance and output consistency. This research provides insights into improving OCL rule generation using RAG and underscores the need for tailoring retrieval. 

---
# Unlearning for Federated Online Learning to Rank: A Reproducibility Study 

**Authors**: Yiling Tao, Shuyi Wang, Jiaxi Yang, Guido Zuccon  

**Link**: [PDF](https://arxiv.org/pdf/2505.12791)  

**Abstract**: This paper reports on findings from a comparative study on the effectiveness and efficiency of federated unlearning strategies within Federated Online Learning to Rank (FOLTR), with specific attention to systematically analysing the unlearning capabilities of methods in a verifiable manner.
Federated approaches to ranking of search results have recently garnered attention to address users privacy concerns. In FOLTR, privacy is safeguarded by collaboratively training ranking models across decentralized data sources, preserving individual user data while optimizing search results based on implicit feedback, such as clicks.
Recent legislation introduced across numerous countries is establishing the so called "the right to be forgotten", according to which services based on machine learning models like those in FOLTR should provide capabilities that allow users to remove their own data from those used to train models. This has sparked the development of unlearning methods, along with evaluation practices to measure whether unlearning of a user data successfully occurred. Current evaluation practices are however often controversial, necessitating the use of multiple metrics for a more comprehensive assessment -- but previous proposals of unlearning methods only used single evaluation metrics.
This paper addresses this limitation: our study rigorously assesses the effectiveness of unlearning strategies in managing both under-unlearning and over-unlearning scenarios using adapted, and newly proposed evaluation metrics. Thanks to our detailed analysis, we uncover the strengths and limitations of five unlearning strategies, offering valuable insights into optimizing federated unlearning to balance data privacy and system performance within FOLTR. We publicly release our code and complete results at this https URL. 

---
# Towards A Generalist Code Embedding Model Based On Massive Data Synthesis 

**Authors**: Chaofan Li, Jianlyu Chen, Yingxia Shao, Defu Lian, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12697)  

**Abstract**: Code embedding models attract increasing attention due to the widespread popularity of retrieval-augmented generation (RAG) in software development. These models are expected to capture the rich semantic relationships inherent to code, which differ significantly from those found in text. However, existing models remain severely limited due to the scarcity of high-quality training data. In this work, we introduce \textbf{CodeR} (\underline{Code} \underline{R}etrieval), a state-of-the-art embedding model for general-purpose code retrieval. The superior performance of CodeR is built upon CodeR-Pile, a large-scale synthetic dataset constructed under the DRU (Diversity, Reliability, Usability) principle via a novel data synthesis pipeline. To optimize training effectiveness, we propose Annealing, a curriculum learning strategy that enables effective knowledge transfer across heterogeneous sources of data. We evaluate CodeR based on 16 diverse code retrieval tasks, where it significantly outperforms existing baselines and exhibits strong out-of-domain generalization performance. We have publicly released our code and the well-trained model to facilitate further research in this critical area. this https URL. 

---
# LLM-based Query Expansion Fails for Unfamiliar and Ambiguous Queries 

**Authors**: Kenya Abe, Kunihiro Takeoka, Makoto P. Kato, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2505.12694)  

**Abstract**: Query expansion (QE) enhances retrieval by incorporating relevant terms, with large language models (LLMs) offering an effective alternative to traditional rule-based and statistical methods. However, LLM-based QE suffers from a fundamental limitation: it often fails to generate relevant knowledge, degrading search performance. Prior studies have focused on hallucination, yet its underlying cause--LLM knowledge deficiencies--remains underexplored. This paper systematically examines two failure cases in LLM-based QE: (1) when the LLM lacks query knowledge, leading to incorrect expansions, and (2) when the query is ambiguous, causing biased refinements that narrow search coverage. We conduct controlled experiments across multiple datasets, evaluating the effects of knowledge and query ambiguity on retrieval performance using sparse and dense retrieval models. Our results reveal that LLM-based QE can significantly degrade the retrieval effectiveness when knowledge in the LLM is insufficient or query ambiguity is high. We introduce a framework for evaluating QE under these conditions, providing insights into the limitations of LLM-based retrieval augmentation. 

---
# PoisonArena: Uncovering Competing Poisoning Attacks in Retrieval-Augmented Generation 

**Authors**: Liuji Chen, Xiaofang Yang, Yuanzhuo Lu, Jinghao Zhang, Xin Sun, Qiang Liu, Shu Wu, Jing Dong, Liang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12574)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems, widely used to improve the factual grounding of large language models (LLMs), are increasingly vulnerable to poisoning attacks, where adversaries inject manipulated content into the retriever's corpus. While prior research has predominantly focused on single-attacker settings, real-world scenarios often involve multiple, competing attackers with conflicting objectives. In this work, we introduce PoisonArena, the first benchmark to systematically study and evaluate competing poisoning attacks in RAG. We formalize the multi-attacker threat model, where attackers vie to control the answer to the same query using mutually exclusive misinformation. PoisonArena leverages the Bradley-Terry model to quantify each method's competitive effectiveness in such adversarial environments. Through extensive experiments on the Natural Questions and MS MARCO datasets, we demonstrate that many attack strategies successful in isolation fail under competitive pressure. Our findings highlight the limitations of conventional evaluation metrics like Attack Success Rate (ASR) and F1 score and underscore the need for competitive evaluation to assess real-world attack robustness. PoisonArena provides a standardized framework to benchmark and develop future attack and defense strategies under more realistic, multi-adversary conditions. Project page: this https URL. 

---
# Batched Self-Consistency Improves LLM Relevance Assessment and Ranking 

**Authors**: Anton Korikov, Pan Du, Scott Sanner, Navid Rekabsaz  

**Link**: [PDF](https://arxiv.org/pdf/2505.12570)  

**Abstract**: Given some information need, Large Language Models (LLMs) are increasingly used for candidate text relevance assessment, typically using a one-by-one pointwise (PW) strategy where each LLM call evaluates one candidate at a time. Meanwhile, it has been shown that LLM performance can be improved through self-consistency: prompting the LLM to do the same task multiple times (possibly in perturbed ways) and then aggregating the responses. To take advantage of self-consistency, we hypothesize that batched PW strategies, where multiple passages are judged in one LLM call, are better suited than one-by-one PW methods since a larger input context can induce more diverse LLM sampling across self-consistency calls. We first propose several candidate batching strategies to create prompt diversity across self-consistency calls through subset reselection and permutation. We then test our batched PW methods on relevance assessment and ranking tasks against one-by-one PW and listwise LLM ranking baselines with and without self-consistency, using three passage retrieval datasets and GPT-4o, Claude Sonnet 3, and Amazon Nova Pro. We find that batched PW methods outperform all baselines, and show that batching can greatly amplify the positive effects of self-consistency. For instance, on our legal search dataset, GPT-4o one-by-one PW ranking NDCG@10 improves only from 44.9% to 46.8% without self-consistency vs. with 15 self consistency calls, while batched PW ranking improves from 43.8% to 51.3%, respectively. 

---
# LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization 

**Authors**: Hailong Luo, Bin Wu, Hongyong Jia, Qingqing Zhu, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2505.12396)  

**Abstract**: Graph neural networks (GNNs) have advanced recommender systems by modeling interaction relationships. However, existing graph-based recommenders rely on sparse ID features and do not fully exploit textual information, resulting in low information density within representations. Furthermore, graph contrastive learning faces challenges. Random negative sampling can introduce false negative samples, while fixed temperature coefficients cannot adapt to the heterogeneity of different nodes. In addition, current efforts to enhance recommendations with large language models (LLMs) have not fully utilized their Chain-of-Thought (CoT) reasoning capabilities to guide representation learning. To address these limitations, we introduces LGHRec (LLM-CoT Enhanced Graph Neural Recommendation with Harmonized Group Policy Optimization). This framework leverages the CoT reasoning ability of LLMs to generate semantic IDs, enriching reasoning processes and improving information density and semantic quality of representations. Moreover, we design a reinforcement learning algorithm, Harmonized Group Policy Optimization (HGPO), to optimize negative sampling strategies and temperature coefficients in contrastive learning. This approach enhances long-tail recommendation performance and ensures optimization consistency across different groups. Experimental results on three datasets demonstrate that LGHRec improves representation quality through semantic IDs generated by LLM's CoT reasoning and effectively boosts contrastive learning with HGPO. Our method outperforms several baseline models. The code is available at: this https URL. 

---
# Addressing Missing Data Issue for Diffusion-based Recommendation 

**Authors**: Wenyu Mao, Zhengyi Yang, Jiancan Wu, Haozhe Liu, Yancheng Yuan, Xiang Wang, Xiangnan He  

**Link**: [PDF](https://arxiv.org/pdf/2505.12283)  

**Abstract**: Diffusion models have shown significant potential in generating oracle items that best match user preference with guidance from user historical interaction sequences. However, the quality of guidance is often compromised by unpredictable missing data in observed sequence, leading to suboptimal item generation. Since missing data is uncertain in both occurrence and content, recovering it is impractical and may introduce additional errors. To tackle this challenge, we propose a novel dual-side Thompson sampling-based Diffusion Model (TDM), which simulates extra missing data in the guidance signals and allows diffusion models to handle existing missing data through extrapolation. To preserve user preference evolution in sequences despite extra missing data, we introduce Dual-side Thompson Sampling to implement simulation with two probability models, sampling by exploiting user preference from both item continuity and sequence stability. TDM strategically removes items from sequences based on dual-side Thompson sampling and treats these edited sequences as guidance for diffusion models, enhancing models' robustness to missing data through consistency regularization. Additionally, to enhance the generation efficiency, TDM is implemented under the denoising diffusion implicit models to accelerate the reverse process. Extensive experiments and theoretical analysis validate the effectiveness of TDM in addressing missing data in sequential recommendations. 

---
# A Survey on Side Information-driven Session-based Recommendation: From a Data-centric Perspective 

**Authors**: Xiaokun Zhang, Bo Xu, Chenliang Li, Bowei He, Hongfei Lin, Chen Ma, Fenglong Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.12279)  

**Abstract**: Session-based recommendation is gaining increasing attention due to its practical value in predicting the intents of anonymous users based on limited behaviors. Emerging efforts incorporate various side information to alleviate inherent data scarcity issues in this task, leading to impressive performance improvements. The core of side information-driven session-based recommendation is the discovery and utilization of diverse data. In this survey, we provide a comprehensive review of this task from a data-centric perspective. Specifically, this survey commences with a clear formulation of the task. This is followed by a detailed exploration of various benchmarks rich in side information that are pivotal for advancing research in this field. Afterwards, we delve into how different types of side information enhance the task, underscoring data characteristics and utility. Moreover, we discuss the usage of various side information, including data encoding, data injection, and involved techniques. A systematic review of research progress is then presented, with the taxonomy by the types of side information. Finally, we summarize the current limitations and present the future prospects of this vibrant topic. 

---
# LightRetriever: A LLM-based Hybrid Retrieval Architecture with 1000x Faster Query Inference 

**Authors**: Guangyuan Ma, Yongliang Ma, Xuanrui Gou, Zhenpeng Su, Ming Zhou, Songlin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12260)  

**Abstract**: Large Language Models (LLMs)-based hybrid retrieval uses LLMs to encode queries and documents into low-dimensional dense or high-dimensional sparse vectors. It retrieves documents relevant to search queries based on vector similarities. Documents are pre-encoded offline, while queries arrive in real-time, necessitating an efficient online query encoder. Although LLMs significantly enhance retrieval capabilities, serving deeply parameterized LLMs slows down query inference throughput and increases demands for online deployment resources. In this paper, we propose LightRetriever, a novel LLM-based hybrid retriever with extremely lightweight query encoders. Our method retains a full-sized LLM for document encoding, but reduces the workload of query encoding to no more than an embedding lookup. Compared to serving a full-sized LLM on an H800 GPU, our approach achieves over a 1000x speedup for query inference with GPU acceleration, and even a 20x speedup without GPU. Experiments on large-scale retrieval benchmarks demonstrate that our method generalizes well across diverse retrieval tasks, retaining an average of 95% full-sized performance. 

---
# Let's have a chat with the EU AI Act 

**Authors**: Adam Kovari, Yasin Ghafourian, Csaba Hegedus, Belal Abu Naim, Kitti Mezei, Pal Varga, Markus Tauber  

**Link**: [PDF](https://arxiv.org/pdf/2505.11946)  

**Abstract**: As artificial intelligence (AI) regulations evolve and the regulatory landscape develops and becomes more complex, ensuring compliance with ethical guidelines and legal frameworks remains a challenge for AI developers. This paper introduces an AI-driven self-assessment chatbot designed to assist users in navigating the European Union AI Act and related standards. Leveraging a Retrieval-Augmented Generation (RAG) framework, the chatbot enables real-time, context-aware compliance verification by retrieving relevant regulatory texts and providing tailored guidance. By integrating both public and proprietary standards, it streamlines regulatory adherence, reduces complexity, and fosters responsible AI development. The paper explores the chatbot's architecture, comparing naive and graph-based RAG models, and discusses its potential impact on AI governance. 

---
# Basic model for ranking microfinance institutions 

**Authors**: Dmitry Dudukalov, Evgeny Prokopenko  

**Link**: [PDF](https://arxiv.org/pdf/2505.11944)  

**Abstract**: This paper discusses the challenges encountered in building a ranking model for aggregator site products, using the example of ranking microfinance institutions (MFIs) based on post-click conversion. We suggest which features of MFIs should be considered, and using an algorithm based on Markov chains, we demonstrate the ``usefulness'' of these features on real data. The ideas developed in this work can be applied to aggregator websites in microinsurance, especially when personal data is unavailable. Since we did not find similar datasets in the public domain, we are publishing our dataset with a detailed description of its attributes. 

---
# Conversational Recommendation System using NLP and Sentiment Analysis 

**Authors**: Piyush Talegaonkar, Siddhant Hole, Shrinesh Kamble, Prashil Gulechha, Deepali Salapurkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.11933)  

**Abstract**: In today's digitally-driven world, the demand for personalized and context-aware recommendations has never been greater. Traditional recommender systems have made significant strides in this direction, but they often lack the ability to tap into the richness of conversational data. This paper represents a novel approach to recommendation systems by integrating conversational insights into the recommendation process. The Conversational Recommender System integrates cutting-edge technologies such as deep learning, leveraging machine learning algorithms like Apriori for Association Rule Mining, Convolutional Neural Networks (CNN), Recurrent Neural Networks (RNN), and Long Short-Term Memory (LTSM). Furthermore, sophisticated voice recognition technologies, including Hidden Markov Models (HMMs) and Dynamic Time Warping (DTW) algorithms, play a crucial role in accurate speech-to-text conversion, ensuring robust performance in diverse environments. The methodology incorporates a fusion of content-based and collaborative recommendation approaches, enhancing them with NLP techniques. This innovative integration ensures a more personalized and context-aware recommendation experience, particularly in marketing applications. 

---
# Telco-oRAG: Optimizing Retrieval-augmented Generation for Telecom Queries via Hybrid Retrieval and Neural Routing 

**Authors**: Andrei-Laurentiu Bornea, Fadhel Ayed, Antonio De Domenico, Nicola Piovesan, Tareq Si Salem, Ali Maatouk  

**Link**: [PDF](https://arxiv.org/pdf/2505.11856)  

**Abstract**: Artificial intelligence will be one of the key pillars of the next generation of mobile networks (6G), as it is expected to provide novel added-value services and improve network performance. In this context, large language models have the potential to revolutionize the telecom landscape through intent comprehension, intelligent knowledge retrieval, coding proficiency, and cross-domain orchestration capabilities. This paper presents Telco-oRAG, an open-source Retrieval-Augmented Generation (RAG) framework optimized for answering technical questions in the telecommunications domain, with a particular focus on 3GPP standards. Telco-oRAG introduces a hybrid retrieval strategy that combines 3GPP domain-specific retrieval with web search, supported by glossary-enhanced query refinement and a neural router for memory-efficient retrieval. Our results show that Telco-oRAG improves the accuracy in answering 3GPP-related questions by up to 17.6% and achieves a 10.6% improvement in lexicon queries compared to baselines. Furthermore, Telco-oRAG reduces memory usage by 45% through targeted retrieval of relevant 3GPP series compared to baseline RAG, and enables open-source LLMs to reach GPT-4-level accuracy on telecom benchmarks. 

---
# The Effects of Demographic Instructions on LLM Personas 

**Authors**: Angel Felipe Magnossão de Paula, J. Shane Culpepper, Alistair Moffat, Sachin Pathiyan Cherumanal, Falk Scholer, Johanne Trippas  

**Link**: [PDF](https://arxiv.org/pdf/2505.11795)  

**Abstract**: Social media platforms must filter sexist content in compliance with governmental regulations. Current machine learning approaches can reliably detect sexism based on standardized definitions, but often neglect the subjective nature of sexist language and fail to consider individual users' perspectives. To address this gap, we adopt a perspectivist approach, retaining diverse annotations rather than enforcing gold-standard labels or their aggregations, allowing models to account for personal or group-specific views of sexism. Using demographic data from Twitter, we employ large language models (LLMs) to personalize the identification of sexism. 

---
# Second SIGIR Workshop on Simulations for Information Access (Sim4IA 2025) 

**Authors**: Philipp Schaer, Christin Katharina Kreutz, Krisztian Balog, Timo Breuer, Andreas Konstantin Kruff  

**Link**: [PDF](https://arxiv.org/pdf/2505.11687)  

**Abstract**: Simulations in information access (IA) have recently gained interest, as shown by various tutorials and workshops around that topic. Simulations can be key contributors to central IA research and evaluation questions, especially around interactive settings when real users are unavailable, or their participation is impossible due to ethical reasons. In addition, simulations in IA can help contribute to a better understanding of users, reduce complexity of evaluation experiments, and improve reproducibility. Building on recent developments in methods and toolkits, the second iteration of our Sim4IA workshop aims to again bring together researchers and practitioners to form an interactive and engaging forum for discussions on the future perspectives of the field. An additional aim is to plan an upcoming TREC/CLEF campaign. 

---
# Terminators: Terms of Service Parsing and Auditing Agents 

**Authors**: Maruf Ahmed Mridul, Inwon Kang, Oshani Seneviratne  

**Link**: [PDF](https://arxiv.org/pdf/2505.11672)  

**Abstract**: Terms of Service (ToS) documents are often lengthy and written in complex legal language, making them difficult for users to read and understand. To address this challenge, we propose Terminators, a modular agentic framework that leverages large language models (LLMs) to parse and audit ToS documents. Rather than treating ToS understanding as a black-box summarization problem, Terminators breaks the task down to three interpretable steps: term extraction, verification, and accountability planning. We demonstrate the effectiveness of our method on the OpenAI ToS using GPT-4o, highlighting strategies to minimize hallucinations and maximize auditability. Our results suggest that structured, agent-based LLM workflows can enhance both the usability and enforceability of complex legal documents. By translating opaque terms into actionable, verifiable components, Terminators promotes ethical use of web content by enabling greater transparency, empowering users to understand their digital rights, and supporting automated policy audits for regulatory or civic oversight. 

---
# MIRACL-VISION: A Large, multilingual, visual document retrieval benchmark 

**Authors**: Radek Osmulsk, Gabriel de Souza P. Moreira, Ronay Ak, Mengyao Xu, Benedikt Schifferer, Even Oldridge  

**Link**: [PDF](https://arxiv.org/pdf/2505.11651)  

**Abstract**: Document retrieval is an important task for search and Retrieval-Augmented Generation (RAG) applications. Large Language Models (LLMs) have contributed to improving the accuracy of text-based document retrieval. However, documents with complex layout and visual elements like tables, charts and infographics are not perfectly represented in textual format. Recently, image-based document retrieval pipelines have become popular, which use visual large language models (VLMs) to retrieve relevant page images given a query. Current evaluation benchmarks on visual document retrieval are limited, as they primarily focus only English language, rely on synthetically generated questions and offer a small corpus size. Therefore, we introduce MIRACL-VISION, a multilingual visual document retrieval evaluation benchmark. MIRACL-VISION covers 18 languages, and is an extension of the MIRACL dataset, a popular benchmark to evaluate text-based multilingual retrieval pipelines. MIRACL was built using a human-intensive annotation process to generate high-quality questions. In order to reduce MIRACL-VISION corpus size to make evaluation more compute friendly while keeping the datasets challenging, we have designed a method for eliminating the "easy" negatives from the corpus. We conducted extensive experiments comparing MIRACL-VISION with other benchmarks, using popular public text and image models. We observe a gap in state-of-the-art VLM-based embedding models on multilingual capabilities, with up to 59.7% lower retrieval accuracy than a text-based retrieval models. Even for the English language, the visual models retrieval accuracy is 12.1% lower compared to text-based models. MIRACL-VISION is a challenging, representative, multilingual evaluation benchmark for visual retrieval pipelines and will help the community build robust models for document retrieval. 

---
# Comparing Lexical and Semantic Vector Search Methods When Classifying Medical Documents 

**Authors**: Lee Harris, Philippe De Wilde, James Bentham  

**Link**: [PDF](https://arxiv.org/pdf/2505.11582)  

**Abstract**: Classification is a common AI problem, and vector search is a typical solution. This transforms a given body of text into a numerical representation, known as an embedding, and modern improvements to vector search focus on optimising speed and predictive accuracy. This is often achieved through neural methods that aim to learn language semantics. However, our results suggest that these are not always the best solution. Our task was to classify rigidly-structured medical documents according to their content, and we found that using off-the-shelf semantic vector search produced slightly worse predictive accuracy than creating a bespoke lexical vector search model, and that it required significantly more time to execute. These findings suggest that traditional methods deserve to be contenders in the information retrieval toolkit, despite the prevalence and success of neural models. 

---
# GSPRec: Temporal-Aware Graph Spectral Filtering for Recommendation 

**Authors**: Ahmad Bin Rabiah, Julian McAuley  

**Link**: [PDF](https://arxiv.org/pdf/2505.11552)  

**Abstract**: Graph-based recommendation systems are effective at modeling collaborative patterns but often suffer from two limitations: overreliance on low-pass filtering, which suppresses user-specific signals, and omission of sequential dynamics in graph construction. We introduce GSPRec, a graph spectral model that integrates temporal transitions through sequentially-informed graph construction and applies frequency-aware filtering in the spectral domain. GSPRec encodes item transitions via multi-hop diffusion to enable the use of symmetric Laplacians for spectral processing. To capture user preferences, we design a dual-filtering mechanism: a Gaussian bandpass filter to extract mid-frequency, user-level patterns, and a low-pass filter to retain global trends. Extensive experiments on four public datasets show that GSPRec consistently outperforms baselines, with an average improvement of 6.77% in NDCG@10. Ablation studies show the complementary benefits of both sequential graph augmentation and bandpass filtering. 

---
# TARGET: Benchmarking Table Retrieval for Generative Tasks 

**Authors**: Xingyu Ji, Parker Glenn, Aditya G. Parameswaran, Madelon Hulsebos  

**Link**: [PDF](https://arxiv.org/pdf/2505.11545)  

**Abstract**: The data landscape is rich with structured data, often of high value to organizations, driving important applications in data analysis and machine learning. Recent progress in representation learning and generative models for such data has led to the development of natural language interfaces to structured data, including those leveraging text-to-SQL. Contextualizing interactions, either through conversational interfaces or agentic components, in structured data through retrieval-augmented generation can provide substantial benefits in the form of freshness, accuracy, and comprehensiveness of answers. The key question is: how do we retrieve the right table(s) for the analytical query or task at hand? To this end, we introduce TARGET: a benchmark for evaluating TAble Retrieval for GEnerative Tasks. With TARGET we analyze the retrieval performance of different retrievers in isolation, as well as their impact on downstream tasks. We find that dense embedding-based retrievers far outperform a BM25 baseline which is less effective than it is for retrieval over unstructured text. We also surface the sensitivity of retrievers across various metadata (e.g., missing table titles), and demonstrate a stark variation of retrieval performance across datasets and tasks. TARGET is available at this https URL. 

---
# GMM-Based Comprehensive Feature Extraction and Relative Distance Preservation For Few-Shot Cross-Modal Retrieval 

**Authors**: Chengsong Sun, Weiping Li, Xiang Li, Yuankun Liu, Lianlei Shan  

**Link**: [PDF](https://arxiv.org/pdf/2505.13306)  

**Abstract**: Few-shot cross-modal retrieval focuses on learning cross-modal representations with limited training samples, enabling the model to handle unseen classes during inference. Unlike traditional cross-modal retrieval tasks, which assume that both training and testing data share the same class distribution, few-shot retrieval involves data with sparse representations across modalities. Existing methods often fail to adequately model the multi-peak distribution of few-shot cross-modal data, resulting in two main biases in the latent semantic space: intra-modal bias, where sparse samples fail to capture intra-class diversity, and inter-modal bias, where misalignments between image and text distributions exacerbate the semantic gap. These biases hinder retrieval accuracy. To address these issues, we propose a novel method, GCRDP, for few-shot cross-modal retrieval. This approach effectively captures the complex multi-peak distribution of data using a Gaussian Mixture Model (GMM) and incorporates a multi-positive sample contrastive learning mechanism for comprehensive feature modeling. Additionally, we introduce a new strategy for cross-modal semantic alignment, which constrains the relative distances between image and text feature distributions, thereby improving the accuracy of cross-modal representations. We validate our approach through extensive experiments on four benchmark datasets, demonstrating superior performance over six state-of-the-art methods. 

---
# CPRet: A Dataset, Benchmark, and Model for Retrieval in Competitive Programming 

**Authors**: Han Deng, Yuan Meng, Shixiang Tang, Wanli Ouyang, Xinzhu Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.12925)  

**Abstract**: Competitive programming benchmarks are widely used in scenarios such as programming contests and large language model assessments. However, the growing presence of duplicate or highly similar problems raises concerns not only about competition fairness, but also about the validity of competitive programming as a benchmark for model evaluation. In this paper, we propose a new problem -- similar question retrieval -- to address this issue. Due to the lack of both data and models, solving this problem is challenging. To this end, we introduce CPRet, a retrieval-oriented benchmark suite for competitive programming, covering four retrieval tasks: two code-centric (i.e., Text-to-Code and Code-to-Code) and two newly proposed problem-centric tasks (i.e., Problem-to-Duplicate and Simplified-to-Full), built from a combination of automatically crawled problem-solution data and manually curated annotations. Our contribution includes both high-quality training data and temporally separated test sets for reliable evaluation. In addition, we develop two task-specialized retrievers based on this dataset: CPRetriever-Code, trained with a novel Group-InfoNCE loss for problem-code alignment, and CPRetriever-Prob, fine-tuned for identifying problem-level similarity. Both models achieve strong results and are open-sourced for local use. Finally, we analyze LiveCodeBench and find that high-similarity problems inflate model pass rates and reduce differentiation, underscoring the need for similarity-aware evaluation in future benchmarks.
Code and data are available at: this https URL 

---
# AdaToken-3D: Dynamic Spatial Gating for Efficient 3D Large Multimodal-Models Reasoning 

**Authors**: Kai Zhang, Xingyu Chen, Xiaofeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12782)  

**Abstract**: Large Multimodal Models (LMMs) have become a pivotal research focus in deep learning, demonstrating remarkable capabilities in 3D scene understanding. However, current 3D LMMs employing thousands of spatial tokens for multimodal reasoning suffer from critical inefficiencies: excessive computational overhead and redundant information flows. Unlike 2D VLMs processing single images, 3D LMMs exhibit inherent architectural redundancy due to the heterogeneous mechanisms between spatial tokens and visual tokens. To address this challenge, we propose AdaToken-3D, an adaptive spatial token optimization framework that dynamically prunes redundant tokens through spatial contribution analysis. Our method automatically tailors pruning strategies to different 3D LMM architectures by quantifying token-level information flows via attention pattern mining. Extensive experiments on LLaVA-3D (a 7B parameter 3D-LMM) demonstrate that AdaToken-3D achieves 21\% faster inference speed and 63\% FLOPs reduction while maintaining original task accuracy. Beyond efficiency gains, this work systematically investigates redundancy patterns in multimodal spatial information flows through quantitative token interaction analysis. Our findings reveal that over 60\% of spatial tokens contribute minimally ($<$5\%) to the final predictions, establishing theoretical foundations for efficient 3D multimodal learning. 

---
# Think Before You Attribute: Improving the Performance of LLMs Attribution Systems 

**Authors**: João Eduardo Batista, Emil Vatai, Mohamed Wahib  

**Link**: [PDF](https://arxiv.org/pdf/2505.12621)  

**Abstract**: Large Language Models (LLMs) are increasingly applied in various science domains, yet their broader adoption remains constrained by a critical challenge: the lack of trustworthy, verifiable outputs. Current LLMs often generate answers without reliable source attribution, or worse, with incorrect attributions, posing a barrier to their use in scientific and high-stakes settings, where traceability and accountability are non-negotiable. To be reliable, attribution systems need high accuracy and retrieve data with short lengths, i.e., attribute to a sentence within a document rather than a whole document. We propose a sentence-level pre-attribution step for Retrieve-Augmented Generation (RAG) systems that classify sentences into three categories: not attributable, attributable to a single quote, and attributable to multiple quotes. By separating sentences before attribution, a proper attribution method can be selected for the type of sentence, or the attribution can be skipped altogether. Our results indicate that classifiers are well-suited for this task. In this work, we propose a pre-attribution step to reduce the computational complexity of attribution, provide a clean version of the HAGRID dataset, and provide an end-to-end attribution system that works out of the box. 

---
# Rebalancing Contrastive Alignment with Learnable Semantic Gaps in Text-Video Retrieval 

**Authors**: Jian Xiao, Zijie Song, Jialong Hu, Hao Cheng, Zhenzhen Hu, Jia Li, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2505.12499)  

**Abstract**: Recent advances in text-video retrieval have been largely driven by contrastive learning frameworks. However, existing methods overlook a key source of optimization tension: the separation between text and video distributions in the representation space (referred to as the modality gap), and the prevalence of false negatives in batch sampling. These factors lead to conflicting gradients under the InfoNCE loss, impeding stable alignment. To mitigate this, we propose GARE, a Gap-Aware Retrieval framework that introduces a learnable, pair-specific increment Delta_ij between text t_i and video v_j to offload the tension from the global anchor representation. We first derive the ideal form of Delta_ij via a coupled multivariate first-order Taylor approximation of the InfoNCE loss under a trust-region constraint, revealing it as a mechanism for resolving gradient conflicts by guiding updates along a locally optimal descent direction. Due to the high cost of directly computing Delta_ij, we introduce a lightweight neural module conditioned on the semantic gap between each video-text pair, enabling structure-aware correction guided by gradient supervision. To further stabilize learning and promote interpretability, we regularize Delta using three components: a trust-region constraint to prevent oscillation, a directional diversity term to promote semantic coverage, and an information bottleneck to limit redundancy. Experiments across four retrieval benchmarks show that GARE consistently improves alignment accuracy and robustness to noisy supervision, confirming the effectiveness of gap-aware tension mitigation. 

---
# Introspective Growth: Automatically Advancing LLM Expertise in Technology Judgment 

**Authors**: Siyang Wu, Honglin Bao, Nadav Kunievsky, James A. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2505.12452)  

**Abstract**: Large language models (LLMs) increasingly demonstrate signs of conceptual understanding, yet much of their internal knowledge remains latent, loosely structured, and difficult to access or evaluate. We propose self-questioning as a lightweight and scalable strategy to improve LLMs' understanding, particularly in domains where success depends on fine-grained semantic distinctions. To evaluate this approach, we introduce a challenging new benchmark of 1.3 million post-2015 computer science patent pairs, characterized by dense technical jargon and strategically complex writing. The benchmark centers on a pairwise differentiation task: can a model distinguish between closely related but substantively different inventions? We show that prompting LLMs to generate and answer their own questions - targeting the background knowledge required for the task - significantly improves performance. These self-generated questions and answers activate otherwise underutilized internal knowledge. Allowing LLMs to retrieve answers from external scientific texts further enhances performance, suggesting that model knowledge is compressed and lacks the full richness of the training data. We also find that chain-of-thought prompting and self-questioning converge, though self-questioning remains more effective for improving understanding of technical concepts. Notably, we uncover an asymmetry in prompting: smaller models often generate more fundamental, more open-ended, better-aligned questions for mid-sized models than large models with better understanding do, revealing a new strategy for cross-model collaboration. Altogether, our findings establish self-questioning as both a practical mechanism for automatically improving LLM comprehension, especially in domains with sparse and underrepresented knowledge, and a diagnostic probe of how internal and external knowledge are organized. 

---
# Scalable Time-Tagged Data Acquisition for Entanglement Distribution in Quantum Networks 

**Authors**: Abderrahim Amlou, Thomas Gerrits, Anouar Rahmouni, Amar Abane, Mheni Merzouki, Ya-Shian Li-Baboud, Ahmed Lbath, Abdella Battou, Oliver Slattery  

**Link**: [PDF](https://arxiv.org/pdf/2505.12102)  

**Abstract**: In distributed quantum applications such as entanglement distribution, precise time synchronization and efficient time-tagged data handling are essential. Traditional systems often suffer from overflow, synchronization drift, and storage inefficiencies. We propose a modular Time Tagging (TT) agent that uses a 1 pulse per second (PPS) signal from White Rabbit (WR) devices to achieve network-wide synchronization, while applying real-time calibration, overflow mitigation, and compression. A live two-lab entanglement distribution experiment validated the system's performance, achieving synchronized coincidence detection at 25,000 counts/sec. 

---
# Demystifying and Enhancing the Efficiency of Large Language Model Based Search Agents 

**Authors**: Tiannuo Yang, Zebin Yao, Bowen Jin, Lixiao Cui, Yusen Li, Gang Wang, Xiaoguang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.12065)  

**Abstract**: Large Language Model (LLM)-based search agents have shown remarkable capabilities in solving complex tasks by dynamically decomposing problems and addressing them through interleaved reasoning and retrieval. However, this interleaved paradigm introduces substantial efficiency bottlenecks. First, we observe that both highly accurate and overly approximate retrieval methods degrade system efficiency: exact search incurs significant retrieval overhead, while coarse retrieval requires additional reasoning steps during generation. Second, we identify inefficiencies in system design, including improper scheduling and frequent retrieval stalls, which lead to cascading latency -- where even minor delays in retrieval amplify end-to-end inference time. To address these challenges, we introduce SearchAgent-X, a high-efficiency inference framework for LLM-based search agents. SearchAgent-X leverages high-recall approximate retrieval and incorporates two key techniques: priority-aware scheduling and non-stall retrieval. Extensive experiments demonstrate that SearchAgent-X consistently outperforms state-of-the-art systems such as vLLM and HNSW-based retrieval across diverse tasks, achieving up to 3.4$\times$ higher throughput and 5$\times$ lower latency, without compromising generation quality. SearchAgent-X is available at this https URL. 

---
# Neuro-Symbolic Query Compiler 

**Authors**: Yuyao Zhang, Zhicheng Dou, Xiaoxi Li, Jiajie Jin, Yongkang Wu, Zhonghua Li, Qi Ye, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.11932)  

**Abstract**: Precise recognition of search intent in Retrieval-Augmented Generation (RAG) systems remains a challenging goal, especially under resource constraints and for complex queries with nested structures and dependencies. This paper presents QCompiler, a neuro-symbolic framework inspired by linguistic grammar rules and compiler design, to bridge this gap. It theoretically designs a minimal yet sufficient Backus-Naur Form (BNF) grammar $G[q]$ to formalize complex queries. Unlike previous methods, this grammar maintains completeness while minimizing redundancy. Based on this, QCompiler includes a Query Expression Translator, a Lexical Syntax Parser, and a Recursive Descent Processor to compile queries into Abstract Syntax Trees (ASTs) for execution. The atomicity of the sub-queries in the leaf nodes ensures more precise document retrieval and response generation, significantly improving the RAG system's ability to address complex queries. 

---
# Recursive Question Understanding for Complex Question Answering over Heterogeneous Personal Data 

**Authors**: Philipp Christmann, Gerhard Weikum  

**Link**: [PDF](https://arxiv.org/pdf/2505.11900)  

**Abstract**: Question answering over mixed sources, like text and tables, has been advanced by verbalizing all contents and encoding it with a language model. A prominent case of such heterogeneous data is personal information: user devices log vast amounts of data every day, such as calendar entries, workout statistics, shopping records, streaming history, and more. Information needs range from simple look-ups to queries of analytical nature. The challenge is to provide humans with convenient access with small footprint, so that all personal data stays on the user devices. We present ReQAP, a novel method that creates an executable operator tree for a given question, via recursive decomposition. Operators are designed to enable seamless integration of structured and unstructured sources, and the execution of the operator tree yields a traceable answer. We further release the PerQA benchmark, with persona-based data and questions, covering a diverse spectrum of realistic user needs. 

---
