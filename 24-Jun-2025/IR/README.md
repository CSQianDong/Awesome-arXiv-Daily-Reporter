# An Audio-centric Multi-task Learning Framework for Streaming Ads Targeting on Spotify 

**Authors**: Shivam Verma, Vivian Chen, Darren Mei  

**Link**: [PDF](https://arxiv.org/pdf/2506.18735)  

**Abstract**: Spotify, a large-scale multimedia platform, attracts over 675 million monthly active users who collectively consume millions of hours of music, podcasts, audiobooks, and video content. This diverse content consumption pattern introduces unique challenges for computational advertising, which must effectively integrate a variety of ad modalities, including audio, video, and display, within a single user experience. Traditional ad recommendation models, primarily designed for foregrounded experiences, often struggle to reconcile the platform's inherent audio-centrality with the demands of optimizing ad performance across multiple formats and modalities. To overcome these challenges, we introduce Cross-modal Adaptive Mixture-of-Experts (CAMoE), a novel framework for optimizing click-through rate (CTR) prediction in both audio-centric and multi-modal settings. CAMoE enhances traditional mixture-of-experts models by incorporating modality-aware task grouping, adaptive loss masking, and deep-cross networks (DCN) to capture complex feature interactions within a multi-modal ad ecosystem. Through extensive ablation studies, we demonstrate that this approach achieves near Pareto-optimal performance across audio, video, and display ad formats, significantly improving AUC-PR compared to conventional single-task and content-based multi-task learning baselines. When deployed at scale on Spotify's ad serving platform, CAMoE delivered substantial gains, yielding a 14.5% increase in CTR for audio ads, a 1.3% increase for video ads, and a 4.8% reduction in expected cost-per-click (eCPC) for audio slots. 

---
# Harnessing the Power of Reinforcement Learning for Language-Model-Based Information Retriever via Query-Document Co-Augmentation 

**Authors**: Jingming Liu, Yumeng Li, Wei Shi, Yao-Xiang Ding, Hui Su, Kun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.18670)  

**Abstract**: Recent studies have proposed leveraging Large Language Models (LLMs) as information retrievers through query rewriting. However, for challenging corpora, we argue that enhancing queries alone is insufficient for robust semantic matching; the LLM should also have sufficient understanding of the corpus by directly handling and augmenting the documents themselves. To this end, we present an LLM-based retriever empowered to augment both user queries and corpus documents, with its policy fully explored via reinforcement learning (RL) and minimal human inductive bias. Notably, we find that simply allowing the LLM to modify documents yields little benefit unless paired with our carefully designed bidirectional RL framework, which enables the LLM to simultaneously learn and collaborate on both query and document augmentation policies. A key technical challenge in realizing such a framework lies in jointly updating both policies during training, where the rewards for the two directions depend on each other, making their entangled reward intractable. Our approach addresses this by introducing a reward sampling strategy and a specifically designed RL algorithm that enables effective training with these sampled rewards. Experimental results demonstrate that our approach significantly enhances LLM-based retrieval performance in both sparse and dense settings, particularly in difficult retrieval domains, and achieves strong cross-benchmark generalization. Our code is released at this https URL. 

---
# Rethinking Click Models in Light of Carousel Interfaces: Theory-Based Categorization and Design of Click Models 

**Authors**: Jingwei Kang, Maarten de Rijke, Santiago de Leon-Martinez, Harrie Oosterhuis  

**Link**: [PDF](https://arxiv.org/pdf/2506.18548)  

**Abstract**: Click models are a well-established for modeling user interactions with web interfaces. Previous work has mainly focused on traditional single-list web search settings; this includes existing surveys that introduced categorizations based on the first generation of probabilistic graphical model (PGM) click models that have become standard. However, these categorizations have become outdated, as their conceptualizations are unable to meaningfully compare PGM with neural network (NN) click models nor generalize to newer interfaces, such as carousel interfaces. We argue that this outdated view fails to adequately explain the fundamentals of click model designs, thus hindering the development of novel click models.
This work reconsiders what should be the fundamental concepts in click model design, grounding them - unlike previous approaches - in their mathematical properties. We propose three fundamental key-design choices that explain what statistical patterns a click model can capture, and thus indirectly, what user behaviors they can capture. Based on these choices, we create a novel click model taxonomy that allows a meaningful comparison of all existing click models; this is the first taxonomy of single-list, grid and carousel click models that includes PGMs and NNs. Finally, we show how our conceptualization provides a foundation for future click model design by an example derivation of a novel design for carousel interfaces. 

---
# PERSCEN: Learning Personalized Interaction Pattern and Scenario Preference for Multi-Scenario Matching 

**Authors**: Haotong Du, Yaqing Wang, Fei Xiong, Lei Shao, Ming Liu, Hao Gu, Quanming Yao, Zhen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18382)  

**Abstract**: With the expansion of business scales and scopes on online platforms, multi-scenario matching has become a mainstream solution to reduce maintenance costs and alleviate data sparsity. The key to effective multi-scenario recommendation lies in capturing both user preferences shared across all scenarios and scenario-aware preferences specific to each scenario. However, existing methods often overlook user-specific modeling, limiting the generation of personalized user representations. To address this, we propose PERSCEN, an innovative approach that incorporates user-specific modeling into multi-scenario matching. PERSCEN constructs a user-specific feature graph based on user characteristics and employs a lightweight graph neural network to capture higher-order interaction patterns, enabling personalized extraction of preferences shared across scenarios. Additionally, we leverage vector quantization techniques to distil scenario-aware preferences from users' behavior sequence within individual scenarios, facilitating user-specific and scenario-aware preference modeling. To enhance efficient and flexible information transfer, we introduce a progressive scenario-aware gated linear unit that allows fine-grained, low-latency fusion. Extensive experiments demonstrate that PERSCEN outperforms existing methods. Further efficiency analysis confirms that PERSCEN effectively balances performance with computational cost, ensuring its practicality for real-world industrial systems. 

---
# Bias vs Bias -- Dawn of Justice: A Fair Fight in Recommendation Systems 

**Authors**: Tahsin Alamgir Kheya, Mohamed Reda Bouadjenek, Sunil Aryal  

**Link**: [PDF](https://arxiv.org/pdf/2506.18327)  

**Abstract**: Recommendation systems play a crucial role in our daily lives by impacting user experience across various domains, including e-commerce, job advertisements, entertainment, etc. Given the vital role of such systems in our lives, practitioners must ensure they do not produce unfair and imbalanced recommendations. Previous work addressing bias in recommendations overlooked bias in certain item categories, potentially leaving some biases unaddressed. Additionally, most previous work on fair re-ranking focused on binary-sensitive attributes. In this paper, we address these issues by proposing a fairness-aware re-ranking approach that helps mitigate bias in different categories of items. This re-ranking approach leverages existing biases to correct disparities in recommendations across various demographic groups. We show how our approach can mitigate bias on multiple sensitive attributes, including gender, age, and occupation. We experimented on three real-world datasets to evaluate the effectiveness of our re-ranking scheme in mitigating bias in recommendations. Our results show how this approach helps mitigate social bias with little to no degradation in performance. 

---
# Team LA at SCIDOCA shared task 2025: Citation Discovery via relation-based zero-shot retrieval 

**Authors**: Trieu An, Long Nguyen, Minh Le Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18316)  

**Abstract**: The Citation Discovery Shared Task focuses on predicting the correct citation from a given candidate pool for a given paragraph. The main challenges stem from the length of the abstract paragraphs and the high similarity among candidate abstracts, making it difficult to determine the exact paper to cite. To address this, we develop a system that first retrieves the top-k most similar abstracts based on extracted relational features from the given paragraph. From this subset, we leverage a Large Language Model (LLM) to accurately identify the most relevant citation. We evaluate our framework on the training dataset provided by the SCIDOCA 2025 organizers, demonstrating its effectiveness in citation prediction. 

---
# Enhancing Document Retrieval in COVID-19 Research: Leveraging Large Language Models for Hidden Relation Extraction 

**Authors**: Hoang-An Trieu, Dinh-Truong Do, Chau Nguyen, Vu Tran, Minh Le Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2506.18311)  

**Abstract**: In recent years, with the appearance of the COVID-19 pandemic, numerous publications relevant to this disease have been issued. Because of the massive volume of publications, an efficient retrieval system is necessary to provide researchers with useful information if an unexpected pandemic happens so suddenly, like COVID-19. In this work, we present a method to help the retrieval system, the Covrelex-SE system, to provide more high-quality search results. We exploited the power of the large language models (LLMs) to extract the hidden relationships inside the unlabeled publication that cannot be found by the current parsing tools that the system is using. Since then, help the system to have more useful information during retrieval progress. 

---
# LettinGo: Explore User Profile Generation for Recommendation System 

**Authors**: Lu Wang, Di Zhang, Fangkai Yang, Pu Zhao, Jianfeng Liu, Yuefeng Zhan, Hao Sun, Qingwei Lin, Weiwei Deng, Dongmei Zhang, Feng Sun, Qi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.18309)  

**Abstract**: User profiling is pivotal for recommendation systems, as it transforms raw user interaction data into concise and structured representations that drive personalized recommendations. While traditional embedding-based profiles lack interpretability and adaptability, recent advances with large language models (LLMs) enable text-based profiles that are semantically richer and more transparent. However, existing methods often adhere to fixed formats that limit their ability to capture the full diversity of user behaviors. In this paper, we introduce LettinGo, a novel framework for generating diverse and adaptive user profiles. By leveraging the expressive power of LLMs and incorporating direct feedback from downstream recommendation tasks, our approach avoids the rigid constraints imposed by supervised fine-tuning (SFT). Instead, we employ Direct Preference Optimization (DPO) to align the profile generator with task-specific performance, ensuring that the profiles remain adaptive and effective. LettinGo operates in three stages: (1) exploring diverse user profiles via multiple LLMs, (2) evaluating profile quality based on their impact in recommendation systems, and (3) aligning the profile generation through pairwise preference data derived from task performance. Experimental results demonstrate that our framework significantly enhances recommendation accuracy, flexibility, and contextual awareness. This work enhances profile generation as a key innovation for next-generation recommendation systems. 

---
# Comparative Analysis of Lion and AdamW Optimizers for Cross-Encoder Reranking with MiniLM, GTE, and ModernBERT 

**Authors**: Shahil Kumar, Manu Pande, Anay Yatin Damle  

**Link**: [PDF](https://arxiv.org/pdf/2506.18297)  

**Abstract**: Modern information retrieval systems often employ a two-stage pipeline: an efficient initial retrieval stage followed by a computationally intensive reranking stage. Cross-encoders have shown strong effectiveness for reranking due to their deep analysis of query-document pairs. This paper studies the impact of the Lion optimizer, a recent alternative to AdamW, during fine-tuning of cross-encoder rerankers. We fine-tune three transformer models-MiniLM, GTE, and ModernBERT-on the MS MARCO passage ranking dataset using both optimizers. GTE and ModernBERT support extended context lengths (up to 8192 tokens). We evaluate effectiveness using TREC 2019 Deep Learning Track and MS MARCO dev set (MRR@10). Experiments, run on the Modal cloud platform, reveal that ModernBERT with Lion achieves the best NDCG@10 (0.7225) and MAP (0.5121) on TREC DL 2019, while MiniLM with Lion ties ModernBERT for MRR@10 (0.5988) on MS MARCO dev. Lion also provides superior GPU efficiency, improving utilization by 2.67% to 10.33% across models. We analyze performance trends using standard IR metrics and discuss the optimizer's impact on training dynamics across architectures. 

---
# LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation 

**Authors**: Wangyu Wu, Zhenhong Chen, Xianglin Qiu, Siqi Song, Xiaowei Huang, Fei Ma, Jimin Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17966)  

**Abstract**: Cross-Domain Sequential Recommendation (CDSR) predicts user behavior by leveraging historical interactions across multiple domains, focusing on modeling cross-domain preferences and capturing both intra- and inter-sequence item relationships. We propose LLM-Enhanced Multimodal Fusion for Cross-Domain Sequential Recommendation (LLM-EMF), a novel and advanced approach that enhances textual information with Large Language Models (LLM) knowledge and significantly improves recommendation performance through the fusion of visual and textual data. Using the frozen CLIP model, we generate image and text embeddings, thereby enriching item representations with multimodal data. A multiple attention mechanism jointly learns both single-domain and cross-domain preferences, effectively capturing and understanding complex user interests across diverse domains. Evaluations conducted on four e-commerce datasets demonstrate that LLM-EMF consistently outperforms existing methods in modeling cross-domain user preferences, thereby highlighting the effectiveness of multimodal data integration and its advantages in enhancing sequential recommendation systems. Our source code will be released. 

---
# A GenAI System for Improved FAIR Independent Biological Database Integration 

**Authors**: Syed N. Sakib, Kallol Naha, Sajratul Y. Rubaiat, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17934)  

**Abstract**: Life sciences research increasingly requires identifying, accessing, and effectively processing data from an ever-evolving array of information sources on the Linked Open Data (LOD) network. This dynamic landscape places a significant burden on researchers, as the quality of query responses depends heavily on the selection and semantic integration of data sources --processes that are often labor-intensive, error-prone, and costly. While the adoption of FAIR (Findable, Accessible, Interoperable, and Reusable) data principles has aimed to address these challenges, barriers to efficient and accurate scientific data processing persist.
In this paper, we introduce FAIRBridge, an experimental natural language-based query processing system designed to empower scientists to discover, access, and query biological databases, even when they are not FAIR-compliant. FAIRBridge harnesses the capabilities of AI to interpret query intents, map them to relevant databases described in scientific literature, and generate executable queries via intelligent resource access plans. The system also includes robust tools for mitigating low-quality query processing, ensuring high fidelity and responsiveness in the information delivered.
FAIRBridge's autonomous query processing framework enables users to explore alternative data sources, make informed choices at every step, and leverage community-driven crowd curation when needed. By providing a user-friendly, automated hypothesis-testing platform in natural English, FAIRBridge significantly enhances the integration and processing of scientific data, offering researchers a powerful new tool for advancing their inquiries. 

---
# Expanding Relevance Judgments for Medical Case-based Retrieval Task with Multimodal LLMs 

**Authors**: Catarina Pires, Sérgio Nunes, Luís Filipe Teixeira  

**Link**: [PDF](https://arxiv.org/pdf/2506.17782)  

**Abstract**: Evaluating Information Retrieval (IR) systems relies on high-quality manual relevance judgments (qrels), which are costly and time-consuming to obtain. While pooling reduces the annotation effort, it results in only partially labeled datasets. Large Language Models (LLMs) offer a promising alternative to reducing reliance on manual judgments, particularly in complex domains like medical case-based retrieval, where relevance assessment requires analyzing both textual and visual information. In this work, we explore using a Multimodal Large Language Model (MLLM) to expand relevance judgments, creating a new dataset of automated judgments. Specifically, we employ Gemini 1.5 Pro on the ImageCLEFmed 2013 case-based retrieval task, simulating human assessment through an iteratively refined, structured prompting strategy that integrates binary scoring, instruction-based evaluation, and few-shot learning. We systematically experimented with various prompt configurations to maximize agreement with human judgments. To evaluate agreement between the MLLM and human judgments, we use Cohen's Kappa, achieving a substantial agreement score of 0.6, comparable to inter-annotator agreement typically observed in multimodal retrieval tasks. Starting from the original 15,028 manual judgments (4.72% relevant) across 35 topics, our MLLM-based approach expanded the dataset by over 37x to 558,653 judgments, increasing relevant annotations to 5,950. On average, each medical case query received 15,398 new annotations, with approximately 99% being non-relevant, reflecting the high sparsity typical in this domain. Our results demonstrate the potential of MLLMs to scale relevance judgment collection, offering a promising direction for supporting retrieval evaluation in medical and multimodal IR tasks. 

---
# CARTS: Collaborative Agents for Recommendation Textual Summarization 

**Authors**: Jiao Chen, Kehui Yao, Reza Yousefi Maragheh, Kai Zhao, Jianpeng Xu, Jason Cho, Evren Korpeoglu, Sushant Kumar, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2506.17765)  

**Abstract**: Current recommendation systems often require some form of textual data summarization, such as generating concise and coherent titles for product carousels or other grouped item displays. While large language models have shown promise in NLP domains for textual summarization, these approaches do not directly apply to recommendation systems, where explanations must be highly relevant to the core features of item sets, adhere to strict word limit constraints. In this paper, we propose CARTS (Collaborative Agents for Recommendation Textual Summarization), a multi-agent LLM framework designed for structured summarization in recommendation systems. CARTS decomposes the task into three stages-Generation Augmented Generation (GAG), refinement circle, and arbitration, where successive agent roles are responsible for extracting salient item features, iteratively refining candidate titles based on relevance and length feedback, and selecting the final title through a collaborative arbitration process. Experiments on large-scale e-commerce data and live A/B testing show that CARTS significantly outperforms single-pass and chain-of-thought LLM baselines, delivering higher title relevance and improved user engagement metrics. 

---
# Reinforcing User Interest Evolution in Multi-Scenario Learning for recommender systems 

**Authors**: Zhijian Feng, Wenhao Zheng, Xuanji Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.17682)  

**Abstract**: In real-world recommendation systems, users would engage in variety scenarios, such as homepages, search pages, and related recommendation pages. Each of these scenarios would reflect different aspects users focus on. However, the user interests may be inconsistent in different scenarios, due to differences in decision-making processes and preference expression. This variability complicates unified modeling, making multi-scenario learning a significant challenge. To address this, we propose a novel reinforcement learning approach that models user preferences across scenarios by modeling user interest evolution across multiple scenarios. Our method employs Double Q-learning to enhance next-item prediction accuracy and optimizes contrastive learning loss using Q-value to make model performance better. Experimental results demonstrate that our approach surpasses state-of-the-art methods in multi-scenario recommendation tasks. Our work offers a fresh perspective on multi-scenario modeling and highlights promising directions for future research. 

---
# A novel fast short-time root music method for vibration monitoring of high-speed spindles 

**Authors**: Huiguang Zhang, Baoguo Liu, Wei Feng, Zongtang Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.17600)  

**Abstract**: Ultra-high-speed spindle bearings challenge traditional vibration monitoring due to broadband noise, non-stationarity, and limited time-frequency resolution. We present a fast Short-Time Root-MUSIC (fSTrM) algorithm that exploits
FFT-accelerated Lanczos bidiagonalization to reduce computational complexity from $\mathcal{O}(N^3)$ to $SN\log_2N+S^2(N+S)+M^2(N+M)$
while preserving parametric super-resolution. The method constructs Hankel matrices from 16 ms signal frames and extracts fault frequencies through polynomial rooting on the unit circle. Experimental validation on the Politecnico di Torino bearing dataset demonstrates breakthrough micro-defect detection capabilities. The algorithm reliably identifies 150 $\mu$m defects -- previously undetectable by conventional methods -- providing 72+ hours additional warning time. Compared to STFT and wavelet methods, fSTrM achieves 1.2 Hz frequency resolution (vs. 12.5 Hz), 93\% detection rate at $-$5 dB SNR, and quantifies defect severity through harmonic content analysis. Critically, the algorithm processes each frame in 2.4 ms on embedded ARM Cortex-M7 hardware, enabling real-time deployment. This advancement transforms bearing monitoring from failure prevention to continuous degradation assessment, establishing a new paradigm for predictive maintenance in aerospace and precision machining. 

---
# Context-Aware Scientific Knowledge Extraction on Linked Open Data using Large Language Models 

**Authors**: Sajratul Y. Rubaiat, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17580)  

**Abstract**: The exponential growth of scientific literature challenges researchers extracting and synthesizing knowledge. Traditional search engines return many sources without direct, detailed answers, while general-purpose LLMs may offer concise responses that lack depth or omit current information. LLMs with search capabilities are also limited by context window, yielding short, incomplete answers. This paper introduces WISE (Workflow for Intelligent Scientific Knowledge Extraction), a system addressing these limits by using a structured workflow to extract, refine, and rank query-specific knowledge. WISE uses an LLM-powered, tree-based architecture to refine data, focusing on query-aligned, context-aware, and non-redundant information. Dynamic scoring and ranking prioritize unique contributions from each source, and adaptive stopping criteria minimize processing overhead. WISE delivers detailed, organized answers by systematically exploring and synthesizing knowledge from diverse sources. Experiments on HBB gene-associated diseases demonstrate WISE reduces processed text by over 80% while achieving significantly higher recall over baselines like search engines and other LLM-based approaches. ROUGE and BLEU metrics reveal WISE's output is more unique than other systems, and a novel level-based metric shows it provides more in-depth information. We also explore how the WISE workflow can be adapted for diverse domains like drug discovery, material science, and social science, enabling efficient knowledge extraction and synthesis from unstructured scientific papers and web sources. 

---
# PreQRAG -- Classify and Rewrite for Enhanced RAG 

**Authors**: Damian Martinez, Catalina Riano, Hui Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17493)  

**Abstract**: This paper presents the submission of the UDInfo team to the SIGIR 2025 LiveRAG Challenge. We introduce PreQRAG, a Retrieval Augmented Generation (RAG) architecture designed to improve retrieval and generation quality through targeted question preprocessing. PreQRAG incorporates a pipeline that first classifies each input question as either single-document or multi-document type. For single-document questions, we employ question rewriting techniques to improve retrieval precision and generation relevance. For multi-document questions, we decompose complex queries into focused sub-questions that can be processed more effectively by downstream components. This classification and rewriting strategy improves the RAG performance. Experimental evaluation of the LiveRAG Challenge dataset demonstrates the effectiveness of our question-type-aware architecture, with PreQRAG achieving the preliminary second place in Session 2 of the LiveRAG challenge. 

---
# SlimRAG: Retrieval without Graphs via Entity-Aware Context Selection 

**Authors**: Jiale Zhang, Jiaxiang Chen, Zhucong Li, Jie Ding, Kui Zhao, Zenglin Xu, Xin Pang, Yinghui Xu  

**Link**: [PDF](https://arxiv.org/pdf/2506.17288)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances language models by incorporating external knowledge at inference time. However, graph-based RAG systems often suffer from structural overhead and imprecise retrieval: they require costly pipelines for entity linking and relation extraction, yet frequently return subgraphs filled with loosely related or tangential content. This stems from a fundamental flaw -- semantic similarity does not imply semantic relevance. We introduce SlimRAG, a lightweight framework for retrieval without graphs. SlimRAG replaces structure-heavy components with a simple yet effective entity-aware mechanism. At indexing time, it constructs a compact entity-to-chunk table based on semantic embeddings. At query time, it identifies salient entities, retrieves and scores associated chunks, and assembles a concise, contextually relevant input -- without graph traversal or edge construction. To quantify retrieval efficiency, we propose Relative Index Token Utilization (RITU), a metric measuring the compactness of retrieved content. Experiments across multiple QA benchmarks show that SlimRAG outperforms strong flat and graph-based baselines in accuracy while reducing index size and RITU (e.g., 16.31 vs. 56+), highlighting the value of structure-free, entity-centric context selection. The code will be released soon. this https URL 

---
# Recommendation systems in e-commerce applications with machine learning methods 

**Authors**: Aneta Poniszewska-Maranda, Magdalena Pakula, Bozena Borowska  

**Link**: [PDF](https://arxiv.org/pdf/2506.17287)  

**Abstract**: E-commerce platforms are increasingly reliant on recommendation systems to enhance user experience, retain customers, and, in most cases, drive sales. The integration of machine learning methods into these systems has significantly improved their efficiency, personalization, and scalability. This paper aims to highlight the current trends in e-commerce recommendation systems, identify challenges, and evaluate the effectiveness of various machine learning methods used, including collaborative filtering, content-based filtering, and hybrid models. A systematic literature review (SLR) was conducted, analyzing 38 publications from 2013 to 2025. The methods used were evaluated and compared to determine their performance and effectiveness in addressing e-commerce challenges. 

---
# A Framework for Generating Conversational Recommendation Datasets from Behavioral Interactions 

**Authors**: Vinaik Chhetri, Yousaf Reza, Moghis Fereidouni, Srijata Maji, Umar Farooq, AB Siddique  

**Link**: [PDF](https://arxiv.org/pdf/2506.17285)  

**Abstract**: Modern recommendation systems typically follow two complementary paradigms: collaborative filtering, which models long-term user preferences from historical interactions, and conversational recommendation systems (CRS), which interact with users in natural language to uncover immediate needs. Each captures a different dimension of user intent. While CRS models lack collaborative signals, leading to generic or poorly personalized suggestions, traditional recommenders lack mechanisms to interactively elicit immediate needs. Unifying these paradigms promises richer personalization but remains challenging due to the lack of large-scale conversational datasets grounded in real user behavior. We present ConvRecStudio, a framework that uses large language models (LLMs) to simulate realistic, multi-turn dialogs grounded in timestamped user-item interactions and reviews. ConvRecStudio follows a three-stage pipeline: (1) Temporal Profiling, which constructs user profiles and community-level item sentiment trajectories over fine-grained aspects; (2) Semantic Dialog Planning, which generates a structured plan using a DAG of flexible super-nodes; and (3) Multi-Turn Simulation, which instantiates the plan using paired LLM agents for the user and system, constrained by executional and behavioral fidelity checks. We apply ConvRecStudio to three domains -- MobileRec, Yelp, and Amazon Electronics -- producing over 12K multi-turn dialogs per dataset. Human and automatic evaluations confirm the naturalness, coherence, and behavioral grounding of the generated conversations. To demonstrate utility, we build a cross-attention transformer model that jointly encodes user history and dialog context, achieving gains in Hit@K and NDCG@K over baselines using either signal alone or naive fusion. Notably, our model achieves a 10.9% improvement in Hit@1 on Yelp over the strongest baseline. 

---
# Automating Financial Statement Audits with Large Language Models 

**Authors**: Rushi Wang, Jiateng Liu, Weijie Zhao, Shenglan Li, Denghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17282)  

**Abstract**: Financial statement auditing is essential for stakeholders to understand a company's financial health, yet current manual processes are inefficient and error-prone. Even with extensive verification procedures, auditors frequently miss errors, leading to inaccurate financial statements that fail to meet stakeholder expectations for transparency and reliability. To this end, we harness large language models (LLMs) to automate financial statement auditing and rigorously assess their capabilities, providing insights on their performance boundaries in the scenario of automated auditing. Our work introduces a comprehensive benchmark using a curated dataset combining real-world financial tables with synthesized transaction data. In the benchmark, we developed a rigorous five-stage evaluation framework to assess LLMs' auditing capabilities. The benchmark also challenges models to map specific financial statement errors to corresponding violations of accounting standards, simulating real-world auditing scenarios through test cases. Our testing reveals that current state-of-the-art LLMs successfully identify financial statement errors when given historical transaction data. However, these models demonstrate significant limitations in explaining detected errors and citing relevant accounting standards. Furthermore, LLMs struggle to execute complete audits and make necessary financial statement revisions. These findings highlight a critical gap in LLMs' domain-specific accounting knowledge. Future research must focus on enhancing LLMs' understanding of auditing principles and procedures. Our benchmark and evaluation framework establish a foundation for developing more effective automated auditing tools that will substantially improve the accuracy and efficiency of real-world financial statement auditing. 

---
# CORONA: A Coarse-to-Fine Framework for Graph-based Recommendation with Large Language Models 

**Authors**: Junze Chen, Xinjie Yang, Cheng Yang, Junfei Bao, Zeyuan Guo, Yawen Li, Chuan Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.17281)  

**Abstract**: Recommender systems (RSs) are designed to retrieve candidate items a user might be interested in from a large pool. A common approach is using graph neural networks (GNNs) to capture high-order interaction relationships. As large language models (LLMs) have shown strong capabilities across domains, researchers are exploring their use to enhance recommendation. However, prior work limits LLMs to re-ranking results or dataset augmentation, failing to utilize their power during candidate filtering - which may lead to suboptimal performance. Instead, we propose to leverage LLMs' reasoning abilities during the candidate filtering process, and introduce Chain Of Retrieval ON grAphs (CORONA) to progressively narrow down the range of candidate items on interaction graphs with the help of LLMs: (1) First, LLM performs preference reasoning based on user profiles, with the response serving as a query to extract relevant users and items from the interaction graph as preference-assisted retrieval; (2) Then, using the information retrieved in the previous step along with the purchase history of target user, LLM conducts intent reasoning to help refine an even smaller interaction subgraph as intent-assisted retrieval; (3) Finally, we employ a GNN to capture high-order collaborative filtering information from the extracted subgraph, performing GNN-enhanced retrieval to generate the final recommendation results. The proposed framework leverages the reasoning capabilities of LLMs during the retrieval process, while seamlessly integrating GNNs to enhance overall recommendation performance. Extensive experiments on various datasets and settings demonstrate that our proposed CORONA achieves state-of-the-art performance with an 18.6% relative improvement in recall and an 18.4% relative improvement in NDCG on average. 

---
# Chunk Twice, Embed Once: A Systematic Study of Segmentation and Representation Trade-offs in Chemistry-Aware Retrieval-Augmented Generation 

**Authors**: Mahmoud Amiri, Thomas Bocklitz  

**Link**: [PDF](https://arxiv.org/pdf/2506.17277)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are increasingly vital for navigating the ever-expanding body of scientific literature, particularly in high-stakes domains such as chemistry. Despite the promise of RAG, foundational design choices -- such as how documents are segmented and represented -- remain underexplored in domain-specific contexts. This study presents the first large-scale, systematic evaluation of chunking strategies and embedding models tailored to chemistry-focused RAG systems. We investigate 25 chunking configurations across five method families and evaluate 48 embedding models on three chemistry-specific benchmarks, including the newly introduced QuestChemRetrieval dataset. Our results reveal that recursive token-based chunking (specifically R100-0) consistently outperforms other approaches, offering strong performance with minimal resource overhead. We also find that retrieval-optimized embeddings -- such as Nomic and Intfloat E5 variants -- substantially outperform domain-specialized models like SciBERT. By releasing our datasets, evaluation framework, and empirical benchmarks, we provide actionable guidelines for building effective and efficient chemistry-aware RAG systems. 

---
# QUST_NLP at SemEval-2025 Task 7: A Three-Stage Retrieval Framework for Monolingual and Crosslingual Fact-Checked Claim Retrieval 

**Authors**: Youzheng Liu, Jiyan Liu, Xiaoman Xu, Taihang Wang, Yimin Wang, Ye Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.17272)  

**Abstract**: This paper describes the participation of QUST_NLP in the SemEval-2025 Task 7. We propose a three-stage retrieval framework specifically designed for fact-checked claim retrieval. Initially, we evaluate the performance of several retrieval models and select the one that yields the best results for candidate retrieval. Next, we employ multiple re-ranking models to enhance the candidate results, with each model selecting the Top-10 outcomes. In the final stage, we utilize weighted voting to determine the final retrieval outcomes. Our approach achieved 5th place in the monolingual track and 7th place in the crosslingual track. We release our system code at: this https URL 

---
# jina-embeddings-v4: Universal Embeddings for Multimodal Multilingual Retrieval 

**Authors**: Michael Günther, Saba Sturua, Mohammad Kalim Akram, Isabelle Mohr, Andrei Ungureanu, Sedigheh Eslami, Scott Martens, Bo Wang, Nan Wang, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2506.18902)  

**Abstract**: We introduce jina-embeddings-v4, a 3.8 billion parameter multimodal embedding model that unifies text and image representations through a novel architecture supporting both single-vector and multi-vector embeddings in the late interaction style. The model incorporates task-specific Low-Rank Adaptation (LoRA) adapters to optimize performance across diverse retrieval scenarios, including query-based information retrieval, cross-modal semantic similarity, and programming code search. Comprehensive evaluations demonstrate that jina-embeddings-v4 achieves state-of-the-art performance on both single- modal and cross-modal retrieval tasks, with particular strength in processing visually rich content such as tables, charts, diagrams, and mixed-media formats. To facilitate evaluation of this capability, we also introduce Jina-VDR, a novel benchmark specifically designed for visually rich image retrieval. 

---
# When Fine-Tuning Fails: Lessons from MS MARCO Passage Ranking 

**Authors**: Manu Pande, Shahil Kumar, Anay Yatin Damle  

**Link**: [PDF](https://arxiv.org/pdf/2506.18535)  

**Abstract**: This paper investigates the counterintuitive phenomenon where fine-tuning pre-trained transformer models degrades performance on the MS MARCO passage ranking task. Through comprehensive experiments involving five model variants-including full parameter fine-tuning and parameter efficient LoRA adaptations-we demonstrate that all fine-tuning approaches underperform the base sentence-transformers/all- MiniLM-L6-v2 model (MRR@10: 0.3026). Our analysis reveals that fine-tuning disrupts the optimal embedding space structure learned during the base model's extensive pre-training on 1 billion sentence pairs, including 9.1 million MS MARCO samples. UMAP visualizations show progressive embedding space flattening, while training dynamics analysis and computational efficiency metrics further support our findings. These results challenge conventional wisdom about transfer learning effectiveness on saturated benchmarks and suggest architectural innovations may be necessary for meaningful improvements. 

---
# Mapping the Evolution of Research Contributions using KnoVo 

**Authors**: Sajratul Y. Rubaiat, Syed N. Sakib, Hasan M. Jamil  

**Link**: [PDF](https://arxiv.org/pdf/2506.17508)  

**Abstract**: This paper presents KnoVo (Knowledge Evolution), an intelligent framework designed for quantifying and analyzing the evolution of research novelty in the scientific literature. Moving beyond traditional citation analysis, which primarily measures impact, KnoVo determines a paper's novelty relative to both prior and subsequent work within its multilayered citation network. Given a target paper's abstract, KnoVo utilizes Large Language Models (LLMs) to dynamically extract dimensions of comparison (e.g., methodology, application, dataset). The target paper is then compared to related publications along these same extracted dimensions. This comparative analysis, inspired by tournament selection, yields quantitative novelty scores reflecting the relative improvement, equivalence, or inferiority of the target paper in specific aspects. By aggregating these scores and visualizing their progression, for instance, through dynamic evolution graphs and comparative radar charts, KnoVo facilitates researchers not only to assess originality and identify similar work, but also to track knowledge evolution along specific research dimensions, uncover research gaps, and explore cross-disciplinary connections. We demonstrate these capabilities through a detailed analysis of 20 diverse papers from multiple scientific fields and report on the performance of various open-source LLMs within the KnoVo framework. 

---
# From Drawings to Decisions: A Hybrid Vision-Language Framework for Parsing 2D Engineering Drawings into Structured Manufacturing Knowledge 

**Authors**: Muhammad Tayyab Khan, Lequn Chen, Zane Yong, Jun Ming Tan, Wenhe Feng, Seung Ki Moon  

**Link**: [PDF](https://arxiv.org/pdf/2506.17374)  

**Abstract**: Efficient and accurate extraction of key information from 2D engineering drawings is essential for advancing digital manufacturing workflows. Such information includes geometric dimensioning and tolerancing (GD&T), measures, material specifications, and textual annotations. Manual extraction is slow and labor-intensive, while generic OCR models often fail due to complex layouts, engineering symbols, and rotated text, leading to incomplete and unreliable outputs. These limitations result in incomplete and unreliable outputs. To address these challenges, we propose a hybrid vision-language framework that integrates a rotation-aware object detection model (YOLOv11-obb) with a transformer-based vision-language parser. Our structured pipeline applies YOLOv11-OBB to localize annotations and extract oriented bounding box (OBB) patches, which are then parsed into structured outputs using a fine-tuned, lightweight vision-language model (VLM). We curate a dataset of 1,367 2D mechanical drawings annotated across nine key categories. YOLOv11-OBB is trained on this dataset to detect OBBs and extract annotation patches. These are parsed using two open-source VLMs: Donut and Florence-2. Both models are lightweight and well-suited for specialized industrial tasks under limited computational overhead. Following fine-tuning of both models on the curated dataset of image patches paired with structured annotation labels, a comparative experiment is conducted to evaluate parsing performance across four key metrics. Donut outperforms Florence-2, achieving 88.5% precision, 99.2% recall, and a 93.5% F1-score, with a hallucination rate of 11.5%. Finally, a case study demonstrates how the extracted structured information supports downstream manufacturing tasks such as process and tool selection, showcasing the practical utility of the proposed framework in modernizing 2D drawing interpretation. 

---
