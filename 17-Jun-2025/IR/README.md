# OneRec Technical Report 

**Authors**: Guorui Zhou, Jiaxin Deng, Jinghao Zhang, Kuo Cai, Lejian Ren, Qiang Luo, Qianqian Wang, Qigen Hu, Rui Huang, Shiyao Wang, Weifeng Ding, Wuchao Li, Xinchen Luo, Xingmei Wang, Zexuan Cheng, Zixing Zhang, Bin Zhang, Boxuan Wang, Chaoyi Ma, Chengru Song, Chenhui Wang, Di Wang, Dongxue Meng, Fan Yang, Fangyu Zhang, Feng Jiang, Fuxing Zhang, Gang Wang, Guowang Zhang, Han Li, Hengrui Hu, Hezheng Lin, Hongtao Cheng, Hongyang Cao, Huanjie Wang, Jiaming Huang, Jiapeng Chen, Jiaqiang Liu, Jinghui Jia, Kun Gai, Lantao Hu, Liang Zeng, Liao Yu, Qiang Wang, Qidong Zhou, Shengzhe Wang, Shihui He, Shuang Yang, Shujie Yang, Sui Huang, Tao Wu, Tiantian He, Tingting Gao, Wei Yuan, Xiao Liang, Xiaoxiao Xu, Xugang Liu, Yan Wang, Yi Wang, Yiwu Liu, Yue Song, Yufei Zhang, Yunfan Wu, Yunfeng Zhao, Zhanyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.13695)  

**Abstract**: Recommender systems have been widely used in various large-scale user-oriented platforms for many years. However, compared to the rapid developments in the AI community, recommendation systems have not achieved a breakthrough in recent years. For instance, they still rely on a multi-stage cascaded architecture rather than an end-to-end approach, leading to computational fragmentation and optimization inconsistencies, and hindering the effective application of key breakthrough technologies from the AI community in recommendation scenarios.
To address these issues, we propose OneRec, which reshapes the recommendation system through an end-to-end generative approach and achieves promising results. Firstly, we have enhanced the computational FLOPs of the current recommendation model by 10 $\times$ and have identified the scaling laws for recommendations within certain boundaries. Secondly, reinforcement learning techniques, previously difficult to apply for optimizing recommendations, show significant potential in this framework. Lastly, through infrastructure optimizations, we have achieved 23.7% and 28.8% Model FLOPs Utilization (MFU) on flagship GPUs during training and inference, respectively, aligning closely with the LLM community. This architecture significantly reduces communication and storage overhead, resulting in operating expense that is only 10.6% of traditional recommendation pipelines. Deployed in Kuaishou/Kuaishou Lite APP, it handles 25% of total queries per second, enhancing overall App Stay Time by 0.54% and 1.24%, respectively. Additionally, we have observed significant increases in metrics such as 7-day Lifetime, which is a crucial indicator of recommendation experience. We also provide practical lessons and insights derived from developing, optimizing, and maintaining a production-scale recommendation system with significant real-world impact. 

---
# Tree-Based Text Retrieval via Hierarchical Clustering in RAGFrameworks: Application on Taiwanese Regulations 

**Authors**: Chia-Heng Yu, Yen-Lung Tsai  

**Link**: [PDF](https://arxiv.org/pdf/2506.13607)  

**Abstract**: Traditional Retrieval-Augmented Generation (RAG) systems employ brute-force inner product search to retrieve the top-k most similar documents, then combined with the user query and passed to a language model. This allows the model to access external knowledge and reduce hallucinations. However, selecting an appropriate k value remains a significant challenge in practical applications: a small k may fail to retrieve sufficient information, while a large k can introduce excessive and irrelevant content. To address this, we propose a hierarchical clustering-based retrieval method that eliminates the need to predefine k. Our approach maintains the accuracy and relevance of system responses while adaptively selecting semantically relevant content. In the experiment stage, we applied our method to a Taiwanese legal dataset with expert-graded queries. The results show that our approach achieves superior performance in expert evaluations and maintains high precision while eliminating the need to predefine k, demonstrating improved accuracy and interpretability in legal text retrieval tasks. Our framework is simple to implement and easily integrates with existing RAG pipelines, making it a practical solution for real-world applications under limited resources. 

---
# Beyond One-Size-Fits-All: A Study of Neural and Behavioural Variability Across Different Recommendation Categories 

**Authors**: Georgios Koutroumpas, Sebastian Idesis, Mireia Masias Bruns, Carlos Segura, Joemon M. Jose, Sergi Abadal, Ioannis Arapakis  

**Link**: [PDF](https://arxiv.org/pdf/2506.13409)  

**Abstract**: Traditionally, Recommender Systems (RS) have primarily measured performance based on the accuracy and relevance of their recommendations. However, this algorithmic-centric approach overlooks how different types of recommendations impact user engagement and shape the overall quality of experience. In this paper, we shift the focus to the user and address for the first time the challenge of decoding the neural and behavioural variability across distinct recommendation categories, considering more than just relevance. Specifically, we conducted a controlled study using a comprehensive e-commerce dataset containing various recommendation types, and collected Electroencephalography and behavioural data. We analysed both neural and behavioural responses to recommendations that were categorised as Exact, Substitute, Complement, or Irrelevant products within search query results. Our findings offer novel insights into user preferences and decision-making processes, revealing meaningful relationships between behavioural and neural patterns for each category, but also indicate inter-subject variability. 

---
# Gated Rotary-Enhanced Linear Attention for Long-term Sequential Recommendation 

**Authors**: Juntao Hu, Wei Zhou, Huayi Shen, Xiao Du, Jie Liao, Junhao Wen, Min Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.13315)  

**Abstract**: In Sequential Recommendation Systems (SRSs), Transformer models show remarkable performance but face computation cost challenges when modeling long-term user behavior sequences due to the quadratic complexity of the dot-product attention mechanism. By approximating the dot-product attention, linear attention provides an efficient option with linear complexity. However, existing linear attention methods face two limitations: 1) they often use learnable position encodings, which incur extra computational costs in long-term sequence scenarios, and 2) they may not consider the user's fine-grained local preferences and confuse these with the actual change of long-term interests. To remedy these drawbacks, we propose a long-term sequential Recommendation model with Gated Rotary Enhanced Linear Attention (RecGRELA). Specifically, we first propose a Rotary-Enhanced Linear Attention (RELA) module to model long-range dependency within the user's historical information using rotary position encodings. We then introduce a local short operation to incorporate local preferences and demonstrate the theoretical insight. We further introduce a SiLU-based Gated mechanism for RELA (GRELA) to help the model determine whether a user's behavior indicates local interest or a genuine shift in long-term preferences. Experimental results on four public datasets demonstrate that our RecGRELA achieves state-of-the-art performance compared to existing SRSs while maintaining low memory overhead. 

---
# SPOT: Bridging Natural Language and Geospatial Search for Investigative Journalists 

**Authors**: Lynn Khellaf, Ipek Baris Schlicht, Tilman Mirass, Julia Bayer, Tilman Wagner, Ruben Bouwmeester  

**Link**: [PDF](https://arxiv.org/pdf/2506.13188)  

**Abstract**: OpenStreetMap (OSM) is a vital resource for investigative journalists doing geolocation verification. However, existing tools to query OSM data such as Overpass Turbo require familiarity with complex query languages, creating barriers for non-technical users. We present SPOT, an open source natural language interface that makes OSM's rich, tag-based geographic data more accessible through intuitive scene descriptions. SPOT interprets user inputs as structured representations of geospatial object configurations using fine-tuned Large Language Models (LLMs), with results being displayed in an interactive map interface. While more general geospatial search tasks are conceivable, SPOT is specifically designed for use in investigative journalism, addressing real-world challenges such as hallucinations in model output, inconsistencies in OSM tagging, and the noisy nature of user input. It combines a novel synthetic data pipeline with a semantic bundling system to enable robust, accurate query generation. To our knowledge, SPOT is the first system to achieve reliable natural language access to OSM data at this level of accuracy. By lowering the technical barrier to geolocation verification, SPOT contributes a practical tool to the broader efforts to support fact-checking and combat disinformation. 

---
# Identifying and Investigating Global News Coverage of Critical Events Such as Disasters and Terrorist Attacks 

**Authors**: Erica Cai, Xi Chen, Reagan Grey Keeney, Ethan Zuckerman, Brendan O'Connor, Przemyslaw A. Grabowicz  

**Link**: [PDF](https://arxiv.org/pdf/2506.12925)  

**Abstract**: Comparative studies of news coverage are challenging to conduct because methods to identify news articles about the same event in different languages require expertise that is difficult to scale. We introduce an AI-powered method for identifying news articles based on an event FINGERPRINT, which is a minimal set of metadata required to identify critical events. Our event coverage identification method, FINGERPRINT TO ARTICLE MATCHING FOR EVENTS (FAME), efficiently identifies news articles about critical world events, specifically terrorist attacks and several types of natural disasters. FAME does not require training data and is able to automatically and efficiently identify news articles that discuss an event given its fingerprint: time, location, and class (such as storm or flood). The method achieves state-of-the-art performance and scales to massive databases of tens of millions of news articles and hundreds of events happening globally. We use FAME to identify 27,441 articles that cover 470 natural disaster and terrorist attack events that happened in 2020. To this end, we use a massive database of news articles in three languages from MediaCloud, and three widely used, expert-curated databases of critical events: EM-DAT, USGS, and GTD. Our case study reveals patterns consistent with prior literature: coverage of disasters and terrorist attacks correlates to death counts, to the GDP of a country where the event occurs, and to trade volume between the reporting country and the country where the event occurred. We share our NLP annotations and cross-country media attention data to support the efforts of researchers and media monitoring organizations. 

---
# Hierarchical Group-wise Ranking Framework for Recommendation Models 

**Authors**: YaChen Yan, Liubo Li, Ravi Choudhary  

**Link**: [PDF](https://arxiv.org/pdf/2506.12756)  

**Abstract**: In modern recommender systems, CTR/CVR models are increasingly trained with ranking objectives to improve item ranking quality. While this shift aligns training more closely with serving goals, most existing methods rely on in-batch negative sampling, which predominantly surfaces easy negatives. This limits the model's ability to capture fine-grained user preferences and weakens overall ranking performance. To address this, we propose a Hierarchical Group-wise Ranking Framework with two key components. First, we apply residual vector quantization to user embeddings to generate hierarchical user codes that partition users into hierarchical, trie-structured clusters. Second, we apply listwise ranking losses to user-item pairs at each level of the hierarchy, where shallow levels group loosely similar users and deeper levels group highly similar users, reinforcing learning-to-rank signals through progressively harder negatives. Since users with similar preferences and content exposure tend to yield more informative negatives, applying ranking losses within these hierarchical user groups serves as an effective approximation of hard negative mining. Our approach improves ranking performance without requiring complex real-time context collection or retrieval infrastructure. Extensive experiments demonstrate that the proposed framework consistently enhances both model calibration and ranking accuracy, offering a scalable and practical solution for industrial recommender systems. 

---
# Device-Cloud Collaborative Correction for On-Device Recommendation 

**Authors**: Tianyu Zhan, Shengyu Zhang, Zheqi Lv, Jieming Zhu, Jiwei Li, Fan Wu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.12687)  

**Abstract**: With the rapid development of recommendation models and device computing power, device-based recommendation has become an important research area due to its better real-time performance and privacy protection. Previously, Transformer-based sequential recommendation models have been widely applied in this field because they outperform Recurrent Neural Network (RNN)-based recommendation models in terms of performance. However, as the length of interaction sequences increases, Transformer-based models introduce significantly more space and computational overhead compared to RNN-based models, posing challenges for device-based recommendation. To balance real-time performance and high performance on devices, we propose Device-Cloud \underline{Co}llaborative \underline{Corr}ection Framework for On-Device \underline{Rec}ommendation (CoCorrRec). CoCorrRec uses a self-correction network (SCN) to correct parameters with extremely low time cost. By updating model parameters during testing based on the input token, it achieves performance comparable to current optimal but more complex Transformer-based models. Furthermore, to prevent SCN from overfitting, we design a global correction network (GCN) that processes hidden states uploaded from devices and provides a global correction solution. Extensive experiments on multiple datasets show that CoCorrRec outperforms existing Transformer-based and RNN-based device recommendation models in terms of performance, with fewer parameters and lower FLOPs, thereby achieving a balance between real-time performance and high efficiency. 

---
# INTERPOS: Interaction Rhythm Guided Positional Morphing for Mobile App Recommender Systems 

**Authors**: M.H. Maqbool, Moghis Fereidouni, Umar Farooq, A.B. Siddique, Hassan Foroosh  

**Link**: [PDF](https://arxiv.org/pdf/2506.12661)  

**Abstract**: The mobile app market has expanded exponentially, offering millions of apps with diverse functionalities, yet research in mobile app recommendation remains limited. Traditional sequential recommender systems utilize the order of items in users' historical interactions to predict the next item for the users. Position embeddings, well-established in transformer-based architectures for natural language processing tasks, effectively distinguish token positions in sequences. In sequential recommendation systems, position embeddings can capture the order of items in a user's historical interaction sequence. Nevertheless, this ordering does not consider the time elapsed between two interactions of the same user (e.g., 1 day, 1 week, 1 month), referred to as "user rhythm". In mobile app recommendation datasets, the time between consecutive user interactions is notably longer compared to other domains like movies, posing significant challenges for sequential recommender systems. To address this phenomenon in the mobile app domain, we introduce INTERPOS, an Interaction Rhythm Guided Positional Morphing strategy for autoregressive mobile app recommender systems. INTERPOS incorporates rhythm-guided position embeddings, providing a more comprehensive representation that considers both the sequential order of interactions and the temporal gaps between them. This approach enables a deep understanding of users' rhythms at a fine-grained level, capturing the intricacies of their interaction patterns over time. We propose three strategies to incorporate the morphed positional embeddings in two transformer-based sequential recommendation system architectures. Our extensive evaluations show that INTERPOS outperforms state-of-the-art models using 7 mobile app recommendation datasets on NDCG@K and HIT@K metrics. The source code of INTERPOS is available at this https URL. 

---
# A Gradient Meta-Learning Joint Optimization for Beamforming and Antenna Position in Pinching-Antenna Systems 

**Authors**: Kang Zhou, Weixi Zhou, Donghong Cai, Xianfu Lei, Yanqing Xu, Zhiguo Ding, Pingzhi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2506.12583)  

**Abstract**: In this paper, we consider a novel optimization design for multi-waveguide pinching-antenna systems, aiming to maximize the weighted sum rate (WSR) by jointly optimizing beamforming coefficients and antenna position. To handle the formulated non-convex problem, a gradient-based meta-learning joint optimization (GML-JO) algorithm is proposed. Specifically, the original problem is initially decomposed into two sub-problems of beamforming optimization and antenna position optimization through equivalent substitution. Then, the convex approximation methods are used to deal with the nonconvex constraints of sub-problems, and two sub-neural networks are constructed to calculate the sub-problems separately. Different from alternating optimization (AO), where two sub-problems are solved alternately and the solutions are influenced by the initial values, two sub-neural networks of proposed GML-JO with fixed channel coefficients are considered as local sub-tasks and the computation results are used to calculate the loss function of joint optimization. Finally, the parameters of sub-networks are updated using the average loss function over different sub-tasks and the solution that is robust to the initial value is obtained. Simulation results demonstrate that the proposed GML-JO algorithm achieves 5.6 bits/s/Hz WSR within 100 iterations, yielding a 32.7\% performance enhancement over conventional AO with substantially reduced computational complexity. Moreover, the proposed GML-JO algorithm is robust to different choices of initialization and yields better performance compared with the existing optimization methods. 

---
# T-TExTS (Teaching Text Expansion for Teacher Scaffolding): Enhancing Text Selection in High School Literature through Knowledge Graph-Based Recommendation 

**Authors**: Nirmal Gelal, Chloe Snow, Ambyr Rios, Hande Küçük McGinty  

**Link**: [PDF](https://arxiv.org/pdf/2506.12075)  

**Abstract**: The implementation of transformational pedagogy in secondary education classrooms requires a broad multiliteracy approach. Due to limited planning time and resources, high school English Literature teachers often struggle to curate diverse, thematically aligned literature text sets. This study addresses the critical need for a tool that provides scaffolds for novice educators in selecting literature texts that are diverse -- in terms of genre, theme, subtheme, and author -- yet similar in context and pedagogical merits. We have developed a recommendation system, Teaching Text Expansion for Teacher Scaffolding (T-TExTS), that suggests high school English Literature books based on pedagogical merits, genre, and thematic relevance using a knowledge graph. We constructed a domain-specific ontology using the KNowledge Acquisition and Representation Methodology (KNARM), transformed into a knowledge graph, which was then embedded using DeepWalk, biased random walk, and a hybrid of both approaches. The system was evaluated using link prediction and recommendation performance metrics, including Area Under the Curve (AUC), Mean Reciprocal Rank (MRR), Hits@K, and normalized Discounted Cumulative Gain (nDCG). DeepWalk outperformed in most ranking metrics, with the highest AUC (0.9431), whereas the hybrid model offered balanced performance. These findings demonstrate the importance of semantic, ontology-driven approaches in recommendation systems and suggest that T-TExTS can significantly ease the burden of English Literature text selection for high school educators, promoting more informed and inclusive curricular decisions. The source code for T-TExTS is available at: this https URL 

---
# WebTrust: An AI-Driven Data Scoring System for Reliable Information Retrieval 

**Authors**: Joydeep Chandra, Aleksandr Algazinov, Satyam Kumar Navneet, Rim El Filali, Matt Laing, Andrew Hanna  

**Link**: [PDF](https://arxiv.org/pdf/2506.12072)  

**Abstract**: As access to information becomes more open and widespread, people are increasingly using AI tools for assistance. However, many of these tools struggle to estimate the trustworthiness of the information. Although today's search engines include AI features, they often fail to offer clear indicators of data reliability. To address this gap, we introduce WebTrust, a system designed to simplify the process of finding and judging credible information online. Built on a fine-tuned version of IBM's Granite-1B model and trained on a custom dataset, WebTrust works by assigning a reliability score (from 0.1 to 1) to each statement it processes. In addition, it offers a clear justification for why a piece of information received that score. Evaluated using prompt engineering, WebTrust consistently achieves superior performance compared to other small-scale LLMs and rule-based approaches, outperforming them across all experiments on MAE, RMSE, and R2. User testing showed that when reliability scores are displayed alongside search results, people feel more confident and satisfied with the information they find. With its accuracy, transparency, and ease of use, WebTrust offers a practical solution to help combat misinformation and make trustworthy information more accessible to everyone. 

---
# T$^2$-RAGBench: Text-and-Table Benchmark for Evaluating Retrieval-Augmented Generation 

**Authors**: Jan Strich, Enes Kutay Isgorur, Maximilian Trescher, Chris Biemann, Martin Semmann  

**Link**: [PDF](https://arxiv.org/pdf/2506.12071)  

**Abstract**: While most financial documents contain a combination of textual and tabular information, robust Retrieval-Augmented Generation (RAG) systems are essential for effectively accessing and reasoning over such content to perform complex numerical tasks. This paper introduces T$^2$-RAGBench, a benchmark comprising 32,908 question-context-answer triples, designed to evaluate RAG methods on real-world financial data. Unlike typical QA datasets that operate under Oracle-context settings, where the relevant context is explicitly provided, T$^2$-RAGBench challenges models to first retrieve the correct context before conducting numerical reasoning. Existing QA datasets involving text and tables typically contain context-dependent questions, which may yield multiple correct answers depending on the provided context. To address this, we transform these datasets into a context-independent format, enabling reliable RAG evaluation. We conduct a comprehensive evaluation of popular RAG methods. Our analysis identifies Hybrid BM25, a technique that combines dense and sparse vectors, as the most effective approach for text-and-table data. However, results demonstrate that T$^2$-RAGBench remains challenging even for SOTA LLMs and RAG methods. Further ablation studies examine the impact of embedding models and corpus size on retrieval performance. T$^2$-RAGBench provides a realistic and rigorous benchmark for existing RAG methods on text-and-table data. Code and dataset are available online. 

---
# Algorithms for estimating linear function in data mining 

**Authors**: Thomas Hoang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12069)  

**Abstract**: The main goal of this topic is to showcase several studied algorithms for estimating the linear utility function to predict the users preferences. For example, if a user comes to buy a car that has several attributes including speed, color, age, etc in a linear function, the algorithms that we present in this paper help with estimating this linear function to filter out a small subset that would be of best interest to the user among a million tuples in a very large database. In addition, the estimating linear function could also be applicable in getting to know what the data can do or predicting the future based on the data that is used in data science, which is demonstrated by the GNN, PLOD algorithms. In the ever-evolving field of data science, deriving valuable insights from large datasets is critical for informed decision-making, particularly in predictive applications. Data analysts often identify high-quality datasets without missing values, duplicates, or inconsistencies before merging diverse attributes for analysis. Taking housing price prediction as a case study, various attributes must be considered, including location factors (proximity to urban centers, crime rates), property features (size, style, modernity), and regional policies (tax implications). Experts in the field typically rank these attributes to establish a predictive utility function, which machine learning models use to forecast outcomes like housing prices. Several data discovery algorithms, including those that address the challenges of predefined utility functions and human input for attribute ranking, which often result in a time-consuming iterative process, that the work of cannot overcome. 

---
# LTRR: Learning To Rank Retrievers for LLMs 

**Authors**: To Eun Kim, Fernando Diaz  

**Link**: [PDF](https://arxiv.org/pdf/2506.13743)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems typically rely on a single fixed retriever, despite growing evidence that no single retriever performs optimally across all query types. In this paper, we explore a query routing approach that dynamically selects from a pool of retrievers based on the query, using both train-free heuristics and learned routing models. We frame routing as a learning-to-rank (LTR) problem and introduce LTRR, a framework that learns to rank retrievers by their expected utility gain to downstream LLM performance. Our experiments, conducted on synthetic QA data with controlled query type variations, show that routing-based RAG systems can outperform the best single-retriever-based systems. Performance gains are especially pronounced in models trained with the Answer Correctness (AC) metric and with pairwise learning approaches, especially with XGBoost. We also observe improvements in generalization to out-of-distribution queries. As part of the SIGIR 2025 LiveRAG challenge, our submitted system demonstrated the practical viability of our approach, achieving competitive performance in both answer correctness and faithfulness. These findings highlight the importance of both training methodology and metric selection in query routing for RAG systems. 

---
# Hierarchical Multi-Positive Contrastive Learning for Patent Image Retrieval 

**Authors**: Kshitij Kavimandan, Angelos Nalmpantis, Emma Beauxis-Aussalet, Robert-Jan Sips  

**Link**: [PDF](https://arxiv.org/pdf/2506.13496)  

**Abstract**: Patent images are technical drawings that convey information about a patent's innovation. Patent image retrieval systems aim to search in vast collections and retrieve the most relevant images. Despite recent advances in information retrieval, patent images still pose significant challenges due to their technical intricacies and complex semantic information, requiring efficient fine-tuning for domain adaptation. Current methods neglect patents' hierarchical relationships, such as those defined by the Locarno International Classification (LIC) system, which groups broad categories (e.g., "furnishing") into subclasses (e.g., "seats" and "beds") and further into specific patent designs. In this work, we introduce a hierarchical multi-positive contrastive loss that leverages the LIC's taxonomy to induce such relations in the retrieval process. Our approach assigns multiple positive pairs to each patent image within a batch, with varying similarity scores based on the hierarchical taxonomy. Our experimental analysis with various vision and multimodal models on the DeepPatent2 dataset shows that the proposed method enhances the retrieval results. Notably, our method is effective with low-parameter models, which require fewer computational resources and can be deployed on environments with limited hardware. 

---
# Decompositional Reasoning for Graph Retrieval with Large Language Models 

**Authors**: Valentin Six, Evan Dufraisse, Gaël de Chalendar  

**Link**: [PDF](https://arxiv.org/pdf/2506.13380)  

**Abstract**: Large Language Models (LLMs) excel at many NLP tasks, but struggle with multi-hop reasoning and factual consistency, limiting their effectiveness on knowledge-intensive tasks like complex question answering (QA). Linking Knowledge Graphs (KG) and LLMs has shown promising results, but LLMs generally lack the ability to reason efficiently over graph-structured information. To tackle this problem, we propose a novel retrieval approach that integrates textual knowledge graphs into the LLM reasoning process via query decomposition. Our method decomposes complex questions into sub-questions, retrieves relevant textual subgraphs, and composes a question-specific knowledge graph to guide answer generation. For that, we use a weighted similarity function that focuses on both the complex question and the generated subquestions to extract a relevant subgraph, which allows efficient and precise retrieval for complex questions and improves the performance of LLMs on multi-hop QA tasks. This structured reasoning pipeline enhances factual grounding and interpretability while leveraging the generative strengths of LLMs. We evaluate our method on standard multi-hop QA benchmarks and show that it achieves comparable or superior performance to competitive existing methods, using smaller models and fewer LLM calls. 

---
# Digital Transformation of Urban Planning in Australia: Influencing Factors and Key Challenges 

**Authors**: Soheil Sabri, Sherah Kurnia  

**Link**: [PDF](https://arxiv.org/pdf/2506.13333)  

**Abstract**: Over the past two decades, several governments in developing and developed countries have started their journey toward digital transformation. However, the pace and maturity of digital technologies and strategies are different between public services. Current literature indicates that research on the digital transformation of urban planning is still developing. Therefore, the aim of this study is to understand the influencing factors and key challenges for the digital transformation of urban planning in Australia. The study adopts the inter-organisational theory and Planning Support Science (PSScience) under the Technological, Organisational, and External Environmental (TOE) framework. It involves a multiple case study, administered semi-structured interviews with thirteen IT and urban planning experts across Victoria and New South Wales governments and private industries. The study findings indicate that the main challenges for digital transformation of the Australian urban planning system are related to organisational and external environmental factors. Furthermore, a digital maturity model is absent in the Australian urban planning industry. This study offers important implications to research and practice related to digital transformation in urban planning. 

---
# Accessibility Barriers in Multi-Terabyte Public Datasets: The Gap Between Promise and Practice 

**Authors**: Marc Bara  

**Link**: [PDF](https://arxiv.org/pdf/2506.13256)  

**Abstract**: The promise of "free and open" multi-terabyte datasets often collides with harsh realities. While these datasets may be technically accessible, practical barriers -- from processing complexity to hidden costs -- create a system that primarily serves well-funded institutions. This study examines accessibility challenges across web crawls, satellite imagery, scientific data, and collaborative projects, revealing a consistent two-tier system where theoretical openness masks practical exclusivity. Our analysis demonstrates that datasets marketed as "publicly accessible" typically require minimum investments of \$1,000+ for meaningful analysis, with complex processing pipelines demanding \$10,000-100,000+ in infrastructure costs. The infrastructure requirements -- distributed computing knowledge, domain expertise, and substantial budgets -- effectively gatekeep these datasets despite their "open" status, limiting practical accessibility to those with institutional support or substantial resources. 

---
# Vector Ontologies as an LLM world view extraction method 

**Authors**: Kaspar Rothenfusser, Bekk Blando  

**Link**: [PDF](https://arxiv.org/pdf/2506.13252)  

**Abstract**: Large Language Models (LLMs) possess intricate internal representations of the world, yet these latent structures are notoriously difficult to interpret or repurpose beyond the original prediction task. Building on our earlier work (Rothenfusser, 2025), which introduced the concept of vector ontologies as a framework for translating high-dimensional neural representations into interpretable geometric structures, this paper provides the first empirical validation of that approach. A vector ontology defines a domain-specific vector space spanned by ontologically meaningful dimensions, allowing geometric analysis of concepts and relationships within a domain. We construct an 8-dimensional vector ontology of musical genres based on Spotify audio features and test whether an LLM's internal world model of music can be consistently and accurately projected into this space. Using GPT-4o-mini, we extract genre representations through multiple natural language prompts and analyze the consistency of these projections across linguistic variations and their alignment with ground-truth data. Our results show (1) high spatial consistency of genre projections across 47 query formulations, (2) strong alignment between LLM-inferred genre locations and real-world audio feature distributions, and (3) evidence of a direct relationship between prompt phrasing and spatial shifts in the LLM's inferred vector ontology. These findings demonstrate that LLMs internalize structured, repurposable knowledge and that vector ontologies offer a promising method for extracting and analyzing this knowledge in a transparent and verifiable way. 

---
# Efficient Neuro-Symbolic Retrieval-Augmented Generation through Adaptive Query Routing 

**Authors**: Safayat Bin Hakim, Muhammad Adil, Alvaro Velasquez, Houbing Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2506.12981)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems address factual inconsistencies in Large Language Models by grounding generation in external knowledge, yet they face a fundamental efficiency problem: simple queries consume computational resources equivalent to complex multi-hop reasoning tasks. We present SymRAG, a neuro-symbolic framework that introduces adaptive query routing based on real-time complexity and system load assessments. SymRAG dynamically selects symbolic, neural, or hybrid processing paths to align resource use with query demands. Evaluated on 2,000 queries from HotpotQA and DROP using Llama-3.2-3B and Mistral-7B models, SymRAG achieves 97.6--100.0% exact match accuracy with significantly lower CPU utilization (3.6--6.2%) and processing time (0.985--3.165s). Disabling adaptive logic results in 169--1151% increase in processing time, highlighting the framework's impact. These results underscore the potential of adaptive neuro-symbolic routing for scalable, sustainable AI systems. 

---
# Assessing the Performance Gap Between Lexical and Semantic Models for Information Retrieval With Formulaic Legal Language 

**Authors**: Larissa Mori, Carlos Sousa de Oliveira, Yuehwern Yih, Mario Ventresca  

**Link**: [PDF](https://arxiv.org/pdf/2506.12895)  

**Abstract**: Legal passage retrieval is an important task that assists legal practitioners in the time-intensive process of finding relevant precedents to support legal arguments. This study investigates the task of retrieving legal passages or paragraphs from decisions of the Court of Justice of the European Union (CJEU), whose language is highly structured and formulaic, leading to repetitive patterns. Understanding when lexical or semantic models are more effective at handling the repetitive nature of legal language is key to developing retrieval systems that are more accurate, efficient, and transparent for specific legal domains. To this end, we explore when this routinized legal language is better suited for retrieval using methods that rely on lexical and statistical features, such as BM25, or dense retrieval models trained to capture semantic and contextual information. A qualitative and quantitative analysis with three complementary metrics shows that both lexical and dense models perform well in scenarios with more repetitive usage of language, whereas BM25 performs better than the dense models in more nuanced scenarios where repetition and verbatim~quotes are less prevalent and in longer queries. Our experiments also show that BM25 is a strong baseline, surpassing off-the-shelf dense models in 4 out of 7 performance metrics. However, fine-tuning a dense model on domain-specific data led to improved performance, surpassing BM25 in most metrics, and we analyze the effect of the amount of data used in fine-tuning on the model's performance and temporal robustness. The code, dataset and appendix related to this work are available on: this https URL. 

---
# Versatile and Fast Location-Based Private Information Retrieval with Fully Homomorphic Encryption over the Torus 

**Authors**: Joon Soo Yoo, Taeho Kim, Ji Won Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2506.12761)  

**Abstract**: Location-based services often require users to share sensitive locational data, raising privacy concerns due to potential misuse or exploitation by untrusted servers. In response, we present VeLoPIR, a versatile location-based private information retrieval (PIR) system designed to preserve user privacy while enabling efficient and scalable query processing. VeLoPIR introduces three operational modes-interval validation, coordinate validation, and identifier matching-that support a broad range of real-world applications, including information and emergency alerts. To enhance performance, VeLoPIR incorporates multi-level algorithmic optimizations with parallel structures, achieving significant scalability across both CPU and GPU platforms. We also provide formal security and privacy proofs, confirming the system's robustness under standard cryptographic assumptions. Extensive experiments on real-world datasets demonstrate that VeLoPIR achieves up to 11.55 times speed-up over a prior baseline. The implementation of VeLoPIR is publicly available at this https URL. 

---
# SciSage: A Multi-Agent Framework for High-Quality Scientific Survey Generation 

**Authors**: Xiaofeng Shi, Qian Kou, Yuduo Li, Ning Tang, Jinxin Xie, Longbin Yu, Songjing Wang, Hua Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.12689)  

**Abstract**: The rapid growth of scientific literature demands robust tools for automated survey-generation. However, current large language model (LLM)-based methods often lack in-depth analysis, structural coherence, and reliable citations. To address these limitations, we introduce SciSage, a multi-agent framework employing a reflect-when-you-write paradigm. SciSage features a hierarchical Reflector agent that critically evaluates drafts at outline, section, and document levels, collaborating with specialized agents for query interpretation, content retrieval, and refinement. We also release SurveyScope, a rigorously curated benchmark of 46 high-impact papers (2020-2025) across 11 computer science domains, with strict recency and citation-based quality controls. Evaluations demonstrate that SciSage outperforms state-of-the-art baselines (LLM x MapReduce-V2, AutoSurvey), achieving +1.73 points in document coherence and +32% in citation F1 scores. Human evaluations reveal mixed outcomes (3 wins vs. 7 losses against human-written surveys), but highlight SciSage's strengths in topical breadth and retrieval efficiency. Overall, SciSage offers a promising foundation for research-assistive writing tools. 

---
# DoTA-RAG: Dynamic of Thought Aggregation RAG 

**Authors**: Saksorn Ruangtanusak, Natthapath Rungseesiripak, Peerawat Rojratchadakorn, Monthol Charattrakool, Natapong Nitarach  

**Link**: [PDF](https://arxiv.org/pdf/2506.12571)  

**Abstract**: In this paper, we introduce DoTA-RAG (Dynamic-of-Thought Aggregation RAG), a retrieval-augmented generation system optimized for high-throughput, large-scale web knowledge indexes. Traditional RAG pipelines often suffer from high latency and limited accuracy over massive, diverse datasets. DoTA-RAG addresses these challenges with a three-stage pipeline: query rewriting, dynamic routing to specialized sub-indexes, and multi-stage retrieval and ranking. We further enhance retrieval by evaluating and selecting a superior embedding model, re-embedding the large FineWeb-10BT corpus. Moreover, we create a diverse Q&A dataset of 500 questions generated via the DataMorgana setup across a broad range of WebOrganizer topics and formats. DoTA-RAG improves the answer correctness score from 0.752 (baseline, using LiveRAG pre-built vector store) to 1.478 while maintaining low latency, and it achieves a 0.929 correctness score on the Live Challenge Day. These results highlight DoTA-RAG's potential for practical deployment in domains requiring fast, reliable access to large and evolving knowledge sources. 

---
# FlexRAG: A Flexible and Comprehensive Framework for Retrieval-Augmented Generation 

**Authors**: Zhuocheng Zhang, Yang Feng, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.12494)  

**Abstract**: Retrieval-Augmented Generation (RAG) plays a pivotal role in modern large language model applications, with numerous existing frameworks offering a wide range of functionalities to facilitate the development of RAG systems. However, we have identified several persistent challenges in these frameworks, including difficulties in algorithm reproduction and sharing, lack of new techniques, and high system overhead. To address these limitations, we introduce \textbf{FlexRAG}, an open-source framework specifically designed for research and prototyping. FlexRAG supports text-based, multimodal, and network-based RAG, providing comprehensive lifecycle support alongside efficient asynchronous processing and persistent caching capabilities. By offering a robust and flexible solution, FlexRAG enables researchers to rapidly develop, deploy, and share advanced RAG systems. Our toolkit and resources are available at \href{this https URL}{this https URL}. 

---
