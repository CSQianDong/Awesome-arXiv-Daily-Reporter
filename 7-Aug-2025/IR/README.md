# Query Attribute Modeling: Improving search relevance with Semantic Search and Meta Data Filtering 

**Authors**: Karthik Menon, Batool Arhamna Haider, Muhammad Arham, Kanwal Mehreen, Ram Mohan Rao Kadiyala, Hamza Farooq  

**Link**: [PDF](https://arxiv.org/pdf/2508.04683)  

**Abstract**: This study introduces Query Attribute Modeling (QAM), a hybrid framework that enhances search precision and relevance by decomposing open text queries into structured metadata tags and semantic elements. QAM addresses traditional search limitations by automatically extracting metadata filters from free-form text queries, reducing noise and enabling focused retrieval of relevant items.
Experimental evaluation using the Amazon Toys Reviews dataset (10,000 unique items with 40,000+ reviews and detailed product attributes) demonstrated QAM's superior performance, achieving a mean average precision at 5 (mAP@5) of 52.99\%. This represents significant improvement over conventional methods, including BM25 keyword search, encoder-based semantic similarity search, cross-encoder re-ranking, and hybrid search combining BM25 and semantic results via Reciprocal Rank Fusion (RRF). The results establish QAM as a robust solution for Enterprise Search applications, particularly in e-commerce systems. 

---
# HiD-VAE: Interpretable Generative Recommendation via Hierarchical and Disentangled Semantic IDs 

**Authors**: Dengzhao Fang, Jingtong Gao, Chengcheng Zhu, Yu Li, Xiangyu Zhao, Yi Chang  

**Link**: [PDF](https://arxiv.org/pdf/2508.04618)  

**Abstract**: Recommender systems are indispensable for helping users navigate the immense item catalogs of modern online platforms. Recently, generative recommendation has emerged as a promising paradigm, unifying the conventional retrieve-and-rank pipeline into an end-to-end model capable of dynamic generation. However, existing generative methods are fundamentally constrained by their unsupervised tokenization, which generates semantic IDs suffering from two critical flaws: (1) they are semantically flat and uninterpretable, lacking a coherent hierarchy, and (2) they are prone to representation entanglement (i.e., ``ID collisions''), which harms recommendation accuracy and diversity. To overcome these limitations, we propose HiD-VAE, a novel framework that learns hierarchically disentangled item representations through two core innovations. First, HiD-VAE pioneers a hierarchically-supervised quantization process that aligns discrete codes with multi-level item tags, yielding more uniform and disentangled IDs. Crucially, the trained codebooks can predict hierarchical tags, providing a traceable and interpretable semantic path for each recommendation. Second, to combat representation entanglement, HiD-VAE incorporates a novel uniqueness loss that directly penalizes latent space overlap. This mechanism not only resolves the critical ID collision problem but also promotes recommendation diversity by ensuring a more comprehensive utilization of the item representation space. These high-quality, disentangled IDs provide a powerful foundation for downstream generative models. Extensive experiments on three public benchmarks validate HiD-VAE's superior performance against state-of-the-art methods. The code is available at this https URL. 

---
# A Reproducible, Scalable Pipeline for Synthesizing Autoregressive Model Literature 

**Authors**: Faruk Alpay, Bugra Kilictas, Hamdi Alakkad  

**Link**: [PDF](https://arxiv.org/pdf/2508.04612)  

**Abstract**: The accelerating pace of research on autoregressive generative models has produced thousands of papers, making manual literature surveys and reproduction studies increasingly impractical. We present a fully open-source, reproducible pipeline that automatically retrieves candidate documents from public repositories, filters them for relevance, extracts metadata, hyper-parameters and reported results, clusters topics, produces retrieval-augmented summaries and generates containerised scripts for re-running selected experiments. Quantitative evaluation on 50 manually-annotated papers shows F1 scores above 0.85 for relevance classification, hyper-parameter extraction and citation identification. Experiments on corpora of up to 1000 papers demonstrate near-linear scalability with eight CPU workers. Three case studies -- AWD-LSTM on WikiText-2, Transformer-XL on WikiText-103 and an autoregressive music model on the Lakh MIDI dataset -- confirm that the extracted settings support faithful reproduction, achieving test perplexities within 1--3% of the original reports. 

---
# Do Recommender Systems Really Leverage Multimodal Content? A Comprehensive Analysis on Multimodal Representations for Recommendation 

**Authors**: Claudio Pomo, Matteo Attimonelli, Danilo Danese, Fedelucio Narducci, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2508.04571)  

**Abstract**: Multimodal Recommender Systems aim to improve recommendation accuracy by integrating heterogeneous content, such as images and textual metadata. While effective, it remains unclear whether their gains stem from true multimodal understanding or increased model complexity. This work investigates the role of multimodal item embeddings, emphasizing the semantic informativeness of the representations. Initial experiments reveal that embeddings from standard extractors (e.g., ResNet50, Sentence-Bert) enhance performance, but rely on modality-specific encoders and ad hoc fusion strategies that lack control over cross-modal alignment. To overcome these limitations, we leverage Large Vision-Language Models (LVLMs) to generate multimodal-by-design embeddings via structured prompts. This approach yields semantically aligned representations without requiring any fusion. Experiments across multiple settings show notable performance improvements. Furthermore, LVLMs embeddings offer a distinctive advantage: they can be decoded into structured textual descriptions, enabling direct assessment of their multimodal comprehension. When such descriptions are incorporated as side content into recommender systems, they improve recommendation performance, empirically validating the semantic depth and alignment encoded within LVLMs outputs. Our study highlights the importance of semantically rich representations and positions LVLMs as a compelling foundation for building robust and meaningful multimodal representations in recommendation tasks. 

---
# TRAIL: Joint Inference and Refinement of Knowledge Graphs with Large Language Models 

**Authors**: Xinkui Zhao, Haode Li, Yifan Zhang, Guanjie Cheng, Yueshen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.04474)  

**Abstract**: Recent advances in large language models (LLMs) have unlocked powerful reasoning and decision-making capabilities. However, their inherent dependence on static parametric memory fundamentally limits their adaptability, factual accuracy, and interpretability in knowledge-intensive scenarios. Knowledge graphs (KGs), as structured repositories of explicit relational knowledge, offer a promising approach for augmenting LLMs with external, interpretable memory. Nevertheless, most existing methods that combine LLMs with KGs treat reasoning and knowledge updating as separate processes, resulting in suboptimal utilization of new information and hindering real-time updates. In this work, we propose TRAIL: a novel, unified framework for Thinking, Reasoning, And Incremental Learning that couples joint inference and dynamic KG refinement with large language models. TRAIL enables LLM agents to iteratively explore, update, and refine knowledge graphs during the reasoning process, employing a confidence-driven mechanism for the generation, validation, and pruning of new facts. This plug-and-play architecture facilitates seamless integration with various LLMs, supporting continual adaptation without the need for retraining. Extensive experiments on multiple benchmarks demonstrate that TRAIL outperforms existing KG-augmented and retrieval-augmented LLM baselines by 3% to 13%. More importantly, these results represent a significant step toward developing adaptive, memory-augmented language models capable of continual learning and reliable, transparent reasoning. 

---
# Algorithm Selection for Recommender Systems via Meta-Learning on Algorithm Characteristics 

**Authors**: Jarne Mathi Decker, Joeran Beel  

**Link**: [PDF](https://arxiv.org/pdf/2508.04419)  

**Abstract**: The Algorithm Selection Problem for recommender systems-choosing the best algorithm for a given user or context-remains a significant challenge. Traditional meta-learning approaches often treat algorithms as categorical choices, ignoring their intrinsic properties. Recent work has shown that explicitly characterizing algorithms with features can improve model performance in other domains. Building on this, we propose a per-user meta-learning approach for recommender system selection that leverages both user meta-features and automatically extracted algorithm features from source code. Our preliminary results, averaged over six diverse datasets, show that augmenting a meta-learner with algorithm features improves its average NDCG@10 performance by 8.83% from 0.135 (user features only) to 0.147. This enhanced model outperforms the Single Best Algorithm baseline (0.131) and successfully closes 10.5% of the performance gap to a theoretical oracle selector. These findings show that even static source code metrics provide a valuable predictive signal, presenting a promising direction for building more robust and intelligent recommender systems. 

---
# Comparative Analysis of Novel NIRMAL Optimizer Against Adam and SGD with Momentum 

**Authors**: Nirmal Gaud, Surej Mouli, Preeti Katiyar, Vaduguru Venkata Ramya  

**Link**: [PDF](https://arxiv.org/pdf/2508.04293)  

**Abstract**: This study proposes NIRMAL (Novel Integrated Robust Multi-Adaptation Learning), a novel optimization algorithm that combines multiple strategies inspired by the movements of the chess piece. These strategies include gradient descent, momentum, stochastic perturbations, adaptive learning rates, and non-linear transformations. We carefully evaluated NIRMAL against two widely used and successful optimizers, Adam and SGD with Momentum, on four benchmark image classification datasets: MNIST, FashionMNIST, CIFAR-10, and CIFAR-100. The custom convolutional neural network (CNN) architecture is applied on each dataset. The experimental results show that NIRMAL achieves competitive performance, particularly on the more challenging CIFAR-100 dataset, where it achieved a test accuracy of 45.32\%and a weighted F1-score of 0.4328. This performance surpasses Adam (41.79\% accuracy, 0.3964 F1-score) and closely matches SGD with Momentum (46.97\% accuracy, 0.4531 F1-score). Also, NIRMAL exhibits robust convergence and strong generalization capabilities, especially on complex datasets, as evidenced by stable training results in loss and accuracy curves. These findings underscore NIRMAL's significant ability as a versatile and effective optimizer for various deep learning tasks. 

---
# I$^3$-MRec: Invariant Learning with Information Bottleneck for Incomplete Modality Recommendation 

**Authors**: Huilin Chen, Miaomiao Cai, Fan Liu, Zhiyong Cheng, Richang Hong, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.04247)  

**Abstract**: Multimodal recommender systems (MRS) improve recommendation performance by integrating diverse semantic information from multiple modalities. However, the assumption of the availability of all modalities rarely holds in practice due to missing images, incomplete descriptions, or inconsistent user content. These challenges significantly degrade the robustness and generalization capabilities of current models. To address these challenges, we introduce a novel method called \textbf{I$^3$-MRec}, which uses \textbf{I}nvariant learning with \textbf{I}nformation bottleneck principle for \textbf{I}ncomplete \textbf{M}odality \textbf{Rec}ommendation. To achieve robust performance in missing modality scenarios, I$^3$-MRec enforces two pivotal properties: (i) cross-modal preference invariance, which ensures consistent user preference modeling across varying modality environments, and (ii) compact yet effective modality representation, which filters out task-irrelevant modality information while maximally preserving essential features relevant to recommendation. By treating each modality as a distinct semantic environment, I$^3$-MRec employs invariant risk minimization (IRM) to learn modality-specific item representations. In parallel, a missing-aware fusion module grounded in the Information Bottleneck (IB) principle extracts compact and effective item embeddings by suppressing modality noise and preserving core user preference signals. Extensive experiments conducted on three real-world datasets demonstrate that I$^3$-MRec consistently outperforms existing state-of-the-art MRS methods across various modality-missing scenarios, highlighting its effectiveness and robustness in practical applications. The code and processed datasets are released at this https URL. 

---
# Discrete-event Tensor Factorization: Learning a Smooth Embedding for Continuous Domains 

**Authors**: Joey De Pauw, Bart Goethals  

**Link**: [PDF](https://arxiv.org/pdf/2508.04221)  

**Abstract**: Recommender systems learn from past user behavior to predict future user preferences. Intuitively, it has been established that the most recent interactions are more indicative of future preferences than older interactions. Many recommendation algorithms use this notion to either drop older interactions or to assign them a lower weight, so the model can focus on the more informative, recent information. However, very few approaches model the flow of time explicitly.
This paper analyzes how time can be encoded in factorization-style recommendation models. By including absolute time as a feature, our models can learn varying user preferences and changing item perception over time. In addition to simple binning approaches, we also propose a novel, fully continuous time encoding mechanism. Through the use of a polynomial fit inside the loss function, our models completely avoid the need for discretization, and they are able to capture the time dimension in arbitrary resolution.
We perform a comparative study on three real-world datasets that span multiple years, where long user histories are present, and items stay relevant for a longer time. Empirical results show that, by explicitly modeling time, our models are very effective at capturing temporal signals, such as varying item popularities over time. Despite this however, our experiments also indicate that a simple post-hoc popularity adjustment is often sufficient to achieve the best performance on the unseen test set. This teaches us that, for the recommendation task, predicting the future is more important than capturing past trends. As such, we argue that specialized mechanisms are needed for extrapolation to future data. 

---
# ViLLA-MMBench: A Unified Benchmark Suite for LLM-Augmented Multimodal Movie Recommendation 

**Authors**: Fatemeh Nazary, Ali Tourani, Yashar Deldjoo, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2508.04206)  

**Abstract**: Recommending long-form video content demands joint modeling of visual, audio, and textual modalities, yet most benchmarks address only raw features or narrow fusion. We present ViLLA-MMBench, a reproducible, extensible benchmark for LLM-augmented multimodal movie recommendation. Built on MovieLens and MMTF-14K, it aligns dense item embeddings from three modalities: audio (block-level, i-vector), visual (CNN, AVF), and text. Missing or sparse metadata is automatically enriched using state-of-the-art LLMs (e.g., OpenAI Ada), generating high-quality synopses for thousands of movies. All text (raw or augmented) is embedded with configurable encoders (Ada, LLaMA-2, Sentence-T5), producing multiple ready-to-use sets. The pipeline supports interchangeable early-, mid-, and late-fusion (concatenation, PCA, CCA, rank-aggregation) and multiple backbones (MF, VAECF, VBPR, AMR, VMF) for ablation. Experiments are fully declarative via a single YAML file. Evaluation spans accuracy (Recall, nDCG) and beyond-accuracy metrics: cold-start rate, coverage, novelty, diversity, fairness. Results show LLM-based augmentation and strong text embeddings boost cold-start and coverage, especially when fused with audio-visual features. Systematic benchmarking reveals universal versus backbone- or metric-specific combinations. Open-source code, embeddings, and configs enable reproducible, fair multimodal RS research and advance principled generative AI integration in large-scale recommendation. Code: this https URL 

---
# SSEmb: A Joint Structural and Semantic Embedding Framework for Mathematical Formula Retrieval 

**Authors**: Ruyin Li, Xiaoyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.04162)  

**Abstract**: Formula retrieval is an important topic in Mathematical Information Retrieval. We propose SSEmb, a novel embedding framework capable of capturing both structural and semantic features of mathematical formulas. Structurally, we employ Graph Contrastive Learning to encode formulas represented as Operator Graphs. To enhance structural diversity while preserving mathematical validity of these formula graphs, we introduce a novel graph data augmentation approach through a substitution strategy. Semantically, we utilize Sentence-BERT to encode the surrounding text of formulas. Finally, for each query and its candidates, structural and semantic similarities are calculated separately and then fused through a weighted scheme. In the ARQMath-3 formula retrieval task, SSEmb outperforms existing embedding-based methods by over 5 percentage points on P'@10 and nDCG'@10. Furthermore, SSEmb enhances the performance of all runs of other methods and achieves state-of-the-art results when combined with Approach0. 

---
# Bridging Search and Recommendation through Latent Cross Reasoning 

**Authors**: Teng Shi, Weicong Qin, Weijie Yu, Xiao Zhang, Ming He, Jianping Fan, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.04152)  

**Abstract**: Search and recommendation (S&R) are fundamental components of modern online platforms, yet effectively leveraging search behaviors to improve recommendation remains a challenging problem. User search histories often contain noisy or irrelevant signals that can even degrade recommendation performance, while existing approaches typically encode S&R histories either jointly or separately without explicitly identifying which search behaviors are truly useful. Inspired by the human decision-making process, where one first identifies recommendation intent and then reasons about relevant evidence, we design a latent cross reasoning framework that first encodes user S&R histories to capture global interests and then iteratively reasons over search behaviors to extract signals beneficial for recommendation. Contrastive learning is employed to align latent reasoning states with target items, and reinforcement learning is further introduced to directly optimize ranking performance. Extensive experiments on public benchmarks demonstrate consistent improvements over strong baselines, validating the importance of reasoning in enhancing search-aware recommendation. 

---
# Benefit from Rich: Tackling Search Interaction Sparsity in Search Enhanced Recommendation 

**Authors**: Teng Shi, Weijie Yu, Xiao Zhang, Ming He, Jianping Fan, Jun Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.04145)  

**Abstract**: In modern online platforms, search and recommendation (S&R) often coexist, offering opportunities for performance improvement through search-enhanced approaches. Existing studies show that incorporating search signals boosts recommendation performance. However, the effectiveness of these methods relies heavily on rich search interactions. They primarily benefit a small subset of users with abundant search behavior, while offering limited improvements for the majority of users who exhibit only sparse search activity. To address the problem of sparse search data in search-enhanced recommendation, we face two key challenges: (1) how to learn useful search features for users with sparse search interactions, and (2) how to design effective training objectives under sparse conditions. Our idea is to leverage the features of users with rich search interactions to enhance those of users with sparse search interactions. Based on this idea, we propose GSERec, a method that utilizes message passing on the User-Code Graphs to alleviate data sparsity in Search-Enhanced Recommendation. Specifically, we utilize Large Language Models (LLMs) with vector quantization to generate discrete codes, which connect similar users and thereby construct the graph. Through message passing on this graph, embeddings of users with rich search data are propagated to enhance the embeddings of users with sparse interactions. To further ensure that the message passing captures meaningful information from truly similar users, we introduce a contrastive loss to better model user similarities. The enhanced user representations are then integrated into downstream search-enhanced recommendation models. Experiments on three real-world datasets show that GSERec consistently outperforms baselines, especially for users with sparse search behaviors. 

---
# Enhancing Serendipity Recommendation System by Constructing Dynamic User Knowledge Graphs with Large Language Models 

**Authors**: Qian Yong, Yanhui Li, Jialiang Shi, Yaguang Dou, Tian Qi  

**Link**: [PDF](https://arxiv.org/pdf/2508.04032)  

**Abstract**: The feedback loop in industrial recommendation systems reinforces homogeneous content, creates filter bubble effects, and diminishes user satisfaction. Recently, large language models(LLMs) have demonstrated potential in serendipity recommendation, thanks to their extensive world knowledge and superior reasoning capabilities. However, these models still face challenges in ensuring the rationality of the reasoning process, the usefulness of the reasoning results, and meeting the latency requirements of industrial recommendation systems (RSs). To address these challenges, we propose a method that leverages llm to dynamically construct user knowledge graphs, thereby enhancing the serendipity of recommendation systems. This method comprises a two stage framework:(1) two-hop interest reasoning, where user static profiles and historical behaviors are utilized to dynamically construct user knowledge graphs via llm. Two-hop reasoning, which can enhance the quality and accuracy of LLM reasoning results, is then performed on the constructed graphs to identify users' potential interests; and(2) Near-line adaptation, a cost-effective approach to deploying the aforementioned models in industrial recommendation systems. We propose a u2i (user-to-item) retrieval model that also incorporates i2i (item-to-item) retrieval capabilities, the retrieved items not only exhibit strong relevance to users' newly emerged interests but also retain the high conversion rate of traditional u2i retrieval. Our online experiments on the Dewu app, which has tens of millions of users, indicate that the method increased the exposure novelty rate by 4.62%, the click novelty rate by 4.85%, the average view duration per person by 0.15%, unique visitor click through rate by 0.07%, and unique visitor interaction penetration by 0.30%, enhancing user experience. 

---
# ConvMix: A Mixed-Criteria Data Augmentation Framework for Conversational Dense Retrieval 

**Authors**: Fengran Mo, Jinghan Zhang, Yuchen Hui, Jia Ao Sun, Zhichao Xu, Zhan Su, Jian-Yun Nie  

**Link**: [PDF](https://arxiv.org/pdf/2508.04001)  

**Abstract**: Conversational search aims to satisfy users' complex information needs via multiple-turn interactions. The key challenge lies in revealing real users' search intent from the context-dependent queries. Previous studies achieve conversational search by fine-tuning a conversational dense retriever with relevance judgments between pairs of context-dependent queries and documents. However, this training paradigm encounters data scarcity issues. To this end, we propose ConvMix, a mixed-criteria framework to augment conversational dense retrieval, which covers more aspects than existing data augmentation frameworks. We design a two-sided relevance judgment augmentation schema in a scalable manner via the aid of large language models. Besides, we integrate the framework with quality control mechanisms to obtain semantically diverse samples and near-distribution supervisions to combine various annotated data. Experimental results on five widely used benchmarks show that the conversational dense retriever trained by our ConvMix framework outperforms previous baseline methods, which demonstrates our superior effectiveness. 

---
# Measuring the stability and plasticity of recommender systems 

**Authors**: Maria João Lavoura, Robert Jungnickel, João Vinagre  

**Link**: [PDF](https://arxiv.org/pdf/2508.03941)  

**Abstract**: The typical offline protocol to evaluate recommendation algorithms is to collect a dataset of user-item interactions and then use a part of this dataset to train a model, and the remaining data to measure how closely the model recommendations match the observed user interactions. This protocol is straightforward, useful and practical, but it only captures performance of a particular model trained at some point in the past. We know, however, that online systems evolve over time. In general, it is a good idea that models reflect such changes, so models are frequently retrained with recent data. But if this is the case, to what extent can we trust previous evaluations? How will a model perform when a different pattern (re)emerges? In this paper we propose a methodology to study how recommendation models behave when they are retrained. The idea is to profile algorithms according to their ability to, on the one hand, retain past patterns -- stability -- and, on the other hand, (quickly) adapt to changes -- plasticity. We devise an offline evaluation protocol that provides detail on the long-term behavior of models, and that is agnostic to datasets, algorithms and metrics. To illustrate the potential of this framework, we present preliminary results of three different types of algorithms on the GoodReads dataset that suggest different stability and plasticity profiles depending on the algorithmic technique, and a possible trade-off between stability and this http URL additional experiments will be necessary to confirm these observations, they already illustrate the usefulness of the proposed framework to gain insights on the long term dynamics of recommendation models. 

---
# A Social Data-Driven System for Identifying Estate-related Events and Topics 

**Authors**: Wenchuan Mu, Menglin Li, Kwan Hui Lim  

**Link**: [PDF](https://arxiv.org/pdf/2508.03711)  

**Abstract**: Social media platforms such as Twitter and Facebook have become deeply embedded in our everyday life, offering a dynamic stream of localized news and personal experiences. The ubiquity of these platforms position them as valuable resources for identifying estate-related issues, especially in the context of growing urban populations. In this work, we present a language model-based system for the detection and classification of estate-related events from social media content. Our system employs a hierarchical classification framework to first filter relevant posts and then categorize them into actionable estate-related topics. Additionally, for posts lacking explicit geotags, we apply a transformer-based geolocation module to infer posting locations at the point-of-interest level. This integrated approach supports timely, data-driven insights for urban management, operational response and situational awareness. 

---
# Evaluating Generative AI Tools for Personalized Offline Recommendations: A Comparative Study 

**Authors**: Rafael Salinas-Buestan, Otto Parra, Nelly Condori-Fernandez, Maria Fernanda Granda  

**Link**: [PDF](https://arxiv.org/pdf/2508.03710)  

**Abstract**: Background: Generative AI tools have become increasingly relevant in supporting personalized recommendations across various domains. However, their effectiveness in health-related behavioral interventions, especially those aiming to reduce the use of technology, remains underexplored. Aims: This study evaluates the performance and user satisfaction of the five most widely used generative AI tools when recommending non-digital activities tailored to individuals at risk of repetitive strain injury. Method: Following the Goal/Question/Metric (GQM) paradigm, this proposed experiment involves generative AI tools that suggest offline activities based on predefined user profiles and intervention scenarios. The evaluation is focused on quantitative performance (precision, recall, F1-score and MCC-score) and qualitative aspects (user satisfaction and perceived recommendation relevance). Two research questions were defined: RQ1 assessed which tool delivers the most accurate recommendations, and RQ2 evaluated how tool choice influences user satisfaction. 

---
# Privacy Risks of LLM-Empowered Recommender Systems: An Inversion Attack Perspective 

**Authors**: Yubo Wang, Min Tang, Nuo Shen, Shujie Cui, Weiqing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.03703)  

**Abstract**: The large language model (LLM) powered recommendation paradigm has been proposed to address the limitations of traditional recommender systems, which often struggle to handle cold start users or items with new IDs. Despite its effectiveness, this study uncovers that LLM empowered recommender systems are vulnerable to reconstruction attacks that can expose both system and user privacy. To examine this threat, we present the first systematic study on inversion attacks targeting LLM empowered recommender systems, where adversaries attempt to reconstruct original prompts that contain personal preferences, interaction histories, and demographic attributes by exploiting the output logits of recommendation models. We reproduce the vec2text framework and optimize it using our proposed method called Similarity Guided Refinement, enabling more accurate reconstruction of textual prompts from model generated logits. Extensive experiments across two domains (movies and books) and two representative LLM based recommendation models demonstrate that our method achieves high fidelity reconstructions. Specifically, we can recover nearly 65 percent of the user interacted items and correctly infer age and gender in 87 percent of the cases. The experiments also reveal that privacy leakage is largely insensitive to the victim model's performance but highly dependent on domain consistency and prompt complexity. These findings expose critical privacy vulnerabilities in LLM empowered recommender systems. 

---
# Suggest, Complement, Inspire: Story of Two Tower Recommendations at Allegro.com 

**Authors**: Aleksandra Osowska-Kurczab, Klaudia Nazarko, Mateusz Marzec, Lidia Wojciechowska, Eliška Kremeňová  

**Link**: [PDF](https://arxiv.org/pdf/2508.03702)  

**Abstract**: Building large-scale e-commerce recommendation systems requires addressing three key technical challenges: (1) designing a universal recommendation architecture across dozens of placements, (2) decreasing excessive maintenance costs, and (3) managing a highly dynamic product catalogue. This paper presents a unified content-based recommendation system deployed at this http URL, the largest e-commerce platform of European origin. The system is built on a prevalent Two Tower retrieval framework, representing products using textual and structured attributes, which enables efficient retrieval via Approximate Nearest Neighbour search. We demonstrate how the same model architecture can be adapted to serve three distinct recommendation tasks: similarity search, complementary product suggestions, and inspirational content discovery, by modifying only a handful of components in either the model or the serving logic. Extensive A/B testing over two years confirms significant gains in engagement and profit-based metrics across desktop and mobile app channels. Our results show that a flexible, scalable architecture can serve diverse user intents with minimal maintenance overhead. 

---
# Lightweight Transformers for Zero-Shot and Fine-Tuned Text-to-SQL Generation Using Spider 

**Authors**: Chirag Seth, Utkarsh Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.04623)  

**Abstract**: Text-to-SQL translation enables non-expert users to query relational databases using natural language, with applications in education and business intelligence. This study evaluates three lightweight transformer models - T5-Small, BART-Small, and GPT-2 - on the Spider dataset, focusing on low-resource settings. We developed a reusable, model-agnostic pipeline that tailors schema formatting to each model's architecture, training them across 1000 to 5000 iterations and evaluating on 1000 test samples using Logical Form Accuracy (LFAcc), BLEU, and Exact Match (EM) metrics. Fine-tuned T5-Small achieves the highest LFAcc (27.8%), outperforming BART-Small (23.98%) and GPT-2 (20.1%), highlighting encoder-decoder models' superiority in schema-aware SQL generation. Despite resource constraints limiting performance, our pipeline's modularity supports future enhancements, such as advanced schema linking or alternative base models. This work underscores the potential of compact transformers for accessible text-to-SQL solutions in resource-scarce environments. 

---
# TURA: Tool-Augmented Unified Retrieval Agent for AI Search 

**Authors**: Zhejun Zhao, Yuehu Dong, Alley Liu, Lixue Zheng, Pingsheng Liu, Dongdong Shen, Long Xia, Jiashu Zhao, Dawei Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.04604)  

**Abstract**: The advent of Large Language Models (LLMs) is transforming search engines into conversational AI search products, primarily using Retrieval-Augmented Generation (RAG) on web corpora. However, this paradigm has significant industrial limitations. Traditional RAG approaches struggle with real-time needs and structured queries that require accessing dynamically generated content like ticket availability or inventory. Limited to indexing static pages, search engines cannot perform the interactive queries needed for such time-sensitive data. Academic research has focused on optimizing RAG for static content, overlooking complex intents and the need for dynamic sources like databases and real-time APIs. To bridge this gap, we introduce TURA (Tool-Augmented Unified Retrieval Agent for AI Search), a novel three-stage framework that combines RAG with agentic tool-use to access both static content and dynamic, real-time information. TURA has three key components: an Intent-Aware Retrieval module to decompose queries and retrieve information sources encapsulated as Model Context Protocol (MCP) Servers, a DAG-based Task Planner that models task dependencies as a Directed Acyclic Graph (DAG) for optimal parallel execution, and a lightweight Distilled Agent Executor for efficient tool calling. TURA is the first architecture to systematically bridge the gap between static RAG and dynamic information sources for a world-class AI search product. Serving tens of millions of users, it leverages an agentic framework to deliver robust, real-time answers while meeting the low-latency demands of a large-scale industrial system. 

---
# Improving Crash Data Quality with Large Language Models: Evidence from Secondary Crash Narratives in Kentucky 

**Authors**: Xu Zhang, Mei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.04399)  

**Abstract**: This study evaluates advanced natural language processing (NLP) techniques to enhance crash data quality by mining crash narratives, using secondary crash identification in Kentucky as a case study. Drawing from 16,656 manually reviewed narratives from 2015-2022, with 3,803 confirmed secondary crashes, we compare three model classes: zero-shot open-source large language models (LLMs) (LLaMA3:70B, DeepSeek-R1:70B, Qwen3:32B, Gemma3:27B); fine-tuned transformers (BERT, DistilBERT, RoBERTa, XLNet, Longformer); and traditional logistic regression as baseline. Models were calibrated on 2015-2021 data and tested on 1,771 narratives from 2022. Fine-tuned transformers achieved superior performance, with RoBERTa yielding the highest F1-score (0.90) and accuracy (95%). Zero-shot LLaMA3:70B reached a comparable F1 of 0.86 but required 139 minutes of inference; the logistic baseline lagged well behind (F1:0.66). LLMs excelled in recall for some variants (e.g., GEMMA3:27B at 0.94) but incurred high computational costs (up to 723 minutes for DeepSeek-R1:70B), while fine-tuned models processed the test set in seconds after brief training. Further analysis indicated that mid-sized LLMs (e.g., DeepSeek-R1:32B) can rival larger counterparts in performance while reducing runtime, suggesting opportunities for optimized deployments. Results highlight trade-offs between accuracy, efficiency, and data requirements, with fine-tuned transformer models balancing precision and recall effectively on Kentucky data. Practical deployment considerations emphasize privacy-preserving local deployment, ensemble approaches for improved accuracy, and incremental processing for scalability, providing a replicable scheme for enhancing crash-data quality with advanced NLP. 

---
# Modelling and Classifying the Components of a Literature Review 

**Authors**: Francisco Bolaños, Angelo Salatino, Francesco Osborne, Enrico Motta  

**Link**: [PDF](https://arxiv.org/pdf/2508.04337)  

**Abstract**: Previous work has demonstrated that AI methods for analysing scientific literature benefit significantly from annotating sentences in papers according to their rhetorical roles, such as research gaps, results, limitations, extensions of existing methodologies, and others. Such representations also have the potential to support the development of a new generation of systems capable of producing high-quality literature reviews. However, achieving this goal requires the definition of a relevant annotation schema and effective strategies for large-scale annotation of the literature. This paper addresses these challenges by 1) introducing a novel annotation schema specifically designed to support literature review generation and 2) conducting a comprehensive evaluation of a wide range of state-of-the-art large language models (LLMs) in classifying rhetorical roles according to this schema. To this end, we also present Sci-Sentence, a novel multidisciplinary benchmark comprising 700 sentences manually annotated by domain experts and 2,240 sentences automatically labelled using LLMs. We evaluate 37 LLMs on this benchmark, spanning diverse model families and sizes, using both zero-shot learning and fine-tuning approaches. The experiments yield several novel insights that advance the state of the art in this challenging domain. First, the current generation of LLMs performs remarkably well on this task when fine-tuned on high-quality data, achieving performance levels above 96\% F1. Second, while large proprietary models like GPT-4o achieve the best results, some lightweight open-source alternatives also demonstrate excellent performance. Finally, enriching the training data with semi-synthetic examples generated by LLMs proves beneficial, enabling small encoders to achieve robust results and significantly enhancing the performance of several open decoder models. 

---
# A Hybrid AI Methodology for Generating Ontologies of Research Topics from Scientific Paper Corpora 

**Authors**: Alessia Pisu, Livio Pompianu, Francesco Osborne, Diego Reforgiato Recupero, Daniele Riboni, Angelo Salatino  

**Link**: [PDF](https://arxiv.org/pdf/2508.04213)  

**Abstract**: Taxonomies and ontologies of research topics (e.g., MeSH, UMLS, CSO, NLM) play a central role in providing the primary framework through which intelligent systems can explore and interpret the literature. However, these resources have traditionally been manually curated, a process that is time-consuming, prone to obsolescence, and limited in granularity. This paper presents Sci-OG, a semi-auto\-mated methodology for generating research topic ontologies, employing a multi-step approach: 1) Topic Discovery, extracting potential topics from research papers; 2) Relationship Classification, determining semantic relationships between topic pairs; and 3) Ontology Construction, refining and organizing topics into a structured ontology. The relationship classification component, which constitutes the core of the system, integrates an encoder-based language model with features describing topic occurrence in the scientific literature. We evaluate this approach against a range of alternative solutions using a dataset of 21,649 manually annotated semantic triples. Our method achieves the highest F1 score (0.951), surpassing various competing approaches, including a fine-tuned SciBERT model and several LLM baselines, such as the fine-tuned GPT4-mini. Our work is corroborated by a use case which illustrates the practical application of our system to extend the CSO ontology in the area of cybersecurity. The presented solution is designed to improve the accessibility, organization, and analysis of scientific knowledge, thereby supporting advancements in AI-enabled literature management and research exploration. 

---
# Dual Prompt Learning for Adapting Vision-Language Models to Downstream Image-Text Retrieval 

**Authors**: Yifan Wang, Tao Wang, Chenwei Tang, Caiyang Yu, Zhengqing Zang, Mengmi Zhang, Shudong Huang, Jiancheng Lv  

**Link**: [PDF](https://arxiv.org/pdf/2508.04028)  

**Abstract**: Recently, prompt learning has demonstrated remarkable success in adapting pre-trained Vision-Language Models (VLMs) to various downstream tasks such as image classification. However, its application to the downstream Image-Text Retrieval (ITR) task is more challenging. We find that the challenge lies in discriminating both fine-grained attributes and similar subcategories of the downstream data. To address this challenge, we propose Dual prompt Learning with Joint Category-Attribute Reweighting (DCAR), a novel dual-prompt learning framework to achieve precise image-text matching. The framework dynamically adjusts prompt vectors from both semantic and visual dimensions to improve the performance of CLIP on the downstream ITR task. Based on the prompt paradigm, DCAR jointly optimizes attribute and class features to enhance fine-grained representation learning. Specifically, (1) at the attribute level, it dynamically updates the weights of attribute descriptions based on text-image mutual information correlation; (2) at the category level, it introduces negative samples from multiple perspectives with category-matching weighting to learn subcategory distinctions. To validate our method, we construct the Fine-class Described Retrieval Dataset (FDRD), which serves as a challenging benchmark for ITR in downstream data domains. It covers over 1,500 downstream fine categories and 230,000 image-caption pairs with detailed attribute annotations. Extensive experiments on FDRD demonstrate that DCAR achieves state-of-the-art performance over existing baselines. 

---
# Prototype-Driven Structure Synergy Network for Remote Sensing Images Segmentation 

**Authors**: Junyi Wang, Jinjiang Li, Guodong Fan, Yakun Ju, Xiang Fang, Alex C. Kot  

**Link**: [PDF](https://arxiv.org/pdf/2508.04022)  

**Abstract**: In the semantic segmentation of remote sensing images, acquiring complete ground objects is critical for achieving precise analysis. However, this task is severely hindered by two major challenges: high intra-class variance and high inter-class similarity. Traditional methods often yield incomplete segmentation results due to their inability to effectively unify class representations and distinguish between similar features. Even emerging class-guided approaches are limited by coarse class prototype representations and a neglect of target structural information.
Therefore, this paper proposes a Prototype-Driven Structure Synergy Network (PDSSNet). The design of this network is based on a core concept, a complete ground object is jointly defined by its invariant class semantics and its variant spatial structure. To implement this, we have designed three key modules. First, the Adaptive Prototype Extraction Module (APEM) ensures semantic accuracy from the source by encoding the ground truth to extract unbiased class prototypes. Subsequently, the designed Semantic-Structure Coordination Module (SSCM) follows a hierarchical semantics-first, structure-second principle. This involves first establishing a global semantic cognition, then leveraging structural information to constrain and refine the semantic representation, thereby ensuring the integrity of class information. Finally, the Channel Similarity Adjustment Module (CSAM) employs a dynamic step-size adjustment mechanism to focus on discriminative features between classes.
Extensive experiments demonstrate that PDSSNet outperforms state-of-the-art methods. The source code is available at this https URL. 

---
# RAVID: Retrieval-Augmented Visual Detection: A Knowledge-Driven Approach for AI-Generated Image Identification 

**Authors**: Mamadou Keita, Wassim Hamidouche, Hessen Bougueffa Eutamene, Abdelmalik Taleb-Ahmed, Abdenour Hadid  

**Link**: [PDF](https://arxiv.org/pdf/2508.03967)  

**Abstract**: In this paper, we introduce RAVID, the first framework for AI-generated image detection that leverages visual retrieval-augmented generation (RAG). While RAG methods have shown promise in mitigating factual inaccuracies in foundation models, they have primarily focused on text, leaving visual knowledge underexplored. Meanwhile, existing detection methods, which struggle with generalization and robustness, often rely on low-level artifacts and model-specific features, limiting their adaptability. To address this, RAVID dynamically retrieves relevant images to enhance detection. Our approach utilizes a fine-tuned CLIP image encoder, RAVID CLIP, enhanced with category-related prompts to improve representation learning. We further integrate a vision-language model (VLM) to fuse retrieved images with the query, enriching the input and improving accuracy. Given a query image, RAVID generates an embedding using RAVID CLIP, retrieves the most relevant images from a database, and combines these with the query image to form an enriched input for a VLM (e.g., Qwen-VL or Openflamingo). Experiments on the UniversalFakeDetect benchmark, which covers 19 generative models, show that RAVID achieves state-of-the-art performance with an average accuracy of 93.85%. RAVID also outperforms traditional methods in terms of robustness, maintaining high accuracy even under image degradations such as Gaussian blur and JPEG compression. Specifically, RAVID achieves an average accuracy of 80.27% under degradation conditions, compared to 63.44% for the state-of-the-art model C2P-CLIP, demonstrating consistent improvements in both Gaussian blur and JPEG compression scenarios. The code will be publicly available upon acceptance. 

---
# Recommending With, Not For: Co-Designing Recommender Systems for Social Good 

**Authors**: Michael D. Ekstrand, Afsaneh Razi, Aleksandra Sarcevic, Maria Soledad Pera, Robin Burke, Katherine Landau Wright  

**Link**: [PDF](https://arxiv.org/pdf/2508.03792)  

**Abstract**: Recommender systems are usually designed by engineers, researchers, designers, and other members of development teams. These systems are then evaluated based on goals set by the aforementioned teams and other business units of the platforms operating the recommender systems. This design approach emphasizes the designers' vision for how the system can best serve the interests of users, providers, businesses, and other stakeholders. Although designers may be well-informed about user needs through user experience and market research, they are still the arbiters of the system's design and evaluation, with other stakeholders' interests less emphasized in user-centered design and evaluation. When extended to recommender systems for social good, this approach results in systems that reflect the social objectives as envisioned by the designers and evaluated as the designers understand them. Instead, social goals and operationalizations should be developed through participatory and democratic processes that are accountable to their stakeholders. We argue that recommender systems aimed at improving social good should be designed *by* and *with*, not just *for*, the people who will experience their benefits and harms. That is, they should be designed in collaboration with their users, creators, and other stakeholders as full co-designers, not only as user study participants. 

---
# A Robust and Efficient Pipeline for Enterprise-Level Large-Scale Entity Resolution 

**Authors**: Sandeepa Kannangara, Arman Abrahamyan, Daniel Elias, Thomas Kilby, Nadav Dar, Luiz Pizzato, Anna Leontjeva, Dan Jermyn  

**Link**: [PDF](https://arxiv.org/pdf/2508.03767)  

**Abstract**: Entity resolution (ER) remains a significant challenge in data management, especially when dealing with large datasets. This paper introduces MERAI (Massive Entity Resolution using AI), a robust and efficient pipeline designed to address record deduplication and linkage issues in high-volume datasets at an enterprise level. The pipeline's resilience and accuracy have been validated through various large-scale record deduplication and linkage projects. To evaluate MERAI's performance, we compared it with two well-known entity resolution libraries, Dedupe and Splink. While Dedupe failed to scale beyond 2 million records due to memory constraints, MERAI successfully processed datasets of up to 15.7 million records and produced accurate results across all experiments. Experimental data demonstrates that MERAI outperforms both baseline systems in terms of matching accuracy, with consistently higher F1 scores in both deduplication and record linkage tasks. MERAI offers a scalable and reliable solution for enterprise-level large-scale entity resolution, ensuring data integrity and consistency in real-world applications. 

---
