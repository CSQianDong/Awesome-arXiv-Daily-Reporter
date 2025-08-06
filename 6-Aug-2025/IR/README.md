# Personalized Recommendation of Dish and Restaurant Collections on iFood 

**Authors**: Fernando F. Granado, Davi A. Bezerra, Iuri Queiroz, Nathan Oliveira, Pedro Fernandes, Bruno Schock  

**Link**: [PDF](https://arxiv.org/pdf/2508.03670)  

**Abstract**: Food delivery platforms face the challenge of helping users navigate vast catalogs of restaurants and dishes to find meals they truly enjoy. This paper presents RED, an automated recommendation system designed for iFood, Latin America's largest on-demand food delivery platform, to personalize the selection of curated food collections displayed to millions of users. Our approach employs a LightGBM classifier that scores collections based on three feature groups: collection characteristics, user-collection similarity, and contextual information. To address the cold-start problem of recommending newly created collections, we develop content-based representations using item embeddings and implement monotonicity constraints to improve generalization. We tackle data scarcity by bootstrapping from category carousel interactions and address visibility bias through unbiased sampling of impressions and purchases in production. The system demonstrates significant real-world impact through extensive A/B testing with 5-10% of iFood's user base. Online results of our A/B tests add up to 97% improvement in Card Conversion Rate and 1.4% increase in overall App Conversion Rate compared to popularity-based baselines. Notably, our offline accuracy metrics strongly correlate with online performance, enabling reliable impact prediction before deployment. To our knowledge, this is the first work to detail large-scale recommendation of curated food collections in a dynamic commercial environment. 

---
# LLMDistill4Ads: Using Cross-Encoders to Distill from LLM Signals for Advertiser Keyphrase Recommendations at eBay 

**Authors**: Soumik Dey, Benjamin Braun, Naveen Ravipati, Hansi Wu, Binbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.03628)  

**Abstract**: Sellers at eBay are recommended keyphrases to bid on to enhance the performance of their advertising campaigns. The relevance of these keyphrases is crucial in avoiding the overcrowding of search systems with irrelevant items and maintaining a positive seller perception. It is essential that keyphrase recommendations align with both seller and Search judgments regarding auctions. Due to the difficulty in procuring negative human judgment at scale, employing LLM-as-a-judge to mimic seller judgment has been established as the norm in several studies. This study introduces a novel two-step LLM distillation process from a LLM-judge used to debias our Embedding Based Retrieval (EBR) model from the various biases that exist in click-data. We distill from an LLM teacher via a cross-encoder assistant into a bi-encoder student using a multi-task training approach, ultimately employing the student bi-encoder to retrieve relevant advertiser keyphrases. We show that integrating a knowledge distillation process from LLMs in a multi-task training setup enhances bi-encoder performance in retrieving relevant advertiser keyphrases at eBay. 

---
# Demystifying Sequential Recommendations: Counterfactual Explanations via Genetic Algorithms 

**Authors**: Domiziano Scarcelli, Filippo Betello, Giuseppe Perelli, Fabrizio Silvestri, Gabriele Tolomei  

**Link**: [PDF](https://arxiv.org/pdf/2508.03606)  

**Abstract**: Sequential Recommender Systems (SRSs) have demonstrated remarkable effectiveness in capturing users' evolving preferences. However, their inherent complexity as "black box" models poses significant challenges for explainability. This work presents the first counterfactual explanation technique specifically developed for SRSs, introducing a novel approach in this space, addressing the key question: What minimal changes in a user's interaction history would lead to different recommendations? To achieve this, we introduce a specialized genetic algorithm tailored for discrete sequences and show that generating counterfactual explanations for sequential data is an NP-Complete problem. We evaluate these approaches across four experimental settings, varying between targeted-untargeted and categorized-uncategorized scenarios, to comprehensively assess their capability in generating meaningful explanations. Using three different datasets and three models, we are able to demonstrate that our methods successfully generate interpretable counterfactual explanation while maintaining model fidelity close to one. Our findings contribute to the growing field of Explainable AI by providing a framework for understanding sequential recommendation decisions through the lens of "what-if" scenarios, ultimately enhancing user trust and system transparency. 

---
# PyLate: Flexible Training and Retrieval for Late Interaction Models 

**Authors**: Antoine Chaffin, Raphaël Sourty  

**Link**: [PDF](https://arxiv.org/pdf/2508.03555)  

**Abstract**: Neural ranking has become a cornerstone of modern information retrieval. While single vector search remains the dominant paradigm, it suffers from the shortcoming of compressing all the information into a single vector. This compression leads to notable performance degradation in out-of-domain, long-context, and reasoning-intensive retrieval tasks. Multi-vector approaches pioneered by ColBERT aim to address these limitations by preserving individual token embeddings and computing similarity via the MaxSim operator. This architecture has demonstrated superior empirical advantages, including enhanced out-of-domain generalization, long-context handling, and performance in complex retrieval scenarios. Despite these compelling empirical results and clear theoretical advantages, the practical adoption and public availability of late interaction models remain low compared to their single-vector counterparts, primarily due to a lack of accessible and modular tools for training and experimenting with such models. To bridge this gap, we introduce PyLate, a streamlined library built on top of Sentence Transformers to support multi-vector architectures natively, inheriting its efficient training, advanced logging, and automated model card generation while requiring minimal code changes to code templates users are already familiar with. By offering multi-vector-specific features such as efficient indexes, PyLate aims to accelerate research and real-world application of late interaction models, thereby unlocking their full potential in modern IR systems. Finally, PyLate has already enabled the development of state-of-the-art models, including GTE-ModernColBERT and Reason-ModernColBERT, demonstrating its practical utility for both research and production environments. 

---
# MultiRAG: A Knowledge-guided Framework for Mitigating Hallucination in Multi-source Retrieval Augmented Generation 

**Authors**: Wenlong Wu, Haofen Wang, Bohan Li, Peixuan Huang, Xinzhe Zhao, Lei Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.03553)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as a promising solution to address hallucination issues in Large Language Models (LLMs). However, the integration of multiple retrieval sources, while potentially more informative, introduces new challenges that can paradoxically exacerbate hallucination problems. These challenges manifest primarily in two aspects: the sparse distribution of multi-source data that hinders the capture of logical relationships and the inherent inconsistencies among different sources that lead to information conflicts. To address these challenges, we propose MultiRAG, a novel framework designed to mitigate hallucination in multi-source retrieval-augmented generation through knowledge-guided approaches. Our framework introduces two key innovations: (1) a knowledge construction module that employs multi-source line graphs to efficiently aggregate logical relationships across different knowledge sources, effectively addressing the sparse data distribution issue; and (2) a sophisticated retrieval module that implements a multi-level confidence calculation mechanism, performing both graph-level and node-level assessments to identify and eliminate unreliable information nodes, thereby reducing hallucinations caused by inter-source inconsistencies. Extensive experiments on four multi-domain query datasets and two multi-hop QA datasets demonstrate that MultiRAG significantly enhances the reliability and efficiency of knowledge retrieval in complex multi-source scenarios. \textcolor{blue}{Our code is available in this https URL. 

---
# Parameter-Efficient Single Collaborative Branch for Recommendation 

**Authors**: Marta Moscati, Shah Nawaz, Markus Schedl  

**Link**: [PDF](https://arxiv.org/pdf/2508.03518)  

**Abstract**: Recommender Systems (RS) often rely on representations of users and items in a joint embedding space and on a similarity metric to compute relevance scores. In modern RS, the modules to obtain user and item representations consist of two distinct and separate neural networks (NN). In multimodal representation learning, weight sharing has been proven effective in reducing the distance between multiple modalities of a same item. Inspired by these approaches, we propose a novel RS that leverages weight sharing between the user and item NN modules used to obtain the latent representations in the shared embedding space. The proposed framework consists of a single Collaborative Branch for Recommendation (CoBraR). We evaluate CoBraR by means of quantitative experiments on e-commerce and movie recommendation. Our experiments show that by reducing the number of parameters and improving beyond-accuracy aspects without compromising accuracy, CoBraR has the potential to be applied and extended for real-world scenarios. 

---
# Reliable Evaluation Protocol for Low-Precision Retrieval 

**Authors**: Kisu Yang, Yoonna Jang, Hwanseok Jang, Kenneth Choi, Isabelle Augenstein, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2508.03306)  

**Abstract**: Lowering the numerical precision of model parameters and computations is widely adopted to improve the efficiency of retrieval systems. However, when computing relevance scores between the query and documents in low-precision, we observe spurious ties due to the reduced granularity. This introduces high variability in the results based on tie resolution, making the evaluation less reliable. To address this, we propose a more robust retrieval evaluation protocol designed to reduce score variation. It consists of: (1) High-Precision Scoring (HPS), which upcasts the final scoring step to higher precision to resolve tied candidates with minimal computational cost; and (2) Tie-aware Retrieval Metrics (TRM), which report expected scores, range, and bias to quantify order uncertainty of tied candidates. Our experiments test multiple models with three scoring functions on two retrieval datasets to demonstrate that HPS dramatically reduces tie-induced instability, and TRM accurately recovers expected metric values. This combination enables a more consistent and reliable evaluation system for lower-precision retrievals. 

---
# Dual-disentangle Framework for Diversified Sequential Recommendation 

**Authors**: Haoran Zhang, Jingtong Liu, Jiangzhou Deng, Junpeng Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.03172)  

**Abstract**: Sequential recommendation predicts user preferences over time and has achieved remarkable success. However, the growing length of user interaction sequences and the complex entanglement of evolving user interests and intentions introduce significant challenges to diversity. To address these, we propose a model-agnostic Dual-disentangle framework for Diversified Sequential Recommendation (DDSRec). The framework refines user interest and intention modeling by adopting disentangling perspectives in interaction modeling and representation learning, thereby balancing accuracy and diversity in sequential recommendations. Extensive experiments on multiple public datasets demonstrate the effectiveness and superiority of DDSRec in terms of accuracy and diversity for sequential recommendations. 

---
# ADSeeker: A Knowledge-Infused Framework for Anomaly Detection and Reasoning 

**Authors**: Kai Zhang, Zekai Zhang, Xihe Sun, Jingmeng Nie, Qinghui Chen, Han Hao, Jianyuan Guo, Jinglin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.03088)  

**Abstract**: Automatic vision inspection holds significant importance in industry inspection. While multimodal large language models (MLLMs) exhibit strong language understanding capabilities and hold promise for this task, their performance remains significantly inferior to that of human experts. In this context, we identify two key challenges: (i) insufficient integration of anomaly detection (AD) knowledge during pre-training, and (ii) the lack of technically precise and conte-aware language generation for anomaly reasoning. To address these issues, we propose ADSeeker, an anomaly task assistant designed to enhance inspection performance through knowledge-grounded reasoning. ADSeeker leverages a curated visual document knowledge base, SEEK-MVTec&VisA (SEEK-M&V), which we construct to address the limitations of existing resources that rely solely on unstructured text. SEEK-M&V includes semantic-rich descriptions and image-document pairs, enabling more comprehensive anomaly understanding. To effectively retrieve and utilize this knowledge, we introduce the Query Image-Knowledge Retrieval-Augmented Generation (Q2K RAG) framework. To further enhance the performance in zero-shot anomaly detection (ZSAD), ADSeeker leverages the Hierarchical Sparse Prompt mechanism and type-level features to efficiently extract anomaly patterns. Furthermore, to tackle the challenge of limited in industry anomaly detection (IAD) data, we introduce the largest-scale AD dataset, Multi-type Anomaly (MulA), encompassing 72 multi-scale defect types across 26 Categories. Extensive experiments show that our plug-and-play framework, ADSeeker, achieves state-of-the-art zero-shot performance on several benchmark datasets. 

---
# KBest: Efficient Vector Search on Kunpeng CPU 

**Authors**: Kaihao MA, Meiling Wang, Senkevich Oleg, Zijian LI, Daihao Xue, Dmitriy Malyshev, Yangming Lv, Shihai Xiao, Xiao Yan, Radionov Alexander, Weidi Zeng, Yuanzhan Gao, Zhiyu Zou, Yao xin, Liu Lin, Junhao Wu, Yiding Liu, Yaoyao Fu, Gongyi Wang, Gong Zhang, Fei Yi, Yingfan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.03016)  

**Abstract**: Vector search, which returns the vectors most similar to a given query vector from a large vector dataset, underlies many important applications such as search, recommendation, and LLMs. To be economic, vector search needs to be efficient to reduce the resources required by a given query workload. However, existing vector search libraries (e.g., Faiss and DiskANN) are optimized for x86 CPU architectures (i.e., Intel and AMD CPUs) while Huawei Kunpeng CPUs are based on the ARM architecture and competitive in compute power. In this paper, we present KBest as a vector search library tailored for the latest Kunpeng 920 CPUs. To be efficient, KBest incorporates extensive hardware-aware and algorithmic optimizations, which include single-instruction-multiple-data (SIMD) accelerated distance computation, data prefetch, index refinement, early termination, and vector quantization. Experiment results show that KBest outperforms SOTA vector search libraries running on x86 CPUs, and our optimizations can improve the query throughput by over 2x. Currently, KBest serves applications from both our internal business and external enterprise clients with tens of millions of queries on a daily basis. 

---
# SustainableQA: A Comprehensive Question Answering Dataset for Corporate Sustainability and EU Taxonomy Reporting 

**Authors**: Mohammed Ali, Abdelrahman Abdallah, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2508.03000)  

**Abstract**: The growing demand for corporate sustainability transparency, particularly under new regulations like the EU Taxonomy, necessitates precise data extraction from large, unstructured corporate reports. Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems, requires high-quality, domain-specific question-answering (QA) datasets to excel at particular domains. To address this, we introduce SustainableQA, a novel dataset and a scalable pipeline for generating a comprehensive QA datasets from corporate sustainability reports and annual reports. Our approach integrates semantic chunk classification, a hybrid span extraction pipeline combining fine-tuned Named Entity Recognition (NER), rule-based methods, and LLM-driven refinement, alongside a specialized table-to-paragraph transformation. With over 195,000 diverse factoid and non-factoid QA pairs, SustainableQA is an effective resource for developing and benchmarking advanced knowledge assistants capable of navigating complex sustainability compliance 

---
# LLM-based IR-system for Bank Supervisors 

**Authors**: Ilias Aarab  

**Link**: [PDF](https://arxiv.org/pdf/2508.02945)  

**Abstract**: Bank supervisors face the complex task of ensuring that new measures are consistently aligned with historical precedents. To address this challenge, we introduce a novel Information Retrieval (IR) System tailored to assist supervisors in drafting both consistent and effective measures. This system ingests findings from on-site investigations. It then retrieves the most relevant historical findings and their associated measures from a comprehensive database, providing a solid basis for supervisors to write well-informed measures for new findings. Utilizing a blend of lexical, semantic, and Capital Requirements Regulation (CRR) fuzzy set matching techniques, the IR system ensures the retrieval of findings that closely align with current cases. The performance of this system, particularly in scenarios with partially labeled data, is validated through a Monte Carlo methodology, showcasing its robustness and accuracy. Enhanced by a Transformer-based Denoising AutoEncoder for fine-tuning, the final model achieves a Mean Average Precision (MAP@100) of 0.83 and a Mean Reciprocal Rank (MRR@100) of 0.92. These scores surpass those of both standalone lexical models such as BM25 and semantic BERT-like models. 

---
# Realizing Scaling Laws in Recommender Systems: A Foundation-Expert Paradigm for Hyperscale Model Deployment 

**Authors**: Dai Li, Kevin Course, Wei Li, Hongwei Li, Jie Hua, Yiqi Chen, Zhao Zhu, Rui Jian, Xuan Cao, Bi Xue, Yu Shi, Jing Qian, Kai Ren, Matt Ma, Qunshu Zhang, Rui Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.02929)  

**Abstract**: While scaling laws promise significant performance gains for recommender systems, efficiently deploying hyperscale models remains a major unsolved challenge. In contrast to fields where FMs are already widely adopted such as natural language processing and computer vision, progress in recommender systems is hindered by unique challenges including the need to learn from online streaming data under shifting data distributions, the need to adapt to different recommendation surfaces with a wide diversity in their downstream tasks and their input distributions, and stringent latency and computational constraints. To bridge this gap, we propose to leverage the Foundation-Expert Paradigm: a framework designed for the development and deployment of hyperscale recommendation FMs. In our approach, a central FM is trained on lifelong, cross-surface, multi-modal user data to learn generalizable knowledge. This knowledge is then efficiently transferred to various lightweight, surface-specific ``expert" models via target-aware embeddings, allowing them to adapt to local data distributions and optimization goals with minimal overhead. To meet our training, inference and development needs, we built HyperCast, a production-grade infrastructure system that re-engineers training, serving, logging and iteration to power this decoupled paradigm. Our approach is now deployed at Meta serving tens of billions of user requests daily, demonstrating online metric improvements over our previous one-stage production system while improving developer velocity and maintaining infrastructure efficiency. To the best of our knowledge, this work represents the first successful deployment of a Foundation-Expert paradigm at this scale, offering a proven, compute-efficient, and developer-friendly blueprint to realize the promise of scaling laws in recommender systems. 

---
# Are We on the Right Way for Assessing Document Retrieval-Augmented Generation? 

**Authors**: Wenxuan Shen, Mingjia Wang, Yaochen Wang, Dongping Chen, Junjie Yang, Yao Wan, Weiwei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.03644)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems using Multimodal Large Language Models (MLLMs) show great promise for complex document understanding, yet their development is critically hampered by inadequate evaluation. Current benchmarks often focus on specific part of document RAG system and use synthetic data with incomplete ground truth and evidence labels, therefore failing to reflect real-world bottlenecks and challenges. To overcome these limitations, we introduce Double-Bench: a new large-scale, multilingual, and multimodal evaluation system that is able to produce fine-grained assessment to each component within document RAG systems. It comprises 3,276 documents (72,880 pages) and 5,168 single- and multi-hop queries across 6 languages and 4 document types with streamlined dynamic update support for potential data contamination issues. Queries are grounded in exhaustively scanned evidence pages and verified by human experts to ensure maximum quality and completeness. Our comprehensive experiments across 9 state-of-the-art embedding models, 4 MLLMs and 4 end-to-end document RAG frameworks demonstrate the gap between text and visual embedding models is narrowing, highlighting the need in building stronger document retrieval models. Our findings also reveal the over-confidence dilemma within current document RAG frameworks that tend to provide answer even without evidence support. We hope our fully open-source Double-Bench provide a rigorous foundation for future research in advanced document RAG systems. We plan to retrieve timely corpus and release new benchmarks on an annual basis. 

---
# OpenLifelogQA: An Open-Ended Multi-Modal Lifelog Question-Answering Dataset 

**Authors**: Quang-Linh Tran, Binh Nguyen, Gareth J. F. Jones, Cathal Gurrin  

**Link**: [PDF](https://arxiv.org/pdf/2508.03583)  

**Abstract**: Lifelogging refers to the process of passively collecting, storing, and analysing personal daily life data using wearable devices. This data can support applications in memory preservation and enhancement. For example, using an ask-and-answer strategy, question-answering (QA) on lifelog data opens an interactive and interesting way to explore memorable events and insights into daily life. However, research resources for QA on lifelog data are limited to small-sized or synthetic QA datasets. In this paper, we present a novel lifelog QA dataset called OpenLifelogQA, building upon an 18-month lifelog dataset. Our dataset focuses on an open-ended and practical QA with real-world application in daily lifelog usage. We construct 14,187 pairs of Q&A with diverse types and difficulty levels. A baseline experiment is reported for this dataset with competitive average performance of 89.7% BERT Score, 25.87% ROUGE-L and 3.9665 LLM Score from LLaVA-NeXT-Interleave 7B model. We release this Q&A dataset to the research community to support new research into lifelog technologies, such as enabling personal chat-based assistants for lifelog data to become a reality. 

---
# fact check AI at SemEval-2025 Task 7: Multilingual and Crosslingual Fact-checked Claim Retrieval 

**Authors**: Pranshu Rastogi  

**Link**: [PDF](https://arxiv.org/pdf/2508.03475)  

**Abstract**: SemEval-2025 Task 7: Multilingual and Crosslingual Fact-Checked Claim Retrieval is approached as a Learning-to-Rank task using a bi-encoder model fine-tuned from a pre-trained transformer optimized for sentence similarity. Training used both the source languages and their English translations for multilingual retrieval and only English translations for cross-lingual retrieval. Using lightweight models with fewer than 500M parameters and training on Kaggle T4 GPUs, the method achieved 92% Success@10 in multilingual and 80% Success@10 in 5th in crosslingual and 10th in multilingual tracks. 

---
# Taggus: An Automated Pipeline for the Extraction of Characters' Social Networks from Portuguese Fiction Literature 

**Authors**: Tiago G Canário, Catarina Duarte, Flávio L. Pinheiro, João L.M. Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2508.03358)  

**Abstract**: Automatically identifying characters and their interactions from fiction books is, arguably, a complex task that requires pipelines that leverage multiple Natural Language Processing (NLP) methods, such as Named Entity Recognition (NER) and Part-of-speech (POS) tagging. However, these methods are not optimized for the task that leads to the construction of Social Networks of Characters. Indeed, the currently available methods tend to underperform, especially in less-represented languages, due to a lack of manually annotated data for training. Here, we propose a pipeline, which we call Taggus, to extract social networks from literary fiction works in Portuguese. Our results show that compared to readily available State-of-the-Art tools -- off-the-shelf NER tools and Large Language Models (ChatGPT) -- the resulting pipeline, which uses POS tagging and a combination of heuristics, achieves satisfying results with an average F1-Score of $94.1\%$ in the task of identifying characters and solving for co-reference and $75.9\%$ in interaction detection. These represent, respectively, an increase of $50.7\%$ and $22.3\%$ on results achieved by the readily available State-of-the-Art tools. Further steps to improve results are outlined, such as solutions for detecting relationships between characters. Limitations on the size and scope of our testing samples are acknowledged. The Taggus pipeline is publicly available to encourage development in this field for the Portuguese language.2 

---
# Investigating the Cognitive Response of Brake Lights in Initiating Braking Action Using EEG 

**Authors**: Ramaswamy Palaniappan, Surej Mouli, Howard Bowman, Ian McLoughlin  

**Link**: [PDF](https://arxiv.org/pdf/2508.03274)  

**Abstract**: Half of all road accidents result from either lack of driver attention or from maintaining insufficient separation between vehicles. Collision from the rear, in particular, has been identified as the most common class of accident in the UK, and its influencing factors have been widely studied for many years. Rear-mounted stop lamps, illuminated when braking, are the primary mechanism to alert following drivers to the need to reduce speed or brake. This paper develops a novel brain response approach to measuring subject reaction to different brake light designs. A variety of off-the-shelf brake light assemblies are tested in a physical simulated driving environment to assess the cognitive reaction times of 22 subjects. Eight pairs of LED-based and two pairs of incandescent bulb-based brake light assemblies are used and electroencephalogram (EEG) data recorded. Channel Pz is utilised to extract the P3 component evoked during the decision making process that occurs in the brain when a participant decides to lift their foot from the accelerator and depress the brake. EEG analysis shows that both incandescent bulb-based lights are statistically slower to evoke cognitive responses than all tested LED-based lights. Between the LED designs, differences are evident, but not statistically significant, attributed to the significant amount of movement artifact in the EEG signal. 

---
# A Multi-Agent System for Complex Reasoning in Radiology Visual Question Answering 

**Authors**: Ziruo Yi, Jinyu Liu, Ting Xiao, Mark V. Albert  

**Link**: [PDF](https://arxiv.org/pdf/2508.02841)  

**Abstract**: Radiology visual question answering (RVQA) provides precise answers to questions about chest X-ray images, alleviating radiologists' workload. While recent methods based on multimodal large language models (MLLMs) and retrieval-augmented generation (RAG) have shown promising progress in RVQA, they still face challenges in factual accuracy, hallucinations, and cross-modal misalignment. We introduce a multi-agent system (MAS) designed to support complex reasoning in RVQA, with specialized agents for context understanding, multimodal reasoning, and answer validation. We evaluate our system on a challenging RVQA set curated via model disagreement filtering, comprising consistently hard cases across multiple MLLMs. Extensive experiments demonstrate the superiority and effectiveness of our system over strong MLLM baselines, with a case study illustrating its reliability and interpretability. This work highlights the potential of multi-agent approaches to support explainable and trustworthy clinical AI applications that require complex reasoning. 

---
# Defending Against Knowledge Poisoning Attacks During Retrieval-Augmented Generation 

**Authors**: Kennedy Edemacu, Vinay M. Shashidhar, Micheal Tuape, Dan Abudu, Beakcheol Jang, Jong Wook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.02835)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful approach to boost the capabilities of large language models (LLMs) by incorporating external, up-to-date knowledge sources. However, this introduces a potential vulnerability to knowledge poisoning attacks, where attackers can compromise the knowledge source to mislead the generation model. One such attack is the PoisonedRAG in which the injected adversarial texts steer the model to generate an attacker-chosen response to a target question. In this work, we propose novel defense methods, FilterRAG and ML-FilterRAG, to mitigate the PoisonedRAG attack. First, we propose a new property to uncover distinct properties to differentiate between adversarial and clean texts in the knowledge data source. Next, we employ this property to filter out adversarial texts from clean ones in the design of our proposed approaches. Evaluation of these methods using benchmark datasets demonstrate their effectiveness, with performances close to those of the original RAG systems. 

---
