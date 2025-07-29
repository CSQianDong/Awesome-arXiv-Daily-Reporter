# Watermarking Large Language Model-based Time Series Forecasting 

**Authors**: Wei Yuan, Chaoqun Yang, Yu Xing, Tong Chen, Nguyen Quoc Viet Hung, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2507.20762)  

**Abstract**: Large Language Model-based Time Series Forecasting (LLMTS) has shown remarkable promise in handling complex and diverse temporal data, representing a significant step toward foundation models for time series analysis. However, this emerging paradigm introduces two critical challenges. First, the substantial commercial potential and resource-intensive development raise urgent concerns about intellectual property (IP) protection. Second, their powerful time series forecasting capabilities may be misused to produce misleading or fabricated deepfake time series data. To address these concerns, we explore watermarking the outputs of LLMTS models, that is, embedding imperceptible signals into the generated time series data that remain detectable by specialized algorithms. We propose a novel post-hoc watermarking framework, Waltz, which is broadly compatible with existing LLMTS models. Waltz is inspired by the empirical observation that time series patch embeddings are rarely aligned with a specific set of LLM tokens, which we term ``cold tokens''. Leveraging this insight, Waltz embeds watermarks by rewiring the similarity statistics between patch embeddings and cold token embeddings, and detects watermarks using similarity z-scores. To minimize potential side effects, we introduce a similarity-based embedding position identification strategy and employ projected gradient descent to constrain the watermark noise within a defined boundary. Extensive experiments using two popular LLMTS models across seven benchmark datasets demonstrate that Waltz achieves high watermark detection accuracy with minimal impact on the quality of the generated time series. 

---
# Industry Insights from Comparing Deep Learning and GBDT Models for E-Commerce Learning-to-Rank 

**Authors**: Yunus Lutz, Timo Wilm, Philipp Duwe  

**Link**: [PDF](https://arxiv.org/pdf/2507.20753)  

**Abstract**: In e-commerce recommender and search systems, tree-based models, such as LambdaMART, have set a strong baseline for Learning-to-Rank (LTR) tasks. Despite their effectiveness and widespread adoption in industry, the debate continues whether deep neural networks (DNNs) can outperform traditional tree-based models in this domain. To contribute to this discussion, we systematically benchmark DNNs against our production-grade LambdaMART model. We evaluate multiple DNN architectures and loss functions on a proprietary dataset from OTTO and validate our findings through an 8-week online A/B test. The results show that a simple DNN architecture outperforms a strong tree-based baseline in terms of total clicks and revenue, while achieving parity in total units sold. 

---
# Beyond Interactions: Node-Level Graph Generation for Knowledge-Free Augmentation in Recommender Systems 

**Authors**: Zhaoyan Wang, Hyunjun Ahn, In-Young Ko  

**Link**: [PDF](https://arxiv.org/pdf/2507.20578)  

**Abstract**: Recent advances in recommender systems rely on external resources such as knowledge graphs or large language models to enhance recommendations, which limit applicability in real-world settings due to data dependency and computational overhead. Although knowledge-free models are able to bolster recommendations by direct edge operations as well, the absence of augmentation primitives drives them to fall short in bridging semantic and structural gaps as high-quality paradigm substitutes. Unlike existing diffusion-based works that remodel user-item interactions, this work proposes NodeDiffRec, a pioneering knowledge-free augmentation framework that enables fine-grained node-level graph generation for recommendations and expands the scope of restricted augmentation primitives via diffusion. By synthesizing pseudo-items and corresponding interactions that align with the underlying distribution for injection, and further refining user preferences through a denoising preference modeling process, NodeDiffRec dramatically enhances both semantic diversity and structural connectivity without external knowledge. Extensive experiments across diverse datasets and recommendation algorithms demonstrate the superiority of NodeDiffRec, achieving State-of-the-Art (SOTA) performance, with maximum average performance improvement 98.6% in Recall@5 and 84.0% in NDCG@5 over selected baselines. 

---
# Improving Community Detection in Academic Networks by Handling Publication Bias 

**Authors**: Md Asaduzzaman Noor, John Sheppard, Jason Clark  

**Link**: [PDF](https://arxiv.org/pdf/2507.20449)  

**Abstract**: Finding potential research collaborators is a challenging task, especially in today's fast-growing and interdisciplinary research landscape. While traditional methods often rely on observable relationships such as co-authorships and citations to construct the research network, in this work, we focus solely on publication content to build a topic-based research network using BERTopic with a fine-tuned SciBERT model that connects and recommends researchers across disciplines based on shared topical interests. A major challenge we address is publication imbalance, where some researchers publish much more than others, often across several topics. Without careful handling, their less frequent interests are hidden under dominant topics, limiting the network's ability to detect their full research scope. To tackle this, we introduce a cloning strategy that clusters a researcher's publications and treats each cluster as a separate node. This allows researchers to be part of multiple communities, improving the detection of interdisciplinary links. Evaluation on the proposed method shows that the cloned network structure leads to more meaningful communities and uncovers a broader set of collaboration opportunities. 

---
# TADT-CSA: Temporal Advantage Decision Transformer with Contrastive State Abstraction for Generative Recommendation 

**Authors**: Xiang Gao, Tianyuan Liu, Yisha Li, Jingxin Liu, Lexi Gao, Xin Li, Haiyang Lu, Liyin Hong  

**Link**: [PDF](https://arxiv.org/pdf/2507.20327)  

**Abstract**: With the rapid advancement of Transformer-based Large Language Models (LLMs), generative recommendation has shown great potential in enhancing both the accuracy and semantic understanding of modern recommender systems. Compared to LLMs, the Decision Transformer (DT) is a lightweight generative model applied to sequential recommendation tasks. However, DT faces challenges in trajectory stitching, often producing suboptimal trajectories. Moreover, due to the high dimensionality of user states and the vast state space inherent in recommendation scenarios, DT can incur significant computational costs and struggle to learn effective state representations. To overcome these issues, we propose a novel Temporal Advantage Decision Transformer with Contrastive State Abstraction (TADT-CSA) model. Specifically, we combine the conventional Return-To-Go (RTG) signal with a novel temporal advantage (TA) signal that encourages the model to capture both long-term returns and their sequential trend. Furthermore, we integrate a contrastive state abstraction module into the DT framework to learn more effective and expressive state representations. Within this module, we introduce a TA-conditioned State Vector Quantization (TAC-SVQ) strategy, where the TA score guides the state codebooks to incorporate contextual token information. Additionally, a reward prediction network and a contrastive transition prediction (CTP) network are employed to ensure the state codebook preserves both the reward information of the current state and the transition information between adjacent states. Empirical results on both public datasets and an online recommendation system demonstrate the effectiveness of the TADT-CSA model and its superiority over baseline methods. 

---
# CTR-Driven Ad Text Generation via Online Feedback Preference Optimization 

**Authors**: Yanda Chen, Zihui Ren, Qixiang Gao, Jiale Chen, Si Chen, Xubin Li, Tiezheng Ge, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2507.20227)  

**Abstract**: Advertising text plays a critical role in determining click-through rates (CTR) in online advertising. Large Language Models (LLMs) offer significant efficiency advantages over manual ad text creation. However, LLM-generated ad texts do not guarantee higher CTR performance compared to human-crafted texts, revealing a gap between generation quality and online performance of ad texts. In this work, we propose a novel ad text generation method which optimizes for CTR through preference optimization from online feedback. Our approach adopts an innovative two-stage framework: (1) diverse ad text sampling via one-shot in-context learning, using retrieval-augmented generation (RAG) to provide exemplars with chain-of-thought (CoT) reasoning; (2) CTR-driven preference optimization from online feedback, which weighs preference pairs according to their CTR gains and confidence levels. Through our method, the resulting model enables end-to-end generation of high-CTR ad texts. Extensive experiments have demonstrated the effectiveness of our method in both offline and online metrics. Notably, we have applied our method on a large-scale online shopping platform and achieved significant CTR improvements, showcasing its strong applicability and effectiveness in advertising systems. 

---
# Practical Multi-Task Learning for Rare Conversions in Ad Tech 

**Authors**: Yuval Dishi, Ophir Friedler, Yonatan Karni, Natalia Silberstein, Yulia Stolin  

**Link**: [PDF](https://arxiv.org/pdf/2507.20161)  

**Abstract**: We present a Multi-Task Learning (MTL) approach for improving predictions for rare (e.g., <1%) conversion events in online advertising. The conversions are classified into "rare" or "frequent" types based on historical statistics. The model learns shared representations across all signals while specializing through separate task towers for each type. The approach was tested and fully deployed to production, demonstrating consistent improvements in both offline (0.69% AUC lift) and online KPI performance metric (2% Cost per Action reduction). 

---
# Integrating LLM-Derived Multi-Semantic Intent into Graph Model for Session-based Recommendation 

**Authors**: Shuo Zhang, Xiao Li, Jiayi Wu, Fan Yang, Xiang Li, Ming Gao  

**Link**: [PDF](https://arxiv.org/pdf/2507.20147)  

**Abstract**: Session-based recommendation (SBR) is mainly based on anonymous user interaction sequences to recommend the items that the next user is most likely to click. Currently, the most popular and high-performing SBR methods primarily leverage graph neural networks (GNNs), which model session sequences as graph-structured data to effectively capture user intent. However, most GNNs-based SBR methods primarily focus on modeling the ID sequence information of session sequences, while neglecting the rich semantic information embedded within them. This limitation significantly hampers model's ability to accurately infer users' true intention. To address above challenge, this paper proposes a novel SBR approach called Integrating LLM-Derived Multi-Semantic Intent into Graph Model for Session-based Recommendation (LLM-DMsRec). The method utilizes a pre-trained GNN model to select the top-k items as candidate item sets and designs prompts along with a large language model (LLM) to infer multi-semantic intents from these candidate items. Specifically, we propose an alignment mechanism that effectively integrates the semantic intent inferred by the LLM with the structural intent captured by GNNs. Extensive experiments conducted on the Beauty and ML-1M datasets demonstrate that the proposed method can be seamlessly integrated into GNNs framework, significantly enhancing its recommendation performance. 

---
# A Non-Parametric Choice Model That Learns How Users Choose Between Recommended Options 

**Authors**: Thorsten Krause, Harrie Oosterhuis  

**Link**: [PDF](https://arxiv.org/pdf/2507.20035)  

**Abstract**: Choice models predict which items users choose from presented options. In recommendation settings, they can infer user preferences while countering exposure bias. In contrast with traditional univariate recommendation models, choice models consider which competitors appeared with the chosen item. This ability allows them to distinguish whether a user chose an item due to preference, i.e., they liked it; or competition, i.e., it was the best available option. Each choice model assumes specific user behavior, e.g., the multinomial logit model. However, it is currently unclear how accurately these assumptions capture actual user behavior, how wrong assumptions impact inference, and whether better models exist.
In this work, we propose the learned choice model for recommendation (LCM4Rec), a non-parametric method for estimating the choice model. By applying kernel density estimation, LCM4Rec infers the most likely error distribution that describes the effect of inter-item cannibalization and thereby characterizes the users' choice model. Thus, it simultaneously infers what users prefer and how they make choices. Our experimental results indicate that our method (i) can accurately recover the choice model underlying a dataset; (ii) provides robust user preference inference, in contrast with existing choice models that are only effective when their assumptions match user behavior; and (iii) is more resistant against exposure bias than existing choice models. Thereby, we show that learning choice models, instead of assuming them, can produce more robust predictions. We believe this work provides an important step towards better understanding users' choice behavior. 

---
# Improving the Performance of Sequential Recommendation Systems with an Extended Large Language Model 

**Authors**: Sinnyum Choi, Woong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.19990)  

**Abstract**: Recently, competition in the field of artificial intelligence (AI) has intensified among major technological companies, resulting in the continuous release of new large-language models (LLMs) that exhibit improved language understanding and context-based reasoning capabilities. It is expected that these advances will enable more efficient personalized recommendations in LLM-based recommendation systems through improved quality of training data and architectural design. However, many studies have not considered these recent developments. In this study, it was proposed to improve LLM-based recommendation systems by replacing Llama2 with Llama3 in the LlamaRec framework. To ensure a fair comparison, random seed values were set and identical input data was provided during preprocessing and training. The experimental results show average performance improvements of 38.65\%, 8.69\%, and 8.19\% for the ML-100K, Beauty, and Games datasets, respectively, thus confirming the practicality of this method. Notably, the significant improvements achieved by model replacement indicate that the recommendation quality can be improved cost-effectively without the need to make structural changes to the system. Based on these results, it is our contention that the proposed approach is a viable solution for improving the performance of current recommendation systems. 

---
# Analyzing and Mitigating Repetitions in Trip Recommendation 

**Authors**: Wenzheng Shu, Kangqi Xu, Wenxin Tai, Ting Zhong, Yong Wang, Fan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.19798)  

**Abstract**: Trip recommendation has emerged as a highly sought-after service over the past decade. Although current studies significantly understand human intention consistency, they struggle with undesired repetitive outcomes that need resolution. We make two pivotal discoveries using statistical analyses and experimental designs: (1) The occurrence of repetitions is intricately linked to the models and decoding strategies. (2) During training and decoding, adding perturbations to logits can reduce repetition. Motivated by these observations, we introduce AR-Trip (Anti Repetition for Trip Recommendation), which incorporates a cycle-aware predictor comprising three mechanisms to avoid duplicate Points-of-Interest (POIs) and demonstrates their effectiveness in alleviating repetition. Experiments on four public datasets illustrate that AR-Trip successfully mitigates repetition issues while enhancing precision. 

---
# A Unified Framework for Interactive Visual Graph Matching via Attribute-Structure Synchronization 

**Authors**: Yuhua Liu, Haoxuan Wang, Jiajia Kou, Ling Sun, Heyu Wang, Yongheng Wang, Yigang Wang, Jinchang Lic, Zhiguang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2507.19750)  

**Abstract**: In traditional graph retrieval tools, graph matching is commonly used to retrieve desired graphs from extensive graph datasets according to their structural similarities. However, in real applications, graph nodes have numerous attributes which also contain valuable information for evaluating similarities between graphs. Thus, to achieve superior graph matching results, it is crucial for graph retrieval tools to make full use of the attribute information in addition to structural information. We propose a novel framework for interactive visual graph matching. In the proposed framework, an attribute-structure synchronization method is developed for representing structural and attribute features in a unified embedding space based on Canonical Correlation Analysis (CCA). To support fast and interactive matching, \revise{our method} provides users with intuitive visual query interfaces for traversing, filtering and searching for the target graph in the embedding space conveniently. With the designed interfaces, the users can also specify a new target graph with desired structural and semantic features. Besides, evaluation views are designed for easy validation and interpretation of the matching results. Case studies and quantitative comparisons on real-world datasets have demonstrated the superiorities of our proposed framework in graph matching and large graph exploration. 

---
# Modeling User Behavior from Adaptive Surveys with Supplemental Context 

**Authors**: Aman Shukla, Daniel Patrick Scantlebury, Rishabh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2507.20919)  

**Abstract**: Modeling user behavior is critical across many industries where understanding preferences, intent, or decisions informs personalization, targeting, and strategic outcomes. Surveys have long served as a classical mechanism for collecting such behavioral data due to their interpretability, structure, and ease of deployment. However, surveys alone are inherently limited by user fatigue, incomplete responses, and practical constraints on their length making them insufficient for capturing user behavior. In this work, we present LANTERN (Late-Attentive Network for Enriched Response Modeling), a modular architecture for modeling user behavior by fusing adaptive survey responses with supplemental contextual signals. We demonstrate the architectural value of maintaining survey primacy through selective gating, residual connections and late fusion via cross-attention, treating survey data as the primary signal while incorporating external modalities only when relevant. LANTERN outperforms strong survey-only baselines in multi-label prediction of survey responses. We further investigate threshold sensitivity and the benefits of selective modality reliance through ablation and rare/frequent attribute analysis. LANTERN's modularity supports scalable integration of new encoders and evolving datasets. This work provides a practical and extensible blueprint for behavior modeling in survey-centric applications. 

---
# ZSE-Cap: A Zero-Shot Ensemble for Image Retrieval and Prompt-Guided Captioning 

**Authors**: Duc-Tai Dinh, Duc Anh Khoa Dinh  

**Link**: [PDF](https://arxiv.org/pdf/2507.20564)  

**Abstract**: We present ZSE-Cap (Zero-Shot Ensemble for Captioning), our 4th place system in Event-Enriched Image Analysis (EVENTA) shared task on article-grounded image retrieval and captioning. Our zero-shot approach requires no finetuning on the competition's data. For retrieval, we ensemble similarity scores from CLIP, SigLIP, and DINOv2. For captioning, we leverage a carefully engineered prompt to guide the Gemma 3 model, enabling it to link high-level events from the article to the visual content in the image. Our system achieved a final score of 0.42002, securing a top-4 position on the private test set, demonstrating the effectiveness of combining foundation models through ensembling and prompting. Our code is available at this https URL. 

---
# TIMEST: Temporal Information Motif Estimator Using Sampling Trees 

**Authors**: Yunjie Pan, Omkar Bhalerao, C. Seshadhri, Nishil Talati  

**Link**: [PDF](https://arxiv.org/pdf/2507.20441)  

**Abstract**: The mining of pattern subgraphs, known as motifs, is a core task in the field of graph mining. Edges in real-world networks often have timestamps, so there is a need for temporal motif mining. A temporal motif is a richer structure that imposes timing constraints on the edges of the motif. Temporal motifs have been used to analyze social networks, financial transactions, and biological networks.
Motif counting in temporal graphs is particularly challenging. A graph with millions of edges can have trillions of temporal motifs, since the same edge can occur with multiple timestamps. There is a combinatorial explosion of possibilities, and state-of-the-art algorithms cannot manage motifs with more than four vertices.
In this work, we present TIMEST: a general, fast, and accurate estimation algorithm to count temporal motifs of arbitrary sizes in temporal networks. Our approach introduces a temporal spanning tree sampler that leverages weighted sampling to generate substructures of target temporal motifs. This method carefully takes a subset of temporal constraints of the motif that can be jointly and efficiently sampled. TIMEST uses randomized estimation techniques to obtain accurate estimates of motif counts.
We give theoretical guarantees on the running time and approximation guarantees of TIMEST. We perform an extensive experimental evaluation and show that TIMEST is both faster and more accurate than previous algorithms. Our CPU implementation exhibits an average speedup of 28x over state-of-the-art GPU implementation of the exact algorithm, and 6x speedup over SOTA approximate algorithms while consistently showcasing less than 5% error in most cases. For example, TIMEST can count the number of instances of a financial fraud temporal motif in four minutes with 0.6% error, while exact methods take more than two days. 

---
# Multi-Stage Verification-Centric Framework for Mitigating Hallucination in Multi-Modal RAG 

**Authors**: Baiyu Chen, Wilson Wongso, Xiaoqian Hu, Yue Tan, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2507.20136)  

**Abstract**: This paper presents the technical solution developed by team CRUISE for the KDD Cup 2025 Meta Comprehensive RAG Benchmark for Multi-modal, Multi-turn (CRAG-MM) challenge. The challenge aims to address a critical limitation of modern Vision Language Models (VLMs): their propensity to hallucinate, especially when faced with egocentric imagery, long-tail entities, and complex, multi-hop questions. This issue is particularly problematic in real-world applications where users pose fact-seeking queries that demand high factual accuracy across diverse modalities. To tackle this, we propose a robust, multi-stage framework that prioritizes factual accuracy and truthfulness over completeness. Our solution integrates a lightweight query router for efficiency, a query-aware retrieval and summarization pipeline, a dual-pathways generation and a post-hoc verification. This conservative strategy is designed to minimize hallucinations, which incur a severe penalty in the competition's scoring metric. Our approach achieved 3rd place in Task 1, demonstrating the effectiveness of prioritizing answer reliability in complex multi-modal RAG systems. Our implementation is available at this https URL . 

---
# Leveraging Fine-Tuned Large Language Models for Interpretable Pancreatic Cystic Lesion Feature Extraction and Risk Categorization 

**Authors**: Ebrahim Rasromani, Stella K. Kang, Yanqi Xu, Beisong Liu, Garvit Luhadia, Wan Fung Chui, Felicia L. Pasadyn, Yu Chih Hung, Julie Y. An, Edwin Mathieu, Zehui Gu, Carlos Fernandez-Granda, Ammar A. Javed, Greg D. Sacks, Tamas Gonda, Chenchan Huang, Yiqiu Shen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19973)  

**Abstract**: Background: Manual extraction of pancreatic cystic lesion (PCL) features from radiology reports is labor-intensive, limiting large-scale studies needed to advance PCL research. Purpose: To develop and evaluate large language models (LLMs) that automatically extract PCL features from MRI/CT reports and assign risk categories based on guidelines. Materials and Methods: We curated a training dataset of 6,000 abdominal MRI/CT reports (2005-2024) from 5,134 patients that described PCLs. Labels were generated by GPT-4o using chain-of-thought (CoT) prompting to extract PCL and main pancreatic duct features. Two open-source LLMs were fine-tuned using QLoRA on GPT-4o-generated CoT data. Features were mapped to risk categories per institutional guideline based on the 2017 ACR White Paper. Evaluation was performed on 285 held-out human-annotated reports. Model outputs for 100 cases were independently reviewed by three radiologists. Feature extraction was evaluated using exact match accuracy, risk categorization with macro-averaged F1 score, and radiologist-model agreement with Fleiss' Kappa. Results: CoT fine-tuning improved feature extraction accuracy for LLaMA (80% to 97%) and DeepSeek (79% to 98%), matching GPT-4o (97%). Risk categorization F1 scores also improved (LLaMA: 0.95; DeepSeek: 0.94), closely matching GPT-4o (0.97), with no statistically significant differences. Radiologist inter-reader agreement was high (Fleiss' Kappa = 0.888) and showed no statistically significant difference with the addition of DeepSeek-FT-CoT (Fleiss' Kappa = 0.893) or GPT-CoT (Fleiss' Kappa = 0.897), indicating that both models achieved agreement levels on par with radiologists. Conclusion: Fine-tuned open-source LLMs with CoT supervision enable accurate, interpretable, and efficient phenotyping for large-scale PCL research, achieving performance comparable to GPT-4o. 

---
# A Scalable and High Availability Solution for Recommending Resolutions to Problem Tickets 

**Authors**: Harish S, Chetana K Nayak, Joy Bose  

**Link**: [PDF](https://arxiv.org/pdf/2507.19846)  

**Abstract**: Resolution of incidents or problem tickets is a common theme in service industries in any sector, including billing and charging systems in telecom domain. Machine learning can help to identify patterns and suggest resolutions for the problem tickets, based on patterns in the historical data of the tickets. However, this process may be complicated due to a variety of phenomena such as data drift and issues such as missing data, lack of data pertaining to resolutions of past incidents, too many similar sounding resolutions due to free text and similar sounding text. This paper proposes a robust ML-driven solution employing clustering, supervised learning, and advanced NLP models to tackle these challenges effectively. Building on previous work, we demonstrate clustering-based resolution identification, supervised classification with LDA, Siamese networks, and One-shot learning, Index embedding. Additionally, we present a real-time dashboard and a highly available Kubernetes-based production deployment. Our experiments with both the open-source Bitext customer-support dataset and proprietary telecom datasets demonstrate high prediction accuracy. 

---
# CleANN: Efficient Full Dynamism in Graph-based Approximate Nearest Neighbor Search 

**Authors**: Ziyu Zhang, Yuanhao Wei, Joshua Engels, Julian Shun  

**Link**: [PDF](https://arxiv.org/pdf/2507.19802)  

**Abstract**: Approximate nearest neighbor search (ANNS) has become a quintessential algorithmic problem for various other foundational data tasks for AI workloads. Graph-based ANNS indexes have superb empirical trade-offs in indexing cost, query efficiency, and query approximation quality. Most existing graph-based indexes are designed for the static scenario, where there are no updates to the data after the index is constructed. However, full dynamism (insertions, deletions, and searches) is crucial to providing up-to-date responses in applications using vector databases. It is desirable that the index efficiently supports updates and search queries concurrently. Existing dynamic graph-based indexes suffer from at least one of the following problems: (1) the query quality degrades as updates happen; and (2) the graph structure updates used to maintain the index quality upon updates are global and thus expensive. To solve these problems, we propose the CleANN system which consists of three main components: (1) workload-aware linking of diverse search tree descendants to combat distribution shift; (2)query-adaptive on-the-fly neighborhood consolidation to efficiently handle deleted nodes; and (3) semi-lazy memory cleaning to clean up stale information in the data structure and reduce the work spent by the first two components. We evaluate CleANN on 7 diverse datasets on fully dynamic workloads and find that CleANN has query quality at least as good as if the index had been built statically using the corresponding data. In the in-memory setting using 56 hyper-threads, with all types of queries running concurrently, at the same recall level, CleANN achieves 7-1200x throughput improvement on million-scale real-world datasets. To the best of our knowledge, CleANN is the first concurrent ANNS index to achieve such efficiency while maintaining quality under full dynamism. 

---
# Unlimited Editions: Documenting Human Style in AI Art Generation 

**Authors**: Alex Leitch, Celia Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.19497)  

**Abstract**: As AI art generation becomes increasingly sophisticated, HCI research has focused primarily on questions of detection, authenticity, and automation. This paper argues that such approaches fundamentally misunderstand how artistic value emerges from the concerns that drive human image production. Through examination of historical precedents, we demonstrate that artistic style is not only visual appearance but the resolution of creative struggle, as artists wrestle with influence and technical constraints to develop unique ways of seeing. Current AI systems flatten these human choices into reproducible patterns without preserving their provenance. We propose that HCI's role lies not only in perfecting visual output, but in developing means to document the origins and evolution of artistic style as it appears within generated visual traces. This reframing suggests new technical directions for HCI research in generative AI, focused on automatic documentation of stylistic lineage and creative choice rather than simple reproduction of aesthetic effects. 

---
