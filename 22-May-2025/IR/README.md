# Distance Adaptive Beam Search for Provably Accurate Graph-Based Nearest Neighbor Search 

**Authors**: Yousef Al-Jazzazi, Haya Diwan, Jinrui Gou, Cameron Musco, Christopher Musco, Torsten Suel  

**Link**: [PDF](https://arxiv.org/pdf/2505.15636)  

**Abstract**: Nearest neighbor search is central in machine learning, information retrieval, and databases. For high-dimensional datasets, graph-based methods such as HNSW, DiskANN, and NSG have become popular thanks to their empirical accuracy and efficiency. These methods construct a directed graph over the dataset and perform beam search on the graph to find nodes close to a given query. While significant work has focused on practical refinements and theoretical understanding of graph-based methods, many questions remain. We propose a new distance-based termination condition for beam search to replace the commonly used condition based on beam width. We prove that, as long as the search graph is navigable, our resulting Adaptive Beam Search method is guaranteed to approximately solve the nearest-neighbor problem, establishing a connection between navigability and the performance of graph-based search. We also provide extensive experiments on our new termination condition for both navigable graphs and approximately navigable graphs used in practice, such as HNSW and Vamana graphs. We find that Adaptive Beam Search outperforms standard beam search over a range of recall values, data sets, graph constructions, and target number of nearest neighbors. It thus provides a simple and practical way to improve the performance of popular methods. 

---
# MIRB: Mathematical Information Retrieval Benchmark 

**Authors**: Haocheng Ju, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.15585)  

**Abstract**: Mathematical Information Retrieval (MIR) is the task of retrieving information from mathematical documents and plays a key role in various applications, including theorem search in mathematical libraries, answer retrieval on math forums, and premise selection in automated theorem proving. However, a unified benchmark for evaluating these diverse retrieval tasks has been lacking. In this paper, we introduce MIRB (Mathematical Information Retrieval Benchmark) to assess the MIR capabilities of retrieval models. MIRB includes four tasks: semantic statement retrieval, question-answer retrieval, premise retrieval, and formula retrieval, spanning a total of 12 datasets. We evaluate 13 retrieval models on this benchmark and analyze the challenges inherent to MIR. We hope that MIRB provides a comprehensive framework for evaluating MIR systems and helps advance the development of more effective retrieval models tailored to the mathematical domain. 

---
# Reranking with Compressed Document Representation 

**Authors**: Hervé Déjean, Stéphane Clinchant  

**Link**: [PDF](https://arxiv.org/pdf/2505.15394)  

**Abstract**: Reranking, the process of refining the output of a first-stage retriever, is often considered computationally expensive, especially with Large Language Models. Borrowing from recent advances in document compression for RAG, we reduce the input size by compressing documents into fixed-size embedding representations. We then teach a reranker to use compressed inputs by distillation. Although based on a billion-size model, our trained reranker using this compressed input can challenge smaller rerankers in terms of both effectiveness and efficiency, especially for long documents. Given that text compressors are still in their early development stages, we view this approach as promising. 

---
# Robust Relevance Feedback for Interactive Known-Item Video Search 

**Authors**: Zhixin Ma, Chong-Wah Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2505.15128)  

**Abstract**: Known-item search (KIS) involves only a single search target, making relevance feedback-typically a powerful technique for efficiently identifying multiple positive examples to infer user intent-inapplicable. PicHunter addresses this issue by asking users to select the top-k most similar examples to the unique search target from a displayed set. Under ideal conditions, when the user's perception aligns closely with the machine's perception of similarity, consistent and precise judgments can elevate the target to the top position within a few iterations. However, in practical scenarios, expecting users to provide consistent judgments is often unrealistic, especially when the underlying embedding features used for similarity measurements lack interpretability. To enhance robustness, we first introduce a pairwise relative judgment feedback that improves the stability of top-k selections by mitigating the impact of misaligned feedback. Then, we decompose user perception into multiple sub-perceptions, each represented as an independent embedding space. This approach assumes that users may not consistently align with a single representation but are more likely to align with one or several among multiple representations. We develop a predictive user model that estimates the combination of sub-perceptions based on each user feedback instance. The predictive user model is then trained to filter out the misaligned sub-perceptions. Experimental evaluations on the large-scale open-domain dataset V3C indicate that the proposed model can optimize over 60% search targets to the top rank when their initial ranks at the search depth between 10 and 50. Even for targets initially ranked between 1,000 and 5,000, the model achieves a success rate exceeding 40% in optimizing ranks to the top, demonstrating the enhanced robustness of relevance feedback in KIS despite inconsistent feedback. 

---
# ThinkRec: Thinking-based recommendation via LLM 

**Authors**: Qihang Yu, Kairui Fu, Shengyu Zhang, Zheqi Lv, Fan Wu, Fei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15091)  

**Abstract**: Recent advances in large language models (LLMs) have enabled more semantic-aware recommendations through natural language generation. Existing LLM for recommendation (LLM4Rec) methods mostly operate in a System 1-like manner, relying on superficial features to match similar items based on click history, rather than reasoning through deeper behavioral logic. This often leads to superficial and erroneous recommendations. Motivated by this, we propose ThinkRec, a thinking-based framework that shifts LLM4Rec from System 1 to System 2 (rational system). Technically, ThinkRec introduces a thinking activation mechanism that augments item metadata with keyword summarization and injects synthetic reasoning traces, guiding the model to form interpretable reasoning chains that consist of analyzing interaction histories, identifying user preferences, and making decisions based on target items. On top of this, we propose an instance-wise expert fusion mechanism to reduce the reasoning difficulty. By dynamically assigning weights to expert models based on users' latent features, ThinkRec adapts its reasoning path to individual users, thereby enhancing precision and personalization. Extensive experiments on real-world datasets demonstrate that ThinkRec significantly improves the accuracy and interpretability of recommendations. Our implementations are available in anonymous Github: this https URL. 

---
# An Alternative to FLOPS Regularization to Effectively Productionize SPLADE-Doc 

**Authors**: Aldo Porco, Dhruv Mehra, Igor Malioutov, Karthik Radhakrishnan, Moniba Keymanesh, Daniel Preoţiuc-Pietro, Sean MacAvaney, Pengxiang Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.15070)  

**Abstract**: Learned Sparse Retrieval (LSR) models encode text as weighted term vectors, which need to be sparse to leverage inverted index structures during retrieval. SPLADE, the most popular LSR model, uses FLOPS regularization to encourage vector sparsity during training. However, FLOPS regularization does not ensure sparsity among terms - only within a given query or document. Terms with very high Document Frequencies (DFs) substantially increase latency in production retrieval engines, such as Apache Solr, due to their lengthy posting lists. To address the issue of high DFs, we present a new variant of FLOPS regularization: DF-FLOPS. This new regularization technique penalizes the usage of high-DF terms, thereby shortening posting lists and reducing retrieval latency. Unlike other inference-time sparsification methods, such as stopword removal, DF-FLOPS regularization allows for the selective inclusion of high-frequency terms in cases where the terms are truly salient. We find that DF-FLOPS successfully reduces the prevalence of high-DF terms and lowers retrieval latency (around 10x faster) in a production-grade engine while maintaining effectiveness both in-domain (only a 2.2-point drop in MRR@10) and cross-domain (improved performance in 12 out of 13 tasks on which we tested). With retrieval latencies on par with BM25, this work provides an important step towards making LSR practical for deployment in production-grade search engines. 

---
# Personalized Diffusion Model Reshapes Cold-Start Bundle Recommendation 

**Authors**: Tuan-Nghia Bui, Huy-Son Nguyen, Cam-Van Thi Nguyen, Hoang-Quynh Le, Duc-Trong Le  

**Link**: [PDF](https://arxiv.org/pdf/2505.14901)  

**Abstract**: Bundle recommendation aims to recommend a set of items to each user. However, the sparser interactions between users and bundles raise a big challenge, especially in cold-start scenarios. Traditional collaborative filtering methods do not work well for this kind of problem because these models rely on interactions to update the latent embedding, which is hard to work in a cold-start setting. We propose a new approach (DisCo), which relies on a personalized Diffusion backbone, enhanced by disentangled aspects for the user's interest, to generate a bundle in distribution space for each user to tackle the cold-start challenge. During the training phase, DisCo adjusts an additional objective loss term to avoid bias, a prevalent issue while using the generative model for top-$K$ recommendation purposes. Our empirical experiments show that DisCo outperforms five comparative baselines by a large margin on three real-world datasets. Thereby, this study devises a promising framework and essential viewpoints in cold-start recommendation. Our materials for reproducibility are available at: this https URL. 

---
# The Atlas of In-Context Learning: How Attention Heads Shape In-Context Retrieval Augmentation 

**Authors**: Patrick Kahardipraja, Reduan Achtibat, Thomas Wiegand, Wojciech Samek, Sebastian Lapuschkin  

**Link**: [PDF](https://arxiv.org/pdf/2505.15807)  

**Abstract**: Large language models are able to exploit in-context learning to access external knowledge beyond their training data through retrieval-augmentation. While promising, its inner workings remain unclear. In this work, we shed light on the mechanism of in-context retrieval augmentation for question answering by viewing a prompt as a composition of informational components. We propose an attribution-based method to identify specialized attention heads, revealing in-context heads that comprehend instructions and retrieve relevant contextual information, and parametric heads that store entities' relational knowledge. To better understand their roles, we extract function vectors and modify their attention weights to show how they can influence the answer generation process. Finally, we leverage the gained insights to trace the sources of knowledge used during inference, paving the way towards more safe and transparent language models. 

---
# ConvSearch-R1: Enhancing Query Reformulation for Conversational Search with Reasoning via Reinforcement Learning 

**Authors**: Changtai Zhu, Siyin Wang, Ruijun Feng, Kai Song, Xipeng Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15776)  

**Abstract**: Conversational search systems require effective handling of context-dependent queries that often contain ambiguity, omission, and coreference. Conversational Query Reformulation (CQR) addresses this challenge by transforming these queries into self-contained forms suitable for off-the-shelf retrievers. However, existing CQR approaches suffer from two critical constraints: high dependency on costly external supervision from human annotations or large language models, and insufficient alignment between the rewriting model and downstream retrievers. We present ConvSearch-R1, the first self-driven framework that completely eliminates dependency on external rewrite supervision by leveraging reinforcement learning to optimize reformulation directly through retrieval signals. Our novel two-stage approach combines Self-Driven Policy Warm-Up to address the cold-start problem through retrieval-guided self-distillation, followed by Retrieval-Guided Reinforcement Learning with a specially designed rank-incentive reward shaping mechanism that addresses the sparsity issue in conventional retrieval metrics. Extensive experiments on TopiOCQA and QReCC datasets demonstrate that ConvSearch-R1 significantly outperforms previous state-of-the-art methods, achieving over 10% improvement on the challenging TopiOCQA dataset while using smaller 3B parameter models without any external supervision. 

---
# Do RAG Systems Suffer From Positional Bias? 

**Authors**: Florin Cuconasu, Simone Filice, Guy Horowitz, Yoelle Maarek, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2505.15561)  

**Abstract**: Retrieval Augmented Generation enhances LLM accuracy by adding passages retrieved from an external corpus to the LLM prompt. This paper investigates how positional bias - the tendency of LLMs to weight information differently based on its position in the prompt - affects not only the LLM's capability to capitalize on relevant passages, but also its susceptibility to distracting passages. Through extensive experiments on three benchmarks, we show how state-of-the-art retrieval pipelines, while attempting to retrieve relevant passages, systematically bring highly distracting ones to the top ranks, with over 60% of queries containing at least one highly distracting passage among the top-10 retrieved passages. As a result, the impact of the LLM positional bias, which in controlled settings is often reported as very prominent by related works, is actually marginal in real scenarios since both relevant and distracting passages are, in turn, penalized. Indeed, our findings reveal that sophisticated strategies that attempt to rearrange the passages based on LLM positional preferences do not perform better than random shuffling. 

---
# Directional Non-Commutative Monoidal Structures for Compositional Embeddings in Machine Learning 

**Authors**: Mahesh Godavarti  

**Link**: [PDF](https://arxiv.org/pdf/2505.15507)  

**Abstract**: We introduce a new algebraic structure for multi-dimensional compositional embeddings, built on directional non-commutative monoidal operators. The core contribution of this work is this novel framework, which exhibits appealing theoretical properties (associativity along each dimension and an interchange law ensuring global consistency) while remaining compatible with modern machine learning architectures. Our construction defines a distinct composition operator circ_i for each axis i, ensuring associative combination along each axis without imposing global commutativity. Importantly, all axis-specific operators commute with one another, enforcing a global interchange law that enables consistent crossaxis compositions. This is, to our knowledge, the first approach that provides a common foundation that generalizes classical sequence-modeling paradigms (e.g., structured state-space models (SSMs) and transformer self-attention) to a unified multi-dimensional framework. For example, specific one-dimensional instances of our framework can recover the familiar affine transformation algebra, vanilla self-attention, and the SSM-style recurrence. The higher-dimensional generalizations naturally support recursive, structure-aware operations in embedding spaces. We outline several potential applications unlocked by this structure-including structured positional encodings in Transformers, directional image embeddings, and symbolic modeling of sequences or grids-indicating that it could inform future deep learning model designs. We formally establish the algebraic properties of our framework and discuss efficient implementations. Finally, as our focus is theoretical, we include no experiments here and defer empirical validation to future work, which we plan to undertake. 

---
# Deliberation on Priors: Trustworthy Reasoning of Large Language Models on Knowledge Graphs 

**Authors**: Jie Ma, Ning Qu, Zhitao Gao, Rui Xing, Jun Liu, Hongbin Pei, Jiang Xie, Linyun Song, Pinghui Wang, Jing Tao, Zhou Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.15210)  

**Abstract**: Knowledge graph-based retrieval-augmented generation seeks to mitigate hallucinations in Large Language Models (LLMs) caused by insufficient or outdated knowledge. However, existing methods often fail to fully exploit the prior knowledge embedded in knowledge graphs (KGs), particularly their structural information and explicit or implicit constraints. The former can enhance the faithfulness of LLMs' reasoning, while the latter can improve the reliability of response generation. Motivated by these, we propose a trustworthy reasoning framework, termed Deliberation over Priors (DP), which sufficiently utilizes the priors contained in KGs. Specifically, DP adopts a progressive knowledge distillation strategy that integrates structural priors into LLMs through a combination of supervised fine-tuning and Kahneman-Tversky optimization, thereby improving the faithfulness of relation path generation. Furthermore, our framework employs a reasoning-introspection strategy, which guides LLMs to perform refined reasoning verification based on extracted constraint priors, ensuring the reliability of response generation. Extensive experiments on three benchmark datasets demonstrate that DP achieves new state-of-the-art performance, especially a Hit@1 improvement of 13% on the ComplexWebQuestions dataset, and generates highly trustworthy responses. We also conduct various analyses to verify its flexibility and practicality. The code is available at this https URL. 

---
# An Empirical Study on Reinforcement Learning for Reasoning-Search Interleaved LLM Agents 

**Authors**: Bowen Jin, Jinsung Yoon, Priyanka Kargupta, Sercan O. Arik, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.15117)  

**Abstract**: Reinforcement learning (RL) has demonstrated strong potential in training large language models (LLMs) capable of complex reasoning for real-world problem solving. More recently, RL has been leveraged to create sophisticated LLM-based search agents that adeptly combine reasoning with search engine use. While the use of RL for training search agents is promising, the optimal design of such agents remains not fully understood. In particular, key factors -- such as (1) reward formulation, (2) the choice and characteristics of the underlying LLM, and (3) the role of the search engine in the RL process -- require further investigation. In this work, we conduct comprehensive empirical studies to systematically investigate these and offer actionable insights. We highlight several key findings: format rewards are effective in improving final performance, whereas intermediate retrieval rewards have limited impact; the scale and initialization of the LLM (general-purpose vs. reasoning-specialized) significantly influence RL outcomes; and the choice of search engine plays a critical role in shaping RL training dynamics and the robustness of the trained agent during inference. These establish important guidelines for successfully building and deploying LLM-based search agents in real-world applications. Code is available at this https URL. 

---
# StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization 

**Authors**: Ziliang Wang, Xuhui Zheng, Kang An, Cijun Ouyang, Jialu Cai, Yuhang Wang, Yichao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.15107)  

**Abstract**: Efficient multi-hop reasoning requires Large Language Models (LLMs) based agents to acquire high-value external knowledge iteratively. Previous work has explored reinforcement learning (RL) to train LLMs to perform search-based document retrieval, achieving notable improvements in QA performance, but underperform on complex, multi-hop QA resulting from the sparse rewards from global signal only. To address this gap in existing research, we introduce StepSearch, a framework for search LLMs that trained with step-wise proximal policy optimization method. It consists of richer and more detailed intermediate search rewards and token-level process supervision based on information gain and redundancy penalties to better guide each search step. We constructed a fine-grained question-answering dataset containing sub-question-level search trajectories based on open source datasets through a set of data pipeline method. On standard multi-hop QA benchmarks, it significantly outperforms global-reward baselines, achieving 11.2% and 4.2% absolute improvements for 3B and 7B models over various search with RL baselines using only 19k training data, demonstrating the effectiveness of fine-grained, stepwise supervision in optimizing deep search LLMs. Our implementation is publicly available at this https URL. 

---
# MoTime: A Dataset Suite for Multimodal Time Series Forecasting 

**Authors**: Xin Zhou, Weiqing Wang, Francisco J. Baldán, Wray Buntine, Christoph Bergmeir  

**Link**: [PDF](https://arxiv.org/pdf/2505.15072)  

**Abstract**: While multimodal data sources are increasingly available from real-world forecasting, most existing research remains on unimodal time series. In this work, we present MoTime, a suite of multimodal time series forecasting datasets that pair temporal signals with external modalities such as text, metadata, and images. Covering diverse domains, MoTime supports structured evaluation of modality utility under two scenarios: 1) the common forecasting task, where varying-length history is available, and 2) cold-start forecasting, where no historical data is available. Experiments show that external modalities can improve forecasting performance in both scenarios, with particularly strong benefits for short series in some datasets, though the impact varies depending on data characteristics. By making datasets and findings publicly available, we aim to support more comprehensive and realistic benchmarks in future multimodal time series forecasting research. 

---
# GitHub Repository Complexity Leads to Diminished Web Archive Availability 

**Authors**: David Calano, Michele C. Weigle, Michael L. Nelson  

**Link**: [PDF](https://arxiv.org/pdf/2505.15042)  

**Abstract**: Software is often developed using versioned controlled software, such as Git, and hosted on centralized Web hosts, such as GitHub and GitLab. These Web hosted software repositories are made available to users in the form of traditional HTML Web pages for each source file and directory, as well as a presentational home page and various descriptive pages. We examined more than 12,000 Web hosted Git repository project home pages, primarily from GitHub, to measure how well their presentational components are preserved in the Internet Archive, as well as the source trees of the collected GitHub repositories to assess the extent to which their source code has been preserved. We found that more than 31% of the archived repository home pages examined exhibited some form of minor page damage and 1.6% exhibited major page damage. We also found that of the source trees analyzed, less than 5% of their source files were archived, on average, with the majority of repositories not having source files saved in the Internet Archive at all. The highest concentration of archived source files available were those linked directly from repositories' home pages at a rate of 14.89% across all available repositories and sharply dropping off at deeper levels of a repository's directory tree. 

---
# Are the confidence scores of reviewers consistent with the review content? Evidence from top conference proceedings in AI 

**Authors**: Wenqing Wu, Haixu Xi, Chengzhi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15031)  

**Abstract**: Peer review is vital in academia for evaluating research quality. Top AI conferences use reviewer confidence scores to ensure review reliability, but existing studies lack fine-grained analysis of text-score consistency, potentially missing key details. This work assesses consistency at word, sentence, and aspect levels using deep learning and NLP conference review data. We employ deep learning to detect hedge sentences and aspects, then analyze report length, hedge word/sentence frequency, aspect mentions, and sentiment to evaluate text-score alignment. Correlation, significance, and regression tests examine confidence scores' impact on paper outcomes. Results show high text-score consistency across all levels, with regression revealing higher confidence scores correlate with paper rejection, validating expert assessments and peer review fairness. 

---
# CRAFT: Training-Free Cascaded Retrieval for Tabular QA 

**Authors**: Adarsh Singh, Kushal Raj Bhandari, Jianxi Gao, Soham Dan, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.14984)  

**Abstract**: Table Question Answering (TQA) involves retrieving relevant tables from a large corpus to answer natural language queries. Traditional dense retrieval models, such as DTR and ColBERT, not only incur high computational costs for large-scale retrieval tasks but also require retraining or fine-tuning on new datasets, limiting their adaptability to evolving domains and knowledge. In this work, we propose $\textbf{CRAFT}$, a cascaded retrieval approach that first uses a sparse retrieval model to filter a subset of candidate tables before applying more computationally expensive dense models and neural re-rankers. Our approach achieves better retrieval performance than state-of-the-art (SOTA) sparse, dense, and hybrid retrievers. We further enhance table representations by generating table descriptions and titles using Gemini Flash 1.5. End-to-end TQA results using various Large Language Models (LLMs) on NQ-Tables, a subset of the Natural Questions Dataset, demonstrate $\textbf{CRAFT}$ effectiveness. 

---
# Privacy Preserving Conversion Modeling in Data Clean Room 

**Authors**: Kungang Li, Xiangyi Chen, Ling Leng, Jiajing Xu, Jiankai Sun, Behnam Rezaei  

**Link**: [PDF](https://arxiv.org/pdf/2505.14959)  

**Abstract**: In the realm of online advertising, accurately predicting the conversion rate (CVR) is crucial for enhancing advertising efficiency and user satisfaction. This paper addresses the challenge of CVR prediction while adhering to user privacy preferences and advertiser requirements. Traditional methods face obstacles such as the reluctance of advertisers to share sensitive conversion data and the limitations of model training in secure environments like data clean rooms. We propose a novel model training framework that enables collaborative model training without sharing sample-level gradients with the advertising platform. Our approach introduces several innovative components: (1) utilizing batch-level aggregated gradients instead of sample-level gradients to minimize privacy risks; (2) applying adapter-based parameter-efficient fine-tuning and gradient compression to reduce communication costs; and (3) employing de-biasing techniques to train the model under label differential privacy, thereby maintaining accuracy despite privacy-enhanced label perturbations. Our experimental results, conducted on industrial datasets, demonstrate that our method achieves competitive ROCAUC performance while significantly decreasing communication overhead and complying with both advertiser privacy requirements and user privacy choices. This framework establishes a new standard for privacy-preserving, high-performance CVR prediction in the digital advertising landscape. 

---
