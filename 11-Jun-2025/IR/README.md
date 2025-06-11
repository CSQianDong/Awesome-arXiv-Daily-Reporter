# Multimodal Representation Alignment for Cross-modal Information Retrieval 

**Authors**: Fan Xu, Luis A. Leiva  

**Link**: [PDF](https://arxiv.org/pdf/2506.08774)  

**Abstract**: Different machine learning models can represent the same underlying concept in different ways. This variability is particularly valuable for in-the-wild multimodal retrieval, where the objective is to identify the corresponding representation in one modality given another modality as input. This challenge can be effectively framed as a feature alignment problem. For example, given a sentence encoded by a language model, retrieve the most semantically aligned image based on features produced by an image encoder, or vice versa. In this work, we first investigate the geometric relationships between visual and textual embeddings derived from both vision-language models and combined unimodal models. We then align these representations using four standard similarity metrics as well as two learned ones, implemented via neural networks. Our findings indicate that the Wasserstein distance can serve as an informative measure of the modality gap, while cosine similarity consistently outperforms alternative metrics in feature alignment tasks. Furthermore, we observe that conventional architectures such as multilayer perceptrons are insufficient for capturing the complex interactions between image and text representations. Our study offers novel insights and practical considerations for researchers working in multimodal information retrieval, particularly in real-world, cross-modal applications. 

---
# Bridging RDF Knowledge Graphs with Graph Neural Networks for Semantically-Rich Recommender Systems 

**Authors**: Michael Färber, David Lamprecht, Yuni Susanti  

**Link**: [PDF](https://arxiv.org/pdf/2506.08743)  

**Abstract**: Graph Neural Networks (GNNs) have substantially advanced the field of recommender systems. However, despite the creation of more than a thousand knowledge graphs (KGs) under the W3C standard RDF, their rich semantic information has not yet been fully leveraged in GNN-based recommender systems. To address this gap, we propose a comprehensive integration of RDF KGs with GNNs that utilizes both the topological information from RDF object properties and the content information from RDF datatype properties. Our main focus is an in-depth evaluation of various GNNs, analyzing how different semantic feature initializations and types of graph structure heterogeneity influence their performance in recommendation tasks. Through experiments across multiple recommendation scenarios involving multi-million-node RDF graphs, we demonstrate that harnessing the semantic richness of RDF KGs significantly improves recommender systems and lays the groundwork for GNN-based recommender systems for the Linked Open Data cloud. The code and data are available on our GitHub repository: this https URL 

---
# Leveraging LLMs to Evaluate Usefulness of Document 

**Authors**: Xingzhu Wang, Erhan Zhang, Yiqun Chen, Jinghan Xuan, Yucheng Hou, Yitong Xu, Ying Nie, Shuaiqiang Wang, Dawei Yin, Jiaxin Mao  

**Link**: [PDF](https://arxiv.org/pdf/2506.08626)  

**Abstract**: The conventional Cranfield paradigm struggles to effectively capture user satisfaction due to its weak correlation between relevance and satisfaction, alongside the high costs of relevance annotation in building test collections. To tackle these issues, our research explores the potential of leveraging large language models (LLMs) to generate multilevel usefulness labels for evaluation. We introduce a new user-centric evaluation framework that integrates users' search context and behavioral data into LLMs. This framework uses a cascading judgment structure designed for multilevel usefulness assessments, drawing inspiration from ordinal regression techniques. Our study demonstrates that when well-guided with context and behavioral information, LLMs can accurately evaluate usefulness, allowing our approach to surpass third-party labeling methods. Furthermore, we conduct ablation studies to investigate the influence of key components within the framework. We also apply the labels produced by our method to predict user satisfaction, with real-world experiments indicating that these labels substantially improve the performance of satisfaction prediction models. 

---
# TSRec: Enhancing Repeat-Aware Recommendation from a Temporal-Sequential Perspective 

**Authors**: Shigang Quan, Shui Liu, Zhenzhe Zheng, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08531)  

**Abstract**: Repeat consumption, such as repurchasing items and relistening songs, is a common scenario in daily life. To model repeat consumption, the repeat-aware recommendation has been proposed to predict which item will be re-interacted based on the user-item interactions. In this paper, we investigate various inherent characteristics to enhance the repeat-aware recommendation. Specifically, we explore these characteristics from two aspects: one is from the temporal aspect where we consider the time interval relationship in the user behavior sequence; the other is from the sequential aspect where we consider the sequential-level relationship in the user behavior sequence. And our intuition is that both the temporal pattern and sequential pattern will reflect users' intentions of repeat consumption. By utilizing these two patterns, a novel model called Temporal and Sequential repeat-aware Recommendation(TSRec for short) is proposed to enhance repeat-aware recommendation. TSRec has three main components: 1) User-specific Temporal Representation Module (UTRM), which encodes and extracts user historical repeat temporal information. 2)Item-specific Temporal Representation Module (ITRM), which incorporates item time interval information as side information to alleviate the data sparsity problem of user repeat behavior sequence. 3) Sequential Repeat-Aware Module (SRAM), which represents the similarity between the user's current and the last repeat sequences. Extensive experimental results on three public benchmarks demonstrate the superiority of TSRec over state-of-the-art methods. The implementation code is available this https URL. 

---
# MERIT: A Merchant Incentive Ranking Model for Hotel Search & Ranking 

**Authors**: Shigang Quan, Hailong Tan, Shui Liu, Zhenzhe zheng, Ruihao Zhu, Liangyue Li, Quan Lu, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08442)  

**Abstract**: Online Travel Platforms (OTPs) have been working on improving their hotel Search & Ranking (S&R) systems that facilitate efficient matching between consumers and hotels. Existing OTPs focus almost exclusively on improving platform revenue. In this work, we take a first step in incorporating hotel merchants' objectives into the design of hotel S&R systems to achieve an incentive loop: the OTP tilts impressions and better-ranked positions to merchants with high quality, and in return, the merchants provide better service to consumers. Three critical design challenges need to be resolved to achieve this incentive loop: Matthew Effect in the consumer feedback-loop, unclear relation between hotel quality and performance, and conflicts between short-term and long-term revenue. To address these challenges, we propose MERIT, a MERchant IncenTive ranking model, which can simultaneously take the interests of merchants and consumers into account. We define a new Merchant Competitiveness Index (MCI) to represent hotel merchant quality and propose a new Merchant Tower to model the relation between MCI and ranking scores. Also, we design a monotonic structure for Merchant Tower to provide a clear relation between hotel quality and performance. Finally, we propose a Multi-objective Stratified Pairwise Loss, which can mitigate the conflicts between OTP's short-term and long-term revenue. The offline experiment results indicate that MERIT outperforms these methods in optimizing the demands of consumers and merchants. Furthermore, we conduct an online A/B test and obtain an improvement of 3.02% for the MCI score. 

---
# NAM: A Normalization Attention Model for Personalized Product Search In Fliggy 

**Authors**: Shui Liu, Mingyuan Tao, Maofei Que, Pan Li, Dong Li, Shenghua Ni, Zhuoran Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2506.08382)  

**Abstract**: Personalized product search provides significant benefits to e-commerce platforms by extracting more accurate user preferences from historical behaviors. Previous studies largely focused on the user factors when personalizing the search query, while ignoring the item perspective, which leads to the following two challenges that we summarize in this paper: First, previous approaches relying only on co-occurrence frequency tend to overestimate the conversion rates for popular items and underestimate those for long-tail items, resulting in inaccurate item similarities; Second, user purchasing propensity is highly heterogeneous according to the popularity of the target item: it is less correlated with the user's historical behavior for a popular item and more correlated for a long-tail item. To address these challenges, in this paper we propose NAM, a Normalization Attention Model, which optimizes ''when to personalize'' by utilizing Inverse Item Frequency (IIF) and employing a gating mechanism, as well as optimizes ''how to personalize'' by normalizing the attention mechanism from a global perspective. Through comprehensive experiments, we demonstrate that our proposed NAM model significantly outperforms state-of-the-art baseline models. Furthermore, we conducted an online A/B test at Fliggy, and obtained a significant improvement of 0.8% over the latest production system in conversion rate. 

---
# Reinforcement Fine-Tuning for Reasoning towards Multi-Step Multi-Source Search in Large Language Models 

**Authors**: Wentao Shi, Yiqing Shen  

**Link**: [PDF](https://arxiv.org/pdf/2506.08352)  

**Abstract**: Large language models (LLMs) can face factual limitations when responding to time-sensitive queries about recent events that arise after their knowledge thresholds in the training corpus. Existing search-augmented approaches fall into two categories, each with distinct limitations: multi-agent search frameworks incur substantial computational overhead by separating search planning and response synthesis across multiple LLMs, while single-LLM tool-calling methods restrict themselves to sequential planned, single-query searches from sole search sources. We present Reasoning-Search (R-Search), a single-LLM search framework that unifies multi-step planning, multi-source search execution, and answer synthesis within one coherent inference process. Innovatively, it structure the output into four explicitly defined components, including reasoning steps that guide the search process (<think>), a natural-language directed acyclic graph that represents the search plans with respect to diverse sources (<search>), retrieved results from executing the search plans (<result>), and synthesized final answers (<answer>). To enable effective generation of these structured outputs, we propose a specialized Reinforcement Fine-Tuning (ReFT) method based on GRPO, together with a multi-component reward function that optimizes LLM's answer correctness, structural validity of the generated DAG, and adherence to the defined output format. Experimental evaluation on FinSearchBench-24, SearchExpertBench-25, and seven Q and A benchmarks demonstrates that R-Search outperforms state-of-the-art methods, while achieving substantial efficiency gains through 70% reduction in context token usage and approximately 50% decrease in execution latency. Code is available at this https URL. 

---
# Rule-Assisted Attribute Embedding 

**Authors**: Sibo Zhao, Michael Bewong, Selasi Kwashie, Junwei Hu, Zaiwen Feng  

**Link**: [PDF](https://arxiv.org/pdf/2506.08314)  

**Abstract**: Recommendation systems often overlook the rich attribute information embedded in property graphs, limiting their effectiveness. Existing graph convolutional network (GCN) models either ignore attributes or rely on simplistic <user, item, attribute> triples, failing to capture deeper semantic structures. We propose RAE (Rule- Assisted Approach for Attribute Embedding), a novel method that improves recommendations by mining semantic rules from property graphs to guide attribute embedding. RAE performs rule-based random walks to generate enriched attribute representations, which are integrated into GCNs. Experiments on real-world datasets (BlogCatalog, Flickr) show that RAE outperforms state-of-the-art baselines by 10.6% on average in Recall@20 and NDCG@20. RAE also demonstrates greater robustness to sparse data and missing attributes, highlighting the value of leveraging structured attribute information in recommendation tasks. 

---
# Serendipitous Recommendation with Multimodal LLM 

**Authors**: Haoting Wang, Jianling Wang, Hao Li, Fangjun Yi, Mengyu Fu, Youwei Zhang, Yifan Liu, Liang Liu, Minmin Chen, Ed H. Chi, Lichan Hong, Haokai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08283)  

**Abstract**: Conventional recommendation systems succeed in identifying relevant content but often fail to provide users with surprising or novel items. Multimodal Large Language Models (MLLMs) possess the world knowledge and multimodal understanding needed for serendipity, but their integration into billion-item-scale platforms presents significant challenges. In this paper, we propose a novel hierarchical framework where fine-tuned MLLMs provide high-level guidance to conventional recommendation models, steering them towards more serendipitous suggestions. This approach leverages MLLM strengths in understanding multimodal content and user interests while retaining the efficiency of traditional models for item-level recommendation. This mitigates the complexity of applying MLLMs directly to vast action spaces. We also demonstrate a chain-of-thought strategy enabling MLLMs to discover novel user interests by first understanding video content and then identifying relevant yet unexplored interest clusters. Through live experiments within a commercial short-form video platform serving billions of users, we show that our MLLM-powered approach significantly improves both recommendation serendipity and user satisfaction. 

---
# No Stupid Questions: An Analysis of Question Query Generation for Citation Recommendation 

**Authors**: Brian D. Zimmerman, Julien Aubert-Béduchaud, Florian Boudin, Akiko Aizawa, Olga Vechtomova  

**Link**: [PDF](https://arxiv.org/pdf/2506.08196)  

**Abstract**: Existing techniques for citation recommendation are constrained by their adherence to article contents and metadata. We leverage GPT-4o-mini's latent expertise as an inquisitive assistant by instructing it to ask questions which, when answered, could expose new insights about an excerpt from a scientific article. We evaluate the utility of these questions as retrieval queries, measuring their effectiveness in retrieving and ranking masked target documents. In some cases, generated questions ended up being better queries than extractive keyword queries generated by the same model. We additionally propose MMR-RBO, a variation of Maximal Marginal Relevance (MMR) using Rank-Biased Overlap (RBO) to identify which questions will perform competitively with the keyword baseline. As all question queries yield unique result sets, we contend that there are no stupid questions. 

---
# Hierarchical Lexical Graph for Enhanced Multi-Hop Retrieval 

**Authors**: Abdellah Ghassel, Ian Robinson, Gabriel Tanase, Hal Cooper, Bryan Thompson, Zhen Han, Vassilis N. Ioannidis, Soji Adeshina, Huzefa Rangwala  

**Link**: [PDF](https://arxiv.org/pdf/2506.08074)  

**Abstract**: Retrieval-Augmented Generation (RAG) grounds large language models in external evidence, yet it still falters when answers must be pieced together across semantically distant documents. We close this gap with the Hierarchical Lexical Graph (HLG), a three-tier index that (i) traces every atomic proposition to its source, (ii) clusters propositions into latent topics, and (iii) links entities and relations to expose cross-document paths. On top of HLG we build two complementary, plug-and-play retrievers: StatementGraphRAG, which performs fine-grained entity-aware beam search over propositions for high-precision factoid questions, and TopicGraphRAG, which selects coarse topics before expanding along entity links to supply broad yet relevant context for exploratory queries. Additionally, existing benchmarks lack the complexity required to rigorously evaluate multi-hop summarization systems, often focusing on single-document queries or limited datasets. To address this, we introduce a synthetic dataset generation pipeline that curates realistic, multi-document question-answer pairs, enabling robust evaluation of multi-hop retrieval systems. Extensive experiments across five datasets demonstrate that our methods outperform naive chunk-based RAG achieving an average relative improvement of 23.1% in retrieval recall and correctness. Open-source Python library is available at this https URL. 

---
# Paths to Causality: Finding Informative Subgraphs Within Knowledge Graphs for Knowledge-Based Causal Discovery 

**Authors**: Yuni Susanti, Michael Färber  

**Link**: [PDF](https://arxiv.org/pdf/2506.08771)  

**Abstract**: Inferring causal relationships between variable pairs is crucial for understanding multivariate interactions in complex systems. Knowledge-based causal discovery -- which involves inferring causal relationships by reasoning over the metadata of variables (e.g., names or textual context) -- offers a compelling alternative to traditional methods that rely on observational data. However, existing methods using Large Language Models (LLMs) often produce unstable and inconsistent results, compromising their reliability for causal inference. To address this, we introduce a novel approach that integrates Knowledge Graphs (KGs) with LLMs to enhance knowledge-based causal discovery. Our approach identifies informative metapath-based subgraphs within KGs and further refines the selection of these subgraphs using Learning-to-Rank-based models. The top-ranked subgraphs are then incorporated into zero-shot prompts, improving the effectiveness of LLMs in inferring the causal relationship. Extensive experiments on biomedical and open-domain datasets demonstrate that our method outperforms most baselines by up to 44.4 points in F1 scores, evaluated across diverse LLMs and KGs. Our code and datasets are available on GitHub: this https URL 

---
# Efficient Context Selection for Long-Context QA: No Tuning, No Iteration, Just Adaptive-$k$ 

**Authors**: Chihiro Taguchi, Seiji Maekawa, Nikita Bhutani  

**Link**: [PDF](https://arxiv.org/pdf/2506.08479)  

**Abstract**: Retrieval-augmented generation (RAG) and long-context language models (LCLMs) both address context limitations of LLMs in open-domain question answering (QA). However, optimal external context to retrieve remains an open problem: fixing the retrieval size risks either wasting tokens or omitting key evidence. Existing adaptive methods like Self-RAG and Self-Route rely on iterative LLM prompting and perform well on factoid QA, but struggle with aggregation QA, where the optimal context size is both unknown and variable. We present Adaptive-$k$ retrieval, a simple and effective single-pass method that adaptively selects the number of passages based on the distribution of the similarity scores between the query and the candidate passages. It does not require model fine-tuning, extra LLM inferences or changes to existing retriever-reader pipelines. On both factoid and aggregation QA benchmarks, Adaptive-$k$ matches or outperforms fixed-$k$ baselines while using up to 10x fewer tokens than full-context input, yet still retrieves 70% of relevant passages. It improves accuracy across five LCLMs and two embedding models, highlighting that dynamically adjusting context size leads to more efficient and accurate QA. 

---
# Text Embeddings Should Capture Implicit Semantics, Not Just Surface Meaning 

**Authors**: Yiqun Sun, Qiang Huang, Anthony K. H. Tung, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2506.08354)  

**Abstract**: This position paper argues that the text embedding research community should move beyond surface meaning and embrace implicit semantics as a central modeling goal. Text embedding models have become foundational in modern NLP, powering a wide range of applications and drawing increasing research attention. Yet, much of this progress remains narrowly focused on surface-level semantics. In contrast, linguistic theory emphasizes that meaning is often implicit, shaped by pragmatics, speaker intent, and sociocultural context. Current embedding models are typically trained on data that lacks such depth and evaluated on benchmarks that reward the capture of surface meaning. As a result, they struggle with tasks requiring interpretive reasoning, speaker stance, or social meaning. Our pilot study highlights this gap, showing that even state-of-the-art models perform only marginally better than simplistic baselines on implicit semantics tasks. To address this, we call for a paradigm shift: embedding research should prioritize more diverse and linguistically grounded training data, design benchmarks that evaluate deeper semantic understanding, and explicitly frame implicit meaning as a core modeling objective, better aligning embeddings with real-world language complexity. 

---
# Video-ColBERT: Contextualized Late Interaction for Text-to-Video Retrieval 

**Authors**: Arun Reddy, Alexander Martin, Eugene Yang, Andrew Yates, Kate Sanders, Kenton Murray, Reno Kriz, Celso M. de Melo, Benjamin Van Durme, Rama Chellappa  

**Link**: [PDF](https://arxiv.org/pdf/2503.19009)  

**Abstract**: In this work, we tackle the problem of text-to-video retrieval (T2VR). Inspired by the success of late interaction techniques in text-document, text-image, and text-video retrieval, our approach, Video-ColBERT, introduces a simple and efficient mechanism for fine-grained similarity assessment between queries and videos. Video-ColBERT is built upon 3 main components: a fine-grained spatial and temporal token-wise interaction, query and visual expansions, and a dual sigmoid loss during training. We find that this interaction and training paradigm leads to strong individual, yet compatible, representations for encoding video content. These representations lead to increases in performance on common text-to-video retrieval benchmarks compared to other bi-encoder methods. 

---
