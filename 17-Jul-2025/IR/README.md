# Developing Visual Augmented Q&A System using Scalable Vision Embedding Retrieval & Late Interaction Re-ranker 

**Authors**: Rachna Saxena, Abhijeet Kumar, Suresh Shanmugam  

**Link**: [PDF](https://arxiv.org/pdf/2507.12378)  

**Abstract**: Traditional information extraction systems face challenges with text only language models as it does not consider infographics (visual elements of information) such as tables, charts, images etc. often used to convey complex information to readers. Multimodal LLM (MLLM) face challenges of finding needle in the haystack problem i.e., either longer context length or substantial number of documents as search space. Late interaction mechanism over visual language models has shown state of the art performance in retrieval-based vision augmented Q&A tasks. There are yet few challenges using it for RAG based multi-modal Q&A. Firstly, many popular and widely adopted vector databases do not support native multi-vector retrieval. Secondly, late interaction requires computation which inflates space footprint and can hinder enterprise adoption. Lastly, the current state of late interaction mechanism does not leverage the approximate neighbor search indexing methods for large speed ups in retrieval process. This paper explores a pragmatic approach to make vision retrieval process scalable and efficient without compromising on performance quality. We propose multi-step custom implementation utilizing widely adopted hybrid search (metadata & embedding) and state of the art late interaction re-ranker to retrieve best matching pages. Finally, MLLM are prompted as reader to generate answers from contextualized best matching pages. Through experiments, we observe that the proposed design is scalable (significant speed up) and stable (without degrading performance quality), hence can be used as production systems at enterprises. 

---
# An Ecosystem for Ontology Interoperability 

**Authors**: Zhangcheng Qiang  

**Link**: [PDF](https://arxiv.org/pdf/2507.12311)  

**Abstract**: Ontology interoperability is one of the complicated issues that restricts the use of ontologies in knowledge graphs (KGs). Different ontologies with conflicting and overlapping concepts make it difficult to design, develop, and deploy an interoperable ontology for downstream tasks. We propose an ecosystem for ontology interoperability. The ecosystem employs three state-of-the-art semantic techniques in different phases of the ontology engineering life cycle: ontology design patterns (ODPs) in the design phase, ontology matching and versioning (OM\&OV) in the develop phase, and ontology-compliant knowledge graphs (OCKGs) in the deploy phase, to achieve better ontology interoperability in real-world applications. A case study in the building domain validates the usefulness of the proposed ecosystem. 

---
# Looking for Fairness in Recommender Systems 

**Authors**: Cécile Logé  

**Link**: [PDF](https://arxiv.org/pdf/2507.12242)  

**Abstract**: Recommender systems can be found everywhere today, shaping our everyday experience whenever we're consuming content, ordering food, buying groceries online, or even just reading the news. Let's imagine we're in the process of building a recommender system to make content suggestions to users on social media. When thinking about fairness, it becomes clear there are several perspectives to consider: the users asking for tailored suggestions, the content creators hoping for some limelight, and society at large, navigating the repercussions of algorithmic recommendations. A shared fairness concern across all three is the emergence of filter bubbles, a side-effect that takes place when recommender systems are almost "too good", making recommendations so tailored that users become inadvertently confined to a narrow set of opinions/themes and isolated from alternative ideas. From the user's perspective, this is akin to manipulation. From the small content creator's perspective, this is an obstacle preventing them access to a whole range of potential fans. From society's perspective, the potential consequences are far-reaching, influencing collective opinions, social behavior and political decisions. How can our recommender system be fine-tuned to avoid the creation of filter bubbles, and ensure a more inclusive and diverse content landscape? Approaching this problem involves defining one (or more) performance metric to represent diversity, and tweaking our recommender system's performance through the lens of fairness. By incorporating this metric into our evaluation framework, we aim to strike a balance between personalized recommendations and the broader societal goal of fostering rich and varied cultures and points of view. 

---
# Sparse Autoencoders for Sequential Recommendation Models: Interpretation and Flexible Control 

**Authors**: Anton Klenitskiy, Konstantin Polev, Daria Denisova, Alexey Vasilev, Dmitry Simakov, Gleb Gusev  

**Link**: [PDF](https://arxiv.org/pdf/2507.12202)  

**Abstract**: Many current state-of-the-art models for sequential recommendations are based on transformer architectures. Interpretation and explanation of such black box models is an important research question, as a better understanding of their internals can help understand, influence, and control their behavior, which is very important in a variety of real-world applications. Recently sparse autoencoders (SAE) have been shown to be a promising unsupervised approach for extracting interpretable features from language models. These autoencoders learn to reconstruct hidden states of the transformer's internal layers from sparse linear combinations of directions in their activation space.
This paper is focused on the application of SAE to the sequential recommendation domain. We show that this approach can be successfully applied to the transformer trained on a sequential recommendation task: learned directions turn out to be more interpretable and monosemantic than the original hidden state dimensions. Moreover, we demonstrate that the features learned by SAE can be used to effectively and flexibly control the model's behavior, providing end-users with a straightforward method to adjust their recommendations to different custom scenarios and contexts. 

---
# Context-Aware Search and Retrieval Over Erasure Channels 

**Authors**: Sara Ghasvarianjahromi, Yauhen Yakimenka, Jörg Kliewer  

**Link**: [PDF](https://arxiv.org/pdf/2507.11894)  

**Abstract**: This paper introduces and analyzes a search and retrieval model that adopts key semantic communication principles from retrieval-augmented generation. We specifically present an information-theoretic analysis of a remote document retrieval system operating over a symbol erasure channel. The proposed model encodes the feature vector of a query, derived from term-frequency weights of a language corpus by using a repetition code with an adaptive rate dependent on the contextual importance of the terms. At the decoder, we select between two documents based on the contextual closeness of the recovered query. By leveraging a jointly Gaussian approximation for both the true and reconstructed similarity scores, we derive an explicit expression for the retrieval error probability, i.e., the probability under which the less similar document is selected. Numerical simulations on synthetic and real-world data (Google NQ) confirm the validity of the analysis. They further demonstrate that assigning greater redundancy to critical features effectively reduces the error rate, highlighting the effectiveness of semantic-aware feature encoding in error-prone communication settings. 

---
# Similarity-Guided Diffusion for Contrastive Sequential Recommendation 

**Authors**: Jinkyeong Choi, Yejin Noh, Donghyeon Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.11866)  

**Abstract**: In sequential recommendation systems, data augmentation and contrastive learning techniques have recently been introduced using diffusion models to achieve robust representation learning. However, most of the existing approaches use random augmentation, which risk damaging the contextual information of the original sequence. Accordingly, we propose a Similarity-Guided Diffusion for Contrastive Sequential Recommendation. Our method leverages the similarity between item embedding vectors to generate semantically consistent noise. Moreover, we utilize high confidence score in the denoising process to select our augmentation positions. This approach more effectively reflects contextual and structural information compared to augmentation at random positions. From a contrastive learning perspective, the proposed augmentation technique provides more discriminative positive and negative samples, simultaneously improving training efficiency and recommendation performance. Experimental results on five benchmark datasets show that SimDiffRec outperforms the existing baseline models. 

---
# Advancing Retrieval-Augmented Generation for Structured Enterprise and Internal Data 

**Authors**: Chandana Cheerla  

**Link**: [PDF](https://arxiv.org/pdf/2507.12425)  

**Abstract**: Organizations increasingly rely on proprietary enterprise data, including HR records, structured reports, and tabular documents, for critical decision-making. While Large Language Models (LLMs) have strong generative capabilities, they are limited by static pretraining, short context windows, and challenges in processing heterogeneous data formats. Conventional Retrieval-Augmented Generation (RAG) frameworks address some of these gaps but often struggle with structured and semi-structured data.
This work proposes an advanced RAG framework that combines hybrid retrieval strategies using dense embeddings (all-mpnet-base-v2) and BM25, enhanced by metadata-aware filtering with SpaCy NER and cross-encoder reranking. The framework applies semantic chunking to maintain textual coherence and retains tabular data structures to preserve row-column integrity. Quantized indexing optimizes retrieval efficiency, while human-in-the-loop feedback and conversation memory improve adaptability.
Experiments on enterprise datasets show notable improvements: Precision@5 increased by 15 percent (90 versus 75), Recall@5 by 13 percent (87 versus 74), and Mean Reciprocal Rank by 16 percent (0.85 versus 0.69). Qualitative evaluations show higher scores in Faithfulness (4.6 versus 3.0), Completeness (4.2 versus 2.5), and Relevance (4.5 versus 3.2) on a 5-point Likert scale. These results demonstrate the framework's effectiveness in delivering accurate, comprehensive, and contextually relevant responses for enterprise tasks. Future work includes extending to multimodal data and integrating agent-based retrieval. The source code will be released at this https URL 

---
# AFPM: Alignment-based Frame Patch Modeling for Cross-Dataset EEG Decoding 

**Authors**: Xiaoqing Chen, Siyang Li, Dongrui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.11911)  

**Abstract**: Electroencephalogram (EEG) decoding models for brain-computer interfaces (BCIs) struggle with cross-dataset learning and generalization due to channel layout inconsistencies, non-stationary signal distributions, and limited neurophysiological prior integration. To address these issues, we propose a plug-and-play Alignment-Based Frame-Patch Modeling (AFPM) framework, which has two main components: 1) Spatial Alignment, which selects task-relevant channels based on brain-region priors, aligns EEG distributions across domains, and remaps the selected channels to a unified layout; and, 2) Frame-Patch Encoding, which models multi-dataset signals into unified spatiotemporal patches for EEG decoding. Compared to 17 state-of-the-art approaches that need dataset-specific tuning, the proposed calibration-free AFPM achieves performance gains of up to 4.40% on motor imagery and 3.58% on event-related potential tasks. To our knowledge, this is the first calibration-free cross-dataset EEG decoding framework, substantially enhancing the practicalness of BCIs in real-world applications. 

---
# SIEVE: Effective Filtered Vector Search with Collection of Indexes 

**Authors**: Zhaoheng Li, Silu Huang, Wei Ding, Yongjoo Park, Jianjun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.11907)  

**Abstract**: Many real-world tasks such as recommending videos with the kids tag can be reduced to finding most similar vectors associated with hard predicates. This task, filtered vector search, is challenging as prior state-of-the-art graph-based (unfiltered) similarity search techniques quickly degenerate when hard constraints are considered. That is, effective graph-based filtered similarity search relies on sufficient connectivity for reaching the most similar items within just a few hops. To consider predicates, recent works propose modifying graph traversal to visit only the items that may satisfy predicates. However, they fail to offer the just-a-few-hops property for a wide range of predicates: they must restrict predicates significantly or lose efficiency if only a small fraction of items satisfy predicates.
We propose an opposite approach: instead of constraining traversal, we build many indexes each serving different predicate forms. For effective construction, we devise a three-dimensional analytical model capturing relationships among index size, search time, and recall, with which we follow a workload-aware approach to pack as many useful indexes as possible into a collection. At query time, the analytical model is employed yet again to discern the one that offers the fastest search at a given recall. We show superior performance and support on datasets with varying selectivities and forms: our approach achieves up to 8.06x speedup while having as low as 1% build time versus other indexes, with less than 2.15x memory of a standard HNSW graph and modest knowledge of past workloads. 

---
# AI Wizards at CheckThat! 2025: Enhancing Transformer-Based Embeddings with Sentiment for Subjectivity Detection in News Articles 

**Authors**: Matteo Fasulo, Luca Babboni, Luca Tedeschini  

**Link**: [PDF](https://arxiv.org/pdf/2507.11764)  

**Abstract**: This paper presents AI Wizards' participation in the CLEF 2025 CheckThat! Lab Task 1: Subjectivity Detection in News Articles, classifying sentences as subjective/objective in monolingual, multilingual, and zero-shot settings. Training/development datasets were provided for Arabic, German, English, Italian, and Bulgarian; final evaluation included additional unseen languages (e.g., Greek, Romanian, Polish, Ukrainian) to assess generalization. Our primary strategy enhanced transformer-based classifiers by integrating sentiment scores, derived from an auxiliary model, with sentence representations, aiming to improve upon standard fine-tuning. We explored this sentiment-augmented architecture with mDeBERTaV3-base, ModernBERT-base (English), and Llama3.2-1B. To address class imbalance, prevalent across languages, we employed decision threshold calibration optimized on the development set. Our experiments show sentiment feature integration significantly boosts performance, especially subjective F1 score. This framework led to high rankings, notably 1st for Greek (Macro F1 = 0.51). 

---
