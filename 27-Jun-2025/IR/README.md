# Real-time and personalized product recommendations for large e-commerce platforms 

**Authors**: Matteo Tolloso, Davide Bacciu, Shahab Mokarizadeh, Marco Varesi  

**Link**: [PDF](https://arxiv.org/pdf/2506.21368)  

**Abstract**: We present a methodology to provide real-time and personalized product recommendations for large e-commerce platforms, specifically focusing on fashion retail. Our approach aims to achieve accurate and scalable recommendations with minimal response times, ensuring user satisfaction, leveraging Graph Neural Networks and parsimonious learning methodologies. Extensive experimentation with datasets from one of the largest e-commerce platforms demonstrates the effectiveness of our approach in forecasting purchase sequences and handling multi-interaction scenarios, achieving efficient personalized recommendations under real-world constraints. 

---
# RecCoT: Enhancing Recommendation via Chain-of-Thought 

**Authors**: Shuo Yang, Jiangxia Cao, Haipeng Li, Yuqi Mao, Shuchao Pang  

**Link**: [PDF](https://arxiv.org/pdf/2506.21032)  

**Abstract**: In real-world applications, users always interact with items in multiple aspects, such as through implicit binary feedback (e.g., clicks, dislikes, long views) and explicit feedback (e.g., comments, reviews). Modern recommendation systems (RecSys) learn user-item collaborative signals from these implicit feedback signals as a large-scale binary data-streaming, subsequently recommending other highly similar items based on users' personalized historical interactions. However, from this collaborative-connection perspective, the RecSys does not focus on the actual content of the items themselves but instead prioritizes higher-probability signals of behavioral co-occurrence among items. Consequently, under this binary learning paradigm, the RecSys struggles to understand why a user likes or dislikes certain items. To alleviate it, some works attempt to utilize the content-based reviews to capture the semantic knowledge to enhance recommender models. However, most of these methods focus on predicting the ratings of reviews, but do not provide a human-understandable explanation. 

---
# Response Quality Assessment for Retrieval-Augmented Generation via Conditional Conformal Factuality 

**Authors**: Naihe Feng, Yi Sui, Shiyi Hou, Jesse C. Cresswell, Ga Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.20978)  

**Abstract**: Existing research on Retrieval-Augmented Generation (RAG) primarily focuses on improving overall question-answering accuracy, often overlooking the quality of sub-claims within generated responses. Recent methods that attempt to improve RAG trustworthiness, such as through auto-evaluation metrics, lack probabilistic guarantees or require ground truth answers. To address these limitations, we propose Conformal-RAG, a novel framework inspired by recent applications of conformal prediction (CP) on large language models (LLMs). Conformal-RAG leverages CP and internal information from the RAG mechanism to offer statistical guarantees on response quality. It ensures group-conditional coverage spanning multiple sub-domains without requiring manual labelling of conformal sets, making it suitable for complex RAG applications. Compared to existing RAG auto-evaluation methods, Conformal-RAG offers statistical guarantees on the quality of refined sub-claims, ensuring response reliability without the need for ground truth answers. Additionally, our experiments demonstrate that by leveraging information from the RAG system, Conformal-RAG retains up to 60\% more high-quality sub-claims from the response compared to direct applications of CP to LLMs, while maintaining the same reliability guarantee. 

---
# EraRAG: Efficient and Incremental Retrieval Augmented Generation for Growing Corpora 

**Authors**: Fangyuan Zhang, Zhengjun Huang, Yingli Zhou, Qintian Guo, Zhixun Li, Wensheng Luo, Di Jiang, Yixiang Fang, Xiaofang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2506.20963)  

**Abstract**: Graph-based Retrieval-Augmented Generation (Graph-RAG) enhances large language models (LLMs) by structuring retrieval over an external corpus. However, existing approaches typically assume a static corpus, requiring expensive full-graph reconstruction whenever new documents arrive, limiting their scalability in dynamic, evolving environments. To address these limitations, we introduce EraRAG, a novel multi-layered Graph-RAG framework that supports efficient and scalable dynamic updates. Our method leverages hyperplane-based Locality-Sensitive Hashing (LSH) to partition and organize the original corpus into hierarchical graph structures, enabling efficient and localized insertions of new data without disrupting the existing topology. The design eliminates the need for retraining or costly recomputation while preserving high retrieval accuracy and low latency. Experiments on large-scale benchmarks demonstrate that EraRag achieves up to an order of magnitude reduction in update time and token consumption compared to existing Graph-RAG systems, while providing superior accuracy performance. This work offers a practical path forward for RAG systems that must operate over continually growing corpora, bridging the gap between retrieval efficiency and adaptability. Our code and data are available at this https URL. 

---
# Towards Two-Stage Counterfactual Learning to Rank 

**Authors**: Shashank Gupta, Yiming Liao, Maarten de Rijke  

**Link**: [PDF](https://arxiv.org/pdf/2506.20854)  

**Abstract**: Counterfactual learning to rank (CLTR) aims to learn a ranking policy from user interactions while correcting for the inherent biases in interaction data, such as position bias. Existing CLTR methods assume a single ranking policy that selects top-K ranking from the entire document candidate set. In real-world applications, the candidate document set is on the order of millions, making a single-stage ranking policy impractical. In order to scale to millions of documents, real-world ranking systems are designed in a two-stage fashion, with a candidate generator followed by a ranker. The existing CLTR method for a two-stage offline ranking system only considers the top-1 ranking set-up and only focuses on training the candidate generator, with the ranker fixed. A CLTR method for training both the ranker and candidate generator jointly is missing from the existing literature. In this paper, we propose a two-stage CLTR estimator that considers the interaction between the two stages and estimates the joint value of the two policies offline. In addition, we propose a novel joint optimization method to train the candidate and ranker policies, respectively. To the best of our knowledge, we are the first to propose a CLTR estimator and learning method for two-stage ranking. Experimental results on a semi-synthetic benchmark demonstrate the effectiveness of the proposed joint CLTR method over baselines. 

---
# The Next Phase of Scientific Fact-Checking: Advanced Evidence Retrieval from Complex Structured Academic Papers 

**Authors**: Xingyu Deng, Xi Wang, Mark Stevenson  

**Link**: [PDF](https://arxiv.org/pdf/2506.20844)  

**Abstract**: Scientific fact-checking aims to determine the veracity of scientific claims by retrieving and analysing evidence from research literature. The problem is inherently more complex than general fact-checking since it must accommodate the evolving nature of scientific knowledge, the structural complexity of academic literature and the challenges posed by long-form, multimodal scientific expression. However, existing approaches focus on simplified versions of the problem based on small-scale datasets consisting of abstracts rather than full papers, thereby avoiding the distinct challenges associated with processing complete documents. This paper examines the limitations of current scientific fact-checking systems and reveals the many potential features and resources that could be exploited to advance their performance. It identifies key research challenges within evidence retrieval, including (1) evidence-driven retrieval that addresses semantic limitations and topic imbalance (2) time-aware evidence retrieval with citation tracking to mitigate outdated information, (3) structured document parsing to leverage long-range context, (4) handling complex scientific expressions, including tables, figures, and domain-specific terminology and (5) assessing the credibility of scientific literature. Preliminary experiments were conducted to substantiate these challenges and identify potential solutions. This perspective paper aims to advance scientific fact-checking with a specialised IR system tailored for real-world applications. 

---
# RAG-VisualRec: An Open Resource for Vision- and Text-Enhanced Retrieval-Augmented Generation in Recommendation 

**Authors**: Ali Tourani, Fatemeh Nazary, Yashar Deldjoo  

**Link**: [PDF](https://arxiv.org/pdf/2506.20817)  

**Abstract**: This paper addresses the challenge of developing multimodal recommender systems for the movie domain, where limited metadata (e.g., title, genre) often hinders the generation of robust recommendations. We introduce a resource that combines LLM-generated plot descriptions with trailer-derived visual embeddings in a unified pipeline supporting both Retrieval-Augmented Generation (RAG) and collaborative filtering. Central to our approach is a data augmentation step that transforms sparse metadata into richer textual signals, alongside fusion strategies (e.g., PCA, CCA) that integrate visual cues. Experimental evaluations demonstrate that CCA-based fusion significantly boosts recall compared to unimodal baselines, while an LLM-driven re-ranking step further improves NDCG, particularly in scenarios with limited textual data. By releasing this framework, we invite further exploration of multi-modal recommendation techniques tailored to cold-start, novelty-focused, and domain-specific settings. All code, data, and detailed documentation are publicly available at: this https URL 

---
# Maximal Matching Matters: Preventing Representation Collapse for Robust Cross-Modal Retrieval 

**Authors**: Hani Alomari, Anushka Sivakumar, Andrew Zhang, Chris Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2506.21538)  

**Abstract**: Cross-modal image-text retrieval is challenging because of the diverse possible associations between content from different modalities. Traditional methods learn a single-vector embedding to represent semantics of each sample, but struggle to capture nuanced and diverse relationships that can exist across modalities. Set-based approaches, which represent each sample with multiple embeddings, offer a promising alternative, as they can capture richer and more diverse relationships. In this paper, we show that, despite their promise, these set-based representations continue to face issues including sparse supervision and set collapse, which limits their effectiveness. To address these challenges, we propose Maximal Pair Assignment Similarity to optimize one-to-one matching between embedding sets which preserve semantic diversity within the set. We also introduce two loss functions to further enhance the representations: Global Discriminative Loss to enhance distinction among embeddings, and Intra-Set Divergence Loss to prevent collapse within each set. Our method achieves state-of-the-art performance on MS-COCO and Flickr30k without relying on external data. 

---
# skLEP: A Slovak General Language Understanding Benchmark 

**Authors**: Marek Šuppa, Andrej Ridzik, Daniel Hládek, Tomáš Javůrek, Viktória Ondrejová, Kristína Sásiková, Martin Tamajka, Marián Šimko  

**Link**: [PDF](https://arxiv.org/pdf/2506.21508)  

**Abstract**: In this work, we introduce skLEP, the first comprehensive benchmark specifically designed for evaluating Slovak natural language understanding (NLU) models. We have compiled skLEP to encompass nine diverse tasks that span token-level, sentence-pair, and document-level challenges, thereby offering a thorough assessment of model capabilities. To create this benchmark, we curated new, original datasets tailored for Slovak and meticulously translated established English NLU resources. Within this paper, we also present the first systematic and extensive evaluation of a wide array of Slovak-specific, multilingual, and English pre-trained language models using the skLEP tasks. Finally, we also release the complete benchmark data, an open-source toolkit facilitating both fine-tuning and evaluation of models, and a public leaderboard at this https URL in the hopes of fostering reproducibility and drive future research in Slovak NLU. 

---
# Text2Cypher Across Languages: Evaluating Foundational Models Beyond English 

**Authors**: Makbule Gulcin Ozsoy, William Tai  

**Link**: [PDF](https://arxiv.org/pdf/2506.21445)  

**Abstract**: Recent advances in large language models have enabled natural language interfaces that translate user questions into database queries, such as Text2SQL, Text2SPARQL, and Text2Cypher. While these interfaces enhance database accessibility, most research today focuses solely on English, with limited evaluation in other languages. This paper investigates the performance of foundational LLMs on the Text2Cypher task across multiple languages. We create and release a multilingual test set by translating English questions into Spanish and Turkish while preserving the original Cypher queries, enabling fair cross-lingual comparison. We evaluate multiple foundational models using standardized prompts and metrics. Our results show a consistent performance pattern: highest on English, then Spanish, and lowest on Turkish. We attribute this to differences in training data availability and linguistic characteristics. Additionally, we explore the impact of translating task prompts into Spanish and Turkish. Results show little to no change in evaluation metrics, suggesting prompt translation has minor impact. Our findings highlight the need for more inclusive evaluation and development in multilingual query generation. Future work includes schema localization and fine-tuning across diverse languages. 

---
# Leveraging LLM-Assisted Query Understanding for Live Retrieval-Augmented Generation 

**Authors**: Guanting Dong, Xiaoxi Li, Yuyao Zhang, Mengjie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2506.21384)  

**Abstract**: Real-world live retrieval-augmented generation (RAG) systems face significant challenges when processing user queries that are often noisy, ambiguous, and contain multiple intents. While RAG enhances large language models (LLMs) with external knowledge, current systems typically struggle with such complex inputs, as they are often trained or evaluated on cleaner data. This paper introduces Omni-RAG, a novel framework designed to improve the robustness and effectiveness of RAG systems in live, open-domain settings. Omni-RAG employs LLM-assisted query understanding to preprocess user inputs through three key modules: (1) Deep Query Understanding and Decomposition, which utilizes LLMs with tailored prompts to denoise queries (e.g., correcting spelling errors) and decompose multi-intent queries into structured sub-queries; (2) Intent-Aware Knowledge Retrieval, which performs retrieval for each sub-query from a corpus (i.e., FineWeb using OpenSearch) and aggregates the results; and (3) Reranking and Generation, where a reranker (i.e., BGE) refines document selection before a final response is generated by an LLM (i.e., Falcon-10B) using a chain-of-thought prompt. Omni-RAG aims to bridge the gap between current RAG capabilities and the demands of real-world applications, such as those highlighted by the SIGIR 2025 LiveRAG Challenge, by robustly handling complex and noisy queries. 

---
# Small Encoders Can Rival Large Decoders in Detecting Groundedness 

**Authors**: Istabrak Abbes, Gabriele Prato, Quentin Fournier, Fernando Rodriguez, Alaa Boukhary, Adam Elwood, Sarath Chandar  

**Link**: [PDF](https://arxiv.org/pdf/2506.21288)  

**Abstract**: Augmenting large language models (LLMs) with external context significantly improves their performance in natural language processing (NLP) tasks. However, LLMs struggle to answer queries reliably when the provided context lacks information, often resorting to ungrounded speculation or internal knowledge. Groundedness - generating responses strictly supported by the context - is essential for ensuring factual consistency and trustworthiness. This study focuses on detecting whether a given query is grounded in a document provided in context before the costly answer generation by LLMs. Such a detection mechanism can significantly reduce both inference time and resource consumption. We show that lightweight, task specific encoder models such as RoBERTa and NomicBERT, fine-tuned on curated datasets, can achieve accuracy comparable to state-of-the-art LLMs, such as Llama3 8B and GPT4o, in groundedness detection while reducing inference latency by orders of magnitude. The code is available at : this https URL 

---
# Enhancing Automatic Term Extraction with Large Language Models via Syntactic Retrieval 

**Authors**: Yongchan Chun, Minhyuk Kim, Dongjun Kim, Chanjun Park, Heuiseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2506.21222)  

**Abstract**: Automatic Term Extraction (ATE) identifies domain-specific expressions that are crucial for downstream tasks such as machine translation and information retrieval. Although large language models (LLMs) have significantly advanced various NLP tasks, their potential for ATE has scarcely been examined. We propose a retrieval-based prompting strategy that, in the few-shot setting, selects demonstrations according to \emph{syntactic} rather than semantic similarity. This syntactic retrieval method is domain-agnostic and provides more reliable guidance for capturing term boundaries. We evaluate the approach in both in-domain and cross-domain settings, analyzing how lexical overlap between the query sentence and its retrieved examples affects performance. Experiments on three specialized ATE benchmarks show that syntactic retrieval improves F1-score. These findings highlight the importance of syntactic cues when adapting LLMs to terminology-extraction tasks. 

---
# PeakNetFP: Peak-based Neural Audio Fingerprinting Robust to Extreme Time Stretching 

**Authors**: Guillem Cortès-Sebastià, Benjamin Martin, Emilio Molina, Xavier Serra, Romain Hennequin  

**Link**: [PDF](https://arxiv.org/pdf/2506.21086)  

**Abstract**: This work introduces PeakNetFP, the first neural audio fingerprinting (AFP) system designed specifically around spectral peaks. This novel system is designed to leverage the sparse spectral coordinates typically computed by traditional peak-based AFP methods. PeakNetFP performs hierarchical point feature extraction techniques similar to the computer vision model PointNet++, and is trained using contrastive learning like in the state-of-the-art deep learning AFP, NeuralFP. This combination allows PeakNetFP to outperform conventional AFP systems and achieves comparable performance to NeuralFP when handling challenging time-stretched audio data. In extensive evaluation, PeakNetFP maintains a Top-1 hit rate of over 90% for stretching factors ranging from 50% to 200%. Moreover, PeakNetFP offers significant efficiency advantages: compared to NeuralFP, it has 100 times fewer parameters and uses 11 times smaller input data. These features make PeakNetFP a lightweight and efficient solution for AFP tasks where time stretching is involved. Overall, this system represents a promising direction for future AFP technologies, as it successfully merges the lightweight nature of peak-based AFP with the adaptability and pattern recognition capabilities of neural network-based approaches, paving the way for more scalable and efficient solutions in the field. 

---
# A Semi-supervised Scalable Unified Framework for E-commerce Query Classification 

**Authors**: Chunyuan Yuan, Chong Zhang, Zheng Fang, Ming Pang, Xue Jiang, Changping Peng, Zhangang Lin, Ching Law  

**Link**: [PDF](https://arxiv.org/pdf/2506.21049)  

**Abstract**: Query classification, including multiple subtasks such as intent and category prediction, is vital to e-commerce applications. E-commerce queries are usually short and lack context, and the information between labels cannot be used, resulting in insufficient prior information for modeling. Most existing industrial query classification methods rely on users' posterior click behavior to construct training samples, resulting in a Matthew vicious cycle. Furthermore, the subtasks of query classification lack a unified framework, leading to low efficiency for algorithm optimization.
In this paper, we propose a novel Semi-supervised Scalable Unified Framework (SSUF), containing multiple enhanced modules to unify the query classification tasks. The knowledge-enhanced module uses world knowledge to enhance query representations and solve the problem of insufficient query information. The label-enhanced module uses label semantics and semi-supervised signals to reduce the dependence on posterior labels. The structure-enhanced module enhances the label representation based on the complex label relations. Each module is highly pluggable, and input features can be added or removed as needed according to each subtask. We conduct extensive offline and online A/B experiments, and the results show that SSUF significantly outperforms the state-of-the-art models. 

---
# Metadata Enrichment of Long Text Documents using Large Language Models 

**Authors**: Manika Lamba, You Peng, Sophie Nikolov, Glen Layne-Worthey, J. Stephen Downie  

**Link**: [PDF](https://arxiv.org/pdf/2506.20918)  

**Abstract**: In this project, we semantically enriched and enhanced the metadata of long text documents, theses and dissertations, retrieved from the HathiTrust Digital Library in English published from 1920 to 2020 through a combination of manual efforts and large language models. This dataset provides a valuable resource for advancing research in areas such as computational social science, digital humanities, and information science. Our paper shows that enriching metadata using LLMs is particularly beneficial for digital repositories by introducing additional metadata access points that may not have originally been foreseen to accommodate various content types. This approach is particularly effective for repositories that have significant missing data in their existing metadata fields, enhancing search results and improving the accessibility of the digital repository. 

---
# Engineering RAG Systems for Real-World Applications: Design, Development, and Evaluation 

**Authors**: Md Toufique Hasan, Muhammad Waseem, Kai-Kristian Kemell, Ayman Asad Khan, Mika Saari, Pekka Abrahamsson  

**Link**: [PDF](https://arxiv.org/pdf/2506.20869)  

**Abstract**: Retrieval-Augmented Generation (RAG) systems are emerging as a key approach for grounding Large Language Models (LLMs) in external knowledge, addressing limitations in factual accuracy and contextual relevance. However, there is a lack of empirical studies that report on the development of RAG-based implementations grounded in real-world use cases, evaluated through general user involvement, and accompanied by systematic documentation of lessons learned. This paper presents five domain-specific RAG applications developed for real-world scenarios across governance, cybersecurity, agriculture, industrial research, and medical diagnostics. Each system incorporates multilingual OCR, semantic retrieval via vector embeddings, and domain-adapted LLMs, deployed through local servers or cloud APIs to meet distinct user needs. A web-based evaluation involving a total of 100 participants assessed the systems across six dimensions: (i) Ease of Use, (ii) Relevance, (iii) Transparency, (iv) Responsiveness, (v) Accuracy, and (vi) Likelihood of Recommendation. Based on user feedback and our development experience, we documented twelve key lessons learned, highlighting technical, operational, and ethical challenges affecting the reliability and usability of RAG systems in practice. 

---
