# Learning to Comparison-Shop 

**Authors**: Jie Tang, Daochen Zha, Xin Liu, Huiji Gao, Liwei He, Stephanie Moyerman, Sanjeev Katariya  

**Link**: [PDF](https://arxiv.org/pdf/2512.04009)  

**Abstract**: In online marketplaces like Airbnb, users frequently engage in comparison shopping before making purchase decisions. Despite the prevalence of this behavior, a significant disconnect persists between mainstream e-commerce search engines and users' comparison needs. Traditional ranking models often evaluate items in isolation, disregarding the context in which users compare multiple items on a search results page. While recent advances in deep learning have sought to improve ranking accuracy, diversity, and fairness by encoding listwise context, the challenge of aligning search rankings with user comparison shopping behavior remains inadequately addressed. In this paper, we propose a novel ranking architecture - Learning-to-Comparison-Shop (LTCS) System - that explicitly models and learns users' comparison shopping behaviors. Through extensive offline and online experiments, we demonstrate that our approach yields statistically significant gains in key business metrics - improving NDCG by 1.7% and boosting booking conversion rate by 0.6% in A/B testing - while also enhancing user experience. We also compare our model against state-of-the-art approaches and demonstrate that LTCS significantly outperforms them. 

---
# Algorithms for Boolean Matrix Factorization using Integer Programming and Heuristics 

**Authors**: Christos Kolomvakis, Thomas Bobille, Arnaud Vandaele, Nicolas Gillis  

**Link**: [PDF](https://arxiv.org/pdf/2512.03807)  

**Abstract**: Boolean matrix factorization (BMF) approximates a given binary input matrix as the product of two smaller binary factors. Unlike binary matrix factorization based on standard arithmetic, BMF employs the Boolean OR and AND operations for the matrix product, which improves interpretability and reduces the approximation error. It is also used in role mining and computer vision. In this paper, we first propose algorithms for BMF that perform alternating optimization (AO) of the factor matrices, where each subproblem is solved via integer programming (IP). We then design different approaches to further enhance AO-based algorithms by selecting an optimal subset of rank-one factors from multiple runs. To address the scalability limits of IP-based methods, we introduce new greedy and local-search heuristics. We also construct a new C++ data structure for Boolean vectors and matrices that is significantly faster than existing ones and is of independent interest, allowing our heuristics to scale to large datasets. We illustrate the performance of all our proposed methods and compare them with the state of the art on various real datasets, both with and without missing data, including applications in topic modeling and imaging. 

---
# M3DR: Towards Universal Multilingual Multimodal Document Retrieval 

**Authors**: Adithya S Kolavi, Vyoman Jain  

**Link**: [PDF](https://arxiv.org/pdf/2512.03514)  

**Abstract**: Multimodal document retrieval systems have shown strong progress in aligning visual and textual content for semantic search. However, most existing approaches remain heavily English-centric, limiting their effectiveness in multilingual contexts. In this work, we present M3DR (Multilingual Multimodal Document Retrieval), a framework designed to bridge this gap across languages, enabling applicability across diverse linguistic and cultural contexts. M3DR leverages synthetic multilingual document data and generalizes across different vision-language architectures and model sizes, enabling robust cross-lingual and cross-modal alignment. Using contrastive training, our models learn unified representations for text and document images that transfer effectively across languages. We validate this capability on 22 typologically diverse languages, demonstrating consistent performance and adaptability across linguistic and script variations. We further introduce a comprehensive benchmark that captures real-world multilingual scenarios, evaluating models under monolingual, multilingual, and mixed-language settings. M3DR generalizes across both single dense vector and ColBERT-style token-level multi-vector retrieval paradigms. Our models, NetraEmbed and ColNetraEmbed achieve state-of-the-art performance with ~150% relative improvements on cross-lingual retrieval. 

---
# LLM as Explainable Re-Ranker for Recommendation System 

**Authors**: Yaqi Wang, Haojia Sun, Shuting Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03439)  

**Abstract**: The application of large language models (LLMs) in recommendation systems has recently gained traction. Traditional recommendation systems often lack explainability and suffer from issues such as popularity bias. Previous research has also indicated that LLMs, when used as standalone predictors, fail to achieve accuracy comparable to traditional models. To address these challenges, we propose to use LLM as an explainable re-ranker, a hybrid approach that combines traditional recommendation models with LLMs to enhance both accuracy and interpretability. We constructed a dataset to train the re-ranker LLM and evaluated the alignment between the generated dataset and human expectations. Leveraging a two-stage training process, our model significantly improved NDCG, a key ranking metric. Moreover, the re-ranker outperformed a zero-shot baseline in ranking accuracy and interpretability. These results highlight the potential of integrating traditional recommendation models with LLMs to address limitations in existing systems and pave the way for more explainable and fair recommendation frameworks. 

---
# BookRAG: A Hierarchical Structure-aware Index-based Approach for Retrieval-Augmented Generation on Complex Documents 

**Authors**: Shu Wang, Yingli Zhou, Yixiang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2512.03413)  

**Abstract**: As an effective method to boost the performance of Large Language Models (LLMs) on the question answering (QA) task, Retrieval-Augmented Generation (RAG), which queries highly relevant information from external complex documents, has attracted tremendous attention from both industry and academia. Existing RAG approaches often focus on general documents, and they overlook the fact that many real-world documents (such as books, booklets, handbooks, etc.) have a hierarchical structure, which organizes their content from different granularity levels, leading to poor performance for the QA task. To address these limitations, we introduce BookRAG, a novel RAG approach targeted for documents with a hierarchical structure, which exploits logical hierarchies and traces entity relations to query the highly relevant information. Specifically, we build a novel index structure, called BookIndex, by extracting a hierarchical tree from the document, which serves as the role of its table of contents, using a graph to capture the intricate relationships between entities, and mapping entities to tree nodes. Leveraging the BookIndex, we then propose an agent-based query method inspired by the Information Foraging Theory, which dynamically classifies queries and employs a tailored retrieval workflow. Extensive experiments on three widely adopted benchmarks demonstrate that BookRAG achieves state-of-the-art performance, significantly outperforming baselines in both retrieval recall and QA accuracy while maintaining competitive efficiency. 

---
# AR-Med: Automated Relevance Enhancement in Medical Search via LLM-Driven Information Augmentation 

**Authors**: Chuyue Wang, Jie Feng, Yuxi Wu, Hang Zhang, Zhiguo Fan, Bing Cheng, Wei Lin  

**Link**: [PDF](https://arxiv.org/pdf/2512.03737)  

**Abstract**: Accurate and reliable search on online healthcare platforms is critical for user safety and service efficacy. Traditional methods, however, often fail to comprehend complex and nuanced user queries, limiting their effectiveness. Large language models (LLMs) present a promising solution, offering powerful semantic understanding to bridge this gap. Despite their potential, deploying LLMs in this high-stakes domain is fraught with challenges, including factual hallucinations, specialized knowledge gaps, and high operational costs. To overcome these barriers, we introduce \textbf{AR-Med}, a novel framework for \textbf{A}utomated \textbf{R}elevance assessment for \textbf{Med}ical search that has been successfully deployed at scale on the Online Medical Delivery Platforms. AR-Med grounds LLM reasoning in verified medical knowledge through a retrieval-augmented approach, ensuring high accuracy and reliability. To enable efficient online service, we design a practical knowledge distillation scheme that compresses large teacher models into compact yet powerful student models. We also introduce LocalQSMed, a multi-expert annotated benchmark developed to guide model iteration and ensure strong alignment between offline and online performance. Extensive experiments show AR-Med achieves an offline accuracy of over 93\%, a 24\% absolute improvement over the original online system, and delivers significant gains in online relevance and user satisfaction. Our work presents a practical and scalable blueprint for developing trustworthy, LLM-powered systems in real-world healthcare applications. 

---
# Tuning for TraceTarnish: Techniques, Trends, and Testing Tangible Traits 

**Authors**: Robert Dilworth  

**Link**: [PDF](https://arxiv.org/pdf/2512.03465)  

**Abstract**: In this study, we more rigorously evaluated our attack script $\textit{TraceTarnish}$, which leverages adversarial stylometry principles to anonymize the authorship of text-based messages. To ensure the efficacy and utility of our attack, we sourced, processed, and analyzed Reddit comments--comments that were later alchemized into $\textit{TraceTarnish}$ data--to gain valuable insights. The transformed $\textit{TraceTarnish}$ data was then further augmented by $\textit{StyloMetrix}$ to manufacture stylometric features--features that were culled using the Information Gain criterion, leaving only the most informative, predictive, and discriminative ones. Our results found that function words and function word types ($L\_FUNC\_A$ $\&$ $L\_FUNC\_T$); content words and content word types ($L\_CONT\_A$ $\&$ $L\_CONT\_T$); and the Type-Token Ratio ($ST\_TYPE\_TOKEN\_RATIO\_LEMMAS$) yielded significant Information-Gain readings. The identified stylometric cues--function-word frequencies, content-word distributions, and the Type-Token Ratio--serve as reliable indicators of compromise (IoCs), revealing when a text has been deliberately altered to mask its true author. Similarly, these features could function as forensic beacons, alerting defenders to the presence of an adversarial stylometry attack; granted, in the absence of the original message, this signal may go largely unnoticed, as it appears to depend on a pre- and post-transformation comparison. "In trying to erase a trace, you often imprint a larger one." Armed with this understanding, we framed $\textit{TraceTarnish}$'s operations and outputs around these five isolated features, using them to conceptualize and implement enhancements that further strengthen the attack. 

---
