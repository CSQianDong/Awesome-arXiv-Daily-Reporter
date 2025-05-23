# A Comparative Study of Recommender Systems under Big Data Constraints 

**Authors**: Arimondo Scrivano  

**Link**: [PDF](https://arxiv.org/pdf/2504.08457)  

**Abstract**: Recommender Systems (RS) have become essential tools in a wide range of digital services, from e-commerce and streaming platforms to news and social media. As the volume of user-item interactions grows exponentially, especially in Big Data environments, selecting the most appropriate RS model becomes a critical task. This paper presents a comparative study of several state-of-the-art recommender algorithms, including EASE-R, SLIM, SLIM with ElasticNet regularization, Matrix Factorization (FunkSVD and ALS), P3Alpha, and RP3Beta. We evaluate these models according to key criteria such as scalability, computational complexity, predictive accuracy, and interpretability. The analysis considers both their theoretical underpinnings and practical applicability in large-scale scenarios. Our results highlight that while models like SLIM and SLIM-ElasticNet offer high accuracy and interpretability, they suffer from high computational costs, making them less suitable for real-time applications. In contrast, algorithms such as EASE-R and RP3Beta achieve a favorable balance between performance and scalability, proving more effective in large-scale environments. This study aims to provide guidelines for selecting the most appropriate recommender approach based on specific Big Data constraints and system requirements. 

---
# A Reproducibility Study of Graph-Based Legal Case Retrieval 

**Authors**: Gregor Donabauer, Udo Kruschwitz  

**Link**: [PDF](https://arxiv.org/pdf/2504.08400)  

**Abstract**: Legal retrieval is a widely studied area in Information Retrieval (IR) and a key task in this domain is retrieving relevant cases based on a given query case, often done by applying language models as encoders to model case similarity. Recently, Tang et al. proposed CaseLink, a novel graph-based method for legal case retrieval, which models both cases and legal charges as nodes in a network, with edges representing relationships such as references and shared semantics. This approach offers a new perspective on the task by capturing higher-order relationships of cases going beyond the stand-alone level of documents. However, while this shift in approaching legal case retrieval is a promising direction in an understudied area of graph-based legal IR, challenges in reproducing novel results have recently been highlighted, with multiple studies reporting difficulties in reproducing previous findings. Thus, in this work we reproduce CaseLink, a graph-based legal case retrieval method, to support future research in this area of IR. In particular, we aim to assess its reliability and generalizability by (i) first reproducing the original study setup and (ii) applying the approach to an additional dataset. We then build upon the original implementations by (iii) evaluating the approach's performance when using a more sophisticated graph data representation and (iv) using an open large language model (LLM) in the pipeline to address limitations that are known to result from using closed models accessed via an API. Our findings aim to improve the understanding of graph-based approaches in legal IR and contribute to improving reproducibility in the field. To achieve this, we share all our implementations and experimental artifacts with the community. 

---
# OnSET: Ontology and Semantic Exploration Toolkit 

**Authors**: Benedikt Kantz, Kevin Innerebner, Peter Waldert, Stefan Lengauer, Elisabeth Lex, Tobias Schreck  

**Link**: [PDF](https://arxiv.org/pdf/2504.08373)  

**Abstract**: Retrieval over knowledge graphs is usually performed using dedicated, complex query languages like SPARQL. We propose a novel system, Ontology and Semantic Exploration Toolkit (OnSET) that allows non-expert users to easily build queries with visual user guidance provided by topic modelling and semantic search throughout the application. OnSET allows users without any prior information about the ontology or networked knowledge to start exploring topics of interest over knowledge graphs, including the retrieval and detailed exploration of prototypical sub-graphs and their instances. Existing systems either focus on direct graph explorations or do not foster further exploration of the result set. We, however, provide a node-based editor that can extend on these missing properties of existing systems to support the search over big ontologies with sub-graph instances. Furthermore, OnSET combines efficient and open platforms to deploy the system on commodity hardware. 

---
# RAG-VR: Leveraging Retrieval-Augmented Generation for 3D Question Answering in VR Environments 

**Authors**: Shiyi Ding, Ying Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.08256)  

**Abstract**: Recent advances in large language models (LLMs) provide new opportunities for context understanding in virtual reality (VR). However, VR contexts are often highly localized and personalized, limiting the effectiveness of general-purpose LLMs. To address this challenge, we present RAG-VR, the first 3D question-answering system for VR that incorporates retrieval-augmented generation (RAG), which augments an LLM with external knowledge retrieved from a localized knowledge database to improve the answer quality. RAG-VR includes a pipeline for extracting comprehensive knowledge about virtual environments and user conditions for accurate answer generation. To ensure efficient retrieval, RAG-VR offloads the retrieval process to a nearby edge server and uses only essential information during retrieval. Moreover, we train the retriever to effectively distinguish among relevant, irrelevant, and hard-to-differentiate information in relation to questions. RAG-VR improves answer accuracy by 17.9%-41.8% and reduces end-to-end latency by 34.5%-47.3% compared with two baseline systems. 

---
# How Good Are Large Language Models for Course Recommendation in MOOCs? 

**Authors**: Boxuan Ma, Md Akib Zabed Khan, Tianyuan Yang, Agoritsa Polyzou, Shin'ichi Konomi  

**Link**: [PDF](https://arxiv.org/pdf/2504.08208)  

**Abstract**: Large Language Models (LLMs) have made significant strides in natural language processing and are increasingly being integrated into recommendation systems. However, their potential in educational recommendation systems has yet to be fully explored. This paper investigates the use of LLMs as a general-purpose recommendation model, leveraging their vast knowledge derived from large-scale corpora for course recommendation tasks. We explore a variety of approaches, ranging from prompt-based methods to more advanced fine-tuning techniques, and compare their performance against traditional recommendation models. Extensive experiments were conducted on a real-world MOOC dataset, evaluating using LLMs as course recommendation systems across key dimensions such as accuracy, diversity, and novelty. Our results demonstrate that LLMs can achieve good performance comparable to traditional models, highlighting their potential to enhance educational recommendation systems. These findings pave the way for further exploration and development of LLM-based approaches in the context of educational recommendations. 

---
# PCA-RAG: Principal Component Analysis for Efficient Retrieval-Augmented Generation 

**Authors**: Arman Khaledian, Amirreza Ghadiridehkordi, Nariman Khaledian  

**Link**: [PDF](https://arxiv.org/pdf/2504.08386)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a powerful paradigm for grounding large language models in external knowledge sources, improving the precision of agents responses. However, high-dimensional language model embeddings, often in the range of hundreds to thousands of dimensions, can present scalability challenges in terms of storage and latency, especially when processing massive financial text corpora. This paper investigates the use of Principal Component Analysis (PCA) to reduce embedding dimensionality, thereby mitigating computational bottlenecks without incurring large accuracy losses. We experiment with a real-world dataset and compare different similarity and distance metrics under both full-dimensional and PCA-compressed embeddings. Our results show that reducing vectors from 3,072 to 110 dimensions provides a sizeable (up to $60\times$) speedup in retrieval operations and a $\sim 28.6\times$ reduction in index size, with only moderate declines in correlation metrics relative to human-annotated similarity scores. These findings demonstrate that PCA-based compression offers a viable balance between retrieval fidelity and resource efficiency, essential for real-time systems such as Zanista AI's \textit{Newswitch} platform. Ultimately, our study underscores the practicality of leveraging classical dimensionality reduction techniques to scale RAG architectures for knowledge-intensive applications in finance and trading, where speed, memory efficiency, and accuracy must jointly be optimized. 

---
# Scholar Inbox: Personalized Paper Recommendations for Scientists 

**Authors**: Markus Flicke, Glenn Angrabeit, Madhav Iyengar, Vitalii Protsenko, Illia Shakun, Jovan Cicvaric, Bora Kargi, Haoyu He, Lukas Schuler, Lewin Scholz, Kavyanjali Agnihotri, Yong Cao, Andreas Geiger  

**Link**: [PDF](https://arxiv.org/pdf/2504.08385)  

**Abstract**: Scholar Inbox is a new open-access platform designed to address the challenges researchers face in staying current with the rapidly expanding volume of scientific literature. We provide personalized recommendations, continuous updates from open-access archives (arXiv, bioRxiv, etc.), visual paper summaries, semantic search, and a range of tools to streamline research workflows and promote open research access. The platform's personalized recommendation system is trained on user ratings, ensuring that recommendations are tailored to individual researchers' interests. To further enhance the user experience, Scholar Inbox also offers a map of science that provides an overview of research across domains, enabling users to easily explore specific topics. We use this map to address the cold start problem common in recommender systems, as well as an active learning strategy that iteratively prompts users to rate a selection of papers, allowing the system to learn user preferences quickly. We evaluate the quality of our recommendation system on a novel dataset of 800k user ratings, which we make publicly available, as well as via an extensive user study. this https URL 

---
# eST$^2$ Miner -- Process Discovery Based on Firing Partial Orders 

**Authors**: Sabine Folz-Weinstein, Christian Rennert, Lisa Luise Mannel, Robin Bergenthum, Wil van der Aalst  

**Link**: [PDF](https://arxiv.org/pdf/2504.08372)  

**Abstract**: Process discovery generates process models from event logs. Traditionally, an event log is defined as a multiset of traces, where each trace is a sequence of events. The total order of the events in a sequential trace is typically based on their temporal occurrence. However, real-life processes are partially ordered by nature. Different activities can occur in different parts of the process and, thus, independently of each other. Therefore, the temporal total order of events does not necessarily reflect their causal order, as also causally unrelated events may be ordered in time. Only partial orders allow to express concurrency, duration, overlap, and uncertainty of events. Consequently, there is a growing need for process mining algorithms that can directly handle partially ordered input. In this paper, we combine two well-established and efficient algorithms, the eST Miner from the process mining community and the Firing LPO algorithm from the Petri net community, to introduce the eST$^2$ Miner. The eST$^2$ Miner is a process discovery algorithm that can directly handle partially ordered input, gives strong formal guarantees, offers good runtime and excellent space complexity, and can, thus, be used in real-life applications. 

---
# Topic mining based on fine-tuning Sentence-BERT and LDA 

**Authors**: Jianheng Li, Lirong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.07984)  

**Abstract**: Research background: With the continuous development of society, consumers pay more attention to the key information of product fine-grained attributes when shopping. Research purposes: This study will fine tune the Sentence-BERT word embedding model and LDA model, mine the subject characteristics in online reviews of goods, and show consumers the details of various aspects of goods. Research methods: First, the Sentence-BERT model was fine tuned in the field of e-commerce online reviews, and the online review text was converted into a word vector set with richer semantic information; Secondly, the vectorized word set is input into the LDA model for topic feature extraction; Finally, focus on the key functions of the product through keyword analysis under the theme. Results: This study compared this model with other word embedding models and LDA models, and compared it with common topic extraction methods. The theme consistency of this model is 0.5 higher than that of other models, which improves the accuracy of theme extraction 

---
