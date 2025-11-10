# TeaRAG: A Token-Efficient Agentic Retrieval-Augmented Generation Framework 

**Authors**: Chao Zhang, Yuhao Wang, Derong Xu, Haoxin Zhang, Yuanjie Lyu, Yuhao Chen, Shuochen Liu, Tong Xu, Xiangyu Zhao, Yan Gao, Yao Hu, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.05385)  

**Abstract**: Retrieval-Augmented Generation (RAG) utilizes external knowledge to augment Large Language Models' (LLMs) reliability. For flexibility, agentic RAG employs autonomous, multi-round retrieval and reasoning to resolve queries. Although recent agentic RAG has improved via reinforcement learning, they often incur substantial token overhead from search and reasoning processes. This trade-off prioritizes accuracy over efficiency. To address this issue, this work proposes TeaRAG, a token-efficient agentic RAG framework capable of compressing both retrieval content and reasoning steps. 1) First, the retrieved content is compressed by augmenting chunk-based semantic retrieval with a graph retrieval using concise triplets. A knowledge association graph is then built from semantic similarity and co-occurrence. Finally, Personalized PageRank is leveraged to highlight key knowledge within this graph, reducing the number of tokens per retrieval. 2) Besides, to reduce reasoning steps, Iterative Process-aware Direct Preference Optimization (IP-DPO) is proposed. Specifically, our reward function evaluates the knowledge sufficiency by a knowledge matching mechanism, while penalizing excessive reasoning steps. This design can produce high-quality preference-pair datasets, supporting iterative DPO to improve reasoning conciseness. Across six datasets, TeaRAG improves the average Exact Match by 4% and 2% while reducing output tokens by 61% and 59% on Llama3-8B-Instruct and Qwen2.5-14B-Instruct, respectively. Code is available at this https URL. 

---
# QUESTER: Query Specification for Generative Retrieval 

**Authors**: Arthur Satouf, Yuxuan Zong, Habiboulaye Amadou-Boubacar, Pablo Piantanida, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2511.05301)  

**Abstract**: Generative Retrieval (GR) differs from the traditional index-then-retrieve pipeline by storing relevance in model parameters and directly generating document identifiers. However, GR often struggles to generalize and is costly to scale. We introduce QUESTER (QUEry SpecificaTion gEnerative Retrieval), which reframes GR as query specification generation - in this work, a simple keyword query handled by BM25 - using a (small) LLM. The policy is trained using reinforcement learning techniques (GRPO). Across in- and out-of-domain evaluations, we show that our model is more effective than BM25, and competitive with neural IR models, while maintaining a good efficiency 

---
# Wikipedia-based Datasets in Russian Information Retrieval Benchmark RusBEIR 

**Authors**: Grigory Kovalev, Natalia Loukachevitch, Mikhail Tikhomirov, Olga Babina, Pavel Mamaev  

**Link**: [PDF](https://arxiv.org/pdf/2511.05079)  

**Abstract**: In this paper, we present a novel series of Russian information retrieval datasets constructed from the "Did you know..." section of Russian Wikipedia. Our datasets support a range of retrieval tasks, including fact-checking, retrieval-augmented generation, and full-document retrieval, by leveraging interesting facts and their referenced Wikipedia articles annotated at the sentence level with graded relevance. We describe the methodology for dataset creation that enables the expansion of existing Russian Information Retrieval (IR) resources. Through extensive experiments, we extend the RusBEIR research by comparing lexical retrieval models, such as BM25, with state-of-the-art neural architectures fine-tuned for Russian, as well as multilingual models. Results of our experiments show that lexical methods tend to outperform neural models on full-document retrieval, while neural approaches better capture lexical semantics in shorter texts, such as in fact-checking or fine-grained retrieval. Using our newly created datasets, we also analyze the impact of document length on retrieval performance and demonstrate that combining retrieval with neural reranking consistently improves results. Our contribution expands the resources available for Russian information retrieval research and highlights the importance of accurate evaluation of retrieval models to achieve optimal performance. All datasets are publicly available at HuggingFace. To facilitate reproducibility and future research, we also release the full implementation on GitHub. 

---
# Query Generation Pipeline with Enhanced Answerability Assessment for Financial Information Retrieval 

**Authors**: Hyunkyu Kim, Yeeun Yoo, Youngjun Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2511.05000)  

**Abstract**: As financial applications of large language models (LLMs) gain attention, accurate Information Retrieval (IR) remains crucial for reliable AI services. However, existing benchmarks fail to capture the complex and domain-specific information needs of real-world banking scenarios. Building domain-specific IR benchmarks is costly and constrained by legal restrictions on using real customer data. To address these challenges, we propose a systematic methodology for constructing domain-specific IR benchmarks through LLM-based query generation. As a concrete implementation of this methodology, our pipeline combines single and multi-document query generation with an enhanced and reasoning-augmented answerability assessment method, achieving stronger alignment with human judgments than prior approaches. Using this methodology, we construct KoBankIR, comprising 815 queries derived from 204 official banking documents. Our experiments show that existing retrieval models struggle with the complex multi-document queries in KoBankIR, demonstrating the value of our systematic approach for domain-specific benchmark construction and underscoring the need for improved retrieval techniques in financial domains. 

---
# Search Is Not Retrieval: Decoupling Semantic Matching from Contextual Assembly in RAG 

**Authors**: Harshit Nainwani, Hediyeh Baban  

**Link**: [PDF](https://arxiv.org/pdf/2511.04939)  

**Abstract**: Retrieval systems are essential to contemporary AI pipelines, although most confuse two separate processes: finding relevant information and giving enough context for reasoning. We introduce the Search-Is-Not-Retrieve (SINR) framework, a dual-layer architecture that distinguishes between fine-grained search representations and coarse-grained retrieval contexts. SINR enhances the composability, scalability, and context fidelity of retrieval systems by directly connecting small, semantically accurate search chunks to larger, contextually complete retrieve chunks, all without incurring extra processing costs. This design changes retrieval from a passive step to an active one, making the system architecture more like how people process information. We discuss the SINR framework's conceptual foundation, formal structure, implementation issues, and qualitative outcomes. This provides a practical foundation for the next generation of AI systems that use retrieval. 

---
# Association via Entropy Reduction 

**Authors**: Anthony Gamst, Lawrence Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2511.04901)  

**Abstract**: Prior to recent successes using neural networks, term frequency-inverse document frequency (tf-idf) was clearly regarded as the best choice for identifying documents related to a query. We provide a different score, aver, and observe, on a dataset with ground truth marking for association, that aver does do better at finding assciated pairs than tf-idf. This example involves finding associated vertices in a large graph and that may be an area where neural networks are not currently an obvious best choice. Beyond this one anecdote, we observe that (1) aver has a natural threshold for declaring pairs as unassociated while tf-idf does not, (2) aver can distinguish between pairs of documents for which tf-idf gives a score of 1.0, (3) aver can be applied to larger collections of documents than pairs while tf-idf cannot, and (4) that aver is derived from entropy under a simple statistical model while tf-idf is a construction designed to achieve a certain goal and hence aver may be more "natural." To be fair, we also observe that (1) writing down and computing the aver score for a pair is more complex than for tf-idf and (2) that the fact that the aver score is naturally scale-free makes it more complicated to interpret aver scores. 

---
# Mapping Research Productivity of BRICS Countries with Special Reference to Coronary Artery Disease (CAD): A Scientometric Study 

**Authors**: Muneer Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2511.05211)  

**Abstract**: This study presents a comprehensive scientometric analysis of research productivity on Coronary Artery Disease (CAD) among the BRICS countries, Brazil, Russia, India, China, and South Africa, using data retrieved from the Web of Science database for the period 1990 to 2019. A total of 50,036 records were analyzed to assess publication growth trends, authorship patterns, collaboration levels, and citation impact. The findings reveal a steady increase in CAD-related publications, with China emerging as the leading contributor, followed by Brazil, Russia, India, and South Africa. English dominated as the primary language of communication, accounting for over 93% of publications. Authorship and collaboration analysis indicate a high degree of joint research, with 97.91% of studies being co-authored and a degree of collaboration of 0.98, underscoring the collective nature of scientific inquiry in this domain. The study validates the applicability of Lotkas Law for author productivity, Bradfords Law for journal distribution, and Zipfs Law for keyword frequency, while the Price Square Root Law was found inapplicable. The predominant publication format was journal articles (79.7%), and Kardiologiya (Russia) emerged as the most prolific journal. The results demonstrate significant growth in CAD research output and collaboration within BRICS, though notable disparities persist among member nations. The study recommends enhancing individual author productivity, expanding international collaboration, and supporting CAD research through strategic institutional and governmental initiatives. These findings provide valuable insights for policymakers, funding agencies, and the academic community to strengthen cardiovascular research capacity within developing economies. 

---
# The use of social media among library professionals and patrons: A review of literature 

**Authors**: Abimbola Agboke, Felicia Nkatv Undie  

**Link**: [PDF](https://arxiv.org/pdf/2511.05051)  

**Abstract**: This paper focused on the utilization of social media by library professionals and library users. It provides an understanding of social media, the most popular social media platforms utilized in the libraries. It also mentions the reasons for the adoption of social media in libraries be it academic, public, school libraries and other types of libraries. This is a review paper on the use of social media among library professionals and patrons. The findings reveal the contributions of social media to the libraries. Social media makes things easy for library professionals and library users. It enables them to connect, create awareness to new information, disseminate information instantly, and helps to market the library resources and services. Therefore, it is recommended amongst others that the library management board should encourage the use of social media in libraries. 

---
# EMO100DB: An Open Dataset of Improvised Songs with Emotion Data 

**Authors**: Daeun Hwang, Saebyul Park  

**Link**: [PDF](https://arxiv.org/pdf/2511.04755)  

**Abstract**: In this study, we introduce Emo100DB: a dataset consisting of improvised songs that were recorded and transcribed with emotion data based on Russell's circumplex model of emotion. The dataset was developed by collecting improvised songs that consist of melody, lyrics, and an instrumental accompaniment played, sung, and recorded by 20 young adults. Before recording each song, the participants were asked to report their emotional state, with the axes representing arousal and valence based on Russell's circumplex model of emotions. The dataset is organized into four emotion quadrants, and it includes the lyrics text and MIDI file of the melody extracted from the participant recordings, along with the original audio in WAV format. By providing an integrated composition of data and analysis, this study aims to offer a comprehensive dataset that allows for a diverse exploration of the relationship between music and emotion. 

---
# EncouRAGe: Evaluating RAG Local, Fast, and Reliable 

**Authors**: Jan Strich, Adeline Scharfenberg, Chris Biemann, Martin Semmann  

**Link**: [PDF](https://arxiv.org/pdf/2511.04696)  

**Abstract**: We introduce EncouRAGe, a comprehensive Python framework designed to streamline the development and evaluation of Retrieval-Augmented Generation (RAG) systems using Large Language Models (LLMs) and Embedding Models. EncouRAGe comprises five modular and extensible components: Type Manifest, RAG Factory, Inference, Vector Store, and Metrics, facilitating flexible experimentation and extensible development. The framework emphasizes scientific reproducibility, diverse evaluation metrics, and local deployment, enabling researchers to efficiently assess datasets within RAG workflows. This paper presents implementation details and an extensive evaluation across multiple benchmark datasets, including 25k QA pairs and over 51k documents. Our results show that RAG still underperforms compared to the Oracle Context, while Hybrid BM25 consistently achieves the best results across all four datasets. We further examine the effects of reranking, observing only marginal performance improvements accompanied by higher response latency. 

---
