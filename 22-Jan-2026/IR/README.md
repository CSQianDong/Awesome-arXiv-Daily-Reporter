# Beyond the Geometric Curse: High-Dimensional N-Gram Hashing for Dense Retrieval 

**Authors**: Sangeet Sharma  

**Link**: [PDF](https://arxiv.org/pdf/2601.15205)  

**Abstract**: Why do even the most powerful 7B-parameter embedding models struggle with simple retrieval tasks that the decades old BM25 handles with ease? Recent theory suggests that this happens because of a dimensionality bottleneck. This occurs when we force infinite linguistic nuances into small, fixed-length learned vectors. We developed NUMEN to break this bottleneck by removing the learning process entirely. Instead of training heavy layers to map text to a constrained space, NUMEN uses deterministic character hashing to project language directly onto high-dimensional vectors. This approach requires no training, supports an unlimited vocabulary, and allows the geometric capacity scale as needed. On the LIMIT benchmark, NUMEN achieves 93.90 % Recall@100 at 32,768 dimensions. This makes it the first dense retrieval model to officially surpass the sparse BM25 baseline 93.6 %. Our findings show that the real problem in dense retrieval isn't the architecture, but the embedding layer itself. The solution isn't necessarily smarter training, but simply providing more room to breathe. 

---
# From Insight to Intervention: Interpretable Neuron Steering for Controlling Popularity Bias in Recommender Systems 

**Authors**: Parviz Ahmadov, Masoud Mansoury  

**Link**: [PDF](https://arxiv.org/pdf/2601.15122)  

**Abstract**: Popularity bias is a pervasive challenge in recommender systems, where a few popular items dominate attention while the majority of less popular items remain underexposed. This imbalance can reduce recommendation quality and lead to unfair item exposure. Although existing mitigation methods address this issue to some extent, they often lack transparency in how they operate. In this paper, we propose a post-hoc approach, PopSteer, that leverages a Sparse Autoencoder (SAE) to both interpret and mitigate popularity bias in recommendation models. The SAE is trained to replicate a trained model's behavior while enabling neuron-level interpretability. By introducing synthetic users with strong preferences for either popular or unpopular items, we identify neurons encoding popularity signals through their activation patterns. We then steer recommendations by adjusting the activations of the most biased neurons. Experiments on three public datasets with a sequential recommendation model demonstrate that PopSteer significantly enhances fairness with minimal impact on accuracy, while providing interpretable insights and fine-grained control over the fairness-accuracy trade-off. 

---
# What Should I Cite? A RAG Benchmark for Academic Citation Prediction 

**Authors**: Leqi Zheng, Jiajun Zhang, Canzhi Chen, Chaokun Wang, Hongwei Li, Yuying Li, Yaoxin Mao, Shannan Yan, Zixin Song, Zhiyuan Feng, Zhaolu Kang, Zirong Chen, Hang Zhang, Qiang Liu, Liang Wang, Ziyang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2601.14949)  

**Abstract**: With the rapid growth of Web-based academic publications, more and more papers are being published annually, making it increasingly difficult to find relevant prior work. Citation prediction aims to automatically suggest appropriate references, helping scholars navigate the expanding scientific literature. Here we present \textbf{CiteRAG}, the first comprehensive retrieval-augmented generation (RAG)-integrated benchmark for evaluating large language models on academic citation prediction, featuring a multi-level retrieval strategy, specialized retrievers, and generators. Our benchmark makes four core contributions: (1) We establish two instances of the citation prediction task with different granularity. Task 1 focuses on coarse-grained list-specific citation prediction, while Task 2 targets fine-grained position-specific citation prediction. To enhance these two tasks, we build a dataset containing 7,267 instances for Task 1 and 8,541 instances for Task 2, enabling comprehensive evaluation of both retrieval and generation. (2) We construct a three-level large-scale corpus with 554k papers spanning many major subfields, using an incremental pipeline. (3) We propose a multi-level hybrid RAG approach for citation prediction, fine-tuning embedding models with contrastive learning to capture complex citation relationships, paired with specialized generation models. (4) We conduct extensive experiments across state-of-the-art language models, including closed-source APIs, open-source models, and our fine-tuned generators, demonstrating the effectiveness of our framework. Our open-source toolkit enables reproducible evaluation and focuses on academic literature, providing the first comprehensive evaluation framework for citation prediction and serving as a methodological template for other scientific domains. Our source code and data are released at this https URL. 

---
# PULSE: Socially-Aware User Representation Modeling Toward Parameter-Efficient Graph Collaborative Filtering 

**Authors**: Doyun Choi, Cheonwoo Lee, Biniyam Aschalew Tolera, Taewook Ham, Chanyoung Park, Jaemin Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2601.14720)  

**Abstract**: Graph-based social recommendation (SocialRec) has emerged as a powerful extension of graph collaborative filtering (GCF), which leverages graph neural networks (GNNs) to capture multi-hop collaborative signals from user-item interactions. These methods enrich user representations by incorporating social network information into GCF, thereby integrating additional collaborative signals from social relations. However, existing GCF and graph-based SocialRec approaches face significant challenges: they incur high computational costs and suffer from limited scalability due to the large number of parameters required to assign explicit embeddings to all users and items. In this work, we propose PULSE (Parameter-efficient User representation Learning with Social Knowledge), a framework that addresses this limitation by constructing user representations from socially meaningful signals without creating an explicit learnable embedding for each user. PULSE reduces the parameter size by up to 50% compared to the most lightweight GCF baseline. Beyond parameter efficiency, our method achieves state-of-the-art performance, outperforming 13 GCF and graph-based social recommendation baselines across varying levels of interaction sparsity, from cold-start to highly active users, through a time- and memory-efficient modeling process. 

---
# Unified Multimodal and Multilingual Retrieval via Multi-Task Learning with NLU Integration 

**Authors**: Xinyuan Zhang, Lina Zhang, Lisung Chen, Guangyao Liu, Shuai Nie, Jiaming Xu, Runyu Shi, Ying Huang, Guoquan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2601.14714)  

**Abstract**: Multimodal retrieval systems typically employ Vision Language Models (VLMs) that encode images and text independently into vectors within a shared embedding space. Despite incorporating text encoders, VLMs consistently underperform specialized text models on text-only retrieval tasks. Moreover, introducing additional text encoders increases storage, inference overhead, and exacerbates retrieval inefficiencies, especially in multilingual settings. To address these limitations, we propose a multi-task learning framework that unifies the feature representation across images, long and short texts, and intent-rich queries. To our knowledge, this is the first work to jointly optimize multilingual image retrieval, text retrieval, and natural language understanding (NLU) tasks within a single framework. Our approach integrates image and text retrieval with a shared text encoder that is enhanced by NLU features for intent understanding and retrieval accuracy. 

---
# When Text-as-Vision Meets Semantic IDs in Generative Recommendation: An Empirical Study 

**Authors**: Shutong Qiao, Wei Yuan, Tong Chen, Xiangyu Zhao, Quoc Viet Hung Nguyen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2601.14697)  

**Abstract**: Semantic ID learning is a key interface in Generative Recommendation (GR) models, mapping items to discrete identifiers grounded in side information, most commonly via a pretrained text encoder. However, these text encoders are primarily optimized for well-formed natural language. In real-world recommendation data, item descriptions are often symbolic and attribute-centric, containing numerals, units, and abbreviations. These text encoders can break these signals into fragmented tokens, weakening semantic coherence and distorting relationships among attributes. Worse still, when moving to multimodal GR, relying on standard text encoders introduces an additional obstacle: text and image embeddings often exhibit mismatched geometric structures, making cross-modal fusion less effective and less stable.
In this paper, we revisit representation design for Semantic ID learning by treating text as a visual signal. We conduct a systematic empirical study of OCR-based text representations, obtained by rendering item descriptions into images and encoding them with vision-based OCR models. Experiments across four datasets and two generative backbones show that OCR-text consistently matches or surpasses standard text embeddings for Semantic ID learning in both unimodal and multimodal settings. Furthermore, we find that OCR-based Semantic IDs remain robust under extreme spatial-resolution compression, indicating strong robustness and efficiency in practical deployments. 

---
# Predicting Retrieval Utility and Answer Quality in Retrieval-Augmented Generation 

**Authors**: Fangzheng Tian, Debasis Ganguly, Craig Macdonald  

**Link**: [PDF](https://arxiv.org/pdf/2601.14546)  

**Abstract**: The quality of answers generated by large language models (LLMs) in retrieval-augmented generation (RAG) is largely influenced by the contextual information contained in the retrieved documents. A key challenge for improving RAG is to predict both the utility of retrieved documents -- quantified as the performance gain from using context over generation without context -- and the quality of the final answers in terms of correctness and relevance. In this paper, we define two prediction tasks within RAG. The first is retrieval performance prediction (RPP), which estimates the utility of retrieved documents. The second is generation performance prediction (GPP), which estimates the final answer quality. We hypothesise that in RAG, the topical relevance of retrieved documents correlates with their utility, suggesting that query performance prediction (QPP) approaches can be adapted for RPP and GPP. Beyond these retriever-centric signals, we argue that reader-centric features, such as the LLM's perplexity of the retrieved context conditioned on the input query, can further enhance prediction accuracy for both RPP and GPP. Finally, we propose that features reflecting query-agnostic document quality and readability can also provide useful signals to the predictions. We train linear regression models with the above categories of predictors for both RPP and GPP. Experiments on the Natural Questions (NQ) dataset show that combining predictors from multiple feature categories yields the most accurate estimates of RAG performance. 

---
# Trust Me on This: A User Study of Trustworthiness for RAG Responses 

**Authors**: Weronika Łajewska, Krisztian Balog  

**Link**: [PDF](https://arxiv.org/pdf/2601.14460)  

**Abstract**: The integration of generative AI into information access systems often presents users with synthesized answers that lack transparency. This study investigates how different types of explanations can influence user trust in responses from retrieval-augmented generation systems. We conducted a controlled, two-stage user study where participants chose the more trustworthy response from a pair-one objectively higher quality than the other-both with and without one of three explanation types: (1) source attribution, (2) factual grounding, and (3) information coverage. Our results show that while explanations significantly guide users toward selecting higher quality responses, trust is not dictated by objective quality alone: Users' judgments are also heavily influenced by response clarity, actionability, and their own prior knowledge. 

---
# Legal Retrieval for Public Defenders 

**Authors**: Dominik Stammbach, Kylie Zhang, Patty Liu, Nimra Nadeem, Lucia Zheng, Peter Henderson  

**Link**: [PDF](https://arxiv.org/pdf/2601.14348)  

**Abstract**: AI tools are increasingly suggested as solutions to assist public agencies with heavy workloads. In public defense, where a constitutional right to counsel meets the complexities of law, overwhelming caseloads and constrained resources, practitioners face especially taxing conditions. Yet, there is little evidence of how AI could meaningfully support defenders' day-to-day work. In partnership with the New Jersey Office of the Public Defender, we develop the NJ BriefBank, a retrieval tool which surfaces relevant appellate briefs to streamline legal research and writing. We show that existing legal retrieval benchmarks fail to transfer to public defense search, however adding domain knowledge improves retrieval quality. This includes query expansion with legal reasoning, domain-specific data and curated synthetic examples. To facilitate further research, we provide a taxonomy of realistic defender search queries and release a manually annotated public defense retrieval dataset. Together, our work offers starting points towards building practical, reliable retrieval AI tools for public defense, and towards more realistic legal retrieval benchmarks. 

---
# Supporting Humans in Evaluating AI Summaries of Legal Depositions 

**Authors**: Naghmeh Farzi, Laura Dietz, Dave D. Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2601.15182)  

**Abstract**: While large language models (LLMs) are increasingly used to summarize long documents, this trend poses significant challenges in the legal domain, where the factual accuracy of deposition summaries is crucial. Nugget-based methods have been shown to be extremely helpful for the automated evaluation of summarization approaches. In this work, we translate these methods to the user side and explore how nuggets could directly assist end users. Although prior systems have demonstrated the promise of nugget-based evaluation, its potential to support end users remains underexplored. Focusing on the legal domain, we present a prototype that leverages a factual nugget-based approach to support legal professionals in two concrete scenarios: (1) determining which of two summaries is better, and (2) manually improving an automatically generated summary. 

---
# Incentive-Tuning: Understanding and Designing Incentives for Empirical Human-AI Decision-Making Studies 

**Authors**: Simran Kaur, Sara Salimzadeh, Ujwal Gadiraju  

**Link**: [PDF](https://arxiv.org/pdf/2601.15064)  

**Abstract**: AI has revolutionised decision-making across various fields. Yet human judgement remains paramount for high-stakes decision-making. This has fueled explorations of collaborative decision-making between humans and AI systems, aiming to leverage the strengths of both. To explore this dynamic, researchers conduct empirical studies, investigating how humans use AI assistance for decision-making and how this collaboration impacts results. A critical aspect of conducting these studies is the role of participants, often recruited through crowdsourcing platforms. The validity of these studies hinges on the behaviours of the participants, hence effective incentives that can potentially affect these behaviours are a key part of designing and executing these studies. In this work, we aim to address the critical role of incentive design for conducting empirical human-AI decision-making studies, focusing on understanding, designing, and documenting incentive schemes. Through a thematic review of existing research, we explored the current practices, challenges, and opportunities associated with incentive design for human-AI decision-making empirical studies. We identified recurring patterns, or themes, such as what comprises the components of an incentive scheme, how incentive schemes are manipulated by researchers, and the impact they can have on research outcomes. Leveraging the acquired understanding, we curated a set of guidelines to aid researchers in designing effective incentive schemes for their studies, called the Incentive-Tuning Framework, outlining how researchers can undertake, reflect on, and document the incentive design process. By advocating for a standardised yet flexible approach to incentive design and contributing valuable insights along with practical tools, we hope to pave the way for more reliable and generalizable knowledge in the field of human-AI decision-making. 

---
