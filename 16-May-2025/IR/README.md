# Do LLMs Memorize Recommendation Datasets? A Preliminary Study on MovieLens-1M 

**Authors**: Dario Di Palma, Felice Antonio Merra, Maurizio Sfilio, Vito Walter Anelli, Fedelucio Narducci, Tommaso Di Noia  

**Link**: [PDF](https://arxiv.org/pdf/2505.10212)  

**Abstract**: Large Language Models (LLMs) have become increasingly central to recommendation scenarios due to their remarkable natural language understanding and generation capabilities. Although significant research has explored the use of LLMs for various recommendation tasks, little effort has been dedicated to verifying whether they have memorized public recommendation dataset as part of their training data. This is undesirable because memorization reduces the generalizability of research findings, as benchmarking on memorized datasets does not guarantee generalization to unseen datasets. Furthermore, memorization can amplify biases, for example, some popular items may be recommended more frequently than others.
In this work, we investigate whether LLMs have memorized public recommendation datasets. Specifically, we examine two model families (GPT and Llama) across multiple sizes, focusing on one of the most widely used dataset in recommender systems: MovieLens-1M. First, we define dataset memorization as the extent to which item attributes, user profiles, and user-item interactions can be retrieved by prompting the LLMs. Second, we analyze the impact of memorization on recommendation performance. Lastly, we examine whether memorization varies across model families and model sizes. Our results reveal that all models exhibit some degree of memorization of MovieLens-1M, and that recommendation performance is related to the extent of memorization. We have made all the code publicly available at: this https URL 

---
# Boosting Text-to-Chart Retrieval through Training with Synthesized Semantic Insights 

**Authors**: Yifan Wu, Lutao Yan, Yizhang Zhu, Yinan Mei, Jiannan Wang, Nan Tang, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2505.10043)  

**Abstract**: Charts are crucial for data analysis and this http URL-to-chart retrieval systems have become increasingly important for Business Intelligence (BI), where users need to find relevant charts that match their analytical needs. These needs can be categorized into precise queries that are well-specified and fuzzy queries that are more exploratory -- both require understanding the semantics and context of the charts. However, existing text-to-chart retrieval solutions often fail to capture the semantic content and contextual information of charts, primarily due to the lack of comprehensive metadata (or semantic insights). To address this limitation, we propose a training data development pipeline that automatically synthesizes hierarchical semantic insights for charts, covering visual patterns (visual-oriented), statistical properties (statistics-oriented), and practical applications (task-oriented), which produces 207,498 semantic insights for 69,166 charts. Based on these, we train a CLIP-based model named ChartFinder to learn better representations of charts for text-to-chart retrieval. Our method leverages rich semantic insights during the training phase to develop a model that understands both visual and semantic aspects of this http URL evaluate text-to-chart retrieval performance, we curate the first benchmark, CRBench, for this task with 21,862 charts and 326 text queries from real-world BI applications, with ground-truth labels verified by the crowd this http URL show that ChartFinder significantly outperforms existing methods in text-to-chart retrieval tasks across various settings. For precise queries, ChartFinder achieves up to 66.9% NDCG@10, which is 11.58% higher than state-of-the-art models. In fuzzy query tasks, our method also demonstrates consistent improvements, with an average increase of 5% across nearly all metrics. 

---
# Beyond Pairwise Learning-To-Rank At Airbnb 

**Authors**: Malay Haldar, Daochen Zha, Huiji Gao, Liwei He, Sanjeev Katariya  

**Link**: [PDF](https://arxiv.org/pdf/2505.09795)  

**Abstract**: There are three fundamental asks from a ranking algorithm: it should scale to handle a large number of items, sort items accurately by their utility, and impose a total order on the items for logical consistency. But here's the catch-no algorithm can achieve all three at the same time. We call this limitation the SAT theorem for ranking algorithms. Given the dilemma, how can we design a practical system that meets user needs? Our current work at Airbnb provides an answer, with a working solution deployed at scale. We start with pairwise learning-to-rank (LTR) models-the bedrock of search ranking tech stacks today. They scale linearly with the number of items ranked and perform strongly on metrics like NDCG by learning from pairwise comparisons. They are at a sweet spot of performance vs. cost, making them an ideal choice for several industrial applications. However, they have a drawback-by ignoring interactions between items, they compromise on accuracy. To improve accuracy, we create a "true" pairwise LTR model-one that captures interactions between items during pairwise comparisons. But accuracy comes at the expense of scalability and total order, and we discuss strategies to counter these challenges. For greater accuracy, we take each item in the search result, and compare it against the rest of the items along two dimensions: (1) Superiority: How strongly do searchers prefer the given item over the remaining ones? (2) Similarity: How similar is the given item to all the other items? This forms the basis of our "all-pairwise" LTR framework, which factors in interactions across all items at once. Looking at items on the search result page all together-superiority and similarity combined-gives us a deeper understanding of what searchers truly want. We quantify the resulting improvements in searcher experience through offline and online experiments at Airbnb. 

---
# A Survey on Large Language Models in Multimodal Recommender Systems 

**Authors**: Alejo Lopez-Avila, Jinhua Du  

**Link**: [PDF](https://arxiv.org/pdf/2505.09777)  

**Abstract**: Multimodal recommender systems (MRS) integrate heterogeneous user and item data, such as text, images, and structured information, to enhance recommendation performance. The emergence of large language models (LLMs) introduces new opportunities for MRS by enabling semantic reasoning, in-context learning, and dynamic input handling. Compared to earlier pre-trained language models (PLMs), LLMs offer greater flexibility and generalisation capabilities but also introduce challenges related to scalability and model accessibility. This survey presents a comprehensive review of recent work at the intersection of LLMs and MRS, focusing on prompting strategies, fine-tuning methods, and data adaptation techniques. We propose a novel taxonomy to characterise integration patterns, identify transferable techniques from related recommendation domains, provide an overview of evaluation metrics and datasets, and point to possible future directions. We aim to clarify the emerging role of LLMs in multimodal recommendation and support future research in this rapidly evolving field. 

---
# The Impact of International Collaborations with Highly Publishing Countries in Computer Science 

**Authors**: Alberto Gomez Espes, Michael Faerber, Adam Jatowt  

**Link**: [PDF](https://arxiv.org/pdf/2505.09776)  

**Abstract**: This paper analyzes international collaborations in Computer Science, focusing on three major players: China, the European Union, and the United States. Drawing from a comprehensive literature review, we examine collaboration patterns, research impact, retraction rates, and the role of the Development Index in shaping research outcomes. Our findings show that while China, the EU, and the US lead global research efforts, other regions are narrowing the gap in publication volume. Collaborations involving these key regions tend to have lower retraction rates, reflecting stronger adherence to scientific standards. We also find that countries with a Very High Development Index contribute to research with higher citation rates and fewer retractions. Overall, this study highlights the value of international collaboration and the importance of inclusive, ethical practices in advancing global research in Computer Science. 

---
# LiDDA: Data Driven Attribution at LinkedIn 

**Authors**: John Bencina, Erkut Aykutlug, Yue Chen, Zerui Zhang, Stephanie Sorenson, Shao Tang, Changshuai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.09861)  

**Abstract**: Data Driven Attribution, which assigns conversion credits to marketing interactions based on causal patterns learned from data, is the foundation of modern marketing intelligence and vital to any marketing businesses and advertising platform. In this paper, we introduce a unified transformer-based attribution approach that can handle member-level data, aggregate-level data, and integration of external macro factors. We detail the large scale implementation of the approach at LinkedIn, showcasing significant impact. We also share learning and insights that are broadly applicable to the marketing and ad tech fields. 

---
# Causal Predictive Optimization and Generation for Business AI 

**Authors**: Liyang Zhao, Olurotimi Seton, Himadeep Reddy Reddivari, Suvendu Jena, Shadow Zhao, Rachit Kumar, Changshuai Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.09847)  

**Abstract**: The sales process involves sales functions converting leads or opportunities to customers and selling more products to existing customers. The optimization of the sales process thus is key to success of any B2B business. In this work, we introduce a principled approach to sales optimization and business AI, namely the Causal Predictive Optimization and Generation, which includes three layers: 1) prediction layer with causal ML 2) optimization layer with constraint optimization and contextual bandit 3) serving layer with Generative AI and feedback-loop for system enhancement. We detail the implementation and deployment of the system in LinkedIn, showcasing significant wins over legacy systems and sharing learning and insight broadly applicable to this field. 

---
