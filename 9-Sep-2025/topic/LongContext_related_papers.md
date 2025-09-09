# Tree of Agents: Improving Long-Context Capabilities of Large Language Models through Multi-Perspective Reasoning 

**Authors**: Song Yu, Xiaofei Xu, Ke Deng, Li Li, Lin Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.06436)  

**Abstract**: Large language models (LLMs) face persistent challenges when handling long-context tasks, most notably the lost in the middle issue, where information located in the middle of a long input tends to be underutilized. Some existing methods that reduce input have the risk of discarding key information, while others that extend context windows often lead to attention dispersion. To address these limitations, we propose Tree of Agents (TOA), a multi-agent reasoning framework that segments the input into chunks processed by independent agents. Each agent generates its local cognition, then agents dynamically exchange information for collaborative reasoning along tree-structured paths. TOA enables agents to probe different reasoning orders for multi-perspective understanding, effectively mitigating position bias and reducing hallucinations. To improve processing efficiency, we incorporate prefix-hash caching and adaptive pruning strategies, achieving significant performance improvements with comparable API overhead. Experiments show that TOA, powered by compact LLaMA3.1-8B, significantly outperforms multiple baselines and demonstrates comparable performance to the latest and much larger commercial models, such as Gemini1.5-pro, on various long-context tasks. Code is available at this https URL. 

---
# PL-CA: A Parametric Legal Case Augmentation Framework 

**Authors**: Ao Chang, Yubo Chen, Jun Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.06356)  

**Abstract**: Conventional RAG is considered one of the most effective methods for addressing model knowledge insufficiency and hallucination, particularly in the judicial domain that requires high levels of knowledge rigor, logical consistency, and content integrity. However, the conventional RAG method only injects retrieved documents directly into the model's context, which severely constrains models due to their limited context windows and introduces additional computational overhead through excessively long contexts, thereby disrupting models' attention and degrading performance on downstream tasks. Moreover, many existing benchmarks lack expert annotation and focus solely on individual downstream tasks while real-world legal scenarios consist of multiple mixed legal tasks, indicating conventional benchmarks' inadequacy for reflecting models' true capabilities. To address these limitations, we propose PL-CA, which introduces a parametric RAG (P-RAG) framework to perform data augmentation on corpus knowledge and encode this legal knowledge into parametric vectors, and then integrates this parametric knowledge into the LLM's feed-forward networks (FFN) via LoRA, thereby alleviating models' context pressure. Additionally, we also construct a multi-task legal dataset comprising more than 2000 training and test instances, which are all expert-annotated and manually verified. We conduct our experiments on our dataset, and the experimental results demonstrate that our method reduces the overhead associated with excessively long contexts while maintaining competitive performance on downstream tasks compared to conventional RAG. Our code and dataset are provided in the appendix. 

---
# Reasoning-enhanced Query Understanding through Decomposition and Interpretation 

**Authors**: Yunfei Zhong, Jun Yang, Yixing Fan, Jiafeng Guo, Lixin Su, Maarten de Rijke, Ruqing Zhang, Dawei Yin, Xueqi Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.06544)  

**Abstract**: Accurate inference of user intent is crucial for enhancing document retrieval in modern search engines. While large language models (LLMs) have made significant strides in this area, their effectiveness has predominantly been assessed with short, keyword-based queries. As AI-driven search evolves, long-form queries with intricate intents are becoming more prevalent, yet they remain underexplored in the context of LLM-based query understanding (QU). To bridge this gap, we introduce ReDI: a Reasoning-enhanced approach for query understanding through Decomposition and Interpretation. ReDI leverages the reasoning and comprehension capabilities of LLMs in a three-stage pipeline: (i) it breaks down complex queries into targeted sub-queries to accurately capture user intent; (ii) it enriches each sub-query with detailed semantic interpretations to improve the query-document matching; and (iii) it independently retrieves documents for each sub-query and employs a fusion strategy to aggregate the results for the final ranking. We compiled a large-scale dataset of real-world complex queries from a major search engine and distilled the query understanding capabilities of teacher models into smaller models for practical application. Experiments on BRIGHT and BEIR demonstrate that ReDI consistently surpasses strong baselines in both sparse and dense retrieval paradigms, affirming its effectiveness. 

---
# Compare: A Framework for Scientific Comparisons 

**Authors**: Moritz Staudinger, Wojciech Kusa, Matteo Cancellieri, David Pride, Petr Knoth, Allan Hanbury  

**Link**: [PDF](https://arxiv.org/pdf/2509.06412)  

**Abstract**: Navigating the vast and rapidly increasing sea of academic publications to identify institutional synergies, benchmark research contributions and pinpoint key research contributions has become an increasingly daunting task, especially with the current exponential increase in new publications. Existing tools provide useful overviews or single-document insights, but none supports structured, qualitative comparisons across institutions or publications.
To address this, we demonstrate Compare, a novel framework that tackles this challenge by enabling sophisticated long-context comparisons of scientific contributions. Compare empowers users to explore and analyze research overlaps and differences at both the institutional and publication granularity, all driven by user-defined questions and automatic retrieval over online resources. For this we leverage on Retrieval-Augmented Generation over evolving data sources to foster long context knowledge synthesis. Unlike traditional scientometric tools, Compare goes beyond quantitative indicators by providing qualitative, citation-supported comparisons. 

---
