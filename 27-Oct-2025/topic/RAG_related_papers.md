# Redefining Retrieval Evaluation in the Era of LLMs 

**Authors**: Giovanni Trappolini, Florin Cuconasu, Simone Filice, Yoelle Maarek, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2510.21440)  

**Abstract**: Traditional Information Retrieval (IR) metrics, such as nDCG, MAP, and MRR, assume that human users sequentially examine documents with diminishing attention to lower ranks. This assumption breaks down in Retrieval Augmented Generation (RAG) systems, where search results are consumed by Large Language Models (LLMs), which, unlike humans, process all retrieved documents as a whole rather than sequentially. Additionally, traditional IR metrics do not account for related but irrelevant documents that actively degrade generation quality, rather than merely being ignored. Due to these two major misalignments, namely human vs. machine position discount and human relevance vs. machine utility, classical IR metrics do not accurately predict RAG performance. We introduce a utility-based annotation schema that quantifies both the positive contribution of relevant passages and the negative impact of distracting ones. Building on this foundation, we propose UDCG (Utility and Distraction-aware Cumulative Gain), a metric using an LLM-oriented positional discount to directly optimize the correlation with the end-to-end answer accuracy. Experiments on five datasets and six LLMs demonstrate that UDCG improves correlation by up to 36% compared to traditional metrics. Our work provides a critical step toward aligning IR evaluation with LLM consumers and enables more reliable assessment of RAG components 

---
# NeuroGenPoisoning: Neuron-Guided Attacks on Retrieval-Augmented Generation of LLM via Genetic Optimization of External Knowledge 

**Authors**: Hanyu Zhu, Lance Fiondella, Jiawei Yuan, Kai Zeng, Long Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2510.21144)  

**Abstract**: Retrieval-Augmented Generation (RAG) empowers Large Language Models (LLMs) to dynamically integrate external knowledge during inference, improving their factual accuracy and adaptability. However, adversaries can inject poisoned external knowledge to override the model's internal memory. While existing attacks iteratively manipulate retrieval content or prompt structure of RAG, they largely ignore the model's internal representation dynamics and neuron-level sensitivities. The underlying mechanism of RAG poisoning has not been fully studied and the effect of knowledge conflict with strong parametric knowledge in RAG is not considered. In this work, we propose NeuroGenPoisoning, a novel attack framework that generates adversarial external knowledge in RAG guided by LLM internal neuron attribution and genetic optimization. Our method first identifies a set of Poison-Responsive Neurons whose activation strongly correlates with contextual poisoning knowledge. We then employ a genetic algorithm to evolve adversarial passages that maximally activate these neurons. Crucially, our framework enables massive-scale generation of effective poisoned RAG knowledge by identifying and reusing promising but initially unsuccessful external knowledge variants via observed attribution signals. At the same time, Poison-Responsive Neurons guided poisoning can effectively resolves knowledge conflict. Experimental results across models and datasets demonstrate consistently achieving high Population Overwrite Success Rate (POSR) of over 90% while preserving fluency. Empirical evidence shows that our method effectively resolves knowledge conflict. 

---
# Bridging Language Gaps with Adaptive RAG: Improving Indonesian Language Question Answering 

**Authors**: William Christian, Daniel Adamlu, Adrian Yu, Derwin Suhartono  

**Link**: [PDF](https://arxiv.org/pdf/2510.21068)  

**Abstract**: Question Answering (QA) has seen significant improvements with the advancement of machine learning models, further studies enhanced this question answering system by retrieving external information, called Retrieval-Augmented Generation (RAG) to produce more accurate and informative answers. However, these state-of-the-art-performance is predominantly in English language. To address this gap we made an effort of bridging language gaps by incorporating Adaptive RAG system to Indonesian language. Adaptive RAG system integrates a classifier whose task is to distinguish the question complexity, which in turn determines the strategy for answering the question. To overcome the limited availability of Indonesian language dataset, our study employs machine translation as data augmentation approach. Experiments show reliable question complexity classifier; however, we observed significant inconsistencies in multi-retrieval answering strategy which negatively impacted the overall evaluation when this strategy was applied. These findings highlight both the promise and challenges of question answering in low-resource language suggesting directions for future improvement. 

---
# SBASH: a Framework for Designing and Evaluating RAG vs. Prompt-Tuned LLM Honeypots 

**Authors**: Adetayo Adebimpe, Helmut Neukirchen, Thomas Welsh  

**Link**: [PDF](https://arxiv.org/pdf/2510.21459)  

**Abstract**: Honeypots are decoy systems used for gathering valuable threat intelligence or diverting attackers away from production systems. Maximising attacker engagement is essential to their utility. However research has highlighted that context-awareness, such as the ability to respond to new attack types, systems and attacker agents, is necessary to increase engagement. Large Language Models (LLMs) have been shown as one approach to increase context awareness but suffer from several challenges including accuracy and timeliness of response time, high operational costs and data-protection issues due to cloud deployment. We propose the System-Based Attention Shell Honeypot (SBASH) framework which manages data-protection issues through the use of lightweight local LLMs. We investigate the use of Retrieval Augmented Generation (RAG) supported LLMs and non-RAG LLMs for Linux shell commands and evaluate them using several different metrics such as response time differences, realism from human testers, and similarity to a real system calculated with Levenshtein distance, SBert, and BertScore. We show that RAG improves accuracy for untuned models while models that have been tuned via a system prompt that tells the LLM to respond like a Linux system achieve without RAG a similar accuracy as untuned with RAG, while having a slightly lower latency. 

---
