# SemCORE: A Semantic-Enhanced Generative Cross-Modal Retrieval Framework with MLLMs 

**Authors**: Haoxuan Li, Yi Bin, Yunshan Ma, Guoqing Wang, Yang Yang, See-Kiong Ng, Tat-Seng Chua  

**Link**: [PDF](https://arxiv.org/pdf/2504.13172)  

**Abstract**: Cross-modal retrieval (CMR) is a fundamental task in multimedia research, focused on retrieving semantically relevant targets across different modalities. While traditional CMR methods match text and image via embedding-based similarity calculations, recent advancements in pre-trained generative models have established generative retrieval as a promising alternative. This paradigm assigns each target a unique identifier and leverages a generative model to directly predict identifiers corresponding to input queries without explicit indexing. Despite its great potential, current generative CMR approaches still face semantic information insufficiency in both identifier construction and generation processes. To address these limitations, we propose a novel unified Semantic-enhanced generative Cross-mOdal REtrieval framework (SemCORE), designed to unleash the semantic understanding capabilities in generative cross-modal retrieval task. Specifically, we first construct a Structured natural language IDentifier (SID) that effectively aligns target identifiers with generative models optimized for natural language comprehension and generation. Furthermore, we introduce a Generative Semantic Verification (GSV) strategy enabling fine-grained target discrimination. Additionally, to the best of our knowledge, SemCORE is the first framework to simultaneously consider both text-to-image and image-to-text retrieval tasks within generative cross-modal retrieval. Extensive experiments demonstrate that our framework outperforms state-of-the-art generative cross-modal retrieval methods. Notably, SemCORE achieves substantial improvements across benchmark datasets, with an average increase of 8.65 points in Recall@1 for text-to-image retrieval. 

---
# FreshStack: Building Realistic Benchmarks for Evaluating Retrieval on Technical Documents 

**Authors**: Nandan Thakur, Jimmy Lin, Sam Havens, Michael Carbin, Omar Khattab, Andrew Drozdov  

**Link**: [PDF](https://arxiv.org/pdf/2504.13128)  

**Abstract**: We introduce FreshStack, a reusable framework for automatically building information retrieval (IR) evaluation benchmarks from community-asked questions and answers. FreshStack conducts the following steps: (1) automatic corpus collection from code and technical documentation, (2) nugget generation from community-asked questions and answers, and (3) nugget-level support, retrieving documents using a fusion of retrieval techniques and hybrid architectures. We use FreshStack to build five datasets on fast-growing, recent, and niche topics to ensure the tasks are sufficiently challenging. On FreshStack, existing retrieval models, when applied out-of-the-box, significantly underperform oracle approaches on all five topics, denoting plenty of headroom to improve IR quality. In addition, we identify cases where rerankers do not clearly improve first-stage retrieval accuracy (two out of five topics). We hope that FreshStack will facilitate future work toward constructing realistic, scalable, and uncontaminated IR and RAG evaluation benchmarks. FreshStack datasets are available at: this https URL. 

---
# CSMF: Cascaded Selective Mask Fine-Tuning for Multi-Objective Embedding-Based Retrieval 

**Authors**: Hao Deng, Haibo Xing, Kanefumi Matsuyama, Moyu Zhang, Jinxin Hu, Hong Wen, Yu Zhang, Xiaoyi Zeng, Jing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.12920)  

**Abstract**: Multi-objective embedding-based retrieval (EBR) has become increasingly critical due to the growing complexity of user behaviors and commercial objectives. While traditional approaches often suffer from data sparsity and limited information sharing between objectives, recent methods utilizing a shared network alongside dedicated sub-networks for each objective partially address these limitations. However, such methods significantly increase the model parameters, leading to an increased retrieval latency and a limited ability to model causal relationships between objectives. To address these challenges, we propose the Cascaded Selective Mask Fine-Tuning (CSMF), a novel method that enhances both retrieval efficiency and serving performance for multi-objective EBR. The CSMF framework selectively masks model parameters to free up independent learning space for each objective, leveraging the cascading relationships between objectives during the sequential fine-tuning. Without increasing network parameters or online retrieval overhead, CSMF computes a linearly weighted fusion score for multiple objective probabilities while supporting flexible adjustment of each objective's weight across various recommendation scenarios. Experimental results on real-world datasets demonstrate the superior performance of CSMF, and online experiments validate its significant practical value. 

---
# Building Russian Benchmark for Evaluation of Information Retrieval Models 

**Authors**: Grigory Kovalev, Mikhail Tikhomirov, Evgeny Kozhevnikov, Max Kornilov, Natalia Loukachevitch  

**Link**: [PDF](https://arxiv.org/pdf/2504.12879)  

**Abstract**: We introduce RusBEIR, a comprehensive benchmark designed for zero-shot evaluation of information retrieval (IR) models in the Russian language. Comprising 17 datasets from various domains, it integrates adapted, translated, and newly created datasets, enabling systematic comparison of lexical and neural models. Our study highlights the importance of preprocessing for lexical models in morphologically rich languages and confirms BM25 as a strong baseline for full-document retrieval. Neural models, such as mE5-large and BGE-M3, demonstrate superior performance on most datasets, but face challenges with long-document retrieval due to input size constraints. RusBEIR offers a unified, open-source framework that promotes research in Russian-language information retrieval. 

---
# Towards Lossless Token Pruning in Late-Interaction Retrieval Models 

**Authors**: Yuxuan Zong, Benjamin Piwowarski  

**Link**: [PDF](https://arxiv.org/pdf/2504.12778)  

**Abstract**: Late interaction neural IR models like ColBERT offer a competitive effectiveness-efficiency trade-off across many benchmarks. However, they require a huge memory space to store the contextual representation for all the document tokens. Some works have proposed using either heuristics or statistical-based techniques to prune tokens from each document. This however doesn't guarantee that the removed tokens have no impact on the retrieval score. Our work uses a principled approach to define how to prune tokens without impacting the score between a document and a query. We introduce three regularization losses, that induce a solution with high pruning ratios, as well as two pruning strategies. We study them experimentally (in and out-domain), showing that we can preserve ColBERT's performance while using only 30\% of the tokens. 

---
# Validating LLM-Generated Relevance Labels for Educational Resource Search 

**Authors**: Ratan J. Sebastian, Anett Hoppe  

**Link**: [PDF](https://arxiv.org/pdf/2504.12732)  

**Abstract**: Manual relevance judgements in Information Retrieval are costly and require expertise, driving interest in using Large Language Models (LLMs) for automatic assessment. While LLMs have shown promise in general web search scenarios, their effectiveness for evaluating domain-specific search results, such as educational resources, remains unexplored. To investigate different ways of including domain-specific criteria in LLM prompts for relevance judgement, we collected and released a dataset of 401 human relevance judgements from a user study involving teaching professionals performing search tasks related to lesson planning. We compared three approaches to structuring these prompts: a simple two-aspect evaluation baseline from prior work on using LLMs as relevance judges, a comprehensive 12-dimensional rubric derived from educational literature, and criteria directly informed by the study participants. Using domain-specific frameworks, LLMs achieved strong agreement with human judgements (Cohen's $\kappa$ up to 0.650), significantly outperforming the baseline approach. The participant-derived framework proved particularly robust, with GPT-3.5 achieving $\kappa$ scores of 0.639 and 0.613 for 10-dimension and 5-dimension versions respectively. System-level evaluation showed that LLM judgements reliably identified top-performing retrieval approaches (RBO scores 0.71-0.76) while maintaining reasonable discrimination between systems (RBO 0.52-0.56). These findings suggest that LLMs can effectively evaluate educational resources when prompted with domain-specific criteria, though performance varies with framework complexity and input structure. 

---
# SimUSER: Simulating User Behavior with Large Language Models for Recommender System Evaluation 

**Authors**: Nicolas Bougie, Narimasa Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2504.12722)  

**Abstract**: Recommender systems play a central role in numerous real-life applications, yet evaluating their performance remains a significant challenge due to the gap between offline metrics and online behaviors. Given the scarcity and limits (e.g., privacy issues) of real user data, we introduce SimUSER, an agent framework that serves as believable and cost-effective human proxies. SimUSER first identifies self-consistent personas from historical data, enriching user profiles with unique backgrounds and personalities. Then, central to this evaluation are users equipped with persona, memory, perception, and brain modules, engaging in interactions with the recommender system. SimUSER exhibits closer alignment with genuine humans than prior work, both at micro and macro levels. Additionally, we conduct insightful experiments to explore the effects of thumbnails on click rates, the exposure effect, and the impact of reviews on user engagement. Finally, we refine recommender system parameters based on offline A/B test results, resulting in improved user engagement in the real world. 

---
# Benchmarking LLM-based Relevance Judgment Methods 

**Authors**: Negar Arabzadeh, Charles L. A. Clarke  

**Link**: [PDF](https://arxiv.org/pdf/2504.12558)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in both academic and industry settings to automate the evaluation of information seeking systems, particularly by generating graded relevance judgments. Previous work on LLM-based relevance assessment has primarily focused on replicating graded human relevance judgments through various prompting strategies. However, there has been limited exploration of alternative assessment methods or comprehensive comparative studies. In this paper, we systematically compare multiple LLM-based relevance assessment methods, including binary relevance judgments, graded relevance assessments, pairwise preference-based methods, and two nugget-based evaluation methods~--~document-agnostic and document-dependent. In addition to a traditional comparison based on system rankings using Kendall correlations, we also examine how well LLM judgments align with human preferences, as inferred from relevance grades. We conduct extensive experiments on datasets from three TREC Deep Learning tracks 2019, 2020 and 2021 as well as the ANTIQUE dataset, which focuses on non-factoid open-domain question answering. As part of our data release, we include relevance judgments generated by both an open-source (Llama3.2b) and a commercial (gpt-4o) model. Our goal is to \textit{reproduce} various LLM-based relevance judgment methods to provide a comprehensive comparison. All code, data, and resources are publicly available in our GitHub Repository at this https URL. 

---
# A Human-AI Comparative Analysis of Prompt Sensitivity in LLM-Based Relevance Judgment 

**Authors**: Negar Arabzadeh, Charles L. A . Clarke  

**Link**: [PDF](https://arxiv.org/pdf/2504.12408)  

**Abstract**: Large Language Models (LLMs) are increasingly used to automate relevance judgments for information retrieval (IR) tasks, often demonstrating agreement with human labels that approaches inter-human agreement. To assess the robustness and reliability of LLM-based relevance judgments, we systematically investigate impact of prompt sensitivity on the task. We collected prompts for relevance assessment from 15 human experts and 15 LLMs across three tasks~ -- ~binary, graded, and pairwise~ -- ~yielding 90 prompts in total. After filtering out unusable prompts from three humans and three LLMs, we employed the remaining 72 prompts with three different LLMs as judges to label document/query pairs from two TREC Deep Learning Datasets (2020 and 2021). We compare LLM-generated labels with TREC official human labels using Cohen's $\kappa$ and pairwise agreement measures. In addition to investigating the impact of prompt variations on agreement with human labels, we compare human- and LLM-generated prompts and analyze differences among different LLMs as judges. We also compare human- and LLM-generated prompts with the standard UMBRELA prompt used for relevance assessment by Bing and TREC 2024 Retrieval Augmented Generation (RAG) Track. To support future research in LLM-based evaluation, we release all data and prompts at this https URL. 

---
# Specialized text classification: an approach to classifying Open Banking transactions 

**Authors**: Duc Tuyen TA, Wajdi Ben Saad, Ji Young Oh  

**Link**: [PDF](https://arxiv.org/pdf/2504.12319)  

**Abstract**: With the introduction of the PSD2 regulation in the EU which established the Open Banking framework, a new window of opportunities has opened for banks and fintechs to explore and enrich Bank transaction descriptions with the aim of building a better understanding of customer behavior, while using this understanding to prevent fraud, reduce risks and offer more competitive and tailored services.
And although the usage of natural language processing models and techniques has seen an incredible progress in various applications and domains over the past few years, custom applications based on domain-specific text corpus remain unaddressed especially in the banking sector.
In this paper, we introduce a language-based Open Banking transaction classification system with a focus on the french market and french language text. The system encompasses data collection, labeling, preprocessing, modeling, and evaluation stages. Unlike previous studies that focus on general classification approaches, this system is specifically tailored to address the challenges posed by training a language model with a specialized text corpus (Banking data in the French context). By incorporating language-specific techniques and domain knowledge, the proposed system demonstrates enhanced performance and efficiency compared to generic approaches. 

---
# Should We Tailor the Talk? Understanding the Impact of Conversational Styles on Preference Elicitation in Conversational Recommender Systems 

**Authors**: Ivica Kostric, Krisztian Balog, Ujwal Gadiraju  

**Link**: [PDF](https://arxiv.org/pdf/2504.13095)  

**Abstract**: Conversational recommender systems (CRSs) provide users with an interactive means to express preferences and receive real-time personalized recommendations. The success of these systems is heavily influenced by the preference elicitation process. While existing research mainly focuses on what questions to ask during preference elicitation, there is a notable gap in understanding what role broader interaction patterns including tone, pacing, and level of proactiveness play in supporting users in completing a given task. This study investigates the impact of different conversational styles on preference elicitation, task performance, and user satisfaction with CRSs. We conducted a controlled experiment in the context of scientific literature recommendation, contrasting two distinct conversational styles, high involvement (fast paced, direct, and proactive with frequent prompts) and high considerateness (polite and accommodating, prioritizing clarity and user comfort) alongside a flexible experimental condition where users could switch between the two. Our results indicate that adapting conversational strategies based on user expertise and allowing flexibility between styles can enhance both user satisfaction and the effectiveness of recommendations in CRSs. Overall, our findings hold important implications for the design of future CRSs. 

---
# InstructRAG: Leveraging Retrieval-Augmented Generation on Instruction Graphs for LLM-Based Task Planning 

**Authors**: Zheng Wang, Shu Xian Teo, Jun Jie Chew, Wei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2504.13032)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled their use as agents for planning complex tasks. Existing methods typically rely on a thought-action-observation (TAO) process to enhance LLM performance, but these approaches are often constrained by the LLMs' limited knowledge of complex tasks. Retrieval-augmented generation (RAG) offers new opportunities by leveraging external databases to ground generation in retrieved information. In this paper, we identify two key challenges (enlargability and transferability) in applying RAG to task planning. We propose InstructRAG, a novel solution within a multi-agent meta-reinforcement learning framework, to address these challenges. InstructRAG includes a graph to organize past instruction paths (sequences of correct actions), an RL-Agent with Reinforcement Learning to expand graph coverage for enlargability, and an ML-Agent with Meta-Learning to improve task generalization for transferability. The two agents are trained end-to-end to optimize overall planning performance. Our experiments on four widely used task planning datasets demonstrate that InstructRAG significantly enhances performance and adapts efficiently to new tasks, achieving up to a 19.2% improvement over the best existing approach. 

---
# ConExion: Concept Extraction with Large Language Models 

**Authors**: Ebrahim Norouzi, Sven Hertling, Harald Sack  

**Link**: [PDF](https://arxiv.org/pdf/2504.12915)  

**Abstract**: In this paper, an approach for concept extraction from documents using pre-trained large language models (LLMs) is presented. Compared with conventional methods that extract keyphrases summarizing the important information discussed in a document, our approach tackles a more challenging task of extracting all present concepts related to the specific domain, not just the important ones. Through comprehensive evaluations of two widely used benchmark datasets, we demonstrate that our method improves the F1 score compared to state-of-the-art techniques. Additionally, we explore the potential of using prompts within these models for unsupervised concept extraction. The extracted concepts are intended to support domain coverage evaluation of ontologies and facilitate ontology learning, highlighting the effectiveness of LLMs in concept extraction tasks. Our source code and datasets are publicly available at this https URL. 

---
# FashionDPO:Fine-tune Fashion Outfit Generation Model using Direct Preference Optimization 

**Authors**: Mingzhe Yu, Yunshan Ma, Lei Wu, Changshuo Wang, Xue Li, Lei Meng  

**Link**: [PDF](https://arxiv.org/pdf/2504.12900)  

**Abstract**: Personalized outfit generation aims to construct a set of compatible and personalized fashion items as an outfit. Recently, generative AI models have received widespread attention, as they can generate fashion items for users to complete an incomplete outfit or create a complete outfit. However, they have limitations in terms of lacking diversity and relying on the supervised learning paradigm. Recognizing this gap, we propose a novel framework FashionDPO, which fine-tunes the fashion outfit generation model using direct preference optimization. This framework aims to provide a general fine-tuning approach to fashion generative models, refining a pre-trained fashion outfit generation model using automatically generated feedback, without the need to design a task-specific reward function. To make sure that the feedback is comprehensive and objective, we design a multi-expert feedback generation module which covers three evaluation perspectives, \ie quality, compatibility and personalization. Experiments on two established datasets, \ie iFashion and Polyvore-U, demonstrate the effectiveness of our framework in enhancing the model's ability to align with users' personalized preferences while adhering to fashion compatibility principles. Our code and model checkpoints are available at this https URL. 

---
# Large Language Model-Based Knowledge Graph System Construction for Sustainable Development Goals: An AI-Based Speculative Design Perspective 

**Authors**: Yi-De Lin, Guan-Ze Liao  

**Link**: [PDF](https://arxiv.org/pdf/2504.12309)  

**Abstract**: From 2000 to 2015, the UN's Millennium Development Goals guided global priorities. The subsequent Sustainable Development Goals (SDGs) adopted a more dynamic approach, with annual indicator updates. As 2030 nears and progress lags, innovative acceleration strategies are critical. This study develops an AI-powered knowledge graph system to analyze SDG interconnections, discover potential new goals, and visualize them online. Using official SDG texts, Elsevier's keyword dataset, and 1,127 TED Talk transcripts (2020-2023), a pilot on 269 talks from 2023 applies AI-speculative design, large language models, and retrieval-augmented generation. Key findings include: (1) Heatmap analysis reveals strong associations between Goal 10 and Goal 16, and minimal coverage of Goal 6. (2) In the knowledge graph, simulated dialogue over time reveals new central nodes, showing how richer data supports divergent thinking and goal clarity. (3) Six potential new goals are proposed, centered on equity, resilience, and technology-driven inclusion. This speculative-AI framework offers fresh insights for policymakers and lays groundwork for future multimodal and cross-system SDG applications. 

---
# Unmasking the Reality of PII Masking Models: Performance Gaps and the Call for Accountability 

**Authors**: Devansh Singh, Sundaraparipurnan Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2504.12308)  

**Abstract**: Privacy Masking is a critical concept under data privacy involving anonymization and de-anonymization of personally identifiable information (PII). Privacy masking techniques rely on Named Entity Recognition (NER) approaches under NLP support in identifying and classifying named entities in each text. NER approaches, however, have several limitations including (a) content sensitivity including ambiguous, polysemic, context dependent or domain specific content, (b) phrasing variabilities including nicknames and alias, informal expressions, alternative representations, emerging expressions, evolving naming conventions and (c) formats or syntax variations, typos, misspellings. However, there are a couple of PII datasets that have been widely used by researchers and the open-source community to train models on PII detection or masking. These datasets have been used to train models including Piiranha and Starpii, which have been downloaded over 300k and 580k times on HuggingFace. We examine the quality of the PII masking by these models given the limitations of the datasets and of the NER approaches. We curate a dataset of 17K unique, semi-synthetic sentences containing 16 types of PII by compiling information from across multiple jurisdictions including India, U.K and U.S. We generate sentences (using language models) containing these PII at five different NER detection feature dimensions - (1) Basic Entity Recognition, (2) Contextual Entity Disambiguation, (3) NER in Noisy & Real-World Data, (4) Evolving & Novel Entities Detection and (5) Cross-Lingual or multi-lingual NER) and 1 in adversarial context. We present the results and exhibit the privacy exposure caused by such model use (considering the extent of lifetime downloads of these models). We conclude by highlighting the gaps in measuring performance of the models and the need for contextual disclosure in model cards for such models. 

---
