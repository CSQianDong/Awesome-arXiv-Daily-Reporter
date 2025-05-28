# Bridging the Gap: Self-Optimized Fine-Tuning for LLM-based Recommender Systems 

**Authors**: Heng Tang, Feng Liu, Xinbo Chen, Jiawei Chen, Bohao Wang, Changwang Zhang, Jun Wang, Yuegang Sun, Bingde Hu, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20771)  

**Abstract**: Recent years have witnessed extensive exploration of Large Language Models (LLMs) on the field of Recommender Systems (RS). There are currently two commonly used strategies to enable LLMs to have recommendation capabilities: 1) The "Guidance-Only" strategy uses in-context learning to exploit and amplify the inherent semantic understanding and item recommendation capabilities of LLMs; 2) The "Tuning-Only" strategy uses supervised fine-tuning (SFT) to fine-tune LLMs with the aim of fitting them to real recommendation data. However, neither of these strategies can effectively bridge the gap between the knowledge space of LLMs and recommendation, and their performance do not meet our expectations.
To better enable LLMs to learn recommendation knowledge, we combine the advantages of the above two strategies and proposed a novel "Guidance+Tuning" method called Self-Optimized Fine-Tuning (SOFT), which adopts the idea of curriculum learning. It first employs self-distillation to construct an auxiliary easy-to-learn but meaningful dataset from a fine-tuned LLM. Then it further utilizes a self-adaptive curriculum scheduler to enable LLMs to gradually learn from simpler data (self-distilled data) to more challenging data (real RS data). Extensive experiments demonstrate that SOFT significantly enhances the recommendation accuracy (37.59\% on average) of LLM-based methods. The code is available via this https URL 

---
# LifeIR at the NTCIR-18 Lifelog-6 Task 

**Authors**: Jiahan Chen, Da Li, Keping Bi  

**Link**: [PDF](https://arxiv.org/pdf/2505.20987)  

**Abstract**: In recent years, sharing lifelogs recorded through wearable devices such as sports watches and GoPros, has gained significant popularity. Lifelogs involve various types of information, including images, videos, and GPS data, revealing users' lifestyles, dietary patterns, and physical activities. The Lifelog Semantic Access Task(LSAT) in the NTCIR-18 Lifelog-6 Challenge focuses on retrieving relevant images from a large scale of users' lifelogs based on textual queries describing an action or event. It serves users' need to find images about a scenario in the historical moments of their lifelogs. We propose a multi-stage pipeline for this task of searching images with texts, addressing various challenges in lifelog retrieval. Our pipeline includes: filtering blurred images, rewriting queries to make intents clearer, extending the candidate set based on events to include images with temporal connections, and reranking results using a multimodal large language model(MLLM) with stronger relevance judgment capabilities. The evaluation results of our submissions have shown the effectiveness of each stage and the entire pipeline. 

---
# Cold-Start Recommendation with Knowledge-Guided Retrieval-Augmented Generation 

**Authors**: Wooseong Yang, Weizhi Zhang, Yuqing Liu, Yuwei Han, Yu Wang, Junhyun Lee, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20773)  

**Abstract**: Cold-start items remain a persistent challenge in recommender systems due to their lack of historical user interactions, which collaborative models rely on. While recent zero-shot methods leverage large language models (LLMs) to address this, they often struggle with sparse metadata and hallucinated or incomplete knowledge. We propose ColdRAG, a retrieval-augmented generation approach that builds a domain-specific knowledge graph dynamically to enhance LLM-based recommendation in cold-start scenarios, without requiring task-specific fine-tuning. ColdRAG begins by converting structured item attributes into rich natural-language profiles, from which it extracts entities and relationships to construct a unified knowledge graph capturing item semantics. Given a user's interaction history, it scores edges in the graph using an LLM, retrieves candidate items with supporting evidence, and prompts the LLM to rank them. By enabling multi-hop reasoning over this graph, ColdRAG grounds recommendations in verifiable evidence, reducing hallucinations and strengthening semantic connections. Experiments on three public benchmarks demonstrate that ColdRAG surpasses existing zero-shot baselines in both Recall and NDCG. This framework offers a practical solution to cold-start recommendation by combining knowledge-graph reasoning with retrieval-augmented LLM generation. 

---
# TeroSeek: An AI-Powered Knowledge Base and Retrieval Generation Platform for Terpenoid Research 

**Authors**: Xu Kang, Siqi Jiang, Kangwei Xu, Jiahao Li, Ruibo Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20663)  

**Abstract**: Terpenoids are a crucial class of natural products that have been studied for over 150 years, but their interdisciplinary nature (spanning chemistry, pharmacology, and biology) complicates knowledge integration. To address this, the authors developed TeroSeek, a curated knowledge base (KB) built from two decades of terpenoid literature, coupled with an AI-powered question-answering chatbot and web service. Leveraging a retrieval-augmented generation (RAG) framework, TeroSeek provides structured, high-quality information and outperforms general-purpose large language models (LLMs) in terpenoid-related queries. It serves as a domain-specific expert tool for multidisciplinary research and is publicly available at this http URL. 

---
# Large Language Model-Powered Decision Support for a Metal Additive Manufacturing Knowledge Graph 

**Authors**: Muhammad Tayyab Khan, Lequn Chen, Wenhe Feng, Seung Ki Moon  

**Link**: [PDF](https://arxiv.org/pdf/2505.20308)  

**Abstract**: Metal additive manufacturing (AM) involves complex interdependencies among processes, materials, feedstock, and post-processing steps. However, the underlying relationships and domain knowledge remain fragmented across literature and static databases that often demand expert-level queries, limiting their applicability in design and planning. To address these gaps, we develop a novel and queryable knowledge graph (KG) in Neo4j, encoding 53 distinct metals and alloys across seven material families, nine AM processes, four feedstock types, and associated post-processing requirements. A large language model (LLM) interface, guided by a few-shot prompting strategy, enables natural language querying without the need for formal query syntax. The system supports a range of tasks, including compatibility checks, multi-constraint filtering, and design for AM (DfAM) guidance. User natural language queries are normalized, translated into Cypher, and executed over the KG, with results reformatted into structured responses. This work presents the first real-time, interactive system that integrates a domain-specific metal AM KG with an LLM interface, offering accessible, explainable decision support for engineers and advancing human-centric tools in manufacturing intelligence. 

---
# WikiTermBase: An AI-Augmented Term Base to Standardize Arabic Translation on Wikipedia 

**Authors**: Michel Bakni, Abbad Diraneyya, Wael Tellat  

**Link**: [PDF](https://arxiv.org/pdf/2505.20369)  

**Abstract**: Term bases are recognized as one of the most effective components of translation software in time saving and consistency. In spite of the many recent advances in natural language processing (NLP) and large language models (LLMs), major translation platforms have yet to take advantage of these tools to improve their term bases and support scalable content for underrepresented languages, which often struggle with localizing technical terminology. Language academies in the Arab World, for example, have struggled since the 1940s to unify the way new scientific terms enter the Arabic language at scale. This abstract introduces an open source tool, WikiTermBase, with a systematic approach for building a lexicographical database with over 900K terms, which were collected and mapped from a multitude of sources on a semantic and morphological basis. The tool was successfully implemented on Arabic Wikipedia to standardize translated English and French terms. 

---
# A Lightweight Multi-Expert Generative Language Model System for Engineering Information and Knowledge Extraction 

**Authors**: Bogdan Bogachov, Yaoyao Fiona Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.21109)  

**Abstract**: Despite recent advancements in domain adaptation techniques for large language models, these methods remain computationally intensive, and the resulting models can still exhibit hallucination issues. Most existing adaptation methods do not prioritize reducing the computational resources required for fine-tuning and inference of language models. Hallucination issues have gradually decreased with each new model release. However, they remain prevalent in engineering contexts, where generating well-structured text with minimal errors and inconsistencies is critical. This work introduces a novel approach called the Small Language Graph (SLG), which is a lightweight adaptation solution designed to address the two key challenges outlined above. The system is structured in the form of a graph, where each node represents a lightweight expert - a small language model fine-tuned on specific and concise texts. The results of this study have shown that SLG was able to surpass conventional fine-tuning methods on the Exact Match metric by 3 times. Additionally, the fine-tuning process was 1.7 times faster compared to that of a larger stand-alone language model. These findings introduce a potential for small to medium-sized engineering companies to confidently use generative AI technologies, such as LLMs, without the necessity to invest in expensive computational resources. Also, the graph architecture and the small size of expert nodes offer a possible opportunity for distributed AI systems, thus potentially diverting the global need for expensive centralized compute clusters. 

---
# Beyond Prompt Engineering: Robust Behavior Control in LLMs via Steering Target Atoms 

**Authors**: Mengru Wang, Ziwen Xu, Shengyu Mao, Shumin Deng, Zhaopeng Tu, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20322)  

**Abstract**: Precise control over language model generation is vital for ensuring both safety and reliability. Although prompt engineering and steering are commonly used to intervene in model behaviors, the vast number of parameters in models often results in highly intertwined internal representations. This interdependency can limit control precision and sometimes lead to unintended side effects. Recent research has explored the use of sparse autoencoders (SAE) to disentangle knowledge in high-dimensional spaces for steering. However, these applications have been limited to toy tasks owing to the nontrivial issue of locating atomic knowledge components. In this paper, we propose Steering Target Atoms (STA), a novel method that isolates and manipulates disentangled knowledge components to enhance safety. Comprehensive experiments demonstrate the effectiveness of our approach. Further analysis reveals that steering exhibits superior robustness and flexibility, particularly in adversarial scenarios. We also apply the steering strategy to the large reasoning model, confirming its effectiveness in precise reasoning control. 

---
# What LLMs Miss in Recommendations: Bridging the Gap with Retrieval-Augmented Collaborative Signals 

**Authors**: Shahrooz Pouryousef  

**Link**: [PDF](https://arxiv.org/pdf/2505.20730)  

**Abstract**: User-item interactions contain rich collaborative signals that form the backbone of many successful recommender systems. While recent work has explored the use of large language models (LLMs) for recommendation, it remains unclear whether LLMs can effectively reason over this type of collaborative information. In this paper, we conduct a systematic comparison between LLMs and classical matrix factorization (MF) models to assess LLMs' ability to leverage user-item interaction data. We further introduce a simple retrieval-augmented generation (RAG) method that enhances LLMs by grounding their predictions in structured interaction data. Our experiments reveal that current LLMs often fall short in capturing collaborative patterns inherent to MF models, but that our RAG-based approach substantially improves recommendation quality-highlighting a promising direction for future LLM-based recommenders. 

---
# Do LLMs Need to Think in One Language? Correlation between Latent Language and Task Performance 

**Authors**: Shintaro Ozaki, Tatsuya Hiraoka, Hiroto Otake, Hiroki Ouchi, Masaru Isonuma, Benjamin Heinzerling, Kentaro Inui, Taro Watanabe, Yusuke Miyao, Yohei Oseki, Yu Takagi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21458)  

**Abstract**: Large Language Models (LLMs) are known to process information using a proficient internal language consistently, referred to as latent language, which may differ from the input or output languages. However, how the discrepancy between the latent language and the input and output language affects downstream task performance remains largely unexplored. While many studies research the latent language of LLMs, few address its importance in influencing task performance. In our study, we hypothesize that thinking in latent language consistently enhances downstream task performance. To validate this, our work varies the input prompt languages across multiple downstream tasks and analyzes the correlation between consistency in latent language and task performance. We create datasets consisting of questions from diverse domains such as translation and geo-culture, which are influenced by the choice of latent language. Experimental results across multiple LLMs on translation and geo-culture tasks, which are sensitive to the choice of language, indicate that maintaining consistency in latent language is not always necessary for optimal downstream task performance. This is because these models adapt their internal representations near the final layers to match the target language, reducing the impact of consistency on overall performance. 

---
# How does Alignment Enhance LLMs' Multilingual Capabilities? A Language Neurons Perspective 

**Authors**: Shimao Zhang, Zhejian Lai, Xiang Liu, Shuaijie She, Xiao Liu, Yeyun Gong, Shujian Huang, Jiajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21505)  

**Abstract**: Multilingual Alignment is an effective and representative paradigm to enhance LLMs' multilingual capabilities, which transfers the capabilities from the high-resource languages to the low-resource languages. Meanwhile, some researches on language-specific neurons reveal that there are language-specific neurons that are selectively activated in LLMs when processing different languages. This provides a new perspective to analyze and understand LLMs' mechanisms more specifically in multilingual scenarios. In this work, we propose a new finer-grained neuron identification algorithm, which detects language neurons~(including language-specific neurons and language-related neurons) and language-agnostic neurons. Furthermore, based on the distributional characteristics of different types of neurons, we divide the LLMs' internal process for multilingual inference into four parts: (1) multilingual understanding, (2) shared semantic space reasoning, (3) multilingual output space transformation, and (4) vocabulary space outputting. Additionally, we systematically analyze the models before and after alignment with a focus on different types of neurons. We also analyze the phenomenon of ''Spontaneous Multilingual Alignment''. Overall, our work conducts a comprehensive investigation based on different types of neurons, providing empirical results and valuable insights for better understanding multilingual alignment and multilingual capabilities of LLMs. 

---
# RefTool: Enhancing Model Reasoning with Reference-Guided Tool Creation 

**Authors**: Xiao Liu, Da Yin, Zirui Wu, Yansong Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21413)  

**Abstract**: Tools enhance the reasoning capabilities of large language models (LLMs) in complex problem-solving tasks, but not all tasks have available tools. In the absence of predefined tools, prior works have explored instructing LLMs to generate tools on their own. However, such approaches rely heavily on the models' internal knowledge and would fail in domains beyond the LLMs' knowledge scope. To address this limitation, we propose RefTool, a reference-guided framework for automatic tool creation that leverages structured external materials such as textbooks. RefTool consists of two modules: (1) tool creation, where LLMs generate executable tools from reference content, validate them using illustrative examples, and organize them hierarchically into a toolbox; and (2) tool utilization, where LLMs navigate the toolbox structure to select and apply the appropriate tools to solve problems. Experiments on causality, physics, and chemistry benchmarks demonstrate that RefTool outperforms existing tool-creation and domain-specific reasoning methods by 11.3% on average accuracy, while being cost-efficient and broadly generalizable. Analyses reveal that grounding tool creation in references produces accurate and faithful tools, and that the hierarchical structure facilitates effective tool selection. RefTool enables LLMs to overcome knowledge limitations, demonstrating the value of grounding tool creation in external references for enhanced and generalizable reasoning. 

---
# Silence is Not Consensus: Disrupting Agreement Bias in Multi-Agent LLMs via Catfish Agent for Clinical Decision Making 

**Authors**: Yihan Wang, Qiao Yan, Zhenghao Xing, Lihao Liu, Junjun He, Chi-Wing Fu, Xiaowei Hu, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21503)  

**Abstract**: Large language models (LLMs) have demonstrated strong potential in clinical question answering, with recent multi-agent frameworks further improving diagnostic accuracy via collaborative reasoning. However, we identify a recurring issue of Silent Agreement, where agents prematurely converge on diagnoses without sufficient critical analysis, particularly in complex or ambiguous cases. We present a new concept called Catfish Agent, a role-specialized LLM designed to inject structured dissent and counter silent agreement. Inspired by the ``catfish effect'' in organizational psychology, the Catfish Agent is designed to challenge emerging consensus to stimulate deeper reasoning. We formulate two mechanisms to encourage effective and context-aware interventions: (i) a complexity-aware intervention that modulates agent engagement based on case difficulty, and (ii) a tone-calibrated intervention articulated to balance critique and collaboration. Evaluations on nine medical Q&A and three medical VQA benchmarks show that our approach consistently outperforms both single- and multi-agent LLMs frameworks, including leading commercial models such as GPT-4o and DeepSeek-R1. 

---
# Are Language Models Consequentialist or Deontological Moral Reasoners? 

**Authors**: Keenan Samway, Max Kleiman-Weiner, David Guzman Piedrahita, Rada Mihalcea, Bernhard Schölkopf, Zhijing Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.21479)  

**Abstract**: As AI systems increasingly navigate applications in healthcare, law, and governance, understanding how they handle ethically complex scenarios becomes critical. Previous work has mainly examined the moral judgments in large language models (LLMs), rather than their underlying moral reasoning process. In contrast, we focus on a large-scale analysis of the moral reasoning traces provided by LLMs. Furthermore, unlike prior work that attempted to draw inferences from only a handful of moral dilemmas, our study leverages over 600 distinct trolley problems as probes for revealing the reasoning patterns that emerge within different LLMs. We introduce and test a taxonomy of moral rationales to systematically classify reasoning traces according to two main normative ethical theories: consequentialism and deontology. Our analysis reveals that LLM chains-of-thought tend to favor deontological principles based on moral obligations, while post-hoc explanations shift notably toward consequentialist rationales that emphasize utility. Our framework provides a foundation for understanding how LLMs process and articulate ethical considerations, an important step toward safe and interpretable deployment of LLMs in high-stakes decision-making environments. Our code is available at this https URL . 

---
# RelationalFactQA: A Benchmark for Evaluating Tabular Fact Retrieval from Large Language Models 

**Authors**: Dario Satriani, Enzo Veltri, Donatello Santoro, Paolo Papotti  

**Link**: [PDF](https://arxiv.org/pdf/2505.21409)  

**Abstract**: Factuality in Large Language Models (LLMs) is a persistent challenge. Current benchmarks often assess short factual answers, overlooking the critical ability to generate structured, multi-record tabular outputs from parametric knowledge. We demonstrate that this relational fact retrieval is substantially more difficult than isolated point-wise queries, even when individual facts are known to the model, exposing distinct failure modes sensitive to output dimensionality (e.g., number of attributes or records). To systematically evaluate this under-explored capability, we introduce RelationalFactQA, a new benchmark featuring diverse natural language questions (paired with SQL) and gold-standard tabular answers, specifically designed to assess knowledge retrieval in a structured format. RelationalFactQA enables analysis across varying query complexities, output sizes, and data characteristics. Our experiments reveal that even state-of-the-art LLMs struggle significantly, not exceeding 25% factual accuracy in generating relational outputs, with performance notably degrading as output dimensionality increases. These findings underscore critical limitations in current LLMs' ability to synthesize structured factual knowledge and establish RelationalFactQA as a crucial resource for measuring future progress in LLM factuality. 

---
# AutoJudger: An Agent-Driven Framework for Efficient Benchmarking of MLLMs 

**Authors**: Xuanwen Ding, Chengjun Pan, Zejun Li, Jiwen Zhang, Siyuan Wang, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.21389)  

**Abstract**: Evaluating multimodal large language models (MLLMs) is increasingly expensive, as the growing size and cross-modality complexity of benchmarks demand significant scoring efforts. To tackle with this difficulty, we introduce AutoJudger, an agent-driven framework for efficient and adaptive benchmarking of MLLMs that tackles this escalating cost. AutoJudger employs the Item Response Theory (IRT) to estimate the question difficulty and an autonomous evaluation agent to dynamically select the most informative test questions based on the model's real-time performance. Specifically, AutoJudger incorporates two pivotal components: a semantic-aware retrieval mechanism to ensure that selected questions cover diverse and challenging scenarios across both vision and language modalities, and a dynamic memory that maintains contextual statistics of previously evaluated questions to guide coherent and globally informed question selection throughout the evaluation process. Extensive experiments on four representative multimodal benchmarks demonstrate that our adaptive framework dramatically reduces evaluation expenses, i.e. AutoJudger uses only 4% of the data to achieve over 90% ranking accuracy with the full benchmark evaluation on MMT-Bench. 

---
# Factual Self-Awareness in Language Models: Representation, Robustness, and Scaling 

**Authors**: Hovhannes Tamoyan, Subhabrata Dutta, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2505.21399)  

**Abstract**: Factual incorrectness in generated content is one of the primary concerns in ubiquitous deployment of large language models (LLMs). Prior findings suggest LLMs can (sometimes) detect factual incorrectness in their generated content (i.e., fact-checking post-generation). In this work, we provide evidence supporting the presence of LLMs' internal compass that dictate the correctness of factual recall at the time of generation. We demonstrate that for a given subject entity and a relation, LLMs internally encode linear features in the Transformer's residual stream that dictate whether it will be able to recall the correct attribute (that forms a valid entity-relation-attribute triplet). This self-awareness signal is robust to minor formatting variations. We investigate the effects of context perturbation via different example selection strategies. Scaling experiments across model sizes and training dynamics highlight that self-awareness emerges rapidly during training and peaks in intermediate layers. These findings uncover intrinsic self-monitoring capabilities within LLMs, contributing to their interpretability and reliability. 

---
# DecisionFlow: Advancing Large Language Model as Principled Decision Maker 

**Authors**: Xiusi Chen, Shanyong Wang, Cheng Qian, Hongru Wang, Peixuan Han, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2505.21397)  

**Abstract**: In high-stakes domains such as healthcare and finance, effective decision-making demands not just accurate outcomes but transparent and explainable reasoning. However, current language models often lack the structured deliberation needed for such tasks, instead generating decisions and justifications in a disconnected, post-hoc manner. To address this, we propose DecisionFlow, a novel decision modeling framework that guides models to reason over structured representations of actions, attributes, and constraints. Rather than predicting answers directly from prompts, DecisionFlow builds a semantically grounded decision space and infers a latent utility function to evaluate trade-offs in a transparent, utility-driven manner. This process produces decisions tightly coupled with interpretable rationales reflecting the model's reasoning. Empirical results on two high-stakes benchmarks show that DecisionFlow not only achieves up to 30% accuracy gains over strong prompting baselines but also enhances alignment in outcomes. Our work is a critical step toward integrating symbolic reasoning with LLMs, enabling more accountable, explainable, and reliable LLM decision support systems. We release the data and code at this https URL. 

---
# Improving Research Idea Generation Through Data: An Empirical Investigation in Social Science 

**Authors**: Xiao Liu, Xinyi Dong, Xinyang Gao, Yansong Feng, Xun Pang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21396)  

**Abstract**: Recent advancements in large language models (LLMs) have shown promise in generating novel research ideas. However, these ideas often face challenges related to feasibility and expected effectiveness. This paper explores how augmenting LLMs with relevant data during the idea generation process can enhance the quality of generated ideas. We introduce two ways of incorporating data: (1) providing metadata during the idea generation stage to guide LLMs toward feasible directions, and (2) adding automatic validation during the idea selection stage to assess the empirical plausibility of hypotheses within ideas. We conduct experiments in the social science domain, specifically with climate negotiation topics, and find that metadata improves the feasibility of generated ideas by 20%, while automatic validation improves the overall quality of selected ideas by 7%. A human study shows that LLM-generated ideas, along with their related data and validation processes, inspire researchers to propose research ideas with higher quality. Our work highlights the potential of data-driven research idea generation, and underscores the practical utility of LLM-assisted ideation in real-world academic settings. 

---
# Leveraging large language models and traditional machine learning ensembles for ADHD detection from narrative transcripts 

**Authors**: Yuxin Zhu, Yuting Guo, Noah Marchuck, Abeed Sarker, Yun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21324)  

**Abstract**: Despite rapid advances in large language models (LLMs), their integration with traditional supervised machine learning (ML) techniques that have proven applicability to medical data remains underexplored. This is particularly true for psychiatric applications, where narrative data often exhibit nuanced linguistic and contextual complexity, and can benefit from the combination of multiple models with differing characteristics. In this study, we introduce an ensemble framework for automatically classifying Attention-Deficit/Hyperactivity Disorder (ADHD) diagnosis (binary) using narrative transcripts. Our approach integrates three complementary models: LLaMA3, an open-source LLM that captures long-range semantic structure; RoBERTa, a pre-trained transformer model fine-tuned on labeled clinical narratives; and a Support Vector Machine (SVM) classifier trained using TF-IDF-based lexical features. These models are aggregated through a majority voting mechanism to enhance predictive robustness. The dataset includes 441 instances, including 352 for training and 89 for validation. Empirical results show that the ensemble outperforms individual models, achieving an F$_1$ score of 0.71 (95\% CI: [0.60-0.80]). Compared to the best-performing individual model (SVM), the ensemble improved recall while maintaining competitive precision. This indicates the strong sensitivity of the ensemble in identifying ADHD-related linguistic cues. These findings demonstrate the promise of hybrid architectures that leverage the semantic richness of LLMs alongside the interpretability and pattern recognition capabilities of traditional supervised ML, offering a new direction for robust and generalizable psychiatric text classification. 

---
# ReSCORE: Label-free Iterative Retriever Training for Multi-hop Question Answering with Relevance-Consistency Supervision 

**Authors**: Dosung Lee, Wonjun Oh, Boyoung Kim, Minyoung Kim, Joonsuk Park, Paul Hongsuck Seo  

**Link**: [PDF](https://arxiv.org/pdf/2505.21250)  

**Abstract**: Multi-hop question answering (MHQA) involves reasoning across multiple documents to answer complex questions. Dense retrievers typically outperform sparse methods like BM25 by leveraging semantic embeddings; however, they require labeled query-document pairs for fine-tuning. This poses a significant challenge in MHQA due to the high variability of queries (reformulated) questions throughout the reasoning steps. To overcome this limitation, we introduce Retriever Supervision with Consistency and Relevance (ReSCORE), a novel method for training dense retrievers for MHQA without labeled documents. ReSCORE leverages large language models to capture each documents relevance to the question and consistency with the correct answer and use them to train a retriever within an iterative question-answering framework. Experiments on three MHQA benchmarks demonstrate the effectiveness of ReSCORE, with significant improvements in retrieval, and in turn, the state-of-the-art MHQA performance. Our implementation is available at: this https URL. 

---
# Scaling External Knowledge Input Beyond Context Windows of LLMs via Multi-Agent Collaboration 

**Authors**: Zijun Liu, Zhennan Wan, Peng Li, Ming Yan, Ji Zhang, Fei Huang, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21471)  

**Abstract**: With the rapid advancement of post-training techniques for reasoning and information seeking, large language models (LLMs) can incorporate a large quantity of retrieved knowledge to solve complex tasks. However, the limited context window of LLMs obstructs scaling the amount of external knowledge input, prohibiting further improvement, especially for tasks requiring significant amount of external knowledge. Existing context window extension methods inevitably cause information loss. LLM-based multi-agent methods emerge as a new paradigm to handle massive input in a distributional manner, where we identify two core bottlenecks in existing knowledge synchronization and reasoning processes. In this work, we develop a multi-agent framework, $\textbf{ExtAgents}$, to overcome the bottlenecks and enable better scalability in inference-time knowledge integration without longer-context training. Benchmarked with our enhanced multi-hop question answering test, $\textbf{$\boldsymbol{\infty}$Bench+}$, and other public test sets including long survey generation, ExtAgents significantly enhances the performance over existing non-training methods with the same amount of external knowledge input, regardless of whether it falls $\textit{within or exceeds the context window}$. Moreover, the method maintains high efficiency due to high parallelism. Further study in the coordination of LLM agents on increasing external knowledge input could benefit real-world applications. 

---
# Pretrained LLMs Learn Multiple Types of Uncertainty 

**Authors**: Roi Cohen, Omri Fahn, Gerard de Melo  

**Link**: [PDF](https://arxiv.org/pdf/2505.21218)  

**Abstract**: Large Language Models are known to capture real-world knowledge, allowing them to excel in many downstream tasks. Despite recent advances, these models are still prone to what are commonly known as hallucinations, causing them to emit unwanted and factually incorrect text. In this work, we study how well LLMs capture uncertainty, without explicitly being trained for that. We show that, if considering uncertainty as a linear concept in the model's latent space, it might indeed be captured, even after only pretraining. We further show that, though unintuitive, LLMs appear to capture several different types of uncertainty, each of which can be useful to predict the correctness for a specific task or benchmark. Furthermore, we provide in-depth results such as demonstrating a correlation between our correction prediction and the model's ability to abstain from misinformation using words, and the lack of impact of model scaling for capturing uncertainty. Finally, we claim that unifying the uncertainty types as a single one using instruction-tuning or [IDK]-token tuning is helpful for the model in terms of correctness prediction. 

---
# PEDANTIC: A Dataset for the Automatic Examination of Definiteness in Patent Claims 

**Authors**: Valentin Knappich, Annemarie Friedrich, Anna Hätty, Simon Razniewski  

**Link**: [PDF](https://arxiv.org/pdf/2505.21342)  

**Abstract**: Patent claims define the scope of protection for an invention. If there are ambiguities in a claim, it is rejected by the patent office. In the US, this is referred to as indefiniteness (35 U.S.C § 112(b)) and is among the most frequent reasons for patent application rejection. The development of automatic methods for patent definiteness examination has the potential to make patent drafting and examination more efficient, but no annotated dataset has been published to date.
We introduce PEDANTIC (\underline{P}at\underline{e}nt \underline{D}efiniteness Ex\underline{a}mi\underline{n}a\underline{ti}on \underline{C}orpus), a novel dataset of 14k US patent claims from patent applications relating to Natural Language Processing (NLP), annotated with reasons for indefiniteness. We construct PEDANTIC using a fully automatic pipeline that retrieves office action documents from the USPTO and uses Large Language Models (LLMs) to extract the reasons for indefiniteness. A human validation study confirms the pipeline's accuracy in generating high-quality annotations. To gain insight beyond binary classification metrics, we implement an LLM-as-Judge evaluation that compares the free-form reasoning of every model-cited reason with every examiner-cited reason. We show that LLM agents based on Qwen 2.5 32B and 72B struggle to outperform logistic regression baselines on definiteness prediction, even though they often correctly identify the underlying reasons. PEDANTIC provides a valuable resource for patent AI researchers, enabling the development of advanced examination models. We will publicly release the dataset and code. 

---
# Leveraging Large Language Models for Bengali Math Word Problem Solving with Chain of Thought Reasoning 

**Authors**: Bidyarthi Paul, Jalisha Jashim Era, Mirazur Rahman Zim, Tahmid Sattar Aothoi, Faisal Muhammad Shah  

**Link**: [PDF](https://arxiv.org/pdf/2505.21354)  

**Abstract**: Solving Bengali Math Word Problems (MWPs) remains a major challenge in natural language processing (NLP) due to the language's low-resource status and the multi-step reasoning required. Existing models struggle with complex Bengali MWPs, largely because no human-annotated Bengali dataset has previously addressed this task. This gap has limited progress in Bengali mathematical reasoning. To address this, we created SOMADHAN, a dataset of 8792 complex Bengali MWPs with manually written, step-by-step solutions. We designed this dataset to support reasoning-focused evaluation and model development in a linguistically underrepresented context. Using SOMADHAN, we evaluated a range of large language models (LLMs) - including GPT-4o, GPT-3.5 Turbo, LLaMA series models, Deepseek, and Qwen - through both zero-shot and few-shot prompting with and without Chain of Thought (CoT) reasoning. CoT prompting consistently improved performance over standard prompting, especially in tasks requiring multi-step logic. LLaMA-3.3 70B achieved the highest accuracy of 88% with few-shot CoT prompting. We also applied Low-Rank Adaptation (LoRA) to fine-tune models efficiently, enabling them to adapt to Bengali MWPs with minimal computational cost. Our work fills a critical gap in Bengali NLP by providing a high-quality reasoning dataset and a scalable framework for solving complex MWPs. We aim to advance equitable research in low-resource languages and enhance reasoning capabilities in educational and language technologies. 

---
# Unveiling Instruction-Specific Neurons & Experts: An Analytical Framework for LLM's Instruction-Following Capabilities 

**Authors**: Junyan Zhang, Yubo Gao, Yibo Yan, Jungang Li, Zhaorui Hou, Sicheng Tao, Shuliang Liu, Song Dai, Yonghua Hei, Junzhuo Li, Xuming Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21191)  

**Abstract**: The finetuning of Large Language Models (LLMs) has significantly advanced their instruction-following capabilities, yet the underlying computational mechanisms driving these improvements remain poorly understood. This study systematically examines how fine-tuning reconfigures LLM computations by isolating and analyzing instruction-specific sparse components, i.e., neurons in dense models and both neurons and experts in Mixture-of-Experts (MoE) architectures. In particular, we introduce HexaInst, a carefully curated and balanced instructional dataset spanning six distinct categories, and propose SPARCOM, a novel analytical framework comprising three key contributions: (1) a method for identifying these sparse components, (2) an evaluation of their functional generality and uniqueness, and (3) a systematic comparison of their alterations. Through experiments, we demonstrate functional generality, uniqueness, and the critical role of these components in instruction execution. By elucidating the relationship between fine-tuning-induced adaptations and sparse computational substrates, this work provides deeper insights into how LLMs internalize instruction-following behavior for the trustworthy LLM community. 

---
# Evaluating LLM Adaptation to Sociodemographic Factors: User Profile vs. Dialogue History 

**Authors**: Qishuai Zhong, Zongmin Li, Siqi Fan, Aixin Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.21362)  

**Abstract**: Effective engagement by large language models (LLMs) requires adapting responses to users' sociodemographic characteristics, such as age, occupation, and education level. While many real-world applications leverage dialogue history for contextualization, existing evaluations of LLMs' behavioral adaptation often focus on single-turn prompts. In this paper, we propose a framework to evaluate LLM adaptation when attributes are introduced either (1) explicitly via user profiles in the prompt or (2) implicitly through multi-turn dialogue history. We assess the consistency of model behavior across these modalities. Using a multi-agent pipeline, we construct a synthetic dataset pairing dialogue histories with distinct user profiles and employ questions from the Value Survey Module (VSM 2013) (Hofstede and Hofstede, 2016) to probe value expression. Our findings indicate that most models adjust their expressed values in response to demographic changes, particularly in age and education level, but consistency varies. Models with stronger reasoning capabilities demonstrate greater alignment, indicating the importance of reasoning in robust sociodemographic adaptation. 

---
# Exploring the Latent Capacity of LLMs for One-Step Text Generation 

**Authors**: Gleb Mezentsev, Ivan Oseledets  

**Link**: [PDF](https://arxiv.org/pdf/2505.21189)  

**Abstract**: A recent study showed that large language models (LLMs) can reconstruct surprisingly long texts - up to thousands of tokens - via autoregressive generation from just one specially trained input embedding. In this work, we explore whether such reconstruction is possible without autoregression. We show that frozen LLMs can generate hundreds of accurate tokens in just one forward pass, when provided with only two learned embeddings. This reveals a surprising and underexplored capability of LLMs - multi-token generation without iterative decoding. We investigate the behaviour of these embeddings and provide insight into the type of information they encode. We also empirically show that although these representations are not unique for a given text, they form connected and local regions in embedding space - a property that suggests the potential of learning a dedicated encoder into that space. 

---
# Walk Before You Run! Concise LLM Reasoning via Reinforcement Learning 

**Authors**: Mingyang Song, Mao Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21178)  

**Abstract**: As test-time scaling becomes a pivotal research frontier in Large Language Models (LLMs) development, contemporary and advanced post-training methodologies increasingly focus on extending the generation length of long Chain-of-Thought (CoT) responses to enhance reasoning capabilities toward DeepSeek R1-like performance. However, recent studies reveal a persistent overthinking phenomenon in state-of-the-art reasoning models, manifesting as excessive redundancy or repetitive thinking patterns in long CoT responses. To address this issue, in this paper, we propose a simple yet effective two-stage reinforcement learning framework for achieving concise reasoning in LLMs, named ConciseR. Specifically, the first stage, using more training steps, aims to incentivize the model's reasoning capabilities via Group Relative Policy Optimization with clip-higher and dynamic sampling components (GRPO++), and the second stage, using fewer training steps, explicitly enforces conciseness and improves efficiency via Length-aware Group Relative Policy Optimization (L-GRPO). Significantly, ConciseR only optimizes response length once all rollouts of a sample are correct, following the "walk before you run" principle. Extensive experimental results demonstrate that our ConciseR model, which generates more concise CoT reasoning responses, outperforms recent state-of-the-art reasoning models with zero RL paradigm across AIME 2024, MATH-500, AMC 2023, Minerva, and Olympiad benchmarks. 

---
# Leveraging LLM and Self-Supervised Training Models for Speech Recognition in Chinese Dialects: A Comparative Analysis 

**Authors**: Tianyi Xu, Hongjie Chen, Wang Qing, Lv Hang, Jian Kang, Li Jie, Zhennan Lin, Yongxiang Li, Xie Lei  

**Link**: [PDF](https://arxiv.org/pdf/2505.21138)  

**Abstract**: Large-scale training corpora have significantly improved the performance of ASR models. Unfortunately, due to the relative scarcity of data, Chinese accents and dialects remain a challenge for most ASR models. Recent advancements in self-supervised learning have shown that self-supervised pre- training, combined with large language models (LLM), can effectively enhance ASR performance in low-resource scenarios. We aim to investigate the effectiveness of this paradigm for Chinese dialects. Specifically, we pre-train a Data2vec2 model on 300,000 hours of unlabeled dialect and accented speech data and do alignment training on a supervised dataset of 40,000 hours. Then, we systematically examine the impact of various projectors and LLMs on Mandarin, dialect, and accented speech recognition performance under this paradigm. Our method achieved SOTA results on multiple dialect datasets, including Kespeech. We will open-source our work to promote reproducible research 

---
# rStar-Coder: Scaling Competitive Code Reasoning with a Large-Scale Verified Dataset 

**Authors**: Yifei Liu, Li Lyna Zhang, Yi Zhu, Bingcheng Dong, Xudong Zhou, Ning Shang, Fan Yang, Mao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21297)  

**Abstract**: Advancing code reasoning in large language models (LLMs) is fundamentally limited by the scarcity of high-difficulty datasets, especially those with verifiable input-output test cases necessary for rigorous solution validation at scale. We introduce rStar-Coder, which significantly improves LLM code reasoning capabilities by constructing a large-scale, verified dataset of 418K competition-level code problems, 580K long-reasoning solutions along with rich test cases of varying difficulty. This is achieved through three core contributions: (1) we curate competitive programming code problems and oracle solutions to synthesize new, solvable problems; (2) we introduce a reliable input-output test case synthesis pipeline that decouples the generation into a three-step input generation method and a mutual verification mechanism for effective output labeling; (3) we augment problems with high-quality, test-case-verified long-reasoning solutions. Extensive experiments on Qwen models (1.5B-14B) across various code reasoning benchmarks demonstrate the superiority of rStar-Coder dataset, achieving leading performance comparable to frontier reasoning LLMs with much smaller model sizes. On LiveCodeBench, rStar-Coder improves Qwen2.5-7B from 17.4% to an impressive 57.3%, and Qwen2.5-14B from 23.3% to 62.5%, surpassing o3-mini (low) by3.1%. On the more challenging USA Computing Olympiad, our 7B model achieves an average pass@1 accuracy of 16.15%, outperforming the frontier-level QWQ-32B. Code and the dataset will be released at this https URL. 

---
# Will It Still Be True Tomorrow? Multilingual Evergreen Question Classification to Improve Trustworthy QA 

**Authors**: Sergey Pletenev, Maria Marina, Nikolay Ivanov, Daria Galimzianova, Nikita Krayko, Mikhail Salnikov, Vasily Konovalov, Alexander Panchenko, Viktor Moskvoretskii  

**Link**: [PDF](https://arxiv.org/pdf/2505.21115)  

**Abstract**: Large Language Models (LLMs) often hallucinate in question answering (QA) tasks. A key yet underexplored factor contributing to this is the temporality of questions -- whether they are evergreen (answers remain stable over time) or mutable (answers change). In this work, we introduce EverGreenQA, the first multilingual QA dataset with evergreen labels, supporting both evaluation and training. Using EverGreenQA, we benchmark 12 modern LLMs to assess whether they encode question temporality explicitly (via verbalized judgments) or implicitly (via uncertainty signals). We also train EG-E5, a lightweight multilingual classifier that achieves SoTA performance on this task. Finally, we demonstrate the practical utility of evergreen classification across three applications: improving self-knowledge estimation, filtering QA datasets, and explaining GPT-4o retrieval behavior. 

---
# Thinker: Learning to Think Fast and Slow 

**Authors**: Stephen Chung, Wenyu Du, Jie Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21097)  

**Abstract**: Recent studies show that the reasoning capabilities of Large Language Models (LLMs) can be improved by applying Reinforcement Learning (RL) to question-answering (QA) tasks in areas such as math and coding. With a long context length, LLMs may learn to perform search, as indicated by the self-correction behavior observed in DeepSeek R1. However, this search behavior is often imprecise and lacks confidence, resulting in long, redundant responses and highlighting deficiencies in intuition and verification. Inspired by the Dual Process Theory in psychology, we introduce a simple modification to the QA task that includes four stages: Fast Thinking, where the LLM must answer within a strict token budget; Verification, where the model evaluates its initial response; Slow Thinking, where it refines the initial response with more deliberation; and Summarization, where it distills the refinement from the previous stage into precise steps. Our proposed task improves average accuracy from 24.9% to 27.9% for Qwen2.5-1.5B, and from 45.9% to 49.8% for DeepSeek-R1-Qwen-1.5B. Notably, for Qwen2.5-1.5B, the Fast Thinking mode alone achieves 26.8% accuracy using fewer than 1000 tokens, demonstrating substantial inference efficiency gains. These findings suggest that intuition and deliberative reasoning are distinct, complementary systems benefiting from targeted training. 

---
# LLMs Think, But Not In Your Flow: Reasoning-Level Personalization for Black-Box Large Language Models 

**Authors**: Jieyong Kim, Tongyoung Kim, Soonjin Yoon, Jaehyung Kim, Dongha Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.21082)  

**Abstract**: Large language models (LLMs) have recently achieved impressive performance across a wide range of natural language tasks and are now widely used in real-world applications. Among them, black-box LLMs--served via APIs without access to model internals--are especially dominant due to their scalability and ease of deployment. Despite their strong capabilities, these models typically produce generalized responses that overlook personal preferences and reasoning styles. This has led to growing interest in black-box LLM personalization, which aims to tailor model outputs to user-specific context without modifying model parameters. However, existing approaches primarily focus on response-level personalization, attempting to match final outputs without modeling personal thought process. To address this limitation, we propose RPM, a framework for reasoning-level personalization that aligns the model's reasoning process with a user's personalized logic. RPM first constructs statistical user-specific factors by extracting and grouping response-influential features from user history. It then builds personalized reasoning paths that reflect how these factors are used in context. In the inference stage, RPM retrieves reasoning-aligned examples for new queries via feature-level similarity and performs inference conditioned on the structured factors and retrieved reasoning paths, enabling the model to follow user-specific reasoning trajectories. This reasoning-level personalization enhances both predictive accuracy and interpretability by grounding model outputs in user-specific logic through structured information. Extensive experiments across diverse tasks show that RPM consistently outperforms response-level personalization methods, demonstrating the effectiveness of reasoning-level personalization in black-box LLMs. 

---
# BLUCK: A Benchmark Dataset for Bengali Linguistic Understanding and Cultural Knowledge 

**Authors**: Daeen Kabir, Minhajur Rahman Chowdhury Mahim, Sheikh Shafayat, Adnan Sadik, Arian Ahmed, Eunsu Kim, Alice Oh  

**Link**: [PDF](https://arxiv.org/pdf/2505.21092)  

**Abstract**: In this work, we introduce BLUCK, a new dataset designed to measure the performance of Large Language Models (LLMs) in Bengali linguistic understanding and cultural knowledge. Our dataset comprises 2366 multiple-choice questions (MCQs) carefully curated from compiled collections of several college and job level examinations and spans 23 categories covering knowledge on Bangladesh's culture and history and Bengali linguistics. We benchmarked BLUCK using 6 proprietary and 3 open-source LLMs - including GPT-4o, Claude-3.5-Sonnet, Gemini-1.5-Pro, Llama-3.3-70B-Instruct, and DeepSeekV3. Our results show that while these models perform reasonably well overall, they, however, struggles in some areas of Bengali phonetics. Although current LLMs' performance on Bengali cultural and linguistic contexts is still not comparable to that of mainstream languages like English, our results indicate Bengali's status as a mid-resource language. Importantly, BLUCK is also the first MCQ-based evaluation benchmark that is centered around native Bengali culture, history, and linguistics. 

---
# Predicting Implicit Arguments in Procedural Video Instructions 

**Authors**: Anil Batra, Laura Sevilla-Lara, Marcus Rohrbach, Frank Keller  

**Link**: [PDF](https://arxiv.org/pdf/2505.21068)  

**Abstract**: Procedural texts help AI enhance reasoning about context and action sequences. Transforming these into Semantic Role Labeling (SRL) improves understanding of individual steps by identifying predicate-argument structure like {verb,what,where/with}. Procedural instructions are highly elliptic, for instance, (i) add cucumber to the bowl and (ii) add sliced tomatoes, the second step's where argument is inferred from the context, referring to where the cucumber was placed. Prior SRL benchmarks often miss implicit arguments, leading to incomplete understanding. To address this, we introduce Implicit-VidSRL, a dataset that necessitates inferring implicit and explicit arguments from contextual information in multimodal cooking procedures. Our proposed dataset benchmarks multimodal models' contextual reasoning, requiring entity tracking through visual changes in recipes. We study recent multimodal LLMs and reveal that they struggle to predict implicit arguments of what and where/with from multi-modal procedural data given the verb. Lastly, we propose iSRL-Qwen2-VL, which achieves a 17% relative improvement in F1-score for what-implicit and a 14.7% for where/with-implicit semantic roles over GPT-4o. 

---
# Evaluation of LLMs in Medical Text Summarization: The Role of Vocabulary Adaptation in High OOV Settings 

**Authors**: Gunjan Balde, Soumyadeep Roy, Mainack Mondal, Niloy Ganguly  

**Link**: [PDF](https://arxiv.org/pdf/2505.21242)  

**Abstract**: Large Language Models (LLMs) recently achieved great success in medical text summarization by simply using in-context learning. However, these recent efforts do not perform fine-grained evaluations under difficult settings where LLMs might fail. They typically report performance scores over the entire dataset. Through our benchmarking study, we show that LLMs show a significant performance drop for data points with high concentration of out-of-vocabulary (OOV) words or with high novelty. Vocabulary adaptation is an intuitive solution to this vocabulary mismatch issue where the LLM vocabulary gets updated with certain expert domain (here, medical) words or subwords. An interesting finding from our study is that Llama-3.1, even with a vocabulary size of around 128K tokens, still faces over-fragmentation issue with medical words. To that end, we show vocabulary adaptation helps improve the LLM summarization performance even in difficult settings. Through extensive experimentation of multiple vocabulary adaptation strategies, two continual pretraining strategies, and three benchmark medical summarization datasets, we gain valuable insights into the role of vocabulary adaptation strategies for customizing LLMs to the medical domain. We also performed a human evaluation study with medical experts where they found that vocabulary adaptation results in more relevant and faithful summaries. Our codebase is made publicly available at this https URL. 

---
# Def-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation 

**Authors**: Seungmin Lee, Yongsang Yoo, Minhwa Jung, Min Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.21033)  

**Abstract**: Dialogue Topic Segmentation (DTS) aims to divide dialogues into coherent segments. DTS plays a crucial role in various NLP downstream tasks, but suffers from chronic problems: data shortage, labeling ambiguity, and incremental complexity of recently proposed solutions. On the other hand, Despite advances in Large Language Models (LLMs) and reasoning strategies, these have rarely been applied to DTS. This paper introduces Def-DTS: Deductive Reasoning for Open-domain Dialogue Topic Segmentation, which utilizes LLM-based multi-step deductive reasoning to enhance DTS performance and enable case study using intermediate result. Our method employs a structured prompting approach for bidirectional context summarization, utterance intent classification, and deductive topic shift detection. In the intent classification process, we propose the generalizable intent list for domain-agnostic dialogue intent classification. Experiments in various dialogue settings demonstrate that Def-DTS consistently outperforms traditional and state-of-the-art approaches, with each subtask contributing to improved performance, particularly in reducing type 2 error. We also explore the potential for autolabeling, emphasizing the importance of LLM reasoning techniques in DTS. 

---
# LLMs are Frequency Pattern Learners in Natural Language Inference 

**Authors**: Liang Cheng, Zhaowei Wang, Mark Steedman  

**Link**: [PDF](https://arxiv.org/pdf/2505.21011)  

**Abstract**: While fine-tuning LLMs on NLI corpora improves their inferential performance, the underlying mechanisms driving this improvement remain largely opaque. In this work, we conduct a series of experiments to investigate what LLMs actually learn during fine-tuning. We begin by analyzing predicate frequencies in premises and hypotheses across NLI datasets and identify a consistent frequency bias, where predicates in hypotheses occur more frequently than those in premises for positive instances. To assess the impact of this bias, we evaluate both standard and NLI fine-tuned LLMs on bias-consistent and bias-adversarial cases. We find that LLMs exploit frequency bias for inference and perform poorly on adversarial instances. Furthermore, fine-tuned LLMs exhibit significantly increased reliance on this bias, suggesting that they are learning these frequency patterns from datasets. Finally, we compute the frequencies of hyponyms and their corresponding hypernyms from WordNet, revealing a correlation between frequency bias and textual entailment. These findings help explain why learning frequency patterns can enhance model performance on inference tasks. 

---
# Assessment of L2 Oral Proficiency using Speech Large Language Models 

**Authors**: Rao Ma, Mengjie Qian, Siyuan Tang, Stefano Bannò, Kate M. Knill, Mark J.F. Gales  

**Link**: [PDF](https://arxiv.org/pdf/2505.21148)  

**Abstract**: The growing population of L2 English speakers has increased the demand for developing automatic graders for spoken language assessment (SLA). Historically, statistical models, text encoders, and self-supervised speech models have been utilised for this task. However, cascaded systems suffer from the loss of information, while E2E graders also have limitations. With the recent advancements of multi-modal large language models (LLMs), we aim to explore their potential as L2 oral proficiency graders and overcome these issues. In this work, we compare various training strategies using regression and classification targets. Our results show that speech LLMs outperform all previous competitive baselines, achieving superior performance on two datasets. Furthermore, the trained grader demonstrates strong generalisation capabilities in the cross-part or cross-task evaluation, facilitated by the audio understanding knowledge acquired during LLM pre-training. 

---
# LMCD: Language Models are Zeroshot Cognitive Diagnosis Learners 

**Authors**: Yu He, Zihan Yao, Chentao Song, Tianyu Qi, Jun Liu, Ming Li, Qing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21239)  

**Abstract**: Cognitive Diagnosis (CD) has become a critical task in AI-empowered education, supporting personalized learning by accurately assessing students' cognitive states. However, traditional CD models often struggle in cold-start scenarios due to the lack of student-exercise interaction data. Recent NLP-based approaches leveraging pre-trained language models (PLMs) have shown promise by utilizing textual features but fail to fully bridge the gap between semantic understanding and cognitive profiling. In this work, we propose Language Models as Zeroshot Cognitive Diagnosis Learners (LMCD), a novel framework designed to handle cold-start challenges by harnessing large language models (LLMs). LMCD operates via two primary phases: (1) Knowledge Diffusion, where LLMs generate enriched contents of exercises and knowledge concepts (KCs), establishing stronger semantic links; and (2) Semantic-Cognitive Fusion, where LLMs employ causal attention mechanisms to integrate textual information and student cognitive states, creating comprehensive profiles for both students and exercises. These representations are efficiently trained with off-the-shelf CD models. Experiments on two real-world datasets demonstrate that LMCD significantly outperforms state-of-the-art methods in both exercise-cold and domain-cold settings. The code is publicly available at this https URL 

---
# Who Reasons in the Large Language Models? 

**Authors**: Jie Shao, Jianxin Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20993)  

**Abstract**: Despite the impressive performance of large language models (LLMs), the process of endowing them with new capabilities--such as mathematical reasoning--remains largely empirical and opaque. A critical open question is whether reasoning abilities stem from the entire model, specific modules, or are merely artifacts of overfitting. In this work, we hypothesize that the reasoning capabilities in well-trained LLMs are primarily attributed to the output projection module (oproj) in the Transformer's multi-head self-attention (MHSA) mechanism. To support this hypothesis, we introduce Stethoscope for Networks (SfN), a suite of diagnostic tools designed to probe and analyze the internal behaviors of LLMs. Using SfN, we provide both circumstantial and empirical evidence suggesting that oproj plays a central role in enabling reasoning, whereas other modules contribute more to fluent dialogue. These findings offer a new perspective on LLM interpretability and open avenues for more targeted training strategies, potentially enabling more efficient and specialized LLMs. 

---
# Evaluating and Steering Modality Preferences in Multimodal Large Language Model 

**Authors**: Yu Zhang, Jinlong Ma, Yongshuai Hou, Xuefeng Bai, Kehai Chen, Yang Xiang, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20977)  

**Abstract**: Multimodal large language models (MLLMs) have achieved remarkable performance on complex tasks with multimodal context. However, it is still understudied whether they exhibit modality preference when processing multimodal contexts. To study this question, we first build a \textbf{MC\textsuperscript{2}} benchmark under controlled evidence conflict scenarios to systematically evaluate modality preference, which is the tendency to favor one modality over another when making decisions based on multimodal conflicting evidence. Our extensive evaluation reveals that all 18 tested MLLMs generally demonstrate clear modality bias, and modality preference can be influenced by external interventions. An in-depth analysis reveals that the preference direction can be captured within the latent representations of MLLMs. Built on this, we propose a probing and steering method based on representation engineering to explicitly control modality preference without additional fine-tuning or carefully crafted prompts. Our method effectively amplifies modality preference toward a desired direction and applies to downstream tasks such as hallucination mitigation and multimodal machine translation, yielding promising improvements. 

---
# Contrastive Learning on LLM Back Generation Treebank for Cross-domain Constituency Parsing 

**Authors**: Peiming Guo, Meishan Zhang, Jianling Li, Min Zhang, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20976)  

**Abstract**: Cross-domain constituency parsing is still an unsolved challenge in computational linguistics since the available multi-domain constituency treebank is limited. We investigate automatic treebank generation by large language models (LLMs) in this paper. The performance of LLMs on constituency parsing is poor, therefore we propose a novel treebank generation method, LLM back generation, which is similar to the reverse process of constituency parsing. LLM back generation takes the incomplete cross-domain constituency tree with only domain keyword leaf nodes as input and fills the missing words to generate the cross-domain constituency treebank. Besides, we also introduce a span-level contrastive learning pre-training strategy to make full use of the LLM back generation treebank for cross-domain constituency parsing. We verify the effectiveness of our LLM back generation treebank coupled with contrastive learning pre-training on five target domains of MCTB. Experimental results show that our approach achieves state-of-the-art performance on average results compared with various baselines. 

---
# Reason-Align-Respond: Aligning LLM Reasoning with Knowledge Graphs for KGQA 

**Authors**: Xiangqing Shen, Fanfan Wang, Rui Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.20971)  

**Abstract**: LLMs have demonstrated remarkable capabilities in complex reasoning tasks, yet they often suffer from hallucinations and lack reliable factual grounding. Meanwhile, knowledge graphs (KGs) provide structured factual knowledge but lack the flexible reasoning abilities of LLMs. In this paper, we present Reason-Align-Respond (RAR), a novel framework that systematically integrates LLM reasoning with knowledge graphs for KGQA. Our approach consists of three key components: a Reasoner that generates human-like reasoning chains, an Aligner that maps these chains to valid KG paths, and a Responser that synthesizes the final answer. We formulate this process as a probabilistic model and optimize it using the Expectation-Maximization algorithm, which iteratively refines the reasoning chains and knowledge paths. Extensive experiments on multiple benchmarks demonstrate the effectiveness of RAR, achieving state-of-the-art performance with Hit@1 scores of 93.3% and 91.0% on WebQSP and CWQ respectively. Human evaluation confirms that RAR generates high-quality, interpretable reasoning chains well-aligned with KG paths. Furthermore, RAR exhibits strong zero-shot generalization capabilities and maintains computational efficiency during inference. 

---
# TAT-R1: Terminology-Aware Translation with Reinforcement Learning and Word Alignment 

**Authors**: Zheng Li, Mao Zheng, Mingyang Song, Wenjie Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21172)  

**Abstract**: Recently, deep reasoning large language models(LLMs) like DeepSeek-R1 have made significant progress in tasks such as mathematics and coding. Inspired by this, several studies have employed reinforcement learning(RL) to enhance models' deep reasoning capabilities and improve machine translation(MT) quality. However, the terminology translation, an essential task in MT, remains unexplored in deep reasoning LLMs. In this paper, we propose \textbf{TAT-R1}, a terminology-aware translation model trained with reinforcement learning and word alignment. Specifically, we first extract the keyword translation pairs using a word alignment model. Then we carefully design three types of rule-based alignment rewards with the extracted alignment relationships. With those alignment rewards, the RL-trained translation model can learn to focus on the accurate translation of key information, including terminology in the source text. Experimental results show the effectiveness of TAT-R1. Our model significantly improves terminology translation accuracy compared to the baseline models while maintaining comparable performance on general translation tasks. In addition, we conduct detailed ablation studies of the DeepSeek-R1-like training paradigm for machine translation and reveal several key findings. 

---
# Multi-objective Large Language Model Alignment with Hierarchical Experts 

**Authors**: Zhuo Li, Guodong Du, Weiyang Guo, Yigeng Zhou, Xiucheng Li, Wenya Wang, Fangming Liu, Yequan Wang, Deheng Ye, Min Zhang, Jing Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20925)  

**Abstract**: Aligning large language models (LLMs) to simultaneously satisfy multiple objectives remains a significant challenge, especially given the diverse and often conflicting nature of human preferences. Existing alignment methods struggle to balance trade-offs effectively, often requiring costly retraining or yielding suboptimal results across the Pareto frontier of preferences. In this paper, we introduce \textit{HoE}(Hierarchical Mixture-of-Experts), a \textit{lightweight}, \textit{parameter-efficient}, and \textit{plug-and-play} approach that eliminates the need for model training, while enabling LLMs to adapt across the entire Pareto frontier and accommodate diverse user preferences. In particular, \textit{HoE} consists of three hierarchical components: LoRA Experts, Router Experts and Preference Routing, reaching optimal Pareto frontiers and achieving a trade-off between parameter size, training cost, and performance. We evaluate \textit{HoE} across various tasks on 14 objectives and 200 different preferences among 6 benchmarks, demonstrating superior performance over 15 recent baselines. Code is available in the supplementary materials. 

---
# Uncertainty Unveiled: Can Exposure to More In-context Examples Mitigate Uncertainty for Large Language Models? 

**Authors**: Yifei Wang, Yu Sheng, Linjing Li, Daniel Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21003)  

**Abstract**: Recent advances in handling long sequences have facilitated the exploration of long-context in-context learning (ICL). While much of the existing research emphasizes performance improvements driven by additional in-context examples, the influence on the trustworthiness of generated responses remains underexplored. This paper addresses this gap by investigating how increased examples influence predictive uncertainty, an essential aspect in trustworthiness. We begin by systematically quantifying the uncertainty of ICL with varying shot counts, analyzing the impact of example quantity. Through uncertainty decomposition, we introduce a novel perspective on performance enhancement, with a focus on epistemic uncertainty (EU). Our results reveal that additional examples reduce total uncertainty in both simple and complex tasks by injecting task-specific knowledge, thereby diminishing EU and enhancing performance. For complex tasks, these advantages emerge only after addressing the increased noise and uncertainty associated with longer inputs. Finally, we explore the evolution of internal confidence across layers, unveiling the mechanisms driving the reduction in uncertainty. 

---
# Automated Privacy Information Annotation in Large Language Model Interactions 

**Authors**: Hang Zeng, Xiangyu Liu, Yong Hu, Chaoyue Niu, Fan Wu, Shaojie Tang, Guihai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.20910)  

**Abstract**: Users interacting with large language models (LLMs) under their real identifiers often unknowingly risk disclosing private information. Automatically notifying users whether their queries leak privacy and which phrases leak what private information has therefore become a practical need. Existing privacy detection methods, however, were designed for different objectives and application scenarios, typically tagging personally identifiable information (PII) in anonymous content. In this work, to support the development and evaluation of privacy detection models for LLM interactions that are deployable on local user devices, we construct a large-scale multilingual dataset with 249K user queries and 154K annotated privacy phrases. In particular, we build an automated privacy annotation pipeline with cloud-based strong LLMs to automatically extract privacy phrases from dialogue datasets and annotate leaked information. We also design evaluation metrics at the levels of privacy leakage, extracted privacy phrase, and privacy information. We further establish baseline methods using light-weight LLMs with both tuning-free and tuning-based methods, and report a comprehensive evaluation of their performance. Evaluation results reveal a gap between current performance and the requirements of real-world LLM applications, motivating future research into more effective local privacy detection methods grounded in our dataset. 

---
# Towards Objective Fine-tuning: How LLMs' Prior Knowledge Causes Potential Poor Calibration? 

**Authors**: Ziming Wang, Zeyu Shi, Haoyi Zhou, Shiqi Gao, Qingyun Sun, Jianxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20903)  

**Abstract**: Fine-tuned Large Language Models (LLMs) often demonstrate poor calibration, with their confidence scores misaligned with actual performance. While calibration has been extensively studied in models trained from scratch, the impact of LLMs' prior knowledge on calibration during fine-tuning remains understudied. Our research reveals that LLMs' prior knowledge causes potential poor calibration due to the ubiquitous presence of known data in real-world fine-tuning, which appears harmful for calibration. Specifically, data aligned with LLMs' prior knowledge would induce overconfidence, while new knowledge improves calibration. Our findings expose a tension: LLMs' encyclopedic knowledge, while enabling task versatility, undermines calibration through unavoidable knowledge overlaps. To address this, we propose CogCalib, a cognition-aware framework that applies targeted learning strategies according to the model's prior knowledge. Experiments across 7 tasks using 3 LLM families prove that CogCalib significantly improves calibration while maintaining performance, achieving an average 57\% reduction in ECE compared to standard fine-tuning in Llama3-8B. These improvements generalize well to out-of-domain tasks, enhancing the objectivity and reliability of domain-specific LLMs, and making them more trustworthy for critical human-AI interaction applications. 

---
# EasyDistill: A Comprehensive Toolkit for Effective Knowledge Distillation of Large Language Models 

**Authors**: Chengyu Wang, Junbing Yan, Wenrui Cai, Yuanhao Yue, Jun Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20888)  

**Abstract**: In this paper, we present EasyDistill, a comprehensive toolkit designed for effective black-box and white-box knowledge distillation (KD) of large language models (LLMs). Our framework offers versatile functionalities, including data synthesis, supervised fine-tuning, ranking optimization, and reinforcement learning techniques specifically tailored for KD scenarios. The toolkit accommodates KD functionalities for both System 1 (fast, intuitive) and System 2 (slow, analytical) models. With its modular design and user-friendly interface, EasyDistill empowers researchers and industry practitioners to seamlessly experiment with and implement state-of-the-art KD strategies for LLMs. In addition, EasyDistill provides a series of robust distilled models and KD-based industrial solutions developed by us, along with the corresponding open-sourced datasets, catering to a variety of use cases. Furthermore, we describe the seamless integration of EasyDistill into Alibaba Cloud's Platform for AI (PAI). Overall, the EasyDistill toolkit makes advanced KD techniques for LLMs more accessible and impactful within the NLP community. 

---
# Trans-EnV: A Framework for Evaluating the Linguistic Robustness of LLMs Against English Varieties 

**Authors**: Jiyoung Lee, Seungho Kim, Jieun Han, Jun-Min Lee, Kitaek Kim, Alice Oh, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.20875)  

**Abstract**: Large Language Models (LLMs) are predominantly evaluated on Standard American English (SAE), often overlooking the diversity of global English varieties. This narrow focus may raise fairness concerns as degraded performance on non-standard varieties can lead to unequal benefits for users worldwide. Therefore, it is critical to extensively evaluate the linguistic robustness of LLMs on multiple non-standard English varieties. We introduce Trans-EnV, a framework that automatically transforms SAE datasets into multiple English varieties to evaluate the linguistic robustness. Our framework combines (1) linguistics expert knowledge to curate variety-specific features and transformation guidelines from linguistic literature and corpora, and (2) LLM-based transformations to ensure both linguistic validity and scalability. Using Trans-EnV, we transform six benchmark datasets into 38 English varieties and evaluate seven state-of-the-art LLMs. Our results reveal significant performance disparities, with accuracy decreasing by up to 46.3% on non-standard varieties. These findings highlight the importance of comprehensive linguistic robustness evaluation across diverse English varieties. Each construction of Trans-EnV was validated through rigorous statistical testing and consultation with a researcher in the field of second language acquisition, ensuring its linguistic validity. Our \href{this https URL}{code} and \href{this https URL}{datasets} are publicly available. 

---
# Can LLMs Learn to Map the World from Local Descriptions? 

**Authors**: Sirui Xia, Aili Chen, Xintao Wang, Tinghui Zhu, Yikai Zhang, Jiangjie Chen, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2505.20874)  

**Abstract**: Recent advances in Large Language Models (LLMs) have demonstrated strong capabilities in tasks such as code and mathematics. However, their potential to internalize structured spatial knowledge remains underexplored. This study investigates whether LLMs, grounded in locally relative human observations, can construct coherent global spatial cognition by integrating fragmented relational descriptions. We focus on two core aspects of spatial cognition: spatial perception, where models infer consistent global layouts from local positional relationships, and spatial navigation, where models learn road connectivity from trajectory data and plan optimal paths between unconnected locations. Experiments conducted in a simulated urban environment demonstrate that LLMs not only generalize to unseen spatial relationships between points of interest (POIs) but also exhibit latent representations aligned with real-world spatial distributions. Furthermore, LLMs can learn road connectivity from trajectory descriptions, enabling accurate path planning and dynamic spatial awareness during navigation. 

---
# Concealment of Intent: A Game-Theoretic Analysis 

**Authors**: Xinbo Wu, Abhishek Umrawal, Lav R. Varshney  

**Link**: [PDF](https://arxiv.org/pdf/2505.20841)  

**Abstract**: As large language models (LLMs) grow more capable, concerns about their safe deployment have also grown. Although alignment mechanisms have been introduced to deter misuse, they remain vulnerable to carefully designed adversarial prompts. In this work, we present a scalable attack strategy: intent-hiding adversarial prompting, which conceals malicious intent through the composition of skills. We develop a game-theoretic framework to model the interaction between such attacks and defense systems that apply both prompt and response filtering. Our analysis identifies equilibrium points and reveals structural advantages for the attacker. To counter these threats, we propose and analyze a defense mechanism tailored to intent-hiding attacks. Empirically, we validate the attack's effectiveness on multiple real-world LLMs across a range of malicious behaviors, demonstrating clear advantages over existing adversarial prompting techniques. 

---
# Tracing and Reversing Rank-One Model Edits 

**Authors**: Paul Youssef, Zhixue Zhao, Christin Seifert, Jörg Schlötterer  

**Link**: [PDF](https://arxiv.org/pdf/2505.20819)  

**Abstract**: Knowledge editing methods (KEs) are a cost-effective way to update the factual content of large language models (LLMs), but they pose a dual-use risk. While KEs are beneficial for updating outdated or incorrect information, they can be exploited maliciously to implant misinformation or bias. In order to defend against these types of malicious manipulation, we need robust techniques that can reliably detect, interpret, and mitigate adversarial edits. This work investigates the traceability and reversibility of knowledge edits, focusing on the widely used Rank-One Model Editing (ROME) method. We first show that ROME introduces distinctive distributional patterns in the edited weight matrices, which can serve as effective signals for locating the edited weights. Second, we show that these altered weights can reliably be used to predict the edited factual relation, enabling partial reconstruction of the modified fact. Building on this, we propose a method to infer the edited object entity directly from the modified weights, without access to the editing prompt, achieving over 95% accuracy. Finally, we demonstrate that ROME edits can be reversed, recovering the model's original outputs with $\geq$ 80% accuracy. Our findings highlight the feasibility of detecting, tracing, and reversing edits based on the edited weights, offering a robust framework for safeguarding LLMs against adversarial manipulations. 

---
# Rethinking Information Synthesis in Multimodal Question Answering A Multi-Agent Perspective 

**Authors**: Krishna Singh Rajput, Tejas Anvekar, Chitta Baral, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.20816)  

**Abstract**: Recent advances in multimodal question answering have primarily focused on combining heterogeneous modalities or fine-tuning multimodal large language models. While these approaches have shown strong performance, they often rely on a single, generalized reasoning strategy, overlooking the unique characteristics of each modality ultimately limiting both accuracy and interpretability. To address these limitations, we propose MAMMQA, a multi-agent QA framework for multimodal inputs spanning text, tables, and images. Our system includes two Visual Language Model (VLM) agents and one text-based Large Language Model (LLM) agent. The first VLM decomposes the user query into sub-questions and sequentially retrieves partial answers from each modality. The second VLM synthesizes and refines these results through cross-modal reasoning. Finally, the LLM integrates the insights into a cohesive answer. This modular design enhances interpretability by making the reasoning process transparent and allows each agent to operate within its domain of expertise. Experiments on diverse multimodal QA benchmarks demonstrate that our cooperative, multi-agent framework consistently outperforms existing baselines in both accuracy and robustness. 

---
# Automatic Transmission for LLM Tiers: Optimizing Cost and Accuracy in Large Language Models 

**Authors**: Injae Na, Keonwoong Noh, Woohwan Jung  

**Link**: [PDF](https://arxiv.org/pdf/2505.20921)  

**Abstract**: LLM providers typically offer multiple LLM tiers, varying in performance and price. As NLP tasks become more complex and modularized, selecting the suitable LLM tier for each subtask is a key challenge to balance between cost and performance. To address the problem, we introduce LLM Automatic Transmission (LLM-AT) framework that automatically selects LLM tiers without training. LLM-AT consists of Starter, Generator, and Judge. The starter selects the initial LLM tier expected to solve the given question, the generator produces a response using the LLM of the selected tier, and the judge evaluates the validity of the response. If the response is invalid, LLM-AT iteratively upgrades to a higher-tier model, generates a new response, and re-evaluates until a valid response is obtained. Additionally, we propose accuracy estimator, which enables the suitable initial LLM tier selection without training. Given an input question, accuracy estimator estimates the expected accuracy of each LLM tier by computing the valid response rate across top-k similar queries from past inference records. Experiments demonstrate that LLM-AT achieves superior performance while reducing costs, making it a practical solution for real-world applications. 

---
# MSA at SemEval-2025 Task 3: High Quality Weak Labeling and LLM Ensemble Verification for Multilingual Hallucination Detection 

**Authors**: Baraa Hikal, Ahmed Nasreldin, Ali Hamdi  

**Link**: [PDF](https://arxiv.org/pdf/2505.20880)  

**Abstract**: This paper describes our submission for SemEval-2025 Task 3: Mu-SHROOM, the Multilingual Shared-task on Hallucinations and Related Observable Overgeneration Mistakes. The task involves detecting hallucinated spans in text generated by instruction-tuned Large Language Models (LLMs) across multiple languages. Our approach combines task-specific prompt engineering with an LLM ensemble verification mechanism, where a primary model extracts hallucination spans and three independent LLMs adjudicate their validity through probability-based voting. This framework simulates the human annotation workflow used in the shared task validation and test data. Additionally, fuzzy matching refines span alignment. Our system ranked 1st in Arabic and Basque, 2nd in German, Swedish, and Finnish, and 3rd in Czech, Farsi, and French. 

---
# SpecExtend: A Drop-in Enhancement for Speculative Decoding of Long Sequences 

**Authors**: Jungyoub Cha, Hyunjong Kim, Sungzoon Cho  

**Link**: [PDF](https://arxiv.org/pdf/2505.20776)  

**Abstract**: Speculative decoding is a widely adopted technique for accelerating inference in large language models (LLMs), but its performance degrades on long inputs due to increased attention cost and reduced draft accuracy. We introduce SpecExtend, a drop-in enhancement that improves the performance of speculative decoding on long sequences without any additional training. SpecExtend integrates efficient attention mechanisms such as FlashAttention and Hybrid Tree Attention into both the draft and target models, reducing latency across all stages. To improve draft accuracy and speed, we propose Cross-model Retrieval, a novel KV cache update strategy that uses the target model's attention scores to dynamically select relevant context for the draft model. Extensive evaluations on three long-context understanding datasets show that SpecExtend accelerates standard tree-based speculative decoding by up to 2.22x for inputs up to 16K tokens, providing an effective solution for speculative decoding of long sequences. The code is available at this https URL . 

---
# CogniBench: A Legal-inspired Framework and Dataset for Assessing Cognitive Faithfulness of Large Language Models 

**Authors**: Xiaqiang Tang, Jian Li, Keyu Hu, Du Nan, Xiaolong Li, Xi Zhang, Weigao Sun, Sihong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.20767)  

**Abstract**: Faithfulness hallucination are claims generated by a Large Language Model (LLM) not supported by contexts provided to the LLM. Lacking assessment standard, existing benchmarks only contain "factual statements" that rephrase source materials without marking "cognitive statements" that make inference from the given context, making the consistency evaluation and optimization of cognitive statements difficult. Inspired by how an evidence is assessed in the legislative domain, we design a rigorous framework to assess different levels of faithfulness of cognitive statements and create a benchmark dataset where we reveal insightful statistics. We design an annotation pipeline to create larger benchmarks for different LLMs automatically, and the resulting larger-scale CogniBench-L dataset can be used to train accurate cognitive hallucination detection model. We release our model and dataset at: this https URL 

---
# Beyond Templates: Dynamic Adaptation of Reasoning Demonstrations via Feasibility-Aware Exploration 

**Authors**: Yong Wu, Weihang Pan, Ke Li, Chen Binhui, Ping Li, Binbin Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.20700)  

**Abstract**: Large language models (LLMs) have shown remarkable reasoning capabilities, yet aligning such abilities to small language models (SLMs) remains a challenge due to distributional mismatches and limited model capacity. Existing reasoning datasets, typically designed for powerful LLMs, often lead to degraded performance when directly applied to weaker models. In this work, we introduce Dynamic Adaptation of Reasoning Trajectories (DART), a novel data adaptation framework that bridges the capability gap between expert reasoning trajectories and diverse SLMs. Instead of uniformly imitating expert steps, DART employs a selective imitation strategy guided by step-wise adaptability estimation via solution simulation. When expert steps surpass the student's capacity -- signaled by an Imitation Gap -- the student autonomously explores alternative reasoning paths, constrained by outcome consistency. We validate DART across multiple reasoning benchmarks and model scales, demonstrating that it significantly improves generalization and data efficiency over static fine-tuning. Our method enhances supervision quality by aligning training signals with the student's reasoning capabilities, offering a scalable solution for reasoning alignment in resource-constrained models. 

---
# Silencer: From Discovery to Mitigation of Self-Bias in LLM-as-Benchmark-Generator 

**Authors**: Peiwen Yuan, Yiwei Li, Shaoxiong Feng, Xinglin Wang, Yueqi Zhang, Jiayi Shi, Chuyi Tan, Boyuan Pan, Yao Hu, Kan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20738)  

**Abstract**: LLM-as-Benchmark-Generator methods have been widely studied as a supplement to human annotators for scalable evaluation, while the potential biases within this paradigm remain underexplored. In this work, we systematically define and validate the phenomenon of inflated performance in models evaluated on their self-generated benchmarks, referred to as self-bias, and attribute it to sub-biases arising from question domain, language style, and wrong labels. On this basis, we propose Silencer, a general framework that leverages the heterogeneity between multiple generators at both the sample and benchmark levels to neutralize bias and generate high-quality, self-bias-silenced benchmark. Experimental results across various settings demonstrate that Silencer can suppress self-bias to near zero, significantly improve evaluation effectiveness of the generated benchmark (with an average improvement from 0.655 to 0.833 in Pearson correlation with high-quality human-annotated benchmark), while also exhibiting strong generalizability. 

---
# Pretraining Language Models to Ponder in Continuous Space 

**Authors**: Boyi Zeng, Shixiang Song, Siyuan Huang, Yixuan Wang, He Li, Ziwei He, Xinbing Wang, Zhiyu Li, Zhouhan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.20674)  

**Abstract**: Humans ponder before articulating complex sentence elements, enabling deeper cognitive processing through focused effort. In this work, we introduce this pondering process into language models by repeatedly invoking the forward process within a single token generation step. During pondering, instead of generating an actual token sampled from the prediction distribution, the model ponders by yielding a weighted sum of all token embeddings according to the predicted token distribution. The generated embedding is then fed back as input for another forward pass. We show that the model can learn to ponder in this way through self-supervised learning, without any human annotations. Our method is straightforward and can be seamlessly integrated with various existing language models. Experiments across three widely used open-source architectures-GPT-2, Pythia, and LLaMA-and extensive downstream task evaluations demonstrate the effectiveness and generality of our method. For language modeling tasks, pondering language models achieve performance comparable to vanilla models with twice the number of parameters. On 9 downstream benchmarks, our pondering-enhanced Pythia models significantly outperform the official Pythia models. Notably, pondering-enhanced Pythia-1B is comparable to TinyLlama-1.1B, which is trained on 10 times more data. The code is available at this https URL. 

---
# SELF-PERCEPT: Introspection Improves Large Language Models' Detection of Multi-Person Mental Manipulation in Conversations 

**Authors**: Danush Khanna, Pratinav Seth, Sidhaarth Sredharan Murali, Aditya Kumar Guru, Siddharth Shukla, Tanuj Tyagi, Sandeep Chaurasia, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2505.20679)  

**Abstract**: Mental manipulation is a subtle yet pervasive form of abuse in interpersonal communication, making its detection critical for safeguarding potential victims. However, due to manipulation's nuanced and context-specific nature, identifying manipulative language in complex, multi-turn, and multi-person conversations remains a significant challenge for large language models (LLMs). To address this gap, we introduce the MultiManip dataset, comprising 220 multi-turn, multi-person dialogues balanced between manipulative and non-manipulative interactions, all drawn from reality shows that mimic real-world scenarios. For manipulative interactions, it includes 11 distinct manipulations depicting real-life scenarios. We conduct extensive evaluations of state-of-the-art LLMs, such as GPT-4o and Llama-3.1-8B, employing various prompting strategies. Despite their capabilities, these models often struggle to detect manipulation effectively. To overcome this limitation, we propose SELF-PERCEPT, a novel, two-stage prompting framework inspired by Self-Perception Theory, demonstrating strong performance in detecting multi-person, multi-turn mental manipulation. Our code and data are publicly available at this https URL . 

---
# Self-Route: Automatic Mode Switching via Capability Estimation for Efficient Reasoning 

**Authors**: Yang He, Xiao Ding, Bibo Cai, Yufei Zhang, Kai Xiong, Zhouhao Sun, Bing Qin, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20664)  

**Abstract**: While reasoning-augmented large language models (RLLMs) significantly enhance complex task performance through extended reasoning chains, they inevitably introduce substantial unnecessary token consumption, particularly for simpler problems where Short Chain-of-Thought (Short CoT) suffices. This overthinking phenomenon leads to inefficient resource usage without proportional accuracy gains. To address this issue, we propose Self-Route, a dynamic reasoning framework that automatically selects between general and reasoning modes based on model capability estimation. Our approach introduces a lightweight pre-inference stage to extract capability-aware embeddings from hidden layer representations, enabling real-time evaluation of the model's ability to solve problems. We further construct Gradient-10K, a model difficulty estimation-based dataset with dense complexity sampling, to train the router for precise capability boundary detection. Extensive experiments demonstrate that Self-Route achieves comparable accuracy to reasoning models while reducing token consumption by 30-55\% across diverse benchmarks. The proposed framework demonstrates consistent effectiveness across models with different parameter scales and reasoning paradigms, highlighting its general applicability and practical value. 

---
# Enhancing Transformation from Natural Language to Signal Temporal Logic Using LLMs with Diverse External Knowledge 

**Authors**: Yue Fang, Zhi Jin, Jie An, Hongshen Chen, Xiaohong Chen, Naijun Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.20658)  

**Abstract**: Temporal Logic (TL), especially Signal Temporal Logic (STL), enables precise formal specification, making it widely used in cyber-physical systems such as autonomous driving and robotics. Automatically transforming NL into STL is an attractive approach to overcome the limitations of manual transformation, which is time-consuming and error-prone. However, due to the lack of datasets, automatic transformation currently faces significant challenges and has not been fully explored. In this paper, we propose an NL-STL dataset named STL-Diversity-Enhanced (STL-DivEn), which comprises 16,000 samples enriched with diverse patterns. To develop the dataset, we first manually create a small-scale seed set of NL-STL pairs. Next, representative examples are identified through clustering and used to guide large language models (LLMs) in generating additional NL-STL pairs. Finally, diversity and accuracy are ensured through rigorous rule-based filters and human validation. Furthermore, we introduce the Knowledge-Guided STL Transformation (KGST) framework, a novel approach for transforming natural language into STL, involving a generate-then-refine process based on external knowledge. Statistical analysis shows that the STL-DivEn dataset exhibits more diversity than the existing NL-STL dataset. Moreover, both metric-based and human evaluations indicate that our KGST approach outperforms baseline models in transformation accuracy on STL-DivEn and DeepSTL datasets. 

---
# FinTagging: An LLM-ready Benchmark for Extracting and Structuring Financial Information 

**Authors**: Yan Wang, Yang Ren, Lingfei Qian, Xueqing Peng, Keyi Wang, Yi Han, Dongji Feng, Xiao-Yang Liu, Jimin Huang, Qianqian Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.20650)  

**Abstract**: We introduce FinTagging, the first full-scope, table-aware XBRL benchmark designed to evaluate the structured information extraction and semantic alignment capabilities of large language models (LLMs) in the context of XBRL-based financial reporting. Unlike prior benchmarks that oversimplify XBRL tagging as flat multi-class classification and focus solely on narrative text, FinTagging decomposes the XBRL tagging problem into two subtasks: FinNI for financial entity extraction and FinCL for taxonomy-driven concept alignment. It requires models to jointly extract facts and align them with the full 10k+ US-GAAP taxonomy across both unstructured text and structured tables, enabling realistic, fine-grained evaluation. We assess a diverse set of LLMs under zero-shot settings, systematically analyzing their performance on both subtasks and overall tagging accuracy. Our results reveal that, while LLMs demonstrate strong generalization in information extraction, they struggle with fine-grained concept alignment, particularly in disambiguating closely related taxonomy entries. These findings highlight the limitations of existing LLMs in fully automating XBRL tagging and underscore the need for improved semantic reasoning and schema-aware modeling to meet the demands of accurate financial disclosure. Code is available at our GitHub repository and data is at our Hugging Face repository. 

---
# Test-Time Learning for Large Language Models 

**Authors**: Jinwu Hu, Zhitian Zhang, Guohao Chen, Xutao Wen, Chao Shuai, Wei Luo, Bin Xiao, Yuanqing Li, Mingkui Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.20633)  

**Abstract**: While Large Language Models (LLMs) have exhibited remarkable emergent capabilities through extensive pre-training, they still face critical limitations in generalizing to specialized domains and handling diverse linguistic variations, known as distribution shifts. In this paper, we propose a Test-Time Learning (TTL) paradigm for LLMs, namely TLM, which dynamically adapts LLMs to target domains using only unlabeled test data during testing. Specifically, we first provide empirical evidence and theoretical insights to reveal that more accurate predictions from LLMs can be achieved by minimizing the input perplexity of the unlabeled test data. Based on this insight, we formulate the Test-Time Learning process of LLMs as input perplexity minimization, enabling self-supervised enhancement of LLM performance. Furthermore, we observe that high-perplexity samples tend to be more informative for model optimization. Accordingly, we introduce a Sample Efficient Learning Strategy that actively selects and emphasizes these high-perplexity samples for test-time updates. Lastly, to mitigate catastrophic forgetting and ensure adaptation stability, we adopt Low-Rank Adaptation (LoRA) instead of full-parameter optimization, which allows lightweight model updates while preserving more original knowledge from the model. We introduce the AdaptEval benchmark for TTL and demonstrate through experiments that TLM improves performance by at least 20% compared to original LLMs on domain knowledge adaptation. 

---
# Long Context Scaling: Divide and Conquer via Multi-Agent Question-driven Collaboration 

**Authors**: Sibo Xiao, Zixin Lin, Wenyang Gao, Yue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20625)  

**Abstract**: Processing long contexts has become a critical capability for modern large language models (LLMs). Existing works leverage agent-based divide-and-conquer methods for processing long contexts. But these methods face crucial limitations, including prohibitive accumulated latency and amplified information loss from excessive agent invocations, and the disruption of inherent textual dependencies by immoderate partitioning. In this paper, we propose a novel multi-agent framework XpandA (Expand-Agent) coupled with question-driven workflow and dynamic partitioning for robust long-context processing. XpandA overcomes these limitations through: 1) dynamic partitioning of long texts, which adaptively modulates the filling rate of context windows for input sequences of vastly varying lengths; 2) question-guided protocol to update flat information ensembles within centralized shared memory, constructing consistent inter-agent knowledge across partitions; and 3) selectively replaying specific partitions based on the state-tracking of question-information couples to promote the resolution of inverted-order structures across partitions (e.g., flashbacks). We perform a comprehensive evaluation of XpandA on multiple long-context benchmarks with length varying from 1k to 1M, demonstrating XpandA's feasibility for processing ultra-long sequences and its significant effectiveness in enhancing the long-context capabilities of various LLMs by achieving 20\% improvements and 1.5x inference speedup over baselines of full-context, RAG and previous agent-based methods. 

---
# POLAR: A Benchmark for Multilingual, Multicultural, and Multi-Event Online Polarization 

**Authors**: Usman Naseem, Juan Ren, Saba Anwar, Sarah Kohail, Rudy Alexandro Garrido Veliz, Robert Geislinger, Aisha Jabr, Idris Abdulmumin, Laiba Qureshi, Aarushi Ajay Borkar, Maryam Ibrahim Mukhtar, Abinew Ali Ayele, Ibrahim Said Ahmad, Adem Ali, Martin Semmann, Shamsuddeen Hassan Muhammad, Seid Muhie Yimam  

**Link**: [PDF](https://arxiv.org/pdf/2505.20624)  

**Abstract**: Online polarization poses a growing challenge for democratic discourse, yet most computational social science research remains monolingual, culturally narrow, or event-specific. We introduce POLAR, a multilingual, multicultural, and multievent dataset with over 23k instances in seven languages from diverse online platforms and real-world events. Polarization is annotated along three axes: presence, type, and manifestation, using a variety of annotation platforms adapted to each cultural context. We conduct two main experiments: (1) we fine-tune six multilingual pretrained language models in both monolingual and cross-lingual setups; and (2) we evaluate a range of open and closed large language models (LLMs) in few-shot and zero-shot scenarios. Results show that while most models perform well on binary polarization detection, they achieve substantially lower scores when predicting polarization types and manifestations. These findings highlight the complex, highly contextual nature of polarization and the need for robust, adaptable approaches in NLP and computational social science. All resources will be released to support further research and effective mitigation of digital polarization globally. 

---
# STEER-BENCH: A Benchmark for Evaluating the Steerability of Large Language Models 

**Authors**: Kai Chen, Zihao He, Taiwei Shi, Kristina Lerman  

**Link**: [PDF](https://arxiv.org/pdf/2505.20645)  

**Abstract**: Steerability, or the ability of large language models (LLMs) to adapt outputs to align with diverse community-specific norms, perspectives, and communication styles, is critical for real-world applications but remains under-evaluated. We introduce Steer-Bench, a benchmark for assessing population-specific steering using contrasting Reddit communities. Covering 30 contrasting subreddit pairs across 19 domains, Steer-Bench includes over 10,000 instruction-response pairs and validated 5,500 multiple-choice question with corresponding silver labels to test alignment with diverse community norms. Our evaluation of 13 popular LLMs using Steer-Bench reveals that while human experts achieve an accuracy of 81% with silver labels, the best-performing models reach only around 65% accuracy depending on the domain and configuration. Some models lag behind human-level alignment by over 15 percentage points, highlighting significant gaps in community-sensitive steerability. Steer-Bench is a benchmark to systematically assess how effectively LLMs understand community-specific instructions, their resilience to adversarial steering attempts, and their ability to accurately represent diverse cultural and ideological perspectives. 

---
# REAL-Prover: Retrieval Augmented Lean Prover for Mathematical Reasoning 

**Authors**: Ziju Shen, Naohao Huang, Fanyi Yang, Yutong Wang, Guoxiong Gao, Tianyi Xu, Jiedong Jiang, Wanyi He, Pu Yang, Mengzhou Sun, Haocheng Ju, Peihao Wu, Bryan Dai, Bin Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.20613)  

**Abstract**: Nowadays, formal theorem provers have made monumental progress on high-school and competition-level mathematics, but few of them generalize to more advanced mathematics. In this paper, we present REAL-Prover, a new open-source stepwise theorem prover for Lean 4 to push this boundary. This prover, based on our fine-tuned large language model (REAL-Prover-v1) and integrated with a retrieval system (Leansearch-PS), notably boosts performance on solving college-level mathematics problems. To train REAL-Prover-v1, we developed HERALD-AF, a data extraction pipeline that converts natural language math problems into formal statements, and a new open-source Lean 4 interactive environment (Jixia-interactive) to facilitate synthesis data collection. In our experiments, our prover using only supervised fine-tune achieves competitive results with a 23.7% success rate (Pass@64) on the ProofNet dataset-comparable to state-of-the-art (SOTA) models. To further evaluate our approach, we introduce FATE-M, a new benchmark focused on algebraic problems, where our prover achieves a SOTA success rate of 56.7% (Pass@64). 

---
# Paths Not Taken: Understanding and Mending the Multilingual Factual Recall Pipeline 

**Authors**: Meng Lu, Ruochen Zhang, Ellie Pavlick, Carsten Eickhoff  

**Link**: [PDF](https://arxiv.org/pdf/2505.20546)  

**Abstract**: Multilingual large language models (LLMs) often exhibit factual inconsistencies across languages, with significantly better performance in factual recall tasks in English than in other languages. The causes of these failures, however, remain poorly understood. Using mechanistic analysis techniques, we uncover the underlying pipeline that LLMs employ, which involves using the English-centric factual recall mechanism to process multilingual queries and then translating English answers back into the target language. We identify two primary sources of error: insufficient engagement of the reliable English-centric mechanism for factual recall, and incorrect translation from English back into the target language for the final answer. To address these vulnerabilities, we introduce two vector interventions, both independent of languages and datasets, to redirect the model toward better internal paths for higher factual consistency. Our interventions combined increase the recall accuracy by over 35 percent for the lowest-performing language. Our findings demonstrate how mechanistic insights can be used to unlock latent multilingual capabilities in LLMs. 

---
# Beyond Keywords: Evaluating Large Language Model Classification of Nuanced Ableism 

**Authors**: Naba Rizvi, Harper Strickland, Saleha Ahmedi, Aekta Kallepalli, Isha Khirwadkar, William Wu, Imani N. S. Munyaka, Nedjma Ousidhoum  

**Link**: [PDF](https://arxiv.org/pdf/2505.20500)  

**Abstract**: Large language models (LLMs) are increasingly used in decision-making tasks like résumé screening and content moderation, giving them the power to amplify or suppress certain perspectives. While previous research has identified disability-related biases in LLMs, little is known about how they conceptualize ableism or detect it in text. We evaluate the ability of four LLMs to identify nuanced ableism directed at autistic individuals. We examine the gap between their understanding of relevant terminology and their effectiveness in recognizing ableist content in context. Our results reveal that LLMs can identify autism-related language but often miss harmful or offensive connotations. Further, we conduct a qualitative comparison of human and LLM explanations. We find that LLMs tend to rely on surface-level keyword matching, leading to context misinterpretations, in contrast to human annotators who consider context, speaker identity, and potential impact. On the other hand, both LLMs and humans agree on the annotation scheme, suggesting that a binary classification is adequate for evaluating LLM performance, which is consistent with findings from prior studies involving human annotators. 

---
# InFact: Informativeness Alignment for Improved LLM Factuality 

**Authors**: Roi Cohen, Russa Biswas, Gerard de Melo  

**Link**: [PDF](https://arxiv.org/pdf/2505.20487)  

**Abstract**: Factual completeness is a general term that captures how detailed and informative a factually correct text is. For instance, the factual sentence ``Barack Obama was born in the United States'' is factually correct, though less informative than the factual sentence ``Barack Obama was born in Honolulu, Hawaii, United States''. Despite the known fact that LLMs tend to hallucinate and generate factually incorrect text, they might also tend to choose to generate factual text that is indeed factually correct and yet less informative than other, more informative choices. In this work, we tackle this problem by proposing an informativeness alignment mechanism. This mechanism takes advantage of recent factual benchmarks to propose an informativeness alignment objective. This objective prioritizes answers that are both correct and informative. A key finding of our work is that when training a model to maximize this objective or optimize its preference, we can improve not just informativeness but also factuality. 

---
# AstroVisBench: A Code Benchmark for Scientific Computing and Visualization in Astronomy 

**Authors**: Sebastian Antony Joseph, Syed Murtaza Husain, Stella S. R. Offner, Stéphanie Juneau, Paul Torrey, Adam S. Bolton, Juan P. Farias, Niall Gaffney, Greg Durrett, Junyi Jessy Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20538)  

**Abstract**: Large Language Models (LLMs) are being explored for applications in scientific research, including their capabilities to synthesize literature, answer research questions, generate research ideas, and even conduct computational experiments. Ultimately, our goal is for these to help scientists derive novel scientific insights. In many areas of science, such insights often arise from processing and visualizing data to understand its patterns. However, evaluating whether an LLM-mediated scientific workflow produces outputs conveying the correct scientific insights is challenging to evaluate and has not been addressed in past work. We introduce AstroVisBench, the first benchmark for both scientific computing and visualization in the astronomy domain. AstroVisBench judges a language model's ability to both (1) create astronomy-specific workflows to process and analyze data and (2) visualize the results of these workflows through complex plots. Our evaluation of visualizations uses a novel LLM-as-a-judge workflow, which is validated against annotation by five professional astronomers. Using AstroVisBench we present an evaluation of state-of-the-art language models, showing a significant gap in their ability to engage in astronomy research as useful assistants. This evaluation provides a strong end-to-end evaluation for AI scientists that offers a path forward for the development of visualization-based workflows, which are central to a broad range of domains from physics to biology. 

---
# Amulet: Putting Complex Multi-Turn Conversations on the Stand with LLM Juries 

**Authors**: Sahana Ramnath, Anurag Mudgil, Brihi Joshi, Skyler Hallinan, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.20451)  

**Abstract**: Today, large language models are widely used as judges to evaluate responses from other language models. Hence, it is imperative to benchmark and improve these LLM-judges on real-world language model usage: a typical human-assistant conversation is lengthy, and shows significant diversity in topics, intents, and requirements across turns, e.g. social interactions, task requests, feedback. We present Amulet, a framework that leverages pertinent linguistic concepts of dialog-acts and maxims to improve the accuracy of LLM-judges on preference data with complex, multi-turn conversational context. Amulet presents valuable insights about (a) the communicative structures and intents present in the conversation (dialog acts), and (b) the satisfaction of conversational principles (maxims) by the preference responses, and uses them to make judgments. On four challenging datasets, Amulet shows that (a) humans frequently (60 to 70 percent of the time) change their intents from one turn of the conversation to the next, and (b) in 75 percent of instances, the preference responses can be differentiated via dialog acts and/or maxims, reiterating the latter's significance in judging such data. Amulet can be used either as a judge by applying the framework to a single LLM, or integrated into a jury with different LLM judges; our judges and juries show strong improvements on relevant baselines for all four datasets. 

---
# Large Language Models for IT Automation Tasks: Are We There Yet? 

**Authors**: Md Mahadi Hassan, John Salvador, Akond Rahman, Santu Karmaker  

**Link**: [PDF](https://arxiv.org/pdf/2505.20505)  

**Abstract**: LLMs show promise in code generation, yet their effectiveness for IT automation tasks, particularly for tools like Ansible, remains understudied. Existing benchmarks rely primarily on synthetic tasks that fail to capture the needs of practitioners who use IT automation tools, such as Ansible. We present ITAB (IT Automation Task Benchmark), a benchmark of 126 diverse tasks (e.g., configuring servers, managing files) where each task accounts for state reconciliation: a property unique to IT automation tools. ITAB evaluates LLMs' ability to generate functional Ansible automation scripts via dynamic execution in controlled environments. We evaluate 14 open-source LLMs, none of which accomplish pass@10 at a rate beyond 12%. To explain these low scores, we analyze 1,411 execution failures across the evaluated LLMs and identify two main categories of prevalent semantic errors: failures in state reconciliation related reasoning (44.87% combined from variable (11.43%), host (11.84%), path(11.63%), and template (9.97%) issues) and deficiencies in module-specific execution knowledge (24.37% combined from Attribute and parameter (14.44%) and module (9.93%) errors). Our findings reveal key limitations in open-source LLMs' ability to track state changes and apply specialized module knowledge, indicating that reliable IT automation will require major advances in state reasoning and domain-specific execution understanding. 

---
# In-context Language Learning for Endangered Languages in Speech Recognition 

**Authors**: Zhaolin Li, Jan Niehues  

**Link**: [PDF](https://arxiv.org/pdf/2505.20445)  

**Abstract**: With approximately 7,000 languages spoken worldwide, current large language models (LLMs) support only a small subset. Prior research indicates LLMs can learn new languages for certain tasks without supervised data. We extend this investigation to speech recognition, investigating whether LLMs can learn unseen, low-resource languages through in-context learning (ICL). With experiments on four diverse endangered languages that LLMs have not been trained on, we find that providing more relevant text samples enhances performance in both language modelling and Automatic Speech Recognition (ASR) tasks. Furthermore, we show that the probability-based approach outperforms the traditional instruction-based approach in language learning. Lastly, we show ICL enables LLMs to achieve ASR performance that is comparable to or even surpasses dedicated language models trained specifically for these languages, while preserving the original capabilities of the LLMs. 

---
# HAMburger: Accelerating LLM Inference via Token Smashing 

**Authors**: Jingyu Liu, Ce Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20438)  

**Abstract**: The growing demand for efficient Large Language Model (LLM) inference requires a holistic optimization on algorithms, systems, and hardware. However, very few works have fundamentally changed the generation pattern: each token needs one forward pass and one KV cache. This can be sub-optimal because we found that LLMs are extremely capable of self-identifying the exact dose of information that a single KV cache can store, and many tokens can be generated confidently without global context. Based on this insight, we introduce HAMburger, a Hierarchically Auto-regressive Model that redefines resource allocation in LLMs by moving beyond uniform computation and storage per token during inference. Stacking a compositional embedder and a micro-step decoder in between a base LLM, HAMburger smashes multiple tokens into a single KV and generates several tokens per step. Additionally, HAMburger functions as a speculative decoding framework where it can blindly trust self-drafted tokens. As a result, HAMburger shifts the growth of KV cache and forward FLOPs from linear to sub-linear with respect to output length, and adjusts its inference speed based on query perplexity and output structure. Extensive evaluations show that HAMburger reduces the KV cache computation by up to 2$\times$ and achieves up to 2$\times$ TPS, while maintaining quality in both short- and long-context tasks. Our method explores an extremely challenging inference regime that requires both computation- and memory-efficiency with a hardware-agnostic design. 

---
# GraphGen: Enhancing Supervised Fine-Tuning for LLMs with Knowledge-Driven Synthetic Data Generation 

**Authors**: Zihong Chen, Wanli Jiang, Jinzhe Li, Zhonghang Yuan, Huanjun Kong, Wanli Ouyang, Nanqing Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.20416)  

**Abstract**: Fine-tuning for large language models (LLMs) typically requires substantial amounts of high-quality supervised data, which is both costly and labor-intensive to acquire. While synthetic data generation has emerged as a promising solution, existing approaches frequently suffer from factual inaccuracies, insufficient long-tail coverage, simplistic knowledge structures, and homogenized outputs. To address these challenges, we introduce GraphGen, a knowledge graph-guided framework designed for three key question-answering (QA) scenarios: atomic QA, aggregated QA, and multi-hop QA. It begins by constructing a fine-grained knowledge graph from the source text. It then identifies knowledge gaps in LLMs using the expected calibration error metric, prioritizing the generation of QA pairs that target high-value, long-tail knowledge. Furthermore, GraphGen incorporates multi-hop neighborhood sampling to capture complex relational information and employs style-controlled generation to diversify the resulting QA data. Experimental results on knowledge-intensive tasks under closed-book settings demonstrate that GraphGen outperforms conventional synthetic data methods, offering a more reliable and comprehensive solution to the data scarcity challenge in supervised fine-tuning. The code and data are publicly available at this https URL. 

---
# Rethinking Text-based Protein Understanding: Retrieval or LLM? 

**Authors**: Juntong Wu, Zijing Liu, He Cao, Hao Li, Bin Feng, Zishan Shu, Ke Yu, Li Yuan, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20354)  

**Abstract**: In recent years, protein-text models have gained significant attention for their potential in protein generation and understanding. Current approaches focus on integrating protein-related knowledge into large language models through continued pretraining and multi-modal alignment, enabling simultaneous comprehension of textual descriptions and protein sequences. Through a thorough analysis of existing model architectures and text-based protein understanding benchmarks, we identify significant data leakage issues present in current benchmarks. Moreover, conventional metrics derived from natural language processing fail to accurately assess the model's performance in this domain. To address these limitations, we reorganize existing datasets and introduce a novel evaluation framework based on biological entities. Motivated by our observation, we propose a retrieval-enhanced method, which significantly outperforms fine-tuned LLMs for protein-to-text generation and shows accuracy and efficiency in training-free scenarios. Our code and data can be seen at this https URL. 

---
# Enhancing Logical Reasoning in Language Models via Symbolically-Guided Monte Carlo Process Supervision 

**Authors**: Xingwei Tan, Marco Valentino, Mahmud Akhter, Maria Liakata, Nikolaos Aletras  

**Link**: [PDF](https://arxiv.org/pdf/2505.20415)  

**Abstract**: Large language models (LLMs) have shown promising performance in mathematical and logical reasoning benchmarks. However, recent studies have pointed to memorization, rather than generalization, as one of the leading causes for such performance. LLMs, in fact, are susceptible to content variations, demonstrating a lack of robust symbolic abstractions supporting their reasoning process. To improve reliability, many attempts have been made to combine LLMs with symbolic methods. Nevertheless, existing approaches fail to effectively leverage symbolic representations due to the challenges involved in developing reliable and scalable verification mechanisms. In this paper, we propose to overcome such limitations by generating symbolic reasoning trajectories and select the high-quality ones using a process reward model automatically tuned based on Monte Carlo estimation. The trajectories are then employed via fine-tuning methods to improve logical reasoning and generalization. Our results on logical reasoning benchmarks such as FOLIO and LogicAsker show the effectiveness of the proposed method with large gains on frontier and open-weight models. Moreover, additional experiments on claim verification reveal that fine-tuning on the generated symbolic reasoning trajectories enhances out-of-domain generalizability, suggesting the potential impact of symbolically-guided process supervision in alleviating the effect of memorization on LLM reasoning. 

---
# Assessing the Capability of LLMs in Solving POSCOMP Questions 

**Authors**: Cayo Viegas, Rohit Gheyi, Márcio Ribeiro  

**Link**: [PDF](https://arxiv.org/pdf/2505.20338)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have significantly expanded the capabilities of artificial intelligence in natural language processing tasks. Despite this progress, their performance in specialized domains such as computer science remains relatively unexplored. Understanding the proficiency of LLMs in these domains is critical for evaluating their practical utility and guiding future developments. The POSCOMP, a prestigious Brazilian examination used for graduate admissions in computer science promoted by the Brazlian Computer Society (SBC), provides a challenging benchmark. This study investigates whether LLMs can match or surpass human performance on the POSCOMP exam. Four LLMs - ChatGPT-4, Gemini 1.0 Advanced, Claude 3 Sonnet, and Le Chat Mistral Large - were initially evaluated on the 2022 and 2023 POSCOMP exams. The assessments measured the models' proficiency in handling complex questions typical of the exam. LLM performance was notably better on text-based questions than on image interpretation tasks. In the 2022 exam, ChatGPT-4 led with 57 correct answers out of 69 questions, followed by Gemini 1.0 Advanced (49), Le Chat Mistral (48), and Claude 3 Sonnet (44). Similar trends were observed in the 2023 exam. ChatGPT-4 achieved the highest performance, surpassing all students who took the POSCOMP 2023 exam. LLMs, particularly ChatGPT-4, show promise in text-based tasks on the POSCOMP exam, although image interpretation remains a challenge. Given the rapid evolution of LLMs, we expanded our analysis to include more recent models - o1, Gemini 2.5 Pro, Claude 3.7 Sonnet, and o3-mini-high - evaluated on the 2022-2024 POSCOMP exams. These newer models demonstrate further improvements and consistently surpass both the average and top-performing human participants across all three years. 

---
# SeRL: Self-Play Reinforcement Learning for Large Language Models with Limited Data 

**Authors**: Wenkai Fang, Shunyu Liu, Yang Zhou, Kongcheng Zhang, Tongya Zheng, Kaixuan Chen, Mingli Song, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.20347)  

**Abstract**: Recent advances have demonstrated the effectiveness of Reinforcement Learning (RL) in improving the reasoning capabilities of Large Language Models (LLMs). However, existing works inevitably rely on high-quality instructions and verifiable rewards for effective training, both of which are often difficult to obtain in specialized domains. In this paper, we propose Self-play Reinforcement Learning(SeRL) to bootstrap LLM training with limited initial data. Specifically, SeRL comprises two complementary modules: self-instruction and self-rewarding. The former module generates additional instructions based on the available data at each training step, employing robust online filtering strategies to ensure instruction quality, diversity, and difficulty. The latter module introduces a simple yet effective majority-voting mechanism to estimate response rewards for additional instructions, eliminating the need for external annotations. Finally, SeRL performs conventional RL based on the generated data, facilitating iterative self-play learning. Extensive experiments on various reasoning benchmarks and across different LLM backbones demonstrate that the proposed SeRL yields results superior to its counterparts and achieves performance on par with those obtained by high-quality data with verifiable rewards. Our code is available at this https URL. 

---
# MOSLIM:Align with diverse preferences in prompts through reward classification 

**Authors**: Yu Zhang, Wanli Jiang, Zhengyu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20336)  

**Abstract**: The multi-objective alignment of Large Language Models (LLMs) is essential for ensuring foundational models conform to diverse human preferences. Current research in this field typically involves either multiple policies or multiple reward models customized for various preferences, or the need to train a preference-specific supervised fine-tuning (SFT) model. In this work, we introduce a novel multi-objective alignment method, MOSLIM, which utilizes a single reward model and policy model to address diverse objectives. MOSLIM provides a flexible way to control these objectives through prompting and does not require preference training during SFT phase, allowing thousands of off-the-shelf models to be directly utilized within this training framework. MOSLIM leverages a multi-head reward model that classifies question-answer pairs instead of scoring them and then optimize policy model with a scalar reward derived from a mapping function that converts classification results from reward model into reward scores. We demonstrate the efficacy of our proposed method across several multi-objective benchmarks and conduct ablation studies on various reward model sizes and policy optimization methods. The MOSLIM method outperforms current multi-objective approaches in most results while requiring significantly fewer GPU computing resources compared with existing policy optimization methods. 

---
# Multi-Scale Manifold Alignment: A Unified Framework for Enhanced Explainability of Large Language Models 

**Authors**: Yukun Zhang, Qi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.20333)  

**Abstract**: Recent advances in Large Language Models (LLMs) have achieved strong performance, yet their internal reasoning remains opaque, limiting interpretability and trust in critical applications. We propose a novel Multi_Scale Manifold Alignment framework that decomposes the latent space into global, intermediate, and local semantic manifolds capturing themes, context, and word-level details. Our method introduces cross_scale mapping functions that jointly enforce geometric alignment (e.g., Procrustes analysis) and information preservation (via mutual information constraints like MINE or VIB). We further incorporate curvature regularization and hyperparameter tuning for stable optimization. Theoretical analysis shows that alignment error, measured by KL divergence, can be bounded under mild assumptions. This framework offers a unified explanation of how LLMs structure multi-scale semantics, advancing interpretability and enabling applications such as bias detection and robustness enhancement. 

---
# SEMMA: A Semantic Aware Knowledge Graph Foundation Model 

**Authors**: Arvindh Arun, Sumit Kumar, Mojtaba Nayyeri, Bo Xiong, Ponnurangam Kumaraguru, Antonio Vergari, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2505.20422)  

**Abstract**: Knowledge Graph Foundation Models (KGFMs) have shown promise in enabling zero-shot reasoning over unseen graphs by learning transferable patterns. However, most existing KGFMs rely solely on graph structure, overlooking the rich semantic signals encoded in textual attributes. We introduce SEMMA, a dual-module KGFM that systematically integrates transferable textual semantics alongside structure. SEMMA leverages Large Language Models (LLMs) to enrich relation identifiers, generating semantic embeddings that subsequently form a textual relation graph, which is fused with the structural component. Across 54 diverse KGs, SEMMA outperforms purely structural baselines like ULTRA in fully inductive link prediction. Crucially, we show that in more challenging generalization settings, where the test-time relation vocabulary is entirely unseen, structural methods collapse while SEMMA is 2x more effective. Our findings demonstrate that textual semantics are critical for generalization in settings where structure alone fails, highlighting the need for foundation models that unify structural and linguistic signals in knowledge reasoning. 

---
# PMOA-TTS: Introducing the PubMed Open Access Textual Times Series Corpus 

**Authors**: Shahriar Noroozizadeh, Sayantan Kumar, George H. Chen, Jeremy C. Weiss  

**Link**: [PDF](https://arxiv.org/pdf/2505.20323)  

**Abstract**: Understanding temporal dynamics in clinical narratives is essential for modeling patient trajectories, yet large-scale temporally annotated resources remain limited. We present PMOA-TTS, the first openly available dataset of 124,699 PubMed Open Access (PMOA) case reports, each converted into structured (event, time) timelines via a scalable LLM-based pipeline. Our approach combines heuristic filtering with Llama 3.3 to identify single-patient case reports, followed by prompt-driven extraction using Llama 3.3 and DeepSeek R1, resulting in over 5.6 million timestamped clinical events. To assess timeline quality, we evaluate against a clinician-curated reference set using three metrics: (i) event-level matching (80% match at a cosine similarity threshold of 0.1), (ii) temporal concordance (c-index > 0.90), and (iii) Area Under the Log-Time CDF (AULTC) for timestamp alignment. Corpus-level analysis shows wide diagnostic and demographic coverage. In a downstream survival prediction task, embeddings from extracted timelines achieve time-dependent concordance indices up to 0.82 $\pm$ 0.01, demonstrating the predictive value of temporally structured narratives. PMOA-TTS provides a scalable foundation for timeline extraction, temporal reasoning, and longitudinal modeling in biomedical NLP. The dataset is available at: this https URL . 

---
# Guiding Giants: Lightweight Controllers for Weighted Activation Steering in LLMs 

**Authors**: Amr Hegazy, Mostafa Elhoushi, Amr Alanwar  

**Link**: [PDF](https://arxiv.org/pdf/2505.20309)  

**Abstract**: Controlling undesirable Large Language Model (LLM) behaviors, such as the generation of unsafe content or failing to adhere to safety guidelines, often relies on costly fine-tuning. Activation steering provides an alternative for inference-time control, but existing methods typically lack fine-grained, adaptive mechanisms. We introduce a novel approach using a lightweight, trainable controller network integrated during inference. This controller network observes specific intermediate LLM activations and predicts both a global scaling factor and layer-specific weights. The predicted global scaling factor and layer-specific weights then dynamically modulate the intensity of a steering patch, derived from a pre-computed "refusal direction" vector, applied across the LLM's layers during generation. Trained on activations from both harmful and benign prompts, our controller learns to discriminatively apply nuanced, layer-aware interventions, activating steering primarily for harmful inputs. Experiments using safety benchmarks like ToxicChat & In-The-Wild Jailbreak Prompts demonstrate that our weighted steering controller significantly increases refusal rates compared to the base LLM, achieving targeted behavioral modification without altering the original model parameters. Our experiments with Llama-3.1-8B, Llama-3.2-1B & Mistral-7B show our approach outperforms existing methods, presenting an efficient and adaptive method for fine-grained control over LLM behavior at inference time. 

---
# Dynamic Manifold Evolution Theory: Modeling and Stability Analysis of Latent Representations in Large Language Models 

**Authors**: Yukun Zhang, Qi Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.20340)  

**Abstract**: We introduce Dynamic Manifold Evolution Theory (DMET),a unified framework that models large language model generation as a controlled dynamical system evolving on a low_dimensional semantic manifold. By casting latent_state updates as discrete time Euler approximations of continuous dynamics, we map intrinsic energy_driven flows and context_dependent forces onto Transformer components (residual connections, attention, feed-forward networks). Leveraging Lyapunov stability theory We define three empirical metrics (state continuity, clustering quality, topological persistence) that quantitatively link latent_trajectory properties to text fluency, grammaticality, and semantic coherence. Extensive experiments across decoding parameters validate DMET's predictions and yield principled guidelines for balancing creativity and consistency in text generation. 

---
# Do LLMs have a Gender (Entropy) Bias? 

**Authors**: Sonal Prabhune, Balaji Padmanabhan, Kaushik Dutta  

**Link**: [PDF](https://arxiv.org/pdf/2505.20343)  

**Abstract**: We investigate the existence and persistence of a specific type of gender bias in some of the popular LLMs and contribute a new benchmark dataset, RealWorldQuestioning (released on HuggingFace ), developed from real-world questions across four key domains in business and health contexts: education, jobs, personal financial management, and general health. We define and study entropy bias, which we define as a discrepancy in the amount of information generated by an LLM in response to real questions users have asked. We tested this using four different LLMs and evaluated the generated responses both qualitatively and quantitatively by using ChatGPT-4o (as "LLM-as-judge"). Our analyses (metric-based comparisons and "LLM-as-judge" evaluation) suggest that there is no significant bias in LLM responses for men and women at a category level. However, at a finer granularity (the individual question level), there are substantial differences in LLM responses for men and women in the majority of cases, which "cancel" each other out often due to some responses being better for males and vice versa. This is still a concern since typical users of these tools often ask a specific question (only) as opposed to several varied ones in each of these common yet important areas of life. We suggest a simple debiasing approach that iteratively merges the responses for the two genders to produce a final result. Our approach demonstrates that a simple, prompt-based debiasing strategy can effectively debias LLM outputs, thus producing responses with higher information content than both gendered variants in 78% of the cases, and consistently achieving a balanced integration in the remaining cases. 

---
# Guided by Gut: Efficient Test-Time Scaling with Reinforced Intrinsic Confidence 

**Authors**: Amirhosein Ghasemabadi, Keith G. Mills, Baochun Li, Di Niu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20325)  

**Abstract**: Test-Time Scaling (TTS) methods for enhancing Large Language Model (LLM) reasoning often incur substantial computational costs, primarily due to extensive reliance on external Process Reward Models (PRMs) or sampling methods like Best-of-N (BoN). This paper introduces Guided by Gut (GG), an efficient self-guided TTS framework that achieves PRM-level performance without costly external verifier models. Our method employs a lightweight tree search guided solely by intrinsic LLM signals, token-level confidence and step novelty. One critical innovation is improving the reliability of internal confidence estimates via a targeted reinforcement learning fine-tuning phase. Empirical evaluations on challenging mathematical reasoning benchmarks demonstrate that GG enables smaller models (e.g., 1.5B parameters) to achieve accuracy matching or surpassing significantly larger models (e.g., 32B-70B parameters), while reducing GPU memory usage by up to 10x. Compared to PRM-based methods, GG achieves comparable accuracy with 8x faster inference speeds and 4-5x lower memory usage. Additionally, GG reduces KV cache memory usage by approximately 50% compared to the BoN strategy, facilitating more efficient and practical deployment of TTS techniques. 

---
# PoisonSwarm: Universal Harmful Information Synthesis via Model Crowdsourcing 

**Authors**: Yu Yan, Sheng Sun, Zhifei Zheng, Ziji Hao, Teli Liu, Min Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21184)  

**Abstract**: To construct responsible and secure AI applications, harmful information data is widely utilized for adversarial testing and the development of safeguards. Existing studies mainly leverage Large Language Models (LLMs) to synthesize data to obtain high-quality task datasets at scale, thereby avoiding costly human annotation. However, limited by the safety alignment mechanisms of LLMs, the synthesis of harmful data still faces challenges in generation reliability and content diversity. In this study, we propose a novel harmful information synthesis framework, PoisonSwarm, which applies the model crowdsourcing strategy to generate diverse harmful data while maintaining a high success rate. Specifically, we generate abundant benign data as the based templates in a counterfactual manner. Subsequently, we decompose each based template into multiple semantic units and perform unit-by-unit toxification and final refinement through dynamic model switching, thus ensuring the success of synthesis. Experimental results demonstrate that PoisonSwarm achieves state-of-the-art performance in synthesizing different categories of harmful data with high scalability and diversity. 

---
# Beyond Demonstrations: Dynamic Vector Construction from Latent Representations 

**Authors**: Wang Cai, Hsiu-Yuan Huang, Zhixiang Wang, Yunfang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20318)  

**Abstract**: In-Context derived Vector (ICV) methods extract task-relevant representations from large language models (LLMs) and reinject them during inference, achieving comparable performance to few-shot In-Context Learning (ICL) without repeated demonstration processing. However, existing ICV methods remain sensitive to ICL-specific factors, often use coarse or semantically fragmented representations as the source of the vector, and rely on heuristic-based injection positions, limiting their applicability.
To address these issues, we propose Dynamic Vector (DyVec), which incorporates an Exhaustive Query Rotation (EQR) strategy to extract robust semantically aggregated latent representations by mitigating variance introduced by ICL. It then applies Dynamic Latent Segmentation and Injection to adaptively partition representations based on task complexity and leverages REINFORCE-based optimization to learn optimal injection positions for each segment.
Experiments results show that DyVec outperforms few-shot ICL, LoRA, and prior ICV baselines. Further analysis highlights the effectiveness of dynamically segmenting and injecting semantically aggregated latent representations. DyVec provides a lightweight and data-efficient solution for inference-time task adaptation. 

---
# Creativity in LLM-based Multi-Agent Systems: A Survey 

**Authors**: Yi-Cheng Lin, Kang-Chieh Chen, Zhe-Yan Li, Tzu-Heng Wu, Tzu-Hsuan Wu, Kuan-Yu Chen, Hung-yi Lee, Yun-Nung Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21116)  

**Abstract**: Large language model (LLM)-driven multi-agent systems (MAS) are transforming how humans and AIs collaboratively generate ideas and artifacts. While existing surveys provide comprehensive overviews of MAS infrastructures, they largely overlook the dimension of \emph{creativity}, including how novel outputs are generated and evaluated, how creativity informs agent personas, and how creative workflows are coordinated. This is the first survey dedicated to creativity in MAS. We focus on text and image generation tasks, and present: (1) a taxonomy of agent proactivity and persona design; (2) an overview of generation techniques, including divergent exploration, iterative refinement, and collaborative synthesis, as well as relevant datasets and evaluation metrics; and (3) a discussion of key challenges, such as inconsistent evaluation standards, insufficient bias mitigation, coordination conflicts, and the lack of unified benchmarks. This survey offers a structured framework and roadmap for advancing the development, evaluation, and standardization of creative MAS. 

---
# Position is Power: System Prompts as a Mechanism of Bias in Large Language Models (LLMs) 

**Authors**: Anna Neumann, Elisabeth Kirsten, Muhammad Bilal Zafar, Jatinder Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.21091)  

**Abstract**: System prompts in Large Language Models (LLMs) are predefined directives that guide model behaviour, taking precedence over user inputs in text processing and generation. LLM deployers increasingly use them to ensure consistent responses across contexts. While model providers set a foundation of system prompts, deployers and third-party developers can append additional prompts without visibility into others' additions, while this layered implementation remains entirely hidden from end-users. As system prompts become more complex, they can directly or indirectly introduce unaccounted for side effects. This lack of transparency raises fundamental questions about how the position of information in different directives shapes model outputs. As such, this work examines how the placement of information affects model behaviour. To this end, we compare how models process demographic information in system versus user prompts across six commercially available LLMs and 50 demographic groups. Our analysis reveals significant biases, manifesting in differences in user representation and decision-making scenarios. Since these variations stem from inaccessible and opaque system-level configurations, they risk representational, allocative and potential other biases and downstream harms beyond the user's ability to detect or correct. Our findings draw attention to these critical issues, which have the potential to perpetuate harms if left unexamined. Further, we argue that system prompt analysis must be incorporated into AI auditing processes, particularly as customisable system prompts become increasingly prevalent in commercial AI deployments. 

---
# An LLM-as-Judge Metric for Bridging the Gap with Human Evaluation in SE Tasks 

**Authors**: Xin Zhou, Kisub Kim, Ting Zhang, Martin Weyssow, Luis F. Gomes, Guang Yang, David Lo  

**Link**: [PDF](https://arxiv.org/pdf/2505.20854)  

**Abstract**: Large Language Models (LLMs) and other automated techniques have been increasingly used to support software developers by generating software artifacts such as code snippets, patches, and comments. However, accurately assessing the correctness of these generated artifacts remains a significant challenge. On one hand, human evaluation provides high accuracy but is labor-intensive and lacks scalability. On the other hand, other existing automatic evaluation metrics are scalable and require minimal human effort, but they often fail to accurately reflect the actual correctness of generated software artifacts.
In this paper, we present SWE-Judge, the first evaluation metric for LLM-as-Ensemble-Judge specifically designed to accurately assess the correctness of generated software artifacts. SWE-Judge first defines five distinct evaluation strategies, each implemented as an independent judge. A dynamic team selection mechanism then identifies the most appropriate subset of judges to produce a final correctness score through ensembling. We evaluate SWE-Judge across a diverse set of software engineering (SE) benchmarks, including CoNaLa, Card2Code, HumanEval-X, APPS, APR-Assess, and Summary-Assess. These benchmarks span three SE tasks: code generation, automated program repair, and code summarization. Experimental results demonstrate that SWE-Judge consistently achieves a higher correlation with human judgments, with improvements ranging from 5.9% to 183.8% over existing automatic metrics. Furthermore, SWE-Judge reaches agreement levels with human annotators that are comparable to inter-annotator agreement in code generation and program repair tasks. These findings underscore SWE-Judge's potential as a scalable and reliable alternative to human evaluation. 

---
# Less Context, Same Performance: A RAG Framework for Resource-Efficient LLM-Based Clinical NLP 

**Authors**: Satya Narayana Cheetirala, Ganesh Raut, Dhavalkumar Patel, Fabio Sanatana, Robert Freeman, Matthew A Levin, Girish N. Nadkarni, Omar Dawkins, Reba Miller, Randolph M. Steinhagen, Eyal Klang, Prem Timsina  

**Link**: [PDF](https://arxiv.org/pdf/2505.20320)  

**Abstract**: Long text classification is challenging for Large Language Models (LLMs) due to token limits and high computational costs. This study explores whether a Retrieval Augmented Generation (RAG) approach using only the most relevant text segments can match the performance of processing entire clinical notes with large context LLMs. We begin by splitting clinical documents into smaller chunks, converting them into vector embeddings, and storing these in a FAISS index. We then retrieve the top 4,000 words most pertinent to the classification query and feed these consolidated segments into an LLM. We evaluated three LLMs (GPT4o, LLaMA, and Mistral) on a surgical complication identification task. Metrics such as AUC ROC, precision, recall, and F1 showed no statistically significant differences between the RAG based approach and whole-text processing (p > 0.05p > 0.05). These findings indicate that RAG can significantly reduce token usage without sacrificing classification accuracy, providing a scalable and cost effective solution for analyzing lengthy clinical documents. 

---
# SV-TrustEval-C: Evaluating Structure and Semantic Reasoning in Large Language Models for Source Code Vulnerability Analysis 

**Authors**: Yansong Li, Paula Branco, Alexander M. Hoole, Manish Marwah, Hari Manassery Koduvely, Guy-Vincent Jourdan, Stephan Jou  

**Link**: [PDF](https://arxiv.org/pdf/2505.20630)  

**Abstract**: As Large Language Models (LLMs) evolve in understanding and generating code, accurately evaluating their reliability in analyzing source code vulnerabilities becomes increasingly vital. While studies have examined LLM capabilities in tasks like vulnerability detection and repair, they often overlook the importance of both structure and semantic reasoning crucial for trustworthy vulnerability analysis. To address this gap, we introduce SV-TrustEval-C, a benchmark designed to evaluate LLMs' abilities for vulnerability analysis of code written in the C programming language through two key dimensions: structure reasoning - assessing how models identify relationships between code elements under varying data and control flow complexities; and semantic reasoning - examining their logical consistency in scenarios where code is structurally and semantically perturbed. Our results show that current LLMs are far from satisfactory in understanding complex code relationships and that their vulnerability analyses rely more on pattern matching than on robust logical reasoning. These findings underscore the effectiveness of the SV-TrustEval-C benchmark and highlight critical areas for enhancing the reasoning capabilities and trustworthiness of LLMs in real-world vulnerability analysis tasks. Our initial benchmark dataset is publicly available. 

---
# Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning 

**Authors**: Shenao Zhang, Yaqing Wang, Yinxiao Liu, Tianqi Liu, Peter Grabowski, Eugene Ie, Zhaoran Wang, Yunxuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20561)  

**Abstract**: Large Language Models (LLMs) trained via Reinforcement Learning (RL) have exhibited strong reasoning capabilities and emergent reflective behaviors, such as backtracking and error correction. However, conventional Markovian RL confines exploration to the training phase to learn an optimal deterministic policy and depends on the history contexts only through the current state. Therefore, it remains unclear whether reflective reasoning will emerge during Markovian RL training, or why they are beneficial at test time. To remedy this, we recast reflective exploration within the Bayes-Adaptive RL framework, which explicitly optimizes the expected return under a posterior distribution over Markov decision processes. This Bayesian formulation inherently incentivizes both reward-maximizing exploitation and information-gathering exploration via belief updates. Our resulting algorithm, BARL, instructs the LLM to stitch and switch strategies based on the observed outcomes, offering principled guidance on when and how the model should reflectively explore. Empirical results on both synthetic and mathematical reasoning tasks demonstrate that BARL outperforms standard Markovian RL approaches at test time, achieving superior token efficiency with improved exploration effectiveness. Our code is available at this https URL. 

---
# Cross from Left to Right Brain: Adaptive Text Dreamer for Vision-and-Language Navigation 

**Authors**: Pingrui Zhang, Yifei Su, Pengyuan Wu, Dong An, Li Zhang, Zhigang Wang, Dong Wang, Yan Ding, Bin Zhao, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20897)  

**Abstract**: Vision-and-Language Navigation (VLN) requires the agent to navigate by following natural instructions under partial observability, making it difficult to align perception with language. Recent methods mitigate this by imagining future scenes, yet they rely on vision-based synthesis, leading to high computational cost and redundant details. To this end, we propose to adaptively imagine key environmental semantics via \textit{language} form, enabling a more reliable and efficient strategy. Specifically, we introduce a novel Adaptive Text Dreamer (ATD), a dual-branch self-guided imagination policy built upon a large language model (LLM). ATD is designed with a human-like left-right brain architecture, where the left brain focuses on logical integration, and the right brain is responsible for imaginative prediction of future scenes. To achieve this, we fine-tune only the Q-former within both brains to efficiently activate domain-specific knowledge in the LLM, enabling dynamic updates of logical reasoning and imagination during navigation. Furthermore, we introduce a cross-interaction mechanism to regularize the imagined outputs and inject them into a navigation expert module, allowing ATD to jointly exploit both the reasoning capacity of the LLM and the expertise of the navigation model. We conduct extensive experiments on the R2R benchmark, where ATD achieves state-of-the-art performance with fewer parameters. The code is \href{this https URL}{here}. 

---
# Comparisons between a Large Language Model-based Real-Time Compound Diagnostic Medical AI Interface and Physicians for Common Internal Medicine Cases using Simulated Patients 

**Authors**: Hyungjun Park, Chang-Yun Woo, Seungjo Lim, Seunghwan Lim, Keunho Kwak, Ju Young Jeong, Chong Hyun Suh  

**Link**: [PDF](https://arxiv.org/pdf/2505.20609)  

**Abstract**: Objective To develop an LLM based realtime compound diagnostic medical AI interface and performed a clinical trial comparing this interface and physicians for common internal medicine cases based on the United States Medical License Exam (USMLE) Step 2 Clinical Skill (CS) style exams. Methods A nonrandomized clinical trial was conducted on August 20, 2024. We recruited one general physician, two internal medicine residents (2nd and 3rd year), and five simulated patients. The clinical vignettes were adapted from the USMLE Step 2 CS style exams. We developed 10 representative internal medicine cases based on actual patients and included information available on initial diagnostic evaluation. Primary outcome was the accuracy of the first differential diagnosis. Repeatability was evaluated based on the proportion of agreement. Results The accuracy of the physicians' first differential diagnosis ranged from 50% to 70%, whereas the realtime compound diagnostic medical AI interface achieved an accuracy of 80%. The proportion of agreement for the first differential diagnosis was 0.7. The accuracy of the first and second differential diagnoses ranged from 70% to 90% for physicians, whereas the AI interface achieved an accuracy rate of 100%. The average time for the AI interface (557 sec) was 44.6% shorter than that of the physicians (1006 sec). The AI interface ($0.08) also reduced costs by 98.1% compared to the physicians' average ($4.2). Patient satisfaction scores ranged from 4.2 to 4.3 for care by physicians and were 3.9 for the AI interface Conclusion An LLM based realtime compound diagnostic medical AI interface demonstrated diagnostic accuracy and patient satisfaction comparable to those of a physician, while requiring less time and lower costs. These findings suggest that AI interfaces may have the potential to assist primary care consultations for common internal medicine cases. 

---
# What Changed? Detecting and Evaluating Instruction-Guided Image Edits with Multimodal Large Language Models 

**Authors**: Lorenzo Baraldi, Davide Bucciarelli, Federico Betti, Marcella Cornia, Lorenzo Baraldi, Nicu Sebe, Rita Cucchiara  

**Link**: [PDF](https://arxiv.org/pdf/2505.20405)  

**Abstract**: Instruction-based image editing models offer increased personalization opportunities in generative tasks. However, properly evaluating their results is challenging, and most of the existing metrics lag in terms of alignment with human judgment and explainability. To tackle these issues, we introduce DICE (DIfference Coherence Estimator), a model designed to detect localized differences between the original and the edited image and to assess their relevance to the given modification request. DICE consists of two key components: a difference detector and a coherence estimator, both built on an autoregressive Multimodal Large Language Model (MLLM) and trained using a strategy that leverages self-supervision, distillation from inpainting networks, and full supervision. Through extensive experiments, we evaluate each stage of our pipeline, comparing different MLLMs within the proposed framework. We demonstrate that DICE effectively identifies coherent edits, effectively evaluating images generated by different editing models with a strong correlation with human judgment. We publicly release our source code, models, and data. 

---
# Project Riley: Multimodal Multi-Agent LLM Collaboration with Emotional Reasoning and Voting 

**Authors**: Ana Rita Ortigoso, Gabriel Vieira, Daniel Fuentes, Luis Frazão, Nuno Costa, António Pereira  

**Link**: [PDF](https://arxiv.org/pdf/2505.20521)  

**Abstract**: This paper presents Project Riley, a novel multimodal and multi-model conversational AI architecture oriented towards the simulation of reasoning influenced by emotional states. Drawing inspiration from Pixar's Inside Out, the system comprises five distinct emotional agents - Joy, Sadness, Fear, Anger, and Disgust - that engage in structured multi-round dialogues to generate, criticise, and iteratively refine responses. A final reasoning mechanism synthesises the contributions of these agents into a coherent output that either reflects the dominant emotion or integrates multiple perspectives. The architecture incorporates both textual and visual large language models (LLMs), alongside advanced reasoning and self-refinement processes. A functional prototype was deployed locally in an offline environment, optimised for emotional expressiveness and computational efficiency. From this initial prototype, another one emerged, called Armando, which was developed for use in emergency contexts, delivering emotionally calibrated and factually accurate information through the integration of Retrieval-Augmented Generation (RAG) and cumulative context tracking. The Project Riley prototype was evaluated through user testing, in which participants interacted with the chatbot and completed a structured questionnaire assessing three dimensions: Emotional Appropriateness, Clarity and Utility, and Naturalness and Human-likeness. The results indicate strong performance in structured scenarios, particularly with respect to emotional alignment and communicative clarity. 

---
# Robust Hypothesis Generation: LLM-Automated Language Bias for Inductive Logic Programming 

**Authors**: Yang Yang, Jiemin Wu, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2505.21486)  

**Abstract**: Automating robust hypothesis generation in open environments is pivotal for AI cognition. We introduce a novel framework integrating a multi-agent system, powered by Large Language Models (LLMs), with Inductive Logic Programming (ILP). Our system's LLM agents autonomously define a structured symbolic vocabulary (predicates) and relational templates , i.e., \emph{language bias} directly from raw textual data. This automated symbolic grounding (the construction of the language bias), traditionally an expert-driven bottleneck for ILP, then guides the transformation of text into facts for an ILP solver, which inductively learns interpretable rules. This approach overcomes traditional ILP's reliance on predefined symbolic structures and the noise-sensitivity of pure LLM methods. Extensive experiments in diverse, challenging scenarios validate superior performance, paving a new path for automated, explainable, and verifiable hypothesis generation. 

---
# MME-Reasoning: A Comprehensive Benchmark for Logical Reasoning in MLLMs 

**Authors**: Jiakang Yuan, Tianshuo Peng, Yilei Jiang, Yiting Lu, Renrui Zhang, Kaituo Feng, Chaoyou Fu, Tao Chen, Lei Bai, Bo Zhang, Xiangyu Yue  

**Link**: [PDF](https://arxiv.org/pdf/2505.21327)  

**Abstract**: Logical reasoning is a fundamental aspect of human intelligence and an essential capability for multimodal large language models (MLLMs). Despite the significant advancement in multimodal reasoning, existing benchmarks fail to comprehensively evaluate their reasoning abilities due to the lack of explicit categorization for logical reasoning types and an unclear understanding of reasoning. To address these issues, we introduce MME-Reasoning, a comprehensive benchmark designed to evaluate the reasoning ability of MLLMs, which covers all three types of reasoning (i.e., inductive, deductive, and abductive) in its questions. We carefully curate the data to ensure that each question effectively evaluates reasoning ability rather than perceptual skills or knowledge breadth, and extend the evaluation protocols to cover the evaluation of diverse questions. Our evaluation reveals substantial limitations of state-of-the-art MLLMs when subjected to holistic assessments of logical reasoning capabilities. Even the most advanced MLLMs show limited performance in comprehensive logical reasoning, with notable performance imbalances across reasoning types. In addition, we conducted an in-depth analysis of approaches such as ``thinking mode'' and Rule-based RL, which are commonly believed to enhance reasoning abilities. These findings highlight the critical limitations and performance imbalances of current MLLMs in diverse logical reasoning scenarios, providing comprehensive and systematic insights into the understanding and evaluation of reasoning capabilities. 

---
# Policy Induction: Predicting Startup Success via Explainable Memory-Augmented In-Context Learning 

**Authors**: Xianling Mu, Joseph Ternasky, Fuat Alican, Yigit Ihlamur  

**Link**: [PDF](https://arxiv.org/pdf/2505.21427)  

**Abstract**: Early-stage startup investment is a high-risk endeavor characterized by scarce data and uncertain outcomes. Traditional machine learning approaches often require large, labeled datasets and extensive fine-tuning, yet remain opaque and difficult for domain experts to interpret or improve. In this paper, we propose a transparent and data-efficient investment decision framework powered by memory-augmented large language models (LLMs) using in-context learning (ICL). Central to our method is a natural language policy embedded directly into the LLM prompt, enabling the model to apply explicit reasoning patterns and allowing human experts to easily interpret, audit, and iteratively refine the logic. We introduce a lightweight training process that combines few-shot learning with an in-context learning loop, enabling the LLM to update its decision policy iteratively based on structured feedback. With only minimal supervision and no gradient-based optimization, our system predicts startup success far more accurately than existing benchmarks. It is over 20x more precise than random chance, which succeeds 1.9% of the time. It is also 7.1x more precise than the typical 5.6% success rate of top-tier venture capital (VC) firms. 

---
# Beyond Chemical QA: Evaluating LLM's Chemical Reasoning with Modular Chemical Operations 

**Authors**: Hao Li, He Cao, Bin Feng, Yanjun Shao, Xiangru Tang, Zhiyuan Yan, Li Yuan, Yonghong Tian, Yu Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.21318)  

**Abstract**: While large language models (LLMs) with Chain-of-Thought (CoT) reasoning excel in mathematics and coding, their potential for systematic reasoning in chemistry, a domain demanding rigorous structural analysis for real-world tasks like drug design and reaction engineering, remains untapped. Current benchmarks focus on simple knowledge retrieval, neglecting step-by-step reasoning required for complex tasks such as molecular optimization and reaction prediction. To address this, we introduce ChemCoTBench, a reasoning framework that bridges molecular structure understanding with arithmetic-inspired operations, including addition, deletion, and substitution, to formalize chemical problem-solving into transparent, step-by-step workflows. By treating molecular transformations as modular "chemical operations", the framework enables slow-thinking reasoning, mirroring the logic of mathematical proofs while grounding solutions in real-world chemical constraints. We evaluate models on two high-impact tasks: Molecular Property Optimization and Chemical Reaction Prediction. These tasks mirror real-world challenges while providing structured evaluability. By providing annotated datasets, a reasoning taxonomy, and baseline evaluations, ChemCoTBench bridges the gap between abstract reasoning methods and practical chemical discovery, establishing a foundation for advancing LLMs as tools for AI-driven scientific innovation. 

---
# Complex System Diagnostics Using a Knowledge Graph-Informed and Large Language Model-Enhanced Framework 

**Authors**: Saman Marandi, Yu-Shu Hu, Mohammad Modarres  

**Link**: [PDF](https://arxiv.org/pdf/2505.21291)  

**Abstract**: In this paper, we present a novel diagnostic framework that integrates Knowledge Graphs (KGs) and Large Language Models (LLMs) to support system diagnostics in high-reliability systems such as nuclear power plants. Traditional diagnostic modeling struggles when systems become too complex, making functional modeling a more attractive approach. Our approach introduces a diagnostic framework grounded in the functional modeling principles of the Dynamic Master Logic (DML) model. It incorporates two coordinated LLM components, including an LLM-based workflow for automated construction of DML logic from system documentation and an LLM agent that facilitates interactive diagnostics. The generated logic is encoded into a structured KG, referred to as KG-DML, which supports hierarchical fault reasoning. Expert knowledge or operational data can also be incorporated to refine the model's precision and diagnostic depth. In the interaction phase, users submit natural language queries, which are interpreted by the LLM agent. The agent selects appropriate tools for structured reasoning, including upward and downward propagation across the KG-DML. Rather than embedding KG content into every prompt, the LLM agent distinguishes between diagnostic and interpretive tasks. For diagnostics, the agent selects and executes external tools that perform structured KG reasoning. For general queries, a Graph-based Retrieval-Augmented Generation (Graph-RAG) approach is used, retrieving relevant KG segments and embedding them into the prompt to generate natural explanations. A case study on an auxiliary feedwater system demonstrated the framework's effectiveness, with over 90% accuracy in key elements and consistent tool and argument extraction, supporting its use in safety-critical diagnostics. 

---
# Interpreting Social Bias in LVLMs via Information Flow Analysis and Multi-Round Dialogue Evaluation 

**Authors**: Zhengyang Ji, Yifan Jia, Shang Gao, Yutao Yue  

**Link**: [PDF](https://arxiv.org/pdf/2505.21106)  

**Abstract**: Large Vision Language Models (LVLMs) have achieved remarkable progress in multimodal tasks, yet they also exhibit notable social biases. These biases often manifest as unintended associations between neutral concepts and sensitive human attributes, leading to disparate model behaviors across demographic groups. While existing studies primarily focus on detecting and quantifying such biases, they offer limited insight into the underlying mechanisms within the models. To address this gap, we propose an explanatory framework that combines information flow analysis with multi-round dialogue evaluation, aiming to understand the origin of social bias from the perspective of imbalanced internal information utilization. Specifically, we first identify high-contribution image tokens involved in the model's reasoning process for neutral questions via information flow analysis. Then, we design a multi-turn dialogue mechanism to evaluate the extent to which these key tokens encode sensitive information. Extensive experiments reveal that LVLMs exhibit systematic disparities in information usage when processing images of different demographic groups, suggesting that social bias is deeply rooted in the model's internal reasoning dynamics. Furthermore, we complement our findings from a textual modality perspective, showing that the model's semantic representations already display biased proximity patterns, thereby offering a cross-modal explanation of bias formation. 

---
# Large Language Model-enhanced Reinforcement Learning for Low-Altitude Economy Networking 

**Authors**: Lingyi Cai, Ruichen Zhang, Changyuan Zhao, Yu Zhang, Jiawen Kang, Dusit Niyato, Tao Jiang, Xuemin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21045)  

**Abstract**: Low-Altitude Economic Networking (LAENet) aims to support diverse flying applications below 1,000 meters by deploying various aerial vehicles for flexible and cost-effective aerial networking. However, complex decision-making, resource constraints, and environmental uncertainty pose significant challenges to the development of the LAENet. Reinforcement learning (RL) offers a potential solution in response to these challenges but has limitations in generalization, reward design, and model stability. The emergence of large language models (LLMs) offers new opportunities for RL to mitigate these limitations. In this paper, we first present a tutorial about integrating LLMs into RL by using the capacities of generation, contextual understanding, and structured reasoning of LLMs. We then propose an LLM-enhanced RL framework for the LAENet in terms of serving the LLM as information processor, reward designer, decision-maker, and generator. Moreover, we conduct a case study by using LLMs to design a reward function to improve the learning performance of RL in the LAENet. Finally, we provide a conclusion and discuss future work. 

---
# Step-Wise Formal Verification for LLM-Based Mathematical Problem Solving 

**Authors**: Kuo Zhou, Lu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20869)  

**Abstract**: Large Language Models (LLMs) have demonstrated formidable capabilities in solving mathematical problems, yet they may still commit logical reasoning and computational errors during the problem-solving process. Thus, this paper proposes a framework, MATH-VF, which includes a Formalizer and a Critic, for formally verifying the correctness of the solutions generated by large language models. Our framework first utilizes a Formalizer which employs an LLM to translate a natural language solution into a formal context. Afterward, our Critic (which integrates various external tools such as a Computer Algebra System and an SMT solver) evaluates the correctness of each statement within the formal context, and when a statement is incorrect, our Critic provides corrective feedback. We empirically investigate the effectiveness of MATH-VF in two scenarios: 1) Verification: MATH-VF is utilized to determine the correctness of a solution to a given problem. 2) Refinement: When MATH-VF identifies errors in the solution generated by an LLM-based solution generator for a given problem, it submits the corrective suggestions proposed by the Critic to the solution generator to regenerate the solution. We evaluate our framework on widely used mathematical benchmarks: MATH500 and ProcessBench, demonstrating the superiority of our approach over existing approaches. 

---
# MT-Mol:Multi Agent System with Tool-based Reasoning for Molecular Optimization 

**Authors**: Hyomin Kim, Yunhui Jang, Sungsoo Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.20820)  

**Abstract**: Large language models (LLMs) have large potential for molecular optimization, as they can gather external chemistry tools and enable collaborative interactions to iteratively refine molecular candidates. However, this potential remains underexplored, particularly in the context of structured reasoning, interpretability, and comprehensive tool-grounded molecular optimization. To address this gap, we introduce MT-Mol, a multi-agent framework for molecular optimization that leverages tool-guided reasoning and role-specialized LLM agents. Our system incorporates comprehensive RDKit tools, categorized into five distinct domains: structural descriptors, electronic and topological features, fragment-based functional groups, molecular representations, and miscellaneous chemical properties. Each category is managed by an expert analyst agent, responsible for extracting task-relevant tools and enabling interpretable, chemically grounded feedback. MT-Mol produces molecules with tool-aligned and stepwise reasoning through the interaction between the analyst agents, a molecule-generating scientist, a reasoning-output verifier, and a reviewer agent. As a result, we show that our framework shows the state-of-the-art performance of the PMO-1K benchmark on 17 out of 23 tasks. 

---
# Can Agents Fix Agent Issues? 

**Authors**: Alfin Wijaya Rahardja, Junwei Liu, Weitong Chen, Zhenpeng Chen, Yiling Lou  

**Link**: [PDF](https://arxiv.org/pdf/2505.20749)  

**Abstract**: LLM-based agent systems are emerging as a new software paradigm and have been widely adopted across diverse domains such as medicine, robotics, and programming. However, maintaining these systems requires substantial effort, as they are inevitably prone to bugs and continually evolve to meet changing external requirements. Therefore, automatically resolving agent issues (i.e., bug reports or feature requests) is a crucial and challenging task. While recent software engineering (SE) agents (e.g., SWE-agent) have shown promise in addressing issues in traditional software systems, it remains unclear how effectively they can resolve real-world issues in agent systems, which differ significantly from traditional software. To fill this gap, we first manually analyze 201 real-world agent issues and identify common categories of agent issues. We then spend 500 person-hours constructing AGENTISSUE-BENCH, a reproducible benchmark comprising 50 agent issue resolution tasks (each with an executable environment and failure-triggering tests). We further evaluate state-of-the-art SE agents on AGENTISSUE-BENCH and reveal their limited effectiveness (i.e., with only 3.33% - 12.67% resolution rates). These results underscore the unique challenges of maintaining agent systems compared to traditional software, highlighting the need for further research to develop advanced SE agents for resolving agent issues. Data and code are available at this https URL . 

---
# MSEarth: A Benchmark for Multimodal Scientific Comprehension of Earth Science 

**Authors**: Xiangyu Zhao, Wanghan Xu, Bo Liu, Yuhao Zhou, Fenghua Ling, Ben Fei, Xiaoyu Yue, Lei Bai, Wenlong Zhang, Xiao-Ming Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.20740)  

**Abstract**: The rapid advancement of multimodal large language models (MLLMs) has unlocked new opportunities to tackle complex scientific challenges. Despite this progress, their application in addressing earth science problems, especially at the graduate level, remains underexplored. A significant barrier is the absence of benchmarks that capture the depth and contextual complexity of geoscientific reasoning. Current benchmarks often rely on synthetic datasets or simplistic figure-caption pairs, which do not adequately reflect the intricate reasoning and domain-specific insights required for real-world scientific applications. To address these gaps, we introduce MSEarth, a multimodal scientific benchmark curated from high-quality, open-access scientific publications. MSEarth encompasses the five major spheres of Earth science: atmosphere, cryosphere, hydrosphere, lithosphere, and biosphere, featuring over 7K figures with refined captions. These captions are crafted from the original figure captions and enriched with discussions and reasoning from the papers, ensuring the benchmark captures the nuanced reasoning and knowledge-intensive content essential for advanced scientific tasks. MSEarth supports a variety of tasks, including scientific figure captioning, multiple choice questions, and open-ended reasoning challenges. By bridging the gap in graduate-level benchmarks, MSEarth provides a scalable and high-fidelity resource to enhance the development and evaluation of MLLMs in scientific reasoning. The benchmark is publicly available to foster further research and innovation in this field. Resources related to this benchmark can be found at this https URL and this https URL. 

---
# RRO: LLM Agent Optimization Through Rising Reward Trajectories 

**Authors**: Zilong Wang, Jingfeng Yang, Sreyashi Nag, Samarth Varshney, Xianfeng Tang, Haoming Jiang, Jingbo Shang, Sheikh Muhammad Sarwar  

**Link**: [PDF](https://arxiv.org/pdf/2505.20737)  

**Abstract**: Large language models (LLMs) have exhibited extraordinary performance in a variety of tasks while it remains challenging for them to solve complex multi-step tasks as agents. In practice, agents sensitive to the outcome of certain key steps which makes them likely to fail the task because of a subtle mistake in the planning trajectory. Recent approaches resort to calibrating the reasoning process through reinforcement learning. They reward or penalize every reasoning step with process supervision, as known as Process Reward Models (PRMs). However, PRMs are difficult and costly to scale up with a large number of next action candidates since they require extensive computations to acquire the training data through the per-step trajectory exploration. To mitigate this issue, we focus on the relative reward trend across successive reasoning steps and propose maintaining an increasing reward in the collected trajectories for process supervision, which we term Reward Rising Optimization (RRO). Specifically, we incrementally augment the process supervision until identifying a step exhibiting positive reward differentials, i.e. rising rewards, relative to its preceding iteration. This method dynamically expands the search space for the next action candidates, efficiently capturing high-quality data. We provide mathematical groundings and empirical results on the WebShop and InterCode-SQL benchmarks, showing that our proposed RRO achieves superior performance while requiring much less exploration cost. 

---
# LLM-Guided Reinforcement Learning: Addressing Training Bottlenecks through Policy Modulation 

**Authors**: Heng Tan, Hua Yan, Yu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20671)  

**Abstract**: While reinforcement learning (RL) has achieved notable success in various domains, training effective policies for complex tasks remains challenging. Agents often converge to local optima and fail to maximize long-term rewards. Existing approaches to mitigate training bottlenecks typically fall into two categories: (i) Automated policy refinement, which identifies critical states from past trajectories to guide policy updates, but suffers from costly and uncertain model training; and (ii) Human-in-the-loop refinement, where human feedback is used to correct agent behavior, but this does not scale well to environments with large or continuous action spaces. In this work, we design a large language model-guided policy modulation framework that leverages LLMs to improve RL training without additional model training or human intervention. We first prompt an LLM to identify critical states from a sub-optimal agent's trajectories. Based on these states, the LLM then provides action suggestions and assigns implicit rewards to guide policy refinement. Experiments across standard RL benchmarks demonstrate that our method outperforms state-of-the-art baselines, highlighting the effectiveness of LLM-based explanations in addressing RL training bottlenecks. 

---
# MIRROR: Multi-agent Intra- and Inter-Reflection for Optimized Reasoning in Tool Learning 

**Authors**: Zikang Guo, Benfeng Xu, Xiaorui Wang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2505.20670)  

**Abstract**: Complex tasks involving tool integration pose significant challenges for Large Language Models (LLMs), leading to the emergence of multi-agent workflows as a promising solution. Reflection has emerged as an effective strategy for correcting erroneous trajectories in agentic workflows. However, existing approaches only exploit such capability in the post-action stage, where the agent observes the execution outcomes. We argue that, like humans, LLMs can also engage in reflection before action execution: the agent can anticipate undesirable outcomes from its own decisions, which not only provides a necessarily complementary perspective to evaluate the decision but also prevents the propagation of errors throughout the trajectory. In this paper, we propose MIRROR, a framework that consists of both intra-reflection, which critically assesses intended actions before execution, and inter-reflection, which further adjusts the trajectory based on observations. This design systematically leverages LLM reflection capabilities to eliminate and rectify erroneous actions on a more comprehensive scope. Evaluations on both the StableToolBench and TravelPlanner benchmarks demonstrate MIRROR's superior performance, achieving state-of-the-art results compared to existing approaches. 

---
# Agent-Environment Alignment via Automated Interface Generation 

**Authors**: Kaiming Liu, Xuanyu Lei, Ziyue Wang, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21055)  

**Abstract**: Large language model (LLM) agents have shown impressive reasoning capabilities in interactive decision-making tasks. These agents interact with environment through intermediate interfaces, such as predefined action spaces and interaction rules, which mediate the perception and action. However, mismatches often happen between the internal expectations of the agent regarding the influence of its issued actions and the actual state transitions in the environment, a phenomenon referred to as \textbf{agent-environment misalignment}. While prior work has invested substantially in improving agent strategies and environment design, the critical role of the interface still remains underexplored. In this work, we empirically demonstrate that agent-environment misalignment poses a significant bottleneck to agent performance. To mitigate this issue, we propose \textbf{ALIGN}, an \underline{A}uto-A\underline{l}igned \underline{I}nterface \underline{G}e\underline{n}eration framework that alleviates the misalignment by enriching the interface. Specifically, the ALIGN-generated interface enhances both the static information of the environment and the step-wise observations returned to the agent. Implemented as a lightweight wrapper, this interface achieves the alignment without modifying either the agent logic or the environment code. Experiments across multiple domains including embodied tasks, web navigation and tool-use, show consistent performance improvements, with up to a 45.67\% success rate improvement observed in ALFWorld. Meanwhile, ALIGN-generated interface can generalize across different agent architectures and LLM backbones without interface regeneration. Code and experimental results are available at this https URL. 

---
# CoderAgent: Simulating Student Behavior for Personalized Programming Learning with Large Language Models 

**Authors**: Yi Zhan, Qi Liu, Weibo Gao, Zheng Zhang, Tianfu Wang, Shuanghong Shen, Junyu Lu, Zhenya Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20642)  

**Abstract**: Personalized programming tutoring, such as exercise recommendation, can enhance learners' efficiency, motivation, and outcomes, which is increasingly important in modern digital education. However, the lack of sufficient and high-quality programming data, combined with the mismatch between offline evaluation and real-world learning, hinders the practical deployment of such systems. To address this challenge, many approaches attempt to simulate learner practice data, yet they often overlook the fine-grained, iterative nature of programming learning, resulting in a lack of interpretability and granularity. To fill this gap, we propose a LLM-based agent, CoderAgent, to simulate students' programming processes in a fine-grained manner without relying on real data. Specifically, we equip each human learner with an intelligent agent, the core of which lies in capturing the cognitive states of the human programming practice process. Inspired by ACT-R, a cognitive architecture framework, we design the structure of CoderAgent to align with human cognitive architecture by focusing on the mastery of programming knowledge and the application of coding ability. Recognizing the inherent patterns in multi-layered cognitive reasoning, we introduce the Programming Tree of Thought (PTOT), which breaks down the process into four steps: why, how, where, and what. This approach enables a detailed analysis of iterative problem-solving strategies. Finally, experimental evaluations on real-world datasets demonstrate that CoderAgent provides interpretable insights into learning trajectories and achieves accurate simulations, paving the way for personalized programming education. 

---
# Why Distillation can Outperform Zero-RL: The Role of Flexible Reasoning 

**Authors**: Xiao Hu, Xingyu Lu, Liyuan Mao, YiFan Zhang, Tianke Zhang, Bin Wen, Fan Yang, Tingting Gao, Guorui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.21067)  

**Abstract**: Reinforcement learning (RL) has played an important role in improving the reasoning ability of large language models (LLMs). Some studies apply RL directly to \textit{smaller} base models (known as zero-RL) and also achieve notable progress. However, in this paper, we show that using only 920 examples, a simple distillation method based on the base model can clearly outperform zero-RL, which typically requires much more data and computational cost. By analyzing the token frequency in model outputs, we find that the distilled model shows more flexible reasoning. It uses anthropomorphic tokens and logical connectors much more often than the zero-RL model. Further analysis reveals that distillation enhances the presence of two advanced cognitive behaviors: Multi-Perspective Thinking or Attempting and Metacognitive Awareness. Frequent occurrences of these two advanced cognitive behaviors give rise to flexible reasoning, which is essential for solving complex reasoning problems, while zero-RL fails to significantly boost the frequency of these behaviors. 

---
# Scaling over Scaling: Exploring Test-Time Scaling Pareto in Large Reasoning Models 

**Authors**: Jian Wang, Boyan Zhu, Chak Tou Leong, Yongqi Li, Wenjie Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.20522)  

**Abstract**: Large reasoning models (LRMs) have exhibited the capacity of enhancing reasoning performance via internal test-time scaling. Building upon this, a promising direction is to further scale test-time compute to unlock even greater reasoning capabilities. However, as we push these scaling boundaries, systematically understanding the practical limits and achieving optimal resource allocation becomes a critical challenge. In this paper, we investigate the scaling Pareto of test-time scaling and introduce the Test-Time Scaling Performance Model (TTSPM). We theoretically analyze two fundamental paradigms for such extended scaling, parallel scaling and sequential scaling, from a probabilistic modeling perspective. Our primary contribution is the derivation of the saturation point on the scaling budget for both strategies, identifying thresholds beyond which additional computation yields diminishing returns. Remarkably, despite their distinct mechanisms, both paradigms converge to a unified mathematical structure in their upper bounds. We empirically validate our theoretical findings on challenging reasoning benchmarks, including AIME, MATH-500, and GPQA, demonstrating the practical utility of these bounds for test-time resource allocation. We hope that this work provides insights into the cost-benefit trade-offs of test-time scaling, guiding the development of more resource-efficient inference strategies for large reasoning models. 

---
# SCAR: Shapley Credit Assignment for More Efficient RLHF 

**Authors**: Meng Cao, Shuyuan Zhang, Xiao-Wen Chang, Doina Precup  

**Link**: [PDF](https://arxiv.org/pdf/2505.20417)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) is a widely used technique for aligning Large Language Models (LLMs) with human preferences, yet it often suffers from sparse reward signals, making effective credit assignment challenging. In typical setups, the reward model provides a single scalar score for an entire generated sequence, offering little insight into which token or span-level decisions were responsible for the outcome. To address this, we propose Shapley Credit Assignment Rewards (SCAR), a novel method that leverages Shapley values in cooperative game theory. SCAR distributes the total sequence-level reward among constituent tokens or text spans based on their principled marginal contributions. This creates dense reward signals, crucially, without necessitating the training of auxiliary critique models or recourse to fine-grained human annotations at intermediate generation stages. Unlike prior dense reward methods, SCAR offers a game-theoretic foundation for fair credit attribution. Theoretically, we demonstrate that SCAR preserves the original optimal policy, and empirically, across diverse tasks including sentiment control, text summarization, and instruction tuning, we show that SCAR converges significantly faster and achieves higher final reward scores compared to standard RLHF and attention-based dense reward baselines. Our findings suggest that SCAR provides a more effective and theoretically sound method for credit assignment in RLHF, leading to more efficient alignment of LLMs. 

---
# Reinforcement Speculative Decoding for Fast Ranking 

**Authors**: Yingpeng Du, Tianjun Wei, Zhu Sun, Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20316)  

**Abstract**: Large Language Models (LLMs) have been widely adopted in ranking systems such as information retrieval (IR) systems and recommender systems (RSs). To alleviate the latency of auto-regressive decoding, some studies explore the single (first) token decoding for ranking approximation, but they suffer from severe degradation in tail positions. Although speculative decoding (SD) methods can be a remedy with verification at different positions, they face challenges in ranking systems due to their left-to-right decoding paradigm. Firstly, ranking systems require strict latency constraints, but verification rounds in SD methods remain agnostic; Secondly, SD methods usually discard listwise ranking knowledge about unaccepted items in previous rounds, hindering future multi-token prediction, especially when candidate tokens are the unaccepted items. In this paper, we propose a Reinforcement Speculative Decoding method for fast ranking inference of LLMs. To meet the ranking systems' latency requirement, we propose an up-to-down decoding paradigm that employs an agent to iteratively modify the ranking sequence under a constrained budget. Specifically, we design a ranking-tailored policy optimization, actively exploring optimal multi-round ranking modification policy verified by LLMs via reinforcement learning (RL). To better approximate the target LLM under the constrained budget, we trigger the agent fully utilizing the listwise ranking knowledge about all items verified by LLMs across different rounds in RL, enhancing the modification policy of the agent. More importantly, we demonstrate the theoretical robustness and advantages of our paradigm and implementation. Experiments on both IR and RS tasks show the effectiveness of our proposed method. 

---
# RLJP: Legal Judgment Prediction via First-Order Logic Rule-enhanced with Large Language Models 

**Authors**: Yue Zhang, Zhiliang Tian, Shicheng Zhou, Haiyang Wang, Wenqing Hou, Yuying Liu, Xuechen Zhao, Minlie Huang, Ye Wang, Bin Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.21281)  

**Abstract**: Legal Judgment Prediction (LJP) is a pivotal task in legal AI. Existing semantic-enhanced LJP models integrate judicial precedents and legal knowledge for high performance. But they neglect legal reasoning logic, a critical component of legal judgments requiring rigorous logical analysis. Although some approaches utilize legal reasoning logic for high-quality predictions, their logic rigidity hinders adaptation to case-specific logical frameworks, particularly in complex cases that are lengthy and detailed. This paper proposes a rule-enhanced legal judgment prediction framework based on first-order logic (FOL) formalism and comparative learning (CL) to develop an adaptive adjustment mechanism for legal judgment logic and further enhance performance in LJP. Inspired by the process of human exam preparation, our method follows a three-stage approach: first, we initialize judgment rules using the FOL formalism to capture complex reasoning logic accurately; next, we propose a Confusion-aware Contrastive Learning (CACL) to dynamically optimize the judgment rules through a quiz consisting of confusable cases; finally, we utilize the optimized judgment rules to predict legal judgments. Experimental results on two public datasets show superior performance across all metrics. The code is publicly available{this https URL}. 

---
# Reasoning in Neurosymbolic AI 

**Authors**: Son Tran, Edjard Mota, Artur d'Avila Garcez  

**Link**: [PDF](https://arxiv.org/pdf/2505.20313)  

**Abstract**: Knowledge representation and reasoning in neural networks have been a long-standing endeavor which has attracted much attention recently. The principled integration of reasoning and learning in neural networks is a main objective of the area of neurosymbolic Artificial Intelligence (AI). In this chapter, a simple energy-based neurosymbolic AI system is described that can represent and reason formally about any propositional logic formula. This creates a powerful combination of learning from data and knowledge and logical reasoning. We start by positioning neurosymbolic AI in the context of the current AI landscape that is unsurprisingly dominated by Large Language Models (LLMs). We identify important challenges of data efficiency, fairness and safety of LLMs that might be addressed by neurosymbolic reasoning systems with formal reasoning capabilities. We then discuss the representation of logic by the specific energy-based system, including illustrative examples and empirical evaluation of the correspondence between logical reasoning and energy minimization using Restricted Boltzmann Machines (RBM). Learning from data and knowledge is also evaluated empirically and compared with a symbolic, neural and a neurosymbolic system. Results reported in this chapter in an accessible way are expected to reignite the research on the use of neural networks as massively-parallel models for logical reasoning and promote the principled integration of reasoning and learning in deep networks. We conclude the chapter with a discussion of the importance of positioning neurosymbolic AI within a broader framework of formal reasoning and accountability in AI, discussing the challenges for neurosynbolic AI to tackle the various known problems of reliability of deep learning. 

---
# GIFARC: Synthetic Dataset for Leveraging Human-Intuitive Analogies to Elevate AI Reasoning 

**Authors**: Woochang Sim, Hyunseok Ryu, Kyungmin Choi, Sungwon Han, Sundong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.20672)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC) poses a stringent test of general AI capabilities, requiring solvers to infer abstract patterns from only a handful of examples. Despite substantial progress in deep learning, state-of-the-art models still achieve accuracy rates of merely 40-55% on 2024 ARC Competition, indicative of a significant gap between their performance and human-level reasoning. In this work, we seek to bridge that gap by introducing an analogy-inspired ARC dataset, GIFARC. Leveraging large language models (LLMs) and vision-language models (VLMs), we synthesize new ARC-style tasks from a variety of GIF images that include analogies. Each new task is paired with ground-truth analogy, providing an explicit mapping between visual transformations and everyday concepts. By embedding robust human-intuitive analogies into ARC-style tasks, GIFARC guides AI agents to evaluate the task analogically before engaging in brute-force pattern search, thus efficiently reducing problem complexity and build a more concise and human-understandable solution. We empirically validate that guiding LLM with analogic approach with GIFARC affects task-solving approaches of LLMs to align with analogic approach of human. 

---
# Active-O3: Empowering Multimodal Large Language Models with Active Perception via GRPO 

**Authors**: Muzhi Zhu, Hao Zhong, Canyu Zhao, Zongze Du, Zheng Huang, Mingyu Liu, Hao Chen, Cheng Zou, Jingdong Chen, Ming Yang, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21457)  

**Abstract**: Active vision, also known as active perception, refers to the process of actively selecting where and how to look in order to gather task-relevant information. It is a critical component of efficient perception and decision-making in humans and advanced embodied agents. Recently, the use of Multimodal Large Language Models (MLLMs) as central planning and decision-making modules in robotic systems has gained extensive attention. However, despite the importance of active perception in embodied intelligence, there is little to no exploration of how MLLMs can be equipped with or learn active perception capabilities. In this paper, we first provide a systematic definition of MLLM-based active perception tasks. We point out that the recently proposed GPT-o3 model's zoom-in search strategy can be regarded as a special case of active perception; however, it still suffers from low search efficiency and inaccurate region selection. To address these issues, we propose ACTIVE-O3, a purely reinforcement learning based training framework built on top of GRPO, designed to equip MLLMs with active perception capabilities. We further establish a comprehensive benchmark suite to evaluate ACTIVE-O3 across both general open-world tasks, such as small-object and dense object grounding, and domain-specific scenarios, including small object detection in remote sensing and autonomous driving, as well as fine-grained interactive segmentation. In addition, ACTIVE-O3 also demonstrates strong zero-shot reasoning abilities on the V* Benchmark, without relying on any explicit reasoning data. We hope that our work can provide a simple codebase and evaluation protocol to facilitate future research on active perception in MLLMs. 

---
# Hume: Introducing System-2 Thinking in Visual-Language-Action Model 

**Authors**: Haoming Song, Delin Qu, Yuanqi Yao, Qizhi Chen, Qi Lv, Yiwen Tang, Modi Shi, Guanghui Ren, Maoqing Yao, Bin Zhao, Dong Wang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.21432)  

**Abstract**: Humans practice slow thinking before performing actual actions when handling complex tasks in the physical world. This thinking paradigm, recently, has achieved remarkable advancement in boosting Large Language Models (LLMs) to solve complex tasks in digital domains. However, the potential of slow thinking remains largely unexplored for robotic foundation models interacting with the physical world. In this work, we propose Hume: a dual-system Vision-Language-Action (VLA) model with value-guided System-2 thinking and cascaded action denoising, exploring human-like thinking capabilities of Vision-Language-Action models for dexterous robot control. System 2 of Hume implements value-Guided thinking by extending a Vision-Language-Action Model backbone with a novel value-query head to estimate the state-action value of predicted actions. The value-guided thinking is conducted by repeat sampling multiple action candidates and selecting one according to state-action value. System 1 of Hume is a lightweight reactive visuomotor policy that takes System 2 selected action and performs cascaded action denoising for dexterous robot control. At deployment time, System 2 performs value-guided thinking at a low frequency while System 1 asynchronously receives the System 2 selected action candidate and predicts fluid actions in real time. We show that Hume outperforms the existing state-of-the-art Vision-Language-Action models across multiple simulation benchmark and real-robot deployments. 

---
# Improving LLM-based Global Optimization with Search Space Partitioning 

**Authors**: Andrej Schwanke, Lyubomir Ivanov, David Salinas, Fabio Ferreira, Aaron Klein, Frank Hutter, Arber Zela  

**Link**: [PDF](https://arxiv.org/pdf/2505.21372)  

**Abstract**: Large Language Models (LLMs) have recently emerged as effective surrogate models and candidate generators within global optimization frameworks for expensive blackbox functions. Despite promising results, LLM-based methods often struggle in high-dimensional search spaces or when lacking domain-specific priors, leading to sparse or uninformative suggestions. To overcome these limitations, we propose HOLLM, a novel global optimization algorithm that enhances LLM-driven sampling by partitioning the search space into promising subregions. Each subregion acts as a ``meta-arm'' selected via a bandit-inspired scoring mechanism that effectively balances exploration and exploitation. Within each selected subregion, an LLM then proposes high-quality candidate points, without any explicit domain knowledge. Empirical evaluation on standard optimization benchmarks shows that HOLLM consistently matches or surpasses leading Bayesian optimization and trust-region methods, while substantially outperforming global LLM-based sampling strategies. 

---
# Efficient Large Language Model Inference with Neural Block Linearization 

**Authors**: Mete Erdogan, Francesco Tonin, Volkan Cevher  

**Link**: [PDF](https://arxiv.org/pdf/2505.21077)  

**Abstract**: The high inference demands of transformer-based Large Language Models (LLMs) pose substantial challenges in their deployment. To this end, we introduce Neural Block Linearization (NBL), a novel framework for accelerating transformer model inference by replacing self-attention layers with linear approximations derived from Linear Minimum Mean Squared Error estimators. NBL leverages Canonical Correlation Analysis to compute a theoretical upper bound on the approximation error. Then, we use this bound as a criterion for substitution, selecting the LLM layers with the lowest linearization error. NBL can be efficiently applied to pre-trained LLMs without the need for fine-tuning. In experiments, NBL achieves notable computational speed-ups while preserving competitive accuracy on multiple reasoning benchmarks. For instance, applying NBL to 12 self-attention layers in DeepSeek-R1-Distill-Llama-8B increases the inference speed by 32% with less than 1% accuracy trade-off, making it a flexible and promising solution to improve the inference efficiency of LLMs. 

---
# Red-Teaming Text-to-Image Systems by Rule-based Preference Modeling 

**Authors**: Yichuan Cao, Yibo Miao, Xiao-Shan Gao, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.21074)  

**Abstract**: Text-to-image (T2I) models raise ethical and safety concerns due to their potential to generate inappropriate or harmful images. Evaluating these models' security through red-teaming is vital, yet white-box approaches are limited by their need for internal access, complicating their use with closed-source models. Moreover, existing black-box methods often assume knowledge about the model's specific defense mechanisms, limiting their utility in real-world commercial API scenarios. A significant challenge is how to evade unknown and diverse defense mechanisms. To overcome this difficulty, we propose a novel Rule-based Preference modeling Guided Red-Teaming (RPG-RT), which iteratively employs LLM to modify prompts to query and leverages feedback from T2I systems for fine-tuning the LLM. RPG-RT treats the feedback from each iteration as a prior, enabling the LLM to dynamically adapt to unknown defense mechanisms. Given that the feedback is often labeled and coarse-grained, making it difficult to utilize directly, we further propose rule-based preference modeling, which employs a set of rules to evaluate desired or undesired feedback, facilitating finer-grained control over the LLM's dynamic adaptation process. Extensive experiments on nineteen T2I systems with varied safety mechanisms, three online commercial API services, and T2V models verify the superiority and practicality of our approach. 

---
# Generalizable Heuristic Generation Through Large Language Models with Meta-Optimization 

**Authors**: Yiding Shi, Jianan Zhou, Wen Song, Jieyi Bi, Yaoxin Wu, Jie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20881)  

**Abstract**: Heuristic design with large language models (LLMs) has emerged as a promising approach for tackling combinatorial optimization problems (COPs). However, existing approaches often rely on manually predefined evolutionary computation (EC) optimizers and single-task training schemes, which may constrain the exploration of diverse heuristic algorithms and hinder the generalization of the resulting heuristics. To address these issues, we propose Meta-Optimization of Heuristics (MoH), a novel framework that operates at the optimizer level, discovering effective optimizers through the principle of meta-learning. Specifically, MoH leverages LLMs to iteratively refine a meta-optimizer that autonomously constructs diverse optimizers through (self-)invocation, thereby eliminating the reliance on a predefined EC optimizer. These constructed optimizers subsequently evolve heuristics for downstream tasks, enabling broader heuristic exploration. Moreover, MoH employs a multi-task training scheme to promote its generalization capability. Experiments on classic COPs demonstrate that MoH constructs an effective and interpretable meta-optimizer, achieving state-of-the-art performance across various downstream tasks, particularly in cross-size settings. 

---
# FM-Planner: Foundation Model Guided Path Planning for Autonomous Drone Navigation 

**Authors**: Jiaping Xiao, Cheng Wen Tsao, Yuhang Zhang, Mir Feroskhan  

**Link**: [PDF](https://arxiv.org/pdf/2505.20783)  

**Abstract**: Path planning is a critical component in autonomous drone operations, enabling safe and efficient navigation through complex environments. Recent advances in foundation models, particularly large language models (LLMs) and vision-language models (VLMs), have opened new opportunities for enhanced perception and intelligent decision-making in robotics. However, their practical applicability and effectiveness in global path planning remain relatively unexplored. This paper proposes foundation model-guided path planners (FM-Planner) and presents a comprehensive benchmarking study and practical validation for drone path planning. Specifically, we first systematically evaluate eight representative LLM and VLM approaches using standardized simulation scenarios. To enable effective real-time navigation, we then design an integrated LLM-Vision planner that combines semantic reasoning with visual perception. Furthermore, we deploy and validate the proposed path planner through real-world experiments under multiple configurations. Our findings provide valuable insights into the strengths, limitations, and feasibility of deploying foundation models in real-world drone applications and providing practical implementations in autonomous flight. Project site: this https URL. 

---
# Respond to Change with Constancy: Instruction-tuning with LLM for Non-I.I.D. Network Traffic Classification 

**Authors**: Xinjie Lin, Gang Xiong, Gaopeng Gou, Wenqi Dong, Jing Yu, Zhen Li, Wei Xia  

**Link**: [PDF](https://arxiv.org/pdf/2505.20866)  

**Abstract**: Encrypted traffic classification is highly challenging in network security due to the need for extracting robust features from content-agnostic traffic data. Existing approaches face critical issues: (i) Distribution drift, caused by reliance on the closedworld assumption, limits adaptability to realworld, shifting patterns; (ii) Dependence on labeled data restricts applicability where such data is scarce or unavailable. Large language models (LLMs) have demonstrated remarkable potential in offering generalizable solutions across a wide range of tasks, achieving notable success in various specialized fields. However, their effectiveness in traffic analysis remains constrained by challenges in adapting to the unique requirements of the traffic domain. In this paper, we introduce a novel traffic representation model named Encrypted Traffic Out-of-Distribution Instruction Tuning with LLM (ETooL), which integrates LLMs with knowledge of traffic structures through a self-supervised instruction tuning paradigm. This framework establishes connections between textual information and traffic interactions. ETooL demonstrates more robust classification performance and superior generalization in both supervised and zero-shot traffic classification tasks. Notably, it achieves significant improvements in F1 scores: APP53 (I.I.D.) to 93.19%(6.62%) and 92.11%(4.19%), APP53 (O.O.D.) to 74.88%(18.17%) and 72.13%(15.15%), and ISCX-Botnet (O.O.D.) to 95.03%(9.16%) and 81.95%(12.08%). Additionally, we construct NETD, a traffic dataset designed to support dynamic distributional shifts, and use it to validate ETooL's effectiveness under varying distributional conditions. Furthermore, we evaluate the efficiency gains achieved through ETooL's instruction tuning approach. 

---
# MedSentry: Understanding and Mitigating Safety Risks in Medical LLM Multi-Agent Systems 

**Authors**: Kai Chen, Taihang Zhen, Hewei Wang, Kailai Liu, Xinfeng Li, Jing Huo, Tianpei Yang, Jinfeng Xu, Wei Dong, Yang Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.20824)  

**Abstract**: As large language models (LLMs) are increasingly deployed in healthcare, ensuring their safety, particularly within collaborative multi-agent configurations, is paramount. In this paper we introduce MedSentry, a benchmark comprising 5 000 adversarial medical prompts spanning 25 threat categories with 100 subthemes. Coupled with this dataset, we develop an end-to-end attack-defense evaluation pipeline to systematically analyze how four representative multi-agent topologies (Layers, SharedPool, Centralized, and Decentralized) withstand attacks from 'dark-personality' agents. Our findings reveal critical differences in how these architectures handle information contamination and maintain robust decision-making, exposing their underlying vulnerability mechanisms. For instance, SharedPool's open information sharing makes it highly susceptible, whereas Decentralized architectures exhibit greater resilience thanks to inherent redundancy and isolation. To mitigate these risks, we propose a personality-scale detection and correction mechanism that identifies and rehabilitates malicious agents, restoring system safety to near-baseline levels. MedSentry thus furnishes both a rigorous evaluation framework and practical defense strategies that guide the design of safer LLM-based multi-agent systems in medical domains. 

---
# Accelerating RL for LLM Reasoning with Optimal Advantage Regression 

**Authors**: Kianté Brantley, Mingyu Chen, Zhaolin Gao, Jason D. Lee, Wen Sun, Wenhao Zhan, Xuezhou Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20686)  

**Abstract**: Reinforcement learning (RL) has emerged as a powerful tool for fine-tuning large language models (LLMs) to improve complex reasoning abilities. However, state-of-the-art policy optimization methods often suffer from high computational overhead and memory consumption, primarily due to the need for multiple generations per prompt and the reliance on critic networks or advantage estimates of the current policy. In this paper, we propose $A$*-PO, a novel two-stage policy optimization framework that directly approximates the optimal advantage function and enables efficient training of LLMs for reasoning tasks. In the first stage, we leverage offline sampling from a reference policy to estimate the optimal value function $V$*, eliminating the need for costly online value estimation. In the second stage, we perform on-policy updates using a simple least-squares regression loss with only a single generation per prompt. Theoretically, we establish performance guarantees and prove that the KL-regularized RL objective can be optimized without requiring complex exploration strategies. Empirically, $A$*-PO achieves competitive performance across a wide range of mathematical reasoning benchmarks, while reducing training time by up to 2$\times$ and peak memory usage by over 30% compared to PPO, GRPO, and REBEL. Implementation of $A$*-PO can be found at this https URL. 

---
# Can Past Experience Accelerate LLM Reasoning? 

**Authors**: Bo Pan, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.20643)  

**Abstract**: Allocating more compute to large language models (LLMs) reasoning has generally been demonstrated to improve their effectiveness, but also results in increased inference time. In contrast, humans can perform tasks faster and better with increased experience and exposure. Hence, this paper aims to investigate the question: Can LLMs also become faster at reasoning through recurrent exposure on relevant tasks, and if so, how can it be achieved? To address these questions, we first formalize the problem setting of LLM reasoning speedup systematically in the dimensions of task relevancy and compute budget calculation. We then propose SpeedupLLM, a theoretically guaranteed framework to implement and benchmark such reasoning speedup behaviour based on adaptive compute allocation and memory mechanisms. We further conduct comprehensive experiments to benchmark such behaviour across different question similarity levels, memory methods, and reasoning methods. Results show that LLMs can generally reason faster with past experience, achieving up to a 56% reduction in compute cost when equipped with appropriate memory and reasoning methods. 

---
# Collision- and Reachability-Aware Multi-Robot Control with Grounded LLM Planners 

**Authors**: Jiabao Ji, Yongchao Chen, Yang Zhang, Ramana Rao Kompella, Chuchu Fan, Gaowen Liu, Shiyu Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20573)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance in various robot control tasks. However, their deployment in real-world applications remains constrained. Even state-ofthe-art LLMs, such as GPT-o4mini, frequently produce invalid action plans that violate physical constraints, such as directing a robot to an unreachable location or causing collisions between robots. This issue primarily arises from a lack of awareness of these physical constraints during the reasoning process. To address this issue, we propose a novel framework that integrates reinforcement learning with verifiable rewards (RLVR) to incentivize knowledge of physical constraints into LLMs to induce constraints-aware reasoning during plan generation. In this approach, only valid action plans that successfully complete a control task receive positive rewards. We applied our method to two small-scale LLMs: a non-reasoning Qwen2.5-3B-Instruct and a reasoning Qwen3-4B. The experiment results demonstrate that constraint-aware small LLMs largely outperform large-scale models without constraints, grounded on both the BoxNet task and a newly developed BoxNet3D environment built using MuJoCo. This work highlights the effectiveness of grounding even small LLMs with physical constraints to enable scalable and efficient multi-robot control in complex, physically constrained environments. 

---
# Risk-aware Direct Preference Optimization under Nested Risk Measure 

**Authors**: Lijun Zhang, Lin Li, Yajie Qi, Huizhong Song, Yaodong Yang, Jun Wang, Wei Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.20359)  

**Abstract**: When fine-tuning pre-trained Large Language Models (LLMs) to align with human values and intentions, maximizing the estimated reward can lead to superior performance, but it also introduces potential risks due to deviations from the reference model's intended behavior. Most existing methods typically introduce KL divergence to constrain deviations between the trained model and the reference model; however, this may not be sufficient in certain applications that require tight risk control. In this paper, we introduce Risk-aware Direct Preference Optimization (Ra-DPO), a novel approach that incorporates risk-awareness by employing a class of nested risk measures. This approach formulates a constrained risk-aware advantage function maximization problem and then converts the Bradley-Terry model into a token-level representation. The objective function maximizes the likelihood of the policy while suppressing the deviation between a trained model and the reference model using a sequential risk ratio, thereby enhancing the model's risk-awareness. Experimental results across three open-source datasets: IMDb Dataset, Anthropic HH Dataset, and AlpacaEval, demonstrate the proposed method's superior performance in balancing alignment performance and model drift. Our code is opensourced at this https URL. 

---
# LEGO-Compiler: Enhancing Neural Compilation Through Translation Composability 

**Authors**: Shuoming Zhang, Jiacheng Zhao, Chunwei Xia, Zheng Wang, Yunji Chen, Xiaobing Feng, Huimin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.20356)  

**Abstract**: Large language models (LLMs) have the potential to revolutionize how we design and implement compilers and code translation tools. However, existing LLMs struggle to handle long and complex programs. We introduce LEGO-Compiler, a novel neural compilation system that leverages LLMs to translate high-level languages into assembly code. Our approach centers on three key innovations: LEGO translation, which decomposes the input program into manageable blocks; breaking down the complex compilation process into smaller, simpler verifiable steps by organizing it as a verifiable LLM workflow by external tests; and a feedback mechanism for self-correction. Supported by formal proofs of translation composability, LEGO-Compiler demonstrates high accuracy on multiple datasets, including over 99% on ExeBench and 97.9% on industrial-grade AnsiBench. Additionally, LEGO-Compiler has also acheived near one order-of-magnitude improvement on compilable code size scalability. This work opens new avenues for applying LLMs to system-level tasks, complementing traditional compiler technologies. 

---
# Language Model Distillation: A Temporal Difference Imitation Learning Perspective 

**Authors**: Zishun Yu, Shangzhe Li, Xinhua Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.20335)  

**Abstract**: Large language models have led to significant progress across many NLP tasks, although their massive sizes often incur substantial computational costs. Distillation has become a common practice to compress these large and highly capable models into smaller, more efficient ones. Many existing language model distillation methods can be viewed as behavior cloning from the perspective of imitation learning or inverse reinforcement learning. This viewpoint has inspired subsequent studies that leverage (inverse) reinforcement learning techniques, including variations of behavior cloning and temporal difference learning methods. Rather than proposing yet another specific temporal difference method, we introduce a general framework for temporal difference-based distillation by exploiting the distributional sparsity of the teacher model. Specifically, it is often observed that language models assign most probability mass to a small subset of tokens. Motivated by this observation, we design a temporal difference learning framework that operates on a reduced action space (a subset of vocabulary), and demonstrate how practical algorithms can be derived and the resulting performance improvements. 

---
# Lookahead Q-Cache: Achieving More Consistent KV Cache Eviction via Pseudo Query 

**Authors**: Yixuan Wang, Shiyu Ji, Yijun Liu, Yuzhuang Xu, Yang Xu, Qingfu Zhu, Wanxiang Che  

**Link**: [PDF](https://arxiv.org/pdf/2505.20334)  

**Abstract**: Large language models (LLMs) rely on key-value cache (KV cache) to accelerate decoding by reducing redundant computations. However, the KV cache memory usage grows substantially with longer text sequences, posing challenges for efficient deployment. Existing KV cache eviction methods prune tokens using prefilling-stage attention scores, causing inconsistency with actual inference queries, especially under tight memory budgets. In this paper, we propose Lookahead Q-Cache (LAQ), a novel eviction framework that generates low-cost pseudo lookahead queries to better approximate the true decoding-stage queries. By using these lookahead queries as the observation window for importance estimation, LAQ achieves more consistent and accurate KV cache eviction aligned with real inference scenarios. Experimental results on LongBench and Needle-in-a-Haystack benchmarks show that LAQ outperforms existing methods across various budget levels, achieving a 1 $\sim$ 4 point improvement on LongBench under limited cache budget. Moreover, LAQ is complementary to existing approaches and can be flexibly combined to yield further improvements. 

---
# Let's Get You Hired: A Job Seeker's Perspective on Multi-Agent Recruitment Systems for Explaining Hiring Decisions 

**Authors**: Aditya Bhattacharya, Katrien Verbert  

**Link**: [PDF](https://arxiv.org/pdf/2505.20312)  

**Abstract**: During job recruitment, traditional applicant selection methods often lack transparency. Candidates are rarely given sufficient justifications for recruiting decisions, whether they are made manually by human recruiters or through the use of black-box Applicant Tracking Systems (ATS). To address this problem, our work introduces a multi-agent AI system that uses Large Language Models (LLMs) to guide job seekers during the recruitment process. Using an iterative user-centric design approach, we first conducted a two-phased exploratory study with four active job seekers to inform the design and development of the system. Subsequently, we conducted an in-depth, qualitative user study with 20 active job seekers through individual one-to-one interviews to evaluate the developed prototype. The results of our evaluation demonstrate that participants perceived our multi-agent recruitment system as significantly more actionable, trustworthy, and fair compared to traditional methods. Our study further helped us uncover in-depth insights into factors contributing to these perceived user experiences. Drawing from these insights, we offer broader design implications for building user-aligned, multi-agent explainable AI systems across diverse domains. 

---
# SpatialLLM: From Multi-modality Data to Urban Spatial Intelligence 

**Authors**: Jiabin Chen, Haiping Wang, Jinpeng Li, Yuan Liu, Zhen Dong, Bisheng Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.12703)  

**Abstract**: We propose SpatialLLM, a novel approach advancing spatial intelligence tasks in complex urban scenes. Unlike previous methods requiring geographic analysis tools or domain expertise, SpatialLLM is a unified language model directly addressing various spatial intelligence tasks without any training, fine-tuning, or expert intervention. The core of SpatialLLM lies in constructing detailed and structured scene descriptions from raw spatial data to prompt pre-trained LLMs for scene-based analysis. Extensive experiments show that, with our designs, pretrained LLMs can accurately perceive spatial distribution information and enable zero-shot execution of advanced spatial intelligence tasks, including urban planning, ecological analysis, traffic management, etc. We argue that multi-field knowledge, context length, and reasoning ability are key factors influencing LLM performances in urban analysis. We hope that SpatialLLM will provide a novel viable perspective for urban intelligent analysis and management. The code and dataset are available at this https URL. 

---
