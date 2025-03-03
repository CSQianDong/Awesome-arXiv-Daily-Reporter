# Contextualizing biological perturbation experiments through language 

**Authors**: Menghua Wu, Russell Littman, Jacob Levine, Lin Qiu, Tommaso Biancalani, David Richmond, Jan-Christian Huetter  

**Link**: [PDF](https://arxiv.org/pdf/2502.21290)  

**Abstract**: High-content perturbation experiments allow scientists to probe biomolecular systems at unprecedented resolution, but experimental and analysis costs pose significant barriers to widespread adoption. Machine learning has the potential to guide efficient exploration of the perturbation space and extract novel insights from these data. However, current approaches neglect the semantic richness of the relevant biology, and their objectives are misaligned with downstream biological analyses. In this paper, we hypothesize that large language models (LLMs) present a natural medium for representing complex biological relationships and rationalizing experimental outcomes. We propose PerturbQA, a benchmark for structured reasoning over perturbation experiments. Unlike current benchmarks that primarily interrogate existing knowledge, PerturbQA is inspired by open problems in perturbation modeling: prediction of differential expression and change of direction for unseen perturbations, and gene set enrichment. We evaluate state-of-the-art machine learning and statistical approaches for modeling perturbations, as well as standard LLM reasoning strategies, and we find that current methods perform poorly on PerturbQA. As a proof of feasibility, we introduce Summer (SUMMarize, retrievE, and answeR, a simple, domain-informed LLM framework that matches or exceeds the current state-of-the-art. Our code and data are publicly available at this https URL. 

---
# Re-evaluating Theory of Mind evaluation in large language models 

**Authors**: Jennifer Hu, Felix Sosa, Tomer Ullman  

**Link**: [PDF](https://arxiv.org/pdf/2502.21098)  

**Abstract**: The question of whether large language models (LLMs) possess Theory of Mind (ToM) -- often defined as the ability to reason about others' mental states -- has sparked significant scientific and public interest. However, the evidence as to whether LLMs possess ToM is mixed, and the recent growth in evaluations has not resulted in a convergence. Here, we take inspiration from cognitive science to re-evaluate the state of ToM evaluation in LLMs. We argue that a major reason for the disagreement on whether LLMs have ToM is a lack of clarity on whether models should be expected to match human behaviors, or the computations underlying those behaviors. We also highlight ways in which current evaluations may be deviating from "pure" measurements of ToM abilities, which also contributes to the confusion. We conclude by discussing several directions for future research, including the relationship between ToM and pragmatic communication, which could advance our understanding of artificial systems as well as human cognition. 

---
# Optimizing Large Language Models for ESG Activity Detection in Financial Texts 

**Authors**: Mattia Birti, Francesco Osborne, Andrea Maurino  

**Link**: [PDF](https://arxiv.org/pdf/2502.21112)  

**Abstract**: The integration of Environmental, Social, and Governance (ESG) factors into corporate decision-making is a fundamental aspect of sustainable finance. However, ensuring that business practices align with evolving regulatory frameworks remains a persistent challenge. AI-driven solutions for automatically assessing the alignment of sustainability reports and non-financial disclosures with specific ESG activities could greatly support this process. Yet, this task remains complex due to the limitations of general-purpose Large Language Models (LLMs) in domain-specific contexts and the scarcity of structured, high-quality datasets. In this paper, we investigate the ability of current-generation LLMs to identify text related to environmental activities. Furthermore, we demonstrate that their performance can be significantly enhanced through fine-tuning on a combination of original and synthetically generated data. To this end, we introduce ESG-Activities, a benchmark dataset containing 1,325 labelled text segments classified according to the EU ESG taxonomy. Our experimental results show that fine-tuning on ESG-Activities significantly enhances classification accuracy, with open models such as Llama 7B and Gemma 7B outperforming large proprietary solutions in specific configurations. These findings have important implications for financial analysts, policymakers, and AI researchers seeking to enhance ESG transparency and compliance through advanced natural language processing techniques. 

---
# ARIES: Autonomous Reasoning with LLMs on Interactive Thought Graph Environments 

**Authors**: Pedro Gimenes, Zeyu Cao, Jeffrey Wong, Yiren Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2502.21208)  

**Abstract**: Recent research has shown that LLM performance on reasoning tasks can be enhanced by scaling test-time compute. One promising approach, particularly with decomposable problems, involves arranging intermediate solutions as a graph on which transformations are performed to explore the solution space. However, prior works rely on pre-determined, task-specific transformation schedules which are subject to a set of searched hyperparameters. In this work, we view thought graph transformations as actions in a Markov decision process, and implement policy agents to drive effective action policies for the underlying reasoning LLM agent. In particular, we investigate the ability for another LLM to act as a policy agent on thought graph environments and introduce ARIES, a multi-agent architecture for reasoning with LLMs. In ARIES, reasoning LLM agents solve decomposed subproblems, while policy LLM agents maintain visibility of the thought graph states, and dynamically adapt the problem-solving strategy. Through extensive experiments, we observe that using off-the-shelf LLMs as policy agents with no supervised fine-tuning (SFT) can yield up to $29\%$ higher accuracy on HumanEval relative to static transformation schedules, as well as reducing inference costs by $35\%$ and avoid any search requirements. We also conduct a thorough analysis of observed failure modes, highlighting that limitations on LLM sizes and the depth of problem decomposition can be seen as challenges to scaling LLM-guided reasoning. 

---
# An LLM-based Delphi Study to Predict GenAI Evolution 

**Authors**: Francesco Bertolotti, Luca Mari  

**Link**: [PDF](https://arxiv.org/pdf/2502.21092)  

**Abstract**: Predicting the future trajectory of complex and rapidly evolving systems remains a significant challenge, particularly in domains where data is scarce or unreliable. This study introduces a novel approach to qualitative forecasting by leveraging Large Language Models to conduct Delphi studies. The methodology was applied to explore the future evolution of Generative Artificial Intelligence, revealing insights into key factors such as geopolitical tensions, economic disparities, regulatory frameworks, and ethical considerations. The results highlight how LLM-based Delphi studies can facilitate structured scenario analysis, capturing diverse perspectives while mitigating issues such as respondent fatigue. However, limitations emerge in terms of knowledge cutoffs, inherent biases, and sensitivity to initial conditions. While the approach provides an innovative means for structured foresight, this method could be also considered as a novel form of reasoning. further research is needed to refine its ability to manage heterogeneity, improve reliability, and integrate external data sources. 

---
# Merging Clinical Knowledge into Large Language Models for Medical Research and Applications: A Survey 

**Authors**: Qiyuan Li, Haijiang Liu, Caicai Guo, Deyu Chen, Meng Wang, Feng Gao, Jinguang Gu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20988)  

**Abstract**: Clinical knowledge is the collection of information learned from studies on the causes, prognosis, diagnosis, and treatment of diseases. This type of knowledge can improve curing performances, and promote physical health. With the emergence of large language models (LLMs), medical artificial intelligence (medical AI), which aims to apply academic medical AI systems to real-world medical scenarios, has entered a new age of development, resulting in excellent works such as DoctorGPT and Pangu-Drug from academic and industrial researches. However, the field lacks a comprehensive compendium and comparison of building medical AI systems from academia and industry. Therefore, this survey focuses on the building paradigms of medical AI systems including the use of clinical databases, datasets, training pipelines, integrating medical knowledge graphs, system applications, and evaluation systems. We hope that this survey can help relevant practical researchers understand the current performance of academic models in various fields of healthcare, as well as the potential problems and future directions for implementing these scientific achievements. 

---
# Transforming Tuberculosis Care: Optimizing Large Language Models For Enhanced Clinician-Patient Communication 

**Authors**: Daniil Filienko, Mahek Nizar, Javier Roberti, Denise Galdamez, Haroon Jakher, Sarah Iribarren, Weichao Yuwen, Martine De Cock  

**Link**: [PDF](https://arxiv.org/pdf/2502.21236)  

**Abstract**: Tuberculosis (TB) is the leading cause of death from an infectious disease globally, with the highest burden in low- and middle-income countries. In these regions, limited healthcare access and high patient-to-provider ratios impede effective patient support, communication, and treatment completion. To bridge this gap, we propose integrating a specialized Large Language Model into an efficacious digital adherence technology to augment interactive communication with treatment supporters. This AI-powered approach, operating within a human-in-the-loop framework, aims to enhance patient engagement and improve TB treatment outcomes. 

---
# Why Trust in AI May Be Inevitable 

**Authors**: Nghi Truong, Phanish Puranam, Ilia Testlin  

**Link**: [PDF](https://arxiv.org/pdf/2502.20701)  

**Abstract**: In human-AI interactions, explanation is widely seen as necessary for enabling trust in AI systems. We argue that trust, however, may be a pre-requisite because explanation is sometimes impossible. We derive this result from a formalization of explanation as a search process through knowledge networks, where explainers must find paths between shared concepts and the concept to be explained, within finite time. Our model reveals that explanation can fail even under theoretically ideal conditions - when actors are rational, honest, motivated, can communicate perfectly, and possess overlapping knowledge. This is because successful explanation requires not just the existence of shared knowledge but also finding the connection path within time constraints, and it can therefore be rational to cease attempts at explanation before the shared knowledge is discovered. This result has important implications for human-AI interaction: as AI systems, particularly Large Language Models, become more sophisticated and able to generate superficially compelling but spurious explanations, humans may default to trust rather than demand genuine explanations. This creates risks of both misplaced trust and imperfect knowledge integration. 

---
# NutriGen: Personalized Meal Plan Generator Leveraging Large Language Models to Enhance Dietary and Nutritional Adherence 

**Authors**: Saman Khamesian, Asiful Arefeen, Stephanie M. Carpenter, Hassan Ghasemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2502.20601)  

**Abstract**: Maintaining a balanced diet is essential for overall health, yet many individuals struggle with meal planning due to nutritional complexity, time constraints, and lack of dietary knowledge. Personalized food recommendations can help address these challenges by tailoring meal plans to individual preferences, habits, and dietary restrictions. However, existing dietary recommendation systems often lack adaptability, fail to consider real-world constraints such as food ingredient availability, and require extensive user input, making them impractical for sustainable and scalable daily use. To address these limitations, we introduce NutriGen, a framework based on large language models (LLM) designed to generate personalized meal plans that align with user-defined dietary preferences and constraints. By building a personalized nutrition database and leveraging prompt engineering, our approach enables LLMs to incorporate reliable nutritional references like the USDA nutrition database while maintaining flexibility and ease-of-use. We demonstrate that LLMs have strong potential in generating accurate and user-friendly food recommendations, addressing key limitations in existing dietary recommendation systems by providing structured, practical, and scalable meal plans. Our evaluation shows that Llama 3.1 8B and GPT-3.5 Turbo achieve the lowest percentage errors of 1.55\% and 3.68\%, respectively, producing meal plans that closely align with user-defined caloric targets while minimizing deviation and improving precision. Additionally, we compared the performance of DeepSeek V3 against several established models to evaluate its potential in personalized nutrition planning. 

---
# MV-MATH: Evaluating Multimodal Math Reasoning in Multi-Visual Contexts 

**Authors**: Peijie Wang, Zhongzhi Li, Fei Yin, Dekang Ran, Chenglin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20808)  

**Abstract**: Multimodal Large Language Models (MLLMs) have shown promising capabilities in mathematical reasoning within visual contexts across various datasets. However, most existing multimodal math benchmarks are limited to single-visual contexts, which diverges from the multi-visual scenarios commonly encountered in real-world mathematical applications. To address this gap, we introduce MV-MATH: a meticulously curated dataset of 2,009 high-quality mathematical problems. Each problem integrates multiple images interleaved with text, derived from authentic K-12 scenarios, and enriched with detailed annotations. MV-MATH includes multiple-choice, free-form, and multi-step questions, covering 11 subject areas across 3 difficulty levels, and serves as a comprehensive and rigorous benchmark for assessing MLLMs' mathematical reasoning in multi-visual contexts. Through extensive experimentation, we observe that MLLMs encounter substantial challenges in multi-visual math tasks, with a considerable performance gap relative to human capabilities on MV-MATH. Furthermore, we analyze the performance and error patterns of various models, providing insights into MLLMs' mathematical reasoning capabilities within multi-visual settings. 

---
# Automatic database description generation for Text-to-SQL 

**Authors**: Yingqi Gao, Zhiling Luo  

**Link**: [PDF](https://arxiv.org/pdf/2502.20657)  

**Abstract**: In the context of the Text-to-SQL task, table and column descriptions are crucial for bridging the gap between natural language and database schema. This report proposes a method for automatically generating effective database descriptions when explicit descriptions are unavailable. The proposed method employs a dual-process approach: a coarse-to-fine process, followed by a fine-to-coarse process. The coarse-to-fine approach leverages the inherent knowledge of LLM to guide the understanding process from databases to tables and finally to columns. This approach provides a holistic understanding of the database structure and ensures contextual alignment. Conversely, the fine-to-coarse approach starts at the column level, offering a more accurate and nuanced understanding when stepping back to the table level. Experimental results on the Bird benchmark indicate that using descriptions generated by the proposed improves SQL generation accuracy by 0.93\% compared to not using descriptions, and achieves 37\% of human-level performance. The source code is publicly available at this https URL. 

---
# ProAI: Proactive Multi-Agent Conversational AI with Structured Knowledge Base for Psychiatric Diagnosis 

**Authors**: Yuqi Wu, Guangya Wan, Jingjing Li, Shengming Zhao, Lingfeng Ma, Tianyi Ye, Ion Pop, Yanbo Zhang, Jie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20689)  

**Abstract**: Most LLM-driven conversational AI systems operate reactively, responding to user prompts without guiding the interaction. Most LLM-driven conversational AI systems operate reactively, responding to user prompts without guiding the interaction. However, many real-world applications-such as psychiatric diagnosis, consulting, and interviews-require AI to take a proactive role, asking the right questions and steering conversations toward specific objectives. Using mental health differential diagnosis as an application context, we introduce ProAI, a goal-oriented, proactive conversational AI framework. ProAI integrates structured knowledge-guided memory, multi-agent proactive reasoning, and a multi-faceted evaluation strategy, enabling LLMs to engage in clinician-style diagnostic reasoning rather than simple response generation. Through simulated patient interactions, user experience assessment, and professional clinical validation, we demonstrate that ProAI achieves up to 83.3% accuracy in mental disorder differential diagnosis while maintaining professional and empathetic interaction standards. These results highlight the potential for more reliable, adaptive, and goal-driven AI diagnostic assistants, advancing LLMs beyond reactive dialogue systems. 

---
# FANformer: Improving Large Language Models Through Effective Periodicity Modeling 

**Authors**: Yihong Dong, Ge Li, Xue Jiang, Yongding Tao, Kechi Zhang, Hao Zhu, Huanyu Liu, Jiazheng Ding, Jia Li, Jinliang Deng, Hong Mei  

**Link**: [PDF](https://arxiv.org/pdf/2502.21309)  

**Abstract**: Periodicity, as one of the most important basic characteristics, lays the foundation for facilitating structured knowledge acquisition and systematic cognitive processes within human learning paradigms. However, the potential flaws of periodicity modeling in Transformer affect the learning efficiency and establishment of underlying principles from data for large language models (LLMs) built upon it. In this paper, we demonstrate that integrating effective periodicity modeling can improve the learning efficiency and performance of LLMs. We introduce FANformer, which integrates Fourier Analysis Network (FAN) into attention mechanism to achieve efficient periodicity modeling, by modifying the feature projection process of attention mechanism. Extensive experimental results on language modeling show that FANformer consistently outperforms Transformer when scaling up model size and training tokens, underscoring its superior learning efficiency. To further validate the effectiveness of FANformer, we pretrain a FANformer-1B on 1 trillion tokens. FANformer-1B exhibits marked improvements on downstream tasks compared to open-source LLMs with similar model parameters or training tokens. The results position FANformer as an effective and promising architecture for advancing LLMs. 

---
# ByteScale: Efficient Scaling of LLM Training with a 2048K Context Length on More Than 12,000 GPUs 

**Authors**: Hao Ge, Junda Feng, Qi Huang, Fangcheng Fu, Xiaonan Nie, Lei Zuo, Haibin Lin, Bin Cui, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.21231)  

**Abstract**: Scaling long-context ability is essential for Large Language Models (LLMs). To amortize the memory consumption across multiple devices in long-context training, inter-data partitioning (a.k.a. Data Parallelism) and intra-data partitioning (a.k.a. Context Parallelism) are commonly used. Current training frameworks predominantly treat the two techniques as orthogonal, and establish static communication groups to organize the devices as a static mesh (e.g., a 2D mesh). However, the sequences for LLM training typically vary in lengths, no matter for texts, multi-modalities or reinforcement learning. The mismatch between data heterogeneity and static mesh causes redundant communication and imbalanced computation, degrading the training efficiency.
In this work, we introduce ByteScale, an efficient, flexible, and scalable LLM training framework for large-scale mixed training of long and short sequences. The core of ByteScale is a novel parallelism strategy, namely Hybrid Data Parallelism (HDP), which unifies the inter- and intra-data partitioning with a dynamic mesh design. In particular, we build a communication optimizer, which eliminates the redundant communication for short sequences by data-aware sharding and dynamic communication, and further compresses the communication cost for long sequences by selective offloading. Besides, we also develop a balance scheduler to mitigate the imbalanced computation by parallelism-aware data assignment. We evaluate ByteScale with the model sizes ranging from 7B to 141B, context lengths from 256K to 2048K, on a production cluster with more than 12,000 GPUs. Experiment results show that ByteScale outperforms the state-of-the-art training system by up to 7.89x. 

---
# ECLeKTic: a Novel Challenge Set for Evaluation of Cross-Lingual Knowledge Transfer 

**Authors**: Omer Goldman, Uri Shaham, Dan Malkin, Sivan Eiger, Avinatan Hassidim, Yossi Matias, Joshua Maynez, Adi Mayrav Gilady, Jason Riesa, Shruti Rijhwani, Laura Rimell, Idan Szpektor, Reut Tsarfaty, Matan Eyal  

**Link**: [PDF](https://arxiv.org/pdf/2502.21228)  

**Abstract**: To achieve equitable performance across languages, multilingual large language models (LLMs) must be able to abstract knowledge beyond the language in which it was acquired. However, the current literature lacks reliable ways to measure LLMs' capability of cross-lingual knowledge transfer. To that end, we present ECLeKTic, a multilingual closed-book QA (CBQA) dataset that Evaluates Cross-Lingual Knowledge Transfer in a simple, black-box manner. We detected information with uneven coverage across languages by controlling for presence and absence of Wikipedia articles in 12 languages. We generated knowledge-seeking questions in a source language, for which the answer appears in a relevant Wikipedia article and translated them to all other 11 languages, for which the respective Wikipedias lack equivalent articles. Assuming that Wikipedia reflects the prominent knowledge in the LLM's training data, to solve ECLeKTic's CBQA task the model is required to transfer knowledge between languages. Experimenting with 8 LLMs, we show that SOTA models struggle to effectively share knowledge across, languages even if they can predict the answer well for queries in the same language the knowledge was acquired in. 

---
# Large Language Model Strategic Reasoning Evaluation through Behavioral Game Theory 

**Authors**: Jingru Jia, Zehua Yuan, Junhao Pan, Paul E. McNamara, Deming Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20432)  

**Abstract**: Strategic decision-making involves interactive reasoning where agents adapt their choices in response to others, yet existing evaluations of large language models (LLMs) often emphasize Nash Equilibrium (NE) approximation, overlooking the mechanisms driving their strategic choices. To bridge this gap, we introduce an evaluation framework grounded in behavioral game theory, disentangling reasoning capability from contextual effects. Testing 22 state-of-the-art LLMs, we find that GPT-o3-mini, GPT-o1, and DeepSeek-R1 dominate most games yet also demonstrate that the model scale alone does not determine performance. In terms of prompting enhancement, Chain-of-Thought (CoT) prompting is not universally effective, as it increases strategic reasoning only for models at certain levels while providing limited gains elsewhere. Additionally, we investigate the impact of encoded demographic features on the models, observing that certain assignments impact the decision-making pattern. For instance, GPT-4o shows stronger strategic reasoning with female traits than males, while Gemma assigns higher reasoning levels to heterosexual identities compared to other sexual orientations, indicating inherent biases. These findings underscore the need for ethical standards and contextual alignment to balance improved reasoning with fairness. 

---
# PASemiQA: Plan-Assisted Agent for Question Answering on Semi-Structured Data with Text and Relational Information 

**Authors**: Hansi Yang, Qi Zhang, Wei Jiang, Jianguo Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.21087)  

**Abstract**: Large language models (LLMs) have shown impressive abilities in answering questions across various domains, but they often encounter hallucination issues on questions that require professional and up-to-date knowledge. To address this limitation, retrieval-augmented generation (RAG) techniques have been proposed, which retrieve relevant information from external sources to inform their responses. However, existing RAG methods typically focus on a single type of external data, such as vectorized text database or knowledge graphs, and cannot well handle real-world questions on semi-structured data containing both text and relational information. To bridge this gap, we introduce PASemiQA, a novel approach that jointly leverages text and relational information in semi-structured data to answer questions. PASemiQA first generates a plan to identify relevant text and relational information to answer the question in semi-structured data, and then uses an LLM agent to traverse the semi-structured data and extract necessary information. Our empirical results demonstrate the effectiveness of PASemiQA across different semi-structured datasets from various domains, showcasing its potential to improve the accuracy and reliability of question answering systems on semi-structured data. 

---
# Beyond Words: A Latent Memory Approach to Internal Reasoning in LLMs 

**Authors**: José I. Orlicki  

**Link**: [PDF](https://arxiv.org/pdf/2502.21030)  

**Abstract**: Recent advances in large language models (LLMs) have popularized the chain-of-thought (CoT) paradigm, in which models produce explicit reasoning steps in natural language. Although this approach improves interpretability and facilitates external auditing, it may not represent the most computationally efficient method for internal reasoning. In contrast, human cognition relies on implicit mental representations that recall past sensory and episodic information without requiring complete verbalization. In this paper, we propose a framework that integrates implicit mental representations into the internal reasoning processes of LLMs. Preliminary experiments indicate that incorporating an Implicit Memory Module (IMM) into a simple GPT model yields a reduction of between 35% and 57% in final training loss compared to a regular GPT baseline. The addition of an explicit interpretability channel (e.g., a chain-of-thought decoder) is straightforward to implement within this approach. We outline theoretical foundations, propose technical mechanisms to scale the memory module, and discuss how these ideas may lead to more efficient and robust reasoning, with optional future extensions for explicit auditability. 

---
# A Deep User Interface for Exploring LLaMa 

**Authors**: Divya Perumal, Swaroop Panda  

**Link**: [PDF](https://arxiv.org/pdf/2502.20938)  

**Abstract**: The growing popularity and widespread adoption of large language models (LLMs) necessitates the development of tools that enhance the effectiveness of user interactions with these models. Understanding the structures and functions of these models poses a significant challenge for users. Visual analytics-driven tools enables users to explore and compare, facilitating better decision-making. This paper presents a visual analytics-driven tool equipped with interactive controls for key hyperparameters, including top-p, frequency and presence penalty, enabling users to explore, examine and compare the outputs of LLMs. In a user study, we assessed the tool's effectiveness, which received favorable feedback for its visual design, with particular commendation for the interface layout and ease of navigation. Additionally, the feedback provided valuable insights for enhancing the effectiveness of Human-LLM interaction tools. 

---
# UoR-NCL at SemEval-2025 Task 1: Using Generative LLMs and CLIP Models for Multilingual Multimodal Idiomaticity Representation 

**Authors**: Thanet Markchom, Tong Wu, Liting Huang, Huizhi Liang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20984)  

**Abstract**: SemEval-2025 Task 1 focuses on ranking images based on their alignment with a given nominal compound that may carry idiomatic meaning in both English and Brazilian Portuguese. To address this challenge, this work uses generative large language models (LLMs) and multilingual CLIP models to enhance idiomatic compound representations. LLMs generate idiomatic meanings for potentially idiomatic compounds, enriching their semantic interpretation. These meanings are then encoded using multilingual CLIP models, serving as representations for image ranking. Contrastive learning and data augmentation techniques are applied to fine-tune these embeddings for improved performance. Experimental results show that multimodal representations extracted through this method outperformed those based solely on the original nominal compounds. The fine-tuning approach shows promising outcomes but is less effective than using embeddings without fine-tuning. The source code used in this paper is available at this https URL. 

---
# LADs: Leveraging LLMs for AI-Driven DevOps 

**Authors**: Ahmad Faraz Khan, Azal Ahmad Khan, Anas Mohamed, Haider Ali, Suchithra Moolinti, Sabaat Haroon, Usman Tahir, Mattia Fazzini, Ali R. Butt, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2502.20825)  

**Abstract**: Automating cloud configuration and deployment remains a critical challenge due to evolving infrastructures, heterogeneous hardware, and fluctuating workloads. Existing solutions lack adaptability and require extensive manual tuning, leading to inefficiencies and misconfigurations. We introduce LADs, the first LLM-driven framework designed to tackle these challenges by ensuring robustness, adaptability, and efficiency in automated cloud management. Instead of merely applying existing techniques, LADs provides a principled approach to configuration optimization through in-depth analysis of what optimization works under which conditions. By leveraging Retrieval-Augmented Generation, Few-Shot Learning, Chain-of-Thought, and Feedback-Based Prompt Chaining, LADs generates accurate configurations and learns from deployment failures to iteratively refine system settings. Our findings reveal key insights into the trade-offs between performance, cost, and scalability, helping practitioners determine the right strategies for different deployment scenarios. For instance, we demonstrate how prompt chaining-based adaptive feedback loops enhance fault tolerance in multi-tenant environments and how structured log analysis with example shots improves configuration accuracy. Through extensive evaluations, LADs reduces manual effort, optimizes resource utilization, and improves system reliability. By open-sourcing LADs, we aim to drive further innovation in AI-powered DevOps automation. 

---
# Collective Reasoning Among LLMs A Framework for Answer Validation Without Ground Truth 

**Authors**: Seyed Pouyan Mousavi Davoudi, Alireza Shafiee Fard, Alireza Amiri-Margavi  

**Link**: [PDF](https://arxiv.org/pdf/2502.20758)  

**Abstract**: We present a collaborative framework where multiple large language models, namely GPT-4-0125-preview, Meta-LLaMA-3-70B-Instruct, Claude-3-Opus, and Gemini-1.5-Flash, work together to generate and respond to complex PhD-level probability questions in the absence of definitive ground truth. This study explores how inter-model consensus enhances response reliability and serves as a proxy for assessing the quality of generated questions. To quantify agreement and consistency, we employ statistical methods including chi-square tests, Fleiss' Kappa, and confidence interval analysis, measuring both response precision and question clarity. Our findings highlight that Claude and Gemini generate well-structured and less ambiguous questions, leading to higher inter-model agreement. This is reflected in their narrower confidence intervals and stronger alignment with answering models. Conversely, LLaMA demonstrates increased variability and lower reliability in question formulation, as indicated by broader confidence intervals and reduced consensus rates. These results suggest that multi-model collaboration not only enhances the reliability of responses but also provides a valuable framework for assessing and improving question quality in the absence of explicit ground truth. This research offers meaningful insights into optimizing AI-driven reasoning through collaborative large-language model interactions. 

---
# Teach-to-Reason with Scoring: Self-Explainable Rationale-Driven Multi-Trait Essay Scoring 

**Authors**: Heejin Do, Sangwon Ryu, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.20748)  

**Abstract**: Multi-trait automated essay scoring (AES) systems provide a fine-grained evaluation of an essay's diverse aspects. While they excel in scoring, prior systems fail to explain why specific trait scores are assigned. This lack of transparency leaves instructors and learners unconvinced of the AES outputs, hindering their practical use. To address this, we propose a self-explainable Rationale-Driven Multi-trait automated Essay scoring (RaDME) framework. RaDME leverages the reasoning capabilities of large language models (LLMs) by distilling them into a smaller yet effective scorer. This more manageable student model is optimized to sequentially generate a trait score followed by the corresponding rationale, thereby inherently learning to select a more justifiable score by considering the subsequent rationale during training. Our findings indicate that while LLMs underperform in direct AES tasks, they excel in rationale generation when provided with precise numerical scores. Thus, RaDME integrates the superior reasoning capacities of LLMs into the robust scoring accuracy of an optimized smaller model. Extensive experiments demonstrate that RaDME achieves both accurate and adequate reasoning while supporting high-quality multi-trait scoring, significantly enhancing the transparency of AES. 

---
# Transformers Learn to Implement Multi-step Gradient Descent with Chain of Thought 

**Authors**: Jianhao Huang, Zixuan Wang, Jason D. Lee  

**Link**: [PDF](https://arxiv.org/pdf/2502.21212)  

**Abstract**: Chain of Thought (CoT) prompting has been shown to significantly improve the performance of large language models (LLMs), particularly in arithmetic and reasoning tasks, by instructing the model to produce intermediate reasoning steps. Despite the remarkable empirical success of CoT and its theoretical advantages in enhancing expressivity, the mechanisms underlying CoT training remain largely unexplored. In this paper, we study the training dynamics of transformers over a CoT objective on an in-context weight prediction task for linear regression. We prove that while a one-layer linear transformer without CoT can only implement a single step of gradient descent (GD) and fails to recover the ground-truth weight vector, a transformer with CoT prompting can learn to perform multi-step GD autoregressively, achieving near-exact recovery. Furthermore, we show that the trained transformer effectively generalizes on the unseen data. With our technique, we also show that looped transformers significantly improve final performance compared to transformers without looping in the in-context learning of linear regression. Empirically, we demonstrate that CoT prompting yields substantial performance improvements. 

---
# JAM: Controllable and Responsible Text Generation via Causal Reasoning and Latent Vector Manipulation 

**Authors**: Yingbing Huang, Deming Chen, Abhishek K. Umrawal  

**Link**: [PDF](https://arxiv.org/pdf/2502.20684)  

**Abstract**: While large language models (LLMs) have made significant strides in generating coherent and contextually relevant text, they often function as opaque black boxes, trained on vast unlabeled datasets with statistical objectives, lacking an interpretable framework for responsible control. In this paper, we introduce JAM (Just A Move), a novel framework that interprets and controls text generation by integrating cause-effect analysis within the latent space of LLMs. Based on our observations, we uncover the inherent causality in LLM generation, which is critical for producing responsible and realistic outputs. Moreover, we explore latent vectors as fundamental components in LLM architectures, aiming to understand and manipulate them for more effective and efficient controllable text generation. We evaluate our framework using a range of tools, including the HHH criteria, toxicity reduction benchmarks, and GPT-4 alignment measures. Our results show that JAM achieves up to a 22% improvement over previous Controllable Text Generation (CTG) methods across multiple quantitative metrics and human-centric evaluations. Furthermore, JAM demonstrates greater computational efficiency compared to other CTG methods. These results highlight the effectiveness and efficiency of JAM for responsible and realistic text generation, paving the way for more interpretable and controllable models. 

---
# Consistency Evaluation of News Article Summaries Generated by Large (and Small) Language Models 

**Authors**: Colleen Gilhuly, Haleh Shahzad  

**Link**: [PDF](https://arxiv.org/pdf/2502.20647)  

**Abstract**: Text summarizing is a critical Natural Language Processing (NLP) task with applications ranging from information retrieval to content generation. Large Language Models (LLMs) have shown remarkable promise in generating fluent abstractive summaries but they can produce hallucinated details not grounded in the source text. Regardless of the method of generating a summary, high quality automated evaluations remain an open area of investigation. This paper embarks on an exploration of text summarization with a diverse set of techniques, including TextRank, BART, Mistral-7B-Instruct, and OpenAI GPT-3.5-Turbo. The generated summaries are evaluated using traditional metrics such as the Recall-Oriented Understudy for Gisting Evaluation (ROUGE) Score and Bidirectional Encoder Representations from Transformers (BERT) Score, as well as LLM-powered evaluation methods that directly assess a generated summary's consistency with the source text. We introduce a meta evaluation score which directly assesses the performance of the LLM evaluation system (prompt + model). We find that that all summarization models produce consistent summaries when tested on the XL-Sum dataset, exceeding the consistency of the reference summaries. 

---
# Leveraging Large Language Models for Building Interpretable Rule-Based Data-to-Text Systems 

**Authors**: Jędrzej Warczyński, Mateusz Lango, Ondrej Dusek  

**Link**: [PDF](https://arxiv.org/pdf/2502.20609)  

**Abstract**: We introduce a simple approach that uses a large language model (LLM) to automatically implement a fully interpretable rule-based data-to-text system in pure Python. Experimental evaluation on the WebNLG dataset showed that such a constructed system produces text of better quality (according to the BLEU and BLEURT metrics) than the same LLM prompted to directly produce outputs, and produces fewer hallucinations than a BART language model fine-tuned on the same data. Furthermore, at runtime, the approach generates text in a fraction of the processing time required by neural approaches, using only a single CPU 

---
# Promote, Suppress, Iterate: How Language Models Answer One-to-Many Factual Queries 

**Authors**: Tianyi Lorena Yan, Robin Jia  

**Link**: [PDF](https://arxiv.org/pdf/2502.20475)  

**Abstract**: To answer one-to-many factual queries (e.g., listing cities of a country), a language model (LM) must simultaneously recall knowledge and avoid repeating previous answers. How are these two subtasks implemented and integrated internally? Across multiple datasets and models, we identify a promote-then-suppress mechanism: the model first recalls all answers, and then suppresses previously generated ones. Specifically, LMs use both the subject and previous answer tokens to perform knowledge recall, with attention propagating subject information and MLPs promoting the answers. Then, attention attends to and suppresses previous answer tokens, while MLPs amplify the suppression signal. Our mechanism is corroborated by extensive experimental evidence: in addition to using early decoding and causal tracing, we analyze how components use different tokens by introducing both \emph{Token Lens}, which decodes aggregated attention updates from specified tokens, and a knockout method that analyzes changes in MLP outputs after removing attention to specified tokens. Overall, we provide new insights into how LMs' internal components interact with different input tokens to support complex factual recall. Code is available at this https URL. 

---
# Pause-Tuning for Long-Context Comprehension: A Lightweight Approach to LLM Attention Recalibration 

**Authors**: James Begin, Namit Agrawal, Eshan Singh, Yicheng Fu, Sean O'Brien, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20405)  

**Abstract**: LLMs have demonstrated remarkable proficiency in understanding tasks but continue to struggle with long-context comprehension, particularly with content located in the middle of extensive inputs. This limitation, known as the Lost-in-the-Middle (LITM) problem, hinders models from fully processing and utilizing information across lengthy contexts. To address this issue, we introduce pause-tuning, a technique that redistributes attention to enhance comprehension of long-context inputs. Our approach involves fine-tuning language models on datasets with artificially inserted pause tokens, which serve to segment the input into smaller, more manageable parts. We evaluate pause-tuning against alternative approaches using the Needle-in-a-Haystack benchmark, where models must retrieve information embedded within contexts of up to 128K tokens. Experimental results demonstrate significant performance gains, with the LLaMA 3.2 3B Instruct model and the LLaMA 3.1 8B Instruct model improving by 10.61% and 3.57% respectively on average, suggesting that pause-tuning successfully enhances attention redistribution and improves long-context retention. The code and data are available at this https URL. 

---
# SEKI: Self-Evolution and Knowledge Inspiration based Neural Architecture Search via Large Language Models 

**Authors**: Zicheng Cai, Yaohua Tang, Yutao Lai, Hua Wang, Zhi Chen, Hao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20422)  

**Abstract**: We introduce SEKI, a novel large language model (LLM)-based neural architecture search (NAS) method. Inspired by the chain-of-thought (CoT) paradigm in modern LLMs, SEKI operates in two key stages: self-evolution and knowledge distillation. In the self-evolution stage, LLMs initially lack sufficient reference examples, so we implement an iterative refinement mechanism that enhances architectures based on performance feedback. Over time, this process accumulates a repository of high-performance architectures. In the knowledge distillation stage, LLMs analyze common patterns among these architectures to generate new, optimized designs. Combining these two stages, SEKI greatly leverages the capacity of LLMs on NAS and without requiring any domain-specific data. Experimental results show that SEKI achieves state-of-the-art (SOTA) performance across various datasets and search spaces while requiring only 0.05 GPU-days, outperforming existing methods in both efficiency and accuracy. Furthermore, SEKI demonstrates strong generalization capabilities, achieving SOTA-competitive results across multiple tasks. 

---
# Semantic Volume: Quantifying and Detecting both External and Internal Uncertainty in LLMs 

**Authors**: Xiaomin Li, Zhou Yu, Ziji Zhang, Yingying Zhuang, Swair Shah, Anurag Beniwal  

**Link**: [PDF](https://arxiv.org/pdf/2502.21239)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance across diverse tasks by encoding vast amounts of factual knowledge. However, they are still prone to hallucinations, generating incorrect or misleading information, often accompanied by high uncertainty. Existing methods for hallucination detection primarily focus on quantifying internal uncertainty, which arises from missing or conflicting knowledge within the model. However, hallucinations can also stem from external uncertainty, where ambiguous user queries lead to multiple possible interpretations. In this work, we introduce Semantic Volume, a novel mathematical measure for quantifying both external and internal uncertainty in LLMs. Our approach perturbs queries and responses, embeds them in a semantic space, and computes the determinant of the Gram matrix of the embedding vectors, capturing their dispersion as a measure of uncertainty. Our framework provides a generalizable and unsupervised uncertainty detection method without requiring white-box access to LLMs. We conduct extensive experiments on both external and internal uncertainty detection, demonstrating that our Semantic Volume method consistently outperforms existing baselines in both tasks. Additionally, we provide theoretical insights linking our measure to differential entropy, unifying and extending previous sampling-based uncertainty measures such as the semantic entropy. Semantic Volume is shown to be a robust and interpretable approach to improving the reliability of LLMs by systematically detecting uncertainty in both user queries and model responses. 

---
# LLM Post-Training: A Deep Dive into Reasoning Large Language Models 

**Authors**: Komal Kumar, Tajamul Ashraf, Omkar Thawakar, Rao Muhammad Anwer, Hisham Cholakkal, Mubarak Shah, Ming-Hsuan Yang, Phillip H.S. Torr, Salman Khan, Fahad Shahbaz Khan  

**Link**: [PDF](https://arxiv.org/pdf/2502.21321)  

**Abstract**: Large Language Models (LLMs) have transformed the natural language processing landscape and brought to life diverse applications. Pretraining on vast web-scale data has laid the foundation for these models, yet the research community is now increasingly shifting focus toward post-training techniques to achieve further breakthroughs. While pretraining provides a broad linguistic foundation, post-training methods enable LLMs to refine their knowledge, improve reasoning, enhance factual accuracy, and align more effectively with user intents and ethical considerations. Fine-tuning, reinforcement learning, and test-time scaling have emerged as critical strategies for optimizing LLMs performance, ensuring robustness, and improving adaptability across various real-world tasks. This survey provides a systematic exploration of post-training methodologies, analyzing their role in refining LLMs beyond pretraining, addressing key challenges such as catastrophic forgetting, reward hacking, and inference-time trade-offs. We highlight emerging directions in model alignment, scalable adaptation, and inference-time reasoning, and outline future research directions. We also provide a public repository to continually track developments in this fast-evolving field: this https URL. 

---
# PersuasiveToM: A Benchmark for Evaluating Machine Theory of Mind in Persuasive Dialogues 

**Authors**: Fangxu Yu, Lai Jiang, Shenyi Huang, Zhen Wu, Xinyu Dai  

**Link**: [PDF](https://arxiv.org/pdf/2502.21017)  

**Abstract**: The ability to understand and predict the mental states of oneself and others, known as the Theory of Mind (ToM), is crucial for effective social interactions. Recent research has emerged to evaluate whether Large Language Models (LLMs) exhibit a form of ToM. Although recent studies have evaluated ToM in LLMs, existing benchmarks focus predominantly on physical perception with principles guided by the Sally-Anne test in synthetic stories and conversations, failing to capture the complex psychological activities of mental states in real-life social interactions. To mitigate this gap, we propose PersuasiveToM, a benchmark designed to evaluate the ToM abilities of LLMs in persuasive dialogues. Our framework introduces two categories of questions: (1) ToM Reasoning, assessing the capacity of LLMs to track evolving mental states (e.g., desire shifts in persuadees), and (2) ToM Application, evaluating whether LLMs can take advantage of inferred mental states to select effective persuasion strategies (e.g., emphasize rarity) and evaluate the effectiveness of persuasion strategies. Experiments across eight state-of-the-art LLMs reveal that while models excel on multiple questions, they struggle to answer questions that need tracking the dynamics and shifts of mental states and understanding the mental states in the whole dialogue comprehensively. Our aim with PersuasiveToM is to allow an effective evaluation of the ToM reasoning ability of LLMs with more focus on complex psychological activities. Our code is available at this https URL. 

---
# Arabizi vs LLMs: Can the Genie Understand the Language of Aladdin? 

**Authors**: Perla Al Almaoui, Pierrette Bouillon, Simon Hengchen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20973)  

**Abstract**: In this era of rapid technological advancements, communication continues to evolve as new linguistic phenomena emerge. Among these is Arabizi, a hybrid form of Arabic that incorporates Latin characters and numbers to represent the spoken dialects of Arab communities. Arabizi is widely used on social media and allows people to communicate in an informal and dynamic way, but it poses significant challenges for machine translation due to its lack of formal structure and deeply embedded cultural nuances. This case study arises from a growing need to translate Arabizi for gisting purposes. It evaluates the capacity of different LLMs to decode and translate Arabizi, focusing on multiple Arabic dialects that have rarely been studied up until now. Using a combination of human evaluators and automatic metrics, this research project investigates the model's performance in translating Arabizi into both Modern Standard Arabic and English. Key questions explored include which dialects are translated most effectively and whether translations into English surpass those into Arabic. 

---
# Beware of Your Po! Measuring and Mitigating AI Safety Risks in Role-Play Fine-Tuning of LLMs 

**Authors**: Weixiang Zhao, Yulin Hu, Yang Deng, Jiahe Guo, Xingyu Sui, Xinyang Han, An Zhang, Yanyan Zhao, Bing Qin, Tat-Seng Chua, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20968)  

**Abstract**: Role-playing enables large language models (LLMs) to engage users in immersive and personalized interactions, but it also introduces significant safety risks. Existing role-play fine-tuning techniques improve role adaptability but may degrade safety performance, particularly for villainous characters. In this work, we conduct the first comprehensive assessment of role-play fine-tuning risks by training 95 role-specific LLMs using RoleBench. Our experiments reveal that role-play fine-tuning leads to a noticeable decline in safety performance, with safety risks varying based on character traits. To tackle this challenge, we propose Safety-Aware Role-Play Fine-Tuning (SaRFT), a novel method designed to balance role-playing capabilities and safety. Extensive experiments on LLaMA-3-8B-Instruct, Gemma-2-9B-it, and Qwen2.5-7B-Instruct demonstrate that SaRFT consistently outperforms state-of-the-art baselines under both LoRA and full-parameter fine-tuning settings. Our findings highlight the necessity of role-adaptive safety measures and provide insights into mitigating role-specific safety risks in role-playing LLMs. 

---
# A database to support the evaluation of gender biases in GPT-4o output 

**Authors**: Luise Mehner, Lena Alicija Philine Fiedler, Sabine Ammon, Dorothea Kolossa  

**Link**: [PDF](https://arxiv.org/pdf/2502.20898)  

**Abstract**: The widespread application of Large Language Models (LLMs) involves ethical risks for users and societies. A prominent ethical risk of LLMs is the generation of unfair language output that reinforces or exacerbates harm for members of disadvantaged social groups through gender biases (Weidinger et al., 2022; Bender et al., 2021; Kotek et al., 2023). Hence, the evaluation of the fairness of LLM outputs with respect to such biases is a topic of rising interest. To advance research in this field, promote discourse on suitable normative bases and evaluation methodologies, and enhance the reproducibility of related studies, we propose a novel approach to database construction. This approach enables the assessment of gender-related biases in LLM-generated language beyond merely evaluating their degree of neutralization. 

---
# ProBench: Benchmarking Large Language Models in Competitive Programming 

**Authors**: Lei Yang, Renren Jin, Ling Shi, Jianxiang Peng, Yue Chen, Deyi Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2502.20868)  

**Abstract**: With reasoning language models such as OpenAI-o3 and DeepSeek-R1 emerging, large language models (LLMs) have entered a new phase of development. However, existing benchmarks for coding evaluation are gradually inadequate to assess the capability of advanced LLMs in code reasoning. To bridge the gap for high-level code reasoning assessment, we propose ProBench to benchmark LLMs in competitive programming, drawing inspiration from the International Collegiate Programming Contest. ProBench collects a comprehensive set of competitive programming problems from Codeforces, Luogu, and Nowcoder platforms during the period from July to December 2024, obtaining real test results through online submissions to ensure the fairness and accuracy of the evaluation. We establish a unified problem attribute system, including difficulty grading and algorithm tagging. With carefully collected and annotated data in ProBench, we systematically assess 9 latest LLMs in competitive programming across multiple dimensions, including thought chain analysis, error type diagnosis, and reasoning depth evaluation. Experimental results show that QwQ-32B-Preview achieves the best score of 20.93 followed by DeepSeek-V3 with a score of 16.38, suggesting that models trained with specialized reasoning tasks significantly outperform general-purpose models (even larger than reasoning-oriented models) in programming. Further analysis also reveals key areas for programming capability enhancement, e.g., algorithm adaptability and reasoning sufficiency, providing important insights for the future development of reasoning models. 

---
# The Power of Personality: A Human Simulation Perspective to Investigate Large Language Model Agents 

**Authors**: Yifan Duan, Yihong Tang, Xuefeng Bai, Kehai Chen, Juntao Li, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20859)  

**Abstract**: Large language models (LLMs) excel in both closed tasks (including problem-solving, and code generation) and open tasks (including creative writing), yet existing explanations for their capabilities lack connections to real-world human intelligence. To fill this gap, this paper systematically investigates LLM intelligence through the lens of ``human simulation'', addressing three core questions: (1) How do personality traits affect problem-solving in closed tasks? (2) How do traits shape creativity in open tasks? (3) How does single-agent performance influence multi-agent collaboration? By assigning Big Five personality traits to LLM agents and evaluating their performance in single- and multi-agent settings, we reveal that specific traits significantly influence reasoning accuracy (closed tasks) and creative output (open tasks). Furthermore, multi-agent systems exhibit collective intelligence distinct from individual capabilities, driven by distinguishing combinations of personalities. We demonstrate that LLMs inherently simulate human behavior through next-token prediction, mirroring human language, decision-making, and collaborative dynamics. 

---
# Plan2Align: Predictive Planning Based Test-Time Preference Alignment in Paragraph-Level Machine Translation 

**Authors**: Kuang-Da Wang, Teng-Ruei Chen, Yu Heng Hung, Shuoyang Ding, Yueh-Hua Wu, Yu-Chiang Frank Wang, Chao-Han Huck Yang, Wen-Chih Peng, Ping-Chun Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2502.20795)  

**Abstract**: Machine Translation (MT) has been predominantly designed for sentence-level translation using transformer-based architectures. While next-token prediction based Large Language Models (LLMs) demonstrate strong capabilities in long-text translation, non-extensive language models often suffer from omissions and semantic inconsistencies when processing paragraphs. Existing preference alignment methods improve sentence-level translation but fail to ensure coherence over extended contexts due to the myopic nature of next-token generation. We introduce Plan2Align, a test-time alignment framework that treats translation as a predictive planning problem, adapting Model Predictive Control to iteratively refine translation outputs. Experiments on WMT24 Discourse-Level Literary Translation show that Plan2Align significantly improves paragraph-level translation, achieving performance surpassing or on par with the existing training-time and test-time alignment methods on LLaMA-3.1 8B. 

---
# Learning to Substitute Components for Compositional Generalization 

**Authors**: Zhaoyi Li, Gangwei Jiang, Chenwang Wu, Ying Wei, Defu Lian, Enhong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20834)  

**Abstract**: Despite the rising prevalence of neural language models, recent empirical evidence suggests their deficiency in compositional generalization. One of the current de-facto solutions to this problem is compositional data augmentation, which aims to introduce additional compositional inductive bias. However, existing handcrafted augmentation strategies offer limited improvement when systematic generalization of neural language models requires multi-grained compositional bias (i.e., not limited to either lexical or structural biases alone) or when training sentences have an imbalanced difficulty distribution. To address these challenges, we first propose a novel compositional augmentation strategy called Component Substitution (CompSub), which enables multi-grained composition of substantial substructures across the entire training set. Furthermore, we introduce the Learning Component Substitution (LCS) framework. This framework empowers the learning of component substitution probabilities in CompSub in an end-to-end manner by maximizing the loss of neural language models, thereby prioritizing challenging compositions with elusive concepts and novel contexts. We extend the key ideas of CompSub and LCS to the recently emerging in-context learning scenarios of pre-trained large language models (LLMs), proposing the LCS-ICL algorithm to enhance the few-shot compositional generalization of state-of-the-art (SOTA) LLMs. Theoretically, we provide insights into why applying our algorithms to language models can improve compositional generalization performance. Empirically, our results on four standard compositional generalization benchmarks(SCAN, COGS, GeoQuery, and COGS-QL) demonstrate the superiority of CompSub, LCS, and LCS-ICL, with improvements of up to 66.5%, 10.3%, 1.4%, and 8.8%, respectively. 

---
# The Rise of Darkness: Safety-Utility Trade-Offs in Role-Playing Dialogue Agents 

**Authors**: Yihong Tang, Kehai Chen, Xuefeng Bai, Zhengyu Niu, Bo Wang, Jie Liu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20757)  

**Abstract**: Large Language Models (LLMs) have made remarkable advances in role-playing dialogue agents, demonstrating their utility in character simulations. However, it remains challenging for these agents to balance character portrayal utility with content safety because this essential character simulation often comes with the risk of generating unsafe content. To address this issue, we first conduct a systematic exploration of the safety-utility trade-off across multiple LLMs. Our analysis reveals that risk scenarios created by villain characters and user queries (referred to as risk coupling) contribute to this trade-off. Building on this, we propose a novel Adaptive Dynamic Multi-Preference (ADMP) method, which dynamically adjusts safety-utility preferences based on the degree of risk coupling and guides the model to generate responses biased toward utility or safety. We further introduce Coupling Margin Sampling (CMS) into coupling detection to enhance the model's ability to handle high-risk scenarios. Experimental results demonstrate that our approach improves safety metrics while maintaining utility. 

---
# Rectifying Belief Space via Unlearning to Harness LLMs' Reasoning 

**Authors**: Ayana Niwa, Masahiro Kaneko, Kentaro Inui  

**Link**: [PDF](https://arxiv.org/pdf/2502.20620)  

**Abstract**: Large language models (LLMs) can exhibit advanced reasoning yet still generate incorrect answers. We hypothesize that such errors frequently stem from spurious beliefs, propositions the model internally considers true but are incorrect. To address this, we propose a method to rectify the belief space by suppressing these spurious beliefs while simultaneously enhancing true ones, thereby enabling more reliable inferences. Our approach first identifies the beliefs that lead to incorrect or correct answers by prompting the model to generate textual explanations, using our Forward-Backward Beam Search (FBBS). We then apply unlearning to suppress the identified spurious beliefs and enhance the true ones, effectively rectifying the model's belief space. Empirical results on multiple QA datasets and LLMs show that our method corrects previously misanswered questions without harming overall model performance. Furthermore, our approach yields improved generalization on unseen data, suggesting that rectifying a model's belief space is a promising direction for mitigating errors and enhancing overall reliability. 

---
# CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation 

**Authors**: Zhenyi Shen, Hanqi Yan, Linhai Zhang, Zhanghao Hu, Yali Du, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2502.21074)  

**Abstract**: Chain-of-Thought (CoT) enhances Large Language Models (LLMs) by enabling step-by-step reasoning in natural language. However, the language space may be suboptimal for reasoning. While implicit CoT methods attempt to enable reasoning without explicit CoT tokens, they have consistently lagged behind explicit CoT method in task performance. We propose CODI (Continuous Chain-of-Thought via Self-Distillation), a novel framework that distills CoT into a continuous space, where a shared model acts as both teacher and student, jointly learning explicit and implicit CoT while aligning their hidden activation on the token generating the final answer. CODI is the first implicit CoT method to match explicit CoT's performance on GSM8k while achieving 3.1x compression, surpassing the previous state-of-the-art by 28.2% in accuracy. Furthermore, CODI demonstrates scalability, robustness, and generalizability to more complex CoT datasets. Additionally, CODI retains interpretability by decoding its continuous thoughts, making its reasoning process transparent. Our findings establish implicit CoT as not only a more efficient but a powerful alternative to explicit CoT. 

---
# Supervised Fine-Tuning LLMs to Behave as Pedagogical Agents in Programming Education 

**Authors**: Emily Ross, Yuval Kansal, Jake Renzella, Alexandra Vassar, Andrew Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2502.20527)  

**Abstract**: Large language models (LLMs) are increasingly being explored in higher education, yet their effectiveness as teaching agents remains underexamined. In this paper, we present the development of GuideLM, a fine-tuned LLM designed for programming education. GuideLM has been integrated into the Debugging C Compiler (DCC), an educational C compiler that leverages LLMs to generate pedagogically sound error explanations. Previously, DCC relied on off-the-shelf OpenAI models, which, while accurate, often over-assisted students by directly providing solutions despite contrary prompting.
To address this, we employed supervised fine-tuning (SFT) on a dataset of 528 student-question/teacher-answer pairs, creating two models: GuideLM and GuideLM-mini, fine-tuned on ChatGPT-4o and 4o-mini, respectively. We conducted an expert analysis of 400 responses per model, comparing their pedagogical effectiveness against base OpenAI models. Our evaluation, grounded in constructivism and cognitive load theory, assessed factors such as conceptual scaffolding, clarity, and Socratic guidance.
Results indicate that GuideLM and GuideLM-mini improve pedagogical performance, with an 8% increase in Socratic guidance and a 58% improvement in economy of words compared to GPT-4o. However, this refinement comes at the cost of a slight reduction in general accuracy. While further work is needed, our findings suggest that fine-tuning LLMs with targeted datasets is a promising approach for developing models better suited to educational contexts. 

---
# NANOGPT: A Query-Driven Large Language Model Retrieval-Augmented Generation System for Nanotechnology Research 

**Authors**: Achuth Chandrasekhar, Omid Barati Farimani, Olabode T. Ajenifujah, Janghoon Ock, Amir Barati Farimani  

**Link**: [PDF](https://arxiv.org/pdf/2502.20541)  

**Abstract**: This paper presents the development and application of a Large Language Model Retrieval-Augmented Generation (LLM-RAG) system tailored for nanotechnology research. The system leverages the capabilities of a sophisticated language model to serve as an intelligent research assistant, enhancing the efficiency and comprehensiveness of literature reviews in the nanotechnology domain. Central to this LLM-RAG system is its advanced query backend retrieval mechanism, which integrates data from multiple reputable sources. The system retrieves relevant literature by utilizing Google Scholar's advanced search, and scraping open-access papers from Elsevier, Springer Nature, and ACS Publications. This multifaceted approach ensures a broad and diverse collection of up-to-date scholarly articles and papers. The proposed system demonstrates significant potential in aiding researchers by providing a streamlined, accurate, and exhaustive literature retrieval process, thereby accelerating research advancements in nanotechnology. The effectiveness of the LLM-RAG system is validated through rigorous testing, illustrating its capability to significantly reduce the time and effort required for comprehensive literature reviews, while maintaining high accuracy, query relevance and outperforming standard, publicly available LLMS. 

---
# Few-Shot, No Problem: Descriptive Continual Relation Extraction 

**Authors**: Nguyen Xuan Thanh, Anh Duc Le, Quyen Tran, Thanh-Thien Le, Linh Ngo Van, Thien Huu Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2502.20596)  

**Abstract**: Few-shot Continual Relation Extraction is a crucial challenge for enabling AI systems to identify and adapt to evolving relationships in dynamic real-world domains. Traditional memory-based approaches often overfit to limited samples, failing to reinforce old knowledge, with the scarcity of data in few-shot scenarios further exacerbating these issues by hindering effective data augmentation in the latent space. In this paper, we propose a novel retrieval-based solution, starting with a large language model to generate descriptions for each relation. From these descriptions, we introduce a bi-encoder retrieval training paradigm to enrich both sample and class representation learning. Leveraging these enhanced representations, we design a retrieval-based prediction method where each sample "retrieves" the best fitting relation via a reciprocal rank fusion score that integrates both relation description vectors and class prototypes. Extensive experiments on multiple datasets demonstrate that our method significantly advances the state-of-the-art by maintaining robust performance across sequential tasks, effectively addressing catastrophic forgetting. 

---
# Protecting multimodal large language models against misleading visualizations 

**Authors**: Jonathan Tonglet, Tinne Tuytelaars, Marie-Francine Moens, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2502.20503)  

**Abstract**: We assess the vulnerability of multimodal large language models to misleading visualizations - charts that distort the underlying data using techniques such as truncated or inverted axes, leading readers to draw inaccurate conclusions that may support misinformation or conspiracy theories. Our analysis shows that these distortions severely harm multimodal large language models, reducing their question-answering accuracy to the level of the random baseline. To mitigate this vulnerability, we introduce six inference-time methods to improve performance of MLLMs on misleading visualizations while preserving their accuracy on non-misleading ones. The most effective approach involves (1) extracting the underlying data table and (2) using a text-only large language model to answer questions based on the table. This method improves performance on misleading visualizations by 15.4 to 19.6 percentage points. 

---
# Chain-of-Thought Matters: Improving Long-Context Language Models with Reasoning Path Supervision 

**Authors**: Dawei Zhu, Xiyu Wei, Guangxiang Zhao, Wenhao Wu, Haosheng Zou, Junfeng Ran, Xun Wang, Lin Sun, Xiangzheng Zhang, Sujian Li  

**Link**: [PDF](https://arxiv.org/pdf/2502.20790)  

**Abstract**: Recent advances in Large Language Models (LLMs) have highlighted the challenge of handling long-context tasks, where models need to reason over extensive input contexts to aggregate target information. While Chain-of-Thought (CoT) prompting has shown promise for multi-step reasoning, its effectiveness for long-context scenarios remains underexplored. Through systematic investigation across diverse tasks, we demonstrate that CoT's benefits generalize across most long-context scenarios and amplify with increasing context length. Motivated by this critical observation, we propose LongRePS, a process-supervised framework that teaches models to generate high-quality reasoning paths for enhanced long-context performance. Our framework incorporates a self-sampling mechanism to bootstrap reasoning paths and a novel quality assessment protocol specifically designed for long-context scenarios. Experimental results on various long-context benchmarks demonstrate the effectiveness of our approach, achieving significant improvements over outcome supervision baselines on both in-domain tasks (+13.6/+3.8 points for LLaMA/Qwen on MuSiQue) and cross-domain generalization (+9.3/+8.1 points on average across diverse QA tasks). Our code, data and trained models are made public to facilitate future research. 

---
# FlexPrefill: A Context-Aware Sparse Attention Mechanism for Efficient Long-Sequence Inference 

**Authors**: Xunhao Lai, Jianqiao Lu, Yao Luo, Yiyuan Ma, Xun Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2502.20766)  

**Abstract**: Large language models (LLMs) encounter computational challenges during long-sequence inference, especially in the attention pre-filling phase, where the complexity grows quadratically with the prompt length. Previous efforts to mitigate these challenges have relied on fixed sparse attention patterns or identifying sparse attention patterns based on limited cases. However, these methods lacked the flexibility to efficiently adapt to varying input demands. In this paper, we introduce FlexPrefill, a Flexible sparse Pre-filling mechanism that dynamically adjusts sparse attention patterns and computational budget in real-time to meet the specific requirements of each input and attention head. The flexibility of our method is demonstrated through two key innovations: 1) Query-Aware Sparse Pattern Determination: By measuring Jensen-Shannon divergence, this component adaptively switches between query-specific diverse attention patterns and predefined attention patterns. 2) Cumulative-Attention Based Index Selection: This component dynamically selects query-key indexes to be computed based on different attention patterns, ensuring the sum of attention scores meets a predefined threshold. FlexPrefill adaptively optimizes the sparse pattern and sparse ratio of each attention head based on the prompt, enhancing efficiency in long-sequence inference tasks. Experimental results show significant improvements in both speed and accuracy over prior methods, providing a more flexible and efficient solution for LLM inference. 

---
# ECCOS: Efficient Capability and Cost Coordinated Scheduling for Multi-LLM Serving 

**Authors**: Kai Mei, Wujiang Xu, Shuhang Lin, Yongfeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2502.20576)  

**Abstract**: As large language models (LLMs) are increasingly deployed as service endpoints in systems, the surge in query volume creates significant scheduling challenges. Existing scheduling frameworks mainly target at latency optimization while neglecting the capability of LLMs to serve different level of queries, which could lead to computational resource waste. This paper addresses this challenge by proposing a capability-cost coordinated scheduling framework, ECCOS, for multi-LLM serving, which explicitly constrains response quality and workload to optimize LLM inference cost. Specifically, it introduces the two-stage scheduling by designing a multi-objective predictor and a constrained optimizer. The predictor estimates both model capabilities and computational costs through training-based and retrieval-based approaches, while the optimizer determines cost-optimal assignments under quality and workload constraints. It also introduces QAServe, a dataset collected for sample-wise response quality and costs by zero-shot prompting different LLMs on knowledge QA and mathematical reasoning. Extensive experiments demonstrate that ECCOS improves success rates by 6.30% while reducing costs by 10.15% compared to existing methods, consuming less than 0.5% of LLM response time. The code is available at: this https URL. 

---
# Visual Reasoning at Urban Intersections: FineTuning GPT-4o for Traffic Conflict Detection 

**Authors**: Sari Masri, Huthaifa I. Ashqar, Mohammed Elhenawy  

**Link**: [PDF](https://arxiv.org/pdf/2502.20573)  

**Abstract**: Traffic control in unsignalized urban intersections presents significant challenges due to the complexity, frequent conflicts, and blind spots. This study explores the capability of leveraging Multimodal Large Language Models (MLLMs), such as GPT-4o, to provide logical and visual reasoning by directly using birds-eye-view videos of four-legged intersections. In this proposed method, GPT-4o acts as intelligent system to detect conflicts and provide explanations and recommendations for the drivers. The fine-tuned model achieved an accuracy of 77.14%, while the manual evaluation of the true predicted values of the fine-tuned GPT-4o showed significant achievements of 89.9% accuracy for model-generated explanations and 92.3% for the recommended next actions. These results highlight the feasibility of using MLLMs for real-time traffic management using videos as inputs, offering scalable and actionable insights into intersections traffic management and operation. Code used in this study is available at this https URL. 

---
# Brain-Inspired Exploration of Functional Networks and Key Neurons in Large Language Models 

**Authors**: Yiheng Liu, Xiaohui Gao, Haiyang Sun, Bao Ge, Tianming Liu, Junwei Han, Xintao Hu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20408)  

**Abstract**: In recent years, the rapid advancement of large language models (LLMs) in natural language processing has sparked significant interest among researchers to understand their mechanisms and functional characteristics. Although existing studies have attempted to explain LLM functionalities by identifying and interpreting specific neurons, these efforts mostly focus on individual neuron contributions, neglecting the fact that human brain functions are realized through intricate interaction networks. Inspired by cognitive neuroscience research on functional brain networks (FBNs), this study introduces a novel approach to investigate whether similar functional networks exist within LLMs. We use methods similar to those in the field of functional neuroimaging analysis to locate and identify functional networks in LLM. Experimental results show that, similar to the human brain, LLMs contain functional networks that frequently recur during operation. Further analysis shows that these functional networks are crucial for LLM performance. Masking key functional networks significantly impairs the model's performance, while retaining just a subset of these networks is adequate to maintain effective operation. This research provides novel insights into the interpretation of LLMs and the lightweighting of LLMs for certain downstream tasks. Code is available at this https URL. 

---
# CoTMR: Chain-of-Thought Multi-Scale Reasoning for Training-Free Zero-Shot Composed Image Retrieval 

**Authors**: Zelong Sun, Dong Jing, Zhiwu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2502.20826)  

**Abstract**: Zero-Shot Composed Image Retrieval (ZS-CIR) aims to retrieve target images by integrating information from a composed query (reference image and modification text) without training samples. Existing methods primarily combine caption models and large language models (LLMs) to generate target captions based on composed queries but face various issues such as incompatibility, visual information loss, and insufficient reasoning. In this work, we propose CoTMR, a training-free framework crafted for ZS-CIR with novel Chain-of-thought (CoT) and Multi-scale Reasoning. Instead of relying on caption models for modality transformation, CoTMR employs the Large Vision-Language Model (LVLM) to achieve unified understanding and reasoning for composed queries. To enhance the reasoning reliability, we devise CIRCoT, which guides the LVLM through a step-by-step inference process using predefined subtasks. Considering that existing approaches focus solely on global-level reasoning, our CoTMR incorporates multi-scale reasoning to achieve more comprehensive inference via fine-grained predictions about the presence or absence of key elements at the object scale. Further, we design a Multi-Grained Scoring (MGS) mechanism, which integrates CLIP similarity scores of the above reasoning outputs with candidate images to realize precise retrieval. Extensive experiments demonstrate that our CoTMR not only drastically outperforms previous methods across four prominent benchmarks but also offers appealing interpretability. 

---
