# Multi-agents based User Values Mining for Recommendation 

**Authors**: Lijian Chen, Wei Yuan, Tong Chen, Xiangyu Zhao, Nguyen Quoc Viet Hung, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2505.00981)  

**Abstract**: Recommender systems have rapidly evolved and become integral to many online services. However, existing systems sometimes produce unstable and unsatisfactory recommendations that fail to align with users' fundamental and long-term preferences. This is because they primarily focus on extracting shallow and short-term interests from user behavior data, which is inherently dynamic and challenging to model. Unlike these transient interests, user values are more stable and play a crucial role in shaping user behaviors, such as purchasing items and consuming content. Incorporating user values into recommender systems can help stabilize recommendation performance and ensure results better reflect users' latent preferences. However, acquiring user values is typically difficult and costly. To address this challenge, we leverage the strong language understanding, zero-shot inference, and generalization capabilities of Large Language Models (LLMs) to extract user values from users' historical interactions. Unfortunately, direct extraction using LLMs presents several challenges such as length constraints and hallucination. To overcome these issues, we propose ZOOM, a zero-shot multi-LLM collaborative framework for effective and accurate user value extraction. In ZOOM, we apply text summarization techniques to condense item content while preserving essential meaning. To mitigate hallucinations, ZOOM introduces two specialized agent roles: evaluators and supervisors, to collaboratively generate accurate user values. Extensive experiments on two widely used recommendation datasets with two state-of-the-art recommendation models demonstrate the effectiveness and generalization of our framework in automatic user value mining and recommendation performance improvement. 

---
# Towards Explainable Temporal User Profiling with LLMs 

**Authors**: Milad Sabouri, Masoud Mansoury, Kun Lin, Bamshad Mobasher  

**Link**: [PDF](https://arxiv.org/pdf/2505.00886)  

**Abstract**: Accurately modeling user preferences is vital not only for improving recommendation performance but also for enhancing transparency in recommender systems. Conventional user profiling methods, such as averaging item embeddings, often overlook the evolving, nuanced nature of user interests, particularly the interplay between short-term and long-term preferences. In this work, we leverage large language models (LLMs) to generate natural language summaries of users' interaction histories, distinguishing recent behaviors from more persistent tendencies. Our framework not only models temporal user preferences but also produces natural language profiles that can be used to explain recommendations in an interpretable manner. These textual profiles are encoded via a pre-trained model, and an attention mechanism dynamically fuses the short-term and long-term embeddings into a comprehensive user representation. Beyond boosting recommendation accuracy over multiple baselines, our approach naturally supports explainability: the interpretable text summaries and attention weights can be exposed to end users, offering insights into why specific items are suggested. Experiments on real-world datasets underscore both the performance gains and the promise of generating clearer, more transparent justifications for content-based recommendations. 

---
# On the Limitations of Steering in Language Model Alignment 

**Authors**: Chebrolu Niranjan, Kokil Jaidka, Gerard Christopher Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2505.01162)  

**Abstract**: Steering vectors are a promising approach to aligning language model behavior at inference time. In this paper, we propose a framework to assess the limitations of steering vectors as alignment mechanisms. Using a framework of transformer hook interventions and antonym-based function vectors, we evaluate the role of prompt structure and context complexity in steering effectiveness. Our findings indicate that steering vectors are promising for specific alignment tasks, such as value alignment, but may not provide a robust foundation for general-purpose alignment in LLMs, particularly in complex scenarios. We establish a methodological foundation for future investigations into steering capabilities of reasoning models. 

---
# Helping Big Language Models Protect Themselves: An Enhanced Filtering and Summarization System 

**Authors**: Sheikh Samit Muhaimin, Spyridon Mastorakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.01315)  

**Abstract**: The recent growth in the use of Large Language Models has made them vulnerable to sophisticated adversarial assaults, manipulative prompts, and encoded malicious inputs. Existing countermeasures frequently necessitate retraining models, which is computationally costly and impracticable for deployment. Without the need for retraining or fine-tuning, this study presents a unique defense paradigm that allows LLMs to recognize, filter, and defend against adversarial or malicious inputs on their own. There are two main parts to the suggested framework: (1) A prompt filtering module that uses sophisticated Natural Language Processing (NLP) techniques, including zero-shot classification, keyword analysis, and encoded content detection (e.g. base64, hexadecimal, URL encoding), to detect, decode, and classify harmful inputs; and (2) A summarization module that processes and summarizes adversarial research literature to give the LLM context-aware defense knowledge. This approach strengthens LLMs' resistance to adversarial exploitation by fusing text extraction, summarization, and harmful prompt analysis. According to experimental results, this integrated technique has a 98.71% success rate in identifying harmful patterns, manipulative language structures, and encoded prompts. By employing a modest amount of adversarial research literature as context, the methodology also allows the model to react correctly to harmful inputs with a larger percentage of jailbreak resistance and refusal rate. While maintaining the quality of LLM responses, the framework dramatically increases LLM's resistance to hostile misuse, demonstrating its efficacy as a quick and easy substitute for time-consuming, retraining-based defenses. 

---
# MateICL: Mitigating Attention Dispersion in Large-Scale In-Context Learning 

**Authors**: Murtadha Ahmed, Wenbo, Liu yunfeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.01110)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in In-Context Learning (ICL). However, the fixed position length constraints in pre-trained models limit the number of demonstration examples. Recent efforts to extend context suffer from attention dispersion as the number of demonstrations increases. In this paper, we introduce Mitigating Attention Dispersion in large-scale ICL (MateICL) that enables LLMs to maintain effective self-attention as the context size grows. We first split the context into multiple windows, each filled to the model's context capacity, which are processed separately. Then, we introduce an additional layer to recalibrate the attention weights, prioritizing the query tokens as the number of demonstrations increases. Our empirical results show that MateICL can effectively leverage larger contexts to improve ICL performance. Compared to retrieval-based baselines, MateICL consistently achieves better performance without requiring an externally trained retrieval model. Despite recent advances in inference strategies (e.g., 32k token contexts), our results demonstrate that MateICL remains beneficial in computationally resource-constrained settings. The code is publicly available at this https URL. 

---
# Do We Need a Detailed Rubric for Automated Essay Scoring using Large Language Models? 

**Authors**: Lui Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2505.01035)  

**Abstract**: This study investigates the necessity and impact of a detailed rubric in automated essay scoring (AES) using large language models (LLMs). While using rubrics are standard in LLM-based AES, creating detailed rubrics requires substantial ef-fort and increases token usage. We examined how different levels of rubric detail affect scoring accuracy across multiple LLMs using the TOEFL11 dataset. Our experiments compared three conditions: a full rubric, a simplified rubric, and no rubric, using four different LLMs (Claude 3.5 Haiku, Gemini 1.5 Flash, GPT-4o-mini, and Llama 3 70B Instruct). Results showed that three out of four models maintained similar scoring accuracy with the simplified rubric compared to the detailed one, while significantly reducing token usage. However, one model (Gemini 1.5 Flash) showed decreased performance with more detailed rubrics. The findings suggest that simplified rubrics may be sufficient for most LLM-based AES applications, offering a more efficient alternative without compromis-ing scoring accuracy. However, model-specific evaluation remains crucial as per-formance patterns vary across different LLMs. 

---
# Value Portrait: Understanding Values of LLMs with Human-aligned Benchmark 

**Authors**: Jongwook Han, Dongmin Choi, Woojung Song, Eun-Ju Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2505.01015)  

**Abstract**: The importance of benchmarks for assessing the values of language models has been pronounced due to the growing need of more authentic, human-aligned responses. However, existing benchmarks rely on human or machine annotations that are vulnerable to value-related biases. Furthermore, the tested scenarios often diverge from real-world contexts in which models are commonly used to generate text and express values. To address these issues, we propose the Value Portrait benchmark, a reliable framework for evaluating LLMs' value orientations with two key characteristics. First, the benchmark consists of items that capture real-life user-LLM interactions, enhancing the relevance of assessment results to real-world LLM usage and thus ecological validity. Second, each item is rated by human subjects based on its similarity to their own thoughts, and correlations between these ratings and the subjects' actual value scores are derived. This psychometrically validated approach ensures that items strongly correlated with specific values serve as reliable items for assessing those values. Through evaluating 27 LLMs with our benchmark, we find that these models prioritize Benevolence, Security, and Self-Direction values while placing less emphasis on Tradition, Power, and Achievement values. Also, our analysis reveals biases in how LLMs perceive various demographic groups, deviating from real human data. 

---
# Position: Enough of Scaling LLMs! Lets Focus on Downscaling 

**Authors**: Ayan Sengupta, Yash Goel, Tanmoy Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2505.00985)  

**Abstract**: We challenge the dominant focus on neural scaling laws and advocate for a paradigm shift toward downscaling in the development of large language models (LLMs). While scaling laws have provided critical insights into performance improvements through increasing model and dataset size, we emphasize the significant limitations of this approach, particularly in terms of computational inefficiency, environmental impact, and deployment constraints. To address these challenges, we propose a holistic framework for downscaling LLMs that seeks to maintain performance while drastically reducing resource demands. This paper outlines practical strategies for transitioning away from traditional scaling paradigms, advocating for a more sustainable, efficient, and accessible approach to LLM development. 

---
# Llama-Nemotron: Efficient Reasoning Models 

**Authors**: Akhiad Bercovich, Itay Levy, Izik Golan, Mohammad Dabbah, Ran El-Yaniv, Omri Puny, Ido Galil, Zach Moshe, Tomer Ronen, Najeeb Nabwani, Ido Shahaf, Oren Tropp, Ehud Karpas, Ran Zilberstein, Jiaqi Zeng, Soumye Singhal, Alexander Bukharin, Yian Zhang, Tugrul Konuk, Gerald Shen, Ameya Sunil Mahabaleshwarkar, Bilal Kartal, Yoshi Suhara, Olivier Delalleau, Zijia Chen, Zhilin Wang, David Mosallanezhad, Adi Renduchintala, Haifeng Qian, Dima Rekesh, Fei Jia, Somshubra Majumdar, Vahid Noroozi, Wasi Uddin Ahmad, Sean Narenthiran, Aleksander Ficek, Mehrzad Samadi, Jocelyn Huang, Siddhartha Jain, Igor Gitman, Ivan Moshkov, Wei Du, Shubham Toshniwal, George Armstrong, Branislav Kisacanin, Matvei Novikov, Daria Gitman, Evelina Bakhturina, Jane Polak Scowcroft, John Kamalu, Dan Su, Kezhi Kong, Markus Kliegl, Rabeeh Karimi, Ying Lin, Sanjeev Satheesh, Jupinder Parmar, Pritam Gundecha, Brandon Norick, Joseph Jennings, Shrimai Prabhumoye, Syeda Nahida Akter, Mostofa Patwary, Abhinav Khattar, Deepak Narayanan, Roger Waleffe, Jimmy Zhang, Bor-Yiing Su, Guyue Huang, Terry Kong, Parth Chadha, Sahil Jain, Christine Harvey, Elad Segal, Jining Huang, Sergey Kashirsky, Robert McQueen, Izzy Putterman, George Lam, Arun Venkatesan, Sherry Wu, Vinh Nguyen, Manoj Kilaru, Andrew Wang, Anna Warno, Abhilash Somasamudramath, Sandip Bhaskar, Maka Dong, Nave Assaf, Shahar Mor, Omer Ullman Argov, Scot Junkin, Oleksandr Romanenko, Pedro Larroy, Monika Katariya, Marco Rovinelli, Viji Balas, Nicholas Edelman, Anahita Bhiwandiwalla, Muthu Subramaniam  

**Link**: [PDF](https://arxiv.org/pdf/2505.00949)  

**Abstract**: We introduce the Llama-Nemotron series of models, an open family of heterogeneous reasoning models that deliver exceptional reasoning capabilities, inference efficiency, and an open license for enterprise use. The family comes in three sizes -- Nano (8B), Super (49B), and Ultra (253B) -- and performs competitively with state-of-the-art reasoning models such as DeepSeek-R1 while offering superior inference throughput and memory efficiency. In this report, we discuss the training procedure for these models, which entails using neural architecture search from Llama 3 models for accelerated inference, knowledge distillation, and continued pretraining, followed by a reasoning-focused post-training stage consisting of two main parts: supervised fine-tuning and large scale reinforcement learning. Llama-Nemotron models are the first open-source models to support a dynamic reasoning toggle, allowing users to switch between standard chat and reasoning modes during inference. To further support open research and facilitate model development, we provide the following resources: 1. We release the Llama-Nemotron reasoning models -- LN-Nano, LN-Super, and LN-Ultra -- under the commercially permissive NVIDIA Open Model License Agreement. 2. We release the complete post-training dataset: Llama-Nemotron-Post-Training-Dataset. 3. We also release our training codebases: NeMo, NeMo-Aligner, and Megatron-LM. 

---
# Synthesize-on-Graph: Knowledgeable Synthetic Data Generation for Continue Pre-training of Large Language Models 

**Authors**: Xuhui Jiang, Shengjie Ma, Chengjin Xu, Cehao Yang, Liyu Zhang, Jian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.00979)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable success but remain data-inefficient, especially when learning from small, specialized corpora with limited and proprietary data. Existing synthetic data generation methods for continue pre-training focus on intra-document content and overlook cross-document knowledge associations, limiting content diversity and depth. We propose Synthetic-on-Graph (SoG), a synthetic data generation framework that incorporates cross-document knowledge associations for efficient corpus expansion. SoG constructs a context graph by extracting entities and concepts from the original corpus, representing cross-document associations, and employing a graph walk strategy for knowledge-associated sampling. This enhances synthetic data diversity and coherence, enabling models to learn complex knowledge structures and handle rare knowledge. To further improve synthetic data quality, we integrate Chain-of-Thought (CoT) and Contrastive Clarifying (CC) synthetic, enhancing reasoning processes and discriminative power. Experiments show that SoG outperforms the state-of-the-art (SOTA) method in a multi-hop document Q&A dataset while performing comparably to the SOTA method on the reading comprehension task datasets, which also underscores the better generalization capability of SoG. Our work advances synthetic data generation and provides practical solutions for efficient knowledge acquisition in LLMs, especially in domains with limited data availability. 

---
# Large Language Model-Driven Dynamic Assessment of Grammatical Accuracy in English Language Learner Writing 

**Authors**: Timur Jaganov, John Blake, Julián Villegas, Nicholas Carr  

**Link**: [PDF](https://arxiv.org/pdf/2505.00931)  

**Abstract**: This study investigates the potential for Large Language Models (LLMs) to scale-up Dynamic Assessment (DA). To facilitate such an investigation, we first developed DynaWrite-a modular, microservices-based grammatical tutoring application which supports multiple LLMs to generate dynamic feedback to learners of English. Initial testing of 21 LLMs, revealed GPT-4o and neural chat to have the most potential to scale-up DA in the language learning classroom. Further testing of these two candidates found both models performed similarly in their ability to accurately identify grammatical errors in user sentences. However, GPT-4o consistently outperformed neural chat in the quality of its DA by generating clear, consistent, and progressively explicit hints. Real-time responsiveness and system stability were also confirmed through detailed performance testing, with GPT-4o exhibiting sufficient speed and stability. This study shows that LLMs can be used to scale-up dynamic assessment and thus enable dynamic assessment to be delivered to larger groups than possible in traditional teacher-learner settings. 

---
# Reasoning Capabilities and Invariability of Large Language Models 

**Authors**: Alessandro Raganato, Rafael Peñaloza, Marco Viviani, Gabriella Pasi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00776)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in manipulating natural language across multiple applications, but their ability to handle simple reasoning tasks is often questioned. In this work, we aim to provide a comprehensive analysis of LLMs' reasoning competence, specifically focusing on their prompt dependency. In particular, we introduce a new benchmark dataset with a series of simple reasoning questions demanding shallow logical reasoning. Aligned with cognitive psychology standards, the questions are confined to a basic domain revolving around geometric figures, ensuring that responses are independent of any pre-existing intuition about the world and rely solely on deduction. An empirical analysis involving zero-shot and few-shot prompting across 24 LLMs of different sizes reveals that, while LLMs with over 70 billion parameters perform better in the zero-shot setting, there is still a large room for improvement. An additional test with chain-of-thought prompting over 22 LLMs shows that this additional prompt can aid or damage the performance of models, depending on whether the rationale is required before or after the answer. 

---
# A Survey on Large Language Model based Human-Agent Systems 

**Authors**: Henry Peng Zou, Wei-Chieh Huang, Yaozu Wu, Yankai Chen, Chunyu Miao, Hoang Nguyen, Yue Zhou, Weizhi Zhang, Liancheng Fang, Langzhou He, Yangning Li, Yuwei Cao, Dongyuan Li, Renhe Jiang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00753)  

**Abstract**: Recent advances in large language models (LLMs) have sparked growing interest in building fully autonomous agents. However, fully autonomous LLM-based agents still face significant challenges, including limited reliability due to hallucinations, difficulty in handling complex tasks, and substantial safety and ethical risks, all of which limit their feasibility and trustworthiness in real-world applications. To overcome these limitations, LLM-based human-agent systems (LLM-HAS) incorporate human-provided information, feedback, or control into the agent system to enhance system performance, reliability and safety. This paper provides the first comprehensive and structured survey of LLM-HAS. It clarifies fundamental concepts, systematically presents core components shaping these systems, including environment & profiling, human feedback, interaction types, orchestration and communication, explores emerging applications, and discusses unique challenges and opportunities. By consolidating current knowledge and offering a structured overview, we aim to foster further research and innovation in this rapidly evolving interdisciplinary field. Paper lists and resources are available at this https URL. 

---
# Anti-adversarial Learning: Desensitizing Prompts for Large Language Models 

**Authors**: Xuan Li, Zhe Yin, Xiaodong Gu, Beijun Shen  

**Link**: [PDF](https://arxiv.org/pdf/2505.01273)  

**Abstract**: With the widespread use of LLMs, preserving privacy in user prompts has become crucial, as prompts risk exposing privacy and sensitive data to the cloud LLMs. Traditional techniques like homomorphic encryption, secure multi-party computation, and federated learning face challenges due to heavy computational costs and user participation requirements, limiting their applicability in LLM scenarios. In this paper, we propose PromptObfus, a novel method for desensitizing LLM prompts. The core idea of PromptObfus is "anti-adversarial" learning, which perturbs privacy words in the prompt to obscure sensitive information while retaining the stability of model predictions. Specifically, PromptObfus frames prompt desensitization as a masked language modeling task, replacing privacy-sensitive terms with a [MASK] token. A desensitization model is trained to generate candidate replacements for each masked position. These candidates are subsequently selected based on gradient feedback from a surrogate model, ensuring minimal disruption to the task output. We demonstrate the effectiveness of our approach on three NLP tasks. Results show that PromptObfus effectively prevents privacy inference from remote LLMs while preserving task performance. 

---
# Attack and defense techniques in large language models: A survey and new perspectives 

**Authors**: Zhiyu Liao, Kang Chen, Yuanguo Lin, Kangkang Li, Yunxuan Liu, Hefeng Chen, Xingwang Huang, Yuanhui Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00976)  

**Abstract**: Large Language Models (LLMs) have become central to numerous natural language processing tasks, but their vulnerabilities present significant security and ethical challenges. This systematic survey explores the evolving landscape of attack and defense techniques in LLMs. We classify attacks into adversarial prompt attack, optimized attacks, model theft, as well as attacks on application of LLMs, detailing their mechanisms and implications. Consequently, we analyze defense strategies, including prevention-based and detection-based defense methods. Although advances have been made, challenges remain to adapt to the dynamic threat landscape, balance usability with robustness, and address resource constraints in defense implementation. We highlight open problems, including the need for adaptive scalable defenses, explainable security techniques, and standardized evaluation frameworks. This survey provides actionable insights and directions for developing secure and resilient LLMs, emphasizing the importance of interdisciplinary collaboration and ethical considerations to mitigate risks in real-world applications. 

---
# SmallPlan: Leverage Small Language Models for Sequential Path Planning with Simulation-Powered, LLM-Guided Distillation 

**Authors**: Quang P. M. Pham, Khoi T. N. Nguyen, Nhi H. Doan, Cuong A. Pham, Kentaro Inui, Dezhen Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.00831)  

**Abstract**: Efficient path planning in robotics, particularly within large-scale, dynamic environments, remains a significant hurdle. While Large Language Models (LLMs) offer strong reasoning capabilities, their high computational cost and limited adaptability in dynamic scenarios hinder real-time deployment on edge devices. We present SmallPlan -- a novel framework leveraging LLMs as teacher models to train lightweight Small Language Models (SLMs) for high-level path planning tasks. In SmallPlan, the SLMs provide optimal action sequences to navigate across scene graphs that compactly represent full-scaled 3D scenes. The SLMs are trained in a simulation-powered, interleaved manner with LLM-guided supervised fine-tuning (SFT) and reinforcement learning (RL). This strategy not only enables SLMs to successfully complete navigation tasks but also makes them aware of important factors like travel distance and number of trials. Through experiments, we demonstrate that the fine-tuned SLMs perform competitively with larger models like GPT-4o on sequential path planning, without suffering from hallucination and overfitting. SmallPlan is resource-efficient, making it well-suited for edge-device deployment and advancing practical autonomous robotics. 

---
# VTS-LLM: Domain-Adaptive LLM Agent for Enhancing Awareness in Vessel Traffic Services through Natural Language 

**Authors**: Sijin Sun, Liangbin Zhao, Ming Deng, Xiuju Fu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00989)  

**Abstract**: Vessel Traffic Services (VTS) are essential for maritime safety and regulatory compliance through real-time traffic management. However, with increasing traffic complexity and the prevalence of heterogeneous, multimodal data, existing VTS systems face limitations in spatiotemporal reasoning and intuitive human interaction. In this work, we propose VTS-LLM Agent, the first domain-adaptive large LLM agent tailored for interactive decision support in VTS operations. We formalize risk-prone vessel identification as a knowledge-augmented Text-to-SQL task, combining structured vessel databases with external maritime knowledge. To support this, we construct a curated benchmark dataset consisting of a custom schema, domain-specific corpus, and a query-SQL test set in multiple linguistic styles. Our framework incorporates NER-based relational reasoning, agent-based domain knowledge injection, semantic algebra intermediate representation, and query rethink mechanisms to enhance domain grounding and context-aware understanding. Experimental results show that VTS-LLM outperforms both general-purpose and SQL-focused baselines under command-style, operational-style, and formal natural language queries, respectively. Moreover, our analysis provides the first empirical evidence that linguistic style variation introduces systematic performance challenges in Text-to-SQL modeling. This work lays the foundation for natural language interfaces in vessel traffic services and opens new opportunities for proactive, LLM-driven maritime real-time traffic management. 

---
# Multi-Modal Language Models as Text-to-Image Model Evaluators 

**Authors**: Jiahui Chen, Candace Ross, Reyhane Askari-Hemmat, Koustuv Sinha, Melissa Hall, Michal Drozdzal, Adriana Romero-Soriano  

**Link**: [PDF](https://arxiv.org/pdf/2505.00759)  

**Abstract**: The steady improvements of text-to-image (T2I) generative models lead to slow deprecation of automatic evaluation benchmarks that rely on static datasets, motivating researchers to seek alternative ways to evaluate the T2I progress. In this paper, we explore the potential of multi-modal large language models (MLLMs) as evaluator agents that interact with a T2I model, with the objective of assessing prompt-generation consistency and image aesthetics. We present Multimodal Text-to-Image Eval (MT2IE), an evaluation framework that iteratively generates prompts for evaluation, scores generated images and matches T2I evaluation of existing benchmarks with a fraction of the prompts used in existing static benchmarks. Moreover, we show that MT2IE's prompt-generation consistency scores have higher correlation with human judgment than scores previously introduced in the literature. MT2IE generates prompts that are efficient at probing T2I model performance, producing the same relative T2I model rankings as existing benchmarks while using only 1/80th the number of prompts for evaluation. 

---
# Retrieval Augmented Learning: A Retrial-based Large Language Model Self-Supervised Learning and Autonomous Knowledge Generation 

**Authors**: Zongyuan Li, Pengfei Li, Runnan Qi, Yanan Ni, Lumin Jiang, Hui Wu, Xuebo Zhang, Kuihua Huang, Xian Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.01073)  

**Abstract**: The lack of domain-specific data in the pre-training of Large Language Models (LLMs) severely limits LLM-based decision systems in specialized applications, while post-training a model in the scenarios requires significant computational resources. In this paper, we present Retrial-Augmented Learning (RAL), a reward-free self-supervised learning framework for LLMs that operates without model training. By developing Retrieval-Augmented Generation (RAG) into a module for organizing intermediate data, we realized a three-stage autonomous knowledge generation of proposing a hypothesis, validating the hypothesis, and generating the knowledge. The method is evaluated in the LLM-PySC2 environment, a representative decision-making platform that combines sufficient complexity with domain-specific knowledge requirements. Experiments demonstrate that the proposed method effectively reduces hallucination by generating and utilizing validated knowledge, and increases decision-making performance at an extremely low cost. Meanwhile, the approach exhibits potential in out-of-distribution(OOD) tasks, robustness, and transferability, making it a cost-friendly but effective solution for decision-making problems and autonomous knowledge generation. 

---
# Seeking to Collide: Online Safety-Critical Scenario Generation for Autonomous Driving with Retrieval Augmented Large Language Models 

**Authors**: Yuewen Mei, Tong Nie, Jian Sun, Ye Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.00972)  

**Abstract**: Simulation-based testing is crucial for validating autonomous vehicles (AVs), yet existing scenario generation methods either overfit to common driving patterns or operate in an offline, non-interactive manner that fails to expose rare, safety-critical corner cases. In this paper, we introduce an online, retrieval-augmented large language model (LLM) framework for generating safety-critical driving scenarios. Our method first employs an LLM-based behavior analyzer to infer the most dangerous intent of the background vehicle from the observed state, then queries additional LLM agents to synthesize feasible adversarial trajectories. To mitigate catastrophic forgetting and accelerate adaptation, we augment the framework with a dynamic memorization and retrieval bank of intent-planner pairs, automatically expanding its behavioral library when novel intents arise. Evaluations using the Waymo Open Motion Dataset demonstrate that our model reduces the mean minimum time-to-collision from 1.62 to 1.08 s and incurs a 75% collision rate, substantially outperforming baselines. 

---
# Improving Large Language Model Planning with Action Sequence Similarity 

**Authors**: Xinran Zhao, Hanie Sedghi, Bernd Bohnet, Dale Schuurmans, Azade Nova  

**Link**: [PDF](https://arxiv.org/pdf/2505.01009)  

**Abstract**: Planning is essential for artificial intelligence systems to look ahead and proactively determine a course of actions to reach objectives in the virtual and real world. Recent work on large language models (LLMs) sheds light on their planning capability in various tasks. However, it remains unclear what signals in the context influence the model performance. In this work, we explore how to improve the model planning capability through in-context learning (ICL), specifically, what signals can help select the exemplars. Through extensive experiments, we observe that commonly used problem similarity may result in false positives with drastically different plans, which can mislead the model. In response, we propose to sample and filter exemplars leveraging plan side action sequence similarity (AS). We propose GRASE-DC: a two-stage pipeline that first re-samples high AS exemplars and then curates the selected exemplars with dynamic clustering on AS to achieve a balance of relevance and diversity. Our experimental result confirms that GRASE-DC achieves significant performance improvement on various planning tasks (up to ~11-40 point absolute accuracy improvement with 27.3% fewer exemplars needed on average). With GRASE-DC* + VAL, where we iteratively apply GRASE-DC with a validator, we are able to even boost the performance by 18.9% more.
Extensive analysis validates the consistent performance improvement of GRASE-DC with various backbone LLMs and on both classical planning and natural language planning benchmarks. GRASE-DC can further boost the planning accuracy by ~24 absolute points on harder problems using simpler problems as exemplars over a random baseline. This demonstrates its ability to generalize to out-of-distribution problems. 

---
# Document Retrieval Augmented Fine-Tuning (DRAFT) for safety-critical software assessments 

**Authors**: Regan Bolton, Mohammadreza Sheikhfathollahi, Simon Parkinson, Vanessa Vulovic, Gary Bamford, Dan Basher, Howard Parkinson  

**Link**: [PDF](https://arxiv.org/pdf/2505.01307)  

**Abstract**: Safety critical software assessment requires robust assessment against complex regulatory frameworks, a process traditionally limited by manual evaluation. This paper presents Document Retrieval-Augmented Fine-Tuning (DRAFT), a novel approach that enhances the capabilities of a large language model (LLM) for safety-critical compliance assessment. DRAFT builds upon existing Retrieval-Augmented Generation (RAG) techniques by introducing a novel fine-tuning framework that accommodates our dual-retrieval architecture, which simultaneously accesses both software documentation and applicable reference standards. To fine-tune DRAFT, we develop a semi-automated dataset generation methodology that incorporates variable numbers of relevant documents with meaningful distractors, closely mirroring real-world assessment scenarios. Experiments with GPT-4o-mini demonstrate a 7% improvement in correctness over the baseline model, with qualitative improvements in evidence handling, response structure, and domain-specific reasoning. DRAFT represents a practical approach to improving compliance assessment systems while maintaining the transparency and evidence-based reasoning essential in regulatory domains. 

---
# Enhancing SPARQL Query Rewriting for Complex Ontology Alignments 

**Authors**: Anicet Lepetit Ondo, Laurence Capus, Mamadou Bousso  

**Link**: [PDF](https://arxiv.org/pdf/2505.01309)  

**Abstract**: SPARQL query rewriting is a fundamental mechanism for uniformly querying heterogeneous ontologies in the Linked Data Web. However, the complexity of ontology alignments, particularly rich correspondences (c : c), makes this process challenging. Existing approaches primarily focus on simple (s : s) and partially complex ( s : c) alignments, thereby overlooking the challenges posed by more expressive alignments. Moreover, the intricate syntax of SPARQL presents a barrier for non-expert users seeking to fully exploit the knowledge encapsulated in ontologies. This article proposes an innovative approach for the automatic rewriting of SPARQL queries from a source ontology to a target ontology, based on a user's need expressed in natural language. It leverages the principles of equivalence transitivity as well as the advanced capabilities of large language models such as GPT-4. By integrating these elements, this approach stands out for its ability to efficiently handle complex alignments, particularly (c : c) correspondences , by fully exploiting their expressiveness. Additionally, it facilitates access to aligned ontologies for users unfamiliar with SPARQL, providing a flexible solution for querying heterogeneous data. 

---
# LLM Security: Vulnerabilities, Attacks, Defenses, and Countermeasures 

**Authors**: Francisco Aguilera-Martínez, Fernando Berzal  

**Link**: [PDF](https://arxiv.org/pdf/2505.01177)  

**Abstract**: As large language models (LLMs) continue to evolve, it is critical to assess the security threats and vulnerabilities that may arise both during their training phase and after models have been deployed. This survey seeks to define and categorize the various attacks targeting LLMs, distinguishing between those that occur during the training phase and those that affect already trained models. A thorough analysis of these attacks is presented, alongside an exploration of defense mechanisms designed to mitigate such threats. Defenses are classified into two primary categories: prevention-based and detection-based defenses. Furthermore, our survey summarizes possible attacks and their corresponding defense strategies. It also provides an evaluation of the effectiveness of the known defense mechanisms for the different security threats. Our survey aims to offer a structured framework for securing LLMs, while also identifying areas that require further research to improve and strengthen defenses against emerging security challenges. 

---
# Good News for Script Kiddies? Evaluating Large Language Models for Automated Exploit Generation 

**Authors**: David Jin, Qian Fu, Yuekang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.01065)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in code-related tasks, raising concerns about their potential for automated exploit generation (AEG). This paper presents the first systematic study on LLMs' effectiveness in AEG, evaluating both their cooperativeness and technical proficiency. To mitigate dataset bias, we introduce a benchmark with refactored versions of five software security labs. Additionally, we design an LLM-based attacker to systematically prompt LLMs for exploit generation. Our experiments reveal that GPT-4 and GPT-4o exhibit high cooperativeness, comparable to uncensored models, while Llama3 is the most resistant. However, no model successfully generates exploits for refactored labs, though GPT-4o's minimal errors highlight the potential for LLM-driven AEG advancements. 

---
# Low-Precision Training of Large Language Models: Methods, Challenges, and Opportunities 

**Authors**: Zhiwei Hao, Jianyuan Guo, Li Shen, Yong Luo, Han Hu, Guoxia Wang, Dianhai Yu, Yonggang Wen, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2505.01043)  

**Abstract**: Large language models (LLMs) have achieved impressive performance across various domains. However, the substantial hardware resources required for their training present a significant barrier to efficiency and scalability. To mitigate this challenge, low-precision training techniques have been widely adopted, leading to notable advancements in training efficiency. Despite these gains, low-precision training involves several components$\unicode{x2013}$such as weights, activations, and gradients$\unicode{x2013}$each of which can be represented in different numerical formats. The resulting diversity has created a fragmented landscape in low-precision training research, making it difficult for researchers to gain a unified overview of the field. This survey provides a comprehensive review of existing low-precision training methods. To systematically organize these approaches, we categorize them into three primary groups based on their underlying numerical formats, which is a key factor influencing hardware compatibility, computational efficiency, and ease of reference for readers. The categories are: (1) fixed-point and integer-based methods, (2) floating-point-based methods, and (3) customized format-based methods. Additionally, we discuss quantization-aware training approaches, which share key similarities with low-precision training during forward propagation. Finally, we highlight several promising research directions to advance this field. A collection of papers discussed in this survey is provided in this https URL. 

---
# A Rusty Link in the AI Supply Chain: Detecting Evil Configurations in Model Repositories 

**Authors**: Ziqi Ding, Qian Fu, Junchen Ding, Gelei Deng, Yi Liu, Yuekang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.01067)  

**Abstract**: Recent advancements in large language models (LLMs) have spurred the development of diverse AI applications from code generation and video editing to text generation; however, AI supply chains such as Hugging Face, which host pretrained models and their associated configuration files contributed by the public, face significant security challenges; in particular, configuration files originally intended to set up models by specifying parameters and initial settings can be exploited to execute unauthorized code, yet research has largely overlooked their security compared to that of the models themselves; in this work, we present the first comprehensive study of malicious configurations on Hugging Face, identifying three attack scenarios (file, website, and repository operations) that expose inherent risks; to address these threats, we introduce CONFIGSCAN, an LLM-based tool that analyzes configuration files in the context of their associated runtime code and critical libraries, effectively detecting suspicious elements with low false positive rates and high accuracy; our extensive evaluation uncovers thousands of suspicious repositories and configuration files, underscoring the urgent need for enhanced security validation in AI model hosting platforms. 

---
# From Texts to Shields: Convergence of Large Language Models and Cybersecurity 

**Authors**: Tao Li, Ya-Ting Yang, Yunian Pan, Quanyan Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00841)  

**Abstract**: This report explores the convergence of large language models (LLMs) and cybersecurity, synthesizing interdisciplinary insights from network security, artificial intelligence, formal methods, and human-centered design. It examines emerging applications of LLMs in software and network security, 5G vulnerability analysis, and generative security engineering. The report highlights the role of agentic LLMs in automating complex tasks, improving operational efficiency, and enabling reasoning-driven security analytics. Socio-technical challenges associated with the deployment of LLMs -- including trust, transparency, and ethical considerations -- can be addressed through strategies such as human-in-the-loop systems, role-specific training, and proactive robustness testing. The report further outlines critical research challenges in ensuring interpretability, safety, and fairness in LLM-based systems, particularly in high-stakes domains. By integrating technical advances with organizational and societal considerations, this report presents a forward-looking research agenda for the secure and effective adoption of LLMs in cybersecurity. 

---
# Zoomer: Adaptive Image Focus Optimization for Black-box MLLM 

**Authors**: Jiaxu Qian, Chendong Wang, Yifan Yang, Chaoyun Zhang, Huiqiang Jiang, Xufang Luo, Yu Kang, Qingwei Lin, Anlan Zhang, Shiqi Jiang, Ting Cao, Tianjun Mao, Suman Banerjee, Guyue Liu, Saravan Rajmohan, Dongmei Zhang, Yuqing Yang, Qi Zhang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00742)  

**Abstract**: Recent advancements in multimodal large language models (MLLMs) have broadened the scope of vision-language tasks, excelling in applications like image captioning and interactive question-answering. However, these models struggle with accurately processing visual data, particularly in tasks requiring precise object recognition and fine visual details. Stringent token limits often result in the omission of critical information, hampering performance. To address these limitations, we introduce \SysName, a novel visual prompting mechanism designed to enhance MLLM performance while preserving essential visual details within token limits. \SysName features three key innovations: a prompt-aware strategy that dynamically highlights relevant image regions, a spatial-preserving orchestration schema that maintains object integrity, and a budget-aware prompting method that balances global context with crucial visual details. Comprehensive evaluations across multiple datasets demonstrate that \SysName consistently outperforms baseline methods, achieving up to a $26.9\%$ improvement in accuracy while significantly reducing token consumption. 

---
# Spill The Beans: Exploiting CPU Cache Side-Channels to Leak Tokens from Large Language Models 

**Authors**: Andrew Adiletta, Berk Sunar  

**Link**: [PDF](https://arxiv.org/pdf/2505.00817)  

**Abstract**: Side-channel attacks on shared hardware resources increasingly threaten confidentiality, especially with the rise of Large Language Models (LLMs). In this work, we introduce Spill The Beans, a novel application of cache side-channels to leak tokens generated by an LLM. By co-locating an attack process on the same hardware as the victim model, we flush and reload embedding vectors from the embedding layer, where each token corresponds to a unique embedding vector. When accessed during token generation, it results in a cache hit detectable by our attack on shared lower-level caches.
A significant challenge is the massive size of LLMs, which, by nature of their compute intensive operation, quickly evicts embedding vectors from the cache. We address this by balancing the number of tokens monitored against the amount of information leaked. Monitoring more tokens increases potential vocabulary leakage but raises the chance of missing cache hits due to eviction; monitoring fewer tokens improves detection reliability but limits vocabulary coverage.
Through extensive experimentation, we demonstrate the feasibility of leaking tokens from LLMs via cache side-channels. Our findings reveal a new vulnerability in LLM deployments, highlighting that even sophisticated models are susceptible to traditional side-channel attacks. We discuss the implications for privacy and security in LLM-serving infrastructures and suggest considerations for mitigating such threats. For proof of concept we consider two concrete attack scenarios: Our experiments show that an attacker can recover as much as 80%-90% of a high entropy API key with single shot monitoring. As for English text we can reach a 40% recovery rate with a single shot. We should note that the rate highly depends on the monitored token set and these rates can be improved by targeting more specialized output domains. 

---
