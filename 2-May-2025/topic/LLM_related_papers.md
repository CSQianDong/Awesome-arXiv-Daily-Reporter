# Investigating Task Arithmetic for Zero-Shot Information Retrieval 

**Authors**: Marco Braga, Pranav Kasela, Alessandro Raganato, Gabriella Pasi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00649)  

**Abstract**: Large Language Models (LLMs) have shown impressive zero-shot performance across a variety of Natural Language Processing tasks, including document re-ranking. However, their effectiveness degrades on unseen tasks and domains, largely due to shifts in vocabulary and word distributions. In this paper, we investigate Task Arithmetic, a technique that combines the weights of LLMs pre-trained on different tasks or domains via simple mathematical operations, such as addition or subtraction, to adapt retrieval models without requiring additional fine-tuning. Our method is able to synthesize diverse tasks and domain knowledge into a single model, enabling effective zero-shot adaptation in different retrieval contexts. Extensive experiments on publicly available scientific, biomedical, and multilingual datasets show that our method improves state-of-the-art re-ranking performance by up to 18% in NDCG@10 and 15% in P@10. In addition to these empirical gains, our analysis provides insights into the strengths and limitations of Task Arithmetic as a practical strategy for zero-shot learning and model adaptation. We make our code publicly available at this https URL. 

---
# Enhancing Speech-to-Speech Dialogue Modeling with End-to-End Retrieval-Augmented Generation 

**Authors**: Pengchao Feng, Ziyang Ma, Wenxi Chen, Yao Li, Sheng Wang, Kai Yu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00028)  

**Abstract**: In recent years, end-to-end speech-to-speech (S2S) dialogue systems have garnered increasing research attention due to their advantages over traditional cascaded systems, including achieving lower latency and more natural integration of nonverbal cues such as emotion and speaker identity. However, these end-to-end systems face key challenges, particularly in incorporating external knowledge, a capability commonly addressed by Retrieval-Augmented Generation (RAG) in text-based large language models (LLMs). The core difficulty lies in the modality gap between input speech and retrieved textual knowledge, which hinders effective integration. To address this issue, we propose a novel end-to-end RAG framework that directly retrieves relevant textual knowledge from speech queries, eliminating the need for intermediate speech-to-text conversion via techniques like ASR. Experimental results demonstrate that our method significantly improves the performance of end-to-end S2S dialogue systems while achieving higher retrieval efficiency. Although the overall performance still lags behind cascaded models, our framework offers a promising direction for enhancing knowledge integration in end-to-end S2S systems. We will release the code and dataset to support reproducibility and promote further research in this area. 

---
# Open-Source LLM-Driven Federated Transformer for Predictive IoV Management 

**Authors**: Yazan Otoum, Arghavan Asad, Ishtiaq Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2505.00651)  

**Abstract**: The proliferation of connected vehicles within the Internet of Vehicles (IoV) ecosystem presents critical challenges in ensuring scalable, real-time, and privacy-preserving traffic management. Existing centralized IoV solutions often suffer from high latency, limited scalability, and reliance on proprietary Artificial Intelligence (AI) models, creating significant barriers to widespread deployment, particularly in dynamic and privacy-sensitive environments. Meanwhile, integrating Large Language Models (LLMs) in vehicular systems remains underexplored, especially concerning prompt optimization and effective utilization in federated contexts. To address these challenges, we propose the Federated Prompt-Optimized Traffic Transformer (FPoTT), a novel framework that leverages open-source LLMs for predictive IoV management. FPoTT introduces a dynamic prompt optimization mechanism that iteratively refines textual prompts to enhance trajectory prediction. The architecture employs a dual-layer federated learning paradigm, combining lightweight edge models for real-time inference with cloud-based LLMs to retain global intelligence. A Transformer-driven synthetic data generator is incorporated to augment training with diverse, high-fidelity traffic scenarios in the Next Generation Simulation (NGSIM) format. Extensive evaluations demonstrate that FPoTT, utilizing EleutherAI Pythia-1B, achieves 99.86% prediction accuracy on real-world data while maintaining high performance on synthetic datasets. These results underscore the potential of open-source LLMs in enabling secure, adaptive, and scalable IoV management, offering a promising alternative to proprietary solutions in smart mobility ecosystems. 

---
# UserCentrix: An Agentic Memory-augmented AI Framework for Smart Spaces 

**Authors**: Alaa Saleh, Sasu Tarkoma, Praveen Kumar Donta, Naser Hossein Motlagh, Schahram Dustdar, Susanna Pirttikangas, Lauri Lovén  

**Link**: [PDF](https://arxiv.org/pdf/2505.00472)  

**Abstract**: Agentic AI, with its autonomous and proactive decision-making, has transformed smart environments. By integrating Generative AI (GenAI) and multi-agent systems, modern AI frameworks can dynamically adapt to user preferences, optimize data management, and improve resource allocation. This paper introduces UserCentrix, an agentic memory-augmented AI framework designed to enhance smart spaces through dynamic, context-aware decision-making. This framework integrates personalized Large Language Model (LLM) agents that leverage user preferences and LLM memory management to deliver proactive and adaptive assistance. Furthermore, it incorporates a hybrid hierarchical control system, balancing centralized and distributed processing to optimize real-time responsiveness while maintaining global situational awareness. UserCentrix achieves resource-efficient AI interactions by embedding memory-augmented reasoning, cooperative agent negotiation, and adaptive orchestration strategies. Our key contributions include (i) a self-organizing framework with proactive scaling based on task urgency, (ii) a Value of Information (VoI)-driven decision-making process, (iii) a meta-reasoning personal LLM agent, and (iv) an intelligent multi-agent coordination system for seamless environment adaptation. Experimental results across various models confirm the effectiveness of our approach in enhancing response accuracy, system efficiency, and computational resource management in real-world application. 

---
# Urban Air Mobility as a System of Systems: An LLM-Enhanced Holonic Approach 

**Authors**: Ahmed R. Sadik, Muhammad Ashfaq, Niko Mäkitalo, Tommi Mikkonen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00368)  

**Abstract**: Urban Air Mobility (UAM) is an emerging System of System (SoS) that faces challenges in system architecture, planning, task management, and execution. Traditional architectural approaches struggle with scalability, adaptability, and seamless resource integration within dynamic and complex environments. This paper presents an intelligent holonic architecture that incorporates Large Language Model (LLM) to manage the complexities of UAM. Holons function semi autonomously, allowing for real time coordination among air taxis, ground transport, and vertiports. LLMs process natural language inputs, generate adaptive plans, and manage disruptions such as weather changes or airspace this http URL a case study of multimodal transportation with electric scooters and air taxis, we demonstrate how this architecture enables dynamic resource allocation, real time replanning, and autonomous adaptation without centralized control, creating more resilient and efficient urban transportation networks. By advancing decentralized control and AI driven adaptability, this work lays the groundwork for resilient, human centric UAM ecosystems, with future efforts targeting hybrid AI integration and real world validation. 

---
# RAIL in the Wild: Operationalizing Responsible AI Evaluation Using Anthropic's Value Dataset 

**Authors**: Sumit Verma, Pritam Prasun, Arpit Jaiswal, Pritish Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.00204)  

**Abstract**: As AI systems become embedded in real-world applications, ensuring they meet ethical standards is crucial. While existing AI ethics frameworks emphasize fairness, transparency, and accountability, they often lack actionable evaluation methods. This paper introduces a systematic approach using the Responsible AI Labs (RAIL) framework, which includes eight measurable dimensions to assess the normative behavior of large language models (LLMs). We apply this framework to Anthropic's "Values in the Wild" dataset, containing over 308,000 anonymized conversations with Claude and more than 3,000 annotated value expressions. Our study maps these values to RAIL dimensions, computes synthetic scores, and provides insights into the ethical behavior of LLMs in real-world use. 

---
# Can LLMs Help Improve Analogical Reasoning For Strategic Decisions? Experimental Evidence from Humans and GPT-4 

**Authors**: Phanish Puranam, Prothit Sen, Maciej Workiewicz  

**Link**: [PDF](https://arxiv.org/pdf/2505.00603)  

**Abstract**: This study investigates whether large language models, specifically GPT4, can match human capabilities in analogical reasoning within strategic decision making contexts. Using a novel experimental design involving source to target matching, we find that GPT4 achieves high recall by retrieving all plausible analogies but suffers from low precision, frequently applying incorrect analogies based on superficial similarities. In contrast, human participants exhibit high precision but low recall, selecting fewer analogies yet with stronger causal alignment. These findings advance theory by identifying matching, the evaluative phase of analogical reasoning, as a distinct step that requires accurate causal mapping beyond simple retrieval. While current LLMs are proficient in generating candidate analogies, humans maintain a comparative advantage in recognizing deep structural similarities across domains. Error analysis reveals that AI errors arise from surface level matching, whereas human errors stem from misinterpretations of causal structure. Taken together, the results suggest a productive division of labor in AI assisted organizational decision making where LLMs may serve as broad analogy generators, while humans act as critical evaluators, applying the most contextually appropriate analogies to strategic problems. 

---
# Combining LLMs with Logic-Based Framework to Explain MCTS 

**Authors**: Ziyan An, Xia Wang, Hendrik Baier, Zirong Chen, Abhishek Dubey, Taylor T. Johnson, Jonathan Sprinkle, Ayan Mukhopadhyay, Meiyi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.00610)  

**Abstract**: In response to the lack of trust in Artificial Intelligence (AI) for sequential planning, we design a Computational Tree Logic-guided large language model (LLM)-based natural language explanation framework designed for the Monte Carlo Tree Search (MCTS) algorithm. MCTS is often considered challenging to interpret due to the complexity of its search trees, but our framework is flexible enough to handle a wide range of free-form post-hoc queries and knowledge-based inquiries centered around MCTS and the Markov Decision Process (MDP) of the application domain. By transforming user queries into logic and variable statements, our framework ensures that the evidence obtained from the search tree remains factually consistent with the underlying environmental dynamics and any constraints in the actual stochastic control process. We evaluate the framework rigorously through quantitative assessments, where it demonstrates strong performance in terms of accuracy and factual consistency. 

---
# On the generalization of language models from in-context learning and finetuning: a controlled study 

**Authors**: Andrew K. Lampinen, Arslan Chaudhry, Stephanie C.Y. Chan, Cody Wild, Diane Wan, Alex Ku, Jörg Bornschein, Razvan Pascanu, Murray Shanahan, James L. McClelland  

**Link**: [PDF](https://arxiv.org/pdf/2505.00661)  

**Abstract**: Large language models exhibit exciting capabilities, yet can show surprisingly narrow generalization from finetuning -- from failing to generalize to simple reversals of relations they are trained on, to missing logical deductions that can be made from trained information. These failures to generalize from fine-tuning can hinder practical application of these models. However, language models' in-context learning shows different inductive biases, and can generalize better in some of these cases. Here, we explore these differences in generalization between in-context- and fine-tuning-based learning. To do so, we constructed several novel datasets to evaluate and improve models' ability to generalize from finetuning data. The datasets are constructed to isolate the knowledge in the dataset from that in pretraining, to create clean tests of generalization. We expose pretrained large models to controlled subsets of the information in these datasets -- either in context, or through fine-tuning -- and evaluate their performance on test sets that require various types of generalization. We find overall that in data-matched settings, in-context learning can generalize more flexibly than fine-tuning (though we also find some qualifications of prior findings, such as cases when fine-tuning can generalize to reversals embedded in a larger structure of knowledge). We build on these findings to propose a method to enable improved generalization from fine-tuning: adding in-context inferences to finetuning data. We show that this method improves generalization across various splits of our datasets and other benchmarks. Our results have implications for understanding the inductive biases of different modes of learning in language models, and practically improving their performance. 

---
# DeepCritic: Deliberate Critique with Large Language Models 

**Authors**: Wenkai Yang, Jingwen Chen, Yankai Lin, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00662)  

**Abstract**: As Large Language Models (LLMs) are rapidly evolving, providing accurate feedback and scalable oversight on their outputs becomes an urgent and critical problem. Leveraging LLMs as critique models to achieve automated supervision is a promising solution. In this work, we focus on studying and enhancing the math critique ability of LLMs. Current LLM critics provide critiques that are too shallow and superficial on each step, leading to low judgment accuracy and struggling to offer sufficient feedback for the LLM generator to correct mistakes. To tackle this issue, we propose a novel and effective two-stage framework to develop LLM critics that are capable of deliberately critiquing on each reasoning step of math solutions. In the first stage, we utilize Qwen2.5-72B-Instruct to generate 4.5K long-form critiques as seed data for supervised fine-tuning. Each seed critique consists of deliberate step-wise critiques that includes multi-perspective verifications as well as in-depth critiques of initial critiques for each reasoning step. Then, we perform reinforcement learning on the fine-tuned model with either existing human-labeled data from PRM800K or our automatically annotated data obtained via Monte Carlo sampling-based correctness estimation, to further incentivize its critique ability. Our developed critique model built on Qwen2.5-7B-Instruct not only significantly outperforms existing LLM critics (including the same-sized DeepSeek-R1-distill models and GPT-4o) on various error identification benchmarks, but also more effectively helps the LLM generator refine erroneous steps through more detailed feedback. 

---
# The Illusion of Role Separation: Hidden Shortcuts in LLM Role Learning (and How to Fix Them) 

**Authors**: Zihao Wang, Yibo Jiang, Jiahao Yu, Heqing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00626)  

**Abstract**: Large language models (LLMs) that integrate multiple input roles (e.g., system instructions, user queries, external tool outputs) are increasingly prevalent in practice. Ensuring that the model accurately distinguishes messages from each role -- a concept we call \emph{role separation} -- is crucial for consistent multi-role behavior. Although recent work often targets state-of-the-art prompt injection defenses, it remains unclear whether such methods truly teach LLMs to differentiate roles or merely memorize known triggers. In this paper, we examine \emph{role-separation learning}: the process of teaching LLMs to robustly distinguish system and user tokens. Through a \emph{simple, controlled experimental framework}, we find that fine-tuned models often rely on two proxies for role identification: (1) task type exploitation, and (2) proximity to begin-of-text. Although data augmentation can partially mitigate these shortcuts, it generally leads to iterative patching rather than a deeper fix. To address this, we propose reinforcing \emph{invariant signals} that mark role boundaries by adjusting token-wise cues in the model's input encoding. In particular, manipulating position IDs helps the model learn clearer distinctions and reduces reliance on superficial proxies. By focusing on this mechanism-centered perspective, our work illuminates how LLMs can more reliably maintain consistent multi-role behavior without merely memorizing known prompts or triggers. 

---
# FineScope : Precision Pruning for Domain-Specialized Large Language Models Using SAE-Guided Self-Data Cultivation 

**Authors**: Chaitali Bhattacharyya, Yeseong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00624)  

**Abstract**: Training large language models (LLMs) from scratch requires significant computational resources, driving interest in developing smaller, domain-specific LLMs that maintain both efficiency and strong task performance. Medium-sized models such as LLaMA, llama} have served as starting points for domain-specific adaptation, but they often suffer from accuracy degradation when tested on specialized datasets. We introduce FineScope, a framework for deriving compact, domain-optimized LLMs from larger pretrained models. FineScope leverages the Sparse Autoencoder (SAE) framework, inspired by its ability to produce interpretable feature representations, to extract domain-specific subsets from large datasets. We apply structured pruning with domain-specific constraints, ensuring that the resulting pruned models retain essential knowledge for the target domain. To further enhance performance, these pruned models undergo self-data distillation, leveraging SAE-curated datasets to restore key domain-specific information lost during pruning. Extensive experiments and ablation studies demonstrate that FineScope achieves highly competitive performance, outperforming several large-scale state-of-the-art LLMs in domain-specific tasks. Additionally, our results show that FineScope enables pruned models to regain a substantial portion of their original performance when fine-tuned with SAE-curated datasets. Furthermore, applying these datasets to fine-tune pretrained LLMs without pruning also improves their domain-specific accuracy, highlighting the robustness of our approach. The code will be released. 

---
# Red Teaming Large Language Models for Healthcare 

**Authors**: Vahid Balazadeh, Michael Cooper, David Pellow, Atousa Assadi, Jennifer Bell, Jim Fackler, Gabriel Funingana, Spencer Gable-Cook, Anirudh Gangadhar, Abhishek Jaiswal, Sumanth Kaja, Christopher Khoury, Randy Lin, Kaden McKeen, Sara Naimimohasses, Khashayar Namdar, Aviraj Newatia, Allan Pang, Anshul Pattoo, Sameer Peesapati, Diana Prepelita, Bogdana Rakova, Saba Sadatamin, Rafael Schulman, Ajay Shah, Syed Azhar Shah, Syed Ahmar Shah, Babak Taati, Balagopal Unnikrishnan, Stephanie Williams, Rahul G Krishnan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00467)  

**Abstract**: We present the design process and findings of the pre-conference workshop at the Machine Learning for Healthcare Conference (2024) entitled Red Teaming Large Language Models for Healthcare, which took place on August 15, 2024. Conference participants, comprising a mix of computational and clinical expertise, attempted to discover vulnerabilities -- realistic clinical prompts for which a large language model (LLM) outputs a response that could cause clinical harm. Red-teaming with clinicians enables the identification of LLM vulnerabilities that may not be recognised by LLM developers lacking clinical expertise. We report the vulnerabilities found, categorise them, and present the results of a replication study assessing the vulnerabilities across all LLMs provided. 

---
# Triggering Hallucinations in LLMs: A Quantitative Study of Prompt-Induced Hallucination in Large Language Models 

**Authors**: Makoto Sato  

**Link**: [PDF](https://arxiv.org/pdf/2505.00557)  

**Abstract**: Hallucinations in large language models (LLMs) present a growing challenge across real-world applications, from healthcare to law, where factual reliability is essential. Despite advances in alignment and instruction tuning, LLMs can still generate outputs that are fluent yet fundamentally untrue. Understanding the cognitive dynamics that underlie these hallucinations remains an open problem. In this study, we propose a prompt-based framework to systematically trigger and quantify hallucination: a Hallucination-Inducing Prompt (HIP), which synthetically fuses semantically distant concepts (e.g., periodic table of elements and tarot divination) in a misleading way, and a Hallucination Quantifying Prompt (HQP), which scores the plausibility, confidence, and coherence of the output. Controlled experiments across multiple LLMs revealed that HIPs consistently produced less coherent and more hallucinated responses than their null-fusion controls. These effects varied across models, with reasoning-oriented LLMs showing distinct profiles from general-purpose ones. Our framework provides a reproducible testbed for studying hallucination vulnerability, and opens the door to developing safer, more introspective LLMs that can detect and self-regulate the onset of conceptual instability. 

---
# Data Therapist: Eliciting Domain Knowledge from Subject Matter Experts Using Large Language Models 

**Authors**: Sungbok Shin, Hyeon Jeon, Sanghyun Hong, Niklas Elmqvist  

**Link**: [PDF](https://arxiv.org/pdf/2505.00455)  

**Abstract**: Effective data visualization requires not only technical proficiency but also a deep understanding of the domain-specific context in which data exists. This context often includes tacit knowledge about data provenance, quality, and intended use, which is rarely explicit in the dataset itself. We present the Data Therapist, a web-based tool that helps domain experts externalize this implicit knowledge through a mixed-initiative process combining iterative Q&A with interactive annotation. Powered by a large language model, the system analyzes user-supplied datasets, prompts users with targeted questions, and allows annotation at varying levels of granularity. The resulting structured knowledge base can inform both human and automated visualization design. We evaluated the tool in a qualitative study involving expert pairs from Molecular Biology, Accounting, Political Science, and Usable Security. The study revealed recurring patterns in how experts reason about their data and highlights areas where AI support can improve visualization design. 

---
# KoACD: The First Korean Adolescent Dataset for Cognitive Distortion Analysis 

**Authors**: JunSeo Kim, HyeHyeon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.00367)  

**Abstract**: Cognitive distortion refers to negative thinking patterns that can lead to mental health issues like depression and anxiety in adolescents. Previous studies using natural language processing (NLP) have focused mainly on small-scale adult datasets, with limited research on adolescents. This study introduces KoACD, the first large-scale dataset of cognitive distortions in Korean adolescents, containing 108,717 instances. We applied a multi-Large Language Model (LLM) negotiation method to refine distortion classification and generate synthetic data using two approaches: cognitive clarification for textual clarity and cognitive balancing for diverse distortion representation. Validation through LLMs and expert evaluations showed that while LLMs classified distortions with explicit markers, they struggled with context-dependent reasoning, where human evaluators demonstrated higher accuracy. KoACD aims to enhance future research on cognitive distortion detection. 

---
# Consistency in Language Models: Current Landscape, Challenges, and Future Directions 

**Authors**: Jekaterina Novikova, Carol Anderson, Borhane Blili-Hamelin, Subhabrata Majumdar  

**Link**: [PDF](https://arxiv.org/pdf/2505.00268)  

**Abstract**: The hallmark of effective language use lies in consistency -- expressing similar meanings in similar contexts and avoiding contradictions. While human communication naturally demonstrates this principle, state-of-the-art language models struggle to maintain reliable consistency across different scenarios. This paper examines the landscape of consistency research in AI language systems, exploring both formal consistency (including logical rule adherence) and informal consistency (such as moral and factual coherence). We analyze current approaches to measure aspects of consistency, identify critical research gaps in standardization of definitions, multilingual assessment, and methods to improve consistency. Our findings point to an urgent need for robust benchmarks to measure and interdisciplinary approaches to ensure consistency in the application of language models on domain-specific tasks while preserving the utility and adaptability. 

---
# LLM-Based Threat Detection and Prevention Framework for IoT Ecosystems 

**Authors**: Yazan Otoum, Arghavan Asad, Amiya Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2505.00240)  

**Abstract**: The increasing complexity and scale of the Internet of Things (IoT) have made security a critical concern. This paper presents a novel Large Language Model (LLM)-based framework for comprehensive threat detection and prevention in IoT environments. The system integrates lightweight LLMs fine-tuned on IoT-specific datasets (IoT-23, TON_IoT) for real-time anomaly detection and automated, context-aware mitigation strategies optimized for resource-constrained devices. A modular Docker-based deployment enables scalable and reproducible evaluation across diverse network conditions. Experimental results in simulated IoT environments demonstrate significant improvements in detection accuracy, response latency, and resource efficiency over traditional security methods. The proposed framework highlights the potential of LLM-driven, autonomous security solutions for future IoT ecosystems. 

---
# Between Underthinking and Overthinking: An Empirical Study of Reasoning Length and correctness in LLMs 

**Authors**: Jinyan Su, Jennifer Healey, Preslav Nakov, Claire Cardie  

**Link**: [PDF](https://arxiv.org/pdf/2505.00127)  

**Abstract**: Large language models (LLMs) are increasingly optimized for long reasoning, under the assumption that more reasoning leads to better performance. However, emerging evidence suggests that longer responses can sometimes degrade accuracy rather than improve it. In this paper, we conduct a systematic empirical study of the relationship between reasoning length and answer correctness. We find that LLMs tend to overthink simple problems, generating unnecessarily long outputs, and underthink harder ones, failing to extend their reasoning when it is most needed. This indicates that models might misjudge problem difficulty and fail to calibrate their response length appropriately. Furthermore, we investigate the effects of length reduction with a preference optimization algorithm when simply preferring the shorter responses regardless of answer correctness. Experiments show that the generation length can be significantly reduced while maintaining acceptable accuracy. Our findings highlight generation length as a meaningful signal for reasoning behavior and motivate further exploration into LLMs' self-awareness in reasoning length adaptation. 

---
# Fine-Tuning LLMs for Low-Resource Dialect Translation: The Case of Lebanese 

**Authors**: Silvana Yakhni, Ali Chehab  

**Link**: [PDF](https://arxiv.org/pdf/2505.00114)  

**Abstract**: This paper examines the effectiveness of Large Language Models (LLMs) in translating the low-resource Lebanese dialect, focusing on the impact of culturally authentic data versus larger translated datasets. We compare three fine-tuning approaches: Basic, contrastive, and grammar-hint tuning, using open-source Aya23 models. Experiments reveal that models fine-tuned on a smaller but culturally aware Lebanese dataset (LW) consistently outperform those trained on larger, non-native data. The best results were achieved through contrastive fine-tuning paired with contrastive prompting, which indicates the benefits of exposing translation models to bad examples. In addition, to ensure authentic evaluation, we introduce LebEval, a new benchmark derived from native Lebanese content, and compare it to the existing FLoRes benchmark. Our findings challenge the "More Data is Better" paradigm and emphasize the crucial role of cultural authenticity in dialectal translation. We made our datasets and code available on Github. 

---
# Fact-Consistency Evaluation of Text-to-SQL Generation for Business Intelligence Using Exaone 3.5 

**Authors**: Jeho Choi  

**Link**: [PDF](https://arxiv.org/pdf/2505.00060)  

**Abstract**: Large Language Models (LLMs) have shown promise in enabling natural language interfaces for structured data querying through text-to-SQL generation. However, their application in real-world Business Intelligence (BI) contexts remains limited due to semantic hallucinations, structural errors, and a lack of domain-specific evaluation frameworks. In this study, we propose a Fact-Consistency Evaluation Framework for assessing the semantic accuracy of LLM-generated SQL outputs using Exaone 3.5--an instruction-tuned, bilingual LLM optimized for enterprise tasks. We construct a domain-specific benchmark comprising 219 natural language business questions across five SQL complexity levels, derived from actual sales data in LG Electronics' internal BigQuery environment. Each question is paired with a gold-standard SQL query and a validated ground-truth answer. We evaluate model performance using answer accuracy, execution success rate, semantic error rate, and non-response rate. Experimental results show that while Exaone 3.5 performs well on simple aggregation tasks (93% accuracy in L1), it exhibits substantial degradation in arithmetic reasoning (4% accuracy in H1) and grouped ranking tasks (31% in H4), with semantic errors and non-responses concentrated in complex cases. Qualitative error analysis further identifies common failure types such as misapplied arithmetic logic, incomplete filtering, and incorrect grouping operations. Our findings highlight the current limitations of LLMs in business-critical environments and underscore the need for fact-consistency validation layers and hybrid reasoning approaches. This work contributes a reproducible benchmark and evaluation methodology for advancing reliable natural language interfaces to structured enterprise data systems. 

---
# Improving Phishing Email Detection Performance of Small Large Language Models 

**Authors**: Zijie Lin, Zikang Liu, Hanbo Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00034)  

**Abstract**: Large language models(LLMs) have demonstrated remarkable performance on many natural language processing(NLP) tasks and have been employed in phishing email detection research. However, in current studies, well-performing LLMs typically contain billions or even tens of billions of parameters, requiring enormous computational resources. To reduce computational costs, we investigated the effectiveness of small-parameter LLMs for phishing email detection. These LLMs have around 3 billion parameters and can run on consumer-grade GPUs. However, small LLMs often perform poorly in phishing email detection task. To address these issues, we designed a set of methods including Prompt Engineering, Explanation Augmented Fine-tuning, and Model Ensemble to improve phishing email detection capabilities of small LLMs. We validated the effectiveness of our approach through experiments, significantly improving accuracy on the SpamAssassin dataset from around 0.5 for baseline models like Qwen2.5-1.5B-Instruct to 0.976. 

---
# MDD-LLM: Towards Accuracy Large Language Models for Major Depressive Disorder Diagnosis 

**Authors**: Yuyang Sha, Hongxin Pan, Wei Xu, Weiyu Meng, Gang Luo, Xinyu Du, Xiaobing Zhai, Henry H. Y. Tong, Caijuan Shi, Kefeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.00032)  

**Abstract**: Major depressive disorder (MDD) impacts more than 300 million people worldwide, highlighting a significant public health issue. However, the uneven distribution of medical resources and the complexity of diagnostic methods have resulted in inadequate attention to this disorder in numerous countries and regions. This paper introduces a high-performance MDD diagnosis tool named MDD-LLM, an AI-driven framework that utilizes fine-tuned large language models (LLMs) and extensive real-world samples to tackle challenges in MDD diagnosis. Therefore, we select 274,348 individual information from the UK Biobank cohort to train and evaluate the proposed method. Specifically, we select 274,348 individual records from the UK Biobank cohort and design a tabular data transformation method to create a large corpus for training and evaluating the proposed approach. To illustrate the advantages of MDD-LLM, we perform comprehensive experiments and provide several comparative analyses against existing model-based solutions across multiple evaluation metrics. Experimental results show that MDD-LLM (70B) achieves an accuracy of 0.8378 and an AUC of 0.8919 (95% CI: 0.8799 - 0.9040), significantly outperforming existing machine learning and deep learning frameworks for MDD diagnosis. Given the limited exploration of LLMs in MDD diagnosis, we examine numerous factors that may influence the performance of our proposed method, such as tabular data transformation techniques and different fine-tuning strategies. 

---
# Learning to Plan Before Answering: Self-Teaching LLMs to Learn Abstract Plans for Problem Solving 

**Authors**: Jin Zhang, Flood Sung, Zhilin Yang, Yang Gao, Chongjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00031)  

**Abstract**: In the field of large language model (LLM) post-training, the effectiveness of utilizing synthetic data generated by the LLM itself has been well-presented. However, a key question remains unaddressed: what essential information should such self-generated data encapsulate? Existing approaches only produce step-by-step problem solutions, and fail to capture the abstract meta-knowledge necessary for generalization across similar problems. Drawing insights from cognitive science, where humans employ high-level abstraction to simplify complex problems before delving into specifics, we introduce a novel self-training algorithm: LEarning to Plan before Answering (LEPA). LEPA trains the LLM to formulate anticipatory plans, which serve as abstract meta-knowledge for problem-solving, before engaging with the intricacies of problems. This approach not only outlines the solution generation path but also shields the LLM from the distraction of irrelevant details. During data generation, LEPA first crafts an anticipatory plan based on the problem, and then generates a solution that aligns with both the plan and the problem. LEPA refines the plan through self-reflection, aiming to acquire plans that are instrumental in yielding correct solutions. During model optimization, the LLM is trained to predict both the refined plans and the corresponding solutions. By efficiently extracting and utilizing the anticipatory plans, LEPA demonstrates remarkable superiority over conventional algorithms on various challenging natural language reasoning benchmarks. 

---
# Theory of Mind in Large Language Models: Assessment and Enhancement 

**Authors**: Ruirui Chen, Weifeng Jiang, Chengwei Qin, Cheston Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00026)  

**Abstract**: Theory of Mind (ToM)-the ability to infer and reason about others' mental states-is fundamental to human social intelligence. As Large Language Models (LLMs) become increasingly integrated into daily life, it is crucial to assess and enhance their capacity to interpret and respond to human mental states. In this paper, we review LLMs' ToM capabilities by examining both evaluation benchmarks and the strategies designed to improve them. We focus on widely adopted story-based benchmarks and provide an in-depth analysis of methods aimed at enhancing ToM in LLMs. Furthermore, we outline promising future research directions informed by recent benchmarks and state-of-the-art approaches. Our survey serves as a valuable resource for researchers interested in advancing LLMs' ToM capabilities. 

---
# Nemotron-Research-Tool-N1: Tool-Using Language Models with Reinforced Reasoning 

**Authors**: Shaokun Zhang, Yi Dong, Jieyu Zhang, Jan Kautz, Bryan Catanzaro, Andrew Tao, Qingyun Wu, Zhiding Yu, Guilin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00024)  

**Abstract**: Enabling large language models with external tools has become a pivotal strategy for extending their functionality beyond text generation tasks. Prior work typically enhances tool-use abilities by either applying supervised fine-tuning (SFT) to enforce tool-call correctness or distilling reasoning traces from stronger models for SFT. However, both approaches fall short, either omitting reasoning entirely or producing imitative reasoning that limits generalization. Inspired by the success of DeepSeek-R1 in eliciting reasoning through rule-based reinforcement learning, we develop the Nemotron-Research-Tool-N1 series of tool-using language models using a similar training paradigm. Instead of restrictively supervising intermediate reasoning traces distilled from stronger models, Nemotron-Research-Tool-N1 is optimized with a binary reward that evaluates only the structural validity and functional correctness of tool invocations. This lightweight supervision allows the model to autonomously internalize reasoning strategies, without the need for annotated reasoning trajectories. Experiments on the BFCL and API-Bank benchmarks show that Nemotron-Research-Tool-N1-7B and Nemotron-Research-Tool-N1-14B, built on Qwen-2.5-7B/14B-Instruct, achieve state-of-the-art results, outperforming GPT-4o on both evaluations. 

---
# CoordField: Coordination Field for Agentic UAV Task Allocation In Low-altitude Urban Scenarios 

**Authors**: Tengchao Zhang, Yonglin Tian, Fei Lin, Jun Huang, Rui Qin, Fei-Yue Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00091)  

**Abstract**: With the increasing demand for heterogeneous Unmanned Aerial Vehicle (UAV) swarms to perform complex tasks in urban environments, system design now faces major challenges, including efficient semantic understanding, flexible task planning, and the ability to dynamically adjust coordination strategies in response to evolving environmental conditions and continuously changing task requirements. To address the limitations of existing approaches, this paper proposes coordination field agentic system for coordinating heterogeneous UAV swarms in complex urban scenarios. In this system, large language models (LLMs) is responsible for interpreting high-level human instructions and converting them into executable commands for the UAV swarms, such as patrol and target tracking. Subsequently, a Coordination field mechanism is proposed to guide UAV motion and task selection, enabling decentralized and adaptive allocation of emergent tasks. A total of 50 rounds of comparative testing were conducted across different models in a 2D simulation space to evaluate their performance. Experimental results demonstrate that the proposed system achieves superior performance in terms of task coverage, response time, and adaptability to dynamic changes. 

---
# ReCellTy: Domain-specific knowledge graph retrieval-augmented LLMs workflow for single-cell annotation 

**Authors**: Dezheng Han, Yibin Jia, Ruxiao Chen, Wenjie Han, Shuaishuai Guo, Jianbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00017)  

**Abstract**: To enable precise and fully automated cell type annotation with large language models (LLMs), we developed a graph structured feature marker database to retrieve entities linked to differential genes for cell reconstruction. We further designed a multi task workflow to optimize the annotation process. Compared to general purpose LLMs, our method improves human evaluation scores by up to 0.21 and semantic similarity by 6.1% across 11 tissue types, while more closely aligning with the cognitive logic of manual annotation. 

---
# An Empirical Study on Prompt Compression for Large Language Models 

**Authors**: Zheng Zhang, Jinyi Li, Yihuai Lan, Xiang Wang, Hao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00019)  

**Abstract**: Prompt engineering enables Large Language Models (LLMs) to perform a variety of tasks. However, lengthy prompts significantly increase computational complexity and economic costs. To address this issue, we study six prompt compression methods for LLMs, aiming to reduce prompt length while maintaining LLM response quality. In this paper, we present a comprehensive analysis covering aspects such as generation performance, model hallucinations, efficacy in multimodal tasks, word omission analysis, and more. We evaluate these methods across 13 datasets, including news, scientific articles, commonsense QA, math QA, long-context QA, and VQA datasets. Our experiments reveal that prompt compression has a greater impact on LLM performance in long contexts compared to short ones. In the Longbench evaluation, moderate compression even enhances LLM performance. Our code and data is available at this https URL. 

---
# Aleph-Alpha-GermanWeb: Improving German-language LLM pre-training with model-based data curation and synthetic data generation 

**Authors**: Thomas F Burns, Letitia Parcalabescu, Stephan Wäldchen, Michael Barlow, Gregor Ziegltrum, Volker Stampa, Bastian Harren, Björn Deiseroth  

**Link**: [PDF](https://arxiv.org/pdf/2505.00022)  

**Abstract**: Scaling data quantity is essential for large language models (LLMs), yet recent findings show that data quality can significantly boost performance and training efficiency. We introduce a German-language dataset curation pipeline that combines heuristic and model-based filtering techniques with synthetic data generation. We use our pipeline to create Aleph-Alpha-GermanWeb, a large-scale German pre-training dataset which draws from: (1) Common Crawl web data, (2) FineWeb2, and (3) synthetically-generated data conditioned on actual, organic web data. We evaluate our dataset by pre-training both a 1B Llama-style model and an 8B tokenizer-free hierarchical autoregressive transformer (HAT). A comparison on German-language benchmarks, including MMMLU, shows significant performance gains of Aleph-Alpha-GermanWeb over FineWeb2 alone. This advantage holds at the 8B scale even when FineWeb2 is enriched by human-curated high-quality data sources such as Wikipedia. Our findings support the growing body of evidence that model-based data curation and synthetic data generation can significantly enhance LLM pre-training datasets. 

---
# Sparks of Tabular Reasoning via Text2SQL Reinforcement Learning 

**Authors**: Josefa Lia Stoisser, Marc Boubnovski Martell, Julien Fauqueur  

**Link**: [PDF](https://arxiv.org/pdf/2505.00016)  

**Abstract**: This work reframes the Text-to-SQL task as a pathway for teaching large language models (LLMs) to reason over and manipulate tabular data--moving beyond the traditional focus on query generation. We propose a two-stage framework that leverages SQL supervision to develop transferable table reasoning capabilities. First, we synthesize detailed chain-of-thought (CoT) traces from real-world SQL queries, providing step-by-step, clause-level supervision that teaches the model how to traverse, filter, and aggregate table fields. Second, we introduce a Group Relative Policy Optimization (GRPO) reinforcement learning objective that connects SQL execution accuracy to generalizable reasoning by encouraging steps that extend beyond task-specific syntax and transfer across datasets. Empirically, our approach improves performance on standard Text-to-SQL benchmarks and achieves substantial gains on reasoning-intensive datasets such as BIRD and CRT-QA, demonstrating enhanced generalization and interpretability. Specifically, the distilled-quantized LLaMA model achieved a 20\% increase in accuracy when trained on Text-to-SQL tasks, while Qwen achieved a 5\% increase. These results suggest that SQL can serve not only as a target formalism but also as an effective scaffold for learning robust, transferable reasoning over structured data. 

---
# LangVAE and LangSpace: Building and Probing for Language Model VAEs 

**Authors**: Danilo S. Carvalho, Yingji Zhang, Harriet Unsworth, André Freitas  

**Link**: [PDF](https://arxiv.org/pdf/2505.00004)  

**Abstract**: We present LangVAE, a novel framework for modular construction of variational autoencoders (VAEs) on top of pre-trained large language models (LLMs). Such language model VAEs can encode the knowledge of their pre-trained components into more compact and semantically disentangled representations. The representations obtained in this way can be analysed with the LangVAE companion framework: LangSpace, which implements a collection of probing methods, such as vector traversal and interpolation, disentanglement measures, and cluster visualisations. LangVAE and LangSpace offer a flexible, efficient and scalable way of building and analysing textual representations, with simple integration for models available on the HuggingFace Hub. Additionally, we conducted a set of experiments with different encoder and decoder combinations, as well as annotated inputs, revealing a wide range of interactions across architectural families and sizes w.r.t. generalisation and disentanglement. Our findings demonstrate a promising framework for systematising the experimentation and understanding of textual representations. 

---
# Steering Large Language Models with Register Analysis for Arbitrary Style Transfer 

**Authors**: Xinchen Yang, Marine Carpuat  

**Link**: [PDF](https://arxiv.org/pdf/2505.00679)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in rewriting text across various styles. However, effectively leveraging this ability for example-based arbitrary style transfer, where an input text is rewritten to match the style of a given exemplar, remains an open challenge. A key question is how to describe the style of the exemplar to guide LLMs toward high-quality rewrites. In this work, we propose a prompting method based on register analysis to guide LLMs to perform this task. Empirical evaluations across multiple style transfer tasks show that our prompting approach enhances style transfer strength while preserving meaning more effectively than existing prompting strategies. 

---
# Rethinking Memory in AI: Taxonomy, Operations, Topics, and Future Directions 

**Authors**: Yiming Du, Wenyu Huang, Danna Zheng, Zhaowei Wang, Sebastien Montella, Mirella Lapata, Kam-Fai Wong, Jeff Z. Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00675)  

**Abstract**: Memory is a fundamental component of AI systems, underpinning large language models (LLMs) based agents. While prior surveys have focused on memory applications with LLMs, they often overlook the atomic operations that underlie memory dynamics. In this survey, we first categorize memory representations into parametric, contextual structured, and contextual unstructured and then introduce six fundamental memory operations: Consolidation, Updating, Indexing, Forgetting, Retrieval, and Compression. We systematically map these operations to the most relevant research topics across long-term, long-context, parametric modification, and multi-source memory. By reframing memory systems through the lens of atomic operations and representation types, this survey provides a structured and dynamic perspective on research, benchmark datasets, and tools related to memory in AI, clarifying the functional interplay in LLMs based agents while outlining promising directions for future research\footnote{The paper list, datasets, methods and tools are available at \href{this https URL}{this https URL\_Memory\_in\_AI}.}. 

---
# Block Circulant Adapter for Large Language Models 

**Authors**: Xinyu Ding, Meiqi Wang, Siyu Liao, Zhongfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.00582)  

**Abstract**: Fine-tuning large language models (LLMs) is difficult due to their huge model size. Recent Fourier domain-based methods show potential for reducing fine-tuning costs. We propose a block circulant matrix-based fine-tuning method with a stable training heuristic to leverage the properties of circulant matrices and one-dimensional Fourier transforms to reduce storage and computation costs. Experiments show that our method uses $14\times$ less number of parameters than VeRA, $16\times$ smaller than LoRA and $32\times$ less FLOPs than FourierFT, while maintaining close or better task performance. Our approach presents a promising way in frequency domain to fine-tune large models on downstream tasks. 

---
# 100 Days After DeepSeek-R1: A Survey on Replication Studies and More Directions for Reasoning Language Models 

**Authors**: Chong Zhang, Yue Deng, Xiang Lin, Bin Wang, Dianwen Ng, Hai Ye, Xingxuan Li, Yao Xiao, Zhanfeng Mo, Qi Zhang, Lidong Bing  

**Link**: [PDF](https://arxiv.org/pdf/2505.00551)  

**Abstract**: The recent development of reasoning language models (RLMs) represents a novel evolution in large language models. In particular, the recent release of DeepSeek-R1 has generated widespread social impact and sparked enthusiasm in the research community for exploring the explicit reasoning paradigm of language models. However, the implementation details of the released models have not been fully open-sourced by DeepSeek, including DeepSeek-R1-Zero, DeepSeek-R1, and the distilled small models. As a result, many replication studies have emerged aiming to reproduce the strong performance achieved by DeepSeek-R1, reaching comparable performance through similar training procedures and fully open-source data resources. These works have investigated feasible strategies for supervised fine-tuning (SFT) and reinforcement learning from verifiable rewards (RLVR), focusing on data preparation and method design, yielding various valuable insights. In this report, we provide a summary of recent replication studies to inspire future research. We primarily focus on SFT and RLVR as two main directions, introducing the details for data construction, method design and training procedure of current replication studies. Moreover, we conclude key findings from the implementation details and experimental results reported by these studies, anticipating to inspire future research. We also discuss additional techniques of enhancing RLMs, highlighting the potential of expanding the application scope of these models, and discussing the challenges in development. By this survey, we aim to help researchers and developers of RLMs stay updated with the latest advancements, and seek to inspire new ideas to further enhance RLMs. 

---
# AdaptMI: Adaptive Skill-based In-context Math Instruction for Small Language Models 

**Authors**: Yinghui He, Abhishek Panigrahi, Yong Lin, Sanjeev Arora  

**Link**: [PDF](https://arxiv.org/pdf/2505.00147)  

**Abstract**: In-context learning (ICL) allows a language model to improve its problem-solving capability when provided with suitable information in context. Since the choice of in-context information can be determined based on the problem itself, in-context learning is analogous to human learning from teachers in a classroom. Recent works (Didolkar et al., 2024a; 2024b) show that ICL performance can be improved by leveraging a frontier large language model's (LLM) ability to predict required skills to solve a problem, popularly referred to as an LLM's metacognition, and using the recommended skills to construct necessary in-context examples. While this skill-based strategy boosts ICL performance in larger models, its gains on small language models (SLMs) have been minimal, highlighting a performance gap in ICL capabilities. We investigate this gap and show that skill-based prompting can hurt SLM performance on easy questions by introducing unnecessary information, akin to cognitive overload. To address this, we introduce AdaptMI, an adaptive approach to selecting skill-based in-context Math Instructions for SLMs. Inspired by cognitive load theory from human pedagogy, our method only introduces skill-based examples when the model performs poorly. We further propose AdaptMI+, which adds examples targeted to the specific skills missing from the model's responses. On 5-shot evaluations across popular math benchmarks and five SLMs (1B--7B; Qwen, Llama), AdaptMI+ improves accuracy by up to 6% over naive skill-based strategies. 

---
# ConSens: Assessing context grounding in open-book question answering 

**Authors**: Ivan Vankov, Matyo Ivanov, Adriana Correia, Victor Botev  

**Link**: [PDF](https://arxiv.org/pdf/2505.00065)  

**Abstract**: Large Language Models (LLMs) have demonstrated considerable success in open-book question answering (QA), where the task requires generating answers grounded in a provided external context. A critical challenge in open-book QA is to ensure that model responses are based on the provided context rather than its parametric knowledge, which can be outdated, incomplete, or incorrect. Existing evaluation methods, primarily based on the LLM-as-a-judge approach, face significant limitations, including biases, scalability issues, and dependence on costly external systems. To address these challenges, we propose a novel metric that contrasts the perplexity of the model response under two conditions: when the context is provided and when it is not. The resulting score quantifies the extent to which the model's answer relies on the provided context. The validity of this metric is demonstrated through a series of experiments that show its effectiveness in identifying whether a given answer is grounded in the provided context. Unlike existing approaches, this metric is computationally efficient, interpretable, and adaptable to various use cases, offering a scalable and practical solution to assess context utilization in open-book QA systems. 

---
# A Report on the llms evaluating the high school questions 

**Authors**: Zhu Jiawei, Chen Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.00057)  

**Abstract**: This report aims to evaluate the performance of large language models (LLMs) in solving high school science questions and to explore their potential applications in the educational field. With the rapid development of LLMs in the field of natural language processing, their application in education has attracted widespread attention. This study selected mathematics exam questions from the college entrance examinations (2019-2023) as evaluation data and utilized at least eight LLM APIs to provide answers. A comprehensive assessment was conducted based on metrics such as accuracy, response time, logical reasoning, and creativity. Through an in-depth analysis of the evaluation results, this report reveals the strengths and weaknesses of LLMs in handling high school science questions and discusses their implications for educational practice. The findings indicate that although LLMs perform excellently in certain aspects, there is still room for improvement in logical reasoning and creative problem-solving. This report provides an empirical foundation for further research and application of LLMs in the educational field and offers suggestions for improvement. 

---
# HyPerAlign: Hypotheses-driven Personalized Alignment 

**Authors**: Cristina Garbacea, Chenhao Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.00038)  

**Abstract**: Alignment algorithms are widely used to align large language models (LLMs) to human users based on preference annotations that reflect their intended real-world use cases. Typically these (often divergent) preferences are aggregated over a diverse set of users, resulting in fine-tuned models that are aligned to the ``average-user'' preference. Nevertheless, current models are used by individual users in very specific contexts and situations, emphasizing the need for user-dependent preference control. In this work we address the problem of personalizing LLM outputs to their users, aiming to generate customized responses tailored to individual users, instead of generic outputs that emulate the collective voices of diverse populations. We propose a novel interpretable and sample-efficient hypotheses-driven personalization approach (HyPerAlign) where given few-shot examples written by a particular user, we first infer hypotheses about their communication strategies, personality and writing style, then prompt LLM models with these hypotheses and user specific attributes to generate customized outputs. We conduct experiments on two different personalization tasks, authorship attribution and deliberative alignment, with datasets from diverse domains (news articles, blog posts, emails, jailbreaking benchmarks), and demonstrate the superiority of hypotheses-driven personalization approach when compared to preference-based fine-tuning methods. For deliberative alignment, the helpfulness of LLM models is improved by up to $70\%$ on average. For authorship attribution, results indicate consistently high win-rates (commonly $>90\%$) against state-of-the-art preference fine-tuning approaches for LLM personalization across diverse user profiles and LLM models. Overall, our approach represents an interpretable and sample-efficient strategy for the personalization of LLM models to individual users. 

---
# Enhancing Security and Strengthening Defenses in Automated Short-Answer Grading Systems 

**Authors**: Sahar Yarmohammadtoosky, Yiyun Zhou, Victoria Yaneva, Peter Baldwin, Saed Rezayi, Brian Clauser, Polina Harikeo  

**Link**: [PDF](https://arxiv.org/pdf/2505.00061)  

**Abstract**: This study examines vulnerabilities in transformer-based automated short-answer grading systems used in medical education, with a focus on how these systems can be manipulated through adversarial gaming strategies. Our research identifies three main types of gaming strategies that exploit the system's weaknesses, potentially leading to false positives. To counteract these vulnerabilities, we implement several adversarial training methods designed to enhance the systems' robustness. Our results indicate that these methods significantly reduce the susceptibility of grading systems to such manipulations, especially when combined with ensemble techniques like majority voting and ridge regression, which further improve the system's defense against sophisticated adversarial inputs. Additionally, employing large language models such as GPT-4 with varied prompting techniques has shown promise in recognizing and scoring gaming strategies effectively. The findings underscore the importance of continuous improvements in AI-driven educational tools to ensure their reliability and fairness in high-stakes settings. 

---
# Design and Application of Multimodal Large Language Model Based System for End to End Automation of Accident Dataset Generation 

**Authors**: MD Thamed Bin Zaman Chowdhury, Moazzem Hossain  

**Link**: [PDF](https://arxiv.org/pdf/2505.00015)  

**Abstract**: Road traffic accidents remain a major public safety and socio-economic issue in developing countries like Bangladesh. Existing accident data collection is largely manual, fragmented, and unreliable, resulting in underreporting and inconsistent records. This research proposes a fully automated system using Large Language Models (LLMs) and web scraping techniques to address these challenges. The pipeline consists of four components: automated web scraping code generation, news collection from online sources, accident news classification with structured data extraction, and duplicate removal. The system uses the multimodal generative LLM Gemini-2.0-Flash for seamless automation. The code generation module classifies webpages into pagination, dynamic, or infinite scrolling categories and generates suitable Python scripts for scraping. LLMs also classify and extract key accident information such as date, time, location, fatalities, injuries, road type, vehicle types, and pedestrian involvement. A deduplication algorithm ensures data integrity by removing duplicate reports. The system scraped 14 major Bangladeshi news sites over 111 days (Oct 1, 2024 - Jan 20, 2025), processing over 15,000 news articles and identifying 705 unique accidents. The code generation module achieved 91.3% calibration and 80% validation accuracy. Chittagong reported the highest number of accidents (80), fatalities (70), and injuries (115), followed by Dhaka, Faridpur, Gazipur, and Cox's Bazar. Peak accident times were morning (8-9 AM), noon (12-1 PM), and evening (6-7 PM). A public repository was also developed with usage instructions. This study demonstrates the viability of an LLM-powered, scalable system for accurate, low-effort accident data collection, providing a foundation for data-driven road safety policymaking in Bangladesh. 

---
# The Mind in the Machine: A Survey of Incorporating Psychological Theories in LLMs 

**Authors**: Zizhou Liu, Ziwei Gong, Lin Ai, Zheng Hui, Run Chen, Colin Wayne Leach, Michelle R. Greene, Julia Hirschberg  

**Link**: [PDF](https://arxiv.org/pdf/2505.00003)  

**Abstract**: Psychological insights have long shaped pivotal NLP breakthroughs, including the cognitive underpinnings of attention mechanisms, formative reinforcement learning, and Theory of Mind-inspired social modeling. As Large Language Models (LLMs) continue to grow in scale and complexity, there is a rising consensus that psychology is essential for capturing human-like cognition, behavior, and interaction. This paper reviews how psychological theories can inform and enhance stages of LLM development, including data, pre-training, post-training, and evaluation\&application. Our survey integrates insights from cognitive, developmental, behavioral, social, personality psychology, and psycholinguistics. Our analysis highlights current trends and gaps in how psychological theories are applied. By examining both cross-domain connections and points of tension, we aim to bridge disciplinary divides and promote more thoughtful integration of psychology into future NLP research. 

---
# Rosetta-PL: Propositional Logic as a Benchmark for Large Language Model Reasoning 

**Authors**: Shaun Baek, Shaun Esua-Mensah, Cyrus Tsui, Sejan Vigneswaralingam, Abdullah Alali, Michael Lu, Vasu Sharma, Kevin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.00001)  

**Abstract**: Large Language Models (LLMs) are primarily trained on high-resource natural languages, limiting their effectiveness in low-resource settings and in tasks requiring deep logical reasoning. This research introduces Rosetta-PL, a benchmark designed to evaluate LLMs' logical reasoning and generalization capabilities in a controlled environment. We construct Rosetta-PL by translating a dataset of logical propositions from Lean into a custom logical language, which is then used to fine-tune an LLM (e.g., GPT-4o). Our experiments analyze the impact of the size of the dataset and the translation methodology on the performance of the model. Our results indicate that preserving logical relationships in the translation process significantly boosts precision, with accuracy plateauing beyond roughly 20,000 training samples. These insights provide valuable guidelines for optimizing LLM training in formal reasoning tasks and improving performance in various low-resource language applications. 

---
# Jailbreak Detection in Clinical Training LLMs Using Feature-Based Predictive Models 

**Authors**: Tri Nguyen, Lohith Srikanth Pentapalli, Magnus Sieverding, Laurah Turner, Seth Overla, Weibing Zheng, Chris Zhou, David Furniss, Danielle Weber, Michael Gharib, Matt Kelleher, Michael Shukis, Cameron Pawlik, Kelly Cohen  

**Link**: [PDF](https://arxiv.org/pdf/2505.00010)  

**Abstract**: Jailbreaking in Large Language Models (LLMs) threatens their safe use in sensitive domains like education by allowing users to bypass ethical safeguards. This study focuses on detecting jailbreaks in 2-Sigma, a clinical education platform that simulates patient interactions using LLMs. We annotated over 2,300 prompts across 158 conversations using four linguistic variables shown to correlate strongly with jailbreak behavior. The extracted features were used to train several predictive models, including Decision Trees, Fuzzy Logic-based classifiers, Boosting methods, and Logistic Regression. Results show that feature-based predictive models consistently outperformed Prompt Engineering, with the Fuzzy Decision Tree achieving the best overall performance. Our findings demonstrate that linguistic-feature-based models are effective and explainable alternatives for jailbreak detection. We suggest future work explore hybrid frameworks that integrate prompt-based flexibility with rule-based robustness for real-time, spectrum-based jailbreak monitoring in educational LLMs. 

---
# Self-Generated In-Context Examples Improve LLM Agents for Sequential Decision-Making Tasks 

**Authors**: Vishnu Sarukkai, Zhiqiang Xie, Kayvon Fatahalian  

**Link**: [PDF](https://arxiv.org/pdf/2505.00234)  

**Abstract**: Many methods for improving Large Language Model (LLM) agents for sequential decision-making tasks depend on task-specific knowledge engineering--such as prompt tuning, curated in-context examples, or customized observation and action spaces. Using these approaches, agent performance improves with the quality or amount of knowledge engineering invested. Instead, we investigate how LLM agents can automatically improve their performance by learning in-context from their own successful experiences on similar tasks. Rather than relying on task-specific knowledge engineering, we focus on constructing and refining a database of self-generated examples. We demonstrate that even a naive accumulation of successful trajectories across training tasks boosts test performance on three benchmarks: ALFWorld (73% to 89%), Wordcraft (55% to 64%), and InterCode-SQL (75% to 79%)--matching the performance the initial agent achieves if allowed two to three attempts per task. We then introduce two extensions: (1) database-level selection through population-based training to identify high-performing example collections, and (2) exemplar-level selection that retains individual trajectories based on their empirical utility as in-context examples. These extensions further enhance performance, achieving 91% on ALFWorld--matching more complex approaches that employ task-specific components and prompts. Our results demonstrate that automatic trajectory database construction offers a compelling alternative to labor-intensive knowledge engineering. 

---
# Humanizing LLMs: A Survey of Psychological Measurements with Tools, Datasets, and Human-Agent Applications 

**Authors**: Wenhan Dong, Yuemeng Zhao, Zhen Sun, Yule Liu, Zifan Peng, Jingyi Zheng, Zongmin Zhang, Ziyi Zhang, Jun Wu, Ruiming Wang, Shengmin Xu, Xinyi Huang, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2505.00049)  

**Abstract**: As large language models (LLMs) are increasingly used in human-centered tasks, assessing their psychological traits is crucial for understanding their social impact and ensuring trustworthy AI alignment. While existing reviews have covered some aspects of related research, several important areas have not been systematically discussed, including detailed discussions of diverse psychological tests, LLM-specific psychological datasets, and the applications of LLMs with psychological traits. To address this gap, we systematically review six key dimensions of applying psychological theories to LLMs: (1) assessment tools; (2) LLM-specific datasets; (3) evaluation metrics (consistency and stability); (4) empirical findings; (5) personality simulation methods; and (6) LLM-based behavior simulation. Our analysis highlights both the strengths and limitations of current methods. While some LLMs exhibit reproducible personality patterns under specific prompting schemes, significant variability remains across tasks and settings. Recognizing methodological challenges such as mismatches between psychological tools and LLMs' capabilities, as well as inconsistencies in evaluation practices, this study aims to propose future directions for developing more interpretable, robust, and generalizable psychological assessment frameworks for LLMs. 

---
