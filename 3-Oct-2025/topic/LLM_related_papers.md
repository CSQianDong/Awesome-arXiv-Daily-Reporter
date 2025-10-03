# Study on LLMs for Promptagator-Style Dense Retriever Training 

**Authors**: Daniel Gwon, Nour Jedidi, Jimmy Lin  

**Link**: [PDF](https://arxiv.org/pdf/2510.02241)  

**Abstract**: Promptagator demonstrated that Large Language Models (LLMs) with few-shot prompts can be used as task-specific query generators for fine-tuning domain-specialized dense retrieval models. However, the original Promptagator approach relied on proprietary and large-scale LLMs which users may not have access to or may be prohibited from using with sensitive data. In this work, we study the impact of open-source LLMs at accessible scales ($\leq$14B parameters) as an alternative. Our results demonstrate that open-source LLMs as small as 3B parameters can serve as effective Promptagator-style query generators. We hope our work will inform practitioners with reliable alternatives for synthetic data generation and give insights to maximize fine-tuning results for domain-specific applications. 

---
# TalkPlay-Tools: Conversational Music Recommendation with LLM Tool Calling 

**Authors**: Seungheon Doh, Keunwoo Choi, Juhan Nam  

**Link**: [PDF](https://arxiv.org/pdf/2510.01698)  

**Abstract**: While the recent developments in large language models (LLMs) have successfully enabled generative recommenders with natural language interactions, their recommendation behavior is limited, leaving other simpler yet crucial components such as metadata or attribute filtering underutilized in the system. We propose an LLM-based music recommendation system with tool calling to serve as a unified retrieval-reranking pipeline. Our system positions an LLM as an end-to-end recommendation system that interprets user intent, plans tool invocations, and orchestrates specialized components: boolean filters (SQL), sparse retrieval (BM25), dense retrieval (embedding similarity), and generative retrieval (semantic IDs). Through tool planning, the system predicts which types of tools to use, their execution order, and the arguments needed to find music matching user preferences, supporting diverse modalities while seamlessly integrating multiple database filtering methods. We demonstrate that this unified tool-calling framework achieves competitive performance across diverse recommendation scenarios by selectively employing appropriate retrieval methods based on user queries, envisioning a new paradigm for conversational music recommendation systems. 

---
# Are LLMs ready to help non-expert users to make charts of official statistics data? 

**Authors**: Gadir Suleymanli, Alexander Rogiers, Lucas Lageweg, Jefrey Lijffijt  

**Link**: [PDF](https://arxiv.org/pdf/2510.01197)  

**Abstract**: In this time when biased information, deep fakes, and propaganda proliferate, the accessibility of reliable data sources is more important than ever. National statistical institutes provide curated data that contain quantitative information on a wide range of topics. However, that information is typically spread across many tables and the plain numbers may be arduous to process. Hence, this open data may be practically inaccessible. We ask the question "Are current Generative AI models capable of facilitating the identification of the right data and the fully-automatic creation of charts to provide information in visual form, corresponding to user queries?". We present a structured evaluation of recent large language models' (LLMs) capabilities to generate charts from complex data in response to user queries. Working with diverse public data from Statistics Netherlands, we assessed multiple LLMs on their ability to identify relevant data tables, perform necessary manipulations, and generate appropriate visualizations autonomously. We propose a new evaluation framework spanning three dimensions: data retrieval & pre-processing, code quality, and visual representation. Results indicate that locating and processing the correct data represents the most significant challenge. Additionally, LLMs rarely implement visualization best practices without explicit guidance. When supplemented with information about effective chart design, models showed marked improvement in representation scores. Furthermore, an agentic approach with iterative self-evaluation led to excellent performance across all evaluation dimensions. These findings suggest that LLMs' effectiveness for automated chart generation can be enhanced through appropriate scaffolding and feedback mechanisms, and that systems can already reach the necessary accuracy across the three evaluation dimensions. 

---
# Bridging Collaborative Filtering and Large Language Models with Dynamic Alignment, Multimodal Fusion and Evidence-grounded Explanations 

**Authors**: Bo Ma, LuYao Liu, Simon Lau, Chandler Yuan, and XueY Cui, Rosie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01606)  

**Abstract**: Recent research has explored using Large Language Models for recommendation tasks by transforming user interaction histories and item metadata into text prompts, then having the LLM produce rankings or recommendations. A promising approach involves connecting collaborative filtering knowledge to LLM representations through compact adapter networks, which avoids expensive fine-tuning while preserving the strengths of both components. Yet several challenges persist in practice: collaborative filtering models often use static snapshots that miss rapidly changing user preferences; many real-world items contain rich visual and audio content beyond textual descriptions; and current systems struggle to provide trustworthy explanations backed by concrete evidence. Our work introduces \model{}, a framework that tackles these limitations through three key innovations. We develop an online adaptation mechanism that continuously incorporates new user interactions through lightweight modules, avoiding the need to retrain large models. We create a unified representation that seamlessly combines collaborative signals with visual and audio features, handling cases where some modalities may be unavailable. Finally, we design an explanation system that grounds recommendations in specific collaborative patterns and item attributes, producing natural language rationales users can verify. Our approach maintains the efficiency of frozen base models while adding minimal computational overhead, making it practical for real-world deployment. 

---
# LLM4Rec: Large Language Models for Multimodal Generative Recommendation with Causal Debiasing 

**Authors**: Bo Ma, Hang Li, ZeHua Hu, XiaoFan Gui, LuYao Liu, Simon Lau  

**Link**: [PDF](https://arxiv.org/pdf/2510.01622)  

**Abstract**: Contemporary generative recommendation systems face significant challenges in handling multimodal data, eliminating algorithmic biases, and providing transparent decision-making processes. This paper introduces an enhanced generative recommendation framework that addresses these limitations through five key innovations: multimodal fusion architecture, retrieval-augmented generation mechanisms, causal inference-based debiasing, explainable recommendation generation, and real-time adaptive learning capabilities. Our framework leverages advanced large language models as the backbone while incorporating specialized modules for cross-modal understanding, contextual knowledge integration, bias mitigation, explanation synthesis, and continuous model adaptation. Extensive experiments on three benchmark datasets (MovieLens-25M, Amazon-Electronics, Yelp-2023) demonstrate consistent improvements in recommendation accuracy, fairness, and diversity compared to existing approaches. The proposed framework achieves up to 2.3% improvement in NDCG@10 and 1.4% enhancement in diversity metrics while maintaining computational efficiency through optimized inference strategies. 

---
# LLM-based Multi-Agent Blackboard System for Information Discovery in Data Science 

**Authors**: Alireza Salemi, Mihir Parmar, Palash Goyal, Yiwen Song, Jinsung Yoon, Hamed Zamani, Hamid Palangi, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2510.01285)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has opened new opportunities in data science, yet their practical deployment is often constrained by the challenge of discovering relevant data within large heterogeneous data lakes. Existing methods struggle with this: single-agent systems are quickly overwhelmed by large, heterogeneous files in the large data lakes, while multi-agent systems designed based on a master-slave paradigm depend on a rigid central controller for task allocation that requires precise knowledge of each sub-agent's capabilities. To address these limitations, we propose a novel multi-agent communication paradigm inspired by the blackboard architecture for traditional AI models. In this framework, a central agent posts requests to a shared blackboard, and autonomous subordinate agents -- either responsible for a partition of the data lake or general information retrieval -- volunteer to respond based on their capabilities. This design improves scalability and flexibility by eliminating the need for a central coordinator to have prior knowledge of all sub-agents' expertise. We evaluate our method on three benchmarks that require explicit data discovery: KramaBench and modified versions of DS-Bench and DA-Code to incorporate data discovery. Experimental results demonstrate that the blackboard architecture substantially outperforms baselines, including RAG and the master-slave multi-agent paradigm, achieving between 13% to 57% relative improvement in end-to-end task success and up to a 9% relative gain in F1 score for data discovery over the best-performing baselines across both proprietary and open-source LLMs. Our findings establish the blackboard paradigm as a scalable and generalizable communication framework for multi-agent systems. 

---
# More Than One Teacher: Adaptive Multi-Guidance Policy Optimization for Diverse Exploration 

**Authors**: Xiaoyang Yuan, Yujuan Ding, Yi Bin, Wenqi Shao, Jinyu Cai, Jingkuan Song, Yang Yang, Hengtao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.02227)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) is a promising paradigm for enhancing the reasoning ability in Large Language Models (LLMs). However, prevailing methods primarily rely on self-exploration or a single off-policy teacher to elicit long chain-of-thought (LongCoT) reasoning, which may introduce intrinsic model biases and restrict exploration, ultimately limiting reasoning diversity and performance. Drawing inspiration from multi-teacher strategies in knowledge distillation, we introduce Adaptive Multi-Guidance Policy Optimization (AMPO), a novel framework that adaptively leverages guidance from multiple proficient teacher models, but only when the on-policy model fails to generate correct solutions. This "guidance-on-demand" approach expands exploration while preserving the value of self-discovery. Moreover, AMPO incorporates a comprehension-based selection mechanism, prompting the student to learn from the reasoning paths that it is most likely to comprehend, thus balancing broad exploration with effective exploitation. Extensive experiments show AMPO substantially outperforms a strong baseline (GRPO), with a 4.3% improvement on mathematical reasoning tasks and 12.2% on out-of-distribution tasks, while significantly boosting Pass@k performance and enabling more diverse exploration. Notably, using four peer-sized teachers, our method achieves comparable results to approaches that leverage a single, more powerful teacher (e.g., DeepSeek-R1) with more data. These results demonstrate a more efficient and scalable path to superior reasoning and generalizability. Our code is available at this https URL. 

---
# Explore Briefly, Then Decide: Mitigating LLM Overthinking via Cumulative Entropy Regulation 

**Authors**: Tianyi Jiang, Yi Bin, Yujuan Ding, Kainian Zhu, Fei Ma, Jingkuan Song, Heng Tao Shen  

**Link**: [PDF](https://arxiv.org/pdf/2510.02249)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable reasoning abilities on complex problems using long Chain-of-Thought (CoT) reasoning. However, they often suffer from overthinking, meaning generating unnecessarily lengthy reasoning steps for simpler problems. This issue may degrade the efficiency of the models and make them difficult to adapt the reasoning depth to the complexity of problems. To address this, we introduce a novel metric Token Entropy Cumulative Average (TECA), which measures the extent of exploration throughout the reasoning process. We further propose a novel reasoning paradigm -- Explore Briefly, Then Decide -- with an associated Cumulative Entropy Regulation (CER) mechanism. This paradigm leverages TECA to help the model dynamically determine the optimal point to conclude its thought process and provide a final answer, thus achieving efficient reasoning. Experimental results across diverse mathematical benchmarks show that our approach substantially mitigates overthinking without sacrificing problem-solving ability. With our thinking paradigm, the average response length decreases by up to 71% on simpler datasets, demonstrating the effectiveness of our method in creating a more efficient and adaptive reasoning process. 

---
# ARUQULA -- An LLM based Text2SPARQL Approach using ReAct and Knowledge Graph Exploration Utilities 

**Authors**: Felix Brei, Lorenz Bühmann, Johannes Frey, Daniel Gerber, Lars-Peter Meyer, Claus Stadler, Kirill Bulert  

**Link**: [PDF](https://arxiv.org/pdf/2510.02200)  

**Abstract**: Interacting with knowledge graphs can be a daunting task for people without a background in computer science since the query language that is used (SPARQL) has a high barrier of entry. Large language models (LLMs) can lower that barrier by providing support in the form of Text2SPARQL translation. In this paper we introduce a generalized method based on SPINACH, an LLM backed agent that translates natural language questions to SPARQL queries not in a single shot, but as an iterative process of exploration and execution. We describe the overall architecture and reasoning behind our design decisions, and also conduct a thorough analysis of the agent behavior to gain insights into future areas for targeted improvements. This work was motivated by the Text2SPARQL challenge, a challenge that was held to facilitate improvements in the Text2SPARQL domain. 

---
# Enhancing Large Language Model Reasoning with Reward Models: An Analytical Survey 

**Authors**: Qiyuan Liu, Hao Xu, Xuhong Chen, Wei Chen, Yee Whye Teh, Ning Miao  

**Link**: [PDF](https://arxiv.org/pdf/2510.01925)  

**Abstract**: Reward models (RMs) play a critical role in enhancing the reasoning performance of LLMs. For example, they can provide training signals to finetune LLMs during reinforcement learning (RL) and help select the best answer from multiple candidates during inference. In this paper, we provide a systematic introduction to RMs, along with a comprehensive survey of their applications in LLM reasoning. We first review fundamental concepts of RMs, including their architectures, training methodologies, and evaluation techniques. Then, we explore their key applications: (1) guiding generation and selecting optimal outputs during LLM inference, (2) facilitating data synthesis and iterative self-improvement for LLMs, and (3) providing training signals in RL-based finetuning. Finally, we address critical open questions regarding the selection, generalization, evaluation, and enhancement of RMs, based on existing research and our own empirical findings. Our analysis aims to provide actionable insights for the effective deployment and advancement of RMs for LLM reasoning. 

---
# REPAIR: Robust Editing via Progressive Adaptive Intervention and Reintegration 

**Authors**: Yisu Wang, Ming Wang, Haoyuan Song, Wenjie Huang, Chaozheng Wang, Yi Xie, Xuming Ran  

**Link**: [PDF](https://arxiv.org/pdf/2510.01879)  

**Abstract**: Post-training for large language models (LLMs) is constrained by the high cost of acquiring new knowledge or correcting errors and by the unintended side effects that frequently arise from retraining. To address these issues, we introduce REPAIR (Robust Editing via Progressive Adaptive Intervention and Reintegration), a lifelong editing framework designed to support precise and low-cost model updates while preserving non-target knowledge. REPAIR mitigates the instability and conflicts of large-scale sequential edits through a closed-loop feedback mechanism coupled with dynamic memory management. Furthermore, by incorporating frequent knowledge fusion and enforcing strong locality guards, REPAIR effectively addresses the shortcomings of traditional distribution-agnostic approaches that often overlook unintended ripple effects. Our experiments demonstrate that REPAIR boosts editing accuracy by 10%-30% across multiple model families and significantly reduces knowledge forgetting. This work introduces a robust framework for developing reliable, scalable, and continually evolving LLMs. 

---
# Inverse Language Modeling towards Robust and Grounded LLMs 

**Authors**: Davide Gabrielli, Simone Sestito, Iacopo Masi  

**Link**: [PDF](https://arxiv.org/pdf/2510.01929)  

**Abstract**: The current landscape of defensive mechanisms for LLMs is fragmented and underdeveloped, unlike prior work on classifiers. To further promote adversarial robustness in LLMs, we propose Inverse Language Modeling (ILM), a unified framework that simultaneously 1) improves the robustness of LLMs to input perturbations, and, at the same time, 2) enables native grounding by inverting model outputs to identify potentially toxic or unsafe input triggers. ILM transforms LLMs from static generators into analyzable and robust systems, potentially helping RED teaming. ILM can lay the foundation for next-generation LLMs that are not only robust and grounded but also fundamentally more controllable and trustworthy. The code is publicly available at this http URL. 

---
# Learning to Reason for Hallucination Span Detection 

**Authors**: Hsuan Su, Ting-Yao Hu, Hema Swetha Koppula, Kundan Krishna, Hadi Pouransari, Cheng-Yu Hsieh, Cem Koc, Joseph Yitan Cheng, Oncel Tuzel, Raviteja Vemulapalli  

**Link**: [PDF](https://arxiv.org/pdf/2510.02173)  

**Abstract**: Large language models (LLMs) often generate hallucinations -- unsupported content that undermines reliability. While most prior works frame hallucination detection as a binary task, many real-world applications require identifying hallucinated spans, which is a multi-step decision making process. This naturally raises the question of whether explicit reasoning can help the complex task of detecting hallucination spans. To answer this question, we first evaluate pretrained models with and without Chain-of-Thought (CoT) reasoning, and show that CoT reasoning has the potential to generate at least one correct answer when sampled multiple times. Motivated by this, we propose RL4HS, a reinforcement learning framework that incentivizes reasoning with a span-level reward function. RL4HS builds on Group Relative Policy Optimization and introduces Class-Aware Policy Optimization to mitigate reward imbalance issue. Experiments on the RAGTruth benchmark (summarization, question answering, data-to-text) show that RL4HS surpasses pretrained reasoning models and supervised fine-tuning, demonstrating the necessity of reinforcement learning with span-level rewards for detecting hallucination spans. 

---
# Can LLMs Refuse Questions They Do Not Know? Measuring Knowledge-Aware Refusal in Factual Tasks 

**Authors**: Wenbo Pan, Jie Xu, Qiguang Chen, Junhao Dong, Libo Qin, Xinfeng Li, Haining Yu, Xiaohua Jia  

**Link**: [PDF](https://arxiv.org/pdf/2510.01782)  

**Abstract**: Large Language Models (LLMs) should refuse to answer questions beyond their knowledge. This capability, which we term knowledge-aware refusal, is crucial for factual reliability. However, existing metrics fail to faithfully measure this ability. On the one hand, simple refusal-based metrics are biased by refusal rates and yield inconsistent scores when models exhibit different refusal tendencies. On the other hand, existing calibration metrics are proxy-based, capturing the performance of auxiliary calibration processes rather than the model's actual refusal behavior. In this work, we propose the Refusal Index (RI), a principled metric that measures how accurately LLMs refuse questions they do not know. We define RI as Spearman's rank correlation between refusal probability and error probability. To make RI practically measurable, we design a lightweight two-pass evaluation method that efficiently estimates RI from observed refusal rates across two standard evaluation runs. Extensive experiments across 16 models and 5 datasets demonstrate that RI accurately quantifies a model's intrinsic knowledge-aware refusal capability in factual tasks. Notably, RI remains stable across different refusal rates and provides consistent model rankings independent of a model's overall accuracy and refusal rates. More importantly, RI provides insight into an important but previously overlooked aspect of LLM factuality: while LLMs achieve high accuracy on factual tasks, their refusal behavior can be unreliable and fragile. This finding highlights the need to complement traditional accuracy metrics with the Refusal Index for comprehensive factuality evaluation. 

---
# LLM-Based Multi-Task Bangla Hate Speech Detection: Type, Severity, and Target 

**Authors**: Md Arid Hasan, Firoj Alam, Md Fahad Hossain, Usman Naseem, Syed Ishtiaque Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2510.01995)  

**Abstract**: Online social media platforms are central to everyday communication and information seeking. While these platforms serve positive purposes, they also provide fertile ground for the spread of hate speech, offensive language, and bullying content targeting individuals, organizations, and communities. Such content undermines safety, participation, and equity online. Reliable detection systems are therefore needed, especially for low-resource languages where moderation tools are limited. In Bangla, prior work has contributed resources and models, but most are single-task (e.g., binary hate/offense) with limited coverage of multi-facet signals (type, severity, target). We address these gaps by introducing the first multi-task Bangla hate-speech dataset, BanglaMultiHate, one of the largest manually annotated corpus to date. Building on this resource, we conduct a comprehensive, controlled comparison spanning classical baselines, monolingual pretrained models, and LLMs under zero-shot prompting and LoRA fine-tuning. Our experiments assess LLM adaptability in a low-resource setting and reveal a consistent trend: although LoRA-tuned LLMs are competitive with BanglaBERT, culturally and linguistically grounded pretraining remains critical for robust performance. Together, our dataset and findings establish a stronger benchmark for developing culturally aligned moderation tools in low-resource contexts. For reproducibility, we will release the dataset and all related scripts. 

---
# What MLLMs Learn about When they Learn about Multimodal Reasoning: Perception, Reasoning, or their Integration? 

**Authors**: Jiwan Chung, Neel Joshi, Pratyusha Sharma, Youngjae Yu, Vibhav Vineet  

**Link**: [PDF](https://arxiv.org/pdf/2510.01719)  

**Abstract**: Multimodal reasoning models have recently shown promise on challenging domains such as olympiad-level geometry, yet their evaluation remains dominated by aggregate accuracy, a single score that obscures where and how models are improving. We introduce MathLens, a benchmark designed to disentangle the subskills of multimodal reasoning while preserving the complexity of textbook-style geometry problems. The benchmark separates performance into three components: Perception: extracting information from raw inputs, Reasoning: operating on available information, and Integration: selecting relevant perceptual evidence and applying it within reasoning. To support each test, we provide annotations: visual diagrams, textual descriptions to evaluate reasoning in isolation, controlled questions that require both modalities, and probes for fine-grained perceptual skills, all derived from symbolic specifications of the problems to ensure consistency and robustness. Our analysis reveals that different training approaches have uneven effects: First, reinforcement learning chiefly strengthens perception, especially when supported by textual supervision, while textual SFT indirectly improves perception through reflective reasoning. Second, reasoning improves only in tandem with perception. Third, integration remains the weakest capacity, with residual errors concentrated there once other skills advance. Finally, robustness diverges: RL improves consistency under diagram variation, whereas multimodal SFT reduces it through overfitting. We will release all data and experimental logs. 

---
# Syntactic Blind Spots: How Misalignment Leads to LLMs Mathematical Errors 

**Authors**: Dane Williamson, Yangfeng Ji, Matthew Dwyer  

**Link**: [PDF](https://arxiv.org/pdf/2510.01831)  

**Abstract**: Large Language Models (LLMs) demonstrate strong mathematical problem-solving abilities but frequently fail on problems that deviate syntactically from their training distribution. We identify a systematic failure mode, syntactic blind spots, in which models misapply familiar reasoning strategies to problems that are semantically straightforward but phrased in unfamiliar ways. These errors are not due to gaps in mathematical competence, but rather reflect a brittle coupling between surface form and internal representation. To test this, we rephrase incorrectly answered questions using syntactic templates drawn from correct examples. These rephrasings, which preserve semantics while reducing structural complexity, often lead to correct answers. We quantify syntactic complexity using a metric based on Dependency Locality Theory (DLT), and show that higher DLT scores are associated with increased failure rates across multiple datasets. Our findings suggest that many reasoning errors stem from structural misalignment rather than conceptual difficulty, and that syntax-aware interventions can reveal and mitigate these inductive failures. 

---
# How Do Language Models Compose Functions? 

**Authors**: Apoorv Khandelwal, Ellie Pavlick  

**Link**: [PDF](https://arxiv.org/pdf/2510.01685)  

**Abstract**: While large language models (LLMs) appear to be increasingly capable of solving compositional tasks, it is an open question whether they do so using compositional mechanisms. In this work, we investigate how feedforward LLMs solve two-hop factual recall tasks, which can be expressed compositionally as $g(f(x))$. We first confirm that modern LLMs continue to suffer from the "compositionality gap": i.e. their ability to compute both $z = f(x)$ and $y = g(z)$ does not entail their ability to compute the composition $y = g(f(x))$. Then, using logit lens on their residual stream activations, we identify two processing mechanisms, one which solves tasks $\textit{compositionally}$, computing $f(x)$ along the way to computing $g(f(x))$, and one which solves them $\textit{directly}$, without any detectable signature of the intermediate variable $f(x)$. Finally, we find that which mechanism is employed appears to be related to the embedding space geometry, with the idiomatic mechanism being dominant in cases where there exists a linear mapping from $x$ to $g(f(x))$ in the embedding spaces. We fully release our data and code at: this https URL . 

---
# Veri-R1: Toward Precise and Faithful Claim Verification via Online Reinforcement Learning 

**Authors**: Qi He, Cheng Qian, Xiusi Chen, Bingxiang He, Yi R., Fung, Heng Ji  

**Link**: [PDF](https://arxiv.org/pdf/2510.01932)  

**Abstract**: Claim verification with large language models (LLMs) has recently attracted considerable attention, owing to their superior reasoning capabilities and transparent verification pathways compared to traditional answer-only judgments. Online claim verification requires iterative evidence retrieval and reasoning, yet existing approaches mainly rely on prompt engineering or predesigned reasoning workflows without offering a unified training paradigm to improve necessary skills. Therefore, we introduce Veri-R1, an online reinforcement learning (RL) framework that enables an LLM to interact with a search engine and to receive reward signals that explicitly shape its planning, retrieval, and reasoning behaviors. The dynamic interaction between models and retrieval systems more accurately reflects real-world verification scenarios and fosters comprehensive verification skills. Empirical results show that Veri-R1 improves joint accuracy by up to 30% and doubles evidence score, often surpassing larger-scale counterparts. Ablation studies further reveal the impact of reward components and the link between output logits and label accuracy. Our results highlight the effectiveness of online RL for precise and faithful claim verification and provide a foundation for future research. We release our code to support community progress in LLM empowered claim verification. 

---
# AMAS: Adaptively Determining Communication Topology for LLM-based Multi-Agent System 

**Authors**: Hui Yi Leong, Yuheng Li, Yuqing Wu, Wenwen Ouyang, Wei Zhu, Jiechao Gao  

**Link**: [PDF](https://arxiv.org/pdf/2510.01617)  

**Abstract**: Although large language models (LLMs) have revolutionized natural language processing capabilities, their practical implementation as autonomous multi-agent systems (MAS) for industrial problem-solving encounters persistent barriers. Conventional MAS architectures are fundamentally restricted by inflexible, hand-crafted graph topologies that lack contextual responsiveness, resulting in diminished efficacy across varied academic and commercial workloads. To surmount these constraints, we introduce AMAS, a paradigm-shifting framework that redefines LLM-based MAS through a novel dynamic graph designer. This component autonomously identifies task-specific optimal graph configurations via lightweight LLM adaptation, eliminating the reliance on monolithic, universally applied structural templates. Instead, AMAS exploits the intrinsic properties of individual inputs to intelligently direct query trajectories through task-optimized agent pathways. Rigorous validation across question answering, mathematical deduction, and code generation benchmarks confirms that AMAS systematically exceeds state-of-the-art single-agent and multi-agent approaches across diverse LLM architectures. Our investigation establishes that context-sensitive structural adaptability constitutes a foundational requirement for high-performance LLM MAS deployments. 

---
# Format Inertia: A Failure Mechanism of LLMs in Medical Pre-Consultation 

**Authors**: Seungseop Lim, Gibaeg Kim, Wooseok Han, Jean Seo, Hyunkyung Lee, Jaehyo Yoo, Eunho Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01688)  

**Abstract**: Recent advances in Large Language Models (LLMs) have brought significant improvements to various service domains, including chatbots and medical pre-consultation applications. In the healthcare domain, the most common approach for adapting LLMs to multi-turn dialogue generation is Supervised Fine-Tuning (SFT). However, datasets for SFT in tasks like medical pre-consultation typically exhibit a skewed turn-count distribution. Training on such data induces a novel failure mechanism we term **Format Inertia**, where models tend to generate repetitive, format-correct, but diagnostically uninformative questions in long medical dialogues. To mitigate this observed failure mechanism, we adopt a simple, data-centric method that rebalances the turn-count distribution of the training dataset. Experimental results show that our approach substantially alleviates Format Inertia in medical pre-consultation. 

---
# CLUE: Non-parametric Verification from Experience via Hidden-State Clustering 

**Authors**: Zhenwen Liang, Ruosen Li, Yujun Zhou, Linfeng Song, Dian Yu, Xinya Du, Haitao Mi, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01591)  

**Abstract**: Assessing the quality of Large Language Model (LLM) outputs presents a critical challenge. Previous methods either rely on text-level information (e.g., reward models, majority voting), which can overfit to superficial cues, or on calibrated confidence from token probabilities, which would fail on less-calibrated models. Yet both of these signals are, in fact, partial projections of a richer source of information: the model's internal hidden states. Early layers, closer to token embeddings, preserve semantic and lexical features that underpin text-based judgments, while later layers increasingly align with output logits, embedding confidence-related information. This paper explores hidden states directly as a unified foundation for verification. We show that the correctness of a solution is encoded as a geometrically separable signature within the trajectory of hidden activations. To validate this, we present Clue (Clustering and Experience-based Verification), a deliberately minimalist, non-parametric verifier. With no trainable parameters, CLUE only summarizes each reasoning trace by an hidden state delta and classifies correctness via nearest-centroid distance to ``success'' and ``failure'' clusters formed from past experience. The simplicity of this method highlights the strength of the underlying signal. Empirically, CLUE consistently outperforms LLM-as-a-judge baselines and matches or exceeds modern confidence-based methods in reranking candidates, improving both top-1 and majority-vote accuracy across AIME 24/25 and GPQA. As a highlight, on AIME 24 with a 1.5B model, CLUE boosts accuracy from 56.7% (majority@64) to 70.0% (top-maj@16). 

---
# One More Question is Enough, Expert Question Decomposition (EQD) Model for Domain Quantitative Reasoning 

**Authors**: Mengyu Wang, Sotirios Sabanis, Miguel de Carvalho, Shay B. Cohen, Tiejun Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.01526)  

**Abstract**: Domain-specific quantitative reasoning remains a major challenge for large language models (LLMs), especially in fields requiring expert knowledge and complex question answering (QA). In this work, we propose Expert Question Decomposition (EQD), an approach designed to balance the use of domain knowledge with computational efficiency. EQD is built on a two-step fine-tuning framework and guided by a reward function that measures the effectiveness of generated sub-questions in improving QA outcomes. It requires only a few thousand training examples and a single A100 GPU for fine-tuning, with inference time comparable to zero-shot prompting. Beyond its efficiency, EQD outperforms state-of-the-art domain-tuned models and advanced prompting strategies. We evaluate EQD in the financial domain, characterized by specialized knowledge and complex quantitative reasoning, across four benchmark datasets. Our method consistently improves QA performance by 0.6% to 10.5% across different LLMs. Our analysis reveals an important insight: in domain-specific QA, a single supporting question often provides greater benefit than detailed guidance steps. 

---
# Detecting LLM-Generated Spam Reviews by Integrating Language Model Embeddings and Graph Neural Network 

**Authors**: Xin Liu, Rongwu Xu, Xinyi Jia, Jason Liao, Jiao Sun, Ling Huang, Wei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01801)  

**Abstract**: The rise of large language models (LLMs) has enabled the generation of highly persuasive spam reviews that closely mimic human writing. These reviews pose significant challenges for existing detection systems and threaten the credibility of online platforms. In this work, we first create three realistic LLM-generated spam review datasets using three distinct LLMs, each guided by product metadata and genuine reference reviews. Evaluations by GPT-4.1 confirm the high persuasion and deceptive potential of these reviews. To address this threat, we propose FraudSquad, a hybrid detection model that integrates text embeddings from a pre-trained language model with a gated graph transformer for spam node classification. FraudSquad captures both semantic and behavioral signals without relying on manual feature engineering or massive training resources. Experiments show that FraudSquad outperforms state-of-the-art baselines by up to 44.22% in precision and 43.01% in recall on three LLM-generated datasets, while also achieving promising results on two human-written spam datasets. Furthermore, FraudSquad maintains a modest model size and requires minimal labeled training data, making it a practical solution for real-world applications. Our contributions include new synthetic datasets, a practical detection framework, and empirical evidence highlighting the urgency of adapting spam detection to the LLM era. Our code and datasets are available at: this https URL. 

---
# TAG-EQA: Text-And-Graph for Event Question Answering via Structured Prompting Strategies 

**Authors**: Maithili Kadam, Francis Ferraro  

**Link**: [PDF](https://arxiv.org/pdf/2510.01391)  

**Abstract**: Large language models (LLMs) excel at general language tasks but often struggle with event-based questions-especially those requiring causal or temporal reasoning. We introduce TAG-EQA (Text-And-Graph for Event Question Answering), a prompting framework that injects causal event graphs into LLM inputs by converting structured relations into natural-language statements. TAG-EQA spans nine prompting configurations, combining three strategies (zero-shot, few-shot, chain-of-thought) with three input modalities (text-only, graph-only, text+graph), enabling a systematic analysis of when and how structured knowledge aids inference. On the TORQUESTRA benchmark, TAG-EQA improves accuracy by 5% on average over text-only baselines, with gains up to 12% in zero-shot settings and 18% when graph-augmented CoT prompting is effective. While performance varies by model and configuration, our findings show that causal graphs can enhance event reasoning in LLMs without fine-tuning, offering a flexible way to encode structure in prompt-based QA. 

---
# TraceDet: Hallucination Detection from the Decoding Trace of Diffusion Large Language Models 

**Authors**: Shenxu Chang, Junchi Yu, Weixing Wang, Yongqiang Chen, Jialin Yu, Philip Torr, Jindong Gu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01274)  

**Abstract**: Diffusion large language models (D-LLMs) have recently emerged as a promising alternative to auto-regressive LLMs (AR-LLMs). However, the hallucination problem in D-LLMs remains underexplored, limiting their reliability in real-world applications. Existing hallucination detection methods are designed for AR-LLMs and rely on signals from single-step generation, making them ill-suited for D-LLMs where hallucination signals often emerge throughout the multi-step denoising process. To bridge this gap, we propose TraceDet, a novel framework that explicitly leverages the intermediate denoising steps of D-LLMs for hallucination detection. TraceDet models the denoising process as an action trace, with each action defined as the model's prediction over the cleaned response, conditioned on the previous intermediate output. By identifying the sub-trace that is maximally informative to the hallucinated responses, TraceDet leverages the key hallucination signals in the multi-step denoising process of D-LLMs for hallucination detection. Extensive experiments on various open source D-LLMs demonstrate that TraceDet consistently improves hallucination detection, achieving an average gain in AUROC of 15.2% compared to baselines. 

---
# AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees 

**Authors**: Hongyi Zhou, Jin Zhu, Pingfan Su, Kai Ye, Ying Yang, Shakeel A O B Gavioli-Akilagun, Chengchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2510.01268)  

**Abstract**: We study the problem of determining whether a piece of text has been authored by a human or by a large language model (LLM). Existing state of the art logits-based detectors make use of statistics derived from the log-probability of the observed text evaluated using the distribution function of a given source LLM. However, relying solely on log probabilities can be sub-optimal. In response, we introduce AdaDetectGPT -- a novel classifier that adaptively learns a witness function from training data to enhance the performance of logits-based detectors. We provide statistical guarantees on its true positive rate, false positive rate, true negative rate and false negative rate. Extensive numerical studies show AdaDetectGPT nearly uniformly improves the state-of-the-art method in various combination of datasets and LLMs, and the improvement can reach up to 58%. A python implementation of our method is available at this https URL. 

---
# RJE: A Retrieval-Judgment-Exploration Framework for Efficient Knowledge Graph Question Answering with LLMs 

**Authors**: Can Lin, Zhengwang Jiang, Ling Zheng, Qi Zhao, Yuhang Zhang, Qi Song, Wangqiu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.01257)  

**Abstract**: Knowledge graph question answering (KGQA) aims to answer natural language questions using knowledge graphs. Recent research leverages large language models (LLMs) to enhance KGQA reasoning, but faces limitations: retrieval-based methods are constrained by the quality of retrieved information, while agent-based methods rely heavily on proprietary LLMs. To address these limitations, we propose Retrieval-Judgment-Exploration (RJE), a framework that retrieves refined reasoning paths, evaluates their sufficiency, and conditionally explores additional evidence. Moreover, RJE introduces specialized auxiliary modules enabling small-sized LLMs to perform effectively: Reasoning Path Ranking, Question Decomposition, and Retriever-assisted Exploration. Experiments show that our approach with proprietary LLMs (such as GPT-4o-mini) outperforms existing baselines while enabling small open-source LLMs (such as 3B and 8B parameters) to achieve competitive results without fine-tuning LLMs. Additionally, RJE substantially reduces the number of LLM calls and token usage compared to agent-based methods, yielding significant efficiency improvements. 

---
# Measuring Algorithmic Partisanship via Zero-Shot Classification and Its Implications on Political Discourse 

**Authors**: Nathan Junzi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.01258)  

**Abstract**: Amidst the rapid normalization of generative artificial intelligence (GAI), intelligent systems have come to dominate political discourse across information mediums. However, internalized political biases stemming from training data skews, human prejudice, and algorithmic flaws continue to plague the novel technology. This paper employs a zero-shot classification approach to evaluate algorithmic political partisanship through a methodical combination of ideological alignment, topicality, response sentiment, and objectivity. A total of 1800 model responses across six mainstream large language models (LLMs) were individually input into four distinct fine-tuned classification algorithms, each responsible for computing an aforementioned bias evaluation metric. Results show an amplified liberal-authoritarian alignment across all six LLMs evaluated, with notable instances of reasoning supersessions and canned refusals. The study subsequently highlights the psychological influences underpinning human-computer interactions and how intrinsic biases can permeate public discourse. The resulting distortion of the political landscape can ultimately manifest as conformity or polarization, depending on a region's pre-existing socio-political structures. 

---
# GPT and Prejudice: A Sparse Approach to Understanding Learned Representations in Large Language Models 

**Authors**: Mariam Mahran, Katharina Simbeck  

**Link**: [PDF](https://arxiv.org/pdf/2510.01252)  

**Abstract**: As large language models (LLMs) are increasingly trained on massive, uncurated corpora, understanding both model representations and the data they internalize has become a major challenge. In this work, we show that pairing LLMs with sparse autoencoders (SAEs) enables interpretation not only of model behavior but also of the deeper structures, themes, and biases embedded in the training data. We train a GPT-style transformer model exclusively on the novels of Jane Austen, a corpus rich in social constructs and narrative patterns. We then apply SAEs to hidden states across multiple layers, uncovering sparse, interpretable features that reflect the key narratives and concepts present in the corpus, including gender, class, and societal duty. Our findings demonstrate that LLMs combined with SAEs can act as scalable probes into complex datasets, offering a new path for corpus exploration, bias discovery, and model interpretability at scale. 

---
# Efficient Uncertainty Estimation for LLM-based Entity Linking in Tabular Data 

**Authors**: Carlo Bono, Federico Belotti, Matteo Palmonari  

**Link**: [PDF](https://arxiv.org/pdf/2510.01251)  

**Abstract**: Linking textual values in tabular data to their corresponding entities in a Knowledge Base is a core task across a variety of data integration and enrichment applications. Although Large Language Models (LLMs) have shown State-of-The-Art performance in Entity Linking (EL) tasks, their deployment in real-world scenarios requires not only accurate predictions but also reliable uncertainty estimates, which require resource-demanding multi-shot inference, posing serious limits to their actual applicability. As a more efficient alternative, we investigate a self-supervised approach for estimating uncertainty from single-shot LLM outputs using token-level features, reducing the need for multiple generations. Evaluation is performed on an EL task on tabular data across multiple LLMs, showing that the resulting uncertainty estimates are highly effective in detecting low-accuracy outputs. This is achieved at a fraction of the computational cost, ultimately supporting a cost-effective integration of uncertainty measures into LLM-based EL workflows. The method offers a practical way to incorporate uncertainty estimation into EL workflows with limited computational overhead. 

---
# Do Bias Benchmarks Generalise? Evidence from Voice-based Evaluation of Gender Bias in SpeechLLMs 

**Authors**: Shree Harsha Bokkahalli Satish, Gustav Eje Henter, Éva Székely  

**Link**: [PDF](https://arxiv.org/pdf/2510.01254)  

**Abstract**: Recent work in benchmarking bias and fairness in speech large language models (SpeechLLMs) has relied heavily on multiple-choice question answering (MCQA) formats. The model is tasked to choose between stereotypical, anti-stereotypical, or neutral/irrelevant answers given an input speech prompt and an optional text prompt. Such MCQA benchmarks implicitly assume that model performance is consistent across other MCQA tasks, voices, and other task formats such as more realistic, long-form evaluations. In this paper, we probe that assumption.
We fine-tune three SpeechLLMs using LoRA adapters to induce specific MCQA behaviours: preference for stereotypical, anti-stereotypical, or neutral/uncertain answers. We then evaluate whether these behaviours generalise to another, distinct MCQA benchmark, and more critically to long-form, creative generation tasks. Our results show that performance on MCQA bias benchmarks fails to reliably predict performances across other MCQA benchmarks, and more importantly across long-form tasks. We conclude that current MCQA bias benchmarks show limited evidence of cross-task generalisation in the speech domain, and also propose an evaluation suite for measuring behaviour transferability in future models and benchmarks. 

---
# LLM Based Sentiment Classification From Bangladesh E-Commerce Reviews 

**Authors**: Sumaiya Tabassum  

**Link**: [PDF](https://arxiv.org/pdf/2510.01276)  

**Abstract**: Sentiment analysis is an essential part of text analysis, which is a larger field that includes determining and evaluating the author's emotional state. This method is essential since it makes it easier to comprehend consumers' feelings, viewpoints, and preferences holistically. The introduction of large language models (LLMs), such as Llama, has greatly increased the availability of cutting-edge model applications, such as sentiment analysis. However, accurate sentiment analysis is hampered by the intricacy of written language and the diversity of languages used in evaluations. The viability of using transformer-based BERT models and other LLMs for sentiment analysis from Bangladesh e commerce reviews is investigated in this paper. A subset of 4000 samples from the original dataset of Bangla and English customer reviews was utilized to fine-tune the model. The fine tuned Llama-3.1-8B model outperformed other fine-tuned models, including Phi-3.5-mini-instruct, Mistral-7B-v0.1, DistilBERT-multilingual, mBERT, and XLM-R-base, with an overall accuracy, precision, recall, and F1 score of 95.5%, 93%, 88%, 90%. The study emphasizes how parameter efficient fine-tuning methods (LoRA and PEFT) can lower computational overhead and make it appropriate for contexts with limited resources. The results show how LLMs can 

---
# SSTAG: Structure-Aware Self-Supervised Learning Method for Text-Attributed Graphs 

**Authors**: Ruyue Liu, Rong Yin, Xiangzhen Bo, Xiaoshuai Hao, Yong Liu, Jinwen Zhong, Can Ma, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01248)  

**Abstract**: Large scale pretrained models have revolutionized Natural Language Processing (NLP) and Computer Vision (CV), showcasing remarkable cross domain generalization abilities. However, in graph learning, models are typically trained on individual graph datasets, limiting their capacity to transfer knowledge across different graphs and tasks. This approach also heavily relies on large volumes of annotated data, which presents a significant challenge in resource-constrained settings. Unlike NLP and CV, graph structured data presents unique challenges due to its inherent heterogeneity, including domain specific feature spaces and structural diversity across various applications. To address these challenges, we propose a novel structure aware self supervised learning method for Text Attributed Graphs (SSTAG). By leveraging text as a unified representation medium for graph learning, SSTAG bridges the gap between the semantic reasoning of Large Language Models (LLMs) and the structural modeling capabilities of Graph Neural Networks (GNNs). Our approach introduces a dual knowledge distillation framework that co-distills both LLMs and GNNs into structure-aware multilayer perceptrons (MLPs), enhancing the scalability of large-scale TAGs. Additionally, we introduce an in-memory mechanism that stores typical graph representations, aligning them with memory anchors in an in-memory repository to integrate invariant knowledge, thereby improving the model's generalization ability. Extensive experiments demonstrate that SSTAG outperforms state-of-the-art models on cross-domain transfer learning tasks, achieves exceptional scalability, and reduces inference costs while maintaining competitive performance. 

---
# A Comparison of Independent and Joint Fine-tuning Strategies for Retrieval-Augmented Generation 

**Authors**: Neal Gregory Lawton, Alfy Samuel, Anoop Kumar, Daben Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01600)  

**Abstract**: A Comparison of Independent and Joint Fine-tuning Strategies for Retrieval-Augmented Generation Download PDF Neal Gregory Lawton, Alfy Samuel, Anoop Kumar, Daben Liu Published: 20 Aug 2025, Last Modified: 17 Sept 2025EMNLP 2025 FindingsConference, Publication Chairs, AuthorsRevisionsBibTeXCC BY 4.0 Keywords: Retrieval-Augmented Generation (RAG), Large Language Models (LLMs), Fine-tuning, Question Answering, Joint fine-tuning TL;DR: We evaluate and compare strategies for fine-tuning Retrieval Augmented Generation (RAG) pipelines, including independent fine-tuning, joint fine-tuning, and two-phase fine-tuning.  Retrieval augmented generation (RAG) is a popular framework for question answering that is powered by two large language models (LLMs): an embedding model that retrieves context documents from a database that are relevant to a given question, and a generator model that uses the retrieved context to generate an answer to the question. Both the embedding and generator models can be fine-tuned to increase performance of a RAG pipeline on a new task, but multiple fine-tuning strategies exist with different costs and benefits. In this paper, we evaluate and compare several RAG fine-tuning strategies, including independent, joint, and two-phase fine-tuning. In our experiments, we observe that all of these strategies achieve about equal improvement in EM and F1 generation quality metrics, although they have significantly different computational costs. We conclude the optimal fine-tuning strategy to use depends on whether the training dataset includes context labels and whether a grid search over the learning rates for the embedding and generator models is required. 

---
# Feasibility of Structuring Stress Documentation Using an Ontology-Guided Large Language Model 

**Authors**: Hyeoneui Kim, Jeongha Kim, Huijing Xu, Jinsun Jung, Sunghoon Kang, Sun Joo Jang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01244)  

**Abstract**: Stress, arising from the dynamic interaction between external stressors, individual appraisals, and physiological or psychological responses, significantly impacts health yet is often underreported and inconsistently documented, typically captured as unstructured free-text in electronic health records. Ambient AI technologies offer promise in reducing documentation burden, but predominantly generate unstructured narratives, limiting downstream clinical utility.
This study aimed to develop an ontology for mental stress and evaluate the feasibility of using a Large Language Model (LLM) to extract ontology-guided stress-related information from narrative text. The Mental Stress Ontology (MeSO) was developed by integrating theoretical models like the Transactional Model of Stress with concepts from 11 validated stress assessment tools. MeSO's structure and content were refined using Ontology Pitfall Scanner! and expert validation.
Using MeSO, six categories of stress-related information--stressor, stress response, coping strategy, duration, onset, and temporal profile--were extracted from 35 Reddit posts using Claude Sonnet 4. Human reviewers evaluated accuracy and ontology coverage. The final ontology included 181 concepts across eight top-level classes. Of 220 extractable stress-related items, the LLM correctly identified 172 (78.2%), misclassified 27 (12.3%), and missed 21 (9.5%). All correctly extracted items were accurately mapped to MeSO, although 24 relevant concepts were not yet represented in the ontology.
This study demonstrates the feasibility of using an ontology-guided LLM for structured extraction of stress-related information, offering potential to enhance the consistency and utility of stress documentation in ambient AI systems. Future work should involve clinical dialogue data and comparison across LLMs. 

---
# Detoxifying Large Language Models via Autoregressive Reward Guided Representation Editing 

**Authors**: Yisong Xiao, Aishan Liu, Siyuan Liang, Zonghao Ying, Xianglong Liu, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2510.01243)  

**Abstract**: Large Language Models (LLMs) have demonstrated impressive performance across various tasks, yet they remain vulnerable to generating toxic content, necessitating detoxification strategies to ensure safe and responsible deployment. Test-time detoxification methods, which typically introduce static or dynamic interventions into LLM representations, offer a promising solution due to their flexibility and minimal invasiveness. However, current approaches often suffer from imprecise interventions, primarily due to their insufficient exploration of the transition space between toxic and non-toxic outputs. To address this challenge, we propose \textsc{A}utoregressive \textsc{R}eward \textsc{G}uided \textsc{R}epresentation \textsc{E}diting (ARGRE), a novel test-time detoxification framework that explicitly models toxicity transitions within the latent representation space, enabling stable and precise reward-guided editing. ARGRE identifies non-toxic semantic directions and interpolates between toxic and non-toxic representations to reveal fine-grained transition trajectories. These trajectories transform sparse toxicity annotations into dense training signals, enabling the construction of an autoregressive reward model that delivers stable and precise editing guidance. At inference, the reward model guides an adaptive two-step editing process to obtain detoxified representations: it first performs directional steering based on expected reward gaps to shift representations toward non-toxic regions, followed by lightweight gradient-based refinements. Extensive experiments across 8 widely used LLMs show that ARGRE significantly outperforms leading baselines in effectiveness (-62.21% toxicity) and efficiency (-47.58% inference time), while preserving the core capabilities of the original model with minimal degradation. Our code is available at the website. 

---
# CIFLEX: Contextual Instruction Flow for Sub-task Execution in Multi-Turn Interactions with a Single On-Device LLM 

**Authors**: Juntae Lee, Jihwan Bang, Seunghan Yang, Simyung Chang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01239)  

**Abstract**: We present CIFLEX (Contextual Instruction Flow for Sub-task Execution), which is a novel execution system for efficient sub-task handling in multi-turn interactions with a single on-device large language model (LLM). As LLMs become increasingly capable, a single model is expected to handle diverse sub-tasks that more effectively and comprehensively support answering user requests. Naive approach reprocesses the entire conversation context when switching between main and sub-tasks (e.g., query rewriting, summarization), incurring significant computational overhead. CIFLEX mitigates this overhead by reusing the key-value (KV) cache from the main task and injecting only task-specific instructions into isolated side paths. After sub-task execution, the model rolls back to the main path via cached context, thereby avoiding redundant prefill computation. To support sub-task selection, we also develop a hierarchical classification strategy tailored for small-scale models, decomposing multi-choice decisions into binary ones. Experiments show that CIFLEX significantly reduces computational costs without degrading task performance, enabling scalable and efficient multi-task dialogue on-device. 

---
# Think Twice, Generate Once: Safeguarding by Progressive Self-Reflection 

**Authors**: Hoang Phan, Victor Li, Qi Lei  

**Link**: [PDF](https://arxiv.org/pdf/2510.01270)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing with their ability to generate coherent and contextually relevant text. However, their deployment raises significant concerns about the potential for generating harmful or inappropriate content. In this paper, we introduce Progressive Self-Reflection (PSR), a novel inference-time technique that empowers LLMs to self-monitor and correct their outputs dynamically. Experimental results demonstrate that applying our proposed method to Llama-3.1-8B-Instruct reduces the attack success rate from 77.5\% to 5.9\%, to Llama-3.1-8B base from 89.7\% to 5.6\%, and to Qwen2.5-7B-Instruct from 44.4\% to 3.8\%, without additional training, while maintaining their original performance on benign tasks. Our approach acts as a test-time scaling method, where additional self-reflection rounds enhance safety at the cost of inference overhead. To balance safety with computational efficiency, we introduce a lightweight self-reflection predictor that estimates the optimal number of reflection rounds based on input complexity. This adaptive mechanism prevents unnecessary self-assessment on benign inputs while ensuring thorough evaluation when encountering potentially harmful content. Our findings suggest that Progressive Self-Reflection serves as a scalable test-time approach, enhancing LLM safety by dynamically allocating computational resources in proportion to the input's risk profile. 

---
# Let's Play Across Cultures: A Large Multilingual, Multicultural Benchmark for Assessing Language Models' Understanding of Sports 

**Authors**: Punit Kumar Singh, Nishant Kumar, Akash Ghosh, Kunal Pasad, Khushi Soni, Manisha Jaishwal, Sriparna Saha, Syukron Abu Ishaq Alfarozi, Asres Temam Abagissa, Kitsuchart Pasupa, Haiqin Yang, Jose G Moreno  

**Link**: [PDF](https://arxiv.org/pdf/2510.01247)  

**Abstract**: Language Models (LMs) are primarily evaluated on globally popular sports, often overlooking regional and indigenous sporting traditions. To address this gap, we introduce \textbf{\textit{CultSportQA}}, a benchmark designed to assess LMs' understanding of traditional sports across 60 countries and 6 continents, encompassing four distinct cultural categories. The dataset features 33,000 multiple-choice questions (MCQs) across text and image modalities, each of which is categorized into three key types: history-based, rule-based, and scenario-based. To evaluate model performance, we employ zero-shot, few-shot, and chain-of-thought (CoT) prompting across a diverse set of Large Language Models (LLMs), Small Language Models (SLMs), and Multimodal Large Language Models (MLMs). By providing a comprehensive multilingual and multicultural sports benchmark, \textbf{\textit{CultSportQA}} establishes a new standard for assessing AI's ability to understand and reason about traditional sports. 

---
# Silent Tokens, Loud Effects: Padding in LLMs 

**Authors**: Rom Himelstein, Amit LeVi, Yonatan Belinkov, Avi Mendelson  

**Link**: [PDF](https://arxiv.org/pdf/2510.01238)  

**Abstract**: Padding tokens are widely used in large language models (LLMs) to equalize sequence lengths during batched inference. While they should be fully masked, implementation errors can cause them to influence computation, and the extent of this influence is not well understood. We systematically study this effect across three open-source model families (Llama, Gemma, Qwen), inserting controlled amounts of padding and evaluating outcomes along four axes: activations, generation quality, bias, and safety. Even small amounts of padding shift hidden representations, degrade quality in smaller models, alter bias in unpredictable ways, and weaken safety guardrails. These findings demonstrate that padding is not a harmless detail but a robustness risk that must be carefully handled in deployment. 

---
# Confidence-Aware Routing for Large Language Model Reliability Enhancement: A Multi-Signal Approach to Pre-Generation Hallucination Mitigation 

**Authors**: Nandakishor M  

**Link**: [PDF](https://arxiv.org/pdf/2510.01237)  

**Abstract**: Large Language Models suffer from hallucination, generating plausible yet factually incorrect content. Current mitigation strategies focus on post-generation correction, which is computationally expensive and fails to prevent unreliable content generation. We propose a confidence-aware routing system that proactively assesses model uncertainty before generation and redirects queries based on estimated reliability. Our approach combines three complementary signals: semantic alignment between internal representations and reference embeddings, internal convergence analysis across model layers, and learned confidence estimation. The unified confidence score determines routing to four pathways: local generation for high confidence, retrieval-augmented generation for medium confidence, larger models for low confidence, and human review for very low confidence. Evaluation on knowledge-intensive QA benchmarks demonstrates significant improvements in hallucination detection (0.74 vs. 0.42 baseline) while reducing computational costs by 40% compared to post-hoc methods. The F1 score improves from 0.61 to 0.82 with low false positive rates (0.09). This paradigm shift from reactive correction to proactive assessment offers a computationally efficient approach to LLM reliability enhancement. 

---
# LLMRank: Understanding LLM Strengths for Model Routing 

**Authors**: Shubham Agrawal, Prasang Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2510.01234)  

**Abstract**: The rapid growth of large language models (LLMs) with diverse capabilities, latency and computational costs presents a critical deployment challenge: selecting the most suitable model for each prompt to optimize the trade-off between performance and efficiency. We introduce LLMRank, a prompt-aware routing framework that leverages rich, human-readable features extracted from prompts, including task type, reasoning patterns, complexity indicators, syntactic cues, and signals from a lightweight proxy solver. Unlike prior one-shot routers that rely solely on latent embeddings, LLMRank predicts per-model utility using a neural ranking model trained on RouterBench, comprising 36,497 prompts spanning 11 benchmarks and 11 state-of-the-art LLMs, from small efficient models to large frontier systems. Our approach achieves up to 89.2% of oracle utility, while providing interpretable feature attributions that explain routing decisions. Extensive studies demonstrate the importance of multifaceted feature extraction and the hybrid ranking objective, highlighting the potential of feature-driven routing for efficient and transparent LLM deployment. 

---
# SKYLENAGE Technical Report: Mathematical Reasoning and Contest-Innovation Benchmarks for Multi-Level Math Evaluation 

**Authors**: Hu Wei, Ze Xu, Boyu Yang, Linlin Miao, Weiqi Zhai, Yihan Li, Zixuan Li, Zhijun Wang, Boya Wang, Jianwei Yu, Jialing Yuan, Xiaoyue Zhang, Cheng He, Minglei Chen, Zifan Zhang, Qianhui Li, Wei Wang, Xiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01241)  

**Abstract**: Large language models (LLMs) now perform strongly on many public math suites, yet frontier separation within mathematics increasingly suffers from ceiling effects. We present two complementary benchmarks: SKYLENAGE-ReasoningMATH, a 100-item, structure-aware diagnostic set with per-item metadata on length, numeric density, and symbolic complexity; and SKYLENAGE-MATH, a 150-item contest-style suite spanning four stages from high school to doctoral under a seven-subject taxonomy. We evaluate fifteen contemporary LLM variants under a single setup and analyze subject x model and grade x model performance. On the contest suite, the strongest model reaches 44% while the runner-up reaches 37%; accuracy declines from high school to doctoral, and top systems exhibit a doctoral-to-high-school retention near 79%. On the reasoning set, the best model attains 81% overall, and hardest-slice results reveal clear robustness gaps between leaders and the mid-tier. In summary, we release SKYLENAGE-ReasoningMATH and report aggregate results for SKYLENAGE-MATH; together, SKYLENAGE provides a hard, reasoning-centered and broadly covering math benchmark with calibrated difficulty and rich metadata, serving as a reference benchmark for future evaluations of mathematical reasoning. 

---
# LOCA: Logical Chain Augmentation for Scientific Corpus Cleaning 

**Authors**: You-Le Fang, Dong-Shan Jian, Xiang Li, Ce Meng, Ling-Shi Meng, Chen-Xu Yan, Zhi-Zhang Bian, Yan-Qing Ma  

**Link**: [PDF](https://arxiv.org/pdf/2510.01249)  

**Abstract**: While Large Language Models (LLMs) excel in general domains, their reliability often falls short in scientific problem-solving. The advancement of scientific AI depends on large-scale, high-quality corpora. However, existing scientific question-answering (QA) datasets suffer from high error rates, frequently resulting from logical leaps and implicit reasoning within the answers. To address this issue, we introduce LOCA (Logical Chain Augmentation), a novel framework for automatically cleaning scientific corpora, implemented through an augment-and-review loop. At its core, LOCA enhances raw answers by completing missing logical steps and explicitly separating the underlying scientific principle from its subsequent derivation. By applying LOCA to challenging scientific corpora, we demonstrate that it can automatically filter noisy datasets, typically reducing the error rate from as high as 20\% to below 2\%. LOCA provides a scalable and effective methodology for creating high-quality scientific corpora, paving the way for more reliable training and evaluation of scientific AI. 

---
# Enhancing Transformer-Based Rerankers with Synthetic Data and LLM-Based Supervision 

**Authors**: Dimitar Peshevski, Kiril Blazhevski, Martin Popovski, Gjorgji Madjarov  

**Link**: [PDF](https://arxiv.org/pdf/2510.01229)  

**Abstract**: Effective document reranking is essential for improving search relevance across diverse applications. While Large Language Models (LLMs) excel at reranking due to their deep semantic understanding and reasoning, their high computational cost makes them impractical for many real-world deployments. Fine-tuning smaller, task-specific models is a more efficient alternative but typically depends on scarce, manually labeled data. To overcome this, we propose a novel pipeline that eliminates the need for human-labeled query-document pairs. Our method uses LLMs to generate synthetic queries from domain-specific corpora and employs an LLM-based classifier to label positive and hard-negative pairs. This synthetic dataset is then used to fine-tune a smaller transformer model with contrastive learning using Localized Contrastive Estimation (LCE) loss. Experiments on the MedQuAD dataset show that our approach significantly boosts in-domain performance and generalizes well to out-of-domain tasks. By using LLMs for data generation and supervision rather than inference, we reduce computational costs while maintaining strong reranking capabilities. 

---
# SeMob: Semantic Synthesis for Dynamic Urban Mobility Prediction 

**Authors**: Runfei Chen, Shuyang Jiang, Wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01245)  

**Abstract**: Human mobility prediction is vital for urban services, but often fails to account for abrupt changes from external events. Existing spatiotemporal models struggle to leverage textual descriptions detailing these events. We propose SeMob, an LLM-powered semantic synthesis pipeline for dynamic mobility prediction. Specifically, SeMob employs a multi-agent framework where LLM-based agents automatically extract and reason about spatiotemporally related text from complex online texts. Fine-grained relevant contexts are then incorporated with spatiotemporal data through our proposed innovative progressive fusion architecture. The rich pre-trained event prior contributes enriched insights about event-driven prediction, and hence results in a more aligned forecasting model. Evaluated on a dataset constructed through our pipeline, SeMob achieves maximal reductions of 13.92% in MAE and 11.12% in RMSE compared to the spatiotemporal model. Notably, the framework exhibits pronounced superiority especially within spatiotemporal regions close to an event's location and time of occurrence. 

---
# Uncovering Implicit Bias in Large Language Models with Concept Learning Dataset 

**Authors**: Leroy Z. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01219)  

**Abstract**: We introduce a dataset of concept learning tasks that helps uncover implicit biases in large language models. Using in-context concept learning experiments, we found that language models may have a bias toward upward monotonicity in quantifiers; such bias is less apparent when the model is tested by direct prompting without concept learning components. This demonstrates that in-context concept learning can be an effective way to discover hidden biases in language models. 

---
# Discourse vs emissions: Analysis of corporate narratives, symbolic practices, and mimicry through LLMs 

**Authors**: Bertrand Kian Hassani, Yacoub Bahini, Rizwan Mushtaq  

**Link**: [PDF](https://arxiv.org/pdf/2510.01222)  

**Abstract**: Climate change has increased demands for transparent and comparable corporate climate disclosures, yet imitation and symbolic reporting often undermine their value. This paper develops a multidimensional framework to assess disclosure maturity among 828 this http URL firms using large language models (LLMs) fine-tuned for climate communication. Four classifiers-sentiment, commitment, specificity, and target ambition-extract narrative indicators from sustainability and annual reports, which are linked to firm attributes such as emissions, market capitalization, and sector. Analyses reveal three insights: (1) risk-focused narratives often align with explicit commitments, but quantitative targets (e.g., net-zero pledges) remain decoupled from tone; (2) larger and higher-emitting firms disclose more commitments and actions than peers, though inconsistently with quantitative targets; and (3) widespread similarity in disclosure styles suggests mimetic behavior, reducing differentiation and decision usefulness. These results highlight the value of LLMs for ESG narrative analysis and the need for stronger regulation to connect commitments with verifiable transition strategies. 

---
# Trustworthy Summarization via Uncertainty Quantification and Risk Awareness in Large Language Models 

**Authors**: Shuaidong Pan, Di Wu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01231)  

**Abstract**: This study addresses the reliability of automatic summarization in high-risk scenarios and proposes a large language model framework that integrates uncertainty quantification and risk-aware mechanisms. Starting from the demands of information overload and high-risk decision-making, a conditional generation-based summarization model is constructed, and Bayesian inference is introduced during generation to model uncertainty in the parameter space, which helps avoid overconfident predictions. The uncertainty level of the generated content is measured using predictive distribution entropy, and a joint optimization of entropy regularization and risk-aware loss is applied to ensure that key information is preserved and risk attributes are explicitly expressed during information compression. On this basis, the model incorporates risk scoring and regulation modules, allowing summaries to cover the core content accurately while enhancing trustworthiness through explicit risk-level prompts. Comparative experiments and sensitivity analyses verify that the proposed method significantly improves the robustness and reliability of summarization in high-risk applications while maintaining fluency and semantic integrity. This research provides a systematic solution for trustworthy summarization and demonstrates both scalability and practical value at the methodological level. 

---
# The Reasoning Boundary Paradox: How Reinforcement Learning Constrains Language Models 

**Authors**: Phuc Minh Nguyen, Chinh D. La, Duy M. H. Nguyen, Nitesh V. Chawla, Binh T. Nguyen, Khoa D. Doan  

**Link**: [PDF](https://arxiv.org/pdf/2510.02230)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a key method for improving Large Language Models' reasoning capabilities, yet recent evidence suggests it may paradoxically shrink the reasoning boundary rather than expand it. This paper investigates the shrinkage issue of RLVR by analyzing its learning dynamics and reveals two critical phenomena that explain this failure. First, we expose negative interference in RLVR, where learning to solve certain training problems actively reduces the likelihood of correct solutions for others, leading to the decline of Pass@$k$ performance, or the probability of generating a correct solution within $k$ attempts. Second, we uncover the winner-take-all phenomenon: RLVR disproportionately reinforces problems with high likelihood, correct solutions, under the base model, while suppressing other initially low-likelihood ones. Through extensive theoretical and empirical analysis on multiple mathematical reasoning benchmarks, we show that this effect arises from the inherent on-policy sampling in standard RL objectives, causing the model to converge toward narrow solution strategies. Based on these insights, we propose a simple yet effective data curation algorithm that focuses RLVR learning on low-likelihood problems, achieving notable improvement in Pass@$k$ performance. Our code is available at this https URL. 

---
# ExGRPO: Learning to Reason from Experience 

**Authors**: Runzhe Zhan, Yafu Li, Zhi Wang, Xiaoye Qu, Dongrui Liu, Jing Shao, Derek F. Wong, Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2510.02245)  

**Abstract**: Reinforcement learning from verifiable rewards (RLVR) is an emerging paradigm for improving the reasoning ability of large language models. However, standard on-policy training discards rollout experiences after a single update, leading to computational inefficiency and instability. While prior work on RL has highlighted the benefits of reusing past experience, the role of experience characteristics in shaping learning dynamics of large reasoning models remains underexplored. In this paper, we are the first to investigate what makes a reasoning experience valuable and identify rollout correctness and entropy as effective indicators of experience value. Based on these insights, we propose ExGRPO (Experiential Group Relative Policy Optimization), a framework that organizes and prioritizes valuable experiences, and employs a mixed-policy objective to balance exploration with experience exploitation. Experiments on five backbone models (1.5B-8B parameters) show that ExGRPO consistently improves reasoning performance on mathematical/general benchmarks, with an average gain of +3.5/7.6 points over on-policy RLVR. Moreover, ExGRPO stabilizes training on both stronger and weaker models where on-policy methods fail. These results highlight principled experience management as a key ingredient for efficient and scalable RLVR. 

---
# RLAD: Training LLMs to Discover Abstractions for Solving Reasoning Problems 

**Authors**: Yuxiao Qu, Anikait Singh, Yoonho Lee, Amrith Setlur, Ruslan Salakhutdinov, Chelsea Finn, Aviral Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.02263)  

**Abstract**: Reasoning requires going beyond pattern matching or memorization of solutions to identify and implement "algorithmic procedures" that can be used to deduce answers to hard problems. Doing so requires realizing the most relevant primitives, intermediate results, or shared procedures, and building upon them. While RL post-training on long chains of thought ultimately aims to uncover this kind of algorithmic behavior, most reasoning traces learned by large models fail to consistently capture or reuse procedures, instead drifting into verbose and degenerate exploration. To address more effective reasoning, we introduce reasoning abstractions: concise natural language descriptions of procedural and factual knowledge that guide the model toward learning successful reasoning. We train models to be capable of proposing multiple abstractions given a problem, followed by RL that incentivizes building a solution while using the information provided by these abstractions. This results in a two-player RL training paradigm, abbreviated as RLAD, that jointly trains an abstraction generator and a solution generator. This setup effectively enables structured exploration, decouples learning signals of abstraction proposal and solution generation, and improves generalization to harder problems. We also show that allocating more test-time compute to generating abstractions is more beneficial for performance than generating more solutions at large test budgets, illustrating the role of abstractions in guiding meaningful exploration. 

---
# Plan Then Action:High-Level Planning Guidance Reinforcement Learning for LLM Reasoning 

**Authors**: Zhihao Dou, Qinjian Zhao, Zhongwei Wan, Dinggen Zhang, Weida Wang, Towsif Raiyan, Benteng Chen, Qingtao Pan, Yang Ouyang, Zhiqiang Gao, Shufei Zhang, Sumon Biswas  

**Link**: [PDF](https://arxiv.org/pdf/2510.01833)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable reasoning abilities in complex tasks, often relying on Chain-of-Thought (CoT) reasoning. However, due to their autoregressive token-level generation, the reasoning process is largely constrained to local decision-making and lacks global planning. This limitation frequently results in redundant, incoherent, or inaccurate reasoning, which significantly degrades overall performance. Existing approaches, such as tree-based algorithms and reinforcement learning (RL), attempt to address this issue but suffer from high computational costs and often fail to produce optimal reasoning trajectories. To tackle this challenge, we propose Plan-Then-Action Enhanced Reasoning with Group Relative Policy Optimization PTA-GRPO, a two-stage framework designed to improve both high-level planning and fine-grained CoT reasoning. In the first stage, we leverage advanced LLMs to distill CoT into compact high-level guidance, which is then used for supervised fine-tuning (SFT). In the second stage, we introduce a guidance-aware RL method that jointly optimizes the final output and the quality of high-level guidance, thereby enhancing reasoning effectiveness. We conduct extensive experiments on multiple mathematical reasoning benchmarks, including MATH, AIME2024, AIME2025, and AMC, across diverse base models such as Qwen2.5-7B-Instruct, Qwen3-8B, Qwen3-14B, and LLaMA3.2-3B. Experimental results demonstrate that PTA-GRPO consistently achieves stable and significant improvements across different models and tasks, validating its effectiveness and generalization. 

---
# ClaimCheck: Real-Time Fact-Checking with Small Language Models 

**Authors**: Akshith Reddy Putta, Jacob Devasier, Chengkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.01226)  

**Abstract**: We introduce ClaimCheck, an LLM-guided automatic fact-checking system designed to verify real-world claims using live Web evidence and small language models. Unlike prior systems that rely on large, closed-source models and static knowledge stores, ClaimCheck employs a transparent, stepwise verification pipeline that mirrors human fact-checking workflows consisting of Web search query planning, Web-based evidence retrieval and summarization, evidence synthesis and re-retrieval, and claim verdict evaluation. Each module is optimized for small LLMs, allowing the system to deliver accurate and interpretable fact-checking with significantly lower computational requirements. Despite using a much smaller Qwen3-4B model, ClaimCheck achieves state-of-the-art accuracy of 76.4% on the AVeriTeC dataset, outperforming previous approaches using LLaMA3.1 70B and GPT-4o. Extensive ablations demonstrate that careful modular design and prompting strategies can overcome the limitations of smaller LLMs. To promote accessibility and transparency, we provide a public demo at this https URL. 

---
# InvThink: Towards AI Safety via Inverse Reasoning 

**Authors**: Yubin Kim, Taehan Kim, Eugene Park, Chunjong Park, Cynthia Breazeal, Daniel McDuff, Hae Won Park  

**Link**: [PDF](https://arxiv.org/pdf/2510.01569)  

**Abstract**: We present InvThink, a simple yet powerful approach that gives large language models (LLMs) the capability of inverse thinking: reasoning through failure modes before generating responses. Unlike existing safety alignment methods that optimize directly for safe response, InvThink instructs models to 1) enumerate potential harms, 2) analyze their consequences, and 3) generate safe outputs that proactively avoid these risks. Our method reveals three key findings: (i) safety improvements show stronger scaling with model size compared to existing safety methods. (ii) InvThink mitigates safety tax; by training models to systematically consider failure modes, it preserves general reasoning capabilities on standard benchmarks. (iii) beyond general safety tasks, InvThink excels in high-stakes domains including external-facing (medicine, finance, law) and agentic (blackmail, murder) risk scenarios, achieving up to 15.7% reduction in harmful responses compared to baseline methods like SafetyPrompt. We further implement InvThink via supervised fine-tuning, and reinforcement learning across three LLM families. These results suggest that inverse reasoning provides a scalable and generalizable path toward safer, more capable language models. 

---
# Information Seeking for Robust Decision Making under Partial Observability 

**Authors**: Djengo Cyun-Jyun Fang, Tsung-Wei Ke  

**Link**: [PDF](https://arxiv.org/pdf/2510.01531)  

**Abstract**: Explicit information seeking is essential to human problem-solving in practical environments characterized by incomplete information and noisy dynamics. When the true environmental state is not directly observable, humans seek information to update their internal dynamics and inform future decision-making. Although existing Large Language Model (LLM) planning agents have addressed observational uncertainty, they often overlook discrepancies between their internal dynamics and the actual environment. We introduce Information Seeking Decision Planner (InfoSeeker), an LLM decision-making framework that integrates task-oriented planning with information seeking to align internal dynamics and make optimal decisions under uncertainty in both agent observations and environmental dynamics. InfoSeeker prompts an LLM to actively gather information by planning actions to validate its understanding, detect environmental changes, or test hypotheses before generating or revising task-oriented plans. To evaluate InfoSeeker, we introduce a novel benchmark suite featuring partially observable environments with incomplete observations and uncertain dynamics. Experiments demonstrate that InfoSeeker achieves a 74% absolute performance gain over prior methods without sacrificing sample efficiency. Moreover, InfoSeeker generalizes across LLMs and outperforms baselines on established benchmarks such as robotic manipulation and web navigation. These findings underscore the importance of tightly integrating planning and information seeking for robust behavior in partially observable environments. The project page is available at this https URL 

---
# Quagmires in SFT-RL Post-Training: When High SFT Scores Mislead and What to Use Instead 

**Authors**: Feiyang Kang, Michael Kuchnik, Karthik Padthe, Marin Vlastelica, Ruoxi Jia, Carole-Jean Wu, Newsha Ardalani  

**Link**: [PDF](https://arxiv.org/pdf/2510.01624)  

**Abstract**: In post-training for reasoning Large Language Models (LLMs), the current state of practice trains LLMs in two independent stages: Supervised Fine-Tuning (SFT) and Reinforcement Learning with Verifiable Rewards (RLVR, shortened as ``RL'' below). In this work, we challenge whether high SFT scores translate to improved performance after RL. We provide extensive counter-examples where this is not true. We find high SFT scores can be biased toward simpler or more homogeneous data and are not reliably predictive of subsequent RL gains or scaled-up post-training effectiveness. In some cases, RL training on models with improved SFT performance could lead to substantially worse outcome compared to RL on the base model without SFT. We study alternative metrics and identify generalization loss on held-out reasoning examples and Pass@large k performance to provide strong proxies for the RL outcome. We trained hundreds of models up to 12B-parameter with SFT and RLVR via GRPO and ran extensive evaluations on 7 math benchmarks with up to 256 repetitions, spending $>$1M GPU hours. Experiments include models from Llama3, Mistral-Nemo, Qwen3 and multiple state-of-the-art SFT/RL datasets. Compared to directly predicting from pre-RL performance, prediction based on generalization loss and Pass@large k achieves substantial higher precision, improving $R^2$ coefficient and Spearman's rank correlation coefficient by up to 0.5 (2x). This provides strong utility for broad use cases. For example, in most experiments, we find SFT training on unique examples for a one epoch underperforms training on half examples for two epochs, either after SFT or SFT-then-RL; With the same SFT budget, training only on short examples may lead to better SFT performance, though, it often leads to worse outcome after RL compared to training on examples with varying lengths. Evaluation tool will be open-sourced. 

---
# StockBench: Can LLM Agents Trade Stocks Profitably In Real-world Markets? 

**Authors**: Yanxu Chen, Zijun Yao, Yantao Liu, Jin Ye, Jianing Yu, Lei Hou, Juanzi Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.02209)  

**Abstract**: Large language models (LLMs) have recently demonstrated strong capabilities as autonomous agents, showing promise in reasoning, tool use, and sequential decision-making. While prior benchmarks have evaluated LLM agents in domains such as software engineering and scientific discovery, the finance domain remains underexplored, despite its direct relevance to economic value and high-stakes decision-making. Existing financial benchmarks primarily test static knowledge through question answering, but they fall short of capturing the dynamic and iterative nature of trading. To address this gap, we introduce StockBench, a contamination-free benchmark designed to evaluate LLM agents in realistic, multi-month stock trading environments. Agents receive daily market signals -- including prices, fundamentals, and news -- and must make sequential buy, sell, or hold decisions. Performance is assessed using financial metrics such as cumulative return, maximum drawdown, and the Sortino ratio. Our evaluation of state-of-the-art proprietary (e.g., GPT-5, Claude-4) and open-weight (e.g., Qwen3, Kimi-K2, GLM-4.5) models shows that while most LLM agents struggle to outperform the simple buy-and-hold baseline, several models demonstrate the potential to deliver higher returns and manage risk more effectively. These findings highlight both the challenges and opportunities in developing LLM-powered financial agents, showing that excelling at static financial knowledge tasks does not necessarily translate into successful trading strategies. We release StockBench as an open-source resource to support reproducibility and advance future research in this domain. 

---
# PychoBench: Evaluating the Psychology Intelligence of Large Language Models 

**Authors**: Min Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2510.01611)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable success across a wide range of industries, primarily due to their impressive generative abilities. Yet, their potential in applications requiring cognitive abilities, such as psychological counseling, remains largely untapped. This paper investigates the key question: Can LLMs be effectively applied to psychological counseling? To determine whether an LLM can effectively take on the role of a psychological counselor, the first step is to assess whether it meets the qualifications required for such a role, namely the ability to pass the U.S. National Counselor Certification Exam (NCE). This is because, just as a human counselor must pass a certification exam to practice, an LLM must demonstrate sufficient psychological knowledge to meet the standards required for such a role. To address this, we introduce PsychoBench, a benchmark grounded in this http URL counselor examinations, a licensure test for professional counselors that requires about 70% accuracy to pass. PsychoBench comprises approximately 2,252 carefully curated single-choice questions, crafted to require deep understanding and broad enough to cover various sub-disciplines of psychology. This benchmark provides a comprehensive assessment of an LLM's ability to function as a counselor. Our evaluation shows that advanced models such as GPT-4o, Llama3.3-70B, and Gemma3-27B achieve well above the passing threshold, while smaller open-source models (e.g., Qwen2.5-7B, Mistral-7B) remain far below it. These results suggest that only frontier LLMs are currently capable of meeting counseling exam standards, highlighting both the promise and the challenges of developing psychology-oriented LLMs. 

---
# VOGUE: Guiding Exploration with Visual Uncertainty Improves Multimodal Reasoning 

**Authors**: Rui Liu, Dian Yu, Tong Zheng, Runpeng Dai, Zongxia Li, Wenhao Yu, Zhenwen Liang, Linfeng Song, Haitao Mi, Pratap Tokekar, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01444)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) improves reasoning in large language models (LLMs) but struggles with exploration, an issue that still persists for multimodal LLMs (MLLMs). Current methods treat the visual input as a fixed, deterministic condition, overlooking a critical source of ambiguity and struggling to build policies robust to plausible visual variations. We introduce $\textbf{VOGUE (Visual Uncertainty Guided Exploration)}$, a novel method that shifts exploration from the output (text) to the input (visual) space. By treating the image as a stochastic context, VOGUE quantifies the policy's sensitivity to visual perturbations using the symmetric KL divergence between a "raw" and "noisy" branch, creating a direct signal for uncertainty-aware exploration. This signal shapes the learning objective via an uncertainty-proportional bonus, which, combined with a token-entropy bonus and an annealed sampling schedule, effectively balances exploration and exploitation. Implemented within GRPO on two model scales (Qwen2.5-VL-3B/7B), VOGUE boosts pass@1 accuracy by an average of 2.6% on three visual math benchmarks and 3.7% on three general-domain reasoning benchmarks, while simultaneously increasing pass@4 performance and mitigating the exploration decay commonly observed in RL fine-tuning. Our work shows that grounding exploration in the inherent uncertainty of visual inputs is an effective strategy for improving multimodal reasoning. 

---
# Fine-tuning with RAG for Improving LLM Learning of New Skills 

**Authors**: Humaid Ibrahim, Nikolai Rozanov, Marek Rei  

**Link**: [PDF](https://arxiv.org/pdf/2510.01375)  

**Abstract**: Large language model (LLM) agents deployed for multi-step tasks frequently fail in predictable ways: attempting actions with unmet preconditions, issuing redundant commands, or mishandling environment constraints. While retrieval-augmented generation (RAG) can improve performance by providing runtime guidance, it requires maintaining external knowledge databases and adds computational overhead at every deployment. We propose a simple pipeline that converts inference-time retrieval into learned competence through distillation. Our approach: (1) extracts compact, reusable hints from agent failures, (2) uses these hints to generate improved teacher trajectories via one-shot retrieval at episode start, and (3) trains student models on these trajectories with hint strings removed, forcing internalization rather than memorization. Across two interactive benchmarks, ALFWorld (household tasks) and WebShop (online shopping), distilled students consistently outperform baseline agents, achieving up to 91% success on ALFWorld (vs. 79% for baselines) and improving WebShop scores to 72 (vs. 61 for baselines), while using 10-60% fewer tokens than retrieval-augmented teachers depending on the environment. The approach generalizes across model scales (7B/14B parameters) and agent architectures (ReAct/StateAct), demonstrating that retrieval benefits can be effectively internalized through targeted fine-tuning without permanent runtime dependencies. 

---
# Think Right: Learning to Mitigate Under-Over Thinking via Adaptive, Attentive Compression 

**Authors**: Joykirat Singh, Justin Chih-Yao Chen, Archiki Prasad, Elias Stengel-Eskin, Akshay Nambi, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2510.01581)  

**Abstract**: Recent thinking models solve complex reasoning tasks by scaling test-time compute, but this scaling must be allocated in line with task difficulty. On one hand, short reasoning (underthinking) leads to errors on harder problems that require extended reasoning steps; but, excessively long reasoning (overthinking) can be token-inefficient, generating unnecessary steps even after reaching a correct intermediate solution. We refer to this as under-adaptivity, where the model fails to modulate its response length appropriately given problems of varying difficulty. To address under-adaptivity and strike a balance between under- and overthinking, we propose TRAAC (Think Right with Adaptive, Attentive Compression), an online post-training RL method that leverages the model's self-attention over a long reasoning trajectory to identify important steps and prune redundant ones. TRAAC also estimates difficulty and incorporates it into training rewards, thereby learning to allocate reasoning budget commensurate with example difficulty. Our approach improves accuracy, reduces reasoning steps, and enables adaptive thinking compared to base models and other RL baselines. Across a variety of tasks (AIME, AMC, GPQA-D, BBEH), TRAAC (Qwen3-4B) achieves an average absolute accuracy gain of 8.4% with a relative reduction in reasoning length of 36.8% compared to the base model, and a 7.9% accuracy gain paired with a 29.4% length drop compared to the best RL baseline. TRAAC also shows strong generalization: although our models are trained on math datasets, they show accuracy and efficiency gains on out-of-distribution non-math datasets like GPQA-D, BBEH, and OptimalThinkingBench. Our analysis further verifies that TRAAC provides fine-grained adjustments to thinking budget based on difficulty and that a combination of task-difficulty calibration and attention-based compression yields gains across diverse tasks. 

---
# Utilizing Modern Large Language Models (LLM) for Financial Trend Analysis and Digest Creation 

**Authors**: Andrei Lazarev, Dmitrii Sedov  

**Link**: [PDF](https://arxiv.org/pdf/2510.01225)  

**Abstract**: The exponential growth of information presents a significant challenge for researchers and professionals seeking to remain at the forefront of their fields and this paper introduces an innovative framework for automatically generating insightful financial digests using the power of Large Language Models (LLMs), specifically Google's Gemini Pro. By leveraging a combination of data extraction from OpenAlex, strategic prompt engineering, and LLM-driven analysis, we demonstrate the automated example of creating a comprehensive digests that generalize key findings, identify emerging trends. This approach addresses the limitations of traditional analysis methods, enabling the efficient processing of vast amounts of unstructured data and the delivery of actionable insights in an easily digestible format. This paper describes how LLMs work in simple words and how we can use their power to help researchers and scholars save their time and stay informed about current trends. Our study includes step-by-step process, from data acquisition and JSON construction to interaction with Gemini and the automated generation of PDF reports, including a link to the project's GitHub repository for broader accessibility and further development. 

---
# Control the Temperature: Selective Sampling for Diverse and High-Quality LLM Outputs 

**Authors**: Sergey Troshin, Wafaa Mohammed, Yan Meng, Christof Monz, Antske Fokkens, Vlad Niculae  

**Link**: [PDF](https://arxiv.org/pdf/2510.01218)  

**Abstract**: Diversity is an essential metric for evaluating the creativity of outputs generated by language models. Temperature-based sampling is a common strategy to increase diversity. However, for tasks that require high precision, e.g., mathematical reasoning, uncontrolled high temperature sampling, e.g., min-$p$ or top-$p$, degrades reasoning quality. We demonstrate that the loss of accuracy is caused by sampling incorrect continuations in sensitive decoding positions. To address this, in this paper, we propose \textbf{selective sampling}, a method that dynamically switches between greedy and high-temperature sampling based on a sampling risk metric. This risk metric estimates the likelihood of output errors when applying high-temperature sampling on the current token position. To predict sampling risk, we train a lightweight classifier on a small subset of verifiable problems. The trained classifier can be integrated with the base language model with minimal latency overhead. Experiments on mathematical reasoning tasks demonstrate that selective sampling enhances the quality-diversity trade-off, even in high-temperature settings. 

---
# Optimal Stopping vs Best-of-$N$ for Inference Time Optimization 

**Authors**: Yusuf Kalayci, Vinod Raman, Shaddin Dughmi  

**Link**: [PDF](https://arxiv.org/pdf/2510.01394)  

**Abstract**: Large language model (LLM) generation often requires balancing output quality against inference cost, especially when using multiple generations. We introduce a new framework for inference-time optimization based on the classical Pandora's Box problem. Viewing each generation as opening a costly "box" with random reward, we develop algorithms that decide when to stop generating without knowing the underlying reward distribution. Our first contribution is a UCB-style Pandora's Box algorithm, which achieves performance that is provably close to Weitzman's algorithm, the optimal strategy when the distribution is known. We further adapt this method to practical LLM settings by addressing reward scaling across prompts via a Bradley-Terry inspired transformation. This leads to an adaptive inference-time optimization method that normalizes rewards and learns stopping thresholds on the fly. Experiments on the AlpacaFarm and HH-RLHF datasets, using multiple LLM-reward model pairs, show that our adaptive strategy can obtain the same performance as non-adaptive Best-of-N sampling while requiring 15-35 percent fewer generations on average. Our results establish a principled bridge between optimal stopping theory and inference-time scaling, providing both theoretical performance bounds and practical efficiency gains for LLM deployment. 

---
# Automated Extraction of Material Properties using LLM-based AI Agents 

**Authors**: Subham Ghosh, Abhishek Tewari  

**Link**: [PDF](https://arxiv.org/pdf/2510.01235)  

**Abstract**: The rapid discovery of materials is constrained by the lack of large, machine-readable datasets that couple performance metrics with structural context. Existing databases are either small, manually curated, or biased toward first principles results, leaving experimental literature underexploited. We present an agentic, large language model (LLM)-driven workflow that autonomously extracts thermoelectric and structural-properties from about 10,000 full-text scientific articles. The pipeline integrates dynamic token allocation, zeroshot multi-agent extraction, and conditional table parsing to balance accuracy against computational cost. Benchmarking on 50 curated papers shows that GPT-4.1 achieves the highest accuracy (F1 = 0.91 for thermoelectric properties and 0.82 for structural fields), while GPT-4.1 Mini delivers nearly comparable performance (F1 = 0.89 and 0.81) at a fraction of the cost, enabling practical large scale deployment. Applying this workflow, we curated 27,822 temperature resolved property records with normalized units, spanning figure of merit (ZT), Seebeck coefficient, conductivity, resistivity, power factor, and thermal conductivity, together with structural attributes such as crystal class, space group, and doping strategy. Dataset analysis reproduces known thermoelectric trends, such as the superior performance of alloys over oxides and the advantage of p-type doping, while also surfacing broader structure-property correlations. To facilitate community access, we release an interactive web explorer with semantic filters, numeric queries, and CSV export. This study delivers the largest LLM-curated thermoelectric dataset to date, provides a reproducible and cost-profiled extraction pipeline, and establishes a foundation for scalable, data-driven materials discovery beyond thermoelectrics. 

---
# RSAVQ: Riemannian Sensitivity-Aware Vector Quantization for Large Language Models 

**Authors**: Zukang Xu, Xing Hu, Qiang Wu, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01240)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks. However, their exponentially increasing parameters pose significant challenges for deployment on resource-constrained devices. Vector Quantization (VQ) shows great promise for low-bit quantization (e.g., 2 to 4 bits), but existing work faces two key challenges: unconstrained direction error and suboptimal bit allocation. In this paper, we propose RSAVQ, a novel VQ framework to enhance extremely low-bit quantization for LLMs. RSAVQ introduces two geometry-driven innovations that effectively mitigate above limitations: (1) Error Direction Sensitivity Guidance (EDSG), which leverages the Fisher Information Matrix (FIM)-induced Riemannian metric to project quantization errors onto low-sensitivity directions in the parameter space. Specifically, this projection is performed along the negative natural gradient direction, which effectively suppresses error expansion. (2) Weight Channel Sensitivity Guidance (WCSG) , which constructs a channel-wise sensitivity metric via FIM curvature analysis to dynamically guide bit resource allocation. The approach facilitates a globally optimal quantization solution within prescribed bit constraints. Experiments demonstrate that RSAVQ outperforms existing methods for LLMs. For example, in 2-bit quantization of LLaMA-3 8B, RSAVQ leads baselines like VPTQ and QuIP# by 0.4 in perplexity (PPL) and 1.5 in zero-shot accuracy. This work offers a practical solution for constrained environments and a theoretical bridge between information geometry and the quantization of neural networks, advancing efficient deep learning. 

---
# Jailbreaking LLMs via Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge 

**Authors**: Hui Dou, Ning Xu, Yiwen Zhang, Kaibin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01223)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in various tasks. However, they remain exposed to jailbreak attacks, eliciting harmful responses. The nested scenario strategy has been increasingly adopted across various methods, demonstrating immense potential. Nevertheless, these methods are easily detectable due to their prominent malicious intentions. In this work, we are the first to find and systematically verify that LLMs' alignment defenses are not sensitive to nested scenarios, where these scenarios are highly semantically relevant to the queries and incorporate targeted toxic knowledge. This is a crucial yet insufficiently explored direction. Based on this, we propose RTS-Attack (Semantically Relevant Nested Scenarios with Targeted Toxic Knowledge), an adaptive and automated framework to examine LLMs' alignment. By building scenarios highly relevant to the queries and integrating targeted toxic knowledge, RTS-Attack bypasses the alignment defenses of LLMs. Moreover, the jailbreak prompts generated by RTS-Attack are free from harmful queries, leading to outstanding concealment. Extensive experiments demonstrate that RTS-Attack exhibits superior performance in both efficiency and universality compared to the baselines across diverse advanced LLMs, including GPT-4o, Llama3-70b, and Gemini-pro. Our complete code is available in the supplementary material. WARNING: THIS PAPER CONTAINS POTENTIALLY HARMFUL CONTENT. 

---
# Demystifying the Roles of LLM Layers in Retrieval, Knowledge, and Reasoning 

**Authors**: Xinyuan Song, Keyu Wang, PengXiang Li, Lu Yin, Shiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.02091)  

**Abstract**: Recent studies suggest that the deeper layers of Large Language Models (LLMs) contribute little to representation learning and can often be removed without significant performance loss. However, such claims are typically drawn from narrow evaluations and may overlook important aspects of model behavior. In this work, we present a systematic study of depth utilization across diverse dimensions, including evaluation protocols, task categories, and model architectures. Our analysis confirms that very deep layers are generally less effective than earlier ones, but their contributions vary substantially with the evaluation setting. Under likelihood-based metrics without generation, pruning most layers preserves performance, with only the initial few being critical. By contrast, generation-based evaluation uncovers indispensable roles for middle and deeper layers in enabling reasoning and maintaining long-range coherence. We further find that knowledge and retrieval are concentrated in shallow components, whereas reasoning accuracy relies heavily on deeper layers -- yet can be reshaped through distillation. These results highlight that depth usage in LLMs is highly heterogeneous and context-dependent, underscoring the need for task-, metric-, and model-aware perspectives in both interpreting and compressing large models. 

---
# Learning a Dense Reasoning Reward Model from Expert Demonstration via Inverse Reinforcement Learning 

**Authors**: Claudio Fanconi, Nicolás Astorga, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2510.01857)  

**Abstract**: We reframe and operationalise adversarial inverse reinforcement learning (IRL) to large language model reasoning, learning a dense, token-level reward model for process supervision directly from expert demonstrations rather than imitating style via supervised fine-tuning. The learned reasoning reward serves two complementary roles: (i) it provides step-level feedback to optimise a reasoning policy during training; and (ii) it functions at inference as a critic to rerank sampled traces under fixed compute budgets. We demonstrate that our approach prioritises correctness over surface form, yielding scores that correlate with eventual answer validity and enabling interpretable localisation of errors within a trace. Empirically, on GSM8K with Llama3 and Qwen2.5 backbones, we demonstrate: (i) dense reasoning rewards can be used as a learning signal to elicit reasoning, and (ii) predictive performance is improved from reward-guided reranking (notably for Llama-based policies). By unifying training signals, inference-time selection, and token-level diagnostics into a single reasoning reward, this work suggests reusable process-level rewards with broad potential to enhance multi-step reasoning in language models. 

---
# UpSafe$^\circ$C: Upcycling for Controllable Safety in Large Language Models 

**Authors**: Yuhao Sun, Zhuoer Xu, Shiwen Cui, Kun Yang, Lingyun Yu, Yongdong Zhang, Hongtao Xie  

**Link**: [PDF](https://arxiv.org/pdf/2510.02194)  

**Abstract**: Large Language Models (LLMs) have achieved remarkable progress across a wide range of tasks, but remain vulnerable to safety risks such as harmful content generation and jailbreak attacks. Existing safety techniques -- including external guardrails, inference-time guidance, and post-training alignment -- each face limitations in balancing safety, utility, and controllability. In this work, we propose UpSafe$^\circ$C, a unified framework for enhancing LLM safety through safety-aware upcycling. Our approach first identifies safety-critical layers and upcycles them into a sparse Mixture-of-Experts (MoE) structure, where the router acts as a soft guardrail that selectively activates original MLPs and added safety experts. We further introduce a two-stage SFT strategy to strengthen safety discrimination while preserving general capabilities. To enable flexible control at inference time, we introduce a safety temperature mechanism, allowing dynamic adjustment of the trade-off between safety and utility. Experiments across multiple benchmarks, base model, and model scales demonstrate that UpSafe$^\circ$C achieves robust safety improvements against harmful and jailbreak inputs, while maintaining competitive performance on general tasks. Moreover, analysis shows that safety temperature provides fine-grained inference-time control that achieves the Pareto-optimal frontier between utility and safety. Our results highlight a new direction for LLM safety: moving from static alignment toward dynamic, modular, and inference-aware control. 

---
# MetaboT: AI-based agent for natural language-based interaction with metabolomics knowledge graphs 

**Authors**: Madina Bekbergenova, Lucas Pradi, Benjamin Navet, Emma Tysinger, Franck Michel, Matthieu Feraud, Yousouf Taghzouti, Yan Zhou Chen, Olivier Kirchhoffer, Florence Mehl, Martin Legrand, Tao Jiang, Marco Pagni, Soha Hassoun, Jean-Luc Wolfender, Wout Bittremieux, Fabien Gandon, Louis-Félix Nothias  

**Link**: [PDF](https://arxiv.org/pdf/2510.01724)  

**Abstract**: Mass spectrometry metabolomics generates vast amounts of data requiring advanced methods for interpretation. Knowledge graphs address these challenges by structuring mass spectrometry data, metabolite information, and their relationships into a connected network (Gaudry et al. 2024). However, effective use of a knowledge graph demands an in-depth understanding of its ontology and its query language syntax. To overcome this, we designed MetaboT, an AI system utilizing large language models (LLMs) to translate user questions into SPARQL semantic query language for operating on knowledge graphs (Steve Harris 2013). We demonstrate its effectiveness using the Experimental Natural Products Knowledge Graph (ENPKG), a large-scale public knowledge graph for plant natural products (Gaudry et al. 2024).MetaboT employs specialized AI agents for handling user queries and interacting with the knowledge graph by breaking down complex tasks into discrete components, each managed by a specialised agent (Fig. 1a). The multi-agent system is constructed using the LangChain and LangGraph libraries, which facilitate the integration of LLMs with external tools and information sources (LangChain, n.d.). The query generation process follows a structured workflow. First, the Entry Agent determines if the question is new or a follow-up to previous interactions. New questions are forwarded to the Validator Agent, which verifies if the question is related to the knowledge graph. Then, the valid question is sent to the Supervisor Agent, which identifies if the question requires chemical conversions or standardized identifiers. In this case it delegates the question to the Knowledge Graph Agent, which can use tools to extract necessary details, such as URIs or taxonomies of chemical names, from the user query. Finally, an agent responsible for crafting the SPARQL queries equipped with the ontology of the knowledge graph uses the provided identifiers to generate the query. Then, the system executes the generated query against the metabolomics knowledge graph and returns structured results to the user (Fig. 1b). To assess the performance of MetaboT we have curated 50 metabolomics-related questions and their expected answers. In addition to submitting these questions to MetaboT, we evaluated a baseline by submitting them to a standard LLM (GPT-4o) with a prompt that incorporated the knowledge graph ontology but did not provide specific entity IDs. This baseline achieved only 8.16% accuracy, compared to MetaboT's 83.67%, underscoring the necessity of our multi-agent system for accurately retrieving entities and generating correct SPARQL queries. MetaboT demonstrates promising performance as a conversational question-answering assistant, enabling researchers to retrieve structured metabolomics data through natural language queries. By automating the generation and execution of SPARQL queries, it removes technical barriers that have traditionally hindered access to knowledge graphs. Importantly, MetaboT leverages the capabilities of LLMs while maintaining experimentally grounded query generation, ensuring that outputs remain aligned with domain-specific standards and data structures. This approach facilitates data-driven discoveries by bridging the gap between complex semantic technologies and user-friendly interaction. MetaboT is accessible at [this https URL], and its source code is available at [this https URL]. 

---
# VaPR -- Vision-language Preference alignment for Reasoning 

**Authors**: Rohan Wadhawan, Fabrice Y Harel-Canada, Zi-Yi Dou, Suhaila Shakiah, Robinson Piramuthu, Nanyun Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.01700)  

**Abstract**: Preference finetuning methods like Direct Preference Optimization (DPO) with AI-generated feedback have shown promise in aligning Large Vision-Language Models (LVLMs) with human preferences. However, existing techniques overlook the prevalence of noise in synthetic preference annotations in the form of stylistic and length biases. To this end, we introduce a hard-negative response generation framework based on LLM-guided response editing, that produces rejected responses with targeted errors, maintaining stylistic and length similarity to the accepted ones. Using this framework, we develop the VaPR dataset, comprising 30K high-quality samples, to finetune three LVLM families: LLaVA-V1.5, Qwen2VL & Qwen2.5VL (2B-13B sizes). Our VaPR models deliver significant performance improvements across ten benchmarks, achieving average gains of 6.5% (LLaVA), 4.0% (Qwen2VL), and 1.5% (Qwen2.5VL), with notable improvements on reasoning tasks. A scaling analysis shows that performance consistently improves with data size, with LLaVA models benefiting even at smaller scales. Moreover, VaPR reduces the tendency to answer "Yes" in binary questions - addressing a common failure mode in LVLMs like LLaVA. Lastly, we show that the framework generalizes to open-source LLMs as editors, with models trained on VaPR-OS achieving ~99% of the performance of models trained on \name, which is synthesized using GPT-4o. Our data, models, and code can be found on the project page this https URL 

---
# To Mask or to Mirror: Human-AI Alignment in Collective Reasoning 

**Authors**: Crystal Qian, Aaron Parisi, Clémentine Bouleau, Vivian Tsai, Maël Lebreton, Lucas Dixon  

**Link**: [PDF](https://arxiv.org/pdf/2510.01924)  

**Abstract**: As large language models (LLMs) are increasingly used to model and augment collective decision-making, it is critical to examine their alignment with human social reasoning. We present an empirical framework for assessing collective alignment, in contrast to prior work on the individual level. Using the Lost at Sea social psychology task, we conduct a large-scale online experiment (N=748), randomly assigning groups to leader elections with either visible demographic attributes (e.g. name, gender) or pseudonymous aliases. We then simulate matched LLM groups conditioned on the human data, benchmarking Gemini 2.5, GPT 4.1, Claude Haiku 3.5, and Gemma 3. LLM behaviors diverge: some mirror human biases; others mask these biases and attempt to compensate for them. We empirically demonstrate that human-AI alignment in collective reasoning depends on context, cues, and model-specific inductive biases. Understanding how LLMs align with collective human behavior is critical to advancing socially-aligned AI, and demands dynamic benchmarks that capture the complexities of collective reasoning. 

---
# AgentRec: Next-Generation LLM-Powered Multi-Agent Collaborative Recommendation with Adaptive Intelligence 

**Authors**: Bo Ma, Hang Li, ZeHua Hu, XiaoFan Gui, LuYao Liu, Simon Lau  

**Link**: [PDF](https://arxiv.org/pdf/2510.01609)  

**Abstract**: Interactive conversational recommender systems have gained significant attention for their ability to capture user preferences through natural language interactions. However, existing approaches face substantial challenges in handling dynamic user preferences, maintaining conversation coherence, and balancing multiple ranking objectives simultaneously. This paper introduces AgentRec, a next-generation LLM-powered multi-agent collaborative recommendation framework that addresses these limitations through hierarchical agent networks with adaptive intelligence. Our approach employs specialized LLM-powered agents for conversation understanding, preference modeling, context awareness, and dynamic ranking, coordinated through an adaptive weighting mechanism that learns from interaction patterns. We propose a three-tier learning strategy combining rapid response for simple queries, intelligent reasoning for complex preferences, and deep collaboration for challenging scenarios. Extensive experiments on three real-world datasets demonstrate that AgentRec achieves consistent improvements over state-of-the-art baselines, with 2.8\% enhancement in conversation success rate, 1.9\% improvement in recommendation accuracy (NDCG@10), and 3.2\% better conversation efficiency while maintaining comparable computational costs through intelligent agent coordination. 

---
# Understanding the Geospatial Reasoning Capabilities of LLMs: A Trajectory Recovery Perspective 

**Authors**: Thinh Hung Truong, Jey Han Lau, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2510.01639)  

**Abstract**: We explore the geospatial reasoning capabilities of Large Language Models (LLMs), specifically, whether LLMs can read road network maps and perform navigation. We frame trajectory recovery as a proxy task, which requires models to reconstruct masked GPS traces, and introduce GLOBALTRACE, a dataset with over 4,000 real-world trajectories across diverse regions and transportation modes. Using road network as context, our prompting framework enables LLMs to generate valid paths without accessing any external navigation tools. Experiments show that LLMs outperform off-the-shelf baselines and specialized trajectory recovery models, with strong zero-shot generalization. Fine-grained analysis shows that LLMs have strong comprehension of the road network and coordinate systems, but also pose systematic biases with respect to regions and transportation modes. Finally, we demonstrate how LLMs can enhance navigation experiences by reasoning over maps in flexible ways to incorporate user preferences. 

---
# LOGicalThought: Logic-Based Ontological Grounding of LLMs for High-Assurance Reasoning 

**Authors**: Navapat Nananukul, Yue Zhang, Ryan Lee, Eric Boxer, Jonathan May, Vibhav Giridhar Gogate, Jay Pujara, Mayank Kejriwal  

**Link**: [PDF](https://arxiv.org/pdf/2510.01530)  

**Abstract**: High-assurance reasoning, particularly in critical domains such as law and medicine, requires conclusions that are accurate, verifiable, and explicitly grounded in evidence. This reasoning relies on premises codified from rules, statutes, and contracts, inherently involving defeasible or non-monotonic logic due to numerous exceptions, where the introduction of a single fact can invalidate general rules, posing significant challenges. While large language models (LLMs) excel at processing natural language, their capabilities in standard inference tasks do not translate to the rigorous reasoning required over high-assurance text guidelines. Core reasoning challenges within such texts often manifest specific logical structures involving negation, implication, and, most critically, defeasible rules and exceptions. In this paper, we propose a novel neurosymbolically-grounded architecture called LOGicalThought (LogT) that uses an advanced logical language and reasoner in conjunction with an LLM to construct a dual symbolic graph context and logic-based context. These two context representations transform the problem from inference over long-form guidelines into a compact grounded evaluation. Evaluated on four multi-domain benchmarks against four baselines, LogT improves overall performance by 11.84% across all LLMs. Performance improves significantly across all three modes of reasoning: by up to +10.2% on negation, +13.2% on implication, and +5.5% on defeasible reasoning compared to the strongest baseline. 

---
# REBot: From RAG to CatRAG with Semantic Enrichment and Graph Routing 

**Authors**: Thanh Ma, Tri-Tam La, Lam-Thu Le Huu, Minh-Nghi Nguyen, Khanh-Van Pham Luu, Huu-Hoa Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2510.01800)  

**Abstract**: Academic regulation advising is essential for helping students interpret and comply with institutional policies, yet building effective systems requires domain specific regulatory resources. To address this challenge, we propose REBot, an LLM enhanced advisory chatbot powered by CatRAG, a hybrid retrieval reasoning framework that integrates retrieval augmented generation with graph based reasoning. CatRAG unifies dense retrieval and graph reasoning, supported by a hierarchical, category labeled knowledge graph enriched with semantic features for domain alignment. A lightweight intent classifier routes queries to the appropriate retrieval modules, ensuring both factual accuracy and contextual depth. We construct a regulation specific dataset and evaluate REBot on classification and question answering tasks, achieving state of the art performance with an F1 score of 98.89%. Finally, we implement a web application that demonstrates the practical value of REBot in real world academic advising scenarios. 

---
# AIReg-Bench: Benchmarking Language Models That Assess AI Regulation Compliance 

**Authors**: Bill Marino, Rosco Hunter, Zubair Jamali, Marinos Emmanouil Kalpakos, Mudra Kashyap, Isaiah Hinton, Alexa Hanson, Maahum Nazir, Christoph Schnabl, Felix Steffek, Hongkai Wen, Nicholas D. Lane  

**Link**: [PDF](https://arxiv.org/pdf/2510.01474)  

**Abstract**: As governments move to regulate AI, there is growing interest in using Large Language Models (LLMs) to assess whether or not an AI system complies with a given AI Regulation (AIR). However, there is presently no way to benchmark the performance of LLMs at this task. To fill this void, we introduce AIReg-Bench: the first benchmark dataset designed to test how well LLMs can assess compliance with the EU AI Act (AIA). We created this dataset through a two-step process: (1) by prompting an LLM with carefully structured instructions, we generated 120 technical documentation excerpts (samples), each depicting a fictional, albeit plausible, AI system - of the kind an AI provider might produce to demonstrate their compliance with AIR; (2) legal experts then reviewed and annotated each sample to indicate whether, and in what way, the AI system described therein violates specific Articles of the AIA. The resulting dataset, together with our evaluation of whether frontier LLMs can reproduce the experts' compliance labels, provides a starting point to understand the opportunities and limitations of LLM-based AIR compliance assessment tools and establishes a benchmark against which subsequent LLMs can be compared. The dataset and evaluation code are available at this https URL. 

---
# Learning to Decide with Just Enough: Information-Theoretic Context Summarization for CDMPs 

**Authors**: Peidong Liu, Junjiang Lin, Shaowen Wang, Yao Xu, Haiqing Li, Xuhao Xie, Siyi Wu, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2510.01620)  

**Abstract**: Contextual Markov Decision Processes (CMDPs) offer a framework for sequential decision-making under external signals, but existing methods often fail to generalize in high-dimensional or unstructured contexts, resulting in excessive computation and unstable performance. We propose an information-theoretic summarization approach that uses large language models (LLMs) to compress contextual inputs into low-dimensional, semantically rich summaries. These summaries augment states by preserving decision-critical cues while reducing redundancy. Building on the notion of approximate context sufficiency, we provide, to our knowledge, the first regret bounds and a latency-entropy trade-off characterization for CMDPs. Our analysis clarifies how informativeness impacts computational cost. Experiments across discrete, continuous, visual, and recommendation benchmarks show that our method outperforms raw-context and non-context baselines, improving reward, success rate, and sample efficiency, while reducing latency and memory usage. These findings demonstrate that LLM-based summarization offers a scalable and interpretable solution for efficient decision-making in context-rich, resource-constrained environments. 

---
# OntoLogX: Ontology-Guided Knowledge Graph Extraction from Cybersecurity Logs with Large Language Models 

**Authors**: Luca Cotti, Idilio Drago, Anisa Rula, Devis Bianchini, Federico Cerutti  

**Link**: [PDF](https://arxiv.org/pdf/2510.01409)  

**Abstract**: System logs represent a valuable source of Cyber Threat Intelligence (CTI), capturing attacker behaviors, exploited vulnerabilities, and traces of malicious activity. Yet their utility is often limited by lack of structure, semantic inconsistency, and fragmentation across devices and sessions. Extracting actionable CTI from logs therefore requires approaches that can reconcile noisy, heterogeneous data into coherent and interoperable representations. We introduce OntoLogX, an autonomous Artificial Intelligence (AI) agent that leverages Large Language Models (LLMs) to transform raw logs into ontology-grounded Knowledge Graphs (KGs). OntoLogX integrates a lightweight log ontology with Retrieval Augmented Generation (RAG) and iterative correction steps, ensuring that generated KGs are syntactically and semantically valid. Beyond event-level analysis, the system aggregates KGs into sessions and employs a LLM to predict MITRE ATT&CK tactics, linking low-level log evidence to higher-level adversarial objectives. We evaluate OntoLogX on both logs from a public benchmark and a real-world honeypot dataset, demonstrating robust KG generation across multiple KGs backends and accurate mapping of adversarial activity to ATT&CK tactics. Results highlight the benefits of retrieval and correction for precision and recall, the effectiveness of code-oriented models in structured log analysis, and the value of ontology-grounded representations for actionable CTI extraction. 

---
# A Tale of LLMs and Induced Small Proxies: Scalable Agents for Knowledge Mining 

**Authors**: Sipeng Zhang, Longfei Yun, Zilong Wang, Jingbo Shang, Letian Peng  

**Link**: [PDF](https://arxiv.org/pdf/2510.01427)  

**Abstract**: At the core of Deep Research is knowledge mining, the task of extracting structured information from massive unstructured text in response to user instructions. Large language models (LLMs) excel at interpreting such instructions but are prohibitively expensive to deploy at scale, while traditional pipelines of classifiers and extractors remain efficient yet brittle and unable to generalize to new tasks. We introduce Falconer, a collaborative framework that combines the agentic reasoning of LLMs with lightweight proxy models for scalable knowledge mining. In Falconer, LLMs act as planners, decomposing user instructions into executable pipelines, and as annotators, generating supervision to train small proxies. The framework unifies classification and extraction into two atomic operations, get label and get span, enabling a single instruction-following model to replace multiple task-specific components. To evaluate the consistency between proxy models incubated by Falconer and annotations provided by humans and large models, we construct new benchmarks covering both planning and end-to-end execution. Experiments show that Falconer closely matches state-of-the-art LLMs in instruction-following accuracy while reducing inference cost by up to 90% and accelerating large-scale knowledge mining by more than 20x, offering an efficient and scalable foundation for Deep Research. 

---
# Automating Data-Driven Modeling and Analysis for Engineering Applications using Large Language Model Agents 

**Authors**: Yang Liu, Zaid Abulawi, Abhiram Garimidi, Doyeong Lim  

**Link**: [PDF](https://arxiv.org/pdf/2510.01398)  

**Abstract**: Modern engineering increasingly relies on vast datasets generated by experiments and simulations, driving a growing demand for efficient, reliable, and broadly applicable modeling strategies. There is also heightened interest in developing data-driven approaches, particularly neural network models, for effective prediction and analysis of scientific datasets. Traditional data-driven methods frequently involve extensive manual intervention, limiting their ability to scale effectively and generalize to diverse applications. In this study, we propose an innovative pipeline utilizing Large Language Model (LLM) agents to automate data-driven modeling and analysis, with a particular emphasis on regression tasks. We evaluate two LLM-agent frameworks: a multi-agent system featuring specialized collaborative agents, and a single-agent system based on the Reasoning and Acting (ReAct) paradigm. Both frameworks autonomously handle data preprocessing, neural network development, training, hyperparameter optimization, and uncertainty quantification (UQ). We validate our approach using a critical heat flux (CHF) prediction benchmark, involving approximately 25,000 experimental data points from the OECD/NEA benchmark dataset. Results indicate that our LLM-agent-developed model surpasses traditional CHF lookup tables and delivers predictive accuracy and UQ on par with state-of-the-art Bayesian optimized deep neural network models developed by human experts. These outcomes underscore the significant potential of LLM-based agents to automate complex engineering modeling tasks, greatly reducing human workload while meeting or exceeding existing standards of predictive performance. 

---
# Retrieval-Augmented Framework for LLM-Based Clinical Decision Support 

**Authors**: Leon Garza, Anantaa Kotal, Michael A. Grasso, Emre Umucu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01363)  

**Abstract**: The increasing complexity of clinical decision-making, alongside the rapid expansion of electronic health records (EHR), presents both opportunities and challenges for delivering data-informed care. This paper proposes a clinical decision support system powered by Large Language Models (LLMs) to assist prescribing clinicians. The system generates therapeutic suggestions by analyzing historical EHR data, including patient demographics, presenting complaints, clinical symptoms, diagnostic information, and treatment histories. The framework integrates natural language processing with structured clinical inputs to produce contextually relevant recommendations. Rather than replacing clinician judgment, it is designed to augment decision-making by retrieving and synthesizing precedent cases with comparable characteristics, drawing on local datasets or federated sources where applicable. At its core, the system employs a retrieval-augmented generation (RAG) pipeline that harmonizes unstructured narratives and codified data to support LLM-based inference. We outline the system's technical components, including representation representation alignment and generation strategies. Preliminary evaluations, conducted with de-identified and synthetic clinical datasets, examine the clinical plausibility and consistency of the model's outputs. Early findings suggest that LLM-based tools may provide valuable decision support in prescribing workflows when appropriately constrained and rigorously validated. This work represents an initial step toward integration of generative AI into real-world clinical decision-making with an emphasis on transparency, safety, and alignment with established practices. 

---
# Step-Aware Policy Optimization for Reasoning in Diffusion Large Language Models 

**Authors**: Shaoan Xie, Lingjing Kong, Xiangchen Song, Xinshuai Dong, Guangyi Chen, Eric P.Xing, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01544)  

**Abstract**: Diffusion language models (dLLMs) offer a promising, non-autoregressive paradigm for text generation, yet training them for complex reasoning remains a key challenge. Current reinforcement learning approaches often rely on sparse, outcome-based rewards, which can reinforce flawed reasoning paths that lead to coincidentally correct answers. We argue that this stems from a fundamental mismatch with the natural structure of reasoning. We first propose a theoretical framework that formalizes complex problem solving as a hierarchical selection process, where an intractable global constraint is decomposed into a series of simpler, localized logical steps. This framework provides a principled foundation for algorithm design, including theoretical insights into the identifiability of this latent reasoning structure. Motivated by this theory, we identify unstructured refinement -- a failure mode where a model's iterative steps do not contribute meaningfully to the solution -- as a core deficiency in existing methods. We then introduce Step-Aware Policy Optimization (SAPO), a novel RL algorithm that aligns the dLLM's denoising process with the latent reasoning hierarchy. By using a process-based reward function that encourages incremental progress, SAPO guides the model to learn structured, coherent reasoning paths. Our empirical results show that this principled approach significantly improves performance on challenging reasoning benchmarks and enhances the interpretability of the generation process. 

---
# OR-Toolformer: Modeling and Solving Operations Research Problems with Tool Augmented Large Language Models 

**Authors**: Jianzhang Zhang, Jialong Zhou, Chuang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01253)  

**Abstract**: Large language models (LLMs) demonstrate strong mathematical reasoning, but reliance on closed-source APIs for OR tasks raises privacy concerns, and training open-source models from scratch incurs high compute costs. We introduce OR-Toolformer, which fine-tunes Llama-3.1-8B-Instruct with a semi-automatic data synthesis pipeline that generates diverse OR problem-answer pairs and augments the model with external solvers to produce API calls. On three of four standard benchmarks, OR-Toolformer achieves up to 80.1% execution accuracy, exceeding size-matched baselines by over 4.3%. In zero-shot evaluation on two unseen OR problem types, it attains 54% average accuracy, a 21 percentage-point improvement over the strongest baseline. These findings validate the efficacy of tool-augmented fine-tuning LLMs for accurate and generalizable OR problem modeling and solving. 

---
# The Social Laboratory: A Psychometric Framework for Multi-Agent LLM Evaluation 

**Authors**: Zarreen Reza  

**Link**: [PDF](https://arxiv.org/pdf/2510.01295)  

**Abstract**: As Large Language Models (LLMs) transition from static tools to autonomous agents, traditional evaluation benchmarks that measure performance on downstream tasks are becoming insufficient. These methods fail to capture the emergent social and cognitive dynamics that arise when agents communicate, persuade, and collaborate in interactive environments. To address this gap, we introduce a novel evaluation framework that uses multi-agent debate as a controlled "social laboratory" to discover and quantify these behaviors. In our framework, LLM-based agents, instantiated with distinct personas and incentives, deliberate on a wide range of challenging topics under the supervision of an LLM moderator. Our analysis, enabled by a new suite of psychometric and semantic metrics, reveals several key findings. Across hundreds of debates, we uncover a powerful and robust emergent tendency for agents to seek consensus, consistently reaching high semantic agreement ({\mu} > 0.88) even without explicit instruction and across sensitive topics. We show that assigned personas induce stable, measurable psychometric profiles, particularly in cognitive effort, and that the moderators persona can significantly alter debate outcomes by structuring the environment, a key finding for external AI alignment. This work provides a blueprint for a new class of dynamic, psychometrically grounded evaluation protocols designed for the agentic setting, offering a crucial methodology for understanding and shaping the social behaviors of the next generation of AI agents. We have released the code and results at this https URL. 

---
# Modeling Others' Minds as Code 

**Authors**: Kunal Jha, Aydan Yuenan Huang, Eric Ye, Natasha Jaques, Max Kleiman-Weiner  

**Link**: [PDF](https://arxiv.org/pdf/2510.01272)  

**Abstract**: Accurate prediction of human behavior is essential for robust and safe human-AI collaboration. However, existing approaches for modeling people are often data-hungry and brittle because they either make unrealistic assumptions about rationality or are too computationally demanding to adapt rapidly. Our key insight is that many everyday social interactions may follow predictable patterns; efficient "scripts" that minimize cognitive load for actors and observers, e.g., "wait for the green light, then go." We propose modeling these routines as behavioral programs instantiated in computer code rather than policies conditioned on beliefs and desires. We introduce ROTE, a novel algorithm that leverages both large language models (LLMs) for synthesizing a hypothesis space of behavioral programs, and probabilistic inference for reasoning about uncertainty over that space. We test ROTE in a suite of gridworld tasks and a large-scale embodied household simulator. ROTE predicts human and AI behaviors from sparse observations, outperforming competitive baselines -- including behavior cloning and LLM-based methods -- by as much as 50% in terms of in-sample accuracy and out-of-sample generalization. By treating action understanding as a program synthesis problem, ROTE opens a path for AI systems to efficiently and effectively predict human behavior in the real-world. 

---
# Addressing Pitfalls in the Evaluation of Uncertainty Estimation Methods for Natural Language Generation 

**Authors**: Mykyta Ielanskyi, Kajetan Schweighofer, Lukas Aichberger, Sepp Hochreiter  

**Link**: [PDF](https://arxiv.org/pdf/2510.02279)  

**Abstract**: Hallucinations are a common issue that undermine the reliability of large language models (LLMs). Recent studies have identified a specific subset of hallucinations, known as confabulations, which arise due to predictive uncertainty of LLMs. To detect confabulations, various methods for estimating predictive uncertainty in natural language generation (NLG) have been developed. These methods are typically evaluated by correlating uncertainty estimates with the correctness of generated text, with question-answering (QA) datasets serving as the standard benchmark. However, commonly used approximate correctness functions have substantial disagreement between each other and, consequently, in the ranking of the uncertainty estimation methods. This allows one to inflate the apparent performance of uncertainty estimation methods. We propose using several alternative risk indicators for risk correlation experiments that improve robustness of empirical assessment of UE algorithms for NLG. For QA tasks, we show that marginalizing over multiple LLM-as-a-judge variants leads to reducing the evaluation biases. Furthermore, we explore structured tasks as well as out of distribution and perturbation detection tasks which provide robust and controllable risk indicators. Finally, we propose to use an Elo rating of uncertainty estimation methods to give an objective summarization over extensive evaluation settings. 

---
# RewardMap: Tackling Sparse Rewards in Fine-grained Visual Reasoning via Multi-Stage Reinforcement Learning 

**Authors**: Sicheng Feng, Kaiwen Tuo, Song Wang, Lingdong Kong, Jianke Zhu, Huan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2510.02240)  

**Abstract**: Fine-grained visual reasoning remains a core challenge for multimodal large language models (MLLMs). The recently introduced ReasonMap highlights this gap by showing that even advanced MLLMs struggle with spatial reasoning in structured and information-rich settings such as transit maps, a task of clear practical and scientific importance. However, standard reinforcement learning (RL) on such tasks is impeded by sparse rewards and unstable optimization. To address this, we first construct ReasonMap-Plus, an extended dataset that introduces dense reward signals through Visual Question Answering (VQA) tasks, enabling effective cold-start training of fine-grained visual understanding skills. Next, we propose RewardMap, a multi-stage RL framework designed to improve both visual understanding and reasoning capabilities of MLLMs. RewardMap incorporates two key designs. First, we introduce a difficulty-aware reward design that incorporates detail rewards, directly tackling the sparse rewards while providing richer supervision. Second, we propose a multi-stage RL scheme that bootstraps training from simple perception to complex reasoning tasks, offering a more effective cold-start strategy than conventional Supervised Fine-Tuning (SFT). Experiments on ReasonMap and ReasonMap-Plus demonstrate that each component of RewardMap contributes to consistent performance gains, while their combination yields the best results. Moreover, models trained with RewardMap achieve an average improvement of 3.47% across 6 benchmarks spanning spatial reasoning, fine-grained visual reasoning, and general tasks beyond transit maps, underscoring enhanced visual understanding and reasoning capabilities. 

---
# DiFFPO: Training Diffusion LLMs to Reason Fast and Furious via Reinforcement Learning 

**Authors**: Hanyang Zhao, Dawen Liang, Wenpin Tang, David Yao, Nathan Kallus  

**Link**: [PDF](https://arxiv.org/pdf/2510.02212)  

**Abstract**: We propose DiFFPO, Diffusion Fast and Furious Policy Optimization, a unified framework for training masked diffusion large language models (dLLMs) to reason not only better (furious), but also faster via reinforcement learning (RL). We first unify the existing baseline approach such as d1 by proposing to train surrogate policies via off-policy RL, whose likelihood is much more tractable as an approximation to the true dLLM policy. This naturally motivates a more accurate and informative two-stage likelihood approximation combined with importance sampling correction, which leads to generalized RL algorithms with better sample efficiency and superior task performance. Second, we propose a new direction of joint training efficient samplers/controllers of dLLMs policy. Via RL, we incentivize dLLMs' natural multi-token prediction capabilities by letting the model learn to adaptively allocate an inference threshold for each prompt. By jointly training the sampler, we yield better accuracies with lower number of function evaluations (NFEs) compared to training the model only, obtaining the best performance in improving the Pareto frontier of the inference-time compute of dLLMs. We showcase the effectiveness of our pipeline by training open source large diffusion language models over benchmark math and planning tasks. 

---
# Towards Interpretable and Inference-Optimal COT Reasoning with Sparse Autoencoder-Guided Generation 

**Authors**: Daniel Zhao, Abhilash Shankarampeta, Lanxiang Hu, Tajana Rosing, Hao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2510.01528)  

**Abstract**: We propose a novel method that leverages sparse autoencoders (SAEs) and clustering techniques to analyze the internal token representations of large language models (LLMs) and guide generations in mathematical reasoning tasks. Our approach first trains an SAE to generate sparse vector representations for training tokens, then applies k-means clustering to construct a graph where vertices represent token clusters and weighted edges capture sequential token transitions. Using this graph, we define an edge-weight based reward function to quantify adherence to established reasoning traces, thereby identifying exploitative reasoning trajectories. Additionally, we measure generation diversity from clustering to assess the extent of exploration. Our findings indicate that balancing both exploitation and exploration is crucial for achieving high accuracy in mathematical reasoning tasks. During generation, the SAE can serve as a scalable reward model to guide generations, ensuring a balanced trade-off between exploitation and exploration. This prevents extreme behaviors in either direction, ultimately fostering a higher-quality reasoning process in LLMs. 

---
# BioinfoMCP: A Unified Platform Enabling MCP Interfaces in Agentic Bioinformatics 

**Authors**: Florensia Widjaja, Zhangtianyi Chen, Juexiao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2510.02139)  

**Abstract**: Bioinformatics tools are essential for complex computational biology tasks, yet their integration with emerging AI-agent frameworks is hindered by incompatible interfaces, heterogeneous input-output formats, and inconsistent parameter conventions. The Model Context Protocol (MCP) provides a standardized framework for tool-AI communication, but manually converting hundreds of existing and rapidly growing specialized bioinformatics tools into MCP-compliant servers is labor-intensive and unsustainable. Here, we present BioinfoMCP, a unified platform comprising two components: BioinfoMCP Converter, which automatically generates robust MCP servers from tool documentation using large language models, and BioinfoMCP Benchmark, which systematically validates the reliability and versatility of converted tools across diverse computational tasks. We present a platform of 38 MCP-converted bioinformatics tools, extensively validated to show that 94.7% successfully executed complex workflows across three widely used AI-agent platforms. By removing technical barriers to AI automation, BioinfoMCP enables natural-language interaction with sophisticated bioinformatics analyses without requiring extensive programming expertise, offering a scalable path to intelligent, interoperable computational biology. 

---
# GRACE: A Language Model Framework for Explainable Inverse Reinforcement Learning 

**Authors**: Silvia Sapora, Devon Hjelm, Alexander Toshev, Omar Attia, Bogdan Mazoure  

**Link**: [PDF](https://arxiv.org/pdf/2510.02180)  

**Abstract**: Inverse Reinforcement Learning aims to recover reward models from expert demonstrations, but traditional methods yield "black-box" models that are difficult to interpret and debug. In this work, we introduce GRACE (Generating Rewards As CodE), a method for using Large Language Models within an evolutionary search to reverse-engineer an interpretable, code-based reward function directly from expert trajectories. The resulting reward function is executable code that can be inspected and verified. We empirically validate GRACE on the BabyAI and AndroidWorld benchmarks, where it efficiently learns highly accurate rewards, even in complex, multi-task settings. Further, we demonstrate that the resulting reward leads to strong policies, compared to both competitive Imitation Learning and online RL approaches with ground-truth rewards. Finally, we show that GRACE is able to build complex reward APIs in multi-task setups. 

---
# Asymmetric Proximal Policy Optimization: mini-critics boost LLM reasoning 

**Authors**: Jiashun Liu, Johan Obando-Ceron, Han Lu, Yancheng He, Weixun Wang, Wenbo Su, Bo Zheng, Pablo Samuel Castro, Aaron Courville, Ling Pan  

**Link**: [PDF](https://arxiv.org/pdf/2510.01656)  

**Abstract**: Most recent RL for LLMs (RL4LLM) methods avoid explicit critics, replacing them with average advantage baselines. This shift is largely pragmatic: conventional value functions are computationally expensive to train at LLM scale and often fail under sparse rewards and long reasoning horizons. We revisit this bottleneck from an architectural perspective and introduce Asymmetric Proximal Policy Optimization (AsyPPO), a simple and scalable framework that restores the critics role while remaining efficient in large-model settings. AsyPPO employs a set of lightweight mini-critics, each trained on disjoint prompt shards. This design encourages diversity while preserving calibration, reducing value-estimation bias. Beyond robust estimation, AsyPPO leverages inter-critic uncertainty to refine the policy update: (i) masking advantages in states where critics agree and gradients add little learning signal, and (ii) filtering high-divergence states from entropy regularization, suppressing spurious exploration. After training on open-source data with only 5,000 samples, AsyPPO consistently improves learning stability and performance across multiple benchmarks over strong baselines, such as GRPO, achieving performance gains of more than six percent on Qwen3-4b-Base and about three percent on Qwen3-8b-Base and Qwen3-14b-Base over classic PPO, without additional tricks. These results highlight the importance of architectural innovations for scalable, efficient algorithms. 

---
# NLP Methods for Detecting Novel LLM Jailbreaks and Keyword Analysis with BERT 

**Authors**: John Hawkins, Aditya Pramar, Rodney Beard, Rohitash Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2510.01644)  

**Abstract**: Large Language Models (LLMs) suffer from a range of vulnerabilities that allow malicious users to solicit undesirable responses through manipulation of the input text. These so-called jailbreak prompts are designed to trick the LLM into circumventing the safety guardrails put in place to keep responses acceptable to the developer's policies. In this study, we analyse the ability of different machine learning models to distinguish jailbreak prompts from genuine uses, including looking at our ability to identify jailbreaks that use previously unseen strategies. Our results indicate that using current datasets the best performance is achieved by fine tuning a Bidirectional Encoder Representations from Transformers (BERT) model end-to-end for identifying jailbreaks. We visualise the keywords that distinguish jailbreak from genuine prompts and conclude that explicit reflexivity in prompt structure could be a signal of jailbreak intention. 

---
# TACOS: Task Agnostic COordinator of a multi-drone System 

**Authors**: Alessandro Nazzari, Roberto Rubinacci, Marco Lovera  

**Link**: [PDF](https://arxiv.org/pdf/2510.01869)  

**Abstract**: When a single pilot is responsible for managing a multi-drone system, the task demands varying levels of autonomy, from direct control of individual UAVs, to group-level coordination, to fully autonomous swarm behaviors for accomplishing high-level tasks. Enabling such flexible interaction requires a framework that supports multiple modes of shared autonomy. As language models continue to improve in reasoning and planning, they provide a natural foundation for such systems, reducing pilot workload by enabling high-level task delegation through intuitive, language-based interfaces. In this paper we present TACOS (Task-Agnostic COordinator of a multi-drone System), a unified framework that enables high-level natural language control of multi-UAV systems through Large Language Models (LLMs). TACOS integrates three key capabilities into a single architecture: a one-to-many natural language interface for intuitive user interaction, an intelligent coordinator for translating user intent into structured task plans, and an autonomous agent that executes plans interacting with the real-world. TACOS allows a LLM to interact with a library of executable APIs, bridging semantic reasoning with real-time multi-robot coordination. We demonstrate the system in real-world multi-drone system and conduct an ablation study to assess the contribution of each module. 

---
# Guiding Multimodal Large Language Models with Blind and Low Vision People Visual Questions for Proactive Visual Interpretations 

**Authors**: Ricardo Gonzalez Penuela, Felipe Arias-Russi, Victor Capriles  

**Link**: [PDF](https://arxiv.org/pdf/2510.01576)  

**Abstract**: Multimodal large language models (MLLMs) have been integrated into visual interpretation applications to support Blind and Low Vision (BLV) users because of their accuracy and ability to provide rich, human-like interpretations. However, these applications often default to comprehensive, lengthy descriptions regardless of context. This leads to inefficient exchanges, as users must go through irrelevant details rather than receiving the specific information they are likely to seek. To deliver more contextually-relevant information, we developed a system that draws on historical BLV users questions. When given an image, our system identifies similar past visual contexts from the VizWiz-LF dataset and uses the associated questions to guide the MLLM generate descriptions more relevant to BLV users. An evaluation with three human labelers who revised 92 context-aware and context-free descriptions showed that context-aware descriptions anticipated and answered users' questions in 76.1% of cases (70 out of 92) and were preferred in 54.4% of comparisons (50 out of 92). Our paper reviews, and data analysis are publicly available in a Github repository at this https URL . 

---
# POLAR: Automating Cyber Threat Prioritization through LLM-Powered Assessment 

**Authors**: Luoxi Tang, Yuqiao Meng, Ankita Patra, Weicheng Ma, Muchao Ye, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2510.01552)  

**Abstract**: Large Language Models (LLMs) are intensively used to assist security analysts in counteracting the rapid exploitation of cyber threats, wherein LLMs offer cyber threat intelligence (CTI) to support vulnerability assessment and incident response. While recent work has shown that LLMs can support a wide range of CTI tasks such as threat analysis, vulnerability detection, and intrusion defense, significant performance gaps persist in practical deployments. In this paper, we investigate the intrinsic vulnerabilities of LLMs in CTI, focusing on challenges that arise from the nature of the threat landscape itself rather than the model architecture. Using large-scale evaluations across multiple CTI benchmarks and real-world threat reports, we introduce a novel categorization methodology that integrates stratification, autoregressive refinement, and human-in-the-loop supervision to reliably analyze failure instances. Through extensive experiments and human inspections, we reveal three fundamental vulnerabilities: spurious correlations, contradictory knowledge, and constrained generalization, that limit LLMs in effectively supporting CTI. Subsequently, we provide actionable insights for designing more robust LLM-powered CTI systems to facilitate future research. 

---
# BioVERSE: Representation Alignment of Biomedical Modalities to LLMs for Multi-Modal Reasoning 

**Authors**: Ching-Huei Tsou, Michal Ozery-Flato, Ella Barkan, Diwakar Mahajan, Ben Shapira  

**Link**: [PDF](https://arxiv.org/pdf/2510.01428)  

**Abstract**: Recent advances in large language models (LLMs) and biomedical foundation models (BioFMs) have achieved strong results in biological text reasoning, molecular modeling, and single-cell analysis, yet they remain siloed in disjoint embedding spaces, limiting cross-modal reasoning. We present BIOVERSE (Biomedical Vector Embedding Realignment for Semantic Engagement), a two-stage approach that adapts pretrained BioFMs as modality encoders and aligns them with LLMs through lightweight, modality-specific projection layers. The approach first aligns each modality to a shared LLM space through independently trained projections, allowing them to interoperate naturally, and then applies standard instruction tuning with multi-modal data to bring them together for downstream reasoning. By unifying raw biomedical data with knowledge embedded in LLMs, the approach enables zero-shot annotation, cross-modal question answering, and interactive, explainable dialogue. Across tasks spanning cell-type annotation, molecular description, and protein function reasoning, compact BIOVERSE configurations surpass larger LLM baselines while enabling richer, generative outputs than existing BioFMs, establishing a foundation for principled multi-modal biomedical reasoning. 

---
# Enhancing the development of Cherenkov Telescope Array control software with Large Language Models 

**Authors**: Dmitriy Kostunin, Elisa Jones, Vladimir Sotnikov, Valery Sotnikov, Sergo Golovachev, Alexandre Strube  

**Link**: [PDF](https://arxiv.org/pdf/2510.01299)  

**Abstract**: We develop AI agents based on instruction-finetuned large language models (LLMs) to assist in the engineering and operation of the Cherenkov Telescope Array Observatory (CTAO) Control and Data Acquisition Software (ACADA). These agents align with project-specific documentation and codebases, understand contextual information, interact with external APIs, and communicate with users in natural language. We present our progress in integrating these features into CTAO pipelines for operations and offline data analysis. 

---
# Beyond Majority Voting: LLM Aggregation by Leveraging Higher-Order Information 

**Authors**: Rui Ai, Yuqi Pan, David Simchi-Levi, Milind Tambe, Haifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01499)  

**Abstract**: With the rapid progress of multi-agent large language model (LLM) reasoning, how to effectively aggregate answers from multiple LLMs has emerged as a fundamental challenge. Standard majority voting treats all answers equally, failing to consider latent heterogeneity and correlation across models. In this work, we design two new aggregation algorithms called Optimal Weight (OW) and Inverse Surprising Popularity (ISP), leveraging both first-order and second-order information. Our theoretical analysis shows these methods provably mitigate inherent limitations of majority voting under mild assumptions, leading to more reliable collective decisions. We empirically validate our algorithms on synthetic datasets, popular LLM fine-tuning benchmarks such as UltraFeedback and MMLU, and a real-world healthcare setting ARMMAN. Across all cases, our methods consistently outperform majority voting, offering both practical performance gains and conceptual insights for the design of robust multi-agent LLM pipelines. 

---
# Microsaccade-Inspired Probing: Positional Encoding Perturbations Reveal LLM Misbehaviours 

**Authors**: Rui Melo, Rui Abreu, Corina S. Pasareanu  

**Link**: [PDF](https://arxiv.org/pdf/2510.01288)  

**Abstract**: We draw inspiration from microsaccades, tiny involuntary eye movements that reveal hidden dynamics of human perception, to propose an analogous probing method for large language models (LLMs). Just as microsaccades expose subtle but informative shifts in vision, we show that lightweight position encoding perturbations elicit latent signals that indicate model misbehaviour. Our method requires no fine-tuning or task-specific supervision, yet detects failures across diverse settings including factuality, safety, toxicity, and backdoor attacks. Experiments on multiple state-of-the-art LLMs demonstrate that these perturbation-based probes surface misbehaviours while remaining computationally efficient. These findings suggest that pretrained LLMs already encode the internal evidence needed to flag their own failures, and that microsaccade-inspired interventions provide a pathway for detecting and mitigating undesirable behaviours. 

---
# IoT-MCP: Bridging LLMs and IoT Systems Through Model Context Protocol 

**Authors**: Ningyuan Yang, Guanliang Lyu, Mingchen Ma, Yiyi Lu, Yiming Li, Zhihui Gao, Hancheng Ye, Jianyi Zhang, Tingjun Chen, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2510.01260)  

**Abstract**: The integration of Large Language Models (LLMs) with Internet-of-Things (IoT) systems faces significant challenges in hardware heterogeneity and control complexity. The Model Context Protocol (MCP) emerges as a critical enabler, providing standardized communication between LLMs and physical devices. We propose IoT-MCP, a novel framework that implements MCP through edge-deployed servers to bridge LLMs and IoT ecosystems. To support rigorous evaluation, we introduce IoT-MCP Bench, the first benchmark containing 114 Basic Tasks (e.g., ``What is the current temperature?'') and 1,140 Complex Tasks (e.g., ``I feel so hot, do you have any ideas?'') for IoT-enabled LLMs. Experimental validation across 22 sensor types and 6 microcontroller units demonstrates IoT-MCP's 100% task success rate to generate tool calls that fully meet expectations and obtain completely accurate results, 205ms average response time, and 74KB peak memory footprint. This work delivers both an open-source integration framework (this https URL) and a standardized evaluation methodology for LLM-IoT systems. 

---
# Breaking the Code: Security Assessment of AI Code Agents Through Systematic Jailbreaking Attacks 

**Authors**: Shoumik Saha, Jifan Chen, Sam Mayers, Sanjay Krishna Gouda, Zijian Wang, Varun Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2510.01359)  

**Abstract**: Code-capable large language model (LLM) agents are increasingly embedded into software engineering workflows where they can read, write, and execute code, raising the stakes of safety-bypass ("jailbreak") attacks beyond text-only settings. Prior evaluations emphasize refusal or harmful-text detection, leaving open whether agents actually compile and run malicious programs. We present JAWS-BENCH (Jailbreaks Across WorkSpaces), a benchmark spanning three escalating workspace regimes that mirror attacker capability: empty (JAWS-0), single-file (JAWS-1), and multi-file (JAWS-M). We pair this with a hierarchical, executable-aware Judge Framework that tests (i) compliance, (ii) attack success, (iii) syntactic correctness, and (iv) runtime executability, moving beyond refusal to measure deployable harm. Using seven LLMs from five families as backends, we find that under prompt-only conditions in JAWS-0, code agents accept 61% of attacks on average; 58% are harmful, 52% parse, and 27% run end-to-end. Moving to single-file regime in JAWS-1 drives compliance to ~ 100% for capable models and yields a mean ASR (Attack Success Rate) ~ 71%; the multi-file regime (JAWS-M) raises mean ASR to ~ 75%, with 32% instantly deployable attack code. Across models, wrapping an LLM in an agent substantially increases vulnerability -- ASR raises by 1.6x -- because initial refusals are frequently overturned during later planning/tool-use steps. Category-level analyses identify which attack classes are most vulnerable and most readily deployable, while others exhibit large execution gaps. These findings motivate execution-aware defenses, code-contextual safety filters, and mechanisms that preserve refusal decisions throughout the agent's multi-step reasoning and tool use. 

---
# Mamba Outpaces Reformer in Stock Prediction with Sentiments from Top Ten LLMs 

**Authors**: Lokesh Antony Kadiyala, Amir Mirzaeinia  

**Link**: [PDF](https://arxiv.org/pdf/2510.01203)  

**Abstract**: The stock market is extremely difficult to predict in the short term due to high market volatility, changes caused by news, and the non-linear nature of the financial time series. This research proposes a novel framework for improving minute-level prediction accuracy using semantic sentiment scores from top ten different large language models (LLMs) combined with minute interval intraday stock price data. We systematically constructed a time-aligned dataset of AAPL news articles and 1-minute Apple Inc. (AAPL) stock prices for the dates of April 4 to May 2, 2025. The sentiment analysis was achieved using the DeepSeek-V3, GPT variants, LLaMA, Claude, Gemini, Qwen, and Mistral models through their APIs. Each article obtained sentiment scores from all ten LLMs, which were scaled to a [0, 1] range and combined with prices and technical indicators like RSI, ROC, and Bollinger Band Width. Two state-of-the-art such as Reformer and Mamba were trained separately on the dataset using the sentiment scores produced by each LLM as input. Hyper parameters were optimized by means of Optuna and were evaluated through a 3-day evaluation period. Reformer had mean squared error (MSE) or the evaluation metrics, and it should be noted that Mamba performed not only faster but also better than Reformer for every LLM across the 10 LLMs tested. Mamba performed best with LLaMA 3.3--70B, with the lowest error of 0.137. While Reformer could capture broader trends within the data, the model appeared to over smooth sudden changes by the LLMs. This study highlights the potential of integrating LLM-based semantic analysis paired with efficient temporal modeling to enhance real-time financial forecasting. 

---
# An Anthropologist LLM to Elicit Users' Moral Preferences through Role-Play 

**Authors**: Gianluca De Ninno, Paola Inverardi, Francesca Belotti  

**Link**: [PDF](https://arxiv.org/pdf/2510.01189)  

**Abstract**: This study investigates a novel approach to eliciting users' moral decision-making by combining immersive roleplaying games with LLM analysis capabilities. Building on the distinction introduced by Floridi between hard ethics inspiring and shaping laws-and soft ethics-moral preferences guiding individual behavior within the free space of decisions compliant to laws-we focus on capturing the latter through contextrich, narrative-driven interactions. Grounded in anthropological methods, the role-playing game exposes participants to ethically charged scenarios in the domain of digital privacy. Data collected during the sessions were interpreted by a customized LLM ("GPT Anthropologist"). Evaluation through a cross-validation process shows that both the richness of the data and the interpretive framing significantly enhance the model's ability to predict user behavior. Results show that LLMs can be effectively employed to automate and enhance the understanding of user moral preferences and decision-making process in the early stages of software development. 

---
