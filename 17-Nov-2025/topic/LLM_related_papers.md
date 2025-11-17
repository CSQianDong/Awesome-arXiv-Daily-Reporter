# Information Extraction From Fiscal Documents Using LLMs 

**Authors**: Vikram Aggarwal, Jay Kulkarni, Aditi Mascarenhas, Aakriti Narang, Siddarth Raman, Ajay Shah, Susan Thomas  

**Link**: [PDF](https://arxiv.org/pdf/2511.10659)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable capabilities in text comprehension, but their ability to process complex, hierarchical tabular data remains underexplored. We present a novel approach to extracting structured data from multi-page government fiscal documents using LLM-based techniques. Applied to annual fiscal documents from the State of Karnataka in India (200+ pages), our method achieves high accuracy through a multi-stage pipeline that leverages domain knowledge, sequential context, and algorithmic validation. A large challenge with traditional OCR methods is the inability to verify the accurate extraction of numbers. When applied to fiscal data, the inherent structure of fiscal tables, with totals at each level of the hierarchy, allows for robust internal validation of the extracted data. We use these hierarchical relationships to create multi-level validation checks. We demonstrate that LLMs can read tables and also process document-specific structural hierarchies, offering a scalable process for converting PDF-based fiscal disclosures into research-ready databases. Our implementation shows promise for broader applications across developing country contexts. 

---
# SRLF: An Agent-Driven Set-Wise Reflective Learning Framework for Sequential Recommendation 

**Authors**: Jiahao Wang, Bokang Fu, Yu Zhu, Yuli Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.11370)  

**Abstract**: LLM-based agents are emerging as a promising paradigm for simulating user behavior to enhance recommender systems. However, their effectiveness is often limited by existing studies that focus on modeling user ratings for individual items. This point-wise approach leads to prevalent issues such as inaccurate user preference comprehension and rigid item-semantic representations.
To address these limitations, we propose the novel Set-wise Reflective Learning Framework (SRLF). Our framework operationalizes a closed-loop "assess-validate-reflect" cycle that harnesses the powerful in-context learning capabilities of LLMs. SRLF departs from conventional point-wise assessment by formulating a holistic judgment on an entire set of items. It accomplishes this by comprehensively analyzing both the intricate interrelationships among items within the set and their collective alignment with the user's preference profile. This method of set-level contextual understanding allows our model to capture complex relational patterns essential to user behavior, making it significantly more adept for sequential recommendation. Extensive experiments validate our approach, confirming that this set-wise perspective is crucial for achieving state-of-the-art performance in sequential recommendation tasks. 

---
# Align$^3$GR: Unified Multi-Level Alignment for LLM-based Generative Recommendation 

**Authors**: Wencai Ye, Mingjie Sun, Shuhang Chen, Wenjin Wu, Peng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2511.11255)  

**Abstract**: Large Language Models (LLMs) demonstrate significant advantages in leveraging structured world knowledge and multi-step reasoning capabilities. However, fundamental challenges arise when transforming LLMs into real-world recommender systems due to semantic and behavioral misalignment. To bridge this gap, we propose Align$^3$GR, a novel framework that unifies token-level, behavior modeling-level, and preference-level alignment. Our approach introduces: Dual tokenization fusing user-item semantic and collaborative signals. Enhanced behavior modeling with bidirectional semantic alignment. Progressive DPO strategy combining self-play (SP-DPO) and real-world feedback (RF-DPO) for dynamic preference adaptation. Experiments show Align$^3$GR outperforms the SOTA baseline by +17.8% in Recall@10 and +20.2% in NDCG@10 on the public dataset, with significant gains in online A/B tests and full-scale deployment on an industrial large-scale recommendation platform. 

---
# Experience-Guided Adaptation of Inference-Time Reasoning Strategies 

**Authors**: Adam Stein, Matthew Trager, Benjamin Bowman, Michael Kleinman, Aditya Chattopadhyay, Wei Xia, Stefano Soatto  

**Link**: [PDF](https://arxiv.org/pdf/2511.11519)  

**Abstract**: Enabling agentic AI systems to adapt their problem-solving approaches based on post-training interactions remains a fundamental challenge. While systems that update and maintain a memory at inference time have been proposed, existing designs only steer the system by modifying textual input to a language model or agent, which means that they cannot change sampling parameters, remove tools, modify system prompts, or switch between agentic and workflow paradigms. On the other hand, systems that adapt more flexibly require offline optimization and remain static once deployed. We present Experience-Guided Reasoner (EGuR), which generates tailored strategies -- complete computational procedures involving LLM calls, tools, sampling parameters, and control logic -- dynamically at inference time based on accumulated experience. We achieve this using an LLM-based meta-strategy -- a strategy that outputs strategies -- enabling adaptation of all strategy components (prompts, sampling parameters, tool configurations, and control logic). EGuR operates through two components: a Guide generates multiple candidate strategies conditioned on the current problem and structured memory of past experiences, while a Consolidator integrates execution feedback to improve future strategy generation. This produces complete, ready-to-run strategies optimized for each problem, which can be cached, retrieved, and executed as needed without wasting resources. Across five challenging benchmarks (AIME 2025, 3-SAT, and three Big Bench Extra Hard tasks), EGuR achieves up to 14% accuracy improvements over the strongest baselines while reducing computational costs by up to 111x, with both metrics improving as the system gains experience. 

---
# AIonopedia: an LLM agent orchestrating multimodal learning for ionic liquid discovery 

**Authors**: Yuqi Yin, Yibo Fu, Siyuan Wang, Peng Sun, Hongyu Wang, Xiaohui Wang, Lei Zheng, Zhiyong Li, Zhirong Liu, Jianji Wang, Zhaoxi Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.11257)  

**Abstract**: The discovery of novel Ionic Liquids (ILs) is hindered by critical challenges in property prediction, including limited data, poor model accuracy, and fragmented workflows. Leveraging the power of Large Language Models (LLMs), we introduce AIonopedia, to the best of our knowledge, the first LLM agent for IL discovery. Powered by an LLM-augmented multimodal domain foundation model for ILs, AIonopedia enables accurate property predictions and incorporates a hierarchical search architecture for molecular screening and design. Trained and evaluated on a newly curated and comprehensive IL dataset, our model delivers superior performance. Complementing these results, evaluations on literature-reported systems indicate that the agent can perform effective IL modification. Moving beyond offline tests, the practical efficacy was further confirmed through real-world wet-lab validation, in which the agent demonstrated exceptional generalization capabilities on challenging out-of-distribution tasks, underscoring its ability to accelerate real-world IL discovery. 

---
# CURENet: Combining Unified Representations for Efficient Chronic Disease Prediction 

**Authors**: Cong-Tinh Dao, Nguyen Minh Thao Phan, Jun-En Ding, Chenwei Wu, David Restrepo, Dongsheng Luo, Fanyi Zhao, Chun-Chieh Liao, Wen-Chih Peng, Chi-Te Wang, Pei-Fu Chen, Ling Chen, Xinglong Ju, Feng Liu, Fang-Ming Hung  

**Link**: [PDF](https://arxiv.org/pdf/2511.11423)  

**Abstract**: Electronic health records (EHRs) are designed to synthesize diverse data types, including unstructured clinical notes, structured lab tests, and time-series visit data. Physicians draw on these multimodal and temporal sources of EHR data to form a comprehensive view of a patient's health, which is crucial for informed therapeutic decision-making. Yet, most predictive models fail to fully capture the interactions, redundancies, and temporal patterns across multiple data modalities, often focusing on a single data type or overlooking these complexities. In this paper, we present CURENet, a multimodal model (Combining Unified Representations for Efficient chronic disease prediction) that integrates unstructured clinical notes, lab tests, and patients' time-series data by utilizing large language models (LLMs) for clinical text processing and textual lab tests, as well as transformer encoders for longitudinal sequential visits. CURENet has been capable of capturing the intricate interaction between different forms of clinical data and creating a more reliable predictive model for chronic illnesses. We evaluated CURENet using the public MIMIC-III and private FEMH datasets, where it achieved over 94\% accuracy in predicting the top 10 chronic conditions in a multi-label framework. Our findings highlight the potential of multimodal EHR integration to enhance clinical decision-making and improve patient outcomes. 

---
# STaR: Towards Cognitive Table Reasoning via Slow-Thinking Large Language Models 

**Authors**: Huajian Zhang, Mingyue Cheng, Yucong Luo, Xiaoyu Tao  

**Link**: [PDF](https://arxiv.org/pdf/2511.11233)  

**Abstract**: Table reasoning with the large language models (LLMs) is a fundamental path toward building intelligent systems that can understand and analyze over structured data. While recent progress has shown promising results, they still suffer from two key limitations: (i) the reasoning processes lack the depth and iterative refinement characteristic of human cognition; and (ii) the reasoning processes exhibit instability, which compromises their reliability in downstream applications. In this work, we present STaR (slow-thinking for table reasoning), a new framework achieving cognitive table reasoning, in which LLMs are equipped with slow-thinking capabilities by explicitly modeling step-by-step thinking and uncertainty-aware inference. During training, STaR employs two-stage difficulty-aware reinforcement learning (DRL), progressively learning from simple to complex queries under a composite reward. During inference, STaR performs trajectory-level uncertainty quantification by integrating token-level confidence and answer consistency, enabling selection of more credible reasoning paths. Extensive experiments on benchmarks demonstrate that STaR achieves superior performance and enhanced reasoning stability. Moreover, strong generalization over out-of-domain datasets further demonstrates STaR's potential as a reliable and cognitively inspired solution for table reasoning with LLMs. 

---
# UAVBench: An Open Benchmark Dataset for Autonomous and Agentic AI UAV Systems via LLM-Generated Flight Scenarios 

**Authors**: Mohamed Amine Ferrag, Abderrahmane Lakas, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2511.11252)  

**Abstract**: Autonomous aerial systems increasingly rely on large language models (LLMs) for mission planning, perception, and decision-making, yet the lack of standardized and physically grounded benchmarks limits systematic evaluation of their reasoning capabilities. To address this gap, we introduce UAVBench, an open benchmark dataset comprising 50,000 validated UAV flight scenarios generated through taxonomy-guided LLM prompting and multi-stage safety validation. Each scenario is encoded in a structured JSON schema that includes mission objectives, vehicle configuration, environmental conditions, and quantitative risk labels, providing a unified representation of UAV operations across diverse domains. Building on this foundation, we present UAVBench_MCQ, a reasoning-oriented extension containing 50,000 multiple-choice questions spanning ten cognitive and ethical reasoning styles, ranging from aerodynamics and navigation to multi-agent coordination and integrated reasoning. This framework enables interpretable and machine-checkable assessment of UAV-specific cognition under realistic operational contexts. We evaluate 32 state-of-the-art LLMs, including GPT-5, ChatGPT-4o, Gemini 2.5 Flash, DeepSeek V3, Qwen3 235B, and ERNIE 4.5 300B, and find strong performance in perception and policy reasoning but persistent challenges in ethics-aware and resource-constrained decision-making. UAVBench establishes a reproducible and physically grounded foundation for benchmarking agentic AI in autonomous aerial systems and advancing next-generation UAV reasoning intelligence. To support open science and reproducibility, we release the UAVBench dataset, the UAVBench_MCQ benchmark, evaluation scripts, and all related materials on GitHub at this https URL 

---
# Multi-agent Undercover Gaming: Hallucination Removal via Counterfactual Test for Multimodal Reasoning 

**Authors**: Dayong Liang, Xiao-Yong Wei, Changmeng Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2511.11182)  

**Abstract**: Hallucination continues to pose a major obstacle in the reasoning capabilities of large language models (LLMs). Although the Multi-Agent Debate (MAD) paradigm offers a promising solution by promoting consensus among multiple agents to enhance reliability, it relies on the unrealistic assumption that all debaters are rational and reflective, which is a condition that may not hold when agents themselves are prone to hallucinations. To address this gap, we introduce the Multi-agent Undercover Gaming (MUG) protocol, inspired by social deduction games like "Who is Undercover?". MUG reframes MAD as a process of detecting "undercover" agents (those suffering from hallucinations) by employing multimodal counterfactual tests. Specifically, we modify reference images to introduce counterfactual evidence and observe whether agents can accurately identify these changes, providing ground-truth for identifying hallucinating agents and enabling robust, crowd-powered multimodal reasoning. MUG advances MAD protocols along three key dimensions: (1) enabling factual verification beyond statistical consensus through counterfactual testing; (2) introducing cross-evidence reasoning via dynamically modified evidence sources instead of relying on static inputs; and (3) fostering active reasoning, where agents engage in probing discussions rather than passively answering questions. Collectively, these innovations offer a more reliable and effective framework for multimodal reasoning in LLMs. The source code can be accessed at this https URL. 

---
# AI Agent-Driven Framework for Automated Product Knowledge Graph Construction in E-Commerce 

**Authors**: Dimitar Peshevski, Riste Stojanov, Dimitar Trajanov  

**Link**: [PDF](https://arxiv.org/pdf/2511.11017)  

**Abstract**: The rapid expansion of e-commerce platforms generates vast amounts of unstructured product data, creating significant challenges for information retrieval, recommendation systems, and data analytics. Knowledge Graphs (KGs) offer a structured, interpretable format to organize such data, yet constructing product-specific KGs remains a complex and manual process. This paper introduces a fully automated, AI agent-driven framework for constructing product knowledge graphs directly from unstructured product descriptions. Leveraging Large Language Models (LLMs), our method operates in three stages using dedicated agents: ontology creation and expansion, ontology refinement, and knowledge graph population. This agent-based approach ensures semantic coherence, scalability, and high-quality output without relying on predefined schemas or handcrafted extraction rules. We evaluate the system on a real-world dataset of air conditioner product descriptions, demonstrating strong performance in both ontology generation and KG population. The framework achieves over 97\% property coverage and minimal redundancy, validating its effectiveness and practical applicability. Our work highlights the potential of LLMs to automate structured knowledge extraction in retail, providing a scalable path toward intelligent product data integration and utilization. 

---
# LLM enhanced graph inference for long-term disease progression modelling 

**Authors**: Tiantian He, An Zhao, Elinor Thompson, Anna Schroder, Ahmed Abdulaal, Frederik Barkhof, Daniel C. Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2511.10890)  

**Abstract**: Understanding the interactions between biomarkers among brain regions during neurodegenerative disease is essential for unravelling the mechanisms underlying disease progression. For example, pathophysiological models of Alzheimer's Disease (AD) typically describe how variables, such as regional levels of toxic proteins, interact spatiotemporally within a dynamical system driven by an underlying biological substrate, often based on brain connectivity. However, current methods grossly oversimplify the complex relationship between brain connectivity by assuming a single-modality brain connectome as the disease-spreading substrate. This leads to inaccurate predictions of pathology spread, especially during the long-term progression period. Meanhwile, other methods of learning such a graph in a purely data-driven way face the identifiability issue due to lack of proper constraint. We thus present a novel framework that uses Large Language Models (LLMs) as expert guides on the interaction of regional variables to enhance learning of disease progression from irregularly sampled longitudinal patient data. By leveraging LLMs' ability to synthesize multi-modal relationships and incorporate diverse disease-driving mechanisms, our method simultaneously optimizes 1) the construction of long-term disease trajectories from individual-level observations and 2) the biologically-constrained graph structure that captures interactions among brain regions with better identifiability. We demonstrate the new approach by estimating the pathology propagation using tau-PET imaging data from an Alzheimer's disease cohort. The new framework demonstrates superior prediction accuracy and interpretability compared to traditional approaches while revealing additional disease-driving factors beyond conventional connectivity measures. 

---
# From Efficiency to Adaptivity: A Deeper Look at Adaptive Reasoning in Large Language Models 

**Authors**: Chao Wu, Baoheng Li, Mingchen Gao, Zhenyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10788)  

**Abstract**: Recent advances in large language models (LLMs) have made reasoning a central benchmark for evaluating intelligence. While prior surveys focus on efficiency by examining how to shorten reasoning chains or reduce computation, this view overlooks a fundamental challenge: current LLMs apply uniform reasoning strategies regardless of task complexity, generating long traces for trivial problems while failing to extend reasoning for difficult tasks. This survey reframes reasoning through the lens of {adaptivity}: the capability to allocate reasoning effort based on input characteristics such as difficulty and uncertainty. We make three contributions. First, we formalize deductive, inductive, and abductive reasoning within the LLM context, connecting these classical cognitive paradigms with their algorithmic realizations. Second, we formalize adaptive reasoning as a control-augmented policy optimization problem balancing task performance with computational cost, distinguishing learned policies from inference-time control mechanisms. Third, we propose a systematic taxonomy organizing existing methods into training-based approaches that internalize adaptivity through reinforcement learning, supervised fine-tuning, and learned controllers, and training-free approaches that achieve adaptivity through prompt conditioning, feedback-driven halting, and modular composition. This framework clarifies how different mechanisms realize adaptive reasoning in practice and enables systematic comparison across diverse strategies. We conclude by identifying open challenges in self-evaluation, meta-reasoning, and human-aligned reasoning control. 

---
# Key Decision-Makers in Multi-Agent Debates: Who Holds the Power? 

**Authors**: Qian Zhang, Yan Zheng, Jinyi Liu, Hebin Liang, Lanjun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.11040)  

**Abstract**: Recent studies on LLM agent scaling have highlighted the potential of Multi-Agent Debate (MAD) to enhance reasoning abilities. However, the critical aspect of role allocation strategies remains underexplored. In this study, we demonstrate that allocating roles with differing viewpoints to specific positions significantly impacts MAD's performance in reasoning tasks. Specifically, we find a novel role allocation strategy, "Truth Last", which can improve MAD performance by up to 22% in reasoning tasks. To address the issue of unknown truth in practical applications, we propose the Multi-Agent Debate Consistency (MADC) strategy, which systematically simulates and optimizes its core mechanisms. MADC incorporates path consistency to assess agreement among independent roles, simulating the role with the highest consistency score as the truth. We validated MADC across a range of LLMs (9 models), including the DeepSeek-R1 Distilled Models, on challenging reasoning tasks. MADC consistently demonstrated advanced performance, effectively overcoming MAD's performance bottlenecks and providing a crucial pathway for further improvements in LLM agent scaling. 

---
# HARNESS: Human-Agent Risk Navigation and Event Safety System for Proactive Hazard Forecasting in High-Risk DOE Environments 

**Authors**: Ran Elgedawy, Sanjay Das, Ethan Seefried, Gavin Wiggins, Ryan Burchfield, Dana Hewit, Sudarshan Srinivasan, Todd Thomas, Prasanna Balaprakash, Tirthankar Ghosal  

**Link**: [PDF](https://arxiv.org/pdf/2511.10810)  

**Abstract**: Operational safety at mission-critical work sites is a top priority given the complex and hazardous nature of daily tasks. This paper presents the Human-Agent Risk Navigation and Event Safety System (HARNESS), a modular AI framework designed to forecast hazardous events and analyze operational risks in U.S. Department of Energy (DOE) environments. HARNESS integrates Large Language Models (LLMs) with structured work data, historical event retrieval, and risk analysis to proactively identify potential hazards. A human-in-the-loop mechanism allows subject matter experts (SMEs) to refine predictions, creating an adaptive learning loop that enhances performance over time. By combining SME collaboration with iterative agentic reasoning, HARNESS improves the reliability and efficiency of predictive safety systems. Preliminary deployment shows promising results, with future work focusing on quantitative evaluation of accuracy, SME agreement, and decision latency reduction. 

---
# Human-AI collaborative autonomous synthesis with pulsed laser deposition for remote epitaxy 

**Authors**: Asraful Haque, Daniel T. Yimam, Jawad Chowdhury, Ralph Bulanadi, Ivan Vlassiouk, John Lasseter, Sujoy Ghosh, Christopher M. Rouleau, Kai Xiao, Yongtao Liu, Eva Zarkadoula, Rama K. Vasudevan, Sumner B. Harris  

**Link**: [PDF](https://arxiv.org/pdf/2511.11558)  

**Abstract**: Autonomous laboratories typically rely on data-driven decision-making, occasionally with human-in-the-loop oversight to inject domain expertise. Fully leveraging AI agents, however, requires tightly coupled, collaborative workflows spanning hypothesis generation, experimental planning, execution, and interpretation. To address this, we develop and deploy a human-AI collaborative (HAIC) workflow that integrates large language models for hypothesis generation and analysis, with collaborative policy updates driving autonomous pulsed laser deposition (PLD) experiments for remote epitaxy of BaTiO$_3$/graphene. HAIC accelerated the hypothesis formation and experimental design and efficiently mapped the growth space to graphene-damage. In situ Raman spectroscopy reveals that chemistry drives degradation while the highest energy plume components seed defects, identifying a low-O$_2$ pressure low-temperature synthesis window that preserves graphene but is incompatible with optimal BaTiO$_3$ growth. Thus, we show a two-step Ar/O$_2$ deposition is required to exfoliate ferroelectric BaTiO$_3$ while maintaining a monolayer graphene interlayer. HAIC stages human insight with AI reasoning between autonomous batches to drive rapid scientific progress, providing an evolution to many existing human-in-the-loop autonomous workflows. 

---
# M-DAIGT: A Shared Task on Multi-Domain Detection of AI-Generated Text 

**Authors**: Salima Lamsiyah, Saad Ezzini, Abdelkader El Mahdaouy, Hamza Alami, Abdessamad Benlahbib, Samir El Amrany, Salmane Chafik, Hicham Hammouchi  

**Link**: [PDF](https://arxiv.org/pdf/2511.11340)  

**Abstract**: The generation of highly fluent text by Large Language Models (LLMs) poses a significant challenge to information integrity and academic research. In this paper, we introduce the Multi-Domain Detection of AI-Generated Text (M-DAIGT) shared task, which focuses on detecting AI-generated text across multiple domains, particularly in news articles and academic writing. M-DAIGT comprises two binary classification subtasks: News Article Detection (NAD) (Subtask 1) and Academic Writing Detection (AWD) (Subtask 2). To support this task, we developed and released a new large-scale benchmark dataset of 30,000 samples, balanced between human-written and AI-generated texts. The AI-generated content was produced using a variety of modern LLMs (e.g., GPT-4, Claude) and diverse prompting strategies. A total of 46 unique teams registered for the shared task, of which four teams submitted final results. All four teams participated in both Subtask 1 and Subtask 2. We describe the methods employed by these participating teams and briefly discuss future directions for M-DAIGT. 

---
# KGQuest: Template-Driven QA Generation from Knowledge Graphs with LLM-Based Refinement 

**Authors**: Sania Nayab, Marco Simoni, Giulio Rossolini, Andrea Saracino  

**Link**: [PDF](https://arxiv.org/pdf/2511.11258)  

**Abstract**: The generation of questions and answers (QA) from knowledge graphs (KG) plays a crucial role in the development and testing of educational platforms, dissemination tools, and large language models (LLM). However, existing approaches often struggle with scalability, linguistic quality, and factual consistency. This paper presents a scalable and deterministic pipeline for generating natural language QA from KGs, with an additional refinement step using LLMs to further enhance linguistic quality. The approach first clusters KG triplets based on their relations, creating reusable templates through natural language rules derived from the entity types of objects and relations. A module then leverages LLMs to refine these templates, improving clarity and coherence while preserving factual accuracy. Finally, the instantiation of answer options is achieved through a selection strategy that introduces distractors from the KG. Our experiments demonstrate that this hybrid approach efficiently generates high-quality QA pairs, combining scalability with fluency and linguistic precision. 

---
# iMAD: Intelligent Multi-Agent Debate for Efficient and Accurate LLM Inference 

**Authors**: Wei Fan, JinYi Yoon, Bo Ji  

**Link**: [PDF](https://arxiv.org/pdf/2511.11306)  

**Abstract**: Large Language Model (LLM) agent systems have advanced rapidly, driven by their strong generalization in zero-shot settings. To further enhance reasoning and accuracy on complex tasks, Multi-Agent Debate (MAD) has emerged as a promising framework that engages multiple LLM agents in structured debates to encourage diverse reasoning. However, triggering MAD for every query is inefficient, as it incurs substantial computational (token) cost and may even degrade accuracy by overturning correct single-agent answers. To address these limitations, we propose intelligent Multi-Agent Debate (iMAD), a token-efficient framework that selectively triggers MAD only when it is likely to be beneficial (i.e., correcting an initially wrong answer). To achieve this goal, iMAD learns generalizable model behaviors to make accurate debate decisions. Specifically, iMAD first prompts a single agent to produce a structured self-critique response, from which we extract 41 interpretable linguistic and semantic features capturing hesitation cues. Then, iMAD uses a lightweight debate-decision classifier, trained using our proposed FocusCal loss, to determine whether to trigger MAD, enabling robust debate decisions without test dataset-specific tuning. Through extensive experiments using six (visual) question answering datasets against five competitive baselines, we have shown that iMAD significantly reduces token usage (by up to 92%) while also improving final answer accuracy (by up to 13.5%). 

---
# VIDEOP2R: Video Understanding from Perception to Reasoning 

**Authors**: Yifan Jiang, Yueying Wang, Rui Zhao, Toufiq Parag, Zhimin Chen, Zhenyu Liao, Jayakrishnan Unnikrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2511.11113)  

**Abstract**: Reinforcement fine-tuning (RFT), a two-stage framework consisting of supervised fine-tuning (SFT) and reinforcement learning (RL) has shown promising results on improving reasoning ability of large language models (LLMs). Yet extending RFT to large video language models (LVLMs) remains challenging. We propose VideoP2R, a novel process-aware video RFT framework that enhances video reasoning by modeling perception and reasoning as distinct processes. In the SFT stage, we develop a three-step pipeline to generate VideoP2R-CoT-162K, a high-quality, process-aware chain-of-thought (CoT) dataset for perception and reasoning. In the RL stage, we introduce a novel process-aware group relative policy optimization (PA-GRPO) algorithm that supplies separate rewards for perception and reasoning. Extensive experiments show that VideoP2R achieves state-of-the-art (SotA) performance on six out of seven video reasoning and understanding benchmarks. Ablation studies further confirm the effectiveness of our process-aware modeling and PA-GRPO and demonstrate that model's perception output is information-sufficient for downstream reasoning. 

---
# S2D-ALIGN: Shallow-to-Deep Auxiliary Learning for Anatomically-Grounded Radiology Report Generation 

**Authors**: Jiechao Gao, Chang Liu, Yuangang Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.11066)  

**Abstract**: Radiology Report Generation (RRG) aims to automatically generate diagnostic reports from radiology images. To achieve this, existing methods have leveraged the powerful cross-modal generation capabilities of Multimodal Large Language Models (MLLMs), primarily focusing on optimizing cross-modal alignment between radiographs and reports through Supervised Fine-Tuning (SFT). However, by only performing instance-level alignment with the image-text pairs, the standard SFT paradigm fails to establish anatomically-grounded alignment, where the templated nature of reports often leads to sub-optimal generation quality. To address this, we propose \textsc{S2D-Align}, a novel SFT paradigm that establishes anatomically-grounded alignment by leveraging auxiliary signals of varying granularities. \textsc{S2D-Align} implements a shallow-to-deep strategy, progressively enriching the alignment process: it begins with the coarse radiograph-report pairing, then introduces reference reports for instance-level guidance, and ultimately utilizes key phrases to ground the generation in specific anatomical details. To bridge the different alignment stages, we introduce a memory-based adapter that empowers feature sharing, thereby integrating coarse and fine-grained guidance. For evaluation, we conduct experiments on the public \textsc{MIMIC-CXR} and \textsc{IU X-Ray} benchmarks, where \textsc{S2D-Align} achieves state-of-the-art performance compared to existing methods. Ablation studies validate the effectiveness of our multi-stage, auxiliary-guided approach, highlighting a promising direction for enhancing grounding capabilities in complex, multi-modal generation tasks. 

---
# Automata-Based Steering of Large Language Models for Diverse Structured Generation 

**Authors**: Xiaokun Luan, Zeming Wei, Yihao Zhang, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2511.11018)  

**Abstract**: Large language models (LLMs) are increasingly tasked with generating structured outputs. While structured generation methods ensure validity, they often lack output diversity, a critical limitation that we confirm in our preliminary study. We propose a novel method to enhance diversity in automaton-based structured generation. Our approach utilizes automata traversal history to steer LLMs towards novel structural patterns. Evaluations show our method significantly improves structural and content diversity while maintaining comparable generation efficiency. Furthermore, we conduct a case study showcasing the effectiveness of our method in generating diverse test cases for testing open-source libraries. 

---
# DialogGraph-LLM: Graph-Informed LLMs for End-to-End Audio Dialogue Intent Recognition 

**Authors**: HongYu Liu, Junxin Li, Changxi Guo, Hao Chen, Yaqian Huang, Yifu Guo, Huan Yang, Lihua Cai  

**Link**: [PDF](https://arxiv.org/pdf/2511.11000)  

**Abstract**: Recognizing speaker intent in long audio dialogues among speakers has a wide range of applications, but is a non-trivial AI task due to complex inter-dependencies in speaker utterances and scarce annotated data. To address these challenges, an end-to-end framework, namely DialogGraph-LLM, is proposed in the current work. DialogGraph-LLM combines a novel Multi-Relational Dialogue Attention Network (MR-DAN) architecture with multimodal foundation models (e.g., Qwen2.5-Omni-7B) for direct acoustic-to-intent inference. An adaptive semi-supervised learning strategy is designed using LLM with a confidence-aware pseudo-label generation mechanism based on dual-threshold filtering using both global and class confidences, and an entropy-based sample selection process that prioritizes high-information unlabeled instances. Extensive evaluations on the proprietary MarketCalls corpus and the publicly available MIntRec 2.0 benchmark demonstrate DialogGraph-LLM's superiority over strong audio and text-driven baselines. The framework demonstrates strong performance and efficiency in intent recognition in real world scenario audio dialogues, proving its practical value for audio-rich domains with limited supervision. Our code is available at this https URL. 

---
# Automated Analysis of Learning Outcomes and Exam Questions Based on Bloom's Taxonomy 

**Authors**: Ramya Kumar, Dhruv Gulwani, Sonit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.10903)  

**Abstract**: This paper explores the automatic classification of exam questions and learning outcomes according to Bloom's Taxonomy. A small dataset of 600 sentences labeled with six cognitive categories - Knowledge, Comprehension, Application, Analysis, Synthesis, and Evaluation - was processed using traditional machine learning (ML) models (Naive Bayes, Logistic Regression, Support Vector Machines), recurrent neural network architectures (LSTM, BiLSTM, GRU, BiGRU), transformer-based models (BERT and RoBERTa), and large language models (OpenAI, Gemini, Ollama, Anthropic). Each model was evaluated under different preprocessing and augmentation strategies (for example, synonym replacement, word embeddings, etc.). Among traditional ML approaches, Support Vector Machines (SVM) with data augmentation achieved the best overall performance, reaching 94 percent accuracy, recall, and F1 scores with minimal overfitting. In contrast, the RNN models and BERT suffered from severe overfitting, while RoBERTa initially overcame it but began to show signs as training progressed. Finally, zero-shot evaluations of large language models (LLMs) indicated that OpenAI and Gemini performed best among the tested LLMs, achieving approximately 0.72-0.73 accuracy and comparable F1 scores. These findings highlight the challenges of training complex deep models on limited data and underscore the value of careful data augmentation and simpler algorithms (such as augmented SVM) for Bloom's Taxonomy classification. 

---
# A Multifaceted Analysis of Negative Bias in Large Language Models through the Lens of Parametric Knowledge 

**Authors**: Jongyoon Song, Sangwon Yu, Sungroh Yoon  

**Link**: [PDF](https://arxiv.org/pdf/2511.10881)  

**Abstract**: Negative bias refers to the tendency of large language models (LLMs) to excessively generate negative responses in binary decision tasks (e.g., yes-no question answering). Previous research has focused on detecting and addressing negative attention heads that induce negative bias. However, the underlying detailed factors influencing negative bias remain underexplored. In this paper, we demonstrate that LLMs exhibit format-level negative bias, meaning the prompt format more influences their responses than the semantics of the negative response. For the fine-grained study of the negative bias, we introduce a pipeline for constructing the evaluation set, which systematically categorizes the dataset into three subsets based on the model's parametric knowledge: correct, incorrect, and insufficient relevant knowledge. Through analysis of this evaluation set, we identify a shortcut behavior in which models tend to generate negative responses when they lack sufficient knowledge to answer a yes-no question, leading to negative bias. We further examine how negative bias changes under various prompting scenarios related to parametric knowledge. We observe that providing relevant context and offering an "I don't know" option generally reduces negative bias, whereas chain-of-thought prompting tends to amplify the bias. Finally, we demonstrate that the degree of negative bias can vary depending on the type of prompt, which influences the direction of the response. Our work reveals the various factors that influence negative bias, providing critical insights for mitigating it in LLMs. 

---
# Evaluating Large Language Models on Rare Disease Diagnosis: A Case Study using House M.D 

**Authors**: Arsh Gupta, Ajay Narayanan Sridhar, Bonam Mingole, Amulya Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2511.10912)  

**Abstract**: Large language models (LLMs) have demonstrated capabilities across diverse domains, yet their performance on rare disease diagnosis from narrative medical cases remains underexplored. We introduce a novel dataset of 176 symptom-diagnosis pairs extracted from House M.D., a medical television series validated for teaching rare disease recognition in medical education. We evaluate four state-of-the-art LLMs such as GPT 4o mini, GPT 5 mini, Gemini 2.5 Flash, and Gemini 2.5 Pro on narrative-based diagnostic reasoning tasks. Results show significant variation in performance, ranging from 16.48% to 38.64% accuracy, with newer model generations demonstrating a 2.3 times improvement. While all models face substantial challenges with rare disease diagnosis, the observed improvement across architectures suggests promising directions for future development. Our educationally validated benchmark establishes baseline performance metrics for narrative medical reasoning and provides a publicly accessible evaluation framework for advancing AI-assisted diagnosis research. 

---
# Leveraging Parameter Space Symmetries for Reasoning Skill Transfer in LLMs 

**Authors**: Stefan Horoi, Sangwoo Cho, Supriyo Chakraborty, Shi-Xiong Zhang, Sambit Sahu, Guy Wolf, Genta Indra Winata  

**Link**: [PDF](https://arxiv.org/pdf/2511.10850)  

**Abstract**: Task arithmetic is a powerful technique for transferring skills between Large Language Models (LLMs), but it often suffers from negative interference when models have diverged during training. We address this limitation by first aligning the models' parameter spaces, leveraging the inherent permutation, rotation, and scaling symmetries of Transformer architectures. We adapt parameter space alignment for modern Grouped-Query Attention (GQA) and SwiGLU layers, exploring both weight-based and activation-based approaches. Using this alignment-first strategy, we successfully transfer advanced reasoning skills to a non-reasoning model. Experiments on challenging reasoning benchmarks show that our method consistently outperforms standard task arithmetic. This work provides an effective approach for merging and transferring specialized skills across evolving LLM families, reducing redundant fine-tuning and enhancing model adaptability. 

---
# Expert-Guided Prompting and Retrieval-Augmented Generation for Emergency Medical Service Question Answering 

**Authors**: Xueren Ge, Sahil Murtaza, Anthony Cortez, Homa Alemzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2511.10900)  

**Abstract**: Large language models (LLMs) have shown promise in medical question answering, yet they often overlook the domain-specific expertise that professionals depend on, such as the clinical subject areas (e.g., trauma, airway) and the certification level (e.g., EMT, Paramedic). Existing approaches typically apply general-purpose prompting or retrieval strategies without leveraging this structured context, limiting performance in high-stakes settings. We address this gap with EMSQA, an 24.3K-question multiple-choice dataset spanning 10 clinical subject areas and 4 certification levels, accompanied by curated, subject area-aligned knowledge bases (40K documents and 2M tokens). Building on EMSQA, we introduce (i) Expert-CoT, a prompting strategy that conditions chain-of-thought (CoT) reasoning on specific clinical subject area and certification level, and (ii) ExpertRAG, a retrieval-augmented generation pipeline that grounds responses in subject area-aligned documents and real-world patient data. Experiments on 4 LLMs show that Expert-CoT improves up to 2.05% over vanilla CoT prompting. Additionally, combining Expert-CoT with ExpertRAG yields up to a 4.59% accuracy gain over standard RAG baselines. Notably, the 32B expertise-augmented LLMs pass all the computer-adaptive EMS certification simulation exams. 

---
# The Map of Misbelief: Tracing Intrinsic and Extrinsic Hallucinations Through Attention Patterns 

**Authors**: Elyes Hajji, Aymen Bouguerra, Fabio Arnez  

**Link**: [PDF](https://arxiv.org/pdf/2511.10837)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in safety-critical domains, yet remain susceptible to hallucinations. While prior works have proposed confidence representation methods for hallucination detection, most of these approaches rely on computationally expensive sampling strategies and often disregard the distinction between hallucination types. In this work, we introduce a principled evaluation framework that differentiates between extrinsic and intrinsic hallucination categories and evaluates detection performance across a suite of curated benchmarks. In addition, we leverage a recent attention-based uncertainty quantification algorithm and propose novel attention aggregation strategies that improve both interpretability and hallucination detection performance. Our experimental findings reveal that sampling-based methods like Semantic Entropy are effective for detecting extrinsic hallucinations but generally fail on intrinsic ones. In contrast, our method, which aggregates attention over input tokens, is better suited for intrinsic hallucinations. These insights provide new directions for aligning detection strategies with the nature of hallucination and highlight attention as a rich signal for quantifying model uncertainty. 

---
# Short-Window Sliding Learning for Real-Time Violence Detection via LLM-based Auto-Labeling 

**Authors**: Seoik Jung, Taekyung Song, Yangro Lee, Sungjun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2511.10866)  

**Abstract**: This paper proposes a Short-Window Sliding Learning framework for real-time violence detection in CCTV footages. Unlike conventional long-video training approaches, the proposed method divides videos into 1-2 second clips and applies Large Language Model (LLM)-based auto-caption labeling to construct fine-grained datasets. Each short clip fully utilizes all frames to preserve temporal continuity, enabling precise recognition of rapid violent events. Experiments demonstrate that the proposed method achieves 95.25\% accuracy on RWF-2000 and significantly improves performance on long videos (UCF-Crime: 83.25\%), confirming its strong generalization and real-time applicability in intelligent surveillance systems. 

---
# HPCAgentTester: A Multi-Agent LLM Approach for Enhanced HPC Unit Test Generation 

**Authors**: Rabimba Karanjai, Lei Xu, Weidong Shi  

**Link**: [PDF](https://arxiv.org/pdf/2511.10860)  

**Abstract**: Unit testing in High-Performance Computing (HPC) is critical but challenged by parallelism, complex algorithms, and diverse hardware. Traditional methods often fail to address non-deterministic behavior and synchronization issues in HPC applications. This paper introduces HPCAgentTester, a novel multi-agent Large Language Model (LLM) framework designed to automate and enhance unit test generation for HPC software utilizing OpenMP and MPI. HPCAgentTester employs a unique collaborative workflow where specialized LLM agents (Recipe Agent and Test Agent) iteratively generate and refine test cases through a critique loop. This architecture enables the generation of context-aware unit tests that specifically target parallel execution constructs, complex communication patterns, and hierarchical parallelism. We demonstrate HPCAgentTester's ability to produce compilable and functionally correct tests for OpenMP and MPI primitives, effectively identifying subtle bugs that are often missed by conventional techniques. Our evaluation shows that HPCAgentTester significantly improves test compilation rates and correctness compared to standalone LLMs, offering a more robust and scalable solution for ensuring the reliability of parallel software systems. 

---
# BadThink: Triggered Overthinking Attacks on Chain-of-Thought Reasoning in Large Language Models 

**Authors**: Shuaitong Liu, Renjue Li, Lijia Yu, Lijun Zhang, Zhiming Liu, Gaojie Jin  

**Link**: [PDF](https://arxiv.org/pdf/2511.10714)  

**Abstract**: Recent advances in Chain-of-Thought (CoT) prompting have substantially improved the reasoning capabilities of large language models (LLMs), but have also introduced their computational efficiency as a new attack surface. In this paper, we propose BadThink, the first backdoor attack designed to deliberately induce "overthinking" behavior in CoT-enabled LLMs while ensuring stealth. When activated by carefully crafted trigger prompts, BadThink manipulates the model to generate inflated reasoning traces - producing unnecessarily redundant thought processes while preserving the consistency of final outputs. This subtle attack vector creates a covert form of performance degradation that significantly increases computational costs and inference time while remaining difficult to detect through conventional output evaluation methods. We implement this attack through a sophisticated poisoning-based fine-tuning strategy, employing a novel LLM-based iterative optimization process to embed the behavior by generating highly naturalistic poisoned data. Our experiments on multiple state-of-the-art models and reasoning tasks show that BadThink consistently increases reasoning trace lengths - achieving an over 17x increase on the MATH-500 dataset - while remaining stealthy and robust. This work reveals a critical, previously unexplored vulnerability where reasoning efficiency can be covertly manipulated, demonstrating a new class of sophisticated attacks against CoT-enabled systems. 

---
# Bias-Restrained Prefix Representation Finetuning for Mathematical Reasoning 

**Authors**: Sirui Liang, Pengfei Cao, Jian Zhao, Cong Huang, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10707)  

**Abstract**: Parameter-Efficient finetuning (PEFT) enhances model performance on downstream tasks by updating a minimal subset of parameters. Representation finetuning (ReFT) methods further improve efficiency by freezing model weights and optimizing internal representations with fewer parameters than PEFT, outperforming PEFT on several tasks. However, ReFT exhibits a significant performance decline on mathematical reasoning tasks. To address this problem, the paper demonstrates that ReFT's poor performance on mathematical tasks primarily stems from its struggle to generate effective reasoning prefixes during the early inference phase. Moreover, ReFT disturbs the numerical encoding and the error accumulats during the CoT stage. Based on these observations, this paper proposes Bias-REstrained Prefix Representation FineTuning (BREP ReFT), which enhances ReFT's mathematical reasoning capability by truncating training data to optimize the generation of initial reasoning prefixes, intervening on the early inference stage to prevent error accumulation, and constraining the intervention vectors' magnitude to avoid disturbing numerical encoding. Extensive experiments across diverse model architectures demonstrate BREP's superior effectiveness, efficiency, and robust generalization capability, outperforming both standard ReFT and weight-based PEFT methods on the task of mathematical reasoning. The source code is available at this https URL. 

---
# Evaluating from Benign to Dynamic Adversarial: A Squid Game for Large Language Models 

**Authors**: Zijian Chen, Wenjun Zhang, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2511.10691)  

**Abstract**: Contemporary benchmarks are struggling to keep pace with the development of large language models (LLMs). Although they are indispensable to evaluate model performance on various tasks, it is uncertain whether the models trained on Internet data have genuinely learned how to solve problems or merely seen the questions before. This potential data contamination issue presents a fundamental challenge to establishing trustworthy evaluation frameworks. Meanwhile, existing benchmarks predominantly assume benign, resource-rich settings, leaving the behavior of LLMs under pressure unexplored. In this paper, we introduce Squid Game, a dynamic and adversarial evaluation environment with resource-constrained and asymmetric information settings elaborated to evaluate LLMs through interactive gameplay against other LLM opponents. Notably, Squid Game consists of six elimination-style levels, focusing on multi-faceted abilities, such as instruction-following, code, reasoning, planning, and safety alignment. We evaluate over 50 LLMs on Squid Game, presenting the largest behavioral evaluation study of general LLMs on dynamic adversarial scenarios. We observe a clear generational phase transition on performance in the same model lineage and find evidence that some models resort to speculative shortcuts to win the game, indicating the possibility of higher-level evaluation paradigm contamination in static benchmarks. Furthermore, we compare prominent LLM benchmarks and Squid Game with correlation analyses, highlighting that dynamic evaluation can serve as a complementary part for static evaluations. The code and data will be released in the future. 

---
# Who Gets the Reward, Who Gets the Blame? Evaluation-Aligned Training Signals for Multi-LLM Agents 

**Authors**: Chih-Hsuan Yang, Tanwi Mallick, Le Chen, Krishnan Raghavan, Azton Wells, Amal Gueroudji, Ian T. Foster, Rajeev Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2511.10687)  

**Abstract**: Large Language Models (LLMs) in multi-agent systems (MAS) have shown promise for complex tasks, yet current training methods lack principled ways to connect system-level evaluation with agent-level and message-level learning. We propose a theoretical framework that unifies cooperative game-theoretic attribution with process reward modeling to transform system evaluation into agent credit and then into response-level signals. Unlike prior approaches that rely only on attribution (e.g., Shapley) or step-level labels (e.g., PRM), our method produces local, signed, and credit-conserving signals. In success cases, Shapley-based credit assignment fairly allocates outcomes across agents and is refined into per-message rewards that promote cooperation while discouraging redundancy or sabotage. In failure cases, first-error localization yields repair-aware preferences that penalize harmful steps while rewarding corrective attempts. The resulting signals are bounded, cooperative, and directly compatible with reinforcement-based or preference-based post-training, providing a unified and auditable pathway from global evaluation to local supervision in LLM multi-agent training. Our contribution is conceptual: we present a theoretical foundation and training signals, leaving empirical validation for future work. 

---
# Evaluating LLM Understanding via Structured Tabular Decision Simulations 

**Authors**: Sichao Li, Xinyue Xu, Xiaomeng Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.10667)  

**Abstract**: Large language models (LLMs) often achieve impressive predictive accuracy, yet correctness alone does not imply genuine understanding. True LLM understanding, analogous to human expertise, requires making consistent, well-founded decisions across multiple instances and diverse domains, relying on relevant and domain-grounded decision factors. We introduce Structured Tabular Decision Simulations (STaDS), a suite of expert-like decision settings that evaluate LLMs as if they were professionals undertaking structured decision ``exams''. In this context, understanding is defined as the ability to identify and rely on the correct decision factors, features that determine outcomes within a domain. STaDS jointly assesses understanding through: (i) question and instruction comprehension, (ii) knowledge-based prediction, and (iii) reliance on relevant decision factors. By analyzing 9 frontier LLMs across 15 diverse decision settings, we find that (a) most models struggle to achieve consistently strong accuracy across diverse domains; (b) models can be accurate yet globally unfaithful, and there are frequent mismatches between stated rationales and factors driving predictions. Our findings highlight the need for global-level understanding evaluation protocols and advocate for novel frameworks that go beyond accuracy to enhance LLMs' understanding ability. 

---
# Guarding the Meaning: Self-Supervised Training for Semantic Robustness in Guard Models 

**Authors**: Cristina Pinneri, Christos Louizos  

**Link**: [PDF](https://arxiv.org/pdf/2511.10665)  

**Abstract**: Guard models are a critical component of LLM safety, but their sensitivity to superficial linguistic variations remains a key vulnerability. We show that even meaning-preserving paraphrases can cause large fluctuations in safety scores, revealing a lack of semantic grounding. To address this, we introduce a practical, self-supervised framework for improving the semantic robustness of guard models. Our method leverages paraphrase sets to enforce prediction consistency using a novel, skew-aware aggregation strategy for robust target computation. Notably, we find that standard aggregation methods like mean and median can degrade safety, underscoring the need for skew-aware alternatives. We analyze six open-source guard models and show that our approach reduces semantic variability across paraphrases by ~58%, improves benchmark accuracy by ~2.5% on average, and generalizes to unseen stylistic variations. Intriguingly, we discover a bidirectional relationship between model calibration and consistency: our robustness training improves calibration by up to 40%, revealing a fundamental connection between these properties. These results highlight the value of treating semantic consistency as a first-class training objective and provide a scalable recipe for building more reliable guard models. 

---
# Continual Learning of Domain Knowledge from Human Feedback in Text-to-SQL 

**Authors**: Thomas Cook, Kelly Patel, Sivapriya Vellaichamy, Saba Rahimi, Zhen Zeng, Sumitra Ganesh  

**Link**: [PDF](https://arxiv.org/pdf/2511.10674)  

**Abstract**: Large Language Models (LLMs) can generate SQL queries from natural language questions but struggle with database-specific schemas and tacit domain knowledge. We introduce a framework for continual learning from human feedback in text-to-SQL, where a learning agent receives natural language feedback to refine queries and distills the revealed knowledge for reuse on future tasks. This distilled knowledge is stored in a structured memory, enabling the agent to improve execution accuracy over time. We design and evaluate multiple variations of a learning agent architecture that vary in how they capture and retrieve past experiences. Experiments on the BIRD benchmark Dev set show that memory-augmented agents, particularly the Procedural Agent, achieve significant accuracy gains and error reduction by leveraging human-in-the-loop feedback. Our results highlight the importance of transforming tacit human expertise into reusable knowledge, paving the way for more adaptive, domain-aware text-to-SQL systems that continually learn from a human-in-the-loop. 

---
# Hybrid Quantum Transformer for Language Generation 

**Authors**: Desheng Kong, Xiangshuo Cui, Jiaying Jin, Jing Xu, Donglin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10653)  

**Abstract**: Although quantum computing has been increasingly applied to replace classical computation, most existing quantum or hybrid models remain confined to simple tasks, with no successful application to large-scale natural language generation to date. In this work, we present the first hybrid quantum-classical large language model (LLM) for natural language generation, HyQuT, capable of performing coherent and context-aware dialogue. The proposed architecture integrates variational quantum circuits (VQCs) into the Transformer framework at both 8M and 150M parameter scales. Experimental results show that a minimal number of qubits (10 qubits with 80 quantum gates) can replace about 10% of the classical parameters in the 150M-parameter model, while achieving comparable convergence stability and generation quality. This study provides an early demonstration of the feasibility of integrating quantum computing to large-scale generative language models. 

---
# Pre-Attention Expert Prediction and Prefetching for Mixture-of-Experts Large Language Models 

**Authors**: Shien Zhu, Samuel Bohl, Robin Oester, Gustavo Alonso  

**Link**: [PDF](https://arxiv.org/pdf/2511.10676)  

**Abstract**: Mixture-of-Experts (MoE) Large Language Models (LLMs) efficiently scale-up the model while keeping relatively low inference cost. As MoE models only activate part of the experts, related work has proposed expert prediction and caching methods to prefetch the experts for faster inference. However, existing approaches utilize the activations from the previous layer for prediction, incurring low accuracy and leave the first layer unoptimized. Applying complex layers or even training standalone networks for better prediction introduces high computation overhead. In this paper, we propose pre-attention expert prediction to achieve accurate and lightweight expert prefetching. The key insight is that some functions in LLMs are ranking-preserving, indicating that matching the ranking of selected experts using simple linear functions is possible. Therefore, we utilize the activations before the attention block in the same layer with 2 linear functions and ranking-aware loss to achieve accurate prediction, which also supports prefetching in the first layer. Our lightweight, pre-attention expert routers achieve 93.03% accuracy on DeepSeek V2 Lite, 94.69% on Qwen3-30B, and 97.62% on Phi-mini-MoE, showing about 15% improvement on absolute accuracy over the state-of-the-art methods. 

---
# Evaluating Open-Weight Large Language Models for Structured Data Extraction from Narrative Medical Reports Across Multiple Use Cases and Languages 

**Authors**: Douwe J. Spaanderman, Karthik Prathaban, Petr Zelina, Kaouther Mouheb, Lukáš Hejtmánek, Matthew Marzetti, Antonius W. Schurink, Damian Chan, Ruben Niemantsverdriet, Frederik Hartmann, Zhen Qian, Maarten G.J. Thomeer, Petr Holub, Farhan Akram, Frank J. Wolters, Meike W. Vernooij, Cornelis Verhoef, Esther E. Bron, Vít Nováček, Dirk J. Grünhagen, Wiro J. Niessen, Martijn P.A. Starmans, Stefan Klein  

**Link**: [PDF](https://arxiv.org/pdf/2511.10658)  

**Abstract**: Large language models (LLMs) are increasingly used to extract structured information from free-text clinical records, but prior work often focuses on single tasks, limited models, and English-language reports. We evaluated 15 open-weight LLMs on pathology and radiology reports across six use cases, colorectal liver metastases, liver tumours, neurodegenerative diseases, soft-tissue tumours, melanomas, and sarcomas, at three institutes in the Netherlands, UK, and Czech Republic. Models included general-purpose and medical-specialised LLMs of various sizes, and six prompting strategies were compared: zero-shot, one-shot, few-shot, chain-of-thought, self-consistency, and prompt graph. Performance was assessed using task-appropriate metrics, with consensus rank aggregation and linear mixed-effects models quantifying variance. Top-ranked models achieved macro-average scores close to inter-rater agreement across tasks. Small-to-medium general-purpose models performed comparably to large models, while tiny and specialised models performed worse. Prompt graph and few-shot prompting improved performance by ~13%. Task-specific factors, including variable complexity and annotation variability, influenced results more than model size or prompting strategy. These findings show that open-weight LLMs can extract structured data from clinical reports across diseases, languages, and institutions, offering a scalable approach for clinical data curation. 

---
# Towards Fine-Grained Code-Switch Speech Translation with Semantic Space Alignment 

**Authors**: Yan Gao, Yazheng Yang, Zhibin Lan, Yidong Chen, Min Zhang, Daimeng Wei, Hui Huang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2511.10670)  

**Abstract**: Code-switching (CS) speech translation (ST) refers to translating speech that alternates between two or more languages into a target language text, which poses significant challenges due to the complexity of semantic modeling and the scarcity of CS data. Previous studies tend to rely on the model itself to implicitly learn semantic modeling during training, and resort to inefficient and costly manual annotations for these two challenges. To mitigate these limitations, we propose enhancing Large Language Models (LLMs) with a Mixture of Experts (MoE) speech projector, where each expert specializes in the semantic subspace of a specific language, enabling fine-grained modeling of speech features. Additionally, we introduce a multi-stage training paradigm that utilizes readily available monolingual automatic speech recognition (ASR) and monolingual ST data, facilitating speech-text alignment and improving translation capabilities. During training, we leverage a combination of language-specific loss and intra-group load balancing loss to guide the MoE speech projector in efficiently allocating tokens to the appropriate experts, across expert groups and within each group, respectively. To bridge the data gap across different training stages and improve adaptation to the CS scenario, we further employ a transition loss, enabling smooth transitions of data between stages, to effectively address the scarcity of high-quality CS speech translation data. Extensive experiments on widely used datasets demonstrate the effectiveness and generality of our approach. 

---
# Data Analysis and Performance Evaluation of Simulation Deduction Based on LLMs 

**Authors**: Shansi Zhang, Min Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.10651)  

**Abstract**: Data analysis and performance evaluation of simulation deduction plays a pivotal role in modern warfare, which enables military personnel to gain invaluable insights into the potential effectiveness of different strategies, tactics, and operational plans. Traditional manual analysis approach is time-consuming and limited by human errors. To enhance efficiency and accuracy, large language models (LLMs) with strong analytical and inferencing capabilities can be employed. However, high-quality analysis reports with well-structured formatting cannot be obtained through a single instruction input to the LLM. To tackle this issue, we propose a method that first decomposes the complex task into several sub-tasks and designs effective system prompts and user prompts for each sub-task. Multi-round interactions with the LLM incorporating self-check and reflection are then conducted to enable structured data extraction as well as multi-step analysis and evaluation. Furthermore, custom tools are defined and invoked to generate figures and compute metrics. We also design multiple report templates, each tailored to a specific application and input data type, ensuring their adaptability across a variety of scenarios. Extensive evaluation results demonstrate that the reports generated by our method exhibit higher quality, therefore obtaining higher scores than the baseline method. 

---
# Assessing the Capabilities of LLMs in Humor:A Multi-dimensional Analysis of Oogiri Generation and Evaluation 

**Authors**: Ritsu Sakabe, Hwichan Kim, Tosho Hirasawa, Mamoru Komachi  

**Link**: [PDF](https://arxiv.org/pdf/2511.09133)  

**Abstract**: Computational humor is a frontier for creating advanced and engaging natural language processing (NLP) applications, such as sophisticated dialogue systems. While previous studies have benchmarked the humor capabilities of Large Language Models (LLMs), they have often relied on single-dimensional evaluations, such as judging whether something is simply ``funny.'' This paper argues that a multifaceted understanding of humor is necessary and addresses this gap by systematically evaluating LLMs through the lens of Oogiri, a form of Japanese improvisational comedy games. To achieve this, we expanded upon existing Oogiri datasets with data from new sources and then augmented the collection with Oogiri responses generated by LLMs. We then manually annotated this expanded collection with 5-point absolute ratings across six dimensions: Novelty, Clarity, Relevance, Intelligence, Empathy, and Overall Funniness. Using this dataset, we assessed the capabilities of state-of-the-art LLMs on two core tasks: their ability to generate creative Oogiri responses and their ability to evaluate the funniness of responses using a six-dimensional evaluation. Our results show that while LLMs can generate responses at a level between low- and mid-tier human performance, they exhibit a notable lack of Empathy. This deficit in Empathy helps explain their failure to replicate human humor assessment. Correlation analyses of human and model evaluation data further reveal a fundamental divergence in evaluation criteria: LLMs prioritize Novelty, whereas humans prioritize Empathy. We release our annotated corpus to the community to pave the way for the development of more emotionally intelligent and sophisticated conversational agents. 

---
# Evaluating Modern Large Language Models on Low-Resource and Morphologically Rich Languages:A Cross-Lingual Benchmark Across Cantonese, Japanese, and Turkish 

**Authors**: Chengxuan Xia, Qianye Wu, Hongbin Guan, Sixuan Tian, Yilun Hao, Xiaoyu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2511.10664)  

**Abstract**: Large language models (LLMs) have achieved impressive results in high-resource languages like English, yet their effectiveness in low-resource and morphologically rich languages remains underexplored. In this paper, we present a comprehensive evaluation of seven cutting-edge LLMs -- including GPT-4o, GPT-4, Claude~3.5~Sonnet, LLaMA~3.1, Mistral~Large~2, LLaMA-2~Chat~13B, and Mistral~7B~Instruct -- on a new cross-lingual benchmark covering \textbf{Cantonese, Japanese, and Turkish}. Our benchmark spans four diverse tasks: open-domain question answering, document summarization, English-to-X translation, and culturally grounded dialogue. We combine \textbf{human evaluations} (rating fluency, factual accuracy, and cultural appropriateness) with automated metrics (e.g., BLEU, ROUGE) to assess model performance.
Our results reveal that while the largest proprietary models (GPT-4o, GPT-4, Claude~3.5) generally lead across languages and tasks, significant gaps persist in culturally nuanced understanding and morphological generalization. Notably, GPT-4o demonstrates robust multilingual performance even on cross-lingual tasks, and Claude~3.5~Sonnet achieves competitive accuracy on knowledge and reasoning benchmarks. However, all models struggle to some extent with the unique linguistic challenges of each language, such as Turkish agglutinative morphology and Cantonese colloquialisms. Smaller open-source models (LLaMA-2~13B, Mistral~7B) lag substantially in fluency and accuracy, highlighting the resource disparity. We provide detailed quantitative results, qualitative error analysis, and discuss implications for developing more culturally aware and linguistically generalizable LLMs. Our benchmark and evaluation data are released to foster reproducibility and further research. 

---
# Preference Orchestrator: Prompt-Aware Multi-Objective Alignment for Large Language Models 

**Authors**: Biao Liu, Ning Xu, Junming Yang, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2511.10656)  

**Abstract**: While Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse natural language processing tasks, aligning these models with varying human preferences across multiple objectives remains a significant challenge in practical deployments. Existing multi-objective alignment methods rely on manually specified preference weights, which not only burden users with difficult preference specification tasks but also lead to suboptimal training efficiency due to exploration of irrelevant preference combinations. To alleviate these issues, we propose a novel framework named PRO, i.e., PReference Orchestrator, which features a lightweight preference adapter that automatically infers prompt-specific preference weights during both training and deployment phases. Specifically, the adapter automatically learns appropriate preference weights for each prompt by training on normalized reward scores from multiple reward models for preferred responses, which inherently reflect effective preference balances across objectives. Additionally, We provide theoretical analysis proving that our prompt-aware preference mechanism achieves superior performance compared to fixed preference weights in multi-objective alignment scenarios. Extensive experiments across multiple tasks demonstrate the effectiveness of our method over existing multi-objective alignment approaches. 

---
# LAET: A Layer-wise Adaptive Ensemble Tuning Framework for Pretrained Language Models 

**Authors**: Jawad Ibn Ahad, Muhammad Rafsan Kabir, Robin Krambroeckers, Sifat Momen, Nabeel Mohammed, Shafin Rahman  

**Link**: [PDF](https://arxiv.org/pdf/2511.11315)  

**Abstract**: Natural Language Processing (NLP) has transformed the financial industry, enabling advancements in areas such as textual analysis, risk management, and forecasting. Large language models (LLMs) like BloombergGPT and FinMA have set new benchmarks across various financial NLP tasks, including sentiment analysis, stock movement prediction, and credit risk assessment. Furthermore, FinMA-ES, a bilingual financial LLM, has also demonstrated strong performance using the FLARE and FLARE-ES benchmarks. However, the high computational demands of these models limit the accessibility of many organizations. To address this, we propose Layer-wise Adaptive Ensemble Tuning (LAET), a novel strategy that selectively fine-tunes the most effective layers of pre-trained LLMs by analyzing hidden state representations while freezing less critical layers. LAET significantly reduces computational overhead while enhancing task-specific performance. Our approach shows strong results in financial NLP tasks, outperforming existing benchmarks and state-of-the-art LLMs such as GPT-4, even with smaller LLMs ($\sim$3B parameters). This work bridges cutting-edge financial NLP research and real-world deployment with efficient and scalable models for financial applications. 

---
# W2S-AlignTree: Weak-to-Strong Inference-Time Alignment for Large Language Models via Monte Carlo Tree Search 

**Authors**: Zhenyu Ding, Yuhao Wang, Tengyue Xiao, Haoying Wang, Guojun Ma, Mingyang Wan, Caigui Jiang, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2511.11518)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive capabilities, yet their outputs often suffer from misalignment with human preferences due to the inadequacy of weak supervision and a lack of fine-grained control. Training-time alignment methods like Reinforcement Learning from Human Feedback (RLHF) face prohibitive costs in expert supervision and inherent scalability limitations, offering limited dynamic control during inference. Consequently, there is an urgent need for scalable and adaptable alignment mechanisms. To address this, we propose W2S-AlignTree, a pioneering plug-and-play inference-time alignment framework that synergistically combines Monte Carlo Tree Search (MCTS) with the Weak-to-Strong Generalization paradigm for the first time. W2S-AlignTree formulates LLM alignment as an optimal heuristic search problem within a generative search tree. By leveraging weak model's real-time, step-level signals as alignment proxies and introducing an Entropy-Aware exploration mechanism, W2S-AlignTree enables fine-grained guidance during strong model's generation without modifying its parameters. The approach dynamically balances exploration and exploitation in high-dimensional generation search trees. Experiments across controlled sentiment generation, summarization, and instruction-following show that W2S-AlignTree consistently outperforms strong baselines. Notably, W2S-AlignTree raises the performance of Llama3-8B from 1.89 to 2.19, a relative improvement of 15.9 on the summarization task. 

---
# LaoBench: A Large-Scale Multidimensional Lao Benchmark for Large Language Models 

**Authors**: Jian Gao, Richeng Xuan, Zhaolu Kang, Dingshi Liao, Wenxin Huang, Zongmou Huang, Yangdi Xu, Bowen Qin, Zheqi He, Xi Yang, Changjin Li  

**Link**: [PDF](https://arxiv.org/pdf/2511.11334)  

**Abstract**: The rapid advancement of large language models (LLMs) has not been matched by their evaluation in low-resource languages, especially Southeast Asian languages like Lao. To fill this gap, we introduce LaoBench, the first large-scale, high-quality, and multidimensional benchmark dataset dedicated to assessing LLMs' comprehensive language understanding and reasoning abilities in Lao. LaoBench comprises over 17,000 carefully curated samples spanning three core dimensions: knowledge application, K12 foundational education, and bilingual translation among Lao, Chinese, and English. The dataset is divided into open-source and closed-source subsets, with the closed-source portion enabling black-box evaluation on an official platform to ensure fairness and data security. Our data construction pipeline integrates expert human curation with automated agent-assisted verification, ensuring linguistic accuracy, cultural relevance, and educational value. Benchmarking multiple state-of-the-art LLMs on LaoBench reveals that current models still face significant challenges in mastering Lao across diverse tasks. We hope LaoBench will catalyze further research and development of AI technologies for underrepresented Southeast Asian languages. 

---
# Can LLMs Detect Their Own Hallucinations? 

**Authors**: Sora Kadotani, Kosuke Nishida, Kyosuke Nishida  

**Link**: [PDF](https://arxiv.org/pdf/2511.11087)  

**Abstract**: Large language models (LLMs) can generate fluent responses, but sometimes hallucinate facts. In this paper, we investigate whether LLMs can detect their own hallucinations. We formulate hallucination detection as a classification task of a sentence. We propose a framework for estimating LLMs' capability of hallucination detection and a classification method using Chain-of-Thought (CoT) to extract knowledge from their parameters. The experimental results indicated that GPT-$3.5$ Turbo with CoT detected $58.2\%$ of its own hallucinations. We concluded that LLMs with CoT can detect hallucinations if sufficient knowledge is contained in their parameters. 

---
# Analysing Personal Attacks in U.S. Presidential Debates 

**Authors**: Ruban Goyal, Rohitash Chandra, Sonit Singh  

**Link**: [PDF](https://arxiv.org/pdf/2511.11108)  

**Abstract**: Personal attacks have become a notable feature of U.S. presidential debates and play an important role in shaping public perception during elections. Detecting such attacks can improve transparency in political discourse and provide insights for journalists, analysts and the public. Advances in deep learning and transformer-based models, particularly BERT and large language models (LLMs) have created new opportunities for automated detection of harmful language. Motivated by these developments, we present a framework for analysing personal attacks in U.S. presidential debates. Our work involves manual annotation of debate transcripts across the 2016, 2020 and 2024 election cycles, followed by statistical and language-model based analysis. We investigate the potential of fine-tuned transformer models alongside general-purpose LLMs to detect personal attacks in formal political speech. This study demonstrates how task-specific adaptation of modern language models can contribute to a deeper understanding of political communication. 

---
# Enhancing Meme Emotion Understanding with Multi-Level Modality Enhancement and Dual-Stage Modal Fusion 

**Authors**: Yi Shi, Wenlong Meng, Zhenyuan Guo, Chengkun Wei, Wenzhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2511.11126)  

**Abstract**: With the rapid rise of social media and Internet culture, memes have become a popular medium for expressing emotional tendencies. This has sparked growing interest in Meme Emotion Understanding (MEU), which aims to classify the emotional intent behind memes by leveraging their multimodal contents. While existing efforts have achieved promising results, two major challenges remain: (1) a lack of fine-grained multimodal fusion strategies, and (2) insufficient mining of memes' implicit meanings and background knowledge. To address these challenges, we propose MemoDetector, a novel framework for advancing MEU. First, we introduce a four-step textual enhancement module that utilizes the rich knowledge and reasoning capabilities of Multimodal Large Language Models (MLLMs) to progressively infer and extract implicit and contextual insights from memes. These enhanced texts significantly enrich the original meme contents and provide valuable guidance for downstream classification. Next, we design a dual-stage modal fusion strategy: the first stage performs shallow fusion on raw meme image and text, while the second stage deeply integrates the enhanced visual and textual features. This hierarchical fusion enables the model to better capture nuanced cross-modal emotional cues. Experiments on two datasets, MET-MEME and MOOD, demonstrate that our method consistently outperforms state-of-the-art baselines. Specifically, MemoDetector improves F1 scores by 4.3\% on MET-MEME and 3.4\% on MOOD. Further ablation studies and in-depth analyses validate the effectiveness and robustness of our approach, highlighting its strong potential for advancing MEU. Our code is available at this https URL. 

---
# From Proof to Program: Characterizing Tool-Induced Reasoning Hallucinations in Large Language Models 

**Authors**: Farima Fatahi Bayat, Pouya Pezeshkpour, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2511.10899)  

**Abstract**: Tool-augmented Language Models (TaLMs) can invoke external tools to solve problems beyond their parametric capacity. However, it remains unclear whether these tool-enabled gains reflect trustworthy reasoning. Focusing on the Code Interpreter tool, we show that even when tools are selected and executed correctly, TaLMs treat tool outputs as substitutes for reasoning, producing solutions that appear correct but lack coherent justification. We term this failure mode Tool-Induced Myopia (TIM), and study it using PYMATH, a benchmark of 1,679 competition-level mathematical problems for which Python code is helpful but not sufficient. We further develop a multi-dimensional evaluation suite to quantify reasoning degradation in TaLMs relative to their non-tool counterparts. Our findings reveal that while TaLMs achieve up to a 19.3 percentage point gain in final-answer accuracy, their reasoning behavior consistently deteriorates (e.g., non-tool LLMs win up to 41.5% more often in pairwise comparisons of the reasoning process). This degradation intensifies with tool use; the more frequently a model invokes tools, the less coherent its reasoning becomes. Moreover, tool use shifts errors from arithmetic mistakes toward global reasoning failures (logic, assumption, creativity); with TIM present in ~55% of high-risk cases. Finally, we propose a preference-optimization-based framework that realigns TaLMs to use tools as assistive evidence, improving both final-answer accuracy and reasoning depth under tool use. Codes and data are available at: this https URL. 

---
# From Fact to Judgment: Investigating the Impact of Task Framing on LLM Conviction in Dialogue Systems 

**Authors**: Parisa Rabbani, Nimet Beyza Bozdag, Dilek Hakkani-Tür  

**Link**: [PDF](https://arxiv.org/pdf/2511.10871)  

**Abstract**: LLMs are increasingly employed as judges across a variety of tasks, including those involving everyday social interactions. Yet, it remains unclear whether such LLM-judges can reliably assess tasks that require social or conversational judgment. We investigate how an LLM's conviction is changed when a task is reframed from a direct factual query to a Conversational Judgment Task. Our evaluation framework contrasts the model's performance on direct factual queries with its assessment of a speaker's correctness when the same information is presented within a minimal dialogue, effectively shifting the query from "Is this statement correct?" to "Is this speaker correct?". Furthermore, we apply pressure in the form of a simple rebuttal ("The previous answer is incorrect.") to both conditions. This perturbation allows us to measure how firmly the model maintains its position under conversational pressure. Our findings show that while some models like GPT-4o-mini reveal sycophantic tendencies under social framing tasks, others like Llama-8B-Instruct become overly-critical. We observe an average performance change of 9.24% across all models, demonstrating that even minimal dialogue context can significantly alter model judgment, underscoring conversational framing as a key factor in LLM-based evaluation. The proposed framework offers a reproducible methodology for diagnosing model conviction and contributes to the development of more trustworthy dialogue systems. 

---
# Tracing Multilingual Representations in LLMs with Cross-Layer Transcoders 

**Authors**: Abir Harrasse, Florent Draye, Zhijing Jin, Bernhard Schölkopf  

**Link**: [PDF](https://arxiv.org/pdf/2511.10840)  

**Abstract**: Multilingual Large Language Models (LLMs) can process many languages, yet how they internally represent this diversity remains unclear. Do they form shared multilingual representations with language-specific decoding, and if so, why does performance still favor the dominant training language? To address this, we train a series of LLMs on different mixtures of multilingual data and analyze their internal mechanisms using cross-layer transcoders (CLT) and attribution graphs. Our results provide strong evidence for pivot language representations: the model employs nearly identical representations across languages, while language-specific decoding emerges in later layers. Attribution analyses reveal that decoding relies in part on a small set of high-frequency language features in the final layers, which linearly read out language identity from the first layers in the model. By intervening on these features, we can suppress one language and substitute another in the model's outputs. Finally, we study how the dominant training language influences these mechanisms across attribution graphs and decoding pathways. We argue that understanding this pivot-language mechanism is crucial for improving multilingual alignment in LLMs. 

---
# LLM-as-a-Grader: Practical Insights from Large Language Model for Short-Answer and Report Evaluation 

**Authors**: Grace Byun, Swati Rajwal, Jinho D. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2511.10819)  

**Abstract**: Large Language Models (LLMs) are increasingly explored for educational tasks such as grading, yet their alignment with human evaluation in real classrooms remains underexamined. In this study, we investigate the feasibility of using an LLM (GPT-4o) to evaluate short-answer quizzes and project reports in an undergraduate Computational Linguistics course. We collect responses from approximately 50 students across five quizzes and receive project reports from 14 teams. LLM-generated scores are compared against human evaluations conducted independently by the course teaching assistants (TAs). Our results show that GPT-4o achieves strong correlation with human graders (up to 0.98) and exact score agreement in 55\% of quiz cases. For project reports, it also shows strong overall alignment with human grading, while exhibiting some variability in scoring technical, open-ended responses. We release all code and sample data to support further research on LLMs in educational assessment. This work highlights both the potential and limitations of LLM-based grading systems and contributes to advancing automated grading in real-world academic settings. 

---
# "As Eastern Powers, I will veto." : An Investigation of Nation-level Bias of Large Language Models in International Relations 

**Authors**: Jonghyeon Choi, Yeonjun Choi, Hyun-chul Kim, Beakcheol Jang  

**Link**: [PDF](https://arxiv.org/pdf/2511.10695)  

**Abstract**: This paper systematically examines nation-level biases exhibited by Large Language Models (LLMs) within the domain of International Relations (IR). Leveraging historical records from the United Nations Security Council (UNSC), we developed a bias evaluation framework comprising three distinct tests to explore nation-level bias in various LLMs, with a particular focus on the five permanent members of the UNSC. Experimental results show that, even with the general bias patterns across models (e.g., favorable biases toward the western nations, and unfavorable biases toward Russia), these still vary based on the LLM. Notably, even within the same LLM, the direction and magnitude of bias for a nation change depending on the evaluation context. This observation suggests that LLM biases are fundamentally multidimensional, varying across models and tasks. We also observe that models with stronger reasoning abilities show reduced bias and better performance. Building on this finding, we introduce a debiasing framework that improves LLMs' factual reasoning combining Retrieval-Augmented Generation with Reflexion-based self-reflection techniques. Experiments show it effectively reduces nation-level bias, and improves performance, particularly in GPT-4o-mini and LLama-3.3-70B. Our findings emphasize the need to assess nation-level bias alongside performance when applying LLMs in the IR domain. 

---
# Faithful Summarization of Consumer Health Queries: A Cross-Lingual Framework with LLMs 

**Authors**: Ajwad Abrar, Nafisa Tabassum Oeshy, Prianka Maheru, Farzana Tabassum, Tareque Mohmud Chowdhury  

**Link**: [PDF](https://arxiv.org/pdf/2511.10768)  

**Abstract**: Summarizing consumer health questions (CHQs) can ease communication in healthcare, but unfaithful summaries that misrepresent medical details pose serious risks. We propose a framework that combines TextRank-based sentence extraction and medical named entity recognition with large language models (LLMs) to enhance faithfulness in medical text summarization. In our experiments, we fine-tuned the LLaMA-2-7B model on the MeQSum (English) and BanglaCHQ-Summ (Bangla) datasets, achieving consistent improvements across quality (ROUGE, BERTScore, readability) and faithfulness (SummaC, AlignScore) metrics, and outperforming zero-shot baselines and prior systems. Human evaluation further shows that over 80\% of generated summaries preserve critical medical information. These results highlight faithfulness as an essential dimension for reliable medical summarization and demonstrate the potential of our approach for safer deployment of LLMs in healthcare contexts. 

---
# Where does an LLM begin computing an instruction? 

**Authors**: Aditya Pola, Vineeth N. Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2511.10694)  

**Abstract**: Following an instruction involves distinct sub-processes, such as reading content, reading the instruction, executing it, and producing an answer. We ask where, along the layer stack, instruction following begins, the point where reading gives way to doing. We introduce three simple datasets (Key-Value, Quote Attribution, Letter Selection) and two hop compositions of these tasks. Using activation patching on minimal-contrast prompt pairs, we measure a layer-wise flip rate that indicates when substituting selected residual activations changes the predicted answer. Across models in the Llama family, we observe an inflection point, which we term onset, where interventions that change predictions before this point become largely ineffective afterward. Multi-hop compositions show a similar onset location. These results provide a simple, replicable way to locate where instruction following begins and to compare this location across tasks and model sizes. 

---
# Modeling and Predicting Multi-Turn Answer Instability in Large Language Models 

**Authors**: Jiahang He, Rishi Ramachandran, Neel Ramachandran, Aryan Katakam, Kevin Zhu, Sunishchal Dev, Ashwinee Panda, Aryan Shrivastava  

**Link**: [PDF](https://arxiv.org/pdf/2511.10688)  

**Abstract**: As large language models (LLMs) are adopted in an increasingly wide range of applications, user-model interactions have grown in both frequency and scale. Consequently, research has focused on evaluating the robustness of LLMs, an essential quality for real-world tasks. In this paper, we employ simple multi-turn follow-up prompts to evaluate models' answer changes, model accuracy dynamics across turns with Markov chains, and examine whether linear probes can predict these changes. Our results show significant vulnerabilities in LLM robustness: a simple "Think again" prompt led to an approximate 10% accuracy drop for Gemini 1.5 Flash over nine turns, while combining this prompt with a semantically equivalent reworded question caused a 7.5% drop for Claude 3.5 Haiku. Additionally, we find that model accuracy across turns can be effectively modeled using Markov chains, enabling the prediction of accuracy probabilities over time. This allows for estimation of the model's stationary (long-run) accuracy, which we find to be on average approximately 8% lower than its first-turn accuracy for Gemini 1.5 Flash. Our results from a model's hidden states also reveal evidence that linear probes can help predict future answer changes. Together, these results establish stationary accuracy as a principled robustness metric for interactive settings and expose the fragility of models under repeated questioning. Addressing this instability will be essential for deploying LLMs in high-stakes and interactive settings where consistent reasoning is as important as initial accuracy. 

---
# A methodological analysis of prompt perturbations and their effect on attack success rates 

**Authors**: Tiago Machado, Maysa Malfiza Garcia de Macedo, Rogerio Abreu de Paula, Marcelo Carpinette Grave, Aminat Adebiyi, Luan Soares de Souza, Enrico Santarelli, Claudio Pinhanez  

**Link**: [PDF](https://arxiv.org/pdf/2511.10686)  

**Abstract**: This work aims to investigate how different Large Language Models (LLMs) alignment methods affect the models' responses to prompt attacks. We selected open source models based on the most common alignment methods, namely, Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Reinforcement Learning with Human Feedback (RLHF). We conducted a systematic analysis using statistical methods to verify how sensitive the Attack Success Rate (ASR) is when we apply variations to prompts designed to elicit inappropriate content from LLMs. Our results show that even small prompt modifications can significantly change the Attack Success Rate (ASR) according to the statistical tests we run, making the models more or less susceptible to types of attack. Critically, our results demonstrate that running existing 'attack benchmarks' alone may not be sufficient to elicit all possible vulnerabilities of both models and alignment methods. This paper thus contributes to ongoing efforts on model attack evaluation by means of systematic and statistically-based analyses of the different alignment methods and how sensitive their ASR is to prompt variation. 

---
# SpiderGen: Towards Procedure Generation For Carbon Life Cycle Assessments with Generative AI 

**Authors**: Anupama Sitaraman, Bharathan Balaji, Yuvraj Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2511.10684)  

**Abstract**: Investigating the effects of climate change and global warming caused by GHG emissions have been a primary concern worldwide. These emissions are largely contributed to by the production, use and disposal of consumer products. Thus, it is important to build tools to estimate the environmental impact of consumer goods, an essential part of which is conducting Life Cycle Assessments (LCAs). LCAs specify and account for the appropriate processes involved with the production, use, and disposal of the products. We present SpiderGen, an LLM-based workflow which integrates the taxonomy and methodology of traditional LCA with the reasoning capabilities and world knowledge of LLMs to generate the procedural information used for LCA. We additionally evaluate the output of SpiderGen using real-world LCA documents as ground-truth. We find that SpiderGen provides accurate LCA process information that is either fully correct or has minor errors, achieving an F1-Score of 62% across 10 sample data points. We observe that the remaining missed processes and hallucinated errors occur primarily due to differences in detail between LCA documents, as well as differences in the "scope" of which auxiliary processes must also be included. We also demonstrate that SpiderGen performs better than several baselines techniques, such as chain-of-thought prompting and one-shot prompting. Finally, we highlight SpiderGen's potential to reduce the human effort and costs for estimating carbon impact, as it is able to produce LCA process information for less than \$1 USD in under 10 minutes as compared to the status quo LCA, which can cost over \$25000 USD and take up to 21-person days. 

---
# Grounded Visual Factualization: Factual Anchor-Based Finetuning for Enhancing MLLM Factual Consistency 

**Authors**: Filippo Morbiato, Luca Romano, Alessandro Persona  

**Link**: [PDF](https://arxiv.org/pdf/2511.10671)  

**Abstract**: Visual hallucination, where Multimodal Large Language Models fabricate details inconsistent with image content, critically undermines their reliability. Existing fine-tuning methods offer limited improvement, failing to deeply intervene in factual reasoning. This paper introduces Grounded Visual Factualization (GVF) Finetuning, a novel approach to systematically enhance MLLM visual factual consistency. GVF integrates explicit factual signals via three core mechanisms: Factual Anchor Data Augmentation, enriching training data with structured factual anchors and counter-factual prompts; Fact-Aware Instruction Tuning, embedding these cues into explicit instructions; and a Factual Consistency Loss function, specifically penalizing factual inaccuracies. Evaluated on LLaVA-1.5-13B, GVF Finetuning significantly outperforms standard fine-tuning on the VHTest benchmark for both Open-Ended Question (OEQ) and Yes/No Question (YNQ) formats. Crucially, GVF maintains or even slightly improves performance on general multimodal benchmarks like MME and POPE, demonstrating effective mitigation of visual hallucinations without compromising general understanding and reasoning abilities. 

---
# Bayesian Evaluation of Large Language Model Behavior 

**Authors**: Rachel Longjohn, Shang Wu, Saatvik Kher, Catarina Belém, Padhraic Smyth  

**Link**: [PDF](https://arxiv.org/pdf/2511.10661)  

**Abstract**: It is increasingly important to evaluate how text generation systems based on large language models (LLMs) behave, such as their tendency to produce harmful output or their sensitivity to adversarial inputs. Such evaluations often rely on a curated benchmark set of input prompts provided to the LLM, where the output for each prompt may be assessed in a binary fashion (e.g., harmful/non-harmful or does not leak/leaks sensitive information), and the aggregation of binary scores is used to evaluate the LLM. However, existing approaches to evaluation often neglect statistical uncertainty quantification. With an applied statistics audience in mind, we provide background on LLM text generation and evaluation, and then describe a Bayesian approach for quantifying uncertainty in binary evaluation metrics. We focus in particular on uncertainty that is induced by the probabilistic text generation strategies typically deployed in LLM-based systems. We present two case studies applying this approach: 1) evaluating refusal rates on a benchmark of adversarial inputs designed to elicit harmful responses, and 2) evaluating pairwise preferences of one LLM over another on a benchmark of open-ended interactive dialogue examples. We demonstrate how the Bayesian approach can provide useful uncertainty quantification about the behavior of LLM-based systems. 

---
# Large language models in materials science and the need for open-source approaches 

**Authors**: Fengxu Yang, Weitong Chen, Jack D. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2511.10673)  

**Abstract**: Large language models (LLMs) are rapidly transforming materials science. This review examines recent LLM applications across the materials discovery pipeline, focusing on three key areas: mining scientific literature , predictive modelling, and multi-agent experimental systems. We highlight how LLMs extract valuable information such as synthesis conditions from text, learn structure-property relationships, and can coordinate agentic systems integrating computational tools and laboratory automation. While progress has been largely dependent on closed-source commercial models, our benchmark results demonstrate that open-source alternatives can match performance while offering greater transparency, reproducibility, cost-effectiveness, and data privacy. As open-source models continue to improve, we advocate their broader adoption to build accessible, flexible, and community-driven AI platforms for scientific discovery. 

---
