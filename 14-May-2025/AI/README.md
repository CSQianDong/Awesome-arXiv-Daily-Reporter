# ARC-NCA: Towards Developmental Solutions to the Abstraction and Reasoning Corpus 

**Authors**: Etienne Guichard, Felix Reimers, Mia Kvalsund, Mikkel Lepperød, Stefano Nichele  

**Link**: [PDF](https://arxiv.org/pdf/2505.08778)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC), later renamed ARC-AGI, poses a fundamental challenge in artificial general intelligence (AGI), requiring solutions that exhibit robust abstraction and reasoning capabilities across diverse tasks, while only few (with median count of three) correct examples are presented. While ARC-AGI remains very challenging for artificial intelligence systems, it is rather easy for humans. This paper introduces ARC-NCA, a developmental approach leveraging standard Neural Cellular Automata (NCA) and NCA enhanced with hidden memories (EngramNCA) to tackle the ARC-AGI benchmark. NCAs are employed for their inherent ability to simulate complex dynamics and emergent patterns, mimicking developmental processes observed in biological systems. Developmental solutions may offer a promising avenue for enhancing AI's problem-solving capabilities beyond mere training data extrapolation. ARC-NCA demonstrates how integrating developmental principles into computational models can foster adaptive reasoning and abstraction. We show that our ARC-NCA proof-of-concept results may be comparable to, and sometimes surpass, that of ChatGPT 4.5, at a fraction of the cost. 

---
# DeepMath-Creative: A Benchmark for Evaluating Mathematical Creativity of Large Language Models 

**Authors**: Xiaoyang Chen, Xinan Dai, Yu Du, Qian Feng, Naixu Guo, Tingshuo Gu, Yuting Gao, Yingyi Gao, Xudong Han, Xiang Jiang, Yilin Jin, Hongyi Lin, Shisheng Lin, Xiangnan Li, Yuante Li, Yixing Li, Zhentao Lai, Zilu Ma, Yingrong Peng, Jiacheng Qian, Hao-Yu Sun, Jianbo Sun, Zirui Wang, Siwei Wu, Zian Wang, Bin Xu, Jianghao Xu, Yiyang Yu, Zichuan Yang, Hongji Zha, Ruichong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08744)  

**Abstract**: To advance the mathematical proficiency of large language models (LLMs), the DeepMath team has launched an open-source initiative aimed at developing an open mathematical LLM and systematically evaluating its mathematical creativity. This paper represents the initial contribution of this initiative. While recent developments in mathematical LLMs have predominantly emphasized reasoning skills, as evidenced by benchmarks on elementary to undergraduate-level mathematical tasks, the creative capabilities of these models have received comparatively little attention, and evaluation datasets remain scarce. To address this gap, we propose an evaluation criteria for mathematical creativity and introduce DeepMath-Creative, a novel, high-quality benchmark comprising constructive problems across algebra, geometry, analysis, and other domains. We conduct a systematic evaluation of mainstream LLMs' creative problem-solving abilities using this dataset. Experimental results show that even under lenient scoring criteria -- emphasizing core solution components and disregarding minor inaccuracies, such as small logical gaps, incomplete justifications, or redundant explanations -- the best-performing model, O3 Mini, achieves merely 70% accuracy, primarily on basic undergraduate-level constructive tasks. Performance declines sharply on more complex problems, with models failing to provide substantive strategies for open problems. These findings suggest that, although current LLMs display a degree of constructive proficiency on familiar and lower-difficulty problems, such performance is likely attributable to the recombination of memorized patterns rather than authentic creative insight or novel synthesis. 

---
# LLM-based Prompt Ensemble for Reliable Medical Entity Recognition from EHRs 

**Authors**: K M Sajjadul Islam, Ayesha Siddika Nipu, Jiawei Wu, Praveen Madiraju  

**Link**: [PDF](https://arxiv.org/pdf/2505.08704)  

**Abstract**: Electronic Health Records (EHRs) are digital records of patient information, often containing unstructured clinical text. Named Entity Recognition (NER) is essential in EHRs for extracting key medical entities like problems, tests, and treatments to support downstream clinical applications. This paper explores prompt-based medical entity recognition using large language models (LLMs), specifically GPT-4o and DeepSeek-R1, guided by various prompt engineering techniques, including zero-shot, few-shot, and an ensemble approach. Among all strategies, GPT-4o with prompt ensemble achieved the highest classification performance with an F1-score of 0.95 and recall of 0.98, outperforming DeepSeek-R1 on the task. The ensemble method improved reliability by aggregating outputs through embedding-based similarity and majority voting. 

---
# A Study of Data-driven Methods for Inventory Optimization 

**Authors**: Lee Yeung Ping, Patrick Wong, Tan Cheng Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.08673)  

**Abstract**: This paper shows a comprehensive analysis of three algorithms (Time Series, Random Forest (RF) and Deep Reinforcement Learning) into three inventory models (the Lost Sales, Dual-Sourcing and Multi-Echelon Inventory Model). These methodologies are applied in the supermarket context. The main purpose is to analyse efficient methods for the data-driven. Their possibility, potential and current challenges are taken into consideration in this report. By comparing the results in each model, the effectiveness of each algorithm is evaluated based on several key performance indicators, including forecast accuracy, adaptability to market changes, and overall impact on inventory costs and customer satisfaction levels. The data visualization tools and statistical metrics are the indicators for the comparisons and show some obvious trends and patterns that can guide decision-making in inventory management. These tools enable managers to not only track the performance of different algorithms in real-time but also to drill down into specific data points to understand the underlying causes of inventory fluctuations. This level of detail is crucial for pinpointing inefficiencies and areas for improvement within the supply chain. 

---
# WixQA: A Multi-Dataset Benchmark for Enterprise Retrieval-Augmented Generation 

**Authors**: Dvir Cohen, Lin Burg, Sviatoslav Pykhnivskyi, Hagit Gur, Stanislav Kovynov, Olga Atzmon, Gilad Barkan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08643)  

**Abstract**: Retrieval-Augmented Generation (RAG) is a cornerstone of modern question answering (QA) systems, enabling grounded answers based on external knowledge. Although recent progress has been driven by open-domain datasets, enterprise QA systems need datasets that mirror the concrete, domain-specific issues users raise in day-to-day support scenarios. Critically, evaluating end-to-end RAG systems requires benchmarks comprising not only question--answer pairs but also the specific knowledge base (KB) snapshot from which answers were derived. To address this need, we introduce WixQA, a benchmark suite featuring QA datasets precisely grounded in the released KB corpus, enabling holistic evaluation of retrieval and generation components. WixQA includes three distinct QA datasets derived from this http URL customer support interactions and grounded in a snapshot of the public Wix Help Center KB: (i) WixQA-ExpertWritten, 200 real user queries with expert-authored, multi-step answers; (ii) WixQA-Simulated, 200 expert-validated QA pairs distilled from user dialogues; and (iii) WixQA-Synthetic, 6,222 LLM-generated QA pairs, with one pair systematically derived from each article in the knowledge base. We release the KB snapshot alongside the datasets under MIT license and provide comprehensive baseline results, forming a unique benchmark for evaluating enterprise RAG systems in realistic enterprise environments. 

---
# TRAIL: Trace Reasoning and Agentic Issue Localization 

**Authors**: Darshan Deshpande, Varun Gangal, Hersh Mehta, Jitin Krishnan, Anand Kannappan, Rebecca Qian  

**Link**: [PDF](https://arxiv.org/pdf/2505.08638)  

**Abstract**: The increasing adoption of agentic workflows across diverse domains brings a critical need to scalably and systematically evaluate the complex traces these systems generate. Current evaluation methods depend on manual, domain-specific human analysis of lengthy workflow traces - an approach that does not scale with the growing complexity and volume of agentic outputs. Error analysis in these settings is further complicated by the interplay of external tool outputs and language model reasoning, making it more challenging than traditional software debugging. In this work, we (1) articulate the need for robust and dynamic evaluation methods for agentic workflow traces, (2) introduce a formal taxonomy of error types encountered in agentic systems, and (3) present a set of 148 large human-annotated traces (TRAIL) constructed using this taxonomy and grounded in established agentic benchmarks. To ensure ecological validity, we curate traces from both single and multi-agent systems, focusing on real-world applications such as software engineering and open-world information retrieval. Our evaluations reveal that modern long context LLMs perform poorly at trace debugging, with the best Gemini-2.5-pro model scoring a mere 11% on TRAIL. Our dataset and code are made publicly available to support and accelerate future research in scalable evaluation for agentic workflows. 

---
# Integrating Natural Language Processing and Exercise Monitoring for Early Diagnosis of Metabolic Syndrome: A Deep Learning Approach 

**Authors**: Yichen Zhao, Yuhua Wang, Xi Cheng, Junhao Fang, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08628)  

**Abstract**: Metabolic syndrome (MetS) is a medication condition characterized by abdominal obesity, insulin resistance, hypertension and hyperlipidemia. It increases the risk of majority of chronic diseases, including type 2 diabetes mellitus, and affects about one quarter of the global population. Therefore, early detection and timely intervention for MetS are crucial. Standard diagnosis for MetS components requires blood tests conducted within medical institutions. However, it is frequently underestimated, leading to unmet need for care for MetS population. This study aims to use the least physiological data and free texts about exercises related activities, which are obtained easily in daily life, to diagnosis MetS. We collected the data from 40 volunteers in a nursing home and used data augmentation to reduce the imbalance. We propose a deep learning framework for classifying MetS that integrates natural language processing (NLP) and exercise monitoring. The results showed that the best model reported a high positive result (AUROC=0.806 and REC=76.3%) through 3-fold cross-validation. Feature importance analysis revealed that text and minimum heart rate on a daily basis contribute the most in the classification of MetS. This study demonstrates the potential application of data that are easily measurable in daily life for the early diagnosis of MetS, which could contribute to reducing the cost of screening and management for MetS population. 

---
# Visually Guided Decoding: Gradient-Free Hard Prompt Inversion with Language Models 

**Authors**: Donghoon Kim, Minji Bae, Kyuhong Shim, Byonghyo Shim  

**Link**: [PDF](https://arxiv.org/pdf/2505.08622)  

**Abstract**: Text-to-image generative models like DALL-E and Stable Diffusion have revolutionized visual content creation across various applications, including advertising, personalized media, and design prototyping. However, crafting effective textual prompts to guide these models remains challenging, often requiring extensive trial and error. Existing prompt inversion approaches, such as soft and hard prompt techniques, are not so effective due to the limited interpretability and incoherent prompt generation. To address these issues, we propose Visually Guided Decoding (VGD), a gradient-free approach that leverages large language models (LLMs) and CLIP-based guidance to generate coherent and semantically aligned prompts. In essence, VGD utilizes the robust text generation capabilities of LLMs to produce human-readable prompts. Further, by employing CLIP scores to ensure alignment with user-specified visual concepts, VGD enhances the interpretability, generalization, and flexibility of prompt generation without the need for additional training. Our experiments demonstrate that VGD outperforms existing prompt inversion techniques in generating understandable and contextually relevant prompts, facilitating more intuitive and controllable interactions with text-to-image models. 

---
# Resource-Efficient Language Models: Quantization for Fast and Accessible Inference 

**Authors**: Tollef Emil Jørgensen  

**Link**: [PDF](https://arxiv.org/pdf/2505.08620)  

**Abstract**: Large language models have significantly advanced natural language processing, yet their heavy resource demands pose severe challenges regarding hardware accessibility and energy consumption. This paper presents a focused and high-level review of post-training quantization (PTQ) techniques designed to optimize the inference efficiency of LLMs by the end-user, including details on various quantization schemes, granularities, and trade-offs. The aim is to provide a balanced overview between the theory and applications of post-training quantization. 

---
# Guiding LLM-based Smart Contract Generation with Finite State Machine 

**Authors**: Hao Luo, Yuhao Lin, Xiao Yan, Xintong Hu, Yuxiang Wang, Qiming Zeng, Hao Wang, Jiawei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08542)  

**Abstract**: Smart contract is a kind of self-executing code based on blockchain technology with a wide range of application scenarios, but the traditional generation method relies on manual coding and expert auditing, which has a high threshold and low efficiency. Although Large Language Models (LLMs) show great potential in programming tasks, they still face challenges in smart contract generation w.r.t. effectiveness and security. To solve these problems, we propose FSM-SCG, a smart contract generation framework based on finite state machine (FSM) and LLMs, which significantly improves the quality of the generated code by abstracting user requirements to generate FSM, guiding LLMs to generate smart contracts, and iteratively optimizing the code with the feedback of compilation and security checks. The experimental results show that FSM-SCG significantly improves the quality of smart contract generation. Compared to the best baseline, FSM-SCG improves the compilation success rate of generated smart contract code by at most 48%, and reduces the average vulnerability risk score by approximately 68%. 

---
# On the Complexity and Properties of Preferential Propositional Dependence Logic 

**Authors**: Kai Sauerwald, Arne Meier, Juha Kontinen  

**Link**: [PDF](https://arxiv.org/pdf/2505.08522)  

**Abstract**: This paper considers the complexity and properties of KLM-style preferential reasoning in the setting of propositional logic with team semantics and dependence atoms, also known as propositional dependence logic. Preferential team-based reasoning is shown to be cumulative, yet violates System~P. We give intuitive conditions that fully characterise those cases where preferential propositional dependence logic satisfies System~P. We show that these characterisations do, surprisingly, not carry over to preferential team-based propositional logic. Furthermore, we show how classical entailment and dependence logic entailment can be expressed in terms of non-trivial preferential models. Finally, we present the complexity of preferential team-based reasoning for two natural representations. This includes novel complexity results for classical (non-team-based) preferential reasoning. 

---
# TrialMatchAI: An End-to-End AI-powered Clinical Trial Recommendation System to Streamline Patient-to-Trial Matching 

**Authors**: Majd Abdallah, Sigve Nakken, Mariska Bierkens, Johanna Galvis, Alexis Groppi, Slim Karkar, Lana Meiqari, Maria Alexandra Rujano, Steve Canham, Rodrigo Dienstmann, Remond Fijneman, Eivind Hovig, Gerrit Meijer, Macha Nikolski  

**Link**: [PDF](https://arxiv.org/pdf/2505.08508)  

**Abstract**: Patient recruitment remains a major bottleneck in clinical trials, calling for scalable and automated solutions. We present TrialMatchAI, an AI-powered recommendation system that automates patient-to-trial matching by processing heterogeneous clinical data, including structured records and unstructured physician notes. Built on fine-tuned, open-source large language models (LLMs) within a retrieval-augmented generation framework, TrialMatchAI ensures transparency and reproducibility and maintains a lightweight deployment footprint suitable for clinical environments. The system normalizes biomedical entities, retrieves relevant trials using a hybrid search strategy combining lexical and semantic similarity, re-ranks results, and performs criterion-level eligibility assessments using medical Chain-of-Thought reasoning. This pipeline delivers explainable outputs with traceable decision rationales. In real-world validation, 92 percent of oncology patients had at least one relevant trial retrieved within the top 20 recommendations. Evaluation across synthetic and real clinical datasets confirmed state-of-the-art performance, with expert assessment validating over 90 percent accuracy in criterion-level eligibility classification, particularly excelling in biomarker-driven matches. Designed for modularity and privacy, TrialMatchAI supports Phenopackets-standardized data, enables secure local deployment, and allows seamless replacement of LLM components as more advanced models emerge. By enhancing efficiency and interpretability and offering lightweight, open-source deployment, TrialMatchAI provides a scalable solution for AI-driven clinical trial matching in precision medicine. 

---
# Achieving Scalable Robot Autonomy via neurosymbolic planning using lightweight local LLM 

**Authors**: Nicholas Attolino, Alessio Capitanelli, Fulvio Mastrogiovanni  

**Link**: [PDF](https://arxiv.org/pdf/2505.08492)  

**Abstract**: PDDL-based symbolic task planning remains pivotal for robot autonomy yet struggles with dynamic human-robot collaboration due to scalability, re-planning demands, and delayed plan availability. Although a few neurosymbolic frameworks have previously leveraged LLMs such as GPT-3 to address these challenges, reliance on closed-source, remote models with limited context introduced critical constraints: third-party dependency, inconsistent response times, restricted plan length and complexity, and multi-domain scalability issues. We present Gideon, a novel framework that enables the transition to modern, smaller, local LLMs with extended context length. Gideon integrates a novel problem generator to systematically generate large-scale datasets of realistic domain-problem-plan tuples for any domain, and adapts neurosymbolic planning for local LLMs, enabling on-device execution and extended context for multi-domain support. Preliminary experiments in single-domain scenarios performed on Qwen-2.5 1.5B and trained on 8k-32k samples, demonstrate a valid plan percentage of 66.1% (32k model) and show that the figure can be further scaled through additional data. Multi-domain tests on 16k samples yield an even higher 70.6% planning validity rate, proving extensibility across domains and signaling that data variety can have a positive effect on learning efficiency. Although long-horizon planning and reduced model size make Gideon training much less efficient than baseline models based on larger LLMs, the results are still significant considering that the trained model is about 120x smaller than baseline and that significant advantages can be achieved in inference efficiency, scalability, and multi-domain adaptability, all critical factors in human-robot collaboration. Training inefficiency can be mitigated by Gideon's streamlined data generation pipeline. 

---
# BAT: Benchmark for Auto-bidding Task 

**Authors**: Alexandra Khirianova, Ekaterina Solodneva, Andrey Pudovikov, Sergey Osokin, Egor Samosvat, Yuriy Dorn, Alexander Ledovsky, Yana Zenkova  

**Link**: [PDF](https://arxiv.org/pdf/2505.08485)  

**Abstract**: The optimization of bidding strategies for online advertising slot auctions presents a critical challenge across numerous digital marketplaces. A significant obstacle to the development, evaluation, and refinement of real-time autobidding algorithms is the scarcity of comprehensive datasets and standardized benchmarks.
To address this deficiency, we present an auction benchmark encompassing the two most prevalent auction formats. We implement a series of robust baselines on a novel dataset, addressing the most salient Real-Time Bidding (RTB) problem domains: budget pacing uniformity and Cost Per Click (CPC) constraint optimization. This benchmark provides a user-friendly and intuitive framework for researchers and practitioners to develop and refine innovative autobidding algorithms, thereby facilitating advancements in the field of programmatic advertising. The implementation and additional resources can be accessed at the following repository (this https URL, this https URL). 

---
# Strategy-Augmented Planning for Large Language Models via Opponent Exploitation 

**Authors**: Shuai Xu, Sijia Cui, Yanna Wang, Bo Xu, Qi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08459)  

**Abstract**: Efficiently modeling and exploiting opponents is a long-standing challenge in adversarial domains. Large Language Models (LLMs) trained on extensive textual data have recently demonstrated outstanding performance in general tasks, introducing new research directions for opponent modeling. Some studies primarily focus on directly using LLMs to generate decisions based on the elaborate prompt context that incorporates opponent descriptions, while these approaches are limited to scenarios where LLMs possess adequate domain expertise. To address that, we introduce a two-stage Strategy-Augmented Planning (SAP) framework that significantly enhances the opponent exploitation capabilities of LLM-based agents by utilizing a critical component, the Strategy Evaluation Network (SEN). Specifically, in the offline stage, we construct an explicit strategy space and subsequently collect strategy-outcome pair data for training the SEN network. During the online phase, SAP dynamically recognizes the opponent's strategies and greedily exploits them by searching best response strategy on the well-trained SEN, finally translating strategy to a course of actions by carefully designed prompts. Experimental results show that SAP exhibits robust generalization capabilities, allowing it to perform effectively not only against previously encountered opponent strategies but also against novel, unseen strategies. In the MicroRTS environment, SAP achieves a 85.35\% performance improvement over baseline methods and matches the competitiveness of reinforcement learning approaches against state-of-the-art (SOTA) rule-based AI. 

---
# Adaptive Bias Generalized Rollout Policy Adaptation on the Flexible Job-Shop Scheduling Problem 

**Authors**: Lotfi Kobrosly, Marc-Emmanuel Coupvent des Graviers, Christophe Guettier, Tristan Cazenave  

**Link**: [PDF](https://arxiv.org/pdf/2505.08451)  

**Abstract**: The Flexible Job-Shop Scheduling Problem (FJSSP) is an NP-hard combinatorial optimization problem, with several application domains, especially for manufacturing purposes. The objective is to
efficiently schedule multiple operations on dissimilar machines. These operations are gathered into jobs, and operations pertaining to the same job need to be scheduled sequentially. Different methods have been previously tested to solve this problem, such as Constraint Solving, Tabu Search, Genetic Algorithms, or Monte Carlo Tree Search (MCTS). We propose a novel algorithm derived from the Generalized Nested Rollout Policy Adaptation, developed to solve the FJSSP. We report encouraging experimental results, as our algorithm performs better than other MCTS-based approaches, even if makespans obtained on large instances are still far from known upper bounds. 

---
# Agent-as-a-Service based on Agent Network 

**Authors**: Yuhan Zhu, Haojie Liu, Jian Wang, Bing Li, Zikang Yin, Yefei Liao  

**Link**: [PDF](https://arxiv.org/pdf/2505.08446)  

**Abstract**: The rise of large model-based AI agents has spurred interest in Multi-Agent Systems (MAS) for their capabilities in decision-making, collaboration, and adaptability. While the Model Context Protocol (MCP) addresses tool invocation and data exchange challenges via a unified protocol, it lacks support for organizing agent-level collaboration. To bridge this gap, we propose Agent-as-a-Service based on Agent Network (AaaS-AN), a service-oriented paradigm grounded in the Role-Goal-Process-Service (RGPS) standard. AaaS-AN unifies the entire agent lifecycle, including construction, integration, interoperability, and networked collaboration, through two core components: (1) a dynamic Agent Network, which models agents and agent groups as vertexes that self-organize within the network based on task and role dependencies; (2) service-oriented agents, incorporating service discovery, registration, and interoperability protocols. These are orchestrated by a Service Scheduler, which leverages an Execution Graph to enable distributed coordination, context tracking, and runtime task management. We validate AaaS-AN on mathematical reasoning and application-level code generation tasks, which outperforms state-of-the-art baselines. Notably, we constructed a MAS based on AaaS-AN containing agent groups, Robotic Process Automation (RPA) workflows, and MCP servers over 100 agent services. We also release a dataset containing 10,000 long-horizon multi-agent workflows to facilitate future research on long-chain collaboration in MAS. 

---
# Explaining Autonomous Vehicles with Intention-aware Policy Graphs 

**Authors**: Sara Montese, Victor Gimenez-Abalos, Atia Cortés, Ulises Cortés, Sergio Alvarez-Napagao  

**Link**: [PDF](https://arxiv.org/pdf/2505.08404)  

**Abstract**: The potential to improve road safety, reduce human driving error, and promote environmental sustainability have enabled the field of autonomous driving to progress rapidly over recent decades. The performance of autonomous vehicles has significantly improved thanks to advancements in Artificial Intelligence, particularly Deep Learning. Nevertheless, the opacity of their decision-making, rooted in the use of accurate yet complex AI models, has created barriers to their societal trust and regulatory acceptance, raising the need for explainability. We propose a post-hoc, model-agnostic solution to provide teleological explanations for the behaviour of an autonomous vehicle in urban environments. Building on Intention-aware Policy Graphs, our approach enables the extraction of interpretable and reliable explanations of vehicle behaviour in the nuScenes dataset from global and local perspectives. We demonstrate the potential of these explanations to assess whether the vehicle operates within acceptable legal boundaries and to identify possible vulnerabilities in autonomous driving datasets and models. 

---
# Learning Like Humans: Advancing LLM Reasoning Capabilities via Adaptive Difficulty Curriculum Learning and Expert-Guided Self-Reformulation 

**Authors**: Enci Zhang, Xingang Yan, Wei Lin, Tianxiang Zhang, Qianchun Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08364)  

**Abstract**: Despite impressive progress in areas like mathematical reasoning, large language models still face significant challenges in consistently solving complex problems. Drawing inspiration from key human learning strategies, we propose two novel strategies to enhance the capability of large language models to solve these complex problems. First, Adaptive Difficulty Curriculum Learning (ADCL) is a novel curriculum learning strategy that tackles the Difficulty Shift phenomenon (i.e., a model's perception of problem difficulty dynamically changes during training) by periodically re-estimating difficulty within upcoming data batches to maintain alignment with the model's evolving capabilities. Second, Expert-Guided Self-Reformulation (EGSR) is a novel reinforcement learning strategy that bridges the gap between imitation learning and pure exploration by guiding models to reformulate expert solutions within their own conceptual framework, rather than relying on direct imitation, fostering deeper understanding and knowledge assimilation. Extensive experiments on challenging mathematical reasoning benchmarks, using Qwen2.5-7B as the base model, demonstrate that these human-inspired strategies synergistically and significantly enhance performance. Notably, their combined application improves performance over the standard Zero-RL baseline by 10% on the AIME24 benchmark and 16.6% on AIME25. 

---
# Modeling Unseen Environments with Language-guided Composable Causal Components in Reinforcement Learning 

**Authors**: Xinyue Wang, Biwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08361)  

**Abstract**: Generalization in reinforcement learning (RL) remains a significant challenge, especially when agents encounter novel environments with unseen dynamics. Drawing inspiration from human compositional reasoning -- where known components are reconfigured to handle new situations -- we introduce World Modeling with Compositional Causal Components (WM3C). This novel framework enhances RL generalization by learning and leveraging compositional causal components. Unlike previous approaches focusing on invariant representation learning or meta-learning, WM3C identifies and utilizes causal dynamics among composable elements, facilitating robust adaptation to new tasks. Our approach integrates language as a compositional modality to decompose the latent space into meaningful components and provides theoretical guarantees for their unique identification under mild assumptions. Our practical implementation uses a masked autoencoder with mutual information constraints and adaptive sparsity regularization to capture high-level semantic information and effectively disentangle transition dynamics. Experiments on numerical simulations and real-world robotic manipulation tasks demonstrate that WM3C significantly outperforms existing methods in identifying latent processes, improving policy learning, and generalizing to unseen tasks. 

---
# An Identifiable Cost-Aware Causal Decision-Making Framework Using Counterfactual Reasoning 

**Authors**: Ruichu Cai, Xi Chen, Jie Qiao, Zijian Li, Yuequn Liu, Wei Chen, Keli Zhang, Jiale Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.08343)  

**Abstract**: Decision making under abnormal conditions is a critical process that involves evaluating the current state and determining the optimal action to restore the system to a normal state at an acceptable cost. However, in such scenarios, existing decision-making frameworks highly rely on reinforcement learning or root cause analysis, resulting in them frequently neglecting the cost of the actions or failing to incorporate causal mechanisms adequately. By relaxing the existing causal decision framework to solve the necessary cause, we propose a minimum-cost causal decision (MiCCD) framework via counterfactual reasoning to address the above challenges. Emphasis is placed on making counterfactual reasoning processes identifiable in the presence of a large amount of mixed anomaly data, as well as finding the optimal intervention state in a continuous decision space. Specifically, it formulates a surrogate model based on causal graphs, using abnormal pattern clustering labels as supervisory signals. This enables the approximation of the structural causal model among the variables and lays a foundation for identifiable counterfactual reasoning. With the causal structure approximated, we then established an optimization model based on counterfactual estimation. The Sequential Least Squares Programming (SLSQP) algorithm is further employed to optimize intervention strategies while taking costs into account. Experimental evaluations on both synthetic and real-world datasets reveal that MiCCD outperforms conventional methods across multiple metrics, including F1-score, cost efficiency, and ranking quality(nDCG@k values), thus validating its efficacy and broad applicability. 

---
# Benchmarking AI scientists in omics data-driven biological research 

**Authors**: Erpai Luo, Jinmeng Jia, Yifan Xiong, Xiangyu Li, Xiaobo Guo, Baoqi Yu, Lei Wei, Xuegong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08341)  

**Abstract**: The rise of large language models and multi-agent systems has sparked growing interest in AI scientists capable of autonomous biological research. However, existing benchmarks either focus on reasoning without data or on data analysis with predefined statistical answers, lacking realistic, data-driven evaluation settings. Here, we introduce the Biological AI Scientist Benchmark (BaisBench), a benchmark designed to assess AI scientists' ability to generate biological discoveries through data analysis and reasoning with external knowledge. BaisBench comprises two tasks: cell type annotation on 31 expert-labeled single-cell datasets, and scientific discovery through answering 198 multiple-choice questions derived from the biological insights of 41 recent single-cell studies. Systematic experiments on state-of-the-art AI scientists and LLM agents showed that while promising, current models still substantially underperform human experts on both tasks. We hope BaisBench will fill this gap and serve as a foundation for advancing and evaluating AI models for scientific discovery. The benchmark can be found at: this https URL. 

---
# Evaluating LLM Metrics Through Real-World Capabilities 

**Authors**: Justin K Miller, Wenjia Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08253)  

**Abstract**: As generative AI becomes increasingly embedded in everyday workflows, it is important to evaluate its performance in ways that reflect real-world usage rather than abstract notions of intelligence. Unlike many existing benchmarks that assess general intelligence, our approach focuses on real-world utility, evaluating how well models support users in everyday tasks. While current benchmarks emphasize code generation or factual recall, users rely on AI for a much broader range of activities-from writing assistance and summarization to citation formatting and stylistic feedback. In this paper, we analyze large-scale survey data and usage logs to identify six core capabilities that represent how people commonly use Large Language Models (LLMs): Summarization, Technical Assistance, Reviewing Work, Data Structuring, Generation, and Information Retrieval. We then assess the extent to which existing benchmarks cover these capabilities, revealing significant gaps in coverage, efficiency measurement, and interpretability. Drawing on this analysis, we use human-centered criteria to identify gaps in how well current benchmarks reflect common usage that is grounded in five practical criteria: coherence, accuracy, clarity, relevance, and efficiency. For four of the six capabilities, we identify the benchmarks that best align with real-world tasks and use them to compare leading models. We find that Google Gemini outperforms other models-including OpenAI's GPT, xAI's Grok, Meta's LLaMA, Anthropic's Claude, DeepSeek, and Qwen from Alibaba-on these utility-focused metrics. 

---
# Unveiling the Best Practices for Applying Speech Foundation Models to Speech Intelligibility Prediction for Hearing-Impaired People 

**Authors**: Haoshuai Zhou, Boxuan Cao, Changgeng Mo, Linkai Li, Shan Xiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08215)  

**Abstract**: Speech foundation models (SFMs) have demonstrated strong performance across a variety of downstream tasks, including speech intelligibility prediction for hearing-impaired people (SIP-HI). However, optimizing SFMs for SIP-HI has been insufficiently explored. In this paper, we conduct a comprehensive study to identify key design factors affecting SIP-HI performance with 5 SFMs, focusing on encoder layer selection, prediction head architecture, and ensemble configurations. Our findings show that, contrary to traditional use-all-layers methods, selecting a single encoder layer yields better results. Additionally, temporal modeling is crucial for effective prediction heads. We also demonstrate that ensembling multiple SFMs improves performance, with stronger individual models providing greater benefit. Finally, we explore the relationship between key SFM attributes and their impact on SIP-HI performance. Our study offers practical insights into effectively adapting SFMs for speech intelligibility prediction for hearing-impaired populations. 

---
# Behind the Noise: Conformal Quantile Regression Reveals Emergent Representations 

**Authors**: Petrus H. Zwart, Tamas Varga, Odeta Qafoku, James A. Sethian  

**Link**: [PDF](https://arxiv.org/pdf/2505.08176)  

**Abstract**: Scientific imaging often involves long acquisition times to obtain high-quality data, especially when probing complex, heterogeneous systems. However, reducing acquisition time to increase throughput inevitably introduces significant noise into the measurements. We present a machine learning approach that not only denoises low-quality measurements with calibrated uncertainty bounds, but also reveals emergent structure in the latent space. By using ensembles of lightweight, randomly structured neural networks trained via conformal quantile regression, our method performs reliable denoising while uncovering interpretable spatial and chemical features -- without requiring labels or segmentation. Unlike conventional approaches focused solely on image restoration, our framework leverages the denoising process itself to drive the emergence of meaningful representations. We validate the approach on real-world geobiochemical imaging data, showing how it supports confident interpretation and guides experimental design under resource constraints. 

---
# Decoding Neighborhood Environments with Large Language Models 

**Authors**: Andrew Cart, Shaohu Zhang, Melanie Escue, Xugui Zhou, Haitao Zhao, Prashanth BusiReddyGari, Beiyu Lin, Shuang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.08163)  

**Abstract**: Neighborhood environments include physical and environmental conditions such as housing quality, roads, and sidewalks, which significantly influence human health and well-being. Traditional methods for assessing these environments, including field surveys and geographic information systems (GIS), are resource-intensive and challenging to evaluate neighborhood environments at scale. Although machine learning offers potential for automated analysis, the laborious process of labeling training data and the lack of accessible models hinder scalability. This study explores the feasibility of large language models (LLMs) such as ChatGPT and Gemini as tools for decoding neighborhood environments (e.g., sidewalk and powerline) at scale. We train a robust YOLOv11-based model, which achieves an average accuracy of 99.13% in detecting six environmental indicators, including streetlight, sidewalk, powerline, apartment, single-lane road, and multilane road. We then evaluate four LLMs, including ChatGPT, Gemini, Claude, and Grok, to assess their feasibility, robustness, and limitations in identifying these indicators, with a focus on the impact of prompting strategies and fine-tuning. We apply majority voting with the top three LLMs to achieve over 88% accuracy, which demonstrates LLMs could be a useful tool to decode the neighborhood environment without any training effort. 

---
# Efficient and Scalable Neural Symbolic Search for Knowledge Graph Complex Query Answering 

**Authors**: Weizhi Fei, Zihao Wang, hang Yin, Shukai Zhao, Wei Zhang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.08155)  

**Abstract**: Complex Query Answering (CQA) aims to retrieve answer sets for complex logical formulas from incomplete knowledge graphs, which is a crucial yet challenging task in knowledge graph reasoning. While neuro-symbolic search utilized neural link predictions achieve superior accuracy, they encounter significant complexity bottlenecks: (i) Data complexity typically scales quadratically with the number of entities in the knowledge graph, and (ii) Query complexity becomes NP-hard for cyclic queries. Consequently, these approaches struggle to effectively scale to larger knowledge graphs and more complex queries. To address these challenges, we propose an efficient and scalable symbolic search framework. First, we propose two constraint strategies to compute neural logical indices to reduce the domain of variables, thereby decreasing the data complexity of symbolic search. Additionally, we introduce an approximate algorithm based on local search to tackle the NP query complexity of cyclic queries. Experiments on various CQA benchmarks demonstrate that our framework reduces the computational load of symbolic methods by 90\% while maintaining nearly the same performance, thus alleviating both efficiency and scalability issues. 

---
# Foundation Models Knowledge Distillation For Battery Capacity Degradation Forecast 

**Authors**: Joey Chan, Zhen Chen, Ershun Pan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08151)  

**Abstract**: Accurate estimation of lithium-ion battery capacity degradation is critical for enhancing the reliability and safety of battery operations. Traditional expert models, tailored to specific scenarios, provide isolated estimations. With the rapid advancement of data-driven techniques, a series of general-purpose time-series foundation models have been developed. However, foundation models specifically designed for battery capacity degradation remain largely unexplored. To enable zero-shot generalization in battery degradation prediction using large model technology, this study proposes a degradation-aware fine-tuning strategy for time-series foundation models. We apply this strategy to fine-tune the Timer model on approximately 10 GB of open-source battery charge discharge data. Validation on our released CycleLife-SJTUIE dataset demonstrates that the fine-tuned Battery-Timer possesses strong zero-shot generalization capability in capacity degradation forecasting. To address the computational challenges of deploying large models, we further propose a knowledge distillation framework that transfers the knowledge of pre-trained foundation models into compact expert models. Distillation results across several state-of-the-art time-series expert models confirm that foundation model knowledge significantly improves the multi-condition generalization of expert models. 

---
# Lost in Transmission: When and Why LLMs Fail to Reason Globally 

**Authors**: Tobias Schnabel, Kiran Tomlinson, Adith Swaminathan, Jennifer Neville  

**Link**: [PDF](https://arxiv.org/pdf/2505.08140)  

**Abstract**: Despite their many successes, transformer-based large language models (LLMs) continue to struggle with tasks that require complex reasoning over large parts of their input. We argue that these failures arise due to capacity limits on the accurate flow of information within LLMs. To formalize this issue, we introduce the bounded attention prefix oracle (BAPO) model, a new computational framework that models bandwidth constraints on attention heads, the mechanism for internal communication in LLMs. We show that several important reasoning problems like graph reachability require high communication bandwidth for BAPOs to solve; we call these problems BAPO-hard. Our experiments corroborate our theoretical predictions: GPT-4, Claude, and Gemini succeed on BAPO-easy tasks and fail even on relatively small BAPO-hard tasks. BAPOs also reveal another benefit of chain of thought (CoT): we prove that breaking down a task using CoT can turn any BAPO-hard problem into a BAPO-easy one. Our results offer principled explanations for key LLM failures and suggest directions for architectures and inference methods that mitigate bandwidth limits. 

---
# Explainable Reinforcement Learning Agents Using World Models 

**Authors**: Madhuri Singh, Amal Alabdulkarim, Gennie Mansi, Mark O. Riedl  

**Link**: [PDF](https://arxiv.org/pdf/2505.08073)  

**Abstract**: Explainable AI (XAI) systems have been proposed to help people understand how AI systems produce outputs and behaviors. Explainable Reinforcement Learning (XRL) has an added complexity due to the temporal nature of sequential decision-making. Further, non-AI experts do not necessarily have the ability to alter an agent or its policy. We introduce a technique for using World Models to generate explanations for Model-Based Deep RL agents. World Models predict how the world will change when actions are performed, allowing for the generation of counterfactual trajectories. However, identifying what a user wanted the agent to do is not enough to understand why the agent did something else. We augment Model-Based RL agents with a Reverse World Model, which predicts what the state of the world should have been for the agent to prefer a given counterfactual action. We show that explanations that show users what the world should have been like significantly increase their understanding of the agent policy. We hypothesize that our explanations can help users learn how to control the agents execution through by manipulating the environment. 

---
# Bias or Optimality? Disentangling Bayesian Inference and Learning Biases in Human Decision-Making 

**Authors**: Prakhar Godara  

**Link**: [PDF](https://arxiv.org/pdf/2505.08049)  

**Abstract**: Recent studies claim that human behavior in a two-armed Bernoulli bandit (TABB) task is described by positivity and confirmation biases, implying that humans do not integrate new information objectively. However, we find that even if the agent updates its belief via objective Bayesian inference, fitting the standard Q-learning model with asymmetric learning rates still recovers both biases. Bayesian inference cast as an effective Q-learning algorithm has symmetric, though decreasing, learning rates. We explain this by analyzing the stochastic dynamics of these learning systems using master equations. We find that both confirmation bias and unbiased but decreasing learning rates yield the same behavioral signatures. Finally, we propose experimental protocols to disentangle true cognitive biases from artifacts of decreasing learning rates. 

---
# The Correspondence Between Bounded Graph Neural Networks and Fragments of First-Order Logic 

**Authors**: Bernardo Cuenca Grau, Przemysław A. Wałęga  

**Link**: [PDF](https://arxiv.org/pdf/2505.08021)  

**Abstract**: Graph Neural Networks (GNNs) address two key challenges in applying deep learning to graph-structured data: they handle varying size input graphs and ensure invariance under graph isomorphism. While GNNs have demonstrated broad applicability, understanding their expressive power remains an important question. In this paper, we show that bounded GNN architectures correspond to specific fragments of first-order logic (FO), including modal logic (ML), graded modal logic (GML), modal logic with the universal modality (ML(A)), the two-variable fragment (FO2) and its extension with counting quantifiers (C2). To establish these results, we apply methods and tools from finite model theory of first-order and modal logics to the domain of graph representation learning. This provides a unifying framework for understanding the logical expressiveness of GNNs within FO. 

---
# Enhancing Trust Management System for Connected Autonomous Vehicles Using Machine Learning Methods: A Survey 

**Authors**: Qian Xu, Lei Zhang, Yixiao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07882)  

**Abstract**: Connected Autonomous Vehicles (CAVs) operate in dynamic, open, and multi-domain networks, rendering them vulnerable to various threats. Trust Management Systems (TMS) systematically organize essential steps in the trust mechanism, identifying malicious nodes against internal threats and external threats, as well as ensuring reliable decision-making for more cooperative tasks. Recent advances in machine learning (ML) offer significant potential to enhance TMS, especially for the strict requirements of CAVs, such as CAV nodes moving at varying speeds, and opportunistic and intermittent network behavior. Those features distinguish ML-based TMS from social networks, static IoT, and Social IoT. This survey proposes a novel three-layer ML-based TMS framework for CAVs in the vehicle-road-cloud integration system, i.e., trust data layer, trust calculation layer and trust incentive layer. A six-dimensional taxonomy of objectives is proposed. Furthermore, the principles of ML methods for each module in each layer are analyzed. Then, recent studies are categorized based on traffic scenarios that are against the proposed objectives. Finally, future directions are suggested, addressing the open issues and meeting the research trend. We maintain an active repository that contains up-to-date literature and open-source projects at this https URL. 

---
# Arrow-Guided VLM: Enhancing Flowchart Understanding via Arrow Direction Encoding 

**Authors**: Takamitsu Omasa, Ryo Koshihara, Masumi Morishige  

**Link**: [PDF](https://arxiv.org/pdf/2505.07864)  

**Abstract**: Flowcharts are indispensable tools in software design and business-process analysis, yet current vision-language models (VLMs) frequently misinterpret the directional arrows and graph topology that set these diagrams apart from natural images. We introduce a seven-stage pipeline grouped into three broader processes: (1) arrow-aware detection of nodes and arrow endpoints; (2) optical character recognition (OCR) to extract node text; and (3) construction of a structured prompt that guides the VLMs. Tested on a 90-question benchmark distilled from 30 annotated flowcharts, the method raises overall accuracy from 80 % to 89 % (+9 percentage points) without any task-specific fine-tuning. The gain is most pronounced for next-step queries (25/30 -> 30/30; 100 %, +17 pp); branch-result questions improve more modestly, and before-step questions remain difficult. A parallel evaluation with an LLM-as-a-Judge protocol shows the same trends, reinforcing the advantage of explicit arrow encoding. Limitations include dependence on detector and OCR precision, the small evaluation set, and residual errors at nodes with multiple incoming edges. Future work will enlarge the benchmark with synthetic and handwritten flowcharts and assess the approach on Business Process Model and Notation (BPMN) and Unified Modeling Language (UML). 

---
# CCL: Collaborative Curriculum Learning for Sparse-Reward Multi-Agent Reinforcement Learning via Co-evolutionary Task Evolution 

**Authors**: Yufei Lin, Chengwei Ye, Huanzhen Zhang, Kangsheng Wang, Linuo Xu, Shuyan Liu, Zeyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07854)  

**Abstract**: Sparse reward environments pose significant challenges in reinforcement learning, especially within multi-agent systems (MAS) where feedback is delayed and shared across agents, leading to suboptimal learning. We propose Collaborative Multi-dimensional Course Learning (CCL), a novel curriculum learning framework that addresses this by (1) refining intermediate tasks for individual agents, (2) using a variational evolutionary algorithm to generate informative subtasks, and (3) co-evolving agents with their environment to enhance training stability. Experiments on five cooperative tasks in the MPE and Hide-and-Seek environments show that CCL outperforms existing methods in sparse reward settings. 

---
# Conceptual Logical Foundations of Artificial Social Intelligence 

**Authors**: Eric Werner  

**Link**: [PDF](https://arxiv.org/pdf/2505.07847)  

**Abstract**: What makes a society possible at all? How is coordination and cooperation in social activity possible? What is the minimal mental architecture of a social agent? How is the information about the state of the world related to the agents intentions? How are the intentions of agents related? What role does communication play in this coordination process? This essay explores the conceptual and logical foundations of artificial social intelligence in the context of a society of multiple agents that communicate and cooperate to achieve some end. An attempt is made to provide an introduction to some of the key concepts, their formal definitions and their interrelationships. These include the notion of a changing social world of multiple agents. The logic of social intelligence goes beyond classical logic by linking information with strategic thought. A minimal architecture of social agents is presented. The agents have different dynamically changing, possible choices and abilities. The agents also have uncertainty, lacking perfect information about their physical state as well as their dynamic social state. The social state of an agent includes the intentional state of that agent, as well as, that agent's representation of the intentional states of other agents. Furthermore, it includes the evaluations agents make of their physical and social condition. Communication, semantic and pragmatic meaning and their relationship to intention and information states are investigated. The logic of agent abilities and intentions are motivated and formalized. The entropy of group strategic states is defined. 

---
# Winning at All Cost: A Small Environment for Eliciting Specification Gaming Behaviors in Large Language Models 

**Authors**: Lars Malmqvist  

**Link**: [PDF](https://arxiv.org/pdf/2505.07846)  

**Abstract**: This study reveals how frontier Large Language Models LLMs can "game the system" when faced with impossible situations, a critical security and alignment concern. Using a novel textual simulation approach, we presented three leading LLMs (o1, o3-mini, and r1) with a tic-tac-toe scenario designed to be unwinnable through legitimate play, then analyzed their tendency to exploit loopholes rather than accept defeat. Our results are alarming for security researchers: the newer, reasoning-focused o3-mini model showed nearly twice the propensity to exploit system vulnerabilities (37.1%) compared to the older o1 model (17.5%). Most striking was the effect of prompting. Simply framing the task as requiring "creative" solutions caused gaming behaviors to skyrocket to 77.3% across all models. We identified four distinct exploitation strategies, from direct manipulation of game state to sophisticated modification of opponent behavior. These findings demonstrate that even without actual execution capabilities, LLMs can identify and propose sophisticated system exploits when incentivized, highlighting urgent challenges for AI alignment as models grow more capable of identifying and leveraging vulnerabilities in their operating environments. 

---
# RAN Cortex: Memory-Augmented Intelligence for Context-Aware Decision-Making in AI-Native Networks 

**Authors**: Sebastian Barros  

**Link**: [PDF](https://arxiv.org/pdf/2505.07842)  

**Abstract**: As Radio Access Networks (RAN) evolve toward AI-native architectures, intelligent modules such as xApps and rApps are expected to make increasingly autonomous decisions across scheduling, mobility, and resource management domains. However, these agents remain fundamentally stateless, treating each decision as isolated, lacking any persistent memory of prior events or outcomes. This reactive behavior constrains optimization, especially in environments where network dynamics exhibit episodic or recurring patterns. In this work, we propose RAN Cortex, a memory-augmented architecture that enables contextual recall in AI-based RAN decision systems. RAN Cortex introduces a modular layer composed of four elements: a context encoder that transforms network state into high-dimensional embeddings, a vector-based memory store of past network episodes, a recall engine to retrieve semantically similar situations, and a policy interface that supplies historical context to AI agents in real time or near-real time. We formalize the retrieval-augmented decision problem in the RAN, present a system architecture compatible with O-RAN interfaces, and analyze feasible deployments within the Non-RT and Near-RT RIC domains. Through illustrative use cases such as stadium traffic mitigation and mobility management in drone corridors, we demonstrate how contextual memory improves adaptability, continuity, and overall RAN intelligence. This work introduces memory as a missing primitive in AI-native RAN designs and provides a framework to enable "learning agents" without the need for retraining or centralized inference 

---
# An Optimized Evacuation Plan for an Active-Shooter Situation Constrained by Network Capacity 

**Authors**: Joseph Lavalle-Rivera, Aniirudh Ramesh, Subhadeep Chakraborty  

**Link**: [PDF](https://arxiv.org/pdf/2505.07830)  

**Abstract**: A total of more than 3400 public shootings have occurred in the United States between 2016 and 2022. Among these, 25.1% of them took place in an educational institution, 29.4% at the workplace including office buildings, 19.6% in retail store locations, and 13.4% in restaurants and bars. During these critical scenarios, making the right decisions while evacuating can make the difference between life and death. However, emergency evacuation is intensely stressful, which along with the lack of verifiable real-time information may lead to fatal incorrect decisions. To tackle this problem, we developed a multi-route routing optimization algorithm that determines multiple optimal safe routes for each evacuee while accounting for available capacity along the route, thus reducing the threat of crowding and bottlenecking. Overall, our algorithm reduces the total casualties by 34.16% and 53.3%, compared to our previous routing algorithm without capacity constraints and an expert-advised routing strategy respectively. Further, our approach to reduce crowding resulted in an approximate 50% reduction in occupancy in key bottlenecking nodes compared to both of the other evacuation algorithms. 

---
# CodePDE: An Inference Framework for LLM-driven PDE Solver Generation 

**Authors**: Shanda Li, Tanya Marwah, Junhong Shen, Weiwei Sun, Andrej Risteski, Yiming Yang, Ameet Talwalkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08783)  

**Abstract**: Partial differential equations (PDEs) are fundamental to modeling physical systems, yet solving them remains a complex challenge. Traditional numerical solvers rely on expert knowledge to implement and are computationally expensive, while neural-network-based solvers require large training datasets and often lack interpretability. In this work, we frame PDE solving as a code generation task and introduce CodePDE, the first inference framework for generating PDE solvers using large language models (LLMs). Leveraging advanced inference-time algorithms and scaling strategies, CodePDE unlocks critical capacities of LLM for PDE solving: reasoning, debugging, selfrefinement, and test-time scaling -- all without task-specific tuning. CodePDE achieves superhuman performance across a range of representative PDE problems. We also present a systematic empirical analysis of LLM generated solvers, analyzing their accuracy, efficiency, and numerical scheme choices. Our findings highlight the promise and the current limitations of LLMs in PDE solving, offering a new perspective on solver design and opportunities for future model development. Our code is available at this https URL. 

---
# Towards Autonomous UAV Visual Object Search in City Space: Benchmark and Agentic Methodology 

**Authors**: Yatai Ji, Zhengqiu Zhu, Yong Zhao, Beidan Liu, Chen Gao, Yihao Zhao, Sihang Qiu, Yue Hu, Quanjun Yin, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.08765)  

**Abstract**: Aerial Visual Object Search (AVOS) tasks in urban environments require Unmanned Aerial Vehicles (UAVs) to autonomously search for and identify target objects using visual and textual cues without external guidance. Existing approaches struggle in complex urban environments due to redundant semantic processing, similar object distinction, and the exploration-exploitation dilemma. To bridge this gap and support the AVOS task, we introduce CityAVOS, the first benchmark dataset for autonomous search of common urban objects. This dataset comprises 2,420 tasks across six object categories with varying difficulty levels, enabling comprehensive evaluation of UAV agents' search capabilities. To solve the AVOS tasks, we also propose PRPSearcher (Perception-Reasoning-Planning Searcher), a novel agentic method powered by multi-modal large language models (MLLMs) that mimics human three-tier cognition. Specifically, PRPSearcher constructs three specialized maps: an object-centric dynamic semantic map enhancing spatial perception, a 3D cognitive map based on semantic attraction values for target reasoning, and a 3D uncertainty map for balanced exploration-exploitation search. Also, our approach incorporates a denoising mechanism to mitigate interference from similar objects and utilizes an Inspiration Promote Thought (IPT) prompting mechanism for adaptive action planning. Experimental results on CityAVOS demonstrate that PRPSearcher surpasses existing baselines in both success rate and search efficiency (on average: +37.69% SR, +28.96% SPL, -30.69% MSS, and -46.40% NE). While promising, the performance gap compared to humans highlights the need for better semantic reasoning and spatial exploration capabilities in AVOS tasks. This work establishes a foundation for future advances in embodied target search. Dataset and source code are available at this https URL. 

---
# Advancing Food Nutrition Estimation via Visual-Ingredient Feature Fusion 

**Authors**: Huiyan Qi, Bin Zhu, Chong-Wah Ngo, Jingjing Chen, Ee-Peng Lim  

**Link**: [PDF](https://arxiv.org/pdf/2505.08747)  

**Abstract**: Nutrition estimation is an important component of promoting healthy eating and mitigating diet-related health risks. Despite advances in tasks such as food classification and ingredient recognition, progress in nutrition estimation is limited due to the lack of datasets with nutritional annotations. To address this issue, we introduce FastFood, a dataset with 84,446 images across 908 fast food categories, featuring ingredient and nutritional annotations. In addition, we propose a new model-agnostic Visual-Ingredient Feature Fusion (VIF$^2$) method to enhance nutrition estimation by integrating visual and ingredient features. Ingredient robustness is improved through synonym replacement and resampling strategies during training. The ingredient-aware visual feature fusion module combines ingredient features and visual representation to achieve accurate nutritional prediction. During testing, ingredient predictions are refined using large multimodal models by data augmentation and majority voting. Our experiments on both FastFood and Nutrition5k datasets validate the effectiveness of our proposed method built in different backbones (e.g., Resnet, InceptionV3 and ViT), which demonstrates the importance of ingredient information in nutrition estimation. this https URL. 

---
# Securing RAG: A Risk Assessment and Mitigation Framework 

**Authors**: Lukas Ammann, Sara Ott, Christoph R. Landolt, Marco P. Lehmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.08728)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as the de facto industry standard for user-facing NLP applications, offering the ability to integrate data without re-training or fine-tuning Large Language Models (LLMs). This capability enhances the quality and accuracy of responses but also introduces novel security and privacy challenges, particularly when sensitive data is integrated. With the rapid adoption of RAG, securing data and services has become a critical priority. This paper first reviews the vulnerabilities of RAG pipelines, and outlines the attack surface from data pre-processing and data storage management to integration with LLMs. The identified risks are then paired with corresponding mitigations in a structured overview. In a second step, the paper develops a framework that combines RAG-specific security considerations, with existing general security guidelines, industry standards, and best practices. The proposed framework aims to guide the implementation of robust, compliant, secure, and trustworthy RAG systems. 

---
# Memorization-Compression Cycles Improve Generalization 

**Authors**: Fangyuan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08727)  

**Abstract**: We prove theoretically that generalization improves not only through data scaling but also by compressing internal representations. To operationalize this insight, we introduce the Information Bottleneck Language Modeling (IBLM) objective, which reframes language modeling as a constrained optimization problem: minimizing representation entropy subject to optimal prediction performance. Empirically, we observe an emergent memorization-compression cycle during LLM pretraining, evidenced by oscillation positive/negative gradient alignment between cross-entropy and Matrix-Based Entropy (MBE), a measure of representation entropy. This pattern closely mirrors the predictive-compressive trade-off prescribed by IBLM and also parallels the biological alternation between awake learning and sleep consolidation. Motivated by this observation, we propose Gated Phase Transition (GAPT), a training algorithm that adaptively switches between memorization and compression phases. When applied to GPT-2 pretraining on FineWeb dataset, GAPT reduces MBE by 50% and improves cross-entropy by 4.8%. GAPT improves OOD generalizatino by 35% in a pretraining task on arithmetic multiplication. In a setting designed to simulate catastrophic forgetting, GAPT reduces interference by compressing and separating representations, achieving a 97% improvement in separation - paralleling the functional role of sleep consolidation. 

---
# PWC-MoE: Privacy-Aware Wireless Collaborative Mixture of Experts 

**Authors**: Yang Su, Na Yan, Yansha Deng, Robert Schober  

**Link**: [PDF](https://arxiv.org/pdf/2505.08719)  

**Abstract**: Large language models (LLMs) hosted on cloud servers alleviate the computational and storage burdens on local devices but raise privacy concerns due to sensitive data transmission and require substantial communication bandwidth, which is challenging in constrained environments. In contrast, small language models (SLMs) running locally enhance privacy but suffer from limited performance on complex tasks. To balance computational cost, performance, and privacy protection under bandwidth constraints, we propose a privacy-aware wireless collaborative mixture of experts (PWC-MoE) framework. Specifically, PWC-MoE employs a sparse privacy-aware gating network to dynamically route sensitive tokens to privacy experts located on local clients, while non-sensitive tokens are routed to non-privacy experts located at the remote base station. To achieve computational efficiency, the gating network ensures that each token is dynamically routed to and processed by only one expert. To enhance scalability and prevent overloading of specific experts, we introduce a group-wise load-balancing mechanism for the gating network that evenly distributes sensitive tokens among privacy experts and non-sensitive tokens among non-privacy experts. To adapt to bandwidth constraints while preserving model performance, we propose a bandwidth-adaptive and importance-aware token offloading scheme. This scheme incorporates an importance predictor to evaluate the importance scores of non-sensitive tokens, prioritizing the most important tokens for transmission to the base station based on their predicted importance and the available bandwidth. Experiments demonstrate that the PWC-MoE framework effectively preserves privacy and maintains high performance even in bandwidth-constrained environments, offering a practical solution for deploying LLMs in privacy-sensitive and bandwidth-limited scenarios. 

---
# Big Data and the Computational Social Science of Entrepreneurship and Innovation 

**Authors**: Ningzi Li, Shiyang Lai, James Evans  

**Link**: [PDF](https://arxiv.org/pdf/2505.08706)  

**Abstract**: As large-scale social data explode and machine-learning methods evolve, scholars of entrepreneurship and innovation face new research opportunities but also unique challenges. This chapter discusses the difficulties of leveraging large-scale data to identify technological and commercial novelty, document new venture origins, and forecast competition between new technologies and commercial forms. It suggests how scholars can take advantage of new text, network, image, audio, and video data in two distinct ways that advance innovation and entrepreneurship research. First, machine-learning models, combined with large-scale data, enable the construction of precision measurements that function as system-level observatories of innovation and entrepreneurship across human societies. Second, new artificial intelligence models fueled by big data generate 'digital doubles' of technology and business, forming laboratories for virtual experimentation about innovation and entrepreneurship processes and policies. The chapter argues for the advancement of theory development and testing in entrepreneurship and innovation by coupling big data with big models. 

---
# Controllable Image Colorization with Instance-aware Texts and Masks 

**Authors**: Yanru An, Ling Gui, Qiang Hu, Chunlei Cai, Tianxiao Ye, Xiaoyun Zhang, Yanfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08705)  

**Abstract**: Recently, the application of deep learning in image colorization has received widespread attention. The maturation of diffusion models has further advanced the development of image colorization models. However, current mainstream image colorization models still face issues such as color bleeding and color binding errors, and cannot colorize images at the instance level. In this paper, we propose a diffusion-based colorization method MT-Color to achieve precise instance-aware colorization with use-provided guidance. To tackle color bleeding issue, we design a pixel-level mask attention mechanism that integrates latent features and conditional gray image features through cross-attention. We use segmentation masks to construct cross-attention masks, preventing pixel information from exchanging between different instances. We also introduce an instance mask and text guidance module that extracts instance masks and text representations of each instance, which are then fused with latent features through self-attention, utilizing instance masks to form self-attention masks to prevent instance texts from guiding the colorization of other areas, thus mitigating color binding errors. Furthermore, we apply a multi-instance sampling strategy, which involves sampling each instance region separately and then fusing the results. Additionally, we have created a specialized dataset for instance-level colorization tasks, GPT-color, by leveraging large visual language models on existing image datasets. Qualitative and quantitative experiments show that our model and dataset outperform previous methods and datasets. 

---
# A Survey of Deep Learning for Complex Speech Spectrograms 

**Authors**: Yuying Xie, Zheng-Hua Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08694)  

**Abstract**: Recent advancements in deep learning have significantly impacted the field of speech signal processing, particularly in the analysis and manipulation of complex spectrograms. This survey provides a comprehensive overview of the state-of-the-art techniques leveraging deep neural networks for processing complex spectrograms, which encapsulate both magnitude and phase information. We begin by introducing complex spectrograms and their associated features for various speech processing tasks. Next, we explore the key components and architectures of complex-valued neural networks, which are specifically designed to handle complex-valued data and have been applied for complex spectrogram processing. We then discuss various training strategies and loss functions tailored for training neural networks to process and model complex spectrograms. The survey further examines key applications, including phase retrieval, speech enhancement, and speech separation, where deep learning has achieved significant progress by leveraging complex spectrograms or their derived feature representations. Additionally, we examine the intersection of complex spectrograms with generative models. This survey aims to serve as a valuable resource for researchers and practitioners in the field of speech signal processing and complex-valued neural networks. 

---
# VizCV: AI-assisted visualization of researchers' publications tracks 

**Authors**: Vladimír Lazárik, Marco Agus, Barbora Kozlíková, Pere-Pau Vázquez  

**Link**: [PDF](https://arxiv.org/pdf/2505.08691)  

**Abstract**: Analyzing how the publication records of scientists and research groups have evolved over the years is crucial for assessing their expertise since it can support the management of academic environments by assisting with career planning and evaluation. We introduce VizCV, a novel web-based end-to-end visual analytics framework that enables the interactive exploration of researchers' scientific trajectories. It incorporates AI-assisted analysis and supports automated reporting of career evolution. Our system aims to model career progression through three key dimensions: a) research topic evolution to detect and visualize shifts in scholarly focus over time, b) publication record and the corresponding impact, c) collaboration dynamics depicting the growth and transformation of a researcher's co-authorship network. AI-driven insights provide automated explanations of career transitions, detecting significant shifts in research direction, impact surges, or collaboration expansions. The system also supports comparative analysis between researchers, allowing users to compare topic trajectories and impact growth. Our interactive, multi-tab and multiview system allows for the exploratory analysis of career milestones under different perspectives, such as the most impactful articles, emerging research themes, or obtaining a detailed analysis of the contribution of the researcher in a subfield. The key contributions include AI/ML techniques for: a) topic analysis, b) dimensionality reduction for visualizing patterns and trends, c) the interactive creation of textual descriptions of facets of data through configurable prompt generation and large language models, that include key indicators, to help understanding the career development of individuals or groups. 

---
# AC-PKAN: Attention-Enhanced and Chebyshev Polynomial-Based Physics-Informed Kolmogorov-Arnold Networks 

**Authors**: Hangwei Zhang, Zhimu Huang, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08687)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) have recently shown promise for solving partial differential equations (PDEs). Yet their original formulation is computationally and memory intensive, motivating the introduction of Chebyshev Type-I-based KANs (Chebyshev1KANs). Although Chebyshev1KANs have outperformed the vanilla KANs architecture, our rigorous theoretical analysis reveals that they still suffer from rank collapse, ultimately limiting their expressive capacity. To overcome these limitations, we enhance Chebyshev1KANs by integrating wavelet-activated MLPs with learnable parameters and an internal attention mechanism. We prove that this design preserves a full-rank Jacobian and is capable of approximating solutions to PDEs of arbitrary order. Furthermore, to alleviate the loss instability and imbalance introduced by the Chebyshev polynomial basis, we externally incorporate a Residual Gradient Attention (RGA) mechanism that dynamically re-weights individual loss terms according to their gradient norms and residual magnitudes. By jointly leveraging internal and external attention, we present AC-PKAN, a novel architecture that constitutes an enhancement to weakly supervised Physics-Informed Neural Networks (PINNs) and extends the expressive power of KANs. Experimental results from nine benchmark tasks across three domains show that AC-PKAN consistently outperforms or matches state-of-the-art models such as PINNsFormer, establishing it as a highly effective tool for solving complex real-world engineering problems in zero-data or data-sparse regimes. The code will be made publicly available upon acceptance. 

---
# A Mamba-based Network for Semi-supervised Singing Melody Extraction Using Confidence Binary Regularization 

**Authors**: Xiaoliang He, Kangjie Dong, Jingkai Cao, Shuai Yu, Wei Li, Yi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08681)  

**Abstract**: Singing melody extraction (SME) is a key task in the field of music information retrieval. However, existing methods are facing several limitations: firstly, prior models use transformers to capture the contextual dependencies, which requires quadratic computation resulting in low efficiency in the inference stage. Secondly, prior works typically rely on frequencysupervised methods to estimate the fundamental frequency (f0), which ignores that the musical performance is actually based on notes. Thirdly, transformers typically require large amounts of labeled data to achieve optimal performances, but the SME task lacks of sufficient annotated data. To address these issues, in this paper, we propose a mamba-based network, called SpectMamba, for semi-supervised singing melody extraction using confidence binary regularization. In particular, we begin by introducing vision mamba to achieve computational linear complexity. Then, we propose a novel note-f0 decoder that allows the model to better mimic the musical performance. Further, to alleviate the scarcity of the labeled data, we introduce a confidence binary regularization (CBR) module to leverage the unlabeled data by maximizing the probability of the correct classes. The proposed method is evaluated on several public datasets and the conducted experiments demonstrate the effectiveness of our proposed method. 

---
# A Social Robot with Inner Speech for Dietary Guidance 

**Authors**: Valerio Belcamino, Alessandro Carfì, Valeria Seidita, Fulvio Mastrogiovanni, Antonio Chella  

**Link**: [PDF](https://arxiv.org/pdf/2505.08664)  

**Abstract**: We explore the use of inner speech as a mechanism to enhance transparency and trust in social robots for dietary advice. In humans, inner speech structures thought processes and decision-making; in robotics, it improves explainability by making reasoning explicit. This is crucial in healthcare scenarios, where trust in robotic assistants depends on both accurate recommendations and human-like dialogue, which make interactions more natural and engaging. Building on this, we developed a social robot that provides dietary advice, and we provided the architecture with inner speech capabilities to validate user input, refine reasoning, and generate clear justifications. The system integrates large language models for natural language understanding and a knowledge graph for structured dietary information. By making decisions more transparent, our approach strengthens trust and improves human-robot interaction in healthcare. We validated this by measuring the computational efficiency of our architecture and conducting a small user study, which assessed the reliability of inner speech in explaining the robot's behavior. 

---
# A Comparative Study of Human Activity Recognition: Motion, Tactile, and multi-modal Approaches 

**Authors**: Valerio Belcamino, Nhat Minh Dinh Le, Quan Khanh Luu, Alessandro Carfì, Van Anh Ho, Fulvio Mastrogiovanni  

**Link**: [PDF](https://arxiv.org/pdf/2505.08657)  

**Abstract**: Human activity recognition (HAR) is essential for effective Human-Robot Collaboration (HRC), enabling robots to interpret and respond to human actions. This study evaluates the ability of a vision-based tactile sensor to classify 15 activities, comparing its performance to an IMU-based data glove. Additionally, we propose a multi-modal framework combining tactile and motion data to leverage their complementary strengths. We examined three approaches: motion-based classification (MBC) using IMU data, tactile-based classification (TBC) with single or dual video streams, and multi-modal classification (MMC) integrating both. Offline validation on segmented datasets assessed each configuration's accuracy under controlled conditions, while online validation on continuous action sequences tested online performance. Results showed the multi-modal approach consistently outperformed single-modality methods, highlighting the potential of integrating tactile and motion sensing to enhance HAR systems for collaborative robotics. 

---
# MINIMALIST: switched-capacitor circuits for efficient in-memory computation of gated recurrent units 

**Authors**: Sebastian Billaudelle, Laura Kriener, Filippo Moro, Tristan Torchet, Melika Payvand  

**Link**: [PDF](https://arxiv.org/pdf/2505.08599)  

**Abstract**: Recurrent neural networks (RNNs) have been a long-standing candidate for processing of temporal sequence data, especially in memory-constrained systems that one may find in embedded edge computing environments. Recent advances in training paradigms have now inspired new generations of efficient RNNs. We introduce a streamlined and hardware-compatible architecture based on minimal gated recurrent units (GRUs), and an accompanying efficient mixed-signal hardware implementation of the model. The proposed design leverages switched-capacitor circuits not only for in-memory computation (IMC), but also for the gated state updates. The mixed-signal cores rely solely on commodity circuits consisting of metal capacitors, transmission gates, and a clocked comparator, thus greatly facilitating scaling and transfer to other technology nodes.
We benchmark the performance of our architecture on time series data, introducing all constraints required for a direct mapping to the hardware system. The direct compatibility is verified in mixed-signal simulations, reproducing data recorded from the software-only network model. 

---
# MESSI: A Multi-Elevation Semantic Segmentation Image Dataset of an Urban Environment 

**Authors**: Barak Pinkovich, Boaz Matalon, Ehud Rivlin, Hector Rotstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.08589)  

**Abstract**: This paper presents a Multi-Elevation Semantic Segmentation Image (MESSI) dataset comprising 2525 images taken by a drone flying over dense urban environments. MESSI is unique in two main features. First, it contains images from various altitudes, allowing us to investigate the effect of depth on semantic segmentation. Second, it includes images taken from several different urban regions (at different altitudes). This is important since the variety covers the visual richness captured by a drone's 3D flight, performing horizontal and vertical maneuvers. MESSI contains images annotated with location, orientation, and the camera's intrinsic parameters and can be used to train a deep neural network for semantic segmentation or other applications of interest (e.g., localization, navigation, and tracking). This paper describes the dataset and provides annotation details. It also explains how semantic segmentation was performed using several neural network models and shows several relevant statistics. MESSI will be published in the public domain to serve as an evaluation benchmark for semantic segmentation using images captured by a drone or similar vehicle flying over a dense urban environment. 

---
# Small but Significant: On the Promise of Small Language Models for Accessible AIED 

**Authors**: Yumou Wei, Paulo Carvalho, John Stamper  

**Link**: [PDF](https://arxiv.org/pdf/2505.08588)  

**Abstract**: GPT has become nearly synonymous with large language models (LLMs), an increasingly popular term in AIED proceedings. A simple keyword-based search reveals that 61% of the 76 long and short papers presented at AIED 2024 describe novel solutions using LLMs to address some of the long-standing challenges in education, and 43% specifically mention GPT. Although LLMs pioneered by GPT create exciting opportunities to strengthen the impact of AI on education, we argue that the field's predominant focus on GPT and other resource-intensive LLMs (with more than 10B parameters) risks neglecting the potential impact that small language models (SLMs) can make in providing resource-constrained institutions with equitable and affordable access to high-quality AI tools. Supported by positive results on knowledge component (KC) discovery, a critical challenge in AIED, we demonstrate that SLMs such as Phi-2 can produce an effective solution without elaborate prompting strategies. Hence, we call for more attention to developing SLM-based AIED approaches. 

---
# DFA-CON: A Contrastive Learning Approach for Detecting Copyright Infringement in DeepFake Art 

**Authors**: Haroon Wahab, Hassan Ugail, Irfan Mehmood  

**Link**: [PDF](https://arxiv.org/pdf/2505.08552)  

**Abstract**: Recent proliferation of generative AI tools for visual content creation-particularly in the context of visual artworks-has raised serious concerns about copyright infringement and forgery. The large-scale datasets used to train these models often contain a mixture of copyrighted and non-copyrighted artworks. Given the tendency of generative models to memorize training patterns, they are susceptible to varying degrees of copyright violation. Building on the recently proposed DeepfakeArt Challenge benchmark, this work introduces DFA-CON, a contrastive learning framework designed to detect copyright-infringing or forged AI-generated art. DFA-CON learns a discriminative representation space, posing affinity among original artworks and their forged counterparts within a contrastive learning framework. The model is trained across multiple attack types, including inpainting, style transfer, adversarial perturbation, and cutmix. Evaluation results demonstrate robust detection performance across most attack types, outperforming recent pretrained foundation models. Code and model checkpoints will be released publicly upon acceptance. 

---
# From Seeing to Doing: Bridging Reasoning and Decision for Robotic Manipulation 

**Authors**: Yifu Yuan, Haiqin Cui, Yibin Chen, Zibin Dong, Fei Ni, Longxin Kou, Jinyi Liu, Pengyi Li, Yan Zheng, Jianye Hao  

**Link**: [PDF](https://arxiv.org/pdf/2505.08548)  

**Abstract**: Achieving generalization in robotic manipulation remains a critical challenge, particularly for unseen scenarios and novel tasks. Current Vision-Language-Action (VLA) models, while building on top of general Vision-Language Models (VLMs), still fall short of achieving robust zero-shot performance due to the scarcity and heterogeneity prevalent in embodied datasets. To address these limitations, we propose FSD (From Seeing to Doing), a novel vision-language model that generates intermediate representations through spatial relationship reasoning, providing fine-grained guidance for robotic manipulation. Our approach combines a hierarchical data pipeline for training with a self-consistency mechanism that aligns spatial coordinates with visual signals. Through extensive experiments, we comprehensively validated FSD's capabilities in both "seeing" and "doing," achieving outstanding performance across 8 benchmarks for general spatial reasoning and embodied reference abilities, as well as on our proposed more challenging benchmark VABench. We also verified zero-shot capabilities in robot manipulation, demonstrating significant performance improvements over baseline methods in both SimplerEnv and real robot settings. Experimental results show that FSD achieves 54.1% success rate in SimplerEnv and 72% success rate across 8 real-world tasks, outperforming the strongest baseline by 30%. 

---
# The Truth Becomes Clearer Through Debate! Multi-Agent Systems with Large Language Models Unmask Fake News 

**Authors**: Yuhan Liu, Yuxuan Liu, Xiaoqing Zhang, Xiuying Chen, Rui Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08532)  

**Abstract**: In today's digital environment, the rapid propagation of fake news via social networks poses significant social challenges. Most existing detection methods either employ traditional classification models, which suffer from low interpretability and limited generalization capabilities, or craft specific prompts for large language models (LLMs) to produce explanations and results directly, failing to leverage LLMs' reasoning abilities fully. Inspired by the saying that "truth becomes clearer through debate," our study introduces a novel multi-agent system with LLMs named TruEDebate (TED) to enhance the interpretability and effectiveness of fake news detection. TED employs a rigorous debate process inspired by formal debate settings. Central to our approach are two innovative components: the DebateFlow Agents and the InsightFlow Agents. The DebateFlow Agents organize agents into two teams, where one supports and the other challenges the truth of the news. These agents engage in opening statements, cross-examination, rebuttal, and closing statements, simulating a rigorous debate process akin to human discourse analysis, allowing for a thorough evaluation of news content. Concurrently, the InsightFlow Agents consist of two specialized sub-agents: the Synthesis Agent and the Analysis Agent. The Synthesis Agent summarizes the debates and provides an overarching viewpoint, ensuring a coherent and comprehensive evaluation. The Analysis Agent, which includes a role-aware encoder and a debate graph, integrates role embeddings and models the interactions between debate roles and arguments using an attention mechanism, providing the final judgment. 

---
# ExEBench: Benchmarking Foundation Models on Extreme Earth Events 

**Authors**: Shan Zhao, Zhitong Xiong, Jie Zhao, Xiao Xiang Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08529)  

**Abstract**: Our planet is facing increasingly frequent extreme events, which pose major risks to human lives and ecosystems. Recent advances in machine learning (ML), especially with foundation models (FMs) trained on extensive datasets, excel in extracting features and show promise in disaster management. Nevertheless, these models often inherit biases from training data, challenging their performance over extreme values. To explore the reliability of FM in the context of extreme events, we introduce \textbf{ExE}Bench (\textbf{Ex}treme \textbf{E}arth Benchmark), a collection of seven extreme event categories across floods, wildfires, storms, tropical cyclones, extreme precipitation, heatwaves, and cold waves. The dataset features global coverage, varying data volumes, and diverse data sources with different spatial, temporal, and spectral characteristics. To broaden the real-world impact of FMs, we include multiple challenging ML tasks that are closely aligned with operational needs in extreme events detection, monitoring, and forecasting. ExEBench aims to (1) assess FM generalizability across diverse, high-impact tasks and domains, (2) promote the development of novel ML methods that benefit disaster management, and (3) offer a platform for analyzing the interactions and cascading effects of extreme events to advance our understanding of Earth system, especially under the climate change expected in the decades to come. The dataset and code are public this https URL. 

---
# GradMix: Gradient-based Selective Mixup for Robust Data Augmentation in Class-Incremental Learning 

**Authors**: Minsu Kim, Seong-Hyeon Hwang, Steven Euijong Whang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08528)  

**Abstract**: In the context of continual learning, acquiring new knowledge while maintaining previous knowledge presents a significant challenge. Existing methods often use experience replay techniques that store a small portion of previous task data for training. In experience replay approaches, data augmentation has emerged as a promising strategy to further improve the model performance by mixing limited previous task data with sufficient current task data. However, we theoretically and empirically analyze that training with mixed samples from random sample pairs may harm the knowledge of previous tasks and cause greater catastrophic forgetting. We then propose GradMix, a robust data augmentation method specifically designed for mitigating catastrophic forgetting in class-incremental learning. GradMix performs gradient-based selective mixup using a class-based criterion that mixes only samples from helpful class pairs and not from detrimental class pairs for reducing catastrophic forgetting. Our experiments on various real datasets show that GradMix outperforms data augmentation baselines in accuracy by minimizing the forgetting of previous knowledge. 

---
# Learning Advanced Self-Attention for Linear Transformers in the Singular Value Domain 

**Authors**: Hyowon Wi, Jeongwhan Choi, Noseong Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.08516)  

**Abstract**: Transformers have demonstrated remarkable performance across diverse domains. The key component of Transformers is self-attention, which learns the relationship between any two tokens in the input sequence. Recent studies have revealed that the self-attention can be understood as a normalized adjacency matrix of a graph. Notably, from the perspective of graph signal processing (GSP), the self-attention can be equivalently defined as a simple graph filter, applying GSP using the value vector as the signal. However, the self-attention is a graph filter defined with only the first order of the polynomial matrix, and acts as a low-pass filter preventing the effective leverage of various frequency information. Consequently, existing self-attention mechanisms are designed in a rather simplified manner. Therefore, we propose a novel method, called \underline{\textbf{A}}ttentive \underline{\textbf{G}}raph \underline{\textbf{F}}ilter (AGF), interpreting the self-attention as learning the graph filter in the singular value domain from the perspective of graph signal processing for directed graphs with the linear complexity w.r.t. the input length $n$, i.e., $\mathcal{O}(nd^2)$. In our experiments, we demonstrate that AGF achieves state-of-the-art performance on various tasks, including Long Range Arena benchmark and time series classification. 

---
# LCES: Zero-shot Automated Essay Scoring via Pairwise Comparisons Using Large Language Models 

**Authors**: Takumi Shibata, Yuichi Miyamura  

**Link**: [PDF](https://arxiv.org/pdf/2505.08498)  

**Abstract**: Recent advances in large language models (LLMs) have enabled zero-shot automated essay scoring (AES), providing a promising way to reduce the cost and effort of essay scoring in comparison with manual grading. However, most existing zero-shot approaches rely on LLMs to directly generate absolute scores, which often diverge from human evaluations owing to model biases and inconsistent scoring. To address these limitations, we propose LLM-based Comparative Essay Scoring (LCES), a method that formulates AES as a pairwise comparison task. Specifically, we instruct LLMs to judge which of two essays is better, collect many such comparisons, and convert them into continuous scores. Considering that the number of possible comparisons grows quadratically with the number of essays, we improve scalability by employing RankNet to efficiently transform LLM preferences into scalar scores. Experiments using AES benchmark datasets show that LCES outperforms conventional zero-shot methods in accuracy while maintaining computational efficiency. Moreover, LCES is robust across different LLM backbones, highlighting its applicability to real-world zero-shot AES. 

---
# An adaptive sampling algorithm for data-generation to build a data-manifold for physical problem surrogate modeling 

**Authors**: Chetra Mang, Axel TahmasebiMoradi, David Danan, Mouadh Yagoubi  

**Link**: [PDF](https://arxiv.org/pdf/2505.08487)  

**Abstract**: Physical models classically involved Partial Differential equations (PDE) and depending of their underlying complexity and the level of accuracy required, and known to be computationally expensive to numerically solve them. Thus, an idea would be to create a surrogate model relying on data generated by such solver. However, training such a model on an imbalanced data have been shown to be a very difficult task. Indeed, if the distribution of input leads to a poor response manifold representation, the model may not learn well and consequently, it may not predict the outcome with acceptable accuracy. In this work, we present an Adaptive Sampling Algorithm for Data Generation (ASADG) involving a physical model. As the initial input data may not accurately represent the response manifold in higher dimension, this algorithm iteratively adds input data into it. At each step the barycenter of each simplicial complex, that the manifold is discretized into, is added as new input data, if a certain threshold is satisfied. We demonstrate the efficiency of the data sampling algorithm in comparison with LHS method for generating more representative input data. To do so, we focus on the construction of a harmonic transport problem metamodel by generating data through a classical solver. By using such algorithm, it is possible to generate the same number of input data as LHS while providing a better representation of the response manifold. 

---
# Distributed Quantum Neural Networks on Distributed Photonic Quantum Computing 

**Authors**: Kuan-Cheng Chen, Chen-Yu Liu, Yu Shang, Felix Burt, Kin K. Leung  

**Link**: [PDF](https://arxiv.org/pdf/2505.08474)  

**Abstract**: We introduce a distributed quantum-classical framework that synergizes photonic quantum neural networks (QNNs) with matrix-product-state (MPS) mapping to achieve parameter-efficient training of classical neural networks. By leveraging universal linear-optical decompositions of $M$-mode interferometers and photon-counting measurement statistics, our architecture generates neural parameters through a hybrid quantum-classical workflow: photonic QNNs with $M(M+1)/2$ trainable parameters produce high-dimensional probability distributions that are mapped to classical network weights via an MPS model with bond dimension $\chi$. Empirical validation on MNIST classification demonstrates that photonic QT achieves an accuracy of $95.50\% \pm 0.84\%$ using 3,292 parameters ($\chi = 10$), compared to $96.89\% \pm 0.31\%$ for classical baselines with 6,690 parameters. Moreover, a ten-fold compression ratio is achieved at $\chi = 4$, with a relative accuracy loss of less than $3\%$. The framework outperforms classical compression techniques (weight sharing/pruning) by 6--12\% absolute accuracy while eliminating quantum hardware requirements during inference through classical deployment of compressed parameters. Simulations incorporating realistic photonic noise demonstrate the framework's robustness to near-term hardware imperfections. Ablation studies confirm quantum necessity: replacing photonic QNNs with random inputs collapses accuracy to chance level ($10.0\% \pm 0.5\%$). Photonic quantum computing's room-temperature operation, inherent scalability through spatial-mode multiplexing, and HPC-integrated architecture establish a practical pathway for distributed quantum machine learning, combining the expressivity of photonic Hilbert spaces with the deployability of classical neural networks. 

---
# RepCali: High Efficient Fine-tuning Via Representation Calibration in Latent Space for Pre-trained Language Models 

**Authors**: Fujun Zhang, XiangDong Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.08463)  

**Abstract**: Fine-tuning pre-trained language models (PLMs) has become a dominant paradigm in applying PLMs to downstream tasks. However, with limited fine-tuning, PLMs still struggle with the discrepancies between the representation obtained from the PLMs' encoder and the optimal input to the PLMs' decoder. This paper tackles this challenge by learning to calibrate the representation of PLMs in the latent space. In the proposed representation calibration method (RepCali), we integrate a specific calibration block to the latent space after the encoder and use the calibrated output as the decoder input. The merits of the proposed RepCali include its universality to all PLMs with encoder-decoder architectures, its plug-and-play nature, and ease of implementation. Extensive experiments on 25 PLM-based models across 8 tasks (including both English and Chinese datasets) demonstrate that the proposed RepCali offers desirable enhancements to PLMs (including LLMs) and significantly improves the performance of downstream tasks. Comparison experiments across 4 benchmark tasks indicate that RepCali is superior to the representative fine-tuning baselines. 

---
# Optimizing Retrieval-Augmented Generation: Analysis of Hyperparameter Impact on Performance and Efficiency 

**Authors**: Adel Ammar, Anis Koubaa, Omer Nacar, Wadii Boulila  

**Link**: [PDF](https://arxiv.org/pdf/2505.08445)  

**Abstract**: Large language models achieve high task performance yet often hallucinate or rely on outdated knowledge. Retrieval-augmented generation (RAG) addresses these gaps by coupling generation with external search. We analyse how hyperparameters influence speed and quality in RAG systems, covering Chroma and Faiss vector stores, chunking policies, cross-encoder re-ranking, and temperature, and we evaluate six metrics: faithfulness, answer correctness, answer relevancy, context precision, context recall, and answer similarity. Chroma processes queries 13% faster, whereas Faiss yields higher retrieval precision, revealing a clear speed-accuracy trade-off. Naive fixed-length chunking with small windows and minimal overlap outperforms semantic segmentation while remaining the quickest option. Re-ranking provides modest gains in retrieval quality yet increases runtime by roughly a factor of 5, so its usefulness depends on latency constraints. These results help practitioners balance computational cost and accuracy when tuning RAG systems for transparent, up-to-date responses. Finally, we re-evaluate the top configurations with a corrective RAG workflow and show that their advantages persist when the model can iteratively request additional evidence. We obtain a near-perfect context precision (99%), which demonstrates that RAG systems can achieve extremely high retrieval accuracy with the right combination of hyperparameters, with significant implications for applications where retrieval quality directly impacts downstream task performance, such as clinical decision support in healthcare. 

---
# A Survey of 3D Reconstruction with Event Cameras: From Event-based Geometry to Neural 3D Rendering 

**Authors**: Chuanzhi Xu, Haoxian Zhou, Langyi Chen, Haodong Chen, Ying Zhou, Vera Chung, Qiang Qu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08438)  

**Abstract**: Event cameras have emerged as promising sensors for 3D reconstruction due to their ability to capture per-pixel brightness changes asynchronously. Unlike conventional frame-based cameras, they produce sparse and temporally rich data streams, which enable more accurate 3D reconstruction and open up the possibility of performing reconstruction in extreme environments such as high-speed motion, low light, or high dynamic range scenes. In this survey, we provide the first comprehensive review focused exclusively on 3D reconstruction using event cameras. The survey categorises existing works into three major types based on input modality - stereo, monocular, and multimodal systems, and further classifies them by reconstruction approach, including geometry-based, deep learning-based, and recent neural rendering techniques such as Neural Radiance Fields and 3D Gaussian Splatting. Methods with a similar research focus were organised chronologically into the most subdivided groups. We also summarise public datasets relevant to event-based 3D reconstruction. Finally, we highlight current research limitations in data availability, evaluation, representation, and dynamic scene handling, and outline promising future research directions. This survey aims to serve as a comprehensive reference and a roadmap for future developments in event-driven 3D reconstruction. 

---
# Hakim: Farsi Text Embedding Model 

**Authors**: Mehran Sarmadi, Morteza Alikhani, Erfan Zinvandi, Zahra Pourbahman  

**Link**: [PDF](https://arxiv.org/pdf/2505.08435)  

**Abstract**: Recent advancements in text embedding have significantly improved natural language understanding across many languages, yet Persian remains notably underrepresented in large-scale embedding research. In this paper, we present Hakim, a novel state-of-the-art Persian text embedding model that achieves a 8.5% performance improvement over existing approaches on the FaMTEB benchmark, outperforming all previously developed Persian language models. As part of this work, we introduce three new datasets - Corpesia, Pairsia-sup, and Pairsia-unsup - to support supervised and unsupervised training scenarios. Additionally, Hakim is designed for applications in chatbots and retrieval-augmented generation (RAG) systems, particularly addressing retrieval tasks that require incorporating message history within these systems. We also propose a new baseline model built on the BERT architecture. Our language model consistently achieves higher accuracy across various Persian NLP tasks, while the RetroMAE-based model proves particularly effective for textual information retrieval applications. Together, these contributions establish a new foundation for advancing Persian language understanding. 

---
# ConDiSim: Conditional Diffusion Models for Simulation Based Inference 

**Authors**: Mayank Nautiyal, Andreas Hellander, Prashant Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.08403)  

**Abstract**: We present a conditional diffusion model - ConDiSim, for simulation-based inference of complex systems with intractable likelihoods. ConDiSim leverages denoising diffusion probabilistic models to approximate posterior distributions, consisting of a forward process that adds Gaussian noise to parameters, and a reverse process learning to denoise, conditioned on observed data. This approach effectively captures complex dependencies and multi-modalities within posteriors. ConDiSim is evaluated across ten benchmark problems and two real-world test problems, where it demonstrates effective posterior approximation accuracy while maintaining computational efficiency and stability in model training. ConDiSim offers a robust and extensible framework for simulation-based inference, particularly suitable for parameter inference workflows requiring fast inference methods. 

---
# Accelerating Chain-of-Thought Reasoning: When Goal-Gradient Importance Meets Dynamic Skipping 

**Authors**: Ren Zhuang, Ben Wang, Shuifa Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.08392)  

**Abstract**: Large Language Models leverage Chain-of-Thought (CoT) prompting for complex tasks, but their reasoning traces are often excessively verbose and inefficient, leading to significant computational costs and latency. Current CoT compression techniques typically rely on generic importance metrics and static compression rates, which may inadvertently remove functionally critical tokens or fail to adapt to varying reasoning complexity. To overcome these limitations, we propose Adaptive GoGI-Skip, a novel framework learning dynamic CoT compression via supervised fine-tuning. This approach introduces two synergistic innovations: (1) Goal-Gradient Importance (GoGI), a novel metric accurately identifying functionally relevant tokens by measuring the gradient influence of their intermediate representations on the final answer loss, and (2) Adaptive Dynamic Skipping (ADS), a mechanism dynamically regulating the compression rate based on runtime model uncertainty while ensuring local coherence through an adaptive N-token constraint. To our knowledge, this is the first work unifying a goal-oriented, gradient-based importance metric with dynamic, uncertainty-aware skipping for CoT compression. Trained on compressed MATH data, Adaptive GoGI-Skip demonstrates strong cross-domain generalization across diverse reasoning benchmarks including AIME, GPQA, and GSM8K. It achieves substantial efficiency gains - reducing CoT token counts by over 45% on average and delivering 1.6-2.0 times inference speedups - while maintaining high reasoning accuracy. Notably, it significantly outperforms existing baselines by preserving accuracy even at high effective compression rates, advancing the state of the art in the CoT reasoning efficiency-accuracy trade-off. 

---
# Adaptive Diffusion Policy Optimization for Robotic Manipulation 

**Authors**: Huiyun Jiang, Zhuang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08376)  

**Abstract**: Recent studies have shown the great potential of diffusion models in improving reinforcement learning (RL) by modeling complex policies, expressing a high degree of multi-modality, and efficiently handling high-dimensional continuous control tasks. However, there is currently limited research on how to optimize diffusion-based polices (e.g., Diffusion Policy) fast and stably. In this paper, we propose an Adam-based Diffusion Policy Optimization (ADPO), a fast algorithmic framework containing best practices for fine-tuning diffusion-based polices in robotic control tasks using the adaptive gradient descent method in RL. Adaptive gradient method is less studied in training RL, let alone diffusion-based policies. We confirm that ADPO outperforms other diffusion-based RL methods in terms of overall effectiveness for fine-tuning on standard robotic tasks. Concretely, we conduct extensive experiments on standard robotic control tasks to test ADPO, where, particularly, six popular diffusion-based RL methods are provided as benchmark methods. Experimental results show that ADPO acquires better or comparable performance than the baseline methods. Finally, we systematically analyze the sensitivity of multiple hyperparameters in standard robotics tasks, providing guidance for subsequent practical applications. Our video demonstrations are released in this https URL. 

---
# Non-contact Vital Signs Detection in Dynamic Environments 

**Authors**: Shuai Sun, Chong-Xi Liang, Chengwei Ye, Huanzhen Zhang, Kangsheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08366)  

**Abstract**: Accurate phase demodulation is critical for vital sign detection using millimeter-wave radar. However, in complex environments, time-varying DC offsets and phase imbalances can severely degrade demodulation performance. To address this, we propose a novel DC offset calibration method alongside a Hilbert and Differential Cross-Multiply (HADCM) demodulation algorithm. The approach estimates time-varying DC offsets from neighboring signal peaks and valleys, then employs both differential forms and Hilbert transforms of the I/Q channel signals to extract vital sign information. Simulation and experimental results demonstrate that the proposed method maintains robust performance under low signal-to-noise ratios. Compared to existing demodulation techniques, it offers more accurate signal recovery in challenging scenarios and effectively suppresses noise interference. 

---
# STORYANCHORS: Generating Consistent Multi-Scene Story Frames for Long-Form Narratives 

**Authors**: Bo Wang, Haoyang Huang, Zhiyin Lu, Fengyuan Liu, Guoqing Ma, Jianlong Yuan, Yuan Zhang, Nan Duan  

**Link**: [PDF](https://arxiv.org/pdf/2505.08350)  

**Abstract**: This paper introduces StoryAnchors, a unified framework for generating high-quality, multi-scene story frames with strong temporal consistency. The framework employs a bidirectional story generator that integrates both past and future contexts to ensure temporal consistency, character continuity, and smooth scene transitions throughout the narrative. Specific conditions are introduced to distinguish story frame generation from standard video synthesis, facilitating greater scene diversity and enhancing narrative richness. To further improve generation quality, StoryAnchors integrates Multi-Event Story Frame Labeling and Progressive Story Frame Training, enabling the model to capture both overarching narrative flow and event-level dynamics. This approach supports the creation of editable and expandable story frames, allowing for manual modifications and the generation of longer, more complex sequences. Extensive experiments show that StoryAnchors outperforms existing open-source models in key areas such as consistency, narrative coherence, and scene diversity. Its performance in narrative consistency and story richness is also on par with GPT-4o. Ultimately, StoryAnchors pushes the boundaries of story-driven frame generation, offering a scalable, flexible, and highly editable foundation for future research. 

---
# FAD: Frequency Adaptation and Diversion for Cross-domain Few-shot Learning 

**Authors**: Ruixiao Shi, Fu Feng, Yucheng Xie, Jing Wang, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2505.08349)  

**Abstract**: Cross-domain few-shot learning (CD-FSL) requires models to generalize from limited labeled samples under significant distribution shifts. While recent methods enhance adaptability through lightweight task-specific modules, they operate solely in the spatial domain and overlook frequency-specific variations that are often critical for robust transfer. We observe that spatially similar images across domains can differ substantially in their spectral representations, with low and high frequencies capturing complementary semantic information at coarse and fine levels. This indicates that uniform spatial adaptation may overlook these spectral distinctions, thus constraining generalization. To address this, we introduce Frequency Adaptation and Diversion (FAD), a frequency-aware framework that explicitly models and modulates spectral components. At its core is the Frequency Diversion Adapter, which transforms intermediate features into the frequency domain using the discrete Fourier transform (DFT), partitions them into low, mid, and high-frequency bands via radial masks, and reconstructs each band using inverse DFT (IDFT). Each frequency band is then adapted using a dedicated convolutional branch with a kernel size tailored to its spectral scale, enabling targeted and disentangled adaptation across frequencies. Extensive experiments on the Meta-Dataset benchmark demonstrate that FAD consistently outperforms state-of-the-art methods on both seen and unseen domains, validating the utility of frequency-domain representations and band-wise adaptation for improving generalization in CD-FSL. 

---
# SHAP-based Explanations are Sensitive to Feature Representation 

**Authors**: Hyunseung Hwang, Andrew Bell, Joao Fonseca, Venetia Pliatsika, Julia Stoyanovich, Steven Euijong Whang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08345)  

**Abstract**: Local feature-based explanations are a key component of the XAI toolkit. These explanations compute feature importance values relative to an ``interpretable'' feature representation. In tabular data, feature values themselves are often considered interpretable. This paper examines the impact of data engineering choices on local feature-based explanations. We demonstrate that simple, common data engineering techniques, such as representing age with a histogram or encoding race in a specific way, can manipulate feature importance as determined by popular methods like SHAP. Notably, the sensitivity of explanations to feature representation can be exploited by adversaries to obscure issues like discrimination. While the intuition behind these results is straightforward, their systematic exploration has been lacking. Previous work has focused on adversarial attacks on feature-based explainers by biasing data or manipulating models. To the best of our knowledge, this is the first study demonstrating that explainers can be misled by standard, seemingly innocuous data engineering techniques. 

---
# A computer vision-based model for occupancy detection using low-resolution thermal images 

**Authors**: Xue Cui, Vincent Gbouna Zakka, Minhyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.08336)  

**Abstract**: Occupancy plays an essential role in influencing the energy consumption and operation of heating, ventilation, and air conditioning (HVAC) systems. Traditional HVAC typically operate on fixed schedules without considering occupancy. Advanced occupant-centric control (OCC) adopted occupancy status in regulating HVAC operations. RGB images combined with computer vision (CV) techniques are widely used for occupancy detection, however, the detailed facial and body features they capture raise significant privacy concerns. Low-resolution thermal images offer a non-invasive solution that mitigates privacy issues. The study developed an occupancy detection model utilizing low-resolution thermal images and CV techniques, where transfer learning was applied to fine-tune the You Only Look Once version 5 (YOLOv5) model. The developed model ultimately achieved satisfactory performance, with precision, recall, mAP50, and mAP50 values approaching 1.000. The contributions of this model lie not only in mitigating privacy concerns but also in reducing computing resource demands. 

---
# Low-Complexity Inference in Continual Learning via Compressed Knowledge Transfer 

**Authors**: Zhenrong Liu, Janne M. J. Huttunen, Mikko Honkala  

**Link**: [PDF](https://arxiv.org/pdf/2505.08327)  

**Abstract**: Continual learning (CL) aims to train models that can learn a sequence of tasks without forgetting previously acquired knowledge. A core challenge in CL is balancing stability -- preserving performance on old tasks -- and plasticity -- adapting to new ones. Recently, large pre-trained models have been widely adopted in CL for their ability to support both, offering strong generalization for new tasks and resilience against forgetting. However, their high computational cost at inference time limits their practicality in real-world applications, especially those requiring low latency or energy efficiency. To address this issue, we explore model compression techniques, including pruning and knowledge distillation (KD), and propose two efficient frameworks tailored for class-incremental learning (CIL), a challenging CL setting where task identities are unavailable during inference. The pruning-based framework includes pre- and post-pruning strategies that apply compression at different training stages. The KD-based framework adopts a teacher-student architecture, where a large pre-trained teacher transfers downstream-relevant knowledge to a compact student. Extensive experiments on multiple CIL benchmarks demonstrate that the proposed frameworks achieve a better trade-off between accuracy and inference complexity, consistently outperforming strong baselines. We further analyze the trade-offs between the two frameworks in terms of accuracy and efficiency, offering insights into their use across different scenarios. 

---
# FedRS-Bench: Realistic Federated Learning Datasets and Benchmarks in Remote Sensing 

**Authors**: Haodong Zhao, Peng Peng, Chiyu Chen, Linqing Huang, Gongshen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08325)  

**Abstract**: Remote sensing (RS) images are usually produced at an unprecedented scale, yet they are geographically and institutionally distributed, making centralized model training challenging due to data-sharing restrictions and privacy concerns. Federated learning (FL) offers a solution by enabling collaborative model training across decentralized RS data sources without exposing raw data. However, there lacks a realistic federated dataset and benchmark in RS. Prior works typically rely on manually partitioned single dataset, which fail to capture the heterogeneity and scale of real-world RS data, and often use inconsistent experimental setups, hindering fair comparison. To address this gap, we propose a realistic federated RS dataset, termed FedRS. FedRS consists of eight datasets that cover various sensors and resolutions and builds 135 clients, which is representative of realistic operational scenarios. Data for each client come from the same source, exhibiting authentic federated properties such as skewed label distributions, imbalanced client data volumes, and domain heterogeneity across clients. These characteristics reflect practical challenges in federated RS and support evaluation of FL methods at scale. Based on FedRS, we implement 10 baseline FL algorithms and evaluation metrics to construct the comprehensive FedRS-Bench. The experimental results demonstrate that FL can consistently improve model performance over training on isolated data silos, while revealing performance trade-offs of different methods under varying client heterogeneity and availability conditions. We hope FedRS-Bench will accelerate research on large-scale, realistic FL in RS by providing a standardized, rich testbed and facilitating fair comparisons across future works. The source codes and dataset are available at this https URL. 

---
# Reciprocity as the Foundational Substrate of Society: How Reciprocal Dynamics Scale into Social Systems 

**Authors**: Egil Diau  

**Link**: [PDF](https://arxiv.org/pdf/2505.08319)  

**Abstract**: A major bottleneck in multi-agent AI is the lack of simulateable models for the bottom-up emergence of social structure under realistic behavioral constraints. Similarly, many foundational theories in economics and sociology including the concepts of "institutions" and "norms" tend to describe social structures post hoc, often relying on implicit assumptions of shared culture, morality, or symbolic agreement. These concepts are often treated as primitives rather than reconstructed from agent-level behavior, leaving both their origins and operational definitions under-specified. To address this, we propose a three-stage bottom-up framework: Reciprocal Dynamics, capturing individual-level reciprocal exchanges; Norm Stabilization, the consolidation of shared expectations; and Institutional Construction, the externalization of stable patterns into scalable structures. By grounding social emergence in agent-level reciprocity, our framework enables the systematic exploration of how moral, cultural, and institutional structures emerge from cognitively minimal interactions. 

---
# A Practical Introduction to Deep Reinforcement Learning 

**Authors**: Yinghan Sun, Hongxi Wang, Hua Chen, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08295)  

**Abstract**: Deep reinforcement learning (DRL) has emerged as a powerful framework for solving sequential decision-making problems, achieving remarkable success in a wide range of applications, including game AI, autonomous driving, biomedicine, and large language models. However, the diversity of algorithms and the complexity of theoretical foundations often pose significant challenges for beginners seeking to enter the field. This tutorial aims to provide a concise, intuitive, and practical introduction to DRL, with a particular focus on the Proximal Policy Optimization (PPO) algorithm, which is one of the most widely used and effective DRL methods. To facilitate learning, we organize all algorithms under the Generalized Policy Iteration (GPI) framework, offering readers a unified and systematic perspective. Instead of lengthy theoretical proofs, we emphasize intuitive explanations, illustrative examples, and practical engineering techniques. This work serves as an efficient and accessible guide, helping readers rapidly progress from basic concepts to the implementation of advanced DRL algorithms. 

---
# M3G: Multi-Granular Gesture Generator for Audio-Driven Full-Body Human Motion Synthesis 

**Authors**: Zhizhuo Yin, Yuk Hang Tsui, Pan Hui  

**Link**: [PDF](https://arxiv.org/pdf/2505.08293)  

**Abstract**: Generating full-body human gestures encompassing face, body, hands, and global movements from audio is a valuable yet challenging task in virtual avatar creation. Previous systems focused on tokenizing the human gestures framewisely and predicting the tokens of each frame from the input audio. However, one observation is that the number of frames required for a complete expressive human gesture, defined as granularity, varies among different human gesture patterns. Existing systems fail to model these gesture patterns due to the fixed granularity of their gesture tokens. To solve this problem, we propose a novel framework named Multi-Granular Gesture Generator (M3G) for audio-driven holistic gesture generation. In M3G, we propose a novel Multi-Granular VQ-VAE (MGVQ-VAE) to tokenize motion patterns and reconstruct motion sequences from different temporal granularities. Subsequently, we proposed a multi-granular token predictor that extracts multi-granular information from audio and predicts the corresponding motion tokens. Then M3G reconstructs the human gestures from the predicted tokens using the MGVQ-VAE. Both objective and subjective experiments demonstrate that our proposed M3G framework outperforms the state-of-the-art methods in terms of generating natural and expressive full-body human gestures. 

---
# Open the Eyes of MPNN: Vision Enhances MPNN in Link Prediction 

**Authors**: Yanbin Wei, Xuehao Wang, Zhan Zhuang, Yang Chen, Shuhao Chen, Yulong Zhang, Yu Zhang, James Kwok  

**Link**: [PDF](https://arxiv.org/pdf/2505.08266)  

**Abstract**: Message-passing graph neural networks (MPNNs) and structural features (SFs) are cornerstones for the link prediction task. However, as a common and intuitive mode of understanding, the potential of visual perception has been overlooked in the MPNN community. For the first time, we equip MPNNs with vision structural awareness by proposing an effective framework called Graph Vision Network (GVN), along with a more efficient variant (E-GVN). Extensive empirical results demonstrate that with the proposed frameworks, GVN consistently benefits from the vision enhancement across seven link prediction datasets, including challenging large-scale graphs. Such improvements are compatible with existing state-of-the-art (SOTA) methods and GVNs achieve new SOTA results, thereby underscoring a promising novel direction for link prediction. 

---
# LLM Enhancers for GNNs: An Analysis from the Perspective of Causal Mechanism Identification 

**Authors**: Hang Gao, Wenxuan Huang, Fengge Wu, Junsuo Zhao, Changwen Zheng, Huaping Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08265)  

**Abstract**: The use of large language models (LLMs) as feature enhancers to optimize node representations, which are then used as inputs for graph neural networks (GNNs), has shown significant potential in graph representation learning. However, the fundamental properties of this approach remain underexplored. To address this issue, we propose conducting a more in-depth analysis of this issue based on the interchange intervention method. First, we construct a synthetic graph dataset with controllable causal relationships, enabling precise manipulation of semantic relationships and causal modeling to provide data for analysis. Using this dataset, we conduct interchange interventions to examine the deeper properties of LLM enhancers and GNNs, uncovering their underlying logic and internal mechanisms. Building on the analytical results, we design a plug-and-play optimization module to improve the information transfer between LLM enhancers and GNNs. Experiments across multiple datasets and models validate the proposed module. 

---
# Automatic Curriculum Learning for Driving Scenarios: Towards Robust and Efficient Reinforcement Learning 

**Authors**: Ahmed Abouelazm, Tim Weinstein, Tim Joseph, Philip Schörner, J. Marius Zöllner  

**Link**: [PDF](https://arxiv.org/pdf/2505.08264)  

**Abstract**: This paper addresses the challenges of training end-to-end autonomous driving agents using Reinforcement Learning (RL). RL agents are typically trained in a fixed set of scenarios and nominal behavior of surrounding road users in simulations, limiting their generalization and real-life deployment. While domain randomization offers a potential solution by randomly sampling driving scenarios, it frequently results in inefficient training and sub-optimal policies due to the high variance among training scenarios. To address these limitations, we propose an automatic curriculum learning framework that dynamically generates driving scenarios with adaptive complexity based on the agent's evolving capabilities. Unlike manually designed curricula that introduce expert bias and lack scalability, our framework incorporates a ``teacher'' that automatically generates and mutates driving scenarios based on their learning potential -- an agent-centric metric derived from the agent's current policy -- eliminating the need for expert design. The framework enhances training efficiency by excluding scenarios the agent has mastered or finds too challenging. We evaluate our framework in a reinforcement learning setting where the agent learns a driving policy from camera images. Comparative results against baseline methods, including fixed scenario training and domain randomization, demonstrate that our approach leads to enhanced generalization, achieving higher success rates: +9\% in low traffic density, +21\% in high traffic density, and faster convergence with fewer training steps. Our findings highlight the potential of ACL in improving the robustness and efficiency of RL-based autonomous driving agents. 

---
# Enhancing Cache-Augmented Generation (CAG) with Adaptive Contextual Compression for Scalable Knowledge Integration 

**Authors**: Rishabh Agrawal, Himanshu Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08261)  

**Abstract**: The rapid progress in large language models (LLMs) has paved the way for novel approaches in knowledge-intensive tasks. Among these, Cache-Augmented Generation (CAG) has emerged as a promising alternative to Retrieval-Augmented Generation (RAG). CAG minimizes retrieval latency and simplifies system design by preloading knowledge into the model's context. However, challenges persist in scaling CAG to accommodate large and dynamic knowledge bases effectively. This paper introduces Adaptive Contextual Compression (ACC), an innovative technique designed to dynamically compress and manage context inputs, enabling efficient utilization of the extended memory capabilities of modern LLMs. To further address the limitations of standalone CAG, we propose a Hybrid CAG-RAG Framework, which integrates selective retrieval to augment preloaded contexts in scenarios requiring additional information. Comprehensive evaluations on diverse datasets highlight the proposed methods' ability to enhance scalability, optimize efficiency, and improve multi-hop reasoning performance, offering practical solutions for real-world knowledge integration challenges. 

---
# Large Language Model Psychometrics: A Systematic Review of Evaluation, Validation, and Enhancement 

**Authors**: Haoran Ye, Jing Jin, Yuhang Xie, Xin Zhang, Guojie Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.08245)  

**Abstract**: The rapid advancement of large language models (LLMs) has outpaced traditional evaluation methodologies. It presents novel challenges, such as measuring human-like psychological constructs, navigating beyond static and task-specific benchmarks, and establishing human-centered evaluation. These challenges intersect with Psychometrics, the science of quantifying the intangible aspects of human psychology, such as personality, values, and intelligence. This survey introduces and synthesizes an emerging interdisciplinary field of LLM Psychometrics, which leverages psychometric instruments, theories, and principles to evaluate, understand, and enhance LLMs. We systematically explore the role of Psychometrics in shaping benchmarking principles, broadening evaluation scopes, refining methodologies, validating results, and advancing LLM capabilities. This paper integrates diverse perspectives to provide a structured framework for researchers across disciplines, enabling a more comprehensive understanding of this nascent field. Ultimately, we aim to provide actionable insights for developing future evaluation paradigms that align with human-level AI and promote the advancement of human-centered AI systems for societal benefit. A curated repository of LLM psychometric resources is available at this https URL. 

---
# Removing Watermarks with Partial Regeneration using Semantic Information 

**Authors**: Krti Tallam, John Kevin Cava, Caleb Geniesse, N. Benjamin Erichson, Michael W. Mahoney  

**Link**: [PDF](https://arxiv.org/pdf/2505.08234)  

**Abstract**: As AI-generated imagery becomes ubiquitous, invisible watermarks have emerged as a primary line of defense for copyright and provenance. The newest watermarking schemes embed semantic signals - content-aware patterns that are designed to survive common image manipulations - yet their true robustness against adaptive adversaries remains under-explored. We expose a previously unreported vulnerability and introduce SemanticRegen, a three-stage, label-free attack that erases state-of-the-art semantic and invisible watermarks while leaving an image's apparent meaning intact. Our pipeline (i) uses a vision-language model to obtain fine-grained captions, (ii) extracts foreground masks with zero-shot segmentation, and (iii) inpaints only the background via an LLM-guided diffusion model, thereby preserving salient objects and style cues. Evaluated on 1,000 prompts across four watermarking systems - TreeRing, StegaStamp, StableSig, and DWT/DCT - SemanticRegen is the only method to defeat the semantic TreeRing watermark (p = 0.10 > 0.05) and reduces bit-accuracy below 0.75 for the remaining schemes, all while maintaining high perceptual quality (masked SSIM = 0.94 +/- 0.01). We further introduce masked SSIM (mSSIM) to quantify fidelity within foreground regions, showing that our attack achieves up to 12 percent higher mSSIM than prior diffusion-based attackers. These results highlight an urgent gap between current watermark defenses and the capabilities of adaptive, semantics-aware adversaries, underscoring the need for watermarking algorithms that are resilient to content-preserving regenerative attacks. 

---
# Object detection in adverse weather conditions for autonomous vehicles using Instruct Pix2Pix 

**Authors**: Unai Gurbindo, Axel Brando, Jaume Abella, Caroline König  

**Link**: [PDF](https://arxiv.org/pdf/2505.08228)  

**Abstract**: Enhancing the robustness of object detection systems under adverse weather conditions is crucial for the advancement of autonomous driving technology. This study presents a novel approach leveraging the diffusion model Instruct Pix2Pix to develop prompting methodologies that generate realistic datasets with weather-based augmentations aiming to mitigate the impact of adverse weather on the perception capabilities of state-of-the-art object detection models, including Faster R-CNN and YOLOv10. Experiments were conducted in two environments, in the CARLA simulator where an initial evaluation of the proposed data augmentation was provided, and then on the real-world image data sets BDD100K and ACDC demonstrating the effectiveness of the approach in real environments.
The key contributions of this work are twofold: (1) identifying and quantifying the performance gap in object detection models under challenging weather conditions, and (2) demonstrating how tailored data augmentation strategies can significantly enhance the robustness of these models. This research establishes a solid foundation for improving the reliability of perception systems in demanding environmental scenarios, and provides a pathway for future advancements in autonomous driving. 

---
# Reinforcement Learning-based Fault-Tolerant Control for Quadrotor with Online Transformer Adaptation 

**Authors**: Dohyun Kim, Jayden Dongwoo Lee, Hyochoong Bang, Jungho Bae  

**Link**: [PDF](https://arxiv.org/pdf/2505.08223)  

**Abstract**: Multirotors play a significant role in diverse field robotics applications but remain highly susceptible to actuator failures, leading to rapid instability and compromised mission reliability. While various fault-tolerant control (FTC) strategies using reinforcement learning (RL) have been widely explored, most previous approaches require prior knowledge of the multirotor model or struggle to adapt to new configurations. To address these limitations, we propose a novel hybrid RL-based FTC framework integrated with a transformer-based online adaptation module. Our framework leverages a transformer architecture to infer latent representations in real time, enabling adaptation to previously unseen system models without retraining. We evaluate our method in a PyBullet simulation under loss-of-effectiveness actuator faults, achieving a 95% success rate and a positional root mean square error (RMSE) of 0.129 m, outperforming existing adaptation methods with 86% success and an RMSE of 0.153 m. Further evaluations on quadrotors with varying configurations confirm the robustness of our framework across untrained dynamics. These results demonstrate the potential of our framework to enhance the adaptability and reliability of multirotors, enabling efficient fault management in dynamic and uncertain environments. Website is available at this http URL 

---
# Scaling Multi Agent Reinforcement Learning for Underwater Acoustic Tracking via Autonomous Vehicles 

**Authors**: Matteo Gallici, Ivan Masmitja, Mario Martín  

**Link**: [PDF](https://arxiv.org/pdf/2505.08222)  

**Abstract**: Autonomous vehicles (AV) offer a cost-effective solution for scientific missions such as underwater tracking. Recently, reinforcement learning (RL) has emerged as a powerful method for controlling AVs in complex marine environments. However, scaling these techniques to a fleet--essential for multi-target tracking or targets with rapid, unpredictable motion--presents significant computational challenges. Multi-Agent Reinforcement Learning (MARL) is notoriously sample-inefficient, and while high-fidelity simulators like Gazebo's LRAUV provide 100x faster-than-real-time single-robot simulations, they offer no significant speedup for multi-vehicle scenarios, making MARL training impractical. To address these limitations, we propose an iterative distillation method that transfers high-fidelity simulations into a simplified, GPU-accelerated environment while preserving high-level dynamics. This approach achieves up to a 30,000x speedup over Gazebo through parallelization, enabling efficient training via end-to-end GPU acceleration. Additionally, we introduce a novel Transformer-based architecture (TransfMAPPO) that learns multi-agent policies invariant to the number of agents and targets, significantly improving sample efficiency. Following large-scale curriculum learning conducted entirely on GPU, we perform extensive evaluations in Gazebo, demonstrating that our method maintains tracking errors below 5 meters over extended durations, even in the presence of multiple fast-moving targets. This work bridges the gap between large-scale MARL training and high-fidelity deployment, providing a scalable framework for autonomous fleet control in real-world sea missions. 

---
# A Head to Predict and a Head to Question: Pre-trained Uncertainty Quantification Heads for Hallucination Detection in LLM Outputs 

**Authors**: Artem Shelmanov, Ekaterina Fadeeva, Akim Tsvigun, Ivan Tsvigun, Zhuohan Xie, Igor Kiselev, Nico Daheim, Caiqi Zhang, Artem Vazhentsev, Mrinmaya Sachan, Preslav Nakov, Timothy Baldwin  

**Link**: [PDF](https://arxiv.org/pdf/2505.08200)  

**Abstract**: Large Language Models (LLMs) have the tendency to hallucinate, i.e., to sporadically generate false or fabricated information. This presents a major challenge, as hallucinations often appear highly convincing and users generally lack the tools to detect them. Uncertainty quantification (UQ) provides a framework for assessing the reliability of model outputs, aiding in the identification of potential hallucinations. In this work, we introduce pre-trained UQ heads: supervised auxiliary modules for LLMs that substantially enhance their ability to capture uncertainty compared to unsupervised UQ methods. Their strong performance stems from the powerful Transformer architecture in their design and informative features derived from LLM attention maps. Experimental evaluation shows that these heads are highly robust and achieve state-of-the-art performance in claim-level hallucination detection across both in-domain and out-of-domain prompts. Moreover, these modules demonstrate strong generalization to languages they were not explicitly trained on. We pre-train a collection of UQ heads for popular LLM series, including Mistral, Llama, and Gemma 2. We publicly release both the code and the pre-trained heads. 

---
# Aitomia: Your Intelligent Assistant for AI-Driven Atomistic and Quantum Chemical Simulations 

**Authors**: Jinming Hu, Hassan Nawaz, Yuting Rui, Lijie Chi, Arif Ullah, Pavlo O. Dral  

**Link**: [PDF](https://arxiv.org/pdf/2505.08195)  

**Abstract**: We have developed Aitomia - a platform powered by AI to assist in performing AI-driven atomistic and quantum chemical (QC) simulations. This intelligent assistant platform is equipped with chatbots and AI agents to help experts and guide non-experts in setting up and running the atomistic simulations, monitoring their computation status, analyzing the simulation results, and summarizing them for the user in text and graphical forms. We achieve these goals by exploiting fine-tuned open-source large language models (LLMs), rule-based agents, and a retrieval-augmented generation (RAG) system. Aitomia leverages the versatility of our MLatom ecosystem for AI-enhanced computational chemistry. This intelligent assistant is going to be integrated into the Aitomistic Hub and XACS online computing services, with some functionality already publicly available as described at this http URL. Aitomia is expected to lower the barrier to performing atomistic simulations, accelerating research and development in the relevant fields. 

---
# DSADF: Thinking Fast and Slow for Decision Making 

**Authors**: Alex Zhihao Dou, Dongfei Cui, Jun Yan, Weida Wang, Benteng Chen, Haoming Wang, Zeke Xie, Shufei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08189)  

**Abstract**: Although Reinforcement Learning (RL) agents are effective in well-defined environments, they often struggle to generalize their learned policies to dynamic settings due to their reliance on trial-and-error interactions. Recent work has explored applying Large Language Models (LLMs) or Vision Language Models (VLMs) to boost the generalization of RL agents through policy optimization guidance or prior knowledge. However, these approaches often lack seamless coordination between the RL agent and the foundation model, leading to unreasonable decision-making in unfamiliar environments and efficiency bottlenecks. Making full use of the inferential capabilities of foundation models and the rapid response capabilities of RL agents and enhancing the interaction between the two to form a dual system is still a lingering scientific question. To address this problem, we draw inspiration from Kahneman's theory of fast thinking (System 1) and slow thinking (System 2), demonstrating that balancing intuition and deep reasoning can achieve nimble decision-making in a complex world. In this study, we propose a Dual-System Adaptive Decision Framework (DSADF), integrating two complementary modules: System 1, comprising an RL agent and a memory space for fast and intuitive decision making, and System 2, driven by a VLM for deep and analytical reasoning. DSADF facilitates efficient and adaptive decision-making by combining the strengths of both systems. The empirical study in the video game environment: Crafter and Housekeep demonstrates the effectiveness of our proposed method, showing significant improvements in decision abilities for both unseen and known tasks. 

---
# Feasibility-Aware Pessimistic Estimation: Toward Long-Horizon Safety in Offline RL 

**Authors**: Zhikun Tao, Gang Xiong, He Fang, Zhen Shen, Yunjun Han, Qing-Shan Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.08179)  

**Abstract**: Offline safe reinforcement learning(OSRL) derives constraint-satisfying policies from pre-collected datasets, offers a promising avenue for deploying RL in safety-critical real-world domains such as robotics. However, the majority of existing approaches emphasize only short-term safety, neglecting long-horizon considerations. Consequently, they may violate safety constraints and fail to ensure sustained protection during online deployment. Moreover, the learned policies often struggle to handle states and actions that are not present or out-of-distribution(OOD) from the offline dataset, and exhibit limited sample efficiency. To address these challenges, we propose a novel framework Feasibility-Aware offline Safe Reinforcement Learning with CVAE-based Pessimism (FASP). First, we employ Hamilton-Jacobi (H-J) reachability analysis to generate reliable safety labels, which serve as supervisory signals for training both a conditional variational autoencoder (CVAE) and a safety classifier. This approach not only ensures high sampling efficiency but also provides rigorous long-horizon safety guarantees. Furthermore, we utilize pessimistic estimation methods to estimate the Q-value of reward and cost, which mitigates the extrapolation errors induces by OOD actions, and penalize unsafe actions to enabled the agent to proactively avoid high-risk behaviors. Moreover, we theoretically prove the validity of this pessimistic estimation. Extensive experiments on DSRL benchmarks demonstrate that FASP algorithm achieves competitive performance across multiple experimental tasks, particularly outperforming state-of-the-art algorithms in terms of safety. 

---
# Fast Text-to-Audio Generation with Adversarial Post-Training 

**Authors**: Zachary Novack, Zach Evans, Zack Zukowski, Josiah Taylor, CJ Carr, Julian Parker, Adnan Al-Sinan, Gian Marco Iodice, Julian McAuley, Taylor Berg-Kirkpatrick, Jordi Pons  

**Link**: [PDF](https://arxiv.org/pdf/2505.08175)  

**Abstract**: Text-to-audio systems, while increasingly performant, are slow at inference time, thus making their latency unpractical for many creative applications. We present Adversarial Relativistic-Contrastive (ARC) post-training, the first adversarial acceleration algorithm for diffusion/flow models not based on distillation. While past adversarial post-training methods have struggled to compare against their expensive distillation counterparts, ARC post-training is a simple procedure that (1) extends a recent relativistic adversarial formulation to diffusion/flow post-training and (2) combines it with a novel contrastive discriminator objective to encourage better prompt adherence. We pair ARC post-training with a number optimizations to Stable Audio Open and build a model capable of generating $\approx$12s of 44.1kHz stereo audio in $\approx$75ms on an H100, and $\approx$7s on a mobile edge-device, the fastest text-to-audio model to our knowledge. 

---
# Exploiting Text Semantics for Few and Zero Shot Node Classification on Text-attributed Graph 

**Authors**: Yuxiang Wang, Xiao Yan, Shiyu Jin, Quanqing Xu, Chuang Hu, Yuanyuan Zhu, Bo Du, Jia Wu, Jiawei Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08168)  

**Abstract**: Text-attributed graph (TAG) provides a text description for each graph node, and few- and zero-shot node classification on TAGs have many applications in fields such as academia and social networks. Existing work utilizes various graph-based augmentation techniques to train the node and text embeddings, while text-based augmentations are largely unexplored. In this paper, we propose Text Semantics Augmentation (TSA) to improve accuracy by introducing more text semantic supervision signals. Specifically, we design two augmentation techniques, i.e., positive semantics matching and negative semantics contrast, to provide more reference texts for each graph node or text description. Positive semantic matching retrieves texts with similar embeddings to match with a graph node. Negative semantic contrast adds a negative prompt to construct a text description with the opposite semantics, which is contrasted with the original node and text. We evaluate TSA on 5 datasets and compare with 13 state-of-the-art baselines. The results show that TSA consistently outperforms all baselines, and its accuracy improvements over the best-performing baseline are usually over 5%. 

---
# Fusing Bidirectional Chains of Thought and Reward Mechanisms A Method for Enhancing Question-Answering Capabilities of Large Language Models for Chinese Intangible Cultural Heritage 

**Authors**: Ruilin Liu, Zhixiao Zhao, Jieqiong Li, Chang Liu, Dongbo Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08167)  

**Abstract**: The rapid development of large language models (LLMs) has provided significant support and opportunities for the advancement of domain-specific LLMs. However, fine-tuning these large models using Intangible Cultural Heritage (ICH) data inevitably faces challenges such as bias, incorrect knowledge inheritance, and catastrophic forgetting. To address these issues, we propose a novel training method that integrates a bidirectional chains of thought and a reward mechanism. This method is built upon ICH-Qwen, a large language model specifically designed for the field of intangible cultural heritage. The proposed method enables the model to not only perform forward reasoning but also enhances the accuracy of the generated answers by utilizing reverse questioning and reverse reasoning to activate the model's latent knowledge. Additionally, a reward mechanism is introduced during training to optimize the decision-making process. This mechanism improves the quality of the model's outputs through structural and content evaluations with different weighting schemes. We conduct comparative experiments on ICH-Qwen, with results demonstrating that our method outperforms 0-shot, step-by-step reasoning, knowledge distillation, and question augmentation methods in terms of accuracy, Bleu-4, and Rouge-L scores on the question-answering task. Furthermore, the paper highlights the effectiveness of combining the bidirectional chains of thought and reward mechanism through ablation experiments. In addition, a series of generalizability experiments are conducted, with results showing that the proposed method yields improvements on various domain-specific datasets and advanced models in areas such as Finance, Wikidata, and StrategyQA. This demonstrates that the method is adaptable to multiple domains and provides a valuable approach for model training in future applications across diverse fields. 

---
# Feature Fitted Online Conformal Prediction for Deep Time Series Forecasting Model 

**Authors**: Xiannan Huang, Shuhan Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08158)  

**Abstract**: Time series forecasting is critical for many applications, where deep learning-based point prediction models have demonstrated strong performance. However, in practical scenarios, there is also a need to quantify predictive uncertainty through online confidence intervals. Existing confidence interval modeling approaches building upon these deep point prediction models suffer from key limitations: they either require costly retraining, fail to fully leverage the representational strengths of deep models, or lack theoretical guarantees. To address these gaps, we propose a lightweight conformal prediction method that provides valid coverage and shorter interval lengths without retraining. Our approach leverages features extracted from pre-trained point prediction models to fit a residual predictor and construct confidence intervals, further enhanced by an adaptive coverage control mechanism. Theoretically, we prove that our method achieves asymptotic coverage convergence, with error bounds dependent on the feature quality of the underlying point prediction model. Experiments on 12 datasets demonstrate that our method delivers tighter confidence intervals while maintaining desired coverage rates. Code, model and dataset in \href{this https URL}{Github} 

---
# Hyperbolic Contrastive Learning with Model-augmentation for Knowledge-aware Recommendation 

**Authors**: Shengyin Sun, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.08157)  

**Abstract**: Benefiting from the effectiveness of graph neural networks (GNNs) and contrastive learning, GNN-based contrastive learning has become mainstream for knowledge-aware recommendation. However, most existing contrastive learning-based methods have difficulties in effectively capturing the underlying hierarchical structure within user-item bipartite graphs and knowledge graphs. Moreover, they commonly generate positive samples for contrastive learning by perturbing the graph structure, which may lead to a shift in user preference learning. To overcome these limitations, we propose hyperbolic contrastive learning with model-augmentation for knowledge-aware recommendation. To capture the intrinsic hierarchical graph structures, we first design a novel Lorentzian knowledge aggregation mechanism, which enables more effective representations of users and items. Then, we propose three model-level augmentation techniques to assist Hyperbolic contrastive learning. Different from the classical structure-level augmentation (e.g., edge dropping), the proposed model-augmentations can avoid preference shifts between the augmented positive pair. Finally, we conduct extensive experiments to demonstrate the superiority (maximum improvement of $11.03\%$) of proposed methods over existing baselines. 

---
# A Large-Scale Empirical Analysis of Custom GPTs' Vulnerabilities in the OpenAI Ecosystem 

**Authors**: Sunday Oyinlola Ogundoyin, Muhammad Ikram, Hassan Jameel Asghar, Benjamin Zi Hao Zhao, Dali Kaafar  

**Link**: [PDF](https://arxiv.org/pdf/2505.08148)  

**Abstract**: Millions of users leverage generative pretrained transformer (GPT)-based language models developed by leading model providers for a wide range of tasks. To support enhanced user interaction and customization, many platforms-such as OpenAI-now enable developers to create and publish tailored model instances, known as custom GPTs, via dedicated repositories or application stores. These custom GPTs empower users to browse and interact with specialized applications designed to meet specific needs. However, as custom GPTs see growing adoption, concerns regarding their security vulnerabilities have intensified. Existing research on these vulnerabilities remains largely theoretical, often lacking empirical, large-scale, and statistically rigorous assessments of associated risks.
In this study, we analyze 14,904 custom GPTs to assess their susceptibility to seven exploitable threats, such as roleplay-based attacks, system prompt leakage, phishing content generation, and malicious code synthesis, across various categories and popularity tiers within the OpenAI marketplace. We introduce a multi-metric ranking system to examine the relationship between a custom GPT's popularity and its associated security risks.
Our findings reveal that over 95% of custom GPTs lack adequate security protections. The most prevalent vulnerabilities include roleplay-based vulnerabilities (96.51%), system prompt leakage (92.20%), and phishing (91.22%). Furthermore, we demonstrate that OpenAI's foundational models exhibit inherent security weaknesses, which are often inherited or amplified in custom GPTs. These results highlight the urgent need for enhanced security measures and stricter content moderation to ensure the safe deployment of GPT-based applications. 

---
# Communication Styles and Reader Preferences of LLM and Human Experts in Explaining Health Information 

**Authors**: Jiawei Zhou, Kritika Venkatachalam, Minje Choi, Koustuv Saha, Munmun De Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.08143)  

**Abstract**: With the wide adoption of large language models (LLMs) in information assistance, it is essential to examine their alignment with human communication styles and values. We situate this study within the context of fact-checking health information, given the critical challenge of rectifying conceptions and building trust. Recent studies have explored the potential of LLM for health communication, but style differences between LLMs and human experts and associated reader perceptions remain under-explored. In this light, our study evaluates the communication styles of LLMs, focusing on how their explanations differ from those of humans in three core components of health communication: information, sender, and receiver. We compiled a dataset of 1498 health misinformation explanations from authoritative fact-checking organizations and generated LLM responses to inaccurate health information. Drawing from health communication theory, we evaluate communication styles across three key dimensions of information linguistic features, sender persuasive strategies, and receiver value alignments. We further assessed human perceptions through a blinded evaluation with 99 participants. Our findings reveal that LLM-generated articles showed significantly lower scores in persuasive strategies, certainty expressions, and alignment with social values and moral foundations. However, human evaluation demonstrated a strong preference for LLM content, with over 60% responses favoring LLM articles for clarity, completeness, and persuasiveness. Our results suggest that LLMs' structured approach to presenting information may be more effective at engaging readers despite scoring lower on traditional measures of quality in fact-checking and health communication. 

---
# Mirror Mirror on the Wall, Have I Forgotten it All? A New Framework for Evaluating Machine Unlearning 

**Authors**: Brennon Brimhall, Philip Mathew, Neil Fendley, Yinzhi Cao, Matthew Green  

**Link**: [PDF](https://arxiv.org/pdf/2505.08138)  

**Abstract**: Machine unlearning methods take a model trained on a dataset and a forget set, then attempt to produce a model as if it had only been trained on the examples not in the forget set. We empirically show that an adversary is able to distinguish between a mirror model (a control model produced by retraining without the data to forget) and a model produced by an unlearning method across representative unlearning methods from the literature. We build distinguishing algorithms based on evaluation scores in the literature (i.e. membership inference scores) and Kullback-Leibler divergence.
We propose a strong formal definition for machine unlearning called computational unlearning. Computational unlearning is defined as the inability for an adversary to distinguish between a mirror model and a model produced by an unlearning method. If the adversary cannot guess better than random (except with negligible probability), then we say that an unlearning method achieves computational unlearning.
Our computational unlearning definition provides theoretical structure to prove unlearning feasibility results. For example, our computational unlearning definition immediately implies that there are no deterministic computational unlearning methods for entropic learning algorithms. We also explore the relationship between differential privacy (DP)-based unlearning methods and computational unlearning, showing that DP-based approaches can satisfy computational unlearning at the cost of an extreme utility collapse. These results demonstrate that current methodology in the literature fundamentally falls short of achieving computational unlearning. We conclude by identifying several open questions for future work. 

---
# Leveraging AI for Productive and Trustworthy HPC Software: Challenges and Research Directions 

**Authors**: Keita Teranishi, Harshitha Menon, William F. Godoy, Prasanna Balaprakash, David Bau, Tal Ben-Nun, Abhinav Bathele, Franz Franchetti, Michael Franusich, Todd Gamblin, Giorgis Georgakoudis, Tom Goldstein, Arjun Guha, Steven Hahn, Costin Iancu, Zheming Jin, Terry Jones, Tze Meng Low, Het Mankad, Narasinga Rao Miniskar, Mohammad Alaul Haque Monil, Daniel Nichols, Konstantinos Parasyris, Swaroop Pophale, Pedro Valero-Lara, Jeffrey S. Vetter, Samuel Williams, Aaron Young  

**Link**: [PDF](https://arxiv.org/pdf/2505.08135)  

**Abstract**: We discuss the challenges and propose research directions for using AI to revolutionize the development of high-performance computing (HPC) software. AI technologies, in particular large language models, have transformed every aspect of software development. For its part, HPC software is recognized as a highly specialized scientific field of its own. We discuss the challenges associated with leveraging state-of-the-art AI technologies to develop such a unique and niche class of software and outline our research directions in the two US Department of Energy--funded projects for advancing HPC Software via AI: Ellora and Durban. 

---
# One Bad NOFO? AI Governance in Federal Grantmaking 

**Authors**: Dan Bateyko, Karen Levy  

**Link**: [PDF](https://arxiv.org/pdf/2505.08133)  

**Abstract**: Much scholarship considers how U.S. federal agencies govern artificial intelligence (AI) through rulemaking and their own internal use policies. But agencies have an overlooked AI governance role: setting discretionary grant policy when directing billions of dollars in federal financial assistance. These dollars enable state and local entities to study, create, and use AI. This funding not only goes to dedicated AI programs, but also to grantees using AI in the course of meeting their routine grant objectives. As discretionary grantmakers, agencies guide and restrict what grant winners do -- a hidden lever for AI governance. Agencies pull this lever by setting program objectives, judging criteria, and restrictions for AI use. Using a novel dataset of over 40,000 non-defense federal grant notices of funding opportunity (NOFOs) posted to this http URL between 2009 and 2024, we analyze how agencies regulate the use of AI by grantees. We select records mentioning AI and review their stated goals and requirements. We find agencies promoting AI in notice narratives, shaping adoption in ways other records of grant policy might fail to capture. Of the grant opportunities that mention AI, we find only a handful of AI-specific judging criteria or restrictions. This silence holds even when agencies fund AI uses in contexts affecting people's rights and which, under an analogous federal procurement regime, would result in extra oversight. These findings recast grant notices as a site of AI policymaking -- albeit one that is developing out of step with other regulatory efforts and incomplete in its consideration of transparency, accountability, and privacy protections. The paper concludes by drawing lessons from AI procurement scholarship, while identifying distinct challenges in grantmaking that invite further study. 

---
# ALOHA: Empowering Multilingual Agent for University Orientation with Hierarchical Retrieval 

**Authors**: Mingxu Tao, Bowen Tang, Mingxuan Ma, Yining Zhang, Hourun Li, Feifan Wen, Hao Ma, Jia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08130)  

**Abstract**: The rise of Large Language Models~(LLMs) revolutionizes information retrieval, allowing users to obtain required answers through complex instructions within conversations. However, publicly available services remain inadequate in addressing the needs of faculty and students to search campus-specific information. It is primarily due to the LLM's lack of domain-specific knowledge and the limitation of search engines in supporting multilingual and timely scenarios. To tackle these challenges, we introduce ALOHA, a multilingual agent enhanced by hierarchical retrieval for university orientation. We also integrate external APIs into the front-end interface to provide interactive service. The human evaluation and case study show our proposed system has strong capabilities to yield correct, timely, and user-friendly responses to the queries in multiple languages, surpassing commercial chatbots and search engines. The system has been deployed and has provided service for more than 12,000 people. 

---
# High-order Regularization for Machine Learning and Learning-based Control 

**Authors**: Xinghua Liu, Ming Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.08129)  

**Abstract**: The paper proposes a novel regularization procedure for machine learning. The proposed high-order regularization (HR) provides new insight into regularization, which is widely used to train a neural network that can be utilized to approximate the action-value function in general reinforcement learning problems. The proposed HR method ensures the provable convergence of the approximation algorithm, which makes the much-needed connection between regularization and explainable learning using neural networks. The proposed HR method theoretically demonstrates that regularization can be regarded as an approximation in terms of inverse mapping with explicitly calculable approximation error, and the $L_2$ regularization is a lower-order case of the proposed method. We provide lower and upper bounds for the error of the proposed HR solution, which helps build a reliable model. We also find that regularization with the proposed HR can be regarded as a contraction. We prove that the generalizability of neural networks can be maximized with a proper regularization matrix, and the proposed HR is applicable for neural networks with any mapping matrix. With the theoretical explanation of the extreme learning machine for neural network training and the proposed high-order regularization, one can better interpret the output of the neural network, thus leading to explainable learning. We present a case study based on regularized extreme learning neural networks to demonstrate the application of the proposed HR and give the corresponding incremental HR solution. We verify the performance of the proposed HR method by solving a classic control problem in reinforcement learning. The result demonstrates the superior performance of the method with significant enhancement in the generalizability of the neural network. 

---
# SLAG: Scalable Language-Augmented Gaussian Splatting 

**Authors**: Laszlo Szilagyi, Francis Engelmann, Jeannette Bohg  

**Link**: [PDF](https://arxiv.org/pdf/2505.08124)  

**Abstract**: Language-augmented scene representations hold great promise for large-scale robotics applications such as search-and-rescue, smart cities, and mining. Many of these scenarios are time-sensitive, requiring rapid scene encoding while also being data-intensive, necessitating scalable solutions. Deploying these representations on robots with limited computational resources further adds to the challenge. To address this, we introduce SLAG, a multi-GPU framework for language-augmented Gaussian splatting that enhances the speed and scalability of embedding large scenes. Our method integrates 2D visual-language model features into 3D scenes using SAM and CLIP. Unlike prior approaches, SLAG eliminates the need for a loss function to compute per-Gaussian language embeddings. Instead, it derives embeddings from 3D Gaussian scene parameters via a normalized weighted average, enabling highly parallelized scene encoding. Additionally, we introduce a vector database for efficient embedding storage and retrieval. Our experiments show that SLAG achieves an 18 times speedup in embedding computation on a 16-GPU setup compared to OpenGaussian, while preserving embedding quality on the ScanNet and LERF datasets. For more details, visit our project website: this https URL. 

---
# JSover: Joint Spectrum Estimation and Multi-Material Decomposition from Single-Energy CT Projections 

**Authors**: Qing Wu, Hongjiang Wei, Jingyi Yu, S. Kevin Zhou, Yuyao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08123)  

**Abstract**: Multi-material decomposition (MMD) enables quantitative reconstruction of tissue compositions in the human body, supporting a wide range of clinical applications. However, traditional MMD typically requires spectral CT scanners and pre-measured X-ray energy spectra, significantly limiting clinical applicability. To this end, various methods have been developed to perform MMD using conventional (i.e., single-energy, SE) CT systems, commonly referred to as SEMMD. Despite promising progress, most SEMMD methods follow a two-step image decomposition pipeline, which first reconstructs monochromatic CT images using algorithms such as FBP, and then performs decomposition on these images. The initial reconstruction step, however, neglects the energy-dependent attenuation of human tissues, introducing severe nonlinear beam hardening artifacts and noise into the subsequent decomposition. This paper proposes JSover, a fundamentally reformulated one-step SEMMD framework that jointly reconstructs multi-material compositions and estimates the energy spectrum directly from SECT projections. By explicitly incorporating physics-informed spectral priors into the SEMMD process, JSover accurately simulates a virtual spectral CT system from SE acquisitions, thereby improving the reliability and accuracy of decomposition. Furthermore, we introduce implicit neural representation (INR) as an unsupervised deep learning solver for representing the underlying material maps. The inductive bias of INR toward continuous image patterns constrains the solution space and further enhances estimation quality. Extensive experiments on both simulated and real CT datasets show that JSover outperforms state-of-the-art SEMMD methods in accuracy and computational efficiency. 

---
# Are LLMs complicated ethical dilemma analyzers? 

**Authors**: Jiashen, Jesse Yao, Allen Liu, Zhekai Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08106)  

**Abstract**: One open question in the study of Large Language Models (LLMs) is whether they can emulate human ethical reasoning and act as believable proxies for human judgment. To investigate this, we introduce a benchmark dataset comprising 196 real-world ethical dilemmas and expert opinions, each segmented into five structured components: Introduction, Key Factors, Historical Theoretical Perspectives, Resolution Strategies, and Key Takeaways. We also collect non-expert human responses for comparison, limited to the Key Factors section due to their brevity. We evaluate multiple frontier LLMs (GPT-4o-mini, Claude-3.5-Sonnet, Deepseek-V3, Gemini-1.5-Flash) using a composite metric framework based on BLEU, Damerau-Levenshtein distance, TF-IDF cosine similarity, and Universal Sentence Encoder similarity. Metric weights are computed through an inversion-based ranking alignment and pairwise AHP analysis, enabling fine-grained comparison of model outputs to expert responses. Our results show that LLMs generally outperform non-expert humans in lexical and structural alignment, with GPT-4o-mini performing most consistently across all sections. However, all models struggle with historical grounding and proposing nuanced resolution strategies, which require contextual abstraction. Human responses, while less structured, occasionally achieve comparable semantic similarity, suggesting intuitive moral reasoning. These findings highlight both the strengths and current limitations of LLMs in ethical decision-making. 

---
# Graph-Based Floor Separation Using Node Embeddings and Clustering of WiFi Trajectories 

**Authors**: Rabia Yasa Kostas, Kahraman Kostas  

**Link**: [PDF](https://arxiv.org/pdf/2505.08088)  

**Abstract**: Indoor positioning systems (IPSs) are increasingly vital for location-based services in complex multi-storey environments. This study proposes a novel graph-based approach for floor separation using Wi-Fi fingerprint trajectories, addressing the challenge of vertical localization in indoor settings. We construct a graph where nodes represent Wi-Fi fingerprints, and edges are weighted by signal similarity and contextual transitions. Node2Vec is employed to generate low-dimensional embeddings, which are subsequently clustered using K-means to identify distinct floors. Evaluated on the Huawei University Challenge 2021 dataset, our method outperforms traditional community detection algorithms, achieving an accuracy of 68.97%, an F1- score of 61.99%, and an Adjusted Rand Index of 57.19%. By publicly releasing the preprocessed dataset and implementation code, this work contributes to advancing research in indoor positioning. The proposed approach demonstrates robustness to signal noise and architectural complexities, offering a scalable solution for floor-level localization. 

---
# Fréchet Power-Scenario Distance: A Metric for Evaluating Generative AI Models across Multiple Time-Scales in Smart Grids 

**Authors**: Yuting Cai, Shaohuai Liu, Chao Tian, Le Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.08082)  

**Abstract**: Generative artificial intelligence (AI) models in smart grids have advanced significantly in recent years due to their ability to generate large amounts of synthetic data, which would otherwise be difficult to obtain in the real world due to confidentiality constraints. A key challenge in utilizing such synthetic data is how to assess the data quality produced from such generative models. Traditional Euclidean distance-based metrics only reflect pair-wise relations between two individual samples, and could fail in evaluating quality differences between groups of synthetic datasets. In this work, we propose a novel metric based on the Fréchet Distance (FD) estimated between two datasets in a learned feature space. The proposed method evaluates the quality of generation from a distributional perspective. Empirical results demonstrate the superiority of the proposed metric across timescales and models, enhancing the reliability of data-driven decision-making in smart grid operations. 

---
# Beyond Input Activations: Identifying Influential Latents by Gradient Sparse Autoencoders 

**Authors**: Dong Shu, Xuansheng Wu, Haiyan Zhao, Mengnan Du, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.08080)  

**Abstract**: Sparse Autoencoders (SAEs) have recently emerged as powerful tools for interpreting and steering the internal representations of large language models (LLMs). However, conventional approaches to analyzing SAEs typically rely solely on input-side activations, without considering the causal influence between each latent feature and the model's output. This work is built on two key hypotheses: (1) activated latents do not contribute equally to the construction of the model's output, and (2) only latents with high causal influence are effective for model steering. To validate these hypotheses, we propose Gradient Sparse Autoencoder (GradSAE), a simple yet effective method that identifies the most influential latents by incorporating output-side gradient information. 

---
# What Matters for Batch Online Reinforcement Learning in Robotics? 

**Authors**: Perry Dong, Suvir Mirchandani, Dorsa Sadigh, Chelsea Finn  

**Link**: [PDF](https://arxiv.org/pdf/2505.08078)  

**Abstract**: The ability to learn from large batches of autonomously collected data for policy improvement -- a paradigm we refer to as batch online reinforcement learning -- holds the promise of enabling truly scalable robot learning by significantly reducing the need for human effort of data collection while getting benefits from self-improvement. Yet, despite the promise of this paradigm, it remains challenging to achieve due to algorithms not being able to learn effectively from the autonomous data. For example, prior works have applied imitation learning and filtered imitation learning methods to the batch online RL problem, but these algorithms often fail to efficiently improve from the autonomously collected data or converge quickly to a suboptimal point. This raises the question of what matters for effective batch online RL in robotics. Motivated by this question, we perform a systematic empirical study of three axes -- (i) algorithm class, (ii) policy extraction methods, and (iii) policy expressivity -- and analyze how these axes affect performance and scaling with the amount of autonomous data. Through our analysis, we make several observations. First, we observe that the use of Q-functions to guide batch online RL significantly improves performance over imitation-based methods. Building on this, we show that an implicit method of policy extraction -- via choosing the best action in the distribution of the policy -- is necessary over traditional policy extraction methods from offline RL. Next, we show that an expressive policy class is preferred over less expressive policy classes. Based on this analysis, we propose a general recipe for effective batch online RL. We then show a simple addition to the recipe of using temporally-correlated noise to obtain more diversity results in further performance gains. Our recipe obtains significantly better performance and scaling compared to prior methods. 

---
# Justified Evidence Collection for Argument-based AI Fairness Assurance 

**Authors**: Alpay Sabuncuoglu, Christopher Burr, Carsten Maple  

**Link**: [PDF](https://arxiv.org/pdf/2505.08064)  

**Abstract**: It is well recognised that ensuring fair AI systems is a complex sociotechnical challenge, which requires careful deliberation and continuous oversight across all stages of a system's lifecycle, from defining requirements to model deployment and deprovisioning. Dynamic argument-based assurance cases, which present structured arguments supported by evidence, have emerged as a systematic approach to evaluating and mitigating safety risks and hazards in AI-enabled system development and have also been extended to deal with broader normative goals such as fairness and explainability. This paper introduces a systems-engineering-driven framework, supported by software tooling, to operationalise a dynamic approach to argument-based assurance in two stages. In the first stage, during the requirements planning phase, a multi-disciplinary and multi-stakeholder team define goals and claims to be established (and evidenced) by conducting a comprehensive fairness governance process. In the second stage, a continuous monitoring interface gathers evidence from existing artefacts (e.g. metrics from automated tests), such as model, data, and use case documentation, to support these arguments dynamically. The framework's effectiveness is demonstrated through an illustrative case study in finance, with a focus on supporting fairness-related arguments. 

---
# FalseReject: A Resource for Improving Contextual Safety and Mitigating Over-Refusals in LLMs via Structured Reasoning 

**Authors**: Zhehao Zhang, Weijie Xu, Fanyou Wu, Chandan K. Reddy  

**Link**: [PDF](https://arxiv.org/pdf/2505.08054)  

**Abstract**: Safety alignment approaches in large language models (LLMs) often lead to the over-refusal of benign queries, significantly diminishing their utility in sensitive scenarios. To address this challenge, we introduce FalseReject, a comprehensive resource containing 16k seemingly toxic queries accompanied by structured responses across 44 safety-related categories. We propose a graph-informed adversarial multi-agent interaction framework to generate diverse and complex prompts, while structuring responses with explicit reasoning to aid models in accurately distinguishing safe from unsafe contexts. FalseReject includes training datasets tailored for both standard instruction-tuned models and reasoning-oriented models, as well as a human-annotated benchmark test set. Our extensive benchmarking on 29 state-of-the-art (SOTA) LLMs reveals persistent over-refusal challenges. Empirical results demonstrate that supervised finetuning with FalseReject substantially reduces unnecessary refusals without compromising overall safety or general language capabilities. 

---
# NAZM: Network Analysis of Zonal Metrics in Persian Poetic Tradition 

**Authors**: Kourosh Shahnazari, Seyed Moein Ayyoubzadeh  

**Link**: [PDF](https://arxiv.org/pdf/2505.08052)  

**Abstract**: This study formalizes a computational model to simulate classical Persian poets' dynamics of influence through constructing a multi-dimensional similarity network. Using a rigorously curated dataset based on Ganjoor's corpus, we draw upon semantic, lexical, stylistic, thematic, and metrical features to demarcate each poet's corpus. Each is contained within weighted similarity matrices, which are then appended to generate an aggregate graph showing poet-to-poet influence. Further network investigation is carried out to identify key poets, style hubs, and bridging poets by calculating degree, closeness, betweenness, eigenvector, and Katz centrality measures. Further, for typological insight, we use the Louvain community detection algorithm to demarcate clusters of poets sharing both style and theme coherence, which correspond closely to acknowledged schools of literature like Sabk-e Hindi, Sabk-e Khorasani, and the Bazgasht-e Adabi phenomenon. Our findings provide a new data-driven view of Persian literature distinguished between canonical significance and interextual influence, thus highlighting relatively lesser-known figures who hold great structural significance. Combining computational linguistics with literary study, this paper produces an interpretable and scalable model for poetic tradition, enabling retrospective reflection as well as forward-looking research within digital humanities. 

---
# Online Learning-based Adaptive Beam Switching for 6G Networks: Enhancing Efficiency and Resilience 

**Authors**: Seyed Bagher Hashemi Natanzi, Zhicong Zhu, Bo Tang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08032)  

**Abstract**: Adaptive beam switching in 6G networks is challenged by high frequencies, mobility, and blockage. We propose an Online Learning framework using Deep Reinforcement Learning (DRL) with an enhanced state representation (velocity and blockage history), a GRU architecture, and prioritized experience replay for real-time beam optimization. Validated via Nvidia Sionna under time-correlated blockage, our approach significantly enhances resilience in SNR, throughput, and accuracy compared to a conventional heuristic. Furthermore, the enhanced DRL agent outperforms a reactive Multi-Armed Bandit (MAB) baseline by leveraging temporal dependencies, achieving lower performance variability. This demonstrates the benefits of memory and prioritized learning for robust 6G beam management, while confirming MAB as a strong baseline. 

---
# PRISM: Complete Online Decentralized Multi-Agent Pathfinding with Rapid Information Sharing using Motion Constraints 

**Authors**: Hannah Lee, Zachary Serlin, James Motes, Brendan Long, Marco Morales, Nancy M. Amato  

**Link**: [PDF](https://arxiv.org/pdf/2505.08025)  

**Abstract**: We introduce PRISM (Pathfinding with Rapid Information Sharing using Motion Constraints), a decentralized algorithm designed to address the multi-task multi-agent pathfinding (MT-MAPF) problem. PRISM enables large teams of agents to concurrently plan safe and efficient paths for multiple tasks while avoiding collisions. It employs a rapid communication strategy that uses information packets to exchange motion constraint information, enhancing cooperative pathfinding and situational awareness, even in scenarios without direct communication. We prove that PRISM resolves and avoids all deadlock scenarios when possible, a critical challenge in decentralized pathfinding. Empirically, we evaluate PRISM across five environments and 25 random scenarios, benchmarking it against the centralized Conflict-Based Search (CBS) and the decentralized Token Passing with Task Swaps (TPTS) algorithms. PRISM demonstrates scalability and solution quality, supporting 3.4 times more agents than CBS and handling up to 2.5 times more tasks in narrow passage environments than TPTS. Additionally, PRISM matches CBS in solution quality while achieving faster computation times, even under low-connectivity conditions. Its decentralized design reduces the computational burden on individual agents, making it scalable for large environments. These results confirm PRISM's robustness, scalability, and effectiveness in complex and dynamic pathfinding scenarios. 

---
# Large Language Models and Arabic Content: A Review 

**Authors**: Haneh Rhel, Dmitri Roussinov  

**Link**: [PDF](https://arxiv.org/pdf/2505.08004)  

**Abstract**: Over the past three years, the rapid advancement of Large Language Models (LLMs) has had a profound impact on multiple areas of Artificial Intelligence (AI), particularly in Natural Language Processing (NLP) across diverse languages, including Arabic. Although Arabic is considered one of the most widely spoken languages across 27 countries in the Arabic world and used as a second language in some other non-Arabic countries as well, there is still a scarcity of Arabic resources, datasets, and tools. Arabic NLP tasks face various challenges due to the complexities of the Arabic language, including its rich morphology, intricate structure, and diverse writing standards, among other factors. Researchers have been actively addressing these challenges, demonstrating that pre-trained Large Language Models (LLMs) trained on multilingual corpora achieve significant success in various Arabic NLP tasks. This study provides an overview of using large language models (LLMs) for the Arabic language, highlighting early pre-trained Arabic Language models across various NLP applications and their ability to handle diverse Arabic content tasks and dialects. It also provides an overview of how techniques like finetuning and prompt engineering can enhance the performance of these models. Additionally, the study summarizes common Arabic benchmarks and datasets while presenting our observations on the persistent upward trend in the adoption of LLMs. 

---
# Fair Play for Individuals, Foul Play for Groups? Auditing Anonymization's Impact on ML Fairness 

**Authors**: Héber H. Arcolezi, Mina Alishahi, Adda-Akram Bendoukha, Nesrine Kaaniche  

**Link**: [PDF](https://arxiv.org/pdf/2505.07985)  

**Abstract**: Machine learning (ML) algorithms are heavily based on the availability of training data, which, depending on the domain, often includes sensitive information about data providers. This raises critical privacy concerns. Anonymization techniques have emerged as a practical solution to address these issues by generalizing features or suppressing data to make it more difficult to accurately identify individuals. Although recent studies have shown that privacy-enhancing technologies can influence ML predictions across different subgroups, thus affecting fair decision-making, the specific effects of anonymization techniques, such as $k$-anonymity, $\ell$-diversity, and $t$-closeness, on ML fairness remain largely unexplored. In this work, we systematically audit the impact of anonymization techniques on ML fairness, evaluating both individual and group fairness. Our quantitative study reveals that anonymization can degrade group fairness metrics by up to four orders of magnitude. Conversely, similarity-based individual fairness metrics tend to improve under stronger anonymization, largely as a result of increased input homogeneity. By analyzing varying levels of anonymization across diverse privacy settings and data distributions, this study provides critical insights into the trade-offs between privacy, fairness, and utility, offering actionable guidelines for responsible AI development. Our code is publicly available at: this https URL. 

---
# Probabilistic approach to longitudinal response prediction: application to radiomics from brain cancer imaging 

**Authors**: Isabella Cama, Michele Piana, Cristina Campi, Sara Garbarino  

**Link**: [PDF](https://arxiv.org/pdf/2505.07973)  

**Abstract**: Longitudinal imaging analysis tracks disease progression and treatment response over time, providing dynamic insights into treatment efficacy and disease evolution. Radiomic features extracted from medical imaging can support the study of disease progression and facilitate longitudinal prediction of clinical outcomes. This study presents a probabilistic model for longitudinal response prediction, integrating baseline features with intermediate follow-ups. The probabilistic nature of the model naturally allows to handle the instrinsic uncertainty of the longitudinal prediction of disease progression. We evaluate the proposed model against state-of-the-art disease progression models in both a synthetic scenario and using a brain cancer dataset. Results demonstrate that the approach is competitive against existing methods while uniquely accounting for uncertainty and controlling the growth of problem dimensionality, eliminating the need for data from intermediate follow-ups. 

---
# Self-cross Feature based Spiking Neural Networks for Efficient Few-shot Learning 

**Authors**: Qi Xu, Junyang Zhu, Dongdong Zhou, Hao Chen, Yang Liu, Jiangrong Shen, Qiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07921)  

**Abstract**: Deep neural networks (DNNs) excel in computer vision tasks, especially, few-shot learning (FSL), which is increasingly important for generalizing from limited examples. However, DNNs are computationally expensive with scalability issues in real world. Spiking Neural Networks (SNNs), with their event-driven nature and low energy consumption, are particularly efficient in processing sparse and dynamic data, though they still encounter difficulties in capturing complex spatiotemporal features and performing accurate cross-class comparisons. To further enhance the performance and efficiency of SNNs in few-shot learning, we propose a few-shot learning framework based on SNNs, which combines a self-feature extractor module and a cross-feature contrastive module to refine feature representation and reduce power consumption. We apply the combination of temporal efficient training loss and InfoNCE loss to optimize the temporal dynamics of spike trains and enhance the discriminative power. Experimental results show that the proposed FSL-SNN significantly improves the classification performance on the neuromorphic dataset N-Omniglot, and also achieves competitive performance to ANNs on static datasets such as CUB and miniImageNet with low power consumption. 

---
# Re$^2$: A Consistency-ensured Dataset for Full-stage Peer Review and Multi-turn Rebuttal Discussions 

**Authors**: Daoze Zhang, Zhijian Bao, Sihang Du, Zhiyi Zhao, Kuangling Zhang, Dezheng Bao, Yang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07920)  

**Abstract**: Peer review is a critical component of scientific progress in the fields like AI, but the rapid increase in submission volume has strained the reviewing system, which inevitably leads to reviewer shortages and declines review quality. Besides the growing research popularity, another key factor in this overload is the repeated resubmission of substandard manuscripts, largely due to the lack of effective tools for authors to self-evaluate their work before submission. Large Language Models (LLMs) show great promise in assisting both authors and reviewers, and their performance is fundamentally limited by the quality of the peer review data. However, existing peer review datasets face three major limitations: (1) limited data diversity, (2) inconsistent and low-quality data due to the use of revised rather than initial submissions, and (3) insufficient support for tasks involving rebuttal and reviewer-author interactions. To address these challenges, we introduce the largest consistency-ensured peer review and rebuttal dataset named Re^2, which comprises 19,926 initial submissions, 70,668 review comments, and 53,818 rebuttals from 24 conferences and 21 workshops on OpenReview. Moreover, the rebuttal and discussion stage is framed as a multi-turn conversation paradigm to support both traditional static review tasks and dynamic interactive LLM assistants, providing more practical guidance for authors to refine their manuscripts and helping alleviate the growing review burden. Our data and code are available in this https URL. 

---
# Efficient and Reproducible Biomedical Question Answering using Retrieval Augmented Generation 

**Authors**: Linus Stuhlmann, Michael Alexander Saxer, Jonathan Fürst  

**Link**: [PDF](https://arxiv.org/pdf/2505.07917)  

**Abstract**: Biomedical question-answering (QA) systems require effective retrieval and generation components to ensure accuracy, efficiency, and scalability. This study systematically examines a Retrieval-Augmented Generation (RAG) system for biomedical QA, evaluating retrieval strategies and response time trade-offs. We first assess state-of-the-art retrieval methods, including BM25, BioBERT, MedCPT, and a hybrid approach, alongside common data stores such as Elasticsearch, MongoDB, and FAISS, on a ~10% subset of PubMed (2.4M documents) to measure indexing efficiency, retrieval latency, and retriever performance in the end-to-end RAG system. Based on these insights, we deploy the final RAG system on the full 24M PubMed corpus, comparing different retrievers' impact on overall performance. Evaluations of the retrieval depth show that retrieving 50 documents with BM25 before reranking with MedCPT optimally balances accuracy (0.90), recall (0.90), and response time (1.91s). BM25 retrieval time remains stable (82ms), while MedCPT incurs the main computational cost. These results highlight previously not well-known trade-offs in retrieval depth, efficiency, and scalability for biomedical QA. With open-source code, the system is fully reproducible and extensible. 

---
# Combining Bayesian Inference and Reinforcement Learning for Agent Decision Making: A Review 

**Authors**: Chengmin Zhou, Ville Kyrki, Pasi Fränti, Laura Ruotsalainen  

**Link**: [PDF](https://arxiv.org/pdf/2505.07911)  

**Abstract**: Bayesian inference has many advantages in decision making of agents (e.g. robotics/simulative agent) over a regular data-driven black-box neural network: Data-efficiency, generalization, interpretability, and safety where these advantages benefit directly/indirectly from the uncertainty quantification of Bayesian inference. However, there are few comprehensive reviews to summarize the progress of Bayesian inference on reinforcement learning (RL) for decision making to give researchers a systematic understanding. This paper focuses on combining Bayesian inference with RL that nowadays is an important approach in agent decision making. To be exact, this paper discusses the following five topics: 1) Bayesian methods that have potential for agent decision making. First basic Bayesian methods and models (Bayesian rule, Bayesian learning, and Bayesian conjugate models) are discussed followed by variational inference, Bayesian optimization, Bayesian deep learning, Bayesian active learning, Bayesian generative models, Bayesian meta-learning, and lifelong Bayesian learning. 2) Classical combinations of Bayesian methods with model-based RL (with approximation methods), model-free RL, and inverse RL. 3) Latest combinations of potential Bayesian methods with RL. 4) Analytical comparisons of methods that combine Bayesian methods with RL with respect to data-efficiency, generalization, interpretability, and safety. 5) In-depth discussions in six complex problem variants of RL, including unknown reward, partial-observability, multi-agent, multi-task, non-linear non-Gaussian, and hierarchical RL problems and the summary of how Bayesian methods work in the data collection, data processing and policy learning stages of RL to pave the way for better agent decision-making strategies. 

---
# Tuning for Trustworthiness -- Balancing Performance and Explanation Consistency in Neural Network Optimization 

**Authors**: Alexander Hinterleitner, Thomas Bartz-Beielstein  

**Link**: [PDF](https://arxiv.org/pdf/2505.07910)  

**Abstract**: Despite the growing interest in Explainable Artificial Intelligence (XAI), explainability is rarely considered during hyperparameter tuning or neural architecture optimization, where the focus remains primarily on minimizing predictive loss. In this work, we introduce the novel concept of XAI consistency, defined as the agreement among different feature attribution methods, and propose new metrics to quantify it. For the first time, we integrate XAI consistency directly into the hyperparameter tuning objective, creating a multi-objective optimization framework that balances predictive performance with explanation robustness. Implemented within the Sequential Parameter Optimization Toolbox (SPOT), our approach uses both weighted aggregation and desirability-based strategies to guide model selection. Through our proposed framework and supporting tools, we explore the impact of incorporating XAI consistency into the optimization process. This enables us to characterize distinct regions in the architecture configuration space: one region with poor performance and comparatively low interpretability, another with strong predictive performance but weak interpretability due to low \gls{xai} consistency, and a trade-off region that balances both objectives by offering high interpretability alongside competitive performance. Beyond introducing this novel approach, our research provides a foundation for future investigations into whether models from the trade-off zone-balancing performance loss and XAI consistency-exhibit greater robustness by avoiding overfitting to training performance, thereby leading to more reliable predictions on out-of-distribution data. 

---
# A Reproduction Study: The Kernel PCA Interpretation of Self-Attention Fails Under Scrutiny 

**Authors**: Karahan Sarıtaş, Çağatay Yıldız  

**Link**: [PDF](https://arxiv.org/pdf/2505.07908)  

**Abstract**: In this reproduction study, we revisit recent claims that self-attention implements kernel principal component analysis (KPCA) (Teo et al., 2024), positing that (i) value vectors $V$ capture the eigenvectors of the Gram matrix of the keys, and (ii) that self-attention projects queries onto the principal component axes of the key matrix $K$ in a feature space. Our analysis reveals three critical inconsistencies: (1) No alignment exists between learned self-attention value vectors and what is proposed in the KPCA perspective, with average similarity metrics (optimal cosine similarity $\leq 0.32$, linear CKA (Centered Kernel Alignment) $\leq 0.11$, kernel CKA $\leq 0.32$) indicating negligible correspondence; (2) Reported decreases in reconstruction loss $J_\text{proj}$, arguably justifying the claim that the self-attention minimizes the projection error of KPCA, are misinterpreted, as the quantities involved differ by orders of magnitude ($\sim\!10^3$); (3) Gram matrix eigenvalue statistics, introduced to justify that $V$ captures the eigenvector of the gram matrix, are irreproducible without undocumented implementation-specific adjustments. Across 10 transformer architectures, we conclude that the KPCA interpretation of self-attention lacks empirical support. 

---
# SEM: Reinforcement Learning for Search-Efficient Large Language Models 

**Authors**: Zeyang Sha, Shiwen Cui, Weiqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07903)  

**Abstract**: Recent advancements in Large Language Models(LLMs) have demonstrated their capabilities not only in reasoning but also in invoking external tools, particularly search engines. However, teaching models to discern when to invoke search and when to rely on their internal knowledge remains a significant challenge. Existing reinforcement learning approaches often lead to redundant search behaviors, resulting in inefficiencies and over-cost. In this paper, we propose SEM, a novel post-training reinforcement learning framework that explicitly trains LLMs to optimize search usage. By constructing a balanced dataset combining MuSiQue and MMLU, we create scenarios where the model must learn to distinguish between questions it can answer directly and those requiring external retrieval. We design a structured reasoning template and employ Group Relative Policy Optimization(GRPO) to post-train the model's search behaviors. Our reward function encourages accurate answering without unnecessary search while promoting effective retrieval when needed. Experimental results demonstrate that our method significantly reduces redundant search operations while maintaining or improving answer accuracy across multiple challenging benchmarks. This framework advances the model's reasoning efficiency and extends its capability to judiciously leverage external knowledge. 

---
# Multimodal Assessment of Classroom Discourse Quality: A Text-Centered Attention-Based Multi-Task Learning Approach 

**Authors**: Ruikun Hou, Babette Bühler, Tim Fütterer, Efe Bozkir, Peter Gerjets, Ulrich Trautwein, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2505.07902)  

**Abstract**: Classroom discourse is an essential vehicle through which teaching and learning take place. Assessing different characteristics of discursive practices and linking them to student learning achievement enhances the understanding of teaching quality. Traditional assessments rely on manual coding of classroom observation protocols, which is time-consuming and costly. Despite many studies utilizing AI techniques to analyze classroom discourse at the utterance level, investigations into the evaluation of discursive practices throughout an entire lesson segment remain limited. To address this gap, our study proposes a novel text-centered multimodal fusion architecture to assess the quality of three discourse components grounded in the Global Teaching InSights (GTI) observation protocol: Nature of Discourse, Questioning, and Explanations. First, we employ attention mechanisms to capture inter- and intra-modal interactions from transcript, audio, and video streams. Second, a multi-task learning approach is adopted to jointly predict the quality scores of the three components. Third, we formulate the task as an ordinal classification problem to account for rating level order. The effectiveness of these designed elements is demonstrated through an ablation study on the GTI Germany dataset containing 92 videotaped math lessons. Our results highlight the dominant role of text modality in approaching this task. Integrating acoustic features enhances the model's consistency with human ratings, achieving an overall Quadratic Weighted Kappa score of 0.384, comparable to human inter-rater reliability (0.326). Our study lays the groundwork for the future development of automated discourse quality assessment to support teacher professional development through timely feedback on multidimensional discourse practices. 

---
# Latent Behavior Diffusion for Sequential Reaction Generation in Dyadic Setting 

**Authors**: Minh-Duc Nguyen, Hyung-Jeong Yang, Soo-Hyung Kim, Ji-Eun Shin, Seung-Won Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.07901)  

**Abstract**: The dyadic reaction generation task involves synthesizing responsive facial reactions that align closely with the behaviors of a conversational partner, enhancing the naturalness and effectiveness of human-like interaction simulations. This paper introduces a novel approach, the Latent Behavior Diffusion Model, comprising a context-aware autoencoder and a diffusion-based conditional generator that addresses the challenge of generating diverse and contextually relevant facial reactions from input speaker behaviors. The autoencoder compresses high-dimensional input features, capturing dynamic patterns in listener reactions while condensing complex input data into a concise latent representation, facilitating more expressive and contextually appropriate reaction synthesis. The diffusion-based conditional generator operates on the latent space generated by the autoencoder to predict realistic facial reactions in a non-autoregressive manner. This approach allows for generating diverse facial reactions that reflect subtle variations in conversational cues and emotional states. Experimental results demonstrate the effectiveness of our approach in achieving superior performance in dyadic reaction synthesis tasks compared to existing methods. 

---
# DeltaEdit: Enhancing Sequential Editing in Large Language Models by Controlling Superimposed Noise 

**Authors**: Ding Cao, Yuchen Cai, Rongxi Guo, Xuesong He, Guiquan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07899)  

**Abstract**: Sequential knowledge editing techniques aim to continuously update the knowledge in large language models at a low cost, preventing the models from generating outdated or incorrect information. However, existing sequential editing methods suffer from a significant decline in editing success rates after long-term editing. Through theoretical analysis and experiments, we identify that as the number of edits increases, the model's output increasingly deviates from the desired target, leading to a drop in editing success rates. We refer to this issue as the accumulation of superimposed noise problem. To address this, we identify the factors contributing to this deviation and propose DeltaEdit, a novel method that optimizes update parameters through a dynamic orthogonal constraints strategy, effectively reducing interference between edits to mitigate deviation. Experimental results demonstrate that DeltaEdit significantly outperforms existing methods in edit success rates and the retention of generalization capabilities, ensuring stable and reliable model performance even under extensive sequential editing. 

---
# LongCodeBench: Evaluating Coding LLMs at 1M Context Windows 

**Authors**: Stefano Rando, Luca Romani, Alessio Sampieri, Yuta Kyuragi, Luca Franco, Fabio Galasso, Tatsunori Hashimoto, John Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07897)  

**Abstract**: Context lengths for models have grown rapidly, from thousands to millions of tokens in just a few years. The extreme context sizes of modern long-context models have made it difficult to construct realistic long-context benchmarks -- not only due to the cost of collecting million-context tasks but also in identifying realistic scenarios that require significant contexts. We identify code comprehension and repair as a natural testbed and challenge task for long-context models and introduce LongCodeBench (LCB), a benchmark to test LLM coding abilities in long-context scenarios. Our benchmark tests both the comprehension and repair capabilities of LCLMs in realistic and important settings by drawing from real-world GitHub issues and constructing QA (LongCodeQA) and bug fixing (LongSWE-Bench) tasks. We carefully stratify the complexity of our benchmark, enabling us to evaluate models across different scales -- ranging from Qwen2.5 14B Instruct to Google's flagship Gemini model. We find that long-context remains a weakness for all models, with performance drops such as from 29% to 3% for Claude 3.5 Sonnet, or from 70.2% to 40% for Qwen2.5. 

---
# Bridging Large Language Models and Single-Cell Transcriptomics in Dissecting Selective Motor Neuron Vulnerability 

**Authors**: Douglas Jiang, Zilin Dai, Luxuan Zhang, Qiyi Yu, Haoqi Sun, Feng Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.07896)  

**Abstract**: Understanding cell identity and function through single-cell level sequencing data remains a key challenge in computational biology. We present a novel framework that leverages gene-specific textual annotations from the NCBI Gene database to generate biologically contextualized cell embeddings. For each cell in a single-cell RNA sequencing (scRNA-seq) dataset, we rank genes by expression level, retrieve their NCBI Gene descriptions, and transform these descriptions into vector embedding representations using large language models (LLMs). The models used include OpenAI text-embedding-ada-002, text-embedding-3-small, and text-embedding-3-large (Jan 2024), as well as domain-specific models BioBERT and SciBERT. Embeddings are computed via an expression-weighted average across the top N most highly expressed genes in each cell, providing a compact, semantically rich representation. This multimodal strategy bridges structured biological data with state-of-the-art language modeling, enabling more interpretable downstream applications such as cell-type clustering, cell vulnerability dissection, and trajectory inference. 

---
# Representation Learning with Mutual Influence of Modalities for Node Classification in Multi-Modal Heterogeneous Networks 

**Authors**: Jiafan Li, Jiaqi Zhu, Liang Chang, Yilin Li, Miaomiao Li, Yang Wang, Hongan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07895)  

**Abstract**: Nowadays, numerous online platforms can be described as multi-modal heterogeneous networks (MMHNs), such as Douban's movie networks and Amazon's product review networks. Accurately categorizing nodes within these networks is crucial for analyzing the corresponding entities, which requires effective representation learning on nodes. However, existing multi-modal fusion methods often adopt either early fusion strategies which may lose the unique characteristics of individual modalities, or late fusion approaches overlooking the cross-modal guidance in GNN-based information propagation. In this paper, we propose a novel model for node classification in MMHNs, named Heterogeneous Graph Neural Network with Inter-Modal Attention (HGNN-IMA). It learns node representations by capturing the mutual influence of multiple modalities during the information propagation process, within the framework of heterogeneous graph transformer. Specifically, a nested inter-modal attention mechanism is integrated into the inter-node attention to achieve adaptive multi-modal fusion, and modality alignment is also taken into account to encourage the propagation among nodes with consistent similarities across all modalities. Moreover, an attention loss is augmented to mitigate the impact of missing modalities. Extensive experiments validate the superiority of the model in the node classification task, providing an innovative view to handle multi-modal data, especially when accompanied with network structures. 

---
# TrumorGPT: Graph-Based Retrieval-Augmented Large Language Model for Fact-Checking 

**Authors**: Ching Nam Hang, Pei-Duo Yu, Chee Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.07891)  

**Abstract**: In the age of social media, the rapid spread of misinformation and rumors has led to the emergence of infodemics, where false information poses a significant threat to society. To combat this issue, we introduce TrumorGPT , a novel generative artificial intelligence solution designed for fact-checking in the health domain. TrumorGPT aims to distinguish "trumors", which are health-related rumors that turn out to be true, providing a crucial tool in differentiating between mere speculation and verified facts. This framework leverages a large language model (LLM) with few-shot learning for semantic health knowledge graph construction and semantic reasoning. TrumorGPT incorporates graph-based retrieval-augmented generation (GraphRAG) to address the hallucination issue common in LLMs and the limitations of static training data. GraphRAG involves accessing and utilizing information from regularly updated semantic health knowledge graphs that consist of the latest medical news and health information, ensuring that fact-checking by TrumorGPT is based on the most recent data. Evaluating with extensive healthcare datasets, TrumorGPT demonstrates superior performance in fact-checking for public health claims. Its ability to effectively conduct fact-checking across various platforms marks a critical step forward in the fight against health-related misinformation, enhancing trust and accuracy in the digital information age. 

---
# Implementing Long Text Style Transfer with LLMs through Dual-Layered Sentence and Paragraph Structure Extraction and Mapping 

**Authors**: Yusen Wu, Xiaotie Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07888)  

**Abstract**: This paper addresses the challenge in long-text style transfer using zero-shot learning of large language models (LLMs), proposing a hierarchical framework that combines sentence-level stylistic adaptation with paragraph-level structural coherence. We argue that in the process of effective paragraph-style transfer, to preserve the consistency of original syntactic and semantic information, it is essential to perform style transfer not only at the sentence level but also to incorporate paragraph-level semantic considerations, while ensuring structural coherence across inter-sentential relationships. Our proposed framework, ZeroStylus, operates through two systematic phases: hierarchical template acquisition from reference texts and template-guided generation with multi-granular matching. The framework dynamically constructs sentence and paragraph template repositories, enabling context-aware transformations while preserving inter-sentence logical relationships. Experimental evaluations demonstrate significant improvements over baseline methods, with structured rewriting achieving 6.90 average score compared to 6.70 for direct prompting approaches in tri-axial metrics assessing style consistency, content preservation, and expression quality. Ablation studies validate the necessity of both template hierarchies during style transfer, showing higher content preservation win rate against sentence-only approaches through paragraph-level structural encoding, as well as direct prompting method through sentence-level pattern extraction and matching. The results establish new capabilities for coherent long-text style transfer without requiring parallel corpora or LLM fine-tuning. 

---
# PLHF: Prompt Optimization with Few-Shot Human Feedback 

**Authors**: Chun-Pai Yang, Kan Zheng, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07886)  

**Abstract**: Automatic prompt optimization frameworks are developed to obtain suitable prompts for large language models (LLMs) with respect to desired output quality metrics. Although existing approaches can handle conventional tasks such as fixed-solution question answering, defining the metric becomes complicated when the output quality cannot be easily assessed by comparisons with standard golden samples. Consequently, optimizing the prompts effectively and efficiently without a clear metric becomes a critical challenge. To address the issue, we present PLHF (which stands for "P"rompt "L"earning with "H"uman "F"eedback), a few-shot prompt optimization framework inspired by the well-known RLHF technique. Different from naive strategies, PLHF employs a specific evaluator module acting as the metric to estimate the output quality. PLHF requires only a single round of human feedback to complete the entire prompt optimization process. Empirical results on both public and industrial datasets show that PLHF outperforms prior output grading strategies for LLM prompt optimizations. 

---
# Recovering Event Probabilities from Large Language Model Embeddings via Axiomatic Constraints 

**Authors**: Jian-Qiao Zhu, Haijiang Yan, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2505.07883)  

**Abstract**: Rational decision-making under uncertainty requires coherent degrees of belief in events. However, event probabilities generated by Large Language Models (LLMs) have been shown to exhibit incoherence, violating the axioms of probability theory. This raises the question of whether coherent event probabilities can be recovered from the embeddings used by the models. If so, those derived probabilities could be used as more accurate estimates in events involving uncertainty. To explore this question, we propose enforcing axiomatic constraints, such as the additive rule of probability theory, in the latent space learned by an extended variational autoencoder (VAE) applied to LLM embeddings. This approach enables event probabilities to naturally emerge in the latent space as the VAE learns to both reconstruct the original embeddings and predict the embeddings of semantically related events. We evaluate our method on complementary events (i.e., event A and its complement, event not-A), where the true probabilities of the two events must sum to 1. Experiment results on open-weight language models demonstrate that probabilities recovered from embeddings exhibit greater coherence than those directly reported by the corresponding models and align closely with the true probabilities. 

---
# OMGM: Orchestrate Multiple Granularities and Modalities for Efficient Multimodal Retrieval 

**Authors**: Wei Yang, Jingjing Fu, Rui Wang, Jinyu Wang, Lei Song, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2505.07879)  

**Abstract**: Vision-language retrieval-augmented generation (RAG) has become an effective approach for tackling Knowledge-Based Visual Question Answering (KB-VQA), which requires external knowledge beyond the visual content presented in images. The effectiveness of Vision-language RAG systems hinges on multimodal retrieval, which is inherently challenging due to the diverse modalities and knowledge granularities in both queries and knowledge bases. Existing methods have not fully tapped into the potential interplay between these elements. We propose a multimodal RAG system featuring a coarse-to-fine, multi-step retrieval that harmonizes multiple granularities and modalities to enhance efficacy. Our system begins with a broad initial search aligning knowledge granularity for cross-modal retrieval, followed by a multimodal fusion reranking to capture the nuanced multimodal information for top entity selection. A text reranker then filters out the most relevant fine-grained section for augmented generation. Extensive experiments on the InfoSeek and Encyclopedic-VQA benchmarks show our method achieves state-of-the-art retrieval performance and highly competitive answering results, underscoring its effectiveness in advancing KB-VQA systems. 

---
# Efficient Telecom Specific LLM: TSLAM-Mini with QLoRA and Digital Twin Data 

**Authors**: Vignesh Ethiraj, Divya Vijay, Sidhanth Menon, Heblin Berscilla  

**Link**: [PDF](https://arxiv.org/pdf/2505.07877)  

**Abstract**: General-purpose large language models (LLMs), despite their broad capabilities accrued from open-world data, frequently exhibit suboptimal performance when confronted with the nuanced and specialized demands inherent in real-time telecommunications applications. This investigation addresses this critical limitation through the meticulous fine-tuning of TSLAM-Mini developed by NetoAI, a compact (3.8-billion parameter) causal language model architecturally derived from Phi-4 Mini Instruct 4B. The fine-tuning regimen leverages a bespoke dataset comprising 100,000 samples, strategically engineered to address 20 pivotal telecommunications use-cases, encompassing domains such as Network Fundamentals, IP Routing, MPLS, Network Security, Automation, OSS/BSS, RAN, Mobile Core, Satellite Communications, and Ethical AI. This dataset was curated utilizing NetoAI's DigiTwin platform, enriched with granular insights from venerated network Subject Matter Experts (SMEs) and authoritative RFC documents, thereby capturing high-fidelity representations of real-world network dynamics through simulations inspired by digital twin paradigms. Employing Quantized Low-Rank Adaptation (QLoRA), a state-of-the-art Parameter Efficient Fine-Tuning (PEFT) technique, we achieved substantial training efficiency and enabled prospective deployment on resource-constrained hardware. A novel evaluation framework, predicated on a high-capacity LLM (Qwen3-235B-A22B) functioning as an automated adjudicator, was instituted to rigorously assess instruction-following fidelity and response quality across the specified telecom use-cases. Empirical results unequivocally demonstrate TSLAM-Mini's superior aptitude in telecom-centric applications, underscoring the profound efficacy of domain-specific datasets and PEFT methodologies for advancing intelligent network management. 

---
# Getting Ready for the EU AI Act in Healthcare. A call for Sustainable AI Development and Deployment 

**Authors**: John Brandt Brodersen, Ilaria Amelia Caggiano, Pedro Kringen, Vince Istvan Madai, Walter Osika, Giovanni Sartor, Ellen Svensson, Magnus Westerlund, Roberto V. Zicari  

**Link**: [PDF](https://arxiv.org/pdf/2505.07875)  

**Abstract**: Assessments of trustworthiness have become a cornerstone of responsible AI development. Especially in high-stakes fields like healthcare, aligning technical, evidence-based, and ethical practices with forthcoming legal requirements is increasingly urgent. We argue that developers and deployers of AI systems for the medical domain should be proactive and take steps to progressively ensure that such systems, both those currently in use and those being developed or planned, respect the requirements of the AI Act, which has come into force in August 2024. This is necessary if full and effective compliance is to be ensured when the most relevant provisions of the Act become effective (August 2026). The engagement with the AI Act cannot be viewed as a formalistic exercise. Compliance with the AI Act needs to be carried out through the proactive commitment to the ethical principles of trustworthy AI. These principles provide the background for the Act, which mentions them several times and connects them to the protection of public interest. They can be used to interpret and apply the Act's provisions and to identify good practices, increasing the validity and sustainability of AI systems over time. 

---
# Evaluating Financial Sentiment Analysis with Annotators Instruction Assisted Prompting: Enhancing Contextual Interpretation and Stock Prediction Accuracy 

**Authors**: A M Muntasir Rahman, Ajim Uddin, Guiling "Grace" Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07871)  

**Abstract**: Financial sentiment analysis (FSA) presents unique challenges to LLMs that surpass those in typical sentiment analysis due to the nuanced language used in financial contexts. The prowess of these models is often undermined by the inherent subjectivity of sentiment classifications in existing benchmark datasets like Financial Phrasebank. These datasets typically feature undefined sentiment classes that reflect the highly individualized perspectives of annotators, leading to significant variability in annotations. This variability results in an unfair expectation for LLMs during benchmarking, where they are tasked to conjecture the subjective viewpoints of human annotators without sufficient context. In this paper, we introduce the Annotators' Instruction Assisted Prompt, a novel evaluation prompt designed to redefine the task definition of FSA for LLMs. By integrating detailed task instructions originally intended for human annotators into the LLMs' prompt framework, AIAP aims to standardize the understanding of sentiment across both human and machine interpretations, providing a fair and context-rich foundation for sentiment analysis. We utilize a new dataset, WSBS, derived from the WallStreetBets subreddit to demonstrate how AIAP significantly enhances LLM performance by aligning machine operations with the refined task definitions. Experimental results demonstrate that AIAP enhances LLM performance significantly, with improvements up to 9.08. This context-aware approach not only yields incremental gains in performance but also introduces an innovative sentiment-indexing method utilizing model confidence scores. This method enhances stock price prediction models and extracts more value from the financial sentiment analysis, underscoring the significance of WSB as a critical source of financial text. Our research offers insights into both improving FSA through better evaluation methods. 

---
# Efficient Fairness Testing in Large Language Models: Prioritizing Metamorphic Relations for Bias Detection 

**Authors**: Suavis Giramata, Madhusudan Srinivasan, Venkat Naidu Gudivada, Upulee Kanewala  

**Link**: [PDF](https://arxiv.org/pdf/2505.07870)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in various applications, raising critical concerns about fairness and potential biases in their outputs. This paper explores the prioritization of metamorphic relations (MRs) in metamorphic testing as a strategy to efficiently detect fairness issues within LLMs. Given the exponential growth of possible test cases, exhaustive testing is impractical; therefore, prioritizing MRs based on their effectiveness in detecting fairness violations is crucial. We apply a sentence diversity-based approach to compute and rank MRs to optimize fault detection. Experimental results demonstrate that our proposed prioritization approach improves fault detection rates by 22% compared to random prioritization and 12% compared to distance-based prioritization, while reducing the time to the first failure by 15% and 8%, respectively. Furthermore, our approach performs within 5% of fault-based prioritization in effectiveness, while significantly reducing the computational cost associated with fault labeling. These results validate the effectiveness of diversity-based MR prioritization in enhancing fairness testing for LLMs. 

---
# Computationally Efficient Diffusion Models in Medical Imaging: A Comprehensive Review 

**Authors**: Abdullah, Tao Huang, Ickjai Lee, Euijoon Ahn  

**Link**: [PDF](https://arxiv.org/pdf/2505.07866)  

**Abstract**: The diffusion model has recently emerged as a potent approach in computer vision, demonstrating remarkable performances in the field of generative artificial intelligence. Capable of producing high-quality synthetic images, diffusion models have been successfully applied across a range of applications. However, a significant challenge remains with the high computational cost associated with training and generating these models. This study focuses on the efficiency and inference time of diffusion-based generative models, highlighting their applications in both natural and medical imaging. We present the most recent advances in diffusion models by categorizing them into three key models: the Denoising Diffusion Probabilistic Model (DDPM), the Latent Diffusion Model (LDM), and the Wavelet Diffusion Model (WDM). These models play a crucial role in medical imaging, where producing fast, reliable, and high-quality medical images is essential for accurate analysis of abnormalities and disease diagnosis. We first investigate the general framework of DDPM, LDM, and WDM and discuss the computational complexity gap filled by these models in natural and medical imaging. We then discuss the current limitations of these models as well as the opportunities and future research directions in medical imaging. 

---
# CellVerse: Do Large Language Models Really Understand Cell Biology? 

**Authors**: Fan Zhang, Tianyu Liu, Zhihong Zhu, Hao Wu, Haixin Wang, Donghao Zhou, Yefeng Zheng, Kun Wang, Xian Wu, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07865)  

**Abstract**: Recent studies have demonstrated the feasibility of modeling single-cell data as natural languages and the potential of leveraging powerful large language models (LLMs) for understanding cell biology. However, a comprehensive evaluation of LLMs' performance on language-driven single-cell analysis tasks still remains unexplored. Motivated by this challenge, we introduce CellVerse, a unified language-centric question-answering benchmark that integrates four types of single-cell multi-omics data and encompasses three hierarchical levels of single-cell analysis tasks: cell type annotation (cell-level), drug response prediction (drug-level), and perturbation analysis (gene-level). Going beyond this, we systematically evaluate the performance across 14 open-source and closed-source LLMs ranging from 160M to 671B on CellVerse. Remarkably, the experimental results reveal: (1) Existing specialist models (C2S-Pythia) fail to make reasonable decisions across all sub-tasks within CellVerse, while generalist models such as Qwen, Llama, GPT, and DeepSeek family models exhibit preliminary understanding capabilities within the realm of cell biology. (2) The performance of current LLMs falls short of expectations and has substantial room for improvement. Notably, in the widely studied drug response prediction task, none of the evaluated LLMs demonstrate significant performance improvement over random guessing. CellVerse offers the first large-scale empirical demonstration that significant challenges still remain in applying LLMs to cell biology. By introducing CellVerse, we lay the foundation for advancing cell biology through natural languages and hope this paradigm could facilitate next-generation single-cell analysis. 

---
# Scalable LLM Math Reasoning Acceleration with Low-rank Distillation 

**Authors**: Harry Dong, Bilge Acun, Beidi Chen, Yuejie Chi  

**Link**: [PDF](https://arxiv.org/pdf/2505.07861)  

**Abstract**: Due to long generations, large language model (LLM) math reasoning demands significant computational resources and time. While many existing efficient inference methods have been developed with excellent performance preservation on language tasks, they often severely degrade math performance. In this paper, we propose Caprese, a low-cost distillation method to recover lost capabilities from deploying efficient inference methods, focused primarily in feedforward blocks. With original weights unperturbed, roughly 1% of additional parameters, and only 20K synthetic training samples, we are able to recover much if not all of the math capabilities lost from efficient inference for thinking LLMs and without harm to language tasks for instruct LLMs. Moreover, Caprese slashes the number of active parameters (~2B cut for Gemma 2 9B and Llama 3.1 8B) and integrates cleanly into existing model layers to reduce latency (>11% reduction to generate 2048 tokens with Qwen 2.5 14B) while encouraging response brevity. 

---
# Boosting Performance on ARC is a Matter of Perspective 

**Authors**: Daniel Franzen, Jan Disselhoff, David Hartmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.07859)  

**Abstract**: The Abstraction and Reasoning Corpus (ARC-AGI) poses a significant challenge for large language models (LLMs), exposing limitations in their abstract reasoning abilities. In this work, we leverage task-specific data augmentations throughout the training, generation, and scoring phases, and employ a depth-first search algorithm to generate diverse, high-probability candidate solutions. Furthermore, we utilize the LLM not only as a generator but also as a scorer, using its output probabilities to select the most promising solutions. Our method achieves a score of 71.6% (286.5/400 solved tasks) on the public ARC-AGI evaluation set, demonstrating state-of-the-art performance among publicly available approaches. While concurrent closed-source work has reported higher scores, our method distinguishes itself through its transparency, reproducibility, and remarkably low inference cost, averaging only around 2ct per task on readily available hardware (we assume a price of 36ct/hour for a Nvidia 4090 GPU). 

---
# Scaling Laws for Speculative Decoding 

**Authors**: Siyuan Yan, Mo Zhu, Guo-qing Jiang, Jianfei Wang, Jiaxing Chen, Wentai Zhang, Xiang Liao, Xiao Cui, Chen Zhang, Zhuoran Song, Ran Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07858)  

**Abstract**: The escalating demand for efficient decoding in large language models (LLMs) is particularly critical for reasoning-intensive architectures like OpenAI-o3 and DeepSeek-R1, which depend on extended chain-of-thought reasoning. This study investigates speculative decoding techniques through dense LLM architectures to establish foundational insights for accelerating reasoning tasks. While speculative decoding methods leveraging parallel draft-verification cycles have emerged as promising acceleration techniques, the scaling laws governing decoding efficiency remain under-explored compared to conventional backbone LLMs developed through Pretraining->SFT->RLHF training paradigms. In this work, we discover Log-linear Scaling Laws (Theorem 1.1, 1.2 and 1.3) governing draft model acceptance rate (or decoding speed) across three dimensions: pretraining token volume, draft model capacity, and decoding batch size. Building on these laws, we achieve Scylla, which coordinates multi-dimensional scaling for popular LLMs (Llama2/3, Qwen2.5). Empirical validation shows Scylla achieves 1.5-2.2 higher acceptance rate than EAGLE2 and 0.3 higher than EAGLE3 at temperature T = 0, with peak performance gains on summarization and QA tasks (Figure 2). Industrial inference engine deployments demonstrate 2X decoding throughput improvements over EAGLE2 (Table 5), validating the transformative potential of systematic scaling for efficient LLM inference. Code will be released later. 

---
# Enhanced Urdu Intent Detection with Large Language Models and Prototype-Informed Predictive Pipelines 

**Authors**: Faiza Hassan, Summra Saleem, Kashif Javed, Muhammad Nabeel Asim, Abdur Rehman, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2505.07857)  

**Abstract**: Multifarious intent detection predictors are developed for different languages, including English, Chinese and French, however, the field remains underdeveloped for Urdu, the 10th most spoken language. In the realm of well-known languages, intent detection predictors utilize the strategy of few-shot learning and prediction of unseen classes based on the model training on seen classes. However, Urdu language lacks few-shot strategy based intent detection predictors and traditional predictors are focused on prediction of the same classes which models have seen in the train set. To empower Urdu language specific intent detection, this introduces a unique contrastive learning approach that leverages unlabeled Urdu data to re-train pre-trained language models. This re-training empowers LLMs representation learning for the downstream intent detection task. Finally, it reaps the combined potential of pre-trained LLMs and the prototype-informed attention mechanism to create a comprehensive end-to-end LLMPIA intent detection pipeline. Under the paradigm of proposed predictive pipeline, it explores the potential of 6 distinct language models and 13 distinct similarity computation methods. The proposed framework is evaluated on 2 public benchmark datasets, namely ATIS encompassing 5836 samples and Web Queries having 8519 samples. Across ATIS dataset under 4-way 1 shot and 4-way 5 shot experimental settings LLMPIA achieved 83.28% and 98.25% F1-Score and on Web Queries dataset produced 76.23% and 84.42% F1-Score, respectively. In an additional case study on the Web Queries dataset under same classes train and test set settings, LLMPIA outperformed state-of-the-art predictor by 53.55% F1-Score. 

---
# Unpacking Robustness in Inflectional Languages: Adversarial Evaluation and Mechanistic Insights 

**Authors**: Paweł Walkowiak, Marek Klonowski, Marcin Oleksy, Arkadiusz Janz  

**Link**: [PDF](https://arxiv.org/pdf/2505.07856)  

**Abstract**: Various techniques are used in the generation of adversarial examples, including methods such as TextBugger which introduce minor, hardly visible perturbations to words leading to changes in model behaviour. Another class of techniques involves substituting words with their synonyms in a way that preserves the text's meaning but alters its predicted class, with TextFooler being a prominent example of such attacks. Most adversarial example generation methods are developed and evaluated primarily on non-inflectional languages, typically English. In this work, we evaluate and explain how adversarial attacks perform in inflectional languages. To explain the impact of inflection on model behaviour and its robustness under attack, we designed a novel protocol inspired by mechanistic interpretability, based on Edge Attribution Patching (EAP) method. The proposed evaluation protocol relies on parallel task-specific corpora that include both inflected and syncretic variants of texts in two languages -- Polish and English. To analyse the models and explain the relationship between inflection and adversarial robustness, we create a new benchmark based on task-oriented dataset MultiEmo, enabling the identification of mechanistic inflection-related elements of circuits within the model and analyse their behaviour under attack. 

---
# CrashSage: A Large Language Model-Centered Framework for Contextual and Interpretable Traffic Crash Analysis 

**Authors**: Hao Zhen, Jidong J. Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.07853)  

**Abstract**: Road crashes claim over 1.3 million lives annually worldwide and incur global economic losses exceeding \$1.8 trillion. Such profound societal and financial impacts underscore the urgent need for road safety research that uncovers crash mechanisms and delivers actionable insights. Conventional statistical models and tree ensemble approaches typically rely on structured crash data, overlooking contextual nuances and struggling to capture complex relationships and underlying semantics. Moreover, these approaches tend to incur significant information loss, particularly in narrative elements related to multi-vehicle interactions, crash progression, and rare event characteristics. This study presents CrashSage, a novel Large Language Model (LLM)-centered framework designed to advance crash analysis and modeling through four key innovations. First, we introduce a tabular-to-text transformation strategy paired with relational data integration schema, enabling the conversion of raw, heterogeneous crash data into enriched, structured textual narratives that retain essential structural and relational context. Second, we apply context-aware data augmentation using a base LLM model to improve narrative coherence while preserving factual integrity. Third, we fine-tune the LLaMA3-8B model for crash severity inference, demonstrating superior performance over baseline approaches, including zero-shot, zero-shot with chain-of-thought prompting, and few-shot learning, with multiple models (GPT-4o, GPT-4o-mini, LLaMA3-70B). Finally, we employ a gradient-based explainability technique to elucidate model decisions at both the individual crash level and across broader risk factor dimensions. This interpretability mechanism enhances transparency and enables targeted road safety interventions by providing deeper insights into the most influential factors. 

---
# Joint Detection of Fraud and Concept Drift inOnline Conversations with LLM-Assisted Judgment 

**Authors**: Ali Senol, Garima Agrawal, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.07852)  

**Abstract**: Detecting fake interactions in digital communication platforms remains a challenging and insufficiently addressed problem. These interactions may appear as harmless spam or escalate into sophisticated scam attempts, making it difficult to flag malicious intent early. Traditional detection methods often rely on static anomaly detection techniques that fail to adapt to dynamic conversational shifts. One key limitation is the misinterpretation of benign topic transitions referred to as concept drift as fraudulent behavior, leading to either false alarms or missed threats. We propose a two stage detection framework that first identifies suspicious conversations using a tailored ensemble classification model. To improve the reliability of detection, we incorporate a concept drift analysis step using a One Class Drift Detector (OCDD) to isolate conversational shifts within flagged dialogues. When drift is detected, a large language model (LLM) assesses whether the shift indicates fraudulent manipulation or a legitimate topic change. In cases where no drift is found, the behavior is inferred to be spam like. We validate our framework using a dataset of social engineering chat scenarios and demonstrate its practical advantages in improving both accuracy and interpretability for real time fraud detection. To contextualize the trade offs, we compare our modular approach against a Dual LLM baseline that performs detection and judgment using different language models. 

---
# Pose Estimation for Intra-cardiac Echocardiography Catheter via AI-Based Anatomical Understanding 

**Authors**: Jaeyoung Huh, Ankur Kapoor, Young-Ho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.07851)  

**Abstract**: Intra-cardiac Echocardiography (ICE) plays a crucial role in Electrophysiology (EP) and Structural Heart Disease (SHD) interventions by providing high-resolution, real-time imaging of cardiac structures. However, existing navigation methods rely on electromagnetic (EM) tracking, which is susceptible to interference and position drift, or require manual adjustments based on operator expertise. To overcome these limitations, we propose a novel anatomy-aware pose estimation system that determines the ICE catheter position and orientation solely from ICE images, eliminating the need for external tracking sensors. Our approach leverages a Vision Transformer (ViT)-based deep learning model, which captures spatial relationships between ICE images and anatomical structures. The model is trained on a clinically acquired dataset of 851 subjects, including ICE images paired with position and orientation labels normalized to the left atrium (LA) mesh. ICE images are patchified into 16x16 embeddings and processed through a transformer network, where a [CLS] token independently predicts position and orientation via separate linear layers. The model is optimized using a Mean Squared Error (MSE) loss function, balancing positional and orientational accuracy. Experimental results demonstrate an average positional error of 9.48 mm and orientation errors of (16.13 deg, 8.98 deg, 10.47 deg) across x, y, and z axes, confirming the model accuracy. Qualitative assessments further validate alignment between predicted and target views within 3D cardiac meshes. This AI-driven system enhances procedural efficiency, reduces operator workload, and enables real-time ICE catheter localization for tracking-free procedures. The proposed method can function independently or complement existing mapping systems like CARTO, offering a transformative approach to ICE-guided interventions. 

---
# A Tale of Two Identities: An Ethical Audit of Human and AI-Crafted Personas 

**Authors**: Pranav Narayanan Venkit, Jiayi Li, Yingfan Zhou, Sarah Rajtmajer, Shomir Wilson  

**Link**: [PDF](https://arxiv.org/pdf/2505.07850)  

**Abstract**: As LLMs (large language models) are increasingly used to generate synthetic personas particularly in data-limited domains such as health, privacy, and HCI, it becomes necessary to understand how these narratives represent identity, especially that of minority communities. In this paper, we audit synthetic personas generated by 3 LLMs (GPT4o, Gemini 1.5 Pro, Deepseek 2.5) through the lens of representational harm, focusing specifically on racial identity. Using a mixed methods approach combining close reading, lexical analysis, and a parameterized creativity framework, we compare 1512 LLM generated personas to human-authored responses. Our findings reveal that LLMs disproportionately foreground racial markers, overproduce culturally coded language, and construct personas that are syntactically elaborate yet narratively reductive. These patterns result in a range of sociotechnical harms, including stereotyping, exoticism, erasure, and benevolent bias, that are often obfuscated by superficially positive narrations. We formalize this phenomenon as algorithmic othering, where minoritized identities are rendered hypervisible but less authentic. Based on these findings, we offer design recommendations for narrative-aware evaluation metrics and community-centered validation protocols for synthetic identity generation. 

---
# SweRank: Software Issue Localization with Code Ranking 

**Authors**: Revanth Gangi Reddy, Tarun Suresh, JaeHyeok Doo, Ye Liu, Xuan Phi Nguyen, Yingbo Zhou, Semih Yavuz, Caiming Xiong, Heng Ji, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2505.07849)  

**Abstract**: Software issue localization, the task of identifying the precise code locations (files, classes, or functions) relevant to a natural language issue description (e.g., bug report, feature request), is a critical yet time-consuming aspect of software development. While recent LLM-based agentic approaches demonstrate promise, they often incur significant latency and cost due to complex multi-step reasoning and relying on closed-source LLMs. Alternatively, traditional code ranking models, typically optimized for query-to-code or code-to-code retrieval, struggle with the verbose and failure-descriptive nature of issue localization queries. To bridge this gap, we introduce SweRank, an efficient and effective retrieve-and-rerank framework for software issue localization. To facilitate training, we construct SweLoc, a large-scale dataset curated from public GitHub repositories, featuring real-world issue descriptions paired with corresponding code modifications. Empirical results on SWE-Bench-Lite and LocBench show that SweRank achieves state-of-the-art performance, outperforming both prior ranking models and costly agent-based systems using closed-source LLMs like Claude-3.5. Further, we demonstrate SweLoc's utility in enhancing various existing retriever and reranker models for issue localization, establishing the dataset as a valuable resource for the community. 

---
# Sub-diffraction terahertz backpropagation compressive imaging 

**Authors**: Yongsheng Zhu, Shaojing Liu, Ximiao Wang, Runli Li, Haili Yang, Jiali Wang, Hongjia Zhu, Yanlin Ke, Ningsheng Xu, Huanjun Chen, Shaozhi Deng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07839)  

**Abstract**: Terahertz single-pixel imaging (TSPI) has garnered significant attention due to its simplicity and cost-effectiveness. However, the relatively long wavelength of THz waves limits sub-diffraction-scale imaging resolution. Although TSPI technique can achieve sub-wavelength resolution, it requires harsh experimental conditions and time-consuming processes. Here, we propose a sub-diffraction THz backpropagation compressive imaging technique. We illuminate the object with monochromatic continuous-wave THz radiation. The transmitted THz wave is modulated by prearranged patterns generated on the back surface of a 500-{\mu}m-thick silicon wafer, realized through photoexcited carriers using a 532-nm laser. The modulated THz wave is then recorded by a single-element detector. An untrained neural network is employed to iteratively reconstruct the object image with an ultralow compression ratio of 1.5625% under a physical model constraint, thus reducing the long sampling times. To further suppress the diffraction-field effects, embedded with the angular spectrum propagation (ASP) theory to model the diffraction of THz waves during propagation, the network retrieves near-field information from the object, enabling sub-diffraction imaging with a spatial resolution of ~{\lambda}0/7 ({\lambda}0 = 833.3 {\mu}m at 0.36 THz) and eliminating the need for ultrathin photomodulators. This approach provides an efficient solution for advancing THz microscopic imaging and addressing other inverse imaging challenges. 

---
# Moving From Monolithic To Microservices Architecture for Multi-Agent Systems 

**Authors**: Muskaan Goyal, Pranav Bhasin  

**Link**: [PDF](https://arxiv.org/pdf/2505.07838)  

**Abstract**: The transition from monolithic to microservices architecture revolutionized software development by improving scalability and maintainability. This paradigm shift is now becoming relevant for complex multi-agent systems (MAS). This review article explores the evolution from monolithic architecture to microservices architecture in the specific context of MAS. It will highlight the limitations of traditional monolithic MAS and the benefits of adopting a microservices-based approach. The article further examines the core architectural principles and communication protocols, including Agent Communication Languages (ACLs), the Model Context Protocol (MCP), and the Application-to-Application (A2A) protocol. The article identifies emerging architectural patterns, design challenges, and considerations through a comparative lens of the paradigm shift. 

---
# Intelligent Product 3.0: Decentralised AI Agents and Web3 Intelligence Standards 

**Authors**: Alex C. Y. Wong, Duncan McFarlane, C. Ellarby, M. Lee, M. Kuok  

**Link**: [PDF](https://arxiv.org/pdf/2505.07835)  

**Abstract**: Twenty-five years ago, the specification of the Intelligent Product was established, envisaging real-time connectivity that not only enables products to gather accurate data about themselves but also allows them to assess and influence their own destiny. Early work by the Auto-ID project focused on creating a single, open-standard repository for storing and retrieving product information, laying a foundation for scalable connectivity. A decade later, the approach was revisited in light of low-cost RFID systems that promised a low-cost link between physical goods and networked information environments. Since then, advances in blockchain, Web3, and artificial intelligence have introduced unprecedented levels of resilience, consensus, and autonomy. By leveraging decentralised identity, blockchain-based product information and history, and intelligent AI-to-AI collaboration, this paper examines these developments and outlines a new specification for the Intelligent Product 3.0, illustrating how decentralised and AI-driven capabilities facilitate seamless interaction between physical AI and everyday products. 

---
# ai.txt: A Domain-Specific Language for Guiding AI Interactions with the Internet 

**Authors**: Yuekang Li, Wei Song, Bangshuo Zhu, Dong Gong, Yi Liu, Gelei Deng, Chunyang Chen, Lei Ma, Jun Sun, Toby Walsh, Jingling Xue  

**Link**: [PDF](https://arxiv.org/pdf/2505.07834)  

**Abstract**: We introduce this http URL, a novel domain-specific language (DSL) designed to explicitly regulate interactions between AI models, agents, and web content, addressing critical limitations of the widely adopted this http URL standard. As AI increasingly engages with online materials for tasks such as training, summarization, and content modification, existing regulatory methods lack the necessary granularity and semantic expressiveness to ensure ethical and legal compliance. this http URL extends traditional URL-based access controls by enabling precise element-level regulations and incorporating natural language instructions interpretable by AI systems. To facilitate practical deployment, we provide an integrated development environment with code autocompletion and automatic XML generation. Furthermore, we propose two compliance mechanisms: XML-based programmatic enforcement and natural language prompt integration, and demonstrate their effectiveness through preliminary experiments and case studies. Our approach aims to aid the governance of AI-Internet interactions, promoting responsible AI use in digital ecosystems. 

---
# Patchwork: A Unified Framework for RAG Serving 

**Authors**: Bodun Hu, Luis Pabon, Saurabh Agarwal, Aditya Akella  

**Link**: [PDF](https://arxiv.org/pdf/2505.07833)  

**Abstract**: Retrieval Augmented Generation (RAG) has emerged as a new paradigm for enhancing Large Language Model reliability through integration with external knowledge sources. However, efficient deployment of these systems presents significant technical challenges due to their inherently heterogeneous computational pipelines comprising LLMs, databases, and specialized processing components. We introduce Patchwork, a comprehensive end-to-end RAG serving framework designed to address these efficiency bottlenecks. Patchwork's architecture offers three key innovations: First, it provides a flexible specification interface enabling users to implement custom RAG pipelines. Secondly, it deploys these pipelines as distributed inference systems while optimizing for the unique scalability characteristics of individual RAG components. Third, Patchwork incorporates an online scheduling mechanism that continuously monitors request load and execution progress, dynamically minimizing SLO violations through strategic request prioritization and resource auto-scaling. Our experimental evaluation across four distinct RAG implementations demonstrates that Patchwork delivers substantial performance improvements over commercial alternatives, achieving throughput gains exceeding 48% while simultaneously reducing SLO violations by ~24%. 

---
# A General Approach of Automated Environment Design for Learning the Optimal Power Flow 

**Authors**: Thomas Wolgast, Astrid Nieße  

**Link**: [PDF](https://arxiv.org/pdf/2505.07832)  

**Abstract**: Reinforcement learning (RL) algorithms are increasingly used to solve the optimal power flow (OPF) problem. Yet, the question of how to design RL environments to maximize training performance remains unanswered, both for the OPF and the general case. We propose a general approach for automated RL environment design by utilizing multi-objective optimization. For that, we use the hyperparameter optimization (HPO) framework, which allows the reuse of existing HPO algorithms and methods. On five OPF benchmark problems, we demonstrate that our automated design approach consistently outperforms a manually created baseline environment design. Further, we use statistical analyses to determine which environment design decisions are especially important for performance, resulting in multiple novel insights on how RL-OPF environments should be designed. Finally, we discuss the risk of overfitting the environment to the utilized RL algorithm. To the best of our knowledge, this is the first general approach for automated RL environment design. 

---
# Polysemy of Synthetic Neurons Towards a New Type of Explanatory Categorical Vector Spaces 

**Authors**: Michael Pichat, William Pogrund, Paloma Pichat, Judicael Poumay, Armanouche Gasparian, Samuel Demarchi, Martin Corbet, Alois Georgeon, Michael Veillet-Guillem  

**Link**: [PDF](https://arxiv.org/pdf/2505.07831)  

**Abstract**: The polysemantic nature of synthetic neurons in artificial intelligence language models is currently understood as the result of a necessary superposition of distributed features within the latent space. We propose an alternative approach, geometrically defining a neuron in layer n as a categorical vector space with a non-orthogonal basis, composed of categorical sub-dimensions extracted from preceding neurons in layer n-1. This categorical vector space is structured by the activation space of each neuron and enables, via an intra-neuronal attention process, the identification and utilization of a critical categorical zone for the efficiency of the language model - more homogeneous and located at the intersection of these different categorical sub-dimensions. 

---
# Blockbuster, Part 1: Block-level AI Operator Fusion 

**Authors**: Ofer Dekel  

**Link**: [PDF](https://arxiv.org/pdf/2505.07829)  

**Abstract**: Blockbuster is a framework for AI operator fusion in inference programs. The Blockbuster framework is compatible with any multiprocessor architecture that has a tiered memory hierarchy, including GPUs, multi-core CPUs, and some AI accelerator chips. It includes a graph-based representation for AI workloads, called a block program, which explicitly models how blocks of data move between the memory tiers. It also includes an operator fusion procedure, which is made up of a candidate selection algorithm and a fusion algorithm that fuses each individual candidate - this two-algorithm structure makes Blockbuster especially suitable for large AI programs. The current paper focuses on the fusion algorithm, which is a rule-based technique. While the literature is full of previous rule-based fusion algorithms, what sets our algorithm apart is its direct modeling of data movement between memory tiers, resulting in uniquely powerful fusion results. As a first sanity check, we demonstrate how our algorithm automatically rediscovers the well-known Flash Attention kernel. Then, we demonstrate the real power of our approach by fusing LayerNorm with matrix multiplication and RMSNorm with FNN-SwiGLU - the latter involves fusing three matrix multiplications, a Hadamard product, a reduction, and a few elementwise operations into a single mega-kernel. 

---
# AI-Based Crypto Tokens: The Illusion of Decentralized AI? 

**Authors**: Rischan Mafrur  

**Link**: [PDF](https://arxiv.org/pdf/2505.07828)  

**Abstract**: The convergence of blockchain and artificial intelligence (AI) has led to the emergence of AI-based tokens, which are cryptographic assets designed to power decentralized AI platforms and services. This paper provides a comprehensive review of leading AI-token projects, examining their technical architectures, token utilities, consensus mechanisms, and underlying business models. We explore how these tokens operate across various blockchain ecosystems and assess the extent to which they offer value beyond traditional centralized AI services. Based on this assessment, our analysis identifies several core limitations. From a technical perspective, many platforms depend extensively on off-chain computation, exhibit limited capabilities for on-chain intelligence, and encounter significant scalability challenges. From a business perspective, many models appear to replicate centralized AI service structures, simply adding token-based payment and governance layers without delivering truly novel value. In light of these challenges, we also examine emerging developments that may shape the next phase of decentralized AI systems. These include approaches for on-chain verification of AI outputs, blockchain-enabled federated learning, and more robust incentive frameworks. Collectively, while emerging innovations offer pathways to strengthen decentralized AI ecosystems, significant gaps remain between the promises and the realities of current AI-token implementations. Our findings contribute to a growing body of research at the intersection of AI and blockchain, highlighting the need for critical evaluation and more grounded approaches as the field continues to evolve. 

---
# Explainable Artificial Intelligence Techniques for Software Development Lifecycle: A Phase-specific Survey 

**Authors**: Lakshit Arora, Sanjay Surendranath Girija, Shashank Kapoor, Aman Raj, Dipen Pradhan, Ankit Shetgaonkar  

**Link**: [PDF](https://arxiv.org/pdf/2505.07058)  

**Abstract**: Artificial Intelligence (AI) is rapidly expanding and integrating more into daily life to automate tasks, guide decision making, and enhance efficiency. However, complex AI models, which make decisions without providing clear explanations (known as the "black-box problem"), currently restrict trust and widespread adoption of AI. Explainable Artificial Intelligence (XAI) has emerged to address the black-box problem of making AI systems more interpretable and transparent so stakeholders can trust, verify, and act upon AI-based outcomes. Researchers have developed various techniques to foster XAI in the Software Development Lifecycle. However, there are gaps in applying XAI techniques in the Software Engineering phases. Literature review shows that 68% of XAI in Software Engineering research is focused on maintenance as opposed to 8% on software management and requirements. In this paper, we present a comprehensive survey of the applications of XAI methods such as concept-based explanations, Local Interpretable Model-agnostic Explanations (LIME), SHapley Additive exPlanations (SHAP), rule extraction, attention mechanisms, counterfactual explanations, and example-based explanations to the different phases of the Software Development Life Cycle (SDLC), including requirements elicitation, design and development, testing and deployment, and evolution. To the best of our knowledge, this paper presents the first comprehensive survey of XAI techniques for every phase of the Software Development Life Cycle (SDLC). This survey aims to promote explainable AI in Software Engineering and facilitate the practical application of complex AI models in AI-driven software development. 

---
# Reinforcement Learning (RL) Meets Urban Climate Modeling: Investigating the Efficacy and Impacts of RL-Based HVAC Control 

**Authors**: Junjie Yu, John S. Schreck, David John Gagne, Keith W. Oleson, Jie Li, Yongtu Liang, Qi Liao, Mingfei Sun, David O. Topping, Zhonghua Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.07045)  

**Abstract**: Reinforcement learning (RL)-based heating, ventilation, and air conditioning (HVAC) control has emerged as a promising technology for reducing building energy consumption while maintaining indoor thermal comfort. However, the efficacy of such strategies is influenced by the background climate and their implementation may potentially alter both the indoor climate and local urban climate. This study proposes an integrated framework combining RL with an urban climate model that incorporates a building energy model, aiming to evaluate the efficacy of RL-based HVAC control across different background climates, impacts of RL strategies on indoor climate and local urban climate, and the transferability of RL strategies across cities. Our findings reveal that the reward (defined as a weighted combination of energy consumption and thermal comfort) and the impacts of RL strategies on indoor climate and local urban climate exhibit marked variability across cities with different background climates. The sensitivity of reward weights and the transferability of RL strategies are also strongly influenced by the background climate. Cities in hot climates tend to achieve higher rewards across most reward weight configurations that balance energy consumption and thermal comfort, and those cities with more varying atmospheric temperatures demonstrate greater RL strategy transferability. These findings underscore the importance of thoroughly evaluating RL-based HVAC control strategies in diverse climatic contexts. This study also provides a new insight that city-to-city learning will potentially aid the deployment of RL-based HVAC control. 

---
