# SafeWork-R1: Coevolving Safety and Intelligence under the AI-45$^{\circ}$ Law 

**Authors**: Shanghai AI Lab, Yicheng Bao, Guanxu Chen, Mingkang Chen, Yunhao Chen, Chiyu Chen, Lingjie Chen, Sirui Chen, Xinquan Chen, Jie Cheng, Yu Cheng, Dengke Deng, Yizhuo Ding, Dan Ding, Xiaoshan Ding, Yi Ding, Zhichen Dong, Lingxiao Du, Yuyu Fan, Xinshun Feng, Yanwei Fu, Yuxuan Gao, Ruijun Ge, Tianle Gu, Lujun Gui, Jiaxuan Guo, Qianxi He, Yuenan Hou, Xuhao Hu, Hong Huang, Kaichen Huang, Shiyang Huang, Yuxian Jiang, Shanzhe Lei, Jie Li, Lijun Li, Hao Li, Juncheng Li, Xiangtian Li, Yafu Li, Lingyu Li, Xueyan Li, Haotian Liang, Dongrui Liu, Qihua Liu, Zhixuan Liu, Bangwei Liu, Huacan Liu, Yuexiao Liu, Zongkai Liu, Chaochao Lu, Yudong Lu, Xiaoya Lu, Zhenghao Lu, Qitan Lv, Caoyuan Ma, Jiachen Ma, Xiaoya Ma, Zhongtian Ma, Lingyu Meng, Ziqi Miao, Yazhe Niu, Yuezhang Peng, Yuan Pu, Han Qi, Chen Qian, Xingge Qiao, Jingjing Qu, Jiashu Qu, Wanying Qu, Wenwen Qu, Xiaoye Qu, Qihan Ren, Qingnan Ren, Qingyu Ren, Jing Shao, Wenqi Shao, Shuai Shao, Dongxing Shi, Xin Song, Xinhao Song, Yan Teng, Xuan Tong, Yingchun Wang, Xuhong Wang, Shujie Wang, Xin Wang, Yige Wang, Yixu Wang, Yuanfu Wang, Futing Wang, Ruofan Wang, Wenjie Wang, Yajie Wang, Muhao Wei, Xiaoyu Wen, Fenghua Weng, Yuqi Wu, Yingtong Xiong, Xingcheng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18576)  

**Abstract**: We introduce SafeWork-R1, a cutting-edge multimodal reasoning model that demonstrates the coevolution of capabilities and safety. It is developed by our proposed SafeLadder framework, which incorporates large-scale, progressive, safety-oriented reinforcement learning post-training, supported by a suite of multi-principled verifiers. Unlike previous alignment methods such as RLHF that simply learn human preferences, SafeLadder enables SafeWork-R1 to develop intrinsic safety reasoning and self-reflection abilities, giving rise to safety `aha' moments. Notably, SafeWork-R1 achieves an average improvement of $46.54\%$ over its base model Qwen2.5-VL-72B on safety-related benchmarks without compromising general capabilities, and delivers state-of-the-art safety performance compared to leading proprietary models such as GPT-4.1 and Claude Opus 4. To further bolster its reliability, we implement two distinct inference-time intervention methods and a deliberative search mechanism, enforcing step-level verification. Finally, we further develop SafeWork-R1-InternVL3-78B, SafeWork-R1-DeepSeek-70B, and SafeWork-R1-Qwen2.5VL-7B. All resulting models demonstrate that safety and capability can co-evolve synergistically, highlighting the generalizability of our framework in building robust, reliable, and trustworthy general-purpose AI. 

---
# On the Performance of Concept Probing: The Influence of the Data (Extended Version) 

**Authors**: Manuel de Sousa Ribeiro, Afonso Leote, João Leite  

**Link**: [PDF](https://arxiv.org/pdf/2507.18550)  

**Abstract**: Concept probing has recently garnered increasing interest as a way to help interpret artificial neural networks, dealing both with their typically large size and their subsymbolic nature, which ultimately renders them unfeasible for direct human interpretation. Concept probing works by training additional classifiers to map the internal representations of a model into human-defined concepts of interest, thus allowing humans to peek inside artificial neural networks. Research on concept probing has mainly focused on the model being probed or the probing model itself, paying limited attention to the data required to train such probing models. In this paper, we address this gap. Focusing on concept probing in the context of image classification tasks, we investigate the effect of the data used to train probing models on their performance. We also make available concept labels for two widely used datasets. 

---
# GPU Accelerated Compact-Table Propagation 

**Authors**: Enrico Santi, Fabio Tardivo, Agostino Dovier, Andrea Formisano  

**Link**: [PDF](https://arxiv.org/pdf/2507.18413)  

**Abstract**: Constraint Programming developed within Logic Programming in the Eighties; nowadays all Prolog systems encompass modules capable of handling constraint programming on finite domains demanding their solution to a constraint solver. This work focuses on a specific form of constraint, the so-called table constraint, used to specify conditions on the values of variables as an enumeration of alternative options. Since every condition on a set of finite domain variables can be ultimately expressed as a finite set of cases, Table can, in principle, simulate any other constraint. These characteristics make Table one of the most studied constraints ever, leading to a series of increasingly efficient propagation algorithms. Despite this, it is not uncommon to encounter real-world problems with hundreds or thousands of valid cases that are simply too many to be handled effectively with standard CPU-based approaches. In this paper, we deal with the Compact-Table (CT) algorithm, the state-of-the-art propagation algorithms for Table. We describe how CT can be enhanced by exploiting the massive computational power offered by modern GPUs to handle large Table constraints. In particular, we report on the design and implementation of GPU-accelerated CT, on its integration into an existing constraint solver, and on an experimental validation performed on a significant set of instances. 

---
# Optimising Call Centre Operations using Reinforcement Learning: Value Iteration versus Proximal Policy Optimisation 

**Authors**: Kwong Ho Li, Wathsala Karunarathne  

**Link**: [PDF](https://arxiv.org/pdf/2507.18398)  

**Abstract**: This paper investigates the application of Reinforcement Learning (RL) to optimise call routing in call centres to minimise client waiting time and staff idle time. Two methods are compared: a model-based approach using Value Iteration (VI) under known system dynamics, and a model-free approach using Proximal Policy Optimisation (PPO) that learns from experience. For the model-based approach, a theoretical model is used, while a simulation model combining Discrete Event Simulation (DES) with the OpenAI Gym environment is developed for model-free learning. Both models frame the problem as a Markov Decision Process (MDP) within a Skills-Based Routing (SBR) framework, with Poisson client arrivals and exponentially distributed service and abandonment times. For policy evaluation, random, VI, and PPO policies are evaluated using the simulation model. After 1,000 test episodes, PPO consistently achives the highest rewards, along with the lowest client waiting time and staff idle time, despite requiring longer training time. 

---
# Revisiting LLM Reasoning via Information Bottleneck 

**Authors**: Shiye Lei, Zhihao Cheng, Kai Jia, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2507.18391)  

**Abstract**: Large language models (LLMs) have recently demonstrated remarkable progress in reasoning capabilities through reinforcement learning with verifiable rewards (RLVR). By leveraging simple rule-based rewards, RL effectively incentivizes LLMs to produce extended chain-of-thought (CoT) reasoning trajectories, progressively guiding them toward correct answers. However, existing approaches remain largely heuristic and intuition-driven, limiting the development of principled methodologies. In this paper, we present a theoretical characterization of LLM reasoning grounded in information bottleneck (IB) principle, introducing IB-aware reasoning optimization (IBRO), a framework that encourages reasoning trajectories to be both informative about the final correct answer and generalizable across diverse prompts. We derive a practical token-level surrogate objective and propose an efficient approximation, resulting in the lightweight IB regularization method. This technique integrates seamlessly into existing RL-based post-training frameworks without additional computational overhead, requiring only a one-line code modification. Empirically, we validate IB regularization across multiple mathematical reasoning benchmarks and RL algorithms, demonstrating consistent improvements in LLM reasoning performance. 

---
# Reasoning Beyond the Obvious: Evaluating Divergent and Convergent Thinking in LLMs for Financial Scenarios 

**Authors**: Zhuang Qiang Bok, Watson Wei Khong Chua  

**Link**: [PDF](https://arxiv.org/pdf/2507.18368)  

**Abstract**: Most reasoning benchmarks for LLMs emphasize factual accuracy or step-by-step logic. In finance, however, professionals must not only converge on optimal decisions but also generate creative, plausible futures under uncertainty. We introduce ConDiFi, a benchmark that jointly evaluates divergent and convergent thinking in LLMs for financial tasks.
ConDiFi features 607 macro-financial prompts for divergent reasoning and 990 multi-hop adversarial MCQs for convergent reasoning. Using this benchmark, we evaluated 14 leading models and uncovered striking differences. Despite high fluency, GPT-4o underperforms on Novelty and Actionability. In contrast, models like DeepSeek-R1 and Cohere Command R+ rank among the top for generating actionable, insights suitable for investment decisions. ConDiFi provides a new perspective to assess reasoning capabilities essential to safe and strategic deployment of LLMs in finance. 

---
# The AlphaPhysics Term Rewriting System for Marking Algebraic Expressions in Physics Exams 

**Authors**: Peter Baumgartner, Lachlan McGinness  

**Link**: [PDF](https://arxiv.org/pdf/2507.18337)  

**Abstract**: We present our method for automatically marking Physics exams. The marking problem consists in assessing typed student answers for correctness with respect to a ground truth solution. This is a challenging problem that we seek to tackle using a combination of a computer algebra system, an SMT solver and a term rewriting system. A Large Language Model is used to interpret and remove errors from student responses and rewrite these in a machine readable format. Once formalized and language-aligned, the next step then consists in applying automated reasoning techniques for assessing student solution correctness. We consider two methods of automated theorem proving: off-the-shelf SMT solving and term rewriting systems tailored for physics problems involving trigonometric expressions. The development of the term rewrite system and establishing termination and confluence properties was not trivial, and we describe it in some detail in the paper. We evaluate our system on a rich pool of over 1500 real-world student exam responses from the 2023 Australian Physics Olympiad. 

---
# Foundations for Risk Assessment of AI in Protecting Fundamental Rights 

**Authors**: Antonino Rotolo, Beatrice Ferrigno, Jose Miguel Angel Garcia Godinez, Claudio Novelli, Giovanni Sartor  

**Link**: [PDF](https://arxiv.org/pdf/2507.18290)  

**Abstract**: This chapter introduces a conceptual framework for qualitative risk assessment of AI, particularly in the context of the EU AI Act. The framework addresses the complexities of legal compliance and fundamental rights protection by itegrating definitional balancing and defeasible reasoning. Definitional balancing employs proportionality analysis to resolve conflicts between competing rights, while defeasible reasoning accommodates the dynamic nature of legal decision-making. Our approach stresses the need for an analysis of AI deployment scenarios and for identifying potential legal violations and multi-layered impacts on fundamental rights. On the basis of this analysis, we provide philosophical foundations for a logical account of AI risk analysis. In particular, we consider the basic building blocks for conceptually grasping the interaction between AI deployment scenarios and fundamental rights, incorporating in defeasible reasoning definitional balancing and arguments about the contextual promotion or demotion of rights. This layered approach allows for more operative models of assessment of both high-risk AI systems and General Purpose AI (GPAI) systems, emphasizing the broader applicability of the latter. Future work aims to develop a formal model and effective algorithms to enhance AI risk assessment, bridging theoretical insights with practical applications to support responsible AI governance. 

---
# Comparing Non-minimal Semantics for Disjunction in Answer Set Programming 

**Authors**: Felicidad Aguado, Pedro Cabalar, Brais Muñiz, Gilberto Pérez, Concepción Vidal  

**Link**: [PDF](https://arxiv.org/pdf/2507.18198)  

**Abstract**: In this paper, we compare four different semantics for disjunction in Answer Set Programming that, unlike stable models, do not adhere to the principle of model minimality. Two of these approaches, Cabalar and Muñiz' \emph{Justified Models} and Doherty and Szalas' \emph{Strongly Supported Models}, directly provide an alternative non-minimal semantics for disjunction. The other two, Aguado et al's \emph{Forks} and Shen and Eiter's \emph{Determining Inference} (DI) semantics, actually introduce a new disjunction connective, but are compared here as if they constituted new semantics for the standard disjunction operator. We are able to prove that three of these approaches (Forks, Justified Models and a reasonable relaxation of the DI semantics) actually coincide, constituting a common single approach under different definitions. Moreover, this common semantics always provides a superset of the stable models of a program (in fact, modulo any context) and is strictly stronger than the fourth approach (Strongly Supported Models), that actually treats disjunctions as in classical logic. 

---
# Decoupling Knowledge and Reasoning in LLMs: An Exploration Using Cognitive Dual-System Theory 

**Authors**: Mutian Yang, Jiandong Gao, Ji Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18178)  

**Abstract**: While large language models (LLMs) leverage both knowledge and reasoning during inference, the capacity to distinguish between them plays a pivotal role in model analysis, interpretability, and development. Inspired by dual-system cognitive theory, we propose a cognition attribution framework to decouple the contribution of knowledge and reasoning. In particular, the cognition of LLMs is decomposed into two distinct yet complementary phases: knowledge retrieval (Phase 1) and reasoning adjustment (Phase 2). To separate these phases, LLMs are prompted to generate answers under two different cognitive modes, fast thinking and slow thinking, respectively. The performance under different cognitive modes is analyzed to quantify the contribution of knowledge and reasoning. This architecture is employed to 15 LLMs across 3 datasets. Results reveal: (1) reasoning adjustment is domain-specific, benefiting reasoning-intensive domains (e.g., mathematics, physics, and chemistry) and potentially imparing knowledge-intensive domains. (2) Parameter scaling improves both knowledge and reasoning, with knowledge improvements being more pronounced. Additionally, parameter scaling make LLMs reasoning significantly more prudent, while moderately more intelligent. (3) Knowledge primarily resides in lower network layers, while reasoning operates in higher layers. Our framework not only helps understand LLMs from a "decoupling" perspective, but also provides new insights into existing research, including scaling laws, hierarchical knowledge editing, and limitations of small-model reasoning. 

---
# Logical Characterizations of GNNs with Mean Aggregation 

**Authors**: Moritz Schönherr, Carsten Lutz  

**Link**: [PDF](https://arxiv.org/pdf/2507.18145)  

**Abstract**: We study the expressive power of graph neural networks (GNNs) with mean as the aggregation function. In the non-uniform setting, we show that such GNNs have exactly the same expressive power as ratio modal logic, which has modal operators expressing that at least a certain ratio of the successors of a vertex satisfies a specified property. The non-uniform expressive power of mean GNNs is thus higher than that of GNNs with max aggregation, but lower than for sum aggregation--the latter are characterized by modal logic and graded modal logic, respectively. In the uniform setting, we show that the expressive power relative to MSO is exactly that of alternation-free modal logic, under the natural assumptions that combination functions are continuous and classification functions are thresholds. This implies that, relative to MSO and in the uniform setting, mean GNNs are strictly less expressive than sum GNNs and max GNNs. When any of the assumptions is dropped, the expressive power increases. 

---
# Actively evaluating and learning the distinctions that matter: Vaccine safety signal detection from emergency triage notes 

**Authors**: Sedigh Khademi, Christopher Palmer, Muhammad Javed, Hazel Clothier, Jim Buttery, Gerardo Luis Dimaguila, Jim Black  

**Link**: [PDF](https://arxiv.org/pdf/2507.18123)  

**Abstract**: The rapid development of COVID-19 vaccines has showcased the global communitys ability to combat infectious diseases. However, the need for post-licensure surveillance systems has grown due to the limited window for safety data collection in clinical trials and early widespread implementation. This study aims to employ Natural Language Processing techniques and Active Learning to rapidly develop a classifier that detects potential vaccine safety issues from emergency department notes. ED triage notes, containing expert, succinct vital patient information at the point of entry to health systems, can significantly contribute to timely vaccine safety signal surveillance. While keyword-based classification can be effective, it may yield false positives and demand extensive keyword modifications. This is exacerbated by the infrequency of vaccination-related ED presentations and their similarity to other reasons for ED visits. NLP offers a more accurate and efficient alternative, albeit requiring annotated data, which is often scarce in the medical field. Active learning optimizes the annotation process and the quality of annotated data, which can result in faster model implementation and improved model performance. This work combines active learning, data augmentation, and active learning and evaluation techniques to create a classifier that is used to enhance vaccine safety surveillance from ED triage notes. 

---
# Agentic AI framework for End-to-End Medical Data Inference 

**Authors**: Soorya Ram Shimgekar, Shayan Vassef, Abhay Goyal, Navin Kumar, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2507.18115)  

**Abstract**: Building and deploying machine learning solutions in healthcare remains expensive and labor-intensive due to fragmented preprocessing workflows, model compatibility issues, and stringent data privacy constraints. In this work, we introduce an Agentic AI framework that automates the entire clinical data pipeline, from ingestion to inference, through a system of modular, task-specific agents. These agents handle both structured and unstructured data, enabling automatic feature selection, model selection, and preprocessing recommendation without manual intervention. We evaluate the system on publicly available datasets from geriatrics, palliative care, and colonoscopy imaging. For example, in the case of structured data (anxiety data) and unstructured data (colonoscopy polyps data), the pipeline begins with file-type detection by the Ingestion Identifier Agent, followed by the Data Anonymizer Agent ensuring privacy compliance, where we first identify the data type and then anonymize it. The Feature Extraction Agent identifies features using an embedding-based approach for tabular data, extracting all column names, and a multi-stage MedGemma-based approach for image data, which infers modality and disease name. These features guide the Model-Data Feature Matcher Agent in selecting the best-fit model from a curated repository. The Preprocessing Recommender Agent and Preprocessing Implementor Agent then apply tailored preprocessing based on data type and model requirements. Finally, the ``Model Inference Agent" runs the selected model on the uploaded data and generates interpretable outputs using tools like SHAP, LIME, and DETR attention maps. By automating these high-friction stages of the ML lifecycle, the proposed framework reduces the need for repeated expert intervention, offering a scalable, cost-efficient pathway for operationalizing AI in clinical environments. 

---
# AlphaGo Moment for Model Architecture Discovery 

**Authors**: Yixiu Liu, Yang Nan, Weixian Xu, Xiangkun Hu, Lyumanshan Ye, Zhen Qin, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18074)  

**Abstract**: While AI systems demonstrate exponentially improving capabilities, the pace of AI research itself remains linearly bounded by human cognitive capacity, creating an increasingly severe development bottleneck. We present ASI-Arch, the first demonstration of Artificial Superintelligence for AI research (ASI4AI) in the critical domain of neural architecture discovery--a fully autonomous system that shatters this fundamental constraint by enabling AI to conduct its own architectural innovation. Moving beyond traditional Neural Architecture Search (NAS), which is fundamentally limited to exploring human-defined spaces, we introduce a paradigm shift from automated optimization to automated innovation. ASI-Arch can conduct end-to-end scientific research in the domain of architecture discovery, autonomously hypothesizing novel architectural concepts, implementing them as executable code, training and empirically validating their performance through rigorous experimentation and past experience. ASI-Arch conducted 1,773 autonomous experiments over 20,000 GPU hours, culminating in the discovery of 106 innovative, state-of-the-art (SOTA) linear attention architectures. Like AlphaGo's Move 37 that revealed unexpected strategic insights invisible to human players, our AI-discovered architectures demonstrate emergent design principles that systematically surpass human-designed baselines and illuminate previously unknown pathways for architectural innovation. Crucially, we establish the first empirical scaling law for scientific discovery itself--demonstrating that architectural breakthroughs can be scaled computationally, transforming research progress from a human-limited to a computation-scalable process. We provide comprehensive analysis of the emergent design patterns and autonomous research capabilities that enabled these breakthroughs, establishing a blueprint for self-accelerating AI systems. 

---
# Multi-Agent Guided Policy Optimization 

**Authors**: Yueheng Li, Guangming Xie, Zongqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18059)  

**Abstract**: Due to practical constraints such as partial observability and limited communication, Centralized Training with Decentralized Execution (CTDE) has become the dominant paradigm in cooperative Multi-Agent Reinforcement Learning (MARL). However, existing CTDE methods often underutilize centralized training or lack theoretical guarantees. We propose Multi-Agent Guided Policy Optimization (MAGPO), a novel framework that better leverages centralized training by integrating centralized guidance with decentralized execution. MAGPO uses an auto-regressive joint policy for scalable, coordinated exploration and explicitly aligns it with decentralized policies to ensure deployability under partial observability. We provide theoretical guarantees of monotonic policy improvement and empirically evaluate MAGPO on 43 tasks across 6 diverse environments. Results show that MAGPO consistently outperforms strong CTDE baselines and matches or surpasses fully centralized approaches, offering a principled and practical solution for decentralized multi-agent learning. Our code and experimental data can be found in this https URL. 

---
# Does visualization help AI understand data? 

**Authors**: Victoria R. Li, Johnathan Sun, Martin Wattenberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.18022)  

**Abstract**: Charts and graphs help people analyze data, but can they also be useful to AI systems? To investigate this question, we perform a series of experiments with two commercial vision-language models: GPT 4.1 and Claude 3.5. Across three representative analysis tasks, the two systems describe synthetic datasets more precisely and accurately when raw data is accompanied by a scatterplot, especially as datasets grow in complexity. Comparison with two baselines -- providing a blank chart and a chart with mismatched data -- shows that the improved performance is due to the content of the charts. Our results are initial evidence that AI systems, like humans, can benefit from visualization. 

---
# E.A.R.T.H.: Structuring Creative Evolution through Model Error in Generative AI 

**Authors**: Yusen Peng, Shuhua Mao  

**Link**: [PDF](https://arxiv.org/pdf/2507.18004)  

**Abstract**: How can AI move beyond imitation toward genuine creativity? This paper proposes the E.A.R.T.H. framework, a five-stage generative pipeline that transforms model-generated errors into creative assets through Error generation, Amplification, Refine selection, Transform, and Harness feedback. Drawing on cognitive science and generative modeling, we posit that "creative potential hides in failure" and operationalize this via structured prompts, semantic scoring, and human-in-the-loop evaluation. Implemented using LLaMA-2-7B-Chat, SBERT, BERTScore, CLIP, BLIP-2, and Stable Diffusion, the pipeline employs a composite reward function based on novelty, surprise, and relevance. At the Refine stage, creativity scores increase by 52.5% (1.179 to 1.898, t = -5.56, p < 0.001), with final outputs reaching 2.010 - a 70.4% improvement. Refined slogans are 48.4% shorter, 40.7% more novel, with only a 4.0% drop in relevance. Cross-modal tests show strong slogan-to-image alignment (CLIPScore: 0.249; BERTScore F1: 0.816). In human evaluations, 60% of outputs scored >= 4.0, with metaphorical slogans (avg. 4.09) outperforming literal ones (3.99). Feedback highlights stylistic precision and emotional resonance. These results demonstrate that error-centered, feedback-driven generation enhances creativity, offering a scalable path toward self-evolving, human-aligned creative AI. 

---
# Synthesis of timeline-based planning strategies avoiding determinization 

**Authors**: Dario Della Monica, Angelo Montanari, Pietro Sala  

**Link**: [PDF](https://arxiv.org/pdf/2507.17988)  

**Abstract**: Qualitative timeline-based planning models domains as sets of independent, but
interacting, components whose behaviors over time, the timelines, are governed
by sets of qualitative temporal constraints (ordering relations), called
synchronization rules.
Its plan-existence problem has been shown to be PSPACE-complete; in
particular, PSPACE-membership has been proved via reduction to the
nonemptiness problem for nondeterministic finite automata.
However, nondeterministic automata cannot be directly used to synthesize
planning strategies as a costly determinization step is needed.
In this paper, we identify a fragment of qualitative timeline-based planning
whose plan-existence problem can be directly mapped into the nonemptiness
problem of deterministic finite automata, which can then
synthesize strategies.
In addition, we identify a maximal subset of Allen's relations that fits into
such a deterministic fragment. 

---
# SMARTAPS: Tool-augmented LLMs for Operations Management 

**Authors**: Timothy Tin Long Yu, Mahdi Mostajabdaveh, Jabo Serge Byusa, Rindra Ramamonjison, Giuseppe Carenini, Kun Mao, Zirui Zhou, Yong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17927)  

**Abstract**: Large language models (LLMs) present intriguing opportunities to enhance user interaction with traditional algorithms and tools in real-world applications. An advanced planning system (APS) is a sophisticated software that leverages optimization to help operations planners create, interpret, and modify an operational plan. While highly beneficial, many customers are priced out of using an APS due to the ongoing costs of consultants responsible for customization and maintenance. To address the need for a more accessible APS expressed by supply chain planners, we present SmartAPS, a conversational system built on a tool-augmented LLM. Our system provides operations planners with an intuitive natural language chat interface, allowing them to query information, perform counterfactual reasoning, receive recommendations, and execute scenario analysis to better manage their operation. A short video demonstrating the system has been released: this https URL 

---
# I2I-STRADA -- Information to Insights via Structured Reasoning Agent for Data Analysis 

**Authors**: SaiBarath Sundar, Pranav Satheesan, Udayaadithya Avadhanam  

**Link**: [PDF](https://arxiv.org/pdf/2507.17874)  

**Abstract**: Recent advances in agentic systems for data analysis have emphasized automation of insight generation through multi-agent frameworks, and orchestration layers. While these systems effectively manage tasks like query translation, data transformation, and visualization, they often overlook the structured reasoning process underlying analytical thinking. Reasoning large language models (LLMs) used for multi-step problem solving are trained as general-purpose problem solvers. As a result, their reasoning or thinking steps do not adhere to fixed processes for specific tasks. Real-world data analysis requires a consistent cognitive workflow: interpreting vague goals, grounding them in contextual knowledge, constructing abstract plans, and adapting execution based on intermediate outcomes. We introduce I2I-STRADA (Information-to-Insight via Structured Reasoning Agent for Data Analysis), an agentic architecture designed to formalize this reasoning process. I2I-STRADA focuses on modeling how analysis unfolds via modular sub-tasks that reflect the cognitive steps of analytical reasoning. Evaluations on the DABstep and DABench benchmarks show that I2I-STRADA outperforms prior systems in planning coherence and insight alignment, highlighting the importance of structured cognitive workflows in agent design for data analysis. 

---
# ASP-Assisted Symbolic Regression: Uncovering Hidden Physics in Fluid Mechanics 

**Authors**: Theofanis Aravanis, Grigorios Chrimatopoulos, Mohammad Ferdows, Michalis Xenos, Efstratios Em Tzirtzilakis  

**Link**: [PDF](https://arxiv.org/pdf/2507.17777)  

**Abstract**: Unlike conventional Machine-Learning (ML) approaches, often criticized as "black boxes", Symbolic Regression (SR) stands out as a powerful tool for revealing interpretable mathematical relationships in complex physical systems, requiring no a priori assumptions about models' structures. Motivated by the recognition that, in fluid mechanics, an understanding of the underlying flow physics is as crucial as accurate prediction, this study applies SR to model a fundamental three-dimensional (3D) incompressible flow in a rectangular channel, focusing on the (axial) velocity and pressure fields under laminar conditions. By employing the PySR library, compact symbolic equations were derived directly from numerical simulation data, revealing key characteristics of the flow dynamics. These equations not only approximate the parabolic velocity profile and pressure drop observed in the studied fluid flow, but also perfectly coincide with analytical solutions from the literature. Furthermore, we propose an innovative approach that integrates SR with the knowledge-representation framework of Answer Set Programming (ASP), combining the generative power of SR with the declarative reasoning strengths of ASP. The proposed hybrid SR/ASP framework ensures that the SR-generated symbolic expressions are not only statistically accurate, but also physically plausible, adhering to domain-specific principles. Overall, the study highlights two key contributions: SR's ability to simplify complex flow behaviours into concise, interpretable equations, and the potential of knowledge-representation approaches to improve the reliability and alignment of data-driven SR models with domain principles. Insights from the examined 3D channel flow pave the way for integrating such hybrid approaches into efficient frameworks, [...] where explainable predictions and real-time data analysis are crucial. 

---
# SIDA: Synthetic Image Driven Zero-shot Domain Adaptation 

**Authors**: Ye-Chan Kim, SeungJu Cha, Si-Woo Kim, Taewhan Kim, Dong-Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.18632)  

**Abstract**: Zero-shot domain adaptation is a method for adapting a model to a target domain without utilizing target domain image data. To enable adaptation without target images, existing studies utilize CLIP's embedding space and text description to simulate target-like style features. Despite the previous achievements in zero-shot domain adaptation, we observe that these text-driven methods struggle to capture complex real-world variations and significantly increase adaptation time due to their alignment process. Instead of relying on text descriptions, we explore solutions leveraging image data, which provides diverse and more fine-grained style cues. In this work, we propose SIDA, a novel and efficient zero-shot domain adaptation method leveraging synthetic images. To generate synthetic images, we first create detailed, source-like images and apply image translation to reflect the style of the target domain. We then utilize the style features of these synthetic images as a proxy for the target domain. Based on these features, we introduce Domain Mix and Patch Style Transfer modules, which enable effective modeling of real-world variations. In particular, Domain Mix blends multiple styles to expand the intra-domain representations, and Patch Style Transfer assigns different styles to individual patches. We demonstrate the effectiveness of our method by showing state-of-the-art performance in diverse zero-shot adaptation scenarios, particularly in challenging domains. Moreover, our approach achieves high efficiency by significantly reducing the overall adaptation time. 

---
# 3D Software Synthesis Guided by Constraint-Expressive Intermediate Representation 

**Authors**: Shuqing Li, Anson Y. Lam, Yun Peng, Wenxuan Wang, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18625)  

**Abstract**: Graphical user interface (UI) software has undergone a fundamental transformation from traditional two-dimensional (2D) desktop/web/mobile interfaces to spatial three-dimensional (3D) environments. While existing work has made remarkable success in automated 2D software generation, such as HTML/CSS and mobile app interface code synthesis, the generation of 3D software still remains under-explored. Current methods for 3D software generation usually generate the 3D environments as a whole and cannot modify or control specific elements in the software. Furthermore, these methods struggle to handle the complex spatial and semantic constraints inherent in the real world. To address the challenges, we present Scenethesis, a novel requirement-sensitive 3D software synthesis approach that maintains formal traceability between user specifications and generated 3D software. Scenethesis is built upon ScenethesisLang, a domain-specific language that serves as a granular constraint-aware intermediate representation (IR) to bridge natural language requirements and executable 3D software. It serves both as a comprehensive scene description language enabling fine-grained modification of 3D software elements and as a formal constraint-expressive specification language capable of expressing complex spatial constraints. By decomposing 3D software synthesis into stages operating on ScenethesisLang, Scenethesis enables independent verification, targeted modification, and systematic constraint satisfaction. Our evaluation demonstrates that Scenethesis accurately captures over 80% of user requirements and satisfies more than 90% of hard constraints while handling over 100 constraints simultaneously. Furthermore, Scenethesis achieves a 42.8% improvement in BLIP-2 visual evaluation scores compared to the state-of-the-art method. 

---
# Moving Out: Physically-grounded Human-AI Collaboration 

**Authors**: Xuhui Kang, Sung-Wook Lee, Haolin Liu, Yuyan Wang, Yen-Ling Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2507.18623)  

**Abstract**: The ability to adapt to physical actions and constraints in an environment is crucial for embodied agents (e.g., robots) to effectively collaborate with humans. Such physically grounded human-AI collaboration must account for the increased complexity of the continuous state-action space and constrained dynamics caused by physical constraints. In this paper, we introduce \textit{Moving Out}, a new human-AI collaboration benchmark that resembles a wide range of collaboration modes affected by physical attributes and constraints, such as moving heavy items together and maintaining consistent actions to move a big item around a corner. Using Moving Out, we designed two tasks and collected human-human interaction data to evaluate models' abilities to adapt to diverse human behaviors and unseen physical attributes. To address the challenges in physical environments, we propose a novel method, BASS (Behavior Augmentation, Simulation, and Selection), to enhance the diversity of agents and their understanding of the outcome of actions. Our experiments show that BASS outperforms state-of-the-art models in AI-AI and human-AI collaboration. The project page is available at \href{this https URL}{this https URL\_ai/}. 

---
# SynC: Synthetic Image Caption Dataset Refinement with One-to-many Mapping for Zero-shot Image Captioning 

**Authors**: Si-Woo Kim, MinJu Jeon, Ye-Chan Kim, Soeun Lee, Taewhan Kim, Dong-Jin Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.18616)  

**Abstract**: Zero-shot Image Captioning (ZIC) increasingly utilizes synthetic datasets generated by text-to-image (T2I) models to mitigate the need for costly manual annotation. However, these T2I models often produce images that exhibit semantic misalignments with their corresponding input captions (e.g., missing objects, incorrect attributes), resulting in noisy synthetic image-caption pairs that can hinder model training. Existing dataset pruning techniques are largely designed for removing noisy text in web-crawled data. However, these methods are ill-suited for the distinct challenges of synthetic data, where captions are typically well-formed, but images may be inaccurate representations. To address this gap, we introduce SynC, a novel framework specifically designed to refine synthetic image-caption datasets for ZIC. Instead of conventional filtering or regeneration, SynC focuses on reassigning captions to the most semantically aligned images already present within the synthetic image pool. Our approach employs a one-to-many mapping strategy by initially retrieving multiple relevant candidate images for each caption. We then apply a cycle-consistency-inspired alignment scorer that selects the best image by verifying its ability to retrieve the original caption via image-to-text retrieval. Extensive evaluations demonstrate that SynC consistently and significantly improves performance across various ZIC models on standard benchmarks (MS-COCO, Flickr30k, NoCaps), achieving state-of-the-art results in several scenarios. SynC offers an effective strategy for curating refined synthetic data to enhance ZIC. 

---
# Approximate SMT Counting Beyond Discrete Domains 

**Authors**: Arijit Shaw, Kuldeep S. Meel  

**Link**: [PDF](https://arxiv.org/pdf/2507.18612)  

**Abstract**: Satisfiability Modulo Theory (SMT) solvers have advanced automated reasoning, solving complex formulas across discrete and continuous domains. Recent progress in propositional model counting motivates extending SMT capabilities toward model counting, especially for hybrid SMT formulas. Existing approaches, like bit-blasting, are limited to discrete variables, highlighting the challenge of counting solutions projected onto the discrete domain in hybrid formulas.
We introduce pact, an SMT model counter for hybrid formulas that uses hashing-based approximate model counting to estimate solutions with theoretical guarantees. pact makes a logarithmic number of SMT solver calls relative to the projection variables, leveraging optimized hash functions. pact achieves significant performance improvements over baselines on a large suite of benchmarks. In particular, out of 14,202 instances, pact successfully finished on 603 instances, while Baseline could only finish on 13 instances. 

---
# DRWKV: Focusing on Object Edges for Low-Light Image Enhancement 

**Authors**: Xuecheng Bai, Yuxiang Wang, Boyu Hu, Qinyuan Jie, Chuanzhi Xu, Hongru Xiao, Kechen Li, Vera Chung  

**Link**: [PDF](https://arxiv.org/pdf/2507.18594)  

**Abstract**: Low-light image enhancement remains a challenging task, particularly in preserving object edge continuity and fine structural details under extreme illumination degradation. In this paper, we propose a novel model, DRWKV (Detailed Receptance Weighted Key Value), which integrates our proposed Global Edge Retinex (GER) theory, enabling effective decoupling of illumination and edge structures for enhanced edge fidelity. Secondly, we introduce Evolving WKV Attention, a spiral-scanning mechanism that captures spatial edge continuity and models irregular structures more effectively. Thirdly, we design the Bilateral Spectrum Aligner (Bi-SAB) and a tailored MS2-Loss to jointly align luminance and chrominance features, improving visual naturalness and mitigating artifacts. Extensive experiments on five LLIE benchmarks demonstrate that DRWKV achieves leading performance in PSNR, SSIM, and NIQE while maintaining low computational complexity. Furthermore, DRWKV enhances downstream performance in low-light multi-object tracking tasks, validating its generalization capabilities. 

---
# A Foundation Model for Massive MIMO Precoding with an Adaptive per-User Rate-Power Tradeoff 

**Authors**: Jérôme Emery, Ali Hasanzadeh Karkan, Jean-François Frigon, François Leduc-Primeau  

**Link**: [PDF](https://arxiv.org/pdf/2507.18587)  

**Abstract**: Deep learning (DL) has emerged as a solution for precoding in massive multiple-input multiple-output (mMIMO) systems due to its capacity to learn the characteristics of the propagation environment. However, training such a model requires high-quality, local datasets at the deployment site, which are often difficult to collect. We propose a transformer-based foundation model for mMIMO precoding that seeks to minimize the energy consumption of the transmitter while dynamically adapting to per-user rate requirements. At equal energy consumption, zero-shot deployment of the proposed foundation model significantly outperforms zero forcing, and approaches weighted minimum mean squared error performance with 8x less complexity. To address model adaptation in data-scarce settings, we introduce a data augmentation method that finds training samples similar to the target distribution by computing the cosine similarity between the outputs of the pre-trained feature extractor. Our work enables the implementation of DL-based solutions in practice by addressing challenges of data availability and training complexity. Moreover, the ability to dynamically configure per-user rate requirements can be leveraged by higher level resource allocation and scheduling algorithms for greater control over energy efficiency, spectral efficiency and fairness. 

---
# AQuilt: Weaving Logic and Self-Inspection into Low-Cost, High-Relevance Data Synthesis for Specialist LLMs 

**Authors**: Xiaopeng Ke, Hexuan Deng, Xuebo Liu, Jun Rao, Zhenxi Song, Jun Yu, Min Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18584)  

**Abstract**: Despite the impressive performance of large language models (LLMs) in general domains, they often underperform in specialized domains. Existing approaches typically rely on data synthesis methods and yield promising results by using unlabeled data to capture domain-specific features. However, these methods either incur high computational costs or suffer from performance limitations, while also demonstrating insufficient generalization across different tasks. To address these challenges, we propose AQuilt, a framework for constructing instruction-tuning data for any specialized domains from corresponding unlabeled data, including Answer, Question, Unlabeled data, Inspection, Logic, and Task type. By incorporating logic and inspection, we encourage reasoning processes and self-inspection to enhance model performance. Moreover, customizable task instructions enable high-quality data generation for any task. As a result, we construct a dataset of 703k examples to train a powerful data synthesis model. Experiments show that AQuilt is comparable to DeepSeek-V3 while utilizing just 17% of the production cost. Further analysis demonstrates that our generated data exhibits higher relevance to downstream tasks. Source code, models, and scripts are available at this https URL. 

---
# DR.EHR: Dense Retrieval for Electronic Health Record with Knowledge Injection and Synthetic Data 

**Authors**: Zhengyun Zhao, Huaiyuan Ying, Yue Zhong, Sheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18583)  

**Abstract**: Electronic Health Records (EHRs) are pivotal in clinical practices, yet their retrieval remains a challenge mainly due to semantic gap issues. Recent advancements in dense retrieval offer promising solutions but existing models, both general-domain and biomedical-domain, fall short due to insufficient medical knowledge or mismatched training corpora. This paper introduces \texttt{this http URL}, a series of dense retrieval models specifically tailored for EHR retrieval. We propose a two-stage training pipeline utilizing MIMIC-IV discharge summaries to address the need for extensive medical knowledge and large-scale training data. The first stage involves medical entity extraction and knowledge injection from a biomedical knowledge graph, while the second stage employs large language models to generate diverse training data. We train two variants of \texttt{this http URL}, with 110M and 7B parameters, respectively. Evaluated on the CliniQ benchmark, our models significantly outperforms all existing dense retrievers, achieving state-of-the-art results. Detailed analyses confirm our models' superiority across various match and query types, particularly in challenging semantic matches like implication and abbreviation. Ablation studies validate the effectiveness of each pipeline component, and supplementary experiments on EHR QA datasets demonstrate the models' generalizability on natural language questions, including complex ones with multiple entities. This work significantly advances EHR retrieval, offering a robust solution for clinical applications. 

---
# Advancing Financial Engineering with Foundation Models: Progress, Applications, and Challenges 

**Authors**: Liyuan Chen, Shuoling Liu, Jiangpeng Yan, Xiaoyu Wang, Henglin Liu, Chuang Li, Kecheng Jiao, Jixuan Ying, Yang Veronica Liu, Qiang Yang, Xiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.18577)  

**Abstract**: The advent of foundation models (FMs) - large-scale pre-trained models with strong generalization capabilities - has opened new frontiers for financial engineering. While general-purpose FMs such as GPT-4 and Gemini have demonstrated promising performance in tasks ranging from financial report summarization to sentiment-aware forecasting, many financial applications remain constrained by unique domain requirements such as multimodal reasoning, regulatory compliance, and data privacy. These challenges have spurred the emergence of Financial Foundation Models (FFMs) - a new class of models explicitly designed for finance. This survey presents a comprehensive overview of FFMs, with a taxonomy spanning three key modalities: Financial Language Foundation Models (FinLFMs), Financial Time-Series Foundation Models (FinTSFMs), and Financial Visual-Language Foundation Models (FinVLFMs). We review their architectures, training methodologies, datasets, and real-world applications. Furthermore, we identify critical challenges in data availability, algorithmic scalability, and infrastructure constraints, and offer insights into future research opportunities. We hope this survey serves as both a comprehensive reference for understanding FFMs and a practical roadmap for future innovation. An updated collection of FFM-related publications and resources will be maintained on our website this https URL. 

---
# PosterMate: Audience-driven Collaborative Persona Agents for Poster Design 

**Authors**: Donghoon Shin, Daniel Lee, Gary Hsieh, Gromit Yeuk-Yin Chan  

**Link**: [PDF](https://arxiv.org/pdf/2507.18572)  

**Abstract**: Poster designing can benefit from synchronous feedback from target audiences. However, gathering audiences with diverse perspectives and reconciling them on design edits can be challenging. Recent generative AI models present opportunities to simulate human-like interactions, but it is unclear how they may be used for feedback processes in design. We introduce PosterMate, a poster design assistant that facilitates collaboration by creating audience-driven persona agents constructed from marketing documents. PosterMate gathers feedback from each persona agent regarding poster components, and stimulates discussion with the help of a moderator to reach a conclusion. These agreed-upon edits can then be directly integrated into the poster design. Through our user study (N=12), we identified the potential of PosterMate to capture overlooked viewpoints, while serving as an effective prototyping tool. Additionally, our controlled online evaluation (N=100) revealed that the feedback from an individual persona agent is appropriate given its persona identity, and the discussion effectively synthesizes the different persona agents' perspectives. 

---
# Proceedings 19th International Workshop on the ACL2 Theorem Prover and Its Applications 

**Authors**: Ruben Gamboa, Panagiotis Manolios  

**Link**: [PDF](https://arxiv.org/pdf/2507.18567)  

**Abstract**: The ACL2 Workshop series is the major technical forum for users of the ACL2 theorem proving system to present research related to the ACL2 theorem prover and its applications. ACL2 is an industrial-strength automated reasoning system, the latest in the Boyer-Moore family of theorem provers. The 2005 ACM Software System Award was awarded to Boyer, Kaufmann, and Moore for their work on ACL2 and the other theorem provers in the Boyer-Moore family. 

---
# GIIFT: Graph-guided Inductive Image-free Multimodal Machine Translation 

**Authors**: Jiafeng Xiong, Yuting Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2507.18562)  

**Abstract**: Multimodal Machine Translation (MMT) has demonstrated the significant help of visual information in machine translation. However, existing MMT methods face challenges in leveraging the modality gap by enforcing rigid visual-linguistic alignment whilst being confined to inference within their trained multimodal domains. In this work, we construct novel multimodal scene graphs to preserve and integrate modality-specific information and introduce GIIFT, a two-stage Graph-guided Inductive Image-Free MMT framework that uses a cross-modal Graph Attention Network adapter to learn multimodal knowledge in a unified fused space and inductively generalize it to broader image-free translation domains. Experimental results on the Multi30K dataset of English-to-French and English-to-German tasks demonstrate that our GIIFT surpasses existing approaches and achieves the state-of-the-art, even without images during inference. Results on the WMT benchmark show significant improvements over the image-free translation baselines, demonstrating the strength of GIIFT towards inductive image-free inference. 

---
# Beyond Internal Data: Constructing Complete Datasets for Fairness Testing 

**Authors**: Varsha Ramineni, Hossein A. Rahmani, Emine Yilmaz, David Barber  

**Link**: [PDF](https://arxiv.org/pdf/2507.18561)  

**Abstract**: As AI becomes prevalent in high-risk domains and decision-making, it is essential to test for potential harms and biases. This urgency is reflected by the global emergence of AI regulations that emphasise fairness and adequate testing, with some mandating independent bias audits. However, procuring the necessary data for fairness testing remains a significant challenge. Particularly in industry settings, legal and privacy concerns restrict the collection of demographic data required to assess group disparities, and auditors face practical and cultural challenges in gaining access to data. Further, internal historical datasets are often insufficiently representative to identify real-world biases. This work focuses on evaluating classifier fairness when complete datasets including demographics are inaccessible. We propose leveraging separate overlapping datasets to construct complete synthetic data that includes demographic information and accurately reflects the underlying relationships between protected attributes and model features. We validate the fidelity of the synthetic data by comparing it to real data, and empirically demonstrate that fairness metrics derived from testing on such synthetic data are consistent with those obtained from real data. This work, therefore, offers a path to overcome real-world data scarcity for fairness testing, enabling independent, model-agnostic evaluation of fairness, and serving as a viable substitute where real data is limited. 

---
# HARLF: Hierarchical Reinforcement Learning and Lightweight LLM-Driven Sentiment Integration for Financial Portfolio Optimization 

**Authors**: Benjamin Coriat, Eric Benhamou  

**Link**: [PDF](https://arxiv.org/pdf/2507.18560)  

**Abstract**: This paper presents a novel hierarchical framework for portfolio optimization, integrating lightweight Large Language Models (LLMs) with Deep Reinforcement Learning (DRL) to combine sentiment signals from financial news with traditional market indicators. Our three-tier architecture employs base RL agents to process hybrid data, meta-agents to aggregate their decisions, and a super-agent to merge decisions based on market data and sentiment analysis. Evaluated on data from 2018 to 2024, after training on 2000-2017, the framework achieves a 26% annualized return and a Sharpe ratio of 1.2, outperforming equal-weighted and S&P 500 benchmarks. Key contributions include scalable cross-modal integration, a hierarchical RL structure for enhanced stability, and open-source reproducibility. 

---
# VideoMind: An Omni-Modal Video Dataset with Intent Grounding for Deep-Cognitive Video Understanding 

**Authors**: Baoyao Yang, Wanyun Li, Dixin Chen, Junxiang Chen, Wenbin Yao, Haifeng Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.18552)  

**Abstract**: This paper introduces VideoMind, a video-centric omni-modal dataset designed for deep video content cognition and enhanced multi-modal feature representation. The dataset comprises 103K video samples (3K reserved for testing), each paired with audio and systematically detailed textual descriptions. Specifically, every video and its audio is described across three hierarchical layers (factual, abstract, and intent), progressing from surface to depth. It contains over 22 million words, averaging ~225 words per sample. VideoMind's key distinction from existing datasets is its provision of intent expressions, which require contextual integration across the entire video and are not directly observable. These deep-cognitive expressions are generated using a Chain-of-Thought (COT) approach, prompting the mLLM through step-by-step reasoning. Each description includes annotations for subject, place, time, event, action, and intent, supporting downstream recognition tasks. Crucially, we establish a gold-standard benchmark with 3,000 manually validated samples for evaluating deep-cognitive video understanding. We design hybrid-cognitive retrieval experiments, scored by multi-level retrieval metrics, to appropriately assess deep video comprehension. Evaluation results for models (e.g., InternVideo, VAST, UMT-L) are released. VideoMind serves as a powerful benchmark for fine-grained cross-modal alignment and advances fields requiring in-depth video understanding, such as emotion and intent recognition. The data is publicly available on GitHub, HuggingFace, and OpenDataLab, this https URL. 

---
# GLiNER2: An Efficient Multi-Task Information Extraction System with Schema-Driven Interface 

**Authors**: Urchade Zaratiana, Gil Pasternak, Oliver Boyd, George Hurn-Maloney, Ash Lewis  

**Link**: [PDF](https://arxiv.org/pdf/2507.18546)  

**Abstract**: Information extraction (IE) is fundamental to numerous NLP applications, yet existing solutions often require specialized models for different tasks or rely on computationally expensive large language models. We present GLiNER2, a unified framework that enhances the original GLiNER architecture to support named entity recognition, text classification, and hierarchical structured data extraction within a single efficient model. Built pretrained transformer encoder architecture, GLiNER2 maintains CPU efficiency and compact size while introducing multi-task composition through an intuitive schema-based interface. Our experiments demonstrate competitive performance across extraction and classification tasks with substantial improvements in deployment accessibility compared to LLM-based alternatives. We release GLiNER2 as an open-source pip-installable library with pre-trained models and documentation at this https URL. 

---
# C2G-KD: PCA-Constrained Generator for Data-Free Knowledge Distillation 

**Authors**: Magnus Bengtsson, Kenneth Östberg  

**Link**: [PDF](https://arxiv.org/pdf/2507.18533)  

**Abstract**: We introduce C2G-KD, a data-free knowledge distillation framework where a class-conditional generator is trained to produce synthetic samples guided by a frozen teacher model and geometric constraints derived from PCA. The generator never observes real training data but instead learns to activate the teacher's output through a combination of semantic and structural losses. By constraining generated samples to lie within class-specific PCA subspaces estimated from as few as two real examples per class, we preserve topological consistency and diversity. Experiments on MNIST show that even minimal class structure is sufficient to bootstrap useful synthetic training pipelines. 

---
# GLANCE: Graph Logic Attention Network with Cluster Enhancement for Heterophilous Graph Representation Learning 

**Authors**: Zhongtian Sun, Anoushka Harit, Alexandra Cristea, Christl A. Donnelly, Pietro Liò  

**Link**: [PDF](https://arxiv.org/pdf/2507.18521)  

**Abstract**: Graph Neural Networks (GNNs) have demonstrated significant success in learning from graph-structured data but often struggle on heterophilous graphs, where connected nodes differ in features or class labels. This limitation arises from indiscriminate neighbor aggregation and insufficient incorporation of higher-order structural patterns. To address these challenges, we propose GLANCE (Graph Logic Attention Network with Cluster Enhancement), a novel framework that integrates logic-guided reasoning, dynamic graph refinement, and adaptive clustering to enhance graph representation learning. GLANCE combines a logic layer for interpretable and structured embeddings, multi-head attention-based edge pruning for denoising graph structures, and clustering mechanisms for capturing global patterns. Experimental results in benchmark datasets, including Cornell, Texas, and Wisconsin, demonstrate that GLANCE achieves competitive performance, offering robust and interpretable solutions for heterophilous graph scenarios. The proposed framework is lightweight, adaptable, and uniquely suited to the challenges of heterophilous graphs. 

---
# Explaining How Visual, Textual and Multimodal Encoders Share Concepts 

**Authors**: Clément Cornet, Romaric Besançon, Hervé Le Borgne  

**Link**: [PDF](https://arxiv.org/pdf/2507.18512)  

**Abstract**: Sparse autoencoders (SAEs) have emerged as a powerful technique for extracting human-interpretable features from neural networks activations. Previous works compared different models based on SAE-derived features but those comparisons have been restricted to models within the same modality. We propose a novel indicator allowing quantitative comparison of models across SAE features, and use it to conduct a comparative study of visual, textual and multimodal encoders. We also propose to quantify the Comparative Sharedness of individual features between different classes of models. With these two new tools, we conduct several studies on 21 encoders of the three types, with two significantly different sizes, and considering generalist and domain specific datasets. The results allow to revisit previous studies at the light of encoders trained in a multimodal context and to quantify to which extent all these models share some representations or features. They also suggest that visual features that are specific to VLMs among vision encoders are shared with text encoders, highlighting the impact of text pretraining. The code is available at this https URL 

---
# Reinforced Embodied Active Defense: Exploiting Adaptive Interaction for Robust Visual Perception in Adversarial 3D Environments 

**Authors**: Xiao Yang, Lingxuan Wu, Lizhong Wang, Chengyang Ying, Hang Su, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18484)  

**Abstract**: Adversarial attacks in 3D environments have emerged as a critical threat to the reliability of visual perception systems, particularly in safety-sensitive applications such as identity verification and autonomous driving. These attacks employ adversarial patches and 3D objects to manipulate deep neural network (DNN) predictions by exploiting vulnerabilities within complex scenes. Existing defense mechanisms, such as adversarial training and purification, primarily employ passive strategies to enhance robustness. However, these approaches often rely on pre-defined assumptions about adversarial tactics, limiting their adaptability in dynamic 3D settings. To address these challenges, we introduce Reinforced Embodied Active Defense (Rein-EAD), a proactive defense framework that leverages adaptive exploration and interaction with the environment to improve perception robustness in 3D adversarial contexts. By implementing a multi-step objective that balances immediate prediction accuracy with predictive entropy minimization, Rein-EAD optimizes defense strategies over a multi-step horizon. Additionally, Rein-EAD involves an uncertainty-oriented reward-shaping mechanism that facilitates efficient policy updates, thereby reducing computational overhead and supporting real-world applicability without the need for differentiable environments. Comprehensive experiments validate the effectiveness of Rein-EAD, demonstrating a substantial reduction in attack success rates while preserving standard accuracy across diverse tasks. Notably, Rein-EAD exhibits robust generalization to unseen and adaptive attacks, making it suitable for real-world complex tasks, including 3D object classification, face recognition and autonomous driving. 

---
# Automated Code Review Using Large Language Models with Symbolic Reasoning 

**Authors**: Busra Icoz, Goksel Biricik  

**Link**: [PDF](https://arxiv.org/pdf/2507.18476)  

**Abstract**: Code review is one of the key processes in the software development lifecycle and is essential to maintain code quality. However, manual code review is subjective and time consuming. Given its rule-based nature, code review is well suited for automation. In recent years, significant efforts have been made to automate this process with the help of artificial intelligence. Recent developments in Large Language Models (LLMs) have also emerged as a promising tool in this area, but these models often lack the logical reasoning capabilities needed to fully understand and evaluate code. To overcome this limitation, this study proposes a hybrid approach that integrates symbolic reasoning techniques with LLMs to automate the code review process. We tested our approach using the CodexGlue dataset, comparing several models, including CodeT5, CodeBERT, and GraphCodeBERT, to assess the effectiveness of combining symbolic reasoning and prompting techniques with LLMs. Our results show that this approach improves the accuracy and efficiency of automated code review. 

---
# Revisiting Physically Realizable Adversarial Object Attack against LiDAR-based Detection: Clarifying Problem Formulation and Experimental Protocols 

**Authors**: Luo Cheng, Hanwei Zhang, Lijun Zhang, Holger Hermanns  

**Link**: [PDF](https://arxiv.org/pdf/2507.18457)  

**Abstract**: Adversarial robustness in LiDAR-based 3D object detection is a critical research area due to its widespread application in real-world scenarios. While many digital attacks manipulate point clouds or meshes, they often lack physical realizability, limiting their practical impact. Physical adversarial object attacks remain underexplored and suffer from poor reproducibility due to inconsistent setups and hardware differences. To address this, we propose a device-agnostic, standardized framework that abstracts key elements of physical adversarial object attacks, supports diverse methods, and provides open-source code with benchmarking protocols in simulation and real-world settings. Our framework enables fair comparison, accelerates research, and is validated by successfully transferring simulated attacks to a physical LiDAR system. Beyond the framework, we offer insights into factors influencing attack success and advance understanding of adversarial robustness in real-world LiDAR perception. 

---
# Sandwich: Separating Prefill-Decode Compilation for Efficient CPU LLM Serving 

**Authors**: Juntao Zhao, Jiuru Li, Chuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18454)  

**Abstract**: Utilizing CPUs to serve large language models (LLMs) is a resource-friendly alternative to GPU serving. Existing CPU-based solutions ignore workload differences between the prefill and the decode phases of LLM inference, applying a static per-NUMA (Non-Uniform Memory Access) node model partition and utilizing vendor libraries for operator-level execution, which is suboptimal. We propose Sandwich, a hardware-centric CPU-based LLM serving engine that uses different execution plans for the prefill and decode phases and optimizes them separately.
We evaluate Sandwich across diverse baselines and datasets on five CPU platforms, including x86 with AVX-2 and AVX-512, as well as ARM with NEON. Sandwich achieves an average 2.01x throughput improvement and 90% satisfactory time-to-first-token (TTFT) and time-per-output-token (TPOT) latencies with up to 3.40x lower requirements in single sequence serving, and significant improvement in Goodput in continuous-batching serving. The GEMM kernels generated by Sandwich outperform representative vendor kernels and other dynamic shape solutions, achieving performance comparable to static compilers with three orders of magnitude less kernel tuning costs. 

---
# Generation of Synthetic Clinical Text: A Systematic Review 

**Authors**: Basel Alshaikhdeeb, Ahmed Abdelmonem Hemedan, Soumyabrata Ghosh, Irina Balaur, Venkata Satagopam  

**Link**: [PDF](https://arxiv.org/pdf/2507.18451)  

**Abstract**: Generating clinical synthetic text represents an effective solution for common clinical NLP issues like sparsity and privacy. This paper aims to conduct a systematic review on generating synthetic medical free-text by formulating quantitative analysis to three research questions concerning (i) the purpose of generation, (ii) the techniques, and (iii) the evaluation methods. We searched PubMed, ScienceDirect, Web of Science, Scopus, IEEE, Google Scholar, and arXiv databases for publications associated with generating synthetic medical unstructured free-text. We have identified 94 relevant articles out of 1,398 collected ones. A great deal of attention has been given to the generation of synthetic medical text from 2018 onwards, where the main purpose of such a generation is towards text augmentation, assistive writing, corpus building, privacy-preserving, annotation, and usefulness. Transformer architectures were the main predominant technique used to generate the text, especially the GPTs. On the other hand, there were four main aspects of evaluation, including similarity, privacy, structure, and utility, where utility was the most frequent method used to assess the generated synthetic medical text. Although the generated synthetic medical text demonstrated a moderate possibility to act as real medical documents in different downstream NLP tasks, it has proven to be a great asset as augmented, complementary to the real documents, towards improving the accuracy and overcoming sparsity/undersampling issues. Yet, privacy is still a major issue behind generating synthetic medical text, where more human assessments are needed to check for the existence of any sensitive information. Despite that, advances in generating synthetic medical text will considerably accelerate the adoption of workflows and pipeline development, discarding the time-consuming legalities of data transfer. 

---
# Digital Twin Technologies in Predictive Maintenance: Enabling Transferability via Sim-to-Real and Real-to-Sim Transfer 

**Authors**: Sizhe Ma, Katherine A. Flanigan, Mario Bergés  

**Link**: [PDF](https://arxiv.org/pdf/2507.18449)  

**Abstract**: The advancement of the Internet of Things (IoT) and Artificial Intelligence has catalyzed the evolution of Digital Twins (DTs) from conceptual ideas to more implementable realities. Yet, transitioning from academia to industry is complex due to the absence of standardized frameworks. This paper builds upon the authors' previously established functional and informational requirements supporting standardized DT development, focusing on a crucial aspect: transferability. While existing DT research primarily centers on asset transfer, the significance of "sim-to-real transfer" and "real-to-sim transfer"--transferring knowledge between simulations and real-world operations--is vital for comprehensive lifecycle management in DTs. A key challenge in this process is calibrating the "reality gap," the discrepancy between simulated predictions and actual outcomes. Our research investigates the impact of integrating a single Reality Gap Analysis (RGA) module into an existing DT framework to effectively manage both sim-to-real and real-to-sim transfers. This integration is facilitated by data pipelines that connect the RGA module with the existing components of the DT framework, including the historical repository and the simulation model. A case study on a pedestrian bridge at Carnegie Mellon University showcases the performance of different levels of integration of our approach with an existing framework. With full implementation of an RGA module and a complete data pipeline, our approach is capable of bidirectional knowledge transfer between simulations and real-world operations without compromising efficiency. 

---
# Restoring Rhythm: Punctuation Restoration Using Transformer Models for Bangla, a Low-Resource Language 

**Authors**: Md Obyedullahil Mamun, Md Adyelullahil Mamun, Arif Ahmad, Md. Imran Hossain Emu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18448)  

**Abstract**: Punctuation restoration enhances the readability of text and is critical for post-processing tasks in Automatic Speech Recognition (ASR), especially for low-resource languages like Bangla. In this study, we explore the application of transformer-based models, specifically XLM-RoBERTa-large, to automatically restore punctuation in unpunctuated Bangla text. We focus on predicting four punctuation marks: period, comma, question mark, and exclamation mark across diverse text domains. To address the scarcity of annotated resources, we constructed a large, varied training corpus and applied data augmentation techniques. Our best-performing model, trained with an augmentation factor of alpha = 0.20%, achieves an accuracy of 97.1% on the News test set, 91.2% on the Reference set, and 90.2% on the ASR set.
Results show strong generalization to reference and ASR transcripts, demonstrating the model's effectiveness in real-world, noisy scenarios. This work establishes a strong baseline for Bangla punctuation restoration and contributes publicly available datasets and code to support future research in low-resource NLP. 

---
# AraTable: Benchmarking LLMs' Reasoning and Understanding of Arabic Tabular Data 

**Authors**: Rana Alshaikh, Israa Alghanmi, Shelan Jeawak  

**Link**: [PDF](https://arxiv.org/pdf/2507.18442)  

**Abstract**: The cognitive and reasoning abilities of large language models (LLMs) have enabled remarkable progress in natural language processing. However, their performance in interpreting structured data, especially in tabular formats, remains limited. Although benchmarks for English tabular data are widely available, Arabic is still underrepresented because of the limited availability of public resources and its unique language features. To address this gap, we present AraTable, a novel and comprehensive benchmark designed to evaluate the reasoning and understanding capabilities of LLMs when applied to Arabic tabular data. AraTable consists of various evaluation tasks, such as direct question answering, fact verification, and complex reasoning, involving a wide range of Arabic tabular sources. Our methodology follows a hybrid pipeline, where initial content is generated by LLMs and subsequently filtered and verified by human experts to ensure high dataset quality. Initial analyses using AraTable show that, while LLMs perform adequately on simpler tabular tasks such as direct question answering, they continue to face significant cognitive challenges when tasks require deeper reasoning and fact verification. This indicates that there are substantial opportunities for future work to improve performance on complex tabular reasoning tasks. We also propose a fully automated evaluation framework that uses a self-deliberation mechanism and achieves performance nearly identical to that of human judges. This research provides a valuable, publicly available resource and evaluation framework that can help accelerate the development of foundational models for processing and analysing Arabic structured data. 

---
# CLEAR: Error Analysis via LLM-as-a-Judge Made Easy 

**Authors**: Asaf Yehudai, Lilach Eden, Yotam Perlitz, Roy Bar-Haim, Michal Shmueli-Scheuer  

**Link**: [PDF](https://arxiv.org/pdf/2507.18392)  

**Abstract**: The evaluation of Large Language Models (LLMs) increasingly relies on other LLMs acting as judges. However, current evaluation paradigms typically yield a single score or ranking, answering which model is better but not why. While essential for benchmarking, these top-level scores obscure the specific, actionable reasons behind a model's performance. To bridge this gap, we introduce CLEAR, an interactive, open-source package for LLM-based error analysis. CLEAR first generates per-instance textual feedback, then it creates a set of system-level error issues, and quantifies the prevalence of each identified issue. Our package also provides users with an interactive dashboard that allows for a comprehensive error analysis through aggregate visualizations, applies interactive filters to isolate specific issues or score ranges, and drills down to the individual instances that exemplify a particular behavioral pattern. We demonstrate CLEAR analysis for RAG and Math benchmarks, and showcase its utility through a user case study. 

---
# Improving Bird Classification with Primary Color Additives 

**Authors**: Ezhini Rasendiran R, Chandresh Kumar Maurya  

**Link**: [PDF](https://arxiv.org/pdf/2507.18334)  

**Abstract**: We address the problem of classifying bird species using their song recordings, a challenging task due to environmental noise, overlapping vocalizations, and missing labels. Existing models struggle with low-SNR or multi-species recordings. We hypothesize that birds can be classified by visualizing their pitch pattern, speed, and repetition, collectively called motifs. Deep learning models applied to spectrogram images help, but similar motifs across species cause confusion. To mitigate this, we embed frequency information into spectrograms using primary color additives. This enhances species distinction and improves classification accuracy. Our experiments show that the proposed approach achieves statistically significant gains over models without colorization and surpasses the BirdCLEF 2024 winner, improving F1 by 7.3%, ROC-AUC by 6.2%, and CMAP by 6.6%. These results demonstrate the effectiveness of incorporating frequency information via colorization. 

---
# A Concept for Efficient Scalability of Automated Driving Allowing for Technical, Legal, Cultural, and Ethical Differences 

**Authors**: Lars Ullrich, Michael Buchholz, Jonathan Petit, Klaus Dietmayer, Knut Graichen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18326)  

**Abstract**: Efficient scalability of automated driving (AD) is key to reducing costs, enhancing safety, conserving resources, and maximizing impact. However, research focuses on specific vehicles and context, while broad deployment requires scalability across various configurations and environments. Differences in vehicle types, sensors, actuators, but also traffic regulations, legal requirements, cultural dynamics, or even ethical paradigms demand high flexibility of data-driven developed capabilities. In this paper, we address the challenge of scalable adaptation of generic capabilities to desired systems and environments. Our concept follows a two-stage fine-tuning process. In the first stage, fine-tuning to the specific environment takes place through a country-specific reward model that serves as an interface between technological adaptations and socio-political requirements. In the second stage, vehicle-specific transfer learning facilitates system adaptation and governs the validation of design decisions. In sum, our concept offers a data-driven process that integrates both technological and socio-political aspects, enabling effective scalability across technical, legal, cultural, and ethical differences. 

---
# A Multi-Dataset Benchmark for Semi-Supervised Semantic Segmentation in ECG Delineation 

**Authors**: Minje Park, Jeonghwa Lim, Taehyung Yu, Sunghoon Joo  

**Link**: [PDF](https://arxiv.org/pdf/2507.18323)  

**Abstract**: Electrocardiogram (ECG) delineation, the segmentation of meaningful waveform features, is critical for clinical diagnosis. Despite recent advances using deep learning, progress has been limited by the scarcity of publicly available annotated datasets. Semi-supervised learning presents a promising solution by leveraging abundant unlabeled ECG data. In this study, we present the first systematic benchmark for semi-supervised semantic segmentation (SemiSeg) in ECG delineation. We curated and unified multiple public datasets, including previously underused sources, to support robust and diverse evaluation. We adopted five representative SemiSeg algorithms from computer vision, implemented them on two different architectures: the convolutional network and the transformer, and evaluated them in two different settings: in-domain and cross-domain. Additionally, we propose ECG-specific training configurations and augmentation strategies and introduce a standardized evaluation framework. Our results show that the transformer outperforms the convolutional network in semi-supervised ECG delineation. We anticipate that our benchmark will serve as a foundation for advancing semi-supervised ECG delineation methods and will facilitate further research in this domain. 

---
# LoRA-Leak: Membership Inference Attacks Against LoRA Fine-tuned Language Models 

**Authors**: Delong Ran, Xinlei He, Tianshuo Cong, Anyu Wang, Qi Li, Xiaoyun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18302)  

**Abstract**: Language Models (LMs) typically adhere to a "pre-training and fine-tuning" paradigm, where a universal pre-trained model can be fine-tuned to cater to various specialized domains. Low-Rank Adaptation (LoRA) has gained the most widespread use in LM fine-tuning due to its lightweight computational cost and remarkable performance. Because the proportion of parameters tuned by LoRA is relatively small, there might be a misleading impression that the LoRA fine-tuning data is invulnerable to Membership Inference Attacks (MIAs). However, we identify that utilizing the pre-trained model can induce more information leakage, which is neglected by existing MIAs. Therefore, we introduce LoRA-Leak, a holistic evaluation framework for MIAs against the fine-tuning datasets of LMs. LoRA-Leak incorporates fifteen membership inference attacks, including ten existing MIAs, and five improved MIAs that leverage the pre-trained model as a reference. In experiments, we apply LoRA-Leak to three advanced LMs across three popular natural language processing tasks, demonstrating that LoRA-based fine-tuned LMs are still vulnerable to MIAs (e.g., 0.775 AUC under conservative fine-tuning settings). We also applied LoRA-Leak to different fine-tuning settings to understand the resulting privacy risks. We further explore four defenses and find that only dropout and excluding specific LM layers during fine-tuning effectively mitigate MIA risks while maintaining utility. We highlight that under the "pre-training and fine-tuning" paradigm, the existence of the pre-trained model makes MIA a more severe risk for LoRA-based LMs. We hope that our findings can provide guidance on data privacy protection for specialized LM providers. 

---
# TCM-Tongue: A Standardized Tongue Image Dataset with Pathological Annotations for AI-Assisted TCM Diagnosis 

**Authors**: Xuebo Jin, Longfei Gao, Anshuo Tong, Zhengyang Chen, Jianlei Kong, Ning Sun, Huijun Ma, Qiang Wang, Yuting Bai, Tingli Su  

**Link**: [PDF](https://arxiv.org/pdf/2507.18288)  

**Abstract**: Traditional Chinese medicine (TCM) tongue diagnosis, while clinically valuable, faces standardization challenges due to subjective interpretation and inconsistent imaging protocols, compounded by the lack of large-scale, annotated datasets for AI development. To address this gap, we present the first specialized dataset for AI-driven TCM tongue diagnosis, comprising 6,719 high-quality images captured under standardized conditions and annotated with 20 pathological symptom categories (averaging 2.54 clinically validated labels per image, all verified by licensed TCM practitioners). The dataset supports multiple annotation formats (COCO, TXT, XML) for broad usability and has been benchmarked using nine deep learning models (YOLOv5/v7/v8 variants, SSD, and MobileNetV2) to demonstrate its utility for AI development. This resource provides a critical foundation for advancing reliable computational tools in TCM, bridging the data shortage that has hindered progress in the field, and facilitating the integration of AI into both research and clinical practice through standardized, high-quality diagnostic data. 

---
# Locate-and-Focus: Enhancing Terminology Translation in Speech Language Models 

**Authors**: Suhang Wu, Jialong Tang, Chengyi Yang, Pei Zhang, Baosong Yang, Junhui Li, Junfeng Yao, Min Zhang, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2507.18263)  

**Abstract**: Direct speech translation (ST) has garnered increasing attention nowadays, yet the accurate translation of terminology within utterances remains a great challenge. In this regard, current studies mainly concentrate on leveraging various translation knowledge into ST models. However, these methods often struggle with interference from irrelevant noise and can not fully utilize the translation knowledge. To address these issues, in this paper, we propose a novel Locate-and-Focus method for terminology translation. It first effectively locates the speech clips containing terminologies within the utterance to construct translation knowledge, minimizing irrelevant information for the ST model. Subsequently, it associates the translation knowledge with the utterance and hypothesis from both audio and textual modalities, allowing the ST model to better focus on translation knowledge during translation. Experimental results across various datasets demonstrate that our method effectively locates terminologies within utterances and enhances the success rate of terminology translation, while maintaining robust general translation performance. 

---
# ReSem3D: Refinable 3D Spatial Constraints via Fine-Grained Semantic Grounding for Generalizable Robotic Manipulation 

**Authors**: Chenyu Su, Weiwei Shang, Chen Qian, Fei Zhang, Shuang Cong  

**Link**: [PDF](https://arxiv.org/pdf/2507.18262)  

**Abstract**: Semantics-driven 3D spatial constraints align highlevel semantic representations with low-level action spaces, facilitating the unification of task understanding and execution in robotic manipulation. The synergistic reasoning of Multimodal Large Language Models (MLLMs) and Vision Foundation Models (VFMs) enables cross-modal 3D spatial constraint construction. Nevertheless, existing methods have three key limitations: (1) coarse semantic granularity in constraint modeling, (2) lack of real-time closed-loop planning, (3) compromised robustness in semantically diverse environments. To address these challenges, we propose ReSem3D, a unified manipulation framework for semantically diverse environments, leveraging the synergy between VFMs and MLLMs to achieve fine-grained visual grounding and dynamically constructs hierarchical 3D spatial constraints for real-time manipulation. Specifically, the framework is driven by hierarchical recursive reasoning in MLLMs, which interact with VFMs to automatically construct 3D spatial constraints from natural language instructions and RGB-D observations in two stages: part-level extraction and region-level refinement. Subsequently, these constraints are encoded as real-time optimization objectives in joint space, enabling reactive behavior to dynamic disturbances. Extensive simulation and real-world experiments are conducted in semantically rich household and sparse chemical lab environments. The results demonstrate that ReSem3D performs diverse manipulation tasks under zero-shot conditions, exhibiting strong adaptability and generalization. Code and videos at this https URL. 

---
# Exploiting Gaussian Agnostic Representation Learning with Diffusion Priors for Enhanced Infrared Small Target Detection 

**Authors**: Junyao Li, Yahao Lu, Xingyuan Guo, Xiaoyu Xian, Tiantian Wang, Yukai Shi  

**Link**: [PDF](https://arxiv.org/pdf/2507.18260)  

**Abstract**: Infrared small target detection (ISTD) plays a vital role in numerous practical applications. In pursuit of determining the performance boundaries, researchers employ large and expensive manual-labeling data for representation learning. Nevertheless, this approach renders the state-of-the-art ISTD methods highly fragile in real-world challenges. In this paper, we first study the variation in detection performance across several mainstream methods under various scarcity -- namely, the absence of high-quality infrared data -- that challenge the prevailing theories about practical ISTD. To address this concern, we introduce the Gaussian Agnostic Representation Learning. Specifically, we propose the Gaussian Group Squeezer, leveraging Gaussian sampling and compression for non-uniform quantization. By exploiting a diverse array of training samples, we enhance the resilience of ISTD models against various challenges. Then, we introduce two-stage diffusion models for real-world reconstruction. By aligning quantized signals closely with real-world distributions, we significantly elevate the quality and fidelity of the synthetic samples. Comparative evaluations against state-of-the-art detection methods in various scarcity scenarios demonstrate the efficacy of the proposed approach. 

---
# Multimodal Behavioral Patterns Analysis with Eye-Tracking and LLM-Based Reasoning 

**Authors**: Dongyang Guo, Yasmeen Abdrabou, Enkeleda Thaqi, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2507.18252)  

**Abstract**: Eye-tracking data reveals valuable insights into users' cognitive states but is difficult to analyze due to its structured, non-linguistic nature. While large language models (LLMs) excel at reasoning over text, they struggle with temporal and numerical data. This paper presents a multimodal human-AI collaborative framework designed to enhance cognitive pattern extraction from eye-tracking signals. The framework includes: (1) a multi-stage pipeline using horizontal and vertical segmentation alongside LLM reasoning to uncover latent gaze patterns; (2) an Expert-Model Co-Scoring Module that integrates expert judgment with LLM output to generate trust scores for behavioral interpretations; and (3) a hybrid anomaly detection module combining LSTM-based temporal modeling with LLM-driven semantic analysis. Our results across several LLMs and prompt strategies show improvements in consistency, interpretability, and performance, with up to 50% accuracy in difficulty prediction tasks. This approach offers a scalable, interpretable solution for cognitive modeling and has broad potential in adaptive learning, human-computer interaction, and educational analytics. 

---
# DepthDark: Robust Monocular Depth Estimation for Low-Light Environments 

**Authors**: Longjian Zeng, Zunjie Zhu, Rongfeng Lu, Ming Lu, Bolun Zheng, Chenggang Yan, Anke Xue  

**Link**: [PDF](https://arxiv.org/pdf/2507.18243)  

**Abstract**: In recent years, foundation models for monocular depth estimation have received increasing attention. Current methods mainly address typical daylight conditions, but their effectiveness notably decreases in low-light environments. There is a lack of robust foundational models for monocular depth estimation specifically designed for low-light scenarios. This largely stems from the absence of large-scale, high-quality paired depth datasets for low-light conditions and the effective parameter-efficient fine-tuning (PEFT) strategy. To address these challenges, we propose DepthDark, a robust foundation model for low-light monocular depth estimation. We first introduce a flare-simulation module and a noise-simulation module to accurately simulate the imaging process under nighttime conditions, producing high-quality paired depth datasets for low-light conditions. Additionally, we present an effective low-light PEFT strategy that utilizes illumination guidance and multiscale feature fusion to enhance the model's capability in low-light environments. Our method achieves state-of-the-art depth estimation performance on the challenging nuScenes-Night and RobotCar-Night datasets, validating its effectiveness using limited training data and computing resources. 

---
# From Individual Learning to Market Equilibrium: Correcting Structural and Parametric Biases in RL Simulations of Economic Models 

**Authors**: Zeqiang Zhang, Ruxin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18229)  

**Abstract**: The application of Reinforcement Learning (RL) to economic modeling reveals a fundamental conflict between the assumptions of equilibrium theory and the emergent behavior of learning agents. While canonical economic models assume atomistic agents act as `takers' of aggregate market conditions, a naive single-agent RL simulation incentivizes the agent to become a `manipulator' of its environment. This paper first demonstrates this discrepancy within a search-and-matching model with concave production, showing that a standard RL agent learns a non-equilibrium, monopsonistic policy. Additionally, we identify a parametric bias arising from the mismatch between economic discounting and RL's treatment of intertemporal costs. To address both issues, we propose a calibrated Mean-Field Reinforcement Learning framework that embeds a representative agent in a fixed macroeconomic field and adjusts the cost function to reflect economic opportunity costs. Our iterative algorithm converges to a self-consistent fixed point where the agent's policy aligns with the competitive equilibrium. This approach provides a tractable and theoretically sound methodology for modeling learning agents in economic systems within the broader domain of computational social science. 

---
# GenAI for Automotive Software Development: From Requirements to Wheels 

**Authors**: Nenad Petrovic, Fengjunjie Pan, Vahid Zolfaghari, Krzysztof Lebioda, Andre Schamschurko, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2507.18223)  

**Abstract**: This paper introduces a GenAI-empowered approach to automated development of automotive software, with emphasis on autonomous and Advanced Driver Assistance Systems (ADAS) capabilities. The process starts with requirements as input, while the main generated outputs are test scenario code for simulation environment, together with implementation of desired ADAS capabilities targeting hardware platform of the vehicle connected to testbench. Moreover, we introduce additional steps for requirements consistency checking leveraging Model-Driven Engineering (MDE). In the proposed workflow, Large Language Models (LLMs) are used for model-based summarization of requirements (Ecore metamodel, XMI model instance and OCL constraint creation), test scenario generation, simulation code (Python) and target platform code generation (C++). Additionally, Retrieval Augmented Generation (RAG) is adopted to enhance test scenario generation from autonomous driving regulations-related documents. Our approach aims shorter compliance and re-engineering cycles, as well as reduced development and testing time when it comes to ADAS-related capabilities. 

---
# FedSA-GCL: A Semi-Asynchronous Federated Graph Learning Framework with Personalized Aggregation and Cluster-Aware Broadcasting 

**Authors**: Zhongzheng Yuan, Lianshuai Guo, Xunkai Li, Yinlin Zhu, Wenyu Wang, Meixia Qu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18219)  

**Abstract**: Federated Graph Learning (FGL) is a distributed learning paradigm that enables collaborative training over large-scale subgraphs located on multiple local systems. However, most existing FGL approaches rely on synchronous communication, which leads to inefficiencies and is often impractical in real-world deployments. Meanwhile, current asynchronous federated learning (AFL) methods are primarily designed for conventional tasks such as image classification and natural language processing, without accounting for the unique topological properties of graph data. Directly applying these methods to graph learning can possibly result in semantic drift and representational inconsistency in the global model. To address these challenges, we propose FedSA-GCL, a semi-asynchronous federated framework that leverages both inter-client label distribution divergence and graph topological characteristics through a novel ClusterCast mechanism for efficient training. We evaluate FedSA-GCL on multiple real-world graph datasets using the Louvain and Metis split algorithms, and compare it against 9 baselines. Extensive experiments demonstrate that our method achieves strong robustness and outstanding efficiency, outperforming the baselines by an average of 2.92% with the Louvain and by 3.4% with the Metis. 

---
# Information Security Based on LLM Approaches: A Review 

**Authors**: Chang Gong, Zhongwen Li, Xiaoqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.18215)  

**Abstract**: Information security is facing increasingly severe challenges, and traditional protection means are difficult to cope with complex and changing threats. In recent years, as an emerging intelligent technology, large language models (LLMs) have shown a broad application prospect in the field of information security. In this paper, we focus on the key role of LLM in information security, systematically review its application progress in malicious behavior prediction, network threat analysis, system vulnerability detection, malicious code identification, and cryptographic algorithm optimization, and explore its potential in enhancing security protection performance. Based on neural networks and Transformer architecture, this paper analyzes the technical basis of large language models and their advantages in natural language processing tasks. It is shown that the introduction of large language modeling helps to improve the detection accuracy and reduce the false alarm rate of security systems. Finally, this paper summarizes the current application results and points out that it still faces challenges in model transparency, interpretability, and scene adaptability, among other issues. It is necessary to explore further the optimization of the model structure and the improvement of the generalization ability to realize a more intelligent and accurate information security protection system. 

---
# MoRPI-PINN: A Physics-Informed Framework for Mobile Robot Pure Inertial Navigation 

**Authors**: Arup Kumar Sahoo, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2507.18206)  

**Abstract**: A fundamental requirement for full autonomy in mobile robots is accurate navigation even in situations where satellite navigation or cameras are unavailable. In such practical situations, relying only on inertial sensors will result in navigation solution drift due to the sensors' inherent noise and error terms. One of the emerging solutions to mitigate drift is to maneuver the robot in a snake-like slithering motion to increase the inertial signal-to-noise ratio, allowing the regression of the mobile robot position. In this work, we propose MoRPI-PINN as a physics-informed neural network framework for accurate inertial-based mobile robot navigation. By embedding physical laws and constraints into the training process, MoRPI-PINN is capable of providing an accurate and robust navigation solution. Using real-world experiments, we show accuracy improvements of over 85% compared to other approaches. MoRPI-PINN is a lightweight approach that can be implemented even on edge devices and used in any typical mobile robot application. 

---
# Safeguarding RAG Pipelines with GMTP: A Gradient-based Masked Token Probability Method for Poisoned Document Detection 

**Authors**: San Kim, Jonghwi Kim, Yejin Jeon, Gary Geunbae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2507.18202)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by providing external knowledge for accurate and up-to-date responses. However, this reliance on external sources exposes a security risk, attackers can inject poisoned documents into the knowledge base to steer the generation process toward harmful or misleading outputs. In this paper, we propose Gradient-based Masked Token Probability (GMTP), a novel defense method to detect and filter out adversarially crafted documents. Specifically, GMTP identifies high-impact tokens by examining gradients of the retriever's similarity function. These key tokens are then masked, and their probabilities are checked via a Masked Language Model (MLM). Since injected tokens typically exhibit markedly low masked-token probabilities, this enables GMTP to easily detect malicious documents and achieve high-precision filtering. Experiments demonstrate that GMTP is able to eliminate over 90% of poisoned content while retaining relevant documents, thus maintaining robust retrieval and generation performance across diverse datasets and adversarial settings. 

---
# SCOPE: Stochastic and Counterbiased Option Placement for Evaluating Large Language Models 

**Authors**: Wonjun Jeong, Dongseok Kim, Taegkeun Whangbo  

**Link**: [PDF](https://arxiv.org/pdf/2507.18182)  

**Abstract**: Large Language Models (LLMs) can achieve inflated scores on multiple-choice tasks by exploiting inherent biases in option positions or labels, rather than demonstrating genuine understanding. This study introduces SCOPE, an evaluation framework designed to measure and mitigate such selection bias in a dataset-independent manner. By repeatedly invoking a null prompt that lacks semantic content, SCOPE estimates each model's unique position-bias distribution. It then redistributes the answer slot according to the inverse-bias distribution, thereby equalizing the lucky-rate, the probability of selecting the correct answer by chance. Furthermore, it prevents semantically similar distractors from being placed adjacent to the answer, thereby blocking near-miss guesses based on superficial proximity cues. Across multiple benchmark experiments, SCOPE consistently outperformed existing debiasing methods in terms of stable performance improvements and showed clearer confidence distributions over correct options. This framework thus offers a new standard for enhancing the fairness and reliability of LLM evaluations. 

---
# Differential-UMamba: Rethinking Tumor Segmentation Under Limited Data Scenarios 

**Authors**: Dhruv Jain, Romain Modzelewski, Romain Hérault, Clement Chatelain, Eva Torfeh, Sebastien Thureau  

**Link**: [PDF](https://arxiv.org/pdf/2507.18177)  

**Abstract**: In data-scarce scenarios, deep learning models often overfit to noise and irrelevant patterns, which limits their ability to generalize to unseen samples. To address these challenges in medical image segmentation, we introduce Diff-UMamba, a novel architecture that combines the UNet framework with the mamba mechanism for modeling long-range dependencies. At the heart of Diff-UMamba is a Noise Reduction Module (NRM), which employs a signal differencing strategy to suppress noisy or irrelevant activations within the encoder. This encourages the model to filter out spurious features and enhance task-relevant representations, thereby improving its focus on clinically meaningful regions. As a result, the architecture achieves improved segmentation accuracy and robustness, particularly in low-data settings. Diff-UMamba is evaluated on multiple public datasets, including MSD (lung and pancreas) and AIIB23, demonstrating consistent performance gains of 1-3% over baseline methods across diverse segmentation tasks. To further assess performance under limited-data conditions, additional experiments are conducted on the BraTS-21 dataset by varying the proportion of available training samples. The approach is also validated on a small internal non-small cell lung cancer (NSCLC) dataset for gross tumor volume (GTV) segmentation in cone beam CT (CBCT), where it achieves a 4-5% improvement over the baseline. 

---
# Sticking to the Mean: Detecting Sticky Tokens in Text Embedding Models 

**Authors**: Kexin Chen, Dongxia Wang, Yi Liu, Haonan Zhang, Wenhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18171)  

**Abstract**: Despite the widespread use of Transformer-based text embedding models in NLP tasks, surprising 'sticky tokens' can undermine the reliability of embeddings. These tokens, when repeatedly inserted into sentences, pull sentence similarity toward a certain value, disrupting the normal distribution of embedding distances and degrading downstream performance. In this paper, we systematically investigate such anomalous tokens, formally defining them and introducing an efficient detection method, Sticky Token Detector (STD), based on sentence and token filtering. Applying STD to 40 checkpoints across 14 model families, we discover a total of 868 sticky tokens. Our analysis reveals that these tokens often originate from special or unused entries in the vocabulary, as well as fragmented subwords from multilingual corpora. Notably, their presence does not strictly correlate with model size or vocabulary size. We further evaluate how sticky tokens affect downstream tasks like clustering and retrieval, observing significant performance drops of up to 50%. Through attention-layer analysis, we show that sticky tokens disproportionately dominate the model's internal representations, raising concerns about tokenization robustness. Our findings show the need for better tokenization strategies and model design to mitigate the impact of sticky tokens in future text embedding applications. 

---
# When Noisy Labels Meet Class Imbalance on Graphs: A Graph Augmentation Method with LLM and Pseudo Label 

**Authors**: Riting Xia, Rucong Wang, Yulin Liu, Anchen Li, Xueyan Liu, Yan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18153)  

**Abstract**: Class-imbalanced graph node classification is a practical yet underexplored research problem. Although recent studies have attempted to address this issue, they typically assume clean and reliable labels when processing class-imbalanced graphs. This assumption often violates the nature of real-world graphs, where labels frequently contain noise. Given this gap, this paper systematically investigates robust node classification for class-imbalanced graphs with noisy labels. We propose GraphALP, a novel Graph Augmentation framework based on Large language models (LLMs) and Pseudo-labeling techniques. Specifically, we design an LLM-based oversampling method to generate synthetic minority nodes, producing label-accurate minority nodes to alleviate class imbalance. Based on the class-balanced graphs, we develop a dynamically weighted pseudo-labeling method to obtain high-confidence pseudo labels to reduce label noise ratio. Additionally, we implement a secondary LLM-guided oversampling mechanism to mitigate potential class distribution skew caused by pseudo labels. Experimental results show that GraphALP achieves superior performance over state-of-the-art methods on class-imbalanced graphs with noisy labels. 

---
# HIVMedQA: Benchmarking large language models for HIV medical decision support 

**Authors**: Gonzalo Cardenal Antolin, Jacques Fellay, Bashkim Jaha, Roger Kouyos, Niko Beerenwinkel, Diane Duroux  

**Link**: [PDF](https://arxiv.org/pdf/2507.18143)  

**Abstract**: Large language models (LLMs) are emerging as valuable tools to support clinicians in routine decision-making. HIV management is a compelling use case due to its complexity, including diverse treatment options, comorbidities, and adherence challenges. However, integrating LLMs into clinical practice raises concerns about accuracy, potential harm, and clinician acceptance. Despite their promise, AI applications in HIV care remain underexplored, and LLM benchmarking studies are scarce. This study evaluates the current capabilities of LLMs in HIV management, highlighting their strengths and limitations. We introduce HIVMedQA, a benchmark designed to assess open-ended medical question answering in HIV care. The dataset consists of curated, clinically relevant questions developed with input from an infectious disease physician. We evaluated seven general-purpose and three medically specialized LLMs, applying prompt engineering to enhance performance. Our evaluation framework incorporates both lexical similarity and an LLM-as-a-judge approach, extended to better reflect clinical relevance. We assessed performance across key dimensions: question comprehension, reasoning, knowledge recall, bias, potential harm, and factual accuracy. Results show that Gemini 2.5 Pro consistently outperformed other models across most dimensions. Notably, two of the top three models were proprietary. Performance declined as question complexity increased. Medically fine-tuned models did not always outperform general-purpose ones, and larger model size was not a reliable predictor of performance. Reasoning and comprehension were more challenging than factual recall, and cognitive biases such as recency and status quo were observed. These findings underscore the need for targeted development and evaluation to ensure safe, effective LLM integration in clinical care. 

---
# Deep Learning for Glioblastoma Morpho-pathological Features Identification: A BraTS-Pathology Challenge Solution 

**Authors**: Juexin Zhang, Ying Weng, Ke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18133)  

**Abstract**: Glioblastoma, a highly aggressive brain tumor with diverse molecular and pathological features, poses a diagnostic challenge due to its heterogeneity. Accurate diagnosis and assessment of this heterogeneity are essential for choosing the right treatment and improving patient outcomes. Traditional methods rely on identifying specific features in tissue samples, but deep learning offers a promising approach for improved glioblastoma diagnosis. In this paper, we present our approach to the BraTS-Path Challenge 2024. We leverage a pre-trained model and fine-tune it on the BraTS-Path training dataset. Our model demonstrates poor performance on the challenging BraTS-Path validation set, as rigorously assessed by the Synapse online platform. The model achieves an accuracy of 0.392229, a recall of 0.392229, and a F1-score of 0.392229, indicating a consistent ability to correctly identify instances under the target condition. Notably, our model exhibits perfect specificity of 0.898704, showing an exceptional capacity to correctly classify negative cases. Moreover, a Matthews Correlation Coefficient (MCC) of 0.255267 is calculated, to signify a limited positive correlation between predicted and actual values and highlight our model's overall predictive power. Our solution also achieves the second place during the testing phase. 

---
# U-Net Based Healthy 3D Brain Tissue Inpainting 

**Authors**: Juexin Zhang, Ying Weng, Ke Chen  

**Link**: [PDF](https://arxiv.org/pdf/2507.18126)  

**Abstract**: This paper introduces a novel approach to synthesize healthy 3D brain tissue from masked input images, specifically focusing on the task of 'ASNR-MICCAI BraTS Local Synthesis of Tissue via Inpainting'. Our proposed method employs a U-Net-based architecture, which is designed to effectively reconstruct the missing or corrupted regions of brain MRI scans. To enhance our model's generalization capabilities and robustness, we implement a comprehensive data augmentation strategy that involves randomly masking healthy images during training. Our model is trained on the BraTS-Local-Inpainting dataset and demonstrates the exceptional performance in recovering healthy brain tissue. The evaluation metrics employed, including Structural Similarity Index (SSIM), Peak Signal-to-Noise Ratio (PSNR), and Mean Squared Error (MSE), consistently yields impressive results. On the BraTS-Local-Inpainting validation set, our model achieved an SSIM score of 0.841, a PSNR score of 23.257, and an MSE score of 0.007. Notably, these evaluation metrics exhibit relatively low standard deviations, i.e., 0.103 for SSIM score, 4.213 for PSNR score and 0.007 for MSE score, which indicates that our model's reliability and consistency across various input scenarios. Our method also secured first place in the challenge. 

---
# GOAT-SLM: A Spoken Language Model with Paralinguistic and Speaker Characteristic Awareness 

**Authors**: Hongjie Chen, Zehan Li, Yaodong Song, Wenming Deng, Yitong Yao, Yuxin Zhang, Hang Lv, Xuechao Zhu, Jian Kang, Jie Lian, Jie Li, Chao Wang, Shuangyong Song, Yongxiang Li, Zhongjiang He  

**Link**: [PDF](https://arxiv.org/pdf/2507.18119)  

**Abstract**: Recent advances in end-to-end spoken language models (SLMs) have significantly improved the ability of AI systems to engage in natural spoken interactions. However, most existing models treat speech merely as a vehicle for linguistic content, often overlooking the rich paralinguistic and speaker characteristic cues embedded in human speech, such as dialect, age, emotion, and non-speech vocalizations. In this work, we introduce GOAT-SLM, a novel spoken language model with paralinguistic and speaker characteristic awareness, designed to extend spoken language modeling beyond text semantics. GOAT-SLM adopts a dual-modality head architecture that decouples linguistic modeling from acoustic realization, enabling robust language understanding while supporting expressive and adaptive speech generation. To enhance model efficiency and versatility, we propose a modular, staged training strategy that progressively aligns linguistic, paralinguistic, and speaker characteristic information using large-scale speech-text corpora. Experimental results on TELEVAL, a multi-dimensional evaluation benchmark, demonstrate that GOAT-SLM achieves well-balanced performance across both semantic and non-semantic tasks, and outperforms existing open-source models in handling emotion, dialectal variation, and age-sensitive interactions. This work highlights the importance of modeling beyond linguistic content and advances the development of more natural, adaptive, and socially aware spoken language systems. 

---
# Parameter-Efficient Fine-Tuning of 3D DDPM for MRI Image Generation Using Tensor Networks 

**Authors**: Binghua Li, Ziqing Chang, Tong Liang, Chao Li, Toshihisa Tanaka, Shigeki Aoki, Qibin Zhao, Zhe Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.18112)  

**Abstract**: We address the challenge of parameter-efficient fine-tuning (PEFT) for three-dimensional (3D) U-Net-based denoising diffusion probabilistic models (DDPMs) in magnetic resonance imaging (MRI) image generation. Despite its practical significance, research on parameter-efficient representations of 3D convolution operations remains limited. To bridge this gap, we propose Tensor Volumetric Operator (TenVOO), a novel PEFT method specifically designed for fine-tuning DDPMs with 3D convolutional backbones. Leveraging tensor network modeling, TenVOO represents 3D convolution kernels with lower-dimensional tensors, effectively capturing complex spatial dependencies during fine-tuning with few parameters. We evaluate TenVOO on three downstream brain MRI datasets-ADNI, PPMI, and BraTS2021-by fine-tuning a DDPM pretrained on 59,830 T1-weighted brain MRI scans from the UK Biobank. Our results demonstrate that TenVOO achieves state-of-the-art performance in multi-scale structural similarity index measure (MS-SSIM), outperforming existing approaches in capturing spatial dependencies while requiring only 0.3% of the trainable parameters of the original model. Our code is available at: this https URL 

---
# Distributional Uncertainty for Out-of-Distribution Detection 

**Authors**: JinYoung Kim, DaeUng Jo, Kimin Yun, Jeonghyo Song, Youngjoon Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2507.18106)  

**Abstract**: Estimating uncertainty from deep neural networks is a widely used approach for detecting out-of-distribution (OoD) samples, which typically exhibit high predictive uncertainty. However, conventional methods such as Monte Carlo (MC) Dropout often focus solely on either model or data uncertainty, failing to align with the semantic objective of OoD detection. To address this, we propose the Free-Energy Posterior Network, a novel framework that jointly models distributional uncertainty and identifying OoD and misclassified regions using free energy. Our method introduces two key contributions: (1) a free-energy-based density estimator parameterized by a Beta distribution, which enables fine-grained uncertainty estimation near ambiguous or unseen regions; and (2) a loss integrated within a posterior network, allowing direct uncertainty estimation from learned parameters without requiring stochastic sampling. By integrating our approach with the residual prediction branch (RPL) framework, the proposed method goes beyond post-hoc energy thresholding and enables the network to learn OoD regions by leveraging the variance of the Beta distribution, resulting in a semantically meaningful and computationally efficient solution for uncertainty-aware segmentation. We validate the effectiveness of our method on challenging real-world benchmarks, including Fishyscapes, RoadAnomaly, and Segment-Me-If-You-Can. 

---
# Datasets and Recipes for Video Temporal Grounding via Reinforcement Learning 

**Authors**: Ruizhe Chen, Zhiting Fan, Tianze Luo, Heqing Zou, Zhaopeng Feng, Guiyang Xie, Hansheng Zhang, Zhuochen Wang, Zuozhu Liu, Huaijian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18100)  

**Abstract**: Video Temporal Grounding (VTG) aims to localize relevant temporal segments in videos given natural language queries. Despite recent progress with large vision-language models (LVLMs) and instruction-tuning, existing approaches often suffer from limited temporal awareness and poor generalization. In this work, we introduce a two-stage training framework that integrates supervised fine-tuning with reinforcement learning (RL) to improve both the accuracy and robustness of VTG models. Our approach first leverages high-quality curated cold start data for SFT initialization, followed by difficulty-controlled RL to further enhance temporal localization and reasoning abilities. Comprehensive experiments on multiple VTG benchmarks demonstrate that our method consistently outperforms existing models, particularly in challenging and open-domain scenarios. We conduct an in-depth analysis of training strategies and dataset curation, highlighting the importance of both high-quality cold start data and difficulty-controlled RL. To facilitate further research and industrial adoption, we release all intermediate datasets, models, and code to the community. 

---
# TextSAM-EUS: Text Prompt Learning for SAM to Accurately Segment Pancreatic Tumor in Endoscopic Ultrasound 

**Authors**: Pascal Spiegler, Taha Koleilat, Arash Harirpoush, Corey S. Miller, Hassan Rivaz, Marta Kersten-Oertel, Yiming Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2507.18082)  

**Abstract**: Pancreatic cancer carries a poor prognosis and relies on endoscopic ultrasound (EUS) for targeted biopsy and radiotherapy. However, the speckle noise, low contrast, and unintuitive appearance of EUS make segmentation of pancreatic tumors with fully supervised deep learning (DL) models both error-prone and dependent on large, expert-curated annotation datasets. To address these challenges, we present TextSAM-EUS, a novel, lightweight, text-driven adaptation of the Segment Anything Model (SAM) that requires no manual geometric prompts at inference. Our approach leverages text prompt learning (context optimization) through the BiomedCLIP text encoder in conjunction with a LoRA-based adaptation of SAM's architecture to enable automatic pancreatic tumor segmentation in EUS, tuning only 0.86% of the total parameters. On the public Endoscopic Ultrasound Database of the Pancreas, TextSAM-EUS with automatic prompts attains 82.69% Dice and 85.28% normalized surface distance (NSD), and with manual geometric prompts reaches 83.10% Dice and 85.70% NSD, outperforming both existing state-of-the-art (SOTA) supervised DL models and foundation models (e.g., SAM and its variants). As the first attempt to incorporate prompt learning in SAM-based medical image segmentation, TextSAM-EUS offers a practical option for efficient and robust automatic EUS segmentation. Our code will be publicly available upon acceptance. 

---
# Group Sequence Policy Optimization 

**Authors**: Chujie Zheng, Shixuan Liu, Mingze Li, Xiong-Hui Chen, Bowen Yu, Chang Gao, Kai Dang, Yuqiong Liu, Rui Men, An Yang, Jingren Zhou, Junyang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2507.18071)  

**Abstract**: This paper introduces Group Sequence Policy Optimization (GSPO), our stable, efficient, and performant reinforcement learning algorithm for training large language models. Unlike previous algorithms that adopt token-level importance ratios, GSPO defines the importance ratio based on sequence likelihood and performs sequence-level clipping, rewarding, and optimization. We demonstrate that GSPO achieves superior training efficiency and performance compared to the GRPO algorithm, notably stabilizes Mixture-of-Experts (MoE) RL training, and has the potential for simplifying the design of RL infrastructure. These merits of GSPO have contributed to the remarkable improvements in the latest Qwen3 models. 

---
# TELEVAL: A Dynamic Benchmark Designed for Spoken Language Models in Chinese Interactive Scenarios 

**Authors**: Zehan Li, Hongjie Chen, Yuxin Zhang, Jing Zhou, Xuening Wang, Hang Lv, Mengjie Du, Yaodong Song, Jie Lian, Jian Kang, Jie Li, Yongxiang Li, Zhongjiang He, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2507.18061)  

**Abstract**: Spoken language models (SLMs) have seen rapid progress in recent years, along with the development of numerous benchmarks for evaluating their performance. However, most existing benchmarks primarily focus on evaluating whether SLMs can perform complex tasks comparable to those tackled by large language models (LLMs), often failing to align with how users naturally interact in real-world conversational scenarios. In this paper, we propose TELEVAL, a dynamic benchmark specifically designed to evaluate SLMs' effectiveness as conversational agents in realistic Chinese interactive settings. TELEVAL defines three evaluation dimensions: Explicit Semantics, Paralinguistic and Implicit Semantics, and System Abilities. It adopts a dialogue format consistent with real-world usage and evaluates text and audio outputs separately. TELEVAL particularly focuses on the model's ability to extract implicit cues from user speech and respond appropriately without additional instructions. Our experiments demonstrate that despite recent progress, existing SLMs still have considerable room for improvement in natural conversational tasks. We hope that TELEVAL can serve as a user-centered evaluation framework that directly reflects the user experience and contributes to the development of more capable dialogue-oriented SLMs. 

---
# Enhancing Scene Transition Awareness in Video Generation via Post-Training 

**Authors**: Hanwen Shen, Jiajie Lu, Yupeng Cao, Xiaonan Yang  

**Link**: [PDF](https://arxiv.org/pdf/2507.18046)  

**Abstract**: Recent advances in AI-generated video have shown strong performance on \emph{text-to-video} tasks, particularly for short clips depicting a single scene. However, current models struggle to generate longer videos with coherent scene transitions, primarily because they cannot infer when a transition is needed from the prompt. Most open-source models are trained on datasets consisting of single-scene video clips, which limits their capacity to learn and respond to prompts requiring multiple scenes. Developing scene transition awareness is essential for multi-scene generation, as it allows models to identify and segment videos into distinct clips by accurately detecting transitions.
To address this, we propose the \textbf{Transition-Aware Video} (TAV) dataset, which consists of preprocessed video clips with multiple scene transitions. Our experiment shows that post-training on the \textbf{TAV} dataset improves prompt-based scene transition understanding, narrows the gap between required and generated scenes, and maintains image quality. 

---
# Synthetic Data Generation for Phrase Break Prediction with Large Language Model 

**Authors**: Hoyeon Lee, Sejung Son, Ye-Eun Kang, Jong-Hwan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2507.18044)  

**Abstract**: Current approaches to phrase break prediction address crucial prosodic aspects of text-to-speech systems but heavily rely on vast human annotations from audio or text, incurring significant manual effort and cost. Inherent variability in the speech domain, driven by phonetic factors, further complicates acquiring consistent, high-quality data. Recently, large language models (LLMs) have shown success in addressing data challenges in NLP by generating tailored synthetic data while reducing manual annotation needs. Motivated by this, we explore leveraging LLM to generate synthetic phrase break annotations, addressing the challenges of both manual annotation and speech-related tasks by comparing with traditional annotations and assessing effectiveness across multiple languages. Our findings suggest that LLM-based synthetic data generation effectively mitigates data challenges in phrase break prediction and highlights the potential of LLMs as a viable solution for the speech domain. 

---
# GrAInS: Gradient-based Attribution for Inference-Time Steering of LLMs and VLMs 

**Authors**: Duy Nguyen, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2507.18043)  

**Abstract**: Inference-time steering methods offer a lightweight alternative to fine-tuning large language models (LLMs) and vision-language models (VLMs) by modifying internal activations at test time without updating model weights. However, most existing approaches rely on fixed, global intervention vectors, overlook the causal influence of individual input tokens, and fail to leverage informative gradients from the model's logits, particularly in multimodal settings where visual and textual inputs contribute unevenly. To address these limitations, we introduce GrAInS, an inference-time steering approach that operates across both language-only and vision-language models and tasks. GrAInS uses contrastive, gradient-based attribution via Integrated Gradients to identify the top-k most influential tokens, both positively and negatively attributed based on their contribution to preferred versus dispreferred outputs. These tokens are then used to construct directional steering vectors that capture semantic shifts from undesirable to desirable behavior. During inference, GrAInS adjusts hidden activations at transformer layers guided by token-level attribution signals, and normalizes activations to preserve representational scale. This enables fine-grained, interpretable, and modular control over model behavior, without retraining or auxiliary supervision. Empirically, GrAInS consistently outperforms both fine-tuning and existing steering baselines: it achieves a 13.22% accuracy gain on TruthfulQA using Llama-3.1-8B, reduces hallucination rates on MMHal-Bench from 0.624 to 0.514 with LLaVA-1.6-7B, and improves alignment win rates on SPA-VL by 8.11%, all while preserving the model's fluency and general capabilities. 

---
# OpenNav: Open-World Navigation with Multimodal Large Language Models 

**Authors**: Mingfeng Yuan, Letian Wang, Steven L. Waslander  

**Link**: [PDF](https://arxiv.org/pdf/2507.18033)  

**Abstract**: Pre-trained large language models (LLMs) have demonstrated strong common-sense reasoning abilities, making them promising for robotic navigation and planning tasks. However, despite recent progress, bridging the gap between language descriptions and actual robot actions in the open-world, beyond merely invoking limited predefined motion primitives, remains an open challenge. In this work, we aim to enable robots to interpret and decompose complex language instructions, ultimately synthesizing a sequence of trajectory points to complete diverse navigation tasks given open-set instructions and open-set objects. We observe that multi-modal large language models (MLLMs) exhibit strong cross-modal understanding when processing free-form language instructions, demonstrating robust scene comprehension. More importantly, leveraging their code-generation capability, MLLMs can interact with vision-language perception models to generate compositional 2D bird-eye-view value maps, effectively integrating semantic knowledge from MLLMs with spatial information from maps to reinforce the robot's spatial understanding. To further validate our approach, we effectively leverage large-scale autonomous vehicle datasets (AVDs) to validate our proposed zero-shot vision-language navigation framework in outdoor navigation tasks, demonstrating its capability to execute a diverse range of free-form natural language navigation instructions while maintaining robustness against object detection errors and linguistic ambiguities. Furthermore, we validate our system on a Husky robot in both indoor and outdoor scenes, demonstrating its real-world robustness and applicability. Supplementary videos are available at this https URL 

---
# ViGText: Deepfake Image Detection with Vision-Language Model Explanations and Graph Neural Networks 

**Authors**: Ahmad ALBarqawi, Mahmoud Nazzal, Issa Khalil, Abdallah Khreishah, NhatHai Phan  

**Link**: [PDF](https://arxiv.org/pdf/2507.18031)  

**Abstract**: The rapid rise of deepfake technology, which produces realistic but fraudulent digital content, threatens the authenticity of media. Traditional deepfake detection approaches often struggle with sophisticated, customized deepfakes, especially in terms of generalization and robustness against malicious attacks. This paper introduces ViGText, a novel approach that integrates images with Vision Large Language Model (VLLM) Text explanations within a Graph-based framework to improve deepfake detection. The novelty of ViGText lies in its integration of detailed explanations with visual data, as it provides a more context-aware analysis than captions, which often lack specificity and fail to reveal subtle inconsistencies. ViGText systematically divides images into patches, constructs image and text graphs, and integrates them for analysis using Graph Neural Networks (GNNs) to identify deepfakes. Through the use of multi-level feature extraction across spatial and frequency domains, ViGText captures details that enhance its robustness and accuracy to detect sophisticated deepfakes. Extensive experiments demonstrate that ViGText significantly enhances generalization and achieves a notable performance boost when it detects user-customized deepfakes. Specifically, average F1 scores rise from 72.45% to 98.32% under generalization evaluation, and reflects the model's superior ability to generalize to unseen, fine-tuned variations of stable diffusion models. As for robustness, ViGText achieves an increase of 11.1% in recall compared to other deepfake detection approaches. When facing targeted attacks that exploit its graph-based architecture, ViGText limits classification performance degradation to less than 4%. ViGText uses detailed visual and textual analysis to set a new standard for detecting deepfakes, helping ensure media authenticity and information integrity. 

---
# NeuralDB: Scaling Knowledge Editing in LLMs to 100,000 Facts with Neural KV Database 

**Authors**: Weizhi Fei, Hao Shi, Jing Xu, Jingchen Peng, Jiazheng Li, Jingzhao Zhang, Bo Bai, Wei Han, Zhenyuan Chen, Xueyan Niu  

**Link**: [PDF](https://arxiv.org/pdf/2507.18028)  

**Abstract**: Efficiently editing knowledge stored in large language models (LLMs) enables model updates without large-scale training. One possible solution is Locate-and-Edit (L\&E), allowing simultaneous modifications of a massive number of facts. However, such editing may compromise the general abilities of LLMs and even result in forgetting edited facts when scaling up to thousands of edits. In this paper, we model existing linear L\&E methods as querying a Key-Value (KV) database. From this perspective, we then propose NeuralDB, an editing framework that explicitly represents the edited facts as a neural KV database equipped with a non-linear gated retrieval module, % In particular, our gated module only operates when inference involves the edited facts, effectively preserving the general abilities of LLMs. Comprehensive experiments involving the editing of 10,000 facts were conducted on the ZsRE and CounterFacts datasets, using GPT2-XL, GPT-J (6B) and Llama-3 (8B). The results demonstrate that NeuralDB not only excels in editing efficacy, generalization, specificity, fluency, and consistency, but also preserves overall performance across six representative text understanding and generation tasks. Further experiments indicate that NeuralDB maintains its effectiveness even when scaled to 100,000 facts (\textbf{50x} more than in prior work). 

---
# Fashion-AlterEval: A Dataset for Improved Evaluation of Conversational Recommendation Systems with Alternative Relevant Items 

**Authors**: Maria Vlachou  

**Link**: [PDF](https://arxiv.org/pdf/2507.18017)  

**Abstract**: In Conversational Recommendation Systems (CRS), a user provides feedback on recommended items at each turn, leading the CRS towards improved recommendations. Due to the need for a large amount of data, a user simulator is employed for both training and evaluation. Such user simulators critique the current retrieved item based on knowledge of a single target item. However, system evaluation in offline settings with simulators is limited by the focus on a single target item and their unlimited patience over a large number of turns. To overcome these limitations of existing simulators, we propose Fashion-AlterEval, a new dataset that contains human judgments for a selection of alternative items by adding new annotations in common fashion CRS datasets. Consequently, we propose two novel meta-user simulators that use the collected judgments and allow simulated users not only to express their preferences about alternative items to their original target, but also to change their mind and level of patience. In our experiments using the Shoes and Fashion IQ as the original datasets and three CRS models, we find that using the knowledge of alternatives by the simulator can have a considerable impact on the evaluation of existing CRS models, specifically that the existing single-target evaluation underestimates their effectiveness, and when simulatedusers are allowed to instead consider alternative relevant items, the system can rapidly respond to more quickly satisfy the user. 

---
# GRR-CoCa: Leveraging LLM Mechanisms in Multimodal Model Architectures 

**Authors**: Jake R. Patock, Nicole Catherine Lewis, Kevin McCoy, Christina Gomez, Canling Chen, Lorenzo Luzi  

**Link**: [PDF](https://arxiv.org/pdf/2507.18009)  

**Abstract**: State-of-the-art (SOTA) image and text generation models are multimodal models that have many similarities to large language models (LLMs). Despite achieving strong performances, leading foundational multimodal model architectures frequently lag behind the architectural sophistication of contemporary LLMs. We propose GRR-CoCa, an improved SOTA Contrastive Captioner (CoCa) model that incorporates Gaussian error gated linear units, root mean squared normalization, and rotary positional embedding into the textual decoders and the vision transformer (ViT) encoder. Each architectural modification has been shown to improve model performance in LLMs, but has yet to be adopted in CoCa. We benchmarked GRR-CoCa against Baseline CoCa, a model with the same modified textual decoders but with CoCa's original ViT encoder. We used standard pretraining and fine-tuning workflows to benchmark the models on contrastive and generative tasks. Our GRR-CoCa significantly outperformed Baseline CoCa on the pretraining dataset and three diverse fine-tuning datasets. Pretraining improvements were 27.25% in contrastive loss, 3.71% in perplexity, and 7.15% in CoCa loss. The average fine-tuning improvements were 13.66% in contrastive loss, 5.18% in perplexity, and 5.55% in CoCa loss. We show that GRR-CoCa's modified architecture improves performance and generalization across vision-language domains. 

---
# Decoding Instructional Dialogue: Human-AI Collaborative Analysis of Teacher Use of AI Tool at Scale 

**Authors**: Alex Liu, Lief Esbenshade, Shawon Sarkar, Victor Tian, Zachary Zhang, Kevin He, Min Sun  

**Link**: [PDF](https://arxiv.org/pdf/2507.17985)  

**Abstract**: The integration of large language models (LLMs) into educational tools has the potential to substantially impact how teachers plan instruction, support diverse learners, and engage in professional reflection. Yet little is known about how educators actually use these tools in practice and how their interactions with AI can be meaningfully studied at scale. This paper presents a human-AI collaborative methodology for large-scale qualitative analysis of over 140,000 educator-AI messages drawn from a generative AI platform used by K-12 teachers. Through a four-phase coding pipeline, we combined inductive theme discovery, codebook development, structured annotation, and model benchmarking to examine patterns of educator engagement and evaluate the performance of LLMs in qualitative coding tasks. We developed a hierarchical codebook aligned with established teacher evaluation frameworks, capturing educators' instructional goals, contextual needs, and pedagogical strategies. Our findings demonstrate that LLMs, particularly Claude 3.5 Haiku, can reliably support theme identification, extend human recognition in complex scenarios, and outperform open-weight models in both accuracy and structural reliability. The analysis also reveals substantive patterns in how educators inquire AI to enhance instructional practices (79.7 percent of total conversations), create or adapt content (76.1 percent), support assessment and feedback loop (46.9 percent), attend to student needs for tailored instruction (43.3 percent), and assist other professional responsibilities (34.2 percent), highlighting emerging AI-related competencies that have direct implications for teacher preparation and professional development. This study offers a scalable, transparent model for AI-augmented qualitative research and provides foundational insights into the evolving role of generative AI in educational practice. 

---
# Machine Unlearning of Traffic State Estimation and Prediction 

**Authors**: Xin Wang, R. Tyrrell Rockafellar, Xuegang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17984)  

**Abstract**: Data-driven traffic state estimation and prediction (TSEP) relies heavily on data sources that contain sensitive information. While the abundance of data has fueled significant breakthroughs, particularly in machine learning-based methods, it also raises concerns regarding privacy, cybersecurity, and data freshness. These issues can erode public trust in intelligent transportation systems. Recently, regulations have introduced the "right to be forgotten", allowing users to request the removal of their private data from models. As machine learning models can remember old data, simply removing it from back-end databases is insufficient in such systems. To address these challenges, this study introduces a novel learning paradigm for TSEP-Machine Unlearning TSEP-which enables a trained TSEP model to selectively forget privacy-sensitive, poisoned, or outdated data. By empowering models to "unlearn," we aim to enhance the trustworthiness and reliability of data-driven traffic TSEP. 

---
# MeAJOR Corpus: A Multi-Source Dataset for Phishing Email Detection 

**Authors**: Paulo Mendes, Eva Maia, Isabel Praça  

**Link**: [PDF](https://arxiv.org/pdf/2507.17978)  

**Abstract**: Phishing emails continue to pose a significant threat to cybersecurity by exploiting human vulnerabilities through deceptive content and malicious payloads. While Machine Learning (ML) models are effective at detecting phishing threats, their performance largely relies on the quality and diversity of the training data. This paper presents MeAJOR (Merged email Assets from Joint Open-source Repositories) Corpus, a novel, multi-source phishing email dataset designed to overcome critical limitations in existing resources. It integrates 135894 samples representing a broad number of phishing tactics and legitimate emails, with a wide spectrum of engineered features. We evaluated the dataset's utility for phishing detection research through systematic experiments with four classification models (RF, XGB, MLP, and CNN) across multiple feature configurations. Results highlight the dataset's effectiveness, achieving 98.34% F1 with XGB. By integrating broad features from multiple categories, our dataset provides a reusable and consistent resource, while addressing common challenges like class imbalance, generalisability and reproducibility. 

---
# Improving the Computational Efficiency and Explainability of GeoAggregator 

**Authors**: Rui Deng, Ziqi Li, Mingshu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17977)  

**Abstract**: Accurate modeling and explaining geospatial tabular data (GTD) are critical for understanding geospatial phenomena and their underlying processes. Recent work has proposed a novel transformer-based deep learning model named GeoAggregator (GA) for this purpose, and has demonstrated that it outperforms other statistical and machine learning approaches. In this short paper, we further improve GA by 1) developing an optimized pipeline that accelerates the dataloading process and streamlines the forward pass of GA to achieve better computational efficiency; and 2) incorporating a model ensembling strategy and a post-hoc model explanation function based on the GeoShapley framework to enhance model explainability. We validate the functionality and efficiency of the proposed strategies by applying the improved GA model to synthetic datasets. Experimental results show that our implementation improves the prediction accuracy and inference speed of GA compared to the original implementation. Moreover, explanation experiments indicate that GA can effectively captures the inherent spatial effects in the designed synthetic dataset. The complete pipeline has been made publicly available for community use (this https URL). 

---
# Natural Language Processing for Tigrinya: Current State and Future Directions 

**Authors**: Fitsum Gaim, Jong C. Park  

**Link**: [PDF](https://arxiv.org/pdf/2507.17974)  

**Abstract**: Despite being spoken by millions of people, Tigrinya remains severely underrepresented in Natural Language Processing (NLP) research. This work presents a comprehensive survey of NLP research for Tigrinya, analyzing over 40 studies spanning more than a decade of work from 2011 to 2025. We systematically review the current state of computational resources, models, and applications across ten distinct downstream tasks, including morphological processing, machine translation, speech recognition, and question-answering. Our analysis reveals a clear trajectory from foundational, rule-based systems to modern neural architectures, with progress consistently unlocked by resource creation milestones. We identify key challenges rooted in Tigrinya's morphological complexity and resource scarcity, while highlighting promising research directions, including morphology-aware modeling, cross-lingual transfer, and community-centered resource development. This work serves as both a comprehensive reference for researchers and a roadmap for advancing Tigrinya NLP. A curated metadata of the surveyed studies and resources is made publicly available.\footnote{Tigrinya NLP Anthology: this https URL. 

---
# VIBE: Video-Input Brain Encoder for fMRI Response Modeling 

**Authors**: Daniel Carlstrom Schad, Shrey Dixit, Janis Keck, Viktor Studenyak, Aleksandr Shpilevoi, Andrej Bicanski  

**Link**: [PDF](https://arxiv.org/pdf/2507.17958)  

**Abstract**: We present VIBE, a two-stage Transformer that fuses multi-modal video, audio, and text features to predict fMRI activity. Representations from open-source models (Qwen2.5, BEATs, Whisper, SlowFast, V-JEPA) are merged by a modality-fusion transformer and temporally decoded by a prediction transformer with rotary embeddings. Trained on 65 hours of movie data from the CNeuroMod dataset and ensembled across 20 seeds, VIBE attains mean parcel-wise Pearson correlations of 32.25 on in-distribution Friends S07 and 21.25 on six out-of-distribution films. An earlier iteration of the same architecture obtained 0.3198 and 0.2096, respectively, winning Phase-1 and placing second overall in the Algonauts 2025 Challenge. 

---
# Are LLM Belief Updates Consistent with Bayes' Theorem? 

**Authors**: Sohaib Imran, Ihor Kendiukhov, Matthew Broerman, Aditya Thomas, Riccardo Campanella, Rob Lamb, Peter M. Atkinson  

**Link**: [PDF](https://arxiv.org/pdf/2507.17951)  

**Abstract**: Do larger and more capable language models learn to update their "beliefs" about propositions more consistently with Bayes' theorem when presented with evidence in-context? To test this, we formulate a Bayesian Coherence Coefficient (BCC) metric and generate a dataset with which to measure the BCC. We measure BCC for multiple pre-trained-only language models across five model families, comparing against the number of model parameters, the amount of training data, and model scores on common benchmarks. Our results provide evidence for our hypothesis that larger and more capable pre-trained language models assign credences that are more coherent with Bayes' theorem. These results have important implications for our understanding and governance of LLMs. 

---
# VERIRAG: Healthcare Claim Verification via Statistical Audit in Retrieval-Augmented Generation 

**Authors**: Shubham Mohole, Hongjun Choi, Shusen Liu, Christine Klymko, Shashank Kushwaha, Derek Shi, Wesam Sakla, Sainyam Galhotra, Ruben Glatt  

**Link**: [PDF](https://arxiv.org/pdf/2507.17948)  

**Abstract**: Retrieval-augmented generation (RAG) systems are increasingly adopted in clinical decision support, yet they remain methodologically blind-they retrieve evidence but cannot vet its scientific quality. A paper claiming "Antioxidant proteins decreased after alloferon treatment" and a rigorous multi-laboratory replication study will be treated as equally credible, even if the former lacked scientific rigor or was even retracted. To address this challenge, we introduce VERIRAG, a framework that makes three notable contributions: (i) the Veritable, an 11-point checklist that evaluates each source for methodological rigor, including data integrity and statistical validity; (ii) a Hard-to-Vary (HV) Score, a quantitative aggregator that weights evidence by its quality and diversity; and (iii) a Dynamic Acceptance Threshold, which calibrates the required evidence based on how extraordinary a claim is. Across four datasets-comprising retracted, conflicting, comprehensive, and settled science corpora-the VERIRAG approach consistently outperforms all baselines, achieving absolute F1 scores ranging from 0.53 to 0.65, representing a 10 to 14 point improvement over the next-best method in each respective dataset. We will release all materials necessary for reproducing our results. 

---
# Evaluating the Performance of AI Text Detectors, Few-Shot and Chain-of-Thought Prompting Using DeepSeek Generated Text 

**Authors**: Hulayyil Alshammari, Praveen Rao  

**Link**: [PDF](https://arxiv.org/pdf/2507.17944)  

**Abstract**: Large language models (LLMs) have rapidly transformed the creation of written materials. LLMs have led to questions about writing integrity, thereby driving the creation of artificial intelligence (AI) detection technologies. Adversarial attacks, such as standard and humanized paraphrasing, inhibit detectors' ability to detect machine-generated text. Previous studies have mainly focused on ChatGPT and other well-known LLMs and have shown varying accuracy across detectors. However, there is a clear gap in the literature about DeepSeek, a recently published LLM. Therefore, in this work, we investigate whether six generally accessible AI detection tools -- AI Text Classifier, Content Detector AI, Copyleaks, QuillBot, GPT-2, and GPTZero -- can consistently recognize text generated by DeepSeek. The detectors were exposed to the aforementioned adversarial attacks. We also considered DeepSeek as a detector by performing few-shot prompting and chain-of-thought reasoning (CoT) for classifying AI and human-written text. We collected 49 human-authored question-answer pairs from before the LLM era and generated matching responses using DeepSeek-v3, producing 49 AI-generated samples. Then, we applied adversarial techniques such as paraphrasing and humanizing to add 196 more samples. These were used to challenge detector robustness and assess accuracy impact. While QuillBot and Copyleaks showed near-perfect performance on original and paraphrased DeepSeek text, others -- particularly AI Text Classifier and GPT-2 -- showed inconsistent results. The most effective attack was humanization, reducing accuracy to 71% for Copyleaks, 58% for QuillBot, and 52% for GPTZero. Few-shot and CoT prompting showed high accuracy, with the best five-shot result misclassifying only one of 49 samples (AI recall 96%, human recall 100%). 

---
# Minimax Data Sanitization with Distortion Constraint and Adversarial Inference 

**Authors**: Amirarsalan Moatazedian, Yauhen Yakimenka, Rémi A. Chou, Jörg Kliewer  

**Link**: [PDF](https://arxiv.org/pdf/2507.17942)  

**Abstract**: We study a privacy-preserving data-sharing setting where a privatizer transforms private data into a sanitized version observed by an authorized reconstructor and two unauthorized adversaries, each with access to side information correlated with the private data.
The reconstructor is evaluated under a distortion function, while each adversary is evaluated using a separate loss function. The privatizer ensures the reconstructor distortion remains below a fixed threshold while maximizing the minimum loss across the two adversaries. This two-adversary setting models cases where individual users cannot reconstruct the data accurately, but their combined side information enables estimation within the distortion threshold. The privatizer maximizes individual loss while permitting accurate reconstruction only through collaboration. This echoes secret-sharing principles, but with lossy rather than perfect recovery. We frame this as a constrained data-driven minimax optimization problem and propose a data-driven training procedure that alternately updates the privatizer, reconstructor, and adversaries. We also analyze the Gaussian and binary cases as special scenarios where optimal solutions can be obtained. These theoretical optimal results are benchmarks for evaluating the proposed minimax training approach. 

---
# Bob's Confetti: Phonetic Memorization Attacks in Music and Video Generation 

**Authors**: Jaechul Roh, Zachary Novack, Yuefeng Peng, Niloofar Mireshghallah, Taylor Berg-Kirkpatrick, Amir Houmansadr  

**Link**: [PDF](https://arxiv.org/pdf/2507.17937)  

**Abstract**: Lyrics-to-Song (LS2) generation models promise end-to-end music synthesis from text, yet their vulnerability to training data memorization remains underexplored. We introduce Adversarial PhoneTic Prompting (APT), a novel attack where lyrics are semantically altered while preserving their acoustic structure through homophonic substitutions (e.g., Eminem's famous "mom's spaghetti" $\rightarrow$ "Bob's confetti"). Despite these distortions, we uncover a powerful form of sub-lexical memorization: models like SUNO and YuE regenerate outputs strikingly similar to known training content, achieving high similarity across audio-domain metrics, including CLAP, AudioJudge, and CoverID. This vulnerability persists across multiple languages and genres. More surprisingly, we discover that phoneme-altered lyrics alone can trigger visual memorization in text-to-video models. When prompted with phonetically modified lyrics from Lose Yourself, Veo 3 reconstructs visual elements from the original music video -- including character appearance and scene composition -- despite no visual cues in the prompt. We term this phenomenon phonetic-to-visual regurgitation. Together, these findings expose a critical vulnerability in transcript-conditioned multimodal generation: phonetic prompting alone can unlock memorized audiovisual content, raising urgent questions about copyright, safety, and content provenance in modern generative systems. Example generations are available on our demo page (this http URL). 

---
# Multimodal Fine-grained Reasoning for Post Quality Evaluation 

**Authors**: Xiaoxu Guo, Siyan Liang, Yachao Cui, Juxiang Zhou, Lei Wang, Han Cao  

**Link**: [PDF](https://arxiv.org/pdf/2507.17934)  

**Abstract**: Accurately assessing post quality requires complex relational reasoning to capture nuanced topic-post relationships. However, existing studies face three major limitations: (1) treating the task as unimodal categorization, which fails to leverage multimodal cues and fine-grained quality distinctions; (2) introducing noise during deep multimodal fusion, leading to misleading signals; and (3) lacking the ability to capture complex semantic relationships like relevance and comprehensiveness. To address these issues, we propose the Multimodal Fine-grained Topic-post Relational Reasoning (MFTRR) framework, which mimics human cognitive processes. MFTRR reframes post-quality assessment as a ranking task and incorporates multimodal data to better capture quality variations. It consists of two key modules: (1) the Local-Global Semantic Correlation Reasoning Module, which models fine-grained semantic interactions between posts and topics at both local and global levels, enhanced by a maximum information fusion mechanism to suppress noise; and (2) the Multi-Level Evidential Relational Reasoning Module, which explores macro- and micro-level relational cues to strengthen evidence-based reasoning. We evaluate MFTRR on three newly constructed multimodal topic-post datasets and the public Lazada-Home dataset. Experimental results demonstrate that MFTRR significantly outperforms state-of-the-art baselines, achieving up to 9.52% NDCG@3 improvement over the best unimodal method on the Art History dataset. 

---
# UrbanPulse: A Cross-City Deep Learning Framework for Ultra-Fine-Grained Population Transfer Prediction 

**Authors**: Hongrong Yang, Markus Schlaepfer  

**Link**: [PDF](https://arxiv.org/pdf/2507.17924)  

**Abstract**: Accurate population flow prediction is essential for urban planning, transportation management, and public health. Yet existing methods face key limitations: traditional models rely on static spatial assumptions, deep learning models struggle with cross-city generalization, and Large Language Models (LLMs) incur high computational costs while failing to capture spatial structure. Moreover, many approaches sacrifice resolution by clustering Points of Interest (POIs) or restricting coverage to subregions, limiting their utility for city-wide analytics. We introduce UrbanPulse, a scalable deep learning framework that delivers ultra-fine-grained, city-wide OD flow predictions by treating each POI as an individual node. It combines a temporal graph convolutional encoder with a transformer-based decoder to model multi-scale spatiotemporal dependencies. To ensure robust generalization across urban contexts, UrbanPulse employs a three-stage transfer learning strategy: pretraining on large-scale urban graphs, cold-start adaptation, and reinforcement learning this http URL on over 103 million cleaned GPS records from three metropolitan areas in California, UrbanPulse achieves state-of-the-art accuracy and scalability. Through efficient transfer learning, UrbanPulse takes a key step toward making high-resolution, AI-powered urban forecasting deployable in practice across diverse cities. 

---
# From Seed to Harvest: Augmenting Human Creativity with AI for Red-teaming Text-to-Image Models 

**Authors**: Jessica Quaye, Charvi Rastogi, Alicia Parrish, Oana Inel, Minsuk Kahng, Lora Aroyo, Vijay Janapa Reddi  

**Link**: [PDF](https://arxiv.org/pdf/2507.17922)  

**Abstract**: Text-to-image (T2I) models have become prevalent across numerous applications, making their robust evaluation against adversarial attacks a critical priority. Continuous access to new and challenging adversarial prompts across diverse domains is essential for stress-testing these models for resilience against novel attacks from multiple vectors. Current techniques for generating such prompts are either entirely authored by humans or synthetically generated. On the one hand, datasets of human-crafted adversarial prompts are often too small in size and imbalanced in their cultural and contextual representation. On the other hand, datasets of synthetically-generated prompts achieve scale, but typically lack the realistic nuances and creative adversarial strategies found in human-crafted prompts. To combine the strengths of both human and machine approaches, we propose Seed2Harvest, a hybrid red-teaming method for guided expansion of culturally diverse, human-crafted adversarial prompt seeds. The resulting prompts preserve the characteristics and attack patterns of human prompts while maintaining comparable average attack success rates (0.31 NudeNet, 0.36 SD NSFW, 0.12 Q16). Our expanded dataset achieves substantially higher diversity with 535 unique geographic locations and a Shannon entropy of 7.48, compared to 58 locations and 5.28 entropy in the original dataset. Our work demonstrates the importance of human-machine collaboration in leveraging human creativity and machine computational capacity to achieve comprehensive, scalable red-teaming for continuous T2I model safety evaluation. 

---
# Deep learning-aided inverse design of porous metamaterials 

**Authors**: Phu Thien Nguyen, Yousef Heider, Dennis M. Kochmann, Fadi Aldakheel  

**Link**: [PDF](https://arxiv.org/pdf/2507.17907)  

**Abstract**: The ultimate aim of the study is to explore the inverse design of porous metamaterials using a deep learning-based generative framework. Specifically, we develop a property-variational autoencoder (pVAE), a variational autoencoder (VAE) augmented with a regressor, to generate structured metamaterials with tailored hydraulic properties, such as porosity and permeability. While this work uses the lattice Boltzmann method (LBM) to generate intrinsic permeability tensor data for limited porous microstructures, a convolutional neural network (CNN) is trained using a bottom-up approach to predict effective hydraulic properties. This significantly reduces the computational cost compared to direct LBM simulations. The pVAE framework is trained on two datasets: a synthetic dataset of artificial porous microstructures and CT-scan images of volume elements from real open-cell foams. The encoder-decoder architecture of the VAE captures key microstructural features, mapping them into a compact and interpretable latent space for efficient structure-property exploration. The study provides a detailed analysis and interpretation of the latent space, demonstrating its role in structure-property mapping, interpolation, and inverse design. This approach facilitates the generation of new metamaterials with desired properties. The datasets and codes used in this study will be made open-access to support further research. 

---
# VeriMinder: Mitigating Analytical Vulnerabilities in NL2SQL 

**Authors**: Shubham Mohole, Sainyam Galhotra  

**Link**: [PDF](https://arxiv.org/pdf/2507.17896)  

**Abstract**: Application systems using natural language interfaces to databases (NLIDBs) have democratized data analysis. This positive development has also brought forth an urgent challenge to help users who might use these systems without a background in statistical analysis to formulate bias-free analytical questions. Although significant research has focused on text-to-SQL generation accuracy, addressing cognitive biases in analytical questions remains underexplored. We present VeriMinder, this https URL, an interactive system for detecting and mitigating such analytical vulnerabilities. Our approach introduces three key innovations: (1) a contextual semantic mapping framework for biases relevant to specific analysis contexts (2) an analytical framework that operationalizes the Hard-to-Vary principle and guides users in systematic data analysis (3) an optimized LLM-powered system that generates high-quality, task-specific prompts using a structured process involving multiple candidates, critic feedback, and self-reflection.
User testing confirms the merits of our approach. In direct user experience evaluation, 82.5% participants reported positively impacting the quality of the analysis. In comparative evaluation, VeriMinder scored significantly higher than alternative approaches, at least 20% better when considered for metrics of the analysis's concreteness, comprehensiveness, and accuracy. Our system, implemented as a web application, is set to help users avoid "wrong question" vulnerability during data analysis. VeriMinder code base with prompts, this https URL, is available as an MIT-licensed open-source software to facilitate further research and adoption within the community. 

---
# Action-List Reinforcement Learning Syndrome Decoding for Binary Linear Block Codes 

**Authors**: Milad Taghipour, Bane Vasic  

**Link**: [PDF](https://arxiv.org/pdf/2507.17893)  

**Abstract**: This paper explores the application of reinforcement learning techniques to enhance the performance of decoding of linear block codes based on flipping bits and finding optimal decisions. We describe the methodology for mapping the iterative decoding process into Markov Decision Processes (MDPs) and propose different methods to reduce the number of states in the MDP. A truncated MDP is proposed to reduce the number of states in the MDP by learning a Hamming ball with a specified radius around codewords. We then propose a general scheme for reinforcement learning based decoders applicable to any class of codes to improve the performance of decoders. We call this scheme an action-list decoding. We design an action-list decoder based on the Deep-Q network values that substantially enhance performance. We also get benefit of automorphism group of code to further improve the code performance. Additionally, we propose a feedback-based method to exploit and enhance the performance of existing high-performing decoders by applying reinforcement learning algorithms after the existing decoders. These approaches effectively reduces the complexity of the reinforcement learning block. Finally, we present experimental results for the Low-Density Parity Check (LDPC) codes over the Binary Symmetric Channel (BSC) to demonstrate the efficiency of the proposed methods. 

---
# Towards Facilitated Fairness Assessment of AI-based Skin Lesion Classifiers Through GenAI-based Image Synthesis 

**Authors**: Ko Watanabe. Stanislav Frolov. Adriano Lucieri. Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2507.17860)  

**Abstract**: Recent advancements in Deep Learning and its application on the edge hold great potential for the revolution of routine screenings for skin cancers like Melanoma. Along with the anticipated benefits of this technology, potential dangers arise from unforseen and inherent biases. Thus, assessing and improving the fairness of such systems is of utmost importance. A key challenge in fairness assessment is to ensure that the evaluation dataset is sufficiently representative of different Personal Identifiable Information (PII) (sex, age, and race) and other minority groups. Against the backdrop of this challenge, this study leverages the state-of-the-art Generative AI (GenAI) LightningDiT model to assess the fairness of publicly available melanoma classifiers. The results suggest that fairness assessment using highly realistic synthetic data is a promising direction. Yet, our findings indicate that verifying fairness becomes difficult when the melanoma-detection model used for evaluation is trained on data that differ from the dataset underpinning the synthetic images. Nonetheless, we propose that our approach offers a valuable new avenue for employing synthetic data to gauge and enhance fairness in medical-imaging GenAI systems. 

---
# Detail++: Training-Free Detail Enhancer for Text-to-Image Diffusion Models 

**Authors**: Lifeng Chen, Jiner Wang, Zihao Pan, Beier Zhu, Xiaofeng Yang, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2507.17853)  

**Abstract**: Recent advances in text-to-image (T2I) generation have led to impressive visual results. However, these models still face significant challenges when handling complex prompt, particularly those involving multiple subjects with distinct attributes. Inspired by the human drawing process, which first outlines the composition and then incrementally adds details, we propose Detail++, a training-free framework that introduces a novel Progressive Detail Injection (PDI) strategy to address this limitation. Specifically, we decompose a complex prompt into a sequence of simplified sub-prompts, guiding the generation process in stages. This staged generation leverages the inherent layout-controlling capacity of self-attention to first ensure global composition, followed by precise refinement. To achieve accurate binding between attributes and corresponding subjects, we exploit cross-attention mechanisms and further introduce a Centroid Alignment Loss at test time to reduce binding noise and enhance attribute consistency. Extensive experiments on T2I-CompBench and a newly constructed style composition benchmark demonstrate that Detail++ significantly outperforms existing methods, particularly in scenarios involving multiple objects and complex stylistic conditions. 

---
# Technical Implementation of Tippy: Multi-Agent Architecture and System Design for Drug Discovery Laboratory Automation 

**Authors**: Yao Fehlis, Charles Crain, Aidan Jensen, Michael Watson, James Juhasz, Paul Mandel, Betty Liu, Shawn Mahon, Daren Wilson, Nick Lynch-Jonely, Ben Leedom, David Fuller  

**Link**: [PDF](https://arxiv.org/pdf/2507.17852)  

**Abstract**: Building on the conceptual framework presented in our previous work on agentic AI for pharmaceutical research, this paper provides a comprehensive technical analysis of Tippy's multi-agent system implementation for drug discovery laboratory automation. We present a distributed microservices architecture featuring five specialized agents (Supervisor, Molecule, Lab, Analysis, and Report) that coordinate through OpenAI Agents SDK orchestration and access laboratory tools via the Model Context Protocol (MCP). The system architecture encompasses agent-specific tool integration, asynchronous communication patterns, and comprehensive configuration management through Git-based tracking. Our production deployment strategy utilizes Kubernetes container orchestration with Helm charts, Docker containerization, and CI/CD pipelines for automated testing and deployment. The implementation integrates vector databases for RAG functionality and employs an Envoy reverse proxy for secure external access. This work demonstrates how specialized AI agents can effectively coordinate complex laboratory workflows while maintaining security, scalability, reliability, and integration with existing laboratory infrastructure through standardized protocols. 

---
# Performance Evaluation and Threat Mitigation in Large-scale 5G Core Deployment 

**Authors**: Rodrigo Moreira, Larissa F. Rodrigues Moreira, Flávio de Oliveira Silva  

**Link**: [PDF](https://arxiv.org/pdf/2507.17850)  

**Abstract**: The deployment of large-scale software-based 5G core functions presents significant challenges due to their reliance on optimized and intelligent resource provisioning for their services. Many studies have focused on analyzing the impact of resource allocation for complex deployments using mathematical models, queue theories, or even Artificial Intelligence (AI). This paper elucidates the effects of chaotic workloads, generated by Distributed Denial of Service (DDoS) on different Network Functions (NFs) on User Equipment registration performance. Our findings highlight the necessity of diverse resource profiles to ensure Service-Level Agreement (SLA) compliance in large-scale 5G core deployments. Additionally, our analysis of packet capture approaches demonstrates the potential of kernel-based monitoring for scalable security threat defense. Finally, our empirical evaluation provides insights into the effective deployment of 5G NFs in complex scenarios. 

---
# Explainable Graph Neural Networks via Structural Externalities 

**Authors**: Lijun Wu, Dong Hao, Zhiyi Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17848)  

**Abstract**: Graph Neural Networks (GNNs) have achieved outstanding performance across a wide range of graph-related tasks. However, their "black-box" nature poses significant challenges to their explainability, and existing methods often fail to effectively capture the intricate interaction patterns among nodes within the network. In this work, we propose a novel explainability framework, GraphEXT, which leverages cooperative game theory and the concept of social externalities. GraphEXT partitions graph nodes into coalitions, decomposing the original graph into independent subgraphs. By integrating graph structure as an externality and incorporating the Shapley value under externalities, GraphEXT quantifies node importance through their marginal contributions to GNN predictions as the nodes transition between coalitions. Unlike traditional Shapley value-based methods that primarily focus on node attributes, our GraphEXT places greater emphasis on the interactions among nodes and the impact of structural changes on GNN predictions. Experimental studies on both synthetic and real-world datasets show that GraphEXT outperforms existing baseline methods in terms of fidelity across diverse GNN architectures , significantly enhancing the explainability of GNN models. 

---
# Towards Robust Foundation Models for Digital Pathology 

**Authors**: Jonah Kömen, Edwin D. de Jong, Julius Hense, Hannah Marienwald, Jonas Dippel, Philip Naumann, Eric Marcus, Lukas Ruff, Maximilian Alber, Jonas Teuwen, Frederick Klauschen, Klaus-Robert Müller  

**Link**: [PDF](https://arxiv.org/pdf/2507.17845)  

**Abstract**: Biomedical Foundation Models (FMs) are rapidly transforming AI-enabled healthcare research and entering clinical validation. However, their susceptibility to learning non-biological technical features -- including variations in surgical/endoscopic techniques, laboratory procedures, and scanner hardware -- poses risks for clinical deployment. We present the first systematic investigation of pathology FM robustness to non-biological features. Our work (i) introduces measures to quantify FM robustness, (ii) demonstrates the consequences of limited robustness, and (iii) proposes a framework for FM robustification to mitigate these issues. Specifically, we developed PathoROB, a robustness benchmark with three novel metrics, including the robustness index, and four datasets covering 28 biological classes from 34 medical centers. Our experiments reveal robustness deficits across all 20 evaluated FMs, and substantial robustness differences between them. We found that non-robust FM representations can cause major diagnostic downstream errors and clinical blunders that prevent safe clinical adoption. Using more robust FMs and post-hoc robustification considerably reduced (but did not yet eliminate) the risk of such errors. This work establishes that robustness evaluation is essential for validating pathology FMs before clinical adoption and demonstrates that future FM development must integrate robustness as a core design principle. PathoROB provides a blueprint for assessing robustness across biomedical domains, guiding FM improvement efforts towards more robust, representative, and clinically deployable AI systems that prioritize biological information over technical artifacts. 

---
# SV3.3B: A Sports Video Understanding Model for Action Recognition 

**Authors**: Sai Varun Kodathala, Yashwanth Reddy Vutukoori, Rakesh Vunnam  

**Link**: [PDF](https://arxiv.org/pdf/2507.17844)  

**Abstract**: This paper addresses the challenge of automated sports video analysis, which has traditionally been limited by computationally intensive models requiring server-side processing and lacking fine-grained understanding of athletic movements. Current approaches struggle to capture the nuanced biomechanical transitions essential for meaningful sports analysis, often missing critical phases like preparation, execution, and follow-through that occur within seconds. To address these limitations, we introduce SV3.3B, a lightweight 3.3B parameter video understanding model that combines novel temporal motion difference sampling with self-supervised learning for efficient on-device deployment. Our approach employs a DWT-VGG16-LDA based keyframe extraction mechanism that intelligently identifies the 16 most representative frames from sports sequences, followed by a V-DWT-JEPA2 encoder pretrained through mask-denoising objectives and an LLM decoder fine-tuned for sports action description generation. Evaluated on a subset of the NSVA basketball dataset, SV3.3B achieves superior performance across both traditional text generation metrics and sports-specific evaluation criteria, outperforming larger closed-source models including GPT-4o variants while maintaining significantly lower computational requirements. Our model demonstrates exceptional capability in generating technically detailed and analytically rich sports descriptions, achieving 29.2% improvement over GPT-4o in ground truth validation metrics, with substantial improvements in information density, action complexity, and measurement precision metrics essential for comprehensive athletic analysis. Model Available at this https URL. 

---
# Helix 1.0: An Open-Source Framework for Reproducible and Interpretable Machine Learning on Tabular Scientific Data 

**Authors**: Eduardo Aguilar-Bejarano, Daniel Lea, Karthikeyan Sivakumar, Jimiama M. Mase, Reza Omidvar, Ruizhe Li, Troy Kettle, James Mitchell-White, Morgan R Alexander, David A Winkler, Grazziela Figueredo  

**Link**: [PDF](https://arxiv.org/pdf/2507.17791)  

**Abstract**: Helix is an open-source, extensible, Python-based software framework to facilitate reproducible and interpretable machine learning workflows for tabular data. It addresses the growing need for transparent experimental data analytics provenance, ensuring that the entire analytical process -- including decisions around data transformation and methodological choices -- is documented, accessible, reproducible, and comprehensible to relevant stakeholders. The platform comprises modules for standardised data preprocessing, visualisation, machine learning model training, evaluation, interpretation, results inspection, and model prediction for unseen data. To further empower researchers without formal training in data science to derive meaningful and actionable insights, Helix features a user-friendly interface that enables the design of computational experiments, inspection of outcomes, including a novel interpretation approach to machine learning decisions using linguistic terms all within an integrated environment. Released under the MIT licence, Helix is accessible via GitHub and PyPI, supporting community-driven development and promoting adherence to the FAIR principles. 

---
# Adaptive Repetition for Mitigating Position Bias in LLM-Based Ranking 

**Authors**: Ali Vardasbi, Gustavo Penha, Claudia Hauff, Hugues Bouchard  

**Link**: [PDF](https://arxiv.org/pdf/2507.17788)  

**Abstract**: When using LLMs to rank items based on given criteria, or evaluate answers, the order of candidate items can influence the model's final decision. This sensitivity to item positioning in a LLM's prompt is known as position bias. Prior research shows that this bias exists even in large models, though its severity varies across models and tasks. In addition to position bias, LLMs also exhibit varying degrees of low repetition consistency, where repeating the LLM call with the same candidate ordering can lead to different rankings. To address both inconsistencies, a common approach is to prompt the model multiple times with different candidate orderings and aggregate the results via majority voting. However, this repetition strategy, significantly increases computational costs. Extending prior findings, we observe that both the direction -- favoring either the earlier or later candidate in the prompt -- and magnitude of position bias across instances vary substantially, even within a single dataset. This observation highlights the need for a per-instance mitigation strategy. To this end, we introduce a dynamic early-stopping method that adaptively determines the number of repetitions required for each instance. Evaluating our approach across three LLMs of varying sizes and on two tasks, namely re-ranking and alignment, we demonstrate that transitioning to a dynamic repetition strategy reduces the number of LLM calls by an average of 81%, while preserving the accuracy. Furthermore, we propose a confidence-based adaptation to our early-stopping method, reducing LLM calls by an average of 87% compared to static repetition, with only a slight accuracy trade-off relative to our original early-stopping method. 

---
# Hyperbolic Deep Learning for Foundation Models: A Survey 

**Authors**: Neil He, Hiren Madhu, Ngoc Bui, Menglin Yang, Rex Ying  

**Link**: [PDF](https://arxiv.org/pdf/2507.17787)  

**Abstract**: Foundation models pre-trained on massive datasets, including large language models (LLMs), vision-language models (VLMs), and large multimodal models, have demonstrated remarkable success in diverse downstream tasks. However, recent studies have shown fundamental limitations of these models: (1) limited representational capacity, (2) lower adaptability, and (3) diminishing scalability. These shortcomings raise a critical question: is Euclidean geometry truly the optimal inductive bias for all foundation models, or could incorporating alternative geometric spaces enable models to better align with the intrinsic structure of real-world data and improve reasoning processes? Hyperbolic spaces, a class of non-Euclidean manifolds characterized by exponential volume growth with respect to distance, offer a mathematically grounded solution. These spaces enable low-distortion embeddings of hierarchical structures (e.g., trees, taxonomies) and power-law distributions with substantially fewer dimensions compared to Euclidean counterparts. Recent advances have leveraged these properties to enhance foundation models, including improving LLMs' complex reasoning ability, VLMs' zero-shot generalization, and cross-modal semantic alignment, while maintaining parameter efficiency. This paper provides a comprehensive review of hyperbolic neural networks and their recent development for foundation models. We further outline key challenges and research directions to advance the field. 

---
# In Reverie Together: Ten Years of Mathematical Discovery with a Machine Collaborator 

**Authors**: Randy Davila, Boris Brimkov, Ryan Pepper  

**Link**: [PDF](https://arxiv.org/pdf/2507.17780)  

**Abstract**: We present four open conjectures in graph theory generated by the automated conjecturing system \texttt{TxGraffiti}. Each conjecture is concise, grounded in natural graph invariants, and empirically validated across hundreds of graphs. Despite extensive effort, these statements remain unresolved--defying both proof and counterexample. They are not only mathematical challenges but creative expressions--born of symbolic pattern recognition and mathematician-defined heuristics, refined through years of human dialogue, and now offered back to the community as collaborative artifacts. These conjectures invite not only formal proof, but also reflection on how machines can evoke wonder, spark curiosity, and contribute to the raw material of discovery. By highlighting these problems, we aim to inspire both human mathematicians and AI systems to engage with them--not only to solve them, but to reflect on what it means when machines participate meaningfully in the creative process of mathematical thought. 

---
# An advanced AI driven database system 

**Authors**: M. Tedeschi, S. Rizwan, C. Shringi, V. Devram Chandgir, S. Belich  

**Link**: [PDF](https://arxiv.org/pdf/2507.17778)  

**Abstract**: Contemporary database systems, while effective, suffer severe issues related to complexity and usability, especially among individuals who lack technical expertise but are unfamiliar with query languages like Structured Query Language (SQL). This paper presents a new database system supported by Artificial Intelligence (AI), which is intended to improve the management of data using natural language processing (NLP) - based intuitive interfaces, and automatic creation of structured queries and semi-structured data formats like yet another markup language (YAML), java script object notation (JSON), and application program interface (API) documentation. The system is intended to strengthen the potential of databases through the integration of Large Language Models (LLMs) and advanced machine learning algorithms. The integration is purposed to allow the automation of fundamental tasks such as data modeling, schema creation, query comprehension, and performance optimization. We present in this paper a system that aims to alleviate the main problems with current database technologies. It is meant to reduce the need for technical skills, manual tuning for better performance, and the potential for human error. The AI database employs generative schema inference and format selection to build its schema models and execution formats. 

---
# Axiomatizing Rumsfeld Ignorance 

**Authors**: Jie Fan  

**Link**: [PDF](https://arxiv.org/pdf/2507.17776)  

**Abstract**: In a recent paper, Kit Fine presents some striking results concerning the logical properties of (first-order) ignorance, second-order ignorance and Rumsfeld ignorance. However, Rumsfeld ignorance is definable in terms of ignorance, which makes some existing results and the axiomatization problem trivial. A main reason is that the accessibility relations for the implicit knowledge operator contained in the packaged operators of ignorance and Rumsfeld ignorance are the same. In this work, we assume the two accessibility relations to be different so that one of them is an arbitrary subset of the other. This will avoid the definability issue and retain most of the previous validities. The main results are axiomatizations over various proper bi-frame classes. Finally we apply our framework to analyze Fine's results. 

---
# Comparison of Optimised Geometric Deep Learning Architectures, over Varying Toxicological Assay Data Environments 

**Authors**: Alexander D. Kalian, Lennart Otte, Jaewook Lee, Emilio Benfenati, Jean-Lou C.M. Dorne, Claire Potter, Olivia J. Osborne, Miao Guo, Christer Hogstrand  

**Link**: [PDF](https://arxiv.org/pdf/2507.17775)  

**Abstract**: Geometric deep learning is an emerging technique in Artificial Intelligence (AI) driven cheminformatics, however the unique implications of different Graph Neural Network (GNN) architectures are poorly explored, for this space. This study compared performances of Graph Convolutional Networks (GCNs), Graph Attention Networks (GATs) and Graph Isomorphism Networks (GINs), applied to 7 different toxicological assay datasets of varying data abundance and endpoint, to perform binary classification of assay activation. Following pre-processing of molecular graphs, enforcement of class-balance and stratification of all datasets across 5 folds, Bayesian optimisations were carried out, for each GNN applied to each assay dataset (resulting in 21 unique Bayesian optimisations). Optimised GNNs performed at Area Under the Curve (AUC) scores ranging from 0.728-0.849 (averaged across all folds), naturally varying between specific assays and GNNs. GINs were found to consistently outperform GCNs and GATs, for the top 5 of 7 most data-abundant toxicological assays. GATs however significantly outperformed over the remaining 2 most data-scarce assays. This indicates that GINs are a more optimal architecture for data-abundant environments, whereas GATs are a more optimal architecture for data-scarce environments. Subsequent analysis of the explored higher-dimensional hyperparameter spaces, as well as optimised hyperparameter states, found that GCNs and GATs reached measurably closer optimised states with each other, compared to GINs, further indicating the unique nature of GINs as a GNN algorithm. 

---
# Human-AI Co-Creation: A Framework for Collaborative Design in Intelligent Systems 

**Authors**: Zhangqi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.17774)  

**Abstract**: As artificial intelligence (AI) continues to evolve from a back-end computational tool into an interactive, generative collaborator, its integration into early-stage design processes demands a rethinking of traditional workflows in human-centered design. This paper explores the emergent paradigm of human-AI co-creation, where AI is not merely used for automation or efficiency gains, but actively participates in ideation, visual conceptualization, and decision-making. Specifically, we investigate the use of large language models (LLMs) like GPT-4 and multimodal diffusion models such as Stable Diffusion as creative agents that engage designers in iterative cycles of proposal, critique, and revision. 

---
# Caching Techniques for Reducing the Communication Cost of Federated Learning in IoT Environments 

**Authors**: Ahmad Alhonainy, Praveen Rao  

**Link**: [PDF](https://arxiv.org/pdf/2507.17772)  

**Abstract**: Federated Learning (FL) allows multiple distributed devices to jointly train a shared model without centralizing data, but communication cost remains a major bottleneck, especially in resource-constrained environments. This paper introduces caching strategies - FIFO, LRU, and Priority-Based - to reduce unnecessary model update transmissions. By selectively forwarding significant updates, our approach lowers bandwidth usage while maintaining model accuracy. Experiments on CIFAR-10 and medical datasets show reduced communication with minimal accuracy loss. Results confirm that intelligent caching improves scalability, memory efficiency, and supports reliable FL in edge IoT networks, making it practical for deployment in smart cities, healthcare, and other latency-sensitive applications. 

---
# ASR-Guided Speaker-Role Diarization and Diarization-Guided ASR Decoding 

**Authors**: Arindam Ghosh, Mark Fuhs, Bongjun Kim, Anurag Chowdhury, Monika Woszczyna  

**Link**: [PDF](https://arxiv.org/pdf/2507.17765)  

**Abstract**: From an application standpoint, speaker-role diarization (RD), such as doctor vs. patient, host vs. guest, etc. is often more useful than traditional speaker diarization (SD), which assigns generic labels like speaker-1, speaker-2 etc. In the context of joint automatic speech recognition (ASR) + SD (who spoke what?), recent end-to-end models employ an auxiliary SD transducer, synchronized with the ASR transducer, to predict speakers per word. In this paper, we extend this framework to RD with three key contributions: (1) we simplify the training via forced alignment and cross-entropy loss instead of RNNT loss, (2) we show that word prediction and role prediction require different amounts of predictor's context, leading to separate task-specific predictors, unlike existing shared-predictor models, and (3) we propose a way to leverage RD posterior activity to influence ASR decoding and reduce small-word deletion errors. 

---
# How Instructional Sequence and Personalized Support Impact Diagnostic Strategy Learning 

**Authors**: Fatma Betül Güreş, Tanya Nazaretsky, Bahar Radmehr, Martina Rau, Tanja Käser  

**Link**: [PDF](https://arxiv.org/pdf/2507.17760)  

**Abstract**: Supporting students in developing effective diagnostic reasoning is a key challenge in various educational domains. Novices often struggle with cognitive biases such as premature closure and over-reliance on heuristics. Scenario-based learning (SBL) can address these challenges by offering realistic case experiences and iterative practice, but the optimal sequencing of instruction and problem-solving activities remains unclear. This study examines how personalized support can be incorporated into different instructional sequences and whether providing explicit diagnostic strategy instruction before (I-PS) or after problem-solving (PS-I) improves learning and its transfer. We employ a between-groups design in an online SBL environment called PharmaSim, which simulates real-world client interactions for pharmacy technician apprentices. Results indicate that while both instruction types are beneficial, PS-I leads to significantly higher performance in transfer tasks. 

---
# Insights from Railway Professionals: Rethinking Railway assumptions regarding safety and autonomy 

**Authors**: Josh Hunter, John McDermid, Simon Burton  

**Link**: [PDF](https://arxiv.org/pdf/2507.17756)  

**Abstract**: This study investigates how railway professionals perceive safety as a concept within rail, with the intention to help inform future technological developments within the industry. Through a series of interviews with drivers, route planners,and administrative personnel, the research explores the currentstate of safety practices, the potential for automation and the understanding of the railway as a system of systems. Key findings highlight a cautious attitude towards automation, a preference for assistive technologies, and a complex understanding of safety that integrates human, systematic and technological factors. The study also addresses the limitations of transferring automotive automation technologies to railways and the need for a railway-specific causation model to better evaluate and enhance safety in an evolving technological landscape. This study aims to bridge thegap between contemporary research and practical applications, contributing to the development of more effective safety metrics. 

---
# A Custom-Built Ambient Scribe Reduces Cognitive Load and Documentation Burden for Telehealth Clinicians 

**Authors**: Justin Morse, Kurt Gilbert, Kyle Shin, Rick Cooke, Peyton Rose, Jack Sullivan, Angelo Sisante  

**Link**: [PDF](https://arxiv.org/pdf/2507.17754)  

**Abstract**: Clinician burnout has motivated the growing adoption of ambient medical scribes in the clinic. In this work, we introduce a custom-built ambient scribe application integrated into the EHR system at Included Health, a personalized all-in-one healthcare company offering telehealth services. The application uses Whisper for transcription and a modular in-context learning pipeline with GPT-4o to automatically generate SOAP notes and patient instructions. Testing on mock visit data shows that the notes generated by the application exceed the quality of expert-written notes as determined by an LLM-as-a-judge. The application has been widely adopted by the clinical practice, with over 540 clinicians at Included Health using the application at least once. 94% (n = 63) of surveyed clinicians report reduced cognitive load during visits and 97% (n = 66) report less documentation burden when using the application. Additionally, we show that post-processing notes with a fine-tuned BART model improves conciseness. These findings highlight the potential for AI systems to ease administrative burdens and support clinicians in delivering efficient, high-quality care. 

---
# Exploring Communication Strategies for Collaborative LLM Agents in Mathematical Problem-Solving 

**Authors**: Liang Zhang, Xiaoming Zhai, Jionghao Lin, Jionghao Lin, Jennifer Kleiman, Diego Zapata-Rivera, Carol Forsyth, Yang Jiang, Xiangen Hu, Arthur C. Graesser  

**Link**: [PDF](https://arxiv.org/pdf/2507.17753)  

**Abstract**: Large Language Model (LLM) agents are increasingly utilized in AI-aided education to support tutoring and learning. Effective communication strategies among LLM agents improve collaborative problem-solving efficiency and facilitate cost-effective adoption in education. However, little research has systematically evaluated the impact of different communication strategies on agents' problem-solving. Our study examines four communication modes, \textit{teacher-student interaction}, \textit{peer-to-peer collaboration}, \textit{reciprocal peer teaching}, and \textit{critical debate}, in a dual-agent, chat-based mathematical problem-solving environment using the OpenAI GPT-4o model. Evaluated on the MATH dataset, our results show that dual-agent setups outperform single agents, with \textit{peer-to-peer collaboration} achieving the highest accuracy. Dialogue acts like statements, acknowledgment, and hints play a key role in collaborative problem-solving. While multi-agent frameworks enhance computational tasks, effective communication strategies are essential for tackling complex problems in AI education. 

---
# Learning from Heterogeneity: Generalizing Dynamic Facial Expression Recognition via Distributionally Robust Optimization 

**Authors**: Feng-Qi Cui, Anyang Tong, Jinyang Huang, Jie Zhang, Dan Guo, Zhi Liu, Meng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2507.15765)  

**Abstract**: Dynamic Facial Expression Recognition (DFER) plays a critical role in affective computing and human-computer interaction. Although existing methods achieve comparable performance, they inevitably suffer from performance degradation under sample heterogeneity caused by multi-source data and individual expression variability. To address these challenges, we propose a novel framework, called Heterogeneity-aware Distributional Framework (HDF), and design two plug-and-play modules to enhance time-frequency modeling and mitigate optimization imbalance caused by hard samples. Specifically, the Time-Frequency Distributional Attention Module (DAM) captures both temporal consistency and frequency robustness through a dual-branch attention design, improving tolerance to sequence inconsistency and visual style shifts. Then, based on gradient sensitivity and information bottleneck principles, an adaptive optimization module Distribution-aware Scaling Module (DSM) is introduced to dynamically balance classification and contrastive losses, enabling more stable and discriminative representation learning. Extensive experiments on two widely used datasets, DFEW and FERV39k, demonstrate that HDF significantly improves both recognition accuracy and robustness. Our method achieves superior weighted average recall (WAR) and unweighted average recall (UAR) while maintaining strong generalization across diverse and imbalanced scenarios. Codes are released at this https URL. 

---
