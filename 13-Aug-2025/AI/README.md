# BrowseMaster: Towards Scalable Web Browsing via Tool-Augmented Programmatic Agent Pair 

**Authors**: Xianghe Pang, Shuo Tang, Rui Ye, Yuwen Du, Yaxin Du, Siheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.09129)  

**Abstract**: Effective information seeking in the vast and ever-growing digital landscape requires balancing expansive search with strategic reasoning. Current large language model (LLM)-based agents struggle to achieve this balance due to limitations in search breadth and reasoning depth, where slow, serial querying restricts coverage of relevant sources and noisy raw inputs disrupt the continuity of multi-step reasoning. To address these challenges, we propose BrowseMaster, a scalable framework built around a programmatically augmented planner-executor agent pair. The planner formulates and adapts search strategies based on task constraints, while the executor conducts efficient, targeted retrieval to supply the planner with concise, relevant evidence. This division of labor preserves coherent, long-horizon reasoning while sustaining broad and systematic exploration, overcoming the trade-off that limits existing agents. Extensive experiments on challenging English and Chinese benchmarks show that BrowseMaster consistently outperforms open-source and proprietary baselines, achieving scores of 30.0 on BrowseComp-en and 46.5 on BrowseComp-zh, which demonstrates its strong capability in complex, reasoning-heavy information-seeking tasks at scale. 

---
# OpenCUA: Open Foundations for Computer-Use Agents 

**Authors**: Xinyuan Wang, Bowen Wang, Dunjie Lu, Junlin Yang, Tianbao Xie, Junli Wang, Jiaqi Deng, Xiaole Guo, Yiheng Xu, Chen Henry Wu, Zhennan Shen, Zhuokai Li, Ryan Li, Xiaochuan Li, Junda Chen, Boyuan Zheng, Peihang Li, Fangyu Lei, Ruisheng Cao, Yeqiao Fu, Dongchan Shin, Martin Shin, Jiarui Hu, Yuyan Wang, Jixuan Chen, Yuxiao Ye, Danyang Zhang, Dikang Du, Hao Hu, Huarong Chen, Zaida Zhou, Yipu Wang, Heng Wang, Diyi Yang, Victor Zhong, Flood Sung, Y.Charles, Zhilin Yang, Tao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09123)  

**Abstract**: Vision-language models have demonstrated impressive capabilities as computer-use agents (CUAs) capable of automating diverse computer tasks. As their commercial potential grows, critical details of the most capable CUA systems remain closed. As these agents will increasingly mediate digital interactions and execute consequential decisions on our behalf, the research community needs access to open CUA frameworks to study their capabilities, limitations, and risks. To bridge this gap, we propose OpenCUA, a comprehensive open-source framework for scaling CUA data and foundation models. Our framework consists of: (1) an annotation infrastructure that seamlessly captures human computer-use demonstrations; (2) AgentNet, the first large-scale computer-use task dataset spanning 3 operating systems and 200+ applications and websites; (3) a scalable pipeline that transforms demonstrations into state-action pairs with reflective long Chain-of-Thought reasoning that sustain robust performance gains as data scales. Our end-to-end agent models demonstrate strong performance across CUA benchmarks. In particular, OpenCUA-32B achieves an average success rate of 34.8% on OSWorld-Verified, establishing a new state-of-the-art (SOTA) among open-source models and surpassing OpenAI CUA (GPT-4o). Further analysis confirms that our approach generalizes well across domains and benefits significantly from increased test-time computation. We release our annotation tool, datasets, code, and models to build open foundations for further CUA research. 

---
# SMA: Who Said That? Auditing Membership Leakage in Semi-Black-box RAG Controlling 

**Authors**: Shixuan Sun, Siyuan Liang, Ruoyu Chen, Jianjie Huang, Jingzhi Li, Xiaochun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.09105)  

**Abstract**: Retrieval-Augmented Generation (RAG) and its Multimodal Retrieval-Augmented Generation (MRAG) significantly improve the knowledge coverage and contextual understanding of Large Language Models (LLMs) by introducing external knowledge sources. However, retrieval and multimodal fusion obscure content provenance, rendering existing membership inference methods unable to reliably attribute generated outputs to pre-training, external retrieval, or user input, thus undermining privacy leakage accountability
To address these challenges, we propose the first Source-aware Membership Audit (SMA) that enables fine-grained source attribution of generated content in a semi-black-box setting with retrieval control this http URL address the environmental constraints of semi-black-box auditing, we further design an attribution estimation mechanism based on zero-order optimization, which robustly approximates the true influence of input tokens on the output through large-scale perturbation sampling and ridge regression modeling. In addition, SMA introduces a cross-modal attribution technique that projects image inputs into textual descriptions via MLLMs, enabling token-level attribution in the text modality, which for the first time facilitates membership inference on image retrieval traces in MRAG systems. This work shifts the focus of membership inference from 'whether the data has been memorized' to 'where the content is sourced from', offering a novel perspective for auditing data provenance in complex generative systems. 

---
# CVCM Track Circuits Pre-emptive Failure Diagnostics for Predictive Maintenance Using Deep Neural Networks 

**Authors**: Debdeep Mukherjee, Eduardo Di Santi, Clément Lefebvre, Nenad Mijatovic, Victor Martin, Thierry Josse, Jonathan Brown, Kenza Saiah  

**Link**: [PDF](https://arxiv.org/pdf/2508.09054)  

**Abstract**: Track circuits are critical for railway operations, acting as the main signalling sub-system to locate trains. Continuous Variable Current Modulation (CVCM) is one such technology. Like any field-deployed, safety-critical asset, it can fail, triggering cascading disruptions. Many failures originate as subtle anomalies that evolve over time, often not visually apparent in monitored signals. Conventional approaches, which rely on clear signal changes, struggle to detect them early. Early identification of failure types is essential to improve maintenance planning, minimising downtime and revenue loss. Leveraging deep neural networks, we propose a predictive maintenance framework that classifies anomalies well before they escalate into failures. Validated on 10 CVCM failure cases across different installations, the method is ISO-17359 compliant and outperforms conventional techniques, achieving 99.31% overall accuracy with detection within 1% of anomaly onset. Through conformal prediction, we provide uncertainty estimates, reaching 99% confidence with consistent coverage across classes. Given CVCMs global deployment, the approach is scalable and adaptable to other track circuits and railway systems, enhancing operational reliability. 

---
# A First Look at Predictability and Explainability of Pre-request Passenger Waiting Time in Ridesharing Systems 

**Authors**: Jie Wang, Guang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09027)  

**Abstract**: Passenger waiting time prediction plays a critical role in enhancing both ridesharing user experience and platform efficiency. While most existing research focuses on post-request waiting time prediction with knowing the matched driver information, pre-request waiting time prediction (i.e., before submitting a ride request and without matching a driver) is also important, as it enables passengers to plan their trips more effectively and enhance the experience of both passengers and drivers. However, it has not been fully studied by existing works. In this paper, we take the first step toward understanding the predictability and explainability of pre-request passenger waiting time in ridesharing systems. Particularly, we conduct an in-depth data-driven study to investigate the impact of demand&supply dynamics on passenger waiting time. Based on this analysis and feature engineering, we propose FiXGBoost, a novel feature interaction-based XGBoost model designed to predict waiting time without knowing the assigned driver information. We further perform an importance analysis to quantify the contribution of each factor. Experiments on a large-scale real-world ridesharing dataset including over 30 million trip records show that our FiXGBoost can achieve a good performance for pre-request passenger waiting time prediction with high explainability. 

---
# Activation Steering for Bias Mitigation: An Interpretable Approach to Safer LLMs 

**Authors**: Shivam Dubey  

**Link**: [PDF](https://arxiv.org/pdf/2508.09019)  

**Abstract**: As large language models (LLMs) become more integrated into societal systems, the risk of them perpetuating and amplifying harmful biases becomes a critical safety concern. Traditional methods for mitigating bias often rely on data filtering or post-hoc output moderation, which treat the model as an opaque black box. In this work, we introduce a complete, end-to-end system that uses techniques from mechanistic interpretability to both identify and actively mitigate bias directly within a model's internal workings. Our method involves two primary stages. First, we train linear "probes" on the internal activations of a model to detect the latent representations of various biases (e.g., gender, race, age). Our experiments on \texttt{gpt2-large} demonstrate that these probes can identify biased content with near-perfect accuracy, revealing that bias representations become most salient in the model's later layers. Second, we leverage these findings to compute "steering vectors" by contrasting the model's activation patterns for biased and neutral statements. By adding these vectors during inference, we can actively steer the model's generative process away from producing harmful, stereotypical, or biased content in real-time. We demonstrate the efficacy of this activation steering technique, showing that it successfully alters biased completions toward more neutral alternatives. We present our work as a robust and reproducible system that offers a more direct and interpretable approach to building safer and more accountable LLMs. 

---
# Intrinsic Memory Agents: Heterogeneous Multi-Agent LLM Systems through Structured Contextual Memory 

**Authors**: Sizhe Yuen, Francisco Gomez Medina, Ting Su, Yali Du, Adam J. Sobey  

**Link**: [PDF](https://arxiv.org/pdf/2508.08997)  

**Abstract**: Multi-agent systems built on Large Language Models (LLMs) show exceptional promise for complex collaborative problem-solving, yet they face fundamental challenges stemming from context window limitations that impair memory consistency, role adherence, and procedural integrity. This paper introduces Intrinsic Memory Agents, a novel framework that addresses these limitations through structured agent-specific memories that evolve intrinsically with agent outputs. Specifically, our method maintains role-aligned memory templates that preserve specialized perspectives while focusing on task-relevant information. We benchmark our approach on the PDDL dataset, comparing its performance to existing state-of-the-art multi-agentic memory approaches and showing an improvement of 38.6\% with the highest token efficiency. An additional evaluation is performed on a complex data pipeline design task, we demonstrate that our approach produces higher quality designs when comparing 5 metrics: scalability, reliability, usability, cost-effectiveness and documentation with additional qualitative evidence of the improvements. Our findings suggest that addressing memory limitations through structured, intrinsic approaches can improve the capabilities of multi-agent LLM systems on structured planning tasks. 

---
# Prospect Theory Fails for LLMs: Revealing Instability of Decision-Making under Epistemic Uncertainty 

**Authors**: Rui Wang, Qihan Lin, Jiayu Liu, Qing Zong, Tianshi Zheng, Weiqi Wang, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2508.08992)  

**Abstract**: Prospect Theory (PT) models human decision-making under uncertainty, while epistemic markers (e.g., maybe) serve to express uncertainty in language. However, it remains largely unexplored whether Prospect Theory applies to contemporary Large Language Models and whether epistemic markers, which express human uncertainty, affect their decision-making behaviour. To address these research gaps, we design a three-stage experiment based on economic questionnaires. We propose a more general and precise evaluation framework to model LLMs' decision-making behaviour under PT, introducing uncertainty through the empirical probability values associated with commonly used epistemic markers in comparable contexts. We then incorporate epistemic markers into the evaluation framework based on their corresponding probability values to examine their influence on LLM decision-making behaviours. Our findings suggest that modelling LLMs' decision-making with PT is not consistently reliable, particularly when uncertainty is expressed in diverse linguistic forms. Our code is released in this https URL. 

---
# Safe Semantics, Unsafe Interpretations: Tackling Implicit Reasoning Safety in Large Vision-Language Models 

**Authors**: Wei Cai, Jian Zhao, Yuchu Jiang, Tianle Zhang, Xuelong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.08926)  

**Abstract**: Large Vision-Language Models face growing safety challenges with multimodal inputs. This paper introduces the concept of Implicit Reasoning Safety, a vulnerability in LVLMs. Benign combined inputs trigger unsafe LVLM outputs due to flawed or hidden reasoning. To showcase this, we developed Safe Semantics, Unsafe Interpretations, the first dataset for this critical issue. Our demonstrations show that even simple In-Context Learning with SSUI significantly mitigates these implicit multimodal threats, underscoring the urgent need to improve cross-modal implicit reasoning. 

---
# Compass-Thinker-7B Technical Report 

**Authors**: Anxiang Zeng, Haibo Zhang, Kaixiang Mo, Long Zhang, Shuman Liu, Yanhui Huang, Yawen Liu, Yuepeng Sheng, Yuwei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08909)  

**Abstract**: Recent R1-Zero-like research further demonstrates that reasoning extension has given large language models (LLMs) unprecedented reasoning capabilities, and Reinforcement Learning is the core tech- nology to elicit its complex reasoning. However, conducting RL experiments directly on hyperscale models involves high computational costs and resource demands, posing significant risks. We pro- pose the Compass-Thinker-7B model, which aims to explore the potential of Reinforcement Learn- ing with less computational resources and costs, and provides insights for further research into RL recipes for larger models. Compass-Thinker-7B is trained from an open source model through a spe- cially designed Reinforcement Learning Pipeline. we curate a dataset of 30k verifiable mathematics problems for the Reinforcement Learning Pipeline. By configuring data and training settings with dif- ferent difficulty distributions for different stages, the potential of the model is gradually released and the training efficiency is improved. Extensive evaluations show that Compass-Thinker-7B possesses exceptional reasoning potential, and achieves superior performance on mathematics compared to the same-sized RL this http URL in the challenging AIME2024 evaluation, Compass-Thinker-7B achieves 40% accuracy. 

---
# Reducing Cognitive Load in Multi-Agent Reinforcement Learning for Mathematical Problem Solving: Decoupling Reasoning and Code Generation 

**Authors**: Dayu Wang, Jiaye Yang, Weikang Li, Jiahui Liang, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.08882)  

**Abstract**: Current tool-integrated mathematical reasoning systems often adopt a single-agent paradigm, where one large language model handles problem reasoning, code generation, and code execution in an integrated workflow. While this design eases coordination, we hypothesize that it imposes cognitive load interference, as the agent must interleave long-horizon reasoning with precise program synthesis. We validate this hypothesis through a controlled comparison between a reasoning-only agent and a reasoning-plus-code agent, finding that the latter produces significantly fewer correct reasoning paths despite having tool-calling capabilities. To address this, we propose a dual-agent hybrid framework: a Reasoning Agent performs stepwise problem decomposition, and a Code Agent handles code generation and execution. Training combines imitation learning and reinforcement learning: the Code Agent receives strong rewards for matching intermediate ground-truth programs and weaker rewards for valid execution, while the Reasoning Agent is optimized chiefly via final-answer accuracy using advantage estimation to credit intermediate steps. This decoupled role design reduces cognitive interference and promotes stable reasoning-coding coordination. 

---
# Silicon Minds versus Human Hearts: The Wisdom of Crowds Beats the Wisdom of AI in Emotion Recognition 

**Authors**: Mustafa Akben, Vinayaka Gude, Haya Ajjan  

**Link**: [PDF](https://arxiv.org/pdf/2508.08830)  

**Abstract**: The ability to discern subtle emotional cues is fundamental to human social intelligence. As artificial intelligence (AI) becomes increasingly common, AI's ability to recognize and respond to human emotions is crucial for effective human-AI interactions. In particular, whether such systems can match or surpass human experts remains to be seen. However, the emotional intelligence of AI, particularly multimodal large language models (MLLMs), remains largely unexplored. This study evaluates the emotion recognition abilities of MLLMs using the Reading the Mind in the Eyes Test (RMET) and its multiracial counterpart (MRMET), and compares their performance against human participants. Results show that, on average, MLLMs outperform humans in accurately identifying emotions across both tests. This trend persists even when comparing performance across low, medium, and expert-level performing groups. Yet when we aggregate independent human decisions to simulate collective intelligence, human groups significantly surpass the performance of aggregated MLLM predictions, highlighting the wisdom of the crowd. Moreover, a collaborative approach (augmented intelligence) that combines human and MLLM predictions achieves greater accuracy than either humans or MLLMs alone. These results suggest that while MLLMs exhibit strong emotion recognition at the individual level, the collective intelligence of humans and the synergistic potential of human-AI collaboration offer the most promising path toward effective emotional AI. We discuss the implications of these findings for the development of emotionally intelligent AI systems and future research directions. 

---
# Efficient Agent: Optimizing Planning Capability for Multimodal Retrieval Augmented Generation 

**Authors**: Yuechen Wang, Yuming Qiao, Dan Meng, Jun Yang, Haonan Lu, Zhenyu Yang, Xudong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08816)  

**Abstract**: Multimodal Retrieval-Augmented Generation (mRAG) has emerged as a promising solution to address the temporal limitations of Multimodal Large Language Models (MLLMs) in real-world scenarios like news analysis and trending topics. However, existing approaches often suffer from rigid retrieval strategies and under-utilization of visual information. To bridge this gap, we propose E-Agent, an agent framework featuring two key innovations: a mRAG planner trained to dynamically orchestrate multimodal tools based on contextual reasoning, and a task executor employing tool-aware execution sequencing to implement optimized mRAG workflows. E-Agent adopts a one-time mRAG planning strategy that enables efficient information retrieval while minimizing redundant tool invocations. To rigorously assess the planning capabilities of mRAG systems, we introduce the Real-World mRAG Planning (RemPlan) benchmark. This novel benchmark contains both retrieval-dependent and retrieval-independent question types, systematically annotated with essential retrieval tools required for each instance. The benchmark's explicit mRAG planning annotations and diverse question design enhance its practical relevance by simulating real-world scenarios requiring dynamic mRAG decisions. Experiments across RemPlan and three established benchmarks demonstrate E-Agent's superiority: 13% accuracy gain over state-of-the-art mRAG methods while reducing redundant searches by 37%. 

---
# GRainsaCK: a Comprehensive Software Library for Benchmarking Explanations of Link Prediction Tasks on Knowledge Graphs 

**Authors**: Roberto Barile, Claudia d'Amato, Nicola Fanizzi  

**Link**: [PDF](https://arxiv.org/pdf/2508.08815)  

**Abstract**: Since Knowledge Graphs are often incomplete, link prediction methods are adopted for predicting missing facts. Scalable embedding based solutions are mostly adopted for this purpose, however, they lack comprehensibility, which may be crucial in several domains. Explanation methods tackle this issue by identifying supporting knowledge explaining the predicted facts. Regretfully, evaluating/comparing quantitatively the resulting explanations is challenging as there is no standard evaluation protocol and overall benchmarking resource. We fill this important gap by proposing GRainsaCK, a reusable software resource that fully streamlines all the tasks involved in benchmarking explanations, i.e., from model training to evaluation of explanations along the same evaluation protocol. Moreover, GRainsaCK furthers modularity/extensibility by implementing the main components as functions that can be easily replaced. Finally, fostering its reuse, we provide extensive documentation including a tutorial. 

---
# A Dual-Axis Taxonomy of Knowledge Editing for LLMs: From Mechanisms to Functions 

**Authors**: Amir Mohammad Salehoof, Ali Ramezani, Yadollah Yaghoobzadeh, Majid Nili Ahmadabadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.08795)  

**Abstract**: Large language models (LLMs) acquire vast knowledge from large text corpora, but this information can become outdated or inaccurate. Since retraining is computationally expensive, knowledge editing offers an efficient alternative -- modifying internal knowledge without full retraining. These methods aim to update facts precisely while preserving the model's overall capabilities. While existing surveys focus on the mechanism of editing (e.g., parameter changes vs. external memory), they often overlook the function of the knowledge being edited. This survey introduces a novel, complementary function-based taxonomy to provide a more holistic view. We examine how different mechanisms apply to various knowledge types -- factual, temporal, conceptual, commonsense, and social -- highlighting how editing effectiveness depends on the nature of the target knowledge. By organizing our review along these two axes, we map the current landscape, outline the strengths and limitations of existing methods, define the problem formally, survey evaluation tasks and datasets, and conclude with open challenges and future directions. 

---
# Designing Memory-Augmented AR Agents for Spatiotemporal Reasoning in Personalized Task Assistance 

**Authors**: Dongwook Choi, Taeyoon Kwon, Dongil Yang, Hyojun Kim, Jinyoung Yeo  

**Link**: [PDF](https://arxiv.org/pdf/2508.08774)  

**Abstract**: Augmented Reality (AR) systems are increasingly integrating foundation models, such as Multimodal Large Language Models (MLLMs), to provide more context-aware and adaptive user experiences. This integration has led to the development of AR agents to support intelligent, goal-directed interactions in real-world environments. While current AR agents effectively support immediate tasks, they struggle with complex multi-step scenarios that require understanding and leveraging user's long-term experiences and preferences. This limitation stems from their inability to capture, retain, and reason over historical user interactions in spatiotemporal contexts. To address these challenges, we propose a conceptual framework for memory-augmented AR agents that can provide personalized task assistance by learning from and adapting to user-specific experiences over time. Our framework consists of four interconnected modules: (1) Perception Module for multimodal sensor processing, (2) Memory Module for persistent spatiotemporal experience storage, (3) Spatiotemporal Reasoning Module for synthesizing past and present contexts, and (4) Actuator Module for effective AR communication. We further present an implementation roadmap, a future evaluation strategy, a potential target application and use cases to demonstrate the practical applicability of our framework across diverse domains. We aim for this work to motivate future research toward developing more intelligent AR systems that can effectively bridge user's interaction history with adaptive, context-aware task assistance. 

---
# Simulating Generative Social Agents via Theory-Informed Workflow Design 

**Authors**: Yuwei Yan, Jinghua Piao, Xiaochong Lan, Chenyang Shao, Pan Hui, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.08726)  

**Abstract**: Recent advances in large language models have demonstrated strong reasoning and role-playing capabilities, opening new opportunities for agent-based social simulations. However, most existing agents' implementations are scenario-tailored, without a unified framework to guide the design. This lack of a general social agent limits their ability to generalize across different social contexts and to produce consistent, realistic behaviors. To address this challenge, we propose a theory-informed framework that provides a systematic design process for LLM-based social agents. Our framework is grounded in principles from Social Cognition Theory and introduces three key modules: motivation, action planning, and learning. These modules jointly enable agents to reason about their goals, plan coherent actions, and adapt their behavior over time, leading to more flexible and contextually appropriate responses. Comprehensive experiments demonstrate that our theory-driven agents reproduce realistic human behavior patterns under complex conditions, achieving up to 75% lower deviation from real-world behavioral data across multiple fidelity metrics compared to classical generative baselines. Ablation studies further show that removing motivation, planning, or learning modules increases errors by 1.5 to 3.2 times, confirming their distinct and essential contributions to generating realistic and coherent social behaviors. 

---
# STELAR-VISION: Self-Topology-Aware Efficient Learning for Aligned Reasoning in Vision 

**Authors**: Chen Li, Han Zhang, Zhantao Yang, Fangyi Chen, Zihan Wang, Anudeepsekhar Bolimera, Marios Savvides  

**Link**: [PDF](https://arxiv.org/pdf/2508.08688)  

**Abstract**: Vision-language models (VLMs) have made significant strides in reasoning, yet they often struggle with complex multimodal tasks and tend to generate overly verbose outputs. A key limitation is their reliance on chain-of-thought (CoT) reasoning, despite many tasks benefiting from alternative topologies like trees or graphs. To address this, we introduce STELAR-Vision, a training framework for topology-aware reasoning. At its core is TopoAug, a synthetic data pipeline that enriches training with diverse topological structures. Using supervised fine-tuning and reinforcement learning, we post-train Qwen2VL models with both accuracy and efficiency in mind. Additionally, we propose Frugal Learning, which reduces output length with minimal accuracy loss. On MATH-V and VLM-S2H, STELAR-Vision improves accuracy by 9.7% over its base model and surpasses the larger Qwen2VL-72B-Instruct by 7.3%. On five out-of-distribution benchmarks, it outperforms Phi-4-Multimodal-Instruct by up to 28.4% and LLaMA-3.2-11B-Vision-Instruct by up to 13.2%, demonstrating strong generalization. Compared to Chain-Only training, our approach achieves 4.3% higher overall accuracy on in-distribution datasets and consistently outperforms across all OOD benchmarks. We have released datasets, and code will be available. 

---
# Aryabhata: An exam-focused language model for JEE Math 

**Authors**: Ritvik Rastogi, Sachin Dharashivkar, Sandeep Varma  

**Link**: [PDF](https://arxiv.org/pdf/2508.08665)  

**Abstract**: We present $\textbf{Aryabhata 1.0}$, a compact 7B parameter math reasoning model optimized for the Indian academic exam, the Joint Entrance Examination (JEE). Despite rapid progress in large language models (LLMs), current models often remain unsuitable for educational use. Aryabhata 1.0 is built by merging strong open-weight reasoning models, followed by supervised fine-tuning (SFT) with curriculum learning on verified chain-of-thought (CoT) traces curated through best-of-$n$ rejection sampling. To further boost performance, we apply reinforcement learning with verifiable rewards (RLVR) using A2C objective with group-relative advantage estimation alongwith novel exploration strategies such as $\textit{Adaptive Group Resizing}$ and $\textit{Temperature Scaling}$. Evaluated on both in-distribution (JEE Main 2025) and out-of-distribution (MATH, GSM8K) benchmarks, Aryabhata outperforms existing models in accuracy and efficiency, while offering pedagogically useful step-by-step reasoning. We release Aryabhata as a foundation model to advance exam-centric, open-source small language models. This marks our first open release for community feedback ($\href{this https URL}{Aryabhata\ 1.0\ on\ Hugging\ Face}$); PW is actively training future models to further improve learning outcomes for students. 

---
# Hybrid Node-Destroyer Model with Large Neighborhood Search for Solving the Capacitated Vehicle Routing Problem 

**Authors**: Bachtiar Herdianto, Romain Billot, Flavien Lucas, Marc Sevaux, Daniele Vigo  

**Link**: [PDF](https://arxiv.org/pdf/2508.08659)  

**Abstract**: In this research, we propose an iterative learning hybrid optimization solver developed to strengthen the performance of metaheuristic algorithms in solving the Capacitated Vehicle Routing Problem (CVRP). The iterative hybrid mechanism integrates the proposed Node-Destroyer Model, a machine learning hybrid model that utilized Graph Neural Networks (GNNs) such identifies and selects customer nodes to guide the Large Neighborhood Search (LNS) operator within the metaheuristic optimization frameworks. This model leverages the structural properties of the problem and solution that can be represented as a graph, to guide strategic selections concerning node removal. The proposed approach reduces operational complexity and scales down the search space involved in the optimization process. The hybrid approach is applied specifically to the CVRP and does not require retraining across problem instances of different sizes. The proposed hybrid mechanism is able to improve the performance of baseline metaheuristic algorithms. Our approach not only enhances the solution quality for standard CVRP benchmarks but also proves scalability on very large-scale instances with up to 30,000 customer nodes. Experimental evaluations on benchmark datasets show that the proposed hybrid mechanism is capable of improving different baseline algorithms, achieving better quality of solutions under similar settings. 

---
# Prompt-and-Check: Using Large Language Models to Evaluate Communication Protocol Compliance in Simulation-Based Training 

**Authors**: Vishakha Lall, Yisi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08652)  

**Abstract**: Accurate evaluation of procedural communication compliance is essential in simulation-based training, particularly in safety-critical domains where adherence to compliance checklists reflects operational competence. This paper explores a lightweight, deployable approach using prompt-based inference with open-source large language models (LLMs) that can run efficiently on consumer-grade GPUs. We present Prompt-and-Check, a method that uses context-rich prompts to evaluate whether each checklist item in a protocol has been fulfilled, solely based on transcribed verbal exchanges. We perform a case study in the maritime domain with participants performing an identical simulation task, and experiment with models such as LLama 2 7B, LLaMA 3 8B and Mistral 7B, running locally on an RTX 4070 GPU. For each checklist item, a prompt incorporating relevant transcript excerpts is fed into the model, which outputs a compliance judgment. We assess model outputs against expert-annotated ground truth using classification accuracy and agreement scores. Our findings demonstrate that prompting enables effective context-aware reasoning without task-specific training. This study highlights the practical utility of LLMs in augmenting debriefing, performance feedback, and automated assessment in training environments. 

---
# P-CAFE: Personalized Cost-Aware Incremental Feature Selection For Electronic Health Records 

**Authors**: Naama Kashani, Mira Cohen, Uri Shaham  

**Link**: [PDF](https://arxiv.org/pdf/2508.08646)  

**Abstract**: Electronic Health Records (EHR) have revolutionized healthcare by digitizing patient data, improving accessibility, and streamlining clinical workflows. However, extracting meaningful insights from these complex and multimodal datasets remains a significant challenge for researchers. Traditional feature selection methods often struggle with the inherent sparsity and heterogeneity of EHR data, especially when accounting for patient-specific variations and feature costs in clinical applications. To address these challenges, we propose a novel personalized, online and cost-aware feature selection framework tailored specifically for EHR datasets. The features are aquired in an online fashion for individual patients, incorporating budgetary constraints and feature variability costs. The framework is designed to effectively manage sparse and multimodal data, ensuring robust and scalable performance in diverse healthcare contexts. A primary application of our proposed method is to support physicians' decision making in patient screening scenarios. By guiding physicians toward incremental acquisition of the most informative features within budget constraints, our approach aims to increase diagnostic confidence while optimizing resource utilization. 

---
# Diminution: On Reducing the Size of Grounding ASP Programs 

**Authors**: HuanYu Yang, Fengming Zhu, YangFan Wu, Jianmin Ji  

**Link**: [PDF](https://arxiv.org/pdf/2508.08633)  

**Abstract**: Answer Set Programming (ASP) is often hindered by the grounding bottleneck: large Herbrand universes generate ground programs so large that solving becomes difficult. Many methods employ ad-hoc heuristics to improve grounding performance, motivating the need for a more formal and generalizable strategy. We introduce the notion of diminution, defined as a selected subset of the Herbrand universe used to generate a reduced ground program before solving. We give a formal definition of diminution, analyze its key properties, and study the complexity of identifying it. We use a specific encoding that enables off-the-shelf ASP solver to evaluate candidate subsets. Our approach integrates seamlessly with existing grounders via domain predicates. In extensive experiments on five benchmarks, applying diminutions selected by our strategy yields significant performance improvements, reducing grounding time by up to 70% on average and decreasing the size of grounding files by up to 85%. These results demonstrate that leveraging diminutions constitutes a robust and general-purpose approach for alleviating the grounding bottleneck in ASP. 

---
# AgriGPT: a Large Language Model Ecosystem for Agriculture 

**Authors**: Bo Yang, Yu Zhang, Lanfei Feng, Yunkui Chen, Jianyu Zhang, Xiao Xu, Nueraili Aierken, Yurui Li, Yuxuan Chen, Guijun Yang, Yong He, Runhe Huang, Shijian Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.08632)  

**Abstract**: Despite the rapid progress of Large Language Models (LLMs), their application in agriculture remains limited due to the lack of domain-specific models, curated datasets, and robust evaluation frameworks. To address these challenges, we propose AgriGPT, a domain-specialized LLM ecosystem for agricultural usage. At its core, we design a multi-agent scalable data engine that systematically compiles credible data sources into Agri-342K, a high-quality, standardized question-answer (QA) dataset. Trained on this dataset, AgriGPT supports a broad range of agricultural stakeholders, from practitioners to policy-makers. To enhance factual grounding, we employ Tri-RAG, a three-channel Retrieval-Augmented Generation framework combining dense retrieval, sparse retrieval, and multi-hop knowledge graph reasoning, thereby improving the LLM's reasoning reliability. For comprehensive evaluation, we introduce AgriBench-13K, a benchmark suite comprising 13 tasks with varying types and complexities. Experiments demonstrate that AgriGPT significantly outperforms general-purpose LLMs on both domain adaptation and reasoning. Beyond the model itself, AgriGPT represents a modular and extensible LLM ecosystem for agriculture, comprising structured data construction, retrieval-enhanced generation, and domain-specific evaluation. This work provides a generalizable framework for developing scientific and industry-specialized LLMs. All models, datasets, and code will be released to empower agricultural communities, especially in underserved regions, and to promote open, impactful research. 

---
# UGM2N: An Unsupervised and Generalizable Mesh Movement Network via M-Uniform Loss 

**Authors**: Zhichao Wang, Xinhai Chen, Qinglin Wang, Xiang Gao, Qingyang Zhang, Menghan Jia, Xiang Zhang, Jie Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08615)  

**Abstract**: Partial differential equations (PDEs) form the mathematical foundation for modeling physical systems in science and engineering, where numerical solutions demand rigorous accuracy-efficiency tradeoffs. Mesh movement techniques address this challenge by dynamically relocating mesh nodes to rapidly-varying regions, enhancing both simulation accuracy and computational efficiency. However, traditional approaches suffer from high computational complexity and geometric inflexibility, limiting their applicability, and existing supervised learning-based approaches face challenges in zero-shot generalization across diverse PDEs and mesh this http URL this paper, we present an Unsupervised and Generalizable Mesh Movement Network (UGM2N). We first introduce unsupervised mesh adaptation through localized geometric feature learning, eliminating the dependency on pre-adapted meshes. We then develop a physics-constrained loss function, M-Uniform loss, that enforces mesh equidistribution at the nodal this http URL results demonstrate that the proposed network exhibits equation-agnostic generalization and geometric independence in efficient mesh adaptation. It demonstrates consistent superiority over existing methods, including robust performance across diverse PDEs and mesh geometries, scalability to multi-scale resolutions and guaranteed error reduction without mesh tangling. 

---
# SynLLM: A Comparative Analysis of Large Language Models for Medical Tabular Synthetic Data Generation via Prompt Engineering 

**Authors**: Arshia Ilaty, Hossein Shirazi, Hajar Homayouni  

**Link**: [PDF](https://arxiv.org/pdf/2508.08529)  

**Abstract**: Access to real-world medical data is often restricted due to privacy regulations, posing a significant barrier to the advancement of healthcare research. Synthetic data offers a promising alternative; however, generating realistic, clinically valid, and privacy-conscious records remains a major challenge. Recent advancements in Large Language Models (LLMs) offer new opportunities for structured data generation; however, existing approaches frequently lack systematic prompting strategies and comprehensive, multi-dimensional evaluation frameworks.
In this paper, we present SynLLM, a modular framework for generating high-quality synthetic medical tabular data using 20 state-of-the-art open-source LLMs, including LLaMA, Mistral, and GPT variants, guided by structured prompts. We propose four distinct prompt types, ranging from example-driven to rule-based constraints, that encode schema, metadata, and domain knowledge to control generation without model fine-tuning. Our framework features a comprehensive evaluation pipeline that rigorously assesses generated data across statistical fidelity, clinical consistency, and privacy preservation.
We evaluate SynLLM across three public medical datasets, including Diabetes, Cirrhosis, and Stroke, using 20 open-source LLMs. Our results show that prompt engineering significantly impacts data quality and privacy risk, with rule-based prompts achieving the best privacy-quality balance. SynLLM establishes that, when guided by well-designed prompts and evaluated with robust, multi-metric criteria, LLMs can generate synthetic medical data that is both clinically plausible and privacy-aware, paving the way for safer and more effective data sharing in healthcare research. 

---
# GVGAI-LLM: Evaluating Large Language Model Agents with Infinite Games 

**Authors**: Yuchen Li, Cong Lin, Muhammad Umair Nasir, Philip Bontrager, Jialin Liu, Julian Togelius  

**Link**: [PDF](https://arxiv.org/pdf/2508.08501)  

**Abstract**: We introduce GVGAI-LLM, a video game benchmark for evaluating the reasoning and problem-solving capabilities of large language models (LLMs). Built on the General Video Game AI framework, it features a diverse collection of arcade-style games designed to test a model's ability to handle tasks that differ from most existing LLM benchmarks. The benchmark leverages a game description language that enables rapid creation of new games and levels, helping to prevent overfitting over time. Each game scene is represented by a compact set of ASCII characters, allowing for efficient processing by language models. GVGAI-LLM defines interpretable metrics, including the meaningful step ratio, step efficiency, and overall score, to assess model behavior. Through zero-shot evaluations across a broad set of games and levels with diverse challenges and skill depth, we reveal persistent limitations of LLMs in spatial reasoning and basic planning. Current models consistently exhibit spatial and logical errors, motivating structured prompting and spatial grounding techniques. While these interventions lead to partial improvements, the benchmark remains very far from solved. GVGAI-LLM provides a reproducible testbed for advancing research on language model capabilities, with a particular emphasis on agentic behavior and contextual reasoning. 

---
# Large Language Models as Oracles for Ontology Alignment 

**Authors**: Sviatoslav Lushnei, Dmytro Shumskyi, Severyn Shykula, Ernesto Jimenez-Ruiz, Artur d'Avila Garcez  

**Link**: [PDF](https://arxiv.org/pdf/2508.08500)  

**Abstract**: Ontology alignment plays a crucial role in integrating diverse data sources across domains. There is a large plethora of systems that tackle the ontology alignment problem, yet challenges persist in producing highly quality correspondences among a set of input ontologies. Human-in-the-loop during the alignment process is essential in applications requiring very accurate mappings. User involvement is, however, expensive when dealing with large ontologies. In this paper, we explore the feasibility of using Large Language Models (LLM) as an alternative to the domain expert. The use of the LLM focuses only on the validation of the subset of correspondences where an ontology alignment system is very uncertain. We have conducted an extensive evaluation over several matching tasks of the Ontology Alignment Evaluation Initiative (OAEI), analysing the performance of several state-of-the-art LLMs using different ontology-driven prompt templates. The LLM results are also compared against simulated Oracles with variable error rates. 

---
# POMO+: Leveraging starting nodes in POMO for solving Capacitated Vehicle Routing Problem 

**Authors**: Szymon Jakubicz, Karol Kuźniak, Jan Wawszczak, Paweł Gora  

**Link**: [PDF](https://arxiv.org/pdf/2508.08493)  

**Abstract**: In recent years, reinforcement learning (RL) methods have emerged as a promising approach for solving combinatorial problems. Among RL-based models, POMO has demonstrated strong performance on a variety of tasks, including variants of the Vehicle Routing Problem (VRP). However, there is room for improvement for these tasks. In this work, we improved POMO, creating a method (\textbf{POMO+}) that leverages the initial nodes to find a solution in a more informed way. We ran experiments on our new model and observed that our solution converges faster and achieves better results. We validated our models on the CVRPLIB dataset and noticed improvements in problem instances with up to 100 customers. We hope that our research in this project can lead to further advancements in the field. 

---
# Beyond Ordinal Preferences: Why Alignment Needs Cardinal Human Feedback 

**Authors**: Parker Whitfill, Stewy Slocum  

**Link**: [PDF](https://arxiv.org/pdf/2508.08486)  

**Abstract**: Alignment techniques for LLMs rely on optimizing preference-based objectives -- where these preferences are typically elicited as ordinal, binary choices between responses. Recent work has focused on improving label quality or mitigating particular biases, but we identify a more fundamental limitation: these methods collect the wrong kind of data. We prove an impossibility result: no algorithm relying solely on ordinal comparisons can systematically recover the most preferred model. Intuitively, ordinal data lacks the information needed to resolve tradeoffs -- e.g., fixing a factual error on one prompt versus improving style on another. We show that selecting the optimal model requires recovering preferences over \emph{models} (rather than just responses), which can only be identified given cardinal feedback about response quality. To address this, we collect and publicly release a dataset of 25,000 cardinal judgments using willingness-to-pay elicitations, a well-established tool from experimental economics. Empirically, we find that incorporating cardinal feedback into preference fine-tuning allows models to prioritize high-impact improvements and outperform ordinal-only methods on downstream benchmarks, such as Arena-Hard. 

---
# A Fast GRASP Metaheuristic for the Trigger Arc TSP with MIP-Based Construction and Multi-Neighborhood Local Search 

**Authors**: Joan Salvà Soler, Grégoire de Lambertye  

**Link**: [PDF](https://arxiv.org/pdf/2508.08477)  

**Abstract**: The Trigger Arc Traveling Salesman Problem (TA-TSP) extends the classical TSP by introducing dynamic arc costs that change when specific \textit{trigger} arcs are traversed, modeling scenarios such as warehouse operations with compactable storage systems. This paper introduces a GRASP-based metaheuristic that combines multiple construction heuristics with a multi-neighborhood local search. The construction phase uses mixed-integer programming (MIP) techniques to transform the TA-TSP into a sequence of tailored TSP instances, while the improvement phase applies 2-Opt, Swap, and Relocate operators. Computational experiments on MESS 2024 competition instances achieved average optimality gaps of 0.77\% and 0.40\% relative to the best-known solutions within a 60-second limit. On smaller, synthetically generated datasets, the method produced solutions 11.3\% better than the Gurobi solver under the same time constraints. The algorithm finished in the top three at MESS 2024, demonstrating its suitability for real-time routing applications with state-dependent travel costs. 

---
# OverFill: Two-Stage Models for Efficient Language Model Decoding 

**Authors**: Woojeong Kim, Junxiong Wang, Jing Nathan Yan, Mohamed Abdelfattah, Alexander M. Rush  

**Link**: [PDF](https://arxiv.org/pdf/2508.08446)  

**Abstract**: Large language models (LLMs) excel across diverse tasks but face significant deployment challenges due to high inference costs. LLM inference comprises prefill (compute-bound) and decode (memory-bound) stages, with decode dominating latency particularly for long sequences. Current decoder-only models handle both stages uniformly, despite their distinct computational profiles. We propose OverFill, which decouples these stages to optimize accuracy-efficiency tradeoffs. OverFill begins with a full model for prefill, processing system and user inputs in parallel. It then switches to a dense pruned model, while generating tokens sequentially. Leveraging more compute during prefill, OverFill improves generation quality with minimal latency overhead. Our 3B-to-1B OverFill configuration outperforms 1B pruned models by 83.2%, while the 8B-to-3B configuration improves over 3B pruned models by 79.2% on average across standard benchmarks. OverFill matches the performance of same-sized models trained from scratch, while using significantly less training data. Our code is available at this https URL. 

---
# Solver-Aided Expansion of Loops to Avoid Generate-and-Test 

**Authors**: Niklas Dewally, Özgür Akgün  

**Link**: [PDF](https://arxiv.org/pdf/2508.08442)  

**Abstract**: Constraint modelling languages like MiniZinc and Essence rely on unrolling loops (in the form of quantified expressions and comprehensions) during compilation. Standard approaches generate all combinations of induction variables and use partial evaluation to discard those that simplify to identity elements of associative-commutative operators (e.g. true for conjunction, 0 for summation). This can be inefficient for problems where most combinations are ultimately irrelevant. We present a method that avoids full enumeration by using a solver to compute only the combinations required to generate the final set of constraints. The resulting model is identical to that produced by conventional flattening, but compilation can be significantly faster. This improves the efficiency of translating high-level user models into solver-ready form, particularly when induction variables range over large domains with selective preconditions. 

---
# Bilevel MCTS for Amortized O(1) Node Selection in Classical Planning 

**Authors**: Masataro Asai  

**Link**: [PDF](https://arxiv.org/pdf/2508.08385)  

**Abstract**: We study an efficient implementation of Multi-Armed Bandit (MAB)-based Monte-Carlo Tree Search (MCTS) for classical planning. One weakness of MCTS is that it spends a significant time deciding which node to expand next. While selecting a node from an OPEN list with $N$ nodes has $O(1)$ runtime complexity with traditional array-based priority-queues for dense integer keys, the tree-based OPEN list used by MCTS requires $O(\log N)$, which roughly corresponds to the search depth $d$. In classical planning, $d$ is arbitrarily large (e.g., $2^k-1$ in $k$-disk Tower-of-Hanoi) and the runtime for node selection is significant, unlike in game tree search, where the cost is negligible compared to the node evaluation (rollouts) because $d$ is inherently limited by the game (e.g., $d\leq 361$ in Go). To improve this bottleneck, we propose a bilevel modification to MCTS that runs a best-first search from each selected leaf node with an expansion budget proportional to $d$, which achieves amortized $O(1)$ runtime for node selection, equivalent to the traditional queue-based OPEN list. In addition, we introduce Tree Collapsing, an enhancement that reduces action selection steps and further improves the performance. 

---
# UrzaGPT: LoRA-Tuned Large Language Models for Card Selection in Collectible Card Games 

**Authors**: Timo Bertram  

**Link**: [PDF](https://arxiv.org/pdf/2508.08382)  

**Abstract**: Collectible card games (CCGs) are a difficult genre for AI due to their partial observability, long-term decision-making, and evolving card sets. Due to this, current AI models perform vastly worse than human players at CCG tasks such as deckbuilding and gameplay. In this work, we introduce $\textit{UrzaGPT}$, a domain-adapted large language model that recommends real-time drafting decisions in $\textit{Magic: The Gathering}$. Starting from an open-weight LLM, we use Low-Rank Adaptation fine-tuning on a dataset of annotated draft logs. With this, we leverage the language modeling capabilities of LLM, and can quickly adapt to different expansions of the game. We benchmark $\textit{UrzaGPT}$ in comparison to zero-shot LLMs and the state-of-the-art domain-specific model. Untuned, small LLMs like Llama-3-8B are completely unable to draft, but the larger GPT-4o achieves a zero-shot performance of $43\%$. Using UrzaGPT to fine-tune smaller models, we achieve an accuracy of $66.2\%$ using only 10,000 steps. Despite this not reaching the capability of domain-specific models, we show that solely using LLMs to draft is possible and conclude that using LLMs can enable performant, general, and update-friendly drafting AIs in the future. 

---
# What Breaks Knowledge Graph based RAG? Empirical Insights into Reasoning under Incomplete Knowledge 

**Authors**: Dongzhuoran Zhou, Yuqicheng Zhu, Xiaxia Wang, Hongkuan Zhou, Yuan He, Jiaoyan Chen, Evgeny Kharlamov, Steffen Staab  

**Link**: [PDF](https://arxiv.org/pdf/2508.08344)  

**Abstract**: Knowledge Graph-based Retrieval-Augmented Generation (KG-RAG) is an increasingly explored approach for combining the reasoning capabilities of large language models with the structured evidence of knowledge graphs. However, current evaluation practices fall short: existing benchmarks often include questions that can be directly answered using existing triples in KG, making it unclear whether models perform reasoning or simply retrieve answers directly. Moreover, inconsistent evaluation metrics and lenient answer matching criteria further obscure meaningful comparisons. In this work, we introduce a general method for constructing benchmarks, together with an evaluation protocol, to systematically assess KG-RAG methods under knowledge incompleteness. Our empirical results show that current KG-RAG methods have limited reasoning ability under missing knowledge, often rely on internal memorization, and exhibit varying degrees of generalization depending on their design. 

---
# First Ask Then Answer: A Framework Design for AI Dialogue Based on Supplementary Questioning with Large Language Models 

**Authors**: Chuanruo Fu, Yuncheng Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.08308)  

**Abstract**: Large Language Models (LLMs) often struggle to deliver accurate and actionable answers when user-provided information is incomplete or ill-specified. We propose a new interaction paradigm, First Ask Then Answer (FATA), in which, through prompt words, LLMs are guided to proactively generate multidimensional supplementary questions for users prior to response generation. Subsequently, by integrating user-provided supplementary information with the original query through sophisticated prompting techniques, we achieve substantially improved response quality and relevance. In contrast to existing clarification approaches -- such as the CLAM framework oriented to ambiguity and the self-interrogation Self-Ask method -- FATA emphasizes completeness (beyond mere disambiguation) and user participation (inviting human input instead of relying solely on model-internal reasoning). It also adopts a single-turn strategy: all clarifying questions are produced at once, thereby reducing dialogue length and improving efficiency. Conceptually, FATA uses the reasoning power of LLMs to scaffold user expression, enabling non-expert users to formulate more comprehensive and contextually relevant queries. To evaluate FATA, we constructed a multi-domain benchmark and compared it with two controls: a baseline prompt (B-Prompt) and a context-enhanced expert prompt (C-Prompt). Experimental results show that FATA outperforms B-Prompt by approximately 40% in aggregate metrics and exhibits a coefficient of variation 8% lower than C-Prompt, indicating superior stability. 

---
# LLM-BI: Towards Fully Automated Bayesian Inference with Large Language Models 

**Authors**: Yongchao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08300)  

**Abstract**: A significant barrier to the widespread adoption of Bayesian inference is the specification of prior distributions and likelihoods, which often requires specialized statistical expertise. This paper investigates the feasibility of using a Large Language Model (LLM) to automate this process. We introduce LLM-BI (Large Language Model-driven Bayesian Inference), a conceptual pipeline for automating Bayesian workflows. As a proof-of-concept, we present two experiments focused on Bayesian linear regression. In Experiment I, we demonstrate that an LLM can successfully elicit prior distributions from natural language. In Experiment II, we show that an LLM can specify the entire model structure, including both priors and the likelihood, from a single high-level problem description. Our results validate the potential of LLMs to automate key steps in Bayesian modeling, enabling the possibility of an automated inference pipeline for probabilistic programming. 

---
# An Efficient Application of Goal Programming to Tackle Multiobjective Problems with Recurring Fitness Landscapes 

**Authors**: Rodrigo Lankaites Pinheiro, Dario Landa-Silva, Wasakorn Laesanklang, Ademir Aparecido Constantino  

**Link**: [PDF](https://arxiv.org/pdf/2508.08297)  

**Abstract**: Many real-world applications require decision-makers to assess the quality of solutions while considering multiple conflicting objectives. Obtaining good approximation sets for highly constrained many-objective problems is often a difficult task even for modern multiobjective algorithms. In some cases, multiple instances of the problem scenario present similarities in their fitness landscapes. That is, there are recurring features in the fitness landscapes when searching for solutions to different problem instances. We propose a methodology to exploit this characteristic by solving one instance of a given problem scenario using computationally expensive multiobjective algorithms to obtain a good approximation set and then using Goal Programming with efficient single-objective algorithms to solve other instances of the same problem scenario. We use three goal-based objective functions and show that on benchmark instances of the multiobjective vehicle routing problem with time windows, the methodology is able to produce good results in short computation time. The methodology allows to combine the effectiveness of state-of-the-art multiobjective algorithms with the efficiency of goal programming to find good compromise solutions in problem scenarios where instances have similar fitness landscapes. 

---
# Topos Causal Models 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2508.08295)  

**Abstract**: We propose topos causal models (TCMs), a novel class of causal models that exploit the key properties of a topos category: they are (co)complete, meaning all (co)limits exist, they admit a subobject classifier, and allow exponential objects. The main goal of this paper is to show that these properties are central to many applications in causal inference. For example, subobject classifiers allow a categorical formulation of causal intervention, which creates sub-models. Limits and colimits allow causal diagrams of arbitrary complexity to be ``solved", using a novel interpretation of causal approximation. Exponential objects enable reasoning about equivalence classes of operations on causal models, such as covered edge reversal and causal homotopy. Analogous to structural causal models (SCMs), TCMs are defined by a collection of functions, each defining a ``local autonomous" causal mechanism that assemble to induce a unique global function from exogenous to endogenous variables. Since the category of TCMs is (co)complete, which we prove in this paper, every causal diagram has a ``solution" in the form of a (co)limit: this implies that any arbitrary causal model can be ``approximated" by some global function with respect to the morphisms going into or out of the diagram. Natural transformations are crucial in measuring the quality of approximation. In addition, we show that causal interventions are modeled by subobject classifiers: any sub-model is defined by a monic arrow into its parent model. Exponential objects permit reasoning about entire classes of causal equivalences and interventions. Finally, as TCMs form a topos, they admit an internal logic defined as a Mitchell-Benabou language with an associated Kripke-Joyal semantics. We show how to reason about causal models in TCMs using this internal logic. 

---
# Topos Theory for Generative AI and LLMs 

**Authors**: Sridhar Mahadevan  

**Link**: [PDF](https://arxiv.org/pdf/2508.08293)  

**Abstract**: We propose the design of novel categorical generative AI architectures (GAIAs) using topos theory, a type of category that is ``set-like": a topos has all (co)limits, is Cartesian closed, and has a subobject classifier. Previous theoretical results on the Transformer model have shown that it is a universal sequence-to-sequence function approximator, and dense in the space of all continuous functions with compact support on the Euclidean space of embeddings of tokens. Building on this theoretical result, we explore novel architectures for LLMs that exploit the property that the category of LLMs, viewed as functions, forms a topos. Previous studies of large language models (LLMs) have focused on daisy-chained linear architectures or mixture-of-experts. In this paper, we use universal constructions in category theory to construct novel LLM architectures based on new types of compositional structures. In particular, these new compositional structures are derived from universal properties of LLM categories, and include pullback, pushout, (co) equalizers, exponential objects, and subobject classifiers. We theoretically validate these new compositional structures by showing that the category of LLMs is (co)complete, meaning that all diagrams have solutions in the form of (co)limits. Building on this completeness result, we then show that the category of LLMs forms a topos, a ``set-like" category, which requires showing the existence of exponential objects as well as subobject classifiers. We use a functorial characterization of backpropagation to define a potential implementation of an LLM topos architecture. 

---
# Time Is a Feature: Exploiting Temporal Dynamics in Diffusion Language Models 

**Authors**: Wen Wang, Bozhen Fang, Chenchen Jing, Yongliang Shen, Yangyi Shen, Qiuyu Wang, Hao Ouyang, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.09138)  

**Abstract**: Diffusion large language models (dLLMs) generate text through iterative denoising, yet current decoding strategies discard rich intermediate predictions in favor of the final output. Our work here reveals a critical phenomenon, temporal oscillation, where correct answers often emerge in the middle process, but are overwritten in later denoising steps. To address this issue, we introduce two complementary methods that exploit temporal consistency: 1) Temporal Self-Consistency Voting, a training-free, test-time decoding strategy that aggregates predictions across denoising steps to select the most consistent output; and 2) a post-training method termed Temporal Consistency Reinforcement, which uses Temporal Semantic Entropy (TSE), a measure of semantic stability across intermediate predictions, as a reward signal to encourage stable generations. Empirical results across multiple benchmarks demonstrate the effectiveness of our approach. Using the negative TSE reward alone, we observe a remarkable average improvement of 24.7% on the Countdown dataset over an existing dLLM. Combined with the accuracy reward, we achieve absolute gains of 2.0% on GSM8K, 4.3% on MATH500, 6.6% on SVAMP, and 25.3% on Countdown, respectively. Our findings underscore the untapped potential of temporal dynamics in dLLMs and offer two simple yet effective tools to harness them. 

---
# Training-Free Text-Guided Color Editing with Multi-Modal Diffusion Transformer 

**Authors**: Zixin Yin, Xili Dai, Ling-Hao Chen, Deyu Zhou, Jianan Wang, Duomin Wang, Gang Yu, Lionel M. Ni, Heung-Yeung Shum  

**Link**: [PDF](https://arxiv.org/pdf/2508.09131)  

**Abstract**: Text-guided color editing in images and videos is a fundamental yet unsolved problem, requiring fine-grained manipulation of color attributes, including albedo, light source color, and ambient lighting, while preserving physical consistency in geometry, material properties, and light-matter interactions. Existing training-free methods offer broad applicability across editing tasks but struggle with precise color control and often introduce visual inconsistency in both edited and non-edited regions. In this work, we present ColorCtrl, a training-free color editing method that leverages the attention mechanisms of modern Multi-Modal Diffusion Transformers (MM-DiT). By disentangling structure and color through targeted manipulation of attention maps and value tokens, our method enables accurate and consistent color editing, along with word-level control of attribute intensity. Our method modifies only the intended regions specified by the prompt, leaving unrelated areas untouched. Extensive experiments on both SD3 and FLUX.1-dev demonstrate that ColorCtrl outperforms existing training-free approaches and achieves state-of-the-art performances in both edit quality and consistency. Furthermore, our method surpasses strong commercial models such as FLUX.1 Kontext Max and GPT-4o Image Generation in terms of consistency. When extended to video models like CogVideoX, our approach exhibits greater advantages, particularly in maintaining temporal coherence and editing stability. Finally, our method also generalizes to instruction-based editing diffusion models such as Step1X-Edit and FLUX.1 Kontext dev, further demonstrating its versatility. 

---
# Towards Universal Neural Inference 

**Authors**: Shreyas Bhat Brahmavar, Yang Li, Junier Oliva  

**Link**: [PDF](https://arxiv.org/pdf/2508.09100)  

**Abstract**: Real-world data often appears in diverse, disjoint forms -- with varying schemas, inconsistent semantics, and no fixed feature ordering -- making it challenging to build general-purpose models that can leverage information across datasets. We introduce ASPIRE, Arbitrary Set-based Permutation-Invariant Reasoning Engine, a Universal Neural Inference model for semantic reasoning and prediction over heterogeneous structured data. ASPIRE combines a permutation-invariant, set-based Transformer with a semantic grounding module that incorporates natural language descriptions, dataset metadata, and in-context examples to learn cross-dataset feature dependencies. This architecture allows ASPIRE to ingest arbitrary sets of feature--value pairs and support examples, align semantics across disjoint tables, and make predictions for any specified target. Once trained, ASPIRE generalizes to new inference tasks without additional tuning. In addition to delivering strong results across diverse benchmarks, ASPIRE naturally supports cost-aware active feature acquisition in an open-world setting, selecting informative features under test-time budget constraints for an arbitrary unseen dataset. These capabilities position ASPIRE as a step toward truly universal, semantics-aware inference over structured data. 

---
# SPARC: Soft Probabilistic Adaptive multi-interest Retrieval Model via Codebooks for recommender system 

**Authors**: Jialiang Shi, Yaguang Dou, Tian Qi  

**Link**: [PDF](https://arxiv.org/pdf/2508.09090)  

**Abstract**: Modeling multi-interests has arisen as a core problem in real-world RS. Current multi-interest retrieval methods pose three major challenges: 1) Interests, typically extracted from predefined external knowledge, are invariant. Failed to dynamically evolve with users' real-time consumption preferences. 2) Online inference typically employs an over-exploited strategy, mainly matching users' existing interests, lacking proactive exploration and discovery of novel and long-tail interests. To address these challenges, we propose a novel retrieval framework named SPARC(Soft Probabilistic Adaptive Retrieval Model via Codebooks). Our contribution is two folds. First, the framework utilizes Residual Quantized Variational Autoencoder (RQ-VAE) to construct a discretized interest space. It achieves joint training of the RQ-VAE with the industrial large scale recommendation model, mining behavior-aware interests that can perceive user feedback and evolve dynamically. Secondly, a probabilistic interest module that predicts the probability distribution over the entire dynamic and discrete interest space. This facilitates an efficient "soft-search" strategy during online inference, revolutionizing the retrieval paradigm from "passive matching" to "proactive exploration" and thereby effectively promoting interest discovery. Online A/B tests on an industrial platform with tens of millions daily active users, have achieved substantial gains in business metrics: +0.9% increase in user view duration, +0.4% increase in user page views (PV), and a +22.7% improvement in PV500(new content reaching 500 PVs in 24 hours). Offline evaluations are conducted on open-source Amazon Product datasets. Metrics, such as Recall@K and Normalized Discounted Cumulative Gain@K(NDCG@K), also showed consistent improvement. Both online and offline experiments validate the efficacy and practical value of the proposed method. 

---
# Dynamic Uncertainty-aware Multimodal Fusion for Outdoor Health Monitoring 

**Authors**: Zihan Fang, Zheng Lin, Senkang Hu, Yihang Tao, Yiqin Deng, Xianhao Chen, Yuguang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09085)  

**Abstract**: Outdoor health monitoring is essential to detect early abnormal health status for safeguarding human health and safety. Conventional outdoor monitoring relies on static multimodal deep learning frameworks, which requires extensive data training from scratch and fails to capture subtle health status changes. Multimodal large language models (MLLMs) emerge as a promising alternative, utilizing only small datasets to fine-tune pre-trained information-rich models for enabling powerful health status monitoring. Unfortunately, MLLM-based outdoor health monitoring also faces significant challenges: I) sensor data contains input noise stemming from sensor data acquisition and fluctuation noise caused by sudden changes in physiological signals due to dynamic outdoor environments, thus degrading the training performance; ii) current transformer based MLLMs struggle to achieve robust multimodal fusion, as they lack a design for fusing the noisy modality; iii) modalities with varying noise levels hinder accurate recovery of missing data from fluctuating distributions. To combat these challenges, we propose an uncertainty-aware multimodal fusion framework, named DUAL-Health, for outdoor health monitoring in dynamic and noisy environments. First, to assess the impact of noise, we accurately quantify modality uncertainty caused by input and fluctuation noise with current and temporal features. Second, to empower efficient muitimodal fusion with low-quality modalities,we customize the fusion weight for each modality based on quantified and calibrated uncertainty. Third, to enhance data recovery from fluctuating noisy modalities, we align modality distributions within a common semantic space. Extensive experiments demonstrate that our DUAL-Health outperforms state-of-the-art baselines in detection accuracy and robustness. 

---
# Can We Trust AI to Govern AI? Benchmarking LLM Performance on Privacy and AI Governance Exams 

**Authors**: Zane Witherspoon, Thet Mon Aye, YingYing Hao  

**Link**: [PDF](https://arxiv.org/pdf/2508.09036)  

**Abstract**: The rapid emergence of large language models (LLMs) has raised urgent questions across the modern workforce about this new technology's strengths, weaknesses, and capabilities. For privacy professionals, the question is whether these AI systems can provide reliable support on regulatory compliance, privacy program management, and AI governance. In this study, we evaluate ten leading open and closed LLMs, including models from OpenAI, Anthropic, Google DeepMind, Meta, and DeepSeek, by benchmarking their performance on industry-standard certification exams: CIPP/US, CIPM, CIPT, and AIGP from the International Association of Privacy Professionals (IAPP). Each model was tested using official sample exams in a closed-book setting and compared to IAPP's passing thresholds. Our findings show that several frontier models such as Gemini 2.5 Pro and OpenAI's GPT-5 consistently achieve scores exceeding the standards for professional human certification - demonstrating substantial expertise in privacy law, technical controls, and AI governance. The results highlight both the strengths and domain-specific gaps of current LLMs and offer practical insights for privacy officers, compliance leads, and technologists assessing the readiness of AI tools for high-stakes data governance roles. This paper provides an overview for professionals navigating the intersection of AI advancement and regulatory risk and establishes a machine benchmark based on human-centric evaluations. 

---
# Spatial Traces: Enhancing VLA Models with Spatial-Temporal Understanding 

**Authors**: Maxim A. Patratskiy, Alexey K. Kovalev, Aleksandr I. Panov  

**Link**: [PDF](https://arxiv.org/pdf/2508.09032)  

**Abstract**: Vision-Language-Action models have demonstrated remarkable capabilities in predicting agent movements within virtual environments and real-world scenarios based on visual observations and textual instructions. Although recent research has focused on enhancing spatial and temporal understanding independently, this paper presents a novel approach that integrates both aspects through visual prompting. We introduce a method that projects visual traces of key points from observations onto depth maps, enabling models to capture both spatial and temporal information simultaneously. The experiments in SimplerEnv show that the mean number of tasks successfully solved increased for 4% compared to SpatialVLA and 19% compared to TraceVLA. Furthermore, we show that this enhancement can be achieved with minimal training data, making it particularly valuable for real-world applications where data collection is challenging. The project page is available at this https URL. 

---
# E3-Rewrite: Learning to Rewrite SQL for Executability, Equivalence,and Efficiency 

**Authors**: Dongjie Xu, Yue Cui, Weijie Shi, Qingzhi Ma, Hanghui Guo, Jiaming Li, Yao Zhao, Ruiyuan Zhang, Shimin Di, Jia Zhu, Kai Zheng, Jiajie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.09023)  

**Abstract**: SQL query rewriting aims to reformulate a query into a more efficient form while preserving equivalence. Most existing methods rely on predefined rewrite rules. However, such rule-based approaches face fundamental limitations: (1) fixed rule sets generalize poorly to novel query patterns and struggle with complex queries; (2) a wide range of effective rewriting strategies cannot be fully captured by declarative rules. To overcome these issues, we propose using large language models (LLMs) to generate rewrites. LLMs can capture complex strategies, such as evaluation reordering and CTE rewriting. Despite this potential, directly applying LLMs often results in suboptimal or non-equivalent rewrites due to a lack of execution awareness and semantic grounding. To address these challenges, We present E3-Rewrite, an LLM-based SQL rewriting framework that produces executable, equivalent, and efficient queries. It integrates two core components: a context construction module and a reinforcement learning framework. First, the context module leverages execution plans and retrieved demonstrations to build bottleneck-aware prompts that guide inference-time rewriting. Second, we design a reward function targeting executability, equivalence, and efficiency, evaluated via syntax checks, equivalence verification, and cost estimation. Third, to ensure stable multi-objective learning, we adopt a staged curriculum that first emphasizes executability and equivalence, then gradually incorporates efficiency. Extensive experiments show that E3-Rewrite achieves up to a 25.6\% reduction in query execution time compared to state-of-the-art methods across multiple SQL benchmarks. Moreover, it delivers up to 24.4\% more successful rewrites, expanding coverage to complex queries that previous systems failed to handle. 

---
# When Deepfakes Look Real: Detecting AI-Generated Faces with Unlabeled Data due to Annotation Challenges 

**Authors**: Zhiqiang Yang, Renshuai Tao, Xiaolong Zheng, Guodong Yang, Chunjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.09022)  

**Abstract**: Existing deepfake detection methods heavily depend on labeled training data. However, as AI-generated content becomes increasingly realistic, even \textbf{human annotators struggle to distinguish} between deepfakes and authentic images. This makes the labeling process both time-consuming and less reliable. Specifically, there is a growing demand for approaches that can effectively utilize large-scale unlabeled data from online social networks. Unlike typical unsupervised learning tasks, where categories are distinct, AI-generated faces closely mimic real image distributions and share strong similarities, causing performance drop in conventional strategies. In this paper, we introduce the Dual-Path Guidance Network (DPGNet), to tackle two key challenges: (1) bridging the domain gap between faces from different generation models, and (2) utilizing unlabeled image samples. The method features two core modules: text-guided cross-domain alignment, which uses learnable prompts to unify visual and textual embeddings into a domain-invariant feature space, and curriculum-driven pseudo label generation, which dynamically exploit more informative unlabeled samples. To prevent catastrophic forgetting, we also facilitate bridging between domains via cross-domain knowledge distillation. Extensive experiments on \textbf{11 popular datasets}, show that DPGNet outperforms SoTA approaches by \textbf{6.3\%}, highlighting its effectiveness in leveraging unlabeled data to address the annotation challenges posed by the increasing realism of deepfakes. 

---
# Attacks and Defenses Against LLM Fingerprinting 

**Authors**: Kevin Kurian, Ethan Holland, Sean Oesch  

**Link**: [PDF](https://arxiv.org/pdf/2508.09021)  

**Abstract**: As large language models are increasingly deployed in sensitive environments, fingerprinting attacks pose significant privacy and security risks. We present a study of LLM fingerprinting from both offensive and defensive perspectives. Our attack methodology uses reinforcement learning to automatically optimize query selection, achieving better fingerprinting accuracy with only 3 queries compared to randomly selecting 3 queries from the same pool. Our defensive approach employs semantic-preserving output filtering through a secondary LLM to obfuscate model identity while maintaining semantic integrity. The defensive method reduces fingerprinting accuracy across tested models while preserving output quality. These contributions show the potential to improve fingerprinting tools capabilities while providing practical mitigation strategies against fingerprinting attacks. 

---
# LyS at SemEval 2025 Task 8: Zero-Shot Code Generation for Tabular QA 

**Authors**: Adrián Gude, Roi Santos-Ríos, Francisco Prado-Valiño, Ana Ezquerro, Jesús Vilares  

**Link**: [PDF](https://arxiv.org/pdf/2508.09012)  

**Abstract**: This paper describes our participation in SemEval 2025 Task 8, focused on Tabular Question Answering. We developed a zero-shot pipeline that leverages an Large Language Model to generate functional code capable of extracting the relevant information from tabular data based on an input question. Our approach consists of a modular pipeline where the main code generator module is supported by additional components that identify the most relevant columns and analyze their data types to improve extraction accuracy. In the event that the generated code fails, an iterative refinement process is triggered, incorporating the error feedback into a new generation prompt to enhance robustness. Our results show that zero-shot code generation is a valid approach for Tabular QA, achieving rank 33 of 53 in the test phase despite the lack of task-specific fine-tuning. 

---
# Retrospective Sparse Attention for Efficient Long-Context Generation 

**Authors**: Seonghwan Choi, Beomseok Kang, Dongwon Jo, Jae-Joon Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.09001)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in long-context tasks such as reasoning, code generation, and multi-turn dialogue. However, inference over extended contexts is bottlenecked by the Key-Value (KV) cache, whose memory footprint grows linearly with sequence length and dominates latency at each decoding step. While recent KV cache compression methods identify and load important tokens, they focus predominantly on input contexts and fail to address the cumulative attention errors that arise during long decoding. In this paper, we introduce RetroAttention, a novel KV cache update technique that retrospectively revises past attention outputs using newly arrived KV entries from subsequent decoding steps. By maintaining a lightweight output cache, RetroAttention enables past queries to efficiently access more relevant context, while incurring minimal latency overhead. This breaks the fixed-attention-output paradigm and allows continual correction of prior approximations. Extensive experiments on long-generation benchmarks show that RetroAttention consistently outperforms state-of-the-art (SOTA) KV compression methods, increasing effective KV exposure by up to 1.6$\times$ and accuracy by up to 21.9\%. 

---
# Rational Inverse Reasoning 

**Authors**: Ben Zandonati, Tomás Lozano-Pérez, Leslie Pack Kaelbling  

**Link**: [PDF](https://arxiv.org/pdf/2508.08983)  

**Abstract**: Humans can observe a single, imperfect demonstration and immediately generalize to very different problem settings. Robots, in contrast, often require hundreds of examples and still struggle to generalize beyond the training conditions. We argue that this limitation arises from the inability to recover the latent explanations that underpin intelligent behavior, and that these explanations can take the form of structured programs consisting of high-level goals, sub-task decomposition, and execution constraints. In this work, we introduce Rational Inverse Reasoning (RIR), a framework for inferring these latent programs through a hierarchical generative model of behavior. RIR frames few-shot imitation as Bayesian program induction: a vision-language model iteratively proposes structured symbolic task hypotheses, while a planner-in-the-loop inference scheme scores each by the likelihood of the observed demonstration under that hypothesis. This loop yields a posterior over concise, executable programs. We evaluate RIR on a suite of continuous manipulation tasks designed to test one-shot and few-shot generalization across variations in object pose, count, geometry, and layout. With as little as one demonstration, RIR infers the intended task structure and generalizes to novel settings, outperforming state-of-the-art vision-language model baselines. 

---
# Unsupervised Skill Discovery as Exploration for Learning Agile Locomotion 

**Authors**: Seungeun Rho, Kartik Garg, Morgan Byrd, Sehoon Ha  

**Link**: [PDF](https://arxiv.org/pdf/2508.08982)  

**Abstract**: Exploration is crucial for enabling legged robots to learn agile locomotion behaviors that can overcome diverse obstacles. However, such exploration is inherently challenging, and we often rely on extensive reward engineering, expert demonstrations, or curriculum learning - all of which limit generalizability. In this work, we propose Skill Discovery as Exploration (SDAX), a novel learning framework that significantly reduces human engineering effort. SDAX leverages unsupervised skill discovery to autonomously acquire a diverse repertoire of skills for overcoming obstacles. To dynamically regulate the level of exploration during training, SDAX employs a bi-level optimization process that autonomously adjusts the degree of exploration. We demonstrate that SDAX enables quadrupedal robots to acquire highly agile behaviors including crawling, climbing, leaping, and executing complex maneuvers such as jumping off vertical walls. Finally, we deploy the learned policy on real hardware, validating its successful transfer to the real world. 

---
# Urban-STA4CLC: Urban Theory-Informed Spatio-Temporal Attention Model for Predicting Post-Disaster Commercial Land Use Change 

**Authors**: Ziyi Guo, Yan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08976)  

**Abstract**: Natural disasters such as hurricanes and wildfires increasingly introduce unusual disturbance on economic activities, which are especially likely to reshape commercial land use pattern given their sensitive to customer visitation. However, current modeling approaches are limited in capturing such complex interplay between human activities and commercial land use change under and following disturbances. Such interactions have been more effectively captured in current resilient urban planning theories. This study designs and calibrates a Urban Theory-Informed Spatio-Temporal Attention Model for Predicting Post-Disaster Commercial Land Use Change (Urban-STA4CLC) to predict both the yearly decline and expansion of commercial land use at census block level under cumulative impact of disasters on human activities over two years. Guided by urban theories, Urban-STA4CLC integrates both spatial and temporal attention mechanisms with three theory-informed modules. Resilience theory guides a disaster-aware temporal attention module that captures visitation dynamics. Spatial economic theory informs a multi-relational spatial attention module for inter-block representation. Diffusion theory contributes a regularization term that constrains land use transitions. The model performs significantly better than non-theoretical baselines in predicting commercial land use change under the scenario of recurrent hurricanes, with around 19% improvement in F1 score (0.8763). The effectiveness of the theory-guided modules was further validated through ablation studies. The research demonstrates that embedding urban theory into commercial land use modeling models may substantially enhance the capacity to capture its gains and losses. These advances in commercial land use modeling contribute to land use research that accounts for cumulative impacts of recurrent disasters and shifts in economic activity patterns. 

---
# Revealing the Role of Audio Channels in ASR Performance Degradation 

**Authors**: Kuan-Tang Huang, Li-Wei Chen, Hung-Shin Lee, Berlin Chen, Hsin-Min Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08967)  

**Abstract**: Pre-trained automatic speech recognition (ASR) models have demonstrated strong performance on a variety of tasks. However, their performance can degrade substantially when the input audio comes from different recording channels. While previous studies have demonstrated this phenomenon, it is often attributed to the mismatch between training and testing corpora. This study argues that variations in speech characteristics caused by different recording channels can fundamentally harm ASR performance. To address this limitation, we propose a normalization technique designed to mitigate the impact of channel variation by aligning internal feature representations in the ASR model with those derived from a clean reference channel. This approach significantly improves ASR performance on previously unseen channels and languages, highlighting its ability to generalize across channel and language differences. 

---
# QAMRO: Quality-aware Adaptive Margin Ranking Optimization for Human-aligned Assessment of Audio Generation Systems 

**Authors**: Chien-Chun Wang, Kuan-Tang Huang, Cheng-Yeh Yang, Hung-Shin Lee, Hsin-Min Wang, Berlin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08957)  

**Abstract**: Evaluating audio generation systems, including text-to-music (TTM), text-to-speech (TTS), and text-to-audio (TTA), remains challenging due to the subjective and multi-dimensional nature of human perception. Existing methods treat mean opinion score (MOS) prediction as a regression problem, but standard regression losses overlook the relativity of perceptual judgments. To address this limitation, we introduce QAMRO, a novel Quality-aware Adaptive Margin Ranking Optimization framework that seamlessly integrates regression objectives from different perspectives, aiming to highlight perceptual differences and prioritize accurate ratings. Our framework leverages pre-trained audio-text models such as CLAP and Audiobox-Aesthetics, and is trained exclusively on the official AudioMOS Challenge 2025 dataset. It demonstrates superior alignment with human evaluations across all dimensions, significantly outperforming robust baseline models. 

---
# Generalising Traffic Forecasting to Regions without Traffic Observations 

**Authors**: Xinyu Su, Majid Sarvi, Feng Liu, Egemen Tanin, Jianzhong Qi  

**Link**: [PDF](https://arxiv.org/pdf/2508.08947)  

**Abstract**: Traffic forecasting is essential for intelligent transportation systems. Accurate forecasting relies on continuous observations collected by traffic sensors. However, due to high deployment and maintenance costs, not all regions are equipped with such sensors. This paper aims to forecast for regions without traffic sensors, where the lack of historical traffic observations challenges the generalisability of existing models. We propose a model named GenCast, the core idea of which is to exploit external knowledge to compensate for the missing observations and to enhance generalisation. We integrate physics-informed neural networks into GenCast, enabling physical principles to regularise the learning process. We introduce an external signal learning module to explore correlations between traffic states and external signals such as weather conditions, further improving model generalisability. Additionally, we design a spatial grouping module to filter localised features that hinder model generalisability. Extensive experiments show that GenCast consistently reduces forecasting errors on multiple real-world datasets. 

---
# Train Long, Think Short: Curriculum Learning for Efficient Reasoning 

**Authors**: Hasan Abed Al Kader Hammoud, Kumail Alhamoud, Abed Hammoud, Elie Bou-Zeid, Marzyeh Ghassemi, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2508.08940)  

**Abstract**: Recent work on enhancing the reasoning abilities of large language models (LLMs) has introduced explicit length control as a means of constraining computational cost while preserving accuracy. However, existing approaches rely on fixed-length training budgets, which do not take advantage of the natural progression from exploration to compression during learning. In this work, we propose a curriculum learning strategy for length-controlled reasoning using Group Relative Policy Optimization (GRPO). Our method starts with generous token budgets and gradually tightens them over training, encouraging models to first discover effective solution strategies and then distill them into more concise reasoning traces. We augment GRPO with a reward function that balances three signals: task correctness (via verifier feedback), length efficiency, and formatting adherence (via structural tags). Experiments on GSM8K, MATH500, SVAMP, College Math, and GSM+ demonstrate that curriculum-based training consistently outperforms fixed-budget baselines at the same final budget, achieving higher accuracy and significantly improved token efficiency. We further ablate the impact of reward weighting and decay schedule design, showing that progressive constraint serves as a powerful inductive bias for training efficient reasoning models. Our code and checkpoints are released at: this https URL. 

---
# EGGCodec: A Robust Neural Encodec Framework for EGG Reconstruction and F0 Extraction 

**Authors**: Rui Feng, Yuang Chen, Yu Hu, Jun Du, Jiahong Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.08924)  

**Abstract**: This letter introduces EGGCodec, a robust neural Encodec framework engineered for electroglottography (EGG) signal reconstruction and F0 extraction. We propose a multi-scale frequency-domain loss function to capture the nuanced relationship between original and reconstructed EGG signals, complemented by a time-domain correlation loss to improve generalization and accuracy. Unlike conventional Encodec models that extract F0 directly from features, EGGCodec leverages reconstructed EGG signals, which more closely correspond to F0. By removing the conventional GAN discriminator, we streamline EGGCodec's training process without compromising efficiency, incurring only negligible performance degradation. Trained on a widely used EGG-inclusive dataset, extensive evaluations demonstrate that EGGCodec outperforms state-of-the-art F0 extraction schemes, reducing mean absolute error (MAE) from 14.14 Hz to 13.69 Hz, and improving voicing decision error (VDE) by 38.2\%. Moreover, extensive ablation experiments validate the contribution of each component of EGGCodec. 

---
# Shape Completion and Real-Time Visualization in Robotic Ultrasound Spine Acquisitions 

**Authors**: Miruna-Alexandra Gafencu, Reem Shaban, Yordanka Velikova, Mohammad Farid Azampour, Nassir Navab  

**Link**: [PDF](https://arxiv.org/pdf/2508.08923)  

**Abstract**: Ultrasound (US) imaging is increasingly used in spinal procedures due to its real-time, radiation-free capabilities; however, its effectiveness is hindered by shadowing artifacts that obscure deeper tissue structures. Traditional approaches, such as CT-to-US registration, incorporate anatomical information from preoperative CT scans to guide interventions, but they are limited by complex registration requirements, differences in spine curvature, and the need for recent CT imaging. Recent shape completion methods can offer an alternative by reconstructing spinal structures in US data, while being pretrained on large set of publicly available CT scans. However, these approaches are typically offline and have limited reproducibility. In this work, we introduce a novel integrated system that combines robotic ultrasound with real-time shape completion to enhance spinal visualization. Our robotic platform autonomously acquires US sweeps of the lumbar spine, extracts vertebral surfaces from ultrasound, and reconstructs the complete anatomy using a deep learning-based shape completion network. This framework provides interactive, real-time visualization with the capability to autonomously repeat scans and can enable navigation to target locations. This can contribute to better consistency, reproducibility, and understanding of the underlying anatomy. We validate our approach through quantitative experiments assessing shape completion accuracy and evaluations of multiple spine acquisition protocols on a phantom setup. Additionally, we present qualitative results of the visualization on a volunteer scan. 

---
# Munsit at NADI 2025 Shared Task 2: Pushing the Boundaries of Multidialectal Arabic ASR with Weakly Supervised Pretraining and Continual Supervised Fine-tuning 

**Authors**: Mahmoud Salhab, Shameed Sait, Mohammad Abusheikh, Hasan Abusheikh  

**Link**: [PDF](https://arxiv.org/pdf/2508.08912)  

**Abstract**: Automatic speech recognition (ASR) plays a vital role in enabling natural human-machine interaction across applications such as virtual assistants, industrial automation, customer support, and real-time transcription. However, developing accurate ASR systems for low-resource languages like Arabic remains a significant challenge due to limited labeled data and the linguistic complexity introduced by diverse dialects. In this work, we present a scalable training pipeline that combines weakly supervised learning with supervised fine-tuning to develop a robust Arabic ASR model. In the first stage, we pretrain the model on 15,000 hours of weakly labeled speech covering both Modern Standard Arabic (MSA) and various Dialectal Arabic (DA) variants. In the subsequent stage, we perform continual supervised fine-tuning using a mixture of filtered weakly labeled data and a small, high-quality annotated dataset. Our approach achieves state-of-the-art results, ranking first in the multi-dialectal Arabic ASR challenge. These findings highlight the effectiveness of weak supervision paired with fine-tuning in overcoming data scarcity and delivering high-quality ASR for low-resource, dialect-rich languages. 

---
# ASPD: Unlocking Adaptive Serial-Parallel Decoding by Exploring Intrinsic Parallelism in LLMs 

**Authors**: Keyu Chen, Zhifeng Shen, Daohai Yu, Haoqian Wu, Wei Wen, Jianfeng He, Ruizhi Qiao, Xing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.08895)  

**Abstract**: The increasing scale and complexity of large language models (LLMs) pose significant inference latency challenges, primarily due to their autoregressive decoding paradigm characterized by the sequential nature of next-token prediction. By re-examining the outputs of autoregressive models, we observed that some segments exhibit parallelizable structures, which we term intrinsic parallelism. Decoding each parallelizable branch simultaneously (i.e. parallel decoding) can significantly improve the overall inference speed of LLMs. In this paper, we propose an Adaptive Serial-Parallel Decoding (ASPD), which addresses two core challenges: automated construction of parallelizable data and efficient parallel decoding mechanism. More specifically, we introduce a non-invasive pipeline that automatically extracts and validates parallelizable structures from the responses of autoregressive models. To empower efficient adaptive serial-parallel decoding, we implement a Hybrid Decoding Engine which enables seamless transitions between serial and parallel decoding modes while maintaining a reusable KV cache, maximizing computational efficiency. Extensive evaluations across General Tasks, Retrieval-Augmented Generation, Mathematical Reasoning, demonstrate that ASPD achieves unprecedented performance in both effectiveness and efficiency. Notably, on Vicuna Bench, our method achieves up to 3.19x speedup (1.85x on average) while maintaining response quality within 1% difference compared to autoregressive models, realizing significant acceleration without compromising generation quality. Our framework sets a groundbreaking benchmark for efficient LLM parallel inference, paving the way for its deployment in latency-sensitive applications such as AI-powered customer service bots and answer retrieval engines. 

---
# Position: Causal Machine Learning Requires Rigorous Synthetic Experiments for Broader Adoption 

**Authors**: Audrey Poinsot, Panayiotis Panayiotou, Alessandro Leite, Nicolas Chesneau, Özgür Şimşek, Marc Schoenauer  

**Link**: [PDF](https://arxiv.org/pdf/2508.08883)  

**Abstract**: Causal machine learning has the potential to revolutionize decision-making by combining the predictive power of machine learning algorithms with the theory of causal inference. However, these methods remain underutilized by the broader machine learning community, in part because current empirical evaluations do not permit assessment of their reliability and robustness, undermining their practical utility. Specifically, one of the principal criticisms made by the community is the extensive use of synthetic experiments. We argue, on the contrary, that synthetic experiments are essential and necessary to precisely assess and understand the capabilities of causal machine learning methods. To substantiate our position, we critically review the current evaluation practices, spotlight their shortcomings, and propose a set of principles for conducting rigorous empirical analyses with synthetic data. Adopting the proposed principles will enable comprehensive evaluations that build trust in causal machine learning methods, driving their broader adoption and impactful real-world use. 

---
# Entangled in Representations: Mechanistic Investigation of Cultural Biases in Large Language Models 

**Authors**: Haeun Yu, Seogyeong Jeong, Siddhesh Pawar, Jisu Shin, Jiho Jin, Junho Myung, Alice Oh, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.08879)  

**Abstract**: The growing deployment of large language models (LLMs) across diverse cultural contexts necessitates a better understanding of how the overgeneralization of less documented cultures within LLMs' representations impacts their cultural understanding. Prior work only performs extrinsic evaluation of LLMs' cultural competence, without accounting for how LLMs' internal mechanisms lead to cultural (mis)representation. To bridge this gap, we propose Culturescope, the first mechanistic interpretability-based method that probes the internal representations of LLMs to elicit the underlying cultural knowledge space. CultureScope utilizes a patching method to extract the cultural knowledge. We introduce a cultural flattening score as a measure of the intrinsic cultural biases. Additionally, we study how LLMs internalize Western-dominance bias and cultural flattening, which allows us to trace how cultural biases emerge within LLMs. Our experimental results reveal that LLMs encode Western-dominance bias and cultural flattening in their cultural knowledge space. We find that low-resource cultures are less susceptible to cultural biases, likely due to their limited training resources. Our work provides a foundation for future research on mitigating cultural biases and enhancing LLMs' cultural understanding. Our codes and data used for experiments are publicly available. 

---
# Oblivionis: A Lightweight Learning and Unlearning Framework for Federated Large Language Models 

**Authors**: Fuyao Zhang, Xinyu Yan, Tiantong Wu, Wenjie Li, Tianxiang Chen, Yang Cao, Ran Yan, Longtao Huang, Wei Yang Bryan Lim, Qiang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08875)  

**Abstract**: Large Language Models (LLMs) increasingly leverage Federated Learning (FL) to utilize private, task-specific datasets for fine-tuning while preserving data privacy. However, while federated LLM frameworks effectively enable collaborative training without raw data sharing, they critically lack built-in mechanisms for regulatory compliance like GDPR's right to be forgotten. Integrating private data heightens concerns over data quality and long-term governance, yet existing distributed training frameworks offer no principled way to selectively remove specific client contributions post-training. Due to distributed data silos, stringent privacy constraints, and the intricacies of interdependent model aggregation, federated LLM unlearning is significantly more complex than centralized LLM unlearning. To address this gap, we introduce Oblivionis, a lightweight learning and unlearning framework that enables clients to selectively remove specific private data during federated LLM training, enhancing trustworthiness and regulatory compliance. By unifying FL and unlearning as a dual optimization objective, we incorporate 6 FL and 5 unlearning algorithms for comprehensive evaluation and comparative analysis, establishing a robust pipeline for federated LLM unlearning. Extensive experiments demonstrate that Oblivionis outperforms local training, achieving a robust balance between forgetting efficacy and model utility, with cross-algorithm comparisons providing clear directions for future LLM development. 

---
# BiasGym: Fantastic Biases and How to Find (and Remove) Them 

**Authors**: Sekh Mainul Islam, Nadav Borenstein, Siddhesh Milind Pawar, Haeun Yu, Arnav Arora, Isabelle Augenstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.08855)  

**Abstract**: Understanding biases and stereotypes encoded in the weights of Large Language Models (LLMs) is crucial for developing effective mitigation strategies. Biased behaviour is often subtle and non-trivial to isolate, even when deliberately elicited, making systematic analysis and debiasing particularly challenging. To address this, we introduce BiasGym, a simple, cost-effective, and generalizable framework for reliably injecting, analyzing, and mitigating conceptual associations within LLMs. BiasGym consists of two components: BiasInject, which injects specific biases into the model via token-based fine-tuning while keeping the model frozen, and BiasScope, which leverages these injected signals to identify and steer the components responsible for biased behavior. Our method enables consistent bias elicitation for mechanistic analysis, supports targeted debiasing without degrading performance on downstream tasks, and generalizes to biases unseen during training. We demonstrate the effectiveness of BiasGym in reducing real-world stereotypes (e.g., people from a country being `reckless drivers') and in probing fictional associations (e.g., people from a country having `blue skin'), showing its utility for both safety interventions and interpretability research. 

---
# Steering Towards Fairness: Mitigating Political Bias in LLMs 

**Authors**: Afrozah Nadeem, Mark Dras, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2508.08846)  

**Abstract**: Recent advancements in large language models (LLMs) have enabled their widespread use across diverse real-world applications. However, concerns remain about their tendency to encode and reproduce ideological biases, particularly along political and economic dimensions. In this paper, we propose a framework for probing and mitigating such biases in decoder-based LLMs through analysis of internal model representations. Grounded in the Political Compass Test (PCT), our method uses contrastive pairs to extract and compare hidden layer activations from models like Mistral and DeepSeek. We introduce a comprehensive activation extraction pipeline capable of layer-wise analysis across multiple ideological axes, revealing meaningful disparities linked to political framing. Our results show that decoder LLMs systematically encode representational bias across layers, which can be leveraged for effective steering vector-based mitigation. This work provides new insights into how political bias is encoded in LLMs and offers a principled approach to debiasing beyond surface-level output interventions. 

---
# The Roots of International Perceptions: Simulating US Attitude Changes Towards China with LLM Agents 

**Authors**: Nicholas Sukiennik, Yichuan Xu, Yuqing Kan, Jinghua Piao, Yuwei Yan, Chen Gao, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.08837)  

**Abstract**: The rise of LLMs poses new possibilities in modeling opinion evolution, a long-standing task in simulation, by leveraging advanced reasoning abilities to recreate complex, large-scale human cognitive trends. While most prior works focus on opinion evolution surrounding specific isolated events or the views within a country, ours is the first to model the large-scale attitude evolution of a population representing an entire country towards another -- US citizens' perspectives towards China. To tackle the challenges of this broad scenario, we propose a framework that integrates media data collection, user profile creation, and cognitive architecture for opinion updates to successfully reproduce the real trend of US attitudes towards China over a 20-year period from 2005 to today. We also leverage LLMs' capabilities to introduce debiased media exposure, extracting neutral events from typically subjective news contents, to uncover the roots of polarized opinion formation, as well as a devils advocate agent to help explain the rare reversal from negative to positive attitudes towards China, corresponding with changes in the way Americans obtain information about the country. The simulation results, beyond validating our framework architecture, also reveal the impact of biased framing and selection bias in shaping attitudes. Overall, our work contributes to a new paradigm for LLM-based modeling of cognitive behaviors in a large-scale, long-term, cross-border social context, providing insights into the formation of international biases and offering valuable implications for media consumers to better understand the factors shaping their perspectives, and ultimately contributing to the larger social need for bias reduction and cross-cultural tolerance. 

---
# EditMF: Drawing an Invisible Fingerprint for Your Large Language Models 

**Authors**: Jiaxuan Wu, Yinghan Zhou, Wanli Peng, Yiming Xue, Juan Wen, Ping Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2508.08836)  

**Abstract**: Training large language models (LLMs) is resource-intensive and expensive, making protecting intellectual property (IP) for LLMs crucial. Recently, embedding fingerprints into LLMs has emerged as a prevalent method for establishing model ownership. However, existing back-door-based methods suffer from limited stealth and efficiency. To simultaneously address these issues, we propose EditMF, a training-free fingerprinting paradigm that achieves highly imperceptible fingerprint embedding with minimal computational overhead. Ownership bits are mapped to compact, semantically coherent triples drawn from an encrypted artificial knowledge base (e.g., virtual author-novel-protagonist facts). Causal tracing localizes the minimal set of layers influencing each triple, and a zero-space update injects the fingerprint without perturbing unrelated knowledge. Verification requires only a single black-box query and succeeds when the model returns the exact pre-embedded protagonist. Empirical results on LLaMA and Qwen families show that EditMF combines high imperceptibility with negligible model's performance loss, while delivering robustness far beyond LoRA-based fingerprinting and approaching that of SFT embeddings. Extensive experiments demonstrate that EditMF is an effective and low-overhead solution for secure LLM ownership verification. 

---
# An Investigation of Robustness of LLMs in Mathematical Reasoning: Benchmarking with Mathematically-Equivalent Transformation of Advanced Mathematical Problems 

**Authors**: Yuren Hao, Xiang Wan, Chengxiang Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2508.08833)  

**Abstract**: In this paper, we introduce a systematic framework beyond conventional method to assess LLMs' mathematical-reasoning robustness by stress-testing them on advanced math problems that are mathematically equivalent but with linguistic and parametric variation. These transformations allow us to measure the sensitivity of LLMs to non-mathematical perturbations, thereby enabling a more accurate evaluation of their mathematical reasoning capabilities. Using this new evaluation methodology, we created PutnamGAP, a new benchmark dataset with multiple mathematically-equivalent variations of competition-level math problems. With the new dataset, we evaluate multiple families of representative LLMs and examine their robustness. Across 18 commercial and open-source models we observe sharp performance degradation on the variants. OpenAI's flagship reasoning model, O3, scores 49 % on the originals but drops by 4 percentage points on surface variants, and by 10.5 percentage points on core-step-based variants, while smaller models fare far worse. Overall, the results show that the proposed new evaluation methodology is effective for deepening our understanding of the robustness of LLMs and generating new insights for further improving their mathematical reasoning capabilities. 

---
# Geometry-Aware Global Feature Aggregation for Real-Time Indirect Illumination 

**Authors**: Meng Gai, Guoping Wang, Sheng Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.08826)  

**Abstract**: Real-time rendering with global illumination is crucial to afford the user realistic experience in virtual environments. We present a learning-based estimator to predict diffuse indirect illumination in screen space, which then is combined with direct illumination to synthesize globally-illuminated high dynamic range (HDR) results. Our approach tackles the challenges of capturing long-range/long-distance indirect illumination when employing neural networks and is generalized to handle complex lighting and scenarios.
From the neural network thinking of the solver to the rendering equation, we present a novel network architecture to predict indirect illumination. Our network is equipped with a modified attention mechanism that aggregates global information guided by spacial geometry features, as well as a monochromatic design that encodes each color channel individually.
We conducted extensive evaluations, and the experimental results demonstrate our superiority over previous learning-based techniques. Our approach excels at handling complex lighting such as varying-colored lighting and environment lighting. It can successfully capture distant indirect illumination and simulates the interreflections between textured surfaces well (i.e., color bleeding effects); it can also effectively handle new scenes that are not present in the training dataset. 

---
# Wavelet Mixture of Experts for Time Series Forecasting 

**Authors**: Zheng Zhou, Yu-Jie Xiong, Jia-Chen Zhang, Chun-Ming Xia, Xi-Jiong Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.08825)  

**Abstract**: The field of time series forecasting is rapidly advancing, with recent large-scale Transformers and lightweight Multilayer Perceptron (MLP) models showing strong predictive performance. However, conventional Transformer models are often hindered by their large number of parameters and their limited ability to capture non-stationary features in data through smoothing. Similarly, MLP models struggle to manage multi-channel dependencies effectively. To address these limitations, we propose a novel, lightweight time series prediction model, WaveTS-B. This model combines wavelet transforms with MLP to capture both periodic and non-stationary characteristics of data in the wavelet domain. Building on this foundation, we propose a channel clustering strategy that incorporates a Mixture of Experts (MoE) framework, utilizing a gating mechanism and expert network to handle multi-channel dependencies efficiently. We propose WaveTS-M, an advanced model tailored for multi-channel time series prediction. Empirical evaluation across eight real-world time series datasets demonstrates that our WaveTS series models achieve state-of-the-art (SOTA) performance with significantly fewer parameters. Notably, WaveTS-M shows substantial improvements on multi-channel datasets, highlighting its effectiveness. 

---
# OISMA: On-the-fly In-memory Stochastic Multiplication Architecture for Matrix-Multiplication Workloads 

**Authors**: Shady Agwa, Yihan Pan, Georgios Papandroulidakis, Themis Prodromakis  

**Link**: [PDF](https://arxiv.org/pdf/2508.08822)  

**Abstract**: Artificial Intelligence models are currently driven by a significant up-scaling of their complexity, with massive matrix multiplication workloads representing the major computational bottleneck. In-memory computing architectures are proposed to avoid the Von Neumann bottleneck. However, both digital/binary-based and analogue in-memory computing architectures suffer from various limitations, which significantly degrade the performance and energy efficiency gains. This work proposes OISMA, a novel in-memory computing architecture that utilizes the computational simplicity of a quasi-stochastic computing domain (Bent-Pyramid system), while keeping the same efficiency, scalability, and productivity of digital memories. OISMA converts normal memory read operations into in-situ stochastic multiplication operations with a negligible cost. An accumulation periphery then accumulates the output multiplication bitstreams, achieving the matrix multiplication functionality. Extensive matrix multiplication benchmarking was conducted to analyze the accuracy of the Bent-Pyramid system, using matrix dimensions ranging from 4x4 to 512x512. The accuracy results show a significant decrease in the average relative Frobenius error, from 9.42% (for 4x4) to 1.81% (for 512x512), compared to 64-bit double precision floating-point format. A 1T1R OISMA array of 4 KB capacity was implemented using a commercial 180nm technology node and in-house RRAM technology. At 50 MHz, OISMA achieves 0.891 TOPS/W and 3.98 GOPS/mm2 for energy and area efficiency, respectively, occupying an effective computing area of 0.804241 mm2. Scaling OISMA from 180nm to 22nm technology shows a significant improvement of two orders of magnitude in energy efficiency and one order of magnitude in area efficiency, compared to dense matrix multiplication in-memory computing architectures. 

---
# TempOpt -- Unsupervised Alarm Relation Learning for Telecommunication Networks 

**Authors**: Sathiyanaryanan Sampath, Pratyush Uppuluri, Thirumaran Ekambaram  

**Link**: [PDF](https://arxiv.org/pdf/2508.08814)  

**Abstract**: In a telecommunications network, fault alarms generated by network nodes are monitored in a Network Operations Centre (NOC) to ensure network availability and continuous network operations. The monitoring process comprises of tasks such as active alarms analysis, root alarm identification, and resolution of the underlying problem. Each network node potentially can generate alarms of different types, while nodes can be from multiple vendors, a network can have hundreds of nodes thus resulting in an enormous volume of alarms at any time. Since network nodes are inter-connected, a single fault in the network would trigger multiple sequences of alarms across a variety of nodes and from a monitoring point of view, it is a challenging task for a NOC engineer to be aware of relations between the various alarms, when trying to identify, for example, a root alarm on which an action needs to be taken. To effectively identify root alarms, it is essential to learn relation among the alarms for accurate and faster resolution. In this work we propose a novel unsupervised alarm relation learning technique Temporal Optimization (TempOpt) that is practical and overcomes the limitations of an existing class of alarm relational learning method-temporal dependency methods. Experiments have been carried on real-world network datasets, that demonstrate the improved quality of alarm relations learned by TempOpt as compared to temporal dependency method. 

---
# Not in My Backyard! Temporal Voting Over Public Chores 

**Authors**: Edith Elkind, Tzeh Yuan Neoh, Nicholas Teh  

**Link**: [PDF](https://arxiv.org/pdf/2508.08810)  

**Abstract**: We study a temporal voting model where voters have dynamic preferences over a set of public chores -- projects that benefit society, but impose individual costs on those affected by their implementation. We investigate the computational complexity of optimizing utilitarian and egalitarian welfare. Our results show that while optimizing the former is computationally straightforward, minimizing the latter is computationally intractable, even in very restricted cases. Nevertheless, we identify several settings where this problem can be solved efficiently, either exactly or by an approximation algorithm. We also examine the effects of enforcing temporal fairness and its impact on social welfare, and analyze the competitive ratio of online algorithms. We then explore the strategic behavior of agents, providing insights into potential malfeasance in such decision-making environments. Finally, we discuss a range of fairness measures and their suitability for our setting. 

---
# Opening Musical Creativity? Embedded Ideologies in Generative-AI Music Systems 

**Authors**: Liam Pram, Fabio Morreale  

**Link**: [PDF](https://arxiv.org/pdf/2508.08805)  

**Abstract**: AI systems for music generation are increasingly common and easy to use, granting people without any musical background the ability to create music. Because of this, generative-AI has been marketed and celebrated as a means of democratizing music making. However, inclusivity often functions as marketable rhetoric rather than a genuine guiding principle in these industry settings. In this paper, we look at four generative-AI music making systems available to the public as of mid-2025 (AIVA, Stable Audio, Suno, and Udio) and track how they are rhetoricized by their developers, and received by users. Our aim is to investigate ideologies that are driving the early-stage development and adoption of generative-AI in music making, with a particular focus on democratization. A combination of autoethnography and digital ethnography is used to examine patterns and incongruities in rhetoric when positioned against product functionality. The results are then collated to develop a nuanced, contextual discussion. The shared ideology we map between producers and consumers is individualist, globalist, techno-liberal, and ethically evasive. It is a 'total ideology' which obfuscates individual responsibility, and through which the nature of music and musical practice is transfigured to suit generative outcomes. 

---
# TechOps: Technical Documentation Templates for the AI Act 

**Authors**: Laura Lucaj, Alex Loosley, Hakan Jonsson, Urs Gasser, Patrick van der Smagt  

**Link**: [PDF](https://arxiv.org/pdf/2508.08804)  

**Abstract**: Operationalizing the EU AI Act requires clear technical documentation to ensure AI systems are transparent, traceable, and accountable. Existing documentation templates for AI systems do not fully cover the entire AI lifecycle while meeting the technical documentation requirements of the AI Act.
This paper addresses those shortcomings by introducing open-source templates and examples for documenting data, models, and applications to provide sufficient documentation for certifying compliance with the AI Act. These templates track the system status over the entire AI lifecycle, ensuring traceability, reproducibility, and compliance with the AI Act. They also promote discoverability and collaboration, reduce risks, and align with best practices in AI documentation and governance.
The templates are evaluated and refined based on user feedback to enable insights into their usability and implementability. We then validate the approach on real-world scenarios, providing examples that further guide their implementation: the data template is followed to document a skin tones dataset created to support fairness evaluations of downstream computer vision models and human-centric applications; the model template is followed to document a neural network for segmenting human silhouettes in photos. The application template is tested on a system deployed for construction site safety using real-time video analytics and sensor data. Our results show that TechOps can serve as a practical tool to enable oversight for regulatory compliance and responsible AI development. 

---
# Feedback-Driven Tool-Use Improvements in Large Language Models via Automated Build Environments 

**Authors**: Junjie Ye, Changhao Jiang, Zhengyin Du, Yufei Xu, Xuesong Yao, Zhiheng Xi, Xiaoran Fan, Qi Zhang, Xuanjing Huang, Jiecao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08791)  

**Abstract**: Effective tool use is essential for large language models (LLMs) to interact meaningfully with their environment. However, progress is limited by the lack of efficient reinforcement learning (RL) frameworks specifically designed for tool use, due to challenges in constructing stable training environments and designing verifiable reward mechanisms. To address this, we propose an automated environment construction pipeline, incorporating scenario decomposition, document generation, function integration, complexity scaling, and localized deployment. This enables the creation of high-quality training environments that provide detailed and measurable feedback without relying on external tools. Additionally, we introduce a verifiable reward mechanism that evaluates both the precision of tool use and the completeness of task execution. When combined with trajectory data collected from the constructed environments, this mechanism integrates seamlessly with standard RL algorithms to facilitate feedback-driven model training. Experiments on LLMs of varying scales demonstrate that our approach significantly enhances the models' tool-use performance without degrading their general capabilities, regardless of inference modes or training algorithms. Our analysis suggests that these gains result from improved context understanding and reasoning, driven by updates to the lower-layer MLP parameters in models. 

---
# ReQuestNet: A Foundational Learning model for Channel Estimation 

**Authors**: Kumar Pratik, Pouriya Sadeghi, Gabriele Cesa, Sanaz Barghi, Joseph B. Soriaga, Yuanning Yu, Supratik Bhattacharjee, Arash Behboodi  

**Link**: [PDF](https://arxiv.org/pdf/2508.08790)  

**Abstract**: In this paper, we present a novel neural architecture for channel estimation (CE) in 5G and beyond, the Recurrent Equivariant UERS Estimation Network (ReQuestNet). It incorporates several practical considerations in wireless communication systems, such as ability to handle variable number of resource block (RB), dynamic number of transmit layers, physical resource block groups (PRGs) bundling size (BS), demodulation reference signal (DMRS) patterns with a single unified model, thereby, drastically simplifying the CE pipeline. Besides it addresses several limitations of the legacy linear MMSE solutions, for example, by being independent of other reference signals and particularly by jointly processing MIMO layers and differently precoded channels with unknown precoding at the receiver. ReQuestNet comprises of two sub-units, CoarseNet followed by RefinementNet. CoarseNet performs per PRG, per transmit-receive (Tx-Rx) stream channel estimation, while RefinementNet refines the CoarseNet channel estimate by incorporating correlations across differently precoded PRGs, and correlation across multiple input multiple output (MIMO) channel spatial dimensions (cross-MIMO). Simulation results demonstrate that ReQuestNet significantly outperforms genie minimum mean squared error (MMSE) CE across a wide range of channel conditions, delay-Doppler profiles, achieving up to 10dB gain at high SNRs. Notably, ReQuestNet generalizes effectively to unseen channel profiles, efficiently exploiting inter-PRG and cross-MIMO correlations under dynamic PRG BS and varying transmit layer allocations. 

---
# Evaluating Podcast Recommendations with Profile-Aware LLM-as-a-Judge 

**Authors**: Francesco Fabbri, Gustavo Penha, Edoardo D'Amico, Alice Wang, Marco De Nadai, Jackie Doremus, Paul Gigioli, Andreas Damianou, Oskar Stal, Mounia Lalmas  

**Link**: [PDF](https://arxiv.org/pdf/2508.08777)  

**Abstract**: Evaluating personalized recommendations remains a central challenge, especially in long-form audio domains like podcasts, where traditional offline metrics suffer from exposure bias and online methods such as A/B testing are costly and operationally constrained. In this paper, we propose a novel framework that leverages Large Language Models (LLMs) as offline judges to assess the quality of podcast recommendations in a scalable and interpretable manner. Our two-stage profile-aware approach first constructs natural-language user profiles distilled from 90 days of listening history. These profiles summarize both topical interests and behavioral patterns, serving as compact, interpretable representations of user preferences. Rather than prompting the LLM with raw data, we use these profiles to provide high-level, semantically rich context-enabling the LLM to reason more effectively about alignment between a user's interests and recommended episodes. This reduces input complexity and improves interpretability. The LLM is then prompted to deliver fine-grained pointwise and pairwise judgments based on the profile-episode match. In a controlled study with 47 participants, our profile-aware judge matched human judgments with high fidelity and outperformed or matched a variant using raw listening histories. The framework enables efficient, profile-aware evaluation for iterative testing and model selection in recommender systems. 

---
# Bridging the Gap: A Framework for Real-World Video Deepfake Detection via Social Network Compression Emulation 

**Authors**: Andrea Montibeller, Dasara Shullani, Daniele Baracchi, Alessandro Piva, Giulia Boato  

**Link**: [PDF](https://arxiv.org/pdf/2508.08765)  

**Abstract**: The growing presence of AI-generated videos on social networks poses new challenges for deepfake detection, as detectors trained under controlled conditions often fail to generalize to real-world scenarios. A key factor behind this gap is the aggressive, proprietary compression applied by platforms like YouTube and Facebook, which launder low-level forensic cues. However, replicating these transformations at scale is difficult due to API limitations and data-sharing constraints. For these reasons, we propose a first framework that emulates the video sharing pipelines of social networks by estimating compression and resizing parameters from a small set of uploaded videos. These parameters enable a local emulator capable of reproducing platform-specific artifacts on large datasets without direct API access. Experiments on FaceForensics++ videos shared via social networks demonstrate that our emulated data closely matches the degradation patterns of real uploads. Furthermore, detectors fine-tuned on emulated videos achieve comparable performance to those trained on actual shared media. Our approach offers a scalable and practical solution for bridging the gap between lab-based training and real-world deployment of deepfake detectors, particularly in the underexplored domain of compressed video content. 

---
# DevNous: An LLM-Based Multi-Agent System for Grounding IT Project Management in Unstructured Conversation 

**Authors**: Stavros Doropoulos, Stavros Vologiannidis, Ioannis Magnisalis  

**Link**: [PDF](https://arxiv.org/pdf/2508.08761)  

**Abstract**: The manual translation of unstructured team dialogue into the structured artifacts required for Information Technology (IT) project governance is a critical bottleneck in modern information systems management. We introduce DevNous, a Large Language Model-based (LLM) multi-agent expert system, to automate this unstructured-to-structured translation process. DevNous integrates directly into team chat environments, identifying actionable intents from informal dialogue and managing stateful, multi-turn workflows for core administrative tasks like automated task formalization and progress summary synthesis. To quantitatively evaluate the system, we introduce a new benchmark of 160 realistic, interactive conversational turns. The dataset was manually annotated with a multi-label ground truth and is publicly available. On this benchmark, DevNous achieves an exact match turn accuracy of 81.3\% and a multiset F1-Score of 0.845, providing strong evidence for its viability. The primary contributions of this work are twofold: (1) a validated architectural pattern for developing ambient administrative agents, and (2) the introduction of the first robust empirical baseline and public benchmark dataset for this challenging problem domain. 

---
# Visual Prompting for Robotic Manipulation with Annotation-Guided Pick-and-Place Using ACT 

**Authors**: Muhammad A. Muttaqien, Tomohiro Motoda, Ryo Hanai, Yukiyasu Domae  

**Link**: [PDF](https://arxiv.org/pdf/2508.08748)  

**Abstract**: Robotic pick-and-place tasks in convenience stores pose challenges due to dense object arrangements, occlusions, and variations in object properties such as color, shape, size, and texture. These factors complicate trajectory planning and grasping. This paper introduces a perception-action pipeline leveraging annotation-guided visual prompting, where bounding box annotations identify both pickable objects and placement locations, providing structured spatial guidance. Instead of traditional step-by-step planning, we employ Action Chunking with Transformers (ACT) as an imitation learning algorithm, enabling the robotic arm to predict chunked action sequences from human demonstrations. This facilitates smooth, adaptive, and data-driven pick-and-place operations. We evaluate our system based on success rate and visual analysis of grasping behavior, demonstrating improved grasp accuracy and adaptability in retail environments. 

---
# SciRerankBench: Benchmarking Rerankers Towards Scientific Retrieval-Augmented Generated LLMs 

**Authors**: Haotian Chen, Qingqing Long, Meng Xiao, Xiao Luo, Wei Ju, Chengrui Wang, Xuezhi Wang, Yuanchun Zhou, Hengshu Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08742)  

**Abstract**: Scientific literature question answering is a pivotal step towards new scientific discoveries. Recently, \textit{two-stage} retrieval-augmented generated large language models (RAG-LLMs) have shown impressive advancements in this domain. Such a two-stage framework, especially the second stage (reranker), is particularly essential in the scientific domain, where subtle differences in terminology may have a greatly negative impact on the final factual-oriented or knowledge-intensive answers. Despite this significant progress, the potential and limitations of these works remain unexplored. In this work, we present a Scientific Rerank-oriented RAG Benchmark (SciRerankBench), for evaluating rerankers within RAG-LLMs systems, spanning five scientific subjects. To rigorously assess the reranker performance in terms of noise resilience, relevance disambiguation, and factual consistency, we develop three types of question-context-answer (Q-C-A) pairs, i.e., Noisy Contexts (NC), Semantically Similar but Logically Irrelevant Contexts (SSLI), and Counterfactual Contexts (CC). Through systematic evaluation of 13 widely used rerankers on five families of LLMs, we provide detailed insights into their relative strengths and limitations. To the best of our knowledge, SciRerankBench is the first benchmark specifically developed to evaluate rerankers within RAG-LLMs, which provides valuable observations and guidance for their future development. 

---
# IROTE: Human-like Traits Elicitation of Large Language Model via In-Context Self-Reflective Optimization 

**Authors**: Yuzhuo Bai, Shitong Duan, Muhua Huang, Jing Yao, Zhenghao Liu, Peng Zhang, Tun Lu, Xiaoyuan Yi, Maosong Sun, Xing Xie  

**Link**: [PDF](https://arxiv.org/pdf/2508.08719)  

**Abstract**: Trained on various human-authored corpora, Large Language Models (LLMs) have demonstrated a certain capability of reflecting specific human-like traits (e.g., personality or values) by prompting, benefiting applications like personalized LLMs and social simulations. However, existing methods suffer from the superficial elicitation problem: LLMs can only be steered to mimic shallow and unstable stylistic patterns, failing to embody the desired traits precisely and consistently across diverse tasks like humans. To address this challenge, we propose IROTE, a novel in-context method for stable and transferable trait elicitation. Drawing on psychological theories suggesting that traits are formed through identity-related reflection, our method automatically generates and optimizes a textual self-reflection within prompts, which comprises self-perceived experience, to stimulate LLMs' trait-driven behavior. The optimization is performed by iteratively maximizing an information-theoretic objective that enhances the connections between LLMs' behavior and the target trait, while reducing noisy redundancy in reflection without any fine-tuning, leading to evocative and compact trait reflection. Extensive experiments across three human trait systems manifest that one single IROTE-generated self-reflection can induce LLMs' stable impersonation of the target trait across diverse downstream tasks beyond simple questionnaire answering, consistently outperforming existing strong baselines. 

---
# Generative Modeling for Robust Deep Reinforcement Learning on the Traveling Salesman Problem 

**Authors**: Michael Li, Eric Bae, Christopher Haberland, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2508.08718)  

**Abstract**: The Traveling Salesman Problem (TSP) is a classic NP-hard combinatorial optimization task with numerous practical applications. Classic heuristic solvers can attain near-optimal performance for small problem instances, but become computationally intractable for larger problems. Real-world logistics problems such as dynamically re-routing last-mile deliveries demand a solver with fast inference time, which has led researchers to investigate specialized neural network solvers. However, neural networks struggle to generalize beyond the synthetic data they were trained on. In particular, we show that there exist TSP distributions that are realistic in practice, which also consistently lead to poor worst-case performance for existing neural approaches. To address this issue of distribution robustness, we present Combinatorial Optimization with Generative Sampling (COGS), where training data is sampled from a generative TSP model. We show that COGS provides better data coverage and interpolation in the space of TSP training distributions. We also present TSPLib50, a dataset of realistically distributed TSP samples, which tests real-world generalization ability without conflating this issue with instance size. We evaluate our method on various synthetic datasets as well as TSPLib50, and compare to state-of-the-art neural baselines. We demonstrate that COGS improves distribution robustness, with most performance gains coming from worst-case scenarios. 

---
# MultiAiTutor: Child-Friendly Educational Multilingual Speech Generation Tutor with LLMs 

**Authors**: Xiaoxue Gao, Huayun Zhang, Nancy F. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08715)  

**Abstract**: Generative speech models have demonstrated significant potential in personalizing teacher-student interactions, offering valuable real-world applications for language learning in children's education. However, achieving high-quality, child-friendly speech generation remains challenging, particularly for low-resource languages across diverse languages and cultural contexts. In this paper, we propose MultiAiTutor, an educational multilingual generative AI tutor with child-friendly designs, leveraging LLM architecture for speech generation tailored for educational purposes. We propose to integrate age-appropriate multilingual speech generation using LLM architectures, facilitating young children's language learning through culturally relevant image-description tasks in three low-resource languages: Singaporean-accent Mandarin, Malay, and Tamil. Experimental results from both objective metrics and subjective evaluations demonstrate the superior performance of the proposed MultiAiTutor compared to baseline methods. 

---
# A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models 

**Authors**: Lingzhe Zhang, Liancheng Fang, Chiming Duan, Minghua He, Leyi Pan, Pei Xiao, Shiyu Huang, Yunpeng Zhai, Xuming Hu, Philip S. Yu, Aiwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08712)  

**Abstract**: As text generation has become a core capability of modern Large Language Models (LLMs), it underpins a wide range of downstream applications. However, most existing LLMs rely on autoregressive (AR) generation, producing one token at a time based on previously generated context-resulting in limited generation speed due to the inherently sequential nature of the process. To address this challenge, an increasing number of researchers have begun exploring parallel text generation-a broad class of techniques aimed at breaking the token-by-token generation bottleneck and improving inference efficiency. Despite growing interest, there remains a lack of comprehensive analysis on what specific techniques constitute parallel text generation and how they improve inference performance. To bridge this gap, we present a systematic survey of parallel text generation methods. We categorize existing approaches into AR-based and Non-AR-based paradigms, and provide a detailed examination of the core techniques within each category. Following this taxonomy, we assess their theoretical trade-offs in terms of speed, quality, and efficiency, and examine their potential for combination and comparison with alternative acceleration strategies. Finally, based on our findings, we highlight recent advancements, identify open challenges, and outline promising directions for future research in parallel text generation. 

---
# SafeFix: Targeted Model Repair via Controlled Image Generation 

**Authors**: Ouyang Xu, Baoming Zhang, Ruiyu Mao, Yunhui Guo  

**Link**: [PDF](https://arxiv.org/pdf/2508.08701)  

**Abstract**: Deep learning models for visual recognition often exhibit systematic errors due to underrepresented semantic subpopulations. Although existing debugging frameworks can pinpoint these failures by identifying key failure attributes, repairing the model effectively remains difficult. Current solutions often rely on manually designed prompts to generate synthetic training images -- an approach prone to distribution shift and semantic errors. To overcome these challenges, we introduce a model repair module that builds on an interpretable failure attribution pipeline. Our approach uses a conditional text-to-image model to generate semantically faithful and targeted images for failure cases. To preserve the quality and relevance of the generated samples, we further employ a large vision-language model (LVLM) to filter the outputs, enforcing alignment with the original data distribution and maintaining semantic consistency. By retraining vision models with this rare-case-augmented synthetic dataset, we significantly reduce errors associated with rare cases. Our experiments demonstrate that this targeted repair strategy improves model robustness without introducing new bugs. Code is available at this https URL 

---
# MMIF-AMIN: Adaptive Loss-Driven Multi-Scale Invertible Dense Network for Multimodal Medical Image Fusion 

**Authors**: Tao Luo, Weihua Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08679)  

**Abstract**: Multimodal medical image fusion (MMIF) aims to integrate images from different modalities to produce a comprehensive image that enhances medical diagnosis by accurately depicting organ structures, tissue textures, and metabolic information. Capturing both the unique and complementary information across multiple modalities simultaneously is a key research challenge in MMIF. To address this challenge, this paper proposes a novel image fusion method, MMIF-AMIN, which features a new architecture that can effectively extract these unique and complementary features. Specifically, an Invertible Dense Network (IDN) is employed for lossless feature extraction from individual modalities. To extract complementary information between modalities, a Multi-scale Complementary Feature Extraction Module (MCFEM) is designed, which incorporates a hybrid attention mechanism, convolutional layers of varying sizes, and Transformers. An adaptive loss function is introduced to guide model learning, addressing the limitations of traditional manually-designed loss functions and enhancing the depth of data mining. Extensive experiments demonstrate that MMIF-AMIN outperforms nine state-of-the-art MMIF methods, delivering superior results in both quantitative and qualitative analyses. Ablation experiments confirm the effectiveness of each component of the proposed method. Additionally, extending MMIF-AMIN to other image fusion tasks also achieves promising performance. 

---
# Imposing AI: Deceptive design patterns against sustainability 

**Authors**: Anaëlle Beignon, Thomas Thibault, Nolwenn Maudet  

**Link**: [PDF](https://arxiv.org/pdf/2508.08672)  

**Abstract**: Generative AI is being massively deployed in digital services, at a scale that will result in significant environmental harm. We document how tech companies are transforming established user interfaces to impose AI use and show how and to what extent these strategies fit within established deceptive pattern categories. We identify two main design strategies that are implemented to impose AI use in both personal and professional contexts: imposing AI features in interfaces at the expense of existing non-AI features and promoting narratives about AI that make it harder to resist using it. We discuss opportunities for regulating the imposed adoption of AI features, which would inevitably lead to negative environmental effects. 

---
# Hallucinations in Code Change to Natural Language Generation: Prevalence and Evaluation of Detection Metrics 

**Authors**: Chunhua Liu, Hong Yi Lin, Patanamon Thongtanunam  

**Link**: [PDF](https://arxiv.org/pdf/2508.08661)  

**Abstract**: Language models have shown strong capabilities across a wide range of tasks in software engineering, such as code generation, yet they suffer from hallucinations. While hallucinations have been studied independently in natural language and code generation, their occurrence in tasks involving code changes which have a structurally complex and context-dependent format of code remains largely unexplored. This paper presents the first comprehensive analysis of hallucinations in two critical tasks involving code change to natural language generation: commit message generation and code review comment generation. We quantify the prevalence of hallucinations in recent language models and explore a range of metric-based approaches to automatically detect them. Our findings reveal that approximately 50\% of generated code reviews and 20\% of generated commit messages contain hallucinations. Whilst commonly used metrics are weak detectors on their own, combining multiple metrics substantially improves performance. Notably, model confidence and feature attribution metrics effectively contribute to hallucination detection, showing promise for inference-time detection.\footnote{All code and data will be released upon acceptance. 

---
# $\text{M}^{2}$LLM: Multi-view Molecular Representation Learning with Large Language Models 

**Authors**: Jiaxin Ju, Yizhen Zheng, Huan Yee Koh, Can Wang, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.08657)  

**Abstract**: Accurate molecular property prediction is a critical challenge with wide-ranging applications in chemistry, materials science, and drug discovery. Molecular representation methods, including fingerprints and graph neural networks (GNNs), achieve state-of-the-art results by effectively deriving features from molecular structures. However, these methods often overlook decades of accumulated semantic and contextual knowledge. Recent advancements in large language models (LLMs) demonstrate remarkable reasoning abilities and prior knowledge across scientific domains, leading us to hypothesize that LLMs can generate rich molecular representations when guided to reason in multiple perspectives. To address these gaps, we propose $\text{M}^{2}$LLM, a multi-view framework that integrates three perspectives: the molecular structure view, the molecular task view, and the molecular rules view. These views are fused dynamically to adapt to task requirements, and experiments demonstrate that $\text{M}^{2}$LLM achieves state-of-the-art performance on multiple benchmarks across classification and regression tasks. Moreover, we demonstrate that representation derived from LLM achieves exceptional performance by leveraging two core functionalities: the generation of molecular embeddings through their encoding capabilities and the curation of molecular features through advanced reasoning processes. 

---
# LLM driven Text-to-Table Generation through Sub-Tasks Guidance and Iterative Refinement 

**Authors**: Rajmohan C, Sarthak Harne, Arvind Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2508.08653)  

**Abstract**: Transforming unstructured text into structured data is a complex task, requiring semantic understanding, reasoning, and structural comprehension. While Large Language Models (LLMs) offer potential, they often struggle with handling ambiguous or domain-specific data, maintaining table structure, managing long inputs, and addressing numerical reasoning. This paper proposes an efficient system for LLM-driven text-to-table generation that leverages novel prompting techniques. Specifically, the system incorporates two key strategies: breaking down the text-to-table task into manageable, guided sub-tasks and refining the generated tables through iterative self-feedback. We show that this custom task decomposition allows the model to address the problem in a stepwise manner and improves the quality of the generated table. Furthermore, we discuss the benefits and potential risks associated with iterative self-feedback on the generated tables while highlighting the trade-offs between enhanced performance and computational cost. Our methods achieve strong results compared to baselines on two complex text-to-table generation datasets available in the public domain. 

---
# MiGrATe: Mixed-Policy GRPO for Adaptation at Test-Time 

**Authors**: Peter Phan, Dhruv Agarwal, Kavitha Srinivas, Horst Samulowitz, Pavan Kapanipathi, Andrew McCallum  

**Link**: [PDF](https://arxiv.org/pdf/2508.08641)  

**Abstract**: Large language models (LLMs) are increasingly being applied to black-box optimization tasks, from program synthesis to molecule design. Prior work typically leverages in-context learning to iteratively guide the model towards better solutions. Such methods, however, often struggle to balance exploration of new solution spaces with exploitation of high-reward ones. Recently, test-time training (TTT) with synthetic data has shown promise in improving solution quality. However, the need for hand-crafted training data tailored to each task limits feasibility and scalability across domains. To address this problem, we introduce MiGrATe-a method for online TTT that uses GRPO as a search algorithm to adapt LLMs at inference without requiring external training data. MiGrATe operates via a mixed-policy group construction procedure that combines on-policy sampling with two off-policy data selection techniques: greedy sampling, which selects top-performing past completions, and neighborhood sampling (NS), which generates completions structurally similar to high-reward ones. Together, these components bias the policy gradient towards exploitation of promising regions in solution space, while preserving exploration through on-policy sampling. We evaluate MiGrATe on three challenging domains-word search, molecule optimization, and hypothesis+program induction on the Abstraction and Reasoning Corpus (ARC)-and find that it consistently outperforms both inference-only and TTT baselines, demonstrating the potential of online TTT as a solution for complex search tasks without external supervision. 

---
# Securing Educational LLMs: A Generalised Taxonomy of Attacks on LLMs and DREAD Risk Assessment 

**Authors**: Farzana Zahid, Anjalika Sewwandi, Lee Brandon, Vimal Kumar, Roopak Sinha  

**Link**: [PDF](https://arxiv.org/pdf/2508.08629)  

**Abstract**: Due to perceptions of efficiency and significant productivity gains, various organisations, including in education, are adopting Large Language Models (LLMs) into their workflows. Educator-facing, learner-facing, and institution-facing LLMs, collectively, Educational Large Language Models (eLLMs), complement and enhance the effectiveness of teaching, learning, and academic operations. However, their integration into an educational setting raises significant cybersecurity concerns. A comprehensive landscape of contemporary attacks on LLMs and their impact on the educational environment is missing. This study presents a generalised taxonomy of fifty attacks on LLMs, which are categorized as attacks targeting either models or their infrastructure. The severity of these attacks is evaluated in the educational sector using the DREAD risk assessment framework. Our risk assessment indicates that token smuggling, adversarial prompts, direct injection, and multi-step jailbreak are critical attacks on eLLMs. The proposed taxonomy, its application in the educational environment, and our risk assessment will help academic and industrial practitioners to build resilient solutions that protect learners and institutions. 

---
# QoE-Aware Service Provision for Mobile AR Rendering: An Agent-Driven Approach 

**Authors**: Conghao Zhou, Lulu Sun, Xiucheng Wang, Peng Yang, Feng Lyu, Sihan Lu, Xuemin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.08627)  

**Abstract**: Mobile augmented reality (MAR) is envisioned as a key immersive application in 6G, enabling virtual content rendering aligned with the physical environment through device pose estimation. In this paper, we propose a novel agent-driven communication service provisioning approach for edge-assisted MAR, aiming to reduce communication overhead between MAR devices and the edge server while ensuring the quality of experience (QoE). First, to address the inaccessibility of MAR application-specific information to the network controller, we establish a digital agent powered by large language models (LLMs) on behalf of the MAR service provider, bridging the data and function gap between the MAR service and network domains. Second, to cope with the user-dependent and dynamic nature of data traffic patterns for individual devices, we develop a user-level QoE modeling method that captures the relationship between communication resource demands and perceived user QoE, enabling personalized, agent-driven communication resource management. Trace-driven simulation results demonstrate that the proposed approach outperforms conventional LLM-based QoE-aware service provisioning methods in both user-level QoE modeling accuracy and communication resource efficiency. 

---
# Transferable Model-agnostic Vision-Language Model Adaptation for Efficient Weak-to-Strong Generalization 

**Authors**: Jihwan Park, Taehoon song, Sanghyeok Lee, Miso Choi, Hyunwoo J. Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.08604)  

**Abstract**: Vision-Language Models (VLMs) have been widely used in various visual recognition tasks due to their remarkable generalization capabilities. As these models grow in size and complexity, fine-tuning becomes costly, emphasizing the need to reuse adaptation knowledge from 'weaker' models to efficiently enhance 'stronger' ones. However, existing adaptation transfer methods exhibit limited transferability across models due to their model-specific design and high computational demands. To tackle this, we propose Transferable Model-agnostic adapter (TransMiter), a light-weight adapter that improves vision-language models 'without backpropagation'. TransMiter captures the knowledge gap between pre-trained and fine-tuned VLMs, in an 'unsupervised' manner. Once trained, this knowledge can be seamlessly transferred across different models without the need for backpropagation. Moreover, TransMiter consists of only a few layers, inducing a negligible additional inference cost. Notably, supplementing the process with a few labeled data further yields additional performance gain, often surpassing a fine-tuned stronger model, with a marginal training cost. Experimental results and analyses demonstrate that TransMiter effectively and efficiently transfers adaptation knowledge while preserving generalization abilities across VLMs of different sizes and architectures in visual recognition tasks. 

---
# Yan: Foundational Interactive Video Generation 

**Authors**: Yan Team  

**Link**: [PDF](https://arxiv.org/pdf/2508.08601)  

**Abstract**: We present Yan, a foundational framework for interactive video generation, covering the entire pipeline from simulation and generation to editing. Specifically, Yan comprises three core modules. AAA-level Simulation: We design a highly-compressed, low-latency 3D-VAE coupled with a KV-cache-based shift-window denoising inference process, achieving real-time 1080P/60FPS interactive simulation. Multi-Modal Generation: We introduce a hierarchical autoregressive caption method that injects game-specific knowledge into open-domain multi-modal video diffusion models (VDMs), then transforming the VDM into a frame-wise, action-controllable, real-time infinite interactive video generator. Notably, when the textual and visual prompts are sourced from different domains, the model demonstrates strong generalization, allowing it to blend and compose the style and mechanics across domains flexibly according to user prompts. Multi-Granularity Editing: We propose a hybrid model that explicitly disentangles interactive mechanics simulation from visual rendering, enabling multi-granularity video content editing during interaction through text. Collectively, Yan offers an integration of these modules, pushing interactive video generation beyond isolated capabilities toward a comprehensive AI-driven interactive creation paradigm, paving the way for the next generation of creative tools, media, and entertainment. The project page is: this https URL. 

---
# Generative AI for Critical Infrastructure in Smart Grids: A Unified Framework for Synthetic Data Generation and Anomaly Detection 

**Authors**: Aydin Zaboli, Junho Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.08593)  

**Abstract**: In digital substations, security events pose significant challenges to the sustained operation of power systems. To mitigate these challenges, the implementation of robust defense strategies is critically important. A thorough process of anomaly identification and detection in information and communication technology (ICT) frameworks is crucial to ensure secure and reliable communication and coordination between interconnected devices within digital substations. Hence, this paper addresses the critical cybersecurity challenges confronting IEC61850-based digital substations within modern smart grids, where the integration of advanced communication protocols, e.g., generic object-oriented substation event (GOOSE), has enhanced energy management and introduced significant vulnerabilities to cyberattacks. Focusing on the limitations of traditional anomaly detection systems (ADSs) in detecting threats, this research proposes a transformative approach by leveraging generative AI (GenAI) to develop robust ADSs. The primary contributions include the suggested advanced adversarial traffic mutation (AATM) technique to generate synthesized and balanced datasets for GOOSE messages, ensuring protocol compliance and enabling realistic zero-day attack pattern creation to address data scarcity. Then, the implementation of GenAI-based ADSs incorporating the task-oriented dialogue (ToD) processes has been explored for improved detection of attack patterns. Finally, a comparison of the GenAI-based ADS with machine learning (ML)-based ADSs has been implemented to showcase the outperformance of the GenAI-based frameworks considering the AATM-generated GOOSE datasets and standard/advanced performance evaluation metrics. 

---
# DepressLLM: Interpretable domain-adapted language model for depression detection from real-world narratives 

**Authors**: Sehwan Moon, Aram Lee, Jeong Eun Kim, Hee-Ju Kang, Il-Seon Shin, Sung-Wan Kim, Jae-Min Kim, Min Jhon, Ju-Wan Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.08591)  

**Abstract**: Advances in large language models (LLMs) have enabled a wide range of applications. However, depression prediction is hindered by the lack of large-scale, high-quality, and rigorously annotated datasets. This study introduces DepressLLM, trained and evaluated on a novel corpus of 3,699 autobiographical narratives reflecting both happiness and distress. DepressLLM provides interpretable depression predictions and, via its Score-guided Token Probability Summation (SToPS) module, delivers both improved classification performance and reliable confidence estimates, achieving an AUC of 0.789, which rises to 0.904 on samples with confidence $\geq$ 0.95. To validate its robustness to heterogeneous data, we evaluated DepressLLM on in-house datasets, including an Ecological Momentary Assessment (EMA) corpus of daily stress and mood recordings, and on public clinical interview data. Finally, a psychiatric review of high-confidence misclassifications highlighted key model and data limitations that suggest directions for future refinements. These findings demonstrate that interpretable AI can enable earlier diagnosis of depression and underscore the promise of medical AI in psychiatry. 

---
# AI Security Map: Holistic Organization of AI Security Technologies and Impacts on Stakeholders 

**Authors**: Hiroya Kato, Kentaro Kita, Kento Hasegawa, Seira Hidano  

**Link**: [PDF](https://arxiv.org/pdf/2508.08583)  

**Abstract**: As the social implementation of AI has been steadily progressing, research and development related to AI security has also been increasing. However, existing studies have been limited to organizing related techniques, attacks, defenses, and risks in terms of specific domains or AI elements. Thus, it extremely difficult to understand the relationships among them and how negative impacts on stakeholders are brought about. In this paper, we argue that the knowledge, technologies, and social impacts related to AI security should be holistically organized to help understand relationships among them. To this end, we first develop an AI security map that holistically organizes interrelationships among elements related to AI security as well as negative impacts on information systems and stakeholders. This map consists of the two aspects, namely the information system aspect (ISA) and the external influence aspect (EIA). The elements that AI should fulfill within information systems are classified under the ISA. The EIA includes elements that affect stakeholders as a result of AI being attacked or misused. For each element, corresponding negative impacts are identified. By referring to the AI security map, one can understand the potential negative impacts, along with their causes and countermeasures. Additionally, our map helps clarify how the negative impacts on AI-based systems relate to those on stakeholders. We show some findings newly obtained by referring to our map. We also provide several recommendations and open problems to guide future AI security communities. 

---
# Who pays the RENT? Implications of Spatial Inequality for Prediction-Based Allocation Policies 

**Authors**: Tasfia Mashiat, Patrick J. Fowler, Sanmay Das  

**Link**: [PDF](https://arxiv.org/pdf/2508.08573)  

**Abstract**: AI-powered scarce resource allocation policies rely on predictions to target either specific individuals (e.g., high-risk) or settings (e.g., neighborhoods). Recent research on individual-level targeting demonstrates conflicting results; some models show that targeting is not useful when inequality is high, while other work demonstrates potential benefits. To study and reconcile this apparent discrepancy, we develop a stylized framework based on the Mallows model to understand how the spatial distribution of inequality affects the effectiveness of door-to-door outreach policies. We introduce the RENT (Relative Efficiency of Non-Targeting) metric, which we use to assess the effectiveness of targeting approaches compared with neighborhood-based approaches in preventing tenant eviction when high-risk households are more versus less spatially concentrated. We then calibrate the model parameters to eviction court records collected in a medium-sized city in the USA. Results demonstrate considerable gains in the number of high-risk households canvassed through individually targeted policies, even in a highly segregated metro area with concentrated risks of eviction. We conclude that apparent discrepancies in the prior literature can be reconciled by considering 1) the source of deployment costs and 2) the observed versus modeled concentrations of risk. Our results inform the deployment of AI-based solutions in social service provision that account for particular applications and geographies. 

---
# Superclass-Guided Representation Disentanglement for Spurious Correlation Mitigation 

**Authors**: Chenruo Liu, Hongjun Liu, Zeyu Lai, Yiqiu Shen, Chen Zhao, Qi Lei  

**Link**: [PDF](https://arxiv.org/pdf/2508.08570)  

**Abstract**: To enhance group robustness to spurious correlations, prior work often relies on auxiliary annotations for groups or spurious features and assumes identical sets of groups across source and target domains. These two requirements are both unnatural and impractical in real-world settings. To overcome these limitations, we propose a method that leverages the semantic structure inherent in class labels--specifically, superclass information--to naturally reduce reliance on spurious features. Our model employs gradient-based attention guided by a pre-trained vision-language model to disentangle superclass-relevant and irrelevant features. Then, by promoting the use of all superclass-relevant features for prediction, our approach achieves robustness to more complex spurious correlations without the need to annotate any source samples. Experiments across diverse datasets demonstrate that our method significantly outperforms baselines in domain generalization tasks, with clear improvements in both quantitative metrics and qualitative visualizations. 

---
# UQGNN: Uncertainty Quantification of Graph Neural Networks for Multivariate Spatiotemporal Prediction 

**Authors**: Dahai Yu, Dingyi Zhuang, Lin Jiang, Rongchao Xu, Xinyue Ye, Yuheng Bu, Shenhao Wang, Guang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08551)  

**Abstract**: Spatiotemporal prediction plays a critical role in numerous real-world applications such as urban planning, transportation optimization, disaster response, and pandemic control. In recent years, researchers have made significant progress by developing advanced deep learning models for spatiotemporal prediction. However, most existing models are deterministic, i.e., predicting only the expected mean values without quantifying uncertainty, leading to potentially unreliable and inaccurate outcomes. While recent studies have introduced probabilistic models to quantify uncertainty, they typically focus on a single phenomenon (e.g., taxi, bike, crime, or traffic crashes), thereby neglecting the inherent correlations among heterogeneous urban phenomena. To address the research gap, we propose a novel Graph Neural Network with Uncertainty Quantification, termed UQGNN for multivariate spatiotemporal prediction. UQGNN introduces two key innovations: (i) an Interaction-aware Spatiotemporal Embedding Module that integrates a multivariate diffusion graph convolutional network and an interaction-aware temporal convolutional network to effectively capture complex spatial and temporal interaction patterns, and (ii) a multivariate probabilistic prediction module designed to estimate both expected mean values and associated uncertainties. Extensive experiments on four real-world multivariate spatiotemporal datasets from Shenzhen, New York City, and Chicago demonstrate that UQGNN consistently outperforms state-of-the-art baselines in both prediction accuracy and uncertainty quantification. For example, on the Shenzhen dataset, UQGNN achieves a 5% improvement in both prediction accuracy and uncertainty quantification. 

---
# OmniLLP: Enhancing LLM-based Log Level Prediction with Context-Aware Retrieval 

**Authors**: Youssef Esseddiq Ouatiti, Mohammed Sayagh, Bram Adams, Ahmed E. Hassan  

**Link**: [PDF](https://arxiv.org/pdf/2508.08545)  

**Abstract**: Developers insert logging statements in source code to capture relevant runtime information essential for maintenance and debugging activities. Log level choice is an integral, yet tricky part of the logging activity as it controls log verbosity and therefore influences systems' observability and performance. Recent advances in ML-based log level prediction have leveraged large language models (LLMs) to propose log level predictors (LLPs) that demonstrated promising performance improvements (AUC between 0.64 and 0.8). Nevertheless, current LLM-based LLPs rely on randomly selected in-context examples, overlooking the structure and the diverse logging practices within modern software projects. In this paper, we propose OmniLLP, a novel LLP enhancement framework that clusters source files based on (1) semantic similarity reflecting the code's functional purpose, and (2) developer ownership cohesion. By retrieving in-context learning examples exclusively from these semantic and ownership aware clusters, we aim to provide more coherent prompts to LLPs leveraging LLMs, thereby improving their predictive accuracy. Our results show that both semantic and ownership-aware clusterings statistically significantly improve the accuracy (by up to 8\% AUC) of the evaluated LLM-based LLPs compared to random predictors (i.e., leveraging randomly selected in-context examples from the whole project). Additionally, our approach that combines the semantic and ownership signal for in-context prediction achieves an impressive 0.88 to 0.96 AUC across our evaluated projects. Our findings highlight the value of integrating software engineering-specific context, such as code semantic and developer ownership signals into LLM-LLPs, offering developers a more accurate, contextually-aware approach to logging and therefore, enhancing system maintainability and observability. 

---
# AI Agents and the Law 

**Authors**: Mark O. Riedl, Deven R. Desai  

**Link**: [PDF](https://arxiv.org/pdf/2508.08544)  

**Abstract**: As AI becomes more "agentic," it faces technical and socio-legal issues it must address if it is to fulfill its promise of increased economic productivity and efficiency. This paper uses technical and legal perspectives to explain how things change when AI systems start being able to directly execute tasks on behalf of a user. We show how technical conceptions of agents track some, but not all, socio-legal conceptions of agency. That is, both computer science and the law recognize the problems of under-specification for an agent, and both disciplines have robust conceptions of how to address ensuring an agent does what the programmer, or in the law, the principal desires and no more. However, to date, computer science has under-theorized issues related to questions of loyalty and to third parties that interact with an agent, both of which are central parts of the law of agency. First, we examine the correlations between implied authority in agency law and the principle of value-alignment in AI, wherein AI systems must operate under imperfect objective specification. Second, we reveal gaps in the current computer science view of agents pertaining to the legal concepts of disclosure and loyalty, and how failure to account for them can result in unintended effects in AI ecommerce agents. In surfacing these gaps, we show a path forward for responsible AI agent development and deployment. 

---
# M3-Net: A Cost-Effective Graph-Free MLP-Based Model for Traffic Prediction 

**Authors**: Guangyin Jin, Sicong Lai, Xiaoshuai Hao, Mingtao Zhang, Jinlei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08543)  

**Abstract**: Achieving accurate traffic prediction is a fundamental but crucial task in the development of current intelligent transportation this http URL of the mainstream methods that have made breakthroughs in traffic prediction rely on spatio-temporal graph neural networks, spatio-temporal attention mechanisms, etc. The main challenges of the existing deep learning approaches are that they either depend on a complete traffic network structure or require intricate model designs to capture complex spatio-temporal dependencies. These limitations pose significant challenges for the efficient deployment and operation of deep learning models on large-scale datasets. To address these challenges, we propose a cost-effective graph-free Multilayer Perceptron (MLP) based model M3-Net for traffic prediction. Our proposed model not only employs time series and spatio-temporal embeddings for efficient feature processing but also first introduces a novel MLP-Mixer architecture with a mixture of experts (MoE) mechanism. Extensive experiments conducted on multiple real datasets demonstrate the superiority of the proposed model in terms of prediction performance and lightweight deployment. 

---
# LLM-Driven Adaptive 6G-Ready Wireless Body Area Networks: Survey and Framework 

**Authors**: Azin Sabzian, Mohammad Jalili Torkamani, Negin Mahmoudi, Kiana Kiashemshaki  

**Link**: [PDF](https://arxiv.org/pdf/2508.08535)  

**Abstract**: Wireless Body Area Networks (WBANs) enable continuous monitoring of physiological signals for applications ranging from chronic disease management to emergency response. Recent advances in 6G communications, post-quantum cryptography, and energy harvesting have the potential to enhance WBAN performance. However, integrating these technologies into a unified, adaptive system remains a challenge. This paper surveys some of the most well-known Wireless Body Area Network (WBAN) architectures, routing strategies, and security mechanisms, identifying key gaps in adaptability, energy efficiency, and quantum-resistant security. We propose a novel Large Language Model-driven adaptive WBAN framework in which a Large Language Model acts as a cognitive control plane, coordinating routing, physical layer selection, micro-energy harvesting, and post-quantum security in real time. Our review highlights the limitations of current heuristic-based designs and outlines a research agenda for resource-constrained, 6G-ready medical systems. This approach aims to enable ultra-reliable, secure, and self-optimizing WBANs for next-generation mobile health applications. 

---
# Playing Atari Space Invaders with Sparse Cosine Optimized Policy Evolution 

**Authors**: Jim O'Connor, Jay B. Nash, Derin Gezgin, Gary B. Parker  

**Link**: [PDF](https://arxiv.org/pdf/2508.08526)  

**Abstract**: Evolutionary approaches have previously been shown to be effective learning methods for a diverse set of domains. However, the domain of game-playing poses a particular challenge for evolutionary methods due to the inherently large state space of video games. As the size of the input state expands, the size of the policy must also increase in order to effectively learn the temporal patterns in the game space. Consequently, a larger policy must contain more trainable parameters, exponentially increasing the size of the search space. Any increase in search space is highly problematic for evolutionary methods, as increasing the number of trainable parameters is inversely correlated with convergence speed. To reduce the size of the input space while maintaining a meaningful representation of the original space, we introduce Sparse Cosine Optimized Policy Evolution (SCOPE). SCOPE utilizes the Discrete Cosine Transform (DCT) as a pseudo attention mechanism, transforming an input state into a coefficient matrix. By truncating and applying sparsification to this matrix, we reduce the dimensionality of the input space while retaining the highest energy features of the original input. We demonstrate the effectiveness of SCOPE as the policy for the Atari game Space Invaders. In this task, SCOPE with CMA-ES outperforms evolutionary methods that consider an unmodified input state, such as OpenAI-ES and HyperNEAT. SCOPE also outperforms simple reinforcement learning methods, such as DQN and A3C. SCOPE achieves this result through reducing the input size by 53% from 33,600 to 15,625 then using a bilinear affine mapping of sparse DCT coefficients to policy actions learned by the CMA-ES algorithm. 

---
# StreetViewAI: Making Street View Accessible Using Context-Aware Multimodal AI 

**Authors**: Jon E. Froehlich, Alexander Fiannaca, Nimer Jaber, Victor Tsara, Shaun Kane  

**Link**: [PDF](https://arxiv.org/pdf/2508.08524)  

**Abstract**: Interactive streetscape mapping tools such as Google Street View (GSV) and Meta Mapillary enable users to virtually navigate and experience real-world environments via immersive 360° imagery but remain fundamentally inaccessible to blind users. We introduce StreetViewAI, the first-ever accessible street view tool, which combines context-aware, multimodal AI, accessible navigation controls, and conversational speech. With StreetViewAI, blind users can virtually examine destinations, engage in open-world exploration, or virtually tour any of the over 220 billion images and 100+ countries where GSV is deployed. We iteratively designed StreetViewAI with a mixed-visual ability team and performed an evaluation with eleven blind users. Our findings demonstrate the value of an accessible street view in supporting POI investigations and remote route planning. We close by enumerating key guidelines for future work. 

---
# VISOR: Visual Input-based Steering for Output Redirection in Vision-Language Models 

**Authors**: Mansi Phute, Ravikumar Balakrishnan  

**Link**: [PDF](https://arxiv.org/pdf/2508.08521)  

**Abstract**: Vision Language Models (VLMs) are increasingly being used in a broad range of applications, bringing their security and behavioral control to the forefront. While existing approaches for behavioral control or output redirection, like system prompting in VLMs, are easily detectable and often ineffective, activation-based steering vectors require invasive runtime access to model internals--incompatible with API-based services and closed-source deployments. We introduce VISOR (Visual Input-based Steering for Output Redirection), a novel method that achieves sophisticated behavioral control through optimized visual inputs alone. By crafting universal steering images that induce target activation patterns, VISOR enables practical deployment across all VLM serving modalities while remaining imperceptible compared to explicit textual instructions. We validate VISOR on LLaVA-1.5-7B across three critical alignment tasks: refusal, sycophancy and survival instinct. A single 150KB steering image matches steering vector performance within 1-2% for positive behavioral shifts while dramatically exceeding it for negative steering--achieving up to 25% shifts from baseline compared to steering vectors' modest changes. Unlike system prompting (3-4% shifts), VISOR provides robust bidirectional control while maintaining 99.9% performance on 14,000 unrelated MMLU tasks. Beyond eliminating runtime overhead and model access requirements, VISOR exposes a critical security vulnerability: adversaries can achieve sophisticated behavioral manipulation through visual channels alone, bypassing text-based defenses. Our work fundamentally re-imagines multimodal model control and highlights the urgent need for defenses against visual steering attacks. 

---
# Using LLMs to Capture Users' Temporal Context for Recommendation 

**Authors**: Milad Sabouri, Masoud Mansoury, Kun Lin, Bamshad Mobasher  

**Link**: [PDF](https://arxiv.org/pdf/2508.08512)  

**Abstract**: Effective recommender systems demand dynamic user understanding, especially in complex, evolving environments. Traditional user profiling often fails to capture the nuanced, temporal contextual factors of user preferences, such as transient short-term interests and enduring long-term tastes. This paper presents an assessment of Large Language Models (LLMs) for generating semantically rich, time-aware user profiles. We do not propose a novel end-to-end recommendation architecture; instead, the core contribution is a systematic investigation into the degree of LLM effectiveness in capturing the dynamics of user context by disentangling short-term and long-term preferences. This approach, framing temporal preferences as dynamic user contexts for recommendations, adaptively fuses these distinct contextual components into comprehensive user embeddings. The evaluation across Movies&TV and Video Games domains suggests that while LLM-generated profiles offer semantic depth and temporal structure, their effectiveness for context-aware recommendations is notably contingent on the richness of user interaction histories. Significant gains are observed in dense domains (e.g., Movies&TV), whereas improvements are less pronounced in sparse environments (e.g., Video Games). This work highlights LLMs' nuanced potential in enhancing user profiling for adaptive, context-aware recommendations, emphasizing the critical role of dataset characteristics for practical applicability. 

---
# Steerable Pluralism: Pluralistic Alignment via Few-Shot Comparative Regression 

**Authors**: Jadie Adams, Brian Hu, Emily Veenhuis, David Joy, Bharadwaj Ravichandran, Aaron Bray, Anthony Hoogs, Arslan Basharat  

**Link**: [PDF](https://arxiv.org/pdf/2508.08509)  

**Abstract**: Large language models (LLMs) are currently aligned using techniques such as reinforcement learning from human feedback (RLHF). However, these methods use scalar rewards that can only reflect user preferences on average. Pluralistic alignment instead seeks to capture diverse user preferences across a set of attributes, moving beyond just helpfulness and harmlessness. Toward this end, we propose a steerable pluralistic model based on few-shot comparative regression that can adapt to individual user preferences. Our approach leverages in-context learning and reasoning, grounded in a set of fine-grained attributes, to compare response options and make aligned choices. To evaluate our algorithm, we also propose two new steerable pluralistic benchmarks by adapting the Moral Integrity Corpus (MIC) and the HelpSteer2 datasets, demonstrating the applicability of our approach to value-aligned decision-making and reward modeling, respectively. Our few-shot comparative regression approach is interpretable and compatible with different attributes and LLMs, while outperforming multiple baseline and state-of-the-art methods. Our work provides new insights and research directions in pluralistic alignment, enabling a more fair and representative use of LLMs and advancing the state-of-the-art in ethical AI. 

---
# When the Domain Expert Has No Time and the LLM Developer Has No Clinical Expertise: Real-World Lessons from LLM Co-Design in a Safety-Net Hospital 

**Authors**: Avni Kothari, Patrick Vossler, Jean Digitale, Mohammad Forouzannia, Elise Rosenberg, Michele Lee, Jennee Bryant, Melanie Molina, James Marks, Lucas Zier, Jean Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.08504)  

**Abstract**: Large language models (LLMs) have the potential to address social and behavioral determinants of health by transforming labor intensive workflows in resource-constrained settings. Creating LLM-based applications that serve the needs of underserved communities requires a deep understanding of their local context, but it is often the case that neither LLMs nor their developers possess this local expertise, and the experts in these communities often face severe time/resource constraints. This creates a disconnect: how can one engage in meaningful co-design of an LLM-based application for an under-resourced community when the communication channel between the LLM developer and domain expert is constrained? We explored this question through a real-world case study, in which our data science team sought to partner with social workers at a safety net hospital to build an LLM application that summarizes patients' social needs. Whereas prior works focus on the challenge of prompt tuning, we found that the most critical challenge in this setting is the careful and precise specification of \what information to surface to providers so that the LLM application is accurate, comprehensive, and verifiable. Here we present a novel co-design framework for settings with limited access to domain experts, in which the summary generation task is first decomposed into individually-optimizable attributes and then each attribute is efficiently refined and validated through a multi-tier cascading approach. 

---
# Momentum Point-Perplexity Mechanics in Large Language Models 

**Authors**: Lorenzo Tomaz, Judd Rosenblatt, Thomas Berry Jones, Diogo Schwerz de Lucena  

**Link**: [PDF](https://arxiv.org/pdf/2508.08492)  

**Abstract**: We take a physics-based approach to studying how the internal hidden states of large language models change from token to token during inference. Across 20 open-source transformer models (135M-3B parameters), we find that a quantity combining the rate of change in hidden states and the model's next-token certainty, analogous to energy in physics, remains nearly constant. Random-weight models conserve this "energy" more tightly than pre-trained ones, while training shifts models into a faster, more decisive regime with greater variability. Using this "log-Lagrangian" view, we derive a control method called Jacobian steering, which perturbs hidden states in the minimal way needed to favor a target token. This approach maintained near-constant energy in two tested models and produced continuations rated higher in semantic quality than the models' natural outputs. Viewing transformers through this mechanics lens offers a principled basis for interpretability, anomaly detection, and low-risk steering. This could help make powerful models more predictable and aligned with human intent. 

---
# MAViS: A Multi-Agent Framework for Long-Sequence Video Storytelling 

**Authors**: Qian Wang, Ziqi Huang, Ruoxi Jia, Paul Debevec, Ning Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08487)  

**Abstract**: Despite recent advances, long-sequence video generation frameworks still suffer from significant limitations: poor assistive capability, suboptimal visual quality, and limited expressiveness. To mitigate these limitations, we propose MAViS, an end-to-end multi-agent collaborative framework for long-sequence video storytelling. MAViS orchestrates specialized agents across multiple stages, including script writing, shot designing, character modeling, keyframe generation, video animation, and audio generation. In each stage, agents operate under the 3E Principle -- Explore, Examine, and Enhance -- to ensure the completeness of intermediate outputs. Considering the capability limitations of current generative models, we propose the Script Writing Guidelines to optimize compatibility between scripts and generative tools. Experimental results demonstrate that MAViS achieves state-of-the-art performance in assistive capability, visual quality, and video expressiveness. Its modular framework further enables scalability with diverse generative models and tools. With just a brief user prompt, MAViS is capable of producing high-quality, expressive long-sequence video storytelling, enriching inspirations and creativity for users. To the best of our knowledge, MAViS is the only framework that provides multimodal design output -- videos with narratives and background music. 

---
# Empowering Children to Create AI-Enabled Augmented Reality Experiences 

**Authors**: Lei Zhang, Shuyao Zhou, Amna Liaqat, Tinney Mak, Brian Berengard, Emily Qian, Andrés Monroy-Hernández  

**Link**: [PDF](https://arxiv.org/pdf/2508.08467)  

**Abstract**: Despite their potential to enhance children's learning experiences, AI-enabled AR technologies are predominantly used in ways that position children as consumers rather than creators. We introduce Capybara, an AR-based and AI-powered visual programming environment that empowers children to create, customize, and program 3D characters overlaid onto the physical world. Capybara enables children to create virtual characters and accessories using text-to-3D generative AI models, and to animate these characters through auto-rigging and body tracking. In addition, our system employs vision-based AI models to recognize physical objects, allowing children to program interactive behaviors between virtual characters and their physical surroundings. We demonstrate the expressiveness of Capybara through a set of novel AR experiences. We conducted user studies with 20 children in the United States and Argentina. Our findings suggest that Capybara can empower children to harness AI in authoring personalized and engaging AR experiences that seamlessly bridge the virtual and physical worlds. 

---
# Temporal User Profiling with LLMs: Balancing Short-Term and Long-Term Preferences for Recommendations 

**Authors**: Milad Sabouri, Masoud Mansoury, Kun Lin, Bamshad Mobasher  

**Link**: [PDF](https://arxiv.org/pdf/2508.08454)  

**Abstract**: Accurately modeling user preferences is crucial for improving the performance of content-based recommender systems. Existing approaches often rely on simplistic user profiling methods, such as averaging or concatenating item embeddings, which fail to capture the nuanced nature of user preference dynamics, particularly the interactions between long-term and short-term preferences. In this work, we propose LLM-driven Temporal User Profiling (LLM-TUP), a novel method for user profiling that explicitly models short-term and long-term preferences by leveraging interaction timestamps and generating natural language representations of user histories using a large language model (LLM). These representations are encoded into high-dimensional embeddings using a pre-trained BERT model, and an attention mechanism is applied to dynamically fuse the short-term and long-term embeddings into a comprehensive user profile. Experimental results on real-world datasets demonstrate that LLM-TUP achieves substantial improvements over several baselines, underscoring the effectiveness of our temporally aware user-profiling approach and the use of semantically rich user profiles, generated by LLMs, for personalized content-based recommendation. 

---
# Fast weight programming and linear transformers: from machine learning to neurobiology 

**Authors**: Kazuki Irie, Samuel J. Gershman  

**Link**: [PDF](https://arxiv.org/pdf/2508.08435)  

**Abstract**: Recent advances in artificial neural networks for machine learning, and language modeling in particular, have established a family of recurrent neural network (RNN) architectures that, unlike conventional RNNs with vector-form hidden states, use two-dimensional (2D) matrix-form hidden states. Such 2D-state RNNs, known as Fast Weight Programmers (FWPs), can be interpreted as a neural network whose synaptic weights (called fast weights) dynamically change over time as a function of input observations, and serve as short-term memory storage; corresponding synaptic weight modifications are controlled or programmed by another network (the programmer) whose parameters are trained (e.g., by gradient descent). In this Primer, we review the technical foundations of FWPs, their computational characteristics, and their connections to transformers and state space models. We also discuss connections between FWPs and models of synaptic plasticity in the brain, suggesting a convergence of natural and artificial intelligence. 

---
# Rethinking Tokenization for Rich Morphology: The Dominance of Unigram over BPE and Morphological Alignment 

**Authors**: Saketh Reddy Vemula, Dipti Mishra Sharma, Parameswari Krishnamurthy  

**Link**: [PDF](https://arxiv.org/pdf/2508.08424)  

**Abstract**: Prior work on language modeling showed conflicting findings about whether morphologically aligned approaches to tokenization improve performance, particularly for languages with complex morphology. To investigate this, we select a typologically diverse set of languages: Telugu (agglutinative), Hindi (primarily fusional with some agglutination), and English (fusional). We conduct a comprehensive evaluation of language models -- starting from tokenizer training and extending through the finetuning and downstream task evaluation. To account for the consistent performance differences observed across tokenizer variants, we focus on two key factors: morphological alignment and tokenization quality. To assess morphological alignment of tokenizers in Telugu, we create a dataset containing gold morpheme segmentations of 600 derivational and 7000 inflectional word forms.
Our experiments reveal that better morphological alignment correlates positively -- though moderately -- with performance in syntax-based tasks such as Parts-of-Speech tagging, Named Entity Recognition and Dependency Parsing. However, we also find that the tokenizer algorithm (Byte-pair Encoding vs. Unigram) plays a more significant role in influencing downstream performance than morphological alignment alone. Naive Unigram tokenizers outperform others across most settings, though hybrid tokenizers that incorporate morphological segmentation significantly improve performance within the BPE framework. In contrast, intrinsic metrics like Corpus Token Count (CTC) and Rényi entropy showed no correlation with downstream performance. 

---
# Neural Tangent Knowledge Distillation for Optical Convolutional Networks 

**Authors**: Jinlin Xiang, Minho Choi, Yubo Zhang, Zhihao Zhou, Arka Majumdar, Eli Shlizerman  

**Link**: [PDF](https://arxiv.org/pdf/2508.08421)  

**Abstract**: Hybrid Optical Neural Networks (ONNs, typically consisting of an optical frontend and a digital backend) offer an energy-efficient alternative to fully digital deep networks for real-time, power-constrained systems. However, their adoption is limited by two main challenges: the accuracy gap compared to large-scale networks during training, and discrepancies between simulated and fabricated systems that further degrade accuracy. While previous work has proposed end-to-end optimizations for specific datasets (e.g., MNIST) and optical systems, these approaches typically lack generalization across tasks and hardware designs. To address these limitations, we propose a task-agnostic and hardware-agnostic pipeline that supports image classification and segmentation across diverse optical systems. To assist optical system design before training, we estimate achievable model accuracy based on user-specified constraints such as physical size and the dataset. For training, we introduce Neural Tangent Knowledge Distillation (NTKD), which aligns optical models with electronic teacher networks, thereby narrowing the accuracy gap. After fabrication, NTKD also guides fine-tuning of the digital backend to compensate for implementation errors. Experiments on multiple datasets (e.g., MNIST, CIFAR, Carvana Masking) and hardware configurations show that our pipeline consistently improves ONN performance and enables practical deployment in both pre-fabrication simulations and physical implementations. 

---
# Generating Query-Relevant Document Summaries via Reinforcement Learning 

**Authors**: Nitin Yadav, Changsung Kang, Hongwei Shang, Ming Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.08404)  

**Abstract**: E-commerce search engines often rely solely on product titles as input for ranking models with latency constraints. However, this approach can result in suboptimal relevance predictions, as product titles often lack sufficient detail to capture query intent. While product descriptions provide richer information, their verbosity and length make them unsuitable for real-time ranking, particularly for computationally expensive architectures like cross-encoder ranking models. To address this challenge, we propose ReLSum, a novel reinforcement learning framework designed to generate concise, query-relevant summaries of product descriptions optimized for search relevance. ReLSum leverages relevance scores as rewards to align the objectives of summarization and ranking, effectively overcoming limitations of prior methods, such as misaligned learning targets. The framework employs a trainable large language model (LLM) to produce summaries, which are then used as input for a cross-encoder ranking model. Experimental results demonstrate significant improvements in offline metrics, including recall and NDCG, as well as online user engagement metrics. ReLSum provides a scalable and efficient solution for enhancing search relevance in large-scale e-commerce systems. 

---
# Spatiotemporally Consistent Indoor Lighting Estimation with Diffusion Priors 

**Authors**: Mutian Tong, Rundi Wu, Changxi Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.08384)  

**Abstract**: Indoor lighting estimation from a single image or video remains a challenge due to its highly ill-posed nature, especially when the lighting condition of the scene varies spatially and temporally. We propose a method that estimates from an input video a continuous light field describing the spatiotemporally varying lighting of the scene. We leverage 2D diffusion priors for optimizing such light field represented as a MLP. To enable zero-shot generalization to in-the-wild scenes, we fine-tune a pre-trained image diffusion model to predict lighting at multiple locations by jointly inpainting multiple chrome balls as light probes. We evaluate our method on indoor lighting estimation from a single image or video and show superior performance over compared baselines. Most importantly, we highlight results on spatiotemporally consistent lighting estimation from in-the-wild videos, which is rarely demonstrated in previous works. 

---
# The DNA of nuclear models: How AI predicts nuclear masses 

**Authors**: Kate A. Richardson, Sokratis Trifinopoulos, Mike Williams  

**Link**: [PDF](https://arxiv.org/pdf/2508.08370)  

**Abstract**: Obtaining high-precision predictions of nuclear masses, or equivalently nuclear binding energies, $E_b$, remains an important goal in nuclear-physics research. Recently, many AI-based tools have shown promising results on this task, some achieving precision that surpasses the best physics models. However, the utility of these AI models remains in question given that predictions are only useful where measurements do not exist, which inherently requires extrapolation away from the training (and testing) samples. Since AI models are largely black boxes, the reliability of such an extrapolation is difficult to assess. We present an AI model that not only achieves cutting-edge precision for $E_b$, but does so in an interpretable manner. For example, we find (and explain why) that the most important dimensions of its internal representation form a double helix, where the analog of the hydrogen bonds in DNA here link the number of protons and neutrons found in the most stable nucleus of each isotopic chain. Furthermore, we show that the AI prediction of $E_b$ can be factorized and ordered hierarchically, with the most important terms corresponding to well-known symbolic models (such as the famous liquid drop). Remarkably, the improvement of the AI model over symbolic ones can almost entirely be attributed to an observation made by Jaffe in 1969. The end result is a fully interpretable data-driven model of nuclear masses. 

---
# Processing of synthetic data in AI development for healthcare and the definition of personal data in EU law 

**Authors**: Vibeke Binz Vallevik, Anne Kjersti C. Befring, Severin Elvatun, Jan Franz Nygaard  

**Link**: [PDF](https://arxiv.org/pdf/2508.08353)  

**Abstract**: Artificial intelligence (AI) has the potential to transform healthcare, but it requires access to health data. Synthetic data that is generated through machine learning models trained on real data, offers a way to share data while preserving privacy. However, uncertainties in the practical application of the General Data Protection Regulation (GDPR) create an administrative burden, limiting the benefits of synthetic data. Through a systematic analysis of relevant legal sources and an empirical study, this article explores whether synthetic data should be classified as personal data under the GDPR. The study investigates the residual identification risk through generating synthetic data and simulating inference attacks, challenging common perceptions of technical identification risk. The findings suggest synthetic data is likely anonymous, depending on certain factors, but highlights uncertainties about what constitutes reasonably likely risk. To promote innovation, the study calls for clearer regulations to balance privacy protection with the advancement of AI in healthcare. 

---
# Fuzzy-Pattern Tsetlin Machine 

**Authors**: Artem Hnilov  

**Link**: [PDF](https://arxiv.org/pdf/2508.08350)  

**Abstract**: The "all-or-nothing" clause evaluation strategy is a core mechanism in the Tsetlin Machine (TM) family of algorithms. In this approach, each clause - a logical pattern composed of binary literals mapped to input data - is disqualified from voting if even a single literal fails. Due to this strict requirement, standard TMs must employ thousands of clauses to achieve competitive accuracy. This paper introduces the Fuzzy-Pattern Tsetlin Machine (FPTM), a novel variant where clause evaluation is fuzzy rather than strict. If some literals in a clause fail, the remaining ones can still contribute to the overall vote with a proportionally reduced score. As a result, each clause effectively consists of sub-patterns that adapt individually to the input, enabling more flexible, efficient, and robust pattern matching. The proposed fuzzy mechanism significantly reduces the required number of clauses, memory footprint, and training time, while simultaneously improving accuracy. On the IMDb dataset, FPTM achieves 90.15% accuracy with only one clause per class, a 50x reduction in clauses and memory over the Coalesced Tsetlin Machine. FPTM trains up to 316x faster (45 seconds vs. 4 hours) and fits within 50 KB, enabling online learning on microcontrollers. Inference throughput reaches 34.5 million predictions/second (51.4 GB/s). On Fashion-MNIST, accuracy reaches 92.18% (2 clauses), 93.19% (20 clauses) and 94.68% (8000 clauses), a ~400x clause reduction compared to the Composite TM's 93.00% (8000 clauses). On the Amazon Sales dataset with 20% noise, FPTM achieves 85.22% accuracy, significantly outperforming the Graph Tsetlin Machine (78.17%) and a Graph Convolutional Neural Network (66.23%). 

---
# Do AI Companies Make Good on Voluntary Commitments to the White House? 

**Authors**: Jennifer Wang, Kayla Huang, Kevin Klyman, Rishi Bommasani  

**Link**: [PDF](https://arxiv.org/pdf/2508.08345)  

**Abstract**: Voluntary commitments are central to international AI governance, as demonstrated by recent voluntary guidelines from the White House to the G7, from Bletchley Park to Seoul. How do major AI companies make good on their commitments? We score companies based on their publicly disclosed behavior by developing a detailed rubric based on their eight voluntary commitments to the White House in 2023. We find significant heterogeneity: while the highest-scoring company (OpenAI) scores a 83% overall on our rubric, the average score across all companies is just 52%. The companies demonstrate systemically poor performance for their commitment to model weight security with an average score of 17%: 11 of the 16 companies receive 0% for this commitment. Our analysis highlights a clear structural shortcoming that future AI governance initiatives should correct: when companies make public commitments, they should proactively disclose how they meet their commitments to provide accountability, and these disclosures should be verifiable. To advance policymaking on corporate AI governance, we provide three directed recommendations that address underspecified commitments, the role of complex AI supply chains, and public transparency that could be applied towards AI governance initiatives worldwide. 

---
# Maximizing GPU Efficiency via Optimal Adapter Caching: An Analytical Approach for Multi-Tenant LLM Serving 

**Authors**: Ferran Agullo, Joan Oliveras, Chen Wang, Alberto Gutierrez-Torre, Olivier Tardieu, Alaa Youssef, Jordi Torres, Josep Ll. Berral  

**Link**: [PDF](https://arxiv.org/pdf/2508.08343)  

**Abstract**: Serving LLM adapters has gained significant attention as an effective approach to adapt general-purpose language models to diverse, task-specific use cases. However, serving a wide range of adapters introduces several and substantial overheads, leading to performance degradation and challenges in optimal placement. To address these challenges, we present an analytical, AI-driven pipeline that accurately determines the optimal allocation of adapters in single-node setups. This allocation maximizes performance, effectively using GPU resources, while preventing request starvation. Crucially, the proposed allocation is given based on current workload patterns. These insights in single-node setups can be leveraged in multi-replica deployments for overall placement, load balancing and server configuration, ultimately enhancing overall performance and improving resource efficiency. Our approach builds on an in-depth analysis of LLM adapter serving, accounting for overheads and performance variability, and includes the development of the first Digital Twin capable of replicating online LLM-adapter serving systems with matching key performance metrics. The experimental results demonstrate that the Digital Twin achieves a SMAPE difference of no more than 5.5% in throughput compared to real results, and the proposed pipeline accurately predicts the optimal placement with minimal latency. 

---
# ImageDDI: Image-enhanced Molecular Motif Sequence Representation for Drug-Drug Interaction Prediction 

**Authors**: Yuqin He, Tengfei Ma, Chaoyi Li, Pengsen Ma, Hongxin Xiang, Jianmin Wang, Yiping Liu, Bosheng Song, Xiangxiang Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2508.08338)  

**Abstract**: To mitigate the potential adverse health effects of simultaneous multi-drug use, including unexpected side effects and interactions, accurately identifying and predicting drug-drug interactions (DDIs) is considered a crucial task in the field of deep learning. Although existing methods have demonstrated promising performance, they suffer from the bottleneck of limited functional motif-based representation learning, as DDIs are fundamentally caused by motif interactions rather than the overall drug structures. In this paper, we propose an Image-enhanced molecular motif sequence representation framework for \textbf{DDI} prediction, called ImageDDI, which represents a pair of drugs from both global and local structures. Specifically, ImageDDI tokenizes molecules into functional motifs. To effectively represent a drug pair, their motifs are combined into a single sequence and embedded using a transformer-based encoder, starting from the local structure representation. By leveraging the associations between drug pairs, ImageDDI further enhances the spatial representation of molecules using global molecular image information (e.g. texture, shadow, color, and planar spatial relationships). To integrate molecular visual information into functional motif sequence, ImageDDI employs Adaptive Feature Fusion, enhancing the generalization of ImageDDI by dynamically adapting the fusion process of feature representations. Experimental results on widely used datasets demonstrate that ImageDDI outperforms state-of-the-art methods. Moreover, extensive experiments show that ImageDDI achieved competitive performance in both 2D and 3D image-enhanced scenarios compared to other models. 

---
# Algorithmic Fairness amid Social Determinants: Reflection, Characterization, and Approach 

**Authors**: Zeyu Tang, Alex John London, Atoosa Kasirzadeh, Sanmi Koyejo, Peter Spirtes, Kun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08337)  

**Abstract**: Social determinants are variables that, while not directly pertaining to any specific individual, capture key aspects of contexts and environments that have direct causal influences on certain attributes of an individual. Previous algorithmic fairness literature has primarily focused on sensitive attributes, often overlooking the role of social determinants. Our paper addresses this gap by introducing formal and quantitative rigor into a space that has been shaped largely by qualitative proposals regarding the use of social determinants. To demonstrate theoretical perspectives and practical applicability, we examine a concrete setting of college admissions, using region as a proxy for social determinants. Our approach leverages a region-based analysis with Gamma distribution parameterization to model how social determinants impact individual outcomes. Despite its simplicity, our method quantitatively recovers findings that resonate with nuanced insights in previous qualitative debates, that are often missed by existing algorithmic fairness approaches. Our findings suggest that mitigation strategies centering solely around sensitive attributes may introduce new structural injustice when addressing existing discrimination. Considering both sensitive attributes and social determinants facilitates a more comprehensive explication of benefits and burdens experienced by individuals from diverse demographic backgrounds as well as contextual environments, which is essential for understanding and achieving fairness effectively and transparently. 

---
# HSA-Net: Hierarchical and Structure-Aware Framework for Efficient and Scalable Molecular Language Modeling 

**Authors**: Zihang Shao, Wentao Lei, Lei Wang, Wencai Ye, Li Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08334)  

**Abstract**: Molecular representation learning, a cornerstone for downstream tasks like molecular captioning and molecular property prediction, heavily relies on Graph Neural Networks (GNN). However, GNN suffers from the over-smoothing problem, where node-level features collapse in deep GNN layers. While existing feature projection methods with cross-attention have been introduced to mitigate this issue, they still perform poorly in deep features. This motivated our exploration of using Mamba as an alternative projector for its ability to handle complex sequences. However, we observe that while Mamba excels at preserving global topological information from deep layers, it neglects fine-grained details in shallow layers. The capabilities of Mamba and cross-attention exhibit a global-local trade-off. To resolve this critical global-local trade-off, we propose Hierarchical and Structure-Aware Network (HSA-Net), a novel framework with two modules that enables a hierarchical feature projection and fusion. Firstly, a Hierarchical Adaptive Projector (HAP) module is introduced to process features from different graph layers. It learns to dynamically switch between a cross-attention projector for shallow layers and a structure-aware Graph-Mamba projector for deep layers, producing high-quality, multi-level features. Secondly, to adaptively merge these multi-level features, we design a Source-Aware Fusion (SAF) module, which flexibly selects fusion experts based on the characteristics of the aggregation features, ensuring a precise and effective final representation fusion. Extensive experiments demonstrate that our HSA-Net framework quantitatively and qualitatively outperforms current state-of-the-art (SOTA) methods. 

---
# Normative Moral Pluralism for AI: A Framework for Deliberation in Complex Moral Contexts 

**Authors**: David-Doron Yaacov  

**Link**: [PDF](https://arxiv.org/pdf/2508.08333)  

**Abstract**: The conceptual framework proposed in this paper centers on the development of a deliberative moral reasoning system - one designed to process complex moral situations by generating, filtering, and weighing normative arguments drawn from diverse ethical perspectives. While the framework is rooted in Machine Ethics, it also makes a substantive contribution to Value Alignment by outlining a system architecture that links structured moral reasoning to action under time constraints. Grounded in normative moral pluralism, this system is not constructed to imitate behavior but is built on reason-sensitive deliberation over structured moral content in a transparent and principled manner. Beyond its role as a deliberative system, it also serves as the conceptual foundation for a novel two-level architecture: functioning as a moral reasoning teacher envisioned to train faster models that support real-time responsiveness without reproducing the full structure of deliberative reasoning. Together, the deliberative and intuitive components are designed to enable both deep reflection and responsive action. A key design feature is the dual-hybrid structure: a universal layer that defines a moral threshold through top-down and bottom-up learning, and a local layer that learns to weigh competing considerations in context while integrating culturally specific normative content, so long as it remains within the universal threshold. By extending the notion of moral complexity to include not only conflicting beliefs but also multifactorial dilemmas, multiple stakeholders, and the integration of non-moral considerations, the framework aims to support morally grounded decision-making in realistic, high-stakes contexts. 

---
# Energy-Aware Code Generation with LLMs: Benchmarking Small vs. Large Language Models for Sustainable AI Programming 

**Authors**: Humza Ashraf, Syed Muhammad Danish, Aris Leivadeas, Yazan Otoum, Zeeshan Sattar  

**Link**: [PDF](https://arxiv.org/pdf/2508.08332)  

**Abstract**: Large Language Models (LLMs) are widely used for code generation. However, commercial models like ChatGPT require significant computing power, which leads to high energy use and carbon emissions. This has raised concerns about their environmental impact. In this study, we evaluate open-source Small Language Models (SLMs) trained explicitly for code generation and compare their performance and energy efficiency against large LLMs and efficient human-written Python code. The goal is to investigate whether SLMs can match the performance of LLMs on certain types of programming problems while producing more energy-efficient code. We evaluate 150 coding problems from LeetCode, evenly distributed across three difficulty levels: easy, medium, and hard. Our comparison includes three small open-source models, StableCode-3B, StarCoderBase-3B, and Qwen2.5-Coder-3B-Instruct, and two large commercial models, GPT-4.0 and DeepSeek-Reasoner. The generated code is evaluated using four key metrics: run-time, memory usage, energy consumption, and correctness. We use human-written solutions as a baseline to assess the quality and efficiency of the model-generated code. Results indicate that LLMs achieve the highest correctness across all difficulty levels, but SLMs are often more energy-efficient when their outputs are correct. In over 52% of the evaluated problems, SLMs consumed the same or less energy than LLMs. 

---
# Algorithmic Collusion of Pricing and Advertising on E-commerce Platforms 

**Authors**: Hangcheng Zhao, Ron Berman  

**Link**: [PDF](https://arxiv.org/pdf/2508.08325)  

**Abstract**: Online sellers have been adopting AI learning algorithms to automatically make product pricing and advertising decisions on e-commerce platforms. When sellers compete using such algorithms, one concern is that of tacit collusion - the algorithms learn to coordinate on higher than competitive. We empirically investigate whether these concerns are valid when sellers make pricing and advertising decisions together, i.e., two-dimensional decisions. Our empirical strategy is to analyze competition with multi-agent reinforcement learning, which we calibrate to a large-scale dataset collected from this http URL products. Our first contribution is to find conditions under which learning algorithms can facilitate win-win-win outcomes that are beneficial for consumers, sellers, and even the platform, when consumers have high search costs. In these cases the algorithms learn to coordinate on prices that are lower than competitive prices. The intuition is that the algorithms learn to coordinate on lower advertising bids, which lower advertising costs, leading to lower prices. Our second contribution is an analysis of a large-scale, high-frequency keyword-product dataset for more than 2 million products on this http URL. Our estimates of consumer search costs show a wide range of costs for different product keywords. We generate an algorithm usage and find a negative interaction between the estimated consumer search costs and the algorithm usage index, providing empirical evidence of beneficial collusion. Finally, we analyze the platform's strategic response. We find that reserve price adjustments will not increase profits for the platform, but commission adjustments will. Our analyses help alleviate some worries about the potentially harmful effects of competing learning algorithms, and can help sellers, platforms and policymakers to decide on whether to adopt or regulate such algorithms. 

---
# Context Engineering for Multi-Agent LLM Code Assistants Using Elicit, NotebookLM, ChatGPT, and Claude Code 

**Authors**: Muhammad Haseeb  

**Link**: [PDF](https://arxiv.org/pdf/2508.08322)  

**Abstract**: Large Language Models (LLMs) have shown promise in automating code generation and software engineering tasks, yet they often struggle with complex, multi-file projects due to context limitations and knowledge gaps. We propose a novel context engineering workflow that combines multiple AI components: an Intent Translator (GPT-5) for clarifying user requirements, an Elicit-powered semantic literature retrieval for injecting domain knowledge, NotebookLM-based document synthesis for contextual understanding, and a Claude Code multi-agent system for code generation and validation. Our integrated approach leverages intent clarification, retrieval-augmented generation, and specialized sub-agents orchestrated via Claude's agent framework. We demonstrate that this method significantly improves the accuracy and reliability of code assistants in real-world repositories, yielding higher single-shot success rates and better adherence to project context than baseline single-agent approaches. Qualitative results on a large this http URL codebase show the multi-agent system effectively plans, edits, and tests complex features with minimal human intervention. We compare our system with recent frameworks like CodePlan, MASAI, and HyperAgent, highlighting how targeted context injection and agent role decomposition lead to state-of-the-art performance. Finally, we discuss the implications for deploying LLM-based coding assistants in production, along with lessons learned on context management and future research directions. 

---
# Between Fear and Desire, the Monster Artificial Intelligence (AI): Analysis through the Lenses of Monster Theory 

**Authors**: Ahmed Tlili  

**Link**: [PDF](https://arxiv.org/pdf/2508.08318)  

**Abstract**: With the increasing adoption of Artificial Intelligence (AI) in all fields and daily activities, a heated debate is found about the advantages and challenges of AI and the need for navigating the concerns associated with AI to make the best of it. To contribute to this literature and the ongoing debate related to it, this study draws on the Monster theory to explain the conflicting representation of AI. It suggests that studying monsters in popular culture can provide an in-depth understanding of AI and its monstrous effects. Specifically, this study aims to discuss AI perception and development through the seven theses of Monster theory. The obtained results revealed that, just like monsters, AI is complex in nature, and it should not be studied as a separate entity but rather within a given society or culture. Similarly, readers may perceive and interpret AI differently, just as readers may interpret monsters differently. The relationship between AI and monsters, as depicted in this study, does not seem to be as odd as it might be at first. 

---
# Evaluation of State-of-the-Art Deep Learning Techniques for Plant Disease and Pest Detection 

**Authors**: Saptarshi Banerjee, Tausif Mallick, Amlan Chakroborty, Himadri Nath Saha, Nityananda T. Takur  

**Link**: [PDF](https://arxiv.org/pdf/2508.08317)  

**Abstract**: Addressing plant diseases and pests is critical for enhancing crop production and preventing economic losses. Recent advances in artificial intelligence (AI), machine learning (ML), and deep learning (DL) have significantly improved the precision and efficiency of detection methods, surpassing the limitations of manual identification. This study reviews modern computer-based techniques for detecting plant diseases and pests from images, including recent AI developments. The methodologies are organized into five categories: hyperspectral imaging, non-visualization techniques, visualization approaches, modified deep learning architectures, and transformer models. This structured taxonomy provides researchers with detailed, actionable insights for selecting advanced state-of-the-art detection methods. A comprehensive survey of recent work and comparative studies demonstrates the consistent superiority of modern AI-based approaches, which often outperform older image analysis methods in speed and accuracy. In particular, vision transformers such as the Hierarchical Vision Transformer (HvT) have shown accuracy exceeding 99.3% in plant disease detection, outperforming architectures like MobileNetV3. The study concludes by discussing system design challenges, proposing solutions, and outlining promising directions for future research. 

---
# EU Digital Regulation and Guatemala: AI, 5G, and Cybersecurity 

**Authors**: Victor Lopez Juarez  

**Link**: [PDF](https://arxiv.org/pdf/2508.08315)  

**Abstract**: The paper examines how EU rules in AI, 5G, and cybersecurity operate as transnational governance and shape policy in Guatemala. It outlines the AI Act's risk approach, the 5G Action Plan and Security Toolbox, and the cybersecurity regime built on ENISA, NIS2, the Cybersecurity Act, and the Cyber Resilience Act. It traces extraterritorial channels such as the Brussels effect, private standards, supply chain clauses, and data transfer controls. Guatemala specific impacts include SME compliance costs, procurement limits, environmental trade-offs in rollout, rights risks, and capacity gaps. The paper maps current national measures and proposes five guardrails: digital constitutionalism, green IT duties, third country impact assessment, standards co-design, and recognition of regulatory diversity. 

---
# Assessing the Quality of AI-Generated Exams: A Large-Scale Field Study 

**Authors**: Calvin Isley, Joshua Gilbert, Evangelos Kassos, Michaela Kocher, Allen Nie, Emma Brunskill, Ben Domingue, Jake Hofman, Joscha Legewie, Teddy Svoronos, Charlotte Tuminelli, Sharad Goel  

**Link**: [PDF](https://arxiv.org/pdf/2508.08314)  

**Abstract**: While large language models (LLMs) challenge conventional methods of teaching and learning, they present an exciting opportunity to improve efficiency and scale high-quality instruction. One promising application is the generation of customized exams, tailored to specific course content. There has been significant recent excitement on automatically generating questions using artificial intelligence, but also comparatively little work evaluating the psychometric quality of these items in real-world educational settings. Filling this gap is an important step toward understanding generative AI's role in effective test design. In this study, we introduce and evaluate an iterative refinement strategy for question generation, repeatedly producing, assessing, and improving questions through cycles of LLM-generated critique and revision. We evaluate the quality of these AI-generated questions in a large-scale field study involving 91 classes -- covering computer science, mathematics, chemistry, and more -- in dozens of colleges across the United States, comprising nearly 1700 students. Our analysis, based on item response theory (IRT), suggests that for students in our sample the AI-generated questions performed comparably to expert-created questions designed for standardized exams. Our results illustrate the power of AI to make high-quality assessments more readily available, benefiting both teachers and students. 

---
# Constrained PSLQ Search for Machin-like Identities Achieving Record-Low Lehmer Measures 

**Authors**: Nick Craig-Wood  

**Link**: [PDF](https://arxiv.org/pdf/2508.08307)  

**Abstract**: Machin-like arctangent relations are classical tools for computing $\pi$, with efficiency quantified by the Lehmer measure ($\lambda$). We present a framework for discovering low-measure relations by coupling the PSLQ integer-relation algorithm with number-theoretic filters derived from the algebraic structure of Gaussian integers, making large scale search tractable. Our search yields new 5 and 6 term relations with record-low Lehmer measures ($\lambda=1.4572, \lambda=1.3291$). We also demonstrate how discovered relations can serve as a basis for generating new, longer formulae through algorithmic extensions. This combined approach of a constrained PSLQ search and algorithmic extension provides a robust method for future explorations. 

---
# Channel-Wise MLPs Improve the Generalization of Recurrent Convolutional Networks 

**Authors**: Nathan Breslow  

**Link**: [PDF](https://arxiv.org/pdf/2508.08298)  

**Abstract**: We investigate the impact of channel-wise mixing via multi-layer perceptrons (MLPs) on the generalization capabilities of recurrent convolutional networks. Specifically, we compare two architectures: DARC (Depth Aware Recurrent Convolution), which employs a simple recurrent convolutional structure, and DAMP (Depth Aware Multi-layer Perceptron), which extends DARC with a gated MLP for channel mixing. Using the Re-ARC benchmark, we find that DAMP significantly outperforms DARC in both in-distribution and out-of-distribution generalization under exact-match grading criteria. These results suggest that explicit channel mixing through MLPs enables recurrent convolutional networks to learn more robust and generalizable computational patterns. Our findings have implications for neural program synthesis and highlight the potential of DAMP as a target architecture for hypernetwork approaches. 

---
# Putnam-AXIOM: A Functional and Static Benchmark 

**Authors**: Aryan Gulati, Brando Miranda, Eric Chen, Emily Xia, Kai Fronsdal, Bruno Dumont, Elyas Obbad, Sanmi Koyejo  

**Link**: [PDF](https://arxiv.org/pdf/2508.08292)  

**Abstract**: Current mathematical reasoning benchmarks for large language models (LLMs) are approaching saturation, with some achieving > 90% accuracy, and are increasingly compromised by training-set contamination. We introduce Putnam-AXIOM, a benchmark of 522 university-level competition problems drawn from the prestigious William Lowell Putnam Mathematical Competition, and Putnam-AXIOM Variation, an unseen companion set of 100 functional variants generated by programmatically perturbing variables and constants. The variation protocol produces an unlimited stream of equally difficult, unseen instances -- yielding a contamination-resilient test bed. On the Original set, OpenAI's o1-preview -- the strongest evaluated model -- scores 41.9%, but its accuracy drops by 19.6% (46.8% relative decrease) on the paired Variations. The remaining eighteen models show the same downward trend, ten of them with non-overlapping 95% confidence intervals. These gaps suggest memorization and highlight the necessity of dynamic benchmarks. We complement "boxed" accuracy with Teacher-Forced Accuracy (TFA), a lightweight metric that directly scores reasoning traces and automates natural language proof evaluations. Putnam-AXIOM therefore provides a rigorous, contamination-resilient evaluation framework for assessing advanced mathematical reasoning of LLMs. Data and evaluation code are publicly available at this https URL. 

---
# Understanding Transformers through the Lens of Pavlovian Conditioning 

**Authors**: Mu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.08289)  

**Abstract**: Transformer architectures have revolutionized artificial intelligence (AI) through their attention mechanisms, yet the computational principles underlying their success remain opaque. We present a novel theoretical framework that reinterprets the core computation of attention as Pavlovian conditioning. Our model finds a direct mathematical analogue in linear attention, which simplifies the analysis of the underlying associative process. We demonstrate that attention's queries, keys, and values can be mapped to the three elements of classical conditioning: test stimuli that probe associations, conditional stimuli (CS) that serve as retrieval cues, and unconditional stimuli (US) that contain response information. Through this lens, we suggest that each attention operation constructs a transient associative memory via a Hebbian rule, where CS-US pairs form dynamic associations that test stimuli can later retrieve. Our framework yields several theoretical insights grounded in this linearized model: (1) a capacity theorem showing that attention heads can store O($\sqrt{d_k}$) associations before interference degrades retrieval; (2) an error propagation analysis revealing fundamental architectural trade-offs of balancing model depth, width, and head redundancy to maintain reliability; and (3) an understanding of how biologically plausible learning rules could enhance transformer architectures. By establishing this deep connection, we suggest that the success of modern AI may stem not from architectural novelty alone, but from implementing computational principles that biology optimized over millions of years of evolution. 

---
# Sacred or Synthetic? Evaluating LLM Reliability and Abstention for Religious Questions 

**Authors**: Farah Atif, Nursultan Askarbekuly, Kareem Darwish, Monojit Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2508.08287)  

**Abstract**: Despite the increasing usage of Large Language Models (LLMs) in answering questions in a variety of domains, their reliability and accuracy remain unexamined for a plethora of domains including the religious domains. In this paper, we introduce a novel benchmark FiqhQA focused on the LLM generated Islamic rulings explicitly categorized by the four major Sunni schools of thought, in both Arabic and English. Unlike prior work, which either overlooks the distinctions between religious school of thought or fails to evaluate abstention behavior, we assess LLMs not only on their accuracy but also on their ability to recognize when not to answer. Our zero-shot and abstention experiments reveal significant variation across LLMs, languages, and legal schools of thought. While GPT-4o outperforms all other models in accuracy, Gemini and Fanar demonstrate superior abstention behavior critical for minimizing confident incorrect answers. Notably, all models exhibit a performance drop in Arabic, highlighting the limitations in religious reasoning for languages other than English. To the best of our knowledge, this is the first study to benchmark the efficacy of LLMs for fine-grained Islamic school of thought specific ruling generation and to evaluate abstention for Islamic jurisprudence queries. Our findings underscore the need for task-specific evaluation and cautious deployment of LLMs in religious applications. 

---
# The Illusion of Progress: Re-evaluating Hallucination Detection in LLMs 

**Authors**: Denis Janiak, Jakub Binkowski, Albert Sawczyn, Bogdan Gabrys, Ravid Schwartz-Ziv, Tomasz Kajdanowicz  

**Link**: [PDF](https://arxiv.org/pdf/2508.08285)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing, yet their tendency to hallucinate poses serious challenges for reliable deployment. Despite numerous hallucination detection methods, their evaluations often rely on ROUGE, a metric based on lexical overlap that misaligns with human judgments. Through comprehensive human studies, we demonstrate that while ROUGE exhibits high recall, its extremely low precision leads to misleading performance estimates. In fact, several established detection methods show performance drops of up to 45.9\% when assessed using human-aligned metrics like LLM-as-Judge. Moreover, our analysis reveals that simple heuristics based on response length can rival complex detection techniques, exposing a fundamental flaw in current evaluation practices. We argue that adopting semantically aware and robust evaluation frameworks is essential to accurately gauge the true performance of hallucination detection methods, ultimately ensuring the trustworthiness of LLM outputs. 

---
# MinionsLLM: a Task-adaptive Framework For The Training and Control of Multi-Agent Systems Through Natural Language 

**Authors**: Andres Garcia Rincon, Eliseo Ferrante  

**Link**: [PDF](https://arxiv.org/pdf/2508.08283)  

**Abstract**: This paper presents MinionsLLM, a novel framework that integrates Large Language Models (LLMs) with Behavior Trees (BTs) and Formal Grammars to enable natural language control of multi-agent systems within arbitrary, user-defined environments. MinionsLLM provides standardized interfaces for defining environments, agents, and behavioral primitives, and introduces two synthetic dataset generation methods (Method A and Method B) to fine-tune LLMs for improved syntactic validity and semantic task relevance. We validate our approach using Google's Gemma 3 model family at three parameter scales (1B, 4B, and 12B) and demonstrate substantial gains: Method B increases syntactic validity to 92.6% and achieves a mean task performance improvement of 33% over baseline. Notably, our experiments show that smaller models benefit most from fine-tuning, suggesting promising directions for deploying compact, locally hosted LLMs in resource-constrained multi-agent control scenarios. The framework and all resources are released open-source to support reproducibility and future research. 

---
# Multi-grained spatial-temporal feature complementarity for accurate online cellular traffic prediction 

**Authors**: Ningning Fu, Shengheng Liu, Weiliang Xie, Yongming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08281)  

**Abstract**: Knowledge discovered from telecom data can facilitate proactive understanding of network dynamics and user behaviors, which in turn empowers service providers to optimize cellular traffic scheduling and resource allocation. Nevertheless, the telecom industry still heavily relies on manual expert intervention. Existing studies have been focused on exhaustively explore the spatial-temporal correlations. However, they often overlook the underlying characteristics of cellular traffic, which are shaped by the sporadic and bursty nature of telecom services. Additionally, concept drift creates substantial obstacles to maintaining satisfactory accuracy in continuous cellular forecasting tasks. To resolve these problems, we put forward an online cellular traffic prediction method grounded in Multi-Grained Spatial-Temporal feature Complementarity (MGSTC). The proposed method is devised to achieve high-precision predictions in practical continuous forecasting scenarios. Concretely, MGSTC segments historical data into chunks and employs the coarse-grained temporal attention to offer a trend reference for the prediction horizon. Subsequently, fine-grained spatial attention is utilized to capture detailed correlations among network elements, which enables localized refinement of the established trend. The complementarity of these multi-grained spatial-temporal features facilitates the efficient transmission of valuable information. To accommodate continuous forecasting needs, we implement an online learning strategy that can detect concept drift in real-time and promptly switch to the appropriate parameter update stage. Experiments carried out on four real-world datasets demonstrate that MGSTC outperforms eleven state-of-the-art baselines consistently. 

---
# MoSSDA: A Semi-Supervised Domain Adaptation Framework for Multivariate Time-Series Classification using Momentum Encoder 

**Authors**: Seonyoung Kim, Dongil Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.08280)  

**Abstract**: Deep learning has emerged as the most promising approach in various fields; however, when the distributions of training and test data are different (domain shift), the performance of deep learning models can degrade. Semi-supervised domain adaptation (SSDA) is a major approach for addressing this issue, assuming that a fully labeled training set (source domain) is available, but the test set (target domain) provides labels only for a small subset. In this study, we propose a novel two-step momentum encoder-utilized SSDA framework, MoSSDA, for multivariate time-series classification. Time series data are highly sensitive to noise, and sequential dependencies cause domain shifts resulting in critical performance degradation. To obtain a robust, domain-invariant and class-discriminative representation, MoSSDA employs a domain-invariant encoder to learn features from both source and target domains. Subsequently, the learned features are fed to a mixup-enhanced positive contrastive module consisting of an online momentum encoder. The final classifier is trained with learned features that exhibit consistency and discriminability with limited labeled target domain data, without data augmentation. We applied a two-stage process by separating the gradient flow between the encoders and the classifier to obtain rich and complex representations. Through extensive experiments on six diverse datasets, MoSSDA achieved state-of-the-art performance for three different backbones and various unlabeled ratios in the target domain data. The Ablation study confirms that each module, including two-stage learning, is effective in improving the performance. Our code is available at this https URL 

---
# XFMNet: Decoding Cross-Site and Nonstationary Water Patterns via Stepwise Multimodal Fusion for Long-Term Water Quality Forecasting 

**Authors**: Ziqi Wang, Hailiang Zhao, Cheng Bao, Wenzhuo Qian, Yuhao Yang, Xueqiang Sun, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2508.08279)  

**Abstract**: Long-term time-series forecasting is critical for environmental monitoring, yet water quality prediction remains challenging due to complex periodicity, nonstationarity, and abrupt fluctuations induced by ecological factors. These challenges are further amplified in multi-site scenarios that require simultaneous modeling of temporal and spatial dynamics. To tackle this, we introduce XFMNet, a stepwise multimodal fusion network that integrates remote sensing precipitation imagery to provide spatial and environmental context in river networks. XFMNet first aligns temporal resolutions between water quality series and remote sensing inputs via adaptive downsampling, followed by locally adaptive decomposition to disentangle trend and cycle components. A cross-attention gated fusion module dynamically integrates temporal patterns with spatial and ecological cues, enhancing robustness to nonstationarity and site-specific anomalies. Through progressive and recursive fusion, XFMNet captures both long-term trends and short-term fluctuations. Extensive experiments on real-world datasets demonstrate substantial improvements over state-of-the-art baselines, highlighting the effectiveness of XFMNet for spatially distributed time series prediction. 

---
# Towards Heterogeneity-Aware and Energy-Efficient Topology Optimization for Decentralized Federated Learning in Edge Environment 

**Authors**: Yuze Liu, Tiehua Zhang, Zhishu Shen, Libing Wu, Shiping Chen, Jiong Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.08278)  

**Abstract**: Federated learning (FL) has emerged as a promising paradigm within edge computing (EC) systems, enabling numerous edge devices to collaboratively train artificial intelligence (AI) models while maintaining data privacy. To overcome the communication bottlenecks associated with centralized parameter servers, decentralized federated learning (DFL), which leverages peer-to-peer (P2P) communication, has been extensively explored in the research community. Although researchers design a variety of DFL approach to ensure model convergence, its iterative learning process inevitably incurs considerable cost along with the growth of model complexity and the number of participants. These costs are largely influenced by the dynamic changes of topology in each training round, particularly its sparsity and connectivity conditions. Furthermore, the inherent resources heterogeneity in the edge environments affects energy efficiency of learning process, while data heterogeneity degrades model performance. These factors pose significant challenges to the design of an effective DFL framework for EC systems. To this end, we propose Hat-DFed, a heterogeneity-aware and coset-effective decentralized federated learning (DFL) framework. In Hat-DFed, the topology construction is formulated as a dual optimization problem, which is then proven to be NP-hard, with the goal of maximizing model performance while minimizing cumulative energy consumption in complex edge environments. To solve this problem, we design a two-phase algorithm that dynamically constructs optimal communication topologies while unbiasedly estimating their impact on both model performance and energy cost. Additionally, the algorithm incorporates an importance-aware model aggregation mechanism to mitigate performance degradation caused by data heterogeneity. 

---
# Evaluating Contrast Localizer for Identifying Causal Unitsin Social & Mathematical Tasks in Language Models 

**Authors**: Yassine Jamaa, Badr AlKhamissi, Satrajit Ghosh, Martin Schrimpf  

**Link**: [PDF](https://arxiv.org/pdf/2508.08276)  

**Abstract**: This work adapts a neuroscientific contrast localizer to pinpoint causally relevant units for Theory of Mind (ToM) and mathematical reasoning tasks in large language models (LLMs) and vision-language models (VLMs). Across 11 LLMs and 5 VLMs ranging in size from 3B to 90B parameters, we localize top-activated units using contrastive stimulus sets and assess their causal role via targeted ablations. We compare the effect of lesioning functionally selected units against low-activation and randomly selected units on downstream accuracy across established ToM and mathematical benchmarks. Contrary to expectations, low-activation units sometimes produced larger performance drops than the highly activated ones, and units derived from the mathematical localizer often impaired ToM performance more than those from the ToM localizer. These findings call into question the causal relevance of contrast-based localizers and highlight the need for broader stimulus sets and more accurately capture task-specific units. 

---
# MLLM-CBench:A Comprehensive Benchmark for Continual Instruction Tuning of Multimodal LLMs with Chain-of-Thought Reasoning Analysis 

**Authors**: Haiyun Guo, ZhiYan Hou, Yu Chen, Jinghan He, Yandu Sun, Yuzhe Zhou, Shujing Guo, Kuan Zhu, Jinqiao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08275)  

**Abstract**: Multimodal Large Language Models (MLLMs) rely on continual instruction tuning to adapt to the evolving demands of real-world applications. However, progress in this area is hindered by the lack of rigorous and systematic benchmarks. To address this gap, we present MLLM-CTBench, a comprehensive evaluation benchmark with three key contributions: (1) Multidimensional Evaluation: We combine final answer accuracy with fine-grained CoT reasoning quality assessment, enabled by a specially trained CoT evaluator; (2) Comprehensive Evaluation of Algorithms and Training Paradigms: We benchmark eight continual learning algorithms across four major categories and systematically compare reinforcement learning with supervised fine-tuning paradigms; (3) Carefully Curated Tasks: We select and organize 16 datasets from existing work, covering six challenging domains. Our key findings include: (i) Models with stronger general capabilities exhibit greater robustness to forgetting during continual learning; (ii) Reasoning chains degrade more slowly than final answers, supporting the hierarchical forgetting hypothesis; (iii) The effectiveness of continual learning algorithms is highly dependent on both model capability and task order; (iv) In reinforcement learning settings, incorporating KL-divergence constraints helps maintain policy stability and plays a crucial role in mitigating forgetting. MLLM-CTBench establishes a rigorous standard for continual instruction tuning of MLLMs and offers practical guidance for algorithm design and evaluation. 

---
# Distilling Knowledge from Large Language Models: A Concept Bottleneck Model for Hate and Counter Speech Recognition 

**Authors**: Roberto Labadie-Tamayo, Djordje Slijepčević, Xihui Chen, Adrian Jaques Böck, Andreas Babic, Liz Freimann, Christiane Atzmüller Matthias Zeppelzauer  

**Link**: [PDF](https://arxiv.org/pdf/2508.08274)  

**Abstract**: The rapid increase in hate speech on social media has exposed an unprecedented impact on society, making automated methods for detecting such content important. Unlike prior black-box models, we propose a novel transparent method for automated hate and counter speech recognition, i.e., "Speech Concept Bottleneck Model" (SCBM), using adjectives as human-interpretable bottleneck concepts. SCBM leverages large language models (LLMs) to map input texts to an abstract adjective-based representation, which is then sent to a light-weight classifier for downstream tasks. Across five benchmark datasets spanning multiple languages and platforms (e.g., Twitter, Reddit, YouTube), SCBM achieves an average macro-F1 score of 0.69 which outperforms the most recently reported results from the literature on four out of five datasets. Aside from high recognition accuracy, SCBM provides a high level of both local and global interpretability. Furthermore, fusing our adjective-based concept representation with transformer embeddings, leads to a 1.8% performance increase on average across all datasets, showing that the proposed representation captures complementary information. Our results demonstrate that adjective-based concept representations can serve as compact, interpretable, and effective encodings for hate and counter speech recognition. With adapted adjectives, our method can also be applied to other NLP tasks. 

---
# Doctor Sun: A Bilingual Multimodal Large Language Model for Biomedical AI 

**Authors**: Dong Xue, Ziyao Shao, Zhaoyang Duan, Fangzhou Liu, Bing Li, Zhongheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.08270)  

**Abstract**: Large multimodal models (LMMs) have demonstrated significant potential in providing innovative solutions for various biomedical tasks, including pathology analysis, radiology report generation, and biomedical assistance. However, the existing multimodal biomedical AI is typically based on foundation LLMs, thus hindering the understanding of intricate medical concepts with limited medical training data. Moreover, recent LLaVA-induced medical LMMs struggle to effectively capture the intricate relationship between the texts and the images. Therefore, we introduce Doctor Sun, a large multimodal generative model specialized in medicine, developed to encode, integrate, and interpret diverse biomedical data modalities such as text and images. In particular, Doctor Sun integrates a pre-trained vision encoder with a medical LLM and conducts two-stage training on various medical datasets, focusing on feature alignment and instruction tuning. Moreover, we release SunMed-VL, a wide-range bilingual medical multimodal dataset, along with all associated models, code, and resources, to freely support the advancement of biomedical multimodal research. 

---
# emg2tendon: From sEMG Signals to Tendon Control in Musculoskeletal Hands 

**Authors**: Sagar Verma  

**Link**: [PDF](https://arxiv.org/pdf/2508.08269)  

**Abstract**: Tendon-driven robotic hands offer unparalleled dexterity for manipulation tasks, but learning control policies for such systems presents unique challenges. Unlike joint-actuated robotic hands, tendon-driven systems lack a direct one-to-one mapping between motion capture (mocap) data and tendon controls, making the learning process complex and expensive. Additionally, visual tracking methods for real-world applications are prone to occlusions and inaccuracies, further complicating joint tracking. Wrist-wearable surface electromyography (sEMG) sensors present an inexpensive, robust alternative to capture hand motion. However, mapping sEMG signals to tendon control remains a significant challenge despite the availability of EMG-to-pose data sets and regression-based models in the existing literature.
We introduce the first large-scale EMG-to-Tendon Control dataset for robotic hands, extending the emg2pose dataset, which includes recordings from 193 subjects, spanning 370 hours and 29 stages with diverse gestures. This dataset incorporates tendon control signals derived using the MyoSuite MyoHand model, addressing limitations such as invalid poses in prior methods. We provide three baseline regression models to demonstrate emg2tendon utility and propose a novel diffusion-based regression model for predicting tendon control from sEMG recordings. This dataset and modeling framework marks a significant step forward for tendon-driven dexterous robotic manipulation, laying the groundwork for scalable and accurate tendon control in robotic hands. this https URL 

---
# Benchmarking Large Language Models for Geolocating Colonial Virginia Land Grants 

**Authors**: Ryan Mioduski  

**Link**: [PDF](https://arxiv.org/pdf/2508.08266)  

**Abstract**: Virginia's seventeenth- and eighteenth-century land patents survive primarily as narrative metes-and-bounds descriptions, limiting spatial analysis. This study systematically evaluates current-generation large language models (LLMs) in converting these prose abstracts into geographically accurate latitude/longitude coordinates within a focused evaluation context. A digitized corpus of 5,471 Virginia patent abstracts (1695-1732) is released, with 43 rigorously verified test cases serving as an initial, geographically focused benchmark. Six OpenAI models across three architectures (o-series, GPT-4-class, and GPT-3.5) were tested under two paradigms: direct-to-coordinate and tool-augmented chain-of-thought invoking external geocoding APIs. Results were compared with a GIS-analyst baseline, the Stanford NER geoparser, Mordecai-3, and a county-centroid heuristic.
The top single-call model, o3-2025-04-16, achieved a mean error of 23 km (median 14 km), outperforming the median LLM (37.4 km) by 37.5%, the weakest LLM (50.3 km) by 53.5%, and external baselines by 67% (GIS analyst) and 70% (Stanford NER). A five-call ensemble further reduced errors to 19 km (median 12 km) at minimal additional cost (approx. USD 0.20 per grant), outperforming the median LLM by 48.6%. A patentee-name-redaction ablation increased error by about 9%, indicating reliance on textual landmark and adjacency descriptions rather than memorization. The cost-efficient gpt-4o-2024-08-06 model maintained a 28 km mean error at USD 1.09 per 1,000 grants, establishing a strong cost-accuracy benchmark; external geocoding tools offered no measurable benefit in this evaluation.
These findings demonstrate the potential of LLMs for scalable, accurate, and cost-effective historical georeferencing. 

---
# TurQUaz at CheckThat! 2025: Debating Large Language Models for Scientific Web Discourse Detection 

**Authors**: Tarık Saraç, Selin Mergen, Mucahid Kutlu  

**Link**: [PDF](https://arxiv.org/pdf/2508.08265)  

**Abstract**: In this paper, we present our work developed for the scientific web discourse detection task (Task 4a) of CheckThat! 2025. We propose a novel council debate method that simulates structured academic discussions among multiple large language models (LLMs) to identify whether a given tweet contains (i) a scientific claim, (ii) a reference to a scientific study, or (iii) mentions of scientific entities. We explore three debating methods: i) single debate, where two LLMs argue for opposing positions while a third acts as a judge; ii) team debate, in which multiple models collaborate within each side of the debate; and iii) council debate, where multiple expert models deliberate together to reach a consensus, moderated by a chairperson model. We choose council debate as our primary model as it outperforms others in the development test set. Although our proposed method did not rank highly for identifying scientific claims (8th out of 10) or mentions of scientific entities (9th out of 10), it ranked first in detecting references to scientific studies. 

---
# On the Effects of Smoothing Rugged Landscape by Different Toy Problems: A Case Study on UBQP 

**Authors**: Wei Wang, Jialong Shi, Jianyong Sun, Arnaud Liefooghe, Qingfu Zhang, Ye Fan  

**Link**: [PDF](https://arxiv.org/pdf/2407.19676)  

**Abstract**: The hardness of the Unconstrained Binary Quadratic Program (UBQP) problem is due its rugged landscape. Various algorithms have been proposed for UBQP, including the Landscape Smoothing Iterated Local Search (LSILS). Different from other UBQP algorithms, LSILS tries to smooth the rugged landscape by building a convex combination of the original UBQP and a toy UBQP. In this paper, our study further investigates the impact of smoothing rugged landscapes using different toy UBQP problems, including a toy UBQP with matrix ^Q1 (construct by "+/-1"), a toy UBQP with matrix ^Q2 (construct by "+/-i") and a toy UBQP with matrix ^Q3 (construct randomly). We first assess the landscape flatness of the three toy UBQPs. Subsequently, we test the efficiency of LSILS with different toy UBQPs. Results reveal that the toy UBQP with ^Q1 (construct by "+/-1") exhibits the flattest landscape among the three, while the toy UBQP with ^Q3 (construct randomly) presents the most non-flat landscape. Notably, LSILS using the toy UBQP with ^Q2 (construct by "+/-i") emerges as the most effective, while ^Q3 (construct randomly) has the poorest result. These findings contribute to a detailed understanding of landscape smoothing techniques in optimizing UBQP. 

---
# A New Parallel Cooperative Landscape Smoothing Algorithm and Its Applications on TSP and UBQP 

**Authors**: Wei Wang, Jialong Shi, Jianyong Sun, Arnaud Liefooghe, Qingfu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2401.03237)  

**Abstract**: Combinatorial optimization problem (COP) is difficult to solve because of the massive number of local optimal solutions in his solution space. Various methods have been put forward to smooth the solution space of COPs, including homotopic convex (HC) transformation for the traveling salesman problem (TSP). This paper extends the HC transformation approach to unconstrained binary quadratic programming (UBQP) by proposing a method to construct a unimodal toy UBQP of any size. We theoretically prove the unimodality of the constructed toy UBQP. After that, we apply this unimodal toy UBQP to smooth the original UBQP by using the HC transformation framework and empirically verify the smoothing effects. Subsequently, we introduce an iterative algorithmic framework incorporating HC transformation, referred as landscape smoothing iterated local search (LSILS). Our experimental analyses, conducted on various UBQP instances show the effectiveness of LSILS. Furthermore, this paper proposes a parallel cooperative variant of LSILS, denoted as PC-LSILS and apply it to both the UBQP and the TSP. Our experimental findings highlight that PC-LSILS improves the smoothing performance of the HC transformation, and further improves the overall performance of the algorithm. 

---
