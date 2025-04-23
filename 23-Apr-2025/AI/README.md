# Approximate matrices of systems of max-min fuzzy relational equations 

**Authors**: Ismaïl Baaj  

**Link**: [PDF](https://arxiv.org/pdf/2504.16042)  

**Abstract**: In this article, we address the inconsistency of a system of max-min fuzzy relational equations by minimally modifying the matrix governing the system in order to achieve consistency. Our method yields consistent systems that approximate the original inconsistent system in the following sense: the right-hand side vector of each consistent system is that of the inconsistent system, and the coefficients of the matrix governing each consistent system are obtained by modifying, exactly and minimally, the entries of the original matrix that must be corrected to achieve consistency, while leaving all other entries unchanged.
To obtain a consistent system that closely approximates the considered inconsistent system, we study the distance (in terms of a norm among $L_1$, $L_2$ or $L_\infty$) between the matrix of the inconsistent system and the set formed by the matrices of consistent systems that use the same right-hand side vector as the inconsistent system. We show that our method allows us to directly compute matrices of consistent systems that use the same right-hand side vector as the inconsistent system whose distance in terms of $L_\infty$ norm to the matrix of the inconsistent system is minimal (the computational costs are higher when using $L_1$ norm or $L_2$ norm). We also give an explicit analytical formula for computing this minimal $L_\infty$ distance. Finally, we translate our results for systems of min-max fuzzy relational equations and present some potential applications. 

---
# Impact of Noise on LLM-Models Performance in Abstraction and Reasoning Corpus (ARC) Tasks with Model Temperature Considerations 

**Authors**: Nikhil Khandalkar, Pavan Yadav, Krishna Shinde, Lokesh B. Ramegowda, Rajarshi Das  

**Link**: [PDF](https://arxiv.org/pdf/2504.15903)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have generated growing interest in their structured reasoning capabilities, particularly in tasks involving abstraction and pattern recognition. The Abstraction and Reasoning Corpus (ARC) benchmark plays a crucial role in evaluating these capabilities by testing how well AI models generalize to novel problems. While GPT-4o demonstrates strong performance by solving all ARC tasks under zero-noise conditions, other models like DeepSeek R1 and LLaMA 3.2 fail to solve any, suggesting limitations in their ability to reason beyond simple pattern matching. To explore this gap, we systematically evaluate these models across different noise levels and temperature settings. Our results reveal that the introduction of noise consistently impairs model performance, regardless of architecture. This decline highlights a shared vulnerability: current LLMs, despite showing signs of abstract reasoning, remain highly sensitive to input perturbations. Such fragility raises concerns about their real-world applicability, where noise and uncertainty are common. By comparing how different model architectures respond to these challenges, we offer insights into the structural weaknesses of modern LLMs in reasoning tasks. This work underscores the need for developing more robust and adaptable AI systems capable of handling the ambiguity and variability inherent in real-world scenarios. Our findings aim to guide future research toward enhancing model generalization, robustness, and alignment with human-like cognitive flexibility. 

---
# CARE: Compatibility-Aware Incentive Mechanisms for Federated Learning with Budgeted Requesters 

**Authors**: Xiang Liu, Hau Chan, Minming Li, Xianlong Zeng, Chenchen Fu, Weiwei Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15847)  

**Abstract**: Federated learning (FL) is a promising approach that allows requesters (\eg, servers) to obtain local training models from workers (e.g., clients). Since workers are typically unwilling to provide training services/models freely and voluntarily, many incentive mechanisms in FL are designed to incentivize participation by offering monetary rewards from requesters. However, existing studies neglect two crucial aspects of real-world FL scenarios. First, workers can possess inherent incompatibility characteristics (e.g., communication channels and data sources), which can lead to degradation of FL efficiency (e.g., low communication efficiency and poor model generalization). Second, the requesters are budgeted, which limits the amount of workers they can hire for their tasks. In this paper, we investigate the scenario in FL where multiple budgeted requesters seek training services from incompatible workers with private training costs. We consider two settings: the cooperative budget setting where requesters cooperate to pool their budgets to improve their overall utility and the non-cooperative budget setting where each requester optimizes their utility within their own budgets. To address efficiency degradation caused by worker incompatibility, we develop novel compatibility-aware incentive mechanisms, CARE-CO and CARE-NO, for both settings to elicit true private costs and determine workers to hire for requesters and their rewards while satisfying requester budget constraints. Our mechanisms guarantee individual rationality, truthfulness, budget feasibility, and approximation performance. We conduct extensive experiments using real-world datasets to show that the proposed mechanisms significantly outperform existing baselines. 

---
# Generative AI for Research Data Processing: Lessons Learnt From Three Use Cases 

**Authors**: Modhurita Mitra, Martine G. de Vos, Nicola Cortinovis, Dawa Ometto  

**Link**: [PDF](https://arxiv.org/pdf/2504.15829)  

**Abstract**: There has been enormous interest in generative AI since ChatGPT was launched in 2022. However, there are concerns about the accuracy and consistency of the outputs of generative AI. We have carried out an exploratory study on the application of this new technology in research data processing. We identified tasks for which rule-based or traditional machine learning approaches were difficult to apply, and then performed these tasks using generative AI.
We demonstrate the feasibility of using the generative AI model Claude 3 Opus in three research projects involving complex data processing tasks:
1) Information extraction: We extract plant species names from historical seedlists (catalogues of seeds) published by botanical gardens.
2) Natural language understanding: We extract certain data points (name of drug, name of health indication, relative effectiveness, cost-effectiveness, etc.) from documents published by Health Technology Assessment organisations in the EU.
3) Text classification: We assign industry codes to projects on the crowdfunding website Kickstarter.
We share the lessons we learnt from these use cases: How to determine if generative AI is an appropriate tool for a given data processing task, and if so, how to maximise the accuracy and consistency of the results obtained. 

---
# Crisp complexity of fuzzy classifiers 

**Authors**: Raquel Fernandez-Peralta, Javier Fumanal-Idocin, Javier Andreu-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2504.15791)  

**Abstract**: Rule-based systems are a very popular form of explainable AI, particularly in the fuzzy community, where fuzzy rules are widely used for control and classification problems. However, fuzzy rule-based classifiers struggle to reach bigger traction outside of fuzzy venues, because users sometimes do not know about fuzzy and because fuzzy partitions are not so easy to interpret in some situations. In this work, we propose a methodology to reduce fuzzy rule-based classifiers to crisp rule-based classifiers. We study different possible crisp descriptions and implement an algorithm to obtain them. Also, we analyze the complexity of the resulting crisp classifiers. We believe that our results can help both fuzzy and non-fuzzy practitioners understand better the way in which fuzzy rule bases partition the feature space and how easily one system can be translated to another and vice versa. Our complexity metric can also help to choose between different fuzzy classifiers based on what the equivalent crisp partitions look like. 

---
# WALL-E 2.0: World Alignment by NeuroSymbolic Learning improves World Model-based LLM Agents 

**Authors**: Siyu Zhou, Tianyi Zhou, Yijun Yang, Guodong Long, Deheng Ye, Jing Jiang, Chengqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15785)  

**Abstract**: Can we build accurate world models out of large language models (LLMs)? How can world models benefit LLM agents? The gap between the prior knowledge of LLMs and the specified environment's dynamics usually bottlenecks LLMs' performance as world models. To bridge the gap, we propose a training-free "world alignment" that learns an environment's symbolic knowledge complementary to LLMs. The symbolic knowledge covers action rules, knowledge graphs, and scene graphs, which are extracted by LLMs from exploration trajectories and encoded into executable codes to regulate LLM agents' policies. We further propose an RL-free, model-based agent "WALL-E 2.0" through the model-predictive control (MPC) framework. Unlike classical MPC requiring costly optimization on the fly, we adopt an LLM agent as an efficient look-ahead optimizer of future steps' actions by interacting with the neurosymbolic world model. While the LLM agent's strong heuristics make it an efficient planner in MPC, the quality of its planned actions is also secured by the accurate predictions of the aligned world model. They together considerably improve learning efficiency in a new environment. On open-world challenges in Mars (Minecraft like) and ALFWorld (embodied indoor environments), WALL-E 2.0 significantly outperforms existing methods, e.g., surpassing baselines in Mars by 16.1%-51.6% of success rate and by at least 61.7% in score. In ALFWorld, it achieves a new record 98% success rate after only 4 iterations. 

---
# TrustGeoGen: Scalable and Formal-Verified Data Engine for Trustworthy Multi-modal Geometric Problem Solving 

**Authors**: Daocheng Fu, Zijun Chen, Renqiu Xia, Qi Liu, Yuan Feng, Hongbin Zhou, Renrui Zhang, Shiyang Feng, Peng Gao, Junchi Yan, Botian Shi, Bo Zhang, Yu Qiao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15780)  

**Abstract**: Mathematical geometric problem solving (GPS) often requires effective integration of multimodal information and verifiable logical coherence. Despite the fast development of large language models in general problem solving, it remains unresolved regarding with both methodology and benchmarks, especially given the fact that exiting synthetic GPS benchmarks are often not self-verified and contain noise and self-contradicted information due to the illusion of LLMs. In this paper, we propose a scalable data engine called TrustGeoGen for problem generation, with formal verification to provide a principled benchmark, which we believe lays the foundation for the further development of methods for GPS. The engine synthesizes geometric data through four key innovations: 1) multimodal-aligned generation of diagrams, textual descriptions, and stepwise solutions; 2) formal verification ensuring rule-compliant reasoning paths; 3) a bootstrapping mechanism enabling complexity escalation via recursive state generation and 4) our devised GeoExplore series algorithms simultaneously produce multi-solution variants and self-reflective backtracking traces. By formal logical verification, TrustGeoGen produces GeoTrust-200K dataset with guaranteed modality integrity, along with GeoTrust-test testset. Experiments reveal the state-of-the-art models achieve only 49.17\% accuracy on GeoTrust-test, demonstrating its evaluation stringency. Crucially, models trained on GeoTrust achieve OOD generalization on GeoQA, significantly reducing logical inconsistencies relative to pseudo-label annotated by OpenAI-o1. Our code is available at this https URL 

---
# Implementing Rational Choice Functions with LLMs and Measuring their Alignment with User Preferences 

**Authors**: Anna Karnysheva, Christian Drescher, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2504.15719)  

**Abstract**: As large language models (LLMs) become integral to intelligent user interfaces (IUIs), their role as decision-making agents raises critical concerns about alignment. Although extensive research has addressed issues such as factuality, bias, and toxicity, comparatively little attention has been paid to measuring alignment to preferences, i.e., the relative desirability of different alternatives, a concept used in decision making, economics, and social choice theory. However, a reliable decision-making agent makes choices that align well with user preferences.
In this paper, we generalize existing methods that exploit LLMs for ranking alternative outcomes by addressing alignment with the broader and more flexible concept of user preferences, which includes both strict preferences and indifference among alternatives. To this end, we put forward design principles for using LLMs to implement rational choice functions, and provide the necessary tools to measure preference satisfaction. We demonstrate the applicability of our approach through an empirical study in a practical application of an IUI in the automotive domain. 

---
# DianJin-R1: Evaluating and Enhancing Financial Reasoning in Large Language Models 

**Authors**: Jie Zhu, Qian Chen, Huaixia Dou, Junhui Li, Lifan Guo, Feng Chen, Chi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15716)  

**Abstract**: Effective reasoning remains a core challenge for large language models (LLMs) in the financial domain, where tasks often require domain-specific knowledge, precise numerical calculations, and strict adherence to compliance rules. We propose DianJin-R1, a reasoning-enhanced framework designed to address these challenges through reasoning-augmented supervision and reinforcement learning. Central to our approach is DianJin-R1-Data, a high-quality dataset constructed from CFLUE, FinQA, and a proprietary compliance corpus (Chinese Compliance Check, CCC), combining diverse financial reasoning scenarios with verified annotations. Our models, DianJin-R1-7B and DianJin-R1-32B, are fine-tuned from Qwen2.5-7B-Instruct and Qwen2.5-32B-Instruct using a structured format that generates both reasoning steps and final answers. To further refine reasoning quality, we apply Group Relative Policy Optimization (GRPO), a reinforcement learning method that incorporates dual reward signals: one encouraging structured outputs and another rewarding answer correctness. We evaluate our models on five benchmarks: three financial datasets (CFLUE, FinQA, and CCC) and two general reasoning benchmarks (MATH-500 and GPQA-Diamond). Experimental results show that DianJin-R1 models consistently outperform their non-reasoning counterparts, especially on complex financial tasks. Moreover, on the real-world CCC dataset, our single-call reasoning models match or even surpass the performance of multi-agent systems that require significantly more computational cost. These findings demonstrate the effectiveness of DianJin-R1 in enhancing financial reasoning through structured supervision and reward-aligned learning, offering a scalable and practical solution for real-world applications. 

---
# Advancing Embodied Agent Security: From Safety Benchmarks to Input Moderation 

**Authors**: Ning Wang, Zihan Yan, Weiyang Li, Chuan Ma, He Chen, Tao Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15699)  

**Abstract**: Embodied agents exhibit immense potential across a multitude of domains, making the assurance of their behavioral safety a fundamental prerequisite for their widespread deployment. However, existing research predominantly concentrates on the security of general large language models, lacking specialized methodologies for establishing safety benchmarks and input moderation tailored to embodied agents. To bridge this gap, this paper introduces a novel input moderation framework, meticulously designed to safeguard embodied agents. This framework encompasses the entire pipeline, including taxonomy definition, dataset curation, moderator architecture, model training, and rigorous evaluation. Notably, we introduce EAsafetyBench, a meticulously crafted safety benchmark engineered to facilitate both the training and stringent assessment of moderators specifically designed for embodied agents. Furthermore, we propose Pinpoint, an innovative prompt-decoupled input moderation scheme that harnesses a masked attention mechanism to effectively isolate and mitigate the influence of functional prompts on moderation tasks. Extensive experiments conducted on diverse benchmark datasets and models validate the feasibility and efficacy of the proposed approach. The results demonstrate that our methodologies achieve an impressive average detection accuracy of 94.58%, surpassing the performance of existing state-of-the-art techniques, alongside an exceptional moderation processing time of merely 0.002 seconds per instance. 

---
# Exploring Inevitable Waypoints for Unsolvability Explanation in Hybrid Planning Problems 

**Authors**: Mir Md Sajid Sarwar, Rajarshi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2504.15668)  

**Abstract**: Explaining unsolvability of planning problems is of significant research interest in Explainable AI Planning. AI planning literature has reported several research efforts on generating explanations of solutions to planning problems. However, explaining the unsolvability of planning problems remains a largely open and understudied problem. A widely practiced approach to plan generation and automated problem solving, in general, is to decompose tasks into sub-problems that help progressively converge towards the goal. In this paper, we propose to adopt the same philosophy of sub-problem identification as a mechanism for analyzing and explaining unsolvability of planning problems in hybrid systems. In particular, for a given unsolvable planning problem, we propose to identify common waypoints, which are universal obstacles to plan existence; in other words, they appear on every plan from the source to the planning goal. This work envisions such waypoints as sub-problems of the planning problem and the unreachability of any of these waypoints as an explanation for the unsolvability of the original planning problem. We propose a novel method of waypoint identification by casting the problem as an instance of the longest common subsequence problem, a widely popular problem in computer science, typically considered as an illustrative example for the dynamic programming paradigm. Once the waypoints are identified, we perform symbolic reachability analysis on them to identify the earliest unreachable waypoint and report it as the explanation of unsolvability. We present experimental results on unsolvable planning problems in hybrid domains. 

---
# A LoRA-Based Approach to Fine-Tuning LLMs for Educational Guidance in Resource-Constrained Settings 

**Authors**: Md Millat, Md Motiur  

**Link**: [PDF](https://arxiv.org/pdf/2504.15610)  

**Abstract**: The current study describes a cost-effective method for adapting large language models (LLMs) for academic advising with study-abroad contexts in mind and for application in low-resource methods for acculturation. With the Mistral-7B-Instruct model applied with a Low-Rank Adaptation (LoRA) method and a 4-bit quantization method, the model underwent training in two distinct stages related to this study's purpose to enhance domain specificity while maintaining computational efficiency. In Phase 1, the model was conditioned with a synthetic dataset via the Gemini Pro API, and in Phase 2, it was trained with manually curated datasets from the StudyAbroadGPT project to achieve enhanced, contextualized responses. Technical innovations entailed memory-efficient quantization, parameter-efficient adaptation, and continuous training analytics via Weights & Biases. After training, this study demonstrated a reduction in training loss by 52.7%, 92% accuracy in domain-specific recommendations, achieved 95% markdown-based formatting support, and a median run-rate of 100 samples per second on off-the-shelf GPU equipment. These findings support the effective application of instruction-tuned LLMs within educational advisers, especially in low-resource institutional scenarios. Limitations included decreased generalizability and the application of a synthetically generated dataset, but this framework is scalable for adding new multilingual-augmented and real-time academic advising processes. Future directions may include plans for the integration of retrieval-augmented generation, applying dynamic quantization routines, and connecting to real-time academic databases to increase adaptability and accuracy. 

---
# A Multi-Agent Framework for Automated Qinqiang Opera Script Generation Using Large Language Models 

**Authors**: Gengxian Cao, Fengyuan Li, Hong Duan, Ye Yang, Bofeng Wang, Donghe Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.15552)  

**Abstract**: This paper introduces a novel multi-Agent framework that automates the end to end production of Qinqiang opera by integrating Large Language Models , visual generation, and Text to Speech synthesis. Three specialized agents collaborate in sequence: Agent1 uses an LLM to craft coherent, culturally grounded scripts;Agent2 employs visual generation models to render contextually accurate stage scenes; and Agent3 leverages TTS to produce synchronized, emotionally expressive vocal performances. In a case study on Dou E Yuan, the system achieved expert ratings of 3.8 for script fidelity, 3.5 for visual coherence, and 3.8 for speech accuracy-culminating in an overall score of 3.6, a 0.3 point improvement over a Single Agent baseline. Ablation experiments demonstrate that removing Agent2 or Agent3 leads to drops of 0.4 and 0.5 points, respectively, underscoring the value of modular collaboration. This work showcases how AI driven pipelines can streamline and scale the preservation of traditional performing arts, and points toward future enhancements in cross modal alignment, richer emotional nuance, and support for additional opera genres. 

---
# Learning Adaptive Parallel Reasoning with Language Models 

**Authors**: Jiayi Pan, Xiuyu Li, Long Lian, Charlie Snell, Yifei Zhou, Adam Yala, Trevor Darrell, Kurt Keutzer, Alane Suhr  

**Link**: [PDF](https://arxiv.org/pdf/2504.15466)  

**Abstract**: Scaling inference-time computation has substantially improved the reasoning capabilities of language models. However, existing methods have significant limitations: serialized chain-of-thought approaches generate overly long outputs, leading to increased latency and exhausted context windows, while parallel methods such as self-consistency suffer from insufficient coordination, resulting in redundant computations and limited performance gains. To address these shortcomings, we propose Adaptive Parallel Reasoning (APR), a novel reasoning framework that enables language models to orchestrate both serialized and parallel computations end-to-end. APR generalizes existing reasoning methods by enabling adaptive multi-threaded inference using spawn() and join() operations. A key innovation is our end-to-end reinforcement learning strategy, optimizing both parent and child inference threads to enhance task success rate without requiring predefined reasoning structures. Experiments on the Countdown reasoning task demonstrate significant benefits of APR: (1) higher performance within the same context window (83.4% vs. 60.0% at 4k context); (2) superior scalability with increased computation (80.1% vs. 66.6% at 20k total tokens); (3) improved accuracy at equivalent latency (75.2% vs. 57.3% at approximately 5,000ms). APR represents a step towards enabling language models to autonomously optimize their reasoning processes through adaptive allocation of computation. 

---
# Improving Human-AI Coordination through Adversarial Training and Generative Models 

**Authors**: Paresh Chaudhary, Yancheng Liang, Daphne Chen, Simon S. Du, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2504.15457)  

**Abstract**: Being able to cooperate with new people is an important component of many economically valuable AI tasks, from household robotics to autonomous driving. However, generalizing to novel humans requires training on data that captures the diversity of human behaviors. Adversarial training is one avenue for searching for such data and ensuring that agents are robust. However, it is difficult to apply in the cooperative setting because adversarial policies intentionally learn to sabotage the task instead of simulating valid cooperation partners. To address this challenge, we propose a novel strategy for overcoming self-sabotage that combines a pre-trained generative model to simulate valid cooperative agent policies with adversarial training to maximize regret. We call our method GOAT: Generative Online Adversarial Training. In this framework, the GOAT dynamically searches for and generates coordination strategies where the learning policy -- the Cooperator agent -- underperforms. GOAT enables better generalization by exposing the Cooperator to various challenging interaction scenarios. We maintain realistic coordination strategies by updating only the generative model's embedding while keeping its parameters frozen, thus avoiding adversarial exploitation. We evaluate GOAT with real human partners, and the results demonstrate state-of-the-art performance on the Overcooked benchmark, highlighting its effectiveness in generalizing to diverse human behaviors. 

---
# AGI Is Coming... Right After AI Learns to Play Wordle 

**Authors**: Sarath Shekkizhar, Romain Cosentino  

**Link**: [PDF](https://arxiv.org/pdf/2504.15434)  

**Abstract**: This paper investigates multimodal agents, in particular, OpenAI's Computer-User Agent (CUA), trained to control and complete tasks through a standard computer interface, similar to humans. We evaluated the agent's performance on the New York Times Wordle game to elicit model behaviors and identify shortcomings. Our findings revealed a significant discrepancy in the model's ability to recognize colors correctly depending on the context. The model had a $5.36\%$ success rate over several hundred runs across a week of Wordle. Despite the immense enthusiasm surrounding AI agents and their potential to usher in Artificial General Intelligence (AGI), our findings reinforce the fact that even simple tasks present substantial challenges for today's frontier AI models. We conclude with a discussion of the potential underlying causes, implications for future development, and research directions to improve these AI systems. 

---
# KeDiff: Key Similarity-Based KV Cache Eviction for Long-Context LLM Inference in Resource-Constrained Environments 

**Authors**: Junyoung Park, Dalton Jones, Matt Morse, Raghavv Goel, Mingu Lee, Chris Lott  

**Link**: [PDF](https://arxiv.org/pdf/2504.15364)  

**Abstract**: In this work, we demonstrate that distinctive keys during LLM inference tend to have high attention scores. We explore this phenomenon and propose KeyDiff, a training-free KV cache eviction method based on key similarity. This method facilitates the deployment of LLM-based application requiring long input prompts in resource-constrained environments with limited memory and compute budgets. Unlike other KV cache eviction methods, KeyDiff can process arbitrarily long prompts within strict resource constraints and efficiently generate responses. We demonstrate that KeyDiff computes the optimal solution to a KV cache selection problem that maximizes key diversity, providing a theoretical understanding of KeyDiff. Notably,KeyDiff does not rely on attention scores, allowing the use of optimized attention mechanisms like FlashAttention. We demonstrate the effectiveness of KeyDiff across diverse tasks and models, illustrating a performance gap of less than 0.04\% with 8K cache budget ($\sim$ 23\% KV cache reduction) from the non-evicting baseline on the LongBench benchmark for Llama 3.1-8B and Llama 3.2-3B. 

---
# Reliable Classification with Conformal Learning and Interval-Type 2 Fuzzy Sets 

**Authors**: Javier Fumanal-Idocin, Javier Andreu-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2504.15360)  

**Abstract**: Classical machine learning classifiers tend to be overconfident can be unreliable outside of the laboratory benchmarks. Properly assessing the reliability of the output of the model per sample is instrumental for real-life scenarios where these systems are deployed. Because of this, different techniques have been employed to properly quantify the quality of prediction for a given model. These are most commonly Bayesian statistics and, more recently, conformal learning. Given a calibration set, conformal learning can produce outputs that are guaranteed to cover the target class with a desired significance level, and are more reliable than the standard confidence intervals used by Bayesian methods. In this work, we propose to use conformal learning with fuzzy rule-based systems in classification and show some metrics of their performance. Then, we discuss how the use of type 2 fuzzy sets can improve the quality of the output of the system compared to both fuzzy and crisp rules. Finally, we also discuss how the fine-tuning of the system can be adapted to improve the quality of the conformal prediction. 

---
# PolicyEvol-Agent: Evolving Policy via Environment Perception and Self-Awareness with Theory of Mind 

**Authors**: Yajie Yu, Yue Feng  

**Link**: [PDF](https://arxiv.org/pdf/2504.15313)  

**Abstract**: Multi-agents has exhibited significant intelligence in real-word simulations with Large language models (LLMs) due to the capabilities of social cognition and knowledge retrieval. However, existing research on agents equipped with effective cognition chains including reasoning, planning, decision-making and reflecting remains limited, especially in the dynamically interactive scenarios. In addition, unlike human, prompt-based responses face challenges in psychological state perception and empirical calibration during uncertain gaming process, which can inevitably lead to cognition bias. In light of above, we introduce PolicyEvol-Agent, a comprehensive LLM-empowered framework characterized by systematically acquiring intentions of others and adaptively optimizing irrational strategies for continual enhancement. Specifically, PolicyEvol-Agent first obtains reflective expertise patterns and then integrates a range of cognitive operations with Theory of Mind alongside internal and external perspectives. Simulation results, outperforming RL-based models and agent-based methods, demonstrate the superiority of PolicyEvol-Agent for final gaming victory. Moreover, the policy evolution mechanism reveals the effectiveness of dynamic guideline adjustments in both automatic and human evaluation. 

---
# Can Machine Learning Agents Deal with Hard Choices? 

**Authors**: Kangyu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15304)  

**Abstract**: Machine Learning ML agents have been increasingly used in decision-making across a wide range of tasks and environments. These ML agents are typically designed to balance multiple objectives when making choices. Understanding how their decision-making processes align with or diverge from human reasoning is essential. Human agents often encounter hard choices, that is, situations where options are incommensurable; neither option is preferred, yet the agent is not indifferent between them. In such cases, human agents can identify hard choices and resolve them through deliberation. In contrast, current ML agents, due to fundamental limitations in Multi-Objective Optimisation or MOO methods, cannot identify hard choices, let alone resolve them. Neither Scalarised Optimisation nor Pareto Optimisation, the two principal MOO approaches, can capture incommensurability. This limitation generates three distinct alignment problems: the alienness of ML decision-making behaviour from a human perspective; the unreliability of preference-based alignment strategies for hard choices; and the blockage of alignment strategies pursuing multiple objectives. Evaluating two potential technical solutions, I recommend an ensemble solution that appears most promising for enabling ML agents to identify hard choices and mitigate alignment problems. However, no known technique allows ML agents to resolve hard choices through deliberation, as they cannot autonomously change their goals. This underscores the distinctiveness of human agency and urges ML researchers to reconceptualise machine autonomy and develop frameworks and methods that can better address this fundamental gap. 

---
# LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities 

**Authors**: Thomas Schmied, Jörg Bornschein, Jordi Grau-Moya, Markus Wulfmeier, Razvan Pascanu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16078)  

**Abstract**: The success of Large Language Models (LLMs) has sparked interest in various agentic applications. A key hypothesis is that LLMs, leveraging common sense and Chain-of-Thought (CoT) reasoning, can effectively explore and efficiently solve complex domains. However, LLM agents have been found to suffer from sub-optimal exploration and the knowing-doing gap, the inability to effectively act on knowledge present in the model. In this work, we systematically study why LLMs perform sub-optimally in decision-making scenarios. In particular, we closely examine three prevalent failure modes: greediness, frequency bias, and the knowing-doing gap. We propose mitigation of these shortcomings by fine-tuning via Reinforcement Learning (RL) on self-generated CoT rationales. Our experiments across multi-armed bandits, contextual bandits, and Tic-tac-toe, demonstrate that RL fine-tuning enhances the decision-making abilities of LLMs by increasing exploration and narrowing the knowing-doing gap. Finally, we study both classic exploration mechanisms, such as $\epsilon$-greedy, and LLM-specific approaches, such as self-correction and self-consistency, to enable more effective fine-tuning of LLMs for decision-making. 

---
# Describe Anything: Detailed Localized Image and Video Captioning 

**Authors**: Long Lian, Yifan Ding, Yunhao Ge, Sifei Liu, Hanzi Mao, Boyi Li, Marco Pavone, Ming-Yu Liu, Trevor Darrell, Adam Yala, Yin Cui  

**Link**: [PDF](https://arxiv.org/pdf/2504.16072)  

**Abstract**: Generating detailed and accurate descriptions for specific regions in images and videos remains a fundamental challenge for vision-language models. We introduce the Describe Anything Model (DAM), a model designed for detailed localized captioning (DLC). DAM preserves both local details and global context through two key innovations: a focal prompt, which ensures high-resolution encoding of targeted regions, and a localized vision backbone, which integrates precise localization with its broader context. To tackle the scarcity of high-quality DLC data, we propose a Semi-supervised learning (SSL)-based Data Pipeline (DLC-SDP). DLC-SDP starts with existing segmentation datasets and expands to unlabeled web images using SSL. We introduce DLC-Bench, a benchmark designed to evaluate DLC without relying on reference captions. DAM sets new state-of-the-art on 7 benchmarks spanning keyword-level, phrase-level, and detailed multi-sentence localized image and video captioning. 

---
# Vision language models are unreliable at trivial spatial cognition 

**Authors**: Sangeet Khemlani, Tyler Tran, Nathaniel Gyory, Anthony M. Harrison, Wallace E. Lawson, Ravenna Thielstrom, Hunter Thompson, Taaren Singh, J. Gregory Trafton  

**Link**: [PDF](https://arxiv.org/pdf/2504.16061)  

**Abstract**: Vision language models (VLMs) are designed to extract relevant visuospatial information from images. Some research suggests that VLMs can exhibit humanlike scene understanding, while other investigations reveal difficulties in their ability to process relational information. To achieve widespread applicability, VLMs must perform reliably, yielding comparable competence across a wide variety of related tasks. We sought to test how reliable these architectures are at engaging in trivial spatial cognition, e.g., recognizing whether one object is left of another in an uncluttered scene. We developed a benchmark dataset -- TableTest -- whose images depict 3D scenes of objects arranged on a table, and used it to evaluate state-of-the-art VLMs. Results show that performance could be degraded by minor variations of prompts that use logically equivalent descriptions. These analyses suggest limitations in how VLMs may reason about spatial relations in real-world applications. They also reveal novel opportunities for bolstering image caption corpora for more efficient training and testing. 

---
# LongMamba: Enhancing Mamba's Long Context Capabilities via Training-Free Receptive Field Enlargement 

**Authors**: Zhifan Ye, Kejing Xia, Yonggan Fu, Xin Dong, Jihoon Hong, Xiangchi Yuan, Shizhe Diao, Jan Kautz, Pavlo Molchanov, Yingyan Celine Lin  

**Link**: [PDF](https://arxiv.org/pdf/2504.16053)  

**Abstract**: State space models (SSMs) have emerged as an efficient alternative to Transformer models for language modeling, offering linear computational complexity and constant memory usage as context length increases. However, despite their efficiency in handling long contexts, recent studies have shown that SSMs, such as Mamba models, generally underperform compared to Transformers in long-context understanding tasks. To address this significant shortfall and achieve both efficient and accurate long-context understanding, we propose LongMamba, a training-free technique that significantly enhances the long-context capabilities of Mamba models. LongMamba builds on our discovery that the hidden channels in Mamba can be categorized into local and global channels based on their receptive field lengths, with global channels primarily responsible for long-context capability. These global channels can become the key bottleneck as the input context lengthens. Specifically, when input lengths largely exceed the training sequence length, global channels exhibit limitations in adaptively extend their receptive fields, leading to Mamba's poor long-context performance. The key idea of LongMamba is to mitigate the hidden state memory decay in these global channels by preventing the accumulation of unimportant tokens in their memory. This is achieved by first identifying critical tokens in the global channels and then applying token filtering to accumulate only those critical tokens. Through extensive benchmarking across synthetic and real-world long-context scenarios, LongMamba sets a new standard for Mamba's long-context performance, significantly extending its operational range without requiring additional training. Our code is available at this https URL. 

---
# Evaluating Vision Language Models (VLMs) for Radiology: A Comprehensive Analysis 

**Authors**: Frank Li, Hari Trivedi, Bardia Khosravi, Theo Dapamede, Mohammadreza Chavoshi, Abdulhameed Dere, Rohan Satya Isaac, Aawez Mansuri, Janice Newsome, Saptarshi Purkayastha, Judy Gichoya  

**Link**: [PDF](https://arxiv.org/pdf/2504.16047)  

**Abstract**: Foundation models, trained on vast amounts of data using self-supervised techniques, have emerged as a promising frontier for advancing artificial intelligence (AI) applications in medicine. This study evaluates three different vision-language foundation models (RAD-DINO, CheXagent, and BiomedCLIP) on their ability to capture fine-grained imaging features for radiology tasks. The models were assessed across classification, segmentation, and regression tasks for pneumothorax and cardiomegaly on chest radiographs. Self-supervised RAD-DINO consistently excelled in segmentation tasks, while text-supervised CheXagent demonstrated superior classification performance. BiomedCLIP showed inconsistent performance across tasks. A custom segmentation model that integrates global and local features substantially improved performance for all foundation models, particularly for challenging pneumothorax segmentation. The findings highlight that pre-training methodology significantly influences model performance on specific downstream tasks. For fine-grained segmentation tasks, models trained without text supervision performed better, while text-supervised models offered advantages in classification and interpretability. These insights provide guidance for selecting foundation models based on specific clinical applications in radiology. 

---
# Muon Optimizer Accelerates Grokking 

**Authors**: Amund Tveit, Bjørn Remseth, Arve Skogvold  

**Link**: [PDF](https://arxiv.org/pdf/2504.16041)  

**Abstract**: This paper investigates the impact of different optimizers on the grokking phenomenon, where models exhibit delayed generalization. We conducted experiments across seven numerical tasks (primarily modular arithmetic) using a modern Transformer architecture. The experimental configuration systematically varied the optimizer (Muon vs. AdamW) and the softmax activation function (standard softmax, stablemax, and sparsemax) to assess their combined effect on learning dynamics. Our empirical evaluation reveals that the Muon optimizer, characterized by its use of spectral norm constraints and second-order information, significantly accelerates the onset of grokking compared to the widely used AdamW optimizer. Specifically, Muon reduced the mean grokking epoch from 153.09 to 102.89 across all configurations, a statistically significant difference (t = 5.0175, p = 6.33e-08). This suggests that the optimizer choice plays a crucial role in facilitating the transition from memorization to generalization. 

---
# LLMs meet Federated Learning for Scalable and Secure IoT Management 

**Authors**: Yazan Otoum, Arghavan Asad, Amiya Nayak  

**Link**: [PDF](https://arxiv.org/pdf/2504.16032)  

**Abstract**: The rapid expansion of IoT ecosystems introduces severe challenges in scalability, security, and real-time decision-making. Traditional centralized architectures struggle with latency, privacy concerns, and excessive resource consumption, making them unsuitable for modern large-scale IoT deployments. This paper presents a novel Federated Learning-driven Large Language Model (FL-LLM) framework, designed to enhance IoT system intelligence while ensuring data privacy and computational efficiency. The framework integrates Generative IoT (GIoT) models with a Gradient Sensing Federated Strategy (GSFS), dynamically optimizing model updates based on real-time network conditions. By leveraging a hybrid edge-cloud processing architecture, our approach balances intelligence, scalability, and security in distributed IoT environments. Evaluations on the IoT-23 dataset demonstrate that our framework improves model accuracy, reduces response latency, and enhances energy efficiency, outperforming traditional FL techniques (i.e., FedAvg, FedOpt). These findings highlight the potential of integrating LLM-powered federated learning into large-scale IoT ecosystems, paving the way for more secure, scalable, and adaptive IoT management solutions. 

---
# Benchmarking LLM for Code Smells Detection: OpenAI GPT-4.0 vs DeepSeek-V3 

**Authors**: Ahmed R. Sadik, Siddhata Govind  

**Link**: [PDF](https://arxiv.org/pdf/2504.16027)  

**Abstract**: Determining the most effective Large Language Model for code smell detection presents a complex challenge. This study introduces a structured methodology and evaluation matrix to tackle this issue, leveraging a curated dataset of code samples consistently annotated with known smells. The dataset spans four prominent programming languages Java, Python, JavaScript, and C++; allowing for cross language comparison. We benchmark two state of the art LLMs, OpenAI GPT 4.0 and DeepSeek-V3, using precision, recall, and F1 score as evaluation metrics. Our analysis covers three levels of detail: overall performance, category level performance, and individual code smell type performance. Additionally, we explore cost effectiveness by comparing the token based detection approach of GPT 4.0 with the pattern-matching techniques employed by DeepSeek V3. The study also includes a cost analysis relative to traditional static analysis tools such as SonarQube. The findings offer valuable guidance for practitioners in selecting an efficient, cost effective solution for automated code smell detection 

---
# Trends in AI Supercomputers 

**Authors**: Konstantin F. Pilz, James Sanders, Robi Rahman, Lennart Heim  

**Link**: [PDF](https://arxiv.org/pdf/2504.16026)  

**Abstract**: Frontier AI development relies on powerful AI supercomputers, yet analysis of these systems is limited. We create a dataset of 500 AI supercomputers from 2019 to 2025 and analyze key trends in performance, power needs, hardware cost, ownership, and global distribution. We find that the computational performance of AI supercomputers has doubled every nine months, while hardware acquisition cost and power needs both doubled every year. The leading system in March 2025, xAI's Colossus, used 200,000 AI chips, had a hardware cost of \$7B, and required 300 MW of power, as much as 250,000 households. As AI supercomputers evolved from tools for science to industrial machines, companies rapidly expanded their share of total AI supercomputer performance, while the share of governments and academia diminished. Globally, the United States accounts for about 75% of total performance in our dataset, with China in second place at 15%. If the observed trends continue, the leading AI supercomputer in 2030 will achieve $2\times10^{22}$ 16-bit FLOP/s, use two million AI chips, have a hardware cost of \$200 billion, and require 9 GW of power. Our analysis provides visibility into the AI supercomputer landscape, allowing policymakers to assess key AI trends like resource needs, ownership, and national competitiveness. 

---
# Navigating the State of Cognitive Flow: Context-Aware AI Interventions for Effective Reasoning Support 

**Authors**: Dinithi Dissanayake, Suranga Nanayakkara  

**Link**: [PDF](https://arxiv.org/pdf/2504.16021)  

**Abstract**: Flow theory describes an optimal cognitive state where individuals experience deep focus and intrinsic motivation when a task's difficulty aligns with their skill level. In AI-augmented reasoning, interventions that disrupt the state of cognitive flow can hinder rather than enhance decision-making. This paper proposes a context-aware cognitive augmentation framework that adapts interventions based on three key contextual factors: type, timing, and scale. By leveraging multimodal behavioral cues (e.g., gaze behavior, typing hesitation, interaction speed), AI can dynamically adjust cognitive support to maintain or restore flow. We introduce the concept of cognitive flow, an extension of flow theory in AI-augmented reasoning, where interventions are personalized, adaptive, and minimally intrusive. By shifting from static interventions to context-aware augmentation, our approach ensures that AI systems support deep engagement in complex decision-making and reasoning without disrupting cognitive immersion. 

---
# AlphaGrad: Non-Linear Gradient Normalization Optimizer 

**Authors**: Soham Sane  

**Link**: [PDF](https://arxiv.org/pdf/2504.16020)  

**Abstract**: We introduce AlphaGrad, a memory-efficient, conditionally stateless optimizer addressing the memory overhead and hyperparameter complexity of adaptive methods like Adam. AlphaGrad enforces scale invariance via tensor-wise L2 gradient normalization followed by a smooth hyperbolic tangent transformation, $g' = \tanh(\alpha \cdot \tilde{g})$, controlled by a single steepness parameter $\alpha$. Our contributions include: (1) the AlphaGrad algorithm formulation; (2) a formal non-convex convergence analysis guaranteeing stationarity; (3) extensive empirical evaluation on diverse RL benchmarks (DQN, TD3, PPO). Compared to Adam, AlphaGrad demonstrates a highly context-dependent performance profile. While exhibiting instability in off-policy DQN, it provides enhanced training stability with competitive results in TD3 (requiring careful $\alpha$ tuning) and achieves substantially superior performance in on-policy PPO. These results underscore the critical importance of empirical $\alpha$ selection, revealing strong interactions between the optimizer's dynamics and the underlying RL algorithm. AlphaGrad presents a compelling alternative optimizer for memory-constrained scenarios and shows significant promise for on-policy learning regimes where its stability and efficiency advantages can be particularly impactful. 

---
# CAPO: Cost-Aware Prompt Optimization 

**Authors**: Tom Zehle, Moritz Schlager, Timo Heiß, Matthias Feurer  

**Link**: [PDF](https://arxiv.org/pdf/2504.16005)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing by solving a wide range of tasks simply guided by a prompt. Yet their performance is highly sensitive to prompt formulation. While automated prompt optimization addresses this challenge by finding optimal prompts, current methods require a substantial number of LLM calls and input tokens, making prompt optimization expensive. We introduce CAPO (Cost-Aware Prompt Optimization), an algorithm that enhances prompt optimization efficiency by integrating AutoML techniques. CAPO is an evolutionary approach with LLMs as operators, incorporating racing to save evaluations and multi-objective optimization to balance performance with prompt length. It jointly optimizes instructions and few-shot examples while leveraging task descriptions for improved robustness. Our extensive experiments across diverse datasets and LLMs demonstrate that CAPO outperforms state-of-the-art discrete prompt optimization methods in 11/15 cases with improvements up to 21%p. Our algorithm achieves better performances already with smaller budgets, saves evaluations through racing, and decreases average prompt length via a length penalty, making it both cost-efficient and cost-aware. Even without few-shot examples, CAPO outperforms its competitors and generally remains robust to initial prompts. CAPO represents an important step toward making prompt optimization more powerful and accessible by improving cost-efficiency. 

---
# How Private is Your Attention? Bridging Privacy with In-Context Learning 

**Authors**: Soham Bonnerjee, Zhen Wei, Yeon, Anna Asch, Sagnik Nandy, Promit Ghosal  

**Link**: [PDF](https://arxiv.org/pdf/2504.16000)  

**Abstract**: In-context learning (ICL)-the ability of transformer-based models to perform new tasks from examples provided at inference time-has emerged as a hallmark of modern language models. While recent works have investigated the mechanisms underlying ICL, its feasibility under formal privacy constraints remains largely unexplored. In this paper, we propose a differentially private pretraining algorithm for linear attention heads and present the first theoretical analysis of the privacy-accuracy trade-off for ICL in linear regression. Our results characterize the fundamental tension between optimization and privacy-induced noise, formally capturing behaviors observed in private training via iterative methods. Additionally, we show that our method is robust to adversarial perturbations of training prompts, unlike standard ridge regression. All theoretical findings are supported by extensive simulations across diverse settings. 

---
# OPUS-VFL: Incentivizing Optimal Privacy-Utility Tradeoffs in Vertical Federated Learning 

**Authors**: Sindhuja Madabushi, Ahmad Faraz Khan, Haider Ali, Jin-Hee Cho  

**Link**: [PDF](https://arxiv.org/pdf/2504.15995)  

**Abstract**: Vertical Federated Learning (VFL) enables organizations with disjoint feature spaces but shared user bases to collaboratively train models without sharing raw data. However, existing VFL systems face critical limitations: they often lack effective incentive mechanisms, struggle to balance privacy-utility tradeoffs, and fail to accommodate clients with heterogeneous resource capabilities. These challenges hinder meaningful participation, degrade model performance, and limit practical deployment. To address these issues, we propose OPUS-VFL, an Optimal Privacy-Utility tradeoff Strategy for VFL. OPUS-VFL introduces a novel, privacy-aware incentive mechanism that rewards clients based on a principled combination of model contribution, privacy preservation, and resource investment. It employs a lightweight leave-one-out (LOO) strategy to quantify feature importance per client, and integrates an adaptive differential privacy mechanism that enables clients to dynamically calibrate noise levels to optimize their individual utility. Our framework is designed to be scalable, budget-balanced, and robust to inference and poisoning attacks. Extensive experiments on benchmark datasets (MNIST, CIFAR-10, and CIFAR-100) demonstrate that OPUS-VFL significantly outperforms state-of-the-art VFL baselines in both efficiency and robustness. It reduces label inference attack success rates by up to 20%, increases feature inference reconstruction error (MSE) by over 30%, and achieves up to 25% higher incentives for clients that contribute meaningfully while respecting privacy and cost constraints. These results highlight the practicality and innovation of OPUS-VFL as a secure, fair, and performance-driven solution for real-world VFL. 

---
# W-PCA Based Gradient-Free Proxy for Efficient Search of Lightweight Language Models 

**Authors**: Shang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15983)  

**Abstract**: The demand for efficient natural language processing (NLP) systems has led to the development of lightweight language models. Previous work in this area has primarily focused on manual design or training-based neural architecture search (NAS) methods. Recently, zero-shot NAS methods have been proposed for evaluating language models without the need for training. However, prevailing approaches to zero-shot NAS often face challenges such as biased evaluation metrics and computational inefficiencies. In this paper, we introduce weight-weighted PCA (W-PCA), a novel zero-shot NAS method specifically tailored for lightweight language models. Our approach utilizes two evaluation proxies: the parameter count and the number of principal components with cumulative contribution exceeding $\eta$ in the feed-forward neural (FFN) layer. Additionally, by eliminating the need for gradient computations, we optimize the evaluation time, thus enhancing the efficiency of designing and evaluating lightweight language models. We conduct a comparative analysis on the GLUE and SQuAD datasets to evaluate our approach. The results demonstrate that our method significantly reduces training time compared to one-shot NAS methods and achieves higher scores in the testing phase compared to previous state-of-the-art training-based methods. Furthermore, we perform ranking evaluations on a dataset sampled from the FlexiBERT search space. Our approach exhibits superior ranking correlation and further reduces solving time compared to other zero-shot NAS methods that require gradient computation. 

---
# Bug Destiny Prediction in Large Open-Source Software Repositories through Sentiment Analysis and BERT Topic Modeling 

**Authors**: Sophie C. Pope, Andrew Barovic, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15972)  

**Abstract**: This study explores a novel approach to predicting key bug-related outcomes, including the time to resolution, time to fix, and ultimate status of a bug, using data from the Bugzilla Eclipse Project. Specifically, we leverage features available before a bug is resolved to enhance predictive accuracy. Our methodology incorporates sentiment analysis to derive both an emotionality score and a sentiment classification (positive or negative). Additionally, we integrate the bug's priority level and its topic, extracted using a BERTopic model, as features for a Convolutional Neural Network (CNN) and a Multilayer Perceptron (MLP). Our findings indicate that the combination of BERTopic and sentiment analysis can improve certain model performance metrics. Furthermore, we observe that balancing model inputs enhances practical applicability, albeit at the cost of a significant reduction in accuracy in most cases. To address our primary objectives, predicting time-to-resolution, time-to-fix, and bug destiny, we employ both binary classification and exact time value predictions, allowing for a comparative evaluation of their predictive effectiveness. Results demonstrate that sentiment analysis serves as a valuable predictor of a bug's eventual outcome, particularly in determining whether it will be fixed. However, its utility is less pronounced when classifying bugs into more complex or unconventional outcome categories. 

---
# Universal Approximation with Softmax Attention 

**Authors**: Jerry Yao-Chieh Hu, Hude Liu, Hong-Yu Chen, Weimin Wu, Han Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15956)  

**Abstract**: We prove that with linear transformations, both (i) two-layer self-attention and (ii) one-layer self-attention followed by a softmax function are universal approximators for continuous sequence-to-sequence functions on compact domains. Our main technique is a new interpolation-based method for analyzing attention's internal mechanism. This leads to our key insight: self-attention is able to approximate a generalized version of ReLU to arbitrary precision, and hence subsumes many known universal approximators. Building on these, we show that two-layer multi-head attention alone suffices as a sequence-to-sequence universal approximator. In contrast, prior works rely on feed-forward networks to establish universal approximation in Transformers. Furthermore, we extend our techniques to show that, (softmax-)attention-only layers are capable of approximating various statistical models in-context. We believe these techniques hold independent interest. 

---
# FairTranslate: An English-French Dataset for Gender Bias Evaluation in Machine Translation by Overcoming Gender Binarity 

**Authors**: Fanny Jourdan, Yannick Chevalier, Cécile Favre  

**Link**: [PDF](https://arxiv.org/pdf/2504.15941)  

**Abstract**: Large Language Models (LLMs) are increasingly leveraged for translation tasks but often fall short when translating inclusive language -- such as texts containing the singular 'they' pronoun or otherwise reflecting fair linguistic protocols. Because these challenges span both computational and societal domains, it is imperative to critically evaluate how well LLMs handle inclusive translation with a well-founded framework.
This paper presents FairTranslate, a novel, fully human-annotated dataset designed to evaluate non-binary gender biases in machine translation systems from English to French. FairTranslate includes 2418 English-French sentence pairs related to occupations, annotated with rich metadata such as the stereotypical alignment of the occupation, grammatical gender indicator ambiguity, and the ground-truth gender label (male, female, or inclusive).
We evaluate four leading LLMs (Gemma2-2B, Mistral-7B, Llama3.1-8B, Llama3.3-70B) on this dataset under different prompting procedures. Our results reveal substantial biases in gender representation across LLMs, highlighting persistent challenges in achieving equitable outcomes in machine translation. These findings underscore the need for focused strategies and interventions aimed at ensuring fair and inclusive language usage in LLM-based translation systems.
We make the FairTranslate dataset publicly available on Hugging Face, and disclose the code for all experiments on GitHub. 

---
# Meta-Entity Driven Triplet Mining for Aligning Medical Vision-Language Models 

**Authors**: Saban Ozturk, Melih B. Yilmaz, Muti Kara, M. Talat Yavuz, Aykut Koç, Tolga Çukur  

**Link**: [PDF](https://arxiv.org/pdf/2504.15929)  

**Abstract**: Diagnostic imaging relies on interpreting both images and radiology reports, but the growing data volumes place significant pressure on medical experts, yielding increased errors and workflow backlogs. Medical vision-language models (med-VLMs) have emerged as a powerful framework to efficiently process multimodal imaging data, particularly in chest X-ray (CXR) evaluations, albeit their performance hinges on how well image and text representations are aligned. Existing alignment methods, predominantly based on contrastive learning, prioritize separation between disease classes over segregation of fine-grained pathology attributes like location, size or severity, leading to suboptimal representations. Here, we propose MedTrim (Meta-entity-driven Triplet mining), a novel method that enhances image-text alignment through multimodal triplet learning synergistically guided by disease class as well as adjectival and directional pathology descriptors. Unlike common alignment methods that separate broad disease classes, MedTrim leverages structured meta-entity information to preserve subtle but clinically significant intra-class variations. For this purpose, we first introduce an ontology-based entity recognition module that extracts pathology-specific meta-entities from CXR reports, as annotations on pathology attributes are rare in public datasets. For refined sample selection in triplet mining, we then introduce a novel score function that captures an aggregate measure of inter-sample similarity based on disease classes and adjectival/directional descriptors. Lastly, we introduce a multimodal triplet alignment objective for explicit within- and cross-modal alignment between samples sharing detailed pathology characteristics. Our demonstrations indicate that MedTrim improves performance in downstream retrieval and classification tasks compared to state-of-the-art alignment methods. 

---
# A Clinician-Friendly Platform for Ophthalmic Image Analysis Without Technical Barriers 

**Authors**: Meng Wang, Tian Lin, Qingshan Hou, Aidi Lin, Jingcheng Wang, Qingsheng Peng, Truong X. Nguyen, Danqi Fang, Ke Zou, Ting Xu, Cancan Xue, Ten Cheer Quek, Qinkai Yu, Minxin Liu, Hui Zhou, Zixuan Xiao, Guiqin He, Huiyu Liang, Tingkun Shi, Man Chen, Linna Liu, Yuanyuan Peng, Lianyu Wang, Qiuming Hu, Junhong Chen, Zhenhua Zhang, Cheng Chen, Yitian Zhao, Dianbo Liu, Jianhua Wu, Xinjian Chen, Changqing Zhang, Triet Thanh Nguyen, Yanda Meng, Yalin Zheng, Yih Chung Tham, Carol Y. Cheung, Huazhu Fu, Haoyu Chen, Ching-Yu Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2504.15928)  

**Abstract**: Artificial intelligence (AI) shows remarkable potential in medical imaging diagnostics, but current models typically require retraining when deployed across different clinical centers, limiting their widespread adoption. We introduce GlobeReady, a clinician-friendly AI platform that enables ocular disease diagnosis without retraining/fine-tuning or technical expertise. GlobeReady achieves high accuracy across imaging modalities: 93.9-98.5% for an 11-category fundus photo dataset and 87.2-92.7% for a 15-category OCT dataset. Through training-free local feature augmentation, it addresses domain shifts across centers and populations, reaching an average accuracy of 88.9% across five centers in China, 86.3% in Vietnam, and 90.2% in the UK. The built-in confidence-quantifiable diagnostic approach further boosted accuracy to 94.9-99.4% (fundus) and 88.2-96.2% (OCT), while identifying out-of-distribution cases at 86.3% (49 CFP categories) and 90.6% (13 OCT categories). Clinicians from multiple countries rated GlobeReady highly (average 4.6 out of 5) for its usability and clinical relevance. These results demonstrate GlobeReady's robust, scalable diagnostic capability and potential to support ophthalmic care without technical barriers. 

---
# New Recipe for Semi-supervised Community Detection: Clique Annealing under Crystallization Kinetics 

**Authors**: Ling Cheng, Jiashu Pu, Ruicheng Liang, Qian Shao, Hezhe Qiao, Feida Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15927)  

**Abstract**: Semi-supervised community detection methods are widely used for identifying specific communities due to the label scarcity. Existing semi-supervised community detection methods typically involve two learning stages learning in both initial identification and subsequent adjustment, which often starts from an unreasonable community core candidate. Moreover, these methods encounter scalability issues because they depend on reinforcement learning and generative adversarial networks, leading to higher computational costs and restricting the selection of candidates. To address these limitations, we draw a parallel between crystallization kinetics and community detection to integrate the spontaneity of the annealing process into community detection. Specifically, we liken community detection to identifying a crystal subgrain (core) that expands into a complete grain (community) through a process similar to annealing. Based on this finding, we propose CLique ANNealing (CLANN), which applies kinetics concepts to community detection by integrating these principles into the optimization process to strengthen the consistency of the community core. Subsequently, a learning-free Transitive Annealer was employed to refine the first-stage candidates by merging neighboring cliques and repositioning the community core, enabling a spontaneous growth process that enhances scalability. Extensive experiments on \textbf{43} different network settings demonstrate that CLANN outperforms state-of-the-art methods across multiple real-world datasets, showcasing its exceptional efficacy and efficiency in community detection. 

---
# Achieving Distributive Justice in Federated Learning via Uncertainty Quantification 

**Authors**: Alycia Carey, Xintao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15924)  

**Abstract**: Client-level fairness metrics for federated learning are used to ensure that all clients in a federation either: a) have similar final performance on their local data distributions (i.e., client parity), or b) obtain final performance on their local data distributions relative to their contribution to the federated learning process (i.e., contribution fairness). While a handful of works that propose either client-parity or contribution-based fairness metrics ground their definitions and decisions in social theories of equality -- such as distributive justice -- most works arbitrarily choose what notion of fairness to align with which makes it difficult for practitioners to choose which fairness metric aligns best with their fairness ethics. In this work, we propose UDJ-FL (Uncertainty-based Distributive Justice for Federated Learning), a flexible federated learning framework that can achieve multiple distributive justice-based client-level fairness metrics. Namely, by utilizing techniques inspired by fair resource allocation, in conjunction with performing aleatoric uncertainty-based client weighing, our UDJ-FL framework is able to achieve egalitarian, utilitarian, Rawls' difference principle, or desert-based client-level fairness. We empirically show the ability of UDJ-FL to achieve all four defined distributive justice-based client-level fairness metrics in addition to providing fairness equivalent to (or surpassing) other popular fair federated learning works. Further, we provide justification for why aleatoric uncertainty weighing is necessary to the construction of our UDJ-FL framework as well as derive theoretical guarantees for the generalization bounds of UDJ-FL. Our code is publicly available at this https URL. 

---
# Ask2Loc: Learning to Locate Instructional Visual Answers by Asking Questions 

**Authors**: Chang Zong, Bin Li, Shoujun Zhou, Jian Wan, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15918)  

**Abstract**: Locating specific segments within an instructional video is an efficient way to acquire guiding knowledge. Generally, the task of obtaining video segments for both verbal explanations and visual demonstrations is known as visual answer localization (VAL). However, users often need multiple interactions to obtain answers that align with their expectations when using the system. During these interactions, humans deepen their understanding of the video content by asking themselves questions, thereby accurately identifying the location. Therefore, we propose a new task, named In-VAL, to simulate the multiple interactions between humans and videos in the procedure of obtaining visual answers. The In-VAL task requires interactively addressing several semantic gap issues, including 1) the ambiguity of user intent in the input questions, 2) the incompleteness of language in video subtitles, and 3) the fragmentation of content in video segments. To address these issues, we propose Ask2Loc, a framework for resolving In-VAL by asking questions. It includes three key modules: 1) a chatting module to refine initial questions and uncover clear intentions, 2) a rewriting module to generate fluent language and create complete descriptions, and 3) a searching module to broaden local context and provide integrated content. We conduct extensive experiments on three reconstructed In-VAL datasets. Compared to traditional end-to-end and two-stage methods, our proposed Ask2Loc can improve performance by up to 14.91 (mIoU) on the In-VAL task. Our code and datasets can be accessed at this https URL. 

---
# Automated Bug Report Prioritization in Large Open-Source Projects 

**Authors**: Riley Pierson, Armin Moin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15912)  

**Abstract**: Large open-source projects receive a large number of issues (known as bugs), including software defect (i.e., bug) reports and new feature requests from their user and developer communities at a fast rate. The often limited project resources do not allow them to deal with all issues. Instead, they have to prioritize them according to the project's priorities and the issues' severities. In this paper, we propose a novel approach to automated bug prioritization based on the natural language text of the bug reports that are stored in the open bug repositories of the issue-tracking systems. We conduct topic modeling using a variant of LDA called TopicMiner-MTM and text classification with the BERT large language model to achieve a higher performance level compared to the state-of-the-art. Experimental results using an existing reference dataset containing 85,156 bug reports of the Eclipse Platform project indicate that we outperform existing approaches in terms of Accuracy, Precision, Recall, and F1-measure of the bug report priority prediction. 

---
# GraphEdge: Dynamic Graph Partition and Task Scheduling for GNNs Computing in Edge Network 

**Authors**: Wenjing Xiao, Chenglong Shi, Miaojiang Chen, Zhiquan Liu, Min Chen, H. Herbert Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.15905)  

**Abstract**: With the exponential growth of Internet of Things (IoT) devices, edge computing (EC) is gradually playing an important role in providing cost-effective services. However, existing approaches struggle to perform well in graph-structured scenarios where user data is correlated, such as traffic flow prediction and social relationship recommender systems. In particular, graph neural network (GNN)-based approaches lead to expensive server communication cost. To address this problem, we propose GraphEdge, an efficient GNN-based EC architecture. It considers the EC system of GNN tasks, where there are associations between users and it needs to take into account the task data of its neighbors when processing the tasks of a user. Specifically, the architecture first perceives the user topology and represents their data associations as a graph layout at each time step. Then the graph layout is optimized by calling our proposed hierarchical traversal graph cut algorithm (HiCut), which cuts the graph layout into multiple weakly associated subgraphs based on the aggregation characteristics of GNN, and the communication cost between different subgraphs during GNN inference is minimized. Finally, based on the optimized graph layout, our proposed deep reinforcement learning (DRL) based graph offloading algorithm (DRLGO) is executed to obtain the optimal offloading strategy for the tasks of users, the offloading strategy is subgraph-based, it tries to offload user tasks in a subgraph to the same edge server as possible while minimizing the task processing time and energy consumption of the EC system. Experimental results show the good effectiveness and dynamic adaptation of our proposed architecture and it also performs well even in dynamic scenarios. 

---
# Dynamic Early Exit in Reasoning Models 

**Authors**: Chenxu Yang, Qingyi Si, Yongjie Duan, Zheliang Zhu, Chenyu Zhu, Zheng Lin, Li Cao, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15895)  

**Abstract**: Recent advances in large reasoning language models (LRLMs) rely on test-time scaling, which extends long chain-of-thought (CoT) generation to solve complex tasks. However, overthinking in long CoT not only slows down the efficiency of problem solving, but also risks accuracy loss due to the extremely detailed or redundant reasoning steps. We propose a simple yet effective method that allows LLMs to self-truncate CoT sequences by early exit during generation. Instead of relying on fixed heuristics, the proposed method monitors model behavior at potential reasoning transition points (e.g.,"Wait" tokens) and dynamically terminates the next reasoning chain's generation when the model exhibits high confidence in a trial answer. Our method requires no additional training and can be seamlessly integrated into existing o1-like reasoning LLMs. Experiments on multiple reasoning benchmarks MATH-500, AMC 2023, GPQA Diamond and AIME 2024 show that the proposed method is consistently effective on deepseek-series reasoning LLMs, reducing the length of CoT sequences by an average of 31% to 43% while improving accuracy by 1.7% to 5.7%. 

---
# Supporting Data-Frame Dynamics in AI-assisted Decision Making 

**Authors**: Chengbo Zheng, Tim Miller, Alina Bialkowski, H Peter Soyer, Monika Janda  

**Link**: [PDF](https://arxiv.org/pdf/2504.15894)  

**Abstract**: High stakes decision-making often requires a continuous interplay between evolving evidence and shifting hypotheses, a dynamic that is not well supported by current AI decision support systems. In this paper, we introduce a mixed-initiative framework for AI assisted decision making that is grounded in the data-frame theory of sensemaking and the evaluative AI paradigm. Our approach enables both humans and AI to collaboratively construct, validate, and adapt hypotheses. We demonstrate our framework with an AI-assisted skin cancer diagnosis prototype that leverages a concept bottleneck model to facilitate interpretable interactions and dynamic updates to diagnostic hypotheses. 

---
# Integrating Non-Linear Radon Transformation for Diabetic Retinopathy Grading 

**Authors**: Farida Mohsen, Samir Belhaouari, Zubair Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.15883)  

**Abstract**: Diabetic retinopathy is a serious ocular complication that poses a significant threat to patients' vision and overall health. Early detection and accurate grading are essential to prevent vision loss. Current automatic grading methods rely heavily on deep learning applied to retinal fundus images, but the complex, irregular patterns of lesions in these images, which vary in shape and distribution, make it difficult to capture subtle changes. This study introduces RadFuse, a multi-representation deep learning framework that integrates non-linear RadEx-transformed sinogram images with traditional fundus images to enhance diabetic retinopathy detection and grading. Our RadEx transformation, an optimized non-linear extension of the Radon transform, generates sinogram representations to capture complex retinal lesion patterns. By leveraging both spatial and transformed domain information, RadFuse enriches the feature set available to deep learning models, improving the differentiation of severity levels. We conducted extensive experiments on two benchmark datasets, APTOS-2019 and DDR, using three convolutional neural networks (CNNs): ResNeXt-50, MobileNetV2, and VGG19. RadFuse showed significant improvements over fundus-image-only models across all three CNN architectures and outperformed state-of-the-art methods on both datasets. For severity grading across five stages, RadFuse achieved a quadratic weighted kappa of 93.24%, an accuracy of 87.07%, and an F1-score of 87.17%. In binary classification between healthy and diabetic retinopathy cases, the method reached an accuracy of 99.09%, precision of 98.58%, and recall of 99.6%, surpassing previously established models. These results demonstrate RadFuse's capacity to capture complex non-linear features, advancing diabetic retinopathy classification and promoting the integration of advanced mathematical transforms in medical image analysis. 

---
# Bidirectional Task-Motion Planning Based on Hierarchical Reinforcement Learning for Strategic Confrontation 

**Authors**: Qizhen Wu Lei Chen, Kexin Liu, Jinhu Lü  

**Link**: [PDF](https://arxiv.org/pdf/2504.15876)  

**Abstract**: In swarm robotics, confrontation scenarios, including strategic confrontations, require efficient decision-making that integrates discrete commands and continuous actions. Traditional task and motion planning methods separate decision-making into two layers, but their unidirectional structure fails to capture the interdependence between these layers, limiting adaptability in dynamic environments. Here, we propose a novel bidirectional approach based on hierarchical reinforcement learning, enabling dynamic interaction between the layers. This method effectively maps commands to task allocation and actions to path planning, while leveraging cross-training techniques to enhance learning across the hierarchical framework. Furthermore, we introduce a trajectory prediction model that bridges abstract task representations with actionable planning goals. In our experiments, it achieves over 80\% in confrontation win rate and under 0.01 seconds in decision time, outperforming existing approaches. Demonstrations through large-scale tests and real-world robot experiments further emphasize the generalization capabilities and practical applicability of our method. 

---
# MedNNS: Supernet-based Medical Task-Adaptive Neural Network Search 

**Authors**: Lotfi Abdelkrim Mecharbat, Ibrahim Elmakky, Martin Takac, Mohammed Yaqub  

**Link**: [PDF](https://arxiv.org/pdf/2504.15865)  

**Abstract**: Deep learning (DL) has achieved remarkable progress in the field of medical imaging. However, adapting DL models to medical tasks remains a significant challenge, primarily due to two key factors: (1) architecture selection, as different tasks necessitate specialized model designs, and (2) weight initialization, which directly impacts the convergence speed and final performance of the models. Although transfer learning from ImageNet is a widely adopted strategy, its effectiveness is constrained by the substantial differences between natural and medical images. To address these challenges, we introduce Medical Neural Network Search (MedNNS), the first Neural Network Search framework for medical imaging applications. MedNNS jointly optimizes architecture selection and weight initialization by constructing a meta-space that encodes datasets and models based on how well they perform together. We build this space using a Supernetwork-based approach, expanding the model zoo size by 51x times over previous state-of-the-art (SOTA) methods. Moreover, we introduce rank loss and Fréchet Inception Distance (FID) loss into the construction of the space to capture inter-model and inter-dataset relationships, thereby achieving more accurate alignment in the meta-space. Experimental results across multiple datasets demonstrate that MedNNS significantly outperforms both ImageNet pre-trained DL models and SOTA Neural Architecture Search (NAS) methods, achieving an average accuracy improvement of 1.7% across datasets while converging substantially faster. The code and the processed meta-space is available at this https URL. 

---
# DualOptim: Enhancing Efficacy and Stability in Machine Unlearning with Dual Optimizers 

**Authors**: Xuyang Zhong, Haochen Luo, Chen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15827)  

**Abstract**: Existing machine unlearning (MU) approaches exhibit significant sensitivity to hyperparameters, requiring meticulous tuning that limits practical deployment. In this work, we first empirically demonstrate the instability and suboptimal performance of existing popular MU methods when deployed in different scenarios. To address this issue, we propose Dual Optimizer (DualOptim), which incorporates adaptive learning rate and decoupled momentum factors. Empirical and theoretical evidence demonstrates that DualOptim contributes to effective and stable unlearning. Through extensive experiments, we show that DualOptim can significantly boost MU efficacy and stability across diverse tasks, including image classification, image generation, and large language models, making it a versatile approach to empower existing MU algorithms. 

---
# Human-Imperceptible Physical Adversarial Attack for NIR Face Recognition Models 

**Authors**: Songyan Xie, Jinghang Wen, Encheng Su, Qiucheng Yu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15823)  

**Abstract**: Near-infrared (NIR) face recognition systems, which can operate effectively in low-light conditions or in the presence of makeup, exhibit vulnerabilities when subjected to physical adversarial attacks. To further demonstrate the potential risks in real-world applications, we design a novel, stealthy, and practical adversarial patch to attack NIR face recognition systems in a black-box setting. We achieved this by utilizing human-imperceptible infrared-absorbing ink to generate multiple patches with digitally optimized shapes and positions for infrared images. To address the optimization mismatch between digital and real-world NIR imaging, we develop a light reflection model for human skin to minimize pixel-level discrepancies by simulating NIR light reflection.
Compared to state-of-the-art (SOTA) physical attacks on NIR face recognition systems, the experimental results show that our method improves the attack success rate in both digital and physical domains, particularly maintaining effectiveness across various face postures. Notably, the proposed approach outperforms SOTA methods, achieving an average attack success rate of 82.46% in the physical domain across different models, compared to 64.18% for existing methods. The artifact is available at this https URL. 

---
# Fusing Reward and Dueling Feedback in Stochastic Bandits 

**Authors**: Xuchuang Wang, Qirun Zeng, Jinhang Zuo, Xutong Liu, Mohammad Hajiesmaili, John C.S. Lui, Adam Wierman  

**Link**: [PDF](https://arxiv.org/pdf/2504.15812)  

**Abstract**: This paper investigates the fusion of absolute (reward) and relative (dueling) feedback in stochastic bandits, where both feedback types are gathered in each decision round. We derive a regret lower bound, demonstrating that an efficient algorithm may incur only the smaller among the reward and dueling-based regret for each individual arm. We propose two fusion approaches: (1) a simple elimination fusion algorithm that leverages both feedback types to explore all arms and unifies collected information by sharing a common candidate arm set, and (2) a decomposition fusion algorithm that selects the more effective feedback to explore the corresponding arms and randomly assigns one feedback type for exploration and the other for exploitation in each round. The elimination fusion experiences a suboptimal multiplicative term of the number of arms in regret due to the intrinsic suboptimality of dueling elimination. In contrast, the decomposition fusion achieves regret matching the lower bound up to a constant under a common assumption. Extensive experiments confirm the efficacy of our algorithms and theoretical results. 

---
# DAE-KAN: A Kolmogorov-Arnold Network Model for High-Index Differential-Algebraic Equations 

**Authors**: Kai Luo, Juan Tang, Mingchao Cai, Xiaoqing Zeng, Manqi Xie, Ming Yan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15806)  

**Abstract**: Kolmogorov-Arnold Networks (KANs) have emerged as a promising alternative to Multi-Layer Perceptrons (MLPs) due to their superior function-fitting abilities in data-driven modeling. In this paper, we propose a novel framework, DAE-KAN, for solving high-index differential-algebraic equations (DAEs) by integrating KANs with Physics-Informed Neural Networks (PINNs). This framework not only preserves the ability of traditional PINNs to model complex systems governed by physical laws but also enhances their performance by leveraging the function-fitting strengths of KANs. Numerical experiments demonstrate that for DAE systems ranging from index-1 to index-3, DAE-KAN reduces the absolute errors of both differential and algebraic variables by 1 to 2 orders of magnitude compared to traditional PINNs. To assess the effectiveness of this approach, we analyze the drift-off error and find that both PINNs and DAE-KAN outperform classical numerical methods in controlling this phenomenon. Our results highlight the potential of neural network methods, particularly DAE-KAN, in solving high-index DAEs with substantial computational accuracy and generalization, offering a promising solution for challenging partial differential-algebraic equations. 

---
# Insights from Verification: Training a Verilog Generation LLM with Reinforcement Learning with Testbench Feedback 

**Authors**: Ning Wang, Bingkun Yao, Jie Zhou, Yuchen Hu, Xi Wang, Nan Guan, Zhe Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15804)  

**Abstract**: Large language models (LLMs) have shown strong performance in Verilog generation from natural language description. However, ensuring the functional correctness of the generated code remains a significant challenge. This paper introduces a method that integrates verification insights from testbench into the training of Verilog generation LLMs, aligning the training with the fundamental goal of hardware design: functional correctness. The main obstacle in using LLMs for Verilog code generation is the lack of sufficient functional verification data, particularly testbenches paired with design specifications and code. To address this problem, we introduce an automatic testbench generation pipeline that decomposes the process and uses feedback from the Verilog compiler simulator (VCS) to reduce hallucination and ensure correctness. We then use the testbench to evaluate the generated codes and collect them for further training, where verification insights are introduced. Our method applies reinforcement learning (RL), specifically direct preference optimization (DPO), to align Verilog code generation with functional correctness by training preference pairs based on testbench outcomes. In evaluations on VerilogEval-Machine, VerilogEval-Human, RTLLM v1.1, RTLLM v2, and VerilogEval v2, our approach consistently outperforms state-of-the-art baselines in generating functionally correct Verilog code. We open source all training code, data, and models at this https URL. 

---
# A closer look at how large language models trust humans: patterns and biases 

**Authors**: Valeria Lerman, Yaniv Dover  

**Link**: [PDF](https://arxiv.org/pdf/2504.15801)  

**Abstract**: As large language models (LLMs) and LLM-based agents increasingly interact with humans in decision-making contexts, understanding the trust dynamics between humans and AI agents becomes a central concern. While considerable literature studies how humans trust AI agents, it is much less understood how LLM-based agents develop effective trust in humans. LLM-based agents likely rely on some sort of implicit effective trust in trust-related contexts (e.g., evaluating individual loan applications) to assist and affect decision making. Using established behavioral theories, we develop an approach that studies whether LLMs trust depends on the three major trustworthiness dimensions: competence, benevolence and integrity of the human subject. We also study how demographic variables affect effective trust. Across 43,200 simulated experiments, for five popular language models, across five different scenarios we find that LLM trust development shows an overall similarity to human trust development. We find that in most, but not all cases, LLM trust is strongly predicted by trustworthiness, and in some cases also biased by age, religion and gender, especially in financial scenarios. This is particularly true for scenarios common in the literature and for newer models. While the overall patterns align with human-like mechanisms of effective trust formation, different models exhibit variation in how they estimate trust; in some cases, trustworthiness and demographic factors are weak predictors of effective trust. These findings call for a better understanding of AI-to-human trust dynamics and monitoring of biases and trust development patterns to prevent unintended and potentially harmful outcomes in trust-sensitive applications of AI. 

---
# Automated Creativity Evaluation for Large Language Models: A Reference-Based Approach 

**Authors**: Ruizhe Li, Chiwei Zhu, Benfeng Xu, Xiaorui Wang, Zhendong Mao  

**Link**: [PDF](https://arxiv.org/pdf/2504.15784)  

**Abstract**: Creative writing is a key capability of Large Language Models (LLMs), with potential applications in literature, storytelling, and various creative domains. However, evaluating the creativity of machine-generated texts remains a significant challenge, as existing methods either rely on costly manual annotations or fail to align closely with human assessments. In this paper, we propose an effective automated evaluation method based on the Torrance Test of Creative Writing (TTCW), which evaluates creativity as product. Our method employs a reference-based Likert-style approach, scoring generated creative texts relative to high-quality reference texts across various tests. Experimental results demonstrate that our method significantly improves the alignment between LLM evaluations and human assessments, achieving a pairwise accuracy of 0.75 (+15\%). 

---
# Shannon invariants: A scalable approach to information decomposition 

**Authors**: Aaron J. Gutknecht, Fernando E. Rosas, David A. Ehrlich, Abdullah Makkeh, Pedro A. M. Mediano, Michael Wibral  

**Link**: [PDF](https://arxiv.org/pdf/2504.15779)  

**Abstract**: Distributed systems, such as biological and artificial neural networks, process information via complex interactions engaging multiple subsystems, resulting in high-order patterns with distinct properties across scales. Investigating how these systems process information remains challenging due to difficulties in defining appropriate multivariate metrics and ensuring their scalability to large systems. To address these challenges, we introduce a novel framework based on what we call "Shannon invariants" -- quantities that capture essential properties of high-order information processing in a way that depends only on the definition of entropy and can be efficiently calculated for large systems. Our theoretical results demonstrate how Shannon invariants can be used to resolve long-standing ambiguities regarding the interpretation of widely used multivariate information-theoretic measures. Moreover, our practical results reveal distinctive information-processing signatures of various deep learning architectures across layers, which lead to new insights into how these systems process information and how this evolves during training. Overall, our framework resolves fundamental limitations in analyzing high-order phenomena and offers broad opportunities for theoretical developments and empirical analyses. 

---
# Clifford Group Equivariant Diffusion Models for 3D Molecular Generation 

**Authors**: Cong Liu, Sharvaree Vadgama, David Ruhe, Erik Bekkers, Patrick Forrè  

**Link**: [PDF](https://arxiv.org/pdf/2504.15773)  

**Abstract**: This paper explores leveraging the Clifford algebra's expressive power for $\E(n)$-equivariant diffusion models. We utilize the geometric products between Clifford multivectors and the rich geometric information encoded in Clifford subspaces in \emph{Clifford Diffusion Models} (CDMs). We extend the diffusion process beyond just Clifford one-vectors to incorporate all higher-grade multivector subspaces. The data is embedded in grade-$k$ subspaces, allowing us to apply latent diffusion across complete multivectors. This enables CDMs to capture the joint distribution across different subspaces of the algebra, incorporating richer geometric information through higher-order features. We provide empirical results for unconditional molecular generation on the QM9 dataset, showing that CDMs provide a promising avenue for generative modeling. 

---
# Dynamic Intent Queries for Motion Transformer-based Trajectory Prediction 

**Authors**: Tobias Demmler, Lennart Hartung, Andreas Tamke, Thao Dang, Alexander Hegai, Karsten Haug, Lars Mikelsons  

**Link**: [PDF](https://arxiv.org/pdf/2504.15766)  

**Abstract**: In autonomous driving, accurately predicting the movements of other traffic participants is crucial, as it significantly influences a vehicle's planning processes. Modern trajectory prediction models strive to interpret complex patterns and dependencies from agent and map data. The Motion Transformer (MTR) architecture and subsequent work define the most accurate methods in common benchmarks such as the Waymo Open Motion Benchmark. The MTR model employs pre-generated static intention points as initial goal points for trajectory prediction. However, the static nature of these points frequently leads to misalignment with map data in specific traffic scenarios, resulting in unfeasible or unrealistic goal points. Our research addresses this limitation by integrating scene-specific dynamic intention points into the MTR model. This adaptation of the MTR model was trained and evaluated on the Waymo Open Motion Dataset. Our findings demonstrate that incorporating dynamic intention points has a significant positive impact on trajectory prediction accuracy, especially for predictions over long time horizons. Furthermore, we analyze the impact on ground truth trajectories which are not compliant with the map data or are illegal maneuvers. 

---
# iMedic: Towards Smartphone-based Self-Auscultation Tool for AI-Powered Pediatric Respiratory Assessment 

**Authors**: Seung Gyu Jeong, Sung Woo Nam, Seong Kwan Jung, Seong-Eun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.15743)  

**Abstract**: Respiratory auscultation is crucial for early detection of pediatric pneumonia, a condition that can quickly worsen without timely intervention. In areas with limited physician access, effective auscultation is challenging. We present a smartphone-based system that leverages built-in microphones and advanced deep learning algorithms to detect abnormal respiratory sounds indicative of pneumonia risk. Our end-to-end deep learning framework employs domain generalization to integrate a large electronic stethoscope dataset with a smaller smartphone-derived dataset, enabling robust feature learning for accurate respiratory assessments without expensive equipment. The accompanying mobile application guides caregivers in collecting high-quality lung sound samples and provides immediate feedback on potential pneumonia risks. User studies show strong classification performance and high acceptance, demonstrating the system's ability to facilitate proactive interventions and reduce preventable childhood pneumonia deaths. By seamlessly integrating into ubiquitous smartphones, this approach offers a promising avenue for more equitable and comprehensive remote pediatric care. 

---
# Collaborative Split Federated Learning with Parallel Training and Aggregation 

**Authors**: Yiannis Papageorgiou, Yannis Thomas, Alexios Filippakopoulos, Ramin Khalili, Iordanis Koutsopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2504.15724)  

**Abstract**: Federated learning (FL) operates based on model exchanges between the server and the clients, and it suffers from significant client-side computation and communication burden. Split federated learning (SFL) arises a promising solution by splitting the model into two parts, that are trained sequentially: the clients train the first part of the model (client-side model) and transmit it to the server that trains the second (server-side model). Existing SFL schemes though still exhibit long training delays and significant communication overhead, especially when clients of different computing capability participate. Thus, we propose Collaborative-Split Federated Learning~(C-SFL), a novel scheme that splits the model into three parts, namely the model parts trained at the computationally weak clients, the ones trained at the computationally strong clients, and the ones at the server. Unlike existing works, C-SFL enables parallel training and aggregation of model's parts at the clients and at the server, resulting in reduced training delays and commmunication overhead while improving the model's accuracy. Experiments verify the multiple gains of C-SFL against the existing schemes. 

---
# RePOPE: Impact of Annotation Errors on the POPE Benchmark 

**Authors**: Yannic Neuhaus, Matthias Hein  

**Link**: [PDF](https://arxiv.org/pdf/2504.15707)  

**Abstract**: Since data annotation is costly, benchmark datasets often incorporate labels from established image datasets. In this work, we assess the impact of label errors in MSCOCO on the frequently used object hallucination benchmark POPE. We re-annotate the benchmark images and identify an imbalance in annotation errors across different subsets. Evaluating multiple models on the revised labels, which we denote as RePOPE, we observe notable shifts in model rankings, highlighting the impact of label quality. Code and data are available at this https URL . 

---
# FADEL: Uncertainty-aware Fake Audio Detection with Evidential Deep Learning 

**Authors**: Ju Yeon Kang, Ji Won Yoon, Semin Kim, Min Hyun Han, Nam Soo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2504.15663)  

**Abstract**: Recently, fake audio detection has gained significant attention, as advancements in speech synthesis and voice conversion have increased the vulnerability of automatic speaker verification (ASV) systems to spoofing attacks. A key challenge in this task is generalizing models to detect unseen, out-of-distribution (OOD) attacks. Although existing approaches have shown promising results, they inherently suffer from overconfidence issues due to the usage of softmax for classification, which can produce unreliable predictions when encountering unpredictable spoofing attempts. To deal with this limitation, we propose a novel framework called fake audio detection with evidential learning (FADEL). By modeling class probabilities with a Dirichlet distribution, FADEL incorporates model uncertainty into its predictions, thereby leading to more robust performance in OOD scenarios. Experimental results on the ASVspoof2019 Logical Access (LA) and ASVspoof2021 LA datasets indicate that the proposed method significantly improves the performance of baseline models. Furthermore, we demonstrate the validity of uncertainty estimation by analyzing a strong correlation between average uncertainty and equal error rate (EER) across different spoofing algorithms. 

---
# VeriCoder: Enhancing LLM-Based RTL Code Generation through Functional Correctness Validation 

**Authors**: Anjiang Wei, Huanmi Tan, Tarun Suresh, Daniel Mendoza, Thiago S. F. X. Teixeira, Ke Wang, Caroline Trippel, Alex Aiken  

**Link**: [PDF](https://arxiv.org/pdf/2504.15659)  

**Abstract**: Recent advances in Large Language Models (LLMs) have sparked growing interest in applying them to Electronic Design Automation (EDA) tasks, particularly Register Transfer Level (RTL) code generation. While several RTL datasets have been introduced, most focus on syntactic validity rather than functional validation with tests, leading to training examples that compile but may not implement the intended behavior. We present VERICODER, a model for RTL code generation fine-tuned on a dataset validated for functional correctness. This fine-tuning dataset is constructed using a novel methodology that combines unit test generation with feedback-directed refinement. Given a natural language specification and an initial RTL design, we prompt a teacher model (GPT-4o-mini) to generate unit tests and iteratively revise the RTL design based on its simulation results using the generated tests. If necessary, the teacher model also updates the tests to ensure they comply with the natural language specification. As a result of this process, every example in our dataset is functionally validated, consisting of a natural language description, an RTL implementation, and passing tests. Fine-tuned on this dataset of over 125,000 examples, VERICODER achieves state-of-the-art metrics in functional correctness on VerilogEval and RTLLM, with relative gains of up to 71.7% and 27.4% respectively. An ablation study further shows that models trained on our functionally validated dataset outperform those trained on functionally non-validated datasets, underscoring the importance of high-quality datasets in RTL code generation. 

---
# A Vision-Enabled Prosthetic Hand for Children with Upper Limb Disabilities 

**Authors**: Md Abdul Baset Sarker, Art Nguyen, Sigmond Kukla, Kevin Fite, Masudul H. Imtiaz  

**Link**: [PDF](https://arxiv.org/pdf/2504.15654)  

**Abstract**: This paper introduces a novel AI vision-enabled pediatric prosthetic hand designed to assist children aged 10-12 with upper limb disabilities. The prosthesis features an anthropomorphic appearance, multi-articulating functionality, and a lightweight design that mimics a natural hand, making it both accessible and affordable for low-income families. Using 3D printing technology and integrating advanced machine vision, sensing, and embedded computing, the prosthetic hand offers a low-cost, customizable solution that addresses the limitations of current myoelectric prostheses. A micro camera is interfaced with a low-power FPGA for real-time object detection and assists with precise grasping. The onboard DL-based object detection and grasp classification models achieved accuracies of 96% and 100% respectively. In the force prediction, the mean absolute error was found to be 0.018. The features of the proposed prosthetic hand can thus be summarized as: a) a wrist-mounted micro camera for artificial sensing, enabling a wide range of hand-based tasks; b) real-time object detection and distance estimation for precise grasping; and c) ultra-low-power operation that delivers high performance within constrained power and resource limits. 

---
# Cost-Effective Text Clustering with Large Language Models 

**Authors**: Hongtao Wang, Taiyan Zhang, Renchi Yang, Jianliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15640)  

**Abstract**: Text clustering aims to automatically partition a collection of text documents into distinct clusters based on linguistic features. In the literature, this task is usually framed as metric clustering based on text embeddings from pre-trained encoders or a graph clustering problem upon pairwise similarities from an oracle, e.g., a large ML model. Recently, large language models (LLMs) bring significant advancement in this field by offering contextualized text embeddings and highly accurate similarity scores, but meanwhile, present grand challenges to cope with substantial computational and/or financial overhead caused by numerous API-based queries or inference calls to the models.
In response, this paper proposes TECL, a cost-effective framework that taps into the feedback from LLMs for accurate text clustering within a limited budget of queries to LLMs. Under the hood, TECL adopts our EdgeLLM or TriangleLLM to construct must-link/cannot-link constraints for text pairs, and further leverages such constraints as supervision signals input to our weighted constrained clustering approach to generate clusters. Particularly, EdgeLLM (resp. TriangleLLM) enables the identification of informative text pairs (resp. triplets) for querying LLMs via well-thought-out greedy algorithms and accurate extraction of pairwise constraints through carefully-crafted prompting techniques. Our experiments on multiple benchmark datasets exhibit that TECL consistently and considerably outperforms existing solutions in unsupervised text clustering under the same query cost for LLMs. 

---
# DR.FIX: Automatically Fixing Data Races at Industry Scale 

**Authors**: Farnaz Behrang, Zhizhou Zhang, Georgian-Vlad Saioc, Peng Liu, Milind Chabbi  

**Link**: [PDF](https://arxiv.org/pdf/2504.15637)  

**Abstract**: Data races are a prevalent class of concurrency bugs in shared-memory parallel programs, posing significant challenges to software reliability and reproducibility. While there is an extensive body of research on detecting data races and a wealth of practical detection tools across various programming languages, considerably less effort has been directed toward automatically fixing data races at an industrial scale. In large codebases, data races are continuously introduced and exhibit myriad patterns, making automated fixing particularly challenging.
In this paper, we tackle the problem of automatically fixing data races at an industrial scale. We present this http URL, a tool that combines large language models (LLMs) with program analysis to generate fixes for data races in real-world settings, effectively addressing a broad spectrum of racy patterns in complex code contexts. Implemented for Go--the programming language widely used in modern microservice architectures where concurrency is pervasive and data races are this http URL seamlessly integrates into existing development workflows. We detail the design of this http URL and examine how individual design choices influence the quality of the fixes produced. Over the past 18 months, this http URL has been integrated into developer workflows at Uber demonstrating its practical utility. During this period, this http URL produced patches for 224 (55%) from a corpus of 404 data races spanning various categories; 193 of these patches (86%) were accepted by more than a hundred developers via code reviews and integrated into the codebase. 

---
# Enhancing Reinforcement learning in 3-Dimensional Hydrophobic-Polar Protein Folding Model with Attention-based layers 

**Authors**: Peizheng Liu, Hitoshi Iba  

**Link**: [PDF](https://arxiv.org/pdf/2504.15634)  

**Abstract**: Transformer-based architectures have recently propelled advances in sequence modeling across domains, but their application to the hydrophobic-hydrophilic (H-P) model for protein folding remains relatively unexplored. In this work, we adapt a Deep Q-Network (DQN) integrated with attention mechanisms (Transformers) to address the 3D H-P protein folding problem. Our system formulates folding decisions as a self-avoiding walk in a reinforced environment, and employs a specialized reward function based on favorable hydrophobic interactions. To improve performance, the method incorporates validity check including symmetry-breaking constraints, dueling and double Q-learning, and prioritized replay to focus learning on critical transitions. Experimental evaluations on standard benchmark sequences demonstrate that our approach achieves several known best solutions for shorter sequences, and obtains near-optimal results for longer chains. This study underscores the promise of attention-based reinforcement learning for protein folding, and created a prototype of Transformer-based Q-network structure for 3-dimensional lattice models. 

---
# Exploring Next Token Prediction in Theory of Mind (ToM) Tasks: Comparative Experiments with GPT-2 and LLaMA-2 AI Models 

**Authors**: Pavan Yadav, Nikhil Khandalkar, Krishna Shinde, Lokesh B. Ramegowda, Rajarshi Das  

**Link**: [PDF](https://arxiv.org/pdf/2504.15604)  

**Abstract**: Language models have made significant progress in generating coherent text and predicting next tokens based on input prompts. This study compares the next-token prediction performance of two well-known models: OpenAI's GPT-2 and Meta's Llama-2-7b-chat-hf on Theory of Mind (ToM) tasks. To evaluate their capabilities, we built a dataset from 10 short stories sourced from the Explore ToM Dataset. We enhanced these stories by programmatically inserting additional sentences (infills) using GPT-4, creating variations that introduce different levels of contextual complexity. This setup enables analysis of how increasing context affects model performance. We tested both models under four temperature settings (0.01, 0.5, 1.0, 2.0) and evaluated their ability to predict the next token across three reasoning levels. Zero-order reasoning involves tracking the state, either current (ground truth) or past (memory). First-order reasoning concerns understanding another's mental state (e.g., "Does Anne know the apple is salted?"). Second-order reasoning adds recursion (e.g., "Does Anne think that Charles knows the apple is salted?").
Our results show that adding more infill sentences slightly reduces prediction accuracy, as added context increases complexity and ambiguity. Llama-2 consistently outperforms GPT-2 in prediction accuracy, especially at lower temperatures, demonstrating greater confidence in selecting the most probable token. As reasoning complexity rises, model responses diverge more. Notably, GPT-2 and Llama-2 display greater variability in predictions during first- and second-order reasoning tasks. These findings illustrate how model architecture, temperature, and contextual complexity influence next-token prediction, contributing to a better understanding of the strengths and limitations of current language models. 

---
# MetaMolGen: A Neural Graph Motif Generation Model for De Novo Molecular Design 

**Authors**: Zimo Yan, Jie Zhang, Zheng Xie, Chang Liu, Yizhen Liu, Yiping Song  

**Link**: [PDF](https://arxiv.org/pdf/2504.15587)  

**Abstract**: Molecular generation plays an important role in drug discovery and materials science, especially in data-scarce scenarios where traditional generative models often struggle to achieve satisfactory conditional generalization. To address this challenge, we propose MetaMolGen, a first-order meta-learning-based molecular generator designed for few-shot and property-conditioned molecular generation. MetaMolGen standardizes the distribution of graph motifs by mapping them to a normalized latent space, and employs a lightweight autoregressive sequence model to generate SMILES sequences that faithfully reflect the underlying molecular structure. In addition, it supports conditional generation of molecules with target properties through a learnable property projector integrated into the generative this http URL results demonstrate that MetaMolGen consistently generates valid and diverse SMILES sequences under low-data regimes, outperforming conventional baselines. This highlights its advantage in fast adaptation and efficient conditional generation for practical molecular design. 

---
# A Comprehensive Survey in LLM(-Agent) Full Stack Safety: Data, Training and Deployment 

**Authors**: Kun Wang, Guibin Zhang, Zhenhong Zhou, Jiahao Wu, Miao Yu, Shiqian Zhao, Chenlong Yin, Jinhu Fu, Yibo Yan, Hanjun Luo, Liang Lin, Zhihao Xu, Haolang Lu, Xinye Cao, Xinyun Zhou, Weifei Jin, Fanci Meng, Junyuan Mao, Hao Wu, Minghe Wang, Fan Zhang, Junfeng Fang, Chengwei Liu, Yifan Zhang, Qiankun Li, Chongye Guo, Yalan Qin, Yi Ding, Donghai Hong, Jiaming Ji, Xinfeng Li, Yifan Jiang, Dongxia Wang, Yihao Huang, Yufei Guo, Jen-tse Huang, Yanwei Yue, Wenke Huang, Guancheng Wan, Tianlin Li, Lei Bai, Jie Zhang, Qing Guo, Jingyi Wang, Tianlong Chen, Joey Tianyi Zhou, Xiaojun Jia, Weisong Sun, Cong Wu, Jing Chen, Xuming Hu, Yiming Li, Xiao Wang, Ningyu Zhang, Luu Anh Tuan, Guowen Xu, Tianwei Zhang, Xingjun Ma, Xiang Wang, Bo An, Jun Sun, Mohit Bansal, Shirui Pan, Yuval Elovici, Bhavya Kailkhura, Bo Li, Yaodong Yang, Hongwei Li, Wenyuan Xu, Yizhou Sun, Wei Wang, Qing Li, Ke Tang, Yu-Gang Jiang, Felix Juefei-Xu, Hui Xiong, Xiaofeng Wang, Shuicheng Yan, Dacheng Tao, Philip S. Yu, Qingsong Wen, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2504.15585)  

**Abstract**: The remarkable success of Large Language Models (LLMs) has illuminated a promising pathway toward achieving Artificial General Intelligence for both academic and industrial communities, owing to their unprecedented performance across various applications. As LLMs continue to gain prominence in both research and commercial domains, their security and safety implications have become a growing concern, not only for researchers and corporations but also for every nation. Currently, existing surveys on LLM safety primarily focus on specific stages of the LLM lifecycle, e.g., deployment phase or fine-tuning phase, lacking a comprehensive understanding of the entire "lifechain" of LLMs. To address this gap, this paper introduces, for the first time, the concept of "full-stack" safety to systematically consider safety issues throughout the entire process of LLM training, deployment, and eventual commercialization. Compared to the off-the-shelf LLM safety surveys, our work demonstrates several distinctive advantages: (I) Comprehensive Perspective. We define the complete LLM lifecycle as encompassing data preparation, pre-training, post-training, deployment and final commercialization. To our knowledge, this represents the first safety survey to encompass the entire lifecycle of LLMs. (II) Extensive Literature Support. Our research is grounded in an exhaustive review of over 800+ papers, ensuring comprehensive coverage and systematic organization of security issues within a more holistic understanding. (III) Unique Insights. Through systematic literature analysis, we have developed reliable roadmaps and perspectives for each chapter. Our work identifies promising research directions, including safety in data generation, alignment techniques, model editing, and LLM-based agent systems. These insights provide valuable guidance for researchers pursuing future work in this field. 

---
# A Large-scale Class-level Benchmark Dataset for Code Generation with LLMs 

**Authors**: Musfiqur Rahman, SayedHassan Khatoonabadi, Emad Shihab  

**Link**: [PDF](https://arxiv.org/pdf/2504.15564)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated promising capabilities in code generation tasks. However, most existing benchmarks focus on isolated functions and fail to capture the complexity of real-world, class-level software structures. To address this gap, we introduce a large-scale, Python class-level dataset curated from $13{,}174$ real-world open-source projects. The dataset contains over 842,000 class skeletons, each including class and method signatures, along with associated docstrings when available. We preserve structural and contextual dependencies critical to realistic software development scenarios and enrich the dataset with static code metrics to support downstream analysis. To evaluate the usefulness of this dataset, we use extracted class skeletons as prompts for GPT-4 to generate full class implementations. Results show that the LLM-generated classes exhibit strong lexical and structural similarity to human-written counterparts, with average ROUGE@L, BLEU, and TSED scores of 0.80, 0.59, and 0.73, respectively. These findings confirm that well-structured prompts derived from real-world class skeletons significantly enhance LLM performance in class-level code generation. This dataset offers a valuable resource for benchmarking, training, and improving LLMs in realistic software engineering contexts. 

---
# Do It For Me vs. Do It With Me: Investigating User Perceptions of Different Paradigms of Automation in Copilots for Feature-Rich Software 

**Authors**: Anjali Khurana, Xiaotian Su, April Yi Wang, Parmit K Chilana  

**Link**: [PDF](https://arxiv.org/pdf/2504.15549)  

**Abstract**: Large Language Model (LLM)-based in-application assistants, or copilots, can automate software tasks, but users often prefer learning by doing, raising questions about the optimal level of automation for an effective user experience. We investigated two automation paradigms by designing and implementing a fully automated copilot (AutoCopilot) and a semi-automated copilot (GuidedCopilot) that automates trivial steps while offering step-by-step visual guidance. In a user study (N=20) across data analysis and visual design tasks, GuidedCopilot outperformed AutoCopilot in user control, software utility, and learnability, especially for exploratory and creative tasks, while AutoCopilot saved time for simpler visual tasks. A follow-up design exploration (N=10) enhanced GuidedCopilot with task-and state-aware features, including in-context preview clips and adaptive instructions. Our findings highlight the critical role of user control and tailored guidance in designing the next generation of copilots that enhance productivity, support diverse skill levels, and foster deeper software engagement. 

---
# A Framework for Testing and Adapting REST APIs as LLM Tools 

**Authors**: Jayachandu Bandlamudi, Ritwik Chaudhuri, Neelamadhav Gantayat, Kushal Mukherjee, Prerna Agarwal, Renuka Sindhgatta, Sameep Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2504.15546)  

**Abstract**: Large Language Models (LLMs) are enabling autonomous agents to perform complex workflows using external tools or functions, often provided via REST APIs in enterprise systems. However, directly utilizing these APIs as tools poses challenges due to their complex input schemas, elaborate responses, and often ambiguous documentation. Current benchmarks for tool testing do not adequately address these complexities, leading to a critical gap in evaluating API readiness for agent-driven automation. In this work, we present a novel testing framework aimed at evaluating and enhancing the readiness of REST APIs to function as tools for LLM-based agents. Our framework transforms apis as tools, generates comprehensive test cases for the APIs, translates tests cases into natural language instructions suitable for agents, enriches tool definitions and evaluates the agent's ability t correctly invoke the API and process its inputs and responses. To provide actionable insights, we analyze the outcomes of 750 test cases, presenting a detailed taxonomy of errors, including input misinterpretation, output handling inconsistencies, and schema mismatches. Additionally, we classify these test cases to streamline debugging and refinement of tool integrations. This work offers a foundational step toward enabling enterprise APIs as tools, improving their usability in agent-based applications. 

---
# IPBench: Benchmarking the Knowledge of Large Language Models in Intellectual Property 

**Authors**: Qiyao Wang, Guhong Chen, Hongbo Wang, Huaren Liu, Minghui Zhu, Zhifei Qin, Linwei Li, Yilin Yue, Shiqiang Wang, Jiayan Li, Yihang Wu, Ziqiang Liu, Longze Chen, Run Luo, Liyang Fan, Jiaming Li, Lei Zhang, Kan Xu, Hongfei Lin, Hamid Alinejad-Rokny, Shiwen Ni, Yuan Lin, Min Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15524)  

**Abstract**: Intellectual Property (IP) is a unique domain that integrates technical and legal knowledge, making it inherently complex and knowledge-intensive. As large language models (LLMs) continue to advance, they show great potential for processing IP tasks, enabling more efficient analysis, understanding, and generation of IP-related content. However, existing datasets and benchmarks either focus narrowly on patents or cover limited aspects of the IP field, lacking alignment with real-world scenarios. To bridge this gap, we introduce the first comprehensive IP task taxonomy and a large, diverse bilingual benchmark, IPBench, covering 8 IP mechanisms and 20 tasks. This benchmark is designed to evaluate LLMs in real-world intellectual property applications, encompassing both understanding and generation. We benchmark 16 LLMs, ranging from general-purpose to domain-specific models, and find that even the best-performing model achieves only 75.8% accuracy, revealing substantial room for improvement. Notably, open-source IP and law-oriented models lag behind closed-source general-purpose models. We publicly release all data and code of IPBench and will continue to update it with additional IP-related tasks to better reflect real-world challenges in the intellectual property domain. 

---
# Transport f divergences 

**Authors**: Wuchen Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.15515)  

**Abstract**: We define a class of divergences to measure differences between probability density functions in one-dimensional sample space. The construction is based on the convex function with the Jacobi operator of mapping function that pushforwards one density to the other. We call these information measures {\em transport $f$-divergences}. We present several properties of transport $f$-divergences, including invariances, convexities, variational formulations, and Taylor expansions in terms of mapping functions. Examples of transport $f$-divergences in generative models are provided. 

---
# Guillotine: Hypervisors for Isolating Malicious AIs 

**Authors**: James Mickens, Sarah Radway, Ravi Netravali  

**Link**: [PDF](https://arxiv.org/pdf/2504.15499)  

**Abstract**: As AI models become more embedded in critical sectors like finance, healthcare, and the military, their inscrutable behavior poses ever-greater risks to society. To mitigate this risk, we propose Guillotine, a hypervisor architecture for sandboxing powerful AI models -- models that, by accident or malice, can generate existential threats to humanity. Although Guillotine borrows some well-known virtualization techniques, Guillotine must also introduce fundamentally new isolation mechanisms to handle the unique threat model posed by existential-risk AIs. For example, a rogue AI may try to introspect upon hypervisor software or the underlying hardware substrate to enable later subversion of that control plane; thus, a Guillotine hypervisor requires careful co-design of the hypervisor software and the CPUs, RAM, NIC, and storage devices that support the hypervisor software, to thwart side channel leakage and more generally eliminate mechanisms for AI to exploit reflection-based vulnerabilities. Beyond such isolation at the software, network, and microarchitectural layers, a Guillotine hypervisor must also provide physical fail-safes more commonly associated with nuclear power plants, avionic platforms, and other types of mission critical systems. Physical fail-safes, e.g., involving electromechanical disconnection of network cables, or the flooding of a datacenter which holds a rogue AI, provide defense in depth if software, network, and microarchitectural isolation is compromised and a rogue AI must be temporarily shut down or permanently destroyed. 

---
# Scalable APT Malware Classification via Parallel Feature Extraction and GPU-Accelerated Learning 

**Authors**: Noah Subedar, Taeui Kim, Saathwick Venkataramalingam  

**Link**: [PDF](https://arxiv.org/pdf/2504.15497)  

**Abstract**: This paper presents an underlying framework for both automating and accelerating malware classification, more specifically, mapping malicious executables to known Advanced Persistent Threat (APT) groups. The main feature of this analysis is the assembly-level instructions present in executables which are also known as opcodes. The collection of such opcodes on many malicious samples is a lengthy process; hence, open-source reverse engineering tools are used in tandem with scripts that leverage parallel computing to analyze multiple files at once. Traditional and deep learning models are applied to create models capable of classifying malware samples. One-gram and two-gram datasets are constructed and used to train models such as SVM, KNN, and Decision Tree; however, they struggle to provide adequate results without relying on metadata to support n-gram sequences. The computational limitations of such models are overcome with convolutional neural networks (CNNs) and heavily accelerated using graphical compute unit (GPU) resources. 

---
# CAPTURe: Evaluating Spatial Reasoning in Vision Language Models via Occluded Object Counting 

**Authors**: Atin Pothiraj, Elias Stengel-Eskin, Jaemin Cho, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2504.15485)  

**Abstract**: Recognizing and reasoning about occluded (partially or fully hidden) objects is vital to understanding visual scenes, as occlusions frequently occur in real-world environments and act as obstacles for spatial comprehension. To test models' ability to reason about multiple occluded objects, we introduce a novel task, Counting Amodally for Patterns Through Unseen REgions (CAPTURe), which requires a model to count objects arranged in a pattern by inferring how the pattern continues behind an occluder (an object which blocks parts of the scene). CAPTURe requires both recognizing visual patterns and reasoning, making it a useful testbed for evaluating vision-language models (VLMs) on whether they understand occluded patterns and possess spatial understanding skills. By requiring models to reason about occluded objects, CAPTURe also tests VLMs' ability to form world models that would allow them to fill in missing information. CAPTURe consists of two parts: (1) CAPTURe-real, with manually filtered images of real objects in patterns and (2) CAPTURe-synthetic, a controlled diagnostic with generated patterned images. We evaluate four strong VLMs (GPT-4o, Intern-VL2, Molmo, and Qwen2-VL) on CAPTURe, finding that models struggle to count on both occluded and unoccluded patterns. Crucially, we find that models perform worse with occlusion, suggesting that VLMs are also deficient in inferring unseen spatial relationships: even the strongest VLMs like GPT-4o fail to count with occlusion. In contrast, we find that humans achieve very little error on CAPTURe. We also find that providing auxiliary information of occluded object locations increases performance, underscoring that the model error comes both from an inability to handle occlusion as well as difficulty counting in images. 

---
# Demand for LLMs: Descriptive Evidence on Substitution, Market Expansion, and Multihoming 

**Authors**: Andrey Fradkin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15440)  

**Abstract**: This paper documents three stylized facts about the demand for Large Language Models (LLMs) using data from OpenRouter, a prominent LLM marketplace. First, new models experience rapid initial adoption that stabilizes within weeks. Second, model releases differ substantially in whether they primarily attract new users or substitute demand from competing models. Third, multihoming, using multiple models simultaneously, is common among apps. These findings suggest significant horizontal and vertical differentiation in the LLM market, implying opportunities for providers to maintain demand and pricing power despite rapid technological advances. 

---
# Trillion 7B Technical Report 

**Authors**: Sungjun Han, Juyoung Suk, Suyeong An, Hyungguk Kim, Kyuseok Kim, Wonsuk Yang, Seungtaek Choi, Jamin Shin  

**Link**: [PDF](https://arxiv.org/pdf/2504.15431)  

**Abstract**: We introduce Trillion-7B, the most token-efficient Korean-centric multilingual LLM available. Our novel Cross-lingual Document Attention (XLDA) mechanism enables highly efficient and effective knowledge transfer from English to target languages like Korean and Japanese. Combined with optimized data mixtures, language-specific filtering, and tailored tokenizer construction, Trillion-7B achieves competitive performance while dedicating only 10\% of its 2T training tokens to multilingual data and requiring just 59.4K H100 GPU hours (\$148K) for full training. Comprehensive evaluations across 27 benchmarks in four languages demonstrate Trillion-7B's robust multilingual performance and exceptional cross-lingual consistency. 

---
# Solving Multi-Agent Safe Optimal Control with Distributed Epigraph Form MARL 

**Authors**: Songyuan Zhang, Oswin So, Mitchell Black, Zachary Serlin, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15425)  

**Abstract**: Tasks for multi-robot systems often require the robots to collaborate and complete a team goal while maintaining safety. This problem is usually formalized as a constrained Markov decision process (CMDP), which targets minimizing a global cost and bringing the mean of constraint violation below a user-defined threshold. Inspired by real-world robotic applications, we define safety as zero constraint violation. While many safe multi-agent reinforcement learning (MARL) algorithms have been proposed to solve CMDPs, these algorithms suffer from unstable training in this setting. To tackle this, we use the epigraph form for constrained optimization to improve training stability and prove that the centralized epigraph form problem can be solved in a distributed fashion by each agent. This results in a novel centralized training distributed execution MARL algorithm named Def-MARL. Simulation experiments on 8 different tasks across 2 different simulators show that Def-MARL achieves the best overall performance, satisfies safety constraints, and maintains stable training. Real-world hardware experiments on Crazyflie quadcopters demonstrate the ability of Def-MARL to safely coordinate agents to complete complex collaborative tasks compared to other methods. 

---
# LLM-Assisted Translation of Legacy FORTRAN Codes to C++: A Cross-Platform Study 

**Authors**: Nishath Rajiv Ranasinghe, Shawn M. Jones, Michal Kucer, Ayan Biswas, Daniel O'Malley, Alexander Buschmann Most, Selma Liliane Wanna, Ajay Sreekumar  

**Link**: [PDF](https://arxiv.org/pdf/2504.15424)  

**Abstract**: Large Language Models (LLMs) are increasingly being leveraged for generating and translating scientific computer codes by both domain-experts and non-domain experts. Fortran has served as one of the go to programming languages in legacy high-performance computing (HPC) for scientific discoveries. Despite growing adoption, LLM-based code translation of legacy code-bases has not been thoroughly assessed or quantified for its usability. Here, we studied the applicability of LLM-based translation of Fortran to C++ as a step towards building an agentic-workflow using open-weight LLMs on two different computational platforms. We statistically quantified the compilation accuracy of the translated C++ codes, measured the similarity of the LLM translated code to the human translated C++ code, and statistically quantified the output similarity of the Fortran to C++ translation. 

---
# On the Boolean Network Theory of Datalog$^\neg$ 

**Authors**: Van-Giang Trinh, Belaid Benhamou, Sylvain Soliman, François Fages  

**Link**: [PDF](https://arxiv.org/pdf/2504.15417)  

**Abstract**: Datalog$^\neg$ is a central formalism used in a variety of domains ranging from deductive databases and abstract argumentation frameworks to answer set programming. Its model theory is the finite counterpart of the logical semantics developed for normal logic programs, mainly based on the notions of Clark's completion and two-valued or three-valued canonical models including supported, stable, regular and well-founded models. In this paper we establish a formal link between Datalog$^\neg$ and Boolean network theory, which was initially introduced by Stuart Kaufman and René Thomas to reason about gene regulatory networks. We use previous results from Boolean network theory to prove that in the absence of odd cycles in a Datalog$^\neg$ program, the regular models coincide with the stable models, which entails the existence of stable models, and in the absence of even cycles, we show the uniqueness of stable partial models, which entails the uniqueness of regular models. These results on regular models have been claimed by You and Yuan in 1994 for normal logic programs but we show problems in their definition of well-founded stratification and in their proofs that we can fix for negative normal logic programs only. We also give upper bounds on the numbers of stable partial, regular, and stable models of a Datalog$^\neg$ program using the cardinality of a feedback vertex set in its atom dependency graph. Interestingly, our connection to Boolean network theory also points us to the notion of trap spaces for Datalog$^\neg$ programs. We relate the notions of supported or stable trap spaces to the other semantics of Datalog$^\neg$, and show the equivalence between subset-minimal stable trap spaces and regular models. 

---
# Towards Understanding Camera Motions in Any Video 

**Authors**: Zhiqiu Lin, Siyuan Cen, Daniel Jiang, Jay Karhade, Hewei Wang, Chancharik Mitra, Tiffany Ling, Yuhan Huang, Sifan Liu, Mingyu Chen, Rushikesh Zawar, Xue Bai, Yilun Du, Chuang Gan, Deva Ramanan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15376)  

**Abstract**: We introduce CameraBench, a large-scale dataset and benchmark designed to assess and improve camera motion understanding. CameraBench consists of ~3,000 diverse internet videos, annotated by experts through a rigorous multi-stage quality control process. One of our contributions is a taxonomy of camera motion primitives, designed in collaboration with cinematographers. We find, for example, that some motions like "follow" (or tracking) require understanding scene content like moving subjects. We conduct a large-scale human study to quantify human annotation performance, revealing that domain expertise and tutorial-based training can significantly enhance accuracy. For example, a novice may confuse zoom-in (a change of intrinsics) with translating forward (a change of extrinsics), but can be trained to differentiate the two. Using CameraBench, we evaluate Structure-from-Motion (SfM) and Video-Language Models (VLMs), finding that SfM models struggle to capture semantic primitives that depend on scene content, while VLMs struggle to capture geometric primitives that require precise estimation of trajectories. We then fine-tune a generative VLM on CameraBench to achieve the best of both worlds and showcase its applications, including motion-augmented captioning, video question answering, and video-text retrieval. We hope our taxonomy, benchmark, and tutorials will drive future efforts towards the ultimate goal of understanding camera motions in any video. 

---
# Solving New Tasks by Adapting Internet Video Knowledge 

**Authors**: Calvin Luo, Zilai Zeng, Yilun Du, Chen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2504.15369)  

**Abstract**: Video generative models demonstrate great promise in robotics by serving as visual planners or as policy supervisors. When pretrained on internet-scale data, such video models intimately understand alignment with natural language, and can thus facilitate generalization to novel downstream behavior through text-conditioning. However, they may not be sensitive to the specificities of the particular environment the agent inhabits. On the other hand, training video models on in-domain examples of robotic behavior naturally encodes environment-specific intricacies, but the scale of available demonstrations may not be sufficient to support generalization to unseen tasks via natural language specification. In this work, we investigate different adaptation techniques that integrate in-domain information with large-scale pretrained video models, and explore the extent to which they enable novel text-conditioned generalization for robotic tasks, while also considering their independent data and resource considerations. We successfully demonstrate across robotic environments that adapting powerful video models with small scales of example data can successfully facilitate generalization to novel behaviors. In particular, we present a novel adaptation strategy, termed Inverse Probabilistic Adaptation, that not only consistently achieves strong generalization performance across robotic tasks and settings, but also exhibits robustness to the quality of adaptation data, successfully solving novel tasks even when only suboptimal in-domain demonstrations are available. 

---
# Med-CoDE: Medical Critique based Disagreement Evaluation Framework 

**Authors**: Mohit Gupta, Akiko Aizawa, Rajiv Ratn Shah  

**Link**: [PDF](https://arxiv.org/pdf/2504.15330)  

**Abstract**: The emergence of large language models (LLMs) has significantly influenced numerous fields, including healthcare, by enhancing the capabilities of automated systems to process and generate human-like text. However, despite their advancements, the reliability and accuracy of LLMs in medical contexts remain critical concerns. Current evaluation methods often lack robustness and fail to provide a comprehensive assessment of LLM performance, leading to potential risks in clinical settings. In this work, we propose Med-CoDE, a specifically designed evaluation framework for medical LLMs to address these challenges. The framework leverages a critique-based approach to quantitatively measure the degree of disagreement between model-generated responses and established medical ground truths. This framework captures both accuracy and reliability in medical settings. The proposed evaluation framework aims to fill the existing gap in LLM assessment by offering a systematic method to evaluate the quality and trustworthiness of medical LLMs. Through extensive experiments and case studies, we illustrate the practicality of our framework in providing a comprehensive and reliable evaluation of medical LLMs. 

---
# Bayesian Federated Learning for Continual Training 

**Authors**: Usevalad Milasheuski, Luca Barbieri, Sanaz Kianoush, Monica Nicoli, Stefano Savazzi  

**Link**: [PDF](https://arxiv.org/pdf/2504.15328)  

**Abstract**: Bayesian Federated Learning (BFL) enables uncertainty quantification and robust adaptation in distributed learning. In contrast to the frequentist approach, it estimates the posterior distribution of a global model, offering insights into model reliability. However, current BFL methods neglect continual learning challenges in dynamic environments where data distributions shift over time. We propose a continual BFL framework applied to human sensing with radar data collected over several days. Using Stochastic Gradient Langevin Dynamics (SGLD), our approach sequentially updates the model, leveraging past posteriors to construct the prior for the new tasks. We assess the accuracy, the expected calibration error (ECE) and the convergence speed of our approach against several baselines. Results highlight the effectiveness of continual Bayesian updates in preserving knowledge and adapting to evolving data. 

---
# Significativity Indices for Agreement Values 

**Authors**: Alberto Casagrande, Francesco Fabris, Rossano Girometti, Roberto Pagliarini  

**Link**: [PDF](https://arxiv.org/pdf/2504.15325)  

**Abstract**: Agreement measures, such as Cohen's kappa or intraclass correlation, gauge the matching between two or more classifiers. They are used in a wide range of contexts from medicine, where they evaluate the effectiveness of medical treatments and clinical trials, to artificial intelligence, where they can quantify the approximation due to the reduction of a classifier. The consistency of different classifiers to a golden standard can be compared simply by using the order induced by their agreement measure with respect to the golden standard itself. Nevertheless, labelling an approach as good or bad exclusively by using the value of an agreement measure requires a scale or a significativity index. Some quality scales have been proposed in the literature for Cohen's kappa, but they are mainly naive, and their boundaries are arbitrary. This work proposes a general approach to evaluate the significativity of any agreement value between two classifiers and introduces two significativity indices: one dealing with finite data sets, the other one handling classification probability distributions. Moreover, this manuscript considers the computational issues of evaluating such indices and identifies some efficient algorithms to evaluate them. 

---
# A Graph Based Raman Spectral Processing Technique for Exosome Classification 

**Authors**: Vuong M. Ngo, Edward Bolger, Stan Goodwin, John O'Sullivan, Dinh Viet Cuong, Mark Roantree  

**Link**: [PDF](https://arxiv.org/pdf/2504.15324)  

**Abstract**: Exosomes are small vesicles crucial for cell signaling and disease biomarkers. Due to their complexity, an "omics" approach is preferable to individual biomarkers. While Raman spectroscopy is effective for exosome analysis, it requires high sample concentrations and has limited sensitivity to lipids and proteins. Surface-enhanced Raman spectroscopy helps overcome these challenges. In this study, we leverage Neo4j graph databases to organize 3,045 Raman spectra of exosomes, enhancing data generalization. To further refine spectral analysis, we introduce a novel spectral filtering process that integrates the PageRank Filter with optimal Dimensionality Reduction. This method improves feature selection, resulting in superior classification performance. Specifically, the Extra Trees model, using our spectral processing approach, achieves 0.76 and 0.857 accuracy in classifying hyperglycemic, hypoglycemic, and normal exosome samples based on Raman spectra and surface, respectively, with group 10-fold cross-validation. Our results show that graph-based spectral filtering combined with optimal dimensionality reduction significantly improves classification accuracy by reducing noise while preserving key biomarker signals. This novel framework enhances Raman-based exosome analysis, expanding its potential for biomedical applications, disease diagnostics, and biomarker discovery. 

---
# HyperFlow: Gradient-Free Emulation of Few-Shot Fine-Tuning 

**Authors**: Donggyun Kim, Chanwoo Kim, Seunghoon Hong  

**Link**: [PDF](https://arxiv.org/pdf/2504.15323)  

**Abstract**: While test-time fine-tuning is beneficial in few-shot learning, the need for multiple backpropagation steps can be prohibitively expensive in real-time or low-resource scenarios. To address this limitation, we propose an approach that emulates gradient descent without computing gradients, enabling efficient test-time adaptation. Specifically, we formulate gradient descent as an Euler discretization of an ordinary differential equation (ODE) and train an auxiliary network to predict the task-conditional drift using only the few-shot support set. The adaptation then reduces to a simple numerical integration (e.g., via the Euler method), which requires only a few forward passes of the auxiliary network -- no gradients or forward passes of the target model are needed. In experiments on cross-domain few-shot classification using the Meta-Dataset and CDFSL benchmarks, our method significantly improves out-of-domain performance over the non-fine-tuned baseline while incurring only 6\% of the memory cost and 0.02\% of the computation time of standard fine-tuning, thus establishing a practical middle ground between direct transfer and fully fine-tuned approaches. 

---
# How to systematically develop an effective AI-based bias correction model? 

**Authors**: Xiao Zhou, Yuze Sun, Jie Wu, Xiaomeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15322)  

**Abstract**: This study introduces ReSA-ConvLSTM, an artificial intelligence (AI) framework for systematic bias correction in numerical weather prediction (NWP). We propose three innovations by integrating dynamic climatological normalization, ConvLSTM with temporal causality constraints, and residual self-attention mechanisms. The model establishes a physics-aware nonlinear mapping between ECMWF forecasts and ERA5 reanalysis data. Using 41 years (1981-2021) of global atmospheric data, the framework reduces systematic biases in 2-m air temperature (T2m), 10-m winds (U10/V10), and sea-level pressure (SLP), achieving up to 20% RMSE reduction over 1-7 day forecasts compared to operational ECMWF outputs. The lightweight architecture (10.6M parameters) enables efficient generalization to multiple variables and downstream applications, reducing retraining time by 85% for cross-variable correction while improving ocean model skill through bias-corrected boundary conditions. The ablation experiments demonstrate that our innovations significantly improve the model's correction performance, suggesting that incorporating variable characteristics into the model helps enhance forecasting skills. 

---
# Enhancing DR Classification with Swin Transformer and Shifted Window Attention 

**Authors**: Meher Boulaabi, Takwa Ben Aïcha Gader, Afef Kacem Echi, Zied Bouraoui  

**Link**: [PDF](https://arxiv.org/pdf/2504.15317)  

**Abstract**: Diabetic retinopathy (DR) is a leading cause of blindness worldwide, underscoring the importance of early detection for effective treatment. However, automated DR classification remains challenging due to variations in image quality, class imbalance, and pixel-level similarities that hinder model training. To address these issues, we propose a robust preprocessing pipeline incorporating image cropping, Contrast-Limited Adaptive Histogram Equalization (CLAHE), and targeted data augmentation to improve model generalization and resilience. Our approach leverages the Swin Transformer, which utilizes hierarchical token processing and shifted window attention to efficiently capture fine-grained features while maintaining linear computational complexity. We validate our method on the Aptos and IDRiD datasets for multi-class DR classification, achieving accuracy rates of 89.65% and 97.40%, respectively. These results demonstrate the effectiveness of our model, particularly in detecting early-stage DR, highlighting its potential for improving automated retinal screening in clinical settings. 

---
# Diffusion-Driven Inertial Generated Data for Smartphone Location Classification 

**Authors**: Noa Cohen, Rotem Dror, Itzik Klein  

**Link**: [PDF](https://arxiv.org/pdf/2504.15315)  

**Abstract**: Despite the crucial role of inertial measurements in motion tracking and navigation systems, the time-consuming and resource-intensive nature of collecting extensive inertial data has hindered the development of robust machine learning models in this field. In recent years, diffusion models have emerged as a revolutionary class of generative models, reshaping the landscape of artificial data generation. These models surpass generative adversarial networks and other state-of-the-art approaches to complex tasks. In this work, we propose diffusion-driven specific force-generated data for smartphone location recognition. We provide a comprehensive evaluation methodology by comparing synthetic and real recorded specific force data across multiple metrics. Our results demonstrate that our diffusion-based generative model successfully captures the distinctive characteristics of specific force signals across different smartphone placement conditions. Thus, by creating diverse, realistic synthetic data, we can reduce the burden of extensive data collection while providing high-quality training data for machine learning models. 

---
# RINN: One Sample Radio Frequency Imaging based on Physics Informed Neural Network 

**Authors**: Fei Shang, Haohua Du, Dawei Yan, Panlong Yang, Xiang-Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.15311)  

**Abstract**: Due to its ability to work in non-line-of-sight and low-light environments, radio frequency (RF) imaging technology is expected to bring new possibilities for embodied intelligence and multimodal sensing. However, widely used RF devices (such as Wi-Fi) often struggle to provide high-precision electromagnetic measurements and large-scale datasets, hindering the application of RF imaging technology. In this paper, we combine the ideas of PINN to design the RINN network, using physical constraints instead of true value comparison constraints and adapting it with the characteristics of ubiquitous RF signals, allowing the RINN network to achieve RF imaging using only one sample without phase and with amplitude noise. Our numerical evaluation results show that compared with 5 classic algorithms based on phase data for imaging results, RINN's imaging results based on phaseless data are good, with indicators such as RRMSE (0.11) performing similarly well. RINN provides new possibilities for the universal development of radio frequency imaging technology. 

---
# Power Transformer Health Index and Life Span Assessment: A Comprehensive Review of Conventional and Machine Learning based Approaches 

**Authors**: Syeda Tahreem Zahra, Syed Kashif Imdad, Sohail Khan, Sohail Khalid, Nauman Anwar Baig  

**Link**: [PDF](https://arxiv.org/pdf/2504.15310)  

**Abstract**: Power transformers play a critical role within the electrical power system, making their health assessment and the prediction of their remaining lifespan paramount for the purpose of ensuring efficient operation and facilitating effective maintenance planning. This paper undertakes a comprehensive examination of existent literature, with a primary focus on both conventional and cutting-edge techniques employed within this domain. The merits and demerits of recent methodologies and techniques are subjected to meticulous scrutiny and explication. Furthermore, this paper expounds upon intelligent fault diagnosis methodologies and delves into the most widely utilized intelligent algorithms for the assessment of transformer conditions. Diverse Artificial Intelligence (AI) approaches, including Artificial Neural Networks (ANN) and Convolutional Neural Network (CNN), Support Vector Machine (SVM), Random Forest (RF), Genetic Algorithm (GA), and Particle Swarm Optimization (PSO), are elucidated offering pragmatic solutions for enhancing the performance of transformer fault diagnosis. The amalgamation of multiple AI methodologies and the exploration of timeseries analysis further contribute to the augmentation of diagnostic precision and the early detection of faults in transformers. By furnishing a comprehensive panorama of AI applications in the field of transformer fault diagnosis, this study lays the groundwork for future research endeavors and the progression of this critical area of study. 

---
# High-Throughput LLM inference on Heterogeneous Clusters 

**Authors**: Yi Xiong, Jinqi Huang, Wenjie Huang, Xuebing Yu, Entong Li, Zhixiong Ning, Jinhua Zhou, Li Zeng, Xin Chen  

**Link**: [PDF](https://arxiv.org/pdf/2504.15303)  

**Abstract**: Nowadays, many companies possess various types of AI accelerators, forming heterogeneous clusters. Efficiently leveraging these clusters for high-throughput large language model (LLM) inference services can significantly reduce costs and expedite task processing. However, LLM inference on heterogeneous clusters presents two main challenges. Firstly, different deployment configurations can result in vastly different performance. The number of possible configurations is large, and evaluating the effectiveness of a specific setup is complex. Thus, finding an optimal configuration is not an easy task. Secondly, LLM inference instances within a heterogeneous cluster possess varying processing capacities, leading to different processing speeds for handling inference requests. Evaluating these capacities and designing a request scheduling algorithm that fully maximizes the potential of each instance is challenging. In this paper, we propose a high-throughput inference service system on heterogeneous clusters. First, the deployment configuration is optimized by modeling the resource amount and expected throughput and using the exhaustive search method. Second, a novel mechanism is proposed to schedule requests among instances, which fully considers the different processing capabilities of various instances. Extensive experiments show that the proposed scheduler improves throughput by 122.5% and 33.6% on two heterogeneous clusters, respectively. 

---
# A biologically Inspired Trust Model for Open Multi-Agent Systems that is Resilient to Rapid Performance Fluctuations 

**Authors**: Zoi Lygizou, Dimitris Kalles  

**Link**: [PDF](https://arxiv.org/pdf/2504.15301)  

**Abstract**: Trust management provides an alternative solution for securing open, dynamic, and distributed multi-agent systems, where conventional cryptographic methods prove to be impractical. However, existing trust models face challenges related to agent mobility, changing behaviors, and the cold start problem. To address these issues we introduced a biologically inspired trust model in which trustees assess their own capabilities and store trust data locally. This design improves mobility support, reduces communication overhead, resists disinformation, and preserves privacy. Despite these advantages, prior evaluations revealed limitations of our model in adapting to provider population changes and continuous performance fluctuations. This study proposes a novel algorithm, incorporating a self-classification mechanism for providers to detect performance drops potentially harmful for the service consumers. Simulation results demonstrate that the new algorithm outperforms its original version and FIRE, a well-known trust and reputation model, particularly in handling dynamic trustee behavior. While FIRE remains competitive under extreme environmental changes, the proposed algorithm demonstrates greater adaptability across various conditions. In contrast to existing trust modeling research, this study conducts a comprehensive evaluation of our model using widely recognized trust model criteria, assessing its resilience against common trust-related attacks while identifying strengths, weaknesses, and potential countermeasures. Finally, several key directions for future research are proposed. 

---
# D$^{2}$MoE: Dual Routing and Dynamic Scheduling for Efficient On-Device MoE-based LLM Serving 

**Authors**: Haodong Wang, Qihua Zhou, Zicong Hong, Song Guo  

**Link**: [PDF](https://arxiv.org/pdf/2504.15299)  

**Abstract**: The mixture of experts (MoE) model is a sparse variant of large language models (LLMs), designed to hold a better balance between intelligent capability and computational overhead. Despite its benefits, MoE is still too expensive to deploy on resource-constrained edge devices, especially with the demands of on-device inference services. Recent research efforts often apply model compression techniques, such as quantization, pruning and merging, to restrict MoE complexity. Unfortunately, due to their predefined static model optimization strategies, they cannot always achieve the desired quality-overhead trade-off when handling multiple requests, finally degrading the on-device quality of service. These limitations motivate us to propose the D$^2$MoE, an algorithm-system co-design framework that matches diverse task requirements by dynamically allocating the most proper bit-width to each expert. Specifically, inspired by the nested structure of matryoshka dolls, we propose the matryoshka weight quantization (MWQ) to progressively compress expert weights in a bit-nested manner and reduce the required runtime memory. On top of it, we further optimize the I/O-computation pipeline and design a heuristic scheduling algorithm following our hottest-expert-bit-first (HEBF) principle, which maximizes the expert parallelism between I/O and computation queue under constrained memory budgets, thus significantly reducing the idle temporal bubbles waiting for the experts to load. Evaluations on real edge devices show that D$^2$MoE improves the overall inference throughput by up to 1.39$\times$ and reduces the peak memory footprint by up to 53% over the latest on-device inference frameworks, while still preserving comparable serving accuracy as its INT8 counterparts. 

---
# Scalability Optimization in Cloud-Based AI Inference Services: Strategies for Real-Time Load Balancing and Automated Scaling 

**Authors**: Yihong Jin, Ze Yang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15296)  

**Abstract**: The rapid expansion of AI inference services in the cloud necessitates a robust scalability solution to manage dynamic workloads and maintain high performance. This study proposes a comprehensive scalability optimization framework for cloud AI inference services, focusing on real-time load balancing and autoscaling strategies. The proposed model is a hybrid approach that combines reinforcement learning for adaptive load distribution and deep neural networks for accurate demand forecasting. This multi-layered approach enables the system to anticipate workload fluctuations and proactively adjust resources, ensuring maximum resource utilisation and minimising latency. Furthermore, the incorporation of a decentralised decision-making process within the model serves to enhance fault tolerance and reduce response time in scaling operations. Experimental results demonstrate that the proposed model enhances load balancing efficiency by 35\ and reduces response delay by 28\, thereby exhibiting a substantial optimization effect in comparison with conventional scalability solutions. 

---
# CUBETESTERAI: Automated JUnit Test Generation using the LLaMA Model 

**Authors**: Daniele Gorla, Shivam Kumar, Pietro Nicolaus Roselli Lorenzini, Alireza Alipourfaz  

**Link**: [PDF](https://arxiv.org/pdf/2504.15286)  

**Abstract**: This paper presents an approach to automating JUnit test generation for Java applications using the Spring Boot framework, leveraging the LLaMA (Large Language Model Architecture) model to enhance the efficiency and accuracy of the testing process. The resulting tool, called CUBETESTERAI, includes a user-friendly web interface and the integration of a CI/CD pipeline using GitLab and Docker. These components streamline the automated test generation process, allowing developers to generate JUnit tests directly from their code snippets with minimal manual intervention. The final implementation executes the LLaMA models through RunPod, an online GPU service, which also enhances the privacy of our tool. Using the advanced natural language processing capabilities of the LLaMA model, CUBETESTERAI is able to generate test cases that provide high code coverage and accurate validation of software functionalities in Java-based Spring Boot applications. Furthermore, it efficiently manages resource-intensive operations and refines the generated tests to address common issues like missing imports and handling of private methods. By comparing CUBETESTERAI with some state-of-the-art tools, we show that our proposal consistently demonstrates competitive and, in many cases, better performance in terms of code coverage in different real-life Java programs. 

---
