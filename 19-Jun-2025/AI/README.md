# Embodied Web Agents: Bridging Physical-Digital Realms for Integrated Agent Intelligence 

**Authors**: Yining Hong, Rui Sun, Bingxuan Li, Xingcheng Yao, Maxine Wu, Alexander Chien, Da Yin, Ying Nian Wu, Zhecan James Wang, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15677)  

**Abstract**: AI agents today are mostly siloed - they either retrieve and reason over vast amount of digital information and knowledge obtained online; or interact with the physical world through embodied perception, planning and action - but rarely both. This separation limits their ability to solve tasks that require integrated physical and digital intelligence, such as cooking from online recipes, navigating with dynamic map data, or interpreting real-world landmarks using web knowledge. We introduce Embodied Web Agents, a novel paradigm for AI agents that fluidly bridge embodiment and web-scale reasoning. To operationalize this concept, we first develop the Embodied Web Agents task environments, a unified simulation platform that tightly integrates realistic 3D indoor and outdoor environments with functional web interfaces. Building upon this platform, we construct and release the Embodied Web Agents Benchmark, which encompasses a diverse suite of tasks including cooking, navigation, shopping, tourism, and geolocation - all requiring coordinated reasoning across physical and digital realms for systematic assessment of cross-domain intelligence. Experimental results reveal significant performance gaps between state-of-the-art AI systems and human capabilities, establishing both challenges and opportunities at the intersection of embodied cognition and web-scale knowledge access. All datasets, codes and websites are publicly available at our project page this https URL. 

---
# SwarmAgentic: Towards Fully Automated Agentic System Generation via Swarm Intelligence 

**Authors**: Yao Zhang, Chenyang Lin, Shijie Tang, Haokun Chen, Shijie Zhou, Yunpu Ma, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2506.15672)  

**Abstract**: The rapid progress of Large Language Models has advanced agentic systems in decision-making, coordination, and task execution. Yet, existing agentic system generation frameworks lack full autonomy, missing from-scratch agent generation, self-optimizing agent functionality, and collaboration, limiting adaptability and scalability. We propose SwarmAgentic, a framework for fully automated agentic system generation that constructs agentic systems from scratch and jointly optimizes agent functionality and collaboration as interdependent components through language-driven exploration. To enable efficient search over system-level structures, SwarmAgentic maintains a population of candidate systems and evolves them via feedback-guided updates, drawing inspiration from Particle Swarm Optimization (PSO). We evaluate our method on six real-world, open-ended, and exploratory tasks involving high-level planning, system-level coordination, and creative reasoning. Given only a task description and an objective function, SwarmAgentic outperforms all baselines, achieving a +261.8% relative improvement over ADAS on the TravelPlanner benchmark, highlighting the effectiveness of full automation in structurally unconstrained tasks. This framework marks a significant step toward scalable and autonomous agentic system design, bridging swarm intelligence with fully automated system multi-agent generation. Our code is publicly released at this https URL. 

---
# Exploring and Exploiting the Inherent Efficiency within Large Reasoning Models for Self-Guided Efficiency Enhancement 

**Authors**: Weixiang Zhao, Jiahe Guo, Yang Deng, Xingyu Sui, Yulin Hu, Yanyan Zhao, Wanxiang Che, Bing Qin, Tat-Seng Chua, Ting Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15647)  

**Abstract**: Recent advancements in large reasoning models (LRMs) have significantly enhanced language models' capabilities in complex problem-solving by emulating human-like deliberative thinking. However, these models often exhibit overthinking (i.e., the generation of unnecessarily verbose and redundant content), which hinders efficiency and inflates inference cost. In this work, we explore the representational and behavioral origins of this inefficiency, revealing that LRMs inherently possess the capacity for more concise reasoning. Empirical analyses show that correct reasoning paths vary significantly in length, and the shortest correct responses often suffice, indicating untapped efficiency potential. Exploiting these findings, we propose two lightweight methods to enhance LRM efficiency. First, we introduce Efficiency Steering, a training-free activation steering technique that modulates reasoning behavior via a single direction in the model's representation space. Second, we develop Self-Rewarded Efficiency RL, a reinforcement learning framework that dynamically balances task accuracy and brevity by rewarding concise correct solutions. Extensive experiments on seven LRM backbones across multiple mathematical reasoning benchmarks demonstrate that our methods significantly reduce reasoning length while preserving or improving task performance. Our results highlight that reasoning efficiency can be improved by leveraging and guiding the intrinsic capabilities of existing models in a self-guided manner. 

---
# The AI Policy Module: Developing Computer Science Student Competency in AI Ethics and Policy 

**Authors**: James Weichert, Daniel Dunlap, Mohammed Farghally, Hoda Eldardiry  

**Link**: [PDF](https://arxiv.org/pdf/2506.15639)  

**Abstract**: As artificial intelligence (AI) further embeds itself into many settings across personal and professional contexts, increasing attention must be paid not only to AI ethics, but also to the governance and regulation of AI technologies through AI policy. However, the prevailing post-secondary computing curriculum is currently ill-equipped to prepare future AI practitioners to confront increasing demands to implement abstract ethical principles and normative policy preferences into the design and development of AI systems. We believe that familiarity with the 'AI policy landscape' and the ability to translate ethical principles to practices will in the future constitute an important responsibility for even the most technically-focused AI engineers.
Toward preparing current computer science (CS) students for these new expectations, we developed an AI Policy Module to introduce discussions of AI policy into the CS curriculum. Building on a successful pilot in fall 2024, in this innovative practice full paper we present an updated and expanded version of the module, including a technical assignment on "AI regulation". We present the findings from our pilot of the AI Policy Module 2.0, evaluating student attitudes towards AI ethics and policy through pre- and post-module surveys. Following the module, students reported increased concern about the ethical impacts of AI technologies while also expressing greater confidence in their abilities to engage in discussions about AI regulation. Finally, we highlight the AI Regulation Assignment as an effective and engaging tool for exploring the limits of AI alignment and emphasizing the role of 'policy' in addressing ethical challenges. 

---
# The Effect of State Representation on LLM Agent Behavior in Dynamic Routing Games 

**Authors**: Lyle Goodyear, Rachel Guo, Ramesh Johari  

**Link**: [PDF](https://arxiv.org/pdf/2506.15624)  

**Abstract**: Large Language Models (LLMs) have shown promise as decision-makers in dynamic settings, but their stateless nature necessitates creating a natural language representation of history. We present a unifying framework for systematically constructing natural language "state" representations for prompting LLM agents in repeated multi-agent games. Previous work on games with LLM agents has taken an ad hoc approach to encoding game history, which not only obscures the impact of state representation on agents' behavior, but also limits comparability between studies. Our framework addresses these gaps by characterizing methods of state representation along three axes: action informativeness (i.e., the extent to which the state representation captures actions played); reward informativeness (i.e., the extent to which the state representation describes rewards obtained); and prompting style (or natural language compression, i.e., the extent to which the full text history is summarized).
We apply this framework to a dynamic selfish routing game, chosen because it admits a simple equilibrium both in theory and in human subject experiments \cite{rapoport_choice_2009}. Despite the game's relative simplicity, we find that there are key dependencies of LLM agent behavior on the natural language state representation. In particular, we observe that representations which provide agents with (1) summarized, rather than complete, natural language representations of past history; (2) information about regrets, rather than raw payoffs; and (3) limited information about others' actions lead to behavior that more closely matches game theoretic equilibrium predictions, and with more stable game play by the agents. By contrast, other representations can exhibit either large deviations from equilibrium, higher variation in dynamic game play over time, or both. 

---
# Managing Complex Failure Analysis Workflows with LLM-based Reasoning and Acting Agents 

**Authors**: Aline Dobrovsky, Konstantin Schekotihin, Christian Burmer  

**Link**: [PDF](https://arxiv.org/pdf/2506.15567)  

**Abstract**: Failure Analysis (FA) is a highly intricate and knowledge-intensive process. The integration of AI components within the computational infrastructure of FA labs has the potential to automate a variety of tasks, including the detection of non-conformities in images, the retrieval of analogous cases from diverse data sources, and the generation of reports from annotated images. However, as the number of deployed AI models increases, the challenge lies in orchestrating these components into cohesive and efficient workflows that seamlessly integrate with the FA process.
This paper investigates the design and implementation of a Large Language Model (LLM)-based Planning Agent (LPA) to assist FA engineers in solving their analysis cases. The LPA integrates LLMs with advanced planning capabilities and external tool utilization, enabling autonomous processing of complex queries, retrieval of relevant data from external systems, and generation of human-readable responses. Evaluation results demonstrate the agent's operational effectiveness and reliability in supporting FA tasks. 

---
# Efficient and Generalizable Environmental Understanding for Visual Navigation 

**Authors**: Ruoyu Wang, Xinshu Li, Chen Wang, Lina Yao  

**Link**: [PDF](https://arxiv.org/pdf/2506.15377)  

**Abstract**: Visual Navigation is a core task in Embodied AI, enabling agents to navigate complex environments toward given objectives. Across diverse settings within Navigation tasks, many necessitate the modelling of sequential data accumulated from preceding time steps. While existing methods perform well, they typically process all historical observations simultaneously, overlooking the internal association structure within the data, which may limit the potential for further improvements in task performance. We address this by examining the unique characteristics of Navigation tasks through the lens of causality, introducing a causal framework to highlight the limitations of conventional sequential methods. Leveraging this insight, we propose Causality-Aware Navigation (CAN), which incorporates a Causal Understanding Module to enhance the agent's environmental understanding capability. Empirical evaluations show that our approach consistently outperforms baselines across various tasks and simulation environments. Extensive ablations studies attribute these gains to the Causal Understanding Module, which generalizes effectively in both Reinforcement and Supervised Learning settings without computational overhead. 

---
# Joint Computation Offloading and Resource Allocation for Uncertain Maritime MEC via Cooperation of UAVs and Vessels 

**Authors**: Jiahao You, Ziye Jia, Chao Dong, Qihui Wu, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2506.15225)  

**Abstract**: The computation demands from the maritime Internet of Things (MIoT) increase rapidly in recent years, and the unmanned aerial vehicles (UAVs) and vessels based multi-access edge computing (MEC) can fulfill these MIoT requirements. However, the uncertain maritime tasks present significant challenges of inefficient computation offloading and resource allocation. In this paper, we focus on the maritime computation offloading and resource allocation through the cooperation of UAVs and vessels, with consideration of uncertain tasks. Specifically, we propose a cooperative MEC framework for computation offloading and resource allocation, including MIoT devices, UAVs and vessels. Then, we formulate the optimization problem to minimize the total execution time. As for the uncertain MIoT tasks, we leverage Lyapunov optimization to tackle the unpredictable task arrivals and varying computational resource availability. 
By converting the long-term constraints into short-term constraints, we obtain a set of small-scale optimization problems. Further, considering the heterogeneity of actions and resources of UAVs and vessels, we reformulate the small-scale optimization problem into a Markov game (MG). Moreover, a heterogeneous-agent soft actor-critic is proposed to sequentially update various neural networks and effectively solve the MG problem. 
Finally, simulations are conducted to verify the effectiveness in addressing computational offloading and resource allocation. 

---
# Multi-Agent Reinforcement Learning for Autonomous Multi-Satellite Earth Observation: A Realistic Case Study 

**Authors**: Mohamad A. Hady, Siyi Hu, Mahardhika Pratama, Jimmy Cao, Ryszard Kowalczyk  

**Link**: [PDF](https://arxiv.org/pdf/2506.15207)  

**Abstract**: The exponential growth of Low Earth Orbit (LEO) satellites has revolutionised Earth Observation (EO) missions, addressing challenges in climate monitoring, disaster management, and more. However, autonomous coordination in multi-satellite systems remains a fundamental challenge. Traditional optimisation approaches struggle to handle the real-time decision-making demands of dynamic EO missions, necessitating the use of Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL). In this paper, we investigate RL-based autonomous EO mission planning by modelling single-satellite operations and extending to multi-satellite constellations using MARL frameworks. We address key challenges, including energy and data storage limitations, uncertainties in satellite observations, and the complexities of decentralised coordination under partial observability. By leveraging a near-realistic satellite simulation environment, we evaluate the training stability and performance of state-of-the-art MARL algorithms, including PPO, IPPO, MAPPO, and HAPPO. Our results demonstrate that MARL can effectively balance imaging and resource management while addressing non-stationarity and reward interdependency in multi-satellite coordination. The insights gained from this study provide a foundation for autonomous satellite operations, offering practical guidelines for improving policy learning in decentralised EO missions. 

---
# HeurAgenix: Leveraging LLMs for Solving Complex Combinatorial Optimization Challenges 

**Authors**: Xianliang Yang, Ling Zhang, Haolong Qian, Lei Song, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2506.15196)  

**Abstract**: Heuristic algorithms play a vital role in solving combinatorial optimization (CO) problems, yet traditional designs depend heavily on manual expertise and struggle to generalize across diverse instances. We introduce \textbf{HeurAgenix}, a two-stage hyper-heuristic framework powered by large language models (LLMs) that first evolves heuristics and then selects among them automatically. In the heuristic evolution phase, HeurAgenix leverages an LLM to compare seed heuristic solutions with higher-quality solutions and extract reusable evolution strategies. During problem solving, it dynamically picks the most promising heuristic for each problem state, guided by the LLM's perception ability. For flexibility, this selector can be either a state-of-the-art LLM or a fine-tuned lightweight model with lower inference cost. To mitigate the scarcity of reliable supervision caused by CO complexity, we fine-tune the lightweight heuristic selector with a dual-reward mechanism that jointly exploits singals from selection preferences and state perception, enabling robust selection under noisy annotations. Extensive experiments on canonical benchmarks show that HeurAgenix not only outperforms existing LLM-based hyper-heuristics but also matches or exceeds specialized solvers. Code is available at this https URL. 

---
# Truncated Proximal Policy Optimization 

**Authors**: Tiantian Fan, Lingjun Liu, Yu Yue, Jiaze Chen, Chengyi Wang, Qiying Yu, Chi Zhang, Zhiqi Lin, Ruofei Zhu, Yufeng Yuan, Xiaochen Zuo, Bole Ma, Mofan Zhang, Gaohong Liu, Ru Zhang, Haotian Zhou, Cong Xie, Ruidong Zhu, Zhi Zhang, Xin Liu, Mingxuan Wang, Lin Yan, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15050)  

**Abstract**: Recently, test-time scaling Large Language Models (LLMs) have demonstrated exceptional reasoning capabilities across scientific and professional tasks by generating long chains-of-thought (CoT). As a crucial component for developing these reasoning models, reinforcement learning (RL), exemplified by Proximal Policy Optimization (PPO) and its variants, allows models to learn through trial and error. However, PPO can be time-consuming due to its inherent on-policy nature, which is further exacerbated by increasing response lengths. In this work, we propose Truncated Proximal Policy Optimization (T-PPO), a novel extension to PPO that improves training efficiency by streamlining policy update and length-restricted response generation. T-PPO mitigates the issue of low hardware utilization, an inherent drawback of fully synchronized long-generation procedures, where resources often sit idle during the waiting periods for complete rollouts. Our contributions are two-folds. First, we propose Extended Generalized Advantage Estimation (EGAE) for advantage estimation derived from incomplete responses while maintaining the integrity of policy learning. Second, we devise a computationally optimized mechanism that allows for the independent optimization of the policy and value models. By selectively filtering prompt and truncated tokens, this mechanism reduces redundant computations and accelerates the training process without sacrificing convergence performance. We demonstrate the effectiveness and efficacy of T-PPO on AIME 2024 with a 32B base model. The experimental results show that T-PPO improves the training efficiency of reasoning LLMs by up to 2.5x and outperforms its existing competitors. 

---
# MEAL: A Benchmark for Continual Multi-Agent Reinforcement Learning 

**Authors**: Tristan Tomilin, Luka van den Boogaard, Samuel Garcin, Bram Grooten, Meng Fang, Mykola Pechenizkiy  

**Link**: [PDF](https://arxiv.org/pdf/2506.14990)  

**Abstract**: Benchmarks play a crucial role in the development and analysis of reinforcement learning (RL) algorithms, with environment availability strongly impacting research. One particularly underexplored intersection is continual learning (CL) in cooperative multi-agent settings. To remedy this, we introduce MEAL (Multi-agent Environments for Adaptive Learning), the first benchmark tailored for continual multi-agent reinforcement learning (CMARL). Existing CL benchmarks run environments on the CPU, leading to computational bottlenecks and limiting the length of task sequences. MEAL leverages JAX for GPU acceleration, enabling continual learning across sequences of 100 tasks on a standard desktop PC in a few hours. We show that naively combining popular CL and MARL methods yields strong performance on simple environments, but fails to scale to more complex settings requiring sustained coordination and adaptation. Our ablation study identifies architectural and algorithmic features critical for CMARL on MEAL. 

---
# CALM: Contextual Analog Logic with Multimodality 

**Authors**: Maxwell J. Jacobson, Corey J. Maley, Yexiang Xue  

**Link**: [PDF](https://arxiv.org/pdf/2506.14936)  

**Abstract**: In this work, we introduce Contextual Analog Logic with Multimodality (CALM). CALM unites symbolic reasoning with neural generation, enabling systems to make context-sensitive decisions grounded in real-world multi-modal data.
Background: Classic bivalent logic systems cannot capture the nuance of human decision-making. They also require human grounding in multi-modal environments, which can be ad-hoc, rigid, and brittle. Neural networks are good at extracting rich contextual information from multi-modal data, but lack interpretable structures for reasoning.
Objectives: CALM aims to bridge the gap between logic and neural perception, creating an analog logic that can reason over multi-modal inputs. Without this integration, AI systems remain either brittle or unstructured, unable to generalize robustly to real-world tasks. In CALM, symbolic predicates evaluate to analog truth values computed by neural networks and constrained search.
Methods: CALM represents each predicate using a domain tree, which iteratively refines its analog truth value when the contextual groundings of its entities are determined. The iterative refinement is predicted by neural networks capable of capturing multi-modal information and is filtered through a symbolic reasoning module to ensure constraint satisfaction.
Results: In fill-in-the-blank object placement tasks, CALM achieved 92.2% accuracy, outperforming classical logic (86.3%) and LLM (59.4%) baselines. It also demonstrated spatial heatmap generation aligned with logical constraints and delicate human preferences, as shown by a human study.
Conclusions: CALM demonstrates the potential to reason with logic structure while aligning with preferences in multi-modal environments. It lays the foundation for next-gen AI systems that require the precision and interpretation of logic and the multimodal information processing of neural networks. 

---
# Dense SAE Latents Are Features, Not Bugs 

**Authors**: Xiaoqing Sun, Alessandro Stolfo, Joshua Engels, Ben Wu, Senthooran Rajamanoharan, Mrinmaya Sachan, Max Tegmark  

**Link**: [PDF](https://arxiv.org/pdf/2506.15679)  

**Abstract**: Sparse autoencoders (SAEs) are designed to extract interpretable features from language models by enforcing a sparsity constraint. Ideally, training an SAE would yield latents that are both sparse and semantically meaningful. However, many SAE latents activate frequently (i.e., are \emph{dense}), raising concerns that they may be undesirable artifacts of the training procedure. In this work, we systematically investigate the geometry, function, and origin of dense latents and show that they are not only persistent but often reflect meaningful model representations. We first demonstrate that dense latents tend to form antipodal pairs that reconstruct specific directions in the residual stream, and that ablating their subspace suppresses the emergence of new dense features in retrained SAEs -- suggesting that high density features are an intrinsic property of the residual space. We then introduce a taxonomy of dense latents, identifying classes tied to position tracking, context binding, entropy regulation, letter-specific output signals, part-of-speech, and principal component reconstruction. Finally, we analyze how these features evolve across layers, revealing a shift from structural features in early layers, to semantic features in mid layers, and finally to output-oriented signals in the last layers of the model. Our findings indicate that dense latents serve functional roles in language model computation and should not be dismissed as training noise. 

---
# Sekai: A Video Dataset towards World Exploration 

**Authors**: Zhen Li, Chuanhao Li, Xiaofeng Mao, Shaoheng Lin, Ming Li, Shitian Zhao, Zhaopan Xu, Xinyue Li, Yukang Feng, Jianwen Sun, Zizhen Li, Fanrui Zhang, Jiaxin Ai, Zhixiang Wang, Yuwei Wu, Tong He, Jiangmiao Pang, Yu Qiao, Yunde Jia, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15675)  

**Abstract**: Video generation techniques have made remarkable progress, promising to be the foundation of interactive world exploration. However, existing video generation datasets are not well-suited for world exploration training as they suffer from some limitations: limited locations, short duration, static scenes, and a lack of annotations about exploration and the world. In this paper, we introduce Sekai (meaning ``world'' in Japanese), a high-quality first-person view worldwide video dataset with rich annotations for world exploration. It consists of over 5,000 hours of walking or drone view (FPV and UVA) videos from over 100 countries and regions across 750 cities. We develop an efficient and effective toolbox to collect, pre-process and annotate videos with location, scene, weather, crowd density, captions, and camera trajectories. Experiments demonstrate the quality of the dataset. And, we use a subset to train an interactive video world exploration model, named YUME (meaning ``dream'' in Japanese). We believe Sekai will benefit the area of video generation and world exploration, and motivate valuable applications. 

---
# Leaky Thoughts: Large Reasoning Models Are Not Private Thinkers 

**Authors**: Tommaso Green, Martin Gubri, Haritz Puerto, Sangdoo Yun, Seong Joon Oh  

**Link**: [PDF](https://arxiv.org/pdf/2506.15674)  

**Abstract**: We study privacy leakage in the reasoning traces of large reasoning models used as personal agents. Unlike final outputs, reasoning traces are often assumed to be internal and safe. We challenge this assumption by showing that reasoning traces frequently contain sensitive user data, which can be extracted via prompt injections or accidentally leak into outputs. Through probing and agentic evaluations, we demonstrate that test-time compute approaches, particularly increased reasoning steps, amplify such leakage. While increasing the budget of those test-time compute approaches makes models more cautious in their final answers, it also leads them to reason more verbosely and leak more in their own thinking. This reveals a core tension: reasoning improves utility but enlarges the privacy attack surface. We argue that safety efforts must extend to the model's internal thinking, not just its outputs. 

---
# AutoRule: Reasoning Chain-of-thought Extracted Rule-based Rewards Improve Preference Learning 

**Authors**: Tevin Wang, Chenyan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.15651)  

**Abstract**: Rule-based rewards offer a promising strategy for improving reinforcement learning from human feedback (RLHF), but current approaches often rely on manual rule engineering. We present AutoRule, a fully automated method for extracting rules from preference feedback and formulating them into rule-based rewards. AutoRule extraction operates in three stages: it leverages a reasoning model to interpret user preferences, identifies candidate rules from the reasoning chain of these interpretations, and synthesizes them into a unified rule set. Leveraging the finalized rule set, we employ language-model verifiers to compute the fraction of rules satisfied by each output, using this metric as an auxiliary reward alongside the learned reward model during policy optimization. Training a Llama-3-8B model with AutoRule results in a 28.6\% relative improvement in length-controlled win rate on AlpacaEval2.0, and a 6.1\% relative gain in second-turn performance on a held-out MT-Bench subset, compared to a GRPO baseline trained with the same learned reward model but without the rule-based auxiliary reward. Our analysis confirms that the extracted rules exhibit good agreement with dataset preference. We find that AutoRule demonstrates reduced reward hacking compared to a learned reward model when run over two episodes. Finally, our case study suggests that the extracted rules capture unique qualities valued in different datasets. The extracted rules are provided in the appendix, and the code is open-sourced at this https URL. 

---
# Demystifying the Visual Quality Paradox in Multimodal Large Language Models 

**Authors**: Shuo Xing, Lanqing Guo, Hongyuan Hua, Seoyoung Lee, Peiran Li, Yufei Wang, Zhangyang Wang, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15645)  

**Abstract**: Recent Multimodal Large Language Models (MLLMs) excel on benchmark vision-language tasks, yet little is known about how input visual quality shapes their responses. Does higher perceptual quality of images already translate to better MLLM understanding? We conduct the first systematic study spanning leading MLLMs and a suite of vision-language benchmarks, applying controlled degradations and stylistic shifts to each image. Surprisingly, we uncover a visual-quality paradox: model, task, and even individual-instance performance can improve when images deviate from human-perceived fidelity. Off-the-shelf restoration pipelines fail to reconcile these idiosyncratic preferences. To close the gap, we introduce Visual-Quality Test-Time Tuning (VQ-TTT)-a lightweight adaptation module that: (1) inserts a learnable, low-rank kernel before the frozen vision encoder to modulate frequency content; and (2) fine-tunes only shallow vision-encoder layers via LoRA. VQ-TTT dynamically adjusts each input image in a single forward pass, aligning it with task-specific model preferences. Across the evaluated MLLMs and all datasets, VQ-TTT lifts significant average accuracy, with no external models, cached features, or extra training data. These findings redefine ``better'' visual inputs for MLLMs and highlight the need for adaptive, rather than universally ``clean'', imagery, in the new era of AI being the main data customer. 

---
# Revisiting Compositional Generalization Capability of Large Language Models Considering Instruction Following Ability 

**Authors**: Yusuke Sakai, Hidetaka Kamigaito, Taro Watanabe  

**Link**: [PDF](https://arxiv.org/pdf/2506.15629)  

**Abstract**: In generative commonsense reasoning tasks such as CommonGen, generative large language models (LLMs) compose sentences that include all given concepts. However, when focusing on instruction-following capabilities, if a prompt specifies a concept order, LLMs must generate sentences that adhere to the specified order. To address this, we propose Ordered CommonGen, a benchmark designed to evaluate the compositional generalization and instruction-following abilities of LLMs. This benchmark measures ordered coverage to assess whether concepts are generated in the specified order, enabling a simultaneous evaluation of both abilities. We conducted a comprehensive analysis using 36 LLMs and found that, while LLMs generally understand the intent of instructions, biases toward specific concept order patterns often lead to low-diversity outputs or identical results even when the concept order is altered. Moreover, even the most instruction-compliant LLM achieved only about 75% ordered coverage, highlighting the need for improvements in both instruction-following and compositional generalization capabilities. 

---
# Federated Learning for MRI-based BrainAGE: a multicenter study on post-stroke functional outcome prediction 

**Authors**: Vincent Roca, Marc Tommasi, Paul Andrey, Aurélien Bellet, Markus D. Schirmer, Hilde Henon, Laurent Puy, Julien Ramon, Grégory Kuchcinski, Martin Bretzner, Renaud Lopes  

**Link**: [PDF](https://arxiv.org/pdf/2506.15626)  

**Abstract**: $\textbf{Objective:}$ Brain-predicted age difference (BrainAGE) is a neuroimaging biomarker reflecting brain health. However, training robust BrainAGE models requires large datasets, often restricted by privacy concerns. This study evaluates the performance of federated learning (FL) for BrainAGE estimation in ischemic stroke patients treated with mechanical thrombectomy, and investigates its association with clinical phenotypes and functional outcomes.
$\textbf{Methods:}$ We used FLAIR brain images from 1674 stroke patients across 16 hospital centers. We implemented standard machine learning and deep learning models for BrainAGE estimates under three data management strategies: centralized learning (pooled data), FL (local training at each site), and single-site learning. We reported prediction errors and examined associations between BrainAGE and vascular risk factors (e.g., diabetes mellitus, hypertension, smoking), as well as functional outcomes at three months post-stroke. Logistic regression evaluated BrainAGE's predictive value for these outcomes, adjusting for age, sex, vascular risk factors, stroke severity, time between MRI and arterial puncture, prior intravenous thrombolysis, and recanalisation outcome.
$\textbf{Results:}$ While centralized learning yielded the most accurate predictions, FL consistently outperformed single-site models. BrainAGE was significantly higher in patients with diabetes mellitus across all models. Comparisons between patients with good and poor functional outcomes, and multivariate predictions of these outcomes showed the significance of the association between BrainAGE and post-stroke recovery.
$\textbf{Conclusion:}$ FL enables accurate age predictions without data centralization. The strong association between BrainAGE, vascular risk factors, and post-stroke recovery highlights its potential for prognostic modeling in stroke care. 

---
# GFLC: Graph-based Fairness-aware Label Correction for Fair Classification 

**Authors**: Modar Sulaiman, Kallol Roy  

**Link**: [PDF](https://arxiv.org/pdf/2506.15620)  

**Abstract**: Fairness in machine learning (ML) has a critical importance for building trustworthy machine learning system as artificial intelligence (AI) systems increasingly impact various aspects of society, including healthcare decisions and legal judgments. Moreover, numerous studies demonstrate evidence of unfair outcomes in ML and the need for more robust fairness-aware methods. However, the data we use to train and develop debiasing techniques often contains biased and noisy labels. As a result, the label bias in the training data affects model performance and misrepresents the fairness of classifiers during testing. To tackle this problem, our paper presents Graph-based Fairness-aware Label Correction (GFLC), an efficient method for correcting label noise while preserving demographic parity in datasets. In particular, our approach combines three key components: prediction confidence measure, graph-based regularization through Ricci-flow-optimized graph Laplacians, and explicit demographic parity incentives. Our experimental findings show the effectiveness of our proposed approach and show significant improvements in the trade-off between performance and fairness metrics compared to the baseline. 

---
# The Compositional Architecture of Regret in Large Language Models 

**Authors**: Xiangxiang Cui, Shu Yang, Tianjin Huang, Wanyu Lin, Lijie Hu, Di Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15617)  

**Abstract**: Regret in Large Language Models refers to their explicit regret expression when presented with evidence contradicting their previously generated misinformation. Studying the regret mechanism is crucial for enhancing model reliability and helps in revealing how cognition is coded in neural networks. To understand this mechanism, we need to first identify regret expressions in model outputs, then analyze their internal representation. This analysis requires examining the model's hidden states, where information processing occurs at the neuron level. However, this faces three key challenges: (1) the absence of specialized datasets capturing regret expressions, (2) the lack of metrics to find the optimal regret representation layer, and (3) the lack of metrics for identifying and analyzing regret neurons. Addressing these limitations, we propose: (1) a workflow for constructing a comprehensive regret dataset through strategically designed prompting scenarios, (2) the Supervised Compression-Decoupling Index (S-CDI) metric to identify optimal regret representation layers, and (3) the Regret Dominance Score (RDS) metric to identify regret neurons and the Group Impact Coefficient (GIC) to analyze activation patterns. Our experimental results successfully identified the optimal regret representation layer using the S-CDI metric, which significantly enhanced performance in probe classification experiments. Additionally, we discovered an M-shaped decoupling pattern across model layers, revealing how information processing alternates between coupling and decoupling phases. Through the RDS metric, we categorized neurons into three distinct functional groups: regret neurons, non-regret neurons, and dual neurons. 

---
# LoX: Low-Rank Extrapolation Robustifies LLM Safety Against Fine-tuning 

**Authors**: Gabrel J. Perin, Runjin Chen, Xuxi Chen, Nina S. T. Hirata, Zhangyang Wang, Junyuan Hong  

**Link**: [PDF](https://arxiv.org/pdf/2506.15606)  

**Abstract**: Large Language Models (LLMs) have become indispensable in real-world applications. However, their widespread adoption raises significant safety concerns, particularly in responding to socially harmful questions. Despite substantial efforts to improve model safety through alignment, aligned models can still have their safety protections undermined by subsequent fine-tuning - even when the additional training data appears benign. In this paper, we empirically demonstrate that this vulnerability stems from the sensitivity of safety-critical low-rank subspaces in LLM parameters to fine-tuning. Building on this insight, we propose a novel training-free method, termed Low-Rank Extrapolation (LoX), to enhance safety robustness by extrapolating the safety subspace of an aligned LLM. Our experimental results confirm the effectiveness of LoX, demonstrating significant improvements in robustness against both benign and malicious fine-tuning attacks while preserving the model's adaptability to new tasks. For instance, LoX leads to 11% to 54% absolute reductions in attack success rates (ASR) facing benign or malicious fine-tuning attacks. By investigating the ASR landscape of parameters, we attribute the success of LoX to that the extrapolation moves LLM parameters to a flatter zone, thereby less sensitive to perturbations. The code is available at this http URL. 

---
# From Model to Classroom: Evaluating Generated MCQs for Portuguese with Narrative and Difficulty Concerns 

**Authors**: Bernardo Leite, Henrique Lopes Cardoso, Pedro Pinto, Abel Ferreira, Luís Abreu, Isabel Rangel, Sandra Monteiro  

**Link**: [PDF](https://arxiv.org/pdf/2506.15598)  

**Abstract**: While MCQs are valuable for learning and evaluation, manually creating them with varying difficulty levels and targeted reading skills remains a time-consuming and costly task. Recent advances in generative AI provide an opportunity to automate MCQ generation efficiently. However, assessing the actual quality and reliability of generated MCQs has received limited attention -- particularly regarding cases where generation fails. This aspect becomes particularly important when the generated MCQs are meant to be applied in real-world settings. Additionally, most MCQ generation studies focus on English, leaving other languages underexplored. This paper investigates the capabilities of current generative models in producing MCQs for reading comprehension in Portuguese, a morphologically rich language. Our study focuses on generating MCQs that align with curriculum-relevant narrative elements and span different difficulty levels. We evaluate these MCQs through expert review and by analyzing the psychometric properties extracted from student responses to assess their suitability for elementary school students. Our results show that current models can generate MCQs of comparable quality to human-authored ones. However, we identify issues related to semantic clarity and answerability. Also, challenges remain in generating distractors that engage students and meet established criteria for high-quality MCQ option design. 

---
# WikiMixQA: A Multimodal Benchmark for Question Answering over Tables and Charts 

**Authors**: Negar Foroutan, Angelika Romanou, Matin Ansaripour, Julian Martin Eisenschlos, Karl Aberer, Rémi Lebret  

**Link**: [PDF](https://arxiv.org/pdf/2506.15594)  

**Abstract**: Documents are fundamental to preserving and disseminating information, often incorporating complex layouts, tables, and charts that pose significant challenges for automatic document understanding (DU). While vision-language large models (VLLMs) have demonstrated improvements across various tasks, their effectiveness in processing long-context vision inputs remains unclear. This paper introduces WikiMixQA, a benchmark comprising 1,000 multiple-choice questions (MCQs) designed to evaluate cross-modal reasoning over tables and charts extracted from 4,000 Wikipedia pages spanning seven distinct topics. Unlike existing benchmarks, WikiMixQA emphasizes complex reasoning by requiring models to synthesize information from multiple modalities. We evaluate 12 state-of-the-art vision-language models, revealing that while proprietary models achieve ~70% accuracy when provided with direct context, their performance deteriorates significantly when retrieval from long documents is required. Among these, GPT-4-o is the only model exceeding 50% accuracy in this setting, whereas open-source models perform considerably worse, with a maximum accuracy of 27%. These findings underscore the challenges of long-context, multi-modal reasoning and establish WikiMixQA as a crucial benchmark for advancing document understanding research. 

---
# One-Step Diffusion for Detail-Rich and Temporally Consistent Video Super-Resolution 

**Authors**: Yujing Sun, Lingchen Sun, Shuaizheng Liu, Rongyuan Wu, Zhengqiang Zhang, Lei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15591)  

**Abstract**: It is a challenging problem to reproduce rich spatial details while maintaining temporal consistency in real-world video super-resolution (Real-VSR), especially when we leverage pre-trained generative models such as stable diffusion (SD) for realistic details synthesis. Existing SD-based Real-VSR methods often compromise spatial details for temporal coherence, resulting in suboptimal visual quality. We argue that the key lies in how to effectively extract the degradation-robust temporal consistency priors from the low-quality (LQ) input video and enhance the video details while maintaining the extracted consistency priors. To achieve this, we propose a Dual LoRA Learning (DLoRAL) paradigm to train an effective SD-based one-step diffusion model, achieving realistic frame details and temporal consistency simultaneously. Specifically, we introduce a Cross-Frame Retrieval (CFR) module to aggregate complementary information across frames, and train a Consistency-LoRA (C-LoRA) to learn robust temporal representations from degraded inputs. After consistency learning, we fix the CFR and C-LoRA modules and train a Detail-LoRA (D-LoRA) to enhance spatial details while aligning with the temporal space defined by C-LoRA to keep temporal coherence. The two phases alternate iteratively for optimization, collaboratively delivering consistent and detail-rich outputs. During inference, the two LoRA branches are merged into the SD model, allowing efficient and high-quality video restoration in a single diffusion step. Experiments show that DLoRAL achieves strong performance in both accuracy and speed. Code and models are available at this https URL. 

---
# Towards Explainable Indoor Localization: Interpreting Neural Network Learning on Wi-Fi Fingerprints Using Logic Gates 

**Authors**: Danish Gufran, Sudeep Pasricha  

**Link**: [PDF](https://arxiv.org/pdf/2506.15559)  

**Abstract**: Indoor localization using deep learning (DL) has demonstrated strong accuracy in mapping Wi-Fi RSS fingerprints to physical locations; however, most existing DL frameworks function as black-box models, offering limited insight into how predictions are made or how models respond to real-world noise over time. This lack of interpretability hampers our ability to understand the impact of temporal variations - caused by environmental dynamics - and to adapt models for long-term reliability. To address this, we introduce LogNet, a novel logic gate-based framework designed to interpret and enhance DL-based indoor localization. LogNet enables transparent reasoning by identifying which access points (APs) are most influential for each reference point (RP) and reveals how environmental noise disrupts DL-driven localization decisions. This interpretability allows us to trace and diagnose model failures and adapt DL systems for more stable long-term deployments. Evaluations across multiple real-world building floorplans and over two years of temporal variation show that LogNet not only interprets the internal behavior of DL models but also improves performance-achieving up to 1.1x to 2.8x lower localization error, 3.4x to 43.3x smaller model size, and 1.5x to 3.6x lower latency compared to prior DL-based models. 

---
# DAILOC: Domain-Incremental Learning for Indoor Localization using Smartphones 

**Authors**: Akhil Singampalli, Danish Gufran, Sudeep Pasricha  

**Link**: [PDF](https://arxiv.org/pdf/2506.15554)  

**Abstract**: Wi-Fi fingerprinting-based indoor localization faces significant challenges in real-world deployments due to domain shifts arising from device heterogeneity and temporal variations within indoor environments. Existing approaches often address these issues independently, resulting in poor generalization and susceptibility to catastrophic forgetting over time. In this work, we propose DAILOC, a novel domain-incremental learning framework that jointly addresses both temporal and device-induced domain shifts. DAILOC introduces a novel disentanglement strategy that separates domain shifts from location-relevant features using a multi-level variational autoencoder. Additionally, we introduce a novel memory-guided class latent alignment mechanism to address the effects of catastrophic forgetting over time. Experiments across multiple smartphones, buildings, and time instances demonstrate that DAILOC significantly outperforms state-of-the-art methods, achieving up to 2.74x lower average error and 4.6x lower worst-case error. 

---
# CLAIM: Clinically-Guided LGE Augmentation for Realistic and Diverse Myocardial Scar Synthesis and Segmentation 

**Authors**: Farheen Ramzan, Yusuf Kiberu, Nikesh Jathanna, Shahnaz Jamil-Copley, Richard H. Clayton, Chen, Chen  

**Link**: [PDF](https://arxiv.org/pdf/2506.15549)  

**Abstract**: Deep learning-based myocardial scar segmentation from late gadolinium enhancement (LGE) cardiac MRI has shown great potential for accurate and timely diagnosis and treatment planning for structural cardiac diseases. However, the limited availability and variability of LGE images with high-quality scar labels restrict the development of robust segmentation models. To address this, we introduce CLAIM: \textbf{C}linically-Guided \textbf{L}GE \textbf{A}ugmentation for Real\textbf{i}stic and Diverse \textbf{M}yocardial Scar Synthesis and Segmentation framework, a framework for anatomically grounded scar generation and segmentation. At its core is the SMILE module (Scar Mask generation guided by cLinical knowledgE), which conditions a diffusion-based generator on the clinically adopted AHA 17-segment model to synthesize images with anatomically consistent and spatially diverse scar patterns. In addition, CLAIM employs a joint training strategy in which the scar segmentation network is optimized alongside the generator, aiming to enhance both the realism of synthesized scars and the accuracy of the scar segmentation performance. Experimental results show that CLAIM produces anatomically coherent scar patterns and achieves higher Dice similarity with real scar distributions compared to baseline models. Our approach enables controllable and realistic myocardial scar synthesis and has demonstrated utility for downstream medical imaging task. 

---
# Learning Algorithms in the Limit 

**Authors**: Hristo Papazov, Nicolas Flammarion  

**Link**: [PDF](https://arxiv.org/pdf/2506.15543)  

**Abstract**: This paper studies the problem of learning computable functions in the limit by extending Gold's inductive inference framework to incorporate \textit{computational observations} and \textit{restricted input sources}. Complimentary to the traditional Input-Output Observations, we introduce Time-Bound Observations, and Policy-Trajectory Observations to study the learnability of general recursive functions under more realistic constraints. While input-output observations do not suffice for learning the class of general recursive functions in the limit, we overcome this learning barrier by imposing computational complexity constraints or supplementing with approximate time-bound observations. Further, we build a formal framework around observations of \textit{computational agents} and show that learning computable functions from policy trajectories reduces to learning rational functions from input and output, thereby revealing interesting connections to finite-state transducer inference. On the negative side, we show that computable or polynomial-mass characteristic sets cannot exist for the class of linear-time computable functions even for policy-trajectory observations. 

---
# Intrinsic and Extrinsic Organized Attention: Softmax Invariance and Network Sparsity 

**Authors**: Oluwadamilola Fasina, Ruben V.C. Pohle, Pei-Chun Su, Ronald R. Coifman  

**Link**: [PDF](https://arxiv.org/pdf/2506.15541)  

**Abstract**: We examine the intrinsic (within the attention head) and extrinsic (amongst the attention heads) structure of the self-attention mechanism in transformers. Theoretical evidence for invariance of the self-attention mechanism to softmax activation is obtained by appealing to paradifferential calculus, (and is supported by computational examples), which relies on the intrinsic organization of the attention heads. Furthermore, we use an existing methodology for hierarchical organization of tensors to examine network structure by constructing hierarchal partition trees with respect to the query, key, and head axes of network 3-tensors. Such an organization is consequential since it allows one to profitably execute common signal processing tasks on a geometry where the organized network 3-tensors exhibit regularity. We exemplify this qualitatively, by visualizing the hierarchical organization of the tree comprised of attention heads and the diffusion map embeddings, and quantitatively by investigating network sparsity with the expansion coefficients of individual attention heads and the entire network with respect to the bi and tri-haar bases (respectively) on the space of queries, keys, and heads of the network. To showcase the utility of our theoretical and methodological findings, we provide computational examples using vision and language transformers. The ramifications of these findings are two-fold: (1) a subsequent step in interpretability analysis is theoretically admitted, and can be exploited empirically for downstream interpretability tasks (2) one can use the network 3-tensor organization for empirical network applications such as model pruning (by virtue of network sparsity) and network architecture comparison. 

---
# Capturing Polysemanticity with PRISM: A Multi-Concept Feature Description Framework 

**Authors**: Laura Kopf, Nils Feldhus, Kirill Bykov, Philine Lou Bommer, Anna Hedström, Marina M.-C. Höhne, Oliver Eberle  

**Link**: [PDF](https://arxiv.org/pdf/2506.15538)  

**Abstract**: Automated interpretability research aims to identify concepts encoded in neural network features to enhance human understanding of model behavior. Current feature description methods face two critical challenges: limited robustness and the flawed assumption that each neuron encodes only a single concept (monosemanticity), despite growing evidence that neurons are often polysemantic. This assumption restricts the expressiveness of feature descriptions and limits their ability to capture the full range of behaviors encoded in model internals. To address this, we introduce Polysemantic FeatuRe Identification and Scoring Method (PRISM), a novel framework that captures the inherent complexity of neural network features. Unlike prior approaches that assign a single description per feature, PRISM provides more nuanced descriptions for both polysemantic and monosemantic features. We apply PRISM to language models and, through extensive benchmarking against existing methods, demonstrate that our approach produces more accurate and faithful feature descriptions, improving both overall description quality (via a description score) and the ability to capture distinct concepts when polysemanticity is present (via a polysemanticity score). 

---
# RePCS: Diagnosing Data Memorization in LLM-Powered Retrieval-Augmented Generation 

**Authors**: Le Vu Anh, Nguyen Viet Anh, Mehmet Dik, Luong Van Nghia  

**Link**: [PDF](https://arxiv.org/pdf/2506.15513)  

**Abstract**: Retrieval-augmented generation (RAG) has become a common strategy for updating large language model (LLM) responses with current, external information. However, models may still rely on memorized training data, bypass the retrieved evidence, and produce contaminated outputs. We introduce Retrieval-Path Contamination Scoring (RePCS), a diagnostic method that detects such behavior without requiring model access or retraining. RePCS compares two inference paths: (i) a parametric path using only the query, and (ii) a retrieval-augmented path using both the query and retrieved context by computing the Kullback-Leibler (KL) divergence between their output distributions. A low divergence suggests that the retrieved context had minimal impact, indicating potential memorization. This procedure is model-agnostic, requires no gradient or internal state access, and adds only a single additional forward pass. We further derive PAC-style guarantees that link the KL threshold to user-defined false positive and false negative rates. On the Prompt-WNQA benchmark, RePCS achieves a ROC-AUC of 0.918. This result outperforms the strongest prior method by 6.5 percentage points while keeping latency overhead below 4.7% on an NVIDIA T4 GPU. RePCS offers a lightweight, black-box safeguard to verify whether a RAG system meaningfully leverages retrieval, making it especially valuable in safety-critical applications. 

---
# Optimizing Web-Based AI Query Retrieval with GPT Integration in LangChain A CoT-Enhanced Prompt Engineering Approach 

**Authors**: Wenqi Guan, Yang Fang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15512)  

**Abstract**: Large Language Models have brought a radical change in the process of remote learning students, among other aspects of educative activities. Current retrieval of remote learning resources lacks depth in contextual meaning that provides comprehensive information on complex student queries. This work proposes a novel approach to enhancing remote learning retrieval by integrating GPT-based models within the LangChain framework. We achieve this system in a more intuitive and productive manner using CoT reasoning and prompt engineering. The framework we propose puts much emphasis on increasing the precision and relevance of the retrieval results to return comprehensive and contextually enriched explanations and resources that best suit each student's needs. We also assess the effectiveness of our approach against paradigmatic LLMs and report improvements in user satisfaction and learning outcomes. 

---
# Over-squashing in Spatiotemporal Graph Neural Networks 

**Authors**: Ivan Marisca, Jacob Bamberger, Cesare Alippi, Michael M. Bronstein  

**Link**: [PDF](https://arxiv.org/pdf/2506.15507)  

**Abstract**: Graph Neural Networks (GNNs) have achieved remarkable success across various domains. However, recent theoretical advances have identified fundamental limitations in their information propagation capabilities, such as over-squashing, where distant nodes fail to effectively exchange information. While extensively studied in static contexts, this issue remains unexplored in Spatiotemporal GNNs (STGNNs), which process sequences associated with graph nodes. Nonetheless, the temporal dimension amplifies this challenge by increasing the information that must be propagated. In this work, we formalize the spatiotemporal over-squashing problem and demonstrate its distinct characteristics compared to the static case. Our analysis reveals that counterintuitively, convolutional STGNNs favor information propagation from points temporally distant rather than close in time. Moreover, we prove that architectures that follow either time-and-space or time-then-space processing paradigms are equally affected by this phenomenon, providing theoretical justification for computationally efficient implementations. We validate our findings on synthetic and real-world datasets, providing deeper insights into their operational dynamics and principled guidance for more effective designs. 

---
# Pixel-level Certified Explanations via Randomized Smoothing 

**Authors**: Alaa Anani, Tobias Lorenz, Mario Fritz, Bernt Schiele  

**Link**: [PDF](https://arxiv.org/pdf/2506.15499)  

**Abstract**: Post-hoc attribution methods aim to explain deep learning predictions by highlighting influential input pixels. However, these explanations are highly non-robust: small, imperceptible input perturbations can drastically alter the attribution map while maintaining the same prediction. This vulnerability undermines their trustworthiness and calls for rigorous robustness guarantees of pixel-level attribution scores. We introduce the first certification framework that guarantees pixel-level robustness for any black-box attribution method using randomized smoothing. By sparsifying and smoothing attribution maps, we reformulate the task as a segmentation problem and certify each pixel's importance against $\ell_2$-bounded perturbations. We further propose three evaluation metrics to assess certified robustness, localization, and faithfulness. An extensive evaluation of 12 attribution methods across 5 ImageNet models shows that our certified attributions are robust, interpretable, and faithful, enabling reliable use in downstream tasks. Our code is at this https URL. 

---
# SPARE: Single-Pass Annotation with Reference-Guided Evaluation for Automatic Process Supervision and Reward Modelling 

**Authors**: Md Imbesat Hassan Rizvi, Xiaodan Zhu, Iryna Gurevych  

**Link**: [PDF](https://arxiv.org/pdf/2506.15498)  

**Abstract**: Process or step-wise supervision has played a crucial role in advancing complex multi-step reasoning capabilities of Large Language Models (LLMs). However, efficient, high-quality automated process annotation remains a significant challenge. To address this, we introduce Single-Pass Annotation with Reference-Guided Evaluation (SPARE), a novel structured framework that enables single-pass, per-step annotation by aligning each solution step to one or multiple steps in a reference solution, accompanied by explicit reasoning for evaluation. We show that reference-guided step-level evaluation effectively facilitates process supervision on four datasets spanning three domains: mathematical reasoning, multi-hop compositional question answering, and spatial reasoning. We demonstrate that SPARE, when compared to baselines, improves reasoning performance when used for: (1) fine-tuning models in an offline RL setup for inference-time greedy-decoding, and (2) training reward models for ranking/aggregating multiple LLM-generated outputs. Additionally, SPARE achieves competitive performance on challenging mathematical datasets while offering 2.6 times greater efficiency, requiring only 38% of the runtime, compared to tree search-based automatic annotation. The codebase, along with a trained SPARE-PRM model, is publicly released to facilitate further research and reproducibility. 

---
# GenHOI: Generalizing Text-driven 4D Human-Object Interaction Synthesis for Unseen Objects 

**Authors**: Shujia Li, Haiyu Zhang, Xinyuan Chen, Yaohui Wang, Yutong Ban  

**Link**: [PDF](https://arxiv.org/pdf/2506.15483)  

**Abstract**: While diffusion models and large-scale motion datasets have advanced text-driven human motion synthesis, extending these advances to 4D human-object interaction (HOI) remains challenging, mainly due to the limited availability of large-scale 4D HOI datasets. In our study, we introduce GenHOI, a novel two-stage framework aimed at achieving two key objectives: 1) generalization to unseen objects and 2) the synthesis of high-fidelity 4D HOI sequences. In the initial stage of our framework, we employ an Object-AnchorNet to reconstruct sparse 3D HOI keyframes for unseen objects, learning solely from 3D HOI datasets, thereby mitigating the dependence on large-scale 4D HOI datasets. Subsequently, we introduce a Contact-Aware Diffusion Model (ContactDM) in the second stage to seamlessly interpolate sparse 3D HOI keyframes into densely temporally coherent 4D HOI sequences. To enhance the quality of generated 4D HOI sequences, we propose a novel Contact-Aware Encoder within ContactDM to extract human-object contact patterns and a novel Contact-Aware HOI Attention to effectively integrate the contact signals into diffusion models. Experimental results show that we achieve state-of-the-art results on the publicly available OMOMO and 3D-FUTURE datasets, demonstrating strong generalization abilities to unseen objects, while enabling high-fidelity 4D HOI generation. 

---
# Context-Informed Grounding Supervision 

**Authors**: Hyunji Lee, Seunghyun Yoon, Yunjae Won, Hanseok Oh, Geewook Kim, Trung Bui, Franck Dernoncourt, Elias Stengel-Eskin, Mohit Bansal, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2506.15480)  

**Abstract**: Large language models (LLMs) are often supplemented with external knowledge to provide information not encoded in their parameters or to reduce hallucination. In such cases, we expect the model to generate responses by grounding its response in the provided external context. However, prior work has shown that simply appending context at inference time does not ensure grounded generation. To address this, we propose Context-INformed Grounding Supervision (CINGS), a post-training supervision in which the model is trained with relevant context prepended to the response, while computing the loss only over the response tokens and masking out the context. Our experiments demonstrate that models trained with CINGS exhibit stronger grounding in both textual and visual domains compared to standard instruction-tuned models. In the text domain, CINGS outperforms other training methods across 11 information-seeking datasets and is complementary to inference-time grounding techniques. In the vision-language domain, replacing a vision-language model's LLM backbone with a CINGS-trained model reduces hallucinations across four benchmarks and maintains factual consistency throughout the generated response. This improved grounding comes without degradation in general downstream performance. Finally, we analyze the mechanism underlying the enhanced grounding in CINGS and find that it induces a shift in the model's prior knowledge and behavior, implicitly encouraging greater reliance on the external context. 

---
# Co-Creative Learning via Metropolis-Hastings Interaction between Humans and AI 

**Authors**: Ryota Okumura, Tadahiro Taniguchi, Akira Taniguchi, Yoshinobu Hagiwara  

**Link**: [PDF](https://arxiv.org/pdf/2506.15468)  

**Abstract**: We propose co-creative learning as a novel paradigm where humans and AI, i.e., biological and artificial agents, mutually integrate their partial perceptual information and knowledge to construct shared external representations, a process we interpret as symbol emergence. Unlike traditional AI teaching based on unilateral knowledge transfer, this addresses the challenge of integrating information from inherently different modalities. We empirically test this framework using a human-AI interaction model based on the Metropolis-Hastings naming game (MHNG), a decentralized Bayesian inference mechanism. In an online experiment, 69 participants played a joint attention naming game (JA-NG) with one of three computer agent types (MH-based, always-accept, or always-reject) under partial observability. Results show that human-AI pairs with an MH-based agent significantly improved categorization accuracy through interaction and achieved stronger convergence toward a shared sign system. Furthermore, human acceptance behavior aligned closely with the MH-derived acceptance probability. These findings provide the first empirical evidence for co-creative learning emerging in human-AI dyads via MHNG-based interaction. This suggests a promising path toward symbiotic AI systems that learn with humans, rather than from them, by dynamically aligning perceptual experiences, opening a new venue for symbiotic AI alignment. 

---
# RE-IMAGINE: Symbolic Benchmark Synthesis for Reasoning Evaluation 

**Authors**: Xinnuo Xu, Rachel Lawrence, Kshitij Dubey, Atharva Pandey, Risa Ueno, Fabian Falck, Aditya V. Nori, Rahul Sharma, Amit Sharma, Javier Gonzalez  

**Link**: [PDF](https://arxiv.org/pdf/2506.15455)  

**Abstract**: Recent Large Language Models (LLMs) have reported high accuracy on reasoning benchmarks. However, it is still unclear whether the observed results arise from true reasoning or from statistical recall of the training set. Inspired by the ladder of causation (Pearl, 2009) and its three levels (associations, interventions and counterfactuals), this paper introduces RE-IMAGINE, a framework to characterize a hierarchy of reasoning ability in LLMs, alongside an automated pipeline to generate problem variations at different levels of the hierarchy. By altering problems in an intermediate symbolic representation, RE-IMAGINE generates arbitrarily many problems that are not solvable using memorization alone. Moreover, the framework is general and can work across reasoning domains, including math, code, and logic. We demonstrate our framework on four widely-used benchmarks to evaluate several families of LLMs, and observe reductions in performance when the models are queried with problem variations. These assessments indicate a degree of reliance on statistical recall for past performance, and open the door to further research targeting skills across the reasoning hierarchy. 

---
# Uncovering Intention through LLM-Driven Code Snippet Description Generation 

**Authors**: Yusuf Sulistyo Nugroho, Farah Danisha Salam, Brittany Reid, Raula Gaikovina Kula, Kazumasa Shimari, Kenichi Matsumoto  

**Link**: [PDF](https://arxiv.org/pdf/2506.15453)  

**Abstract**: Documenting code snippets is essential to pinpoint key areas where both developers and users should pay attention. Examples include usage examples and other Application Programming Interfaces (APIs), which are especially important for third-party libraries. With the rise of Large Language Models (LLMs), the key goal is to investigate the kinds of description developers commonly use and evaluate how well an LLM, in this case Llama, can support description generation. We use NPM Code Snippets, consisting of 185,412 packages with 1,024,579 code snippets. From there, we use 400 code snippets (and their descriptions) as samples. First, our manual classification found that the majority of original descriptions (55.5%) highlight example-based usage. This finding emphasizes the importance of clear documentation, as some descriptions lacked sufficient detail to convey intent. Second, the LLM correctly identified the majority of original descriptions as "Example" (79.75%), which is identical to our manual finding, showing a propensity for generalization. Third, compared to the originals, the produced description had an average similarity score of 0.7173, suggesting relevance but room for improvement. Scores below 0.9 indicate some irrelevance. Our results show that depending on the task of the code snippet, the intention of the document may differ from being instructions for usage, installations, or descriptive learning examples for any user of a library. 

---
# Warping and Matching Subsequences Between Time Series 

**Authors**: Simiao Lin, Wannes Meert, Pieter Robberechts, Hendrik Blockeel  

**Link**: [PDF](https://arxiv.org/pdf/2506.15452)  

**Abstract**: Comparing time series is essential in various tasks such as clustering and classification. While elastic distance measures that allow warping provide a robust quantitative comparison, a qualitative comparison on top of them is missing. Traditional visualizations focus on point-to-point alignment and do not convey the broader structural relationships at the level of subsequences. This limitation makes it difficult to understand how and where one time series shifts, speeds up or slows down with respect to another. To address this, we propose a novel technique that simplifies the warping path to highlight, quantify and visualize key transformations (shift, compression, difference in amplitude). By offering a clearer representation of how subsequences match between time series, our method enhances interpretability in time series comparison. 

---
# Zero-Shot Reinforcement Learning Under Partial Observability 

**Authors**: Scott Jeen, Tom Bewley, Jonathan M. Cullen  

**Link**: [PDF](https://arxiv.org/pdf/2506.15446)  

**Abstract**: Recent work has shown that, under certain assumptions, zero-shot reinforcement learning (RL) methods can generalise to any unseen task in an environment after reward-free pre-training. Access to Markov states is one such assumption, yet, in many real-world applications, the Markov state is only partially observable. Here, we explore how the performance of standard zero-shot RL methods degrades when subjected to partially observability, and show that, as in single-task RL, memory-based architectures are an effective remedy. We evaluate our memory-based zero-shot RL methods in domains where the states, rewards and a change in dynamics are partially observed, and show improved performance over memory-free baselines. Our code is open-sourced via: this https URL. 

---
# Hunyuan3D 2.1: From Images to High-Fidelity 3D Assets with Production-Ready PBR Material 

**Authors**: Team Hunyuan3D, Shuhui Yang, Mingxin Yang, Yifei Feng, Xin Huang, Sheng Zhang, Zebin He, Di Luo, Haolin Liu, Yunfei Zhao, Qingxiang Lin, Zeqiang Lai, Xianghui Yang, Huiwen Shi, Zibo Zhao, Bowen Zhang, Hongyu Yan, Lifu Wang, Sicong Liu, Jihong Zhang, Meng Chen, Liang Dong, Yiwen Jia, Yulin Cai, Jiaao Yu, Yixuan Tang, Dongyuan Guo, Junlin Yu, Hao Zhang, Zheng Ye, Peng He, Runzhou Wu, Shida Wei, Chao Zhang, Yonghao Tan, Yifu Sun, Lin Niu, Shirui Huang, Bojian Zheng, Shu Liu, Shilin Chen, Xiang Yuan, Xiaofeng Yang, Kai Liu, Jianchen Zhu, Peng Chen, Tian Liu, Di Wang, Yuhong Liu, Linus, Jie Jiang, Jingwei Huang, Chunchao Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.15442)  

**Abstract**: 3D AI-generated content (AIGC) is a passionate field that has significantly accelerated the creation of 3D models in gaming, film, and design. Despite the development of several groundbreaking models that have revolutionized 3D generation, the field remains largely accessible only to researchers, developers, and designers due to the complexities involved in collecting, processing, and training 3D models. To address these challenges, we introduce Hunyuan3D 2.1 as a case study in this tutorial. This tutorial offers a comprehensive, step-by-step guide on processing 3D data, training a 3D generative model, and evaluating its performance using Hunyuan3D 2.1, an advanced system for producing high-resolution, textured 3D assets. The system comprises two core components: the Hunyuan3D-DiT for shape generation and the Hunyuan3D-Paint for texture synthesis. We will explore the entire workflow, including data preparation, model architecture, training strategies, evaluation metrics, and deployment. By the conclusion of this tutorial, you will have the knowledge to finetune or develop a robust 3D generative model suitable for applications in gaming, virtual reality, and industrial design. 

---
# Reward Models in Deep Reinforcement Learning: A Survey 

**Authors**: Rui Yu, Shenghua Wan, Yucen Wang, Chen-Xiao Gao, Le Gan, Zongzhang Zhang, De-Chuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15421)  

**Abstract**: In reinforcement learning (RL), agents continually interact with the environment and use the feedback to refine their behavior. To guide policy optimization, reward models are introduced as proxies of the desired objectives, such that when the agent maximizes the accumulated reward, it also fulfills the task designer's intentions. Recently, significant attention from both academic and industrial researchers has focused on developing reward models that not only align closely with the true objectives but also facilitate policy optimization. In this survey, we provide a comprehensive review of reward modeling techniques within the deep RL literature. We begin by outlining the background and preliminaries in reward modeling. Next, we present an overview of recent reward modeling approaches, categorizing them based on the source, the mechanism, and the learning paradigm. Building on this understanding, we discuss various applications of these reward modeling techniques and review methods for evaluating reward models. Finally, we conclude by highlighting promising research directions in reward modeling. Altogether, this survey includes both established and emerging methods, filling the vacancy of a systematic review of reward models in current literature. 

---
# Unifying VXAI: A Systematic Review and Framework for the Evaluation of Explainable AI 

**Authors**: David Dembinsky, Adriano Lucieri, Stanislav Frolov, Hiba Najjar, Ko Watanabe, Andreas Dengel  

**Link**: [PDF](https://arxiv.org/pdf/2506.15408)  

**Abstract**: Modern AI systems frequently rely on opaque black-box models, most notably Deep Neural Networks, whose performance stems from complex architectures with millions of learned parameters. While powerful, their complexity poses a major challenge to trustworthiness, particularly due to a lack of transparency. Explainable AI (XAI) addresses this issue by providing human-understandable explanations of model behavior. However, to ensure their usefulness and trustworthiness, such explanations must be rigorously evaluated. Despite the growing number of XAI methods, the field lacks standardized evaluation protocols and consensus on appropriate metrics. To address this gap, we conduct a systematic literature review following the Preferred Reporting Items for Systematic Reviews and Meta-Analyses (PRISMA) guidelines and introduce a unified framework for the eValuation of XAI (VXAI). We identify 362 relevant publications and aggregate their contributions into 41 functionally similar metric groups. In addition, we propose a three-dimensional categorization scheme spanning explanation type, evaluation contextuality, and explanation quality desiderata. Our framework provides the most comprehensive and structured overview of VXAI to date. It supports systematic metric selection, promotes comparability across methods, and offers a flexible foundation for future extensions. 

---
# MCOO-SLAM: A Multi-Camera Omnidirectional Object SLAM System 

**Authors**: Miaoxin Pan, Jinnan Li, Yaowen Zhang, Yi Yang, Yufeng Yue  

**Link**: [PDF](https://arxiv.org/pdf/2506.15402)  

**Abstract**: Object-level SLAM offers structured and semantically meaningful environment representations, making it more interpretable and suitable for high-level robotic tasks. However, most existing approaches rely on RGB-D sensors or monocular views, which suffer from narrow fields of view, occlusion sensitivity, and limited depth perception-especially in large-scale or outdoor environments. These limitations often restrict the system to observing only partial views of objects from limited perspectives, leading to inaccurate object modeling and unreliable data association. In this work, we propose MCOO-SLAM, a novel Multi-Camera Omnidirectional Object SLAM system that fully leverages surround-view camera configurations to achieve robust, consistent, and semantically enriched mapping in complex outdoor scenarios. Our approach integrates point features and object-level landmarks enhanced with open-vocabulary semantics. A semantic-geometric-temporal fusion strategy is introduced for robust object association across multiple views, leading to improved consistency and accurate object modeling, and an omnidirectional loop closure module is designed to enable viewpoint-invariant place recognition using scene-level descriptors. Furthermore, the constructed map is abstracted into a hierarchical 3D scene graph to support downstream reasoning tasks. Extensive experiments in real-world demonstrate that MCOO-SLAM achieves accurate localization and scalable object-level mapping with improved robustness to occlusion, pose variation, and environmental complexity. 

---
# A Real-time Endoscopic Image Denoising System 

**Authors**: Yu Xing, Shishi Huang, Meng Lv, Guo Chen, Huailiang Wang, Lingzhi Sui  

**Link**: [PDF](https://arxiv.org/pdf/2506.15395)  

**Abstract**: Endoscopes featuring a miniaturized design have significantly enhanced operational flexibility, portability, and diagnostic capability while substantially reducing the invasiveness of medical procedures. Recently, single-use endoscopes equipped with an ultra-compact analogue image sensor measuring less than 1mm x 1mm bring revolutionary advancements to medical diagnosis. They reduce the structural redundancy and large capital expenditures associated with reusable devices, eliminate the risk of patient infections caused by inadequate disinfection, and alleviate patient suffering. However, the limited photosensitive area results in reduced photon capture per pixel, requiring higher photon sensitivity settings to maintain adequate brightness. In high-contrast medical imaging scenarios, the small-sized sensor exhibits a constrained dynamic range, making it difficult to simultaneously capture details in both highlights and shadows, and additional localized digital gain is required to compensate. Moreover, the simplified circuit design and analog signal transmission introduce additional noise sources. These factors collectively contribute to significant noise issues in processed endoscopic images. In this work, we developed a comprehensive noise model for analog image sensors in medical endoscopes, addressing three primary noise types: fixed-pattern noise, periodic banding noise, and mixed Poisson-Gaussian noise. Building on this analysis, we propose a hybrid denoising system that synergistically combines traditional image processing algorithms with advanced learning-based techniques for captured raw frames from sensors. Experiments demonstrate that our approach effectively reduces image noise without fine detail loss or color distortion, while achieving real-time performance on FPGA platforms and an average PSNR improvement from 21.16 to 33.05 on our test dataset. 

---
# Evaluation Pipeline for systematically searching for Anomaly Detection Systems 

**Authors**: Florian Rokohl, Alexander Lehnert, Marc Reichenbach  

**Link**: [PDF](https://arxiv.org/pdf/2506.15388)  

**Abstract**: Digitalization in the medical world provides major benefits while making it a target for attackers and thus hard to secure. To deal with network intruders we propose an anomaly detection system on hardware to detect malicious clients in real-time. We meet real-time and power restrictions using FPGAs. Overall system performance is achieved via the presented holistic system evaluation. 

---
# Open-World Object Counting in Videos 

**Authors**: Niki Amini-Naieni, Andrew Zisserman  

**Link**: [PDF](https://arxiv.org/pdf/2506.15368)  

**Abstract**: We introduce a new task of open-world object counting in videos: given a text description, or an image example, that specifies the target object, the objective is to enumerate all the unique instances of the target objects in the video. This task is especially challenging in crowded scenes with occlusions and similar objects, where avoiding double counting and identifying reappearances is crucial. To this end, we make the following contributions: we introduce a model, CountVid, for this task. It leverages an image-based counting model, and a promptable video segmentation and tracking model to enable automated, open-world object counting across video frames. To evaluate its performance, we introduce VideoCount, a new dataset for our novel task built from the TAO and MOT20 tracking datasets, as well as from videos of penguins and metal alloy crystallization captured by x-rays. Using this dataset, we demonstrate that CountVid provides accurate object counts, and significantly outperforms strong baselines. The VideoCount dataset, the CountVid model, and all the code are available at this https URL. 

---
# When and How Unlabeled Data Provably Improve In-Context Learning 

**Authors**: Yingcong Li, Xiangyu Chang, Muti Kara, Xiaofeng Liu, Amit Roy-Chowdhury, Samet Oymak  

**Link**: [PDF](https://arxiv.org/pdf/2506.15329)  

**Abstract**: Recent research shows that in-context learning (ICL) can be effective even when demonstrations have missing or incorrect labels. To shed light on this capability, we examine a canonical setting where the demonstrations are drawn according to a binary Gaussian mixture model (GMM) and a certain fraction of the demonstrations have missing labels. We provide a comprehensive theoretical study to show that: (1) The loss landscape of one-layer linear attention models recover the optimal fully-supervised estimator but completely fail to exploit unlabeled data; (2) In contrast, multilayer or looped transformers can effectively leverage unlabeled data by implicitly constructing estimators of the form $\sum_{i\ge 0} a_i (X^\top X)^iX^\top y$ with $X$ and $y$ denoting features and partially-observed labels (with missing entries set to zero). We characterize the class of polynomials that can be expressed as a function of depth and draw connections to Expectation Maximization, an iterative pseudo-labeling algorithm commonly used in semi-supervised learning. Importantly, the leading polynomial power is exponential in depth, so mild amount of depth/looping suffices. As an application of theory, we propose looping off-the-shelf tabular foundation models to enhance their semi-supervision capabilities. Extensive evaluations on real-world datasets show that our method significantly improves the semisupervised tabular learning performance over the standard single pass inference. 

---
# J3DAI: A tiny DNN-Based Edge AI Accelerator for 3D-Stacked CMOS Image Sensor 

**Authors**: Benoit Tain, Raphael Millet, Romain Lemaire, Michal Szczepanski, Laurent Alacoque, Emmanuel Pluchart, Sylvain Choisnet, Rohit Prasad, Jerome Chossat, Pascal Pierunek, Pascal Vivet, Sebastien Thuries  

**Link**: [PDF](https://arxiv.org/pdf/2506.15316)  

**Abstract**: This paper presents J3DAI, a tiny deep neural network-based hardware accelerator for a 3-layer 3D-stacked CMOS image sensor featuring an artificial intelligence (AI) chip integrating a Deep Neural Network (DNN)-based accelerator. The DNN accelerator is designed to efficiently perform neural network tasks such as image classification and segmentation. This paper focuses on the digital system of J3DAI, highlighting its Performance-Power-Area (PPA) characteristics and showcasing advanced edge AI capabilities on a CMOS image sensor. To support hardware, we utilized the Aidge comprehensive software framework, which enables the programming of both the host processor and the DNN accelerator. Aidge supports post-training quantization, significantly reducing memory footprint and computational complexity, making it crucial for deploying models on resource-constrained hardware like J3DAI. Our experimental results demonstrate the versatility and efficiency of this innovative design in the field of edge AI, showcasing its potential to handle both simple and computationally intensive tasks. Future work will focus on further optimizing the architecture and exploring new applications to fully leverage the capabilities of J3DAI. As edge AI continues to grow in importance, innovations like J3DAI will play a crucial role in enabling real-time, low-latency, and energy-efficient AI processing at the edge. 

---
# MapFM: Foundation Model-Driven HD Mapping with Multi-Task Contextual Learning 

**Authors**: Leonid Ivanov, Vasily Yuryev, Dmitry Yudin  

**Link**: [PDF](https://arxiv.org/pdf/2506.15313)  

**Abstract**: In autonomous driving, high-definition (HD) maps and semantic maps in bird's-eye view (BEV) are essential for accurate localization, planning, and decision-making. This paper introduces an enhanced End-to-End model named MapFM for online vectorized HD map generation. We show significantly boost feature representation quality by incorporating powerful foundation model for encoding camera images. To further enrich the model's understanding of the environment and improve prediction quality, we integrate auxiliary prediction heads for semantic segmentation in the BEV representation. This multi-task learning approach provides richer contextual supervision, leading to a more comprehensive scene representation and ultimately resulting in higher accuracy and improved quality of the predicted vectorized HD maps. The source code is available at this https URL. 

---
# Active Learning-Guided Seq2Seq Variational Autoencoder for Multi-target Inhibitor Generation 

**Authors**: Júlia Vilalta-Mor, Alexis Molina, Laura Ortega Varga, Isaac Filella-Merce, Victor Guallar  

**Link**: [PDF](https://arxiv.org/pdf/2506.15309)  

**Abstract**: Simultaneously optimizing molecules against multiple therapeutic targets remains a profound challenge in drug discovery, particularly due to sparse rewards and conflicting design constraints. We propose a structured active learning (AL) paradigm integrating a sequence-to-sequence (Seq2Seq) variational autoencoder (VAE) into iterative loops designed to balance chemical diversity, molecular quality, and multi-target affinity. Our method alternates between expanding chemically feasible regions of latent space and progressively constraining molecules based on increasingly stringent multi-target docking thresholds. In a proof-of-concept study targeting three related coronavirus main proteases (SARS-CoV-2, SARS-CoV, MERS-CoV), our approach efficiently generated a structurally diverse set of pan-inhibitor candidates. We demonstrate that careful timing and strategic placement of chemical filters within this active learning pipeline markedly enhance exploration of beneficial chemical space, transforming the sparse-reward, multi-objective drug design problem into an accessible computational task. Our framework thus provides a generalizable roadmap for efficiently navigating complex polypharmacological landscapes. 

---
# ConLID: Supervised Contrastive Learning for Low-Resource Language Identification 

**Authors**: Negar Foroutan, Jakhongir Saydaliev, Ye Eun Kim, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2506.15304)  

**Abstract**: Language identification (LID) is a critical step in curating multilingual LLM pretraining corpora from web crawls. While many studies on LID model training focus on collecting diverse training data to improve performance, low-resource languages -- often limited to single-domain data, such as the Bible -- continue to perform poorly. To resolve these class imbalance and bias issues, we propose a novel supervised contrastive learning (SCL) approach to learn domain-invariant representations for low-resource languages. Through an extensive analysis, we show that our approach improves LID performance on out-of-domain data for low-resource languages by 3.2%, demonstrating its effectiveness in enhancing LID models. 

---
# Cohort Discovery: A Survey on LLM-Assisted Clinical Trial Recruitment 

**Authors**: Shrestha Ghosh, Moritz Schneider, Carina Reinicke, Carsten Eickhoff  

**Link**: [PDF](https://arxiv.org/pdf/2506.15301)  

**Abstract**: Recent advances in LLMs have greatly improved general-domain NLP tasks. Yet, their adoption in critical domains, such as clinical trial recruitment, remains limited. As trials are designed in natural language and patient data is represented as both structured and unstructured text, the task of matching trials and patients benefits from knowledge aggregation and reasoning abilities of LLMs. Classical approaches are trial-specific and LLMs with their ability to consolidate distributed knowledge hold the potential to build a more general solution. Yet recent applications of LLM-assisted methods rely on proprietary models and weak evaluation benchmarks. In this survey, we are the first to analyze the task of trial-patient matching and contextualize emerging LLM-based approaches in clinical trial recruitment. We critically examine existing benchmarks, approaches and evaluation frameworks, the challenges to adopting LLM technologies in clinical research and exciting future directions. 

---
# Human Motion Capture from Loose and Sparse Inertial Sensors with Garment-aware Diffusion Models 

**Authors**: Andela Ilic, Jiaxi Jiang, Paul Streli, Xintong Liu, Christian Holz  

**Link**: [PDF](https://arxiv.org/pdf/2506.15290)  

**Abstract**: Motion capture using sparse inertial sensors has shown great promise due to its portability and lack of occlusion issues compared to camera-based tracking. Existing approaches typically assume that IMU sensors are tightly attached to the human body. However, this assumption often does not hold in real-world scenarios. In this paper, we present a new task of full-body human pose estimation using sparse, loosely attached IMU sensors. To solve this task, we simulate IMU recordings from an existing garment-aware human motion dataset. We developed transformer-based diffusion models to synthesize loose IMU data and estimate human poses based on this challenging loose IMU data. In addition, we show that incorporating garment-related parameters while training the model on simulated loose data effectively maintains expressiveness and enhances the ability to capture variations introduced by looser or tighter garments. Experiments show that our proposed diffusion methods trained on simulated and synthetic data outperformed the state-of-the-art methods quantitatively and qualitatively, opening up a promising direction for future research. 

---
# Unlocking Post-hoc Dataset Inference with Synthetic Data 

**Authors**: Bihe Zhao, Pratyush Maini, Franziska Boenisch, Adam Dziedzic  

**Link**: [PDF](https://arxiv.org/pdf/2506.15271)  

**Abstract**: The remarkable capabilities of Large Language Models (LLMs) can be mainly attributed to their massive training datasets, which are often scraped from the internet without respecting data owners' intellectual property rights. Dataset Inference (DI) offers a potential remedy by identifying whether a suspect dataset was used in training, thereby enabling data owners to verify unauthorized use. However, existing DI methods require a private set-known to be absent from training-that closely matches the compromised dataset's distribution. Such in-distribution, held-out data is rarely available in practice, severely limiting the applicability of DI. In this work, we address this challenge by synthetically generating the required held-out set. Our approach tackles two key obstacles: (1) creating high-quality, diverse synthetic data that accurately reflects the original distribution, which we achieve via a data generator trained on a carefully designed suffix-based completion task, and (2) bridging likelihood gaps between real and synthetic data, which is realized through post-hoc calibration. Extensive experiments on diverse text datasets show that using our generated data as a held-out set enables DI to detect the original training sets with high confidence, while maintaining a low false positive rate. This result empowers copyright owners to make legitimate claims on data usage and demonstrates our method's reliability for real-world litigations. Our code is available at this https URL. 

---
# Domain Adaptation for Image Classification of Defects in Semiconductor Manufacturing 

**Authors**: Adrian Poniatowski, Natalie Gentner, Manuel Barusco, Davide Dalle Pezze, Samuele Salti, Gian Antonio Susto  

**Link**: [PDF](https://arxiv.org/pdf/2506.15260)  

**Abstract**: In the semiconductor sector, due to high demand but also strong and increasing competition, time to market and quality are key factors in securing significant market share in various application areas. Thanks to the success of deep learning methods in recent years in the computer vision domain, Industry 4.0 and 5.0 applications, such as defect classification, have achieved remarkable success. In particular, Domain Adaptation (DA) has proven highly effective since it focuses on using the knowledge learned on a (source) domain to adapt and perform effectively on a different but related (target) domain. By improving robustness and scalability, DA minimizes the need for extensive manual re-labeling or re-training of models. This not only reduces computational and resource costs but also allows human experts to focus on high-value tasks. Therefore, we tested the efficacy of DA techniques in semi-supervised and unsupervised settings within the context of the semiconductor field. Moreover, we propose the DBACS approach, a CycleGAN-inspired model enhanced with additional loss terms to improve performance. All the approaches are studied and validated on real-world Electron Microscope images considering the unsupervised and semi-supervised settings, proving the usefulness of our method in advancing DA techniques for the semiconductor field. 

---
# RAS-Eval: A Comprehensive Benchmark for Security Evaluation of LLM Agents in Real-World Environments 

**Authors**: Yuchuan Fu, Xiaohan Yuan, Dongxia Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15253)  

**Abstract**: The rapid deployment of Large language model (LLM) agents in critical domains like healthcare and finance necessitates robust security frameworks. To address the absence of standardized evaluation benchmarks for these agents in dynamic environments, we introduce RAS-Eval, a comprehensive security benchmark supporting both simulated and real-world tool execution. RAS-Eval comprises 80 test cases and 3,802 attack tasks mapped to 11 Common Weakness Enumeration (CWE) categories, with tools implemented in JSON, LangGraph, and Model Context Protocol (MCP) formats. We evaluate 6 state-of-the-art LLMs across diverse scenarios, revealing significant vulnerabilities: attacks reduced agent task completion rates (TCR) by 36.78% on average and achieved an 85.65% success rate in academic settings. Notably, scaling laws held for security capabilities, with larger models outperforming smaller counterparts. Our findings expose critical risks in real-world agent deployments and provide a foundational framework for future security research. Code and data are available at this https URL. 

---
# Singular Value Decomposition on Kronecker Adaptation for Large Language Model 

**Authors**: Yee Hin Chong, Peng Qu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15251)  

**Abstract**: Large pre-trained Transformer models achieve state-of-the-art results across diverse language and reasoning tasks, but full fine-tuning incurs substantial storage, memory, and computational overhead. Parameter-efficient fine-tuning (PEFT) methods mitigate these costs by learning only a small subset of task-specific parameters, yet existing approaches either introduce inference-time latency (adapter modules), suffer from suboptimal convergence (randomly initialized low-rank updates), or rely on fixed rank choices that may not match task complexity (Kronecker-based decompositions).
We propose SoKA (SVD on Kronecker Adaptation), a novel PEFT strategy that combines Kronecker-product tensor factorization with SVD-driven initialization and spectrum-aware dynamic rank selection. Our Kronecker-Product SVD (KPSVD) procedure extracts principal components of the full weight update into compact Kronecker factors, while an adaptive rank selection algorithm uses energy-threshold and elbow-point criteria to prune negligible components.
Empirical evaluation on LLaMA2-7B across arithmetic reasoning (GSM8K), formal mathematics (MATH), and code generation (MBPP) demonstrates that SoKA requires only 0.99M trainable parameters, 25% fewer than LoRA/PiSSA, while matching or exceeding baseline performance. Moreover, SoKA exhibits faster convergence and more stable gradients, highlighting its robustness and efficiency for large-scale model adaptation. 

---
# A Comparative Study of Task Adaptation Techniques of Large Language Models for Identifying Sustainable Development Goals 

**Authors**: Andrea Cadeddu, Alessandro Chessa, Vincenzo De Leo, Gianni Fenu, Enrico Motta, Francesco Osborne, Diego Reforgiato Recupero, Angelo Salatino, Luca Secchi  

**Link**: [PDF](https://arxiv.org/pdf/2506.15208)  

**Abstract**: In 2012, the United Nations introduced 17 Sustainable Development Goals (SDGs) aimed at creating a more sustainable and improved future by 2030. However, tracking progress toward these goals is difficult because of the extensive scale and complexity of the data involved. Text classification models have become vital tools in this area, automating the analysis of vast amounts of text from a variety of sources. Additionally, large language models (LLMs) have recently proven indispensable for many natural language processing tasks, including text classification, thanks to their ability to recognize complex linguistic patterns and semantics. This study analyzes various proprietary and open-source LLMs for a single-label, multi-class text classification task focused on the SDGs. Then, it also evaluates the effectiveness of task adaptation techniques (i.e., in-context learning approaches), namely Zero-Shot and Few-Shot Learning, as well as Fine-Tuning within this domain. The results reveal that smaller models, when optimized through prompt engineering, can perform on par with larger models like OpenAI's GPT (Generative Pre-trained Transformer). 

---
# Accessible Gesture-Driven Augmented Reality Interaction System 

**Authors**: Yikan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15189)  

**Abstract**: Augmented reality (AR) offers immersive interaction but remains inaccessible for users with motor impairments or limited dexterity due to reliance on precise input methods. This study proposes a gesture-based interaction system for AR environments, leveraging deep learning to recognize hand and body gestures from wearable sensors and cameras, adapting interfaces to user capabilities. The system employs vision transformers (ViTs), temporal convolutional networks (TCNs), and graph attention networks (GATs) for gesture processing, with federated learning ensuring privacy-preserving model training across diverse users. Reinforcement learning optimizes interface elements like menu layouts and interaction modes. Experiments demonstrate a 20% improvement in task completion efficiency and a 25% increase in user satisfaction for motor-impaired users compared to baseline AR systems. This approach enhances AR accessibility and scalability. Keywords: Deep learning, Federated learning, Gesture recognition, Augmented reality, Accessibility, Human-computer interaction 

---
# Classification of Multi-Parametric Body MRI Series Using Deep Learning 

**Authors**: Boah Kim, Tejas Sudharshan Mathai, Kimberly Helm, Peter A. Pinto, Ronald M. Summers  

**Link**: [PDF](https://arxiv.org/pdf/2506.15182)  

**Abstract**: Multi-parametric magnetic resonance imaging (mpMRI) exams have various series types acquired with different imaging protocols. The DICOM headers of these series often have incorrect information due to the sheer diversity of protocols and occasional technologist errors. To address this, we present a deep learning-based classification model to classify 8 different body mpMRI series types so that radiologists read the exams efficiently. Using mpMRI data from various institutions, multiple deep learning-based classifiers of ResNet, EfficientNet, and DenseNet are trained to classify 8 different MRI series, and their performance is compared. Then, the best-performing classifier is identified, and its classification capability under the setting of different training data quantities is studied. Also, the model is evaluated on the out-of-training-distribution datasets. Moreover, the model is trained using mpMRI exams obtained from different scanners in two training strategies, and its performance is tested. Experimental results show that the DenseNet-121 model achieves the highest F1-score and accuracy of 0.966 and 0.972 over the other classification models with p-value$<$0.05. The model shows greater than 0.95 accuracy when trained with over 729 studies of the training data, whose performance improves as the training data quantities grew larger. On the external data with the DLDS and CPTAC-UCEC datasets, the model yields 0.872 and 0.810 accuracy for each. These results indicate that in both the internal and external datasets, the DenseNet-121 model attains high accuracy for the task of classifying 8 body MRI series types. 

---
# LLM Agent for Hyper-Parameter Optimization 

**Authors**: Wanzhe Wang, Jianqiu Peng, Menghao Hu, Weihuang Zhong, Tong Zhang, Shuai Wang, Yixin Zhang, Mingjie Shao, Wanli Ni  

**Link**: [PDF](https://arxiv.org/pdf/2506.15167)  

**Abstract**: Hyper-parameters are essential and critical for the performance of communication algorithms. However, current hyper-parameters tuning methods for warm-start particles swarm optimization with cross and mutation (WS-PSO-CM) algortihm for radio map-enabled unmanned aerial vehicle (UAV) trajectory and communication are primarily heuristic-based, exhibiting low levels of automation and unsatisfactory performance. In this paper, we design an large language model (LLM) agent for automatic hyper-parameters-tuning, where an iterative framework and model context protocol (MCP) are applied. In particular, the LLM agent is first setup via a profile, which specifies the mission, background, and output format. Then, the LLM agent is driven by the prompt requirement, and iteratively invokes WS-PSO-CM algorithm for exploration. Finally, the LLM agent autonomously terminates the loop and returns a set of hyper-parameters. Our experiment results show that the minimal sum-rate achieved by hyper-parameters generated via our LLM agent is significantly higher than those by both human heuristics and random generation methods. This indicates that an LLM agent with PSO knowledge and WS-PSO-CM algorithm background is useful in finding high-performance hyper-parameters. 

---
# SonicVerse: Multi-Task Learning for Music Feature-Informed Captioning 

**Authors**: Anuradha Chopra, Abhinaba Roy, Dorien Herremans  

**Link**: [PDF](https://arxiv.org/pdf/2506.15154)  

**Abstract**: Detailed captions that accurately reflect the characteristics of a music piece can enrich music databases and drive forward research in music AI. This paper introduces a multi-task music captioning model, SonicVerse, that integrates caption generation with auxiliary music feature detection tasks such as key detection, vocals detection, and more, so as to directly capture both low-level acoustic details as well as high-level musical attributes. The key contribution is a projection-based architecture that transforms audio input into language tokens, while simultaneously detecting music features through dedicated auxiliary heads. The outputs of these heads are also projected into language tokens, to enhance the captioning input. This framework not only produces rich, descriptive captions for short music fragments but also directly enables the generation of detailed time-informed descriptions for longer music pieces, by chaining the outputs using a large-language model. To train the model, we extended the MusicBench dataset by annotating it with music features using MIRFLEX, a modular music feature extractor, resulting in paired audio, captions and music feature data. Experimental results show that incorporating features in this way improves the quality and detail of the generated captions. 

---
# Thunder-Tok: Minimizing Tokens per Word in Tokenizing Korean Texts for Generative Language Models 

**Authors**: Gyeongje Cho, Yeonkyoun So, Chanwoo Park, Sangmin Lee, Sungmok Jung, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2506.15138)  

**Abstract**: This paper introduces Thunder-Tok, a new Korean tokenizer designed to reduce token fertility without compromising model performance. Our approach uses a rule-based pre-tokenization method that aligns with the linguistic structure of the Korean language. We also create a seed vocabulary containing tokens that resemble linguistic units and employ a branching entropy-based selection algorithm. These techniques increase the average token length, thus lowering fertility while preserving linguistic information. Experimental results indicate that Thunder-Tok reduces fertility by approximately 10% (i.e., reduces the number of tokens by 10%, improving the inference speed by 10%) compared to BPE without compromising performance across various downstream tasks. These findings demonstrate that our linguistically informed approach is effective and practical for designing efficient tokenizers for language models. 

---
# Modeling the One-to-Many Property in Open-Domain Dialogue with LLMs 

**Authors**: Jing Yang Lee, Kong-Aik Lee, Woon-Seng Gan  

**Link**: [PDF](https://arxiv.org/pdf/2506.15131)  

**Abstract**: Open-domain Dialogue (OD) exhibits a one-to-many (o2m) property, whereby multiple appropriate responses exist for a single dialogue context. Despite prior research showing that modeling this property boosts response diversity, most modern LLM-based dialogue agents do not explicitly do so. In this work, we model the o2m property of OD in LLMs by decomposing OD generation into two key tasks: Multi-Response Generation (MRG) and Preference-based Selection (PS), which entail generating a set of n semantically and lexically diverse high-quality responses for a given dialogue context, followed by selecting a single response based on human preference, respectively. To facilitate MRG and PS, we introduce o2mDial, a dialogue corpus explicitly designed to capture the o2m property by featuring multiple plausible responses for each context. Leveraging o2mDial, we propose new in-context learning and instruction-tuning strategies, as well as novel evaluation metrics for MRG, alongside a model-based approach for PS. Empirical results demonstrate that applying the proposed two-stage framework to smaller LLMs for OD generation enhances overall response diversity while maintaining contextual coherence, improving response quality by up to 90%, bringing them closer to the performance of larger models. 

---
# Advancing Loss Functions in Recommender Systems: A Comparative Study with a Rényi Divergence-Based Solution 

**Authors**: Shengjia Zhang, Jiawei Chen, Changdong Li, Sheng Zhou, Qihao Shi, Yan Feng, Chun Chen, Can Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15120)  

**Abstract**: Loss functions play a pivotal role in optimizing recommendation models. Among various loss functions, Softmax Loss (SL) and Cosine Contrastive Loss (CCL) are particularly effective. Their theoretical connections and differences warrant in-depth exploration. This work conducts comprehensive analyses of these losses, yielding significant insights: 1) Common strengths -- both can be viewed as augmentations of traditional losses with Distributional Robust Optimization (DRO), enhancing robustness to distributional shifts; 2) Respective limitations -- stemming from their use of different distribution distance metrics in DRO optimization, SL exhibits high sensitivity to false negative instances, whereas CCL suffers from low data utilization. To address these limitations, this work proposes a new loss function, DrRL, which generalizes SL and CCL by leveraging Rényi-divergence in DRO optimization. DrRL incorporates the advantageous structures of both SL and CCL, and can be demonstrated to effectively mitigate their limitations. Extensive experiments have been conducted to validate the superiority of DrRL on both recommendation accuracy and robustness. 

---
# Transit for All: Mapping Equitable Bike2Subway Connection using Region Representation Learning 

**Authors**: Min Namgung, JangHyeon Lee, Fangyi Ding, Yao-Yi Chiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.15113)  

**Abstract**: Ensuring equitable public transit access remains challenging, particularly in densely populated cities like New York City (NYC), where low-income and minority communities often face limited transit accessibility. Bike-sharing systems (BSS) can bridge these equity gaps by providing affordable first- and last-mile connections. However, strategically expanding BSS into underserved neighborhoods is difficult due to uncertain bike-sharing demand at newly planned ("cold-start") station locations and limitations in traditional accessibility metrics that may overlook realistic bike usage potential. We introduce Transit for All (TFA), a spatial computing framework designed to guide the equitable expansion of BSS through three components: (1) spatially-informed bike-sharing demand prediction at cold-start stations using region representation learning that integrates multimodal geospatial data, (2) comprehensive transit accessibility assessment leveraging our novel weighted Public Transport Accessibility Level (wPTAL) by combining predicted bike-sharing demand with conventional transit accessibility metrics, and (3) strategic recommendations for new bike station placements that consider potential ridership and equity enhancement. Using NYC as a case study, we identify transit accessibility gaps that disproportionately impact low-income and minority communities in historically underserved neighborhoods. Our results show that strategically placing new stations guided by wPTAL notably reduces disparities in transit access related to economic and demographic factors. From our study, we demonstrate that TFA provides practical guidance for urban planners to promote equitable transit and enhance the quality of life in underserved urban communities. 

---
# Improving Dialogue Discourse Parsing through Discourse-aware Utterance Clarification 

**Authors**: Yaxin Fan, Peifeng Li, Qiaoming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15081)  

**Abstract**: Dialogue discourse parsing aims to identify and analyze discourse relations between the utterances within dialogues. However, linguistic features in dialogues, such as omission and idiom, frequently introduce ambiguities that obscure the intended discourse relations, posing significant challenges for parsers. To address this issue, we propose a Discourse-aware Clarification Module (DCM) to enhance the performance of the dialogue discourse parser. DCM employs two distinct reasoning processes: clarification type reasoning and discourse goal reasoning. The former analyzes linguistic features, while the latter distinguishes the intended relation from the ambiguous one. Furthermore, we introduce Contribution-aware Preference Optimization (CPO) to mitigate the risk of erroneous clarifications, thereby reducing cascading errors. CPO enables the parser to assess the contributions of the clarifications from DCM and provide feedback to optimize the DCM, enhancing its adaptability and alignment with the parser's requirements. Extensive experiments on the STAC and Molweni datasets demonstrate that our approach effectively resolves ambiguities and significantly outperforms the state-of-the-art (SOTA) baselines. 

---
# Sequential Policy Gradient for Adaptive Hyperparameter Optimization 

**Authors**: Zheng Li, Jerry Cheng, Huanying Helen Gu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15051)  

**Abstract**: Reinforcement learning is essential for neural architecture search and hyperparameter optimization, but the conventional approaches impede widespread use due to prohibitive time and computational costs. Inspired by DeepSeek-V3 multi-token prediction architecture, we propose Sequential Policy Gradient modeling (SPG), a novel trajectory generation paradigm for lightweight online hyperparameter optimization. In contrast to conventional policy gradient methods, SPG extends the base model with temporary modules, enabling it to generate state-action (padded) trajectories in a single forward pass. Our experiments demonstrate that models gain performance when retrained with SPG on their original datasets and also outperform standard transfer fine-tuning. We evaluate on five datasets spanning computer vision (ImageNet, COCO), natural language processing (GLUE, SQuAD), and audio (SUPERB) to assess the industrial applicability of SPG. The proposed method demonstrates consistent improvements across widely adopted models, achieving performance gains of $+0.2\sim7\%$, with significantly low computational costs. Fully reproducible code and pre-trained models: this https URL. 

---
# Mapping Caregiver Needs to AI Chatbot Design: Strengths and Gaps in Mental Health Support for Alzheimer's and Dementia Caregivers 

**Authors**: Jiayue Melissa Shi, Dong Whi Yoo, Keran Wang, Violeta J. Rodriguez, Ravi Karkar, Koustuv Saha  

**Link**: [PDF](https://arxiv.org/pdf/2506.15047)  

**Abstract**: Family caregivers of individuals with Alzheimer's Disease and Related Dementia (AD/ADRD) face significant emotional and logistical challenges that place them at heightened risk for stress, anxiety, and depression. Although recent advances in generative AI -- particularly large language models (LLMs) -- offer new opportunities to support mental health, little is known about how caregivers perceive and engage with such technologies. To address this gap, we developed Carey, a GPT-4o-based chatbot designed to provide informational and emotional support to AD/ADRD caregivers. Using Carey as a technology probe, we conducted semi-structured interviews with 16 family caregivers following scenario-driven interactions grounded in common caregiving stressors. Through inductive coding and reflexive thematic analysis, we surface a systemic understanding of caregiver needs and expectations across six themes -- on-demand information access, emotional support, safe space for disclosure, crisis management, personalization, and data privacy. For each of these themes, we also identified the nuanced tensions in the caregivers' desires and concerns. We present a mapping of caregiver needs, AI chatbot's strengths, gaps, and design recommendations. Our findings offer theoretical and practical insights to inform the design of proactive, trustworthy, and caregiver-centered AI systems that better support the evolving mental health needs of AD/ADRD caregivers. 

---
# Advanced Prediction of Hypersonic Missile Trajectories with CNN-LSTM-GRU Architectures 

**Authors**: Amir Hossein Baradaran  

**Link**: [PDF](https://arxiv.org/pdf/2506.15043)  

**Abstract**: Advancements in the defense industry are paramount for ensuring the safety and security of nations, providing robust protection against emerging threats. Among these threats, hypersonic missiles pose a significant challenge due to their extreme speeds and maneuverability, making accurate trajectory prediction a critical necessity for effective countermeasures. This paper addresses this challenge by employing a novel hybrid deep learning approach, integrating Convolutional Neural Networks (CNNs), Long Short-Term Memory (LSTM) networks, and Gated Recurrent Units (GRUs). By leveraging the strengths of these architectures, the proposed method successfully predicts the complex trajectories of hypersonic missiles with high accuracy, offering a significant contribution to defense strategies and missile interception technologies. This research demonstrates the potential of advanced machine learning techniques in enhancing the predictive capabilities of defense systems. 

---
# Optimal Embedding Learning Rate in LLMs: The Effect of Vocabulary Size 

**Authors**: Soufiane Hayou, Liyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2506.15025)  

**Abstract**: Pretraining large language models is a costly process. To make this process more efficient, several methods have been proposed to optimize model architecture/parametrization and hardware use. On the parametrization side, $\mu P$ (Maximal Update Parametrization) parametrizes model weights and learning rate (LR) in a way that makes hyperparameters (HPs) transferable with width (embedding dimension): HPs can be tuned for a small model and used for larger models without additional tuning. While $\mu$P showed impressive results in practice, recent empirical studies have reported conflicting observations when applied to LLMs. One limitation of the theory behind $\mu$P is the fact that input dimension (vocabulary size in LLMs) is considered fixed when taking the width to infinity. This is unrealistic since vocabulary size is generally much larger than width in practice. In this work, we provide a theoretical analysis of the effect of vocabulary size on training dynamics, and subsequently show that as vocabulary size increases, the training dynamics \emph{interpolate between the $\mu$P regime and another regime that we call Large Vocab (LV) Regime}, where optimal scaling rules are different from those predicted by $\mu$P. Our analysis reveals that in the LV regime, the optimal embedding LR to hidden LR ratio should roughly scale as $\Theta(\sqrt{width})$, surprisingly close to the empirical findings previously reported in the literature, and different from the $\Theta(width)$ ratio predicted by $\mu$P. We conduct several experiments to validate our theory, and pretrain a 1B model from scratch to show the benefit of our suggested scaling rule for the embedding LR. 

---
# SFT-GO: Supervised Fine-Tuning with Group Optimization for Large Language Models 

**Authors**: Gyuhak Kim, Sumiran Singh Thakur, Su Min Park, Wei Wei, Yujia Bao  

**Link**: [PDF](https://arxiv.org/pdf/2506.15021)  

**Abstract**: Supervised fine-tuning (SFT) has become an essential step in tailoring large language models (LLMs) to align with human expectations and specific downstream tasks. However, existing SFT methods typically treat each training instance as a uniform sequence, giving equal importance to all tokens regardless of their relevance. This overlooks the fact that only a subset of tokens often contains critical, task-specific information. To address this limitation, we introduce Supervised Fine-Tuning with Group Optimization (SFT-GO), a novel approach that treats groups of tokens differently based on their this http URL-GO groups tokens in each sample based on their importance values and optimizes the LLM using a weighted combination of the worst-group loss and the standard cross-entropy loss. This mechanism adaptively emphasizes the most challenging token groups and guides the model to better handle different group distributions, thereby improving overall learning dynamics. We provide a theoretical analysis of SFT-GO's convergence rate, demonstrating its efficiency. Empirically, we apply SFT-GO with three different token grouping strategies and show that models trained with SFT-GO consistently outperform baseline approaches across popular LLM benchmarks. These improvements hold across various datasets and base models, demonstrating the robustness and the effectiveness of our method. 

---
# Stable CDE Autoencoders with Acuity Regularization for Offline Reinforcement Learning in Sepsis Treatment 

**Authors**: Yue Gao  

**Link**: [PDF](https://arxiv.org/pdf/2506.15019)  

**Abstract**: Effective reinforcement learning (RL) for sepsis treatment depends on learning stable, clinically meaningful state representations from irregular ICU time series. While previous works have explored representation learning for this task, the critical challenge of training instability in sequential representations and its detrimental impact on policy performance has been overlooked. This work demonstrates that Controlled Differential Equations (CDE) state representation can achieve strong RL policies when two key factors are met: (1) ensuring training stability through early stopping or stabilization methods, and (2) enforcing acuity-aware representations by correlation regularization with clinical scores (SOFA, SAPS-II, OASIS). Experiments on the MIMIC-III sepsis cohort reveal that stable CDE autoencoder produces representations strongly correlated with acuity scores and enables RL policies with superior performance (WIS return $> 0.9$). In contrast, unstable CDE representation leads to degraded representations and policy failure (WIS return $\sim$ 0). Visualizations of the latent space show that stable CDEs not only separate survivor and non-survivor trajectories but also reveal clear acuity score gradients, whereas unstable training fails to capture either pattern. These findings highlight practical guidelines for using CDEs to encode irregular medical time series in clinical RL, emphasizing the need for training stability in sequential representation learning. 

---
# Insights Informed Generative AI for Design: Incorporating Real-world Data for Text-to-Image Output 

**Authors**: Richa Gupta, Alexander Htet Kyaw  

**Link**: [PDF](https://arxiv.org/pdf/2506.15008)  

**Abstract**: Generative AI, specifically text-to-image models, have revolutionized interior architectural design by enabling the rapid translation of conceptual ideas into visual representations from simple text prompts. While generative AI can produce visually appealing images they often lack actionable data for designers In this work, we propose a novel pipeline that integrates DALL-E 3 with a materials dataset to enrich AI-generated designs with sustainability metrics and material usage insights. After the model generates an interior design image, a post-processing module identifies the top ten materials present and pairs them with carbon dioxide equivalent (CO2e) values from a general materials dictionary. This approach allows designers to immediately evaluate environmental impacts and refine prompts accordingly. We evaluate the system through three user tests: (1) no mention of sustainability to the user prior to the prompting process with generative AI, (2) sustainability goals communicated to the user before prompting, and (3) sustainability goals communicated along with quantitative CO2e data included in the generative AI outputs. Our qualitative and quantitative analyses reveal that the introduction of sustainability metrics in the third test leads to more informed design decisions, however, it can also trigger decision fatigue and lower overall satisfaction. Nevertheless, the majority of participants reported incorporating sustainability principles into their workflows in the third test, underscoring the potential of integrated metrics to guide more ecologically responsible practices. Our findings showcase the importance of balancing design freedom with practical constraints, offering a clear path toward holistic, data-driven solutions in AI-assisted architectural design. 

---
# Scaling Intelligence: Designing Data Centers for Next-Gen Language Models 

**Authors**: Jesmin Jahan Tithi, Hanjiang Wu, Avishaii Abuhatzera, Fabrizio Petrini  

**Link**: [PDF](https://arxiv.org/pdf/2506.15006)  

**Abstract**: The explosive growth of Large Language Models (LLMs) - such as GPT-4 with 1.8 trillion parameters - demands a radical rethinking of data center architecture to ensure scalability, efficiency, and cost-effectiveness. Our work provides a comprehensive co-design framework that jointly explores FLOPS, HBM bandwidth and capacity, multiple network topologies (two-tier vs. FullFlat optical), the size of the scale-out domain, and popular parallelism/optimization strategies used in LLMs. We introduce and evaluate FullFlat network architectures, which provide uniform high-bandwidth, low-latency connectivity between all nodes, and demonstrate their transformative impact on performance and scalability. Through detailed sensitivity analyses, we quantify the benefits of overlapping compute and communication, leveraging hardware-accelerated collectives, wider scale-out domains, and larger memory capacity. Our study spans both sparse (mixture of experts) and dense transformer-based LLMs, revealing how system design choices affect Model FLOPS Utilization (MFU = Model flops per token x Observed tokens per sec / Peak flops of the hardware) and overall throughput. For the co-design study, we extended and validated a performance modeling tool capable of predicting LLM runtime within 10% of real-world measurements. Our findings offer actionable insights and a practical roadmap for designing AI data centers that can efficiently support trillion-parameter models, reduce optimization complexity, and sustain the rapid evolution of AI capabilities. 

---
# Memory Tokens: Large Language Models Can Generate Reversible Sentence Embeddings 

**Authors**: Ignacio Sastre, Aiala Rosá  

**Link**: [PDF](https://arxiv.org/pdf/2506.15001)  

**Abstract**: In this work, we observe an interesting phenomenon: it is possible to generate reversible sentence embeddings that allow an LLM to reconstruct the original text exactly, without modifying the model's weights. This is achieved by introducing a special memory token, whose embedding is optimized through training on a fixed sequence. When prompted with this embedding, the model reconstructs the fixed sequence exactly. We evaluate this phenomenon across English and Spanish datasets, sequences of up to approximately 240 tokens, and model scales ranging from 100M to 8B parameters. Notably, Llama 3.1 8B successfully reconstructs all tested sequences. Our findings highlight an interesting capability of LLMs and suggest potential applications in memory-based retrieval, compression, and controlled text generation. 

---
# Improved Image Reconstruction and Diffusion Parameter Estimation Using a Temporal Convolutional Network Model of Gradient Trajectory Errors 

**Authors**: Jonathan B. Martin, Hannah E. Alderson, John C. Gore, Mark D. Does, Kevin D. Harkins  

**Link**: [PDF](https://arxiv.org/pdf/2506.14995)  

**Abstract**: Summary: Errors in gradient trajectories introduce significant artifacts and distortions in magnetic resonance images, particularly in non-Cartesian imaging sequences, where imperfect gradient waveforms can greatly reduce image quality. Purpose: Our objective is to develop a general, nonlinear gradient system model that can accurately predict gradient distortions using convolutional networks. Methods: A set of training gradient waveforms were measured on a small animal imaging system, and used to train a temporal convolutional network to predict the gradient waveforms produced by the imaging system. Results: The trained network was able to accurately predict nonlinear distortions produced by the gradient system. Network prediction of gradient waveforms was incorporated into the image reconstruction pipeline and provided improvements in image quality and diffusion parameter mapping compared to both the nominal gradient waveform and the gradient impulse response function. Conclusion: Temporal convolutional networks can more accurately model gradient system behavior than existing linear methods and may be used to retrospectively correct gradient errors. 

---
# Fair Algorithms with Probing for Multi-Agent Multi-Armed Bandits 

**Authors**: Tianyi Xu, Jiaxin Liu, Zizhan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.14988)  

**Abstract**: We propose a multi-agent multi-armed bandit (MA-MAB) framework aimed at ensuring fair outcomes across agents while maximizing overall system performance. A key challenge in this setting is decision-making under limited information about arm rewards. To address this, we introduce a novel probing framework that strategically gathers information about selected arms before allocation. In the offline setting, where reward distributions are known, we leverage submodular properties to design a greedy probing algorithm with a provable performance bound. For the more complex online setting, we develop an algorithm that achieves sublinear regret while maintaining fairness. Extensive experiments on synthetic and real-world datasets show that our approach outperforms baseline methods, achieving better fairness and efficiency. 

---
# Thinking in Directivity: Speech Large Language Model for Multi-Talker Directional Speech Recognition 

**Authors**: Jiamin Xie, Ju Lin, Yiteng Huang, Tyler Vuong, Zhaojiang Lin, Zhaojun Yang, Peng Su, Prashant Rawat, Sangeeta Srivastava, Ming Sun, Florian Metze  

**Link**: [PDF](https://arxiv.org/pdf/2506.14973)  

**Abstract**: Recent studies have demonstrated that prompting large language models (LLM) with audio encodings enables effective speech recognition capabilities. However, the ability of Speech LLMs to comprehend and process multi-channel audio with spatial cues remains a relatively uninvestigated area of research. In this work, we present directional-SpeechLlama, a novel approach that leverages the microphone array of smart glasses to achieve directional speech recognition, source localization, and bystander cross-talk suppression. To enhance the model's ability to understand directivity, we propose two key techniques: serialized directional output training (S-DOT) and contrastive direction data augmentation (CDDA). Experimental results show that our proposed directional-SpeechLlama effectively captures the relationship between textual cues and spatial audio, yielding strong performance in both speech recognition and source localization tasks. 

---
# FEAST: A Flexible Mealtime-Assistance System Towards In-the-Wild Personalization 

**Authors**: Rajat Kumar Jenamani, Tom Silver, Ben Dodson, Shiqin Tong, Anthony Song, Yuting Yang, Ziang Liu, Benjamin Howe, Aimee Whitneck, Tapomayukh Bhattacharjee  

**Link**: [PDF](https://arxiv.org/pdf/2506.14968)  

**Abstract**: Physical caregiving robots hold promise for improving the quality of life of millions worldwide who require assistance with feeding. However, in-home meal assistance remains challenging due to the diversity of activities (e.g., eating, drinking, mouth wiping), contexts (e.g., socializing, watching TV), food items, and user preferences that arise during deployment. In this work, we propose FEAST, a flexible mealtime-assistance system that can be personalized in-the-wild to meet the unique needs of individual care recipients. Developed in collaboration with two community researchers and informed by a formative study with a diverse group of care recipients, our system is guided by three key tenets for in-the-wild personalization: adaptability, transparency, and safety. FEAST embodies these principles through: (i) modular hardware that enables switching between assisted feeding, drinking, and mouth-wiping, (ii) diverse interaction methods, including a web interface, head gestures, and physical buttons, to accommodate diverse functional abilities and preferences, and (iii) parameterized behavior trees that can be safely and transparently adapted using a large language model. We evaluate our system based on the personalization requirements identified in our formative study, demonstrating that FEAST offers a wide range of transparent and safe adaptations and outperforms a state-of-the-art baseline limited to fixed customizations. To demonstrate real-world applicability, we conduct an in-home user study with two care recipients (who are community researchers), feeding them three meals each across three diverse scenarios. We further assess FEAST's ecological validity by evaluating with an Occupational Therapist previously unfamiliar with the system. In all cases, users successfully personalize FEAST to meet their individual needs and preferences. Website: this https URL 

---
# Revisiting Reinforcement Learning for LLM Reasoning from A Cross-Domain Perspective 

**Authors**: Zhoujun Cheng, Shibo Hao, Tianyang Liu, Fan Zhou, Yutao Xie, Feng Yao, Yuexin Bian, Yonghao Zhuang, Nilabjo Dey, Yuheng Zha, Yi Gu, Kun Zhou, Yuqi Wang, Yuan Li, Richard Fan, Jianshu She, Chengqian Gao, Abulhair Saparov, Haonan Li, Taylor W. Killian, Mikhail Yurochkin, Zhengzhong Liu, Eric P. Xing, Zhiting Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14965)  

**Abstract**: Reinforcement learning (RL) has emerged as a promising approach to improve large language model (LLM) reasoning, yet most open efforts focus narrowly on math and code, limiting our understanding of its broader applicability to general reasoning. A key challenge lies in the lack of reliable, scalable RL reward signals across diverse reasoning domains. We introduce Guru, a curated RL reasoning corpus of 92K verifiable examples spanning six reasoning domains--Math, Code, Science, Logic, Simulation, and Tabular--each built through domain-specific reward design, deduplication, and filtering to ensure reliability and effectiveness for RL training. Based on Guru, we systematically revisit established findings in RL for LLM reasoning and observe significant variation across domains. For example, while prior work suggests that RL primarily elicits existing knowledge from pretrained models, our results reveal a more nuanced pattern: domains frequently seen during pretraining (Math, Code, Science) easily benefit from cross-domain RL training, while domains with limited pretraining exposure (Logic, Simulation, and Tabular) require in-domain training to achieve meaningful performance gains, suggesting that RL is likely to facilitate genuine skill acquisition. Finally, we present Guru-7B and Guru-32B, two models that achieve state-of-the-art performance among open models RL-trained with publicly available data, outperforming best baselines by 7.9% and 6.7% on our 17-task evaluation suite across six reasoning domains. We also show that our models effectively improve the Pass@k performance of their base models, particularly on complex tasks less likely to appear in pretraining data. We release data, models, training and evaluation code to facilitate general-purpose reasoning at: this https URL 

---
# Flat Channels to Infinity in Neural Loss Landscapes 

**Authors**: Flavio Martinelli, Alexander Van Meegen, Berfin Şimşek, Wulfram Gerstner, Johanni Brea  

**Link**: [PDF](https://arxiv.org/pdf/2506.14951)  

**Abstract**: The loss landscapes of neural networks contain minima and saddle points that may be connected in flat regions or appear in isolation. We identify and characterize a special structure in the loss landscape: channels along which the loss decreases extremely slowly, while the output weights of at least two neurons, $a_i$ and $a_j$, diverge to $\pm$infinity, and their input weight vectors, $\mathbf{w_i}$ and $\mathbf{w_j}$, become equal to each other. At convergence, the two neurons implement a gated linear unit: $a_i\sigma(\mathbf{w_i} \cdot \mathbf{x}) + a_j\sigma(\mathbf{w_j} \cdot \mathbf{x}) \rightarrow \sigma(\mathbf{w} \cdot \mathbf{x}) + (\mathbf{v} \cdot \mathbf{x}) \sigma'(\mathbf{w} \cdot \mathbf{x})$. Geometrically, these channels to infinity are asymptotically parallel to symmetry-induced lines of critical points. Gradient flow solvers, and related optimization methods like SGD or ADAM, reach the channels with high probability in diverse regression settings, but without careful inspection they look like flat local minima with finite parameter values. Our characterization provides a comprehensive picture of these quasi-flat regions in terms of gradient dynamics, geometry, and functional interpretation. The emergence of gated linear units at the end of the channels highlights a surprising aspect of the computational capabilities of fully connected layers. 

---
# Determinação Automática de Limiar de Detecção de Ataques em Redes de Computadores Utilizando Autoencoders 

**Authors**: Luan Gonçalves Miranda, Pedro Ivo da Cruz, Murilo Bellezoni Loiola  

**Link**: [PDF](https://arxiv.org/pdf/2506.14937)  

**Abstract**: Currently, digital security mechanisms like Anomaly Detection Systems using Autoencoders (AE) show great potential for bypassing problems intrinsic to the data, such as data imbalance. Because AE use a non-trivial and nonstandardized separation threshold to classify the extracted reconstruction error, the definition of this threshold directly impacts the performance of the detection process. Thus, this work proposes the automatic definition of this threshold using some machine learning algorithms. For this, three algorithms were evaluated: the K-Nearst Neighbors, the K-Means and the Support Vector Machine. 

---
# Explain First, Trust Later: LLM-Augmented Explanations for Graph-Based Crypto Anomaly Detection 

**Authors**: Adriana Watson  

**Link**: [PDF](https://arxiv.org/pdf/2506.14933)  

**Abstract**: The decentralized finance (DeFi) community has grown rapidly in recent years, pushed forward by cryptocurrency enthusiasts interested in the vast untapped potential of new markets. The surge in popularity of cryptocurrency has ushered in a new era of financial crime. Unfortunately, the novelty of the technology makes the task of catching and prosecuting offenders particularly challenging. Thus, it is necessary to implement automated detection tools related to policies to address the growing criminality in the cryptocurrency realm. 

---
# MDBench: A Synthetic Multi-Document Reasoning Benchmark Generated with Knowledge Guidance 

**Authors**: Joseph J. Peper, Wenzhao Qiu, Ali Payani, Lu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14927)  

**Abstract**: Natural language processing evaluation has made significant progress, largely driven by the proliferation of powerful large language mod-els (LLMs). New evaluation benchmarks are of increasing priority as the reasoning capabilities of LLMs are expanding at a rapid pace. In particular, while multi-document (MD) reasoning is an area of extreme relevance given LLM capabilities in handling longer-context inputs, few benchmarks exist to rigorously examine model behavior in this setting. Moreover, the multi-document setting is historically challenging for benchmark creation due to the expensive cost of annotating long inputs. In this work, we introduce MDBench, a new dataset for evaluating LLMs on the task of multi-document reasoning. Notably, MDBench is created through a novel synthetic generation process, allowing us to controllably and efficiently generate challenging document sets and the corresponding question-answer (QA) examples. Our novel technique operates on condensed structured seed knowledge, modifying it through LLM-assisted edits to induce MD-specific reasoning challenges. We then convert this structured knowledge into a natural text surface form, generating a document set and corresponding QA example. We analyze the behavior of popular LLMs and prompting techniques, finding that MDBENCH poses significant challenges for all methods, even with relatively short document sets. We also see our knowledge-guided generation technique (1) allows us to readily perform targeted analysis of MD-specific reasoning capabilities and (2) can be adapted quickly to account for new challenges and future modeling improvements. 

---
# Forecasting the spatiotemporal evolution of fluid-induced microearthquakes with deep learning 

**Authors**: Jaehong Chung, Michael Manga, Timothy Kneafsey, Tapan Mukerji, Mengsu Hu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14923)  

**Abstract**: Microearthquakes (MEQs) generated by subsurface fluid injection record the evolving stress state and permeability of reservoirs. Forecasting their full spatiotemporal evolution is therefore critical for applications such as enhanced geothermal systems (EGS), CO$_2$ sequestration and other geo-engineering applications. We present a transformer-based deep learning model that ingests hydraulic stimulation history and prior MEQ observations to forecast four key quantities: cumulative MEQ count, cumulative logarithmic seismic moment, and the 50th- and 95th-percentile extents ($P_{50}, P_{95}$) of the MEQ cloud. Applied to the EGS Collab Experiment 1 dataset, the model achieves $R^2 >0.98$ for the 1-second forecast horizon and $R^2 >0.88$ for the 15-second forecast horizon across all targets, and supplies uncertainty estimates through a learned standard deviation term. These accurate, uncertainty-quantified forecasts enable real-time inference of fracture propagation and permeability evolution, demonstrating the strong potential of deep-learning approaches to improve seismic-risk assessment and guide mitigation strategies in future fluid-injection operations. 

---
# Foundation Artificial Intelligence Models for Health Recognition Using Face Photographs (FAHR-Face) 

**Authors**: Fridolin Haugg, Grace Lee, John He, Leonard Nürnberg, Dennis Bontempi, Danielle S. Bitterman, Paul Catalano, Vasco Prudente, Dmitrii Glubokov, Andrew Warrington, Suraj Pai, Dirk De Ruysscher, Christian Guthier, Benjamin H. Kann, Vadim N. Gladyshev, Hugo JWL Aerts, Raymond H. Mak  

**Link**: [PDF](https://arxiv.org/pdf/2506.14909)  

**Abstract**: Background: Facial appearance offers a noninvasive window into health. We built FAHR-Face, a foundation model trained on >40 million facial images and fine-tuned it for two distinct tasks: biological age estimation (FAHR-FaceAge) and survival risk prediction (FAHR-FaceSurvival).
Methods: FAHR-FaceAge underwent a two-stage, age-balanced fine-tuning on 749,935 public images; FAHR-FaceSurvival was fine-tuned on 34,389 photos of cancer patients. Model robustness (cosmetic surgery, makeup, pose, lighting) and independence (saliency mapping) was tested extensively. Both models were clinically tested in two independent cancer patient datasets with survival analyzed by multivariable Cox models and adjusted for clinical prognostic factors.
Findings: For age estimation, FAHR-FaceAge had the lowest mean absolute error of 5.1 years on public datasets, outperforming benchmark models and maintaining accuracy across the full human lifespan. In cancer patients, FAHR-FaceAge outperformed a prior facial age estimation model in survival prognostication. FAHR-FaceSurvival demonstrated robust prediction of mortality, and the highest-risk quartile had more than triple the mortality of the lowest (adjusted hazard ratio 3.22; P<0.001). These findings were validated in the independent cohort and both models showed generalizability across age, sex, race and cancer subgroups. The two algorithms provided distinct, complementary prognostic information; saliency mapping revealed each model relied on distinct facial regions. The combination of FAHR-FaceAge and FAHR-FaceSurvival improved prognostic accuracy.
Interpretation: A single foundation model can generate inexpensive, scalable facial biomarkers that capture both biological ageing and disease-related mortality risk. The foundation model enabled effective training using relatively small clinical datasets. 

---
# PeRL: Permutation-Enhanced Reinforcement Learning for Interleaved Vision-Language Reasoning 

**Authors**: Yizhen Zhang, Yang Ding, Shuoshuo Zhang, Xinchen Zhang, Haoling Li, Zhong-zhi Li, Peijie Wang, Jie Wu, Lei Ji, Yelong Shen, Yujiu Yang, Yeyun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2506.14907)  

**Abstract**: Inspired by the impressive reasoning capabilities demonstrated by reinforcement learning approaches like DeepSeek-R1, recent emerging research has begun exploring the use of reinforcement learning (RL) to enhance vision-language models (VLMs) for multimodal reasoning tasks. However, most existing multimodal reinforcement learning approaches remain limited to spatial reasoning within single-image contexts, yet still struggle to generalize to more complex and real-world scenarios involving multi-image positional reasoning, where understanding the relationships across images is crucial. To address this challenge, we propose a general reinforcement learning approach PeRL tailored for interleaved multimodal tasks, and a multi-stage strategy designed to enhance the exploration-exploitation trade-off, thereby improving learning efficiency and task performance. Specifically, we introduce permutation of image sequences to simulate varied positional relationships to explore more spatial and positional diversity. Furthermore, we design a rollout filtering mechanism for resampling to focus on trajectories that contribute most to learning optimal behaviors to exploit learned policies effectively. We evaluate our model on 5 widely-used multi-image benchmarks and 3 single-image benchmarks. Our experiments confirm that PeRL trained model consistently surpasses R1-related and interleaved VLM baselines by a large margin, achieving state-of-the-art performance on multi-image benchmarks, while preserving comparable performance on single-image tasks. 

---
# Preparing for the Intelligence Explosion 

**Authors**: William MacAskill, Fin Moorhouse  

**Link**: [PDF](https://arxiv.org/pdf/2506.14863)  

**Abstract**: AI that can accelerate research could drive a century of technological progress over just a few years. During such a period, new technological or political developments will raise consequential and hard-to-reverse decisions, in rapid succession. We call these developments grand challenges. These challenges include new weapons of mass destruction, AI-enabled autocracies, races to grab offworld resources, and digital beings worthy of moral consideration, as well as opportunities to dramatically improve quality of life and collective decision-making. We argue that these challenges cannot always be delegated to future AI systems, and suggest things we can do today to meaningfully improve our prospects. AGI preparedness is therefore not just about ensuring that advanced AI systems are aligned: we should be preparing, now, for the disorienting range of developments an intelligence explosion would bring. 

---
# Identifiability by common backdoor in summary causal graphs of time series 

**Authors**: Clément Yvernes, Charles K. Assaad, Emilie Devijver, Eric Gaussier  

**Link**: [PDF](https://arxiv.org/pdf/2506.14862)  

**Abstract**: The identifiability problem for interventions aims at assessing whether the total effect of some given interventions can be written with a do-free formula, and thus be computed from observational data only. We study this problem, considering multiple interventions and multiple effects, in the context of time series when only abstractions of the true causal graph in the form of summary causal graphs are available. We focus in this study on identifiability by a common backdoor set, and establish, for time series with and without consistency throughout time, conditions under which such a set exists. We also provide algorithms of limited complexity to decide whether the problem is identifiable or not. 

---
# BMFM-RNA: An Open Framework for Building and Evaluating Transcriptomic Foundation Models 

**Authors**: Bharath Dandala, Michael M. Danziger, Ella Barkan, Tanwi Biswas, Viatcheslav Gurev, Jianying Hu, Matthew Madgwick, Akira Koseki, Tal Kozlovski, Michal Rosen-Zvi, Yishai Shimoni, Ching-Huei Tsou  

**Link**: [PDF](https://arxiv.org/pdf/2506.14861)  

**Abstract**: Transcriptomic foundation models (TFMs) have recently emerged as powerful tools for analyzing gene expression in cells and tissues, supporting key tasks such as cell-type annotation, batch correction, and perturbation prediction. However, the diversity of model implementations and training strategies across recent TFMs, though promising, makes it challenging to isolate the contribution of individual design choices or evaluate their potential synergies. This hinders the field's ability to converge on best practices and limits the reproducibility of insights across studies. We present BMFM-RNA, an open-source, modular software package that unifies diverse TFM pretraining and fine-tuning objectives within a single framework. Leveraging this capability, we introduce a novel training objective, whole cell expression decoder (WCED), which captures global expression patterns using an autoencoder-like CLS bottleneck representation. In this paper, we describe the framework, supported input representations, and training objectives. We evaluated four model checkpoints pretrained on CELLxGENE using combinations of masked language modeling (MLM), WCED and multitask learning. Using the benchmarking capabilities of BMFM-RNA, we show that WCED-based models achieve performance that matches or exceeds state-of-the-art approaches like scGPT across more than a dozen datasets in both zero-shot and fine-tuning tasks. BMFM-RNA, available as part of the biomed-multi-omics project ( this https URL ), offers a reproducible foundation for systematic benchmarking and community-driven exploration of optimal TFM training strategies, enabling the development of more effective tools to leverage the latest advances in AI for understanding cell biology. 

---
# Peering into the Unknown: Active View Selection with Neural Uncertainty Maps for 3D Reconstruction 

**Authors**: Zhengquan Zhang, Feng Xu, Mengmi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14856)  

**Abstract**: Some perspectives naturally provide more information than others. How can an AI system determine which viewpoint offers the most valuable insight for accurate and efficient 3D object reconstruction? Active view selection (AVS) for 3D reconstruction remains a fundamental challenge in computer vision. The aim is to identify the minimal set of views that yields the most accurate 3D reconstruction. Instead of learning radiance fields, like NeRF or 3D Gaussian Splatting, from a current observation and computing uncertainty for each candidate viewpoint, we introduce a novel AVS approach guided by neural uncertainty maps predicted by a lightweight feedforward deep neural network, named UPNet. UPNet takes a single input image of a 3D object and outputs a predicted uncertainty map, representing uncertainty values across all possible candidate viewpoints. By leveraging heuristics derived from observing many natural objects and their associated uncertainty patterns, we train UPNet to learn a direct mapping from viewpoint appearance to uncertainty in the underlying volumetric representations. Next, our approach aggregates all previously predicted neural uncertainty maps to suppress redundant candidate viewpoints and effectively select the most informative one. Using these selected viewpoints, we train 3D neural rendering models and evaluate the quality of novel view synthesis against other competitive AVS methods. Remarkably, despite using half of the viewpoints than the upper bound, our method achieves comparable reconstruction accuracy. In addition, it significantly reduces computational overhead during AVS, achieving up to a 400 times speedup along with over 50\% reductions in CPU, RAM, and GPU usage compared to baseline methods. Notably, our approach generalizes effectively to AVS tasks involving novel object categories, without requiring any additional training. 

---
# Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation -- Adios low-level controllers 

**Authors**: Tommaso Belvedere, Michael Ziegltrum, Giulio Turrisi, Valerio Modugno  

**Link**: [PDF](https://arxiv.org/pdf/2506.14855)  

**Abstract**: Model Predictive Path Integral control is a powerful sampling-based approach suitable for complex robotic tasks due to its flexibility in handling nonlinear dynamics and non-convex costs. However, its applicability in real-time, highfrequency robotic control scenarios is limited by computational demands. This paper introduces Feedback-MPPI (F-MPPI), a novel framework that augments standard MPPI by computing local linear feedback gains derived from sensitivity analysis inspired by Riccati-based feedback used in gradient-based MPC. These gains allow for rapid closed-loop corrections around the current state without requiring full re-optimization at each timestep. We demonstrate the effectiveness of F-MPPI through simulations and real-world experiments on two robotic platforms: a quadrupedal robot performing dynamic locomotion on uneven terrain and a quadrotor executing aggressive maneuvers with onboard computation. Results illustrate that incorporating local feedback significantly improves control performance and stability, enabling robust, high-frequency operation suitable for complex robotic systems. 

---
# Efficient Retail Video Annotation: A Robust Key Frame Generation Approach for Product and Customer Interaction Analysis 

**Authors**: Varun Mannam, Zhenyu Shi  

**Link**: [PDF](https://arxiv.org/pdf/2506.14854)  

**Abstract**: Accurate video annotation plays a vital role in modern retail applications, including customer behavior analysis, product interaction detection, and in-store activity recognition. However, conventional annotation methods heavily rely on time-consuming manual labeling by human annotators, introducing non-robust frame selection and increasing operational costs. To address these challenges in the retail domain, we propose a deep learning-based approach that automates key-frame identification in retail videos and provides automatic annotations of products and customers. Our method leverages deep neural networks to learn discriminative features by embedding video frames and incorporating object detection-based techniques tailored for retail environments. Experimental results showcase the superiority of our approach over traditional methods, achieving accuracy comparable to human annotator labeling while enhancing the overall efficiency of retail video annotation. Remarkably, our approach leads to an average of 2 times cost savings in video annotation. By allowing human annotators to verify/adjust less than 5% of detected frames in the video dataset, while automating the annotation process for the remaining frames without reducing annotation quality, retailers can significantly reduce operational costs. The automation of key-frame detection enables substantial time and effort savings in retail video labeling tasks, proving highly valuable for diverse retail applications such as shopper journey analysis, product interaction detection, and in-store security monitoring. 

---
# Cost-Efficient Serving of LLM Agents via Test-Time Plan Caching 

**Authors**: Qizheng Zhang, Michael Wornow, Kunle Olukotun  

**Link**: [PDF](https://arxiv.org/pdf/2506.14852)  

**Abstract**: LLM-based agentic applications have shown increasingly remarkable capabilities in complex workflows but incur substantial costs due to extensive planning and reasoning requirements. Existing LLM caching techniques (like context caching and semantic caching), primarily designed for serving chatbots, are insufficient for agentic applications where outputs depend on external data or environmental contexts. We propose agentic plan caching, a novel approach that extracts, stores, adapts, and reuses structured plan templates from planning stages of agentic applications across semantically similar tasks to reduce the cost of serving. Unlike traditional semantic caching, our system extracts plan templates from completed agent executions at test-time, employs keyword extraction to match new requests against cached plans, and utilizes lightweight models to adapt these templates to task-specific plans with contexts. Evaluation across multiple real-world agentic applications shows that our system can reduce costs by 46.62% on average while maintaining performance, offering a more efficient solution for serving LLM-based agents that complements existing LLM serving infrastructures. 

---
# Efficient Serving of LLM Applications with Probabilistic Demand Modeling 

**Authors**: Yifei Liu, Zuo Gan, Zhenghao Gan, Weiye Wang, Chen Chen, Yizhou Shan, Xusheng Chen, Zhenhua Han, Yifei Zhu, Shixuan Sun, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2506.14851)  

**Abstract**: Applications based on Large Language Models (LLMs) contains a series of tasks to address real-world problems with boosted capability, which have dynamic demand volumes on diverse backends. Existing serving systems treat the resource demands of LLM applications as a blackbox, compromising end-to-end efficiency due to improper queuing order and backend warm up latency. We find that the resource demands of LLM applications can be modeled in a general and accurate manner with Probabilistic Demand Graph (PDGraph). We then propose Hermes, which leverages PDGraph for efficient serving of LLM applications. Confronting probabilistic demand description, Hermes applies the Gittins policy to determine the scheduling order that can minimize the average application completion time. It also uses the PDGraph model to help prewarm cold backends at proper moments. Experiments with diverse LLM applications confirm that Hermes can effectively improve the application serving efficiency, reducing the average completion time by over 70% and the P95 completion time by over 80%. 

---
# Finding Optimal Kernel Size and Dimension in Convolutional Neural Networks An Architecture Optimization Approach 

**Authors**: Shreyas Rajeev, B Sathish Babu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14846)  

**Abstract**: Kernel size selection in Convolutional Neural Networks (CNNs) is a critical but often overlooked design decision that affects receptive field, feature extraction, computational cost, and model accuracy. This paper proposes the Best Kernel Size Estimation Function (BKSEF), a mathematically grounded and empirically validated framework for optimal, layer-wise kernel size determination. BKSEF balances information gain, computational efficiency, and accuracy improvements by integrating principles from information theory, signal processing, and learning theory. Extensive experiments on CIFAR-10, CIFAR-100, ImageNet-lite, ChestX-ray14, and GTSRB datasets demonstrate that BKSEF-guided architectures achieve up to 3.1 percent accuracy improvement and 42.8 percent reduction in FLOPs compared to traditional models using uniform 3x3 kernels. Two real-world case studies further validate the approach: one for medical image classification in a cloud-based setup, and another for traffic sign recognition on edge devices. The former achieved enhanced interpretability and accuracy, while the latter reduced latency and model size significantly, with minimal accuracy trade-off. These results show that kernel size can be an active, optimizable parameter rather than a fixed heuristic. BKSEF provides practical heuristics and theoretical support for researchers and developers seeking efficient and application-aware CNN designs. It is suitable for integration into neural architecture search pipelines and real-time systems, offering a new perspective on CNN optimization. 

---
# PictSure: Pretraining Embeddings Matters for In-Context Learning Image Classifiers 

**Authors**: Lukas Schiesser, Cornelius Wolff, Sophie Haas, Simon Pukrop  

**Link**: [PDF](https://arxiv.org/pdf/2506.14842)  

**Abstract**: Building image classification models remains cumbersome in data-scarce domains, where collecting large labeled datasets is impractical. In-context learning (ICL) has emerged as a promising paradigm for few-shot image classification (FSIC), enabling models to generalize across domains without gradient-based adaptation. However, prior work has largely overlooked a critical component of ICL-based FSIC pipelines: the role of image embeddings. In this work, we present PictSure, an ICL framework that places the embedding model -- its architecture, pretraining, and training dynamics -- at the center of analysis. We systematically examine the effects of different visual encoder types, pretraining objectives, and fine-tuning strategies on downstream FSIC performance. Our experiments show that the training success and the out-of-domain performance are highly dependent on how the embedding models are pretrained. Consequently, PictSure manages to outperform existing ICL-based FSIC models on out-of-domain benchmarks that differ significantly from the training distribution, while maintaining comparable results on in-domain tasks. Code can be found at this https URL. 

---
# Improved Iterative Refinement for Chart-to-Code Generation via Structured Instruction 

**Authors**: Chengzhi Xu, Yuyang Wang, Lai Wei, Lichao Sun, Weiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14837)  

**Abstract**: Recently, multimodal large language models (MLLMs) have attracted increasing research attention due to their powerful visual understanding capabilities. While they have achieved impressive results on various vision tasks, their performance on chart-to-code generation remains suboptimal. This task requires MLLMs to generate executable code that can reproduce a given chart, demanding not only precise visual understanding but also accurate translation of visual elements into structured code. Directly prompting MLLMs to perform this complex task often yields unsatisfactory results. To address this challenge, we propose {ChartIR}, an iterative refinement method based on structured instruction. First, we distinguish two tasks: visual understanding and code translation. To accomplish the visual understanding component, we design two types of structured instructions: description and difference. The description instruction captures the visual elements of the reference chart, while the difference instruction characterizes the discrepancies between the reference chart and the generated chart. These instructions effectively transform visual features into language representations, thereby facilitating the subsequent code translation process. Second, we decompose the overall chart generation pipeline into two stages: initial code generation and iterative refinement, enabling progressive enhancement of the final output. Experimental results show that, compared to other method, our method achieves superior performance on both the open-source model Qwen2-VL and the closed-source model GPT-4o. 

---
# Deploying and Evaluating Multiple Deep Learning Models on Edge Devices for Diabetic Retinopathy Detection 

**Authors**: Akwasi Asare, Dennis Agyemanh Nana Gookyi, Derrick Boateng, Fortunatus Aabangbio Wulnye  

**Link**: [PDF](https://arxiv.org/pdf/2506.14834)  

**Abstract**: Diabetic Retinopathy (DR), a leading cause of vision impairment in individuals with diabetes, affects approximately 34.6% of diabetes patients globally, with the number of cases projected to reach 242 million by 2045. Traditional DR diagnosis relies on the manual examination of retinal fundus images, which is both time-consuming and resource intensive. This study presents a novel solution using Edge Impulse to deploy multiple deep learning models for real-time DR detection on edge devices. A robust dataset of over 3,662 retinal fundus images, sourced from the Kaggle EyePACS dataset, was curated, and enhanced through preprocessing techniques, including augmentation and normalization. Using TensorFlow, various Convolutional Neural Networks (CNNs), such as MobileNet, ShuffleNet, SqueezeNet, and a custom Deep Neural Network (DNN), were designed, trained, and optimized for edge deployment. The models were converted to TensorFlowLite and quantized to 8-bit integers to reduce their size and enhance inference speed, with minimal trade-offs in accuracy. Performance evaluations across different edge hardware platforms, including smartphones and microcontrollers, highlighted key metrics such as inference speed, accuracy, precision, and resource utilization. MobileNet achieved an accuracy of 96.45%, while SqueezeNet demonstrated strong real-time performance with a small model size of 176 KB and latency of just 17 ms on GPU. ShuffleNet and the custom DNN achieved moderate accuracy but excelled in resource efficiency, making them suitable for lower-end devices. This integration of edge AI technology into healthcare presents a scalable, cost-effective solution for early DR detection, providing timely and accurate diagnosis, especially in resource-constrained and remote healthcare settings. 

---
# Real-Time, Low-Latency Surveillance Using Entropy-Based Adaptive Buffering and MobileNetV2 on Edge Devices 

**Authors**: Poojashree Chandrashekar Pankaj M Sajjanar  

**Link**: [PDF](https://arxiv.org/pdf/2506.14833)  

**Abstract**: This paper describes a high-performance, low-latency video surveillance system designed for resource-constrained environments. We have proposed a formal entropy-based adaptive frame buffering algorithm and integrated that with MobileNetV2 to achieve high throughput with low latency. The system is capable of processing live streams of video with sub-50ms end-to-end inference latency on resource-constrained devices (embedding platforms) such as Raspberry Pi, Amazon, and NVIDIA Jetson Nano. Our method maintains over 92% detection accuracy on standard datasets focused on video surveillance and exhibits robustness to varying lighting, backgrounds, and speeds. A number of comparative and ablation experiments validate the effectiveness of our design. Finally, our architecture is scalable, inexpensive, and compliant with stricter data privacy regulations than common surveillance systems, so that the system could coexist in a smart city or embedded security architecture. 

---
# ArchShapeNet:An Interpretable 3D-CNN Framework for Evaluating Architectural Shapes 

**Authors**: Jun Yin, Jing Zhong, Pengyu Zeng, Peilin Li, Zixuan Dai, Miao Zhang, Shuai Lu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14832)  

**Abstract**: In contemporary architectural design, the growing complexity and diversity of design demands have made generative plugin tools essential for quickly producing initial concepts and exploring novel 3D forms. However, objectively analyzing the differences between human-designed and machine-generated 3D forms remains a challenge, limiting our understanding of their respective strengths and hindering the advancement of generative tools.
To address this, we built ArchForms-4000, a dataset containing 2,000 architect-designed and 2,000 Evomass-generated 3D forms; Proposed ArchShapeNet, a 3D convolutional neural network tailored for classifying and analyzing architectural forms, incorporating a saliency module to highlight key spatial features aligned with architectural reasoning; And conducted comparative experiments showing our model outperforms human experts in distinguishing form origins, achieving 94.29% accuracy, 96.2% precision, and 98.51% recall.
This study not only highlights the distinctive advantages of human-designed forms in spatial organization, proportional harmony, and detail refinement but also provides valuable insights for enhancing generative design tools in the future. 

---
# Optimization of bi-directional gated loop cell based on multi-head attention mechanism for SSD health state classification model 

**Authors**: Zhizhao Wen, Ruoxin Zhang, Chao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14830)  

**Abstract**: Aiming at the critical role of SSD health state prediction in data reliability assurance, this study proposes a hybrid BiGRU-MHA model that incorporates a multi-head attention mechanism to enhance the accuracy and stability of storage device health classification. The model innovatively integrates temporal feature extraction and key information focusing capabilities. Specifically, it leverages the bidirectional timing modeling advantages of the BiGRU network to capture both forward and backward dependencies of SSD degradation features. Simultaneously, the multi-head attention mechanism dynamically assigns feature weights, improving the model's sensitivity to critical health indicators. Experimental results show that the proposed model achieves classification accuracies of 92.70% on the training set and 92.44% on the test set, with a minimal performance gap of only 0.26%, demonstrating excellent generalization ability. Further analysis using the receiver operating characteristic (ROC) curve shows an area under the curve (AUC) of 0.94 on the test set, confirming the model's robust binary classification performance. This work not only presents a new technical approach for SSD health prediction but also addresses the generalization bottleneck of traditional models, offering a verifiable method with practical value for preventive maintenance of industrial-grade storage systems. The results show the model can significantly reduce data loss risks by providing early failure warnings and help optimize maintenance costs, supporting intelligent decision-making in building reliable storage systems for cloud computing data centers and edge storage environments. 

---
# The Hardness of Achieving Impact in AI for Social Impact Research: A Ground-Level View of Challenges & Opportunities 

**Authors**: Aditya Majumdar, Wenbo Zhang, Kashvi Prawal, Amulya Yadav  

**Link**: [PDF](https://arxiv.org/pdf/2506.14829)  

**Abstract**: In an attempt to tackle the UN SDGs, AI for Social Impact (AI4SI) projects focus on harnessing AI to address societal issues in areas such as healthcare, social justice, etc. Unfortunately, despite growing interest in AI4SI, achieving tangible, on-the-ground impact remains a significant challenge. For example, identifying and engaging motivated collaborators who are willing to co-design and deploy AI based solutions in real-world settings is often difficult. Even when such partnerships are established, many AI4SI projects "fail" to progress beyond the proof-of-concept stage, and hence, are unable to transition to at-scale production-level solutions. Furthermore, the unique challenges faced by AI4SI researchers are not always fully recognized within the broader AI community, where such work is sometimes viewed as primarily applied and not aligning with the traditional criteria for novelty emphasized in core AI venues. This paper attempts to shine a light on the diverse challenges faced in AI4SI research by diagnosing a multitude of factors that prevent AI4SI partnerships from achieving real-world impact on the ground. Drawing on semi-structured interviews with six leading AI4SI researchers - complemented by the authors' own lived experiences in conducting AI4SI research - this paper attempts to understand the day-to-day difficulties faced in developing and deploying socially impactful AI solutions. Through thematic analysis, we identify structural and organizational, communication, collaboration, and operational challenges as key barriers to deployment. While there are no easy fixes, we synthesize best practices and actionable strategies drawn from these interviews and our own work in this space. In doing so, we hope this paper serves as a practical reference guide for AI4SI researchers and partner organizations seeking to engage more effectively in socially impactful AI collaborations. 

---
# DAVID-XR1: Detecting AI-Generated Videos with Explainable Reasoning 

**Authors**: Yifeng Gao, Yifan Ding, Hongyu Su, Juncheng Li, Yunhan Zhao, Lin Luo, Zixing Chen, Li Wang, Xin Wang, Yixu Wang, Xingjun Ma, Yu-Gang Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14827)  

**Abstract**: As AI-generated video becomes increasingly pervasive across media platforms, the ability to reliably distinguish synthetic content from authentic footage has become both urgent and essential. Existing approaches have primarily treated this challenge as a binary classification task, offering limited insight into where or why a model identifies a video as AI-generated. However, the core challenge extends beyond simply detecting subtle artifacts; it requires providing fine-grained, persuasive evidence that can convince auditors and end-users alike. To address this critical gap, we introduce DAVID-X, the first dataset to pair AI-generated videos with detailed defect-level, temporal-spatial annotations and written rationales. Leveraging these rich annotations, we present DAVID-XR1, a video-language model designed to deliver an interpretable chain of visual reasoning-including defect categorization, temporal-spatial localization, and natural language explanations. This approach fundamentally transforms AI-generated video detection from an opaque black-box decision into a transparent and verifiable diagnostic process. We demonstrate that a general-purpose backbone, fine-tuned on our compact dataset and enhanced with chain-of-thought distillation, achieves strong generalization across a variety of generators and generation modes. Our results highlight the promise of explainable detection methods for trustworthy identification of AI-generated video content. 

---
# Collaborative Interest-aware Graph Learning for Group Identification 

**Authors**: Rui Zhao, Beihong Jin, Beibei Li, Yiyuan Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2506.14826)  

**Abstract**: With the popularity of social media, an increasing number of users are joining group activities on online social platforms. This elicits the requirement of group identification (GI), which is to recommend groups to users. We reveal that users are influenced by both group-level and item-level interests, and these dual-level interests have a collaborative evolution relationship: joining a group expands the user's item interests, further prompting the user to join new groups. Ultimately, the two interests tend to align dynamically. However, existing GI methods fail to fully model this collaborative evolution relationship, ignoring the enhancement of group-level interests on item-level interests, and suffering from false-negative samples when aligning cross-level interests. In order to fully model the collaborative evolution relationship between dual-level user interests, we propose CI4GI, a Collaborative Interest-aware model for Group Identification. Specifically, we design an interest enhancement strategy that identifies additional interests of users from the items interacted with by the groups they have joined as a supplement to item-level interests. In addition, we adopt the distance between interest distributions of two users to optimize the identification of negative samples for a user, mitigating the interference of false-negative samples during cross-level interests alignment. The results of experiments on three real-world datasets demonstrate that CI4GI significantly outperforms state-of-the-art models. 

---
# GraphGSOcc: Semantic and Geometric Graph Transformer for 3D Gaussian Splating-based Occupancy Prediction 

**Authors**: Ke Song, Yunhe Wu, Chunchit Siu, Huiyuan Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2506.14825)  

**Abstract**: Addressing the task of 3D semantic occupancy prediction for autonomous driving, we tackle two key issues in existing 3D Gaussian Splating (3DGS) methods: (1) unified feature aggregation neglecting semantic correlations among similar categories and across regions, and (2) boundary ambiguities caused by the lack of geometric constraints in MLP iterative optimization. We propose the GraphGSOcc model, a novel framework that combines semantic and geometric graph Transformer for 3D Gaussian Splating-based Occupancy Prediction. We propose the Dual Gaussians Graph Attenntion, which dynamically constructs dual graph structures: a geometric graph adaptively calculating KNN search radii based on Gaussian poses, enabling large-scale Gaussians to aggregate features from broader neighborhoods while compact Gaussians focus on local geometric consistency; a semantic graph retaining top-M highly correlated nodes via cosine similarity to explicitly encode semantic relationships within and across instances. Coupled with the Multi-scale Graph Attention framework, fine-grained attention at lower layers optimizes boundary details, while coarse-grained attention at higher layers models object-level topology. Experiments on the SurroundOcc dataset achieve an mIoU of 24.10%, reducing GPU memory to 6.1 GB, demonstrating a 1.97% mIoU improvement and 13.7% memory reduction compared to GaussianWorld 

---
# FedNano: Toward Lightweight Federated Tuning for Pretrained Multimodal Large Language Models 

**Authors**: Yao Zhang, Hewei Gao, Haokun Chen, Weiguo Li, Yunpu Ma, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2506.14824)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel in tasks like multimodal reasoning and cross-modal retrieval but face deployment challenges in real-world scenarios due to distributed multimodal data and strict privacy requirements. Federated Learning (FL) offers a solution by enabling collaborative model training without centralizing data. However, realizing FL for MLLMs presents significant challenges, including high computational demands, limited client capacity, substantial communication costs, and heterogeneous client data. Existing FL methods assume client-side deployment of full models, an assumption that breaks down for large-scale MLLMs due to their massive size and communication demands. To address these limitations, we propose FedNano, the first FL framework that centralizes the LLM on the server while introducing NanoEdge, a lightweight module for client-specific adaptation. NanoEdge employs modality-specific encoders, connectors, and trainable NanoAdapters with low-rank adaptation. This design eliminates the need to deploy LLM on clients, reducing client-side storage by 95%, and limiting communication overhead to only 0.01% of the model parameters. By transmitting only compact NanoAdapter updates, FedNano handles heterogeneous client data and resource constraints while preserving privacy. Experiments demonstrate that FedNano outperforms prior FL baselines, bridging the gap between MLLM scale and FL feasibility, and enabling scalable, decentralized multimodal AI systems. 

---
# ViLLa: A Neuro-Symbolic approach for Animal Monitoring 

**Authors**: Harsha Koduri  

**Link**: [PDF](https://arxiv.org/pdf/2506.14823)  

**Abstract**: Monitoring animal populations in natural environments requires systems that can interpret both visual data and human language queries. This work introduces ViLLa (Vision-Language-Logic Approach), a neuro-symbolic framework designed for interpretable animal monitoring. ViLLa integrates three core components: a visual detection module for identifying animals and their spatial locations in images, a language parser for understanding natural language queries, and a symbolic reasoning layer that applies logic-based inference to answer those queries. Given an image and a question such as "How many dogs are in the scene?" or "Where is the buffalo?", the system grounds visual detections into symbolic facts and uses predefined rules to compute accurate answers related to count, presence, and location. Unlike end-to-end black-box models, ViLLa separates perception, understanding, and reasoning, offering modularity and transparency. The system was evaluated on a range of animal imagery tasks and demonstrates the ability to bridge visual content with structured, human-interpretable queries. 

---
# Reinforcing VLMs to Use Tools for Detailed Visual Reasoning Under Resource Constraints 

**Authors**: Sunil Kumar, Bowen Zhao, Leo Dirac, Paulina Varshavskaya  

**Link**: [PDF](https://arxiv.org/pdf/2506.14821)  

**Abstract**: Despite tremendous recent advances in large model reasoning ability, vision-language models (VLMs) still struggle with detailed visual reasoning, especially when compute resources are limited. To address this challenge, we draw inspiration from methods like Deepseek-r1 for VLMs and train smaller-scale models with Group Relative Policy Optimization (GRPO) to use external tools such as zoom. The greatest benefit is obtained with a combination of GRPO learning, a simple reward structure, a simplified tool-calling interface, allocating additional tokens to the result of the tool call, and a training data mix that over-represents visually difficult examples. Compared to similarly-sized baseline models, our method achieves better performance on some visual question-answering (VQA) tasks, thanks to the detailed visual information gathered from the external tool. 

---
# Next-Generation Conflict Forecasting: Unleashing Predictive Patterns through Spatiotemporal Learning 

**Authors**: Simon P. von der Maase  

**Link**: [PDF](https://arxiv.org/pdf/2506.14817)  

**Abstract**: Forecasting violent conflict at high spatial and temporal resolution remains a central challenge for both researchers and policymakers. This study presents a novel neural network architecture for forecasting three distinct types of violence -- state-based, non-state, and one-sided -- at the subnational (priogrid-month) level, up to 36 months in advance. The model jointly performs classification and regression tasks, producing both probabilistic estimates and expected magnitudes of future events. It achieves state-of-the-art performance across all tasks and generates approximate predictive posterior distributions to quantify forecast uncertainty.
The architecture is built on a Monte Carlo Dropout Long Short-Term Memory (LSTM) U-Net, integrating convolutional layers to capture spatial dependencies with recurrent structures to model temporal dynamics. Unlike many existing approaches, it requires no manual feature engineering and relies solely on historical conflict data. This design enables the model to autonomously learn complex spatiotemporal patterns underlying violent conflict.
Beyond achieving state-of-the-art predictive performance, the model is also highly extensible: it can readily integrate additional data sources and jointly forecast auxiliary variables. These capabilities make it a promising tool for early warning systems, humanitarian response planning, and evidence-based peacebuilding initiatives. 

---
# Training with Confidence: Catching Silent Errors in Deep Learning Training with Automated Proactive Checks 

**Authors**: Yuxuan Jiang, Ziming Zhou, Boyu Xu, Beijie Liu, Runhui Xu, Peng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14813)  

**Abstract**: Training deep learning (DL) models is a complex process, making it prone to silent errors that are challenging to detect and diagnose. This paper presents TRAINCHECK, a framework that takes a proactive checking approach to address silent training errors. TRAINCHECK automatically infers invariants tailored for DL training. It uses these invariants to proactively detect silent errors during the training process while providing debugging help. To evaluate TRAINCHECK, we reproduce 20 real-world silent training errors with diverse root causes. TRAINCHECK successfully detects 18 errors within a single training iteration. It also uncovers 6 unknown bugs in popular training libraries that lead to silent errors. 

---
# Argus Inspection: Do Multimodal Large Language Models Possess the Eye of Panoptes? 

**Authors**: Yang Yao, Lingyu Li, Jiaxin Song, Chiyu Chen, Zhenqi He, Yixu Wang, Xin Wang, Tianle Gu, Jie Li, Yan Teng, Yingchun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.14805)  

**Abstract**: As Multimodal Large Language Models (MLLMs) continue to evolve, their cognitive and reasoning capabilities have seen remarkable progress. However, challenges in visual fine-grained perception and commonsense causal inference persist. This paper introduces Argus Inspection, a multimodal benchmark with two levels of difficulty, emphasizing detailed visual recognition while incorporating real-world commonsense understanding to evaluate causal reasoning abilities. Expanding on it, we present the Eye of Panoptes framework, which integrates a binary parametric Sigmoid metric with an indicator function, enabling a more holistic evaluation of MLLMs' responses in opinion-based reasoning tasks. Experiments conducted on 26 mainstream MLLMs reveal that the highest performance in visual fine-grained reasoning reaches only 0.46, highlighting considerable potential for enhancement. Our research offers valuable perspectives for the continued refinement of MLLMs. 

---
# ss-Mamba: Semantic-Spline Selective State-Space Model 

**Authors**: Zuochen Ye  

**Link**: [PDF](https://arxiv.org/pdf/2506.14802)  

**Abstract**: We propose ss-Mamba, a novel foundation model that enhances time series forecasting by integrating semantic-aware embeddings and adaptive spline-based temporal encoding within a selective state-space modeling framework. Building upon the recent success of Transformer architectures, ss-Mamba adopts the Mamba selective state space model as an efficient alternative that achieves comparable performance while significantly reducing computational complexity from quadratic to linear time. Semantic index embeddings, initialized from pretrained language models, allow effective generalization to previously unseen series through meaningful semantic priors. Additionally, spline-based Kolmogorov-Arnold Networks (KAN) dynamically and interpretably capture complex seasonalities and non-stationary temporal effects, providing a powerful enhancement over conventional temporal feature encodings. Extensive experimental evaluations confirm that ss-Mamba delivers superior accuracy, robustness, and interpretability, demonstrating its capability as a versatile and computationally efficient alternative to traditional Transformer-based models in time-series forecasting. 

---
# Analyzing Character Representation in Media Content using Multimodal Foundation Model: Effectiveness and Trust 

**Authors**: Evdoxia Taka, Debadyuti Bhattacharya, Joanne Garde-Hansen, Sanjay Sharma, Tanaya Guha  

**Link**: [PDF](https://arxiv.org/pdf/2506.14799)  

**Abstract**: Recent advances in AI has enabled automated analysis of complex media content at scale and generate actionable insights regarding character representation along such dimensions as gender and age. Past work focused on quantifying representation from audio/video/text using various ML models, but without having the audience in the loop. We ask, even if character distribution along demographic dimensions are available, how useful are they to the general public? Do they actually trust the numbers generated by AI models? Our work addresses these questions through a user study, while proposing a new AI-based character representation and visualization tool. Our tool based on the Contrastive Language Image Pretraining (CLIP) foundation model to analyze visual screen data to quantify character representation across dimensions of age and gender. We also designed effective visualizations suitable for presenting such analytics to lay audience. Next, we conducted a user study to seek empirical evidence on the usefulness and trustworthiness of the AI-generated results for carefully chosen movies presented in the form of our visualizations. We note that participants were able to understand the analytics from our visualization, and deemed the tool `overall useful'. Participants also indicated a need for more detailed visualizations to include more demographic categories and contextual information of the characters. Participants' trust in AI-based gender and age models is seen to be moderate to low, although they were not against the use of AI in this context. Our tool including code, benchmarking, and data from the user study can be found here: this https URL 

---
# MODS: Multi-source Observations Conditional Diffusion Model for Meteorological State Downscaling 

**Authors**: Siwei Tu, Jingyi Xu, Weidong Yang, Lei Bai, Ben Fei  

**Link**: [PDF](https://arxiv.org/pdf/2506.14798)  

**Abstract**: Accurate acquisition of high-resolution surface meteorological conditions is critical for forecasting and simulating meteorological variables. Directly applying spatial interpolation methods to derive meteorological values at specific locations from low-resolution grid fields often yields results that deviate significantly from the actual conditions. Existing downscaling methods primarily rely on the coupling relationship between geostationary satellites and ERA5 variables as a condition. However, using brightness temperature data from geostationary satellites alone fails to comprehensively capture all the changes in meteorological variables in ERA5 maps. To address this limitation, we can use a wider range of satellite data to make more full use of its inversion effects on various meteorological variables, thus producing more realistic results across different meteorological variables. To further improve the accuracy of downscaling meteorological variables at any location, we propose the Multi-source Observation Down-Scaling Model (MODS). It is a conditional diffusion model that fuses data from multiple geostationary satellites GridSat, polar-orbiting satellites (AMSU-A, HIRS, and MHS), and topographic data (GEBCO), as conditions, and is pre-trained on the ERA5 reanalysis dataset. During training, latent features from diverse conditional inputs are extracted separately and fused into ERA5 maps via a multi-source cross-attention module. By exploiting the inversion relationships between reanalysis data and multi-source atmospheric variables, MODS generates atmospheric states that align more closely with real-world conditions. During sampling, MODS enhances downscaling consistency by incorporating low-resolution ERA5 maps and station-level meteorological data as guidance. Experimental results demonstrate that MODS achieves higher fidelity when downscaling ERA5 maps to a 6.25 km resolution. 

---
# Bound by semanticity: universal laws governing the generalization-identification tradeoff 

**Authors**: Marco Nurisso, Jesseba Fernando, Raj Deshpande, Alan Perotti, Raja Marjieh, Steven M. Frankland, Richard L. Lewis, Taylor W. Webb, Declan Campbell, Francesco Vaccarino, Jonathan D. Cohen, Giovanni Petri  

**Link**: [PDF](https://arxiv.org/pdf/2506.14797)  

**Abstract**: Intelligent systems must deploy internal representations that are simultaneously structured -- to support broad generalization -- and selective -- to preserve input identity. We expose a fundamental limit on this tradeoff. For any model whose representational similarity between inputs decays with finite semantic resolution $\varepsilon$, we derive closed-form expressions that pin its probability of correct generalization $p_S$ and identification $p_I$ to a universal Pareto front independent of input space geometry. Extending the analysis to noisy, heterogeneous spaces and to $n>2$ inputs predicts a sharp $1/n$ collapse of multi-input processing capacity and a non-monotonic optimum for $p_S$. A minimal ReLU network trained end-to-end reproduces these laws: during learning a resolution boundary self-organizes and empirical $(p_S,p_I)$ trajectories closely follow theoretical curves for linearly decaying similarity. Finally, we demonstrate that the same limits persist in two markedly more complex settings -- a convolutional neural network and state-of-the-art vision-language models -- confirming that finite-resolution similarity is a fundamental emergent informational constraint, not merely a toy-model artifact. Together, these results provide an exact theory of the generalization-identification trade-off and clarify how semantic resolution shapes the representational capacity of deep networks and brains alike. 

---
# PFMBench: Protein Foundation Model Benchmark 

**Authors**: Zhangyang Gao, Hao Wang, Cheng Tan, Chenrui Xu, Mengdi Liu, Bozhen Hu, Linlin Chao, Xiaoming Zhang, Stan Z. Li  

**Link**: [PDF](https://arxiv.org/pdf/2506.14796)  

**Abstract**: This study investigates the current landscape and future directions of protein foundation model research. While recent advancements have transformed protein science and engineering, the field lacks a comprehensive benchmark for fair evaluation and in-depth understanding. Since ESM-1B, numerous protein foundation models have emerged, each with unique datasets and methodologies. However, evaluations often focus on limited tasks tailored to specific models, hindering insights into broader generalization and limitations. Specifically, researchers struggle to understand the relationships between tasks, assess how well current models perform across them, and determine the criteria in developing new foundation models. To fill this gap, we present PFMBench, a comprehensive benchmark evaluating protein foundation models across 38 tasks spanning 8 key areas of protein science. Through hundreds of experiments on 17 state-of-the-art models across 38 tasks, PFMBench reveals the inherent correlations between tasks, identifies top-performing models, and provides a streamlined evaluation protocol. Code is available at \href{this https URL}{\textcolor{blue}{GitHub}}. 

---
# Comparative Analysis of QNN Architectures for Wind Power Prediction: Feature Maps and Ansatz Configurations 

**Authors**: Batuhan Hangun, Emine Akpinar, Oguz Altun, Onder Eyecioglu  

**Link**: [PDF](https://arxiv.org/pdf/2506.14795)  

**Abstract**: Quantum Machine Learning (QML) is an emerging field at the intersection of quantum computing and machine learning, aiming to enhance classical machine learning methods by leveraging quantum mechanics principles such as entanglement and superposition. However, skepticism persists regarding the practical advantages of QML, mainly due to the current limitations of noisy intermediate-scale quantum (NISQ) devices. This study addresses these concerns by extensively assessing Quantum Neural Networks (QNNs)-quantum-inspired counterparts of Artificial Neural Networks (ANNs), demonstrating their effectiveness compared to classical methods. We systematically construct and evaluate twelve distinct QNN configurations, utilizing two unique quantum feature maps combined with six different entanglement strategies for ansatz design. Experiments conducted on a wind energy dataset reveal that QNNs employing the Z feature map achieve up to 93% prediction accuracy when forecasting wind power output using only four input parameters. Our findings show that QNNs outperform classical methods in predictive tasks, underscoring the potential of QML in real-world applications. 

---
# Assembly of Experts: Linear-time construction of the Chimera LLM variants with emergent and adaptable behaviors 

**Authors**: Henrik Klagges, Robert Dahlke, Fabian Klemm, Benjamin Merkel, Daniel Klingmann, David A. Reiss, Dan Zecha  

**Link**: [PDF](https://arxiv.org/pdf/2506.14794)  

**Abstract**: Requiring $10^{13}$-$10^{15}$ FLOPs to calculate one 8 bit weight in an LLM during pretraining is extremely expensive and seems inefficient. To better leverage the huge investments made into pretrained models, we develop the new "Assembly-of-Experts" (AoE) construction method to create capable child variants of existing Mixture-of-Experts parent models in linear time. Model weight tensors get interpolated individually, allowing to enhance or suppress semantic features of the parents.
Varying the proportion of weights taken from the parent models, we observe some properties of the AoE child model changing gradually, while other behavioral traits emerge with a sharp transition. Surprisingly, nearly every generated model is functional and capable, which makes searching the model space straightforward.
We construct the DeepSeek R1T "Chimera", a 671B open-weights hybrid model combining DeepSeek's V3-0324 and R1 model variants. The child inherits only the routed expert tensors of R1, but still achieves about R1-level intelligence. At the same time, it uses about 40\% fewer output tokens, close to V3 speed. Constructed without any fine-tuning or distillation, the Chimera exhibits surprisingly compact, orderly reasoning compared to its parent models. 

---
# Topology-Aware and Highly Generalizable Deep Reinforcement Learning for Efficient Retrieval in Multi-Deep Storage Systems 

**Authors**: Funing Li, Yuan Tian, Ruben Noortwyck, Jifeng Zhou, Liming Kuang, Robert Schulz  

**Link**: [PDF](https://arxiv.org/pdf/2506.14787)  

**Abstract**: In modern industrial and logistics environments, the rapid expansion of fast delivery services has heightened the demand for storage systems that combine high efficiency with increased density. Multi-deep autonomous vehicle storage and retrieval systems (AVS/RS) present a viable solution for achieving greater storage density. However, these systems encounter significant challenges during retrieval operations due to lane blockages. A conventional approach to mitigate this issue involves storing items with homogeneous characteristics in a single lane, but this strategy restricts the flexibility and adaptability of multi-deep storage systems.
In this study, we propose a deep reinforcement learning-based framework to address the retrieval problem in multi-deep storage systems with heterogeneous item configurations. Each item is associated with a specific due date, and the objective is to minimize total tardiness. To effectively capture the system's topology, we introduce a graph-based state representation that integrates both item attributes and the local topological structure of the multi-deep warehouse. To process this representation, we design a novel neural network architecture that combines a Graph Neural Network (GNN) with a Transformer model. The GNN encodes topological and item-specific information into embeddings for all directly accessible items, while the Transformer maps these embeddings into global priority assignments. The Transformer's strong generalization capability further allows our approach to be applied to storage systems with diverse layouts. Extensive numerical experiments, including comparisons with heuristic methods, demonstrate the superiority of the proposed neural network architecture and the effectiveness of the trained agent in optimizing retrieval tardiness. 

---
# PIPE: Physics-Informed Position Encoding for Alignment of Satellite Images and Time Series 

**Authors**: Haobo Li, Eunseo Jung, Zixin Chen, Zhaowei Wang, Yueya Wang, Huamin Qu, Alexis Kai Hon Lau  

**Link**: [PDF](https://arxiv.org/pdf/2506.14786)  

**Abstract**: Multimodal time series forecasting is foundational in various fields, such as utilizing satellite imagery and numerical data for predicting typhoons in climate science. However, existing multimodal approaches primarily focus on utilizing text data to help time series forecasting, leaving the visual data in existing time series datasets untouched. Furthermore, it is challenging for models to effectively capture the physical information embedded in visual data, such as satellite imagery's temporal and geospatial context, which extends beyond images themselves. To address this gap, we propose physics-informed positional encoding (PIPE), a lightweight method that embeds physical information into vision language models (VLMs). PIPE introduces two key innovations: (1) a physics-informed positional indexing scheme for mapping physics to positional IDs, and (2) a variant-frequency positional encoding mechanism for encoding frequency information of physical variables and sequential order of tokens within the embedding space. By preserving both the physical information and sequential order information, PIPE significantly improves multimodal alignment and forecasting accuracy. Through the experiments on the most representative and the largest open-sourced satellite image dataset, PIPE achieves state-of-the-art performance in both deep learning forecasting and climate domain methods, demonstrating superiority across benchmarks, including a 12% improvement in typhoon intensity forecasting over prior works. Our code is provided in the supplementary material. 

---
# WebXAII: an open-source web framework to study human-XAI interaction 

**Authors**: Jules Leguy, Pierre-Antoine Jean, Felipe Torres Figueroa, Sébastien Harispe  

**Link**: [PDF](https://arxiv.org/pdf/2506.14777)  

**Abstract**: This article introduces WebXAII, an open-source web framework designed to facilitate research on human interaction with eXplainable Artificial Intelligence (XAI) systems. The field of XAI is rapidly expanding, driven by the growing societal implications of the widespread adoption of AI (and in particular machine learning) across diverse applications. Researchers who study the interaction between humans and XAI techniques typically develop ad hoc interfaces in order to conduct their studies. These interfaces are usually not shared alongside the results of the studies, which limits their reusability and the reproducibility of experiments. In response, we design and implement WebXAII, a web-based platform that can embody full experimental protocols, meaning that it can present all aspects of the experiment to human participants and record their responses. The experimental protocols are translated into a composite architecture of generic views and modules, which offers a lot of flexibility. The architecture is defined in a structured configuration file, so that protocols can be implemented with minimal programming skills. We demonstrate that WebXAII can effectively embody relevant protocols, by reproducing the protocol of a state-of-the-art study of the literature. The framework is available at this https URL. 

---
# See What I Mean? CUE: A Cognitive Model of Understanding Explanations 

**Authors**: Tobias Labarta, Nhi Hoang, Katharina Weitz, Wojciech Samek, Sebastian Lapuschkin, Leander Weber  

**Link**: [PDF](https://arxiv.org/pdf/2506.14775)  

**Abstract**: As machine learning systems increasingly inform critical decisions, the need for human-understandable explanations grows. Current evaluations of Explainable AI (XAI) often prioritize technical fidelity over cognitive accessibility which critically affects users, in particular those with visual impairments. We propose CUE, a model for Cognitive Understanding of Explanations, linking explanation properties to cognitive sub-processes: legibility (perception), readability (comprehension), and interpretability (interpretation). In a study (N=455) testing heatmaps with varying colormaps (BWR, Cividis, Coolwarm), we found comparable task performance but lower confidence/effort for visually impaired users. Unlike expected, these gaps were not mitigated and sometimes worsened by accessibility-focused color maps like Cividis. These results challenge assumptions about perceptual optimization and support the need for adaptive XAI interfaces. They also validate CUE by demonstrating that altering explanation legibility affects understandability. We contribute: (1) a formalized cognitive model for explanation understanding, (2) an integrated definition of human-centered explanation properties, and (3) empirical evidence motivating accessible, user-tailored XAI. 

---
# MedSyn: Enhancing Diagnostics with Human-AI Collaboration 

**Authors**: Burcu Sayin, Ipek Baris Schlicht, Ngoc Vo Hong, Sara Allievi, Jacopo Staiano, Pasquale Minervini, Andrea Passerini  

**Link**: [PDF](https://arxiv.org/pdf/2506.14774)  

**Abstract**: Clinical decision-making is inherently complex, often influenced by cognitive biases, incomplete information, and case ambiguity. Large Language Models (LLMs) have shown promise as tools for supporting clinical decision-making, yet their typical one-shot or limited-interaction usage may overlook the complexities of real-world medical practice. In this work, we propose a hybrid human-AI framework, MedSyn, where physicians and LLMs engage in multi-step, interactive dialogues to refine diagnoses and treatment decisions. Unlike static decision-support tools, MedSyn enables dynamic exchanges, allowing physicians to challenge LLM suggestions while the LLM highlights alternative perspectives. Through simulated physician-LLM interactions, we assess the potential of open-source LLMs as physician assistants. Results show open-source LLMs are promising as physician assistants in the real world. Future work will involve real physician interactions to further validate MedSyn's usefulness in diagnostic accuracy and patient outcomes. 

---
# MixRep: Hidden Representation Mixup for Low-Resource Speech Recognition 

**Authors**: Jiamin Xie, John H.L. Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2310.18450)  

**Abstract**: In this paper, we present MixRep, a simple and effective data augmentation strategy based on mixup for low-resource ASR. MixRep interpolates the feature dimensions of hidden representations in the neural network that can be applied to both the acoustic feature input and the output of each layer, which generalizes the previous MixSpeech method. Further, we propose to combine the mixup with a regularization along the time axis of the input, which is shown as complementary. We apply MixRep to a Conformer encoder of an E2E LAS architecture trained with a joint CTC loss. We experiment on the WSJ dataset and subsets of the SWB dataset, covering reading and telephony conversational speech. Experimental results show that MixRep consistently outperforms other regularization methods for low-resource ASR. Compared to a strong SpecAugment baseline, MixRep achieves a +6.5\% and a +6.7\% relative WER reduction on the eval92 set and the Callhome part of the eval'2000 set. 

---
# Dynamic ASR Pathways: An Adaptive Masking Approach Towards Efficient Pruning of A Multilingual ASR Model 

**Authors**: Jiamin Xie, Ke Li, Jinxi Guo, Andros Tjandra, Yuan Shangguan, Leda Sari, Chunyang Wu, Junteng Jia, Jay Mahadeokar, Ozlem Kalinli  

**Link**: [PDF](https://arxiv.org/pdf/2309.13018)  

**Abstract**: Neural network pruning offers an effective method for compressing a multilingual automatic speech recognition (ASR) model with minimal performance loss. However, it entails several rounds of pruning and re-training needed to be run for each language. In this work, we propose the use of an adaptive masking approach in two scenarios for pruning a multilingual ASR model efficiently, each resulting in sparse monolingual models or a sparse multilingual model (named as Dynamic ASR Pathways). Our approach dynamically adapts the sub-network, avoiding premature decisions about a fixed sub-network structure. We show that our approach outperforms existing pruning methods when targeting sparse monolingual models. Further, we illustrate that Dynamic ASR Pathways jointly discovers and trains better sub-networks (pathways) of a single multilingual model by adapting from different sub-network initializations, thereby reducing the need for language-specific pruning. 

---
# DEFORMER: Coupling Deformed Localized Patterns with Global Context for Robust End-to-end Speech Recognition 

**Authors**: Jiamin Xie, John H.L. Hansen  

**Link**: [PDF](https://arxiv.org/pdf/2207.01732)  

**Abstract**: Convolutional neural networks (CNN) have improved speech recognition performance greatly by exploiting localized time-frequency patterns. But these patterns are assumed to appear in symmetric and rigid kernels by the conventional CNN operation. It motivates the question: What about asymmetric kernels? In this study, we illustrate adaptive views can discover local features which couple better with attention than fixed views of the input. We replace depthwise CNNs in the Conformer architecture with a deformable counterpart, dubbed this "Deformer". By analyzing our best-performing model, we visualize both local receptive fields and global attention maps learned by the Deformer and show increased feature associations on the utterance level. The statistical analysis of learned kernel offsets provides an insight into the change of information in features with the network depth. Finally, replacing only half of the layers in the encoder, the Deformer improves +5.6% relative WER without a LM and +6.4% relative WER with a LM over the Conformer baseline on the WSJ eval92 set. 

---
