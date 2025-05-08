# Qualitative Analysis of $ω$-Regular Objectives on Robust MDPs 

**Authors**: Ali Asadi, Krishnendu Chatterjee, Ehsan Kafshdar Goharshady, Mehrdad Karrabi, Ali Shafiee  

**Link**: [PDF](https://arxiv.org/pdf/2505.04539)  

**Abstract**: Robust Markov Decision Processes (RMDPs) generalize classical MDPs that consider uncertainties in transition probabilities by defining a set of possible transition functions. An objective is a set of runs (or infinite trajectories) of the RMDP, and the value for an objective is the maximal probability that the agent can guarantee against the adversarial environment. We consider (a) reachability objectives, where given a target set of states, the goal is to eventually arrive at one of them; and (b) parity objectives, which are a canonical representation for $\omega$-regular objectives. The qualitative analysis problem asks whether the objective can be ensured with probability 1.
In this work, we study the qualitative problem for reachability and parity objectives on RMDPs without making any assumption over the structures of the RMDPs, e.g., unichain or aperiodic. Our contributions are twofold. We first present efficient algorithms with oracle access to uncertainty sets that solve qualitative problems of reachability and parity objectives. We then report experimental results demonstrating the effectiveness of our oracle-based approach on classical RMDP examples from the literature scaling up to thousands of states. 

---
# Beyond Theorem Proving: Formulation, Framework and Benchmark for Formal Problem-Solving 

**Authors**: Qi Liu, Xinhao Zheng, Renqiu Xia, Xingzhi Qi, Qinxiang Cao, Junchi Yan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04528)  

**Abstract**: As a seemingly self-explanatory task, problem-solving has been a significant component of science and engineering. However, a general yet concrete formulation of problem-solving itself is missing. With the recent development of AI-based problem-solving agents, the demand for process-level verifiability is rapidly increasing yet underexplored. To fill these gaps, we present a principled formulation of problem-solving as a deterministic Markov decision process; a novel framework, FPS (Formal Problem-Solving), which utilizes existing FTP (formal theorem proving) environments to perform process-verified problem-solving; and D-FPS (Deductive FPS), decoupling solving and answer verification for better human-alignment. The expressiveness, soundness and completeness of the frameworks are proven. We construct three benchmarks on problem-solving: FormalMath500, a formalization of a subset of the MATH500 benchmark; MiniF2F-Solving and PutnamBench-Solving, adaptations of FTP benchmarks MiniF2F and PutnamBench. For faithful, interpretable, and human-aligned evaluation, we propose RPE (Restricted Propositional Equivalence), a symbolic approach to determine the correctness of answers by formal verification. We evaluate four prevalent FTP models and two prompting methods as baselines, solving at most 23.77% of FormalMath500, 27.47% of MiniF2F-Solving, and 0.31% of PutnamBench-Solving. 

---
# On some improvements to Unbounded Minimax 

**Authors**: Quentin Cohen-Solal, Tristan Cazenave  

**Link**: [PDF](https://arxiv.org/pdf/2505.04525)  

**Abstract**: This paper presents the first experimental evaluation of four previously untested modifications of Unbounded Best-First Minimax algorithm. This algorithm explores the game tree by iteratively expanding the most promising sequences of actions based on the current partial game tree. We first evaluate the use of transposition tables, which convert the game tree into a directed acyclic graph by merging duplicate states. Second, we compare the original algorithm by Korf & Chickering with the variant proposed by Cohen-Solal, which differs in its backpropagation strategy: instead of stopping when a stable value is encountered, it updates values up to the root. This change slightly improves performance when value ties or transposition tables are involved. Third, we assess replacing the exact terminal evaluation function with the learned heuristic function. While beneficial when exact evaluations are costly, this modification reduces performance in inexpensive settings. Finally, we examine the impact of the completion technique that prioritizes resolved winning states and avoids resolved losing states. This technique also improves performance. Overall, our findings highlight how targeted modifications can enhance the efficiency of Unbounded Best-First Minimax. 

---
# TrajEvo: Designing Trajectory Prediction Heuristics via LLM-driven Evolution 

**Authors**: Zhikai Zhao, Chuanbo Hua, Federico Berto, Kanghoon Lee, Zihan Ma, Jiachen Li, Jinkyoo Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.04480)  

**Abstract**: Trajectory prediction is a crucial task in modeling human behavior, especially in fields as social robotics and autonomous vehicle navigation. Traditional heuristics based on handcrafted rules often lack accuracy, while recently proposed deep learning approaches suffer from computational cost, lack of explainability, and generalization issues that limit their practical adoption. In this paper, we introduce TrajEvo, a framework that leverages Large Language Models (LLMs) to automatically design trajectory prediction heuristics. TrajEvo employs an evolutionary algorithm to generate and refine prediction heuristics from past trajectory data. We introduce a Cross-Generation Elite Sampling to promote population diversity and a Statistics Feedback Loop allowing the LLM to analyze alternative predictions. Our evaluations show TrajEvo outperforms previous heuristic methods on the ETH-UCY datasets, and remarkably outperforms both heuristics and deep learning methods when generalizing to the unseen SDD dataset. TrajEvo represents a first step toward automated design of fast, explainable, and generalizable trajectory prediction heuristics. We make our source code publicly available to foster future research at this https URL. 

---
# Uncertain Machine Ethics Planning 

**Authors**: Simon Kolker, Louise A. Dennis, Ramon Fraga Pereira, Mengwei Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04352)  

**Abstract**: Machine Ethics decisions should consider the implications of uncertainty over decisions. Decisions should be made over sequences of actions to reach preferable outcomes long term. The evaluation of outcomes, however, may invoke one or more moral theories, which might have conflicting judgements. Each theory will require differing representations of the ethical situation. For example, Utilitarianism measures numerical values, Deontology analyses duties, and Virtue Ethics emphasises moral character. While balancing potentially conflicting moral considerations, decisions may need to be made, for example, to achieve morally neutral goals with minimal costs. In this paper, we formalise the problem as a Multi-Moral Markov Decision Process and a Multi-Moral Stochastic Shortest Path Problem. We develop a heuristic algorithm based on Multi-Objective AO*, utilising Sven-Ove Hansson's Hypothetical Retrospection procedure for ethical reasoning under uncertainty. Our approach is validated by a case study from Machine Ethics literature: the problem of whether to steal insulin for someone who needs it. 

---
# Mastering Multi-Drone Volleyball through Hierarchical Co-Self-Play Reinforcement Learning 

**Authors**: Ruize Zhang, Sirui Xiang, Zelai Xu, Feng Gao, Shilong Ji, Wenhao Tang, Wenbo Ding, Chao Yu, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04317)  

**Abstract**: In this paper, we tackle the problem of learning to play 3v3 multi-drone volleyball, a new embodied competitive task that requires both high-level strategic coordination and low-level agile control. The task is turn-based, multi-agent, and physically grounded, posing significant challenges due to its long-horizon dependencies, tight inter-agent coupling, and the underactuated dynamics of quadrotors. To address this, we propose Hierarchical Co-Self-Play (HCSP), a hierarchical reinforcement learning framework that separates centralized high-level strategic decision-making from decentralized low-level motion control. We design a three-stage population-based training pipeline to enable both strategy and skill to emerge from scratch without expert demonstrations: (I) training diverse low-level skills, (II) learning high-level strategy via self-play with fixed low-level controllers, and (III) joint fine-tuning through co-self-play. Experiments show that HCSP achieves superior performance, outperforming non-hierarchical self-play and rule-based hierarchical baselines with an average 82.9\% win rate and a 71.5\% win rate against the two-stage variant. Moreover, co-self-play leads to emergent team behaviors such as role switching and coordinated formations, demonstrating the effectiveness of our hierarchical design and training scheme. 

---
# KERAIA: An Adaptive and Explainable Framework for Dynamic Knowledge Representation and Reasoning 

**Authors**: Stephen Richard Varey, Alessandro Di Stefano, Anh Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.04313)  

**Abstract**: In this paper, we introduce KERAIA, a novel framework and software platform for symbolic knowledge engineering designed to address the persistent challenges of representing, reasoning with, and executing knowledge in dynamic, complex, and context-sensitive environments. The central research question that motivates this work is: How can unstructured, often tacit, human expertise be effectively transformed into computationally tractable algorithms that AI systems can efficiently utilise? KERAIA seeks to bridge this gap by building on foundational concepts such as Minsky's frame-based reasoning and K-lines, while introducing significant innovations. These include Clouds of Knowledge for dynamic aggregation, Dynamic Relations (DRels) for context-sensitive inheritance, explicit Lines of Thought (LoTs) for traceable reasoning, and Cloud Elaboration for adaptive knowledge transformation. This approach moves beyond the limitations of traditional, often static, knowledge representation paradigms. KERAIA is designed with Explainable AI (XAI) as a core principle, ensuring transparency and interpretability, particularly through the use of LoTs. The paper details the framework's architecture, the KSYNTH representation language, and the General Purpose Paradigm Builder (GPPB) to integrate diverse inference methods within a unified structure. We validate KERAIA's versatility, expressiveness, and practical applicability through detailed analysis of multiple case studies spanning naval warfare simulation, industrial diagnostics in water treatment plants, and strategic decision-making in the game of RISK. Furthermore, we provide a comparative analysis against established knowledge representation paradigms (including ontologies, rule-based systems, and knowledge graphs) and discuss the implementation aspects and computational considerations of the KERAIA platform. 

---
# Flow Models for Unbounded and Geometry-Aware Distributional Reinforcement Learning 

**Authors**: Simo Alami C., Rim Kaddah, Jesse Read, Marie-Paule Cani  

**Link**: [PDF](https://arxiv.org/pdf/2505.04310)  

**Abstract**: We introduce a new architecture for Distributional Reinforcement Learning (DistRL) that models return distributions using normalizing flows. This approach enables flexible, unbounded support for return distributions, in contrast to categorical approaches like C51 that rely on fixed or bounded representations. It also offers richer modeling capacity to capture multi-modality, skewness, and tail behavior than quantile based approaches. Our method is significantly more parameter-efficient than categorical approaches. Standard metrics used to train existing models like KL divergence or Wasserstein distance either are scale insensitive or have biased sample gradients, especially when return supports do not overlap. To address this, we propose a novel surrogate for the Cramèr distance, that is geometry-aware and computable directly from the return distribution's PDF, avoiding the costly CDF computation. We test our model on the ATARI-5 sub-benchmark and show that our approach outperforms PDF based models while remaining competitive with quantile based methods. 

---
# Polynomial-Time Relational Probabilistic Inference in Open Universes 

**Authors**: Luise Ge, Brendan Juba, Kris Nilsson  

**Link**: [PDF](https://arxiv.org/pdf/2505.04115)  

**Abstract**: Reasoning under uncertainty is a fundamental challenge in Artificial Intelligence. As with most of these challenges, there is a harsh dilemma between the expressive power of the language used, and the tractability of the computational problem posed by reasoning. Inspired by human reasoning, we introduce a method of first-order relational probabilistic inference that satisfies both criteria, and can handle hybrid (discrete and continuous) variables. Specifically, we extend sum-of-squares logic of expectation to relational settings, demonstrating that lifted reasoning in the bounded-degree fragment for knowledge bases of bounded quantifier rank can be performed in polynomial time, even with an a priori unknown and/or countably infinite set of objects. Crucially, our notion of tractability is framed in proof-theoretic terms, which extends beyond the syntactic properties of the language or queries. We are able to derive the tightest bounds provable by proofs of a given degree and size and establish completeness in our sum-of-squares refutations for fixed degrees. 

---
# Extending Decision Predicate Graphs for Comprehensive Explanation of Isolation Forest 

**Authors**: Matteo Ceschin, Leonardo Arrighi, Luca Longo, Sylvio Barbon Junior  

**Link**: [PDF](https://arxiv.org/pdf/2505.04019)  

**Abstract**: The need to explain predictive models is well-established in modern machine learning. However, beyond model interpretability, understanding pre-processing methods is equally essential. Understanding how data modifications impact model performance improvements and potential biases and promoting a reliable pipeline is mandatory for developing robust machine learning solutions. Isolation Forest (iForest) is a widely used technique for outlier detection that performs well. Its effectiveness increases with the number of tree-based learners. However, this also complicates the explanation of outlier selection and the decision boundaries for inliers. This research introduces a novel Explainable AI (XAI) method, tackling the problem of global explainability. In detail, it aims to offer a global explanation for outlier detection to address its opaque nature. Our approach is based on the Decision Predicate Graph (DPG), which clarifies the logic of ensemble methods and provides both insights and a graph-based metric to explain how samples are identified as outliers using the proposed Inlier-Outlier Propagation Score (IOP-Score). Our proposal enhances iForest's explainability and provides a comprehensive view of the decision-making process, detailing which features contribute to outlier identification and how the model utilizes them. This method advances the state-of-the-art by providing insights into decision boundaries and a comprehensive view of holistic feature usage in outlier identification. -- thus promoting a fully explainable machine learning pipeline. 

---
# An alignment safety case sketch based on debate 

**Authors**: Marie Davidsen Buhl, Jacob Pfau, Benjamin Hilton, Geoffrey Irving  

**Link**: [PDF](https://arxiv.org/pdf/2505.03989)  

**Abstract**: If AI systems match or exceed human capabilities on a wide range of tasks, it may become difficult for humans to efficiently judge their actions -- making it hard to use human feedback to steer them towards desirable traits. One proposed solution is to leverage another superhuman system to point out flaws in the system's outputs via a debate. This paper outlines the value of debate for AI safety, as well as the assumptions and further research required to make debate work. It does so by sketching an ``alignment safety case'' -- an argument that an AI system will not autonomously take actions which could lead to egregious harm, despite being able to do so. The sketch focuses on the risk of an AI R\&D agent inside an AI company sabotaging research, for example by producing false results. To prevent this, the agent is trained via debate, subject to exploration guarantees, to teach the system to be honest. Honesty is maintained throughout deployment via online training. The safety case rests on four key claims: (1) the agent has become good at the debate game, (2) good performance in the debate game implies that the system is mostly honest, (3) the system will not become significantly less honest during deployment, and (4) the deployment context is tolerant of some errors. We identify open research problems that, if solved, could render this a compelling argument that an AI system is safe. 

---
# LogiDebrief: A Signal-Temporal Logic based Automated Debriefing Approach with Large Language Models Integration 

**Authors**: Zirong Chen, Ziyan An, Jennifer Reynolds, Kristin Mullen, Stephen Martini, Meiyi Ma  

**Link**: [PDF](https://arxiv.org/pdf/2505.03985)  

**Abstract**: Emergency response services are critical to public safety, with 9-1-1 call-takers playing a key role in ensuring timely and effective emergency operations. To ensure call-taking performance consistency, quality assurance is implemented to evaluate and refine call-takers' skillsets. However, traditional human-led evaluations struggle with high call volumes, leading to low coverage and delayed assessments. We introduce LogiDebrief, an AI-driven framework that automates traditional 9-1-1 call debriefing by integrating Signal-Temporal Logic (STL) with Large Language Models (LLMs) for fully-covered rigorous performance evaluation. LogiDebrief formalizes call-taking requirements as logical specifications, enabling systematic assessment of 9-1-1 calls against procedural guidelines. It employs a three-step verification process: (1) contextual understanding to identify responder types, incident classifications, and critical conditions; (2) STL-based runtime checking with LLM integration to ensure compliance; and (3) automated aggregation of results into quality assurance reports. Beyond its technical contributions, LogiDebrief has demonstrated real-world impact. Successfully deployed at Metro Nashville Department of Emergency Communications, it has assisted in debriefing 1,701 real-world calls, saving 311.85 hours of active engagement. Empirical evaluation with real-world data confirms its accuracy, while a case study and extensive user study highlight its effectiveness in enhancing call-taking performance. 

---
# The Power of Stories: Narrative Priming Shapes How LLM Agents Collaborate and Compete 

**Authors**: Gerrit Großmann, Larisa Ivanova, Sai Leela Poduru, Mohaddeseh Tabrizian, Islam Mesabah, David A. Selby, Sebastian J. Vollmer  

**Link**: [PDF](https://arxiv.org/pdf/2505.03961)  

**Abstract**: According to Yuval Noah Harari, large-scale human cooperation is driven by shared narratives that encode common beliefs and values. This study explores whether such narratives can similarly nudge LLM agents toward collaboration. We use a finitely repeated public goods game in which LLM agents choose either cooperative or egoistic spending strategies. We prime agents with stories highlighting teamwork to different degrees and test how this influences negotiation outcomes. Our experiments explore four questions:(1) How do narratives influence negotiation behavior? (2) What differs when agents share the same story versus different ones? (3) What happens when the agent numbers grow? (4) Are agents resilient against self-serving negotiators? We find that story-based priming significantly affects negotiation strategies and success rates. Common stories improve collaboration, benefiting each agent. By contrast, priming agents with different stories reverses this effect, and those agents primed toward self-interest prevail. We hypothesize that these results carry implications for multi-agent system design and AI alignment. 

---
# Frog Soup: Zero-Shot, In-Context, and Sample-Efficient Frogger Agents 

**Authors**: Xiang Li, Yiyang Hao, Doug Fulop  

**Link**: [PDF](https://arxiv.org/pdf/2505.03947)  

**Abstract**: One of the primary aspirations in reinforcement learning research is developing general-purpose agents capable of rapidly adapting to and mastering novel tasks. While RL gaming agents have mastered many Atari games, they remain slow and costly to train for each game. In this work, we demonstrate that latest reasoning LLMs with out-of-domain RL post-training can play a challenging Atari game called Frogger under a zero-shot setting. We then investigate the effect of in-context learning and the amount of reasoning effort on LLM performance. Lastly, we demonstrate a way to bootstrap traditional RL method with LLM demonstrations, which significantly improves their performance and sample efficiency. Our implementation is open sourced at this https URL. 

---
# GRAML: Dynamic Goal Recognition As Metric Learning 

**Authors**: Matan Shamir, Reuth Mirsky  

**Link**: [PDF](https://arxiv.org/pdf/2505.03941)  

**Abstract**: Goal Recognition (GR) is the problem of recognizing an agent's objectives based on observed actions. Recent data-driven approaches for GR alleviate the need for costly, manually crafted domain models. However, these approaches can only reason about a pre-defined set of goals, and time-consuming training is needed for new emerging goals. To keep this model-learning automated while enabling quick adaptation to new goals, this paper introduces GRAML: Goal Recognition As Metric Learning. GRAML uses a Siamese network to treat GR as a deep metric learning task, employing an RNN that learns a metric over an embedding space, where the embeddings for observation traces leading to different goals are distant, and embeddings of traces leading to the same goals are close. This metric is especially useful when adapting to new goals, even if given just one example observation trace per goal. Evaluated on a versatile set of environments, GRAML shows speed, flexibility, and runtime improvements over the state-of-the-art GR while maintaining accurate recognition. 

---
# Design description of Wisdom Computing Persperctive 

**Authors**: TianYi Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03800)  

**Abstract**: This course design aims to develop and research a handwriting matrix recognition and step-by-step visual calculation process display system, addressing the issue of abstract formulas and complex calculation steps that students find difficult to understand when learning mathematics. By integrating artificial intelligence with visualization animation technology, the system enhances precise recognition of handwritten matrix content through the introduction of Mamba backbone networks, completes digital extraction and matrix reconstruction using the YOLO model, and simultaneously combines CoordAttention coordinate attention mechanisms to improve the accurate grasp of character spatial positions. The calculation process is demonstrated frame by frame through the Manim animation engine, vividly showcasing each mathematical calculation step, helping students intuitively understand the intrinsic logic of mathematical operations. Through dynamically generating animation processes for different computational tasks, the system exhibits high modularity and flexibility, capable of generating various mathematical operation examples in real-time according to student needs. By innovating human-computer interaction methods, it brings mathematical calculation processes to life, helping students bridge the gap between knowledge and understanding on a deeper level, ultimately achieving a learning experience where "every step is understood." The system's scalability and interactivity make it an intuitive, user-friendly, and efficient auxiliary tool in education. 

---
# Proceedings of 1st Workshop on Advancing Artificial Intelligence through Theory of Mind 

**Authors**: Mouad Abrini, Omri Abend, Dina Acklin, Henny Admoni, Gregor Aichinger, Nitay Alon, Zahra Ashktorab, Ashish Atreja, Moises Auron, Alexander Aufreiter, Raghav Awasthi, Soumya Banerjee, Joe M. Barnby, Rhea Basappa, Severin Bergsmann, Djallel Bouneffouf, Patrick Callaghan, Marc Cavazza, Thierry Chaminade, Sonia Chernova, Mohamed Chetouan, Moumita Choudhury, Axel Cleeremans, Jacek B. Cywinski, Fabio Cuzzolin, Hokin Deng, N'yoma Diamond, Camilla Di Pasquasio, Guillaume Dumas, Max van Duijn, Mahapatra Dwarikanath, Qingying Gao, Ashok Goel, Rebecca Goldstein, Matthew Gombolay, Gabriel Enrique Gonzalez, Amar Halilovic, Tobias Halmdienst, Mahimul Islam, Julian Jara-Ettinger, Natalie Kastel, Renana Keydar, Ashish K. Khanna, Mahdi Khoramshahi, JiHyun Kim, MiHyeon Kim, YoungBin Kim, Senka Krivic, Nikita Krasnytskyi, Arun Kumar, JuneHyoung Kwon, Eunju Lee, Shane Lee, Peter R. Lewis, Xue Li, Yijiang Li, Michal Lewandowski, Nathan Lloyd, Matthew B. Luebbers, Dezhi Luo, Haiyun Lyu, Dwarikanath Mahapatra, Kamal Maheshwari, Mallika Mainali, Piyush Mathur, Patrick Mederitsch, Shuwa Miura, Manuel Preston de Miranda, Reuth Mirsky, Shreya Mishra, Nina Moorman, Katelyn Morrison, John Muchovej, Bernhard Nessler, Felix Nessler, Hieu Minh Jord Nguyen, Abby Ortego, Francis A. Papay, Antoine Pasquali, Hamed Rahimi, Charumathi Raghu, Amanda Royka, Stefan Sarkadi, Jaelle Scheuerman, Simon Schmid, Paul Schrater, Anik Sen, Zahra Sheikhbahaee, Ke Shi, Reid Simmons, Nishant Singh, Mason O. Smith, Ramira van der Meulen, Anthia Solaki, Haoran Sun, Viktor Szolga, Matthew E. Taylor, Travis Taylor, Sanne Van Waveren, Juan David Vargas  

**Link**: [PDF](https://arxiv.org/pdf/2505.03770)  

**Abstract**: This volume includes a selection of papers presented at the Workshop on Advancing Artificial Intelligence through Theory of Mind held at AAAI 2025 in Philadelphia US on 3rd March 2025. The purpose of this volume is to provide an open access and curated anthology for the ToM and AI research community. 

---
# EchoInk-R1: Exploring Audio-Visual Reasoning in Multimodal LLMs via Reinforcement Learning 

**Authors**: Zhenghao Xing, Xiaowei Hu, Chi-Wing Fu, Wenhai Wang, Jifeng Dai, Pheng-Ann Heng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04623)  

**Abstract**: Multimodal large language models (MLLMs) have advanced perception across text, vision, and audio, yet they often struggle with structured cross-modal reasoning, particularly when integrating audio and visual signals. We introduce EchoInk-R1, a reinforcement learning framework that enhances such reasoning in MLLMs. Built upon the Qwen2.5-Omni-7B foundation and optimized with Group Relative Policy Optimization (GRPO), EchoInk-R1 tackles multiple-choice question answering over synchronized audio-image pairs. To enable this, we curate AVQA-R1-6K, a dataset pairing such audio-image inputs with multiple-choice questions derived from OmniInstruct-v1. EchoInk-R1-7B achieves 85.77% accuracy on the validation set, outperforming the base model, which scores 80.53%, using only 562 reinforcement learning steps. Beyond accuracy, EchoInk-R1 demonstrates reflective reasoning by revisiting initial interpretations and refining responses when facing ambiguous multimodal inputs. These results suggest that lightweight reinforcement learning fine-tuning enhances cross-modal reasoning in MLLMs. EchoInk-R1 is the first framework to unify audio, visual, and textual modalities for general open-world reasoning via reinforcement learning. Code and data are publicly released to facilitate further research. 

---
# Score Distillation Sampling for Audio: Source Separation, Synthesis, and Beyond 

**Authors**: Jessie Richter-Powell, Antonio Torralba, Jonathan Lorraine  

**Link**: [PDF](https://arxiv.org/pdf/2505.04621)  

**Abstract**: We introduce Audio-SDS, a generalization of Score Distillation Sampling (SDS) to text-conditioned audio diffusion models. While SDS was initially designed for text-to-3D generation using image diffusion, its core idea of distilling a powerful generative prior into a separate parametric representation extends to the audio domain. Leveraging a single pretrained model, Audio-SDS enables a broad range of tasks without requiring specialized datasets. In particular, we demonstrate how Audio-SDS can guide physically informed impact sound simulations, calibrate FM-synthesis parameters, and perform prompt-specified source separation. Our findings illustrate the versatility of distillation-based methods across modalities and establish a robust foundation for future work using generative priors in audio tasks. 

---
# WATCH: Weighted Adaptive Testing for Changepoint Hypotheses via Weighted-Conformal Martingales 

**Authors**: Drew Prinster, Xing Han, Anqi Liu, Suchi Saria  

**Link**: [PDF](https://arxiv.org/pdf/2505.04608)  

**Abstract**: Responsibly deploying artificial intelligence (AI) / machine learning (ML) systems in high-stakes settings arguably requires not only proof of system reliability, but moreover continual, post-deployment monitoring to quickly detect and address any unsafe behavior. Statistical methods for nonparametric change-point detection -- especially the tools of conformal test martingales (CTMs) and anytime-valid inference -- offer promising approaches to this monitoring task. However, existing methods are restricted to monitoring limited hypothesis classes or ``alarm criteria,'' such as data shifts that violate certain exchangeability assumptions, or do not allow for online adaptation in response to shifts. In this paper, we expand the scope of these monitoring methods by proposing a weighted generalization of conformal test martingales (WCTMs), which lay a theoretical foundation for online monitoring for any unexpected changepoints in the data distribution while controlling false-alarms. For practical applications, we propose specific WCTM algorithms that accommodate online adaptation to mild covariate shifts (in the marginal input distribution) while raising alarms in response to more severe shifts, such as concept shifts (in the conditional label distribution) or extreme (out-of-support) covariate shifts that cannot be easily adapted to. On real-world datasets, we demonstrate improved performance relative to state-of-the-art baselines. 

---
# AI Governance to Avoid Extinction: The Strategic Landscape and Actionable Research Questions 

**Authors**: Peter Barnett, Aaron Scher  

**Link**: [PDF](https://arxiv.org/pdf/2505.04592)  

**Abstract**: Humanity appears to be on course to soon develop AI systems that substantially outperform human experts in all cognitive domains and activities. We believe the default trajectory has a high likelihood of catastrophe, including human extinction. Risks come from failure to control powerful AI systems, misuse of AI by malicious rogue actors, war between great powers, and authoritarian lock-in. This research agenda has two aims: to describe the strategic landscape of AI development and to catalog important governance research questions. These questions, if answered, would provide important insight on how to successfully reduce catastrophic risks.
We describe four high-level scenarios for the geopolitical response to advanced AI development, cataloging the research questions most relevant to each. Our favored scenario involves building the technical, legal, and institutional infrastructure required to internationally restrict dangerous AI development and deployment (which we refer to as an Off Switch), which leads into an internationally coordinated Halt on frontier AI activities at some point in the future. The second scenario we describe is a US National Project for AI, in which the US Government races to develop advanced AI systems and establish unilateral control over global AI development. We also describe two additional scenarios: a Light-Touch world similar to that of today and a Threat of Sabotage situation where countries use sabotage and deterrence to slow AI development.
In our view, apart from the Off Switch and Halt scenario, all of these trajectories appear to carry an unacceptable risk of catastrophic harm. Urgent action is needed from the US National Security community and AI governance ecosystem to answer key research questions, build the capability to halt dangerous AI activities, and prepare for international AI agreements. 

---
# Fight Fire with Fire: Defending Against Malicious RL Fine-Tuning via Reward Neutralization 

**Authors**: Wenjun Cao  

**Link**: [PDF](https://arxiv.org/pdf/2505.04578)  

**Abstract**: Reinforcement learning (RL) fine-tuning transforms large language models while creating a vulnerability we experimentally verify: Our experiment shows that malicious RL fine-tuning dismantles safety guardrails with remarkable efficiency, requiring only 50 steps and minimal adversarial prompts, with harmful escalating from 0-2 to 7-9. This attack vector particularly threatens open-source models with parameter-level access. Existing defenses targeting supervised fine-tuning prove ineffective against RL's dynamic feedback mechanisms. We introduce Reward Neutralization, the first defense framework specifically designed against RL fine-tuning attacks, establishing concise rejection patterns that render malicious reward signals ineffective. Our approach trains models to produce minimal-information rejections that attackers cannot exploit, systematically neutralizing attempts to optimize toward harmful outputs. Experiments validate that our approach maintains low harmful scores (no greater than 2) after 200 attack steps, while standard models rapidly deteriorate. This work provides the first constructive proof that robust defense against increasingly accessible RL attacks is achievable, addressing a critical security gap for open-weight models. 

---
# Purity Law for Generalizable Neural TSP Solvers 

**Authors**: Wenzhao Liu, Haoran Li, Congying Han, Zicheng Zhang, Anqi Li, Tiande Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.04558)  

**Abstract**: Achieving generalization in neural approaches across different scales and distributions remains a significant challenge for the Traveling Salesman Problem~(TSP). A key obstacle is that neural networks often fail to learn robust principles for identifying universal patterns and deriving optimal solutions from diverse instances. In this paper, we first uncover Purity Law (PuLa), a fundamental structural principle for optimal TSP solutions, defining that edge prevalence grows exponentially with the sparsity of surrounding vertices. Statistically validated across diverse instances, PuLa reveals a consistent bias toward local sparsity in global optima. Building on this insight, we propose Purity Policy Optimization~(PUPO), a novel training paradigm that explicitly aligns characteristics of neural solutions with PuLa during the solution construction process to enhance generalization. Extensive experiments demonstrate that PUPO can be seamlessly integrated with popular neural solvers, significantly enhancing their generalization performance without incurring additional computational overhead during inference. 

---
# Risk-sensitive Reinforcement Learning Based on Convex Scoring Functions 

**Authors**: Shanyu Han, Yang Liu, Xiang Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04553)  

**Abstract**: We propose a reinforcement learning (RL) framework under a broad class of risk objectives, characterized by convex scoring functions. This class covers many common risk measures, such as variance, Expected Shortfall, entropic Value-at-Risk, and mean-risk utility. To resolve the time-inconsistency issue, we consider an augmented state space and an auxiliary variable and recast the problem as a two-state optimization problem. We propose a customized Actor-Critic algorithm and establish some theoretical approximation guarantees. A key theoretical contribution is that our results do not require the Markov decision process to be continuous. Additionally, we propose an auxiliary variable sampling method inspired by the alternating minimization algorithm, which is convergent under certain conditions. We validate our approach in simulation experiments with a financial application in statistical arbitrage trading, demonstrating the effectiveness of the algorithm. 

---
# Overcoming Data Scarcity in Generative Language Modelling for Low-Resource Languages: A Systematic Review 

**Authors**: Josh McGiff, Nikola S. Nikolov  

**Link**: [PDF](https://arxiv.org/pdf/2505.04531)  

**Abstract**: Generative language modelling has surged in popularity with the emergence of services such as ChatGPT and Google Gemini. While these models have demonstrated transformative potential in productivity and communication, they overwhelmingly cater to high-resource languages like English. This has amplified concerns over linguistic inequality in natural language processing (NLP). This paper presents the first systematic review focused specifically on strategies to address data scarcity in generative language modelling for low-resource languages (LRL). Drawing from 54 studies, we identify, categorise and evaluate technical approaches, including monolingual data augmentation, back-translation, multilingual training, and prompt engineering, across generative tasks. We also analyse trends in architecture choices, language family representation, and evaluation methods. Our findings highlight a strong reliance on transformer-based models, a concentration on a small subset of LRLs, and a lack of consistent evaluation across studies. We conclude with recommendations for extending these methods to a wider range of LRLs and outline open challenges in building equitable generative language systems. Ultimately, this review aims to support researchers and developers in building inclusive AI tools for underrepresented languages, a necessary step toward empowering LRL speakers and the preservation of linguistic diversity in a world increasingly shaped by large-scale language technologies. 

---
# DFVO: Learning Darkness-free Visible and Infrared Image Disentanglement and Fusion All at Once 

**Authors**: Qi Zhou, Yukai Shi, Xiaojun Yang, Xiaoyu Xian, Lunjia Liao, Ruimao Zhang, Liang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04526)  

**Abstract**: Visible and infrared image fusion is one of the most crucial tasks in the field of image fusion, aiming to generate fused images with clear structural information and high-quality texture features for high-level vision tasks. However, when faced with severe illumination degradation in visible images, the fusion results of existing image fusion methods often exhibit blurry and dim visual effects, posing major challenges for autonomous driving. To this end, a Darkness-Free network is proposed to handle Visible and infrared image disentanglement and fusion all at Once (DFVO), which employs a cascaded multi-task approach to replace the traditional two-stage cascaded training (enhancement and fusion), addressing the issue of information entropy loss caused by hierarchical data transmission. Specifically, we construct a latent-common feature extractor (LCFE) to obtain latent features for the cascaded tasks strategy. Firstly, a details-extraction module (DEM) is devised to acquire high-frequency semantic information. Secondly, we design a hyper cross-attention module (HCAM) to extract low-frequency information and preserve texture features from source images. Finally, a relevant loss function is designed to guide the holistic network learning, thereby achieving better image fusion. Extensive experiments demonstrate that our proposed approach outperforms state-of-the-art alternatives in terms of qualitative and quantitative evaluations. Particularly, DFVO can generate clearer, more informative, and more evenly illuminated fusion results in the dark environments, achieving best performance on the LLVIP dataset with 63.258 dB PSNR and 0.724 CC, providing more effective information for high-level vision tasks. Our code is publicly accessible at this https URL. 

---
# Defining and Quantifying Creative Behavior in Popular Image Generators 

**Authors**: Aditi Ramaswamy  

**Link**: [PDF](https://arxiv.org/pdf/2505.04497)  

**Abstract**: Creativity of generative AI models has been a subject of scientific debate in the last years, without a conclusive answer. In this paper, we study creativity from a practical perspective and introduce quantitative measures that help the user to choose a suitable AI model for a given task. We evaluated our measures on a number of popular image-to-image generation models, and the results of this suggest that our measures conform to human intuition. 

---
# Model-Based AI planning and Execution Systems for Robotics 

**Authors**: Or Wertheim, Ronen I. Brafman  

**Link**: [PDF](https://arxiv.org/pdf/2505.04493)  

**Abstract**: Model-based planning and execution systems offer a principled approach to building flexible autonomous robots that can perform diverse tasks by automatically combining a host of basic skills. This idea is almost as old as modern robotics. Yet, while diverse general-purpose reasoning architectures have been proposed since, general-purpose systems that are integrated with modern robotic platforms have emerged only recently, starting with the influential ROSPlan system. Since then, a growing number of model-based systems for robot task-level control have emerged. In this paper, we consider the diverse design choices and issues existing systems attempt to address, the different solutions proposed so far, and suggest avenues for future development. 

---
# "I Can See Forever!": Evaluating Real-time VideoLLMs for Assisting Individuals with Visual Impairments 

**Authors**: Ziyi Zhang, Zhen Sun, Zongmin Zhang, Zifan Peng, Yuemeng Zhao, Zichun Wang, Zeren Luo, Ruiting Zuo, Xinlei He  

**Link**: [PDF](https://arxiv.org/pdf/2505.04488)  

**Abstract**: The visually impaired population, especially the severely visually impaired, is currently large in scale, and daily activities pose significant challenges for them. Although many studies use large language and vision-language models to assist the blind, most focus on static content and fail to meet real-time perception needs in dynamic and complex environments, such as daily activities. To provide them with more effective intelligent assistance, it is imperative to incorporate advanced visual understanding technologies. Although real-time vision and speech interaction VideoLLMs demonstrate strong real-time visual understanding, no prior work has systematically evaluated their effectiveness in assisting visually impaired individuals. In this work, we conduct the first such evaluation. First, we construct a benchmark dataset (VisAssistDaily), covering three categories of assistive tasks for visually impaired individuals: Basic Skills, Home Life Tasks, and Social Life Tasks. The results show that GPT-4o achieves the highest task success rate. Next, we conduct a user study to evaluate the models in both closed-world and open-world scenarios, further exploring the practical challenges of applying VideoLLMs in assistive contexts. One key issue we identify is the difficulty current models face in perceiving potential hazards in dynamic environments. To address this, we build an environment-awareness dataset named SafeVid and introduce a polling mechanism that enables the model to proactively detect environmental risks. We hope this work provides valuable insights and inspiration for future research in this field. 

---
# Efficient Flow Matching using Latent Variables 

**Authors**: Anirban Samaddar, Yixuan Sun, Viktor Nilsson, Sandeep Madireddy  

**Link**: [PDF](https://arxiv.org/pdf/2505.04486)  

**Abstract**: Flow matching models have shown great potential in image generation tasks among probabilistic generative models. Building upon the ideas of continuous normalizing flows, flow matching models generalize the transport path of the diffusion models from a simple prior distribution to the data. Most flow matching models in the literature do not explicitly model the underlying structure/manifold in the target data when learning the flow from a simple source distribution like the standard Gaussian. This leads to inefficient learning, especially for many high-dimensional real-world datasets, which often reside in a low-dimensional manifold. Existing strategies of incorporating manifolds, including data with underlying multi-modal distribution, often require expensive training and hence frequently lead to suboptimal performance. To this end, we present \texttt{Latent-CFM}, which provides simplified training/inference strategies to incorporate multi-modal data structures using pretrained deep latent variable models. Through experiments on multi-modal synthetic data and widely used image benchmark datasets, we show that \texttt{Latent-CFM} exhibits improved generation quality with significantly less training ($\sim 50\%$ less in some cases) and computation than state-of-the-art flow matching models. Using a 2d Darcy flow dataset, we demonstrate that our approach generates more physically accurate samples than competitive approaches. In addition, through latent space analysis, we demonstrate that our approach can be used for conditional image generation conditioned on latent features. 

---
# Spectral and Temporal Denoising for Differentially Private Optimization 

**Authors**: Hyeju Shin, Kyudan Jung, Seongwon Yun, Juyoung Yun  

**Link**: [PDF](https://arxiv.org/pdf/2505.04468)  

**Abstract**: This paper introduces the FFT-Enhanced Kalman Filter (FFTKF), a differentially private optimization method that addresses the challenge of preserving performance in DP-SGD, where added noise typically degrades model utility. FFTKF integrates frequency-domain noise shaping with Kalman filtering to enhance gradient quality while preserving $(\varepsilon, \delta)$-DP guarantees. It employs a high-frequency shaping mask in the Fourier domain to concentrate differential privacy noise in less informative spectral components, preserving low-frequency gradient signals. A scalar-gain Kalman filter with finite-difference Hessian approximation further refines the denoised gradients. With a per-iteration complexity of $\mathcal{O}(d \log d)$, FFTKF demonstrates improved test accuracy over DP-SGD and DiSK across MNIST, CIFAR-10, CIFAR-100, and Tiny-ImageNet datasets using CNNs, Wide ResNets, and Vision Transformers. Theoretical analysis confirms that FFTKF maintains equivalent privacy guarantees while achieving a tighter privacy-utility trade-off through reduced noise and controlled bias. 

---
# Discriminative Ordering Through Ensemble Consensus 

**Authors**: Louis Ohl, Fredrik Lindsten  

**Link**: [PDF](https://arxiv.org/pdf/2505.04464)  

**Abstract**: Evaluating the performance of clustering models is a challenging task where the outcome depends on the definition of what constitutes a cluster. Due to this design, current existing metrics rarely handle multiple clustering models with diverse cluster definitions, nor do they comply with the integration of constraints when available. In this work, we take inspiration from consensus clustering and assume that a set of clustering models is able to uncover hidden structures in the data. We propose to construct a discriminative ordering through ensemble clustering based on the distance between the connectivity of a clustering model and the consensus matrix. We first validate the proposed method with synthetic scenarios, highlighting that the proposed score ranks the models that best match the consensus first. We then show that this simple ranking score significantly outperforms other scoring methods when comparing sets of different clustering algorithms that are not restricted to a fixed number of clusters and is compatible with clustering constraints. 

---
# A Survey on Temporal Interaction Graph Representation Learning: Progress, Challenges, and Opportunities 

**Authors**: Pengfei Jiao, Hongjiang Chen, Xuan Guo, Zhidong Zhao, Dongxiao He, Di Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04461)  

**Abstract**: Temporal interaction graphs (TIGs), defined by sequences of timestamped interaction events, have become ubiquitous in real-world applications due to their capability to model complex dynamic system behaviors. As a result, temporal interaction graph representation learning (TIGRL) has garnered significant attention in recent years. TIGRL aims to embed nodes in TIGs into low-dimensional representations that effectively preserve both structural and temporal information, thereby enhancing the performance of downstream tasks such as classification, prediction, and clustering within constantly evolving data environments. In this paper, we begin by introducing the foundational concepts of TIGs and emphasize the critical role of temporal dependencies. We then propose a comprehensive taxonomy of state-of-the-art TIGRL methods, systematically categorizing them based on the types of information utilized during the learning process to address the unique challenges inherent to TIGs. To facilitate further research and practical applications, we curate the source of datasets and benchmarks, providing valuable resources for empirical investigations. Finally, we examine key open challenges and explore promising research directions in TIGRL, laying the groundwork for future advancements that have the potential to shape the evolution of this field. 

---
# Automatic Music Transcription using Convolutional Neural Networks and Constant-Q transform 

**Authors**: Yohannis Telila, Tommaso Cucinotta, Davide Bacciu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04451)  

**Abstract**: Automatic music transcription (AMT) is the problem of analyzing an audio recording of a musical piece and detecting notes that are being played. AMT is a challenging problem, particularly when it comes to polyphonic music. The goal of AMT is to produce a score representation of a music piece, by analyzing a sound signal containing multiple notes played simultaneously. In this work, we design a processing pipeline that can transform classical piano audio files in .wav format into a music score representation. The features from the audio signals are extracted using the constant-Q transform, and the resulting coefficients are used as an input to the convolutional neural network (CNN) model. 

---
# FedBWO: Enhancing Communication Efficiency in Federated Learning 

**Authors**: Vahideh Hayyolalam, Öznur Özkasap  

**Link**: [PDF](https://arxiv.org/pdf/2505.04435)  

**Abstract**: Federated Learning (FL) is a distributed Machine Learning (ML) setup, where a shared model is collaboratively trained by various clients using their local datasets while keeping the data private. Considering resource-constrained devices, FL clients often suffer from restricted transmission capacity. Aiming to enhance the system performance, the communication between clients and server needs to be diminished. Current FL strategies transmit a tremendous amount of data (model weights) within the FL process, which needs a high communication bandwidth. Considering resource constraints, increasing the number of clients and, consequently, the amount of data (model weights) can lead to a bottleneck. In this paper, we introduce the Federated Black Widow Optimization (FedBWO) technique to decrease the amount of transmitted data by transmitting only a performance score rather than the local model weights from clients. FedBWO employs the BWO algorithm to improve local model updates. The conducted experiments prove that FedBWO remarkably improves the performance of the global model and the communication efficiency of the overall system. According to the experimental outcomes, FedBWO enhances the global model accuracy by an average of 21% over FedAvg, and 12% over FedGWO. Furthermore, FedBWO dramatically decreases the communication cost compared to other methods. 

---
# Recognizing Ornaments in Vocal Indian Art Music with Active Annotation 

**Authors**: Sumit Kumar, Parampreet Singh, Vipul Arora  

**Link**: [PDF](https://arxiv.org/pdf/2505.04419)  

**Abstract**: Ornamentations, embellishments, or microtonal inflections are essential to melodic expression across many musical traditions, adding depth, nuance, and emotional impact to performances. Recognizing ornamentations in singing voices is key to MIR, with potential applications in music pedagogy, singer identification, genre classification, and controlled singing voice generation. However, the lack of annotated datasets and specialized modeling approaches remains a major obstacle for progress in this research area. In this work, we introduce Rāga Ornamentation Detection (ROD), a novel dataset comprising Indian classical music recordings curated by expert musicians. The dataset is annotated using a custom Human-in-the-Loop tool for six vocal ornaments marked as event-based labels. Using this dataset, we develop an ornamentation detection model based on deep time-series analysis, preserving ornament boundaries during the chunking of long audio recordings. We conduct experiments using different train-test configurations within the ROD dataset and also evaluate our approach on a separate, manually annotated dataset of Indian classical concert recordings. Our experimental results support the superior performance of our proposed approach over the baseline CRNN. 

---
# OBLIVIATE: Robust and Practical Machine Unlearning for Large Language Models 

**Authors**: Xiaoyu Xu, Minxin Du, Qingqing Ye, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04416)  

**Abstract**: Large language models (LLMs) trained over extensive corpora risk memorizing sensitive, copyrighted, or toxic content. To address this, we propose OBLIVIATE, a robust unlearning framework that removes targeted data while preserving model utility. The framework follows a structured process: extracting target tokens, building retain sets, and fine-tuning with a tailored loss function comprising three components -- masking, distillation, and world fact. Using low-rank adapters (LoRA), it ensures efficiency without compromising unlearning quality. We conduct experiments on multiple datasets, including the Harry Potter series, WMDP, and TOFU, using a comprehensive suite of metrics: forget quality (new document-level memorization score), model utility, and fluency. Results demonstrate its effectiveness in resisting membership inference attacks, minimizing the impact on retained data, and maintaining robustness across diverse scenarios. 

---
# YABLoCo: Yet Another Benchmark for Long Context Code Generation 

**Authors**: Aidar Valeev, Roman Garaev, Vadim Lomshakov, Irina Piontkovskaya, Vladimir Ivanov, Israel Adewuyi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04406)  

**Abstract**: Large Language Models demonstrate the ability to solve various programming tasks, including code generation. Typically, the performance of LLMs is measured on benchmarks with small or medium-sized context windows of thousands of lines of code. At the same time, in real-world software projects, repositories can span up to millions of LoC. This paper closes this gap by contributing to the long context code generation benchmark (YABLoCo). The benchmark featured a test set of 215 functions selected from four large repositories with thousands of functions. The dataset contained metadata of functions, contexts of the functions with different levels of dependencies, docstrings, functions bodies, and call graphs for each repository. This paper presents three key aspects of the contribution. First, the benchmark aims at function body generation in large repositories in C and C++, two languages not covered by previous benchmarks. Second, the benchmark contains large repositories from 200K to 2,000K LoC. Third, we contribute a scalable evaluation pipeline for efficient computing of the target metrics and a tool for visual analysis of generated code. Overall, these three aspects allow for evaluating code generation in large repositories in C and C++. 

---
# High-speed multiwavelength photonic temporal integration using silicon photonics 

**Authors**: Yi Zhang, Nikolaos Farmakidis, Ioannis Roumpos, Miltiadis Moralis-Pegios, Apostolos Tsakyridis, June Sang Lee, Bowei Dong, Yuhan He, Samarth Aggarwal, Nikolaos Pleros, Harish Bhaskaran  

**Link**: [PDF](https://arxiv.org/pdf/2505.04405)  

**Abstract**: Optical systems have been pivotal for energy-efficient computing, performing high-speed, parallel operations in low-loss carriers. While these predominantly analog optical accelerators bypass digitization to perform parallel floating-point computations, scaling optical hardware to map large-vector sizes for AI tasks remains challenging. Here, we overcome this limitation by unfolding scalar operations in time and introducing a photonic-heater-in-lightpath (PHIL) unit for all-optical temporal integration. Counterintuitively, we exploit a slow heat dissipation process to integrate optical signals modulated at 50 GHz bridging the speed gap between the widely applied thermo-optic effects and ultrafast photonics. This architecture supports optical end-to-end signal processing, eliminates inefficient electro-optical conversions, and enables both linear and nonlinear operations within a unified framework. Our results demonstrate a scalable path towards high-speed photonic computing through thermally driven integration. 

---
# In-Context Adaptation to Concept Drift for Learned Database Operations 

**Authors**: Jiaqi Zhu, Shaofeng Cai, Yanyan Shen, Gang Chen, Fang Deng, Beng Chin Ooi  

**Link**: [PDF](https://arxiv.org/pdf/2505.04404)  

**Abstract**: Machine learning has demonstrated transformative potential for database operations, such as query optimization and in-database data analytics. However, dynamic database environments, characterized by frequent updates and evolving data distributions, introduce concept drift, which leads to performance degradation for learned models and limits their practical applicability. Addressing this challenge requires efficient frameworks capable of adapting to shifting concepts while minimizing the overhead of retraining or fine-tuning.
In this paper, we propose FLAIR, an online adaptation framework that introduces a new paradigm called \textit{in-context adaptation} for learned database operations. FLAIR leverages the inherent property of data systems, i.e., immediate availability of execution results for predictions, to enable dynamic context construction. By formalizing adaptation as $f:(\mathbf{x} \,| \,\mathcal{C}_t) \to \mathbf{y}$, with $\mathcal{C}_t$ representing a dynamic context memory, FLAIR delivers predictions aligned with the current concept, eliminating the need for runtime parameter optimization. To achieve this, FLAIR integrates two key modules: a Task Featurization Module for encoding task-specific features into standardized representations, and a Dynamic Decision Engine, pre-trained via Bayesian meta-training, to adapt seamlessly using contextual information at runtime. Extensive experiments across key database tasks demonstrate that FLAIR outperforms state-of-the-art baselines, achieving up to 5.2x faster adaptation and reducing error by 22.5% for cardinality estimation. 

---
# Deep residual learning with product units 

**Authors**: Ziyuan Li, Uwe Jaekel, Babette Dellen  

**Link**: [PDF](https://arxiv.org/pdf/2505.04397)  

**Abstract**: We propose a deep product-unit residual neural network (PURe) that integrates product units into residual blocks to improve the expressiveness and parameter efficiency of deep convolutional networks. Unlike standard summation neurons, product units enable multiplicative feature interactions, potentially offering a more powerful representation of complex patterns. PURe replaces conventional convolutional layers with 2D product units in the second layer of each residual block, eliminating nonlinear activation functions to preserve structural information. We validate PURe on three benchmark datasets. On Galaxy10 DECaLS, PURe34 achieves the highest test accuracy of 84.89%, surpassing the much deeper ResNet152, while converging nearly five times faster and demonstrating strong robustness to Poisson noise. On ImageNet, PURe architectures outperform standard ResNet models at similar depths, with PURe34 achieving a top-1 accuracy of 80.27% and top-5 accuracy of 95.78%, surpassing deeper ResNet variants (ResNet50, ResNet101) while utilizing significantly fewer parameters and computational resources. On CIFAR-10, PURe consistently outperforms ResNet variants across varying depths, with PURe272 reaching 95.01% test accuracy, comparable to ResNet1001 but at less than half the model size. These results demonstrate that PURe achieves a favorable balance between accuracy, efficiency, and robustness. Compared to traditional residual networks, PURe not only achieves competitive classification performance with faster convergence and fewer parameters, but also demonstrates greater robustness to noise. Its effectiveness across diverse datasets highlights the potential of product-unit-based architectures for scalable and reliable deep learning in computer vision. 

---
# The Aloe Family Recipe for Open and Specialized Healthcare LLMs 

**Authors**: Dario Garcia-Gasulla, Jordi Bayarri-Planas, Ashwin Kumar Gururajan, Enrique Lopez-Cuena, Adrian Tormos, Daniel Hinjos, Pablo Bernabeu-Perez, Anna Arias-Duart, Pablo Agustin Martin-Torres, Marta Gonzalez-Mallo, Sergio Alvarez-Napagao, Eduard Ayguadé-Parra, Ulises Cortés  

**Link**: [PDF](https://arxiv.org/pdf/2505.04388)  

**Abstract**: Purpose: With advancements in Large Language Models (LLMs) for healthcare, the need arises for competitive open-source models to protect the public interest. This work contributes to the field of open medical LLMs by optimizing key stages of data preprocessing and training, while showing how to improve model safety (through DPO) and efficacy (through RAG). The evaluation methodology used, which includes four different types of tests, defines a new standard for the field. The resultant models, shown to be competitive with the best private alternatives, are released with a permisive license.
Methods: Building on top of strong base models like Llama 3.1 and Qwen 2.5, Aloe Beta uses a custom dataset to enhance public data with synthetic Chain of Thought examples. The models undergo alignment with Direct Preference Optimization, emphasizing ethical and policy-aligned performance in the presence of jailbreaking attacks. Evaluation includes close-ended, open-ended, safety and human assessments, to maximize the reliability of results.
Results: Recommendations are made across the entire pipeline, backed by the solid performance of the Aloe Family. These models deliver competitive performance across healthcare benchmarks and medical fields, and are often preferred by healthcare professionals. On bias and toxicity, the Aloe Beta models significantly improve safety, showing resilience to unseen jailbreaking attacks. For a responsible release, a detailed risk assessment specific to healthcare is attached to the Aloe Family models.
Conclusion: The Aloe Beta models, and the recipe that leads to them, are a significant contribution to the open-source medical LLM field, offering top-of-the-line performance while maintaining high ethical requirements. This work sets a new standard for developing and reporting aligned LLMs in healthcare. 

---
# Consensus-Aware AV Behavior: Trade-offs Between Safety, Interaction, and Performance in Mixed Urban Traffic 

**Authors**: Mohammad Elayan, Wissam Kontar  

**Link**: [PDF](https://arxiv.org/pdf/2505.04379)  

**Abstract**: Transportation systems have long been shaped by complexity and heterogeneity, driven by the interdependency of agent actions and traffic outcomes. The deployment of automated vehicles (AVs) in such systems introduces a new challenge: achieving consensus across safety, interaction quality, and traffic performance. In this work, we position consensus as a fundamental property of the traffic system and aim to quantify it. We use high-resolution trajectory data from the Third Generation Simulation (TGSIM) dataset to empirically analyze AV and human-driven vehicle (HDV) behavior at a signalized urban intersection and around vulnerable road users (VRUs). Key metrics, including Time-to-Collision (TTC), Post-Encroachment Time (PET), deceleration patterns, headways, and string stability, are evaluated across the three performance dimensions. Results show that full consensus across safety, interaction, and performance is rare, with only 1.63% of AV-VRU interaction frames meeting all three conditions. These findings highlight the need for AV models that explicitly balance multi-dimensional performance in mixed-traffic environments. Full reproducibility is supported via our open-source codebase on this https URL. 

---
# Balancing Accuracy, Calibration, and Efficiency in Active Learning with Vision Transformers Under Label Noise 

**Authors**: Moseli Mots'oehli, Hope Mogale, Kyungim Baek  

**Link**: [PDF](https://arxiv.org/pdf/2505.04375)  

**Abstract**: Fine-tuning pre-trained convolutional neural networks on ImageNet for downstream tasks is well-established. Still, the impact of model size on the performance of vision transformers in similar scenarios, particularly under label noise, remains largely unexplored. Given the utility and versatility of transformer architectures, this study investigates their practicality under low-budget constraints and noisy labels. We explore how classification accuracy and calibration are affected by symmetric label noise in active learning settings, evaluating four vision transformer configurations (Base and Large with 16x16 and 32x32 patch sizes) and three Swin Transformer configurations (Tiny, Small, and Base) on CIFAR10 and CIFAR100 datasets, under varying label noise rates. Our findings show that larger ViT models (ViTl32 in particular) consistently outperform their smaller counterparts in both accuracy and calibration, even under moderate to high label noise, while Swin Transformers exhibit weaker robustness across all noise levels. We find that smaller patch sizes do not always lead to better performance, as ViTl16 performs consistently worse than ViTl32 while incurring a higher computational cost. We also find that information-based Active Learning strategies only provide meaningful accuracy improvements at moderate label noise rates, but they result in poorer calibration compared to models trained on randomly acquired labels, especially at high label noise rates. We hope these insights provide actionable guidance for practitioners looking to deploy vision transformers in resource-constrained environments, where balancing model complexity, label noise, and compute efficiency is critical in model fine-tuning or distillation. 

---
# Optimization Problem Solving Can Transition to Evolutionary Agentic Workflows 

**Authors**: Wenhao Li, Bo Jin, Mingyi Hong, Changhong Lu, Xiangfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04354)  

**Abstract**: This position paper argues that optimization problem solving can transition from expert-dependent to evolutionary agentic workflows. Traditional optimization practices rely on human specialists for problem formulation, algorithm selection, and hyperparameter tuning, creating bottlenecks that impede industrial adoption of cutting-edge methods. We contend that an evolutionary agentic workflow, powered by foundation models and evolutionary search, can autonomously navigate the optimization space, comprising problem, formulation, algorithm, and hyperparameter spaces. Through case studies in cloud resource scheduling and ADMM parameter adaptation, we demonstrate how this approach can bridge the gap between academic innovation and industrial implementation. Our position challenges the status quo of human-centric optimization workflows and advocates for a more scalable, adaptive approach to solving real-world optimization problems. 

---
# Multi-Granular Attention based Heterogeneous Hypergraph Neural Network 

**Authors**: Hong Jin, Kaicheng Zhou, Jie Yin, Lan You, Zhifeng Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.04340)  

**Abstract**: Heterogeneous graph neural networks (HeteGNNs) have demonstrated strong abilities to learn node representations by effectively extracting complex structural and semantic information in heterogeneous graphs. Most of the prevailing HeteGNNs follow the neighborhood aggregation paradigm, leveraging meta-path based message passing to learn latent node representations. However, due to the pairwise nature of meta-paths, these models fail to capture high-order relations among nodes, resulting in suboptimal performance. Additionally, the challenge of ``over-squashing'', where long-range message passing in HeteGNNs leads to severe information distortion, further limits the efficacy of these models. To address these limitations, this paper proposes MGA-HHN, a Multi-Granular Attention based Heterogeneous Hypergraph Neural Network for heterogeneous graph representation learning. MGA-HHN introduces two key innovations: (1) a novel approach for constructing meta-path based heterogeneous hypergraphs that explicitly models higher-order semantic information in heterogeneous graphs through multiple views, and (2) a multi-granular attention mechanism that operates at both the node and hyperedge levels. This mechanism enables the model to capture fine-grained interactions among nodes sharing the same semantic context within a hyperedge type, while preserving the diversity of semantics across different hyperedge types. As such, MGA-HHN effectively mitigates long-range message distortion and generates more expressive node representations. Extensive experiments on real-world benchmark datasets demonstrate that MGA-HHN outperforms state-of-the-art models, showcasing its effectiveness in node classification, node clustering and visualization tasks. 

---
# Detecting Concept Drift in Neural Networks Using Chi-squared Goodness of Fit Testing 

**Authors**: Jacob Glenn Ayers, Buvaneswari A. Ramanan, Manzoor A. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04318)  

**Abstract**: As the adoption of deep learning models has grown beyond human capacity for verification, meta-algorithms are needed to ensure reliable model inference. Concept drift detection is a field dedicated to identifying statistical shifts that is underutilized in monitoring neural networks that may encounter inference data with distributional characteristics diverging from their training data. Given the wide variety of model architectures, applications, and datasets, it is important that concept drift detection algorithms are adaptable to different inference scenarios. In this paper, we introduce an application of the $\chi^2$ Goodness of Fit Hypothesis Test as a drift detection meta-algorithm applied to a multilayer perceptron, a convolutional neural network, and a transformer trained for machine vision as they are exposed to simulated drift during inference. To that end, we demonstrate how unexpected drops in accuracy due to concept drift can be detected without directly examining the inference outputs. Our approach enhances safety by ensuring models are continually evaluated for reliability across varying conditions. 

---
# Guardians of the Web: The Evolution and Future of Website Information Security 

**Authors**: Md Saiful Islam, Li Xiangdong  

**Link**: [PDF](https://arxiv.org/pdf/2505.04308)  

**Abstract**: Website information security has become a critical concern in the digital age. This article explores the evolution of website information security, examining its historical development, current practices, and future directions. The early beginnings from the 1960s to the 1980s laid the groundwork for modern cybersecurity, with the development of ARPANET, TCP/IP, public-key cryptography, and the first antivirus programs. The 1990s marked a transformative era, driven by the commercialization of the Internet and the emergence of web-based services. As the Internet grew, so did the range and sophistication of cyber threats, leading to advancements in security technologies such as the Secure Sockets Layer (SSL) protocol, password protection, and firewalls. Current practices in website information security involve a multi-layered approach, including encryption, secure coding practices, regular security audits, and user education. The future of website information security is expected to be shaped by emerging technologies such as artificial intelligence, blockchain, and quantum computing, as well as the increasing importance of international cooperation and standardization efforts. As cyber threats continue to evolve, ongoing research and innovation in website information security will be essential to protect sensitive information and maintain trust in the digital world. 

---
# Sparsity is All You Need: Rethinking Biological Pathway-Informed Approaches in Deep Learning 

**Authors**: Isabella Caranzano, Corrado Pancotti, Cesare Rollo, Flavio Sartori, Pietro Liò, Piero Fariselli, Tiziana Sanavia  

**Link**: [PDF](https://arxiv.org/pdf/2505.04300)  

**Abstract**: Biologically-informed neural networks typically leverage pathway annotations to enhance performance in biomedical applications. We hypothesized that the benefits of pathway integration does not arise from its biological relevance, but rather from the sparsity it introduces. We conducted a comprehensive analysis of all relevant pathway-based neural network models for predictive tasks, critically evaluating each study's contributions. From this review, we curated a subset of methods for which the source code was publicly available. The comparison of the biologically informed state-of-the-art deep learning models and their randomized counterparts showed that models based on randomized information performed equally well as biologically informed ones across different metrics and datasets. Notably, in 3 out of the 15 analyzed models, the randomized versions even outperformed their biologically informed counterparts. Moreover, pathway-informed models did not show any clear advantage in interpretability, as randomized models were still able to identify relevant disease biomarkers despite lacking explicit pathway information. Our findings suggest that pathway annotations may be too noisy or inadequately explored by current methods. Therefore, we propose a methodology that can be applied to different domains and can serve as a robust benchmark for systematically comparing novel pathway-informed models against their randomized counterparts. This approach enables researchers to rigorously determine whether observed performance improvements can be attributed to biological insights. 

---
# GASCADE: Grouped Summarization of Adverse Drug Event for Enhanced Cancer Pharmacovigilance 

**Authors**: Sofia Jamil, Aryan Dabad, Bollampalli Areen Reddy, Sriparna Saha, Rajiv Misra, Adil A. Shakur  

**Link**: [PDF](https://arxiv.org/pdf/2505.04284)  

**Abstract**: In the realm of cancer treatment, summarizing adverse drug events (ADEs) reported by patients using prescribed drugs is crucial for enhancing pharmacovigilance practices and improving drug-related decision-making. While the volume and complexity of pharmacovigilance data have increased, existing research in this field has predominantly focused on general diseases rather than specifically addressing cancer. This work introduces the task of grouped summarization of adverse drug events reported by multiple patients using the same drug for cancer treatment. To address the challenge of limited resources in cancer pharmacovigilance, we present the MultiLabeled Cancer Adverse Drug Reaction and Summarization (MCADRS) dataset. This dataset includes pharmacovigilance posts detailing patient concerns regarding drug efficacy and adverse effects, along with extracted labels for drug names, adverse drug events, severity, and adversity of reactions, as well as summaries of ADEs for each drug. Additionally, we propose the Grouping and Abstractive Summarization of Cancer Adverse Drug events (GASCADE) framework, a novel pipeline that combines the information extraction capabilities of Large Language Models (LLMs) with the summarization power of the encoder-decoder T5 model. Our work is the first to apply alignment techniques, including advanced algorithms like Direct Preference Optimization, to encoder-decoder models using synthetic datasets for summarization tasks. Through extensive experiments, we demonstrate the superior performance of GASCADE across various metrics, validated through both automated assessments and human evaluations. This multitasking approach enhances drug-related decision-making and fosters a deeper understanding of patient concerns, paving the way for advancements in personalized and responsive cancer care. The code and dataset used in this work are publicly available. 

---
# Non-stationary Diffusion For Probabilistic Time Series Forecasting 

**Authors**: Weiwei Ye, Zhuopeng Xu, Ning Gui  

**Link**: [PDF](https://arxiv.org/pdf/2505.04278)  

**Abstract**: Due to the dynamics of underlying physics and external influences, the uncertainty of time series often varies over time. However, existing Denoising Diffusion Probabilistic Models (DDPMs) often fail to capture this non-stationary nature, constrained by their constant variance assumption from the additive noise model (ANM). In this paper, we innovatively utilize the Location-Scale Noise Model (LSNM) to relax the fixed uncertainty assumption of ANM. A diffusion-based probabilistic forecasting framework, termed Non-stationary Diffusion (NsDiff), is designed based on LSNM that is capable of modeling the changing pattern of uncertainty. Specifically, NsDiff combines a denoising diffusion-based conditional generative model with a pre-trained conditional mean and variance estimator, enabling adaptive endpoint distribution modeling. Furthermore, we propose an uncertainty-aware noise schedule, which dynamically adjusts the noise levels to accurately reflect the data uncertainty at each step and integrates the time-varying variances into the diffusion process. Extensive experiments conducted on nine real-world and synthetic datasets demonstrate the superior performance of NsDiff compared to existing approaches. Code is available at this https URL. 

---
# Object-Shot Enhanced Grounding Network for Egocentric Video 

**Authors**: Yisen Feng, Haoyu Zhang, Meng Liu, Weili Guan, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.04270)  

**Abstract**: Egocentric video grounding is a crucial task for embodied intelligence applications, distinct from exocentric video moment localization. Existing methods primarily focus on the distributional differences between egocentric and exocentric videos but often neglect key characteristics of egocentric videos and the fine-grained information emphasized by question-type queries. To address these limitations, we propose OSGNet, an Object-Shot enhanced Grounding Network for egocentric video. Specifically, we extract object information from videos to enrich video representation, particularly for objects highlighted in the textual query but not directly captured in the video features. Additionally, we analyze the frequent shot movements inherent to egocentric videos, leveraging these features to extract the wearer's attention information, which enhances the model's ability to perform modality alignment. Experiments conducted on three datasets demonstrate that OSGNet achieves state-of-the-art performance, validating the effectiveness of our approach. Our code can be found at this https URL. 

---
# Weaponizing Language Models for Cybersecurity Offensive Operations: Automating Vulnerability Assessment Report Validation; A Review Paper 

**Authors**: Abdulrahman S Almuhaidib, Azlan Mohd Zain, Zalmiyah Zakaria, Izyan Izzati Kamsani, Abdulaziz S Almuhaidib  

**Link**: [PDF](https://arxiv.org/pdf/2505.04265)  

**Abstract**: This, with the ever-increasing sophistication of cyberwar, calls for novel solutions. In this regard, Large Language Models (LLMs) have emerged as a highly promising tool for defensive and offensive cybersecurity-related strategies. While existing literature has focused much on the defensive use of LLMs, when it comes to their offensive utilization, very little has been reported-namely, concerning Vulnerability Assessment (VA) report validation. Consequentially, this paper tries to fill that gap by investigating the capabilities of LLMs in automating and improving the validation process of the report of the VA. From the critical review of the related literature, this paper hereby proposes a new approach to using the LLMs in the automation of the analysis and within the validation process of the report of the VA that could potentially reduce the number of false positives and generally enhance efficiency. These results are promising for LLM automatization for improving validation on reports coming from VA in order to improve accuracy while reducing human effort and security postures. The contribution of this paper provides further evidence about the offensive and defensive LLM capabilities and therefor helps in devising more appropriate cybersecurity strategies and tools accordingly. 

---
# Steerable Chatbots: Personalizing LLMs with Preference-Based Activation Steering 

**Authors**: Jessica Y. Bo, Tianyu Xu, Ishan Chatterjee, Katrina Passarella-Ward, Achin Kulshrestha, D Shin  

**Link**: [PDF](https://arxiv.org/pdf/2505.04260)  

**Abstract**: As large language models (LLMs) improve in their capacity to serve as personal AI assistants, their ability to output uniquely tailored, personalized responses that align with the soft preferences of their users is essential for enhancing user satisfaction and retention. However, untrained lay users have poor prompt specification abilities and often struggle with conveying their latent preferences to AI assistants. To address this, we leverage activation steering to guide LLMs to align with interpretable preference dimensions during inference. In contrast to memory-based personalization methods that require longer user history, steering is extremely lightweight and can be easily controlled by the user via an linear strength factor. We embed steering into three different interactive chatbot interfaces and conduct a within-subjects user study (n=14) to investigate how end users prefer to personalize their conversations. The results demonstrate the effectiveness of preference-based steering for aligning real-world conversations with hidden user preferences, and highlight further insights on how diverse values around control, usability, and transparency lead users to prefer different interfaces. 

---
# Facilitating Trustworthy Human-Agent Collaboration in LLM-based Multi-Agent System oriented Software Engineering 

**Authors**: Krishna Ronanki  

**Link**: [PDF](https://arxiv.org/pdf/2505.04251)  

**Abstract**: Multi-agent autonomous systems (MAS) are better at addressing challenges that spans across multiple domains than singular autonomous agents. This holds true within the field of software engineering (SE) as well. The state-of-the-art research on MAS within SE focuses on integrating LLMs at the core of autonomous agents to create LLM-based multi-agent autonomous (LMA) systems. However, the introduction of LMA systems into SE brings a plethora of challenges. One of the major challenges is the strategic allocation of tasks between humans and the LMA system in a trustworthy manner. To address this challenge, a RACI-based framework is proposed in this work in progress article, along with implementation guidelines and an example implementation of the framework. The proposed framework can facilitate efficient collaboration, ensure accountability, and mitigate potential risks associated with LLM-driven automation while aligning with the Trustworthy AI guidelines. The future steps for this work delineating the planned empirical validation method are also presented. 

---
# FRAIN to Train: A Fast-and-Reliable Solution for Decentralized Federated Learning 

**Authors**: Sanghyeon Park, Soo-Mook Moon  

**Link**: [PDF](https://arxiv.org/pdf/2505.04223)  

**Abstract**: Federated learning (FL) enables collaborative model training across distributed clients while preserving data locality. Although FedAvg pioneered synchronous rounds for global model averaging, slower devices can delay collective progress. Asynchronous FL (e.g., FedAsync) addresses stragglers by continuously integrating client updates, yet naive implementations risk client drift due to non-IID data and stale contributions. Some Blockchain-based FL approaches (e.g., BRAIN) employ robust weighting or scoring of updates to resist malicious or misaligned proposals. However, performance drops can still persist under severe data heterogeneity or high staleness, and synchronization overhead has emerged as a new concern due to its aggregator-free architectures.
We introduce Fast-and-Reliable AI Network, FRAIN, a new asynchronous FL method that mitigates these limitations by incorporating two key ideas. First, our FastSync strategy eliminates the need to replay past model versions, enabling newcomers and infrequent participants to efficiently approximate the global model. Second, we adopt spherical linear interpolation (SLERP) when merging parameters, preserving models' directions and alleviating destructive interference from divergent local training.
Experiments with a CNN image-classification model and a Transformer-based language model demonstrate that FRAIN achieves more stable and robust convergence than FedAvg, FedAsync, and BRAIN, especially under harsh environments: non-IID data distributions, networks that experience delays and require frequent re-synchronization, and the presence of malicious nodes. 

---
# To Judge or not to Judge: Using LLM Judgements for Advertiser Keyphrase Relevance at eBay 

**Authors**: Soumik Dey, Hansi Wu, Binbin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.04209)  

**Abstract**: E-commerce sellers are recommended keyphrases based on their inventory on which they advertise to increase buyer engagement (clicks/sales). The relevance of advertiser keyphrases plays an important role in preventing the inundation of search systems with numerous irrelevant items that compete for attention in auctions, in addition to maintaining a healthy seller perception. In this work, we describe the shortcomings of training Advertiser keyphrase relevance filter models on click/sales/search relevance signals and the importance of aligning with human judgment, as sellers have the power to adopt or reject said keyphrase recommendations. In this study, we frame Advertiser keyphrase relevance as a complex interaction between 3 dynamical systems -- seller judgment, which influences seller adoption of our product, Advertising, which provides the keyphrases to bid on, and Search, who holds the auctions for the same keyphrases. This study discusses the practicalities of using human judgment via a case study at eBay Advertising and demonstrate that using LLM-as-a-judge en-masse as a scalable proxy for seller judgment to train our relevance models achieves a better harmony across the three systems -- provided that they are bound by a meticulous evaluation framework grounded in business metrics. 

---
# An Enhanced YOLOv8 Model for Real-Time and Accurate Pothole Detection and Measurement 

**Authors**: Mustafa Yurdakul, Şakir Tasdemir  

**Link**: [PDF](https://arxiv.org/pdf/2505.04207)  

**Abstract**: Potholes cause vehicle damage and traffic accidents, creating serious safety and economic problems. Therefore, early and accurate detection of potholes is crucial. Existing detection methods are usually only based on 2D RGB images and cannot accurately analyze the physical characteristics of potholes. In this paper, a publicly available dataset of RGB-D images (PothRGBD) is created and an improved YOLOv8-based model is proposed for both pothole detection and pothole physical features analysis. The Intel RealSense D415 depth camera was used to collect RGB and depth data from the road surfaces, resulting in a PothRGBD dataset of 1000 images. The data was labeled in YOLO format suitable for segmentation. A novel YOLO model is proposed based on the YOLOv8n-seg architecture, which is structurally improved with Dynamic Snake Convolution (DSConv), Simple Attention Module (SimAM) and Gaussian Error Linear Unit (GELU). The proposed model segmented potholes with irregular edge structure more accurately, and performed perimeter and depth measurements on depth maps with high accuracy. The standard YOLOv8n-seg model achieved 91.9% precision, 85.2% recall and 91.9% mAP@50. With the proposed model, the values increased to 93.7%, 90.4% and 93.8% respectively. Thus, an improvement of 1.96% in precision, 6.13% in recall and 2.07% in mAP was achieved. The proposed model performs pothole detection as well as perimeter and depth measurement with high accuracy and is suitable for real-time applications due to its low model complexity. In this way, a lightweight and effective model that can be used in deep learning-based intelligent transportation solutions has been acquired. 

---
# VideoPath-LLaVA: Pathology Diagnostic Reasoning Through Video Instruction Tuning 

**Authors**: Trinh T.L. Vuong, Jin Tae Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2505.04192)  

**Abstract**: We present VideoPath-LLaVA, the first large multimodal model (LMM) in computational pathology that integrates three distinct image scenarios, single patch images, automatically keyframe-extracted clips, and manually segmented video pathology images, to mimic the natural diagnostic process of pathologists. By generating detailed histological descriptions and culminating in a definitive sign-out diagnosis, VideoPath-LLaVA bridges visual narratives with diagnostic reasoning.
Central to our approach is the VideoPath-Instruct dataset, comprising 4278 video and diagnosis-specific chain-of-thought instructional pairs sourced from educational histopathology videos on YouTube. Although high-quality data is critical for enhancing diagnostic reasoning, its creation is time-intensive and limited in volume. To overcome this challenge, we transfer knowledge from existing single-image instruction datasets to train on weakly annotated, keyframe-extracted clips, followed by fine-tuning on manually segmented videos. VideoPath-LLaVA establishes a new benchmark in pathology video analysis and offers a promising foundation for future AI systems that support clinical decision-making through integrated visual and diagnostic reasoning. Our code, data, and model are publicly available at this https URL. 

---
# S3D: Sketch-Driven 3D Model Generation 

**Authors**: Hail Song, Wonsik Shin, Naeun Lee, Soomin Chung, Nojun Kwak, Woontack Woo  

**Link**: [PDF](https://arxiv.org/pdf/2505.04185)  

**Abstract**: Generating high-quality 3D models from 2D sketches is a challenging task due to the inherent ambiguity and sparsity of sketch data. In this paper, we present S3D, a novel framework that converts simple hand-drawn sketches into detailed 3D models. Our method utilizes a U-Net-based encoder-decoder architecture to convert sketches into face segmentation masks, which are then used to generate a 3D representation that can be rendered from novel views. To ensure robust consistency between the sketch domain and the 3D output, we introduce a novel style-alignment loss that aligns the U-Net bottleneck features with the initial encoder outputs of the 3D generation module, significantly enhancing reconstruction fidelity. To further enhance the network's robustness, we apply augmentation techniques to the sketch dataset. This streamlined framework demonstrates the effectiveness of S3D in generating high-quality 3D models from sketch inputs. The source code for this project is publicly available at this https URL. 

---
# DOTA: Deformable Optimized Transformer Architecture for End-to-End Text Recognition with Retrieval-Augmented Generation 

**Authors**: Naphat Nithisopa, Teerapong Panboonyuen  

**Link**: [PDF](https://arxiv.org/pdf/2505.04175)  

**Abstract**: Text recognition in natural images remains a challenging yet essential task, with broad applications spanning computer vision and natural language processing. This paper introduces a novel end-to-end framework that combines ResNet and Vision Transformer backbones with advanced methodologies, including Deformable Convolutions, Retrieval-Augmented Generation, and Conditional Random Fields (CRF). These innovations collectively enhance feature representation and improve Optical Character Recognition (OCR) performance. Specifically, the framework substitutes standard convolution layers in the third and fourth blocks with Deformable Convolutions, leverages adaptive dropout for regularization, and incorporates CRF for more refined sequence modeling. Extensive experiments conducted on six benchmark datasets IC13, IC15, SVT, IIIT5K, SVTP, and CUTE80 validate the proposed method's efficacy, achieving notable accuracies: 97.32% on IC13, 58.26% on IC15, 88.10% on SVT, 74.13% on IIIT5K, 82.17% on SVTP, and 66.67% on CUTE80, resulting in an average accuracy of 77.77%. These results establish a new state-of-the-art for text recognition, demonstrating the robustness of the approach across diverse and challenging datasets. 

---
# On-Device LLM for Context-Aware Wi-Fi Roaming 

**Authors**: Ju-Hyung Lee, Yanqing Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.04174)  

**Abstract**: Wireless roaming is a critical yet challenging task for maintaining seamless connectivity in dynamic mobile environments. Conventional threshold-based or heuristic schemes often fail, leading to either sticky or excessive handovers. We introduce the first cross-layer use of an on-device large language model (LLM): high-level reasoning in the application layer that issues real-time actions executed in the PHY/MAC stack. The LLM addresses two tasks: (i) context-aware AP selection, where structured prompts fuse environmental cues (e.g., location, time) to choose the best BSSID; and (ii) dynamic threshold adjustment, where the model adaptively decides when to roam. To satisfy the tight latency and resource budgets of edge hardware, we apply a suite of optimizations-chain-of-thought prompting, parameter-efficient fine-tuning, and quantization. Experiments on indoor and outdoor datasets show that our approach surpasses legacy heuristics and DRL baselines, achieving a strong balance between roaming stability and signal quality. These findings underscore the promise of application-layer LLM reasoning for lower-layer wireless control in future edge systems. 

---
# TS-SNN: Temporal Shift Module for Spiking Neural Networks 

**Authors**: Kairong Yu, Tianqing Zhang, Qi Xu, Gang Pan, Hongwei Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.04165)  

**Abstract**: Spiking Neural Networks (SNNs) are increasingly recognized for their biological plausibility and energy efficiency, positioning them as strong alternatives to Artificial Neural Networks (ANNs) in neuromorphic computing applications. SNNs inherently process temporal information by leveraging the precise timing of spikes, but balancing temporal feature utilization with low energy consumption remains a challenge. In this work, we introduce Temporal Shift module for Spiking Neural Networks (TS-SNN), which incorporates a novel Temporal Shift (TS) module to integrate past, present, and future spike features within a single timestep via a simple yet effective shift operation. A residual combination method prevents information loss by integrating shifted and original features. The TS module is lightweight, requiring only one additional learnable parameter, and can be seamlessly integrated into existing architectures with minimal additional computational cost. TS-SNN achieves state-of-the-art performance on benchmarks like CIFAR-10 (96.72\%), CIFAR-100 (80.28\%), and ImageNet (70.61\%) with fewer timesteps, while maintaining low energy consumption. This work marks a significant step forward in developing efficient and accurate SNN architectures. 

---
# R^3-VQA: "Read the Room" by Video Social Reasoning 

**Authors**: Lixing Niu, Jiapeng Li, Xingping Yu, Shu Wang, Ruining Feng, Bo Wu, Ping Wei, Yisen Wang, Lifeng Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.04147)  

**Abstract**: "Read the room" is a significant social reasoning capability in human daily life. Humans can infer others' mental states from subtle social cues. Previous social reasoning tasks and datasets lack complexity (e.g., simple scenes, basic interactions, incomplete mental state variables, single-step reasoning, etc.) and fall far short of the challenges present in real-life social interactions. In this paper, we contribute a valuable, high-quality, and comprehensive video dataset named R^3-VQA with precise and fine-grained annotations of social events and mental states (i.e., belief, intent, desire, and emotion) as well as corresponding social causal chains in complex social scenarios. Moreover, we include human-annotated and model-generated QAs. Our task R^3-VQA includes three aspects: Social Event Understanding, Mental State Estimation, and Social Causal Reasoning. As a benchmark, we comprehensively evaluate the social reasoning capabilities and consistencies of current state-of-the-art large vision-language models (LVLMs). Comprehensive experiments show that (i) LVLMs are still far from human-level consistent social reasoning in complex social scenarios; (ii) Theory of Mind (ToM) prompting can help LVLMs perform better on social reasoning tasks. We provide some of our dataset and codes in supplementary material and will release our full dataset and codes upon acceptance. 

---
# Unmasking the Canvas: A Dynamic Benchmark for Image Generation Jailbreaking and LLM Content Safety 

**Authors**: Variath Madhupal Gautham Nair, Vishal Varma Dantuluri  

**Link**: [PDF](https://arxiv.org/pdf/2505.04146)  

**Abstract**: Existing large language models (LLMs) are advancing rapidly and produce outstanding results in image generation tasks, yet their content safety checks remain vulnerable to prompt-based jailbreaks. Through preliminary testing on platforms such as ChatGPT, MetaAI, and Grok, we observed that even short, natural prompts could lead to the generation of compromising images ranging from realistic depictions of forged documents to manipulated images of public figures.
We introduce Unmasking the Canvas (UTC Benchmark; UTCB), a dynamic and scalable benchmark dataset to evaluate LLM vulnerability in image generation. Our methodology combines structured prompt engineering, multilingual obfuscation (e.g., Zulu, Gaelic, Base64), and evaluation using Groq-hosted LLaMA-3. The pipeline supports both zero-shot and fallback prompting strategies, risk scoring, and automated tagging. All generations are stored with rich metadata and curated into Bronze (non-verified), Silver (LLM-aided verification), and Gold (manually verified) tiers. UTCB is designed to evolve over time with new data sources, prompt templates, and model behaviors.
Warning: This paper includes visual examples of adversarial inputs designed to test model safety. All outputs have been redacted to ensure responsible disclosure. 

---
# Bringing legal knowledge to the public by constructing a legal question bank using large-scale pre-trained language model 

**Authors**: Mingruo Yuan, Ben Kao, Tien-Hsuan Wu, Michael M. K. Cheung, Henry W. H. Chan, Anne S. Y. Cheung, Felix W. H. Chan, Yongxi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.04132)  

**Abstract**: Access to legal information is fundamental to access to justice. Yet accessibility refers not only to making legal documents available to the public, but also rendering legal information comprehensible to them. A vexing problem in bringing legal information to the public is how to turn formal legal documents such as legislation and judgments, which are often highly technical, to easily navigable and comprehensible knowledge to those without legal education. In this study, we formulate a three-step approach for bringing legal knowledge to laypersons, tackling the issues of navigability and comprehensibility. First, we translate selected sections of the law into snippets (called CLIC-pages), each being a small piece of article that focuses on explaining certain technical legal concept in layperson's terms. Second, we construct a Legal Question Bank (LQB), which is a collection of legal questions whose answers can be found in the CLIC-pages. Third, we design an interactive CLIC Recommender (CRec). Given a user's verbal description of a legal situation that requires a legal solution, CRec interprets the user's input and shortlists questions from the question bank that are most likely relevant to the given legal situation and recommends their corresponding CLIC pages where relevant legal knowledge can be found. In this paper we focus on the technical aspects of creating an LQB. We show how large-scale pre-trained language models, such as GPT-3, can be used to generate legal questions. We compare machine-generated questions (MGQs) against human-composed questions (HCQs) and find that MGQs are more scalable, cost-effective, and more diversified, while HCQs are more precise. We also show a prototype of CRec and illustrate through an example how our 3-step approach effectively brings relevant legal knowledge to the public. 

---
# LLMs' Suitability for Network Security: A Case Study of STRIDE Threat Modeling 

**Authors**: AbdulAziz AbdulGhaffar, Ashraf Matrawy  

**Link**: [PDF](https://arxiv.org/pdf/2505.04101)  

**Abstract**: Artificial Intelligence (AI) is expected to be an integral part of next-generation AI-native 6G networks. With the prevalence of AI, researchers have identified numerous use cases of AI in network security. However, there are almost nonexistent studies that analyze the suitability of Large Language Models (LLMs) in network security. To fill this gap, we examine the suitability of LLMs in network security, particularly with the case study of STRIDE threat modeling. We utilize four prompting techniques with five LLMs to perform STRIDE classification of 5G threats. From our evaluation results, we point out key findings and detailed insights along with the explanation of the possible underlying factors influencing the behavior of LLMs in the modeling of certain threats. The numerical results and the insights support the necessity for adjusting and fine-tuning LLMs for network security use cases. 

---
# An Empirical Study of OpenAI API Discussions on Stack Overflow 

**Authors**: Xiang Chen, Jibin Wang, Chaoyang Gao, Xiaolin Ju, Zhanqi Cui  

**Link**: [PDF](https://arxiv.org/pdf/2505.04084)  

**Abstract**: The rapid advancement of large language models (LLMs), represented by OpenAI's GPT series, has significantly impacted various domains such as natural language processing, software development, education, healthcare, finance, and scientific research. However, OpenAI APIs introduce unique challenges that differ from traditional APIs, such as the complexities of prompt engineering, token-based cost management, non-deterministic outputs, and operation as black boxes. To the best of our knowledge, the challenges developers encounter when using OpenAI APIs have not been explored in previous empirical studies. To fill this gap, we conduct the first comprehensive empirical study by analyzing 2,874 OpenAI API-related discussions from the popular Q&A forum Stack Overflow. We first examine the popularity and difficulty of these posts. After manually categorizing them into nine OpenAI API-related categories, we identify specific challenges associated with each category through topic modeling analysis. Based on our empirical findings, we finally propose actionable implications for developers, LLM vendors, and researchers. 

---
# Plexus: Taming Billion-edge Graphs with 3D Parallel GNN Training 

**Authors**: Aditya K. Ranjan, Siddharth Singh, Cunyang Wei, Abhinav Bhatele  

**Link**: [PDF](https://arxiv.org/pdf/2505.04083)  

**Abstract**: Graph neural networks have emerged as a potent class of neural networks capable of leveraging the connectivity and structure of real-world graphs to learn intricate properties and relationships between nodes. Many real-world graphs exceed the memory capacity of a GPU due to their sheer size, and using GNNs on them requires techniques such as mini-batch sampling to scale. However, this can lead to reduced accuracy in some cases, and sampling and data transfer from the CPU to the GPU can also slow down training. On the other hand, distributed full-graph training suffers from high communication overhead and load imbalance due to the irregular structure of graphs. We propose Plexus, a three-dimensional (3D) parallel approach for full-graph training that tackles these issues and scales to billion-edge graphs. Additionally, we introduce optimizations such as a permutation scheme for load balancing, and a performance model to predict the optimal 3D configuration. We evaluate Plexus on several graph datasets and show scaling results for up to 2048 GPUs on Perlmutter, which is 33% of the machine, and 2048 GCDs on Frontier. Plexus achieves unprecedented speedups of 2.3x-12.5x over existing methods and a reduction in the time to solution by 5.2-8.7x on Perlmutter and 7-54.2x on Frontier. 

---
# LLM-e Guess: Can LLMs Capabilities Advance Without Hardware Progress? 

**Authors**: Teddy Foley, Spencer Guo, Henry Josephson, Anqi Qu, Jack Sanderson  

**Link**: [PDF](https://arxiv.org/pdf/2505.04075)  

**Abstract**: This paper examines whether large language model (LLM) capabilities can continue to advance without additional compute by analyzing the development and role of algorithms used in state-of-the-art LLMs. Motivated by regulatory efforts that have largely focused on restricting access to high-performance hardware, we ask: Can LLMs progress in a compute-constrained environment, and how do algorithmic innovations perform under such conditions?
To address these questions, we introduce a novel classification framework that distinguishes between compute-dependent innovations -- which yield disproportionate benefits at high compute levels (e.g., the Transformer architecture and mixture-of-experts models) and compute-independent innovations, which improve efficiency across all compute scales (e.g., rotary positional encoding, FlashAttention, or layer normalization). We quantify these contributions using a metric called compute-equivalent gain (CEG), which estimates the additional compute that would be required to achieve similar improvements without these algorithmic advancements.
To validate this framework, we conduct small-scale training experiments with a scaled-down GPT-2 model. Our results confirm that compute-independent advancements yield meaningful performance gains even in resource-constrained settings, with a CEG of up to $3.5\times$ over a baseline model. By contrast, compute-dependent advancements provided little benefit or even degraded performance at the small scale, reinforcing the importance of compute availability for certain algorithmic gains. 

---
# Advancing and Benchmarking Personalized Tool Invocation for LLMs 

**Authors**: Xu Huang, Yuefeng Huang, Weiwen Liu, Xingshan Zeng, Yasheng Wang, Ruiming Tang, Hong Xie, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.04072)  

**Abstract**: Tool invocation is a crucial mechanism for extending the capabilities of Large Language Models (LLMs) and has recently garnered significant attention. It enables LLMs to solve complex problems through tool calls while accessing up-to-date world knowledge. However, existing work primarily focuses on the fundamental ability of LLMs to invoke tools for problem-solving, without considering personalized constraints in tool invocation. In this work, we introduce the concept of Personalized Tool Invocation and define two key tasks: Tool Preference and Profile-dependent Query. Tool Preference addresses user preferences when selecting among functionally similar tools, while Profile-dependent Query considers cases where a user query lacks certain tool parameters, requiring the model to infer them from the user profile. To tackle these challenges, we propose PTool, a data synthesis framework designed for personalized tool invocation. Additionally, we construct \textbf{PTBench}, the first benchmark for evaluating personalized tool invocation. We then fine-tune various open-source models, demonstrating the effectiveness of our framework and providing valuable insights. Our benchmark is public at this https URL. 

---
# Izhikevich-Inspired Temporal Dynamics for Enhancing Privacy, Efficiency, and Transferability in Spiking Neural Networks 

**Authors**: Ayana Moshruba, Hamed Poursiami, Maryam Parsa  

**Link**: [PDF](https://arxiv.org/pdf/2505.04034)  

**Abstract**: Biological neurons exhibit diverse temporal spike patterns, which are believed to support efficient, robust, and adaptive neural information processing. While models such as Izhikevich can replicate a wide range of these firing dynamics, their complexity poses challenges for directly integrating them into scalable spiking neural networks (SNN) training pipelines. In this work, we propose two probabilistically driven, input-level temporal spike transformations: Poisson-Burst and Delayed-Burst that introduce biologically inspired temporal variability directly into standard Leaky Integrate-and-Fire (LIF) neurons. This enables scalable training and systematic evaluation of how spike timing dynamics affect privacy, generalization, and learning performance. Poisson-Burst modulates burst occurrence based on input intensity, while Delayed-Burst encodes input strength through burst onset timing. Through extensive experiments across multiple benchmarks, we demonstrate that Poisson-Burst maintains competitive accuracy and lower resource overhead while exhibiting enhanced privacy robustness against membership inference attacks, whereas Delayed-Burst provides stronger privacy protection at a modest accuracy trade-off. These findings highlight the potential of biologically grounded temporal spike dynamics in improving the privacy, generalization and biological plausibility of neuromorphic learning systems. 

---
# Prism: Unleashing GPU Sharing for Cost-Efficient Multi-LLM Serving 

**Authors**: Shan Yu, Jiarong Xing, Yifan Qiao, Mingyuan Ma, Yangmin Li, Yang Wang, Shuo Yang, Zhiqiang Xie, Shiyi Cao, Ke Bao, Ion Stoica, Harry Xu, Ying Sheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04021)  

**Abstract**: Serving large language models (LLMs) is expensive, especially for providers hosting many models, making cost reduction essential. The unique workload patterns of serving multiple LLMs (i.e., multi-LLM serving) create new opportunities and challenges for this task. The long-tail popularity of models and their long idle periods present opportunities to improve utilization through GPU sharing. However, existing GPU sharing systems lack the ability to adjust their resource allocation and sharing policies at runtime, making them ineffective at meeting latency service-level objectives (SLOs) under rapidly fluctuating workloads.
This paper presents Prism, a multi-LLM serving system that unleashes the full potential of GPU sharing to achieve both cost efficiency and SLO attainment. At its core, Prism tackles a key limitation of existing systems$\unicode{x2014}$the lack of $\textit{cross-model memory coordination}$, which is essential for flexibly sharing GPU memory across models under dynamic workloads. Prism achieves this with two key designs. First, it supports on-demand memory allocation by dynamically mapping physical to virtual memory pages, allowing flexible memory redistribution among models that space- and time-share a GPU. Second, it improves memory efficiency through a two-level scheduling policy that dynamically adjusts sharing strategies based on models' runtime demands. Evaluations on real-world traces show that Prism achieves more than $2\times$ cost savings and $3.3\times$ SLO attainment compared to state-of-the-art systems. 

---
# SLOT: Structuring the Output of Large Language Models 

**Authors**: Darren Yow-Bang Wang, Zhengyuan Shen, Soumya Smruti Mishra, Zhichao Xu, Yifei Teng, Haibo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.04016)  

**Abstract**: Structured outputs are essential for large language models (LLMs) in critical applications like agents and information extraction. Despite their capabilities, LLMs often generate outputs that deviate from predefined schemas, significantly hampering reliable application development. We present SLOT (Structured LLM Output Transformer), a model-agnostic approach that transforms unstructured LLM outputs into precise structured formats. While existing solutions predominantly rely on constrained decoding techniques or are tightly coupled with specific models, SLOT employs a fine-tuned lightweight language model as a post-processing layer, achieving flexibility across various LLMs and schema specifications. We introduce a systematic pipeline for data curation and synthesis alongside a formal evaluation methodology that quantifies both schema accuracy and content fidelity. Our results demonstrate that fine-tuned Mistral-7B model with constrained decoding achieves near perfect schema accuracy (99.5%) and content similarity (94.0%), outperforming Claude-3.5-Sonnet by substantial margins (+25 and +20 percentage points, respectively). Notably, even compact models like Llama-3.2-1B can match or exceed the structured output capabilities of much larger proprietary models when equipped with SLOT, enabling reliable structured generation in resource-constrained environments. 

---
# MergeGuard: Efficient Thwarting of Trojan Attacks in Machine Learning Models 

**Authors**: Soheil Zibakhsh Shabgahi, Yaman Jandali, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2505.04015)  

**Abstract**: This paper proposes MergeGuard, a novel methodology for mitigation of AI Trojan attacks. Trojan attacks on AI models cause inputs embedded with triggers to be misclassified to an adversary's target class, posing a significant threat to model usability trained by an untrusted third party. The core of MergeGuard is a new post-training methodology for linearizing and merging fully connected layers which we show simultaneously improves model generalizability and performance. Our Proof of Concept evaluation on Transformer models demonstrates that MergeGuard maintains model accuracy while decreasing trojan attack success rate, outperforming commonly used (post-training) Trojan mitigation by fine-tuning methodologies. 

---
# PARC: Physics-based Augmentation with Reinforcement Learning for Character Controllers 

**Authors**: Michael Xu, Yi Shi, KangKang Yin, Xue Bin Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.04002)  

**Abstract**: Humans excel in navigating diverse, complex environments with agile motor skills, exemplified by parkour practitioners performing dynamic maneuvers, such as climbing up walls and jumping across gaps. Reproducing these agile movements with simulated characters remains challenging, in part due to the scarcity of motion capture data for agile terrain traversal behaviors and the high cost of acquiring such data. In this work, we introduce PARC (Physics-based Augmentation with Reinforcement Learning for Character Controllers), a framework that leverages machine learning and physics-based simulation to iteratively augment motion datasets and expand the capabilities of terrain traversal controllers. PARC begins by training a motion generator on a small dataset consisting of core terrain traversal skills. The motion generator is then used to produce synthetic data for traversing new terrains. However, these generated motions often exhibit artifacts, such as incorrect contacts or discontinuities. To correct these artifacts, we train a physics-based tracking controller to imitate the motions in simulation. The corrected motions are then added to the dataset, which is used to continue training the motion generator in the next iteration. PARC's iterative process jointly expands the capabilities of the motion generator and tracker, creating agile and versatile models for interacting with complex environments. PARC provides an effective approach to develop controllers for agile terrain traversal, which bridges the gap between the scarcity of motion data and the need for versatile character controllers. 

---
# Can Large Language Models Predict Parallel Code Performance? 

**Authors**: Gregory Bolet, Giorgis Georgakoudis, Harshitha Menon, Konstantinos Parasyris, Niranjan Hasabnis, Hayden Estes, Kirk W. Cameron, Gal Oren  

**Link**: [PDF](https://arxiv.org/pdf/2505.03988)  

**Abstract**: Accurate determination of the performance of parallel GPU code typically requires execution-time profiling on target hardware -- an increasingly prohibitive step due to limited access to high-end GPUs. This paper explores whether Large Language Models (LLMs) can offer an alternative approach for GPU performance prediction without relying on hardware. We frame the problem as a roofline classification task: given the source code of a GPU kernel and the hardware specifications of a target GPU, can an LLM predict whether the GPU kernel is compute-bound or bandwidth-bound?
For this study, we build a balanced dataset of 340 GPU kernels, obtained from HeCBench benchmark and written in CUDA and OpenMP, along with their ground-truth labels obtained via empirical GPU profiling. We evaluate LLMs across four scenarios: (1) with access to profiling data of the kernel source, (2) zero-shot with source code only, (3) few-shot with code and label pairs, and (4) fine-tuned on a small custom dataset.
Our results show that state-of-the-art LLMs have a strong understanding of the Roofline model, achieving 100% classification accuracy when provided with explicit profiling data. We also find that reasoning-capable LLMs significantly outperform standard LLMs in zero- and few-shot settings, achieving up to 64% accuracy on GPU source codes, without profiling information. Lastly, we find that LLM fine-tuning will require much more data than what we currently have available.
This work is among the first to use LLMs for source-level roofline performance prediction via classification, and illustrates their potential to guide optimization efforts when runtime profiling is infeasible. Our findings suggest that with better datasets and prompt strategies, LLMs could become practical tools for HPC performance analysis and performance portability. 

---
# Diffusion Models are Secretly Exchangeable: Parallelizing DDPMs via Autospeculation 

**Authors**: Hengyuan Hu, Aniket Das, Dorsa Sadigh, Nima Anari  

**Link**: [PDF](https://arxiv.org/pdf/2505.03983)  

**Abstract**: Denoising Diffusion Probabilistic Models (DDPMs) have emerged as powerful tools for generative modeling. However, their sequential computation requirements lead to significant inference-time bottlenecks. In this work, we utilize the connection between DDPMs and Stochastic Localization to prove that, under an appropriate reparametrization, the increments of DDPM satisfy an exchangeability property. This general insight enables near-black-box adaptation of various performance optimization techniques from autoregressive models to the diffusion setting. To demonstrate this, we introduce \emph{Autospeculative Decoding} (ASD), an extension of the widely used speculative decoding algorithm to DDPMs that does not require any auxiliary draft models. Our theoretical analysis shows that ASD achieves a $\tilde{O} (K^{\frac{1}{3}})$ parallel runtime speedup over the $K$ step sequential DDPM. We also demonstrate that a practical implementation of autospeculative decoding accelerates DDPM inference significantly in various domains. 

---
# X-Reasoner: Towards Generalizable Reasoning Across Modalities and Domains 

**Authors**: Qianchu Liu, Sheng Zhang, Guanghui Qin, Timothy Ossowski, Yu Gu, Ying Jin, Sid Kiblawi, Sam Preston, Mu Wei, Paul Vozila, Tristan Naumann, Hoifung Poon  

**Link**: [PDF](https://arxiv.org/pdf/2505.03981)  

**Abstract**: Recent proprietary models (e.g., o3) have begun to demonstrate strong multimodal reasoning capabilities. Yet, most existing open-source research concentrates on training text-only reasoning models, with evaluations limited to mainly mathematical and general-domain tasks. Therefore, it remains unclear how to effectively extend reasoning capabilities beyond text input and general domains. This paper explores a fundamental research question: Is reasoning generalizable across modalities and domains? Our findings support an affirmative answer: General-domain text-based post-training can enable such strong generalizable reasoning. Leveraging this finding, we introduce X-Reasoner, a vision-language model post-trained solely on general-domain text for generalizable reasoning, using a two-stage approach: an initial supervised fine-tuning phase with distilled long chain-of-thoughts, followed by reinforcement learning with verifiable rewards. Experiments show that X-Reasoner successfully transfers reasoning capabilities to both multimodal and out-of-domain settings, outperforming existing state-of-the-art models trained with in-domain and multimodal data across various general and medical benchmarks (Figure 1). Additionally, we find that X-Reasoner's performance in specialized domains can be further enhanced through continued training on domain-specific text-only data. Building upon this, we introduce X-Reasoner-Med, a medical-specialized variant that achieves new state of the art on numerous text-only and multimodal medical benchmarks. 

---
# Deep Learning Framework for Infrastructure Maintenance: Crack Detection and High-Resolution Imaging of Infrastructure Surfaces 

**Authors**: Nikhil M. Pawar, Jorge A. Prozzi, Feng Hong, Surya Sarat Chandra Congress  

**Link**: [PDF](https://arxiv.org/pdf/2505.03974)  

**Abstract**: Recently, there has been an impetus for the application of cutting-edge data collection platforms such as drones mounted with camera sensors for infrastructure asset management. However, the sensor characteristics, proximity to the structure, hard-to-reach access, and environmental conditions often limit the resolution of the datasets. A few studies used super-resolution techniques to address the problem of low-resolution images. Nevertheless, these techniques were observed to increase computational cost and false alarms of distress detection due to the consideration of all the infrastructure images i.e., positive and negative distress classes. In order to address the pre-processing of false alarm and achieve efficient super-resolution, this study developed a framework consisting of convolutional neural network (CNN) and efficient sub-pixel convolutional neural network (ESPCNN). CNN accurately classified both the classes. ESPCNN, which is the lightweight super-resolution technique, generated high-resolution infrastructure image of positive distress obtained from CNN. The ESPCNN outperformed bicubic interpolation in all the evaluation metrics for super-resolution. Based on the performance metrics, the combination of CNN and ESPCNN was observed to be effective in preprocessing the infrastructure images with negative distress, reducing the computational cost and false alarms in the next step of super-resolution. The visual inspection showed that EPSCNN is able to capture crack propagation, complex geometry of even minor cracks. The proposed framework is expected to help the highway agencies in accurately performing distress detection and assist in efficient asset management practices. 

---
# Decentralized Distributed Proximal Policy Optimization (DD-PPO) for High Performance Computing Scheduling on Multi-User Systems 

**Authors**: Matthew Sgambati, Aleksandar Vakanski, Matthew Anderson  

**Link**: [PDF](https://arxiv.org/pdf/2505.03946)  

**Abstract**: Resource allocation in High Performance Computing (HPC) environments presents a complex and multifaceted challenge for job scheduling algorithms. Beyond the efficient allocation of system resources, schedulers must account for and optimize multiple performance metrics, including job wait time and system utilization. While traditional rule-based scheduling algorithms dominate the current deployments of HPC systems, the increasing heterogeneity and scale of those systems is expected to challenge the efficiency and flexibility of those algorithms in minimizing job wait time and maximizing utilization. Recent research efforts have focused on leveraging advancements in Reinforcement Learning (RL) to develop more adaptable and intelligent scheduling strategies. Recent RL-based scheduling approaches have explored a range of algorithms, from Deep Q-Networks (DQN) to Proximal Policy Optimization (PPO), and more recently, hybrid methods that integrate Graph Neural Networks with RL techniques. However, a common limitation across these methods is their reliance on relatively small datasets, and these methods face scalability issues when using large datasets. This study introduces a novel RL-based scheduler utilizing the Decentralized Distributed Proximal Policy Optimization (DD-PPO) algorithm, which supports large-scale distributed training across multiple workers without requiring parameter synchronization at every step. By eliminating reliance on centralized updates to a shared policy, the DD-PPO scheduler enhances scalability, training efficiency, and sample utilization. The validation dataset leveraged over 11.5 million real HPC job traces for comparing DD-PPO performance between traditional and advanced scheduling approaches, and the experimental results demonstrate improved scheduling performance in comparison to both rule-based schedulers and existing RL-based scheduling algorithms. 

---
# AI-Driven Security in Cloud Computing: Enhancing Threat Detection, Automated Response, and Cyber Resilience 

**Authors**: Shamnad Mohamed Shaffi, Sunish Vengathattil, Jezeena Nikarthil Sidhick, Resmi Vijayan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03945)  

**Abstract**: Cloud security concerns have been greatly realized in recent years due to the increase of complicated threats in the computing world. Many traditional solutions do not work well in real-time to detect or prevent more complex threats. Artificial intelligence is today regarded as a revolution in determining a protection plan for cloud data architecture through machine learning, statistical visualization of computing infrastructure, and detection of security breaches followed by counteraction. These AI-enabled systems make work easier as more network activities are scrutinized, and any anomalous behavior that might be a precursor to a more serious breach is prevented. This paper examines ways AI can enhance cloud security by applying predictive analytics, behavior-based security threat detection, and AI-stirring encryption. It also outlines the problems of the previous security models and how AI overcomes them. For a similar reason, issues like data privacy, biases in the AI model, and regulatory compliance are also covered. So, AI improves the protection of cloud computing contexts; however, more efforts are needed in the subsequent phases to extend the technology's reliability, modularity, and ethical aspects. This means that AI can be blended with other new computing technologies, including blockchain, to improve security frameworks further. The paper discusses the current trends in securing cloud data architecture using AI and presents further research and application directions. 

---
# A Graphical Global Optimization Framework for Parameter Estimation of Statistical Models with Nonconvex Regularization Functions 

**Authors**: Danial Davarnia, Mohammadreza Kiaghadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03899)  

**Abstract**: Optimization problems with norm-bounding constraints arise in a variety of applications, including portfolio optimization, machine learning, and feature selection. A common approach to these problems involves relaxing the norm constraint via Lagrangian relaxation, transforming it into a regularization term in the objective function. A particularly challenging class includes the zero-norm function, which promotes sparsity in statistical parameter estimation. Most existing exact methods for solving these problems introduce binary variables and artificial bounds to reformulate them as higher-dimensional mixed-integer programs, solvable by standard solvers. Other exact approaches exploit specific structural properties of the objective, making them difficult to generalize across different problem types. Alternative methods employ nonconvex penalties with favorable statistical characteristics, but these are typically addressed using heuristic or local optimization techniques due to their structural complexity. In this paper, we propose a novel graph-based method to globally solve optimization problems involving generalized norm-bounding constraints. Our approach encompasses standard $\ell_p$-norms for $p \in [0, \infty)$ and nonconvex penalties such as SCAD and MCP. We leverage decision diagrams to construct strong convex relaxations directly in the original variable space, eliminating the need for auxiliary variables or artificial bounds. Integrated into a spatial branch-and-cut framework, our method guarantees convergence to the global optimum. We demonstrate its effectiveness through preliminary computational experiments on benchmark sparse linear regression problems involving complex nonconvex penalties, which are not tractable using existing global optimization techniques. 

---
# Novel Extraction of Discriminative Fine-Grained Feature to Improve Retinal Vessel Segmentation 

**Authors**: Shuang Zeng, Chee Hong Lee, Micky C Nnamdi, Wenqi Shi, J Ben Tamo, Lei Zhu, Hangzhou He, Xinliang Zhang, Qian Chen, May D. Wang, Yanye Lu, Qiushi Ren  

**Link**: [PDF](https://arxiv.org/pdf/2505.03896)  

**Abstract**: Retinal vessel segmentation is a vital early detection method for several severe ocular diseases. Despite significant progress in retinal vessel segmentation with the advancement of Neural Networks, there are still challenges to overcome. Specifically, retinal vessel segmentation aims to predict the class label for every pixel within a fundus image, with a primary focus on intra-image discrimination, making it vital for models to extract more discriminative features. Nevertheless, existing methods primarily focus on minimizing the difference between the output from the decoder and the label, but ignore fully using feature-level fine-grained representations from the encoder. To address these issues, we propose a novel Attention U-shaped Kolmogorov-Arnold Network named AttUKAN along with a novel Label-guided Pixel-wise Contrastive Loss for retinal vessel segmentation. Specifically, we implement Attention Gates into Kolmogorov-Arnold Networks to enhance model sensitivity by suppressing irrelevant feature activations and model interpretability by non-linear modeling of KAN blocks. Additionally, we also design a novel Label-guided Pixel-wise Contrastive Loss to supervise our proposed AttUKAN to extract more discriminative features by distinguishing between foreground vessel-pixel pairs and background pairs. Experiments are conducted across four public datasets including DRIVE, STARE, CHASE_DB1, HRF and our private dataset. AttUKAN achieves F1 scores of 82.50%, 81.14%, 81.34%, 80.21% and 80.09%, along with MIoU scores of 70.24%, 68.64%, 68.59%, 67.21% and 66.94% in the above datasets, which are the highest compared to 11 networks for retinal vessel segmentation. Quantitative and qualitative results show that our AttUKAN achieves state-of-the-art performance and outperforms existing retinal vessel segmentation methods. Our code will be available at this https URL. 

---
# Scratch Copilot: Supporting Youth Creative Coding with AI 

**Authors**: Stefania Druga, Amy J. Ko  

**Link**: [PDF](https://arxiv.org/pdf/2505.03867)  

**Abstract**: Creative coding platforms like Scratch have democratized programming for children, yet translating imaginative ideas into functional code remains a significant hurdle for many young learners. While AI copilots assist adult programmers, few tools target children in block-based environments. Building on prior research \cite{druga_how_2021,druga2023ai, druga2023scratch}, we present Cognimates Scratch Copilot: an AI-powered assistant integrated into a Scratch-like environment, providing real-time support for ideation, code generation, debugging, and asset creation. This paper details the system architecture and findings from an exploratory qualitative evaluation with 18 international children (ages 7--12). Our analysis reveals how the AI Copilot supported key creative coding processes, particularly aiding ideation and debugging. Crucially, it also highlights how children actively negotiated the use of AI, demonstrating strong agency by adapting or rejecting suggestions to maintain creative control. Interactions surfaced design tensions between providing helpful scaffolding and fostering independent problem-solving, as well as learning opportunities arising from navigating AI limitations and errors. Findings indicate Cognimates Scratch Copilot's potential to enhance creative self-efficacy and engagement. Based on these insights, we propose initial design guidelines for AI coding assistants that prioritize youth agency and critical interaction alongside supportive scaffolding. 

---
# From Glue-Code to Protocols: A Critical Analysis of A2A and MCP Integration for Scalable Agent Systems 

**Authors**: Qiaomu Li, Ying Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.03864)  

**Abstract**: Artificial intelligence is rapidly evolving towards multi-agent systems where numerous AI agents collaborate and interact with external tools. Two key open standards, Google's Agent to Agent (A2A) protocol for inter-agent communication and Anthropic's Model Context Protocol (MCP) for standardized tool access, promise to overcome the limitations of fragmented, custom integration approaches. While their potential synergy is significant, this paper argues that effectively integrating A2A and MCP presents unique, emergent challenges at their intersection, particularly concerning semantic interoperability between agent tasks and tool capabilities, the compounded security risks arising from combined discovery and execution, and the practical governance required for the envisioned "Agent Economy". This work provides a critical analysis, moving beyond a survey to evaluate the practical implications and inherent difficulties of combining these horizontal and vertical integration standards. We examine the benefits (e.g., specialization, scalability) while critically assessing their dependencies and trade-offs in an integrated context. We identify key challenges increased by the integration, including novel security vulnerabilities, privacy complexities, debugging difficulties across protocols, and the need for robust semantic negotiation mechanisms. In summary, A2A+MCP offers a vital architectural foundation, but fully realizing its potential requires substantial advancements to manage the complexities of their combined operation. 

---
# Data-Driven Falsification of Cyber-Physical Systems 

**Authors**: Atanu Kundu, Sauvik Gon, Rajarshi Ray  

**Link**: [PDF](https://arxiv.org/pdf/2505.03863)  

**Abstract**: Cyber-Physical Systems (CPS) are abundant in safety-critical domains such as healthcare, avionics, and autonomous vehicles. Formal verification of their operational safety is, therefore, of utmost importance. In this paper, we address the falsification problem, where the focus is on searching for an unsafe execution in the system instead of proving their absence. The contribution of this paper is a framework that (a) connects the falsification of CPS with the falsification of deep neural networks (DNNs) and (b) leverages the inherent interpretability of Decision Trees for faster falsification of CPS. This is achieved by: (1) building a surrogate model of the CPS under test, either as a DNN model or a Decision Tree, (2) application of various DNN falsification tools to falsify CPS, and (3) a novel falsification algorithm guided by the explanations of safety violations of the CPS model extracted from its Decision Tree surrogate. The proposed framework has the potential to exploit a repertoire of \emph{adversarial attack} algorithms designed to falsify robustness properties of DNNs, as well as state-of-the-art falsification algorithms for DNNs. Although the presented methodology is applicable to systems that can be executed/simulated in general, we demonstrate its effectiveness, particularly in CPS. We show that our framework, implemented as a tool \textsc{FlexiFal}, can detect hard-to-find counterexamples in CPS that have linear and non-linear dynamics. Decision tree-guided falsification shows promising results in efficiently finding multiple counterexamples in the ARCH-COMP 2024 falsification benchmarks~\cite{khandait2024arch}. 

---
# Deepfakes on Demand: the rise of accessible non-consensual deepfake image generators 

**Authors**: Will Hawkins, Chris Russell, Brent Mittelstadt  

**Link**: [PDF](https://arxiv.org/pdf/2505.03859)  

**Abstract**: Advances in multimodal machine learning have made text-to-image (T2I) models increasingly accessible and popular. However, T2I models introduce risks such as the generation of non-consensual depictions of identifiable individuals, otherwise known as deepfakes. This paper presents an empirical study exploring the accessibility of deepfake model variants online. Through a metadata analysis of thousands of publicly downloadable model variants on two popular repositories, Hugging Face and Civitai, we demonstrate a huge rise in easily accessible deepfake models. Almost 35,000 examples of publicly downloadable deepfake model variants are identified, primarily hosted on Civitai. These deepfake models have been downloaded almost 15 million times since November 2022, with the models targeting a range of individuals from global celebrities to Instagram users with under 10,000 followers. Both Stable Diffusion and Flux models are used for the creation of deepfake models, with 96% of these targeting women and many signalling intent to generate non-consensual intimate imagery (NCII). Deepfake model variants are often created via the parameter-efficient fine-tuning technique known as low rank adaptation (LoRA), requiring as few as 20 images, 24GB VRAM, and 15 minutes of time, making this process widely accessible via consumer-grade computers. Despite these models violating the Terms of Service of hosting platforms, and regulation seeking to prevent dissemination, these results emphasise the pressing need for greater action to be taken against the creation of deepfakes and NCII. 

---
# An Active Inference Model of Covert and Overt Visual Attention 

**Authors**: Tin Mišić, Karlo Koledić, Fabio Bonsignorio, Ivan Petrović, Ivan Marković  

**Link**: [PDF](https://arxiv.org/pdf/2505.03856)  

**Abstract**: The ability to selectively attend to relevant stimuli while filtering out distractions is essential for agents that process complex, high-dimensional sensory input. This paper introduces a model of covert and overt visual attention through the framework of active inference, utilizing dynamic optimization of sensory precisions to minimize free-energy. The model determines visual sensory precisions based on both current environmental beliefs and sensory input, influencing attentional allocation in both covert and overt modalities. To test the effectiveness of the model, we analyze its behavior in the Posner cueing task and a simple target focus task using two-dimensional(2D) visual data. Reaction times are measured to investigate the interplay between exogenous and endogenous attention, as well as valid and invalid cueing. The results show that exogenous and valid cues generally lead to faster reaction times compared to endogenous and invalid cues. Furthermore, the model exhibits behavior similar to inhibition of return, where previously attended locations become suppressed after a specific cue-target onset asynchrony interval. Lastly, we investigate different aspects of overt attention and show that involuntary, reflexive saccades occur faster than intentional ones, but at the expense of adaptability. 

---
# GRAPE: Heterogeneous Graph Representation Learning for Genetic Perturbation with Coding and Non-Coding Biotype 

**Authors**: Changxi Chi, Jun Xia, Jingbo Zhou, Jiabei Cheng, Chang Yu, Stan Z. Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.03853)  

**Abstract**: Predicting genetic perturbations enables the identification of potentially crucial genes prior to wet-lab experiments, significantly improving overall experimental efficiency. Since genes are the foundation of cellular life, building gene regulatory networks (GRN) is essential to understand and predict the effects of genetic perturbations. However, current methods fail to fully leverage gene-related information, and solely rely on simple evaluation metrics to construct coarse-grained GRN. More importantly, they ignore functional differences between biotypes, limiting the ability to capture potential gene interactions. In this work, we leverage pre-trained large language model and DNA sequence model to extract features from gene descriptions and DNA sequence data, respectively, which serve as the initialization for gene representations. Additionally, we introduce gene biotype information for the first time in genetic perturbation, simulating the distinct roles of genes with different biotypes in regulating cellular processes, while capturing implicit gene relationships through graph structure learning (GSL). We propose GRAPE, a heterogeneous graph neural network (HGNN) that leverages gene representations initialized with features from descriptions and sequences, models the distinct roles of genes with different biotypes, and dynamically refines the GRN through GSL. The results on publicly available datasets show that our method achieves state-of-the-art performance. 

---
# Impact Analysis of Inference Time Attack of Perception Sensors on Autonomous Vehicles 

**Authors**: Hanlin Chen, Simin Chen, Wenyu Li, Wei Yang, Yiheng Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.03850)  

**Abstract**: As a safety-critical cyber-physical system, cybersecurity and related safety issues for Autonomous Vehicles (AVs) have been important research topics for a while. Among all the modules on AVs, perception is one of the most accessible attack surfaces, as drivers and AVs have no control over the outside environment. Most current work targeting perception security for AVs focuses on perception correctness. In this work, we propose an impact analysis based on inference time attacks for autonomous vehicles. We demonstrate in a simulation system that such inference time attacks can also threaten the safety of both the ego vehicle and other traffic participants. 

---
# Advanced Clustering Framework for Semiconductor Image Analytics Integrating Deep TDA with Self-Supervised and Transfer Learning Techniques 

**Authors**: Janhavi Giri, Attila Lengyel, Don Kent, Edward Kibardin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03848)  

**Abstract**: Semiconductor manufacturing generates vast amounts of image data, crucial for defect identification and yield optimization, yet often exceeds manual inspection capabilities. Traditional clustering techniques struggle with high-dimensional, unlabeled data, limiting their effectiveness in capturing nuanced patterns. This paper introduces an advanced clustering framework that integrates deep Topological Data Analysis (TDA) with self-supervised and transfer learning techniques, offering a novel approach to unsupervised image clustering. TDA captures intrinsic topological features, while self-supervised learning extracts meaningful representations from unlabeled data, reducing reliance on labeled datasets. Transfer learning enhances the framework's adaptability and scalability, allowing fine-tuning to new datasets without retraining from scratch. Validated on synthetic and open-source semiconductor image datasets, the framework successfully identifies clusters aligned with defect patterns and process variations. This study highlights the transformative potential of combining TDA, self-supervised learning, and transfer learning, providing a scalable solution for proactive process monitoring and quality control in semiconductor manufacturing and other domains with large-scale image datasets. 

---
# GAME: Learning Multimodal Interactions via Graph Structures for Personality Trait Estimation 

**Authors**: Kangsheng Wang, Yuhang Li, Chengwei Ye, Yufei Lin, Huanzhen Zhang, Bohan Hu, Linuo Xu, Shuyan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03846)  

**Abstract**: Apparent personality analysis from short videos poses significant chal-lenges due to the complex interplay of visual, auditory, and textual cues. In this paper, we propose GAME, a Graph-Augmented Multimodal Encoder designed to robustly model and fuse multi-source features for automatic personality prediction. For the visual stream, we construct a facial graph and introduce a dual-branch Geo Two-Stream Network, which combines Graph Convolutional Networks (GCNs) and Convolutional Neural Net-works (CNNs) with attention mechanisms to capture both structural and appearance-based facial cues. Complementing this, global context and iden-tity features are extracted using pretrained ResNet18 and VGGFace back-bones. To capture temporal dynamics, frame-level features are processed by a BiGRU enhanced with temporal attention modules. Meanwhile, audio representations are derived from the VGGish network, and linguistic se-mantics are captured via the XLM-Roberta transformer. To achieve effective multimodal integration, we propose a Channel Attention-based Fusion module, followed by a Multi-Layer Perceptron (MLP) regression head for predicting personality traits. Extensive experiments show that GAME con-sistently outperforms existing methods across multiple benchmarks, vali-dating its effectiveness and generalizability. 

---
# A Deep Learning approach for Depressive Symptoms assessment in Parkinson's disease patients using facial videos 

**Authors**: Ioannis Kyprakis, Vasileios Skaramagkas, Iro Boura, Georgios Karamanis, Dimitrios I. Fotiadis, Zinovia Kefalopoulou, Cleanthe Spanaki, Manolis Tsiknakis  

**Link**: [PDF](https://arxiv.org/pdf/2505.03845)  

**Abstract**: Parkinson's disease (PD) is a neurodegenerative disorder, manifesting with motor and non-motor symptoms. Depressive symptoms are prevalent in PD, affecting up to 45% of patients. They are often underdiagnosed due to overlapping motor features, such as hypomimia. This study explores deep learning (DL) models-ViViT, Video Swin Tiny, and 3D CNN-LSTM with attention layers-to assess the presence and severity of depressive symptoms, as detected by the Geriatric Depression Scale (GDS), in PD patients through facial video analysis. The same parameters were assessed in a secondary analysis taking into account whether patients were one hour after (ON-medication state) or 12 hours without (OFF-medication state) dopaminergic medication. Using a dataset of 1,875 videos from 178 patients, the Video Swin Tiny model achieved the highest performance, with up to 94% accuracy and 93.7% F1-score in binary classification (presence of absence of depressive symptoms), and 87.1% accuracy with an 85.4% F1-score in multiclass tasks (absence or mild or severe depressive symptoms). 

---
# From Spaceborn to Airborn: SAR Image Synthesis Using Foundation Models for Multi-Scale Adaptation 

**Authors**: Solène Debuysère, Nicolas Trouvé, Nathan Letheule, Olivier Lévêque, Elise Colin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03844)  

**Abstract**: The availability of Synthetic Aperture Radar (SAR) satellite imagery has increased considerably in recent years, with datasets commercially available. However, the acquisition of high-resolution SAR images in airborne configurations, remains costly and limited. Thus, the lack of open source, well-labeled, or easily exploitable SAR text-image datasets is a barrier to the use of existing foundation models in remote sensing applications. In this context, synthetic image generation is a promising solution to augment this scarce data, enabling a broader range of applications. Leveraging over 15 years of ONERA's extensive archival airborn data from acquisition campaigns, we created a comprehensive training dataset of 110 thousands SAR images to exploit a 3.5 billion parameters pre-trained latent diffusion model. In this work, we present a novel approach utilizing spatial conditioning techniques within a foundation model to transform satellite SAR imagery into airborne SAR representations. Additionally, we demonstrate that our pipeline is effective for bridging the realism of simulated images generated by ONERA's physics-based simulator EMPRISE. Our method explores a key application of AI in advancing SAR imaging technology. To the best of our knowledge, we are the first to introduce this approach in the literature. 

---
# CoCoB: Adaptive Collaborative Combinatorial Bandits for Online Recommendation 

**Authors**: Cairong Yan, Jinyi Han, Jin Ju, Yanting Zhang, Zijian Wang, Xuan Shao  

**Link**: [PDF](https://arxiv.org/pdf/2505.03840)  

**Abstract**: Clustering bandits have gained significant attention in recommender systems by leveraging collaborative information from neighboring users to better capture target user preferences. However, these methods often lack a clear definition of similar users and face challenges when users with unique preferences lack appropriate neighbors. In such cases, relying on divergent preferences of misidentified neighbors can degrade recommendation quality. To address these limitations, this paper proposes an adaptive Collaborative Combinatorial Bandits algorithm (CoCoB). CoCoB employs an innovative two-sided bandit architecture, applying bandit principles to both the user and item sides. The user-bandit employs an enhanced Bayesian model to explore user similarity, identifying neighbors based on a similarity probability threshold. The item-bandit treats items as arms, generating diverse recommendations informed by the user-bandit's output. CoCoB dynamically adapts, leveraging neighbor preferences when available or focusing solely on the target user otherwise. Regret analysis under a linear contextual bandit setting and experiments on three real-world datasets demonstrate CoCoB's effectiveness, achieving an average 2.4% improvement in F1 score over state-of-the-art methods. 

---
# IntelliCardiac: An Intelligent Platform for Cardiac Image Segmentation and Classification 

**Authors**: Ting Yu Tsai, An Yu, Meghana Spurthi Maadugundu, Ishrat Jahan Mohima, Umme Habiba Barsha, Mei-Hwa F. Chen, Balakrishnan Prabhakaran, Ming-Ching Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03838)  

**Abstract**: Precise and effective processing of cardiac imaging data is critical for the identification and management of the cardiovascular diseases. We introduce IntelliCardiac, a comprehensive, web-based medical image processing platform for the automatic segmentation of 4D cardiac images and disease classification, utilizing an AI model trained on the publicly accessible ACDC dataset. The system, intended for patients, cardiologists, and healthcare professionals, offers an intuitive interface and uses deep learning models to identify essential heart structures and categorize cardiac diseases. The system supports analysis of both the right and left ventricles as well as myocardium, and then classifies patient's cardiac images into five diagnostic categories: dilated cardiomyopathy, myocardial infarction, hypertrophic cardiomyopathy, right ventricular abnormality, and no disease. IntelliCardiac combines a deep learning-based segmentation model with a two-step classification pipeline. The segmentation module gains an overall accuracy of 92.6\%. The classification module, trained on characteristics taken from segmented heart structures, achieves 98\% accuracy in five categories. These results exceed the performance of the existing state-of-the-art methods that integrate both segmentation and classification models. IntelliCardiac, which supports real-time visualization, workflow integration, and AI-assisted diagnostics, has great potential as a scalable, accurate tool for clinical decision assistance in cardiac imaging and diagnosis. 

---
# Explainable Face Recognition via Improved Localization 

**Authors**: Rashik Shadman, Daqing Hou, Faraz Hussain, M G Sarwar Murshed  

**Link**: [PDF](https://arxiv.org/pdf/2505.03837)  

**Abstract**: Biometric authentication has become one of the most widely used tools in the current technological era to authenticate users and to distinguish between genuine users and imposters. Face is the most common form of biometric modality that has proven effective. Deep learning-based face recognition systems are now commonly used across different domains. However, these systems usually operate like black-box models that do not provide necessary explanations or justifications for their decisions. This is a major disadvantage because users cannot trust such artificial intelligence-based biometric systems and may not feel comfortable using them when clear explanations or justifications are not provided. This paper addresses this problem by applying an efficient method for explainable face recognition systems. We use a Class Activation Mapping (CAM)-based discriminative localization (very narrow/specific localization) technique called Scaled Directed Divergence (SDD) to visually explain the results of deep learning-based face recognition systems. We perform fine localization of the face features relevant to the deep learning model for its prediction/decision. Our experiments show that the SDD Class Activation Map (CAM) highlights the relevant face features very specifically compared to the traditional CAM and very accurately. The provided visual explanations with narrow localization of relevant features can ensure much-needed transparency and trust for deep learning-based face recognition systems. 

---
# OBD-Finder: Explainable Coarse-to-Fine Text-Centric Oracle Bone Duplicates Discovery 

**Authors**: Chongsheng Zhang, Shuwen Wu, Yingqi Chen, Matthias Aßenmacher, Christian Heumann, Yi Men, Gaojuan Fan, João Gama  

**Link**: [PDF](https://arxiv.org/pdf/2505.03836)  

**Abstract**: Oracle Bone Inscription (OBI) is the earliest systematic writing system in China, while the identification of Oracle Bone (OB) duplicates is a fundamental issue in OBI research. In this work, we design a progressive OB duplicate discovery framework that combines unsupervised low-level keypoints matching with high-level text-centric content-based matching to refine and rank the candidate OB duplicates with semantic awareness and interpretability. We compare our approach with state-of-the-art content-based image retrieval and image matching methods, showing that our approach yields comparable recall performance and the highest simplified mean reciprocal rank scores for both Top-5 and Top-15 retrieval results, and with significantly accelerated computation efficiency. We have discovered over 60 pairs of new OB duplicates in real-world deployment, which were missed by OBI researchers for decades. The models, video illustration and demonstration of this work are available at: this https URL. 

---
# The Shift Towards Preprints in AI Policy Research: A Comparative Study of Preprint Trends in the U.S., Europe, and South Korea 

**Authors**: Simon Suh, Jihyuk Bang, Ji Woo Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.03835)  

**Abstract**: The adoption of open science has quickly changed how artificial intelligence (AI) policy research is distributed globally. This study examines the regional trends in the citation of preprints, specifically focusing on the impact of two major disruptive events: the COVID-19 pandemic and the release of ChatGPT, on research dissemination patterns in the United States, Europe, and South Korea from 2015 to 2024. Using bibliometrics data from the Web of Science, this study tracks how global disruptive events influenced the adoption of preprints in AI policy research and how such shifts vary by region. By marking the timing of these disruptive events, the analysis reveals that while all regions experienced growth in preprint citations, the magnitude and trajectory of change varied significantly. The United States exhibited sharp, event-driven increases; Europe demonstrated institutional growth; and South Korea maintained consistent, linear growth in preprint adoption. These findings suggest that global disruptions may have accelerated preprint adoption, but the extent and trajectory are shaped by local research cultures, policy environments, and levels of open science maturity. This paper emphasizes the need for future AI governance strategies to consider regional variability in research dissemination and highlights opportunities for further longitudinal and comparative research to deepen our understanding of open-access adoption in AI policy development. 

---
# PointExplainer: Towards Transparent Parkinson's Disease Diagnosis 

**Authors**: Xuechao Wang, Sven Nomm, Junqing Huang, Kadri Medijainen, Aaro Toomela, Michael Ruzhansky  

**Link**: [PDF](https://arxiv.org/pdf/2505.03833)  

**Abstract**: Deep neural networks have shown potential in analyzing digitized hand-drawn signals for early diagnosis of Parkinson's disease. However, the lack of clear interpretability in existing diagnostic methods presents a challenge to clinical trust. In this paper, we propose PointExplainer, an explainable diagnostic strategy to identify hand-drawn regions that drive model diagnosis. Specifically, PointExplainer assigns discrete attribution values to hand-drawn segments, explicitly quantifying their relative contributions to the model's decision. Its key components include: (i) a diagnosis module, which encodes hand-drawn signals into 3D point clouds to represent hand-drawn trajectories, and (ii) an explanation module, which trains an interpretable surrogate model to approximate the local behavior of the black-box diagnostic model. We also introduce consistency measures to further address the issue of faithfulness in explanations. Extensive experiments on two benchmark datasets and a newly constructed dataset show that PointExplainer can provide intuitive explanations with no diagnostic performance degradation. The source code is available at this https URL. 

---
# Video Forgery Detection for Surveillance Cameras: A Review 

**Authors**: Noor B. Tayfor, Tarik A. Rashid, Shko M. Qader, Bryar A. Hassan, Mohammed H. Abdalla, Jafar Majidpour, Aram M. Ahmed, Hussein M. Ali, Aso M. Aladdin, Abdulhady A. Abdullah, Ahmed S. Shamsaldin, Haval M. Sidqi, Abdulrahman Salih, Zaher M. Yaseen, Azad A. Ameen, Janmenjoy Nayak, Mahmood Yashar Hamza  

**Link**: [PDF](https://arxiv.org/pdf/2505.03832)  

**Abstract**: The widespread availability of video recording through smartphones and digital devices has made video-based evidence more accessible than ever. Surveillance footage plays a crucial role in security, law enforcement, and judicial processes. However, with the rise of advanced video editing tools, tampering with digital recordings has become increasingly easy, raising concerns about their authenticity. Ensuring the integrity of surveillance videos is essential, as manipulated footage can lead to misinformation and undermine judicial decisions. This paper provides a comprehensive review of existing forensic techniques used to detect video forgery, focusing on their effectiveness in verifying the authenticity of surveillance recordings. Various methods, including compression-based analysis, frame duplication detection, and machine learning-based approaches, are explored. The findings highlight the growing necessity for more robust forensic techniques to counteract evolving forgery methods. Strengthening video forensic capabilities will ensure that surveillance recordings remain credible and admissible as legal evidence. 

---
# VideoLLM Benchmarks and Evaluation: A Survey 

**Authors**: Yogesh Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03829)  

**Abstract**: The rapid development of Large Language Models (LLMs) has catalyzed significant advancements in video understanding technologies. This survey provides a comprehensive analysis of benchmarks and evaluation methodologies specifically designed or used for Video Large Language Models (VideoLLMs). We examine the current landscape of video understanding benchmarks, discussing their characteristics, evaluation protocols, and limitations. The paper analyzes various evaluation methodologies, including closed-set, open-set, and specialized evaluations for temporal and spatiotemporal understanding tasks. We highlight the performance trends of state-of-the-art VideoLLMs across these benchmarks and identify key challenges in current evaluation frameworks. Additionally, we propose future research directions to enhance benchmark design, evaluation metrics, and protocols, including the need for more diverse, multimodal, and interpretability-focused benchmarks. This survey aims to equip researchers with a structured understanding of how to effectively evaluate VideoLLMs and identify promising avenues for advancing the field of video understanding with large language models. 

---
# Sentiment-Aware Recommendation Systems in E-Commerce: A Review from a Natural Language Processing Perspective 

**Authors**: Yogesh Gajula  

**Link**: [PDF](https://arxiv.org/pdf/2505.03828)  

**Abstract**: E-commerce platforms generate vast volumes of user feedback, such as star ratings, written reviews, and comments. However, most recommendation engines rely primarily on numerical scores, often overlooking the nuanced opinions embedded in free text. This paper comprehensively reviews sentiment-aware recommendation systems from a natural language processing perspective, covering advancements from 2023 to early 2025. It highlights the benefits of integrating sentiment analysis into e-commerce recommenders to enhance prediction accuracy and explainability through detailed opinion extraction. Our survey categorizes recent work into four main approaches: deep learning classifiers that combine sentiment embeddings with user item interactions, transformer based methods for nuanced feature extraction, graph neural networks that propagate sentiment signals, and conversational recommenders that adapt in real time to user feedback. We summarize model architectures and demonstrate how sentiment flows through recommendation pipelines, impacting dialogue-based suggestions. Key challenges include handling noisy or sarcastic text, dynamic user preferences, and bias mitigation. Finally, we outline research gaps and provide a roadmap for developing smarter, fairer, and more user-centric recommendation tools. 

---
# MISE: Meta-knowledge Inheritance for Social Media-Based Stressor Estimation 

**Authors**: Xin Wang, Ling Feng, Huijun Zhang, Lei Cao, Kaisheng Zeng, Qi Li, Yang Ding, Yi Dai, David Clifton  

**Link**: [PDF](https://arxiv.org/pdf/2505.03827)  

**Abstract**: Stress haunts people in modern society, which may cause severe health issues if left unattended. With social media becoming an integral part of daily life, leveraging social media to detect stress has gained increasing attention. While the majority of the work focuses on classifying stress states and stress categories, this study introduce a new task aimed at estimating more specific stressors (like exam, writing paper, etc.) through users' posts on social media. Unfortunately, the diversity of stressors with many different classes but a few examples per class, combined with the consistent arising of new stressors over time, hinders the machine understanding of stressors. To this end, we cast the stressor estimation problem within a practical scenario few-shot learning setting, and propose a novel meta-learning based stressor estimation framework that is enhanced by a meta-knowledge inheritance mechanism. This model can not only learn generic stressor context through meta-learning, but also has a good generalization ability to estimate new stressors with little labeled data. A fundamental breakthrough in our approach lies in the inclusion of the meta-knowledge inheritance mechanism, which equips our model with the ability to prevent catastrophic forgetting when adapting to new stressors. The experimental results show that our model achieves state-of-the-art performance compared with the baselines. Additionally, we construct a social media-based stressor estimation dataset that can help train artificial intelligence models to facilitate human well-being. The dataset is now public at \href{this https URL}{\underline{Kaggle}} and \href{this https URL}{\underline{Hugging Face}}. 

---
# In-situ and Non-contact Etch Depth Prediction in Plasma Etching via Machine Learning (ANN & BNN) and Digital Image Colorimetry 

**Authors**: Minji Kang, Seongho Kim, Eunseo Go, Donghyeon Paek, Geon Lim, Muyoung Kim, Soyeun Kim, Sung Kyu Jang, Min Sup Choi, Woo Seok Kang, Jaehyun Kim, Jaekwang Kim, Hyeong-U Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.03826)  

**Abstract**: Precise monitoring of etch depth and the thickness of insulating materials, such as Silicon dioxide and silicon nitride, is critical to ensuring device performance and yield in semiconductor manufacturing. While conventional ex-situ analysis methods are accurate, they are constrained by time delays and contamination risks. To address these limitations, this study proposes a non-contact, in-situ etch depth prediction framework based on machine learning (ML) techniques. Two scenarios are explored. In the first scenario, an artificial neural network (ANN) is trained to predict average etch depth from process parameters, achieving a significantly lower mean squared error (MSE) compared to a linear baseline model. The approach is then extended to incorporate variability from repeated measurements using a Bayesian Neural Network (BNN) to capture both aleatoric and epistemic uncertainty. Coverage analysis confirms the BNN's capability to provide reliable uncertainty estimates. In the second scenario, we demonstrate the feasibility of using RGB data from digital image colorimetry (DIC) as input for etch depth prediction, achieving strong performance even in the absence of explicit process parameters. These results suggest that the integration of DIC and ML offers a viable, cost-effective alternative for real-time, in-situ, and non-invasive monitoring in plasma etching processes, contributing to enhanced process stability, and manufacturing efficiency. 

---
# Intelligently Augmented Contrastive Tensor Factorization: Empowering Multi-dimensional Time Series Classification in Low-Data Environments 

**Authors**: Anushiya Arunan, Yan Qin, Xiaoli Li, Yuen Chau  

**Link**: [PDF](https://arxiv.org/pdf/2505.03825)  

**Abstract**: Classification of multi-dimensional time series from real-world systems require fine-grained learning of complex features such as cross-dimensional dependencies and intra-class variations-all under the practical challenge of low training data availability. However, standard deep learning (DL) struggles to learn generalizable features in low-data environments due to model overfitting. We propose a versatile yet data-efficient framework, Intelligently Augmented Contrastive Tensor Factorization (ITA-CTF), to learn effective representations from multi-dimensional time series. The CTF module learns core explanatory components of the time series (e.g., sensor factors, temporal factors), and importantly, their joint dependencies. Notably, unlike standard tensor factorization (TF), the CTF module incorporates a new contrastive loss optimization to induce similarity learning and class-awareness into the learnt representations for better classification performance. To strengthen this contrastive learning, the preceding ITA module generates targeted but informative augmentations that highlight realistic intra-class patterns in the original data, while preserving class-wise properties. This is achieved by dynamically sampling a "soft" class prototype to guide the warping of each query data sample, which results in an augmentation that is intelligently pattern-mixed between the "soft" class prototype and the query sample. These augmentations enable the CTF module to recognize complex intra-class variations despite the limited original training data, and seek out invariant class-wise properties for accurate classification performance. The proposed method is comprehensively evaluated on five different classification tasks. Compared to standard TF and several DL benchmarks, notable performance improvements up to 18.7% were achieved. 

---
# Memory Assisted LLM for Personalized Recommendation System 

**Authors**: Jiarui Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.03824)  

**Abstract**: Large language models (LLMs) have demonstrated significant potential in solving recommendation tasks. With proven capabilities in understanding user preferences, LLM personalization has emerged as a critical area for providing tailored responses to individuals. Current studies explore personalization through prompt design and fine-tuning, paving the way for further research in personalized LLMs. However, existing approaches are either costly and inefficient in capturing diverse user preferences or fail to account for timely updates to user history. To address these gaps, we propose the Memory-Assisted Personalized LLM (MAP). Through user interactions, we first create a history profile for each user, capturing their preferences, such as ratings for historical items. During recommendation, we extract relevant memory based on similarity, which is then incorporated into the prompts to enhance personalized recommendations. In our experiments, we evaluate MAP using a sequential rating prediction task under two scenarios: single domain, where memory and tasks are from the same category (e.g., movies), and cross-domain (e.g., memory from movies and recommendation tasks in books). The results show that MAP outperforms regular LLM-based recommenders that integrate user history directly through prompt design. Moreover, as user history grows, MAP's advantage increases in both scenarios, making it more suitable for addressing successive personalized user requests. 

---
# DRSLF: Double Regularized Second-Order Low-Rank Representation for Web Service QoS Prediction 

**Authors**: Hao Wu, Jialiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03822)  

**Abstract**: Quality-of-Service (QoS) data plays a crucial role in cloud service selection. Since users cannot access all services, QoS can be represented by a high-dimensional and incomplete (HDI) matrix. Latent factor analysis (LFA) models have been proven effective as low-rank representation techniques for addressing this issue. However, most LFA models rely on first-order optimizers and use L2-norm regularization, which can lead to lower QoS prediction accuracy. To address this issue, this paper proposes a double regularized second-order latent factor (DRSLF) model with two key ideas: a) integrating L1-norm and L2-norm regularization terms to enhance the low-rank representation performance; b) incorporating second-order information by calculating the Hessian-vector product in each conjugate gradient step. Experimental results on two real-world response-time QoS datasets demonstrate that DRSLF has a higher low-rank representation capability than two baselines. 

---
# Beyond Recognition: Evaluating Visual Perspective Taking in Vision Language Models 

**Authors**: Gracjan Góral, Alicja Ziarko, Piotr Miłoś, Michał Nauman, Maciej Wołczyk, Michał Kosiński  

**Link**: [PDF](https://arxiv.org/pdf/2505.03821)  

**Abstract**: We investigate the ability of Vision Language Models (VLMs) to perform visual perspective taking using a novel set of visual tasks inspired by established human tests. Our approach leverages carefully controlled scenes, in which a single humanoid minifigure is paired with a single object. By systematically varying spatial configurations - such as object position relative to the humanoid minifigure and the humanoid minifigure's orientation - and using both bird's-eye and surface-level views, we created 144 unique visual tasks. Each visual task is paired with a series of 7 diagnostic questions designed to assess three levels of visual cognition: scene understanding, spatial reasoning, and visual perspective taking. Our evaluation of several state-of-the-art models, including GPT-4-Turbo, GPT-4o, Llama-3.2-11B-Vision-Instruct, and variants of Claude Sonnet, reveals that while they excel in scene understanding, the performance declines significantly on spatial reasoning and further deteriorates on perspective-taking. Our analysis suggests a gap between surface-level object recognition and the deeper spatial and perspective reasoning required for complex visual tasks, pointing to the need for integrating explicit geometric representations and tailored training protocols in future VLM development. 

---
# Focus on the Likely: Test-time Instance-based Uncertainty Removal 

**Authors**: Johannes Schneider  

**Link**: [PDF](https://arxiv.org/pdf/2505.03819)  

**Abstract**: We propose two novel test-time fine-tuning methods to improve uncertain model predictions. Our methods require no auxiliary data and use the given test instance only. Instead of performing a greedy selection of the most likely class to make a prediction, we introduce an additional focus on the likely classes step during inference. By applying a single-step gradient descent, we refine predictions when an initial forward pass indicates high uncertainty. This aligns predictions more closely with the ideal of assigning zero probability to less plausible outcomes. Our theoretical discussion provides a deeper understanding highlighting the impact on shared and non-shared features among (focus) classes. The experimental evaluation highlights accuracy gains on samples exhibiting high decision uncertainty for a diverse set of models from both the text and image domain using the same hyperparameters. 

---
# Program Semantic Inequivalence Game with Large Language Models 

**Authors**: Antonio Valerio Miceli-Barone, Vaishak Belle, Ali Payani  

**Link**: [PDF](https://arxiv.org/pdf/2505.03818)  

**Abstract**: Large Language Models (LLMs) can achieve strong performance on everyday coding tasks, but they can fail on complex tasks that require non-trivial reasoning about program semantics. Finding training examples to teach LLMs to solve these tasks can be challenging.
In this work, we explore a method to synthetically generate code reasoning training data based on a semantic inequivalence game SInQ: a generator agent creates program variants that are semantically distinct, derived from a dataset of real-world programming tasks, while an evaluator agent has to identify input examples that cause the original programs and the generated variants to diverge in their behaviour, with the agents training each other semi-adversarially. We prove that this setup enables theoretically unlimited improvement through self-play in the limit of infinite computational resources.
We evaluated our approach on multiple code generation and understanding benchmarks, including cross-language vulnerability detection (Lu et al., 2021), where our method improves vulnerability detection in C/C++ code despite being trained exclusively on Python code, and the challenging Python builtin identifier swap benchmark (Miceli-Barone et al., 2023), showing that whereas modern LLMs still struggle with this benchmark, our approach yields substantial improvements.
We release the code needed to replicate the experiments, as well as the generated synthetic data, which can be used to fine-tune LLMs. 

---
# Modeling Behavioral Preferences of Cyber Adversaries Using Inverse Reinforcement Learning 

**Authors**: Aditya Shinde, Prashant Doshi  

**Link**: [PDF](https://arxiv.org/pdf/2505.03817)  

**Abstract**: This paper presents a holistic approach to attacker preference modeling from system-level audit logs using inverse reinforcement learning (IRL). Adversary modeling is an important capability in cybersecurity that lets defenders characterize behaviors of potential attackers, which enables attribution to known cyber adversary groups. Existing approaches rely on documenting an ever-evolving set of attacker tools and techniques to track known threat actors. Although attacks evolve constantly, attacker behavioral preferences are intrinsic and less volatile. Our approach learns the behavioral preferences of cyber adversaries from forensics data on their tools and techniques. We model the attacker as an expert decision-making agent with unknown behavioral preferences situated in a computer host. We leverage attack provenance graphs of audit logs to derive a state-action trajectory of the attack. We test our approach on open datasets of audit logs containing real attack data. Our results demonstrate for the first time that low-level forensics data can automatically reveal an adversary's subjective preferences, which serves as an additional dimension to modeling and documenting cyber adversaries. Attackers' preferences tend to be invariant despite their different tools and indicate predispositions that are inherent to the attacker. As such, these inferred preferences can potentially serve as unique behavioral signatures of attackers and improve threat attribution. 

---
# Geospatial and Temporal Trends in Urban Transportation: A Study of NYC Taxis and Pathao Food Deliveries 

**Authors**: Bidyarthi Paul, Fariha Tasnim Chowdhury, Dipta Biswas, Meherin Sultana  

**Link**: [PDF](https://arxiv.org/pdf/2505.03816)  

**Abstract**: Urban transportation plays a vital role in modern city life, affecting how efficiently people and goods move around. This study analyzes transportation patterns using two datasets: the NYC Taxi Trip dataset from New York City and the Pathao Food Trip dataset from Dhaka, Bangladesh. Our goal is to identify key trends in demand, peak times, and important geographical hotspots. We start with Exploratory Data Analysis (EDA) to understand the basic characteristics of the datasets. Next, we perform geospatial analysis to map out high-demand and low-demand regions. We use the SARIMAX model for time series analysis to forecast demand patterns, capturing seasonal and weekly variations. Lastly, we apply clustering techniques to identify significant areas of high and low demand. Our findings provide valuable insights for optimizing fleet management and resource allocation in both passenger transport and food delivery services. These insights can help improve service efficiency, better meet customer needs, and enhance urban transportation systems in diverse urban environments. 

---
# Cer-Eval: Certifiable and Cost-Efficient Evaluation Framework for LLMs 

**Authors**: Ganghua Wang, Zhaorun Chen, Bo Li, Haifeng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03814)  

**Abstract**: As foundation models continue to scale, the size of trained models grows exponentially, presenting significant challenges for their evaluation. Current evaluation practices involve curating increasingly large datasets to assess the performance of large language models (LLMs). However, there is a lack of systematic analysis and guidance on determining the sufficiency of test data or selecting informative samples for evaluation. This paper introduces a certifiable and cost-efficient evaluation framework for LLMs. Our framework adapts to different evaluation objectives and outputs confidence intervals that contain true values with high probability. We use ``test sample complexity'' to quantify the number of test points needed for a certifiable evaluation and derive tight bounds on test sample complexity. Based on the developed theory, we develop a partition-based algorithm, named Cer-Eval, that adaptively selects test points to minimize the cost of LLM evaluation. Real-world experiments demonstrate that Cer-Eval can save 20% to 40% test points across various benchmarks, while maintaining an estimation error level comparable to the current evaluation process and providing a 95% confidence guarantee. 

---
# ScarceGAN: Discriminative Classification Framework for Rare Class Identification for Longitudinal Data with Weak Prior 

**Authors**: Surajit Chakrabarty, Rukma Talwadker, Tridib Mukherjee  

**Link**: [PDF](https://arxiv.org/pdf/2505.03811)  

**Abstract**: This paper introduces ScarceGAN which focuses on identification of extremely rare or scarce samples from multi-dimensional longitudinal telemetry data with small and weak label prior. We specifically address: (i) severe scarcity in positive class, stemming from both underlying organic skew in the data, as well as extremely limited labels; (ii) multi-class nature of the negative samples, with uneven density distributions and partially overlapping feature distributions; and (iii) massively unlabelled data leading to tiny and weak prior on both positive and negative classes, and possibility of unseen or unknown behavior in the unlabelled set, especially in the negative class. Although related to PU learning problems, we contend that knowledge (or lack of it) on the negative class can be leveraged to learn the compliment of it (i.e., the positive class) better in a semi-supervised manner. To this effect, ScarceGAN re-formulates semi-supervised GAN by accommodating weakly labelled multi-class negative samples and the available positive samples. It relaxes the supervised discriminator's constraint on exact differentiation between negative samples by introducing a 'leeway' term for samples with noisy prior. We propose modifications to the cost objectives of discriminator, in supervised and unsupervised path as well as that of the generator. For identifying risky players in skill gaming, this formulation in whole gives us a recall of over 85% (~60% jump over vanilla semi-supervised GAN) on our scarce class with very minimal verbosity in the unknown space. Further ScarceGAN outperforms the recall benchmarks established by recent GAN based specialized models for the positive imbalanced class identification and establishes a new benchmark in identifying one of rare attack classes (0.09%) in the intrusion dataset from the KDDCUP99 challenge. 

---
# Grouped Sequency-arranged Rotation: Optimizing Rotation Transformation for Quantization for Free 

**Authors**: Euntae Choi, Sumin Song, Woosang Lim, Sungjoo Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03810)  

**Abstract**: Large Language Models (LLMs) face deployment challenges due to high computational costs, and while Post-Training Quantization (PTQ) offers a solution, existing rotation-based methods struggle at very low bit-widths like 2-bit. We introduce a novel, training-free approach to construct an improved rotation matrix, addressing the limitations of current methods. The key contributions include leveraging the Walsh-Hadamard transform with sequency ordering, which clusters similar frequency components to reduce quantization error compared to standard Hadamard matrices, significantly improving performance. Furthermore, we propose a Grouped Sequency-arranged Rotation (GSR) using block-diagonal matrices with smaller Walsh blocks, effectively isolating outlier impacts and achieving performance comparable to optimization-based methods without requiring any training. Our method demonstrates robust performance on reasoning tasks and Perplexity (PPL) score on WikiText-2. Our method also enhances results even when applied over existing learned rotation techniques. 

---
# When Dynamic Data Selection Meets Data Augmentation 

**Authors**: Suorong Yang, Peng Ye, Furao Shen, Dongzhan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.03809)  

**Abstract**: Dynamic data selection aims to accelerate training with lossless performance. However, reducing training data inherently limits data diversity, potentially hindering generalization. While data augmentation is widely used to enhance diversity, it is typically not optimized in conjunction with selection. As a result, directly combining these techniques fails to fully exploit their synergies. To tackle the challenge, we propose a novel online data training framework that, for the first time, unifies dynamic data selection and augmentation, achieving both training efficiency and enhanced performance. Our method estimates each sample's joint distribution of local density and multimodal semantic consistency, allowing for the targeted selection of augmentation-suitable samples while suppressing the inclusion of noisy or ambiguous data. This enables a more significant reduction in dataset size without sacrificing model generalization. Experimental results demonstrate that our method outperforms existing state-of-the-art approaches on various benchmark datasets and architectures, e.g., reducing 50\% training costs on ImageNet-1k with lossless performance. Furthermore, our approach enhances noise resistance and improves model robustness, reinforcing its practical utility in real-world scenarios. 

---
# Facilitating Video Story Interaction with Multi-Agent Collaborative System 

**Authors**: Yiwen Zhang, Jianing Hao, Zhan Wang, Hongling Sheng, Wei Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2505.03807)  

**Abstract**: Video story interaction enables viewers to engage with and explore narrative content for personalized experiences. However, existing methods are limited to user selection, specially designed narratives, and lack customization. To address this, we propose an interactive system based on user intent. Our system uses a Vision Language Model (VLM) to enable machines to understand video stories, combining Retrieval-Augmented Generation (RAG) and a Multi-Agent System (MAS) to create evolving characters and scene experiences. It includes three stages: 1) Video story processing, utilizing VLM and prior knowledge to simulate human understanding of stories across three modalities. 2) Multi-space chat, creating growth-oriented characters through MAS interactions based on user queries and story stages. 3) Scene customization, expanding and visualizing various story scenes mentioned in dialogue. Applied to the Harry Potter series, our study shows the system effectively portrays emergent character social behavior and growth, enhancing the interactive experience in the video story world. 

---
# Perception-Informed Neural Networks: Beyond Physics-Informed Neural Networks 

**Authors**: Mehran Mazandarani, Marzieh Najariyan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03806)  

**Abstract**: This article introduces Perception-Informed Neural Networks (PrINNs), a framework designed to incorporate perception-based information into neural networks, addressing both systems with known and unknown physics laws or differential equations. Moreover, PrINNs extend the concept of Physics-Informed Neural Networks (PINNs) and their variants, offering a platform for the integration of diverse forms of perception precisiation, including singular, probability distribution, possibility distribution, interval, and fuzzy graph. In fact, PrINNs allow neural networks to model dynamical systems by integrating expert knowledge and perception-based information through loss functions, enabling the creation of modern data-driven models. Some of the key contributions include Mixture of Experts Informed Neural Networks (MOEINNs), which combine heterogeneous expert knowledge into the network, and Transformed-Knowledge Informed Neural Networks (TKINNs), which facilitate the incorporation of meta-information for enhanced model performance. Additionally, Fuzzy-Informed Neural Networks (FINNs) as a modern class of fuzzy deep neural networks leverage fuzzy logic constraints within a deep learning architecture, allowing online training without pre-training and eliminating the need for defuzzification. PrINNs represent a significant step forward in bridging the gap between traditional physics-based modeling and modern data-driven approaches, enabling neural networks to learn from both structured physics laws and flexible perception-based rules. This approach empowers neural networks to operate in uncertain environments, model complex systems, and discover new forms of differential equations, making PrINNs a powerful tool for advancing computational science and engineering. 

---
# MoEQuant: Enhancing Quantization for Mixture-of-Experts Large Language Models via Expert-Balanced Sampling and Affinity Guidance 

**Authors**: Xing Hu, Zhixuan Chen, Dawei Yang, Zukang Xu, Chen Xu, Zhihang Yuan, Sifan Zhou, Jiangyong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.03804)  

**Abstract**: Mixture-of-Experts (MoE) large language models (LLMs), which leverage dynamic routing and sparse activation to enhance efficiency and scalability, have achieved higher performance while reducing computational costs. However, these models face significant memory overheads, limiting their practical deployment and broader adoption. Post-training quantization (PTQ), a widely used method for compressing LLMs, encounters severe accuracy degradation and diminished generalization performance when applied to MoE models. This paper investigates the impact of MoE's sparse and dynamic characteristics on quantization and identifies two primary challenges: (1) Inter-expert imbalance, referring to the uneven distribution of samples across experts, which leads to insufficient and biased calibration for less frequently utilized experts; (2) Intra-expert imbalance, arising from MoE's unique aggregation mechanism, which leads to varying degrees of correlation between different samples and their assigned experts. To address these challenges, we propose MoEQuant, a novel quantization framework tailored for MoE LLMs. MoE-Quant includes two novel techniques: 1) Expert-Balanced Self-Sampling (EBSS) is an efficient sampling method that efficiently constructs a calibration set with balanced expert distributions by leveraging the cumulative probabilities of tokens and expert balance metrics as guiding factors. 2) Affinity-Guided Quantization (AGQ), which incorporates affinities between experts and samples into the quantization process, thereby accurately assessing the impact of individual samples on different experts within the MoE layer. Experiments demonstrate that MoEQuant achieves substantial performance gains (more than 10 points accuracy gain in the HumanEval for DeepSeekMoE-16B under 4-bit quantization) and boosts efficiency. 

---
# RWKVQuant: Quantizing the RWKV Family with Proxy Guided Hybrid of Scalar and Vector Quantization 

**Authors**: Chen Xu, Yuxuan Yue, Zukang Xu, Xing Hu, Jiangyong Yu, Zhixuan Chen, Sifan Zhou, Zhihang Yuan, Dawei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03803)  

**Abstract**: RWKV is a modern RNN architecture with comparable performance to Transformer, but still faces challenges when deployed to resource-constrained devices. Post Training Quantization (PTQ), which is a an essential technique to reduce model size and inference latency, has been widely used in Transformer models. However, it suffers significant degradation of performance when applied to RWKV. This paper investigates and identifies two key constraints inherent in the properties of RWKV: (1) Non-linear operators hinder the parameter-fusion of both smooth- and rotation-based quantization, introducing extra computation overhead. (2) The larger amount of uniformly distributed weights poses challenges for cluster-based quantization, leading to reduced accuracy. To this end, we propose RWKVQuant, a PTQ framework tailored for RWKV models, consisting of two novel techniques: (1) a coarse-to-fine proxy capable of adaptively selecting different quantization approaches by assessing the uniformity and identifying outliers in the weights, and (2) a codebook optimization algorithm that enhances the performance of cluster-based quantization methods for element-wise multiplication in RWKV. Experiments show that RWKVQuant can quantize RWKV-6-14B into about 3-bit with less than 1% accuracy loss and 2.14x speed up. 

---
# Efficient Fine-Tuning of Quantized Models via Adaptive Rank and Bitwidth 

**Authors**: Changhai Zhou, Yuhua Zhou, Qian Qiao, Weizhong Zhang, Cheng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03802)  

**Abstract**: QLoRA effectively combines low-bit quantization and LoRA to achieve memory-friendly fine-tuning for large language models (LLM). Recently, methods based on SVD for continuous update iterations to initialize LoRA matrices to accommodate quantization errors have generally failed to consistently improve performance. Dynamic mixed precision is a natural idea for continuously improving the fine-tuning performance of quantized models, but previous methods often optimize low-rank subspaces or quantization components separately, without considering their synergy. To address this, we propose \textbf{QR-Adaptor}, a unified, gradient-free strategy that uses partial calibration data to jointly search the quantization components and the rank of low-rank spaces for each layer, thereby continuously improving model performance. QR-Adaptor does not minimize quantization error but treats precision and rank allocation as a discrete optimization problem guided by actual downstream performance and memory usage. Compared to state-of-the-art (SOTA) quantized LoRA fine-tuning methods, our approach achieves a 4.89\% accuracy improvement on GSM8K, and in some cases even outperforms the 16-bit fine-tuned model while maintaining the memory footprint of the 4-bit setting. 

---
# Large Language Model Compression with Global Rank and Sparsity Optimization 

**Authors**: Changhai Zhou, Qian Qiao, Weizhong Zhang, Cheng Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.03801)  

**Abstract**: Low-rank and sparse composite approximation is a natural idea to compress Large Language Models (LLMs). However, such an idea faces two primary challenges that adversely affect the performance of existing methods. The first challenge relates to the interaction and cooperation between low-rank and sparse matrices, while the second involves determining weight allocation across different layers, as redundancy varies considerably among them. To address these challenges, we propose a novel two-stage LLM compression method with the capability of global rank and sparsity optimization. It is noteworthy that the overall optimization space is vast, making comprehensive optimization computationally prohibitive. Therefore, to reduce the optimization space, our first stage utilizes robust principal component analysis to decompose the weight matrices of LLMs into low-rank and sparse components, which span the low dimensional and sparse spaces containing the resultant low-rank and sparse matrices, respectively. In the second stage, we propose a probabilistic global optimization technique to jointly identify the low-rank and sparse structures within the above two spaces. The appealing feature of our approach is its ability to automatically detect the redundancy across different layers and to manage the interaction between the sparse and low-rank components. Extensive experimental results indicate that our method significantly surpasses state-of-the-art techniques for sparsification and composite approximation. 

---
# Scalability Matters: Overcoming Challenges in InstructGLM with Similarity-Degree-Based Sampling 

**Authors**: Hyun Lee, Chris Yi, Maminur Islam, B.D.S. Aritra  

**Link**: [PDF](https://arxiv.org/pdf/2505.03799)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong capabilities in various natural language processing tasks; however, their application to graph-related problems remains limited, primarily due to scalability constraints and the absence of dedicated mechanisms for processing graph structures. Existing approaches predominantly integrate LLMs with Graph Neural Networks (GNNs), using GNNs as feature encoders or auxiliary components. However, directly encoding graph structures within LLMs has been underexplored, particularly in the context of large-scale graphs where token limitations hinder effective representation. To address these challenges, we propose SDM-InstructGLM, a novel instruction-tuned Graph Language Model (InstructGLM) framework that enhances scalability and efficiency without relying on GNNs. Our method introduces a similarity-degree-based biased random walk mechanism, which selectively samples and encodes graph information based on node-feature similarity and degree centrality, ensuring an adaptive and structured representation within the LLM. This approach significantly improves token efficiency, mitigates information loss due to random sampling, and enhances performance on graph-based tasks such as node classification and link prediction. Furthermore, our results demonstrate the feasibility of LLM-only graph processing, enabling scalable and interpretable Graph Language Models (GLMs) optimized through instruction-based fine-tuning. This work paves the way for GNN-free approaches to graph learning, leveraging LLMs as standalone graph reasoning models. Our source code is available on GitHub. 

---
# Position: Foundation Models Need Digital Twin Representations 

**Authors**: Yiqing Shen, Hao Ding, Lalithkumar Seenivasan, Tianmin Shu, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2505.03798)  

**Abstract**: Current foundation models (FMs) rely on token representations that directly fragment continuous real-world multimodal data into discrete tokens. They limit FMs to learning real-world knowledge and relationships purely through statistical correlation rather than leveraging explicit domain knowledge. Consequently, current FMs struggle with maintaining semantic coherence across modalities, capturing fine-grained spatial-temporal dynamics, and performing causal reasoning. These limitations cannot be overcome by simply scaling up model size or expanding datasets. This position paper argues that the machine learning community should consider digital twin (DT) representations, which are outcome-driven digital representations that serve as building blocks for creating virtual replicas of physical processes, as an alternative to the token representation for building FMs. Finally, we discuss how DT representations can address these challenges by providing physically grounded representations that explicitly encode domain knowledge and preserve the continuous nature of real-world processes. 

---
# AI-Driven IRM: Transforming insider risk management with adaptive scoring and LLM-based threat detection 

**Authors**: Lokesh Koli, Shubham Kalra, Rohan Thakur, Anas Saifi, Karanpreet Singh  

**Link**: [PDF](https://arxiv.org/pdf/2505.03796)  

**Abstract**: Insider threats pose a significant challenge to organizational security, often evading traditional rule-based detection systems due to their subtlety and contextual nature. This paper presents an AI-powered Insider Risk Management (IRM) system that integrates behavioral analytics, dynamic risk scoring, and real-time policy enforcement to detect and mitigate insider threats with high accuracy and adaptability. We introduce a hybrid scoring mechanism - transitioning from the static PRISM model to an adaptive AI-based model utilizing an autoencoder neural network trained on expert-annotated user activity data. Through iterative feedback loops and continuous learning, the system reduces false positives by 59% and improves true positive detection rates by 30%, demonstrating substantial gains in detection precision. Additionally, the platform scales efficiently, processing up to 10 million log events daily with sub-300ms query latency, and supports automated enforcement actions for policy violations, reducing manual intervention. The IRM system's deployment resulted in a 47% reduction in incident response times, highlighting its operational impact. Future enhancements include integrating explainable AI, federated learning, graph-based anomaly detection, and alignment with Zero Trust principles to further elevate its adaptability, transparency, and compliance-readiness. This work establishes a scalable and proactive framework for mitigating emerging insider risks in both on-premises and hybrid environments. 

---
# Modeling Human Behavior in a Strategic Network Game with Complex Group Dynamics 

**Authors**: Jacob W. Crandall, Jonathan Skaggs  

**Link**: [PDF](https://arxiv.org/pdf/2505.03795)  

**Abstract**: Human networks greatly impact important societal outcomes, including wealth and health inequality, poverty, and bullying. As such, understanding human networks is critical to learning how to promote favorable societal outcomes. As a step toward better understanding human networks, we compare and contrast several methods for learning models of human behavior in a strategic network game called the Junior High Game (JHG). These modeling methods differ with respect to the assumptions they use to parameterize human behavior (behavior vs. community-aware behavior) and the statistical moments they model (mean vs. distribution). Results show that the highest-performing method models the population's distribution rather than the mean and assumes humans use community-aware behavior rather than behavior matching. When applied to small societies (6-11 individuals), this learned model, called hCAB, closely mirrors the population dynamics of human groups (with some differences). Additionally, a user study reveals that human participants were unable to distinguish hCAB agents from other humans, thus illustrating that individual hCAB behavior plausibly mirrors human behavior in this strategic network game. 

---
# LENSLLM: Unveiling Fine-Tuning Dynamics for LLM Selection 

**Authors**: Xinyue Zeng, Haohui Wang, Junhong Lin, Jun Wu, Tyler Cody, Dawei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.03793)  

**Abstract**: The proliferation of open-sourced Large Language Models (LLMs) and diverse downstream tasks necessitates efficient model selection, given the impracticality of fine-tuning all candidates due to computational constraints. Despite the recent advances in LLM selection, a fundamental research question largely remains nascent: how can we model the dynamic behaviors of LLMs during fine-tuning, thereby enhancing our understanding of their generalization performance across diverse downstream tasks? In this work, we propose a novel theoretical framework that provides a proper lens to assess the generalization capabilities of LLMs, thereby enabling accurate and efficient LLM selection for downstream applications. In particular, we first derive a Hessian-based PAC-Bayes generalization bound that unveils fine-tuning dynamics of LLMs and then introduce LENSLLM, a Neural Tangent Kernel(NTK)-based Rectified Scaling Model that enables accurate performance predictions across diverse tasks while maintaining computational efficiency. Extensive empirical results on 3 large-scale benchmarks demonstrate that our model achieves up to 91.1% accuracy and reduces up to 88.5% computational cost in LLM selection, outperforming 5 state-of-the-art methods. We open-source our proposed LENSLLM model and corresponding results at the Github link: this https URL. 

---
# Towards Efficient Online Tuning of VLM Agents via Counterfactual Soft Reinforcement Learning 

**Authors**: Lang Feng, Weihao Tan, Zhiyi Lyu, Longtao Zheng, Haiyang Xu, Ming Yan, Fei Huang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2505.03792)  

**Abstract**: Online fine-tuning vision-language model (VLM) agents with reinforcement learning (RL) has shown promise for equipping agents with multi-step, goal-oriented capabilities in dynamic environments. However, their open-ended textual action space and non-end-to-end nature of action generation present significant challenges to effective online exploration in RL, e.g., explosion of the exploration space. We propose a novel online fine-tuning method, Counterfactual Soft Reinforcement Learning (CoSo), better suited to the textual output space of VLM agents. Compared to prior methods that assign uniform uncertainty to all tokens, CoSo leverages counterfactual reasoning to dynamically assess the causal influence of individual tokens on post-processed actions. By prioritizing the exploration of action-critical tokens while reducing the impact of semantically redundant or low-impact tokens, CoSo enables a more targeted and efficient online rollout process. We provide theoretical analysis proving CoSo's convergence and policy improvement guarantees, and extensive empirical evaluations supporting CoSo's effectiveness. Our results across a diverse set of agent tasks, including Android device control, card gaming, and embodied AI, highlight its remarkable ability to enhance exploration efficiency and deliver consistent performance gains. The code is available at this https URL. 

---
# Practical Boolean Backpropagation 

**Authors**: Simon Golbert  

**Link**: [PDF](https://arxiv.org/pdf/2505.03791)  

**Abstract**: Boolean neural networks offer hardware-efficient alternatives to real-valued models. While quantization is common, purely Boolean training remains underexplored. We present a practical method for purely Boolean backpropagation for networks based on a single specific gate we chose, operating directly in Boolean algebra involving no numerics. Initial experiments confirm its feasibility. 

---
# A Time-Series Data Augmentation Model through Diffusion and Transformer Integration 

**Authors**: Yuren Zhang, Zhongnan Pu, Lei Jing  

**Link**: [PDF](https://arxiv.org/pdf/2505.03790)  

**Abstract**: With the development of Artificial Intelligence, numerous real-world tasks have been accomplished using technology integrated with deep learning. To achieve optimal performance, deep neural networks typically require large volumes of data for training. Although advances in data augmentation have facilitated the acquisition of vast datasets, most of this data is concentrated in domains like images and speech. However, there has been relatively less focus on augmenting time-series data. To address this gap and generate a substantial amount of time-series data, we propose a simple and effective method that combines the Diffusion and Transformer models. By utilizing an adjusted diffusion denoising model to generate a large volume of initial time-step action data, followed by employing a Transformer model to predict subsequent actions, and incorporating a weighted loss function to achieve convergence, the method demonstrates its effectiveness. Using the performance improvement of the model after applying augmented data as a benchmark, and comparing the results with those obtained without data augmentation or using traditional data augmentation methods, this approach shows its capability to produce high-quality augmented data. 

---
# Calibrating Uncertainty Quantification of Multi-Modal LLMs using Grounding 

**Authors**: Trilok Padhi, Ramneet Kaur, Adam D. Cobb, Manoj Acharya, Anirban Roy, Colin Samplawski, Brian Matejek, Alexander M. Berenbeim, Nathaniel D. Bastian, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2505.03788)  

**Abstract**: We introduce a novel approach for calibrating uncertainty quantification (UQ) tailored for multi-modal large language models (LLMs). Existing state-of-the-art UQ methods rely on consistency among multiple responses generated by the LLM on an input query under diverse settings. However, these approaches often report higher confidence in scenarios where the LLM is consistently incorrect. This leads to a poorly calibrated confidence with respect to accuracy. To address this, we leverage cross-modal consistency in addition to self-consistency to improve the calibration of the multi-modal models. Specifically, we ground the textual responses to the visual inputs. The confidence from the grounding model is used to calibrate the overall confidence. Given that using a grounding model adds its own uncertainty in the pipeline, we apply temperature scaling - a widely accepted parametric calibration technique - to calibrate the grounding model's confidence in the accuracy of generated responses. We evaluate the proposed approach across multiple multi-modal tasks, such as medical question answering (Slake) and visual question answering (VQAv2), considering multi-modal models such as LLaVA-Med and LLaVA. The experiments demonstrate that the proposed framework achieves significantly improved calibration on both tasks. 

---
# ArrhythmiaVision: Resource-Conscious Deep Learning Models with Visual Explanations for ECG Arrhythmia Classification 

**Authors**: Zuraiz Baig, Sidra Nasir, Rizwan Ahmed Khan, Muhammad Zeeshan Ul Haque  

**Link**: [PDF](https://arxiv.org/pdf/2505.03787)  

**Abstract**: Cardiac arrhythmias are a leading cause of life-threatening cardiac events, highlighting the urgent need for accurate and timely detection. Electrocardiography (ECG) remains the clinical gold standard for arrhythmia diagnosis; however, manual interpretation is time-consuming, dependent on clinical expertise, and prone to human error. Although deep learning has advanced automated ECG analysis, many existing models abstract away the signal's intrinsic temporal and morphological features, lack interpretability, and are computationally intensive-hindering their deployment on resource-constrained platforms. In this work, we propose two novel lightweight 1D convolutional neural networks, ArrhythmiNet V1 and V2, optimized for efficient, real-time arrhythmia classification on edge devices. Inspired by MobileNet's depthwise separable convolutional design, these models maintain memory footprints of just 302.18 KB and 157.76 KB, respectively, while achieving classification accuracies of 0.99 (V1) and 0.98 (V2) on the MIT-BIH Arrhythmia Dataset across five classes: Normal Sinus Rhythm, Left Bundle Branch Block, Right Bundle Branch Block, Atrial Premature Contraction, and Premature Ventricular Contraction. In order to ensure clinical transparency and relevance, we integrate Shapley Additive Explanations and Gradient-weighted Class Activation Mapping, enabling both local and global interpretability. These techniques highlight physiologically meaningful patterns such as the QRS complex and T-wave that contribute to the model's predictions. We also discuss performance-efficiency trade-offs and address current limitations related to dataset diversity and generalizability. Overall, our findings demonstrate the feasibility of combining interpretability, predictive accuracy, and computational efficiency in practical, wearable, and embedded ECG monitoring systems. 

---
# GPU Performance Portability needs Autotuning 

**Authors**: Burkhard Ringlein, Thomas Parnell, Radu Stoica  

**Link**: [PDF](https://arxiv.org/pdf/2505.03780)  

**Abstract**: As LLMs grow in complexity, achieving state-of-the-art performance requires tight co-design across algorithms, software, and hardware. Today's reliance on a single dominant platform limits portability, creates vendor lock-in, and raises barriers for new AI hardware. In this work, we make the case for combining just-in-time (JIT) compilation with kernel parameter autotuning to enable portable, state-of-the-art performance LLM execution without code changes. Focusing on flash attention -- a widespread performance-critical LLM kernel -- we demonstrate that this approach explores up to 15x more kernel parameter configurations, produces significantly more diverse code across multiple dimensions, and even outperforms vendor-optimized implementations by up to 230%, all while reducing kernel code size by 70x and eliminating manual code optimizations. Our results highlight autotuning as a promising path to unlocking model portability across GPU vendors. 

---
# The Influence of Text Variation on User Engagement in Cross-Platform Content Sharing 

**Authors**: Yibo Hu, Yiqiao Jin, Meng Ye, Ajay Divakaran, Srijan Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.03769)  

**Abstract**: In today's cross-platform social media landscape, understanding factors that drive engagement for multimodal content, especially text paired with visuals, remains complex. This study investigates how rewriting Reddit post titles adapted from YouTube video titles affects user engagement. First, we build and analyze a large dataset of Reddit posts sharing YouTube videos, revealing that 21% of post titles are minimally modified. Statistical analysis demonstrates that title rewrites measurably improve engagement. Second, we design a controlled, multi-phase experiment to rigorously isolate the effects of textual variations by neutralizing confounding factors like video popularity, timing, and community norms. Comprehensive statistical tests reveal that effective title rewrites tend to feature emotional resonance, lexical richness, and alignment with community-specific norms. Lastly, pairwise ranking prediction experiments using a fine-tuned BERT classifier achieves 74% accuracy, significantly outperforming near-random baselines, including GPT-4o. These results validate that our controlled dataset effectively minimizes confounding effects, allowing advanced models to both learn and demonstrate the impact of textual features on engagement. By bridging quantitative rigor with qualitative insights, this study uncovers engagement dynamics and offers a robust framework for future cross-platform, multimodal content strategies. 

---
# Ultra-Low-Power Spiking Neurons in 7 nm FinFET Technology: A Comparative Analysis of Leaky Integrate-and-Fire, Morris-Lecar, and Axon-Hillock Architectures 

**Authors**: Logan Larsh, Raiyan Siddique, Sarah Sharif Yaser Mike Banad  

**Link**: [PDF](https://arxiv.org/pdf/2505.03764)  

**Abstract**: Neuromorphic computing aims to replicate the brain's remarkable energy efficiency and parallel processing capabilities for large-scale artificial intelligence applications. In this work, we present a comprehensive comparative study of three spiking neuron circuit architectures-Leaky-Integrate-and-Fire (LIF), Morris-Lecar (ML), and Axon-Hillock (AH)-implemented in a 7 nm FinFET technology. Through extensive SPICE simulations, we explore the optimization of spiking frequency, energy per spike, and static power consumption. Our results show that the AH design achieves the highest throughput, demonstrating multi-gigahertz firing rates (up to 3 GHz) with attojoule energy costs. By contrast, the ML architecture excels in subthreshold to near-threshold regimes, offering robust low-power operation (as low as 0.385 aJ/spike) and biological bursting behavior. Although LIF benefits from a decoupled current mirror for high-frequency operation, it exhibits slightly higher static leakage compared to ML and AH at elevated supply voltages. Comparisons with previous node implementations (22 nm planar, 28 nm) reveal that 7 nm FinFETs can drastically boost energy efficiency and speed albeit at the cost of increased subthreshold leakage in deep subthreshold regions. By quantifying design trade-offs for each neuron architecture, our work provides a roadmap for optimizing spiking neuron circuits in advanced nanoscale technologies to deliver neuromorphic hardware capable of both ultra-low-power operation and high computational throughput. 

---
# Splitwiser: Efficient LM inference with constrained resources 

**Authors**: Asad Aali, Adney Cardoza, Melissa Capo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03763)  

**Abstract**: Efficient inference of LLMs remains a crucial challenge, with two main phases: a compute-intensive prompt computation and a memory-intensive token generation. Despite existing batching and scheduling techniques, token generation phases fail to fully utilize compute resources, especially when compared to prompt computation phases. To address these challenges, we propose Splitwiser, a methodology that splits the two phases of an LLM inference request onto the same GPU, thereby reducing overhead and improving memory access and cache utilization. By eliminating the need to transfer data across devices, Splitwiser aims to minimize network-related overheads. In this report, we describe the basic structure of our proposed pipeline while sharing preliminary results and analysis. We implement our proposed multiprocessing design on two widely-used and independent LLM architectures: Huggingface and vLLM. We open-source our code for the respective implementations: 1) Huggingface (this https URL), and 2) vLLM (this https URL). 

---
# Deep Reinforcement Learning for Investor-Specific Portfolio Optimization: A Volatility-Guided Asset Selection Approach 

**Authors**: Arishi Orra, Aryan Bhambu, Himanshu Choudhary, Manoj Thakur, Selvaraju Natarajan  

**Link**: [PDF](https://arxiv.org/pdf/2505.03760)  

**Abstract**: Portfolio optimization requires dynamic allocation of funds by balancing the risk and return tradeoff under dynamic market conditions. With the recent advancements in AI, Deep Reinforcement Learning (DRL) has gained prominence in providing adaptive and scalable strategies for portfolio optimization. However, the success of these strategies depends not only on their ability to adapt to market dynamics but also on the careful pre-selection of assets that influence overall portfolio performance. Incorporating the investor's preference in pre-selecting assets for a portfolio is essential in refining their investment strategies. This study proposes a volatility-guided DRL-based portfolio optimization framework that dynamically constructs portfolios based on investors' risk profiles. The Generalized Autoregressive Conditional Heteroscedasticity (GARCH) model is utilized for volatility forecasting of stocks and categorizes them based on their volatility as aggressive, moderate, and conservative. The DRL agent is then employed to learn an optimal investment policy by interacting with the historical market data. The efficacy of the proposed methodology is established using stocks from the Dow $30$ index. The proposed investor-specific DRL-based portfolios outperformed the baseline strategies by generating consistent risk-adjusted returns. 

---
# Improving the Serving Performance of Multi-LoRA Large Language Models via Efficient LoRA and KV Cache Management 

**Authors**: Hang Zhang, Jiuchen Shi, Yixiao Wang, Quan Chen, Yizhou Shan, Minyi Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.03756)  

**Abstract**: Multiple Low-Rank Adapters (Multi-LoRAs) are gaining popularity for task-specific Large Language Model (LLM) applications. For multi-LoRA serving, caching hot KV caches and LoRA adapters in high bandwidth memory of accelerations can improve inference performance. However, existing Multi-LoRA inference systems fail to optimize serving performance like Time-To-First-Toke (TTFT), neglecting usage dependencies when caching LoRAs and KVs. We therefore propose FASTLIBRA, a Multi-LoRA caching system to optimize the serving performance. FASTLIBRA comprises a dependency-aware cache manager and a performance-driven cache swapper. The cache manager maintains the usage dependencies between LoRAs and KV caches during the inference with a unified caching pool. The cache swapper determines the swap-in or out of LoRAs and KV caches based on a unified cost model, when the HBM is idle or busy, respectively. Experimental results show that ELORA reduces the TTFT by 63.4% on average, compared to state-of-the-art works. 

---
# AI-Powered Agile Analog Circuit Design and Optimization 

**Authors**: Jinhai Hu, Wang Ling Goh, Yuan Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.03750)  

**Abstract**: Artificial intelligence (AI) techniques are transforming analog circuit design by automating device-level tuning and enabling system-level co-optimization. This paper integrates two approaches: (1) AI-assisted transistor sizing using Multi-Objective Bayesian Optimization (MOBO) for direct circuit parameter optimization, demonstrated on a linearly tunable transconductor; and (2) AI-integrated circuit transfer function modeling for system-level optimization in a keyword spotting (KWS) application, demonstrated by optimizing an analog bandpass filter within a machine learning training loop. The combined insights highlight how AI can improve analog performance, reduce design iteration effort, and jointly optimize analog components and application-level metrics. 

---
# APSQ: Additive Partial Sum Quantization with Algorithm-Hardware Co-Design 

**Authors**: Yonghao Tan, Pingcheng Dong, Yongkun Wu, Yu Liu, Xuejiao Liu, Peng Luo, Shih-Yang Liu, Xijie Huang, Dong Zhang, Luhong Liang, Kwang-Ting Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2505.03748)  

**Abstract**: DNN accelerators, significantly advanced by model compression and specialized dataflow techniques, have marked considerable progress. However, the frequent access of high-precision partial sums (PSUMs) leads to excessive memory demands in architectures utilizing input/weight stationary dataflows. Traditional compression strategies have typically overlooked PSUM quantization, which may account for 69% of power consumption. This study introduces a novel Additive Partial Sum Quantization (APSQ) method, seamlessly integrating PSUM accumulation into the quantization framework. A grouping strategy that combines APSQ with PSUM quantization enhanced by a reconfigurable architecture is further proposed. The APSQ performs nearly lossless on NLP and CV tasks across BERT, Segformer, and EfficientViT models while compressing PSUMs to INT8. This leads to a notable reduction in energy costs by 28-87%. Extended experiments on LLaMA2-7B demonstrate the potential of APSQ for large language models. Code is available at this https URL. 

---
# The Evolution of Rough Sets 1970s-1981 

**Authors**: Viktor Marek, Ewa Orłowska, Ivo Düntsch  

**Link**: [PDF](https://arxiv.org/pdf/2505.03747)  

**Abstract**: In this note research and publications by Zdzisław Pawlak and his collaborators from 1970s and 1981 are recalled. Focus is placed on the sources of inspiration which one can identify on the basis of those publications. Finally, developments from 1981 related to rough sets and information systems are outlined. 

---
# Promoting Security and Trust on Social Networks: Explainable Cyberbullying Detection Using Large Language Models in a Stream-Based Machine Learning Framework 

**Authors**: Silvia García-Méndez, Francisco De Arriba-Pérez  

**Link**: [PDF](https://arxiv.org/pdf/2505.03746)  

**Abstract**: Social media platforms enable instant and ubiquitous connectivity and are essential to social interaction and communication in our technological society. Apart from its advantages, these platforms have given rise to negative behaviors in the online community, the so-called cyberbullying. Despite the many works involving generative Artificial Intelligence (AI) in the literature lately, there remain opportunities to study its performance apart from zero/few-shot learning strategies. Accordingly, we propose an innovative and real-time solution for cyberbullying detection that leverages stream-based Machine Learning (ML) models able to process the incoming samples incrementally and Large Language Models (LLMS) for feature engineering to address the evolving nature of abusive and hate speech online. An explainability dashboard is provided to promote the system's trustworthiness, reliability, and accountability. Results on experimental data report promising performance close to 90 % in all evaluation metrics and surpassing those obtained by competing works in the literature. Ultimately, our proposal contributes to the safety of online communities by timely detecting abusive behavior to prevent long-lasting harassment and reduce the negative consequences in society. 

---
# AccLLM: Accelerating Long-Context LLM Inference Via Algorithm-Hardware Co-Design 

**Authors**: Yanbiao Liang, Huihong Shi, Haikuo Shao, Zhongfeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.03745)  

**Abstract**: Recently, large language models (LLMs) have achieved huge success in the natural language processing (NLP) field, driving a growing demand to extend their deployment from the cloud to edge devices. However, deploying LLMs on resource-constrained edge devices poses significant challenges, including (1) intensive computations and huge model sizes, (2) great memory and bandwidth demands introduced by the autoregressive generation process, and (3) limited scalability for handling long sequences. To address these challenges, we propose AccLLM, a comprehensive acceleration framework that enables efficient and fast long-context LLM inference through algorithm and hardware co-design. At the algorithmic level, we integrate (1) pruning, (2) {\Lambda}-shaped attention, and (3) an innovative W2A8KV4 (2-bit weights, 8-bit activations, and 4-bit KV cache) quantization scheme, thus effectively reducing memory and bandwidth requirements while facilitating LLMs' long-sequence generation. At the hardware level, we design a dedicated FPGA-based accelerator with a reconfigurable computing engine to effectively and flexibly accommodate diverse operations arising from our compression algorithm, thereby fully translating the algorithmic innovations into tangible hardware efficiency. We validate AccLLM on the Xilinx Alveo U280 FPGA, demonstrating a 4.07x energy efficiency and a 2.98x throughput compared to the state-of-the-art work FlightLLM. 

---
# Beyond Misinformation: A Conceptual Framework for Studying AI Hallucinations in (Science) Communication 

**Authors**: Anqi Shao  

**Link**: [PDF](https://arxiv.org/pdf/2504.13777)  

**Abstract**: This paper proposes a conceptual framework for understanding AI hallucinations as a distinct form of misinformation. While misinformation scholarship has traditionally focused on human intent, generative AI systems now produce false yet plausible outputs absent of such intent. I argue that these AI hallucinations should not be treated merely as technical failures but as communication phenomena with social consequences. Drawing on a supply-and-demand model and the concept of distributed agency, the framework outlines how hallucinations differ from human-generated misinformation in production, perception, and institutional response. I conclude by outlining a research agenda for communication scholars to investigate the emergence, dissemination, and audience reception of hallucinated content, with attention to macro (institutional), meso (group), and micro (individual) levels. This work urges communication researchers to rethink the boundaries of misinformation theory in light of probabilistic, non-human actors increasingly embedded in knowledge production. 

---
