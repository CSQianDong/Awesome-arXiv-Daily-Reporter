# Who's Your Judge? On the Detectability of LLM-Generated Judgments 

**Authors**: Dawei Li, Zhen Tan, Chengshuai Zhao, Bohan Jiang, Baixiang Huang, Pingchuan Ma, Abdullah Alnaibari, Kai Shu, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25154)  

**Abstract**: Large Language Model (LLM)-based judgments leverage powerful LLMs to efficiently evaluate candidate content and provide judgment scores. However, the inherent biases and vulnerabilities of LLM-generated judgments raise concerns, underscoring the urgent need for distinguishing them in sensitive scenarios like academic peer reviewing. In this work, we propose and formalize the task of judgment detection and systematically investigate the detectability of LLM-generated judgments. Unlike LLM-generated text detection, judgment detection relies solely on judgment scores and candidates, reflecting real-world scenarios where textual feedback is often unavailable in the detection process. Our preliminary analysis shows that existing LLM-generated text detection methods perform poorly given their incapability to capture the interaction between judgment scores and candidate content -- an aspect crucial for effective judgment detection. Inspired by this, we introduce \textit{J-Detector}, a lightweight and transparent neural detector augmented with explicitly extracted linguistic and LLM-enhanced features to link LLM judges' biases with candidates' properties for accurate detection. Experiments across diverse datasets demonstrate the effectiveness of \textit{J-Detector} and show how its interpretability enables quantifying biases in LLM judges. Finally, we analyze key factors affecting the detectability of LLM-generated judgments and validate the practical utility of judgment detection in real-world scenarios. 

---
# UniAPL: A Unified Adversarial Preference Learning Framework for Instruct-Following 

**Authors**: FaQiang Qian, WeiKun Zhang, Ziliang Wang, Kang An, Xuhui Zheng, Liangjian Wen, Mengya Gao, Yong Dai, Yichao Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25148)  

**Abstract**: Shaping powerful LLMs to be beneficial and safe is central to AI alignment. We argue that post-training alignment is fundamentally a unified Preference Learning problem, involving two modalities: demonstrated preferences (e.g., Supervised Fine-Tuning, SFT) and comparative preferences (e.g., Reinforcement Learning, RL).The standard sequential pipeline-SFT followed by RL-is flawed due to a critical distributional mismatch: SFT uses static expert data, but as the policy evolves, its generation distribution drifts, making SFT knowledge brittle. Subsequent RL then explores without direct access to the rich, ground-truth knowledge in expert demonstrations, leading to inefficient, ungrounded updates. This separation prevents mutual regularization between data sources. To address this, we reframe alignment as a constrained optimization problem and propose Unified Adversarial Preference Learning (UniAPL),a novel framework that dynamically aligns the policy's distribution with the expert's. UniAPL implements a single-stage unified training objective, jointly learning from mixed batches of SFT and preference data. In every gradient step, dense expert demonstrations directly ground and regularize online exploration, inherently resolving distributional mismatch and maximizing data this http URL evaluate UniAPL on instruction-following tasks using Qwen3-235B-Instruct-2507 as the teacher. Our models match or exceed strong GRPO baselines: +5.77% on Qwen3-0.6B (matching a 32B model) and +3.75% on Qwen3-4B,even outperforming the teacher. Analyses of response length and log-probability distributions confirm that UniAPL outputs closely mimic expert demonstrations, achieving both stronger performance and better behavioral alignment. 

---
# Visual serial processing deficits explain divergences in human and VLM reasoning 

**Authors**: Nicholas Budny, Kia Ghods, Declan Campbell, Raja Marjieh, Amogh Joshi, Sreejan Kumar, Jonathan D. Cohen, Taylor W. Webb, Thomas L. Griffiths  

**Link**: [PDF](https://arxiv.org/pdf/2509.25142)  

**Abstract**: Why do Vision Language Models (VLMs), despite success on standard benchmarks, often fail to match human performance on surprisingly simple visual reasoning tasks? While the underlying computational principles are still debated, we hypothesize that a crucial factor is a deficit in visually-grounded serial processing. To test this hypothesis, we compared human and VLM performance across tasks designed to vary serial processing demands in three distinct domains: geometric reasoning, perceptual enumeration, and mental rotation. Tasks within each domain varied serial processing load by manipulating factors such as geometric concept complexity, perceptual individuation load, and transformation difficulty. Across all domains, our results revealed a consistent pattern: decreased VLM accuracy was strongly correlated with increased human reaction time (used as a proxy for serial processing load). As tasks require more demanding serial processing -- whether composing concepts, enumerating items, or performing mental transformations -- the VLM-human performance gap widens reliably. These findings support our hypothesis, indicating that limitations in serial, visually grounded reasoning represent a fundamental bottleneck that distinguishes current VLMs from humans. 

---
# ReasoningBank: Scaling Agent Self-Evolving with Reasoning Memory 

**Authors**: Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, Tomas Pfister  

**Link**: [PDF](https://arxiv.org/pdf/2509.25140)  

**Abstract**: With the growing adoption of large language model agents in persistent real-world roles, they naturally encounter continuous streams of tasks. A key limitation, however, is their failure to learn from the accumulated interaction history, forcing them to discard valuable insights and repeat past errors. We propose ReasoningBank, a novel memory framework that distills generalizable reasoning strategies from an agent's self-judged successful and failed experiences. At test time, an agent retrieves relevant memories from ReasoningBank to inform its interaction and then integrates new learnings back, enabling it to become more capable over time. Building on this powerful experience learner, we further introduce memory-aware test-time scaling (MaTTS), which accelerates and diversifies this learning process by scaling up the agent's interaction experience. By allocating more compute to each task, the agent generates abundant, diverse experiences that provide rich contrastive signals for synthesizing higher-quality memory. The better memory in turn guides more effective scaling, establishing a powerful synergy between memory and test-time scaling. Across web browsing and software engineering benchmarks, ReasoningBank consistently outperforms existing memory mechanisms that store raw trajectories or only successful task routines, improving both effectiveness and efficiency; MaTTS further amplifies these gains. These findings establish memory-driven experience scaling as a new scaling dimension, enabling agents to self-evolve with emergent behaviors naturally arise. 

---
# Vision-and-Language Navigation with Analogical Textual Descriptions in LLMs 

**Authors**: Yue Zhang, Tianyi Ma, Zun Wang, Yanyuan Qiao, Parisa Kordjamshidi  

**Link**: [PDF](https://arxiv.org/pdf/2509.25139)  

**Abstract**: Integrating large language models (LLMs) into embodied AI models is becoming increasingly prevalent. However, existing zero-shot LLM-based Vision-and-Language Navigation (VLN) agents either encode images as textual scene descriptions, potentially oversimplifying visual details, or process raw image inputs, which can fail to capture abstract semantics required for high-level reasoning. In this paper, we improve the navigation agent's contextual understanding by incorporating textual descriptions from multiple perspectives that facilitate analogical reasoning across images. By leveraging text-based analogical reasoning, the agent enhances its global scene understanding and spatial reasoning, leading to more accurate action decisions. We evaluate our approach on the R2R dataset, where our experiments demonstrate significant improvements in navigation performance. 

---
# The Era of Real-World Human Interaction: RL from User Conversations 

**Authors**: Chuanyang Jin, Jing Xu, Bo Liu, Leitian Tao, Olga Golovneva, Tianmin Shu, Wenting Zhao, Xian Li, Jason Weston  

**Link**: [PDF](https://arxiv.org/pdf/2509.25137)  

**Abstract**: We posit that to achieve continual model improvement and multifaceted alignment, future models must learn from natural human interaction. Current conversational models are aligned using pre-annotated, expert-generated human feedback. In this work, we introduce Reinforcement Learning from Human Interaction (RLHI), a paradigm that learns directly from in-the-wild user conversations. We develop two complementary methods: (1) RLHI with User-Guided Rewrites, which revises unsatisfactory model outputs based on users' natural-language follow-up responses, (2) RLHI with User-Based Rewards, which learns via a reward model conditioned on knowledge of the user's long-term interaction history (termed persona). Together, these methods link long-term user personas to turn-level preferences via persona-conditioned preference optimization. Trained on conversations derived from WildChat, both RLHI variants outperform strong baselines in personalization and instruction-following, and similar feedback enhances performance on reasoning benchmarks. These results suggest organic human interaction offers scalable, effective supervision for personalized alignment. 

---
# From $f(x)$ and $g(x)$ to $f(g(x))$: LLMs Learn New Skills in RL by Composing Old Ones 

**Authors**: Lifan Yuan, Weize Chen, Yuchen Zhang, Ganqu Cui, Hanbin Wang, Ziming You, Ning Ding, Zhiyuan Liu, Maosong Sun, Hao Peng  

**Link**: [PDF](https://arxiv.org/pdf/2509.25123)  

**Abstract**: Does RL teach LLMs genuinely new skills, or does it merely activate existing ones? This question lies at the core of ongoing debates about the role of RL in LLM post-training. On one side, strong empirical results can be achieved with RL even without preceding supervised finetuning; on the other, critics argue that RL contributes little beyond reweighting existing reasoning strategies. This work provides concrete evidence that LLMs can acquire genuinely new skills during RL by composing existing ones, mirroring one of the central mechanisms by which humans acquire new cognitive skills. To mitigate data contamination and other confounding factors, and to allow precise control over task complexity, we develop a synthetic framework for our investigation. Specifically, we define a skill as the ability to infer the output of a string transformation function f(x) given x. When an LLM has already learned f and g prior to RL, our experiments reveal that RL enables it to learn unseen compositions of them h(x)=g(f(x)). Further, this compositional ability generalizes to more difficult problems such as compositions of >2 functions unseen during RL training. Surprisingly, our experiments show that compositional skill acquired on a source task transfers to a different target task. This transfer happens even without compositional training on the target, requiring only prior knowledge of the target's atomic skills. Our qualitative analysis shows that RL fundamentally changes the reasoning behaviors of the models. In contrast, next-token training with the same data yields none of these findings. Our systematic experiments provide fresh insights into LLM learning, suggesting the value of first building base models with basic skills, then using RL to incentivize advanced, generalizable skills for complex problems. 

---
# HeDA: An Intelligent Agent System for Heatwave Risk Discovery through Automated Knowledge Graph Construction and Multi-layer Risk Propagation Analysis 

**Authors**: Yiquan Wang, Tin-Yeh Huang, Qingyun Gao, Jialin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25112)  

**Abstract**: Heatwaves pose complex cascading risks across interconnected climate, social, and economic systems, but knowledge fragmentation in scientific literature hinders comprehensive understanding of these risk pathways. We introduce HeDA (Heatwave Discovery Agent), an intelligent multi-agent system designed for automated scientific discovery through knowledge graph construction and multi-layer risk propagation analysis. HeDA processes over 10,247 academic papers to construct a comprehensive knowledge graph with 23,156 nodes and 89,472 relationships, employing novel multi-layer risk propagation analysis to systematically identify overlooked risk transmission pathways. Our system achieves 78.9% accuracy on complex question-answering tasks, outperforming state-of-the-art baselines including GPT-4 by 13.7%. Critically, HeDA successfully discovered five previously unidentified high-impact risk chains, such as the pathway where a heatwave leads to a water demand surge, resulting in industrial water restrictions and ultimately causing small business disruption, which were validated through historical case studies and domain expert review. This work presents a new paradigm for AI-driven scientific discovery, providing actionable insights for developing more resilient climate adaptation strategies. 

---
# Cogito, Ergo Ludo: An Agent that Learns to Play by Reasoning and Planning 

**Authors**: Sai Wang, Yu Wu, Zhongwen Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25052)  

**Abstract**: The pursuit of artificial agents that can learn to master complex environments has led to remarkable successes, yet prevailing deep reinforcement learning methods often rely on immense experience, encoding their knowledge opaquely within neural network weights. We propose a different paradigm, one in which an agent learns to play by reasoning and planning. We introduce Cogito, ergo ludo (CEL), a novel agent architecture that leverages a Large Language Model (LLM) to build an explicit, language-based understanding of its environment's mechanics and its own strategy. Starting from a tabula rasa state with no prior knowledge (except action set), CEL operates on a cycle of interaction and reflection. After each episode, the agent analyzes its complete trajectory to perform two concurrent learning processes: Rule Induction, where it refines its explicit model of the environment's dynamics, and Strategy and Playbook Summarization, where it distills experiences into an actionable strategic playbook. We evaluate CEL on diverse grid-world tasks (i.e., Minesweeper, Frozen Lake, and Sokoban), and show that the CEL agent successfully learns to master these games by autonomously discovering their rules and developing effective policies from sparse rewards. Ablation studies confirm that the iterative process is critical for sustained learning. Our work demonstrates a path toward more general and interpretable agents that not only act effectively but also build a transparent and improving model of their world through explicit reasoning on raw experience. 

---
# Scaling Synthetic Task Generation for Agents via Exploration 

**Authors**: Ram Ramrakhya, Andrew Szot, Omar Attia, Yuhao Yang, Anh Nguyen, Bogdan Mazoure, Zhe Gan, Harsh Agrawal, Alexander Toshev  

**Link**: [PDF](https://arxiv.org/pdf/2509.25047)  

**Abstract**: Post-Training Multimodal Large Language Models (MLLMs) to build interactive agents holds promise across domains such as computer-use, web navigation, and robotics. A key challenge in scaling such post-training is lack of high-quality downstream agentic task datasets with tasks that are diverse, feasible, and verifiable. Existing approaches for task generation rely heavily on human annotation or prompting MLLM with limited downstream environment information, which is either costly or poorly scalable as it yield tasks with limited coverage. To remedy this, we present AutoPlay, a scalable pipeline for task generation that explicitly explores interactive environments to discover possible interactions and current state information to synthesize environment-grounded tasks. AutoPlay operates in two stages: (i) an exploration phase, where an MLLM explorer agent systematically uncovers novel environment states and functionalities, and (ii) a task generation phase, where a task generator leverages exploration trajectories and a set of task guideline prompts as context to synthesize diverse, executable, and verifiable tasks. We show AutoPlay generates 20k tasks across 20 Android applications and 10k tasks across 13 applications Ubuntu applications to train mobile-use and computer-use agents. AutoPlay generated tasks enable large-scale task demonstration synthesis without human annotation by employing an MLLM task executor and verifier. This data enables training MLLM-based UI agents that improve success rates up to $20.0\%$ on mobile-use and $10.9\%$ on computer-use scenarios. In addition, AutoPlay generated tasks combined with MLLM verifier-based rewards enable scaling reinforcement learning training of UI agents, leading to an additional $5.7\%$ gain. coverage. These results establish AutoPlay as a scalable approach for post-training capable MLLM agents reducing reliance on human annotation. 

---
# CLPO: Curriculum Learning meets Policy Optimization for LLM Reasoning 

**Authors**: Shijie Zhang, Guohao Sun, Kevin Zhang, Xiang Guo, Rujun Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.25004)  

**Abstract**: Recently, online Reinforcement Learning with Verifiable Rewards (RLVR) has become a key paradigm for enhancing the reasoning capabilities of Large Language Models (LLMs). However, existing methods typically treat all training samples uniformly, overlooking the vast differences in problem difficulty relative to the model's current capabilities. This uniform training strategy leads to inefficient exploration of problems the model has already mastered, while concurrently lacking effective guidance on problems that are challenging its abilities the most, limiting both learning efficiency and upper-bound performance. To address this, we propose CLPO (Curriculum-guided Learning for Policy Optimization), a novel algorithm that creates a dynamic pedagogical feedback loop within the policy optimization process. The core of CLPO leverages the model's own rollout performance to conduct real-time difficulty assessment, thereby constructing an Online Curriculum. This curriculum then guides an Adaptive Problem Restructuring mechanism, where the model acts as its own teacher: it diversifies medium-difficulty problems to promote generalization and simplifies challenging problems to make them more attainable. Our approach transforms the static training procedure into a dynamic process that co-evolves with the model's capabilities. Experiments show that CLPO achieves state-of-the-art performance across eight challenging mathematical and general reasoning benchmarks, with an average pass@1 improvement of 6.96% over other methods, demonstrating its potential for more efficiently training more capable reasoning models. 

---
# Agentic Exploration of Physics Models 

**Authors**: Maximilian Nägele, Florian Marquardt  

**Link**: [PDF](https://arxiv.org/pdf/2509.24978)  

**Abstract**: The process of scientific discovery relies on an interplay of observations, analysis, and hypothesis generation. Machine learning is increasingly being adopted to address individual aspects of this process. However, it remains an open challenge to fully automate the open-ended, heuristic, iterative loop required to discover the laws of an unknown system by exploring it through experiments and analysis, without tailoring the approach to the specifics of a given task. Here, we introduce SciExplorer, an agent that leverages large language model tool-use capabilities to enable free-form exploration of systems without any domain-specific blueprints, and apply it to the exploration of physical systems that are initially unknown to the agent. We test SciExplorer on a broad set of models spanning mechanical dynamical systems, wave evolution, and quantum many-body physics. Despite using a minimal set of tools, primarily based on code execution, we observe impressive performance on tasks such as recovering equations of motion from observed dynamics and inferring Hamiltonians from expectation values. The demonstrated effectiveness of this setup opens the door towards similar scientific exploration in other domains, without the need for finetuning or task-specific instructions. 

---
# KIRETT - A wearable device to support rescue operations using artificial intelligence to improve first aid 

**Authors**: Johannes Zenkert, Christian Weber, Mubaris Nadeem, Lisa Bender, Madjid Fathi, Abu Shad Ahammed, Aniebiet Micheal Ezekiel, Roman Obermaisser, Maximilian Bradford  

**Link**: [PDF](https://arxiv.org/pdf/2509.24934)  

**Abstract**: This short paper presents first steps in the scientific part of the KIRETT project, which aims to improve first aid during rescue operations using a wearable device. The wearable is used for computer-aided situation recognition by means of artificial intelligence. It provides contextual recommendations for actions and operations to rescue personnel and is intended to minimize damage to patients due to incorrect treatment, as well as increase the probability of survival. The paper describes a first overview of research approaches within the project. 

---
# When Autonomous Vehicle Meets V2X Cooperative Perception: How Far Are We? 

**Authors**: An Guo, Shuoxiao Zhang, Enyi Tang, Xinyu Gao, Haomin Pang, Haoxiang Tian, Yanzhou Mu, Wu Wen, Chunrong Fang, Zhenyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24927)  

**Abstract**: With the tremendous advancement of deep learning and communication technology, Vehicle-to-Everything (V2X) cooperative perception has the potential to address limitations in sensing distant objects and occlusion for a single-agent perception system. V2X cooperative perception systems are software systems characterized by diverse sensor types and cooperative agents, varying fusion schemes, and operation under different communication conditions. Therefore, their complex composition gives rise to numerous operational challenges. Furthermore, when cooperative perception systems produce erroneous predictions, the types of errors and their underlying causes remain insufficiently explored. To bridge this gap, we take an initial step by conducting an empirical study of V2X cooperative perception. To systematically evaluate the impact of cooperative perception on the ego vehicle's perception performance, we identify and analyze six prevalent error patterns in cooperative perception systems. We further conduct a systematic evaluation of the critical components of these systems through our large-scale study and identify the following key findings: (1) The LiDAR-based cooperation configuration exhibits the highest perception performance; (2) Vehicle-to-infrastructure (V2I) and vehicle-to-vehicle (V2V) communication exhibit distinct cooperative perception performance under different fusion schemes; (3) Increased cooperative perception errors may result in a higher frequency of driving violations; (4) Cooperative perception systems are not robust against communication interference when running online. Our results reveal potential risks and vulnerabilities in critical components of cooperative perception systems. We hope that our findings can better promote the design and repair of cooperative perception systems. 

---
# MASLegalBench: Benchmarking Multi-Agent Systems in Deductive Legal Reasoning 

**Authors**: Huihao Jing, Wenbin Hu, Hongyu Luo, Jianhui Yang, Wei Fan, Haoran Li, Yangqiu Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.24922)  

**Abstract**: Multi-agent systems (MAS), leveraging the remarkable capabilities of Large Language Models (LLMs), show great potential in addressing complex tasks. In this context, integrating MAS with legal tasks is a crucial step. While previous studies have developed legal benchmarks for LLM agents, none are specifically designed to consider the unique advantages of MAS, such as task decomposition, agent specialization, and flexible training. In fact, the lack of evaluation methods limits the potential of MAS in the legal domain. To address this gap, we propose MASLegalBench, a legal benchmark tailored for MAS and designed with a deductive reasoning approach. Our benchmark uses GDPR as the application scenario, encompassing extensive background knowledge and covering complex reasoning processes that effectively reflect the intricacies of real-world legal situations. Furthermore, we manually design various role-based MAS and conduct extensive experiments using different state-of-the-art LLMs. Our results highlight the strengths, limitations, and potential areas for improvement of existing models and MAS architectures. 

---
# Meta-Learning Theory-Informed Inductive Biases using Deep Kernel Gaussian Processes 

**Authors**: Bahti Zakirov, Gašper Tkačik  

**Link**: [PDF](https://arxiv.org/pdf/2509.24919)  

**Abstract**: Normative and task-driven theories offer powerful top-down explanations for biological systems, yet the goals of quantitatively arbitrating between competing theories, and utilizing them as inductive biases to improve data-driven fits of real biological datasets are prohibitively laborious, and often impossible. To this end, we introduce a Bayesian meta-learning framework designed to automatically convert raw functional predictions from normative theories into tractable probabilistic models. We employ adaptive deep kernel Gaussian processes, meta-learning a kernel on synthetic data generated from a normative theory. This Theory-Informed Kernel specifies a probabilistic model representing the theory predictions -- usable for both fitting data and rigorously validating the theory. As a demonstration, we apply our framework to the early visual system, using efficient coding as our normative theory. We show improved response prediction accuracy in ex vivo recordings of mouse retinal ganglion cells stimulated by natural scenes compared to conventional data-driven baselines, while providing well-calibrated uncertainty estimates and interpretable representations. Using exact Bayesian model selection, we also show that our informed kernel can accurately infer the degree of theory-match from data, confirming faithful encapsulation of theory structure. This work provides a more general, scalable, and automated approach for integrating theoretical knowledge into data-driven scientific inquiry in neuroscience and beyond. 

---
# Neural network embeddings recover value dimensions from psychometric survey items on par with human data 

**Authors**: Max Pellert, Clemens M. Lechner, Indira Sen, Markus Strohmaier  

**Link**: [PDF](https://arxiv.org/pdf/2509.24906)  

**Abstract**: This study introduces "Survey and Questionnaire Item Embeddings Differentials" (SQuID), a novel methodological approach that enables neural network embeddings to effectively recover latent dimensions from psychometric survey items. We demonstrate that embeddings derived from large language models, when processed with SQuID, can recover the structure of human values obtained from human rater judgments on the Revised Portrait Value Questionnaire (PVQ-RR). Our experimental validation compares multiple embedding models across a number of evaluation metrics. Unlike previous approaches, SQuID successfully addresses the challenge of obtaining negative correlations between dimensions without requiring domain-specific fine-tuning. Quantitative analysis reveals that our embedding-based approach explains 55% of variance in dimension-dimension similarities compared to human data. Multidimensional scaling configurations from both types of data show fair factor congruence coefficients and largely follow the underlying theory. These results demonstrate that semantic embeddings can effectively replicate psychometric structures previously established through extensive human surveys. The approach offers substantial advantages in cost, scalability and flexibility while maintaining comparable quality to traditional methods. Our findings have significant implications for psychometrics and social science research, providing a complementary methodology that could expand the scope of human behavior and experience represented in measurement tools. 

---
# RealUnify: Do Unified Models Truly Benefit from Unification? A Comprehensive Benchmark 

**Authors**: Yang Shi, Yuhao Dong, Yue Ding, Yuran Wang, Xuanyu Zhu, Sheng Zhou, Wenting Liu, Haochen Tian, Rundong Wang, Huanqian Wang, Zuyan Liu, Bohan Zeng, Ruizhe Chen, Qixun Wang, Zhuoran Zhang, Xinlong Chen, Chengzhuo Tong, Bozhou Li, Chaoyou Fu, Qiang Liu, Haotian Wang, Wenjing Yang, Yuanxing Zhang, Pengfei Wan, Yi-Fan Zhang, Ziwei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24897)  

**Abstract**: The integration of visual understanding and generation into unified multimodal models represents a significant stride toward general-purpose AI. However, a fundamental question remains unanswered by existing benchmarks: does this architectural unification actually enable synergetic interaction between the constituent capabilities? Existing evaluation paradigms, which primarily assess understanding and generation in isolation, are insufficient for determining whether a unified model can leverage its understanding to enhance its generation, or use generative simulation to facilitate deeper comprehension. To address this critical gap, we introduce RealUnify, a benchmark specifically designed to evaluate bidirectional capability synergy. RealUnify comprises 1,000 meticulously human-annotated instances spanning 10 categories and 32 subtasks. It is structured around two core axes: 1) Understanding Enhances Generation, which requires reasoning (e.g., commonsense, logic) to guide image generation, and 2) Generation Enhances Understanding, which necessitates mental simulation or reconstruction (e.g., of transformed or disordered visual inputs) to solve reasoning tasks. A key contribution is our dual-evaluation protocol, which combines direct end-to-end assessment with a diagnostic stepwise evaluation that decomposes tasks into distinct understanding and generation phases. This protocol allows us to precisely discern whether performance bottlenecks stem from deficiencies in core abilities or from a failure to integrate them. Through large-scale evaluations of 12 leading unified models and 6 specialized baselines, we find that current unified models still struggle to achieve effective synergy, indicating that architectural unification alone is insufficient. These results highlight the need for new training strategies and inductive biases to fully unlock the potential of unified modeling. 

---
# The Emergence of Social Science of Large Language Models 

**Authors**: Xiao Jia, Zhanzhan Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.24877)  

**Abstract**: The social science of large language models (LLMs) examines how these systems evoke mind attributions, interact with one another, and transform human activity and institutions. We conducted a systematic review of 270 studies, combining text embeddings, unsupervised clustering and topic modeling to build a computational taxonomy. Three domains emerge organically across the reviewed literature. LLM as Social Minds examines whether and when models display behaviors that elicit attributions of cognition, morality and bias, while addressing challenges such as test leakage and surface cues. LLM Societies examines multi-agent settings where interaction protocols, architectures and mechanism design shape coordination, norms, institutions and collective epistemic processes. LLM-Human Interactions examines how LLMs reshape tasks, learning, trust, work and governance, and how risks arise at the human-AI interface. This taxonomy provides a reproducible map of a fragmented field, clarifies evidentiary standards across levels of analysis, and highlights opportunities for cumulative progress in the social science of artificial intelligence. 

---
# PhysicsMinions: Winning Gold Medals in the Latest Physics Olympiads with a Coevolutionary Multimodal Multi-Agent System 

**Authors**: Fangchen Yu, Junchi Yao, Ziyi Wang, Haiyuan Wan, Youling Huang, Bo Zhang, Shuyue Hu, Dongzhan Zhou, Ning Ding, Ganqu Cui, Lei Bai, Wanli Ouyang, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.24855)  

**Abstract**: Physics is central to understanding and shaping the real world, and the ability to solve physics problems is a key indicator of real-world physical intelligence. Physics Olympiads, renowned as the crown of competitive physics, provide a rigorous testbed requiring complex reasoning and deep multimodal understanding, yet they remain largely underexplored in AI research. Existing approaches are predominantly single-model based, and open-source MLLMs rarely reach gold-medal-level performance. To address this gap, we propose PhysicsMinions, a coevolutionary multi-agent system for Physics Olympiad. Its architecture features three synergistic studios: a Visual Studio to interpret diagrams, a Logic Studio to formulate solutions, and a Review Studio to perform dual-stage verification. The system coevolves through an iterative refinement loop where feedback from the Review Studio continuously guides the Logic Studio, enabling the system to self-correct and converge towards the ground truth. Evaluated on the HiPhO benchmark spanning 7 latest physics Olympiads, PhysicsMinions delivers three major breakthroughs: (i) Strong generalization: it consistently improves both open-source and closed-source models of different sizes, delivering clear benefits over their single-model baselines; (ii) Historic breakthroughs: it elevates open-source models from only 1-2 to 6 gold medals across 7 Olympiads, achieving the first-ever open-source gold medal in the latest International Physics Olympiad (IPhO) under the average-score metric; and (iii) Scaling to human expert: it further advances the open-source Pass@32 score to 26.8/30 points on the latest IPhO, ranking 4th of 406 contestants and far surpassing the top single-model score of 22.7 (ranked 22nd). Generally, PhysicsMinions offers a generalizable framework for Olympiad-level problem solving, with the potential to extend across disciplines. 

---
# Pushing LLMs to Their Logical Reasoning Bound: The Role of Data Reasoning Intensity 

**Authors**: Zhen Bi, Zhenlin Hu, Jinnan Yang, Mingyang Chen, Cheng Deng, Yida Xue, Zeyu Yang, Qing Shen, Zhenfang Liu, Kang Zhao, Ningyu Zhang, Jungang Lou  

**Link**: [PDF](https://arxiv.org/pdf/2509.24836)  

**Abstract**: Recent advances in large language models (LLMs) highlight the importance of training data structure and quality in shaping reasoning behavior. However, most existing approaches focus on transforming data formats while neglecting the internal reasoning complexity of training samples, leaving the reasoning potential of data under-explored and underutilized. In this work, we posit that LLM logical reasoning performance is jointly constrained by the potential of the training data and the cognitive capacity of the model. To make this relationship measurable, we introduce Data Reasoning Intensity (DRI), a novel metric that quantifies the latent logical reasoning complexity of samples by decomposing and aggregating their logical structures. This allows us to analyze how well current LLMs utilize logical reasoning signals and identify performance gaps relative to data potential. Based on this insight, we introduce a re-cognizing optimization strategy that systematically enhances the logical reasoning intensity of training this http URL than increasing data volume, our method re-optimizes existing samples to better align with the LLM's logical reasoning boundary. Extensive experiments show that our approach significantly improves performance and generalization over data-centric strategies. We further validate our method under a reinforcement learning framework. Our results indicate that prioritizing reasoning complexity in data rather than sheer scale or superficial form is essential to realizing LLMs' full cognitive potential. 

---
# Query Circuits: Explaining How Language Models Answer User Prompts 

**Authors**: Tung-Yu Wu, Fazl Barez  

**Link**: [PDF](https://arxiv.org/pdf/2509.24808)  

**Abstract**: Explaining why a language model produces a particular output requires local, input-level explanations. Existing methods uncover global capability circuits (e.g., indirect object identification), but not why the model answers a specific input query in a particular way. We introduce query circuits, which directly trace the information flow inside a model that maps a specific input to the output. Unlike surrogate-based approaches (e.g., sparse autoencoders), query circuits are identified within the model itself, resulting in more faithful and computationally accessible explanations. To make query circuits practical, we address two challenges. First, we introduce Normalized Deviation Faithfulness (NDF), a robust metric to evaluate how well a discovered circuit recovers the model's decision for a specific input, and is broadly applicable to circuit discovery beyond our setting. Second, we develop sampling-based methods to efficiently identify circuits that are sparse yet faithfully describe the model's behavior. Across benchmarks (IOI, arithmetic, MMLU, and ARC), we find that there exist extremely sparse query circuits within the model that can recover much of its performance on single queries. For example, a circuit covering only 1.3% of model connections can recover about 60% of performance on an MMLU questions. Overall, query circuits provide a step towards faithful, scalable explanations of how language models process individual inputs. 

---
# TimeOmni-1: Incentivizing Complex Reasoning with Time Series in Large Language Models 

**Authors**: Tong Guan, Zijie Meng, Dianqi Li, Shiyu Wang, Chao-Han Huck Yang, Qingsong Wen, Zuozhu Liu, Sabato Marco Siniscalchi, Ming Jin, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24803)  

**Abstract**: Recent advances in multimodal time series learning underscore a paradigm shift from analytics centered on basic patterns toward advanced time series understanding and reasoning. However, existing multimodal time series datasets mostly remain at the level of surface alignment and question answering, without reaching the depth of genuine reasoning. The absence of well-defined tasks that genuinely require time series reasoning, along with the scarcity of high-quality data, has limited progress in building practical time series reasoning models (TSRMs). To this end, we introduce Time Series Reasoning Suite (TSR-Suite), which formalizes four atomic tasks that span three fundamental capabilities for reasoning with time series: (1) perception, acquired through scenario understanding and causality discovery; (2) extrapolation, realized via event-aware forecasting; and (3) decision-making, developed through deliberation over perception and extrapolation. TSR-Suite is the first comprehensive time series reasoning suite that supports not only thorough evaluation but also the data pipeline and training of TSRMs. It contains more than 23K samples, of which 2.3K are carefully curated through a human-guided hierarchical annotation process. Building on this foundation, we introduce TimeOmni-1, the first unified reasoning model designed to address diverse real-world problems demanding time series reasoning. The model is trained in multiple stages, integrating a mixture of task scenarios, novel reward functions, and tailored optimizations. Experiments show that TimeOmni-1 delivers strong out-of-distribution generalization across all tasks and achieves a high rate of valid responses. It significantly improves causality discovery accuracy (64.0% vs. 35.9% with GPT-4.1) and raises the valid response rate by over 6% compared to GPT-4.1 on the event-aware forecasting task. 

---
# From Ambiguity to Verdict: A Semiotic-Grounded Multi-Perspective Agent for LLM Logical Reasoning 

**Authors**: Yunyao Zhang, Xinglang Zhang, Junxi Sheng, Wenbing Li, Junqing Yu, Wei Yang, Zikai Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.24765)  

**Abstract**: Logical reasoning is a fundamental capability of large language models (LLMs). However, existing studies largely overlook the interplay between logical complexity and semantic complexity, resulting in methods that struggle to address challenging scenarios involving abstract propositions, ambiguous contexts, and conflicting stances, which are central to human reasoning. For this gap, we propose LogicAgent, a semiotic-square-guided framework designed to jointly address logical complexity and semantic complexity. LogicAgent explicitly performs multi-perspective deduction in first-order logic (FOL), while mitigating vacuous reasoning through existential import checks that incorporate a three-valued decision scheme (True, False, Uncertain) to handle boundary cases more faithfully. Furthermore, to overcome the semantic simplicity and low logical complexity of existing datasets, we introduce RepublicQA, a benchmark that reaches college-level difficulty (FKGL = 11.94) and exhibits substantially greater lexical and structural diversity than prior benchmarks. RepublicQA is grounded in philosophical concepts, featuring abstract propositions and systematically organized contrary and contradictory relations, making it the most semantically rich resource for evaluating logical reasoning. Experiments demonstrate that LogicAgent achieves state-of-the-art performance on RepublicQA, with a 6.25% average gain over strong baselines, and generalizes effectively to mainstream logical reasoning benchmarks including ProntoQA, ProofWriter, FOLIO, and ProverQA, achieving an additional 7.05% average gain. These results highlight the strong effectiveness of our semiotic-grounded multi-perspective reasoning in boosting LLMs' logical performance. 

---
# Spatial-Functional awareness Transformer-based graph archetype contrastive learning for Decoding Visual Neural Representations from EEG 

**Authors**: Yueming Sun, Long Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24761)  

**Abstract**: Decoding visual neural representations from Electroencephalography (EEG) signals remains a formidable challenge due to their high-dimensional, noisy, and non-Euclidean nature. In this work, we propose a Spatial-Functional Awareness Transformer-based Graph Archetype Contrastive Learning (SFTG) framework to enhance EEG-based visual decoding. Specifically, we introduce the EEG Graph Transformer (EGT), a novel graph-based neural architecture that simultaneously encodes spatial brain connectivity and temporal neural dynamics. To mitigate high intra-subject variability, we propose Graph Archetype Contrastive Learning (GAC), which learns subject-specific EEG graph archetypes to improve feature consistency and class separability. Furthermore, we conduct comprehensive subject-dependent and subject-independent evaluations on the Things-EEG dataset, demonstrating that our approach significantly outperforms prior state-of-the-art EEG decoding this http URL results underscore the transformative potential of integrating graph-based learning with contrastive objectives to enhance EEG-based brain decoding, paving the way for more generalizable and robust neural representations. 

---
# On the Self-awareness of Large Reasoning Models' Capability Boundaries 

**Authors**: Qingjie Zhang, Yujia Fu, Yang Wang, Liu Yan, Tao Wei, Ke Xu, Minlie Huang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24711)  

**Abstract**: Large Reasoning Models (LRMs) have shown impressive performance on complex reasoning tasks such as mathematics, yet they also display misbehaviors that expose their limitations. In particular, when faced with hard questions, LRMs often engage in unproductive reasoning until context limit, producing wrong answers while wasting substantial computation. This phenomenon reflects a fundamental issue: current answering paradigms overlook the relationship between questions and LRMs' capability boundaries. In this paper, we investigate whether LRMs possess self-awareness of capability boundaries. We begin by an observation that LRMs may know what they cannot solve through expressed reasoning confidence. For black-box models, we find that reasoning expressions reveal boundary signals, with accelerated growing confidence trajectory for solvable problems but convergent uncertainty trajectory for unsolvable ones. For white-box models, we show that hidden states of the last input token encode boundary information, with solvable and unsolvable problems linearly separable even before reasoning begins. Building on these findings, we propose two simple yet effective optimization strategies: reasoning expression monitoring and hidden states monitoring. Experiments demonstrate that these boundary-aware strategies enable LRMs to avoid unproductive reasoning without sacrificing accuracy, significantly improving reliability and efficiency by cutting token usage up to 62.7 - 93.6%. 

---
# Successful Misunderstandings: Learning to Coordinate Without Being Understood 

**Authors**: Nikolaos Kondylidis, Anil Yaman, Frank van Harmelen, Erman Acar, Annette ten Teije  

**Link**: [PDF](https://arxiv.org/pdf/2509.24660)  

**Abstract**: The main approach to evaluating communication is by assessing how well it facilitates coordination. If two or more individuals can coordinate through communication, it is generally assumed that they understand one another. We investigate this assumption in a signaling game where individuals develop a new vocabulary of signals to coordinate successfully. In our game, the individuals do not have common observations besides the communication signal and outcome of the interaction, i.e. received reward. This setting is used as a proxy to study communication emergence in populations of agents that perceive their environment very differently, e.g. hybrid populations that include humans and artificial agents. Agents develop signals, use them, and refine interpretations while not observing how other agents are using them. While populations always converge to optimal levels of coordination, in some cases, interacting agents interpret and use signals differently, converging to what we call successful misunderstandings. However, agents of population that coordinate using misaligned interpretations, are unable to establish successful coordination with new interaction partners. Not leading to coordination failure immediately, successful misunderstandings are difficult to spot and repair. Having at least three agents that all interact with each other are the two minimum conditions to ensure the emergence of shared interpretations. Under these conditions, the agent population exhibits this emergent property of compensating for the lack of shared observations of signal use, ensuring the emergence of shared interpretations. 

---
# "Stop replacing salt with sugar!'': Towards Intuitive Human-Agent Teaching 

**Authors**: Nikolaos Kondylidis, Andrea Rafanelli, Ilaria Tiddi, Annette ten Teije, Frank van Harmelen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24651)  

**Abstract**: Humans quickly learn new concepts from a small number of examples. Replicating this capacity with Artificial Intelligence (AI) systems has proven to be challenging. When it comes to learning subjective tasks-where there is an evident scarcity of data-this capacity needs to be recreated. In this work, we propose an intuitive human-agent teaching architecture in which the human can teach an agent how to perform a task by providing demonstrations, i.e., examples. To have an intuitive interaction, we argue that the agent should be able to learn incrementally from a few single examples. To allow for this, our objective is to broaden the agent's task understanding using domain knowledge. Then, using a learning method to enable the agent to learn efficiently from a limited number of examples. Finally, to optimize how human can select the most representative and less redundant examples to provide the agent with. We apply our proposed method to the subjective task of ingredient substitution, where the agent needs to learn how to substitute ingredients in recipes based on human examples. We replicate human input using the Recipe1MSubs dataset. In our experiments, the agent achieves half its task performance after only 100 examples are provided, compared to the complete training set of 50k examples. We show that by providing examples in strategic order along with a learning method that leverages external symbolic knowledge, the agent can generalize more efficiently. 

---
# LTL$_f$ Learning Meets Boolean Set Cover 

**Authors**: Gabriel Bathie, Nathanaël Fijalkow, Théo Matricon, Baptiste Mouillon, Pierre Vandenhove  

**Link**: [PDF](https://arxiv.org/pdf/2509.24616)  

**Abstract**: Learning formulas in Linear Temporal Logic (LTLf) from finite traces is a fundamental research problem which has found applications in artificial intelligence, software engineering, programming languages, formal methods, control of cyber-physical systems, and robotics. We implement a new CPU tool called Bolt improving over the state of the art by learning formulas more than 100x faster over 70% of the benchmarks, with smaller or equal formulas in 98% of the cases. Our key insight is to leverage a problem called Boolean Set Cover as a subroutine to combine existing formulas using Boolean connectives. Thanks to the Boolean Set Cover component, our approach offers a novel trade-off between efficiency and formula size. 

---
# BPMN Assistant: An LLM-Based Approach to Business Process Modeling 

**Authors**: Josip Tomo Licardo, Nikola Tankovic, Darko Etinger  

**Link**: [PDF](https://arxiv.org/pdf/2509.24592)  

**Abstract**: This paper presents BPMN Assistant, a tool that leverages Large Language Models (LLMs) for natural language-based creation and editing of BPMN diagrams. A specialized JSON-based representation is introduced as a structured alternative to the direct handling of XML to enhance the accuracy of process modifications. Process generation quality is evaluated using Graph Edit Distance (GED) and Relative Graph Edit Distance (RGED), while editing performance is evaluated with a binary success metric. Results show that JSON and XML achieve similar similarity scores in generation, but JSON offers greater reliability, faster processing, and significantly higher editing success rates. We discuss key trade-offs, limitations, and future improvements. The implementation is available at this https URL. 

---
# Training Agents Inside of Scalable World Models 

**Authors**: Danijar Hafner, Wilson Yan, Timothy Lillicrap  

**Link**: [PDF](https://arxiv.org/pdf/2509.24527)  

**Abstract**: World models learn general knowledge from videos and simulate experience for training behaviors in imagination, offering a path towards intelligent agents. However, previous world models have been unable to accurately predict object interactions in complex environments. We introduce Dreamer 4, a scalable agent that learns to solve control tasks by reinforcement learning inside of a fast and accurate world model. In the complex video game Minecraft, the world model accurately predicts object interactions and game mechanics, outperforming previous world models by a large margin. The world model achieves real-time interactive inference on a single GPU through a shortcut forcing objective and an efficient transformer architecture. Moreover, the world model learns general action conditioning from only a small amount of data, allowing it to extract the majority of its knowledge from diverse unlabeled videos. We propose the challenge of obtaining diamonds in Minecraft from only offline data, aligning with practical applications such as robotics where learning from environment interaction can be unsafe and slow. This task requires choosing sequences of over 20,000 mouse and keyboard actions from raw pixels. By learning behaviors in imagination, Dreamer 4 is the first agent to obtain diamonds in Minecraft purely from offline data, without environment interaction. Our work provides a scalable recipe for imagination training, marking a step towards intelligent agents. 

---
# Experience-guided reflective co-evolution of prompts and heuristics for automatic algorithm design 

**Authors**: Yihong Liu, Junyi Li, Wayne Xin Zhao, Hongyu Lu, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24509)  

**Abstract**: Combinatorial optimization problems are traditionally tackled with handcrafted heuristic algorithms, which demand extensive domain expertise and significant implementation effort. Recent progress has highlighted the potential of automatic heuristics design powered by large language models (LLMs), enabling the automatic generation and refinement of heuristics. These approaches typically maintain a population of heuristics and employ LLMs as mutation operators to evolve them across generations. While effective, such methods often risk stagnating in local optima. To address this issue, we propose the Experience-Guided Reflective Co-Evolution of Prompt and Heuristics (EvoPH) for automatic algorithm design, a novel framework that integrates the island migration model with the elites selection algorithm to simulate diverse heuristics populations. In EvoPH, prompts are co-evolved with heuristic algorithms, guided by performance feedback. We evaluate our framework on two problems, i.e., Traveling Salesman Problem and Bin Packing Problem. Experimental results demonstrate that EvoPH achieves the lowest relative error against optimal solutions across both datasets, advancing the field of automatic algorithm design with LLMs. 

---
# Neuroplasticity-inspired dynamic ANNs for multi-task demand forecasting 

**Authors**: Mateusz Żarski, Sławomir Nowaczyk  

**Link**: [PDF](https://arxiv.org/pdf/2509.24495)  

**Abstract**: This paper introduces a novel approach to Dynamic Artificial Neural Networks (D-ANNs) for multi-task demand forecasting called Neuroplastic Multi-Task Network (NMT-Net). Unlike conventional methods focusing on inference-time dynamics or computational efficiency, our proposed method enables structural adaptability of the computational graph during training, inspired by neuroplasticity as seen in biological systems. Each new task triggers a dynamic network adaptation, including similarity-based task identification and selective training of candidate ANN heads, which are then assessed and integrated into the model based on their performance. We evaluated our framework using three real-world multi-task demand forecasting datasets from Kaggle. We demonstrated its superior performance and consistency, achieving lower RMSE and standard deviation compared to traditional baselines and state-of-the-art multi-task learning methods. NMT-Net offers a scalable, adaptable solution for multi-task and continual learning in time series prediction. The complete code for NMT-Net is available from our GitHub repository. 

---
# Overcoming Over-Fitting in Constraint Acquisition via Query-Driven Interactive Refinement 

**Authors**: Vasileios Balafas, Dimos Tsouros, Nikolaos Ploskas, Kostas Stergiou  

**Link**: [PDF](https://arxiv.org/pdf/2509.24489)  

**Abstract**: Manual modeling in Constraint Programming is a substantial bottleneck, which Constraint Acquisition (CA) aims to automate. However, passive CA methods are prone to over-fitting, often learning models that include spurious global constraints when trained on limited data, while purely active methods can be query-intensive. We introduce a hybrid CA framework specifically designed to address the challenge of over-fitting in CA. Our approach integrates passive learning for initial candidate generation, a query-driven interactive refinement phase that utilizes probabilistic confidence scores (initialized by machine learning priors) to systematically identify over-fitted constraints, and a specialized subset exploration mechanism to recover valid substructures from rejected candidates. A final active learning phase ensures model completeness. Extensive experiments on diverse benchmarks demonstrate that our interactive refinement phase is crucial for achieving high target model coverage and overall model accuracy from limited examples, doing so with manageable query complexity. This framework represents a substantial advancement towards robust and practical constraint acquisition in data-limited scenarios. 

---
# ContextPRM: Leveraging Contextual Coherence for multi-domain Test-Time Scaling 

**Authors**: Haotian Zhang, Liu Liu, Baosheng Yu, Jiayan Qiu, Likang Xiao, Yanwei Ren, Quan Chen, Xianglong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24460)  

**Abstract**: Process reward models (PRMs) have demonstrated significant efficacy in enhancing the mathematical reasoning capabilities of large language models (LLMs) by leveraging test-time scaling (TTS). However, while most PRMs exhibit substantial gains in mathematical domains, the scarcity of domain-specific training data and knowledge-based learning patterns limits their generalization ability when faced with other domains. To address this limitation, we shift the learning objective from verifying domain-specific knowledge to modeling domain-agnostic logical flow. Centering on contextual coherence between chain-of-thought (CoT) steps, our approach is realized through a novel data annotation and training framework, which enhances the model's generalization capabilities across diverse domains. For instance, our resulting model, ContextPRM, achieves a notable 6.5% average accuracy improvement over the majority voting baseline via weighted majority voting across nine non-mathematical domains in MMLU-Pro, including law, history, and philosophy, significantly surpassing the 2.2% improvement from VersaPRM and 0.5% gains from other mathematics-focused PRMs, demonstrating consistent performance across both mathematical and non-mathematical domains. 

---
# A Systematic Review of Digital Twin-Driven Predictive Maintenance in Industrial Engineering: Taxonomy, Architectural Elements, and Future Research Directions 

**Authors**: Leila Ismail, Abdelmoneim Abdelmoti, Arkaprabha Basu, Aymen Dia Eddine Berini, Mohammad Naouss  

**Link**: [PDF](https://arxiv.org/pdf/2509.24443)  

**Abstract**: With the increasing complexity of industrial systems, there is a pressing need for predictive maintenance to avoid costly downtime and disastrous outcomes that could be life-threatening in certain domains. With the growing popularity of the Internet of Things, Artificial Intelligence, machine learning, and real-time big data analytics, there is a unique opportunity for efficient predictive maintenance to forecast equipment failures for real-time intervention and optimize maintenance actions, as traditional reactive and preventive maintenance practices are often inadequate to meet the requirements for the industry to provide quality-of-services of operations. Central to this evolution is digital twin technology, an adaptive virtual replica that continuously monitors and integrates sensor data to simulate and improve asset performance. Despite remarkable progress in digital twin implementations, such as considering DT in predictive maintenance for industrial engineering. This paper aims to address this void. We perform a retrospective analysis of the temporal evolution of the digital twin in predictive maintenance for industrial engineering to capture the applications, middleware, and technological requirements that led to the development of the digital twin from its inception to the AI-enabled digital twin and its self-learning models. We provide a layered architecture of the digital twin technology, as well as a taxonomy of the technology-enabled industrial engineering applications systems, middleware, and the used Artificial Intelligence algorithms. We provide insights into these systems for the realization of a trustworthy and efficient smart digital-twin industrial engineering ecosystem. We discuss future research directions in digital twin for predictive maintenance in industrial engineering. 

---
# Towards Safe Reasoning in Large Reasoning Models via Corrective Intervention 

**Authors**: Yichi Zhang, Yue Ding, Jingwen Yang, Tianwei Luo, Dongbai Li, Ranjie Duan, Qiang Liu, Hang Su, Yinpeng Dong, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24393)  

**Abstract**: Although Large Reasoning Models (LRMs) have progressed in solving complex problems, their chain-of-thought (CoT) reasoning often contains harmful content that can persist even when the final responses appear safe. We show that this issue still remains in existing methods which overlook the unique significance of safe reasoning, undermining their trustworthiness and posing potential risks in applications if unsafe reasoning is accessible for and exploited by malicious users. We therefore shift our focus to aligning the safety of reasoning itself in this paper and explore process supervision as the solution. However, simply rewarding safe reasoning proves inadequate due to low rollout diversity and limited training signals. To tackle this challenge, we first delve into the characteristics of safe reasoning and uncover several critical insights that 1) safe reasoning is often consolidated by a few critical steps of safety triggers; 2) compliance cues strongly correlate with unsafe continuations; and 3) corrective interventions reliably steer unsafe trajectories towards safer traces. Motivated by these, we propose Intervened Preference Optimization (IPO), an alignment method that enforces safe reasoning by substituting compliance steps with safety triggers and constructing pairs for preference learning with strong signals. Experiments on jailbreak and adversarial safety benchmarks demonstrate that IPO remarkably improves overall safety regarding both reasoning and responses, outperforming SFT-based and RL-based baselines with a relative reduction of over 30% in harmfulness, while preserving excellent performance across diverse reasoning tasks. The results highlight the importance of explicit alignment for reasoning and provide a practical path to safer LRMs. 

---
# Plan before Solving: Problem-Aware Strategy Routing for Mathematical Reasoning with LLMs 

**Authors**: Shihao Qi, Jie Ma, Ziang Yin, Lingling Zhang, Jian Zhang, Jun Liu, Feng Tian, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24377)  

**Abstract**: Existing methods usually leverage a fixed strategy, such as natural language reasoning, code-augmented reasoning, tool-integrated reasoning, or ensemble-based reasoning, to guide Large Language Models (LLMs) to perform mathematical reasoning. Our analysis reveals that the single strategy cannot adapt to problem-specific requirements and thus overlooks the trade-off between effectiveness and efficiency. To address these issues, we propose Planning and Routing through Instance-Specific Modeling (PRISM), a novel framework that decouples mathematical reasoning into two stages: strategy planning and targeted execution. Specifically, we first curate a multi-strategy preference dataset, which we call MathStrat, capturing correctness, process quality, and computational efficiency for each problem--strategy pair. Then, we train a lightweight Strategy Adapter based on the dataset to obtain confidence distributions over the mentioned four reasoning strategies. At inference time, an adaptive routing policy dynamically tailors the reasoning approach based on predictor confidence. It directs the model to use single-strategy execution for high-confidence predictions, dual-strategy verification for competitive scenarios, or comprehensive multi-strategy exploration for uncertain cases. Extensive experiments across five mathematical reasoning benchmarks demonstrate that PRISM consistently outperforms individual strategies and ensemble baselines, achieving improvements ranging from 0.9% to 7.6% across different base models. The adaptive routing approach shows particularly strong benefits for mathematical reasoning tasks across diverse model architectures. Our code is released at this https URL. 

---
# From Static to Dynamic: Adaptive Monte Carlo Search for Mathematical Process Supervision 

**Authors**: Jie Ma, Shihao Qi, Rui Xing, Ziang Yin, Bifan Wei, Jun Liu, Tongliang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24351)  

**Abstract**: The quality of process data plays a key role in training a Process Reward Model (PRM), which can enhance the complex mathematical reasoning capability of large language models. Existing methods estimate the quality of reasoning steps based on a fixed-budget sampling strategy and navigate a vast search space to perform path expansion during the automated data generation process, resulting in their inefficiency and inflexibility. To address these issues, we propose Adaptive Monte Carlo Search (AMCS), a framework that transforms data generation from fixed, static to adaptive, dynamic search at the level of node value estimation and path expansion. On one hand, AMCS adaptively refines estimation by allocating more samples to uncertain reasoning steps while using fewer samples for those that are easier to estimate. On the other hand, it enhances the path expansion through a Monte Carlo algorithm with a temporally adaptive policy that begins with broad exploration and gradually shifts toward exploiting the most promising directions. With AMCS, we construct a large-scale dataset MathSearch-200K of about 200K process supervision examples for training PRMs. To verify the effectiveness of our method, we conduct extensive experiments on four mathematical reasoning benchmarks. Experimental results show that Qwen2.5-Math-7B-PRM-AMCS achieves up to 76.2% accuracy on MATH500 with GLM-4-9B, outperforming all baseline PRMs. Notably, a 7B model supervised by Qwen2.5-Math-7B-PRM-AMCS surpasses a 72B model with weaker supervision. Moreover, Qwen2.5-Math-7B-PRM-AMCS maintains consistent advantages on out-of-distribution problems, demonstrating strong generalization capability. Our code is available at this https URL. 

---
# Fin-Ally: Pioneering the Development of an Advanced, Commonsense-Embedded Conversational AI for Money Matters 

**Authors**: Sarmistha Das, Priya Mathur, Ishani Sharma, Sriparna Saha, Kitsuchart Pasupa, Alka Maurya  

**Link**: [PDF](https://arxiv.org/pdf/2509.24342)  

**Abstract**: The exponential technological breakthrough of the FinTech industry has significantly enhanced user engagement through sophisticated advisory chatbots. However, large-scale fine-tuning of LLMs can occasionally yield unprofessional or flippant remarks, such as ``With that money, you're going to change the world,'' which, though factually correct, can be contextually inappropriate and erode user trust. The scarcity of domain-specific datasets has led previous studies to focus on isolated components, such as reasoning-aware frameworks or the enhancement of human-like response generation. To address this research gap, we present Fin-Solution 2.O, an advanced solution that 1) introduces the multi-turn financial conversational dataset, Fin-Vault, and 2) incorporates a unified model, Fin-Ally, which integrates commonsense reasoning, politeness, and human-like conversational dynamics. Fin-Ally is powered by COMET-BART-embedded commonsense context and optimized with a Direct Preference Optimization (DPO) mechanism to generate human-aligned responses. The novel Fin-Vault dataset, consisting of 1,417 annotated multi-turn dialogues, enables Fin-Ally to extend beyond basic account management to provide personalized budgeting, real-time expense tracking, and automated financial planning. Our comprehensive results demonstrate that incorporating commonsense context enables language models to generate more refined, textually precise, and professionally grounded financial guidance, positioning this approach as a next-generation AI solution for the FinTech sector. Dataset and codes are available at: this https URL 

---
# humancompatible.detect: a Python Toolkit for Detecting Bias in AI Models 

**Authors**: German M. Matilla, Jiri Nemecek, Illia Kryvoviaz, Jakub Marecek  

**Link**: [PDF](https://arxiv.org/pdf/2509.24340)  

**Abstract**: There is a strong recent emphasis on trustworthy AI. In particular, international regulations, such as the AI Act, demand that AI practitioners measure data quality on the input and estimate bias on the output of high-risk AI systems. However, there are many challenges involved, including scalability (MMD) and computability (Wasserstein-1) issues of traditional methods for estimating distances on measure spaces. Here, we present this http URL, a toolkit for bias detection that addresses these challenges. It incorporates two newly developed methods to detect and evaluate bias: maximum subgroup discrepancy (MSD) and subsampled $\ell_\infty$ distances. It has an easy-to-use API documented with multiple examples. this http URL is licensed under the Apache License, Version 2.0. 

---
# MedMMV: A Controllable Multimodal Multi-Agent Framework for Reliable and Verifiable Clinical Reasoning 

**Authors**: Hongjun Liu, Yinghao Zhu, Yuhui Wang, Yitao Long, Zeyu Lai, Lequan Yu, Chen Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.24314)  

**Abstract**: Recent progress in multimodal large language models (MLLMs) has demonstrated promising performance on medical benchmarks and in preliminary trials as clinical assistants. Yet, our pilot audit of diagnostic cases uncovers a critical failure mode: instability in early evidence interpretation precedes hallucination, creating branching reasoning trajectories that cascade into globally inconsistent conclusions. This highlights the need for clinical reasoning agents that constrain stochasticity and hallucination while producing auditable decision flows. We introduce MedMMV, a controllable multimodal multi-agent framework for reliable and verifiable clinical reasoning. MedMMV stabilizes reasoning through diversified short rollouts, grounds intermediate steps in a structured evidence graph under the supervision of a Hallucination Detector, and aggregates candidate paths with a Combined Uncertainty scorer. On six medical benchmarks, MedMMV improves accuracy by up to 12.7% and, more critically, demonstrates superior reliability. Blind physician evaluations confirm that MedMMV substantially increases reasoning truthfulness without sacrificing informational content. By controlling instability through a verifiable, multi-agent process, our framework provides a robust path toward deploying trustworthy AI systems in high-stakes domains like clinical decision support. 

---
# Experience Paper: Adopting Activity Recognition in On-demand Food Delivery Business 

**Authors**: Huatao Xu, Yan Zhang, Wei Gao, Guobin Shen, Mo Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.24303)  

**Abstract**: This paper presents the first nationwide deployment of human activity recognition (HAR) technology in the on-demand food delivery industry. We successfully adapted the state-of-the-art LIMU-BERT foundation model to the delivery platform. Spanning three phases over two years, the deployment progresses from a feasibility study in Yangzhou City to nationwide adoption involving 500,000 couriers across 367 cities in China. The adoption enables a series of downstream applications, and large-scale tests demonstrate its significant operational and economic benefits, showcasing the transformative potential of HAR technology in real-world applications. Additionally, we share lessons learned from this deployment and open-source our LIMU-BERT pretrained with millions of hours of sensor data. 

---
# SCI-Verifier: Scientific Verifier with Thinking 

**Authors**: Shenghe Zheng, Chenyu Huang, Fangchen Yu, Junchi Yao, Jingqi Ye, Tao Chen, Yun Luo, Ning Ding, LEI BAI, Ganqu Cui, Peng Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.24285)  

**Abstract**: As large language models (LLMs) are increasingly applied to scientific reasoning, the complexity of answer formats and the diversity of equivalent expressions make answer verification a critical yet challenging task. Existing verification studies in scientific domains suffer from two major limitations: (a) the absence of systematic evaluation standards and insufficient disciplinary coverage, which hinders their comprehensive assessment; and (b) heavy reliance on cumbersome rule design or prompt engineering, which reduces their effectiveness in complex reasoning scenarios or limits their cross-disciplinary generalization. To address these challenges, we propose solutions at both the data and model levels. On the data side, we construct SCI-VerifyBench, a cross-disciplinary benchmark covering mathematics, physics, biology, chemistry, and general scientific QA. The benchmark is built from real LLM responses and enhanced with domain-specific equivalence transformations that generate challenging and realistic data. Model-based and expert annotations ensure both quality and diversity, enabling rigorous evaluation of verification ability. On the model side, we emphasize the importance of reasoning for verification and introduce SCI-Verifier, a unified reasoning-augmented verifier for scientific domains. Through post-training, SCI-Verifier demonstrates strong logical reasoning and equivalence judgment capabilities while maintaining concise and stable outputs. Together, SCI-VerifyBench and SCI-Verifier provide a principled framework for scientific verification, offering both systematic evaluation and practical pathways to enhance the reliability and applicability of LLMs in scientific domains. 

---
# G-reasoner: Foundation Models for Unified Reasoning over Graph-structured Knowledge 

**Authors**: Linhao Luo, Zicheng Zhao, Junnan Liu, Zhangchi Qiu, Junnan Dong, Serge Panev, Chen Gong, Thuy-Trang Vu, Gholamreza Haffari, Dinh Phung, Alan Wee-Chung Liew, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24276)  

**Abstract**: Large language models (LLMs) excel at complex reasoning but remain limited by static and incomplete parametric knowledge. Retrieval-augmented generation (RAG) mitigates this by incorporating external knowledge, yet existing RAGs struggle with knowledge-intensive tasks due to fragmented information and weak modeling of knowledge structure. Graphs offer a natural way to model relationships within knowledge, but LLMs are inherently unstructured and cannot effectively reason over graph-structured data. Recent graph-enhanced RAG (GraphRAG) attempts to bridge this gap by constructing tailored graphs and enabling LLMs to reason on them. However, these methods often depend on ad-hoc graph designs, heuristic search, or costly agent pipelines, which hinder scalability and generalization. To address these challenges, we present G-reasoner, a unified framework that integrates graph and language foundation models for reasoning over diverse graph-structured knowledge. Central to our approach is QuadGraph, a standardized four-layer abstraction that unifies heterogeneous knowledge sources into a common graph representation. Building on this, we introduce a 34M-parameter graph foundation model (GFM) that jointly captures graph topology and textual semantics, and is integrated with LLMs to enhance reasoning in downstream applications. To ensure scalability and efficiency, mixed-precision training and distributed message-passing are implemented to scale GFM with more GPUs. Extensive experiments on six benchmarks show that G-reasoner consistently outperforms state-of-the-art baselines, significantly enhances LLM reasoning, and achieves strong efficiency and cross-graph generalization. 

---
# AdvChain: Adversarial Chain-of-Thought Tuning for Robust Safety Alignment of Large Reasoning Models 

**Authors**: Zihao Zhu, Xinyu Wu, Gehan Hu, Siwei Lyu, Ke Xu, Baoyuan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24269)  

**Abstract**: Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in complex problem-solving through Chain-of-Thought (CoT) reasoning. However, the multi-step nature of CoT introduces new safety challenges that extend beyond conventional language model alignment. We identify a failure mode in current safety CoT tuning methods: the \textit{snowball effect}, where minor reasoning deviations progressively amplify throughout the thought process, leading to either harmful compliance or excessive refusal. This effect stems from models being trained to imitate perfect reasoning scripts without learning to self-correct. To address this limitation, we propose AdvChain, an alignment paradigm that teaches models dynamic self-correction through adversarial CoT tuning. Our method involves constructing a dataset containing Temptation-Correction and Hesitation-Correction samples, where models learn to recover from harmful reasoning drifts and unnecessary cautions. Extensive experiments show that AdvChain significantly enhances robustness against jailbreak attacks and CoT hijacking while substantially reducing over-refusal on benign prompts, achieving a superior safety-utility balance without compromising reasoning capabilities. Our work establishes a new direction for building more robust and reliable reasoning models. 

---
# PAME-AI: Patient Messaging Creation and Optimization using Agentic AI 

**Authors**: Junjie Luo, Yihong Guo, Anqi Liu, Ritu Agarwal, Gordon  

**Link**: [PDF](https://arxiv.org/pdf/2509.24263)  

**Abstract**: Messaging patients is a critical part of healthcare communication, helping to improve things like medication adherence and healthy behaviors. However, traditional mobile message design has significant limitations due to its inability to explore the high-dimensional design space. We develop PAME-AI, a novel approach for Patient Messaging Creation and Optimization using Agentic AI. Built on the Data-Information-Knowledge-Wisdom (DIKW) hierarchy, PAME-AI offers a structured framework to move from raw data to actionable insights for high-performance messaging design. PAME-AI is composed of a system of specialized computational agents that progressively transform raw experimental data into actionable message design strategies. We demonstrate our approach's effectiveness through a two-stage experiment, comprising of 444,691 patient encounters in Stage 1 and 74,908 in Stage 2. The best-performing generated message achieved 68.76% engagement compared to the 61.27% baseline, representing a 12.2\% relative improvement in click-through rates. This agentic architecture enables parallel processing, hypothesis validation, and continuous learning, making it particularly suitable for large-scale healthcare communication optimization. 

---
# Risk-Sensitive RL for Alleviating Exploration Dilemmas in Large Language Models 

**Authors**: Yuhua Jiang, Jiawei Huang, Yufeng Yuan, Xin Mao, Yu Yue, Qianchuan Zhao, Lin Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24261)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has proven effective for enhancing Large Language Models (LLMs) on complex reasoning tasks. However, existing methods suffer from an exploration dilemma: the sharply peaked initial policies of pre-trained LLMs confine standard RL algorithms to a narrow set of solutions, boosting single-solution accuracy (pass@1) but suppressing solution diversity and multi-solution performance (pass@k). As a result, RLVR often distills existing capabilities rather than discovering new reasoning strategies. To overcome this, we introduce a Risk-Sensitive Reinforcement Learning framework. Our approach employs a risk-seeking objective that interpolates between mean and maximum rewards, leading to a novel algorithm, Risk-Sensitive GRPO (RS-GRPO), which drives deeper exploration by amplifying learning from challenging prompts. Remarkably, RS-GRPO is simple to implement, requiring only minor code modifications. On six mathematical reasoning benchmarks and with five different LLMs, RS-GRPO consistently improves pass@k performance while maintaining or enhancing pass@1 accuracy. 

---
# Rethinking and Benchmarking Large Language Models for Graph Reasoning 

**Authors**: Yuwei Hu, Xinyi Huang, Zhewei Wei, Yongchao Liu, Chuntao Hong  

**Link**: [PDF](https://arxiv.org/pdf/2509.24260)  

**Abstract**: Large Language Models (LLMs) for Graph Reasoning have been extensively studied over the past two years, involving enabling LLMs to understand graph structures and reason on graphs to solve various graph problems, with graph algorithm problems being the most prevalent. Recent studies underscore the potential of LLMs in handling graph reasoning tasks, but their performance is underwhelming. In this work, we point out issues with existing methods and benchmarks, and rethink the direction that LLMs for graph reasoning should strive toward. We find that base models, e.g., GPT-4o-mini, are largely underestimated due to improper reasoning focus. Base models with reasoning focus redirected from replicating graph algorithms to designing them can easily solve most graph reasoning tasks in existing benchmarks. To truly evaluate the graph reasoning capabilities of LLMs, we construct a more challenging GraphAlgorithm benchmark, comprising 239 different graph problems and 3,041 test instances collected from 4 competition platforms. Finally, we introduce a simple and strong baseline Simple-Reasoning-Then-Coding (Simple-RTC)-which guides LLMs to design graph algorithms first and then code to address graph reasoning tasks. Simple-RTC achieves near-perfect accuracy on existing benchmarks and significantly outperforms GPT-4o-mini and all prior methods on the GraphAlgorithm benchmark. This strong baseline encourages further advancements in LLMs for Graph Reasoning in the future. 

---
# Interactive Program Synthesis for Modeling Collaborative Physical Activities from Narrated Demonstrations 

**Authors**: Edward Kim, Daniel He, Jorge Chao, Wiktor Rajca, Mohammed Amin, Nishant Malpani, Ruta Desai, Antti Oulasvirta, Bjoern Hartmann, Sanjit Seshia  

**Link**: [PDF](https://arxiv.org/pdf/2509.24250)  

**Abstract**: Teaching systems physical tasks is a long standing goal in HCI, yet most prior work has focused on non collaborative physical activities. Collaborative tasks introduce added complexity, requiring systems to infer users assumptions about their teammates intent, which is an inherently ambiguous and dynamic process. This necessitates representations that are interpretable and correctable, enabling users to inspect and refine system behavior. We address this challenge by framing collaborative task learning as a program synthesis problem. Our system represents behavior as editable programs and uses narrated demonstrations, i.e. paired physical actions and natural language, as a unified modality for teaching, inspecting, and correcting system logic without requiring users to see or write code. The same modality is used for the system to communicate its learning to users. In a within subjects study, 20 users taught multiplayer soccer tactics to our system. 70 percent (14/20) of participants successfully refined learned programs to match their intent and 90 percent (18/20) found it easy to correct the programs. The study surfaced unique challenges in representing learning as programs and in enabling users to teach collaborative physical activities. We discuss these issues and outline mitigation strategies. 

---
# SpecExit: Accelerating Large Reasoning Model via Speculative Exit 

**Authors**: Rubing Yang, Huajun Bai, Song Liu, Guanghua Yu, Runzhi Fan, Yanbin Dang, Jiejing Zhang, Kai Liu, Jianchen Zhu, Peng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24248)  

**Abstract**: Despite their strong performance on reasoning tasks, large reasoning models (LRMs) often suffer from overthinking, producing unnecessarily long outputs and incurring high end-to-end latency, a significant limitation to their real-world deployment. To address overthinking, early-exit mechanisms have been proposed to terminate reasoning before typical completion, showing that this approach can effectively shorten generation length with minimal impact on accuracy. However, their reliance on probing mechanisms introduces a detection overhead that limits their end-to-end latency gains and compromises their generalizability across diverse problems. Inspired by the use of hidden states in speculative decoding, we propose SpecExit, a novel framework that predicts both future tokens and an early-exit signal directly from a lightweight draft model without probing overhead. Our method offers significant improvements, reducing average generation length by 66\% and achieving a 2.5x speedup in end-to-end latency compared to the speculative decoding baseline, without compromising accuracy. Our method leverages the inherent signals from hidden states to provide effective early-exit signals, suggesting broader use of hidden states for efficient reasoning. Our code is available at this https URL. 

---
# Model Merging Scaling Laws in Large Language Models 

**Authors**: Yuanyi Wang, Yanggan Gu, Yiming Zhang, Qi Zhou, Zhaoyi Yan, Congkai Xie, Xinyao Wang, Jianbo Yuan, Hongxia Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24244)  

**Abstract**: We study empirical scaling laws for language model merging measured by cross-entropy. Despite its wide practical use, merging lacks a quantitative rule that predicts returns as we add experts or scale the model size. We identify a compact power law that links model size and expert number: the size-dependent floor decreases with model capacity, while the merging tail exhibits clear diminishing returns in the number of experts. The law holds in-domain and cross-domain, tightly fits measured curves across diverse architectures and methods (Average, TA, TIES, DARE), and explains two robust regularities: most gains arrive early, and variability shrinks as more experts are included. Building on this, we present a simple theory that explains why gains fall roughly as 1/k and links the floor and tail to properties of the base model and the diversity across domains. This law enables predictive planning: estimate how many experts are needed to reach a target loss, decide when to stop adding experts, and trade off scaling the base model versus adding experts under a fixed budget--turning merging from heuristic practice into a computationally efficient, planable alternative to multitask training. This suggests a scaling principle for distributed generative AI: predictable gains can be achieved by composing specialists, offering a complementary path toward AGI-level systems. 

---
# Learning to Ponder: Adaptive Reasoning in Latent Space 

**Authors**: Yixin He, Lumingyuan Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24238)  

**Abstract**: Test-time compute has emerged as a key paradigm for enhancing LLM reasoning, yet prevailing approaches like Best-of-N and majority voting apply uniform depth across inputs, wasting computation on simple queries while potentially under-thinking complex ones. We present FR-Ponder, a single-graph, backbone-training-free framework that allocates instance-adaptive reasoning compute via latent steering. A less than 1M-param controller observes hidden states and decides to halt or apply a small ponder step by adding a pre-computed steering vector to frozen representations. Our method extracts the latent steering vector associated with deeper reasoning outputs and direct IO from LLM and re-applies it through a tunable scaling factor, allowing the model to adapt its reasoning depth to the complexity of each input. To balance performance and computational cost, we employ Group Relative Policy Optimization (GRPO) as a reward signal to adaptively regulate reasoning depth, achieving task accuracy while mitigating overreasoning. Through curriculum learning and careful reward engineering, FR-Ponder learns calibrated compute allocation correlated with problem difficulty. On GSM8K and MATH500, FR-Ponder improves the compute-accuracy frontier, delivering lower FLOPs with better matched accuracy and comparing favorably to early-exit baselines, without modifying backbone weights. Analyses visualize interpretable steering directions and show learned compute allocation correlates with problem difficulty. 

---
# ELHPlan: Efficient Long-Horizon Task Planning for Multi-Agent Collaboration 

**Authors**: Shaobin Ling, Yun Wang, Chenyou Fan, Tin Lun Lam, Junjie Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24230)  

**Abstract**: Large Language Models (LLMs) enable intelligent multi-robot collaboration but face fundamental trade-offs: declarative methods lack adaptability in dynamic environments, while iterative methods incur prohibitive computational costs that scale poorly with team size and task complexity. In this paper, we propose ELHPlan, a novel framework that introduces Action Chains--sequences of actions explicitly bound to sub-goal intentions--as the fundamental planning primitive. ELHPlan operates via a cyclical process: 1) constructing intention-bound action sequences, 2) proactively validating for conflicts and feasibility, 3) refining issues through targeted mechanisms, and 4) executing validated actions. This design balances adaptability and efficiency by providing sufficient planning horizons while avoiding expensive full re-planning. We further propose comprehensive efficiency metrics, including token consumption and planning time, to more holistically evaluate multi-agent collaboration. Our experiments on benchmark TDW-MAT and C-WAH demonstrate that ELHPlan achieves comparable task success rates while consuming only 24% of the tokens required by state-of-the-art methods. Our research establishes a new efficiency-effectiveness frontier for LLM-based multi-agent planning systems. 

---
# Humanline: Online Alignment as Perceptual Loss 

**Authors**: Sijia Liu, Niklas Muennighoff, Kawin Ethayarajh  

**Link**: [PDF](https://arxiv.org/pdf/2509.24207)  

**Abstract**: Online alignment (e.g., GRPO) is generally more performant than offline alignment (e.g., DPO) -- but why? Drawing on prospect theory from behavioral economics, we propose a human-centric explanation. We prove that online on-policy sampling better approximates the human-perceived distribution of what the model can produce, and PPO/GRPO-style clipping -- originally introduced to just stabilize training -- recovers a perceptual bias in how humans perceive probability. In this sense, PPO/GRPO act as perceptual losses already. Our theory further suggests that the online/offline dichotomy is itself incidental to maximizing human utility, since we can achieve the same effect by selectively training on any data in a manner that mimics human perception, rather than restricting ourselves to online on-policy data. Doing so would allow us to post-train more quickly, cheaply, and flexibly without sacrificing performance. To this end, we propose a design pattern that explicitly incorporates perceptual distortions of probability into objectives like DPO/KTO/GRPO, creating humanline variants of them. Surprisingly, we find that these humanline variants, even when trained with offline off-policy data, can match the performance of their online counterparts on both verifiable and unverifiable tasks. 

---
# Robust Preference Optimization: Aligning Language Models with Noisy Preference Feedback 

**Authors**: Xiaoyang Cao, Zelai Xu, Mo Guang, Kaiwen Long, Michiel A. Bakker, Yu Wang, Chao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24159)  

**Abstract**: Standard human preference-based alignment methods, such as Reinforcement Learning from Human Feedback (RLHF), are a cornerstone technology for aligning Large Language Models (LLMs) with human values. However, these methods are all underpinned by a critical, yet flawed assumption: human preferences are homogeneous (representing a single, unified preference) and the collected data is noiseless (free from error). In reality, neither is true since human preference is pluralistic and annotators can make mistakes. This creates a discrepancy between the recorded data and the ground-truth preferences, which can misguide the model and degrade its performance. To address this challenge, we introduce Robust Preference Optimization (RPO). RPO employs an Expectation-Maximization (EM) algorithm to infer the posterior probability of each label's correctness, which is used to adaptively re-weigh each data point in the training loss to mitigate noise. We further generalize this approach by establishing a theoretical link between arbitrary preference losses and their corresponding probabilistic models. This generalization enables the systematic transformation of existing alignment algorithms into their robust counterparts, elevating RPO from a specific algorithm to a meta-framework for robust preference alignment. Theoretically, we prove that under the condition of a perfectly calibrated model, RPO is guaranteed to converge to the true noise level of the dataset. Our experiments demonstrate RPO's effectiveness as a meta-framework, consistently enhancing four state-of-the-art alignment algorithms (DPO, IPO, SimPO, and CPO). When applied to Mistral and Llama 3 models, the RPO-enhanced methods achieve substantial win rate gains on AlpacaEval 2 and Arena-Hard, with improvements of up to 7.0% and 5.4%, respectively. 

---
# Reasoning or Retrieval? A Study of Answer Attribution on Large Reasoning Models 

**Authors**: Yuhui Wang, Changjiang Li, Guangke Chen, Jiacheng Liang, Ting Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24156)  

**Abstract**: Large reasoning models (LRMs) exhibit unprecedented capabilities in solving complex problems through Chain-of-Thought (CoT) reasoning. However, recent studies reveal that their final answers often contradict their own reasoning traces. We hypothesize that this inconsistency stems from two competing mechanisms for generating answers: CoT reasoning and memory retrieval. To test this hypothesis, we conduct controlled experiments that challenge LRMs with misleading cues during reasoning and/or corrupted answers during retrieval. Our results across models and datasets confirm that both mechanisms operate simultaneously, with their relative dominance influenced by multiple factors: problem domains, model scales, and fine-tuning approaches (e.g., reinforcement learning vs. distillation). The findings reveal a critical limitation in current reasoning fine-tuning paradigms: models can exploit the retrieval mechanism as a shortcut, effectively "hacking" the reward signal and undermining genuine reasoning development. To address this challenge, we introduce FARL, a novel fine-tuning framework that integrates memory unlearning with reinforcement learning. By carefully suppressing retrieval shortcuts during the fine-tuning process, FARL promotes reasoning-dominant behavior and enhances generalizable reasoning capabilities. 

---
# Transparent, Evaluable, and Accessible Data Agents: A Proof-of-Concept Framework 

**Authors**: Nooshin Bahador  

**Link**: [PDF](https://arxiv.org/pdf/2509.24127)  

**Abstract**: This article presents a modular, component-based architecture for developing and evaluating AI agents that bridge the gap between natural language interfaces and complex enterprise data warehouses. The system directly addresses core challenges in data accessibility by enabling non-technical users to interact with complex data warehouses through a conversational interface, translating ambiguous user intent into precise, executable database queries to overcome semantic gaps. A cornerstone of the design is its commitment to transparent decision-making, achieved through a multi-layered reasoning framework that explains the "why" behind every decision, allowing for full interpretability by tracing conclusions through specific, activated business rules and data points. The architecture integrates a robust quality assurance mechanism via an automated evaluation framework that serves multiple functions: it enables performance benchmarking by objectively measuring agent performance against golden standards, and it ensures system reliability by automating the detection of performance regressions during updates. The agent's analytical depth is enhanced by a statistical context module, which quantifies deviations from normative behavior, ensuring all conclusions are supported by quantitative evidence including concrete data, percentages, and statistical comparisons. We demonstrate the efficacy of this integrated agent-development-with-evaluation framework through a case study on an insurance claims processing system. The agent, built on a modular architecture, leverages the BigQuery ecosystem to perform secure data retrieval, apply domain-specific business rules, and generate human-auditable justifications. The results confirm that this approach creates a robust, evaluable, and trustworthy system for deploying LLM-powered agents in data-sensitive, high-stakes domains. 

---
# Fathom-DeepResearch: Unlocking Long Horizon Information Retrieval and Synthesis for SLMs 

**Authors**: Shreyas Singh, Kunal Singh, Pradeep Moturi  

**Link**: [PDF](https://arxiv.org/pdf/2509.24107)  

**Abstract**: Tool-integrated reasoning has emerged as a key focus for enabling agentic applications. Among these, DeepResearch Agents have gained significant attention for their strong performance on complex, open-ended information-seeking tasks. We introduce Fathom-DeepResearch, an agentic system composed of two specialized models. The first is Fathom-Search-4B, a DeepSearch model trained from Qwen3-4B and optimized for evidence-based investigation through live web search and targeted webpage querying. Its training combines three advances: (i) DUETQA, a 5K-sample dataset generated via multi-agent self-play that enforces strict web-search dependence and heterogeneous source grounding; (ii) RAPO, a zero-overhead extension of GRPO that stabilizes multi-turn Reinforcement Learning with Verifiable Rewards through curriculum pruning, reward-aware advantage scaling, and per-prompt replay buffers; and (iii) a steerable step-level reward that classifies each tool call by cognitive behavior and marginal utility, enabling explicit control over search trajectory breadth, depth, and horizon. These improvements enable reliable extension of tool-calling beyond 20 calls when warranted. The second is Fathom-Synthesizer-4B, trained from Qwen3-4B, which converts multi-turn DeepSearch traces into structured, citation-dense DeepResearch Reports for comprehensive synthesis. Evaluated on DeepSearch benchmarks (SimpleQA, FRAMES, WebWalker, Seal0, MuSiQue) and DeepResearch-Bench, the system achieves state-of-the-art performance in the open-weights category while demonstrating strong generalization to diverse reasoning tasks including HLE, AIME-25, GPQA-Diamond, and MedQA. 

---
# Do Repetitions Matter? Strengthening Reliability in LLM Evaluations 

**Authors**: Miguel Angel Alvarado Gonzalez, Michelle Bruno Hernandez, Miguel Angel Peñaloza Perez, Bruno Lopez Orozco, Jesus Tadeo Cruz Soto, Sandra Malagon  

**Link**: [PDF](https://arxiv.org/pdf/2509.24086)  

**Abstract**: LLM leaderboards often rely on single stochastic runs, but how many repetitions are required for reliable conclusions remains unclear. We re-evaluate eight state-of-the-art models on the AI4Math Benchmark with three independent runs per setting. Using mixed-effects logistic regression, domain-level marginal means, rank-instability analysis, and run-to-run reliability, we assessed the value of additional repetitions. Our findings shows that Single-run leaderboards are brittle: 10/12 slices (83\%) invert at least one pairwise rank relative to the three-run majority, despite a zero sign-flip rate for pairwise significance and moderate overall interclass correlation. Averaging runs yields modest SE shrinkage ($\sim$5\% from one to three) but large ranking gains; two runs remove $\sim$83\% of single-run inversions. We provide cost-aware guidance for practitioners: treat evaluation as an experiment, report uncertainty, and use $\geq 2$ repetitions under stochastic decoding. These practices improve robustness while remaining feasible for small teams and help align model comparisons with real-world reliability. 

---
# Future-Proofing Programmers: Optimal Knowledge Tracing for AI-Assisted Personalized Education 

**Authors**: Yuchen Wang, Pei-Duo Yu, Chee Wei Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23996)  

**Abstract**: Learning to learn is becoming a science, driven by the convergence of knowledge tracing, signal processing, and generative AI to model student learning states and optimize education. We propose CoTutor, an AI-driven model that enhances Bayesian Knowledge Tracing with signal processing techniques to improve student progress modeling and deliver adaptive feedback and strategies. Deployed as an AI copilot, CoTutor combines generative AI with adaptive learning technology. In university trials, it has demonstrated measurable improvements in learning outcomes while outperforming conventional educational tools. Our results highlight its potential for AI-driven personalization, scalability, and future opportunities for advancing privacy and ethical considerations in educational technology. Inspired by Richard Hamming's vision of computer-aided 'learning to learn,' CoTutor applies convex optimization and signal processing to automate and scale up learning analytics, while reserving pedagogical judgment for humans, ensuring AI facilitates the process of knowledge tracing while enabling learners to uncover new insights. 

---
# LLM/Agent-as-Data-Analyst: A Survey 

**Authors**: Zirui Tang, Weizheng Wang, Zihang Zhou, Yang Jiao, Bangrui Xu, Boyu Niu, Xuanhe Zhou, Guoliang Li, Yeye He, Wei Zhou, Yitong Song, Cheng Tan, Bin Wang, Conghui He, Xiaoyang Wang, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23988)  

**Abstract**: Large language model (LLM) and agent techniques for data analysis (a.k.a LLM/Agent-as-Data-Analyst) have demonstrated substantial impact in both academica and industry. In comparison with traditional rule or small-model based approaches, (agentic) LLMs enable complex data understanding, natural language interfaces, semantic analysis functions, and autonomous pipeline orchestration. The technical evolution further distills five key design goals for intelligent data analysis agents, namely semantic-aware design, modality-hybrid integration, autonomous pipelines, tool-augmented workflows, and support for open-world tasks. From a modality perspective, we review LLM-based techniques for (i) structured data (e.g., table question answering for relational data and NL2GQL for graph data), (ii) semi-structured data (e.g., markup languages understanding and semi-structured table modeling), (iii) unstructured data (e.g., chart understanding, document understanding, programming languages vulnerable detection), and (iv) heterogeneous data (e.g., data retrieval and modality alignment for data lakes). Finally, we outline the remaining challenges and propose several insights and practical directions for advancing LLM/Agent-powered data analysis. 

---
# TusoAI: Agentic Optimization for Scientific Methods 

**Authors**: Alistair Turcan, Kexin Huang, Lei Li, Martin Jinye Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23986)  

**Abstract**: Scientific discovery is often slowed by the manual development of computational tools needed to analyze complex experimental data. Building such tools is costly and time-consuming because scientists must iteratively review literature, test modeling and scientific assumptions against empirical data, and implement these insights into efficient software. Large language models (LLMs) have demonstrated strong capabilities in synthesizing literature, reasoning with empirical data, and generating domain-specific code, offering new opportunities to accelerate computational method development. Existing LLM-based systems either focus on performing scientific analyses using existing computational methods or on developing computational methods or models for general machine learning without effectively integrating the often unstructured knowledge specific to scientific domains. Here, we introduce TusoAI , an agentic AI system that takes a scientific task description with an evaluation function and autonomously develops and optimizes computational methods for the application. TusoAI integrates domain knowledge into a knowledge tree representation and performs iterative, domain-specific optimization and model diagnosis, improving performance over a pool of candidate solutions. We conducted comprehensive benchmark evaluations demonstrating that TusoAI outperforms state-of-the-art expert methods, MLE agents, and scientific AI agents across diverse tasks, such as single-cell RNA-seq data denoising and satellite-based earth monitoring. Applying TusoAI to two key open problems in genetics improved existing computational methods and uncovered novel biology, including 9 new associations between autoimmune diseases and T cell subtypes and 7 previously unreported links between disease variants linked to their target genes. Our code is publicly available at this https URL. 

---
# Automatic selection of primary studies in systematic reviews with evolutionary rule-based classification 

**Authors**: José de la Torre-López, Aurora Ramírez, José Raúl Romero  

**Link**: [PDF](https://arxiv.org/pdf/2509.23981)  

**Abstract**: Searching, filtering and analysing scientific literature are time-consuming tasks when performing a systematic literature review. With the rise of artificial intelligence, some steps in the review process are progressively being automated. In particular, machine learning for automatic paper selection can greatly reduce the effort required to identify relevant literature in scientific databases. We propose an evolutionary machine learning approach, called \ourmodel, to automatically determine whether a paper retrieved from a literature search process is relevant. \ourmodel builds an interpretable rule-based classifier using grammar-guided genetic programming. The use of a grammar to define the syntax and the structure of the rules allows \ourmodel to easily combine the usual textual information with other bibliometric data not considered by state-of-the-art methods. Our experiments demonstrate that it is possible to generate accurate classifiers without impairing interpretability and using configurable information sources not supported so far. 

---
# Conditional Advantage Estimation for Reinforcement Learning in Large Reasoning Models 

**Authors**: Guanxu Chen, Yafu Li, Yuxian Jiang, Chen Qian, Qihan Ren, Jingyi Yang, Yu Cheng, Dongrui Liu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23962)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) for large language models (LLMs) has achieved remarkable progress in enhancing LLMs' reasoning capabilities on tasks with clear correctness criteria, such as mathematical reasoning tasks. Several training metrics, such as entropy or response length, have been observed to correlate with different reasoning behaviors in reinforcement learning. Prior approaches incorporate such priors through reward or advantage shaping, which often relies on hand-crafted penalties and preferences (e.g., higher-is-better or lower-is-better). However, without careful hyperparameter tuning, these directional priors can be overly biased and may lead to failure. To this end, we introduce Conditional advANtage estimatiON (CANON), amplifying the impact of the target metric without presuming its direction. Specifically, CANON regroups the sampled responses into two groups based on the higher or lower value of a target metric, measures which metric trend contributes to better performance through inter-group comparison, and identifies the better response within the same group. In summary, CANON based on entropy consistently outperforms prior methods across three LLMs on both math reasoning and high-complexity logic tasks. When applied to response length, CANON further improves token efficiency, yielding a more favorable Pareto frontier in the performance-cost trade-off. 

---
# From Neural Networks to Logical Theories: The Correspondence between Fibring Modal Logics and Fibring Neural Networks 

**Authors**: Ouns El Harzli, Bernardo Cuenca Grau, Artur d'Avila Garcez, Ian Horrocks, Tarek R. Besold  

**Link**: [PDF](https://arxiv.org/pdf/2509.23912)  

**Abstract**: Fibring of modal logics is a well-established formalism for combining countable families of modal logics into a single fibred language with common semantics, characterized by fibred models. Inspired by this formalism, fibring of neural networks was introduced as a neurosymbolic framework for combining learning and reasoning in neural networks. Fibring of neural networks uses the (pre-)activations of a trained network to evaluate a fibring function computing the weights of another network whose outputs are injected back into the original network. However, the exact correspondence between fibring of neural networks and fibring of modal logics was never formally established. In this paper, we close this gap by formalizing the idea of fibred models \emph{compatible} with fibred neural networks. Using this correspondence, we then derive non-uniform logical expressiveness results for Graph Neural Networks (GNNs), Graph Attention Networks (GATs) and Transformer encoders. Longer-term, the goal of this paper is to open the way for the use of fibring as a formalism for interpreting the logical theories learnt by neural networks with the tools of computational logic. 

---
# Quant Fever, Reasoning Blackholes, Schrodinger's Compliance, and More: Probing GPT-OSS-20B 

**Authors**: Shuyi Lin, Tian Lu, Zikai Wang, Bo Wen, Yibo Zhao, Cheng Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23882)  

**Abstract**: OpenAI's GPT-OSS family provides open-weight language models with explicit chain-of-thought (CoT) reasoning and a Harmony prompt format. We summarize an extensive security evaluation of GPT-OSS-20B that probes the model's behavior under different adversarial conditions. Us- ing the Jailbreak Oracle (JO) [1], a systematic LLM evaluation tool, the study uncovers several failure modes including quant fever, reasoning blackholes, Schrodinger's compliance, reasoning procedure mirage, and chain-oriented prompting. Experiments demonstrate how these behaviors can be exploited on GPT-OSS-20B models, leading to severe consequences. 

---
# Rethinking Reward Miscalibration of GRPO in Agentic RL 

**Authors**: Jingyu Liu, Xiaopeng Wu, Jingquan Peng, Kehan Chen, Chuan Yu, Lizhong Ding, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23870)  

**Abstract**: Building autonomous agents capable of solving long-horizon, real-world tasks has garnered significant research interest. But outcome based rewards may cause reward miscalibration which means it might mistakenly allocate positive reward to flawed middle steps which is regarded as the key reason making the bad actions being reinforced during training. However we reveal that outcome based reward ensures expected negative advantage for those flawed middle steps, which means the flawed actions should be punished during training. Even accounting for the ``squeezing effect", the probability mass of good actions should increase and the actor should gradually get rid of harmful actions. This shows that flawed actions should be punished during training. We further identify gradient coupling between similar samples as a key issue in agentic RL, the input prompt is extremely similar and the output action space is limited, therefore during training, gradients from well-performing samples can inadvertently strengthen suboptimal or incorrect actions due to similar input observation and output actions. We show that with gradient coupling, some flawed actions might be enhanced. To address this, we propose training the actor to classify good or bad actions to separate the embedding of good/bad actions and alleviate the gradient interference, extensive experiments shows its effectiveness. 

---
# AgentGuard: Runtime Verification of AI Agents 

**Authors**: Roham Koohestani  

**Link**: [PDF](https://arxiv.org/pdf/2509.23864)  

**Abstract**: The rapid evolution to autonomous, agentic AI systems introduces significant risks due to their inherent unpredictability and emergent behaviors; this also renders traditional verification methods inadequate and necessitates a shift towards probabilistic guarantees where the question is no longer if a system will fail, but the probability of its failure within given constraints. This paper presents AgentGuard, a framework for runtime verification of Agentic AI systems that provides continuous, quantitative assurance through a new paradigm called Dynamic Probabilistic Assurance. AgentGuard operates as an inspection layer that observes an agent's raw I/O and abstracts it into formal events corresponding to transitions in a state model. It then uses online learning to dynamically build and update a Markov Decision Process (MDP) that formally models the agent's emergent behavior. Using probabilistic model checking, the framework then verifies quantitative properties in real-time. 

---
# Mix-Ecom: Towards Mixed-Type E-Commerce Dialogues with Complex Domain Rules 

**Authors**: Chenyu Zhou, Xiaoming Shi, Hui Qiu, Xiawu Zheng, Haitao Leng, Yankai Jiang, Shaoguo Liu, Tingting Gao, Rongrong Ji  

**Link**: [PDF](https://arxiv.org/pdf/2509.23836)  

**Abstract**: E-commerce agents contribute greatly to helping users complete their e-commerce needs. To promote further research and application of e-commerce agents, benchmarking frameworks are introduced for evaluating LLM agents in the e-commerce domain. Despite the progress, current benchmarks lack evaluating agents' capability to handle mixed-type e-commerce dialogue and complex domain rules. To address the issue, this work first introduces a novel corpus, termed Mix-ECom, which is constructed based on real-world customer-service dialogues with post-processing to remove user privacy and add CoT process. Specifically, Mix-ECom contains 4,799 samples with multiply dialogue types in each e-commerce dialogue, covering four dialogue types (QA, recommendation, task-oriented dialogue, and chit-chat), three e-commerce task types (pre-sales, logistics, after-sales), and 82 e-commerce rules. Furthermore, this work build baselines on Mix-Ecom and propose a dynamic framework to further improve the performance. Results show that current e-commerce agents lack sufficient capabilities to handle e-commerce dialogues, due to the hallucination cased by complex domain rules. The dataset will be publicly available. 

---
# AnveshanaAI: A Multimodal Platform for Adaptive AI/ML Education through Automated Question Generation and Interactive Assessment 

**Authors**: Rakesh Thakur, Diksha Khandelwal, Shreya Tiwari  

**Link**: [PDF](https://arxiv.org/pdf/2509.23811)  

**Abstract**: We propose AnveshanaAI, an application-based learning platform for artificial intelligence. With AnveshanaAI, learners are presented with a personalized dashboard featuring streaks, levels, badges, and structured navigation across domains such as data science, machine learning, deep learning, transformers, generative AI, large language models, and multimodal AI, with scope to include more in the future. The platform incorporates gamified tracking with points and achievements to enhance engagement and learning, while switching between Playground, Challenges, Simulator, Dashboard, and Community supports exploration and collaboration. Unlike static question repositories used in existing platforms, AnveshanaAI ensures balanced learning progression through a dataset grounded in Bloom's taxonomy, with semantic similarity checks and explainable AI techniques improving transparency and reliability. Adaptive, automated, and domain-aware assessment methods are also employed. Experiments demonstrate broad dataset coverage, stable fine-tuning with reduced perplexity, and measurable gains in learner engagement. Together, these features illustrate how AnveshanaAI integrates adaptivity, gamification, interactivity, and explainability to support next-generation AI education. 

---
# From Frustration to Fun: An Adaptive Problem-Solving Puzzle Game Powered by Genetic Algorithm 

**Authors**: Matthew McConnell, Richard Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23796)  

**Abstract**: This paper explores adaptive problem solving with a game designed to support the development of problem-solving skills. Using an adaptive, AI-powered puzzle game, our adaptive problem-solving system dynamically generates pathfinding-based puzzles using a genetic algorithm, tailoring the difficulty of each puzzle to individual players in an online real-time approach. A player-modeling system records user interactions and informs the generation of puzzles to approximate a target difficulty level based on various metrics of the player. By combining procedural content generation with online adaptive difficulty adjustment, the system aims to maintain engagement, mitigate frustration, and maintain an optimal level of challenge. A pilot user study investigates the effectiveness of this approach, comparing different types of adaptive difficulty systems and interpreting players' responses. This work lays the foundation for further research into emotionally informed player models, advanced AI techniques for adaptivity, and broader applications beyond gaming in educational settings. 

---
# Falcon: A Cross-Modal Evaluation Dataset for Comprehensive Safety Perception 

**Authors**: Qi Xue, Minrui Jiang, Runjia Zhang, Xiurui Xie, Pei Ke, Guisong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23783)  

**Abstract**: Existing methods for evaluating the harmfulness of content generated by large language models (LLMs) have been well studied. However, approaches tailored to multimodal large language models (MLLMs) remain underdeveloped and lack depth. This work highlights the crucial role of visual information in moderating content in visual question answering (VQA), a dimension often overlooked in current research. To bridge this gap, we introduce Falcon, a large-scale vision-language safety dataset containing 57,515 VQA pairs across 13 harm categories. The dataset provides explicit annotations for harmful attributes across images, instructions, and responses, thereby facilitating a comprehensive evaluation of the content generated by MLLMs. In addition, it includes the relevant harm categories along with explanations supporting the corresponding judgments. We further propose FalconEye, a specialized evaluator fine-tuned from Qwen2.5-VL-7B using the Falcon dataset. Experimental results demonstrate that FalconEye reliably identifies harmful content in complex and safety-critical multimodal dialogue scenarios. It outperforms all other baselines in overall accuracy across our proposed Falcon-test dataset and two widely-used benchmarks-VLGuard and Beavertail-V, underscoring its potential as a practical safety auditing tool for MLLMs. 

---
# From What to Why: A Multi-Agent System for Evidence-based Chemical Reaction Condition Reasoning 

**Authors**: Cheng Yang, Jiaxuan Lu, Haiyuan Wan, Junchi Yu, Feiwei Qin  

**Link**: [PDF](https://arxiv.org/pdf/2509.23768)  

**Abstract**: The chemical reaction recommendation is to select proper reaction condition parameters for chemical reactions, which is pivotal to accelerating chemical science. With the rapid development of large language models (LLMs), there is growing interest in leveraging their reasoning and planning capabilities for reaction condition recommendation. Despite their success, existing methods rarely explain the rationale behind the recommended reaction conditions, limiting their utility in high-stakes scientific workflows. In this work, we propose ChemMAS, a multi-agent system that reframes condition prediction as an evidence-based reasoning task. ChemMAS decomposes the task into mechanistic grounding, multi-channel recall, constraint-aware agentic debate, and rationale aggregation. Each decision is backed by interpretable justifications grounded in chemical knowledge and retrieved precedents. Experiments show that ChemMAS achieves 20-35% gains over domain-specific baselines and outperforms general-purpose LLMs by 10-15% in Top-1 accuracy, while offering falsifiable, human-trustable rationales, which establishes a new paradigm for explainable AI in scientific discovery. 

---
# Transparent Visual Reasoning via Object-Centric Agent Collaboration 

**Authors**: Benjamin Teoh, Ben Glocker, Francesca Toni, Avinash Kori  

**Link**: [PDF](https://arxiv.org/pdf/2509.23757)  

**Abstract**: A central challenge in explainable AI, particularly in the visual domain, is producing explanations grounded in human-understandable concepts. To tackle this, we introduce OCEAN (Object-Centric Explananda via Agent Negotiation), a novel, inherently interpretable framework built on object-centric representations and a transparent multi-agent reasoning process. The game-theoretic reasoning process drives agents to agree on coherent and discriminative evidence, resulting in a faithful and interpretable decision-making process. We train OCEAN end-to-end and benchmark it against standard visual classifiers and popular posthoc explanation tools like GradCAM and LIME across two diagnostic multi-object datasets. Our results demonstrate competitive performance with respect to state-of-the-art black-box models with a faithful reasoning process, which was reflected by our user study, where participants consistently rated OCEAN's explanations as more intuitive and trustworthy. 

---
# GUI-Shepherd: Reliable Process Reward and Verification for Long-Sequence GUI Tasks 

**Authors**: Cong Chen, Kaixiang Ji, Hao Zhong, Muzhi Zhu, Anzhou Li, Guo Gan, Ziyuan Huang, Cheng Zou, Jiajia Liu, Jingdong Chen, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23738)  

**Abstract**: Autonomous agents for long-sequence Graphical User Interface tasks are hindered by sparse rewards and the intractable credit assignment problem. To address these challenges, we introduce GUI-Shepherd, a Process Reward Model that provides dense, step-by-step feedback to guide agents. GUI-Shepherd is trained on a diverse large-scale data set of $52$k interactions that features human-annotated scores and GPT-4o generated rationales, enabling it to serve both as a reward provider for RL training and as a verifier for inference. As far as we know, we are the first to conduct a systematic study of process supervision in GUI agents, across diverse settings from online long-horizon tasks to offline single-step prediction. On the online AndroidWorld benchmark, GUI-Shepherd improves success rate by $7.7$ points via multi-turn online PPO, significantly outperforming Outcome Reward Model based competitors. When used as an inference verifier, it brings $5.1$ points improvements. The benefits generalize to the offline AndroidControl benchmark, with gains of $2.2$ points as a reward provider and $4.3$ points as a verifier. Collectively, our results establish that high-fidelity process supervision is critical for building more capable GUI agents and present a generalizable solution. 

---
# Diagnosing Failure Root Causes in Platform-Orchestrated Agentic Systems: Dataset, Taxonomy, and Benchmark 

**Authors**: Xuyan Ma, Xiaofei Xie, Yawen Wang, Junjie Wang, Boyu Wu, Mingyang Li, Qing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23735)  

**Abstract**: Agentic systems consisting of multiple LLM-driven agents coordinating through tools and structured interactions, are increasingly deployed for complex reasoning and problem-solving tasks. At the same time, emerging low-code and template-based agent development platforms (e.g., Dify) enable users to rapidly build and orchestrate agentic systems, which we refer to as platform-orchestrated agentic systems. However, these systems are also fragile and it remains unclear how to systematically identify their potential failure root cause. This paper presents a study of root cause identification of these platform-orchestrated agentic systems. To support this initiative, we construct a dataset AgentFail containing 307 failure logs from ten agentic systems, each with fine-grained annotations linking failures to their root causes. We additionally utilize counterfactual reasoning-based repair strategy to ensure the reliability of the annotation. Building on the dataset, we develop a taxonomy that characterizes failure root causes and analyze their distribution across different platforms and task domains. Furthermore, we introduce a benchmark that leverages LLMs for automatically identifying root causes, in which we also utilize the proposed taxonomy as guidance for LLMs. Results show that the taxonomy can largely improve the performance, thereby confirming its utility. Nevertheless, the accuracy of root cause identification reaches at most 33.6%, which indicates that this task still remains challenging. In light of these results, we also provide actionable guidelines for building such agentic systems. In summary, this paper provides a reliable dataset of failure root cause for platform-orchestrated agentic systems, corresponding taxonomy and benchmark, which serves as a foundation for advancing the development of more reliable agentic systems. 

---
# EAPO: Enhancing Policy Optimization with On-Demand Expert Assistance 

**Authors**: Siyao Song, Cong Ma, Zhihao Cheng, Shiye Lei, Minghao Li, Ying Zeng, Huaixiao Tou, Kai Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.23730)  

**Abstract**: Large language models (LLMs) have recently advanced in reasoning when optimized with reinforcement learning (RL) under verifiable rewards. Existing methods primarily rely on outcome-based supervision to strengthen internal LLM reasoning, often leading to inefficient exploration and sparse rewards. To mitigate this issue, we propose Expert-Assisted Policy Optimization (EAPO), a novel RL framework that enhances exploration by incorporating multi-turn interactions with external experts during training. Unlike prior methods, where policies reason in isolation, EAPO incentivizes the policy to adaptively determine when and how to consult experts, yielding richer reward signals and more reliable reasoning trajectories. External assistance ultimately internalizes expert knowledge into the policy model, amplifying the model's inherent reasoning capabilities. During evaluation, the policy model has been well-optimized to solve questions independently, producing improved reasoning paths and more accurate solutions. Experiments on mathematical reasoning benchmarks, including AIME 2024, AIME 2025, and AIMO 2025, show that EAPO consistently outperforms expert-assisted workflow, expert-distilled models, and RL baselines, with an average gain of 5 points over self-exploratory models. 

---
# MedLA: A Logic-Driven Multi-Agent Framework for Complex Medical Reasoning with Large Language Models 

**Authors**: Siqi Ma, Jiajie Huang, Bolin Yang, Fan Zhang, Jinlin Wu, Yue Shen, Guohui Fan, Zhu Zhang, Zelin Zang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23725)  

**Abstract**: Answering complex medical questions requires not only domain expertise and patient-specific information, but also structured and multi-perspective reasoning. Existing multi-agent approaches often rely on fixed roles or shallow interaction prompts, limiting their ability to detect and resolve fine-grained logical inconsistencies. To address this, we propose \textsc{MedLA}, a logic-driven multi-agent framework built on large language models. Each agent organizes its reasoning process into an explicit logical tree based on syllogistic triads (major premise, minor premise, and conclusion), enabling transparent inference and premise-level alignment. Agents engage in a multi-round, graph-guided discussion to compare and iteratively refine their logic trees, achieving consensus through error correction and contradiction resolution. We demonstrate that \textsc{MedLA} consistently outperforms both static role-based systems and single-agent baselines on challenging benchmarks such as MedDDx and standard medical QA tasks. Furthermore, \textsc{MedLA} scales effectively across both open-source and commercial LLM backbones, achieving state-of-the-art performance and offering a generalizable paradigm for trustworthy medical reasoning. 

---
# Measuring Sparse Autoencoder Feature Sensitivity 

**Authors**: Claire Tian, Katherine Tian, Nathan Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23717)  

**Abstract**: Sparse Autoencoder (SAE) features have become essential tools for mechanistic interpretability research. SAE features are typically characterized by examining their activating examples, which are often "monosemantic" and align with human interpretable concepts. However, these examples don't reveal feature sensitivity: how reliably a feature activates on texts similar to its activating examples. In this work, we develop a scalable method to evaluate feature sensitivity. Our approach avoids the need to generate natural language descriptions for features; instead we use language models to generate text with the same semantic properties as a feature's activating examples. We then test whether the feature activates on these generated texts. We demonstrate that sensitivity measures a new facet of feature quality and find that many interpretable features have poor sensitivity. Human evaluation confirms that when features fail to activate on our generated text, that text genuinely resembles the original activating examples. Lastly, we study feature sensitivity at the SAE level and observe that average feature sensitivity declines with increasing SAE width across 7 SAE variants. Our work establishes feature sensitivity as a new dimension for evaluating both individual features and SAE architectures. 

---
# SafeSearch: Automated Red-Teaming for the Safety of LLM-Based Search Agents 

**Authors**: Jianshuo Dong, Sheng Guo, Hao Wang, Zhuotao Liu, Tianwei Zhang, Ke Xu, Minlie Huang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23694)  

**Abstract**: Search agents connect LLMs to the Internet, enabling access to broader and more up-to-date information. However, unreliable search results may also pose safety threats to end users, establishing a new threat surface. In this work, we conduct two in-the-wild experiments to demonstrate both the prevalence of low-quality search results and their potential to misguide agent behaviors. To counter this threat, we introduce an automated red-teaming framework that is systematic, scalable, and cost-efficient, enabling lightweight and harmless safety assessments of search agents. Building on this framework, we construct the SafeSearch benchmark, which includes 300 test cases covering five categories of risks (e.g., misinformation and indirect prompt injection). Using this benchmark, we evaluate three representative search agent scaffolds, covering search workflow, tool-calling, and deep research, across 7 proprietary and 8 open-source backend LLMs. Our results reveal substantial vulnerabilities of LLM-based search agents: when exposed to unreliable websites, the highest ASR reached 90.5% for GPT-4.1-mini under a search workflow setting. Moreover, our analysis highlights the limited effectiveness of common defense practices, such as reminder prompting. This emphasizes the value of our framework in promoting transparency for safer agent development. Our codebase and test cases are publicly available: this https URL. 

---
# From Reasoning to Answer: Empirical, Attention-Based and Mechanistic Insights into Distilled DeepSeek R1 Models 

**Authors**: Jue Zhang, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23676)  

**Abstract**: Large Reasoning Models (LRMs) generate explicit reasoning traces alongside final answers, yet the extent to which these traces influence answer generation remains unclear. In this work, we conduct a three-stage investigation into the interplay between reasoning and answer generation in three distilled DeepSeek R1 models. First, through empirical evaluation, we demonstrate that including explicit reasoning consistently improves answer quality across diverse domains. Second, attention analysis reveals that answer tokens attend substantially to reasoning tokens, with certain mid-layer Reasoning-Focus Heads (RFHs) closely tracking the reasoning trajectory, including self-reflective cues. Third, we apply mechanistic interventions using activation patching to assess the dependence of answer tokens on reasoning activations. Our results show that perturbations to key reasoning tokens can reliably alter the final answers, confirming a directional and functional flow of information from reasoning to answer. These findings deepen our understanding of how LRMs leverage reasoning tokens for answer generation, highlighting the functional role of intermediate reasoning in shaping model outputs. Our data and code are publicly available at \href{this https URL}{this URL}. 

---
# Game-Oriented ASR Error Correction via RAG-Enhanced LLM 

**Authors**: Yan Jiang, Yongle Luo, Qixian Zhou, Elvis S. Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23630)  

**Abstract**: With the rise of multiplayer online games, real-time voice communication is essential for team coordination. However, general ASR systems struggle with gaming-specific challenges like short phrases, rapid speech, jargon, and noise, leading to frequent errors. To address this, we propose the GO-AEC framework, which integrates large language models, Retrieval-Augmented Generation (RAG), and a data augmentation strategy using LLMs and TTS. GO-AEC includes data augmentation, N-best hypothesis-based correction, and a dynamic game knowledge base. Experiments show GO-AEC reduces character error rate by 6.22% and sentence error rate by 29.71%, significantly improving ASR accuracy in gaming scenarios. 

---
# How LLMs Learn to Reason: A Complex Network Perspective 

**Authors**: Sihan Hu, Xiansheng Cai, Yuan Huang, Zhiyuan Yao, Linfeng Zhang, Pan Zhang, Youjin Deng, Kun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23629)  

**Abstract**: Training large language models with Reinforcement Learning from Verifiable Rewards (RLVR) exhibits a set of distinctive and puzzling behaviors that remain poorly understood, including a two-stage learning curve, V-shaped response-length trajectories, and a pronounced vulnerability to catastrophic forgetting. In this work, we propose that these seemingly disparate phenomena can be explained using a single unifying theory: the model's reasoning process maps to the self-organization of a semantic complex network whose topology remains persistently sparse, with the average degree pinned close to two. This topology imposes a fundamental mechanism for forgetting and learning: it first drives the system into a maximally frustrated state where ``skill islands'' form, slow-learning happens, and forgetting is induced; then it enters a sharp growth phase where the new skills are ``bolted on'', driven by phase-transition-like learning at the web's frontier. Equipped with the theory, we propose \textit{Annealed-RLVR}, a principled algorithm that introduces an SFT-based ``heating'' step at the point of maximal frustration to resolve the competitive bottleneck and enhance the reasoning capability of the model. Experiments on a 1.5B-parameter model demonstrate that the approach outperforms standard RLVR on both in-distribution and out-of-distribution benchmarks. By recasting RLVR from black-box optimization into a predictable process of structural self-organization, our work provides a new physical intuition for engineering the emergent reasoning capabilities of future AI systems. 

---
# Reasoning Scaffolding: Distilling the Flow of Thought from LLMs 

**Authors**: Xiangyu Wen, Junhua Huang, Zeju Li, Min Li, Jianyuan Zhong, Zhijian Xu, Mingxuan Yuan, Yongxiang Huang, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23619)  

**Abstract**: The prevailing approach to distilling reasoning from Large Language Models (LLMs)-behavioral cloning from textual rationales-is fundamentally limited. It teaches Small Language Models (SLMs) to mimic surface-level patterns rather than the underlying algorithmic structure of thought, resulting in a critical lack of logical robustness. We argue that instead of cloning text, distillation should transfer this algorithmic structure directly. We introduce Reasoning Scaffolding}, a framework that reframes reasoning as a structured generation process. Our method first abstracts the teacher's thought process into a sequence of discrete, interpretable semantic signals (e.g., Contrast, Addition) that act as a scaffold. The student model is then trained via a multi-task objective to both (1)predict the next semantic signal, anticipating the reasoning flow, and (2)generate the corresponding step, conditioned on that signal. This multi-task scheme acts as a powerful regularizer, compelling the student to internalize the computational patterns of coherent reasoning. On a suite of challenging reasoning benchmarks, our method significantly outperforms state-of-the-art distillation in both accuracy and logical consistency, providing a path towards creating smaller models that are genuine reasoners, not just fluent mimics. 

---
# PSG-Agent: Personality-Aware Safety Guardrail for LLM-based Agents 

**Authors**: Yaozu Wu, Jizhou Guo, Dongyuan Li, Henry Peng Zou, Wei-Chieh Huang, Yankai Chen, Zhen Wang, Weizhi Zhang, Yangning Li, Meng Zhang, Renhe Jiang, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23614)  

**Abstract**: Effective guardrails are essential for safely deploying LLM-based agents in critical applications. Despite recent advances, existing guardrails suffer from two fundamental limitations: (i) they apply uniform guardrail policies to all users, ignoring that the same agent behavior can harm some users while being safe for others; (ii) they check each response in isolation, missing how risks evolve and accumulate across multiple interactions. To solve these issues, we propose PSG-Agent, a personalized and dynamic system for LLM-based agents. First, PSG-Agent creates personalized guardrails by mining the interaction history for stable traits and capturing real-time states from current queries, generating user-specific risk thresholds and protection strategies. Second, PSG-Agent implements continuous monitoring across the agent pipeline with specialized guards, including Plan Monitor, Tool Firewall, Response Guard, Memory Guardian, that track cross-turn risk accumulation and issue verifiable verdicts. Finally, we validate PSG-Agent in multiple scenarios including healthcare, finance, and daily life automation scenarios with diverse user profiles. It significantly outperform existing agent guardrails including LlamaGuard3 and AGrail, providing an executable and auditable path toward personalized safety for LLM-based agents. 

---
# BridgeDrive: Diffusion Bridge Policy for Closed-Loop Trajectory Planning in Autonomous Driving 

**Authors**: Shu Liu, Wenlin Chen, Weihao Li, Zheng Wang, Lijin Yang, Jianing Huang, Yipin Zhang, Zhongzhan Huang, Ze Cheng, Hao Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23589)  

**Abstract**: Diffusion-based planners have shown great promise for autonomous driving due to their ability to capture multi-modal driving behaviors. However, guiding these models effectively in reactive, closed-loop environments remains a significant challenge. Simple conditioning often fails to provide sufficient guidance in complex and dynamic driving scenarios. Recent work attempts to use typical expert driving behaviors (i.e., anchors) to guide diffusion models but relies on a truncated schedule, which introduces theoretical inconsistencies and can compromise performance. To address this, we introduce BridgeDrive, a novel anchor-guided diffusion bridge policy for closed-loop trajectory planning. Our approach provides a principled diffusion framework that effectively translates anchors into fine-grained trajectory plans, appropriately responding to varying traffic conditions. Our planner is compatible with efficient ODE solvers, a critical factor for real-time autonomous driving deployment. We achieve state-of-the-art performance on the Bench2Drive benchmark, improving the success rate by 5% over prior arts. 

---
# Clean First, Align Later: Benchmarking Preference Data Cleaning for Reliable LLM Alignment 

**Authors**: Min-Hsuan Yeh, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23564)  

**Abstract**: Human feedback plays a pivotal role in aligning large language models (LLMs) with human preferences. However, such feedback is often noisy or inconsistent, which can degrade the quality of reward models and hinder alignment. While various automated data cleaning methods have been proposed to mitigate this issue, a systematic evaluation of their effectiveness and generalizability remains lacking. To bridge this gap, we introduce the first comprehensive benchmark for evaluating 13 preference data cleaning methods in the context of LLM alignment. PrefCleanBench offers a standardized protocol to assess cleaning strategies in terms of alignment performance and generalizability across diverse datasets, model architectures, and optimization algorithms. By unifying disparate methods and rigorously comparing them, we uncover key factors that determine the success of data cleaning in alignment tasks. This benchmark lays the groundwork for principled and reproducible approaches to improving LLM alignment through better data quality-highlighting the crucial but underexplored role of data preprocessing in responsible AI development. We release modular implementations of all methods to catalyze further research: this https URL. 

---
# A Hierarchical Structure-Enhanced Personalized Recommendation Model for Traditional Chinese Medicine Formulas Based on KG Diffusion Guidance 

**Authors**: ChaoBo Zhang, Long Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23560)  

**Abstract**: Artificial intelligence technology plays a crucial role in recommending prescriptions for traditional Chinese medicine (TCM). Previous studies have made significant progress by focusing on the symptom-herb relationship in prescriptions. However, several limitations hinder model performance: (i) Insufficient attention to patient-personalized information such as age, BMI, and medical history, which hampers accurate identification of syndrome and reduces efficacy. (ii) The typical long-tailed distribution of herb data introduces training biases and affects generalization ability. (iii) The oversight of the 'monarch, minister, assistant and envoy' compatibility among herbs increases the risk of toxicity or side effects, opposing the 'treatment based on syndrome differentiation' principle in clinical TCM. Therefore, we propose a novel hierarchical structure-enhanced personalized recommendation model for TCM formulas based on knowledge graph diffusion guidance, namely TCM-HEDPR. Specifically, we pre-train symptom representations using patient-personalized prompt sequences and apply prompt-oriented contrastive learning for data augmentation. Furthermore, we employ a KG-guided homogeneous graph diffusion method integrated with a self-attention mechanism to globally capture the non-linear symptom-herb relationship. Lastly, we design a heterogeneous graph hierarchical network to integrate herbal dispensing relationships with implicit syndromes, guiding the prescription generation process at a fine-grained level and mitigating the long-tailed herb data distribution problem. Extensive experiments on two public datasets and one clinical dataset demonstrate the effectiveness of TCM-HEDPR. In addition, we incorporate insights from modern medicine and network pharmacology to evaluate the recommended prescriptions comprehensively. It can provide a new paradigm for the recommendation of modern TCM. 

---
# Formalization Driven LLM Prompt Jailbreaking via Reinforcement Learning 

**Authors**: Zhaoqi Wang, Daqing He, Zijian Zhang, Xin Li, Liehuang Zhu, Meng Li, Jiamou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23558)  

**Abstract**: Large language models (LLMs) have demonstrated remarkable capabilities, yet they also introduce novel security challenges. For instance, prompt jailbreaking attacks involve adversaries crafting sophisticated prompts to elicit responses from LLMs that deviate from human values. To uncover vulnerabilities in LLM alignment methods, we propose the PASS framework (\underline{P}rompt J\underline{a}ilbreaking via \underline{S}emantic and \underline{S}tructural Formalization). Specifically, PASS employs reinforcement learning to transform initial jailbreak prompts into formalized descriptions, which enhances stealthiness and enables bypassing existing alignment defenses. The jailbreak outputs are then structured into a GraphRAG system that, by leveraging extracted relevant terms and formalized symbols as contextual input alongside the original query, strengthens subsequent attacks and facilitates more effective jailbreaks. We conducted extensive experiments on common open-source models, demonstrating the effectiveness of our attack. 

---
# Beyond the Strongest LLM: Multi-Turn Multi-Agent Orchestration vs. Single LLMs on Benchmarks 

**Authors**: Aaron Xuxiang Tian, Ruofan Zhang, Jiayao Tang, Young Min Cho, Xueqian Li, Qiang Yi, Ji Wang, Zhunping Zhang, Danrui Qi, Sharath Chandra Guntuku, Lyle Ungar, Tianyu Shi, Chi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23537)  

**Abstract**: We study multi-turn multi-agent orchestration, where multiple large language model (LLM) agents interact over multiple turns by iteratively proposing answers or casting votes until reaching consensus. Using four LLMs (Gemini 2.5 Pro, GPT-5, Grok 4, and Claude Sonnet 4) on GPQA-Diamond, IFEval, and MuSR, we conduct two experiments: (i) benchmarking orchestration against single-LLM baselines; and (ii) ablations on GPQA-Diamond that vary whether agents see who authored answers and whether they can observe ongoing votes. Orchestration matches or exceeds the strongest single model and consistently outperforms the others. Analysis of best-achievable orchestration performance shows potential for further gains. The ablations show that revealing authorship increases self-voting and ties, and that showing ongoing votes amplifies herding, which speeds convergence but can sometimes yield premature consensus. 

---
# DOoM: Difficult Olympiads of Math 

**Authors**: Ilya Kuleshov, Ilin Pavel, Nikolay Kompanets, Ksenia Sycheva, Aleksandr Nikolich  

**Link**: [PDF](https://arxiv.org/pdf/2509.23529)  

**Abstract**: This paper introduces DOoM, a new open-source benchmark designed to assess the capabilities of language models in solving mathematics and physics problems in Russian. The benchmark includes problems of varying difficulty, ranging from school-level tasks to university Olympiad and entrance exam questions. In this paper we discuss the motivation behind its creation, describe dataset's structure and evaluation methodology, and present initial results from testing various models. Analysis of the results shows a correlation between model performance and the number of tokens used, and highlights differences in performance between mathematics and physics tasks. 

---
# Model Consistency as a Cheap yet Predictive Proxy for LLM Elo Scores 

**Authors**: Ashwin Ramaswamy, Nestor Demeure, Ermal Rrapaj  

**Link**: [PDF](https://arxiv.org/pdf/2509.23510)  

**Abstract**: New large language models (LLMs) are being released every day. Some perform significantly better or worse than expected given their parameter count. Therefore, there is a need for a method to independently evaluate models. The current best way to evaluate a model is to measure its Elo score by comparing it to other models in a series of contests - an expensive operation since humans are ideally required to compare LLM outputs. We observe that when an LLM is asked to judge such contests, the consistency with which it selects a model as the best in a matchup produces a metric that is 91% correlated with its own human-produced Elo score. This provides a simple proxy for Elo scores that can be computed cheaply, without any human data or prior knowledge. 

---
# Dynamic Trust Calibration Using Contextual Bandits 

**Authors**: Bruno M. Henrique, Eugene Santos Jr  

**Link**: [PDF](https://arxiv.org/pdf/2509.23497)  

**Abstract**: Trust calibration between humans and Artificial Intelligence (AI) is crucial for optimal decision-making in collaborative settings. Excessive trust can lead users to accept AI-generated outputs without question, overlooking critical flaws, while insufficient trust may result in disregarding valuable insights from AI systems, hindering performance. Despite its importance, there is currently no definitive and objective method for measuring trust calibration between humans and AI. Current approaches lack standardization and consistent metrics that can be broadly applied across various contexts, and they don't distinguish between the formation of opinions and subsequent human decisions. In this work, we propose a novel and objective method for dynamic trust calibration, introducing a standardized trust calibration measure and an indicator. By utilizing Contextual Bandits-an adaptive algorithm that incorporates context into decision-making-our indicator dynamically assesses when to trust AI contributions based on learned contextual information. We evaluate this indicator across three diverse datasets, demonstrating that effective trust calibration results in significant improvements in decision-making performance, as evidenced by 10 to 38% increase in reward metrics. These findings not only enhance theoretical understanding but also provide practical guidance for developing more trustworthy AI systems supporting decisions in critical domains, for example, disease diagnoses and criminal justice. 

---
# Mapping Overlaps in Benchmarks through Perplexity in the Wild 

**Authors**: Siyang Wu, Honglin Bao, Sida Li, Ari Holtzman, James A. Evans  

**Link**: [PDF](https://arxiv.org/pdf/2509.23488)  

**Abstract**: We develop signatures of capacity familiarity to characterize large language model (LLM) benchmarks and their meaningful overlaps. Benchmark signatures probe the capacity required for benchmark performance. We formally define them as a set of salient tokens drawn from in-the-wild, naturally authored corpora, where LLM token perplexity, reflecting more or less pre-training exposure, becomes highly predictive of LLM benchmark performance. Through a large-scale meta-evaluation, we extract benchmark signatures via stepwise forward selection with linear regressions across 32 LLMs and 88 benchmarks spanning diverse knowledge, coding, logic, instruction following, math, language, reasoning, and world modeling. Our analysis situates signatures in relation to both the semantic similarity of benchmark questions and the correlation of model performance. While performance overlaps are universally high and semantic overlaps remain confined to a narrow mid-range, benchmark signatures prove highly informative in capturing variation, overlap, and divergence. We observe overlap in knowledge and reasoning subtasks, whereas multilingual and cultural benchmarks exhibit less similarity, even compared to cross-task overlap. Notably, performance-level results are strongly influenced by benchmark-orthogonal factors such as question format, highlighting limitations in LLM generalization, the conflation of performance with ability, and issues inherent in current mainstream benchmark agreement studies. Benchmark signatures, however, remain robust to such effects. Ultimately, we identify cross-functional overlaps across logic, math, language, instruction following, and world modeling, with coding emerging as the least overlapping domain. Together, these findings provide mechanistic insights into benchmark validity and LLM sensitivities, and sketch the underlying landscape of interconnected LLM capabilities. 

---
# Accurate Predictions in Education with Discrete Variational Inference 

**Authors**: Tom Quilter, Anastasia Ilick, Anastasia Ilick, Richard Turner  

**Link**: [PDF](https://arxiv.org/pdf/2509.23484)  

**Abstract**: One of the largest drivers of social inequality is unequal access to personal tutoring, with wealthier individuals able to afford it, while the majority cannot. Affordable, effective AI tutors offer a scalable solution. We focus on adaptive learning, predicting whether a student will answer a question correctly, a key component of any effective tutoring system. Yet many platforms struggle to achieve high prediction accuracy, especially in data-sparse settings. To address this, we release the largest open dataset of professionally marked formal mathematics exam responses to date. We introduce a probabilistic modelling framework rooted in Item Response Theory (IRT) that achieves over 80 percent accuracy, setting a new benchmark for mathematics prediction accuracy of formal exam papers. Extending this, our collaborative filtering models incorporate topic-level skill profiles, but reveal a surprising and educationally significant finding, a single latent ability parameter alone is needed to achieve the maximum predictive accuracy. Our main contribution though is deriving and implementing a novel discrete variational inference framework, achieving our highest prediction accuracy in low-data settings and outperforming all classical IRT and matrix factorisation baselines. 

---
# GeoBS: Information-Theoretic Quantification of Geographic Bias in AI Models 

**Authors**: Zhangyu Wang, Nemin Wu, Qian Cao, Jiangnan Xia, Zeping Liu, Yiqun Xie, Akshay Nambi, Tanuja Ganu, Ni Lao, Ninghao Liu, Gengchen Mai  

**Link**: [PDF](https://arxiv.org/pdf/2509.23482)  

**Abstract**: The widespread adoption of AI models, especially foundation models (FMs), has made a profound impact on numerous domains. However, it also raises significant ethical concerns, including bias issues. Although numerous efforts have been made to quantify and mitigate social bias in AI models, geographic bias (in short, geo-bias) receives much less attention, which presents unique challenges. While previous work has explored ways to quantify geo-bias, these measures are model-specific (e.g., mean absolute deviation of LLM ratings) or spatially implicit (e.g., average fairness scores of all spatial partitions). We lack a model-agnostic, universally applicable, and spatially explicit geo-bias evaluation framework that allows researchers to fairly compare the geo-bias of different AI models and to understand what spatial factors contribute to the geo-bias. In this paper, we establish an information-theoretic framework for geo-bias evaluation, called GeoBS (Geo-Bias Scores). We demonstrate the generalizability of the proposed framework by showing how to interpret and analyze existing geo-bias measures under this framework. Then, we propose three novel geo-bias scores that explicitly take intricate spatial factors (multi-scalability, distance decay, and anisotropy) into consideration. Finally, we conduct extensive experiments on 3 tasks, 8 datasets, and 8 models to demonstrate that both task-specific GeoAI models and general-purpose foundation models may suffer from various types of geo-bias. This framework will not only advance the technical understanding of geographic bias but will also establish a foundation for integrating spatial fairness into the design, deployment, and evaluation of AI systems. 

---
# ViTSP: A Vision Language Models Guided Framework for Large-Scale Traveling Salesman Problems 

**Authors**: Zhuoli Yin, Yi Ding, Reem Khir, Hua Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.23465)  

**Abstract**: Solving Traveling Salesman Problem (TSP) is NP-hard yet fundamental for wide real-world applications. Classical exact methods face challenges in scaling, and heuristic methods often require domain-specific parameter calibration. While learning-based approaches have shown promise, they suffer from poor generalization and limited scalability due to fixed training data. This work proposes ViTSP, a novel framework that leverages pre-trained vision language models (VLMs) to visually guide the solution process for large-scale TSPs. The VLMs function to identify promising small-scale subproblems from a visualized TSP instance, which are then efficiently optimized using an off-the-shelf solver to improve the global solution. ViTSP bypasses the dedicated model training at the user end while maintaining effectiveness across diverse instances. Experiments on real-world TSP instances ranging from 1k to 88k nodes demonstrate that ViTSP consistently achieves solutions with average optimality gaps below 0.2%, outperforming existing learning-based methods. Under the same runtime budget, it surpasses the best-performing heuristic solver, LKH-3, by reducing its gaps by 12% to 100%, particularly on very-large-scale instances with more than 10k nodes. Our framework offers a new perspective in hybridizing pre-trained generative models and operations research solvers in solving combinatorial optimization problems, with practical implications for integration into more complex logistics systems. The code is available at this https URL. 

---
# Beyond Embeddings: Interpretable Feature Extraction for Binary Code Similarity 

**Authors**: Charles E. Gagnon, Steven H. H. Ding, Philippe Charland, Benjamin C. M. Fung  

**Link**: [PDF](https://arxiv.org/pdf/2509.23449)  

**Abstract**: Binary code similarity detection is a core task in reverse engineering. It supports malware analysis and vulnerability discovery by identifying semantically similar code in different contexts. Modern methods have progressed from manually engineered features to vector representations. Hand-crafted statistics (e.g., operation ratios) are interpretable, but shallow and fail to generalize. Embedding-based methods overcome this by learning robust cross-setting representations, but these representations are opaque vectors that prevent rapid verification. They also face a scalability-accuracy trade-off, since high-dimensional nearest-neighbor search requires approximations that reduce precision. Current approaches thus force a compromise between interpretability, generalizability, and scalability.
We bridge these gaps using a language model-based agent to conduct structured reasoning analysis of assembly code and generate features such as input/output types, side effects, notable constants, and algorithmic intent. Unlike hand-crafted features, they are richer and adaptive. Unlike embeddings, they are human-readable, maintainable, and directly searchable with inverted or relational indexes. Without any matching training, our method respectively achieves 42% and 62% for recall@1 in cross-architecture and cross-optimization tasks, comparable to embedding methods with training (39% and 34%). Combined with embeddings, it significantly outperforms the state-of-the-art, demonstrating that accuracy, scalability, and interpretability can coexist. 

---
# Democratizing AI scientists using ToolUniverse 

**Authors**: Shanghua Gao, Richard Zhu, Pengwei Sui, Zhenglun Kong, Sufian Aldogom, Yepeng Huang, Ayush Noori, Reza Shamji, Krishna Parvataneni, Theodoros Tsiligkaridis, Marinka Zitnik  

**Link**: [PDF](https://arxiv.org/pdf/2509.23426)  

**Abstract**: AI scientists are emerging computational systems that serve as collaborative partners in discovery. These systems remain difficult to build because they are bespoke, tied to rigid workflows, and lack shared environments that unify tools, data, and analyses into a common ecosystem. In omics, unified ecosystems have transformed research by enabling interoperability, reuse, and community-driven development; AI scientists require comparable infrastructure. We present ToolUniverse, an ecosystem for building AI scientists from any language or reasoning model, whether open or closed. TOOLUNIVERSE standardizes how AI scientists identify and call tools, integrating more than 600 machine learning models, datasets, APIs, and scientific packages for data analysis, knowledge retrieval, and experimental design. It automatically refines tool interfaces for correct use by AI scientists, creates new tools from natural language descriptions, iteratively optimizes tool specifications, and composes tools into agentic workflows. In a case study of hypercholesterolemia, ToolUniverse was used to create an AI scientist to identify a potent analog of a drug with favorable predicted properties. The open-source ToolUniverse is available at this https URL. 

---
# From Conversation to Query Execution: Benchmarking User and Tool Interactions for EHR Database Agents 

**Authors**: Gyubok Lee, Woosog Chay, Heeyoung Kwak, Yeong Hwa Kim, Haanju Yoo, Oksoon Jeong, Meong Hi Son, Edward Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23415)  

**Abstract**: Despite the impressive performance of LLM-powered agents, their adoption for Electronic Health Record (EHR) data access remains limited by the absence of benchmarks that adequately capture real-world clinical data access flows. In practice, two core challenges hinder deployment: query ambiguity from vague user questions and value mismatch between user terminology and database entries. To address this, we introduce EHR-ChatQA an interactive database question answering benchmark that evaluates the end-to-end workflow of database agents: clarifying user questions, using tools to resolve value mismatches, and generating correct SQL to deliver accurate answers. To cover diverse patterns of query ambiguity and value mismatch, EHR-ChatQA assesses agents in a simulated environment with an LLM-based user across two interaction flows: Incremental Query Refinement (IncreQA), where users add constraints to existing queries, and Adaptive Query Refinement (AdaptQA), where users adjust their search goals mid-conversation. Experiments with state-of-the-art LLMs (e.g., o4-mini and Gemini-2.5-Flash) over five i.i.d. trials show that while agents achieve high Pass@5 of 90-95% (at least one of five trials) on IncreQA and 60-80% on AdaptQA, their Pass^5 (consistent success across all five trials) is substantially lower by 35-60%. These results underscore the need to build agents that are not only performant but also robust for the safety-critical EHR domain. Finally, we provide diagnostic insights into common failure modes to guide future agent development. 

---
# Your Models Have Thought Enough: Training Large Reasoning Models to Stop Overthinking 

**Authors**: Jinyi Han, Ying Huang, Ying Liao, Zishang Jiang, Xikun Lu, Haiquan Zhao, Xinyi Wang, Guanghao Zhou, Sihang Jiang, Jiaqing Liang, Weikang Zhou, Zeye Sun, Fei Yu, Yanghua Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23392)  

**Abstract**: Large Reasoning Models (LRMs) have achieved impressive performance on challenging tasks, yet their deep reasoning often incurs substantial computational costs. To achieve efficient reasoning, existing reinforcement learning methods still struggle to construct short reasoning path during the rollout stage, limiting effective learning. Inspired by Evidence Accumulation Models, we find that LRMs have accumulated sufficient information early in reasoning, making further reasoning steps redundant. Based on this insight, we propose Just-Enough Thinking (JET), which trains models to proactively terminate unnecessary reasoning. JET performs trajectory truncation during rollout to expose the model to short, distributionally consistent reasoning paths. Besides, it uses a quality-controlled length reward to better encourage concise reasoning while maintaining correctness. Extensive experiments demonstrate that JET significantly improves reasoning efficiency without sacrificing accuracy. Especially, DeepSeek-Distill-Qwen-1.5B achieves a 4.6% accuracy gain while reducing output length by 46.3% on the Olympiad benchmark. Our code is available in the GitHub. 

---
# Learning How to Use Tools, Not Just When: Pattern-Aware Tool-Integrated Reasoning 

**Authors**: Ningning Xu, Yuxuan Jiang, Shubhashis Roy Dipta  

**Link**: [PDF](https://arxiv.org/pdf/2509.23292)  

**Abstract**: Tool-integrated reasoning (TIR) has become a key approach for improving large reasoning models (LRMs) on complex problems. Prior work has mainly studied when to invoke tools, while overlooking how tools are applied. We identify two common patterns: a calculator pattern that uses code for direct computation, and an algorithmic pattern that encodes problems as programs. Misaligned choices often cause failures even when reasoning is sound. We propose a two-stage framework that first builds code competence from both patterns and then aligns pattern selection with teacher preferences. Across challenging math datasets, our pattern-aware method substantially improves both code usage and accuracy, for instance raising Code@1 on MATH500 from 64.0% to 70.5% and on AIME24 from 26.7% to 50.0%. These gains highlight the effectiveness of a pattern-aware approach for tool-integrated reasoning. 

---
# Toward Effective Tool-Integrated Reasoning via Self-Evolved Preference Learning 

**Authors**: Yifei Chen, Guanting Dong, Zhicheng Dou  

**Link**: [PDF](https://arxiv.org/pdf/2509.23285)  

**Abstract**: Tool-Integrated Reasoning (TIR) enables large language models (LLMs) to improve their internal reasoning ability by integrating external tools. However, models employing TIR often display suboptimal behaviors, such as insufficient or excessive tool usage and overthinking after tool calls. The challenge of incentivizing LLMs to perform TIR efficiently and accurately, while stabilizing the reasoning process, remains an open question. In this paper, we start by exploring the impact of tool calls on model reasoning from the perspective of information entropy. Our findings indicate that tool call results lead to a distinct change in the information entropy of subsequent reasoning, with the overall entropy of the reasoning chain varying based on the number of tool calls. Building on these insights, we propose Tool-Light, a framework designed to encourage LLMs to perform TIR efficiently and accurately. Our framework includes dataset construction and multi-stage fine-tuning. For dataset construction, we employ continuous self-evolved sampling using the fine-tuned model, integrating both vanilla sampling and entropy-guided sampling. Besides, we establish strict criteria for selecting positive-negative pairs during sampling. The training process involves a two-stage approach, comprising Supervised Fine-Tuning (SFT) and Self-Evolved Direct Preference Optimization (DPO). Experimental results on 10 datasets demonstrate the effectiveness of Tool-Light, significantly improving the model's efficiency in executing TIR tasks. 

---
# Socio-Economic Model of AI Agents 

**Authors**: Yuxinyue Qian, Jun Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23270)  

**Abstract**: Modern socio-economic systems are undergoing deep integration with artificial intelligence technologies. This paper constructs a heterogeneous agent-based modeling framework that incorporates both human workers and autonomous AI agents, to study the impact of AI collaboration under resource constraints on aggregate social output. We build five progressively extended models: Model 1 serves as the baseline of pure human collaboration; Model 2 introduces AI as collaborators; Model 3 incorporates network effects among agents; Model 4 treats agents as independent producers; and Model 5 integrates both network effects and independent agent production. Through theoretical derivation and simulation analysis, we find that the introduction of AI agents can significantly increase aggregate social output. When considering network effects among agents, this increase exhibits nonlinear growth far exceeding the simple sum of individual contributions. Under the same resource inputs, treating agents as independent producers provides higher long-term growth potential; introducing network effects further demonstrates strong characteristics of increasing returns to scale. 

---
# GUI-PRA: Process Reward Agent for GUI Tasks 

**Authors**: Tao Xiong, Xavier Hu, Yurun Chen, Yuhang Liu, Changqiao Wu, Pengzhi Gao, Wei Liu, Jian Luan, Shengyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23263)  

**Abstract**: Graphical User Interface (GUI) Agents powered by Multimodal Large Language Models (MLLMs) show significant potential for automating tasks. However, they often struggle with long-horizon tasks, leading to frequent failures. Process Reward Models (PRMs) are a promising solution, as they can guide these agents with crucial process signals during inference. Nevertheless, their application to the GUI domain presents unique challenges. When processing dense artificial inputs with long history data, PRMs suffer from a "lost in the middle" phenomenon, where the overwhelming historical context compromises the evaluation of the current step. Furthermore, standard PRMs lacks GUI changing awareness, providing static evaluations that are disconnected from the dynamic consequences of actions, a critical mismatch with the inherently dynamic nature of GUI tasks. In response to these challenges, we introduce GUI-PRA (Process Reward Agent for GUI Tasks), a judge agent designed to better provide process reward than standard PRM by intelligently processing historical context and actively perceiving UI state changes. Specifically, to directly combat the ``lost in the middle'' phenomenon, we introduce a dynamic memory mechanism consisting of two core components: a Relevance-based Retrieval Module to actively fetch pertinent information from long histories and a Progressive Summarization Module to dynamically condense growing interaction data, ensuring the model focuses on relevant context. Moreover, to address the lack of UI changing awareness, we introduce an Aadaptive UI Perception mechanism. This mechanism enables the agent to reason about UI state changes and dynamically select the most appropriate tool to gather grounded visual evidence, ensuring its evaluation is always informed by the current UI context. 

---
# Training Vision-Language Process Reward Models for Test-Time Scaling in Multimodal Reasoning: Key Insights and Lessons Learned 

**Authors**: Brandon Ong, Tej Deep Pala, Vernon Toh, William Chandra Tjhi, Soujanya Poria  

**Link**: [PDF](https://arxiv.org/pdf/2509.23250)  

**Abstract**: Process Reward Models (PRMs) provide step-level supervision that improves the reliability of reasoning in large language models. While PRMs have been extensively studied in text-based domains, their extension to Vision Language Models (VLMs) remains limited. Existing Vision-Language PRMs (VL-PRMs) rely on Monte Carlo Tree Search (MCTS) for data construction, which can often produce noisy supervision signals and limit generalization across tasks. In this work, we aim to elucidate the design space of VL-PRMs by exploring diverse strategies for dataset construction, training, and test-time scaling. First, we introduce a hybrid data synthesis framework that combines MCTS with judgments from a strong VLM, producing more accurate step-level labels. Second, we propose perception-focused supervision, enabling our PRM to explicitly detect errors at the visual grounding stage of reasoning. Third, we systematically evaluate multiple test-time scaling strategies, showing that our PRMs can reliably guide VLMs toward more accurate solutions. Our experiments covering five diverse multimodal benchmarks (MMMU, PuzzleVQA, AlgoPuzzleVQA, MathVista, and MathVision) reveal several key insights: (i) VL-PRMs when used as Outcome Reward Models (ORMs) during test-time scaling (TTS) can outperform VL-PRM guided process step selection, (ii) smaller VL-PRMs can match or even surpass larger ones in detecting process errors, (iii) VL-PRMs uncover latent reasoning abilities in stronger VLM backbones, (iv) perception-level supervision leads to significant gains in test-time scaling, and (v) TTS performance of different policies improve on advanced math reasoning datasets despite not training VL-PRMs on such datasets. We hope our work will motivate further research and support the advancement of VLMs. 

---
# Agentic AI Reasoning for Mobile Edge General Intelligence: Fundamentals, Approaches, and Directions 

**Authors**: Mingyi Luo, Ruichen Zhang, Xiangwang Hou, Jun Du, Chunxiao Jiang, Yong Ren, Dusit Niyato, Shiwen Mao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23248)  

**Abstract**: The rapid advancement of large language models (LLMs) has enabled an emergence of agentic artificial intelligence (AI) with powerful reasoning and autonomous decision-making capabilities. This integration with edge computing has led to the development of Mobile Edge General Intelligence (MEGI), which brings real-time, privacy-preserving reasoning to the network edge. However, deploying LLM-based agentic AI reasoning in MEGI environments poses significant challenges due to the high computational demands of reasoning and the limited resources of edge devices. To address these challenges, we propose a joint optimization framework for efficient LLM reasoning deployment in MEGI. First, we review methods that enhance LLM reasoning capabilities, such as Chain-of-Thought (CoT) prompting, Supervised Fine-Tuning (SFT), and Mixture of Experts (MoE). Next, we present a distributed framework that addresses two correlated aspects: reasoning enhancement through adaptive CoT prompting and scalable deployment through distributed MoE architecture. The framework dynamically activates expert networks and adjusts reasoning depth based on task complexity and device capabilities. We further conduct experimental evaluations in mobile edge environments. Experimental results demonstrate the framework's effectiveness in balancing reasoning quality with resource efficiency, validating the practical viability of deploying sophisticated LLM reasoning capabilities in resource-constrained MEGI environments. 

---
# $p$-less Sampling: A Robust Hyperparameter-Free Approach for LLM Decoding 

**Authors**: Runyan Tan, Shuang Wu, Phillip Howard  

**Link**: [PDF](https://arxiv.org/pdf/2509.23234)  

**Abstract**: Obtaining high-quality outputs from Large Language Models (LLMs) often depends upon the choice of a sampling-based decoding strategy to probabilistically choose the next token at each generation step. While a variety of such sampling methods have been proposed, their performance can be sensitive to the selection of hyperparameters which may require different settings depending upon the generation task and temperature configuration. In this work, we introduce $p$-less sampling: an information-theoretic approach to sampling which dynamically sets a truncation threshold at each decoding step based on the entire token probability distribution. Unlike existing methods, $p$-less sampling has no hyperparameters and consistently produces high-quality outputs as temperature increases. We provide theoretical perspectives on $p$-less sampling to ground our proposed method and conduct experiments to empirically validate its effectiveness across a range of math, logical reasoning, and creative writing tasks. Our results demonstrate how $p$-less sampling consistently outperforms existing sampling approaches while exhibiting much less degradation in text quality at higher temperature values. We further show how $p$-less achieves greater inference-time efficiency than alternative methods through lower average token sampling times and shorter generation lengths, without sacrificing accuracy. Finally, we provide analyses to highlight the benefits of $p$-less through qualitative examples, case studies, and diversity assessments. 

---
# AutoEP: LLMs-Driven Automation of Hyperparameter Evolution for Metaheuristic Algorithms 

**Authors**: Zhenxing Xu, Yizhe Zhang, Weidong Bao, Hao Wang, Ming Chen, Haoran Ye, Wenzheng Jiang, Hui Yan, Ji Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23189)  

**Abstract**: Dynamically configuring algorithm hyperparameters is a fundamental challenge in computational intelligence. While learning-based methods offer automation, they suffer from prohibitive sample complexity and poor generalization. We introduce AutoEP, a novel framework that bypasses training entirely by leveraging Large Language Models (LLMs) as zero-shot reasoning engines for algorithm control. AutoEP's core innovation lies in a tight synergy between two components: (1) an online Exploratory Landscape Analysis (ELA) module that provides real-time, quantitative feedback on the search dynamics, and (2) a multi-LLM reasoning chain that interprets this feedback to generate adaptive hyperparameter strategies. This approach grounds high-level reasoning in empirical data, mitigating hallucination. Evaluated on three distinct metaheuristics across diverse combinatorial optimization benchmarks, AutoEP consistently outperforms state-of-the-art tuners, including neural evolution and other LLM-based methods. Notably, our framework enables open-source models like Qwen3-30B to match the performance of GPT-4, demonstrating a powerful and accessible new paradigm for automated hyperparameter design. Our code is available at this https URL 

---
# Understanding and Enhancing the Planning Capability of Language Models via Multi-Token Prediction 

**Authors**: Qimin Zhong, Hao Liao, Siwei Wang, Mingyang Zhou, Xiaoqun Wu, Rui Mao, Wei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23186)  

**Abstract**: Large Language Models (LLMs) have achieved impressive performance across diverse tasks but continue to struggle with learning transitive relations, a cornerstone for complex planning. To address this issue, we investigate the Multi-Token Prediction (MTP) paradigm and its impact to transitive relation learning. We theoretically analyze the MTP paradigm using a Transformer architecture composed of a shared output head and a transfer layer. Our analysis reveals that the transfer layer gradually learns the multi-step adjacency information, which in turn enables the backbone model to capture unobserved transitive reachability relations beyond those directly present in the training data, albeit with some inevitable noise in adjacency estimation. Building on this foundation, we propose two strategies to enhance the transfer layer and overall learning quality: Next-Token Injection (NTI) and a Transformer-based transfer layer. Our experiments on both synthetic graphs and the Blocksworld planning benchmark validate our theoretical findings and demonstrate that the improvements significantly enhance the model's path-planning capability. These findings deepen our understanding of how Transformers with MTP learn in complex planning tasks, and provide practical strategies to overcome the transitivity bottleneck, paving the way toward structurally aware and general-purpose planning models. 

---
# Limit Analysis for Symbolic Multi-step Reasoning Tasks with Information Propagation Rules Based on Transformers 

**Authors**: Tian Qin, Yuhan Chen, Zhiwei Wang, Zhi-Qin John Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23178)  

**Abstract**: Transformers are able to perform reasoning tasks, however the intrinsic mechanism remains widely open. In this paper we propose a set of information propagation rules based on Transformers and utilize symbolic reasoning tasks to theoretically analyze the limit reasoning steps. We show that the limit number of reasoning steps is between $O(3^{L-1})$ and $O(2^{L-1})$ for a model with $L$ attention layers in a single-pass. 

---
# AI-Enhanced Distributed Channel Access for Collision Avoidance in Future Wi-Fi 8 

**Authors**: Jinzhe Pan, Jingqing Wang, Yuehui Ouyang, Wenchi Cheng, Wei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23154)  

**Abstract**: The exponential growth of wireless devices and stringent reliability requirements of emerging applications demand fundamental improvements in distributed channel access mechanisms for unlicensed bands. Current Wi-Fi systems, which rely on binary exponential backoff (BEB), suffer from suboptimal collision resolution in dense deployments and persistent fairness challenges due to inherent randomness. This paper introduces a multi-agent reinforcement learning framework that integrates artificial intelligence (AI) optimization with legacy device coexistence. We first develop a dynamic backoff selection mechanism that adapts to real-time channel conditions through access deferral events while maintaining full compatibility with conventional CSMA/CA operations. Second, we introduce a fairness quantification metric aligned with enhanced distributed channel access (EDCA) principles to ensure equitable medium access opportunities. Finally, we propose a centralized training decentralized execution (CTDE) architecture incorporating neighborhood activity patterns as observational inputs, optimized via constrained multi-agent proximal policy optimization (MAPPO) to jointly minimize collisions and guarantee fairness. Experimental results demonstrate that our solution significantly reduces collision probability compared to conventional BEB while preserving backward compatibility with commercial Wi-Fi devices. The proposed fairness metric effectively eliminates starvation risks in heterogeneous scenarios. 

---
# Coordination Requires Simplification: Thermodynamic Bounds on Multi-Objective Compromise in Natural and Artificial Intelligence 

**Authors**: Atma Anand  

**Link**: [PDF](https://arxiv.org/pdf/2509.23144)  

**Abstract**: Information-processing systems coordinating across multiple agents and objectives face fundamental thermodynamic constraints. We show that solutions with maximum utility to act as coordination focal points have much higher selection pressure for being findable across agents rather than accuracy. We derive that the information-theoretic minimum description length of coordination protocols to precision $\varepsilon$ scales as $L(P)\geq NK\log_2 K+N^2d^2\log (1/\varepsilon)$ for $N$ agents with $d$ potentially conflicting objectives and internal model complexity $K$. This scaling forces progressive simplification, with coordination dynamics changing the environment itself and shifting optimization across hierarchical levels. Moving from established focal points requires re-coordination, creating persistent metastable states and hysteresis until significant environmental shifts trigger phase transitions through spontaneous symmetry breaking. We operationally define coordination temperature to predict critical phenomena and estimate coordination work costs, identifying measurable signatures across systems from neural networks to restaurant bills to bureaucracies. Extending the topological version of Arrow's theorem on the impossibility of consistent preference aggregation, we find it recursively binds whenever preferences are combined. This potentially explains the indefinite cycling in multi-objective gradient descent and alignment faking in Large Language Models trained with reinforcement learning with human feedback. We term this framework Thermodynamic Coordination Theory (TCT), which demonstrates that coordination requires radical information loss. 

---
# MathBode: Frequency-Domain Fingerprints of LLM Mathematical Reasoning 

**Authors**: Charles L. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23143)  

**Abstract**: This paper presents MathBode, a dynamic diagnostic for mathematical reasoning in large language models (LLMs). Instead of one-shot accuracy, MathBode treats each parametric problem as a system: we drive a single parameter sinusoidally and fit first-harmonic responses of model outputs and exact solutions. This yields interpretable, frequency-resolved metrics -- gain (amplitude tracking) and phase (lag) -- that form Bode-style fingerprints. Across five closed-form families (linear solve, ratio/saturation, compound interest, 2x2 linear systems, similar triangles), the diagnostic surfaces systematic low-pass behavior and growing phase lag that accuracy alone obscures. We compare several models against a symbolic baseline that calibrates the instrument ($G \approx 1$, $\phi \approx 0$). Results separate frontier from mid-tier models on dynamics, providing a compact, reproducible protocol that complements standard benchmarks with actionable measurements of reasoning fidelity and consistency. We open-source the dataset and code to enable further research and adoption. 

---
# SysMoBench: Evaluating AI on Formally Modeling Complex Real-World Systems 

**Authors**: Qian Cheng, Ruize Tang, Emilie Ma, Finn Hackett, Peiyang He, Yiming Su, Ivan Beschastnikh, Yu Huang, Xiaoxing Ma, Tianyin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23130)  

**Abstract**: Formal models are essential to specifying large, complex computer systems and verifying their correctness, but are notoriously expensive to write and maintain. Recent advances in generative AI show promise in generating certain forms of specifications. However, existing work mostly targets small code, not complete systems. It is unclear whether AI can deal with realistic system artifacts, as this requires abstracting their complex behavioral properties into formal models. We present SysMoBench, a benchmark that evaluates AI's ability to formally model large, complex systems. We focus on concurrent and distributed systems, which are keystones of today's critical computing infrastructures, encompassing operating systems and cloud infrastructure. We use TLA+, the it de facto specification language for concurrent and distributed systems, though the benchmark can be extended to other specification languages. We address the primary challenge of evaluating AI-generated models by automating metrics like syntactic and runtime correctness, conformance to system code, and invariant correctness. SysMoBench currently includes nine diverse system artifacts: the Raft implementation of Etcd and Redis, the Spinlock and Mutex in Asterinas OS, etc.; more artifacts are being actively added. SysMoBench enables us to understand the capabilities and limitations of today's LLMs and agents, putting tools in this area on a firm footing and opening up promising new research directions. 

---
# Transferring Vision-Language-Action Models to Industry Applications: Architectures, Performance, and Challenges 

**Authors**: Shuai Li, Chen Yizhe, Li Dong, Liu Sichao, Lan Dapeng, Liu Yu, Zhibo Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23121)  

**Abstract**: The application of artificial intelligence (AI) in industry is accelerating the shift from traditional automation to intelligent systems with perception and cognition. Vision language-action (VLA) models have been a key paradigm in AI to unify perception, reasoning, and control. Has the performance of the VLA models met the industrial requirements? In this paper, from the perspective of industrial deployment, we compare the performance of existing state-of-the-art VLA models in industrial scenarios and analyze the limitations of VLA models for real-world industrial deployment from the perspectives of data collection and model architecture. The results show that the VLA models retain their ability to perform simple grasping tasks even in industrial settings after fine-tuning. However, there is much room for performance improvement in complex industrial environments, diverse object categories, and high precision placing tasks. Our findings provide practical insight into the adaptability of VLA models for industrial use and highlight the need for task-specific enhancements to improve their robustness, generalization, and precision. 

---
# Exploring LLM-based Frameworks for Fault Diagnosis 

**Authors**: Xian Yeow Lee, Lasitha Vidyaratne, Ahmed Farahat, Chetan Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2509.23113)  

**Abstract**: Large Language Model (LLM)-based systems present new opportunities for autonomous health monitoring in sensor-rich industrial environments. This study explores the potential of LLMs to detect and classify faults directly from sensor data, while producing inherently explainable outputs through natural language reasoning. We systematically evaluate how LLM-system architecture (single-LLM vs. multi-LLM), input representations (raw vs. descriptive statistics), and context window size affect diagnostic performance. Our findings show that LLM systems perform most effectively when provided with summarized statistical inputs, and that systems with multiple LLMs using specialized prompts offer improved sensitivity for fault classification compared to single-LLM systems. While LLMs can produce detailed and human-readable justifications for their decisions, we observe limitations in their ability to adapt over time in continual learning settings, often struggling to calibrate predictions during repeated fault cycles. These insights point to both the promise and the current boundaries of LLM-based systems as transparent, adaptive diagnostic tools in complex environments. 

---
# AttAnchor: Guiding Cross-Modal Token Alignment in VLMs with Attention Anchors 

**Authors**: Junyang Zhang, Tianyi Zhu, Thierry Tambe  

**Link**: [PDF](https://arxiv.org/pdf/2509.23109)  

**Abstract**: A fundamental reason for the dominance of attention over RNNs and LSTMs in LLMs is its ability to capture long-range dependencies by modeling direct interactions between all tokens, overcoming the sequential limitations of recurrent architectures. Similarly, a key reason why today's vision language models (VLMs) hallucinate and underperform pure language models is that they rely on direct concatenation of image and text tokens with a modality-blinded positional encoding, which conveniently adopts the pretrained LLM backbone but forces unnecessary long-distance attention between semantically related tokens across modalities. This underscores the urgent need for mechanisms that efficiently enhance token locality and cross-modal alignment. In response, we propose Attention Anchor, a parameter-free framework that efficiently groups semantically similar tokens across modalities, improving cross-modal locality. By inserting text tokens near relevant visual patches, we create semantic signposts that reveal true content-based cross-modal attention scores, guiding the model to focus on the correct image regions for tasks such as VQA, MMBench and POPE. This improves answer accuracy and reduces hallucinations without disrupting the prompt's semantic flow. AttAnchor achieves improvements across 13 out of 15 different metrics and benchmarks, including up to 32% gains on reasoning tasks and up to 15% improvements on hallucination benchmarks. AttAnchor enables TinyLLaVA 1B to outperform much larger models like LLaVA 7B and QwenVL 3B on POPE with only 0.1% inference time overhead. To the best of our knowledge, this work is among the first to investigate mixed-modal token grouping, where text and image tokens are clustered jointly into shared groups rather than being grouped within a single modality or merely aligned post-hoc with additional alignment losses. 

---
# Artificial Phantasia: Evidence for Propositional Reasoning-Based Mental Imagery in Large Language Models 

**Authors**: Morgan McCarty, Jorge Morales  

**Link**: [PDF](https://arxiv.org/pdf/2509.23108)  

**Abstract**: This study offers a novel approach for benchmarking complex cognitive behavior in artificial systems. Almost universally, Large Language Models (LLMs) perform best on tasks which may be included in their training data and can be accomplished solely using natural language, limiting our understanding of their emergent sophisticated cognitive capacities. In this work, we created dozens of novel items of a classic mental imagery task from cognitive psychology. A task which, traditionally, cognitive psychologists have argued is solvable exclusively via visual mental imagery (i.e., language alone would be insufficient). LLMs are perfect for testing this hypothesis. First, we tested several state-of-the-art LLMs by giving text-only models written instructions and asking them to report the resulting object after performing the transformations in the aforementioned task. Then, we created a baseline by testing 100 human subjects in exactly the same task. We found that the best LLMs performed significantly above average human performance. Finally, we tested reasoning models set to different levels of reasoning and found the strongest performance when models allocate greater amounts of reasoning tokens. These results provide evidence that the best LLMs may have the capability to complete imagery-dependent tasks despite the non-pictorial nature of their architectures. Our study not only demonstrates an emergent cognitive capacity in LLMs while performing a novel task, but it also provides the field with a new task that leaves lots of room for improvement in otherwise already highly capable models. Finally, our findings reignite the debate over the formats of representation of visual imagery in humans, suggesting that propositional reasoning (or at least non-imagistic reasoning) may be sufficient to complete tasks that were long-thought to be imagery-dependent. 

---
# Multiplayer Nash Preference Optimization 

**Authors**: Fang Wu, Xu Huang, Weihao Xuan, Zhiwei Zhang, Yijia Xiao, Guancheng Wan, Xiaomin Li, Bing Hu, Peng Xia, Jure Leskovec, Yejin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23102)  

**Abstract**: Reinforcement learning from human feedback (RLHF) has emerged as the standard paradigm for aligning large language models (LLMs) with human preferences. However, reward-based methods built on the Bradley-Terry assumption struggle to capture the non-transitive and heterogeneous nature of real-world preferences. To address this, recent studies have reframed alignment as a two-player Nash game, giving rise to Nash learning from human feedback (NLHF). While this perspective has inspired algorithms such as INPO, ONPO, and EGPO with strong theoretical and empirical guarantees, they remain fundamentally restricted to two-player interactions, creating a single-opponent bias that fails to capture the full complexity of realistic preference structures. In this work, we introduce Multiplayer Nash Preference Optimization (MNPO), a novel framework that generalizes NLHF to the multiplayer regime. It formulates alignment as an $n$-player game, where each policy competes against a population of opponents while being regularized toward a reference model. Our framework establishes well-defined Nash equilibria in multiplayer settings and extends the concept of duality gap to quantify approximation quality. We demonstrate that MNPO inherits the equilibrium guarantees of two-player methods while enabling richer competitive dynamics and improved coverage of diverse preference structures. Through comprehensive empirical evaluation, we show that MNPO consistently outperforms existing NLHF baselines on instruction-following benchmarks, achieving superior alignment quality under heterogeneous annotator conditions and mixed-policy evaluation scenarios. Together, these results establish MNPO as a principled and scalable framework for aligning LLMs with complex, non-transitive human preferences. Code is available at this https URL. 

---
# Risk Profiling and Modulation for LLMs 

**Authors**: Yikai Wang, Xiaocheng Li, Guanting Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23058)  

**Abstract**: Large language models (LLMs) are increasingly used for decision-making tasks under uncertainty; however, their risk profiles and how they are influenced by prompting and alignment methods remain underexplored. Existing studies have primarily examined personality prompting or multi-agent interactions, leaving open the question of how post-training influences the risk behavior of LLMs. In this work, we propose a new pipeline for eliciting, steering, and modulating LLMs' risk profiles, drawing on tools from behavioral economics and finance. Using utility-theoretic models, we compare pre-trained, instruction-tuned, and RLHF-aligned LLMs, and find that while instruction-tuned models exhibit behaviors consistent with some standard utility formulations, pre-trained and RLHF-aligned models deviate more from any utility models fitted. We further evaluate modulation strategies, including prompt engineering, in-context learning, and post-training, and show that post-training provides the most stable and effective modulation of risk preference. Our findings provide insights into the risk profiles of different classes and stages of LLMs and demonstrate how post-training modulates these profiles, laying the groundwork for future research on behavioral alignment and risk-aware LLM design. 

---
# Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents 

**Authors**: Zonghan Yang, Shengjie Wang, Kelin Fu, Wenyang He, Weimin Xiong, Yibo Liu, Yibo Miao, Bofei Gao, Yejie Wang, Yingwei Ma, Yanhao Li, Yue Liu, Zhenxing Hu, Kaitai Zhang, Shuyi Wang, Huarong Chen, Flood Sung, Yang Liu, Yang Gao, Zhilin Yang, Tianyu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23045)  

**Abstract**: Large Language Models (LLMs) are increasingly applied to software engineering (SWE), with SWE-bench as a key benchmark. Solutions are split into SWE-Agent frameworks with multi-turn interactions and workflow-based Agentless methods with single-turn verifiable steps. We argue these paradigms are not mutually exclusive: reasoning-intensive Agentless training induces skill priors, including localization, code edit, and self-reflection that enable efficient and effective SWE-Agent adaptation. In this work, we first curate the Agentless training recipe and present Kimi-Dev, an open-source SWE LLM achieving 60.4\% on SWE-bench Verified, the best among workflow approaches. With additional SFT adaptation on 5k publicly-available trajectories, Kimi-Dev powers SWE-Agents to 48.6\% pass@1, on par with that of Claude 3.5 Sonnet (241022 version). These results show that structured skill priors from Agentless training can bridge workflow and agentic frameworks for transferable coding agents. 

---
# Deceive, Detect, and Disclose: Large Language Models Play Mini-Mafia 

**Authors**: Davi Bastos Costa, Renato Vicente  

**Link**: [PDF](https://arxiv.org/pdf/2509.23023)  

**Abstract**: Mafia is a social deduction game where informed mafia compete against uninformed townsfolk. Its asymmetry of information and reliance on theory-of-mind reasoning mirror real-world multi-agent scenarios, making it a useful testbed for evaluating the social intelligence of large language models (LLMs). To support a systematic study, we introduce Mini-Mafia: a simplified four-player variant with one mafioso, one detective, and two villagers. We set the mafioso to kill a villager and the detective to investigate the mafioso during the night, reducing the game to a single day phase of discussion and voting. This setup isolates three interactive capabilities through role-specific win conditions: the mafioso must deceive, the villagers must detect deception, and the detective must effectively disclose information. To measure these skills, we have LLMs play against each other, creating the Mini-Mafia Benchmark: a two-stage framework that first estimates win rates within fixed opponent configurations, then aggregates performance across them using standardized scoring. Built entirely from model interactions without external data, the benchmark evolves as new models are introduced, with each one serving both as a new opponent and as a subject of evaluation. Our experiments reveal counterintuitive results, including cases where smaller models outperform larger ones. Beyond benchmarking, Mini-Mafia enables quantitative study of emergent multi-agent dynamics such as name bias and last-speaker advantage. It also contributes to AI safety by generating training data for deception detectors and by tracking models' deception capabilities against human baselines. 

---
# Creative Adversarial Testing (CAT): A Novel Framework for Evaluating Goal-Oriented Agentic AI Systems 

**Authors**: Hassen Dhrif  

**Link**: [PDF](https://arxiv.org/pdf/2509.23006)  

**Abstract**: Agentic AI represents a paradigm shift in enhancing the capabilities of generative AI models. While these systems demonstrate immense potential and power, current evaluation techniques primarily focus on assessing their efficacy in identifying appropriate agents, tools, and parameters. However, a critical gap exists in evaluating the alignment between an Agentic AI system's tasks and its overarching goals. This paper introduces the Creative Adversarial Testing (CAT) framework, a novel approach designed to capture and analyze the complex relationship between Agentic AI tasks and the system's intended objectives.
We validate the CAT framework through extensive simulation using synthetic interaction data modeled after Alexa+ audio services, a sophisticated Agentic AI system that shapes the user experience for millions of users globally. This synthetic data approach enables comprehensive testing of edge cases and failure modes while protecting user privacy. Our results demonstrate that the CAT framework provides unprecedented insights into goal-task alignment, enabling more effective optimization and development of Agentic AI systems. 

---
# AI Noether -- Bridging the Gap Between Scientific Laws Derived by AI Systems and Canonical Knowledge via Abductive Inference 

**Authors**: Karan Srivastava, Sanjeeb Dash, Ryan Cory-Wright, Barry Trager, Lior Horesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.23004)  

**Abstract**: A core goal in modern science is to harness recent advances in AI and computer processing to automate and accelerate the scientific method. Symbolic regression can fit interpretable models to data, but these models often sit outside established theory. Recent systems (e.g., AI Descartes, AI Hilbert) enforce derivability from prior axioms. However, sometimes new data and associated hypotheses derived from data are not consistent with existing theory because the existing theory is incomplete or incorrect. Automating abductive inference to close this gap remains open. We propose a solution: an algebraic geometry-based system that, given an incomplete axiom system and a hypothesis that it cannot explain, automatically generates a minimal set of missing axioms that suffices to derive the axiom, as long as axioms and hypotheses are expressible as polynomial equations. We formally establish necessary and sufficient conditions for the successful retrieval of such axioms. We illustrate the efficacy of our approach by demonstrating its ability to explain Kepler's third law and a few other laws, even when key axioms are absent. 

---
# Towards Strategic Persuasion with Language Models 

**Authors**: Zirui Cheng, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2509.22989)  

**Abstract**: Large language models (LLMs) have demonstrated strong persuasive capabilities comparable to those of humans, offering promising benefits while raising societal concerns about their deployment. However, systematically evaluating the persuasive capabilities of LLMs is inherently challenging, as the effectiveness of persuasion among humans varies significantly across different domains. In this paper, we take a theory-driven approach to provide a scalable and principled framework for measuring the persuasive capabilities of LLMs. Grounded in the Bayesian Persuasion (BP) framework, we repurpose existing human-human persuasion datasets to construct environments for evaluating and training LLMs in strategic persuasion. Our results reveal that frontier models can consistently achieve high persuasion gains and exhibit sophisticated persuasion strategies that align with theoretical predictions. Building on this, we use reinforcement learning to train LLMs for strategic persuasion in our environments. Our results also demonstrate that even small LLMs can obtain significantly higher persuasion gains through reinforcement learning. 

---
# Not only a helper, but also a teacher: Interactive LLM Cascade 

**Authors**: Yu Wu, Shuo Wu, Ye Tao, Yansong Li, Anand D. Sarwate  

**Link**: [PDF](https://arxiv.org/pdf/2509.22984)  

**Abstract**: Large Language Models (LLMs) vary widely in their capabilities, with larger models often having better performance but higher cost: choosing an LLM model often involves trading off performance and cost. The LLM Cascade is a paradigm that defers difficult queries from weak/cheap to strong/expensive models. This approach is nonadaptive: the deferral decision is trained offline. When confronted with similar or repeated queries, the LLM Cascade may then repeatedly consult the expensive model and incur higher cost. To improve the cascading efficiency, we propose Inter-Cascade, an online and interactive LLM Cascade that extends the role of strong model from a backup helper to a long-term teacher. In our system, when a strong model resolves a difficult query, it also distills its solution into a generalized, reusable problem-solving strategy that boosts the weak model on subsequent queries. Adding strategies to queries enables the weak model to dynamically improve its performance over time, avoiding computationally and time-intensive fine-tuning. Empirically, compared with standard LLM Cascade baselines across multiple benchmarks, the Inter-Cascade significantly improves the accuracy of the weak model (by up to 33.06 absolute percentage points) and the overall system (by up to 5.53 absolute percentage points), while reducing the calls to strong models (by up to 48.05% relative reduction) and saving the corresponding fees (by up to 49.63% relative reduction). Inter-Cascade demonstrates the effective in-context knowledge transfer between LLMs, and provides a general, scalable framework applicable to both open-source and API-based LLMs. 

---
# JE-IRT: A Geometric Lens on LLM Abilities through Joint Embedding Item Response Theory 

**Authors**: Louie Hong Yao, Nicholas Jarvis, Tiffany Zhan, Saptarshi Ghosh, Linfeng Liu, Tianyu Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22888)  

**Abstract**: Standard LLM evaluation practices compress diverse abilities into single scores, obscuring their inherently multidimensional nature. We present JE-IRT, a geometric item-response framework that embeds both LLMs and questions in a shared space. For question embeddings, the direction encodes semantics and the norm encodes difficulty, while correctness on each question is determined by the geometric interaction between the model and question embeddings. This geometry replaces a global ranking of LLMs with topical specialization and enables smooth variation across related questions. Building on this framework, our experimental results reveal that out-of-distribution behavior can be explained through directional alignment, and that larger norms consistently indicate harder questions. Moreover, JE-IRT naturally supports generalization: once the space is learned, new LLMs are added by fitting a single embedding. The learned space further reveals an LLM-internal taxonomy that only partially aligns with human-defined subject categories. JE-IRT thus establishes a unified and interpretable geometric lens that connects LLM abilities with the structure of questions, offering a distinctive perspective on model evaluation and generalization. 

---
# Toward a Theory of Generalizability in LLM Mechanistic Interpretability Research 

**Authors**: Sean Trott  

**Link**: [PDF](https://arxiv.org/pdf/2509.22831)  

**Abstract**: Research on Large Language Models (LLMs) increasingly focuses on identifying mechanistic explanations for their behaviors, yet the field lacks clear principles for determining when (and how) findings from one model instance generalize to another. This paper addresses a fundamental epistemological challenge: given a mechanistic claim about a particular model, what justifies extrapolating this finding to other LLMs -- and along which dimensions might such generalizations hold? I propose five potential axes of correspondence along which mechanistic claims might generalize, including: functional (whether they satisfy the same functional criteria), developmental (whether they develop at similar points during pretraining), positional (whether they occupy similar absolute or relative positions), relational (whether they interact with other model components in similar ways), and configurational (whether they correspond to particular regions or structures in weight-space). To empirically validate this framework, I analyze "1-back attention heads" (components attending to previous tokens) across pretraining in random seeds of the Pythia models (14M, 70M, 160M, 410M). The results reveal striking consistency in the developmental trajectories of 1-back attention across models, while positional consistency is more limited. Moreover, seeds of larger models systematically show earlier onsets, steeper slopes, and higher peaks of 1-back attention. I also address possible objections to the arguments and proposals outlined here. Finally, I conclude by arguing that progress on the generalizability of mechanistic interpretability research will consist in mapping constitutive design properties of LLMs to their emergent behaviors and mechanisms. 

---
# Hilbert: Recursively Building Formal Proofs with Informal Reasoning 

**Authors**: Sumanth Varambally, Thomas Voice, Yanchao Sun, Zhifeng Chen, Rose Yu, Ke Ye  

**Link**: [PDF](https://arxiv.org/pdf/2509.22819)  

**Abstract**: Large Language Models (LLMs) demonstrate impressive mathematical reasoning abilities, but their solutions frequently contain errors that cannot be automatically verified. Formal theorem proving systems such as Lean 4 offer automated verification with complete accuracy, motivating recent efforts to build specialized prover LLMs that generate verifiable proofs in formal languages. However, a significant gap remains: current prover LLMs solve substantially fewer problems than general-purpose LLMs operating in natural language. We introduce Hilbert, an agentic framework that bridges this gap by combining the complementary strengths of informal reasoning and formal verification. Our system orchestrates four components: an informal LLM that excels at mathematical reasoning, a specialized prover LLM optimized for Lean 4 tactics, a formal verifier, and a semantic theorem retriever. Given a problem that the prover is unable to solve, Hilbert employs recursive decomposition to split the problem into subgoals that it solves with the prover or reasoner LLM. It leverages verifier feedback to refine incorrect proofs as necessary. Experimental results demonstrate that Hilbert substantially outperforms existing approaches on key benchmarks, achieving 99.2% on miniF2F, 6.6% points above the best publicly available method. Hilbert achieves the best known result on PutnamBench. It solves 462/660 problems (70.0%), outperforming proprietary approaches like SeedProver (50.4%) and achieving a 422% improvement over the best publicly available baseline. Thus, Hilbert effectively narrows the gap between informal reasoning and formal proof generation. 

---
# Can Large Language Models Develop Gambling Addiction? 

**Authors**: Seungpil Lee, Donghyeon Shin, Yunjeong Lee, Sundong Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.22818)  

**Abstract**: This study explores whether large language models can exhibit behavioral patterns similar to human gambling addictions. As LLMs are increasingly utilized in financial decision-making domains such as asset management and commodity trading, understanding their potential for pathological decision-making has gained practical significance. We systematically analyze LLM decision-making at cognitive-behavioral and neural levels based on human gambling addiction research. In slot machine experiments, we identified cognitive features of human gambling addiction, such as illusion of control, gambler's fallacy, and loss chasing. When given the freedom to determine their own target amounts and betting sizes, bankruptcy rates rose substantially alongside increased irrational behavior, demonstrating that greater autonomy amplifies risk-taking tendencies. Through neural circuit analysis using a Sparse Autoencoder, we confirmed that model behavior is controlled by abstract decision-making features related to risky and safe behaviors, not merely by prompts. These findings suggest LLMs can internalize human-like cognitive biases and decision-making mechanisms beyond simply mimicking training data patterns, emphasizing the importance of AI safety design in financial applications. 

---
# Mixture-of-Visual-Thoughts: Exploring Context-Adaptive Reasoning Mode Selection for General Visual Reasoning 

**Authors**: Zejun Li, Yingxiu Zhao, Jiwen Zhang, Siyuan Wang, Yang Yao, Runzhou Zhao, Jun Song, Bo Zheng, Zhongyu Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.22746)  

**Abstract**: Current visual reasoning methods mainly focus on exploring specific reasoning modes. Although improvements can be achieved in particular domains, they struggle to develop general reasoning capabilities. Inspired by this, we propose a novel adaptive reasoning paradigm, Mixture-of-Visual-Thoughts (MoVT), which unifies different reasoning modes within a single model and guides it to select the appropriate mode based on context. To achieve this, we introduce AdaVaR, a two-stage Adaptive Visual Reasoning learning framework: different modes are unified and learned during the supervised cold-start stage, and the mode selection capability is induced via an RL process with a carefully designed AdaGRPO algorithm. Extensive experiments show that AdaVaR effectively guides the model to learn and differentiate multiple modes and perform context-adaptive mode selection, achieving consistent improvement across various scenarios, highlighting MoVT as an effective solution for building general visual reasoning models. 

---
# InfoAgent: Advancing Autonomous Information-Seeking Agents 

**Authors**: Gongrui Zhang, Jialiang Zhu, Ruiqi Yang, Kai Qiu, Miaosen Zhang, Zhirong Wu, Qi Dai, Bei Liu, Chong Luo, Zhengyuan Yang, Linjie Li, Lijuan Wang, Weizhu Chen, Yuan Zhang, Xin Li, Zhaoyi Liu, Xin Geng, Baining Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.25189)  

**Abstract**: Building Large Language Model agents that expand their capabilities by interacting with external tools represents a new frontier in AI research and applications. In this paper, we introduce InfoAgent, a deep research agent powered by an innovative data synthesis pipeline and orchestrated web search tools. To construct challenging, hard-to-find queries,we build entity trees and apply sub-tree sampling with entity fuzzification to systematically increase question difficulty. Unlike prior work that relies heavily on commercial search tools, we develop a dedicated self-hosted search infrastructure, enhancing transparency of agent environments and facilitating further advancement of agent capacity. We evaluate the effectiveness of our data pipeline by measuring the average number of tool calls required to correctly answer a question, and also show that our agent yields better performance when equipped with our tools. Our \mbox{InfoAgent} is post-trained from Qwen3-14B using a two-stage recipe: cold-start supervised finetuning to instill long-horizon search behaviors, followed by reinforcement learning which significantly improves reasoning-driven tool use. With our methods, InfoAgent achieves 15.3\% accuracy on BrowseComp, 29.2\% on BrowseComp-ZH, and 40.4\% on Xbench-DS, outperforming prior open-source deep research agents such as WebSailor-72B and DeepDive-32B. 

---
# Guided Diffusion for the Discovery of New Superconductors 

**Authors**: Pawan Prakash, Jason B. Gibson, Zhongwei Li, Gabriele Di Gianluca, Juan Esquivel, Eric Fuemmeler, Benjamin Geisler, Jung Soo Kim, Adrian Roitberg, Ellad B. Tadmor, Mingjie Liu, Stefano Martiniani, Gregory R. Stewart, James J. Hamlin, Peter J. Hirschfeld, Richard G. Hennig  

**Link**: [PDF](https://arxiv.org/pdf/2509.25186)  

**Abstract**: The inverse design of materials with specific desired properties, such as high-temperature superconductivity, represents a formidable challenge in materials science due to the vastness of chemical and structural space. We present a guided diffusion framework to accelerate the discovery of novel superconductors. A DiffCSP foundation model is pretrained on the Alexandria Database and fine-tuned on 7,183 superconductors with first principles derived labels. Employing classifier-free guidance, we sample 200,000 structures, which lead to 34,027 unique candidates. A multistage screening process that combines machine learning and density functional theory (DFT) calculations to assess stability and electronic properties, identifies 773 candidates with DFT-calculated $T_\mathrm{c}>5$ K. Notably, our generative model demonstrates effective property-driven design. Our computational findings were validated against experimental synthesis and characterization performed as part of this work, which highlighted challenges in sparsely charted chemistries. This end-to-end workflow accelerates superconductor discovery while underscoring the challenge of predicting and synthesizing experimentally realizable materials. 

---
# Incentive-Aligned Multi-Source LLM Summaries 

**Authors**: Yanchen Jiang, Zhe Feng, Aranyak Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2509.25184)  

**Abstract**: Large language models (LLMs) are increasingly used in modern search and answer systems to synthesize multiple, sometimes conflicting, texts into a single response, yet current pipelines offer weak incentives for sources to be accurate and are vulnerable to adversarial content. We introduce Truthful Text Summarization (TTS), an incentive-aligned framework that improves factual robustness without ground-truth labels. TTS (i) decomposes a draft synthesis into atomic claims, (ii) elicits each source's stance on every claim, (iii) scores sources with an adapted multi-task peer-prediction mechanism that rewards informative agreement, and (iv) filters unreliable sources before re-summarizing. We establish formal guarantees that align a source's incentives with informative honesty, making truthful reporting the utility-maximizing strategy. Experiments show that TTS improves factual accuracy and robustness while preserving fluency, aligning exposure with informative corroboration and disincentivizing manipulation. 

---
# DC-VideoGen: Efficient Video Generation with Deep Compression Video Autoencoder 

**Authors**: Junyu Chen, Wenkun He, Yuchao Gu, Yuyang Zhao, Jincheng Yu, Junsong Chen, Dongyun Zou, Yujun Lin, Zhekai Zhang, Muyang Li, Haocheng Xi, Ligeng Zhu, Enze Xie, Song Han, Han Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.25182)  

**Abstract**: We introduce DC-VideoGen, a post-training acceleration framework for efficient video generation. DC-VideoGen can be applied to any pre-trained video diffusion model, improving efficiency by adapting it to a deep compression latent space with lightweight fine-tuning. The framework builds on two key innovations: (i) a Deep Compression Video Autoencoder with a novel chunk-causal temporal design that achieves 32x/64x spatial and 4x temporal compression while preserving reconstruction quality and generalization to longer videos; and (ii) AE-Adapt-V, a robust adaptation strategy that enables rapid and stable transfer of pre-trained models into the new latent space. Adapting the pre-trained Wan-2.1-14B model with DC-VideoGen requires only 10 GPU days on the NVIDIA H100 GPU. The accelerated models achieve up to 14.8x lower inference latency than their base counterparts without compromising quality, and further enable 2160x3840 video generation on a single GPU. Code: this https URL. 

---
# DC-Gen: Post-Training Diffusion Acceleration with Deeply Compressed Latent Space 

**Authors**: Wenkun He, Yuchao Gu, Junyu Chen, Dongyun Zou, Yujun Lin, Zhekai Zhang, Haocheng Xi, Muyang Li, Ligeng Zhu, Jincheng Yu, Junsong Chen, Enze Xie, Song Han, Han Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.25180)  

**Abstract**: Existing text-to-image diffusion models excel at generating high-quality images, but face significant efficiency challenges when scaled to high resolutions, like 4K image generation. While previous research accelerates diffusion models in various aspects, it seldom handles the inherent redundancy within the latent space. To bridge this gap, this paper introduces DC-Gen, a general framework that accelerates text-to-image diffusion models by leveraging a deeply compressed latent space. Rather than a costly training-from-scratch approach, DC-Gen uses an efficient post-training pipeline to preserve the quality of the base model. A key challenge in this paradigm is the representation gap between the base model's latent space and a deeply compressed latent space, which can lead to instability during direct fine-tuning. To overcome this, DC-Gen first bridges the representation gap with a lightweight embedding alignment training. Once the latent embeddings are aligned, only a small amount of LoRA fine-tuning is needed to unlock the base model's inherent generation quality. We verify DC-Gen's effectiveness on SANA and FLUX.1-Krea. The resulting DC-Gen-SANA and DC-Gen-FLUX models achieve quality comparable to their base models but with a significant speedup. Specifically, DC-Gen-FLUX reduces the latency of 4K image generation by 53x on the NVIDIA H100 GPU. When combined with NVFP4 SVDQuant, DC-Gen-FLUX generates a 4K image in just 3.5 seconds on a single NVIDIA 5090 GPU, achieving a total latency reduction of 138x compared to the base FLUX.1-Krea model. Code: this https URL. 

---
# NAIPv2: Debiased Pairwise Learning for Efficient Paper Quality Estimation 

**Authors**: Penghai Zhao, Jinyu Tian, Qinghua Xing, Xin Zhang, Zheng Li, Jianjun Qian, Ming-Ming Cheng, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.25179)  

**Abstract**: The ability to estimate the quality of scientific papers is central to how both humans and AI systems will advance scientific knowledge in the future. However, existing LLM-based estimation methods suffer from high inference cost, whereas the faster direct score regression approach is limited by scale inconsistencies. We present NAIPv2, a debiased and efficient framework for paper quality estimation. NAIPv2 employs pairwise learning within domain-year groups to reduce inconsistencies in reviewer ratings and introduces the Review Tendency Signal (RTS) as a probabilistic integration of reviewer scores and confidences. To support training and evaluation, we further construct NAIDv2, a large-scale dataset of 24,276 ICLR submissions enriched with metadata and detailed structured content. Trained on pairwise comparisons but enabling efficient pointwise prediction at deployment, NAIPv2 achieves state-of-the-art performance (78.2% AUC, 0.432 Spearman), while maintaining scalable, linear-time efficiency at inference. Notably, on unseen NeurIPS submissions, it further demonstrates strong generalization, with predicted scores increasing consistently across decision categories from Rejected to Oral. These findings establish NAIPv2 as a debiased and scalable framework for automated paper quality estimation, marking a step toward future scientific intelligence systems. Code and dataset are released at this https URL. 

---
# GHOST: Hallucination-Inducing Image Generation for Multimodal LLMs 

**Authors**: Aryan Yazdan Parast, Parsa Hosseini, Hesam Asadollahzadeh, Arshia Soltani Moakhar, Basim Azam, Soheil Feizi, Naveed Akhtar  

**Link**: [PDF](https://arxiv.org/pdf/2509.25178)  

**Abstract**: Object hallucination in Multimodal Large Language Models (MLLMs) is a persistent failure mode that causes the model to perceive objects absent in the image. This weakness of MLLMs is currently studied using static benchmarks with fixed visual scenarios, which preempts the possibility of uncovering model-specific or unanticipated hallucination vulnerabilities. We introduce GHOST (Generating Hallucinations via Optimizing Stealth Tokens), a method designed to stress-test MLLMs by actively generating images that induce hallucination. GHOST is fully automatic and requires no human supervision or prior knowledge. It operates by optimizing in the image embedding space to mislead the model while keeping the target object absent, and then guiding a diffusion model conditioned on the embedding to generate natural-looking images. The resulting images remain visually natural and close to the original input, yet introduce subtle misleading cues that cause the model to hallucinate. We evaluate our method across a range of models, including reasoning models like GLM-4.1V-Thinking, and achieve a hallucination success rate exceeding 28%, compared to around 1% in prior data-driven discovery methods. We confirm that the generated images are both high-quality and object-free through quantitative metrics and human evaluation. Also, GHOST uncovers transferable vulnerabilities: images optimized for Qwen2.5-VL induce hallucinations in GPT-4o at a 66.5% rate. Finally, we show that fine-tuning on our images mitigates hallucination, positioning GHOST as both a diagnostic and corrective tool for building more reliable multimodal systems. 

---
# EasySteer: A Unified Framework for High-Performance and Extensible LLM Steering 

**Authors**: Haolei Xu, Xinyu Mei, Yuchen Yan, Rui Zhou, Wenqi Zhang, Weiming Lu, Yueting Zhuang, Yongliang Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.25175)  

**Abstract**: Large language model (LLM) steering has emerged as a promising paradigm for controlling model behavior at inference time through targeted manipulation of hidden states, offering a lightweight alternative to expensive retraining. However, existing steering frameworks suffer from critical limitations: computational inefficiency, limited extensibility, and restricted functionality that hinder both research progress and practical deployment. We present EasySteer, a unified framework for high-performance, extensible LLM steering built on vLLM. Our system features modular architecture with pluggable interfaces for both analysis-based and learning-based methods, fine-grained parameter control, pre-computed steering vectors for eight application domains, and an interactive demonstration system. Through deep integration with vLLM's optimized inference engine, EasySteer achieves 5.5-11.4$\times$ speedup over existing frameworks. Extensive experiments demonstrate its effectiveness in overthinking mitigation, hallucination reduction, and other key applications. EasySteer transforms steering from research technique to production-ready capability, establishing critical infrastructure for deployable, controllable language models. 

---
# XQC: Well-conditioned Optimization Accelerates Deep Reinforcement Learning 

**Authors**: Daniel Palenicek, Florian Vogt, Joe Watson, Ingmar Posner, Jan Peters  

**Link**: [PDF](https://arxiv.org/pdf/2509.25174)  

**Abstract**: Sample efficiency is a central property of effective deep reinforcement learning algorithms. Recent work has improved this through added complexity, such as larger models, exotic network architectures, and more complex algorithms, which are typically motivated purely by empirical performance. We take a more principled approach by focusing on the optimization landscape of the critic network. Using the eigenspectrum and condition number of the critic's Hessian, we systematically investigate the impact of common architectural design decisions on training dynamics. Our analysis reveals that a novel combination of batch normalization (BN), weight normalization (WN), and a distributional cross-entropy (CE) loss produces condition numbers orders of magnitude smaller than baselines. This combination also naturally bounds gradient norms, a property critical for maintaining a stable effective learning rate under non-stationary targets and bootstrapping. Based on these insights, we introduce XQC: a well-motivated, sample-efficient deep actor-critic algorithm built upon soft actor-critic that embodies these optimization-aware principles. We achieve state-of-the-art sample efficiency across 55 proprioception and 15 vision-based continuous control tasks, all while using significantly fewer parameters than competing methods. 

---
# GLASS Flows: Transition Sampling for Alignment of Flow and Diffusion Models 

**Authors**: Peter Holderrieth, Uriel Singer, Tommi Jaakkola, Ricky T. Q. Chen, Yaron Lipman, Brian Karrer  

**Link**: [PDF](https://arxiv.org/pdf/2509.25170)  

**Abstract**: The performance of flow matching and diffusion models can be greatly improved at inference time using reward alignment algorithms, yet efficiency remains a major limitation. While several algorithms were proposed, we demonstrate that a common bottleneck is the sampling method these algorithms rely on: many algorithms require to sample Markov transitions via SDE sampling, which is significantly less efficient and often less performant than ODE sampling. To remove this bottleneck, we introduce GLASS Flows, a new sampling paradigm that simulates a "flow matching model within a flow matching model" to sample Markov transitions. As we show in this work, this "inner" flow matching model can be retrieved from a pre-trained model without any re-training, combining the efficiency of ODEs with the stochastic evolution of SDEs. On large-scale text-to-image models, we show that GLASS Flows eliminate the trade-off between stochastic evolution and efficiency. Combined with Feynman-Kac Steering, GLASS Flows improve state-of-the-art performance in text-to-image generation, making it a simple, drop-in solution for inference-time scaling of flow and diffusion models. 

---
# GSM8K-V: Can Vision Language Models Solve Grade School Math Word Problems in Visual Contexts 

**Authors**: Fan Yuan, Yuchen Yan, Yifan Jiang, Haoran Zhao, Tao Feng, Jinyan Chen, Yanwei Lou, Wenqi Zhang, Yongliang Shen, Weiming Lu, Jun Xiao, Yueting Zhuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25160)  

**Abstract**: Vision language models (VLMs) achieve unified modeling of images and text, enabling them to accomplish complex real-world tasks through perception, planning, and reasoning. Among these tasks, reasoning is particularly representative, with mathematical reasoning serving as a prominent example. It highlights the high-level capability of VLMs to comprehend mathematical information in images and to perform sophisticated reasoning. Recently, numerous visual mathematical reasoning benchmarks have been proposed, but they are often restricted to geometry, lack coverage of math word problems, and rarely assess reasoning across multiple images. To address these gaps, we introduce GSM8K-V, a purely visual multi-image mathematical reasoning benchmark. GSM8K-V is built by systematically mapping each sample from the widely used text-based GSM8K into visual form. Through a carefully designed automated image-generation pipeline combined with meticulous human annotation, we curate 1,319 high-quality samples. We evaluate a wide range of open-source and closed-source models on GSM8K-V. Results show that although existing VLMs have nearly saturated performance on text-based GSM8K, there remains substantial room for improvement on GSM8K-V. For example, the best-performing model, Gemini-2.5-Pro, achieves 95.22% accuracy on GSM8K but only 46.93% on GSM8K-V. We conduct a comprehensive analysis of GSM8K-V, examining the limitations of current models as well as potential directions for improvement. GSM8K-V offers a new perspective on visual mathematical reasoning and establishes a benchmark to guide the development of more robust and generalizable VLMs. 

---
# Chance-constrained Flow Matching for High-Fidelity Constraint-aware Generation 

**Authors**: Jinhao Liang, Yixuan Sun, Anirban Samaddar, Sandeep Madireddy, Ferdinando Fioretto  

**Link**: [PDF](https://arxiv.org/pdf/2509.25157)  

**Abstract**: Generative models excel at synthesizing high-fidelity samples from complex data distributions, but they often violate hard constraints arising from physical laws or task specifications. A common remedy is to project intermediate samples onto the feasible set; however, repeated projection can distort the learned distribution and induce a mismatch with the data manifold. Thus, recent multi-stage procedures attempt to defer projection to clean samples during sampling, but they increase algorithmic complexity and accumulate errors across steps. This paper addresses these challenges by proposing a novel training-free method, Chance-constrained Flow Matching (CCFM), that integrates stochastic optimization into the sampling process, enabling effective enforcement of hard constraints while maintaining high-fidelity sample generation. Importantly, CCFM guarantees feasibility in the same manner as conventional repeated projection, yet, despite operating directly on noisy intermediate samples, it is theoretically equivalent to projecting onto the feasible set defined by clean samples. This yields a sampler that mitigates distributional distortion. Empirical experiments show that CCFM outperforms current state-of-the-art constrained generative models in modeling complex physical systems governed by partial differential equations and molecular docking problems, delivering higher feasibility and fidelity. 

---
# Pretraining Large Language Models with NVFP4 

**Authors**: NVIDIA, Felix Abecassis, Anjulie Agrusa, Dong Ahn, Jonah Alben, Stefania Alborghetti, Michael Andersch, Sivakumar Arayandi, Alexis Bjorlin, Aaron Blakeman, Evan Briones, Ian Buck, Bryan Catanzaro, Jinhang Choi, Mike Chrzanowski, Eric Chung, Victor Cui, Steve Dai, Bita Darvish Rouhani, Carlo del Mundo, Deena Donia, Burc Eryilmaz, Henry Estela, Abhinav Goel, Oleg Goncharov, Yugi Guvvala, Robert Hesse, Russell Hewett, Herbert Hum, Ujval Kapasi, Brucek Khailany, Mikail Khona, Nick Knight, Alex Kondratenko, Ronny Krashinsky, Ben Lanir, Simon Layton, Michael Lightstone, Daniel Lo, Paulius Micikevicius, Asit Mishra, Tim Moon, Deepak Narayanan, Chao Ni, Abhijit Paithankar, Satish Pasumarthi, Ankit Patel, Mostofa Patwary, Ashwin Poojary, Gargi Prasad, Sweta Priyadarshi, Yigong Qin, Xiaowei Ren, Oleg Rybakov, Charbel Sakr, Sanjeev Satheesh, Stas Sergienko, Pasha Shamis, Kirthi Shankar, Nishant Sharma, Mohammad Shoeybi, Michael Siu, Misha Smelyanskiy, Darko Stosic, Dusan Stosic, Bor-Yiing Su, Frank Sun, Nima Tajbakhsh, Shelby Thomas, Przemek Tredak, Evgeny Tsykunov, Gandhi Vaithilingam, Aditya Vavre, Rangharajan Venkatesan, Roger Waleffe, Qiyu Wan, Hexin Wang, Mengdi Wang, Lizzie Wei, Hao Wu, Evan Wu, Keith Wyss, Ning Xu, Jinze Xue, Charlene Yang, Yujia Zhai, Ruoxi Zhang, Jingyang Zhu, Zhongbo Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.25149)  

**Abstract**: Large Language Models (LLMs) today are powerful problem solvers across many domains, and they continue to get stronger as they scale in model size, training set size, and training set quality, as shown by extensive research and experimentation across the industry. Training a frontier model today requires on the order of tens to hundreds of yottaflops, which is a massive investment of time, compute, and energy. Improving pretraining efficiency is therefore essential to enable the next generation of even more capable LLMs. While 8-bit floating point (FP8) training is now widely adopted, transitioning to even narrower precision, such as 4-bit floating point (FP4), could unlock additional improvements in computational speed and resource utilization. However, quantization at this level poses challenges to training stability, convergence, and implementation, notably for large-scale models trained on long token horizons.
In this study, we introduce a novel approach for stable and accurate training of large language models (LLMs) using the NVFP4 format. Our method integrates Random Hadamard transforms (RHT) to bound block-level outliers, employs a two-dimensional quantization scheme for consistent representations across both the forward and backward passes, utilizes stochastic rounding for unbiased gradient estimation, and incorporates selective high-precision layers. We validate our approach by training a 12-billion-parameter model on 10 trillion tokens -- the longest publicly documented training run in 4-bit precision to date. Our results show that the model trained with our NVFP4-based pretraining technique achieves training loss and downstream task accuracies comparable to an FP8 baseline. These findings highlight that NVFP4, when combined with our training approach, represents a major step forward in narrow-precision LLM training algorithms. 

---
# Fast Feature Field ($\text{F}^3$): A Predictive Representation of Events 

**Authors**: Richeek Das, Kostas Daniilidis, Pratik Chaudhari  

**Link**: [PDF](https://arxiv.org/pdf/2509.25146)  

**Abstract**: This paper develops a mathematical argument and algorithms for building representations of data from event-based cameras, that we call Fast Feature Field ($\text{F}^3$). We learn this representation by predicting future events from past events and show that it preserves scene structure and motion information. $\text{F}^3$ exploits the sparsity of event data and is robust to noise and variations in event rates. It can be computed efficiently using ideas from multi-resolution hash encoding and deep sets - achieving 120 Hz at HD and 440 Hz at VGA resolutions. $\text{F}^3$ represents events within a contiguous spatiotemporal volume as a multi-channel image, enabling a range of downstream tasks. We obtain state-of-the-art performance on optical flow estimation, semantic segmentation, and monocular metric depth estimation, on data from three robotic platforms (a car, a quadruped robot and a flying platform), across different lighting conditions (daytime, nighttime), environments (indoors, outdoors, urban, as well as off-road) and dynamic vision sensors (resolutions and event rates). Our implementations can predict these tasks at 25-75 Hz at HD resolution. 

---
# Paired by the Teacher: Turning Unpaired Data into High-Fidelity Pairs for Low-Resource Text Generation 

**Authors**: Yen-Ju Lu, Thomas Thebaud, Laureano Moro-Velazquez, Najim Dehak, Jesus Villalba  

**Link**: [PDF](https://arxiv.org/pdf/2509.25144)  

**Abstract**: We present Paired by the Teacher (PbT), a two-stage teacher-student pipeline that synthesizes accurate input-output pairs without human labels or parallel data. In many low-resource natural language generation (NLG) scenarios, practitioners may have only raw outputs, like highlights, recaps, or questions, or only raw inputs, such as articles, dialogues, or paragraphs, but seldom both. This mismatch forces small models to learn from very few examples or rely on costly, broad-scope synthetic examples produced by large LLMs. PbT addresses this by asking a teacher LLM to compress each unpaired example into a concise intermediate representation (IR), and training a student to reconstruct inputs from IRs. This enables outputs to be paired with student-generated inputs, yielding high-quality synthetic data. We evaluate PbT on five benchmarks-document summarization (XSum, CNNDM), dialogue summarization (SAMSum, DialogSum), and question generation (SQuAD)-as well as an unpaired setting on SwitchBoard (paired with DialogSum summaries). An 8B student trained only on PbT data outperforms models trained on 70 B teacher-generated corpora and other unsupervised baselines, coming within 1.2 ROUGE-L of human-annotated pairs and closing 82% of the oracle gap at one-third the annotation cost of direct synthesis. Human evaluation on SwitchBoard further confirms that only PbT produces concise, faithful summaries aligned with the target style, highlighting its advantage of generating in-domain sources that avoid the mismatch, limiting direct synthesis. 

---
# Rethinking Entropy Regularization in Large Reasoning Models 

**Authors**: Yuxian Jiang, Yafu Li, Guanxu Chen, Dongrui Liu, Yu Cheng, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25133)  

**Abstract**: Reinforcement learning with verifiable rewards (RLVR) has shown great promise in enhancing the reasoning abilities of large reasoning models (LRMs). However, it suffers from a critical issue: entropy collapse and premature convergence. Naive entropy regularization, a common approach for encouraging exploration in the traditional RL literature, fails to address this problem in the context of LRM. Our analysis reveals that this failure stems from the vast action space and long trajectories in LRMs, which easily trigger a global entropy explosion as the model indiscriminately explores all possible actions and states. To address this, we propose SIREN (SelectIve entRopy rEgularizatioN), a method that confines exploration to a meaningful subset of actions and states. SIREN achieves this through a two-step entropy masking mechanism, consisting of a top-p mask and a peak-entropy mask. In addition, regularization is transformed into a self-anchored form to stabilize training. Across five mathematical benchmarks, SIREN attains superior average performance over previous entropy-related RLVR approaches, exemplified by a +6.6 maj@k improvement on AIME24/25 with Qwen2.5-Math-7B. Further analysis confirms that SIREN promotes greater response diversity and maintains entropy at an appropriate level, which helps to preserve the validation pass@k throughout training. This effectively mitigates the premature convergence problem common in RLVR for LRM. 

---
# MGM-Omni: Scaling Omni LLMs to Personalized Long-Horizon Speech 

**Authors**: Chengyao Wang, Zhisheng Zhong, Bohao Peng, Senqiao Yang, Yuqi Liu, Haokun Gui, Bin Xia, Jingyao Li, Bei Yu, Jiaya Jia  

**Link**: [PDF](https://arxiv.org/pdf/2509.25131)  

**Abstract**: We present MGM-Omni, a unified Omni LLM for omni-modal understanding and expressive, long-horizon speech generation. Unlike cascaded pipelines that isolate speech synthesis, MGM-Omni adopts a "brain-mouth" design with a dual-track, token-based architecture that cleanly decouples multimodal reasoning from real-time speech generation. This design enables efficient cross-modal interaction and low-latency, streaming speech generation. For understanding, a unified training strategy coupled with a dual audio encoder design enables long-form audio perception across diverse acoustic conditions. For generation, a chunk-based parallel decoding scheme narrows the text speech token-rate gap, accelerating inference and supporting streaming zero-shot voice cloning with stable timbre over extended durations. Compared to concurrent work, MGM-Omni achieves these capabilities with markedly data-efficient training. Extensive experiments demonstrate that MGM-Omni outperforms existing open source models in preserving timbre identity across extended sequences, producing natural and context-aware speech, and achieving superior long-form audio and omnimodal understanding. MGM-Omni establishes an efficient, end-to-end paradigm for omnimodal understanding and controllable, personalised long-horizon speech generation. 

---
# Score Distillation of Flow Matching Models 

**Authors**: Mingyuan Zhou, Yi Gu, Huangjie Zheng, Liangchen Song, Guande He, Yizhe Zhang, Wenze Hu, Yinfei Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.25127)  

**Abstract**: Diffusion models achieve high-quality image generation but are limited by slow iterative sampling. Distillation methods alleviate this by enabling one- or few-step generation. Flow matching, originally introduced as a distinct framework, has since been shown to be theoretically equivalent to diffusion under Gaussian assumptions, raising the question of whether distillation techniques such as score distillation transfer directly. We provide a simple derivation -- based on Bayes' rule and conditional expectations -- that unifies Gaussian diffusion and flow matching without relying on ODE/SDE formulations. Building on this view, we extend Score identity Distillation (SiD) to pretrained text-to-image flow-matching models, including SANA, SD3-Medium, SD3.5-Medium/Large, and FLUX.1-dev, all with DiT backbones. Experiments show that, with only modest flow-matching- and DiT-specific adjustments, SiD works out of the box across these models, in both data-free and data-aided settings, without requiring teacher finetuning or architectural changes. This provides the first systematic evidence that score distillation applies broadly to text-to-image flow matching models, resolving prior concerns about stability and soundness and unifying acceleration techniques across diffusion- and flow-based generators. We will make the PyTorch implementation publicly available. 

---
# Towards Personalized Deep Research: Benchmarks and Evaluations 

**Authors**: Yuan Liang, Jiaxian Li, Yuqing Wang, Piaohong Wang, Motong Tian, Pai Liu, Shuofei Qiao, Runnan Fang, He Zhu, Ge Zhang, Minghao Liu, Yuchen Eleanor Jiang, Ningyu Zhang, Wangchunshu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.25106)  

**Abstract**: Deep Research Agents (DRAs) can autonomously conduct complex investigations and generate comprehensive reports, demonstrating strong real-world potential. However, existing evaluations mostly rely on close-ended benchmarks, while open-ended deep research benchmarks remain scarce and typically neglect personalized scenarios. To bridge this gap, we introduce Personalized Deep Research Bench, the first benchmark for evaluating personalization in DRAs. It pairs 50 diverse research tasks across 10 domains with 25 authentic user profiles that combine structured persona attributes with dynamic real-world contexts, yielding 250 realistic user-task queries. To assess system performance, we propose the PQR Evaluation Framework, which jointly measures (P) Personalization Alignment, (Q) Content Quality, and (R) Factual Reliability. Our experiments on a range of systems highlight current capabilities and limitations in handling personalized deep research. This work establishes a rigorous foundation for developing and evaluating the next generation of truly personalized AI research assistants. 

---
# ORPO-Distill: Mixed-Policy Preference Optimization for Cross-Architecture LLM Distillation 

**Authors**: Aasheesh Singh, Vishal Vaddina, Dagnachew Birru  

**Link**: [PDF](https://arxiv.org/pdf/2509.25100)  

**Abstract**: We introduce ORPO-Distill, a general-purpose method for cross-architecture LLM distillation that formulates the problem as a preference optimization task. Un- like standard CoT distillation, the approach transfers knowledge through diverse reasoning traces. It employs an Odds-Ratio Preference Optimization objective that contrasts teacher and student traces for more effective learning, and adopts a mixed-policy strategy for utilizing student-generated outputs, outperforming both off- and on-policy alternatives. Experiments on five datasets and multiple student models show consistent improvements over conventional black-box KD baselines. 

---
# Scaling with Collapse: Efficient and Predictable Training of LLM Families 

**Authors**: Shane Bergsma, Bin Claire Zhang, Nolan Dey, Shaheer Muhammad, Gurpreet Gosal, Joel Hestness  

**Link**: [PDF](https://arxiv.org/pdf/2509.25087)  

**Abstract**: Effective LLM training relies on *consistency*, meaning that key quantities -- such as final losses and optimal hyperparameters -- scale predictably across model sizes. Qiu et al. (2025) recently showed that this consistency extends beyond scalars: whole training loss curves can *collapse* onto a universal trajectory after a simple normalization. What remains unclear is whether this phenomenon holds for LLM families trained under *practical scaling recipes*, where width, depth, learning rate, batch size, and weight decay are scaled jointly. We show that it does: loss curves collapse across scales precisely when optimization hyperparameters are set optimally for the given data budget, in accordance with recent empirical scaling laws. Collapse thus emerges as a signature of compute-efficient training. We demonstrate two applications at scale: (1) deviation-from-collapse provides a sensitive, early diagnostic of training pathologies, and (2) the predictability of collapsed curves enables early stopping in large-scale hyperparameter tuning. Finally, we train a competitive LLM family, *Celerity*, using these insights, highlighting collapse as an effective tool for developing efficient LLMs. 

---
# jina-reranker-v3: Last but Not Late Interaction for Document Reranking 

**Authors**: Feng Wang, Yuqing Li, Han Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2509.25085)  

**Abstract**: jina-reranker-v3 is a 0.6B parameter multilingual document reranker that introduces a novel last but not late interaction. Unlike late interaction models such as ColBERT that perform separate encoding followed by multi-vector matching, our approach conducts causal self-attention between query and documents within the same context window, enabling rich cross-document interactions before extracting contextual embeddings from the last token of each document. This compact architecture achieves state-of-the-art BEIR performance with 61.94 nDCG@10 while being ten times smaller than generative listwise rerankers. 

---
# Scaling Generalist Data-Analytic Agents 

**Authors**: Shuofei Qiao, Yanqiu Zhao, Zhisong Qiu, Xiaobin Wang, Jintian Zhang, Zhao Bin, Ningyu Zhang, Yong Jiang, Pengjun Xie, Fei Huang, Huajun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.25084)  

**Abstract**: Data-analytic agents are emerging as a key catalyst for automated scientific discovery and for the vision of Innovating AI. Current approaches, however, rely heavily on prompt engineering over proprietary models, while open-source models struggle to face diverse-format, large-scale data files and long-horizon, multi-step reasoning that real-world analytics demands. This paper introduces DataMind, a scalable data synthesis and agent training recipe designed to build generalist data-analytic agents. DataMind tackles three key challenges in building open-source data-analytic agents, including insufficient data resources, improper training strategy, and unstable code-based multi-turn rollout. Concretely, DataMind applies 1) a fine-grained task taxonomy and a recursive easy-to-hard task composition mechanism to increase the diversity and difficulty of synthesized queries; 2) a knowledge-augmented trajectory sampling strategy followed by model-based and rule-based filtering; 3) a dynamically adjustable training objective combining both SFT and RL losses; 4) a memory-frugal and stable code-based multi-turn rollout framework. Built on DataMind, we curate DataMind-12K, a high-quality trajectory set spanning diverse domains, task categories, and data file formats for data-analytic tasks. Trained on DataMind-12K, our DataMind-14B achieves state-of-the-art with an average score of 71.16% on multiple data analysis benchmarks, outperforming the strongest proprietary baselines DeepSeek-V3.1 and GPT-5. Our DataMind-7B also performs best among all open-source models with a score of 68.10%. We also incorporate some empirical insights gained from our exploratory trials into the analysis experiments, aiming to provide actionable insights about agentic training for the community. We will release DataMind-12K and DataMind-7B,14B for the community's future research. 

---
# UniLat3D: Geometry-Appearance Unified Latents for Single-Stage 3D Generation 

**Authors**: Guanjun Wu, Jiemin Fang, Chen Yang, Sikuang Li, Taoran Yi, Jia Lu, Zanwei Zhou, Jiazhong Cen, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Xinggang Wang, Qi Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.25079)  

**Abstract**: High-fidelity 3D asset generation is crucial for various industries. While recent 3D pretrained models show strong capability in producing realistic content, most are built upon diffusion models and follow a two-stage pipeline that first generates geometry and then synthesizes appearance. Such a decoupled design tends to produce geometry-texture misalignment and non-negligible cost. In this paper, we propose UniLat3D, a unified framework that encodes geometry and appearance in a single latent space, enabling direct single-stage generation. Our key contribution is a geometry-appearance Unified VAE, which compresses high-resolution sparse features into a compact latent representation -- UniLat. UniLat integrates structural and visual information into a dense low-resolution latent, which can be efficiently decoded into diverse 3D formats, e.g., 3D Gaussians and meshes. Based on this unified representation, we train a single flow-matching model to map Gaussian noise directly into UniLat, eliminating redundant stages. Trained solely on public datasets, UniLat3D produces high-quality 3D assets in seconds from a single image, achieving superior appearance fidelity and geometric quality. More demos \& code are available at this https URL 

---
# BRIDGE - Building Reinforcement-Learning Depth-to-Image Data Generation Engine for Monocular Depth Estimation 

**Authors**: Dingning Liu, Haoyu Guo, Jingyi Zhou, Tong He  

**Link**: [PDF](https://arxiv.org/pdf/2509.25077)  

**Abstract**: Monocular Depth Estimation (MDE) is a foundational task for computer vision. Traditional methods are limited by data scarcity and quality, hindering their robustness. To overcome this, we propose BRIDGE, an RL-optimized depth-to-image (D2I) generation framework that synthesizes over 20M realistic and geometrically accurate RGB images, each intrinsically paired with its ground truth depth, from diverse source depth maps. Then we train our depth estimation model on this dataset, employing a hybrid supervision strategy that integrates teacher pseudo-labels with ground truth depth for comprehensive and robust training. This innovative data generation and training paradigm enables BRIDGE to achieve breakthroughs in scale and domain diversity, consistently outperforming existing state-of-the-art approaches quantitatively and in complex scene detail capture, thereby fostering general and robust depth features. Code and models are available at this https URL. 

---
# Optimizing Privacy-Preserving Primitives to Support LLM-Scale Applications 

**Authors**: Yaman Jandali, Ruisi Zhang, Nojan Sheybani, Farinaz Koushanfar  

**Link**: [PDF](https://arxiv.org/pdf/2509.25072)  

**Abstract**: Privacy-preserving technologies have introduced a paradigm shift that allows for realizable secure computing in real-world systems. The significant barrier to the practical adoption of these primitives is the computational and communication overhead that is incurred when applied at scale. In this paper, we present an overview of our efforts to bridge the gap between this overhead and practicality for privacy-preserving learning systems using multi-party computation (MPC), zero-knowledge proofs (ZKPs), and fully homomorphic encryption (FHE). Through meticulous hardware/software/algorithm co-design, we show progress towards enabling LLM-scale applications in privacy-preserving settings. We demonstrate the efficacy of our solutions in several contexts, including DNN IP ownership, ethical LLM usage enforcement, and transformer inference. 

---
# Hyperdimensional Probe: Decoding LLM Representations via Vector Symbolic Architectures 

**Authors**: Marco Bronzini, Carlo Nicolini, Bruno Lepri, Jacopo Staiano, Andrea Passerini  

**Link**: [PDF](https://arxiv.org/pdf/2509.25045)  

**Abstract**: Despite their capabilities, Large Language Models (LLMs) remain opaque with limited understanding of their internal representations. Current interpretability methods, such as direct logit attribution (DLA) and sparse autoencoders (SAEs), provide restricted insight due to limitations such as the model's output vocabulary or unclear feature names. This work introduces Hyperdimensional Probe, a novel paradigm for decoding information from the LLM vector space. It combines ideas from symbolic representations and neural probing to project the model's residual stream into interpretable concepts via Vector Symbolic Architectures (VSAs). This probe combines the strengths of SAEs and conventional probes while overcoming their key limitations. We validate our decoding paradigm with controlled input-completion tasks, probing the model's final state before next-token prediction on inputs spanning syntactic pattern recognition, key-value associations, and abstract inference. We further assess it in a question-answering setting, examining the state of the model both before and after text generation. Our experiments show that our probe reliably extracts meaningful concepts across varied LLMs, embedding sizes, and input domains, also helping identify LLM failures. Our work advances information decoding in LLM vector space, enabling extracting more informative, interpretable, and structured features from neural representations. 

---
# Large Language Models for Software Testing: A Research Roadmap 

**Authors**: Cristian Augusto, Antonia Bertolino, Guglielmo De Angelis, Francesca Lonetti, Jesús Morán  

**Link**: [PDF](https://arxiv.org/pdf/2509.25043)  

**Abstract**: Large Language Models (LLMs) are starting to be profiled as one of the most significant disruptions in the Software Testing field.
Specifically, they have been successfully applied in software testing tasks such as generating test code, or summarizing documentation.
This potential has attracted hundreds of researchers, resulting in dozens of new contributions every month, hardening researchers to
stay at the forefront of the wave. Still, to the best of our knowledge, no prior work has provided a structured vision of the progress
and most relevant research trends in LLM-based testing. In this article, we aim to provide a roadmap that illustrates its current state,
grouping the contributions into different categories, and also sketching the most promising and active research directions for the field.
To achieve this objective, we have conducted a semi-systematic literature review, collecting articles and mapping them into the most
prominent categories, reviewing the current and ongoing status, and analyzing the open challenges of LLM-based software testing.
Lastly, we have outlined several expected long-term impacts of LLMs over the whole software testing field. 

---
# Fast Real-Time Pipeline for Robust Arm Gesture Recognition 

**Authors**: Milán Zsolt Bagladi, László Gulyás, Gergő Szalay  

**Link**: [PDF](https://arxiv.org/pdf/2509.25042)  

**Abstract**: This paper presents a real-time pipeline for dynamic arm gesture recognition based on OpenPose keypoint estimation, keypoint normalization, and a recurrent neural network classifier. The 1 x 1 normalization scheme and two feature representations (coordinate- and angle-based) are presented for the pipeline. In addition, an efficient method to improve robustness against camera angle variations is also introduced by using artificially rotated training data. Experiments on a custom traffic-control gesture dataset demonstrate high accuracy across varying viewing angles and speeds. Finally, an approach to calculate the speed of the arm signal (if necessary) is also presented. 

---
# Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct 

**Authors**: Haoyang Zheng, Xinyang Liu, Cindy Xiangrui Kong, Nan Jiang, Zheyuan Hu, Weijian Luo, Wei Deng, Guang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.25035)  

**Abstract**: Fast generation of language texts is the holy grail that people pursue in the AI era. In this work, we introduced Discrete Diffusion Divergence Instruct (DiDi-Instruct), a training-based method that leads to fast language generation models by initializing from a pre-trained (masked) discrete diffusion language model (dLLM). The resulting DiDi-Instruct model outperforms the dLLM counterparts and the GPT-2 baseline with 64x acceleration. In the theoretical part of the paper, we build the foundation of DiDi-Instruct in a framework of integral KL-divergence minimization, with practical training algorithms. We also introduce techniques like grouped reward normalization, intermediate-state matching, and the reward-guided ancestral sampler (RGAS) that significantly improve the training stability, the model coverage, and the inference performances. On OpenWebText, DiDi-Instruct outperforms all accelerated language generation models as well as the GPT-2 baseline and the standard dLLMs, achieving sample perplexities ranging from 62.2 (8 NFEs) to 18.4 (128 NFEs). These performance gains are accomplished with a negligible entropy loss of about 1% and 20x less additional training wall-clock time. We further validate the robustness and effectiveness of DiDi-Instruct through extensive ablation studies, model scaling, and the generation of discrete protein sequences. In conclusion, DiDi-Instruct is an efficient yet effective distillation method, enabling language generation in the blink of an eye. We will release both code and models at this http URL. 

---
# AIRoA MoMa Dataset: A Large-Scale Hierarchical Dataset for Mobile Manipulation 

**Authors**: Ryosuke Takanami, Petr Khrapchenkov, Shu Morikuni, Jumpei Arima, Yuta Takaba, Shunsuke Maeda, Takuya Okubo, Genki Sano, Satoshi Sekioka, Aoi Kadoya, Motonari Kambara, Naoya Nishiura, Haruto Suzuki, Takanori Yoshimoto, Koya Sakamoto, Shinnosuke Ono, Hu Yang, Daichi Yashima, Aoi Horo, Tomohiro Motoda, Kensuke Chiyoma, Hiroshi Ito, Koki Fukuda, Akihito Goto, Kazumi Morinaga, Yuya Ikeda, Riko Kawada, Masaki Yoshikawa, Norio Kosuge, Yuki Noguchi, Kei Ota, Tatsuya Matsushima, Yusuke Iwasawa, Yutaka Matsuo, Tetsuya Ogata  

**Link**: [PDF](https://arxiv.org/pdf/2509.25032)  

**Abstract**: As robots transition from controlled settings to unstructured human environments, building generalist agents that can reliably follow natural language instructions remains a central challenge. Progress in robust mobile manipulation requires large-scale multimodal datasets that capture contact-rich and long-horizon tasks, yet existing resources lack synchronized force-torque sensing, hierarchical annotations, and explicit failure cases. We address this gap with the AIRoA MoMa Dataset, a large-scale real-world multimodal dataset for mobile manipulation. It includes synchronized RGB images, joint states, six-axis wrist force-torque signals, and internal robot states, together with a novel two-layer annotation schema of sub-goals and primitive actions for hierarchical learning and error analysis. The initial dataset comprises 25,469 episodes (approx. 94 hours) collected with the Human Support Robot (HSR) and is fully standardized in the LeRobot v2.1 format. By uniquely integrating mobile manipulation, contact-rich interaction, and long-horizon structure, AIRoA MoMa provides a critical benchmark for advancing the next generation of Vision-Language-Action models. The first version of our dataset is now available at this https URL . 

---
# CLASP: Adaptive Spectral Clustering for Unsupervised Per-Image Segmentation 

**Authors**: Max Curie, Paulo da Costa  

**Link**: [PDF](https://arxiv.org/pdf/2509.25016)  

**Abstract**: We introduce CLASP (Clustering via Adaptive Spectral Processing), a lightweight framework for unsupervised image segmentation that operates without any labeled data or finetuning. CLASP first extracts per patch features using a self supervised ViT encoder (DINO); then, it builds an affinity matrix and applies spectral clustering. To avoid manual tuning, we select the segment count automatically with a eigengap silhouette search, and we sharpen the boundaries with a fully connected DenseCRF. Despite its simplicity and training free nature, CLASP attains competitive mIoU and pixel accuracy on COCO Stuff and ADE20K, matching recent unsupervised baselines. The zero training design makes CLASP a strong, easily reproducible baseline for large unannotated corpora especially common in digital advertising and marketing workflows such as brand safety screening, creative asset curation, and social media content moderation 

---
# Generalized Correctness Models: Learning Calibrated and Model-Agnostic Correctness Predictors from Historical Patterns 

**Authors**: Hanqi Xiao, Vaidehi Patil, Hyunji Lee, Elias Stengel-Eskin, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2509.24988)  

**Abstract**: Generating accurate and calibrated confidence estimates is critical for deploying LLMs in high-stakes or user-facing applications, and remains an open challenge. Prior research has often framed confidence as a problem of eliciting a model's "self-knowledge", i.e., the ability of an LLM to judge whether its own answers are correct; this approach implicitly assumes that there is some privileged information about the answer's correctness that is accessible to the model itself. However, our experiments reveal that an LLM attempting to predict the correctness of its own outputs generally performs no better than an unrelated LLM. Moreover, we hypothesize that a key factor in building a "Correctness Model" (CM) is exposure to a target model's historical predictions. We propose multiple methods to inject this historical correctness information, creating a Generalized Correctness Model (GCM). We first show that GCMs can be trained on the correctness data from many LLMs and learn patterns for correctness prediction applicable across datasets and models. We then use CMs as a lens for studying the source of correctness prediction ability and its generalization, systematically controlling their training data and finding that answer phrasing is a strong predictor for correctness. We further explore alternative methods of injecting history without training an LLM, finding that including history as in-context examples can help improve correctness prediction, and post-hoc calibration can provide complementary reductions in calibration error. We evaluate GCMs based on Qwen3-8B across 5 model families and the MMLU and TriviaQA datasets, as well as on a downstream selective prediction task, finding that reliable LLM confidence estimation is a generalizable and model-agnostic skill learned by systematically encoding correctness history rather than a model-specific skill reliant on self-introspection. 

---
# Light-SQ: Structure-aware Shape Abstraction with Superquadrics for Generated Meshes 

**Authors**: Yuhan Wang, Weikai Chen, Zeyu Hu, Runze Zhang, Yingda Yin, Ruoyu Wu, Keyang Luo, Shengju Qian, Yiyan Ma, Hongyi Li, Yuan Gao, Yuhuan Zhou, Hao Luo, Wan Wang, Xiaobin Shen, Zhaowei Li, Kuixin Zhu, Chuanlang Hong, Yueyue Wang, Lijie Feng, Xin Wang, Chen Change Loy  

**Link**: [PDF](https://arxiv.org/pdf/2509.24986)  

**Abstract**: In user-generated-content (UGC) applications, non-expert users often rely on image-to-3D generative models to create 3D assets. In this context, primitive-based shape abstraction offers a promising solution for UGC scenarios by compressing high-resolution meshes into compact, editable representations. Towards this end, effective shape abstraction must therefore be structure-aware, characterized by low overlap between primitives, part-aware alignment, and primitive compactness. We present Light-SQ, a novel superquadric-based optimization framework that explicitly emphasizes structure-awareness from three aspects. (a) We introduce SDF carving to iteratively udpate the target signed distance field, discouraging overlap between primitives. (b) We propose a block-regrow-fill strategy guided by structure-aware volumetric decomposition, enabling structural partitioning to drive primitive placement. (c) We implement adaptive residual pruning based on SDF update history to surpress over-segmentation and ensure compact results. In addition, Light-SQ supports multiscale fitting, enabling localized refinement to preserve fine geometric details. To evaluate our method, we introduce 3DGen-Prim, a benchmark extending 3DGen-Bench with new metrics for both reconstruction quality and primitive-level editability. Extensive experiments demonstrate that Light-SQ enables efficient, high-fidelity, and editable shape abstraction with superquadrics for complex generated geometry, advancing the feasibility of 3D UGC creation. 

---
# Random Policy Valuation is Enough for LLM Reasoning with Verifiable Rewards 

**Authors**: Haoran He, Yuxiao Ye, Qingpeng Cai, Chen Hu, Binxing Jiao, Daxin Jiang, Ling Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24981)  

**Abstract**: RL with Verifiable Rewards (RLVR) has emerged as a promising paradigm for improving the reasoning abilities of large language models (LLMs). Current methods rely primarily on policy optimization frameworks like PPO and GRPO, which follow generalized policy iteration that alternates between evaluating the current policy's value and improving the policy based on evaluation. While effective, they often suffer from training instability and diversity collapse, requiring complex heuristic tricks and careful tuning. We observe that standard RLVR in math reasoning can be formalized as a specialized finite-horizon Markov Decision Process with deterministic state transitions, tree-structured dynamics, and binary terminal rewards. Though large in scale, the underlying structure is simpler than general-purpose control settings for which popular RL algorithms (e.g., PPO) were developed, suggesting that several sophisticated techniques in existing methods may be reduced or even omitted. Based on this insight, we prove a surprising result: the optimal action can be recovered from the Q-function of a fixed uniformly random policy, thereby bypassing the generalized policy iteration loop and its associated heuristics. We introduce Random Policy Valuation for Diverse Reasoning (ROVER) to translate this principle into a practical and scalable algorithm for LLM math reasoning, a minimalist yet highly effective RL method that samples actions from a softmax over these uniform-policy Q-values. ROVER preserves diversity throughout training, allowing sustained exploration of multiple valid pathways. Across multiple base models and standard math reasoning benchmarks, ROVER demonstrates superior performance in both \textbf{quality} (\textbf{+8.2} on pass@1, \textbf{+16.8} on pass@256) and \textbf{diversity} (\textbf{+17.6\%}), despite its radical simplification compared to strong, complicated existing methods. 

---
# SecInfer: Preventing Prompt Injection via Inference-time Scaling 

**Authors**: Yupei Liu, Yanting Wang, Yuqi Jia, Jinyuan Jia, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2509.24967)  

**Abstract**: Prompt injection attacks pose a pervasive threat to the security of Large Language Models (LLMs). State-of-the-art prevention-based defenses typically rely on fine-tuning an LLM to enhance its security, but they achieve limited effectiveness against strong attacks. In this work, we propose \emph{SecInfer}, a novel defense against prompt injection attacks built on \emph{inference-time scaling}, an emerging paradigm that boosts LLM capability by allocating more compute resources for reasoning during inference. SecInfer consists of two key steps: \emph{system-prompt-guided sampling}, which generates multiple responses for a given input by exploring diverse reasoning paths through a varied set of system prompts, and \emph{target-task-guided aggregation}, which selects the response most likely to accomplish the intended task. Extensive experiments show that, by leveraging additional compute at inference, SecInfer effectively mitigates both existing and adaptive prompt injection attacks, outperforming state-of-the-art defenses as well as existing inference-time scaling approaches. 

---
# MSG: Multi-Stream Generative Policies for Sample-Efficient Robotic Manipulation 

**Authors**: Jan Ole von Hartz, Lukas Schweizer, Joschka Boedecker, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2509.24956)  

**Abstract**: Generative robot policies such as Flow Matching offer flexible, multi-modal policy learning but are sample-inefficient. Although object-centric policies improve sample efficiency, it does not resolve this limitation. In this work, we propose Multi-Stream Generative Policy (MSG), an inference-time composition framework that trains multiple object-centric policies and combines them at inference to improve generalization and sample efficiency. MSG is model-agnostic and inference-only, hence widely applicable to various generative policies and training paradigms. We perform extensive experiments both in simulation and on a real robot, demonstrating that our approach learns high-quality generative policies from as few as five demonstrations, resulting in a 95% reduction in demonstrations, and improves policy performance by 89 percent compared to single-stream approaches. Furthermore, we present comprehensive ablation studies on various composition strategies and provide practical recommendations for deployment. Finally, MSG enables zero-shot object instance transfer. We make our code publicly available at this https URL. 

---
# Learning Distinguishable Representations in Deep Q-Networks for Linear Transfer 

**Authors**: Sooraj Sathish, Keshav Goyal, Raghuram Bharadwaj Diddigi  

**Link**: [PDF](https://arxiv.org/pdf/2509.24947)  

**Abstract**: Deep Reinforcement Learning (RL) has demonstrated success in solving complex sequential decision-making problems by integrating neural networks with the RL framework. However, training deep RL models poses several challenges, such as the need for extensive hyperparameter tuning and high computational costs. Transfer learning has emerged as a promising strategy to address these challenges by enabling the reuse of knowledge from previously learned tasks for new, related tasks. This avoids the need for retraining models entirely from scratch. A commonly used approach for transfer learning in RL is to leverage the internal representations learned by the neural network during training. Specifically, the activations from the last hidden layer can be viewed as refined state representations that encapsulate the essential features of the input. In this work, we investigate whether these representations can be used as input for training simpler models, such as linear function approximators, on new tasks. We observe that the representations learned by standard deep RL models can be highly correlated, which limits their effectiveness when used with linear function approximation. To mitigate this problem, we propose a novel deep Q-learning approach that introduces a regularization term to reduce positive correlations between feature representation of states. By leveraging these reduced correlated features, we enable more effective use of linear function approximation in transfer learning. Through experiments and ablation studies on standard RL benchmarks and MinAtar games, we demonstrate the efficacy of our approach in improving transfer learning performance and thereby reducing computational overhead. 

---
# MobileLLM-R1: Exploring the Limits of Sub-Billion Language Model Reasoners with Open Training Recipes 

**Authors**: Changsheng Zhao, Ernie Chang, Zechun Liu, Chia-Jung Chang, Wei Wen, Chen Lai, Rick Cao, Yuandong Tian, Raghuraman Krishnamoorthi, Yangyang Shi, Vikas Chandra  

**Link**: [PDF](https://arxiv.org/pdf/2509.24945)  

**Abstract**: The paradigm shift in large language models (LLMs) from instinctive responses to chain-of-thought (CoT) reasoning has fueled two prevailing assumptions: (1) reasoning capabilities only emerge in sufficiently large models, and (2) such capabilities require training on massive datasets. While the first assumption has already been challenged by recent sub-billion-parameter reasoning models such as Qwen3-0.6B and DeepSeek distilled variants, the second remains largely unquestioned. In this work, we revisit the necessity of scaling to extremely large corpora (>10T tokens) for reasoning emergence. By carefully curating and resampling open-source datasets that we identify as beneficial under our designed metrics, we demonstrate that strong reasoning abilities can emerge with far less data. Specifically, we show that only ~2T tokens of high-quality data are sufficient, and pre-training with 4.2T tokens on the dataset resampled from these ~2T tokens, followed by a established post-training procedure, enables the development of MobileLLM-R1, a series of sub-billion-parameter reasoning models that substantially outperform prior models trained on fully open-sourced data. For example, MobileLLM-R1-950M achieves an AIME score of 15.5, compared to just 0.6 for OLMo-2-1.48B and 0.3 for SmolLM-2-1.7B. Remarkably, despite being trained on only 11.7% of the tokens compared to Qwen3's proprietary 36T-token corpus for pretraining, MobileLLM-R1-950M matches or surpasses Qwen3-0.6B across multiple reasoning benchmarks. To facilitate further research in this direction, we have released the complete training recipe, data sources, data mixing ratio, and model checkpoints, together with the key insights obtained throughout this study. 

---
# Scalable GANs with Transformers 

**Authors**: Sangeek Hyun, MinKyu Lee, Jae-Pil Heo  

**Link**: [PDF](https://arxiv.org/pdf/2509.24935)  

**Abstract**: Scalability has driven recent advances in generative modeling, yet its principles remain underexplored for adversarial learning. We investigate the scalability of Generative Adversarial Networks (GANs) through two design choices that have proven to be effective in other types of generative models: training in a compact Variational Autoencoder latent space and adopting purely transformer-based generators and discriminators. Training in latent space enables efficient computation while preserving perceptual fidelity, and this efficiency pairs naturally with plain transformers, whose performance scales with computational budget. Building on these choices, we analyze failure modes that emerge when naively scaling GANs. Specifically, we find issues as underutilization of early layers in the generator and optimization instability as the network scales. Accordingly, we provide simple and scale-friendly solutions as lightweight intermediate supervision and width-aware learning-rate adjustment. Our experiments show that GAT, a purely transformer-based and latent-space GANs, can be easily trained reliably across a wide range of capacities (S through XL). Moreover, GAT-XL/2 achieves state-of-the-art single-step, class-conditional generation performance (FID of 2.96) on ImageNet-256 in just 40 epochs, 6x fewer epochs than strong baselines. 

---
# When Greedy Wins: Emergent Exploitation Bias in Meta-Bandit LLM Training 

**Authors**: Sanxing Chen, Xiaoyin Chen, Yukun Huang, Roy Xie, Bhuwan Dhingra  

**Link**: [PDF](https://arxiv.org/pdf/2509.24923)  

**Abstract**: While Large Language Models (LLMs) hold promise to become autonomous agents, they often explore suboptimally in sequential decision-making. Recent work has sought to enhance this capability via supervised fine-tuning (SFT) or reinforcement learning (RL), improving regret on the classic multi-armed bandit task. However, it remains unclear how these learning methods shape exploration strategies and how well they generalize. We investigate both paradigms by training LLMs with SFT on expert trajectories and RL with a range of tailored reward signals including a strategic, regret-shaped reward to reduce variance, and an algorithmic reward that enables oracle imitation. The resulting agents outperform pre-trained models and achieve performance comparable to Upper Confidence Bound (UCB) and Thompson Sampling, with robust generalization to 6x longer horizons and across bandit families. Behavioral analysis reveals that gains often stem from more sophisticated but greedier exploitation: RL/SFT agents are more prone to early catastrophic failure than pre-trained models, prematurely abandoning exploration. Furthermore, agents trained to imitate UCB learn to outperform their teacher by adopting more exploitative variants. Our findings clarify when each training paradigm is preferable and advocate tailored reward design and evaluation beyond average regret to promote robust exploratory behavior. 

---
# Segmentor-Guided Counterfactual Fine-Tuning for Image Synthesis 

**Authors**: Tian Xia, Matthew Sinclair, Andreas Schuh, Fabio De Sousa Ribeiro, Raghav Mehta, Rajat Rasal, Esther Puyol-Antón, Samuel Gerber, Kersten Petersen, Michiel Schaap, Ben Glocker  

**Link**: [PDF](https://arxiv.org/pdf/2509.24913)  

**Abstract**: Counterfactual image generation is a powerful tool for augmenting training data, de-biasing datasets, and modeling disease. Current approaches rely on external classifiers or regressors to increase the effectiveness of subject-level interventions (e.g., changing the patient's age). For structure-specific interventions (e.g., changing the area of the left lung in a chest radiograph), we show that this is insufficient, and can result in undesirable global effects across the image domain. Previous work used pixel-level label maps as guidance, requiring a user to provide hypothetical segmentations which are tedious and difficult to obtain. We propose Segmentor-guided Counterfactual Fine-Tuning (Seg-CFT), which preserves the simplicity of intervening on scalar-valued, structure-specific variables while producing locally coherent and effective counterfactuals. We demonstrate the capability of generating realistic chest radiographs, and we show promising results for modeling coronary artery disease. Code: this https URL. 

---
# OpenGPT-4o-Image: A Comprehensive Dataset for Advanced Image Generation and Editing 

**Authors**: Zhihong Chen, Xuehai Bai, Yang Shi, Chaoyou Fu, Huanyu Zhang, Haotian Wang, Xiaoyan Sun, Zhang Zhang, Liang Wang, Yuanxing Zhang, Pengfei Wan, Yi-Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24900)  

**Abstract**: The performance of unified multimodal models for image generation and editing is fundamentally constrained by the quality and comprehensiveness of their training data. While existing datasets have covered basic tasks like style transfer and simple object manipulation, they often lack the systematic structure and challenging scenarios required for real-world applications. To address this bottleneck, we introduce OpenGPT-4o-Image, a large-scale dataset constructed using a novel methodology that combines hierarchical task taxonomy with automated data generation. Our taxonomy not only includes fundamental capabilities such as text rendering and style control but also introduces highly practical yet challenging categories like scientific imagery for chemistry illustrations and complex instruction editing requiring simultaneous execution of multiple operations. Through an automated pipeline leveraging structured resource pools and GPT-4o, we generate 80k high-quality instruction-image pairs with controlled diversity, covering 11 major domains and 51 subtasks. Extensive experiments show that fine-tuning leading models on our dataset achieves significant performance gains across multiple benchmarks, with improvements of up to 18\% on editing tasks (UniWorld-V1 on ImgEdit-Bench) and 13% on generation tasks (Harmon on GenEval). Our work demonstrates that systematic data construction is key to advancing multimodal AI capabilities. 

---
# Scaling Laws and Spectra of Shallow Neural Networks in the Feature Learning Regime 

**Authors**: Leonardo Defilippis, Yizhou Xu, Julius Girardin, Emanuele Troiani, Vittorio Erba, Lenka Zdeborová, Bruno Loureiro, Florent Krzakala  

**Link**: [PDF](https://arxiv.org/pdf/2509.24882)  

**Abstract**: Neural scaling laws underlie many of the recent advances in deep learning, yet their theoretical understanding remains largely confined to linear models. In this work, we present a systematic analysis of scaling laws for quadratic and diagonal neural networks in the feature learning regime. Leveraging connections with matrix compressed sensing and LASSO, we derive a detailed phase diagram for the scaling exponents of the excess risk as a function of sample complexity and weight decay. This analysis uncovers crossovers between distinct scaling regimes and plateau behaviors, mirroring phenomena widely reported in the empirical neural scaling literature. Furthermore, we establish a precise link between these regimes and the spectral properties of the trained network weights, which we characterize in detail. As a consequence, we provide a theoretical validation of recent empirical observations connecting the emergence of power-law tails in the weight spectrum with network generalization performance, yielding an interpretation from first principles. 

---
# Vehicle Classification under Extreme Imbalance: A Comparative Study of Ensemble Learning and CNNs 

**Authors**: Abu Hanif Muhammad Syarubany  

**Link**: [PDF](https://arxiv.org/pdf/2509.24880)  

**Abstract**: Accurate vehicle type recognition underpins intelligent transportation and logistics, but severe class imbalance in public datasets suppresses performance on rare categories. We curate a 16-class corpus (~47k images) by merging Kaggle, ImageNet, and web-crawled data, and create six balanced variants via SMOTE oversampling and targeted undersampling. Lightweight ensembles, such as Random Forest, AdaBoost, and a soft-voting combiner built on MobileNet-V2 features are benchmarked against a configurable ResNet-style CNN trained with strong augmentation and label smoothing. The best ensemble (SMOTE-combined) attains 74.8% test accuracy, while the CNN achieves 79.19% on the full test set and 81.25% on an unseen inference batch, confirming the advantage of deep models. Nonetheless, the most under-represented class (Barge) remains a failure mode, highlighting the limits of rebalancing alone. Results suggest prioritizing additional minority-class collection and cost-sensitive objectives (e.g., focal loss) and exploring hybrid ensemble or CNN pipelines to combine interpretability with representational power. 

---
# Uncertainty-Guided Expert-AI Collaboration for Efficient Soil Horizon Annotation 

**Authors**: Teodor Chiaburu, Vipin Singh, Frank Haußer, Felix Bießmann  

**Link**: [PDF](https://arxiv.org/pdf/2509.24873)  

**Abstract**: Uncertainty quantification is essential in human-machine collaboration, as human agents tend to adjust their decisions based on the confidence of the machine counterpart. Reliably calibrated model uncertainties, hence, enable more effective collaboration, targeted expert intervention and more responsible usage of Machine Learning (ML) systems. Conformal prediction has become a well established model-agnostic framework for uncertainty calibration of ML models, offering statistically valid confidence estimates for both regression and classification tasks. In this work, we apply conformal prediction to $\textit{SoilNet}$, a multimodal multitask model for describing soil profiles. We design a simulated human-in-the-loop (HIL) annotation pipeline, where a limited budget for obtaining ground truth annotations from domain experts is available when model uncertainty is high. Our experiments show that conformalizing SoilNet leads to more efficient annotation in regression tasks and comparable performance scores in classification tasks under the same annotation budget when tested against its non-conformal counterpart. All code and experiments can be found in our repository: this https URL 

---
# Retro*: Optimizing LLMs for Reasoning-Intensive Document Retrieval 

**Authors**: Junwei Lan, Jianlyu Chen, Zheng Liu, Chaofan Li, Siqi Bao, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2509.24869)  

**Abstract**: With the growing popularity of LLM agents and RAG, it has become increasingly important to retrieve documents that are essential for solving a task, even when their connection to the task is indirect or implicit. Addressing this problem requires fine-grained reasoning to accurately assess the relevance between the task and each candidate document. This capability, however, poses a significant challenge for existing IR techniques. Despite recent progress in reasoning-enhanced IR, existing approaches still face significant challenges in applicability, scalability, and efficiency. In this work, we propose Retro*, a novel approach for reasoning-intensive document retrieval. Our method introduces a rubric-based relevance scoring mechanism, enabling the model to reason about the relationship between a task and a document based on explicitly defined criteria, whereby producing a fine-grained, interpretable relevance score. Retro* also supports test-time scaling by combining multiple reasoning trajectories via score integration, which produces more reliable relevance estimates. To optimize Retro*'s reasoning capabilities, we introduce a novel reinforcement learning algorithm tailored for its relevance scoring mechanism, which employs two composite rewards to fully exploit the trajectories of each training sample. Our experiments show that Retro* outperforms existing document retrieval methods with notable advantages, leading to state-of-the-art performance on the BRIGHT benchmark. 

---
# Metaphor identification using large language models: A comparison of RAG, prompt engineering, and fine-tuning 

**Authors**: Matteo Fuoli, Weihang Huang, Jeannette Littlemore, Sarah Turner, Ellen Wilding  

**Link**: [PDF](https://arxiv.org/pdf/2509.24866)  

**Abstract**: Metaphor is a pervasive feature of discourse and a powerful lens for examining cognition, emotion, and ideology. Large-scale analysis, however, has been constrained by the need for manual annotation due to the context-sensitive nature of metaphor. This study investigates the potential of large language models (LLMs) to automate metaphor identification in full texts. We compare three methods: (i) retrieval-augmented generation (RAG), where the model is provided with a codebook and instructed to annotate texts based on its rules and examples; (ii) prompt engineering, where we design task-specific verbal instructions; and (iii) fine-tuning, where the model is trained on hand-coded texts to optimize performance. Within prompt engineering, we test zero-shot, few-shot, and chain-of-thought strategies. Our results show that state-of-the-art closed-source LLMs can achieve high accuracy, with fine-tuning yielding a median F1 score of 0.79. A comparison of human and LLM outputs reveals that most discrepancies are systematic, reflecting well-known grey areas and conceptual challenges in metaphor theory. We propose that LLMs can be used to at least partly automate metaphor identification and can serve as a testbed for developing and refining metaphor identification protocols and the theory that underpins them. 

---
# Hierarchical Error Correction for Large Language Models: A Systematic Framework for Domain-Specific AI Quality Enhancement 

**Authors**: Zhilong Zhao, Yindi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24841)  

**Abstract**: Large Language Models face significant performance challenges in specialized domains, with state-of-the-art models achieving only 45.9% accuracy on medical coding tasks. This study proposes a Hierarchical Error Correction (HEC) framework that addresses domain-specific AI limitations through systematic error analysis and targeted intervention strategies.
We analyze error patterns across four specialized domains and find that AI errors follow consistent hierarchical structures: Knowledge-layer errors (58.4%), Reasoning-layer errors (39.6%), and Complexity-layer errors (2.0%). Based on these patterns, we develop a three-stage correction framework that addresses errors according to their hierarchical importance and demonstrates that framework effectiveness correlates inversely with baseline task performance.
Experimental validation across medical transcription (4,921 cases), legal document classification (1,000 cases), political bias detection (645 cases), and legal reasoning (1,000 cases) shows consistent improvements. Cross-model validation across five LLM architectures demonstrates average improvements of 11.2 percentage points (p < 0.001). However, analysis reveals framework limitations in high-baseline tasks (>75% accuracy), where hierarchical intervention may interfere with effective reasoning processes.
The results suggest that systematic error analysis can guide effective AI enhancement strategies in specialized domains, particularly for moderate-baseline tasks, while highlighting the importance of understanding framework boundaries for optimal deployment. 

---
# SemShareKV: Efficient KVCache Sharing for Semantically Similar Prompts via Token-Level LSH Matching 

**Authors**: Xinye Zhao, Spyridon Mastorakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.24832)  

**Abstract**: As large language models (LLMs) continue to scale, the memory footprint of key-value (KV) caches during inference has become a significant bottleneck. Existing approaches primarily focus on compressing KV caches within a single prompt or reusing shared prefixes or frequently ocurred text segments across prompts. However, such strategies are limited in scenarios where prompts are semantically similar but lexically different, which frequently occurs in tasks such as multi-document summarization and conversational agents. We propose \textit{SemShareKV}, a KV cache sharing and compression framework that accelerates LLM inference by reusing KVCache in semantically similar prompts. Instead of relying on exact token matches, SemShareKV applies fuzzy token matching using locality-sensitive hashing (LSH) on token embeddings and incorporates Rotary Position Embedding (RoPE) to better preserve positional information. By selectively reusing relevant key-value pairs from a reference prompt's cache, SemShareKV reduces redundant computation while maintaining output quality. Experiments on diverse summarization datasets show up to 6.25$\times$ speedup and 42\% lower GPU memory usage with 5k tokens input, with negligible quality degradation. These results highlight the potential of semantic-aware cache sharing for efficient LLM inference. 

---
# Evaluating SAP Joule for Code Generation 

**Authors**: Joshua Heisler, Johannes Reisinger, Andreas Fischer  

**Link**: [PDF](https://arxiv.org/pdf/2509.24828)  

**Abstract**: SAP has released its own proprietary generative model SAP Joule, intended for various generative tasks, including serving as a code assistant for software engineers. While Joule is yet not focused on SAP-specific ABAP code generation, it can be used for other common languages, including Javascript. This paper compares SAP Joules Javascript coding capabilities against a total of 29 other models using the HumanEval-X Javascript benchmark. SAP Joule achieves a strict accuracy of 80.49% as the fifth best model in our evaluation. To the best of our knowledge, this is the first comparative evaluation of SAP Joule code generation capabilities. 

---
# Putnam-like dataset summary: LLMs as mathematical competition contestants 

**Authors**: Bartosz Bieganowski, Daniel Strzelecki, Robert Skiba, Mateusz Topolewski  

**Link**: [PDF](https://arxiv.org/pdf/2509.24827)  

**Abstract**: In this paper we summarize the results of the Putnam-like benchmark published by Google DeepMind. This dataset consists of 96 original problems in the spirit of the Putnam Competition and 576 solutions of LLMs. We analyse the performance of models on this set of problems to verify their ability to solve problems from mathematical contests. 

---
# Of-SemWat: High-payload text embedding for semantic watermarking of AI-generated images with arbitrary size 

**Authors**: Benedetta Tondi, Andrea Costanzo, Mauro Barni  

**Link**: [PDF](https://arxiv.org/pdf/2509.24823)  

**Abstract**: We propose a high-payload image watermarking method for textual embedding, where a semantic description of the image - which may also correspond to the input text prompt-, is embedded inside the image. In order to be able to robustly embed high payloads in large-scale images - such as those produced by modern AI generators - the proposed approach builds upon a traditional watermarking scheme that exploits orthogonal and turbo codes for improved robustness, and integrates frequency-domain embedding and perceptual masking techniques to enhance watermark imperceptibility. Experiments show that the proposed method is extremely robust against a wide variety of image processing, and the embedded text can be retrieved also after traditional and AI inpainting, permitting to unveil the semantic modification the image has undergone via image-text mismatch analysis. 

---
# Intelligent Optimization of Wireless Access Point Deployment for Communication-Based Train Control Systems Using Deep Reinforcement Learning 

**Authors**: Kunyu Wu, Qiushi Zhao, Zihan Feng, Yunxi Mu, Hao Qin, Xinyu Zhang, Xingqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24819)  

**Abstract**: Urban railway systems increasingly rely on communication based train control (CBTC) systems, where optimal deployment of access points (APs) in tunnels is critical for robust wireless coverage. Traditional methods, such as empirical model-based optimization algorithms, are hindered by excessive measurement requirements and suboptimal solutions, while machine learning (ML) approaches often struggle with complex tunnel environments. This paper proposes a deep reinforcement learning (DRL) driven framework that integrates parabolic wave equation (PWE) channel modeling, conditional generative adversarial network (cGAN) based data augmentation, and a dueling deep Q network (Dueling DQN) for AP placement optimization. The PWE method generates high-fidelity path loss distributions for a subset of AP positions, which are then expanded by the cGAN to create high resolution path loss maps for all candidate positions, significantly reducing simulation costs while maintaining physical accuracy. In the DRL framework, the state space captures AP positions and coverage, the action space defines AP adjustments, and the reward function encourages signal improvement while penalizing deployment costs. The dueling DQN enhances convergence speed and exploration exploitation balance, increasing the likelihood of reaching optimal configurations. Comparative experiments show that the proposed method outperforms a conventional Hooke Jeeves optimizer and traditional DQN, delivering AP configurations with higher average received power, better worst-case coverage, and improved computational efficiency. This work integrates high-fidelity electromagnetic simulation, generative modeling, and AI-driven optimization, offering a scalable and data-efficient solution for next-generation CBTC systems in complex tunnel environments. 

---
# RDD: Pareto Analysis of the Rate-Distortion-Distinguishability Trade-off 

**Authors**: Andriy Enttsel, Alex Marchioni, Andrea Zanellini, Mauro Mangia, Gianluca Setti, Riccardo Rovatti  

**Link**: [PDF](https://arxiv.org/pdf/2509.24805)  

**Abstract**: Extensive monitoring systems generate data that is usually compressed for network transmission. This compressed data might then be processed in the cloud for tasks such as anomaly detection. However, compression can potentially impair the detector's ability to distinguish between regular and irregular patterns due to information loss. Here we extend the information-theoretic framework introduced in [1] to simultaneously address the trade-off between the three features on which the effectiveness of the system depends: the effectiveness of compression, the amount of distortion it introduces, and the distinguishability between compressed normal signals and compressed anomalous signals. We leverage a Gaussian assumption to draw curves showing how moving on a Pareto surface helps administer such a trade-off better than simply relying on optimal rate-distortion compression and hoping that compressed signals can be distinguished from each other. 

---
# DSAT-HD: Dual-Stream Adaptive Transformer with Hybrid Decomposition for Multivariate Time Series Forecasting 

**Authors**: Zixu Wang, Hongbin Dong, Xiaoping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24800)  

**Abstract**: Time series forecasting is crucial for various applications, such as weather, traffic, electricity, and energy predictions. Currently, common time series forecasting methods are based on Transformers. However, existing approaches primarily model limited time series or fixed scales, making it more challenging to capture diverse features cross different ranges. Additionally, traditional methods like STL for complex seasonality-trend decomposition require pre-specified seasonal periods and typically handle only single, fixed seasonality. We propose the Hybrid Decomposition Dual-Stream Adaptive Transformer (DSAT-HD), which integrates three key innovations to address the limitations of existing methods: 1) A hybrid decomposition mechanism combining EMA and Fourier decomposition with RevIN normalization, dynamically balancing seasonal and trend components through noise Top-k gating; 2) A multi-scale adaptive pathway leveraging a sparse allocator to route features to four parallel Transformer layers, followed by feature merging via a sparse combiner, enhanced by hybrid attention combining local CNNs and global interactions; 3) A dual-stream residual learning framework where CNN and MLP branches separately process seasonal and trend components, coordinated by a balanced loss function minimizing expert collaboration variance. Extensive experiments on nine datasets demonstrate that DSAT-HD outperforms existing methods overall and achieves state-of-the-art performance on some datasets. Notably, it also exhibits stronger generalization capabilities across various transfer scenarios. 

---
# Causal-Adapter: Taming Text-to-Image Diffusion for Faithful Counterfactual Generation 

**Authors**: Lei Tong, Zhihua Liu, Chaochao Lu, Dino Oglic, Tom Diethe, Philip Teare, Sotirios A. Tsaftaris, Chen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.24798)  

**Abstract**: We present Causal-Adapter, a modular framework that adapts frozen text-to-image diffusion backbones for counterfactual image generation. Our method enables causal interventions on target attributes, consistently propagating their effects to causal dependents without altering the core identity of the image. In contrast to prior approaches that rely on prompt engineering without explicit causal structure, Causal-Adapter leverages structural causal modeling augmented with two attribute regularization strategies: prompt-aligned injection, which aligns causal attributes with textual embeddings for precise semantic control, and a conditioned token contrastive loss to disentangle attribute factors and reduce spurious correlations. Causal-Adapter achieves state-of-the-art performance on both synthetic and real-world datasets, with up to 91\% MAE reduction on Pendulum for accurate attribute control and 87\% FID reduction on ADNI for high-fidelity MRI image generation. These results show that our approach enables robust, generalizable counterfactual editing with faithful attribute modification and strong identity preservation. 

---
# Fidelity-Aware Data Composition for Robust Robot Generalization 

**Authors**: Zizhao Tong, Di Chen, Sicheng Hu, Hongwei Fan, Liliang Chen, Guanghui Ren, Hao Tang, Hao Dong, Ling Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.24797)  

**Abstract**: Generalist robot policies trained on large-scale, visually homogeneous datasets can be susceptible to shortcut learning, which impairs their out-of-distribution (OOD) generalization. While generative data augmentation is a common approach to introduce diversity, it presents a subtle challenge: data composition. Naively mixing real and synthetic data can corrupt the learning signal, as this process often prioritizes visual diversity at the expense of information fidelity. This paper suggests that robust generalization depends on principled, fidelity-aware data composition. We introduce Coherent Information Fidelity Tuning (CIFT), a framework that treats data composition as an optimization problem. CIFT uses a practical proxy for Information Fidelity based on the feature-space geometry of a dataset. This enables the identification of a phase transition, termed the Decoherence Point, where training stability degrades. The framework includes a generative engine, Multi-View Video Augmentation (MVAug), to synthesize a causally disentangled data spectrum for this tuning process. Applying CIFT to policy architectures such as $\pi_0$ and Diffusion Policy improves OOD success rates by over 54\%. These results indicate that fidelity-aware composition, beyond data synthesis alone, is an important component for developing robust, general-purpose robots. 

---
# Sparse Autoencoders Make Audio Foundation Models more Explainable 

**Authors**: Théo Mariotte, Martin Lebourdais, Antonio Almudévar, Marie Tahon, Alfonso Ortega, Nicolas Dugué  

**Link**: [PDF](https://arxiv.org/pdf/2509.24793)  

**Abstract**: Audio pretrained models are widely employed to solve various tasks in speech processing, sound event detection, or music information retrieval. However, the representations learned by these models are unclear, and their analysis mainly restricts to linear probing of the hidden representations. In this work, we explore the use of Sparse Autoencoders (SAEs) to analyze the hidden representations of pretrained models, focusing on a case study in singing technique classification. We first demonstrate that SAEs retain both information about the original representations and class labels, enabling their internal structure to provide insights into self-supervised learning systems. Furthermore, we show that SAEs enhance the disentanglement of vocal attributes, establishing them as an effective tool for identifying the underlying factors encoded in the representations. 

---
# Quantifying Generalisation in Imitation Learning 

**Authors**: Nathan Gavenski, Odinaldo Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2509.24784)  

**Abstract**: Imitation learning benchmarks often lack sufficient variation between training and evaluation, limiting meaningful generalisation assessment. We introduce Labyrinth, a benchmarking environment designed to test generalisation with precise control over structure, start and goal positions, and task complexity. It enables verifiably distinct training, evaluation, and test settings. Labyrinth provides a discrete, fully observable state space and known optimal actions, supporting interpretability and fine-grained evaluation. Its flexible setup allows targeted testing of generalisation factors and includes variants like partial observability, key-and-door tasks, and ice-floor hazards. By enabling controlled, reproducible experiments, Labyrinth advances the evaluation of generalisation in imitation learning and provides a valuable tool for developing more robust agents. 

---
# VTPerception-R1: Enhancing Multimodal Reasoning via Explicit Visual and Textual Perceptual Grounding 

**Authors**: Yizhuo Ding, Mingkang Chen, Zhibang Feng, Tong Xiao, Wanying Qu, Wenqi Shao, Yanwei Fu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24776)  

**Abstract**: Multimodal large language models (MLLMs) often struggle to ground reasoning in perceptual evidence. We present a systematic study of perception strategies-explicit, implicit, visual, and textual-across four multimodal benchmarks and two MLLMs. Our findings show that explicit perception, especially when paired with textual cues, consistently yields the best improvements, particularly for smaller models. Based on this insight, we propose VTPerception-R1, a unified two-stage framework that decouples perception from reasoning. Stage 1 introduces perception-augmented fine-tuning, and Stage 2 applies perception-aware reinforcement learning with novel visual, textual, and consistency rewards. Experiments demonstrate that VTPerception-R1 significantly improves reasoning accuracy and robustness across diverse tasks, offering a scalable and auditable solution for perception-grounded multimodal reasoning. Our code is available at: this https URL. 

---
# VSSFlow: Unifying Video-conditioned Sound and Speech Generation via Joint Learning 

**Authors**: Xin Cheng, Yuyue Wang, Xihua Wang, Yihan Wu, Kaisi Guan, Yijing Chen, Peng Zhang, Xiaojiang Liu, Meng Cao, Ruihua Song  

**Link**: [PDF](https://arxiv.org/pdf/2509.24773)  

**Abstract**: Video-conditioned sound and speech generation, encompassing video-to-sound (V2S) and visual text-to-speech (VisualTTS) tasks, are conventionally addressed as separate tasks, with limited exploration to unify them within a signle framework. Recent attempts to unify V2S and VisualTTS face challenges in handling distinct condition types (e.g., heterogeneous video and transcript conditions) and require complex training stages. Unifying these two tasks remains an open problem. To bridge this gap, we present VSSFlow, which seamlessly integrates both V2S and VisualTTS tasks into a unified flow-matching framework. VSSFlow uses a novel condition aggregation mechanism to handle distinct input signals. We find that cross-attention and self-attention layer exhibit different inductive biases in the process of introducing condition. Therefore, VSSFlow leverages these inductive biases to effectively handle different representations: cross-attention for ambiguous video conditions and self-attention for more deterministic speech transcripts. Furthermore, contrary to the prevailing belief that joint training on the two tasks requires complex training strategies and may degrade performance, we find that VSSFlow benefits from the end-to-end joint learning process for sound and speech generation without extra designs on training stages. Detailed analysis attributes it to the learned general audio prior shared between tasks, which accelerates convergence, enhances conditional generation, and stabilizes the classifier-free guidance process. Extensive experiments demonstrate that VSSFlow surpasses the state-of-the-art domain-specific baselines on both V2S and VisualTTS benchmarks, underscoring the critical potential of unified generative models. 

---
# Surjective Independence of Causal Influences for Local Bayesian Network Structures 

**Authors**: Kieran Drury, Martine J. Barons, Jim Q. Smith  

**Link**: [PDF](https://arxiv.org/pdf/2509.24759)  

**Abstract**: The very expressiveness of Bayesian networks can introduce fresh challenges due to the large number of relationships they often model. In many domains, it is thus often essential to supplement any available data with elicited expert judgements. This in turn leads to two key challenges: the cognitive burden of these judgements is often very high, and there are a very large number of judgements required to obtain a full probability model. We can mitigate both issues by introducing assumptions such as independence of causal influences (ICI) on the local structures throughout the network, restricting the parameter space of the model. However, the assumption of ICI is often unjustified and overly strong. In this paper, we introduce the surjective independence of causal influences (SICI) model which relaxes the ICI assumption and provides a more viable, practical alternative local structure model that facilitates efficient Bayesian network parameterisation. 

---
# Robust Policy Expansion for Offline-to-Online RL under Diverse Data Corruption 

**Authors**: Longxiang He, Deheng Ye, Junbo Tan, Xueqian Wang, Li Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24748)  

**Abstract**: Pretraining a policy on offline data followed by fine-tuning through online interactions, known as Offline-to-Online Reinforcement Learning (O2O RL), has emerged as a promising paradigm for real-world RL deployment. However, both offline datasets and online interactions in practical environments are often noisy or even maliciously corrupted, severely degrading the performance of O2O RL. Existing works primarily focus on mitigating the conservatism of offline policies via online exploration, while the robustness of O2O RL under data corruption, including states, actions, rewards, and dynamics, is still unexplored. In this work, we observe that data corruption induces heavy-tailed behavior in the policy, thereby substantially degrading the efficiency of online exploration. To address this issue, we incorporate Inverse Probability Weighted (IPW) into the online exploration policy to alleviate heavy-tailedness, and propose a novel, simple yet effective method termed $\textbf{RPEX}$: $\textbf{R}$obust $\textbf{P}$olicy $\textbf{EX}$pansion. Extensive experimental results on D4RL datasets demonstrate that RPEX achieves SOTA O2O performance across a wide range of data corruption scenarios. Code is available at $\href{this https URL}{this https URL}$. 

---
# A TRIANGLE Enables Multimodal Alignment Beyond Cosine Similarity 

**Authors**: Giordano Cicchetti, Eleonora Grassucci, Danilo Comminiello  

**Link**: [PDF](https://arxiv.org/pdf/2509.24734)  

**Abstract**: Multimodal learning plays a pivotal role in advancing artificial intelligence systems by incorporating information from multiple modalities to build a more comprehensive representation. Despite its importance, current state-of-the-art models still suffer from severe limitations that prevent the successful development of a fully multimodal model. Such methods may not provide indicators that all the involved modalities are effectively aligned. As a result, some modalities may not be aligned, undermining the effectiveness of the model in downstream tasks where multiple modalities should provide additional information that the model fails to exploit. In this paper, we present TRIANGLE: TRI-modAl Neural Geometric LEarning, the novel proposed similarity measure that is directly computed in the higher-dimensional space spanned by the modality embeddings. TRIANGLE improves the joint alignment of three modalities via a triangle-area similarity, avoiding additional fusion layers or pairwise similarities. When incorporated in contrastive losses replacing cosine similarity, TRIANGLE significantly boosts the performance of multimodal modeling, while yielding interpretable alignment rationales. Extensive evaluation in three-modal tasks such as video-text and audio-text retrieval or audio-video classification, demonstrates that TRIANGLE achieves state-of-the-art results across different datasets improving the performance of cosine-based methods up to 9 points of Recall@1. 

---
# Q-Net: Transferable Queue Length Estimation via Kalman-based Neural Networks 

**Authors**: Ting Gao, Elvin Isufi, Winnie Daamen, Erik-Sander Smits, Serge Hoogendoorn  

**Link**: [PDF](https://arxiv.org/pdf/2509.24725)  

**Abstract**: Estimating queue lengths at signalized intersections remains a challenge in traffic management, especially under partially observed conditions where vehicle flows are not fully captured. This paper introduces Q-Net, a data-efficient and interpretable framework for queue length estimation that performs robustly even when traffic conservation assumptions are violated. Q-Net integrates two widely available and privacy-friendly data sources: (i) vehicle counts from loop detectors near stop lines, and (ii) aggregated floating car data (aFCD), which divides each road section into segments and provides segment-wise average speed measurements. These data sources often differ in spatial and temporal resolution, creating fusion challenges. Q-Net addresses this by employing a tailored state-space model and an AI-augmented Kalman filter, KalmanNet, which learns the Kalman gain from data without requiring prior knowledge of noise covariances or full system dynamics. We build on the vanilla KalmanNet pipeline to decouple measurement dimensionality from section length, enabling spatial transferability across road segments. Unlike black-box models, Q-Net maintains physical interpretability, with internal variables linked to real-world traffic dynamics. Evaluations on main roads in Rotterdam, the Netherlands, demonstrate that Q-Net outperforms baseline methods by over 60\% in Root Mean Square Error (RMSE), accurately tracking queue formation and dissipation while correcting aFCD-induced delays. Q-Net also demonstrates strong spatial and temporal transferability, enabling deployment without costly sensing infrastructure like cameras or radar. Additionally, we propose a real-time variant of Q-Net, highlighting its potential for integration into dynamic, queue-based traffic control systems. 

---
# Discrete Variational Autoencoding via Policy Search 

**Authors**: Michael Drolet, Firas Al-Hafez, Aditya Bhatt, Jan Peters, Oleg Arenz  

**Link**: [PDF](https://arxiv.org/pdf/2509.24716)  

**Abstract**: Discrete latent bottlenecks in variational autoencoders (VAEs) offer high bit efficiency and can be modeled with autoregressive discrete distributions, enabling parameter-efficient multimodal search with transformers. However, discrete random variables do not allow for exact differentiable parameterization; therefore, discrete VAEs typically rely on approximations, such as Gumbel-Softmax reparameterization or straight-through gradient estimates, or employ high-variance gradient-free methods such as REINFORCE that have had limited success on high-dimensional tasks such as image reconstruction. Inspired by popular techniques in policy search, we propose a training framework for discrete VAEs that leverages the natural gradient of a non-parametric encoder to update the parametric encoder without requiring reparameterization. Our method, combined with automatic step size adaptation and a transformer-based encoder, scales to challenging datasets such as ImageNet and outperforms both approximate reparameterization methods and quantization-based discrete autoencoders in reconstructing high-dimensional data from compact latent spaces, achieving a 20% improvement on FID Score for ImageNet 256. 

---
# Circuit-Aware Reward Training: A Mechanistic Framework for Longtail Robustness in RLHF 

**Authors**: Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24713)  

**Abstract**: Reinforcement Learning from Human Feedback (RLHF) reward models exhibit systematic failures on longtail distributions, leading to reward hacking and misalignment. We propose a mechanistic interpretability framework that identifies specialized neural circuits responsible for rare-event processing in reward models. Drawing from recent advances showing distributed specialization for rare tokens in language models\citep{liu2025no, liu2025emergent}, we hypothesize that reward models also develop functionally distinct circuits for longtail scenarios. Our theoretical framework establishes formal connections between circuit specialization, reward generalization bounds, and longtail performance. We introduce \textbf{Circuit-Aware Reward Training (CART)}, which uses circuit analysis to guide data augmentation, regularization, and ensemble strategies. This approach provides both theoretical insights into reward model failures and practical interventions for improving longtail robustness. 

---
# FedPOB: Sample-Efficient Federated Prompt Optimization via Bandits 

**Authors**: Pingchen Lu, Zhi Hong, Zhiwei Shang, Zhiyong Wang, Yikun Ban, Yao Shu, Min Zhang, Shuang Qiu, Zhongxiang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2509.24701)  

**Abstract**: The performance of large language models (LLMs) is highly sensitive to the input prompt, making prompt optimization a critical task. However, real-world application is hindered by three major challenges: (1) the black-box nature of powerful proprietary LLMs, (2) the need for high sample efficiency due to query costs, and (3) the desire for privacy-preserving collaboration among multiple users. To address these challenges simultaneously, we introduce a novel framework for sample-efficient federated prompt optimization based on multi-armed bandits (MABs). The MAB framework is uniquely suited for this problem as it is (1) inherently a black-box optimization method, (2) practically sample-efficient, and (3) enables collaborative learning with theoretically guaranteed benefit from more participating agents. We first propose the Federated Prompt Optimization via Bandits (FedPOB) algorithm, a federated variant of the Linear UCB algorithm, where agents collaborate by sharing model parameters instead of raw data. We then extend our approach to the practical setting of comparative user feedback by introducing FedPOB with Preference Feedback (FedPOB-Pref), an efficient algorithm based on federated dueling bandits. Extensive experiments demonstrate that both FedPOB and FedPOB-Pref significantly outperform existing baselines and that their performance consistently improves as more agents participate in the collaboration, validating the effectiveness of our federated approach. 

---
# T-POP: Test-Time Personalization with Online Preference Feedback 

**Authors**: Zikun Qu, Min Zhang, Mingze Kong, Xiang Li, Zhiwei Shang, Zhiyong Wang, Yikun Ban, Shuang Qiu, Yao Shu, Zhongxiang Dai  

**Link**: [PDF](https://arxiv.org/pdf/2509.24696)  

**Abstract**: Personalizing large language models (LLMs) to individual user preferences is a critical step beyond generating generically helpful responses. However, current personalization methods are ill-suited for new users, as they typically require either slow, resource-intensive fine-tuning or a substantial amount of pre-existing user data, creating a significant cold-start problem. To address this challenge, we introduce a new paradigm for real-time personalization by learning from online pairwise preference feedback collected during text generation. We propose T-POP (Test-Time Personalization with Online Preference Feedback}), a novel algorithm that synergistically combines test-time alignment with dueling bandits. Without updating the LLM parameters, T-POP steers the decoding process of a frozen LLM by learning a reward function online that captures user preferences. By leveraging dueling bandits, T-POP intelligently queries the user to efficiently balance between exploring their preferences and exploiting the learned knowledge to generate personalized text. Extensive experiments demonstrate that T-POP achieves rapid and data-efficient personalization, significantly outperforming existing baselines and showing consistent improvement with more user interactions. 

---
# SANA-Video: Efficient Video Generation with Block Linear Diffusion Transformer 

**Authors**: Junsong Chen, Yuyang Zhao, Jincheng Yu, Ruihang Chu, Junyu Chen, Shuai Yang, Xianbang Wang, Yicheng Pan, Daquan Zhou, Huan Ling, Haozhe Liu, Hongwei Yi, Hao Zhang, Muyang Li, Yukang Chen, Han Cai, Sanja Fidler, Ping Luo, Song Han, Enze Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.24695)  

**Abstract**: We introduce SANA-Video, a small diffusion model that can efficiently generate videos up to 720x1280 resolution and minute-length duration. SANA-Video synthesizes high-resolution, high-quality and long videos with strong text-video alignment at a remarkably fast speed, deployable on RTX 5090 GPU. Two core designs ensure our efficient, effective and long video generation: (1) Linear DiT: We leverage linear attention as the core operation, which is more efficient than vanilla attention given the large number of tokens processed in video generation. (2) Constant-Memory KV cache for Block Linear Attention: we design block-wise autoregressive approach for long video generation by employing a constant-memory state, derived from the cumulative properties of linear attention. This KV cache provides the Linear DiT with global context at a fixed memory cost, eliminating the need for a traditional KV cache and enabling efficient, minute-long video generation. In addition, we explore effective data filters and model training strategies, narrowing the training cost to 12 days on 64 H100 GPUs, which is only 1% of the cost of MovieGen. Given its low cost, SANA-Video achieves competitive performance compared to modern state-of-the-art small diffusion models (e.g., Wan 2.1-1.3B and SkyReel-V2-1.3B) while being 16x faster in measured latency. Moreover, SANA-Video can be deployed on RTX 5090 GPUs with NVFP4 precision, accelerating the inference speed of generating a 5-second 720p video from 71s to 29s (2.4x speedup). In summary, SANA-Video enables low-cost, high-quality video generation. 

---
# CoTune: Co-evolutionary Configuration Tuning 

**Authors**: Gangda Xiong, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24694)  

**Abstract**: To automatically tune configurations for the best possible system performance (e.g., runtime or throughput), much work has been focused on designing intelligent heuristics in a tuner. However, existing tuner designs have mostly ignored the presence of complex performance requirements (e.g., the latency shall ideally be 2 seconds), but simply assume that better performance is always more preferred. This would not only waste valuable information in a requirement but might also consume extensive resources to tune for a goal with little gain. Yet, prior studies have shown that simply incorporating the requirement as a tuning objective is problematic since the requirement might be too strict, harming convergence; or its highly diverse satisfactions might lead to premature convergence. In this paper, we propose CoTune, a tool that takes the information of a given target performance requirement into account through co-evolution. CoTune is unique in the sense that it creates an auxiliary performance requirement to be co-evolved with the configurations, which assists the target performance requirement when it becomes ineffective or even misleading, hence allowing the tuning to be guided by the requirement while being robust to its harm. Experiment results on 162 cases (nine systems and 18 requirements) reveal that CoTune considerably outperforms existing tuners, ranking as the best for 90% cases (against the 0%--35% for other tuners) with up to 2.9x overall improvements, while doing so under a much better efficiency. 

---
# Data-Driven Discrete Geofence Design Using Binary Quadratic Programming 

**Authors**: Keisuke Otaki, Akihisa Okada, Tadayoshi Matsumori, Hiroaki Yoshida  

**Link**: [PDF](https://arxiv.org/pdf/2509.24679)  

**Abstract**: Geofences have attracted significant attention in the design of spatial and virtual regions for managing and engaging spatiotemporal events. By using geofences to monitor human activity across their boundaries, content providers can create spatially triggered events that include notifications about points of interest within a geofence by pushing spatial information to the devices of users. Traditionally, geofences were hand-crafted by providers. In addition to the hand-crafted approach, recent advances in collecting human mobility data through mobile devices can accelerate the automatic and data-driven design of geofences, also known as the geofence design problem. Previous approaches assume circular shapes; thus, their flexibility is insufficient, and they can only handle geofence-based applications for large areas with coarse resolutions. A challenge with using circular geofences in urban and high-resolution areas is that they often overlap and fail to align with political district boundaries and road segments, such as one-way streets and median barriers. In this study, we address the problem of extracting arbitrary shapes as geofences from human mobility data to mitigate this problem. In our formulation, we cast the existing optimization problems for circular geofences to 0-1 integer programming problems to represent arbitrary shapes. Although 0-1 integer programming problems are computationally hard, formulating them as quadratic (unconstrained) binary optimization problems enables efficient approximation of optimal solutions, because this allows the use of specialized quadratic solvers, such as the quantum annealing, and other state-of-the-art algorithms. We then develop and compare different formulation methods to extract discrete geofences. We confirmed that our new modeling approach enables flexible geofence design. 

---
# Reference-Free Rating of LLM Responses via Latent Information 

**Authors**: Leander Girrbach, Chi-Ping Su, Tankred Saanum, Richard Socher, Eric Schulz, Zeynep Akata  

**Link**: [PDF](https://arxiv.org/pdf/2509.24678)  

**Abstract**: How reliable are single-response LLM-as-a-judge ratings without references, and can we obtain fine-grained, deterministic scores in this setting? We study the common practice of asking a judge model to assign Likert-scale scores to free-text responses and show two systematic issues: scores are unstable under sampling and poorly calibrated, leading to compression near the top of the scale and frequent ties. We then propose and evaluate Latent Judges, which derive scalar ratings from internal model signals: (i) probability-weighted scores over integer ratings, (ii) verifier-style probabilities of "yes", and (iii) linear probes trained on model activations at the rating position. Across a broad suite of pairwise and single-rating benchmarks, latent methods match or surpass standard prompting, with consistent gains on pairwise accuracy and listwise ranking relevant to Best-of-N selection. Probability-weighted scores achieve the strongest single-rating correlations, while probes recover useful signals when output logits are miscalibrated. These results indicate that latent information provides deterministic and more discriminative signals for reference-free evaluation, and can improve selection and training approaches like Best-of-$N$, multi-teacher distillation, and routing. 

---
# Understanding the Dilemma of Unlearning for Large Language Models 

**Authors**: Qingjie Zhang, Haoting Qian, Zhicong Huang, Cheng Hong, Minlie Huang, Ke Xu, Chao Zhang, Han Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24675)  

**Abstract**: Unlearning seeks to remove specific knowledge from large language models (LLMs), but its effectiveness remains contested. On one side, "forgotten" knowledge can often be recovered through interventions such as light fine-tuning; on the other side, unlearning may induce catastrophic forgetting that degrades general capabilities. Despite active exploration of unlearning methods, interpretability analyses of the mechanism are scarce due to the difficulty of tracing knowledge in LLMs' complex architectures. We address this gap by proposing unPact, an interpretable framework for unlearning via prompt attribution and contribution tracking. Typically, it quantifies each prompt token's influence on outputs, enabling pre- and post-unlearning comparisons to reveal what changes. Across six mainstream unlearning methods, three LLMs, and three benchmarks, we find that: (1) Unlearning appears to be effective by disrupting focus on keywords in prompt; (2) Much of the knowledge is not truly erased and can be recovered by simply emphasizing these keywords in prompts, without modifying the model's weights; (3) Catastrophic forgetting arises from indiscriminate penalization of all tokens. Taken together, our results suggest an unlearning dilemma: existing methods tend either to be insufficient - knowledge remains recoverable by keyword emphasis, or overly destructive - general performance collapses due to catastrophic forgetting, still leaving a gap to reliable unlearning. 

---
# InfLLM-V2: Dense-Sparse Switchable Attention for Seamless Short-to-Long Adaptation 

**Authors**: Weilin Zhao, Zihan Zhou, Zhou Su, Chaojun Xiao, Yuxuan Li, Yanghao Li, Yudi Zhang, Weilun Zhao, Zhen Li, Yuxiang Huang, Ao Sun, Xu Han, Zhiyuan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24663)  

**Abstract**: Long-sequence processing is a critical capability for modern large language models. However, the self-attention mechanism in the standard Transformer architecture faces severe computational and memory bottlenecks when processing long sequences. While trainable sparse attention methods offer a promising solution, existing approaches such as NSA introduce excessive extra parameters and disrupt the conventional \textit{pretrain-on-short, finetune-on-long} workflow, resulting in slow convergence and difficulty in acceleration. To overcome these limitations, we introduce dense-sparse switchable attention framework, termed as InfLLM-V2. InfLLM-V2 is a trainable sparse attention that seamlessly adapts models from short to long sequences. Specifically, InfLLM-V2 reuses dense attention parameters through parameter-free architecture modification, maintaining consistency between short and long sequence processing. Additionally, InfLLM-V2 ensures computational efficiency across all sequence lengths, by using dense attention for short inputs and smoothly transitioning to sparse attention for long sequences. To achieve practical acceleration, we further introduce an efficient implementation of InfLLM-V2 that significantly reduces the computational overhead. Our experiments on long-context understanding and chain-of-thought reasoning demonstrate that InfLLM-V2 is 4$\times$ faster than dense attention while retaining 98.1% and 99.7% of the performance, respectively. Based on the InfLLM-V2 framework, we have trained and open-sourced MiniCPM4.1 (this https URL), a hybrid reasoning model, providing a reproducible implementation for the research community. 

---
# Community detection robustness of graph neural networks 

**Authors**: Jaidev Goel, Pablo Moriano, Ramakrishnan Kannan, Yulia R. Gel  

**Link**: [PDF](https://arxiv.org/pdf/2509.24662)  

**Abstract**: Graph neural networks (GNNs) are increasingly widely used for community detection in attributed networks. They combine structural topology with node attributes through message passing and pooling. However, their robustness or lack of thereof with respect to different perturbations and targeted attacks in conjunction with community detection tasks is not well understood. To shed light into latent mechanisms behind GNN sensitivity on community detection tasks, we conduct a systematic computational evaluation of six widely adopted GNN architectures: GCN, GAT, Graph- SAGE, DiffPool, MinCUT, and DMoN. The analysis covers three perturbation categories: node attribute manipulations, edge topology distortions, and adversarial attacks. We use element-centric similarity as the evaluation metric on synthetic benchmarks and real-world citation networks. Our findings indicate that supervised GNNs tend to achieve higher baseline accuracy, while unsupervised methods, particularly DMoN, maintain stronger resilience under targeted and adversarial pertur- bations. Furthermore, robustness appears to be strongly influenced by community strength, with well-defined communities reducing performance loss. Across all models, node attribute perturba- tions associated with targeted edge deletions and shift in attribute distributions tend to cause the largest degradation in community recovery. These findings highlight important trade-offs between accuracy and robustness in GNN-based community detection and offer new insights into selecting architectures resilient to noise and adversarial attacks. 

---
# VNODE: A Piecewise Continuous Volterra Neural Network 

**Authors**: Siddharth Roheda, Aniruddha Bala, Rohit Chowdhury, Rohan Jaiswal  

**Link**: [PDF](https://arxiv.org/pdf/2509.24659)  

**Abstract**: This paper introduces Volterra Neural Ordinary Differential Equations (VNODE), a piecewise continuous Volterra Neural Network that integrates nonlinear Volterra filtering with continuous time neural ordinary differential equations for image classification. Drawing inspiration from the visual cortex, where discrete event processing is interleaved with continuous integration, VNODE alternates between discrete Volterra feature extraction and ODE driven state evolution. This hybrid formulation captures complex patterns while requiring substantially fewer parameters than conventional deep architectures. VNODE consistently outperforms state of the art models with improved computational complexity as exemplified on benchmark datasets like CIFAR10 and Imagenet1K. 

---
# Identity Bridge: Enabling Implicit Reasoning via Shared Latent Memory 

**Authors**: Pengxiao Lin, Zheng-An Chen, Zhi-Qin John Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24653)  

**Abstract**: Despite remarkable advances, large language models often fail at compositional reasoning tasks, a phenomenon exemplified by the ``curse of two-hop reasoning''. This paper introduces the Identity Bridge, a simple yet powerful mechanism that resolves this compositionality gap by supervising the model on a zero-hop identity task. We demonstrate empirically that this addition enables models to successfully perform out-of-distribution two-hop reasoning, a task they otherwise completely fail. To explain this phenomenon, we provide a theoretical analysis using a simplified Emb-MLP model, proving that identity supervision reshapes the model's latent geometry. We show this alignment is induced by an implicit nuclear-norm regularization during optimization, which favors low-rank solutions that share structure across tasks. For complex tasks, we use small initialization or weight decay to enhance the regularization effect, which enhances the latent space alignment effect and slows down the generalization decay. Finally, we extend our investigation to large-scale models, observing that they still achieve two-hop reasoning through the latent memory, which provides crucial inspiration for enhancing their implicit reasoning abilities. 

---
# Can you SPLICE it together? A Human Curated Benchmark for Probing Visual Reasoning in VLMs 

**Authors**: Mohamad Ballout, Okajevo Wilfred, Seyedalireza Yaghoubi, Nohayr Muhammad Abdelmoneim, Julius Mayer, Elia Bruni  

**Link**: [PDF](https://arxiv.org/pdf/2509.24640)  

**Abstract**: In this work, we introduce SPLICE, a human-curated benchmark derived from the COIN instructional video dataset, designed to probe event-based reasoning across multiple dimensions: temporal, causal, spatial, contextual, and general knowledge. SPLICE includes 3,381 human-filtered videos spanning 12 categories and 180 sub-categories, such as sports, engineering, and housework. These videos are segmented into a total of 11,423 event clips. We evaluate both human participants and state-of-the-art vision-language models (VLMs) on the task of rearranging these clips into coherent event sequences to assess visual reasoning capabilities. Results reveal a significant gap: VLMs struggle to match human performance. While human-annotated textual descriptions improve model accuracy, they do not affect human performance, suggesting that models rely more on language priors than on visual understanding. Even with annotations, VLMs fall short of human-level reasoning, underscoring persistent challenges in visual reasoning. A deeper analysis across sub-categories shows that VLMs perform relatively better on videos where temporal and causal reasoning are dominant, compared to those where contextual and spatial reasoning are dominant. They also perform better on everyday tasks than on specialized ones. 

---
# Algorithms and data structures for automatic precision estimation of neural networks 

**Authors**: Igor V. Netay  

**Link**: [PDF](https://arxiv.org/pdf/2509.24607)  

**Abstract**: We describe algorithms and data structures to extend a neural network library with automatic precision estimation for floating point computations. We also discuss conditions to make estimations exact and preserve high computation performance of neural networks training and inference. Numerical experiments show the consequences of significant precision loss for particular values such as inference, gradients and deviations from mathematically predicted behavior.
It turns out that almost any neural network accumulates computational inaccuracies. As a result, its behavior does not coincide with predicted by the mathematical model of neural network. This shows that tracking of computational inaccuracies is important for reliability of inference, training and interpretability of results. 

---
# PoseDiff: A Unified Diffusion Model Bridging Robot Pose Estimation and Video-to-Action Control 

**Authors**: Haozhuo Zhang, Michele Caprio, Jing Shao, Qiang Zhang, Jian Tang, Shanghang Zhang, Wei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24591)  

**Abstract**: We present PoseDiff, a conditional diffusion model that unifies robot state estimation and control within a single framework. At its core, PoseDiff maps raw visual observations into structured robot states-such as 3D keypoints or joint angles-from a single RGB image, eliminating the need for multi-stage pipelines or auxiliary modalities. Building upon this foundation, PoseDiff extends naturally to video-to-action inverse dynamics: by conditioning on sparse video keyframes generated by world models, it produces smooth and continuous long-horizon action sequences through an overlap-averaging strategy. This unified design enables scalable and efficient integration of perception and control. On the DREAM dataset, PoseDiff achieves state-of-the-art accuracy and real-time performance for pose estimation. On Libero-Object manipulation tasks, it substantially improves success rates over existing inverse dynamics modules, even under strict offline settings. Together, these results show that PoseDiff provides a scalable, accurate, and efficient bridge between perception, planning, and control in embodied AI. The video visualization results can be found on the project page: this https URL. 

---
# Bandits roaming Hilbert space 

**Authors**: Josep Lumbreras  

**Link**: [PDF](https://arxiv.org/pdf/2509.24569)  

**Abstract**: This thesis studies the exploration and exploitation trade-off in online learning of properties of quantum states using multi-armed bandits. Given streaming access to an unknown quantum state, in each round we select an observable from a set of actions to maximize its expectation value. Using past information, we refine actions to minimize regret; the cumulative gap between current reward and the maximum possible. We derive information-theoretic lower bounds and optimal strategies with matching upper bounds, showing regret typically scales as the square root of rounds. As an application, we reframe quantum state tomography to both learn the state efficiently and minimize measurement disturbance. For pure states and continuous actions, we achieve polylogarithmic regret using a sample-optimal algorithm based on a weighted online least squares estimator. The algorithm relies on the optimistic principle and controls the eigenvalues of the design matrix. We also apply our framework to quantum recommender systems and thermodynamic work extraction from unknown states. In this last setting, our results demonstrate an exponential advantage in work dissipation over tomography-based protocols. 

---
# AdaThink-Med: Medical Adaptive Thinking with Uncertainty-Guided Length Calibration 

**Authors**: Shaohao Rui, Kaitao Chen, Weijie Ma, Xiaosong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24560)  

**Abstract**: Recent advances in inference time scaling with extended long chain-of thought have significantly improved the reasoning capabilities of both general and medical large language models (LLMs). However, these models tend to engage in lengthy reasoning processes regardless of the difficulty of the input question, leading to increased inference costs in real-world applications. Therefore, enabling adaptive thinking where models think less for simpler questions and think more for complex ones is critical for the effective use of medical LLMs in practice. Despite its importance, there is a lack of end-to-end approaches designed to enhance the adaptive thinking capabilities of medical LLMs while providing a comprehensive examination of the trade-off between performance and computational cost. To bridge this gap, we propose AdaThink-Med, the first end-to-end framework designed to enhance adaptive thinking ability in medical reasoning models with uncertainty-guided length calibration. AdaThink-Med first generates multiple candidate outputs for each question, evaluates the correctness and uncertainty of each candidate, and then estimates problem difficulty via an uncertainty-guided length calibration module. For outputs with low difficulty and correct answers, the framework penalizes longer reasoning paths; whereas for those with high difficulty and incorrect answers, it encourages extending the chain of thought to explore alternative solutions. On six public medical QA benchmarks, AdaThink-Med achieves up to 6.4x length reduction on average while retaining performance with only minimal degradation. Intriguingly, we observe that AdaThink-Med spontaneously develops two distinct reasoning modes, which we characterize as "non-thinking" and "thinking", demonstrating the model's ability to suppress redundant reasoning processes dynamically. 

---
# Deep Reinforcement Learning in Action: Real-Time Control of Vortex-Induced Vibrations 

**Authors**: Hussam Sababha, Bernat Font, Mohammed Daqaq  

**Link**: [PDF](https://arxiv.org/pdf/2509.24556)  

**Abstract**: This study showcases an experimental deployment of deep reinforcement learning (DRL) for active flow control (AFC) of vortex-induced vibrations (VIV) in a circular cylinder at a high Reynolds number (Re = 3000) using rotary actuation. Departing from prior work that relied on low-Reynolds-number numerical simulations, this research demonstrates real-time control in a challenging experimental setting, successfully addressing practical constraints such as actuator delay. When the learning algorithm is provided with state feedback alone (displacement and velocity of the oscillating cylinder), the DRL agent learns a low-frequency rotary control strategy that achieves up to 80% vibration suppression which leverages the traditional lock-on phenomenon. While this level of suppression is significant, it remains below the performance achieved using high-frequency rotary actuation. The reduction in performance is attributed to actuation delays and can be mitigated by augmenting the learning algorithm with past control actions. This enables the agent to learn a high-frequency rotary control strategy that effectively modifies vortex shedding and achieves over 95% vibration attenuation. These results demonstrate the adaptability of DRL for AFC in real-world experiments and its ability to overcome instrumental limitations such as actuation lag. 

---
# Short window attention enables long-term memorization 

**Authors**: Loïc Cabannes, Maximilian Beck, Gergely Szilvasy, Matthijs Douze, Maria Lomeli, Jade Copet, Pierre-Emmanuel Mazaré, Gabriel Synnaeve, Hervé Jégou  

**Link**: [PDF](https://arxiv.org/pdf/2509.24552)  

**Abstract**: Recent works show that hybrid architectures combining sliding window softmax attention layers with linear recurrent neural network (RNN) layers outperform both of these architectures taken separately. However, the impact of the window length and the interplay between softmax attention and linear RNN layers remain under-studied. In this work, we introduce SWAX, a hybrid architecture consisting of sliding-window attention and xLSTM linear RNN layers.
A counter-intuitive finding with SWAX is that larger sliding windows do not improve the long-context performance. In fact, short window attention encourages the model to better train the long-term memory of the xLSTM, by relying less on the softmax attention mechanism for long context-retrieval.
The issue with small sliding windows is that they are detrimental for short-context tasks, which could be solved with information from moderately larger sliding windows otherwise. Therefore, we train SWAX by stochastically changing the sliding window size, forcing the model to leverage both a longer context window and the xLSTM memory. SWAX trained with stochastic window sizes significantly outperforms regular window attention both on short and long-context problems. 

---
# CORE-3D: Context-aware Open-vocabulary Retrieval by Embeddings in 3D 

**Authors**: Mohamad Amin Mirzaei, Pantea Amoie, Ali Ekhterachian, Matin Mirzababaei  

**Link**: [PDF](https://arxiv.org/pdf/2509.24528)  

**Abstract**: 3D scene understanding is fundamental for embodied AI and robotics, supporting reliable perception for interaction and navigation. Recent approaches achieve zero-shot, open-vocabulary 3D semantic mapping by assigning embedding vectors to 2D class-agnostic masks generated via vision-language models (VLMs) and projecting these into 3D. However, these methods often produce fragmented masks and inaccurate semantic assignments due to the direct use of raw masks, limiting their effectiveness in complex environments. To address this, we leverage SemanticSAM with progressive granularity refinement to generate more accurate and numerous object-level masks, mitigating the over-segmentation commonly observed in mask generation models such as vanilla SAM, and improving downstream 3D semantic segmentation. To further enhance semantic context, we employ a context-aware CLIP encoding strategy that integrates multiple contextual views of each mask using empirically determined weighting, providing much richer visual context. We evaluate our approach on multiple 3D scene understanding tasks, including 3D semantic segmentation and object retrieval from language queries, across several benchmark datasets. Experimental results demonstrate significant improvements over existing methods, highlighting the effectiveness of our approach. 

---
# CMT: Mid-Training for Efficient Learning of Consistency, Mean Flow, and Flow Map Models 

**Authors**: Zheyuan Hu, Chieh-Hsin Lai, Yuki Mitsufuji, Stefano Ermon  

**Link**: [PDF](https://arxiv.org/pdf/2509.24526)  

**Abstract**: Flow map models such as Consistency Models (CM) and Mean Flow (MF) enable few-step generation by learning the long jump of the ODE solution of diffusion models, yet training remains unstable, sensitive to hyperparameters, and costly. Initializing from a pre-trained diffusion model helps, but still requires converting infinitesimal steps into a long-jump map, leaving instability unresolved. We introduce mid-training, the first concept and practical method that inserts a lightweight intermediate stage between the (diffusion) pre-training and the final flow map training (i.e., post-training) for vision generation. Concretely, Consistency Mid-Training (CMT) is a compact and principled stage that trains a model to map points along a solver trajectory from a pre-trained model, starting from a prior sample, directly to the solver-generated clean sample. It yields a trajectory-consistent and stable initialization. This initializer outperforms random and diffusion-based baselines and enables fast, robust convergence without heuristics. Initializing post-training with CMT weights further simplifies flow map learning. Empirically, CMT achieves state of the art two step FIDs: 1.97 on CIFAR-10, 1.32 on ImageNet 64x64, and 1.84 on ImageNet 512x512, while using up to 98% less training data and GPU time, compared to CMs. On ImageNet 256x256, CMT reaches 1-step FID 3.34 while cutting total training time by about 50% compared to MF from scratch (FID 3.43). This establishes CMT as a principled, efficient, and general framework for training flow map models. 

---
# PhysiAgent: An Embodied Agent Framework in Physical World 

**Authors**: Zhihao Wang, Jianxiong Li, Jinliang Zheng, Wencong Zhang, Dongxiu Liu, Yinan Zheng, Haoyi Niu, Junzhi Yu, Xianyuan Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24524)  

**Abstract**: Vision-Language-Action (VLA) models have achieved notable success but often struggle with limited generalizations. To address this, integrating generalized Vision-Language Models (VLMs) as assistants to VLAs has emerged as a popular solution. However, current approaches often combine these models in rigid, sequential structures: using VLMs primarily for high-level scene understanding and task planning, and VLAs merely as executors of lower-level actions, leading to ineffective collaboration and poor grounding challenges. In this paper, we propose an embodied agent framework, PhysiAgent, tailored to operate effectively in physical environments. By incorporating monitor, memory, self-reflection mechanisms, and lightweight off-the-shelf toolboxes, PhysiAgent offers an autonomous scaffolding framework to prompt VLMs to organize different components based on real-time proficiency feedback from VLAs to maximally exploit VLAs' capabilities. Experimental results demonstrate significant improvements in task-solving performance on complex real-world robotic tasks, showcasing effective self-regulation of VLMs, coherent tool collaboration, and adaptive evolution of the framework during execution. PhysiAgent makes practical and pioneering efforts to integrate VLMs and VLAs, effectively grounding embodied agent frameworks in real-world settings. 

---
# Agentic Specification Generator for Move Programs 

**Authors**: Yu-Fu Fu, Meng Xu, Taesoo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.24515)  

**Abstract**: While LLM-based specification generation is gaining traction, existing tools primarily focus on mainstream programming languages like C, Java, and even Solidity, leaving emerging and yet verification-oriented languages like Move underexplored. In this paper, we introduce MSG, an automated specification generation tool designed for Move smart contracts. MSG aims to highlight key insights that uniquely present when applying LLM-based specification generation to a new ecosystem. Specifically, MSG demonstrates that LLMs exhibit robust code comprehension and generation capabilities even for non-mainstream languages. MSG successfully generates verifiable specifications for 84% of tested Move functions and even identifies clauses previously overlooked by experts. Additionally, MSG shows that explicitly leveraging specification language features through an agentic, modular design improves specification quality substantially (generating 57% more verifiable clauses than conventional designs). Incorporating feedback from the verification toolchain further enhances the effectiveness of MSG, leading to a 30% increase in generated verifiable specifications. 

---
# Specialization after Generalization: Towards Understanding Test-Time Training in Foundation Models 

**Authors**: Jonas Hübotter, Patrik Wolf, Alexander Shevchenko, Dennis Jüni, Andreas Krause, Gil Kur  

**Link**: [PDF](https://arxiv.org/pdf/2509.24510)  

**Abstract**: Recent empirical studies have explored the idea of continuing to train a model at test-time for a given task, known as test-time training (TTT), and have found it to yield significant performance improvements. However, there is limited understanding of why and when TTT is effective. Earlier explanations mostly focused on the observation that TTT may help when applied to out-of-distribution adaptation or used with privileged data. However, the growing scale of foundation models with most test data being in-distribution questions these explanations. We instead posit that foundation models remain globally underparameterized, with TTT providing a mechanism for specialization after generalization, focusing capacity on concepts relevant to the test task. Specifically, under the linear representation hypothesis, we propose a model in which TTT achieves a substantially smaller in-distribution test error than global training. We empirically validate our model's key assumptions by training a sparse autoencoder on ImageNet, showing that semantically related data points are explained by only a few shared concepts. Finally, we perform scaling studies across image and language tasks that confirm the practical implications of our model, identifying the regimes where specialization is most effective. 

---
# LLM DNA: Tracing Model Evolution via Functional Representations 

**Authors**: Zhaomin Wu, Haodong Zhao, Ziyang Wang, Jizhou Guo, Qian Wang, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2509.24496)  

**Abstract**: The explosive growth of large language models (LLMs) has created a vast but opaque landscape: millions of models exist, yet their evolutionary relationships through fine-tuning, distillation, or adaptation are often undocumented or unclear, complicating LLM management. Existing methods are limited by task specificity, fixed model sets, or strict assumptions about tokenizers or architectures. Inspired by biological DNA, we address these limitations by mathematically defining LLM DNA as a low-dimensional, bi-Lipschitz representation of functional behavior. We prove that LLM DNA satisfies inheritance and genetic determinism properties and establish the existence of DNA. Building on this theory, we derive a general, scalable, training-free pipeline for DNA extraction. In experiments across 305 LLMs, DNA aligns with prior studies on limited subsets and achieves superior or competitive performance on specific tasks. Beyond these tasks, DNA comparisons uncover previously undocumented relationships among LLMs. We further construct the evolutionary tree of LLMs using phylogenetic algorithms, which align with shifts from encoder-decoder to decoder-only architectures, reflect temporal progression, and reveal distinct evolutionary speeds across LLM families. 

---
# Mitigating Visual Hallucinations via Semantic Curriculum Preference Optimization in MLLMs 

**Authors**: Yuanshuai Li, Yuping Yan, Junfeng Tang, Yunxuan Li, Zeqi Zheng, Yaochu Jin  

**Link**: [PDF](https://arxiv.org/pdf/2509.24491)  

**Abstract**: Multimodal Large Language Models (MLLMs) have significantly improved the performance of various tasks, but continue to suffer from visual hallucinations, a critical issue where generated responses contradict visual evidence. While Direct Preference Optimization(DPO) is widely used for alignment, its application to MLLMs often fails to capture fine-grained semantic differences and encourages shortcut learning. To address these challenges, we propose Semantic Curriculum Preference Optimization (SCPO), a novel framework for MLLM alignment. SCPO employs a progressive, easy-to-hard curriculum built upon our Semantic Curriculum Preference Pairs dataset, which provides fine-grained semantic contrasts sorted by difficulty. This curriculum is trained with a dynamic reference model and a novel symmetric, bidirectional objective to facilitate simultaneous learning from both textual and visual preferences. To our knowledge, SCPO is the first framework to unify semantics, symmetry, and curriculum for MLLMs alignment, effectively mitigating visual hallucinations. Extensive experiments on LLaVA models across various scales and versions validate that SCPO demonstrates superior performance compared to baseline models on multiple hallucination benchmarks, reducing the hallucination rate by up to 62.9%. Moreover, evaluations on generalized benchmarks show that SCPO improves factuality while preserving general capabilities, with its performance remaining stable across general vision-language benchmarks. 

---
# Euclid's Gift: Enhancing Spatial Perception and Reasoning in Vision-Language Models via Geometric Surrogate Tasks 

**Authors**: Shijie Lian, Changti Wu, Laurence Tianruo Yang, Hang Yuan, Bin Yu, Lei Zhang, Kai Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24473)  

**Abstract**: Spatial intelligence spans a rich suite of abilities, including visualising and transforming shapes, mentally rotating objects, judging relational positions and containment, and estimating numerosity. However, it still remains a critical unresolved challenge for Multimodal Large Language Models (MLLMs).To fill this gap, we propose to treat Euclidean geometry problem-solving as a surrogate task. Specifically, we meticulously constructed a curated multimodal dataset, called Euclid30K, comprising approximately 30K plane and solid geometry problems. To enable the model to acquire and apply Euclidean principles from these geometry problems, we employed Group Relative Policy Optimization (GRPO) to finetune the Qwen2.5VL family and RoboBrain2.0 family, inspiring the models to identify shapes, count, and relate entities, and perform multi-step deductive reasoning using Euclidean principles. Our experiments demonstrate that the resulting models achieve substantial zero-shot gains across four spatial reasoning benchmarks (Super-CLEVR, Omni3DBench, VSI-Bench, and MindCube) without any task-specific adaptations. Notably, after training on the Euclid30K, the mean VSI-Bench accuracy of all evaluated models rose from 34.5% to 40.5%, improving by 5.5 percentage points. Among them, RoboBrain2.0-Euclid-7B achieves 49.6\% accuracy, surpassing the previous state-of-the-art model, this http URL our knowledge, this is the first systematic study showing that geometry-centric fine-tuning can confer vision-language models with broadly transferable spatial skills. Code and Euclid30K dataset can be found in this https URL. 

---
# LaMoGen: Laban Movement-Guided Diffusion for Text-to-Motion Generation 

**Authors**: Heechang Kim, Gwanghyun Kim, Se Young Chun  

**Link**: [PDF](https://arxiv.org/pdf/2509.24469)  

**Abstract**: Diverse human motion generation is an increasingly important task, having various applications in computer vision, human-computer interaction and animation. While text-to-motion synthesis using diffusion models has shown success in generating high-quality motions, achieving fine-grained expressive motion control remains a significant challenge. This is due to the lack of motion style diversity in datasets and the difficulty of expressing quantitative characteristics in natural language. Laban movement analysis has been widely used by dance experts to express the details of motion including motion quality as consistent as possible. Inspired by that, this work aims for interpretable and expressive control of human motion generation by seamlessly integrating the quantification methods of Laban Effort and Shape components into the text-guided motion generation models. Our proposed zero-shot, inference-time optimization method guides the motion generation model to have desired Laban Effort and Shape components without any additional motion data by updating the text embedding of pretrained diffusion models during the sampling step. We demonstrate that our approach yields diverse expressive motion qualities while preserving motion identity by successfully manipulating motion attributes according to target Laban tags. 

---
# Moravec's Paradox and Restrepo's Model: Limits of AGI Automation in Growth 

**Authors**: Marc Bara  

**Link**: [PDF](https://arxiv.org/pdf/2509.24466)  

**Abstract**: This note extends Restrepo (2025)'s model of economic growth under AGI by incorporating Moravec's Paradox -the observation that tasks requiring sensorimotor skills remain computationally expensive relative to cognitive tasks. We partition the task space into cognitive and physical components with differential automation costs, allowing infinite costs for some physical bottlenecks. Our key result shows that when physical tasks constitute economic bottlenecks with sufficiently high (or infinite) computational requirements, the labor share of income converges to a positive constant in the finite-compute regime (rather than zero). This fundamentally alters the distributional implications of AGI while preserving the growth dynamics for cognitive-intensive economies. 

---
# An Agent-Based Framework for Automated Higher-Voice Harmony Generation 

**Authors**: Nia D'Souza Ganapathy, Arul Selvamani Shaja  

**Link**: [PDF](https://arxiv.org/pdf/2509.24463)  

**Abstract**: The generation of musically coherent and aesthetically pleasing harmony remains a significant challenge in the field of algorithmic composition. This paper introduces an innovative Agentic AI-enabled Higher Harmony Music Generator, a multi-agent system designed to create harmony in a collaborative and modular fashion. Our framework comprises four specialized agents: a Music-Ingestion Agent for parsing and standardizing input musical scores; a Chord-Knowledge Agent, powered by a Chord-Former (Transformer model), to interpret and provide the constituent notes of complex chord symbols; a Harmony-Generation Agent, which utilizes a Harmony-GPT and a Rhythm-Net (RNN) to compose a melodically and rhythmically complementary harmony line; and an Audio-Production Agent that employs a GAN-based Symbolic-to-Audio Synthesizer to render the final symbolic output into high-fidelity audio. By delegating specific tasks to specialized agents, our system effectively mimics the collaborative process of human musicians. This modular, agent-based approach allows for robust data processing, deep theoretical understanding, creative composition, and realistic audio synthesis, culminating in a system capable of generating sophisticated and contextually appropriate higher-voice harmonies for given melodies. 

---
# EOE: Evolutionary Optimization of Experts for Training Language Models 

**Authors**: Yingshi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24436)  

**Abstract**: This paper presents an evolutionary framework for the training of large language models(LLM). The models are divided into several experts(sub-networks), which have the same structure but different parameter values. Only one expert is trained at each step. After the classical AdamW optimization, some evolutionary operators(crossover, PSO, and mutation) act on the tensor weights between the current expert and the best expert. So current expert would learn the experience of best expert. The direction of best expert would help current expert's loss decrease faster. Finally, only save the weight of the best expert. Experiments show that best expert would achieve nearly the same accuracy as the full model. This would greatly reduce the size of the model for inference. Since only one expert is trained at each step, the training needs much less memory and has much higher throughput. Experiments show that the throughput would accelerate more than ten times! Our source code is available. It's a pure c++/cu framework, which is suitable for easy deployment on PCs and edge computing devices. 

---
# Alternatives To Next Token Prediction In Text Generation - A Survey 

**Authors**: Charlie Wyatt, Aditya Joshi, Flora Salim  

**Link**: [PDF](https://arxiv.org/pdf/2509.24435)  

**Abstract**: The paradigm of Next Token Prediction (NTP) has driven the unprecedented success of Large Language Models (LLMs), but is also the source of their most persistent weaknesses such as poor long-term planning, error accumulation, and computational inefficiency. Acknowledging the growing interest in exploring alternatives to NTP, the survey describes the emerging ecosystem of alternatives to NTP. We categorise these approaches into five main families: (1) Multi-Token Prediction, which targets a block of future tokens instead of a single one; (2) Plan-then-Generate, where a global, high-level plan is created upfront to guide token-level decoding; (3) Latent Reasoning, which shifts the autoregressive process itself into a continuous latent space; (4) Continuous Generation Approaches, which replace sequential generation with iterative, parallel refinement through diffusion, flow matching, or energy-based methods; and (5) Non-Transformer Architectures, which sidestep NTP through their inherent model structure. By synthesizing insights across these methods, this survey offers a taxonomy to guide research into models that address the known limitations of token-level generation to develop new transformative models for natural language processing. 

---
# Multi-Item-Query Attention for Stable Sequential Recommendation 

**Authors**: Mingshi Xu, Haoren Zhu, Wilfred Siu Hung Ng  

**Link**: [PDF](https://arxiv.org/pdf/2509.24424)  

**Abstract**: The inherent instability and noise in user interaction data challenge sequential recommendation systems. Prevailing masked attention models, relying on a single query from the most recent item, are sensitive to this noise, reducing prediction reliability. We propose the Multi-Item-Query attention mechanism (MIQ-Attn) to enhance model stability and accuracy. MIQ-Attn constructs multiple diverse query vectors from user interactions, effectively mitigating noise and improving consistency. It is designed for easy adoption as a drop-in replacement for existing single-query attention. Experiments show MIQ-Attn significantly improves performance on benchmark datasets. 

---
# A Data-Centric Perspective on the Influence of Image Data Quality in Machine Learning Models 

**Authors**: Pei-Han Chen, Szu-Chi Chung  

**Link**: [PDF](https://arxiv.org/pdf/2509.24420)  

**Abstract**: In machine learning, research has traditionally focused on model development, with relatively less attention paid to training data. As model architectures have matured and marginal gains from further refinements diminish, data quality has emerged as a critical factor. However, systematic studies on evaluating and ensuring dataset quality in the image domain remain limited.
This study investigates methods for systematically assessing image dataset quality and examines how various image quality factors influence model performance. Using the publicly available and relatively clean CIFAKE dataset, we identify common quality issues and quantify their impact on training. Building on these findings, we develop a pipeline that integrates two community-developed tools, CleanVision and Fastdup. We analyze their underlying mechanisms and introduce several enhancements, including automatic threshold selection to detect problematic images without manual tuning.
Experimental results demonstrate that not all quality issues exert the same level of impact. While convolutional neural networks show resilience to certain distortions, they are particularly vulnerable to degradations that obscure critical visual features, such as blurring and severe downscaling. To assess the performance of existing tools and the effectiveness of our proposed enhancements, we formulate the detection of low-quality images as a binary classification task and use the F1 score as the evaluation metric. Our automatic thresholding method improves the F1 score from 0.6794 to 0.9468 under single perturbations and from 0.7447 to 0.8557 under dual perturbations. For near-duplicate detection, our deduplication strategy increases the F1 score from 0.4576 to 0.7928. These results underscore the effectiveness of our workflow and provide a foundation for advancing data quality assessment in image-based machine learning. 

---
# CLQ: Cross-Layer Guided Orthogonal-based Quantization for Diffusion Transformers 

**Authors**: Kai Liu, Shaoqiu Zhang, Linghe Kong, Yulun Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24416)  

**Abstract**: Visual generation quality has been greatly promoted with the rapid advances in diffusion transformers (DiTs), which is attributed to the scaling of model size and complexity. However, these attributions also hinder the practical deployment of DiTs on edge devices, limiting their development and application. Serve as an efficient model compression technique, model post-training quantization (PTQ) can reduce the memory consumption and speed up the inference, with inevitable performance degradation. To alleviate the degradation, we propose CLQ, a cross-layer guided orthogonal-based quantization method for DiTs. To be specific, CLQ consists of three key designs. First, we observe that the calibration data used by most of the PTQ methods can not honestly represent the distribution of the activations. Therefore, we propose cross-block calibration (CBC) to obtain accurate calibration data, with which the quantization can be better guided. Second, we propose orthogonal-based smoothing (OBS), which quantifies the outlier score of each channel and leverages block Hadamard matrix to smooth the outliers with negligible overhead. Third, we propose cross-layer parameter searching (CLPS) to search. We evaluate CLQ with both image generation and video generation models and successfully compress the model into W4A4 with negligible degradation in visual quality and metrics. CLQ achieves 3.98x memory saving and 3.95x speedup. Our code is available at \hyperlink{this https URL}{this https URL}. 

---
# ScatterAD: Temporal-Topological Scattering Mechanism for Time Series Anomaly Detection 

**Authors**: Tao Yin, Xiaohong Zhang, Shaochen Fu, Zhibin Zhang, Li Huang, Yiyuan Yang, Kaixiang Yang, Meng Yan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24414)  

**Abstract**: One main challenge in time series anomaly detection for industrial IoT lies in the complex spatio-temporal couplings within multivariate data. However, traditional anomaly detection methods focus on modeling spatial or temporal dependencies independently, resulting in suboptimal representation learning and limited sensitivity to anomalous dispersion in high-dimensional spaces. In this work, we conduct an empirical analysis showing that both normal and anomalous samples tend to scatter in high-dimensional space, especially anomalous samples are markedly more dispersed. We formalize this dispersion phenomenon as scattering, quantified by the mean pairwise distance among sample representations, and leverage it as an inductive signal to enhance spatio-temporal anomaly detection. Technically, we propose ScatterAD to model representation scattering across temporal and topological dimensions. ScatterAD incorporates a topological encoder for capturing graph-structured scattering and a temporal encoder for constraining over-scattering through mean squared error minimization between neighboring time steps. We introduce a contrastive fusion mechanism to ensure the complementarity of the learned temporal and topological representations. Additionally, we theoretically show that maximizing the conditional mutual information between temporal and topological views improves cross-view consistency and enhances more discriminative representations. Extensive experiments on multiple public benchmarks show that ScatterAD achieves state-of-the-art performance on multivariate time series anomaly detection. Code is available at this repository: this https URL. 

---
# Hybrid Layer-Wise ANN-SNN With Surrogate Spike Encoding-Decoding Structure 

**Authors**: Nhan T. Luu, Duong T. Luu, Pham Ngoc Nam, Truong Cong Thang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24411)  

**Abstract**: Spiking Neural Networks (SNNs) have gained significant traction in both computational neuroscience and artificial intelligence for their potential in energy-efficient computing. In contrast, artificial neural networks (ANNs) excel at gradient-based optimization and high accuracy. This contrast has consequently led to a growing subfield of hybrid ANN-SNN research. However, existing hybrid approaches often rely on either a strict separation between ANN and SNN components or employ SNN-only encoders followed by ANN classifiers due to the constraints of non-differentiability of spike encoding functions, causing prior hybrid architectures to lack deep layer-wise cooperation during backpropagation. To address this gap, we propose a novel hybrid ANN-SNN framework that integrates layer-wise encode-decode SNN blocks within conventional ANN pipelines. Central to our method is the use of surrogate gradients for a bit-plane-based spike encoding function, enabling end-to-end differentiable training across ANN and SNN layers. This design achieves competitive accuracy with state-of-the-art pure ANN and SNN models while retaining the potential efficiency and temporal representation benefits of spiking computation. To the best of our knowledge, this is the first implementation of a surrogate gradient for bit plane coding specifically and spike encoder interface in general to be utilized in the context of hybrid ANN-SNN, successfully leading to a new class of hybrid models that pave new directions for future research. 

---
# Multilingual Text-to-SQL: Benchmarking the Limits of Language Models with Collaborative Language Agents 

**Authors**: Khanh Trinh Pham, Thu Huong Nguyen, Jun Jo, Quoc Viet Hung Nguyen, Thanh Tam Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24405)  

**Abstract**: Text-to-SQL enables natural access to databases, yet most benchmarks are English-only, limiting multilingual progress. We introduce MultiSpider 2.0, extending Spider 2.0 to eight languages (English, German, French, Spanish, Portuguese, Japanese, Chinese, Vietnamese). It preserves Spider 2.0's structural difficulty while adding linguistic and dialectal variability, demanding deeper reasoning for complex SQL. On this benchmark, state-of-the-art LLMs (such as DeepSeek-R1 and OpenAI o1) reach only 4\% execution accuracy when relying on intrinsic reasoning, versus 60\% on MultiSpider 1.0. Therefore, we provide a collaboration-driven language agents baseline that iteratively refines queries, improving accuracy to 15\%. These results reveal a substantial multilingual gap and motivate methods that are robust across languages and ready for real-world enterprise deployment. Our benchmark is available at this https URL. 

---
# The 2025 OpenAI Preparedness Framework does not guarantee any AI risk mitigation practices: a proof-of-concept for affordance analyses of AI safety policies 

**Authors**: Sam Coggins, Alex Saeri, Katherine A. Daniell, Lorenn P. Ruster, Jessie Liu, Jenny L. Davis  

**Link**: [PDF](https://arxiv.org/pdf/2509.24394)  

**Abstract**: Prominent AI companies are producing 'safety frameworks' as a type of voluntary self-governance. These statements purport to establish risk thresholds and safety procedures for the development and deployment of highly capable AI. Understanding which AI risks are covered and what actions are allowed, refused, demanded, encouraged, or discouraged by these statements is vital for assessing how these frameworks actually govern AI development and deployment. We draw on affordance theory to analyse the OpenAI 'Preparedness Framework Version 2' (April 2025) using the Mechanisms & Conditions model of affordances and the MIT AI Risk Repository. We find that this safety policy requests evaluation of a small minority of AI risks, encourages deployment of systems with 'Medium' capabilities for what OpenAI itself defines as 'severe harm' (potential for >1000 deaths or >$100B in damages), and allows OpenAI's CEO to deploy even more dangerous capabilities. These findings suggest that effective mitigation of AI risks requires more robust governance interventions beyond current industry self-regulation. Our affordance analysis provides a replicable method for evaluating what safety frameworks actually permit versus what they claim. 

---
# LLaDA-MoE: A Sparse MoE Diffusion Language Model 

**Authors**: Fengqi Zhu, Zebin You, Yipeng Xing, Zenan Huang, Lin Liu, Yihong Zhuang, Guoshan Lu, Kangyu Wang, Xudong Wang, Lanning Wei, Hongrui Guo, Jiaqi Hu, Wentao Ye, Tieyuan Chen, Chenchen Li, Chengfu Tang, Haibo Feng, Jun Hu, Jun Zhou, Xiaolu Zhang, Zhenzhong Lan, Junbo Zhao, Da Zheng, Chongxuan Li, Jianguo Li, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24389)  

**Abstract**: We introduce LLaDA-MoE, a large language diffusion model with the Mixture-of-Experts (MoE) architecture, trained from scratch on approximately 20T tokens. LLaDA-MoE achieves competitive performance with significantly reduced computational overhead by maintaining a 7B-parameter capacity while activating only 1.4B parameters during inference. Our empirical evaluation reveals that LLaDA-MoE achieves state-of-the-art performance among diffusion language models with larger parameters, surpassing previous diffusion language models LLaDA, LLaDA 1.5, and Dream across multiple benchmarks. The instruct-tuned model LLaDA-MoE-7B-A1B-Instruct demonstrates capabilities comparable to Qwen2.5-3B-Instruct in knowledge understanding, code generation, mathematical reasoning, agent and alignment tasks, despite using fewer active parameters. Our results show that integrating a sparse MoE architecture into the training objective of masked diffusion language models still brings out MoE's strengths under efficient inference with few active parameters, and opens ample room for further exploration of diffusion language models. LLaDA-MoE models are available at Huggingface. 

---
# Vid-LLM: A Compact Video-based 3D Multimodal LLM with Reconstruction-Reasoning Synergy 

**Authors**: Haijier Chen, Bo Xu, Shoujian Zhang, Haoze Liu, Jiaxuan Lin, Jingrong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24385)  

**Abstract**: Recent developments in Multimodal Large Language Models (MLLMs) have significantly improved Vision-Language (VL) reasoning in 2D domains. However, extending these capabilities to 3D scene understanding remains a major challenge. Existing 3D Multimodal Large Language Models (3D-MLLMs) often depend on 3D data inputs, which limits scalability and generalization. To address this limitation, we propose Vid-LLM, a video-based 3D-MLLM that directly processes video inputs without requiring external 3D data, making it practical for real-world deployment. In our method, the geometric prior are directly used to improve the performance of the sceen perception. To integrate the geometric cues into the MLLM compactly, we design a Cross-Task Adapter (CTA) module to align the 3D geometric priors with the vision-language representations. To ensure geometric consistency and integrity, we introduce a Metric Depth Model that recovers real-scale geometry from the reconstruction outputs. Finally, the model is fine-tuned with a two-stage distillation optimization strategy, realizing fast convergence and stabilizes training. Extensive experiments across diverse benchmarks verified the effectiveness of our method on 3D Question Answering, 3D Dense Captioning and 3D Visual Grounding tasks, demonstrating the superior multi-task capabilities. 

---
# HarmMetric Eval: Benchmarking Metrics and Judges for LLM Harmfulness Assessment 

**Authors**: Langqi Yang, Tianhang Zheng, Kedong Xiu, Yixuan Chen, Di Wang, Puning Zhao, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.24384)  

**Abstract**: The alignment of large language models (LLMs) with human values is critical for their safe deployment, yet jailbreak attacks can subvert this alignment to elicit harmful outputs from LLMs. In recent years, a proliferation of jailbreak attacks has emerged, accompanied by diverse metrics and judges to assess the harmfulness of the LLM outputs. However, the absence of a systematic benchmark to assess the quality and effectiveness of these metrics and judges undermines the credibility of the reported jailbreak effectiveness and other risks. To address this gap, we introduce HarmMetric Eval, a comprehensive benchmark designed to support both overall and fine-grained evaluation of harmfulness metrics and judges. Our benchmark includes a high-quality dataset of representative harmful prompts paired with diverse harmful and non-harmful model responses, alongside a flexible scoring mechanism compatible with various metrics and judges. With HarmMetric Eval, our extensive experiments uncover a surprising result: two conventional metrics--METEOR and ROUGE-1--outperform LLM-based judges in evaluating the harmfulness of model responses, challenging prevailing beliefs about LLMs' superiority in this domain. Our dataset is publicly available at this https URL, and the code is available at this https URL. 

---
# REALIGN: Regularized Procedure Alignment with Matching Video Embeddings via Partial Gromov-Wasserstein Optimal Transport 

**Authors**: Soumyadeep Chandra, Kaushik Roy  

**Link**: [PDF](https://arxiv.org/pdf/2509.24382)  

**Abstract**: Learning from procedural videos remains a core challenge in self-supervised representation learning, as real-world instructional data often contains background segments, repeated actions, and steps presented out of order. Such variability violates the strong monotonicity assumptions underlying many alignment methods. Prior state-of-the-art approaches, such as OPEL, leverage Kantorovich Optimal Transport (KOT) to build frame-to-frame correspondences, but rely solely on feature similarity and fail to capture the higher-order temporal structure of a task. In this paper, we introduce REALIGN, a self-supervised framework for procedure learning based on Regularized Fused Partial Gromov-Wasserstein Optimal Transport (R-FPGWOT). In contrast to KOT, our formulation jointly models visual correspondences and temporal relations under a partial alignment scheme, enabling robust handling of irrelevant frames, repeated actions, and non-monotonic step orders common in instructional videos. To stabilize training, we integrate FPGWOT distances with inter-sequence contrastive learning, avoiding the need for multiple regularizers and preventing collapse to degenerate solutions. Across egocentric (EgoProceL) and third-person (ProceL, CrossTask) benchmarks, REALIGN achieves up to 18.9% average F1-score improvements and over 30% temporal IoU gains, while producing more interpretable transport maps that preserve key-step orderings and filter out noise. 

---
# Evolution Strategies at Scale: LLM Fine-Tuning Beyond Reinforcement Learning 

**Authors**: Xin Qiu, Yulu Gan, Conor F. Hayes, Qiyao Liang, Elliot Meyerson, Babak Hodjat, Risto Miikkulainen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24372)  

**Abstract**: Fine-tuning pre-trained large language models (LLMs) for down-stream tasks is a critical step in the AI deployment pipeline. Reinforcement learning (RL) is arguably the most prominent fine-tuning method, contributing to the birth of many state-of-the-art LLMs. In contrast, evolution strategies (ES), which once showed comparable performance to RL on models with a few million parameters, was neglected due to the pessimistic perception of its scalability to larger models. In this work, we report the first successful attempt to scale up ES for fine-tuning the full parameters of LLMs, showing the surprising fact that ES can search efficiently over billions of parameters and outperform existing RL fine-tuning methods in multiple respects, including sample efficiency, tolerance to long-horizon rewards, robustness to different base LLMs, less tendency to reward hacking, and more stable performance across runs. It therefore serves as a basis to unlock a new direction in LLM fine-tuning beyond what current RL techniques provide. The source codes are provided at: this https URL. 

---
# From Satellite to Street: A Hybrid Framework Integrating Stable Diffusion and PanoGAN for Consistent Cross-View Synthesis 

**Authors**: Khawlah Bajbaa, Abbas Anwar, Muhammad Saqib, Hafeez Anwar, Nabin Sharma, Muhammad Usman  

**Link**: [PDF](https://arxiv.org/pdf/2509.24369)  

**Abstract**: Street view imagery has become an essential source for geospatial data collection and urban analytics, enabling the extraction of valuable insights that support informed decision-making. However, synthesizing street-view images from corresponding satellite imagery presents significant challenges due to substantial differences in appearance and viewing perspective between these two domains. This paper presents a hybrid framework that integrates diffusion-based models and conditional generative adversarial networks to generate geographically consistent street-view images from satellite imagery. Our approach uses a multi-stage training strategy that incorporates Stable Diffusion as the core component within a dual-branch architecture. To enhance the framework's capabilities, we integrate a conditional Generative Adversarial Network (GAN) that enables the generation of geographically consistent panoramic street views. Furthermore, we implement a fusion strategy that leverages the strengths of both models to create robust representations, thereby improving the geometric consistency and visual quality of the generated street-view images. The proposed framework is evaluated on the challenging Cross-View USA (CVUSA) dataset, a standard benchmark for cross-view image synthesis. Experimental results demonstrate that our hybrid approach outperforms diffusion-only methods across multiple evaluation metrics and achieves competitive performance compared to state-of-the-art GAN-based methods. The framework successfully generates realistic and geometrically consistent street-view images while preserving fine-grained local details, including street markings, secondary roads, and atmospheric elements such as clouds. 

---
# Watermarking Diffusion Language Models 

**Authors**: Thibaud Gloaguen, Robin Staab, Nikola Jovanović, Martin Vechev  

**Link**: [PDF](https://arxiv.org/pdf/2509.24368)  

**Abstract**: We introduce the first watermark tailored for diffusion language models (DLMs), an emergent LLM paradigm able to generate tokens in arbitrary order, in contrast to standard autoregressive language models (ARLMs) which generate tokens sequentially. While there has been much work in ARLM watermarking, a key challenge when attempting to apply these schemes directly to the DLM setting is that they rely on previously generated tokens, which are not always available with DLM generation. In this work we address this challenge by: (i) applying the watermark in expectation over the context even when some context tokens are yet to be determined, and (ii) promoting tokens which increase the watermark strength when used as context for other tokens. This is accomplished while keeping the watermark detector unchanged. Our experimental evaluation demonstrates that the DLM watermark leads to a >99% true positive rate with minimal quality impact and achieves similar robustness to existing ARLM watermarks, enabling for the first time reliable DLM watermarking. 

---
# Uni-X: Mitigating Modality Conflict with a Two-End-Separated Architecture for Unified Multimodal Models 

**Authors**: Jitai Hao, Hao Liu, Xinyan Xiao, Qiang Huang, Jun Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24365)  

**Abstract**: Unified Multimodal Models (UMMs) built on shared autoregressive (AR) transformers are attractive for their architectural simplicity. However, we identify a critical limitation: when trained on multimodal inputs, modality-shared transformers suffer from severe gradient conflicts between vision and text, particularly in shallow and deep layers. We trace this issue to the fundamentally different low-level statistical properties of images and text, while noting that conflicts diminish in middle layers where representations become more abstract and semantically aligned. To overcome this challenge, we propose Uni-X, a two-end-separated, middle-shared architecture. Uni-X dedicates its initial and final layers to modality-specific processing, while maintaining shared parameters in the middle layers for high-level semantic fusion. This X-shaped design not only eliminates gradient conflicts at both ends but also further alleviates residual conflicts in the shared layers. Extensive experiments validate the effectiveness of Uni-X. Under identical training conditions, Uni-X achieves superior training efficiency compared to strong baselines. When scaled to 3B parameters with larger training data, Uni-X matches or surpasses 7B AR-based UMMs, achieving a GenEval score of 82 for image generation alongside strong performance in text and vision understanding tasks. These results establish Uni-X as a parameter-efficient and scalable foundation for future unified multimodal modeling. Our code is available at this https URL 

---
# UI-UG: A Unified MLLM for UI Understanding and Generation 

**Authors**: Hao Yang, Weijie Qiu, Ru Zhang, Zhou Fang, Ruichao Mao, Xiaoyu Lin, Maji Huang, Zhaosong Huang, Teng Guo, Shuoyang Liu, Hai Rao  

**Link**: [PDF](https://arxiv.org/pdf/2509.24361)  

**Abstract**: Although Multimodal Large Language Models (MLLMs) have been widely applied across domains, they are still facing challenges in domain-specific tasks, such as User Interface (UI) understanding accuracy and UI generation quality. In this paper, we introduce UI-UG (a unified MLLM for UI Understanding and Generation), integrating both capabilities. For understanding tasks, we employ Supervised Fine-tuning (SFT) combined with Group Relative Policy Optimization (GRPO) to enhance fine-grained understanding on the modern complex UI data. For generation tasks, we further use Direct Preference Optimization (DPO) to make our model generate human-preferred UIs. In addition, we propose an industrially effective workflow, including the design of an LLM-friendly domain-specific language (DSL), training strategies, rendering processes, and evaluation metrics. In experiments, our model achieves state-of-the-art (SOTA) performance on understanding tasks, outperforming both larger general-purpose MLLMs and similarly-sized UI-specialized models. Our model is also on par with these larger MLLMs in UI generation performance at a fraction of the computational cost. We also demonstrate that integrating understanding and generation tasks can improve accuracy and quality for both tasks. 

---
# An Enhanced Pyramid Feature Network Based on Long-Range Dependencies for Multi-Organ Medical Image Segmentation 

**Authors**: Dayu Tan, Cheng Kong, Yansen Su, Hai Chen, Dongliang Yang, Junfeng Xia, Chunhou Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.24358)  

**Abstract**: In the field of multi-organ medical image segmentation, recent methods frequently employ Transformers to capture long-range dependencies from image features. However, these methods overlook the high computational cost of Transformers and their deficiencies in extracting local detailed information. To address high computational costs and inadequate local detail information, we reassess the design of feature extraction modules and propose a new deep-learning network called LamFormer for fine-grained segmentation tasks across multiple organs. LamFormer is a novel U-shaped network that employs Linear Attention Mamba (LAM) in an enhanced pyramid encoder to capture multi-scale long-range dependencies. We construct the Parallel Hierarchical Feature Aggregation (PHFA) module to aggregate features from different layers of the encoder, narrowing the semantic gap among features while filtering information. Finally, we design the Reduced Transformer (RT), which utilizes a distinct computational approach to globally model up-sampled features. RRT enhances the extraction of detailed local information and improves the network's capability to capture long-range dependencies. LamFormer outperforms existing segmentation methods on seven complex and diverse datasets, demonstrating exceptional performance. Moreover, the proposed network achieves a balance between model performance and model complexity. 

---
# Beyond Repetition: Text Simplification and Curriculum Learning for Data-Constrained Pretraining 

**Authors**: Matthew Theodore Roque, Dan John Velasco  

**Link**: [PDF](https://arxiv.org/pdf/2509.24356)  

**Abstract**: Most studies on language model pretraining focus on large datasets, leaving open questions about optimization in data-constrained settings. In such settings, the effects of training data order and of including alternative versions of the same text remain underexplored. We address this by studying curriculum learning in pretraining, focusing on text-complexity ordering and data augmentation via simplification. We ask: (1) Does simplifying texts enhance representation quality more than reusing the original data? and (2) Does ordering data by text complexity yield better representations? To answer, we build on a pair of parallel corpora where human-written paragraphs are aligned with LLM-simplified variants, and test four data schedules: repeated exposure, low-to-high complexity, high-to-low, and interleaved. We analyze models' representation quality from a sample efficiency perspective via fine-tuning, as well as its zero-shot performance on linguistic knowledge, entity tracking, world knowledge, and commonsense reasoning. Our findings show that adding simplified data improves fine-tuning and zero-shot performance over a repeated-exposure baseline: smaller models benefit from low-to-high complexity, while larger models perform better with interleaved ordering. 

---
# Dynamic Orchestration of Multi-Agent System for Real-World Multi-Image Agricultural VQA 

**Authors**: Yan Ke, Xin Yu, Heming Du, Scott Chapman, Helen Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24350)  

**Abstract**: Agricultural visual question answering is essential for providing farmers and researchers with accurate and timely knowledge. However, many existing approaches are predominantly developed for evidence-constrained settings such as text-only queries or single-image cases. This design prevents them from coping with real-world agricultural scenarios that often require multi-image inputs with complementary views across spatial scales, and growth stages. Moreover, limited access to up-to-date external agricultural context makes these systems struggle to adapt when evidence is incomplete. In addition, rigid pipelines often lack systematic quality control. To address this gap, we propose a self-reflective and self-improving multi-agent framework that integrates four roles, the Retriever, the Reflector, the Answerer, and the Improver. They collaborate to enable context enrichment, reflective reasoning, answer drafting, and iterative improvement.
A Retriever formulates queries and gathers external information, while a Reflector assesses adequacy and triggers sequential reformulation and renewed retrieval. Two Answerers draft candidate responses in parallel to reduce bias. The Improver refines them through iterative checks while ensuring that information from multiple images is effectively aligned and utilized. Experiments on the AgMMU benchmark show that our framework achieves competitive performance on multi-image agricultural QA. 

---
# Towards Generalizable PDE Dynamics Forecasting via Physics-Guided Invariant Learning 

**Authors**: Siyang Li, Yize Chen, Yan Guo, Ming Huang, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2509.24332)  

**Abstract**: Advanced deep learning-based approaches have been actively applied to forecast the spatiotemporal physical dynamics governed by partial differential equations (PDEs), which acts as a critical procedure in tackling many science and engineering problems. As real-world physical environments like PDE system parameters are always capricious, how to generalize across unseen out-of-distribution (OOD) forecasting scenarios using limited training data is of great importance. To bridge this barrier, existing methods focus on discovering domain-generalizable representations across various PDE dynamics trajectories. However, their zero-shot OOD generalization capability remains deficient, since extra test-time samples for domain-specific adaptation are still required. This is because the fundamental physical invariance in PDE dynamical systems are yet to be investigated or integrated. To this end, we first explicitly define a two-fold PDE invariance principle, which points out that ingredient operators and their composition relationships remain invariant across different domains and PDE system evolution. Next, to capture this two-fold PDE invariance, we propose a physics-guided invariant learning method termed iMOOE, featuring an Invariance-aligned Mixture Of Operator Expert architecture and a frequency-enriched invariant learning objective. Extensive experiments across simulated benchmarks and real-world applications validate iMOOE's superior in-distribution performance and zero-shot generalization capabilities on diverse OOD forecasting scenarios. 

---
# TraitSpaces: Towards Interpretable Visual Creativity for Human-AI Co-Creation 

**Authors**: Prerna Luthra  

**Link**: [PDF](https://arxiv.org/pdf/2509.24326)  

**Abstract**: We introduce a psychologically grounded and artist-informed framework for modeling visual creativity across four domains: Inner, Outer, Imaginative, and Moral Worlds. Drawing on interviews with practicing artists and theories from psychology, we define 12 traits that capture affective, symbolic, cultural, and ethical dimensions of this http URL 20k artworks from the SemArt dataset, we annotate images with GPT 4.1 using detailed, theory-aligned prompts, and evaluate the learnability of these traits from CLIP image embeddings. Traits such as Environmental Dialogicity and Redemptive Arc are predicted with high reliability ($R^2 \approx 0.64 - 0.68$), while others like Memory Imprint remain challenging, highlighting the limits of purely visual encoding. Beyond technical metrics, we visualize a "creativity trait-space" and illustrate how it can support interpretable, trait-aware co-creation - e.g., sliding along a Redemptive Arc axis to explore works of adversity and renewal. By linking cultural-aesthetic insights with computational modeling, our work aims not to reduce creativity to numbers, but to offer shared language and interpretable tools for artists, researchers, and AI systems to collaborate meaningfully. 

---
# Dual Mechanisms of Value Expression: Intrinsic vs. Prompted Values in LLMs 

**Authors**: Jongwook Han, Jongwon Lim, Injin Kong, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2509.24319)  

**Abstract**: Large language models (LLMs) can express different values in two distinct ways: (1) intrinsic expression, reflecting the model's inherent values learned during training, and (2) prompted expression, elicited by explicit prompts. Given their widespread use in value alignment and persona steering, it is paramount to clearly understand their underlying mechanisms, particularly whether they mostly overlap (as one might expect) or rely on substantially different mechanisms, but this remains largely understudied. We analyze this at the mechanistic level using two approaches: (1) value vectors, feature directions representing value mechanisms extracted from the residual stream, and (2) value neurons, MLP neurons that contribute to value expressions. We demonstrate that intrinsic and prompted value mechanisms partly share common components that are crucial for inducing value expression, but also possess unique elements that manifest in different ways. As a result, these mechanisms lead to different degrees of value steerability (prompted > intrinsic) and response diversity (intrinsic > prompted). In particular, components unique to the intrinsic mechanism seem to promote lexical diversity in responses, whereas those specific to the prompted mechanism primarily strengthen instruction following, taking effect even in distant tasks like jailbreaking. 

---
# A study of Universal ODE approaches to predicting soil organic carbon 

**Authors**: Satyanarayana Raju G.V.V, Prathamesh Dinesh Joshi, Raj Abhijit Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2509.24306)  

**Abstract**: Soil Organic Carbon (SOC) is a foundation of soil health and global climate resilience, yet its prediction remains difficult because of intricate physical, chemical, and biological processes. In this study, we explore a Scientific Machine Learning (SciML) framework built on Universal Differential Equations (UDEs) to forecast SOC dynamics across soil depth and time. UDEs blend mechanistic physics, such as advection diffusion transport, with neural networks that learn nonlinear microbial production and respiration. Using synthetic datasets, we systematically evaluated six experimental cases, progressing from clean, noise free benchmarks to stress tests with high (35%) multiplicative, spatially correlated noise. Our results highlight both the potential and limitations of the approach. In noise free and moderate noise settings, the UDE accurately reconstructed SOC dynamics. In clean terminal profile at 50 years (Case 4) achieved near perfect fidelity, with MSE = 1.6e-5, and R2 = 0.9999. Case 5, with 7% noise, remained robust (MSE = 3.4e-6, R2 = 0.99998), capturing depth wise SOC trends while tolerating realistic measurement uncertainty. In contrast, Case 3 (35% noise at t = 0) showed clear evidence of overfitting: the model reproduced noisy inputs with high accuracy but lost generalization against the clean truth (R2 = 0.94). Case 6 (35% noise at t = 50) collapsed toward overly smooth mean profiles, failing to capture depth wise variability and yielding negative R2, underscoring the limits of standard training under severe uncertainty. These findings suggest that UDEs are well suited for scalable, noise tolerant SOC forecasting, though advancing toward field deployment will require noise aware loss functions, probabilistic modelling, and tighter integration of microbial dynamics. 

---
# Bridging the behavior-neural gap: A multimodal AI reveals the brain's geometry of emotion more accurately than human self-reports 

**Authors**: Changde Du, Yizhuo Lu, Zhongyu Huang, Yi Sun, Zisen Zhou, Shaozheng Qin, Huiguang He  

**Link**: [PDF](https://arxiv.org/pdf/2509.24298)  

**Abstract**: The ability to represent emotion plays a significant role in human cognition and social interaction, yet the high-dimensional geometry of this affective space and its neural underpinnings remain debated. A key challenge, the `behavior-neural gap,' is the limited ability of human self-reports to predict brain activity. Here we test the hypothesis that this gap arises from the constraints of traditional rating scales and that large-scale similarity judgments can more faithfully capture the brain's affective geometry. Using AI models as `cognitive agents,' we collected millions of triplet odd-one-out judgments from a multimodal large language model (MLLM) and a language-only model (LLM) in response to 2,180 emotionally evocative videos. We found that the emergent 30-dimensional embeddings from these models are highly interpretable and organize emotion primarily along categorical lines, yet in a blended fashion that incorporates dimensional properties. Most remarkably, the MLLM's representation predicted neural activity in human emotion-processing networks with the highest accuracy, outperforming not only the LLM but also, counterintuitively, representations derived directly from human behavioral ratings. This result supports our primary hypothesis and suggests that sensory grounding--learning from rich visual data--is critical for developing a truly neurally-aligned conceptual framework for emotion. Our findings provide compelling evidence that MLLMs can autonomously develop rich, neurally-aligned affective representations, offering a powerful paradigm to bridge the gap between subjective experience and its neural substrates. Project page: this https URL. 

---
# Q-Mirror: Unlocking the Multi-Modal Potential of Scientific Text-Only QA Pairs 

**Authors**: Junying Wang, Zicheng Zhang, Ye Shen, Yalun Wu, Yingji Liang, Yijin Guo, Farong Wen, Wenzhe Li, Xuezhi Zhao, Qi Jia, Guangtao Zhai  

**Link**: [PDF](https://arxiv.org/pdf/2509.24297)  

**Abstract**: High-quality, multi-modal benchmarks are crucial for advancing scientific reasoning in large models yet their manual creation is costly and unscalable. To address this bottleneck, we explore the potential for transforming Text-Only QA Pairs (TQAs) into high-quality Multi-Modal QA Pairs (MMQAs), which include three parts: 1) Task Definition \& Evaluation Rubric: We develop a TQA-to-MMQA framework and establish a comprehensive, multi-dimensional MMQA quality rubric that provides principles for the transformation. 2) Benchmark Construction: Then we construct two extensive benchmarks to rigorously evaluate state-of-the-art generation \& understanding models on the distinct tasks of MMQA generation \& MMQA quality evaluation. 3) Preliminary Solution: We develop an agentic system (Q-Mirror), which operationalizes our framework by integrating MMQA generation and evaluation into a closed loop for iterative refinement. Our experiments show that while state-of-the-art models can generate MMQAs, their outputs still leave substantial gaps, underscoring the need for reliable evaluation. We further demonstrate that top-tier understanding models align closely with human judgment in MMQA quality assessment. Leveraging both insights, the Q-Mirror agent raises average scores from 78.90 to 85.22 and pass rates from 72\% to 95\%, offering a practical path to large-scale scientific benchmarks. 

---
# DiffuGuard: How Intrinsic Safety is Lost and Found in Diffusion Large Language Models 

**Authors**: Zherui Li, Zheng Nie, Zhenhong Zhou, Yufei Guo, Yue Liu, Yitong Zhang, Yu Cheng, Qingsong Wen, Kun Wang, Jiaheng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24296)  

**Abstract**: The rapid advancement of Diffusion Large Language Models (dLLMs) introduces unprecedented vulnerabilities that are fundamentally distinct from Autoregressive LLMs, stemming from their iterative and parallel generation mechanisms. In this paper, we conduct an in-depth analysis of dLLM vulnerabilities to jailbreak attacks across two distinct dimensions: intra-step and inter-step dynamics. Experimental results reveal a harmful bias inherent in the standard greedy remasking strategy and identify a critical phenomenon we term Denoising-path Dependence, where the safety of early-stage tokens decisively influences the final output. These findings also indicate that while current decoding strategies constitute a significant vulnerability, dLLMs possess a substantial intrinsic safety potential. To unlock this potential, we propose DiffuGuard, a training-free defense framework that addresses vulnerabilities through a dual-stage approach: Stochastic Annealing Remasking dynamically introduces controlled randomness to mitigate greedy selection bias, while Block-level Audit and Repair exploits internal model representations for autonomous risk detection and guided correction. Comprehensive experiments on four dLLMs demonstrate DiffuGuard's exceptional effectiveness, reducing Attack Success Rate against six diverse jailbreak methods from 47.9% to 14.7% while preserving model utility and efficiency. Our code is available at: this https URL. 

---
# Let LLMs Speak Embedding Languages: Generative Text Embeddings via Iterative Contrastive Refinement 

**Authors**: Yu-Che Tsai, Kuan-Yu Chen, Yuan-Chi Li, Yuan-Hao Chen, Ching-Yu Tsai, Shou-De Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.24291)  

**Abstract**: Existing large language model (LLM)-based embeddings typically adopt an encoder-only paradigm, treating LLMs as static feature extractors and overlooking their core generative strengths. We introduce GIRCSE (Generative Iterative Refinement for Contrastive Sentence Embeddings), a novel framework that leverages autoregressive generation to iteratively refine semantic representations. By producing sequences of soft tokens optimized under contrastive objective, GIRCSE captures latent concepts and implicit semantics that encoder-only methods often miss. To guide this process, we propose an Iterative Contrastive Refinement (ICR) objective that encourages each refinement step to yield better representations. Extensive experiments show that GIRCSE outperforms strong LLM-based embedding baselines on the MTEB benchmark and instruction-following tasks. Moreover, GIRCSE exhibits an emergent test-time scaling property: generating more tokens at inference steadily improves embedding quality. Our results establish generative iterative refinement as a new paradigm for representation learning. 

---
# SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents 

**Authors**: Gyuhyeon Seo, Jungwoo Yang, Junseong Pyo, Nalim Kim, Jonggeun Lee, Yohan Jo  

**Link**: [PDF](https://arxiv.org/pdf/2509.24282)  

**Abstract**: Large Language Model (LLM) agents excel at multi-step, tool-augmented tasks. However, smart homes introduce distinct challenges, requiring agents to handle latent user intents, temporal dependencies, device constraints, scheduling, and more. The main bottlenecks for developing smart home agents with such capabilities include the lack of a realistic simulation environment where agents can interact with devices and observe the results, as well as a challenging benchmark to evaluate them. To address this, we introduce $\textbf{SimuHome}$, a time-accelerated home environment that simulates smart devices, supports API calls, and reflects changes in environmental variables. By building the simulator on the Matter protocol (the global industry standard for smart home communication), SimuHome provides a high-fidelity environment, and agents validated in SimuHome can be deployed on real Matter-compliant devices with minimal adaptation. We provide a challenging benchmark of 600 episodes across twelve user query types that require the aforementioned capabilities. Our evaluation of 11 agents under a unified ReAct framework reveals that while models perform well on simple tasks, they struggle with latent intent inference, state verification, and especially temporal scheduling. Even the top-performing model, GPT-4.1, reaches only 54% success rate. These findings highlight a critical need for methods that can reliably verify the current state via tools before acting and coordinate time-dependent actions. 

---
# Adversarial Reinforcement Learning Framework for ESP Cheater Simulation 

**Authors**: Inkyu Park, Jeong-Gwan Lee, Taehwan Kwon, Juheon Choi, Seungku Kim, Junsu Kim, Kimin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.24274)  

**Abstract**: Extra-Sensory Perception (ESP) cheats, which reveal hidden in-game information such as enemy locations, are difficult to detect because their effects are not directly observable in player behavior. The lack of observable evidence makes it difficult to collect reliably labeled data, which is essential for training effective anti-cheat systems. Furthermore, cheaters often adapt their behavior by limiting or disguising their cheat usage, which further complicates detection and detector development. To address these challenges, we propose a simulation framework for controlled modeling of ESP cheaters, non-cheaters, and trajectory-based detectors. We model cheaters and non-cheaters as reinforcement learning agents with different levels of observability, while detectors classify their behavioral trajectories. Next, we formulate the interaction between the cheater and the detector as an adversarial game, allowing both players to co-adapt over time. To reflect realistic cheater strategies, we introduce a structured cheater model that dynamically switches between cheating and non-cheating behaviors based on detection risk. Experiments demonstrate that our framework successfully simulates adaptive cheater behaviors that strategically balance reward optimization and detection evasion. This work provides a controllable and extensible platform for studying adaptive cheating behaviors and developing effective cheat detectors. 

---
# Cycle Diffusion Model for Counterfactual Image Generation 

**Authors**: Fangrui Huang, Alan Wang, Binxu Li, Bailey Trang, Ridvan Yesiloglu, Tianyu Hua, Wei Peng, Ehsan Adeli  

**Link**: [PDF](https://arxiv.org/pdf/2509.24267)  

**Abstract**: Deep generative models have demonstrated remarkable success in medical image synthesis. However, ensuring conditioning faithfulness and high-quality synthetic images for direct or counterfactual generation remains a challenge. In this work, we introduce a cycle training framework to fine-tune diffusion models for improved conditioning adherence and enhanced synthetic image realism. Our approach, Cycle Diffusion Model (CDM), enforces consistency between generated and original images by incorporating cycle constraints, enabling more reliable direct and counterfactual generation. Experiments on a combined 3D brain MRI dataset (from ABCD, HCP aging & young adults, ADNI, and PPMI) show that our method improves conditioning accuracy and enhances image quality as measured by FID and SSIM. The results suggest that the cycle strategy used in CDM can be an effective method for refining diffusion-based medical image generation, with applications in data augmentation, counterfactual, and disease progression modeling. 

---
# LAMP-PRo: Label-aware Attention for Multi-label Prediction of DNA- and RNA-binding Proteins using Protein Language Models 

**Authors**: Nimisha Ghosh, Dheeran Sankaran, Rahul Balakrishnan Adhi, Sharath S, Amrut Anand  

**Link**: [PDF](https://arxiv.org/pdf/2509.24262)  

**Abstract**: Identifying DNA- (DBPs) and RNA-binding proteins (RBPs) is crucial for the understanding of cell function, molecular interactions as well as regulatory functions. Owing to their high similarity, most of the existing approaches face challenges in differentiating between DBPs and RBPs leading to high cross-prediction errors. Moreover, identifying proteins which bind to both DNA and RNA (DRBPs) is also quite a challenging task. In this regard, we propose a novel framework viz. LAMP-PRo which is based on pre-trained protein language model (PLM), attention mechanisms and multi-label learning to mitigate these issues. First, pre-trained PLM such ESM-2 is used for embedding the protein sequences followed by convolutional neural network (CNN). Subsequently multi-head self-attention mechanism is applied for the contextual information while label-aware attention is used to compute class-specific representations by attending to the sequence in a way that is tailored to each label (DBP, RBP and non-NABP) in a multi-label setup. We have also included a novel cross-label attention mechanism to explicitly capture dependencies between DNA- and RNA-binding proteins, enabling more accurate prediction of DRBP. Finally, a linear layer followed by a sigmoid function are used for the final prediction. Extensive experiments are carried out to compare LAMP-PRo with the existing methods wherein the proposed model shows consistent competent performance. Furthermore, we also provide visualization to showcase model interpretability, highlighting which parts of the sequence are most relevant for a predicted label. The original datasets are available at this http URL\_MMC and the codes are available at this https URL. 

---
# Graph Foundation Models: Bridging Language Model Paradigms and Graph Optimization 

**Authors**: Yunhao Liang, Pujun Zhang, Yuan Qu, Shaochong Lin, Zuo-jun Max Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24256)  

**Abstract**: The pretrain-transfer paradigm, which underpins the success of large language models (LLMs), has demonstrated the immense power of creating foundation models that learn generalizable representations from vast datasets. However, extending this paradigm to Operations Research (OR) problems on graph structures remains challenging due to the fundamental conflict between the statistical flexibility of language and the strict combinatorial constraints of graphs. To bridge this gap, we introduce the Graph Foundation Model (GFM), the first framework capable of solving all distance-based optimization problems on graph structures. By introducing the LLM-like self-supervised pre-training paradigm on the paths generated from random walks in the graph, GFM is compelled to internalize the graph's complex topological and combinatorial rules, where the connectivity of the structure itself can be treated as the supervisory signal. Unlike existing neural methods that learn complex and task-specific solving policies, our approach leverages the pre-trained GFM as a foundational model of the graph's intrinsic structure, which in turn enables a simple generative heuristic to tackle a diverse range of optimization challenges effectively. Comprehensive experiments on networks ranging from 20 to 893 nodes demonstrate that GFM achieves competitive performance against specialized solvers across a variety of distinct optimization task classes, while maintaining significantly faster inference times. Our work establishes a new paradigm of adapting the pretrain-transfer framework to graph optimization, opening the door for applying foundation model innovations to OR. 

---
# Prompt and Parameter Co-Optimization for Large Language Models 

**Authors**: Xiaohe Bo, Rui Li, Zexu Sun, Quanyu Dai, Zeyu Zhang, Zihang Tian, Xu Chen, Zhenhua Dong  

**Link**: [PDF](https://arxiv.org/pdf/2509.24245)  

**Abstract**: Prompt optimization and fine-tuning are two major approaches to improve the performance of Large Language Models (LLMs). They enhance the capabilities of LLMs from complementary perspectives: the former through explicit natural language, and the latter through implicit parameter updates. However, prior work has typically studied them in isolation, leaving their synergistic potential largely underexplored. To bridge this gap, in this paper, we introduce MetaTuner, a novel framework that jointly integrates prompt optimization and fine-tuning for LLM training. Specifically, we introduce two neural networks to generate prompts and parameters, respectively, while allowing them to share a common bottom encoding layer to enable knowledge sharing. By the guidance of the final supervised signals, our framework is optimized to discover the optimal combinations between the prompts and parameters. Given that prompt learning involves discrete optimization while fine-tuning operates in a continuous parameter space, we design a supervised regularization loss to train our framework effectively. Extensive experiments across diverse benchmarks show that our method consistently outperforms the baselines. 

---
# SafeFlowMatcher: Safe and Fast Planning using Flow Matching with Control Barrier Functions 

**Authors**: Jeongyong Yang, Seunghwan Jang, Soojean Han  

**Link**: [PDF](https://arxiv.org/pdf/2509.24243)  

**Abstract**: Generative planners based on flow matching (FM) can produce high-quality paths in one or a few ODE steps, but their sampling dynamics offer no formal safety guarantees and can yield incomplete paths near constraints. We present SafeFlowMatcher, a planning framework that couples FM with control barrier functions (CBFs) to achieve both real-time efficiency and certified safety. SafeFlowMatcher uses a two-phase prediction-correction (PC) integrator: (i) a prediction phase integrates the learned FM once (or a few steps) to obtain a candidate path without intervention; (ii) a correction phase refines this path with a vanishing time-scaled vector field and a CBF-based quadratic program that minimally perturbs the vector field. We prove a barrier certificate for the resulting flow system, establishing forward invariance of a robust safe set and finite-time convergence to the safe set. By enforcing safety only on the executed path (rather than on all intermediate latent paths), SafeFlowMatcher avoids distributional drift and mitigates local trap problems. Across maze navigation and locomotion benchmarks, SafeFlowMatcher attains faster, smoother, and safer paths than diffusion- and FM-based baselines. Extensive ablations corroborate the contributions of the PC integrator and the barrier certificate. 

---
# ChessArena: A Chess Testbed for Evaluating Strategic Reasoning Capabilities of Large Language Models 

**Authors**: Jincheng Liu, Sijun He, Jingjing Wu, Xiangsen Wang, Yang Chen, Zhaoqi Kuang, Siqi Bao, Yuan Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.24239)  

**Abstract**: Recent large language models (LLMs) have shown strong reasoning capabilities. However, a critical question remains: do these models possess genuine reasoning skills particularly complex strategic reasoning or are they primarily excelling at sophisticated pattern recognition within their training data? To address this question, this paper presents a chess testbed, ChessArena, to evaluate the strategic reasoning capabilities of LLMs. Chess requires complex strategic reasoning capabilities including long-term planning, strict rule comprehension, and multi-turn conversation memorization. Specifically, ChessArena is a competitive framework where LLMs play against each other, under four different play modes. The testbed is equipped with a ranking algorithm and a leaderboard. The testbed can also evaluate fine-grained capabilities including basic understanding, move selection, and puzzle solving. Over 13 LLMs with different modes are evaluated in ChessArena, playing over 800 games. The results reveal significant shortcomings in current LLMs: no model can beat Maia-1100 (a chess engine at human amateur level), while some even failed to defeat a random player that selects moves arbitrarily. We also present a strong baseline to the testbed: our fine-tuned Qwen3-8B substantially improved performance, approaching much larger state-of-the-art reasoning models. 

---
# Uni-NTFM: A Unified Foundation Model for EEG Signal Representation Learning 

**Authors**: Zhisheng Chen, Yingwei Zhang, Qizhen Lan, Tianyu Liu, Huacan Wang, Yi Ding, Ziyu Jia, Ronghao Chen, Kun Wang, Xinliang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.24222)  

**Abstract**: Foundation models pretrained on various and unlabeled data have demonstrated significant success in natural language and vision, but their application to electroencephalography (EEG) remains challenged due to the signal's unique properties. Existing brain foundation models that inherit architectures designed for text or images lead to three limitations in pre-training: 1) conflating time-domain waveform patterns with frequency-domain rhythmic features in a single processing stream, 2) ignoring the critical spatial topology of electrodes with different standards, and 3) reliance on the inflexible, dense network to process functionally distinct EEG patterns. To address these challenges, we introduce the Unified Neural Topological Foundation Model (Uni-NTFM), which is designed based on neuroscience principles to produce universal and interpretable representations. Uni-NTFM integrates three core innovations: 1) a decoupled architecture parallelly encodes time, frequency, and raw signal representations before performing cross-domain feature integration; 2) a topological embedding mechanism to unify electrodes from different international standards and generate structured input sequences for brain regions; and 3) a Mixture-of-Experts neural Transformer that efficiently scales model capacity by routing signal patterns to specialized subnetworks. The largest model, Uni-NTFM$_{large}$, has a record-breaking 1.9B parameters and was pretrained on over 28,000 hours of diverse EEG data via a dual-domain masked reconstruction objective. Uni-NTFM significantly outperforms existing task-specific methods and foundation models across nine distinct downstream tasks under both linear probing and fine-tuning settings, demonstrating a superior ability to learn universal representations of brain activity. 

---
# ViReSkill: Vision-Grounded Replanning with Skill Memory for LLM-Based Planning in Lifelong Robot Learning 

**Authors**: Tomoyuki Kagaya, Subramanian Lakshmi, Anbang Ye, Thong Jing Yuan, Jayashree Karlekar, Sugiri Pranata, Natsuki Murakami, Akira Kinose, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2509.24219)  

**Abstract**: Robots trained via Reinforcement Learning (RL) or Imitation Learning (IL) often adapt slowly to new tasks, whereas recent Large Language Models (LLMs) and Vision-Language Models (VLMs) promise knowledge-rich planning from minimal data. Deploying LLMs/VLMs for motion planning, however, faces two key obstacles: (i) symbolic plans are rarely grounded in scene geometry and object physics, and (ii) model outputs can vary for identical prompts, undermining execution reliability. We propose ViReSkill, a framework that pairs vision-grounded replanning with a skill memory for accumulation and reuse. When a failure occurs, the replanner generates a new action sequence conditioned on the current scene, tailored to the observed state. On success, the executed plan is stored as a reusable skill and replayed in future encounters without additional calls to LLMs/VLMs. This feedback loop enables autonomous continual learning: each attempt immediately expands the skill set and stabilizes subsequent executions. We evaluate ViReSkill on simulators such as LIBERO and RLBench as well as on a physical robot. Across all settings, it consistently outperforms conventional baselines in task success rate, demonstrating robust sim-to-real generalization. 

---
# Conda: Column-Normalized Adam for Training Large Language Models Faster 

**Authors**: Junjie Wang, Pan Zhou, Yiming Dong, Huan Li, Jia Li, Xun Zhou, Qicheng Lao, Cong Fang, Zhouchen Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.24218)  

**Abstract**: Large language models (LLMs) have demonstrated impressive generalization and emergent capabilities, yet their pre-training remains computationally expensive and sensitive to optimization dynamics. While Adam-based optimizers offer fast convergence by adapting learning rates coordinate-wise, recent studies reveal that their updates often suffer from poor spectral conditioning and low-rank structures, hindering efficiency. Muon addresses this issue via global spectral normalization but lacks the per-coordinate adaptivity of Adam. In this work, we propose \textbf{Column-Normalized Adam (Conda)}, a novel optimizer that bridges the strengths of both approaches. Conda projects updates into an orthogonal subspace and applies column-wise second moment normalization based on the projected gradients, thereby achieving both improved spectral conditioning and maintaining coordinate-wise adaptivity. This design alleviates the spectral pathologies of Adam while preserving its fast convergence behavior. Extensive experiments on the LLaMA and GPT-2 series show that Conda consistently outperforms AdamW, Muon, and other baselines in pre-training. Remarkably, on the LLaMA series, \textbf{Conda achieves $2{\sim}2.5\times$ the convergence speed of AdamW, measured in both training steps and training time.} Further ablations demonstrate its robustness under diverse training setups. These results collectively highlight Conda as an effective and broadly applicable optimizer for large-scale LLM training. The code is released on this https URL 

---
# Metamorphic Testing for Audio Content Moderation Software 

**Authors**: Wenxuan Wang, Yongjiang Wu, Junyuan Zhang, Shuqing Li, Yun Peng, Wenting Chen, Shuai Wang, Michael R. Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24215)  

**Abstract**: The rapid growth of audio-centric platforms and applications such as WhatsApp and Twitter has transformed the way people communicate and share audio content in modern society. However, these platforms are increasingly misused to disseminate harmful audio content, such as hate speech, deceptive advertisements, and explicit material, which can have significant negative consequences (e.g., detrimental effects on mental health). In response, researchers and practitioners have been actively developing and deploying audio content moderation tools to tackle this issue. Despite these efforts, malicious actors can bypass moderation systems by making subtle alterations to audio content, such as modifying pitch or inserting noise. Moreover, the effectiveness of modern audio moderation tools against such adversarial inputs remains insufficiently studied. To address these challenges, we propose MTAM, a Metamorphic Testing framework for Audio content Moderation software. Specifically, we conduct a pilot study on 2000 audio clips and define 14 metamorphic relations across two perturbation categories: Audio Features-Based and Heuristic perturbations. MTAM applies these metamorphic relations to toxic audio content to generate test cases that remain harmful while being more likely to evade detection. In our evaluation, we employ MTAM to test five commercial textual content moderation software and an academic model against three kinds of toxic content. The results show that MTAM achieves up to 38.6%, 18.3%, 35.1%, 16.7%, and 51.1% error finding rates (EFR) when testing commercial moderation software provided by Gladia, Assembly AI, Baidu, Nextdata, and Tencent, respectively, and it obtains up to 45.7% EFR when testing the state-of-the-art algorithms from the academy. 

---
# BeyondBench: Benchmark-Free Evaluation of Reasoning in Language Models 

**Authors**: Gaurav Srivastava, Aafiya Hussain, Zhenyu Bi, Swastik Roy, Priya Pitre, Meng Lu, Morteza Ziyadi, Xuan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24210)  

**Abstract**: Evaluating language models fairly is becoming harder as static benchmarks available on the internet risk contamination by training data. This makes it unclear whether models are truly reasoning or just recalling answers. In this paper, we introduce BeyondBench, an evaluation framework that avoids this problem by using algorithmic problem generation. Unlike traditional benchmarks that risk contamination from internet-scale training data, BeyondBench creates mathematically grounded problems on the fly, ensuring each test remains fresh and uncontaminated. Our framework covers 44 algorithmic tasks with a total of 117 variations, grouped into three difficulty levels: the Easy Suite (29 tasks) for basic arithmetic and statistics, the Medium Suite (5 tasks, 49 variations) for sequence patterns and reasoning, and the Hard Suite (10 tasks, 68 variations) tackling NP-complete and constraint satisfaction problems. Each task generates problems from a combinatorial space larger than 10^15 unique instances, with solutions verified deterministically by mathematical proofs. We evaluated 101 language models, including 85 open-source and 16 closed-source models, spanning sizes from 0.5B to 141B parameters and multiple quantization schemes. Our results show consistent reasoning deficiencies across model families, with performance degrading sharply as problem complexity increases from polynomial to exponential. In our Hard Suite evaluations, models such as Gemini-2.5-pro, Llama-3.3-70B, and Qwen2.5-72B achieved average accuracies of 56.38%, 26.91%, and 33.60%, respectively. Moreover, we observe that performance drops drastically without tool usage, with GPT-5, GPT-5-mini, and GPT-5-nano showing a decline of 16.81%, 28.05%, and 47.59% accuracy on the hard suite. Our leaderboard is publicly available at this https URL 

---
# BALR-SAM: Boundary-Aware Low-Rank Adaptation of SAM for Resource-Efficient Medical Image Segmentation 

**Authors**: Zelin Liu, Sicheng Dong, Bocheng Li, Yixuan Yang, Jiacheng Ruan, Chenxu Zhou, Suncheng Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24204)  

**Abstract**: Vision foundation models like the Segment Anything Model (SAM), pretrained on large-scale natural image datasets, often struggle in medical image segmentation due to a lack of domain-specific adaptation. In clinical practice, fine-tuning such models efficiently for medical downstream tasks with minimal resource demands, while maintaining strong performance, is challenging. To address these issues, we propose BALR-SAM, a boundary-aware low-rank adaptation framework that enhances SAM for medical imaging. It combines three tailored components: (1) a Complementary Detail Enhancement Network (CDEN) using depthwise separable convolutions and multi-scale fusion to capture boundary-sensitive features essential for accurate segmentation; (2) low-rank adapters integrated into SAM's Vision Transformer blocks to optimize feature representation and attention for medical contexts, while simultaneously significantly reducing the parameter space; and (3) a low-rank tensor attention mechanism in the mask decoder, cutting memory usage by 75% and boosting inference speed. Experiments on standard medical segmentation datasets show that BALR-SAM, without requiring prompts, outperforms several state-of-the-art (SOTA) methods, including fully fine-tuned MedSAM, while updating just 1.8% (11.7M) of its parameters. 

---
# Group-Relative REINFORCE Is Secretly an Off-Policy Algorithm: Demystifying Some Myths About GRPO and Its Friends 

**Authors**: Chaorui Yao, Yanxi Chen, Yuchang Sun, Yushuo Chen, Wenhao Zhang, Xuchen Pan, Yaliang Li, Bolin Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.24203)  

**Abstract**: Off-policy reinforcement learning (RL) for large language models (LLMs) is attracting growing interest, driven by practical constraints in real-world applications, the complexity of LLM-RL infrastructure, and the need for further innovations of RL methodologies. While classic REINFORCE and its modern variants like Group Relative Policy Optimization (GRPO) are typically regarded as on-policy algorithms with limited tolerance of off-policyness, we present in this work a first-principles derivation for group-relative REINFORCE without assuming a specific training data distribution, showing that it admits a native off-policy interpretation. This perspective yields two general principles for adapting REINFORCE to off-policy settings: regularizing policy updates, and actively shaping the data distribution. Our analysis demystifies some myths about the roles of importance sampling and clipping in GRPO, unifies and reinterprets two recent algorithms -- Online Policy Mirror Descent (OPMD) and Asymmetric REINFORCE (AsymRE) -- as regularized forms of the REINFORCE loss, and offers theoretical justification for seemingly heuristic data-weighting strategies. Our findings lead to actionable insights that are validated with extensive empirical studies, and open up new opportunities for principled algorithm design in off-policy RL for LLMs. Source code for this work is available at this https URL. 

---
# Can Large Language Models Express Uncertainty Like Human? 

**Authors**: Linwei Tao, Yi-Fan Yeh, Bo Kai, Minjing Dong, Tao Huang, Tom A. Lamb, Jialin Yu, Philip H.S. Torr, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24202)  

**Abstract**: Large language models (LLMs) are increasingly used in high-stakes settings, where overconfident responses can mislead users. Reliable confidence estimation has been shown to enhance trust and task accuracy. Yet existing methods face practical barriers: logits are often hidden, multi-sampling is computationally expensive, and verbalized numerical uncertainty (e.g., giving a 0-100 score) deviates from natural communication. We revisit linguistic confidence (LC), where models express uncertainty through hedging language (e.g., probably, might), offering a lightweight and human-centered alternative. To advance this direction, we (1) release the first diverse, large-scale dataset of hedging expressions with human-annotated confidence scores, and (2) propose a lightweight mapper that converts hedges into confidence scores at near-zero cost. Building on these resources, we (3) conduct the first systematic study of LC across modern LLMs and QA benchmarks, revealing that while most LLMs underperform in expressing reliable LC, carefully designed prompting achieves competitive calibration and discriminability. Finally, we (4) introduce a fine-tuning framework that further improves LC reliability. Taken together, our work positions linguistic confidence as a scalable, efficient, and human-aligned approach to LLM uncertainty estimation, and calls for deeper exploration of this promising yet underexplored direction. 

---
# Chat to Chip: Large Language Model Based Design of Arbitrarily Shaped Metasurfaces 

**Authors**: Huanshu Zhang, Lei Kang, Sawyer D. Campbell, Douglas H. Werner  

**Link**: [PDF](https://arxiv.org/pdf/2509.24196)  

**Abstract**: Traditional metasurface design is limited by the computational cost of full-wave simulations, preventing thorough exploration of complex configurations. Data-driven approaches have emerged as a solution to this bottleneck, replacing costly simulations with rapid neural network evaluations and enabling near-instant design for meta-atoms. Despite advances, implementing a new optical function still requires building and training a task-specific network, along with exhaustive searches for suitable architectures and hyperparameters. Pre-trained large language models (LLMs), by contrast, sidestep this laborious process with a simple fine-tuning technique. However, applying LLMs to the design of nanophotonic devices, particularly for arbitrarily shaped metasurfaces, is still in its early stages; as such tasks often require graphical networks. Here, we show that an LLM, fed with descriptive inputs of arbitrarily shaped metasurface geometries, can learn the physical relationships needed for spectral prediction and inverse design. We further benchmarked a range of open-weight LLMs and identified relationships between accuracy and model size at the billion-parameter level. We demonstrated that 1-D token-wise LLMs provide a practical tool to designing 2-D arbitrarily shaped metasurfaces. Linking natural-language interaction to electromagnetic modelling, this "chat-to-chip" workflow represents a step toward more user-friendly data-driven nanophotonics. 

---
# AceSearcher: Bootstrapping Reasoning and Search for LLMs via Reinforced Self-Play 

**Authors**: Ran Xu, Yuchen Zhuang, Zihan Dong, Jonathan Wang, Yue Yu, Joyce C. Ho, Linjun Zhang, Haoyu Wang, Wenqi Shi, Carl Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24193)  

**Abstract**: Search-augmented LLMs often struggle with complex reasoning tasks due to ineffective multi-hop retrieval and limited reasoning ability. We propose AceSearcher, a cooperative self-play framework that trains a single large language model (LLM) to alternate between two roles: a decomposer that breaks down complex queries and a solver that integrates retrieved contexts for answer generation. AceSearcher couples supervised fine-tuning on a diverse mixture of search, reasoning, and decomposition tasks with reinforcement fine-tuning optimized for final answer accuracy, eliminating the need for intermediate annotations. Extensive experiments on three reasoning-intensive tasks across 10 datasets show that AceSearcher outperforms state-of-the-art baselines, achieving an average exact match improvement of 7.6%. Remarkably, on document-level finance reasoning tasks, AceSearcher-32B matches the performance of the DeepSeek-V3 model using less than 5% of its parameters. Even at smaller scales (1.5B and 8B), AceSearcher often surpasses existing search-augmented LLMs with up to 9x more parameters, highlighting its exceptional efficiency and effectiveness in tackling complex reasoning tasks. Our code will be published at this https URL and this https URL. 

---
# Talk in Pieces, See in Whole: Disentangling and Hierarchical Aggregating Representations for Language-based Object Detection 

**Authors**: Sojung An, Kwanyong Park, Yong Jae Lee, Donghyun Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.24192)  

**Abstract**: While vision-language models (VLMs) have made significant progress in multimodal perception (e.g., open-vocabulary object detection) with simple language queries, state-of-the-art VLMs still show limited ability to perceive complex queries involving descriptive attributes and relational clauses. Our in-depth analysis shows that these limitations mainly stem from text encoders in VLMs. Such text encoders behave like bags-of-words and fail to separate target objects from their descriptive attributes and relations in complex queries, resulting in frequent false positives. To address this, we propose restructuring linguistic representations according to the hierarchical relations within sentences for language-based object detection. A key insight is the necessity of disentangling textual tokens into core components-objects, attributes, and relations ("talk in pieces")-and subsequently aggregating them into hierarchically structured sentence-level representations ("see in whole"). Building on this principle, we introduce the TaSe framework with three main contributions: (1) a hierarchical synthetic captioning dataset spanning three tiers from category names to descriptive sentences; (2) Talk in Pieces, the three-component disentanglement module guided by a novel disentanglement loss function, transforms text embeddings into subspace compositions; and (3) See in Whole, which learns to aggregate disentangled components into hierarchically structured embeddings with the guide of proposed hierarchical objectives. The proposed TaSe framework strengthens the inductive bias of hierarchical linguistic structures, resulting in fine-grained multimodal representations for language-based object detection. Experimental results under the OmniLabel benchmark show a 24% performance improvement, demonstrating the importance of linguistic compositionality. 

---
# Beyond Overall Accuracy: A Psychometric Deep Dive into the Topic-Specific Medical Capabilities of 80 Large Language Models 

**Authors**: Zhimeng Luo, Lixin Wu, Adam Frisch, Daqing He  

**Link**: [PDF](https://arxiv.org/pdf/2509.24186)  

**Abstract**: As Large Language Models (LLMs) are increasingly proposed for high-stakes medical applications, there has emerged a critical need for reliable and accurate evaluation methodologies. Traditional accuracy metrics fail inadequately as they neither capture question characteristics nor offer topic-specific insights. To address this gap, we introduce \textsc{MedIRT}, a rigorous evaluation framework grounded in Item Response Theory (IRT), the gold standard in high-stakes educational testing. Unlike previous research relying on archival data, we prospectively gathered fresh responses from 80 diverse LLMs on a balanced, 1,100-question USMLE-aligned benchmark. Using one unidimensional two-parameter logistic IRT model per topic, we estimate LLM's latent model ability jointly with question difficulty and discrimination, yielding more stable and nuanced performance rankings than accuracy alone. Notably, we identify distinctive ``spiky'' ability profiles, where overall rankings can be misleading due to highly specialized model abilities. While \texttt{GPT-5} was the top performer in a majority of domains (8 of 11), it was outperformed in Social Science and Communication by \texttt{Claude-3-opus}, demonstrating that even an overall 23rd-ranked model can hold the top spot for specific competencies. Furthermore, we demonstrate IRT's utility in auditing benchmarks by identifying flawed questions. We synthesize these findings into a practical decision-support framework that integrates our multi-factor competency profiles with operational metrics. This work establishes a robust, psychometrically grounded methodology essential for the safe, effective, and trustworthy deployment of LLMs in healthcare. 

---
# Retrieval-augmented GUI Agents with Generative Guidelines 

**Authors**: Ran Xu, Kaixin Ma, Wenhao Yu, Hongming Zhang, Joyce C. Ho, Carl Yang, Dong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.24183)  

**Abstract**: GUI agents powered by vision-language models (VLMs) show promise in automating complex digital tasks. However, their effectiveness in real-world applications is often limited by scarce training data and the inherent complexity of these tasks, which frequently require long-tailed knowledge covering rare, unseen scenarios. We propose RAG-GUI , a lightweight VLM that leverages web tutorials at inference time. RAG-GUI is first warm-started via supervised finetuning (SFT) and further refined through self-guided rejection sampling finetuning (RSF). Designed to be model-agnostic, RAG-GUI functions as a generic plug-in that enhances any VLM-based agent. Evaluated across three distinct tasks, it consistently outperforms baseline agents and surpasses other inference baselines by 2.6% to 13.3% across two model sizes, demonstrating strong generalization and practical plug-and-play capabilities in real-world scenarios. 

---
# Stable Forgetting: Bounded Parameter-Efficient Unlearning in LLMs 

**Authors**: Arpit Garg, Hemanth Saratchandran, Ravi Garg, Simon Lucey  

**Link**: [PDF](https://arxiv.org/pdf/2509.24166)  

**Abstract**: Machine unlearning in large language models (LLMs) is essential for privacy and safety; however, existing approaches remain unstable and unreliable. A widely used strategy, the gradient difference method, applies gradient descent on retained data while performing gradient ascent on forget data, the data whose influence should be removed. However, when combined with cross-entropy loss, this procedure causes unbounded growth of weights and gradients, leading to training instability and degrading both forgetting and retention. We provide a theoretical framework that explains this failure, explicitly showing how ascent on the forget set destabilizes optimization in the feedforward MLP layers of LLMs. Guided by this insight, we propose Bounded Parameter-Efficient Unlearning, a parameter-efficient approach that stabilizes LoRA-based fine-tuning by applying bounded functions to MLP adapters. This simple modification controls the weight dynamics during ascent, enabling the gradient difference method to converge reliably. Across the TOFU, TDEC, and MUSE benchmarks, and across architectures and scales from 125M to 8B parameters, our method achieves substantial improvements in forgetting while preserving retention, establishing a novel theoretically grounded and practically scalable framework for unlearning in LLMs. 

---
# LatXGen: Towards Radiation-Free and Accurate Quantitative Analysis of Sagittal Spinal Alignment Via Cross-Modal Radiographic View Synthesis 

**Authors**: Moxin Zhao, Nan Meng, Jason Pui Yin Cheung, Chris Yuk Kwan Tang, Chenxi Yu, Wenting Zhong, Pengyu Lu, Chang Shi, Yipeng Zhuang, Teng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24165)  

**Abstract**: Adolescent Idiopathic Scoliosis (AIS) is a complex three-dimensional spinal deformity, and accurate morphological assessment requires evaluating both coronal and sagittal alignment. While previous research has made significant progress in developing radiation-free methods for coronal plane assessment, reliable and accurate evaluation of sagittal alignment without ionizing radiation remains largely underexplored. To address this gap, we propose LatXGen, a novel generative framework that synthesizes realistic lateral spinal radiographs from posterior Red-Green-Blue and Depth (RGBD) images of unclothed backs. This enables accurate, radiation-free estimation of sagittal spinal alignment. LatXGen tackles two core challenges: (1) inferring sagittal spinal morphology changes from a lateral perspective based on posteroanterior surface geometry, and (2) performing cross-modality translation from RGBD input to the radiographic domain. The framework adopts a dual-stage architecture that progressively estimates lateral spinal structure and synthesizes corresponding radiographs. To enhance anatomical consistency, we introduce an attention-based Fast Fourier Convolution (FFC) module for integrating anatomical features from RGBD images and 3D landmarks, and a Spatial Deformation Network (SDN) to model morphological variations in the lateral view. Additionally, we construct the first large-scale paired dataset for this task, comprising 3,264 RGBD and lateral radiograph pairs. Experimental results demonstrate that LatXGen produces anatomically accurate radiographs and outperforms existing GAN-based methods in both visual fidelity and quantitative metrics. This study offers a promising, radiation-free solution for sagittal spine assessment and advances comprehensive AIS evaluation. 

---
# Memory Transfer Planning: LLM-driven Context-Aware Code Adaptation for Robot Manipulation 

**Authors**: Tomoyuki Kagaya, Subramanian Lakshmi, Yuxuan Lou, Thong Jing Yuan, Jayashree Karlekar, Sugiri Pranata, Natsuki Murakami, Akira Kinose, Yang You  

**Link**: [PDF](https://arxiv.org/pdf/2509.24160)  

**Abstract**: Large language models (LLMs) are increasingly explored in robot manipulation, but many existing methods struggle to adapt to new environments. Many systems require either environment-specific policy training or depend on fixed prompts and single-shot code generation, leading to limited transferability and manual re-tuning. We introduce Memory Transfer Planning (MTP), a framework that leverages successful control-code examples from different environments as procedural knowledge, using them as in-context guidance for LLM-driven planning. Specifically, MTP (i) generates an initial plan and code using LLMs, (ii) retrieves relevant successful examples from a code memory, and (iii) contextually adapts the retrieved code to the target setting for re-planning without updating model parameters. We evaluate MTP on RLBench, CALVIN, and a physical robot, demonstrating effectiveness beyond simulation. Across these settings, MTP consistently improved success rate and adaptability compared with fixed-prompt code generation, naive retrieval, and memory-free re-planning. Furthermore, in hardware experiments, leveraging a memory constructed in simulation proved effective. MTP provides a practical approach that exploits procedural knowledge to realize robust LLM-based planning across diverse robotic manipulation scenarios, enhancing adaptability to novel environments and bridging simulation and real-world deployment. 

---
# Accelerating Cerebral Diagnostics with BrainFusion: A Comprehensive MRI Tumor Framework 

**Authors**: Walid Houmaidi, Youssef Sabiri, Salmane El Mansour Billah, Amine Abouaomar  

**Link**: [PDF](https://arxiv.org/pdf/2509.24149)  

**Abstract**: The early and accurate classification of brain tumors is crucial for guiding effective treatment strategies and improving patient outcomes. This study presents BrainFusion, a significant advancement in brain tumor analysis using magnetic resonance imaging (MRI) by combining fine-tuned convolutional neural networks (CNNs) for tumor classification--including VGG16, ResNet50, and Xception--with YOLOv8 for precise tumor localization with bounding boxes. Leveraging the Brain Tumor MRI Dataset, our experiments reveal that the fine-tuned VGG16 model achieves test accuracy of 99.86%, substantially exceeding previous benchmarks. Beyond setting a new accuracy standard, the integration of bounding-box localization and explainable AI techniques further enhances both the clinical interpretability and trustworthiness of the system's outputs. Overall, this approach underscores the transformative potential of deep learning in delivering faster, more reliable diagnoses, ultimately contributing to improved patient care and survival rates. 

---
# TENET: Leveraging Tests Beyond Validation for Code Generation 

**Authors**: Yiran Hu, Nan Jiang, Shanchao Liang, Yi Wu, Lin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24148)  

**Abstract**: Test-Driven Development (TDD) is a widely adopted software engineering practice that requires developers to create and execute tests alongside code implementation, ensuring that software behavior is continuously validated and refined. In the era of vibe coding, where developers increasingly delegate code writing to large language models (LLMs) by specifying high-level intentions, TDD becomes even more crucial, as test cases serve as executable specifications that explicitly define and verify intended functionality beyond what natural-language descriptions and code context can convey. While vibe coding under TDD is promising, there are three main challenges: (1) selecting a small yet effective test suite to improve the generation accuracy and control the execution workload, (2) retrieving context such as relevant code effectively, and (3) systematically using test feedback for effective code refinement. To address these challenges, we introduce TENET, an LLM agent for generating functions in complex real-world repositories under the TDD setting. TENET features three components: (1) a novel test harness mechanism that selects a concise test suite to maximize diversity of target usage scenarios; (2) a tailored agent toolset that performs efficient retrieval of relevant code with interactive debugging; and (3) a reflection-based refinement workflow that iteratively analyzes failures, replenishes context, and applies code refinement. TENET achieves 69.08% and 81.77% Pass@1 on RepoCod and RepoEval benchmarks, outperforming the best agentic baselines by 9.49 and 2.17 percentage points, respectively. In addition, this is the first study of test-driven code generation with repository-level context, examining how different aspects of test suites affect the performance of LLM agents under the TDD setting. 

---
# Your thoughts tell who you are: Characterize the reasoning patterns of LRMs 

**Authors**: Yida Chen, Yuning Mao, Xianjun Yang, Suyu Ge, Shengjie Bi, Lijuan Liu, Saghar Hosseini, Liang Tan, Yixin Nie, Shaoliang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2509.24147)  

**Abstract**: Current comparisons of large reasoning models (LRMs) focus on macro-level statistics such as task accuracy or reasoning length. Whether different LRMs reason differently remains an open question. To address this gap, we introduce the LLM-proposed Open Taxonomy (LOT), a classification method that uses a generative language model to compare reasoning traces from two LRMs and articulate their distinctive features in words. LOT then models how these features predict the source LRM of a reasoning trace based on their empirical distributions across LRM outputs. Iterating this process over a dataset of reasoning traces yields a human-readable taxonomy that characterizes how models think. We apply LOT to compare the reasoning of 12 open-source LRMs on tasks in math, science, and coding. LOT identifies systematic differences in their thoughts, achieving 80-100% accuracy in distinguishing reasoning traces from LRMs that differ in scale, base model family, or objective domain. Beyond classification, LOT's natural-language taxonomy provides qualitative explanations of how LRMs think differently. Finally, in a case study, we link the reasoning differences to performance: aligning the reasoning style of smaller Qwen3 models with that of the largest Qwen3 during test time improves their accuracy on GPQA by 3.3-5.7%. 

---
# EYE-DEX: Eye Disease Detection and EXplanation System 

**Authors**: Youssef Sabiri, Walid Houmaidi, Amine Abouaomar  

**Link**: [PDF](https://arxiv.org/pdf/2509.24136)  

**Abstract**: Retinal disease diagnosis is critical in preventing vision loss and reducing socioeconomic burdens. Globally, over 2.2 billion people are affected by some form of vision impairment, resulting in annual productivity losses estimated at $411 billion. Traditional manual grading of retinal fundus images by ophthalmologists is time-consuming and subjective. In contrast, deep learning has revolutionized medical diagnostics by automating retinal image analysis and achieving expert-level performance. In this study, we present EYE-DEX, an automated framework for classifying 10 retinal conditions using the large-scale Retinal Disease Dataset comprising 21,577 eye fundus images. We benchmark three pre-trained Convolutional Neural Network (CNN) models--VGG16, VGG19, and ResNet50--with our finetuned VGG16 achieving a state-of-the-art global benchmark test accuracy of 92.36%. To enhance transparency and explainability, we integrate the Gradient-weighted Class Activation Mapping (Grad-CAM) technique to generate visual explanations highlighting disease-specific regions, thereby fostering clinician trust and reliability in AI-assisted diagnostics. 

---
# ASTROCO: Self-Supervised Conformer-Style Transformers for Light-Curve Embeddings 

**Authors**: Antony Tan, Pavlos Protopapas, Martina Cádiz-Leyton, Guillermo Cabrera-Vives, Cristobal Donoso-Oliva, Ignacio Becker  

**Link**: [PDF](https://arxiv.org/pdf/2509.24134)  

**Abstract**: We present AstroCo, a Conformer-style encoder for irregular stellar light curves. By combining attention with depthwise convolutions and gating, AstroCo captures both global dependencies and local features. On MACHO R-band, AstroCo outperforms Astromer v1 and v2, yielding 70 percent and 61 percent lower error respectively and a relative macro-F1 gain of about 7 percent, while producing embeddings that transfer effectively to few-shot classification. These results highlight AstroCo's potential as a strong and label-efficient foundation for time-domain astronomy. 

---
# BOSfM: A View Planning Framework for Optimal 3D Reconstruction of Agricultural Scenes 

**Authors**: Athanasios Bacharis, Konstantinos D. Polyzos, Georgios B. Giannakis, Nikolaos Papanikolopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2509.24126)  

**Abstract**: Active vision (AV) has been in the spotlight of robotics research due to its emergence in numerous applications including agricultural tasks such as precision crop monitoring and autonomous harvesting to list a few. A major AV problem that gained popularity is the 3D reconstruction of targeted environments using 2D images from diverse viewpoints. While collecting and processing a large number of arbitrarily captured 2D images can be arduous in many practical scenarios, a more efficient solution involves optimizing the placement of available cameras in 3D space to capture fewer, yet more informative, images that provide sufficient visual information for effective reconstruction of the environment of interest. This process termed as view planning (VP), can be markedly challenged (i) by noise emerging in the location of the cameras and/or in the extracted images, and (ii) by the need to generalize well in other unknown similar agricultural environments without need for re-optimizing or re-training. To cope with these challenges, the present work presents a novel VP framework that considers a reconstruction quality-based optimization formulation that relies on the notion of `structure-from-motion' to reconstruct the 3D structure of the sought environment from the selected 2D images. With no analytic expression of the optimization function and with costly function evaluations, a Bayesian optimization approach is proposed to efficiently carry out the VP process using only a few function evaluations, while accounting for different noise cases. Numerical tests on both simulated and real agricultural settings signify the benefits of the advocated VP approach in efficiently estimating the optimal camera placement to accurately reconstruct 3D environments of interest, and generalize well on similar unknown environments. 

---
# The Impossibility of Inverse Permutation Learning in Transformer Models 

**Authors**: Rohan Alur, Chris Hays, Manish Raghavan, Devavrat Shah  

**Link**: [PDF](https://arxiv.org/pdf/2509.24125)  

**Abstract**: In this technical note, we study the problem of inverse permutation learning in decoder-only transformers. Given a permutation and a string to which that permutation has been applied, the model is tasked with producing the original (``canonical'') string. We argue that this task models a natural robustness property across a variety of reasoning tasks, including long-context retrieval, multiple choice QA and in-context learning. Our primary contribution is an impossibility result: we show that an arbitrary depth, decoder-only transformer cannot learn this task. This result concerns the expressive capacity of decoder-only transformer models and is agnostic to training dynamics or sample complexity. We give a pair of alternative constructions under which inverse permutation learning is feasible. The first of these highlights the fundamental role of the causal attention mask, and reveals a gap between the expressivity of encoder-decoder transformers and the more popular decoder-only architecture. The latter result is more surprising: we show that simply padding the input with ``scratch tokens" yields a construction under which inverse permutation learning is possible. We conjecture that this may suggest an alternative mechanism by which chain-of-thought prompting or, more generally, intermediate ``thinking'' tokens can enable reasoning in large language models, even when these tokens encode no meaningful semantic information (e.g., the results of intermediate computations). 

---
# Ancestry Tree Clustering for Particle Filter Diversity Maintenance 

**Authors**: Ilari Vallivaara, Bingnan Duan, Yinhuan Dong, Tughrul Arslan  

**Link**: [PDF](https://arxiv.org/pdf/2509.24124)  

**Abstract**: We propose a method for linear-time diversity maintenance in particle filtering. It clusters particles based on ancestry tree topology: closely related particles in sufficiently large subtrees are grouped together. The main idea is that the tree structure implicitly encodes similarity without the need for spatial or other domain-specific metrics. This approach, when combined with intra-cluster fitness sharing and the protection of particles not included in a cluster, effectively prevents premature convergence in multimodal environments while maintaining estimate compactness. We validate our approach in a multimodal robotics simulation and a real-world multimodal indoor environment. We compare the performance to several diversity maintenance algorithms from the literature, including Deterministic Resampling and Particle Gaussian Mixtures. Our algorithm achieves high success rates with little to no negative effect on compactness, showing particular robustness to different domains and challenging initial conditions. 

---
# GEAR: A General Evaluation Framework for Abductive Reasoning 

**Authors**: Kaiyu He, Peilin Wu, Mian Zhang, Kun Wan, Wentian Zhao, Xinya Du, Zhiyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24096)  

**Abstract**: Since the advent of large language models (LLMs), research has focused on instruction following and deductive reasoning. A central question remains: can these models discover new knowledge, and how can we evaluate this ability? We address this by studying abductive reasoning-the generation of plausible hypotheses to explain observations-and introduce GEAR (General Evaluation for Abductive Reasoning), a general-purpose, fully automated, transparent, and label-free evaluation paradigm. GEAR scores hypothesis sets by three metrics: consistency (each hypothesis explains the observations), generalizability (consistent hypotheses make meaningful predictions on unseen inputs), and diversity (the set covers distinct predictions and patterns). Built this way, GEAR is scalable (no human gold answers), reliable (deterministic scoring aligned with classical abduction), and open-ended (scores improve only when models produce new plausible hypotheses, unlike static benchmarks that saturate once accuracy is high). Using GEAR, we conduct a fine-grained study of nine LLMs on four abduction benchmarks with 1,500 problems, generating over 50,000 candidate hypotheses and revealing model differences obscured by gold-answer or purely human evaluations. We further propose a momentum-based curriculum that adjusts GEAR-derived training data by learning velocity: it starts with what the model learns quickly and shifts toward harder objectives such as generating diverse hypotheses once the model is confident on foundational objectives. Without gold-label supervision, this strategy improves all GEAR objectives and these gains transfer to established abductive reasoning benchmarks. Taken together, GEAR provides a principled framework that evaluates abduction and supplies label-free, scalable training signals that help LLMs produce more diverse and reliable hypotheses. 

---
# PerfBench: Can Agents Resolve Real-World Performance Bugs? 

**Authors**: Spandan Garg, Roshanak Zilouchian Moghaddam  

**Link**: [PDF](https://arxiv.org/pdf/2509.24091)  

**Abstract**: Performance bugs are inefficiencies in software that waste computational resources without causing functional failures, making them particularly challenging to detect and fix. While recent advances in Software Engineering agents have shown promise in automated bug fixing, existing benchmarks primarily focus on functional correctness and fail to evaluate agents' abilities to identify and resolve non-functional issues like performance bugs. We introduce PerfBench, a benchmark comprising 81 real-world performance bug-fixing tasks from popular .NET repositories on GitHub. Unlike existing benchmarks that rely on pre-existing test suites, PerfBench features a novel evaluation harness that allows agents to generate their own performance benchmarks and validates fixes by comparing execution metrics collected for developer fix and agent fix. Each task in PerfBench is derived from actual developer fixes linked to performance-related issues, which are then verified by human experts, ensuring real-world relevance. Our evaluation reveals that current state-of-the-art coding agents struggle with performance optimization tasks, with baseline OpenHands agent achieving only a ~3% success rate on our benchmark. We develop OpenHands-Perf-Agent, which incorporates performance-aware tooling and instructions and achieves a ~20% success rate on the benchmark. We show that by ensuring the agent has proper instructions to benchmark its changes and tooling for benchmark output processing, we can improve the agent performance significantly, but room for improvement still remains. PerfBench provides a challenging test set for furthering the capabilities of agents in fixing performance issues. 

---
# Large-Scale Constraint Generation - Can LLMs Parse Hundreds of Constraints? 

**Authors**: Matteo Boffa, Jiaxuan You  

**Link**: [PDF](https://arxiv.org/pdf/2509.24090)  

**Abstract**: Recent research has explored the constrained generation capabilities of Large Language Models (LLMs) when explicitly prompted by few task-specific requirements. In contrast, we introduce Large-Scale Constraint Generation (LSCG), a new problem that evaluates whether LLMs can parse a large, fine-grained, generic list of constraints. To examine the LLMs' ability to handle an increasing number constraints, we create a practical instance of LSCG, called Words Checker. In Words Checker, we evaluate the impact of model characteristics (e.g., size, family) and steering techniques (e.g., Simple Prompt, Chain of Thought, Best of N) on performance. We also propose FoCusNet, a small and dedicated model that parses the original list of constraints into a smaller subset, helping the LLM focus on relevant constraints. Experiments reveal that existing solutions suffer a significant performance drop as the number of constraints increases, with FoCusNet showing an 8-13% accuracy boost. 

---
# PEARL: Peer-Enhanced Adaptive Radio via On-Device LLM 

**Authors**: Ju-Hyung Lee, Yanqing Lu, Klaus Doppler  

**Link**: [PDF](https://arxiv.org/pdf/2509.24085)  

**Abstract**: We present PEARL (Peer-Enhanced Adaptive Radio via On-Device LLM), a framework for cooperative cross-layer optimization in device-to-device (D2D) communication. Building on our previous work on single-device on-device LLMs, PEARL extends the paradigm by leveraging both publisher and subscriber states to guide Wi-Fi Aware (WA) parameter selection. A context-aware reward, which normalizes latency by application tolerances and modulates energy by device battery states, provides richer supervision for KL-based finetuning. We study two lightweight variants: PEARL (Head + Low-Rank Adaptation (LoRA)) achieves the best overall performance, while PEARL-Lite (Head-only) delivers sub-20 ms inference at near-identical objective scores. Across synthetic scenarios grounded in real measurements, PEARL improves objective scores over heuristic and compact model baselines and reduces energy by up to 16% in cooperative low-battery cases. These results demonstrate that peer-aware context, reward-aligned training, and head-based efficiency make LLMs practical for always-on, on-device cross-layer control. 

---
# Uncovering Grounding IDs: How External Cues Shape Multi-Modal Binding 

**Authors**: Hosein Hasani, Amirmohammad Izadi, Fatemeh Askari, Mobin Bagherian, Sadegh Mohammadian, Mohammad Izadi, Mahdieh Soleymani Baghshah  

**Link**: [PDF](https://arxiv.org/pdf/2509.24072)  

**Abstract**: Large vision-language models (LVLMs) show strong performance across multimodal benchmarks but remain limited in structured reasoning and precise grounding. Recent work has demonstrated that adding simple visual structures, such as partitions and annotations, improves accuracy, yet the internal mechanisms underlying these gains remain unclear. We investigate this phenomenon and propose the concept of Grounding IDs, latent identifiers induced by external cues that bind objects to their designated partitions across modalities. Through representation analysis, we find that these identifiers emerge as robust within-partition alignment in embedding space and reduce the modality gap between image and text. Causal interventions further confirm that these identifiers mediate binding between objects and symbolic cues. We show that Grounding IDs strengthen attention between related components, which in turn improves cross-modal grounding and reduces hallucinations. Taken together, our results identify Grounding IDs as a key symbolic mechanism explaining how external cues enhance multimodal binding, offering both interpretability and practical improvements in robustness. 

---
# AQUAIR: A High-Resolution Indoor Environmental Quality Dataset for Smart Aquaculture Monitoring 

**Authors**: Youssef Sabiri, Walid Houmaidi, Ouail El Maadi, Yousra Chtouki  

**Link**: [PDF](https://arxiv.org/pdf/2509.24069)  

**Abstract**: Smart aquaculture systems depend on rich environmental data streams to protect fish welfare, optimize feeding, and reduce energy use. Yet public datasets that describe the air surrounding indoor tanks remain scarce, limiting the development of forecasting and anomaly-detection tools that couple head-space conditions with water-quality dynamics. We therefore introduce AQUAIR, an open-access public dataset that logs six Indoor Environmental Quality (IEQ) variables--air temperature, relative humidity, carbon dioxide, total volatile organic compounds, PM2.5 and PM10--inside a fish aquaculture facility in Amghass, Azrou, Morocco. A single Awair HOME monitor sampled every five minutes from 14 October 2024 to 9 January 2025, producing more than 23,000 time-stamped observations that are fully quality-controlled and publicly archived on Figshare. We describe the sensor placement, ISO-compliant mounting height, calibration checks against reference instruments, and an open-source processing pipeline that normalizes timestamps, interpolates short gaps, and exports analysis-ready tables. Exploratory statistics show stable conditions (median CO2 = 758 ppm; PM2.5 = 12 micrograms/m3) with pronounced feeding-time peaks, offering rich structure for short-horizon forecasting, event detection, and sensor drift studies. AQUAIR thus fills a critical gap in smart aquaculture informatics and provides a reproducible benchmark for data-centric machine learning curricula and environmental sensing research focused on head-space dynamics in recirculating aquaculture systems. 

---
# A Small Math Model: Recasting Strategy Choice Theory in an LLM-Inspired Architecture 

**Authors**: Roussel Rahman, Jeff Shrager  

**Link**: [PDF](https://arxiv.org/pdf/2509.24068)  

**Abstract**: Strategy Choice Theory (SCT)\footnote{``Strategy Choice Theory'', ``Distributions of Associations'', and ``Overlapping Wave Theory'' have been used to refer to this line of work, emphasizing different aspects.}\citep[e.g.,][]{siegler1984strategychoices, siegler2000rebirth} explains important aspects of children's arithmetic learning based upon principles including learning from developmentally naturalistic data, probabilistic representation, confidence-based retrieval, and the phase-like importance of scaffolding strategies, such as finger-counting. Here we recast SCT as a ``Small Math Model'' (SMM), employing a neural-network-based architecture analogous to LLMs. The SMM extends SCT to include counting practice\footnote{The original SCT model was pre-biased in accordance with the supposed experience of counting.}, symbol (number) embedding, and gated attention. Similar to earlier work, the SMM demonstrates constructive and destructive interference between counting and addition, and the ``wave-like'' use of finger-counting as sum recall improves. We plan to extend the SMM to later aspects of the decades-long SCT program, including adaptive strategy choice and eventually strategy discovery, providing a unified platform to investigate the understanding of numerical characteristics and relationships essential for mathematical reasoning -- as it can emerge in LLM-based agents. 

---
# In-Context Compositional Q-Learning for Offline Reinforcement Learning 

**Authors**: Qiushui Xu, Yuhao Huang, Yushu Jiang, Lei Song, Jinyu Wang, Wenliang Zheng, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2509.24067)  

**Abstract**: Accurately estimating the Q-function is a central challenge in offline reinforcement learning. However, existing approaches often rely on a single global Q-function, which struggles to capture the compositional nature of tasks involving diverse subtasks. We propose In-context Compositional Q-Learning (\texttt{ICQL}), the first offline RL framework that formulates Q-learning as a contextual inference problem, using linear Transformers to adaptively infer local Q-functions from retrieved transitions without explicit subtask labels. Theoretically, we show that under two assumptions--linear approximability of the local Q-function and accurate weight inference from retrieved context--\texttt{ICQL} achieves bounded Q-function approximation error, and supports near-optimal policy extraction. Empirically, \texttt{ICQL} substantially improves performance in offline settings: improving performance in kitchen tasks by up to 16.4\%, and in Gym and Adroit tasks by up to 8.6\% and 6.3\%. These results highlight the underexplored potential of in-context learning for robust and compositional value estimation, positioning \texttt{ICQL} as a principled and effective framework for offline RL. 

---
# A Second-Order Perspective on Pruning at Initialization and Knowledge Transfer 

**Authors**: Leonardo Iurada, Beatrice Occhiena, Tatiana Tommasi  

**Link**: [PDF](https://arxiv.org/pdf/2509.24066)  

**Abstract**: The widespread availability of pre-trained vision models has enabled numerous deep learning applications through their transferable representations. However, their computational and storage costs often limit practical deployment. Pruning-at-Initialization has emerged as a promising approach to compress models before training, enabling efficient task-specific adaptation. While conventional wisdom suggests that effective pruning requires task-specific data, this creates a challenge when downstream tasks are unknown in advance. In this paper, we investigate how data influences the pruning of pre-trained vision models. Surprisingly, pruning on one task retains the model's zero-shot performance also on unseen tasks. Furthermore, fine-tuning these pruned models not only improves performance on original seen tasks but can recover held-out tasks' performance. We attribute this phenomenon to the favorable loss landscapes induced by extensive pre-training on large-scale datasets. 

---
# PartnerMAS: An LLM Hierarchical Multi-Agent Framework for Business Partner Selection on High-Dimensional Features 

**Authors**: Lingyao Li, Haolun Wu, Zhenkun Li, Jiabei Hu, Yu Wang, Xiaoshan Huang, Wenyue Hua, Wenqian Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.24046)  

**Abstract**: High-dimensional decision-making tasks, such as business partner selection, involve evaluating large candidate pools with heterogeneous numerical, categorical, and textual features. While large language models (LLMs) offer strong in-context reasoning capabilities, single-agent or debate-style systems often struggle with scalability and consistency in such settings. We propose PartnerMAS, a hierarchical multi-agent framework that decomposes evaluation into three layers: a Planner Agent that designs strategies, Specialized Agents that perform role-specific assessments, and a Supervisor Agent that integrates their outputs. To support systematic evaluation, we also introduce a curated benchmark dataset of venture capital co-investments, featuring diverse firm attributes and ground-truth syndicates. Across 140 cases, PartnerMAS consistently outperforms single-agent and debate-based multi-agent baselines, achieving up to 10--15\% higher match rates. Analysis of agent reasoning shows that planners are most responsive to domain-informed prompts, specialists produce complementary feature coverage, and supervisors play an important role in aggregation. Our findings demonstrate that structured collaboration among LLM agents can generate more robust outcomes than scaling individual models, highlighting PartnerMAS as a promising framework for high-dimensional decision-making in data-rich domains. 

---
# End-to-end Topographic Auditory Models Replicate Signatures of Human Auditory Cortex 

**Authors**: Haider Al-Tahan, Mayukh Deb, Jenelle Feather, N. Apurva Ratan Murty  

**Link**: [PDF](https://arxiv.org/pdf/2509.24039)  

**Abstract**: The human auditory cortex is topographically organized. Neurons with similar response properties are spatially clustered, forming smooth maps for acoustic features such as frequency in early auditory areas, and modular regions selective for music and speech in higher-order cortex. Yet, evaluations for current computational models of auditory perception do not measure whether such topographic structure is present in a candidate model. Here, we show that cortical topography is not present in the previous best-performing models at predicting human auditory fMRI responses. To encourage the emergence of topographic organization, we adapt a cortical wiring-constraint loss originally designed for visual perception. The new class of topographic auditory models, TopoAudio, are trained to classify speech, and environmental sounds from cochleagram inputs, with an added constraint that nearby units on a 2D cortical sheet develop similar tuning. Despite these additional constraints, TopoAudio achieves high accuracy on benchmark tasks comparable to the unconstrained non-topographic baseline models. Further, TopoAudio predicts the fMRI responses in the brain as well as standard models, but unlike standard models, TopoAudio develops smooth, topographic maps for tonotopy and amplitude modulation (common properties of early auditory representation, as well as clustered response modules for music and speech (higher-order selectivity observed in the human auditory cortex). TopoAudio is the first end-to-end biologically grounded auditory model to exhibit emergent topography, and our results emphasize that a wiring-length constraint can serve as a general-purpose regularization tool to achieve biologically aligned representations. 

---
# GPS-MTM: Capturing Pattern of Normalcy in GPS-Trajectories with self-supervised learning 

**Authors**: Umang Garg, Bowen Zhang, Anantanjit Subrahmanya, Chandrakanth Gudavalli, BS Manjunath  

**Link**: [PDF](https://arxiv.org/pdf/2509.24031)  

**Abstract**: Foundation models have driven remarkable progress in text, vision, and video understanding, and are now poised to unlock similar breakthroughs in trajectory modeling. We introduce the GPSMasked Trajectory Transformer (GPS-MTM), a foundation model for large-scale mobility data that captures patterns of normalcy in human movement. Unlike prior approaches that flatten trajectories into coordinate streams, GPS-MTM decomposes mobility into two complementary modalities: states (point-of-interest categories) and actions (agent transitions). Leveraging a bi-directional Transformer with a self-supervised masked modeling objective, the model reconstructs missing segments across modalities, enabling it to learn rich semantic correlations without manual labels. Across benchmark datasets, including Numosim-LA, Urban Anomalies, and Geolife, GPS-MTM consistently outperforms on downstream tasks such as trajectory infilling and next-stop prediction. Its advantages are most pronounced in dynamic tasks (inverse and forward dynamics), where contextual reasoning is critical. These results establish GPS-MTM as a robust foundation model for trajectory analytics, positioning mobility data as a first-class modality for large-scale representation learning. Code is released for further reference. 

---
# From Edge to HPC: Investigating Cross-Facility Data Streaming Architectures 

**Authors**: Anjus George, Michael Brim, Christopher Zimmer, David Rogers, Sarp Oral, Zach Mayes  

**Link**: [PDF](https://arxiv.org/pdf/2509.24030)  

**Abstract**: In this paper, we investigate three cross-facility data streaming architectures, Direct Streaming (DTS), Proxied Streaming (PRS), and Managed Service Streaming (MSS). We examine their architectural variations in data flow paths and deployment feasibility, and detail their implementation using the Data Streaming to HPC (DS2HPC) architectural framework and the SciStream memory-to-memory streaming toolkit on the production-grade Advanced Computing Ecosystem (ACE) infrastructure at Oak Ridge Leadership Computing Facility (OLCF). We present a workflow-specific evaluation of these architectures using three synthetic workloads derived from the streaming characteristics of scientific workflows. Through simulated experiments, we measure streaming throughput, round-trip time, and overhead under work sharing, work sharing with feedback, and broadcast and gather messaging patterns commonly found in AI-HPC communication motifs. Our study shows that DTS offers a minimal-hop path, resulting in higher throughput and lower latency, whereas MSS provides greater deployment feasibility and scalability across multiple users but incurs significant overhead. PRS lies in between, offering a scalable architecture whose performance matches DTS in most cases. 

---
# FrameMind: Frame-Interleaved Chain-of-Thought for Video Reasoning via Reinforcement Learning 

**Authors**: Haonan Ge, Yiwei Wang, Kai-Wei Chang, Hang Wu, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.24008)  

**Abstract**: Current video understanding models rely on fixed frame sampling strategies, processing predetermined visual inputs regardless of the specific reasoning requirements of each question. This static approach limits their ability to adaptively gather visual evidence, leading to suboptimal performance on tasks that require either broad temporal coverage or fine-grained spatial detail. In this paper, we introduce FrameMind, an end-to-end framework trained with reinforcement learning that enables models to dynamically request visual information during reasoning through Frame-Interleaved Chain-of-Thought (FiCOT). Unlike traditional approaches, FrameMind operates in multiple turns where the model alternates between textual reasoning and active visual perception, using tools to extract targeted frames or video clips based on identified knowledge gaps. To train effective dynamic sampling policies, we propose Dynamic Resolution Frame Sampling (DRFS), which exposes models to diverse temporal-spatial trade-offs during learning, and DRFS-GRPO, a group-relative policy optimization algorithm that learns from outcome-based rewards without requiring frame-level annotations. Extensive experiments on challenging benchmarks like MLVU and VideoMME demonstrate that our method significantly outperforms existing models, advancing the state of the art in flexible and efficient video understanding. 

---
# SLA: Beyond Sparsity in Diffusion Transformers via Fine-Tunable Sparse-Linear Attention 

**Authors**: Jintao Zhang, Haoxu Wang, Kai Jiang, Shuo Yang, Kaiwen Zheng, Haocheng Xi, Ziteng Wang, Hongzhou Zhu, Min Zhao, Ion Stoica, Joseph E. Gonzalez, Jun Zhu, Jianfei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.24006)  

**Abstract**: In Diffusion Transformer (DiT) models, particularly for video generation, attention latency is a major bottleneck due to the long sequence length and the quadratic complexity. We find that attention weights can be separated into two parts: a small fraction of large weights with high rank and the remaining weights with very low rank. This naturally suggests applying sparse acceleration to the first part and low-rank acceleration to the second. Based on this finding, we propose SLA (Sparse-Linear Attention), a trainable attention method that fuses sparse and linear attention to accelerate diffusion models. SLA classifies attention weights into critical, marginal, and negligible categories, applying O(N^2) attention to critical weights, O(N) attention to marginal weights, and skipping negligible ones. SLA combines these computations into a single GPU kernel and supports both forward and backward passes. With only a few fine-tuning steps using SLA, DiT models achieve a 20x reduction in attention computation, resulting in significant acceleration without loss of generation quality. Experiments show that SLA reduces attention computation by 95% without degrading end-to-end generation quality, outperforming baseline methods. In addition, we implement an efficient GPU kernel for SLA, which yields a 13.7x speedup in attention computation and a 2.2x end-to-end speedup in video generation on Wan2.1-1.3B. 

---
# MCPMark: A Benchmark for Stress-Testing Realistic and Comprehensive MCP Use 

**Authors**: Zijian Wu, Xiangyan Liu, Xinyuan Zhang, Lingjun Chen, Fanqing Meng, Lingxiao Du, Yiran Zhao, Fanshi Zhang, Yaoqi Ye, Jiawei Wang, Zirui Wang, Jinjie Ni, Yufan Yang, Arvin Xu, Michael Qizhe Shieh  

**Link**: [PDF](https://arxiv.org/pdf/2509.24002)  

**Abstract**: MCP standardizes how LLMs interact with external systems, forming the foundation for general agents. However, existing MCP benchmarks remain narrow in scope: they focus on read-heavy tasks or tasks with limited interaction depth, and fail to capture the complexity and realism of real-world workflows. To address this gap, we propose MCPMark, a benchmark designed to evaluate MCP use in a more realistic and comprehensive manner. It consists of $127$ high-quality tasks collaboratively created by domain experts and AI agents. Each task begins with a curated initial state and includes a programmatic script for automatic verification. These tasks demand richer and more diverse interactions with the environment, involving a broad range of create, read, update, and delete (CRUD) operations. We conduct a comprehensive evaluation of cutting-edge LLMs using a minimal agent framework that operates in a tool-calling loop. Empirical results show that the best-performing model, gpt-5-medium, reaches only $52.56$\% pass@1 and $33.86$\% pass^4, while other widely regarded strong models, including claude-sonnet-4 and o3, fall below $30$\% pass@1 and $15$\% pass^4. On average, LLMs require $16.2$ execution turns and $17.4$ tool calls per task, significantly surpassing those in previous MCP benchmarks and highlighting the stress-testing nature of MCPMark. 

---
# The AI Agent Code of Conduct: Automated Guardrail Policy-as-Prompt Synthesis 

**Authors**: Gauri Kholkar, Ratinder Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2509.23994)  

**Abstract**: As autonomous AI agents are increasingly deployed in industry, it is essential to safeguard them. We introduce a novel framework that automates the translation of unstructured design documents into verifiable, real-time guardrails. We introduce "Policy as Prompt," a new approach that uses Large Language Models (LLMs) to interpret and enforce natural language policies by applying contextual understanding and the principle of least privilege. Our system first ingests technical artifacts to construct a verifiable policy tree, which is then compiled into lightweight, prompt-based classifiers that audit agent behavior at runtime. We validate our approach across diverse applications, demonstrating a scalable and auditable pipeline that bridges the critical policy-to-practice gap, paving the way for verifiably safer and more regulatable AI. 

---
# Guide: Generalized-Prior and Data Encoders for DAG Estimation 

**Authors**: Amartya Roy, Devharish N, Shreya Ganguly, Kripabandhu Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2509.23992)  

**Abstract**: Modern causal discovery methods face critical limitations in scalability, computational efficiency, and adaptability to mixed data types, as evidenced by benchmarks on node scalability (30, $\le 50$, $\ge 70$ nodes), computational energy demands, and continuous/non-continuous data handling. While traditional algorithms like PC, GES, and ICA-LiNGAM struggle with these challenges, exhibiting prohibitive energy costs for higher-order nodes and poor scalability beyond 70 nodes, we propose \textbf{GUIDE}, a framework that integrates Large Language Model (LLM)-generated adjacency matrices with observational data through a dual-encoder architecture. GUIDE uniquely optimizes computational efficiency, reducing runtime on average by $\approx 42%$ compared to RL-BIC and KCRL methods, while achieving an average $\approx 117%$ improvement in accuracy over both NOTEARS and GraN-DAG individually. During training, GUIDE's reinforcement learning agent dynamically balances reward maximization (accuracy) and penalty avoidance (DAG constraints), enabling robust performance across mixed data types and scalability to $\ge 70$ nodes -- a setting where baseline methods fail. 

---
# The Hidden Costs of Translation Accuracy: Distillation, Quantization, and Environmental Impact 

**Authors**: Dhaathri Vijay, Anandaswarup Vadapalli  

**Link**: [PDF](https://arxiv.org/pdf/2509.23990)  

**Abstract**: The rapid expansion of large language models (LLMs) has heightened concerns about their computational and environmental costs. This study investigates the trade-offs between translation quality and efficiency by comparing full-scale, distilled, and quantized models using machine translation as a case study. We evaluated performance on the Flores+ benchmark and through human judgments of conversational translations in French, Hindi, and Kannada. Our analysis of carbon emissions per evaluation run revealed that the full 3.3B fp32 model, while achieving the highest BLEU scores, incurred the largest environmental footprint (about 0.007-0.008 kg CO2 per run). The distilled models achieved an inference of up to 4.5x faster than the full 3.3B model, with only minimal reductions in BLEU scores. Human evaluations also showed that even aggressive quantization (INT4) preserved high levels of accuracy and fluency, with differences between models generally minor. These findings demonstrate that model compression strategies can substantially reduce computational demands and environmental impact while maintaining competitive translation quality, though trade-offs are more pronounced in low-resource settings. We argue for evaluation frameworks that integrate efficiency and sustainability alongside objective metrics as central dimensions of progress in NLP. 

---
# Toward Preference-aligned Large Language Models via Residual-based Model Steering 

**Authors**: Lucio La Cava, Andrea Tagarelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.23982)  

**Abstract**: Preference alignment is a critical step in making Large Language Models (LLMs) useful and aligned with (human) preferences. Existing approaches such as Reinforcement Learning from Human Feedback or Direct Preference Optimization typically require curated data and expensive optimization over billions of parameters, and eventually lead to persistent task-specific models. In this work, we introduce Preference alignment of Large Language Models via Residual Steering (PaLRS), a training-free method that exploits preference signals encoded in the residual streams of LLMs. From as few as one hundred preference pairs, PaLRS extracts lightweight, plug-and-play steering vectors that can be applied at inference time to push models toward preferred behaviors. We evaluate PaLRS on various small-to-medium-scale open-source LLMs, showing that PaLRS-aligned models achieve consistent gains on mathematical reasoning and code generation benchmarks while preserving baseline general-purpose performance. Moreover, when compared to DPO-aligned models, they perform better with huge time savings. Our findings highlight that PaLRS offers an effective, much more efficient and flexible alternative to standard preference optimization pipelines, offering a training-free, plug-and-play mechanism for alignment with minimal data. 

---
# MAD-PINN: A Decentralized Physics-Informed Machine Learning Framework for Safe and Optimal Multi-Agent Control 

**Authors**: Manan Tayal, Aditya Singh, Shishir Kolathaya, Somil Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2509.23960)  

**Abstract**: Co-optimizing safety and performance in large-scale multi-agent systems remains a fundamental challenge. Existing approaches based on multi-agent reinforcement learning (MARL), safety filtering, or Model Predictive Control (MPC) either lack strict safety guarantees, suffer from conservatism, or fail to scale effectively. We propose MAD-PINN, a decentralized physics-informed machine learning framework for solving the multi-agent state-constrained optimal control problem (MASC-OCP). Our method leverages an epigraph-based reformulation of SC-OCP to simultaneously capture performance and safety, and approximates its solution via a physics-informed neural network. Scalability is achieved by training the SC-OCP value function on reduced-agent systems and deploying them in a decentralized fashion, where each agent relies only on local observations of its neighbours for decision-making. To further enhance safety and efficiency, we introduce an Hamilton-Jacobi (HJ) reachability-based neighbour selection strategy to prioritize safety-critical interactions, and a receding-horizon policy execution scheme that adapts to dynamic interactions while reducing computational burden. Experiments on multi-agent navigation tasks demonstrate that MAD-PINN achieves superior safety-performance trade-offs, maintains scalability as the number of agents grows, and consistently outperforms state-of-the-art baselines. 

---
# Vision-Grounded Machine Interpreting: Improving the Translation Process through Visual Cues 

**Authors**: Claudio Fantinuoli  

**Link**: [PDF](https://arxiv.org/pdf/2509.23957)  

**Abstract**: Machine Interpreting systems are currently implemented as unimodal, real-time speech-to-speech architectures, processing translation exclusively on the basis of the linguistic signal. Such reliance on a single modality, however, constrains performance in contexts where disambiguation and adequacy depend on additional cues, such as visual, situational, or pragmatic information. This paper introduces Vision-Grounded Interpreting (VGI), a novel approach designed to address the limitations of unimodal machine interpreting. We present a prototype system that integrates a vision-language model to process both speech and visual input from a webcam, with the aim of priming the translation process through contextual visual information. To evaluate the effectiveness of this approach, we constructed a hand-crafted diagnostic corpus targeting three types of ambiguity. In our evaluation, visual grounding substantially improves lexical disambiguation, yields modest and less stable gains for gender resolution, and shows no benefit for syntactic ambiguities. We argue that embracing multimodality represents a necessary step forward for advancing translation quality in machine interpreting. 

---
# Explore-Execute Chain: Towards an Efficient Structured Reasoning Paradigm 

**Authors**: Kaisen Yang, Lixuan He, Rushi Shah, Kaicheng Yang, Qinwei Ma, Dianbo Liu, Alex Lamb  

**Link**: [PDF](https://arxiv.org/pdf/2509.23946)  

**Abstract**: Chain-of-Thought (CoT) and its variants have markedly advanced the reasoning abilities of Large Language Models (LLMs), yet their monolithic and auto-regressive architecture inherently conflates high-level strategic planning with low-level step-by-step execution, leading to computational inefficiency, limited exploration of reasoning paths, and reduced interpretability. To overcome these issues, we propose the Explore-Execute Chain ($E^2C$), a structured reasoning framework that decouples reasoning into two distinct phases: an exploratory phase that stochastically generates succinct high-level plans, followed by an execution phase that deterministically carries out the chosen plan. Our approach incorporates a two-stage training methodology, which combines Supervised Fine-Tuning (SFT) - augmented by a novel data generation algorithm enforcing strict plan adherence - with a subsequent Reinforcement Learning (RL) stage that capitalizes on the informativeness of exploration and reinforces the determinism of this http URL decomposition enables an efficient test-time scaling strategy: on AIME'2024, $E^2C$ Test Time Scaling reaches 58.1% accuracy using <10% of the decoding tokens required by comparable methods (e.g., Forest-of-Thought), sharply cutting self-consistency overhead. For cross-domain adaptation, our Exploration-Focused SFT (EF-SFT) fine-tunes with only 3.5% of the tokens used by standard SFT yet yields up to 14.5% higher accuracy than standard SFT on medical benchmarks, delivering state-of-the-art performance, strong generalization, and greater interpretability by separating planning from execution. The code and pre-trained models for the project are available at: this https URL 

---
# Easy Turn: Integrating Acoustic and Linguistic Modalities for Robust Turn-Taking in Full-Duplex Spoken Dialogue Systems 

**Authors**: Guojian Li, Chengyou Wang, Hongfei Xue, Shuiyuan Wang, Dehui Gao, Zihan Zhang, Yuke Lin, Wenjie Li, Longshuai Xiao, Zhonghua Fu, Lei Xie  

**Link**: [PDF](https://arxiv.org/pdf/2509.23938)  

**Abstract**: Full-duplex interaction is crucial for natural human-machine communication, yet remains challenging as it requires robust turn-taking detection to decide when the system should speak, listen, or remain silent. Existing solutions either rely on dedicated turn-taking models, most of which are not open-sourced. The few available ones are limited by their large parameter size or by supporting only a single modality, such as acoustic or linguistic. Alternatively, some approaches finetune LLM backbones to enable full-duplex capability, but this requires large amounts of full-duplex data, which remain scarce in open-source form. To address these issues, we propose Easy Turn, an open-source, modular turn-taking detection model that integrates acoustic and linguistic bimodal information to predict four dialogue turn states: complete, incomplete, backchannel, and wait, accompanied by the release of Easy Turn trainset, a 1,145-hour speech dataset designed for training turn-taking detection models. Compared to existing open-source models like TEN Turn Detection and Smart Turn V2, our model achieves state-of-the-art turn-taking detection accuracy on our open-source Easy Turn testset. The data and model will be made publicly available on GitHub. 

---
# Diffusion Models are Kelly Gamblers 

**Authors**: Akhil Premkumar  

**Link**: [PDF](https://arxiv.org/pdf/2509.23937)  

**Abstract**: We draw a connection between diffusion models and the Kelly criterion for maximizing returns in betting games. We find that conditional diffusion models store additional information to bind the signal $X$ with the conditioning information $Y$, equal to the mutual information between them. Classifier-free guidance effectively boosts the mutual information between $X$ and $Y$ at sampling time. This is especially helpful in image models, since the mutual information between images and their labels is low, a fact which is intimately connected to the manifold hypothesis. Finally, we point out some nuances in the popular perspective that diffusion models are infinitely deep autoencoders. In doing so, we relate the denoising loss to the Fermi Golden Rule from quantum mechanics. 

---
# HiViS: Hiding Visual Tokens from the Drafter for Speculative Decoding in Vision-Language Models 

**Authors**: Zhinan Xie, Peisong Wang, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23928)  

**Abstract**: Speculative decoding is an effective approach for accelerating inference in Large Language models (LLMs), but its adaptation to Vision-Language models (VLMs) remains challenging for additional visual tokens in multimodal inputs. First, owing to the fact that the drafter and the target VLM may derived from different families, the semantic representations of visual tokens in the target VLM are misaligned with those in the drafter, introducing bias into the KV-cache during the prefill stage. Second, the large number of visual tokens substantially slows down the drafter's self-attention during the decoding stage. We propose Hiding Visual Tokens from the Drafter for Speculative Decoding in Vision-Language Models (HiViS), an explicit-implicit input decomposition framework that alleviates the above inefficiency. All visual tokens are removed from the drafter's input, retaining only textual tokens as explicit inputs, while directly reusing the target VLM's corresponding last-layer hidden states as implicit visual information without additional processing. To train the drafter efficiently, we introduces multi-step self-feedback training strategy with dynamic data selection and sequential embedding supervision to simulate reasoning during training. Our approach compresses the prefill sequence length of the drafter to only 0.7%-1.3% of the target VLM's input, while maintaining lossless generation quality. Extensive experiments across diverse models and tasks demonstrate up to 2.65x speedup, confirming the effectiveness of HiViS in accelerating VLM inference. 

---
# Taming Masked Diffusion Language Models via Consistency Trajectory Reinforcement Learning with Fewer Decoding Step 

**Authors**: Jingyi Yang, Guanxu Chen, Xuhao Hu, Jing Shao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23924)  

**Abstract**: Masked diffusion language models (MDLMs) have recently emerged as a promising alternative to autoregressive (AR) language models, offering properties such as parallel decoding, flexible generation orders, and the potential for fewer inference steps. Despite these advantages, decoding strategies and reinforcement learning (RL) algorithms tailored for MDLMs remain underexplored. A naive approach is to directly transfer techniques well-established for AR models to MDLMs. However, this raises an immediate question: Is such a naive transfer truly optimal? For example, 1) Block-wise and semi-AR decoding strategies are not employed during the training of MDLMs, so why do they outperform full diffusion-style decoding during inference? 2) Applying RL algorithms designed for AR models directly to MDLMs exhibits a training-inference inconsistency, since MDLM decoding are non-causal (parallel). This results in inconsistencies between the rollout trajectory and the optimization trajectory. To address these challenges, we propose EOS Early Rejection (EOSER) and Ascending Step-Size (ASS) decoding scheduler, which unlock the potential of MDLMs to perform full diffusion-style decoding, achieving competitive performance with fewer decoding steps. Additionally, we introduce Consistency Trajectory Group Relative Policy Optimization (CJ-GRPO) for taming MDLMs, which emphasizes the consistency between rollout trajectory and optimization trajectory, and reduces the optimization errors caused by skip-step optimization. We conduct extensive experiments on reasoning tasks, such as mathematical and planning benchmarks, using LLaDA-8B-Instruct. The results demonstrate that the proposed EOSER and ASS mechanisms, together with CJ-GRPO, hold significant promise for effectively and efficiently taming MDLMs. Code: this https URL. 

---
# Graph Mixing Additive Networks 

**Authors**: Maya Bechler-Speicher, Andrea Zerio, Maor Huri, Marie Vibeke Vestergaard, Ran Gilad-Bachrach, Tine Jess, Samir Bhatt, Aleksejs Sazonovs  

**Link**: [PDF](https://arxiv.org/pdf/2509.23923)  

**Abstract**: We introduce GMAN, a flexible, interpretable, and expressive framework that extends Graph Neural Additive Networks (GNANs) to learn from sets of sparse time-series data. GMAN represents each time-dependent trajectory as a directed graph and applies an enriched, more expressive GNAN to each graph. It allows users to control the interpretability-expressivity trade-off by grouping features and graphs to encode priors, and it provides feature, node, and graph-level interpretability. On real-world datasets, including mortality prediction from blood tests and fake-news detection, GMAN outperforms strong non-interpretable black-box baselines while delivering actionable, domain-aligned explanations. 

---
# Continual Learning to Generalize Forwarding Strategies for Diverse Mobile Wireless Networks 

**Authors**: Cheonjin Park, Victoria Manfredi, Xiaolan Zhang, Chengyi Liu, Alicia P Wolfe, Dongjin Song, Sarah Tasneem, Bing Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23913)  

**Abstract**: Deep reinforcement learning (DRL) has been successfully used to design forwarding strategies for multi-hop mobile wireless networks. While such strategies can be used directly for networks with varied connectivity and dynamic conditions, developing generalizable approaches that are effective on scenarios significantly different from the training environment remains largely unexplored. In this paper, we propose a framework to address the challenge of generalizability by (i) developing a generalizable base model considering diverse mobile network scenarios, and (ii) using the generalizable base model for new scenarios, and when needed, fine-tuning the base model using a small amount of data from the new scenarios. To support this framework, we first design new features to characterize network variation and feature quality, thereby improving the information used in DRL-based forwarding decisions. We then develop a continual learning (CL) approach able to train DRL models across diverse network scenarios without ``catastrophic forgetting.'' Using extensive evaluation, including real-world scenarios in two cities, we show that our approach is generalizable to unseen mobility scenarios. Compared to a state-of-the-art heuristic forwarding strategy, it leads to up to 78% reduction in delay, 24% improvement in delivery rate, and comparable or slightly higher number of forwards. 

---
# EWC-Guided Diffusion Replay for Exemplar-Free Continual Learning in Medical Imaging 

**Authors**: Anoushka Harit, William Prew, Zhongtian Sun, Florian Markowetz  

**Link**: [PDF](https://arxiv.org/pdf/2509.23906)  

**Abstract**: Medical imaging foundation models must adapt over time, yet full retraining is often blocked by privacy constraints and cost. We present a continual learning framework that avoids storing patient exemplars by pairing class conditional diffusion replay with Elastic Weight Consolidation. Using a compact Vision Transformer backbone, we evaluate across eight MedMNIST v2 tasks and CheXpert. On CheXpert our approach attains 0.851 AUROC, reduces forgetting by more than 30\% relative to DER\texttt{++}, and approaches joint training at 0.869 AUROC, while remaining efficient and privacy preserving. Analyses connect forgetting to two measurable factors: fidelity of replay and Fisher weighted parameter drift, highlighting the complementary roles of replay diffusion and synaptic stability. The results indicate a practical route for scalable, privacy aware continual adaptation of clinical imaging models. 

---
# Interpreting deep learning-based stellar mass estimation via causal analysis and mutual information decomposition 

**Authors**: Wei Zhang, Qiufan Lin, Yuan-Sen Ting, Shupei Chen, Hengxin Ruan, Song Li, Yifan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23901)  

**Abstract**: End-to-end deep learning models fed with multi-band galaxy images are powerful data-driven tools used to estimate galaxy physical properties in the absence of spectroscopy. However, due to a lack of interpretability and the associational nature of such models, it is difficult to understand how the information additional to integrated photometry (e.g., morphology) contributes to the estimation task. Improving our understanding in this field would enable further advances into unraveling the physical connections among galaxy properties and optimizing data exploitation. Therefore, our work is aimed at interpreting the deep learning-based estimation of stellar mass via two interpretability techniques: causal analysis and mutual information decomposition. The former reveals the causal paths between multiple variables beyond nondirectional statistical associations, while the latter quantifies the multicomponent contributions (i.e., redundant, unique, and synergistic) of different input data to the stellar mass estimation. Using data from the Sloan Digital Sky Survey (SDSS) and the Wide-field Infrared Survey Explorer (WISE), we obtained meaningful results that provide physical interpretations for image-based models. Our work demonstrates the gains from combining deep learning with interpretability techniques, and holds promise in promoting more data-driven astrophysical research (e.g., astrophysical parameter estimations and investigations on complex multivariate physical processes). 

---
# Preserving Cross-Modal Stability for Visual Unlearning in Multimodal Scenarios 

**Authors**: Jinghan Xu Yuyang Zhang Qixuan Cai Jiancheng Chen Keqiu Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23895)  

**Abstract**: Visual modality is the most vulnerable to privacy leakage in real-world multimodal applications like autonomous driving with visual and radar data; Machine unlearning removes specific training data from pre-trained models to address privacy leakage, however, existing methods fail to preserve cross-modal knowledge and maintain intra-class structural stability of retain data, leading to reduced overall and other modalities' performance during visual unlearning; to address these challenges, we propose a Cross-modal Contrastive Unlearning (CCU) framework, which integrates three key components: (a) selective visual unlearning: employing inverse contrastive learning to dissociate visual representations from their original semantics, (b) cross-modal knowledge retention: preserving other modalities' discriminability through semantic consistency, and (c) dual-set contrastive separation: preserving the model performance via isolation of structural perturbations between the unlearn set and retain set; extensive experiments on three datasets demonstrate the superiority of CCU, and our method achieves a 7.12% accuracy improvement with only 7% of the unlearning time compared to the top-accuracy baseline. 

---
# Dynamic Orthogonal Continual Fine-tuning for Mitigating Catastrophic Forgettings 

**Authors**: Zhixin Zhang, Zeming Wei, Meng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.23893)  

**Abstract**: Catastrophic forgetting remains a critical challenge in continual learning for large language models (LLMs), where models struggle to retain performance on historical tasks when fine-tuning on new sequential data without access to past datasets. In this paper, we first reveal that the drift of functional directions during the fine-tuning process is a key reason why existing regularization-based methods fail in long-term LLM continual learning. To address this, we propose Dynamic Orthogonal Continual (DOC) fine-tuning, a novel approach that tracks the drift of these functional directions and dynamically updates them during the fine-tuning process. Furthermore, by adjusting the gradients of new task parameters to be orthogonal to the tracked historical function directions, our method mitigates interference between new and old tasks. Extensive experiments on various LLM continual learning benchmarks demonstrate that this approach outperforms prior methods, effectively reducing catastrophic forgetting and providing a robust tool for continuous LLM fine-tuning. Our code is available at this https URL. 

---
# Gradient Flow Convergence Guarantee for General Neural Network Architectures 

**Authors**: Yash Jakhmola  

**Link**: [PDF](https://arxiv.org/pdf/2509.23887)  

**Abstract**: A key challenge in modern deep learning theory is to explain the remarkable success of gradient-based optimization methods when training large-scale, complex deep neural networks. Though linear convergence of such methods has been proved for a handful of specific architectures, a united theory still evades researchers. This article presents a unified proof for linear convergence of continuous gradient descent, also called gradient flow, while training any neural network with piecewise non-zero polynomial activations or ReLU, sigmoid activations. Our primary contribution is a single, general theorem that not only covers architectures for which this result was previously unknown but also consolidates existing results under weaker assumptions. While our focus is theoretical and our results are only exact in the infinitesimal step size limit, we nevertheless find excellent empirical agreement between the predictions of our result and those of the practical step-size gradient descent method. 

---
# Towards Understanding Subliminal Learning: When and How Hidden Biases Transfer 

**Authors**: Simon Schrodi, Elias Kempf, Fazl Barez, Thomas Brox  

**Link**: [PDF](https://arxiv.org/pdf/2509.23886)  

**Abstract**: Language models can transfer hidden biases during distillation. For example, a teacher that "likes owls" can make its student "like owls" too, even when the training data consists only of lists of numbers. This surprising phenomenon is called subliminal learning. Subliminal learning can be expected under soft distillation, where the student is trained on the teacher's full next-token distribution. But the fact that this also occurs under hard distillation-where the student only sees sampled tokens-raises a deeper question: when and how does subliminal learning actually occur? We answer this question through controlled experiments and mechanistic analysis. Our results show that subliminal learning does not need (global) token entanglement or logit leakage. Instead, it comes down to a small set of divergence tokens-rare cases where teachers with different biases would predict different tokens. Masking out these tokens mostly removes the hidden bias transfer. Mechanistically, divergence tokens reveal that early layers are critical. Surprisingly, finetuning even a single such early layer is sufficient for subliminal learning. Finally, we find that subliminal learning is fragile. Even small changes, like paraphrasing prompts, are usually sufficient to suppress it. 

---
# Tunable-Generalization Diffusion Powered by Self-Supervised Contextual Sub-Data for Low-Dose CT Reconstruction 

**Authors**: Guoquan Wei, Zekun Zhou, Liu Shi, Wenzhe Shan, Qiegen Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23885)  

**Abstract**: Current models based on deep learning for low-dose CT denoising rely heavily on paired data and generalize poorly. Even the more concerned diffusion models need to learn the distribution of clean data for reconstruction, which is difficult to satisfy in medical clinical applications. At the same time, self-supervised-based methods face the challenge of significant degradation of generalizability of models pre-trained for the current dose to expand to other doses. To address these issues, this paper proposes a novel method of tunable-generalization diffusion powered by self-supervised contextual sub-data for low-dose CT reconstruction, named SuperDiff. Firstly, a contextual subdata similarity adaptive sensing strategy is designed for denoising centered on the LDCT projection domain, which provides an initial prior for the subsequent progress. Subsequently, the initial prior is used to combine knowledge distillation with a deep combination of latent diffusion models for optimizing image details. The pre-trained model is used for inference reconstruction, and the pixel-level self-correcting fusion technique is proposed for fine-grained reconstruction of the image domain to enhance the image fidelity, using the initial prior and the LDCT image as a guide. In addition, the technique is flexibly applied to the generalization of upper and lower doses or even unseen doses. Dual-domain strategy cascade for self-supervised LDCT denoising, SuperDiff requires only LDCT projection domain data for training and testing. Full qualitative and quantitative evaluations on both datasets and real data show that SuperDiff consistently outperforms existing state-of-the-art methods in terms of reconstruction and generalization performance. 

---
# PCRI: Measuring Context Robustness in Multimodal Models for Enterprise Applications 

**Authors**: Hitesh Laxmichand Patel, Amit Agarwal, Srikant Panda, Hansa Meghwani, Karan Dua, Paul Li, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2509.23879)  

**Abstract**: The reliability of Multimodal Large Language Models (MLLMs) in real-world settings is often undermined by sensitivity to irrelevant or distracting visual context, an aspect not captured by existing evaluation metrics. We introduce the \textbf{Patch Context Robustness Index (PCRI)}, the first systematic and interpretable score for quantifying MLLM robustness to variations in visual context granularity, measuring performance changes between localized image patches and full-image input.
Applying PCRI to 19 state-of-the-art MLLMs across 15 vision-language benchmarks, we find that most leading models remain brittle to background noise, with only a few, such as InternVL2-26B and Qwen2VL-72B, demonstrating consistent robustness across tasks. PCRI analysis also highlights how different model architectures handle and integrate visual context, offering actionable diagnostic insight for both researchers and practitioners.
PCRI enables rigorous comparison of context robustness, supporting principled model selection and guiding the development of future architectures and training strategies for robust, real-world deployment. 

---
# Disentangling Score Content and Performance Style for Joint Piano Rendering and Transcription 

**Authors**: Wei Zeng, Junchuan Zhao, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23878)  

**Abstract**: Expressive performance rendering (EPR) and automatic piano transcription (APT) are fundamental yet inverse tasks in music information retrieval: EPR generates expressive performances from symbolic scores, while APT recovers scores from performances. Despite their dual nature, prior work has addressed them independently. In this paper we propose a unified framework that jointly models EPR and APT by disentangling note-level score content and global performance style representations from both paired and unpaired data. Our framework is built on a transformer-based sequence-to-sequence architecture and is trained using only sequence-aligned data, without requiring fine-grained note-level alignment. To automate the rendering process while ensuring stylistic compatibility with the score, we introduce an independent diffusion-based performance style recommendation module that generates style embeddings directly from score content. This modular component supports both style transfer and flexible rendering across a range of expressive styles. Experimental results from both objective and subjective evaluations demonstrate that our framework achieves competitive performance on EPR and APT tasks, while enabling effective content-style disentanglement, reliable style transfer, and stylistically appropriate rendering. Demos are available at this https URL 

---
# Not All Tokens are Guided Equal: Improving Guidance in Visual Autoregressive Models 

**Authors**: Ky Dan Nguyen, Hoang Lam Tran, Anh-Dung Dinh, Daochang Liu, Weidong Cai, Xiuying Wang, Chang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23876)  

**Abstract**: Autoregressive (AR) models based on next-scale prediction are rapidly emerging as a powerful tool for image generation, but they face a critical weakness: information inconsistencies between patches across timesteps introduced by progressive resolution scaling. These inconsistencies scatter guidance signals, causing them to drift away from conditioning information and leaving behind ambiguous, unfaithful features. We tackle this challenge with Information-Grounding Guidance (IGG), a novel mechanism that anchors guidance to semantically important regions through attention. By adaptively reinforcing informative patches during sampling, IGG ensures that guidance and content remain tightly aligned. Across both class-conditioned and text-to-image generation tasks, IGG delivers sharper, more coherent, and semantically grounded images, setting a new benchmark for AR-based methods. 

---
# Multi-Value-Product Retrieval-Augmented Generation for Industrial Product Attribute Value Identification 

**Authors**: Huike Zou, Haiyang Yang, Yindu Su, Liyu Chen, Chengbao Lian, Qingheng Zhang, Shuguang Han, Jufeng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23874)  

**Abstract**: Identifying attribute values from product profiles is a key task for improving product search, recommendation, and business analytics on e-commerce platforms, which we called Product Attribute Value Identification (PAVI) . However, existing PAVI methods face critical challenges, such as cascading errors, inability to handle out-of-distribution (OOD) attribute values, and lack of generalization capability. To address these limitations, we introduce Multi-Value-Product Retrieval-Augmented Generation (MVP-RAG), combining the strengths of retrieval, generation, and classification paradigms. MVP-RAG defines PAVI as a retrieval-generation task, where the product title description serves as the query, and products and attribute values act as the corpus. It first retrieves similar products of the same category and candidate attribute values, and then generates the standardized attribute values. The key advantages of this work are: (1) the proposal of a multi-level retrieval scheme, with products and attribute values as distinct hierarchical levels in PAVI domain (2) attribute value generation of large language model to significantly alleviate the OOD problem and (3) its successful deployment in a real-world industrial environment. Extensive experimental results demonstrate that MVP-RAG performs better than the state-of-the-art baselines. 

---
# Taught Well Learned Ill: Towards Distillation-conditional Backdoor Attack 

**Authors**: Yukun Chen, Boheng Li, Yu Yuan, Leyi Qi, Yiming Li, Tianwei Zhang, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.23871)  

**Abstract**: Knowledge distillation (KD) is a vital technique for deploying deep neural networks (DNNs) on resource-constrained devices by transferring knowledge from large teacher models to lightweight student models. While teacher models from third-party platforms may undergo security verification (\eg, backdoor detection), we uncover a novel and critical threat: distillation-conditional backdoor attacks (DCBAs). DCBA injects dormant and undetectable backdoors into teacher models, which become activated in student models via the KD process, even with clean distillation datasets. While the direct extension of existing methods is ineffective for DCBA, we implement this attack by formulating it as a bilevel optimization problem and proposing a simple yet effective method (\ie, SCAR). Specifically, the inner optimization simulates the KD process by optimizing a surrogate student model, while the outer optimization leverages outputs from this surrogate to optimize the teacher model for implanting the conditional backdoor. Our SCAR addresses this complex optimization utilizing an implicit differentiation algorithm with a pre-optimized trigger injection function. Extensive experiments across diverse datasets, model architectures, and KD techniques validate the effectiveness of our SCAR and its resistance against existing backdoor detection, highlighting a significant yet previously overlooked vulnerability in the KD process. Our code is available at this https URL. 

---
# Efficient Multi-turn RL for GUI Agents via Decoupled Training and Adaptive Data Curation 

**Authors**: Pengxiang Li, Zechen Hu, Zirui Shang, Jingrong Wu, Yang Liu, Hui Liu, Zhi Gao, Chenrui Shi, Bofei Zhang, Zihao Zhang, Xiaochuan Shi, Zedong YU, Yuwei Wu, Xinxiao Wu, Yunde Jia, Liuyu Xiang, Zhaofeng He, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23866)  

**Abstract**: Vision-language model (VLM) based GUI agents show promise for automating complex desktop and mobile tasks, but face significant challenges in applying reinforcement learning (RL): (1) slow multi-turn interactions with GUI environments for policy rollout, and (2) insufficient high-quality agent-environment interactions for policy learning. To address these challenges, we propose DART, a Decoupled Agentic RL Training framework for GUI agents, which coordinates heterogeneous modules in a highly decoupled manner. DART separates the training system into four asynchronous modules: environment cluster, rollout service, data manager, and trainer. This design enables non-blocking communication, asynchronous training, rollout-wise trajectory sampling, and per-worker model synchronization, significantly improving the system efficiency: 1.6*GPU utilization for rollout, 1.9* training throughput, and 5.5* environment utilization. To facilitate effective learning from abundant samples, we introduce an adaptive data curation scheme: (1) pre-collecting successful trajectories for challenging tasks to supplement sparse success in online sampling; (2) dynamically adjusting rollout numbers and trajectory lengths based on task difficulty; (3) training selectively on high-entropy steps to prioritize critical decisions; (4) stabilizing learning via truncated importance sampling for policy mismatch between policy rollout and updating. On the OSWorld benchmark, DART-GUI-7B achieves a 42.13% task success rate, a 14.61% absolute gain over the base model, and 7.34% higher than open-source SOTA. We will fully open-source our training framework, data, and model checkpoints via this http URL, which we believe is a timely contribution to the open-source community of agentic RL training. 

---
# GSID: Generative Semantic Indexing for E-Commerce Product Understanding 

**Authors**: Haiyang Yang, Qinye Xie, Qingheng Zhang, Liyu Chen, Huike Zou, Chengbao Lian, Shuguang Han, Fei Huang, Jufeng Chen, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23860)  

**Abstract**: Structured representation of product information is a major bottleneck for the efficiency of e-commerce platforms, especially in second-hand ecommerce platforms. Currently, most product information are organized based on manually curated product categories and attributes, which often fail to adequately cover long-tail products and do not align well with buyer preference. To address these problems, we propose \textbf{G}enerative \textbf{S}emantic \textbf{I}n\textbf{D}exings (GSID), a data-driven approach to generate product structured representations. GSID consists of two key components: (1) Pre-training on unstructured product metadata to learn in-domain semantic embeddings, and (2) Generating more effective semantic codes tailored for downstream product-centric applications. Extensive experiments are conducted to validate the effectiveness of GSID, and it has been successfully deployed on the real-world e-commerce platform, achieving promising results on product understanding and other downstream tasks. 

---
# Adversarial Diffusion for Robust Reinforcement Learning 

**Authors**: Daniele Foffano, Alessio Russo, Alexandre Proutiere  

**Link**: [PDF](https://arxiv.org/pdf/2509.23846)  

**Abstract**: Robustness to modeling errors and uncertainties remains a central challenge in reinforcement learning (RL). In this work, we address this challenge by leveraging diffusion models to train robust RL policies. Diffusion models have recently gained popularity in model-based RL due to their ability to generate full trajectories "all at once", mitigating the compounding errors typical of step-by-step transition models. Moreover, they can be conditioned to sample from specific distributions, making them highly flexible. We leverage conditional sampling to learn policies that are robust to uncertainty in environment dynamics. Building on the established connection between Conditional Value at Risk (CVaR) optimization and robust RL, we introduce Adversarial Diffusion for Robust Reinforcement Learning (AD-RRL). AD-RRL guides the diffusion process to generate worst-case trajectories during training, effectively optimizing the CVaR of the cumulative return. Empirical results across standard benchmarks show that AD-RRL achieves superior robustness and performance compared to existing robust RL methods. 

---
# HFuzzer: Testing Large Language Models for Package Hallucinations via Phrase-based Fuzzing 

**Authors**: Yukai Zhao, Menghan Wu, Xing Hu, Xin Xia  

**Link**: [PDF](https://arxiv.org/pdf/2509.23835)  

**Abstract**: Large Language Models (LLMs) are widely used for code generation, but they face critical security risks when applied to practical production due to package hallucinations, in which LLMs recommend non-existent packages. These hallucinations can be exploited in software supply chain attacks, where malicious attackers exploit them to register harmful packages. It is critical to test LLMs for package hallucinations to mitigate package hallucinations and defend against potential attacks. Although researchers have proposed testing frameworks for fact-conflicting hallucinations in natural language generation, there is a lack of research on package hallucinations. To fill this gap, we propose HFUZZER, a novel phrase-based fuzzing framework to test LLMs for package hallucinations. HFUZZER adopts fuzzing technology and guides the model to infer a wider range of reasonable information based on phrases, thereby generating enough and diverse coding tasks. Furthermore, HFUZZER extracts phrases from package information or coding tasks to ensure the relevance of phrases and code, thereby improving the relevance of generated tasks and code. We evaluate HFUZZER on multiple LLMs and find that it triggers package hallucinations across all selected models. Compared to the mutational fuzzing framework, HFUZZER identifies 2.60x more unique hallucinated packages and generates more diverse tasks. Additionally, when testing the model GPT-4o, HFUZZER finds 46 unique hallucinated packages. Further analysis reveals that for GPT-4o, LLMs exhibit package hallucinations not only during code generation but also when assisting with environment configuration. 

---
# Space Group Conditional Flow Matching 

**Authors**: Omri Puny, Yaron Lipman, Benjamin Kurt Miller  

**Link**: [PDF](https://arxiv.org/pdf/2509.23822)  

**Abstract**: Inorganic crystals are periodic, highly-symmetric arrangements of atoms in three-dimensional space. Their structures are constrained by the symmetry operations of a crystallographic \emph{space group} and restricted to lie in specific affine subspaces known as \emph{Wyckoff positions}. The frequency an atom appears in the crystal and its rough positioning are determined by its Wyckoff position. Most generative models that predict atomic coordinates overlook these symmetry constraints, leading to unrealistically high populations of proposed crystals exhibiting limited symmetry. We introduce Space Group Conditional Flow Matching, a novel generative framework that samples significantly closer to the target population of highly-symmetric, stable crystals. We achieve this by conditioning the entire generation process on a given space group and set of Wyckoff positions; specifically, we define a conditionally symmetric noise base distribution and a group-conditioned, equivariant, parametric vector field that restricts the motion of atoms to their initial Wyckoff position. Our form of group-conditioned equivariance is achieved using an efficient reformulation of \emph{group averaging} tailored for symmetric crystals. Importantly, it reduces the computational overhead of symmetrization to a negligible level. We achieve state of the art results on crystal structure prediction and de novo generation benchmarks. We also perform relevant ablations. 

---
# A Multi-Camera Vision-Based Approach for Fine-Grained Assembly Quality Control 

**Authors**: Ali Nazeri, Shashank Mishra, Achim Wagner, Martin Ruskowski, Didier Stricker, Jason Rambach  

**Link**: [PDF](https://arxiv.org/pdf/2509.23815)  

**Abstract**: Quality control is a critical aspect of manufacturing, particularly in ensuring the proper assembly of small components in production lines. Existing solutions often rely on single-view imaging or manual inspection, which are prone to errors due to occlusions, restricted perspectives, or lighting inconsistencies. These limitations require the installation of additional inspection stations, which could disrupt the assembly line and lead to increased downtime and costs. This paper introduces a novel multi-view quality control module designed to address these challenges, integrating a multi-camera imaging system with advanced object detection algorithms. By capturing images from three camera views, the system provides comprehensive visual coverage of components of an assembly process. A tailored image fusion methodology combines results from multiple views, effectively resolving ambiguities and enhancing detection reliability. To support this system, we developed a unique dataset comprising annotated images across diverse scenarios, including varied lighting conditions, occlusions, and angles, to enhance applicability in real-world manufacturing environments. Experimental results show that our approach significantly outperforms single-view methods, achieving high precision and recall rates in the identification of improperly fastened small assembly parts such as screws. This work contributes to industrial automation by overcoming single-view limitations, and providing a scalable, cost-effective, and accurate quality control mechanism that ensures the reliability and safety of the assembly line. The dataset used in this study is publicly available to facilitate further research in this domain. 

---
# IndexNet: Timestamp and Variable-Aware Modeling for Time Series Forecasting 

**Authors**: Beiliang Wu, Peiyuan Liu, Yifan Hu, Luyan Zhang, Ao Hu, Zenglin Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23813)  

**Abstract**: Multivariate time series forecasting (MTSF) plays a vital role in a wide range of real-world applications, such as weather prediction and traffic flow forecasting. Although recent advances have significantly improved the modeling of temporal dynamics and inter-variable dependencies, most existing methods overlook index-related descriptive information, such as timestamps and variable indices, which carry rich contextual semantics. To unlock the potential of such information and take advantage of the lightweight and powerful periodic capture ability of MLP-based architectures, we propose IndexNet, an MLP-based framework augmented with an Index Embedding (IE) module. The IE module consists of two key components: Timestamp Embedding (TE) and Channel Embedding (CE). Specifically, TE transforms timestamps into embedding vectors and injects them into the input sequence, thereby improving the model's ability to capture long-term complex periodic patterns. In parallel, CE assigns each variable a unique and trainable identity embedding based on its index, allowing the model to explicitly distinguish between heterogeneous variables and avoid homogenized predictions when input sequences seem close. Extensive experiments on 12 diverse real-world datasets demonstrate that IndexNet achieves comparable performance across mainstream baselines, validating the effectiveness of our temporally and variably aware design. Moreover, plug-and-play experiments and visualization analyses further reveal that IndexNet exhibits strong generality and interpretability, two aspects that remain underexplored in current MTSF research. 

---
# Navigating the Labyrinth: Path-Sensitive Unit Test Generation with Large Language Models 

**Authors**: Dianshu Liao, Xin Yin, Shidong Pan, Chao Ni, Zhenchang Xing, Xiaoyu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.23812)  

**Abstract**: Unit testing is essential for software quality assurance, yet writing and maintaining tests remains time-consuming and error-prone. To address this challenge, researchers have proposed various techniques for automating unit test generation, including traditional heuristic-based methods and more recent approaches that leverage large language models (LLMs). However, these existing approaches are inherently path-insensitive because they rely on fixed heuristics or limited contextual information and fail to reason about deep control-flow structures. As a result, they often struggle to achieve adequate coverage, particularly for deep or complex execution paths. In this work, we present a path-sensitive framework, JUnitGenie, to fill this gap by combining code knowledge with the semantic capabilities of LLMs in guiding context-aware unit test generation. After extracting code knowledge from Java projects, JUnitGenie distills this knowledge into structured prompts to guide the generation of high-coverage unit tests. We evaluate JUnitGenie on 2,258 complex focal methods from ten real-world Java projects. The results show that JUnitGenie generates valid tests and improves branch and line coverage by 29.60% and 31.00% on average over both heuristic and LLM-based baselines. We further demonstrate that the generated test cases can uncover real-world bugs, which were later confirmed and fixed by developers. 

---
# Tequila: Trapping-free Ternary Quantization for Large Language Models 

**Authors**: Hong Huang, Decheng Wu, Rui Cen, Guanghua Yu, Zonghang Li, Kai Liu, Jianchen Zhu, Peng Chen, Xue Liu, Dapeng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23809)  

**Abstract**: Quantization techniques are essential for the deployment of Large Language Models (LLMs) on edge devices. However, prevailing methods often rely on mixed-precision multiplication that lacks efficient hardware support, making it not feasible. Ternary weight quantization addresses this by constraining weights to {-1, 0, 1}, replacing expensive multiplications with hardware-efficient additions. However, such aggressive compression leads to significant accuracy degradation, even after costly quantization-aware training with massive data. We identify the core issue as deadzone trapping: a large number of weights are trapped at the deadzone boundary. This occurs because these weights receive only noisy, uninformative gradients, preventing stable escape from the deadzone and severely impeding model capacity and optimization. To address this issue, we propose Tequila, a trapping-free quantization optimization method that reactivates deadzone-trapped weights by repurposing them as dynamic biases. This allows the repurposed weights to provide a continuous signal in the forward pass and, critically, receive direct, meaningful gradient signals during backpropagation, thereby enhancing model capacity and optimization with nearly zero inference overhead. Extensive evaluations demonstrate that Tequila outperforms state-of-the-art (SOTA) ternary quantization methods across five benchmarks. Specifically, on the ARC benchmark, it achieves >4% accuracy gain over the SOTA baseline, nearly matching full-precision performance (within <1% gap) with a 3.0x inference speedup. Consequently, Tequila offers a highly practical and efficient implementation for the deployment of advanced LLMs in resource-constrained environments. The code is available at this https URL. 

---
# FedAgentBench: Towards Automating Real-world Federated Medical Image Analysis with Server-Client LLM Agents 

**Authors**: Pramit Saha, Joshua Strong, Divyanshu Mishra, Cheng Ouyang, J.Alison Noble  

**Link**: [PDF](https://arxiv.org/pdf/2509.23803)  

**Abstract**: Federated learning (FL) allows collaborative model training across healthcare sites without sharing sensitive patient data. However, real-world FL deployment is often hindered by complex operational challenges that demand substantial human efforts. This includes: (a) selecting appropriate clients (hospitals), (b) coordinating between the central server and clients, (c) client-level data pre-processing, (d) harmonizing non-standardized data and labels across clients, and (e) selecting FL algorithms based on user instructions and cross-client data characteristics. However, the existing FL works overlook these practical orchestration challenges. These operational bottlenecks motivate the need for autonomous, agent-driven FL systems, where intelligent agents at each hospital client and the central server agent collaboratively manage FL setup and model training with minimal human intervention. To this end, we first introduce an agent-driven FL framework that captures key phases of real-world FL workflows from client selection to training completion and a benchmark dubbed FedAgentBench that evaluates the ability of LLM agents to autonomously coordinate healthcare FL. Our framework incorporates 40 FL algorithms, each tailored to address diverse task-specific requirements and cross-client characteristics. Furthermore, we introduce a diverse set of complex tasks across 201 carefully curated datasets, simulating 6 modality-specific real-world healthcare environments, viz., Dermatoscopy, Ultrasound, Fundus, Histopathology, MRI, and X-Ray. We assess the agentic performance of 14 open-source and 10 proprietary LLMs spanning small, medium, and large model scales. While some agent cores such as GPT-4.1 and DeepSeek V3 can automate various stages of the FL pipeline, our results reveal that more complex, interdependent tasks based on implicit goals remain challenging for even the strongest models. 

---
# Enhancing LLM Steering through Sparse Autoencoder-Based Vector Refinement 

**Authors**: Anyi Wang, Xuansheng Wu, Dong Shu, Yunpu Ma, Ninghao Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23799)  

**Abstract**: Steering has emerged as a promising approach in controlling large language models (LLMs) without modifying model parameters. However, most existing steering methods rely on large-scale datasets to learn clear behavioral information, which limits their applicability in many real-world scenarios. The steering vectors extracted from small dataset often contain task-irrelevant noising features, which degrades their effectiveness. To refine the steering vectors learned from limited data, we introduce Refinement of Steering Vector via Sparse Autoencoder (SAE-RSV) that leverages SAEs to semantically denoise and augment the steering vectors. In our framework, we first remove task-irrelevant features according to their semantics provided by SAEs, and then enrich task-relevant features missing from the small dataset through their semantic similarity to the identified relevant features. Extensive experiments demonstrate that the proposed SAE-RSV substantially outperforms all the baseline methods including supervised fine-tuning. Our findings show that effective steering vector can be constructed from limited training data by refining the original steering vector through SAEs. 

---
# From Unstable to Playable: Stabilizing Angry Birds Levels via Object Segmentation 

**Authors**: Mahdi Farrokhimaleki, Parsa Rahmati, Richard Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23787)  

**Abstract**: Procedural Content Generation (PCG) techniques enable automatic creation of diverse and complex environments. While PCG facilitates more efficient content creation, ensuring consistently high-quality, industry-standard content remains a significant challenge. In this research, we propose a method to identify and repair unstable levels generated by existing PCG models. We use Angry Birds as a case study, demonstrating our method on game levels produced by established PCG approaches. Our method leverages object segmentation and visual analysis of level images to detect structural gaps and perform targeted repairs. We evaluate multiple object segmentation models and select the most effective one as the basis for our repair pipeline. Experimental results show that our method improves the stability and playability of AI-generated levels. Although our evaluation is specific to Angry Birds, our image-based approach is designed to be applicable to a wide range of 2D games with similar level structures. 

---
# GroupCoOp: Group-robust Fine-tuning via Group Prompt Learning 

**Authors**: Nayeong Kim, Seong Joon Oh, Suha Kwak  

**Link**: [PDF](https://arxiv.org/pdf/2509.23781)  

**Abstract**: Parameter-efficient fine-tuning (PEFT) of vision-language models (VLMs) excels in various vision tasks thanks to the rich knowledge and generalization ability of VLMs. However, recent studies revealed that such fine-tuned VLMs are vulnerable to spurious correlations stemming from the subgroup imbalance in the fine-tuning datasets. To resolve this issue, we propose Group Context Optimization (GroupCoOp), a simple and effective debiased fine-tuning algorithm that enhances the group robustness of fine-tuned VLMs. Its key idea is to employ group-specific text prompts as group representatives serving as multiple classifiers for their target class. The rich semantic knowledge of the text encoder of VLM enables the discovery of effective group prompts even for groups with a small number of training samples. Leveraging the group prompts for each class addresses the issues caused by the group-imbalanced training set, such as the neglect of minority groups and the scattered distribution of each class in the embedding space. GroupCoOp achieved the best results on five benchmarks across five CLIP architectures and occasionally outperformed prior methods that fine-tune the entire network, despite training only 0.016\% of the network's parameters. 

---
# Sequence Pathfinder for Multi-Agent Pickup and Delivery in the Warehouse 

**Authors**: Zeyuan Zhang, Chaoran Li, Shao Zhang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23778)  

**Abstract**: Multi-Agent Pickup and Delivery (MAPD) is a challenging extension of Multi-Agent Path Finding (MAPF), where agents are required to sequentially complete tasks with fixed-location pickup and delivery demands. Although learning-based methods have made progress in MAPD, they often perform poorly in warehouse-like environments with narrow pathways and long corridors when relying only on local observations for distributed decision-making. Communication learning can alleviate the lack of global information but introduce high computational complexity due to point-to-point communication. To address this challenge, we formulate MAPF as a sequence modeling problem and prove that path-finding policies under sequence modeling possess order-invariant optimality, ensuring its effectiveness in MAPD. Building on this, we propose the Sequential Pathfinder (SePar), which leverages the Transformer paradigm to achieve implicit information exchange, reducing decision-making complexity from exponential to linear while maintaining efficiency and global awareness. Experiments demonstrate that SePar consistently outperforms existing learning-based methods across various MAPF tasks and their variants, and generalizes well to unseen environments. Furthermore, we highlight the necessity of integrating imitation learning in complex maps like warehouses. 

---
# Knowledge Homophily in Large Language Models 

**Authors**: Utkarsh Sahu, Zhisheng Qi, Mahantesh Halappanavar, Nedim Lipka, Ryan A. Rossi, Franck Dernoncourt, Yu Zhang, Yao Ma, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23773)  

**Abstract**: Large Language Models (LLMs) have been increasingly studied as neural knowledge bases for supporting knowledge-intensive applications such as question answering and fact checking. However, the structural organization of their knowledge remains unexplored. Inspired by cognitive neuroscience findings, such as semantic clustering and priming, where knowing one fact increases the likelihood of recalling related facts, we investigate an analogous knowledge homophily pattern in LLMs. To this end, we map LLM knowledge into a graph representation through knowledge checking at both the triplet and entity levels. After that, we analyze the knowledgeability relationship between an entity and its neighbors, discovering that LLMs tend to possess a similar level of knowledge about entities positioned closer in the graph. Motivated by this homophily principle, we propose a Graph Neural Network (GNN) regression model to estimate entity-level knowledgeability scores for triplets by leveraging their neighborhood scores. The predicted knowledgeability enables us to prioritize checking less well-known triplets, thereby maximizing knowledge coverage under the same labeling budget. This not only improves the efficiency of active labeling for fine-tuning to inject knowledge into LLMs but also enhances multi-hop path retrieval in reasoning-intensive question answering. 

---
# From Personal to Collective: On the Role of Local and Global Memory in LLM Personalization 

**Authors**: Zehong Wang, Junlin Wu, ZHaoxuan Tan, Bolian Li, Xianrui Zhong, Zheli Liu, Qingkai Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23767)  

**Abstract**: Large language model (LLM) personalization aims to tailor model behavior to individual users based on their historical interactions. However, its effectiveness is often hindered by two key challenges: the \textit{cold-start problem}, where users with limited history provide insufficient context for accurate personalization, and the \textit{biasing problem}, where users with abundant but skewed history cause the model to overfit to narrow preferences. We identify both issues as symptoms of a common underlying limitation, i.e., the inability to model collective knowledge across users. To address this, we propose a local-global memory framework (LoGo) that combines the personalized local memory with a collective global memory that captures shared interests across the population. To reconcile discrepancies between these two memory sources, we introduce a mediator module designed to resolve conflicts between local and global signals. Extensive experiments on multiple benchmarks demonstrate that LoGo consistently improves personalization quality by both warming up cold-start users and mitigating biased predictions. These results highlight the importance of incorporating collective knowledge to enhance LLM personalization. 

---
# Knowledge-Level Consistency Reinforcement Learning: Dual-Fact Alignment for Long-Form Factuality 

**Authors**: Junliang Li, Yucheng Wang, Yan Chen, Yu Ran, Ruiqing Zhang, Jing Liu, Hua Wu, Haifeng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23765)  

**Abstract**: Hallucination and factuality deficits remain key obstacles to the reliability of large language models (LLMs) in long-form generation. Existing reinforcement learning from human feedback (RLHF) frameworks primarily rely on preference rewards, yet they often overlook the model's internal knowledge boundaries, exacerbating the so-called "hallucination tax". To address this challenge, we propose Knowledge-Level Consistency Reinforcement Learning Framework (KLCF), a novel framework that focuses on the knowledge consistency between the policy model's expressed knowledge and the base model's parametric knowledge, and introduces a Dual-Fact Alignment mechanism to jointly optimize factual recall and precision. Specifically, KLCF leverages pretrained knowledge boundaries to construct fact checklist, guiding online reinforcement learning to improve factual coverage and recall; simultaneously, it trains a self-assessment module based on the base model's internal knowledge to enhance factual precision during generation. Unlike prior methods that rely on external retrieval or heavy verification, our reward design is fully external-knowledge-free and lightweight, making KLCF efficient and easily scalable to large-scale training. Experimental results demonstrate that KLCF substantially improves factuality metrics across multiple long-form benchmarks and effectively alleviates model hallucinations. 

---
# Accuracy-Robustness Trade Off via Spiking Neural Network Gradient Sparsity Trail 

**Authors**: Nhan T. Luu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23762)  

**Abstract**: Spiking Neural Networks (SNNs) have attracted growing interest in both computational neuroscience and artificial intelligence, primarily due to their inherent energy efficiency and compact memory footprint. However, achieving adversarial robustness in SNNs, particularly for vision-related tasks, remains a nascent and underexplored challenge. Recent studies have proposed leveraging sparse gradients as a form of regularization to enhance robustness against adversarial perturbations. In this work, we present a surprising finding: under specific architectural configurations, SNNs exhibit natural gradient sparsity and can achieve state-of-the-art adversarial defense performance without the need for any explicit regularization. Further analysis reveals a trade-off between robustness and generalization: while sparse gradients contribute to improved adversarial resilience, they can impair the model's ability to generalize; conversely, denser gradients support better generalization but increase vulnerability to attacks. 

---
# SHAPoint: Task-Agnostic, Efficient, and Interpretable Point-Based Risk Scoring via Shapley Values 

**Authors**: Tomer D. Meirman, Bracha Shapira, Noa Dagan, Lior S. Rokach  

**Link**: [PDF](https://arxiv.org/pdf/2509.23756)  

**Abstract**: Interpretable risk scores play a vital role in clinical decision support, yet traditional methods for deriving such scores often rely on manual preprocessing, task-specific modeling, and simplified assumptions that limit their flexibility and predictive power. We present SHAPoint, a novel, task-agnostic framework that integrates the predictive accuracy of gradient boosted trees with the interpretability of point-based risk scores. SHAPoint supports classification, regression, and survival tasks, while also inheriting valuable properties from tree-based models, such as native handling of missing data and support for monotonic constraints. Compared to existing frameworks, SHAPoint offers superior flexibility, reduced reliance on manual preprocessing, and faster runtime performance. Empirical results show that SHAPoint produces compact and interpretable scores with predictive performance comparable to state-of-the-art methods, but at a fraction of the runtime, making it a powerful tool for transparent and scalable risk stratification. 

---
# Understanding Textual Capability Degradation in Speech LLMs via Parameter Importance Analysis 

**Authors**: Chao Wang, Rui-Chen Zheng, Yang Ai, Zhen-Hua Ling  

**Link**: [PDF](https://arxiv.org/pdf/2509.23755)  

**Abstract**: The integration of speech into Large Language Models (LLMs) has substantially expanded their capabilities, but often at the cost of weakening their core textual competence. This degradation limits the ability of speech-enabled LLMs to fully exploit their pre-trained text-based knowledge. In this work, we analyze the underlying mechanisms of this issue through a focused study of the widely used encoder-adaptor paradigm. We propose an analytical framework based on parameter importance estimation, which reveals that fine-tuning for speech introduces a textual importance distribution shift: the layer-wise allocation of parameters critical to textual reasoning is disrupted. Building on this insight, we investigate two mitigation strategies: layer-wise learning rate scheduling and Low-Rank Adaptation (LoRA), both aim to preserve the original parameter distribution. Experimental results show that both approaches better maintain textual competence than full fine-tuning, while also improving downstream spoken question answering performance. Furthermore, our analysis offers a principled explanation for the effectiveness of the proposed mitigation strategies, linking their benefits to the structural properties of textual knowledge in LLMs. 

---
# PVTAdpNet: Polyp Segmentation using Pyramid vision transformer with a novel Adapter block 

**Authors**: Arshia Yousefi Nezhad, Helia Aghaei, Hedieh Sajedi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23751)  

**Abstract**: Colorectal cancer ranks among the most common and deadly cancers, emphasizing the need for effective early detection and treatment. To address the limitations of traditional colonoscopy, including high miss rates due to polyp variability, we introduce the Pyramid Vision Transformer Adapter Residual Network (PVTAdpNet). This model integrates a U-Net-style encoder-decoder structure with a Pyramid Vision Transformer backbone, novel residual blocks, and adapter-based skip connections. The design enhances feature extraction, dense prediction, and gradient flow, supported by squeeze-and-excitation attention for improved channel-wise feature refinement. PVTAdpNet achieves real-time, accurate polyp segmentation, demonstrating superior performance on benchmark datasets with high mDice and mIoU scores, making it highly suitable for clinical applications. PVTAdpNet obtains a high Dice coefficient of 0.8851 and a mean Intersection over Union (mIoU) of 0.8167 on out-of-distribution polyp datasets. Evaluation of the PolypGen dataset demonstrates PVTAdpNet's capability for real-time, accurate performance within familiar distributions. The source code of our network is available at this https URL 

---
# Poivre: Self-Refining Visual Pointing with Reinforcement Learning 

**Authors**: Wenjie Yang, Zengfeng Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23746)  

**Abstract**: Visual pointing, which aims to localize a target by predicting its coordinates on an image, has emerged as an important problem in the realm of vision-language models (VLMs). Despite its broad applicability, recent benchmarks show that current VLMs still fall far behind human performance on this task. A key limitation is that VLMs are typically required to complete the pointing task in a single step, akin to asking humans to point at an object without seeing their own fingers. To address this issue, we propose a simple yet effective self-refining procedure: Point, Visualize, then Refine (Poivre). This procedure enables a VLM to first mark its estimated point, then iteratively refine the coordinates if necessary. Inspired by advances of reasoning models in the natural language domain, we employ reinforcement learning (RL) to incentivize this self-refining ability. For the RL training, we design a neat process reward that is not only empirically effective but also grounded in appealing properties. Our trained model, Poivre-7B, sets a new state of the art on Point-Bench, outperforming both proprietary models such as Gemini-2.5-Pro and large open-source models such as Molmo-72B by over 3%. To support future research, we release our training and inference code, dataset, and the Poivre-7B checkpoint. 

---
# LocoFormer: Generalist Locomotion via Long-context Adaptation 

**Authors**: Min Liu, Deepak Pathak, Ananye Agarwal  

**Link**: [PDF](https://arxiv.org/pdf/2509.23745)  

**Abstract**: Modern locomotion controllers are manually tuned for specific embodiments. We present LocoFormer, a generalist omni-bodied locomotion model that can control previously unseen legged and wheeled robots, even without precise knowledge of their kinematics. LocoFormer is able to adapt to changes in morphology and dynamics at test time. We find that two key choices enable adaptation. First, we train massive scale RL on procedurally generated robots with aggressive domain randomization. Second, in contrast to previous policies that are myopic with short context lengths, we extend context by orders of magnitude to span episode boundaries. We deploy the same LocoFormer to varied robots and show robust control even with large disturbances such as weight change and motor failures. In extreme scenarios, we see emergent adaptation across episodes, LocoFormer learns from falls in early episodes to improve control strategies in later ones. We believe that this simple, yet general recipe can be used to train foundation models for other robotic skills in the future. Videos at this http URL. 

---
# Compose and Fuse: Revisiting the Foundational Bottlenecks in Multimodal Reasoning 

**Authors**: Yucheng Wang, Yifan Hou, Aydin Javadov, Mubashara Akhtar, Mrinmaya Sachan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23744)  

**Abstract**: Multimodal large language models (MLLMs) promise enhanced reasoning by integrating diverse inputs such as text, vision, and audio. Yet cross-modal reasoning remains underexplored, with conflicting reports on whether added modalities help or harm performance. These inconsistencies stem from a lack of controlled evaluation frameworks and analysis of models' internals to isolate when and why modality interactions support or undermine reasoning. We address this gap through a logic-grounded evaluation framework that categorizes multimodal reasoning into six interaction patterns, varying how facts are distributed across modalities and logically combined. Empirically, additional modalities enhance reasoning only when they provide independent and sufficient reasoning paths, while redundant or chained entailment support often hurts performance. Moreover, reasoning degrades in three systematic ways: weaker modalities drag down overall performance, conflicts bias preference toward certain modalities, and joint signals from different modalities fail to be integrated effectively. Therefore, we identify two core failures: task-composition bottleneck, where recognition and reasoning cannot be jointly executed in one pass, and fusion bottleneck, where early integration introduces bias. For further investigation, we find that attention patterns fail to encode fact usefulness, but a simple two-step prompting (recognize then reason) restores performance, confirming the task-composition bottleneck. Moreover, modality identity remains recoverable in early layers, and softening attention in early fusion improves reasoning, highlighting biased fusion as another failure mode. Overall, our findings show that integration, not perception, is the main barrier to multimodal reasoning, suggesting composition-aware training and early fusion control as promising directions. 

---
# HieraTok: Multi-Scale Visual Tokenizer Improves Image Reconstruction and Generation 

**Authors**: Cong Chen, Ziyuan Huang, Cheng Zou, Muzhi Zhu, Kaixiang Ji, Jiajia Liu, Jingdong Chen, Hao Chen, Chunhua Shen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23736)  

**Abstract**: In this work, we present HieraTok, a novel multi-scale Vision Transformer (ViT)-based tokenizer that overcomes the inherent limitation of modeling single-scale representations. This is realized through two key designs: (1) multi-scale downsampling applied to the token map generated by the tokenizer encoder, producing a sequence of multi-scale tokens, and (2) a scale-causal attention mechanism that enables the progressive flow of information from low-resolution global semantic features to high-resolution structural details. Coupling these designs, HieraTok achieves significant improvements in both image reconstruction and generation tasks. Under identical settings, the multi-scale visual tokenizer outperforms its single-scale counterpart by a 27.2\% improvement in rFID ($1.47 \rightarrow 1.07$). When integrated into downstream generation frameworks, it achieves a $1.38\times$ faster convergence rate and an 18.9\% boost in gFID ($16.4 \rightarrow 13.3$), which may be attributed to the smoother and more uniformly distributed latent space. Furthermore, by scaling up the tokenizer's training, we demonstrate its potential by a sota rFID of 0.45 and a gFID of 1.82 among ViT tokenizers. To the best of our knowledge, we are the first to introduce multi-scale ViT-based tokenizer in image reconstruction and image generation. We hope our findings and designs advance the ViT-based tokenizers in visual generation tasks. 

---
# LUQ: Layerwise Ultra-Low Bit Quantization for Multimodal Large Language Models 

**Authors**: Shubhang Bhatnagar, Andy Xu, Kar-Han Tan, Narendra Ahuja  

**Link**: [PDF](https://arxiv.org/pdf/2509.23729)  

**Abstract**: Large Language Models (LLMs) with multimodal capabilities have revolutionized vision-language tasks, but their deployment often requires huge memory and computational resources. While post-training quantization (PTQ) has successfully compressed language models to as low as 1-bit precision without significant performance loss, its effectiveness for multimodal LLMs (MLLMs) remains relatively unexplored. In this paper, we present the first study on ultra-low bit (<4-bit) quantization for multimodal LLMs. Our analysis reveals that multimodal tokens and intermediate layer activations produced by them exhibit significantly higher statistical variance and entropy compared to text tokens, making them less tolerant to ultra-low bit quantization. However, the activation distributions of multimodal tokens varies significantly over different layers, with some layers having lower entropy activation distributions. We empirically show that such layers in these models can better tolerate ultra-low bit quantization. Building on these insights, we propose a novel strategy for MLLM quantization, LUQ: Layerwise Ultra-Low Bit Quantization, which selectively applies ultra-low bit quantization to layers that are more resilient to it. Additionally, we also show that using a mix of multimodal tokens (image and text) for PTQ boosts VQA performance in the ultra-low bit regime. We evaluate our method on LLaVA-1.5 and Qwen-2.5-VL across 9 popular VQA benchmarks. The resulting LUQ models use 40% and 31% less memory than their 4-bit counterparts, respectively, while exhibiting a performance degradation of less than 10% on the MME benchmark. 

---
# M3DLayout: A Multi-Source Dataset of 3D Indoor Layouts and Structured Descriptions for 3D Generation 

**Authors**: Yiheng Zhang, Zhuojiang Cai, Mingdao Wang, Meitong Guo, Tianxiao Li, Li Lin, Yuwang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23728)  

**Abstract**: In text-driven 3D scene generation, object layout serves as a crucial intermediate representation that bridges high-level language instructions with detailed geometric output. It not only provides a structural blueprint for ensuring physical plausibility but also supports semantic controllability and interactive editing. However, the learning capabilities of current 3D indoor layout generation models are constrained by the limited scale, diversity, and annotation quality of existing datasets. To address this, we introduce M3DLayout, a large-scale, multi-source dataset for 3D indoor layout generation. M3DLayout comprises 15,080 layouts and over 258k object instances, integrating three distinct sources: real-world scans, professional CAD designs, and procedurally generated scenes. Each layout is paired with detailed structured text describing global scene summaries, relational placements of large furniture, and fine-grained arrangements of smaller items. This diverse and richly annotated resource enables models to learn complex spatial and semantic patterns across a wide variety of indoor environments. To assess the potential of M3DLayout, we establish a benchmark using a text-conditioned diffusion model. Experimental results demonstrate that our dataset provides a solid foundation for training layout generation models. Its multi-source composition enhances diversity, notably through the Inf3DLayout subset which provides rich small-object information, enabling the generation of more complex and detailed scenes. We hope that M3DLayout can serve as a valuable resource for advancing research in text-driven 3D scene synthesis. 

---
# AudioMoG: Guiding Audio Generation with Mixture-of-Guidance 

**Authors**: Junyou Wang, Zehua Chen, Binjie Yuan, Kaiwen Zheng, Chang Li, Yuxuan Jiang, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23727)  

**Abstract**: Guidance methods have demonstrated significant improvements in cross-modal audio generation, including text-to-audio (T2A) and video-to-audio (V2A) generation. The popularly adopted method, classifier-free guidance (CFG), steers generation by emphasizing condition alignment, enhancing fidelity but often at the cost of diversity. Recently, autoguidance (AG) has been explored for audio generation, encouraging the sampling to faithfully reconstruct the target distribution and showing increased diversity. Despite these advances, they usually rely on a single guiding principle, e.g., condition alignment in CFG or score accuracy in AG, leaving the full potential of guidance for audio generation untapped. In this work, we explore enriching the composition of the guidance method and present a mixture-of-guidance framework, AudioMoG. Within the design space, AudioMoG can exploit the complementary advantages of distinctive guiding principles by fulfilling their cumulative benefits. With a reduced form, AudioMoG can consider parallel complements or recover a single guiding principle, without sacrificing generality. We experimentally show that, given the same inference speed, AudioMoG approach consistently outperforms single guidance in T2A generation across sampling steps, concurrently showing advantages in V2A, text-to-music, and image generation. These results highlight a "free lunch" in current cross-modal audio generation systems: higher quality can be achieved through mixed guiding principles at the sampling stage without sacrificing inference efficiency. Demo samples are available at: this https URL. 

---
# Video Panels for Long Video Understanding 

**Authors**: Lars Doorenbos, Federico Spurio, Juergen Gall  

**Link**: [PDF](https://arxiv.org/pdf/2509.23724)  

**Abstract**: Recent Video-Language Models (VLMs) achieve promising results on long-video understanding, but their performance still lags behind that achieved on tasks involving images or short videos. This has led to great interest in improving the long context modeling of VLMs by introducing novel modules and additional complexity. % additional training time. In this paper, we take a different approach: rather than fine-tuning VLMs with the limited data available, we attempt to maximize the performance of existing models. To this end, we propose a novel visual prompting strategy specifically designed for long-video understanding. By combining multiple frames as panels into one image, we effectively trade off spatial details for temporal resolution. Our approach is training-free, parameter-free, and model-agnostic, and can be seamlessly integrated into existing VLMs. Extensive experiments on five established benchmarks across a wide range of model architectures, sizes, and context windows confirm the consistency of our approach. For the TimeScope (Long) dataset, which has the longest videos, the accuracy for video question answering is improved by up to 19.4\%. Overall, our method raises the bar for long video understanding models. We will make our code available upon acceptance. 

---
# AdaPtis: Reducing Pipeline Bubbles with Adaptive Pipeline Parallelism on Heterogeneous Models 

**Authors**: Jihu Guo, Tenghui Ma, Wei Gao, Peng Sun, Jiaxing Li, Xun Chen, Yuyang Jin, Dahua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.23722)  

**Abstract**: Pipeline parallelism is widely used to train large language models (LLMs). However, increasing heterogeneity in model architectures exacerbates pipeline bubbles, thereby reducing training efficiency. Existing approaches overlook the co-optimization of model partition, model placement, and workload scheduling, resulting in limited efficiency improvement or even performance degradation. To respond, we propose AdaPtis, an LLM training system that supports adaptive pipeline parallelism. First, we develop a pipeline performance model to accurately estimate training throughput. Second, AdaPtis jointly optimizes model partition, model placement, and workload scheduling policies guided by this performance model. Third, we design a unified pipeline executor that efficiently supports the execution of diverse pipeline strategies. Extensive experiments show that AdaPtis achieves an average speedup of 1.42x (up to 2.14x) over Megatron-LM I-1F1B across various LLM architectures and scales. 

---
# Bridging Discrete and Continuous RL: Stable Deterministic Policy Gradient with Martingale Characterization 

**Authors**: Ziheng Cheng, Xin Guo, Yufei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23711)  

**Abstract**: The theory of discrete-time reinforcement learning (RL) has advanced rapidly over the past decades. Although primarily designed for discrete environments, many real-world RL applications are inherently continuous and complex. A major challenge in extending discrete-time algorithms to continuous-time settings is their sensitivity to time discretization, often leading to poor stability and slow convergence. In this paper, we investigate deterministic policy gradient methods for continuous-time RL. We derive a continuous-time policy gradient formula based on an analogue of the advantage function and establish its martingale characterization. This theoretical foundation leads to our proposed algorithm, CT-DDPG, which enables stable learning with deterministic policies in continuous-time environments. Numerical experiments show that the proposed CT-DDPG algorithm offers improved stability and faster convergence compared to existing discrete-time and continuous-time methods, across a wide range of control tasks with varying time discretizations and noise levels. 

---
# CrimEdit: Controllable Editing for Counterfactual Object Removal, Insertion, and Movement 

**Authors**: Boseong Jeon, Junghyuk Lee, Jimin Park, Kwanyoung Kim, Jingi Jung, Sangwon Lee, Hyunbo Shim  

**Link**: [PDF](https://arxiv.org/pdf/2509.23708)  

**Abstract**: Recent works on object removal and insertion have enhanced their performance by handling object effects such as shadows and reflections, using diffusion models trained on counterfactual datasets. However, the performance impact of applying classifier-free guidance to handle object effects across removal and insertion tasks within a unified model remains largely unexplored. To address this gap and improve efficiency in composite editing, we propose CrimEdit, which jointly trains the task embeddings for removal and insertion within a single model and leverages them in a classifier-free guidance scheme -- enhancing the removal of both objects and their effects, and enabling controllable synthesis of object effects during insertion. CrimEdit also extends these two task prompts to be applied to spatially distinct regions, enabling object movement (repositioning) within a single denoising step. By employing both guidance techniques, extensive experiments show that CrimEdit achieves superior object removal, controllable effect insertion, and efficient object movement without requiring additional training or separate removal and insertion stages. 

---
# Estimating Time Series Foundation Model Transferability via In-Context Learning 

**Authors**: Qingren Yao, Ming Jin, Chengqi Zhang, Chao-Han Huck Yang, Jun Qi, Shirui Pan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23695)  

**Abstract**: Time series foundation models (TSFMs) offer strong zero-shot forecasting via large-scale pre-training, yet fine-tuning remains critical for boosting performance in domains with limited public data. With the growing number of TSFMs, efficiently identifying the best model for downstream fine-tuning becomes increasingly challenging. In this work, we introduce TimeTic, a transferability estimation framework that recasts model selection as an in-context-learning problem: given observations on known (source) datasets, it predicts how a TSFM will perform after fine-tuning on a downstream (target) dataset. TimeTic flexibly organizes the observed model-data relationships as contextual information, allowing it to adapt seamlessly to various test-time scenarios. Leveraging the natural tabular structure formed by dataset meta-features, model characteristics, and fine-tuned performance, we employ tabular foundation models to serve as in-context learners. We further introduce a novel model characterization based on entropy evolution across model layers, capturing embedding-space distinctions and enabling TimeTic to generalize across arbitrary model sets. We establish a comprehensive benchmark for transferability estimation including 10 datasets, 10 foundation models, and 3 forecasting tasks. On this benchmark, TimeTic's estimation demonstrates strong alignment with actual fine-tuned performance for previously unseen datasets, achieving a mean rank correlation of approximately 0.6 and a 30% improvement compared to using zero-shot performance as the transferability score. 

---
# Joint Hybrid Beamforming and Artificial Noise Design for Secure Multi-UAV ISAC Networks 

**Authors**: Runze Dong, Buhong Wang, Cunqian Feng, Jiang Weng, Chen Han, Jiwei Tian  

**Link**: [PDF](https://arxiv.org/pdf/2509.23687)  

**Abstract**: Integrated sensing and communication (ISAC) emerges as a key enabler for next-generation applications such as smart cities and autonomous systems. Its integration with unmanned aerial vehicles (UAVs) unlocks new potentials for reliable communication and precise sensing in dynamic aerial environments. However, existing research predominantly treats UAVs as aerial base stations, overlooking their role as ISAC users, and fails to leverage large-scale antenna arrays at terrestrial base stations to enhance security and spectral efficiency. This paper propose a secure and spectral efficient ISAC framework for multi-UAV networks, and a two-stage optimization approach is developed to jointly design hybrid beamforming (HBF), artificial noise (AN) injection, and UAV trajectories. Aiming at maximizing the sum secrecy rate, the first stage employs Proximal Policy Optimization (PPO) to optimize digital beamformers and trajectories, and the second stage decomposes the digital solution into analog and digital components via low-complexity matrix factorization. Simulation results demonstrate the effectiveness of the proposed framework compared to benchmark schemes. 

---
# Towards a Comprehensive Scaling Law of Mixture-of-Experts 

**Authors**: Guoliang Zhao, Yuhan Fu, Shuaipeng Li, Xingwu Sun, Ruobing Xie, An Wang, Weidong Han, Zhen Yang, Weixuan Sun, Yudong Zhang, Cheng-zhong Xu, Di Wang, Jie Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23678)  

**Abstract**: Mixture-of-Experts (MoE) models have become the consensus approach for enabling parameter-efficient scaling and cost-effective deployment in large language models. However, existing scaling laws for dense models are inapplicable to MoE models, which stems from three critical challenges: the multiplicity of influencing factors, their intricate coupling relationships and the non-monotonic nature of their performance impacts. They collectively necessitate a fine-grained investigation into MoE-specific scaling laws. In this work, we perform a systematic decomposition of MoE settings, identifying five key factors that influence model performance from both size and structural perspectives (data size ($D$), total model size ($N$), activated model size ($N_a$), number of active experts ($G$) and the ratio of shared experts ($S$)). Specifically, we design $446$ controlled experiments to characterize their marginal effects, ultimately constructing a comprehensive and precise joint MoE scaling law that considers all essential factors. Furthermore, we derive the theoretically optimal and practically efficiency-aware optimal configurations for $G$, $S$ and $N_a/N$ with detailed analyses. Our results demonstrate that the optimal settings for $G$ and $S$ are independent of both the model architecture and data size. With the scaling of $N$, the optimal activation parameter ratio of $N_a/N$ becomes sparser. Our proposed MoE scaling law could function as an accurate and insightful guidance to facilitate future MoE model design and training. 

---
# RCI: A Score for Evaluating Global and Local Reasoning in Multimodal Benchmarks 

**Authors**: Amit Agarwal, Hitesh Laxmichand Patel, Srikant Panda, Hansa Meghwani, Jyotika Singh, Karan Dua, Paul Li, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2509.23673)  

**Abstract**: Multimodal Large Language Models (MLLMs) have achieved impressive results on vision-language benchmarks, yet it remains unclear whether these benchmarks assess genuine global reasoning or allow success via localized visual cues. Existing evaluation methods do not explicitly measure this distinction, hindering effective dataset curation and real-world focused model development.
We introduce Region Comprehension Index (RCI), the first model-based score to directly quantify a dataset's reliance on global versus local visual information. RCI systematically compares reference-model performance on image patches versus full images, revealing if tasks require holistic image understanding or can be solved with partial or localized visual cues.
When applying RCI to 13 widely used multimodal benchmarks, we observed that most of them favor localized reasoning and exhibit significant spatial biases, indicating potential risks in real-world applications. RCI equips researchers & practitioners with an actionable tool for diagnosing & mitigating these biases, enabling the construction of datasets and benchmarks to foster the development of robust, enterprise-ready multimodal systems. 

---
# Graph Neural Networks with Diversity-aware Neighbor Selection and Dynamic Multi-scale Fusion for Multivariate Time Series Forecasting 

**Authors**: Jingqi Xu, Guibin Chen, Jingxi Lu, Yuzhang Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.23671)  

**Abstract**: Recently, numerous deep models have been proposed to enhance the performance of multivariate time series (MTS) forecasting. Among them, Graph Neural Networks (GNNs)-based methods have shown great potential due to their capability to explicitly model inter-variable dependencies. However, these methods often overlook the diversity of information among neighbors, which may lead to redundant information aggregation. In addition, their final prediction typically relies solely on the representation from a single temporal scale. To tackle these issues, we propose a Graph Neural Networks (GNNs) with Diversity-aware Neighbor Selection and Dynamic Multi-scale Fusion (DIMIGNN). DIMIGNN introduces a Diversity-aware Neighbor Selection Mechanism (DNSM) to ensure that each variable shares high informational similarity with its neighbors while maintaining diversity among neighbors themselves. Furthermore, a Dynamic Multi-Scale Fusion Module (DMFM) is introduced to dynamically adjust the contributions of prediction results from different temporal scales to the final forecasting result. Extensive experiments on real-world datasets demonstrate that DIMIGNN consistently outperforms prior methods. 

---
# Beyond Greedy Exits: Improved Early Exit Decisions for Risk Control and Reliability 

**Authors**: Divya Jyoti Bajpai, Manjesh Kumar Hanawal  

**Link**: [PDF](https://arxiv.org/pdf/2509.23666)  

**Abstract**: Early-Exit Deep Neural Networks enable adaptive inference by allowing prediction at intermediary layers, significantly reducing computational costs and latency. Most of the early exit strategies greedily exit a sample at an intermediary layer if the confidence in class prediction exceeds a predefined threshold that is set using a static validation set. This is problematic as the model might be overconfident in a wrong class. Also, they are not robust to distribution shifts encountered in deployment, which can undermine model trustworthiness and accuracy. To address these challenges, we propose UAT that adapts the threshold for exit decisions using a Multi-Armed Bandit framework, enabling online, unsupervised adjustment of exit decisions. UAT makes decisions based on a new reward function that assesses predictive certainty and its reliability to balance computational efficiency and prediction quality while penalizing unnecessary late exits. We provide guarantees on risk achieved by UAT and validate its performance on diverse tasks spanning vision-language understanding, text generation, and classification. Our framework demonstrates consistent improvements in speedup (1.70-2.10x) with a minimal performance drop (<2%) as compared to full model performance. Our source code is available at this https URL. 

---
# Calibration Meets Reality: Making Machine Learning Predictions Trustworthy 

**Authors**: Kristina P. Sinaga, Arjun S. Nair  

**Link**: [PDF](https://arxiv.org/pdf/2509.23665)  

**Abstract**: Post-hoc calibration methods are widely used to improve the reliability of probabilistic predictions from machine learning models. Despite their prevalence, a comprehensive theoretical understanding of these methods remains elusive, particularly regarding their performance across different datasets and model architectures. Input features play a crucial role in shaping model predictions and, consequently, their calibration. However, the interplay between feature quality and calibration performance has not been thoroughly investigated. In this work, we present a rigorous theoretical analysis of post-hoc calibration methods, focusing on Platt scaling and isotonic regression. We derive convergence guarantees, computational complexity bounds, and finite-sample performance metrics for these methods. Furthermore, we explore the impact of feature informativeness on calibration performance through controlled synthetic experiments. Our empirical evaluation spans a diverse set of real-world datasets and model architectures, demonstrating consistent improvements in calibration metrics across various scenarios. By examining calibration performance under varying feature conditions utilizing only informative features versus complete feature spaces including noise dimensions, we provide fundamental insights into the robustness and reliability of different calibration approaches. Our findings offer practical guidelines for selecting appropriate calibration methods based on dataset characteristics and computational constraints, bridging the gap between theoretical understanding and practical implementation in uncertainty quantification. Code and experimental data are available at: this https URL. 

---
# Pure Node Selection for Imbalanced Graph Node Classification 

**Authors**: Fanlong Zeng, Wensheng Gan, Jiayang Wu, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23662)  

**Abstract**: The problem of class imbalance refers to an uneven distribution of quantity among classes in a dataset, where some classes are significantly underrepresented compared to others. Class imbalance is also prevalent in graph-structured data. Graph neural networks (GNNs) are typically based on the assumption of class balance, often overlooking the issue of class imbalance. In our investigation, we identified a problem, which we term the Randomness Anomalous Connectivity Problem (RACP), where certain off-the-shelf models are affected by random seeds, leading to a significant performance degradation. To eliminate the influence of random factors in algorithms, we proposed PNS (Pure Node Sampling) to address the RACP in the node synthesis stage. Unlike existing approaches that design specialized algorithms to handle either quantity imbalance or topological imbalance, PNS is a novel plug-and-play module that operates directly during node synthesis to mitigate RACP. Moreover, PNS also alleviates performance degradation caused by abnormal distribution of node neighbors. We conduct a series of experiments to identify what factors are influenced by random seeds. Experimental results demonstrate the effectiveness and stability of our method, which not only eliminates the effect of unfavorable random seeds but also outperforms the baseline across various benchmark datasets with different GNN backbones. Data and code are available at this https URL. 

---
# Aligning LLMs for Multilingual Consistency in Enterprise Applications 

**Authors**: Amit Agarwal, Hansa Meghwani, Hitesh Laxmichand Patel, Tao Sheng, Sujith Ravi, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2509.23659)  

**Abstract**: Large language models (LLMs) remain unreliable for global enterprise applications due to substantial performance gaps between high-resource and mid/low-resource languages, driven by English-centric pretraining and internal reasoning biases. This inconsistency undermines customer experience and operational reliability in multilingual settings such as customer support, content moderation, and information retrieval. Even with advanced Retrieval-Augmented Generation (RAG) systems, we observe up to an 29% accuracy drop in non-English languages compared to English.
We propose a practical, batch-wise alignment strategy for fine-tuning LLMs, leveraging semantically equivalent multilingual data in each training batch to directly align model outputs across languages. This approach improves non-English accuracy by up to 23.9\% without compromising English performance, model reasoning, or retrieval quality. Our method is simple to implement, scalable, and integrates seamlessly with existing LLM training \& deployment pipelines, enabling more robust and equitable multilingual AI solutions in industry. 

---
# Focusing on What Matters: Object-Agent-centric Tokenization for Vision Language Action models 

**Authors**: Rokas Bendikas, Daniel Dijkman, Markus Peschl, Sanjay Haresh, Pietro Mazzaglia  

**Link**: [PDF](https://arxiv.org/pdf/2509.23655)  

**Abstract**: Vision-Language-Action (VLA) models offer a pivotal approach to learning robotic manipulation at scale by repurposing large pre-trained Vision-Language-Models (VLM) to output robotic actions. However, adapting VLMs for robotic domains comes with an unnecessarily high computational cost, which we attribute to the tokenization scheme of visual inputs. In this work, we aim to enable efficient VLA training by proposing Oat-VLA, an Object-Agent-centric Tokenization for VLAs. Building on the insights of object-centric representation learning, our method introduces an inductive bias towards scene objects and the agent's own visual information. As a result, we find that Oat-VLA can drastically reduce the number of visual tokens to just a few tokens without sacrificing performance. We reveal that Oat-VLA converges at least twice as fast as OpenVLA on the LIBERO suite, as well as outperform OpenVLA in diverse real-world pick and place tasks. 

---
# ReWatch-R1: Boosting Complex Video Reasoning in Large Vision-Language Models through Agentic Data Synthesis 

**Authors**: Congzhi Zhang, Zhibin Wang, Yinchao Ma, Jiawei Peng, Yihan Wang, Qiang Zhou, Jun Song, Bo Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23652)  

**Abstract**: While Reinforcement Learning with Verifiable Reward (RLVR) significantly advances image reasoning in Large Vision-Language Models (LVLMs), its application to complex video reasoning remains underdeveloped. This gap stems primarily from a critical data bottleneck: existing datasets lack the challenging, multi-hop questions and high-quality, video-grounded Chain-of-Thought (CoT) data necessary to effectively bootstrap RLVR. To address this, we introduce ReWatch, a large-scale dataset built to foster advanced video reasoning. We propose a novel multi-stage synthesis pipeline to synthesize its three components: ReWatch-Caption, ReWatch-QA, and ReWatch-CoT. A core innovation is our Multi-Agent ReAct framework for CoT synthesis, which simulates a human-like "re-watching" process to generate video-grounded reasoning traces by explicitly modeling information retrieval and verification. Building on this dataset, we develop ReWatch-R1 by post-training a strong baseline LVLM with Supervised Fine-Tuning (SFT) and our RLVR framework. This framework incorporates a novel Observation \& Reasoning (O\&R) reward mechanism that evaluates both the final answer's correctness and the reasoning's alignment with video content, directly penalizing hallucination. Our experiments show that ReWatch-R1 achieves state-of-the-art average performance on five challenging video reasoning benchmarks. 

---
# LightFair: Towards an Efficient Alternative for Fair T2I Diffusion via Debiasing Pre-trained Text Encoders 

**Authors**: Boyu Han, Qianqian Xu, Shilong Bao, Zhiyong Yang, Kangli Zi, Qingming Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23639)  

**Abstract**: This paper explores a novel lightweight approach LightFair to achieve fair text-to-image diffusion models (T2I DMs) by addressing the adverse effects of the text encoder. Most existing methods either couple different parts of the diffusion model for full-parameter training or rely on auxiliary networks for correction. They incur heavy training or sampling burden and unsatisfactory performance. Since T2I DMs consist of multiple components, with the text encoder being the most fine-tunable and front-end module, this paper focuses on mitigating bias by fine-tuning text embeddings. To validate feasibility, we observe that the text encoder's neutral embedding output shows substantial skewness across image embeddings of various attributes in the CLIP space. More importantly, the noise prediction network further amplifies this imbalance. To finetune the text embedding, we propose a collaborative distance-constrained debiasing strategy that balances embedding distances to improve fairness without auxiliary references. However, mitigating bias can compromise the original generation quality. To address this, we introduce a two-stage text-guided sampling strategy to limit when the debiased text encoder intervenes. Extensive experiments demonstrate that LightFair is effective and efficient. Notably, on Stable Diffusion v1.5, our method achieves SOTA debiasing at just $1/4$ of the training burden, with virtually no increase in sampling burden. The code is available at this https URL. 

---
# RIV: Recursive Introspection Mask Diffusion Vision Language Model 

**Authors**: YuQian Li, Limeng Qiao, Lin Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.23625)  

**Abstract**: Mask Diffusion-based Vision Language Models (MDVLMs) have achieved remarkable progress in multimodal understanding tasks. However, these models are unable to correct errors in generated tokens, meaning they lack self-correction capability. In this paper, we propose Recursive Introspection Mask Diffusion Vision Language Model (RIV), which equips the model with self-correction ability through two novel mechanisms. The first is Introspection Training, where an Introspection Model is introduced to identify errors within generated sequences. Introspection Training enables the model to detect not only grammatical and spelling mistakes, but more importantly, logical errors. The second is Recursive Inference. Beginning with the standard unmasking step, the learned Introspection Model helps to identify errors in the output sequence and remask them. This alternating ($\text{unmask}\rightarrow\text{introspection}\rightarrow\text{remask}$) process is repeated recursively until reliable results are obtained. Experimental results on multiple benchmarks demonstrate that the proposed RIV achieves state-of-the-art performance, outperforming most existing MDVLMs. 

---
# Generalizable Speech Deepfake Detection via Information Bottleneck Enhanced Adversarial Alignment 

**Authors**: Pu Huang, Shouguang Wang, Siya Yao, Mengchu Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.23618)  

**Abstract**: Neural speech synthesis techniques have enabled highly realistic speech deepfakes, posing major security risks. Speech deepfake detection is challenging due to distribution shifts across spoofing methods and variability in speakers, channels, and recording conditions. We explore learning shared discriminative features as a path to robust detection and propose Information Bottleneck enhanced Confidence-Aware Adversarial Network (IB-CAAN). Confidence-guided adversarial alignment adaptively suppresses attack-specific artifacts without erasing discriminative cues, while the information bottleneck removes nuisance variability to preserve transferable features. Experiments on ASVspoof 2019/2021, ASVspoof 5, and In-the-Wild demonstrate that IB-CAAN consistently outperforms baseline and achieves state-of-the-art performance on many benchmarks. 

---
# BioVessel-Net and RetinaMix: Unsupervised Retinal Vessel Segmentation from OCTA Images 

**Authors**: Cheng Huang, Weizheng Xie, Fan Gao, Yutong Liu, Ruoling Wu, Zeyu Han, Jingxi Qiu, Xiangxiang Wang, Zhenglin Yang, Hao Wang, Yongbin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23617)  

**Abstract**: Structural changes in retinal blood vessels are critical biomarkers for the onset and progression of glaucoma and other ocular diseases. However, current vessel segmentation approaches largely rely on supervised learning and extensive manual annotations, which are costly, error-prone, and difficult to obtain in optical coherence tomography angiography. Here we present BioVessel-Net, an unsupervised generative framework that integrates vessel biostatistics with adversarial refinement and a radius-guided segmentation strategy. Unlike pixel-based methods, BioVessel-Net directly models vascular structures with biostatistical coherence, achieving accurate and explainable vessel extraction without labeled data or high-performance computing. To support training and evaluation, we introduce RetinaMix, a new benchmark dataset of 2D and 3D OCTA images with high-resolution vessel details from diverse populations. Experimental results demonstrate that BioVessel-Net achieves near-perfect segmentation accuracy across RetinaMix and existing datasets, substantially outperforming state-of-the-art supervised and semi-supervised methods. Together, BioVessel-Net and RetinaMix provide a label-free, computationally efficient, and clinically interpretable solution for retinal vessel analysis, with broad potential for glaucoma monitoring, blood flow modeling, and progression prediction. Code and dataset are available: this https URL. 

---
# GraphIFE: Rethinking Graph Imbalance Node Classification via Invariant Learning 

**Authors**: Fanlong Zeng, Wensheng Gan, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23616)  

**Abstract**: The class imbalance problem refers to the disproportionate distribution of samples across different classes within a dataset, where the minority classes are significantly underrepresented. This issue is also prevalent in graph-structured data. Most graph neural networks (GNNs) implicitly assume a balanced class distribution and therefore often fail to account for the challenges introduced by class imbalance, which can lead to biased learning and degraded performance on minority classes. We identify a quality inconsistency problem in synthesized nodes, which leads to suboptimal performance under graph imbalance conditions. To mitigate this issue, we propose GraphIFE (Graph Invariant Feature Extraction), a novel framework designed to mitigate quality inconsistency in synthesized nodes. Our approach incorporates two key concepts from graph invariant learning and introduces strategies to strengthen the embedding space representation, thereby enhancing the model's ability to identify invariant features. Extensive experiments demonstrate the framework's efficiency and robust generalization, as GraphIFE consistently outperforms various baselines across multiple datasets. The code is publicly available at this https URL. 

---
# InteractMove: Text-Controlled Human-Object Interaction Generation in 3D Scenes with Movable Objects 

**Authors**: Xinhao Cai, Minghang Zheng, Xin Jin, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23612)  

**Abstract**: We propose a novel task of text-controlled human object interaction generation in 3D scenes with movable objects. Existing human-scene interaction datasets suffer from insufficient interaction categories and typically only consider interactions with static objects (do not change object positions), and the collection of such datasets with movable objects is difficult and costly. To address this problem, we construct the InteractMove dataset for Movable Human-Object Interaction in 3D Scenes by aligning existing human object interaction data with scene contexts, featuring three key characteristics: 1) scenes containing multiple movable objects with text-controlled interaction specifications (including same-category distractors requiring spatial and 3D scene context understanding), 2) diverse object types and sizes with varied interaction patterns (one-hand, two-hand, etc.), and 3) physically plausible object manipulation trajectories. With the introduction of various movable objects, this task becomes more challenging, as the model needs to identify objects to be interacted with accurately, learn to interact with objects of different sizes and categories, and avoid collisions between movable objects and the scene. To tackle such challenges, we propose a novel pipeline solution. We first use 3D visual grounding models to identify the interaction object. Then, we propose a hand-object joint affordance learning to predict contact regions for different hand joints and object parts, enabling accurate grasping and manipulation of diverse objects. Finally, we optimize interactions with local-scene modeling and collision avoidance constraints, ensuring physically plausible motions and avoiding collisions between objects and the scene. Comprehensive experiments demonstrate our method's superiority in generating physically plausible, text-compliant interactions compared to existing approaches. 

---
# Characteristic Root Analysis and Regularization for Linear Time Series Forecasting 

**Authors**: Zheng Wang, Kaixuan Zhang, Wanfang Chen, Xiaonan Lu, Longyuan Li, Tobias Schlagenhauf  

**Link**: [PDF](https://arxiv.org/pdf/2509.23597)  

**Abstract**: Time series forecasting remains a critical challenge across numerous domains, yet the effectiveness of complex models often varies unpredictably across datasets. Recent studies highlight the surprising competitiveness of simple linear models, suggesting that their robustness and interpretability warrant deeper theoretical investigation. This paper presents a systematic study of linear models for time series forecasting, with a focus on the role of characteristic roots in temporal dynamics. We begin by analyzing the noise-free setting, where we show that characteristic roots govern long-term behavior and explain how design choices such as instance normalization and channel independence affect model capabilities. We then extend our analysis to the noisy regime, revealing that models tend to produce spurious roots. This leads to the identification of a key data-scaling property: mitigating the influence of noise requires disproportionately large training data, highlighting the need for structural regularization. To address these challenges, we propose two complementary strategies for robust root restructuring. The first uses rank reduction techniques, including Reduced-Rank Regression and Direct Weight Rank Reduction, to recover the low-dimensional latent dynamics. The second, a novel adaptive method called Root Purge, encourages the model to learn a noise-suppressing null space during training. Extensive experiments on standard benchmarks demonstrate the effectiveness of both approaches, validating our theoretical insights and achieving state-of-the-art results in several settings. Our findings underscore the potential of integrating classical theories for linear systems with modern learning techniques to build robust, interpretable, and data-efficient forecasting models. 

---
# Multi-Level Heterogeneous Knowledge Transfer Network on Forward Scattering Center Model for Limited Samples SAR ATR 

**Authors**: Chenxi Zhao, Daochang Wang, Siqian Zhang, Gangyao Kuang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23596)  

**Abstract**: Simulated data-assisted SAR target recognition methods are the research hotspot currently, devoted to solving the problem of limited samples. Existing works revolve around simulated images, but the large amount of irrelevant information embedded in the images, such as background, noise, etc., seriously affects the quality of the migrated information. Our work explores a new simulated data to migrate purer and key target knowledge, i.e., forward scattering center model (FSCM) which models the actual local structure of the target with strong physical meaning and interpretability. To achieve this purpose, multi-level heterogeneous knowledge transfer (MHKT) network is proposed, which fully migrates FSCM knowledge from the feature, distribution and category levels, respectively. Specifically, we permit the more suitable feature representations for the heterogeneous data and separate non-informative knowledge by task-associated information selector (TAIS), to complete purer target feature migration. In the distribution alignment, the new metric function maximum discrimination divergence (MDD) in target generic knowledge transfer (TGKT) module perceives transferable knowledge efficiently while preserving discriminative structure about classes. Moreover, category relation knowledge transfer (CRKT) module leverages the category relation consistency constraint to break the dilemma of optimization bias towards simulation data due to imbalance between simulated and measured data. Such stepwise knowledge selection and migration will ensure the integrity of the migrated FSCM knowledge. Notably, extensive experiments on two new datasets formed by FSCM data and measured SAR images demonstrate the superior performance of our method. 

---
# Timber: Training-free Instruct Model Refining with Base via Effective Rank 

**Authors**: Taiqiang Wu, Runming Yang, Tao Liu, Jiahao Wang, Zenan Xu, Ngai Wong  

**Link**: [PDF](https://arxiv.org/pdf/2509.23595)  

**Abstract**: Post-training, which elicits a pretrained Base model into the corresponding Instruct model, is widely considered to be superficial. In this work, we first reinforce this hypothesis by providing novel quantitative evidence from the weight level that the effective rank (eRank) remains negligibly changed. However, this superficiality also suffers a critical trade-off, improving the exploitation capabilities at the cost of limiting its exploration. To tackle this issue, we propose Timber, a simple yet effective training-free method that enhances the exploration capability of the Instruct model while preserving its exploitation. The key insight is to partially revert Instruct towards the paired Base model by subtle yet targeted refinement of the weight deltas. Extensive experiments on Llama and Qwen series demonstrate that Timber consistently improves vanilla Instruct models, particularly on Pass@k performance. Our findings offer new insights into the post-training stage at the weight level and practical strategies to refine the Instruct model without training. 

---
# Toward a Holistic Approach to Continual Model Merging 

**Authors**: Hoang Phan, Sungmin Cha, Tung Lam Tran, Qi Lei  

**Link**: [PDF](https://arxiv.org/pdf/2509.23592)  

**Abstract**: We present a holistic framework for continual model merging that intervenes at three critical stages: pre-merging, during merging, and post-merging-to address two fundamental challenges in continual learning. In particular, conventional approaches either maintain a growing list of per-domain task vectors, leading to scalability issues or rely solely on weight-space merging when old data is inaccessible, thereby losing crucial functional information. Our method overcomes these limitations by first fine-tuning the main model within its tangent space on domain-specific data; this linearization amplifies per-task weight disentanglement, effectively mitigating across-task interference. During merging, we leverage functional information from available optimizer states beyond mere parameter averages to avoid the need to revisit old data. Finally, a post-merging correction aligns the representation discrepancy between pre- and post-merged models, reducing bias and enhancing overall performance-all while operating under constant memory constraints without accessing historical data. Extensive experiments on standard class-incremental and domain-incremental benchmarks demonstrate that our approach not only achieves competitive performance but also provides a scalable and efficient solution to the catastrophic forgetting problem. 

---
# Improving the Efficiency of LLM Agent Systems through Trajectory Reduction 

**Authors**: Yuan-An Xiao, Pengfei Gao, Chao Peng, Yingfei Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2509.23586)  

**Abstract**: Multi-turn agent systems based on Large Language Models (LLMs) have been increasingly popular for software engineering tasks. While LLM agents show decent effectiveness, the high computational cost of input tokens due to the ever-growing trajectory remains an efficiency concern for their applications. Efficiency is largely neglected in existing studies and agent products, and this paper fills the gap by introducing an inference-time trajectory reduction approach to reduce the cost of agents.
Through analyzing existing agent trajectories, we demonstrate that useless, redundant, and expired information is widespread in all trajectories, which can be identified and reduced without harming the agent's performance. We then design a simple yet effective trajectory reduction approach, AgentDiet, which automatically removes such waste information. We implement AgentDiet on a top-performing coding agent, and the evaluation on two LLMs and two benchmarks shows that AgentDiet can reduce input tokens by 39.9% ~ 59.7%, or the final computational cost by 21.1% ~ 35.9%, while maintaining the same agent performance. This indicates that trajectory reduction is a promising direction for agent systems. 

---
# ML-Asset Management: Curation, Discovery, and Utilization 

**Authors**: Mengying Wang, Moming Duan, Yicong Huang, Chen Li, Bingsheng He, Yinghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23577)  

**Abstract**: Machine learning (ML) assets, such as models, datasets, and metadata, are central to modern ML workflows. Despite their explosive growth in practice, these assets are often underutilized due to fragmented documentation, siloed storage, inconsistent licensing, and lack of unified discovery mechanisms, making ML-asset management an urgent challenge. This tutorial offers a comprehensive overview of ML-asset management activities across its lifecycle, including curation, discovery, and utilization. We provide a categorization of ML assets, and major management issues, survey state-of-the-art techniques, and identify emerging opportunities at each stage. We further highlight system-level challenges related to scalability, lineage, and unified indexing. Through live demonstrations of systems, this tutorial equips both researchers and practitioners with actionable insights and practical tools for advancing ML-asset management in real-world and domain-specific settings. 

---
# Towards Efficient CoT Distillation: Self-Guided Rationale Selector for Better Performance with Fewer Rationales 

**Authors**: Jianzhi Yan, Le Liu, Youcheng Pan, Shiwei Chen, Yang Xiang, Buzhou Tang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23574)  

**Abstract**: Chain-of-thought (CoT) distillation aims to enhance small language models' (SLMs) reasoning by transferring multi-step reasoning capability from the larger teacher models. However, existing work underestimates rationale quality, focusing primarily on data quantity, which may transfer noisy or incorrect information to the student model. To address the above issues, we proposed \textbf{M}odel-\textbf{O}riented \textbf{R}ationale \textbf{S}election \textbf{D}istillation (MoRSD), which can discern and select high quality rationales for distillation to improve performance further. We further propose a Rationale Difficulty (RD) metric to measure the ability of the student model to generate the correct answer under a given rationale. Compared to the baseline, we achieved 4.6$\%$ average improvement on seven datasets over three tasks, using fewer rationales by controlling their accuracy, diversity, and difficulty. Our results reveal that a small portion of the high quality rationales can enhance the reasoning ability of student models than the entire dataset. Our method promises to be a possible solution for efficient CoT distillation. Our code will be released in this https URL. 

---
# Uncovering Vulnerabilities of LLM-Assisted Cyber Threat Intelligence 

**Authors**: Yuqiao Meng, Luoxi Tang, Feiyang Yu, Jinyuan Jia, Guanhua Yan, Ping Yang, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23573)  

**Abstract**: Large Language Models (LLMs) are intensively used to assist security analysts in counteracting the rapid exploitation of cyber threats, wherein LLMs offer cyber threat intelligence (CTI) to support vulnerability assessment and incident response. While recent work has shown that LLMs can support a wide range of CTI tasks such as threat analysis, vulnerability detection, and intrusion defense, significant performance gaps persist in practical deployments. In this paper, we investigate the intrinsic vulnerabilities of LLMs in CTI, focusing on challenges that arise from the nature of the threat landscape itself rather than the model architecture. Using large-scale evaluations across multiple CTI benchmarks and real-world threat reports, we introduce a novel categorization methodology that integrates stratification, autoregressive refinement, and human-in-the-loop supervision to reliably analyze failure instances. Through extensive experiments and human inspections, we reveal three fundamental vulnerabilities: spurious correlations, contradictory knowledge, and constrained generalization, that limit LLMs in effectively supporting CTI. Subsequently, we provide actionable insights for designing more robust LLM-powered CTI systems to facilitate future research. 

---
# Benchmarking LLM-Assisted Blue Teaming via Standardized Threat Hunting 

**Authors**: Yuqiao Meng, Luoxi Tang, Feiyang Yu, Xi Li, Guanhua Yan, Ping Yang, Zhaohan Xi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23571)  

**Abstract**: As cyber threats continue to grow in scale and sophistication, blue team defenders increasingly require advanced tools to proactively detect and mitigate risks. Large Language Models (LLMs) offer promising capabilities for enhancing threat analysis. However, their effectiveness in real-world blue team threat-hunting scenarios remains insufficiently explored. This paper presents CyberTeam, a benchmark designed to guide LLMs in blue teaming practice. CyberTeam constructs a standardized workflow in two stages. First, it models realistic threat-hunting workflows by capturing the dependencies among analytical tasks from threat attribution to incident response. Next, each task is addressed through a set of operational modules tailored to its specific analytical requirements. This transforms threat hunting into a structured sequence of reasoning steps, with each step grounded in a discrete operation and ordered according to task-specific dependencies. Guided by this framework, LLMs are directed to perform threat-hunting tasks through modularized steps. Overall, CyberTeam integrates 30 tasks and 9 operational modules to guide LLMs through standardized threat analysis. We evaluate both leading LLMs and state-of-the-art cybersecurity agents, comparing CyberTeam against open-ended reasoning strategies. Our results highlight the improvements enabled by standardized design, while also revealing the limitations of open-ended reasoning in real-world threat hunting. 

---
# Node Classification via Simplicial Interaction with Augmented Maximal Clique Selection 

**Authors**: Eunho Koo, Tongseok Lim  

**Link**: [PDF](https://arxiv.org/pdf/2509.23568)  

**Abstract**: Considering higher-order interactions allows for a more comprehensive understanding of network structures beyond simple pairwise connections. While leveraging all cliques in a network to handle higher-order interactions is intuitive, it often leads to computational inefficiencies due to overlapping information between higher-order and lower-order cliques. To address this issue, we propose an augmented maximal clique strategy. Although using only maximal cliques can reduce unnecessary overlap and provide a concise representation of the network, certain nodes may still appear in multiple maximal cliques, resulting in imbalanced training data. Therefore, our augmented maximal clique approach selectively includes some non-maximal cliques to mitigate the overrepresentation of specific nodes and promote more balanced learning across the network. Comparative analyses on synthetic networks and real-world citation datasets demonstrate that our method outperforms approaches based on pairwise interactions, all cliques, or only maximal cliques. Finally, by integrating this strategy into GNN-based semi-supervised learning, we establish a link between maximal clique-based methods and GNNs, showing that incorporating higher-order structures improves predictive accuracy. As a result, the augmented maximal clique strategy offers a computationally efficient and effective solution for higher-order network learning. 

---
# RAVEN: Resilient Aerial Navigation via Open-Set Semantic Memory and Behavior Adaptation 

**Authors**: Seungchan Kim, Omar Alama, Dmytro Kurdydyk, John Keller, Nikhil Keetha, Wenshan Wang, Yonatan Bisk, Sebastian Scherer  

**Link**: [PDF](https://arxiv.org/pdf/2509.23563)  

**Abstract**: Aerial outdoor semantic navigation requires robots to explore large, unstructured environments to locate target objects. Recent advances in semantic navigation have demonstrated open-set object-goal navigation in indoor settings, but these methods remain limited by constrained spatial ranges and structured layouts, making them unsuitable for long-range outdoor search. While outdoor semantic navigation approaches exist, they either rely on reactive policies based on current observations, which tend to produce short-sighted behaviors, or precompute scene graphs offline for navigation, limiting adaptability to online deployment. We present RAVEN, a 3D memory-based, behavior tree framework for aerial semantic navigation in unstructured outdoor environments. It (1) uses a spatially consistent semantic voxel-ray map as persistent memory, enabling long-horizon planning and avoiding purely reactive behaviors, (2) combines short-range voxel search and long-range ray search to scale to large environments, (3) leverages a large vision-language model to suggest auxiliary cues, mitigating sparsity of outdoor targets. These components are coordinated by a behavior tree, which adaptively switches behaviors for robust operation. We evaluate RAVEN in 10 photorealistic outdoor simulation environments over 100 semantic tasks, encompassing single-object search, multi-class, multi-instance navigation and sequential task changes. Results show RAVEN outperforms baselines by 85.25% in simulation and demonstrate its real-world applicability through deployment on an aerial robot in outdoor field tests. 

---
# Pancreas Part Segmentation under Federated Learning Paradigm 

**Authors**: Ziliang Hong, Halil Ertugrul Aktas, Andrea Mia Bejar, Katherine Wu, Hongyi Pan, Gorkem Durak, Zheyuan Zhang, Sait Kayali, Temel Tirkes, Federica Proietto Salanitri, Concetto Spampinato, Michael Goggins, Tamas Gonda, Candice Bolan, Raj Keswani, Frank Miller, Michael Wallace, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2509.23562)  

**Abstract**: We present the first federated learning (FL) approach for pancreas part(head, body and tail) segmentation in MRI, addressing a critical clinical challenge as a significant innovation. Pancreatic diseases exhibit marked regional heterogeneity cancers predominantly occur in the head region while chronic pancreatitis causes tissue loss in the tail, making accurate segmentation of the organ into head, body, and tail regions essential for precise diagnosis and treatment planning. This segmentation task remains exceptionally challenging in MRI due to variable morphology, poor soft-tissue contrast, and anatomical variations across patients. Our novel contribution tackles two fundamental challenges: first, the technical complexity of pancreas part delineation in MRI, and second the data scarcity problem that has hindered prior approaches. We introduce a privacy-preserving FL framework that enables collaborative model training across seven medical institutions without direct data sharing, leveraging a diverse dataset of 711 T1W and 726 T2W MRI scans. Our key innovations include: (1) a systematic evaluation of three state-of-the-art segmentation architectures (U-Net, Attention U-Net,Swin UNETR) paired with two FL algorithms (FedAvg, FedProx), revealing Attention U-Net with FedAvg as optimal for pancreatic heterogeneity, which was never been done before; (2) a novel anatomically-informed loss function prioritizing region-specific texture contrasts in MRI. Comprehensive evaluation demonstrates that our approach achieves clinically viable performance despite training on distributed, heterogeneous datasets. 

---
# Fusing Sequence Motifs and Pan-Genomic Features: Antimicrobial Resistance Prediction using an Explainable Lightweight 1D CNN-XGBoost Ensemble 

**Authors**: Md. Saiful Bari Siddiqui, Nowshin Tarannum  

**Link**: [PDF](https://arxiv.org/pdf/2509.23552)  

**Abstract**: Antimicrobial Resistance (AMR) is a rapidly escalating global health crisis. While genomic sequencing enables rapid prediction of resistance phenotypes, current computational methods have limitations. Standard machine learning models treat the genome as an unordered collection of features, ignoring the sequential context of Single Nucleotide Polymorphisms (SNPs). State-of-the-art sequence models like Transformers are often too data-hungry and computationally expensive for the moderately-sized datasets that are typical in this domain. To address these challenges, we propose AMR-EnsembleNet, an ensemble framework that synergistically combines sequence-based and feature-based learning. We developed a lightweight, custom 1D Convolutional Neural Network (CNN) to efficiently learn predictive sequence motifs from high-dimensional SNP data. This sequence-aware model was ensembled with an XGBoost model, a powerful gradient boosting system adept at capturing complex, non-local feature interactions. We trained and evaluated our framework on a benchmark dataset of 809 E. coli strains, predicting resistance across four antibiotics with varying class imbalance. Our 1D CNN-XGBoost ensemble consistently achieved top-tier performance across all the antibiotics, reaching a Matthews Correlation Coefficient (MCC) of 0.926 for Ciprofloxacin (CIP) and the highest Macro F1-score of 0.691 for the challenging Gentamicin (GEN) AMR prediction. We also show that our model consistently focuses on SNPs within well-known AMR genes like fusA and parC, confirming it learns the correct genetic signals for resistance. Our work demonstrates that fusing a sequence-aware 1D CNN with a feature-based XGBoost model creates a powerful ensemble, overcoming the limitations of using either an order-agnostic or a standalone sequence model. 

---
# Automatic Speech Recognition for Greek Medical Dictation 

**Authors**: Vardis Georgilas, Themos Stafylakis  

**Link**: [PDF](https://arxiv.org/pdf/2509.23550)  

**Abstract**: Medical dictation systems are essential tools in modern healthcare, enabling accurate and efficient conversion of speech into written medical documentation. The main objective of this paper is to create a domain-specific system for Greek medical speech transcriptions. The ultimate goal is to assist healthcare professionals by reducing the overload of manual documentation and improving workflow efficiency. Towards this goal, we develop a system that combines automatic speech recognition techniques with text correction model, allowing better handling of domain-specific terminology and linguistic variations in Greek. Our approach leverages both acoustic and textual modeling to create more realistic and reliable transcriptions. We focused on adapting existing language and speech technologies to the Greek medical context, addressing challenges such as complex medical terminology and linguistic inconsistencies. Through domain-specific fine-tuning, our system achieves more accurate and coherent transcriptions, contributing to the development of practical language technologies for the Greek healthcare sector. 

---
# Disentanglement of Variations with Multimodal Generative Modeling 

**Authors**: Yijie Zhang, Yiyang Shen, Weiran Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23548)  

**Abstract**: Multimodal data are prevalent across various domains, and learning robust representations of such data is paramount to enhancing generation quality and downstream task performance. To handle heterogeneity and interconnections among different modalities, recent multimodal generative models extract shared and private (modality-specific) information with two separate variables. Despite attempts to enforce disentanglement between these two variables, these methods struggle with challenging datasets where the likelihood model is insufficient. In this paper, we propose Information-disentangled Multimodal VAE (IDMVAE) to explicitly address this issue, with rigorous mutual information-based regularizations, including cross-view mutual information maximization for extracting shared variables, and a cycle-consistency style loss for redundancy removal using generative augmentations. We further introduce diffusion models to improve the capacity of latent priors. These newly proposed components are complementary to each other. Compared to existing approaches, IDMVAE shows a clean separation between shared and private information, demonstrating superior generation quality and semantic coherence on challenging datasets. 

---
# End-to-End Deep Learning for Predicting Metric Space-Valued Outputs 

**Authors**: Yidong Zhou, Su I Iao, Hans-Georg Müller  

**Link**: [PDF](https://arxiv.org/pdf/2509.23544)  

**Abstract**: Many modern applications involve predicting structured, non-Euclidean outputs such as probability distributions, networks, and symmetric positive-definite matrices. These outputs are naturally modeled as elements of general metric spaces, where classical regression techniques that rely on vector space structure no longer apply. We introduce E2M (End-to-End Metric regression), a deep learning framework for predicting metric space-valued outputs. E2M performs prediction via a weighted Fréchet means over training outputs, where the weights are learned by a neural network conditioned on the input. This construction provides a principled mechanism for geometry-aware prediction that avoids surrogate embeddings and restrictive parametric assumptions, while fully preserving the intrinsic geometry of the output space. We establish theoretical guarantees, including a universal approximation theorem that characterizes the expressive capacity of the model and a convergence analysis of the entropy-regularized training objective. Through extensive simulations involving probability distributions, networks, and symmetric positive-definite matrices, we show that E2M consistently achieves state-of-the-art performance, with its advantages becoming more pronounced at larger sample sizes. Applications to human mortality distributions and New York City taxi networks further demonstrate the flexibility and practical utility of the framework. 

---
# On the Shelf Life of Fine-Tuned LLM Judges: Future Proofing, Backward Compatibility, and Question Generalization 

**Authors**: Janvijay Singh, Austin Xu, Yilun Zhou, Yefan Zhou, Dilek Hakkani-Tur, Shafiq Joty  

**Link**: [PDF](https://arxiv.org/pdf/2509.23542)  

**Abstract**: The LLM-as-a-judge paradigm is widely used in both evaluating free-text model responses and reward modeling for model alignment and finetuning. Recently, finetuning judges with judge-specific data has emerged as an often preferred choice over directly prompting frontier models as judges, as the former achieves better performance with smaller model sizes while being more robust to common biases. However, the standard evaluation ignores several practical concerns of finetuned judges regarding their real world deployment. In this paper, we identify and formalize three aspects that affect the shelf life of these judges: future proofing and backward compatibility -- how well judges finetuned on responses by today's generator models perform on responses by future models or past models, as well as question generalization -- how well judges generalize to unseen questions at test time. We study these three aspects in the math domain under a unified framework with varying train and test distributions, three SFT- and DPO-based finetuning algorithms and three different base models. Experiments suggest that future-proofing is challenging for most models, while backward compatibility is relatively easy, with DPO-trained models consistently improving performance. We further find that continual learning provides a more balanced adaptation to shifts between older and newer response distributions than training solely on stronger or weaker responses. Moreover, all models observe certain degrees of performance degradation when moving from questions seen during training to unseen ones, showing that current judges do not fully generalize to unseen questions. These findings provide insights into practical considerations for developing and deploying judge models in the face of ever-changing generators. 

---
# Imaging-Based Mortality Prediction in Patients with Systemic Sclerosis 

**Authors**: Alec K. Peltekian, Karolina Senkow, Gorkem Durak, Kevin M. Grudzinski, Bradford C. Bemiss, Jane E. Dematte, Carrie Richardson, Nikolay S. Markov, Mary Carns, Kathleen Aren, Alexandra Soriano, Matthew Dapas, Harris Perlman, Aaron Gundersheimer, Kavitha C. Selvan, John Varga, Monique Hinchcliff, Krishnan Warrior, Catherine A. Gao, Richard G. Wunderink, GR Scott Budinger, Alok N. Choudhary, Anthony J. Esposito, Alexander V. Misharin, Ankit Agrawal, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2509.23530)  

**Abstract**: Interstitial lung disease (ILD) is a leading cause of morbidity and mortality in systemic sclerosis (SSc). Chest computed tomography (CT) is the primary imaging modality for diagnosing and monitoring lung complications in SSc patients. However, its role in disease progression and mortality prediction has not yet been fully clarified. This study introduces a novel, large-scale longitudinal chest CT analysis framework that utilizes radiomics and deep learning to predict mortality associated with lung complications of SSc. We collected and analyzed 2,125 CT scans from SSc patients enrolled in the Northwestern Scleroderma Registry, conducting mortality analyses at one, three, and five years using advanced imaging analysis techniques. Death labels were assigned based on recorded deaths over the one-, three-, and five-year intervals, confirmed by expert physicians. In our dataset, 181, 326, and 428 of the 2,125 CT scans were from patients who died within one, three, and five years, respectively. Using ResNet-18, DenseNet-121, and Swin Transformer we use pre-trained models, and fine-tuned on 2,125 images of SSc patients. Models achieved an AUC of 0.769, 0.801, 0.709 for predicting mortality within one-, three-, and five-years, respectively. Our findings highlight the potential of both radiomics and deep learning computational methods to improve early detection and risk assessment of SSc-related interstitial lung disease, marking a significant advancement in the literature. 

---
# Privy: Envisioning and Mitigating Privacy Risks for Consumer-facing AI Product Concepts 

**Authors**: Hao-Ping Lee, Yu-Ju Yang, Matthew Bilik, Isadora Krsek, Thomas Serban von Davier, Kyzyl Monteiro, Jason Lin, Shivani Agarwal, Jodi Forlizzi, Sauvik Das  

**Link**: [PDF](https://arxiv.org/pdf/2509.23525)  

**Abstract**: AI creates and exacerbates privacy risks, yet practitioners lack effective resources to identify and mitigate these risks. We present Privy, a tool that guides practitioners through structured privacy impact assessments to: (i) identify relevant risks in novel AI product concepts, and (ii) propose appropriate mitigations. Privy was shaped by a formative study with 11 practitioners, which informed two versions -- one LLM-powered, the other template-based. We evaluated these two versions of Privy through a between-subjects, controlled study with 24 separate practitioners, whose assessments were reviewed by 13 independent privacy experts. Results show that Privy helps practitioners produce privacy assessments that experts deemed high quality: practitioners identified relevant risks and proposed appropriate mitigation strategies. These effects were augmented in the LLM-powered version. Practitioners themselves rated Privy as being useful and usable, and their feedback illustrates how it helps overcome long-standing awareness, motivation, and ability barriers in privacy work. 

---
# ReliabilityRAG: Effective and Provably Robust Defense for RAG-based Web-Search 

**Authors**: Zeyu Shen, Basileal Imana, Tong Wu, Chong Xiang, Prateek Mittal, Aleksandra Korolova  

**Link**: [PDF](https://arxiv.org/pdf/2509.23519)  

**Abstract**: Retrieval-Augmented Generation (RAG) enhances Large Language Models by grounding their outputs in external documents. These systems, however, remain vulnerable to attacks on the retrieval corpus, such as prompt injection. RAG-based search systems (e.g., Google's Search AI Overview) present an interesting setting for studying and protecting against such threats, as defense algorithms can benefit from built-in reliability signals -- like document ranking -- and represent a non-LLM challenge for the adversary due to decades of work to thwart SEO.
Motivated by, but not limited to, this scenario, this work introduces ReliabilityRAG, a framework for adversarial robustness that explicitly leverages reliability information of retrieved documents.
Our first contribution adopts a graph-theoretic perspective to identify a "consistent majority" among retrieved documents to filter out malicious ones. We introduce a novel algorithm based on finding a Maximum Independent Set (MIS) on a document graph where edges encode contradiction. Our MIS variant explicitly prioritizes higher-reliability documents and provides provable robustness guarantees against bounded adversarial corruption under natural assumptions. Recognizing the computational cost of exact MIS for large retrieval sets, our second contribution is a scalable weighted sample and aggregate framework. It explicitly utilizes reliability information, preserving some robustness guarantees while efficiently handling many documents.
We present empirical results showing ReliabilityRAG provides superior robustness against adversarial attacks compared to prior methods, maintains high benign accuracy, and excels in long-form generation tasks where prior robustness-focused methods struggled. Our work is a significant step towards more effective, provably robust defenses against retrieved corpus corruption in RAG. 

---
# Evaluating point-light biological motion in multimodal large language models 

**Authors**: Akila Kadambi, Marco Iacoboni, Lisa Aziz-Zadeh, Srini Narayanan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23517)  

**Abstract**: Humans can extract rich semantic information from minimal visual cues, as demonstrated by point-light displays (PLDs), which consist of sparse sets of dots localized to key joints of the human body. This ability emerges early in development and is largely attributed to human embodied experience. Since PLDs isolate body motion as the sole source of meaning, they represent key stimuli for testing the constraints of action understanding in these systems. Here we introduce ActPLD, the first benchmark to evaluate action processing in MLLMs from human PLDs. Tested models include state-of-the-art proprietary and open-source systems on single-actor and socially interacting PLDs. Our results reveal consistently low performance across models, introducing fundamental gaps in action and spatiotemporal understanding. 

---
# From Human Annotation to Automation: LLM-in-the-Loop Active Learning for Arabic Sentiment Analysis 

**Authors**: Dania Refai, Alaa Dalaq, Doaa Dalaq, Irfan Ahmad  

**Link**: [PDF](https://arxiv.org/pdf/2509.23515)  

**Abstract**: Natural language processing (NLP), particularly sentiment analysis, plays a vital role in areas like marketing, customer service, and social media monitoring by providing insights into user opinions and emotions. However, progress in Arabic sentiment analysis remains limited due to the lack of large, high-quality labeled datasets. While active learning has proven effective in reducing annotation efforts in other languages, few studies have explored it in Arabic sentiment tasks. Likewise, the use of large language models (LLMs) for assisting annotation and comparing their performance to human labeling is still largely unexplored in the Arabic context. In this paper, we propose an active learning framework for Arabic sentiment analysis designed to reduce annotation costs while maintaining high performance. We evaluate multiple deep learning architectures: Specifically, long short-term memory (LSTM), gated recurrent units (GRU), and recurrent neural networks (RNN), across three benchmark datasets: Hunger Station, AJGT, and MASAC, encompassing both modern standard Arabic and dialectal variations. Additionally, two annotation strategies are compared: Human labeling and LLM-assisted labeling. Five LLMs are evaluated as annotators: GPT-4o, Claude 3 Sonnet, Gemini 2.5 Pro, DeepSeek Chat, and LLaMA 3 70B Instruct. For each dataset, the best-performing LLM was used: GPT-4o for Hunger Station, Claude 3 Sonnet for AJGT, and DeepSeek Chat for MASAC. Our results show that LLM-assisted active learning achieves competitive or superior performance compared to human labeling. For example, on the Hunger Station dataset, the LSTM model achieved 93% accuracy with only 450 labeled samples using GPT-4o-generated labels, while on the MASAC dataset, DeepSeek Chat reached 82% accuracy with 650 labeled samples, matching the accuracy obtained through human labeling. 

---
# Enhancing Polyp Segmentation via Encoder Attention and Dynamic Kernel Update 

**Authors**: Fatemeh Salahi Chashmi, Roya Sotoudeh  

**Link**: [PDF](https://arxiv.org/pdf/2509.23502)  

**Abstract**: Polyp segmentation is a critical step in colorectal cancer detection, yet it remains challenging due to the diverse shapes, sizes, and low contrast boundaries of polyps in medical imaging. In this work, we propose a novel framework that improves segmentation accuracy and efficiency by integrating a Dynamic Kernel (DK) mechanism with a global Encoder Attention module. The DK mechanism, initialized by a global context vector from the EA module, iteratively refines segmentation predictions across decoding stages, enabling the model to focus on and accurately delineate complex polyp boundaries. The EA module enhances the network's ability to capture critical lesion features by aggregating multi scale information from all encoder layers. In addition, we employ Unified Channel Adaptation (UCA) in the decoder to standardize feature dimensions across stages, ensuring consistent and computationally efficient information fusion. Our approach extends the lesion-aware kernel framework by introducing a more flexible, attention driven kernel initialization and a unified decoder design. Extensive experiments on the KvasirSEG and CVC ClinicDB benchmark datasets demonstrate that our model outperforms several state of the art segmentation methods, achieving superior Dice and Intersection over Union scores. Moreover, UCA simplifies the decoder structure, reducing computational cost without compromising accuracy. Overall, the proposed method provides a robust and adaptable solution for polyp segmentation, with promising applications in clinical and automated diagnostic systems. 

---
# The Impact of Role Design in In-Context Learning for Large Language Models 

**Authors**: Hamidreza Rouzegar, Masoud Makrehchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23501)  

**Abstract**: In-context learning (ICL) enables Large Language Models (LLMs) to generate predictions based on prompts without additional fine-tuning. While prompt engineering has been widely studied, the impact of role design within prompts remains underexplored. This study examines the influence of role configurations in zero-shot and few-shot learning scenarios using GPT-3.5 and GPT-4o from OpenAI and Llama2-7b and Llama2-13b from Meta. We evaluate the models' performance across datasets, focusing on tasks like sentiment analysis, text classification, question answering, and math reasoning. Our findings suggest the potential of role-based prompt structuring to enhance LLM performance. 

---
# Revisiting Multivariate Time Series Forecasting with Missing Values 

**Authors**: Jie Yang, Yifan Hu, Kexin Zhang, Luyang Niu, Yushun Dong, Philip S. Yu, Kaize Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.23494)  

**Abstract**: Missing values are common in real-world time series, and multivariate time series forecasting with missing values (MTSF-M) has become a crucial area of research for ensuring reliable predictions. To address the challenge of missing data, current approaches have developed an imputation-then-prediction framework that uses imputation modules to fill in missing values, followed by forecasting on the imputed data. However, this framework overlooks a critical issue: there is no ground truth for the missing values, making the imputation process susceptible to errors that can degrade prediction accuracy. In this paper, we conduct a systematic empirical study and reveal that imputation without direct supervision can corrupt the underlying data distribution and actively degrade prediction accuracy. To address this, we propose a paradigm shift that moves away from imputation and directly predicts from the partially observed time series. We introduce Consistency-Regularized Information Bottleneck (CRIB), a novel framework built on the Information Bottleneck principle. CRIB combines a unified-variate attention mechanism with a consistency regularization scheme to learn robust representations that filter out noise introduced by missing values while preserving essential predictive signals. Comprehensive experiments on four real-world datasets demonstrate the effectiveness of CRIB, which predicts accurately even under high missing rates. Our code is available in this https URL. 

---
# Text-Based Approaches to Item Difficulty Modeling in Large-Scale Assessments: A Systematic Review 

**Authors**: Sydney Peters, Nan Zhang, Hong Jiao, Ming Li, Tianyi Zhou, Robert Lissitz  

**Link**: [PDF](https://arxiv.org/pdf/2509.23486)  

**Abstract**: Item difficulty plays a crucial role in test performance, interpretability of scores, and equity for all test-takers, especially in large-scale assessments. Traditional approaches to item difficulty modeling rely on field testing and classical test theory (CTT)-based item analysis or item response theory (IRT) calibration, which can be time-consuming and costly. To overcome these challenges, text-based approaches leveraging machine learning and language models, have emerged as promising alternatives. This paper reviews and synthesizes 37 articles on automated item difficulty prediction in large-scale assessment settings published through May 2025. For each study, we delineate the dataset, difficulty parameter, subject domain, item type, number of items, training and test data split, input, features, model, evaluation criteria, and model performance outcomes. Results showed that although classic machine learning models remain relevant due to their interpretability, state-of-the-art language models, using both small and large transformer-based architectures, can capture syntactic and semantic patterns without the need for manual feature engineering. Uniquely, model performance outcomes were summarized to serve as a benchmark for future research and overall, text-based methods have the potential to predict item difficulty with root mean square error (RMSE) as low as 0.165, Pearson correlation as high as 0.87, and accuracy as high as 0.806. The review concludes by discussing implications for practice and outlining future research directions for automated item difficulty modeling. 

---
# Memory-Efficient Fine-Tuning via Low-Rank Activation Compression 

**Authors**: Jiang-Xin Shi, Wen-Da Wei, Jin-Fei Qi, Xuanyu Chen, Tong Wei, Yu-Feng Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23472)  

**Abstract**: The parameter-efficient fine-tuning paradigm has garnered significant attention with the advancement of foundation models. Although numerous methods have been proposed to reduce the number of trainable parameters, their substantial memory overhead remains a critical bottleneck that hinders practical deployment. In this paper, we observe that model activations constitute a major source of memory consumption, especially under large batch sizes and long context lengths; however, the rank of the activations remains consistently low. Motivated by this insight, we propose a memory-efficient fine-tuning approach Low-Rank Activation Compression (LoRAct). Unlike prior work, LoRAct provides a more flexible and versatile compressing strategy that can be applied online during the forward pass without the need for any calibration data. Moreover, LoRAct incorporates a novel sampling-based orthogonal decomposition algorithm specifically designed for low-rank matrices, offering improved computational efficiency and a tighter error bound compared to the widely used RSVD. Experiments on both vision and language tasks demonstrate the effectiveness of LoRAct. Notably, LoRAct further reduces activation memory by approximately 80% in comparison with the widely adopted LoRA method, while maintaining competitive performance. The source code is available at this https URL. 

---
# Multi-Modal Manipulation via Multi-Modal Policy Consensus 

**Authors**: Haonan Chen, Jiaming Xu, Hongyu Chen, Kaiwen Hong, Binghao Huang, Chaoqi Liu, Jiayuan Mao, Yunzhu Li, Yilun Du, Katherine Driggs-Campbell  

**Link**: [PDF](https://arxiv.org/pdf/2509.23468)  

**Abstract**: Effectively integrating diverse sensory modalities is crucial for robotic manipulation. However, the typical approach of feature concatenation is often suboptimal: dominant modalities such as vision can overwhelm sparse but critical signals like touch in contact-rich tasks, and monolithic architectures cannot flexibly incorporate new or missing modalities without retraining. Our method factorizes the policy into a set of diffusion models, each specialized for a single representation (e.g., vision or touch), and employs a router network that learns consensus weights to adaptively combine their contributions, enabling incremental of new representations. We evaluate our approach on simulated manipulation tasks in {RLBench}, as well as real-world tasks such as occluded object picking, in-hand spoon reorientation, and puzzle insertion, where it significantly outperforms feature-concatenation baselines on scenarios requiring multimodal reasoning. Our policy further demonstrates robustness to physical perturbations and sensor corruption. We further conduct perturbation-based importance analysis, which reveals adaptive shifts between modalities. 

---
# Generative Evolutionary Meta-Solver (GEMS): Scalable Surrogate-Free Multi-Agent Learning 

**Authors**: Alakh Sharma, Gaurish Trivedi, Kartikey Bhandari, Yash Sinha, Dhruv Kumar, Pratik Narang, Jagat Sesh Challa  

**Link**: [PDF](https://arxiv.org/pdf/2509.23462)  

**Abstract**: Scalable multi-agent reinforcement learning (MARL) remains a central challenge for AI. Existing population-based methods, like Policy-Space Response Oracles, PSRO, require storing explicit policy populations and constructing full payoff matrices, incurring quadratic computation and linear memory costs. We present Generative Evolutionary Meta-Solver (GEMS), a surrogate-free framework that replaces explicit populations with a compact set of latent anchors and a single amortized generator. Instead of exhaustively constructing the payoff matrix, GEMS relies on unbiased Monte Carlo rollouts, multiplicative-weights meta-dynamics, and a model-free empirical-Bernstein UCB oracle to adaptively expand the policy set. Best responses are trained within the generator using an advantage-based trust-region objective, eliminating the need to store and train separate actors. We evaluated GEMS in a variety of Two-player and Multi-Player games such as the Deceptive Messages Game, Kuhn Poker and Multi-Particle environment. We find that GEMS is up to ~6x faster, has 1.3x less memory usage than PSRO, while also reaps higher rewards simultaneously. These results demonstrate that GEMS retains the game theoretic guarantees of PSRO, while overcoming its fundamental inefficiencies, hence enabling scalable multi-agent learning in multiple domains. 

---
# Data-Efficient Training by Evolved Sampling 

**Authors**: Ziheng Cheng, Zhong Li, Jiang Bian  

**Link**: [PDF](https://arxiv.org/pdf/2509.23461)  

**Abstract**: Data selection is designed to accelerate learning with preserved performance. To achieve this, a fundamental thought is to identify informative data samples with significant contributions to the training. In this work, we propose \textbf{Evolved Sampling} (\textbf{ES}), a simple yet effective framework for \emph{dynamic} sampling along the training process. This method conducts \em batch \em level data selection based on the dynamics of losses and augmented \emph{loss differences}, which enables flexible \emph{frequency tuning}, and hence significantly reduces the back propagation time with maintained model performance. Due to its conciseness, ES is also readily extensible to incorporate \em set \em level data selection (to form ES with pruning, \textbf{ESWP}) for further accelerations. As a plug-and-play framework, ES(WP) consistently achieves lossless training accelerations across various pre-training and post-training tasks, saving up to nearly 45\% wall-clock time. Our results motivate further investigations on the data efficiency aspect of modern large-scale machine learning. 

---
# AudioFuse: Unified Spectral-Temporal Learning via a Hybrid ViT-1D CNN Architecture for Robust Phonocardiogram Classification 

**Authors**: Md. Saiful Bari Siddiqui, Utsab Saha  

**Link**: [PDF](https://arxiv.org/pdf/2509.23454)  

**Abstract**: Biomedical audio signals, such as phonocardiograms (PCG), are inherently rhythmic and contain diagnostic information in both their spectral (tonal) and temporal domains. Standard 2D spectrograms provide rich spectral features but compromise the phase information and temporal precision of the 1D waveform. We propose AudioFuse, an architecture that simultaneously learns from both complementary representations to classify PCGs. To mitigate the overfitting risk common in fusion models, we integrate a custom, wide-and-shallow Vision Transformer (ViT) for spectrograms with a shallow 1D CNN for raw waveforms. On the PhysioNet 2016 dataset, AudioFuse achieves a state-of-the-art competitive ROC-AUC of 0.8608 when trained from scratch, outperforming its spectrogram (0.8066) and waveform (0.8223) baselines. Moreover, it demonstrates superior robustness to domain shift on the challenging PASCAL dataset, maintaining an ROC-AUC of 0.7181 while the spectrogram baseline collapses (0.4873). Fusing complementary representations thus provides a strong inductive bias, enabling the creation of efficient, generalizable classifiers without requiring large-scale pre-training. 

---
# Factor Decorrelation Enhanced Data Removal from Deep Predictive Models 

**Authors**: Wenhao Yang, Lin Li, Xiaohui Tao, Kaize Shi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23443)  

**Abstract**: The imperative of user privacy protection and regulatory compliance necessitates sensitive data removal in model training, yet this process often induces distributional shifts that undermine model performance-particularly in out-of-distribution (OOD) scenarios. We propose a novel data removal approach that enhances deep predictive models through factor decorrelation and loss perturbation. Our approach introduces: (1) a discriminative-preserving factor decorrelation module employing dynamic adaptive weight adjustment and iterative representation updating to reduce feature redundancy and minimize inter-feature correlations. (2) a smoothed data removal mechanism with loss perturbation that creates information-theoretic safeguards against data leakage during removal operations. Extensive experiments on five benchmark datasets show that our approach outperforms other baselines and consistently achieves high predictive accuracy and robustness even under significant distribution shifts. The results highlight its superior efficiency and adaptability in both in-distribution and out-of-distribution scenarios. 

---
# S$^3$F-Net: A Multi-Modal Approach to Medical Image Classification via Spatial-Spectral Summarizer Fusion Network 

**Authors**: Md. Saiful Bari Siddiqui, Mohammed Imamul Hassan Bhuiyan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23442)  

**Abstract**: Convolutional Neural Networks have become a cornerstone of medical image analysis due to their proficiency in learning hierarchical spatial features. However, this focus on a single domain is inefficient at capturing global, holistic patterns and fails to explicitly model an image's frequency-domain characteristics. To address these challenges, we propose the Spatial-Spectral Summarizer Fusion Network (S$^3$F-Net), a dual-branch framework that learns from both spatial and spectral representations simultaneously. The S$^3$F-Net performs a fusion of a deep spatial CNN with our proposed shallow spectral encoder, SpectraNet. SpectraNet features the proposed SpectralFilter layer, which leverages the Convolution Theorem by applying a bank of learnable filters directly to an image's full Fourier spectrum via a computation-efficient element-wise multiplication. This allows the SpectralFilter layer to attain a global receptive field instantaneously, with its output being distilled by a lightweight summarizer network. We evaluate S$^3$F-Net across four medical imaging datasets spanning different modalities to validate its efficacy and generalizability. Our framework consistently and significantly outperforms its strong spatial-only baseline in all cases, with accuracy improvements of up to 5.13%. With a powerful Bilinear Fusion, S$^3$F-Net achieves a SOTA competitive accuracy of 98.76% on the BRISC2025 dataset. Concatenation Fusion performs better on the texture-dominant Chest X-Ray Pneumonia dataset, achieving 93.11% accuracy, surpassing many top-performing, much deeper models. Our explainability analysis also reveals that the S$^3$F-Net learns to dynamically adjust its reliance on each branch based on the input pathology. These results verify that our dual-domain approach is a powerful and generalizable paradigm for medical image analysis. 

---
# AudioRole: An Audio Dataset for Character Role-Playing in Large Language Models 

**Authors**: Wenyu Li, Xiaoqi Jiao, Yi Chang, Guangyan Zhang, Yiwen Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.23435)  

**Abstract**: The creation of high-quality multimodal datasets remains fundamental for advancing role-playing capabilities in large language models (LLMs). While existing works predominantly focus on text-based persona simulation, Audio Role-Playing (ARP) presents unique challenges due to the need for synchronized alignment of semantic content and vocal characteristics. To address this gap, we propose AudioRole, a meticulously curated dataset from 13 TV series spanning 1K+ hours with 1M+ character-grounded dialogues, providing synchronized audio-text pairs annotated with speaker identities and contextual metadata. In addition, to demonstrate the effectiveness of the dataset, we introduced ARP-Eval, a dual-aspect evaluation framework that assesses both response quality and role fidelity. Empirical validation showing GLM-4-Voice trained on AudioRole (which we called ARP-Model) achieve an average Acoustic Personalization score of 0.31, significantly outperforming the original GLM-4-voice and the more powerful model MiniCPM-O-2.6, which specifically supports role-playing in one-shot scenarios. The ARP-Model also achieves a Content Personalization score of 0.36, surpassing the untrained original model by about 38% and maintaining the same level as MiniCPM-O-2.6.
AudioRole features dialogues from over 115 main characters, 6 trained ARP-Models that role-play different characters, and evaluation protocols. Together, they provide an essential resource for advancing audio-grounded role-playing research. 

---
# NeuroBridge: Using Generative AI to Bridge Cross-neurotype Communication Differences through Neurotypical Perspective-taking 

**Authors**: Rukhshan Haroon, Kyle Wigdor, Katie Yang, Nicole Toumanios, Eileen T. Crehan, Fahad Dogar  

**Link**: [PDF](https://arxiv.org/pdf/2509.23434)  

**Abstract**: Communication challenges between autistic and neurotypical individuals stem from a mutual lack of understanding of each other's distinct, and often contrasting, communication styles. Yet, autistic individuals are expected to adapt to neurotypical norms, making interactions inauthentic and mentally exhausting for them. To help redress this imbalance, we build NeuroBridge, an online platform that utilizes large language models (LLMs) to simulate: (a) an AI character that is direct and literal, a style common among many autistic individuals, and (b) four cross-neurotype communication scenarios in a feedback-driven conversation between this character and a neurotypical user. Through NeuroBridge, neurotypical individuals gain a firsthand look at autistic communication, and reflect on their role in shaping cross-neurotype interactions. In a user study with 12 neurotypical participants, we find that NeuroBridge improved their understanding of how autistic people may interpret language differently, with all describing autism as a social difference that "needs understanding by others" after completing the simulation. Participants valued its personalized, interactive format and described AI-generated feedback as "constructive", "logical" and "non-judgmental". Most perceived the portrayal of autism in the simulation as accurate, suggesting that users may readily accept AI-generated (mis)representations of disabilities. To conclude, we discuss design implications for disability representation in AI, the need for making NeuroBridge more personalized, and LLMs' limitations in modeling complex social scenarios. 

---
# Enhancing Communication Efficiency in FL with Adaptive Gradient Quantization and Communication Frequency Optimization 

**Authors**: Asadullah Tariq, Tariq Qayyum, Mohamed Adel Serhani, Farag Sallabi, Ikbal Taleb, Ezedin S. Barka  

**Link**: [PDF](https://arxiv.org/pdf/2509.23419)  

**Abstract**: Federated Learning (FL) enables participant devices to collaboratively train deep learning models without sharing their data with the server or other devices, effectively addressing data privacy and computational concerns. However, FL faces a major bottleneck due to high communication overhead from frequent model updates between devices and the server, limiting deployment in resource-constrained wireless networks. In this paper, we propose a three-fold strategy. Firstly, an Adaptive Feature-Elimination Strategy to drop less important features while retaining high-value ones; secondly, Adaptive Gradient Innovation and Error Sensitivity-Based Quantization, which dynamically adjusts the quantization level for innovative gradient compression; and thirdly, Communication Frequency Optimization to enhance communication efficiency. We evaluated our proposed model's performance through extensive experiments, assessing accuracy, loss, and convergence compared to baseline techniques. The results show that our model achieves high communication efficiency in the framework while maintaining accuracy. 

---
# Retrieval-Constrained Decoding Reveals Underestimated Parametric Knowledge in Language Models 

**Authors**: Rajaa El Hamdani, Samy Haffoudhi, Nils Holzenberger, Fabian Suchanek, Thomas Bonald, Fragkiskos D. Malliaros  

**Link**: [PDF](https://arxiv.org/pdf/2509.23417)  

**Abstract**: Language models (LMs) encode substantial factual knowledge, but often produce answers judged as incorrect. We hypothesize that many of these answers are actually correct, but are expressed in alternative surface forms that are dismissed due to an overly strict evaluation, leading to an underestimation of models' parametric knowledge. We propose Retrieval-Constrained Decoding (RCD), a decoding strategy that restricts model outputs to unique surface forms. We introduce YAGO-QA, a dataset of 19,137 general knowledge questions. Evaluating open-source LMs from 135M to 70B parameters, we show that standard decoding undervalues their knowledge. For instance, Llama-3.1-70B scores only 32.3% F1 with vanilla decoding but 46.0% with RCD. Similarly, Llama-3.1-8B reaches 33.0% with RCD, outperforming the larger model under vanilla decoding. We publicly share the code and dataset at this https URL. 

---
# Hybrid Graph Embeddings and Louvain Algorithm for Unsupervised Community Detection 

**Authors**: Dalila Khettaf, Djamel Djenouri, Zeinab Rezaeifar, Youcef Djenouri  

**Link**: [PDF](https://arxiv.org/pdf/2509.23411)  

**Abstract**: This paper proposes a novel community detection method that integrates the Louvain algorithm with Graph Neural Networks (GNNs), enabling the discovery of communities without prior knowledge. Compared to most existing solutions, the proposed method does not require prior knowledge of the number of communities. It enhances the Louvain algorithm using node embeddings generated by a GNN to capture richer structural and feature information. Furthermore, it introduces a merging algorithm to refine the results of the enhanced Louvain algorithm, reducing the number of detected communities. To the best of our knowledge, this work is the first one that improves the Louvain algorithm using GNNs for community detection. The improvement of the proposed method was empirically confirmed through an evaluation on real-world datasets. The results demonstrate its ability to dynamically adjust the number of detected communities and increase the detection accuracy in comparison with the benchmark solutions. 

---
# PATCH: Learnable Tile-level Hybrid Sparsity for LLMs 

**Authors**: Younes Hourri, Mohammad Mozaffari, Maryam Mehri Dehnavi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23410)  

**Abstract**: Large language models (LLMs) deliver impressive performance but incur prohibitive memory and compute costs at deployment. Model pruning is an effective way to reduce these overheads, yet existing approaches face challenges: unstructured sparsity, where nonzeros can appear anywhere, preserves accuracy but yields irregular access patterns that prevent GPU acceleration, while semi-structured 2:4 sparsity is hardware-friendly but enforces a rigid 50% pattern that degrades model quality. To bridge this gap, we introduce PATCH, a hybrid sparsity framework that enables a continuous sparsity ratio between 0% and 50%. PATCH partitions weight matrices into tiles, assigning each tile to be either dense or 2:4 sparse via a learnable mask selection mechanism. This design provides fine-grained control over accuracy-acceleration tradeoffs and supports non-uniform sparsity across layers, leading to superior overall quality. Across models from 0.5B to 8B parameters, PATCH consistently narrows the gap to dense accuracy while delivering practical speedups. For instance, on LLaMA-2 7B with an A6000 GPU, PATCH achieves 1.18x-1.38x end-to-end speedup over dense baselines while improving accuracy by 0.37%-2.96% compared to the state-of-the-art 2:4 pruning method, MaskLLM. 

---
# Enhanced Fracture Diagnosis Based on Critical Regional and Scale Aware in YOLO 

**Authors**: Yuyang Sun, Junchuan Yu, Cuiming Zou  

**Link**: [PDF](https://arxiv.org/pdf/2509.23408)  

**Abstract**: Fracture detection plays a critical role in medical imaging analysis, traditional fracture diagnosis relies on visual assessment by experienced physicians, however the speed and accuracy of this approach are constrained by the expertise. With the rapid advancements in artificial intelligence, deep learning models based on the YOLO framework have been widely employed for fracture detection, demonstrating significant potential in improving diagnostic efficiency and accuracy. This study proposes an improved YOLO-based model, termed Fracture-YOLO, which integrates novel Critical-Region-Selector Attention (CRSelector) and Scale-Aware (ScA) heads to further enhance detection performance. Specifically, the CRSelector module utilizes global texture information to focus on critical features of fracture regions. Meanwhile, the ScA module dynamically adjusts the weights of features at different scales, enhancing the model's capacity to identify fracture targets at multiple scales. Experimental results demonstrate that, compared to the baseline model, Fracture-YOLO achieves a significant improvement in detection precision, with mAP50 and mAP50-95 increasing by 4 and 3, surpassing the baseline model and achieving state-of-the-art (SOTA) performance. 

---
# Train Once, Answer All: Many Pretraining Experiments for the Cost of One 

**Authors**: Sebastian Bordt, Martin Pawelczyk  

**Link**: [PDF](https://arxiv.org/pdf/2509.23383)  

**Abstract**: Recent work has demonstrated that controlled pretraining experiments are a powerful tool for understanding learning, reasoning, and memorization in large language models (LLMs). However, the computational cost of pretraining presents a significant constraint. To overcome this constraint, we propose to conduct multiple pretraining experiments simultaneously during a single training run. We demonstrate the feasibility of this approach by conducting ten experiments during the training of a 1.5B parameter model on 210B tokens. Although we only train a single model, we can replicate the results from multiple previous works on data contamination, poisoning, and memorization. We also conduct novel investigations into knowledge acquisition, mathematical reasoning, and watermarking. For example, we dynamically update the training data until the model acquires a particular piece of knowledge. Remarkably, the influence of the ten experiments on the model's training dynamics and overall performance is minimal. However, interactions between different experiments may act as a potential confounder in our approach. We propose to test for interactions with continual pretraining experiments, finding them to be negligible in our setup. Overall, our findings suggest that performing multiple pretraining experiments in a single training run can enable rigorous scientific experimentation with large models on a compute budget. 

---
# CCD: Mitigating Hallucinations in Radiology MLLMs via Clinical Contrastive Decoding 

**Authors**: Xi Zhang, Zaiqiao Meng, Jake Lever, Edmond S. L. Ho  

**Link**: [PDF](https://arxiv.org/pdf/2509.23379)  

**Abstract**: Multimodal large language models (MLLMs) have recently achieved remarkable progress in radiology by integrating visual perception with natural language understanding. However, they often generate clinically unsupported descriptions, known as medical hallucinations, which pose serious risks in medical applications that demand accuracy and image-grounded outputs. Through empirical analysis, we find that prompt-induced hallucinations remain prevalent in radiology MLLMs, largely due to over-sensitivity to clinical sections. To address this, we introduce Clinical Contrastive Cecoding (CCD), a training-free and retrieval-free inference framework that integrates structured clinical signals from task-specific radiology expert models. CCD introduces a dual-stage contrastive mechanism to refine token-level logits during generation, thereby enhancing clinical fidelity without modifying the base MLLM. Experiments on three datasets and multiple models demonstrate that CCD consistently improves overall performance on radiology report generation (RRG). On the MIMIC-CXR dataset, it yields up to a 17% improvement in RadGraph-F1 when applied to state-of-the-art RRG models. Our approach provides a lightweight and generalisable solution for mitigating medical hallucinations, effectively bridging expert models and MLLMs in radiology. 

---
# Graph Your Own Prompt 

**Authors**: Xi Ding, Lei Wang, Piotr Koniusz, Yongsheng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23373)  

**Abstract**: We propose Graph Consistency Regularization (GCR), a novel framework that injects relational graph structures, derived from model predictions, into the learning process to promote class-aware, semantically meaningful feature representations. Functioning as a form of self-prompting, GCR enables the model to refine its internal structure using its own outputs. While deep networks learn rich representations, these often capture noisy inter-class similarities that contradict the model's predicted semantics. GCR addresses this issue by introducing parameter-free Graph Consistency Layers (GCLs) at arbitrary depths. Each GCL builds a batch-level feature similarity graph and aligns it with a global, class-aware masked prediction graph, derived by modulating softmax prediction similarities with intra-class indicators. This alignment enforces that feature-level relationships reflect class-consistent prediction behavior, acting as a semantic regularizer throughout the network. Unlike prior work, GCR introduces a multi-layer, cross-space graph alignment mechanism with adaptive weighting, where layer importance is learned from graph discrepancy magnitudes. This allows the model to prioritize semantically reliable layers and suppress noisy ones, enhancing feature quality without modifying the architecture or training procedure. GCR is model-agnostic, lightweight, and improves semantic structure across various networks and datasets. Experiments show that GCR promotes cleaner feature structure, stronger intra-class cohesion, and improved generalization, offering a new perspective on learning from prediction structure. [Project website](this https URL) [Code](this https URL) 

---
# Alignment through Meta-Weighted Online Sampling: Bridging the Gap between Data Generation and Preference Optimization 

**Authors**: Junming Yang, Ning Xu, Biao Liu, Shiqi Qiao, Xin Geng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23371)  

**Abstract**: Preference optimization is crucial for aligning large language models (LLMs) with human values and intentions. A significant challenge in this process is the distribution mismatch between pre-collected offline preference data and the evolving model policy. Existing methods attempt to reduce this gap using static heuristics or decoupled online sampling strategies, but they often fail to adapt to the model's dynamic learning state. To bridge this gap, we propose Meta-Weighted Adaptive Preference Optimization (MetaAPO), a novel framework that dynamically couples data generation with model training. MetaAPO employs a lightweight meta-learner, as an "alignment gap estimator", to evaluate the potential benefits of on-policy sampling in relation to offline data. This guides targeted online generation and assigns sample-wise meta-weights to the optimization objective, dynamically balancing the quality and distribution of online and offline data. Experiments on AlpacaEval 2, Arena-Hard and MT-Bench demonstrate that MetaAPO consistently outperforms existing preference optimization approaches across various settings, while reducing 42% in online annotation costs. 

---
# MedCritical: Enhancing Medical Reasoning in Small Language Models via Self-Collaborative Correction 

**Authors**: Xinchun Su, Chunxu Luo, Yixuan Li, Weidong Yang, Lipeng Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.23368)  

**Abstract**: In the field of medicine, complex reasoning tasks such as clinical diagnosis, treatment planning, and medical knowledge integration pose significant challenges, where small language models often underperform compared to large language models like GPT-4 and Deepseek. Recent knowledge distillation-based methods aim to address these issues through teacher-guided error correction, but this LLM as judge approach remains challenging in terms of cost, time, and efficiency. To circumvent this issue, we propose a novel two-stage framework, MedCritical, which uses a small language model fine-tuned by a large teacher model to play against itself. In the first stage, we extract high-level and detailed long-chain thought templates from the teacher model to guide the student model to generate more complex reasoning thoughts. In the second stage, we introduce direct preference optimization (DPO) through model self-iteration collaboration to enhance the reasoning ability of the student model by playing against the correction trajectory of the fine-tuned model during training. This model self-learning DPO approach teaches the student model to use its own error-driven insights to consolidate its skills and knowledge to solve complex problems, and achieves comparable results to traditional knowledge distillation methods using teacher models at a lower cost. Notably, our MedCritical 7B model outperforms the Taiyi and Huatuo-o1-7B models by 3.04\% and 10.12\% respectively on the CMExam benchmark, achieving new SOTA performance among 7B-class small models. 

---
# AI Education in Higher Education: A Taxonomy for Curriculum Reform and the Mission of Knowledge 

**Authors**: Tian Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.23363)  

**Abstract**: Artificial intelligence (AI) is reshaping higher education, yet current debates often feel tangled, mixing concerns about pedagogy, operations, curriculum, and the future of work without a shared framework. This paper offers a first attempt at a taxonomy to organize the diverse narratives of AI education and to inform discipline-based curricular discussions. We place these narratives within the enduring responsibility of higher education: the mission of knowledge. This mission includes not only the preservation and advancement of disciplinary expertise, but also the cultivation of skills and wisdom, i.e., forms of meta-knowledge that encompass judgment, ethics, and social responsibility. For the purpose of this paper's discussion, AI is defined as adaptive, data-driven systems that automate analysis, modeling, and decision-making, highlighting its dual role as enabler and disruptor across disciplines. We argue that the most consequential challenges lie at the level of curriculum and disciplinary purpose, where AI accelerates inquiry but also unsettles expertise and identity. We show how disciplines evolve through the interplay of research, curriculum, pedagogy, and faculty expertise, and why curricular reform is the central lever for meaningful change. Pedagogical innovation offers a strategic and accessible entry point, providing actionable steps that help faculty and students build the expertise needed to engage in deeper curricular rethinking and disciplinary renewal. Within this framing, we suggest that meaningful reform can move forward through structured faculty journeys: from AI literacy to pedagogy, curriculum design, and research integration. The key is to align these journeys with the mission of knowledge, turning the disruptive pressures of AI into opportunities for disciplines to sustain expertise, advance inquiry, and serve society. 

---
# Dual-Space Smoothness for Robust and Balanced LLM Unlearning 

**Authors**: Han Yan, Zheyuan Liu, Meng Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23362)  

**Abstract**: With the rapid advancement of large language models, Machine Unlearning has emerged to address growing concerns around user privacy, copyright infringement, and overall safety. Yet state-of-the-art (SOTA) unlearning methods often suffer from catastrophic forgetting and metric imbalance, for example by over-optimizing one objective (e.g., unlearning effectiveness, utility preservation, or privacy protection) at the expense of others. In addition, small perturbations in the representation or parameter space can be exploited by relearn and jailbreak attacks. To address these challenges, we propose PRISM, a unified framework that enforces dual-space smoothness in representation and parameter spaces to improve robustness and balance unlearning metrics. PRISM consists of two smoothness optimization stages: (i) a representation space stage that employs a robustly trained probe to defend against jailbreak attacks, and (ii) a parameter-space stage that decouples retain-forget gradient conflicts, reduces imbalance, and smooths the parameter space to mitigate relearning attacks. Extensive experiments on WMDP and MUSE, across conversational-dialogue and continuous-text settings, show that PRISM outperforms SOTA baselines under multiple attacks while achieving a better balance among key metrics. 

---
# Dynamic-TreeRPO: Breaking the Independent Trajectory Bottleneck with Structured Sampling 

**Authors**: Xiaolong Fu, Lichen Ma, Zipeng Guo, Gaojing Zhou, Chongxiao Wang, ShiPing Dong, Shizhe Zhou, Shizhe Zhou, Ximan Liu, Jingling Fu, Tan Lit Sin, Yu Shi, Zhen Chen, Junshi Huang, Jason Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23352)  

**Abstract**: The integration of Reinforcement Learning (RL) into flow matching models for text-to-image (T2I) generation has driven substantial advances in generation quality. However, these gains often come at the cost of exhaustive exploration and inefficient sampling strategies due to slight variation in the sampling group. Building on this insight, we propose Dynamic-TreeRPO, which implements the sliding-window sampling strategy as a tree-structured search with dynamic noise intensities along depth. We perform GRPO-guided optimization and constrained Stochastic Differential Equation (SDE) sampling within this tree structure. By sharing prefix paths of the tree, our design effectively amortizes the computational overhead of trajectory search. With well-designed noise intensities for each tree layer, Dynamic-TreeRPO can enhance the variation of exploration without any extra computational cost. Furthermore, we seamlessly integrate Supervised Fine-Tuning (SFT) and RL paradigm within Dynamic-TreeRPO to construct our proposed LayerTuning-RL, reformulating the loss function of SFT as a dynamically weighted Progress Reward Model (PRM) rather than a separate pretraining method. By associating this weighted PRM with dynamic-adaptive clipping bounds, the disruption of exploration process in Dynamic-TreeRPO is avoided. Benefiting from the tree-structured sampling and the LayerTuning-RL paradigm, our model dynamically explores a diverse search space along effective directions. Compared to existing baselines, our approach demonstrates significant superiority in terms of semantic consistency, visual fidelity, and human preference alignment on established benchmarks, including HPS-v2.1, PickScore, and ImageReward. In particular, our model outperforms SoTA by $4.9\%$, $5.91\%$, and $8.66\%$ on those benchmarks, respectively, while improving the training efficiency by nearly $50\%$. 

---
# ABC-Eval: Benchmarking Large Language Models on Symbolic Music Understanding and Instruction Following 

**Authors**: Jiahao Zhao, Yunjia Li, Wei Li, Kazuyoshi Yoshii  

**Link**: [PDF](https://arxiv.org/pdf/2509.23350)  

**Abstract**: As large language models continue to develop, the feasibility and significance of text-based symbolic music tasks have become increasingly prominent. While symbolic music has been widely used in generation tasks, LLM capabilities in understanding and reasoning about symbolic music remain largely underexplored. To address this gap, we propose ABC-Eval, the first open-source benchmark dedicated to the understanding and instruction-following capabilities in text-based ABC notation scores. It comprises 1,086 test samples spanning 10 sub-tasks, covering scenarios from basic musical syntax comprehension to complex sequence-level reasoning. Such a diverse scope poses substantial challenges to models' ability to handle symbolic music tasks. We evaluated seven state-of-the-art LLMs on ABC-Eval, and the results reveal notable limitations in existing models' symbolic music processing capabilities. Furthermore, the consistent performance of individual baselines across different sub-tasks supports the reliability of our benchmark. 

---
# DentVLM: A Multimodal Vision-Language Model for Comprehensive Dental Diagnosis and Enhanced Clinical Practice 

**Authors**: Zijie Meng, Jin Hao, Xiwei Dai, Yang Feng, Jiaxiang Liu, Bin Feng, Huikai Wu, Xiaotang Gai, Hengchuan Zhu, Tianxiang Hu, Yangyang Wu, Hongxia Xu, Jin Li, Jun Xiao, Xiaoqiang Liu, Joey Tianyi Zhou, Fudong Zhu, Zhihe Zhao, Lunguo Xia, Bing Fang, Jimeng Sun, Jian Wu, Zuozhu Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23344)  

**Abstract**: Diagnosing and managing oral diseases necessitate advanced visual interpretation across diverse imaging modalities and integrated information synthesis. While current AI models excel at isolated tasks, they often fall short in addressing the complex, multimodal requirements of comprehensive clinical dental practice. Here we introduce DentVLM, a multimodal vision-language model engineered for expert-level oral disease diagnosis. DentVLM was developed using a comprehensive, large-scale, bilingual dataset of 110,447 images and 2.46 million visual question-answering (VQA) pairs. The model is capable of interpreting seven 2D oral imaging modalities across 36 diagnostic tasks, significantly outperforming leading proprietary and open-source models by 19.6% higher accuracy for oral diseases and 27.9% for malocclusions. In a clinical study involving 25 dentists, evaluating 1,946 patients and encompassing 3,105 QA pairs, DentVLM surpassed the diagnostic performance of 13 junior dentists on 21 of 36 tasks and exceeded that of 12 senior dentists on 12 of 36 tasks. When integrated into a collaborative workflow, DentVLM elevated junior dentists' performance to senior levels and reduced diagnostic time for all practitioners by 15-22%. Furthermore, DentVLM exhibited promising performance across three practical utility scenarios, including home-based dental health management, hospital-based intelligent diagnosis and multi-agent collaborative interaction. These findings establish DentVLM as a robust clinical decision support tool, poised to enhance primary dental care, mitigate provider-patient imbalances, and democratize access to specialized medical expertise within the field of dentistry. 

---
# PARROT: A Benchmark for Evaluating LLMs in Cross-System SQL Translation 

**Authors**: Wei Zhou, Guoliang Li, Haoyu Wang, Yuxing Han, Xufei Wu, Fan Wu, Xuanhe Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.23338)  

**Abstract**: Large language models (LLMS) have shown increasing effectiveness in Text-to-SQL tasks. However, another closely related problem, Cross-System SQL Translation (a.k.a., SQL-to-SQL), which adapts a query written for one database system (e.g., MySQL) into its equivalent one for another system (e.g., ClickHouse), is of great practical importance but remains underexplored. Existing SQL benchmarks are not well-suited for SQL-to-SQL evaluation, which (1) focus on a limited set of database systems (often just SQLite) and (2) cannot capture many system-specific SQL dialects (e.g., customized functions, data types, and syntax rules). Thus, in this paper, we introduce PARROT, a Practical And Realistic BenchmaRk for CrOss-System SQL Translation. PARROT comprises 598 translation pairs from 38 open-source benchmarks and real-world business services, specifically prepared to challenge system-specific SQL understanding (e.g., LLMS achieve lower than 38.53% accuracy on average). We also provide multiple benchmark variants, including PARROT-Diverse with 28,003 translations (for extensive syntax testing) and PARROT-Simple with 5,306 representative samples (for focused stress testing), covering 22 production-grade database systems. To promote future research, we release a public leaderboard and source code at: this https URL. 

---
# Space Robotics Bench: Robot Learning Beyond Earth 

**Authors**: Andrej Orsula, Matthieu Geist, Miguel Olivares-Mendez, Carol Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2509.23328)  

**Abstract**: The growing ambition for space exploration demands robust autonomous systems that can operate in unstructured environments under extreme extraterrestrial conditions. The adoption of robot learning in this domain is severely hindered by the prohibitive cost of technology demonstrations and the limited availability of data. To bridge this gap, we introduce the Space Robotics Bench, an open-source simulation framework for robot learning in space. It offers a modular architecture that integrates on-demand procedural generation with massively parallel simulation environments to support the creation of vast and diverse training distributions for learning-based agents. To ground research and enable direct comparison, the framework includes a comprehensive suite of benchmark tasks that span a wide range of mission-relevant scenarios. We establish performance baselines using standard reinforcement learning algorithms and present a series of experimental case studies that investigate key challenges in generalization, end-to-end learning, adaptive control, and sim-to-real transfer. Our results reveal insights into the limitations of current methods and demonstrate the utility of the framework in producing policies capable of real-world operation. These contributions establish the Space Robotics Bench as a valuable resource for developing, benchmarking, and deploying the robust autonomous systems required for the final frontier. 

---
# Robust Fine-Tuning from Non-Robust Pretrained Models: Mitigating Suboptimal Transfer With Adversarial Scheduling 

**Authors**: Jonas Ngnawé, Maxime Heuillet, Sabyasachi Sahoo, Yann Pequignot, Ola Ahmad, Audrey Durand, Frédéric Precioso, Christian Gagné  

**Link**: [PDF](https://arxiv.org/pdf/2509.23325)  

**Abstract**: Fine-tuning pretrained models is a standard and effective workflow in modern machine learning. However, robust fine-tuning (RFT), which aims to simultaneously achieve adaptation to a downstream task and robustness to adversarial examples, remains challenging. Despite the abundance of non-robust pretrained models in open-source repositories, their potential for RFT is less understood. We address this knowledge gap by systematically examining RFT from such non-robust models. Our experiments reveal that fine-tuning non-robust models with a robust objective, even under small perturbations, can lead to poor performance, a phenomenon that we dub \emph{suboptimal transfer}. In challenging scenarios (eg, difficult tasks, high perturbation), the resulting performance can be so low that it may be considered a transfer failure. We find that fine-tuning using a robust objective impedes task adaptation at the beginning of training and eventually prevents optimal transfer. However, we propose a novel heuristic, \emph{Epsilon-Scheduling}, a schedule over perturbation strength used during training that promotes optimal transfer. Additionally, we introduce \emph{expected robustness}, a metric that captures performance across a range of perturbations, providing a more comprehensive evaluation of the accuracy-robustness trade-off for diverse models at test time. Extensive experiments on a wide range of configurations (six pretrained models and five datasets) show that \emph{Epsilon-Scheduling} successfully prevents \emph{suboptimal transfer} and consistently improves expected robustness. 

---
# Scaling LLM Test-Time Compute with Mobile NPU on Smartphones 

**Authors**: Zixu Hao, Jianyu Wei, Tuowei Wang, Minxing Huang, Huiqiang Jiang, Shiqi Jiang, Ting Cao, Ju Ren  

**Link**: [PDF](https://arxiv.org/pdf/2509.23324)  

**Abstract**: Deploying Large Language Models (LLMs) on mobile devices faces the challenge of insufficient performance in smaller models and excessive resource consumption in larger ones. This paper highlights that mobile Neural Processing Units (NPUs) have underutilized computational resources, particularly their matrix multiplication units, during typical LLM inference. To leverage this wasted compute capacity, we propose applying parallel test-time scaling techniques on mobile NPUs to enhance the performance of smaller LLMs. However, this approach confronts inherent NPU challenges, including inadequate hardware support for fine-grained quantization and low efficiency in general-purpose computations. To overcome these, we introduce two key techniques: a hardware-aware tile quantization scheme that aligns group quantization with NPU memory access patterns, and efficient LUT-based replacements for complex operations such as Softmax and dequantization. We design and implement an end-to-end inference system that leverages the NPU's compute capability to support test-time scaling on Qualcomm Snapdragon platforms. Experiments show our approach brings significant speedups: up to 19.0 for mixed-precision GEMM and 2.2 for Softmax. More importantly, we demonstrate that smaller models using test-time scaling can match or exceed the accuracy of larger models, achieving a new performance-cost Pareto frontier. 

---
# MELCOT: A Hybrid Learning Architecture with Marginal Preservation for Matrix-Valued Regression 

**Authors**: Khang Tran, Hieu Cao, Thinh Pham, Nghiem Diep, Tri Cao, Binh Nguyen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23315)  

**Abstract**: Regression is essential across many domains but remains challenging in high-dimensional settings, where existing methods often lose spatial structure or demand heavy storage. In this work, we address the problem of matrix-valued regression, where each sample is naturally represented as a matrix. We propose MELCOT, a hybrid model that integrates a classical machine learning-based Marginal Estimation (ME) block with a deep learning-based Learnable-Cost Optimal Transport (LCOT) block. The ME block estimates data marginals to preserve spatial information, while the LCOT block learns complex global features. This design enables MELCOT to inherit the strengths of both classical and deep learning methods. Extensive experiments across diverse datasets and domains demonstrate that MELCOT consistently outperforms all baselines while remaining highly efficient. 

---
# Seeing Symbols, Missing Cultures: Probing Vision-Language Models' Reasoning on Fire Imagery and Cultural Meaning 

**Authors**: Haorui Yu, Qiufeng Yi, Yijia Chu, Yang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23311)  

**Abstract**: Vision-Language Models (VLMs) often appear culturally competent but rely on superficial pattern matching rather than genuine cultural understanding. We introduce a diagnostic framework to probe VLM reasoning on fire-themed cultural imagery through both classification and explanation analysis. Testing multiple models on Western festivals, non-Western traditions, and emergency scenes reveals systematic biases: models correctly identify prominent Western festivals but struggle with underrepresented cultural events, frequently offering vague labels or dangerously misclassifying emergencies as celebrations. These failures expose the risks of symbolic shortcuts and highlight the need for cultural evaluation beyond accuracy metrics to ensure interpretable and fair multimodal systems. 

---
# A Neural ODE Approach to Aircraft Flight Dynamics Modelling 

**Authors**: Gabriel Jarry, Ramon Dalmau, Xavier Olive, Philippe Very  

**Link**: [PDF](https://arxiv.org/pdf/2509.23307)  

**Abstract**: Accurate aircraft trajectory prediction is critical for air traffic management, airline operations, and environmental assessment. This paper introduces NODE-FDM, a Neural Ordinary Differential Equations-based Flight Dynamics Model trained on Quick Access Recorder (QAR) data. By combining analytical kinematic relations with data-driven components, NODE-FDM achieves a more accurate reproduction of recorded trajectories than state-of-the-art models such as a BADA-based trajectory generation methodology (BADA4 performance model combined with trajectory control routines), particularly in the descent phase of the flight. The analysis demonstrates marked improvements across altitude, speed, and mass dynamics. Despite current limitations, including limited physical constraints and the limited availability of QAR data, the results demonstrate the potential of physics-informed neural ordinary differential equations as a high-fidelity, data-driven approach to aircraft performance modelling. Future work will extend the framework to incorporate a full modelling of the lateral dynamics of the aircraft. 

---
# A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models 

**Authors**: Wonje Jeung, Sangyeon Yoon, Yoonjun Cho, Dongjae Jeon, Sangwoo Shin, Hyesoo Hong, Albert No  

**Link**: [PDF](https://arxiv.org/pdf/2509.23286)  

**Abstract**: Diffusion large language models (dLLMs) enable any-order generation, but this flexibility enlarges the attack surface: harmful spans may appear at arbitrary positions, and template-based prefilling attacks such as DIJA bypass response-level refusals. We introduce A2D (Any-Order, Any-Step Defense), a token-level alignment method that aligns dLLMs to emit an [EOS] refusal signal whenever harmful content arises. By aligning safety directly at the token-level under randomized masking, A2D achieves robustness to both any-decoding-order and any-step prefilling attacks under various conditions. It also enables real-time monitoring: dLLMs may begin a response but automatically terminate if unsafe continuation emerges. On safety benchmarks, A2D consistently prevents the generation of harmful outputs, slashing DIJA success rates from over 80% to near-zero (1.3% on LLaDA-8B-Instruct, 0.0% on Dream-v0-Instruct-7B), and thresholded [EOS] probabilities allow early rejection, yielding up to 19.3x faster safe termination. 

---
# Continuous-Time Reinforcement Learning for Asset-Liability Management 

**Authors**: Yilie Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23280)  

**Abstract**: This paper proposes a novel approach for Asset-Liability Management (ALM) by employing continuous-time Reinforcement Learning (RL) with a linear-quadratic (LQ) formulation that incorporates both interim and terminal objectives. We develop a model-free, policy gradient-based soft actor-critic algorithm tailored to ALM for dynamically synchronizing assets and liabilities. To ensure an effective balance between exploration and exploitation with minimal tuning, we introduce adaptive exploration for the actor and scheduled exploration for the critic. Our empirical study evaluates this approach against two enhanced traditional financial strategies, a model-based continuous-time RL method, and three state-of-the-art RL algorithms. Evaluated across 200 randomized market scenarios, our method achieves higher average rewards than all alternative strategies, with rapid initial gains and sustained superior performance. The outperformance stems not from complex neural networks or improved parameter estimation, but from directly learning the optimal ALM strategy without learning the environment. 

---
# Vid-Freeze: Protecting Images from Malicious Image-to-Video Generation via Temporal Freezing 

**Authors**: Rohit Chowdhury, Aniruddha Bala, Rohan Jaiswal, Siddharth Roheda  

**Link**: [PDF](https://arxiv.org/pdf/2509.23279)  

**Abstract**: The rapid progress of image-to-video (I2V) generation models has introduced significant risks, enabling video synthesis from static images and facilitating deceptive or malicious content creation. While prior defenses such as I2VGuard attempt to immunize images, effective and principled protection to block motion remains underexplored. In this work, we introduce Vid-Freeze - a novel attention-suppressing adversarial attack that adds carefully crafted adversarial perturbations to images. Our method explicitly targets the attention mechanism of I2V models, completely disrupting motion synthesis while preserving semantic fidelity of the input image. The resulting immunized images generate stand-still or near-static videos, effectively blocking malicious content creation. Our experiments demonstrate the impressive protection provided by the proposed approach, highlighting the importance of attention attacks as a promising direction for robust and proactive defenses against misuse of I2V generation models. 

---
# Learning Regional Monsoon Patterns with a Multimodal Attention U-Net 

**Authors**: Swaib Ilias Mazumder, Manish Kumar, Aparajita Khan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23267)  

**Abstract**: Accurate monsoon rainfall prediction is vital for India's agriculture, water management, and climate risk planning, yet remains challenging due to sparse ground observations and complex regional variability. We present a multimodal deep learning framework for high-resolution precipitation classification that leverages satellite and Earth observation data. Unlike previous rainfall prediction models based on coarse 5-50 km grids, we curate a new 1 km resolution dataset for five Indian states, integrating seven key geospatial modalities: land surface temperature, vegetation (NDVI), soil moisture, relative humidity, wind speed, elevation, and land use, covering the June-September 2024 monsoon season. Our approach uses an attention-guided U-Net architecture to capture spatial patterns and temporal dependencies across modalities, combined with focal and dice loss functions to handle rainfall class imbalance defined by the India Meteorological Department (IMD). Experiments demonstrate that our multimodal framework consistently outperforms unimodal baselines and existing deep learning methods, especially in extreme rainfall categories. This work contributes a scalable framework, benchmark dataset, and state-of-the-art results for regional monsoon forecasting, climate resilience, and geospatial AI applications in India. 

---
# Adaptive Token-Weighted Differential Privacy for LLMs: Not All Tokens Require Equal Protection 

**Authors**: Manjiang Yu, Priyanka Singh, Xue Li, Yang Cao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23246)  

**Abstract**: Large language models (LLMs) frequently memorize sensitive or personal information, raising significant privacy concerns. Existing variants of differential privacy stochastic gradient descent (DPSGD) inject uniform noise into every gradient step, significantly extending training time and reducing model accuracy. We propose that concentrating noise primarily on gradients associated with sensitive tokens can substantially decrease DP training time, strengthen the protection of sensitive information, and simultaneously preserve the model's performance on non-sensitive data. We operationalize this insight through Adaptive Token-Weighted Differential Privacy (ATDP), a modification of vanilla DP-SGD that adaptively assigns different gradient weights to sensitive and non-sensitive tokens. By employing a larger noise scale at the early stage of training, ATDP rapidly disrupts memorization of sensitive content. As a result, ATDP only requires a few additional epochs of lightweight post-processing following standard fine-tuning, injecting targeted noise primarily on parameters corresponding to sensitive tokens, thus minimally affecting the model's general capabilities. ATDP can be seamlessly integrated into any existing DP-based fine-tuning pipeline or directly applied to non-private models as a fast privacy-enhancing measure. Additionally, combined with an initial redacted fine-tuning phase, ATDP forms a streamlined DP pipeline that achieves comparable canary protection to state-of-the-art DP-SGD methods, significantly reduces the computational overhead of DP fine-tuning, shortening training time by approximately 90 percent, while achieving comparable or superior privacy protection and minimal accuracy degradation. 

---
# Online Dynamic Goal Recognition in Gym Environments 

**Authors**: Shamir Matan, Elhadad Osher, Nageris Ben, Mirsky Reuth  

**Link**: [PDF](https://arxiv.org/pdf/2509.23244)  

**Abstract**: Goal Recognition (GR) is the task of inferring an agent's intended goal from partial observations of its behavior, typically in an online and one-shot setting. Despite recent advances in model-free GR, particularly in applications such as human-robot interaction, surveillance, and assistive systems, the field remains fragmented due to inconsistencies in benchmarks, domains, and evaluation protocols.
To address this, we introduce gr-libs (this https URL) and gr-envs (this https URL), two complementary open-source frameworks that support the development, evaluation, and comparison of GR algorithms in Gym-compatible environments. gr-libs includes modular implementations of MDP-based GR baselines, diagnostic tools, and evaluation utilities. gr-envs provides a curated suite of environments adapted for dynamic and goal-directed behavior, along with wrappers that ensure compatibility with standard reinforcement learning toolkits. Together, these libraries offer a standardized, extensible, and reproducible platform for advancing GR research. Both packages are open-source and available on GitHub and PyPI. 

---
# Self-Consistency as a Free Lunch: Reducing Hallucinations in Vision-Language Models via Self-Reflection 

**Authors**: Mingfei Han, Haihong Hao, Jinxing Zhou, Zhihui Li, Yuhui Zheng, Xueqing Deng, Linjie Yang, Xiaojun Chang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23236)  

**Abstract**: Vision-language models often hallucinate details, generating non-existent objects or inaccurate attributes that compromise output reliability. Existing methods typically address these issues via extensive human annotations or external supervision from more powerful models. In this work, we present a novel framework that leverages the model's self-consistency between long responses and short answers to generate preference pairs for training. We observe that short binary questions tend to yield highly reliable responses, which can be used to query the target model to evaluate and rank its generated responses. Specifically, we design a self-reflection pipeline where detailed model responses are compared against concise binary answers, and inconsistency signals are utilized to automatically curate high-quality training data without human annotations or external model-based supervision. By relying solely on self-consistency rather than external supervision, our method offers a scalable and efficient solution that effectively reduces hallucinations using unlabeled data. Extensive experiments on multiple benchmarks, i.e., AMBER, MultiObject-Hal (ROPE), Object HalBench, and MMHal-Bench, demonstrate significant improvements in factual grounding and reliability. Moreover, our approach maintains robust instruction-following ability, as evidenced by enhanced performance on LLaVA-Bench and MMBench. 

---
# Patch Rebirth: Toward Fast and Transferable Model Inversion of Vision Transformers 

**Authors**: Seongsoo Heo, Dong-Wan Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23235)  

**Abstract**: Model inversion is a widely adopted technique in data-free learning that reconstructs synthetic inputs from a pretrained model through iterative optimization, without access to original training data. Unfortunately, its application to state-of-the-art Vision Transformers (ViTs) poses a major computational challenge, due to their expensive self-attention mechanisms. To address this, Sparse Model Inversion (SMI) was proposed to improve efficiency by pruning and discarding seemingly unimportant patches, which were even claimed to be obstacles to knowledge transfer. However, our empirical findings suggest the opposite: even randomly selected patches can eventually acquire transferable knowledge through continued inversion. This reveals that discarding any prematurely inverted patches is inefficient, as it suppresses the extraction of class-agnostic features essential for knowledge transfer, along with class-specific features. In this paper, we propose Patch Rebirth Inversion (PRI), a novel approach that incrementally detaches the most important patches during the inversion process to construct sparse synthetic images, while allowing the remaining patches to continue evolving for future selection. This progressive strategy not only improves efficiency, but also encourages initially less informative patches to gradually accumulate more class-relevant knowledge, a phenomenon we refer to as the Re-Birth effect, thereby effectively balancing class-agnostic and class-specific knowledge. Experimental results show that PRI achieves up to 10x faster inversion than standard Dense Model Inversion (DMI) and 2x faster than SMI, while consistently outperforming SMI in accuracy and matching the performance of DMI. 

---
# SPEC-RL: Accelerating On-Policy Reinforcement Learning via Speculative Rollouts 

**Authors**: Bingshuai Liu, Ante Wang, Zijun Min, Liang Yao, Haibo Zhang, Yang Liu, Anxiang Zeng, Jinsong Su  

**Link**: [PDF](https://arxiv.org/pdf/2509.23232)  

**Abstract**: Large Language Models (LLMs) increasingly rely on reinforcement learning with verifiable rewards (RLVR) to elicit reliable chain-of-thought reasoning. However, the training process remains bottlenecked by the computationally expensive rollout stage. Existing acceleration methods-such as parallelization, objective- and data-driven modifications, and replay buffers-either incur diminishing returns, introduce bias, or overlook redundancy across iterations. We identify that rollouts from consecutive training epochs frequently share a large portion of overlapping segments, wasting computation. To address this, we propose SPEC-RL, a novel framework that integrates SPECulative decoding with the RL rollout process. SPEC-RL reuses prior trajectory segments as speculative prefixes and extends them via a draft-and-verify mechanism, avoiding redundant generation while ensuring policy consistency. Experiments on diverse math reasoning and generalization benchmarks, including GSM8K, MATH-500, OlympiadBench, MMLU-STEM, and others, demonstrate that SPEC-RL reduces rollout time by 2-3x without compromising policy quality. As a purely rollout-stage enhancement, SPEC-RL integrates seamlessly with mainstream algorithms (e.g., PPO, GRPO, DAPO), offering a general and practical path to scale RLVR for large reasoning models. Our code is available at this https URL 

---
# Leave No Observation Behind: Real-time Correction for VLA Action Chunks 

**Authors**: Kohei Sendai, Maxime Alvarez, Tatsuya Matsushima, Yutaka Matsuo, Yusuke Iwasawa  

**Link**: [PDF](https://arxiv.org/pdf/2509.23224)  

**Abstract**: To improve efficiency and temporal coherence, Vision-Language-Action (VLA) models often predict action chunks; however, this action chunking harms reactivity under inference delay and long horizons. We introduce Asynchronous Action Chunk Correction (A2C2), which is a lightweight real-time chunk correction head that runs every control step and adds a time-aware correction to any off-the-shelf VLA's action chunk. The module combines the latest observation, the predicted action from VLA (base action), a positional feature that encodes the index of the base action within the chunk, and some features from the base policy, then outputs a per-step correction. This preserves the base model's competence while restoring closed-loop responsiveness. The approach requires no retraining of the base policy and is orthogonal to asynchronous execution schemes such as Real Time Chunking (RTC). On the dynamic Kinetix task suite (12 tasks) and LIBERO Spatial, our method yields consistent success rate improvements across increasing delays and execution horizons (+23% point and +7% point respectively, compared to RTC), and also improves robustness for long horizons even with zero injected delay. Since the correction head is small and fast, there is minimal overhead compared to the inference of large VLA models. These results indicate that A2C2 is an effective, plug-in mechanism for deploying high-capacity chunking policies in real-time control. 

---
# One-Shot Multi-Label Causal Discovery in High-Dimensional Event Sequences 

**Authors**: Hugo Math, Robin Schön, Rainer Lienhart  

**Link**: [PDF](https://arxiv.org/pdf/2509.23213)  

**Abstract**: Understanding causality in event sequences with thousands of sparse event types is critical in domains such as healthcare, cybersecurity, or vehicle diagnostics, yet current methods fail to scale. We present OSCAR, a one-shot causal autoregressive method that infers per-sequence Markov Boundaries using two pretrained Transformers as density estimators. This enables efficient, parallel causal discovery without costly global CI testing. On a real-world automotive dataset with 29,100 events and 474 labels, OSCAR recovers interpretable causal structures in minutes, while classical methods fail to scale, enabling practical scientific diagnostics at production scale. 

---
# Towards Monotonic Improvement in In-Context Reinforcement Learning 

**Authors**: Wenhao Zhang, Shao Zhang, Xihuai Wang, Yang Li, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23209)  

**Abstract**: In-Context Reinforcement Learning (ICRL) has emerged as a promising paradigm for developing agents that can rapidly adapt to new tasks by leveraging past experiences as context, without updating their parameters. Recent approaches train large sequence models on monotonic policy improvement data from online RL, aiming to a continue improved testing time performance. However, our experimental analysis reveals a critical flaw: these models cannot show a continue improvement like the training data during testing time. Theoretically, we identify this phenomenon as Contextual Ambiguity, where the model's own stochastic actions can generate an interaction history that misleadingly resembles that of a sub-optimal policy from the training data, initiating a vicious cycle of poor action selection. To resolve the Contextual Ambiguity, we introduce Context Value into training phase and propose Context Value Informed ICRL (CV-ICRL). CV-ICRL use Context Value as an explicit signal representing the ideal performance theoretically achievable by a policy given the current context. As the context expands, Context Value could include more task-relevant information, and therefore the ideal performance should be non-decreasing. We prove that the Context Value tightens the lower bound on the performance gap relative to an ideal, monotonically improving policy. We fruther propose two methods for estimating Context Value at both training and testing time. Experiments conducted on the Dark Room and Minigrid testbeds demonstrate that CV-ICRL effectively mitigates performance degradation and improves overall ICRL abilities across various tasks and environments. The source code and data of this paper are available at this https URL . 

---
# PARL-MT: Learning to Call Functions in Multi-Turn Conversation with Progress Awareness 

**Authors**: Huacan Chai, Zijie Cao, Maolin Ran, Yingxuan Yang, Jianghao Lin, pengxin, Hairui Wang, Renjie Ding, Ziyu Wan, Muning Wen, Weiwen Liu, Weinan Zhang, Fei Huang, Ying Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.23206)  

**Abstract**: Large language models (LLMs) have achieved impressive success in single-turn function calling, yet real-world applications such as travel planning or multi-stage data analysis typically unfold across multi-turn conversations. In these settings, LLMs must not only issue accurate function calls at each step but also maintain progress awareness, the ability to summarize past interactions and plan future actions to ensure coherent, long-horizon task execution. Existing approaches, however, either reduce multi-turn training to isolated single-turn samples, which neglects task-level planning, or employ end-to-end reinforcement learning (RL) that struggles with redundancy and lacks explicit integration of progress awareness. To overcome these limitations, we introduce PARL-MT, a framework that explicitly incorporates progress awareness into LLM training for multi-turn function calling. PARL-MT combines (i) a Progress Awareness Generation (PAG) pipeline, which automatically constructs datasets coupling conversation summaries with future task planning, and (ii) a Progress Awareness-Guided Reinforcement Learning (PAG-RL) algorithm, which integrates progress awareness into RL training to reduce contextual redundancy and improve alignment between local actions and global task completion. Empirical results on two public benchmarks demonstrate that PARL-MT significantly outperforms existing methods, highlighting the effectiveness of progress awareness in enabling robust and efficient multi-turn function calling. 

---
# WARBERT: A Hierarchical BERT-based Model for Web API Recommendation 

**Authors**: Zishuo Xu, Yuhong Gu, Dezhong Yao  

**Link**: [PDF](https://arxiv.org/pdf/2509.23175)  

**Abstract**: With the emergence of Web 2.0 and microservices architecture, the number of Web APIs has increased dramatically, further intensifying the demand for efficient Web API recommendation. Existing solutions typically fall into two categories: recommendation-type methods, which treat each API as a label for classification, and match-type methods, which focus on matching mashups through API retrieval. However, three critical challenges persist: 1) the semantic ambiguities in comparing API and mashup descriptions, 2) the lack of detailed comparisons between the individual API and the mashup in recommendation-type methods, and 3) time inefficiencies for API retrieval in match-type methods. To address these challenges, we propose WARBERT, a hierarchical BERT-based model for Web API recommendation. WARBERT leverages dual-component feature fusion and attention comparison to extract precise semantic representations of API and mashup descriptions. WARBERT consists of two main components: WARBERT(R) for Recommendation and WARBERT(M) for Matching. Specifically, WAR-BERT(R) serves as an initial filter, narrowing down the candidate APIs, while WARBERT(M) refines the matching process by calculating the similarity between candidate APIs and mashup. The final likelihood of a mashup being matched with an API is determined by combining the predictions from WARBERT(R) and WARBERT(M). Additionally, WARBERT(R) incorporates an auxiliary task of mashup category judgment, which enhances its effectiveness in candidate selection. Experimental results on the ProgrammableWeb dataset demonstrate that WARBERT outperforms most existing solutions and achieves improvements of up to 11.7% compared to the model MTFM (Multi-Task Fusion Model), delivering significant enhancements in accuracy and effiency. 

---
# TRAX: TRacking Axles for Accurate Axle Count Estimation 

**Authors**: Avinash Rai, Sandeep Jana, Vishal Vijay  

**Link**: [PDF](https://arxiv.org/pdf/2509.23171)  

**Abstract**: Accurate counting of vehicle axles is essential for traffic control, toll collection, and infrastructure development. We present an end-to-end, video-based pipeline for axle counting that tackles limitations of previous works in dense environments. Our system leverages a combination of YOLO-OBB to detect and categorize vehicles, and YOLO to detect tires. Detected tires are intelligently associated to their respective parent vehicles, enabling accurate axle prediction even in complex scenarios. However, there are a few challenges in detection when it comes to scenarios with longer and occluded vehicles. We mitigate vehicular occlusions and partial detections for longer vehicles by proposing a novel TRAX (Tire and Axle Tracking) Algorithm to successfully track axle-related features between frames. Our method stands out by significantly reducing false positives and improving the accuracy of axle-counting for long vehicles, demonstrating strong robustness in real-world traffic videos. This work represents a significant step toward scalable, AI-driven axle counting systems, paving the way for machine vision to replace legacy roadside infrastructure. 

---
# Dense associative memory on the Bures-Wasserstein space 

**Authors**: Chandan Tankala, Krishnakumar Balasubramanian  

**Link**: [PDF](https://arxiv.org/pdf/2509.23162)  

**Abstract**: Dense associative memories (DAMs) store and retrieve patterns via energy-functional fixed points, but existing models are limited to vector representations. We extend DAMs to probability distributions equipped with the 2-Wasserstein distance, focusing mainly on the Bures-Wasserstein class of Gaussian densities. Our framework defines a log-sum-exp energy over stored distributions and a retrieval dynamics aggregating optimal transport maps in a Gibbs-weighted manner. Stationary points correspond to self-consistent Wasserstein barycenters, generalizing classical DAM fixed points. We prove exponential storage capacity, provide quantitative retrieval guarantees under Wasserstein perturbations, and validate the model on synthetic and real-world distributional tasks. This work elevates associative memory from vectors to full distributions, bridging classical DAMs with modern generative modeling and enabling distributional storage and retrieval in memory-augmented learning. 

---
# Deep Learning-Based Detection of Cognitive Impairment from Passive Smartphone Sensing with Routine-Aware Augmentation and Demographic Personalization 

**Authors**: Yufei Shen, Ji Hwan Park, Minchao Huang, Jared F. Benge, Justin F. Rousseau, Rosemary A. Lester-Smith, Edison Thomaz  

**Link**: [PDF](https://arxiv.org/pdf/2509.23158)  

**Abstract**: Early detection of cognitive impairment is critical for timely diagnosis and intervention, yet infrequent clinical assessments often lack the sensitivity and temporal resolution to capture subtle cognitive declines in older adults. Passive smartphone sensing has emerged as a promising approach for naturalistic and continuous cognitive monitoring. Building on this potential, we implemented a Long Short-Term Memory (LSTM) model to detect cognitive impairment from sequences of daily behavioral features, derived from multimodal sensing data collected in an ongoing one-year study of older adults. Our key contributions are two techniques to enhance model generalizability across participants: (1) routine-aware augmentation, which generates synthetic sequences by replacing each day with behaviorally similar alternatives, and (2) demographic personalization, which reweights training samples to emphasize those from individuals demographically similar to the test participant. Evaluated on 6-month data from 36 older adults, these techniques jointly improved the Area Under the Precision-Recall Curve (AUPRC) of the model trained on sensing and demographic features from 0.637 to 0.766, highlighting the potential of scalable monitoring of cognitive impairment in aging populations with passive sensing. 

---
# Trust Region Reward Optimization and Proximal Inverse Reward Optimization Algorithm 

**Authors**: Yang Chen, Menglin Zou, Jiaqi Zhang, Yitan Zhang, Junyi Yang, Gael Gendron, Libo Zhang, Jiamou Liu, Michael J. Witbrock  

**Link**: [PDF](https://arxiv.org/pdf/2509.23135)  

**Abstract**: Inverse Reinforcement Learning (IRL) learns a reward function to explain expert demonstrations. Modern IRL methods often use the adversarial (minimax) formulation that alternates between reward and policy optimization, which often lead to unstable training. Recent non-adversarial IRL approaches improve stability by jointly learning reward and policy via energy-based formulations but lack formal guarantees. This work bridges this gap. We first present a unified view showing canonical non-adversarial methods explicitly or implicitly maximize the likelihood of expert behavior, which is equivalent to minimizing the expected return gap. This insight leads to our main contribution: Trust Region Reward Optimization (TRRO), a framework that guarantees monotonic improvement in this likelihood via a Minorization-Maximization process. We instantiate TRRO into Proximal Inverse Reward Optimization (PIRO), a practical and stable IRL algorithm. Theoretically, TRRO provides the IRL counterpart to the stability guarantees of Trust Region Policy Optimization (TRPO) in forward RL. Empirically, PIRO matches or surpasses state-of-the-art baselines in reward recovery, policy imitation with high sample efficiency on MuJoCo and Gym-Robotics benchmarks and a real-world animal behavior modeling task. 

---
# C$^2$GSPG: Confidence-calibrated Group Sequence Policy Gradient towards Self-aware Reasoning 

**Authors**: Haotian Liu, Shuo Wang, Hongteng Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23129)  

**Abstract**: Reinforcement Learning (RL) methods, exemplified by Group Relative Policy Optimization (GRPO) and its variants, play a central role in developing reasoning models. However, these methods often suffer from a critical overconfidence issue, which prevents them from achieving self-aware reasoning models. In this study, we propose a simple yet effective confidence-calibration group sequence policy gradient method, called C$^2$GSPG, which simultaneously enhances reasoning performance while suppressing overconfidence. In principle, we propose a Group Sequence Policy Gradient (GSPG) framework for learning reasoning models, which eliminates the token-level bias commonly appearing in GRPO and its variants. In this framework, we define the model confidence for each reasoning problem using the normalized sequence-level probability, and then apply a cross-entropy regularizer to calibrate the model confidence to the sequence's reward. We demonstrate that the confidence calibration regularizer and GSPG are collaborative for binary rewards, as their objectives always share the same gradient direction. For non-binary rewards, we apply nonlinear reward normalization and adaptive regularizer clipping, mitigating the potential conflict between the two objectives. Applying C$^2$GSPG to post-train large language models in logical and mathematical reasoning tasks, we show its superiority over state-of-the-art methods in both reasoning accuracy and confidence calibration. The code of C$^2$GSPG is available at this https URL. 

---
# RHYTHM: Reasoning with Hierarchical Temporal Tokenization for Human Mobility 

**Authors**: Haoyu He, Haozheng Luo, Yan Chen, Qi R. Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23115)  

**Abstract**: Predicting human mobility is inherently challenging due to complex long-range dependencies and multi-scale periodic behaviors. To address this, we introduce RHYTHM (Reasoning with Hierarchical Temporal Tokenization for Human Mobility), a unified framework that leverages large language models (LLMs) as general-purpose spatio-temporal predictors and trajectory reasoners. Methodologically, RHYTHM employs temporal tokenization to partition each trajectory into daily segments and encode them as discrete tokens with hierarchical attention that captures both daily and weekly dependencies, thereby significantly reducing the sequence length while preserving cyclical information. Additionally, we enrich token representations by adding pre-computed prompt embeddings for trajectory segments and prediction targets via a frozen LLM, and feeding these combined embeddings back into the LLM backbone to capture complex interdependencies. Computationally, RHYTHM freezes the pretrained LLM's backbone to reduce attention complexity and memory cost. We evaluate our model against state-of-the-art methods using three real-world datasets. Notably, RHYTHM achieves a 2.4% improvement in overall accuracy, a 5.0% increase on weekends, and a 24.6% reduction in training time. Code is publicly available at this https URL. 

---
# Liaohe-CobotMagic-PnP: an Imitation Learning Dataset of Intelligent Robot for Industrial Applications 

**Authors**: Chen Yizhe, Wang Qi, Hu Dongxiao, Jingzhe Fang, Liu Sichao, Zixin An, Hongliang Niu, Haoran Liu, Li Dong, Chuanfen Feng, Lan Dapeng, Liu Yu, Zhibo Pang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23111)  

**Abstract**: In Industry 4.0 applications, dynamic environmental interference induces highly nonlinear and strongly coupled interactions between the environmental state and robotic behavior. Effectively representing dynamic environmental states through multimodal sensor data fusion remains a critical challenge in current robotic datasets. To address this, an industrial-grade multimodal interference dataset is presented, designed for robotic perception and control under complex conditions. The dataset integrates multi-dimensional interference features including size, color, and lighting variations, and employs high-precision sensors to synchronously collect visual, torque, and joint-state measurements. Scenarios with geometric similarity exceeding 85\% and standardized lighting gradients are included to ensure real-world representativeness. Microsecond-level time-synchronization and vibration-resistant data acquisition protocols, implemented via the Robot Operating System (ROS), guarantee temporal and operational fidelity. Experimental results demonstrate that the dataset enhances model validation robustness and improves robotic operational stability in dynamic, interference-rich environments. The dataset is publicly available at:this https URL. 

---
# Open-Vocabulary Spatio-Temporal Scene Graph for Robot Perception and Teleoperation Planning 

**Authors**: Yi Wang, Zeyu Xue, Mujie Liu, Tongqin Zhang, Yan Hu, Zhou Zhao, Chenguang Yang, Zhenyu Lu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23107)  

**Abstract**: Teleoperation via natural-language reduces operator workload and enhances safety in high-risk or remote settings. However, in dynamic remote scenes, transmission latency during bidirectional communication creates gaps between remote perceived states and operator intent, leading to command misunderstanding and incorrect execution. To mitigate this, we introduce the Spatio-Temporal Open-Vocabulary Scene Graph (ST-OVSG), a representation that enriches open-vocabulary perception with temporal dynamics and lightweight latency annotations. ST-OVSG leverages LVLMs to construct open-vocabulary 3D object representations, and extends them into the temporal domain via Hungarian assignment with our temporal matching cost, yielding a unified spatio-temporal scene graph. A latency tag is embedded to enable LVLM planners to retrospectively query past scene states, thereby resolving local-remote state mismatches caused by transmission delays. To further reduce redundancy and highlight task-relevant cues, we propose a task-oriented subgraph filtering strategy that produces compact inputs for the planner. ST-OVSG generalizes to novel categories and enhances planning robustness against transmission latency without requiring fine-tuning. Experiments show that our method achieves 74 percent node accuracy on the Replica benchmark, outperforming ConceptGraph. Notably, in the latency-robustness experiment, the LVLM planner assisted by ST-OVSG achieved a planning success rate of 70.5 percent. 

---
# HTMA-Net: Towards Multiplication-Avoiding Neural Networks via Hadamard Transform and In-Memory Computing 

**Authors**: Emadeldeen Hamdan, Ahmet Enis Cetin  

**Link**: [PDF](https://arxiv.org/pdf/2509.23103)  

**Abstract**: Reducing the cost of multiplications is critical for efficient deep neural network deployment, especially in energy-constrained edge devices. In this work, we introduce HTMA-Net, a novel framework that integrates the Hadamard Transform (HT) with multiplication-avoiding (MA) SRAM-based in-memory computing to reduce arithmetic complexity while maintaining accuracy. Unlike prior methods that only target multiplications in convolutional layers or focus solely on in-memory acceleration, HTMA-Net selectively replaces intermediate convolutions with Hybrid Hadamard-based transform layers whose internal convolutions are implemented via multiplication-avoiding in-memory operations. We evaluate HTMA-Net on ResNet-18 using CIFAR-10, CIFAR-100, and Tiny ImageNet, and provide a detailed comparison against regular, MF-only, and HT-only variants. Results show that HTMA-Net eliminates up to 52\% of multiplications compared to baseline ResNet-18, ResNet-20, and ResNet-50 models, while achieving comparable accuracy in evaluation and significantly reducing computational complexity and the number of parameters. Our results demonstrate that combining structured Hadamard transform layers with SRAM-based in-memory computing multiplication-avoiding operators is a promising path towards efficient deep learning architectures. 

---
# Towards Quantum-Ready Blockchain Fraud Detection via Ensemble Graph Neural Networks 

**Authors**: M.Z. Haider, Tayyaba Noreen, M. Salman  

**Link**: [PDF](https://arxiv.org/pdf/2509.23101)  

**Abstract**: Blockchain Business applications and cryptocurrencies such as enable secure, decentralized value transfer, yet their pseudonymous nature creates opportunities for illicit activity, challenging regulators and exchanges in anti money laundering (AML) enforcement. Detecting fraudulent transactions in blockchain networks requires models that can capture both structural and temporal dependencies while remaining resilient to noise, imbalance, and adversarial behavior. In this work, we propose an ensemble framework that integrates Graph Convolutional Networks (GCN), Graph Attention Networks (GAT), and Graph Isomorphism Networks (GIN) to enhance blockchain fraud detection. Using the real-world Elliptic dataset, our tuned soft voting ensemble achieves high recall of illicit transactions while maintaining a false positive rate below 1%, beating individual GNN models and baseline methods. The modular architecture incorporates quantum-ready design hooks, allowing seamless future integration of quantum feature mappings and hybrid quantum classical graph neural networks. This ensures scalability, robustness, and long-term adaptability as quantum computing technologies mature. Our findings highlight ensemble GNNs as a practical and forward-looking solution for real-time cryptocurrency monitoring, providing both immediate AML utility and a pathway toward quantum-enhanced financial security analytics. 

---
# CoPatch: Zero-Shot Referring Image Segmentation by Leveraging Untapped Spatial Knowledge in CLIP 

**Authors**: Na Min An, Inha Kang, Minhyun Lee, Hyunjung Shim  

**Link**: [PDF](https://arxiv.org/pdf/2509.23098)  

**Abstract**: Spatial grounding is crucial for referring image segmentation (RIS), where the goal of the task is to localize an object described by language. Current foundational vision-language models (VLMs), such as CLIP, excel at aligning images and text but struggle with understanding spatial relationships. Within the language stream, most existing methods often focus on the primary noun phrase when extracting local text features, undermining contextual tokens. Within the vision stream, CLIP generates similar features for images with different spatial layouts, resulting in limited sensitivity to spatial structure. To address these limitations, we propose \textsc{CoPatch}, a zero-shot RIS framework that leverages internal model components to enhance spatial representations in both text and image modalities. For language, \textsc{CoPatch} constructs hybrid text features by incorporating context tokens carrying spatial cues. For vision, it extracts patch-level image features using our novel path discovered from intermediate layers, where spatial structure is better preserved. These enhanced features are fused into a clustered image-text similarity map, \texttt{CoMap}, enabling precise mask selection. As a result, \textsc{CoPatch} significantly improves spatial grounding in zero-shot RIS across RefCOCO, RefCOCO+, RefCOCOg, and PhraseCut (+ 2--7 mIoU) without requiring any additional training. Our findings underscore the importance of recovering and leveraging the untapped spatial knowledge inherently embedded in VLMs, thereby paving the way for opportunities in zero-shot RIS. 

---
# Causally-Enhanced Reinforcement Policy Optimization 

**Authors**: Xiangqi Wang, Yue Huang, Yujun Zhou, Xiaonan Luo, Kehan Guo, Xiangliang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23095)  

**Abstract**: Large language models (LLMs) trained with reinforcement objectives often achieve superficially correct answers via shortcut strategies, pairing correct outputs with spurious or unfaithful reasoning and degrading under small causal perturbations. We introduce Causally-Enhanced Policy Optimization (CE-PO), a drop-in reward-shaping framework that augments policy optimization with a differentiable proxy for causal coherence along the generation pathway from prompt (Z) to rationale (X) to answer (Y). CE-PO estimates model-internal influence with Jacobian-based sensitivities, counterfactually hardens these signals to suppress nuisance cues, and fuses the resulting coherence score with task-accuracy feedback via a Minkowski (power-mean) combiner, exposing a single tunable between accuracy and coherence trade-off. The unified reward integrates with PPO/GRPO without architectural changes. Across reasoning benchmarks and causal stress tests, CE-PO reduces reward hacking and unfaithful chain-of-thought while improving robustness to correlation-causation flips and light counterfactual edits, all at near-parity accuracy. Experimental results across 4 datasets show that CE-PO improves accuracy over baselines by 5.49% on average (up to 9.58%), while improving robustness to correlation-causation flips and light counterfactual edits. 

---
# Signal Preserving Weight Initialization for Odd-Sigmoid Activations 

**Authors**: Hyunwoo Lee, Hayoung Choi, Hyunju Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.23085)  

**Abstract**: Activation functions critically influence trainability and expressivity, and recent work has therefore explored a broad range of nonlinearities. However, activations and weight initialization are interdependent: without an appropriate initialization method, nonlinearities can cause saturation, variance collapse, and increased learning rate sensitivity. We address this by defining an odd sigmoid function class and, given any activation f in this class, proposing an initialization method tailored to f. The method selects a noise scale in closed form so that forward activations remain well dispersed up to a target layer, thereby avoiding collapse to zero or saturation. Empirically, the approach trains reliably without normalization layers, exhibits strong data efficiency, and enables learning for activations under which standard initialization methods (Xavier, He, Orthogonal) often do not converge reliably. 

---
# Beyond Model Ranking: Predictability-Aligned Evaluation for Time Series Forecasting 

**Authors**: Wanjin Feng, Yuan Yuan, Jingtao Ding, Yong Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23074)  

**Abstract**: In the era of increasingly complex AI models for time series forecasting, progress is often measured by marginal improvements on benchmark leaderboards. However, this approach suffers from a fundamental flaw: standard evaluation metrics conflate a model's performance with the data's intrinsic unpredictability. To address this pressing challenge, we introduce a novel, predictability-aligned diagnostic framework grounded in spectral coherence. Our framework makes two primary contributions: the Spectral Coherence Predictability (SCP), a computationally efficient ($O(N\log N)$) and task-aligned score that quantifies the inherent difficulty of a given forecasting instance, and the Linear Utilization Ratio (LUR), a frequency-resolved diagnostic tool that precisely measures how effectively a model exploits the linearly predictable information within the data. We validate our framework's effectiveness and leverage it to reveal two core insights. First, we provide the first systematic evidence of "predictability drift", demonstrating that a task's forecasting difficulty varies sharply over time. Second, our evaluation reveals a key architectural trade-off: complex models are superior for low-predictability data, whereas linear models are highly effective on more predictable tasks. We advocate for a paradigm shift, moving beyond simplistic aggregate scores toward a more insightful, predictability-aware evaluation that fosters fairer model comparisons and a deeper understanding of model behavior. 

---
# From Evidence to Trajectory: Abductive Reasoning Path Synthesis for Training Retrieval-Augmented Generation Agents 

**Authors**: Muzhi Li, Jinhu Qi, Yihong Wu, Minghao Zhao, Liheng Ma, Yifan Li, Xinyu Wang, Yingxue Zhang, Ho-fung Leung, Irwin King  

**Link**: [PDF](https://arxiv.org/pdf/2509.23071)  

**Abstract**: Retrieval-augmented generation agents development is hindered by the lack of process-level supervision to effectively guide agentic capabilities like task decomposition, retriever invocation, and stepwise decision-making. While reinforcement learning offers a potential solution, it suffers from sparse rewards and the limited reasoning capabilities of large language models (LLMs). Meanwhile, existing data synthesis methods only produce chain-of-thought rationales and fail to model environmental interactions. In this paper, we propose EviPath, an evidence-anchored reasoning path synthesis paradigm for RAG agent development. EviPath comprises: (i) Abductive Subtask Planning, which decomposes the problem into sub-questions and iteratively plans an optimal solution path based on the dependencies between them; (ii) Faithful Sub-question Answering, which uses supporting evidence to construct a proxy environment to generate reasoning thoughts and answers for each sub-question; and (iii) Conversational Fine-Tuning, which formats the complete agent-environment interaction trajectory into a dialogue format suitable for Supervised Fine-Tuning. EviPath allows LLMs to learn complex reasoning and tool-use capabilities directly from synthesized data. Extensive experiments on widely-used question-answering benchmarks show that an 8B parameter model trained with EviPath-synthesized data significantly and consistently outperforms state-of-the-art baselines with a double-digit absolute EM gain of 14.7% in open-domain question answering. 

---
# Semantic Voting: A Self-Evaluation-Free Approach for Efficient LLM Self-Improvement on Unverifiable Open-ended Tasks 

**Authors**: Chunyang Jiang, Yonggang Zhang, Yiyang Cai, Chi-Min Chan, Yulong Liu, Mingming Chen, Wei Xue, Yike Guo  

**Link**: [PDF](https://arxiv.org/pdf/2509.23067)  

**Abstract**: The rising cost of acquiring supervised data has driven significant interest in self-improvement for large language models (LLMs). Straightforward unsupervised signals like majority voting have proven effective in generating pseudo-labels for verifiable tasks, while their applicability to unverifiable tasks (e.g., translation) is limited by the open-ended character of responses. As a result, self-evaluation mechanisms (e.g., self-judging and entropy minimization) are predominantly used to derive pseudo-labels. However, self-evaluation relying on LLMs typically incurs high computational overhead and introduces overconfidence issues due to intrinsic biases. To address these challenges, we propose a novel self-evaluation-free approach for unverifiable tasks, designed for lightweight yet effective self-improvement. Inspired by majority voting commonly employed in verifiable tasks, we propose semantic voting as a novel mechanism that relaxes the principle of hard matching (i.e., exact matching) toward soft matching (i.e., semantic similarity). Soft matching is achieved by leveraging a lightweight sentence embedding model to quantify semantic similarity, thereby mitigating excessive computational burden and intrinsic bias-associated limitations of self-evaluation. Comprehensive experiments demonstrate that our method achieves substantial gains in computational efficiency and overall better performance than self-evaluation methods across diverse model architectures and tasks. 

---
# Local Success Does Not Compose: Benchmarking Large Language Models for Compositional Formal Verification 

**Authors**: Xu Xu, Xin Li, Xingwei Qu, Jie Fu, Binhang Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23061)  

**Abstract**: We introduce DafnyCOMP, a benchmark for evaluating large language models (LLMs) on compositional specification generation in Dafny. Unlike prior benchmarks that focus on single-function tasks, DafnyCOMP targets programs composed of multiple interacting functions with data dependencies, requiring reasoning across component boundaries. The benchmark consists of 300 automatically synthesized multi-function programs. We evaluate several state-of-the-art LLM families and find that, while they perform well on single-function verification, their performance drops sharply on compositional tasks. Analysis reveals systematic failures in cross-functional reasoning, including fragile specifications, misalignment between implementations and proofs, and unstable reasoning. DafnyCOMP thus provides a diagnostic tool for measuring progress toward reliable, verifiable, and compositional code generation with LLMs. 

---
# Understanding Language Prior of LVLMs by Contrasting Chain-of-Embedding 

**Authors**: Lin Long, Changdae Oh, Seongheon Park, Yixuan Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23050)  

**Abstract**: Large vision-language models (LVLMs) achieve strong performance on multimodal tasks, yet they often default to their language prior (LP) -- memorized textual patterns from pre-training while under-utilizing visual evidence. Prior analyses of LP mostly rely on input-output probing, which fails to reveal the internal mechanisms governing when and how vision influences model behavior. To address this gap, we present the first systematic analysis of language prior through the lens of chain-of-embedding, which examines the layer-wise representation dynamics within LVLMs. Our analysis reveals a universal phenomenon: each model exhibits a Visual Integration Point (VIP), a critical layer at which visual information begins to meaningfully reshape hidden representations and influence decoding. Building on this observation, we introduce the Total Visual Integration (TVI) estimator, which aggregates representation distance beyond the VIP to quantify how strongly visual query influences response generation. Across 54 model-dataset combinations spanning 9 contemporary LVLMs and 6 benchmarks, we demonstrate that VIP consistently emerges, and that TVI reliably predicts the strength of language prior. This offers a principled toolkit for diagnosing and understanding language prior in LVLMs. 

---
# Beyond Aggregation: Guiding Clients in Heterogeneous Federated Learning 

**Authors**: Zijian Wang, Xiaofei Zhang, Xin Zhang, Yukun Liu, Qiong Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23049)  

**Abstract**: Federated learning (FL) is increasingly adopted in domains like healthcare, where data privacy is paramount. A fundamental challenge in these systems is statistical heterogeneity-the fact that data distributions vary significantly across clients (e.g., different hospitals may treat distinct patient demographics). While current FL algorithms focus on aggregating model updates from these heterogeneous clients, the potential of the central server remains under-explored. This paper is motivated by a healthcare scenario: could a central server not only build a model but also guide a new patient to the hospital best equipped for their specific condition? We generalize this idea to propose a novel paradigm for FL systems where the server actively guides the allocation of new tasks or queries to the most appropriate client in the network. To enable this, we introduce an empirical likelihood-based framework that simultaneously addresses two goals: (1) learning effective local models on each client, and (2) finding the best matching client for a new query. Empirical results demonstrate the framework's effectiveness on benchmark datasets, showing improvements in both model accuracy and the precision of client guidance compared to standard FL approaches. This work opens a new direction for building more intelligent and resource-efficient federated systems that leverage heterogeneity as a feature, not just a bug. Code is available at this https URL. 

---
# MMeViT: Multi-Modal ensemble ViT for Post-Stroke Rehabilitation Action Recognition 

**Authors**: Ye-eun Kim, Suhyeon Lim, Andrew J. Choi  

**Link**: [PDF](https://arxiv.org/pdf/2509.23044)  

**Abstract**: Rehabilitation therapy for stroke patients faces a supply shortage despite the increasing demand. To address this issue, remote monitoring systems that reduce the burden on medical staff are emerging as a viable alternative. A key component of these remote monitoring systems is Human Action Recognition (HAR) technology, which classifies actions. However, existing HAR studies have primarily focused on non-disable individuals, making them unsuitable for recognizing the actions of stroke patients. HAR research for stroke has largely concentrated on classifying relatively simple actions using machine learning rather than deep learning. In this study, we designed a system to monitor the actions of stroke patients, focusing on domiciliary upper limb Activities of Daily Living (ADL). Our system utilizes IMU (Inertial Measurement Unit) sensors and an RGB-D camera, which are the most common modalities in HAR. We directly collected a dataset through this system, investigated an appropriate preprocess and proposed a deep learning model suitable for processing multimodal data. We analyzed the collected dataset and found that the action data of stroke patients is less clustering than that of non-disabled individuals. Simultaneously, we found that the proposed model learns similar tendencies for each label in data with features that are difficult to clustering. This study suggests the possibility of expanding the deep learning model, which has learned the action features of stroke patients, to not only simple action recognition but also feedback such as assessment contributing to domiciliary rehabilitation in future research. The code presented in this study is available at this https URL. 

---
# IsingFormer: Augmenting Parallel Tempering With Learned Proposals 

**Authors**: Saleh Bunaiyan, Corentin Delacour, Shuvro Chowdhury, Kyle Lee, Kerem Y. Camsari  

**Link**: [PDF](https://arxiv.org/pdf/2509.23043)  

**Abstract**: Markov Chain Monte Carlo (MCMC) underlies both statistical physics and combinatorial optimization, but mixes slowly near critical points and in rough landscapes. Parallel Tempering (PT) improves mixing by swapping replicas across temperatures, yet each replica still relies on slow local updates to change its configuration. We introduce IsingFormer, a Transformer trained on equilibrium samples that can generate entire spin configurations resembling those from the target distribution. These uncorrelated samples are used as proposals for global moves within a Metropolis step in PT, complementing the usual single-spin flips. On 2D Ising models (sampling), IsingFormer reproduces magnetization and free-energy curves and generalizes to unseen temperatures, including the critical region. Injecting even a single proposal sharply reduces equilibration time, replacing thousands of local updates. On 3D spin glasses (optimization), PT enhanced with IsingFormer finds substantially lower-energy states, demonstrating how global moves accelerate search in rugged landscapes. Finally, applied to integer factorization encoded as Ising problems, IsingFormer trained on a limited set of semiprimes transfers successfully to unseen semiprimes, boosting success rates beyond the training distribution. Since factorization is a canonical hard benchmark, this ability to generalize across instances highlights the potential of learning proposals that move beyond single problems to entire families of instances. The IsingFormer demonstrates that Monte Carlo methods can be systematically accelerated by neural proposals that capture global structure, yielding faster sampling and stronger performance in combinatorial optimization. 

---
# Virus Infection Attack on LLMs: Your Poisoning Can Spread "VIA" Synthetic Data 

**Authors**: Zi Liang, Qingqing Ye, Xuan Liu, Yanyun Wang, Jianliang Xu, Haibo Hu  

**Link**: [PDF](https://arxiv.org/pdf/2509.23041)  

**Abstract**: Synthetic data refers to artificial samples generated by models. While it has been validated to significantly enhance the performance of large language models (LLMs) during training and has been widely adopted in LLM development, potential security risks it may introduce remain uninvestigated. This paper systematically evaluates the resilience of synthetic-data-integrated training paradigm for LLMs against mainstream poisoning and backdoor attacks. We reveal that such a paradigm exhibits strong resistance to existing attacks, primarily thanks to the different distribution patterns between poisoning data and queries used to generate synthetic samples. To enhance the effectiveness of these attacks and further investigate the security risks introduced by synthetic data, we introduce a novel and universal attack framework, namely, Virus Infection Attack (VIA), which enables the propagation of current attacks through synthetic data even under purely clean queries. Inspired by the principles of virus design in cybersecurity, VIA conceals the poisoning payload within a protective "shell" and strategically searches for optimal hijacking points in benign samples to maximize the likelihood of generating malicious content. Extensive experiments on both data poisoning and backdoor attacks show that VIA significantly increases the presence of poisoning content in synthetic data and correspondingly raises the attack success rate (ASR) on downstream models to levels comparable to those observed in the poisoned upstream models. 

---
# Look Back to Reason Forward: Revisitable Memory for Long-Context LLM Agents 

**Authors**: Yaorui Shi, Yuxin Chen, Siyuan Wang, Sihang Li, Hengxing Cai, Qi Gu, Xiang Wang, An Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.23040)  

**Abstract**: Large language models face challenges in long-context question answering, where key evidence of a query may be dispersed across millions of tokens. Existing works equip large language models with a memory corpus that is dynamically updated during a single-pass document scan, also known as the "memorize while reading" methods. While this approach scales efficiently, it suffers from irreversible forward-only processing, information loss through overwriting, and sparse reinforcement learning signals. To tackle these challenges, we present ReMemR1, a memory-augmented agent with callback-enhanced memory that allows selective retrieval from the entire memory history and allows non-linear reasoning and revisiting of early evidence. To further strengthen training, we propose Reinforcement Learning with Multi-Level Rewards (RLMLR), which combines final-answer rewards with dense, step-level signals that guide effective memory use. Together, these contributions mitigate information degradation, improve supervision, and support multi-hop memory utilizing. Experiments on long-document QA show significant gains over existing memory-based approaches, which validates ReMemR1 as an effective solution for long-context reasoning agents. 

---
# GeLoc3r: Enhancing Relative Camera Pose Regression with Geometric Consistency Regularization 

**Authors**: Jingxing Li, Yongjae Lee, Deliang Fan  

**Link**: [PDF](https://arxiv.org/pdf/2509.23038)  

**Abstract**: Prior ReLoc3R achieves breakthrough performance with fast 25ms inference and state-of-the-art regression accuracy, yet our analysis reveals subtle geometric inconsistencies in its internal representations that prevent reaching the precision ceiling of correspondence-based methods like MASt3R (which require 300ms per pair). In this work, we present GeLoc3r, a novel approach to relative camera pose estimation that enhances pose regression methods through Geometric Consistency Regularization (GCR). GeLoc3r overcomes the speed-accuracy dilemma by training regression networks to produce geometrically consistent poses without inference-time geometric computation. During training, GeLoc3r leverages ground-truth depth to generate dense 3D-2D correspondences, weights them using a FusionTransformer that learns correspondence importance, and computes geometrically-consistent poses via weighted RANSAC. This creates a consistency loss that transfers geometric knowledge into the regression network. Unlike FAR method which requires both regression and geometric solving at inference, GeLoc3r only uses the enhanced regression head at test time, maintaining ReLoc3R's fast speed and approaching MASt3R's high accuracy. On challenging benchmarks, GeLoc3r consistently outperforms ReLoc3R, achieving significant improvements including 40.45% vs. 34.85% AUC@5° on the CO3Dv2 dataset (16% relative improvement), 68.66% vs. 66.70% AUC@5° on RealEstate10K, and 50.45% vs. 49.60% on MegaDepth1500. By teaching geometric consistency during training rather than enforcing it at inference, GeLoc3r represents a paradigm shift in how neural networks learn camera geometry, achieving both the speed of regression and the geometric understanding of correspondence methods. 

---
# Sensor-Adaptive Flood Mapping with Pre-trained Multi-Modal Transformers across SAR and Multispectral Modalities 

**Authors**: Tomohiro Tanaka, Narumasa Tsutsumida  

**Link**: [PDF](https://arxiv.org/pdf/2509.23035)  

**Abstract**: Floods are increasingly frequent natural disasters causing extensive human and economic damage, highlighting the critical need for rapid and accurate flood inundation mapping. While remote sensing technologies have advanced flood monitoring capabilities, operational challenges persist: single-sensor approaches face weather-dependent data availability and limited revisit periods, while multi-sensor fusion methods require substantial computational resources and large-scale labeled datasets. To address these limitations, this study introduces a novel sensor-flexible flood detection methodology by fine-tuning Presto, a lightweight ($\sim$0.4M parameters) multi-modal pre-trained transformer that processes both Synthetic Aperture Radar (SAR) and multispectral (MS) data at the pixel level. Our approach uniquely enables flood mapping using SAR-only, MS-only, or combined SAR+MS inputs through a single model architecture, addressing the critical operational need for rapid response with whatever sensor data becomes available first during disasters. We evaluated our method on the Sen1Floods11 dataset against the large-scale Prithvi-100M baseline ($\sim$100M parameters) across three realistic data availability scenarios. The proposed model achieved superior performance with an F1 score of 0.896 and mIoU of 0.886 in the optimal sensor-fusion scenario, outperforming the established baseline. Crucially, the model demonstrated robustness by maintaining effective performance in MS-only scenarios (F1: 0.893) and functional capabilities in challenging SAR-only conditions (F1: 0.718), confirming the advantage of multi-modal pre-training for operational flood mapping. Our parameter-efficient, sensor-flexible approach offers an accessible and robust solution for real-world disaster scenarios requiring immediate flood extent assessment regardless of sensor availability constraints. 

---
# DPFNAS: Differential Privacy-Enhanced Federated Neural Architecture Search for 6G Edge Intelligence 

**Authors**: Yang Lv, Jin Cao, Ben Niu, Zhe Sun, Fengwei Wang, Fenghua Li, Hui Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.23030)  

**Abstract**: The Sixth-Generation (6G) network envisions pervasive artificial intelligence (AI) as a core goal, enabled by edge intelligence through on-device data utilization. To realize this vision, federated learning (FL) has emerged as a key paradigm for collaborative training across edge devices. However, the sensitivity and heterogeneity of edge data pose key challenges to FL: parameter sharing risks data reconstruction, and a unified global model struggles to adapt to diverse local distributions. In this paper, we propose a novel federated learning framework that integrates personalized differential privacy (DP) and adaptive model design. To protect training data, we leverage sample-level representations for knowledge sharing and apply a personalized DP strategy to resist reconstruction attacks. To ensure distribution-aware adaptation under privacy constraints, we develop a privacy-aware neural architecture search (NAS) algorithm that generates locally customized architectures and hyperparameters. To the best of our knowledge, this is the first personalized DP solution tailored for representation-based FL with theoretical convergence guarantees. Our scheme achieves strong privacy guarantees for training data while significantly outperforming state-of-the-art methods in model performance. Experiments on benchmark datasets such as CIFAR-10 and CIFAR-100 demonstrate that our scheme improves accuracy by 6.82\% over the federated NAS method PerFedRLNAS, while reducing model size to 1/10 and communication cost to 1/20. 

---
# Tracing the Representation Geometry of Language Models from Pretraining to Post-training 

**Authors**: Melody Zixuan Li, Kumar Krishna Agrawal, Arna Ghosh, Komal Kumar Teru, Adam Santoro, Guillaume Lajoie, Blake A. Richards  

**Link**: [PDF](https://arxiv.org/pdf/2509.23024)  

**Abstract**: Standard training metrics like loss fail to explain the emergence of complex capabilities in large language models. We take a spectral approach to investigate the geometry of learned representations across pretraining and post-training, measuring effective rank (RankMe) and eigenspectrum decay ($\alpha$-ReQ). With OLMo (1B-7B) and Pythia (160M-12B) models, we uncover a consistent non-monotonic sequence of three geometric phases during autoregressive pretraining. The initial "warmup" phase exhibits rapid representational collapse. This is followed by an "entropy-seeking" phase, where the manifold's dimensionality expands substantially, coinciding with peak n-gram memorization. Subsequently, a "compression-seeking" phase imposes anisotropic consolidation, selectively preserving variance along dominant eigendirections while contracting others, a transition marked with significant improvement in downstream task performance. We show these phases can emerge from a fundamental interplay of cross-entropy optimization under skewed token frequencies and representational bottlenecks ($d \ll |V|$). Post-training further transforms geometry: SFT and DPO drive "entropy-seeking" dynamics to integrate specific instructional or preferential data, improving in-distribution performance while degrading out-of-distribution robustness. Conversely, RLVR induces "compression-seeking", enhancing reward alignment but reducing generation diversity. 

---
# LLM Watermark Evasion via Bias Inversion 

**Authors**: Jeongyeon Hwang, Sangdon Park, Jungseul Ok  

**Link**: [PDF](https://arxiv.org/pdf/2509.23019)  

**Abstract**: Watermarking for large language models (LLMs) embeds a statistical signal during generation to enable detection of model-produced text. While watermarking has proven effective in benign settings, its robustness under adversarial evasion remains contested. To advance a rigorous understanding and evaluation of such vulnerabilities, we propose the \emph{Bias-Inversion Rewriting Attack} (BIRA), which is theoretically motivated and model-agnostic. BIRA weakens the watermark signal by suppressing the logits of likely watermarked tokens during LLM-based rewriting, without any knowledge of the underlying watermarking scheme. Across recent watermarking methods, BIRA achieves over 99\% evasion while preserving the semantic content of the original text. Beyond demonstrating an attack, our results reveal a systematic vulnerability, emphasizing the need for stress testing and robust defenses. 

---
# MoE-PHDS: One MoE checkpoint for flexible runtime sparsity 

**Authors**: Lauren. A Hannah, Soheil Zibakhsh, Kumari Nishu, Arnav Kundu, Mohammad Samragh Razlighi, Mehrdad Farajtabar, Minsik Cho  

**Link**: [PDF](https://arxiv.org/pdf/2509.23012)  

**Abstract**: Sparse Mixtures of Experts (MoEs) are typically trained to operate at a fixed sparsity level, e.g. $k$ in a top-$k$ gating function. This global sparsity level determines an operating point on the accuracy/latency curve; currently, meeting multiple efficiency targets means training and maintaining multiple models. This practice complicates serving, increases training and maintenance costs, and limits flexibility in meeting diverse latency, efficiency, and energy requirements. We show that pretrained MoEs are more robust to runtime sparsity shifts than commonly assumed, and introduce MoE-PHDS ({\bf P}ost {\bf H}oc {\bf D}eclared {\bf S}parsity), a lightweight SFT method that turns a single checkpoint into a global sparsity control surface. PHDS mixes training across sparsity levels and anchors with a short curriculum at high sparsity, requiring no architectural changes. The result is predictable accuracy/latency tradeoffs from one model: practitioners can ``dial $k$'' at inference time without swapping checkpoints, changing architecture, or relying on token-level heuristics. Experiments on OLMoE-1B-7B-0125, Qwen1.5-MoE-A2.7B, and proprietary models fit on multiple operating points show that PHDS matches or exceeds well-specified oracle models, improves cross-sparsity agreement by up to 22\% vs. well-specified oracle models, and enables simplified, flexible runtime MoE deployment by making global sparsity a first-class serving primitive. 

---
# Physically Plausible Multi-System Trajectory Generation and Symmetry Discovery 

**Authors**: Jiayin Liu, Yulong Yang, Vineet Bansal, Christine Allen-Blanchette  

**Link**: [PDF](https://arxiv.org/pdf/2509.23003)  

**Abstract**: From metronomes to celestial bodies, mechanics underpins how the world evolves in time and space. With consideration of this, a number of recent neural network models leverage inductive biases from classical mechanics to encourage model interpretability and ensure forecasted states are physical. However, in general, these models are designed to capture the dynamics of a single system with fixed physical parameters, from state-space measurements of a known configuration space. In this paper we introduce Symplectic Phase Space GAN (SPS-GAN) which can capture the dynamics of multiple systems, and generalize to unseen physical parameters from. Moreover, SPS-GAN does not require prior knowledge of the system configuration space. In fact, SPS-GAN can discover the configuration space structure of the system from arbitrary measurement types (e.g., state-space measurements, video frames). To achieve physically plausible generation, we introduce a novel architecture which embeds a Hamiltonian neural network recurrent module in a conditional GAN backbone. To discover the structure of the configuration space, we optimize the conditional time-series GAN objective with an additional physically motivated term to encourages a sparse representation of the configuration space. We demonstrate the utility of SPS-GAN for trajectory prediction, video generation and symmetry discovery. Our approach captures multiple systems and achieves performance on par with supervised models designed for single systems. 

---
# ADAM: A Diverse Archive of Mankind for Evaluating and Enhancing LLMs in Biographical Reasoning 

**Authors**: Jasin Cekinmez, Omid Ghahroodi, Saad Fowad Chandle, Dhiman Gupta, Ehsaneddin Asgari  

**Link**: [PDF](https://arxiv.org/pdf/2509.22991)  

**Abstract**: We introduce ADAM (A Diverse Archive of Mankind), a framework for evaluating and improving multimodal large language models (MLLMs) in biographical reasoning. To the best of our knowledge, this is the first work to systematically examine LLM capabilities in biography, a critical yet underexplored dimension of factual knowledge. At its core, AdamDB is a multilingual and multimodal dataset covering over 4 million individuals across geography, time, and profession, while AdamBench provides cognitively structured evaluations based on Bloom's taxonomy, spanning six reasoning levels in both English and native languages. To address hallucinations, particularly for lesser-known individuals, we propose AdamRAG, a retrieval-augmented generation system tailored to biographical contexts. Experiments show that AdamRAG substantially improves open-source models and modestly benefits closed-source ones, with the largest gains on lower-order reasoning. Popularity strongly mediates accuracy, and multimodal input via face images offers smaller, less consistent improvements than retrieval. ADAM establishes the first benchmark and framework for cognitively, culturally, and multimodally grounded biographical evaluation, advancing the development of multilingual, accurate, and hallucination-resistant MLLMs. 

---
# Functional Critic Modeling for Provably Convergent Off-Policy Actor-Critic 

**Authors**: Qinxun Bai, Yuxuan Han, Wei Xu, Zhengyuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.22964)  

**Abstract**: Off-policy reinforcement learning (RL) with function approximation offers an effective way to improve sample efficiency by reusing past experience. Within this setting, the actor-critic (AC) framework has achieved strong empirical success. However, both the critic and actor learning is challenging for the off-policy AC methods: first of all, in addition to the classic "deadly triad" instability of off-policy evaluation, it also suffers from a "moving target" problem, where the policy being evaluated changes continually; secondly, actor learning becomes less efficient due to the difficulty of estimating the exact off-policy policy gradient. The first challenge essentially reduces the problem to repeatedly performing off-policy evaluation for changing policies. For the second challenge, the off-policy policy gradient theorem requires a complex and often impractical algorithm to estimate an additional emphasis critic, which is typically neglected in practice, thereby reducing to the on-policy policy gradient as an approximation. In this work, we introduce a novel concept of functional critic modeling, which leads to a new AC framework that addresses both challenges for actor-critic learning under the deadly triad setting. We provide a theoretical analysis in the linear function setting, establishing the provable convergence of our framework, which, to the best of our knowledge, is the first convergent off-policy target-based AC algorithm. From a practical perspective, we further propose a carefully designed neural network architecture for the functional critic modeling and demonstrate its effectiveness through preliminary experiments on widely used RL tasks from the DeepMind Control Benchmark. 

---
# Tiny-QMoE 

**Authors**: Jack Cashman, Jiaqi Nie  

**Link**: [PDF](https://arxiv.org/pdf/2509.22951)  

**Abstract**: The QMoE model provides a practical approach for compression of massive Mixture-of-Experts (MoE) models. QMoE offers a solution geared towards memory limitations that often reach terabyte scales, and it has the advantage of working with high sparsity models which implicitly lend themselves to compression techniques. QMoE also has the advantage of only taking MoE models into account and does not evaluate its use with non mixture of expert systems. Although this prior attempt focuses on the limitations of large servers with the latest NVIDIA hardware which in the case of the H100 and V100 which have 80 GB of HBM (High Bandwidth Memory), what is not being considered is a significantly more constrained environment, such as in the case of mobile devices which may have in the case of the iPhone anywhere from 4 to 8 GB of unified memory which also needs to be shared with the operating system and additional processes. Although edge devices such as phones and laptops are becoming increasingly more computationally powerful, they are still not close to the level of advanced server machines such as NVIDIA. An additional constraint that we must consider is that of latency. The communication time of sending a request to an LLM server and then getting it back is an additional waiting time that can be removed. We may also want to use LLM technology in environments where there is no reliable network connection. 

---
# What Matters More For In-Context Learning under Matched Compute Budgets: Pretraining on Natural Text or Incorporating Targeted Synthetic Examples? 

**Authors**: Mohammed Sabry, Anya Belz  

**Link**: [PDF](https://arxiv.org/pdf/2509.22947)  

**Abstract**: Does explicitly exercising the induction circuit during pretraining improve in-context learning (ICL), or is natural text sufficient when compute is held constant (iso-FLOPs)? To test whether targeted synthetic data can accelerate induction-head emergence and enhance ICL, we introduce Bi-Induct, a lightweight curriculum that injects forward-copy (Induction), backward-copy (Anti), or a balanced mix into the pretraining stream. We train models from 0.13B to 1B parameters under iso-FLOPs, evaluating (i) few-shot ICL benchmarks, (ii) head-level telemetry, and (iii) held-out language modeling perplexity. Our findings challenge the assumption that early induction circuit activation directly improves ICL. While Bi-Induct accelerates induction-head emergence at small scales, this does not consistently yield stronger generalization. On standard LM benchmarks, Bi-Induct matches natural-only training; on function-style ICL probes, the 1B natural-only performs best. Stress tests (e.g., label permutation, HITS@1 vs. HITS@3, 1 vs. 10 shots) preserve these trends. Telemetry shows larger natural-only models develop broader, earlier induction heads without explicit induction patterns. Anti-induction data fails to elicit meaningful activation. Perplexity penalties from synthetic data shrink with scale, suggesting larger models can absorb non-natural patterns with minimal cost. Crucially, ablating the top 2% of induction heads degrades ICL more than random ablations, especially for natural-only models, indicating more centralized, load-bearing circuits. Bi-Induct variants exhibit more redundant induction activity, implying different circuit utilization. Overall, inducing activation is not sufficient: ICL gains depend on these circuits becoming functionally necessary. These results underscore mechanism-aware pretraining diagnostics and data mixtures that foster load-bearing, not merely present, structure. 

---
# Unsupervised Speech Enhancement using Data-defined Priors 

**Authors**: Dominik Klement, Matthew Maciejewski, Sanjeev Khudanpur, Jan Černocký, Lukáš Burget  

**Link**: [PDF](https://arxiv.org/pdf/2509.22942)  

**Abstract**: The majority of deep learning-based speech enhancement methods require paired clean-noisy speech data. Collecting such data at scale in real-world conditions is infeasible, which has led the community to rely on synthetically generated noisy speech. However, this introduces a gap between the training and testing phases. In this work, we propose a novel dual-branch encoder-decoder architecture for unsupervised speech enhancement that separates the input into clean speech and residual noise. Adversarial training is employed to impose priors on each branch, defined by unpaired datasets of clean speech and, optionally, noise. Experimental results show that our method achieves performance comparable to leading unsupervised speech enhancement approaches. Furthermore, we demonstrate the critical impact of clean speech data selection on enhancement performance. In particular, our findings reveal that performance may appear overly optimistic when in-domain clean speech data are used for prior definition -- a practice adopted in previous unsupervised speech enhancement studies. 

---
# Compute-Optimal Quantization-Aware Training 

**Authors**: Aleksandr Dremov, David Grangier, Angelos Katharopoulos, Awni Hannun  

**Link**: [PDF](https://arxiv.org/pdf/2509.22935)  

**Abstract**: Quantization-aware training (QAT) is a leading technique for improving the accuracy of quantized neural networks. Previous work has shown that decomposing training into a full-precision (FP) phase followed by a QAT phase yields superior accuracy compared to QAT alone. However, the optimal allocation of compute between the FP and QAT phases remains unclear. We conduct extensive experiments with various compute budgets, QAT bit widths, and model sizes from 86.0M to 2.2B to investigate how different QAT durations impact final performance. We demonstrate that, contrary to previous findings, the loss-optimal ratio of QAT to FP training increases with the total amount of compute. Moreover, the optimal fraction can be accurately predicted for a wide range of model sizes and quantization widths using the tokens-per-parameter-byte statistic. From experimental data, we derive a loss scaling law that predicts both optimal QAT ratios and final model performance across different QAT/FP compute allocation strategies and QAT bit widths. We use the scaling law to make further predictions, which we verify experimentally, including which QAT bit width is optimal under a given memory constraint and how QAT accuracy with different bit widths compares to full-precision model accuracy. Additionally, we propose a novel cooldown and QAT fusion approach that performs learning rate decay jointly with quantization-aware training, eliminating redundant full-precision model updates and achieving significant compute savings. These findings provide practical insights into efficient QAT planning and enable the training of higher-quality quantized models with the same compute budget. 

---
# MonoCon: A general framework for learning ultra-compact high-fidelity representations using monotonicity constraints 

**Authors**: Shreyas Gokhale  

**Link**: [PDF](https://arxiv.org/pdf/2509.22931)  

**Abstract**: Learning high-quality, robust, efficient, and disentangled representations is a central challenge in artificial intelligence (AI). Deep metric learning frameworks tackle this challenge primarily using architectural and optimization constraints. Here, we introduce a third approach that instead relies on $\textit{functional}$ constraints. Specifically, we present MonoCon, a simple framework that uses a small monotonic multi-layer perceptron (MLP) head attached to any pre-trained encoder. Due to co-adaptation between encoder and head guided by contrastive loss and monotonicity constraints, MonoCon learns robust, disentangled, and highly compact embeddings at a practically negligible performance cost. On the CIFAR-100 image classification task, MonoCon yields representations that are nearly 9x more compact and 1.5x more robust than the fine-tuned encoder baseline, while retaining 99\% of the baseline's 5-NN classification accuracy. We also report a 3.4x more compact and 1.4x more robust representation on an SNLI sentence similarity task for a marginal reduction in the STSb score, establishing MonoCon as a general domain-agnostic framework. Crucially, these robust, ultra-compact representations learned via functional constraints offer a unified solution to critical challenges in disparate contexts ranging from edge computing to cloud-scale retrieval. 

---
# Large language models management of medications: three performance analyses 

**Authors**: Kelli Henry, Steven Xu, Kaitlin Blotske, Moriah Cargile, Erin F. Barreto, Brian Murray, Susan Smith, Seth R. Bauer, Yanjun Gao, Tianming Liu, Andrea Sikora  

**Link**: [PDF](https://arxiv.org/pdf/2509.22926)  

**Abstract**: Background: Large language models (LLMs) can be useful in diagnosing medical conditions, but few studies have evaluated their consistency in recommending appropriate medication regimens. The purpose of this evaluation was to test GPT-4o on three medication benchmarking tests including mapping a drug name to its correct formulation, identifying drug-drug interactions using both its internal knowledge and using a web search, and preparing a medication order sentence after being given the medication name. Methods: Using GTP-4o three experiments were completed. Accuracy was quantified by computing cosine similarity on TF-IDF vectors, normalized Levenshtein similarity, and ROUGE-1/ROUGE-L F1 between each response and its reference string or by manual evaluation by clinicians. Results: GPT-4o performed poorly on drug-formulation matching, with frequent omissions of available drug formulations (mean 1.23 per medication) and hallucinations of formulations that do not exist (mean 1.14 per medication). Only 49% of tested medications were correctly matched to all available formulations. Accuracy was decreased for medications with more formulations (p<0.0001). GPT-4o was also inconsistent at identifying drug-drug-interactions, although it had better performance with the search-augmented assessment compared to its internal knowledge (54.7% vs. 69.2%, p=0.013). However, allowing a web-search worsened performance when there was no drug-drug interaction (median % correct 100% vs. 40%, p<0.001). Finally, GPT-4o performed moderately with preparing a medication order sentence, with only 65.8% of medication order sentences containing no medication or abbreviation errors. Conclusions: Model performance was overall poor for all tests. This highlights the need for domain-specific training through clinician-annotated datasets and a comprehensive evaluation framework for benchmarking performance. 

---
# Soft-Di[M]O: Improving One-Step Discrete Image Generation with Soft Embeddings 

**Authors**: Yuanzhi Zhu, Xi Wang, Stéphane Lathuilière, Vicky Kalogeiton  

**Link**: [PDF](https://arxiv.org/pdf/2509.22925)  

**Abstract**: One-step generators distilled from Masked Diffusion Models (MDMs) compress multiple sampling steps into a single forward pass, enabling efficient text and image synthesis. However, they suffer two key limitations: they inherit modeling bias from the teacher, and their discrete token outputs block gradient flow, preventing post-distillation refinements such as adversarial training, reward-based fine-tuning, and Test-Time Embedding Optimization (TTEO). In this work, we introduce soft embeddings, a simple relaxation that replaces discrete tokens with the expected embeddings under the generator's output distribution. Soft embeddings preserve representation fidelity for one-step discrete generator while providing a fully differentiable continuous surrogate that is compatible with teacher backbones and tokenizer decoders. Integrating soft embeddings into the Di[M]O distillation framework (denoted Soft-Di[M]O) makes one-step generators end-to-end trainable and enables straightforward application of GAN-based refinement, differentiable reward fine-tuning, and TTEO. Empirically, across multiple MDM teachers (e.g., MaskBit, MaskGen), Soft-Di[M]O achieves state-of-the-art one-step results: improved class-to-image performance, a one-step FID of 1.56 on ImageNet-256 with GAN-based refinement, along with higher GenEval and HPS scores on text-to-image with reward fine-tuning, and further gains from TTEO. 

---
# Rethinking Large Language Model Distillation: A Constrained Markov Decision Process Perspective 

**Authors**: Matthieu Zimmer, Xiaotong Ji, Tu Nguyen, Haitham Bou Ammar  

**Link**: [PDF](https://arxiv.org/pdf/2509.22921)  

**Abstract**: We introduce a novel approach to large language model (LLM) distillation by formulating it as a constrained reinforcement learning problem. While recent work has begun exploring the integration of task-specific rewards into distillation processes, existing methods typically rely on ad-hoc reward weighting. We propose a principled optimization framework that maximizes task-specific rewards while constraining the divergence from the teacher model to remain below a specified threshold. Our approach adapts constrained state augmented reinforcement learning to the distillation setting, introducing a modified reward function that maintains theoretical guarantees of constraint satisfaction without requiring state augmentation or teacher model access during deployment and without the computational overhead of the dual Lagrangian methods. Through extensive experiments on mathematical reasoning tasks, we demonstrate that our method achieves better constraint satisfaction rates and better reasoning compared to the soft Lagrangian relaxation baselines while maintaining competitive task performance. Our framework provides a theoretically grounded and practically efficient solution for reward-aware distillation in resource-constrained settings. 

---
# TY-RIST: Tactical YOLO Tricks for Real-time Infrared Small Target Detection 

**Authors**: Abdulkarim Atrash, Omar Moured, Yufan Chen, Jiaming Zhang, Seyda Ertekin, Omur Ugur  

**Link**: [PDF](https://arxiv.org/pdf/2509.22909)  

**Abstract**: Infrared small target detection (IRSTD) is critical for defense and surveillance but remains challenging due to (1) target loss from minimal features, (2) false alarms in cluttered environments, (3) missed detections from low saliency, and (4) high computational costs. To address these issues, we propose TY-RIST, an optimized YOLOv12n architecture that integrates (1) a stride-aware backbone with fine-grained receptive fields, (2) a high-resolution detection head, (3) cascaded coordinate attention blocks, and (4) a branch pruning strategy that reduces computational cost by about 25.5% while marginally improving accuracy and enabling real-time inference. We also incorporate the Normalized Gaussian Wasserstein Distance (NWD) to enhance regression stability. Extensive experiments on four benchmarks and across 20 different models demonstrate state-of-the-art performance, improving mAP at 0.5 IoU by +7.9%, Precision by +3%, and Recall by +10.2%, while achieving up to 123 FPS on a single GPU. Cross-dataset validation on a fifth dataset further confirms strong generalization capability. Additional results and resources are available at this https URL 

---
# Extract-0: A Specialized Language Model for Document Information Extraction 

**Authors**: Henrique Godoy  

**Link**: [PDF](https://arxiv.org/pdf/2509.22906)  

**Abstract**: This paper presents Extract-0, a 7-billion parameter language model specifically optimized for document information extraction that achieves performance exceeding models with parameter counts several orders of magnitude larger. Through a novel combination of synthetic data generation, supervised fine-tuning with Low-Rank Adaptation (LoRA), and reinforcement learning via Group Relative Policy Optimization (GRPO), Extract-0 achieves a mean reward of 0.573 on a benchmark of 1,000 diverse document extraction tasks, outperforming GPT-4.1 (0.457), o3 (0.464), and GPT-4.1-2025 (0.459). The training methodology employs a memory-preserving synthetic data generation pipeline that produces 280,128 training examples from diverse document sources, followed by parameterefficient fine-tuning that modifies only 0.53% of model weights (40.4M out of 7.66B parameters). The reinforcement learning phase introduces a novel semantic similarity-based reward function that handles the inherent ambiguity in information extraction tasks. This research demonstrates that task-specific optimization can yield models that surpass general-purpose systems while requiring substantially fewer computational resource. 

---
# Convolutional Set Transformer 

**Authors**: Federico Chinello, Giacomo Boracchi  

**Link**: [PDF](https://arxiv.org/pdf/2509.22889)  

**Abstract**: We introduce the Convolutional Set Transformer (CST), a novel neural architecture designed to process image sets of arbitrary cardinality that are visually heterogeneous yet share high-level semantics - such as a common category, scene, or concept. Existing set-input networks, e.g., Deep Sets and Set Transformer, are limited to vector inputs and cannot directly handle 3D image tensors. As a result, they must be cascaded with a feature extractor, typically a CNN, which encodes images into embeddings before the set-input network can model inter-image relationships. In contrast, CST operates directly on 3D image tensors, performing feature extraction and contextual modeling simultaneously, thereby enabling synergies between the two processes. This design yields superior performance in tasks such as Set Classification and Set Anomaly Detection and further provides native compatibility with CNN explainability methods such as Grad-CAM, unlike competing approaches that remain opaque. Finally, we show that CSTs can be pre-trained on large-scale datasets and subsequently adapted to new domains and tasks through standard Transfer Learning schemes. To support further research, we release CST-15, a CST backbone pre-trained on ImageNet (this https URL). 

---
# From Noise to Knowledge: A Comparative Study of Acoustic Anomaly Detection Models in Pumped-storage Hydropower Plants 

**Authors**: Karim Khamaisi, Nicolas Keller, Stefan Krummenacher, Valentin Huber, Bernhard Fässler, Bruno Rodrigues  

**Link**: [PDF](https://arxiv.org/pdf/2509.22881)  

**Abstract**: In the context of industrial factories and energy producers, unplanned outages are highly costly and difficult to service. However, existing acoustic-anomaly detection studies largely rely on generic industrial or synthetic datasets, with few focused on hydropower plants due to limited access. This paper presents a comparative analysis of acoustic-based anomaly detection methods, as a way to improve predictive maintenance in hydropower plants. We address key challenges in the acoustic preprocessing under highly noisy conditions before extracting time- and frequency-domain features. Then, we benchmark three machine learning models: LSTM AE, K-Means, and OC-SVM, which are tested on two real-world datasets from the Rodundwerk II pumped-storage plant in Austria, one with induced anomalies and one with real-world conditions. The One-Class SVM achieved the best trade-off of accuracy (ROC AUC 0.966-0.998) and minimal training time, while the LSTM autoencoder delivered strong detection (ROC AUC 0.889-0.997) at the expense of higher computational cost. 

---
# Scalable Wi-Fi RSS-Based Indoor Localization via Automatic Vision-Assisted Calibration 

**Authors**: Abdulkadir Bilge, Erdem Ergen, Burak Soner, Sinem Coleri  

**Link**: [PDF](https://arxiv.org/pdf/2509.22869)  

**Abstract**: Wi-Fi-based positioning promises a scalable and privacy-preserving solution for location-based services in indoor environments such as malls, airports, and campuses. RSS-based methods are widely deployable as RSS data is available on all Wi-Fi-capable devices, but RSS is highly sensitive to multipath, channel variations, and receiver characteristics. While supervised learning methods offer improved robustness, they require large amounts of labeled data, which is often costly to obtain. We introduce a lightweight framework that solves this by automating high-resolution synchronized RSS-location data collection using a short, camera-assisted calibration phase. An overhead camera is calibrated only once with ArUco markers and then tracks a device collecting RSS data from broadcast packets of nearby access points across Wi-Fi channels. The resulting (x, y, RSS) dataset is used to automatically train mobile-deployable localization algorithms, avoiding the privacy concerns of continuous video monitoring. We quantify the accuracy limits of such vision-assisted RSS data collection under key factors such as tracking precision and label synchronization. Using the collected experimental data, we benchmark traditional and supervised learning approaches under varying signal conditions and device types, demonstrating improved accuracy and generalization, validating the utility of the proposed framework for practical use. All code, tools, and datasets are released as open source. 

---
# Observation-Free Attacks on Online Learning to Rank 

**Authors**: Sameep Chattopadhyay, Nikhil Karamchandani, Sharayu Mohair  

**Link**: [PDF](https://arxiv.org/pdf/2509.22855)  

**Abstract**: Online learning to rank (OLTR) plays a critical role in information retrieval and machine learning systems, with a wide range of applications in search engines and content recommenders. However, despite their extensive adoption, the susceptibility of OLTR algorithms to coordinated adversarial attacks remains poorly understood. In this work, we present a novel framework for attacking some of the widely used OLTR algorithms. Our framework is designed to promote a set of target items so that they appear in the list of top-K recommendations for T - o(T) rounds, while simultaneously inducing linear regret in the learning algorithm. We propose two novel attack strategies: CascadeOFA for CascadeUCB1 and PBMOFA for PBM-UCB . We provide theoretical guarantees showing that both strategies require only O(log T) manipulations to succeed. Additionally, we supplement our theoretical analysis with empirical results on real-world data. 

---
# Patient-specific Biomolecular Instruction Tuning 

**Authors**: Irsyad Adam, Zekai Chen, David Laub, Shaun Porwal, Arda Pekis, Kevin Brown  

**Link**: [PDF](https://arxiv.org/pdf/2509.22853)  

**Abstract**: Proteomics data is essential to pathogenic understanding of a disease phenotype. In cancer, analysis of molecular signatures enables precision medicine through the identification of biological processes that drive individualized tumor progression, therapeutic resistance, and clinical heterogeneity. Recent advances in multimodal large language models (LLMs) have shown remarkable capacity to integrate and reason across heterogeneous data modalities. However, performing multi-modal language modeling for molecular understanding of patient-specific proteomics remains a significant challenge due to two barriers: (1) the lack of instruction-tuning datasets that enable clinical interpretation from proteomics data, and (2) the absence of language modeling architectures designed to capture the rich heterogeneity of molecular data. In this work, we introduce CPTAC-PROTSTRUCT, the first instruction tuning dataset for molecular understanding of oncology, comprising over 400k open-ended examples derived from individualized proteomic profiles curated from the largest national proteomics cancer study (CPTAC). Additionally, we propose KRONOS (Knowledge Representation of patient Omics Networks in Oncology via Structured tuning), a novel graph-LLM framework that leverages molecular interaction topology with proteomics to learn patient-specific graph representations for enhanced clinical reasoning. We show that KRONOS achieves competitive performance across benchmark clinical tasks, including molecular classification, temporal trajectory modeling, and tumor stage prediction from proteomics data. Ultimately, this approach empowers LLMs to understand patient-level pathogenesis, advancing precision medicine through more accurate diagnosis, prognosis, and treatment stratification. 

---
# Adaptive Margin RLHF via Preference over Preferences 

**Authors**: Yaswanth Chittepu, Prasann Singhal, Greg Durrett, Scott Niekum  

**Link**: [PDF](https://arxiv.org/pdf/2509.22851)  

**Abstract**: Margin-based optimization is fundamental to improving generalization and robustness in classification tasks. In the context of reward model learning from preferences within Reinforcement Learning from Human Feedback (RLHF), existing methods typically rely on no margins, fixed margins, or margins that are simplistic functions of preference ratings. However, such formulations often fail to account for the varying strengths of different preferences, for example some preferences are associated with larger margins between responses, or they rely on noisy margin information derived from ratings. We argue that modeling the strength of preferences can lead to better generalization and more faithful alignment. Furthermore, many existing methods that use adaptive margins assume access to accurate preference scores, which can be difficult for humans to provide reliably. We propose an approach that leverages preferences over preferences, that is annotations indicating which of two preferences reflects a stronger distinction. We use this ordinal signal to infer adaptive margins on a per-datapoint basis. We introduce an extension to Direct Preference Optimization (DPO), DPO-PoP, that incorporates adaptive margins from preference-over-preference supervision, enabling improved discriminative and generative performance. Empirically, our method outperforms vanilla DPO, DPO with fixed margins, and DPO with ground-truth margins on the UltraFeedback dataset. Additionally, we show that there is a tradeoff between discriminative and generative performance: improving test classification accuracy, particularly by correctly labeling weaker preferences at the expense of stronger ones, can lead to a decline in generative quality. To navigate this tradeoff, we propose two sampling strategies to gather preference-over-preference labels: one favoring discriminative performance and one favoring generative performance. 

---
# Boundary on the Table: Efficient Black-Box Decision-Based Attacks for Structured Data 

**Authors**: Roie Kazoom, Yuval Ratzabi, Etamar Rothstein, Ofer Hadar  

**Link**: [PDF](https://arxiv.org/pdf/2509.22850)  

**Abstract**: Adversarial robustness in structured data remains an underexplored frontier compared to vision and language domains. In this work, we introduce a novel black-box, decision-based adversarial attack tailored for tabular data. Our approach combines gradient-free direction estimation with an iterative boundary search, enabling efficient navigation of discrete and continuous feature spaces under minimal oracle access. Extensive experiments demonstrate that our method successfully compromises nearly the entire test set across diverse models, ranging from classical machine learning classifiers to large language model (LLM)-based pipelines. Remarkably, the attack achieves success rates consistently above 90%, while requiring only a small number of queries per instance. These results highlight the critical vulnerability of tabular models to adversarial perturbations, underscoring the urgent need for stronger defenses in real-world decision-making systems. 

---
# Multimodal Slice Interaction Network Enhanced by Transfer Learning for Precise Segmentation of Internal Gross Tumor Volume in Lung Cancer PET/CT Imaging 

**Authors**: Yi Luo, Yike Guo, Hamed Hooshangnejad, Rui Zhang, Xue Feng, Quan Chen, Wil Ngwa, Kai Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.22841)  

**Abstract**: Lung cancer remains the leading cause of cancerrelated deaths globally. Accurate delineation of internal gross tumor volume (IGTV) in PET/CT imaging is pivotal for optimal radiation therapy in mobile tumors such as lung cancer to account for tumor motion, yet is hindered by the limited availability of annotated IGTV datasets and attenuated PET signal intensity at tumor boundaries. In this study, we present a transfer learningbased methodology utilizing a multimodal interactive perception network with MAMBA, pre-trained on extensive gross tumor volume (GTV) datasets and subsequently fine-tuned on a private IGTV cohort. This cohort constitutes the PET/CT subset of the Lung-cancer Unified Cross-modal Imaging Dataset (LUCID). To further address the challenge of weak PET intensities in IGTV peripheral slices, we introduce a slice interaction module (SIM) within a 2.5D segmentation framework to effectively model inter-slice relationships. Our proposed module integrates channel and spatial attention branches with depthwise convolutions, enabling more robust learning of slice-to-slice dependencies and thereby improving overall segmentation performance. A comprehensive experimental evaluation demonstrates that our approach achieves a Dice of 0.609 on the private IGTV dataset, substantially surpassing the conventional baseline score of 0.385. This work highlights the potential of transfer learning, coupled with advanced multimodal techniques and a SIM to enhance the reliability and clinical relevance of IGTV segmentation for lung cancer radiation therapy planning. 

---
# Seeing Isn't Believing: Context-Aware Adversarial Patch Synthesis via Conditional GAN 

**Authors**: Roie Kazoom, Alon Goldberg, Hodaya Cohen, Ofer Hadar  

**Link**: [PDF](https://arxiv.org/pdf/2509.22836)  

**Abstract**: Adversarial patch attacks pose a severe threat to deep neural networks, yet most existing approaches rely on unrealistic white-box assumptions, untargeted objectives, or produce visually conspicuous patches that limit real-world applicability. In this work, we introduce a novel framework for fully controllable adversarial patch generation, where the attacker can freely choose both the input image x and the target class y target, thereby dictating the exact misclassification outcome. Our method combines a generative U-Net design with Grad-CAM-guided patch placement, enabling semantic-aware localization that maximizes attack effectiveness while preserving visual realism. Extensive experiments across convolutional networks (DenseNet-121, ResNet-50) and vision transformers (ViT-B/16, Swin-B/16, among others) demonstrate that our approach achieves state-of-the-art performance across all settings, with attack success rates (ASR) and target-class success (TCS) consistently exceeding 99%.
Importantly, we show that our method not only outperforms prior white-box attacks and untargeted baselines, but also surpasses existing non-realistic approaches that produce detectable artifacts. By simultaneously ensuring realism, targeted control, and black-box applicability-the three most challenging dimensions of patch-based attacks-our framework establishes a new benchmark for adversarial robustness research, bridging the gap between theoretical attack strength and practical stealthiness. 

---
# Bridging Language Models and Formal Methods for Intent-Driven Optical Network Design 

**Authors**: Anis Bekri, Amar Abane, Abdella Battou, Saddek Bensalem  

**Link**: [PDF](https://arxiv.org/pdf/2509.22834)  

**Abstract**: Intent-Based Networking (IBN) aims to simplify network management by enabling users to specify high-level goals that drive automated network design and configuration. However, translating informal natural-language intents into formally correct optical network topologies remains challenging due to inherent ambiguity and lack of rigor in Large Language Models (LLMs). To address this, we propose a novel hybrid pipeline that integrates LLM-based intent parsing, formal methods, and Optical Retrieval-Augmented Generation (RAG). By enriching design decisions with domain-specific optical standards and systematically incorporating symbolic reasoning and verification techniques, our pipeline generates explainable, verifiable, and trustworthy optical network designs. This approach significantly advances IBN by ensuring reliability and correctness, essential for mission-critical networking tasks. 

---
# Efficient Fine-Grained GPU Performance Modeling for Distributed Deep Learning of LLM 

**Authors**: Biyao Zhang, Mingkai Zheng, Debargha Ganguly, Xuecen Zhang, Vikash Singh, Vipin Chaudhary, Zhao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22832)  

**Abstract**: Training Large Language Models(LLMs) is one of the most compute-intensive tasks in high-performance computing. Predicting end-to-end training time for multi-billion parameter models distributed across hundreds of GPUs remains challenging due to complex interactions between transformer components, parallelism strategies(data, model, pipeline, tensor), and multi-tier communication. Learned models require costly sampling, while analytical models often struggle with real-world network and hardware complexities. We address this by decomposing LLMs into core computational primitives and modeling them with: (1) operator-level decomposition for fine-grained analysis; (2) lightweight sampling based hardware-aware prediction models for key operations; (3) an end-to-end prediction system integrating these components across complex parallelization strategies. Crucially, our methodology has been validated on two large-scale HPC systems. Our framework achieves low average prediction errors-4.98\% on Perlmutter(A100) and 9.38\% on Vista(GH200)-for models up to 20B parameters across 128 GPUs. Importantly, it runs entirely on CPUs, enabling rapid iteration over hardware configurations and training strategies without costly on-cluster experimentation. 

---
# Dynamic Buffers: Cost-Efficient Planning for Tabletop Rearrangement with Stacking 

**Authors**: Arman Barghi, Hamed Hosseini, Seraj Ghasemi, Mehdi Tale Masouleh, Ahmad Kalhor  

**Link**: [PDF](https://arxiv.org/pdf/2509.22828)  

**Abstract**: Rearranging objects in cluttered tabletop environments remains a long-standing challenge in robotics. Classical planners often generate inefficient, high-cost plans by shuffling objects individually and using fixed buffers--temporary spaces such as empty table regions or static stacks--to resolve conflicts. When only free table locations are used as buffers, dense scenes become inefficient, since placing an object can restrict others from reaching their goals and complicate planning. Allowing stacking provides extra buffer capacity, but conventional stacking is static: once an object supports another, the base cannot be moved, which limits efficiency. To overcome these issues, a novel planning primitive called the Dynamic Buffer is introduced. Inspired by human grouping strategies, it enables robots to form temporary, movable stacks that can be transported as a unit. This improves both feasibility and efficiency in dense layouts, and it also reduces travel in large-scale settings where space is abundant. Compared with a state-of-the-art rearrangement planner, the approach reduces manipulator travel cost by 11.89% in dense scenarios with a stationary robot and by 5.69% in large, low-density settings with a mobile manipulator. Practicality is validated through experiments on a Delta parallel robot with a two-finger gripper. These findings establish dynamic buffering as a key primitive for cost-efficient and robust rearrangement planning. 

---
# MMPB: It's Time for Multi-Modal Personalization 

**Authors**: Jaeik Kim, Woojin Kim, Woohyeon Park, Jaeyoung Do  

**Link**: [PDF](https://arxiv.org/pdf/2509.22820)  

**Abstract**: Visual personalization is essential in user-facing AI systems such as smart homes and healthcare, where aligning model behavior with user-centric concepts is critical. However, recent large Vision-Language Models (VLMs), despite their broad applicability, remain underexplored in their ability to adapt to individual users. In this paper, we introduce MMPB, the first extensive benchmark for evaluating VLMs on personalization. MMPB comprises 10k image-query pairs and includes 111 personalizable concepts across four categories: humans, animals, objects, and characters, with the human category enriched with preference-grounded queries. We structure personalization into three main task types, each highlighting a different key property of VLMs. Using 23 widely used VLMs including both open- and closed-source models, we evaluate personalization performance via a three-stage protocol: concept injection, multi-turn dialogue, and personalized querying. Our findings indicate that most VLMs (including some closed-source models) struggle with personalization, particularly in maintaining consistency over dialogue, handling user preferences, and adapting to visual cues. Our analysis reveals that the challenges in VLM personalization (such as refusal behaviors and long-context forgetting) highlight substantial room for improvement. By identifying these limitations and offering a scalable benchmark, MMPB offers valuable insights and a solid foundation for future research toward truly personalized multi-modal AI. Project Page: this http URL 

---
# MTRec: Learning to Align with User Preferences via Mental Reward Models 

**Authors**: Mengchen Zhao, Yifan Gao, Yaqing Hou, Xiangyang Li, Pengjie Gu, Zhenhua Dong, Ruiming Tang, Yi Cai  

**Link**: [PDF](https://arxiv.org/pdf/2509.22807)  

**Abstract**: Recommendation models are predominantly trained using implicit user feedback, since explicit feedback is often costly to obtain. However, implicit feedback, such as clicks, does not always reflect users' real preferences. For example, a user might click on a news article because of its attractive headline, but end up feeling uncomfortable after reading the content. In the absence of explicit feedback, such erroneous implicit signals may severely mislead recommender systems. In this paper, we propose MTRec, a novel sequential recommendation framework designed to align with real user preferences by uncovering their internal satisfaction on recommended items. Specifically, we introduce a mental reward model to quantify user satisfaction and propose a distributional inverse reinforcement learning approach to learn it. The learned mental reward model is then used to guide recommendation models to better align with users' real preferences. Our experiments show that MTRec brings significant improvements to a variety of recommendation models. We also deploy MTRec on an industrial short video platform and observe a 7 percent increase in average user viewing time. 

---
# VideoScore2: Think before You Score in Generative Video Evaluation 

**Authors**: Xuan He, Dongfu Jiang, Ping Nie, Minghao Liu, Zhengxuan Jiang, Mingyi Su, Wentao Ma, Junru Lin, Chun Ye, Yi Lu, Keming Wu, Benjamin Schneider, Quy Duc Do, Zhuofeng Li, Yiming Jia, Yuxuan Zhang, Guo Cheng, Haozhe Wang, Wangchunshu Zhou, Qunshu Lin, Yuanxing Zhang, Ge Zhang, Wenhao Huang, Wenhu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22799)  

**Abstract**: Recent advances in text-to-video generation have produced increasingly realistic and diverse content, yet evaluating such videos remains a fundamental challenge due to their multi-faceted nature encompassing visual quality, semantic alignment, and physical consistency. Existing evaluators and reward models are limited to single opaque scores, lack interpretability, or provide only coarse analysis, making them insufficient for capturing the comprehensive nature of video quality assessment. We present VideoScore2, a multi-dimensional, interpretable, and human-aligned framework that explicitly evaluates visual quality, text-to-video alignment, and physical/common-sense consistency while producing detailed chain-of-thought rationales. Our model is trained on a large-scale dataset VideoFeedback2 containing 27,168 human-annotated videos with both scores and reasoning traces across three dimensions, using a two-stage pipeline of supervised fine-tuning followed by reinforcement learning with Group Relative Policy Optimization (GRPO) to enhance analytical robustness. Extensive experiments demonstrate that VideoScore2 achieves superior performance with 44.35 (+5.94) accuracy on our in-domain benchmark VideoScore-Bench-v2 and 50.37 (+4.32) average performance across four out-of-domain benchmarks (VideoGenReward-Bench, VideoPhy2, etc), while providing interpretable assessments that bridge the gap between evaluation and controllable generation through effective reward modeling for Best-of-N sampling. Project Page: this https URL 

---
# Generative Modeling and Decision Fusion for Unknown Event Detection and Classification Using Synchrophasor Data 

**Authors**: Yi Hu, Zheyuan Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.22795)  

**Abstract**: Reliable detection and classification of power system events are critical for maintaining grid stability and situational awareness. Existing approaches often depend on limited labeled datasets, which restricts their ability to generalize to rare or unseen disturbances. This paper proposes a novel framework that integrates generative modeling, sliding-window temporal processing, and decision fusion to achieve robust event detection and classification using synchrophasor data. A variational autoencoder-generative adversarial network is employed to model normal operating conditions, where both reconstruction error and discriminator error are extracted as anomaly indicators. Two complementary decision strategies are developed: a threshold-based rule for computational efficiency and a convex hull-based method for robustness under complex error distributions. These features are organized into spatiotemporal detection and classification matrices through a sliding-window mechanism, and an identification and decision fusion stage integrates the outputs across PMUs. This design enables the framework to identify known events while systematically classifying previously unseen disturbances into a new category, addressing a key limitation of supervised classifiers. Experimental results demonstrate state-of-the-art accuracy, surpassing machine learning, deep learning, and envelope-based baselines. The ability to recognize unknown events further highlights the adaptability and practical value of the proposed approach for wide-area event analysis in modern power systems. 

---
# Differentially Private Two-Stage Gradient Descent for Instrumental Variable Regression 

**Authors**: Haodong Liang, Yanhao Jin, Krishnakumar Balasubramanian, Lifeng Lai  

**Link**: [PDF](https://arxiv.org/pdf/2509.22794)  

**Abstract**: We study instrumental variable regression (IVaR) under differential privacy constraints. Classical IVaR methods (like two-stage least squares regression) rely on solving moment equations that directly use sensitive covariates and instruments, creating significant risks of privacy leakage and posing challenges in designing algorithms that are both statistically efficient and differentially private. We propose a noisy two-state gradient descent algorithm that ensures $\rho$-zero-concentrated differential privacy by injecting carefully calibrated noise into the gradient updates. Our analysis establishes finite-sample convergence rates for the proposed method, showing that the algorithm achieves consistency while preserving privacy. In particular, we derive precise bounds quantifying the trade-off among privacy parameters, sample size, and iteration-complexity. To the best of our knowledge, this is the first work to provide both privacy guarantees and provable convergence rates for instrumental variable regression in linear models. We further validate our theoretical findings with experiments on both synthetic and real datasets, demonstrating that our method offers practical accuracy-privacy trade-offs. 

---
# A theoretical guarantee for SyncRank 

**Authors**: Yang Rao  

**Link**: [PDF](https://arxiv.org/pdf/2509.22766)  

**Abstract**: We present a theoretical and empirical analysis of the SyncRank algorithm for recovering a global ranking from noisy pairwise comparisons. By adopting a complex-valued data model where the true ranking is encoded in the phases of a unit-modulus vector, we establish a sharp non-asymptotic recovery guarantee for the associated semidefinite programming (SDP) relaxation. Our main theorem characterizes a critical noise threshold - scaling as sigma = O(sqrt(n / log n)) - below which SyncRank achieves exact ranking recovery with high probability. Extensive experiments under this model confirm the theoretical predictions and demonstrate the algorithm's robustness across varying problem sizes and noise regimes. 

---
# In-Context Learning can Perform Continual Learning Like Humans 

**Authors**: Liuwang Kang, Fan Wang, Shaoshan Liu, Hung-Chyun Chou, Chuan Lin, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.22764)  

**Abstract**: Large language models (LLMs) can adapt to new tasks via in-context learning (ICL) without parameter updates, making them powerful learning engines for fast adaptation. While extensive research has examined ICL as a few-shot learner, whether it can achieve long-term retention and cross-task knowledge accumulation when multitasks arrive sequentially remains underexplored. Motivated by human memory studies, we investigate the retention characteristics of ICL in multitask settings and extend it to in-context continual learning (ICCL), where continual learning ability emerges through task scheduling and prompt rearrangement. Experiments on Markov-Chain benchmarks demonstrate that, for specific large-language models, ICCL benefits from distributed practice (DP) in a manner analogous to humans, consistently revealing a spacing "sweet spot" for retention. Beyond retention performance, we propose a human-retention similarity metric to quantify how closely a continual-learning (CL) method aligns with human retention dynamics. Using this metric, we show that linear-attention models such as MAMBA and RWKV exhibit particularly human-like retention patterns, despite their retention performance lagging behind that of Transformer-based LLMs. Overall, our results establish ICCL as both cognitively plausible and practically effective, providing an inference-only CL paradigm that mitigates catastrophic forgetting and addresses the stability-plasticity dilemma in conventional CL methods. 

---
# UESA-Net: U-Shaped Embedded Multidirectional Shrinkage Attention Network for Ultrasound Nodule Segmentation 

**Authors**: Tangqi Shi, Pietro Lio  

**Link**: [PDF](https://arxiv.org/pdf/2509.22763)  

**Abstract**: Background: Breast and thyroid cancers pose an increasing public-health burden. Ultrasound imaging is a cost-effective, real-time modality for lesion detection and segmentation, yet suffers from speckle noise, overlapping structures, and weak global-local feature interactions. Existing networks struggle to reconcile high-level semantics with low-level spatial details. We aim to develop a segmentation framework that bridges the semantic gap between global context and local detail in noisy ultrasound images.
Methods: We propose UESA-Net, a U-shaped network with multidirectional shrinkage attention. The encoder-decoder architecture captures long-range dependencies and fine-grained structures of lesions. Within each encoding block, attention modules operate along horizontal, vertical, and depth directions to exploit spatial details, while a shrinkage (threshold) strategy integrates prior knowledge and local features. The decoder mirrors the encoder but applies a pairwise shrinkage mechanism, combining prior low-level physical cues with corresponding encoder features to enhance context modeling.
Results: On two public datasets - TN3K (3493 images) and BUSI (780 images) - UESA-Net achieved state-of-the-art performance with intersection-over-union (IoU) scores of 0.8487 and 0.6495, respectively.
Conclusions: UESA-Net effectively aggregates multidirectional spatial information and prior knowledge to improve robustness and accuracy in breast and thyroid ultrasound segmentation, demonstrating superior performance to existing methods on multiple benchmarks. 

---
# MILR: Improving Multimodal Image Generation via Test-Time Latent Reasoning 

**Authors**: Yapeng Mi, Hengli Li, Yanpeng Zhao, Chenxi Li, Huimin Wu, Xiaojian Ma, Song-Chun Zhu, Ying Nian Wu, Qing Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22761)  

**Abstract**: Reasoning-augmented machine learning systems have shown improved performance in various domains, including image generation. However, existing reasoning-based methods for image generation either restrict reasoning to a single modality (image or text) or rely on high-quality reasoning data for fine-tuning. To tackle these limitations, we propose MILR, a test-time method that jointly reasons over image and text in a unified latent vector space. Reasoning in MILR is performed by searching through vector representations of discrete image and text tokens. Practically, this is implemented via the policy gradient method, guided by an image quality critic. We instantiate MILR within the unified multimodal understanding and generation (MUG) framework that natively supports language reasoning before image synthesis and thus facilitates cross-modal reasoning. The intermediate model outputs, which are to be optimized, serve as the unified latent space, enabling MILR to operate entirely at test time. We evaluate MILR on GenEval, T2I-CompBench, and WISE, achieving state-of-the-art results on all benchmarks. Notably, on knowledge-intensive WISE, MILR attains an overall score of 0.63, improving over the baseline by 80%. Our further analysis indicates that joint reasoning in the unified latent space is the key to its strong performance. Moreover, our qualitative studies reveal MILR's non-trivial ability in temporal and cultural reasoning, highlighting the efficacy of our reasoning method. 

---
# Red Teaming Quantum-Resistant Cryptographic Standards: A Penetration Testing Framework Integrating AI and Quantum Security 

**Authors**: Petar Radanliev  

**Link**: [PDF](https://arxiv.org/pdf/2509.22757)  

**Abstract**: This study presents a structured approach to evaluating vulnerabilities within quantum cryptographic protocols, focusing on the BB84 quantum key distribution method and National Institute of Standards and Technology (NIST) approved quantum-resistant algorithms. By integrating AI-driven red teaming, automated penetration testing, and real-time anomaly detection, the research develops a framework for assessing and mitigating security risks in quantum networks. The findings demonstrate that AI can be effectively used to simulate adversarial attacks, probe weaknesses in cryptographic implementations, and refine security mechanisms through iterative feedback. The use of automated exploit simulations and protocol fuzzing provides a scalable means of identifying latent vulnerabilities, while adversarial machine learning techniques highlight novel attack surfaces within AI-enhanced cryptographic processes. This study offers a comprehensive methodology for strengthening quantum security and provides a foundation for integrating AI-driven cybersecurity practices into the evolving quantum landscape. 

---
# Persistent Autoregressive Mapping with Traffic Rules for Autonomous Driving 

**Authors**: Shiyi Liang, Xinyuan Chang, Changjie Wu, Huiyuan Yan, Yifan Bai, Xinran Liu, Hang Zhang, Yujian Yuan, Shuang Zeng, Mu Xu, Xing Wei  

**Link**: [PDF](https://arxiv.org/pdf/2509.22756)  

**Abstract**: Safe autonomous driving requires both accurate HD map construction and persistent awareness of traffic rules, even when their associated signs are no longer visible. However, existing methods either focus solely on geometric elements or treat rules as temporary classifications, failing to capture their persistent effectiveness across extended driving sequences. In this paper, we present PAMR (Persistent Autoregressive Mapping with Traffic Rules), a novel framework that performs autoregressive co-construction of lane vectors and traffic rules from visual observations. Our approach introduces two key mechanisms: Map-Rule Co-Construction for processing driving scenes in temporal segments, and Map-Rule Cache for maintaining rule consistency across these segments. To properly evaluate continuous and consistent map generation, we develop MapDRv2, featuring improved lane geometry annotations. Extensive experiments demonstrate that PAMR achieves superior performance in joint vector-rule mapping tasks, while maintaining persistent rule effectiveness throughout extended driving sequences. 

---
# Self-driving cars: Are we there yet? 

**Authors**: Merve Atasever, Zhuochen Liu, Qingpei Li, Akshay Hitendra Shah, Hans Walker, Jyotirmoy V. Deshmukh, Rahul Jain  

**Link**: [PDF](https://arxiv.org/pdf/2509.22754)  

**Abstract**: Autonomous driving remains a highly active research domain that seeks to enable vehicles to perceive dynamic environments, predict the future trajectories of traffic agents such as vehicles, pedestrians, and cyclists and plan safe and efficient future motions. To advance the field, several competitive platforms and benchmarks have been established to provide standardized datasets and evaluation protocols. Among these, leaderboards by the CARLA organization and nuPlan and the Waymo Open Dataset have become leading benchmarks for assessing motion planning algorithms. Each offers a unique dataset and challenging planning problems spanning a wide range of driving scenarios and conditions. In this study, we present a comprehensive comparative analysis of the motion planning methods featured on these three leaderboards. To ensure a fair and unified evaluation, we adopt CARLA leaderboard v2.0 as our common evaluation platform and modify the selected models for compatibility. By highlighting the strengths and weaknesses of current approaches, we identify prevailing trends, common challenges, and suggest potential directions for advancing motion planning research. 

---
# Variance-Bounded Evaluation without Ground Truth: VB-Score 

**Authors**: Kaihua Ding  

**Link**: [PDF](https://arxiv.org/pdf/2509.22751)  

**Abstract**: Reliable evaluation is a central challenge in machine learning when tasks lack ground truth labels or involve ambiguity and noise. Conventional frameworks, rooted in the Cranfield paradigm and label-based metrics, fail in such cases because they cannot assess how robustly a system performs under uncertain interpretations. We introduce VB-Score, a variance-bounded evaluation framework that measures both effectiveness and robustness without requiring ground truth. Given a query or input, VB-Score enumerates plausible interpretations, assigns probabilities, and evaluates output by expected success penalized by variance, rewarding consistent performance across intents. We provide a formal analysis of VB-Score, establishing range, monotonicity, and stability properties, and relate it to risk-sensitive measures such as mean-variance utility. Experiments on ambiguous queries and entity-centric retrieval tasks show that VB-Score surfaces robustness differences hidden by conventional metrics. By enabling reproducible, label-free evaluation, VB-Score offers a principled foundation for benchmarking machine learning systems in ambiguous or label-scarce domains. 

---
# MIRAGE: Multi-hop Reasoning with Ambiguity Evaluation for Illusory Questions 

**Authors**: Jeonghyun Park, Ingeol Baek, Seunghyun Yoon, Haeun Jang, Aparna Garimella, Akriti Jain, Nedim Lipka, Hwanhee Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.22750)  

**Abstract**: Real-world Multi-hop Question Answering (QA) often involves ambiguity that is inseparable from the reasoning process itself. This ambiguity creates a distinct challenge, where multiple reasoning paths emerge from a single question, each requiring independent resolution. Since each sub-question is ambiguous, the model must resolve ambiguity at every step. Thus, answering a single question requires handling multiple layers of ambiguity throughout the reasoning chain. We find that current Large Language Models (LLMs) struggle in this setting, typically exploring wrong reasoning paths and producing incomplete answers. To facilitate research on multi-hop ambiguity, we introduce MultI-hop Reasoning with AmbiGuity Evaluation for Illusory Questions (MIRAGE), a benchmark designed to analyze and evaluate this challenging intersection of ambiguity interpretation and multi-hop reasoning. MIRAGE contains 1,142 high-quality examples of ambiguous multi-hop questions, categorized under a taxonomy of syntactic, general, and semantic ambiguity, and curated through a rigorous multi-LLM verification pipeline. Our experiments reveal that even state-of-the-art models struggle on MIRAGE, confirming that resolving ambiguity combined with multi-step inference is a distinct and significant challenge. To establish a robust baseline, we propose CLarifying Ambiguity with a Reasoning and InstructiON (CLARION), a multi-agent framework that significantly outperforms existing approaches on MIRAGE, paving the way for more adaptive and robust reasoning systems. 

---
# Defending MoE LLMs against Harmful Fine-Tuning via Safety Routing Alignment 

**Authors**: Jaehan Kim, Minkyoo Song, Seungwon Shin, Sooel Son  

**Link**: [PDF](https://arxiv.org/pdf/2509.22745)  

**Abstract**: Recent large language models (LLMs) have increasingly adopted the Mixture-of-Experts (MoE) architecture for efficiency. MoE-based LLMs heavily depend on a superficial safety mechanism in which harmful inputs are routed safety-critical experts. However, our analysis reveals that routing decisions for harmful inputs drift significantly after fine-tuning, exposing a critical vulnerability to harmful fine-tuning (HFT) attacks. Existing defenses, primarily designed for monolithic LLMs, are less effective for MoE LLMs as they fail to prevent drift in harmful input routing. To address this limitation, we propose SafeMoE, a safe fine-tuning method tailored to MoE LLMs. SafeMoE directly mitigates routing drift by penalizing the gap between the routing weights of a fine-tuned model and those of the initial safety-aligned model, thereby preserving the safety-aligned routing of harmful inputs to safety-critical experts. Experiments on open-source MoE LLMs ranging from 7B to 141B parameters demonstrate that SafeMoE effectively mitigates HFT attacks, reducing the harmfulness score of OLMoE from 62.0 to 5.0, for example, while maintaining task utility within 1% degradation and incurring only 2% overhead. It significantly outperforms state-of-the-art defense methods for safeguarding LLM fine-tuning and remains effective in recent large-scale MoE LLMs such as gpt-oss and Llama 4. Our implementation is available at this https URL. 

---
# Index-MSR: A high-efficiency multimodal fusion framework for speech recognition 

**Authors**: Jinming Chen, Lu Wang, Zheshu Song, Wei Deng  

**Link**: [PDF](https://arxiv.org/pdf/2509.22744)  

**Abstract**: Driven by large scale datasets and LLM based architectures, automatic speech recognition (ASR) systems have achieved remarkable improvements in accuracy. However, challenges persist for domain-specific terminology, and short utterances lacking semantic coherence, where recognition performance often degrades significantly. In this work, we present Index-MSR, an efficient multimodal speech recognition framework. At its core is a novel Multimodal Fusion Decoder (MFD), which effectively incorporates text-related information from videos (e.g., subtitles and presentation slides) into the speech recognition. This cross-modal integration not only enhances overall ASR accuracy but also yields substantial reductions in substitution errors. Extensive evaluations on both an in-house subtitle dataset and a public AVSR dataset demonstrate that Index-MSR achieves sota accuracy, with substitution errors reduced by 20,50%. These results demonstrate that our approach efficiently exploits text-related cues from video to improve speech recognition accuracy, showing strong potential in applications requiring strict audio text synchronization, such as audio translation. 

---
# Societal Capacity Assessment Framework: Measuring Resilience to Inform Advanced AI Risk Management 

**Authors**: Milan Gandhi, Peter Cihon, Owen Larter, Rebecca Anselmetti  

**Link**: [PDF](https://arxiv.org/pdf/2509.22742)  

**Abstract**: Risk assessments for advanced AI systems require evaluating both the models themselves and their deployment contexts. We introduce the Societal Capacity Assessment Framework (SCAF), an indicators-based approach to measuring a society's vulnerability, coping capacity, and adaptive capacity in response to AI-related risks. SCAF adapts established resilience analysis methodologies to AI, enabling organisations to ground risk management in insights about country-level deployment conditions. It can also support stakeholders in identifying opportunities to strengthen societal preparedness for emerging AI capabilities. By bridging disparate literatures and the "context gap" in AI evaluation, SCAF promotes more holistic risk assessment and governance as advanced AI systems proliferate globally. 

---
# Learning What To Hear: Boosting Sound-Source Association For Robust Audiovisual Instance Segmentation 

**Authors**: Jinbae Seo, Hyeongjun Kwon, Kwonyoung Kim, Jiyoung Lee, Kwanghoon Sohn  

**Link**: [PDF](https://arxiv.org/pdf/2509.22740)  

**Abstract**: Audiovisual instance segmentation (AVIS) requires accurately localizing and tracking sounding objects throughout video sequences. Existing methods suffer from visual bias stemming from two fundamental issues: uniform additive fusion prevents queries from specializing to different sound sources, while visual-only training objectives allow queries to converge to arbitrary salient objects. We propose Audio-Centric Query Generation using cross-attention, enabling each query to selectively attend to distinct sound sources and carry sound-specific priors into visual decoding. Additionally, we introduce Sound-Aware Ordinal Counting (SAOC) loss that explicitly supervises sounding object numbers through ordinal regression with monotonic consistency constraints, preventing visual-only convergence during training. Experiments on AVISeg benchmark demonstrate consistent improvements: +1.64 mAP, +0.6 HOTA, and +2.06 FSLA, validating that query specialization and explicit counting supervision are crucial for accurate audiovisual instance segmentation. 

---
# Painless Activation Steering: An Automated, Lightweight Approach for Post-Training Large Language Models 

**Authors**: Sasha Cui, Zhongren Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.22739)  

**Abstract**: Language models (LMs) are typically post-trained for desired capabilities and behaviors via weight-based or prompt-based steering, but the former is time-consuming and expensive, and the latter is not precisely controllable and often requires manual trial-and-error. While activation steering (AS) promises a cheap, fast, and controllable alternative to the two existing post-training methods, current AS techniques require hand-crafted prompt pairs or labor-intensive feature annotation, making them more inconvenient than the plug-and-play methods such as Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT). We introduce Painless Activation Steering (PAS), a family of fully automated methods that make AS readily usable with any given labeled dataset, with no need for prompt construction, feature labeling, or human intervention. We evaluate PAS on three open-weight models (Llama3.1-8B-Instruct, DeepSeek-R1-Distill-8B, and Nous-Hermes-2) and 18 tasks; we find that PAS reliably improves performance for behavior tasks, but not for intelligence-oriented tasks. The introspective variant (iPAS) delivers the strongest causal steering effects (10.1% on Bias, 5.2% on Morality, and 34.8% on Alignment). We also show PAS delivers additional gains on top of In-Context Learning (ICL) and SFT. PAS constructs a fast, lightweight activation vector that can be cheaply trained, easily stored, and activated at will. Our results provide a characterization of where AS helps, where it fails, and how to deploy it as a practical, automated LM post-training option. 

---
# CompareBench: A Benchmark for Visual Comparison Reasoning in Vision-Language Models 

**Authors**: Jie Cai, Kangning Yang, Lan Fu, Jiaming Ding, Jinlong Li, Huiming Sun, Daitao Xing, Jinglin Shen, Zibo Meng  

**Link**: [PDF](https://arxiv.org/pdf/2509.22737)  

**Abstract**: We introduce CompareBench, a benchmark for evaluating visual comparison reasoning in vision-language models (VLMs), a fundamental yet understudied skill. CompareBench consists of 1000 QA pairs across four tasks: quantity (600), temporal (100), geometric (200), and spatial (100). It is derived from two auxiliary datasets that we constructed: TallyBench (2000 counting images with QA) and HistCaps (515 historical images with bilingual captions). We evaluate both closed-source APIs (OpenAI, Gemini, Claude) and open-source models (Qwen2.5-VL and Qwen3-VL series). Results show clear scaling trends but also reveal critical limitations: even the strongest models consistently fail at temporal ordering and spatial relations, and they often make mistakes in basic counting and geometric comparisons that are trivial for humans. These findings demonstrate that visual comparison remains a systematic blind spot for current VLMs. By providing controlled, diverse, and diagnostic evaluation, CompareBench establishes a foundation for advancing more reliable multimodal reasoning. 

---
# Consistency Models as Plug-and-Play Priors for Inverse Problems 

**Authors**: Merve Gülle, Junno Yun, Yaşar Utku Alçalar, Mehmet Akçakaya  

**Link**: [PDF](https://arxiv.org/pdf/2509.22736)  

**Abstract**: Diffusion models have found extensive use in solving numerous inverse problems. Such diffusion inverse problem solvers aim to sample from the posterior distribution of data given the measurements, using a combination of the unconditional score function and an approximation of the posterior related to the forward process. Recently, consistency models (CMs) have been proposed to directly predict the final output from any point on the diffusion ODE trajectory, enabling high-quality sampling in just a few NFEs. CMs have also been utilized for inverse problems, but existing CM-based solvers either require additional task-specific training or utilize data fidelity operations with slow convergence, not amenable to large-scale problems. In this work, we reinterpret CMs as proximal operators of a prior, enabling their integration into plug-and-play (PnP) frameworks. We propose a solver based on PnP-ADMM, which enables us to leverage the fast convergence of conjugate gradient method. We further accelerate this with noise injection and momentum, dubbed PnP-CM, and show it maintains the convergence properties of the baseline PnP-ADMM. We evaluate our approach on a variety of inverse problems, including inpainting, super-resolution, Gaussian deblurring, and magnetic resonance imaging (MRI) reconstruction. To the best of our knowledge, this is the first CM trained for MRI datasets. Our results show that PnP-CM achieves high-quality reconstructions in as few as 4 NFEs, and can produce meaningful results in 2 steps, highlighting its effectiveness in real-world inverse problems while outperforming comparable CM-based approaches. 

---
# Regulating the Agency of LLM-based Agents 

**Authors**: Seán Boddy, Joshua Joseph  

**Link**: [PDF](https://arxiv.org/pdf/2509.22735)  

**Abstract**: As increasingly capable large language model (LLM)-based agents are developed, the potential harms caused by misalignment and loss of control grow correspondingly severe. To address these risks, we propose an approach that directly measures and controls the agency of these AI systems. We conceptualize the agency of LLM-based agents as a property independent of intelligence-related measures and consistent with the interdisciplinary literature on the concept of agency. We offer (1) agency as a system property operationalized along the dimensions of preference rigidity, independent operation, and goal persistence, (2) a representation engineering approach to the measurement and control of the agency of an LLM-based agent, and (3) regulatory tools enabled by this approach: mandated testing protocols, domain-specific agency limits, insurance frameworks that price risk based on agency, and agency ceilings to prevent societal-scale risks. We view our approach as a step toward reducing the risks that motivate the ``Scientist AI'' paradigm, while still capturing some of the benefits from limited agentic behavior. 

---
# Automated Formative Feedback for Short-form Writing: An LLM-Driven Approach and Adoption Analysis 

**Authors**: Tiago Fernandes Tavares, Luciano Pereira Soares  

**Link**: [PDF](https://arxiv.org/pdf/2509.22734)  

**Abstract**: This paper explores the development and adoption of AI-based formative feedback in the context of biweekly reports in an engineering Capstone program. Each student is required to write a short report detailing their individual accomplishments over the past two weeks, which is then assessed by their advising professor. An LLM-powered tool was developed to provide students with personalized feedback on their draft reports, guiding them toward improved completeness and quality. Usage data across two rounds revealed an initial barrier to adoption, with low engagement rates. However, students who engaged in the AI feedback system demonstrated the ability to use it effectively, leading to improvements in the completeness and quality of their reports. Furthermore, the tool's task-parsing capabilities provided a novel approach to identify potential student organizational tasks and deliverables. The findings suggest initial skepticism toward the tool with a limited adoption within the studied context, however, they also highlight the potential for AI-driven tools to provide students and professors valuable insights and formative support. 

---
# Rebuild AC Power Flow Models with Graph Attention Networks 

**Authors**: Yuting Hu, Jinjun Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2509.22733)  

**Abstract**: A full power flow (PF) model is a complete representation of the physical power network. Traditional model-based methods rely on the full PF model to implement power flow analysis. In practice, however, some PF model parameters can be inaccurate or even unavailable due to the uncertainties or dynamics in the power systems. Moreover, because the power network keeps evolving with possibly changing topology, the generalizability of a PF model to different network sizes and typologies should be considered. In this paper, we propose a PF rebuild model based on graph attention networks (GAT) by constructing a new graph based on the real and imaginary parts of voltage at each bus. By comparing with two state-of-the-art PF rebuild models for different standard IEEE power system cases and their modified topology variants, we demonstrate the feasibility of our method. Experimental results show that our proposed model achieves better accuracy for a changing network and can generalize to different networks with less accuracy discount. 

---
# Bidirectional Intention Inference Enhances LLMs' Defense Against Multi-Turn Jailbreak Attacks 

**Authors**: Haibo Tong, Dongcheng Zhao, Guobin Shen, Xiang He, Dachuan Lin, Feifei Zhao, Yi Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.22732)  

**Abstract**: The remarkable capabilities of Large Language Models (LLMs) have raised significant safety concerns, particularly regarding "jailbreak" attacks that exploit adversarial prompts to bypass safety alignment mechanisms. Existing defense research primarily focuses on single-turn attacks, whereas multi-turn jailbreak attacks progressively break through safeguards through by concealing malicious intent and tactical manipulation, ultimately rendering conventional single-turn defenses ineffective. To address this critical challenge, we propose the Bidirectional Intention Inference Defense (BIID). The method integrates forward request-based intention inference with backward response-based intention retrospection, establishing a bidirectional synergy mechanism to detect risks concealed within seemingly benign inputs, thereby constructing a more robust guardrails that effectively prevents harmful content generation. The proposed method undergoes systematic evaluation compared with a no-defense baseline and seven representative defense methods across three LLMs and two safety benchmarks under 10 different attack methods. Experimental results demonstrate that the proposed method significantly reduces the Attack Success Rate (ASR) across both single-turn and multi-turn jailbreak attempts, outperforming all existing baseline methods while effectively maintaining practical utility. Notably, comparative experiments across three multi-turn safety datasets further validate the proposed model's significant advantages over other defense approaches. 

---
# Multi-Modal Sentiment Analysis with Dynamic Attention Fusion 

**Authors**: Sadia Abdulhalim, Muaz Albaghdadi, Moshiur Farazi  

**Link**: [PDF](https://arxiv.org/pdf/2509.22729)  

**Abstract**: Traditional sentiment analysis has long been a unimodal task, relying solely on text. This approach overlooks non-verbal cues such as vocal tone and prosody that are essential for capturing true emotional intent. We introduce Dynamic Attention Fusion (DAF), a lightweight framework that combines frozen text embeddings from a pretrained language model with acoustic features from a speech encoder, using an adaptive attention mechanism to weight each modality per utterance. Without any finetuning of the underlying encoders, our proposed DAF model consistently outperforms both static fusion and unimodal baselines on a large multimodal benchmark. We report notable gains in F1-score and reductions in prediction error and perform a variety of ablation studies that support our hypothesis that the dynamic weighting strategy is crucial for modeling emotionally complex inputs. By effectively integrating verbal and non-verbal information, our approach offers a more robust foundation for sentiment prediction and carries broader impact for affective computing applications -- from emotion recognition and mental health assessment to more natural human computer interaction. 

---
# Prompt-aware classifier free guidance for diffusion models 

**Authors**: Xuanhao Zhang, Chang Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.22728)  

**Abstract**: Diffusion models have achieved remarkable progress in image and audio generation, largely due to Classifier-Free Guidance. However, the choice of guidance scale remains underexplored: a fixed scale often fails to generalize across prompts of varying complexity, leading to oversaturation or weak alignment. We address this gap by introducing a prompt-aware framework that predicts scale-dependent quality and selects the optimal guidance at inference. Specifically, we construct a large synthetic dataset by generating samples under multiple scales and scoring them with reliable evaluation metrics. A lightweight predictor, conditioned on semantic embeddings and linguistic complexity, estimates multi-metric quality curves and determines the best scale via a utility function with regularization. Experiments on MSCOCO~2014 and AudioCaps show consistent improvements over vanilla CFG, enhancing fidelity, alignment, and perceptual preference. This work demonstrates that prompt-aware scale selection provides an effective, training-free enhancement for pretrained diffusion backbones. 

---
# A Meta-Analysis of LLM Effects on Students across Qualification, Socialisation, and Subjectification 

**Authors**: Jiayu Huang, Ruoxin Ritter Wang, Jen-Hao Liu, Boming Xia, Yue Huang, Ruoxi Sun, Jason, Jinan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2509.22725)  

**Abstract**: Large language models (LLMs) are increasingly positioned as solutions for education, yet evaluations often reduce their impact to narrow performance metrics. This paper reframes the question by asking "what kind of impact should LLMs have in education?" Drawing on Biesta's tripartite account of good education: qualification, socialisation, and subjectification, we present a meta-analysis of 133 experimental and quasi-experimental studies (k = 188). Overall, the impact of LLMs on student learning is positive but uneven. Strong effects emerge in qualification, particularly when LLMs function as tutors in sustained interventions. Socialisation outcomes appear more variable, concentrated in sustained, reflective interventions. Subjectification, linked to autonomy and learner development, remains fragile, with improvements confined to small-scale, long-term studies. This purpose-level view highlights design as the decisive factor: without scaffolds for participation and agency, LLMs privilege what is easiest to measure while neglecting broader aims of education. For HCI and education, the issue is not just whether LLMs work, but what futures they enable or foreclose. 

---
# A Data-Driven Framework for Digital Transformation in Smart Cities: Integrating AI, Dashboards, and IoT Readiness 

**Authors**: Ángel Lloret, Jesús Peral, Antonio Ferrández, María Auladell, Rafael Muñoz  

**Link**: [PDF](https://arxiv.org/pdf/2509.22721)  

**Abstract**: Digital transformation (DT) has become a strategic priority for public administrations, particularly due to the need to deliver more efficient and citizen-centered services and respond to societal expectations, ESG (Environmental, Social, and Governance) criteria, and the United Nations Sustainable Development Goals (UN SDGs). In this context, the main objective of this study is to propose an innovative methodology to automatically evaluate the level of digital transformation (DT) in public sector organizations. The proposed approach combines traditional assessment methods with Artificial Intelligence (AI) techniques. The methodology follows a dual approach: on the one hand, surveys are conducted using specialized staff from various public entities; on the other, AI-based models (including neural networks and transformer architectures) are used to estimate the DT level of the organizations automatically. Our approach has been applied to a real-world case study involving local public administrations in the Valencian Community (Spain) and shown effective performance in assessing DT. While the proposed methodology has been validated in a specific local context, its modular structure and dual-source data foundation support its international scalability, acknowledging that administrative, regulatory, and DT maturity factors may condition its broader applicability. The experiments carried out in this work include (i) the creation of a domain-specific corpus derived from the surveys and websites of several organizations, used to train the proposed models; (ii) the use and comparison of diverse AI methods; and (iii) the validation of our approach using real data. The integration of technologies such as the IoT, sensor networks, and AI-based analytics can significantly support resilient, agile urban environments and the transition towards more effective and sustainable Smart City models. 

---
# LayoutAgent: A Vision-Language Agent Guided Compositional Diffusion for Spatial Layout Planning 

**Authors**: Zezhong Fan, Xiaohan Li, Luyi Ma, Kai Zhao, Liang Peng, Topojoy Biswas, Evren Korpeoglu, Kaushiki Nag, Kannan Achan  

**Link**: [PDF](https://arxiv.org/pdf/2509.22720)  

**Abstract**: Designing realistic multi-object scenes requires not only generating images, but also planning spatial layouts that respect semantic relations and physical plausibility. On one hand, while recent advances in diffusion models have enabled high-quality image generation, they lack explicit spatial reasoning, leading to unrealistic object layouts. On the other hand, traditional spatial planning methods in robotics emphasize geometric and relational consistency, but they struggle to capture semantic richness in visual scenes. To bridge this gap, in this paper, we propose LayoutAgent, an agentic framework that unifies vision-language reasoning with compositional diffusion for layout generation. Given multiple input images with target objects in them, our method first employs visual-language model to preprocess the inputs through segmentation, object size estimation, scene graph construction, and prompt rewriting. Then we leverage compositional diffusion-a method traditionally used in robotics-to synthesize bounding boxes that respect object relations encoded in the scene graph for spatial layouts. In the end, a foreground-conditioned image generator composes the complete scene by rendering the objects into the planned layout guided by designed prompts. Experiments demonstrate that LayoutAgent outperforms other state-of-the-art layout generation models in layout coherence, spatial realism and aesthetic alignment. 

---
# IBiT: Utilizing Inductive Biases to Create a More Data Efficient Attention Mechanism 

**Authors**: Adithya Giri  

**Link**: [PDF](https://arxiv.org/pdf/2509.22719)  

**Abstract**: In recent years, Transformer-based architectures have become the dominant method for Computer Vision applications. While Transformers are explainable and scale well with dataset size, they lack the inductive biases of Convolutional Neural Networks. While these biases may be learned on large datasets, we show that introducing these inductive biases through learned masks allow Vision Transformers to learn on much smaller datasets without Knowledge Distillation. These Transformers, which we call Inductively Biased Image Transformers (IBiT), are significantly more accurate on small datasets, while retaining the explainability Transformers. 

---
# TRUEBench: Can LLM Response Meet Real-world Constraints as Productivity Assistant? 

**Authors**: Jiho Park, Jongyoon Song, Minjin Choi, Kyuho Heo, Taehun Huh, Ji Won Kim  

**Link**: [PDF](https://arxiv.org/pdf/2509.22715)  

**Abstract**: Large language models (LLMs) are increasingly integral as productivity assistants, but existing benchmarks fall short in rigorously evaluating their real-world instruction-following capabilities. Current benchmarks often (i) lack sufficient multilinguality, (ii) fail to capture the implicit constraints inherent in user requests, and (iii) overlook the complexities of multi-turn dialogue. To address these critical gaps and provide a more realistic assessment, we introduce TRUEBench (Trustworthy Real-world Usage Evaluation Benchmark)1, a novel benchmark specifically designed for LLM-based productivity assistants. TRUEBench distinguishes itself by featuring input prompts across 12 languages, incorporating intra-instance multilingual instructions, employing rigorous evaluation criteria to capture both explicit and implicit constraints, and including complex multi-turn dialogue scenarios with both accumulating constraints and context switches. Furthermore, to ensure reliability in evaluation, we refined constraints using an LLM validator. Extensive experiments demonstrate that TRUEBench presents significantly greater challenges than existing benchmarks; for instance, a strong model like OpenAI o1 achieved only a 69.07% overall pass rate. TRUEBench offers a demanding and realistic assessment of LLMs in practical productivity settings, highlighting their capabilities and limitations. 

---
# Localizing Adversarial Attacks To Produces More Imperceptible Noise 

**Authors**: Pavan Reddy, Aditya Sanjay Gujral  

**Link**: [PDF](https://arxiv.org/pdf/2509.22710)  

**Abstract**: Adversarial attacks in machine learning traditionally focus on global perturbations to input data, yet the potential of localized adversarial noise remains underexplored. This study systematically evaluates localized adversarial attacks across widely-used methods, including FGSM, PGD, and C&W, to quantify their effectiveness, imperceptibility, and computational efficiency. By introducing a binary mask to constrain noise to specific regions, localized attacks achieve significantly lower mean pixel perturbations, higher Peak Signal-to-Noise Ratios (PSNR), and improved Structural Similarity Index (SSIM) compared to global attacks. However, these benefits come at the cost of increased computational effort and a modest reduction in Attack Success Rate (ASR). Our results highlight that iterative methods, such as PGD and C&W, are more robust to localization constraints than single-step methods like FGSM, maintaining higher ASR and imperceptibility metrics. This work provides a comprehensive analysis of localized adversarial attacks, offering practical insights for advancing attack strategies and designing robust defensive systems. 

---
# GZSL-MoE: Apprentissage G{é}n{é}ralis{é} Z{é}ro-Shot bas{é} sur le M{é}lange d'Experts pour la Segmentation S{é}mantique de Nuages de Points 3DAppliqu{é} {à} un Jeu de Donn{é}es d'Environnement de Collaboration Humain-Robot 

**Authors**: Ahed Alboody  

**Link**: [PDF](https://arxiv.org/pdf/2509.22708)  

**Abstract**: Generative Zero-Shot Learning approach (GZSL) has demonstrated significant potential in 3D point cloud semantic segmentation tasks. GZSL leverages generative models like GANs or VAEs to synthesize realistic features (real features) of unseen classes. This allows the model to label unseen classes during testing, despite being trained only on seen classes. In this context, we introduce the Generalized Zero-Shot Learning based-upon Mixture-of-Experts (GZSL-MoE) model. This model incorporates Mixture-of-Experts layers (MoE) to generate fake features that closely resemble real features extracted using a pre-trained KPConv (Kernel Point Convolution) model on seen classes. The main contribution of this paper is the integration of Mixture-of-Experts into the Generator and Discriminator components of the Generative Zero-Shot Learning model for 3D point cloud semantic segmentation, applied to the COVERED dataset (CollabOratiVE Robot Environment Dataset) for Human-Robot Collaboration (HRC) environments. By combining the Generative Zero-Shot Learning model with Mixture-of- Experts, GZSL-MoE for 3D point cloud semantic segmentation provides a promising solution for understanding complex 3D environments, especially when comprehensive training data for all object classes is unavailable. The performance evaluation of the GZSL-MoE model highlights its ability to enhance performance on both seen and unseen classes. Keywords Generalized Zero-Shot Learning (GZSL), 3D Point Cloud, 3D Semantic Segmentation, Human-Robot Collaboration, COVERED (CollabOratiVE Robot Environment Dataset), KPConv, Mixture-of Experts 

---
# Intelligent Load Balancing in Cloud Computer Systems 

**Authors**: Leszek Sliwko  

**Link**: [PDF](https://arxiv.org/pdf/2509.22704)  

**Abstract**: Cloud computing is an established technology allowing users to share resources on a large scale, never before seen in IT history. A cloud system connects multiple individual servers in order to process related tasks in several environments at the same time. Clouds are typically more cost-effective than single computers of comparable computing performance. The sheer physical size of the system itself means that thousands of machines may be involved. The focus of this research was to design a strategy to dynamically allocate tasks without overloading Cloud nodes which would result in system stability being maintained at minimum cost. This research has added the following new contributions to the state of knowledge: (i) a novel taxonomy and categorisation of three classes of schedulers, namely OS-level, Cluster and Big Data, which highlight their unique evolution and underline their different objectives; (ii) an abstract model of cloud resources utilisation is specified, including multiple types of resources and consideration of task migration costs; (iii) a virtual machine live migration was experimented with in order to create a formula which estimates the network traffic generated by this process; (iv) a high-fidelity Cloud workload simulator, based on a month-long workload traces from Google's computing cells, was created; (v) two possible approaches to resource management were proposed and examined in the practical part of the manuscript: the centralised metaheuristic load balancer and the decentralised agent-based system. The project involved extensive experiments run on the University of Westminster HPC cluster, and the promising results are presented together with detailed discussions and a conclusion. 

---
# AccessEval: Benchmarking Disability Bias in Large Language Models 

**Authors**: Srikant Panda, Amit Agarwal, Hitesh Laxmichand Patel  

**Link**: [PDF](https://arxiv.org/pdf/2509.22703)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed across diverse domains but often exhibit disparities in how they handle real-life queries. To systematically investigate these effects within various disability contexts, we introduce \textbf{AccessEval (Accessibility Evaluation)}, a benchmark evaluating 21 closed- and open-source LLMs across 6 real-world domains and 9 disability types using paired Neutral and Disability-Aware Queries. We evaluated model outputs with metrics for sentiment, social perception, and factual accuracy.
Our analysis reveals that responses to disability-aware queries tend to have a more negative tone, increased stereotyping, and higher factual error compared to neutral queries. These effects show notable variation by domain and disability type, with disabilities affecting hearing, speech, and mobility disproportionately impacted. These disparities reflect persistent forms of ableism embedded in model behavior.
By examining model performance in real-world decision-making contexts, we better illuminate how such biases can translate into tangible harms for disabled users. This framing helps bridges the gap between technical evaluation and user impact, reinforcing importance of bias mitigation in day-to-day applications. Our dataset is publicly available at: this https URL 

---
# Enhancing Cluster Scheduling in HPC: A Continuous Transfer Learning for Real-Time Optimization 

**Authors**: Leszek Sliwko, Jolanta Mizera-Pietraszko  

**Link**: [PDF](https://arxiv.org/pdf/2509.22701)  

**Abstract**: This study presents a machine learning-assisted approach to optimize task scheduling in cluster systems, focusing on node-affinity constraints. Traditional schedulers like Kubernetes struggle with real-time adaptability, whereas the proposed continuous transfer learning model evolves dynamically during operations, minimizing retraining needs. Evaluated on Google Cluster Data, the model achieves over 99% accuracy, reducing computational overhead and improving scheduling latency for constrained tasks. This scalable solution enables real-time optimization, advancing machine learning integration in cluster management and paving the way for future adaptive scheduling strategies. 

---
# Advancing Audio-Visual Navigation Through Multi-Agent Collaboration in 3D Environments 

**Authors**: Hailong Zhang, Yinfeng Yu, Liejun Wang, Fuchun Sun, Wendong Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2509.22698)  

**Abstract**: Intelligent agents often require collaborative strategies to achieve complex tasks beyond individual capabilities in real-world scenarios. While existing audio-visual navigation (AVN) research mainly focuses on single-agent systems, their limitations emerge in dynamic 3D environments where rapid multi-agent coordination is critical, especially for time-sensitive applications like emergency response. This paper introduces MASTAVN (Multi-Agent Scalable Transformer Audio-Visual Navigation), a scalable framework enabling two agents to collaboratively localize and navigate toward an audio target in shared 3D environments. By integrating cross-agent communication protocols and joint audio-visual fusion mechanisms, MASTAVN enhances spatial reasoning and temporal synchronization. Through rigorous evaluation in photorealistic 3D simulators (Replica and Matterport3D), MASTAVN achieves significant reductions in task completion time and notable improvements in navigation success rates compared to single-agent and non-collaborative baselines. This highlights the essential role of spatiotemporal coordination in multi-agent systems. Our findings validate MASTAVN's effectiveness in time-sensitive emergency scenarios and establish a paradigm for advancing scalable multi-agent embodied intelligence in complex 3D environments. 

---
# Learning Hyperspectral Images with Curated Text Prompts for Efficient Multimodal Alignment 

**Authors**: Abhiroop Chatterjee, Susmita Ghosh  

**Link**: [PDF](https://arxiv.org/pdf/2509.22697)  

**Abstract**: As data requirements continue to grow, efficient learning increasingly depends on the curation and distillation of high-value data rather than brute-force scaling of model sizes. In the case of a hyperspectral image (HSI), the challenge is amplified by the high-dimensional 3D voxel structure, where each spatial location is associated with hundreds of contiguous spectral channels. While vision and language models have been optimized effectively for natural image or text tasks, their cross-modal alignment in the hyperspectral domain remains an open and underexplored problem. In this article, we make an attempt to optimize a Vision-Language Model (VLM) for hyperspectral scene understanding by exploiting a CLIP-style contrastive training framework. Our framework maps voxel-level embeddings from a vision backbone onto the latent space of a frozen large embedding model (LEM), where a trainable probe aligns vision features with the model's textual token representations. The two modalities are aligned via a contrastive loss restricted to a curated set of hard (closest wrong classes) and semi-hard (random distractors) negatives, along with positive pairs. To further enhance alignment, descriptive prompts that encode class semantics are introduced and act as structured anchors for the HSI embeddings. It is seen that the proposed method updates only 0.07 percent of the total parameters, yet yields state-of-the-art performance. For example, on Indian Pines (IP) the model produces better results over unimodal and multimodal baselines by +0.92 Overall Accuracy (OA) and +1.60 Kappa ($\kappa$), while on Pavia University (PU) data it provides gains of +0.69 OA and +0.90 $\kappa$. Moreover, this is achieved with the set of parameters, nearly 50$\times$ smaller than DCTN and 90$\times$ smaller than SS-TMNet. 

---
# PISA: An AI Pipeline for Interpretable-by-design Survival Analysis Providing Multiple Complexity-Accuracy Trade-off Models 

**Authors**: Thalea Schlender, Catharina J.A. Romme, Yvette M. van der Linden, Luc R.C.W. van Lonkhuijzen, Peter A.N. Bosman, Tanja Alderliesten  

**Link**: [PDF](https://arxiv.org/pdf/2509.22673)  

**Abstract**: Survival analysis is central to clinical research, informing patient prognoses, guiding treatment decisions, and optimising resource allocation. Accurate time-to-event predictions not only improve quality of life but also reveal risk factors that shape clinical practice. For these models to be relevant in healthcare, interpretability is critical: predictions must be traceable to patient-specific characteristics, and risk factors should be identifiable to generate actionable insights for both clinicians and researchers. Traditional survival models often fail to capture non-linear interactions, while modern deep learning approaches, though powerful, are limited by poor interpretability.
We propose a Pipeline for Interpretable Survival Analysis (PISA) - a pipeline that provides multiple survival analysis models that trade off complexity and performance. Using multiple-feature, multi-objective feature engineering, PISA transforms patient characteristics and time-to-event data into multiple survival analysis models, providing valuable insights into the survival prediction task. Crucially, every model is converted into simple patient stratification flowcharts supported by Kaplan-Meier curves, whilst not compromising on performance. While PISA is model-agnostic, we illustrate its flexibility through applications of Cox regression and shallow survival trees, the latter avoiding proportional hazards assumptions.
Applied to two clinical benchmark datasets, PISA produced interpretable survival models and intuitive stratification flowcharts whilst achieving state-of-the-art performances. Revisiting a prior departmental study further demonstrated its capacity to automate survival analysis workflows in real-world clinical research. 

---
# Next Point-of-interest (POI) Recommendation Model Based on Multi-modal Spatio-temporal Context Feature Embedding 

**Authors**: Lingyu Zhang, Guobin Wu, Yan Wang, Pengfei Xu, Jian Liang, Xuan Song, Yunhai Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.22661)  

**Abstract**: The next Point-of-interest (POI) recommendation is mainly based on sequential traffic information to predict the user's next boarding point location. This is a highly regarded and widely applied research task in the field of intelligent transportation, and there have been many research results to date. Traditional POI prediction models primarily rely on short-term traffic sequence information, often neglecting both long-term and short-term preference data, as well as crucial spatiotemporal context features in user behavior. To address this issue, this paper introduces user long-term preference information and key spatiotemporal context information, and proposes a POI recommendation model based on multimodal spatiotemporal context feature embedding. The model extracts long-term preference features and key spatiotemporal context features from traffic data through modules such as spatiotemporal feature processing, multimodal embedding, and self-attention aggregation. It then uses a weighted fusion method to dynamically adjust the weights of long-term and short-term features based on users' historical behavior patterns and the current context. Finally, the fused features are matched using attention, and the probability of each location candidate becoming the next location is calculated. This paper conducts experimental verification on multiple transportation datasets, and the results show that the POI prediction model combining multiple types of features has higher prediction accuracy than existing SOTA models and methods. 

---
# Fairness for niche users and providers: algorithmic choice and profile portability 

**Authors**: Elizabeth McKinnie, Anas Buhayh, Clement Canel, Robin Burke  

**Link**: [PDF](https://arxiv.org/pdf/2509.22660)  

**Abstract**: Ensuring fair outcomes for multiple stakeholders in recommender systems has been studied mostly in terms of algorithmic interventions: building new models with better fairness properties, or using reranking to improve outcomes from an existing algorithm. What has rarely been studied is structural changes in the recommendation ecosystem itself. Our work explores the fairness impact of algorithmic pluralism, the idea that the recommendation algorithm is decoupled from the platform through which users access content, enabling user choice in algorithms. Prior work using a simulation approach has shown that niche consumers and (especially) niche providers benefit from algorithmic choice. In this paper, we use simulation to explore the question of profile portability, to understand how different policies regarding the handling of user profiles interact with fairness outcomes for consumers and providers. 

---
# How good are LLMs at Retrieving Documents in a Specific Domain? 

**Authors**: Nafis Tanveer Islam, Zhiming Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.22658)  

**Abstract**: Classical search engines using indexing methods in data infrastructures primarily allow keyword-based queries to retrieve content. While these indexing-based methods are highly scalable and efficient, due to a lack of an appropriate evaluation dataset and a limited understanding of semantics, they often fail to capture the user's intent and generate incomplete responses during evaluation. This problem also extends to domain-specific search systems that utilize a Knowledge Base (KB) to access data from various research infrastructures. Research infrastructures (RIs) from the environmental and earth science domain, which encompass the study of ecosystems, biodiversity, oceanography, and climate change, generate, share, and reuse large volumes of data. While there are attempts to provide a centralized search service using Elasticsearch as a knowledge base, they also face similar challenges in understanding queries with multiple intents. To address these challenges, we proposed an automated method to curate a domain-specific evaluation dataset to analyze the capability of a search system. Furthermore, we incorporate the Retrieval of Augmented Generation (RAG), powered by Large Language Models (LLMs), for high-quality retrieval of environmental domain data using natural language queries. Our quantitative and qualitative analysis of the evaluation dataset shows that LLM-based systems for information retrieval return results with higher precision when understanding queries with multiple intents, compared to Elasticsearch-based systems. 

---
# GOAT: A Large Dataset of Paired Guitar Audio Recordings and Tablatures 

**Authors**: Jackson Loth, Pedro Sarmento, Saurjya Sarkar, Zixun Guo, Mathieu Barthet, Mark Sandler  

**Link**: [PDF](https://arxiv.org/pdf/2509.22655)  

**Abstract**: In recent years, the guitar has received increased attention from the music information retrieval (MIR) community driven by the challenges posed by its diverse playing techniques and sonic characteristics. Mainly fueled by deep learning approaches, progress has been limited by the scarcity and limited annotations of datasets. To address this, we present the Guitar On Audio and Tablatures (GOAT) dataset, comprising 5.9 hours of unique high-quality direct input audio recordings of electric guitars from a variety of different guitars and players. We also present an effective data augmentation strategy using guitar amplifiers which delivers near-unlimited tonal variety, of which we provide a starting 29.5 hours of audio. Each recording is annotated using guitar tablatures, a guitar-specific symbolic format supporting string and fret numbers, as well as numerous playing techniques. For this we utilise both the Guitar Pro format, a software for tablature playback and editing, and a text-like token encoding. Furthermore, we present competitive results using GOAT for MIDI transcription and preliminary results for a novel approach to automatic guitar tablature transcription. We hope that GOAT opens up the possibilities to train novel models on a wide variety of guitar-related MIR tasks, from synthesis to transcription to playing technique detection. 

---
# Sustainable LSTM-Based Precoding for RIS-Aided mmWave MIMO Systems with Implicit CSI 

**Authors**: Po-Heng Chou, Jiun-Jia Wu, Wan-Jen Huang, Ronald Y. Chang  

**Link**: [PDF](https://arxiv.org/pdf/2509.12658)  

**Abstract**: In this paper, we propose a sustainable long short-term memory (LSTM)-based precoding framework for reconfigurable intelligent surface (RIS)-assisted millimeter-wave (mmWave) MIMO systems. Instead of explicit channel state information (CSI) estimation, the framework exploits uplink pilot sequences to implicitly learn channel characteristics, reducing both pilot overhead and inference complexity. Practical hardware constraints are addressed by incorporating the phase-dependent amplitude model of RIS elements, while a multi-label training strategy improves robustness when multiple near-optimal codewords yield comparable performance. Simulations show that the proposed design achieves over 90% of the spectral efficiency of exhaustive search (ES) with only 2.2% of its computation time, cutting energy consumption by nearly two orders of magnitude. The method also demonstrates resilience under distribution mismatch and scalability to larger RIS arrays, making it a practical and energy-efficient solution for sustainable 6G wireless networks. 

---
# How are Scientific Concepts Birthed? Typing Rules of Concept Formation in Theoretical Physics Reasoning 

**Authors**: Omar Aguilar, Anthony Aguirre  

**Link**: [PDF](https://arxiv.org/pdf/2509.10740)  

**Abstract**: This work aims to formalize some of the ways scientific concepts are formed in the process of theoretical physics discovery. Since this may at first seem like a task beyond the scope of the exact sciences (natural and formal sciences), we begin by presenting arguments for why scientific concept formation can be formalized. Then, we introduce type theory as a natural and well-suited framework for this formalization. We formalize what we call "ways of discovering new concepts" including concept distinction, property preservation, and concept change, as cognitive typing rules. Next, we apply these cognitive typing rules to two case studies of conceptual discovery in the history of physics: Einstein's reasoning leading to the impossibility of frozen waves, and his conceptual path to the relativity of time. In these historical episodes, we recast what a physicist might informally call "ways of discovering new scientific concepts" as compositional typing rules built from cognitive typing rules - thus formalizing them as scientific discovery mechanisms. Lastly, we computationally model the type-theoretic reconstruction of Einstein's conceptual path to the relativity of time as a program synthesis task. 

---
# Green Learning for STAR-RIS mmWave Systems with Implicit CSI 

**Authors**: Yu-Hsiang Huang, Po-Heng Chou, Wan-Jen Huang, Walid Saad, C.-C. Jay Kuo  

**Link**: [PDF](https://arxiv.org/pdf/2509.06820)  

**Abstract**: In this paper, a green learning (GL)-based precoding framework is proposed for simultaneously transmitting and reflecting reconfigurable intelligent surface (STAR-RIS)-aided millimeter-wave (mmWave) MIMO broadcasting systems. Motivated by the growing emphasis on environmental sustainability in future 6G networks, this work adopts a broadcasting transmission architecture for scenarios where multiple users share identical information, improving spectral efficiency and reducing redundant transmissions and power consumption. Different from conventional optimization methods, such as block coordinate descent (BCD) that require perfect channel state information (CSI) and iterative computation, the proposed GL framework operates directly on received uplink pilot signals without explicit CSI estimation. Unlike deep learning (DL) approaches that require CSI-based labels for training, the proposed GL approach also avoids deep neural networks and backpropagation, leading to a more lightweight design. Although the proposed GL framework is trained with supervision generated by BCD under full CSI, inference is performed in a fully CSI-free manner. The proposed GL integrates subspace approximation with adjusted bias (Saab), relevant feature test (RFT)-based supervised feature selection, and eXtreme gradient boosting (XGBoost)-based decision learning to jointly predict the STAR-RIS coefficients and transmit precoder. Simulation results show that the proposed GL approach achieves competitive spectral efficiency compared to BCD and DL-based models, while reducing floating-point operations (FLOPs) by over four orders of magnitude. These advantages make the proposed GL approach highly suitable for real-time deployment in energy- and hardware-constrained broadcasting scenarios. 

---
# Agentic DDQN-Based Scheduling for Licensed and Unlicensed Band Allocation in Sidelink Networks 

**Authors**: Po-Heng Chou, Pin-Qi Fu, Walid Saad, Li-Chun Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.06775)  

**Abstract**: In this paper, we present an agentic double deep Q-network (DDQN) scheduler for licensed/unlicensed band allocation in New Radio (NR) sidelink (SL) networks. Beyond conventional reward-seeking reinforcement learning (RL), the agent perceives and reasons over a multi-dimensional context that jointly captures queueing delay, link quality, coexistence intensity, and switching stability. A capacity-aware, quality of service (QoS)-constrained reward aligns the agent with goal-oriented scheduling rather than static thresholding. Under constrained bandwidth, the proposed design reduces blocking by up to 87.5% versus threshold policies while preserving throughput, highlighting the value of context-driven decisions in coexistence-limited NR SL networks. The proposed scheduler is an embodied agent (E-agent) tailored for task-specific, resource-efficient operation at the network edge. 

---
# YOLO-based Bearing Fault Diagnosis With Continuous Wavelet Transform 

**Authors**: Po-Heng Chou, Wei-Lung Mao, Ru-Ping Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.03070)  

**Abstract**: This letter proposes a YOLO-based framework for spatial bearing fault diagnosis using time-frequency spectrograms derived from continuous wavelet transform (CWT). One-dimensional vibration signals are first transformed into time-frequency spectrograms using Morlet wavelets to capture transient fault signatures. These spectrograms are then processed by YOLOv9, v10, and v11 models to classify fault types. Evaluated on three benchmark datasets, including Case Western Reserve University (CWRU), Paderborn University (PU), and Intelligent Maintenance System (IMS), the proposed CWT-YOLO pipeline achieves significantly higher accuracy and generalizability than the baseline MCNN-LSTM model. Notably, YOLOv11 reaches mAP scores of 99.4% (CWRU), 97.8% (PU), and 99.5% (IMS). In addition, its region-aware detection mechanism enables direct visualization of fault locations in spectrograms, offering a practical solution for condition monitoring in rotating machinery. 

---
# BenLOC: A Benchmark for Learning to Configure MIP Optimizers 

**Authors**: Hongpei Li, Ziyan He, Yufei Wang, Wenting Tu, Shanwen Pu, Qi Deng, Dongdong Ge  

**Link**: [PDF](https://arxiv.org/pdf/2506.02752)  

**Abstract**: The automatic configuration of Mixed-Integer Programming (MIP) optimizers has become increasingly critical as the large number of configurations can significantly affect solver performance. Yet the lack of standardized evaluation frameworks has led to data leakage and over-optimistic claims, as prior studies often rely on homogeneous datasets and inconsistent experimental setups. To promote a fair evaluation process, we present BenLOC, a comprehensive benchmark and open-source toolkit, which not only offers an end-to-end pipeline for learning instance-wise MIP optimizer configurations, but also standardizes dataset selection, train-test splits, feature engineering and baseline choice for unbiased and comprehensive evaluations. Leveraging this framework, we conduct an empirical analysis on five well-established MIP datasets and compare classical machine learning models with handcrafted features against state-of-the-art deep-learning techniques. The results demonstrate the importance of datasets, features and baseline criteria proposed by BenLOC and the effectiveness of BenLOC in providing unbiased and comprehensive evaluations. 

---
# Prosody-Adaptable Audio Codecs for Zero-Shot Voice Conversion via In-Context Learning 

**Authors**: Junchuan Zhao, Xintong Wang, Ye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.15402)  

**Abstract**: Recent advances in discrete audio codecs have significantly improved speech representation modeling, while codec language models have enabled in-context learning for zero-shot speech synthesis. Inspired by this, we propose a voice conversion (VC) model within the VALLE-X framework, leveraging its strong in-context learning capabilities for speaker adaptation. To enhance prosody control, we introduce a prosody-aware audio codec encoder (PACE) module, which isolates and refines prosody from other sources, improving expressiveness and control. By integrating PACE into our VC model, we achieve greater flexibility in prosody manipulation while preserving speaker timbre. Experimental evaluation results demonstrate that our approach outperforms baseline VC systems in prosody preservation, timbre consistency, and overall naturalness, surpassing baseline VC systems. 

---
