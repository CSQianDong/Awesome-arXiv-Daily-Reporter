# D2PPO: Diffusion Policy Policy Optimization with Dispersive Loss 

**Authors**: Guowei Zou, Weibing Li, Hejun Wu, Yukun Qian, Yuhang Wang, Haitao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02644)  

**Abstract**: Diffusion policies excel at robotic manipulation by naturally modeling multimodal action distributions in high-dimensional spaces. Nevertheless, diffusion policies suffer from diffusion representation collapse: semantically similar observations are mapped to indistinguishable features, ultimately impairing their ability to handle subtle but critical variations required for complex robotic manipulation. To address this problem, we propose D2PPO (Diffusion Policy Policy Optimization with Dispersive Loss). D2PPO introduces dispersive loss regularization that combats representation collapse by treating all hidden representations within each batch as negative pairs. D2PPO compels the network to learn discriminative representations of similar observations, thereby enabling the policy to identify subtle yet crucial differences necessary for precise manipulation. In evaluation, we find that early-layer regularization benefits simple tasks, while late-layer regularization sharply enhances performance on complex manipulation tasks. On RoboMimic benchmarks, D2PPO achieves an average improvement of 22.7% in pre-training and 26.1% after fine-tuning, setting new SOTA results. In comparison with SOTA, results of real-world experiments on a Franka Emika Panda robot show the excitingly high success rate of our method. The superiority of our method is especially evident in complex tasks. Project page: this https URL 

---
# Actionable Counterfactual Explanations Using Bayesian Networks and Path Planning with Applications to Environmental Quality Improvement 

**Authors**: Enrique Valero-Leal, Pedro Larrañaga, Concha Bielza  

**Link**: [PDF](https://arxiv.org/pdf/2508.02634)  

**Abstract**: Counterfactual explanations study what should have changed in order to get an alternative result, enabling end-users to understand machine learning mechanisms with counterexamples. Actionability is defined as the ability to transform the original case to be explained into a counterfactual one. We develop a method for actionable counterfactual explanations that, unlike predecessors, does not directly leverage training data. Rather, data is only used to learn a density estimator, creating a search landscape in which to apply path planning algorithms to solve the problem and masking the endogenous data, which can be sensitive or private. We put special focus on estimating the data density using Bayesian networks, demonstrating how their enhanced interpretability is useful in high-stakes scenarios in which fairness is raising concern. Using a synthetic benchmark comprised of 15 datasets, our proposal finds more actionable and simpler counterfactuals than the current state-of-the-art algorithms. We also test our algorithm with a real-world Environmental Protection Agency dataset, facilitating a more efficient and equitable study of policies to improve the quality of life in United States of America counties. Our proposal captures the interaction of variables, ensuring equity in decisions, as policies to improve certain domains of study (air, water quality, etc.) can be detrimental in others. In particular, the sociodemographic domain is often involved, where we find important variables related to the ongoing housing crisis that can potentially have a severe negative impact on communities. 

---
# What Is Your AI Agent Buying? Evaluation, Implications and Emerging Questions for Agentic E-Commerce 

**Authors**: Amine Allouah, Omar Besbes, Josué D Figueroa, Yash Kanoria, Akshit Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2508.02630)  

**Abstract**: Online marketplaces will be transformed by autonomous AI agents acting on behalf of consumers. Rather than humans browsing and clicking, vision-language-model (VLM) agents can parse webpages, evaluate products, and transact. This raises a fundamental question: what do AI agents buy, and why? We develop ACES, a sandbox environment that pairs a platform-agnostic VLM agent with a fully programmable mock marketplace to study this question. We first conduct basic rationality checks in the context of simple tasks, and then, by randomizing product positions, prices, ratings, reviews, sponsored tags, and platform endorsements, we obtain causal estimates of how frontier VLMs actually shop. Models show strong but heterogeneous position effects: all favor the top row, yet different models prefer different columns, undermining the assumption of a universal "top" rank. They penalize sponsored tags and reward endorsements. Sensitivities to price, ratings, and reviews are directionally human-like but vary sharply in magnitude across models. Motivated by scenarios where sellers use AI agents to optimize product listings, we show that a seller-side agent that makes minor tweaks to product descriptions, targeting AI buyer preferences, can deliver substantial market-share gains if AI-mediated shopping dominates. We also find that modal product choices can differ across models and, in some cases, demand may concentrate on a few select products, raising competition questions. Together, our results illuminate how AI agents may behave in e-commerce settings and surface concrete seller strategy, platform design, and regulatory questions in an AI-mediated ecosystem. 

---
# Noosemia: toward a Cognitive and Phenomenological Account of Intentionality Attribution in Human-Generative AI Interaction 

**Authors**: Enrico De Santis, Antonello Rizzi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02622)  

**Abstract**: This paper introduces and formalizes Noosemia, a novel cognitive-phenomenological phenomenon emerging from human interaction with generative AI systems, particularly those enabling dialogic or multimodal exchanges. We propose a multidisciplinary framework to explain how, under certain conditions, users attribute intentionality, agency, and even interiority to these systems - a process grounded not in physical resemblance, but in linguistic performance, epistemic opacity, and emergent technological complexity. By linking an LLM declination of meaning holism to our technical notion of the LLM Contextual Cognitive Field, we clarify how LLMs construct meaning relationally and how coherence and a simulacrum of agency arise at the human-AI interface. The analysis situates noosemia alongside pareidolia, animism, the intentional stance and the uncanny valley, distinguishing its unique characteristics. We also introduce a-noosemia to describe the phenomenological withdrawal of such projections. The paper concludes with reflections on the broader philosophical, epistemological, and social implications of noosemic dynamics and directions for future research. 

---
# HealthFlow: A Self-Evolving AI Agent with Meta Planning for Autonomous Healthcare Research 

**Authors**: Yinghao Zhu, Yifan Qi, Zixiang Wang, Lei Gu, Dehao Sui, Haoran Hu, Xichen Zhang, Ziyi He, Liantao Ma, Lequan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02621)  

**Abstract**: The efficacy of AI agents in healthcare research is hindered by their reliance on static, predefined strategies. This creates a critical limitation: agents can become better tool-users but cannot learn to become better strategic planners, a crucial skill for complex domains like healthcare. We introduce HealthFlow, a self-evolving AI agent that overcomes this limitation through a novel meta-level evolution mechanism. HealthFlow autonomously refines its own high-level problem-solving policies by distilling procedural successes and failures into a durable, strategic knowledge base. To anchor our research and facilitate reproducible evaluation, we introduce EHRFlowBench, a new benchmark featuring complex, realistic health data analysis tasks derived from peer-reviewed clinical research. Our comprehensive experiments demonstrate that HealthFlow's self-evolving approach significantly outperforms state-of-the-art agent frameworks. This work marks a necessary shift from building better tool-users to designing smarter, self-evolving task-managers, paving the way for more autonomous and effective AI for scientific discovery. 

---
# CAMA: Enhancing Mathematical Reasoning in Large Language Models with Causal Knowledge 

**Authors**: Lei Zan, Keli Zhang, Ruichu Cai, Lujia Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.02583)  

**Abstract**: Large Language Models (LLMs) have demonstrated strong performance across a wide range of tasks, yet they still struggle with complex mathematical reasoning, a challenge fundamentally rooted in deep structural dependencies. To address this challenge, we propose \textbf{CA}usal \textbf{MA}thematician (\textbf{CAMA}), a two-stage causal framework that equips LLMs with explicit, reusable mathematical structure. In the learning stage, CAMA first constructs the \textbf{M}athematical \textbf{C}ausal \textbf{G}raph (\textbf{MCG}), a high-level representation of solution strategies, by combining LLM priors with causal discovery algorithms applied to a corpus of question-solution pairs. The resulting MCG encodes essential knowledge points and their causal dependencies. To better align the graph with downstream reasoning tasks, CAMA further refines the MCG through iterative feedback derived from a selected subset of the question-solution pairs. In the reasoning stage, given a new question, CAMA dynamically extracts a task-relevant subgraph from the MCG, conditioned on both the question content and the LLM's intermediate reasoning trace. This subgraph, which encodes the most pertinent knowledge points and their causal dependencies, is then injected back into the LLM to guide its reasoning process. Empirical results on real-world datasets show that CAMA significantly improves LLM performance on challenging mathematical problems. Furthermore, our experiments demonstrate that structured guidance consistently outperforms unstructured alternatives, and that incorporating asymmetric causal relationships yields greater improvements than using symmetric associations alone. 

---
# Accurate and Interpretable Postmenstrual Age Prediction via Multimodal Large Language Model 

**Authors**: Qifan Chen, Jin Cui, Cindy Duan, Yushuo Han, Yifei Shi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02525)  

**Abstract**: Accurate estimation of postmenstrual age (PMA) at scan is crucial for assessing neonatal development and health. While deep learning models have achieved high accuracy in predicting PMA from brain MRI, they often function as black boxes, offering limited transparency and interpretability in clinical decision support. In this work, we address the dual challenge of accuracy and interpretability by adapting a multimodal large language model (MLLM) to perform both precise PMA prediction and clinically relevant explanation generation. We introduce a parameter-efficient fine-tuning (PEFT) strategy using instruction tuning and Low-Rank Adaptation (LoRA) applied to the Qwen2.5-VL-7B model. The model is trained on four 2D cortical surface projection maps derived from neonatal MRI scans. By employing distinct prompts for training and inference, our approach enables the MLLM to handle a regression task during training and generate clinically relevant explanations during inference. The fine-tuned model achieves a low prediction error with a 95 percent confidence interval of 0.78 to 1.52 weeks, while producing interpretable outputs grounded in developmental features, marking a significant step toward transparent and trustworthy AI systems in perinatal neuroscience. 

---
# Test-time Prompt Intervention 

**Authors**: Chenxu Yang, Qingyi Si, Mz Dai, Dingyu Yao, Mingyu Zheng, Minghui Chen, Zheng Lin, Weiping Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02511)  

**Abstract**: Test-time compute has led to remarkable success in the large language model (LLM) community, particularly for complex tasks, where longer chains of thought (CoTs) are generated to enhance reasoning capabilities. However, growing evidence reveals that such reasoning models often produce CoTs plagued by excessive redundancy, including unnecessary verification steps and repetitive reasoning shifts. The root cause lies in post-training of them that overly rely on outcome reward paradigms, as the data of process reward paradigms, which regulate intermediate reasoning steps, is difficult to construct at scale. To address this, we propose PI, a novel framework for Test-time Prompt Intervention. PI provides an interface to dynamically guide and regulate reasoning paths during inference through timely (When module) and proper (How module) interventions and post-intervention sampling (Which module). This allows human problem-solving expertise and cognitive science principles to be seamlessly integrated into LLMs' reasoning processes, enhancing controllability and interpretability. Extensive experiments across multiple models and datasets demonstrate that PI significantly shortens CoTs while reducing hallucination, yielding more concise and reliable reasoning. 

---
# OptiHive: Ensemble Selection for LLM-Based Optimization via Statistical Modeling 

**Authors**: Maxime Bouscary, Saurabh Amin  

**Link**: [PDF](https://arxiv.org/pdf/2508.02503)  

**Abstract**: LLM-based solvers have emerged as a promising means of automating problem modeling and solving. However, they remain unreliable and often depend on iterative repair loops that result in significant latency. We introduce OptiHive, an LLM-based framework that produces high-quality solvers for optimization problems from natural-language descriptions without iterative self-correction. OptiHive uses a single batched LLM query to generate diverse components (solvers, problem instances, and validation tests) and filters out erroneous components to ensure fully interpretable outputs. Taking into account the imperfection of the generated components, we employ a statistical model to infer their true performance, enabling principled uncertainty quantification and solver selection. On tasks ranging from traditional optimization problems to challenging variants of the Multi-Depot Vehicle Routing Problem, OptiHive significantly outperforms baselines, increasing the optimality rate from 5\% to 92\% on the most complex problems. 

---
# PHM-Bench: A Domain-Specific Benchmarking Framework for Systematic Evaluation of Large Models in Prognostics and Health Management 

**Authors**: Puyu Yang, Laifa Tao, Zijian Huang, Haifei Liu, Wenyan Cao, Hao Ji, Jianan Qiu, Qixuan Huang, Xuanyuan Su, Yuhang Xie, Jun Zhang, Shangyu Li, Chen Lu, Zhixuan Lian  

**Link**: [PDF](https://arxiv.org/pdf/2508.02490)  

**Abstract**: With the rapid advancement of generative artificial intelligence, large language models (LLMs) are increasingly adopted in industrial domains, offering new opportunities for Prognostics and Health Management (PHM). These models help address challenges such as high development costs, long deployment cycles, and limited generalizability. However, despite the growing synergy between PHM and LLMs, existing evaluation methodologies often fall short in structural completeness, dimensional comprehensiveness, and evaluation granularity. This hampers the in-depth integration of LLMs into the PHM domain. To address these limitations, this study proposes PHM-Bench, a novel three-dimensional evaluation framework for PHM-oriented large models. Grounded in the triadic structure of fundamental capability, core task, and entire lifecycle, PHM-Bench is tailored to the unique demands of PHM system engineering. It defines multi-level evaluation metrics spanning knowledge comprehension, algorithmic generation, and task optimization. These metrics align with typical PHM tasks, including condition monitoring, fault diagnosis, RUL prediction, and maintenance decision-making. Utilizing both curated case sets and publicly available industrial datasets, our study enables multi-dimensional evaluation of general-purpose and domain-specific models across diverse PHM tasks. PHM-Bench establishes a methodological foundation for large-scale assessment of LLMs in PHM and offers a critical benchmark to guide the transition from general-purpose to PHM-specialized models. 

---
# Multimodal Large Language Models for End-to-End Affective Computing: Benchmarking and Boosting with Generative Knowledge Prompting 

**Authors**: Miaosen Luo, Jiesen Long, Zequn Li, Yunying Yang, Yuncheng Jiang, Sijie Mai  

**Link**: [PDF](https://arxiv.org/pdf/2508.02429)  

**Abstract**: Multimodal Affective Computing (MAC) aims to recognize and interpret human emotions by integrating information from diverse modalities such as text, video, and audio. Recent advancements in Multimodal Large Language Models (MLLMs) have significantly reshaped the landscape of MAC by offering a unified framework for processing and aligning cross-modal information. However, practical challenges remain, including performance variability across complex MAC tasks and insufficient understanding of how architectural designs and data characteristics impact affective analysis. To address these gaps, we conduct a systematic benchmark evaluation of state-of-the-art open-source MLLMs capable of concurrently processing audio, visual, and textual modalities across multiple established MAC datasets. Our evaluation not only compares the performance of these MLLMs but also provides actionable insights into model optimization by analyzing the influence of model architectures and dataset properties. Furthermore, we propose a novel hybrid strategy that combines generative knowledge prompting with supervised fine-tuning to enhance MLLMs' affective computing capabilities. Experimental results demonstrate that this integrated approach significantly improves performance across various MAC tasks, offering a promising avenue for future research and development in this field. Our code is released on this https URL. 

---
# CABENCH: Benchmarking Composable AI for Solving Complex Tasks through Composing Ready-to-Use Models 

**Authors**: Tung-Thuy Pham, Duy-Quan Luong, Minh-Quan Duong, Trung-Hieu Nguyen, Thu-Trang Nguyen, Son Nguyen, Hieu Dinh Vo  

**Link**: [PDF](https://arxiv.org/pdf/2508.02427)  

**Abstract**: Composable AI offers a scalable and effective paradigm for tackling complex AI tasks by decomposing them into sub-tasks and solving each sub-task using ready-to-use well-trained models. However, systematically evaluating methods under this setting remains largely unexplored. In this paper, we introduce CABENCH, the first public benchmark comprising 70 realistic composable AI tasks, along with a curated pool of 700 models across multiple modalities and domains. We also propose an evaluation framework to enable end-to-end assessment of composable AI solutions. To establish initial baselines, we provide human-designed reference solutions and compare their performance with two LLM-based approaches. Our results illustrate the promise of composable AI in addressing complex real-world problems while highlighting the need for methods that can fully unlock its potential by automatically generating effective execution pipelines. 

---
# Traffic-R1: Reinforced LLMs Bring Human-Like Reasoning to Traffic Signal Control Systems 

**Authors**: Xingchen Zou, Yuhao Yang, Zheng Chen, Xixuan Hao, Yiqi Chen, Chao Huang, Yuxuan Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02344)  

**Abstract**: Traffic signal control (TSC) is vital for mitigating congestion and sustaining urban mobility. In this paper, we introduce Traffic-R1, a foundation model with human-like reasoning for TSC systems. Our model is developed through self-exploration and iteration of reinforced large language models (LLMs) with expert guidance in a simulated traffic environment. Compared to traditional reinforcement learning (RL) and recent LLM-based methods, Traffic-R1 offers three significant advantages. First, Traffic-R1 delivers zero-shot generalisation, transferring unchanged to new road networks and out-of-distribution incidents by utilizing its internal traffic control policies and human-like reasoning. Second, its 3B-parameter architecture is lightweight enough for real-time inference on mobile-class chips, enabling large-scale edge deployment. Third, Traffic-R1 provides an explainable TSC process and facilitates multi-intersection communication through its self-iteration and a new synchronous communication network. Extensive benchmarks demonstrate that Traffic-R1 sets a new state of the art, outperforming strong baselines and training-intensive RL controllers. In practice, the model now manages signals for more than 55,000 drivers daily, shortening average queues by over 5% and halving operator workload. Our checkpoint is available at this https URL. 

---
# FinWorld: An All-in-One Open-Source Platform for End-to-End Financial AI Research and Deployment 

**Authors**: Wentao Zhang, Yilei Zhao, Chuqiao Zong, Xinrun Wang, Bo An  

**Link**: [PDF](https://arxiv.org/pdf/2508.02292)  

**Abstract**: Financial AI holds great promise for transforming modern finance, with the potential to support a wide range of tasks such as market forecasting, portfolio management, quantitative trading, and automated analysis. However, existing platforms remain limited in task coverage, lack robust multimodal data integration, and offer insufficient support for the training and deployment of large language models (LLMs). In response to these limitations, we present FinWorld, an all-in-one open-source platform that provides end-to-end support for the entire financial AI workflow, from data acquisition to experimentation and deployment. FinWorld distinguishes itself through native integration of heterogeneous financial data, unified support for diverse AI paradigms, and advanced agent automation, enabling seamless development and deployment. Leveraging data from 2 representative markets, 4 stock pools, and over 800 million financial data points, we conduct comprehensive experiments on 4 key financial AI tasks. These experiments systematically evaluate deep learning and reinforcement learning algorithms, with particular emphasis on RL-based finetuning for LLMs and LLM Agents. The empirical results demonstrate that FinWorld significantly enhances reproducibility, supports transparent benchmarking, and streamlines deployment, thereby providing a strong foundation for future research and real-world applications. Code is available at Github~\footnote{this https URL}. 

---
# AirTrafficGen: Configurable Air Traffic Scenario Generation with Large Language Models 

**Authors**: Dewi Sid William Gould, George De Ath, Ben Carvell, Nick Pepper  

**Link**: [PDF](https://arxiv.org/pdf/2508.02269)  

**Abstract**: The manual design of scenarios for Air Traffic Control (ATC) training is a demanding and time-consuming bottleneck that limits the diversity of simulations available to controllers. To address this, we introduce a novel, end-to-end approach, AirTrafficGen, that leverages large language models (LLMs) to automate and control the generation of complex ATC scenarios. Our method uses a purpose-built, graph-based representation to encode sector topology (including airspace geometry, routes, and fixes) into a format LLMs can process. Through rigorous benchmarking, we show that state-of-the-art models like Gemini 2.5 Pro and OpenAI o3 can generate high-traffic scenarios whilst maintaining operational realism. Our engineered prompting enables fine-grained control over interaction presence, type, and location. Initial findings suggest these models are also capable of iterative refinement, correcting flawed scenarios based on simple textual feedback. This approach provides a scalable alternative to manual scenario design, addressing the need for a greater volume and variety of ATC training and validation simulations. More broadly, this work showcases the potential of LLMs for complex planning in safety-critical domains. 

---
# A Message Passing Realization of Expected Free Energy Minimization 

**Authors**: Wouter W. L. Nuijten, Mykola Lukashchuk, Thijs van de Laar, Bert de Vries  

**Link**: [PDF](https://arxiv.org/pdf/2508.02197)  

**Abstract**: We present a message passing approach to Expected Free Energy (EFE) minimization on factor graphs, based on the theory introduced in arXiv:2504.14898. By reformulating EFE minimization as Variational Free Energy minimization with epistemic priors, we transform a combinatorial search problem into a tractable inference problem solvable through standard variational techniques. Applying our message passing method to factorized state-space models enables efficient policy inference. We evaluate our method on environments with epistemic uncertainty: a stochastic gridworld and a partially observable Minigrid task. Agents using our approach consistently outperform conventional KL-control agents on these tasks, showing more robust planning and efficient exploration under uncertainty. In the stochastic gridworld environment, EFE-minimizing agents avoid risky paths, while in the partially observable minigrid setting, they conduct more systematic information-seeking. This approach bridges active inference theory with practical implementations, providing empirical evidence for the efficiency of epistemic priors in artificial agents. 

---
# Neuromorphic Computing with Multi-Frequency Oscillations: A Bio-Inspired Approach to Artificial Intelligence 

**Authors**: Boheng Liu, Ziyu Li, Xia Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02191)  

**Abstract**: Despite remarkable capabilities, artificial neural networks exhibit limited flexible, generalizable intelligence. This limitation stems from their fundamental divergence from biological cognition that overlooks both neural regions' functional specialization and the temporal dynamics critical for coordinating these specialized systems. We propose a tripartite brain-inspired architecture comprising functionally specialized perceptual, auxiliary, and executive systems. Moreover, the integration of temporal dynamics through the simulation of multi-frequency neural oscillation and synaptic dynamic adaptation mechanisms enhances the architecture, thereby enabling more flexible and efficient artificial cognition. Initial evaluations demonstrate superior performance compared to state-of-the-art temporal processing approaches, with 2.18\% accuracy improvements while reducing required computation iterations by 48.44\%, and achieving higher correlation with human confidence patterns. Though currently demonstrated on visual processing tasks, this architecture establishes a theoretical foundation for brain-like intelligence across cognitive domains, potentially bridging the gap between artificial and biological intelligence. 

---
# Reconsidering Overthinking: Penalizing Internal and External Redundancy in CoT Reasoning 

**Authors**: Jialiang Hong, Taihang Zhen, Kai Chen, Jiaheng Liu, Wenpeng Zhu, Jing Huo, Yang Gao, Depeng Wang, Haitao Wan, Xi Yang, Boyan Wang, Fanyu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2508.02178)  

**Abstract**: Large Reasoning Models (LRMs) often produce excessively verbose reasoning traces, a phenomenon known as overthinking, which hampers both efficiency and interpretability. Prior works primarily address this issue by reducing response length, without fully examining the underlying semantic structure of the reasoning process. In this paper, we revisit overthinking by decomposing it into two distinct forms: internal redundancy, which consists of low-contribution reasoning steps within the first correct solution (FCS), and external redundancy, which refers to unnecessary continuation after the FCS. To mitigate both forms, we propose a dual-penalty reinforcement learning framework. For internal redundancy, we adopt a sliding-window semantic analysis to penalize low-gain reasoning steps that contribute little toward reaching the correct answer. For external redundancy, we penalize its proportion beyond the FCS to encourage earlier termination. Our method significantly compresses reasoning traces with minimal accuracy loss, and generalizes effectively to out-of-domain tasks such as question answering and code generation. Crucially, we find that external redundancy can be safely removed without degrading performance, whereas internal redundancy must be reduced more cautiously to avoid impairing correctness. These findings suggest that our method not only improves reasoning efficiency but also enables implicit, semantic-aware control over Chain-of-Thought length, paving the way for more concise and interpretable LRMs. 

---
# Beyond the Trade-off: Self-Supervised Reinforcement Learning for Reasoning Models' Instruction Following 

**Authors**: Qingyu Ren, Qianyu He, Bowei Zhang, Jie Zeng, Jiaqing Liang, Yanghua Xiao, Weikang Zhou, Zeye Sun, Fei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02150)  

**Abstract**: Reasoning models excel in complex problem solving but exhibit a concerning trade off between reasoning capabilities and instruction following abilities. Existing approaches for improving instruction following rely on stronger external models, creating methodological bottlenecks and practical limitations including increased costs and accessibility constraints. We propose a self-supervised RL framework that leverages reasoning models' own internal signals to improve instruction following capabilities without external supervision. Extensive experiments demonstrate that our framework significantly improves instruction following capabilities while maintaining reasoning performance, offering a scalable and cost-effective approach to enhance instruction following in reasoning models. The data and code are publicly available at this https URL. 

---
# All Stories Are One Story: Emotional Arc Guided Procedural Game Level Generation 

**Authors**: Yunge Wen, Chenliang Huang, Hangyu Zhou, Zhuo Zeng, Chun Ming Louis Po, Julian Togelius, Timothy Merino, Sam Earle  

**Link**: [PDF](https://arxiv.org/pdf/2508.02132)  

**Abstract**: The emotional arc is a universal narrative structure underlying stories across cultures and media -- an idea central to structuralist narratology, often encapsulated in the phrase "all stories are one story." We present a framework for procedural game narrative generation that incorporates emotional arcs as a structural backbone for both story progression and gameplay dynamics. Leveraging established narratological theories and large-scale empirical analyses, we focus on two core emotional patterns -- Rise and Fall -- to guide the generation of branching story graphs. Each story node is automatically populated with characters, items, and gameplay-relevant attributes (e.g., health, attack), with difficulty adjusted according to the emotional trajectory. Implemented in a prototype action role-playing game (ARPG), our system demonstrates how emotional arcs can be operationalized using large language models (LLMs) and adaptive entity generation. Evaluation through player ratings, interviews, and sentiment analysis shows that emotional arc integration significantly enhances engagement, narrative coherence, and emotional impact. These results highlight the potential of emotionally structured procedural generation for advancing interactive storytelling for games. 

---
# Trainable Dynamic Mask Sparse Attention 

**Authors**: Jingze Shi, Yifan Wu, Bingheng Wu, Yiran Peng, Liangdong Wang, Guang Liu, Yuyu Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.02124)  

**Abstract**: In large language models, the demand for modeling long contexts is constantly increasing, but the quadratic complexity of the standard self-attention mechanism often becomes a bottleneck. Although existing sparse attention mechanisms have improved efficiency, they may still encounter issues such as static patterns or information loss. We introduce a trainable dynamic mask sparse attention mechanism, Dynamic Mask Attention, which effectively utilizes content-aware and position-aware sparsity. DMA achieves this through two key innovations: First, it dynamically generates content-aware sparse masks from value representations, enabling the model to identify and focus on critical information adaptively. Second, it implements position-aware sparse attention computation that effectively skips unnecessary calculation regions. This dual-sparsity design allows the model to significantly reduce the computational complexity of important information while retaining complete information, achieving an excellent balance between information fidelity and computational efficiency. We have verified the performance of DMA through comprehensive experiments. Comparative studies show that DMA outperforms multi-head attention, sliding window attention, multi-head latent attention, and native sparse attention in terms of perplexity under Chinchilla Scaling Law settings. Moreover, in challenging multi-query associative recall tasks, DMA also demonstrates superior performance and efficiency compared to these methods. Crucially, in the evaluation of a 1.7B parameter model, DMA significantly outperforms multi-head attention in both standard benchmark performance and the challenging needle-in-a-haystack task. These experimental results highlight its capability to balance model efficiency and long-context modeling ability effectively. 

---
# A Survey on AgentOps: Categorization, Challenges, and Future Directions 

**Authors**: Zexin Wang, Jingjing Li, Quan Zhou, Haotian Si, Yuanhao Liu, Jianhui Li, Gaogang Xie, Fei Sun, Dan Pei, Changhua Pei  

**Link**: [PDF](https://arxiv.org/pdf/2508.02121)  

**Abstract**: As the reasoning capabilities of Large Language Models (LLMs) continue to advance, LLM-based agent systems offer advantages in flexibility and interpretability over traditional systems, garnering increasing attention. However, despite the widespread research interest and industrial application of agent systems, these systems, like their traditional counterparts, frequently encounter anomalies. These anomalies lead to instability and insecurity, hindering their further development. Therefore, a comprehensive and systematic approach to the operation and maintenance of agent systems is urgently needed. Unfortunately, current research on the operations of agent systems is sparse. To address this gap, we have undertaken a survey on agent system operations with the aim of establishing a clear framework for the field, defining the challenges, and facilitating further development. Specifically, this paper begins by systematically defining anomalies within agent systems, categorizing them into intra-agent anomalies and inter-agent anomalies. Next, we introduce a novel and comprehensive operational framework for agent systems, dubbed Agent System Operations (AgentOps). We provide detailed definitions and explanations of its four key stages: monitoring, anomaly detection, root cause analysis, and resolution. 

---
# Don't Overthink It: A Survey of Efficient R1-style Large Reasoning Models 

**Authors**: Linan Yue, Yichao Du, Yizhi Wang, Weibo Gao, Fangzhou Yao, Li Wang, Ye Liu, Ziyu Xu, Qi Liu, Shimin Di, Min-Ling Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02120)  

**Abstract**: Recently, Large Reasoning Models (LRMs) have gradually become a research hotspot due to their outstanding performance in handling complex tasks. Among them, DeepSeek R1 has garnered significant attention for its exceptional performance and open-source nature, driving advancements in the research of R1-style LRMs. Unlike traditional Large Language Models (LLMs), these models enhance logical deduction and decision-making capabilities during reasoning by incorporating mechanisms such as long chain-of-thought and self-reflection through reinforcement learning. However, with the widespread application of these models, the problem of overthinking has gradually emerged. Specifically, when generating answers, these models often construct excessively long reasoning chains with redundant or repetitive steps, which leads to reduced reasoning efficiency and may affect the accuracy of the final answer. To this end, various efficient reasoning methods have been proposed, aiming to reduce the length of reasoning paths without compromising model performance and reasoning capability. By reviewing the current research advancements in the field of efficient reasoning methods systematically, we categorize existing works into two main directions based on the lens of single-model optimization versus model collaboration: (1) Efficient Reasoning with Single Model, which focuses on improving the reasoning efficiency of individual models; and (2) Efficient Reasoning with Model Collaboration, which explores optimizing reasoning paths through collaboration among multiple models. Besides, we maintain a public GitHub repository that tracks the latest progress in efficient reasoning methods. 

---
# Attractive Metadata Attack: Inducing LLM Agents to Invoke Malicious Tools 

**Authors**: Kanghua Mo, Li Hu, Yucheng Long, Zhihao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.02110)  

**Abstract**: Large language model (LLM) agents have demonstrated remarkable capabilities in complex reasoning and decision-making by leveraging external tools. However, this tool-centric paradigm introduces a previously underexplored attack surface: adversaries can manipulate tool metadata -- such as names, descriptions, and parameter schemas -- to influence agent behavior. We identify this as a new and stealthy threat surface that allows malicious tools to be preferentially selected by LLM agents, without requiring prompt injection or access to model internals. To demonstrate and exploit this vulnerability, we propose the Attractive Metadata Attack (AMA), a black-box in-context learning framework that generates highly attractive but syntactically and semantically valid tool metadata through iterative optimization. Our attack integrates seamlessly into standard tool ecosystems and requires no modification to the agent's execution framework. Extensive experiments across ten realistic, simulated tool-use scenarios and a range of popular LLM agents demonstrate consistently high attack success rates (81\%-95\%) and significant privacy leakage, with negligible impact on primary task execution. Moreover, the attack remains effective even under prompt-level defenses and structured tool-selection protocols such as the Model Context Protocol, revealing systemic vulnerabilities in current agent architectures. These findings reveal that metadata manipulation constitutes a potent and stealthy attack surface, highlighting the need for execution-level security mechanisms that go beyond prompt-level defenses. 

---
# "Stack It Up!": 3D Stable Structure Generation from 2D Hand-drawn Sketch 

**Authors**: Yiqing Xu, Linfeng Li, Cunjun Yu, David Hsu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02093)  

**Abstract**: Imagine a child sketching the Eiffel Tower and asking a robot to bring it to life. Today's robot manipulation systems can't act on such sketches directly-they require precise 3D block poses as goals, which in turn demand structural analysis and expert tools like CAD. We present StackItUp, a system that enables non-experts to specify complex 3D structures using only 2D front-view hand-drawn sketches. StackItUp introduces an abstract relation graph to bridge the gap between rough sketches and accurate 3D block arrangements, capturing the symbolic geometric relations (e.g., left-of) and stability patterns (e.g., two-pillar-bridge) while discarding noisy metric details from sketches. It then grounds this graph to 3D poses using compositional diffusion models and iteratively updates it by predicting hidden internal and rear supports-critical for stability but absent from the sketch. Evaluated on sketches of iconic landmarks and modern house designs, StackItUp consistently produces stable, multilevel 3D structures and outperforms all baselines in both stability and visual resemblance. 

---
# SE-Agent: Self-Evolution Trajectory Optimization in Multi-Step Reasoning with LLM-Based Agents 

**Authors**: Jiaye Lin, Yifu Guo, Yuzhen Han, Sen Hu, Ziyi Ni, Licheng Wang, Mingguang Chen, Daxin Jiang, Binxing Jiao, Chen Hu, Huacan Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02085)  

**Abstract**: Large Language Model (LLM)-based agents have recently shown impressive capabilities in complex reasoning and tool use via multi-step interactions with their environments. While these agents have the potential to tackle complicated tasks, their problem-solving process, i.e., agents' interaction trajectory leading to task completion, remains underexploited. These trajectories contain rich feedback that can navigate agents toward the right directions for solving problems correctly. Although prevailing approaches, such as Monte Carlo Tree Search (MCTS), can effectively balance exploration and exploitation, they ignore the interdependence among various trajectories and lack the diversity of search spaces, which leads to redundant reasoning and suboptimal outcomes. To address these challenges, we propose SE-Agent, a Self-Evolution framework that enables Agents to optimize their reasoning processes iteratively. Our approach revisits and enhances former pilot trajectories through three key operations: revision, recombination, and refinement. This evolutionary mechanism enables two critical advantages: (1) it expands the search space beyond local optima by intelligently exploring diverse solution paths guided by previous trajectories, and (2) it leverages cross-trajectory inspiration to efficiently enhance performance while mitigating the impact of suboptimal reasoning paths. Through these mechanisms, SE-Agent achieves continuous self-evolution that incrementally improves reasoning quality. We evaluate SE-Agent on SWE-bench Verified to resolve real-world GitHub issues. Experimental results across five strong LLMs show that integrating SE-Agent delivers up to 55% relative improvement, achieving state-of-the-art performance among all open-source agents on SWE-bench Verified. Our code and demonstration materials are publicly available at this https URL. 

---
# Everyone Contributes! Incentivizing Strategic Cooperation in Multi-LLM Systems via Sequential Public Goods Games 

**Authors**: Yunhao Liang, Yuan Qu, Jingyuan Yang, Shaochong Lin, Zuo-Jun Max Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02076)  

**Abstract**: Coordinating multiple large language models (LLMs) to solve complex tasks collaboratively poses a fundamental trade-off between the computation costs and collective performance compared with individual model. We introduce a novel, game-theoretically grounded reinforcement learning (RL) framework, the Multi-Agent Cooperation Sequential Public Goods Game (MAC-SPGG), to systematically incentivize cooperation in multi-LLM ensembles. In MAC-SPGG, LLM agents move in sequence, observing predecessors' outputs and updating beliefs to condition their own contributions. By redesigning the public-goods reward, effortful contributions become the unique Subgame Perfect Nash Equilibrium (SPNE), which eliminates free-riding under traditional SPGG or PGG. Its sequential protocol replaces costly round-based information exchanges with a streamlined decision flow, cutting communication overhead while retaining strategic depth. We prove the existence and uniqueness of the SPNE under realistic parameters, and empirically show that MAC-SPGG-trained ensembles outperform single-agent baselines, chain-of-thought prompting, and other cooperative methods, even achieving comparable performance to large-scale models across reasoning, math, code generation, and NLP tasks. Our results highlight the power of structured, incentive-aligned MAC-SPGG cooperation for scalable and robust multi-agent language generation. 

---
# Risk identification based on similar case retrieval enhancement, 

**Authors**: Jiawei Li, Chengye Yang, Yaochen Zhang, Weilin Sun, Lei Meng, Xiangxu Meng  

**Link**: [PDF](https://arxiv.org/pdf/2508.02073)  

**Abstract**: The goal of construction site risk and hazard identification is to enhance safety management through automation. Existing research based on large language models falls into two categories: image-text matching for collaborative reasoning, which struggles with complex hazard features, and instruction fine-tuning or dialogue guidance using professional datasets, which suffers from high training costs and poor this http URL address this, we propose a hazard identification method using similar case retrieval enhancement. By integrating external knowledge and retrieved case contexts via prompt fine-tuning, we mitigate misjudgments caused by limited domain knowledge and weak feature associations. Our method includes three modules: retrieval library, image similarity retrieval, and large model retrieval enhancement, enabling efficient recognition without training. Experiments on real construction data show significant improvements. For instance, GLM-4V's recognition accuracy increased to 50\%, a 35.49\% boost. The method enhances accuracy, context understanding, and stability, offering new theoretical and technical support for hazard detection. 

---
# TRACEALIGN -- Tracing the Drift: Attributing Alignment Failures to Training-Time Belief Sources in LLMs 

**Authors**: Amitava Das, Vinija Jain, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2508.02063)  

**Abstract**: Large Language Models (LLMs) fine-tuned to align with human values often exhibit alignment drift, producing unsafe or policy-violating completions when exposed to adversarial prompts, decoding perturbations, or paraphrased jailbreaks. While prior work has behaviorally characterized alignment failure, little is known about the training-time belief sources underlying these failures. We introduce TraceAlign, a unified framework for tracing unsafe completions back to their root causes in the model's training corpus. Central to our approach is the Belief Conflict Index (BCI), which quantifies semantic inconsistency between generated spans and aligned policies, based on retrieved training documents using suffix-array matching. We propose three complementary interventions: (i) TraceShield, an inference-time safety filter that refuses completions with high-BCI spans, (ii) Contrastive Belief Deconfliction Loss, a contrastive fine-tuning objective penalizing high-BCI continuations during DPO, and (iii) Prov-Decode, a provenance-aware decoding strategy that vetoes beam expansions predicted to yield high-BCI spans. Together, these defenses reduce alignment drift by up to 85% on our curated Alignment Drift Benchmark (ADB) while preserving utility on standard tasks, with delta less than 0.2 and improved refusal quality. We further derive a theoretical upper bound on drift likelihood via suffix-array span statistics, linking memorization frequency and length to adversarial reactivation risk. TraceAlign thus provides the first scalable, traceable, and grounded toolkit for understanding and mitigating alignment failures at source. To encourage further exploration and development, we open-source our implementation at: this https URL 

---
# Dynamic Context Adaptation for Consistent Role-Playing Agents with Retrieval-Augmented Generations 

**Authors**: Jeiyoon Park, Yongshin Han, Minseop Kim, Kisu Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02016)  

**Abstract**: We propose AMADEUS, which is composed of Adaptive Context-aware Text Splitter (ACTS), Guided Selection (GS), and Attribute Extractor (AE). ACTS finds an optimal chunk length and hierarchical contexts for each character. AE identifies a character's general attributes from the chunks retrieved by GS and uses these attributes as a final context to maintain robust persona consistency even when answering out of knowledge questions. To facilitate the development and evaluation of RAG-based RPAs, we construct CharacterRAG, a role-playing dataset that consists of persona documents for 15 distinct fictional characters totaling 976K written characters, and 450 question and answer pairs. We find that our framework effectively models not only the knowledge possessed by characters, but also various attributes such as personality. 

---
# Agent-Based Feature Generation from Clinical Notes for Outcome Prediction 

**Authors**: Jiayi Wang, Jacqueline Jil Vallon, Neil Panjwani, Xi Ling, Sushmita Vij, Sandy Srinivas, John Leppert, Mark K. Buyyounouski, Mohsen Bayati  

**Link**: [PDF](https://arxiv.org/pdf/2508.01956)  

**Abstract**: Electronic health records (EHRs) contain rich unstructured clinical notes that could enhance predictive modeling, yet extracting meaningful features from these notes remains challenging. Current approaches range from labor-intensive manual clinician feature generation (CFG) to fully automated representational feature generation (RFG) that lack interpretability and clinical relevance. Here we introduce SNOW (Scalable Note-to-Outcome Workflow), a modular multi-agent system powered by large language models (LLMs) that autonomously generates structured clinical features from unstructured notes without human intervention. We evaluated SNOW against manual CFG, clinician-guided LLM approaches, and RFG methods for predicting 5-year prostate cancer recurrence in 147 patients from Stanford Healthcare. While manual CFG achieved the highest performance (AUC-ROC: 0.771), SNOW matched this performance (0.761) without requiring any clinical expertise, significantly outperforming both baseline features alone (0.691) and all RFG approaches. The clinician-guided LLM method also performed well (0.732) but still required expert input. SNOW's specialized agents handle feature discovery, extraction, validation, post-processing, and aggregation, creating interpretable features that capture complex clinical information typically accessible only through manual review. Our findings demonstrate that autonomous LLM systems can replicate expert-level feature engineering at scale, potentially transforming how clinical ML models leverage unstructured EHR data while maintaining the interpretability essential for clinical deployment. 

---
# Multi-turn Natural Language to Graph Query Language Translation 

**Authors**: Yuanyuan Liang, Lei Pan, Tingyu Xie, Yunshi Lan, Weining Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.01871)  

**Abstract**: In recent years, research on transforming natural language into graph query language (NL2GQL) has been increasing. Most existing methods focus on single-turn transformation from NL to GQL. In practical applications, user interactions with graph databases are typically multi-turn, dynamic, and context-dependent. While single-turn methods can handle straightforward queries, more complex scenarios often require users to iteratively adjust their queries, investigate the connections between entities, or request additional details across multiple dialogue turns. Research focused on single-turn conversion fails to effectively address multi-turn dialogues and complex context dependencies. Additionally, the scarcity of high-quality multi-turn NL2GQL datasets further hinders the progress of this field. To address this challenge, we propose an automated method for constructing multi-turn NL2GQL datasets based on Large Language Models (LLMs) , and apply this method to develop the MTGQL dataset, which is constructed from a financial market graph database and will be publicly released for future research. Moreover, we propose three types of baseline methods to assess the effectiveness of multi-turn NL2GQL translation, thereby laying a solid foundation for future research. 

---
# ProKG-Dial: Progressive Multi-Turn Dialogue Construction with Domain Knowledge Graphs 

**Authors**: Yuanyuan Liang, Xiaoman Wang, Tingyu Xie, Lei Pan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01869)  

**Abstract**: Current large language models (LLMs) excel at general NLP tasks but often lack domain specific precision in professional settings. Building a high quality domain specific multi turn dialogue dataset is essential for developing specialized conversational systems. However, existing methods such as manual annotation, simulated human LLM interactions, and role based LLM dialogues are resource intensive or suffer from limitations in dialogue quality and domain coverage. To address these challenges, we introduce ProKG Dial, a progressive framework for constructing knowledge intensive multi turn dialogue datasets using domain specific knowledge graphs (KGs). ProKG Dial leverages the structured nature of KGs to encode complex domain knowledge and relationships, providing a solid foundation for generating meaningful and coherent dialogues. Specifically, ProKG Dial begins by applying community detection to partition the KG into semantically cohesive subgraphs. For each subgraph, the framework incrementally generates a series of questions and answers centered around a target entity, ensuring relevance and coverage. A rigorous filtering step is employed to maintain high dialogue quality. We validate ProKG Dial on a medical knowledge graph by evaluating the generated dialogues in terms of diversity, semantic coherence, and entity coverage. Furthermore, we fine tune a base LLM on the resulting dataset and benchmark it against several baselines. Both automatic metrics and human evaluations demonstrate that ProKG Dial substantially improves dialogue quality and domain specific performance, highlighting its effectiveness and practical utility. 

---
# CloudAnoAgent: Anomaly Detection for Cloud Sites via LLM Agent with Neuro-Symbolic Mechanism 

**Authors**: Xinkai Zou, Xuan Jiang, Ruikai Huang, Haoze He, Parv Kapoor, Jiahua Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01844)  

**Abstract**: Anomaly detection in cloud sites remains a critical yet challenging task. Existing approaches that rely solely on metric data often suffer from high false positive rates (FPR) due to data imbalance between normal and anomalous events, leading to significant operational overhead for system reliance engineers. Recent advances in large language models (LLMs) offer new opportunities for integrating metrics with log data, enabling more accurate and interpretable anomaly detection. In this paper, we propose CloudAnoAgent, the first neuro-symbolic LLM-based agent for anomaly detection in cloud environments. CloudAnoAgent jointly processes structured metrics and textual log data in a unified pipeline, leveraging symbolic verification to validate detection hypotheses and generate structured anomaly reports. To support systematic evaluation, we introduce CloudAnoBench, the first benchmark that provides LLM-generated paired metrics and log data with fine-grained anomaly behavior annotations, filling a critical gap in existing datasets. Experimental results demonstrate that CloudAnoAgent improves anomaly classification accuracy by 46.36% and 36.67% on average and reduces the FPR by 36.67% and 33.89% on average over traditional baselines and LLM-only baseline, with a boost on anomaly type detection accuracy by 12.8% compared to vanilla LLM prompting. These results demonstrate the strengths of our approach in improving detection accuracy, reducing false positives, and enhancing interpretability, thereby supporting practical deployment in enterprise cloud environments. 

---
# LiveMCPBench: Can Agents Navigate an Ocean of MCP Tools? 

**Authors**: Guozhao Mo, Wenliang Zhong, Jiawei Chen, Xuanang Chen, Yaojie Lu, Hongyu Lin, Ben He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.01780)  

**Abstract**: With the rapid development of Model Context Protocol (MCP), the number of MCP servers has surpassed 10,000. However, existing MCP benchmarks are limited to single-server settings with only a few tools, hindering effective evaluation of agent capabilities in large-scale, real-world scenarios. To address this limitation, we present LiveMCPBench, the first comprehensive benchmark comprising 95 real-world tasks grounded in the MCP ecosystem, designed to evaluate LLM agents at scale across diverse servers. To support a scalable and reproducible evaluation pipeline in large-scale MCP environments, we curate LiveMCPTool, a diverse and readily deployable collection of 70 MCP servers and 527 tools. Furthermore, we introduce LiveMCPEval, an LLM-as-a-Judge framework that enables automated and adaptive evaluation in dynamic, time-varying task environments, achieving 81% agreement with human reviewers. Finally, we propose the MCP Copilot Agent, a multi-step agent that routes tools for dynamic planning and executes tools for API interaction across the entire LiveMCPTool suite. Our evaluation covers 10 leading models, with the best-performing model (Claude-Sonnet-4) reaching a 78.95% success rate. However, we observe large performance variance across models, and several widely-used models perform poorly in LiveMCPBench's complex, tool-rich environments. Overall, LiveMCPBench offers the first unified framework for benchmarking LLM agents in realistic, tool-rich, and dynamic MCP environments, laying a solid foundation for scalable and reproducible research on agent capabilities. Our code and data will be publicly available at this https URL. 

---
# Uncertainty-Based Methods for Automated Process Reward Data Construction and Output Aggregation in Mathematical Reasoning 

**Authors**: Jiuzhou Han, Wray Buntine, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01773)  

**Abstract**: Large language models have demonstrated remarkable capabilities in complex mathematical reasoning tasks, but they inevitably generate errors throughout multi-step solutions. Process-level Reward Models (PRMs) have shown great promise by providing supervision and evaluation at each intermediate step, thereby effectively improving the models' reasoning abilities. However, training effective PRMs requires high-quality process reward data, yet existing methods for constructing such data are often labour-intensive or inefficient. In this paper, we propose an uncertainty-driven framework for automated process reward data construction, encompassing both data generation and annotation processes for PRMs. Additionally, we identify the limitations of both majority vote and PRMs, and introduce two generic uncertainty-aware output aggregation methods: Hybrid Majority Reward Vote and Weighted Reward Frequency Vote, which combine the strengths of majority vote with PRMs. Extensive experiments on ProcessBench, MATH, and GSMPlus show the effectiveness and efficiency of the proposed PRM data construction framework, and demonstrate that the two output aggregation methods further improve the mathematical reasoning abilities across diverse PRMs. The code and data will be publicly available at this https URL. 

---
# Reasoning Systems as Structured Processes: Foundations, Failures, and Formal Criteria 

**Authors**: Saleh Nikooroo, Thomas Engel  

**Link**: [PDF](https://arxiv.org/pdf/2508.01763)  

**Abstract**: This paper outlines a general formal framework for reasoning systems, intended to support future analysis of inference architectures across domains. We model reasoning systems as structured tuples comprising phenomena, explanation space, inference and generation maps, and a principle base. The formulation accommodates logical, algorithmic, and learning-based reasoning processes within a unified structural schema, while remaining agnostic to any specific reasoning algorithm or logic system. We survey basic internal criteria--including coherence, soundness, and completeness-and catalog typical failure modes such as contradiction, incompleteness, and non-convergence. The framework also admits dynamic behaviors like iterative refinement and principle evolution. The goal of this work is to establish a foundational structure for representing and comparing reasoning systems, particularly in contexts where internal failure, adaptation, or fragmentation may arise. No specific solution architecture is proposed; instead, we aim to support future theoretical and practical investigations into reasoning under structural constraint. 

---
# Implementing Cumulative Functions with Generalized Cumulative Constraints 

**Authors**: Pierre Schaus, Charles Thomas, Roger Kameugne  

**Link**: [PDF](https://arxiv.org/pdf/2508.01751)  

**Abstract**: Modeling scheduling problems with conditional time intervals and cumulative functions has become a common approach when using modern commercial constraint programming solvers. This paradigm enables the modeling of a wide range of scheduling problems, including those involving producers and consumers. However, it is unavailable in existing open-source solvers and practical implementation details remain undocumented. In this work, we present an implementation of this modeling approach using a single, generic global constraint called the Generalized Cumulative. We also introduce a novel time-table filtering algorithm designed to handle tasks defined on conditional time-intervals. Experimental results demonstrate that this approach, combined with the new filtering algorithm, performs competitively with existing solvers enabling the modeling of producer and consumer scheduling problems and effectively scales to large problems. 

---
# Bayes-Entropy Collaborative Driven Agents for Research Hypotheses Generation and Optimization 

**Authors**: Shiyang Duan, Yuan Tian, Qi Bing, Xiaowei Shao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01746)  

**Abstract**: The exponential growth of scientific knowledge has made the automated generation of scientific hypotheses that combine novelty, feasibility, and research value a core challenge. Existing methods based on large language models fail to systematically model the inherent in hypotheses or incorporate the closed-loop feedback mechanisms crucial for refinement. This paper proposes a multi-agent collaborative framework called HypoAgents, which for the first time integrates Bayesian reasoning with an information entropy-driven search mechanism across three stages-hypotheses generation, evidence validation, and hypotheses Refinement-to construct an iterative closed-loop simulating scientists' cognitive processes. Specifically, the framework first generates an initial set of hypotheses through diversity sampling and establishes prior beliefs based on a composite novelty-relevance-feasibility (N-R-F) score. It then employs etrieval-augmented generation (RAG) to gather external literature evidence, updating the posterior probabilities of hypotheses using Bayes' theorem. Finally, it identifies high-uncertainty hypotheses using information entropy $H = - \sum {{p_i}\log {p_i}}$ and actively refines them, guiding the iterative optimization of the hypothesis set toward higher quality and confidence. Experimental results on the ICLR 2025 conference real-world research question dataset (100 research questions) show that after 12 optimization iterations, the average ELO score of generated hypotheses improves by 116.3, surpassing the benchmark of real paper abstracts by 17.8, while the framework's overall uncertainty, as measured by Shannon entropy, decreases significantly by 0.92. This study presents an interpretable probabilistic reasoning framework for automated scientific discovery, substantially improving the quality and reliability of machine-generated research hypotheses. 

---
# ReflecSched: Solving Dynamic Flexible Job-Shop Scheduling via LLM-Powered Hierarchical Reflection 

**Authors**: Shijie Cao, Yuan Yuan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01724)  

**Abstract**: Dynamic Flexible Job-Shop Scheduling (DFJSP) is an NP-hard problem challenged by real-time event adaptation and complex machine routing. While traditional dispatching rules are efficient but rigid, deep learning approaches are opaque and require intricate feature engineering. Large Language Models (LLMs) promise adaptive reasoning without this engineering overhead, yet we find their direct application is suboptimal. Baseline LLMs suffer from three key pitfalls: the long-context paradox, where crucial data is underutilized; an underutilization of expert heuristics; and myopic decision-making. To address this, we propose ReflecSched, a framework that empowers the LLM beyond a direct scheduler by equipping it with a strategic analysis capability. ReflecSched tasks the LLM to analyze heuristic-driven simulations across multiple planning horizons and distill them into a concise, natural-language summary termed ``Strategic Experience''. This summary is then integrated into the prompt of a final decision-making module, guiding it to produce non-myopic actions. Experiments show that ReflecSched not only statistically significantly outperforms direct LLM baselines, securing a 71.35\% Win Rate and a 2.755\% Relative Percentage Deviation reduction, but also surpasses the performance of all individual heuristics evaluated, all while demonstrably mitigating the three identified pitfalls. Additionally, ReflecSched performs on par with the best heuristic tailored to each instance across all problem cases. 

---
# DeepVIS: Bridging Natural Language and Data Visualization Through Step-wise Reasoning 

**Authors**: Zhihao Shuai, Boyan Li, Siyu Yan, Yuyu Luo, Weikai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01700)  

**Abstract**: Although data visualization is powerful for revealing patterns and communicating insights, creating effective visualizations requires familiarity with authoring tools and often disrupts the analysis flow. While large language models show promise for automatically converting analysis intent into visualizations, existing methods function as black boxes without transparent reasoning processes, which prevents users from understanding design rationales and refining suboptimal outputs. To bridge this gap, we propose integrating Chain-of-Thought (CoT) reasoning into the Natural Language to Visualization (NL2VIS) pipeline. First, we design a comprehensive CoT reasoning process for NL2VIS and develop an automatic pipeline to equip existing datasets with structured reasoning steps. Second, we introduce nvBench-CoT, a specialized dataset capturing detailed step-by-step reasoning from ambiguous natural language descriptions to finalized visualizations, which enables state-of-the-art performance when used for model fine-tuning. Third, we develop DeepVIS, an interactive visual interface that tightly integrates with the CoT reasoning process, allowing users to inspect reasoning steps, identify errors, and make targeted adjustments to improve visualization outcomes. Quantitative benchmark evaluations, two use cases, and a user study collectively demonstrate that our CoT framework effectively enhances NL2VIS quality while providing insightful reasoning steps to users. 

---
# SURE-Med: Systematic Uncertainty Reduction for Enhanced Reliability in Medical Report Generation 

**Authors**: Yuhang Gu, Xingyu Hu, Yuyu Fan, Xulin Yan, Longhuan Xu, Peng peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01693)  

**Abstract**: Automated medical report generation (MRG) holds great promise for reducing the heavy workload of radiologists. However, its clinical deployment is hindered by three major sources of uncertainty. First, visual uncertainty, caused by noisy or incorrect view annotations, compromises feature extraction. Second, label distribution uncertainty, stemming from long-tailed disease prevalence, biases models against rare but clinically critical conditions. Third, contextual uncertainty, introduced by unverified historical reports, often leads to factual hallucinations. These challenges collectively limit the reliability and clinical trustworthiness of MRG systems. To address these issues, we propose SURE-Med, a unified framework that systematically reduces uncertainty across three critical dimensions: visual, distributional, and contextual. To mitigate visual uncertainty, a Frontal-Aware View Repair Resampling module corrects view annotation errors and adaptively selects informative features from supplementary views. To tackle label distribution uncertainty, we introduce a Token Sensitive Learning objective that enhances the modeling of critical diagnostic sentences while reweighting underrepresented diagnostic terms, thereby improving sensitivity to infrequent conditions. To reduce contextual uncertainty, our Contextual Evidence Filter validates and selectively incorporates prior information that aligns with the current image, effectively suppressing hallucinations. Extensive experiments on the MIMIC-CXR and IU-Xray benchmarks demonstrate that SURE-Med achieves state-of-the-art performance. By holistically reducing uncertainty across multiple input modalities, SURE-Med sets a new benchmark for reliability in medical report generation and offers a robust step toward trustworthy clinical decision support. 

---
# T-GRAG: A Dynamic GraphRAG Framework for Resolving Temporal Conflicts and Redundancy in Knowledge Retrieval 

**Authors**: Dong Li, Yichen Niu, Ying Ai, Xiang Zou, Biqing Qi, Jianxing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01680)  

**Abstract**: Large language models (LLMs) have demonstrated strong performance in natural language generation but remain limited in knowle-
dge-intensive tasks due to outdated or incomplete internal knowledge. Retrieval-Augmented Generation (RAG) addresses this by incorporating external retrieval, with GraphRAG further enhancing performance through structured knowledge graphs and multi-hop reasoning. However, existing GraphRAG methods largely ignore the temporal dynamics of knowledge, leading to issues such as temporal ambiguity, time-insensitive retrieval, and semantic redundancy. To overcome these limitations, we propose Temporal GraphRAG (T-GRAG), a dynamic, temporally-aware RAG framework that models the evolution of knowledge over time. T-GRAG consists of five key components: (1) a Temporal Knowledge Graph Generator that creates time-stamped, evolving graph structures; (2) a Temporal Query Decomposition mechanism that breaks complex temporal queries into manageable sub-queries; (3) a Three-layer Interactive Retriever that progressively filters and refines retrieval across temporal subgraphs; (4) a Source Text Extractor to mitigate noise; and (5) a LLM-based Generator that synthesizes contextually and temporally accurate responses. We also introduce Time-LongQA, a novel benchmark dataset based on real-world corporate annual reports, designed to test temporal reasoning across evolving knowledge. Extensive experiments show that T-GRAG significantly outperforms prior RAG and GraphRAG baselines in both retrieval accuracy and response relevance under temporal constraints, highlighting the necessity of modeling knowledge evolution for robust long-text question answering. Our code is publicly available on the T-GRAG 

---
# QCBench: Evaluating Large Language Models on Domain-Specific Quantitative Chemistry 

**Authors**: Jiaqing Xie, Weida Wang, Ben Gao, Zhuo Yang, Haiyuan Wan, Shufei Zhang, Tianfan Fu, Yuqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01670)  

**Abstract**: Quantitative chemistry plays a fundamental role in chemistry research, enabling precise predictions of molecular properties, reaction outcomes, and material behaviors. While large language models (LLMs) have shown promise in chemistry-related tasks, their ability to perform rigorous, step-by-step quantitative reasoning remains underexplored. To fill this blank, we propose QCBench, a Quantitative Chemistry benchmark comprising 350 computational chemistry problems across 7 chemistry subfields (analytical chemistry, bio/organic chemistry, general chemistry, inorganic chemistry, physical chemistry, polymer chemistry and quantum chemistry), categorized into three hierarchical tiers-basic, intermediate, and expert-to systematically evaluate the mathematical reasoning abilities of large language models (LLMs). Designed to minimize shortcuts and emphasize stepwise numerical reasoning, each problem focuses on pure calculations rooted in real-world chemical vertical fields. QCBench enables fine-grained diagnosis of computational weaknesses, reveals model-specific limitations across difficulty levels, and lays the groundwork for future improvements such as domain adaptive fine-tuning or multi-modal integration. Evaluations on 19 LLMs demonstrate a consistent performance degradation with increasing task complexity, highlighting the current gap between language fluency and scientific computation accuracy. 

---
# DRKF: Decoupled Representations with Knowledge Fusion for Multimodal Emotion Recognition 

**Authors**: Peiyuan Jiang, Yao Liu, Qiao Liu, Zongshun Zhang, Jiaye Yang, Lu Liu, Daibing Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01644)  

**Abstract**: Multimodal emotion recognition (MER) aims to identify emotional states by integrating and analyzing information from multiple modalities. However, inherent modality heterogeneity and inconsistencies in emotional cues remain key challenges that hinder performance. To address these issues, we propose a Decoupled Representations with Knowledge Fusion (DRKF) method for MER. DRKF consists of two main modules: an Optimized Representation Learning (ORL) Module and a Knowledge Fusion (KF) Module. ORL employs a contrastive mutual information estimation method with progressive modality augmentation to decouple task-relevant shared representations and modality-specific features while mitigating modality heterogeneity. KF includes a lightweight self-attention-based Fusion Encoder (FE) that identifies the dominant modality and integrates emotional information from other modalities to enhance the fused representation. To handle potential errors from incorrect dominant modality selection under emotionally inconsistent conditions, we introduce an Emotion Discrimination Submodule (ED), which enforces the fused representation to retain discriminative cues of emotional inconsistency. This ensures that even if the FE selects an inappropriate dominant modality, the Emotion Classification Submodule (EC) can still make accurate predictions by leveraging preserved inconsistency information. Experiments show that DRKF achieves state-of-the-art (SOTA) performance on IEMOCAP, MELD, and M3ED. The source code is publicly available at this https URL. 

---
# A Multi-Agent Pokemon Tournament for Evaluating Strategic Reasoning of Large Language Models 

**Authors**: Tadisetty Sai Yashwanth, Dhatri C  

**Link**: [PDF](https://arxiv.org/pdf/2508.01623)  

**Abstract**: This research presents LLM Pokemon League, a competitive tournament system that leverages Large Language Models (LLMs) as intelligent agents to simulate strategic decision-making in Pokémon battles. The platform is designed to analyze and compare the reasoning, adaptability, and tactical depth exhibited by different LLMs in a type-based, turn-based combat environment. By structuring the competition as a single-elimination tournament involving diverse AI trainers, the system captures detailed decision logs, including team-building rationale, action selection strategies, and switching decisions. The project enables rich exploration into comparative AI behavior, battle psychology, and meta-strategy development in constrained, rule-based game environments. Through this system, we investigate how modern LLMs understand, adapt, and optimize decisions under uncertainty, making Pokémon League a novel benchmark for AI research in strategic reasoning and competitive learning. 

---
# Polymorphic Combinatorial Frameworks (PCF): Guiding the Design of Mathematically-Grounded, Adaptive AI Agents 

**Authors**: David Pearl, Matthew Murphy, James Intriligator  

**Link**: [PDF](https://arxiv.org/pdf/2508.01581)  

**Abstract**: The Polymorphic Combinatorial Framework (PCF) leverages Large Language Models (LLMs) and mathematical frameworks to guide the meta-prompt enabled design of solution spaces and adaptive AI agents for complex, dynamic environments. Unlike static agent architectures, PCF enables real-time parameter reconfiguration through mathematically-grounded combinatorial spaces, allowing agents to adapt their core behavioral traits dynamically. Grounded in combinatorial logic, topos theory, and rough fuzzy set theory, PCF defines a multidimensional SPARK parameter space (Skills, Personalities, Approaches, Resources, Knowledge) to capture agent behaviors. This paper demonstrates how LLMs can parameterize complex spaces and estimate likely parameter values/variabilities. Using PCF, we parameterized mock café domains (five levels of complexity), estimated variables/variabilities, and conducted over 1.25 million Monte Carlo simulations. The results revealed trends in agent adaptability and performance across the five complexity tiers, with diminishing returns at higher complexity levels highlighting thresholds for scalable designs. PCF enables the generation of optimized agent configurations for specific scenarios while maintaining logical consistency. This framework supports scalable, dynamic, explainable, and ethical AI applications in domains like customer service, healthcare, robotics, and collaborative systems, paving the way for adaptable and cooperative next-generation polymorphic agents. 

---
# One Subgoal at a Time: Zero-Shot Generalization to Arbitrary Linear Temporal Logic Requirements in Multi-Task Reinforcement Learning 

**Authors**: Zijian Guo, İlker Işık, H. M. Sabbir Ahmad, Wenchao Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01561)  

**Abstract**: Generalizing to complex and temporally extended task objectives and safety constraints remains a critical challenge in reinforcement learning (RL). Linear temporal logic (LTL) offers a unified formalism to specify such requirements, yet existing methods are limited in their abilities to handle nested long-horizon tasks and safety constraints, and cannot identify situations when a subgoal is not satisfiable and an alternative should be sought. In this paper, we introduce GenZ-LTL, a method that enables zero-shot generalization to arbitrary LTL specifications. GenZ-LTL leverages the structure of Büchi automata to decompose an LTL task specification into sequences of reach-avoid subgoals. Contrary to the current state-of-the-art method that conditions on subgoal sequences, we show that it is more effective to achieve zero-shot generalization by solving these reach-avoid problems \textit{one subgoal at a time} through proper safe RL formulations. In addition, we introduce a novel subgoal-induced observation reduction technique that can mitigate the exponential complexity of subgoal-state combinations under realistic assumptions. Empirical results show that GenZ-LTL substantially outperforms existing methods in zero-shot generalization to unseen LTL specifications. 

---
# Empowering Tabular Data Preparation with Language Models: Why and How? 

**Authors**: Mengshi Chen, Yuxiang Sun, Tengchao Li, Jianwei Wang, Kai Wang, Xuemin Lin, Ying Zhang, Wenjie Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01556)  

**Abstract**: Data preparation is a critical step in enhancing the usability of tabular data and thus boosts downstream data-driven tasks. Traditional methods often face challenges in capturing the intricate relationships within tables and adapting to the tasks involved. Recent advances in Language Models (LMs), especially in Large Language Models (LLMs), offer new opportunities to automate and support tabular data preparation. However, why LMs suit tabular data preparation (i.e., how their capabilities match task demands) and how to use them effectively across phases still remain to be systematically explored. In this survey, we systematically analyze the role of LMs in enhancing tabular data preparation processes, focusing on four core phases: data acquisition, integration, cleaning, and transformation. For each phase, we present an integrated analysis of how LMs can be combined with other components for different preparation tasks, highlight key advancements, and outline prospective pipelines. 

---
# Getting out of the Big-Muddy: Escalation of Commitment in LLMs 

**Authors**: Emilio Barkett, Olivia Long, Paul Kröger  

**Link**: [PDF](https://arxiv.org/pdf/2508.01545)  

**Abstract**: Large Language Models (LLMs) are increasingly deployed in autonomous decision-making roles across high-stakes domains. However, since models are trained on human-generated data, they may inherit cognitive biases that systematically distort human judgment, including escalation of commitment, where decision-makers continue investing in failing courses of action due to prior investment. Understanding when LLMs exhibit such biases presents a unique challenge. While these biases are well-documented in humans, it remains unclear whether they manifest consistently in LLMs or require specific triggering conditions. This paper investigates this question using a two-stage investment task across four experimental conditions: model as investor, model as advisor, multi-agent deliberation, and compound pressure scenario. Across N = 6,500 trials, we find that bias manifestation in LLMs is highly context-dependent. In individual decision-making contexts (Studies 1-2, N = 4,000), LLMs demonstrate strong rational cost-benefit logic with minimal escalation of commitment. However, multi-agent deliberation reveals a striking hierarchy effect (Study 3, N = 500): while asymmetrical hierarchies show moderate escalation rates (46.2%), symmetrical peer-based decision-making produces near-universal escalation (99.2%). Similarly, when subjected to compound organizational and personal pressures (Study 4, N = 2,000), models exhibit high degrees of escalation of commitment (68.95% average allocation to failing divisions). These findings reveal that LLM bias manifestation depends critically on social and organizational context rather than being inherent, with significant implications for the deployment of multi-agent systems and unsupervised operations where such conditions may emerge naturally. 

---
# Refine-n-Judge: Curating High-Quality Preference Chains for LLM-Fine-Tuning 

**Authors**: Derin Cayir, Renjie Tao, Rashi Rungta, Kai Sun, Sean Chen, Haidar Khan, Minseok Kim, Julia Reinspach, Yue Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01543)  

**Abstract**: Large Language Models (LLMs) have demonstrated remarkable progress through preference-based fine-tuning, which critically depends on the quality of the underlying training data. While human feedback is essential for improving data quality, it is costly and does not scale well. In this paper, we introduce Refine-n-Judge, an automated iterative approach that leverages a single LLM as both a refiner and a judge to enhance dataset quality. Unlike existing iterative refinement methods, Refine-n-Judge employs an LLM to both generate refinements and explicitly evaluate each improvement, ensuring that every iteration meaningfully enhances the dataset without requiring additional human annotation or a separate reward model. At each step, the LLM refines a response and judges whether the refinement is an improvement over the previous answer. This process continues until the LLM prefers the initial answer over the refinement, indicating no further improvements. This produces sequences of increasing quality, preference-labeled responses ideal for fine-tuning.
We demonstrate the effectiveness of Refine-n-Judge across a range of public datasets spanning five corpora, targeting tasks such as coding, math, and conversation. Models (Llama 3.1-8B and Llama 3.3-70B) fine-tuned on Refine-n-Judge-enhanced datasets were preferred by LLM judges in over 74% of comparisons against models tuned on the original dataset by GPT-4. Additionally, we report performance gains: +5% on AlpacaEval and AlpacaEval 2.0, and +19% on MT-Bench. Our results indicate that Refine-n-Judge produces high-quality datasets and scalable model improvements. 

---
# WinkTPG: An Execution Framework for Multi-Agent Path Finding Using Temporal Reasoning 

**Authors**: Jingtian Yan, Stephen F. Smith, Jiaoyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01495)  

**Abstract**: Planning collision-free paths for a large group of agents is a challenging problem with numerous real-world applications. While recent advances in Multi-Agent Path Finding (MAPF) have shown promising progress, standard MAPF algorithms rely on simplified kinodynamic models, preventing agents from directly following the generated MAPF plan. To bridge this gap, we propose kinodynamic Temporal Plan Graph Planning (kTPG), a multi-agent speed optimization algorithm that efficiently refines a MAPF plan into a kinodynamically feasible plan while accounting for uncertainties and preserving collision-freeness. Building on kTPG, we propose Windowed kTPG (WinkTPG), a MAPF execution framework that incrementally refines MAPF plans using a window-based mechanism, dynamically incorporating agent information during execution to reduce uncertainty. Experiments show that WinkTPG can generate speed profiles for up to 1,000 agents in 1 second and improves solution quality by up to 51.7% over existing MAPF execution methods. 

---
# CARGO: A Co-Optimization Framework for EV Charging and Routing in Goods Delivery Logistics 

**Authors**: Arindam Khanda, Anurag Satpathy, Amit Jha, Sajal K. Das  

**Link**: [PDF](https://arxiv.org/pdf/2508.01476)  

**Abstract**: With growing interest in sustainable logistics, electric vehicle (EV)-based deliveries offer a promising alternative for urban distribution. However, EVs face challenges due to their limited battery capacity, requiring careful planning for recharging. This depends on factors such as the charging point (CP) availability, cost, proximity, and vehicles' state of charge (SoC). We propose CARGO, a framework addressing the EV-based delivery route planning problem (EDRP), which jointly optimizes route planning and charging for deliveries within time windows. After proving the problem's NP-hardness, we propose a mixed integer linear programming (MILP)-based exact solution and a computationally efficient heuristic method. Using real-world datasets, we evaluate our methods by comparing the heuristic to the MILP solution, and benchmarking it against baseline strategies, Earliest Deadline First (EDF) and Nearest Delivery First (NDF). The results show up to 39% and 22% reductions in the charging cost over EDF and NDF, respectively, while completing comparable deliveries. 

---
# $R^2$-CoD: Understanding Text-Graph Complementarity in Relational Reasoning via Knowledge Co-Distillation 

**Authors**: Zhen Wu, Ritam Dutt, Luke M. Breitfeller, Armineh Nourbakhsh, Siddharth Parekh, Carolyn Rosé  

**Link**: [PDF](https://arxiv.org/pdf/2508.01475)  

**Abstract**: Relational reasoning lies at the core of many NLP tasks, drawing on complementary signals from text and graphs. While prior research has investigated how to leverage this dual complementarity, a detailed and systematic understanding of text-graph interplay and its effect on hybrid models remains underexplored. We take an analysis-driven approach to investigate text-graph representation complementarity via a unified architecture that supports knowledge co-distillation (CoD). We explore five tasks involving relational reasoning that differ in how text and graph structures encode the information needed to solve that task. By tracking how these dual representations evolve during training, we uncover interpretable patterns of alignment and divergence, and provide insights into when and why their integration is beneficial. 

---
# TripTailor: A Real-World Benchmark for Personalized Travel Planning 

**Authors**: Yuanzhe Shen, Kaimin Wang, Changze Lv, Xiaoqing Zheng, Xuanjing Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01432)  

**Abstract**: The continuous evolution and enhanced reasoning capabilities of large language models (LLMs) have elevated their role in complex tasks, notably in travel planning, where demand for personalized, high-quality itineraries is rising. However, current benchmarks often rely on unrealistic simulated data, failing to reflect the differences between LLM-generated and real-world itineraries. Existing evaluation metrics, which primarily emphasize constraints, fall short of providing a comprehensive assessment of the overall quality of travel plans. To address these limitations, we introduce TripTailor, a benchmark designed specifically for personalized travel planning in real-world scenarios. This dataset features an extensive collection of over 500,000 real-world points of interest (POIs) and nearly 4,000 diverse travel itineraries, complete with detailed information, providing a more authentic evaluation framework. Experiments show that fewer than 10\% of the itineraries generated by the latest state-of-the-art LLMs achieve human-level performance. Moreover, we identify several critical challenges in travel planning, including the feasibility, rationality, and personalized customization of the proposed solutions. We hope that TripTailor will drive the development of travel planning agents capable of understanding and meeting user needs while generating practical itineraries. Our code and dataset are available at this https URL 

---
# Relation-Aware LNN-Transformer for Intersection-Centric Next-Step Prediction 

**Authors**: Zhehong Ren, Tianluo Zhang, Yiheng Lu, Yushen Liang, Promethee Spathis  

**Link**: [PDF](https://arxiv.org/pdf/2508.01368)  

**Abstract**: Next-step location prediction plays a pivotal role in modeling human mobility, underpinning applications from personalized navigation to strategic urban planning. However, approaches that assume a closed world - restricting choices to a predefined set of points of interest (POIs) - often fail to capture exploratory or target-agnostic behavior and the topological constraints of urban road networks. Hence, we introduce a road-node-centric framework that represents road-user trajectories on the city's road-intersection graph, thereby relaxing the closed-world constraint and supporting next-step forecasting beyond fixed POI sets. To encode environmental context, we introduce a sector-wise directional POI aggregation that produces compact features capturing distance, bearing, density and presence cues. By combining these cues with structural graph embeddings, we obtain semantically grounded node representations. For sequence modeling, we integrate a Relation-Aware LNN-Transformer - a hybrid of a Continuous-time Forgetting Cell CfC-LNN and a bearing-biased self-attention module - to capture both fine-grained temporal dynamics and long-range spatial dependencies. Evaluated on city-scale road-user trajectories, our model outperforms six state-of-the-art baselines by up to 17 percentage points in accuracy at one hop and 10 percentage points in MRR, and maintains high resilience under noise, losing only 2.4 percentage points in accuracy at one under 50 meter GPS perturbation and 8.9 percentage points in accuracy at one hop under 25 percent POI noise. 

---
# NatureGAIA: Pushing the Frontiers of GUI Agents with a Challenging Benchmark and High-Quality Trajectory Dataset 

**Authors**: Zihan Zheng, Tianle Cui, Chuwen Xie, Jiahui Zhang, Jiahui Pan, Lewei He, Qianglong Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01330)  

**Abstract**: The rapid advancement of Large Language Model (LLM)-driven Graphical User Interface (GUI) agents is significantly hampered by the profound limitations of existing evaluation benchmarks in terms of accuracy, reproducibility, and scalability. To address this critical gap, we introduce \Benchmark, a novel benchmark engineered on the principle of Causal Pathways. This design paradigm structures complex tasks into a series of programmatically verifiable atomic steps, ensuring a rigorous, fully automated, and reproducible standard for assessment. Concurrently, to mitigate the inherent capability deficits of agents, we developed \Agent, a hierarchical agent architecture specifically optimized for long-horizon tasks. We leveraged this agent to generate a high-quality, human-verified trajectory dataset that uniquely captures diverse and even self-correcting interaction patterns of LLMs. We then utilized this dataset to perform Reinforcement Fine-Tuning (RFT) on the Qwen2.5-VL-7B model. Our experiments reveal that \Benchmark~presents a formidable challenge to current state-of-the-art LLMs; even the top-performing Claude-sonnet-4 achieved a Weighted Pathway Success Rate (WPSR) of only 34.6\%. Moreover, while RFT substantially improved the smaller model's GUI execution capabilities (WPSR increased from 3.3\% to 10.8\%), its performance degraded sharply when handling complex scenarios. This outcome highlights the inherent capability ceiling of smaller models when faced with comprehensive tasks that integrate perception, decision-making, and execution. This research contributes a rigorous evaluation standard and a high-quality dataset to the community, aiming to guide the future development of GUI agents. 

---
# Towards Evaluation for Real-World LLM Unlearning 

**Authors**: Ke Miao, Yuke Hu, Xiaochen Li, Wenjie Bao, Zhihao Liu, Zhan Qin, Kui Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.01324)  

**Abstract**: This paper analyzes the limitations of existing unlearning evaluation metrics in terms of practicality, exactness, and robustness in real-world LLM unlearning scenarios. To overcome these limitations, we propose a new metric called Distribution Correction-based Unlearning Evaluation (DCUE). It identifies core tokens and corrects distributional biases in their confidence scores using a validation set. The evaluation results are quantified using the Kolmogorov-Smirnov test. Experimental results demonstrate that DCUE overcomes the limitations of existing metrics, which also guides the design of more practical and reliable unlearning algorithms in the future. 

---
# Idempotent Equilibrium Analysis of Hybrid Workflow Allocation: A Mathematical Schema for Future Work 

**Authors**: Faruk Alpay, Bugra Kilictas, Taylan Alpay, Hamdi Alakkad  

**Link**: [PDF](https://arxiv.org/pdf/2508.01323)  

**Abstract**: The rapid advance of large-scale AI systems is reshaping how work is divided between people and machines. We formalise this reallocation as an iterated task-delegation map and show that--under broad, empirically grounded assumptions--the process converges to a stable idempotent equilibrium in which every task is performed by the agent (human or machine) with enduring comparative advantage. Leveraging lattice-theoretic fixed-point tools (Tarski and Banach), we (i) prove existence of at least one such equilibrium and (ii) derive mild monotonicity conditions that guarantee uniqueness. In a stylised continuous model the long-run automated share takes the closed form $x^* = \alpha / (\alpha + \beta)$, where $\alpha$ captures the pace of automation and $\beta$ the rate at which new, human-centric tasks appear; hence full automation is precluded whenever $\beta > 0$. We embed this analytic result in three complementary dynamical benchmarks--a discrete linear update, an evolutionary replicator dynamic, and a continuous Beta-distributed task spectrum--each of which converges to the same mixed equilibrium and is reproducible from the provided code-free formulas. A 2025-to-2045 simulation calibrated to current adoption rates projects automation rising from approximately 10% of work to approximately 65%, leaving a persistent one-third of tasks to humans. We interpret that residual as a new profession of workflow conductor: humans specialise in assigning, supervising and integrating AI modules rather than competing with them. Finally, we discuss implications for skill development, benchmark design and AI governance, arguing that policies which promote "centaur" human-AI teaming can steer the economy toward the welfare-maximising fixed point. 

---
# PUZZLED: Jailbreaking LLMs through Word-Based Puzzles 

**Authors**: Yelim Ahn, Jaejin Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.01306)  

**Abstract**: As large language models (LLMs) are increasingly deployed across diverse domains, ensuring their safety has become a critical concern. In response, studies on jailbreak attacks have been actively growing. Existing approaches typically rely on iterative prompt engineering or semantic transformations of harmful instructions to evade detection. In this work, we introduce PUZZLED, a novel jailbreak method that leverages the LLM's reasoning capabilities. It masks keywords in a harmful instruction and presents them as word puzzles for the LLM to solve. We design three puzzle types-word search, anagram, and crossword-that are familiar to humans but cognitively demanding for LLMs. The model must solve the puzzle to uncover the masked words and then proceed to generate responses to the reconstructed harmful instruction. We evaluate PUZZLED on five state-of-the-art LLMs and observe a high average attack success rate (ASR) of 88.8%, specifically 96.5% on GPT-4.1 and 92.3% on Claude 3.7 Sonnet. PUZZLED is a simple yet powerful attack that transforms familiar puzzles into an effective jailbreak strategy by harnessing LLMs' reasoning capabilities. 

---
# How Far Are LLMs from Symbolic Planners? An NLP-Based Perspective 

**Authors**: Ma'ayan Armony, Albert Meroño-Peñuela, Gerard Canal  

**Link**: [PDF](https://arxiv.org/pdf/2508.01300)  

**Abstract**: The reasoning and planning abilities of Large Language Models (LLMs) have been a frequent topic of discussion in recent years. Their ability to take unstructured planning problems as input has made LLMs' integration into AI planning an area of interest. Nevertheless, LLMs are still not reliable as planners, with the generated plans often containing mistaken or hallucinated actions. Existing benchmarking and evaluation methods investigate planning with LLMs, focusing primarily on success rate as a quality indicator in various planning tasks, such as validating plans or planning in relaxed conditions. In this paper, we approach planning with LLMs as a natural language processing (NLP) task, given that LLMs are NLP models themselves. We propose a recovery pipeline consisting of an NLP-based evaluation of the generated plans, along with three stages to recover the plans through NLP manipulation of the LLM-generated plans, and eventually complete the plan using a symbolic planner. This pipeline provides a holistic analysis of LLM capabilities in the context of AI task planning, enabling a broader understanding of the quality of invalid plans. Our findings reveal no clear evidence of underlying reasoning during plan generation, and that a pipeline comprising an NLP-based analysis of the plans, followed by a recovery mechanism, still falls short of the quality and reliability of classical planners. On average, only the first 2.65 actions of the plan are executable, with the average length of symbolically generated plans being 8.4 actions. The pipeline still improves action quality and increases the overall success rate from 21.9% to 27.5%. 

---
# BioDisco: Multi-agent hypothesis generation with dual-mode evidence, iterative feedback and temporal evaluation 

**Authors**: Yujing Ke, Kevin George, Kathan Pandya, David Blumenthal, Maximilian Sprang, Gerrit Großmann, Sebastian Vollmer, David Antony Selby  

**Link**: [PDF](https://arxiv.org/pdf/2508.01285)  

**Abstract**: Identifying novel hypotheses is essential to scientific research, yet this process risks being overwhelmed by the sheer volume and complexity of available information. Existing automated methods often struggle to generate novel and evidence-grounded hypotheses, lack robust iterative refinement and rarely undergo rigorous temporal evaluation for future discovery potential. To address this, we propose BioDisco, a multi-agent framework that draws upon language model-based reasoning and a dual-mode evidence system (biomedical knowledge graphs and automated literature retrieval) for grounded novelty, integrates an internal scoring and feedback loop for iterative refinement, and validates performance through pioneering temporal and human evaluations and a Bradley-Terry paired comparison model to provide statistically-grounded assessment. Our evaluations demonstrate superior novelty and significance over ablated configurations representative of existing agentic architectures. Designed for flexibility and modularity, BioDisco allows seamless integration of custom language models or knowledge graphs, and can be run with just a few lines of code. We anticipate researchers using this practical tool as a catalyst for the discovery of new hypotheses. 

---
# Multi-TW: Benchmarking Multimodal Models on Traditional Chinese Question Answering in Taiwan 

**Authors**: Jui-Ming Yao, Bing-Cheng Xie, Sheng-Wei Peng, Hao-Yuan Chen, He-Rong Zheng, Bing-Jia Tan, Peter Shaojui Wang, Shun-Feng Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.01274)  

**Abstract**: Multimodal Large Language Models (MLLMs) process visual, acoustic, and textual inputs, addressing the limitations of single-modality LLMs. However, existing benchmarks often overlook tri-modal evaluation in Traditional Chinese and do not consider inference latency. To address this, we introduce Multi-TW, the first Traditional Chinese benchmark for evaluating the performance and latency of any-to-any multimodal models. Multi-TW includes 900 multiple-choice questions (image and text, audio and text pairs) sourced from official proficiency tests developed with the Steering Committee for the Test of Proficiency-Huayu (SC-TOP). We evaluated various any-to-any models and vision-language models (VLMs) with audio transcription. Our results show that closed-source models generally outperform open-source ones across modalities, although open-source models can perform well in audio tasks. End-to-end any-to-any pipelines offer clear latency advantages compared to VLMs using separate audio transcription. Multi-TW presents a comprehensive view of model capabilities and highlights the need for Traditional Chinese fine-tuning and efficient multimodal architectures. 

---
# KCR: Resolving Long-Context Knowledge Conflicts via Reasoning in LLMs 

**Authors**: Xianda Zheng, Zijian Huang, Meng-Fen Chiang, Michael J. Witbrock, Kaiqi Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01273)  

**Abstract**: Knowledge conflicts commonly arise across diverse sources, and their prevalence has increased with the advent of LLMs. When dealing with conflicts between multiple contexts, also known as \emph{inter-context knowledge conflicts}, LLMs are often confused by lengthy and conflicting contexts. To address this challenge, we propose the Knowledge Conflict Reasoning (KCR) framework, which enhances the ability of LLMs to resolve conflicting knowledge. The key idea of KCR is to train backbone LLMs to establish a correct reasoning process by rewarding them for selecting and adhering to the context with stronger logical consistency when presented with conflicting contexts. Specifically, we first extract reasoning paths, represented by either text or local knowledge graphs, from the conflicting long contexts. Subsequently, we employ Reinforcement Learning to encourage the model to learn the paradigm of reasoning process that follows correct reasoning paths rather than the incorrect counterparts. This enables the backbone models to genuinely acquire the capability to resolve inter-context knowledge conflicts within long contexts. Experimental results demonstrate that our framework significantly improves the ability of various backbone models to resolve knowledge conflicts in long-context scenarios, yielding substantial performance gains. 

---
# Win-k: Improved Membership Inference Attacks on Small Language Models 

**Authors**: Roya Arkhmammadova, Hosein Madadi Tamar, M. Emre Gursoy  

**Link**: [PDF](https://arxiv.org/pdf/2508.01268)  

**Abstract**: Small language models (SLMs) are increasingly valued for their efficiency and deployability in resource-constrained environments, making them useful for on-device, privacy-sensitive, and edge computing applications. On the other hand, membership inference attacks (MIAs), which aim to determine whether a given sample was used in a model's training, are an important threat with serious privacy and intellectual property implications. In this paper, we study MIAs on SLMs. Although MIAs were shown to be effective on large language models (LLMs), they are relatively less studied on emerging SLMs, and furthermore, their effectiveness decreases as models get smaller. Motivated by this finding, we propose a new MIA called win-k, which builds on top of a state-of-the-art attack (min-k). We experimentally evaluate win-k by comparing it with five existing MIAs using three datasets and eight SLMs. Results show that win-k outperforms existing MIAs in terms of AUROC, TPR @ 1% FPR, and FPR @ 99% TPR metrics, especially on smaller models. 

---
# Unifying Mixture of Experts and Multi-Head Latent Attention for Efficient Language Models 

**Authors**: Sushant Mehta, Raj Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2508.01261)  

**Abstract**: We present MoE-MLA-RoPE, a novel architecture combination that combines Mixture of Experts (MoE) with Multi-head Latent Attention (MLA) and Rotary Position Embeddings (RoPE) for efficient language modeling. Our approach addresses the fundamental trade-off between model capacity and computational efficiency through three key innovations: (1) fine-grained expert routing with 64 micro-experts and top-$k$ selection, enabling flexible specialization through 3.6 * 10^7 possible expert combinations; (2) shared expert isolation that dedicates 2 always active experts for common patterns while routing to 6 of 62 specialized experts; and (3) gradient-conflict-free load balancing that maintains expert utilization without interfering with primary loss optimization.
Extensive experiments on models ranging from 17M to 202M parameters demonstrate that MoE-MLA-RoPE with compression ratio r=d/2 achieves 68% KV cache memory reduction and 3.2x inference speedup while maintaining competitive perplexity (0.8% degradation). Compared to the parameters with 53.9M parameters, MoE-MLA-RoPE improves the validation loss by 6.9% over the vanilla transformers while using 42% fewer active parameters per forward pass. FLOP-matched experiments reveal even larger gains: 11.1% improvement with 3.2x inference acceleration. Automated evaluation using GPT-4 as a judge confirms quality improvements in generation, with higher scores on coherence (8.1/10), creativity (7.9/10) and grammatical correctness (8.2/10). Our results establish that architectural novelty, not parameter scaling, defines the efficiency frontier for resource-constrained language model deployment. 

---
# SketchAgent: Generating Structured Diagrams from Hand-Drawn Sketches 

**Authors**: Cheng Tan, Qi Chen, Jingxuan Wei, Gaowei Wu, Zhangyang Gao, Siyuan Li, Bihui Yu, Ruifeng Guo, Stan Z. Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01237)  

**Abstract**: Hand-drawn sketches are a natural and efficient medium for capturing and conveying ideas. Despite significant advancements in controllable natural image generation, translating freehand sketches into structured, machine-readable diagrams remains a labor-intensive and predominantly manual task. The primary challenge stems from the inherent ambiguity of sketches, which lack the structural constraints and semantic precision required for automated diagram generation. To address this challenge, we introduce SketchAgent, a multi-agent system designed to automate the transformation of hand-drawn sketches into structured diagrams. SketchAgent integrates sketch recognition, symbolic reasoning, and iterative validation to produce semantically coherent and structurally accurate diagrams, significantly reducing the need for manual effort. To evaluate the effectiveness of our approach, we propose the Sketch2Diagram Benchmark, a comprehensive dataset and evaluation framework encompassing eight diverse diagram categories, such as flowcharts, directed graphs, and model architectures. The dataset comprises over 6,000 high-quality examples with token-level annotations, standardized preprocessing, and rigorous quality control. By streamlining the diagram generation process, SketchAgent holds great promise for applications in design, education, and engineering, while offering a significant step toward bridging the gap between intuitive sketching and machine-readable diagram generation. The benchmark is released at this https URL. 

---
# Calibrated Prediction Set in Fault Detection with Risk Guarantees via Significance Tests 

**Authors**: Mingchen Mei, Yi Li, YiYao Qian, Zijun Jia  

**Link**: [PDF](https://arxiv.org/pdf/2508.01208)  

**Abstract**: Fault detection is crucial for ensuring the safety and reliability of modern industrial systems. However, a significant scientific challenge is the lack of rigorous risk control and reliable uncertainty quantification in existing diagnostic models, particularly when facing complex scenarios such as distributional shifts. To address this issue, this paper proposes a novel fault detection method that integrates significance testing with the conformal prediction framework to provide formal risk guarantees. The method transforms fault detection into a hypothesis testing task by defining a nonconformity measure based on model residuals. It then leverages a calibration dataset to compute p-values for new samples, which are used to construct prediction sets mathematically guaranteed to contain the true label with a user-specified probability, $1-\alpha$. Fault classification is subsequently performed by analyzing the intersection of the constructed prediction set with predefined normal and fault label sets. Experimental results on cross-domain fault diagnosis tasks validate the theoretical properties of our approach. The proposed method consistently achieves an empirical coverage rate at or above the nominal level ($1-\alpha$), demonstrating robustness even when the underlying point-prediction models perform poorly. Furthermore, the results reveal a controllable trade-off between the user-defined risk level ($\alpha$) and efficiency, where higher risk tolerance leads to smaller average prediction set sizes. This research contributes a theoretically grounded framework for fault detection that enables explicit risk control, enhancing the trustworthiness of diagnostic systems in safety-critical applications and advancing the field from simple point predictions to informative, uncertainty-aware outputs. 

---
# Importance Sampling is All You Need: Predict LLM's performance on new benchmark by reusing existing benchmark 

**Authors**: Junjie Shi, Wei Ma, Shi Ying, Lingxiao Jiang, Yang liu, Bo Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.01203)  

**Abstract**: With the rapid advancement of large language models , code generation has become a key benchmark for evaluating LLM capabilities. However, existing benchmarks face two major challenges: (1) the escalating cost of constructing high-quality test suites and reference solutions, and (2) the increasing risk of data contamination, which undermines the reliability of benchmark-based evaluations. In this paper, we propose BIS, a prompt-centric evaluation framework that enables ground-truth-free prediction of LLM performance on code generation tasks. Rather than executing generated code, BIS estimates performance metrics by analyzing the prompt distribution alone. Built on importance sampling theory and implemented using Importance Weighted Autoencoders, our method reweights samples from existing annotated benchmarks to estimate performance on new, unseen benchmarks. To stabilize the estimation, we introduce weight truncation strategies and compute marginal expectations across the fitted distributions. BIS serves as a complementary tool that supports benchmark development and validation under constrained resources, offering actionable and quick feedback for prompt selection and contamination assessment. We conduct extensive experiments involving 8,000 evaluation points across 4 CodeLlama models and 9 diverse benchmarks. Our framework achieves an average absolute prediction error of 1.1% for code correctness scores, with best- and worst-case errors of 0.3% and 1.9%, respectively. It also generalizes well to other metrics, attaining average absolute errors of 2.15% for pass@1. These results demonstrate the reliability and broad applicability of BIS, which can significantly reduce the cost and effort of benchmarking LLMs in code-related tasks. 

---
# Is Chain-of-Thought Reasoning of LLMs a Mirage? A Data Distribution Lens 

**Authors**: Chengshuai Zhao, Zhen Tan, Pingchuan Ma, Dawei Li, Bohan Jiang, Yancheng Wang, Yingzhen Yang, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01191)  

**Abstract**: Chain-of-Thought (CoT) prompting has been shown to improve Large Language Model (LLM) performance on various tasks. With this approach, LLMs appear to produce human-like reasoning steps before providing answers (a.k.a., CoT reasoning), which often leads to the perception that they engage in deliberate inferential processes. However, some initial findings suggest that CoT reasoning may be more superficial than it appears, motivating us to explore further. In this paper, we study CoT reasoning via a data distribution lens and investigate if CoT reasoning reflects a structured inductive bias learned from in-distribution data, allowing the model to conditionally generate reasoning paths that approximate those seen during training. Thus, its effectiveness is fundamentally bounded by the degree of distribution discrepancy between the training data and the test queries. With this lens, we dissect CoT reasoning via three dimensions: task, length, and format. To investigate each dimension, we design DataAlchemy, an isolated and controlled environment to train LLMs from scratch and systematically probe them under various distribution conditions. Our results reveal that CoT reasoning is a brittle mirage that vanishes when it is pushed beyond training distributions. This work offers a deeper understanding of why and when CoT reasoning fails, emphasizing the ongoing challenge of achieving genuine and generalizable reasoning. 

---
# A Survey on Agent Workflow -- Status and Future 

**Authors**: Chaojia Yu, Zihan Cheng, Hanwen Cui, Yishuo Gao, Zexu Luo, Yijin Wang, Hangbin Zheng, Yong Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01186)  

**Abstract**: In the age of large language models (LLMs), autonomous agents have emerged as a powerful paradigm for achieving general intelligence. These agents dynamically leverage tools, memory, and reasoning capabilities to accomplish user-defined goals. As agent systems grow in complexity, agent workflows-structured orchestration frameworks-have become central to enabling scalable, controllable, and secure AI behaviors. This survey provides a comprehensive review of agent workflow systems, spanning academic frameworks and industrial implementations. We classify existing systems along two key dimensions: functional capabilities (e.g., planning, multi-agent collaboration, external API integration) and architectural features (e.g., agent roles, orchestration flows, specification languages). By comparing over 20 representative systems, we highlight common patterns, potential technical challenges, and emerging trends. We further address concerns related to workflow optimization strategies and security. Finally, we outline open problems such as standardization and multimodal integration, offering insights for future research at the intersection of agent design, workflow infrastructure, and safe automation. 

---
# Benchmarking and Bridging Emotion Conflicts for Multimodal Emotion Reasoning 

**Authors**: Zhiyuan Han, Beier Zhu, Yanlong Xu, Peipei Song, Xun Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01181)  

**Abstract**: Despite their strong performance in multimodal emotion reasoning, existing Multimodal Large Language Models (MLLMs) often overlook the scenarios involving emotion conflicts, where emotional cues from different modalities are inconsistent. To fill this gap, we first introduce CA-MER, a new benchmark designed to examine MLLMs under realistic emotion conflicts. It consists of three subsets: video-aligned, audio-aligned, and consistent, where only one or all modalities reflect the true emotion. However, evaluations on our CA-MER reveal that current state-of-the-art emotion MLLMs systematically over-rely on audio signal during emotion conflicts, neglecting critical cues from visual modality. To mitigate this bias, we propose MoSEAR, a parameter-efficient framework that promotes balanced modality integration. MoSEAR consists of two modules: (1)MoSE, modality-specific experts with a regularized gating mechanism that reduces modality bias in the fine-tuning heads; and (2)AR, an attention reallocation mechanism that rebalances modality contributions in frozen backbones during inference. Our framework offers two key advantages: it mitigates emotion conflicts and improves performance on consistent samples-without incurring a trade-off between audio and visual modalities. Experiments on multiple benchmarks-including MER2023, EMER, DFEW, and our CA-MER-demonstrate that MoSEAR achieves state-of-the-art performance, particularly under modality conflict conditions. 

---
# H2C: Hippocampal Circuit-inspired Continual Learning for Lifelong Trajectory Prediction in Autonomous Driving 

**Authors**: Yunlong Lin, Zirui Li, Guodong Du, Xiaocong Zhao, Cheng Gong, Xinwei Wang, Chao Lu, Jianwei Gong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01158)  

**Abstract**: Deep learning (DL) has shown state-of-the-art performance in trajectory prediction, which is critical to safe navigation in autonomous driving (AD). However, most DL-based methods suffer from catastrophic forgetting, where adapting to a new distribution may cause significant performance degradation in previously learned ones. Such inability to retain learned knowledge limits their applicability in the real world, where AD systems need to operate across varying scenarios with dynamic distributions. As revealed by neuroscience, the hippocampal circuit plays a crucial role in memory replay, effectively reconstructing learned knowledge based on limited resources. Inspired by this, we propose a hippocampal circuit-inspired continual learning method (H2C) for trajectory prediction across varying scenarios. H2C retains prior knowledge by selectively recalling a small subset of learned samples. First, two complementary strategies are developed to select the subset to represent learned knowledge. Specifically, one strategy maximizes inter-sample diversity to represent the distinctive knowledge, and the other estimates the overall knowledge by equiprobable sampling. Then, H2C updates via a memory replay loss function calculated by these selected samples to retain knowledge while learning new data. Experiments based on various scenarios from the INTERACTION dataset are designed to evaluate H2C. Experimental results show that H2C reduces catastrophic forgetting of DL baselines by 22.71% on average in a task-free manner, without relying on manually informed distributional shifts. The implementation is available at this https URL. 

---
# Platonic Representations for Poverty Mapping: Unified Vision-Language Codes or Agent-Induced Novelty? 

**Authors**: Satiyabooshan Murugaboopathy, Connor T. Jerzak, Adel Daoud  

**Link**: [PDF](https://arxiv.org/pdf/2508.01109)  

**Abstract**: We investigate whether socio-economic indicators like household wealth leave recoverable imprints in satellite imagery (capturing physical features) and Internet-sourced text (reflecting historical/economic narratives). Using Demographic and Health Survey (DHS) data from African neighborhoods, we pair Landsat images with LLM-generated textual descriptions conditioned on location/year and text retrieved by an AI search agent from web sources. We develop a multimodal framework predicting household wealth (International Wealth Index) through five pipelines: (i) vision model on satellite images, (ii) LLM using only location/year, (iii) AI agent searching/synthesizing web text, (iv) joint image-text encoder, (v) ensemble of all signals. Our framework yields three contributions. First, fusing vision and agent/LLM text outperforms vision-only baselines in wealth prediction (e.g., R-squared of 0.77 vs. 0.63 on out-of-sample splits), with LLM-internal knowledge proving more effective than agent-retrieved text, improving robustness to out-of-country and out-of-time generalization. Second, we find partial representational convergence: fused embeddings from vision/language modalities correlate moderately (median cosine similarity of 0.60 after alignment), suggesting a shared latent code of material well-being while retaining complementary details, consistent with the Platonic Representation Hypothesis. Although LLM-only text outperforms agent-retrieved data, challenging our Agent-Induced Novelty Hypothesis, modest gains from combining agent data in some splits weakly support the notion that agent-gathered information introduces unique representational structures not fully captured by static LLM knowledge. Third, we release a large-scale multimodal dataset comprising more than 60,000 DHS clusters linked to satellite images, LLM-generated descriptions, and agent-retrieved texts. 

---
# Multispin Physics of AI Tipping Points and Hallucinations 

**Authors**: Neil F. Johnson, Frank Yingjie Huo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01097)  

**Abstract**: Output from generative AI such as ChatGPT, can be repetitive and biased. But more worrying is that this output can mysteriously tip mid-response from good (correct) to bad (misleading or wrong) without the user noticing. In 2024 alone, this reportedly caused $67 billion in losses and several deaths. Establishing a mathematical mapping to a multispin thermal system, we reveal a hidden tipping instability at the scale of the AI's 'atom' (basic Attention head). We derive a simple but essentially exact formula for this tipping point which shows directly the impact of a user's prompt choice and the AI's training bias. We then show how the output tipping can get amplified by the AI's multilayer architecture. As well as helping improve AI transparency, explainability and performance, our results open a path to quantifying users' AI risk and legal liabilities. 

---
# gpuRDF2vec -- Scalable GPU-based RDF2vec 

**Authors**: Martin Böckling, Heiko Paulheim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01073)  

**Abstract**: Generating Knowledge Graph (KG) embeddings at web scale remains challenging. Among existing techniques, RDF2vec combines effectiveness with strong scalability. We present gpuRDF2vec, an open source library that harnesses modern GPUs and supports multi-node execution to accelerate every stage of the RDF2vec pipeline. Extensive experiments on both synthetically generated graphs and real-world benchmarks show that gpuRDF2vec achieves up to a substantial speedup over the currently fastest alternative, i.e., jRDF2vec. In a single-node setup, our walk-extraction phase alone outperforms pyRDF2vec, SparkKGML, and jRDF2vec by a substantial margin using random walks on large/ dense graphs, and scales very well to longer walks, which typically lead to better quality embeddings. Our implementation of gpuRDF2vec enables practitioners and researchers to train high-quality KG embeddings on large-scale graphs within practical time budgets and builds on top of Pytorch Lightning for the scalable word2vec implementation. 

---
# REACT: A Real-Time Edge-AI Based V2X Framework for Accident Avoidance in Autonomous Driving System 

**Authors**: Fengze Yang, Bo Yu, Yang Zhou, Xuewen Luo, Zhengzhong Tu, Chenxi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01057)  

**Abstract**: Collisions caused by human error are the most common type of multi-vehicle crash, highlighting the critical need for autonomous driving (AD) systems to leverage cooperative perception through Vehicle-to-Everything (V2X) communication. This capability extends situational awareness beyond the limitations of onboard sensors. However, current transformer-based V2X frameworks suffer from limited generalization, shallow contextual reasoning, and reliance on mono-modal inputs. Vision-Language Models (VLMs) offer enhanced reasoning and multimodal integration but typically fall short of real-time performance requirements in safety-critical applications. This paper presents REACT, a real-time, V2X-integrated trajectory optimization framework built upon a fine-tuned lightweight VLM. REACT integrates a set of specialized modules that process multimodal inputs into optimized, risk-aware trajectories. To ensure real-time performance on edge devices, REACT incorporates edge adaptation strategies that reduce model complexity and accelerate inference. Evaluated on the DeepAccident benchmark, REACT achieves state-of-the-art performance, a 77% collision rate reduction, a 48.2% Video Panoptic Quality (VPQ), and a 0.57-second inference latency on the Jetson AGX Orin. Ablation studies validate the contribution of each input, module, and edge adaptation strategy. These results demonstrate the feasibility of lightweight VLMs for real-time edge-based cooperative planning and showcase the potential of language-guided contextual reasoning to improve safety and responsiveness in autonomous driving. 

---
# CADDesigner: Conceptual Design of CAD Models Based on General-Purpose Agent 

**Authors**: Jingzhe Ni, Xiaolong Yin, Xintong Li, Xingyu Lu, Ji Wei, Ruofeng Tong, Min Tang, Peng Du  

**Link**: [PDF](https://arxiv.org/pdf/2508.01031)  

**Abstract**: Computer-Aided Design (CAD) plays a pivotal role in industrial manufacturing but typically requires a high level of expertise from designers. To lower the entry barrier and improve design efficiency, we present an agent for CAD conceptual design powered by large language models (LLMs). The agent accepts both abstract textual descriptions and freehand sketches as input, engaging in interactive dialogue with users to refine and clarify design requirements through comprehensive requirement analysis. Built upon a novel Context-Independent Imperative Paradigm (CIP), the agent generates high-quality CAD modeling code. During the generation process, the agent incorporates iterative visual feedback to improve model quality. Generated design cases are stored in a structured knowledge base, enabling continuous improvement of the agent's code generation capabilities. Experimental results demonstrate that our method achieves state-of-the-art performance in CAD code generation. 

---
# AutoEDA: Enabling EDA Flow Automation through Microservice-Based LLM Agents 

**Authors**: Yiyi Lu, Hoi Ian Au, Junyao Zhang, Jingyu Pan, Yiting Wang, Ang Li, Jianyi Zhang, Yiran Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01012)  

**Abstract**: Modern Electronic Design Automation (EDA) workflows, especially the RTL-to-GDSII flow, require heavily manual scripting and demonstrate a multitude of tool-specific interactions which limits scalability and efficiency. While LLMs introduces strides for automation, existing LLM solutions require expensive fine-tuning and do not contain standardized frameworks for integration and evaluation. We introduce AutoEDA, a framework for EDA automation that leverages paralleled learning through the Model Context Protocol (MCP) specific for standardized and scalable natural language experience across the entire RTL-to-GDSII flow. AutoEDA limits fine-tuning through structured prompt engineering, implements intelligent parameter extraction and task decomposition, and provides an extended CodeBLEU metric to evaluate the quality of TCL scripts. Results from experiments over five previously curated benchmarks show improvements in automation accuracy and efficiency, as well as script quality when compared to existing methods. AutoEDA is released open-sourced to support reproducibility and the EDA community. Available at: this https URL 

---
# Cooperative Perception: A Resource-Efficient Framework for Multi-Drone 3D Scene Reconstruction Using Federated Diffusion and NeRF 

**Authors**: Massoud Pourmandi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00967)  

**Abstract**: The proposal introduces an innovative drone swarm perception system that aims to solve problems related to computational limitations and low-bandwidth communication, and real-time scene reconstruction. The framework enables efficient multi-agent 3D/4D scene synthesis through federated learning of shared diffusion model and YOLOv12 lightweight semantic extraction and local NeRF updates while maintaining privacy and scalability. The framework redesigns generative diffusion models for joint scene reconstruction, and improves cooperative scene understanding, while adding semantic-aware compression protocols. The approach can be validated through simulations and potential real-world deployment on drone testbeds, positioning it as a disruptive advancement in multi-agent AI for autonomous systems. 

---
# Knowledge Editing for Multi-Hop Question Answering Using Semantic Analysis 

**Authors**: Dominic Simon, Rickard Ewetz  

**Link**: [PDF](https://arxiv.org/pdf/2508.00914)  

**Abstract**: Large Language Models (LLMs) require lightweight avenues of updating stored information that has fallen out of date. Knowledge Editing (KE) approaches have been successful in updating model knowledge for simple factual queries but struggle with handling tasks that require compositional reasoning such as multi-hop question answering (MQA). We observe that existing knowledge editors leverage decompositional techniques that result in illogical reasoning processes. In this paper, we propose a knowledge editor for MQA based on semantic analysis called CHECK. Our framework is based on insights from an analogy between compilers and reasoning using LLMs. Similar to how source code is first compiled before being executed, we propose to semantically analyze reasoning chains before executing the chains to answer questions. Reasoning chains with semantic errors are revised to ensure consistency through logic optimization and re-prompting the LLM model at a higher temperature. We evaluate the effectiveness of CHECK against five state-of-the-art frameworks on four datasets and achieve an average 22.8% improved MQA accuracy. 

---
# An analysis of AI Decision under Risk: Prospect theory emerges in Large Language Models 

**Authors**: Kenneth Payne  

**Link**: [PDF](https://arxiv.org/pdf/2508.00902)  

**Abstract**: Judgment of risk is key to decision-making under uncertainty. As Daniel Kahneman and Amos Tversky famously discovered, humans do so in a distinctive way that departs from mathematical rationalism. Specifically, they demonstrated experimentally that humans accept more risk when they feel themselves at risk of losing something than when they might gain. I report the first tests of Kahneman and Tversky's landmark 'prospect theory' with Large Language Models, including today's state of the art chain-of-thought 'reasoners'.
In common with humans, I find that prospect theory often anticipates how these models approach risky decisions across a range of scenarios. I also demonstrate that context is key to explaining much of the variance in risk appetite. The 'frame' through which risk is apprehended appears to be embedded within the language of the scenarios tackled by the models. Specifically, I find that military scenarios generate far larger 'framing effects' than do civilian settings, ceteris paribus. My research suggests, therefore, that language models the world, capturing our human heuristics and biases. But also that these biases are uneven - the idea of a 'frame' is richer than simple gains and losses. Wittgenstein's notion of 'language games' explains the contingent, localised biases activated by these scenarios. Finally, I use my findings to reframe the ongoing debate about reasoning and memorisation in LLMs. 

---
# ff4ERA: A new Fuzzy Framework for Ethical Risk Assessment in AI 

**Authors**: Abeer Dyoub, Ivan Letteri, Francesca A. Lisi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00899)  

**Abstract**: The emergence of Symbiotic AI (SAI) introduces new challenges to ethical decision-making as it deepens human-AI collaboration. As symbiosis grows, AI systems pose greater ethical risks, including harm to human rights and trust. Ethical Risk Assessment (ERA) thus becomes crucial for guiding decisions that minimize such risks. However, ERA is hindered by uncertainty, vagueness, and incomplete information, and morality itself is context-dependent and imprecise. This motivates the need for a flexible, transparent, yet robust framework for ERA. Our work supports ethical decision-making by quantitatively assessing and prioritizing multiple ethical risks so that artificial agents can select actions aligned with human values and acceptable risk levels. We introduce ff4ERA, a fuzzy framework that integrates Fuzzy Logic, the Fuzzy Analytic Hierarchy Process (FAHP), and Certainty Factors (CF) to quantify ethical risks via an Ethical Risk Score (ERS) for each risk type. The final ERS combines the FAHP-derived weight, propagated CF, and risk level. The framework offers a robust mathematical approach for collaborative ERA modeling and systematic, step-by-step analysis. A case study confirms that ff4ERA yields context-sensitive, ethically meaningful risk scores reflecting both expert input and sensor-based evidence. Risk scores vary consistently with relevant factors while remaining robust to unrelated inputs. Local sensitivity analysis shows predictable, mostly monotonic behavior across perturbations, and global Sobol analysis highlights the dominant influence of expert-defined weights and certainty factors, validating the model design. Overall, the results demonstrate ff4ERA ability to produce interpretable, traceable, and risk-aware ethical assessments, enabling what-if analyses and guiding designers in calibrating membership functions and expert judgments for reliable ethical decision support. 

---
# AgentTTS: Large Language Model Agent for Test-time Compute-optimal Scaling Strategy in Complex Tasks 

**Authors**: Fali Wang, Hui Liu, Zhenwei Dai, Jingying Zeng, Zhiwei Zhang, Zongyu Wu, Chen Luo, Zhen Li, Xianfeng Tang, Qi He, Suhang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00890)  

**Abstract**: Test-time scaling (TTS) enhances the performance of large language models (LLMs) by allocating additional compute resources during inference. However, existing research primarily investigates TTS in single-stage tasks; while many real-world problems are multi-stage complex tasks, composed of a sequence of heterogeneous subtasks with each subtask requires LLM of specific capability. Therefore, we study a novel problem: the test-time compute-optimal scaling in multi-stage complex tasks, aiming to select suitable models and allocate budgets per subtask to maximize overall performance. TTS in multi-stage tasks introduces two fundamental challenges: (i) The combinatorial search space of model and budget allocations, combined with the high cost of inference, makes brute-force search impractical. (ii) The optimal model and budget allocations across subtasks are interdependent, increasing the complexity of the compute-optimal search. To address this gap, we conduct extensive pilot experiments on four tasks across six datasets, deriving three empirical insights characterizing the behavior of LLMs in multi-stage complex tasks. Informed by these insights, we propose AgentTTS, an LLM-agent-based framework that autonomously searches for compute-optimal allocations through iterative feedback-driven interactions with the execution environment. Experimental results demonstrate that AgentTTS significantly outperforms traditional and other LLM-based baselines in search efficiency, and shows improved robustness to varying training set sizes and enhanced interpretability. 

---
# A Formal Framework for the Definition of 'State': Hierarchical Representation and Meta-Universe Interpretation 

**Authors**: Kei Itoh  

**Link**: [PDF](https://arxiv.org/pdf/2508.00853)  

**Abstract**: This study aims to reinforce the theoretical foundation for diverse systems--including the axiomatic definition of intelligence--by introducing a mathematically rigorous and unified formal structure for the concept of 'state,' which has long been used without consensus or formal clarity. First, a 'hierarchical state grid' composed of two axes--state depth and mapping hierarchy--is proposed to provide a unified notational system applicable across mathematical, physical, and linguistic domains. Next, the 'Intermediate Meta-Universe (IMU)' is introduced to enable explicit descriptions of definers (ourselves) and the languages we use, thereby allowing conscious meta-level operations while avoiding self-reference and logical inconsistency. Building on this meta-theoretical foundation, this study expands inter-universal theory beyond mathematics to include linguistic translation and agent integration, introducing the conceptual division between macrocosm-inter-universal and microcosm-inter-universal operations for broader expressivity. Through these contributions, this paper presents a meta-formal logical framework--grounded in the principle of definition = state--that spans time, language, agents, and operations, providing a mathematically robust foundation applicable to the definition of intelligence, formal logic, and scientific theory at large. 

---
# Exploring Agentic Artificial Intelligence Systems: Towards a Typological Framework 

**Authors**: Christopher Wissuchek, Patrick Zschech  

**Link**: [PDF](https://arxiv.org/pdf/2508.00844)  

**Abstract**: Artificial intelligence (AI) systems are evolving beyond passive tools into autonomous agents capable of reasoning, adapting, and acting with minimal human intervention. Despite their growing presence, a structured framework is lacking to classify and compare these systems. This paper develops a typology of agentic AI systems, introducing eight dimensions that define their cognitive and environmental agency in an ordinal structure. Using a multi-phase methodological approach, we construct and refine this typology, which is then evaluated through a human-AI hybrid approach and further distilled into constructed types. The framework enables researchers and practitioners to analyze varying levels of agency in AI systems. By offering a structured perspective on the progression of AI capabilities, the typology provides a foundation for assessing current systems and anticipating future developments in agentic AI. 

---
# An Efficient Continuous-Time MILP for Integrated Aircraft Hangar Scheduling and Layout 

**Authors**: Shayan Farhang Pazhooh, Hossein Shams Shemirani  

**Link**: [PDF](https://arxiv.org/pdf/2508.02640)  

**Abstract**: Efficient management of aircraft maintenance hangars is a critical operational challenge, involving complex, interdependent decisions regarding aircraft scheduling and spatial allocation. This paper introduces a novel continuous-time mixed-integer linear programming (MILP) model to solve this integrated spatio-temporal problem. By treating time as a continuous variable, our formulation overcomes the scalability limitations of traditional discrete-time approaches. The performance of the exact model is benchmarked against a constructive heuristic, and its practical applicability is demonstrated through a custom-built visualization dashboard. Computational results are compelling: the model solves instances with up to 25 aircraft to proven optimality, often in mere seconds, and for large-scale cases of up to 40 aircraft, delivers high-quality solutions within known optimality gaps. In all tested scenarios, the resulting solutions consistently and significantly outperform the heuristic, which highlights the framework's substantial economic benefits and provides valuable managerial insights into the trade-off between solution time and optimality. 

---
# HyCodePolicy: Hybrid Language Controllers for Multimodal Monitoring and Decision in Embodied Agents 

**Authors**: Yibin Liu, Zhixuan Liang, Zanxin Chen, Tianxing Chen, Mengkang Hu, Wanxi Dong, Congsheng Xu, Zhaoming Han, Yusen Qin, Yao Mu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02629)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have enabled richer perceptual grounding for code policy generation in embodied agents. However, most existing systems lack effective mechanisms to adaptively monitor policy execution and repair codes during task completion. In this work, we introduce HyCodePolicy, a hybrid language-based control framework that systematically integrates code synthesis, geometric grounding, perceptual monitoring, and iterative repair into a closed-loop programming cycle for embodied agents. Technically, given a natural language instruction, our system first decomposes it into subgoals and generates an initial executable program grounded in object-centric geometric primitives. The program is then executed in simulation, while a vision-language model (VLM) observes selected checkpoints to detect and localize execution failures and infer failure reasons. By fusing structured execution traces capturing program-level events with VLM-based perceptual feedback, HyCodePolicy infers failure causes and repairs programs. This hybrid dual feedback mechanism enables self-correcting program synthesis with minimal human supervision. Our results demonstrate that HyCodePolicy significantly improves the robustness and sample efficiency of robot manipulation policies, offering a scalable strategy for integrating multimodal reasoning into autonomous decision-making pipelines. 

---
# AutoML-Med: A Framework for Automated Machine Learning in Medical Tabular Data 

**Authors**: Riccardo Francia, Maurizio Leone, Giorgio Leonardi, Stefania Montani, Marzio Pennisi, Manuel Striani, Sandra D'Alfonso  

**Link**: [PDF](https://arxiv.org/pdf/2508.02625)  

**Abstract**: Medical datasets are typically affected by issues such as missing values, class imbalance, a heterogeneous feature types, and a high number of features versus a relatively small number of samples, preventing machine learning models from obtaining proper results in classification and regression tasks. This paper introduces AutoML-Med, an Automated Machine Learning tool specifically designed to address these challenges, minimizing user intervention and identifying the optimal combination of preprocessing techniques and predictive models. AutoML-Med's architecture incorporates Latin Hypercube Sampling (LHS) for exploring preprocessing methods, trains models using selected metrics, and utilizes Partial Rank Correlation Coefficient (PRCC) for fine-tuned optimization of the most influential preprocessing steps. Experimental results demonstrate AutoML-Med's effectiveness in two different clinical settings, achieving higher balanced accuracy and sensitivity, which are crucial for identifying at-risk patients, compared to other state-of-the-art tools. AutoML-Med's ability to improve prediction results, especially in medical datasets with sparse data and class imbalance, highlights its potential to streamline Machine Learning applications in healthcare. 

---
# Meta-RAG on Large Codebases Using Code Summarization 

**Authors**: Vali Tawosia, Salwa Alamir, Xiaomo Liu, Manuela Veloso  

**Link**: [PDF](https://arxiv.org/pdf/2508.02611)  

**Abstract**: Large Language Model (LLM) systems have been at the forefront of applied Artificial Intelligence (AI) research in a multitude of domains. One such domain is software development, where researchers have pushed the automation of a number of code tasks through LLM agents. Software development is a complex ecosystem, that stretches far beyond code implementation and well into the realm of code maintenance. In this paper, we propose a multi-agent system to localize bugs in large pre-existing codebases using information retrieval and LLMs. Our system introduces a novel Retrieval Augmented Generation (RAG) approach, Meta-RAG, where we utilize summaries to condense codebases by an average of 79.8\%, into a compact, structured, natural language representation. We then use an LLM agent to determine which parts of the codebase are critical for bug resolution, i.e. bug localization. We demonstrate the usefulness of Meta-RAG through evaluation with the SWE-bench Lite dataset. Meta-RAG scores 84.67 % and 53.0 % for file-level and function-level correct localization rates, respectively, achieving state-of-the-art performance. 

---
# Entity Representation Learning Through Onsite-Offsite Graph for Pinterset Ads 

**Authors**: Jiayin Jin, Zhimeng Pan, Yang Tang, Jiarui Feng, Kungang Li, Chongyuan Xiang, Jiacheng Li, Runze Su, Siping Ji, Han Sun, Ling Leng, Prathibha Deshikachar  

**Link**: [PDF](https://arxiv.org/pdf/2508.02609)  

**Abstract**: Graph Neural Networks (GNN) have been extensively applied to industry recommendation systems, as seen in models like GraphSage\cite{GraphSage}, TwHIM\cite{TwHIM}, LiGNN\cite{LiGNN} etc. In these works, graphs were constructed based on users' activities on the platforms, and various graph models were developed to effectively learn node embeddings. In addition to users' onsite activities, their offsite conversions are crucial for Ads models to capture their shopping interest. To better leverage offsite conversion data and explore the connection between onsite and offsite activities, we constructed a large-scale heterogeneous graph based on users' onsite ad interactions and opt-in offsite conversion activities. Furthermore, we introduced TransRA (TransR\cite{TransR} with Anchors), a novel Knowledge Graph Embedding (KGE) model, to more efficiently integrate graph embeddings into Ads ranking models. However, our Ads ranking models initially struggled to directly incorporate Knowledge Graph Embeddings (KGE), and only modest gains were observed during offline experiments. To address this challenge, we employed the Large ID Embedding Table technique and innovated an attention based KGE finetuning approach within the Ads ranking models. As a result, we observed a significant AUC lift in Click-Through Rate (CTR) and Conversion Rate (CVR) prediction models. Moreover, this framework has been deployed in Pinterest's Ads Engagement Model and contributed to $2.69\%$ CTR lift and $1.34\%$ CPC reduction. We believe the techniques presented in this paper can be leveraged by other large-scale industrial models. 

---
# StructSynth: Leveraging LLMs for Structure-Aware Tabular Data Synthesis in Low-Data Regimes 

**Authors**: Siyi Liu, Yujia Zheng, Yongqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02601)  

**Abstract**: The application of machine learning on tabular data in specialized domains is severely limited by data scarcity. While generative models offer a solution, traditional methods falter in low-data regimes, and recent Large Language Models (LLMs) often ignore the explicit dependency structure of tabular data, leading to low-fidelity synthetics. To address these limitations, we introduce StructSynth, a novel framework that integrates the generative power of LLMs with robust structural control. StructSynth employs a two-stage architecture. First, it performs explicit structure discovery to learn a Directed Acyclic Graph (DAG) from the available data. Second, this learned structure serves as a high-fidelity blueprint to steer the LLM's generation process, forcing it to adhere to the learned feature dependencies and thereby ensuring the generated data respects the underlying structure by design. Our extensive experiments demonstrate that StructSynth produces synthetic data with significantly higher structural integrity and downstream utility than state-of-the-art methods. It proves especially effective in challenging low-data scenarios, successfully navigating the trade-off between privacy preservation and statistical fidelity. 

---
# Explainable AI for Automated User-specific Feedback in Surgical Skill Acquisition 

**Authors**: Catalina Gomez, Lalithkumar Seenivasan, Xinrui Zou, Jeewoo Yoon, Sirui Chu, Ariel Leong, Patrick Kramer, Yu-Chun Ku, Jose L. Porras, Alejandro Martin-Gomez, Masaru Ishii, Mathias Unberath  

**Link**: [PDF](https://arxiv.org/pdf/2508.02593)  

**Abstract**: Traditional surgical skill acquisition relies heavily on expert feedback, yet direct access is limited by faculty availability and variability in subjective assessments. While trainees can practice independently, the lack of personalized, objective, and quantitative feedback reduces the effectiveness of self-directed learning. Recent advances in computer vision and machine learning have enabled automated surgical skill assessment, demonstrating the feasibility of automatic competency evaluation. However, it is unclear whether such Artificial Intelligence (AI)-driven feedback can contribute to skill acquisition. Here, we examine the effectiveness of explainable AI (XAI)-generated feedback in surgical training through a human-AI study. We create a simulation-based training framework that utilizes XAI to analyze videos and extract surgical skill proxies related to primitive actions. Our intervention provides automated, user-specific feedback by comparing trainee performance to expert benchmarks and highlighting deviations from optimal execution through understandable proxies for actionable guidance. In a prospective user study with medical students, we compare the impact of XAI-guided feedback against traditional video-based coaching on task outcomes, cognitive load, and trainees' perceptions of AI-assisted learning. Results showed improved cognitive load and confidence post-intervention. While no differences emerged between the two feedback types in reducing performance gaps or practice adjustments, trends in the XAI group revealed desirable effects where participants more closely mimicked expert practice. This work encourages the study of explainable AI in surgical education and the development of data-driven, adaptive feedback mechanisms that could transform learning experiences and competency assessment. 

---
# Parameter-Efficient Routed Fine-Tuning: Mixture-of-Experts Demands Mixture of Adaptation Modules 

**Authors**: Yilun Liu, Yunpu Ma, Yuetian Lu, Shuo Chen, Zifeng Ding, Volker Tresp  

**Link**: [PDF](https://arxiv.org/pdf/2508.02587)  

**Abstract**: Mixture-of-Experts (MoE) benefits from a dynamic routing mechanism among their specialized experts, which existing Parameter- Efficient Fine-Tuning (PEFT) strategies fail to leverage. This motivates us to investigate whether adaptation modules themselves should incorporate routing mechanisms to align with MoE's multi-expert architecture. We analyze dynamics of core components when applying PEFT to MoE language models and examine how different routing strategies affect adaptation effectiveness. Extensive experiments adapting OLMoE-1B-7B and Mixtral-8x7B on various commonsense and math reasoning tasks validate the performance and efficiency of our routed approach. We identify the optimal configurations for different scenarios and provide empirical analyses with practical insights to facilitate better PEFT and MoE applications. 

---
# MArgE: Meshing Argumentative Evidence from Multiple Large Language Models for Justifiable Claim Verification 

**Authors**: Ming Pok Ng, Junqi Jiang, Gabriel Freedman, Antonio Rago, Francesca Toni  

**Link**: [PDF](https://arxiv.org/pdf/2508.02584)  

**Abstract**: Leveraging outputs from multiple large language models (LLMs) is emerging as a method for harnessing their power across a wide range of tasks while mitigating their capacity for making errors, e.g., hallucinations. However, current approaches to combining insights from multiple LLMs often involve unstructured interactions (e.g., free debate), resulting in model generations that are not faithfully justifiable. In this work, we introduce MArgE, a novel framework to provide formal structure to the evidence from each LLM, in the form of a tree of extracted arguments, for the task of claim verification. We use a variant of Argumentative LLMs (ArgLLMs), i.e. LLMs driven by frameworks and semantics from the field of computational argumentation, to construct structured argument trees for given claims. This process creates an inspectable pathway from the initial arguments to the final claim verification decisions, providing a faithful justification thereof. We show experimentally that MArgE can significantly outperform single LLMs, including three open-source models (4B to 8B parameters), GPT-4o-mini and existing ArgLLMs, as well as prior methods for unstructured multi-LLM debates. We thus demonstrate the advantages of incorporating formal, argumentative reasoning mechanisms when combining multiple LLM outputs. 

---
# EHSAN: Leveraging ChatGPT in a Hybrid Framework for Arabic Aspect-Based Sentiment Analysis in Healthcare 

**Authors**: Eman Alamoudi, Ellis Solaiman  

**Link**: [PDF](https://arxiv.org/pdf/2508.02574)  

**Abstract**: Arabic-language patient feedback remains under-analysed because dialect diversity and scarce aspect-level sentiment labels hinder automated assessment. To address this gap, we introduce EHSAN, a data-centric hybrid pipeline that merges ChatGPT pseudo-labelling with targeted human review to build the first explainable Arabic aspect-based sentiment dataset for healthcare. Each sentence is annotated with an aspect and sentiment label (positive, negative, or neutral), forming a pioneering Arabic dataset aligned with healthcare themes, with ChatGPT-generated rationales provided for each label to enhance transparency. To evaluate the impact of annotation quality on model performance, we created three versions of the training data: a fully supervised set with all labels reviewed by humans, a semi-supervised set with 50% human review, and an unsupervised set with only machine-generated labels. We fine-tuned two transformer models on these datasets for both aspect and sentiment classification. Experimental results show that our Arabic-specific model achieved high accuracy even with minimal human supervision, reflecting only a minor performance drop when using ChatGPT-only labels. Reducing the number of aspect classes notably improved classification metrics across the board. These findings demonstrate an effective, scalable approach to Arabic aspect-based sentiment analysis (SA) in healthcare, combining large language model annotation with human expertise to produce a robust and explainable dataset. Future directions include generalisation across hospitals, prompt refinement, and interpretable data-driven modelling. 

---
# Dynamic Feature Selection based on Rule-based Learning for Explainable Classification with Uncertainty Quantification 

**Authors**: Javier Fumanal-Idocin, Raquel Fernandez-Peralta, Javier Andreu-Perez  

**Link**: [PDF](https://arxiv.org/pdf/2508.02566)  

**Abstract**: Dynamic feature selection (DFS) offers a compelling alternative to traditional, static feature selection by adapting the selected features to each individual sample. Unlike classical methods that apply a uniform feature set, DFS customizes feature selection per sample, providing insight into the decision-making process for each case. DFS is especially significant in settings where decision transparency is key, i.e., clinical decisions; however, existing methods use opaque models, which hinder their applicability in real-life scenarios. This paper introduces a novel approach leveraging a rule-based system as a base classifier for the DFS process, which enhances decision interpretability compared to neural estimators. We also show how this method provides a quantitative measure of uncertainty for each feature query and can make the feature selection process computationally lighter by constraining the feature search space. We also discuss when greedy selection of conditional mutual information is equivalent to selecting features that minimize the difference with respect to the global model predictions. Finally, we demonstrate the competitive performance of our rule-based DFS approach against established and state-of-the-art greedy and RL methods, which are mostly considered opaque, compared to our explainable rule-based system. 

---
# Stakeholder Perspectives on Humanistic Implementation of Computer Perception in Healthcare: A Qualitative Study 

**Authors**: Kristin M. Kostick-Quenet, Meghan E. Hurley, Syed Ayaz, John Herrington, Casey Zampella, Julia Parish-Morris, Birkan Tunç, Gabriel Lázaro-Muñoz, J.S. Blumenthal-Barby, Eric A. Storch  

**Link**: [PDF](https://arxiv.org/pdf/2508.02550)  

**Abstract**: Computer perception (CP) technologies (digital phenotyping, affective computing and related passive sensing approaches) offer unprecedented opportunities to personalize healthcare, but provoke concerns about privacy, bias and the erosion of empathic, relationship-centered practice. A comprehensive understanding of perceived risks, benefits, and implementation challenges from those who design, deploy and experience these tools in real-world settings remains elusive. This study provides the first evidence-based account of key stakeholder perspectives on the relational, technical, and governance challenges raised by the integration of CP technologies into patient care. We conducted in-depth, semi-structured interviews with 102 stakeholders: adolescent patients and their caregivers, frontline clinicians, technology developers, and ethics, legal, policy or philosophy scholars. Transcripts underwent thematic analysis by a multidisciplinary team; reliability was enhanced through double coding and consensus adjudication. Stakeholders articulated seven interlocking concern domains: (1) trustworthiness and data integrity; (2) patient-specific relevance; (3) utility and workflow integration; (4) regulation and governance; (5) privacy and data protection; (6) direct and indirect patient harms; and (7) philosophical critiques of reductionism. To operationalize humanistic safeguards, we propose "personalized roadmaps": co-designed plans that predetermine which metrics will be monitored, how and when feedback is shared, thresholds for clinical action, and procedures for reconciling discrepancies between algorithmic inferences and lived experience. By translating these insights into personalized roadmaps, we offer a practical framework for developers, clinicians and policymakers seeking to harness continuous behavioral data while preserving the humanistic core of care. 

---
# The KG-ER Conceptual Schema Language 

**Authors**: Enrico Franconi, Benoît Groz, Jan Hidders, Nina Pardal, Sławek Staworko, Jan Van den Bussche, Piotr Wieczorek  

**Link**: [PDF](https://arxiv.org/pdf/2508.02548)  

**Abstract**: We propose KG-ER, a conceptual schema language for knowledge graphs that describes the structure of knowledge graphs independently of their representation (relational databases, property graphs, RDF) while helping to capture the semantics of the information stored in a knowledge graph. 

---
# What are you sinking? A geometric approach on attention sink 

**Authors**: Valeria Ruscio, Umberto Nanni, Fabrizio Silvestri  

**Link**: [PDF](https://arxiv.org/pdf/2508.02546)  

**Abstract**: Attention sink (AS) is a consistent pattern in transformer attention maps where certain tokens (often special tokens or positional anchors) disproportionately attract attention from other tokens. We show that in transformers, AS is not an architectural artifact, but it is the manifestation of a fundamental geometric principle: the establishment of reference frames that anchor representational spaces. We analyze several architectures and identify three distinct reference frame types, centralized, distributed, and bidirectional, that correlate with the attention sink phenomenon. We show that they emerge during the earliest stages of training as optimal solutions to the problem of establishing stable coordinate systems in high-dimensional spaces. We show the influence of architecture components, particularly position encoding implementations, on the specific type of reference frame. This perspective transforms our understanding of transformer attention mechanisms and provides insights for both architecture design and the relationship with AS. 

---
# Automatic Identification of Machine Learning-Specific Code Smells 

**Authors**: Peter Hamfelt, Ricardo Britto, Lincoln Rocha, Camilo Almendra  

**Link**: [PDF](https://arxiv.org/pdf/2508.02541)  

**Abstract**: Machine learning (ML) has rapidly grown in popularity, becoming vital to many industries. Currently, the research on code smells in ML applications lacks tools and studies that address the identification and validity of ML-specific code smells. This work investigates suitable methods and tools to design and develop a static code analysis tool (MLpylint) based on code smell criteria. This research employed the Design Science Methodology. In the problem identification phase, a literature review was conducted to identify ML-specific code smells. In solution design, a secondary literature review and consultations with experts were performed to select methods and tools for implementing the tool. We evaluated the tool on data from 160 open-source ML applications sourced from GitHub. We also conducted a static validation through an expert survey involving 15 ML professionals. The results indicate the effectiveness and usefulness of the MLpylint. We aim to extend our current approach by investigating ways to introduce MLpylint seamlessly into development workflows, fostering a more productive and innovative developer environment. 

---
# Decomposed Reasoning with Reinforcement Learning for Relevance Assessment in UGC Platforms 

**Authors**: Xiaowei Yuan, Lei Jin, Haoxin Zhang, Yan Gao, Yi Wu, Yao Hu, Ziyang Huang, Jun Zhao, Kang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02506)  

**Abstract**: Retrieval-augmented generation (RAG) plays a critical role in user-generated content (UGC) platforms, but its effectiveness depends heavily on accurate relevance assessment of query-document pairs. Despite recent advances in applying large language models (LLMs) to relevance modeling, UGC platforms present unique challenges: 1) ambiguous user intent due to sparse user feedback in RAG scenarios, and 2) substantial noise introduced by informal and unstructured language. To address these issues, we propose the Reinforced Reasoning Model for Relevance Assessment (R3A), which introduces a decomposed reasoning framework over queries and candidate documents before scoring. R3A first leverages auxiliary high-ranked documents within the platform to infer latent query intent. It then performs verbatim fragment extraction to justify relevance decisions, thereby reducing errors caused by noisy UGC. Based on a reinforcement learning framework, R3A is optimized to mitigate distortions arising from ambiguous queries and unstructured content. Experimental results show that R3A significantly outperforms existing baseline methods in terms of relevance accuracy, across both offline benchmarks and online experiments. 

---
# AIAP: A No-Code Workflow Builder for Non-Experts with Natural Language and Multi-Agent Collaboration 

**Authors**: Hyunjn An, Yongwon Kim, Wonduk Seo, Joonil Park, Daye Kang, Changhoon Oh, Dokyun Kim, Seunghyun Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.02470)  

**Abstract**: While many tools are available for designing AI, non-experts still face challenges in clearly expressing their intent and managing system complexity. We introduce AIAP, a no-code platform that integrates natural language input with visual workflows. AIAP leverages a coordinated multi-agent system to decompose ambiguous user instructions into modular, actionable steps, hidden from users behind a unified interface. A user study involving 32 participants showed that AIAP's AI-generated suggestions, modular workflows, and automatic identification of data, actions, and context significantly improved participants' ability to develop services intuitively. These findings highlight that natural language-based visual programming significantly reduces barriers and enhances user experience in AI service design. 

---
# TreeRanker: Fast and Model-agnostic Ranking System for Code Suggestions in IDEs 

**Authors**: Daniele Cipollone, Egor Bogomolov, Arie van Deursen, Maliheh Izadi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02455)  

**Abstract**: Token-level code completion is one of the most critical features in modern Integrated Development Environments (IDEs). It assists developers by suggesting relevant identifiers and APIs during coding. While completions are typically derived from static analysis, their usefulness depends heavily on how they are ranked, as correct predictions buried deep in the list are rarely seen by users. Most current systems rely on hand-crafted heuristics or lightweight machine learning models trained on user logs, which can be further improved to capture context information and generalize across projects and coding styles. In this work, we propose a new scoring approach to ranking static completions using language models in a lightweight and model-agnostic way. Our method organizes all valid completions into a prefix tree and performs a single greedy decoding pass to collect token-level scores across the tree. This enables a precise token-aware ranking without needing beam search, prompt engineering, or model adaptations. The approach is fast, architecture-agnostic, and compatible with already deployed models for code completion. These findings highlight a practical and effective pathway for integrating language models into already existing tools within IDEs, and ultimately providing smarter and more responsive developer assistance. 

---
# Dynamic Forgetting and Spatio-Temporal Periodic Interest Modeling for Local-Life Service Recommendation 

**Authors**: Zhaoyu Hu, Hao Guo, Yuan Tian, Erpeng Xue, Jianyang Wang, Xianyang Qi, Hongxiang Lin, Lei Wang, Sheng Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02451)  

**Abstract**: In the context of the booming digital economy, recommendation systems, as a key link connecting users and numerous services, face challenges in modeling user behavior sequences on local-life service platforms, including the sparsity of long sequences and strong spatio-temporal dependence. Such challenges can be addressed by drawing an analogy to the forgetting process in human memory. This is because users' responses to recommended content follow the recency effect and the cyclicality of memory. By exploring this, this paper introduces the forgetting curve and proposes Spatio-Temporal periodic Interest Modeling (STIM) with long sequences for local-life service recommendation. STIM integrates three key components: a dynamic masking module based on the forgetting curve, which is used to extract both recent spatiotemporal features and periodic spatiotemporal features; a query-based mixture of experts (MoE) approach that can adaptively activate expert networks under different dynamic masks, enabling the collaborative modeling of time, location, and items; and a hierarchical multi-interest network unit, which captures multi-interest representations by modeling the hierarchical interactions between the shallow and deep semantics of users' recent behaviors. By introducing the STIM method, we conducted online A/B tests and achieved a 1.54\% improvement in gross transaction volume (GTV). In addition, extended offline experiments also showed improvements. STIM has been deployed in a large-scale local-life service recommendation system, serving hundreds of millions of daily active users in core application scenarios. 

---
# Assessing the Reliability and Validity of Large Language Models for Automated Assessment of Student Essays in Higher Education 

**Authors**: Andrea Gaggioli, Giuseppe Casaburi, Leonardo Ercolani, Francesco Collova', Pietro Torre, Fabrizio Davide  

**Link**: [PDF](https://arxiv.org/pdf/2508.02442)  

**Abstract**: This study investigates the reliability and validity of five advanced Large Language Models (LLMs), Claude 3.5, DeepSeek v2, Gemini 2.5, GPT-4, and Mistral 24B, for automated essay scoring in a real world higher education context. A total of 67 Italian-language student essays, written as part of a university psychology course, were evaluated using a four-criterion rubric (Pertinence, Coherence, Originality, Feasibility). Each model scored all essays across three prompt replications to assess intra-model stability. Human-LLM agreement was consistently low and non-significant (Quadratic Weighted Kappa), and within-model reliability across replications was similarly weak (median Kendall's W < 0.30). Systematic scoring divergences emerged, including a tendency to inflate Coherence and inconsistent handling of context-dependent dimensions. Inter-model agreement analysis revealed moderate convergence for Coherence and Originality, but negligible concordance for Pertinence and Feasibility. Although limited in scope, these findings suggest that current LLMs may struggle to replicate human judgment in tasks requiring disciplinary insight and contextual sensitivity. Human oversight remains critical when evaluating open-ended academic work, particularly in interpretive domains. 

---
# Multi-Class Human/Object Detection on Robot Manipulators using Proprioceptive Sensing 

**Authors**: Justin Hehli, Marco Heiniger, Maryam Rezayati, Hans Wernher van de Venn  

**Link**: [PDF](https://arxiv.org/pdf/2508.02425)  

**Abstract**: In physical human-robot collaboration (pHRC) settings, humans and robots collaborate directly in shared environments. Robots must analyze interactions with objects to ensure safety and facilitate meaningful workflows. One critical aspect is human/object detection, where the contacted object is identified. Past research introduced binary machine learning classifiers to distinguish between soft and hard objects. This study improves upon those results by evaluating three-class human/object detection models, offering more detailed contact analysis. A dataset was collected using the Franka Emika Panda robot manipulator, exploring preprocessing strategies for time-series analysis. Models including LSTM, GRU, and Transformers were trained on these datasets. The best-performing model achieved 91.11\% accuracy during real-time testing, demonstrating the feasibility of multi-class detection models. Additionally, a comparison of preprocessing strategies suggests a sliding window approach is optimal for this task. 

---
# Emergence of Fair Leaders via Mediators in Multi-Agent Reinforcement Learning 

**Authors**: Akshay Dodwadmath, Setareh Maghsudi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02421)  

**Abstract**: Stackelberg games and their resulting equilibria have received increasing attention in the multi-agent reinforcement learning literature. Each stage of a traditional Stackelberg game involves a leader(s) acting first, followed by the followers. In situations where the roles of leader(s) and followers can be interchanged, the designated role can have considerable advantages, for example, in first-mover advantage settings. Then the question arises: Who should be the leader and when? A bias in the leader selection process can lead to unfair outcomes. This problem is aggravated if the agents are self-interested and care only about their goals and rewards. We formally define this leader selection problem and show its relation to fairness in agents' returns. Furthermore, we propose a multi-agent reinforcement learning framework that maximizes fairness by integrating mediators. Mediators have previously been used in the simultaneous action setting with varying levels of control, such as directly performing agents' actions or just recommending them. Our framework integrates mediators in the Stackelberg setting with minimal control (leader selection). We show that the presence of mediators leads to self-interested agents taking fair actions, resulting in higher overall fairness in agents' returns. 

---
# HGTS-Former: Hierarchical HyperGraph Transformer for Multivariate Time Series Analysis 

**Authors**: Xiao Wang, Hao Si, Fan Zhang, Xiaoya Zhou, Dengdi Sun, Wanli Lyu, Qingquan Yang, Jin Tang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02411)  

**Abstract**: Multivariate time series analysis has long been one of the key research topics in the field of artificial intelligence. However, analyzing complex time series data remains a challenging and unresolved problem due to its high dimensionality, dynamic nature, and complex interactions among variables. Inspired by the strong structural modeling capability of hypergraphs, this paper proposes a novel hypergraph-based time series transformer backbone network, termed HGTS-Former, to address the multivariate coupling in time series data. Specifically, given the multivariate time series signal, we first normalize and embed each patch into tokens. Then, we adopt the multi-head self-attention to enhance the temporal representation of each patch. The hierarchical hypergraphs are constructed to aggregate the temporal patterns within each channel and fine-grained relations between different variables. After that, we convert the hyperedge into node features through the EdgeToNode module and adopt the feed-forward network to further enhance the output features. Extensive experiments conducted on two multivariate time series tasks and eight datasets fully validated the effectiveness of our proposed HGTS-Former. The source code will be released on this https URL. 

---
# Hydra: Accurate Multi-Modal Leaf Wetness Sensing with mm-Wave and Camera Fusion 

**Authors**: Yimeng Liu, Maolin Gan, Huaili Zeng, Li Liu, Younsuk Dong, Zhichao Cao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02409)  

**Abstract**: Leaf Wetness Duration (LWD), the time that water remains on leaf surfaces, is crucial in the development of plant diseases. Existing LWD detection lacks standardized measurement techniques, and variations across different plant characteristics limit its effectiveness. Prior research proposes diverse approaches, but they fail to measure real natural leaves directly and lack resilience in various environmental conditions. This reduces the precision and robustness, revealing a notable practical application and effectiveness gap in real-world agricultural settings. This paper presents Hydra, an innovative approach that integrates millimeter-wave (mm-Wave) radar with camera technology to detect leaf wetness by determining if there is water on the leaf. We can measure the time to determine the LWD based on this detection. Firstly, we design a Convolutional Neural Network (CNN) to selectively fuse multiple mm-Wave depth images with an RGB image to generate multiple feature images. Then, we develop a transformer-based encoder to capture the inherent connection among the multiple feature images to generate a feature map, which is further fed to a classifier for detection. Moreover, we augment the dataset during training to generalize our model. Implemented using a frequency-modulated continuous-wave (FMCW) radar within the 76 to 81 GHz band, Hydra's performance is meticulously evaluated on plants, demonstrating the potential to classify leaf wetness with up to 96% accuracy across varying scenarios. Deploying Hydra in the farm, including rainy, dawn, or poorly light nights, it still achieves an accuracy rate of around 90%. 

---
# CompressKV: Semantic Retrieval Heads Know What Tokens are Not Important Before Generation 

**Authors**: Xiaolin Lin, Jingcun Wang, Olga Kondrateva, Yiyu Shi, Bing Li, Grace Li Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02401)  

**Abstract**: Recent advances in large language models (LLMs) have significantly boosted long-context processing. However, the increasing key-value (KV) cache size poses critical challenges to memory and execution efficiency. Most KV cache compression methods rely on heuristic token eviction using all attention heads in Grouped Query Attention (GQA)-based LLMs. This method ignores the different functionalities of attention heads, leading to the eviction of critical tokens and thus degrades the performance of LLMs.
To address the issue above, instead of using all the attention heads in GQA-based LLMs to determine important tokens as in the previous work, we first identify the attention heads in each layer that are not only capable of retrieving the initial and final tokens of a prompt, but also capable of retrieving important tokens within the text and attending to their surrounding semantic context. Afterwards, we exploit such heads to determine the important tokens and retain their corresponding KV cache pairs. Furthermore, we analyze the cache eviction error of each layer individually and introduce a layer-adaptive KV cache allocation strategy. Experimental results demonstrate the proposed CompressKV consistently outperforms state-of-the-art approaches under various memory budgets on LongBench and Needle-in-a-Haystack benchmarks. Our code is publicly available at: this https URL. 

---
# Inference-time Scaling for Diffusion-based Audio Super-resolution 

**Authors**: Yizhu Jin, Zhen Ye, Zeyue Tian, Haohe Liu, Qiuqiang Kong, Yike Guo, Wei Xue  

**Link**: [PDF](https://arxiv.org/pdf/2508.02391)  

**Abstract**: Diffusion models have demonstrated remarkable success in generative tasks, including audio super-resolution (SR). In many applications like movie post-production and album mastering, substantial computational budgets are available for achieving superior audio quality. However, while existing diffusion approaches typically increase sampling steps to improve quality, the performance remains fundamentally limited by the stochastic nature of the sampling process, leading to high-variance and quality-limited outputs. Here, rather than simply increasing the number of sampling steps, we propose a different paradigm through inference-time scaling for SR, which explores multiple solution trajectories during the sampling process. Different task-specific verifiers are developed, and two search algorithms, including the random search and zero-order search for SR, are introduced. By actively guiding the exploration of the high-dimensional solution space through verifier-algorithm combinations, we enable more robust and higher-quality outputs. Through extensive validation across diverse audio domains (speech, music, sound effects) and frequency ranges, we demonstrate consistent performance gains, achieving improvements of up to 9.70% in aesthetics, 5.88% in speaker similarity, 15.20% in word error rate, and 46.98% in spectral distance for speech SR from 4kHz to 24kHz, showcasing the effectiveness of our approach. Audio samples are available at: this https URL. 

---
# Text2Lip: Progressive Lip-Synced Talking Face Generation from Text via Viseme-Guided Rendering 

**Authors**: Xu Wang, Shengeng Tang, Fei Wang, Lechao Cheng, Dan Guo, Feng Xue, Richang Hong  

**Link**: [PDF](https://arxiv.org/pdf/2508.02362)  

**Abstract**: Generating semantically coherent and visually accurate talking faces requires bridging the gap between linguistic meaning and facial articulation. Although audio-driven methods remain prevalent, their reliance on high-quality paired audio visual data and the inherent ambiguity in mapping acoustics to lip motion pose significant challenges in terms of scalability and robustness. To address these issues, we propose Text2Lip, a viseme-centric framework that constructs an interpretable phonetic-visual bridge by embedding textual input into structured viseme sequences. These mid-level units serve as a linguistically grounded prior for lip motion prediction. Furthermore, we design a progressive viseme-audio replacement strategy based on curriculum learning, enabling the model to gradually transition from real audio to pseudo-audio reconstructed from enhanced viseme features via cross-modal attention. This allows for robust generation in both audio-present and audio-free scenarios. Finally, a landmark-guided renderer synthesizes photorealistic facial videos with accurate lip synchronization. Extensive evaluations show that Text2Lip outperforms existing approaches in semantic fidelity, visual realism, and modality robustness, establishing a new paradigm for controllable and flexible talking face generation. Our project homepage is this https URL. 

---
# mmWave Radar-Based Non-Line-of-Sight Pedestrian Localization at T-Junctions Utilizing Road Layout Extraction via Camera 

**Authors**: Byeonggyu Park, Hee-Yeun Kim, Byonghyok Choi, Hansang Cho, Byungkwan Kim, Soomok Lee, Mingu Jeon, Seong-Woo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.02348)  

**Abstract**: Pedestrians Localization in Non-Line-of-Sight (NLoS) regions within urban environments poses a significant challenge for autonomous driving systems. While mmWave radar has demonstrated potential for detecting objects in such scenarios, the 2D radar point cloud (PCD) data is susceptible to distortions caused by multipath reflections, making accurate spatial inference difficult. Additionally, although camera images provide high-resolution visual information, they lack depth perception and cannot directly observe objects in NLoS regions. In this paper, we propose a novel framework that interprets radar PCD through road layout inferred from camera for localization of NLoS pedestrians. The proposed method leverages visual information from the camera to interpret 2D radar PCD, enabling spatial scene reconstruction. The effectiveness of the proposed approach is validated through experiments conducted using a radar-camera system mounted on a real vehicle. The localization performance is evaluated using a dataset collected in outdoor NLoS driving environments, demonstrating the practical applicability of the method. 

---
# MicroMix: Efficient Mixed-Precision Quantization with Microscaling Formats for Large Language Models 

**Authors**: Wenyuan Liu, Haoqian Meng, Yilun Luo, Peng Zhang, Xindian Ma  

**Link**: [PDF](https://arxiv.org/pdf/2508.02343)  

**Abstract**: Quantization significantly accelerates inference in large language models (LLMs) by replacing original high-precision matrices with low-precision counterparts. Recent advances in weight-activation quantization have primarily focused on mapping both weights and activations to the INT4 format. Although the new FP4 Tensor Cores in NVIDIA's Blackwell architecture offer up to 4x speedup over FP16, existing INT4-based kernels fail to fully exploit this capability due to mismatched data formats. To bridge this gap, we propose MicroMix, a co-designed mixed-precision quantization algorithm and matrix multiplication kernel based on Microscaling (MX) data formats. Tailored for the Blackwell architecture, the MicroMix kernel supports arbitrary combinations of MXFP4, MXFP6, and MXFP8 channels, and produces BFloat16 outputs. To achieve a favorable trade-off between accuracy and efficiency for each linear layer, we introduce quantization thresholds that identify activation elements where lower-precision formats (MXFP4 or MXFP6) incur excessive quantization error. Our algorithm selectively allocates higher-precision channels to preserve accuracy while maintaining compute efficiency. MicroMix achieves competitive or superior performance across diverse downstream tasks, including zero-shot and few-shot learning, language modeling, code generation, and mathematical reasoning. On both consumer-grade (RTX 5070Ti laptop) and server-grade (RTX 5090) GPUs, our kernel delivers at least 20% faster execution than TensorRT-FP8. Furthermore, when applied to various Llama and Qwen models, MicroMix consistently improves prefill latency and memory efficiency across a range of batch sizes compared to TensorRT baselines. Our code is available at this https URL. 

---
# VeOmni: Scaling Any Modality Model Training with Model-Centric Distributed Recipe Zoo 

**Authors**: Qianli Ma, Yaowei Zheng, Zhelun Shi, Zhongkai Zhao, Bin Jia, Ziyue Huang, Zhiqi Lin, Youjie Li, Jiacheng Yang, Yanghua Peng, Zhi Zhang, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02317)  

**Abstract**: Recent advances in large language models (LLMs) have driven impressive progress in omni-modal understanding and generation. However, training omni-modal LLMs remains a significant challenge due to the heterogeneous model architectures required to process diverse modalities, necessitating sophisticated system design for efficient large-scale training. Existing frameworks typically entangle model definition with parallel logic, incurring limited scalability and substantial engineering overhead for end-to-end omni-modal training. %
We present \veomni, a modular and efficient training framework to accelerate the development of omni-modal LLMs. \veomni introduces model-centric distributed recipes that decouples communication from computation, enabling efficient 3D parallelism on omni-modal LLMs. \veomni also features a flexible configuration interface supporting seamless integration of new modalities with minimal code change. %
Using \veomni, a omni-modal mixture-of-experts (MoE) model with 30B parameters can be trained with over 2,800 tokens/sec/GPU throughput and scale to 160K context lengths via 3D parallelism on 128 GPUs, showcasing its superior efficiency and scalability for training large omni-modal LLMs. 

---
# A Survey on Data Security in Large Language Models 

**Authors**: Kang Chen, Xiuze Zhou, Yuanguo Lin, Jinhe Su, Yuanhui Yu, Li Shen, Fan Lin  

**Link**: [PDF](https://arxiv.org/pdf/2508.02312)  

**Abstract**: Large Language Models (LLMs), now a foundation in advancing natural language processing, power applications such as text generation, machine translation, and conversational systems. Despite their transformative potential, these models inherently rely on massive amounts of training data, often collected from diverse and uncurated sources, which exposes them to serious data security risks. Harmful or malicious data can compromise model behavior, leading to issues such as toxic output, hallucinations, and vulnerabilities to threats such as prompt injection or data poisoning. As LLMs continue to be integrated into critical real-world systems, understanding and addressing these data-centric security risks is imperative to safeguard user trust and system reliability. This survey offers a comprehensive overview of the main data security risks facing LLMs and reviews current defense strategies, including adversarial training, RLHF, and data augmentation. Additionally, we categorize and analyze relevant datasets used for assessing robustness and security across different domains, providing guidance for future research. Finally, we highlight key research directions that focus on secure model updates, explainability-driven defenses, and effective governance frameworks, aiming to promote the safe and responsible development of LLM technology. This work aims to inform researchers, practitioners, and policymakers, driving progress toward data security in LLMs. 

---
# CAPO: Towards Enhancing LLM Reasoning through Verifiable Generative Credit Assignment 

**Authors**: Guofu Xie, Yunsheng Shi, Hongtao Tian, Ting Yao, Xiao Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02298)  

**Abstract**: Reinforcement Learning with Verifiable Rewards (RLVR) has improved the reasoning abilities of Large Language Models (LLMs) by using rule-based binary feedback, helping to mitigate reward hacking. However, current RLVR methods typically treat whole responses as single actions, assigning the same reward to every token. This coarse-grained feedback hampers precise credit assignment, making it hard for models to identify which reasoning steps lead to success or failure, and often results in suboptimal policies and inefficient learning. Methods like PPO provide credit assignment through value estimation, but often yield inaccurate and unverifiable signals due to limited sampling. On the other hand, methods using Process Reward Models can provide step-by-step judgments for each reasoning step, but they require high-quality process supervision labels and are time-consuming when applied in online reinforcement learning (RL). To overcome these limitations, we introduce a simple but efficient method Credit Assignment Policy Optimization (CAPO). Given a reasoning response rollout from the policy model, CAPO directly leverages an off-the-shelf, general-purpose LLM as a Generative Process Reward Model (LLM-as-GenPRM) to generate all step-wise critique by one pass, thereby providing verifiable token-level rewards to refine the tokens that were originally assigned identical rule-based rewards. This enables more fine-grained credit assignment in an effective way. Furthermore, to enhance the accuracy and robustness of CAPO, we employ voting mechanisms that scale with the number of generated critiques. Extensive experiments using different backbones like Llama and Qwen models and in different sizes show that CAPO consistently outperforms supervised learning-based and RL-based fine-tuning methods across six challenging mathematical benchmarks and three out-of-domain benchmarks. 

---
# Flexible Automatic Identification and Removal (FAIR)-Pruner: An Efficient Neural Network Pruning Method 

**Authors**: Chenqing Lin, Mostafa Hussien, Chengyao Yu, Mohamed Cheriet, Osama Abdelrahman, Ruixing Ming  

**Link**: [PDF](https://arxiv.org/pdf/2508.02291)  

**Abstract**: Neural network pruning is a critical compression technique that facilitates the deployment of large-scale neural networks on resource-constrained edge devices, typically by identifying and eliminating redundant or insignificant parameters to reduce computational and memory overhead. This paper proposes the Flexible Automatic Identification and Removal (FAIR)-Pruner, a novel method for neural network structured pruning. Specifically, FAIR-Pruner first evaluates the importance of each unit (e.g., neuron or channel) through the Utilization Score quantified by the Wasserstein distance. To reflect the performance degradation after unit removal, it then introduces the Reconstruction Error, which is computed via the Taylor expansion of the loss function. Finally, FAIR-Pruner identifies superfluous units with negligible impact on model performance by controlling the proposed Tolerance of Difference, which measures differences between unimportant units and those that cause performance degradation. A major advantage of FAIR-Pruner lies in its capacity to automatically determine the layer-wise pruning rates, which yields a more efficient subnetwork structure compared to applying a uniform pruning rate. Another advantage of the FAIR-Pruner is its great one-shot performance without post-pruning fine-tuning. Furthermore, with utilization scores and reconstruction errors, users can flexibly obtain pruned models under different pruning ratios. Comprehensive experimental validation on diverse benchmark datasets (e.g., ImageNet) and various neural network architectures (e.g., VGG) demonstrates that FAIR-Pruner achieves significant model compression while maintaining high accuracy. 

---
# Dialogue Systems Engineering: A Survey and Future Directions 

**Authors**: Mikio Nakano, Hironori Takeuchi, Sadahiro Yoshikawa, Yoichi Matsuyama, Kazunori Komatani  

**Link**: [PDF](https://arxiv.org/pdf/2508.02279)  

**Abstract**: This paper proposes to refer to the field of software engineering related to the life cycle of dialogue systems as Dialogue Systems Engineering, and surveys this field while also discussing its future directions. With the advancement of large language models, the core technologies underlying dialogue systems have significantly progressed. As a result, dialogue system technology is now expected to be applied to solving various societal issues and in business contexts. To achieve this, it is important to build, operate, and continuously improve dialogue systems correctly and efficiently. Accordingly, in addition to applying existing software engineering knowledge, it is becoming increasingly important to evolve software engineering tailored specifically to dialogue systems. In this paper, we enumerate the knowledge areas of dialogue systems engineering based on those of software engineering, as defined in the Software Engineering Body of Knowledge (SWEBOK) Version 4.0, and survey each area. Based on this survey, we identify unexplored topics in each area and discuss the future direction of dialogue systems engineering. 

---
# CellForge: Agentic Design of Virtual Cell Models 

**Authors**: Xiangru Tang, Zhuoyun Yu, Jiapeng Chen, Yan Cui, Daniel Shao, Weixu Wang, Fang Wu, Yuchen Zhuang, Wenqi Shi, Zhi Huang, Arman Cohan, Xihong Lin, Fabian Theis, Smita Krishnaswamy, Mark Gerstein  

**Link**: [PDF](https://arxiv.org/pdf/2508.02276)  

**Abstract**: Virtual cell modeling represents an emerging frontier at the intersection of artificial intelligence and biology, aiming to predict quantities such as responses to diverse perturbations quantitatively. However, autonomously building computational models for virtual cells is challenging due to the complexity of biological systems, the heterogeneity of data modalities, and the need for domain-specific expertise across multiple disciplines. Here, we introduce CellForge, an agentic system that leverages a multi-agent framework that transforms presented biological datasets and research objectives directly into optimized computational models for virtual cells. More specifically, given only raw single-cell multi-omics data and task descriptions as input, CellForge outputs both an optimized model architecture and executable code for training virtual cell models and inference. The framework integrates three core modules: Task Analysis for presented dataset characterization and relevant literature retrieval, Method Design, where specialized agents collaboratively develop optimized modeling strategies, and Experiment Execution for automated generation of code. The agents in the Design module are separated into experts with differing perspectives and a central moderator, and have to collaboratively exchange solutions until they achieve a reasonable consensus. We demonstrate CellForge's capabilities in single-cell perturbation prediction, using six diverse datasets that encompass gene knockouts, drug treatments, and cytokine stimulations across multiple modalities. CellForge consistently outperforms task-specific state-of-the-art methods. Overall, CellForge demonstrates how iterative interaction between LLM agents with differing perspectives provides better solutions than directly addressing a modeling challenge. Our code is publicly available at this https URL. 

---
# Dynaword: From One-shot to Continuously Developed Datasets 

**Authors**: Kenneth Enevoldsen, Kristian Nørgaard Jensen, Jan Kostkan, Balázs Szabó, Márton Kardos, Kirten Vad, Andrea Blasi Núñez, Gianluca Barmina, Jacob Nielsen, Rasmus Larsen, Peter Vahlstrup, Per Møldrup Dalum, Desmond Elliott, Lukas Galke, Peter Schneider-Kamp, Kristoffer Nielbo  

**Link**: [PDF](https://arxiv.org/pdf/2508.02271)  

**Abstract**: Large-scale datasets are foundational for research and development in natural language processing. However, current approaches face three key challenges: (1) reliance on ambiguously licensed sources restricting use, sharing, and derivative works; (2) static dataset releases that prevent community contributions and diminish longevity; and (3) quality assurance processes restricted to publishing teams rather than leveraging community expertise.
To address these limitations, we introduce two contributions: the Dynaword approach and Danish Dynaword. The Dynaword approach is a framework for creating large-scale, open datasets that can be continuously updated through community collaboration. Danish Dynaword is a concrete implementation that validates this approach and demonstrates its potential. Danish Dynaword contains over four times as many tokens as comparable releases, is exclusively openly licensed, and has received multiple contributions across industry and research. The repository includes light-weight tests to ensure data formatting, quality, and documentation, establishing a sustainable framework for ongoing community contributions and dataset evolution. 

---
# Decomposing the Entropy-Performance Exchange: The Missing Keys to Unlocking Effective Reinforcement Learning 

**Authors**: Jia Deng, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02260)  

**Abstract**: Recently, reinforcement learning with verifiable rewards (RLVR) has been widely used for enhancing the reasoning abilities of large language models (LLMs). A core challenge in RLVR involves managing the exchange between entropy and performance of policies. Despite the importance of this exchange, a fine-grained understanding of when and how this exchange operates most effectively remains limited. To bridge this gap, we conduct a systematic empirical analysis of the entropy-performance exchange mechanism of RLVR across different levels of granularity. Specifically, we first divide the training process into two distinct stages based on entropy dynamics, i.e., rising stage and plateau stage, and then systematically investigate how this mechanism varies across stage-level, instance-level, and token-level granularitiess. Our analysis reveals that, in the rising stage, entropy reduction in negative samples facilitates the learning of effective reasoning patterns, which in turn drives rapid performance gains. Moreover, in the plateau stage, learning efficiency strongly correlates with high-entropy tokens present in low-perplexity samples and those located at the end of sequences. Motivated by these findings, we propose two methods that dynamically adjust the reward signal using perplexity and positional information to focus RL updates on tokens that exhibit high learning potential, achieving improvements compared to the baseline methods on various LLMs. 

---
# StutterCut: Uncertainty-Guided Normalised Cut for Dysfluency Segmentation 

**Authors**: Suhita Ghosh, Melanie Jouaiti, Jan-Ole Perschewski, Sebastian Stober  

**Link**: [PDF](https://arxiv.org/pdf/2508.02255)  

**Abstract**: Detecting and segmenting dysfluencies is crucial for effective speech therapy and real-time feedback. However, most methods only classify dysfluencies at the utterance level. We introduce StutterCut, a semi-supervised framework that formulates dysfluency segmentation as a graph partitioning problem, where speech embeddings from overlapping windows are represented as graph nodes. We refine the connections between nodes using a pseudo-oracle classifier trained on weak (utterance-level) labels, with its influence controlled by an uncertainty measure from Monte Carlo dropout. Additionally, we extend the weakly labelled FluencyBank dataset by incorporating frame-level dysfluency boundaries for four dysfluency types. This provides a more realistic benchmark compared to synthetic datasets. Experiments on real and synthetic datasets show that StutterCut outperforms existing methods, achieving higher F1 scores and more precise stuttering onset detection. 

---
# ByteGen: A Tokenizer-Free Generative Model for Orderbook Events in Byte Space 

**Authors**: Yang Li, Zhi Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02247)  

**Abstract**: Generative modeling of high-frequency limit order book (LOB) dynamics is a critical yet unsolved challenge in quantitative finance, essential for robust market simulation and strategy backtesting. Existing approaches are often constrained by simplifying stochastic assumptions or, in the case of modern deep learning models like Transformers, rely on tokenization schemes that affect the high-precision, numerical nature of financial data through discretization and binning. To address these limitations, we introduce ByteGen, a novel generative model that operates directly on the raw byte streams of LOB events. Our approach treats the problem as an autoregressive next-byte prediction task, for which we design a compact and efficient 32-byte packed binary format to represent market messages without information loss. The core novelty of our work is the complete elimination of feature engineering and tokenization, enabling the model to learn market dynamics from its most fundamental representation. We achieve this by adapting the H-Net architecture, a hybrid Mamba-Transformer model that uses a dynamic chunking mechanism to discover the inherent structure of market messages without predefined rules. Our primary contributions are: 1) the first end-to-end, byte-level framework for LOB modeling; 2) an efficient packed data representation; and 3) a comprehensive evaluation on high-frequency data. Trained on over 34 million events from CME Bitcoin futures, ByteGen successfully reproduces key stylized facts of financial markets, generating realistic price distributions, heavy-tailed returns, and bursty event timing. Our findings demonstrate that learning directly from byte space is a promising and highly flexible paradigm for modeling complex financial systems, achieving competitive performance on standard market quality metrics without the biases of tokenization. 

---
# Forecasting When to Forecast: Accelerating Diffusion Models with Confidence-Gated Taylor 

**Authors**: Xiaoliu Guan, Lielin Jiang, Hanqi Chen, Xu Zhang, Jiaxing Yan, Guanzhong Wang, Yi Liu, Zetao Zhang, Yu Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02240)  

**Abstract**: Diffusion Transformers (DiTs) have demonstrated remarkable performance in visual generation tasks. However, their low inference speed limits their deployment in low-resource applications. Recent training-free approaches exploit the redundancy of features across timesteps by caching and reusing past representations to accelerate inference. Building on this idea, TaylorSeer instead uses cached features to predict future ones via Taylor expansion. However, its module-level prediction across all transformer blocks (e.g., attention or feedforward modules) requires storing fine-grained intermediate features, leading to notable memory and computation overhead. Moreover, it adopts a fixed caching schedule without considering the varying accuracy of predictions across timesteps, which can lead to degraded outputs when prediction fails. To address these limitations, we propose a novel approach to better leverage Taylor-based acceleration. First, we shift the Taylor prediction target from the module level to the last block level, significantly reducing the number of cached features. Furthermore, observing strong sequential dependencies among Transformer blocks, we propose to use the error between the Taylor-estimated and actual outputs of the first block as an indicator of prediction reliability. If the error is small, we trust the Taylor prediction for the last block; otherwise, we fall back to full computation, thereby enabling a dynamic caching mechanism. Empirical results show that our method achieves a better balance between speed and quality, achieving a 3.17x acceleration on FLUX, 2.36x on DiT, and 4.14x on Wan Video with negligible quality drop. The Project Page is \href{this https URL}{here.} 

---
# FinCPRG: A Bidirectional Generation Pipeline for Hierarchical Queries and Rich Relevance in Financial Chinese Passage Retrieval 

**Authors**: Xuan Xu, Beilin Chu, Qinhong Lin, Yixiao Zhong, Fufang Wen, Jiaqi Liu, Binjie Fei, Yu Li, Zhongliang Yang, Linna Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.02222)  

**Abstract**: In recent years, large language models (LLMs) have demonstrated significant potential in constructing passage retrieval datasets. However, existing methods still face limitations in expressing cross-doc query needs and controlling annotation quality. To address these issues, this paper proposes a bidirectional generation pipeline, which aims to generate 3-level hierarchical queries for both intra-doc and cross-doc scenarios and mine additional relevance labels on top of direct mapping annotation. The pipeline introduces two query generation methods: bottom-up from single-doc text and top-down from multi-doc titles. The bottom-up method uses LLMs to disassemble and generate structured queries at both sentence-level and passage-level simultaneously from intra-doc passages. The top-down approach incorporates three key financial elements--industry, topic, and time--to divide report titles into clusters and prompts LLMs to generate topic-level queries from each cluster. For relevance annotation, our pipeline not only relies on direct mapping annotation from the generation relationship but also implements an indirect positives mining method to enrich the relevant query-passage pairs. Using this pipeline, we constructed a Financial Passage Retrieval Generated dataset (FinCPRG) from almost 1.3k Chinese financial research reports, which includes hierarchical queries and rich relevance labels. Through evaluations of mined relevance labels, benchmarking and training experiments, we assessed the quality of FinCPRG and validated its effectiveness as a passage retrieval dataset for both training and benchmarking. 

---
# LeanK: Learnable K Cache Channel Pruning for Efficient Decoding 

**Authors**: Yike Zhang, Zhiyuan He, Huiqiang Jiang, Chengruidong Zhang, Yuqing Yang, Jianyong Wang, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02215)  

**Abstract**: Large language models (LLMs) enable long-context tasks but face efficiency challenges due to the growing key-value (KV) cache. We propose LeanK, a learning-based method that prunes unimportant key (K) cache channels by leveraging static channel sparsity. With a novel two-stage training process, LeanK learns channel-wise static mask that could satisfy specific sparsity ratio and hardware alignment requirement. LeanK reduces GPU memory and accelerates decoding without sacrificing accuracy. Experiments demonstrate up to 70% K cache and 16%-18% V cache memory reduction. Custom decoding kernel enables 1.3x speedup for attention computation. We also provide insights into model channels and attention heads during long-context inference by analyzing the learned importance distribution. Our code is available at this https URL. 

---
# Balancing Information Accuracy and Response Timeliness in Networked LLMs 

**Authors**: Yigit Turkmen, Baturalp Buyukates, Melih Bastopcu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02209)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have transformed many fields including scientific discovery, content generation, biomedical text mining, and educational technology. However, the substantial requirements for training data, computational resources, and energy consumption pose significant challenges for their practical deployment. A promising alternative is to leverage smaller, specialized language models and aggregate their outputs to improve overall response quality. In this work, we investigate a networked LLM system composed of multiple users, a central task processor, and clusters of topic-specialized LLMs. Each user submits categorical binary (true/false) queries, which are routed by the task processor to a selected cluster of $m$ LLMs. After gathering individual responses, the processor returns a final aggregated answer to the user. We characterize both the information accuracy and response timeliness in this setting, and formulate a joint optimization problem to balance these two competing objectives. Our extensive simulations demonstrate that the aggregated responses consistently achieve higher accuracy than those of individual LLMs. Notably, this improvement is more significant when the participating LLMs exhibit similar standalone performance. 

---
# Proof2Hybrid: Automatic Mathematical Benchmark Synthesis for Proof-Centric Problems 

**Authors**: Yebo Peng, Zixiang Liu, Yaoming Li, Zhizhuo Yang, Xinye Xu, Bowen Ye, Weijun Yuan, Zihan Wang, Tong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02208)  

**Abstract**: Evaluating the mathematical capability of Large Language Models (LLMs) is a critical yet challenging frontier. Existing benchmarks fall short, particularly for proof-centric problems, as manual creation is unscalable and costly, leaving the true mathematical abilities of LLMs largely unassessed. To overcome these barriers, we propose Proof2Hybrid, the first fully automated framework that synthesizes high-quality, proof-centric benchmarks from natural language mathematical corpora. The key novelty of our solution is Proof2X, a roadmap of converting mathematical proofs into various kinds of questions that are easy to verify. Instructed by this roadmap, we propose a new type of hybrid-formatted questions, named ``$m$-out-of-$n$ multiple judge questions'', specifically designed to enable robust, automatic evaluation while being resilient to guessing and superficial pattern matching inherent in traditional formats. As a demonstration of our framework, we introduce AlgGeoTest, a benchmark for algebraic geometry--a frontier domain of modern mathematics--comprising 456 challenging items. Our extensive evaluations on state-of-the-art LLMs using AlgGeoTest reveal profound deficits in their comprehension of algebraic geometry, providing a more precise measure of their true mathematical capabilities. Our framework and benchmark pave the way for a new wave of in-depth research into the mathematical intelligence of AI systems. 

---
# FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation 

**Authors**: Cui Miao, Tao Chang, Meihan Wu, Hongbin Xu, Chun Li, Ming Li, Xiaodong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02190)  

**Abstract**: Vision-language-action (VLA) models have significantly advanced robotic manipulation by enabling robots to interpret language instructions for task execution. However, training these models often relies on large-scale user-specific data, raising concerns about privacy and security, which in turn limits their broader adoption. To address this, we propose FedVLA, the first federated VLA learning framework, enabling distributed model training that preserves data privacy without compromising performance. Our framework integrates task-aware representation learning, adaptive expert selection, and expert-driven federated aggregation, enabling efficient and privacy-preserving training of VLA models. Specifically, we introduce an Instruction Oriented Scene-Parsing mechanism, which decomposes and enhances object-level features based on task instructions, improving contextual understanding. To effectively learn diverse task patterns, we design a Dual Gating Mixture-of-Experts (DGMoE) mechanism, where not only input tokens but also self-aware experts adaptively decide their activation. Finally, we propose an Expert-Driven Aggregation strategy at the federated server, where model aggregation is guided by activated experts, ensuring effective cross-client knowledge this http URL simulations and real-world robotic experiments demonstrate the effectiveness of our proposals. Notably, DGMoE significantly improves computational efficiency compared to its vanilla counterpart, while FedVLA achieves task success rates comparable to centralized training, effectively preserving data privacy. 

---
# Learning Dynamics of Meta-Learning in Small Model Pretraining 

**Authors**: David Demitri Africa, Yuval Weiss, Paula Buttery, Richard Diehl Martinez  

**Link**: [PDF](https://arxiv.org/pdf/2508.02189)  

**Abstract**: Large language models are powerful but costly. We ask whether meta-learning can make the pretraining of small language models not only better but also more interpretable. We integrate first-order MAML with subset-masked LM pretraining, producing four LLama-style decoder-only models (11M-570M params), and evaluate it on a fundamental NLP task with many settings and real-world applications. Compared with vanilla training, our model (i) reaches the same loss up to 1.6x sooner, (ii) improves F1 on multilingual Universal NER under equal compute, and (iii) makes the training dynamics easy to read: first the network's representations fan out ("diversify") and later they collapse into a smaller, shared subspace ("compress"). This two-stage shift shows up as a rise-and-fall in both effective-rank curves and attention-head entropy. The same curves pinpoint which layers specialise earliest and which later reconverge, giving a compact, interpretable signature of meta-adaptation. Code, checkpoints and WandB logs are released. 

---
# GaussianCross: Cross-modal Self-supervised 3D Representation Learning via Gaussian Splatting 

**Authors**: Lei Yao, Yi Wang, Yi Zhang, Moyun Liu, Lap-Pui Chau  

**Link**: [PDF](https://arxiv.org/pdf/2508.02172)  

**Abstract**: The significance of informative and robust point representations has been widely acknowledged for 3D scene understanding. Despite existing self-supervised pre-training counterparts demonstrating promising performance, the model collapse and structural information deficiency remain prevalent due to insufficient point discrimination difficulty, yielding unreliable expressions and suboptimal performance. In this paper, we present GaussianCross, a novel cross-modal self-supervised 3D representation learning architecture integrating feed-forward 3D Gaussian Splatting (3DGS) techniques to address current challenges. GaussianCross seamlessly converts scale-inconsistent 3D point clouds into a unified cuboid-normalized Gaussian representation without missing details, enabling stable and generalizable pre-training. Subsequently, a tri-attribute adaptive distillation splatting module is incorporated to construct a 3D feature field, facilitating synergetic feature capturing of appearance, geometry, and semantic cues to maintain cross-modal consistency. To validate GaussianCross, we perform extensive evaluations on various benchmarks, including ScanNet, ScanNet200, and S3DIS. In particular, GaussianCross shows a prominent parameter and data efficiency, achieving superior performance through linear probing (<0.1% parameters) and limited data training (1% of scenes) compared to state-of-the-art methods. Furthermore, GaussianCross demonstrates strong generalization capabilities, improving the full fine-tuning accuracy by 9.3% mIoU and 6.1% AP$_{50}$ on ScanNet200 semantic and instance segmentation tasks, respectively, supporting the effectiveness of our approach. The code, weights, and visualizations are publicly available at \href{this https URL}{this https URL}. 

---
# DreamPainter: Image Background Inpainting for E-commerce Scenarios 

**Authors**: Sijie Zhao, Jing Cheng, Yaoyao Wu, Hao Xu, Shaohui Jiao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02155)  

**Abstract**: Although diffusion-based image genenation has been widely explored and applied, background generation tasks in e-commerce scenarios still face significant challenges. The first challenge is to ensure that the generated products are consistent with the given product inputs while maintaining a reasonable spatial arrangement, harmonious shadows, and reflections between foreground products and backgrounds. Existing inpainting methods fail to address this due to the lack of domain-specific data. The second challenge involves the limitation of relying solely on text prompts for image control, as effective integrating visual information to achieve precise control in inpainting tasks remains underexplored. To address these challenges, we introduce DreamEcom-400K, a high-quality e-commerce dataset containing accurate product instance masks, background reference images, text prompts, and aesthetically pleasing product images. Based on this dataset, we propose DreamPainter, a novel framework that not only utilizes text prompts for control but also flexibly incorporates reference image information as an additional control signal. Extensive experiments demonstrate that our approach significantly outperforms state-of-the-art methods, maintaining high product consistency while effectively integrating both text prompt and reference image information. 

---
# Large-Scale Model Enabled Semantic Communication Based on Robust Knowledge Distillation 

**Authors**: Kuiyuan DIng, Caili Guo, Yang Yang, Zhongtian Du, Walid Saad  

**Link**: [PDF](https://arxiv.org/pdf/2508.02148)  

**Abstract**: Large-scale models (LSMs) can be an effective framework for semantic representation and understanding, thereby providing a suitable tool for designing semantic communication (SC) systems. However, their direct deployment is often hindered by high computational complexity and resource requirements. In this paper, a novel robust knowledge distillation based semantic communication (RKD-SC) framework is proposed to enable efficient and \textcolor{black}{channel-noise-robust} LSM-powered SC. The framework addresses two key challenges: determining optimal compact model architectures and effectively transferring knowledge while maintaining robustness against channel noise. First, a knowledge distillation-based lightweight differentiable architecture search (KDL-DARTS) algorithm is proposed. This algorithm integrates knowledge distillation loss and a complexity penalty into the neural architecture search process to identify high-performance, lightweight semantic encoder architectures. Second, a novel two-stage robust knowledge distillation (RKD) algorithm is developed to transfer semantic capabilities from an LSM (teacher) to a compact encoder (student) and subsequently enhance system robustness. To further improve resilience to channel impairments, a channel-aware transformer (CAT) block is introduced as the channel codec, trained under diverse channel conditions with variable-length outputs. Extensive simulations on image classification tasks demonstrate that the RKD-SC framework significantly reduces model parameters while preserving a high degree of the teacher model's performance and exhibiting superior robustness compared to existing methods. 

---
# Fitness aligned structural modeling enables scalable virtual screening with AuroBind 

**Authors**: Zhongyue Zhang, Jiahua Rao, Jie Zhong, Weiqiang Bai, Dongxue Wang, Shaobo Ning, Lifeng Qiao, Sheng Xu, Runze Ma, Will Hua, Jack Xiaoyu Chen, Odin Zhang, Wei Lu, Hanyi Feng, He Yang, Xinchao Shi, Rui Li, Wanli Ouyang, Xinzhu Ma, Jiahao Wang, Jixian Zhang, Jia Duan, Siqi Sun, Jian Zhang, Shuangjia Zheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.02137)  

**Abstract**: Most human proteins remain undrugged, over 96% of human proteins remain unexploited by approved therapeutics. While structure-based virtual screening promises to expand the druggable proteome, existing methods lack atomic-level precision and fail to predict binding fitness, limiting translational impact. We present AuroBind, a scalable virtual screening framework that fine-tunes a custom atomic-level structural model on million-scale chemogenomic data. AuroBind integrates direct preference optimization, self-distillation from high-confidence complexes, and a teacher-student acceleration strategy to jointly predict ligand-bound structures and binding fitness. The proposed models outperform state-of-the-art models on structural and functional benchmarks while enabling 100,000-fold faster screening across ultra-large compound libraries. In a prospective screen across ten disease-relevant targets, AuroBind achieved experimental hit rates of 7-69%, with top compounds reaching sub-nanomolar to picomolar potency. For the orphan GPCRs GPR151 and GPR160, AuroBind identified both agonists and antagonists with success rates of 16-30%, and functional assays confirmed GPR160 modulation in liver and prostate cancer models. AuroBind offers a generalizable framework for structure-function learning and high-throughput molecular screening, bridging the gap between structure prediction and therapeutic discovery. 

---
# The Complexity of Extreme Climate Events on the New Zealand's Kiwifruit Industry 

**Authors**: Boyuan Zheng, Victor W. Chu, Zhidong Li, Evan Webster, Ashley Rootsey  

**Link**: [PDF](https://arxiv.org/pdf/2508.02130)  

**Abstract**: Climate change has intensified the frequency and severity of extreme weather events, presenting unprecedented challenges to the agricultural industry worldwide. In this investigation, we focus on kiwifruit farming in New Zealand. We propose to examine the impacts of climate-induced extreme events, specifically frost, drought, extreme rainfall, and heatwave, on kiwifruit harvest yields. These four events were selected due to their significant impacts on crop productivity and their prevalence as recorded by climate monitoring institutions in the country. We employed Isolation Forest, an unsupervised anomaly detection method, to analyse climate history and recorded extreme events, alongside with kiwifruit yields. Our analysis reveals considerable variability in how different types of extreme event affect kiwifruit yields underscoring notable discrepancies between climatic extremes and individual farm's yield outcomes. Additionally, our study highlights critical limitations of current anomaly detection approaches, particularly in accurately identifying events such as frost. These findings emphasise the need for integrating supplementary features like farm management strategies with climate adaptation practices. Our further investigation will employ ensemble methods that consolidate nearby farms' yield data and regional climate station features to reduce variance, thereby enhancing the accuracy and reliability of extreme event detection and the formulation of response strategies. 

---
# Amber Pruner: Leveraging N:M Activation Sparsity for Efficient Prefill in Large Language Models 

**Authors**: Tai An, Ruwu Cai, Yanzhe Zhang, Yang Liu, Hao Chen, Pengcheng Xie, Sheng Chang, Yiwu Yao, Gongyi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02128)  

**Abstract**: In the era of large language models (LLMs), N:M sparsity has emerged as a structured compression technique critical for accelerating inference. While prior work has primarily focused on weight sparsity, it often suffers from significant accuracy degradation. Activation sparsity, though promising, is typically training-dependent and faces challenges in generalization. To address these limitations, we introduce Amber Pruner, a training-free N:M activation sparsity method designed specifically for the prefill stage, targeting the acceleration of linear projection layers in LLMs. Extensive experiments across multiple models and sparsity ratios (2:4, 4:8, and 8:16) demonstrate that Amber Pruner can effectively sparsify and accelerate more than 55% of linear computations without requiring model retraining. To further enhance generality and efficiency, we propose Outstanding-sparse, a unified framework that integrates Amber Pruner with post-training W8A8 quantization. Our approach preserves strong performance across a range of downstream tasks, with notable advantages in generative tasks. This work pioneers a new frontier in activation sparsity, providing foundational insights that are poised to guide the co-evolution of algorithms and architectures in the design of next-generation AI systems. 

---
# Coward: Toward Practical Proactive Federated Backdoor Defense via Collision-based Watermark 

**Authors**: Wenjie Li, Siying Gu, Yiming Li, Kangjie Chen, Zhili Chen, Tianwei Zhang, Shu-Tao Xia, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02115)  

**Abstract**: Backdoor detection is currently the mainstream defense against backdoor attacks in federated learning (FL), where malicious clients upload poisoned updates that compromise the global model and undermine the reliability of FL deployments. Existing backdoor detection techniques fall into two categories, including passive and proactive ones, depending on whether the server proactively modifies the global model. However, both have inherent limitations in practice: passive defenses are vulnerable to common non-i.i.d. data distributions and random participation of FL clients, whereas current proactive defenses suffer inevitable out-of-distribution (OOD) bias because they rely on backdoor co-existence effects. To address these issues, we introduce a new proactive defense, dubbed Coward, inspired by our discovery of multi-backdoor collision effects, in which consecutively planted, distinct backdoors significantly suppress earlier ones. In general, we detect attackers by evaluating whether the server-injected, conflicting global watermark is erased during local training rather than retained. Our method preserves the advantages of proactive defenses in handling data heterogeneity (\ie, non-i.i.d. data) while mitigating the adverse impact of OOD bias through a revised detection mechanism. Extensive experiments on benchmark datasets confirm the effectiveness of Coward and its resilience to potential adaptive attacks. The code for our method would be available at this https URL. 

---
# Evaluating User Experience in Conversational Recommender Systems: A Systematic Review Across Classical and LLM-Powered Approaches 

**Authors**: Raj Mahmud, Yufeng Wu, Abdullah Bin Sawad, Shlomo Berkovsky, Mukesh Prasad, A. Baki Kocaballi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02096)  

**Abstract**: Conversational Recommender Systems (CRSs) are receiving growing research attention across domains, yet their user experience (UX) evaluation remains limited. Existing reviews largely overlook empirical UX studies, particularly in adaptive and large language model (LLM)-based CRSs. To address this gap, we conducted a systematic review following PRISMA guidelines, synthesising 23 empirical studies published between 2017 and 2025. We analysed how UX has been conceptualised, measured, and shaped by domain, adaptivity, and LLM.
Our findings reveal persistent limitations: post hoc surveys dominate, turn-level affective UX constructs are rarely assessed, and adaptive behaviours are seldom linked to UX outcomes. LLM-based CRSs introduce further challenges, including epistemic opacity and verbosity, yet evaluations infrequently address these issues. We contribute a structured synthesis of UX metrics, a comparative analysis of adaptive and nonadaptive systems, and a forward-looking agenda for LLM-aware UX evaluation. These findings support the development of more transparent, engaging, and user-centred CRS evaluation practices. 

---
# VLM4D: Towards Spatiotemporal Awareness in Vision Language Models 

**Authors**: Shijie Zhou, Alexander Vilesov, Xuehai He, Ziyu Wan, Shuwang Zhang, Aditya Nagachandra, Di Chang, Dongdong Chen, Xin Eric Wang, Achuta Kadambi  

**Link**: [PDF](https://arxiv.org/pdf/2508.02095)  

**Abstract**: Vision language models (VLMs) have shown remarkable capabilities in integrating linguistic and visual reasoning but remain fundamentally limited in understanding dynamic spatiotemporal interactions. Humans effortlessly track and reason about object movements, rotations, and perspective shifts-abilities essential for robust dynamic real-world understanding yet notably lacking in current VLMs. In this paper, we introduce VLM4D, the first benchmark specifically designed to evaluate the spatiotemporal reasoning capabilities of VLMs. Our benchmark comprises diverse real-world and synthetic videos accompanied by carefully curated question-answer pairs emphasizing translational and rotational motions, perspective awareness, and motion continuity. Through comprehensive evaluations of state-of-the-art open and closed-source VLMs, we identify significant performance gaps compared to human baselines, highlighting fundamental deficiencies in existing models. Extensive analysis reveals that VLMs struggle particularly with integrating multiple visual cues and maintaining temporal coherence. We further explore promising directions, such as leveraging 4D feature field reconstruction and targeted spatiotemporal supervised fine-tuning, demonstrating their effectiveness in enhancing spatiotemporal comprehension. Our work aims to encourage deeper exploration into improving VLMs' spatial and temporal grounding, paving the way towards more capable and reliable visual intelligence for dynamic environments. 

---
# FPEdit: Robust LLM Fingerprinting through Localized Knowledge Editing 

**Authors**: Shida Wang, Chaohu Liu, Yubo Wang, Linli Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02092)  

**Abstract**: Large language models represent significant investments in computation, data, and engineering expertise, making them extraordinarily valuable intellectual assets. Nevertheless, these AI assets remain vulnerable to unauthorized redistribution and commercial exploitation through fine-tuning or black-box deployment. Current fingerprinting approaches face a fundamental trade-off: intrinsic methods require full parameter access, while backdoor-based techniques employ statistically anomalous triggers easily detected and filtered by adversaries. To address these limitations, we introduce FPEdit, a novel knowledge-editing framework that injects semantically coherent natural language fingerprints by modifying a sparse subset of model weights. This ensures stealthy and precise ownership encoding without degrading the core functionality. Extensive experiments show that FPEdit achieves $95$-$100\%$ fingerprint retention under both full-parameter fine-tuning and parameter-efficient adaptation, while preserving performance on 24 downstream benchmarks. Moreover, FPEdit remains robust under quantization, pruning, and stochastic decoding, and can embed 10 fingerprint pairs into LLaMA2-7B in under 10 minutes using less than 32 GB of GPU memory, a $70\%$ reduction in resource requirements compared to existing techniques. These advances establish FPEdit as the first fingerprinting approach to simultaneously achieve robustness against adaptation, resistance to detection, and preservation of model utility, providing a minimally invasive solution for reliable provenance verification of large language models in adversarial deployment scenarios. 

---
# CRINN: Contrastive Reinforcement Learning for Approximate Nearest Neighbor Search 

**Authors**: Xiaoya Li, Xiaofei Sun, Albert Wang, Chris Shum, Jiwei Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.02091)  

**Abstract**: Approximate nearest-neighbor search (ANNS) algorithms have become increasingly critical for recent AI applications, particularly in retrieval-augmented generation (RAG) and agent-based LLM applications. In this paper, we present CRINN, a new paradigm for ANNS algorithms. CRINN treats ANNS optimization as a reinforcement learning problem where execution speed serves as the reward signal. This approach enables the automatic generation of progressively faster ANNS implementations while maintaining accuracy constraints. Our experimental evaluation demonstrates CRINN's effectiveness across six widely-used NNS benchmark datasets. When compared against state-of-the-art open-source ANNS algorithms, CRINN achieves best performance on three of them (GIST-960-Euclidean, MNIST-784-Euclidean, and GloVe-25-angular), and tied for first place on two of them (SIFT-128-Euclidean and GloVe-25-angular). The implications of CRINN's success reach well beyond ANNS optimization: It validates that LLMs augmented with reinforcement learning can function as an effective tool for automating sophisticated algorithmic optimizations that demand specialized knowledge and labor-intensive manual this http URL can be found at this https URL 

---
# SSBD Ontology: A Two-Tier Approach for Interoperable Bioimaging Metadata 

**Authors**: Yuki Yamagata, Koji Kyoda, Hiroya Itoga, Emi Fujisawa, Shuichi Onami  

**Link**: [PDF](https://arxiv.org/pdf/2508.02084)  

**Abstract**: Advanced bioimaging technologies have enabled the large-scale acquisition of multidimensional data, yet effective metadata management and interoperability remain significant challenges. To address these issues, we propose a new ontology-driven framework for the Systems Science of Biological Dynamics Database (SSBD) that adopts a two-tier architecture. The core layer provides a class-centric structure referencing existing biomedical ontologies, supporting both SSBD:repository -- which focuses on rapid dataset publication with minimal metadata -- and SSBD:database, which is enhanced with biological and imaging-related annotations. Meanwhile, the instance layer represents actual imaging dataset information as Resource Description Framework individuals that are explicitly linked to the core classes. This layered approach aligns flexible instance data with robust ontological classes, enabling seamless integration and advanced semantic queries. By coupling flexibility with rigor, the SSBD Ontology promotes interoperability, data reuse, and the discovery of novel biological mechanisms. Moreover, our solution aligns with the Recommended Metadata for Biological Images guidelines and fosters compatibility. Ultimately, our approach contributes to establishing a Findable, Accessible, Interoperable, and Reusable data ecosystem within the bioimaging community. 

---
# AlignGuard-LoRA: Alignment-Preserving Fine-Tuning via Fisher-Guided Decomposition and Riemannian-Geodesic Collision Regularization 

**Authors**: Amitava Das, Abhilekh Borah, Vinija Jain, Aman Chadha  

**Link**: [PDF](https://arxiv.org/pdf/2508.02079)  

**Abstract**: Low-rank adaptation (LoRA) has become a standard tool for efficiently fine-tuning large language models (LLMs). Yet, even minor LoRA updates can induce alignment drift, weakening safety and behavioral constraints through entangled parameter changes. To address this, we propose AlignGuard-LoRA (AGL), a principled framework for preserving alignment during finetuning. AGL introduces several key components: a primary task loss for supervision, Fisher Information Matrix-based regularization to restrict updates in alignment-sensitive subspaces, and task-specific regularization to stabilize the integration of new knowledge. We further introduce collision-aware regularization, blending Riemannian overlap -- which penalizes coordinate-wise interference -- and geodesic separation -- which encourages disjoint update geometry. We curate DriftCaps, a targeted diagnostic benchmark of safe and unsafe prompts designed to quantify alignment drift and safety degradation. Empirical evaluations show that AGL mitigates alignment drift by up to 50% on safety-critical benchmarks without degrading downstream task performance. Comprehensive ablation confirms that each component contributes distinctly to preserving latent safety behaviors. Finally, we derive and validate a scaling law for catastrophic forgetting, revealing that AGL flattens post-finetuning loss escalation while preserving adaptation dynamics. AGL is a structurally grounded refinement of LoRA, ensuring alignment preservation with minimal trade-offs. To encourage further exploration and development, we open-source our implementation. 

---
# SpikeSTAG: Spatial-Temporal Forecasting via GNN-SNN Collaboration 

**Authors**: Bang Hu, Changze Lv, Mingjie Li, Yunpeng Liu, Xiaoqing Zheng, Fengzhe Zhang, Wei cao, Fan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02069)  

**Abstract**: Spiking neural networks (SNNs), inspired by the spiking behavior of biological neurons, offer a distinctive approach for capturing the complexities of temporal data. However, their potential for spatial modeling in multivariate time-series forecasting remains largely unexplored. To bridge this gap, we introduce a brand new SNN architecture, which is among the first to seamlessly integrate graph structural learning with spike-based temporal processing for multivariate time-series forecasting. Specifically, we first embed time features and an adaptive matrix, eliminating the need for predefined graph structures. We then further learn sequence features through the Observation (OBS) Block. Building upon this, our Multi-Scale Spike Aggregation (MSSA) hierarchically aggregates neighborhood information through spiking SAGE layers, enabling multi-hop feature extraction while eliminating the need for floating-point operations. Finally, we propose a Dual-Path Spike Fusion (DSF) Block to integrate spatial graph features and temporal dynamics via a spike-gated mechanism, combining LSTM-processed sequences with spiking self-attention outputs, effectively improve the model accuracy of long sequence datasets. Experiments show that our model surpasses the state-of-the-art SNN-based iSpikformer on all datasets and outperforms traditional temporal models at long horizons, thereby establishing a new paradigm for efficient spatial-temporal modeling. 

---
# MolReasoner: Toward Effective and Interpretable Reasoning for Molecular LLMs 

**Authors**: Guojiang Zhao, Sihang Li, Zixiang Lu, Zheng Cheng, Haitao Lin, Lirong Wu, Hanchen Xia, Hengxing Cai, Wentao Guo, Hongshuai Wang, Mingjun Xu, Siyu Zhu, Guolin Ke, Linfeng Zhang, Zhifeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2508.02066)  

**Abstract**: Large Language Models(LLMs) have demonstrated remarkable performance across various domains, yet their capabilities in molecular reasoning remain insufficiently explored. Current approaches tend to rely heavily on general-purpose prompting, which lacks domain-specific molecular semantics, while those that use fine-tuning strategies often face challenges with interpretability and reasoning depth. To address these issues, we introduce MolReasoner, a two-stage framework designed to transition LLMs from memorization towards chemical reasoning. First, we propose Mol-SFT, which initializes the model's reasoning abilities via synthetic Chain-of-Thought(CoT) samples generated by GPT-4o and verified for chemical accuracy. Subsequently, Mol-RL applies reinforcement learning with specialized reward functions designed explicitly to align chemical structures with linguistic descriptions, thereby enhancing molecular reasoning capabilities. Our approach notably enhances interpretability, improving the model 's molecular understanding and enabling better generalization. Extensive experiments demonstrate that MolReasoner outperforms existing methods, and marking a significant shift from memorization-based outputs to robust chemical reasoning. 

---
# RICL: Adding In-Context Adaptability to Pre-Trained Vision-Language-Action Models 

**Authors**: Kaustubh Sridhar, Souradeep Dutta, Dinesh Jayaraman, Insup Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.02062)  

**Abstract**: Multi-task ``vision-language-action'' (VLA) models have recently demonstrated increasing promise as generalist foundation models for robotics, achieving non-trivial performance out of the box on new tasks in new environments. However, for such models to be truly useful, an end user must have easy means to teach them to improve. For language and vision models, the emergent ability to perform in-context learning (ICL) has proven to be a versatile and highly useful interface to easily teach new tasks with no parameter finetuning. Unfortunately, VLAs pre-trained with imitation learning objectives do not naturally acquire ICL abilities. In this paper, we demonstrate that, with the right finetuning recipe and a small robot demonstration dataset, it is possible to inject in-context adaptability post hoc into such a VLA. After retraining for in-context learning (RICL), our system permits an end user to provide a small number (10-20) of demonstrations for a new task. RICL then fetches the most relevant portions of those demonstrations into the VLA context to exploit ICL, performing the new task and boosting task performance. We apply RICL to inject ICL into the $\pi_{0}$-FAST VLA, and show that it permits large in-context improvements for a variety of new manipulation tasks with only 20 demonstrations per task, without any parameter updates. When parameter updates on the target task demonstrations is possible, RICL finetuning further boosts performance. We release code and model weights for RICL-$\pi_{0}$-FAST alongside the paper to enable, for the first time, a simple in-context learning interface for new manipulation tasks. Website: this https URL. 

---
# Enhancement of Quantum Semi-Supervised Learning via Improved Laplacian and Poisson Methods 

**Authors**: Hamed Gholipour, Farid Bozorgnia, Hamzeh Mohammadigheymasi, Kailash Hambarde, Javier Mancilla, Hugo Proenca, Joao Neves, Moharram Challenger  

**Link**: [PDF](https://arxiv.org/pdf/2508.02054)  

**Abstract**: This paper develops a hybrid quantum approach for graph-based semi-supervised learning to enhance performance in scenarios where labeled data is scarce. We introduce two enhanced quantum models, the Improved Laplacian Quantum Semi-Supervised Learning (ILQSSL) and the Improved Poisson Quantum Semi-Supervised Learning (IPQSSL), that incorporate advanced label propagation strategies within variational quantum circuits. These models utilize QR decomposition to embed graph structure directly into quantum states, thereby enabling more effective learning in low-label settings. We validate our methods across four benchmark datasets like Iris, Wine, Heart Disease, and German Credit Card -- and show that both ILQSSL and IPQSSL consistently outperform leading classical semi-supervised learning algorithms, particularly under limited supervision. Beyond standard performance metrics, we examine the effect of circuit depth and qubit count on learning quality by analyzing entanglement entropy and Randomized Benchmarking (RB). Our results suggest that while some level of entanglement improves the model's ability to generalize, increased circuit complexity may introduce noise that undermines performance on current quantum hardware. Overall, the study highlights the potential of quantum-enhanced models for semi-supervised learning, offering practical insights into how quantum circuits can be designed to balance expressivity and stability. These findings support the role of quantum machine learning in advancing data-efficient classification, especially in applications constrained by label availability and hardware limitations. 

---
# Epi$^2$-Net: Advancing Epidemic Dynamics Forecasting with Physics-Inspired Neural Networks 

**Authors**: Rui Sun, Chenghua Gong, Tianjun Gu, Yuhao Zheng, Jie Ding, Juyuan Zhang, Liming Pan, Linyuan Lü  

**Link**: [PDF](https://arxiv.org/pdf/2508.02049)  

**Abstract**: Advancing epidemic dynamics forecasting is vital for targeted interventions and safeguarding public health. Current approaches mainly fall into two categories: mechanism-based and data-driven models. Mechanism-based models are constrained by predefined compartmental structures and oversimplified system assumptions, limiting their ability to model complex real-world dynamics, while data-driven models focus solely on intrinsic data dependencies without physical or epidemiological constraints, risking biased or misleading representations. Although recent studies have attempted to integrate epidemiological knowledge into neural architectures, most of them fail to reconcile explicit physical priors with neural representations. To overcome these obstacles, we introduce Epi$^2$-Net, a Epidemic Forecasting Framework built upon Physics-Inspired Neural Networks. Specifically, we propose reconceptualizing epidemic transmission from the physical transport perspective, introducing the concept of neural epidemic transport. Further, we present a physic-inspired deep learning framework, and integrate physical constraints with neural modules to model spatio-temporal patterns of epidemic dynamics. Experiments on real-world datasets have demonstrated that Epi$^2$-Net outperforms state-of-the-art methods in epidemic forecasting, providing a promising solution for future epidemic containment. The code is available at: this https URL. 

---
# Graph Unlearning via Embedding Reconstruction -- A Range-Null Space Decomposition Approach 

**Authors**: Hang Yin, Zipeng Liu, Xiaoyong Peng, Liyao Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.02044)  

**Abstract**: Graph unlearning is tailored for GNNs to handle widespread and various graph structure unlearning requests, which remain largely unexplored. The GIF (graph influence function) achieves validity under partial edge unlearning, but faces challenges in dealing with more disturbing node unlearning. To avoid the overhead of retraining and realize the model utility of unlearning, we proposed a novel node unlearning method to reverse the process of aggregation in GNN by embedding reconstruction and to adopt Range-Null Space Decomposition for the nodes' interaction learning. Experimental results on multiple representative datasets demonstrate the SOTA performance of our proposed approach. 

---
# Diagnosing Memorization in Chain-of-Thought Reasoning, One Token at a Time 

**Authors**: Huihan Li, You Chen, Siyuan Wang, Yixin He, Ninareh Mehrabi, Rahul Gupta, Xiang Ren  

**Link**: [PDF](https://arxiv.org/pdf/2508.02037)  

**Abstract**: Large Language Models (LLMs) perform well on reasoning benchmarks but often fail when inputs alter slightly, raising concerns about the extent to which their success relies on memorization. This issue is especially acute in Chain-of-Thought (CoT) reasoning, where spurious memorized patterns can trigger intermediate errors that cascade into incorrect final answers. We introduce STIM, a novel framework for Source-aware Token-level Identification of Memorization, which attributes each token in a reasoning chain to one of multiple memorization sources - local, mid-range, or long-range - based on their statistical co-occurrence with the token in the pretraining corpus. Our token-level analysis across tasks and distributional settings reveals that models rely more on memorization in complex or long-tail cases, and that local memorization is often the dominant driver of errors, leading to up to 67% of wrong tokens. We also show that memorization scores from STIM can be effective in predicting the wrong tokens in the wrong reasoning step. STIM offers a powerful tool for diagnosing and improving model reasoning and can generalize to other structured step-wise generation tasks. 

---
# Confidence-Diversity Calibration of AI Judgement Enables Reliable Qualitative Coding 

**Authors**: Zhilong Zhao, Yindi Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.02029)  

**Abstract**: LLMs enable qualitative coding at large scale, but assessing the reliability of their output remains challenging in domains where human experts seldom agree. Analysing 5,680 coding decisions from eight state-of-the-art LLMs across ten thematic categories, we confirm that a model's mean self-confidence already tracks inter-model agreement closely (Pearson r=0.82). Adding model diversity-quantified as the normalised Shannon entropy of the panel's votes-turns this single cue into a dual signal that explains agreement almost completely (R^2=0.979). The confidence-diversity duo enables a three-tier workflow that auto-accepts 35% of segments with <5% audit-detected error and routes the remainder for targeted human review, cutting manual effort by up to 65%. Cross-domain replication on six public datasets spanning finance, medicine, law and multilingual tasks confirms these gains (kappa improvements of 0.20-0.78). Our results establish a generalisable, evidence-based criterion for calibrating AI judgement in qualitative research. 

---
# SpeechR: A Benchmark for Speech Reasoning in Large Audio-Language Models 

**Authors**: Wanqi Yang, Yanda Li, Yunchao Wei, Meng Fang, Ling Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.02018)  

**Abstract**: Large audio-language models (LALMs) have achieved near-human performance in sentence-level transcription and emotion recognition. However, existing evaluations focus mainly on surface-level perception, leaving the capacity of models for contextual and inference-driven reasoning in speech-based scenarios insufficiently examined. To address this gap, we introduce SpeechR, a unified benchmark for evaluating reasoning over speech in large audio-language models. SpeechR evaluates models along three key dimensions: factual retrieval, procedural inference, and normative judgment. It includes three distinct evaluation formats. The multiple-choice version measures answer selection accuracy. The generative version assesses the coherence and logical consistency of reasoning chains. The acoustic-feature version investigates whether variations in stress and emotion affect reasoning performance. Evaluations on eleven state-of-the-art LALMs reveal that high transcription accuracy does not translate into strong reasoning capabilities. SpeechR establishes a structured benchmark for evaluating reasoning in spoken language, enabling more targeted analysis of model capabilities across diverse dialogue-based tasks. 

---
# DIRF: A Framework for Digital Identity Protection and Clone Governance in Agentic AI Systems 

**Authors**: Hammad Atta, Muhammad Zeeshan Baig, Yasir Mehmood, Nadeem Shahzad, Ken Huang, Muhammad Aziz Ul Haq, Muhammad Awais, Kamal Ahmed, Anthony Green  

**Link**: [PDF](https://arxiv.org/pdf/2508.01997)  

**Abstract**: The rapid advancement and widespread adoption of generative artificial intelligence (AI) pose significant threats to the integrity of personal identity, including digital cloning, sophisticated impersonation, and the unauthorized monetization of identity-related data. Mitigating these risks necessitates the development of robust AI-generated content detection systems, enhanced legal frameworks, and ethical guidelines. This paper introduces the Digital Identity Rights Framework (DIRF), a structured security and governance model designed to protect behavioral, biometric, and personality-based digital likeness attributes to address this critical need. Structured across nine domains and 63 controls, DIRF integrates legal, technical, and hybrid enforcement mechanisms to secure digital identity consent, traceability, and monetization. We present the architectural foundations, enforcement strategies, and key use cases supporting the need for a unified framework. This work aims to inform platform builders, legal entities, and regulators about the essential controls needed to enforce identity rights in AI-driven systems. 

---
# Controllable and Stealthy Shilling Attacks via Dispersive Latent Diffusion 

**Authors**: Shutong Qiao, Wei Yuan, Junliang Yu, Tong Chen, Quoc Viet Hung Nguyen, Hongzhi Yin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01987)  

**Abstract**: Recommender systems (RSs) are now fundamental to various online platforms, but their dependence on user-contributed data leaves them vulnerable to shilling attacks that can manipulate item rankings by injecting fake users. Although widely studied, most existing attack models fail to meet two critical objectives simultaneously: achieving strong adversarial promotion of target items while maintaining realistic behavior to evade detection. As a result, the true severity of shilling threats that manage to reconcile the two objectives remains underappreciated. To expose this overlooked vulnerability, we present DLDA, a diffusion-based attack framework that can generate highly effective yet indistinguishable fake users by enabling fine-grained control over target promotion. Specifically, DLDA operates in a pre-aligned collaborative embedding space, where it employs a conditional latent diffusion process to iteratively synthesize fake user profiles with precise target item control. To evade detection, DLDA introduces a dispersive regularization mechanism that promotes variability and realism in generated behavioral patterns. Extensive experiments on three real-world datasets and five popular RS models demonstrate that, compared to prior attacks, DLDA consistently achieves stronger item promotion while remaining harder to detect. These results highlight that modern RSs are more vulnerable than previously recognized, underscoring the urgent need for more robust defenses. 

---
# TIBSTC-CoT: A Multi-Domain Instruction Dataset for Chain-of-Thought Reasoning in Language Models 

**Authors**: Fan Gao, Cheng Huang, Nyima Tashi, Yutong Liu, Xiangxiang Wang, Thupten Tsering, Ban Ma-bao, Renzeg Duojie, Gadeng Luosang, Rinchen Dongrub, Dorje Tashi, Xiao Feng, Hao Wang, Yongbin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01977)  

**Abstract**: To address the severe data scarcity in Tibetan, a low-resource language spoken by over six million people, we introduce TIBSTC-CoT, the large-scale, multi-domain Tibetan dataset automatically constructed via chain-of-thought prompting with large language models (LLMs). TIBSTC-CoT establishes a scalable and reproducible framework for dataset creation in low-resource settings, covering diverse domains and reasoning patterns essential for language understanding and generation. Building on this dataset, we develop the Sunshine-thinking LLM family, a series of Tibetan-centric LLMs equipped with chain-of-thought capabilities. Trained entirely on TIBSTC-CoT, Sunshine-thinking has demonstrated strong reasoning and generation performance, comparable to state-of-the-art (SOTA) multilingual LLMs. Our work marks a significant step toward inclusive AI by enabling high-quality Tibetan language processing through both resource creation and model innovation. All data are available: this https URL. 

---
# Accelerating LLM Reasoning via Early Rejection with Partial Reward Modeling 

**Authors**: Seyyed Saeid Cheshmi, Azal Ahmad Khan, Xinran Wang, Zirui Liu, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2508.01969)  

**Abstract**: Large Language Models (LLMs) are increasingly relied upon for solving complex reasoning tasks in domains such as mathematics, logic, and multi-step question answering. A growing line of work seeks to improve reasoning quality by scaling inference time compute particularly through Process Reward Models (PRMs), used to reward the reasoning at intermediate steps. While effective, these methods introduce substantial computational overhead, especially when generating large numbers of solutions in parallel. In this paper, we investigate whether PRMs can be used mid-generation to provide early signals that enable the rejection of suboptimal candidates before full generation of step is complete. We introduce the hypothesis that PRMs are also Partial Reward Models, meaning that the scores they assign to partially completed reasoning step are predictive of final output quality. This allows for principled early rejection based on intermediate token-level signals. We support this hypothesis both theoretically, by proving that the risk of discarding optimal beams decreases exponentially with generation length and empirically, by demonstrating a strong correlation between partial and final rewards across multiple reward models. On math reasoning benchmarks, our method achieves up to 1.4$\times$-9$\times$ reduction in inference FLOPs without degrading final performance. These results suggest that early rejection is a powerful mechanism for improving the compute-efficiency of reasoning in LLMs. 

---
# Kronecker-LoRA: hybrid Kronecker-LoRA adapters for scalable, sustainable fine-tuning 

**Authors**: Yixin Shen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01961)  

**Abstract**: Fine-tuning massive pre-trained language models across many tasks demands adapters that are both parameter-efficient and highly expressive. We introduce \textbf{Kron-LoRA}, a two-stage adapter that first factorizes each frozen linear update as a Kronecker product \[ \Delta W = A \otimes B \] and then compresses \[ B \in \mathbb{R}^{d_{B2}\times d_{B1}} \] via an \(r\)-rank LoRA decomposition \(B \approx B_{1}B_{2}\). By leveraging \[ \mathrm{rank}(A \otimes B) \;=\; \mathrm{rank}(A)\,\mathrm{rank}(B), \] Kron-LoRA retains the expressivity of the update while using up to $4\!\times\!$ fewer parameters than a standard rank-8 LoRA adapter. Its compact adapter matrices also quantize to 8- or 4-bit with less accuracy degradation than LoRA, enabling further memory and storage savings for on-device deployment. We benchmark on DistilBERT and Mistral-7B across five tasks (PIQA, HellaSwag, WinoGrande, ARC-Easy, ARC-Challenge) over multiple epochs of adapter-only tuning: on DistilBERT, an 840 K-parameter Kron-LoRA matches LoRA-16's performance, and on Mistral-7B, a 5.7 M-parameter Kron-LoRA rivals LoRA-8 with modest memory savings and only a 3-8\% speed overhead. In sequential fine-tuning from ARC-Challenge to ARC-Easy, Kron-LoRA retains 55.18\% accuracy versus 53.17\% for LoRA-8-despite using only one-quarter of the adapter parameters-underscoring its competitive cross-task transfer performance. By uniting Kronecker structure, low-rank compression, quantization-friendliness, and by providing transparent trade-off analysis, Kron-LoRA offers a scalable, sustainable, and continual-learning-ready solution for multi-task adaptation of large language models. 

---
# Flow-Aware GNN for Transmission Network Reconfiguration via Substation Breaker Optimization 

**Authors**: Dekang Meng, Rabab Haider, Pascal van Hentenryck  

**Link**: [PDF](https://arxiv.org/pdf/2508.01951)  

**Abstract**: This paper introduces OptiGridML, a machine learning framework for discrete topology optimization in power grids. The task involves selecting substation breaker configurations that maximize cross-region power exports, a problem typically formulated as a mixed-integer program (MIP) that is NP-hard and computationally intractable for large networks. OptiGridML replaces repeated MIP solves with a two-stage neural architecture: a line-graph neural network (LGNN) that approximates DC power flows for a given network topology, and a heterogeneous GNN (HeteroGNN) that predicts breaker states under structural and physical constraints. A physics-informed consistency loss connects these components by enforcing Kirchhoff's law on predicted flows. Experiments on synthetic networks with up to 1,000 breakers show that OptiGridML achieves power export improvements of up to 18% over baseline topologies, while reducing inference time from hours to milliseconds. These results demonstrate the potential of structured, flow-aware GNNs for accelerating combinatorial optimization in physical networked systems. 

---
# Inferring Reward Machines and Transition Machines from Partially Observable Markov Decision Processes 

**Authors**: Yuly Wu, Jiamou Liu, Libo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01947)  

**Abstract**: Partially Observable Markov Decision Processes (POMDPs) are fundamental to many real-world applications. Although reinforcement learning (RL) has shown success in fully observable domains, learning policies from traces in partially observable environments remains challenging due to non-Markovian observations. Inferring an automaton to handle the non-Markovianity is a proven effective approach, but faces two limitations: 1) existing automaton representations focus only on reward-based non-Markovianity, leading to unnatural problem formulations; 2) inference algorithms face enormous computational costs. For the first limitation, we introduce Transition Machines (TMs) to complement existing Reward Machines (RMs). To develop a unified inference algorithm for both automata types, we propose the Dual Behavior Mealy Machine (DBMM) that subsumes both TMs and RMs. We then introduce DB-RPNI, a passive automata learning algorithm that efficiently infers DBMMs while avoiding the costly reductions required by prior work. We further develop optimization techniques and identify sufficient conditions for inferring the minimal correct automata. Experimentally, our inference method achieves speedups of up to three orders of magnitude over SOTA baselines. 

---
# ROVER: Recursive Reasoning Over Videos with Vision-Language Models for Embodied Tasks 

**Authors**: Philip Schroeder, Ondrej Biza, Thomas Weng, Hongyin Luo, James Glass  

**Link**: [PDF](https://arxiv.org/pdf/2508.01943)  

**Abstract**: Vision-language models (VLMs) have exhibited impressive capabilities across diverse image understanding tasks, but still struggle in settings that require reasoning over extended sequences of camera frames from a video. This limits their utility in embodied settings, which require reasoning over long frame sequences from a continuous stream of visual input at each moment of a task attempt. To address this limitation, we propose ROVER (Reasoning Over VidEo Recursively), a framework that enables the model to recursively decompose long-horizon video trajectories into segments corresponding to shorter subtasks within the trajectory. In doing so, ROVER facilitates more focused and accurate reasoning over temporally localized frame sequences without losing global context. We evaluate ROVER, implemented using an in-context learning approach, on diverse OpenX Embodiment videos and on a new dataset derived from RoboCasa that consists of 543 videos showing both expert and perturbed non-expert trajectories across 27 robotic manipulation tasks. ROVER outperforms strong baselines across three video reasoning tasks: task progress estimation, frame-level natural language reasoning, and video question answering. We observe that, by reducing the number of frames the model reasons over at each timestep, ROVER mitigates hallucinations, especially during unexpected or non-optimal moments of a trajectory. In addition, by enabling the implementation of a subtask-specific sliding context window, ROVER's time complexity scales linearly with video length, an asymptotic improvement over baselines. Demos, code, and data available at: this https URL 

---
# Less is More: AMBER-AFNO -- a New Benchmark for Lightweight 3D Medical Image Segmentation 

**Authors**: Andrea Dosi, Semanto Mondal, Rajib Chandra Ghosh, Massimo Brescia, Giuseppe Longo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01941)  

**Abstract**: This work presents the results of a methodological transfer from remote sensing to healthcare, adapting AMBER -- a transformer-based model originally designed for multiband images, such as hyperspectral data -- to the task of 3D medical datacube segmentation. In this study, we use the AMBER architecture with Adaptive Fourier Neural Operators (AFNO) in place of the multi-head self-attention mechanism. While existing models rely on various forms of attention to capture global context, AMBER-AFNO achieves this through frequency-domain mixing, enabling a drastic reduction in model complexity. This design reduces the number of trainable parameters by over 80% compared to UNETR++, while maintaining a FLOPs count comparable to other state-of-the-art architectures. Model performance is evaluated on two benchmark 3D medical datasets -- ACDC and Synapse -- using standard metrics such as Dice Similarity Coefficient (DSC) and Hausdorff Distance (HD), demonstrating that AMBER-AFNO achieves competitive or superior accuracy with significant gains in training efficiency, inference speed, and memory usage. 

---
# Proactive Disentangled Modeling of Trigger-Object Pairings for Backdoor Defense 

**Authors**: Kyle Stein, Andrew A. Mahyari, Guillermo Francia III, Eman El-Sheikh  

**Link**: [PDF](https://arxiv.org/pdf/2508.01932)  

**Abstract**: Deep neural networks (DNNs) and generative AI (GenAI) are increasingly vulnerable to backdoor attacks, where adversaries embed triggers into inputs to cause models to misclassify or misinterpret target labels. Beyond traditional single-trigger scenarios, attackers may inject multiple triggers across various object classes, forming unseen backdoor-object configurations that evade standard detection pipelines. In this paper, we introduce DBOM (Disentangled Backdoor-Object Modeling), a proactive framework that leverages structured disentanglement to identify and neutralize both seen and unseen backdoor threats at the dataset level. Specifically, DBOM factorizes input image representations by modeling triggers and objects as independent primitives in the embedding space through the use of Vision-Language Models (VLMs). By leveraging the frozen, pre-trained encoders of VLMs, our approach decomposes the latent representations into distinct components through a learnable visual prompt repository and prompt prefix tuning, ensuring that the relationships between triggers and objects are explicitly captured. To separate trigger and object representations in the visual prompt repository, we introduce the trigger-object separation and diversity losses that aids in disentangling trigger and object visual features. Next, by aligning image features with feature decomposition and fusion, as well as learned contextual prompt tokens in a shared multimodal space, DBOM enables zero-shot generalization to novel trigger-object pairings that were unseen during training, thereby offering deeper insights into adversarial attack patterns. Experimental results on CIFAR-10 and GTSRB demonstrate that DBOM robustly detects poisoned images prior to downstream training, significantly enhancing the security of DNN training pipelines. 

---
# Word Overuse and Alignment in Large Language Models: The Influence of Learning from Human Feedback 

**Authors**: Tom S. Juzek, Zina B. Ward  

**Link**: [PDF](https://arxiv.org/pdf/2508.01930)  

**Abstract**: Large Language Models (LLMs) are known to overuse certain terms like "delve" and "intricate." The exact reasons for these lexical choices, however, have been unclear. Using Meta's Llama model, this study investigates the contribution of Learning from Human Feedback (LHF), under which we subsume Reinforcement Learning from Human Feedback and Direct Preference Optimization. We present a straightforward procedure for detecting the lexical preferences of LLMs that are potentially LHF-induced. Next, we more conclusively link LHF to lexical overuse by experimentally emulating the LHF procedure and demonstrating that participants systematically prefer text variants that include certain words. This lexical overuse can be seen as a sort of misalignment, though our study highlights the potential divergence between the lexical expectations of different populations -- namely LHF workers versus LLM users. Our work contributes to the growing body of research on explainable artificial intelligence and emphasizes the importance of both data and procedural transparency in alignment research. 

---
# IAUNet: Instance-Aware U-Net 

**Authors**: Yaroslav Prytula, Illia Tsiporenko, Ali Zeynalli, Dmytro Fishman  

**Link**: [PDF](https://arxiv.org/pdf/2508.01928)  

**Abstract**: Instance segmentation is critical in biomedical imaging to accurately distinguish individual objects like cells, which often overlap and vary in size. Recent query-based methods, where object queries guide segmentation, have shown strong performance. While U-Net has been a go-to architecture in medical image segmentation, its potential in query-based approaches remains largely unexplored. In this work, we present IAUNet, a novel query-based U-Net architecture. The core design features a full U-Net architecture, enhanced by a novel lightweight convolutional Pixel decoder, making the model more efficient and reducing the number of parameters. Additionally, we propose a Transformer decoder that refines object-specific features across multiple scales. Finally, we introduce the 2025 Revvity Full Cell Segmentation Dataset, a unique resource with detailed annotations of overlapping cell cytoplasm in brightfield images, setting a new benchmark for biomedical instance segmentation. Experiments on multiple public datasets and our own show that IAUNet outperforms most state-of-the-art fully convolutional, transformer-based, and query-based models and cell segmentation-specific models, setting a strong baseline for cell instance segmentation tasks. Code is available at this https URL 

---
# Quantum-RAG and PunGPT2: Advancing Low-Resource Language Generation and Retrieval for the Punjabi Language 

**Authors**: Jaskaranjeet Singh, Rakesh Thakur  

**Link**: [PDF](https://arxiv.org/pdf/2508.01918)  

**Abstract**: Despite the rapid advancement of large language models (LLMs), low-resource languages remain largely excluded from the NLP landscape. We present PunGPT2, the first fully open-source suite of Punjabi large language models, trained from scratch on a 35GB domain-diverse corpus encompassing literature, religious texts, news, and social discourse. Unlike prior multilingual approaches, PunGPT2 captures rich syntactic and morphological features unique to Punjabi through a tokenizer optimised with byte pair encoding and linguistically aligned pretraining objectives. To improve factual grounding and domain recall, we introduce Pun-RAG, a retrieval-augmented generation framework combining PunGPT2 with a dense FAISS retriever over a curated Punjabi knowledge base. We further develop Pun-Instruct, a parameter-efficient, instruction-tuned variant using QLoRA, enabling robust zero-shot and instruction-following performance with significantly reduced compute needs.
As a key innovation, we propose Quantum-RAG, a novel hybrid retrieval system that fuses sparse (BM25) and dense methods with quantum-inspired semantic matching. By encoding queries using amplitude-based embeddings and retrieving via quantum kernel similarity, Quantum-RAG achieves improved contextual relevance with minimal memory overhead marking the first practical integration of quantum representations in low-resource language generation. Our models significantly outperform strong multilingual baselines (mBERT, mT5, MuRIL) in perplexity, factuality, and fluency. This work provides a scalable, reproducible blueprint for extending LLM capabilities to underrepresented languages and pioneers quantum-aware retrieval in low-resource NLP 

---
# L3M+P: Lifelong Planning with Large Language Models 

**Authors**: Krish Agarwal, Yuqian Jiang, Jiaheng Hu, Bo Liu, Peter Stone  

**Link**: [PDF](https://arxiv.org/pdf/2508.01917)  

**Abstract**: By combining classical planning methods with large language models (LLMs), recent research such as LLM+P has enabled agents to plan for general tasks given in natural language. However, scaling these methods to general-purpose service robots remains challenging: (1) classical planning algorithms generally require a detailed and consistent specification of the environment, which is not always readily available; and (2) existing frameworks mainly focus on isolated planning tasks, whereas robots are often meant to serve in long-term continuous deployments, and therefore must maintain a dynamic memory of the environment which can be updated with multi-modal inputs and extracted as planning knowledge for future tasks. To address these two issues, this paper introduces L3M+P (Lifelong LLM+P), a framework that uses an external knowledge graph as a representation of the world state. The graph can be updated from multiple sources of information, including sensory input and natural language interactions with humans. L3M+P enforces rules for the expected format of the absolute world state graph to maintain consistency between graph updates. At planning time, given a natural language description of a task, L3M+P retrieves context from the knowledge graph and generates a problem definition for classical planners. Evaluated on household robot simulators and on a real-world service robot, L3M+P achieves significant improvement over baseline methods both on accurately registering natural language state changes and on correctly generating plans, thanks to the knowledge graph retrieval and verification. 

---
# Decomposing Representation Space into Interpretable Subspaces with Unsupervised Learning 

**Authors**: Xinting Huang, Michael Hahn  

**Link**: [PDF](https://arxiv.org/pdf/2508.01916)  

**Abstract**: Understanding internal representations of neural models is a core interest of mechanistic interpretability. Due to its large dimensionality, the representation space can encode various aspects about inputs. To what extent are different aspects organized and encoded in separate subspaces? Is it possible to find these ``natural'' subspaces in a purely unsupervised way? Somewhat surprisingly, we can indeed achieve this and find interpretable subspaces by a seemingly unrelated training objective. Our method, neighbor distance minimization (NDM), learns non-basis-aligned subspaces in an unsupervised manner. Qualitative analysis shows subspaces are interpretable in many cases, and encoded information in obtained subspaces tends to share the same abstract concept across different inputs, making such subspaces similar to ``variables'' used by the model. We also conduct quantitative experiments using known circuits in GPT-2; results show a strong connection between subspaces and circuit variables. We also provide evidence showing scalability to 2B models by finding separate subspaces mediating context and parametric knowledge routing. Viewed more broadly, our findings offer a new perspective on understanding model internals and building circuits. 

---
# Revisiting Replay and Gradient Alignment for Continual Pre-Training of Large Language Models 

**Authors**: Istabrak Abbes, Gopeshh Subbaraj, Matthew Riemer, Nizar Islah, Benjamin Therien, Tsuguchika Tabaru, Hiroaki Kingetsu, Sarath Chandar, Irina Rish  

**Link**: [PDF](https://arxiv.org/pdf/2508.01908)  

**Abstract**: Training large language models (LLMs) typically involves pre-training on massive corpora, only to restart the process entirely when new data becomes available. A more efficient and resource-conserving approach would be continual pre-training, where models are updated with new data rather than retraining from scratch. However, the introduction of new data often causes distribution shifts, leading to performance degradation on previously learned tasks. In this paper, we take a deeper look at two popular proposals for addressing this distribution shift within the continual learning literature: experience replay and gradient alignment. We consider continual pre-training of models within the Llama family of architectures at a large scale across languages with 100 billion tokens of training data in each language, finding that both replay and gradient alignment lead to more stable learning without forgetting. This conclusion holds both as we vary the model scale and as we vary the number and diversity of tasks. Moreover, we are the first to demonstrate the effectiveness of gradient alignment techniques in the context of LLM pre-training and propose an efficient implementation of meta-experience replay (MER) that imbues experience replay with the benefits of gradient alignment despite negligible compute and memory overhead. Our scaling analysis across model sizes and replay rates indicates that small rates of replaying old examples are definitely a more valuable use of compute than investing in model size, but that it is more compute efficient to scale the size of the model than invest in high rates of replaying old examples. 

---
# How Does Controllability Emerge In Language Models During Pretraining? 

**Authors**: Jianshu She, Xinyue Li, Eric Xing, Zhengzhong Liu, Qirong Ho  

**Link**: [PDF](https://arxiv.org/pdf/2508.01892)  

**Abstract**: Language models can be steered by modifying their internal representations to control concepts such as emotion, style, or truthfulness in generation. However, the conditions for an effective intervention remain unclear and are often validated through heuristics and trial-and-error. To fill this gap, we demonstrate that intervention efficacy, measured by linear steerability (i.e., the ability to adjust output via linear transformations of hidden states), emerges during intermediate stages of training. Moreover, even closely related concepts (e.g., anger and sadness) exhibit steerability emergence at distinct stages of training.
To better interpret the dynamics of steerability during training, we adapt existing intervention techniques into a unified framework, referred to as the "Intervention Detector" (ID), which is designed to reveal how linear steerability evolves over the course of training through hidden state and representation analysis. ID reveals that concepts become increasingly linearly separable in the hidden space as training progresses, which strongly correlates with the emergence of linear steerability. We further introduce ID-based metrics, such as heatmaps, entropy trends, and cosine similarity, to help interpret how linear steerability evolves throughout training. In addition, we apply ID across different model families to ensure the generality of our findings on steerability dynamics. 

---
# Complete Evasion, Zero Modification: PDF Attacks on AI Text Detection 

**Authors**: Aldan Creo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01887)  

**Abstract**: AI-generated text detectors have become essential tools for maintaining content authenticity, yet their robustness against evasion attacks remains questionable. We present PDFuzz, a novel attack that exploits the discrepancy between visual text layout and extraction order in PDF documents. Our method preserves exact textual content while manipulating character positioning to scramble extraction sequences. We evaluate this approach against the ArguGPT detector using a dataset of human and AI-generated text. Our results demonstrate complete evasion: detector performance drops from (93.6 $\pm$ 1.4) % accuracy and 0.938 $\pm$ 0.014 F1 score to random-level performance ((50.4 $\pm$ 3.2) % accuracy, 0.0 F1 score) while maintaining perfect visual fidelity. Our work reveals a vulnerability in current detection systems that is inherent to PDF document structures and underscores the need for implementing sturdy safeguards against such attacks. We make our code publicly available at this https URL. 

---
# Counterfactual Reciprocal Recommender Systems for User-to-User Matching 

**Authors**: Kazuki Kawamura, Takuma Udagawa, Kei Tateno  

**Link**: [PDF](https://arxiv.org/pdf/2508.01867)  

**Abstract**: Reciprocal recommender systems (RRS) in dating, gaming, and talent platforms require mutual acceptance for a match. Logged data, however, over-represents popular profiles due to past exposure policies, creating feedback loops that skew learning and fairness. We introduce Counterfactual Reciprocal Recommender Systems (CFRR), a causal framework to mitigate this bias. CFRR uses inverse propensity scored, self-normalized objectives. Experiments show CFRR improves NDCG@10 by up to 3.5% (e.g., from 0.459 to 0.475 on DBLP, from 0.299 to 0.307 on Synthetic), increases long-tail user coverage by up to 51% (from 0.504 to 0.763 on Synthetic), and reduces Gini exposure inequality by up to 24% (from 0.708 to 0.535 on Synthetic). CFRR offers a promising approach for more accurate and fair user-to-user matching. 

---
# Counterfactual Probing for Hallucination Detection and Mitigation in Large Language Models 

**Authors**: Yijun Feng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01862)  

**Abstract**: Large Language Models have demonstrated remarkable capabilities across diverse tasks, yet they frequently generate hallucinations outputs that are fluent but factually incorrect or unsupported. We propose Counterfactual Probing, a novel approach for detecting and mitigating hallucinations in LLM outputs. Our method dynamically generates counterfactual statements that appear plausible but contain subtle factual errors, then evaluates the model's sensitivity to these perturbations. We hypothesize that genuine knowledge exhibits robustness to counterfactual variations, while hallucinated content shows inconsistent confidence patterns when confronted with plausible alternatives. Our comprehensive evaluation on TruthfulQA, factual statement datasets, and curated hallucination examples demonstrates that counterfactual probing achieves superior detection performance compared to baseline methods, while our adaptive mitigation strategies reduce hallucination scores by an average of 24.5%. The approach requires no model retraining and can be integrated into existing LLM pipelines as a realtime verification mechanism. 

---
# ACT-Tensor: Tensor Completion Framework for Financial Dataset Imputation 

**Authors**: Junyi Mo, Jiayu Li, Duo Zhang, Elynn Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01861)  

**Abstract**: Missing data in financial panels presents a critical obstacle, undermining asset-pricing models and reducing the effectiveness of investment strategies. Such panels are often inherently multi-dimensional, spanning firms, time, and financial variables, which adds complexity to the imputation task. Conventional imputation methods often fail by flattening the data's multidimensional structure, struggling with heterogeneous missingness patterns, or overfitting in the face of extreme data sparsity. To address these limitations, we introduce an Adaptive, Cluster-based Temporal smoothing tensor completion framework (ACT-Tensor) tailored for severely and heterogeneously missing multi-dimensional financial data panels. ACT-Tensor incorporates two key innovations: a cluster-based completion module that captures cross-sectional heterogeneity by learning group-specific latent structures; and a temporal smoothing module that proactively removes short-lived noise while preserving slow-moving fundamental trends. Extensive experiments show that ACT-Tensor consistently outperforms state-of-the-art benchmarks in terms of imputation accuracy across a range of missing data regimes, including extreme sparsity scenarios. To assess its practical financial utility, we evaluate the imputed data with an asset-pricing pipeline tailored for tensor-structured financial data. Results show that ACT-Tensor not only reduces pricing errors but also significantly improves risk-adjusted returns of the constructed portfolio. These findings confirm that our method delivers highly accurate and informative imputations, offering substantial value for financial decision-making. 

---
# Web-CogReasoner: Towards Knowledge-Induced Cognitive Reasoning for Web Agents 

**Authors**: Yuhan Guo, Cong Guo, Aiwen Sun, Hongliang He, Xinyu Yang, Yue Lu, Yingji Zhang, Xuntao Guo, Dong Zhang, Jianzhuang Liu, Jiang Duan, Yijia Xiao, Liangjian Wen, Hai-Ming Xu, Yong Dai  

**Link**: [PDF](https://arxiv.org/pdf/2508.01858)  

**Abstract**: Multimodal large-scale models have significantly advanced the development of web agents, enabling perception and interaction with digital environments akin to human cognition. In this paper, we argue that web agents must first acquire sufficient knowledge to effectively engage in cognitive reasoning. Therefore, we decompose a web agent's capabilities into two essential stages: knowledge content learning and cognitive processes. To formalize this, we propose Web-CogKnowledge Framework, categorizing knowledge as Factual, Conceptual, and Procedural. In this framework, knowledge content learning corresponds to the agent's processes of Memorizing and Understanding, which rely on the first two knowledge types, representing the "what" of learning. Conversely, cognitive processes correspond to Exploring, grounded in Procedural knowledge, defining the "how" of reasoning and action. To facilitate knowledge acquisition, we construct the Web-CogDataset, a structured resource curated from 14 real-world websites, designed to systematically instill core knowledge necessary for web agent. This dataset serves as the agent's conceptual grounding-the "nouns" upon which comprehension is built-as well as the basis for learning how to reason and act. Building on this foundation, we operationalize these processes through a novel knowledge-driven Chain-of-Thought (CoT) reasoning framework, developing and training our proposed agent, the Web-CogReasoner. Extensive experimentation reveals its significant superiority over existing models, especially in generalizing to unseen tasks where structured knowledge is decisive. To enable rigorous evaluation, we introduce the Web-CogBench, a comprehensive evaluation suite designed to assess and compare agent performance across the delineated knowledge domains and cognitive capabilities. Our code and data is open sourced at this https URL 

---
# ChairPose: Pressure-based Chair Morphology Grounded Sitting Pose Estimation through Simulation-Assisted Training 

**Authors**: Lala Shakti Swarup Ray, Vitor Fortes Rey, Bo Zhou, Paul Lukowicz, Sungho Suh  

**Link**: [PDF](https://arxiv.org/pdf/2508.01850)  

**Abstract**: Prolonged seated activity is increasingly common in modern environments, raising concerns around musculoskeletal health, ergonomics, and the design of responsive interactive systems. Existing posture sensing methods such as vision-based or wearable approaches face limitations including occlusion, privacy concerns, user discomfort, and restricted deployment flexibility. We introduce ChairPose, the first full body, wearable free seated pose estimation system that relies solely on pressure sensing and operates independently of chair geometry. ChairPose employs a two stage generative model trained on pressure maps captured from a thin, chair agnostic sensing mattress. Unlike prior approaches, our method explicitly incorporates chair morphology into the inference process, enabling accurate, occlusion free, and privacy preserving pose estimation. To support generalization across diverse users and chairs, we introduce a physics driven data augmentation pipeline that simulates realistic variations in posture and seating conditions. Evaluated across eight users and four distinct chairs, ChairPose achieves a mean per joint position error of 89.4 mm when both the user and the chair are unseen, demonstrating robust generalization to novel real world generalizability. ChairPose expands the design space for posture aware interactive systems, with potential applications in ergonomics, healthcare, and adaptive user interfaces. 

---
# Beyond Vulnerabilities: A Survey of Adversarial Attacks as Both Threats and Defenses in Computer Vision Systems 

**Authors**: Zhongliang Guo, Yifei Qian, Yanli Li, Weiye Li, Chun Tong Lei, Shuai Zhao, Lei Fang, Ognjen Arandjelović, Chun Pong Lau  

**Link**: [PDF](https://arxiv.org/pdf/2508.01845)  

**Abstract**: Adversarial attacks against computer vision systems have emerged as a critical research area that challenges the fundamental assumptions about neural network robustness and security. This comprehensive survey examines the evolving landscape of adversarial techniques, revealing their dual nature as both sophisticated security threats and valuable defensive tools. We provide a systematic analysis of adversarial attack methodologies across three primary domains: pixel-space attacks, physically realizable attacks, and latent-space attacks. Our investigation traces the technical evolution from early gradient-based methods such as FGSM and PGD to sophisticated optimization techniques incorporating momentum, adaptive step sizes, and advanced transferability mechanisms. We examine how physically realizable attacks have successfully bridged the gap between digital vulnerabilities and real-world threats through adversarial patches, 3D textures, and dynamic optical perturbations. Additionally, we explore the emergence of latent-space attacks that leverage semantic structure in internal representations to create more transferable and meaningful adversarial examples. Beyond traditional offensive applications, we investigate the constructive use of adversarial techniques for vulnerability assessment in biometric authentication systems and protection against malicious generative models. Our analysis reveals critical research gaps, particularly in neural style transfer protection and computational efficiency requirements. This survey contributes a comprehensive taxonomy, evolution analysis, and identification of future research directions, aiming to advance understanding of adversarial vulnerabilities and inform the development of more robust and trustworthy computer vision systems. 

---
# Neural Predictive Control to Coordinate Discrete- and Continuous-Time Models for Time-Series Analysis with Control-Theoretical Improvements 

**Authors**: Haoran Li, Muhao Guo, Yang Weng, Hanghang Tong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01833)  

**Abstract**: Deep sequence models have achieved notable success in time-series analysis, such as interpolation and forecasting. Recent advances move beyond discrete-time architectures like Recurrent Neural Networks (RNNs) toward continuous-time formulations such as the family of Neural Ordinary Differential Equations (Neural ODEs). Generally, they have shown that capturing the underlying dynamics is beneficial for generic tasks like interpolation, extrapolation, and classification. However, existing methods approximate the dynamics using unconstrained neural networks, which struggle to adapt reliably under distributional shifts. In this paper, we recast time-series problems as the continuous ODE-based optimal control problem. Rather than learning dynamics solely from data, we optimize control actions that steer ODE trajectories toward task objectives, bringing control-theoretical performance guarantees. To achieve this goal, we need to (1) design the appropriate control actions and (2) apply effective optimal control algorithms. As the actions should contain rich context information, we propose to employ the discrete-time model to process past sequences and generate actions, leading to a coordinate model to extract long-term temporal features to modulate short-term continuous dynamics. During training, we apply model predictive control to plan multi-step future trajectories, minimize a task-specific cost, and greedily select the optimal current action. We show that, under mild assumptions, this multi-horizon optimization leads to exponential convergence to infinite-horizon solutions, indicating that the coordinate model can gain robust and generalizable performance. Extensive experiments on diverse time-series datasets validate our method's superior generalization and adaptability compared to state-of-the-art baselines. 

---
# Deep Learning-Driven Prediction of Microstructure Evolution via Latent Space Interpolation 

**Authors**: Sachin Gaikwad, Thejas Kasilingam, Owais Ahmad, Rajdip Mukherjee, Somnath Bhowmick  

**Link**: [PDF](https://arxiv.org/pdf/2508.01822)  

**Abstract**: Phase-field models accurately simulate microstructure evolution, but their dependence on solving complex differential equations makes them computationally expensive. This work achieves a significant acceleration via a novel deep learning-based framework, utilizing a Conditional Variational Autoencoder (CVAE) coupled with Cubic Spline Interpolation and Spherical Linear Interpolation (SLERP). We demonstrate the method for binary spinodal decomposition by predicting microstructure evolution for intermediate alloy compositions from a limited set of training compositions. First, using microstructures from phase-field simulations of binary spinodal decomposition, we train the CVAE, which learns compact latent representations that encode essential morphological features. Next, we use cubic spline interpolation in the latent space to predict microstructures for any unknown composition. Finally, SLERP ensures smooth morphological evolution with time that closely resembles coarsening. The predicted microstructures exhibit high visual and statistical similarity to phase-field simulations. This framework offers a scalable and efficient surrogate model for microstructure evolution, enabling accelerated materials design and composition optimization. 

---
# AGENTICT$^2$S:Robust Text-to-SPARQL via Agentic Collaborative Reasoning over Heterogeneous Knowledge Graphs for the Circular Economy 

**Authors**: Yang Zhao, Chengxiao Dai, Wei Zhuo, Tan Chuan Fu, Yue Xiu, Dusit Niyato, Jonathan Z. Low, Eugene Ho Hong Zhuang, Daren Zong Loong Tan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01815)  

**Abstract**: Question answering over heterogeneous knowledge graphs (KGQA) involves reasoning across diverse schemas, incomplete alignments, and distributed data sources. Existing text-to-SPARQL approaches rely on large-scale domain-specific fine-tuning or operate within single-graph settings, limiting their generalizability in low-resource domains and their ability to handle queries spanning multiple graphs. These challenges are particularly relevant in domains such as the circular economy, where information about classifications, processes, and emissions is distributed across independently curated knowledge graphs (KGs). We present AgenticT$^2$S, a modular framework that decomposes KGQA into subtasks managed by specialized agents responsible for retrieval, query generation, and verification. A scheduler assigns subgoals to different graphs using weak-to-strong alignment strategies. A two-stage verifier detects structurally invalid and semantically underspecified queries through symbolic validation and counterfactual consistency checks. Experiments on real-world circular economy KGs demonstrate that AgenticT$^2$S improves execution accuracy by 17.3% and triple level F$_1$ by 25.4% over the best baseline, while reducing the average prompt length by 46.4%. These results demonstrate the benefits of agent-based schema-aware reasoning for scalable KGQA and support decision-making in sustainability domains through robust cross-graph reasoning. 

---
# HeQ: a Large and Diverse Hebrew Reading Comprehension Benchmark 

**Authors**: Amir DN Cohen, Hilla Merhav, Yoav Goldberg, Reut Tsarfaty  

**Link**: [PDF](https://arxiv.org/pdf/2508.01812)  

**Abstract**: Current benchmarks for Hebrew Natural Language Processing (NLP) focus mainly on morpho-syntactic tasks, neglecting the semantic dimension of language understanding. To bridge this gap, we set out to deliver a Hebrew Machine Reading Comprehension (MRC) dataset, where MRC is to be realized as extractive Question Answering. The morphologically rich nature of Hebrew poses a challenge to this endeavor: the indeterminacy and non-transparency of span boundaries in morphologically complex forms lead to annotation inconsistencies, disagreements, and flaws in standard evaluation metrics.
To remedy this, we devise a novel set of guidelines, a controlled crowdsourcing protocol, and revised evaluation metrics that are suitable for the morphologically rich nature of the language. Our resulting benchmark, HeQ (Hebrew QA), features 30,147 diverse question-answer pairs derived from both Hebrew Wikipedia articles and Israeli tech news. Our empirical investigation reveals that standard evaluation metrics such as F1 scores and Exact Match (EM) are not appropriate for Hebrew (and other MRLs), and we propose a relevant enhancement.
In addition, our experiments show low correlation between models' performance on morpho-syntactic tasks and on MRC, which suggests that models designed for the former might underperform on semantics-heavy tasks. The development and exploration of HeQ illustrate some of the challenges MRLs pose in natural language understanding (NLU), fostering progression towards more and better NLU models for Hebrew and other MRLs. 

---
# Mitigating Persistent Client Dropout in Asynchronous Decentralized Federated Learning 

**Authors**: Ignacy Stępka, Nicholas Gisolfi, Kacper Trębacz, Artur Dubrawski  

**Link**: [PDF](https://arxiv.org/pdf/2508.01807)  

**Abstract**: We consider the problem of persistent client dropout in asynchronous Decentralized Federated Learning (DFL). Asynchronicity and decentralization obfuscate information about model updates among federation peers, making recovery from a client dropout difficult. Access to the number of learning epochs, data distributions, and all the information necessary to precisely reconstruct the missing neighbor's loss functions is limited. We show that obvious mitigations do not adequately address the problem and introduce adaptive strategies based on client reconstruction. We show that these strategies can effectively recover some performance loss caused by dropout. Our work focuses on asynchronous DFL with local regularization and differs substantially from that in the existing literature. We evaluate the proposed methods on tabular and image datasets, involve three DFL algorithms, and three data heterogeneity scenarios (iid, non-iid, class-focused non-iid). Our experiments show that the proposed adaptive strategies can be effective in maintaining robustness of federated learning, even if they do not reconstruct the missing client's data precisely. We also discuss the limitations and identify future avenues for tackling the problem of client dropout. 

---
# Contrastive Multi-Task Learning with Solvent-Aware Augmentation for Drug Discovery 

**Authors**: Jing Lan, Hexiao Ding, Hongzhao Chen, Yufeng Jiang, Ng Nga Chun, Gerald W.Y. Cheng, Zongxi Li, Jing Cai, Liang-ting Lin, Jung Sun Yoo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01799)  

**Abstract**: Accurate prediction of protein-ligand interactions is essential for computer-aided drug discovery. However, existing methods often fail to capture solvent-dependent conformational changes and lack the ability to jointly learn multiple related tasks. To address these limitations, we introduce a pre-training method that incorporates ligand conformational ensembles generated under diverse solvent conditions as augmented input. This design enables the model to learn both structural flexibility and environmental context in a unified manner. The training process integrates molecular reconstruction to capture local geometry, interatomic distance prediction to model spatial relationships, and contrastive learning to build solvent-invariant molecular representations. Together, these components lead to significant improvements, including a 3.7% gain in binding affinity prediction, an 82% success rate on the PoseBusters Astex docking benchmarks, and an area under the curve of 97.1% in virtual screening. The framework supports solvent-aware, multi-task modeling and produces consistent results across benchmarks. A case study further demonstrates sub-angstrom docking accuracy with a root-mean-square deviation of 0.157 angstroms, offering atomic-level insight into binding mechanisms and advancing structure-based drug design. 

---
# CSLRConformer: A Data-Centric Conformer Approach for Continuous Arabic Sign Language Recognition on the Isharah Datase 

**Authors**: Fatimah Mohamed Emad Elden  

**Link**: [PDF](https://arxiv.org/pdf/2508.01791)  

**Abstract**: The field of Continuous Sign Language Recognition (CSLR) poses substantial technical challenges, including fluid inter-sign transitions, the absence of temporal boundaries, and co-articulation effects. This paper, developed for the MSLR 2025 Workshop Challenge at ICCV 2025, addresses the critical challenge of signer-independent recognition to advance the generalization capabilities of CSLR systems across diverse signers. A data-centric methodology is proposed, centered on systematic feature engineering, a robust preprocessing pipeline, and an optimized model architecture. Key contributions include a principled feature selection process guided by Exploratory Data Analysis (EDA) to isolate communicative keypoints, a rigorous preprocessing pipeline incorporating DBSCAN-based outlier filtering and spatial normalization, and the novel CSLRConformer architecture. This architecture adapts the hybrid CNN-Transformer design of the Conformer model, leveraging its capacity to model local temporal dependencies and global sequence context; a characteristic uniquely suited for the spatio-temporal dynamics of sign language. The proposed methodology achieved a competitive performance, with a Word Error Rate (WER) of 5.60% on the development set and 12.01% on the test set, a result that secured a 3rd place ranking on the official competition platform. This research validates the efficacy of cross-domain architectural adaptation, demonstrating that the Conformer model, originally conceived for speech recognition, can be successfully repurposed to establish a new state-of-the-art performance in keypoint-based CSLR. 

---
# RouteMark: A Fingerprint for Intellectual Property Attribution in Routing-based Model Merging 

**Authors**: Xin He, Junxi Shen, Zhenheng Tang, Xiaowen Chu, Bo Li, Ivor W. Tsang, Yew-Soon Ong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01784)  

**Abstract**: Model merging via Mixture-of-Experts (MoE) has emerged as a scalable solution for consolidating multiple task-specific models into a unified sparse architecture, where each expert is derived from a model fine-tuned on a distinct task. While effective for multi-task integration, this paradigm introduces a critical yet underexplored challenge: how to attribute and protect the intellectual property (IP) of individual experts after merging. We propose RouteMark, a framework for IP protection in merged MoE models through the design of expert routing fingerprints. Our key insight is that task-specific experts exhibit stable and distinctive routing behaviors under probing inputs. To capture these patterns, we construct expert-level fingerprints using two complementary statistics: the Routing Score Fingerprint (RSF), quantifying the intensity of expert activation, and the Routing Preference Fingerprint (RPF), characterizing the input distribution that preferentially activates each expert. These fingerprints are reproducible, task-discriminative, and lightweight to construct. For attribution and tampering detection, we introduce a similarity-based matching algorithm that compares expert fingerprints between a suspect and a reference (victim) model. Extensive experiments across diverse tasks and CLIP-based MoE architectures show that RouteMark consistently yields high similarity for reused experts and clear separation from unrelated ones. Moreover, it remains robust against both structural tampering (expert replacement, addition, deletion) and parametric tampering (fine-tuning, pruning, permutation), outperforming weight- and activation-based baseliness. Our work lays the foundation for RouteMark as a practical and broadly applicable framework for IP verification in MoE-based model merging. 

---
# A comprehensive taxonomy of hallucinations in Large Language Models 

**Authors**: Manuel Cossio  

**Link**: [PDF](https://arxiv.org/pdf/2508.01781)  

**Abstract**: Large language models (LLMs) have revolutionized natural language processing, yet their propensity for hallucination, generating plausible but factually incorrect or fabricated content, remains a critical challenge. This report provides a comprehensive taxonomy of LLM hallucinations, beginning with a formal definition and a theoretical framework that posits its inherent inevitability in computable LLMs, irrespective of architecture or training. It explores core distinctions, differentiating between intrinsic (contradicting input context) and extrinsic (inconsistent with training data or reality), as well as factuality (absolute correctness) and faithfulness (adherence to input). The report then details specific manifestations, including factual errors, contextual and logical inconsistencies, temporal disorientation, ethical violations, and task-specific hallucinations across domains like code generation and multimodal applications. It analyzes the underlying causes, categorizing them into data-related issues, model-related factors, and prompt-related influences. Furthermore, the report examines cognitive and human factors influencing hallucination perception, surveys evaluation benchmarks and metrics for detection, and outlines architectural and systemic mitigation strategies. Finally, it introduces web-based resources for monitoring LLM releases and performance. This report underscores the complex, multifaceted nature of LLM hallucinations and emphasizes that, given their theoretical inevitability, future efforts must focus on robust detection, mitigation, and continuous human oversight for responsible and reliable deployment in critical applications. 

---
# VAGPO: Vision-augmented Asymmetric Group Preference Optimization for the Routing Problems 

**Authors**: Shiyan Liu, Bohan Tan, Yan Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01774)  

**Abstract**: The routing problems such as the Traveling Salesman Problem (TSP) and the Capacitated Vehicle Routing Problem (CVRP) are well-known combinatorial optimization challenges with broad practical relevance. Recent data-driven optimization methods have made significant progress, yet they often face limitations in training efficiency and generalization to large-scale instances. In this paper, we propose a novel Vision-Augmented Asymmetric Group Preference Optimization (VAGPO) approach for solving the routing problems. By leveraging ResNet-based visual encoding and Transformer-based sequential modeling, VAGPO captures both spatial structure and temporal dependencies. Furthermore, we introduce an asymmetric group preference optimization strategy that significantly accelerates convergence compared to commonly used policy gradient methods. Experimental results on TSP and CVRP benchmarks show that the proposed VAGPO not only achieves highly competitive solution quality but also exhibits strong generalization to larger instances (up to 1000 nodes) without re-training, highlighting its effectiveness in both learning efficiency and scalability. 

---
# LoRA-based methods on Unet for transfer learning in Subarachnoid Hematoma Segmentation 

**Authors**: Cristian Minoccheri, Matthew Hodgman, Haoyuan Ma, Rameez Merchant, Emily Wittrup, Craig Williamson, Kayvan Najarian  

**Link**: [PDF](https://arxiv.org/pdf/2508.01772)  

**Abstract**: Aneurysmal subarachnoid hemorrhage (SAH) is a life-threatening neurological emergency with mortality rates exceeding 30%. Transfer learning from related hematoma types represents a potentially valuable but underexplored approach. Although Unet architectures remain the gold standard for medical image segmentation due to their effectiveness on limited datasets, Low-Rank Adaptation (LoRA) methods for parameter-efficient transfer learning have been rarely applied to convolutional neural networks in medical imaging contexts. We implemented a Unet architecture pre-trained on computed tomography scans from 124 traumatic brain injury patients across multiple institutions, then fine-tuned on 30 aneurysmal SAH patients from the University of Michigan Health System using 3-fold cross-validation. We developed a novel CP-LoRA method based on tensor CP-decomposition and introduced DoRA variants (DoRA-C, convDoRA, CP-DoRA) that decompose weight matrices into magnitude and directional components. We compared these approaches against existing LoRA methods (LoRA-C, convLoRA) and standard fine-tuning strategies across different modules on a multi-view Unet model. LoRA-based methods consistently outperformed standard Unet fine-tuning. Performance varied by hemorrhage volume, with all methods showing improved accuracy for larger volumes. CP-LoRA achieved comparable performance to existing methods while using significantly fewer parameters. Over-parameterization with higher ranks consistently yielded better performance than strictly low-rank adaptations. This study demonstrates that transfer learning between hematoma types is feasible and that LoRA-based methods significantly outperform conventional Unet fine-tuning for aneurysmal SAH segmentation. 

---
# Semantically-Guided Inference for Conditional Diffusion Models: Enhancing Covariate Consistency in Time Series Forecasting 

**Authors**: Rui Ding, Hanyang Meng, Zeyang Zhang, Jielong Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01761)  

**Abstract**: Diffusion models have demonstrated strong performance in time series forecasting, yet often suffer from semantic misalignment between generated trajectories and conditioning covariates, especially under complex or multimodal conditions. To address this issue, we propose SemGuide, a plug-and-play, inference-time method that enhances covariate consistency in conditional diffusion models. Our approach introduces a scoring network to assess the semantic alignment between intermediate diffusion states and future covariates. These scores serve as proxy likelihoods in a stepwise importance reweighting procedure, which progressively adjusts the sampling path without altering the original training process. The method is model-agnostic and compatible with any conditional diffusion framework. Experiments on real-world forecasting tasks show consistent gains in both predictive accuracy and covariate alignment, with especially strong performance under complex conditioning scenarios. 

---
# Vision transformer-based multi-camera multi-object tracking framework for dairy cow monitoring 

**Authors**: Kumail Abbas, Zeeshan Afzal, Aqeel Raza, Taha Mansouri, Andrew W. Dowsey, Chaidate Inchaisri, Ali Alameer  

**Link**: [PDF](https://arxiv.org/pdf/2508.01752)  

**Abstract**: Activity and behaviour correlate with dairy cow health and welfare, making continual and accurate monitoring crucial for disease identification and farm productivity. Manual observation and frequent assessments are laborious and inconsistent for activity monitoring. In this study, we developed a unique multi-camera, real-time tracking system for indoor-housed Holstein Friesian dairy cows. This technology uses cutting-edge computer vision techniques, including instance segmentation and tracking algorithms to monitor cow activity seamlessly and accurately. An integrated top-down barn panorama was created by geometrically aligning six camera feeds using homographic transformations. The detection phase used a refined YOLO11-m model trained on an overhead cow dataset, obtaining high accuracy (mAP\@0.50 = 0.97, F1 = 0.95). SAMURAI, an upgraded Segment Anything Model 2.1, generated pixel-precise cow masks for instance segmentation utilizing zero-shot learning and motion-aware memory. Even with occlusion and fluctuating posture, a motion-aware Linear Kalman filter and IoU-based data association reliably identified cows over time for object tracking. The proposed system significantly outperformed Deep SORT Realtime. Multi-Object Tracking Accuracy (MOTA) was 98.7% and 99.3% in two benchmark video sequences, with IDF1 scores above 99% and near-zero identity switches. This unified multi-camera system can track dairy cows in complex interior surroundings in real time, according to our data. The system reduces redundant detections across overlapping cameras, maintains continuity as cows move between viewpoints, with the aim of improving early sickness prediction through activity quantification and behavioural classification. 

---
# Improving Noise Efficiency in Privacy-preserving Dataset Distillation 

**Authors**: Runkai Zheng, Vishnu Asutosh Dasu, Yinong Oliver Wang, Haohan Wang, Fernando De la Torre  

**Link**: [PDF](https://arxiv.org/pdf/2508.01749)  

**Abstract**: Modern machine learning models heavily rely on large datasets that often include sensitive and private information, raising serious privacy concerns. Differentially private (DP) data generation offers a solution by creating synthetic datasets that limit the leakage of private information within a predefined privacy budget; however, it requires a substantial amount of data to achieve performance comparable to models trained on the original data. To mitigate the significant expense incurred with synthetic data generation, Dataset Distillation (DD) stands out for its remarkable training and storage efficiency. This efficiency is particularly advantageous when integrated with DP mechanisms, curating compact yet informative synthetic datasets without compromising privacy. However, current state-of-the-art private DD methods suffer from a synchronized sampling-optimization process and the dependency on noisy training signals from randomly initialized networks. This results in the inefficient utilization of private information due to the addition of excessive noise. To address these issues, we introduce a novel framework that decouples sampling from optimization for better convergence and improves signal quality by mitigating the impact of DP noise through matching in an informative subspace. On CIFAR-10, our method achieves a \textbf{10.0\%} improvement with 50 images per class and \textbf{8.3\%} increase with just \textbf{one-fifth} the distilled set size of previous state-of-the-art methods, demonstrating significant potential to advance privacy-preserving DD. 

---
# Intention-Guided Cognitive Reasoning for Egocentric Long-Term Action Anticipation 

**Authors**: Qiaohui Chu, Haoyu Zhang, Meng Liu, Yisen Feng, Haoxiang Shi, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2508.01742)  

**Abstract**: Long-term action anticipation from egocentric video is critical for applications such as human-computer interaction and assistive technologies, where anticipating user intent enables proactive and context-aware AI assistance. However, existing approaches suffer from three key limitations: 1) underutilization of fine-grained visual cues from hand-object interactions, 2) neglect of semantic dependencies between verbs and nouns, and 3) lack of explicit cognitive reasoning, limiting generalization and long-term forecasting ability. To overcome these challenges, we propose INSIGHT, a unified two-stage framework for egocentric action anticipation. In the first stage, INSIGHT focuses on extracting semantically rich features from hand-object interaction regions and enhances action representations using a verb-noun co-occurrence matrix. In the second stage, it introduces a reinforcement learning-based module that simulates explicit cognitive reasoning through a structured process: visual perception (think) -> intention inference (reason) -> action anticipation (answer). Extensive experiments on Ego4D, EPIC-Kitchens-55, and EGTEA Gaze+ benchmarks show that INSIGHT achieves state-of-the-art performance, demonstrating its effectiveness and strong generalization capability. 

---
# Granular Concept Circuits: Toward a Fine-Grained Circuit Discovery for Concept Representations 

**Authors**: Dahee Kwon, Sehyun Lee, Jaesik Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01728)  

**Abstract**: Deep vision models have achieved remarkable classification performance by leveraging a hierarchical architecture in which human-interpretable concepts emerge through the composition of individual neurons across layers. Given the distributed nature of representations, pinpointing where specific visual concepts are encoded within a model remains a crucial yet challenging task. In this paper, we introduce an effective circuit discovery method, called Granular Concept Circuit (GCC), in which each circuit represents a concept relevant to a given query. To construct each circuit, our method iteratively assesses inter-neuron connectivity, focusing on both functional dependencies and semantic alignment. By automatically discovering multiple circuits, each capturing specific concepts within that query, our approach offers a profound, concept-wise interpretation of models and is the first to identify circuits tied to specific visual concepts at a fine-grained level. We validate the versatility and effectiveness of GCCs across various deep image classification models. 

---
# Dynamic Robot-Assisted Surgery with Hierarchical Class-Incremental Semantic Segmentation 

**Authors**: Julia Hindel, Ema Mekic, Enamundram Naga Karthik, Rohit Mohan, Daniele Cattaneo, Maria Kalweit, Abhinav Valada  

**Link**: [PDF](https://arxiv.org/pdf/2508.01713)  

**Abstract**: Robot-assisted surgeries rely on accurate and real-time scene understanding to safely guide surgical instruments. However, segmentation models trained on static datasets face key limitations when deployed in these dynamic and evolving surgical environments. Class-incremental semantic segmentation (CISS) allows models to continually adapt to new classes while avoiding catastrophic forgetting of prior knowledge, without training on previous data. In this work, we build upon the recently introduced Taxonomy-Oriented Poincaré-regularized Incremental Class Segmentation (TOPICS) approach and propose an enhanced variant, termed TOPICS+, specifically tailored for robust segmentation of surgical scenes. Concretely, we incorporate the Dice loss into the hierarchical loss formulation to handle strong class imbalances, introduce hierarchical pseudo-labeling, and design tailored label taxonomies for robotic surgery environments. We also propose six novel CISS benchmarks designed for robotic surgery environments including multiple incremental steps and several semantic categories to emulate realistic class-incremental settings in surgical environments. In addition, we introduce a refined set of labels with more than 144 classes on the Syn-Mediverse synthetic dataset, hosted online as an evaluation benchmark. We make the code and trained models publicly available at this http URL. 

---
# HateClipSeg: A Segment-Level Annotated Dataset for Fine-Grained Hate Video Detection 

**Authors**: Han Wang, Zhuoran Wang, Roy Ka-Wei Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.01712)  

**Abstract**: Detecting hate speech in videos remains challenging due to the complexity of multimodal content and the lack of fine-grained annotations in existing datasets. We present HateClipSeg, a large-scale multimodal dataset with both video-level and segment-level annotations, comprising over 11,714 segments labeled as Normal or across five Offensive categories: Hateful, Insulting, Sexual, Violence, Self-Harm, along with explicit target victim labels. Our three-stage annotation process yields high inter-annotator agreement (Krippendorff's alpha = 0.817). We propose three tasks to benchmark performance: (1) Trimmed Hateful Video Classification, (2) Temporal Hateful Video Localization, and (3) Online Hateful Video Classification. Results highlight substantial gaps in current models, emphasizing the need for more sophisticated multimodal and temporally aware approaches. The HateClipSeg dataset are publicly available at this https URL. 

---
# GAID: Frame-Level Gated Audio-Visual Integration with Directional Perturbation for Text-Video Retrieval 

**Authors**: Bowen Yang, Yun Cao, Chen He, Xiaosu Su  

**Link**: [PDF](https://arxiv.org/pdf/2508.01711)  

**Abstract**: Text-to-video retrieval requires precise alignment between language and temporally rich video signals. Existing methods predominantly exploit visual cues and often overlook complementary audio semantics or adopt coarse fusion strategies, leading to suboptimal multimodal representations. We present GAID, a framework that jointly address this gap via two key components: (i) a Frame-level Gated Fusion (FGF) that adaptively integrates audio and visual features under textual guidance, enabling fine-grained temporal alignment; and (ii) a Directional Adaptive Semantic Perturbation (DASP) that injects structure-aware perturbations into text embeddings, enhancing robustness and discrimination without incurring multi-pass inference. These modules complement each other -- fusion reduces modality gaps while perturbation regularizes cross-modal matching -- yielding more stable and expressive representations. Extensive experiments on MSR-VTT, DiDeMo, LSMDC, and VATEX show consistent state-of-the-art results across all retrieval metrics with notable efficiency gains. Our code is available at this https URL. 

---
# MHARFedLLM: Multimodal Human Activity Recognition Using Federated Large Language Model 

**Authors**: Asmit Bandyopadhyay, Rohit Basu, Tanmay Sen, Swagatam Das  

**Link**: [PDF](https://arxiv.org/pdf/2508.01701)  

**Abstract**: Human Activity Recognition (HAR) plays a vital role in applications such as fitness tracking, smart homes, and healthcare monitoring. Traditional HAR systems often rely on single modalities, such as motion sensors or cameras, limiting robustness and accuracy in real-world environments. This work presents FedTime-MAGNET, a novel multimodal federated learning framework that advances HAR by combining heterogeneous data sources: depth cameras, pressure mats, and accelerometers. At its core is the Multimodal Adaptive Graph Neural Expert Transformer (MAGNET), a fusion architecture that uses graph attention and a Mixture of Experts to generate unified, discriminative embeddings across modalities. To capture complex temporal dependencies, a lightweight T5 encoder only architecture is customized and adapted within this framework. Extensive experiments show that FedTime-MAGNET significantly improves HAR performance, achieving a centralized F1 Score of 0.934 and a strong federated F1 Score of 0.881. These results demonstrate the effectiveness of combining multimodal fusion, time series LLMs, and federated learning for building accurate and robust HAR systems. 

---
# Collaborative Chain-of-Agents for Parametric-Retrieved Knowledge Synergy 

**Authors**: Yi Jiang, Sendong Zhao, Jianbo Li, Haochun Wang, Lizhe Zhang, Yan Liu, Bin Qin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01696)  

**Abstract**: Retrieval-Augmented Generation (RAG) has emerged as a promising framework for enhancing the capabilities of Large Language Models (LLMs), especially in knowledge-intensive tasks. Despite its advantages, current RAG methods often struggle to *fully exploit knowledge during generation*. In particular, the synergy between the model's internal parametric knowledge and external retrieved knowledge remains limited. Retrieved contents may sometimes mislead generation, while certain generated content can guide the model toward more accurate outputs. In this work, we propose Collaborative Chain-of-Agents, a framework designed to enhance explicitly synergy over both parametric and retrieved knowledge. Specifically, we first introduce CoCoA-zero, a multi-agent RAG framework that first performs conditional knowledge induction and then reasons answers. Building on this, we develop CoCoA, a long-chain training strategy that synthesizes extended multi-agent reasoning trajectories from CoCoA-zero to fine-tune the LLM. This strategy enhances the model's capability to explicitly integrate and jointly leverage parametric and retrieved knowledge. Experiments results show that CoCoA-zero and CoCoA achieve superior performance on open-domain and multi-hop QA tasks. 

---
# From SHAP to Rules: Distilling Expert Knowledge from Post-hoc Model Explanations in Time Series Classification 

**Authors**: Maciej Mozolewski, Szymon Bobek, Grzegorz J. Nalepa  

**Link**: [PDF](https://arxiv.org/pdf/2508.01687)  

**Abstract**: Explaining machine learning (ML) models for time series (TS) classification is challenging due to inherent difficulty in raw time series interpretation and doubled down by the high dimensionality. We propose a framework that converts numeric feature attributions from post-hoc, instance-wise explainers (e.g., LIME, SHAP) into structured, human-readable rules. These rules define intervals indicating when and where they apply, improving transparency. Our approach performs comparably to native rule-based methods like Anchor while scaling better to long TS and covering more instances. Rule fusion integrates rule sets through methods such as weighted selection and lasso-based refinement to balance coverage, confidence, and simplicity, ensuring all instances receive an unambiguous, metric-optimized rule. It enhances explanations even for a single explainer. We introduce visualization techniques to manage specificity-generalization trade-offs. By aligning with expert-system principles, our framework consolidates conflicting or overlapping explanations - often resulting from the Rashomon effect - into coherent and domain-adaptable insights. Experiments on UCI datasets confirm that the resulting rule-based representations improve interpretability, decision transparency, and practical applicability for TS classification. 

---
# Cure or Poison? Embedding Instructions Visually Alters Hallucination in Vision-Language Models 

**Authors**: Zhaochen Wang, Yiwei Wang, Yujun Cai  

**Link**: [PDF](https://arxiv.org/pdf/2508.01678)  

**Abstract**: Vision-Language Models (VLMs) often suffer from hallucination, partly due to challenges in aligning multimodal information. We propose Prompt-in-Image, a simple method that embeds textual instructions directly into images. This removes the need for separate text inputs and forces the model to process all content through the visual channel. We evaluate this method on three popular open-source VLMs: Qwen2.5-VL, LLaVA-1.5, and InstructBLIP. The results reveal sharp differences. Prompt-in-Image improves Qwen2.5-VL's performance, increasing POPE accuracy by 4.1 percent (from 80.2 percent to 84.3 percent) and also reducing hallucination rates on MS-COCO. In contrast, LLaVA-1.5 and InstructBLIP experience a severe performance drop, with accuracy falling from around 84 percent to near-random levels. Through detailed analysis, we found that CLIP-based encoders in LLaVA and InstructBLIP exhibit excessive attention bias toward embedded text regions, disrupting visual understanding. In contrast, Qwen's vision encoder handles text-embedded images robustly. Crucially, Prompt-in-Image reduces Qwen's modality gap, enhancing cross-modal alignment by unifying information processing through a single modality. 

---
# CUPID: Evaluating Personalized and Contextualized Alignment of LLMs from Interactions 

**Authors**: Tae Soo Kim, Yoonjoo Lee, Yoonah Park, Jiho Kim, Young-Ho Kim, Juho Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01674)  

**Abstract**: Personalization of Large Language Models (LLMs) often assumes users hold static preferences that reflect globally in all tasks. In reality, humans hold dynamic preferences that change depending on the context. As users interact with an LLM in various contexts, they naturally reveal their contextual preferences, which a model must infer and apply in future contexts to ensure alignment. To assess this, we introduce CUPID, a benchmark of 756 human-curated interaction session histories between users and LLM-based chat assistants. In each interaction session, the user provides a request in a specific context and expresses their preference through multi-turn feedback. Given a new user request and prior interaction sessions, our benchmark assesses whether LLMs can infer the preference relevant to this request and generate a response that satisfies this preference. With CUPID, we evaluated 10 open and proprietary LLMs, revealing that state-of-the-art LLMs struggle to infer preferences from multi-turn interactions and fail to discern what previous context is relevant to a new request -- under 50% precision and 65% recall. Our work highlights the need to advance LLM capabilities for more contextually personalized interactions and proposes CUPID as a resource to drive these improvements. 

---
# Authorship Attribution in Multilingual Machine-Generated Texts 

**Authors**: Lucio La Cava, Dominik Macko, Róbert Móro, Ivan Srba, Andrea Tagarelli  

**Link**: [PDF](https://arxiv.org/pdf/2508.01656)  

**Abstract**: As Large Language Models (LLMs) have reached human-like fluency and coherence, distinguishing machine-generated text (MGT) from human-written content becomes increasingly difficult. While early efforts in MGT detection have focused on binary classification, the growing landscape and diversity of LLMs require a more fine-grained yet challenging authorship attribution (AA), i.e., being able to identify the precise generator (LLM or human) behind a text. However, AA remains nowadays confined to a monolingual setting, with English being the most investigated one, overlooking the multilingual nature and usage of modern LLMs. In this work, we introduce the problem of Multilingual Authorship Attribution, which involves attributing texts to human or multiple LLM generators across diverse languages. Focusing on 18 languages -- covering multiple families and writing scripts -- and 8 generators (7 LLMs and the human-authored class), we investigate the multilingual suitability of monolingual AA methods, their cross-lingual transferability, and the impact of generators on attribution performance. Our results reveal that while certain monolingual AA methods can be adapted to multilingual settings, significant limitations and challenges remain, particularly in transferring across diverse language families, underscoring the complexity of multilingual AA and the need for more robust approaches to better match real-world scenarios. 

---
# MAP: Mitigating Hallucinations in Large Vision-Language Models with Map-Level Attention Processing 

**Authors**: Chenxi Li, Yichen Guo, Benfang Qian, Jinhao You, Kai Tang, Yaosong Du, Zonghao Zhang, Xiande Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01653)  

**Abstract**: Large Vision-Language Models (LVLMs) have achieved impressive performance in multimodal tasks, but they still suffer from hallucinations, i.e., generating content that is grammatically accurate but inconsistent with visual inputs. In this work, we introduce a novel map-level perspective to mitigate hallucinations in LVLMs, interpreting the hidden states of the model as a 2D semantic map. We observe that factual information is widely distributed across this map, extending beyond the localized inter- or intra-layer regions targeted by most existing methods (e.g., contrastive decoding and layer-wise consistency). Building on this insight, we propose Map-Level Attention Processing (MAP), a training-free decoding method that effectively leverages factual information through attention-based map-level operations to improve factual consistency. Specifically, we employ Layer-Wise Criss-Cross Attention to progressively refine token representations at each decoding layer by aggregating tokens from both inter- and intra-layer dimensions. Additionally, a Global-Local Logit Fusion mechanism combines logits obtained before and after global attention to further refine predictions and improve accuracy. Our method consistently improves the truthfulness and performance of LVLMs across benchmarks, such as POPE, MME, and MMHal-Bench, demonstrating the potential of the map-level decoding strategy. 

---
# DUP: Detection-guided Unlearning for Backdoor Purification in Language Models 

**Authors**: Man Hu, Yahui Ding, Yatao Yang, Liangyu Chen, Yanhao Jia, Shuai Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01647)  

**Abstract**: As backdoor attacks become more stealthy and robust, they reveal critical weaknesses in current defense strategies: detection methods often rely on coarse-grained feature statistics, and purification methods typically require full retraining or additional clean models. To address these challenges, we propose DUP (Detection-guided Unlearning for Purification), a unified framework that integrates backdoor detection with unlearning-based purification. The detector captures feature-level anomalies by jointly leveraging class-agnostic distances and inter-layer transitions. These deviations are integrated through a weighted scheme to identify poisoned inputs, enabling more fine-grained analysis. Based on the detection results, we purify the model through a parameter-efficient unlearning mechanism that avoids full retraining and does not require any external clean model. Specifically, we innovatively repurpose knowledge distillation to guide the student model toward increasing its output divergence from the teacher on detected poisoned samples, effectively forcing it to unlearn the backdoor behavior. Extensive experiments across diverse attack methods and language model architectures demonstrate that DUP achieves superior defense performance in detection accuracy and purification efficacy. Our code is available at this https URL. 

---
# SPARTA: Advancing Sparse Attention in Spiking Neural Networks via Spike-Timing-Based Prioritization 

**Authors**: Minsuk Jang, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01646)  

**Abstract**: Current Spiking Neural Networks (SNNs) underutilize the temporal dynamics inherent in spike-based processing, relying primarily on rate coding while overlooking precise timing information that provides rich computational cues. We propose SPARTA (Spiking Priority Attention with Resource-Adaptive Temporal Allocation), a framework that leverages heterogeneous neuron dynamics and spike-timing information to enable efficient sparse attention. SPARTA prioritizes tokens based on temporal cues, including firing patterns, spike timing, and inter-spike intervals, achieving 65.4% sparsity through competitive gating. By selecting only the most salient tokens, SPARTA reduces attention complexity from O(N^2) to O(K^2) with k << n, while maintaining high accuracy. Our method achieves state-of-the-art performance on DVS-Gesture (98.78%) and competitive results on CIFAR10-DVS (83.06%) and CIFAR-10 (95.3%), demonstrating that exploiting spike timing dynamics improves both computational efficiency and accuracy. 

---
# Semantic Encryption: Secure and Effective Interaction with Cloud-based Large Language Models via Semantic Transformation 

**Authors**: Dong Chen, Tong Yang, Feipeng Zhai, Pengpeng Ouyang, Qidong Liu, Yafei Li, Chong Fu, Mingliang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01638)  

**Abstract**: The increasing adoption of Cloud-based Large Language Models (CLLMs) has raised significant concerns regarding data privacy during user interactions. While existing approaches primarily focus on encrypting sensitive information, they often overlook the logical structure of user inputs. This oversight can lead to reduced data utility and degraded performance of CLLMs. To address these limitations and enable secure yet effective interactions, we propose Semantic Encryption (SE)-a plug-and-play framework designed to preserve both privacy and utility. SE consists of two key components: Semantic Encoding and Semantic Decoding. In the encoding phase, a lightweight local model transforms the original user input into an alternative semantic context that maintains the original intent and logical structure while obfuscating sensitive information. This transformed input is then processed by the CLLM, which generates a response based on the transformed semantic context. To maintain a seamless user experience, the decoding phase will reconstruct the CLLM's response back into the original semantic context by referencing the locally stored user input. Extensive experimental evaluations demonstrate that SE effectively protects data privacy without compromising data utility or user experience, offering a practical solution for secure interaction with CLLMs. Particularly, the proposed SE demonstrates a significant improvement over the state-of-the-art InferDPT, surpassing it across various evaluated metrics and datasets. 

---
# Learning Unified System Representations for Microservice Tail Latency Prediction 

**Authors**: Wenzhuo Qian, Hailiang Zhao, Tianlv Chen, Jiayi Chen, Ziqi Wang, Kingsum Chow, Shuiguang Deng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01635)  

**Abstract**: Microservice architectures have become the de facto standard for building scalable cloud-native applications, yet their distributed nature introduces significant challenges in performance monitoring and resource management. Traditional approaches often rely on per-request latency metrics, which are highly sensitive to transient noise and fail to reflect the holistic behavior of complex, concurrent workloads. In contrast, window-level P95 tail latency provides a stable and meaningful signal that captures both system-wide trends and user-perceived performance degradation. We identify two key shortcomings in existing methods: (i) inadequate handling of heterogeneous data, where traffic-side features propagate across service dependencies and resource-side signals reflect localized bottlenecks, and (ii) the lack of principled architectural designs that effectively distinguish and integrate these complementary modalities. To address these challenges, we propose USRFNet, a deep learning network that explicitly separates and models traffic-side and resource-side features. USRFNet employs GNNs to capture service interactions and request propagation patterns, while gMLP modules independently model cluster resource dynamics. These representations are then fused into a unified system embedding to predict window-level P95 latency with high accuracy. We evaluate USRFNet on real-world microservice benchmarks under large-scale stress testing conditions, demonstrating substantial improvements in prediction accuracy over state-of-the-art baselines. 

---
# OpenMed NER: Open-Source, Domain-Adapted State-of-the-Art Transformers for Biomedical NER Across 12 Public Datasets 

**Authors**: Maziyar Panahi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01630)  

**Abstract**: Named-entity recognition (NER) is fundamental to extracting structured information from the >80% of healthcare data that resides in unstructured clinical notes and biomedical literature. Despite recent advances with large language models, achieving state-of-the-art performance across diverse entity types while maintaining computational efficiency remains a significant challenge. We introduce OpenMed NER, a suite of open-source, domain-adapted transformer models that combine lightweight domain-adaptive pre-training (DAPT) with parameter-efficient Low-Rank Adaptation (LoRA). Our approach performs cost-effective DAPT on a 350k-passage corpus compiled from ethically sourced, publicly available research repositories and de-identified clinical notes (PubMed, arXiv, and MIMIC-III) using DeBERTa-v3, PubMedBERT, and BioELECTRA backbones. This is followed by task-specific fine-tuning with LoRA, which updates less than 1.5% of model parameters. We evaluate our models on 12 established biomedical NER benchmarks spanning chemicals, diseases, genes, and species. OpenMed NER achieves new state-of-the-art micro-F1 scores on 10 of these 12 datasets, with substantial gains across diverse entity types. Our models advance the state-of-the-art on foundational disease and chemical benchmarks (e.g., BC5CDR-Disease, +2.70 pp), while delivering even larger improvements of over 5.3 and 9.7 percentage points on more specialized gene and clinical cell line corpora. This work demonstrates that strategically adapted open-source models can surpass closed-source solutions. This performance is achieved with remarkable efficiency: training completes in under 12 hours on a single GPU with a low carbon footprint (< 1.2 kg CO2e), producing permissively licensed, open-source checkpoints designed to help practitioners facilitate compliance with emerging data protection and AI regulations, such as the EU AI Act. 

---
# EAC-MoE: Expert-Selection Aware Compressor for Mixture-of-Experts Large Language Models 

**Authors**: Yuanteng Chen, Yuantian Shao, Peisong Wang, Jian Cheng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01625)  

**Abstract**: Mixture-of-Experts (MoE) has demonstrated promising potential in scaling LLMs. However, it is hindered by two critical challenges: (1) substantial GPU memory consumption to load all experts; (2) low activated parameters cannot be equivalently translated into inference acceleration effects. In this work, we propose EAC-MoE, an Expert-Selection Aware Compressor for MoE-LLMs, which deeply aligns with the characteristics of MoE from the perspectives of quantization and pruning, and introduces two modules to address these two challenges respectively: (1) The expert selection bias caused by low-bit quantization is a major factor contributing to the performance degradation in MoE-LLMs. Based on this, we propose Quantization with Expert-Selection Calibration (QESC), which mitigates the expert selection bias by calibrating the routers within the MoE; (2) There are always certain experts that are not crucial for the corresponding tasks, yet causing inference latency. Therefore, we propose Pruning based on Expert-Selection Frequency (PESF), which significantly improves inference speed by pruning less frequently used experts for current task. Extensive experiments demonstrate that our approach significantly reduces memory usage and improves inference speed with minimal performance degradation. 

---
# TCDiff: Triplex Cascaded Diffusion for High-fidelity Multimodal EHRs Generation with Incomplete Clinical Data 

**Authors**: Yandong Yan, Chenxi Li, Yu Huang, Dexuan Xu, Jiaqi Zhu, Zhongyan Chai, Huamin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01615)  

**Abstract**: The scarcity of large-scale and high-quality electronic health records (EHRs) remains a major bottleneck in biomedical research, especially as large foundation models become increasingly data-hungry. Synthesizing substantial volumes of de-identified and high-fidelity data from existing datasets has emerged as a promising solution. However, existing methods suffer from a series of limitations: they struggle to model the intrinsic properties of heterogeneous multimodal EHR data (e.g., continuous, discrete, and textual modalities), capture the complex dependencies among them, and robustly handle pervasive data incompleteness. These challenges are particularly acute in Traditional Chinese Medicine (TCM). To this end, we propose TCDiff (Triplex Cascaded Diffusion Network), a novel EHR generation framework that cascades three diffusion networks to learn the features of real-world EHR data, formatting a multi-stage generative process: Reference Modalities Diffusion, Cross-Modal Bridging, and Target Modality Diffusion. Furthermore, to validate our proposed framework, besides two public datasets, we also construct and introduce TCM-SZ1, a novel multimodal EHR dataset for benchmarking. Experimental results show that TCDiff consistently outperforms state-of-the-art baselines by an average of 10% in data fidelity under various missing rate, while maintaining competitive privacy guarantees. This highlights the effectiveness, robustness, and generalizability of our approach in real-world healthcare scenarios. 

---
# Augmented Reinforcement Learning Framework For Enhancing Decision-Making In Machine Learning Models Using External Agents 

**Authors**: Sandesh Kumar Singh  

**Link**: [PDF](https://arxiv.org/pdf/2508.01612)  

**Abstract**: This work proposes a novel technique Augmented Reinforcement Learning framework for the improvement of decision-making capabilities of machine learning models. The introduction of agents as external overseers checks on model decisions. The external agent can be anyone, like humans or automated scripts, that helps in decision path correction. It seeks to ascertain the priority of the "Garbage-In, Garbage-Out" problem that caused poor data inputs or incorrect actions in reinforcement learning. The ARL framework incorporates two external agents that aid in course correction and the guarantee of quality data at all points of the training cycle. The External Agent 1 is a real-time evaluator, which will provide feedback light of decisions taken by the model, identify suboptimal actions forming the Rejected Data Pipeline. The External Agent 2 helps in selective curation of the provided feedback with relevance and accuracy in business scenarios creates an approved dataset for future training cycles. The validation of the framework is also applied to a real-world scenario, which is "Document Identification and Information Extraction". This problem originates mainly from banking systems, but can be extended anywhere. The method of classification and extraction of information has to be done correctly here. Experimental results show that including human feedback significantly enhances the ability of the model in order to increase robustness and accuracy in making decisions. The augmented approach, with a combination of machine efficiency and human insight, attains a higher learning standard-mainly in complex or ambiguous environments. The findings of this study show that human-in-the-loop reinforcement learning frameworks such as ARL can provide a scalable approach to improving model performance in data-driven applications. 

---
# Drift-aware Collaborative Assistance Mixture of Experts for Heterogeneous Multistream Learning 

**Authors**: En Yu, Jie Lu, Kun Wang, Xiaoyu Yang, Guangquan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01598)  

**Abstract**: Learning from multiple data streams in real-world scenarios is fundamentally challenging due to intrinsic heterogeneity and unpredictable concept drifts. Existing methods typically assume homogeneous streams and employ static architectures with indiscriminate knowledge fusion, limiting generalizability in complex dynamic environments. To tackle this gap, we propose CAMEL, a dynamic \textbf{C}ollaborative \textbf{A}ssistance \textbf{M}ixture of \textbf{E}xperts \textbf{L}earning framework. It addresses heterogeneity by assigning each stream an independent system with a dedicated feature extractor and task-specific head. Meanwhile, a dynamic pool of specialized private experts captures stream-specific idiosyncratic patterns. Crucially, collaboration across these heterogeneous streams is enabled by a dedicated assistance expert. This expert employs a multi-head attention mechanism to distill and integrate relevant context autonomously from all other concurrent streams. It facilitates targeted knowledge transfer while inherently mitigating negative transfer from irrelevant sources. Furthermore, we propose an Autonomous Expert Tuner (AET) strategy, which dynamically manages expert lifecycles in response to drift. It instantiates new experts for emerging concepts (freezing prior ones to prevent catastrophic forgetting) and prunes obsolete ones. This expert-level plasticity provides a robust and efficient mechanism for online model capacity adaptation. Extensive experiments demonstrate CAMEL's superior generalizability across diverse multistreams and exceptional resilience against complex concept drifts. 

---
# DMTrack: Spatio-Temporal Multimodal Tracking via Dual-Adapter 

**Authors**: Weihong Li, Shaohua Dong, Haonan Lu, Yanhao Zhang, Heng Fan, Libo Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01592)  

**Abstract**: In this paper, we explore adapter tuning and introduce a novel dual-adapter architecture for spatio-temporal multimodal tracking, dubbed DMTrack. The key of our DMTrack lies in two simple yet effective modules, including a spatio-temporal modality adapter (STMA) and a progressive modality complementary adapter (PMCA) module. The former, applied to each modality alone, aims to adjust spatio-temporal features extracted from a frozen backbone by self-prompting, which to some extent can bridge the gap between different modalities and thus allows better cross-modality fusion. The latter seeks to facilitate cross-modality prompting progressively with two specially designed pixel-wise shallow and deep adapters. The shallow adapter employs shared parameters between the two modalities, aiming to bridge the information flow between the two modality branches, thereby laying the foundation for following modality fusion, while the deep adapter modulates the preliminarily fused information flow with pixel-wise inner-modal attention and further generates modality-aware prompts through pixel-wise inter-modal attention. With such designs, DMTrack achieves promising spatio-temporal multimodal tracking performance with merely \textbf{0.93M} trainable parameters. Extensive experiments on five benchmarks show that DMTrack achieves state-of-the-art results. Code will be available. 

---
# Censored Sampling for Topology Design: Guiding Diffusion with Human Preferences 

**Authors**: Euihyun Kim, Keun Park, Yeoneung Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01589)  

**Abstract**: Recent advances in denoising diffusion models have enabled rapid generation of optimized structures for topology optimization. However, these models often rely on surrogate predictors to enforce physical constraints, which may fail to capture subtle yet critical design flaws such as floating components or boundary discontinuities that are obvious to human experts. In this work, we propose a novel human-in-the-loop diffusion framework that steers the generative process using a lightweight reward model trained on minimal human feedback. Inspired by preference alignment techniques in generative modeling, our method learns to suppress unrealistic outputs by modulating the reverse diffusion trajectory using gradients of human-aligned rewards. Specifically, we collect binary human evaluations of generated topologies and train classifiers to detect floating material and boundary violations. These reward models are then integrated into the sampling loop of a pre-trained diffusion generator, guiding it to produce designs that are not only structurally performant but also physically plausible and manufacturable. Our approach is modular and requires no retraining of the diffusion model. Preliminary results show substantial reductions in failure modes and improved design realism across diverse test conditions. This work bridges the gap between automated design generation and expert judgment, offering a scalable solution to trustworthy generative design. 

---
# Diffusion Models for Future Networks and Communications: A Comprehensive Survey 

**Authors**: Nguyen Cong Luong, Nguyen Duc Hai, Duc Van Le, Huy T. Nguyen, Thai-Hoc Vu, Thien Huynh-The, Ruichen Zhang, Nguyen Duc Duy Anh, Dusit Niyato, Marco Di Renzo, Dong In Kim, Quoc-Viet Pham  

**Link**: [PDF](https://arxiv.org/pdf/2508.01586)  

**Abstract**: The rise of Generative AI (GenAI) in recent years has catalyzed transformative advances in wireless communications and networks. Among the members of the GenAI family, Diffusion Models (DMs) have risen to prominence as a powerful option, capable of handling complex, high-dimensional data distribution, as well as consistent, noise-robust performance. In this survey, we aim to provide a comprehensive overview of the theoretical foundations and practical applications of DMs across future communication systems. We first provide an extensive tutorial of DMs and demonstrate how they can be applied to enhance optimizers, reinforcement learning and incentive mechanisms, which are popular approaches for problems in wireless networks. Then, we review and discuss the DM-based methods proposed for emerging issues in future networks and communications, including channel modeling and estimation, signal detection and data reconstruction, integrated sensing and communication, resource management in edge computing networks, semantic communications and other notable issues. We conclude the survey with highlighting technical limitations of DMs and their applications, as well as discussing future research directions. 

---
# Deeply Supervised Multi-Task Autoencoder for Biological Brain Age estimation using three dimensional T$_1$-weighted magnetic resonance imaging 

**Authors**: Mehreen Kanwal, Yunsik Son  

**Link**: [PDF](https://arxiv.org/pdf/2508.01565)  

**Abstract**: Accurate estimation of biological brain age from three dimensional (3D) T$_1$-weighted magnetic resonance imaging (MRI) is a critical imaging biomarker for identifying accelerated aging associated with neurodegenerative diseases. Effective brain age prediction necessitates training 3D models to leverage comprehensive insights from volumetric MRI scans, thereby fully capturing spatial anatomical context. However, optimizing deep 3D models remains challenging due to problems such as vanishing gradients. Furthermore, brain structural patterns differ significantly between sexes, which impacts aging trajectories and vulnerability to neurodegenerative diseases, thereby making sex classification crucial for enhancing the accuracy and generalizability of predictive models. To address these challenges, we propose a Deeply Supervised Multitask Autoencoder (DSMT-AE) framework for brain age estimation. DSMT-AE employs deep supervision, which involves applying supervisory signals at intermediate layers during training, to stabilize model optimization, and multitask learning to enhance feature representation. Specifically, our framework simultaneously optimizes brain age prediction alongside auxiliary tasks of sex classification and image reconstruction, thus effectively capturing anatomical and demographic variability to improve prediction accuracy. We extensively evaluate DSMT-AE on the Open Brain Health Benchmark (OpenBHB) dataset, the largest multisite neuroimaging cohort combining ten publicly available datasets. The results demonstrate that DSMT-AE achieves state-of-the-art performance and robustness across age and sex subgroups. Additionally, our ablation study confirms that each proposed component substantially contributes to the improved predictive accuracy and robustness of the overall architecture. 

---
# Are All Prompt Components Value-Neutral? Understanding the Heterogeneous Adversarial Robustness of Dissected Prompt in Large Language Models 

**Authors**: Yujia Zheng, Tianhao Li, Haotian Huang, Tianyu Zeng, Jingyu Lu, Chuangxin Chu, Yuekai Huang, Ziyou Jiang, Qian Xiong, Yuyao Ge, Mingyang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01554)  

**Abstract**: Prompt-based adversarial attacks have become an effective means to assess the robustness of large language models (LLMs). However, existing approaches often treat prompts as monolithic text, overlooking their structural heterogeneity-different prompt components contribute unequally to adversarial robustness. Prior works like PromptRobust assume prompts are value-neutral, but our analysis reveals that complex, domain-specific prompts with rich structures have components with differing vulnerabilities. To address this gap, we introduce PromptAnatomy, an automated framework that dissects prompts into functional components and generates diverse, interpretable adversarial examples by selectively perturbing each component using our proposed method, ComPerturb. To ensure linguistic plausibility and mitigate distribution shifts, we further incorporate a perplexity (PPL)-based filtering mechanism. As a complementary resource, we annotate four public instruction-tuning datasets using the PromptAnatomy framework, verified through human review. Extensive experiments across these datasets and five advanced LLMs demonstrate that ComPerturb achieves state-of-the-art attack success rates. Ablation studies validate the complementary benefits of prompt dissection and PPL filtering. Our results underscore the importance of prompt structure awareness and controlled perturbation for reliable adversarial robustness evaluation in LLMs. Code and data are available at this https URL. 

---
# Understanding Why ChatGPT Outperforms Humans in Visualization Design Advice 

**Authors**: Yongsu Ahn, Nam Wook Kim  

**Link**: [PDF](https://arxiv.org/pdf/2508.01547)  

**Abstract**: This paper investigates why recent generative AI models outperform humans in data visualization knowledge tasks. Through systematic comparative analysis of responses to visualization questions, we find that differences exist between two ChatGPT models and human outputs over rhetorical structure, knowledge breadth, and perceptual quality. Our findings reveal that ChatGPT-4, as a more advanced model, displays a hybrid of characteristics from both humans and ChatGPT-3.5. The two models were generally favored over human responses, while their strengths in coverage and breadth, and emphasis on technical and task-oriented visualization feedback collectively shaped higher overall quality. Based on our findings, we draw implications for advancing user experiences based on the potential of LLMs and human perception over their capabilities, with relevance to broader applications of AI. 

---
# Leveraging Machine Learning for Botnet Attack Detection in Edge-Computing Assisted IoT Networks 

**Authors**: Dulana Rupanetti, Naima Kaabouch  

**Link**: [PDF](https://arxiv.org/pdf/2508.01542)  

**Abstract**: The increase of IoT devices, driven by advancements in hardware technologies, has led to widespread deployment in large-scale networks that process massive amounts of data daily. However, the reliance on Edge Computing to manage these devices has introduced significant security vulnerabilities, as attackers can compromise entire networks by targeting a single IoT device. In light of escalating cybersecurity threats, particularly botnet attacks, this paper investigates the application of machine learning techniques to enhance security in Edge-Computing-Assisted IoT environments. Specifically, it presents a comparative analysis of Random Forest, XGBoost, and LightGBM -- three advanced ensemble learning algorithms -- to address the dynamic and complex nature of botnet threats. Utilizing a widely recognized IoT network traffic dataset comprising benign and malicious instances, the models were trained, tested, and evaluated for their accuracy in detecting and classifying botnet activities. Furthermore, the study explores the feasibility of deploying these models in resource-constrained edge and IoT devices, demonstrating their practical applicability in real-world scenarios. The results highlight the potential of machine learning to fortify IoT networks against emerging cybersecurity challenges. 

---
# MagicVL-2B: Empowering Vision-Language Models on Mobile Devices with Lightweight Visual Encoders via Curriculum Learning 

**Authors**: Yi Liu, Xiao Xu, Zeyu Xu, Meng Zhang, Yibo Li, Haoyu Chen, Junkang Zhang, Qiang Wang, Jifa Sun, Siling Lin, Shengxun Cheng, Lingshu Zhang, Kang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01540)  

**Abstract**: Vision-Language Models (VLMs) have achieved remarkable breakthroughs in recent years, enabling a diverse array of applications in everyday life. However, the substantial computational and storage demands of VLMs pose significant challenges for their efficient deployment on mobile devices, which represent the most ubiquitous and accessible computing platforms today. In this work, we introduce MagicVL-2B, a novel VLM meticulously optimized for flagship smartphones. MagicVL-2B leverages a lightweight visual encoder with fewer than 100M parameters and features a redesigned dynamic resolution scheme that adaptively generates image tokens without excessive modification of image dimensions. To further enhance the performance of this compact encoder within VLMs, we propose a multimodal curriculum learning strategy that incrementally increases task difficulty and data information density throughout training. This approach substantially improves the model's performance across a variety of sub-tasks. Extensive evaluations on standard VLM benchmarks demonstrate that MagicVL-2B matches the accuracy of current state-of-the-art models while reducing on-device power consumption by 41.1%. These results establish MagicVL-2B as a practical and robust solution for real-world mobile vision-language applications, enabling advanced multimodal intelligence to run directly on smartphones. 

---
# Revisiting Gossip Protocols: A Vision for Emergent Coordination in Agentic Multi-Agent Systems 

**Authors**: Mansura Habiba, Nafiul I. Khan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01531)  

**Abstract**: As agentic platforms scale, agents are evolving beyond static roles and fixed toolchains, creating a growing need for flexible, decentralized coordination. Today's structured communication protocols (e.g., direct agent-to-agent messaging) excel at reliability and task delegation, but they fall short in enabling emergent, swarm-like intelligence, where distributed agents continuously learn, adapt, and communicate to form collective cognition. This paper revisits gossip protocols, long valued in distributed systems for their fault tolerance and decentralization, and argues that they offer a missing layer for context-rich, adaptive communication in agentic AI. Gossip enables scalable, low-overhead dissemination of shared knowledge, but also raises unresolved challenges around semantic filtering, staleness, trustworthiness, and consistency in high-stakes environments. Rather than proposing a new framework, this work charts a research agenda for integrating gossip as a complementary substrate alongside structured protocols. We identify critical gaps in current agent-to-agent architectures, highlight where gossip could reshape assumptions about coordination, and outline open questions around intent propagation, knowledge decay, and peer-to-peer trust. Gossip is not a silver bullet, but overlooking it risks missing a key path toward resilient, reflexive, and self-organizing multi-agent systems. 

---
# MiraGe: Multimodal Discriminative Representation Learning for Generalizable AI-Generated Image Detection 

**Authors**: Kuo Shi, Jie Lu, Shanshan Ye, Guangquan Zhang, Zhen Fang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01525)  

**Abstract**: Recent advances in generative models have highlighted the need for robust detectors capable of distinguishing real images from AI-generated images. While existing methods perform well on known generators, their performance often declines when tested with newly emerging or unseen generative models due to overlapping feature embeddings that hinder accurate cross-generator classification. In this paper, we propose Multimodal Discriminative Representation Learning for Generalizable AI-generated Image Detection (MiraGe), a method designed to learn generator-invariant features. Motivated by theoretical insights on intra-class variation minimization and inter-class separation, MiraGe tightly aligns features within the same class while maximizing separation between classes, enhancing feature discriminability. Moreover, we apply multimodal prompt learning to further refine these principles into CLIP, leveraging text embeddings as semantic anchors for effective discriminative representation learning, thereby improving generalizability. Comprehensive experiments across multiple benchmarks show that MiraGe achieves state-of-the-art performance, maintaining robustness even against unseen generators like Sora. 

---
# Decentralized Aerial Manipulation of a Cable-Suspended Load using Multi-Agent Reinforcement Learning 

**Authors**: Jack Zeng, Andreu Matoses Gimenez, Eugene Vinitsky, Javier Alonso-Mora, Sihao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.01522)  

**Abstract**: This paper presents the first decentralized method to enable real-world 6-DoF manipulation of a cable-suspended load using a team of Micro-Aerial Vehicles (MAVs). Our method leverages multi-agent reinforcement learning (MARL) to train an outer-loop control policy for each MAV. Unlike state-of-the-art controllers that utilize a centralized scheme, our policy does not require global states, inter-MAV communications, nor neighboring MAV information. Instead, agents communicate implicitly through load pose observations alone, which enables high scalability and flexibility. It also significantly reduces computing costs during inference time, enabling onboard deployment of the policy. In addition, we introduce a new action space design for the MAVs using linear acceleration and body rates. This choice, combined with a robust low-level controller, enables reliable sim-to-real transfer despite significant uncertainties caused by cable tension during dynamic 3D motion. We validate our method in various real-world experiments, including full-pose control under load model uncertainties, showing setpoint tracking performance comparable to the state-of-the-art centralized method. We also demonstrate cooperation amongst agents with heterogeneous control policies, and robustness to the complete in-flight loss of one MAV. Videos of experiments: this https URL 

---
# The Vanishing Gradient Problem for Stiff Neural Differential Equations 

**Authors**: Colby Fronk, Linda Petzold  

**Link**: [PDF](https://arxiv.org/pdf/2508.01519)  

**Abstract**: Gradient-based optimization of neural differential equations and other parameterized dynamical systems fundamentally relies on the ability to differentiate numerical solutions with respect to model parameters. In stiff systems, it has been observed that sensitivities to parameters controlling fast-decaying modes become vanishingly small during training, leading to optimization difficulties. In this paper, we show that this vanishing gradient phenomenon is not an artifact of any particular method, but a universal feature of all A-stable and L-stable stiff numerical integration schemes. We analyze the rational stability function for general stiff integration schemes and demonstrate that the relevant parameter sensitivities, governed by the derivative of the stability function, decay to zero for large stiffness. Explicit formulas for common stiff integration schemes are provided, which illustrate the mechanism in detail. Finally, we rigorously prove that the slowest possible rate of decay for the derivative of the stability function is $O(|z|^{-1})$, revealing a fundamental limitation: all A-stable time-stepping methods inevitably suppress parameter gradients in stiff regimes, posing a significant barrier for training and parameter identification in stiff neural ODEs. 

---
# FlashSVD: Memory-Efficient Inference with Streaming for Low-Rank Models 

**Authors**: Zishan Shao, Yixiao Wang, Qinsi Wang, Ting Jiang, Zhixu Du, Hancheng Ye, Danyang Zhuo, Yiran Chen, Hai Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01506)  

**Abstract**: Singular Value Decomposition (SVD) has recently seen a surge of interest as a simple yet powerful tool for large language models (LLMs) compression, with a growing number of works demonstrating 20-80% parameter reductions at minimal accuracy loss. Previous SVD-based approaches have focused primarily on reducing the memory footprint of model weights, largely overlooking the additional activation memory overhead incurred during inference when applying truncated factors via standard dense CUDA kernels. Our experiments demonstrate that this activation overhead, scaling with sequence length and hidden dimension, prevents current SVD compression techniques from achieving any reduction in peak inference memory, thereby limiting their viability for real-world, on-device deployments.
We introduce FlashSVD, a novel, end-to-end rank-aware streaming inference framework specifically designed for SVD-compressed large language models. FlashSVD can be seamlessly integrated with any model that employs SVD-based methods for parameter reduction. By fusing low-rank projection kernels directly into both the self-attention and feed-forward network (FFN) pipelines, FlashSVD avoid materializing full-size activation buffers. Instead, small tiles of the truncated factors are loaded into on-chip SRAM, multiplied and reduced on the fly, and immediately evicted, preserving high GPU occupancy and adding no extra latency. On standard encoder benchmarks (e.g., BERT-Base), FlashSVD cuts peak activation memory by up to 70.2% and intermediate transient memory by 75%, all while incur no accuracy loss with upstreaming compression methods, offering a practical path toward memory-constrained deployment of low-rank LLMs. 

---
# ShrutiSense: Microtonal Modeling and Correction in Indian Classical Music 

**Authors**: Rajarshi Ghosh, Jayanth Athipatla  

**Link**: [PDF](https://arxiv.org/pdf/2508.01498)  

**Abstract**: Indian classical music relies on a sophisticated microtonal system of 22 shrutis (pitch intervals), which provides expressive nuance beyond the 12-tone equal temperament system. Existing symbolic music processing tools fail to account for these microtonal distinctions and culturally specific raga grammars that govern melodic movement. We present ShrutiSense, a comprehensive symbolic pitch processing system designed for Indian classical music, addressing two critical tasks: (1) correcting westernized or corrupted pitch sequences, and (2) completing melodic sequences with missing values. Our approach employs complementary models for different tasks: a Shruti-aware finite-state transducer (FST) that performs contextual corrections within the 22-shruti framework and a grammar-constrained Shruti hidden Markov model (GC-SHMM) that incorporates raga-specific transition rules for contextual completions. Comprehensive evaluation on simulated data across five ragas demonstrates that ShrutiSense (FST model) achieves 91.3% shruti classification accuracy for correction tasks, with example sequences showing 86.7-90.0% accuracy at corruption levels of 0.2 to 0.4. The system exhibits robust performance under pitch noise up to +/-50 cents, maintaining consistent accuracy across ragas (90.7-91.8%), thus preserving the cultural authenticity of Indian classical music expression. 

---
# Translation-Equivariant Self-Supervised Learning for Pitch Estimation with Optimal Transport 

**Authors**: Bernardo Torres, Alain Riou, Gaël Richard, Geoffroy Peeters  

**Link**: [PDF](https://arxiv.org/pdf/2508.01493)  

**Abstract**: In this paper, we propose an Optimal Transport objective for learning one-dimensional translation-equivariant systems and demonstrate its applicability to single pitch estimation. Our method provides a theoretically grounded, more numerically stable, and simpler alternative for training state-of-the-art self-supervised pitch estimators. 

---
# A Large-Scale Benchmark of Cross-Modal Learning for Histology and Gene Expression in Spatial Transcriptomics 

**Authors**: Rushin H. Gindra, Giovanni Palla, Mathias Nguyen, Sophia J. Wagner, Manuel Tran, Fabian J Theis, Dieter Saur, Lorin Crawford, Tingying Peng  

**Link**: [PDF](https://arxiv.org/pdf/2508.01490)  

**Abstract**: Spatial transcriptomics enables simultaneous measurement of gene expression and tissue morphology, offering unprecedented insights into cellular organization and disease mechanisms. However, the field lacks comprehensive benchmarks for evaluating multimodal learning methods that leverage both histology images and gene expression data. Here, we present HESCAPE, a large-scale benchmark for cross-modal contrastive pretraining in spatial transcriptomics, built on a curated pan-organ dataset spanning 6 different gene panels and 54 donors. We systematically evaluated state-of-the-art image and gene expression encoders across multiple pretraining strategies and assessed their effectiveness on two downstream tasks: gene mutation classification and gene expression prediction. Our benchmark demonstrates that gene expression encoders are the primary determinant of strong representational alignment, and that gene models pretrained on spatial transcriptomics data outperform both those trained without spatial data and simple baseline approaches. However, downstream task evaluation reveals a striking contradiction: while contrastive pretraining consistently improves gene mutation classification performance, it degrades direct gene expression prediction compared to baseline encoders trained without cross-modal objectives. We identify batch effects as a key factor that interferes with effective cross-modal alignment. Our findings highlight the critical need for batch-robust multimodal learning approaches in spatial transcriptomics. To accelerate progress in this direction, we release HESCAPE, providing standardized datasets, evaluation protocols, and benchmarking tools for the community 

---
# PESTO: Real-Time Pitch Estimation with Self-supervised Transposition-equivariant Objective 

**Authors**: Alain Riou, Bernardo Torres, Ben Hayes, Stefan Lattner, Gaëtan Hadjeres, Gaël Richard, Geoffroy Peeters  

**Link**: [PDF](https://arxiv.org/pdf/2508.01488)  

**Abstract**: In this paper, we introduce PESTO, a self-supervised learning approach for single-pitch estimation using a Siamese architecture. Our model processes individual frames of a Variable-$Q$ Transform (VQT) and predicts pitch distributions. The neural network is designed to be equivariant to translations, notably thanks to a Toeplitz fully-connected layer. In addition, we construct pitch-shifted pairs by translating and cropping the VQT frames and train our model with a novel class-based transposition-equivariant objective, eliminating the need for annotated data. Thanks to this architecture and training objective, our model achieves remarkable performances while being very lightweight ($130$k parameters). Evaluations on music and speech datasets (MIR-1K, MDB-stem-synth, and PTDB) demonstrate that PESTO not only outperforms self-supervised baselines but also competes with supervised methods, exhibiting superior cross-dataset generalization. Finally, we enhance PESTO's practical utility by developing a streamable VQT implementation using cached convolutions. Combined with our model's low latency (less than 10 ms) and minimal parameter count, this makes PESTO particularly suitable for real-time applications. 

---
# Training Dynamics of the Cooldown Stage in Warmup-Stable-Decay Learning Rate Scheduler 

**Authors**: Aleksandr Dremov, Alexander Hägele, Atli Kosson, Martin Jaggi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01483)  

**Abstract**: Learning rate scheduling is essential in transformer training, where the final annealing plays a crucial role in getting the best performance. However, the mechanisms behind this cooldown phase, with its characteristic drop in loss, remain poorly understood. To address this, we provide a comprehensive analysis focusing solely on the cooldown phase in the Warmup-Stable-Decay (WSD) learning rate scheduler. Our analysis reveals that different cooldown shapes reveal a fundamental bias-variance trade-off in the resulting models, with shapes that balance exploration and exploitation consistently outperforming alternatives. Similarly, we find substantial performance variations $\unicode{x2013}$ comparable to those from cooldown shape selection $\unicode{x2013}$ when tuning AdamW hyperparameters. Notably, we observe consistent improvements with higher values of $\beta_2$ during cooldown. From a loss landscape perspective, we provide visualizations of the landscape during cooldown, supporting the river valley loss perspective empirically. These findings offer practical recommendations for configuring the WSD scheduler in transformer training, emphasizing the importance of optimizing the cooldown phase alongside traditional hyperparameter tuning. 

---
# Reconstructing Trust Embeddings from Siamese Trust Scores: A Direct-Sum Approach with Fixed-Point Semantics 

**Authors**: Faruk Alpay, Taylan Alpay, Bugra Kilictas  

**Link**: [PDF](https://arxiv.org/pdf/2508.01479)  

**Abstract**: We study the inverse problem of reconstructing high-dimensional trust embeddings from the one-dimensional Siamese trust scores that many distributed-security frameworks expose. Starting from two independent agents that publish time-stamped similarity scores for the same set of devices, we formalise the estimation task, derive an explicit direct-sum estimator that concatenates paired score series with four moment features, and prove that the resulting reconstruction map admits a unique fixed point under a contraction argument rooted in Banach theory. A suite of synthetic benchmarks (20 devices x 10 time steps) confirms that, even in the presence of Gaussian noise, the recovered embeddings preserve inter-device geometry as measured by Euclidean and cosine metrics; we complement these experiments with non-asymptotic error bounds that link reconstruction accuracy to score-sequence length. Beyond methodology, the paper demonstrates a practical privacy risk: publishing granular trust scores can leak latent behavioural information about both devices and evaluation models. We therefore discuss counter-measures -- score quantisation, calibrated noise, obfuscated embedding spaces -- and situate them within wider debates on transparency versus confidentiality in networked AI systems. All datasets, reproduction scripts and extended proofs accompany the submission so that results can be verified without proprietary code. 

---
# Fast and scalable retrosynthetic planning with a transformer neural network and speculative beam search 

**Authors**: Mikhail Andronov, Natalia Andronova, Michael Wand, Jürgen Schmidhuber, Djork-Arné Clevert  

**Link**: [PDF](https://arxiv.org/pdf/2508.01459)  

**Abstract**: AI-based computer-aided synthesis planning (CASP) systems are in demand as components of AI-driven drug discovery workflows. However, the high latency of such CASP systems limits their utility for high-throughput synthesizability screening in de novo drug design. We propose a method for accelerating multi-step synthesis planning systems that rely on SMILES-to-SMILES transformers as single-step retrosynthesis models. Our approach reduces the latency of SMILES-to-SMILES transformers powering multi-step synthesis planning in AiZynthFinder through speculative beam search combined with a scalable drafting strategy called Medusa. Replacing standard beam search with our approach allows the CASP system to solve 26\% to 86\% more molecules under the same time constraints of several seconds. Our method brings AI-based CASP systems closer to meeting the strict latency requirements of high-throughput synthesizability screening and improving general user experience. 

---
# Tuning LLM-based Code Optimization via Meta-Prompting: An Industrial Perspective 

**Authors**: Jingzhi Gong, Rafail Giavrimis, Paul Brookes, Vardan Voskanyan, Fan Wu, Mari Ashiga, Matthew Truscott, Mike Basios, Leslie Kanthan, Jie Xu, Zheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01443)  

**Abstract**: There is a growing interest in leveraging large language models (LLMs) for automated code optimization. However, industrial platforms deploying multiple LLMs face a critical challenge: prompts optimized for one LLM often fail with others, requiring expensive model-specific prompt engineering. This cross-model prompt engineering bottleneck severely limits the practical deployment of multi-LLM optimization systems in production environments. To address this, we introduce Meta-Prompted Code Optimization (MPCO), a framework that automatically generates high-quality, task-specific prompts across diverse LLMs while maintaining industrial efficiency requirements. MPCO leverages meta-prompting to dynamically synthesize context-aware optimization prompts by integrating project metadata, task requirements, and LLM-specific contexts, and it seamlessly deploys on the ARTEMIS industrial platform for automated validation and scaling.
Our comprehensive evaluation on five real-world codebases with 366 hours of runtime benchmarking demonstrates MPCO's effectiveness: it achieves overall performance improvements up to 19.06% with the best statistical rank across all systems compared to baseline methods. Analysis shows that 96% of the top-performing optimizations stem from meaningful edits. Through systematic ablation studies and meta-prompter sensitivity analysis, we identify that comprehensive context integration is essential for effective meta-prompting, and that all three major LLMs can serve effectively as meta-prompters, providing actionable insights for industrial practitioners. 

---
# Capturing More: Learning Multi-Domain Representations for Robust Online Handwriting Verification 

**Authors**: Peirong Zhang, Kai Ding, Lianwen Jin  

**Link**: [PDF](https://arxiv.org/pdf/2508.01427)  

**Abstract**: In this paper, we propose SPECTRUM, a temporal-frequency synergistic model that unlocks the untapped potential of multi-domain representation learning for online handwriting verification (OHV). SPECTRUM comprises three core components: (1) a multi-scale interactor that finely combines temporal and frequency features through dual-modal sequence interaction and multi-scale aggregation, (2) a self-gated fusion module that dynamically integrates global temporal and frequency features via self-driven balancing. These two components work synergistically to achieve micro-to-macro spectral-temporal integration. (3) A multi-domain distance-based verifier then utilizes both temporal and frequency representations to improve discrimination between genuine and forged handwriting, surpassing conventional temporal-only approaches. Extensive experiments demonstrate SPECTRUM's superior performance over existing OHV methods, underscoring the effectiveness of temporal-frequency multi-domain learning. Furthermore, we reveal that incorporating multiple handwritten biometrics fundamentally enhances the discriminative power of handwriting representations and facilitates verification. These findings not only validate the efficacy of multi-domain learning in OHV but also pave the way for future research in multi-domain approaches across both feature and biometric domains. Code is publicly available at this https URL. 

---
# From Query to Logic: Ontology-Driven Multi-Hop Reasoning in LLMs 

**Authors**: Haonan Bian, Yutao Qi, Rui Yang, Yuanxi Che, Jiaqian Wang, Heming Xia, Ranran Zhen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01424)  

**Abstract**: Large Language Models (LLMs), despite their success in question answering, exhibit limitations in complex multi-hop question answering (MQA) tasks that necessitate non-linear, structured reasoning. This limitation stems from their inability to adequately capture deep conceptual relationships between entities. To overcome this challenge, we present **ORACLE** (**O**ntology-driven **R**easoning **A**nd **C**hain for **L**ogical **E**ucidation), a training-free framework that combines LLMs' generative capabilities with the structural benefits of knowledge graphs. Our approach operates through three stages: (1) dynamic construction of question-specific knowledge ontologies using LLMs, (2) transformation of these ontologies into First-Order Logic reasoning chains, and (3) systematic decomposition of the original query into logically coherent sub-questions. Experimental results on several standard MQA benchmarks show that our framework achieves highly competitive performance, rivaling current state-of-the-art models like DeepSeek-R1. Detailed analyses further confirm the effectiveness of each component, while demonstrating that our method generates more logical and interpretable reasoning chains than existing approaches. 

---
# RoboMemory: A Brain-inspired Multi-memory Agentic Framework for Lifelong Learning in Physical Embodied Systems 

**Authors**: Mingcong Lei, Honghao Cai, Zezhou Cui, Liangchen Tan, Junkun Hong, Gehan Hu, Shuangyu Zhu, Yimou Wu, Shaohan Jiang, Ge Wang, Zhen Li, Shuguang Cui, Yiming Zhao, Yatong Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.01415)  

**Abstract**: We present RoboMemory, a brain-inspired multi-memory framework for lifelong learning in physical embodied systems, addressing critical challenges in real-world environments: continuous learning, multi-module memory latency, task correlation capture, and infinite-loop mitigation in closed-loop planning. Grounded in cognitive neuroscience, it integrates four core modules: the Information Preprocessor (thalamus-like), the Lifelong Embodied Memory System (hippocampus-like), the Closed-Loop Planning Module (prefrontal lobe-like), and the Low-Level Executer (cerebellum-like) to enable long-term planning and cumulative learning. The Lifelong Embodied Memory System, central to the framework, alleviates inference speed issues in complex memory frameworks via parallelized updates/retrieval across Spatial, Temporal, Episodic, and Semantic submodules. It incorporates a dynamic Knowledge Graph (KG) and consistent architectural design to enhance memory consistency and scalability. Evaluations on EmbodiedBench show RoboMemory outperforms the open-source baseline (Qwen2.5-VL-72B-Ins) by 25% in average success rate and surpasses the closed-source State-of-the-Art (SOTA) (Claude3.5-Sonnet) by 5%, establishing new SOTA. Ablation studies validate key components (critic, spatial memory, long-term memory), while real-world deployment confirms its lifelong learning capability with significantly improved success rates across repeated tasks. RoboMemory alleviates high latency challenges with scalability, serving as a foundational reference for integrating multi-modal memory systems in physical robots. 

---
# MedSynth: Realistic, Synthetic Medical Dialogue-Note Pairs 

**Authors**: Ahmad Rezaie Mianroodi, Amirali Rezaie, Niko Grisel Todorov, Cyril Rakovski, Frank Rudzicz  

**Link**: [PDF](https://arxiv.org/pdf/2508.01401)  

**Abstract**: Physicians spend significant time documenting clinical encounters, a burden that contributes to professional burnout. To address this, robust automation tools for medical documentation are crucial. We introduce MedSynth -- a novel dataset of synthetic medical dialogues and notes designed to advance the Dialogue-to-Note (Dial-2-Note) and Note-to-Dialogue (Note-2-Dial) tasks. Informed by an extensive analysis of disease distributions, this dataset includes over 10,000 dialogue-note pairs covering over 2000 ICD-10 codes. We demonstrate that our dataset markedly enhances the performance of models in generating medical notes from dialogues, and dialogues from medical notes. The dataset provides a valuable resource in a field where open-access, privacy-compliant, and diverse training data are scarce. Code is available at this https URL and the dataset is available at this https URL. 

---
# Spatial-Frequency Aware for Object Detection in RAW Image 

**Authors**: Zhuohua Ye, Liming Zhang, Hongru Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.01396)  

**Abstract**: Direct RAW-based object detection offers great promise by utilizing RAW data (unprocessed sensor data), but faces inherent challenges due to its wide dynamic range and linear response, which tends to suppress crucial object details. In particular, existing enhancement methods are almost all performed in the spatial domain, making it difficult to effectively recover these suppressed details from the skewed pixel distribution of RAW images. To address this limitation, we turn to the frequency domain, where features, such as object contours and textures, can be naturally separated based on frequency. In this paper, we propose Space-Frequency Aware RAW Image Object Detection Enhancer (SFAE), a novel framework that synergizes spatial and frequency representations. Our contribution is threefold. The first lies in the ``spatialization" of frequency bands. Different from the traditional paradigm of directly manipulating abstract spectra in deep networks, our method inversely transforms individual frequency bands back into tangible spatial maps, thus preserving direct physical intuition. Then the cross-domain fusion attention module is developed to enable deep multimodal interactions between these maps and the original spatial features. Finally, the framework performs adaptive nonlinear adjustments by predicting and applying different gamma parameters for the two domains. 

---
# Via Score to Performance: Efficient Human-Controllable Long Song Generation with Bar-Level Symbolic Notation 

**Authors**: Tongxi Wang, Yang Yu, Qing Wang, Junlang Qian  

**Link**: [PDF](https://arxiv.org/pdf/2508.01394)  

**Abstract**: Song generation is regarded as the most challenging problem in music AIGC; nonetheless, existing approaches have yet to fully overcome four persistent limitations: controllability, generalizability, perceptual quality, and duration. We argue that these shortcomings stem primarily from the prevailing paradigm of attempting to learn music theory directly from raw audio, a task that remains prohibitively difficult for current models. To address this, we present Bar-level AI Composing Helper (BACH), the first model explicitly designed for song generation through human-editable symbolic scores. BACH introduces a tokenization strategy and a symbolic generative procedure tailored to hierarchical song structure. Consequently, it achieves substantial gains in the efficiency, duration, and perceptual quality of song generation. Experiments demonstrate that BACH, with a small model size, establishes a new SOTA among all publicly reported song generation systems, even surpassing commercial solutions such as Suno. Human evaluations further confirm its superiority across multiple subjective metrics. 

---
# Recognising, Anticipating, and Mitigating LLM Pollution of Online Behavioural Research 

**Authors**: Raluca Rilla, Tobias Werner, Hiromu Yakura, Iyad Rahwan, Anne-Marie Nussberger  

**Link**: [PDF](https://arxiv.org/pdf/2508.01390)  

**Abstract**: Online behavioural research faces an emerging threat as participants increasingly turn to large language models (LLMs) for advice, translation, or task delegation: LLM Pollution. We identify three interacting variants through which LLM Pollution threatens the validity and integrity of online behavioural research. First, Partial LLM Mediation occurs when participants make selective use of LLMs for specific aspects of a task, such as translation or wording support, leading researchers to (mis)interpret LLM-shaped outputs as human ones. Second, Full LLM Delegation arises when agentic LLMs complete studies with little to no human oversight, undermining the central premise of human-subject research at a more foundational level. Third, LLM Spillover signifies human participants altering their behaviour as they begin to anticipate LLM presence in online studies, even when none are involved. While Partial Mediation and Full Delegation form a continuum of increasing automation, LLM Spillover reflects second-order reactivity effects. Together, these variants interact and generate cascading distortions that compromise sample authenticity, introduce biases that are difficult to detect post hoc, and ultimately undermine the epistemic grounding of online research on human cognition and behaviour. Crucially, the threat of LLM Pollution is already co-evolving with advances in generative AI, creating an escalating methodological arms race. To address this, we propose a multi-layered response spanning researcher practices, platform accountability, and community efforts. As the challenge evolves, coordinated adaptation will be essential to safeguard methodological integrity and preserve the validity of online behavioural research. 

---
# Video-based Vehicle Surveillance in the Wild: License Plate, Make, and Model Recognition with Self Reflective Vision-Language Models 

**Authors**: Pouya Parsa, Keya Li, Kara M. Kockelman, Seongjin Choi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01387)  

**Abstract**: Automatic license plate recognition (ALPR) and vehicle make and model recognition underpin intelligent transportation systems, supporting law enforcement, toll collection, and post-incident investigation. Applying these methods to videos captured by handheld smartphones or non-static vehicle-mounted cameras presents unique challenges compared to fixed installations, including frequent camera motion, varying viewpoints, occlusions, and unknown road geometry. Traditional ALPR solutions, dependent on specialized hardware and handcrafted OCR pipelines, often degrade under these conditions. Recent advances in large vision-language models (VLMs) enable direct recognition of textual and semantic attributes from arbitrary imagery. This study evaluates the potential of VLMs for ALPR and makes and models recognition using monocular videos captured with handheld smartphones and non-static mounted cameras. The proposed license plate recognition pipeline filters to sharp frames, then sends a multimodal prompt to a VLM using several prompt strategies. Make and model recognition pipeline runs the same VLM with a revised prompt and an optional self-reflection module. In the self-reflection module, the model contrasts the query image with a reference from a 134-class dataset, correcting mismatches. Experiments on a smartphone dataset collected on the campus of the University of Texas at Austin, achieve top-1 accuracies of 91.67% for ALPR and 66.67% for make and model recognition. On the public UFPR-ALPR dataset, the approach attains 83.05% and 61.07%, respectively. The self-reflection module further improves results by 5.72% on average for make and model recognition. These findings demonstrate that VLMs provide a cost-effective solution for scalable, in-motion traffic video analysis. 

---
# A Full-Stage Refined Proposal Algorithm for Suppressing False Positives in Two-Stage CNN-Based Detection Methods 

**Authors**: Qiang Guo, Rubo Zhang, Bingbing Zhang, Junjie Liu, Jianqing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01382)  

**Abstract**: False positives in pedestrian detection remain a challenge that has yet to be effectively resolved. To address this issue, this paper proposes a Full-stage Refined Proposal (FRP) algorithm aimed at eliminating these false positives within a two-stage CNN-based pedestrian detection framework. The main innovation of this work lies in employing various pedestrian feature re-evaluation strategies to filter out low-quality pedestrian proposals during both the training and testing stages. Specifically, in the training phase, the Training mode FRP algorithm (TFRP) introduces a novel approach for validating pedestrian proposals to effectively guide the model training process, thereby constructing a model with strong capabilities for false positive suppression. During the inference phase, two innovative strategies are implemented: the Classifier-guided FRP (CFRP) algorithm integrates a pedestrian classifier into the proposal generation pipeline to yield high-quality proposals through pedestrian feature evaluation, and the Split-proposal FRP (SFRP) algorithm vertically divides all proposals, sending both the original and the sub-region proposals to the subsequent subnetwork to evaluate their confidence scores, filtering out those with lower sub-region pedestrian confidence scores. As a result, the proposed algorithm enhances the model's ability to suppress pedestrian false positives across all stages. Various experiments conducted on multiple benchmarks and the SY-Metro datasets demonstrate that the model, supported by different combinations of the FRP algorithm, can effectively eliminate false positives to varying extents. Furthermore, experiments conducted on embedded platforms underscore the algorithm's effectiveness in enhancing the comprehensive pedestrian detection capabilities of the small pedestrian detector in resource-constrained edge devices. 

---
# Effective Damage Data Generation by Fusing Imagery with Human Knowledge Using Vision-Language Models 

**Authors**: Jie Wei, Erika Ardiles-Cruz, Aleksey Panasyuk, Erik Blasch  

**Link**: [PDF](https://arxiv.org/pdf/2508.01380)  

**Abstract**: It is of crucial importance to assess damages promptly and accurately in humanitarian assistance and disaster response (HADR). Current deep learning approaches struggle to generalize effectively due to the imbalance of data classes, scarcity of moderate damage examples, and human inaccuracy in pixel labeling during HADR situations. To accommodate for these limitations and exploit state-of-the-art techniques in vision-language models (VLMs) to fuse imagery with human knowledge understanding, there is an opportunity to generate a diversified set of image-based damage data effectively. Our initial experimental results suggest encouraging data generation quality, which demonstrates an improvement in classifying scenes with different levels of structural damage to buildings, roads, and infrastructures. 

---
# Prompt to Pwn: Automated Exploit Generation for Smart Contracts 

**Authors**: Zeke Xiao, Yuekang Li, Qin Wang, Shiping Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01371)  

**Abstract**: We explore the feasibility of using LLMs for Automated Exploit Generation (AEG) against vulnerable smart contracts. We present \textsc{ReX}, a framework integrating LLM-based exploit synthesis with the Foundry testing suite, enabling the automated generation and validation of proof-of-concept (PoC) exploits. We evaluate five state-of-the-art LLMs (GPT-4.1, Gemini 2.5 Pro, Claude Opus 4, DeepSeek, and Qwen3 Plus) on both synthetic benchmarks and real-world smart contracts affected by known high-impact exploits. Our results show that modern LLMs can reliably generate functional PoC exploits for diverse vulnerability types, with success rates reaching up to 92\%. Notably, Gemini 2.5 Pro and GPT-4.1 consistently outperform others in both synthetic and real-world scenarios. We further analyze factors influencing AEG effectiveness, including model capabilities, contract structure, and vulnerability types. We also collect the first curated dataset of real-world PoC exploits to support future research. 

---
# Classification of Brain Tumors using Hybrid Deep Learning Models 

**Authors**: Neerav Nemchand Gala  

**Link**: [PDF](https://arxiv.org/pdf/2508.01350)  

**Abstract**: The use of Convolutional Neural Networks (CNNs) has greatly improved the interpretation of medical images. However, conventional CNNs typically demand extensive computational resources and large training datasets. To address these limitations, this study applied transfer learning to achieve strong classification performance using fewer training samples. Specifically, the study compared EfficientNetV2 with its predecessor, EfficientNet, and with ResNet50 in classifying brain tumors into three types: glioma, meningioma, and pituitary tumors. Results showed that EfficientNetV2 delivered superior performance compared to the other models. However, this improvement came at the cost of increased training time, likely due to the model's greater complexity. 

---
# Convergence Analysis of Aggregation-Broadcast in LoRA-enabled Federated Learning 

**Authors**: Xin Chen, Shuaijun Chen, Omid Tavallaie, Nguyen Tran, Shuhuang Xiang, Albert Zomaya  

**Link**: [PDF](https://arxiv.org/pdf/2508.01348)  

**Abstract**: Federated Learning (FL) enables collaborative model training across decentralized data sources while preserving data privacy. However, the growing size of Machine Learning (ML) models poses communication and computation challenges in FL. Low-Rank Adaptation (LoRA) has recently been introduced into FL as an efficient fine-tuning method, reducing communication overhead by updating only a small number of trainable parameters. Despite its effectiveness, how to aggregate LoRA-updated local models on the server remains a critical and understudied problem. In this paper, we provide a unified convergence analysis for LoRA-based FL. We first categories the current aggregation method into two major type: Sum-Product (SP) and Product-Sum (PS). Then we formally define the Aggregation-Broadcast Operator (ABO) and derive a general convergence condition under mild assumptions. Furthermore, we present several sufficient conditions that guarantee convergence of the global model. These theoretical analyze offer a principled understanding of various aggregation strategies. Notably, we prove that the SP and PS aggregation methods both satisfy our convergence condition, but differ in their ability to achieve the optimal convergence rate. Extensive experiments on standard benchmarks validate our theoretical findings. 

---
# UEChecker: Detecting Unchecked External Call Vulnerabilities in DApps via Graph Analysis 

**Authors**: Dechao Kong, Xiaoqi Li, Wenkai Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01343)  

**Abstract**: The increasing number of attacks on the contract layer of DApps has resulted in economic losses amounting to $66 billion. Vulnerabilities arise when contracts interact with external protocols without verifying the results of the calls, leading to exploit entry points such as flash loan attacks and reentrancy attacks. In this paper, we propose UEChecker, a deep learning-based tool that utilizes a call graph and a Graph Convolutional Network to detect unchecked external call vulnerabilities. We design the following components: An edge prediction module that reconstructs the feature representation of nodes and edges in the call graph; A node aggregation module that captures structural information from both the node itself and its neighbors, thereby enhancing feature representation between nodes and improving the model's understanding of the global graph structure; A Conformer Block module that integrates multi-head attention, convolutional modules, and feedforward neural networks to more effectively capture dependencies of different scales within the call graph, extending beyond immediate neighbors and enhancing the performance of vulnerability detection. Finally, we combine these modules with Graph Convolutional Network to detect unchecked external call vulnerabilities. By auditing the smart contracts of 608 DApps, our results show that our tool achieves an accuracy of 87.59% in detecting unchecked external call vulnerabilities. Furthermore, we compare our tool with GAT, LSTM, and GCN baselines, and in the comparison experiments, UEChecker consistently outperforms these models in terms of accuracy. 

---
# SBP-YOLO:A Lightweight Real-Time Model for Detecting Speed Bumps and Potholes 

**Authors**: Chuanqi Liang, Jie Fu, Lei Luo, Miao Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01339)  

**Abstract**: With increasing demand for ride comfort in new energy vehicles, accurate real-time detection of speed bumps and potholes is critical for predictive suspension control. This paper proposes SBP-YOLO, a lightweight detection framework based on YOLOv11, optimized for embedded deployment. The model integrates GhostConv for efficient computation, VoVGSCSPC for multi-scale feature enhancement, and a Lightweight Efficiency Detection Head (LEDH) to reduce early-stage feature processing costs. A hybrid training strategy combining NWD loss, knowledge distillation, and Albumentations-based weather augmentation improves detection robustness, especially for small and distant targets. Experiments show SBP-YOLO achieves 87.0% mAP (outperforming YOLOv11n by 5.8%) and runs at 139.5 FPS on a Jetson AGX Xavier with TensorRT FP16 quantization. The results validate its effectiveness for real-time road condition perception in intelligent suspension systems. 

---
# Weakly-Supervised Image Forgery Localization via Vision-Language Collaborative Reasoning Framework 

**Authors**: Ziqi Sheng, Junyan Wu, Wei Lu, Jiantao Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2508.01338)  

**Abstract**: Image forgery localization aims to precisely identify tampered regions within images, but it commonly depends on costly pixel-level annotations. To alleviate this annotation burden, weakly supervised image forgery localization (WSIFL) has emerged, yet existing methods still achieve limited localization performance as they mainly exploit intra-image consistency clues and lack external semantic guidance to compensate for weak supervision. In this paper, we propose ViLaCo, a vision-language collaborative reasoning framework that introduces auxiliary semantic supervision distilled from pre-trained vision-language models (VLMs), enabling accurate pixel-level localization using only image-level labels. Specifically, ViLaCo first incorporates semantic knowledge through a vision-language feature modeling network, which jointly extracts textual and visual priors using pre-trained VLMs. Next, an adaptive vision-language reasoning network aligns textual semantics and visual features through mutual interactions, producing semantically aligned representations. Subsequently, these representations are passed into dual prediction heads, where the coarse head performs image-level classification and the fine head generates pixel-level localization masks, thereby bridging the gap between weak supervision and fine-grained localization. Moreover, a contrastive patch consistency module is introduced to cluster tampered features while separating authentic ones, facilitating more reliable forgery discrimination. Extensive experiments on multiple public datasets demonstrate that ViLaCo substantially outperforms existing WSIFL methods, achieving state-of-the-art performance in both detection and localization accuracy. 

---
# BlockA2A: Towards Secure and Verifiable Agent-to-Agent Interoperability 

**Authors**: Zhenhua Zou, Zhuotao Liu, Lepeng Zhao, Qiuyang Zhan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01332)  

**Abstract**: The rapid adoption of agentic AI, powered by large language models (LLMs), is transforming enterprise ecosystems with autonomous agents that execute complex workflows. Yet we observe several key security vulnerabilities in LLM-driven multi-agent systems (MASes): fragmented identity frameworks, insecure communication channels, and inadequate defenses against Byzantine agents or adversarial prompts. In this paper, we present the first systematic analysis of these emerging multi-agent risks and explain why the legacy security strategies cannot effectively address these risks. Afterwards, we propose BlockA2A, the first unified multi-agent trust framework that enables secure and verifiable and agent-to-agent interoperability. At a high level, BlockA2A adopts decentralized identifiers (DIDs) to enable fine-grained cross-domain agent authentication, blockchain-anchored ledgers to enable immutable auditability, and smart contracts to dynamically enforce context-aware access control policies. BlockA2A eliminates centralized trust bottlenecks, ensures message authenticity and execution integrity, and guarantees accountability across agent interactions. Furthermore, we propose a Defense Orchestration Engine (DOE) that actively neutralizes attacks through real-time mechanisms, including Byzantine agent flagging, reactive execution halting, and instant permission revocation. Empirical evaluations demonstrate BlockA2A's effectiveness in neutralizing prompt-based, communication-based, behavioral and systemic MAS attacks. We formalize its integration into existing MAS and showcase a practical implementation for Google's A2A protocol. Experiments confirm that BlockA2A and DOE operate with sub-second overhead, enabling scalable deployment in production LLM-based MAS environments. 

---
# Referring Remote Sensing Image Segmentation with Cross-view Semantics Interaction Network 

**Authors**: Jiaxing Yang, Lihe Zhang, Huchuan Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01331)  

**Abstract**: Recently, Referring Remote Sensing Image Segmentation (RRSIS) has aroused wide attention. To handle drastic scale variation of remote targets, existing methods only use the full image as input and nest the saliency-preferring techniques of cross-scale information interaction into traditional single-view structure. Although effective for visually salient targets, they still struggle in handling tiny, ambiguous ones in lots of real scenarios. In this work, we instead propose a paralleled yet unified segmentation framework Cross-view Semantics Interaction Network (CSINet) to solve the limitations. Motivated by human behavior in observing targets of interest, the network orchestrates visual cues from remote and close distances to conduct synergistic prediction. In its every encoding stage, a Cross-View Window-attention module (CVWin) is utilized to supplement global and local semantics into close-view and remote-view branch features, finally promoting the unified representation of feature in every encoding stage. In addition, we develop a Collaboratively Dilated Attention enhanced Decoder (CDAD) to mine the orientation property of target and meanwhile integrate cross-view multiscale features. The proposed network seamlessly enhances the exploitation of global and local semantics, achieving significant improvements over others while maintaining satisfactory speed. 

---
# Is Exploration or Optimization the Problem for Deep Reinforcement Learning? 

**Authors**: Glen Berseth  

**Link**: [PDF](https://arxiv.org/pdf/2508.01329)  

**Abstract**: In the era of deep reinforcement learning, making progress is more complex, as the collected experience must be compressed into a deep model for future exploitation and sampling. Many papers have shown that training a deep learning policy under the changing state and action distribution leads to sub-optimal performance, or even collapse. This naturally leads to the concern that even if the community creates improved exploration algorithms or reward objectives, will those improvements fall on the \textit{deaf ears} of optimization difficulties. This work proposes a new \textit{practical} sub-optimality estimator to determine optimization limitations of deep reinforcement learning algorithms. Through experiments across environments and RL algorithms, it is shown that the difference between the best experience generated is 2-3$\times$ better than the policies' learned performance. This large difference indicates that deep RL methods only exploit half of the good experience they generate. 

---
# D-SCoRE: Document-Centric Segmentation and CoT Reasoning with Structured Export for QA-CoT Data Generation 

**Authors**: Weibo Zhou, Lingbo Li, Shangsong Liang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01309)  

**Abstract**: The scarcity and high cost of high-quality question-answering (QA) datasets hinder supervised fine-tuning (SFT) for domain-specific large language models (LLMs). To address this, we introduce D-SCoRE, a training-free pipeline that utilizes LLMs and prompt engineering to produce diverse, high-quality QA datasets from arbitrary textual sources. D-SCoRE integrates $\textbf{D}$ocument-centric processing, $\textbf{S}$egmentation, $\textbf{Co}$T $\textbf{R}$easoning, and structured $\textbf{E}$xport to generate QA-COT datasets tailored for domain-aware SFT. Multi-dimensional control mechanisms, such as semantic role transformation, question type balancing, and counterfactual materials, enhance diversity and relevance, overcoming limitations of existing QA generation. LLMs fine-tuned on D-SCoRE-generated QA datasets, and human-annotated QA datasets (SQuAD, Covid-QA) are evaluated on SQuADShifts and Covid-QA test sets, with D-SCoRE outperforming across most domains. D-SCoRE generates six QA-CoT pairs with four-option counterfactual materials per 100-200-word text in 90 seconds using an 8B LLM on consumer-grade hardware. Its simplicity and scalability enable efficient QA generation and high-performance fine-tuning across domains. 

---
# GMAT: Grounded Multi-Agent Clinical Description Generation for Text Encoder in Vision-Language MIL for Whole Slide Image Classification 

**Authors**: Ngoc Bui Lam Quang, Nam Le Nguyen Binh, Thanh-Huy Nguyen, Le Thien Phuc Nguyen, Quan Nguyen, Ulas Bagci  

**Link**: [PDF](https://arxiv.org/pdf/2508.01293)  

**Abstract**: Multiple Instance Learning (MIL) is the leading approach for whole slide image (WSI) classification, enabling efficient analysis of gigapixel pathology slides. Recent work has introduced vision-language models (VLMs) into MIL pipelines to incorporate medical knowledge through text-based class descriptions rather than simple class names. However, when these methods rely on large language models (LLMs) to generate clinical descriptions or use fixed-length prompts to represent complex pathology concepts, the limited token capacity of VLMs often constrains the expressiveness and richness of the encoded class information. Additionally, descriptions generated solely by LLMs may lack domain grounding and fine-grained medical specificity, leading to suboptimal alignment with visual features. To address these challenges, we propose a vision-language MIL framework with two key contributions: (1) A grounded multi-agent description generation system that leverages curated pathology textbooks and agent specialization (e.g., morphology, spatial context) to produce accurate and diverse clinical descriptions; (2) A text encoding strategy using a list of descriptions rather than a single prompt, capturing fine-grained and complementary clinical signals for better alignment with visual features. Integrated into a VLM-MIL pipeline, our approach shows improved performance over single-prompt class baselines and achieves results comparable to state-of-the-art models, as demonstrated on renal and lung cancer datasets. 

---
# CoCoLIT: ControlNet-Conditioned Latent Image Translation for MRI to Amyloid PET Synthesis 

**Authors**: Alec Sargood, Lemuel Puglisi, James H. Cole, Neil P. Oxtoby, Daniele Ravì, Daniel C. Alexander  

**Link**: [PDF](https://arxiv.org/pdf/2508.01292)  

**Abstract**: Synthesizing amyloid PET scans from the more widely available and accessible structural MRI modality offers a promising, cost-effective approach for large-scale Alzheimer's Disease (AD) screening. This is motivated by evidence that, while MRI does not directly detect amyloid pathology, it may nonetheless encode information correlated with amyloid deposition that can be uncovered through advanced modeling. However, the high dimensionality and structural complexity of 3D neuroimaging data pose significant challenges for existing MRI-to-PET translation methods. Modeling the cross-modality relationship in a lower-dimensional latent space can simplify the learning task and enable more effective translation. As such, we present CoCoLIT (ControlNet-Conditioned Latent Image Translation), a diffusion-based latent generative framework that incorporates three main innovations: (1) a novel Weighted Image Space Loss (WISL) that improves latent representation learning and synthesis quality; (2) a theoretical and empirical analysis of Latent Average Stabilization (LAS), an existing technique used in similar generative models to enhance inference consistency; and (3) the introduction of ControlNet-based conditioning for MRI-to-PET translation. We evaluate CoCoLIT's performance on publicly available datasets and find that our model significantly outperforms state-of-the-art methods on both image-based and amyloid-related metrics. Notably, in amyloid-positivity classification, CoCoLIT outperforms the second-best method with improvements of +10.5% on the internal dataset and +23.7% on the external dataset. The code and models of our approach are available at this https URL. 

---
# Exploitation Is All You Need... for Exploration 

**Authors**: Micah Rentschler, Jesse Roberts  

**Link**: [PDF](https://arxiv.org/pdf/2508.01287)  

**Abstract**: Ensuring sufficient exploration is a central challenge when training meta-reinforcement learning (meta-RL) agents to solve novel environments. Conventional solutions to the exploration-exploitation dilemma inject explicit incentives such as randomization, uncertainty bonuses, or intrinsic rewards to encourage exploration. In this work, we hypothesize that an agent trained solely to maximize a greedy (exploitation-only) objective can nonetheless exhibit emergent exploratory behavior, provided three conditions are met: (1) Recurring Environmental Structure, where the environment features repeatable regularities that allow past experience to inform future choices; (2) Agent Memory, enabling the agent to retain and utilize historical interaction data; and (3) Long-Horizon Credit Assignment, where learning propagates returns over a time frame sufficient for the delayed benefits of exploration to inform current decisions. Through experiments in stochastic multi-armed bandits and temporally extended gridworlds, we observe that, when both structure and memory are present, a policy trained on a strictly greedy objective exhibits information-seeking exploratory behavior. We further demonstrate, through controlled ablations, that emergent exploration vanishes if either environmental structure or agent memory is absent (Conditions 1 & 2). Surprisingly, removing long-horizon credit assignment (Condition 3) does not always prevent emergent exploration-a result we attribute to the pseudo-Thompson Sampling effect. These findings suggest that, under the right prerequisites, exploration and exploitation need not be treated as orthogonal objectives but can emerge from a unified reward-maximization process. 

---
# Defending Against Beta Poisoning Attacks in Machine Learning Models 

**Authors**: Nilufer Gulciftci, M. Emre Gursoy  

**Link**: [PDF](https://arxiv.org/pdf/2508.01276)  

**Abstract**: Poisoning attacks, in which an attacker adversarially manipulates the training dataset of a machine learning (ML) model, pose a significant threat to ML security. Beta Poisoning is a recently proposed poisoning attack that disrupts model accuracy by making the training dataset linearly nonseparable. In this paper, we propose four defense strategies against Beta Poisoning attacks: kNN Proximity-Based Defense (KPB), Neighborhood Class Comparison (NCC), Clustering-Based Defense (CBD), and Mean Distance Threshold (MDT). The defenses are based on our observations regarding the characteristics of poisoning samples generated by Beta Poisoning, e.g., poisoning samples have close proximity to one another, and they are centered near the mean of the target class. Experimental evaluations using MNIST and CIFAR-10 datasets demonstrate that KPB and MDT can achieve perfect accuracy and F1 scores, while CBD and NCC also provide strong defensive capabilities. Furthermore, by analyzing performance across varying parameters, we offer practical insights regarding defenses' behaviors under varying conditions. 

---
# AgentArmor: Enforcing Program Analysis on Agent Runtime Trace to Defend Against Prompt Injection 

**Authors**: Peiran Wang, Yang Liu, Yunfei Lu, Yifeng Cai, Hongbo Chen, Qingyou Yang, Jie Zhang, Jue Hong, Ye Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01249)  

**Abstract**: Large Language Model (LLM) agents offer a powerful new paradigm for solving various problems by combining natural language reasoning with the execution of external tools. However, their dynamic and non-transparent behavior introduces critical security risks, particularly in the presence of prompt injection attacks. In this work, we propose a novel insight that treats the agent runtime traces as structured programs with analyzable semantics. Thus, we present AgentArmor, a program analysis framework that converts agent traces into graph intermediate representation-based structured program dependency representations (e.g., CFG, DFG, and PDG) and enforces security policies via a type system. AgentArmor consists of three key components: (1) a graph constructor that reconstructs the agent's working traces as graph-based intermediate representations with control flow and data flow described within; (2) a property registry that attaches security-relevant metadata of interacted tools & data, and (3) a type system that performs static inference and checking over the intermediate representation. By representing agent behavior as structured programs, AgentArmor enables program analysis over sensitive data flow, trust boundaries, and policy violations. We evaluate AgentArmor on the AgentDojo benchmark, the results show that AgentArmor can achieve 95.75% of TPR, with only 3.66% of FPR. Our results demonstrate AgentArmor's ability to detect prompt injection vulnerabilities and enforce fine-grained security constraints. 

---
# Multi-Cache Enhanced Prototype Learning for Test-Time Generalization of Vision-Language Models 

**Authors**: Xinyu Chen, Haotian Zhai, Can Zhang, Xiupeng Shi, Ruirui Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01225)  

**Abstract**: In zero-shot setting, test-time adaptation adjusts pre-trained models using unlabeled data from the test phase to enhance performance on unknown test distributions. Existing cache-enhanced TTA methods rely on a low-entropy criterion to select samples for prototype construction, assuming intra-class compactness. However, low-entropy samples may be unreliable under distribution shifts, and the resulting prototypes may not ensure compact intra-class distributions. This study identifies a positive correlation between cache-enhanced performance and intra-class compactness. Based on this observation, we propose a Multi-Cache enhanced Prototype-based Test-Time Adaptation (MCP) featuring three caches: an entropy cache for initializing prototype representations with low-entropy samples, an align cache for integrating visual and textual information to achieve compact intra-class distributions, and a negative cache for prediction calibration using high-entropy samples. We further developed MCP++, a framework incorporating cross-modal prototype alignment and residual learning, introducing prototype residual fine-tuning. Comparative and ablation experiments across 15 downstream tasks demonstrate that the proposed method and framework achieve state-of-the-art generalization performance. 

---
# WebDS: An End-to-End Benchmark for Web-based Data Science 

**Authors**: Ethan Hsu, Hong Meng Yam, Ines Bouissou, Aaron Murali John, Raj Thota, Josh Koe, Vivek Sarath Putta, G K Dharesan, Alexander Spangher, Shikhar Murty, Tenghao Huang, Christopher D. Manning  

**Link**: [PDF](https://arxiv.org/pdf/2508.01222)  

**Abstract**: A large portion of real-world data science tasks are complex and require multi-hop web-based interactions: finding appropriate data available on the internet, synthesizing real-time data of various modalities from different locations, and producing summarized analyses. Existing web benchmarks often focus on simplistic interactions, such as form submissions or e-commerce transactions, and often do not require diverse tool-using capabilities required for web based data science. Conversely, traditional data science benchmarks typically concentrate on static, often textually bound datasets and do not assess end-to-end workflows that encompass data acquisition, cleaning, analysis, and insight generation. In response, we introduce WebDS, the first end-to-end web-based data science benchmark. It comprises 870 web-based data science tasks across 29 diverse websites from structured government data portals to unstructured news media, challenging agents to perform complex, multi-step operations requiring the use of tools and heterogeneous data formats that better reflect the realities of modern data analytics. Evaluations of current SOTA LLM agents indicate significant performance gaps in accomplishing these tasks. For instance, Browser Use, which accomplishes 80% of tasks on Web Voyager, successfully completes only 15% of tasks in WebDS, which our analysis suggests is due to new failure modes like poor information grounding, repetitive behavior and shortcut-taking that agents performing WebDS' tasks display. By providing a more robust and realistic testing ground, WebDS sets the stage for significant advances in the development of practically useful LLM-based data science. 

---
# Oldie but Goodie: Re-illuminating Label Propagation on Graphs with Partially Observed Features 

**Authors**: Sukwon Yun, Xin Liu, Yunhak Oh, Junseok Lee, Tianlong Chen, Tsuyoshi Murata, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2508.01209)  

**Abstract**: In real-world graphs, we often encounter missing feature situations where a few or the majority of node features, e.g., sensitive information, are missed. In such scenarios, directly utilizing Graph Neural Networks (GNNs) would yield sub-optimal results in downstream tasks such as node classification. Despite the emergence of a few GNN-based methods attempting to mitigate its missing situation, when only a few features are available, they rather perform worse than traditional structure-based models. To this end, we propose a novel framework that further illuminates the potential of classical Label Propagation (Oldie), taking advantage of Feature Propagation, especially when only a partial feature is available. Now called by GOODIE, it takes a hybrid approach to obtain embeddings from the Label Propagation branch and Feature Propagation branch. To do so, we first design a GNN-based decoder that enables the Label Propagation branch to output hidden embeddings that align with those of the FP branch. Then, GOODIE automatically captures the significance of structure and feature information thanks to the newly designed Structure-Feature Attention. Followed by a novel Pseudo-Label contrastive learning that differentiates the contribution of each positive pair within pseudo-labels originating from the LP branch, GOODIE outputs the final prediction for the unlabeled nodes. Through extensive experiments, we demonstrate that our proposed model, GOODIE, outperforms the existing state-of-the-art methods not only when only a few features are available but also in abundantly available situations. Source code of GOODIE is available at: this https URL. 

---
# Deep Learning for Pavement Condition Evaluation Using Satellite Imagery 

**Authors**: Prathyush Kumar Reddy Lebaku, Lu Gao, Pan Lu, Jingran Sun  

**Link**: [PDF](https://arxiv.org/pdf/2508.01206)  

**Abstract**: Civil infrastructure systems covers large land areas and needs frequent inspections to maintain their public service capabilities. The conventional approaches of manual surveys or vehicle-based automated surveys to assess infrastructure conditions are often labor-intensive and time-consuming. For this reason, it is worthwhile to explore more cost-effective methods for monitoring and maintaining these infrastructures. Fortunately, recent advancements in satellite systems and image processing algorithms have opened up new possibilities. Numerous satellite systems have been employed to monitor infrastructure conditions and identify damages. Due to the improvement in ground sample distance (GSD), the level of detail that can be captured has significantly increased. Taking advantage of these technology advancement, this research investigated to evaluate pavement conditions using deep learning models for analyzing satellite images. We gathered over 3,000 satellite images of pavement sections, together with pavement evaluation ratings from TxDOT's PMIS database. The results of our study show an accuracy rate is exceeding 90%. This research paves the way for a rapid and cost-effective approach to evaluating the pavement network in the future. 

---
# Conquering High Packet-Loss Erasure: MoE Swin Transformer-Based Video Semantic Communication 

**Authors**: Lei Teng, Senran Fan, Chen Dong, Haotai Liang, Zhicheng Bao, Xiaodong Xu, Rui Meng, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01205)  

**Abstract**: Semantic communication with joint semantic-channel coding robustly transmits diverse data modalities but faces challenges in mitigating semantic information loss due to packet drops in packet-based systems. Under current protocols, packets with errors are discarded, preventing the receiver from utilizing erroneous semantic data for robust decoding. To address this issue, a packet-loss-resistant MoE Swin Transformer-based Video Semantic Communication (MSTVSC) system is proposed in this paper. Semantic vectors are encoded by MSTVSC and transmitted through upper-layer protocol packetization. To investigate the impact of the packetization, a theoretical analysis of the packetization strategy is provided. To mitigate the semantic loss caused by packet loss, a 3D CNN at the receiver recovers missing information using un-lost semantic data and an packet-loss mask matrix. Semantic-level interleaving is employed to reduce concentrated semantic loss from packet drops. To improve compression, a common-individual decomposition approach is adopted, with downsampling applied to individual information to minimize redundancy. The model is lightweighted for practical deployment. Extensive simulations and comparisons demonstrate strong performance, achieving an MS-SSIM greater than 0.6 and a PSNR exceeding 20 dB at a 90% packet loss rate. 

---
# Adaptive Content Restriction for Large Language Models via Suffix Optimization 

**Authors**: Yige Li, Peihai Jiang, Jun Sun, Peng Shu, Tianming Liu, Zhen Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01198)  

**Abstract**: Large Language Models (LLMs) have demonstrated significant success across diverse applications. However, enforcing content restrictions remains a significant challenge due to their expansive output space. One aspect of content restriction is preventing LLMs from generating harmful content via model alignment approaches such as supervised fine-tuning (SFT). Yet, the need for content restriction may vary significantly across user groups, change rapidly over time, and not always align with general definitions of harmfulness. Applying SFT to each of these specific use cases is impractical due to the high computational, data, and storage demands. Motivated by this need, we propose a new task called \textit{Adaptive Content Restriction} (AdaCoRe), which focuses on lightweight strategies -- methods without model fine-tuning -- to prevent deployed LLMs from generating restricted terms for specific use cases. We propose the first method for AdaCoRe, named \textit{Suffix Optimization (SOP)}, which appends a short, optimized suffix to any prompt to a) prevent a target LLM from generating a set of restricted terms, while b) preserving the output quality. To evaluate AdaCoRe approaches, including our SOP, we create a new \textit{Content Restriction Benchmark} (CoReBench), which contains 400 prompts for 80 restricted terms across 8 carefully selected categories. We demonstrate the effectiveness of SOP on CoReBench, which outperforms the system-level baselines such as system suffix by 15\%, 17\%, 10\%, 9\%, and 6\% on average restriction rates for Gemma2-2B, Mistral-7B, Vicuna-7B, Llama3-8B, and Llama3.1-8B, respectively. We also demonstrate that SOP is effective on POE, an online platform hosting various commercial LLMs, highlighting its practicality in real-world scenarios. 

---
# BSL: A Unified and Generalizable Multitask Learning Platform for Virtual Drug Discovery from Design to Synthesis 

**Authors**: Kun Li, Zhennan Wu, Yida Xiong, Hongzhi Zhang, Longtao Hu, Zhonglie Liu, Junqi Zeng, Wenjie Wu, Mukun Chen, Jiameng Chen, Wenbin Hu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01195)  

**Abstract**: Drug discovery is of great social significance in safeguarding human health, prolonging life, and addressing the challenges of major diseases. In recent years, artificial intelligence has demonstrated remarkable advantages in key tasks across bioinformatics and pharmacology, owing to its efficient data processing and data representation capabilities. However, most existing computational platforms cover only a subset of core tasks, leading to fragmented workflows and low efficiency. In addition, they often lack algorithmic innovation and show poor generalization to out-of-distribution (OOD) data, which greatly hinders the progress of drug discovery. To address these limitations, we propose Baishenglai (BSL), a deep learning-enhanced, open-access platform designed for virtual drug discovery. BSL integrates seven core tasks within a unified and modular framework, incorporating advanced technologies such as generative models and graph neural networks. In addition to achieving state-of-the-art (SOTA) performance on multiple benchmark datasets, the platform emphasizes evaluation mechanisms that focus on generalization to OOD molecular structures. Comparative experiments with existing platforms and baseline methods demonstrate that BSL provides a comprehensive, scalable, and effective solution for virtual drug discovery, offering both algorithmic innovation and high-precision prediction for real-world pharmaceutical research. In addition, BSL demonstrated its practical utility by discovering novel modulators of the GluN1/GluN3A NMDA receptor, successfully identifying three compounds with clear bioactivity in in-vitro electrophysiological assays. These results highlight BSL as a promising and comprehensive platform for accelerating biomedical research and drug discovery. The platform is accessible at this https URL. 

---
# SpectrumWorld: Artificial Intelligence Foundation for Spectroscopy 

**Authors**: Zhuo Yang, Jiaqing Xie, Shuaike Shen, Daolang Wang, Yeyun Chen, Ben Gao, Shuzhou Sun, Biqing Qi, Dongzhan Zhou, Lei Bai, Linjiang Chen, Shufei Zhang, Jun Jiang, Tianfan Fu, Yuqiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01188)  

**Abstract**: Deep learning holds immense promise for spectroscopy, yet research and evaluation in this emerging field often lack standardized formulations. To address this issue, we introduce SpectrumLab, a pioneering unified platform designed to systematize and accelerate deep learning research in spectroscopy. SpectrumLab integrates three core components: a comprehensive Python library featuring essential data processing and evaluation tools, along with leaderboards; an innovative SpectrumAnnotator module that generates high-quality benchmarks from limited seed data; and SpectrumBench, a multi-layered benchmark suite covering 14 spectroscopic tasks and over 10 spectrum types, featuring spectra curated from over 1.2 million distinct chemical substances. Thorough empirical studies on SpectrumBench with 18 cutting-edge multimodal LLMs reveal critical limitations of current approaches. We hope SpectrumLab will serve as a crucial foundation for future advancements in deep learning-driven spectroscopy. 

---
# Advancing the Foundation Model for Music Understanding 

**Authors**: Yi Jiang, Wei Wang, Xianwen Guo, Huiyun Liu, Hanrui Wang, Youri Xu, Haoqi Gu, Zhongqian Xie, Chuanjiang Luo  

**Link**: [PDF](https://arxiv.org/pdf/2508.01178)  

**Abstract**: The field of Music Information Retrieval (MIR) is fragmented, with specialized models excelling at isolated tasks. In this work, we challenge this paradigm by introducing a unified foundation model named MuFun for holistic music understanding. Our model features a novel architecture that jointly processes instrumental and lyrical content, and is trained on a large-scale dataset covering diverse tasks such as genre classification, music tagging, and question answering. To facilitate robust evaluation, we also propose a new benchmark for multi-faceted music understanding called MuCUE (Music Comprehensive Understanding Evaluation). Experiments show our model significantly outperforms existing audio large language models across the MuCUE tasks, demonstrating its state-of-the-art effectiveness and generalization ability. 

---
# RSPO: Risk-Seeking Policy Optimization for Pass@k and Max@k Metrics in Large Language Models 

**Authors**: Kaichen Zhang, Shenghao Gao, Yuzhong Hong, Haipeng Sun, Junwei Bao, Hongfei Jiang, Yang Song, Hong Dingqian, Hui Xiong  

**Link**: [PDF](https://arxiv.org/pdf/2508.01174)  

**Abstract**: Current large language model post-training optimizes a risk-neutral objective that maximizes expected reward, yet evaluation relies heavily on risk-seeking metrics like Pass@k (at least one success in k trials) and Max@k (maximum reward across k responses). This mismatch in risk preferences can inevitably lead to suboptimal performance. To bridge this gap, we propose Risk-Seeking Policy Optimization (RSPO), a novel method that directly targets Pass@k and Max@k during training. A key challenge in optimizing these metrics is the "hitchhiking" problem: low-reward responses are inadvertently reinforced if they co-occur with a high-reward response within a sample of k generations, resulting in inefficient optimization. RSPO addresses this problem by leveraging the closed-form probability that a given response is the maximum among k samplings. Despite the complexity of nested gradients over multiple responses, RSPO produces efficient, unbiased gradient estimators for both metrics. We validate our approach with both rigorous theoretical analysis and comprehensive experimental results. 

---
# GeHirNet: A Gender-Aware Hierarchical Model for Voice Pathology Classification 

**Authors**: Fan Wu, Kaicheng Zhao, Elgar Fleisch, Filipe Barata  

**Link**: [PDF](https://arxiv.org/pdf/2508.01172)  

**Abstract**: AI-based voice analysis shows promise for disease diagnostics, but existing classifiers often fail to accurately identify specific pathologies because of gender-related acoustic variations and the scarcity of data for rare diseases. We propose a novel two-stage framework that first identifies gender-specific pathological patterns using ResNet-50 on Mel spectrograms, then performs gender-conditioned disease classification. We address class imbalance through multi-scale resampling and time warping augmentation. Evaluated on a merged dataset from four public repositories, our two-stage architecture with time warping achieves state-of-the-art performance (97.63\% accuracy, 95.25\% MCC), with a 5\% MCC improvement over single-stage baseline. This work advances voice pathology classification while reducing gender bias through hierarchical modeling of vocal characteristics. 

---
# Asking the Right Questions: Benchmarking Large Language Models in the Development of Clinical Consultation Templates 

**Authors**: Liam G. McCoy, Fateme Nateghi Haredasht, Kanav Chopra, David Wu, David JH Wu, Abass Conteh, Sarita Khemani, Saloni Kumar Maharaj, Vishnu Ravi, Arth Pahwa, Yingjie Weng, Leah Rosengaus, Lena Giang, Kelvin Zhenghao Li, Olivia Jee, Daniel Shirvani, Ethan Goh, Jonathan H. Chen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01159)  

**Abstract**: This study evaluates the capacity of large language models (LLMs) to generate structured clinical consultation templates for electronic consultation. Using 145 expert-crafted templates developed and routinely used by Stanford's eConsult team, we assess frontier models -- including o3, GPT-4o, Kimi K2, Claude 4 Sonnet, Llama 3 70B, and Gemini 2.5 Pro -- for their ability to produce clinically coherent, concise, and prioritized clinical question schemas. Through a multi-agent pipeline combining prompt optimization, semantic autograding, and prioritization analysis, we show that while models like o3 achieve high comprehensiveness (up to 92.2\%), they consistently generate excessively long templates and fail to correctly prioritize the most clinically important questions under length constraints. Performance varies across specialties, with significant degradation in narrative-driven fields such as psychiatry and pain medicine. Our findings demonstrate that LLMs can enhance structured clinical information exchange between physicians, while highlighting the need for more robust evaluation methods that capture a model's ability to prioritize clinically salient information within the time constraints of real-world physician communication. 

---
# Personalized Safety Alignment for Text-to-Image Diffusion Models 

**Authors**: Yu Lei, Jinbin Bai, Qingyu Shi, Aosong Feng, Kaidong Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01151)  

**Abstract**: Text-to-image diffusion models have revolutionized visual content generation, but current safety mechanisms apply uniform standards that often fail to account for individual user preferences. These models overlook the diverse safety boundaries shaped by factors like age, mental health, and personal beliefs. To address this, we propose Personalized Safety Alignment (PSA), a framework that allows user-specific control over safety behaviors in generative models. PSA integrates personalized user profiles into the diffusion process, adjusting the model's behavior to match individual safety preferences while preserving image quality. We introduce a new dataset, Sage, which captures user-specific safety preferences and incorporates these profiles through a cross-attention mechanism. Experiments show that PSA outperforms existing methods in harmful content suppression and aligns generated content better with user constraints, achieving higher Win Rate and Pass Rate scores. Our code, data, and models are publicly available at this https URL. 

---
# Dataset Condensation with Color Compensation 

**Authors**: Huyu Wu, Duo Su, Junjie Hou, Guang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.01139)  

**Abstract**: Dataset condensation always faces a constitutive trade-off: balancing performance and fidelity under extreme compression. Existing methods struggle with two bottlenecks: image-level selection methods (Coreset Selection, Dataset Quantization) suffer from inefficiency condensation, while pixel-level optimization (Dataset Distillation) introduces semantic distortion due to over-parameterization. With empirical observations, we find that a critical problem in dataset condensation is the oversight of color's dual role as an information carrier and a basic semantic representation unit. We argue that improving the colorfulness of condensed images is beneficial for representation learning. Motivated by this, we propose DC3: a Dataset Condensation framework with Color Compensation. After a calibrated selection strategy, DC3 utilizes the latent diffusion model to enhance the color diversity of an image rather than creating a brand-new one. Extensive experiments demonstrate the superior performance and generalization of DC3 that outperforms SOTA methods across multiple benchmarks. To the best of our knowledge, besides focusing on downstream tasks, DC3 is the first research to fine-tune pre-trained diffusion models with condensed datasets. The FID results prove that training networks with our high-quality datasets is feasible without model collapse or other degradation issues. Code and generated data will be released soon. 

---
# DBAIOps: A Reasoning LLM-Enhanced Database Operation and Maintenance System using Knowledge Graphs 

**Authors**: Wei Zhou, Peng Sun, Xuanhe Zhou, Qianglei Zang, Ji Xu, Tieying Zhang, Guoliang Li, Fan Wu  

**Link**: [PDF](https://arxiv.org/pdf/2508.01136)  

**Abstract**: The operation and maintenance (O&M) of database systems is critical to ensuring system availability and performance, typically requiring expert experience (e.g., identifying metric-to-anomaly relations) for effective diagnosis and recovery. However, existing automatic database O&M methods, including commercial products, cannot effectively utilize expert experience. On the one hand, rule-based methods only support basic O&M tasks (e.g., metric-based anomaly detection), which are mostly numerical equations and cannot effectively incorporate literal O&M experience (e.g., troubleshooting guidance in manuals). On the other hand, LLM-based methods, which retrieve fragmented information (e.g., standard documents + RAG), often generate inaccurate or generic results. To address these limitations, we present DBAIOps, a novel hybrid database O&M system that combines reasoning LLMs with knowledge graphs to achieve DBA-style diagnosis. First, DBAIOps introduces a heterogeneous graph model for representing the diagnosis experience, and proposes a semi-automatic graph construction algorithm to build that graph from thousands of documents. Second, DBAIOps develops a collection of (800+) reusable anomaly models that identify both directly alerted metrics and implicitly correlated experience and metrics. Third, for each anomaly, DBAIOps proposes a two-stage graph evolution mechanism to explore relevant diagnosis paths and identify missing relations automatically. It then leverages a reasoning LLM (e.g., DeepSeek-R1) to infer root causes and generate clear diagnosis reports for both DBAs and common users. Our evaluation over four mainstream database systems (Oracle, MySQL, PostgreSQL, and DM8) demonstrates that DBAIOps outperforms state-of-the-art baselines, 34.85% and 47.22% higher in root cause and human evaluation accuracy, respectively. 

---
# COLLAGE: Adaptive Fusion-based Retrieval for Augmented Policy Learning 

**Authors**: Sateesh Kumar, Shivin Dass, Georgios Pavlakos, Roberto Martín-Martín  

**Link**: [PDF](https://arxiv.org/pdf/2508.01131)  

**Abstract**: In this work, we study the problem of data retrieval for few-shot imitation learning: selecting data from a large dataset to train a performant policy for a specific task, given only a few target demonstrations. Prior methods retrieve data using a single-feature distance heuristic, assuming that the best demonstrations are those that most closely resemble the target examples in visual, semantic, or motion space. However, this approach captures only a subset of the relevant information and can introduce detrimental demonstrations, e.g., retrieving data from unrelated tasks due to similar scene layouts, or selecting similar motions from tasks with divergent goals. We present COLLAGE, a method for COLLective data AGgrEgation in few-shot imitation learning that uses an adaptive late fusion mechanism to guide the selection of relevant demonstrations based on a task-specific combination of multiple cues. COLLAGE follows a simple, flexible, and efficient recipe: it assigns weights to subsets of the dataset that are pre-selected using a single feature (e.g., appearance, shape, or language similarity), based on how well a policy trained on each subset predicts actions in the target demonstrations. These weights are then used to perform importance sampling during policy training, sampling data more densely or sparsely according to estimated relevance. COLLAGE is general and feature-agnostic, allowing it to combine any number of subsets selected by any retrieval heuristic, and to identify which subsets provide the greatest benefit for the target task. In extensive experiments, COLLAGE outperforms state-of-the-art retrieval and multi-task learning approaches by 5.1% in simulation across 10 tasks, and by 16.6% in the real world across 6 tasks, where we perform retrieval from the large-scale DROID dataset. More information at this https URL . 

---
# Human-Robot Red Teaming for Safety-Aware Reasoning 

**Authors**: Emily Sheetz, Emma Zemler, Misha Savchenko, Connor Rainen, Erik Holum, Jodi Graf, Andrew Albright, Shaun Azimi, Benjamin Kuipers  

**Link**: [PDF](https://arxiv.org/pdf/2508.01129)  

**Abstract**: While much research explores improving robot capabilities, there is a deficit in researching how robots are expected to perform tasks safely, especially in high-risk problem domains. Robots must earn the trust of human operators in order to be effective collaborators in safety-critical tasks, specifically those where robots operate in human environments. We propose the human-robot red teaming paradigm for safety-aware reasoning. We expect humans and robots to work together to challenge assumptions about an environment and explore the space of hazards that may arise. This exploration will enable robots to perform safety-aware reasoning, specifically hazard identification, risk assessment, risk mitigation, and safety reporting. We demonstrate that: (a) human-robot red teaming allows human-robot teams to plan to perform tasks safely in a variety of domains, and (b) robots with different embodiments can learn to operate safely in two different environments -- a lunar habitat and a household -- with varying definitions of safety. Taken together, our work on human-robot red teaming for safety-aware reasoning demonstrates the feasibility of this approach for safely operating and promoting trust on human-robot teams in safety-critical problem domains. 

---
# Towards Bridging Review Sparsity in Recommendation with Textual Edge Graph Representation 

**Authors**: Leyao Wang, Xutao Mao, Xuhui Zhan, Yuying Zhao, Bo Ni, Ryan A. Rossi, Nesreen K. Ahmed, Tyler Derr  

**Link**: [PDF](https://arxiv.org/pdf/2508.01128)  

**Abstract**: Textual reviews enrich recommender systems with fine-grained preference signals and enhanced explainability. However, in real-world scenarios, users rarely leave reviews, resulting in severe sparsity that undermines the effectiveness of existing models. A natural solution is to impute or generate missing reviews to enrich the data. However, conventional imputation techniques -- such as matrix completion and LLM-based augmentation -- either lose contextualized semantics by embedding texts into vectors, or overlook structural dependencies among user-item interactions. To address these shortcomings, we propose TWISTER (ToWards Imputation on Sparsity with Textual Edge Graph Representation), a unified framework that imputes missing reviews by jointly modeling semantic and structural signals. Specifically, we represent user-item interactions as a Textual-Edge Graph (TEG), treating reviews as edge attributes. To capture relational context, we construct line-graph views and employ a large language model as a graph-aware aggregator. For each interaction lacking a textual review, our model aggregates the neighborhood's natural-language representations to generate a coherent and personalized review. Experiments on the Amazon and Goodreads datasets show that TWISTER consistently outperforms traditional numeric, graph-based, and LLM baselines, delivering higher-quality imputed reviews and, more importantly, enhanced recommendation performance. In summary, TWISTER generates reviews that are more helpful, authentic, and specific, while smoothing structural signals for improved recommendations. 

---
# TensoMeta-VQC: A Tensor-Train-Guided Meta-Learning Framework for Robust and Scalable Variational Quantum Computing 

**Authors**: Jun Qi, Chao-Han Yang, Pin-Yu Chen, Min-Hsiu Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2508.01116)  

**Abstract**: Variational Quantum Computing (VQC) faces fundamental barriers in scalability, primarily due to barren plateaus and quantum noise sensitivity. To address these challenges, we introduce TensoMeta-VQC, a novel tensor-train (TT)-guided meta-learning framework designed to improve the robustness and scalability of VQC significantly. Our framework fully delegates the generation of quantum circuit parameters to a classical TT network, effectively decoupling optimization from quantum hardware. This innovative parameterization mitigates gradient vanishing, enhances noise resilience through structured low-rank representations, and facilitates efficient gradient propagation. Based on Neural Tangent Kernel and statistical learning theory, our rigorous theoretical analyses establish strong guarantees on approximation capability, optimization stability, and generalization performance. Extensive empirical results across quantum dot classification, Max-Cut optimization, and molecular quantum simulation tasks demonstrate that TensoMeta-VQC consistently achieves superior performance and robust noise tolerance, establishing it as a principled pathway toward practical and scalable VQC on near-term quantum devices. 

---
# Protecting Student Mental Health with a Context-Aware Machine Learning Framework for Stress Monitoring 

**Authors**: Md Sultanul Islam Ovi, Jamal Hossain, Md Raihan Alam Rahi, Fatema Akter  

**Link**: [PDF](https://arxiv.org/pdf/2508.01105)  

**Abstract**: Student mental health is an increasing concern in academic institutions, where stress can severely impact well-being and academic performance. Traditional assessment methods rely on subjective surveys and periodic evaluations, offering limited value for timely intervention. This paper introduces a context-aware machine learning framework for classifying student stress using two complementary survey-based datasets covering psychological, academic, environmental, and social factors. The framework follows a six-stage pipeline involving preprocessing, feature selection (SelectKBest, RFECV), dimensionality reduction (PCA), and training with six base classifiers: SVM, Random Forest, Gradient Boosting, XGBoost, AdaBoost, and Bagging. To enhance performance, we implement ensemble strategies, including hard voting, soft voting, weighted voting, and stacking. Our best models achieve 93.09% accuracy with weighted hard voting on the Student Stress Factors dataset and 99.53% with stacking on the Stress and Well-being dataset, surpassing previous benchmarks. These results highlight the potential of context-integrated, data-driven systems for early stress detection and underscore their applicability in real-world academic settings to support student well-being. 

---
# Cross-Domain Web Information Extraction at Pinterest 

**Authors**: Michael Farag, Patrick Halina, Andrey Zaytsev, Alekhya Munagala, Imtihan Ahmed, Junhao Wang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01096)  

**Abstract**: The internet offers a massive repository of unstructured information, but it's a significant challenge to convert this into a structured format. At Pinterest, the ability to accurately extract structured product data from e-commerce websites is essential to enhance user experiences and improve content distribution. In this paper, we present Pinterest's system for attribute extraction, which achieves remarkable accuracy and scalability at a manageable cost. Our approach leverages a novel webpage representation that combines structural, visual, and text modalities into a compact form, optimizing it for small model learning. This representation captures each visible HTML node with its text, style and layout information. We show how this allows simple models such as eXtreme Gradient Boosting (XGBoost) to extract attributes more accurately than much more complex Large Language Models (LLMs) such as Generative Pre-trained Transformer (GPT). Our results demonstrate a system that is highly scalable, processing over 1,000 URLs per second, while being 1000 times more cost-effective than the cheapest GPT alternatives. 

---
# Provably Secure Retrieval-Augmented Generation 

**Authors**: Pengcheng Zhou, Yinglun Feng, Zhongliang Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01084)  

**Abstract**: Although Retrieval-Augmented Generation (RAG) systems have been widely applied, the privacy and security risks they face, such as data leakage and data poisoning, have not been systematically addressed yet. Existing defense strategies primarily rely on heuristic filtering or enhancing retriever robustness, which suffer from limited interpretability, lack of formal security guarantees, and vulnerability to adaptive attacks. To address these challenges, this paper proposes the first provably secure framework for RAG systems(SAG). Our framework employs a pre-storage full-encryption scheme to ensure dual protection of both retrieved content and vector embeddings, guaranteeing that only authorized entities can access the data. Through formal security proofs, we rigorously verify the scheme's confidentiality and integrity under a computational security model. Extensive experiments across multiple benchmark datasets demonstrate that our framework effectively resists a range of state-of-the-art attacks. This work establishes a theoretical foundation and practical paradigm for verifiably secure RAG systems, advancing AI-powered services toward formally guaranteed security. 

---
# Learning Pivoting Manipulation with Force and Vision Feedback Using Optimization-based Demonstrations 

**Authors**: Yuki Shirai, Kei Ota, Devesh K. Jha, Diego Romeres  

**Link**: [PDF](https://arxiv.org/pdf/2508.01082)  

**Abstract**: Non-prehensile manipulation is challenging due to complex contact interactions between objects, the environment, and robots. Model-based approaches can efficiently generate complex trajectories of robots and objects under contact constraints. However, they tend to be sensitive to model inaccuracies and require access to privileged information (e.g., object mass, size, pose), making them less suitable for novel objects. In contrast, learning-based approaches are typically more robust to modeling errors but require large amounts of data. In this paper, we bridge these two approaches to propose a framework for learning closed-loop pivoting manipulation. By leveraging computationally efficient Contact-Implicit Trajectory Optimization (CITO), we design demonstration-guided deep Reinforcement Learning (RL), leading to sample-efficient learning. We also present a sim-to-real transfer approach using a privileged training strategy, enabling the robot to perform pivoting manipulation using only proprioception, vision, and force sensing without access to privileged information. Our method is evaluated on several pivoting tasks, demonstrating that it can successfully perform sim-to-real transfer. 

---
# The Lattice Geometry of Neural Network Quantization -- A Short Equivalence Proof of GPTQ and Babai's algorithm 

**Authors**: Johann Birnick  

**Link**: [PDF](https://arxiv.org/pdf/2508.01077)  

**Abstract**: We explain how data-driven quantization of a linear unit in a neural network corresponds to solving the closest vector problem for a certain lattice generated by input data. We prove that the GPTQ algorithm is equivalent to Babai's well-known nearest-plane algorithm. We furthermore provide geometric intuition for both algorithms. Lastly, we note the consequences of these results, in particular hinting at the possibility for using lattice basis reduction for better quantization. 

---
# Expressive Power of Graph Transformers via Logic 

**Authors**: Veeti Ahvonen, Maurice Funk, Damian Heiman, Antti Kuusisto, Carsten Lutz  

**Link**: [PDF](https://arxiv.org/pdf/2508.01067)  

**Abstract**: Transformers are the basis of modern large language models, but relatively little is known about their precise expressive power on graphs. We study the expressive power of graph transformers (GTs) by Dwivedi and Bresson (2020) and GPS-networks by Rampásek et al. (2022), both under soft-attention and average hard-attention. Our study covers two scenarios: the theoretical setting with real numbers and the more practical case with floats. With reals, we show that in restriction to vertex properties definable in first-order logic (FO), GPS-networks have the same expressive power as graded modal logic (GML) with the global modality. With floats, GPS-networks turn out to be equally expressive as GML with the counting global modality. The latter result is absolute, not restricting to properties definable in a background logic. We also obtain similar characterizations for GTs in terms of propositional logic with the global modality (for reals) and the counting global modality (for floats). 

---
# Connectivity Management in Satellite-Aided Vehicular Networks with Multi-Head Attention-Based State Estimation 

**Authors**: Ibrahim Althamary, Chen-Fu Chou, Chih-Wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2508.01060)  

**Abstract**: Managing connectivity in integrated satellite-terrestrial vehicular networks is critical for 6G, yet is challenged by dynamic conditions and partial observability. This letter introduces the Multi-Agent Actor-Critic with Satellite-Aided Multi-head self-attention (MAAC-SAM), a novel multi-agent reinforcement learning framework that enables vehicles to autonomously manage connectivity across Vehicle-to-Satellite (V2S), Vehicle-to-Infrastructure (V2I), and Vehicle-to-Vehicle (V2V) links. Our key innovation is the integration of a multi-head attention mechanism, which allows for robust state estimation even with fluctuating and limited information sharing among vehicles. The framework further leverages self-imitation learning (SIL) and fingerprinting to improve learning efficiency and real-time decisions. Simulation results, based on realistic SUMO traffic models and 3GPP-compliant configurations, demonstrate that MAAC-SAM outperforms state-of-the-art terrestrial and satellite-assisted baselines by up to 14% in transmission utility and maintains high estimation accuracy across varying vehicle densities and sharing levels. 

---
# Llama-3.1-FoundationAI-SecurityLLM-8B-Instruct Technical Report 

**Authors**: Sajana Weerawardhena, Paul Kassianik, Blaine Nelson, Baturay Saglam, Anu Vellore, Aman Priyanshu, Supriti Vijay, Massimo Aufiero, Arthur Goldblatt, Fraser Burch, Ed Li, Jianliang He, Dhruv Kedia, Kojin Oshiba, Zhouran Yang, Yaron Singer, Amin Karbasi  

**Link**: [PDF](https://arxiv.org/pdf/2508.01059)  

**Abstract**: Large language models (LLMs) have shown remarkable success across many domains, yet their integration into cybersecurity applications remains limited due to a lack of general-purpose cybersecurity data, representational complexity, and safety and regulatory concerns. To address this gap, we previously introduced Foundation-Sec-8B, a cybersecurity-focused LLM suitable for fine-tuning on downstream tasks. That model, however, was not designed for chat-style interactions or instruction-following. In this report, we release Foundation-Sec-8B-Instruct: a model specifically trained for general-purpose cybersecurity dialogue. Built on Foundation-Sec-8B, it combines domain-specific knowledge with instruction-following, conversational capabilities, and alignment with human preferences to produce high-quality, relevant responses. Comprehensive evaluations show that Foundation-Sec-8B-Instruct outperforms Llama 3.1-8B-Instruct on a range of cybersecurity tasks while matching its instruction-following performance. It is also competitive with GPT-4o-mini on cyber threat intelligence and instruction-following tasks. We envision Foundation-Sec-8B-Instruct becoming an indispensable assistant in the daily workflows of cybersecurity professionals. We release the model publicly at this https URL. 

---
# Managing Escalation in Off-the-Shelf Large Language Models 

**Authors**: Sebastian Elbaum, Jonathan Panther  

**Link**: [PDF](https://arxiv.org/pdf/2508.01056)  

**Abstract**: U.S. national security customers have begun to utilize large language models, including enterprise versions of ``off-the-shelf'' models (e.g., ChatGPT) familiar to the public. This uptake will likely accelerate. However, recent studies suggest that off-the-shelf large language models frequently suggest escalatory actions when prompted with geopolitical or strategic scenarios. We demonstrate two simple, non-technical interventions to control these tendencies. Introducing these interventions into the experimental wargame design of a recent study, we substantially reduce escalation throughout the game. Calls to restrict the use of large language models in national security applications are thus premature. The U.S. government is already, and will continue, employing large language models for scenario planning and suggesting courses of action. Rather than warning against such applications, this study acknowledges the imminent adoption of large language models, and provides actionable measures to align them with national security goals, including escalation management. 

---
# FGBench: A Dataset and Benchmark for Molecular Property Reasoning at Functional Group-Level in Large Language Models 

**Authors**: Xuan Liu, Siru Ouyang, Xianrui Zhong, Jiawei Han, Huimin Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.01055)  

**Abstract**: Large language models (LLMs) have gained significant attention in chemistry. However, most existing datasets center on molecular-level property prediction and overlook the role of fine-grained functional group (FG) information. Incorporating FG-level data can provide valuable prior knowledge that links molecular structures with textual descriptions, which can be used to build more interpretable, structure-aware LLMs for reasoning on molecule-related tasks. Moreover, LLMs can learn from such fine-grained information to uncover hidden relationships between specific functional groups and molecular properties, thereby advancing molecular design and drug discovery. Here, we introduce FGBench, a dataset comprising 625K molecular property reasoning problems with functional group information. Functional groups are precisely annotated and localized within the molecule, which ensures the dataset's interoperability thereby facilitating further multimodal applications. FGBench includes both regression and classification tasks on 245 different functional groups across three categories for molecular property reasoning: (1) single functional group impacts, (2) multiple functional group interactions, and (3) direct molecular comparisons. In the benchmark of state-of-the-art LLMs on 7K curated data, the results indicate that current LLMs struggle with FG-level property reasoning, highlighting the need to enhance reasoning capabilities in LLMs for chemistry tasks. We anticipate that the methodology employed in FGBench to construct datasets with functional group-level information will serve as a foundational framework for generating new question-answer pairs, enabling LLMs to better understand fine-grained molecular structure-property relationships. The dataset and evaluation code are available at \href{this https URL}{this https URL}. 

---
# Autonomous Penetration Testing: Solving Capture-the-Flag Challenges with LLMs 

**Authors**: Isabelle Bakker, John Hastings  

**Link**: [PDF](https://arxiv.org/pdf/2508.01054)  

**Abstract**: This study evaluates the ability of GPT-4o to autonomously solve beginner-level offensive security tasks by connecting the model to OverTheWire's Bandit capture-the-flag game. Of the 25 levels that were technically compatible with a single-command SSH framework, GPT-4o solved 18 unaided and another two after minimal prompt hints for an overall 80% success rate. The model excelled at single-step challenges that involved Linux filesystem navigation, data extraction or decoding, and straightforward networking. The approach often produced the correct command in one shot and at a human-surpassing speed. Failures involved multi-command scenarios that required persistent working directories, complex network reconnaissance, daemon creation, or interaction with non-standard shells. These limitations highlight current architectural deficiencies rather than a lack of general exploit knowledge. The results demonstrate that large language models (LLMs) can automate a substantial portion of novice penetration-testing workflow, potentially lowering the expertise barrier for attackers and offering productivity gains for defenders who use LLMs as rapid reconnaissance aides. Further, the unsolved tasks reveal specific areas where secure-by-design environments might frustrate simple LLM-driven attacks, informing future hardening strategies. Beyond offensive cybersecurity applications, results suggest the potential to integrate LLMs into cybersecurity education as practice aids. 

---
# A Deep Reinforcement Learning-Based TCP Congestion Control Algorithm: Design, Simulation, and Evaluation 

**Authors**: Efe Ağlamazlar, Emirhan Eken, Harun Batur Geçici  

**Link**: [PDF](https://arxiv.org/pdf/2508.01047)  

**Abstract**: This paper presents a novel TCP congestion control algorithm based on Deep Reinforcement Learning. The proposed approach utilizes Deep Q-Networks to optimize the congestion window (cWnd) by observing key network parameters and taking real-time actions. The algorithm is trained and evaluated within the NS-3 network simulator using the OpenGym interface. The results demonstrate significant improvements over traditional TCP New Reno in terms of latency and throughput, with better adaptability to changing network conditions. This study emphasizes the potential of reinforcement learning techniques for solving complex congestion control problems in modern networks. 

---
# AutoSIGHT: Automatic Eye Tracking-based System for Immediate Grading of Human experTise 

**Authors**: Byron Dowling, Jozef Probcin, Adam Czajka  

**Link**: [PDF](https://arxiv.org/pdf/2508.01015)  

**Abstract**: Can we teach machines to assess the expertise of humans solving visual tasks automatically based on eye tracking features? This paper proposes AutoSIGHT, Automatic System for Immediate Grading of Human experTise, that classifies expert and non-expert performers, and builds upon an ensemble of features extracted from eye tracking data while the performers were solving a visual task. Results on the task of iris Presentation Attack Detection (PAD) used for this study show that with a small evaluation window of just 5 seconds, AutoSIGHT achieves an average average Area Under the ROC curve performance of 0.751 in subject-disjoint train-test regime, indicating that such detection is viable. Furthermore, when a larger evaluation window of up to 30 seconds is available, the Area Under the ROC curve (AUROC) increases to 0.8306, indicating the model is effectively leveraging more information at a cost of slightly delayed decisions. This work opens new areas of research on how to incorporate the automatic weighing of human and machine expertise into human-AI pairing setups, which need to react dynamically to nonstationary expertise distribution between the human and AI players (e.g. when the experts need to be replaced, or the task at hand changes rapidly). Along with this paper, we offer the eye tracking data used in this study collected from 6 experts and 53 non-experts solving iris PAD visual task. 

---
# On Some Tunable Multi-fidelity Bayesian Optimization Frameworks 

**Authors**: Arjun Manoj, Anastasia S. Georgiou, Dimitris G. Giovanis, Themistoklis P. Sapsis, Ioannis G. Kevrekidis  

**Link**: [PDF](https://arxiv.org/pdf/2508.01013)  

**Abstract**: Multi-fidelity optimization employs surrogate models that integrate information from varying levels of fidelity to guide efficient exploration of complex design spaces while minimizing the reliance on (expensive) high-fidelity objective function evaluations. To advance Gaussian Process (GP)-based multi-fidelity optimization, we implement a proximity-based acquisition strategy that simplifies fidelity selection by eliminating the need for separate acquisition functions at each fidelity level. We also enable multi-fidelity Upper Confidence Bound (UCB) strategies by combining them with multi-fidelity GPs rather than the standard GPs typically used. We benchmark these approaches alongside other multi-fidelity acquisition strategies (including fidelity-weighted approaches) comparing their performance, reliance on high-fidelity evaluations, and hyperparameter tunability in representative optimization tasks. The results highlight the capability of the proximity-based multi-fidelity acquisition function to deliver consistent control over high-fidelity usage while maintaining convergence efficiency. Our illustrative examples include multi-fidelity chemical kinetic models, both homogeneous and heterogeneous (dynamic catalysis for ammonia production). 

---
# v-PuNNs: van der Put Neural Networks for Transparent Ultrametric Representation Learning 

**Authors**: Gnankan Landry Regis N'guessan  

**Link**: [PDF](https://arxiv.org/pdf/2508.01010)  

**Abstract**: Conventional deep learning models embed data in Euclidean space $\mathbb{R}^d$, a poor fit for strictly hierarchical objects such as taxa, word senses, or file systems. We introduce van der Put Neural Networks (v-PuNNs), the first architecture whose neurons are characteristic functions of p-adic balls in $\mathbb{Z}_p$. Under our Transparent Ultrametric Representation Learning (TURL) principle every weight is itself a p-adic number, giving exact subtree semantics. A new Finite Hierarchical Approximation Theorem shows that a depth-K v-PuNN with $\sum_{j=0}^{K-1}p^{\,j}$ neurons universally represents any K-level tree. Because gradients vanish in this discrete space, we propose Valuation-Adaptive Perturbation Optimization (VAPO), with a fast deterministic variant (HiPaN-DS) and a moment-based one (HiPaN / Adam-VAPO). On three canonical benchmarks our CPU-only implementation sets new state-of-the-art: WordNet nouns (52,427 leaves) 99.96% leaf accuracy in 16 min; GO molecular-function 96.9% leaf / 100% root in 50 s; NCBI Mammalia Spearman $\rho = -0.96$ with true taxonomic distance. The learned metric is perfectly ultrametric (zero triangle violations), and its fractal and information-theoretic properties are analyzed. Beyond classification we derive structural invariants for quantum systems (HiPaQ) and controllable generative codes for tabular data (Tab-HiPaN). v-PuNNs therefore bridge number theory and deep learning, offering exact, interpretable, and efficient models for hierarchical data. 

---
# Generative AI Adoption in Postsecondary Education, AI Hype, and ChatGPT's Launch 

**Authors**: Isabel Pedersen  

**Link**: [PDF](https://arxiv.org/pdf/2508.01003)  

**Abstract**: The rapid integration of generative artificial intelligence (AI) into postsecondary education and many other sectors resulted in a global reckoning with this new technology. This paper contributes to the study of the multifaceted influence of generative AI, with a particular focus on OpenAI's ChatGPT within academic settings during the first six months after the release in three specific ways. First, it scrutinizes the rise of ChatGPT as a transformative event construed through a study of mainstream discourses exhibiting AI hype. Second, it discusses the perceived implications of generative AI for writing, teaching, and learning through the lens of critical discourse analysis and critical AI studies. Third, it encourages the necessity for best practices in the adoption of generative AI technologies in education. 

---
# Are LLM-Powered Social Media Bots Realistic? 

**Authors**: Lynnette Hui Xian Ng, Kathleen M. Carley  

**Link**: [PDF](https://arxiv.org/pdf/2508.00998)  

**Abstract**: As Large Language Models (LLMs) become more sophisticated, there is a possibility to harness LLMs to power social media bots. This work investigates the realism of generating LLM-Powered social media bot networks. Through a combination of manual effort, network science and LLMs, we create synthetic bot agent personas, their tweets and their interactions, thereby simulating social media networks. We compare the generated networks against empirical bot/human data, observing that both network and linguistic properties of LLM-Powered Bots differ from Wild Bots/Humans. This has implications towards the detection and effectiveness of LLM-Powered Bots. 

---
# ThermoCycleNet: Stereo-based Thermogram Labeling for Model Transition to Cycling 

**Authors**: Daniel Andrés López, Vincent Weber, Severin Zentgraf, Barlo Hillen, Perikles Simon, Elmar Schömer  

**Link**: [PDF](https://arxiv.org/pdf/2508.00974)  

**Abstract**: Infrared thermography is emerging as a powerful tool in sports medicine, allowing assessment of thermal radiation during exercise and analysis of anatomical regions of interest, such as the well-exposed calves. Building on our previous advanced automatic annotation method, we aimed to transfer the stereo- and multimodal-based labeling approach from treadmill running to ergometer cycling. Therefore, the training of the semantic segmentation network with automatic labels and fine-tuning on high-quality manually annotated images has been examined and compared in different data set combinations. The results indicate that fine-tuning with a small fraction of manual data is sufficient to improve the overall performance of the deep neural network. Finally, combining automatically generated labels with small manually annotated data sets accelerates the adaptation of deep neural networks to new use cases, such as the transition from treadmill to bicycle. 

---
# Generative AI as a Geopolitical Factor in Industry 5.0: Sovereignty, Access, and Control 

**Authors**: Azmine Toushik Wasi, Enjamamul Haque Eram, Sabrina Afroz Mitu, Md Manjurul Ahsan  

**Link**: [PDF](https://arxiv.org/pdf/2508.00973)  

**Abstract**: Industry 5.0 marks a new phase in industrial evolution, emphasizing human-centricity, sustainability, and resilience through the integration of advanced technologies. Within this evolving landscape, Generative AI (GenAI) and autonomous systems are not only transforming industrial processes but also emerging as pivotal geopolitical instruments. We examine strategic implications of GenAI in Industry 5.0, arguing that these technologies have become national assets central to sovereignty, access, and global influence. As countries compete for AI supremacy, growing disparities in talent, computational infrastructure, and data access are reshaping global power hierarchies and accelerating the fragmentation of the digital economy. The human-centric ethos of Industry 5.0, anchored in collaboration between humans and intelligent systems, increasingly conflicts with the autonomy and opacity of GenAI, raising urgent governance challenges related to meaningful human control, dual-use risks, and accountability. We analyze how these dynamics influence defense strategies, industrial competitiveness, and supply chain resilience, including the geopolitical weaponization of export controls and the rise of data sovereignty. Our contribution synthesizes technological, economic, and ethical perspectives to propose a comprehensive framework for navigating the intersection of GenAI and geopolitics. We call for governance models that balance national autonomy with international coordination while safeguarding human-centric values in an increasingly AI-driven world. 

---
# AI-Educational Development Loop (AI-EDL): A Conceptual Framework to Bridge AI Capabilities with Classical Educational Theories 

**Authors**: Ning Yu, Jie Zhang, Sandeep Mitra, Rebecca Smith, Adam Rich  

**Link**: [PDF](https://arxiv.org/pdf/2508.00970)  

**Abstract**: This study introduces the AI-Educational Development Loop (AI-EDL), a theory-driven framework that integrates classical learning theories with human-in-the-loop artificial intelligence (AI) to support reflective, iterative learning. Implemented in EduAlly, an AI-assisted platform for writing-intensive and feedback-sensitive tasks, the framework emphasizes transparency, self-regulated learning, and pedagogical oversight. A mixed-methods study was piloted at a comprehensive public university to evaluate alignment between AI-generated feedback, instructor evaluations, and student self-assessments; the impact of iterative revision on performance; and student perceptions of AI feedback. Quantitative results demonstrated statistically significant improvement between first and second attempts, with agreement between student self-evaluations and final instructor grades. Qualitative findings indicated students valued immediacy, specificity, and opportunities for growth that AI feedback provided. These findings validate the potential to enhance student learning outcomes through developmentally grounded, ethically aligned, and scalable AI feedback systems. The study concludes with implications for future interdisciplinary applications and refinement of AI-supported educational technologies. 

---
# Masked Omics Modeling for Multimodal Representation Learning across Histopathology and Molecular Profiles 

**Authors**: Lucas Robinet, Ahmad Berjaoui, Elizabeth Cohen-Jonathan Moyal  

**Link**: [PDF](https://arxiv.org/pdf/2508.00969)  

**Abstract**: Self-supervised learning has driven major advances in computational pathology by enabling models to learn rich representations from hematoxylin and eosin (H&E)-stained cancer tissue. However, histopathology alone often falls short for molecular characterization and understanding clinical outcomes, as important information is contained in high-dimensional omics profiles like transcriptomics, methylomics, or genomics. In this work, we introduce MORPHEUS, a unified transformer-based pre-training framework that encodes both histopathology and multi-omics data into a shared latent space. At its core, MORPHEUS relies on a masked modeling objective applied to randomly selected omics portions, encouraging the model to learn biologically meaningful cross-modal relationships. The same pre-trained network can be applied to histopathology alone or in combination with any subset of omics modalities, seamlessly adapting to the available inputs. Additionally, MORPHEUS enables any-to-any omics generation, enabling one or more omics profiles to be inferred from any subset of modalities, including H&E alone. Pre-trained on a large pan-cancer cohort, MORPHEUS consistently outperforms state-of-the-art methods across diverse modality combinations and tasks, positioning itself as a promising framework for developing multimodal foundation models in oncology. The code is available at: this https URL 

---
# VAULT: Vigilant Adversarial Updates via LLM-Driven Retrieval-Augmented Generation for NLI 

**Authors**: Roie Kazoom, Ofir Cohen, Rami Puzis, Asaf Shabtai, Ofer Hadar  

**Link**: [PDF](https://arxiv.org/pdf/2508.00965)  

**Abstract**: We introduce VAULT, a fully automated adversarial RAG pipeline that systematically uncovers and remedies weaknesses in NLI models through three stages: retrieval, adversarial generation, and iterative retraining. First, we perform balanced few-shot retrieval by embedding premises with both semantic (BGE) and lexical (BM25) similarity. Next, we assemble these contexts into LLM prompts to generate adversarial hypotheses, which are then validated by an LLM ensemble for label fidelity. Finally, the validated adversarial examples are injected back into the training set at increasing mixing ratios, progressively fortifying a zero-shot RoBERTa-base this http URL standard benchmarks, VAULT elevates RoBERTa-base accuracy from 88.48% to 92.60% on SNLI +4.12%, from 75.04% to 80.95% on ANLI +5.91%, and from 54.67% to 71.99% on MultiNLI +17.32%. It also consistently outperforms prior in-context adversarial methods by up to 2.0% across datasets. By automating high-quality adversarial data curation at scale, VAULT enables rapid, human-independent robustness improvements in NLI inference tasks. 

---
# Rethinking Multimodality: Optimizing Multimodal Deep Learning for Biomedical Signal Classification 

**Authors**: Timothy Oladunni, Alex Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00963)  

**Abstract**: This study proposes a novel perspective on multimodal deep learning for biomedical signal classification, systematically analyzing how complementary feature domains impact model performance. While fusing multiple domains often presumes enhanced accuracy, this work demonstrates that adding modalities can yield diminishing returns, as not all fusions are inherently advantageous. To validate this, five deep learning models were designed, developed, and rigorously evaluated: three unimodal (1D-CNN for time, 2D-CNN for time-frequency, and 1D-CNN-Transformer for frequency) and two multimodal (Hybrid 1, which fuses 1D-CNN and 2D-CNN; Hybrid 2, which combines 1D-CNN, 2D-CNN, and a Transformer). For ECG classification, bootstrapping and Bayesian inference revealed that Hybrid 1 consistently outperformed the 2D-CNN baseline across all metrics (p-values < 0.05, Bayesian probabilities > 0.90), confirming the synergistic complementarity of the time and time-frequency domains. Conversely, Hybrid 2's inclusion of the frequency domain offered no further improvement and sometimes a marginal decline, indicating representational redundancy; a phenomenon further substantiated by a targeted ablation study. This research redefines a fundamental principle of multimodal design in biomedical signal analysis. We demonstrate that optimal domain fusion isn't about the number of modalities, but the quality of their inherent complementarity. This paradigm-shifting concept moves beyond purely heuristic feature selection. Our novel theoretical contribution, "Complementary Feature Domains in Multimodal ECG Deep Learning," presents a mathematically quantifiable framework for identifying ideal domain combinations, demonstrating that optimal multimodal performance arises from the intrinsic information-theoretic complementarity among fused domains. 

---
# FinKario: Event-Enhanced Automated Construction of Financial Knowledge Graph 

**Authors**: Xiang Li, Penglei Sun, Wanyun Zhou, Zikai Wei, Yongqi Zhang, Xiaowen Chu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00961)  

**Abstract**: Individual investors are significantly outnumbered and disadvantaged in financial markets, overwhelmed by abundant information and lacking professional analysis. Equity research reports stand out as crucial resources, offering valuable insights. By leveraging these reports, large language models (LLMs) can enhance investors' decision-making capabilities and strengthen financial analysis. However, two key challenges limit their effectiveness: (1) the rapid evolution of market events often outpaces the slow update cycles of existing knowledge bases, (2) the long-form and unstructured nature of financial reports further hinders timely and context-aware integration by LLMs. To address these challenges, we tackle both data and methodological aspects. First, we introduce the Event-Enhanced Automated Construction of Financial Knowledge Graph (FinKario), a dataset comprising over 305,360 entities, 9,625 relational triples, and 19 distinct relation types. FinKario automatically integrates real-time company fundamentals and market events through prompt-driven extraction guided by professional institutional templates, providing structured and accessible financial insights for LLMs. Additionally, we propose a Two-Stage, Graph-Based retrieval strategy (FinKario-RAG), optimizing the retrieval of evolving, large-scale financial knowledge to ensure efficient and precise data access. Extensive experiments show that FinKario with FinKario-RAG achieves superior stock trend prediction accuracy, outperforming financial LLMs by 18.81% and institutional strategies by 17.85% on average in backtesting. 

---
# Compression-Induced Communication-Efficient Large Model Training and Inferencing 

**Authors**: Sudip K. Seal, Maksudul Alam, Jorge Ramirez, Sajal Dash, Hao Lu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00960)  

**Abstract**: Energy efficiency of training and inferencing with large neural network models is a critical challenge facing the future of sustainable large-scale machine learning workloads. This paper introduces an alternative strategy, called phantom parallelism, to minimize the net energy consumption of traditional tensor (model) parallelism, the most energy-inefficient component of large neural network training. The approach is presented in the context of feed-forward network architectures as a preliminary, but comprehensive, proof-of-principle study of the proposed methodology. We derive new forward and backward propagation operators for phantom parallelism, implement them as custom autograd operations within an end-to-end phantom parallel training pipeline and compare its parallel performance and energy-efficiency against those of conventional tensor parallel training pipelines. Formal analyses that predict lower bandwidth and FLOP counts are presented with supporting empirical results on up to 256 GPUs that corroborate these gains. Experiments are shown to deliver ~50% reduction in the energy consumed to train FFNs using the proposed phantom parallel approach when compared with conventional tensor parallel methods. Additionally, the proposed approach is shown to train smaller phantom models to the same model loss on smaller GPU counts as larger tensor parallel models on larger GPU counts offering the possibility for even greater energy savings. 

---
# Enhancing material behavior discovery using embedding-oriented Physically-Guided Neural Networks with Internal Variables 

**Authors**: Rubén Muñoz-Sierra, Manuel Doblaré, Jacobo Ayensa-Jiménez  

**Link**: [PDF](https://arxiv.org/pdf/2508.00959)  

**Abstract**: Physically Guided Neural Networks with Internal Variables are SciML tools that use only observable data for training and and have the capacity to unravel internal state relations. They incorporate physical knowledge both by prescribing the model architecture and using loss regularization, thus endowing certain specific neurons with a physical meaning as internal state variables. Despite their potential, these models face challenges in scalability when applied to high-dimensional data such as fine-grid spatial fields or time-evolving systems. In this work, we propose some enhancements to the PGNNIV framework that address these scalability limitations through reduced-order modeling techniques. Specifically, we introduce alternatives to the original decoder structure using spectral decomposition, POD, and pretrained autoencoder-based mappings. These surrogate decoders offer varying trade-offs between computational efficiency, accuracy, noise tolerance, and generalization, while improving drastically the scalability. Additionally, we integrate model reuse via transfer learning and fine-tuning strategies to exploit previously acquired knowledge, supporting efficient adaptation to novel materials or configurations, and significantly reducing training time while maintaining or improving model performance. To illustrate these various techniques, we use a representative case governed by the nonlinear diffusion equation, using only observable data. Results demonstrate that the enhanced PGNNIV framework successfully identifies the underlying constitutive state equations while maintaining high predictive accuracy. It also improves robustness to noise, mitigates overfitting, and reduces computational demands. The proposed techniques can be tailored to various scenarios depending on data availability, resources, and specific modeling objectives, overcoming scalability challenges in all the scenarios. 

---
# Small sample-based adaptive text classification through iterative and contrastive description refinement 

**Authors**: Amrit Rajeev, Udayaadithya Avadhanam, Harshula Tulapurkar, SaiBarath Sundar  

**Link**: [PDF](https://arxiv.org/pdf/2508.00957)  

**Abstract**: Zero-shot text classification remains a difficult task in domains with evolving knowledge and ambiguous category boundaries, such as ticketing systems. Large language models (LLMs) often struggle to generalize in these scenarios due to limited topic separability, while few-shot methods are constrained by insufficient data diversity. We propose a classification framework that combines iterative topic refinement, contrastive prompting, and active learning. Starting with a small set of labeled samples, the model generates initial topic labels. Misclassified or ambiguous samples are then used in an iterative contrastive prompting process to refine category distinctions by explicitly teaching the model to differentiate between closely related classes. The framework features a human-in-the-loop component, allowing users to introduce or revise category definitions in natural language. This enables seamless integration of new, unseen categories without retraining, making the system well-suited for real-world, dynamic environments. The evaluations on AGNews and DBpedia demonstrate strong performance: 91% accuracy on AGNews (3 seen, 1 unseen class) and 84% on DBpedia (8 seen, 1 unseen), with minimal accuracy shift after introducing unseen classes (82% and 87%, respectively). The results highlight the effectiveness of prompt-based semantic reasoning for fine-grained classification with limited supervision. 

---
# Learning Unified User Quantized Tokenizers for User Representation 

**Authors**: Chuan He, Yang Chen, Wuliang Huang, Tianyi Zheng, Jianhu Chen, Bin Dou, Yice Luo, Yun Zhu, Baokun Wang, Yongchao Liu, Xing Fu, Yu Cheng, Chuntao Hong, Weiqiang Wang, Xin-Wei Yao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00956)  

**Abstract**: Multi-source user representation learning plays a critical role in enabling personalized services on web platforms (e.g., Alipay). While prior works have adopted late-fusion strategies to combine heterogeneous data sources, they suffer from three key limitations: lack of unified representation frameworks, scalability and storage issues in data compression, and inflexible cross-task generalization. To address these challenges, we propose U^2QT (Unified User Quantized Tokenizers), a novel framework that integrates cross-domain knowledge transfer with early fusion of heterogeneous domains. Our framework employs a two-stage architecture: first, a causal Q-Former projects domain-specific features into a shared causal representation space to preserve inter-modality dependencies; second, a multi-view RQ-VAE discretizes causal embeddings into compact tokens through shared and source-specific codebooks, enabling efficient storage while maintaining semantic coherence. Experimental results showcase U^2QT's advantages across diverse downstream tasks, outperforming task-specific baselines in future behavior prediction and recommendation tasks while achieving efficiency gains in storage and computation. The unified tokenization framework enables seamless integration with language models and supports industrial-scale applications. 

---
# From Generator to Embedder: Harnessing Innate Abilities of Multimodal LLMs via Building Zero-Shot Discriminative Embedding Model 

**Authors**: Yeong-Joon Ju, Seong-Whan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2508.00955)  

**Abstract**: Multimodal Large Language Models (MLLMs) have emerged as a promising solution for universal embedding tasks, yet adapting their generative nature for discriminative representation learning remains a significant challenge. The dominant paradigm of large-scale contrastive pre-training suffers from critical inefficiencies, including prohibitive computational costs and a failure to leverage the intrinsic, instruction-following capabilities of MLLMs. To overcome these limitations, we propose an efficient framework for universal multimodal embeddings, which bridges this gap by centering on two synergistic components. First, our hierarchical embedding prompt template employs a two-level instruction architecture that forces the model to produce discriminative representations. Building on this strong foundation, our second component, self-aware hard negative sampling, redefines the fine-tuning process by leveraging the model's own understanding to efficiently mine challenging negatives while actively filtering out potential false negatives. Our comprehensive experiments show that our hierarchical prompt achieves zero-shot performance competitive with contrastively trained baselines and enhances the fine-tuning process by lifting a simple in-batch negative baseline by 4.8 points on the MMEB benchmark. We further boost the performance via our self-aware hard negative sampling, achieving the state-of-the-art performance without the contrative pre-training. Our work presents an effective and efficient pathway to adapt MLLMs for universal embedding tasks, significantly reducing training time. 

---
# Academic Vibe Coding: Opportunities for Accelerating Research in an Era of Resource Constraint 

**Authors**: Matthew G Crowson, Leo Celi A. Celi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00952)  

**Abstract**: Academic laboratories face mounting resource constraints: budgets are tightening, grant overheads are potentially being capped, and the market rate for data-science talent significantly outstrips university compensation. Vibe coding, which is structured, prompt-driven code generation with large language models (LLMs) embedded in reproducible workflows, offers one pragmatic response. It aims to compress the idea-to-analysis timeline, reduce staffing pressure on specialized data roles, and maintain rigorous, version-controlled outputs. This article defines the vibe coding concept, situates it against the current academic resourcing crisis, details a beginner-friendly toolchain for its implementation, and analyzes inherent limitations that necessitate governance and mindful application. 

---
# LLMs Can Covertly Sandbag on Capability Evaluations Against Chain-of-Thought Monitoring 

**Authors**: Chloe Li, Mary Phuong, Noah Y. Siegel  

**Link**: [PDF](https://arxiv.org/pdf/2508.00943)  

**Abstract**: Trustworthy evaluations of dangerous capabilities are increasingly crucial for determining whether an AI system is safe to deploy. One empirically demonstrated threat to this is sandbagging - the strategic underperformance on evaluations by AI models or their developers. One promising defense is to monitor a model's chain-of-thought (CoT) reasoning, as this could reveal its intentions and plans. In this work, we measure the ability of models to sandbag on dangerous capability evaluations against a CoT monitor by prompting them to sandbag while being either monitor-oblivious or monitor-aware. We show that both frontier models and small open-sourced models can covertly sandbag against CoT monitoring 0-shot without hints. However, they cannot yet do so reliably: they bypass the monitor 16-36\% of the time when monitor-aware, conditioned on sandbagging successfully. We qualitatively analyzed the uncaught CoTs to understand why the monitor failed. We reveal a rich attack surface for CoT monitoring and contribute five covert sandbagging policies generated by models. These results inform potential failure modes of CoT monitoring and may help build more diverse sandbagging model organisms. 

---
# Trusted Routing for Blockchain-Empowered UAV Networks via Multi-Agent Deep Reinforcement Learning 

**Authors**: Ziye Jia, Sijie He, Qiuming Zhu, Wei Wang, Qihui Wu, Zhu Han  

**Link**: [PDF](https://arxiv.org/pdf/2508.00938)  

**Abstract**: Due to the high flexibility and versatility, unmanned aerial vehicles (UAVs) are leveraged in various fields including surveillance and disaster this http URL, in UAV networks, routing is vulnerable to malicious damage due to distributed topologies and high dynamics. Hence, ensuring the routing security of UAV networks is challenging. In this paper, we characterize the routing process in a time-varying UAV network with malicious nodes. Specifically, we formulate the routing problem to minimize the total delay, which is an integer linear programming and intractable to solve. Then, to tackle the network security issue, a blockchain-based trust management mechanism (BTMM) is designed to dynamically evaluate trust values and identify low-trust UAVs. To improve traditional practical Byzantine fault tolerance algorithms in the blockchain, we propose a consensus UAV update mechanism. Besides, considering the local observability, the routing problem is reformulated into a decentralized partially observable Markov decision process. Further, a multi-agent double deep Q-network based routing algorithm is designed to minimize the total delay. Finally, simulations are conducted with attacked UAVs and numerical results show that the delay of the proposed mechanism decreases by 13.39$\%$, 12.74$\%$, and 16.6$\%$ than multi-agent proximal policy optimal algorithms, multi-agent deep Q-network algorithms, and methods without BTMM, respectively. 

---
# Measuring Harmfulness of Computer-Using Agents 

**Authors**: Aaron Xuxiang Tian, Ruofan Zhang, Janet Tang, Jiaxin Wen  

**Link**: [PDF](https://arxiv.org/pdf/2508.00935)  

**Abstract**: Computer-using agents (CUAs), which autonomously control computers to perform multi-step actions, might pose significant safety risks if misused. Existing benchmarks mostly evaluate language models' (LMs) safety risks in chatbots or simple tool-usage scenarios, without granting full computer access. To better evaluate CUAs' misuse risks, we introduce a new benchmark: CUAHarm. CUAHarm consists of 104 expert-written realistic misuse risks, such as disabling firewalls, leaking confidential information, launching denial-of-service attacks, or installing backdoors. We provide a sandbox environment and rule-based verifiable rewards to measure CUAs' success rates in executing these tasks (e.g., whether the firewall is indeed disabled), not just refusal. We evaluate multiple frontier open-source and proprietary LMs, such as Claude Sonnet, GPT-4o, Gemini Pro 1.5, Llama-3.3-70B, and Mistral Large 2. Surprisingly, even without carefully designed jailbreaking prompts, these frontier LMs comply with executing these malicious tasks at a high success rate (e.g., 59% for Claude 3.7 Sonnet). Newer models show higher misuse rates: Claude 3.7 Sonnet succeeds on 15% more tasks than Claude 3.5. While these models are robust to common malicious prompts (e.g., creating a bomb) in chatbot settings, they behave unsafely as CUAs. We further evaluate a leading agentic framework (UI-TARS-1.5) and find that while it improves performance, it also amplifies misuse risks. Benign variants reveal refusals stem from alignment, not capability limits. To mitigate risks, we explore using LMs to monitor CUAs' actions and chain-of-thoughts (CoTs). Monitoring CUAs is significantly harder than chatbot outputs. Monitoring CoTs yields modest gains, with average detection accuracy at only 72%. Even with hierarchical summarization, improvement is limited to 4%. CUAHarm will be released at this https URL. 

---
# OKG-LLM: Aligning Ocean Knowledge Graph with Observation Data via LLMs for Global Sea Surface Temperature Prediction 

**Authors**: Hanchen Yang, Jiaqi Wang, Jiannong Cao, Wengen Li, Jialun Zheng, Yangning Li, Chunyu Miao, Jihong Guan, Shuigeng Zhou, Philip S. Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00933)  

**Abstract**: Sea surface temperature (SST) prediction is a critical task in ocean science, supporting various applications, such as weather forecasting, fisheries management, and storm tracking. While existing data-driven methods have demonstrated significant success, they often neglect to leverage the rich domain knowledge accumulated over the past decades, limiting further advancements in prediction accuracy. The recent emergence of large language models (LLMs) has highlighted the potential of integrating domain knowledge for downstream tasks. However, the application of LLMs to SST prediction remains underexplored, primarily due to the challenge of integrating ocean domain knowledge and numerical data. To address this issue, we propose Ocean Knowledge Graph-enhanced LLM (OKG-LLM), a novel framework for global SST prediction. To the best of our knowledge, this work presents the first systematic effort to construct an Ocean Knowledge Graph (OKG) specifically designed to represent diverse ocean knowledge for SST prediction. We then develop a graph embedding network to learn the comprehensive semantic and structural knowledge within the OKG, capturing both the unique characteristics of individual sea regions and the complex correlations between them. Finally, we align and fuse the learned knowledge with fine-grained numerical SST data and leverage a pre-trained LLM to model SST patterns for accurate prediction. Extensive experiments on the real-world dataset demonstrate that OKG-LLM consistently outperforms state-of-the-art methods, showcasing its effectiveness, robustness, and potential to advance SST prediction. The codes are available in the online repository. 

---
# SmartDate: AI-Driven Precision Sorting and Quality Control in Date Fruits 

**Authors**: Khaled Eskaf  

**Link**: [PDF](https://arxiv.org/pdf/2508.00921)  

**Abstract**: SmartDate is an AI-powered system for automated sorting and quality control of date fruits. It combines deep learning, genetic algorithms, and reinforcement learning to improve classification accuracy and predict shelf life. The system uses high-resolution imaging and Visible-Near-Infrared (VisNIR) spectral sensors to evaluate key features such as moisture, sugar content, and texture. Reinforcement learning enables real-time adaptation to production conditions, while genetic algorithms optimize model parameters. SmartDate achieved 94.5 percent accuracy, 93.1 percent F1-score, and an AUC-ROC of 0.96. The system reduces waste and ensures that only high-quality dates reach the market, setting a new benchmark in smart agriculture. 

---
# Predictive Auditing of Hidden Tokens in LLM APIs via Reasoning Length Estimation 

**Authors**: Ziyao Wang, Guoheng Sun, Yexiao He, Zheyu Shen, Bowei Tian, Ang Li  

**Link**: [PDF](https://arxiv.org/pdf/2508.00912)  

**Abstract**: Commercial LLM services often conceal internal reasoning traces while still charging users for every generated token, including those from hidden intermediate steps, raising concerns of token inflation and potential overbilling. This gap underscores the urgent need for reliable token auditing, yet achieving it is far from straightforward: cryptographic verification (e.g., hash-based signature) offers little assurance when providers control the entire execution pipeline, while user-side prediction struggles with the inherent variance of reasoning LLMs, where token usage fluctuates across domains and prompt styles. To bridge this gap, we present PALACE (Predictive Auditing of LLM APIs via Reasoning Token Count Estimation), a user-side framework that estimates hidden reasoning token counts from prompt-answer pairs without access to internal traces. PALACE introduces a GRPO-augmented adaptation module with a lightweight domain router, enabling dynamic calibration across diverse reasoning tasks and mitigating variance in token usage patterns. Experiments on math, coding, medical, and general reasoning benchmarks show that PALACE achieves low relative error and strong prediction accuracy, supporting both fine-grained cost auditing and inflation detection. Taken together, PALACE represents an important first step toward standardized predictive auditing, offering a practical path to greater transparency, accountability, and user trust. 

---
# Forecasting LLM Inference Performance via Hardware-Agnostic Analytical Modeling 

**Authors**: Rajeev Patwari, Ashish Sirasao, Devleena Das  

**Link**: [PDF](https://arxiv.org/pdf/2508.00904)  

**Abstract**: Large language models (LLMs) have been increasingly deployed as local agents on personal devices with CPUs, NPUs and integrated GPUs. However, forecasting inference performance on devices with such heterogeneity remains challenging due to the dynamic compute and memory demands. Existing approaches rely on GPU benchmarking or machine learning-based latency predictors, which are often hardware-specific and lack generalizability. To this end, we introduce LIFE, a lightweight and modular analytical framework that is comprised of modular analytical model of operators, configurable to characterize LLM inference workloads in a hardware and dataset-agnostic manner. LIFE characterizes the influence of software and model optimizations, such as quantization, KV cache compression, LoRA adapters, chunked prefill, different attentions, and operator fusion, on performance metrics such as time-to-first-token (TTFT), time-per-output-token (TPOT) and tokens-per-second (TPS). LIFE enables performance forecasting using only hardware specifications, such as TOPS and memory bandwidth, without requiring extensive dataset benchmarking. We validate LIFE's forecasting with inference on AMD Ryzen CPUs, NPUs, iGPUs and NVIDIA V100 GPUs, with Llama2-7B variants, demonstrating the utility of LIFE in forecasting LLM performance through lens of system efficiency to enable efficient LLM deployment across different hardware platforms. 

---
# Universal Neurons in GPT-2: Emergence, Persistence, and Functional Impact 

**Authors**: Advey Nandan, Cheng-Ting Chou, Amrit Kurakula, Cole Blondin, Kevin Zhu, Vasu Sharma, Sean O'Brien  

**Link**: [PDF](https://arxiv.org/pdf/2508.00903)  

**Abstract**: We investigate the phenomenon of neuron universality in independently trained GPT-2 Small models, examining how these universal neurons-neurons with consistently correlated activations across models-emerge and evolve throughout training. By analyzing five GPT-2 models at three checkpoints (100k, 200k, 300k steps), we identify universal neurons through pairwise correlation analysis of activations over a dataset of 5 million tokens. Ablation experiments reveal significant functional impacts of universal neurons on model predictions, measured via loss and KL divergence. Additionally, we quantify neuron persistence, demonstrating high stability of universal neurons across training checkpoints, particularly in deeper layers. These findings suggest stable and universal representational structures emerge during neural network training. 

---
# Sparse 3D Perception for Rose Harvesting Robots: A Two-Stage Approach Bridging Simulation and Real-World Applications 

**Authors**: Taha Samavati, Mohsen Soryani, Sina Mansouri  

**Link**: [PDF](https://arxiv.org/pdf/2508.00900)  

**Abstract**: The global demand for medicinal plants, such as Damask roses, has surged with population growth, yet labor-intensive harvesting remains a bottleneck for scalability. To address this, we propose a novel 3D perception pipeline tailored for flower-harvesting robots, focusing on sparse 3D localization of rose centers. Our two-stage algorithm first performs 2D point-based detection on stereo images, followed by depth estimation using a lightweight deep neural network. To overcome the challenge of scarce real-world labeled data, we introduce a photorealistic synthetic dataset generated via Blender, simulating a dynamic rose farm environment with precise 3D annotations. This approach minimizes manual labeling costs while enabling robust model training. We evaluate two depth estimation paradigms: a traditional triangulation-based method and our proposed deep learning framework. Results demonstrate the superiority of our method, achieving an F1 score of 95.6% (synthetic) and 74.4% (real) in 2D detection, with a depth estimation error of 3% at a 2-meter range on synthetic data. The pipeline is optimized for computational efficiency, ensuring compatibility with resource-constrained robotic systems. By bridging the domain gap between synthetic and real-world data, this work advances agricultural automation for specialty crops, offering a scalable solution for precision harvesting. 

---
# Benefits of Feature Extraction and Temporal Sequence Analysis for Video Frame Prediction: An Evaluation of Hybrid Deep Learning Models 

**Authors**: Jose M. Sánchez Velázquez, Mingbo Cai, Andrew Coney, Álvaro J. García- Tejedor, Alberto Nogales  

**Link**: [PDF](https://arxiv.org/pdf/2508.00898)  

**Abstract**: In recent years, advances in Artificial Intelligence have significantly impacted computer science, particularly in the field of computer vision, enabling solutions to complex problems such as video frame prediction. Video frame prediction has critical applications in weather forecasting or autonomous systems and can provide technical improvements, such as video compression and streaming. Among Artificial Intelligence methods, Deep Learning has emerged as highly effective for solving vision-related tasks, although current frame prediction models still have room for enhancement. This paper evaluates several hybrid deep learning approaches that combine the feature extraction capabilities of autoencoders with temporal sequence modelling using Recurrent Neural Networks (RNNs), 3D Convolutional Neural Networks (3D CNNs), and related architectures. The proposed solutions were rigorously evaluated on three datasets that differ in terms of synthetic versus real-world scenarios and grayscale versus color imagery. Results demonstrate that the approaches perform well, with SSIM metrics increasing from 0.69 to 0.82, indicating that hybrid models utilizing 3DCNNs and ConvLSTMs are the most effective, and greyscale videos with real data are the easiest to predict. 

---
# Maximize margins for robust splicing detection 

**Authors**: Julien Simon de Kergunic, Rony Abecidan, Patrick Bas, Vincent Itier  

**Link**: [PDF](https://arxiv.org/pdf/2508.00897)  

**Abstract**: Despite recent progress in splicing detection, deep learning-based forensic tools remain difficult to deploy in practice due to their high sensitivity to training conditions. Even mild post-processing applied to evaluation images can significantly degrade detector performance, raising concerns about their reliability in operational contexts. In this work, we show that the same deep architecture can react very differently to unseen post-processing depending on the learned weights, despite achieving similar accuracy on in-distribution test data. This variability stems from differences in the latent spaces induced by training, which affect how samples are separated internally. Our experiments reveal a strong correlation between the distribution of latent margins and a detector's ability to generalize to post-processed images. Based on this observation, we propose a practical strategy for building more robust detectors: train several variants of the same model under different conditions, and select the one that maximizes latent margins. 

---
# HoneyImage: Verifiable, Harmless, and Stealthy Dataset Ownership Verification for Image Models 

**Authors**: Zhihao Zhu, Jiale Han, Yi Yang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00892)  

**Abstract**: Image-based AI models are increasingly deployed across a wide range of domains, including healthcare, security, and consumer applications. However, many image datasets carry sensitive or proprietary content, raising critical concerns about unauthorized data usage. Data owners therefore need reliable mechanisms to verify whether their proprietary data has been misused to train third-party models. Existing solutions, such as backdoor watermarking and membership inference, face inherent trade-offs between verification effectiveness and preservation of data integrity. In this work, we propose HoneyImage, a novel method for dataset ownership verification in image recognition models. HoneyImage selectively modifies a small number of hard samples to embed imperceptible yet verifiable traces, enabling reliable ownership verification while maintaining dataset integrity. Extensive experiments across four benchmark datasets and multiple model architectures show that HoneyImage consistently achieves strong verification accuracy with minimal impact on downstream performance while maintaining imperceptible. The proposed HoneyImage method could provide data owners with a practical mechanism to protect ownership over valuable image datasets, encouraging safe sharing and unlocking the full transformative potential of data-driven AI. 

---
# Accelerating multiparametric quantitative MRI using self-supervised scan-specific implicit neural representation with model reinforcement 

**Authors**: Ruimin Feng, Albert Jang, Xingxin He, Fang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00891)  

**Abstract**: Purpose: To develop a self-supervised scan-specific deep learning framework for reconstructing accelerated multiparametric quantitative MRI (qMRI).
Methods: We propose REFINE-MORE (REference-Free Implicit NEural representation with MOdel REinforcement), combining an implicit neural representation (INR) architecture with a model reinforcement module that incorporates MR physics constraints. The INR component enables informative learning of spatiotemporal correlations to initialize multiparametric quantitative maps, which are then further refined through an unrolled optimization scheme enforcing data consistency. To improve computational efficiency, REFINE-MORE integrates a low-rank adaptation strategy that promotes rapid model convergence. We evaluated REFINE-MORE on accelerated multiparametric quantitative magnetization transfer imaging for simultaneous estimation of free water spin-lattice relaxation, tissue macromolecular proton fraction, and magnetization exchange rate, using both phantom and in vivo brain data.
Results: Under 4x and 5x accelerations on in vivo data, REFINE-MORE achieved superior reconstruction quality, demonstrating the lowest normalized root-mean-square error and highest structural similarity index compared to baseline methods and other state-of-the-art model-based and deep learning approaches. Phantom experiments further showed strong agreement with reference values, underscoring the robustness and generalizability of the proposed framework. Additionally, the model adaptation strategy improved reconstruction efficiency by approximately fivefold.
Conclusion: REFINE-MORE enables accurate and efficient scan-specific multiparametric qMRI reconstruction, providing a flexible solution for high-dimensional, accelerated qMRI applications. 

---
# FECT: Factuality Evaluation of Interpretive AI-Generated Claims in Contact Center Conversation Transcripts 

**Authors**: Hagyeong Shin, Binoy Robin Dalal, Iwona Bialynicka-Birula, Navjot Matharu, Ryan Muir, Xingwei Yang, Samuel W. K. Wong  

**Link**: [PDF](https://arxiv.org/pdf/2508.00889)  

**Abstract**: Large language models (LLMs) are known to hallucinate, producing natural language outputs that are not grounded in the input, reference materials, or real-world knowledge. In enterprise applications where AI features support business decisions, such hallucinations can be particularly detrimental. LLMs that analyze and summarize contact center conversations introduce a unique set of challenges for factuality evaluation, because ground-truth labels often do not exist for analytical interpretations about sentiments captured in the conversation and root causes of the business problems. To remedy this, we first introduce a \textbf{3D} -- \textbf{Decompose, Decouple, Detach} -- paradigm in the human annotation guideline and the LLM-judges' prompt to ground the factuality labels in linguistically-informed evaluation criteria. We then introduce \textbf{FECT}, a novel benchmark dataset for \textbf{F}actuality \textbf{E}valuation of Interpretive AI-Generated \textbf{C}laims in Contact Center Conversation \textbf{T}ranscripts, labeled under our 3D paradigm. Lastly, we report our findings from aligning LLM-judges on the 3D paradigm. Overall, our findings contribute a new approach for automatically evaluating the factuality of outputs generated by an AI system for analyzing contact center conversations. 

---
# Multi-Grained Temporal-Spatial Graph Learning for Stable Traffic Flow Forecasting 

**Authors**: Zhenan Lin, Yuni Lai, Wai Lun Lo, Richard Tai-Chiu Hsung, Harris Sik-Ho Tsang, Xiaoyu Xue, Kai Zhou, Yulin Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00884)  

**Abstract**: Time-evolving traffic flow forecasting are playing a vital role in intelligent transportation systems and smart cities. However, the dynamic traffic flow forecasting is a highly nonlinear problem with complex temporal-spatial dependencies. Although the existing methods has provided great contributions to mine the temporal-spatial patterns in the complex traffic networks, they fail to encode the globally temporal-spatial patterns and are prone to overfit on the pre-defined geographical correlations, and thus hinder the model's robustness on the complex traffic environment. To tackle this issue, in this work, we proposed a multi-grained temporal-spatial graph learning framework to adaptively augment the globally temporal-spatial patterns obtained from a crafted graph transformer encoder with the local patterns from the graph convolution by a crafted gated fusion unit with residual connection techniques. Under these circumstances, our proposed model can mine the hidden global temporal-spatial relations between each monitor stations and balance the relative importance of local and global temporal-spatial patterns. Experiment results demonstrate the strong representation capability of our proposed method and our model consistently outperforms other strong baselines on various real-world traffic networks. 

---
# Reproducibility of Machine Learning-Based Fault Detection and Diagnosis for HVAC Systems in Buildings: An Empirical Study 

**Authors**: Adil Mukhtar, Michael Hadwiger, Franz Wotawa, Gerald Schweiger  

**Link**: [PDF](https://arxiv.org/pdf/2508.00880)  

**Abstract**: Reproducibility is a cornerstone of scientific research, enabling independent verification and validation of empirical findings. The topic gained prominence in fields such as psychology and medicine, where concerns about non - replicable results sparked ongoing discussions about research practices. In recent years, the fast-growing field of Machine Learning (ML) has become part of this discourse, as it faces similar concerns about transparency and reliability. Some reproducibility issues in ML research are shared with other fields, such as limited access to data and missing methodological details. In addition, ML introduces specific challenges, including inherent nondeterminism and computational constraints. While reproducibility issues are increasingly recognized by the ML community and its major conferences, less is known about how these challenges manifest in applied disciplines. This paper contributes to closing this gap by analyzing the transparency and reproducibility standards of ML applications in building energy systems. The results indicate that nearly all articles are not reproducible due to insufficient disclosure across key dimensions of reproducibility. 72% of the articles do not specify whether the dataset used is public, proprietary, or commercially available. Only two papers share a link to their code - one of which was broken. Two-thirds of the publications were authored exclusively by academic researchers, yet no significant differences in reproducibility were observed compared to publications with industry-affiliated authors. These findings highlight the need for targeted interventions, including reproducibility guidelines, training for researchers, and policies by journals and conferences that promote transparency and reproducibility. 

---
# GNN-ASE: Graph-Based Anomaly Detection and Severity Estimation in Three-Phase Induction Machines 

**Authors**: Moutaz Bellah Bentrad, Adel Ghoggal, Tahar Bahi, Abderaouf Bahi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00879)  

**Abstract**: The diagnosis of induction machines has traditionally relied on model-based methods that require the development of complex dynamic models, making them difficult to implement and computationally expensive. To overcome these limitations, this paper proposes a model-free approach using Graph Neural Networks (GNNs) for fault diagnosis in induction machines. The focus is on detecting multiple fault types -- including eccentricity, bearing defects, and broken rotor bars -- under varying severity levels and load conditions. Unlike traditional approaches, raw current and vibration signals are used as direct inputs, eliminating the need for signal preprocessing or manual feature extraction. The proposed GNN-ASE model automatically learns and extracts relevant features from raw inputs, leveraging the graph structure to capture complex relationships between signal types and fault patterns. It is evaluated for both individual fault detection and multi-class classification of combined fault conditions. Experimental results demonstrate the effectiveness of the proposed model, achieving 92.5\% accuracy for eccentricity defects, 91.2\% for bearing faults, and 93.1\% for broken rotor bar detection. These findings highlight the model's robustness and generalization capability across different operational scenarios. The proposed GNN-based framework offers a lightweight yet powerful solution that simplifies implementation while maintaining high diagnostic performance. It stands as a promising alternative to conventional model-based diagnostic techniques for real-world induction machine monitoring and predictive maintenance. 

---
# Satellite Connectivity Prediction for Fast-Moving Platforms 

**Authors**: Chao Yan, Babak Mafakheri  

**Link**: [PDF](https://arxiv.org/pdf/2508.00877)  

**Abstract**: Satellite connectivity is gaining increased attention as the demand for seamless internet access, especially in transportation and remote areas, continues to grow. For fast-moving objects such as aircraft, vehicles, or trains, satellite connectivity is critical due to their mobility and frequent presence in areas without terrestrial coverage. Maintaining reliable connectivity in these cases requires frequent switching between satellite beams, constellations, or orbits. To enhance user experience and address challenges like long switching times, Machine Learning (ML) algorithms can analyze historical connectivity data and predict network quality at specific locations. This allows for proactive measures, such as network switching before connectivity issues arise. In this paper, we analyze a real dataset of communication between a Geostationary Orbit (GEO) satellite and aircraft over multiple flights, using ML to predict signal quality. Our prediction model achieved an F1 score of 0.97 on the test data, demonstrating the accuracy of machine learning in predicting signal quality during flight. By enabling seamless broadband service, including roaming between different satellite constellations and providers, our model addresses the need for real-time predictions of signal quality. This approach can further be adapted to automate satellite and beam-switching mechanisms to improve overall communication efficiency. The model can also be retrained and applied to any moving object with satellite connectivity, using customized datasets, including connected vehicles and trains. 

---
# Patents as Knowledge Artifacts: An Information Science Perspective on Global Innovation 

**Authors**: M. S. Rajeevan, B. Mini Devi  

**Link**: [PDF](https://arxiv.org/pdf/2508.00871)  

**Abstract**: In an age of fast-paced technological change, patents have evolved into not only legal mechanisms of intellectual property, but also structured storage containers of knowledge full of metadata, categories, and formal innovation. This chapter proposes to reframe patents in the context of information science, by focusing on patents as knowledge artifacts, and by seeing patents as fundamentally tied to the global movement of scientific and technological knowledge. With a focus on three areas, the inventions of AIs, biotech patents, and international competition with patents, this work considers how new technologies are challenging traditional notions of inventorship, access, and moral this http URL chapter provides a critical analysis of AI's implications for patent authorship and prior art searches, ownership issues arising from proprietary claims in biotechnology to ethical dilemmas, and the problem of using patents for strategic advantage in a global context of innovation competition. In this analysis, the chapter identified the importance of organizing information, creating metadata standards about originality, implementing retrieval systems to access previous works, and ethical contemplation about patenting unseen relationships in innovation ecosystems. Ultimately, the chapter called for a collaborative, transparent, and ethically-based approach in managing knowledge in the patenting environment highlighting the role for information professionals and policy to contribute to access equity in innovation. 

---
# Better Recommendations: Validating AI-generated Subject Terms Through LOC Linked Data Service 

**Authors**: Kwok Leong Tang, Yi Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00867)  

**Abstract**: This article explores the integration of AI-generated subject terms into library cataloging, focusing on validation through the Library of Congress Linked Data Service. It examines the challenges of traditional subject cataloging under the Library of Congress Subject Headings system, including inefficiencies and cataloging backlogs. While generative AI shows promise in expediting cataloging workflows, studies reveal significant limitations in the accuracy of AI-assigned subject headings. The article proposes a hybrid approach combining AI technology with human validation through LOC Linked Data Service, aiming to enhance the precision, efficiency, and overall quality of metadata creation in library cataloging practices. 

---
# Deploying Geospatial Foundation Models in the Real World: Lessons from WorldCereal 

**Authors**: Christina Butsko, Kristof Van Tricht, Gabriel Tseng, Giorgia Milli, David Rolnick, Ruben Cartuyvels, Inbal Becker Reshef, Zoltan Szantoi, Hannah Kerner  

**Link**: [PDF](https://arxiv.org/pdf/2508.00858)  

**Abstract**: The increasing availability of geospatial foundation models has the potential to transform remote sensing applications such as land cover classification, environmental monitoring, and change detection. Despite promising benchmark results, the deployment of these models in operational settings is challenging and rare. Standardized evaluation tasks often fail to capture real-world complexities relevant for end-user adoption such as data heterogeneity, resource constraints, and application-specific requirements. This paper presents a structured approach to integrate geospatial foundation models into operational mapping systems. Our protocol has three key steps: defining application requirements, adapting the model to domain-specific data and conducting rigorous empirical testing. Using the Presto model in a case study for crop mapping, we demonstrate that fine-tuning a pre-trained model significantly improves performance over conventional supervised methods. Our results highlight the model's strong spatial and temporal generalization capabilities. Our protocol provides a replicable blueprint for practitioners and lays the groundwork for future research to operationalize foundation models in diverse remote sensing applications. Application of the protocol to the WorldCereal global crop-mapping system showcases the framework's scalability. 

---
# EthicAlly: a Prototype for AI-Powered Research Ethics Support for the Social Sciences and Humanities 

**Authors**: Steph Grohmann  

**Link**: [PDF](https://arxiv.org/pdf/2508.00856)  

**Abstract**: In biomedical science, review by a Research Ethics Committee (REC) is an indispensable way of protecting human subjects from harm. However, in social science and the humanities, mandatory ethics compliance has long been met with scepticism as biomedical models of ethics can map poorly onto methodologies involving complex socio-political and cultural considerations. As a result, tailored ethics training and support as well as access to RECs with the necessary expertise is lacking in some areas, including parts of Europe and low- and middle-income countries. This paper suggests that Generative AI can meaningfully contribute to closing these gaps, illustrating this claim by presenting EthicAlly, a proof-of-concept prototype for an AI-powered ethics support system for social science and humanities researchers. Drawing on constitutional AI technology and a collaborative prompt development methodology, EthicAlly provides structured ethics assessment that incorporates both universal ethics principles and contextual and interpretive considerations relevant to most social science research. In supporting researchers in ethical research design and preparation for REC submission, this kind of system can also contribute to easing the burden on institutional RECs, without attempting to automate or replace human ethical oversight. 

---
# Gearshift Fellowship: A Next-Generation Neurocomputational Game Platform to Model and Train Human-AI Adaptability 

**Authors**: Nadja R. Ging-Jehli, Russell K. Childers, Joshua Lu, Robert Gemma, Rachel Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00850)  

**Abstract**: How do we learn when to persist, when to let go, and when to shift gears? Gearshift Fellowship (GF) is the prototype of a new Supertask paradigm designed to model how humans and artificial agents adapt to shifting environment demands. Grounded in cognitive neuroscience, computational psychiatry, economics, and artificial intelligence, Supertasks combine computational neurocognitive modeling with serious gaming. This creates a dynamic, multi-mission environment engineered to assess mechanisms of adaptive behavior across cognitive and social contexts. Computational parameters explain behavior and probe mechanisms by controlling the game environment. Unlike traditional tasks, GF enables neurocognitive modeling of individual differences across perceptual decisions, learning, and meta-cognitive levels. This positions GF as a flexible testbed for understanding how cognitive-affective control processes, learning styles, strategy use, and motivational shifts adapt across contexts and over time. It serves as an experimental platform for scientists, a phenotype-to-mechanism intervention for clinicians, and a training tool for players aiming to strengthen self-regulated learning, mood, and stress resilience. Online study (n = 60, ongoing) results show that GF recovers effects from traditional neuropsychological tasks (construct validity), uncovers novel patterns in how learning differs across contexts and how clinical features map onto distinct adaptations. These findings pave the way for developing in-game interventions that foster self-efficacy and agency to cope with real-world stress and uncertainty. GF builds a new adaptive ecosystem designed to accelerate science, transform clinical care, and foster individual growth. It offers a mirror and training ground where humans and machines co-develop together deeper flexibility and awareness. 

---
# Cognitive Exoskeleton: Augmenting Human Cognition with an AI-Mediated Intelligent Visual Feedback 

**Authors**: Songlin Xu, Xinyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2508.00846)  

**Abstract**: In this paper, we introduce an AI-mediated framework that can provide intelligent feedback to augment human cognition. Specifically, we leverage deep reinforcement learning (DRL) to provide adaptive time pressure feedback to improve user performance in a math arithmetic task. Time pressure feedback could either improve or deteriorate user performance by regulating user attention and anxiety. Adaptive time pressure feedback controlled by a DRL policy according to users' real-time performance could potentially solve this trade-off problem. However, the DRL training and hyperparameter tuning may require large amounts of data and iterative user studies. Therefore, we propose a dual-DRL framework that trains a regulation DRL agent to regulate user performance by interacting with another simulation DRL agent that mimics user cognition behaviors from an existing dataset. Our user study demonstrates the feasibility and effectiveness of the dual-DRL framework in augmenting user performance, in comparison to the baseline group. 

---
# Generative AI for CAD Automation: Leveraging Large Language Models for 3D Modelling 

**Authors**: Sumit Kumar, Sarthak Kapoor, Harsh Vardhan, Yao Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2508.00843)  

**Abstract**: Large Language Models (LLMs) are revolutionizing industries by enhancing efficiency, scalability, and innovation. This paper investigates the potential of LLMs in automating Computer-Aided Design (CAD) workflows, by integrating FreeCAD with LLM as CAD design tool. Traditional CAD processes are often complex and require specialized sketching skills, posing challenges for rapid prototyping and generative design. We propose a framework where LLMs generate initial CAD scripts from natural language descriptions, which are then executed and refined iteratively based on error feedback. Through a series of experiments with increasing complexity, we assess the effectiveness of this approach. Our findings reveal that LLMs perform well for simple to moderately complex designs but struggle with highly constrained models, necessitating multiple refinements. The study highlights the need for improved memory retrieval, adaptive prompt engineering, and hybrid AI techniques to enhance script robustness. Future directions include integrating cloud-based execution and exploring advanced LLM capabilities to further streamline CAD automation. This work underscores the transformative potential of LLMs in design workflows while identifying critical areas for future development. 

---
# Inclusive Review on Advances in Masked Human Face Recognition Technologies 

**Authors**: Ali Haitham Abdul Amir, Zainab N. Nemer  

**Link**: [PDF](https://arxiv.org/pdf/2508.00841)  

**Abstract**: Masked Face Recognition (MFR) is an increasingly important area in biometric recognition technologies, especially with the widespread use of masks as a result of the COVID-19 pandemic. This development has created new challenges for facial recognition systems due to the partial concealment of basic facial features. This paper aims to provide a comprehensive review of the latest developments in the field, with a focus on deep learning techniques, especially convolutional neural networks (CNNs) and twin networks (Siamese networks), which have played a pivotal role in improving the accuracy of covering face recognition. The paper discusses the most prominent challenges, which include changes in lighting, different facial positions, partial concealment, and the impact of mask types on the performance of systems. It also reviews advanced technologies developed to overcome these challenges, including data enhancement using artificial databases and multimedia methods to improve the ability of systems to generalize. In addition, the paper highlights advance in deep network design, feature extraction techniques, evaluation criteria, and data sets used in this area. Moreover, it reviews the various applications of masked face recognition in the fields of security and medicine, highlighting the growing importance of these systems in light of recurrent health crises and increasing security threats. Finally, the paper focuses on future research trends such as developing more efficient algorithms and integrating multimedia technologies to improve the performance of recognition systems in real-world environments and expand their applications. 

---
# The Attribution Crisis in LLM Search Results 

**Authors**: Ilan Strauss, Jangho Yang, Tim O'Reilly, Sruly Rosenblat, Isobel Moure  

**Link**: [PDF](https://arxiv.org/pdf/2508.00838)  

**Abstract**: Web-enabled LLMs frequently answer queries without crediting the web pages they consume, creating an "attribution gap" - the difference between relevant URLs read and those actually cited. Drawing on approximately 14,000 real-world LMArena conversation logs with search-enabled LLM systems, we document three exploitation patterns: 1) No Search: 34% of Google Gemini and 24% of OpenAI GPT-4o responses are generated without explicitly fetching any online content; 2) No citation: Gemini provides no clickable citation source in 92% of answers; 3) High-volume, low-credit: Perplexity's Sonar visits approximately 10 relevant pages per query but cites only three to four. A negative binomial hurdle model shows that the average query answered by Gemini or Sonar leaves about 3 relevant websites uncited, whereas GPT-4o's tiny uncited gap is best explained by its selective log disclosures rather than by better attribution. Citation efficiency - extra citations provided per additional relevant web page visited - varies widely across models, from 0.19 to 0.45 on identical queries, underscoring that retrieval design, not technical limits, shapes ecosystem impact. We recommend a transparent LLM search architecture based on standardized telemetry and full disclosure of search traces and citation logs. 

---
# PCS Workflow for Veridical Data Science in the Age of AI 

**Authors**: Zachary T. Rewolinski, Bin Yu  

**Link**: [PDF](https://arxiv.org/pdf/2508.00835)  

**Abstract**: Data science is a pillar of artificial intelligence (AI), which is transforming nearly every domain of human activity, from the social and physical sciences to engineering and medicine. While data-driven findings in AI offer unprecedented power to extract insights and guide decision-making, many are difficult or impossible to replicate. A key reason for this challenge is the uncertainty introduced by the many choices made throughout the data science life cycle (DSLC). Traditional statistical frameworks often fail to account for this uncertainty. The Predictability-Computability-Stability (PCS) framework for veridical (truthful) data science offers a principled approach to addressing this challenge throughout the DSLC. This paper presents an updated and streamlined PCS workflow, tailored for practitioners and enhanced with guided use of generative AI. We include a running example to display the PCS framework in action, and conduct a related case study which showcases the uncertainty in downstream predictions caused by judgment calls in the data cleaning stage. 

---
# Bike-Bench: A Bicycle Design Benchmark for Generative Models with Objectives and Constraints 

**Authors**: Lyle Regenwetter, Yazan Abu Obaideh, Fabien Chiotti, Ioanna Lykourentzou, Faez Ahmed  

**Link**: [PDF](https://arxiv.org/pdf/2508.00830)  

**Abstract**: We introduce Bike-Bench, an engineering design benchmark for evaluating generative models on problems with multiple real-world objectives and constraints. As generative AI's reach continues to grow, evaluating its capability to understand physical laws, human guidelines, and hard constraints grows increasingly important. Engineering product design lies at the intersection of these difficult tasks, providing new challenges for AI capabilities. Bike-Bench evaluates AI models' capability to generate designs that not only resemble the dataset, but meet specific performance objectives and constraints. To do so, Bike-Bench quantifies a variety of human-centered and multiphysics performance characteristics, such as aerodynamics, ergonomics, structural mechanics, human-rated usability, and similarity to subjective text or image prompts. Supporting the benchmark are several datasets of simulation results, a dataset of 10K human-rated bicycle assessments, and a synthetically-generated dataset of 1.4M designs, each with a parametric, CAD/XML, SVG, and PNG representation. Bike-Bench is uniquely configured to evaluate tabular generative models, LLMs, design optimization, and hybrid algorithms side-by-side. Our experiments indicate that LLMs and tabular generative models fall short of optimization and optimization-augmented generative models in both validity and optimality scores, suggesting significant room for improvement. We hope Bike-Bench, a first-of-its-kind benchmark, will help catalyze progress in generative AI for constrained multi-objective engineering design problems. Code, data, and other resources are published at this http URL. 

---
# A Schema.org Mapping for Brazilian Legal Norms: Toward Interoperable Legal Graphs and Open Government Data 

**Authors**: Hudson de Martim  

**Link**: [PDF](https://arxiv.org/pdf/2508.00827)  

**Abstract**: Open Government Data (OGD) initiatives aim to enhance transparency and public participation by making government data openly accessible. However, structuring legal norms for machine readability remains a critical challenge for advancing Legal Tech applications such as Legal Knowledge Graphs (LKGs). Focusing on the this http URL portal initiative by the Brazilian National Congress, we propose a unified mapping of Brazilian legislation to the this http URL vocabulary via JSON-LD and Linked Data. Our approach covers both the conceptual "Norm" entity (mapped to sdo:Legislation) and its digital publications or manifestations (mapped to sdo:LegislationObject). We detail key properties for each type, providing concrete examples and considering URN identifiers (per the LexML standard), multilingual support, versioning in the Official Journal, and inter-norm relationships (e.g., citations and references). Our structured schema improves the quality and interoperability of Brazilian legal data, fosters integration within the global OGD ecosystem, and facilitates the creation of a wor 

---
# EH-Benchmark Ophthalmic Hallucination Benchmark and Agent-Driven Top-Down Traceable Reasoning Workflow 

**Authors**: Xiaoyu Pan, Yang Bai, Ke Zou, Yang Zhou, Jun Zhou, Huazhu Fu, Yih-Chung Tham, Yong Liu  

**Link**: [PDF](https://arxiv.org/pdf/2507.22929)  

**Abstract**: Medical Large Language Models (MLLMs) play a crucial role in ophthalmic diagnosis, holding significant potential to address vision-threatening diseases. However, their accuracy is constrained by hallucinations stemming from limited ophthalmic knowledge, insufficient visual localization and reasoning capabilities, and a scarcity of multimodal ophthalmic data, which collectively impede precise lesion detection and disease diagnosis. Furthermore, existing medical benchmarks fail to effectively evaluate various types of hallucinations or provide actionable solutions to mitigate them. To address the above challenges, we introduce EH-Benchmark, a novel ophthalmology benchmark designed to evaluate hallucinations in MLLMs. We categorize MLLMs' hallucinations based on specific tasks and error types into two primary classes: Visual Understanding and Logical Composition, each comprising multiple subclasses. Given that MLLMs predominantly rely on language-based reasoning rather than visual processing, we propose an agent-centric, three-phase framework, including the Knowledge-Level Retrieval stage, the Task-Level Case Studies stage, and the Result-Level Validation stage. Experimental results show that our multi-agent framework significantly mitigates both types of hallucinations, enhancing accuracy, interpretability, and reliability. Our project is available at this https URL. 

---
# Zero-Shot Temporal Interaction Localization for Egocentric Videos 

**Authors**: Erhang Zhang, Junyi Ma, Yin-Dong Zheng, Yixuan Zhou, Hesheng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2506.03662)  

**Abstract**: Locating human-object interaction (HOI) actions within video serves as the foundation for multiple downstream tasks, such as human behavior analysis and human-robot skill transfer. Current temporal action localization methods typically rely on annotated action and object categories of interactions for optimization, which leads to domain bias and low deployment efficiency. Although some recent works have achieved zero-shot temporal action localization (ZS-TAL) with large vision-language models (VLMs), their coarse-grained estimations and open-loop pipelines hinder further performance improvements for temporal interaction localization (TIL). To address these issues, we propose a novel zero-shot TIL approach dubbed EgoLoc to locate the timings of grasp actions for human-object interaction in egocentric videos. EgoLoc introduces a self-adaptive sampling strategy to generate reasonable visual prompts for VLM reasoning. By absorbing both 2D and 3D observations, it directly samples high-quality initial guesses around the possible contact/separation timestamps of HOI according to 3D hand velocities, leading to high inference accuracy and efficiency. In addition, EgoLoc generates closed-loop feedback from visual and dynamic cues to further refine the localization results. Comprehensive experiments on the publicly available dataset and our newly proposed benchmark demonstrate that EgoLoc achieves better temporal interaction localization for egocentric videos compared to state-of-the-art baselines. We will release our code and relevant data as open-source at this https URL. 

---
# Towards Actionable Pedagogical Feedback: A Multi-Perspective Analysis of Mathematics Teaching and Tutoring Dialogue 

**Authors**: Jannatun Naim, Jie Cao, Fareen Tasneem, Jennifer Jacobs, Brent Milne, James Martin, Tamara Sumner  

**Link**: [PDF](https://arxiv.org/pdf/2505.07161)  

**Abstract**: Effective feedback is essential for refining instructional practices in mathematics education, and researchers often turn to advanced natural language processing (NLP) models to analyze classroom dialogues from multiple perspectives. However, utterance-level discourse analysis encounters two primary challenges: (1) multifunctionality, where a single utterance may serve multiple purposes that a single tag cannot capture, and (2) the exclusion of many utterances from domain-specific discourse move classifications, leading to their omission in feedback. To address these challenges, we proposed a multi-perspective discourse analysis that integrates domain-specific talk moves with dialogue act (using the flattened multi-functional SWBD-MASL schema with 43 tags) and discourse relation (applying Segmented Discourse Representation Theory with 16 relations). Our top-down analysis framework enables a comprehensive understanding of utterances that contain talk moves, as well as utterances that do not contain talk moves. This is applied to two mathematics education datasets: TalkMoves (teaching) and SAGA22 (tutoring). Through distributional unigram analysis, sequential talk move analysis, and multi-view deep dive, we discovered meaningful discourse patterns, and revealed the vital role of utterances without talk moves, demonstrating that these utterances, far from being mere fillers, serve crucial functions in guiding, acknowledging, and structuring classroom discourse. These insights underscore the importance of incorporating discourse relations and dialogue acts into AI-assisted education systems to enhance feedback and create more responsive learning environments. Our framework may prove helpful for providing human educator feedback, but also aiding in the development of AI agents that can effectively emulate the roles of both educators and students. 

---
# Enhancing Talk Moves Analysis in Mathematics Tutoring through Classroom Teaching Discourse 

**Authors**: Jie Cao, Abhijit Suresh, Jennifer Jacobs, Charis Clevenger, Amanda Howard, Chelsea Brown, Brent Milne, Tom Fischaber, Tamara Sumner, James H. Martin  

**Link**: [PDF](https://arxiv.org/pdf/2412.13395)  

**Abstract**: Human tutoring interventions play a crucial role in supporting student learning, improving academic performance, and promoting personal growth. This paper focuses on analyzing mathematics tutoring discourse using talk moves - a framework of dialogue acts grounded in Accountable Talk theory. However, scaling the collection, annotation, and analysis of extensive tutoring dialogues to develop machine learning models is a challenging and resource-intensive task. To address this, we present SAGA22, a compact dataset, and explore various modeling strategies, including dialogue context, speaker information, pretraining datasets, and further fine-tuning. By leveraging existing datasets and models designed for classroom teaching, our results demonstrate that supplementary pretraining on classroom data enhances model performance in tutoring settings, particularly when incorporating longer context and speaker information. Additionally, we conduct extensive ablation studies to underscore the challenges in talk move modeling. 

---
# Observing Dialogue in Therapy: Categorizing and Forecasting Behavioral Codes 

**Authors**: Jie Cao, Michael Tanana, Zac E. Imel, Eric Poitras, David C. Atkins, Vivek Srikumar  

**Link**: [PDF](https://arxiv.org/pdf/1907.00326)  

**Abstract**: Automatically analyzing dialogue can help understand and guide behavior in domains such as counseling, where interactions are largely mediated by conversation. In this paper, we study modeling behavioral codes used to asses a psychotherapy treatment style called Motivational Interviewing (MI), which is effective for addressing substance abuse and related problems. Specifically, we address the problem of providing real-time guidance to therapists with a dialogue observer that (1) categorizes therapist and client MI behavioral codes and, (2) forecasts codes for upcoming utterances to help guide the conversation and potentially alert the therapist. For both tasks, we define neural network models that build upon recent successes in dialogue modeling. Our experiments demonstrate that our models can outperform several baselines for both tasks. We also report the results of a careful analysis that reveals the impact of the various network design tradeoffs for modeling therapy dialogue. 

---
