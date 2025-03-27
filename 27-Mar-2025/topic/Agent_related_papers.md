# Beyond Believability: Accurate Human Behavior Simulation with Fine-Tuned LLMs 

**Authors**: Yuxuan Lu, Jing Huang, Yan Han, Bennet Bei, Yaochen Xie, Dakuo Wang, Jessie Wang, Qi He  

**Link**: [PDF](https://arxiv.org/pdf/2503.20749)  

**Abstract**: Recent research shows that LLMs can simulate ``believable'' human behaviors to power LLM agents via prompt-only methods. In this work, we focus on evaluating and improving LLM's objective ``accuracy'' rather than the subjective ``believability'' in the web action generation task, leveraging a large-scale, real-world dataset collected from online shopping human actions. We present the first comprehensive quantitative evaluation of state-of-the-art LLMs (e.g., DeepSeek-R1, Llama, and Claude) on the task of web action generation. Our results show that fine-tuning LLMs on real-world behavioral data substantially improves their ability to generate actions compared to prompt-only methods. Furthermore, incorporating synthesized reasoning traces into model training leads to additional performance gains, demonstrating the value of explicit rationale in behavior modeling. This work establishes a new benchmark for evaluating LLMs in behavior simulation and offers actionable insights into how real-world action data and reasoning augmentation can enhance the fidelity of LLM agents. 

---
# TAMA: A Human-AI Collaborative Thematic Analysis Framework Using Multi-Agent LLMs for Clinical Interviews 

**Authors**: Huimin Xu, Seungjun Yi, Terence Lim, Jiawei Xu, Andrew Well, Carlos Mery, Aidong Zhang, Yuji Zhang, Heng Ji, Keshav Pingali, Yan Leng, Ying Ding  

**Link**: [PDF](https://arxiv.org/pdf/2503.20666)  

**Abstract**: Thematic analysis (TA) is a widely used qualitative approach for uncovering latent meanings in unstructured text data. TA provides valuable insights in healthcare but is resource-intensive. Large Language Models (LLMs) have been introduced to perform TA, yet their applications in healthcare remain unexplored. Here, we propose TAMA: A Human-AI Collaborative Thematic Analysis framework using Multi-Agent LLMs for clinical interviews. We leverage the scalability and coherence of multi-agent systems through structured conversations between agents and coordinate the expertise of cardiac experts in TA. Using interview transcripts from parents of children with Anomalous Aortic Origin of a Coronary Artery (AAOCA), a rare congenital heart disease, we demonstrate that TAMA outperforms existing LLM-assisted TA approaches, achieving higher thematic hit rate, coverage, and distinctiveness. TAMA demonstrates strong potential for automated TA in clinical settings by leveraging multi-agent LLM systems with human-in-the-loop integration by enhancing quality while significantly reducing manual workload. 

---
# Open Deep Search: Democratizing Search with Open-source Reasoning Agents 

**Authors**: Salaheddin Alzubi, Creston Brooks, Purva Chiniya, Edoardo Contente, Chiara von Gerlach, Lucas Irwin, Yihan Jiang, Arda Kaz, Windsor Nguyen, Sewoong Oh, Himanshu Tyagi, Pramod Viswanath  

**Link**: [PDF](https://arxiv.org/pdf/2503.20201)  

**Abstract**: We introduce Open Deep Search (ODS) to close the increasing gap between the proprietary search AI solutions, such as Perplexity's Sonar Reasoning Pro and OpenAI's GPT-4o Search Preview, and their open-source counterparts. The main innovation introduced in ODS is to augment the reasoning capabilities of the latest open-source LLMs with reasoning agents that can judiciously use web search tools to answer queries. Concretely, ODS consists of two components that work with a base LLM chosen by the user: Open Search Tool and Open Reasoning Agent. Open Reasoning Agent interprets the given task and completes it by orchestrating a sequence of actions that includes calling tools, one of which is the Open Search Tool. Open Search Tool is a novel web search tool that outperforms proprietary counterparts. Together with powerful open-source reasoning LLMs, such as DeepSeek-R1, ODS nearly matches and sometimes surpasses the existing state-of-the-art baselines on two benchmarks: SimpleQA and FRAMES. For example, on the FRAMES evaluation benchmark, ODS improves the best existing baseline of the recently released GPT-4o Search Preview by 9.7% in accuracy. ODS is a general framework for seamlessly augmenting any LLMs -- for example, DeepSeek-R1 that achieves 82.4% on SimpleQA and 30.1% on FRAMES -- with search and reasoning capabilities to achieve state-of-the-art performance: 88.3% on SimpleQA and 75.3% on FRAMES. 

---
# Perspective-Shifted Neuro-Symbolic World Models: A Framework for Socially-Aware Robot Navigation 

**Authors**: Kevin Alcedo, Pedro U. Lima, Rachid Alami  

**Link**: [PDF](https://arxiv.org/pdf/2503.20425)  

**Abstract**: Navigating in environments alongside humans requires agents to reason under uncertainty and account for the beliefs and intentions of those around them. Under a sequential decision-making framework, egocentric navigation can naturally be represented as a Markov Decision Process (MDP). However, social navigation additionally requires reasoning about the hidden beliefs of others, inherently leading to a Partially Observable Markov Decision Process (POMDP), where agents lack direct access to others' mental states. Inspired by Theory of Mind and Epistemic Planning, we propose (1) a neuro-symbolic model-based reinforcement learning architecture for social navigation, addressing the challenge of belief tracking in partially observable environments; and (2) a perspective-shift operator for belief estimation, leveraging recent work on Influence-based Abstractions (IBA) in structured multi-agent settings. 

---
# OmniNova:A General Multimodal Agent Framework 

**Authors**: Pengfei Du  

**Link**: [PDF](https://arxiv.org/pdf/2503.20028)  

**Abstract**: The integration of Large Language Models (LLMs) with specialized tools presents new opportunities for intelligent automation systems. However, orchestrating multiple LLM-driven agents to tackle complex tasks remains challenging due to coordination difficulties, inefficient resource utilization, and inconsistent information flow. We present OmniNova, a modular multi-agent automation framework that combines language models with specialized tools such as web search, crawling, and code execution capabilities. OmniNova introduces three key innovations: (1) a hierarchical multi-agent architecture with distinct coordinator, planner, supervisor, and specialist agents; (2) a dynamic task routing mechanism that optimizes agent deployment based on task complexity; and (3) a multi-layered LLM integration system that allocates appropriate models to different cognitive requirements. Our evaluations across 50 complex tasks in research, data analysis, and web interaction domains demonstrate that OmniNova outperforms existing frameworks in task completion rate (87\% vs. baseline 62\%), efficiency (41\% reduced token usage), and result quality (human evaluation score of 4.2/5 vs. baseline 3.1/5). We contribute both a theoretical framework for multi-agent system design and an open-source implementation that advances the state-of-the-art in LLM-based automation systems. 

---
# Synthesizing world models for bilevel planning 

**Authors**: Zergham Ahmed, Joshua B. Tenenbaum, Christopher J. Bates, Samuel J. Gershman  

**Link**: [PDF](https://arxiv.org/pdf/2503.20124)  

**Abstract**: Modern reinforcement learning (RL) systems have demonstrated remarkable capabilities in complex environments, such as video games. However, they still fall short of achieving human-like sample efficiency and adaptability when learning new domains. Theory-based reinforcement learning (TBRL) is an algorithmic framework specifically designed to address this gap. Modeled on cognitive theories, TBRL leverages structured, causal world models - "theories" - as forward simulators for use in planning, generalization and exploration. Although current TBRL systems provide compelling explanations of how humans learn to play video games, they face several technical limitations: their theory languages are restrictive, and their planning algorithms are not scalable. To address these challenges, we introduce TheoryCoder, an instantiation of TBRL that exploits hierarchical representations of theories and efficient program synthesis methods for more powerful learning and planning. TheoryCoder equips agents with general-purpose abstractions (e.g., "move to"), which are then grounded in a particular environment by learning a low-level transition model (a Python program synthesized from observations by a large language model). A bilevel planning algorithm can exploit this hierarchical structure to solve large domains. We demonstrate that this approach can be successfully applied to diverse and challenging grid-world games, where approaches based on directly synthesizing a policy perform poorly. Ablation studies demonstrate the benefits of using hierarchical abstractions. 

---
# Graph-Enhanced Model-Free Reinforcement Learning Agents for Efficient Power Grid Topological Control 

**Authors**: Eloy Anguiano Batanero, Ángela Fernández, Álvaro Barbero  

**Link**: [PDF](https://arxiv.org/pdf/2503.20688)  

**Abstract**: The increasing complexity of power grid management, driven by the emergence of prosumers and the demand for cleaner energy solutions, has needed innovative approaches to ensure stability and efficiency. This paper presents a novel approach within the model-free framework of reinforcement learning, aimed at optimizing power network operations without prior expert knowledge. We introduce a masked topological action space, enabling agents to explore diverse strategies for cost reduction while maintaining reliable service using the state logic as a guide for choosing proper actions. Through extensive experimentation across 20 different scenarios in a simulated 5-substation environment, we demonstrate that our approach achieves a consistent reduction in power losses, while ensuring grid stability against potential blackouts. The results underscore the effectiveness of combining dynamic observation formalization with opponent-based training, showing a viable way for autonomous management solutions in modern energy systems or even for building a foundational model for this field. 

---
# GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving 

**Authors**: Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, Gianluca Corrado  

**Link**: [PDF](https://arxiv.org/pdf/2503.20523)  

**Abstract**: Generative models offer a scalable and flexible paradigm for simulating complex environments, yet current approaches fall short in addressing the domain-specific requirements of autonomous driving - such as multi-agent interactions, fine-grained control, and multi-camera consistency. We introduce GAIA-2, Generative AI for Autonomy, a latent diffusion world model that unifies these capabilities within a single generative framework. GAIA-2 supports controllable video generation conditioned on a rich set of structured inputs: ego-vehicle dynamics, agent configurations, environmental factors, and road semantics. It generates high-resolution, spatiotemporally consistent multi-camera videos across geographically diverse driving environments (UK, US, Germany). The model integrates both structured conditioning and external latent embeddings (e.g., from a proprietary driving model) to facilitate flexible and semantically grounded scene synthesis. Through this integration, GAIA-2 enables scalable simulation of both common and rare driving scenarios, advancing the use of generative world models as a core tool in the development of autonomous systems. Videos are available at this https URL. 

---
# A multi-agentic framework for real-time, autonomous freeform metasurface design 

**Authors**: Robert Lupoiu, Yixuan Shao, Tianxiang Dai, Chenkai Mao, Kofi Edee, Jonathan A. Fan  

**Link**: [PDF](https://arxiv.org/pdf/2503.20479)  

**Abstract**: Innovation in nanophotonics currently relies on human experts who synergize specialized knowledge in photonics and coding with simulation and optimization algorithms, entailing design cycles that are time-consuming, computationally demanding, and frequently suboptimal. We introduce MetaChat, a multi-agentic design framework that can translate semantically described photonic design goals into high-performance, freeform device layouts in an automated, nearly real-time manner. Multi-step reasoning is enabled by our Agentic Iterative Monologue (AIM) paradigm, which coherently interfaces agents with code-based tools, other specialized agents, and human designers. Design acceleration is facilitated by Feature-wise Linear Modulation-conditioned Maxwell surrogate solvers that support the generalized evaluation of metasurface structures. We use freeform dielectric metasurfaces as a model system and demonstrate with MetaChat the design of multi-objective, multi-wavelength metasurfaces orders of magnitude faster than conventional methods. These concepts present a scientific computing blueprint for utilizing specialist design agents, surrogate solvers, and human interactions to drive multi-physics innovation and discovery. 

---
# MoLe-VLA: Dynamic Layer-skipping Vision Language Action Model via Mixture-of-Layers for Efficient Robot Manipulation 

**Authors**: Rongyu Zhang, Menghang Dong, Yuan Zhang, Liang Heng, Xiaowei Chi, Gaole Dai, Li Du, Dan Wang, Yuan Du, Shanghang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.20384)  

**Abstract**: Multimodal Large Language Models (MLLMs) excel in understanding complex language and visual data, enabling generalist robotic systems to interpret instructions and perform embodied tasks. Nevertheless, their real-world deployment is hindered by substantial computational and storage demands. Recent insights into the homogeneous patterns in the LLM layer have inspired sparsification techniques to address these challenges, such as early exit and token pruning. However, these methods often neglect the critical role of the final layers that encode the semantic information most relevant to downstream robotic tasks. Aligning with the recent breakthrough of the Shallow Brain Hypothesis (SBH) in neuroscience and the mixture of experts in model sparsification, we conceptualize each LLM layer as an expert and propose a Mixture-of-Layers Vision-Language-Action model (MoLe-VLA, or simply MoLe) architecture for dynamic LLM layer activation. We introduce a Spatial-Temporal Aware Router (STAR) for MoLe to selectively activate only parts of the layers based on the robot's current state, mimicking the brain's distinct signal pathways specialized for cognition and causal reasoning. Additionally, to compensate for the cognitive ability of LLMs lost in MoLe, we devise a Cognition Self-Knowledge Distillation (CogKD) framework. CogKD enhances the understanding of task demands and improves the generation of task-relevant action sequences by leveraging cognitive features. Extensive experiments conducted in both RLBench simulation and real-world environments demonstrate the superiority of MoLe-VLA in both efficiency and performance. Specifically, MoLe-VLA achieves an 8% improvement in the mean success rate across ten tasks while reducing computational costs by up to x5.6 compared to standard LLMs. 

---
# LGR: LLM-Guided Ranking of Frontiers for Object Goal Navigation 

**Authors**: Mitsuaki Uno, Kanji Tanaka, Daiki Iwata, Yudai Noda, Shoya Miyazaki, Kouki Terashima  

**Link**: [PDF](https://arxiv.org/pdf/2503.20241)  

**Abstract**: Object Goal Navigation (OGN) is a fundamental task for robots and AI, with key applications such as mobile robot image databases (MRID). In particular, mapless OGN is essential in scenarios involving unknown or dynamic environments. This study aims to enhance recent modular mapless OGN systems by leveraging the commonsense reasoning capabilities of large language models (LLMs). Specifically, we address the challenge of determining the visiting order in frontier-based exploration by framing it as a frontier ranking problem. Our approach is grounded in recent findings that, while LLMs cannot determine the absolute value of a frontier, they excel at evaluating the relative value between multiple frontiers viewed within a single image using the view image as context. We dynamically manage the frontier list by adding and removing elements, using an LLM as a ranking model. The ranking results are represented as reciprocal rank vectors, which are ideal for multi-view, multi-query information fusion. We validate the effectiveness of our method through evaluations in Habitat-Sim. 

---
# Look Before Leap: Look-Ahead Planning with Uncertainty in Reinforcement Learning 

**Authors**: Yongshuai Liu, Xin Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.20139)  

**Abstract**: Model-based reinforcement learning (MBRL) has demonstrated superior sample efficiency compared to model-free reinforcement learning (MFRL). However, the presence of inaccurate models can introduce biases during policy learning, resulting in misleading trajectories. The challenge lies in obtaining accurate models due to limited diverse training data, particularly in regions with limited visits (uncertain regions). Existing approaches passively quantify uncertainty after sample generation, failing to actively collect uncertain samples that could enhance state coverage and improve model accuracy. Moreover, MBRL often faces difficulties in making accurate multi-step predictions, thereby impacting overall performance. To address these limitations, we propose a novel framework for uncertainty-aware policy optimization with model-based exploratory planning. In the model-based planning phase, we introduce an uncertainty-aware k-step lookahead planning approach to guide action selection at each step. This process involves a trade-off analysis between model uncertainty and value function approximation error, effectively enhancing policy performance. In the policy optimization phase, we leverage an uncertainty-driven exploratory policy to actively collect diverse training samples, resulting in improved model accuracy and overall performance of the RL agent. Our approach offers flexibility and applicability to tasks with varying state/action spaces and reward structures. We validate its effectiveness through experiments on challenging robotic manipulation tasks and Atari games, surpassing state-of-the-art methods with fewer interactions, thereby leading to significant performance improvements. 

---
# BugCraft: End-to-End Crash Bug Reproduction Using LLM Agents in Minecraft 

**Authors**: Eray Yapağcı, Yavuz Alp Sencer Öztürk, Eray Tüzün  

**Link**: [PDF](https://arxiv.org/pdf/2503.20036)  

**Abstract**: Reproducing game bugs, in our case crash bugs in continuously evolving games like Minecraft, is a notoriously manual, time-consuming, and challenging process to automate. Despite the success of LLM-driven bug reproduction in other software domains, games, with their complex interactive environments, remain largely unaddressed. This paper introduces BugCraft, a novel end-to-end framework designed to automate the reproduction of crash bugs in Minecraft directly from user-submitted bug reports, addressing the critical gap in automated game bug reproduction. BugCraft employs a two-stage approach: first, a Step Synthesizer leverages LLMs and Minecraft Wiki knowledge to transform bug reports into high-quality, structured steps to reproduce (S2R). Second, an Action Model, powered by a vision-based LLM agent (GPT-4o) and a custom macro API, executes these S2R steps within Minecraft to trigger the reported crash. To facilitate evaluation, we introduce BugCraft-Bench, a curated dataset of Minecraft crash bug reports. Evaluated on BugCraft-Bench, our framework successfully reproduced 30.23% of crash bugs end-to-end. The Step Synthesizer demonstrated a 66.28% accuracy in generating correct bug reproduction plans, highlighting its effectiveness in interpreting and structuring bug report information. BugCraft demonstrates the feasibility of automated reproduction of crash bugs in complex game environments using LLMs, opening promising avenues for game testing and development. The framework and the BugCraft-Bench dataset pave the way for future research in automated game bug analysis and hold potential for generalization to other interactive game platforms. Finally, we make our code open at this https URL 

---
