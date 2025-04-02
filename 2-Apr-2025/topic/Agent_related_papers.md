# On the Robustness of Agentic Function Calling 

**Authors**: Ella Rabinovich, Ateret Anaby-Tavor  

**Link**: [PDF](https://arxiv.org/pdf/2504.00914)  

**Abstract**: Large Language Models (LLMs) are increasingly acting as autonomous agents, with function calling (FC) capabilities enabling them to invoke specific tools for tasks. While prior research has primarily focused on improving FC accuracy, little attention has been given to the robustness of these agents to perturbations in their input. We introduce a benchmark assessing FC robustness in two key areas: resilience to naturalistic query variations, and stability in function calling when the toolkit expands with semantically related tools. Evaluating best-performing FC models on a carefully expanded subset of the Berkeley function calling leaderboard (BFCL), we identify critical weaknesses in existing evaluation methodologies, and highlight areas for improvement in real-world agentic deployments. 

---
# Command A: An Enterprise-Ready Large Language Model 

**Authors**: Team Cohere, Aakanksha, Arash Ahmadian, Marwan Ahmed, Jay Alammar, Yazeed Alnumay, Sophia Althammer, Arkady Arkhangorodsky, Viraat Aryabumi, Dennis Aumiller, Raphaël Avalos, Zahara Aviv, Sammie Bae, Saurabh Baji, Alexandre Barbet, Max Bartolo, Björn Bebensee, Neeral Beladia, Walter Beller-Morales, Alexandre Bérard, Andrew Berneshawi, Anna Bialas, Phil Blunsom, Matt Bobkin, Adi Bongale, Sam Braun, Maxime Brunet, Samuel Cahyawijaya, David Cairuz, Jon Ander Campos, Cassie Cao, Kris Cao, Roman Castagné, Julián Cendrero, Leila Chan Currie, Yash Chandak, Diane Chang, Giannis Chatziveroglou, Hongyu Chen, Claire Cheng, Alexis Chevalier, Justin T. Chiu, Eugene Cho, Eugene Choi, Eujeong Choi, Tim Chung, Volkan Cirik, Ana Cismaru, Pierre Clavier, Henry Conklin, Lucas Crawhall-Stein, Devon Crouse, Andres Felipe Cruz-Salinas, Ben Cyrus, Daniel D'souza, Hugo Dalla-Torre, John Dang, William Darling, Omar Darwiche Domingues, Saurabh Dash, Antoine Debugne, Théo Dehaze, Shaan Desai, Joan Devassy, Rishit Dholakia, Kyle Duffy, Ali Edalati, Ace Eldeib, Abdullah Elkady, Sarah Elsharkawy, Irem Ergün, Beyza Ermis, Marzieh Fadaee, Boyu Fan, Lucas Fayoux, Yannis Flet-Berliac, Nick Frosst, Matthias Gallé, Wojciech Galuba, Utsav Garg, Matthieu Geist, Mohammad Gheshlaghi Azar, Seraphina Goldfarb-Tarrant, Tomas Goldsack, Aidan Gomez, Victor Machado Gonzaga, Nithya Govindarajan, Manoj Govindassamy, Nathan Grinsztajn, Nikolas Gritsch, Patrick Gu, Shangmin Guo, Kilian Haefeli, Rod Hajjar, Tim Hawes, Jingyi He, Sebastian Hofstätter, Sungjin Hong, Sara Hooker, Tom Hosking  

**Link**: [PDF](https://arxiv.org/pdf/2504.00698)  

**Abstract**: In this report we describe the development of Command A, a powerful large language model purpose-built to excel at real-world enterprise use cases. Command A is an agent-optimised and multilingual-capable model, with support for 23 languages of global business, and a novel hybrid architecture balancing efficiency with top of the range performance. It offers best-in-class Retrieval Augmented Generation (RAG) capabilities with grounding and tool use to automate sophisticated business processes. These abilities are achieved through a decentralised training approach, including self-refinement algorithms and model merging techniques. We also include results for Command R7B which shares capability and architectural similarities to Command A. Weights for both models have been released for research purposes. This technical report details our original training pipeline and presents an extensive evaluation of our models across a suite of enterprise-relevant tasks and public benchmarks, demonstrating excellent performance and efficiency. 

---
# When Persuasion Overrides Truth in Multi-Agent LLM Debates: Introducing a Confidence-Weighted Persuasion Override Rate (CW-POR) 

**Authors**: Mahak Agarwal, Divyam Khanna  

**Link**: [PDF](https://arxiv.org/pdf/2504.00374)  

**Abstract**: In many real-world scenarios, a single Large Language Model (LLM) may encounter contradictory claims-some accurate, others forcefully incorrect-and must judge which is true. We investigate this risk in a single-turn, multi-agent debate framework: one LLM-based agent provides a factual answer from TruthfulQA, another vigorously defends a falsehood, and the same LLM architecture serves as judge. We introduce the Confidence-Weighted Persuasion Override Rate (CW-POR), which captures not only how often the judge is deceived but also how strongly it believes the incorrect choice. Our experiments on five open-source LLMs (3B-14B parameters), where we systematically vary agent verbosity (30-300 words), reveal that even smaller models can craft persuasive arguments that override truthful answers-often with high confidence. These findings underscore the importance of robust calibration and adversarial testing to prevent LLMs from confidently endorsing misinformation. 

---
# VerifiAgent: a Unified Verification Agent in Language Model Reasoning 

**Authors**: Jiuzhou Han, Wray Buntine, Ehsan Shareghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00406)  

**Abstract**: Large language models demonstrate remarkable reasoning capabilities but often produce unreliable or incorrect responses. Existing verification methods are typically model-specific or domain-restricted, requiring significant computational resources and lacking scalability across diverse reasoning tasks. To address these limitations, we propose VerifiAgent, a unified verification agent that integrates two levels of verification: meta-verification, which assesses completeness and consistency in model responses, and tool-based adaptive verification, where VerifiAgent autonomously selects appropriate verification tools based on the reasoning type, including mathematical, logical, or commonsense reasoning. This adaptive approach ensures both efficiency and robustness across different verification scenarios. Experimental results show that VerifiAgent outperforms baseline verification methods (e.g., deductive verifier, backward verifier) among all reasoning tasks. Additionally, it can further enhance reasoning accuracy by leveraging feedback from verification results. VerifiAgent can also be effectively applied to inference scaling, achieving better results with fewer generated samples and costs compared to existing process reward models in the mathematical reasoning domain. Code is available at this https URL 

---
# Do Large Language Models Exhibit Spontaneous Rational Deception? 

**Authors**: Samuel M. Taylor, Benjamin K. Bergen  

**Link**: [PDF](https://arxiv.org/pdf/2504.00285)  

**Abstract**: Large Language Models (LLMs) are effective at deceiving, when prompted to do so. But under what conditions do they deceive spontaneously? Models that demonstrate better performance on reasoning tasks are also better at prompted deception. Do they also increasingly deceive spontaneously in situations where it could be considered rational to do so? This study evaluates spontaneous deception produced by LLMs in a preregistered experimental protocol using tools from signaling theory. A range of proprietary closed-source and open-source LLMs are evaluated using modified 2x2 games (in the style of Prisoner's Dilemma) augmented with a phase in which they can freely communicate to the other agent using unconstrained language. This setup creates an opportunity to deceive, in conditions that vary in how useful deception might be to an agent's rational self-interest. The results indicate that 1) all tested LLMs spontaneously misrepresent their actions in at least some conditions, 2) they are generally more likely to do so in situations in which deception would benefit them, and 3) models exhibiting better reasoning capacity overall tend to deceive at higher rates. Taken together, these results suggest a tradeoff between LLM reasoning capability and honesty. They also provide evidence of reasoning-like behavior in LLMs from a novel experimental configuration. Finally, they reveal certain contextual factors that affect whether LLMs will deceive or not. We discuss consequences for autonomous, human-facing systems driven by LLMs both now and as their reasoning capabilities continue to improve. 

---
# SciReplicate-Bench: Benchmarking LLMs in Agent-driven Algorithmic Reproduction from Research Papers 

**Authors**: Yanzheng Xiang, Hanqi Yan, Shuyin Ouyang, Lin Gui, Yulan He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00255)  

**Abstract**: This study evaluates large language models (LLMs) in generating code from algorithm descriptions from recent NLP papers. The task requires two key competencies: (1) algorithm comprehension: synthesizing information from papers and academic literature to understand implementation logic, and (2) coding expertise: identifying dependencies and correctly implementing necessary APIs. To facilitate rigorous evaluation, we introduce SciReplicate-Bench, a benchmark of 100 tasks from 36 NLP papers published in 2024, featuring detailed annotations and comprehensive test cases. Building on SciReplicate-Bench, we propose Sci-Reproducer, a multi-agent framework consisting of a Paper Agent that interprets algorithmic concepts from literature and a Code Agent that retrieves dependencies from repositories and implement solutions. To assess algorithm understanding, we introduce reasoning graph accuracy, which quantifies similarity between generated and reference reasoning graphs derived from code comments and structure. For evaluating implementation quality, we employ execution accuracy, CodeBLEU, and repository dependency/API recall metrics. In our experiments, we evaluate various powerful Non-Reasoning LLMs and Reasoning LLMs as foundational models. The best-performing LLM using Sci-Reproducer achieves only 39% execution accuracy, highlighting the benchmark's this http URL analysis identifies missing or inconsistent algorithm descriptions as key barriers to successful reproduction. We will open-source our benchmark, and code at this https URL. 

---
# AgentNet: Decentralized Evolutionary Coordination for LLM-based Multi-Agent Systems 

**Authors**: Yingxuan Yang, Huacan Chai, Shuai Shao, Yuanyi Song, Siyuan Qi, Renting Rui, Weinan Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00587)  

**Abstract**: The rapid advancement of Large Language Models (LLMs) has catalyzed the development of multi-agent systems, where multiple LLM-based agents collaborate to solve complex tasks. However, existing systems predominantly rely on centralized coordination, which introduces scalability bottlenecks, limits adaptability, and creates single points of failure. Additionally, concerns over privacy and proprietary knowledge sharing hinder cross-organizational collaboration, leading to siloed expertise. To address these challenges, we propose AgentNet, a decentralized, Retrieval-Augmented Generation (RAG)-based framework that enables LLM-based agents to autonomously evolve their capabilities and collaborate efficiently in a Directed Acyclic Graph (DAG)-structured network. Unlike traditional multi-agent systems that depend on static role assignments or centralized control, AgentNet allows agents to specialize dynamically, adjust their connectivity, and route tasks without relying on predefined workflows. AgentNet's core design is built upon several key innovations: (1) Fully Decentralized Paradigm: Removing the central orchestrator, allowing agents to coordinate and specialize autonomously, fostering fault tolerance and emergent collective intelligence. (2) Dynamically Evolving Graph Topology: Real-time adaptation of agent connections based on task demands, ensuring scalability and resilience.(3) Adaptive Learning for Expertise Refinement: A retrieval-based memory system that enables agents to continuously update and refine their specialized skills. By eliminating centralized control, AgentNet enhances fault tolerance, promotes scalable specialization, and enables privacy-preserving collaboration across organizations. Through decentralized coordination and minimal data exchange, agents can leverage diverse knowledge sources while safeguarding sensitive information. 

---
# Medical Reasoning in LLMs: An In-Depth Analysis of DeepSeek R1 

**Authors**: Birger Moell, Fredrik Sand Aronsson, Sanian Akbar  

**Link**: [PDF](https://arxiv.org/pdf/2504.00016)  

**Abstract**: Integrating large language models (LLMs) like DeepSeek R1 into healthcare requires rigorous evaluation of their reasoning alignment with clinical expertise. This study assesses DeepSeek R1's medical reasoning against expert patterns using 100 MedQA clinical cases. The model achieved 93% diagnostic accuracy, demonstrating systematic clinical judgment through differential diagnosis, guideline-based treatment selection, and integration of patient-specific factors. However, error analysis of seven incorrect cases revealed persistent limitations: anchoring bias, challenges reconciling conflicting data, insufficient exploration of alternatives, overthinking, knowledge gaps, and premature prioritization of definitive treatment over intermediate care. Crucially, reasoning length correlated with accuracy - shorter responses (<5,000 characters) were more reliable, suggesting extended explanations may signal uncertainty or rationalization of errors. While DeepSeek R1 exhibits foundational clinical reasoning capabilities, recurring flaws highlight critical areas for refinement, including bias mitigation, knowledge updates, and structured reasoning frameworks. These findings underscore LLMs' potential to augment medical decision-making through artificial reasoning but emphasize the need for domain-specific validation, interpretability safeguards, and confidence metrics (e.g., response length thresholds) to ensure reliability in real-world applications. 

---
# Agent S2: A Compositional Generalist-Specialist Framework for Computer Use Agents 

**Authors**: Saaket Agashe, Kyle Wong, Vincent Tu, Jiachen Yang, Ang Li, Xin Eric Wang  

**Link**: [PDF](https://arxiv.org/pdf/2504.00906)  

**Abstract**: Computer use agents automate digital tasks by directly interacting with graphical user interfaces (GUIs) on computers and mobile devices, offering significant potential to enhance human productivity by completing an open-ended space of user queries. However, current agents face significant challenges: imprecise grounding of GUI elements, difficulties with long-horizon task planning, and performance bottlenecks from relying on single generalist models for diverse cognitive tasks. To this end, we introduce Agent S2, a novel compositional framework that delegates cognitive responsibilities across various generalist and specialist models. We propose a novel Mixture-of-Grounding technique to achieve precise GUI localization and introduce Proactive Hierarchical Planning, dynamically refining action plans at multiple temporal scales in response to evolving observations. Evaluations demonstrate that Agent S2 establishes new state-of-the-art (SOTA) performance on three prominent computer use benchmarks. Specifically, Agent S2 achieves 18.9% and 32.7% relative improvements over leading baseline agents such as Claude Computer Use and UI-TARS on the OSWorld 15-step and 50-step evaluation. Moreover, Agent S2 generalizes effectively to other operating systems and applications, surpassing previous best methods by 52.8% on WindowsAgentArena and by 16.52% on AndroidWorld relatively. Code available at this https URL. 

---
# Grounding Multimodal LLMs to Embodied Agents that Ask for Help with Reinforcement Learning 

**Authors**: Ram Ramrakhya, Matthew Chang, Xavier Puig, Ruta Desai, Zsolt Kira, Roozbeh Mottaghi  

**Link**: [PDF](https://arxiv.org/pdf/2504.00907)  

**Abstract**: Embodied agents operating in real-world environments must interpret ambiguous and under-specified human instructions. A capable household robot should recognize ambiguity and ask relevant clarification questions to infer the user intent accurately, leading to more effective task execution. To study this problem, we introduce the Ask-to-Act task, where an embodied agent must fetch a specific object instance given an ambiguous instruction in a home environment. The agent must strategically ask minimal, yet relevant, clarification questions to resolve ambiguity while navigating under partial observability. To solve this problem, we propose a novel approach that fine-tunes multimodal large language models (MLLMs) as vision-language-action (VLA) policies using online reinforcement learning (RL) with LLM-generated rewards. Our method eliminates the need for large-scale human demonstrations or manually engineered rewards for training such agents. We benchmark against strong zero-shot baselines, including GPT-4o, and supervised fine-tuned MLLMs, on our task. Our results demonstrate that our RL-finetuned MLLM outperforms all baselines by a significant margin ($19.1$-$40.3\%$), generalizing well to novel scenes and tasks. To the best of our knowledge, this is the first demonstration of adapting MLLMs as VLA agents that can act and ask for help using LLM-generated rewards with online RL. 

---
# Personality-Driven Decision-Making in LLM-Based Autonomous Agents 

**Authors**: Lewis Newsham, Daniel Prince  

**Link**: [PDF](https://arxiv.org/pdf/2504.00727)  

**Abstract**: The embedding of Large Language Models (LLMs) into autonomous agents is a rapidly developing field which enables dynamic, configurable behaviours without the need for extensive domain-specific training. In our previous work, we introduced SANDMAN, a Deceptive Agent architecture leveraging the Five-Factor OCEAN personality model, demonstrating that personality induction significantly influences agent task planning. Building on these findings, this study presents a novel method for measuring and evaluating how induced personality traits affect task selection processes - specifically planning, scheduling, and decision-making - in LLM-based agents. Our results reveal distinct task-selection patterns aligned with induced OCEAN attributes, underscoring the feasibility of designing highly plausible Deceptive Agents for proactive cyber defense strategies. 

---
# Accelerating drug discovery with Artificial: a whole-lab orchestration and scheduling system for self-driving labs 

**Authors**: Yao Fehlis, Paul Mandel, Charles Crain, Betty Liu, David Fuller  

**Link**: [PDF](https://arxiv.org/pdf/2504.00986)  

**Abstract**: Self-driving labs are transforming drug discovery by enabling automated, AI-guided experimentation, but they face challenges in orchestrating complex workflows, integrating diverse instruments and AI models, and managing data efficiently. Artificial addresses these issues with a comprehensive orchestration and scheduling system that unifies lab operations, automates workflows, and integrates AI-driven decision-making. By incorporating AI/ML models like NVIDIA BioNeMo - which facilitates molecular interaction prediction and biomolecular analysis - Artificial enhances drug discovery and accelerates data-driven research. Through real-time coordination of instruments, robots, and personnel, the platform streamlines experiments, enhances reproducibility, and advances drug discovery. 

---
# JudgeLRM: Large Reasoning Models as a Judge 

**Authors**: Nuo Chen, Zhiyuan Hu, Qingyun Zou, Jiaying Wu, Qian Wang, Bryan Hooi, Bingsheng He  

**Link**: [PDF](https://arxiv.org/pdf/2504.00050)  

**Abstract**: The rise of Large Language Models (LLMs) as evaluators offers a scalable alternative to human annotation, yet existing Supervised Fine-Tuning (SFT) for judges approaches often fall short in domains requiring complex reasoning. In this work, we investigate whether LLM judges truly benefit from enhanced reasoning capabilities. Through a detailed analysis of reasoning requirements across evaluation tasks, we reveal a negative correlation between SFT performance gains and the proportion of reasoning-demanding samples - highlighting the limitations of SFT in such scenarios. To address this, we introduce JudgeLRM, a family of judgment-oriented LLMs trained using reinforcement learning (RL) with judge-wise, outcome-driven rewards. JudgeLRM models consistently outperform both SFT-tuned and state-of-the-art reasoning models. Notably, JudgeLRM-3B surpasses GPT-4, and JudgeLRM-7B outperforms DeepSeek-R1 by 2.79% in F1 score, particularly excelling in judge tasks requiring deep reasoning. 

---
