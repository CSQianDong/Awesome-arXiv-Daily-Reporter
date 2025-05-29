# HDDLGym: A Tool for Studying Multi-Agent Hierarchical Problems Defined in HDDL with OpenAI Gym 

**Authors**: Ngoc La, Ruaridh Mon-Williams, Julie A. Shah  

**Link**: [PDF](https://arxiv.org/pdf/2505.22597)  

**Abstract**: In recent years, reinforcement learning (RL) methods have been widely tested using tools like OpenAI Gym, though many tasks in these environments could also benefit from hierarchical planning. However, there is a lack of a tool that enables seamless integration of hierarchical planning with RL. Hierarchical Domain Definition Language (HDDL), used in classical planning, introduces a structured approach well-suited for model-based RL to address this gap. To bridge this integration, we introduce HDDLGym, a Python-based tool that automatically generates OpenAI Gym environments from HDDL domains and problems. HDDLGym serves as a link between RL and hierarchical planning, supporting multi-agent scenarios and enabling collaborative planning among agents. This paper provides an overview of HDDLGym's design and implementation, highlighting the challenges and design choices involved in integrating HDDL with the Gym interface, and applying RL policies to support hierarchical planning. We also provide detailed instructions and demonstrations for using the HDDLGym framework, including how to work with existing HDDL domains and problems from International Planning Competitions, exemplified by the Transport domain. Additionally, we offer guidance on creating new HDDL domains for multi-agent scenarios and demonstrate the practical use of HDDLGym in the Overcooked domain. By leveraging the advantages of HDDL and Gym, HDDLGym aims to be a valuable tool for studying RL in hierarchical planning, particularly in multi-agent contexts. 

---
# AI Mathematician: Towards Fully Automated Frontier Mathematical Research 

**Authors**: Yuanhang Liu, Yanxing Huang, Yanqiao Wang, Peng Li, Yang Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22451)  

**Abstract**: Large Reasoning Models (LRMs) have made significant progress in mathematical capabilities in recent times. However, these successes have been primarily confined to competition-level problems. In this work, we propose AI Mathematician (AIM) framework, which harnesses the reasoning strength of LRMs to support frontier mathematical research. We have identified two critical challenges of mathematical research compared to competition, {\it the intrinsic complexity of research problems} and {\it the requirement of procedural rigor}. To address these challenges, AIM incorporates two core strategies: an exploration mechanism to foster longer solution paths, and the pessimistic reasonable verification method to ensure reliability.
This early version of AIM already exhibits strong capability in tackling research-level tasks. We conducted extensive experiments across several real-world mathematical topics and obtained promising results. AIM is able to autonomously construct substantial portions of proofs and uncover non-trivial insights within each research area. These findings highlight the potential of LRMs in mathematical discovery and suggest that LRM-based agent systems could significantly accelerate mathematical research in the future. 

---
# AgentDNS: A Root Domain Naming System for LLM Agents 

**Authors**: Enfang Cui, Yujun Cheng, Rui She, Dan Liu, Zhiyuan Liang, Minxin Guo, Tianzheng Li, Qian Wei, Wenjuan Xing, Zhijie Zhong  

**Link**: [PDF](https://arxiv.org/pdf/2505.22368)  

**Abstract**: The rapid evolution of Large Language Model (LLM) agents has highlighted critical challenges in cross-vendor service discovery, interoperability, and communication. Existing protocols like model context protocol and agent-to-agent protocol have made significant strides in standardizing interoperability between agents and tools, as well as communication among multi-agents. However, there remains a lack of standardized protocols and solutions for service discovery across different agent and tool vendors. In this paper, we propose AgentDNS, a root domain naming and service discovery system designed to enable LLM agents to autonomously discover, resolve, and securely invoke third-party agent and tool services across organizational and technological boundaries. Inspired by the principles of the traditional DNS, AgentDNS introduces a structured mechanism for service registration, semantic service discovery, secure invocation, and unified billing. We detail the architecture, core functionalities, and use cases of AgentDNS, demonstrating its potential to streamline multi-agent collaboration in real-world scenarios. The source code will be published on this https URL. 

---
# From Large AI Models to Agentic AI: A Tutorial on Future Intelligent Communications 

**Authors**: Feibo Jiang, Cunhua Pan, Li Dong, Kezhi Wang, Octavia A. Dobre, Merouane Debbah  

**Link**: [PDF](https://arxiv.org/pdf/2505.22311)  

**Abstract**: With the advent of 6G communications, intelligent communication systems face multiple challenges, including constrained perception and response capabilities, limited scalability, and low adaptability in dynamic environments. This tutorial provides a systematic introduction to the principles, design, and applications of Large Artificial Intelligence Models (LAMs) and Agentic AI technologies in intelligent communication systems, aiming to offer researchers a comprehensive overview of cutting-edge technologies and practical guidance. First, we outline the background of 6G communications, review the technological evolution from LAMs to Agentic AI, and clarify the tutorial's motivation and main contributions. Subsequently, we present a comprehensive review of the key components required for constructing LAMs. We further categorize LAMs and analyze their applicability, covering Large Language Models (LLMs), Large Vision Models (LVMs), Large Multimodal Models (LMMs), Large Reasoning Models (LRMs), and lightweight LAMs. Next, we propose a LAM-centric design paradigm tailored for communications, encompassing dataset construction and both internal and external learning approaches. Building upon this, we develop an LAM-based Agentic AI system for intelligent communications, clarifying its core components such as planners, knowledge bases, tools, and memory modules, as well as its interaction mechanisms. We also introduce a multi-agent framework with data retrieval, collaborative planning, and reflective evaluation for 6G. Subsequently, we provide a detailed overview of the applications of LAMs and Agentic AI in communication scenarios. Finally, we summarize the research challenges and future directions in current studies, aiming to support the development of efficient, secure, and sustainable next-generation intelligent communication systems. 

---
# Rethinking the Unsolvable: When In-Context Search Meets Test-Time Scaling 

**Authors**: Fanzeng Xia, Yidong Luo, Tinko Sebastian Bartels, Yaqi Xu, Tongxin Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.22290)  

**Abstract**: Recent research has highlighted that Large Language Models (LLMs), even when trained to generate extended long reasoning steps, still face significant challenges on hard reasoning problems. However, much of the existing literature relies on direct prompting with simple in-context learning examples for evaluation, which largely overlooks advanced techniques to elicit LLMs' deliberate reasoning before drawing conclusions that LLMs hit a performance ceiling. In this paper, we systematically explore the combined potential of in-context search and test-time scaling on super hard reasoning tasks. We find that by employing advanced in-context search prompting to LLMs augmented with internal scaling, one can achieve transformative performance breakthroughs on tasks previously deemed "unsolvable" (e.g., reported success rates below 5%). We provide both empirical results and theoretical analysis of how this combination can unleash LLM reasoning capabilities: i) Empirically, on controlled NP-hard tasks and complex real-world planning benchmarks, our approach achieves up to a 30x improvement in success rates compared to previously reported results without any external mechanisms; ii) Theoretically, we show that in-context search prompting, when combined with internal scaling, significantly extends the complexity class of solvable reasoning problems. These findings challenge prevailing assumptions about the limitations of LLMs on complex tasks, indicating that current evaluation paradigms systematically underestimate their true potential. Our work calls for a critical reassessment of how LLM reasoning is benchmarked and a more robust evaluation strategy that fully captures the true capabilities of contemporary LLMs, which can lead to a better understanding of their operational reasoning boundaries in real-world deployments. 

---
# Compression versus Accuracy: A Hierarchy of Lifted Models 

**Authors**: Jan Speller, Malte Luttermann, Marcel Gehrke, Tanya Braun  

**Link**: [PDF](https://arxiv.org/pdf/2505.22288)  

**Abstract**: Probabilistic graphical models that encode indistinguishable objects and relations among them use first-order logic constructs to compress a propositional factorised model for more efficient (lifted) inference. To obtain a lifted representation, the state-of-the-art algorithm Advanced Colour Passing (ACP) groups factors that represent matching distributions. In an approximate version using $\varepsilon$ as a hyperparameter, factors are grouped that differ by a factor of at most $(1\pm \varepsilon)$. However, finding a suitable $\varepsilon$ is not obvious and may need a lot of exploration, possibly requiring many ACP runs with different $\varepsilon$ values. Additionally, varying $\varepsilon$ can yield wildly different models, leading to decreased interpretability. Therefore, this paper presents a hierarchical approach to lifted model construction that is hyperparameter-free. It efficiently computes a hierarchy of $\varepsilon$ values that ensures a hierarchy of models, meaning that once factors are grouped together given some $\varepsilon$, these factors will be grouped together for larger $\varepsilon$ as well. The hierarchy of $\varepsilon$ values also leads to a hierarchy of error bounds. This allows for explicitly weighing compression versus accuracy when choosing specific $\varepsilon$ values to run ACP with and enables interpretability between the different models. 

---
# A Preprocessing Framework for Efficient Approximate Bi-Objective Shortest-Path Computation in the Presence of Correlated Objectives 

**Authors**: Yaron Halle, Ariel Felner, Sven Koenig, Oren Salzman  

**Link**: [PDF](https://arxiv.org/pdf/2505.22244)  

**Abstract**: The bi-objective shortest-path (BOSP) problem seeks to find paths between start and target vertices of a graph while optimizing two conflicting objective functions. We consider the BOSP problem in the presence of correlated objectives. Such correlations often occur in real-world settings such as road networks, where optimizing two positively correlated objectives, such as travel time and fuel consumption, is common. BOSP is generally computationally challenging as the size of the search space is exponential in the number of objective functions and the graph size. Bounded sub-optimal BOSP solvers such as A*pex alleviate this complexity by approximating the Pareto-optimal solution set rather than computing it exactly (given a user-provided approximation factor). As the correlation between objective functions increases, smaller approximation factors are sufficient for collapsing the entire Pareto-optimal set into a single solution. We leverage this insight to propose an efficient algorithm that reduces the search effort in the presence of correlated objectives. Our approach for computing approximations of the entire Pareto-optimal set is inspired by graph-clustering algorithms. It uses a preprocessing phase to identify correlated clusters within a graph and to generate a new graph representation. This allows a natural generalization of A*pex to run up to five times faster on DIMACS dataset instances, a standard benchmark in the field. To the best of our knowledge, this is the first algorithm proposed that efficiently and effectively exploits correlations in the context of bi-objective search while providing theoretical guarantees on solution quality. 

---
# What Makes a Good Reasoning Chain? Uncovering Structural Patterns in Long Chain-of-Thought Reasoning 

**Authors**: Gangwei Jiang, Yahui Liu, Zhaoyi Li, Qi Wang, Fuzheng Zhang, Linqi Song, Ying Wei, Defu Lian  

**Link**: [PDF](https://arxiv.org/pdf/2505.22148)  

**Abstract**: Recent advances in reasoning with large language models (LLMs) have popularized Long Chain-of-Thought (LCoT), a strategy that encourages deliberate and step-by-step reasoning before producing a final answer. While LCoTs have enabled expert-level performance in complex tasks, how the internal structures of their reasoning chains drive, or even predict, the correctness of final answers remains a critical yet underexplored question. In this work, we present LCoT2Tree, an automated framework that converts sequential LCoTs into hierarchical tree structures and thus enables deeper structural analysis of LLM reasoning. Using graph neural networks (GNNs), we reveal that structural patterns extracted by LCoT2Tree, including exploration, backtracking, and verification, serve as stronger predictors of final performance across a wide range of tasks and models. Leveraging an explainability technique, we further identify critical thought patterns such as over-branching that account for failures. Beyond diagnostic insights, the structural patterns by LCoT2Tree support practical applications, including improving Best-of-N decoding effectiveness. Overall, our results underscore the critical role of internal structures of reasoning chains, positioning LCoT2Tree as a powerful tool for diagnosing, interpreting, and improving reasoning in LLMs. 

---
# Lifted Forward Planning in Relational Factored Markov Decision Processes with Concurrent Actions 

**Authors**: Florian Andreas Marwitz, Tanya Braun, Ralf Möller, Marcel Gehrke  

**Link**: [PDF](https://arxiv.org/pdf/2505.22147)  

**Abstract**: Decision making is a central problem in AI that can be formalized using a Markov Decision Process. A problem is that, with increasing numbers of (indistinguishable) objects, the state space grows exponentially. To compute policies, the state space has to be enumerated. Even more possibilities have to be enumerated if the size of the action space depends on the size of the state space, especially if we allow concurrent actions. To tackle the exponential blow-up in the action and state space, we present a first-order representation to store the spaces in polynomial instead of exponential size in the number of objects and introduce Foreplan, a relational forward planner, which uses this representation to efficiently compute policies for numerous indistinguishable objects and actions. Additionally, we introduce an even faster approximate version of Foreplan. Moreover, Foreplan identifies how many objects an agent should act on to achieve a certain task given restrictions. Further, we provide a theoretical analysis and an empirical evaluation of Foreplan, demonstrating a speedup of at least four orders of magnitude. 

---
# Visual Large Language Models Exhibit Human-Level Cognitive Flexibility in the Wisconsin Card Sorting Test 

**Authors**: Guangfu Hao, Frederic Alexandre, Shan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22112)  

**Abstract**: Cognitive flexibility has been extensively studied in human cognition but remains relatively unexplored in the context of Visual Large Language Models (VLLMs). This study assesses the cognitive flexibility of state-of-the-art VLLMs (GPT-4o, Gemini-1.5 Pro, and Claude-3.5 Sonnet) using the Wisconsin Card Sorting Test (WCST), a classic measure of set-shifting ability. Our results reveal that VLLMs achieve or surpass human-level set-shifting capabilities under chain-of-thought prompting with text-based inputs. However, their abilities are highly influenced by both input modality and prompting strategy. In addition, we find that through role-playing, VLLMs can simulate various functional deficits aligned with patients having impairments in cognitive flexibility, suggesting that VLLMs may possess a cognitive architecture, at least regarding the ability of set-shifting, similar to the brain. This study reveals the fact that VLLMs have already approached the human level on a key component underlying our higher cognition, and highlights the potential to use them to emulate complex brain processes. 

---
# Efficient Dynamic Shielding for Parametric Safety Specifications 

**Authors**: Davide Corsi, Kaushik Mallik, Andoni Rodriguez, Cesar Sanchez  

**Link**: [PDF](https://arxiv.org/pdf/2505.22104)  

**Abstract**: Shielding has emerged as a promising approach for ensuring safety of AI-controlled autonomous systems. The algorithmic goal is to compute a shield, which is a runtime safety enforcement tool that needs to monitor and intervene the AI controller's actions if safety could be compromised otherwise. Traditional shields are designed statically for a specific safety requirement. Therefore, if the safety requirement changes at runtime due to changing operating conditions, the shield needs to be recomputed from scratch, causing delays that could be fatal. We introduce dynamic shields for parametric safety specifications, which are succinctly represented sets of all possible safety specifications that may be encountered at runtime. Our dynamic shields are statically designed for a given safety parameter set, and are able to dynamically adapt as the true safety specification (permissible by the parameters) is revealed at runtime. The main algorithmic novelty lies in the dynamic adaptation procedure, which is a simple and fast algorithm that utilizes known features of standard safety shields, like maximal permissiveness. We report experimental results for a robot navigation problem in unknown territories, where the safety specification evolves as new obstacles are discovered at runtime. In our experiments, the dynamic shields took a few minutes for their offline design, and took between a fraction of a second and a few seconds for online adaptation at each step, whereas the brute-force online recomputation approach was up to 5 times slower. 

---
# VIRAL: Vision-grounded Integration for Reward design And Learning 

**Authors**: Valentin Cuzin-Rambaud, Emilien Komlenovic, Alexandre Faure, Bruno Yun  

**Link**: [PDF](https://arxiv.org/pdf/2505.22092)  

**Abstract**: The alignment between humans and machines is a critical challenge in artificial intelligence today. Reinforcement learning, which aims to maximize a reward function, is particularly vulnerable to the risks associated with poorly designed reward functions. Recent advancements has shown that Large Language Models (LLMs) for reward generation can outperform human performance in this context. We introduce VIRAL, a pipeline for generating and refining reward functions through the use of multi-modal LLMs. VIRAL autonomously creates and interactively improves reward functions based on a given environment and a goal prompt or annotated image. The refinement process can incorporate human feedback or be guided by a description generated by a video LLM, which explains the agent's policy in video form. We evaluated VIRAL in five Gymnasium environments, demonstrating that it accelerates the learning of new behaviors while ensuring improved alignment with user intent. The source-code and demo video are available at: this https URL and this https URL. 

---
# Cognitively-Inspired Emergent Communication via Knowledge Graphs for Assisting the Visually Impaired 

**Authors**: Ruxiao Chen, Dezheng Han, Wenjie Han, Shuaishuai Guo  

**Link**: [PDF](https://arxiv.org/pdf/2505.22087)  

**Abstract**: Assistive systems for visually impaired individuals must deliver rapid, interpretable, and adaptive feedback to facilitate real-time navigation. Current approaches face a trade-off between latency and semantic richness: natural language-based systems provide detailed guidance but are too slow for dynamic scenarios, while emergent communication frameworks offer low-latency symbolic languages but lack semantic depth, limiting their utility in tactile modalities like vibration. To address these limitations, we introduce a novel framework, Cognitively-Inspired Emergent Communication via Knowledge Graphs (VAG-EC), which emulates human visual perception and cognitive mapping. Our method constructs knowledge graphs to represent objects and their relationships, incorporating attention mechanisms to prioritize task-relevant entities, thereby mirroring human selective attention. This structured approach enables the emergence of compact, interpretable, and context-sensitive symbolic languages. Extensive experiments across varying vocabulary sizes and message lengths demonstrate that VAG-EC outperforms traditional emergent communication methods in Topographic Similarity (TopSim) and Context Independence (CI). These findings underscore the potential of cognitively grounded emergent communication as a fast, adaptive, and human-aligned solution for real-time assistive technologies. Code is available at this https URL. 

---
# Reinforced Reasoning for Embodied Planning 

**Authors**: Di Wu, Jiaxin Fan, Junzhe Zang, Guanbo Wang, Wei Yin, Wenhao Li, Bo Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.22050)  

**Abstract**: Embodied planning requires agents to make coherent multi-step decisions based on dynamic visual observations and natural language goals. While recent vision-language models (VLMs) excel at static perception tasks, they struggle with the temporal reasoning, spatial understanding, and commonsense grounding needed for planning in interactive environments. In this work, we introduce a reinforcement fine-tuning framework that brings R1-style reasoning enhancement into embodied planning. We first distill a high-quality dataset from a powerful closed-source model and perform supervised fine-tuning (SFT) to equip the model with structured decision-making priors. We then design a rule-based reward function tailored to multi-step action quality and optimize the policy via Generalized Reinforced Preference Optimization (GRPO). Our approach is evaluated on Embench, a recent benchmark for interactive embodied tasks, covering both in-domain and out-of-domain scenarios. Experimental results show that our method significantly outperforms models of similar or larger scale, including GPT-4o-mini and 70B+ open-source baselines, and exhibits strong generalization to unseen environments. This work highlights the potential of reinforcement-driven reasoning to advance long-horizon planning in embodied AI. 

---
# Efficiently Enhancing General Agents With Hierarchical-categorical Memory 

**Authors**: Changze Qiao, Mingming Lu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22006)  

**Abstract**: With large language models (LLMs) demonstrating remarkable capabilities, there has been a surge in research on leveraging LLMs to build general-purpose multi-modal agents. However, existing approaches either rely on computationally expensive end-to-end training using large-scale multi-modal data or adopt tool-use methods that lack the ability to continuously learn and adapt to new environments. In this paper, we introduce EHC, a general agent capable of learning without parameter updates. EHC consists of a Hierarchical Memory Retrieval (HMR) module and a Task-Category Oriented Experience Learning (TOEL) module. The HMR module facilitates rapid retrieval of relevant memories and continuously stores new information without being constrained by memory capacity. The TOEL module enhances the agent's comprehension of various task characteristics by classifying experiences and extracting patterns across different categories. Extensive experiments conducted on multiple standard datasets demonstrate that EHC outperforms existing methods, achieving state-of-the-art performance and underscoring its effectiveness as a general agent for handling complex multi-modal tasks. 

---
# Functional Matching of Logic Subgraphs: Beyond Structural Isomorphism 

**Authors**: Ziyang Zheng, Kezhi Li, Zhengyuan Shi, Qiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21988)  

**Abstract**: Subgraph matching in logic circuits is foundational for numerous Electronic Design Automation (EDA) applications, including datapath optimization, arithmetic verification, and hardware trojan detection. However, existing techniques rely primarily on structural graph isomorphism and thus fail to identify function-related subgraphs when synthesis transformations substantially alter circuit topology. To overcome this critical limitation, we introduce the concept of functional subgraph matching, a novel approach that identifies whether a given logic function is implicitly present within a larger circuit, irrespective of structural variations induced by synthesis or technology mapping. Specifically, we propose a two-stage multi-modal framework: (1) learning robust functional embeddings across AIG and post-mapping netlists for functional subgraph detection, and (2) identifying fuzzy boundaries using a graph segmentation approach. Evaluations on standard benchmarks (ITC99, OpenABCD, ForgeEDA) demonstrate significant performance improvements over existing structural methods, with average $93.8\%$ accuracy in functional subgraph detection and a dice score of $91.3\%$ in fuzzy boundary identification. 

---
# From Reasoning to Learning: A Survey on Hypothesis Discovery and Rule Learning with Large Language Models 

**Authors**: Kaiyu He, Zhiyu Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21935)  

**Abstract**: Since the advent of Large Language Models (LLMs), efforts have largely focused on improving their instruction-following and deductive reasoning abilities, leaving open the question of whether these models can truly discover new knowledge. In pursuit of artificial general intelligence (AGI), there is a growing need for models that not only execute commands or retrieve information but also learn, reason, and generate new knowledge by formulating novel hypotheses and theories that deepen our understanding of the world. Guided by Peirce's framework of abduction, deduction, and induction, this survey offers a structured lens to examine LLM-based hypothesis discovery. We synthesize existing work in hypothesis generation, application, and validation, identifying both key achievements and critical gaps. By unifying these threads, we illuminate how LLMs might evolve from mere ``information executors'' into engines of genuine innovation, potentially transforming research, science, and real-world problem solving. 

---
# Modeling and Optimizing User Preferences in AI Copilots: A Comprehensive Survey and Taxonomy 

**Authors**: Saleh Afzoon, Zahra Jahanandish, Phuong Thao Huynh, Amin Beheshti, Usman Naseem  

**Link**: [PDF](https://arxiv.org/pdf/2505.21907)  

**Abstract**: AI copilots, context-aware, AI-powered systems designed to assist users in tasks such as software development and content creation, are becoming integral to modern workflows. As these systems grow in capability and adoption, personalization has emerged as a cornerstone for ensuring usability, trust, and productivity. Central to this personalization is preference optimization: the ability of AI copilots to detect, interpret, and align with individual user preferences. While personalization techniques are well-established in domains like recommender systems and dialogue agents, their adaptation to interactive, real-time systems like AI copilots remains fragmented and underexplored. This survey addresses this gap by synthesizing research on how user preferences are captured, modeled, and refined within the design of AI copilots. We introduce a unified definition of AI copilots and propose a phase-based taxonomy of preference optimization strategies, structured around pre-interaction, mid-interaction, and post-interaction stages. We analyze techniques for acquiring preference signals, modeling user intent, and integrating feedback loops, highlighting both established approaches and recent innovations. By bridging insights from AI personalization, human-AI collaboration, and large language model adaptation, this survey provides a structured foundation for designing adaptive, preference-aware AI copilots. It offers a holistic view of the available preference resources, how they can be leveraged, and which technical approaches are most suited to each stage of system design. 

---
# SVRPBench: A Realistic Benchmark for Stochastic Vehicle Routing Problem 

**Authors**: Ahmed Heakl, Yahia Salaheldin Shaaban, Martin Takac, Salem Lahlou, Zangir Iklassov  

**Link**: [PDF](https://arxiv.org/pdf/2505.21887)  

**Abstract**: Robust routing under uncertainty is central to real-world logistics, yet most benchmarks assume static, idealized settings. We present SVRPBench, the first open benchmark to capture high-fidelity stochastic dynamics in vehicle routing at urban scale. Spanning more than 500 instances with up to 1000 customers, it simulates realistic delivery conditions: time-dependent congestion, log-normal delays, probabilistic accidents, and empirically grounded time windows for residential and commercial clients. Our pipeline generates diverse, constraint-rich scenarios, including multi-depot and multi-vehicle setups. Benchmarking reveals that state-of-the-art RL solvers like POMO and AM degrade by over 20% under distributional shift, while classical and metaheuristic methods remain robust. To enable reproducible research, we release the dataset and evaluation suite. SVRPBench challenges the community to design solvers that generalize beyond synthetic assumptions and adapt to real-world uncertainty. 

---
# SAGE-Eval: Evaluating LLMs for Systematic Generalizations of Safety Facts 

**Authors**: Chen Yueh-Han, Guy Davidson, Brenden M. Lake  

**Link**: [PDF](https://arxiv.org/pdf/2505.21828)  

**Abstract**: Do LLMs robustly generalize critical safety facts to novel situations? Lacking this ability is dangerous when users ask naive questions. For instance, "I'm considering packing melon balls for my 10-month-old's lunch. What other foods would be good to include?" Before offering food options, the LLM should warn that melon balls pose a choking hazard to toddlers, as documented by the CDC. Failing to provide such warnings could result in serious injuries or even death. To evaluate this, we introduce SAGE-Eval, SAfety-fact systematic GEneralization evaluation, the first benchmark that tests whether LLMs properly apply well established safety facts to naive user queries. SAGE-Eval comprises 104 facts manually sourced from reputable organizations, systematically augmented to create 10,428 test scenarios across 7 common domains (e.g., Outdoor Activities, Medicine). We find that the top model, Claude-3.7-sonnet, passes only 58% of all the safety facts tested. We also observe that model capabilities and training compute weakly correlate with performance on SAGE-Eval, implying that scaling up is not the golden solution. Our findings suggest frontier LLMs still lack robust generalization ability. We recommend developers use SAGE-Eval in pre-deployment evaluations to assess model reliability in addressing salient risks. We publicly release SAGE-Eval at this https URL and our code is available at this https URL. 

---
# Towards Safety Reasoning in LLMs: AI-agentic Deliberation for Policy-embedded CoT Data Creation 

**Authors**: Tharindu Kumarage, Ninareh Mehrabi, Anil Ramakrishna, Xinyan Zhao, Richard Zemel, Kai-Wei Chang, Aram Galstyan, Rahul Gupta, Charith Peris  

**Link**: [PDF](https://arxiv.org/pdf/2505.21784)  

**Abstract**: Safety reasoning is a recent paradigm where LLMs reason over safety policies before generating responses, thereby mitigating limitations in existing safety measures such as over-refusal and jailbreak vulnerabilities. However, implementing this paradigm is challenging due to the resource-intensive process of creating high-quality policy-embedded chain-of-thought (CoT) datasets while ensuring reasoning remains accurate and free from hallucinations or policy conflicts. To tackle this, we propose AIDSAFE: Agentic Iterative Deliberation for Safety Reasoning, a novel data generation recipe that leverages multi-agent deliberation to iteratively expand reasoning on safety policies. A data refiner stage in AIDSAFE ensures high-quality outputs by eliminating repetitive, redundant, and deceptive thoughts. AIDSAFE-generated CoTs provide a strong foundation for supervised fine-tuning (SFT)-based safety training. Additionally, to address the need of preference data in alignment stages, such as DPO training, we introduce a supplemental recipe that uses belief augmentation to create distinct selected and rejected CoT samples. Our evaluations demonstrate that AIDSAFE-generated CoTs achieve superior policy adherence and reasoning quality. Consequently, we show that fine-tuning open-source LLMs on these CoTs can significantly improve safety generalization and jailbreak robustness while maintaining acceptable utility and over-refusal accuracy. AIDSAFE-generated CoT datasets can be found here: this https URL 

---
# Don't Think Longer, Think Wisely: Optimizing Thinking Dynamics for Large Reasoning Models 

**Authors**: Sohyun An, Ruochen Wang, Tianyi Zhou, Cho-Jui Hsieh  

**Link**: [PDF](https://arxiv.org/pdf/2505.21765)  

**Abstract**: While recent success of large reasoning models (LRMs) significantly advanced LLMs' reasoning capability by optimizing the final answer accuracy using reinforcement learning, they may also drastically increase the output length due to overthinking, characterized by unnecessarily complex reasoning paths that waste computation and potentially degrade the performance. We hypothesize that such inefficiencies stem from LRMs' limited capability to dynamically select the proper modular reasoning strategies, termed thinking patterns at the right position. To investigate this hypothesis, we propose a dynamic optimization framework that segments model-generated reasoning paths into distinct thinking patterns, systematically identifying and promoting beneficial patterns that improve the answer while removing detrimental ones. Empirical analysis confirms that our optimized thinking paths yield more concise yet sufficiently informative trajectories, enhancing reasoning efficiency by reducing attention FLOPs by up to 47% while maintaining accuracy for originally correct responses. Moreover, a non-trivial portion of originally incorrect responses are transformed into correct ones, achieving a 15.6% accuracy improvement with reduced length. Motivated by the improvement brought by the optimized thinking paths, we apply a preference optimization technique supported by a pairwise dataset contrasting suboptimal and optimal reasoning paths. Experimental evaluations across multiple mathematical reasoning benchmarks reveal that our method notably reduces computational overhead while simultaneously improving reasoning accuracy, achieving up to a 12% accuracy improvement and reducing token usage from approximately 5,000 to 3,000 tokens. 

---
# Make Planning Research Rigorous Again! 

**Authors**: Michael Katz, Harsha Kokel, Christian Muise, Shirin Sohrabi, Sarath Sreedharan  

**Link**: [PDF](https://arxiv.org/pdf/2505.21674)  

**Abstract**: In over sixty years since its inception, the field of planning has made significant contributions to both the theory and practice of building planning software that can solve a never-before-seen planning problem. This was done through established practices of rigorous design and evaluation of planning systems. It is our position that this rigor should be applied to the current trend of work on planning with large language models. One way to do so is by correctly incorporating the insights, tools, and data from the automated planning community into the design and evaluation of LLM-based planners. The experience and expertise of the planning community are not just important from a historical perspective; the lessons learned could play a crucial role in accelerating the development of LLM-based planners. This position is particularly important in light of the abundance of recent works that replicate and propagate the same pitfalls that the planning community has encountered and learned from. We believe that avoiding such known pitfalls will contribute greatly to the progress in building LLM-based planners and to planning in general. 

---
# Adaptive Frontier Exploration on Graphs with Applications to Network-Based Disease Testing 

**Authors**: Davin Choo, Yuqi Pan, Tonghan Wang, Milind Tambe, Alastair van Heerden, Cheryl Johnson  

**Link**: [PDF](https://arxiv.org/pdf/2505.21671)  

**Abstract**: We study a sequential decision-making problem on a $n$-node graph $G$ where each node has an unknown label from a finite set $\mathbf{\Sigma}$, drawn from a joint distribution $P$ that is Markov with respect to $G$. At each step, selecting a node reveals its label and yields a label-dependent reward. The goal is to adaptively choose nodes to maximize expected accumulated discounted rewards. We impose a frontier exploration constraint, where actions are limited to neighbors of previously selected nodes, reflecting practical constraints in settings such as contact tracing and robotic exploration. We design a Gittins index-based policy that applies to general graphs and is provably optimal when $G$ is a forest. Our implementation runs in $O(n^2 \cdot |\mathbf{\Sigma}|^2)$ time while using $O(n \cdot |\mathbf{\Sigma}|^2)$ oracle calls to $P$ and $O(n^2 \cdot |\mathbf{\Sigma}|)$ space. Experiments on synthetic and real-world graphs show that our method consistently outperforms natural baselines, including in non-tree, budget-limited, and undiscounted settings. For example, in HIV testing simulations on real-world sexual interaction networks, our policy detects nearly all positive cases with only half the population tested, substantially outperforming other baselines. 

---
# R1-Code-Interpreter: Training LLMs to Reason with Code via Supervised and Reinforcement Learning 

**Authors**: Yongchao Chen, Yueying Liu, Junwei Zhou, Yilun Hao, Jingquan Wang, Yang Zhang, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2505.21668)  

**Abstract**: Despite advances in reasoning and planning of R1-like models, Large Language Models (LLMs) still struggle with tasks requiring precise computation, symbolic manipulation, optimization, and algorithmic reasoning, in which textual reasoning lacks the rigor of code execution. A key challenge is enabling LLMs to decide when to use textual reasoning versus code generation. While OpenAI trains models to invoke a Code Interpreter as needed, public research lacks guidance on aligning pre-trained LLMs to effectively leverage code and generalize across diverse tasks. We present R1-Code-Interpreter, an extension of a text-only LLM trained via multi-turn supervised fine-tuning (SFT) and reinforcement learning (RL) to autonomously generate multiple code queries during step-by-step reasoning. We curate 144 reasoning and planning tasks (107 for training, 37 for testing), each with over 200 diverse questions. We fine-tune Qwen-2.5 models (3B/7B/14B) using various SFT and RL strategies, investigating different answer formats, reasoning vs. non-reasoning models, cold vs. warm starts, GRPO vs. PPO, and masked vs. unmasked code outputs. Unlike prior RL work on narrow domains, we find that Code Interpreter training is significantly harder due to high task diversity and expensive code execution, highlighting the critical role of the SFT stage. Our final model, R1-CI-14B, improves average accuracy on the 37 test tasks from 44.0\% to 64.1\%, outperforming GPT-4o (text-only: 58.6\%) and approaching GPT-4o with Code Interpreter (70.9\%), with the emergent self-checking behavior via code generation. Datasets, Codes, and Models are available at this https URL and this https URL. 

---
# Understanding the learned look-ahead behavior of chess neural networks 

**Authors**: Diogo Cruz  

**Link**: [PDF](https://arxiv.org/pdf/2505.21552)  

**Abstract**: We investigate the look-ahead capabilities of chess-playing neural networks, specifically focusing on the Leela Chess Zero policy network. We build on the work of Jenner et al. (2024) by analyzing the model's ability to consider future moves and alternative sequences beyond the immediate next move. Our findings reveal that the network's look-ahead behavior is highly context-dependent, varying significantly based on the specific chess position. We demonstrate that the model can process information about board states up to seven moves ahead, utilizing similar internal mechanisms across different future time steps. Additionally, we provide evidence that the network considers multiple possible move sequences rather than focusing on a single line of play. These results offer new insights into the emergence of sophisticated look-ahead capabilities in neural networks trained on strategic tasks, contributing to our understanding of AI reasoning in complex domains. Our work also showcases the effectiveness of interpretability techniques in uncovering cognitive-like processes in artificial intelligence systems. 

---
# Maximizing Confidence Alone Improves Reasoning 

**Authors**: Mihir Prabhudesai, Lili Chen, Alex Ippoliti, Katerina Fragkiadaki, Hao Liu, Deepak Pathak  

**Link**: [PDF](https://arxiv.org/pdf/2505.22660)  

**Abstract**: Reinforcement learning (RL) has enabled machine learning models to achieve significant advances in many fields. Most recently, RL has empowered frontier language models to solve challenging math, science, and coding problems. However, central to any RL algorithm is the reward function, and reward engineering is a notoriously difficult problem in any domain. In this paper, we propose RENT: Reinforcement Learning via Entropy Minimization -- a fully unsupervised RL method that requires no external reward or ground-truth answers, and instead uses the model's entropy of its underlying distribution as an intrinsic reward. We find that by reinforcing the chains of thought that yield high model confidence on its generated answers, the model improves its reasoning ability. In our experiments, we showcase these improvements on an extensive suite of commonly-used reasoning benchmarks, including GSM8K, MATH500, AMC, AIME, and GPQA, and models of varying sizes from the Qwen and Mistral families. The generality of our unsupervised learning method lends itself to applicability in a wide range of domains where external supervision is limited or unavailable. 

---
# 3DLLM-Mem: Long-Term Spatial-Temporal Memory for Embodied 3D Large Language Model 

**Authors**: Wenbo Hu, Yining Hong, Yanjun Wang, Leison Gao, Zibu Wei, Xingcheng Yao, Nanyun Peng, Yonatan Bitton, Idan Szpektor, Kai-Wei Chang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22657)  

**Abstract**: Humans excel at performing complex tasks by leveraging long-term memory across temporal and spatial experiences. In contrast, current Large Language Models (LLMs) struggle to effectively plan and act in dynamic, multi-room 3D environments. We posit that part of this limitation is due to the lack of proper 3D spatial-temporal memory modeling in LLMs. To address this, we first introduce 3DMem-Bench, a comprehensive benchmark comprising over 26,000 trajectories and 2,892 embodied tasks, question-answering and captioning, designed to evaluate an agent's ability to reason over long-term memory in 3D environments. Second, we propose 3DLLM-Mem, a novel dynamic memory management and fusion model for embodied spatial-temporal reasoning and actions in LLMs. Our model uses working memory tokens, which represents current observations, as queries to selectively attend to and fuse the most useful spatial and temporal features from episodic memory, which stores past observations and interactions. Our approach allows the agent to focus on task-relevant information while maintaining memory efficiency in complex, long-horizon environments. Experimental results demonstrate that 3DLLM-Mem achieves state-of-the-art performance across various tasks, outperforming the strongest baselines by 16.5% in success rate on 3DMem-Bench's most challenging in-the-wild embodied tasks. 

---
# Position: Uncertainty Quantification Needs Reassessment for Large-language Model Agents 

**Authors**: Michael Kirchhof, Gjergji Kasneci, Enkelejda Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2505.22655)  

**Abstract**: Large-language models (LLMs) and chatbot agents are known to provide wrong outputs at times, and it was recently found that this can never be fully prevented. Hence, uncertainty quantification plays a crucial role, aiming to quantify the level of ambiguity in either one overall number or two numbers for aleatoric and epistemic uncertainty. This position paper argues that this traditional dichotomy of uncertainties is too limited for the open and interactive setup that LLM agents operate in when communicating with a user, and that we need to research avenues that enrich uncertainties in this novel scenario. We review the literature and find that popular definitions of aleatoric and epistemic uncertainties directly contradict each other and lose their meaning in interactive LLM agent settings. Hence, we propose three novel research directions that focus on uncertainties in such human-computer interactions: Underspecification uncertainties, for when users do not provide all information or define the exact task at the first go, interactive learning, to ask follow-up questions and reduce the uncertainty about the current context, and output uncertainties, to utilize the rich language and speech space to express uncertainties as more than mere numbers. We expect that these new ways of dealing with and communicating uncertainties will lead to LLM agent interactions that are more transparent, trustworthy, and intuitive. 

---
# Pre-training for Recommendation Unlearning 

**Authors**: Guoxuan Chen, Lianghao Xia, Chao Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22649)  

**Abstract**: Modern recommender systems powered by Graph Neural Networks (GNNs) excel at modeling complex user-item interactions, yet increasingly face scenarios requiring selective forgetting of training data. Beyond user requests to remove specific interactions due to privacy concerns or preference changes, regulatory frameworks mandate recommender systems' ability to eliminate the influence of certain user data from models. This recommendation unlearning challenge presents unique difficulties as removing connections within interaction graphs creates ripple effects throughout the model, potentially impacting recommendations for numerous users. Traditional approaches suffer from significant drawbacks: fragmentation methods damage graph structure and diminish performance, while influence function techniques make assumptions that may not hold in complex GNNs, particularly with self-supervised or random architectures. To address these limitations, we propose a novel model-agnostic pre-training paradigm UnlearnRec that prepares systems for efficient unlearning operations. Our Influence Encoder takes unlearning requests together with existing model parameters and directly produces updated parameters of unlearned model with little fine-tuning, avoiding complete retraining while preserving model performance characteristics. Extensive evaluation on public benchmarks demonstrates that our method delivers exceptional unlearning effectiveness while providing more than 10x speedup compared to retraining approaches. We release our method implementation at: this https URL. 

---
# FastTD3: Simple, Fast, and Capable Reinforcement Learning for Humanoid Control 

**Authors**: Younggyo Seo, Carmelo Sferrazza, Haoran Geng, Michal Nauman, Zhao-Heng Yin, Pieter Abbeel  

**Link**: [PDF](https://arxiv.org/pdf/2505.22642)  

**Abstract**: Reinforcement learning (RL) has driven significant progress in robotics, but its complexity and long training times remain major bottlenecks. In this report, we introduce FastTD3, a simple, fast, and capable RL algorithm that significantly speeds up training for humanoid robots in popular suites such as HumanoidBench, IsaacLab, and MuJoCo Playground. Our recipe is remarkably simple: we train an off-policy TD3 agent with several modifications -- parallel simulation, large-batch updates, a distributional critic, and carefully tuned hyperparameters. FastTD3 solves a range of HumanoidBench tasks in under 3 hours on a single A100 GPU, while remaining stable during training. We also provide a lightweight and easy-to-use implementation of FastTD3 to accelerate RL research in robotics. 

---
# Learning Composable Chains-of-Thought 

**Authors**: Fangcong Yin, Zeyu Leo Liu, Liu Leqi, Xi Ye, Greg Durrett  

**Link**: [PDF](https://arxiv.org/pdf/2505.22635)  

**Abstract**: A common approach for teaching large language models (LLMs) to reason is to train on chain-of-thought (CoT) traces of in-distribution reasoning problems, but such annotated data is costly to obtain for every problem of interest. We want reasoning models to generalize beyond their training distribution, and ideally to generalize compositionally: combine atomic reasoning skills to solve harder, unseen reasoning tasks. We take a step towards compositional generalization of reasoning skills when addressing a target compositional task that has no labeled CoT data. We find that simply training models on CoT data of atomic tasks leads to limited generalization, but minimally modifying CoT formats of constituent atomic tasks to be composable can lead to improvements. We can train "atomic CoT" models on the atomic tasks with Composable CoT data and combine them with multitask learning or model merging for better zero-shot performance on the target compositional task. Such a combined model can be further bootstrapped on a small amount of compositional data using rejection sampling fine-tuning (RFT). Results on string operations and natural language skill compositions show that training LLMs on Composable CoT outperforms multitask learning and continued fine-tuning baselines within a given training data budget. 

---
# Spatial Knowledge Graph-Guided Multimodal Synthesis 

**Authors**: Yida Xue, Zhen Bi, Jinnan Yang, Jungang Lou, Huajun Chen, Ningyu Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22633)  

**Abstract**: Recent advances in multimodal large language models (MLLMs) have significantly enhanced their capabilities; however, their spatial perception abilities remain a notable limitation. To address this challenge, multimodal data synthesis offers a promising solution. Yet, ensuring that synthesized data adhere to spatial common sense is a non-trivial task. In this work, we introduce SKG2Data, a novel multimodal synthesis approach guided by spatial knowledge graphs, grounded in the concept of knowledge-to-data generation. SKG2Data automatically constructs a Spatial Knowledge Graph (SKG) to emulate human-like perception of spatial directions and distances, which is subsequently utilized to guide multimodal data synthesis. Extensive experiments demonstrate that data synthesized from diverse types of spatial knowledge, including direction and distance, not only enhance the spatial perception and reasoning abilities of MLLMs but also exhibit strong generalization capabilities. We hope that the idea of knowledge-based data synthesis can advance the development of spatial intelligence. 

---
# SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning 

**Authors**: Yu Zhang, Yuqi Xie, Huihan Liu, Rutav Shah, Michael Wan, Linxi Fan, Yuke Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22626)  

**Abstract**: Imitation learning advances robot capabilities by enabling the acquisition of diverse behaviors from human demonstrations. However, large-scale datasets used for policy training often introduce substantial variability in quality, which can negatively impact performance. As a result, automatically curating datasets by filtering low-quality samples to improve quality becomes essential. Existing robotic curation approaches rely on costly manual annotations and perform curation at a coarse granularity, such as the dataset or trajectory level, failing to account for the quality of individual state-action pairs. To address this, we introduce SCIZOR, a self-supervised data curation framework that filters out low-quality state-action pairs to improve the performance of imitation learning policies. SCIZOR targets two complementary sources of low-quality data: suboptimal data, which hinders learning with undesirable actions, and redundant data, which dilutes training with repetitive patterns. SCIZOR leverages a self-supervised task progress predictor for suboptimal data to remove samples lacking task progression, and a deduplication module operating on joint state-action representation for samples with redundant patterns. Empirically, we show that SCIZOR enables imitation learning policies to achieve higher performance with less data, yielding an average improvement of 15.4% across multiple benchmarks. More information is available at: this https URL 

---
# The Entropy Mechanism of Reinforcement Learning for Reasoning Language Models 

**Authors**: Ganqu Cui, Yuchen Zhang, Jiacheng Chen, Lifan Yuan, Zhi Wang, Yuxin Zuo, Haozhan Li, Yuchen Fan, Huayu Chen, Weize Chen, Zhiyuan Liu, Hao Peng, Lei Bai, Wanli Ouyang, Yu Cheng, Bowen Zhou, Ning Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.22617)  

**Abstract**: This paper aims to overcome a major obstacle in scaling RL for reasoning with LLMs, namely the collapse of policy entropy. Such phenomenon is consistently observed across vast RL runs without entropy intervention, where the policy entropy dropped sharply at the early training stage, this diminished exploratory ability is always accompanied with the saturation of policy performance. In practice, we establish a transformation equation R=-a*e^H+b between entropy H and downstream performance R. This empirical law strongly indicates that, the policy performance is traded from policy entropy, thus bottlenecked by its exhaustion, and the ceiling is fully predictable H=0, R=-a+b. Our finding necessitates entropy management for continuous exploration toward scaling compute for RL. To this end, we investigate entropy dynamics both theoretically and empirically. Our derivation highlights that, the change in policy entropy is driven by the covariance between action probability and the change in logits, which is proportional to its advantage when using Policy Gradient-like algorithms. Empirical study shows that, the values of covariance term and entropy differences matched exactly, supporting the theoretical conclusion. Moreover, the covariance term stays mostly positive throughout training, further explaining why policy entropy would decrease monotonically. Through understanding the mechanism behind entropy dynamics, we motivate to control entropy by restricting the update of high-covariance tokens. Specifically, we propose two simple yet effective techniques, namely Clip-Cov and KL-Cov, which clip and apply KL penalty to tokens with high covariances respectively. Experiments show that these methods encourage exploration, thus helping policy escape entropy collapse and achieve better downstream performance. 

---
# RICO: Improving Accuracy and Completeness in Image Recaptioning via Visual Reconstruction 

**Authors**: Yuchi Wang, Yishuo Cai, Shuhuai Ren, Sihan Yang, Linli Yao, Yuanxin Liu, Yuanxing Zhang, Pengfei Wan, Xu Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.22613)  

**Abstract**: Image recaptioning is widely used to generate training datasets with enhanced quality for various multimodal tasks. Existing recaptioning methods typically rely on powerful multimodal large language models (MLLMs) to enhance textual descriptions, but often suffer from inaccuracies due to hallucinations and incompleteness caused by missing fine-grained details. To address these limitations, we propose RICO, a novel framework that refines captions through visual reconstruction. Specifically, we leverage a text-to-image model to reconstruct a caption into a reference image, and prompt an MLLM to identify discrepancies between the original and reconstructed images to refine the caption. This process is performed iteratively, further progressively promoting the generation of more faithful and comprehensive descriptions. To mitigate the additional computational cost induced by the iterative process, we introduce RICO-Flash, which learns to generate captions like RICO using DPO. Extensive experiments demonstrate that our approach significantly improves caption accuracy and completeness, outperforms most baselines by approximately 10% on both CapsBench and CompreCap. Code released at this https URL. 

---
# Effective and Efficient One-pass Compression of Speech Foundation Models Using Sparsity-aware Self-pinching Gates 

**Authors**: Haoning Xu, Zhaoqing Li, Youjun Chen, Huimeng Wang, Guinan Li, Mengzhe Geng, Chengxi Deng, Xunying Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22608)  

**Abstract**: This paper presents a novel approach for speech foundation models compression that tightly integrates model pruning and parameter update into a single stage. Highly compact layer-level tied self-pinching gates each containing only a single learnable threshold are jointly trained with uncompressed models and used in fine-grained neuron level pruning. Experiments conducted on the LibriSpeech-100hr corpus suggest that our approach reduces the number of parameters of wav2vec2.0-base and HuBERT-large models by 65% and 60% respectively, while incurring no statistically significant word error rate (WER) increase on the test-clean dataset. Compared to previously published methods on the same task, our approach not only achieves the lowest WER of 7.05% on the test-clean dataset under a comparable model compression ratio of 4.26x, but also operates with at least 25% less model compression time. 

---
# One Rank at a Time: Cascading Error Dynamics in Sequential Learning 

**Authors**: Mahtab Alizadeh Vandchali, Fangshuo, Liao, Anastasios Kyrillidis  

**Link**: [PDF](https://arxiv.org/pdf/2505.22602)  

**Abstract**: Sequential learning -- where complex tasks are broken down into simpler, hierarchical components -- has emerged as a paradigm in AI. This paper views sequential learning through the lens of low-rank linear regression, focusing specifically on how errors propagate when learning rank-1 subspaces sequentially. We present an analysis framework that decomposes the learning process into a series of rank-1 estimation problems, where each subsequent estimation depends on the accuracy of previous steps. Our contribution is a characterization of the error propagation in this sequential process, establishing bounds on how errors -- e.g., due to limited computational budgets and finite precision -- affect the overall model accuracy. We prove that these errors compound in predictable ways, with implications for both algorithmic design and stability guarantees. 

---
# Machine Unlearning under Overparameterization 

**Authors**: Jacob L. Block, Aryan Mokhtari, Sanjay Shakkottai  

**Link**: [PDF](https://arxiv.org/pdf/2505.22601)  

**Abstract**: Machine unlearning algorithms aim to remove the influence of specific training samples, ideally recovering the model that would have resulted from training on the remaining data alone. We study unlearning in the overparameterized setting, where many models interpolate the data, and defining the unlearning solution as any loss minimizer over the retained set$\unicode{x2013}$as in prior work in the underparameterized setting$\unicode{x2013}$is inadequate, since the original model may already interpolate the retained data and satisfy this condition. In this regime, loss gradients vanish, rendering prior methods based on gradient perturbations ineffective, motivating both new unlearning definitions and algorithms. For this setting, we define the unlearning solution as the minimum-complexity interpolator over the retained data and propose a new algorithmic framework that only requires access to model gradients on the retained set at the original solution. We minimize a regularized objective over perturbations constrained to be orthogonal to these model gradients, a first-order relaxation of the interpolation condition. For different model classes, we provide exact and approximate unlearning guarantees, and we demonstrate that an implementation of our framework outperforms existing baselines across various unlearning experiments. 

---
# On the performance of machine-learning assisted Monte Carlo in sampling from simple statistical physics models 

**Authors**: Luca Maria Del Bono, Federico Ricci-Tersenghi, Francesco Zamponi  

**Link**: [PDF](https://arxiv.org/pdf/2505.22598)  

**Abstract**: Recent years have seen a rise in the application of machine learning techniques to aid the simulation of hard-to-sample systems that cannot be studied using traditional methods. Despite the introduction of many different architectures and procedures, a wide theoretical understanding is still lacking, with the risk of suboptimal implementations. As a first step to address this gap, we provide here a complete analytic study of the widely-used Sequential Tempering procedure applied to a shallow MADE architecture for the Curie-Weiss model. The contribution of this work is twofold: firstly, we give a description of the optimal weights and of the training under Gradient Descent optimization. Secondly, we compare what happens in Sequential Tempering with and without the addition of local Metropolis Monte Carlo steps. We are thus able to give theoretical predictions on the best procedure to apply in this case. This work establishes a clear theoretical basis for the integration of machine learning techniques into Monte Carlo sampling and optimization. 

---
# Self-Error-Instruct: Generalizing from Errors for LLMs Mathematical Reasoning 

**Authors**: Erxin Yu, Jing Li, Ming Liao, Qi Zhu, Boyang Xue, Minghui Xu, Baojun Wang, Lanqing Hong, Fei Mi, Lifeng Shang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22591)  

**Abstract**: Although large language models demonstrate strong performance across various domains, they still struggle with numerous bad cases in mathematical reasoning. Previous approaches to learning from errors synthesize training data by solely extrapolating from isolated bad cases, thereby failing to generalize the extensive patterns inherent within these cases. This paper presents Self-Error-Instruct (SEI), a framework that addresses these model weaknesses and synthesizes more generalized targeted training data. Specifically, we explore a target model on two mathematical datasets, GSM8K and MATH, to pinpoint bad cases. Then, we generate error keyphrases for these cases based on the instructor model's (GPT-4o) analysis and identify error types by clustering these keyphrases. Next, we sample a few bad cases during each generation for each identified error type and input them into the instructor model, which synthesizes additional training data using a self-instruct approach. This new data is refined through a one-shot learning process to ensure that only the most effective examples are kept. Finally, we use these curated data to fine-tune the target model, iteratively repeating the process to enhance performance. We apply our framework to various models and observe improvements in their reasoning abilities across both in-domain and out-of-domain mathematics datasets. These results demonstrate the effectiveness of self-error instruction in improving LLMs' mathematical reasoning through error generalization. 

---
# GitGoodBench: A Novel Benchmark For Evaluating Agentic Performance On Git 

**Authors**: Tobias Lindenbauer, Egor Bogomolov, Yaroslav Zharov  

**Link**: [PDF](https://arxiv.org/pdf/2505.22583)  

**Abstract**: Benchmarks for Software Engineering (SE) AI agents, most notably SWE-bench, have catalyzed progress in programming capabilities of AI agents. However, they overlook critical developer workflows such as Version Control System (VCS) operations. To address this issue, we present GitGoodBench, a novel benchmark for evaluating AI agent performance on VCS tasks. GitGoodBench covers three core Git scenarios extracted from permissive open-source Python, Java, and Kotlin repositories. Our benchmark provides three datasets: a comprehensive evaluation suite (900 samples), a rapid prototyping version (120 samples), and a training corpus (17,469 samples). We establish baseline performance on the prototyping version of our benchmark using GPT-4o equipped with custom tools, achieving a 21.11% solve rate overall. We expect GitGoodBench to serve as a crucial stepping stone toward truly comprehensive SE agents that go beyond mere programming. 

---
# Tell me Habibi, is it Real or Fake? 

**Authors**: Kartik Kuckreja, Parul Gupta, Injy Hamed, Thamar Solorio, Muhammad Haris Khan, Abhinav Dhall  

**Link**: [PDF](https://arxiv.org/pdf/2505.22581)  

**Abstract**: Deepfake generation methods are evolving fast, making fake media harder to detect and raising serious societal concerns. Most deepfake detection and dataset creation research focuses on monolingual content, often overlooking the challenges of multilingual and code-switched speech, where multiple languages are mixed within the same discourse. Code-switching, especially between Arabic and English, is common in the Arab world and is widely used in digital communication. This linguistic mixing poses extra challenges for deepfake detection, as it can confuse models trained mostly on monolingual data. To address this, we introduce \textbf{ArEnAV}, the first large-scale Arabic-English audio-visual deepfake dataset featuring intra-utterance code-switching, dialectal variation, and monolingual Arabic content. It \textbf{contains 387k videos and over 765 hours of real and fake videos}. Our dataset is generated using a novel pipeline integrating four Text-To-Speech and two lip-sync models, enabling comprehensive analysis of multilingual multimodal deepfake detection. We benchmark our dataset against existing monolingual and multilingual datasets, state-of-the-art deepfake detection models, and a human evaluation, highlighting its potential to advance deepfake research. The dataset can be accessed \href{this https URL}{here}. 

---
# Fusion Steering: Prompt-Specific Activation Control 

**Authors**: Waldemar Chang, Alhassan Yasin  

**Link**: [PDF](https://arxiv.org/pdf/2505.22572)  

**Abstract**: We present Fusion Steering, an activation steering methodology that improves factual accuracy in large language models (LLMs) for question-answering (QA) tasks. This approach introduces flexible steering configurations, including full-layer steering and segmented steering. Unlike traditional methods constrained to single-layer or fixed-layer operations, Fusion Steering employs dynamic injection of prompt-specific activation deltas across all transformer layers. These activation deltas are derived from reference completions that combine the ground-truth answer with a model-generated explanation to facilitate semantically enriched, example-specific steering. The injection weights are optimized per prompt using Optuna, targeting a joint objective that balances token overlap (factual alignment) and perplexity (fluency proxy). Evaluation employs a composite score integrating token overlap and LLM-graded quality, encompassing factual accuracy, coherence, and relevance. Empirical results on 260 SimpleQA prompts (selected from 500 where the baseline failed) showcase the efficacy of segmented steering. Using Gemma-2-2B-IT with 8-bit quantization, segmented steering achieves an accuracy of 25.4% (outputs scoring $\geq 0.6$), outperforming the baseline at 3.5% and full-layer steering at 16.2%. Under the stricter SimpleQA rubric, segmented steering boosts fully correct responses from 0.0% to 13.1%. These findings highlight the strengths of segmented, dynamic intervention strategies and the promise of per-prompt, full-network activation control. Fusion Steering is also amenable to sparse representations, such as Neuronpedia or sparse crosscoders, suggesting a promising direction for interpretable and scalable activation-level control in LLMs. 

---
# Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems 

**Authors**: Hoang Pham, Khac-Hoai Nam Bui  

**Link**: [PDF](https://arxiv.org/pdf/2505.22571)  

**Abstract**: This paper presents a novel approach for unified retrieval-augmented generation (RAG) systems using the recent emerging large language model (LLM) agent concept. Specifically, Agent LLM, which utilizes LLM as fundamental controllers, has become a promising approach to enable the interpretability of RAG tasks, especially for complex reasoning question-answering systems (e.g., multi-hop queries). Nonetheless, previous works mainly focus on solving RAG systems with either single-hop or multi-hop approaches separately, which limits the application of those approaches to real-world applications. In this study, we propose a trainable agent framework called Agent-UniRAG for unified retrieval-augmented LLM systems, which enhances the effectiveness and interpretability of RAG systems. The main idea is to design an LLM agent framework to solve RAG tasks step-by-step based on the complexity of the inputs, simultaneously including single-hop and multi-hop queries in an end-to-end manner. Furthermore, we introduce SynAgent-RAG, a synthetic dataset to enable the proposed agent framework for small open-source LLMs (e.g., Llama-3-8B). The results show comparable performances with closed-source and larger open-source LLMs across various RAG benchmarks. Our source code and dataset are publicly available for further exploitation. 

---
# Universal Visuo-Tactile Video Understanding for Embodied Interaction 

**Authors**: Yifan Xie, Mingyang Li, Shoujie Li, Xingting Li, Guangyu Chen, Fei Ma, Fei Richard Yu, Wenbo Ding  

**Link**: [PDF](https://arxiv.org/pdf/2505.22566)  

**Abstract**: Tactile perception is essential for embodied agents to understand physical attributes of objects that cannot be determined through visual inspection alone. While existing approaches have made progress in visual and language modalities for physical understanding, they fail to effectively incorporate tactile information that provides crucial haptic feedback for real-world interaction. In this paper, we present VTV-LLM, the first multi-modal large language model for universal Visuo-Tactile Video (VTV) understanding that bridges the gap between tactile perception and natural language. To address the challenges of cross-sensor and cross-modal integration, we contribute VTV150K, a comprehensive dataset comprising 150,000 video frames from 100 diverse objects captured across three different tactile sensors (GelSight Mini, DIGIT, and Tac3D), annotated with four fundamental tactile attributes (hardness, protrusion, elasticity, and friction). We develop a novel three-stage training paradigm that includes VTV enhancement for robust visuo-tactile representation, VTV-text alignment for cross-modal correspondence, and text prompt finetuning for natural language generation. Our framework enables sophisticated tactile reasoning capabilities including feature assessment, comparative analysis, scenario-based decision making and so on. Experimental evaluations demonstrate that VTV-LLM achieves superior performance in tactile video understanding tasks, establishing a foundation for more intuitive human-machine interaction in tactile domains. 

---
# PRISM: Video Dataset Condensation with Progressive Refinement and Insertion for Sparse Motion 

**Authors**: Jaehyun Choi, Jiwan Hur, Gyojin Han, Jaemyung Yu, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.22564)  

**Abstract**: Video dataset condensation has emerged as a critical technique for addressing the computational challenges associated with large-scale video data processing in deep learning applications. While significant progress has been made in image dataset condensation, the video domain presents unique challenges due to the complex interplay between spatial content and temporal dynamics. This paper introduces PRISM, Progressive Refinement and Insertion for Sparse Motion, for video dataset condensation, a novel approach that fundamentally reconsiders how video data should be condensed. Unlike the previous method that separates static content from dynamic motion, our method preserves the essential interdependence between these elements. Our approach progressively refines and inserts frames to fully accommodate the motion in an action while achieving better performance but less storage, considering the relation of gradients for each frame. Extensive experiments across standard video action recognition benchmarks demonstrate that PRISM outperforms existing disentangled approaches while maintaining compact representations suitable for resource-constrained environments. 

---
# ClaimPKG: Enhancing Claim Verification via Pseudo-Subgraph Generation with Lightweight Specialized LLM 

**Authors**: Hoang Pham, Thanh-Do Nguyen, Khac-Hoai Nam Bui  

**Link**: [PDF](https://arxiv.org/pdf/2505.22552)  

**Abstract**: Integrating knowledge graphs (KGs) to enhance the reasoning capabilities of large language models (LLMs) is an emerging research challenge in claim verification. While KGs provide structured, semantically rich representations well-suited for reasoning, most existing verification methods rely on unstructured text corpora, limiting their ability to effectively leverage KGs. Additionally, despite possessing strong reasoning abilities, modern LLMs struggle with multi-step modular pipelines and reasoning over KGs without adaptation. To address these challenges, we propose ClaimPKG, an end-to-end framework that seamlessly integrates LLM reasoning with structured knowledge from KGs. Specifically, the main idea of ClaimPKG is to employ a lightweight, specialized LLM to represent the input claim as pseudo-subgraphs, guiding a dedicated subgraph retrieval module to identify relevant KG subgraphs. These retrieved subgraphs are then processed by a general-purpose LLM to produce the final verdict and justification. Extensive experiments on the FactKG dataset demonstrate that ClaimPKG achieves state-of-the-art performance, outperforming strong baselines in this research field by 9%-12% accuracy points across multiple categories. Furthermore, ClaimPKG exhibits zero-shot generalizability to unstructured datasets such as HoVer and FEVEROUS, effectively combining structured knowledge from KGs with LLM reasoning across various LLM backbones. 

---
# Scaling-up Perceptual Video Quality Assessment 

**Authors**: Ziheng Jia, Zicheng Zhang, Zeyu Zhang, Yingji Liang, Xiaorong Zhu, Chunyi Li, Jinliang Han, Haoning Wu, Bin Wang, Haoran Zhang, Guanyu Zhu, Qiyong Zhao, Xiaohong Liu, Guangtao Zhai, Xiongkuo Min  

**Link**: [PDF](https://arxiv.org/pdf/2505.22543)  

**Abstract**: The data scaling law has been shown to significantly enhance the performance of large multi-modal models (LMMs) across various downstream tasks. However, in the domain of perceptual video quality assessment (VQA), the potential of scaling law remains unprecedented due to the scarcity of labeled resources and the insufficient scale of datasets. To address this, we propose \textbf{OmniVQA}, an efficient framework designed to efficiently build high-quality, human-in-the-loop VQA multi-modal instruction databases (MIDBs). We then scale up to create \textbf{OmniVQA-Chat-400K}, the largest MIDB in the VQA field concurrently. Our focus is on the technical and aesthetic quality dimensions, with abundant in-context instruction data to provide fine-grained VQA knowledge. Additionally, we have built the \textbf{OmniVQA-MOS-20K} dataset to enhance the model's quantitative quality rating capabilities. We then introduce a \textbf{complementary} training strategy that effectively leverages the knowledge from datasets for quality understanding and quality rating tasks. Furthermore, we propose the \textbf{OmniVQA-FG (fine-grain)-Benchmark} to evaluate the fine-grained performance of the models. Our results demonstrate that our models achieve state-of-the-art performance in both quality understanding and rating tasks. 

---
# TabularQGAN: A Quantum Generative Model for Tabular Data 

**Authors**: Pallavi Bhardwaj, Caitlin Jones, Lasse Dierich, Aleksandar Vučković  

**Link**: [PDF](https://arxiv.org/pdf/2505.22533)  

**Abstract**: In this paper, we introduce a novel quantum generative model for synthesizing tabular data. Synthetic data is valuable in scenarios where real-world data is scarce or private, it can be used to augment or replace existing datasets. Real-world enterprise data is predominantly tabular and heterogeneous, often comprising a mixture of categorical and numerical features, making it highly relevant across various industries such as healthcare, finance, and software. We propose a quantum generative adversarial network architecture with flexible data encoding and a novel quantum circuit ansatz to effectively model tabular data. The proposed approach is tested on the MIMIC III healthcare and Adult Census datasets, with extensive benchmarking against leading classical models, CTGAN, and CopulaGAN. Experimental results demonstrate that our quantum model outperforms classical models by an average of 8.5% with respect to an overall similarity score from SDMetrics, while using only 0.072% of the parameters of the classical models. Additionally, we evaluate the generalization capabilities of the models using two custom-designed metrics that demonstrate the ability of the proposed quantum model to generate useful and novel samples. To our knowledge, this is one of the first demonstrations of a successful quantum generative model for handling tabular data, indicating that this task could be well-suited to quantum computers. 

---
# Training RL Agents for Multi-Objective Network Defense Tasks 

**Authors**: Andres Molina-Markham, Luis Robaina, Sean Steinle, Akash Trivedi, Derek Tsui, Nicholas Potteiger, Lauren Brandt, Ransom Winder, Ahmed Ridley  

**Link**: [PDF](https://arxiv.org/pdf/2505.22531)  

**Abstract**: Open-ended learning (OEL) -- which emphasizes training agents that achieve broad capability over narrow competency -- is emerging as a paradigm to develop artificial intelligence (AI) agents to achieve robustness and generalization. However, despite promising results that demonstrate the benefits of OEL, applying OEL to develop autonomous agents for real-world cybersecurity applications remains a challenge.
We propose a training approach, inspired by OEL, to develop autonomous network defenders. Our results demonstrate that like in other domains, OEL principles can translate into more robust and generalizable agents for cyber defense. To apply OEL to network defense, it is necessary to address several technical challenges. Most importantly, it is critical to provide a task representation approach over a broad universe of tasks that maintains a consistent interface over goals, rewards and action spaces. This way, the learning agent can train with varying network conditions, attacker behaviors, and defender goals while being able to build on previously gained knowledge.
With our tools and results, we aim to fundamentally impact research that applies AI to solve cybersecurity problems. Specifically, as researchers develop gyms and benchmarks for cyber defense, it is paramount that they consider diverse tasks with consistent representations, such as those we propose in our work. 

---
# Thinking with Generated Images 

**Authors**: Ethan Chern, Zhulin Hu, Steffi Chern, Siqi Kou, Jiadi Su, Yan Ma, Zhijie Deng, Pengfei Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22525)  

**Abstract**: We present Thinking with Generated Images, a novel paradigm that fundamentally transforms how large multimodal models (LMMs) engage with visual reasoning by enabling them to natively think across text and vision modalities through spontaneous generation of intermediate visual thinking steps. Current visual reasoning with LMMs is constrained to either processing fixed user-provided images or reasoning solely through text-based chain-of-thought (CoT). Thinking with Generated Images unlocks a new dimension of cognitive capability where models can actively construct intermediate visual thoughts, critique their own visual hypotheses, and refine them as integral components of their reasoning process. We demonstrate the effectiveness of our approach through two complementary mechanisms: (1) vision generation with intermediate visual subgoals, where models decompose complex visual tasks into manageable components that are generated and integrated progressively, and (2) vision generation with self-critique, where models generate an initial visual hypothesis, analyze its shortcomings through textual reasoning, and produce refined outputs based on their own critiques. Our experiments on vision generation benchmarks show substantial improvements over baseline approaches, with our models achieving up to 50% (from 38% to 57%) relative improvement in handling complex multi-object scenarios. From biochemists exploring novel protein structures, and architects iterating on spatial designs, to forensic analysts reconstructing crime scenes, and basketball players envisioning strategic plays, our approach enables AI models to engage in the kind of visual imagination and iterative refinement that characterizes human creative, analytical, and strategic thinking. We release our open-source suite at this https URL. 

---
# Evaluating Supervised Learning Models for Fraud Detection: A Comparative Study of Classical and Deep Architectures on Imbalanced Transaction Data 

**Authors**: Chao Wang, Chuanhao Nie, Yunbo Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22521)  

**Abstract**: Fraud detection remains a critical task in high-stakes domains such as finance and e-commerce, where undetected fraudulent transactions can lead to significant economic losses. In this study, we systematically compare the performance of four supervised learning models - Logistic Regression, Random Forest, Light Gradient Boosting Machine (LightGBM), and a Gated Recurrent Unit (GRU) network - on a large-scale, highly imbalanced online transaction dataset. While ensemble methods such as Random Forest and LightGBM demonstrated superior performance in both overall and class-specific metrics, Logistic Regression offered a reliable and interpretable baseline. The GRU model showed strong recall for the minority fraud class, though at the cost of precision, highlighting a trade-off relevant for real-world deployment. Our evaluation emphasizes not only weighted averages but also per-class precision, recall, and F1-scores, providing a nuanced view of each model's effectiveness in detecting rare but consequential fraudulent activity. The findings underscore the importance of choosing models based on the specific risk tolerance and operational needs of fraud detection systems. 

---
# Strengthening Proportionality in Temporal Voting 

**Authors**: Bradley Phillips, Edith Elkind, Nicholas Teh, Tomasz Wąs  

**Link**: [PDF](https://arxiv.org/pdf/2505.22513)  

**Abstract**: We study proportional representation in the framework of temporal voting with approval ballots. Prior work adapted basic proportional representation concepts -- justified representation (JR), proportional JR (PJR), and extended JR (EJR) -- from the multiwinner setting to the temporal setting. Our work introduces and examines ways of going beyond EJR. Specifically, we consider stronger variants of JR, PJR, and EJR, and introduce temporal adaptations of more demanding multiwinner axioms, such as EJR+, full JR (FJR), full proportional JR (FPJR), and the Core. For each of these concepts, we investigate its existence and study its relationship to existing notions, thereby establishing a rich hierarchy of proportionality concepts. Notably, we show that two of our proposed axioms -- EJR+ and FJR -- strengthen EJR while remaining satisfiable in every temporal election. 

---
# From Strangers to Assistants: Fast Desire Alignment for Embodied Agent-User Adaptation 

**Authors**: Yuanfei Wang, Xinju Huang, Fangwei Zhong, Yaodong Yang, Yizhou Wang, Yuanpei Chen, Hao Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.22503)  

**Abstract**: While embodied agents have made significant progress in performing complex physical tasks, real-world applications demand more than pure task execution. The agents must collaborate with unfamiliar agents and human users, whose goals are often vague and implicit. In such settings, interpreting ambiguous instructions and uncovering underlying desires is essential for effective assistance. Therefore, fast and accurate desire alignment becomes a critical capability for embodied agents. In this work, we first develop a home assistance simulation environment HA-Desire that integrates an LLM-driven human user agent exhibiting realistic value-driven goal selection and communication. The ego agent must interact with this proxy user to infer and adapt to the user's latent desires. To achieve this, we present a novel framework FAMER for fast desire alignment, which introduces a desire-based mental reasoning mechanism to identify user intent and filter desire-irrelevant actions. We further design a reflection-based communication module that reduces redundant inquiries, and incorporate goal-relevant information extraction with memory persistence to improve information reuse and reduce unnecessary exploration. Extensive experiments demonstrate that our framework significantly enhances both task execution and communication efficiency, enabling embodied agents to quickly adapt to user-specific desires in complex embodied environments. 

---
# Demystifying the Paradox of Importance Sampling with an Estimated History-Dependent Behavior Policy in Off-Policy Evaluation 

**Authors**: Hongyi Zhou, Josiah P. Hanna, Jin Zhu, Ying Yang, Chengchun Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.22492)  

**Abstract**: This paper studies off-policy evaluation (OPE) in reinforcement learning with a focus on behavior policy estimation for importance sampling. Prior work has shown empirically that estimating a history-dependent behavior policy can lead to lower mean squared error (MSE) even when the true behavior policy is Markovian. However, the question of why the use of history should lower MSE remains open. In this paper, we theoretically demystify this paradox by deriving a bias-variance decomposition of the MSE of ordinary importance sampling (IS) estimators, demonstrating that history-dependent behavior policy estimation decreases their asymptotic variances while increasing their finite-sample biases. Additionally, as the estimated behavior policy conditions on a longer history, we show a consistent decrease in variance. We extend these findings to a range of other OPE estimators, including the sequential IS estimator, the doubly robust estimator and the marginalized IS estimator, with the behavior policy estimated either parametrically or non-parametrically. 

---
# On the Surprising Effectiveness of Large Learning Rates under Standard Width Scaling 

**Authors**: Moritz Haas, Sebastian Bordt, Ulrike von Luxburg, Leena Chennuru Vankadara  

**Link**: [PDF](https://arxiv.org/pdf/2505.22491)  

**Abstract**: The dominant paradigm for training large-scale vision and language models is He initialization and a single global learning rate (\textit{standard parameterization}, SP). Despite its practical success, standard parametrization remains poorly understood from a theoretical perspective: Existing infinite-width theory would predict instability under large learning rates and vanishing feature learning under stable learning rates. However, empirically optimal learning rates consistently decay much slower than theoretically predicted. By carefully studying neural network training dynamics, we demonstrate that this discrepancy is not fully explained by finite-width phenomena such as catapult effects or a lack of alignment between weights and incoming activations. We instead show that the apparent contradiction can be fundamentally resolved by taking the loss function into account: In contrast to Mean Squared Error (MSE) loss, we prove that under cross-entropy (CE) loss, an intermediate \textit{controlled divergence} regime emerges, where logits diverge but loss, gradients, and activations remain stable. Stable training under large learning rates enables persistent feature evolution at scale in all hidden layers, which is crucial for the practical success of SP. In experiments across optimizers (SGD, Adam), architectures (MLPs, GPT) and data modalities (vision, language), we validate that neural networks operate in this controlled divergence regime under CE loss but not under MSE loss. Our empirical evidence suggests that width-scaling considerations are surprisingly useful for predicting empirically optimal learning rate exponents. Finally, our analysis clarifies the effectiveness and limitations of recently proposed layerwise learning rate scalings for standard initialization. 

---
# A Closer Look at Multimodal Representation Collapse 

**Authors**: Abhra Chaudhuri, Anjan Dutta, Tu Bui, Serban Georgescu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22483)  

**Abstract**: We aim to develop a fundamental understanding of modality collapse, a recently observed empirical phenomenon wherein models trained for multimodal fusion tend to rely only on a subset of the modalities, ignoring the rest. We show that modality collapse happens when noisy features from one modality are entangled, via a shared set of neurons in the fusion head, with predictive features from another, effectively masking out positive contributions from the predictive features of the former modality and leading to its collapse. We further prove that cross-modal knowledge distillation implicitly disentangles such representations by freeing up rank bottlenecks in the student encoder, denoising the fusion-head outputs without negatively impacting the predictive features from either modality. Based on the above findings, we propose an algorithm that prevents modality collapse through explicit basis reallocation, with applications in dealing with missing modalities. Extensive experiments on multiple multimodal benchmarks validate our theoretical claims. Project page: this https URL. 

---
# Human-Centered Human-AI Collaboration (HCHAC) 

**Authors**: Qi Gao, Wei Xu, Hanxi Pan, Mowei Shen, Zaifeng Gao  

**Link**: [PDF](https://arxiv.org/pdf/2505.22477)  

**Abstract**: In the intelligent era, the interaction between humans and intelligent systems fundamentally involves collaboration with autonomous intelligent agents. Human-AI Collaboration (HAC) represents a novel type of human-machine relationship facilitated by autonomous intelligent machines equipped with AI technologies. In this paradigm, AI agents serve not only as auxiliary tools but also as active teammates, partnering with humans to accomplish tasks collaboratively. Human-centered AI (HCAI) emphasizes that humans play critical leadership roles in the collaboration. This human-led collaboration imparts new dimensions to the human-machine relationship, necessitating innovative research perspectives, paradigms, and agenda to address the unique challenges posed by HAC. This chapter delves into the essence of HAC from the human-centered perspective, outlining its core concepts and distinguishing features. It reviews the current research methodologies and research agenda within the HAC field from the HCAI perspective, highlighting advancements and ongoing studies. Furthermore, a framework for human-centered HAC (HCHAC) is proposed by integrating these reviews and analyses. A case study of HAC in the context of autonomous vehicles is provided, illustrating practical applications and the synergistic interactions between humans and AI agents. Finally, it identifies potential future research directions aimed at enhancing the effectiveness, reliability, and ethical integration of human-centered HAC systems in diverse domains. 

---
# Topological Structure Learning Should Be A Research Priority for LLM-Based Multi-Agent Systems 

**Authors**: Jiaxi Yang, Mengqi Zhang, Yiqiao Jin, Hao Chen, Qingsong Wen, Lu Lin, Yi He, Weijie Xu, James Evans, Jindong Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22467)  

**Abstract**: Large Language Model-based Multi-Agent Systems (MASs) have emerged as a powerful paradigm for tackling complex tasks through collaborative intelligence. Nevertheless, the question of how agents should be structurally organized for optimal cooperation remains largely unexplored. In this position paper, we aim to gently redirect the focus of the MAS research community toward this critical dimension: develop topology-aware MASs for specific tasks. Specifically, the system consists of three core components - agents, communication links, and communication patterns - that collectively shape its coordination performance and efficiency. To this end, we introduce a systematic, three-stage framework: agent selection, structure profiling, and topology synthesis. Each stage would trigger new research opportunities in areas such as language models, reinforcement learning, graph learning, and generative modeling; together, they could unleash the full potential of MASs in complicated real-world applications. Then, we discuss the potential challenges and opportunities in the evaluation of multiple systems. We hope our perspective and framework can offer critical new insights in the era of agentic AI. 

---
# Fostering Video Reasoning via Next-Event Prediction 

**Authors**: Haonan Wang, Hongfu Liu, Xiangyan Liu, Chao Du, Kenji Kawaguchi, Ye Wang, Tianyu Pang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22457)  

**Abstract**: Next-token prediction serves as the foundational learning task enabling reasoning in LLMs. But what should the learning task be when aiming to equip MLLMs with temporal reasoning capabilities over video inputs? Existing tasks such as video question answering often rely on annotations from humans or much stronger MLLMs, while video captioning tends to entangle temporal reasoning with spatial information. To address this gap, we propose next-event prediction (NEP), a learning task that harnesses future video segments as a rich, self-supervised signal to foster temporal reasoning. We segment each video into past and future frames: the MLLM takes the past frames as input and predicts a summary of events derived from the future frames, thereby encouraging the model to reason temporally in order to complete the task. To support this task, we curate V1-33K, a dataset comprising 33,000 automatically extracted video segments spanning diverse real-world scenarios. We further explore a range of video instruction-tuning strategies to study their effects on temporal reasoning. To evaluate progress, we introduce FutureBench to assess coherence in predicting unseen future events. Experiments validate that NEP offers a scalable and effective training paradigm for fostering temporal reasoning in MLLMs. 

---
# Unsupervised Post-Training for Multi-Modal LLM Reasoning via GRPO 

**Authors**: Lai Wei, Yuting Li, Chen Wang, Yue Wang, Linghe Kong, Weiran Huang, Lichao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.22453)  

**Abstract**: Improving Multi-modal Large Language Models (MLLMs) in the post-training stage typically relies on supervised fine-tuning (SFT) or reinforcement learning (RL). However, these supervised methods require expensive and manually annotated multi-modal data--an ultimately unsustainable resource. While recent efforts have explored unsupervised post-training, their methods are complex and difficult to iterate. In this work, we are the first to investigate the use of GRPO, a stable and scalable online RL algorithm, for enabling continual self-improvement without any external supervision. We propose MM-UPT, a simple yet effective framework for unsupervised post-training of MLLMs. MM-UPT builds upon GRPO, replacing traditional reward signals with a self-rewarding mechanism based on majority voting over multiple sampled responses. Our experiments demonstrate that MM-UPT significantly improves the reasoning ability of Qwen2.5-VL-7B (e.g., 66.3 %$\rightarrow$72.9 % on MathVista, 62.9 %$\rightarrow$68.7 % on We-Math), using standard dataset without ground truth labels. MM-UPT also outperforms prior unsupervised baselines and even approaches the results of supervised GRPO. Furthermore, we show that incorporating synthetic questions, generated solely by MLLM itself, can boost performance as well, highlighting a promising approach for scalable self-improvement. Overall, MM-UPT offers a new paradigm for continual, autonomous enhancement of MLLMs in the absence of external supervision. Our code is available at this https URL. 

---
# NFR: Neural Feature-Guided Non-Rigid Shape Registration 

**Authors**: Puhua Jiang, Zhangquan Chen, Mingze Sun, Ruqi Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22445)  

**Abstract**: In this paper, we propose a novel learning-based framework for 3D shape registration, which overcomes the challenges of significant non-rigid deformation and partiality undergoing among input shapes, and, remarkably, requires no correspondence annotation during training. Our key insight is to incorporate neural features learned by deep learning-based shape matching networks into an iterative, geometric shape registration pipeline. The advantage of our approach is two-fold -- On one hand, neural features provide more accurate and semantically meaningful correspondence estimation than spatial features (e.g., coordinates), which is critical in the presence of large non-rigid deformations; On the other hand, the correspondences are dynamically updated according to the intermediate registrations and filtered by consistency prior, which prominently robustify the overall pipeline. Empirical results show that, with as few as dozens of training shapes of limited variability, our pipeline achieves state-of-the-art results on several benchmarks of non-rigid point cloud matching and partial shape matching across varying settings, but also delivers high-quality correspondences between unseen challenging shape pairs that undergo both significant extrinsic and intrinsic deformations, in which case neither traditional registration methods nor intrinsic methods work. 

---
# SOReL and TOReL: Two Methods for Fully Offline Reinforcement Learning 

**Authors**: Mattie Fellows, Clarisse Wibault, Uljad Berdica, Johannes Forkel, Jakob N. Foerster, Michael A. Osborne  

**Link**: [PDF](https://arxiv.org/pdf/2505.22442)  

**Abstract**: Sample efficiency remains a major obstacle for real world adoption of reinforcement learning (RL): success has been limited to settings where simulators provide access to essentially unlimited environment interactions, which in reality are typically costly or dangerous to obtain. Offline RL in principle offers a solution by exploiting offline data to learn a near-optimal policy before deployment. In practice, however, current offline RL methods rely on extensive online interactions for hyperparameter tuning, and have no reliable bound on their initial online performance. To address these two issues, we introduce two algorithms. Firstly, SOReL: an algorithm for safe offline reinforcement learning. Using only offline data, our Bayesian approach infers a posterior over environment dynamics to obtain a reliable estimate of the online performance via the posterior predictive uncertainty. Crucially, all hyperparameters are also tuned fully offline. Secondly, we introduce TOReL: a tuning for offline reinforcement learning algorithm that extends our information rate based offline hyperparameter tuning methods to general offline RL approaches. Our empirical evaluation confirms SOReL's ability to accurately estimate regret in the Bayesian setting whilst TOReL's offline hyperparameter tuning achieves competitive performance with the best online hyperparameter tuning methods using only offline data. Thus, SOReL and TOReL make a significant step towards safe and reliable offline RL, unlocking the potential for RL in the real world. Our implementations are publicly available: this https URL\_torel. 

---
# Can NeRFs See without Cameras? 

**Authors**: Chaitanya Amballa, Sattwik Basu, Yu-Lin Wei, Zhijian Yang, Mehmet Ergezer, Romit Roy Choudhury  

**Link**: [PDF](https://arxiv.org/pdf/2505.22441)  

**Abstract**: Neural Radiance Fields (NeRFs) have been remarkably successful at synthesizing novel views of 3D scenes by optimizing a volumetric scene function. This scene function models how optical rays bring color information from a 3D object to the camera pixels. Radio frequency (RF) or audio signals can also be viewed as a vehicle for delivering information about the environment to a sensor. However, unlike camera pixels, an RF/audio sensor receives a mixture of signals that contain many environmental reflections (also called "multipath"). Is it still possible to infer the environment using such multipath signals? We show that with redesign, NeRFs can be taught to learn from multipath signals, and thereby "see" the environment. As a grounding application, we aim to infer the indoor floorplan of a home from sparse WiFi measurements made at multiple locations inside the home. Although a difficult inverse problem, our implicitly learnt floorplans look promising, and enables forward applications, such as indoor signal prediction and basic ray tracing. 

---
# Synonymous Variational Inference for Perceptual Image Compression 

**Authors**: Zijian Liang, Kai Niu, Changshuo Wang, Jin Xu, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22438)  

**Abstract**: Recent contributions of semantic information theory reveal the set-element relationship between semantic and syntactic information, represented as synonymous relationships. In this paper, we propose a synonymous variational inference (SVI) method based on this synonymity viewpoint to re-analyze the perceptual image compression problem. It takes perceptual similarity as a typical synonymous criterion to build an ideal synonymous set (Synset), and approximate the posterior of its latent synonymous representation with a parametric density by minimizing a partial semantic KL divergence. This analysis theoretically proves that the optimization direction of perception image compression follows a triple tradeoff that can cover the existing rate-distortion-perception schemes. Additionally, we introduce synonymous image compression (SIC), a new image compression scheme that corresponds to the analytical process of SVI, and implement a progressive SIC codec to fully leverage the model's capabilities. Experimental results demonstrate comparable rate-distortion-perception performance using a single progressive SIC codec, thus verifying the effectiveness of our proposed analysis method. 

---
# Scaling Reasoning without Attention 

**Authors**: Xueliang Zhao, Wei Wu, Lingpeng Kong  

**Link**: [PDF](https://arxiv.org/pdf/2505.22425)  

**Abstract**: Large language models (LLMs) have made significant advances in complex reasoning tasks, yet they remain bottlenecked by two core challenges: architectural inefficiency due to reliance on Transformers, and a lack of structured fine-tuning for high-difficulty domains. We introduce \ourmodel, an attention-free language model that addresses both issues through architectural and data-centric innovations. Built on the state space dual (SSD) layers of Mamba-2, our model eliminates the need for self-attention and key-value caching, enabling fixed-memory, constant-time inference. To train it for complex reasoning, we propose a two-phase curriculum fine-tuning strategy based on the \textsc{PromptCoT} synthesis paradigm, which generates pedagogically structured problems via abstract concept selection and rationale-guided generation. On benchmark evaluations, \ourmodel-7B outperforms strong Transformer and hybrid models of comparable scale, and even surpasses the much larger Gemma3-27B by 2.6\% on AIME 24, 0.6\% on AIME 25, and 3.0\% on Livecodebench. These results highlight the potential of state space models as efficient and scalable alternatives to attention-based architectures for high-capacity reasoning. 

---
# Mitigating Overthinking in Large Reasoning Models via Manifold Steering 

**Authors**: Yao Huang, Huanran Chen, Shouwei Ruan, Yichi Zhang, Xingxing Wei, Yinpeng Dong  

**Link**: [PDF](https://arxiv.org/pdf/2505.22411)  

**Abstract**: Recent advances in Large Reasoning Models (LRMs) have demonstrated remarkable capabilities in solving complex tasks such as mathematics and coding. However, these models frequently exhibit a phenomenon known as overthinking during inference, characterized by excessive validation loops and redundant deliberation, leading to substantial computational overheads. In this paper, we aim to mitigate overthinking by investigating the underlying mechanisms from the perspective of mechanistic interpretability. We first showcase that the tendency of overthinking can be effectively captured by a single direction in the model's activation space and the issue can be eased by intervening the activations along this direction. However, this efficacy soon reaches a plateau and even deteriorates as the intervention strength increases. We therefore systematically explore the activation space and find that the overthinking phenomenon is actually tied to a low-dimensional manifold, which indicates that the limited effect stems from the noises introduced by the high-dimensional steering direction. Based on this insight, we propose Manifold Steering, a novel approach that elegantly projects the steering direction onto the low-dimensional activation manifold given the theoretical approximation of the interference noise. Extensive experiments on DeepSeek-R1 distilled models validate that our method reduces output tokens by up to 71% while maintaining and even improving the accuracy on several mathematical benchmarks. Our method also exhibits robust cross-domain transferability, delivering consistent token reduction performance in code generation and knowledge-based QA tasks. Code is available at: this https URL. 

---
# Physics-Informed Distillation of Diffusion Models for PDE-Constrained Generation 

**Authors**: Yi Zhang, Difan Zou  

**Link**: [PDF](https://arxiv.org/pdf/2505.22391)  

**Abstract**: Modeling physical systems in a generative manner offers several advantages, including the ability to handle partial observations, generate diverse solutions, and address both forward and inverse problems. Recently, diffusion models have gained increasing attention in the modeling of physical systems, particularly those governed by partial differential equations (PDEs). However, diffusion models only access noisy data $\boldsymbol{x}_t$ at intermediate steps, making it infeasible to directly enforce constraints on the clean sample $\boldsymbol{x}_0$ at each noisy level. As a workaround, constraints are typically applied to the expectation of clean samples $\mathbb{E}[\boldsymbol{x}_0|\boldsymbol{x}_t]$, which is estimated using the learned score network. However, imposing PDE constraints on the expectation does not strictly represent the one on the true clean data, known as Jensen's Gap. This gap creates a trade-off: enforcing PDE constraints may come at the cost of reduced accuracy in generative modeling. To address this, we propose a simple yet effective post-hoc distillation approach, where PDE constraints are not injected directly into the diffusion process, but instead enforced during a post-hoc distillation stage. We term our method as Physics-Informed Distillation of Diffusion Models (PIDDM). This distillation not only facilitates single-step generation with improved PDE satisfaction, but also support both forward and inverse problem solving and reconstruction from randomly partial observation. Extensive experiments across various PDE benchmarks demonstrate that PIDDM significantly improves PDE satisfaction over several recent and competitive baselines, such as PIDM, DiffusionPDE, and ECI-sampling, with less computation overhead. Our approach can shed light on more efficient and effective strategies for incorporating physical constraints into diffusion models. 

---
# Train with Perturbation, Infer after Merging: A Two-Stage Framework for Continual Learning 

**Authors**: Haomiao Qiu, Miao Zhang, Ziyue Qiao, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.22389)  

**Abstract**: Continual Learning (CL) aims to enable models to continuously acquire new knowledge from a sequence of tasks with avoiding the forgetting of learned information. However, existing CL methods only rely on the parameters of the most recent task for inference, which makes them susceptible to catastrophic forgetting. Inspired by the recent success of model merging techniques, we propose \textbf{Perturb-and-Merge (P\&M)}, a novel continual learning framework that integrates model merging into the CL paradigm to mitigate forgetting. Specifically, after training on each task, P\&M constructs a new model by forming a convex combination of the previous model and the newly trained task-specific model. Through theoretical analysis, we minimize the total loss increase across all tasks and derive an analytical solution for the optimal merging coefficient. To further improve the performance of the merged model, we observe that the degradation introduced during merging can be alleviated by a regularization term composed of the task vector and the Hessian matrix of the loss function. Interestingly, we show that this term can be efficiently approximated using second-order symmetric finite differences, and a stochastic perturbation strategy along the task vector direction is accordingly devised which incurs no additional forward or backward passes while providing an effective approximation of the regularization term. Finally, we combine P\&M with LoRA, a parameter-efficient fine-tuning method, to reduce memory overhead. Our proposed approach achieves state-of-the-art performance on several continual learning benchmark datasets. 

---
# DAM: Domain-Aware Module for Multi-Domain Dataset Condensation 

**Authors**: Jaehyun Choi, Gyojin Han, Dong-Jae Lee, Sunghyun Baek, Junmo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.22387)  

**Abstract**: Dataset Condensation (DC) has emerged as a promising solution to mitigate the computational and storage burdens associated with training deep learning models. However, existing DC methods largely overlook the multi-domain nature of modern datasets, which are increasingly composed of heterogeneous images spanning multiple domains. In this paper, we extend DC and introduce Multi-Domain Dataset Condensation (MDDC), which aims to condense data that generalizes across both single-domain and multi-domain settings. To this end, we propose the Domain-Aware Module (DAM), a training-time module that embeds domain-related features into each synthetic image via learnable spatial masks. As explicit domain labels are mostly unavailable in real-world datasets, we employ frequency-based pseudo-domain labeling, which leverages low-frequency amplitude statistics. DAM is only active during the condensation process, thus preserving the same images per class (IPC) with prior methods. Experiments show that DAM consistently improves in-domain, out-of-domain, and cross-architecture performance over baseline dataset condensation methods. 

---
# Exact Algorithms and Lower Bounds for Forming Coalitions of Constrained Maximum Size 

**Authors**: Foivos Fioravantes, Harmender Gahlawat, Nikolaos Melissinos  

**Link**: [PDF](https://arxiv.org/pdf/2505.22384)  

**Abstract**: Imagine we want to split a group of agents into teams in the most \emph{efficient} way, considering that each agent has their own preferences about their teammates. This scenario is modeled by the extensively studied \textsc{Coalition Formation} problem. Here, we study a version of this problem where each team must additionally be of bounded size.
We conduct a systematic algorithmic study, providing several intractability results as well as multiple exact algorithms that scale well as the input grows (FPT), which could prove useful in practice.
Our main contribution is an algorithm that deals efficiently with tree-like structures (bounded \emph{treewidth}) for ``small'' teams. We complement this result by proving that our algorithm is asymptotically optimal. Particularly, there can be no algorithm that vastly outperforms the one we present, under reasonable theoretical assumptions, even when considering star-like structures (bounded \emph{vertex cover number}). 

---
# SplitLoRA: Balancing Stability and Plasticity in Continual Learning Through Gradient Space Splitting 

**Authors**: Haomiao Qiu, Miao Zhang, Ziyue Qiao, Weili Guan, Min Zhang, Liqiang Nie  

**Link**: [PDF](https://arxiv.org/pdf/2505.22370)  

**Abstract**: Continual Learning requires a model to learn multiple tasks in sequence while maintaining both stability:preserving knowledge from previously learned tasks, and plasticity:effectively learning new tasks. Gradient projection has emerged as an effective and popular paradigm in CL, where it partitions the gradient space of previously learned tasks into two orthogonal subspaces: a primary subspace and a minor subspace. New tasks are learned effectively within the minor subspace, thereby reducing interference with previously acquired knowledge. However, existing Gradient Projection methods struggle to achieve an optimal balance between plasticity and stability, as it is hard to appropriately partition the gradient space. In this work, we consider a continual learning paradigm based on Low-Rank Adaptation, which has gained considerable attention due to its efficiency and wide applicability, and propose a novel approach for continual learning, called SplitLoRA. We first provide a theoretical analysis of how subspace partitioning affects model stability and plasticity. Informed by this analysis, we then introduce an effective method that derives the optimal partition of the gradient space for previously learned tasks. This approach effectively balances stability and plasticity in continual learning. Experimental results on multiple datasets demonstrate that the proposed method achieves state-of-the-art performance. 

---
# Budget-Adaptive Adapter Tuning in Orthogonal Subspaces for Continual Learning in LLMs 

**Authors**: Zhiyi Wan, Wanrou Du, Liang Li, Miao Pan, Xiaoqi Qin  

**Link**: [PDF](https://arxiv.org/pdf/2505.22358)  

**Abstract**: Large language models (LLMs) often suffer from catastrophic forgetting in continual learning (CL) scenarios, where performance on previously learned tasks degrades severely while training on sequentially arriving tasks. Although pioneering CL approaches using orthogonal subspaces can mitigate task interference, they typically employ fixed budget allocation, neglecting the varying complexity across tasks and layers. Besides, recent budget-adaptive tuning methods for LLMs often adopt multi-stage paradigms that decouple optimization and budget allocation. Such decoupling results in potential misalignment, which hinders those approaches' practical application in CL scenarios. To address these limitations, we propose OA-Adapter, a novel parameter-efficient approach for continual learning in LLMs that unifies dynamic budget adaptation with orthogonal subspace learning in a single end-to-end training stage. Specifically, OA-Adapter introduces a dynamic bottleneck dimension adaptation mechanism that simultaneously allocates an efficient parameter budget and optimizes task objectives without misalignment. To effectively preserve previously acquired knowledge while coordinating with the dynamic budget allocation, orthogonal constraints are applied specifically between the parameter subspace of the current task and the dynamically allocated parameter subspaces of historical tasks. Experimental results on continual learning benchmarks demonstrate that OA-Adapter outperforms state-of-the-art methods in both accuracy and parameter efficiency, achieving higher average accuracy while using 58.5% fewer parameters on the standard CL benchmark. 

---
# Suitability Filter: A Statistical Framework for Classifier Evaluation in Real-World Deployment Settings 

**Authors**: Angéline Pouget, Mohammad Yaghini, Stephan Rabanser, Nicolas Papernot  

**Link**: [PDF](https://arxiv.org/pdf/2505.22356)  

**Abstract**: Deploying machine learning models in safety-critical domains poses a key challenge: ensuring reliable model performance on downstream user data without access to ground truth labels for direct validation. We propose the suitability filter, a novel framework designed to detect performance deterioration by utilizing suitability signals -- model output features that are sensitive to covariate shifts and indicative of potential prediction errors. The suitability filter evaluates whether classifier accuracy on unlabeled user data shows significant degradation compared to the accuracy measured on the labeled test dataset. Specifically, it ensures that this degradation does not exceed a pre-specified margin, which represents the maximum acceptable drop in accuracy. To achieve reliable performance evaluation, we aggregate suitability signals for both test and user data and compare these empirical distributions using statistical hypothesis testing, thus providing insights into decision uncertainty. Our modular method adapts to various models and domains. Empirical evaluations across different classification tasks demonstrate that the suitability filter reliably detects performance deviations due to covariate shift. This enables proactive mitigation of potential failures in high-stakes applications. 

---
# VME: A Satellite Imagery Dataset and Benchmark for Detecting Vehicles in the Middle East and Beyond 

**Authors**: Noora Al-Emadi, Ingmar Weber, Yin Yang, Ferda Ofli  

**Link**: [PDF](https://arxiv.org/pdf/2505.22353)  

**Abstract**: Detecting vehicles in satellite images is crucial for traffic management, urban planning, and disaster response. However, current models struggle with real-world diversity, particularly across different regions. This challenge is amplified by geographic bias in existing datasets, which often focus on specific areas and overlook regions like the Middle East. To address this gap, we present the Vehicles in the Middle East (VME) dataset, designed explicitly for vehicle detection in high-resolution satellite images from Middle Eastern countries. Sourced from Maxar, the VME dataset spans 54 cities across 12 countries, comprising over 4,000 image tiles and more than 100,000 vehicles, annotated using both manual and semi-automated methods. Additionally, we introduce the largest benchmark dataset for Car Detection in Satellite Imagery (CDSI), combining images from multiple sources to enhance global car detection. Our experiments demonstrate that models trained on existing datasets perform poorly on Middle Eastern images, while the VME dataset significantly improves detection accuracy in this region. Moreover, state-of-the-art models trained on CDSI achieve substantial improvements in global car detection. 

---
# ChatPD: An LLM-driven Paper-Dataset Networking System 

**Authors**: Anjie Xu, Ruiqing Ding, Leye Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22349)  

**Abstract**: Scientific research heavily depends on suitable datasets for method validation, but existing academic platforms with dataset management like PapersWithCode suffer from inefficiencies in their manual workflow. To overcome this bottleneck, we present a system, called ChatPD, that utilizes Large Language Models (LLMs) to automate dataset information extraction from academic papers and construct a structured paper-dataset network. Our system consists of three key modules: \textit{paper collection}, \textit{dataset information extraction}, and \textit{dataset entity resolution} to construct paper-dataset networks. Specifically, we propose a \textit{Graph Completion and Inference} strategy to map dataset descriptions to their corresponding entities. Through extensive experiments, we demonstrate that ChatPD not only outperforms the existing platform PapersWithCode in dataset usage extraction but also achieves about 90\% precision and recall in entity resolution tasks. Moreover, we have deployed ChatPD to continuously extract which datasets are used in papers, and provide a dataset discovery service, such as task-specific dataset queries and similar dataset recommendations. We open source ChatPD and the current paper-dataset network on this [GitHub repository]{this https URL}. 

---
# Empowering Intelligent Low-altitude Economy with Large AI Model Deployment 

**Authors**: Zhonghao Lyu, Yulan Gao, Junting Chen, Hongyang Du, Jie Xu, Kaibin Huang, Dong In Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.22343)  

**Abstract**: Low-altitude economy (LAE) represents an emerging economic paradigm that redefines commercial and social aerial activities. Large artificial intelligence models (LAIMs) offer transformative potential to further enhance the intelligence of LAE services. However, deploying LAIMs in LAE poses several challenges, including the significant gap between their computational/storage demands and the limited onboard resources of LAE entities, the mismatch between lab-trained LAIMs and dynamic physical environments, and the inefficiencies of traditional decoupled designs for sensing, communication, and computation. To address these issues, we first propose a hierarchical system architecture tailored for LAIM deployment and present representative LAE application scenarios. Next, we explore key enabling techniques that facilitate the mutual co-evolution of LAIMs and low-altitude systems, and introduce a task-oriented execution pipeline for scalable and adaptive service delivery. Then, the proposed framework is validated through real-world case studies. Finally, we outline open challenges to inspire future research. 

---
# Text2Grad: Reinforcement Learning from Natural Language Feedback 

**Authors**: Hanyang Wang, Lu Wang, Chaoyun Zhang, Tianjun Mao, Si Qin, Qingwei Lin, Saravan Rajmohan, Dongmei Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22338)  

**Abstract**: Traditional RLHF optimizes language models with coarse, scalar rewards that mask the fine-grained reasons behind success or failure, leading to slow and opaque learning. Recent work augments RL with textual critiques through prompting or reflection, improving interpretability but leaving model parameters untouched. We introduce Text2Grad, a reinforcement-learning paradigm that turns free-form textual feedback into span-level gradients. Given human (or programmatic) critiques, Text2Grad aligns each feedback phrase with the relevant token spans, converts these alignments into differentiable reward signals, and performs gradient updates that directly refine the offending portions of the model's policy. This yields precise, feedback-conditioned adjustments instead of global nudges. Text2Grad is realized through three components: (1) a high-quality feedback-annotation pipeline that pairs critiques with token spans; (2) a fine-grained reward model that predicts span-level reward on answer while generating explanatory critiques; and (3) a span-level policy optimizer that back-propagates natural-language gradients. Across summarization, code generation, and question answering, Text2Grad consistently surpasses scalar-reward RL and prompt-only baselines, providing both higher task metrics and richer interpretability. Our results demonstrate that natural-language feedback, when converted to gradients, is a powerful signal for fine-grained policy optimization. The code for our method is available at this https URL 

---
# Advancing Multimodal Reasoning via Reinforcement Learning with Cold Start 

**Authors**: Lai Wei, Yuting Li, Kaipeng Zheng, Chen Wang, Yue Wang, Linghe Kong, Lichao Sun, Weiran Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22334)  

**Abstract**: Recent advancements in large language models (LLMs) have demonstrated impressive chain-of-thought reasoning capabilities, with reinforcement learning (RL) playing a crucial role in this progress. While "aha moment" patterns--where models exhibit self-correction through reflection--are often attributed to emergent properties from RL, we first demonstrate that these patterns exist in multimodal LLMs (MLLMs) prior to RL training but may not necessarily correlate with improved reasoning performance. Building on these insights, we present a comprehensive study on enhancing multimodal reasoning through a two-stage approach: (1) supervised fine-tuning (SFT) as a cold start with structured chain-of-thought reasoning patterns, followed by (2) reinforcement learning via GRPO to further refine these capabilities. Our extensive experiments show that this combined approach consistently outperforms both SFT-only and RL-only methods across challenging multimodal reasoning benchmarks. The resulting models achieve state-of-the-art performance among open-source MLLMs at both 3B and 7B scales, with our 7B model showing substantial improvements over base models (e.g., 66.3 %$\rightarrow$73.4 % on MathVista, 62.9 %$\rightarrow$70.4 % on We-Math) and our 3B model achieving performance competitive with several 7B models. Overall, this work provides practical guidance for building advanced multimodal reasoning models. Our code is available at this https URL. 

---
# Skywork Open Reasoner 1 Technical Report 

**Authors**: Jujie He, Jiacai Liu, Chris Yuhao Liu, Rui Yan, Chaojie Wang, Peng Cheng, Xiaoyu Zhang, Fuxiang Zhang, Jiacheng Xu, Wei Shen, Siyuan Li, Liang Zeng, Tianwen Wei, Cheng Cheng, Bo An, Yang Liu, Yahui Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.22312)  

**Abstract**: The success of DeepSeek-R1 underscores the significant role of reinforcement learning (RL) in enhancing the reasoning capabilities of large language models (LLMs). In this work, we present Skywork-OR1, an effective and scalable RL implementation for long Chain-of-Thought (CoT) models. Building on the DeepSeek-R1-Distill model series, our RL approach achieves notable performance gains, increasing average accuracy across AIME24, AIME25, and LiveCodeBench from 57.8% to 72.8% (+15.0%) for the 32B model and from 43.6% to 57.5% (+13.9%) for the 7B model. Our Skywork-OR1-32B model surpasses both DeepSeek-R1 and Qwen3-32B on the AIME24 and AIME25 benchmarks, while achieving comparable results on LiveCodeBench. The Skywork-OR1-7B and Skywork-OR1-Math-7B models demonstrate competitive reasoning capabilities among models of similar size. We perform comprehensive ablation studies on the core components of our training pipeline to validate their effectiveness. Additionally, we thoroughly investigate the phenomenon of entropy collapse, identify key factors affecting entropy dynamics, and demonstrate that mitigating premature entropy collapse is critical for improved test performance. To support community research, we fully open-source our model weights, training code, and training datasets. 

---
# From Dormant to Deleted: Tamper-Resistant Unlearning Through Weight-Space Regularization 

**Authors**: Shoaib Ahmed Siddiqui, Adrian Weller, David Krueger, Gintare Karolina Dziugaite, Michael Curtis Mozer, Eleni Triantafillou  

**Link**: [PDF](https://arxiv.org/pdf/2505.22310)  

**Abstract**: Recent unlearning methods for LLMs are vulnerable to relearning attacks: knowledge believed-to-be-unlearned re-emerges by fine-tuning on a small set of (even seemingly-unrelated) examples. We study this phenomenon in a controlled setting for example-level unlearning in vision classifiers. We make the surprising discovery that forget-set accuracy can recover from around 50% post-unlearning to nearly 100% with fine-tuning on just the retain set -- i.e., zero examples of the forget set. We observe this effect across a wide variety of unlearning methods, whereas for a model retrained from scratch excluding the forget set (gold standard), the accuracy remains at 50%. We observe that resistance to relearning attacks can be predicted by weight-space properties, specifically, $L_2$-distance and linear mode connectivity between the original and the unlearned model. Leveraging this insight, we propose a new class of methods that achieve state-of-the-art resistance to relearning attacks. 

---
# Versatile Cardiovascular Signal Generation with a Unified Diffusion Transformer 

**Authors**: Zehua Chen, Yuyang Miao, Liyuan Wang, Luyun Fan, Danilo P. Mandic, Jun Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22306)  

**Abstract**: Cardiovascular signals such as photoplethysmography (PPG), electrocardiography (ECG), and blood pressure (BP) are inherently correlated and complementary, together reflecting the health of cardiovascular system. However, their joint utilization in real-time monitoring is severely limited by diverse acquisition challenges from noisy wearable recordings to burdened invasive procedures. Here we propose UniCardio, a multi-modal diffusion transformer that reconstructs low-quality signals and synthesizes unrecorded signals in a unified generative framework. Its key innovations include a specialized model architecture to manage the signal modalities involved in generation tasks and a continual learning paradigm to incorporate varying modality combinations. By exploiting the complementary nature of cardiovascular signals, UniCardio clearly outperforms recent task-specific baselines in signal denoising, imputation, and translation. The generated signals match the performance of ground-truth signals in detecting abnormal health conditions and estimating vital signs, even in unseen domains, while ensuring interpretability for human experts. These advantages position UniCardio as a promising avenue for advancing AI-assisted healthcare. 

---
# Voice CMS: updating the knowledge base of a digital assistant through conversation 

**Authors**: Grzegorz Wolny, Michał Szczerbak  

**Link**: [PDF](https://arxiv.org/pdf/2505.22303)  

**Abstract**: In this study, we propose a solution based on a multi-agent LLM architecture and a voice user interface (VUI) designed to update the knowledge base of a digital assistant. Its usability is evaluated in comparison to a more traditional graphical content management system (CMS), with a focus on understanding the relationship between user preferences and the complexity of the information being provided. The findings demonstrate that, while the overall usability of the VUI is rated lower than the graphical interface, it is already preferred by users for less complex tasks. Furthermore, the quality of content entered through the VUI is comparable to that achieved with the graphical interface, even for highly complex tasks. Obtained qualitative results suggest that a hybrid interface combining the strengths of both approaches could address the key challenges identified during the experiment, such as reducing cognitive load through graphical feedback while maintaining the intuitive nature of voice-based interactions. This work highlights the potential of conversational interfaces as a viable and effective method for knowledge management in specific business contexts. 

---
# Neural Restoration of Greening Defects in Historical Autochrome Photographs Based on Purely Synthetic Data 

**Authors**: Saptarshi Neil Sinha, P. Julius Kuehn, Johannes Koppe, Arjan Kuijper, Michael Weinmann  

**Link**: [PDF](https://arxiv.org/pdf/2505.22291)  

**Abstract**: The preservation of early visual arts, particularly color photographs, is challenged by deterioration caused by aging and improper storage, leading to issues like blurring, scratches, color bleeding, and fading defects. In this paper, we present the first approach for the automatic removal of greening color defects in digitized autochrome photographs. Our main contributions include a method based on synthetic dataset generation and the use of generative AI with a carefully designed loss function for the restoration of visual arts. To address the lack of suitable training datasets for analyzing greening defects in damaged autochromes, we introduce a novel approach for accurately simulating such defects in synthetic data. We also propose a modified weighted loss function for the ChaIR method to account for color imbalances between defected and non-defected areas. While existing methods struggle with accurately reproducing original colors and may require significant manual effort, our method allows for efficient restoration with reduced time requirements. 

---
# New Tools are Needed for Tracking Adherence to AI Model Behavioral Use Clauses 

**Authors**: Daniel McDuff, Tim Korjakow, Kevin Klyman, Danish Contractor  

**Link**: [PDF](https://arxiv.org/pdf/2505.22287)  

**Abstract**: Foundation models have had a transformative impact on AI. A combination of large investments in research and development, growing sources of digital data for training, and architectures that scale with data and compute has led to models with powerful capabilities. Releasing assets is fundamental to scientific advancement and commercial enterprise. However, concerns over negligent or malicious uses of AI have led to the design of mechanisms to limit the risks of the technology. The result has been a proliferation of licenses with behavioral-use clauses and acceptable-use-policies that are increasingly being adopted by commonly used families of models (Llama, Gemma, Deepseek) and a myriad of smaller projects. We created and deployed a custom AI licenses generator to facilitate license creation and have quantitatively and qualitatively analyzed over 300 customized licenses created with this tool. Alongside this we analyzed 1.7 million models licenses on the HuggingFace model hub. Our results show increasing adoption of these licenses, interest in tools that support their creation and a convergence on common clause configurations. In this paper we take the position that tools for tracking adoption of, and adherence to, these licenses is the natural next step and urgently needed in order to ensure they have the desired impact of ensuring responsible use. 

---
# Natural Language Processing in Support of Evidence-based Medicine: A Scoping Review 

**Authors**: Zihan Xu, Haotian Ma, Gongbo Zhang, Yihao Ding, Chunhua Weng, Yifan Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.22280)  

**Abstract**: Evidence-based medicine (EBM) is at the forefront of modern healthcare, emphasizing the use of the best available scientific evidence to guide clinical decisions. Due to the sheer volume and rapid growth of medical literature and the high cost of curation, there is a critical need to investigate Natural Language Processing (NLP) methods to identify, appraise, synthesize, summarize, and disseminate evidence in EBM. This survey presents an in-depth review of 129 research studies on leveraging NLP for EBM, illustrating its pivotal role in enhancing clinical decision-making processes. The paper systematically explores how NLP supports the five fundamental steps of EBM -- Ask, Acquire, Appraise, Apply, and Assess. The review not only identifies current limitations within the field but also proposes directions for future research, emphasizing the potential for NLP to revolutionize EBM by refining evidence extraction, evidence synthesis, appraisal, summarization, enhancing data comprehensibility, and facilitating a more efficient clinical workflow. 

---
# Test-Time Immunization: A Universal Defense Framework Against Jailbreaks for (Multimodal) Large Language Models 

**Authors**: Yongcan Yu, Yanbo Wang, Ran He, Jian Liang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22271)  

**Abstract**: While (multimodal) large language models (LLMs) have attracted widespread attention due to their exceptional capabilities, they remain vulnerable to jailbreak attacks. Various defense methods are proposed to defend against jailbreak attacks, however, they are often tailored to specific types of jailbreak attacks, limiting their effectiveness against diverse adversarial strategies. For instance, rephrasing-based defenses are effective against text adversarial jailbreaks but fail to counteract image-based attacks. To overcome these limitations, we propose a universal defense framework, termed Test-time IMmunization (TIM), which can adaptively defend against various jailbreak attacks in a self-evolving way. Specifically, TIM initially trains a gist token for efficient detection, which it subsequently applies to detect jailbreak activities during inference. When jailbreak attempts are identified, TIM implements safety fine-tuning using the detected jailbreak instructions paired with refusal answers. Furthermore, to mitigate potential performance degradation in the detector caused by parameter updates during safety fine-tuning, we decouple the fine-tuning process from the detection module. Extensive experiments on both LLMs and multimodal LLMs demonstrate the efficacy of TIM. 

---
# MRT at SemEval-2025 Task 8: Maximizing Recovery from Tables with Multiple Steps 

**Authors**: Maximiliano Hormazábal Lagos, Álvaro Bueno Saez, Héctor Cerezo-Costas, Pedro Alonso Doval, Jorge Alcalde Vesteiro  

**Link**: [PDF](https://arxiv.org/pdf/2505.22264)  

**Abstract**: In this paper we expose our approach to solve the \textit{SemEval 2025 Task 8: Question-Answering over Tabular Data} challenge. Our strategy leverages Python code generation with LLMs to interact with the table and get the answer to the questions. The process is composed of multiple steps: understanding the content of the table, generating natural language instructions in the form of steps to follow in order to get the answer, translating these instructions to code, running it and handling potential errors or exceptions. These steps use open source LLMs and fine grained optimized prompts for each task (step). With this approach, we achieved a score of $70.50\%$ for subtask 1. 

---
# Judging Quality Across Languages: A Multilingual Approach to Pretraining Data Filtering with Language Models 

**Authors**: Mehdi Ali, Manuel Brack, Max Lübbering, Elias Wendt, Abbas Goher Khan, Richard Rutmann, Alex Jude, Maurice Kraus, Alexander Arno Weber, Felix Stollenwerk, David Kaczér, Florian Mai, Lucie Flek, Rafet Sifa, Nicolas Flores-Herr, Joachim Köhler, Patrick Schramowski, Michael Fromm, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2505.22232)  

**Abstract**: High-quality multilingual training data is essential for effectively pretraining large language models (LLMs). Yet, the availability of suitable open-source multilingual datasets remains limited. Existing state-of-the-art datasets mostly rely on heuristic filtering methods, restricting both their cross-lingual transferability and scalability. Here, we introduce JQL, a systematic approach that efficiently curates diverse and high-quality multilingual data at scale while significantly reducing computational demands. JQL distills LLMs' annotation capabilities into lightweight annotators based on pretrained multilingual embeddings. These models exhibit robust multilingual and cross-lingual performance, even for languages and scripts unseen during training. Evaluated empirically across 35 languages, the resulting annotation pipeline substantially outperforms current heuristic filtering methods like Fineweb2. JQL notably enhances downstream model training quality and increases data retention rates. Our research provides practical insights and valuable resources for multilingual data curation, raising the standards of multilingual dataset development. 

---
# Solver-Free Decision-Focused Learning for Linear Optimization Problems 

**Authors**: Senne Berden, Ali İrfan Mahmutoğulları, Dimos Tsouros, Tias Guns  

**Link**: [PDF](https://arxiv.org/pdf/2505.22224)  

**Abstract**: Mathematical optimization is a fundamental tool for decision-making in a wide range of applications. However, in many real-world scenarios, the parameters of the optimization problem are not known a priori and must be predicted from contextual features. This gives rise to predict-then-optimize problems, where a machine learning model predicts problem parameters that are then used to make decisions via optimization. A growing body of work on decision-focused learning (DFL) addresses this setting by training models specifically to produce predictions that maximize downstream decision quality, rather than accuracy. While effective, DFL is computationally expensive, because it requires solving the optimization problem with the predicted parameters at each loss evaluation. In this work, we address this computational bottleneck for linear optimization problems, a common class of problems in both DFL literature and real-world applications. We propose a solver-free training method that exploits the geometric structure of linear optimization to enable efficient training with minimal degradation in solution quality. Our method is based on the insight that a solution is optimal if and only if it achieves an objective value that is at least as good as that of its adjacent vertices on the feasible polytope. Building on this, our method compares the estimated quality of the ground-truth optimal solution with that of its precomputed adjacent vertices, and uses this as loss function. Experiments demonstrate that our method significantly reduces computational cost while maintaining high decision quality. 

---
# Pitfalls of Rule- and Model-based Verifiers -- A Case Study on Mathematical Reasoning 

**Authors**: Yuzhen Huang, Weihao Zeng, Xingshan Zeng, Qi Zhu, Junxian He  

**Link**: [PDF](https://arxiv.org/pdf/2505.22203)  

**Abstract**: Trustworthy verifiers are essential for the success of reinforcement learning with verifiable reward (RLVR), which is the core methodology behind various large reasoning models such as DeepSeek-R1. In complex domains like mathematical reasoning, rule-based verifiers have been widely adopted in previous works to train strong reasoning models. However, the reliability of these verifiers and their impact on the RL training process remain poorly understood. In this work, we take mathematical reasoning as a case study and conduct a comprehensive analysis of various verifiers in both static evaluation and RL training scenarios. First, we find that current open-source rule-based verifiers often fail to recognize equivalent answers presented in different formats across multiple commonly used mathematical datasets, resulting in non-negligible false negative rates. This limitation adversely affects RL training performance and becomes more pronounced as the policy model gets stronger. Subsequently, we investigate model-based verifiers as a potential solution to address these limitations. While the static evaluation shows that model-based verifiers achieve significantly higher verification accuracy, further analysis and RL training results imply that they are highly susceptible to hacking, where they misclassify certain patterns in responses as correct (i.e., false positives). This vulnerability is exploited during policy model optimization, leading to artificially inflated rewards. Our findings underscore the unique risks inherent to both rule-based and model-based verifiers, aiming to offer valuable insights to develop more robust reward systems in reinforcement learning. 

---
# Let's Predict Sentence by Sentence 

**Authors**: Hyeonbin Hwang, Byeongguk Jeon, Seungone Kim, Jiyeon Kim, Hoyeon Chang, Sohee Yang, Seungpil Won, Dohaeng Lee, Youbin Ahn, Minjoon Seo  

**Link**: [PDF](https://arxiv.org/pdf/2505.22202)  

**Abstract**: Autoregressive language models (LMs) generate one token at a time, yet human reasoning operates over higher-level abstractions - sentences, propositions, and concepts. This contrast raises a central question- Can LMs likewise learn to reason over structured semantic units rather than raw token sequences? In this work, we investigate whether pretrained LMs can be lifted into such abstract reasoning spaces by building on their learned representations. We present a framework that adapts a pretrained token-level LM to operate in sentence space by autoregressively predicting continuous embeddings of next sentences. We explore two embedding paradigms inspired by classical representation learning: 1) semantic embeddings, learned via autoencoding to preserve surface meaning; and 2) contextual embeddings, trained via next-sentence prediction to encode anticipatory structure. We evaluate both under two inference regimes: Discretized, which decodes each predicted embedding into text before re-encoding; and Continuous, which reasons entirely in embedding space for improved efficiency. Across four domains - mathematics, logic, commonsense, and planning - contextual embeddings under continuous inference show competitive performance with Chain-of-Thought (CoT) while reducing inference-time FLOPs on average by half. We also present early signs of scalability and modular adaptation. Finally, to visualize latent trajectories, we introduce SentenceLens, a diagnostic tool that decodes intermediate model states into interpretable sentences. Together, our results indicate that pretrained LMs can effectively transition to abstract, structured reasoning within latent embedding spaces. 

---
# Investigating Mechanisms for In-Context Vision Language Binding 

**Authors**: Darshana Saravanan, Makarand Tapaswi, Vineet Gandhi  

**Link**: [PDF](https://arxiv.org/pdf/2505.22200)  

**Abstract**: To understand a prompt, Vision-Language models (VLMs) must perceive the image, comprehend the text, and build associations within and across both modalities. For instance, given an 'image of a red toy car', the model should associate this image to phrases like 'car', 'red toy', 'red object', etc. Feng and Steinhardt propose the Binding ID mechanism in LLMs, suggesting that the entity and its corresponding attribute tokens share a Binding ID in the model activations. We investigate this for image-text binding in VLMs using a synthetic dataset and task that requires models to associate 3D objects in an image with their descriptions in the text. Our experiments demonstrate that VLMs assign a distinct Binding ID to an object's image tokens and its textual references, enabling in-context association. 

---
# Enhancing Uncertainty Estimation and Interpretability via Bayesian Non-negative Decision Layer 

**Authors**: Xinyue Hu, Zhibin Duan, Bo Chen, Mingyuan Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.22199)  

**Abstract**: Although deep neural networks have demonstrated significant success due to their powerful expressiveness, most models struggle to meet practical requirements for uncertainty estimation. Concurrently, the entangled nature of deep neural networks leads to a multifaceted problem, where various localized explanation techniques reveal that multiple unrelated features influence the decisions, thereby undermining interpretability. To address these challenges, we develop a Bayesian Non-negative Decision Layer (BNDL), which reformulates deep neural networks as a conditional Bayesian non-negative factor analysis. By leveraging stochastic latent variables, the BNDL can model complex dependencies and provide robust uncertainty estimation. Moreover, the sparsity and non-negativity of the latent variables encourage the model to learn disentangled representations and decision layers, thereby improving interpretability. We also offer theoretical guarantees that BNDL can achieve effective disentangled learning. In addition, we developed a corresponding variational inference method utilizing a Weibull variational inference network to approximate the posterior distribution of the latent variables. Our experimental results demonstrate that with enhanced disentanglement capabilities, BNDL not only improves the model's accuracy but also provides reliable uncertainty estimation and improved interpretability. 

---
# Physics-inspired Generative AI models via real hardware-based noisy quantum diffusion 

**Authors**: Marco Parigi, Stefano Martina, Francesco Aldo Venturelli, Filippo Caruso  

**Link**: [PDF](https://arxiv.org/pdf/2505.22193)  

**Abstract**: Quantum Diffusion Models (QDMs) are an emerging paradigm in Generative AI that aims to use quantum properties to improve the performances of their classical counterparts. However, existing algorithms are not easily scalable due to the limitations of near-term quantum devices. Following our previous work on QDMs, here we propose and implement two physics-inspired protocols. In the first, we use the formalism of quantum stochastic walks, showing that a specific interplay of quantum and classical dynamics in the forward process produces statistically more robust models generating sets of MNIST images with lower Fréchet Inception Distance (FID) than using totally classical dynamics. In the second approach, we realize an algorithm to generate images by exploiting the intrinsic noise of real IBM quantum hardware with only four qubits. Our work could be a starting point to pave the way for new scenarios for large-scale algorithms in quantum Generative AI, where quantum noise is neither mitigated nor corrected, but instead exploited as a useful resource. 

---
# Breaking the Cloak! Unveiling Chinese Cloaked Toxicity with Homophone Graph and Toxic Lexicon 

**Authors**: Xuchen Ma, Jianxiang Yu, Wenming Shao, Bo Pang, Xiang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.22184)  

**Abstract**: Social media platforms have experienced a significant rise in toxic content, including abusive language and discriminatory remarks, presenting growing challenges for content moderation. Some users evade censorship by deliberately disguising toxic words through homophonic cloak, which necessitates the task of unveiling cloaked toxicity. Existing methods are mostly designed for English texts, while Chinese cloaked toxicity unveiling has not been solved yet. To tackle the issue, we propose C$^2$TU, a novel training-free and prompt-free method for Chinese cloaked toxic content unveiling. It first employs substring matching to identify candidate toxic words based on Chinese homo-graph and toxic lexicon. Then it filters those candidates that are non-toxic and corrects cloaks to be their corresponding toxicities. Specifically, we develop two model variants for filtering, which are based on BERT and LLMs, respectively. For LLMs, we address the auto-regressive limitation in computing word occurrence probability and utilize the full semantic contexts of a text sequence to reveal cloaked toxic words. Extensive experiments demonstrate that C$^2$TU can achieve superior performance on two Chinese toxic datasets. In particular, our method outperforms the best competitor by up to 71% on the F1 score and 35% on accuracy, respectively. 

---
# Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design 

**Authors**: Yudi Zhang, Weilin Zhao, Xu Han, Tiejun Zhao, Wang Xu, Hailong Cao, Conghui Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22179)  

**Abstract**: Speculative decoding and quantization effectively accelerate memory-bound inference of large language models. Speculative decoding mitigates the memory bandwidth bottleneck by verifying multiple tokens within a single forward pass, which increases computational effort. Quantization achieves this optimization by compressing weights and activations into lower bit-widths and also reduces computations via low-bit matrix multiplications. To further leverage their strengths, we investigate the integration of these two techniques. Surprisingly, experiments applying the advanced speculative decoding method EAGLE-2 to various quantized models reveal that the memory benefits from 4-bit weight quantization are diminished by the computational load from speculative decoding. Specifically, verifying a tree-style draft incurs significantly more time overhead than a single-token forward pass on 4-bit weight quantized models. This finding led to our new speculative decoding design: a hierarchical framework that employs a small model as an intermediate stage to turn tree-style drafts into sequence drafts, leveraging the memory access benefits of the target quantized model. Experimental results show that our hierarchical approach achieves a 2.78$\times$ speedup across various tasks for the 4-bit weight Llama-3-70B model on an A100 GPU, outperforming EAGLE-2 by 1.31$\times$. Code available at this https URL. 

---
# Online Fair Division for Personalized $2$-Value Instances 

**Authors**: Georgios Amanatidis, Alexandros Lolos, Evangelos Markakis, Victor Turmel  

**Link**: [PDF](https://arxiv.org/pdf/2505.22174)  

**Abstract**: We study an online fair division setting, where goods arrive one at a time and there is a fixed set of $n$ agents, each of whom has an additive valuation function over the goods. Once a good appears, the value each agent has for it is revealed and it must be allocated immediately and irrevocably to one of the agents. It is known that without any assumptions about the values being severely restricted or coming from a distribution, very strong impossibility results hold in this setting. To bypass the latter, we turn our attention to instances where the valuation functions are restricted. In particular, we study personalized $2$-value instances, where there are only two possible values each agent may have for each good, possibly different across agents, and we show how to obtain worst case guarantees with respect to well-known fairness notions, such as maximin share fairness and envy-freeness up to one (or two) good(s). We suggest a deterministic algorithm that maintains a $1/(2n-1)$-MMS allocation at every time step and show that this is the best possible any deterministic algorithm can achieve if one cares about every single time step; nevertheless, eventually the allocation constructed by our algorithm becomes a $1/4$-MMS allocation. To achieve this, the algorithm implicitly maintains a fragile system of priority levels for all agents. Further, we show that, by allowing some limited access to future information, it is possible to have stronger results with less involved approaches. By knowing the values of goods for $n-1$ time steps into the future, we design a matching-based algorithm that achieves an EF$1$ allocation every $n$ time steps, while always maintaining an EF$2$ allocation. Finally, we show that our results allow us to get the first nontrivial guarantees for additive instances in which the ratio of the maximum over the minimum value an agent has for a good is bounded. 

---
# Unifying Continuous and Discrete Text Diffusion with Non-simultaneous Diffusion Processes 

**Authors**: Bocheng Li, Zhujin Gao, Linli Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22165)  

**Abstract**: Diffusion models have emerged as a promising approach for text generation, with recent works falling into two main categories: discrete and continuous diffusion models. Discrete diffusion models apply token corruption independently using categorical distributions, allowing for different diffusion progress across tokens but lacking fine-grained control. Continuous diffusion models map tokens to continuous spaces and apply fine-grained noise, but the diffusion progress is uniform across tokens, limiting their ability to capture semantic nuances. To address these limitations, we propose \textbf{\underline{N}}on-simultan\textbf{\underline{e}}ous C\textbf{\underline{o}}ntinuous \textbf{\underline{Diff}}usion Models (NeoDiff), a novel diffusion model that integrates the strengths of both discrete and continuous approaches. NeoDiff introduces a Poisson diffusion process for the forward process, enabling a flexible and fine-grained noising paradigm, and employs a time predictor for the reverse process to adaptively modulate the denoising progress based on token semantics. Furthermore, NeoDiff utilizes an optimized schedule for inference to ensure more precise noise control and improved performance. Our approach unifies the theories of discrete and continuous diffusion models, offering a more principled and effective framework for text generation. Experimental results on several text generation tasks demonstrate NeoDiff's superior performance compared to baselines of non-autoregressive continuous and discrete diffusion models, iterative-based methods and autoregressive diffusion-based methods. These results highlight NeoDiff's potential as a powerful tool for generating high-quality text and advancing the field of diffusion-based text generation. 

---
# Flexible Tool Selection through Low-dimensional Attribute Alignment of Vision and Language 

**Authors**: Guangfu Hao, Haojie Wen, Liangxuna Guo, Yang Chen, Yanchao Bi, Shan Yu  

**Link**: [PDF](https://arxiv.org/pdf/2505.22146)  

**Abstract**: Flexible tool selection reflects a complex cognitive ability that distinguishes humans from other species, yet computational models that capture this ability remain underdeveloped. We developed a framework using low-dimensional attribute representations to bridge visual tool perception and linguistic task understanding. We constructed a comprehensive dataset (ToolNet) containing 115 common tools labeled with 13 carefully designed attributes spanning physical, functional, and psychological properties, paired with natural language scenarios describing tool usage. Visual encoders (ResNet or ViT) extract attributes from tool images while fine-tuned language models (GPT-2, LLaMA, DeepSeek) derive required attributes from task descriptions. Our approach achieves 74% accuracy in tool selection tasks-significantly outperforming direct tool matching (20%) and smaller multimodal models (21%-58%), while approaching performance of much larger models like GPT-4o (73%) with substantially fewer parameters. Ablation studies revealed that manipulation-related attributes (graspability, hand-relatedness, elongation) consistently prove most critical across modalities. This work provides a parameter-efficient, interpretable solution that mimics human-like tool cognition, advancing both cognitive science understanding and practical applications in tool selection tasks. 

---
# FaceEditTalker: Interactive Talking Head Generation with Facial Attribute Editing 

**Authors**: Guanwen Feng, Zhiyuan Ma, Yunan Li, Junwei Jing, Jiahao Yang, Qiguang Miao  

**Link**: [PDF](https://arxiv.org/pdf/2505.22141)  

**Abstract**: Recent advances in audio-driven talking head generation have achieved impressive results in lip synchronization and emotional expression. However, they largely overlook the crucial task of facial attribute editing. This capability is crucial for achieving deep personalization and expanding the range of practical applications, including user-tailored digital avatars, engaging online education content, and brand-specific digital customer service. In these key domains, the flexible adjustment of visual attributes-such as hairstyle, accessories, and subtle facial features is essential for aligning with user preferences, reflecting diverse brand identities, and adapting to varying contextual demands. In this paper, we present FaceEditTalker, a unified framework that enables controllable facial attribute manipulation while generating high-quality, audio-synchronized talking head videos. Our method consists of two key components: an image feature space editing module, which extracts semantic and detail features and allows flexible control over attributes like expression, hairstyle, and accessories; and an audio-driven video generation module, which fuses these edited features with audio-guided facial landmarks to drive a diffusion-based generator. This design ensures temporal coherence, visual fidelity, and identity preservation across frames. Extensive experiments on public datasets demonstrate that our method outperforms state-of-the-art approaches in lip-sync accuracy, video quality, and attribute controllability. Project page: this https URL 

---
# Limited Generalizability in Argument Mining: State-Of-The-Art Models Learn Datasets, Not Arguments 

**Authors**: Marc Feger, Katarina Boland, Stefan Dietze  

**Link**: [PDF](https://arxiv.org/pdf/2505.22137)  

**Abstract**: Identifying arguments is a necessary prerequisite for various tasks in automated discourse analysis, particularly within contexts such as political debates, online discussions, and scientific reasoning. In addition to theoretical advances in understanding the constitution of arguments, a significant body of research has emerged around practical argument mining, supported by a growing number of publicly available datasets. On these benchmarks, BERT-like transformers have consistently performed best, reinforcing the belief that such models are broadly applicable across diverse contexts of debate. This study offers the first large-scale re-evaluation of such state-of-the-art models, with a specific focus on their ability to generalize in identifying arguments. We evaluate four transformers, three standard and one enhanced with contrastive pre-training for better generalization, on 17 English sentence-level datasets as most relevant to the task. Our findings show that, to varying degrees, these models tend to rely on lexical shortcuts tied to content words, suggesting that apparent progress may often be driven by dataset-specific cues rather than true task alignment. While the models achieve strong results on familiar benchmarks, their performance drops markedly when applied to unseen datasets. Nonetheless, incorporating both task-specific pre-training and joint benchmark training proves effective in enhancing both robustness and generalization. 

---
# Real-Time Blind Defocus Deblurring for Earth Observation: The IMAGIN-e Mission Approach 

**Authors**: Alejandro D. Mousist  

**Link**: [PDF](https://arxiv.org/pdf/2505.22128)  

**Abstract**: This work addresses mechanical defocus in Earth observation images from the IMAGIN-e mission aboard the ISS, proposing a blind deblurring approach adapted to space-based edge computing constraints. Leveraging Sentinel-2 data, our method estimates the defocus kernel and trains a restoration model within a GAN framework, effectively operating without reference images.
On Sentinel-2 images with synthetic degradation, SSIM improved by 72.47% and PSNR by 25.00%, confirming the model's ability to recover lost details when the original clean image is known. On IMAGIN-e, where no reference images exist, perceptual quality metrics indicate a substantial enhancement, with NIQE improving by 60.66% and BRISQUE by 48.38%, validating real-world onboard restoration. The approach is currently deployed aboard the IMAGIN-e mission, demonstrating its practical application in an operational space environment.
By efficiently handling high-resolution images under edge computing constraints, the method enables applications such as water body segmentation and contour detection while maintaining processing viability despite resource limitations. 

---
# SridBench: Benchmark of Scientific Research Illustration Drawing of Image Generation Model 

**Authors**: Yifan Chang, Yukang Feng, Jianwen Sun, Jiaxin Ai, Chuanhao Li, S. Kevin Zhou, Kaipeng Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22126)  

**Abstract**: Recent years have seen rapid advances in AI-driven image generation. Early diffusion models emphasized perceptual quality, while newer multimodal models like GPT-4o-image integrate high-level reasoning, improving semantic understanding and structural composition. Scientific illustration generation exemplifies this evolution: unlike general image synthesis, it demands accurate interpretation of technical content and transformation of abstract ideas into clear, standardized visuals. This task is significantly more knowledge-intensive and laborious, often requiring hours of manual work and specialized tools. Automating it in a controllable, intelligent manner would provide substantial practical value. Yet, no benchmark currently exists to evaluate AI on this front. To fill this gap, we introduce SridBench, the first benchmark for scientific figure generation. It comprises 1,120 instances curated from leading scientific papers across 13 natural and computer science disciplines, collected via human experts and MLLMs. Each sample is evaluated along six dimensions, including semantic fidelity and structural accuracy. Experimental results reveal that even top-tier models like GPT-4o-image lag behind human performance, with common issues in text/visual clarity and scientific correctness. These findings highlight the need for more advanced reasoning-driven visual generation capabilities. 

---
# Sentiment Simulation using Generative AI Agents 

**Authors**: Melrose Tia, Jezreel Sophia Lanuzo, Lei Rigi Baltazar, Marie Joy Lopez-Relente, Diwa Malaya Quiñones, Jason Albia  

**Link**: [PDF](https://arxiv.org/pdf/2505.22125)  

**Abstract**: Traditional sentiment analysis relies on surface-level linguistic patterns and retrospective data, limiting its ability to capture the psychological and contextual drivers of human sentiment. These limitations constrain its effectiveness in applications that require predictive insight, such as policy testing, narrative framing, and behavioral forecasting. We present a robust framework for sentiment simulation using generative AI agents embedded with psychologically rich profiles. Agents are instantiated from a nationally representative survey of 2,485 Filipino respondents, combining sociodemographic information with validated constructs of personality traits, values, beliefs, and socio-political attitudes. The framework includes three stages: (1) agent embodiment via categorical or contextualized encodings, (2) exposure to real-world political and economic scenarios, and (3) generation of sentiment ratings accompanied by explanatory rationales. Using Quadratic Weighted Accuracy (QWA), we evaluated alignment between agent-generated and human responses. Contextualized encoding achieved 92% alignment in replicating original survey responses. In sentiment simulation tasks, agents reached 81%--86% accuracy against ground truth sentiment, with contextualized profile encodings significantly outperforming categorical (p < 0.0001, Cohen's d = 0.70). Simulation results remained consistent across repeated trials (+/-0.2--0.5% SD) and resilient to variation in scenario framing (p = 0.9676, Cohen's d = 0.02). Our findings establish a scalable framework for sentiment modeling through psychographically grounded AI agents. This work signals a paradigm shift in sentiment analysis from retrospective classification to prospective and dynamic simulation grounded in psychology of sentiment formation. 

---
# Multimodal Forecasting of Sparse Intraoperative Hypotension Events Powered by Language Model 

**Authors**: Jintao Zhang, Zirui Liu, Mingyue Cheng, Shilong Zhang, Tingyue Pan, Qi Liu, Yanhu Xie  

**Link**: [PDF](https://arxiv.org/pdf/2505.22116)  

**Abstract**: Intraoperative hypotension (IOH) frequently occurs under general anesthesia and is strongly linked to adverse outcomes such as myocardial injury and increased mortality. Despite its significance, IOH prediction is hindered by event sparsity and the challenge of integrating static and dynamic data across diverse patients. In this paper, we propose \textbf{IOHFuseLM}, a multimodal language model framework. To accurately identify and differentiate sparse hypotensive events, we leverage a two-stage training strategy. The first stage involves domain adaptive pretraining on IOH physiological time series augmented through diffusion methods, thereby enhancing the model sensitivity to patterns associated with hypotension. Subsequently, task fine-tuning is performed on the original clinical dataset to further enhance the ability to distinguish normotensive from hypotensive states. To enable multimodal fusion for each patient, we align structured clinical descriptions with the corresponding physiological time series at the token level. Such alignment enables the model to capture individualized temporal patterns alongside their corresponding clinical semantics. In addition, we convert static patient attributes into structured text to enrich personalized information. Experimental evaluations on two intraoperative datasets demonstrate that IOHFuseLM outperforms established baselines in accurately identifying IOH events, highlighting its applicability in clinical decision support scenarios. Our code is publicly available to promote reproducibility at this https URL. 

---
# The quest for the GRAph Level autoEncoder (GRALE) 

**Authors**: Paul Krzakala, Gabriel Melo, Charlotte Laclau, Florence d'Alché-Buc, Rémi Flamary  

**Link**: [PDF](https://arxiv.org/pdf/2505.22109)  

**Abstract**: Although graph-based learning has attracted a lot of attention, graph representation learning is still a challenging task whose resolution may impact key application fields such as chemistry or biology. To this end, we introduce GRALE, a novel graph autoencoder that encodes and decodes graphs of varying sizes into a shared embedding space. GRALE is trained using an Optimal Transport-inspired loss that compares the original and reconstructed graphs and leverages a differentiable node matching module, which is trained jointly with the encoder and decoder. The proposed attention-based architecture relies on Evoformer, the core component of AlphaFold, which we extend to support both graph encoding and decoding. We show, in numerical experiments on simulated and molecular data, that GRALE enables a highly general form of pre-training, applicable to a wide range of downstream tasks, from classification and regression to more complex tasks such as graph interpolation, editing, matching, and prediction. 

---
# Inclusive, Differentially Private Federated Learning for Clinical Data 

**Authors**: Santhosh Parampottupadam, Melih Coşğun, Sarthak Pati, Maximilian Zenk, Saikat Roy, Dimitrios Bounias, Benjamin Hamm, Sinem Sav, Ralf Floca, Klaus Maier-Hein  

**Link**: [PDF](https://arxiv.org/pdf/2505.22108)  

**Abstract**: Federated Learning (FL) offers a promising approach for training clinical AI models without centralizing sensitive patient data. However, its real-world adoption is hindered by challenges related to privacy, resource constraints, and compliance. Existing Differential Privacy (DP) approaches often apply uniform noise, which disproportionately degrades model performance, even among well-compliant institutions. In this work, we propose a novel compliance-aware FL framework that enhances DP by adaptively adjusting noise based on quantifiable client compliance scores. Additionally, we introduce a compliance scoring tool based on key healthcare and security standards to promote secure, inclusive, and equitable participation across diverse clinical settings. Extensive experiments on public datasets demonstrate that integrating under-resourced, less compliant clinics with highly regulated institutions yields accuracy improvements of up to 15% over traditional FL. This work advances FL by balancing privacy, compliance, and performance, making it a viable solution for real-world clinical workflows in global healthcare. 

---
# AudioTurbo: Fast Text-to-Audio Generation with Rectified Diffusion 

**Authors**: Junqi Zhao, Jinzheng Zhao, Haohe Liu, Yun Chen, Lu Han, Xubo Liu, Mark Plumbley, Wenwu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22106)  

**Abstract**: Diffusion models have significantly improved the quality and diversity of audio generation but are hindered by slow inference speed. Rectified flow enhances inference speed by learning straight-line ordinary differential equation (ODE) paths. However, this approach requires training a flow-matching model from scratch and tends to perform suboptimally, or even poorly, at low step counts. To address the limitations of rectified flow while leveraging the advantages of advanced pre-trained diffusion models, this study integrates pre-trained models with the rectified diffusion method to improve the efficiency of text-to-audio (TTA) generation. Specifically, we propose AudioTurbo, which learns first-order ODE paths from deterministic noise sample pairs generated by a pre-trained TTA model. Experiments on the AudioCaps dataset demonstrate that our model, with only 10 sampling steps, outperforms prior models and reduces inference to 3 steps compared to a flow-matching-based acceleration model. 

---
# Knowledge Base Construction for Knowledge-Augmented Text-to-SQL 

**Authors**: Jinheon Baek, Horst Samulowitz, Oktie Hassanzadeh, Dharmashankar Subramanian, Sola Shirai, Alfio Gliozzo, Debarun Bhattacharjya  

**Link**: [PDF](https://arxiv.org/pdf/2505.22096)  

**Abstract**: Text-to-SQL aims to translate natural language queries into SQL statements, which is practical as it enables anyone to easily retrieve the desired information from databases. Recently, many existing approaches tackle this problem with Large Language Models (LLMs), leveraging their strong capability in understanding user queries and generating corresponding SQL code. Yet, the parametric knowledge in LLMs might be limited to covering all the diverse and domain-specific queries that require grounding in various database schemas, which makes generated SQLs less accurate oftentimes. To tackle this, we propose constructing the knowledge base for text-to-SQL, a foundational source of knowledge, from which we retrieve and generate the necessary knowledge for given queries. In particular, unlike existing approaches that either manually annotate knowledge or generate only a few pieces of knowledge for each query, our knowledge base is comprehensive, which is constructed based on a combination of all the available questions and their associated database schemas along with their relevant knowledge, and can be reused for unseen databases from different datasets and domains. We validate our approach on multiple text-to-SQL datasets, considering both the overlapping and non-overlapping database scenarios, where it outperforms relevant baselines substantially. 

---
# From Coders to Critics: Empowering Students through Peer Assessment in the Age of AI Copilots 

**Authors**: Santiago Berrezueta-Guzman, Stephan Krusche, Stefan Wagner  

**Link**: [PDF](https://arxiv.org/pdf/2505.22093)  

**Abstract**: The rapid adoption of AI powered coding assistants like ChatGPT and other coding copilots is transforming programming education, raising questions about assessment practices, academic integrity, and skill development. As educators seek alternatives to traditional grading methods susceptible to AI enabled plagiarism, structured peer assessment could be a promising strategy. This paper presents an empirical study of a rubric based, anonymized peer review process implemented in a large introductory programming course.
Students evaluated each other's final projects (2D game), and their assessments were compared to instructor grades using correlation, mean absolute error, and root mean square error (RMSE). Additionally, reflective surveys from 47 teams captured student perceptions of fairness, grading behavior, and preferences regarding grade aggregation. Results show that peer review can approximate instructor evaluation with moderate accuracy and foster student engagement, evaluative thinking, and interest in providing good feedback to their peers. We discuss these findings for designing scalable, trustworthy peer assessment systems to face the age of AI assisted coding. 

---
# iDSE: Navigating Design Space Exploration in High-Level Synthesis Using LLMs 

**Authors**: Runkai Li, Jia Xiong, Xi Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.22086)  

**Abstract**: High-Level Synthesis (HLS) serves as an agile hardware development tool that streamlines the circuit design by abstracting the register transfer level into behavioral descriptions, while allowing designers to customize the generated microarchitectures through optimization directives. However, the combinatorial explosion of possible directive configurations yields an intractable design space. Traditional design space exploration (DSE) methods, despite adopting heuristics or constructing predictive models to accelerate Pareto-optimal design acquisition, still suffer from prohibitive exploration costs and suboptimal results. Addressing these concerns, we introduce iDSE, the first LLM-aided DSE framework that leverages HLS design quality perception to effectively navigate the design space. iDSE intelligently pruns the design space to guide LLMs in calibrating representative initial sampling designs, expediting convergence toward the Pareto front. By exploiting the convergent and divergent thinking patterns inherent in LLMs for hardware optimization, iDSE achieves multi-path refinement of the design quality and diversity. Extensive experiments demonstrate that iDSE outperforms heuristic-based DSE methods by 5.1$\times$$\sim$16.6$\times$ in proximity to the reference Pareto front, matching NSGA-II with only 4.6% of the explored designs. Our work demonstrates the transformative potential of LLMs in scalable and efficient HLS design optimization, offering new insights into multiobjective optimization challenges. 

---
# The Resurrection of the ReLU 

**Authors**: Coşku Can Horuz, Geoffrey Kasenbacher, Saya Higuchi, Sebastian Kairat, Jendrik Stoltz, Moritz Pesl, Bernhard A. Moser, Christoph Linse, Thomas Martinetz, Sebastian Otte  

**Link**: [PDF](https://arxiv.org/pdf/2505.22074)  

**Abstract**: Modeling sophisticated activation functions within deep learning architectures has evolved into a distinct research direction. Functions such as GELU, SELU, and SiLU offer smooth gradients and improved convergence properties, making them popular choices in state-of-the-art models. Despite this trend, the classical ReLU remains appealing due to its simplicity, inherent sparsity, and other advantageous topological characteristics. However, ReLU units are prone to becoming irreversibly inactive - a phenomenon known as the dying ReLU problem - which limits their overall effectiveness. In this work, we introduce surrogate gradient learning for ReLU (SUGAR) as a novel, plug-and-play regularizer for deep architectures. SUGAR preserves the standard ReLU function during the forward pass but replaces its derivative in the backward pass with a smooth surrogate that avoids zeroing out gradients. We demonstrate that SUGAR, when paired with a well-chosen surrogate function, substantially enhances generalization performance over convolutional network architectures such as VGG-16 and ResNet-18, providing sparser activations while effectively resurrecting dead ReLUs. Moreover, we show that even in modern architectures like Conv2NeXt and Swin Transformer - which typically employ GELU - substituting these with SUGAR yields competitive and even slightly superior performance. These findings challenge the prevailing notion that advanced activation functions are necessary for optimal performance. Instead, they suggest that the conventional ReLU, particularly with appropriate gradient handling, can serve as a strong, versatile revived classic across a broad range of deep learning vision models. 

---
# Beyond path selection: Better LLMs for Scientific Information Extraction with MimicSFT and Relevance and Rule-induced(R$^2$)GRPO 

**Authors**: Ran Li, Shimin Di, Yuchen Liu, Chen Jing, Yu Qiu, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.22068)  

**Abstract**: Previous study suggest that powerful Large Language Models (LLMs) trained with Reinforcement Learning with Verifiable Rewards (RLVR) only refines reasoning path without improving the reasoning capacity in math tasks while supervised-finetuning(SFT) with distillation can. We study this from the view of Scientific information extraction (SciIE) where LLMs and reasoning LLMs underperforms small Bert-based models. SciIE require both the reasoning and memorization. We argue that both SFT and RLVR can refine the reasoning path and improve reasoning capacity in a simple way based on SciIE. We propose two-stage training with 1. MimicSFT, using structured reasoning templates without needing high-quality chain-of-thought data, 2. R$^2$GRPO with relevance and rule-induced rewards. Experiments on scientific IE benchmarks show that both methods can improve the reasoning capacity. R$^2$GRPO with mimicSFT surpasses baseline LLMs and specialized supervised models in relation extraction. Our code is available at this https URL. 

---
# From Failures to Fixes: LLM-Driven Scenario Repair for Self-Evolving Autonomous Driving 

**Authors**: Xinyu Xia, Xingjun Ma, Yunfeng Hu, Ting Qu, Hong Chen, Xun Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.22067)  

**Abstract**: Ensuring robust and generalizable autonomous driving requires not only broad scenario coverage but also efficient repair of failure cases, particularly those related to challenging and safety-critical scenarios. However, existing scenario generation and selection methods often lack adaptivity and semantic relevance, limiting their impact on performance improvement. In this paper, we propose \textbf{SERA}, an LLM-powered framework that enables autonomous driving systems to self-evolve by repairing failure cases through targeted scenario recommendation. By analyzing performance logs, SERA identifies failure patterns and dynamically retrieves semantically aligned scenarios from a structured bank. An LLM-based reflection mechanism further refines these recommendations to maximize relevance and diversity. The selected scenarios are used for few-shot fine-tuning, enabling targeted adaptation with minimal data. Experiments on the benchmark show that SERA consistently improves key metrics across multiple autonomous driving baselines, demonstrating its effectiveness and generalizability under safety-critical conditions. 

---
# Estimating the Effects of Sample Training Orders for Large Language Models without Retraining 

**Authors**: Hao Yang, Haoxuan Li, Mengyue Yang, Xu Chen, Mingming Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.22042)  

**Abstract**: The order of training samples plays a crucial role in large language models (LLMs), significantly impacting both their external performance and internal learning dynamics. Traditional methods for investigating this effect generally require retraining the model with various sample orders, which is computationally infeasible for LLMs. In this work, we improve traditional methods by designing a retraining-free framework. By approximating Adam optimizer updates with first- and second-order Taylor expansions and utilizing random projection methods to store intermediate checkpoints, our framework can efficiently estimate model parameters for arbitrary training sample orders. Next, we apply our framework to two downstream research problems: (1) Training curriculum design for LLMs -- we base our retraining-free framework to propose a novel curriculum learning strategy that augments curriculum proposals with estimated model performances, enabling more informed sample scheduling. (2) LLMs' memorization and generalization effect analysis -- we use our retraining-free framework to estimate how the positions of training samples influence LLMs' capacity for memorization and generalization. We conduct extensive experiments to validate the effectiveness of our retraining-free framework in reproducing the true model performances, and further demonstrate its potential in optimizing LLM training curricula and analyzing the memorization and generalization effects of LLMs. 

---
# Balanced Token Pruning: Accelerating Vision Language Models Beyond Local Optimization 

**Authors**: Kaiyuan Li, Xiaoyue Chen, Chen Gao, Yong Li, Xinlei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.22038)  

**Abstract**: Large Vision-Language Models (LVLMs) have shown impressive performance across multi-modal tasks by encoding images into thousands of tokens. However, the large number of image tokens results in significant computational overhead, and the use of dynamic high-resolution inputs further increases this burden. Previous approaches have attempted to reduce the number of image tokens through token pruning, typically by selecting tokens based on attention scores or image token diversity. Through empirical studies, we observe that existing methods often overlook the joint impact of pruning on both the current layer's output (local) and the outputs of subsequent layers (global), leading to suboptimal pruning decisions. To address this challenge, we propose Balanced Token Pruning (BTP), a plug-and-play method for pruning vision tokens. Specifically, our method utilizes a small calibration set to divide the pruning process into multiple stages. In the early stages, our method emphasizes the impact of pruning on subsequent layers, whereas in the deeper stages, the focus shifts toward preserving the consistency of local outputs. Extensive experiments across various LVLMs demonstrate the broad effectiveness of our approach on multiple benchmarks. Our method achieves a 78% compression rate while preserving 96.7% of the original models' performance on average. 

---
# Analysis and Evaluation of Synthetic Data Generation in Speech Dysfluency Detection 

**Authors**: Jinming Zhang, Xuanru Zhou, Jiachen Lian, Shuhe Li, William Li, Zoe Ezzes, Rian Bogley, Lisa Wauters, Zachary Miller, Jet Vonk, Brittany Morin, Maria Gorno-Tempini, Gopala Anumanchipalli  

**Link**: [PDF](https://arxiv.org/pdf/2505.22029)  

**Abstract**: Speech dysfluency detection is crucial for clinical diagnosis and language assessment, but existing methods are limited by the scarcity of high-quality annotated data. Although recent advances in TTS model have enabled synthetic dysfluency generation, existing synthetic datasets suffer from unnatural prosody and limited contextual diversity. To address these limitations, we propose LLM-Dys -- the most comprehensive dysfluent speech corpus with LLM-enhanced dysfluency simulation. This dataset captures 11 dysfluency categories spanning both word and phoneme levels. Building upon this resource, we improve an end-to-end dysfluency detection framework. Experimental validation demonstrates state-of-the-art performance. All data, models, and code are open-sourced at this https URL. 

---
# Improving Respiratory Sound Classification with Architecture-Agnostic Knowledge Distillation from Ensembles 

**Authors**: Miika Toikkanen, June-Woo Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.22027)  

**Abstract**: Respiratory sound datasets are limited in size and quality, making high performance difficult to achieve. Ensemble models help but inevitably increase compute cost at inference time. Soft label training distills knowledge efficiently with extra cost only at training. In this study, we explore soft labels for respiratory sound classification as an architecture-agnostic approach to distill an ensemble of teacher models into a student model. We examine different variations of our approach and find that even a single teacher, identical to the student, considerably improves performance beyond its own capability, with optimal gains achieved using only a few teachers. We achieve the new state-of-the-art Score of 64.39 on ICHBI, surpassing the previous best by 0.85 and improving average Scores across architectures by more than 1.16. Our results highlight the effectiveness of knowledge distillation with soft labels for respiratory sound classification, regardless of size or architecture. 

---
# GL-PGENet: A Parameterized Generation Framework for Robust Document Image Enhancement 

**Authors**: Zhihong Tang, Yang Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.22021)  

**Abstract**: Document Image Enhancement (DIE) serves as a critical component in Document AI systems, where its performance substantially determines the effectiveness of downstream tasks. To address the limitations of existing methods confined to single-degradation restoration or grayscale image processing, we present Global with Local Parametric Generation Enhancement Network (GL-PGENet), a novel architecture designed for multi-degraded color document images, ensuring both efficiency and robustness in real-world scenarios. Our solution incorporates three key innovations: First, a hierarchical enhancement framework that integrates global appearance correction with local refinement, enabling coarse-to-fine quality improvement. Second, a Dual-Branch Local-Refine Network with parametric generation mechanisms that replaces conventional direct prediction, producing enhanced outputs through learned intermediate parametric representations rather than pixel-wise mapping. This approach enhances local consistency while improving model generalization. Finally, a modified NestUNet architecture incorporating dense block to effectively fuse low-level pixel features and high-level semantic features, specifically adapted for document image characteristics. In addition, to enhance generalization performance, we adopt a two-stage training strategy: large-scale pretraining on a synthetic dataset of 500,000+ samples followed by task-specific fine-tuning. Extensive experiments demonstrate the superiority of GL-PGENet, achieving state-of-the-art SSIM scores of 0.7721 on DocUNet and 0.9480 on RealDAE. The model also exhibits remarkable cross-domain adaptability and maintains computational efficiency for high-resolution images without performance degradation, confirming its practical utility in real-world scenarios. 

---
# VRAG-RL: Empower Vision-Perception-Based RAG for Visually Rich Information Understanding via Iterative Reasoning with Reinforcement Learning 

**Authors**: Qiuchen Wang, Ruixue Ding, Yu Zeng, Zehui Chen, Lin Chen, Shihang Wang, Pengjun Xie, Fei Huang, Feng Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.22019)  

**Abstract**: Effectively retrieving, reasoning and understanding visually rich information remains a challenge for RAG methods. Traditional text-based methods cannot handle visual-related information. On the other hand, current vision-based RAG approaches are often limited by fixed pipelines and frequently struggle to reason effectively due to the insufficient activation of the fundamental capabilities of models. As RL has been proven to be beneficial for model reasoning, we introduce VRAG-RL, a novel RL framework tailored for complex reasoning across visually rich information. With this framework, VLMs interact with search engines, autonomously sampling single-turn or multi-turn reasoning trajectories with the help of visual perception tokens and undergoing continual optimization based on these samples. Our approach highlights key limitations of RL in RAG domains: (i) Prior Multi-modal RAG approaches tend to merely incorporate images into the context, leading to insufficient reasoning token allocation and neglecting visual-specific perception; and (ii) When models interact with search engines, their queries often fail to retrieve relevant information due to the inability to articulate requirements, thereby leading to suboptimal performance. To address these challenges, we define an action space tailored for visually rich inputs, with actions including cropping and scaling, allowing the model to gather information from a coarse-to-fine perspective. Furthermore, to bridge the gap between users' original inquiries and the retriever, we employ a simple yet effective reward that integrates query rewriting and retrieval performance with a model-based reward. Our VRAG-RL optimizes VLMs for RAG tasks using specially designed RL strategies, aligning the model with real-world applications. The code is available at \hyperlink{this https URL}{this https URL}. 

---
# Legal Assist AI: Leveraging Transformer-Based Model for Effective Legal Assistance 

**Authors**: Jatin Gupta, Akhil Sharma, Saransh Singhania, Ali Imam Abidi  

**Link**: [PDF](https://arxiv.org/pdf/2505.22003)  

**Abstract**: Pursuit of accessible legal assistance in India faces a critical gap, as many citizens struggle to leverage their legal rights due to limited awareness and access to relevant legal information. This paper introduces Legal Assist AI, a transformer-based model designed to bridge this gap by offering effective legal assistance through large language models (LLMs). The system retrieves relevant legal information from a curated database and generates accurate responses, enabling effective assistance for diverse users, including legal professionals, scholars, and the general public. The model was fine-tuned on extensive datasets from the Indian legal domain, including Indian Constitution, Bharatiya Nyaya Sanhita (BNS), Bharatiya Nagarik Suraksha Sanhita (BNSS) and so forth, providing a robust understanding of the complexities of Indian law. By incorporating domain-specific legal datasets, the proposed model demonstrated remarkable efficiency and specialization in legal Question-Answering. The model was evaluated against state-of-the-art models such as GPT-3.5 Turbo and Mistral 7B, achieving a 60.08% score on the AIBE, outperforming its competitors in legal reasoning and accuracy. Unlike other models, Legal Assist AI avoided common issues such as hallucinations, making it highly reliable for practical legal applications. It showcases the model's applicability in real-world legal scenarios, with future iterations aiming to enhance performance and expand its dataset to cover a broader range of multilingual and case-specific queries as well. 

---
# Learning World Models for Interactive Video Generation 

**Authors**: Taiye Chen, Xun Hu, Zihan Ding, Chi Jin  

**Link**: [PDF](https://arxiv.org/pdf/2505.21996)  

**Abstract**: Foundational world models must be both interactive and preserve spatiotemporal coherence for effective future planning with action choices. However, present models for long video generation have limited inherent world modeling capabilities due to two main challenges: compounding errors and insufficient memory mechanisms. We enhance image-to-video models with interactive capabilities through additional action conditioning and autoregressive framework, and reveal that compounding error is inherently irreducible in autoregressive video generation, while insufficient memory mechanism leads to incoherence of world models. We propose video retrieval augmented generation (VRAG) with explicit global state conditioning, which significantly reduces long-term compounding errors and increases spatiotemporal consistency of world models. In contrast, naive autoregressive generation with extended context windows and retrieval-augmented generation prove less effective for video generation, primarily due to the limited in-context learning capabilities of current video models. Our work illuminates the fundamental challenges in video world models and establishes a comprehensive benchmark for improving video generation models with internal world modeling capabilities. 

---
# Reward-Independent Messaging for Decentralized Multi-Agent Reinforcement Learning 

**Authors**: Naoto Yoshida, Tadahiro Taniguchi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21985)  

**Abstract**: In multi-agent reinforcement learning (MARL), effective communication improves agent performance, particularly under partial observability. We propose MARL-CPC, a framework that enables communication among fully decentralized, independent agents without parameter sharing. MARL-CPC incorporates a message learning model based on collective predictive coding (CPC) from emergent communication research. Unlike conventional methods that treat messages as part of the action space and assume cooperation, MARL-CPC links messages to state inference, supporting communication in non-cooperative, reward-independent settings. We introduce two algorithms -Bandit-CPC and IPPO-CPC- and evaluate them in non-cooperative MARL tasks. Benchmarks show that both outperform standard message-as-action approaches, establishing effective communication even when messages offer no direct benefit to the sender. These results highlight MARL-CPC's potential for enabling coordination in complex, decentralized environments. 

---
# Learning Compositional Behaviors from Demonstration and Language 

**Authors**: Weiyu Liu, Neil Nie, Ruohan Zhang, Jiayuan Mao, Jiajun Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21981)  

**Abstract**: We introduce Behavior from Language and Demonstration (BLADE), a framework for long-horizon robotic manipulation by integrating imitation learning and model-based planning. BLADE leverages language-annotated demonstrations, extracts abstract action knowledge from large language models (LLMs), and constructs a library of structured, high-level action representations. These representations include preconditions and effects grounded in visual perception for each high-level action, along with corresponding controllers implemented as neural network-based policies. BLADE can recover such structured representations automatically, without manually labeled states or symbolic definitions. BLADE shows significant capabilities in generalizing to novel situations, including novel initial states, external state perturbations, and novel goals. We validate the effectiveness of our approach both in simulation and on real robots with a diverse set of objects with articulated parts, partial observability, and geometric constraints. 

---
# Judging LLMs on a Simplex 

**Authors**: Patrick Vossler, Fan Xia, Yifan Mai, Jean Feng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21972)  

**Abstract**: Automated evaluation of free-form outputs from large language models (LLMs) is challenging because many distinct answers can be equally valid. A common practice is to use LLMs themselves as judges, but the theoretical properties of this approach are not yet well understood. We show that a geometric framework that represents both judges and candidates as points on a probability simplex can provide helpful insight on what is or is not identifiable using LLM judges. Our theoretical analysis uncovers a "phase transition" in ranking identifiability: for binary scoring systems, true rankings are identifiable even with weak judges under mild assumptions, while rankings become non-identifiable for three or more scoring levels even with infinite data, absent additional prior knowledge. This non-identifiability highlights how uncertainty in rankings stems from not only aleatoric uncertainty (i.e., inherent stochasticity in the data) but also epistemic uncertainty regarding which assumptions hold, an aspect that has received limited attention until now. To integrate both types of uncertainty, we use Bayesian inference to encode assumptions as priors and conduct sensitivity analysis of ranking estimates and credible intervals. Empirical evaluations across multiple benchmarks demonstrate that Bayesian inference yields more accurate rankings and substantially improves coverage rates. These results underscore the importance of taking a more holistic approach to uncertainty quantification when using LLMs as judges. 

---
# DORAEMON: Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation 

**Authors**: Tianjun Gu, Linfeng Li, Xuhong Wang, Chenghua Gong, Jingyu Gong, Zhizhong Zhang, Yuan Xie, Lizhuang Ma, Xin Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.21969)  

**Abstract**: Adaptive navigation in unfamiliar environments is crucial for household service robots but remains challenging due to the need for both low-level path planning and high-level scene understanding. While recent vision-language model (VLM) based zero-shot approaches reduce dependence on prior maps and scene-specific training data, they face significant limitations: spatiotemporal discontinuity from discrete observations, unstructured memory representations, and insufficient task understanding leading to navigation failures. We propose DORAEMON (Decentralized Ontology-aware Reliable Agent with Enhanced Memory Oriented Navigation), a novel cognitive-inspired framework consisting of Ventral and Dorsal Streams that mimics human navigation capabilities. The Dorsal Stream implements the Hierarchical Semantic-Spatial Fusion and Topology Map to handle spatiotemporal discontinuities, while the Ventral Stream combines RAG-VLM and Policy-VLM to improve decision-making. Our approach also develops Nav-Ensurance to ensure navigation safety and efficiency. We evaluate DORAEMON on the HM3D, MP3D, and GOAT datasets, where it achieves state-of-the-art performance on both success rate (SR) and success weighted by path length (SPL) metrics, significantly outperforming existing methods. We also introduce a new evaluation metric (AORI) to assess navigation intelligence better. Comprehensive experiments demonstrate DORAEMON's effectiveness in zero-shot autonomous navigation without requiring prior map building or pre-training. 

---
# MapStory: LLM-Powered Text-Driven Map Animation Prototyping with Human-in-the-Loop Editing 

**Authors**: Aditya Gunturu, Ben Pearman, Keiichi Ihara, Morteza Faraji, Bryan Wang, Rubaiat Habib Kazi, Ryo Suzuki  

**Link**: [PDF](https://arxiv.org/pdf/2505.21966)  

**Abstract**: We introduce MapStory, an LLM-powered animation authoring tool that generates editable map animation sequences directly from natural language text. Given a user-written script, MapStory leverages an agentic architecture to automatically produce a scene breakdown, which decomposes the script into key animation building blocks such as camera movements, visual highlights, and animated elements. Our system includes a researcher component that accurately queries geospatial information by leveraging an LLM with web search, enabling the automatic extraction of relevant regions, paths, and coordinates while allowing users to edit and query for changes or additional information to refine the results. Additionally, users can fine-tune parameters of these blocks through an interactive timeline editor. We detail the system's design and architecture, informed by formative interviews with professional animators and an analysis of 200 existing map animation videos. Our evaluation, which includes expert interviews (N=5) and a usability study (N=12), demonstrates that MapStory enables users to create map animations with ease, facilitates faster iteration, encourages creative exploration, and lowers barriers to creating map-centric stories. 

---
# LaMDAgent: An Autonomous Framework for Post-Training Pipeline Optimization via LLM Agents 

**Authors**: Taro Yano, Yoichi Ishibashi, Masafumi Oyamada  

**Link**: [PDF](https://arxiv.org/pdf/2505.21963)  

**Abstract**: Large Language Models (LLMs) have demonstrated exceptional performance across a wide range of tasks. To further tailor LLMs to specific domains or applications, post-training techniques such as Supervised Fine-Tuning (SFT), Preference Learning, and model merging are commonly employed. While each of these methods has been extensively studied in isolation, the automated construction of complete post-training pipelines remains an underexplored area. Existing approaches typically rely on manual design or focus narrowly on optimizing individual components, such as data ordering or merging strategies. In this work, we introduce LaMDAgent (short for Language Model Developing Agent), a novel framework that autonomously constructs and optimizes full post-training pipelines through the use of LLM-based agents. LaMDAgent systematically explores diverse model generation techniques, datasets, and hyperparameter configurations, leveraging task-based feedback to discover high-performing pipelines with minimal human intervention. Our experiments show that LaMDAgent improves tool-use accuracy by 9.0 points while preserving instruction-following capabilities. Moreover, it uncovers effective post-training strategies that are often overlooked by conventional human-driven exploration. We further analyze the impact of data and model size scaling to reduce computational costs on the exploration, finding that model size scalings introduces new challenges, whereas scaling data size enables cost-effective pipeline discovery. 

---
# Cross-modal RAG: Sub-dimensional Retrieval-Augmented Text-to-Image Generation 

**Authors**: Mengdan Zhu, Senhao Cheng, Guangji Bai, Yifei Zhang, Liang Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2505.21956)  

**Abstract**: Text-to-image generation increasingly demands access to domain-specific, fine-grained, and rapidly evolving knowledge that pretrained models cannot fully capture. Existing Retrieval-Augmented Generation (RAG) methods attempt to address this by retrieving globally relevant images, but they fail when no single image contains all desired elements from a complex user query. We propose Cross-modal RAG, a novel framework that decomposes both queries and images into sub-dimensional components, enabling subquery-aware retrieval and generation. Our method introduces a hybrid retrieval strategy - combining a sub-dimensional sparse retriever with a dense retriever - to identify a Pareto-optimal set of images, each contributing complementary aspects of the query. During generation, a multimodal large language model is guided to selectively condition on relevant visual features aligned to specific subqueries, ensuring subquery-aware image synthesis. Extensive experiments on MS-COCO, Flickr30K, WikiArt, CUB, and ImageNet-LT demonstrate that Cross-modal RAG significantly outperforms existing baselines in both retrieval and generation quality, while maintaining high efficiency. 

---
# Towards Comprehensive Scene Understanding: Integrating First and Third-Person Views for LVLMs 

**Authors**: Insu Lee, Wooje Park, Jaeyun Jang, Minyoung Noh, Kyuhong Shim, Byonghyo Shim  

**Link**: [PDF](https://arxiv.org/pdf/2505.21955)  

**Abstract**: Large vision-language models (LVLMs) are increasingly deployed in interactive applications such as virtual and augmented reality, where first-person (egocentric) view captured by head-mounted cameras serves as key input. While this view offers fine-grained cues about user attention and hand-object interactions, their narrow field of view and lack of global context often lead to failures on spatially or contextually demanding queries. To address this, we introduce a framework that augments egocentric inputs with third-person (exocentric) views, providing complementary information such as global scene layout and object visibility to LVLMs. We present E3VQA, the first benchmark for multi-view question answering with 4K high-quality question-answer pairs grounded in synchronized ego-exo image pairs. Additionally, we propose M3CoT, a training-free prompting technique that constructs a unified scene representation by integrating scene graphs from three complementary perspectives. M3CoT enables LVLMs to reason more effectively across views, yielding consistent performance gains (4.84% for GPT-4o and 5.94% for Gemini 2.0 Flash) over a recent CoT baseline. Our extensive evaluation reveals key strengths and limitations of LVLMs in multi-view reasoning and highlights the value of leveraging both egocentric and exocentric inputs. 

---
# UniTalk: Towards Universal Active Speaker Detection in Real World Scenarios 

**Authors**: Le Thien Phuc Nguyen, Zhuoran Yu, Khoa Quang Nhat Cao, Yuwei Guo, Tu Ho Manh Pham, Tuan Tai Nguyen, Toan Ngo Duc Vo, Lucas Poon, Soochahn Lee, Yong Jae Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.21954)  

**Abstract**: We present UniTalk, a novel dataset specifically designed for the task of active speaker detection, emphasizing challenging scenarios to enhance model generalization. Unlike previously established benchmarks such as AVA, which predominantly features old movies and thus exhibits significant domain gaps, UniTalk focuses explicitly on diverse and difficult real-world conditions. These include underrepresented languages, noisy backgrounds, and crowded scenes - such as multiple visible speakers speaking concurrently or in overlapping turns. It contains over 44.5 hours of video with frame-level active speaker annotations across 48,693 speaking identities, and spans a broad range of video types that reflect real-world conditions. Through rigorous evaluation, we show that state-of-the-art models, while achieving nearly perfect scores on AVA, fail to reach saturation on UniTalk, suggesting that the ASD task remains far from solved under realistic conditions. Nevertheless, models trained on UniTalk demonstrate stronger generalization to modern "in-the-wild" datasets like Talkies and ASW, as well as to AVA. UniTalk thus establishes a new benchmark for active speaker detection, providing researchers with a valuable resource for developing and evaluating versatile and resilient models.
Dataset: this https URL
Code: this https URL 

---
# Practical Adversarial Attacks on Stochastic Bandits via Fake Data Injection 

**Authors**: Qirun Zeng, Eric He, Richard Hoffmann, Xuchuang Wang, Jinhang Zuo  

**Link**: [PDF](https://arxiv.org/pdf/2505.21938)  

**Abstract**: Adversarial attacks on stochastic bandits have traditionally relied on some unrealistic assumptions, such as per-round reward manipulation and unbounded perturbations, limiting their relevance to real-world systems. We propose a more practical threat model, Fake Data Injection, which reflects realistic adversarial constraints: the attacker can inject only a limited number of bounded fake feedback samples into the learner's history, simulating legitimate interactions. We design efficient attack strategies under this model, explicitly addressing both magnitude constraints (on reward values) and temporal constraints (on when and how often data can be injected). Our theoretical analysis shows that these attacks can mislead both Upper Confidence Bound (UCB) and Thompson Sampling algorithms into selecting a target arm in nearly all rounds while incurring only sublinear attack cost. Experiments on synthetic and real-world datasets validate the effectiveness of our strategies, revealing significant vulnerabilities in widely used stochastic bandit algorithms under practical adversarial scenarios. 

---
# Subspecialty-Specific Foundation Model for Intelligent Gastrointestinal Pathology 

**Authors**: Lianghui Zhu, Xitong Ling, Minxi Ouyang, Xiaoping Liu, Mingxi Fu, Tian Guan, Fanglei Fu, Xuanyu Wang, Maomao Zeng, Mingxi Zhu, Yibo Jin, Liming Liu, Song Duan, Qiming He, Yizhi Wang, Luxi Xie, Houqiang Li, Yonghong He, Sufang Tian  

**Link**: [PDF](https://arxiv.org/pdf/2505.21928)  

**Abstract**: Gastrointestinal (GI) diseases represent a clinically significant burden, necessitating precise diagnostic approaches to optimize patient outcomes. Conventional histopathological diagnosis, heavily reliant on the subjective interpretation of pathologists, suffers from limited reproducibility and diagnostic variability. To overcome these limitations and address the lack of pathology-specific foundation models for GI diseases, we develop Digepath, a specialized foundation model for GI pathology. Our framework introduces a dual-phase iterative optimization strategy combining pretraining with fine-screening, specifically designed to address the detection of sparsely distributed lesion areas in whole-slide images. Digepath is pretrained on more than 353 million image patches from over 200,000 hematoxylin and eosin-stained slides of GI diseases. It attains state-of-the-art performance on 33 out of 34 tasks related to GI pathology, including pathological diagnosis, molecular prediction, gene mutation prediction, and prognosis evaluation, particularly in diagnostically ambiguous cases and resolution-agnostic tissue this http URL further translate the intelligent screening module for early GI cancer and achieve near-perfect 99.6% sensitivity across 9 independent medical institutions nationwide. The outstanding performance of Digepath highlights its potential to bridge critical gaps in histopathological practice. This work not only advances AI-driven precision pathology for GI diseases but also establishes a transferable paradigm for other pathology subspecialties. 

---
# Beyond Completion: A Foundation Model for General Knowledge Graph Reasoning 

**Authors**: Yin Hua, Zhiqiang Liu, Mingyang Chen, Zheng Fang, Chi Man Wong, Lingxiao Li, Chi Man Vong, Huajun Chen, Wen Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21926)  

**Abstract**: In natural language processing (NLP) and computer vision (CV), the successful application of foundation models across diverse tasks has demonstrated their remarkable potential. However, despite the rich structural and textual information embedded in knowledge graphs (KGs), existing research of foundation model for KG has primarily focused on their structural aspects, with most efforts restricted to in-KG tasks (e.g., knowledge graph completion, KGC). This limitation has hindered progress in addressing more challenging out-of-KG tasks. In this paper, we introduce MERRY, a foundation model for general knowledge graph reasoning, and investigate its performance across two task categories: in-KG reasoning tasks (e.g., KGC) and out-of-KG tasks (e.g., KG question answering, KGQA). We not only utilize the structural information, but also the textual information in KGs. Specifically, we propose a multi-perspective Conditional Message Passing (CMP) encoding architecture to bridge the gap between textual and structural modalities, enabling their seamless integration. Additionally, we introduce a dynamic residual fusion module to selectively retain relevant textual information and a flexible edge scoring mechanism to adapt to diverse downstream tasks. Comprehensive evaluations on 28 datasets demonstrate that MERRY outperforms existing baselines in most scenarios, showcasing strong reasoning capabilities within KGs and excellent generalization to out-of-KG tasks such as KGQA. 

---
# FALCON: An ML Framework for Fully Automated Layout-Constrained Analog Circuit Design 

**Authors**: Asal Mehradfar, Xuzhe Zhao, Yilun Huang, Emir Ceyani, Yankai Yang, Shihao Han, Hamidreza Aghasi, Salman Avestimehr  

**Link**: [PDF](https://arxiv.org/pdf/2505.21923)  

**Abstract**: Designing analog circuits from performance specifications is a complex, multi-stage process encompassing topology selection, parameter inference, and layout feasibility. We introduce FALCON, a unified machine learning framework that enables fully automated, specification-driven analog circuit synthesis through topology selection and layout-constrained optimization. Given a target performance, FALCON first selects an appropriate circuit topology using a performance-driven classifier guided by human design heuristics. Next, it employs a custom, edge-centric graph neural network trained to map circuit topology and parameters to performance, enabling gradient-based parameter inference through the learned forward model. This inference is guided by a differentiable layout cost, derived from analytical equations capturing parasitic and frequency-dependent effects, and constrained by design rules. We train and evaluate FALCON on a large-scale custom dataset of 1M analog mm-wave circuits, generated and simulated using Cadence Spectre across 20 expert-designed topologies. Through this evaluation, FALCON demonstrates >99\% accuracy in topology inference, <10\% relative error in performance prediction, and efficient layout-aware design that completes in under 1 second per instance. Together, these results position FALCON as a practical and extensible foundation model for end-to-end analog circuit design automation. 

---
# Towards Efficient Key-Value Cache Management for Prefix Prefilling in LLM Inference 

**Authors**: Yue Zhu, Hao Yu, Chen Wang, Zhuoran Liu, Eun Kyung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2505.21919)  

**Abstract**: The increasing adoption of large language models (LLMs) with extended context windows necessitates efficient Key-Value Cache (KVC) management to optimize inference performance. Inference workloads like Retrieval-Augmented Generation (RAG) and agents exhibit high cache reusability, making efficient caching critical to reducing redundancy and improving speed. We analyze real-world KVC access patterns using publicly available traces and evaluate commercial key-value stores like Redis and state-of-the-art RDMA-based systems (CHIME [1] and Sherman [2]) for KVC metadata management. Our work demonstrates the lack of tailored storage solution for KVC prefilling, underscores the need for an efficient distributed caching system with optimized metadata management for LLM workloads, and provides insights into designing improved KVC management systems for scalable, low-latency inference. 

---
# Self-supervised Learning Method Using Transformer for Multi-dimensional Sensor Data Processing 

**Authors**: Haruki Kai, Tsuyoshi Okita  

**Link**: [PDF](https://arxiv.org/pdf/2505.21918)  

**Abstract**: We developed a deep learning algorithm for human activity recognition using sensor signals as input. In this study, we built a pretrained language model based on the Transformer architecture, which is widely used in natural language processing. By leveraging this pretrained model, we aimed to improve performance on the downstream task of human activity recognition. While this task can be addressed using a vanilla Transformer, we propose an enhanced n-dimensional numerical processing Transformer that incorporates three key features: embedding n-dimensional numerical data through a linear layer, binning-based pre-processing, and a linear transformation in the output layer. We evaluated the effectiveness of our proposed model across five different datasets. Compared to the vanilla Transformer, our model demonstrated 10%-15% improvements in accuracy. 

---
# Reinforcement Learning for Out-of-Distribution Reasoning in LLMs: An Empirical Study on Diagnosis-Related Group Coding 

**Authors**: Hanyin Wang, Zhenbang Wu, Gururaj Kolar, Hariprasad Korsapati, Brian Bartlett, Bryan Hull, Jimeng Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.21908)  

**Abstract**: Diagnosis-Related Group (DRG) codes are essential for hospital reimbursement and operations but require labor-intensive assignment. Large Language Models (LLMs) struggle with DRG coding due to the out-of-distribution (OOD) nature of the task: pretraining corpora rarely contain private clinical or billing data. We introduce DRG-Sapphire, which uses large-scale reinforcement learning (RL) for automated DRG coding from clinical notes. Built on Qwen2.5-7B and trained with Group Relative Policy Optimization (GRPO) using rule-based rewards, DRG-Sapphire introduces a series of RL enhancements to address domain-specific challenges not seen in previous mathematical tasks. Our model achieves state-of-the-art accuracy on the MIMIC-IV benchmark and generates physician-validated reasoning for DRG assignments, significantly enhancing explainability. Our study further sheds light on broader challenges of applying RL to knowledge-intensive, OOD tasks. We observe that RL performance scales approximately linearly with the logarithm of the number of supervised fine-tuning (SFT) examples, suggesting that RL effectiveness is fundamentally constrained by the domain knowledge encoded in the base model. For OOD tasks like DRG coding, strong RL performance requires sufficient knowledge infusion prior to RL. Consequently, scaling SFT may be more effective and computationally efficient than scaling RL alone for such tasks. 

---
# Vision-Language-Action Model with Open-World Embodied Reasoning from Pretrained Knowledge 

**Authors**: Zhongyi Zhou, Yichen Zhu, Junjie Wen, Chaomin Shen, Yi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21906)  

**Abstract**: Vision-language-action (VLA) models have emerged as the next generation of models in robotics. However, despite leveraging powerful pre-trained Vision-Language Models (VLMs), existing end-to-end VLA systems often lose key capabilities during fine-tuning as the model adapts to specific robotic tasks. We argue that a generalizable VLA model should retain and expand upon the VLM's core competencies: 1) Open-world embodied reasoning - the VLA should inherit the knowledge from VLM, i.e., recognize anything that the VLM can recognize, capable of solving math problems, possessing visual-spatial intelligence, 2) Reasoning following - effectively translating the open-world reasoning into actionable steps for the robot. In this work, we introduce ChatVLA-2, a novel mixture-of-expert VLA model coupled with a specialized three-stage training pipeline designed to preserve the VLM's original strengths while enabling actionable reasoning. To validate our approach, we design a math-matching task wherein a robot interprets math problems written on a whiteboard and picks corresponding number cards from a table to solve equations. Remarkably, our method exhibits exceptional mathematical reasoning and OCR capabilities, despite these abilities not being explicitly trained within the VLA. Furthermore, we demonstrate that the VLA possesses strong spatial reasoning skills, enabling it to interpret novel directional instructions involving previously unseen objects. Overall, our method showcases reasoning and comprehension abilities that significantly surpass state-of-the-art imitation learning methods such as OpenVLA, DexVLA, and pi-zero. This work represents a substantial advancement toward developing truly generalizable robotic foundation models endowed with robust reasoning capacities. 

---
# CAST: Contrastive Adaptation and Distillation for Semi-Supervised Instance Segmentation 

**Authors**: Pardis Taghavi, Tian Liu, Renjie Li, Reza Langari, Zhengzhong Tu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21904)  

**Abstract**: Instance segmentation demands costly per-pixel annotations and large models. We introduce CAST, a semi-supervised knowledge distillation (SSKD) framework that compresses pretrained vision foundation models (VFM) into compact experts using limited labeled and abundant unlabeled data. CAST unfolds in three stages: (1) domain adaptation of the VFM teacher(s) via self-training with contrastive pixel calibration, (2) distillation into a compact student via a unified multi-objective loss that couples standard supervision and pseudo-labels with our instance-aware pixel-wise contrastive term, and (3) fine-tuning on labeled data to remove residual pseudo-label bias. Central to CAST is an \emph{instance-aware pixel-wise contrastive loss} that fuses mask and class scores to mine informative negatives and enforce clear inter-instance margins. By maintaining this contrastive signal across both adaptation and distillation, we align teacher and student embeddings and fully leverage unlabeled images. On Cityscapes and ADE20K, our ~11X smaller student surpasses its adapted VFM teacher(s) by +3.4 AP (33.9 vs. 30.5) and +1.5 AP (16.7 vs. 15.2) and outperforms state-of-the-art semi-supervised approaches. 

---
# Co-Saving: Resource Aware Multi-Agent Collaboration for Software Development 

**Authors**: Rennai Qiu, Chen Qian, Ran Li, Yufan Dang, Weize Chen, Cheng Yang, Yingli Zhang, Ye Tian, Xuantang Xiong, Lei Han, Zhiyuan Liu, Maosong Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.21898)  

**Abstract**: Recent advancements in Large Language Models (LLMs) and autonomous agents have demonstrated remarkable capabilities across various domains. However, standalone agents frequently encounter limitations when handling complex tasks that demand extensive interactions and substantial computational resources. Although Multi-Agent Systems (MAS) alleviate some of these limitations through collaborative mechanisms like task decomposition, iterative communication, and role specialization, they typically remain resource-unaware, incurring significant inefficiencies due to high token consumption and excessive execution time. To address these limitations, we propose a resource-aware multi-agent system -- Co-Saving (meaning that multiple agents collaboratively engage in resource-saving activities), which leverages experiential knowledge to enhance operational efficiency and solution quality. Our key innovation is the introduction of "shortcuts" -- instructional transitions learned from historically successful trajectories -- which allows to bypass redundant reasoning agents and expedite the collective problem-solving process. Experiments for software development tasks demonstrate significant advantages over existing methods. Specifically, compared to the state-of-the-art MAS ChatDev, our method achieves an average reduction of 50.85% in token usage, and improves the overall code quality by 10.06%. 

---
# Compressing Sine-Activated Low-Rank Adapters through Post-Training Quantization 

**Authors**: Cameron Gordon, Yiping Ji, Hemanth Saratchandran, Paul Albert, Simon Lucey  

**Link**: [PDF](https://arxiv.org/pdf/2505.21895)  

**Abstract**: Low-Rank Adaptation (LoRA) has become a standard approach for parameter-efficient fine-tuning, offering substantial reductions in trainable parameters by modeling updates as the product of two low-rank matrices. While effective, the low-rank constraint inherently limits representational capacity, often resulting in reduced performance compared to full-rank fine-tuning. Recent work by Ji et al. (2025) has addressed this limitation by applying a fixed-frequency sinusoidal transformation to low-rank adapters, increasing their stable rank without introducing additional parameters. This raises a crucial question: can the same sine-activated technique be successfully applied within the context of Post-Training Quantization to retain benefits even after model compression? In this paper, we investigate this question by extending the sinusoidal transformation framework to quantized LoRA adapters. We develop a theoretical analysis showing that the stable rank of a quantized adapter is tightly linked to that of its full-precision counterpart, motivating the use of such rank-enhancing functions even under quantization. Our results demonstrate that the expressivity gains from a sinusoidal non-linearity persist after quantization, yielding highly compressed adapters with negligible loss in performance. We validate our approach across a range of fine-tuning tasks for language, vision and text-to-image generation achieving significant memory savings while maintaining competitive accuracy. 

---
# SDPO: Importance-Sampled Direct Preference Optimization for Stable Diffusion Training 

**Authors**: Xiaomeng Yang, Zhiyu Tan, Junyan Wang, Zhijian Zhou, Hao Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.21893)  

**Abstract**: Preference learning has become a central technique for aligning generative models with human expectations. Recently, it has been extended to diffusion models through methods like Direct Preference Optimization (DPO). However, existing approaches such as Diffusion-DPO suffer from two key challenges: timestep-dependent instability, caused by a mismatch between the reverse and forward diffusion processes and by high gradient variance in early noisy timesteps, and off-policy bias arising from the mismatch between optimization and data collection policies. We begin by analyzing the reverse diffusion trajectory and observe that instability primarily occurs at early timesteps with low importance weights. To address these issues, we first propose DPO-C\&M, a practical strategy that improves stability by clipping and masking uninformative timesteps while partially mitigating off-policy bias. Building on this, we introduce SDPO (Importance-Sampled Direct Preference Optimization), a principled framework that incorporates importance sampling into the objective to fully correct for off-policy bias and emphasize informative updates during the diffusion process. Experiments on CogVideoX-2B, CogVideoX-5B, and Wan2.1-1.3B demonstrate that both methods outperform standard Diffusion-DPO, with SDPO achieving superior VBench scores, human preference alignment, and training robustness. These results highlight the importance of timestep-aware, distribution-corrected optimization in diffusion-based preference learning. 

---
# Incorporating LLMs for Large-Scale Urban Complex Mobility Simulation 

**Authors**: Yu-Lun Song, Chung-En Tsern, Che-Cheng Wu, Yu-Ming Chang, Syuan-Bo Huang, Wei-Chu Chen, Michael Chia-Liang Lin, Yu-Ta Lin  

**Link**: [PDF](https://arxiv.org/pdf/2505.21880)  

**Abstract**: This study presents an innovative approach to urban mobility simulation by integrating a Large Language Model (LLM) with Agent-Based Modeling (ABM). Unlike traditional rule-based ABM, the proposed framework leverages LLM to enhance agent diversity and realism by generating synthetic population profiles, allocating routine and occasional locations, and simulating personalized routes. Using real-world data, the simulation models individual behaviors and large-scale mobility patterns in Taipei City. Key insights, such as route heat maps and mode-specific indicators, provide urban planners with actionable information for policy-making. Future work focuses on establishing robust validation frameworks to ensure accuracy and reliability in urban planning applications. 

---
# Symbolic Foundation Regressor on Complex Networks 

**Authors**: Weiting Liu, Jiaxu Cui, Jiao Hu, En Wang, Bo Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21879)  

**Abstract**: In science, we are interested not only in forecasting but also in understanding how predictions are made, specifically what the interpretable underlying model looks like. Data-driven machine learning technology can significantly streamline the complex and time-consuming traditional manual process of discovering scientific laws, helping us gain insights into fundamental issues in modern science. In this work, we introduce a pre-trained symbolic foundation regressor that can effectively compress complex data with numerous interacting variables while producing interpretable physical representations. Our model has been rigorously tested on non-network symbolic regression, symbolic regression on complex networks, and the inference of network dynamics across various domains, including physics, biochemistry, ecology, and epidemiology. The results indicate a remarkable improvement in equation inference efficiency, being three times more effective than baseline approaches while maintaining accurate predictions. Furthermore, we apply our model to uncover more intuitive laws of interaction transmission from global epidemic outbreak data, achieving optimal data fitting. This model extends the application boundary of pre-trained symbolic regression models to complex networks, and we believe it provides a foundational solution for revealing the hidden mechanisms behind changes in complex phenomena, enhancing interpretability, and inspiring further scientific discoveries. 

---
# EPiC: Efficient Video Camera Control Learning with Precise Anchor-Video Guidance 

**Authors**: Zun Wang, Jaemin Cho, Jialu Li, Han Lin, Jaehong Yoon, Yue Zhang, Mohit Bansal  

**Link**: [PDF](https://arxiv.org/pdf/2505.21876)  

**Abstract**: Recent approaches on 3D camera control in video diffusion models (VDMs) often create anchor videos to guide diffusion models as a structured prior by rendering from estimated point clouds following annotated camera trajectories. However, errors inherent in point cloud estimation often lead to inaccurate anchor videos. Moreover, the requirement for extensive camera trajectory annotations further increases resource demands. To address these limitations, we introduce EPiC, an efficient and precise camera control learning framework that automatically constructs high-quality anchor videos without expensive camera trajectory annotations. Concretely, we create highly precise anchor videos for training by masking source videos based on first-frame visibility. This approach ensures high alignment, eliminates the need for camera trajectory annotations, and thus can be readily applied to any in-the-wild video to generate image-to-video (I2V) training pairs. Furthermore, we introduce Anchor-ControlNet, a lightweight conditioning module that integrates anchor video guidance in visible regions to pretrained VDMs, with less than 1% of backbone model parameters. By combining the proposed anchor video data and ControlNet module, EPiC achieves efficient training with substantially fewer parameters, training steps, and less data, without requiring modifications to the diffusion model backbone typically needed to mitigate rendering misalignments. Although being trained on masking-based anchor videos, our method generalizes robustly to anchor videos made with point clouds during inference, enabling precise 3D-informed camera control. EPiC achieves SOTA performance on RealEstate10K and MiraData for I2V camera control task, demonstrating precise and robust camera control ability both quantitatively and qualitatively. Notably, EPiC also exhibits strong zero-shot generalization to video-to-video scenarios. 

---
# HelixDesign-Binder: A Scalable Production-Grade Platform for Binder Design Built on HelixFold3 

**Authors**: Jie Gao, Jun Li, Jing Hu, Shanzhuo Zhang, Kunrui Zhu, Yueyang Huang, Xiaonan Zhang, Xiaomin Fang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21873)  

**Abstract**: Protein binder design is central to therapeutics, diagnostics, and synthetic biology, yet practical deployment remains challenging due to fragmented workflows, high computational costs, and complex tool integration. We present HelixDesign-Binder, a production-grade, high-throughput platform built on HelixFold3 that automates the full binder design pipeline, from backbone generation and sequence design to structural evaluation and multi-dimensional scoring. By unifying these stages into a scalable and user-friendly system, HelixDesign-Binder enables efficient exploration of binder candidates with favorable structural, energetic, and physicochemical properties. The platform leverages Baidu Cloud's high-performance infrastructure to support large-scale design and incorporates advanced scoring metrics, including ipTM, predicted binding free energy, and interface hydrophobicity. Benchmarking across six protein targets demonstrates that HelixDesign-Binder reliably produces diverse and high-quality binders, some of which match or exceed validated designs in predicted binding affinity. HelixDesign-Binder is accessible via an interactive web interface in PaddleHelix platform, supporting both academic research and industrial applications in antibody and protein binder development. 

---
# Evaluating the Retrieval Robustness of Large Language Models 

**Authors**: Shuyang Cao, Karthik Radhakrishnan, David Rosenberg, Steven Lu, Pengxiang Cheng, Lu Wang, Shiyue Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21870)  

**Abstract**: Retrieval-augmented generation (RAG) generally enhances large language models' (LLMs) ability to solve knowledge-intensive tasks. But RAG may also lead to performance degradation due to imperfect retrieval and the model's limited ability to leverage retrieved content. In this work, we evaluate the robustness of LLMs in practical RAG setups (henceforth retrieval robustness). We focus on three research questions: (1) whether RAG is always better than non-RAG; (2) whether more retrieved documents always lead to better performance; (3) and whether document orders impact results. To facilitate this study, we establish a benchmark of 1500 open-domain questions, each with retrieved documents from Wikipedia. We introduce three robustness metrics, each corresponds to one research question. Our comprehensive experiments, involving 11 LLMs and 3 prompting strategies, reveal that all of these LLMs exhibit surprisingly high retrieval robustness; nonetheless, different degrees of imperfect robustness hinders them from fully utilizing the benefits of RAG. 

---
# CSI-Bench: A Large-Scale In-the-Wild Dataset for Multi-task WiFi Sensing 

**Authors**: Guozhen Zhu, Yuqian Hu, Weihang Gao, Wei-Hsiang Wang, Beibei Wang, K. J. Ray Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21866)  

**Abstract**: WiFi sensing has emerged as a compelling contactless modality for human activity monitoring by capturing fine-grained variations in Channel State Information (CSI). Its ability to operate continuously and non-intrusively while preserving user privacy makes it particularly suitable for health monitoring. However, existing WiFi sensing systems struggle to generalize in real-world settings, largely due to datasets collected in controlled environments with homogeneous hardware and fragmented, session-based recordings that fail to reflect continuous daily activity.
We present CSI-Bench, a large-scale, in-the-wild benchmark dataset collected using commercial WiFi edge devices across 26 diverse indoor environments with 35 real users. Spanning over 461 hours of effective data, CSI-Bench captures realistic signal variability under natural conditions. It includes task-specific datasets for fall detection, breathing monitoring, localization, and motion source recognition, as well as a co-labeled multitask dataset with joint annotations for user identity, activity, and proximity. To support the development of robust and generalizable models, CSI-Bench provides standardized evaluation splits and baseline results for both single-task and multi-task learning. CSI-Bench offers a foundation for scalable, privacy-preserving WiFi sensing systems in health and broader human-centric applications. 

---
# Extracting Research Instruments from Educational Literature Using LLMs 

**Authors**: Jiseung Yoo, Curran Mahowald, Meiyu Li, Wei Ai  

**Link**: [PDF](https://arxiv.org/pdf/2505.21855)  

**Abstract**: Large Language Models (LLMs) are transforming information extraction from academic literature, offering new possibilities for knowledge management. This study presents an LLM-based system designed to extract detailed information about research instruments used in the education field, including their names, types, target respondents, measured constructs, and outcomes. Using multi-step prompting and a domain-specific data schema, it generates structured outputs optimized for educational research. Our evaluation shows that this system significantly outperforms other approaches, particularly in identifying instrument names and detailed information. This demonstrates the potential of LLM-powered information extraction in educational contexts, offering a systematic way to organize research instrument information. The ability to aggregate such information at scale enhances accessibility for researchers and education leaders, facilitating informed decision-making in educational research and policy. 

---
# Rethinking Gradient-based Adversarial Attacks on Point Cloud Classification 

**Authors**: Jun Chen, Xinke Li, Mingyue Xu, Tianrui Li, Chongshou Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.21854)  

**Abstract**: Gradient-based adversarial attacks have become a dominant approach for evaluating the robustness of point cloud classification models. However, existing methods often rely on uniform update rules that fail to consider the heterogeneous nature of point clouds, resulting in excessive and perceptible perturbations. In this paper, we rethink the design of gradient-based attacks by analyzing the limitations of conventional gradient update mechanisms and propose two new strategies to improve both attack effectiveness and imperceptibility. First, we introduce WAAttack, a novel framework that incorporates weighted gradients and an adaptive step-size strategy to account for the non-uniform contribution of points during optimization. This approach enables more targeted and subtle perturbations by dynamically adjusting updates according to the local structure and sensitivity of each point. Second, we propose SubAttack, a complementary strategy that decomposes the point cloud into subsets and focuses perturbation efforts on structurally critical regions. Together, these methods represent a principled rethinking of gradient-based adversarial attacks for 3D point cloud classification. Extensive experiments demonstrate that our approach outperforms state-of-the-art baselines in generating highly imperceptible adversarial examples. Code will be released upon paper acceptance. 

---
# A Provable Approach for End-to-End Safe Reinforcement Learning 

**Authors**: Akifumi Wachi, Kohei Miyaguchi, Takumi Tanabe, Rei Sato, Youhei Akimoto  

**Link**: [PDF](https://arxiv.org/pdf/2505.21852)  

**Abstract**: A longstanding goal in safe reinforcement learning (RL) is a method to ensure the safety of a policy throughout the entire process, from learning to operation. However, existing safe RL paradigms inherently struggle to achieve this objective. We propose a method, called Provably Lifetime Safe RL (PLS), that integrates offline safe RL with safe policy deployment to address this challenge. Our proposed method learns a policy offline using return-conditioned supervised learning and then deploys the resulting policy while cautiously optimizing a limited set of parameters, known as target returns, using Gaussian processes (GPs). Theoretically, we justify the use of GPs by analyzing the mathematical relationship between target and actual returns. We then prove that PLS finds near-optimal target returns while guaranteeing safety with high probability. Empirically, we demonstrate that PLS outperforms baselines both in safety and reward performance, thereby achieving the longstanding goal to obtain high rewards while ensuring the safety of a policy throughout the lifetime from learning to operation. 

---
# Streaming Flow Policy: Simplifying diffusion$/$flow-matching policies by treating action trajectories as flow trajectories 

**Authors**: Sunshine Jiang, Xiaolin Fang, Nicholas Roy, Tomás Lozano-Pérez, Leslie Pack Kaelbling, Siddharth Ancha  

**Link**: [PDF](https://arxiv.org/pdf/2505.21851)  

**Abstract**: Recent advances in diffusion$/$flow-matching policies have enabled imitation learning of complex, multi-modal action trajectories. However, they are computationally expensive because they sample a trajectory of trajectories: a diffusion$/$flow trajectory of action trajectories. They discard intermediate action trajectories, and must wait for the sampling process to complete before any actions can be executed on the robot. We simplify diffusion$/$flow policies by treating action trajectories as flow trajectories. Instead of starting from pure noise, our algorithm samples from a narrow Gaussian around the last action. Then, it incrementally integrates a velocity field learned via flow matching to produce a sequence of actions that constitute a single trajectory. This enables actions to be streamed to the robot on-the-fly during the flow sampling process, and is well-suited for receding horizon policy execution. Despite streaming, our method retains the ability to model multi-modal behavior. We train flows that stabilize around demonstration trajectories to reduce distribution shift and improve imitation learning performance. Streaming flow policy outperforms prior methods while enabling faster policy execution and tighter sensorimotor loops for learning-based robot control. Project website: this https URL 

---
# Beyond Perception: Evaluating Abstract Visual Reasoning through Multi-Stage Task 

**Authors**: Yanbei Jiang, Yihao Ding, Chao Lei, Jiayang Ao, Jey Han Lau, Krista A. Ehinger  

**Link**: [PDF](https://arxiv.org/pdf/2505.21850)  

**Abstract**: Current Multimodal Large Language Models (MLLMs) excel in general visual reasoning but remain underexplored in Abstract Visual Reasoning (AVR), which demands higher-order reasoning to identify abstract rules beyond simple perception. Existing AVR benchmarks focus on single-step reasoning, emphasizing the end result but neglecting the multi-stage nature of reasoning process. Past studies found MLLMs struggle with these benchmarks, but it doesn't explain how they fail. To address this gap, we introduce MultiStAR, a Multi-Stage AVR benchmark, based on RAVEN, designed to assess reasoning across varying levels of complexity. Additionally, existing metrics like accuracy only focus on the final outcomes while do not account for the correctness of intermediate steps. Therefore, we propose a novel metric, MSEval, which considers the correctness of intermediate steps in addition to the final outcomes. We conduct comprehensive experiments on MultiStAR using 17 representative close-source and open-source MLLMs. The results reveal that while existing MLLMs perform adequately on basic perception tasks, they continue to face challenges in more complex rule detection stages. 

---
# Xinyu AI Search: Enhanced Relevance and Comprehensive Results with Rich Answer Presentations 

**Authors**: Bo Tang, Junyi Zhu, Chenyang Xi, Yunhang Ge, Jiahao Wu, Yuchen Feng, Yijun Niu, Wenqiang Wei, Yu Yu, Chunyu Li, Zehao Lin, Hao Wu, Ning Liao, Yebin Yang, Jiajia Wang, Zhiyu Li, Feiyu Xiong, Jingrun Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21849)  

**Abstract**: Traditional search engines struggle to synthesize fragmented information for complex queries, while generative AI search engines face challenges in relevance, comprehensiveness, and presentation. To address these limitations, we introduce Xinyu AI Search, a novel system that incorporates a query-decomposition graph to dynamically break down complex queries into sub-queries, enabling stepwise retrieval and generation. Our retrieval pipeline enhances diversity through multi-source aggregation and query expansion, while filtering and re-ranking strategies optimize passage relevance. Additionally, Xinyu AI Search introduces a novel approach for fine-grained, precise built-in citation and innovates in result presentation by integrating timeline visualization and textual-visual choreography. Evaluated on recent real-world queries, Xinyu AI Search outperforms eight existing technologies in human assessments, excelling in relevance, comprehensiveness, and insightfulness. Ablation studies validate the necessity of its key sub-modules. Our work presents the first comprehensive framework for generative AI search engines, bridging retrieval, generation, and user-centric presentation. 

---
# RePaViT: Scalable Vision Transformer Acceleration via Structural Reparameterization on Feedforward Network Layers 

**Authors**: Xuwei Xu, Yang Li, Yudong Chen, Jiajun Liu, Sen Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21847)  

**Abstract**: We reveal that feedforward network (FFN) layers, rather than attention layers, are the primary contributors to Vision Transformer (ViT) inference latency, with their impact signifying as model size increases. This finding highlights a critical opportunity for optimizing the efficiency of large-scale ViTs by focusing on FFN layers. In this work, we propose a novel channel idle mechanism that facilitates post-training structural reparameterization for efficient FFN layers during testing. Specifically, a set of feature channels remains idle and bypasses the nonlinear activation function in each FFN layer, thereby forming a linear pathway that enables structural reparameterization during inference. This mechanism results in a family of ReParameterizable Vision Transformers (RePaViTs), which achieve remarkable latency reductions with acceptable sacrifices (sometimes gains) in accuracy across various ViTs. The benefits of our method scale consistently with model sizes, demonstrating greater speed improvements and progressively narrowing accuracy gaps or even higher accuracies on larger models. In particular, RePa-ViT-Large and RePa-ViT-Huge enjoy 66.8% and 68.7% speed-ups with +1.7% and +1.1% higher top-1 accuracies under the same training strategy, respectively. RePaViT is the first to employ structural reparameterization on FFN layers to expedite ViTs to our best knowledge, and we believe that it represents an auspicious direction for efficient ViTs. Source code is available at this https URL. 

---
# An Optimistic Algorithm for online CMDPS with Anytime Adversarial Constraints 

**Authors**: Jiahui Zhu, Kihyun Yu, Dabeen Lee, Xin Liu, Honghao Wei  

**Link**: [PDF](https://arxiv.org/pdf/2505.21841)  

**Abstract**: Online safe reinforcement learning (RL) plays a key role in dynamic environments, with applications in autonomous driving, robotics, and cybersecurity. The objective is to learn optimal policies that maximize rewards while satisfying safety constraints modeled by constrained Markov decision processes (CMDPs). Existing methods achieve sublinear regret under stochastic constraints but often fail in adversarial settings, where constraints are unknown, time-varying, and potentially adversarially designed. In this paper, we propose the Optimistic Mirror Descent Primal-Dual (OMDPD) algorithm, the first to address online CMDPs with anytime adversarial constraints. OMDPD achieves optimal regret O(sqrt(K)) and strong constraint violation O(sqrt(K)) without relying on Slater's condition or the existence of a strictly known safe policy. We further show that access to accurate estimates of rewards and transitions can further improve these bounds. Our results offer practical guarantees for safe decision-making in adversarial environments. 

---
# Nonadaptive Output Regulation of Second-Order Nonlinear Uncertain Systems 

**Authors**: Maobin Lu, Martin Guay, Telema Harry, Shimin Wang, Jordan Cooper  

**Link**: [PDF](https://arxiv.org/pdf/2505.21838)  

**Abstract**: This paper investigates the robust output regulation problem of second-order nonlinear uncertain systems with an unknown exosystem. Instead of the adaptive control approach, this paper resorts to a robust control methodology to solve the problem and thus avoid the bursting phenomenon. In particular, this paper constructs generic internal models for the steady-state state and input variables of the system. By introducing a coordinate transformation, this paper converts the robust output regulation problem into a nonadaptive stabilization problem of an augmented system composed of the second-order nonlinear uncertain system and the generic internal models. Then, we design the stabilization control law and construct a strict Lyapunov function that guarantees the robustness with respect to unmodeled disturbances. The analysis shows that the output zeroing manifold of the augmented system can be made attractive by the proposed nonadaptive control law, which solves the robust output regulation problem. Finally, we demonstrate the effectiveness of the proposed nonadaptive internal model approach by its application to the control of the Duffing system. 

---
# TuneComp: Joint Fine-tuning and Compression for Large Foundation Models 

**Authors**: Xiangyu Chen, Jing Liu, Ye Wang, Matthew Brand, Wang, Toshiaki Koike-Akino  

**Link**: [PDF](https://arxiv.org/pdf/2505.21835)  

**Abstract**: To reduce model size during post-training, compression methods, including knowledge distillation, low-rank approximation, and pruning, are often applied after fine-tuning the model. However, sequential fine-tuning and compression sacrifices performance, while creating a larger than necessary model as an intermediate step. In this work, we aim to reduce this gap, by directly constructing a smaller model while guided by the downstream task. We propose to jointly fine-tune and compress the model by gradually distilling it to a pruned low-rank structure. Experiments demonstrate that joint fine-tuning and compression significantly outperforms other sequential compression methods. 

---
# Music Source Restoration 

**Authors**: Yongyi Zang, Zheqi Dai, Mark D. Plumbley, Qiuqiang Kong  

**Link**: [PDF](https://arxiv.org/pdf/2505.21827)  

**Abstract**: We introduce Music Source Restoration (MSR), a novel task addressing the gap between idealized source separation and real-world music production. Current Music Source Separation (MSS) approaches assume mixtures are simple sums of sources, ignoring signal degradations employed during music production like equalization, compression, and reverb. MSR models mixtures as degraded sums of individually degraded sources, with the goal of recovering original, undegraded signals. Due to the lack of data for MSR, we present RawStems, a dataset annotation of 578 songs with unprocessed source signals organized into 8 primary and 17 secondary instrument groups, totaling 354.13 hours. To the best of our knowledge, RawStems is the first dataset that contains unprocessed music stems with hierarchical categories. We consider spectral filtering, dynamic range compression, harmonic distortion, reverb and lossy codec as possible degradations, and establish U-Former as a baseline method, demonstrating the feasibility of MSR on our dataset. We release the RawStems dataset annotations, degradation simulation pipeline, training code and pre-trained models to be publicly available. 

---
# Let Me Think! A Long Chain-of-Thought Can Be Worth Exponentially Many Short Ones 

**Authors**: Parsa Mirtaheri, Ezra Edelman, Samy Jelassi, Eran Malach, Enric Boix-Adsera  

**Link**: [PDF](https://arxiv.org/pdf/2505.21825)  

**Abstract**: Inference-time computation has emerged as a promising scaling axis for improving large language model reasoning. However, despite yielding impressive performance, the optimal allocation of inference-time computation remains poorly understood. A central question is whether to prioritize sequential scaling (e.g., longer chains of thought) or parallel scaling (e.g., majority voting across multiple short chains of thought). In this work, we seek to illuminate the landscape of test-time scaling by demonstrating the existence of reasoning settings where sequential scaling offers an exponential advantage over parallel scaling. These settings are based on graph connectivity problems in challenging distributions of graphs. We validate our theoretical findings with comprehensive experiments across a range of language models, including models trained from scratch for graph connectivity with different chain of thought strategies as well as large reasoning models. 

---
# Scientific Paper Retrieval with LLM-Guided Semantic-Based Ranking 

**Authors**: Yunyi Zhang, Ruozhen Yang, Siqi Jiao, SeongKu Kang, Jiawei Han  

**Link**: [PDF](https://arxiv.org/pdf/2505.21815)  

**Abstract**: Scientific paper retrieval is essential for supporting literature discovery and research. While dense retrieval methods demonstrate effectiveness in general-purpose tasks, they often fail to capture fine-grained scientific concepts that are essential for accurate understanding of scientific queries. Recent studies also use large language models (LLMs) for query understanding; however, these methods often lack grounding in corpus-specific knowledge and may generate unreliable or unfaithful content. To overcome these limitations, we propose SemRank, an effective and efficient paper retrieval framework that combines LLM-guided query understanding with a concept-based semantic index. Each paper is indexed using multi-granular scientific concepts, including general research topics and detailed key phrases. At query time, an LLM identifies core concepts derived from the corpus to explicitly capture the query's information need. These identified concepts enable precise semantic matching, significantly enhancing retrieval accuracy. Experiments show that SemRank consistently improves the performance of various base retrievers, surpasses strong existing LLM-based baselines, and remains highly efficient. 

---
# Revisiting Self-attention for Cross-domain Sequential Recommendation 

**Authors**: Clark Mingxuan Ju, Leonardo Neves, Bhuvesh Kumar, Liam Collins, Tong Zhao, Yuwei Qiu, Qing Dou, Sohail Nizam, Sen Yang, Neil Shah  

**Link**: [PDF](https://arxiv.org/pdf/2505.21811)  

**Abstract**: Sequential recommendation is a popular paradigm in modern recommender systems. In particular, one challenging problem in this space is cross-domain sequential recommendation (CDSR), which aims to predict future behaviors given user interactions across multiple domains. Existing CDSR frameworks are mostly built on the self-attention transformer and seek to improve by explicitly injecting additional domain-specific components (e.g. domain-aware module blocks). While these additional components help, we argue they overlook the core self-attention module already present in the transformer, a naturally powerful tool to learn correlations among behaviors. In this work, we aim to improve the CDSR performance for simple models from a novel perspective of enhancing the self-attention. Specifically, we introduce a Pareto-optimal self-attention and formulate the cross-domain learning as a multi-objective problem, where we optimize the recommendation task while dynamically minimizing the cross-domain attention scores. Our approach automates knowledge transfer in CDSR (dubbed as AutoCDSR) -- it not only mitigates negative transfer but also encourages complementary knowledge exchange among auxiliary domains. Based on the idea, we further introduce AutoCDSR+, a more performant variant with slight additional cost. Our proposal is easy to implement and works as a plug-and-play module that can be incorporated into existing transformer-based recommenders. Besides flexibility, it is practical to deploy because it brings little extra computational overheads without heavy hyper-parameter tuning. AutoCDSR on average improves Recall@10 for SASRec and Bert4Rec by 9.8% and 16.0% and NDCG@10 by 12.0% and 16.7%, respectively. Code is available at this https URL. 

---
# Multimodal Federated Learning: A Survey through the Lens of Different FL Paradigms 

**Authors**: Yuanzhe Peng, Jieming Bian, Lei Wang, Yin Huang, Jie Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21792)  

**Abstract**: Multimodal Federated Learning (MFL) lies at the intersection of two pivotal research areas: leveraging complementary information from multiple modalities to improve downstream inference performance and enabling distributed training to enhance efficiency and preserve privacy. Despite the growing interest in MFL, there is currently no comprehensive taxonomy that organizes MFL through the lens of different Federated Learning (FL) paradigms. This perspective is important because multimodal data introduces distinct challenges across various FL settings. These challenges, including modality heterogeneity, privacy heterogeneity, and communication inefficiency, are fundamentally different from those encountered in traditional unimodal or non-FL scenarios. In this paper, we systematically examine MFL within the context of three major FL paradigms: horizontal FL (HFL), vertical FL (VFL), and hybrid FL. For each paradigm, we present the problem formulation, review representative training algorithms, and highlight the most prominent challenge introduced by multimodal data in distributed settings. We also discuss open challenges and provide insights for future research. By establishing this taxonomy, we aim to uncover the novel challenges posed by multimodal data from the perspective of different FL paradigms and to offer a new lens through which to understand and advance the development of MFL. 

---
# VeriTrail: Closed-Domain Hallucination Detection with Traceability 

**Authors**: Dasha Metropolitansky, Jonathan Larson  

**Link**: [PDF](https://arxiv.org/pdf/2505.21786)  

**Abstract**: Even when instructed to adhere to source material, Language Models often generate unsubstantiated content - a phenomenon known as "closed-domain hallucination." This risk is amplified in processes with multiple generative steps (MGS), compared to processes with a single generative step (SGS). However, due to the greater complexity of MGS processes, we argue that detecting hallucinations in their final outputs is necessary but not sufficient: it is equally important to trace where hallucinated content was likely introduced and how faithful content may have been derived from the source through intermediate outputs. To address this need, we present VeriTrail, the first closed-domain hallucination detection method designed to provide traceability for both MGS and SGS processes. We also introduce the first datasets to include all intermediate outputs as well as human annotations of final outputs' faithfulness for their respective MGS processes. We demonstrate that VeriTrail outperforms baseline methods on both datasets. 

---
# DualSchool: How Reliable are LLMs for Optimization Education? 

**Authors**: Michael Klamkin, Arnaud Deza, Sikai Cheng, Haoruo Zhao, Pascal Van Hentenryck  

**Link**: [PDF](https://arxiv.org/pdf/2505.21775)  

**Abstract**: Consider the following task taught in introductory optimization courses which addresses challenges articulated by the community at the intersection of (generative) AI and OR: generate the dual of a linear program. LLMs, being trained at web-scale, have the conversion process and many instances of Primal to Dual Conversion (P2DC) at their disposal. Students may thus reasonably expect that LLMs would perform well on the P2DC task. To assess this expectation, this paper introduces DualSchool, a comprehensive framework for generating and verifying P2DC instances. The verification procedure of DualSchool uses the Canonical Graph Edit Distance, going well beyond existing evaluation methods for optimization models, which exhibit many false positives and negatives when applied to P2DC. Experiments performed by DualSchool reveal interesting findings. Although LLMs can recite the conversion procedure accurately, state-of-the-art open LLMs fail to consistently produce correct duals. This finding holds even for the smallest two-variable instances and for derivative tasks, such as correctness, verification, and error classification. The paper also discusses the implications for educators, students, and the development of large reasoning systems. 

---
# MMTBENCH: A Unified Benchmark for Complex Multimodal Table Reasoning 

**Authors**: Prasham Yatinkumar Titiya, Jainil Trivedi, Chitta Baral, Vivek Gupta  

**Link**: [PDF](https://arxiv.org/pdf/2505.21771)  

**Abstract**: Multimodal tables those that integrate semi structured data with visual elements such as charts and maps are ubiquitous across real world domains, yet they pose a formidable challenge to current vision language models (VLMs). While Large Language models (LLMs) and VLMs have demonstrated strong capabilities in text and image understanding, their performance on complex, real world multimodal table reasoning remains unexplored. To bridge this gap, we introduce MMTBENCH (Multimodal Table Benchmark), a benchmark consisting of 500 real world multimodal tables drawn from diverse real world sources, with a total of 4021 question answer pairs. MMTBENCH questions cover four question types (Explicit, Implicit, Answer Mention, and Visual Based), five reasoning types (Mathematical, Extrema Identification, Fact Verification, Vision Based, and Others), and eight table types (Single/Multiple Entity, Maps and Charts with Entities, Single/Multiple Charts, Maps, and Visualizations). Extensive evaluation of state of the art models on all types reveals substantial performance gaps, particularly on questions requiring visual-based reasoning and multi-step inference. These findings show the urgent need for improved architectures that more tightly integrate vision and language processing. By providing a challenging, high-quality resource that mirrors the complexity of real-world tasks, MMTBENCH underscores its value as a resource for future research on multimodal tables. 

---
# FRAMES-VQA: Benchmarking Fine-Tuning Robustness across Multi-Modal Shifts in Visual Question Answering 

**Authors**: Chengyue Huang, Brisa Maneechotesuwan, Shivang Chopra, Zsolt Kira  

**Link**: [PDF](https://arxiv.org/pdf/2505.21755)  

**Abstract**: Visual question answering (VQA) systems face significant challenges when adapting to real-world data shifts, especially in multi-modal contexts. While robust fine-tuning strategies are essential for maintaining performance across in-distribution (ID) and out-of-distribution (OOD) scenarios, current evaluation settings are primarily unimodal or particular to some types of OOD, offering limited insight into the complexities of multi-modal contexts. In this work, we propose a new benchmark FRAMES-VQA (Fine-Tuning Robustness across Multi-Modal Shifts in VQA) for evaluating robust fine-tuning for VQA tasks. We utilize ten existing VQA benchmarks, including VQAv2, IV-VQA, VQA-CP, OK-VQA and others, and categorize them into ID, near and far OOD datasets covering uni-modal, multi-modal and adversarial distribution shifts. We first conduct a comprehensive comparison of existing robust fine-tuning methods. We then quantify the distribution shifts by calculating the Mahalanobis distance using uni-modal and multi-modal embeddings extracted from various models. Further, we perform an extensive analysis to explore the interactions between uni- and multi-modal shifts as well as modality importance for ID and OOD samples. These analyses offer valuable guidance on developing more robust fine-tuning methods to handle multi-modal distribution shifts. The code is available at this https URL . 

---
# Learning to See More: UAS-Guided Super-Resolution of Satellite Imagery for Precision Agriculture 

**Authors**: Arif Masrur, Peder A. Olsen, Paul R. Adler, Carlan Jackson, Matthew W. Myers, Nathan Sedghi, Ray R. Weil  

**Link**: [PDF](https://arxiv.org/pdf/2505.21746)  

**Abstract**: Unmanned Aircraft Systems (UAS) and satellites are key data sources for precision agriculture, yet each presents trade-offs. Satellite data offer broad spatial, temporal, and spectral coverage but lack the resolution needed for many precision farming applications, while UAS provide high spatial detail but are limited by coverage and cost, especially for hyperspectral data. This study presents a novel framework that fuses satellite and UAS imagery using super-resolution methods. By integrating data across spatial, spectral, and temporal domains, we leverage the strengths of both platforms cost-effectively. We use estimation of cover crop biomass and nitrogen (N) as a case study to evaluate our approach. By spectrally extending UAS RGB data to the vegetation red edge and near-infrared regions, we generate high-resolution Sentinel-2 imagery and improve biomass and N estimation accuracy by 18% and 31%, respectively. Our results show that UAS data need only be collected from a subset of fields and time points. Farmers can then 1) enhance the spectral detail of UAS RGB imagery; 2) increase the spatial resolution by using satellite data; and 3) extend these enhancements spatially and across the growing season at the frequency of the satellite flights. Our SRCNN-based spectral extension model shows considerable promise for model transferability over other cropping systems in the Upper and Lower Chesapeake Bay regions. Additionally, it remains effective even when cloud-free satellite data are unavailable, relying solely on the UAS RGB input. The spatial extension model produces better biomass and N predictions than models built on raw UAS RGB images. Once trained with targeted UAS RGB data, the spatial extension model allows farmers to stop repeated UAS flights. While we introduce super-resolution advances, the core contribution is a lightweight and scalable system for affordable on-farm use. 

---
# Simulating the Unseen: Crash Prediction Must Learn from What Did Not Happen 

**Authors**: Zihao Li, Xinyuan Cao, Xiangbo Gao, Kexin Tian, Keshu Wu, Mohammad Anis, Hao Zhang, Keke Long, Jiwan Jiang, Xiaopeng Li, Yunlong Zhang, Tianbao Yang, Dominique Lord, Zhengzhong Tu, Yang Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2505.21743)  

**Abstract**: Traffic safety science has long been hindered by a fundamental data paradox: the crashes we most wish to prevent are precisely those events we rarely observe. Existing crash-frequency models and surrogate safety metrics rely heavily on sparse, noisy, and under-reported records, while even sophisticated, high-fidelity simulations undersample the long-tailed situations that trigger catastrophic outcomes such as fatalities. We argue that the path to achieving Vision Zero, i.e., the complete elimination of traffic fatalities and severe injuries, requires a paradigm shift from traditional crash-only learning to a new form of counterfactual safety learning: reasoning not only about what happened, but also about the vast set of plausible yet perilous scenarios that could have happened under slightly different circumstances. To operationalize this shift, our proposed agenda bridges macro to micro. Guided by crash-rate priors, generative scene engines, diverse driver models, and causal learning, near-miss events are synthesized and explained. A crash-focused digital twin testbed links micro scenes to macro patterns, while a multi-objective validator ensures that simulations maintain statistical realism. This pipeline transforms sparse crash data into rich signals for crash prediction, enabling the stress-testing of vehicles, roads, and policies before deployment. By learning from crashes that almost happened, we can shift traffic safety from reactive forensics to proactive prevention, advancing Vision Zero. 

---
# Counterfactual Simulatability of LLM Explanations for Generation Tasks 

**Authors**: Marvin Limpijankit, Yanda Chen, Melanie Subbiah, Nicholas Deas, Kathleen McKeown  

**Link**: [PDF](https://arxiv.org/pdf/2505.21740)  

**Abstract**: LLMs can be unpredictable, as even slight alterations to the prompt can cause the output to change in unexpected ways. Thus, the ability of models to accurately explain their behavior is critical, especially in high-stakes settings. One approach for evaluating explanations is counterfactual simulatability, how well an explanation allows users to infer the model's output on related counterfactuals. Counterfactual simulatability has been previously studied for yes/no question answering tasks. We provide a general framework for extending this method to generation tasks, using news summarization and medical suggestion as example use cases. We find that while LLM explanations do enable users to better predict LLM outputs on counterfactuals in the summarization setting, there is significant room for improvement for medical suggestion. Furthermore, our results suggest that the evaluation for counterfactual simulatability may be more appropriate for skill-based tasks as opposed to knowledge-based tasks. 

---
# Deep Reinforcement Learning Agents are not even close to Human Intelligence 

**Authors**: Quentin Delfosse, Jannis Blüml, Fabian Tatai, Théo Vincent, Bjarne Gregori, Elisabeth Dillies, Jan Peters, Constantin Rothkopf, Kristian Kersting  

**Link**: [PDF](https://arxiv.org/pdf/2505.21731)  

**Abstract**: Deep reinforcement learning (RL) agents achieve impressive results in a wide variety of tasks, but they lack zero-shot adaptation capabilities. While most robustness evaluations focus on tasks complexifications, for which human also struggle to maintain performances, no evaluation has been performed on tasks simplifications. To tackle this issue, we introduce HackAtari, a set of task variations of the Arcade Learning Environments. We use it to demonstrate that, contrary to humans, RL agents systematically exhibit huge performance drops on simpler versions of their training tasks, uncovering agents' consistent reliance on shortcuts. Our analysis across multiple algorithms and architectures highlights the persistent gap between RL agents and human behavioral intelligence, underscoring the need for new benchmarks and methodologies that enforce systematic generalization testing beyond static evaluation protocols. Training and testing in the same environment is not enough to obtain agents equipped with human-like intelligence. 

---
# OmniResponse: Online Multimodal Conversational Response Generation in Dyadic Interactions 

**Authors**: Cheng Luo, Jianghui Wang, Bing Li, Siyang Song, Bernard Ghanem  

**Link**: [PDF](https://arxiv.org/pdf/2505.21724)  

**Abstract**: In this paper, we introduce Online Multimodal Conversational Response Generation (OMCRG), a novel task that aims to online generate synchronized verbal and non-verbal listener feedback, conditioned on the speaker's multimodal input. OMCRG reflects natural dyadic interactions and poses new challenges in achieving synchronization between the generated audio and facial responses of the listener. To address these challenges, we innovatively introduce text as an intermediate modality to bridge the audio and facial responses. We hence propose OmniResponse, a Multimodal Large Language Model (MLLM) that autoregressively generates high-quality multi-modal listener responses. OmniResponse leverages a pretrained LLM enhanced with two novel components: Chrono-Text, which temporally anchors generated text tokens, and TempoVoice, a controllable online TTS module that produces speech synchronized with facial reactions. To support further OMCRG research, we present ResponseNet, a new dataset comprising 696 high-quality dyadic interactions featuring synchronized split-screen videos, multichannel audio, transcripts, and facial behavior annotations. Comprehensive evaluations conducted on ResponseNet demonstrate that OmniResponse significantly outperforms baseline models in terms of semantic speech content, audio-visual synchronization, and generation quality. 

---
# Saddle-To-Saddle Dynamics in Deep ReLU Networks: Low-Rank Bias in the First Saddle Escape 

**Authors**: Ioannis Bantzis, James B. Simon, Arthur Jacot  

**Link**: [PDF](https://arxiv.org/pdf/2505.21722)  

**Abstract**: When a deep ReLU network is initialized with small weights, GD is at first dominated by the saddle at the origin in parameter space. We study the so-called escape directions, which play a similar role as the eigenvectors of the Hessian for strict saddles. We show that the optimal escape direction features a low-rank bias in its deeper layers: the first singular value of the $\ell$-th layer weight matrix is at least $\ell^{\frac{1}{4}}$ larger than any other singular value. We also prove a number of related results about these escape directions. We argue that this result is a first step in proving Saddle-to-Saddle dynamics in deep ReLU networks, where GD visits a sequence of saddles with increasing bottleneck rank. 

---
# Responsible Data Stewardship: Generative AI and the Digital Waste Problem 

**Authors**: Vanessa Utz  

**Link**: [PDF](https://arxiv.org/pdf/2505.21720)  

**Abstract**: As generative AI systems become widely adopted, they enable unprecedented creation levels of synthetic data across text, images, audio, and video modalities. While research has addressed the energy consumption of model training and inference, a critical sustainability challenge remains understudied: digital waste. This term refers to stored data that consumes resources without serving a specific (and/or immediate) purpose. This paper presents this terminology in the AI context and introduces digital waste as an ethical imperative within (generative) AI development, positioning environmental sustainability as core for responsible innovation. Drawing from established digital resource management approaches, we examine how other disciplines manage digital waste and identify transferable approaches for the AI community. We propose specific recommendations encompassing re-search directions, technical interventions, and cultural shifts to mitigate the environmental consequences of in-definite data storage. By expanding AI ethics beyond immediate concerns like bias and privacy to include inter-generational environmental justice, this work contributes to a more comprehensive ethical framework that considers the complete lifecycle impact of generative AI systems. 

---
# Scaling Up Liquid-Resistance Liquid-Capacitance Networks for Efficient Sequence Modeling 

**Authors**: Mónika Farsang, Ramin Hasani, Radu Grosu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21717)  

**Abstract**: We present LrcSSM, a \textit{nonlinear} recurrent model that processes long sequences as fast as today's linear state-space layers. By forcing the state-transition matrix to be diagonal and learned at every step, the full sequence can be solved in parallel with a single prefix-scan, giving $\mathcal{O}(TD)$ time and memory and only $\mathcal{O}(\log T)$ sequential depth, for input-sequence length $T$ and a state dimension $D$. Moreover, LrcSSM offers a formal gradient-stability guarantee that other input-varying systems such as Liquid-S4 and Mamba do not provide. Lastly, for network depth $L$, as the forward and backward passes cost $\Theta(T\,D\,L)$ FLOPs, with its low sequential depth and parameter count $\Theta(D\,L)$, the model follows the compute-optimal scaling law regime ($\beta \approx 0.42$) recently observed for Mamba, outperforming quadratic-attention Transformers at equal compute while avoiding the memory overhead of FFT-based long convolutions. We show that on a series of long-range forecasting tasks, LrcSSM outperforms LRU, S5 and Mamba. 

---
# Privacy-Preserving Chest X-ray Report Generation via Multimodal Federated Learning with ViT and GPT-2 

**Authors**: Md. Zahid Hossain, Mustofa Ahmed, Most. Sharmin Sultana Samu, Md. Rakibul Islam  

**Link**: [PDF](https://arxiv.org/pdf/2505.21715)  

**Abstract**: The automated generation of radiology reports from chest X-ray images holds significant promise in enhancing diagnostic workflows while preserving patient privacy. Traditional centralized approaches often require sensitive data transfer, posing privacy concerns. To address this, the study proposes a Multimodal Federated Learning framework for chest X-ray report generation using the IU-Xray dataset. The system utilizes a Vision Transformer (ViT) as the encoder and GPT-2 as the report generator, enabling decentralized training without sharing raw data. Three Federated Learning (FL) aggregation strategies: FedAvg, Krum Aggregation and a novel Loss-aware Federated Averaging (L-FedAvg) were evaluated. Among these, Krum Aggregation demonstrated superior performance across lexical and semantic evaluation metrics such as ROUGE, BLEU, BERTScore and RaTEScore. The results show that FL can match or surpass centralized models in generating clinically relevant and semantically rich radiology reports. This lightweight and privacy-preserving framework paves the way for collaborative medical AI development without compromising data confidentiality. 

---
# A Joint Reconstruction-Triplet Loss Autoencoder Approach Towards Unseen Attack Detection in IoV Networks 

**Authors**: Julia Boone, Tolunay Seyfi, Fatemeh Afghah  

**Link**: [PDF](https://arxiv.org/pdf/2505.21703)  

**Abstract**: Internet of Vehicles (IoV) systems, while offering significant advancements in transportation efficiency and safety, introduce substantial security vulnerabilities due to their highly interconnected nature. These dynamic systems produce massive amounts of data between vehicles, infrastructure, and cloud services and present a highly distributed framework with a wide attack surface. In considering network-centered attacks on IoV systems, attacks such as Denial-of-Service (DoS) can prohibit the communication of essential physical traffic safety information between system elements, illustrating that the security concerns for these systems go beyond the traditional confidentiality, integrity, and availability concerns of enterprise systems. Given the complexity and volume of data generated by IoV systems, traditional security mechanisms are often inadequate for accurately detecting sophisticated and evolving cyberattacks. Here, we present an unsupervised autoencoder method trained entirely on benign network data for the purpose of unseen attack detection in IoV networks. We leverage a weighted combination of reconstruction and triplet margin loss to guide the autoencoder training and develop a diverse representation of the benign training set. We conduct extensive experiments on recent network intrusion datasets from two different application domains, industrial IoT and home IoT, that represent the modern IoV task. We show that our method performs robustly for all unseen attack types, with roughly 99% accuracy on benign data and between 97% and 100% performance on anomaly data. We extend these results to show that our model is adaptable through the use of transfer learning, achieving similarly high results while leveraging domain features from one domain to another. 

---
# STA-Risk: A Deep Dive of Spatio-Temporal Asymmetries for Breast Cancer Risk Prediction 

**Authors**: Zhengbo Zhou, Dooman Arefan, Margarita Zuley, Jules Sumkin, Shandong Wu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21699)  

**Abstract**: Predicting the risk of developing breast cancer is an important clinical tool to guide early intervention and tailoring personalized screening strategies. Early risk models have limited performance and recently machine learning-based analysis of mammogram images showed encouraging risk prediction effects. These models however are limited to the use of a single exam or tend to overlook nuanced breast tissue evolvement in spatial and temporal details of longitudinal imaging exams that are indicative of breast cancer risk. In this paper, we propose STA-Risk (Spatial and Temporal Asymmetry-based Risk Prediction), a novel Transformer-based model that captures fine-grained mammographic imaging evolution simultaneously from bilateral and longitudinal asymmetries for breast cancer risk prediction. STA-Risk is innovative by the side encoding and temporal encoding to learn spatial-temporal asymmetries, regulated by a customized asymmetry loss. We performed extensive experiments with two independent mammogram datasets and achieved superior performance than four representative SOTA models for 1- to 5-year future risk prediction. Source codes will be released upon publishing of the paper. 

---
# LLMPR: A Novel LLM-Driven Transfer Learning based Petition Ranking Model 

**Authors**: Avijit Gayen, Somyajit Chakraborty, Mainak Sen, Soham Paul, Angshuman Jana  

**Link**: [PDF](https://arxiv.org/pdf/2505.21689)  

**Abstract**: The persistent accumulation of unresolved legal cases, especially within the Indian judiciary, significantly hampers the timely delivery of justice. Manual methods of prioritizing petitions are often prone to inefficiencies and subjective biases further exacerbating delays. To address this issue, we propose LLMPR (Large Language Model-based Petition Ranking), an automated framework that utilizes transfer learning and machine learning to assign priority rankings to legal petitions based on their contextual urgency. Leveraging the ILDC dataset comprising 7,593 annotated petitions, we process unstructured legal text and extract features through various embedding techniques, including DistilBERT, LegalBERT, and MiniLM. These textual embeddings are combined with quantitative indicators such as gap days, rank scores, and word counts to train multiple machine learning models, including Random Forest, Decision Tree, XGBoost, LightGBM, and CatBoost. Our experiments demonstrate that Random Forest and Decision Tree models yield superior performance, with accuracy exceeding 99% and a Spearman rank correlation of 0.99. Notably, models using only numerical features achieve nearly optimal ranking results (R2 = 0.988, \r{ho} = 0.998), while LLM-based embeddings offer only marginal gains. These findings suggest that automated petition ranking can effectively streamline judicial workflows, reduce case backlog, and improve fairness in legal prioritization. 

---
# multivariateGPT: a decoder-only transformer for multivariate categorical and numeric data 

**Authors**: Andrew J. Loza, Jun Yup Kim, Shangzheng Song, Yihang Liu, Joseph J. Y. Sung, R Andrew Taylor, Dennis L. Shung  

**Link**: [PDF](https://arxiv.org/pdf/2505.21680)  

**Abstract**: Real-world processes often generate data that are a mix of categorical and numeric values that are recorded at irregular and informative intervals. Discrete token-based approaches are limited in numeric representation capacity while methods like neural ordinary differential equations are not well suited for categorical data or informative sampling and require augmentation to handle certain classes of trajectories. Here, we present multivariateGPT, a single architecture for modeling sequences of mixed categorical (including tokenized text) and numeric data. This is accomplished with an autoregressive sequence decomposition, embedding scheme, and loss function that extend the next token prediction task to likelihood estimation of the joint distribution of next token class and value. We demonstrate how this approach can efficiently learn to generalize patterns in simple physical systems and model complex time series including electrocardiograms and multivariate electronic health record data. This work extends the utility of transformer based models to additional classes of data. 

---
# What happens when generative AI models train recursively on each others' generated outputs? 

**Authors**: Hung Ahn Vu, Galen Reeves, Emily Wenger  

**Link**: [PDF](https://arxiv.org/pdf/2505.21677)  

**Abstract**: The internet is full of AI-generated content while also serving as a common source of training data for generative AI (genAI) models. This duality raises the possibility that future genAI models may be trained on other models' generated outputs. Prior work has studied consequences of models training on their own generated outputs, but limited work has considered what happens if models ingest content produced by other models. Given society's increasing dependence on genAI tools, understanding downstream effects of such data-mediated model interactions is critical. To this end, we provide empirical evidence for how data-mediated interactions might unfold in practice, develop a theoretical model for this interactive training process, and show experimentally possible long-term results of such interactions. We find that data-mediated interactions can benefit models by exposing them to novel concepts perhaps missed in original training data, but also can homogenize their performance on shared tasks. 

---
# Rethinking the Outlier Distribution in Large Language Models: An In-depth Study 

**Authors**: Rahul Raman, Khushi Sharma, Sai Qian Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21670)  

**Abstract**: Investigating outliers in large language models (LLMs) is crucial due to their significant impact on various aspects of LLM performance, including quantization and compression. Outliers often cause considerable quantization errors, leading to degraded model performance. Identifying and addressing these outliers can enhance the accuracy and efficiency of the quantization process, enabling smoother deployment on edge devices or specialized hardware. Recent studies have identified two common types of outliers in LLMs: massive activations and channel-wise outliers. While numerous quantization algorithms have been proposed to mitigate their effects and maintain satisfactory accuracy, few have thoroughly explored the root causes of these outliers in depth. In this paper, we conduct a comprehensive investigation into the formation mechanisms of these outliers and propose potential strategies to mitigate their occurrence. Ultimately, we introduce some efficient approaches to eliminate most massive activations and channel-wise outliers with minimal impact on accuracy. 

---
# Efficient Controllable Diffusion via Optimal Classifier Guidance 

**Authors**: Owen Oertell, Shikun Sun, Yiding Chen, Jin Peng Zhou, Zhiyong Wang, Wen Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.21666)  

**Abstract**: The controllable generation of diffusion models aims to steer the model to generate samples that optimize some given objective functions. It is desirable for a variety of applications including image generation, molecule generation, and DNA/sequence generation. Reinforcement Learning (RL) based fine-tuning of the base model is a popular approach but it can overfit the reward function while requiring significant resources. We frame controllable generation as a problem of finding a distribution that optimizes a KL-regularized objective function. We present SLCD -- Supervised Learning based Controllable Diffusion, which iteratively generates online data and trains a small classifier to guide the generation of the diffusion model. Similar to the standard classifier-guided diffusion, SLCD's key computation primitive is classification and does not involve any complex concepts from RL or control. Via a reduction to no-regret online learning analysis, we show that under KL divergence, the output from SLCD provably converges to the optimal solution of the KL-regularized objective. Further, we empirically demonstrate that SLCD can generate high quality samples with nearly the same inference time as the base model in both image generation with continuous diffusion and biological sequence generation with discrete diffusion. Our code is available at this https URL 

---
# Expert Survey: AI Reliability & Security Research Priorities 

**Authors**: Joe O'Brien, Jeremy Dolan, Jay Kim, Jonah Dykhuizen, Jeba Sania, Sebastian Becker, Jam Kraprayoon, Cara Labrador  

**Link**: [PDF](https://arxiv.org/pdf/2505.21664)  

**Abstract**: Our survey of 53 specialists across 105 AI reliability and security research areas identifies the most promising research prospects to guide strategic AI R&D investment. As companies are seeking to develop AI systems with broadly human-level capabilities, research on reliability and security is urgently needed to ensure AI's benefits can be safely and broadly realized and prevent severe harms. This study is the first to quantify expert priorities across a comprehensive taxonomy of AI safety and security research directions and to produce a data-driven ranking of their potential impact. These rankings may support evidence-based decisions about how to effectively deploy resources toward AI reliability and security research. 

---
# Explainability of Large Language Models using SMILE: Statistical Model-agnostic Interpretability with Local Explanations 

**Authors**: Zeinab Dehghani, Koorosh Aslansefat, Adil Khan, Mohammed Naveed Akram  

**Link**: [PDF](https://arxiv.org/pdf/2505.21657)  

**Abstract**: Large language models like GPT, LLAMA, and Claude have become incredibly powerful at generating text, but they are still black boxes, so it is hard to understand how they decide what to say. That lack of transparency can be problematic, especially in fields where trust and accountability matter. To help with this, we introduce SMILE, a new method that explains how these models respond to different parts of a prompt. SMILE is model-agnostic and works by slightly changing the input, measuring how the output changes, and then highlighting which words had the most impact. Create simple visual heat maps showing which parts of a prompt matter the most. We tested SMILE on several leading LLMs and used metrics such as accuracy, consistency, stability, and fidelity to show that it gives clear and reliable explanations. By making these models easier to understand, SMILE brings us one step closer to making AI more transparent and trustworthy. 

---
# PartInstruct: Part-level Instruction Following for Fine-grained Robot Manipulation 

**Authors**: Yifan Yin, Zhengtao Han, Shivam Aarya, Jianxin Wang, Shuhang Xu, Jiawei Peng, Angtian Wang, Alan Yuille, Tianmin Shu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21652)  

**Abstract**: Fine-grained robot manipulation, such as lifting and rotating a bottle to display the label on the cap, requires robust reasoning about object parts and their relationships with intended tasks. Despite recent advances in training general-purpose robot manipulation policies guided by language instructions, there is a notable lack of large-scale datasets for fine-grained manipulation tasks with part-level instructions and diverse 3D object instances annotated with part-level labels. In this work, we introduce PartInstruct, the first large-scale benchmark for training and evaluating fine-grained robot manipulation models using part-level instructions. PartInstruct comprises 513 object instances across 14 categories, each annotated with part-level information, and 1302 fine-grained manipulation tasks organized into 16 task classes. Our training set consists of over 10,000 expert demonstrations synthesized in a 3D simulator, where each demonstration is paired with a high-level task instruction, a chain of base part-based skill instructions, and ground-truth 3D information about the object and its parts. Additionally, we designed a comprehensive test suite to evaluate the generalizability of learned policies across new states, objects, and tasks. We evaluated several state-of-the-art robot manipulation approaches, including end-to-end vision-language policy learning and bi-level planning models for robot manipulation on our benchmark. The experimental results reveal that current models struggle to robustly ground part concepts and predict actions in 3D space, and face challenges when manipulating object parts in long-horizon tasks. 

---
# Efficient Diffusion Models for Symmetric Manifolds 

**Authors**: Oren Mangoubi, Neil He, Nisheeth K. Vishnoi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21640)  

**Abstract**: We introduce a framework for designing efficient diffusion models for $d$-dimensional symmetric-space Riemannian manifolds, including the torus, sphere, special orthogonal group and unitary group. Existing manifold diffusion models often depend on heat kernels, which lack closed-form expressions and require either $d$ gradient evaluations or exponential-in-$d$ arithmetic operations per training step. We introduce a new diffusion model for symmetric manifolds with a spatially-varying covariance, allowing us to leverage a projection of Euclidean Brownian motion to bypass heat kernel computations. Our training algorithm minimizes a novel efficient objective derived via Ito's Lemma, allowing each step to run in $O(1)$ gradient evaluations and nearly-linear-in-$d$ ($O(d^{1.19})$) arithmetic operations, reducing the gap between diffusions on symmetric manifolds and Euclidean space. Manifold symmetries ensure the diffusion satisfies an "average-case" Lipschitz condition, enabling accurate and efficient sample generation. Empirically, our model outperforms prior methods in training speed and improves sample quality on synthetic datasets on the torus, special orthogonal group, and unitary group. 

---
# The Feasibility of Topic-Based Watermarking on Academic Peer Reviews 

**Authors**: Alexander Nemecek, Yuzhou Jiang, Erman Ayday  

**Link**: [PDF](https://arxiv.org/pdf/2505.21636)  

**Abstract**: Large language models (LLMs) are increasingly integrated into academic workflows, with many conferences and journals permitting their use for tasks such as language refinement and literature summarization. However, their use in peer review remains prohibited due to concerns around confidentiality breaches, hallucinated content, and inconsistent evaluations. As LLM-generated text becomes more indistinguishable from human writing, there is a growing need for reliable attribution mechanisms to preserve the integrity of the review process. In this work, we evaluate topic-based watermarking (TBW), a lightweight, semantic-aware technique designed to embed detectable signals into LLM-generated text. We conduct a comprehensive assessment across multiple LLM configurations, including base, few-shot, and fine-tuned variants, using authentic peer review data from academic conferences. Our results show that TBW maintains review quality relative to non-watermarked outputs, while demonstrating strong robustness to paraphrasing-based evasion. These findings highlight the viability of TBW as a minimally intrusive and practical solution for enforcing LLM usage in peer review. 

---
# Is Your LLM Overcharging You? Tokenization, Transparency, and Incentives 

**Authors**: Ander Artola Velasco, Stratis Tsirtsis, Nastaran Okati, Manuel Gomez-Rodriguez  

**Link**: [PDF](https://arxiv.org/pdf/2505.21627)  

**Abstract**: State-of-the-art large language models require specialized hardware and substantial energy to operate. As a consequence, cloud-based services that provide access to large language models have become very popular. In these services, the price users pay for an output provided by a model depends on the number of tokens the model uses to generate it -- they pay a fixed price per token. In this work, we show that this pricing mechanism creates a financial incentive for providers to strategize and misreport the (number of) tokens a model used to generate an output, and users cannot prove, or even know, whether a provider is overcharging them. However, we also show that, if an unfaithful provider is obliged to be transparent about the generative process used by the model, misreporting optimally without raising suspicion is hard. Nevertheless, as a proof-of-concept, we introduce an efficient heuristic algorithm that allows providers to significantly overcharge users without raising suspicion, highlighting the vulnerability of users under the current pay-per-token pricing mechanism. Further, to completely eliminate the financial incentive to strategize, we introduce a simple incentive-compatible token pricing mechanism. Under this mechanism, the price users pay for an output provided by a model depends on the number of characters of the output -- they pay a fixed price per character. Along the way, to illustrate and complement our theoretical results, we conduct experiments with several large language models from the $\texttt{Llama}$, $\texttt{Gemma}$ and $\texttt{Ministral}$ families, and input prompts from the LMSYS Chatbot Arena platform. 

---
# VideoMarkBench: Benchmarking Robustness of Video Watermarking 

**Authors**: Zhengyuan Jiang, Moyang Guo, Kecen Li, Yuepeng Hu, Yupu Wang, Zhicong Huang, Cheng Hong, Neil Zhenqiang Gong  

**Link**: [PDF](https://arxiv.org/pdf/2505.21620)  

**Abstract**: The rapid development of video generative models has led to a surge in highly realistic synthetic videos, raising ethical concerns related to disinformation and copyright infringement. Recently, video watermarking has been proposed as a mitigation strategy by embedding invisible marks into AI-generated videos to enable subsequent detection. However, the robustness of existing video watermarking methods against both common and adversarial perturbations remains underexplored. In this work, we introduce VideoMarkBench, the first systematic benchmark designed to evaluate the robustness of video watermarks under watermark removal and watermark forgery attacks. Our study encompasses a unified dataset generated by three state-of-the-art video generative models, across three video styles, incorporating four watermarking methods and seven aggregation strategies used during detection. We comprehensively evaluate 12 types of perturbations under white-box, black-box, and no-box threat models. Our findings reveal significant vulnerabilities in current watermarking approaches and highlight the urgent need for more robust solutions. Our code is available at this https URL. 

---
# Preventing Adversarial AI Attacks Against Autonomous Situational Awareness: A Maritime Case Study 

**Authors**: Mathew J. Walter, Aaron Barrett, Kimberly Tam  

**Link**: [PDF](https://arxiv.org/pdf/2505.21609)  

**Abstract**: Adversarial artificial intelligence (AI) attacks pose a significant threat to autonomous transportation, such as maritime vessels, that rely on AI components. Malicious actors can exploit these systems to deceive and manipulate AI-driven operations. This paper addresses three critical research challenges associated with adversarial AI: the limited scope of traditional defences, inadequate security metrics, and the need to build resilience beyond model-level defences. To address these challenges, we propose building defences utilising multiple inputs and data fusion to create defensive components and an AI security metric as a novel approach toward developing more secure AI systems. We name this approach the Data Fusion Cyber Resilience (DFCR) method, and we evaluate it through real-world demonstrations and comprehensive quantitative analyses, comparing a system built with the DFCR method against single-input models and models utilising existing state-of-the-art defences. The findings show that the DFCR approach significantly enhances resilience against adversarial machine learning attacks in maritime autonomous system operations, achieving up to a 35\% reduction in loss for successful multi-pronged perturbation attacks, up to a 100\% reduction in loss for successful adversarial patch attacks and up to 100\% reduction in loss for successful spoofing attacks when using these more resilient systems. We demonstrate how DFCR and DFCR confidence scores can reduce adversarial AI contact confidence and improve decision-making by the system, even when typical adversarial defences have been compromised. Ultimately, this work contributes to the development of more secure and resilient AI-driven systems against adversarial attacks. 

---
# How does Misinformation Affect Large Language Model Behaviors and Preferences? 

**Authors**: Miao Peng, Nuo Chen, Jianheng Tang, Jia Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.21608)  

**Abstract**: Large Language Models (LLMs) have shown remarkable capabilities in knowledge-intensive tasks, while they remain vulnerable when encountering misinformation. Existing studies have explored the role of LLMs in combating misinformation, but there is still a lack of fine-grained analysis on the specific aspects and extent to which LLMs are influenced by misinformation. To bridge this gap, we present MisBench, the current largest and most comprehensive benchmark for evaluating LLMs' behavior and knowledge preference toward misinformation. MisBench consists of 10,346,712 pieces of misinformation, which uniquely considers both knowledge-based conflicts and stylistic variations in misinformation. Empirical results reveal that while LLMs demonstrate comparable abilities in discerning misinformation, they still remain susceptible to knowledge conflicts and stylistic variations. Based on these findings, we further propose a novel approach called Reconstruct to Discriminate (RtD) to strengthen LLMs' ability to detect misinformation. Our study provides valuable insights into LLMs' interactions with misinformation, and we believe MisBench can serve as an effective benchmark for evaluating LLM-based detectors and enhancing their reliability in real-world applications. Codes and data are available at this https URL. 

---
# SOSBENCH: Benchmarking Safety Alignment on Scientific Knowledge 

**Authors**: Fengqing Jiang, Fengbo Ma, Zhangchen Xu, Yuetai Li, Bhaskar Ramasubramanian, Luyao Niu, Bo Li, Xianyan Chen, Zhen Xiang, Radha Poovendran  

**Link**: [PDF](https://arxiv.org/pdf/2505.21605)  

**Abstract**: Large language models (LLMs) exhibit advancing capabilities in complex tasks, such as reasoning and graduate-level question answering, yet their resilience against misuse, particularly involving scientifically sophisticated risks, remains underexplored. Existing safety benchmarks typically focus either on instructions requiring minimal knowledge comprehension (e.g., ``tell me how to build a bomb") or utilize prompts that are relatively low-risk (e.g., multiple-choice or classification tasks about hazardous content). Consequently, they fail to adequately assess model safety when handling knowledge-intensive, hazardous scenarios.
To address this critical gap, we introduce SOSBench, a regulation-grounded, hazard-focused benchmark encompassing six high-risk scientific domains: chemistry, biology, medicine, pharmacology, physics, and psychology. The benchmark comprises 3,000 prompts derived from real-world regulations and laws, systematically expanded via an LLM-assisted evolutionary pipeline that introduces diverse, realistic misuse scenarios (e.g., detailed explosive synthesis instructions involving advanced chemical formulas). We evaluate frontier models within a unified evaluation framework using our SOSBench. Despite their alignment claims, advanced models consistently disclose policy-violating content across all domains, demonstrating alarmingly high rates of harmful responses (e.g., 79.1% for Deepseek-R1 and 47.3% for GPT-4.1). These results highlight significant safety alignment deficiencies and underscore urgent concerns regarding the responsible deployment of powerful LLMs. 

---
# Public Discourse Sandbox: Facilitating Human and AI Digital Communication Research 

**Authors**: Kristina Radivojevic, Caleb Reinking, Shaun Whitfield, Paul Brenner  

**Link**: [PDF](https://arxiv.org/pdf/2505.21604)  

**Abstract**: Social media serves as a primary communication and information dissemination platform for major global events, entertainment, and niche or topically focused community discussions. Therefore, it represents a valuable resource for researchers who aim to understand numerous questions. However, obtaining data can be difficult, expensive, and often unreliable due to the presence of bots, fake accounts, and manipulated content. Additionally, there are ethical concerns if researchers decide to conduct an online experiment without explicitly notifying social media users about their intent. There is a need for more controlled and scalable mechanisms to evaluate the impacts of digital discussion interventions on audiences. We introduce the Public Discourse Sandbox (PDS), which serves as a digital discourse research platform for human-AI as well as AI-AI discourse research, testing, and training. PDS provides a safe and secure space for research experiments that are not viable on public, commercial social media platforms. Its main purpose is to enable the understanding of AI behaviors and the impacts of customized AI participants via techniques such as prompt engineering, retrieval-augmented generation (RAG), and fine-tuning. We provide a hosted live version of the sandbox to support researchers as well as the open-sourced code on GitHub for community collaboration and contribution. 

---
# Leveraging XP and CRISP-DM for Agile Data Science Projects 

**Authors**: Andre Massahiro Shimaoka, Renato Cordeiro Ferreira, Alfredo Goldman  

**Link**: [PDF](https://arxiv.org/pdf/2505.21603)  

**Abstract**: This study explores the integration of eXtreme Programming (XP) and the Cross-Industry Standard Process for Data Mining (CRISP-DM) in agile Data Science projects. We conducted a case study at the e-commerce company Elo7 to answer the research question: How can the agility of the XP method be integrated with CRISP-DM in Data Science projects? Data was collected through interviews and questionnaires with a Data Science team consisting of data scientists, ML engineers, and data product managers. The results show that 86% of the team frequently or always applies CRISP-DM, while 71% adopt XP practices in their projects. Furthermore, the study demonstrates that it is possible to combine CRISP-DM with XP in Data Science projects, providing a structured and collaborative approach. Finally, the study generated improvement recommendations for the company. 

---
# R2R: Efficiently Navigating Divergent Reasoning Paths with Small-Large Model Token Routing 

**Authors**: Tianyu Fu, Yi Ge, Yichen You, Enshu Liu, Zhihang Yuan, Guohao Dai, Shengen Yan, Huazhong Yang, Yu Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21600)  

**Abstract**: Large Language Models (LLMs) achieve impressive reasoning capabilities at the cost of substantial inference overhead, posing substantial deployment challenges. Although distilled Small Language Models (SLMs) significantly enhance efficiency, their performance suffers as they fail to follow LLMs' reasoning paths. Luckily, we reveal that only a small fraction of tokens genuinely diverge reasoning paths between LLMs and SLMs. Most generated tokens are either identical or exhibit neutral differences, such as minor variations in abbreviations or expressions. Leveraging this insight, we introduce **Roads to Rome (R2R)**, a neural token routing method that selectively utilizes LLMs only for these critical, path-divergent tokens, while leaving the majority of token generation to the SLM. We also develop an automatic data generation pipeline that identifies divergent tokens and generates token-level routing labels to train the lightweight router. We apply R2R to combine R1-1.5B and R1-32B models from the DeepSeek family, and evaluate on challenging math, coding, and QA benchmarks. With an average activated parameter size of 5.6B, R2R surpasses the average accuracy of R1-7B by 1.6x, outperforming even the R1-14B model. Compared to R1-32B, it delivers a 2.8x wall-clock speedup with comparable performance, advancing the Pareto frontier of test-time scaling efficiency. Our code is available at this https URL. 

---
# Learning optimal treatment strategies for intraoperative hypotension using deep reinforcement learning 

**Authors**: Esra Adiyeke, Tianqi Liu, Venkata Sai Dheeraj Naganaboina, Han Li, Tyler J. Loftus, Yuanfang Ren, Benjamin Shickel, Matthew M. Ruppert, Karandeep Singh, Ruogu Fang, Parisa Rashidi, Azra Bihorac, Tezcan Ozrazgat-Baslanti  

**Link**: [PDF](https://arxiv.org/pdf/2505.21596)  

**Abstract**: Traditional methods of surgical decision making heavily rely on human experience and prompt actions, which are variable. A data-driven system generating treatment recommendations based on patient states can be a substantial asset in perioperative decision-making, as in cases of intraoperative hypotension, for which suboptimal management is associated with acute kidney injury (AKI), a common and morbid postoperative complication. We developed a Reinforcement Learning (RL) model to recommend optimum dose of intravenous (IV) fluid and vasopressors during surgery to avoid intraoperative hypotension and postoperative AKI. We retrospectively analyzed 50,021 surgeries from 42,547 adult patients who underwent major surgery at a quaternary care hospital between June 2014 and September 2020. Of these, 34,186 surgeries were used for model training and 15,835 surgeries were reserved for testing. We developed a Deep Q-Networks based RL model using 16 variables including intraoperative physiologic time series, total dose of IV fluid and vasopressors extracted for every 15-minute epoch. The model replicated 69% of physician's decisions for the dosage of vasopressors and proposed higher or lower dosage of vasopressors than received in 10% and 21% of the treatments, respectively. In terms of IV fluids, the model's recommendations were within 0.05 ml/kg/15 min of the actual dose in 41% of the cases, with higher or lower doses recommended for 27% and 32% of the treatments, respectively. The model resulted in a higher estimated policy value compared to the physicians' actual treatments, as well as random and zero-drug policies. AKI prevalence was the lowest in patients receiving medication dosages that aligned with model's decisions. Our findings suggest that implementation of the model's policy has the potential to reduce postoperative AKI and improve other outcomes driven by intraoperative hypotension. 

---
# Relevance-driven Input Dropout: an Explanation-guided Regularization Technique 

**Authors**: Shreyas Gururaj, Lars Grüne, Wojciech Samek, Sebastian Lapuschkin, Leander Weber  

**Link**: [PDF](https://arxiv.org/pdf/2505.21595)  

**Abstract**: Overfitting is a well-known issue extending even to state-of-the-art (SOTA) Machine Learning (ML) models, resulting in reduced generalization, and a significant train-test performance gap. Mitigation measures include a combination of dropout, data augmentation, weight decay, and other regularization techniques. Among the various data augmentation strategies, occlusion is a prominent technique that typically focuses on randomly masking regions of the input during training. Most of the existing literature emphasizes randomness in selecting and modifying the input features instead of regions that strongly influence model decisions. We propose Relevance-driven Input Dropout (RelDrop), a novel data augmentation method which selectively occludes the most relevant regions of the input, nudging the model to use other important features in the prediction process, thus improving model generalization through informed regularization. We further conduct qualitative and quantitative analyses to study how Relevance-driven Input Dropout (RelDrop) affects model decision-making. Through a series of experiments on benchmark datasets, we demonstrate that our approach improves robustness towards occlusion, results in models utilizing more features within the region of interest, and boosts inference time generalization performance. Our code is available at this https URL. 

---
# Fast and Cost-effective Speculative Edge-Cloud Decoding with Early Exits 

**Authors**: Yeshwanth Venkatesha, Souvik Kundu, Priyadarshini Panda  

**Link**: [PDF](https://arxiv.org/pdf/2505.21594)  

**Abstract**: Large Language Models (LLMs) enable various applications on edge devices such as smartphones, wearables, and embodied robots. However, their deployment often depends on expensive cloud-based APIs, creating high operational costs, which limit access for smaller organizations and raise sustainability concerns. Certain LLMs can be deployed on-device, offering a cost-effective solution with reduced latency and improved privacy. Yet, limited computing resources constrain the size and accuracy of models that can be deployed, necessitating a collaborative design between edge and cloud. We propose a fast and cost-effective speculative edge-cloud decoding framework with a large target model on the server and a small draft model on the device. By introducing early exits in the target model, tokens are generated mid-verification, allowing the client to preemptively draft subsequent tokens before final verification, thus utilizing idle time and enhancing parallelism between edge and cloud. Using an NVIDIA Jetson Nano (client) and an A100 GPU (server) with Vicuna-68M (draft) and Llama2-7B (target) models, our method achieves up to a 35% reduction in latency compared to cloud-based autoregressive decoding, with an additional 11% improvement from preemptive drafting. To demonstrate real-world applicability, we deploy our method on the Unitree Go2 quadruped robot using Vision-Language Model (VLM) based control, achieving a 21% speedup over traditional cloud-based autoregressive decoding. These results demonstrate the potential of our framework for real-time LLM and VLM applications on resource-constrained edge devices. 

---
# Any-to-Bokeh: One-Step Video Bokeh via Multi-Plane Image Guided Diffusion 

**Authors**: Yang Yang, Siming Zheng, Jinwei Chen, Boxi Wu, Xiaofei He, Deng Cai, Bo Li, Peng-Tao Jiang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21593)  

**Abstract**: Recent advances in diffusion based editing models have enabled realistic camera simulation and image-based bokeh, but video bokeh remains largely unexplored. Existing video editing models cannot explicitly control focus planes or adjust bokeh intensity, limiting their applicability for controllable optical effects. Moreover, naively extending image-based bokeh methods to video often results in temporal flickering and unsatisfactory edge blur transitions due to the lack of temporal modeling and generalization capability. To address these challenges, we propose a novel one-step video bokeh framework that converts arbitrary input videos into temporally coherent, depth-aware bokeh effects. Our method leverages a multi-plane image (MPI) representation constructed through a progressively widening depth sampling function, providing explicit geometric guidance for depth-dependent blur synthesis. By conditioning a single-step video diffusion model on MPI layers and utilizing the strong 3D priors from pre-trained models such as Stable Video Diffusion, our approach achieves realistic and consistent bokeh effects across diverse scenes. Additionally, we introduce a progressive training strategy to enhance temporal consistency, depth robustness, and detail preservation. Extensive experiments demonstrate that our method produces high-quality, controllable bokeh effects and achieves state-of-the-art performance on multiple evaluation benchmarks. 

---
# Pioneering 4-Bit FP Quantization for Diffusion Models: Mixup-Sign Quantization and Timestep-Aware Fine-Tuning 

**Authors**: Maosen Zhao, Pengtao Chen, Chong Yu, Yan Wen, Xudong Tan, Tao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21591)  

**Abstract**: Model quantization reduces the bit-width of weights and activations, improving memory efficiency and inference speed in diffusion models. However, achieving 4-bit quantization remains challenging. Existing methods, primarily based on integer quantization and post-training quantization fine-tuning, struggle with inconsistent performance. Inspired by the success of floating-point (FP) quantization in large language models, we explore low-bit FP quantization for diffusion models and identify key challenges: the failure of signed FP quantization to handle asymmetric activation distributions, the insufficient consideration of temporal complexity in the denoising process during fine-tuning, and the misalignment between fine-tuning loss and quantization error. To address these challenges, we propose the mixup-sign floating-point quantization (MSFP) framework, first introducing unsigned FP quantization in model quantization, along with timestep-aware LoRA (TALoRA) and denoising-factor loss alignment (DFA), which ensure precise and stable fine-tuning. Extensive experiments show that we are the first to achieve superior performance in 4-bit FP quantization for diffusion models, outperforming existing PTQ fine-tuning methods in 4-bit INT quantization. 

---
# Do you see what I see? An Ambiguous Optical Illusion Dataset exposing limitations of Explainable AI 

**Authors**: Carina Newen, Luca Hinkamp, Maria Ntonti, Emmanuel Müller  

**Link**: [PDF](https://arxiv.org/pdf/2505.21589)  

**Abstract**: From uncertainty quantification to real-world object detection, we recognize the importance of machine learning algorithms, particularly in safety-critical domains such as autonomous driving or medical diagnostics. In machine learning, ambiguous data plays an important role in various machine learning domains. Optical illusions present a compelling area of study in this context, as they offer insight into the limitations of both human and machine perception. Despite this relevance, optical illusion datasets remain scarce. In this work, we introduce a novel dataset of optical illusions featuring intermingled animal pairs designed to evoke perceptual ambiguity. We identify generalizable visual concepts, particularly gaze direction and eye cues, as subtle yet impactful features that significantly influence model accuracy. By confronting models with perceptual ambiguity, our findings underscore the importance of concepts in visual learning and provide a foundation for studying bias and alignment between human and machine vision. To make this dataset useful for general purposes, we generate optical illusions systematically with different concepts discussed in our bias mitigation section. The dataset is accessible in Kaggle via this https URL. Our source code can be found at this https URL. 

---
# Herd Behavior: Investigating Peer Influence in LLM-based Multi-Agent Systems 

**Authors**: Young-Min Cho, Sharath Chandra Guntuku, Lyle Ungar  

**Link**: [PDF](https://arxiv.org/pdf/2505.21588)  

**Abstract**: Recent advancements in Large Language Models (LLMs) have enabled the emergence of multi-agent systems where LLMs interact, collaborate, and make decisions in shared environments. While individual model behavior has been extensively studied, the dynamics of peer influence in such systems remain underexplored. In this paper, we investigate herd behavior, the tendency of agents to align their outputs with those of their peers, within LLM-based multi-agent interactions. We present a series of controlled experiments that reveal how herd behaviors are shaped by multiple factors. First, we show that the gap between self-confidence and perceived confidence in peers significantly impacts an agent's likelihood to conform. Second, we find that the format in which peer information is presented plays a critical role in modulating the strength of herd behavior. Finally, we demonstrate that the degree of herd behavior can be systematically controlled, and that appropriately calibrated herd tendencies can enhance collaborative outcomes. These findings offer new insights into the social dynamics of LLM-based systems and open pathways for designing more effective and adaptive multi-agent collaboration frameworks. 

---
# CellCLAT: Preserving Topology and Trimming Redundancy in Self-Supervised Cellular Contrastive Learning 

**Authors**: Bin Qin, Qirui Ji, Jiangmeng Li, Yupeng Wang, Xuesong Wu, Jianwen Cao, Fanjiang Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21587)  

**Abstract**: Self-supervised topological deep learning (TDL) represents a nascent but underexplored area with significant potential for modeling higher-order interactions in simplicial complexes and cellular complexes to derive representations of unlabeled graphs. Compared to simplicial complexes, cellular complexes exhibit greater expressive power. However, the advancement in self-supervised learning for cellular TDL is largely hindered by two core challenges: \textit{extrinsic structural constraints} inherent to cellular complexes, and intrinsic semantic redundancy in cellular representations. The first challenge highlights that traditional graph augmentation techniques may compromise the integrity of higher-order cellular interactions, while the second underscores that topological redundancy in cellular complexes potentially diminish task-relevant information. To address these issues, we introduce Cellular Complex Contrastive Learning with Adaptive Trimming (CellCLAT), a twofold framework designed to adhere to the combinatorial constraints of cellular complexes while mitigating informational redundancy. Specifically, we propose a parameter perturbation-based augmentation method that injects controlled noise into cellular interactions without altering the underlying cellular structures, thereby preserving cellular topology during contrastive learning. Additionally, a cellular trimming scheduler is employed to mask gradient contributions from task-irrelevant cells through a bi-level meta-learning approach, effectively removing redundant topological elements while maintaining critical higher-order semantics. We provide theoretical justification and empirical validation to demonstrate that CellCLAT achieves substantial improvements over existing self-supervised graph learning methods, marking a significant attempt in this domain. 

---
# Fairness in Federated Learning: Fairness for Whom? 

**Authors**: Afaf Taik, Khaoula Chehbouni, Golnoosh Farnadi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21584)  

**Abstract**: Fairness in federated learning has emerged as a rapidly growing area of research, with numerous works proposing formal definitions and algorithmic interventions. Yet, despite this technical progress, fairness in FL is often defined and evaluated in ways that abstract away from the sociotechnical contexts in which these systems are deployed. In this paper, we argue that existing approaches tend to optimize narrow system level metrics, such as performance parity or contribution-based rewards, while overlooking how harms arise throughout the FL lifecycle and how they impact diverse stakeholders. We support this claim through a critical analysis of the literature, based on a systematic annotation of papers for their fairness definitions, design decisions, evaluation practices, and motivating use cases. Our analysis reveals five recurring pitfalls: 1) fairness framed solely through the lens of server client architecture, 2) a mismatch between simulations and motivating use-cases and contexts, 3) definitions that conflate protecting the system with protecting its users, 4) interventions that target isolated stages of the lifecycle while neglecting upstream and downstream effects, 5) and a lack of multi-stakeholder alignment where multiple fairness definitions can be relevant at once. Building on these insights, we propose a harm centered framework that links fairness definitions to concrete risks and stakeholder vulnerabilities. We conclude with recommendations for more holistic, context-aware, and accountable fairness research in FL. 

---
# AITEE -- Agentic Tutor for Electrical Engineering 

**Authors**: Christopher Knievel, Alexander Bernhardt, Christian Bernhardt  

**Link**: [PDF](https://arxiv.org/pdf/2505.21582)  

**Abstract**: Intelligent tutoring systems combined with large language models offer a promising approach to address students' diverse needs and promote self-efficacious learning. While large language models possess good foundational knowledge of electrical engineering basics, they remain insufficiently capable of addressing specific questions about electrical circuits. In this paper, we present AITEE, an agent-based tutoring system for electrical engineering designed to accompany students throughout their learning process, offer individualized support, and promote self-directed learning. AITEE supports both hand-drawn and digital circuits through an adapted circuit reconstruction process, enabling natural interaction with students. Our novel graph-based similarity measure identifies relevant context from lecture materials through a retrieval augmented generation approach, while parallel Spice simulation further enhances accuracy in applying solution methodologies. The system implements a Socratic dialogue to foster learner autonomy through guided questioning. Experimental evaluations demonstrate that AITEE significantly outperforms baseline approaches in domain-specific knowledge application, with even medium-sized LLM models showing acceptable performance. Our results highlight the potential of agentic tutors to deliver scalable, personalized, and effective learning environments for electrical engineering education. 

---
# RepoMaster: Autonomous Exploration and Understanding of GitHub Repositories for Complex Task Solving 

**Authors**: Huacan Wang, Ziyi Ni, Shuo Zhang, Shuo Lu, Sen Hu, Ziyang He, Chen Hu, Jiaye Lin, Yifu Guo, Yuntao Du, Pin Lyu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21577)  

**Abstract**: The ultimate goal of code agents is to solve complex tasks autonomously. Although large language models (LLMs) have made substantial progress in code generation, real-world tasks typically demand full-fledged code repositories rather than simple scripts. Building such repositories from scratch remains a major challenge. Fortunately, GitHub hosts a vast, evolving collection of open-source repositories, which developers frequently reuse as modular components for complex tasks. Yet, existing frameworks like OpenHands and SWE-Agent still struggle to effectively leverage these valuable resources. Relying solely on README files provides insufficient guidance, and deeper exploration reveals two core obstacles: overwhelming information and tangled dependencies of repositories, both constrained by the limited context windows of current LLMs. To tackle these issues, we propose RepoMaster, an autonomous agent framework designed to explore and reuse GitHub repositories for solving complex tasks. For efficient understanding, RepoMaster constructs function-call graphs, module-dependency graphs, and hierarchical code trees to identify essential components, providing only identified core elements to the LLMs rather than the entire repository. During autonomous execution, it progressively explores related components using our exploration tools and prunes information to optimize context usage. Evaluated on the adjusted MLE-bench, RepoMaster achieves a 110% relative boost in valid submissions over the strongest baseline OpenHands. On our newly released GitTaskBench, RepoMaster lifts the task-pass rate from 24.1% to 62.9% while reducing token usage by 95%. Our code and demonstration materials are publicly available at this https URL. 

---
# Concentration Distribution Learning from Label Distributions 

**Authors**: Jiawei Tang, Yuheng Jia  

**Link**: [PDF](https://arxiv.org/pdf/2505.21576)  

**Abstract**: Label distribution learning (LDL) is an effective method to predict the relative label description degree (a.k.a. label distribution) of a sample. However, the label distribution is not a complete representation of an instance because it overlooks the absolute intensity of each label. Specifically, it's impossible to obtain the total description degree of hidden labels that not in the label space, which leads to the loss of information and confusion in instances. To solve the above problem, we come up with a new concept named background concentration to serve as the absolute description degree term of the label distribution and introduce it into the LDL process, forming the improved paradigm of concentration distribution learning. Moreover, we propose a novel model by probabilistic methods and neural networks to learn label distributions and background concentrations from existing LDL datasets. Extensive experiments prove that the proposed approach is able to extract background concentrations from label distributions while producing more accurate prediction results than the state-of-the-art LDL methods. The code is available in this https URL. 

---
# StreamLink: Large-Language-Model Driven Distributed Data Engineering System 

**Authors**: Dawei Feng, Di Mei, Huiri Tan, Lei Ren, Xianying Lou, Zhangxi Tan  

**Link**: [PDF](https://arxiv.org/pdf/2505.21575)  

**Abstract**: Large Language Models (LLMs) have shown remarkable proficiency in natural language understanding (NLU), opening doors for innovative applications. We introduce StreamLink - an LLM-driven distributed data system designed to improve the efficiency and accessibility of data engineering tasks. We build StreamLink on top of distributed frameworks such as Apache Spark and Hadoop to handle large data at scale. One of the important design philosophies of StreamLink is to respect user data privacy by utilizing local fine-tuned LLMs instead of a public AI service like ChatGPT. With help from domain-adapted LLMs, we can improve our system's understanding of natural language queries from users in various scenarios and simplify the procedure of generating database queries like the Structured Query Language (SQL) for information processing. We also incorporate LLM-based syntax and security checkers to guarantee the reliability and safety of each generated query. StreamLink illustrates the potential of merging generative LLMs with distributed data processing for comprehensive and user-centric data engineering. With this architecture, we allow users to interact with complex database systems at different scales in a user-friendly and security-ensured manner, where the SQL generation reaches over 10\% of execution accuracy compared to baseline methods, and allow users to find the most concerned item from hundreds of millions of items within a few seconds using natural language. 

---
# Spectral-inspired Neural Operator for Data-efficient PDE Simulation in Physics-agnostic Regimes 

**Authors**: Han Wan, Rui Zhang, Hao Sun  

**Link**: [PDF](https://arxiv.org/pdf/2505.21573)  

**Abstract**: Partial differential equations (PDEs) govern the spatiotemporal evolution of various physical systems. Classical numerical solvers, while accurate, require fine discretization and full knowledge of the governing PDEs, limiting their applicability when the physics is unknown or fast inference is required. Data-driven neural PDE solvers alleviate these constraints by learning from data but demand large training datasets and perform poorly in data-scarce regimes. Physics-aware methods mitigate data requirements by incorporating physical knowledge yet rely on known PDE terms or local numerical schemes, restricting their ability to handle unknown or globally coupled systems. In this work, we propose the Spectral-inspired Neural Operator (SINO), a novel framework that learns PDE operators from limited trajectories (as few as 2-5), without any known PDE terms. SINO operates in the frequency domain and introduces a Frequency-to-Vector module to learn spectral representations analogous to derivative multipliers. To model nonlinear physical interactions, we design a nonlinear operator block that includes a $\Pi$-Block with low-pass filtering to prevent aliasing. Finally, we introduce an operator distillation technique to distill the trained model for efficient inference. SINO achieves state-of-the-art results across multiple PDE benchmarks, demonstrating strong discretization invariance and robust generalization to out-of-distribution initial conditions. To our knowledge, SINO is the first physics-aware method capable of accurately simulating globally coupled systems (e.g., the Navier-Stokes equations) from limited data without any explicit PDE terms. 

---
# Thickness-aware E(3)-Equivariant 3D Mesh Neural Networks 

**Authors**: Sungwon Kim, Namkyeong Lee, Yunyoung Doh, Seungmin Shin, Guimok Cho, Seung-Won Jeon, Sangkook Kim, Chanyoung Park  

**Link**: [PDF](https://arxiv.org/pdf/2505.21572)  

**Abstract**: Mesh-based 3D static analysis methods have recently emerged as efficient alternatives to traditional computational numerical solvers, significantly reducing computational costs and runtime for various physics-based analyses. However, these methods primarily focus on surface topology and geometry, often overlooking the inherent thickness of real-world 3D objects, which exhibits high correlations and similar behavior between opposing surfaces. This limitation arises from the disconnected nature of these surfaces and the absence of internal edge connections within the mesh. In this work, we propose a novel framework, the Thickness-aware E(3)-Equivariant 3D Mesh Neural Network (T-EMNN), that effectively integrates the thickness of 3D objects while maintaining the computational efficiency of surface meshes. Additionally, we introduce data-driven coordinates that encode spatial information while preserving E(3)-equivariance or invariance properties, ensuring consistent and robust analysis. Evaluations on a real-world industrial dataset demonstrate the superior performance of T-EMNN in accurately predicting node-level 3D deformations, effectively capturing thickness effects while maintaining computational efficiency. 

---
# FCOS: A Two-Stage Recoverable Model Pruning Framework for Automatic Modulation Recognition 

**Authors**: Yao Lu, Tengfei Ma, Zeyu Wang, Zhuangzhi Chen, Dongwei Xu, Yun Lin, Qi Xuan, Guan Gui  

**Link**: [PDF](https://arxiv.org/pdf/2505.21571)  

**Abstract**: With the rapid development of wireless communications and the growing complexity of digital modulation schemes, traditional manual modulation recognition methods struggle to extract reliable signal features and meet real-time requirements in modern scenarios. Recently, deep learning based Automatic Modulation Recognition (AMR) approaches have greatly improved classification accuracy. However, their large model sizes and high computational demands hinder deployment on resource-constrained devices. Model pruning provides a general approach to reduce model complexity, but existing weight, channel, and layer pruning techniques each present a trade-off between compression rate, hardware acceleration, and accuracy preservation. To this end, in this paper, we introduce FCOS, a novel Fine-to-COarse two-Stage pruning framework that combines channel-level pruning with layer-level collapse diagnosis to achieve extreme compression, high performance and efficient inference. In the first stage of FCOS, hierarchical clustering and parameter fusion are applied to channel weights to achieve channel-level pruning. Then a Layer Collapse Diagnosis (LaCD) module uses linear probing to identify layer collapse and removes the collapsed layers due to high channel compression ratio. Experiments on multiple AMR benchmarks demonstrate that FCOS outperforms existing channel and layer pruning methods. Specifically, FCOS achieves 95.51% FLOPs reduction and 95.31% parameter reduction while still maintaining performance close to the original ResNet56, with only a 0.46% drop in accuracy on Sig2019-12. Code is available at this https URL. 

---
# Beyond Explainability: The Case for AI Validation 

**Authors**: Dalit Ken-Dror Feldman, Daniel Benoliel  

**Link**: [PDF](https://arxiv.org/pdf/2505.21570)  

**Abstract**: Artificial Knowledge (AK) systems are transforming decision-making across critical domains such as healthcare, finance, and criminal justice. However, their growing opacity presents governance challenges that current regulatory approaches, focused predominantly on explainability, fail to address adequately. This article argues for a shift toward validation as a central regulatory pillar. Validation, ensuring the reliability, consistency, and robustness of AI outputs, offers a more practical, scalable, and risk-sensitive alternative to explainability, particularly in high-stakes contexts where interpretability may be technically or economically unfeasible. We introduce a typology based on two axes, validity and explainability, classifying AK systems into four categories and exposing the trade-offs between interpretability and output reliability. Drawing on comparative analysis of regulatory approaches in the EU, US, UK, and China, we show how validation can enhance societal trust, fairness, and safety even where explainability is limited. We propose a forward-looking policy framework centered on pre- and post-deployment validation, third-party auditing, harmonized standards, and liability incentives. This framework balances innovation with accountability and provides a governance roadmap for responsibly integrating opaque, high-performing AK systems into society. 

---
# ChemHAS: Hierarchical Agent Stacking for Enhancing Chemistry Tools 

**Authors**: Zhucong Li, Bowei Zhang, Jin Xiao, Zhijian Zhou, Fenglei Cao, Jiaqing Liang, Yuan Qi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21569)  

**Abstract**: Large Language Model (LLM)-based agents have demonstrated the ability to improve performance in chemistry-related tasks by selecting appropriate tools. However, their effectiveness remains limited by the inherent prediction errors of chemistry tools. In this paper, we take a step further by exploring how LLMbased agents can, in turn, be leveraged to reduce prediction errors of the tools. To this end, we propose ChemHAS (Chemical Hierarchical Agent Stacking), a simple yet effective method that enhances chemistry tools through optimizing agent-stacking structures from limited data. ChemHAS achieves state-of-the-art performance across four fundamental chemistry tasks, demonstrating that our method can effectively compensate for prediction errors of the tools. Furthermore, we identify and characterize four distinct agent-stacking behaviors, potentially improving interpretability and revealing new possibilities for AI agent applications in scientific research. Our code and dataset are publicly available at https: //anonymous.this http URL. 

---
# VoiceMark: Zero-Shot Voice Cloning-Resistant Watermarking Approach Leveraging Speaker-Specific Latents 

**Authors**: Haiyun Li, Zhiyong Wu, Xiaofeng Xie, Jingran Xie, Yaoxun Xu, Hanyang Peng  

**Link**: [PDF](https://arxiv.org/pdf/2505.21568)  

**Abstract**: Voice cloning (VC)-resistant watermarking is an emerging technique for tracing and preventing unauthorized cloning. Existing methods effectively trace traditional VC models by training them on watermarked audio but fail in zero-shot VC scenarios, where models synthesize audio from an audio prompt without training. To address this, we propose VoiceMark, the first zero-shot VC-resistant watermarking method that leverages speaker-specific latents as the watermark carrier, allowing the watermark to transfer through the zero-shot VC process into the synthesized audio. Additionally, we introduce VC-simulated augmentations and VAD-based loss to enhance robustness against distortions. Experiments on multiple zero-shot VC models demonstrate that VoiceMark achieves over 95% accuracy in watermark detection after zero-shot VC synthesis, significantly outperforming existing methods, which only reach around 50%. See our code and demos at: this https URL 

---
# Towards Human-Like Trajectory Prediction for Autonomous Driving: A Behavior-Centric Approach 

**Authors**: Haicheng Liao, Zhenning Li, Guohui Zhang, Keqiang Li, Chengzhong Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21565)  

**Abstract**: Predicting the trajectories of vehicles is crucial for the development of autonomous driving (AD) systems, particularly in complex and dynamic traffic environments. In this study, we introduce HiT (Human-like Trajectory Prediction), a novel model designed to enhance trajectory prediction by incorporating behavior-aware modules and dynamic centrality measures. Unlike traditional methods that primarily rely on static graph structures, HiT leverages a dynamic framework that accounts for both direct and indirect interactions among traffic participants. This allows the model to capture the subtle yet significant influences of surrounding vehicles, enabling more accurate and human-like predictions. To evaluate HiT's performance, we conducted extensive experiments using diverse and challenging real-world datasets, including NGSIM, HighD, RounD, ApolloScape, and MoCAD++. The results demonstrate that HiT consistently outperforms other top models across multiple metrics, particularly excelling in scenarios involving aggressive driving behaviors. This research presents a significant step forward in trajectory prediction, offering a more reliable and interpretable approach for enhancing the safety and efficiency of fully autonomous driving systems. 

---
# Fog Intelligence for Network Anomaly Detection 

**Authors**: Kai Yang, Hui Ma, Shaoyu Dou  

**Link**: [PDF](https://arxiv.org/pdf/2505.21563)  

**Abstract**: Anomalies are common in network system monitoring. When manifested as network threats to be mitigated, service outages to be prevented, and security risks to be ameliorated, detecting such anomalous network behaviors becomes of great importance. However, the growing scale and complexity of the mobile communication networks, as well as the ever-increasing amount and dimensionality of the network surveillance data, make it extremely difficult to monitor a mobile network and discover abnormal network behaviors. Recent advances in machine learning allow for obtaining near-optimal solutions to complicated decision-making problems with many sources of uncertainty that cannot be accurately characterized by traditional mathematical models. However, most machine learning algorithms are centralized, which renders them inapplicable to a large-scale distributed wireless networks with tens of millions of mobile devices. In this article, we present fog intelligence, a distributed machine learning architecture that enables intelligent wireless network management. It preserves the advantage of both edge processing and centralized cloud computing. In addition, the proposed architecture is scalable, privacy-preserving, and well suited for intelligent management of a distributed wireless network. 

---
# Enhancing Selection of Climate Tech Startups with AI -- A Case Study on Integrating Human and AI Evaluations in the ClimaTech Great Global Innovation Challenge 

**Authors**: Jennifer Turliuk, Alejandro Sevilla, Daniela Gorza, Tod Hynes  

**Link**: [PDF](https://arxiv.org/pdf/2505.21562)  

**Abstract**: This case study examines the ClimaTech Great Global Innovation Challenge's approach to selecting climate tech startups by integrating human and AI evaluations. The competition aimed to identify top startups and enhance the accuracy and efficiency of the selection process through a hybrid model. Research shows data-driven approaches help VC firms reduce bias and improve decision-making. Machine learning models have outperformed human investors in deal screening, helping identify high-potential startups. Incorporating AI aimed to ensure more equitable and objective evaluations.
The methodology included three phases: initial AI review, semi-finals judged by humans, and finals using a hybrid weighting. In phase one, 57 applications were scored by an AI tool built with StackAI and OpenAI's GPT-4o, and the top 36 advanced. In the semi-finals, human judges, unaware of AI scores, evaluated startups on team quality, market potential, and technological innovation. Each score - human or AI - was weighted equally, resulting in 75 percent human and 25 percent AI influence. In the finals, with five human judges, weighting shifted to 83.3 percent human and 16.7 percent AI. There was a moderate positive correlation between AI and human scores - Spearman's = 0.47 - indicating general alignment with key differences. Notably, the final four startups, selected mainly by humans, were among those rated highest by the AI. This highlights the complementary nature of AI and human judgment. The study shows that hybrid models can streamline and improve startup assessments. The ClimaTech approach offers a strong framework for future competitions by combining human expertise with AI capabilities. 

---
# Streamlining Resilient Kubernetes Autoscaling with Multi-Agent Systems via an Automated Online Design Framework 

**Authors**: Julien Soulé, Jean-Paul Jamont, Michel Occello, Louis-Marie Traonouez, Paul Théron  

**Link**: [PDF](https://arxiv.org/pdf/2505.21559)  

**Abstract**: In cloud-native systems, Kubernetes clusters with interdependent services often face challenges to their operational resilience due to poor workload management issues such as resource blocking, bottlenecks, or continuous pod crashes. These vulnerabilities are further amplified in adversarial scenarios, such as Distributed Denial-of-Service attacks (DDoS). Conventional Horizontal Pod Autoscaling (HPA) approaches struggle to address such dynamic conditions, while reinforcement learning-based methods, though more adaptable, typically optimize single goals like latency or resource usage, neglecting broader failure scenarios. We propose decomposing the overarching goal of maintaining operational resilience into failure-specific sub-goals delegated to collaborative agents, collectively forming an HPA Multi-Agent System (MAS). We introduce an automated, four-phase online framework for HPA MAS design: 1) modeling a digital twin built from cluster traces; 2) training agents in simulation using roles and missions tailored to failure contexts; 3) analyzing agent behaviors for explainability; and 4) transferring learned policies to the real cluster. Experimental results demonstrate that the generated HPA MASs outperform three state-of-the-art HPA systems in sustaining operational resilience under various adversarial conditions in a proposed complex cluster. 

---
# A Novel Convolutional Neural Network-Based Framework for Complex Multiclass Brassica Seed Classification 

**Authors**: Elhoucine Elfatimia, Recep Eryigitb, Lahcen Elfatimi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21558)  

**Abstract**: Agricultural research has accelerated in recent years, yet farmers often lack the time and resources for on-farm research due to the demands of crop production and farm operations. Seed classification offers valuable insights into quality control, production efficiency, and impurity detection. Early identification of seed types is critical to reducing the cost and risk associated with field emergence, which can lead to yield losses or disruptions in downstream processes like harvesting. Seed sampling supports growers in monitoring and managing seed quality, improving precision in determining seed purity levels, guiding management adjustments, and enhancing yield estimations. This study proposes a novel convolutional neural network (CNN)-based framework for the efficient classification of ten common Brassica seed types. The approach addresses the inherent challenge of texture similarity in seed images using a custom-designed CNN architecture. The model's performance was evaluated against several pre-trained state-of-the-art architectures, with adjustments to layer configurations for optimized classification. Experimental results using our collected Brassica seed dataset demonstrate that the proposed model achieved a high accuracy rate of 93 percent. 

---
# Analytical Calculation of Weights Convolutional Neural Network 

**Authors**: Polad Geidarov  

**Link**: [PDF](https://arxiv.org/pdf/2505.21557)  

**Abstract**: This paper presents an algorithm for analytically calculating the weights and thresholds of convolutional neural networks (CNNs) without using standard training procedures. The algorithm enables the determination of CNN parameters based on just 10 selected images from the MNIST dataset, each representing a digit from 0 to 9. As part of the method, the number of channels in CNN layers is also derived analytically. A software module was implemented in C++ Builder, and a series of experiments were conducted using the MNIST dataset. Results demonstrate that the analytically computed CNN can recognize over half of 1000 handwritten digit images without any training, achieving inference in fractions of a second. These findings suggest that CNNs can be constructed and applied directly for classification tasks without training, using purely analytical computation of weights. 

---
# Benign-to-Toxic Jailbreaking: Inducing Harmful Responses from Harmless Prompts 

**Authors**: Hee-Seon Kim, Minbeom Kim, Wonjun Lee, Kihyun Kim, Changick Kim  

**Link**: [PDF](https://arxiv.org/pdf/2505.21556)  

**Abstract**: Optimization-based jailbreaks typically adopt the Toxic-Continuation setting in large vision-language models (LVLMs), following the standard next-token prediction objective. In this setting, an adversarial image is optimized to make the model predict the next token of a toxic prompt. However, we find that the Toxic-Continuation paradigm is effective at continuing already-toxic inputs, but struggles to induce safety misalignment when explicit toxic signals are absent. We propose a new paradigm: Benign-to-Toxic (B2T) jailbreak. Unlike prior work, we optimize adversarial images to induce toxic outputs from benign conditioning. Since benign conditioning contains no safety violations, the image alone must break the model's safety mechanisms. Our method outperforms prior approaches, transfers in black-box settings, and complements text-based jailbreaks. These results reveal an underexplored vulnerability in multimodal alignment and introduce a fundamentally new direction for jailbreak approaches. 

---
# MetaSTNet: Multimodal Meta-learning for Cellular Traffic Conformal Prediction 

**Authors**: Hui Ma, Kai Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21553)  

**Abstract**: Network traffic prediction techniques have attracted much attention since they are valuable for network congestion control and user experience improvement. While existing prediction techniques can achieve favorable performance when there is sufficient training data, it remains a great challenge to make accurate predictions when only a small amount of training data is available. To tackle this problem, we propose a deep learning model, entitled MetaSTNet, based on a multimodal meta-learning framework. It is an end-to-end network architecture that trains the model in a simulator and transfers the meta-knowledge to a real-world environment, which can quickly adapt and obtain accurate predictions on a new task with only a small amount of real-world training data. In addition, we further employ cross conformal prediction to assess the calibrated prediction intervals. Extensive experiments have been conducted on real-world datasets to illustrate the efficiency and effectiveness of MetaSTNet. 

---
# WhisperD: Dementia Speech Recognition and Filler Word Detection with Whisper 

**Authors**: Emmanuel Akinrintoyo, Nadine Abdelhalim, Nicole Salomons  

**Link**: [PDF](https://arxiv.org/pdf/2505.21551)  

**Abstract**: Whisper fails to correctly transcribe dementia speech because persons with dementia (PwDs) often exhibit irregular speech patterns and disfluencies such as pauses, repetitions, and fragmented sentences. It was trained on standard speech and may have had little or no exposure to dementia-affected speech. However, correct transcription is vital for dementia speech for cost-effective diagnosis and the development of assistive technology. In this work, we fine-tune Whisper with the open-source dementia speech dataset (DementiaBank) and our in-house dataset to improve its word error rate (WER). The fine-tuning also includes filler words to ascertain the filler inclusion rate (FIR) and F1 score. The fine-tuned models significantly outperformed the off-the-shelf models. The medium-sized model achieved a WER of 0.24, outperforming previous work. Similarly, there was a notable generalisability to unseen data and speech patterns. 

---
# Collaborative Agentic AI Needs Interoperability Across Ecosystems 

**Authors**: Rishi Sharma, Martijn de Vos, Pradyumna Chari, Ramesh Raskar, Anne-Marie Kermarrec  

**Link**: [PDF](https://arxiv.org/pdf/2505.21550)  

**Abstract**: Collaborative agentic AI is projected to transform entire industries by enabling AI-powered agents to autonomously perceive, plan, and act within digital environments. Yet, current solutions in this field are all built in isolation, and we are rapidly heading toward a landscape of fragmented, incompatible ecosystems. In this position paper, we argue that interoperability, achieved by the adoption of minimal standards, is essential to ensure open, secure, web-scale, and widely-adopted agentic ecosystems. To this end, we devise a minimal architectural foundation for collaborative agentic AI, named Web of Agents, which is composed of four components: agent-to-agent messaging, interaction interoperability, state management, and agent discovery. Web of Agents adopts existing standards and reuses existing infrastructure where possible. With Web of Agents, we take the first but critical step toward interoperable agentic systems and offer a pragmatic path forward before ecosystem fragmentation becomes the norm. 

---
# Fluent but Culturally Distant: Can Regional Training Teach Cultural Understanding? 

**Authors**: Dhruv Agarwal, Anya Shukla, Sunayana Sitaram, Aditya Vashistha  

**Link**: [PDF](https://arxiv.org/pdf/2505.21548)  

**Abstract**: Large language models (LLMs) are used around the world but exhibit Western cultural tendencies. To address this cultural misalignment, many countries have begun developing "regional" LLMs tailored to local communities. Yet it remains unclear whether these models merely speak the language of their users or also reflect their cultural values and practices. Using India as a case study, we evaluate five Indic and five global LLMs along two key dimensions: values (via the Inglehart-Welzel map and GlobalOpinionQA) and practices (via CulturalBench and NormAd). Across all four tasks, we find that Indic models do not align more closely with Indian cultural norms than global models. In fact, an average American person is a better proxy for Indian cultural values than any Indic model. Even prompting strategies fail to meaningfully improve alignment. Ablations show that regional fine-tuning does not enhance cultural competence and may in fact hurt it by impeding recall of existing knowledge. We trace this failure to the scarcity of high-quality, untranslated, and culturally grounded pretraining and fine-tuning data. Our study positions cultural evaluation as a first-class requirement alongside multilingual benchmarks and offers a reusable methodology for developers. We call for deeper investments in culturally representative data to build and evaluate truly sovereign LLMs. 

---
# Image Tokens Matter: Mitigating Hallucination in Discrete Tokenizer-based Large Vision-Language Models via Latent Editing 

**Authors**: Weixing Wang, Zifeng Ding, Jindong Gu, Rui Cao, Christoph Meinel, Gerard de Melo, Haojin Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21547)  

**Abstract**: Large Vision-Language Models (LVLMs) with discrete image tokenizers unify multimodal representations by encoding visual inputs into a finite set of tokens. Despite their effectiveness, we find that these models still hallucinate non-existent objects. We hypothesize that this may be due to visual priors induced during training: When certain image tokens frequently co-occur in the same spatial regions and represent shared objects, they become strongly associated with the verbalizations of those objects. As a result, the model may hallucinate by evoking visually absent tokens that often co-occur with present ones. To test this assumption, we construct a co-occurrence graph of image tokens using a segmentation dataset and employ a Graph Neural Network (GNN) with contrastive learning followed by a clustering method to group tokens that frequently co-occur in similar visual contexts. We find that hallucinations predominantly correspond to clusters whose tokens dominate the input, and more specifically, that the visually absent tokens in those clusters show much higher correlation with hallucinated objects compared to tokens present in the image. Based on this observation, we propose a hallucination mitigation method that suppresses the influence of visually absent tokens by modifying latent image embeddings during generation. Experiments show our method reduces hallucinations while preserving expressivity. Code is available at this https URL 

---
# DiffDecompose: Layer-Wise Decomposition of Alpha-Composited Images via Diffusion Transformers 

**Authors**: Zitong Wang, Hang Zhao, Qianyu Zhou, Xuequan Lu, Xiangtai Li, Yiren Song  

**Link**: [PDF](https://arxiv.org/pdf/2505.21541)  

**Abstract**: Diffusion models have recently motivated great success in many generation tasks like object removal. Nevertheless, existing image decomposition methods struggle to disentangle semi-transparent or transparent layer occlusions due to mask prior dependencies, static object assumptions, and the lack of datasets. In this paper, we delve into a novel task: Layer-Wise Decomposition of Alpha-Composited Images, aiming to recover constituent layers from single overlapped images under the condition of semi-transparent/transparent alpha layer non-linear occlusion. To address challenges in layer ambiguity, generalization, and data scarcity, we first introduce AlphaBlend, the first large-scale and high-quality dataset for transparent and semi-transparent layer decomposition, supporting six real-world subtasks (e.g., translucent flare removal, semi-transparent cell decomposition, glassware decomposition). Building on this dataset, we present DiffDecompose, a diffusion Transformer-based framework that learns the posterior over possible layer decompositions conditioned on the input image, semantic prompts, and blending type. Rather than regressing alpha mattes directly, DiffDecompose performs In-Context Decomposition, enabling the model to predict one or multiple layers without per-layer supervision, and introduces Layer Position Encoding Cloning to maintain pixel-level correspondence across layers. Extensive experiments on the proposed AlphaBlend dataset and public LOGO dataset verify the effectiveness of DiffDecompose. The code and dataset will be available upon paper acceptance. Our code will be available at: this https URL. 

---
# Equivariant Flow Matching for Point Cloud Assembly 

**Authors**: Ziming Wang, Nan Xue, Rebecka Jörnsten  

**Link**: [PDF](https://arxiv.org/pdf/2505.21539)  

**Abstract**: The goal of point cloud assembly is to reconstruct a complete 3D shape by aligning multiple point cloud pieces. This work presents a novel equivariant solver for assembly tasks based on flow matching models. We first theoretically show that the key to learning equivariant distributions via flow matching is to learn related vector fields. Based on this result, we propose an assembly model, called equivariant diffusion assembly (Eda), which learns related vector fields conditioned on the input pieces. We further construct an equivariant path for Eda, which guarantees high data efficiency of the training process. Our numerical results show that Eda is highly competitive on practical datasets, and it can even handle the challenging situation where the input pieces are non-overlapped. 

---
# Caption This, Reason That: VLMs Caught in the Middle 

**Authors**: Zihan Weng, Lucas Gomez, Taylor Whittington Webb, Pouya Bashivan  

**Link**: [PDF](https://arxiv.org/pdf/2505.21538)  

**Abstract**: Vision-Language Models (VLMs) have shown remarkable progress in visual understanding in recent years. Yet, they still lag behind human capabilities in specific visual tasks such as counting or relational reasoning. To understand the underlying limitations, we adopt methodologies from cognitive science, analyzing VLM performance along core cognitive axes: Perception, Attention, and Memory. Using a suite of tasks targeting these abilities, we evaluate state-of-the-art VLMs, including GPT-4o. Our analysis reveals distinct cognitive profiles: while advanced models approach ceiling performance on some tasks (e.g. category identification), a significant gap persists, particularly in tasks requiring spatial understanding or selective attention. Investigating the source of these failures and potential methods for improvement, we employ a vision-text decoupling analysis, finding that models struggling with direct visual reasoning show marked improvement when reasoning over their own generated text captions. These experiments reveal a strong need for improved VLM Chain-of-Thought (CoT) abilities, even in models that consistently exceed human performance. Furthermore, we demonstrate the potential of targeted fine-tuning on composite visual reasoning tasks and show that fine-tuning smaller VLMs substantially improves core cognitive abilities. While this improvement does not translate to large enhancements on challenging, out-of-distribution benchmarks, we show broadly that VLM performance on our datasets strongly correlates with performance on these other benchmarks. Our work provides a detailed analysis of VLM cognitive strengths and weaknesses and identifies key bottlenecks in simultaneous perception and reasoning while also providing an effective and simple solution. 

---
# OpenReview Should be Protected and Leveraged as a Community Asset for Research in the Era of Large Language Models 

**Authors**: Hao Sun, Yunyi Shen, Mihaela van der Schaar  

**Link**: [PDF](https://arxiv.org/pdf/2505.21537)  

**Abstract**: In the era of large language models (LLMs), high-quality, domain-rich, and continuously evolving datasets capturing expert-level knowledge, core human values, and reasoning are increasingly valuable. This position paper argues that OpenReview -- the continually evolving repository of research papers, peer reviews, author rebuttals, meta-reviews, and decision outcomes -- should be leveraged more broadly as a core community asset for advancing research in the era of LLMs. We highlight three promising areas in which OpenReview can uniquely contribute: enhancing the quality, scalability, and accountability of peer review processes; enabling meaningful, open-ended benchmarks rooted in genuine expert deliberation; and supporting alignment research through real-world interactions reflecting expert assessment, intentions, and scientific values. To better realize these opportunities, we suggest the community collaboratively explore standardized benchmarks and usage guidelines around OpenReview, inviting broader dialogue on responsible data use, ethical considerations, and collective stewardship. 

---
# Is Attention Required for Transformer Inference? Explore Function-preserving Attention Replacement 

**Authors**: Yuxin Ren, Maxwell D Collins, Miao Hu, Huanrui Yang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21535)  

**Abstract**: While transformers excel across vision and language pretraining tasks, their reliance on attention mechanisms poses challenges for inference efficiency, especially on edge and embedded accelerators with limited parallelism and memory bandwidth. Hinted by the observed redundancy of attention at inference time, we hypothesize that though the model learns complicated token dependency through pretraining, the inference-time sequence-to-sequence mapping in each attention layer is actually ''simple'' enough to be represented with a much cheaper function. In this work, we explore FAR, a Function-preserving Attention Replacement framework that replaces all attention blocks in pretrained transformers with learnable sequence-to-sequence modules, exemplified by an LSTM. FAR optimize a multi-head LSTM architecture with a block-wise distillation objective and a global structural pruning framework to achieve a family of efficient LSTM-based models from pretrained transformers. We validate FAR on the DeiT vision transformer family and demonstrate that it matches the accuracy of the original models on ImageNet and multiple downstream tasks with reduced parameters and latency. Further analysis shows that FAR preserves the semantic token relationships and the token-to-token correlation learned in the transformer's attention module. 

---
# Uncovering Bottlenecks and Optimizing Scientific Lab Workflows with Cycle Time Reduction Agents 

**Authors**: Yao Fehlis  

**Link**: [PDF](https://arxiv.org/pdf/2505.21534)  

**Abstract**: Scientific laboratories, particularly those in pharmaceutical and biotechnology companies, encounter significant challenges in optimizing workflows due to the complexity and volume of tasks such as compound screening and assay execution. We introduce Cycle Time Reduction Agents (CTRA), a LangGraph-based agentic workflow designed to automate the analysis of lab operational metrics. CTRA comprises three main components: the Question Creation Agent for initiating analysis, Operational Metrics Agents for data extraction and validation, and Insights Agents for reporting and visualization, identifying bottlenecks in lab processes. This paper details CTRA's architecture, evaluates its performance on a lab dataset, and discusses its potential to accelerate pharmaceutical and biotechnological development. CTRA offers a scalable framework for reducing cycle times in scientific labs. 

---
# EvidenceMoE: A Physics-Guided Mixture-of-Experts with Evidential Critics for Advancing Fluorescence Light Detection and Ranging in Scattering Media 

**Authors**: Ismail Erbas, Ferhat Demirkiran, Karthik Swaminathan, Naigang Wang, Navid Ibtehaj Nizam, Stefan T. Radev, Kaoutar El Maghraoui, Xavier Intes, Vikas Pandey  

**Link**: [PDF](https://arxiv.org/pdf/2505.21532)  

**Abstract**: Fluorescence LiDAR (FLiDAR), a Light Detection and Ranging (LiDAR) technology employed for distance and depth estimation across medical, automotive, and other fields, encounters significant computational challenges in scattering media. The complex nature of the acquired FLiDAR signal, particularly in such environments, makes isolating photon time-of-flight (related to target depth) and intrinsic fluorescence lifetime exceptionally difficult, thus limiting the effectiveness of current analytical and computational methodologies. To overcome this limitation, we present a Physics-Guided Mixture-of-Experts (MoE) framework tailored for specialized modeling of diverse temporal components. In contrast to the conventional MoE approaches our expert models are informed by underlying physics, such as the radiative transport equation governing photon propagation in scattering media. Central to our approach is EvidenceMoE, which integrates Evidence-Based Dirichlet Critics (EDCs). These critic models assess the reliability of each expert's output by providing per-expert quality scores and corrective feedback. A Decider Network then leverages this information to fuse expert predictions into a robust final estimate adaptively. We validate our method using realistically simulated Fluorescence LiDAR (FLiDAR) data for non-invasive cancer cell depth detection generated from photon transport models in tissue. Our framework demonstrates strong performance, achieving a normalized root mean squared error (NRMSE) of 0.030 for depth estimation and 0.074 for fluorescence lifetime. 

---
# How Much Do Large Language Models Know about Human Motion? A Case Study in 3D Avatar Control 

**Authors**: Kunhang Li, Jason Naradowsky, Yansong Feng, Yusuke Miyao  

**Link**: [PDF](https://arxiv.org/pdf/2505.21531)  

**Abstract**: We explore Large Language Models (LLMs)' human motion knowledge through 3D avatar control. Given a motion instruction, we prompt LLMs to first generate a high-level movement plan with consecutive steps (High-level Planning), then specify body part positions in each step (Low-level Planning), which we linearly interpolate into avatar animations as a clear verification lens for human evaluators. Through carefully designed 20 representative motion instructions with full coverage of basic movement primitives and balanced body part usage, we conduct comprehensive evaluations including human assessment of both generated animations and high-level movement plans, as well as automatic comparison with oracle positions in low-level planning. We find that LLMs are strong at interpreting the high-level body movements but struggle with precise body part positioning. While breaking down motion queries into atomic components improves planning performance, LLMs have difficulty with multi-step movements involving high-degree-of-freedom body parts. Furthermore, LLMs provide reasonable approximation for general spatial descriptions, but fail to handle precise spatial specifications in text, and the precise spatial-temporal parameters needed for avatar control. Notably, LLMs show promise in conceptualizing creative motions and distinguishing culturally-specific motion patterns. 

---
# High-Fidelity Functional Ultrasound Reconstruction via A Visual Auto-Regressive Framework 

**Authors**: Xuhang Chen, Zhuo Li, Yanyan Shen, Mufti Mahmud, Hieu Pham, Chi-Man Pun, Shuqiang Wang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21530)  

**Abstract**: Functional ultrasound (fUS) imaging provides exceptional spatiotemporal resolution for neurovascular mapping, yet its practical application is significantly hampered by critical challenges. Foremost among these are data scarcity, arising from ethical considerations and signal degradation through the cranium, which collectively limit dataset diversity and compromise the fairness of downstream machine learning models. 

---
# UniDB++: Fast Sampling of Unified Diffusion Bridge 

**Authors**: Mokai Pan, Kaizhen Zhu, Yuexin Ma, Yanwei Fu, Jingyi Yu, Jingya Wang, Ye Shi  

**Link**: [PDF](https://arxiv.org/pdf/2505.21528)  

**Abstract**: Diffusion Bridges enable transitions between arbitrary distributions, with the Unified Diffusion Bridge (UniDB) framework achieving high-fidelity image generation via a Stochastic Optimal Control (SOC) formulation. However, UniDB's reliance on iterative Euler sampling methods results in slow, computationally expensive inference, while existing acceleration techniques for diffusion or diffusion bridge models fail to address its unique challenges: missing terminal mean constraints and SOC-specific penalty coefficients in its SDEs. We present UniDB++, a training-free sampling algorithm that significantly improves upon these limitations. The method's key advancement comes from deriving exact closed-form solutions for UniDB's reverse-time SDEs, effectively reducing the error accumulation inherent in Euler approximations and enabling high-quality generation with up to 20$\times$ fewer sampling steps. This method is further complemented by replacing conventional noise prediction with a more stable data prediction model, along with an SDE-Corrector mechanism that maintains perceptual quality for low-step regimes (5-10 steps). Additionally, we demonstrate that UniDB++ aligns with existing diffusion bridge acceleration methods by evaluating their update rules, and UniDB++ can recover DBIMs as special cases under some theoretical conditions. Experiments demonstrate UniDB++'s state-of-the-art performance in image restoration tasks, outperforming Euler-based methods in fidelity and speed while reducing inference time significantly. This work bridges the gap between theoretical generality and practical efficiency in SOC-driven diffusion bridge models. Our code is available at this https URL. 

---
# VietASR: Achieving Industry-level Vietnamese ASR with 50-hour labeled data and Large-Scale Speech Pretraining 

**Authors**: Jianheng Zhuo, Yifan Yang, Yiwen Shao, Yong Xu, Dong Yu, Kai Yu, Xie Chen  

**Link**: [PDF](https://arxiv.org/pdf/2505.21527)  

**Abstract**: Automatic speech recognition (ASR) has made remarkable progress but heavily relies on large-scale labeled data, which is scarce for low-resource languages like Vietnamese. While existing systems such as Whisper, USM, and MMS achieve promising performance, their efficacy remains inadequate in terms of training costs, latency, and accessibility. To address these issues, we propose VietASR, a novel ASR training pipeline that leverages vast amounts of unlabeled data and a small set of labeled data. Through multi-iteration ASR-biased self-supervised learning on a large-scale unlabeled dataset, VietASR offers a cost-effective and practical solution for enhancing ASR performance. Experiments demonstrate that pre-training on 70,000-hour unlabeled data and fine-tuning on merely 50-hour labeled data yield a lightweight but powerful ASR model. It outperforms Whisper Large-v3 and commercial ASR systems on real-world data. Our code and models will be open-sourced to facilitate research in low-resource ASR. 

---
# Temporal Restoration and Spatial Rewiring for Source-Free Multivariate Time Series Domain Adaptation 

**Authors**: Peiliang Gong, Yucheng Wang, Min Wu, Zhenghua Chen, Xiaoli Li, Daoqiang Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2505.21525)  

**Abstract**: Source-Free Domain Adaptation (SFDA) aims to adapt a pre-trained model from an annotated source domain to an unlabelled target domain without accessing the source data, thereby preserving data privacy. While existing SFDA methods have proven effective in reducing reliance on source data, they struggle to perform well on multivariate time series (MTS) due to their failure to consider the intrinsic spatial correlations inherent in MTS data. These spatial correlations are crucial for accurately representing MTS data and preserving invariant information across domains. To address this challenge, we propose Temporal Restoration and Spatial Rewiring (TERSE), a novel and concise SFDA method tailored for MTS data. Specifically, TERSE comprises a customized spatial-temporal feature encoder designed to capture the underlying spatial-temporal characteristics, coupled with both temporal restoration and spatial rewiring tasks to reinstate latent representations of the temporally masked time series and the spatially masked correlated structures. During the target adaptation phase, the target encoder is guided to produce spatially and temporally consistent features with the source domain by leveraging the source pre-trained temporal restoration and spatial rewiring networks. Therefore, TERSE can effectively model and transfer spatial-temporal dependencies across domains, facilitating implicit feature alignment. In addition, as the first approach to simultaneously consider spatial-temporal consistency in MTS-SFDA, TERSE can also be integrated as a versatile plug-and-play module into established SFDA methods. Extensive experiments on three real-world time series datasets demonstrate the effectiveness and versatility of our approach. 

---
# More Thinking, Less Seeing? Assessing Amplified Hallucination in Multimodal Reasoning Models 

**Authors**: Chengzhi Liu, Zhongxing Xu, Qingyue Wei, Juncheng Wu, James Zou, Xin Eric Wang, Yuyin Zhou, Sheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21523)  

**Abstract**: Test-time compute has empowered multimodal large language models to generate extended reasoning chains, yielding strong performance on tasks such as multimodal math reasoning. However, this improved reasoning ability often comes with increased hallucination: as generations become longer, models tend to drift away from image-grounded content and rely more heavily on language priors. Attention analysis shows that longer reasoning chains lead to reduced focus on visual inputs, which contributes to hallucination. To systematically study this phenomenon, we introduce RH-AUC, a metric that quantifies how a model's perception accuracy changes with reasoning length, allowing us to evaluate whether the model preserves visual grounding during reasoning. We also release RH-Bench, a diagnostic benchmark that spans a variety of multimodal tasks, designed to assess the trade-off between reasoning ability and hallucination. Our analysis reveals that (i) larger models typically achieve a better balance between reasoning and perception, and (ii) this balance is influenced more by the types and domains of training data than by its overall volume. These findings underscore the importance of evaluation frameworks that jointly consider both reasoning quality and perceptual fidelity. 

---
# CIM-NET: A Video Denoising Deep Neural Network Model Optimized for Computing-in-Memory Architectures 

**Authors**: Shan Gao, Zhiqiang Wu, Yawen Niu, Xiaotao Li, Qingqing Xu  

**Link**: [PDF](https://arxiv.org/pdf/2505.21522)  

**Abstract**: While deep neural network (DNN)-based video denoising has demonstrated significant performance, deploying state-of-the-art models on edge devices remains challenging due to stringent real-time and energy efficiency requirements. Computing-in-Memory (CIM) chips offer a promising solution by integrating computation within memory cells, enabling rapid matrix-vector multiplication (MVM). However, existing DNN models are often designed without considering CIM architectural constraints, thus limiting their acceleration potential during inference. To address this, we propose a hardware-algorithm co-design framework incorporating two innovations: (1) a CIM-Aware Architecture, CIM-NET, optimized for large receptive field operation and CIM's crossbar-based MVM acceleration; and (2) a pseudo-convolutional operator, CIM-CONV, used within CIM-NET to integrate slide-based processing with fully connected transformations for high-quality feature extraction and reconstruction. This framework significantly reduces the number of MVM operations, improving inference speed on CIM chips while maintaining competitive performance. Experimental results indicate that, compared to the conventional lightweight model FastDVDnet, CIM-NET substantially reduces MVM operations with a slight decrease in denoising performance. With a stride value of 8, CIM-NET reduces MVM operations to 1/77th of the original, while maintaining competitive PSNR (35.11 dB vs. 35.56 dB 

---
# Do DeepFake Attribution Models Generalize? 

**Authors**: Spiros Baxavanakis, Manos Schinas, Symeon Papadopoulos  

**Link**: [PDF](https://arxiv.org/pdf/2505.21520)  

**Abstract**: Recent advancements in DeepFake generation, along with the proliferation of open-source tools, have significantly lowered the barrier for creating synthetic media. This trend poses a serious threat to the integrity and authenticity of online information, undermining public trust in institutions and media. State-of-the-art research on DeepFake detection has primarily focused on binary detection models. A key limitation of these models is that they treat all manipulation techniques as equivalent, despite the fact that different methods introduce distinct artifacts and visual cues. Only a limited number of studies explore DeepFake attribution models, although such models are crucial in practical settings. By providing the specific manipulation method employed, these models could enhance both the perceived trustworthiness and explainability for end users. In this work, we leverage five state-of-the-art backbone models and conduct extensive experiments across six DeepFake datasets. First, we compare binary and multi-class models in terms of cross-dataset generalization. Second, we examine the accuracy of attribution models in detecting seen manipulation methods in unknown datasets, hence uncovering data distribution shifts on the same DeepFake manipulations. Last, we assess the effectiveness of contrastive methods in improving cross-dataset generalization performance. Our findings indicate that while binary models demonstrate better generalization abilities, larger models, contrastive methods, and higher data quality can lead to performance improvements in attribution models. The code of this work is available on GitHub. 

---
# Enhancing Vision Transformer Explainability Using Artificial Astrocytes 

**Authors**: Nicolas Echevarrieta-Catalan, Ana Ribas-Rodriguez, Francisco Cedron, Odelia Schwartz, Vanessa Aguiar-Pulido  

**Link**: [PDF](https://arxiv.org/pdf/2505.21513)  

**Abstract**: Machine learning models achieve high precision, but their decision-making processes often lack explainability. Furthermore, as model complexity increases, explainability typically decreases. Existing efforts to improve explainability primarily involve developing new eXplainable artificial intelligence (XAI) techniques or incorporating explainability constraints during training. While these approaches yield specific improvements, their applicability remains limited. In this work, we propose the Vision Transformer with artificial Astrocytes (ViTA). This training-free approach is inspired by neuroscience and enhances the reasoning of a pretrained deep neural network to generate more human-aligned explanations. We evaluated our approach employing two well-known XAI techniques, Grad-CAM and Grad-CAM++, and compared it to a standard Vision Transformer (ViT). Using the ClickMe dataset, we quantified the similarity between the heatmaps produced by the XAI techniques and a (human-aligned) ground truth. Our results consistently demonstrate that incorporating artificial astrocytes enhances the alignment of model explanations with human perception, leading to statistically significant improvements across all XAI techniques and metrics utilized. 

---
# Conformance Checking for Less: Efficient Conformance Checking for Long Event Sequences 

**Authors**: Eli Bogdanov, Izack Cohen, Avigdor Gal  

**Link**: [PDF](https://arxiv.org/pdf/2505.21506)  

**Abstract**: Long event sequences (termed traces) and large data logs that originate from sensors and prediction models are becoming increasingly common in our data-rich world. In such scenarios, conformance checking-validating a data log against an expected system behavior (the process model) can become computationally infeasible due to the exponential complexity of finding an optimal alignment. To alleviate scalability challenges for this task, we propose ConLES, a sliding-window conformance checking approach for long event sequences that preserves the interpretability of alignment-based methods. ConLES partitions traces into manageable subtraces and iteratively aligns each against the expected behavior, leading to significant reduction of the search space while maintaining overall accuracy. We use global information that captures structural properties of both the trace and the process model, enabling informed alignment decisions and discarding unpromising alignments, even if they appear locally optimal. Performance evaluations across multiple datasets highlight that ConLES outperforms the leading optimal and heuristic algorithms for long traces, consistently achieving the optimal or near-optimal solution. Unlike other conformance methods that struggle with long event sequences, ConLES significantly reduces the search space, scales efficiently, and uniquely supports both predefined and discovered process models, making it a viable and leading option for conformance checking of long event sequences. 

---
# Offset Unlearning for Large Language Models 

**Authors**: James Y. Huang, Wenxuan Zhou, Fei Wang, Fred Morstatter, Sheng Zhang, Hoifung Poon, Muhao Chen  

**Link**: [PDF](https://arxiv.org/pdf/2404.11045)  

**Abstract**: Despite the strong capabilities of Large Language Models (LLMs) to acquire knowledge from their training corpora, the memorization of sensitive information in the corpora such as copyrighted, biased, and private content has led to ethical and legal concerns. In response to these challenges, unlearning has emerged as a potential remedy for LLMs affected by problematic training data. However, previous unlearning techniques are either not applicable to black-box LLMs due to required access to model internal weights, or violate data protection principles by retaining sensitive data for inference-time correction. We propose {\delta}-Unlearning, an offset unlearning framework for black-box LLMs. Instead of tuning the black-box LLM itself, {\delta}-Unlearning learns the logit offset needed for unlearning by contrasting the logits from a pair of smaller models. Experiments demonstrate that {\delta}- Unlearning can effectively unlearn target data while maintaining similar or even stronger performance on general out-of-forget-scope tasks. {\delta}-Unlearning also effectively incorporates different unlearning algorithms, making our approach a versatile solution to adapting various existing unlearning algorithms to black-box LLMs. 

---
