# Advancing Embodied Agent Security: From Safety Benchmarks to Input Moderation 

**Authors**: Ning Wang, Zihan Yan, Weiyang Li, Chuan Ma, He Chen, Tao Xiang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15699)  

**Abstract**: Embodied agents exhibit immense potential across a multitude of domains, making the assurance of their behavioral safety a fundamental prerequisite for their widespread deployment. However, existing research predominantly concentrates on the security of general large language models, lacking specialized methodologies for establishing safety benchmarks and input moderation tailored to embodied agents. To bridge this gap, this paper introduces a novel input moderation framework, meticulously designed to safeguard embodied agents. This framework encompasses the entire pipeline, including taxonomy definition, dataset curation, moderator architecture, model training, and rigorous evaluation. Notably, we introduce EAsafetyBench, a meticulously crafted safety benchmark engineered to facilitate both the training and stringent assessment of moderators specifically designed for embodied agents. Furthermore, we propose Pinpoint, an innovative prompt-decoupled input moderation scheme that harnesses a masked attention mechanism to effectively isolate and mitigate the influence of functional prompts on moderation tasks. Extensive experiments conducted on diverse benchmark datasets and models validate the feasibility and efficacy of the proposed approach. The results demonstrate that our methodologies achieve an impressive average detection accuracy of 94.58%, surpassing the performance of existing state-of-the-art techniques, alongside an exceptional moderation processing time of merely 0.002 seconds per instance. 

---
# Implementing Rational Choice Functions with LLMs and Measuring their Alignment with User Preferences 

**Authors**: Anna Karnysheva, Christian Drescher, Dietrich Klakow  

**Link**: [PDF](https://arxiv.org/pdf/2504.15719)  

**Abstract**: As large language models (LLMs) become integral to intelligent user interfaces (IUIs), their role as decision-making agents raises critical concerns about alignment. Although extensive research has addressed issues such as factuality, bias, and toxicity, comparatively little attention has been paid to measuring alignment to preferences, i.e., the relative desirability of different alternatives, a concept used in decision making, economics, and social choice theory. However, a reliable decision-making agent makes choices that align well with user preferences.
In this paper, we generalize existing methods that exploit LLMs for ranking alternative outcomes by addressing alignment with the broader and more flexible concept of user preferences, which includes both strict preferences and indifference among alternatives. To this end, we put forward design principles for using LLMs to implement rational choice functions, and provide the necessary tools to measure preference satisfaction. We demonstrate the applicability of our approach through an empirical study in a practical application of an IUI in the automotive domain. 

---
# Improving Human-AI Coordination through Adversarial Training and Generative Models 

**Authors**: Paresh Chaudhary, Yancheng Liang, Daphne Chen, Simon S. Du, Natasha Jaques  

**Link**: [PDF](https://arxiv.org/pdf/2504.15457)  

**Abstract**: Being able to cooperate with new people is an important component of many economically valuable AI tasks, from household robotics to autonomous driving. However, generalizing to novel humans requires training on data that captures the diversity of human behaviors. Adversarial training is one avenue for searching for such data and ensuring that agents are robust. However, it is difficult to apply in the cooperative setting because adversarial policies intentionally learn to sabotage the task instead of simulating valid cooperation partners. To address this challenge, we propose a novel strategy for overcoming self-sabotage that combines a pre-trained generative model to simulate valid cooperative agent policies with adversarial training to maximize regret. We call our method GOAT: Generative Online Adversarial Training. In this framework, the GOAT dynamically searches for and generates coordination strategies where the learning policy -- the Cooperator agent -- underperforms. GOAT enables better generalization by exposing the Cooperator to various challenging interaction scenarios. We maintain realistic coordination strategies by updating only the generative model's embedding while keeping its parameters frozen, thus avoiding adversarial exploitation. We evaluate GOAT with real human partners, and the results demonstrate state-of-the-art performance on the Overcooked benchmark, highlighting its effectiveness in generalizing to diverse human behaviors. 

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
# A Multi-Agent Framework for Automated Qinqiang Opera Script Generation Using Large Language Models 

**Authors**: Gengxian Cao, Fengyuan Li, Hong Duan, Ye Yang, Bofeng Wang, Donghe Li  

**Link**: [PDF](https://arxiv.org/pdf/2504.15552)  

**Abstract**: This paper introduces a novel multi-Agent framework that automates the end to end production of Qinqiang opera by integrating Large Language Models , visual generation, and Text to Speech synthesis. Three specialized agents collaborate in sequence: Agent1 uses an LLM to craft coherent, culturally grounded scripts;Agent2 employs visual generation models to render contextually accurate stage scenes; and Agent3 leverages TTS to produce synchronized, emotionally expressive vocal performances. In a case study on Dou E Yuan, the system achieved expert ratings of 3.8 for script fidelity, 3.5 for visual coherence, and 3.8 for speech accuracy-culminating in an overall score of 3.6, a 0.3 point improvement over a Single Agent baseline. Ablation experiments demonstrate that removing Agent2 or Agent3 leads to drops of 0.4 and 0.5 points, respectively, underscoring the value of modular collaboration. This work showcases how AI driven pipelines can streamline and scale the preservation of traditional performing arts, and points toward future enhancements in cross modal alignment, richer emotional nuance, and support for additional opera genres. 

---
# AGI Is Coming... Right After AI Learns to Play Wordle 

**Authors**: Sarath Shekkizhar, Romain Cosentino  

**Link**: [PDF](https://arxiv.org/pdf/2504.15434)  

**Abstract**: This paper investigates multimodal agents, in particular, OpenAI's Computer-User Agent (CUA), trained to control and complete tasks through a standard computer interface, similar to humans. We evaluated the agent's performance on the New York Times Wordle game to elicit model behaviors and identify shortcomings. Our findings revealed a significant discrepancy in the model's ability to recognize colors correctly depending on the context. The model had a $5.36\%$ success rate over several hundred runs across a week of Wordle. Despite the immense enthusiasm surrounding AI agents and their potential to usher in Artificial General Intelligence (AGI), our findings reinforce the fact that even simple tasks present substantial challenges for today's frontier AI models. We conclude with a discussion of the potential underlying causes, implications for future development, and research directions to improve these AI systems. 

---
# WALL-E 2.0: World Alignment by NeuroSymbolic Learning improves World Model-based LLM Agents 

**Authors**: Siyu Zhou, Tianyi Zhou, Yijun Yang, Guodong Long, Deheng Ye, Jing Jiang, Chengqi Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2504.15785)  

**Abstract**: Can we build accurate world models out of large language models (LLMs)? How can world models benefit LLM agents? The gap between the prior knowledge of LLMs and the specified environment's dynamics usually bottlenecks LLMs' performance as world models. To bridge the gap, we propose a training-free "world alignment" that learns an environment's symbolic knowledge complementary to LLMs. The symbolic knowledge covers action rules, knowledge graphs, and scene graphs, which are extracted by LLMs from exploration trajectories and encoded into executable codes to regulate LLM agents' policies. We further propose an RL-free, model-based agent "WALL-E 2.0" through the model-predictive control (MPC) framework. Unlike classical MPC requiring costly optimization on the fly, we adopt an LLM agent as an efficient look-ahead optimizer of future steps' actions by interacting with the neurosymbolic world model. While the LLM agent's strong heuristics make it an efficient planner in MPC, the quality of its planned actions is also secured by the accurate predictions of the aligned world model. They together considerably improve learning efficiency in a new environment. On open-world challenges in Mars (Minecraft like) and ALFWorld (embodied indoor environments), WALL-E 2.0 significantly outperforms existing methods, e.g., surpassing baselines in Mars by 16.1%-51.6% of success rate and by at least 61.7% in score. In ALFWorld, it achieves a new record 98% success rate after only 4 iterations. 

---
# LLMs are Greedy Agents: Effects of RL Fine-tuning on Decision-Making Abilities 

**Authors**: Thomas Schmied, Jörg Bornschein, Jordi Grau-Moya, Markus Wulfmeier, Razvan Pascanu  

**Link**: [PDF](https://arxiv.org/pdf/2504.16078)  

**Abstract**: The success of Large Language Models (LLMs) has sparked interest in various agentic applications. A key hypothesis is that LLMs, leveraging common sense and Chain-of-Thought (CoT) reasoning, can effectively explore and efficiently solve complex domains. However, LLM agents have been found to suffer from sub-optimal exploration and the knowing-doing gap, the inability to effectively act on knowledge present in the model. In this work, we systematically study why LLMs perform sub-optimally in decision-making scenarios. In particular, we closely examine three prevalent failure modes: greediness, frequency bias, and the knowing-doing gap. We propose mitigation of these shortcomings by fine-tuning via Reinforcement Learning (RL) on self-generated CoT rationales. Our experiments across multi-armed bandits, contextual bandits, and Tic-tac-toe, demonstrate that RL fine-tuning enhances the decision-making abilities of LLMs by increasing exploration and narrowing the knowing-doing gap. Finally, we study both classic exploration mechanisms, such as $\epsilon$-greedy, and LLM-specific approaches, such as self-correction and self-consistency, to enable more effective fine-tuning of LLMs for decision-making. 

---
# Solving Multi-Agent Safe Optimal Control with Distributed Epigraph Form MARL 

**Authors**: Songyuan Zhang, Oswin So, Mitchell Black, Zachary Serlin, Chuchu Fan  

**Link**: [PDF](https://arxiv.org/pdf/2504.15425)  

**Abstract**: Tasks for multi-robot systems often require the robots to collaborate and complete a team goal while maintaining safety. This problem is usually formalized as a constrained Markov decision process (CMDP), which targets minimizing a global cost and bringing the mean of constraint violation below a user-defined threshold. Inspired by real-world robotic applications, we define safety as zero constraint violation. While many safe multi-agent reinforcement learning (MARL) algorithms have been proposed to solve CMDPs, these algorithms suffer from unstable training in this setting. To tackle this, we use the epigraph form for constrained optimization to improve training stability and prove that the centralized epigraph form problem can be solved in a distributed fashion by each agent. This results in a novel centralized training distributed execution MARL algorithm named Def-MARL. Simulation experiments on 8 different tasks across 2 different simulators show that Def-MARL achieves the best overall performance, satisfies safety constraints, and maintains stable training. Real-world hardware experiments on Crazyflie quadcopters demonstrate the ability of Def-MARL to safely coordinate agents to complete complex collaborative tasks compared to other methods. 

---
# A Framework for Testing and Adapting REST APIs as LLM Tools 

**Authors**: Jayachandu Bandlamudi, Ritwik Chaudhuri, Neelamadhav Gantayat, Kushal Mukherjee, Prerna Agarwal, Renuka Sindhgatta, Sameep Mehta  

**Link**: [PDF](https://arxiv.org/pdf/2504.15546)  

**Abstract**: Large Language Models (LLMs) are enabling autonomous agents to perform complex workflows using external tools or functions, often provided via REST APIs in enterprise systems. However, directly utilizing these APIs as tools poses challenges due to their complex input schemas, elaborate responses, and often ambiguous documentation. Current benchmarks for tool testing do not adequately address these complexities, leading to a critical gap in evaluating API readiness for agent-driven automation. In this work, we present a novel testing framework aimed at evaluating and enhancing the readiness of REST APIs to function as tools for LLM-based agents. Our framework transforms apis as tools, generates comprehensive test cases for the APIs, translates tests cases into natural language instructions suitable for agents, enriches tool definitions and evaluates the agent's ability t correctly invoke the API and process its inputs and responses. To provide actionable insights, we analyze the outcomes of 750 test cases, presenting a detailed taxonomy of errors, including input misinterpretation, output handling inconsistencies, and schema mismatches. Additionally, we classify these test cases to streamline debugging and refinement of tool integrations. This work offers a foundational step toward enabling enterprise APIs as tools, improving their usability in agent-based applications. 

---
# A biologically Inspired Trust Model for Open Multi-Agent Systems that is Resilient to Rapid Performance Fluctuations 

**Authors**: Zoi Lygizou, Dimitris Kalles  

**Link**: [PDF](https://arxiv.org/pdf/2504.15301)  

**Abstract**: Trust management provides an alternative solution for securing open, dynamic, and distributed multi-agent systems, where conventional cryptographic methods prove to be impractical. However, existing trust models face challenges related to agent mobility, changing behaviors, and the cold start problem. To address these issues we introduced a biologically inspired trust model in which trustees assess their own capabilities and store trust data locally. This design improves mobility support, reduces communication overhead, resists disinformation, and preserves privacy. Despite these advantages, prior evaluations revealed limitations of our model in adapting to provider population changes and continuous performance fluctuations. This study proposes a novel algorithm, incorporating a self-classification mechanism for providers to detect performance drops potentially harmful for the service consumers. Simulation results demonstrate that the new algorithm outperforms its original version and FIRE, a well-known trust and reputation model, particularly in handling dynamic trustee behavior. While FIRE remains competitive under extreme environmental changes, the proposed algorithm demonstrates greater adaptability across various conditions. In contrast to existing trust modeling research, this study conducts a comprehensive evaluation of our model using widely recognized trust model criteria, assessing its resilience against common trust-related attacks while identifying strengths, weaknesses, and potential countermeasures. Finally, several key directions for future research are proposed. 

---
# Guiding VLM Agents with Process Rewards at Inference Time for GUI Navigation 

**Authors**: Zhiyuan Hu, Shiyun Xiong, Yifan Zhang, See-Kiong Ng, Anh Tuan Luu, Bo An, Shuicheng Yan, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2504.16073)  

**Abstract**: Recent advancements in visual language models (VLMs) have notably enhanced their capabilities in handling complex Graphical User Interface (GUI) interaction tasks. Despite these improvements, current frameworks often struggle to generate correct actions in challenging GUI environments. State-of-the-art commercial VLMs are black-boxes, and fine-tuning open-source VLMs for GUI tasks requires significant resources. Additionally, existing trajectory-level evaluation and refinement techniques frequently fall short due to delayed feedback and local optimization issues. To address these challenges, we propose an approach that guides VLM agents with process supervision by a reward model during GUI navigation and control at inference time. This guidance allows the VLM agent to optimize actions at each inference step, thereby improving performance in both static and dynamic environments. In particular, our method demonstrates significant performance gains in three GUI navigation tasks, achieving a 3.4% improvement in single step action accuracy for static environments, along with a around 33% increase in task success rate in one dynamic environment. With further integration of trajectory reflection and retry mechanisms, we also demonstrate even greater enhancement in task success. 

---
