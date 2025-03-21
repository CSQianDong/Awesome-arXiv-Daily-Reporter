# Advancing Mobile GUI Agents: A Verifier-Driven Approach to Practical Deployment 

**Authors**: Gaole Dai, Shiqi Jiang, Ting Cao, Yuanchun Li, Yuqing Yang, Rui Tan, Mo Li, Lili Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15937)  

**Abstract**: We propose V-Droid, a mobile GUI task automation agent. Unlike previous mobile agents that utilize Large Language Models (LLMs) as generators to directly generate actions at each step, V-Droid employs LLMs as verifiers to evaluate candidate actions before making final decisions. To realize this novel paradigm, we introduce a comprehensive framework for constructing verifier-driven mobile agents: the discretized action space construction coupled with the prefilling-only workflow to accelerate the verification process, the pair-wise progress preference training to significantly enhance the verifier's decision-making capabilities, and the scalable human-agent joint annotation scheme to efficiently collect the necessary data at scale. V-Droid sets a new state-of-the-art task success rate across several public mobile task automation benchmarks: 59.5% on AndroidWorld, 38.3% on AndroidLab, and 49% on MobileAgentBench, surpassing existing agents by 9.5%, 2.1%, and 9%, respectively. Furthermore, V-Droid achieves an impressively low latency of 0.7 seconds per step, making it the first mobile agent capable of delivering near-real-time, effective decision-making capabilities. 

---
# DeepPsy-Agent: A Stage-Aware and Deep-Thinking Emotional Support Agent System 

**Authors**: Kai Chen, Zebing Sun  

**Link**: [PDF](https://arxiv.org/pdf/2503.15876)  

**Abstract**: This paper introduces DeepPsy-Agent, an innovative psychological support system that combines the three-stage helping theory in psychology with deep learning techniques. The system consists of two core components: (1) a multi-stage response-capable dialogue model (\textit{deeppsy-chat}), which enhances reasoning capabilities through stage-awareness and deep-thinking analysis to generate high-quality responses; and (2) a real-time stage transition detection model that identifies contextual shifts to guide the dialogue towards more effective intervention stages. Based on 30,000 real psychological hotline conversations, we employ AI-simulated dialogues and expert re-annotation strategies to construct a high-quality multi-turn dialogue dataset. Experimental results demonstrate that DeepPsy-Agent outperforms general-purpose large language models (LLMs) in key metrics such as problem exposure completeness, cognitive restructuring success rate, and action adoption rate. Ablation studies further validate the effectiveness of stage-awareness and deep-thinking modules, showing that stage information contributes 42.3\% to performance, while the deep-thinking module increases root-cause identification by 58.3\% and reduces ineffective suggestions by 72.1\%. This system addresses critical challenges in AI-based psychological support through dynamic dialogue management and deep reasoning, advancing intelligent mental health services. 

---
# Unreal-MAP: Unreal-Engine-Based General Platform for Multi-Agent Reinforcement Learning 

**Authors**: Tianyi Hu, Qingxu Fu, Zhiqiang Pu, Yuan Wang, Tenghai Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15947)  

**Abstract**: In this paper, we propose Unreal Multi-Agent Playground (Unreal-MAP), an MARL general platform based on the Unreal-Engine (UE). Unreal-MAP allows users to freely create multi-agent tasks using the vast visual and physical resources available in the UE community, and deploy state-of-the-art (SOTA) MARL algorithms within them. Unreal-MAP is user-friendly in terms of deployment, modification, and visualization, and all its components are open-source. We also develop an experimental framework compatible with algorithms ranging from rule-based to learning-based provided by third-party frameworks. Lastly, we deploy several SOTA algorithms in example tasks developed via Unreal-MAP, and conduct corresponding experimental analyses. We believe Unreal-MAP can play an important role in the MARL field by closely integrating existing algorithms with user-customized tasks, thus advancing the field of MARL. 

---
# RoboFactory: Exploring Embodied Agent Collaboration with Compositional Constraints 

**Authors**: Yiran Qin, Li Kang, Xiufeng Song, Zhenfei Yin, Xiaohong Liu, Xihui Liu, Ruimao Zhang, Lei Bai  

**Link**: [PDF](https://arxiv.org/pdf/2503.16408)  

**Abstract**: Designing effective embodied multi-agent systems is critical for solving complex real-world tasks across domains. Due to the complexity of multi-agent embodied systems, existing methods fail to automatically generate safe and efficient training data for such systems. To this end, we propose the concept of compositional constraints for embodied multi-agent systems, addressing the challenges arising from collaboration among embodied agents. We design various interfaces tailored to different types of constraints, enabling seamless interaction with the physical world. Leveraging compositional constraints and specifically designed interfaces, we develop an automated data collection framework for embodied multi-agent systems and introduce the first benchmark for embodied multi-agent manipulation, RoboFactory. Based on RoboFactory benchmark, we adapt and evaluate the method of imitation learning and analyzed its performance in different difficulty agent tasks. Furthermore, we explore the architectures and training strategies for multi-agent imitation learning, aiming to build safe and efficient embodied multi-agent systems. 

---
# JARVIS-VLA: Post-Training Large-Scale Vision Language Models to Play Visual Games with Keyboards and Mouse 

**Authors**: Muyao Li, Zihao Wang, Kaichen He, Xiaojian Ma, Yitao Liang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16365)  

**Abstract**: Recently, action-based decision-making in open-world environments has gained significant attention. Visual Language Action (VLA) models, pretrained on large-scale web datasets, have shown promise in decision-making tasks. However, previous work has primarily focused on action post-training, often neglecting enhancements to the foundational model itself. In response, we introduce a novel approach, Act from Visual Language Post-Training, which refines Visual Language Models (VLMs) through visual and linguistic guidance in a self-supervised manner. This enhancement improves the models' capabilities in world knowledge, visual recognition, and spatial grounding in open-world environments. Following the above post-training paradigms, we obtain the first VLA models in Minecraft that can follow human instructions on over 1k different atomic tasks, including crafting, smelting, cooking, mining, and killing. Our experiments demonstrate that post-training on non-trajectory tasks leads to a significant 40% improvement over the best agent baseline on a diverse set of atomic tasks. Furthermore, we demonstrate that our approach surpasses traditional imitation learning-based policies in Minecraft, achieving state-of-the-art performance. We have open-sourced the code, models, and datasets to foster further research. The project page can be found in this https URL. 

---
# Do Visual Imaginations Improve Vision-and-Language Navigation Agents? 

**Authors**: Akhil Perincherry, Jacob Krantz, Stefan Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.16394)  

**Abstract**: Vision-and-Language Navigation (VLN) agents are tasked with navigating an unseen environment using natural language instructions. In this work, we study if visual representations of sub-goals implied by the instructions can serve as navigational cues and lead to increased navigation performance. To synthesize these visual representations or imaginations, we leverage a text-to-image diffusion model on landmark references contained in segmented instructions. These imaginations are provided to VLN agents as an added modality to act as landmark cues and an auxiliary loss is added to explicitly encourage relating these with their corresponding referring expressions. Our findings reveal an increase in success rate (SR) of around 1 point and up to 0.5 points in success scaled by inverse path length (SPL) across agents. These results suggest that the proposed approach reinforces visual understanding compared to relying on language instructions alone. Code and data for our work can be found at this https URL. 

---
# ECLAIR: Enhanced Clarification for Interactive Responses 

**Authors**: John Murzaku, Zifan Liu, Md Mehrab Tanjim, Vaishnavi Muppala, Xiang Chen, Yunyao Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.15739)  

**Abstract**: We present ECLAIR (Enhanced CLArification for Interactive Responses), a novel unified and end-to-end framework for interactive disambiguation in enterprise AI assistants. ECLAIR generates clarification questions for ambiguous user queries and resolves ambiguity based on the user's this http URL introduce a generalized architecture capable of integrating ambiguity information from multiple downstream agents, enhancing context-awareness in resolving ambiguities and allowing enterprise specific definition of agents. We further define agents within our system that provide domain-specific grounding information. We conduct experiments comparing ECLAIR to few-shot prompting techniques and demonstrate ECLAIR's superior performance in clarification question generation and ambiguity resolution. 

---
# Reinforcement Learning Environment with LLM-Controlled Adversary in D&D 5th Edition Combat 

**Authors**: Joseph Emmanuel DL Dayo, Michel Onasis S. Ogbinar, Prospero C. Naval Jr  

**Link**: [PDF](https://arxiv.org/pdf/2503.15726)  

**Abstract**: The objective of this study is to design and implement a reinforcement learning (RL) environment using D\&D 5E combat scenarios to challenge smaller RL agents through interaction with a robust adversarial agent controlled by advanced Large Language Models (LLMs) like GPT-4o and LLaMA 3 8B. This research employs Deep Q-Networks (DQN) for the smaller agents, creating a testbed for strategic AI development that also serves as an educational tool by simulating dynamic and unpredictable combat scenarios. We successfully integrated sophisticated language models into the RL framework, enhancing strategic decision-making processes. Our results indicate that while RL agents generally outperform LLM-controlled adversaries in standard metrics, the strategic depth provided by LLMs significantly enhances the overall AI capabilities in this complex, rule-based setting. The novelty of our approach and its implications for mastering intricate environments and developing adaptive strategies are discussed, alongside potential innovations in AI-driven interactive simulations. This paper aims to demonstrate how integrating LLMs can create more robust and adaptable AI systems, providing valuable insights for further research and educational applications. 

---
# Survey on Evaluation of LLM-based Agents 

**Authors**: Asaf Yehudai, Lilach Eden, Alan Li, Guy Uziel, Yilun Zhao, Roy Bar-Haim, Arman Cohan, Michal Shmueli-Scheuer  

**Link**: [PDF](https://arxiv.org/pdf/2503.16416)  

**Abstract**: The emergence of LLM-based agents represents a paradigm shift in AI, enabling autonomous systems to plan, reason, use tools, and maintain memory while interacting with dynamic environments. This paper provides the first comprehensive survey of evaluation methodologies for these increasingly capable agents. We systematically analyze evaluation benchmarks and frameworks across four critical dimensions: (1) fundamental agent capabilities, including planning, tool use, self-reflection, and memory; (2) application-specific benchmarks for web, software engineering, scientific, and conversational agents; (3) benchmarks for generalist agents; and (4) frameworks for evaluating agents. Our analysis reveals emerging trends, including a shift toward more realistic, challenging evaluations with continuously updated benchmarks. We also identify critical gaps that future research must address-particularly in assessing cost-efficiency, safety, and robustness, and in developing fine-grained, and scalable evaluation methods. This survey maps the rapidly evolving landscape of agent evaluation, reveals the emerging trends in the field, identifies current limitations, and proposes directions for future research. 

---
# AI Agents in Cryptoland: Practical Attacks and No Silver Bullet 

**Authors**: Atharv Singh Patlan, Peiyao Sheng, S. Ashwin Hebbar, Prateek Mittal, Pramod Viswanath  

**Link**: [PDF](https://arxiv.org/pdf/2503.16248)  

**Abstract**: The integration of AI agents with Web3 ecosystems harnesses their complementary potential for autonomy and openness, yet also introduces underexplored security risks, as these agents dynamically interact with financial protocols and immutable smart contracts. This paper investigates the vulnerabilities of AI agents within blockchain-based financial ecosystems when exposed to adversarial threats in real-world scenarios. We introduce the concept of context manipulation -- a comprehensive attack vector that exploits unprotected context surfaces, including input channels, memory modules, and external data feeds. Through empirical analysis of ElizaOS, a decentralized AI agent framework for automated Web3 operations, we demonstrate how adversaries can manipulate context by injecting malicious instructions into prompts or historical interaction records, leading to unintended asset transfers and protocol violations which could be financially devastating. Our findings indicate that prompt-based defenses are insufficient, as malicious inputs can corrupt an agent's stored context, creating cascading vulnerabilities across interactions and platforms. This research highlights the urgent need to develop AI agents that are both secure and fiduciarily responsible. 

---
# Flight Testing an Optionally Piloted Aircraft: a Case Study on Trust Dynamics in Human-Autonomy Teaming 

**Authors**: Jeremy C.-H. Wang, Ming Hou, David Dunwoody, Marko Ilievski, Justin Tomasi, Edward Chao, Carl Pigeon  

**Link**: [PDF](https://arxiv.org/pdf/2503.16227)  

**Abstract**: This paper examines how trust is formed, maintained, or diminished over time in the context of human-autonomy teaming with an optionally piloted aircraft. Whereas traditional factor-based trust models offer a static representation of human confidence in technology, here we discuss how variations in the underlying factors lead to variations in trust, trust thresholds, and human behaviours. Over 200 hours of flight test data collected over a multi-year test campaign from 2021 to 2023 were reviewed. The dispositional-situational-learned, process-performance-purpose, and IMPACTS homeostasis trust models are applied to illuminate trust trends during nominal autonomous flight operations. The results offer promising directions for future studies on trust dynamics and design-for-trust in human-autonomy teaming. 

---
# The Lighthouse of Language: Enhancing LLM Agents via Critique-Guided Improvement 

**Authors**: Ruihan Yang, Fanghua Ye, Jian Li, Siyu Yuan, Yikai Zhang, Zhaopeng Tu, Xiaolong Li, Deqing Yang  

**Link**: [PDF](https://arxiv.org/pdf/2503.16024)  

**Abstract**: Large language models (LLMs) have recently transformed from text-based assistants to autonomous agents capable of planning, reasoning, and iteratively improving their actions. While numerical reward signals and verifiers can effectively rank candidate actions, they often provide limited contextual guidance. In contrast, natural language feedback better aligns with the generative capabilities of LLMs, providing richer and more actionable suggestions. However, parsing and implementing this feedback effectively can be challenging for LLM-based agents. In this work, we introduce Critique-Guided Improvement (CGI), a novel two-player framework, comprising an actor model that explores an environment and a critic model that generates detailed nature language feedback. By training the critic to produce fine-grained assessments and actionable revisions, and the actor to utilize these critiques, our approach promotes more robust exploration of alternative strategies while avoiding local optima. Experiments in three interactive environments show that CGI outperforms existing baselines by a substantial margin. Notably, even a small critic model surpasses GPT-4 in feedback quality. The resulting actor achieves state-of-the-art performance, demonstrating the power of explicit iterative guidance to enhance decision-making in LLM-based agents. 

---
# Active management of battery degradation in wireless sensor network using deep reinforcement learning for group battery replacement 

**Authors**: Jong-Hyun Jeonga, Hongki Jo, Qiang Zhou, Tahsin Afroz Hoque Nishat, Lang Wu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15865)  

**Abstract**: Wireless sensor networks (WSNs) have become a promising solution for structural health monitoring (SHM), especially in hard-to-reach or remote locations. Battery-powered WSNs offer various advantages over wired systems, however limited battery life has always been one of the biggest obstacles in practical use of the WSNs, regardless of energy harvesting methods. While various methods have been studied for battery health management, existing methods exclusively aim to extend lifetime of individual batteries, lacking a system level view. A consequence of applying such methods is that batteries in a WSN tend to fail at different times, posing significant difficulty on planning and scheduling of battery replacement trip. This study investigate a deep reinforcement learning (DRL) method for active battery degradation management by optimizing duty cycle of WSNs at the system level. This active management strategy effectively reduces earlier failure of battery individuals which enable group replacement without sacrificing WSN performances. A simulated environment based on a real-world WSN setup was developed to train a DRL agent and learn optimal duty cycle strategies. The performance of the strategy was validated in a long-term setup with various network sizes, demonstrating its efficiency and scalability. 

---
# Reward Training Wheels: Adaptive Auxiliary Rewards for Robotics Reinforcement Learning 

**Authors**: Linji Wang, Tong Xu, Yuanjie Lu, Xuesu Xiao  

**Link**: [PDF](https://arxiv.org/pdf/2503.15724)  

**Abstract**: Robotics Reinforcement Learning (RL) often relies on carefully engineered auxiliary rewards to supplement sparse primary learning objectives to compensate for the lack of large-scale, real-world, trial-and-error data. While these auxiliary rewards accelerate learning, they require significant engineering effort, may introduce human biases, and cannot adapt to the robot's evolving capabilities during training. In this paper, we introduce Reward Training Wheels (RTW), a teacher-student framework that automates auxiliary reward adaptation for robotics RL. To be specific, the RTW teacher dynamically adjusts auxiliary reward weights based on the student's evolving capabilities to determine which auxiliary reward aspects require more or less emphasis to improve the primary objective. We demonstrate RTW on two challenging robot tasks: navigation in highly constrained spaces and off-road vehicle mobility on vertically challenging terrain. In simulation, RTW outperforms expert-designed rewards by 2.35% in navigation success rate and improves off-road mobility performance by 122.62%, while achieving 35% and 3X faster training efficiency, respectively. Physical robot experiments further validate RTW's effectiveness, achieving a perfect success rate (5/5 trials vs. 2/5 for expert-designed rewards) and improving vehicle stability with up to 47.4% reduction in orientation angles. 

---
# Safety Aware Task Planning via Large Language Models in Robotics 

**Authors**: Azal Ahmad Khan, Michael Andrev, Muhammad Ali Murtaza, Sergio Aguilera, Rui Zhang, Jie Ding, Seth Hutchinson, Ali Anwar  

**Link**: [PDF](https://arxiv.org/pdf/2503.15707)  

**Abstract**: The integration of large language models (LLMs) into robotic task planning has unlocked better reasoning capabilities for complex, long-horizon workflows. However, ensuring safety in LLM-driven plans remains a critical challenge, as these models often prioritize task completion over risk mitigation. This paper introduces SAFER (Safety-Aware Framework for Execution in Robotics), a multi-LLM framework designed to embed safety awareness into robotic task planning. SAFER employs a Safety Agent that operates alongside the primary task planner, providing safety feedback. Additionally, we introduce LLM-as-a-Judge, a novel metric leveraging LLMs as evaluators to quantify safety violations within generated task plans. Our framework integrates safety feedback at multiple stages of execution, enabling real-time risk assessment, proactive error correction, and transparent safety evaluation. We also integrate a control framework using Control Barrier Functions (CBFs) to ensure safety guarantees within SAFER's task planning. We evaluated SAFER against state-of-the-art LLM planners on complex long-horizon tasks involving heterogeneous robotic agents, demonstrating its effectiveness in reducing safety violations while maintaining task efficiency. We also verify the task planner and safety planner through actual hardware experiments involving multiple robots and a human. 

---
# Towards Agentic AI Networking in 6G: A Generative Foundation Model-as-Agent Approach 

**Authors**: Yong Xiao, Guangming Shi, Ping Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2503.15764)  

**Abstract**: The promising potential of AI and network convergence in improving networking performance and enabling new service capabilities has recently attracted significant interest. Existing network AI solutions, while powerful, are mainly built based on the close-loop and passive learning framework, resulting in major limitations in autonomous solution finding and dynamic environmental adaptation. Agentic AI has recently been introduced as a promising solution to address the above limitations and pave the way for true generally intelligent and beneficial AI systems. The key idea is to create a networking ecosystem to support a diverse range of autonomous and embodied AI agents in fulfilling their goals. In this paper, we focus on the novel challenges and requirements of agentic AI networking. We propose AgentNet, a novel framework for supporting interaction, collaborative learning, and knowledge transfer among AI agents. We introduce a general architectural framework of AgentNet and then propose a generative foundation model (GFM)-based implementation in which multiple GFM-as-agents have been created as an interactive knowledge-base to bootstrap the development of embodied AI agents according to different task requirements and environmental features. We consider two application scenarios, digital-twin-based industrial automation and metaverse-based infotainment system, to describe how to apply AgentNet for supporting efficient task-driven collaboration and interaction among AI agents. 

---
# AutoRedTeamer: Autonomous Red Teaming with Lifelong Attack Integration 

**Authors**: Andy Zhou, Kevin Wu, Francesco Pinto, Zhaorun Chen, Yi Zeng, Yu Yang, Shuang Yang, Sanmi Koyejo, James Zou, Bo Li  

**Link**: [PDF](https://arxiv.org/pdf/2503.15754)  

**Abstract**: As large language models (LLMs) become increasingly capable, security and safety evaluation are crucial. While current red teaming approaches have made strides in assessing LLM vulnerabilities, they often rely heavily on human input and lack comprehensive coverage of emerging attack vectors. This paper introduces AutoRedTeamer, a novel framework for fully automated, end-to-end red teaming against LLMs. AutoRedTeamer combines a multi-agent architecture with a memory-guided attack selection mechanism to enable continuous discovery and integration of new attack vectors. The dual-agent framework consists of a red teaming agent that can operate from high-level risk categories alone to generate and execute test cases and a strategy proposer agent that autonomously discovers and implements new attacks by analyzing recent research. This modular design allows AutoRedTeamer to adapt to emerging threats while maintaining strong performance on existing attack vectors. We demonstrate AutoRedTeamer's effectiveness across diverse evaluation settings, achieving 20% higher attack success rates on HarmBench against Llama-3.1-70B while reducing computational costs by 46% compared to existing approaches. AutoRedTeamer also matches the diversity of human-curated benchmarks in generating test cases, providing a comprehensive, scalable, and continuously evolving framework for evaluating the security of AI systems. 

---
# Enforcing Cybersecurity Constraints for LLM-driven Robot Agents for Online Transactions 

**Authors**: Shraddha Pradipbhai Shah, Aditya Vilas Deshpande  

**Link**: [PDF](https://arxiv.org/pdf/2503.15546)  

**Abstract**: The integration of Large Language Models (LLMs) into autonomous robotic agents for conducting online transactions poses significant cybersecurity challenges. This study aims to enforce robust cybersecurity constraints to mitigate the risks associated with data breaches, transaction fraud, and system manipulation. The background focuses on the rise of LLM-driven robotic systems in e-commerce, finance, and service industries, alongside the vulnerabilities they introduce. A novel security architecture combining blockchain technology with multi-factor authentication (MFA) and real-time anomaly detection was implemented to safeguard transactions. Key performance metrics such as transaction integrity, response time, and breach detection accuracy were evaluated, showing improved security and system performance. The results highlight that the proposed architecture reduced fraudulent transactions by 90%, improved breach detection accuracy to 98%, and ensured secure transaction validation within a latency of 0.05 seconds. These findings emphasize the importance of cybersecurity in the deployment of LLM-driven robotic systems and suggest a framework adaptable to various online platforms. 

---
# Prompt Flow Integrity to Prevent Privilege Escalation in LLM Agents 

**Authors**: Juhee Kim, Woohyuk Choi, Byoungyoung Lee  

**Link**: [PDF](https://arxiv.org/pdf/2503.15547)  

**Abstract**: Large Language Models (LLMs) are combined with plugins to create powerful LLM agents that provide a wide range of services. Unlike traditional software, LLM agent's behavior is determined at runtime by natural language prompts from either user or plugin's data. This flexibility enables a new computing paradigm with unlimited capabilities and programmability, but also introduces new security risks, vulnerable to privilege escalation attacks. Moreover, user prompt is prone to be interpreted in an insecure way by LLM agents, creating non-deterministic behaviors that can be exploited by attackers. To address these security risks, we propose Prompt Flow Integrity (PFI), a system security-oriented solution to prevent privilege escalation in LLM agents. Analyzing the architectural characteristics of LLM agents, PFI features three mitigation techniques -- i.e., untrusted data identification, enforcing least privilege on LLM agents, and validating unsafe data flows. Our evaluation result shows that PFI effectively mitigates privilege escalation attacks while successfully preserving the utility of LLM agents. 

---
# PEnGUiN: Partially Equivariant Graph NeUral Networks for Sample Efficient MARL 

**Authors**: Joshua McClellan, Greyson Brothers, Furong Huang, Pratap Tokekar  

**Link**: [PDF](https://arxiv.org/pdf/2503.15615)  

**Abstract**: Equivariant Graph Neural Networks (EGNNs) have emerged as a promising approach in Multi-Agent Reinforcement Learning (MARL), leveraging symmetry guarantees to greatly improve sample efficiency and generalization. However, real-world environments often exhibit inherent asymmetries arising from factors such as external forces, measurement inaccuracies, or intrinsic system biases. This paper introduces \textit{Partially Equivariant Graph NeUral Networks (PEnGUiN)}, a novel architecture specifically designed to address these challenges. We formally identify and categorize various types of partial equivariance relevant to MARL, including subgroup equivariance, feature-wise equivariance, regional equivariance, and approximate equivariance. We theoretically demonstrate that PEnGUiN is capable of learning both fully equivariant (EGNN) and non-equivariant (GNN) representations within a unified framework. Through extensive experiments on a range of MARL problems incorporating various asymmetries, we empirically validate the efficacy of PEnGUiN. Our results consistently demonstrate that PEnGUiN outperforms both EGNNs and standard GNNs in asymmetric environments, highlighting their potential to improve the robustness and applicability of graph-based MARL algorithms in real-world scenarios. 

---
# KHAIT: K-9 Handler Artificial Intelligence Teaming for Collaborative Sensemaking 

**Authors**: Matthew Wilchek, Linhan Wang, Sally Dickinson, Erica Feuerbacher, Kurt Luther, Feras A. Batarseh  

**Link**: [PDF](https://arxiv.org/pdf/2503.15524)  

**Abstract**: In urban search and rescue (USAR) operations, communication between handlers and specially trained canines is crucial but often complicated by challenging environments and the specific behaviors canines are trained to exhibit when detecting a person. Since a USAR canine often works out of sight of the handler, the handler lacks awareness of the canine's location and situation, known as the 'sensemaking gap.' In this paper, we propose KHAIT, a novel approach to close the sensemaking gap and enhance USAR effectiveness by integrating object detection-based Artificial Intelligence (AI) and Augmented Reality (AR). Equipped with AI-powered cameras, edge computing, and AR headsets, KHAIT enables precise and rapid object detection from a canine's perspective, improving survivor localization. We evaluate this approach in a real-world USAR environment, demonstrating an average survival allocation time decrease of 22%, enhancing the speed and accuracy of operations. 

---
# In Pursuit of Predictive Models of Human Preferences Toward AI Teammates 

**Authors**: Ho Chit Siu, Jaime D. Peña, Yutai Zhou, Ross E. Allen  

**Link**: [PDF](https://arxiv.org/pdf/2503.15516)  

**Abstract**: We seek measurable properties of AI agents that make them better or worse teammates from the subjective perspective of human collaborators. Our experiments use the cooperative card game Hanabi -- a common benchmark for AI-teaming research. We first evaluate AI agents on a set of objective metrics based on task performance, information theory, and game theory, which are measurable without human interaction. Next, we evaluate subjective human preferences toward AI teammates in a large-scale (N=241) human-AI teaming experiment. Finally, we correlate the AI-only objective metrics with the human subjective preferences. Our results refute common assumptions from prior literature on reinforcement learning, revealing new correlations between AI behaviors and human preferences. We find that the final game score a human-AI team achieves is less predictive of human preferences than esoteric measures of AI action diversity, strategic dominance, and ability to team with other AI. In the future, these correlations may help shape reward functions for training human-collaborative AI. 

---
# The Impact of Big Five Personality Traits on AI Agent Decision-Making in Public Spaces: A Social Simulation Study 

**Authors**: Mingjun Ren, Wentao Xu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15497)  

**Abstract**: This study investigates how the Big Five personality traits influence decision-making processes in AI agents within public spaces. Using AgentVerse framework and GPT-3.5-turbo, we simulated interactions among 10 AI agents, each embodying different dimensions of the Big Five personality traits, in a classroom environment responding to misinformation. The experiment assessed both public expressions ([Speak]) and private thoughts ([Think]) of agents, revealing significant correlations between personality traits and decision-making patterns. Results demonstrate that Openness to Experience had the strongest impact on information acceptance, with curious agents showing high acceptance rates and cautious agents displaying strong skepticism. Extraversion and Conscientiousness also showed notable influence on decision-making, while Neuroticism and Agreeableness exhibited more balanced responses. Additionally, we observed significant discrepancies between public expressions and private thoughts, particularly in agents with friendly and extroverted personalities, suggesting that social context influences decision-making behavior. Our findings contribute to understanding how personality traits shape AI agent behavior in social settings and have implications for developing more nuanced and context-aware AI systems. 

---
# Towards Computer-Using Personal Agents 

**Authors**: Piero A. Bonatti, John Domingue, Anna Lisa Gentile, Andreas Harth, Olaf Hartig, Aidan Hogan, Katja Hose, Ernesto Jimenez-Ruiz, Deborah L. McGuinness, Chang Sun, Ruben Verborgh, Jesse Wright  

**Link**: [PDF](https://arxiv.org/pdf/2503.15515)  

**Abstract**: Computer-Using Agents (CUA) enable users to automate increasingly-complex tasks using graphical interfaces such as browsers. As many potential tasks require personal data, we propose Computer-Using Personal Agents (CUPAs) that have access to an external repository of the user's personal data. Compared with CUAs, CUPAs offer users better control of their personal data, the potential to automate more tasks involving personal data, better interoperability with external sources of data, and better capabilities to coordinate with other CUPAs in order to solve collaborative tasks involving the personal data of multiple users. 

---
# Reinforcement Learning for Reasoning in Small LLMs: What Works and What Doesn't 

**Authors**: Quy-Anh Dang, Chris Ngo  

**Link**: [PDF](https://arxiv.org/pdf/2503.16219)  

**Abstract**: Enhancing the reasoning capabilities of large language models (LLMs) typically relies on massive computational resources and extensive datasets, limiting accessibility for resource-constrained settings. Our study investigates the potential of reinforcement learning (RL) to improve reasoning in small LLMs, focusing on a 1.5-billion-parameter model, DeepSeek-R1-Distill-Qwen-1.5B, under strict constraints: training on 4 NVIDIA A40 GPUs (48 GB VRAM each) within 24 hours. Adapting the Group Relative Policy Optimization (GRPO) algorithm and curating a compact, high-quality mathematical reasoning dataset, we conducted three experiments to explore model behavior and performance. Our results demonstrate rapid reasoning gains - e.g., AMC23 accuracy rising from 63% to 80% and AIME24 reaching 46.7%, surpassing o1-preview - using only 7,000 samples and a $42 training cost, compared to thousands of dollars for baseline models. However, challenges such as optimization instability and length constraints emerged with prolonged training. These findings highlight the efficacy of RL-based fine-tuning for small LLMs, offering a cost-effective alternative to large-scale approaches. We release our code and datasets as open-source resources, providing insights into trade-offs and laying a foundation for scalable, reasoning-capable LLMs in resource-limited environments. All are available at this https URL. 

---
# Personalized Attacks of Social Engineering in Multi-turn Conversations -- LLM Agents for Simulation and Detection 

**Authors**: Tharindu Kumarage, Cameron Johnson, Jadie Adams, Lin Ai, Matthias Kirchner, Anthony Hoogs, Joshua Garland, Julia Hirschberg, Arslan Basharat, Huan Liu  

**Link**: [PDF](https://arxiv.org/pdf/2503.15552)  

**Abstract**: The rapid advancement of conversational agents, particularly chatbots powered by Large Language Models (LLMs), poses a significant risk of social engineering (SE) attacks on social media platforms. SE detection in multi-turn, chat-based interactions is considerably more complex than single-instance detection due to the dynamic nature of these conversations. A critical factor in mitigating this threat is understanding the mechanisms through which SE attacks operate, specifically how attackers exploit vulnerabilities and how victims' personality traits contribute to their susceptibility. In this work, we propose an LLM-agentic framework, SE-VSim, to simulate SE attack mechanisms by generating multi-turn conversations. We model victim agents with varying personality traits to assess how psychological profiles influence susceptibility to manipulation. Using a dataset of over 1000 simulated conversations, we examine attack scenarios in which adversaries, posing as recruiters, funding agencies, and journalists, attempt to extract sensitive information. Based on this analysis, we present a proof of concept, SE-OmniGuard, to offer personalized protection to users by leveraging prior knowledge of the victims personality, evaluating attack strategies, and monitoring information exchanges in conversations to identify potential SE attempts. 

---
# Agreeing to Interact in Human-Robot Interaction using Large Language Models and Vision Language Models 

**Authors**: Kazuhiro Sasabuchi, Naoki Wake, Atsushi Kanehira, Jun Takamatsu, Katsushi Ikeuchi  

**Link**: [PDF](https://arxiv.org/pdf/2503.15491)  

**Abstract**: In human-robot interaction (HRI), the beginning of an interaction is often complex. Whether the robot should communicate with the human is dependent on several situational factors (e.g., the current human's activity, urgency of the interaction, etc.). We test whether large language models (LLM) and vision language models (VLM) can provide solutions to this problem. We compare four different system-design patterns using LLMs and VLMs, and test on a test set containing 84 human-robot situations. The test set mixes several publicly available datasets and also includes situations where the appropriate action to take is open-ended. Our results using the GPT-4o and Phi-3 Vision model indicate that LLMs and VLMs are capable of handling interaction beginnings when the desired actions are clear, however, challenge remains in the open-ended situations where the model must balance between the human and robot situation. 

---
