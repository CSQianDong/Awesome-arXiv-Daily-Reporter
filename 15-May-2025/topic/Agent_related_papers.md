# Scent of Knowledge: Optimizing Search-Enhanced Reasoning with Information Foraging 

**Authors**: Hongjin Qian, Zheng Liu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09316)  

**Abstract**: Augmenting large language models (LLMs) with external retrieval has become a standard method to address their inherent knowledge cutoff limitations. However, traditional retrieval-augmented generation methods employ static, pre-inference retrieval strategies, making them inadequate for complex tasks involving ambiguous, multi-step, or evolving information needs. Recent advances in test-time scaling techniques have demonstrated significant potential in enabling LLMs to dynamically interact with external tools, motivating the shift toward adaptive inference-time retrieval. Inspired by Information Foraging Theory (IFT), we propose InForage, a reinforcement learning framework that formalizes retrieval-augmented reasoning as a dynamic information-seeking process. Unlike existing approaches, InForage explicitly rewards intermediate retrieval quality, encouraging LLMs to iteratively gather and integrate information through adaptive search behaviors. To facilitate training, we construct a human-guided dataset capturing iterative search and reasoning trajectories for complex, real-world web tasks. Extensive evaluations across general question answering, multi-hop reasoning tasks, and a newly developed real-time web QA dataset demonstrate InForage's superior performance over baseline methods. These results highlight InForage's effectiveness in building robust, adaptive, and efficient reasoning agents. 

---
# Qwen3 Technical Report 

**Authors**: An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Le Yu, Lianghao Deng, Mei Li, Mingfeng Xue, Mingze Li, Pei Zhang, Peng Wang, Qin Zhu, Rui Men, Ruize Gao, Shixuan Liu, Shuang Luo, Tianhao Li, Tianyi Tang, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yinger Zhang, Yu Wan, Yuqiong Liu, Zekun Wang, Zeyu Cui, Zhenru Zhang, Zhipeng Zhou, Zihan Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09388)  

**Abstract**: In this work, we present Qwen3, the latest version of the Qwen model family. Qwen3 comprises a series of large language models (LLMs) designed to advance performance, efficiency, and multilingual capabilities. The Qwen3 series includes models of both dense and Mixture-of-Expert (MoE) architectures, with parameter scales ranging from 0.6 to 235 billion. A key innovation in Qwen3 is the integration of thinking mode (for complex, multi-step reasoning) and non-thinking mode (for rapid, context-driven responses) into a unified framework. This eliminates the need to switch between different models--such as chat-optimized models (e.g., GPT-4o) and dedicated reasoning models (e.g., QwQ-32B)--and enables dynamic mode switching based on user queries or chat templates. Meanwhile, Qwen3 introduces a thinking budget mechanism, allowing users to allocate computational resources adaptively during inference, thereby balancing latency and performance based on task complexity. Moreover, by leveraging the knowledge from the flagship models, we significantly reduce the computational resources required to build smaller-scale models, while ensuring their highly competitive performance. Empirical evaluations demonstrate that Qwen3 achieves state-of-the-art results across diverse benchmarks, including tasks in code generation, mathematical reasoning, agent tasks, etc., competitive against larger MoE models and proprietary models. Compared to its predecessor Qwen2.5, Qwen3 expands multilingual support from 29 to 119 languages and dialects, enhancing global accessibility through improved cross-lingual understanding and generation capabilities. To facilitate reproducibility and community-driven research and development, all Qwen3 models are publicly accessible under Apache 2.0. 

---
# LibVulnWatch: A Deep Assessment Agent System and Leaderboard for Uncovering Hidden Vulnerabilities in Open-Source AI Libraries 

**Authors**: Zekun Wu, Seonglae Cho, Umar Mohammed, Cristian Munoz, Kleyton Costa, Xin Guan, Theo King, Ze Wang, Emre Kazim, Adriano Koshiyama  

**Link**: [PDF](https://arxiv.org/pdf/2505.08842)  

**Abstract**: Open-source AI libraries are foundational to modern AI systems but pose significant, underexamined risks across security, licensing, maintenance, supply chain integrity, and regulatory compliance. We present LibVulnWatch, a graph-based agentic assessment framework that performs deep, source-grounded evaluations of these libraries. Built on LangGraph, the system coordinates a directed acyclic graph of specialized agents to extract, verify, and quantify risk using evidence from trusted sources such as repositories, documentation, and vulnerability databases. LibVulnWatch generates reproducible, governance-aligned scores across five critical domains, publishing them to a public leaderboard for longitudinal ecosystem monitoring. Applied to 20 widely used libraries, including ML frameworks, LLM inference engines, and agent orchestration tools, our system covers up to 88% of OpenSSF Scorecard checks while uncovering up to 19 additional risks per library. These include critical Remote Code Execution (RCE) vulnerabilities, absent Software Bills of Materials (SBOMs), licensing constraints, undocumented telemetry, and widespread gaps in regulatory documentation and auditability. By translating high-level governance principles into practical, verifiable metrics, LibVulnWatch advances technical AI governance with a scalable, transparent mechanism for continuous supply chain risk assessment and informed library selection. 

---
# Language Agents Mirror Human Causal Reasoning Biases. How Can We Help Them Think Like Scientists? 

**Authors**: Anthony GX-Chen, Dongyan Lin, Mandana Samiei, Doina Precup, Blake A. Richards, Rob Fergus, Kenneth Marino  

**Link**: [PDF](https://arxiv.org/pdf/2505.09614)  

**Abstract**: Language model (LM) agents are increasingly used as autonomous decision-makers who need to actively gather information to guide their decisions. A crucial cognitive skill for such agents is the efficient exploration and understanding of the causal structure of the world -- key to robust, scientifically grounded reasoning. Yet, it remains unclear whether LMs possess this capability or exhibit systematic biases leading to erroneous conclusions. In this work, we examine LMs' ability to explore and infer causal relationships, using the well-established "Blicket Test" paradigm from developmental psychology. We find that LMs reliably infer the common, intuitive disjunctive causal relationships but systematically struggle with the unusual, yet equally (or sometimes even more) evidenced conjunctive ones. This "disjunctive bias" persists across model families, sizes, and prompting strategies, and performance further declines as task complexity increases. Interestingly, an analogous bias appears in human adults, suggesting that LMs may have inherited deep-seated reasoning heuristics from their training data. To this end, we quantify similarities between LMs and humans, finding that LMs exhibit adult-like inference profiles (but not children-like). Finally, we propose a test-time sampling method which explicitly samples and eliminates hypotheses about causal relationships from the LM. This scalable approach significantly reduces the disjunctive bias and moves LMs closer to the goal of scientific, causally rigorous reasoning. 

---
# Automated Meta Prompt Engineering for Alignment with the Theory of Mind 

**Authors**: Aaron Baughman, Rahul Agarwal, Eduardo Morales, Gozde Akay  

**Link**: [PDF](https://arxiv.org/pdf/2505.09024)  

**Abstract**: We introduce a method of meta-prompting that jointly produces fluent text for complex tasks while optimizing the similarity of neural states between a human's mental expectation and a Large Language Model's (LLM) neural processing. A technique of agentic reinforcement learning is applied, in which an LLM as a Judge (LLMaaJ) teaches another LLM, through in-context learning, how to produce content by interpreting the intended and unintended generated text traits. To measure human mental beliefs around content production, users modify long form AI-generated text articles before publication at the US Open 2024 tennis Grand Slam. Now, an LLMaaJ can solve the Theory of Mind (ToM) alignment problem by anticipating and including human edits within the creation of text from an LLM. Throughout experimentation and by interpreting the results of a live production system, the expectations of human content reviewers had 100% of alignment with AI 53.8% of the time with an average iteration count of 4.38. The geometric interpretation of content traits such as factualness, novelty, repetitiveness, and relevancy over a Hilbert vector space combines spatial volume (all trait importance) with vertices alignment (individual trait relevance) enabled the LLMaaJ to optimize on Human ToM. This resulted in an increase in content quality by extending the coverage of tennis action. Our work that was deployed at the US Open 2024 has been used across other live events within sports and entertainment. 

---
# Reproducibility Study of "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society of LLM Agents" 

**Authors**: Pedro M. P. Curvo, Mara Dragomir, Salvador Torpes, Mohammadmahdi Rahimi  

**Link**: [PDF](https://arxiv.org/pdf/2505.09289)  

**Abstract**: This study evaluates and extends the findings made by Piatti et al., who introduced GovSim, a simulation framework designed to assess the cooperative decision-making capabilities of large language models (LLMs) in resource-sharing scenarios. By replicating key experiments, we validate claims regarding the performance of large models, such as GPT-4-turbo, compared to smaller models. The impact of the universalization principle is also examined, with results showing that large models can achieve sustainable cooperation, with or without the principle, while smaller models fail without it. In addition, we provide multiple extensions to explore the applicability of the framework to new settings. We evaluate additional models, such as DeepSeek-V3 and GPT-4o-mini, to test whether cooperative behavior generalizes across different architectures and model sizes. Furthermore, we introduce new settings: we create a heterogeneous multi-agent environment, study a scenario using Japanese instructions, and explore an "inverse environment" where agents must cooperate to mitigate harmful resource distributions. Our results confirm that the benchmark can be applied to new models, scenarios, and languages, offering valuable insights into the adaptability of LLMs in complex cooperative tasks. Moreover, the experiment involving heterogeneous multi-agent systems demonstrates that high-performing models can influence lower-performing ones to adopt similar behaviors. This finding has significant implications for other agent-based applications, potentially enabling more efficient use of computational resources and contributing to the development of more effective cooperative AI systems. 

---
# The Influence of Human-inspired Agentic Sophistication in LLM-driven Strategic Reasoners 

**Authors**: Vince Trencsenyi, Agnieszka Mensfelt, Kostas Stathis  

**Link**: [PDF](https://arxiv.org/pdf/2505.09396)  

**Abstract**: The rapid rise of large language models (LLMs) has shifted artificial intelligence (AI) research toward agentic systems, motivating the use of weaker and more flexible notions of agency. However, this shift raises key questions about the extent to which LLM-based agents replicate human strategic reasoning, particularly in game-theoretic settings. In this context, we examine the role of agentic sophistication in shaping artificial reasoners' performance by evaluating three agent designs: a simple game-theoretic model, an unstructured LLM-as-agent model, and an LLM integrated into a traditional agentic framework. Using guessing games as a testbed, we benchmarked these agents against human participants across general reasoning patterns and individual role-based objectives. Furthermore, we introduced obfuscated game scenarios to assess agents' ability to generalise beyond training distributions. Our analysis, covering over 2000 reasoning samples across 25 agent configurations, shows that human-inspired cognitive structures can enhance LLM agents' alignment with human strategic behaviour. Still, the relationship between agentic design complexity and human-likeness is non-linear, highlighting a critical dependence on underlying LLM capabilities and suggesting limits to simple architectural augmentation. 

---
# Generalization in Monitored Markov Decision Processes (Mon-MDPs) 

**Authors**: Montaser Mohammedalamen, Michael Bowling  

**Link**: [PDF](https://arxiv.org/pdf/2505.08988)  

**Abstract**: Reinforcement learning (RL) typically models the interaction between the agent and environment as a Markov decision process (MDP), where the rewards that guide the agent's behavior are always observable. However, in many real-world scenarios, rewards are not always observable, which can be modeled as a monitored Markov decision process (Mon-MDP). Prior work on Mon-MDPs have been limited to simple, tabular cases, restricting their applicability to real-world problems. This work explores Mon-MDPs using function approximation (FA) and investigates the challenges involved. We show that combining function approximation with a learned reward model enables agents to generalize from monitored states with observable rewards, to unmonitored environment states with unobservable rewards. Therefore, we demonstrate that such generalization with a reward model achieves near-optimal policies in environments formally defined as unsolvable. However, we identify a critical limitation of such function approximation, where agents incorrectly extrapolate rewards due to overgeneralization, resulting in undesirable behaviors. To mitigate overgeneralization, we propose a cautious police optimization method leveraging reward uncertainty. This work serves as a step towards bridging this gap between Mon-MDP theory and real-world applications. 

---
# Counterfactual Strategies for Markov Decision Processes 

**Authors**: Paul Kobialka, Lina Gerlach, Francesco Leofante, Erika Ábrahám, Silvia Lizeth Tapia Tarifa, Einar Broch Johnsen  

**Link**: [PDF](https://arxiv.org/pdf/2505.09412)  

**Abstract**: Counterfactuals are widely used in AI to explain how minimal changes to a model's input can lead to a different output. However, established methods for computing counterfactuals typically focus on one-step decision-making, and are not directly applicable to sequential decision-making tasks. This paper fills this gap by introducing counterfactual strategies for Markov Decision Processes (MDPs). During MDP execution, a strategy decides which of the enabled actions (with known probabilistic effects) to execute next. Given an initial strategy that reaches an undesired outcome with a probability above some limit, we identify minimal changes to the initial strategy to reduce that probability below the limit. We encode such counterfactual strategies as solutions to non-linear optimization problems, and further extend our encoding to synthesize diverse counterfactual strategies. We evaluate our approach on four real-world datasets and demonstrate its practical viability in sophisticated sequential decision-making tasks. 

---
# Deep Reinforcement Learning for Power Grid Multi-Stage Cascading Failure Mitigation 

**Authors**: Bo Meng, Chenghao Xu, Yongli Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2505.09012)  

**Abstract**: Cascading failures in power grids can lead to grid collapse, causing severe disruptions to social operations and economic activities. In certain cases, multi-stage cascading failures can occur. However, existing cascading-failure-mitigation strategies are usually single-stage-based, overlooking the complexity of the multi-stage scenario. This paper treats the multi-stage cascading failure problem as a reinforcement learning task and develops a simulation environment. The reinforcement learning agent is then trained via the deterministic policy gradient algorithm to achieve continuous actions. Finally, the effectiveness of the proposed approach is validated on the IEEE 14-bus and IEEE 118-bus systems. 

---
# Monte Carlo Beam Search for Actor-Critic Reinforcement Learning in Continuous Control 

**Authors**: Hazim Alzorgan, Abolfazl Razi  

**Link**: [PDF](https://arxiv.org/pdf/2505.09029)  

**Abstract**: Actor-critic methods, like Twin Delayed Deep Deterministic Policy Gradient (TD3), depend on basic noise-based exploration, which can result in less than optimal policy convergence. In this study, we introduce Monte Carlo Beam Search (MCBS), a new hybrid method that combines beam search and Monte Carlo rollouts with TD3 to improve exploration and action selection. MCBS produces several candidate actions around the policy's output and assesses them through short-horizon rollouts, enabling the agent to make better-informed choices. We test MCBS across various continuous-control benchmarks, including HalfCheetah-v4, Walker2d-v5, and Swimmer-v5, showing enhanced sample efficiency and performance compared to standard TD3 and other baseline methods like SAC, PPO, and A2C. Our findings emphasize MCBS's capability to enhance policy learning through structured look-ahead search while ensuring computational efficiency. Additionally, we offer a detailed analysis of crucial hyperparameters, such as beam width and rollout depth, and explore adaptive strategies to optimize MCBS for complex control tasks. Our method shows a higher convergence rate across different environments compared to TD3, SAC, PPO, and A2C. For instance, we achieved 90% of the maximum achievable reward within around 200 thousand timesteps compared to 400 thousand timesteps for the second-best method. 

---
# Enhancing Aerial Combat Tactics through Hierarchical Multi-Agent Reinforcement Learning 

**Authors**: Ardian Selmonaj, Oleg Szehr, Giacomo Del Rio, Alessandro Antonucci, Adrian Schneider, Michael Rüegsegger  

**Link**: [PDF](https://arxiv.org/pdf/2505.08995)  

**Abstract**: This work presents a Hierarchical Multi-Agent Reinforcement Learning framework for analyzing simulated air combat scenarios involving heterogeneous agents. The objective is to identify effective Courses of Action that lead to mission success within preset simulations, thereby enabling the exploration of real-world defense scenarios at low cost and in a safe-to-fail setting. Applying deep Reinforcement Learning in this context poses specific challenges, such as complex flight dynamics, the exponential size of the state and action spaces in multi-agent systems, and the capability to integrate real-time control of individual units with look-ahead planning. To address these challenges, the decision-making process is split into two levels of abstraction: low-level policies control individual units, while a high-level commander policy issues macro commands aligned with the overall mission targets. This hierarchical structure facilitates the training process by exploiting policy symmetries of individual agents and by separating control from command tasks. The low-level policies are trained for individual combat control in a curriculum of increasing complexity. The high-level commander is then trained on mission targets given pre-trained control policies. The empirical validation confirms the advantages of the proposed framework. 

---
# Beyond the Known: Decision Making with Counterfactual Reasoning Decision Transformer 

**Authors**: Minh Hoang Nguyen, Linh Le Pham Van, Thommen George Karimpanal, Sunil Gupta, Hung Le  

**Link**: [PDF](https://arxiv.org/pdf/2505.09114)  

**Abstract**: Decision Transformers (DT) play a crucial role in modern reinforcement learning, leveraging offline datasets to achieve impressive results across various domains. However, DT requires high-quality, comprehensive data to perform optimally. In real-world applications, the lack of training data and the scarcity of optimal behaviours make training on offline datasets challenging, as suboptimal data can hinder performance. To address this, we propose the Counterfactual Reasoning Decision Transformer (CRDT), a novel framework inspired by counterfactual reasoning. CRDT enhances DT ability to reason beyond known data by generating and utilizing counterfactual experiences, enabling improved decision-making in unseen scenarios. Experiments across Atari and D4RL benchmarks, including scenarios with limited data and altered dynamics, demonstrate that CRDT outperforms conventional DT approaches. Additionally, reasoning counterfactually allows the DT agent to obtain stitching abilities, combining suboptimal trajectories, without architectural modifications. These results highlight the potential of counterfactual reasoning to enhance reinforcement learning agents' performance and generalization capabilities. 

---
# Deploying Foundation Model-Enabled Air and Ground Robots in the Field: Challenges and Opportunities 

**Authors**: Zachary Ravichandran, Fernando Cladera, Jason Hughes, Varun Murali, M. Ani Hsieh, George J. Pappas, Camillo J. Taylor, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.09477)  

**Abstract**: The integration of foundation models (FMs) into robotics has enabled robots to understand natural language and reason about the semantics in their environments. However, existing FM-enabled robots primary operate in closed-world settings, where the robot is given a full prior map or has a full view of its workspace. This paper addresses the deployment of FM-enabled robots in the field, where missions often require a robot to operate in large-scale and unstructured environments. To effectively accomplish these missions, robots must actively explore their environments, navigate obstacle-cluttered terrain, handle unexpected sensor inputs, and operate with compute constraints. We discuss recent deployments of SPINE, our LLM-enabled autonomy framework, in field robotic settings. To the best of our knowledge, we present the first demonstration of large-scale LLM-enabled robot planning in unstructured environments with several kilometers of missions. SPINE is agnostic to a particular LLM, which allows us to distill small language models capable of running onboard size, weight and power (SWaP) limited platforms. Via preliminary model distillation work, we then present the first language-driven UAV planner using on-device language models. We conclude our paper by proposing several promising directions for future research. 

---
# Air-Ground Collaboration for Language-Specified Missions in Unknown Environments 

**Authors**: Fernando Cladera, Zachary Ravichandran, Jason Hughes, Varun Murali, Carlos Nieto-Granda, M. Ani Hsieh, George J. Pappas, Camillo J. Taylor, Vijay Kumar  

**Link**: [PDF](https://arxiv.org/pdf/2505.09108)  

**Abstract**: As autonomous robotic systems become increasingly mature, users will want to specify missions at the level of intent rather than in low-level detail. Language is an expressive and intuitive medium for such mission specification. However, realizing language-guided robotic teams requires overcoming significant technical hurdles. Interpreting and realizing language-specified missions requires advanced semantic reasoning. Successful heterogeneous robots must effectively coordinate actions and share information across varying viewpoints. Additionally, communication between robots is typically intermittent, necessitating robust strategies that leverage communication opportunities to maintain coordination and achieve mission objectives. In this work, we present a first-of-its-kind system where an unmanned aerial vehicle (UAV) and an unmanned ground vehicle (UGV) are able to collaboratively accomplish missions specified in natural language while reacting to changes in specification on the fly. We leverage a Large Language Model (LLM)-enabled planner to reason over semantic-metric maps that are built online and opportunistically shared between an aerial and a ground robot. We consider task-driven navigation in urban and rural areas. Our system must infer mission-relevant semantics and actively acquire information via semantic mapping. In both ground and air-ground teaming experiments, we demonstrate our system on seven different natural-language specifications at up to kilometer-scale navigation. 

---
# SALM: A Multi-Agent Framework for Language Model-Driven Social Network Simulation 

**Authors**: Gaurav Koley  

**Link**: [PDF](https://arxiv.org/pdf/2505.09081)  

**Abstract**: Contemporary approaches to agent-based modeling (ABM) of social systems have traditionally emphasized rule-based behaviors, limiting their ability to capture nuanced dynamics by moving beyond predefined rules and leveraging contextual understanding from LMs of human social interaction. This paper presents SALM (Social Agent LM Framework), a novel approach for integrating language models (LMs) into social network simulation that achieves unprecedented temporal stability in multi-agent scenarios. Our primary contributions include: (1) a hierarchical prompting architecture enabling stable simulation beyond 4,000 timesteps while reducing token usage by 73%, (2) an attention-based memory system achieving 80% cache hit rates (95% CI [78%, 82%]) with sub-linear memory growth of 9.5%, and (3) formal bounds on personality stability. Through extensive validation against SNAP ego networks, we demonstrate the first LLM-based framework capable of modeling long-term social phenomena while maintaining empirically validated behavioral fidelity. 

---
# Continual Reinforcement Learning via Autoencoder-Driven Task and New Environment Recognition 

**Authors**: Zeki Doruk Erden, Donia Gasmi, Boi Faltings  

**Link**: [PDF](https://arxiv.org/pdf/2505.09003)  

**Abstract**: Continual learning for reinforcement learning agents remains a significant challenge, particularly in preserving and leveraging existing information without an external signal to indicate changes in tasks or environments. In this study, we explore the effectiveness of autoencoders in detecting new tasks and matching observed environments to previously encountered ones. Our approach integrates policy optimization with familiarity autoencoders within an end-to-end continual learning system. This system can recognize and learn new tasks or environments while preserving knowledge from earlier experiences and can selectively retrieve relevant knowledge when re-encountering a known environment. Initial results demonstrate successful continual learning without external signals to indicate task changes or reencounters, showing promise for this methodology. 

---
# CellTypeAgent: Trustworthy cell type annotation with Large Language Models 

**Authors**: Jiawen Chen, Jianghao Zhang, Huaxiu Yao, Yun Li  

**Link**: [PDF](https://arxiv.org/pdf/2505.08844)  

**Abstract**: Cell type annotation is a critical yet laborious step in single-cell RNA sequencing analysis. We present a trustworthy large language model (LLM)-agent, CellTypeAgent, which integrates LLMs with verification from relevant databases. CellTypeAgent achieves higher accuracy than existing methods while mitigating hallucinations. We evaluated CellTypeAgent across nine real datasets involving 303 cell types from 36 tissues. This combined approach holds promise for more efficient and reliable cell type annotation. 

---
# Multi-source Plume Tracing via Multi-Agent Reinforcement Learning 

**Authors**: Pedro Antonio Alarcon Granadeno, Theodore Chambers, Jane Cleland-Huang  

**Link**: [PDF](https://arxiv.org/pdf/2505.08825)  

**Abstract**: Industrial catastrophes like the Bhopal disaster (1984) and the Aliso Canyon gas leak (2015) demonstrate the urgent need for rapid and reliable plume tracing algorithms to protect public health and the environment. Traditional methods, such as gradient-based or biologically inspired approaches, often fail in realistic, turbulent conditions. To address these challenges, we present a Multi-Agent Reinforcement Learning (MARL) algorithm designed for localizing multiple airborne pollution sources using a swarm of small uncrewed aerial systems (sUAS). Our method models the problem as a Partially Observable Markov Game (POMG), employing a Long Short-Term Memory (LSTM)-based Action-specific Double Deep Recurrent Q-Network (ADDRQN) that uses full sequences of historical action-observation pairs, effectively approximating latent states. Unlike prior work, we use a general-purpose simulation environment based on the Gaussian Plume Model (GPM), incorporating realistic elements such as a three-dimensional environment, sensor noise, multiple interacting agents, and multiple plume sources. The incorporation of action histories as part of the inputs further enhances the adaptability of our model in complex, partially observable environments. Extensive simulations show that our algorithm significantly outperforms conventional approaches. Specifically, our model allows agents to explore only 1.29\% of the environment to successfully locate pollution sources. 

---
# Self Rewarding Self Improving 

**Authors**: Toby Simonds, Kevin Lopez, Akira Yoshiyama, Dominique Garmier  

**Link**: [PDF](https://arxiv.org/pdf/2505.08827)  

**Abstract**: We demonstrate that large language models can effectively self-improve through self-judging without requiring reference solutions, leveraging the inherent asymmetry between generating and verifying solutions. Our experiments on Countdown puzzles and MIT Integration Bee problems show that models can provide reliable reward signals without ground truth answers, enabling reinforcement learning in domains previously not possible. By implementing self-judging, we achieve significant performance gains maintaining alignment with formal verification. When combined with synthetic question generation, we establish a complete self-improvement loop where models generate practice problems, solve them, and evaluate their own performance-achieving an 8% improvement with Qwen 2.5 7B over baseline and surpassing GPT-4o performance on integration tasks. Our findings demonstrate that LLM judges can provide effective reward signals for training models, unlocking many reinforcement learning environments previously limited by the difficulty of creating programmatic rewards. This suggests a potential paradigm shift toward AI systems that continuously improve through self-directed learning rather than human-guided training, potentially accelerating progress in domains with scarce training data or complex evaluation requirements. 

---
# Security of Internet of Agents: Attacks and Countermeasures 

**Authors**: Yuntao Wang, Yanghe Pan, Shaolong Guo, Zhou Su  

**Link**: [PDF](https://arxiv.org/pdf/2505.08807)  

**Abstract**: With the rise of large language and vision-language models, AI agents have evolved into autonomous, interactive systems capable of perception, reasoning, and decision-making. As they proliferate across virtual and physical domains, the Internet of Agents (IoA) has emerged as a key infrastructure for enabling scalable and secure coordination among heterogeneous agents. This survey offers a comprehensive examination of the security and privacy landscape in IoA systems. We begin by outlining the IoA architecture and its distinct vulnerabilities compared to traditional networks, focusing on four critical aspects: identity authentication threats, cross-agent trust issues, embodied security, and privacy risks. We then review existing and emerging defense mechanisms and highlight persistent challenges. Finally, we identify open research directions to advance the development of resilient and privacy-preserving IoA ecosystems. 

---
