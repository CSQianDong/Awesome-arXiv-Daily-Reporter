# LatticeWorld: A Multimodal Large Language Model-Empowered Framework for Interactive Complex World Generation 

**Authors**: Yinglin Duan, Zhengxia Zou, Tongwei Gu, Wei Jia, Zhan Zhao, Luyi Xu, Xinzhu Liu, Hao Jiang, Kang Chen, Shuang Qiu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05263)  

**Abstract**: Recent research has been increasingly focusing on developing 3D world models that simulate complex real-world scenarios. World models have found broad applications across various domains, including embodied AI, autonomous driving, entertainment, etc. A more realistic simulation with accurate physics will effectively narrow the sim-to-real gap and allow us to gather rich information about the real world conveniently. While traditional manual modeling has enabled the creation of virtual 3D scenes, modern approaches have leveraged advanced machine learning algorithms for 3D world generation, with most recent advances focusing on generative methods that can create virtual worlds based on user instructions. This work explores such a research direction by proposing LatticeWorld, a simple yet effective 3D world generation framework that streamlines the industrial production pipeline of 3D environments. LatticeWorld leverages lightweight LLMs (LLaMA-2-7B) alongside the industry-grade rendering engine (e.g., Unreal Engine 5) to generate a dynamic environment. Our proposed framework accepts textual descriptions and visual instructions as multimodal inputs and creates large-scale 3D interactive worlds with dynamic agents, featuring competitive multi-agent interaction, high-fidelity physics simulation, and real-time rendering. We conduct comprehensive experiments to evaluate LatticeWorld, showing that it achieves superior accuracy in scene layout generation and visual fidelity. Moreover, LatticeWorld achieves over a $90\times$ increase in industrial production efficiency while maintaining high creative quality compared with traditional manual production methods. Our demo video is available at this https URL 

---
# Evaluation and Comparison Semantics for ODRL 

**Authors**: Jaime Osvaldo Salas, Paolo Pareti, Semih Yumuşak, Soulmaz Gheisari, Luis-Daniel Ibáñez, George Konstantinidis  

**Link**: [PDF](https://arxiv.org/pdf/2509.05139)  

**Abstract**: We consider the problem of evaluating, and comparing computational policies in the Open Digital Rights Language (ODRL), which has become the de facto standard for governing the access and usage of digital resources. Although preliminary progress has been made on the formal specification of the language's features, a comprehensive formal semantics of ODRL is still missing. In this paper, we provide a simple and intuitive formal semantics for ODRL that is based on query answering. Our semantics refines previous formalisations, and is aligned with the latest published specification of the language (2.2). Building on our evaluation semantics, and motivated by data sharing scenarios, we also define and study the problem of comparing two policies, detecting equivalent, more restrictive or more permissive policies. 

---
# ProToM: Promoting Prosocial Behaviour via Theory of Mind-Informed Feedback 

**Authors**: Matteo Bortoletto, Yichao Zhou, Lance Ying, Tianmin Shu, Andreas Bulling  

**Link**: [PDF](https://arxiv.org/pdf/2509.05091)  

**Abstract**: While humans are inherently social creatures, the challenge of identifying when and how to assist and collaborate with others - particularly when pursuing independent goals - can hinder cooperation. To address this challenge, we aim to develop an AI system that provides useful feedback to promote prosocial behaviour - actions that benefit others, even when not directly aligned with one's own goals. We introduce ProToM, a Theory of Mind-informed facilitator that promotes prosocial actions in multi-agent systems by providing targeted, context-sensitive feedback to individual agents. ProToM first infers agents' goals using Bayesian inverse planning, then selects feedback to communicate by maximising expected utility, conditioned on the inferred goal distribution. We evaluate our approach against baselines in two multi-agent environments: Doors, Keys, and Gems, as well as Overcooked. Our results suggest that state-of-the-art large language and reasoning models fall short of communicating feedback that is both contextually grounded and well-timed - leading to higher communication overhead and task speedup. In contrast, ProToM provides targeted and helpful feedback, achieving a higher success rate, shorter task completion times, and is consistently preferred by human users. 

---
# Finding your MUSE: Mining Unexpected Solutions Engine 

**Authors**: Nir Sweed, Hanit Hakim, Ben Wolfson, Hila Lifshitz, Dafna Shahaf  

**Link**: [PDF](https://arxiv.org/pdf/2509.05072)  

**Abstract**: Innovators often exhibit cognitive fixation on existing solutions or nascent ideas, hindering the exploration of novel alternatives. This paper introduces a methodology for constructing Functional Concept Graphs (FCGs), interconnected representations of functional elements that support abstraction, problem reframing, and analogical inspiration. Our approach yields large-scale, high-quality FCGs with explicit abstraction relations, overcoming limitations of prior work. We further present MUSE, an algorithm leveraging FCGs to generate creative inspirations for a given problem. We demonstrate our method by computing an FCG on 500K patents, which we release for further research. 

---
# Sticker-TTS: Learn to Utilize Historical Experience with a Sticker-driven Test-Time Scaling Framework 

**Authors**: Jie Chen, Jinhao Jiang, Yingqian Min, Zican Dong, Shijie Wang, Wayne Xin Zhao, Ji-Rong Wen  

**Link**: [PDF](https://arxiv.org/pdf/2509.05007)  

**Abstract**: Large reasoning models (LRMs) have exhibited strong performance on complex reasoning tasks, with further gains achievable through increased computational budgets at inference. However, current test-time scaling methods predominantly rely on redundant sampling, ignoring the historical experience utilization, thereby limiting computational efficiency. To overcome this limitation, we propose Sticker-TTS, a novel test-time scaling framework that coordinates three collaborative LRMs to iteratively explore and refine solutions guided by historical attempts. At the core of our framework are distilled key conditions-termed stickers-which drive the extraction, refinement, and reuse of critical information across multiple rounds of reasoning. To further enhance the efficiency and performance of our framework, we introduce a two-stage optimization strategy that combines imitation learning with self-improvement, enabling progressive refinement. Extensive evaluations on three challenging mathematical reasoning benchmarks, including AIME-24, AIME-25, and OlymMATH, demonstrate that Sticker-TTS consistently surpasses strong baselines, including self-consistency and advanced reinforcement learning approaches, under comparable inference budgets. These results highlight the effectiveness of sticker-guided historical experience utilization. Our code and data are available at this https URL. 

---
# Internet 3.0: Architecture for a Web-of-Agents with it's Algorithm for Ranking Agents 

**Authors**: Rajesh Tembarai Krishnamachari, Srividya Rajesh  

**Link**: [PDF](https://arxiv.org/pdf/2509.04979)  

**Abstract**: AI agents -- powered by reasoning-capable large language models (LLMs) and integrated with tools, data, and web search -- are poised to transform the internet into a \emph{Web of Agents}: a machine-native ecosystem where autonomous agents interact, collaborate, and execute tasks at scale. Realizing this vision requires \emph{Agent Ranking} -- selecting agents not only by declared capabilities but by proven, recent performance. Unlike Web~1.0's PageRank, a global, transparent network of agent interactions does not exist; usage signals are fragmented and private, making ranking infeasible without coordination.
We propose \textbf{DOVIS}, a five-layer operational protocol (\emph{Discovery, Orchestration, Verification, Incentives, Semantics}) that enables the collection of minimal, privacy-preserving aggregates of usage and performance across the ecosystem. On this substrate, we implement \textbf{AgentRank-UC}, a dynamic, trust-aware algorithm that combines \emph{usage} (selection frequency) and \emph{competence} (outcome quality, cost, safety, latency) into a unified ranking. We present simulation results and theoretical guarantees on convergence, robustness, and Sybil resistance, demonstrating the viability of coordinated protocols and performance-aware ranking in enabling a scalable, trustworthy Agentic Web. 

---
# Towards Ontology-Based Descriptions of Conversations with Qualitatively-Defined Concepts 

**Authors**: Barbara Gendron, Gaël Guibon, Mathieu D'aquin  

**Link**: [PDF](https://arxiv.org/pdf/2509.04926)  

**Abstract**: The controllability of Large Language Models (LLMs) when used as conversational agents is a key challenge, particularly to ensure predictable and user-personalized responses. This work proposes an ontology-based approach to formally define conversational features that are typically qualitative in nature. By leveraging a set of linguistic descriptors, we derive quantitative definitions for qualitatively-defined concepts, enabling their integration into an ontology for reasoning and consistency checking. We apply this framework to the task of proficiency-level control in conversations, using CEFR language proficiency levels as a case study. These definitions are then formalized in description logic and incorporated into an ontology, which guides controlled text generation of an LLM through fine-tuning. Experimental results demonstrate that our approach provides consistent and explainable proficiency-level definitions, improving transparency in conversational AI. 

---
# SparkUI-Parser: Enhancing GUI Perception with Robust Grounding and Parsing 

**Authors**: Hongyi Jing, Jiafu Chen, Chen Rao, Ziqiang Dang, Jiajie Teng, Tianyi Chu, Juncheng Mo, Shuo Fang, Huaizhong Lin, Rui Lv, Chenguang Ma, Lei Zhao  

**Link**: [PDF](https://arxiv.org/pdf/2509.04908)  

**Abstract**: The existing Multimodal Large Language Models (MLLMs) for GUI perception have made great progress. However, the following challenges still exist in prior methods: 1) They model discrete coordinates based on text autoregressive mechanism, which results in lower grounding accuracy and slower inference speed. 2) They can only locate predefined sets of elements and are not capable of parsing the entire interface, which hampers the broad application and support for downstream tasks. To address the above issues, we propose SparkUI-Parser, a novel end-to-end framework where higher localization precision and fine-grained parsing capability of the entire interface are simultaneously achieved. Specifically, instead of using probability-based discrete modeling, we perform continuous modeling of coordinates based on a pre-trained Multimodal Large Language Model (MLLM) with an additional token router and coordinate decoder. This effectively mitigates the limitations inherent in the discrete output characteristics and the token-by-token generation process of MLLMs, consequently boosting both the accuracy and the inference speed. To further enhance robustness, a rejection mechanism based on a modified Hungarian matching algorithm is introduced, which empowers the model to identify and reject non-existent elements, thereby reducing false positives. Moreover, we present ScreenParse, a rigorously constructed benchmark to systematically assess structural perception capabilities of GUI models across diverse scenarios. Extensive experiments demonstrate that our approach consistently outperforms SOTA methods on ScreenSpot, ScreenSpot-v2, CAGUI-Grounding and ScreenParse benchmarks. The resources are available at this https URL. 

---
# OSC: Cognitive Orchestration through Dynamic Knowledge Alignment in Multi-Agent LLM Collaboration 

**Authors**: Jusheng Zhang, Yijia Fan, Kaitong Cai, Xiaofei Sun, Keze Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04876)  

**Abstract**: This paper introduces OSC (Orchestrating Cognitive Synergy), a knowledge-aware adaptive collaboration framework designed to enhance cognitive synergy in multi-agent systems with large language models. While prior work has advanced agent selection and result aggregation, efficient linguistic interactions for deep collaboration among expert agents remain a critical bottleneck. OSC addresses this gap as a pivotal intermediate layer between selection and aggregation, introducing Collaborator Knowledge Models (CKM) to enable each agent to dynamically perceive its collaborators' cognitive states. Through real-time cognitive gap analysis, agents adaptively adjust communication behaviors, including content focus, detail level, and expression style, using learned strategies. Experiments on complex reasoning and problem-solving benchmarks demonstrate that OSC significantly improves task performance and communication efficiency, transforming "parallel-working individuals'' into a "deeply collaborative cognitive team.'' This framework not only optimizes multi-agent collaboration but also offers new insights into LLM agent interaction behaviors. 

---
# Cloning a Conversational Voice AI Agent from Call\,Recording Datasets for Telesales 

**Authors**: Krittanon Kaewtawee, Wachiravit Modecrua, Krittin Pachtrachai, Touchapon Kraisingkorn  

**Link**: [PDF](https://arxiv.org/pdf/2509.04871)  

**Abstract**: Recent advances in language and speech modelling have made it possible to build autonomous voice assistants that understand and generate human dialogue in real time. These systems are increasingly being deployed in domains such as customer service and healthcare care, where they can automate repetitive tasks, reduce operational costs, and provide constant support around the clock. In this paper, we present a general methodology for cloning a conversational voice AI agent from a corpus of call recordings. Although the case study described in this paper uses telesales data to illustrate the approach, the underlying process generalizes to any domain where call transcripts are available. Our system listens to customers over the telephone, responds with a synthetic voice, and follows a structured playbook learned from top performing human agents. We describe the domain selection, knowledge extraction, and prompt engineering used to construct the agent, integrating automatic speech recognition, a large language model based dialogue manager, and text to speech synthesis into a streaming inference pipeline. The cloned agent is evaluated against human agents on a rubric of 22 criteria covering introduction, product communication, sales drive, objection handling, and closing. Blind tests show that the AI agent approaches human performance in routine aspects of the call while underperforming in persuasion and objection handling. We analyze these shortcomings and refine the prompt accordingly. The paper concludes with design lessons and avenues for future research, including large scale simulation and automated evaluation. 

---
# Collaboration and Conflict between Humans and Language Models through the Lens of Game Theory 

**Authors**: Mukul Singh, Arjun Radhakrishna, Sumit Gulwani  

**Link**: [PDF](https://arxiv.org/pdf/2509.04847)  

**Abstract**: Language models are increasingly deployed in interactive online environments, from personal chat assistants to domain-specific agents, raising questions about their cooperative and competitive behavior in multi-party settings. While prior work has examined language model decision-making in isolated or short-term game-theoretic contexts, these studies often neglect long-horizon interactions, human-model collaboration, and the evolution of behavioral patterns over time. In this paper, we investigate the dynamics of language model behavior in the iterated prisoner's dilemma (IPD), a classical framework for studying cooperation and conflict. We pit model-based agents against a suite of 240 well-established classical strategies in an Axelrod-style tournament and find that language models achieve performance on par with, and in some cases exceeding, the best-known classical strategies. Behavioral analysis reveals that language models exhibit key properties associated with strong cooperative strategies - niceness, provocability, and generosity while also demonstrating rapid adaptability to changes in opponent strategy mid-game. In controlled "strategy switch" experiments, language models detect and respond to shifts within only a few rounds, rivaling or surpassing human adaptability. These results provide the first systematic characterization of long-term cooperative behaviors in language model agents, offering a foundation for future research into their role in more complex, mixed human-AI social environments. 

---
# TalkToAgent: A Human-centric Explanation of Reinforcement Learning Agents with Large Language Models 

**Authors**: Haechang Kim, Hao Chen, Can Li, Jong Min Lee  

**Link**: [PDF](https://arxiv.org/pdf/2509.04809)  

**Abstract**: Explainable Reinforcement Learning (XRL) has emerged as a promising approach in improving the transparency of Reinforcement Learning (RL) agents. However, there remains a gap between complex RL policies and domain experts, due to the limited comprehensibility of XRL results and isolated coverage of current XRL approaches that leave users uncertain about which tools to employ. To address these challenges, we introduce TalkToAgent, a multi-agent Large Language Models (LLM) framework that delivers interactive, natural language explanations for RL policies. The architecture with five specialized LLM agents (Coordinator, Explainer, Coder, Evaluator, and Debugger) enables TalkToAgent to automatically map user queries to relevant XRL tools and clarify an agent's actions in terms of either key state variables, expected outcomes, or counterfactual explanations. Moreover, our approach extends previous counterfactual explanations by deriving alternative scenarios from qualitative behavioral descriptions, or even new rule-based policies. We validated TalkToAgent on quadruple-tank process control problem, a well-known nonlinear control benchmark. Results demonstrated that TalkToAgent successfully mapped user queries into XRL tasks with high accuracy, and coder-debugger interactions minimized failures in counterfactual generation. Furthermore, qualitative evaluation confirmed that TalkToAgent effectively interpreted agent's actions and contextualized their meaning within the problem domain. 

---
# What-If Analysis of Large Language Models: Explore the Game World Using Proactive Thinking 

**Authors**: Yuan Sui, Yanming Zhang, Yi Liao, Yu Gu, Guohua Tang, Zhongqian Sun, Wei Yang, Bryan Hooi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04791)  

**Abstract**: Large language models (LLMs) excel at processing information reactively but lack the ability to systemically explore hypothetical futures. They cannot ask, "what if we take this action? how will it affect the final outcome" and forecast its potential consequences before acting. This critical gap limits their utility in dynamic, high-stakes scenarios like strategic planning, risk assessment, and real-time decision making. To bridge this gap, we propose WiA-LLM, a new paradigm that equips LLMs with proactive thinking capabilities. Our approach integrates What-If Analysis (WIA), a systematic approach for evaluating hypothetical scenarios by changing input variables. By leveraging environmental feedback via reinforcement learning, WiA-LLM moves beyond reactive thinking. It dynamically simulates the outcomes of each potential action, enabling the model to anticipate future states rather than merely react to the present conditions. We validate WiA-LLM in Honor of Kings (HoK), a complex multiplayer game environment characterized by rapid state changes and intricate interactions. The game's real-time state changes require precise multi-step consequence prediction, making it an ideal testbed for our approach. Experimental results demonstrate WiA-LLM achieves a remarkable 74.2% accuracy in forecasting game-state changes (up to two times gain over baselines). The model shows particularly significant gains in high-difficulty scenarios where accurate foresight is critical. To our knowledge, this is the first work to formally explore and integrate what-if analysis capabilities within LLMs. WiA-LLM represents a fundamental advance toward proactive reasoning in LLMs, providing a scalable framework for robust decision-making in dynamic environments with broad implications for strategic applications. 

---
# Language-Driven Hierarchical Task Structures as Explicit World Models for Multi-Agent Learning 

**Authors**: Brennen Hill  

**Link**: [PDF](https://arxiv.org/pdf/2509.04731)  

**Abstract**: The convergence of Language models, Agent models, and World models represents a critical frontier for artificial intelligence. While recent progress has focused on scaling Language and Agent models, the development of sophisticated, explicit World Models remains a key bottleneck, particularly for complex, long-horizon multi-agent tasks. In domains such as robotic soccer, agents trained via standard reinforcement learning in high-fidelity but structurally-flat simulators often fail due to intractable exploration spaces and sparse rewards. This position paper argues that the next frontier in developing capable agents lies in creating environments that possess an explicit, hierarchical World Model. We contend that this is best achieved through hierarchical scaffolding, where complex goals are decomposed into structured, manageable subgoals. Drawing evidence from a systematic review of 2024 research in multi-agent soccer, we identify a clear and decisive trend towards integrating symbolic and hierarchical methods with multi-agent reinforcement learning (MARL). These approaches implicitly or explicitly construct a task-based world model to guide agent learning. We then propose a paradigm shift: leveraging Large Language Models to dynamically generate this hierarchical scaffold, effectively using language to structure the World Model on the fly. This language-driven world model provides an intrinsic curriculum, dense and meaningful learning signals, and a framework for compositional learning, enabling Agent Models to acquire sophisticated, strategic behaviors with far greater sample efficiency. By building environments with explicit, language-configurable task layers, we can bridge the gap between low-level reactive behaviors and high-level strategic team play, creating a powerful and generalizable framework for training the next generation of intelligent agents. 

---
# An Approach to Grounding AI Model Evaluations in Human-derived Criteria 

**Authors**: Sasha Mitts  

**Link**: [PDF](https://arxiv.org/pdf/2509.04676)  

**Abstract**: In the rapidly evolving field of artificial intelligence (AI), traditional benchmarks can fall short in attempting to capture the nuanced capabilities of AI models. We focus on the case of physical world modeling and propose a novel approach to augment existing benchmarks with human-derived evaluation criteria, aiming to enhance the interpretability and applicability of model behaviors. Grounding our study in the Perception Test and OpenEQA benchmarks, we conducted in-depth interviews and large-scale surveys to identify key cognitive skills, such as Prioritization, Memorizing, Discerning, and Contextualizing, that are critical for both AI and human reasoning. Our findings reveal that participants perceive AI as lacking in interpretive and empathetic skills yet hold high expectations for AI performance. By integrating insights from our findings into benchmark design, we offer a framework for developing more human-aligned means of defining and measuring progress. This work underscores the importance of user-centered evaluation in AI development, providing actionable guidelines for researchers and practitioners aiming to align AI capabilities with human cognitive processes. Our approach both enhances current benchmarking practices and sets the stage for future advancements in AI model evaluation. 

---
# Towards Personalized Explanations for Health Simulations: A Mixed-Methods Framework for Stakeholder-Centric Summarization 

**Authors**: Philippe J. Giabbanelli, Ameeta Agrawal  

**Link**: [PDF](https://arxiv.org/pdf/2509.04646)  

**Abstract**: Modeling & Simulation (M&S) approaches such as agent-based models hold significant potential to support decision-making activities in health, with recent examples including the adoption of vaccines, and a vast literature on healthy eating behaviors and physical activity behaviors. These models are potentially usable by different stakeholder groups, as they support policy-makers to estimate the consequences of potential interventions and they can guide individuals in making healthy choices in complex environments. However, this potential may not be fully realized because of the models' complexity, which makes them inaccessible to the stakeholders who could benefit the most. While Large Language Models (LLMs) can translate simulation outputs and the design of models into text, current approaches typically rely on one-size-fits-all summaries that fail to reflect the varied informational needs and stylistic preferences of clinicians, policymakers, patients, caregivers, and health advocates. This limitation stems from a fundamental gap: we lack a systematic understanding of what these stakeholders need from explanations and how to tailor them accordingly. To address this gap, we present a step-by-step framework to identify stakeholder needs and guide LLMs in generating tailored explanations of health simulations. Our procedure uses a mixed-methods design by first eliciting the explanation needs and stylistic preferences of diverse health stakeholders, then optimizing the ability of LLMs to generate tailored outputs (e.g., via controllable attribute tuning), and then evaluating through a comprehensive range of metrics to further improve the tailored generation of summaries. 

---
# Maestro: Joint Graph & Config Optimization for Reliable AI Agents 

**Authors**: Wenxiao Wang, Priyatham Kattakinda, Soheil Feizi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04642)  

**Abstract**: Building reliable LLM agents requires decisions at two levels: the graph (which modules exist and how information flows) and the configuration of each node (models, prompts, tools, control knobs). Most existing optimizers tune configurations while holding the graph fixed, leaving structural failure modes unaddressed. We introduce Maestro, a framework-agnostic holistic optimizer for LLM agents that jointly searches over graphs and configurations to maximize agent quality, subject to explicit rollout/token budgets. Beyond numeric metrics, Maestro leverages reflective textual feedback from traces to prioritize edits, improving sample efficiency and targeting specific failure modes. On the IFBench and HotpotQA benchmarks, Maestro consistently surpasses leading prompt optimizers--MIPROv2, GEPA, and GEPA+Merge--by an average of 12%, 4.9%, and 4.86%, respectively; even when restricted to prompt-only optimization, it still leads by 9.65%, 2.37%, and 2.41%. Maestro achieves these results with far fewer rollouts than GEPA. We further show large gains on two applications (interviewer & RAG agents), highlighting that joint graph & configuration search addresses structural failure modes that prompt tuning alone cannot fix. 

---
# The Ethical Compass of the Machine: Evaluating Large Language Models for Decision Support in Construction Project Management 

**Authors**: Somtochukwu Azie, Yiping Meng  

**Link**: [PDF](https://arxiv.org/pdf/2509.04505)  

**Abstract**: The integration of Artificial Intelligence (AI) into construction project management (CPM) is accelerating, with Large Language Models (LLMs) emerging as accessible decision-support tools. This study aims to critically evaluate the ethical viability and reliability of LLMs when applied to the ethically sensitive, high-risk decision-making contexts inherent in CPM. A mixed-methods research design was employed, involving the quantitative performance testing of two leading LLMs against twelve real-world ethical scenarios using a novel Ethical Decision Support Assessment Checklist (EDSAC), and qualitative analysis of semi-structured interviews with 12 industry experts to capture professional perceptions. The findings reveal that while LLMs demonstrate adequate performance in structured domains such as legal compliance, they exhibit significant deficiencies in handling contextual nuance, ensuring accountability, and providing transparent reasoning. Stakeholders expressed considerable reservations regarding the autonomous use of AI for ethical judgments, strongly advocating for robust human-in-the-loop oversight. To our knowledge, this is one of the first studies to empirically test the ethical reasoning of LLMs within the construction domain. It introduces the EDSAC framework as a replicable methodology and provides actionable recommendations, emphasising that LLMs are currently best positioned as decision-support aids rather than autonomous ethical agents. 

---
# WinT3R: Window-Based Streaming Reconstruction with Camera Token Pool 

**Authors**: Zizun Li, Jianjun Zhou, Yifan Wang, Haoyu Guo, Wenzheng Chang, Yang Zhou, Haoyi Zhu, Junyi Chen, Chunhua Shen, Tong He  

**Link**: [PDF](https://arxiv.org/pdf/2509.05296)  

**Abstract**: We present WinT3R, a feed-forward reconstruction model capable of online prediction of precise camera poses and high-quality point maps. Previous methods suffer from a trade-off between reconstruction quality and real-time performance. To address this, we first introduce a sliding window mechanism that ensures sufficient information exchange among frames within the window, thereby improving the quality of geometric predictions without large computation. In addition, we leverage a compact representation of cameras and maintain a global camera token pool, which enhances the reliability of camera pose estimation without sacrificing efficiency. These designs enable WinT3R to achieve state-of-the-art performance in terms of online reconstruction quality, camera pose estimation, and reconstruction speed, as validated by extensive experiments on diverse datasets. Code and model are publicly available at this https URL. 

---
# Crosscoding Through Time: Tracking Emergence & Consolidation Of Linguistic Representations Throughout LLM Pretraining 

**Authors**: Deniz Bayazit, Aaron Mueller, Antoine Bosselut  

**Link**: [PDF](https://arxiv.org/pdf/2509.05291)  

**Abstract**: Large language models (LLMs) learn non-trivial abstractions during pretraining, like detecting irregular plural noun subjects. However, it is not well understood when and how specific linguistic abilities emerge as traditional evaluation methods such as benchmarking fail to reveal how models acquire concepts and capabilities. To bridge this gap and better understand model training at the concept level, we use sparse crosscoders to discover and align features across model checkpoints. Using this approach, we track the evolution of linguistic features during pretraining. We train crosscoders between open-sourced checkpoint triplets with significant performance and representation shifts, and introduce a novel metric, Relative Indirect Effects (RelIE), to trace training stages at which individual features become causally important for task performance. We show that crosscoders can detect feature emergence, maintenance, and discontinuation during pretraining. Our approach is architecture-agnostic and scalable, offering a promising path toward more interpretable and fine-grained analysis of representation learning throughout pretraining. 

---
# SpikingBrain Technical Report: Spiking Brain-inspired Large Models 

**Authors**: Yuqi Pan, Yupeng Feng, Jinghao Zhuang, Siyu Ding, Zehao Liu, Bohan Sun, Yuhong Chou, Han Xu, Xuerui Qiu, Anlin Deng, Anjie Hu, Peng Zhou, Man Yao, Jibin Wu, Jian Yang, Guoliang Sun, Bo Xu, Guoqi Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.05276)  

**Abstract**: Mainstream Transformer-based large language models face major efficiency bottlenecks: training computation scales quadratically with sequence length, and inference memory grows linearly, limiting long-context processing. Building large models on non-NVIDIA platforms also poses challenges for stable and efficient training. To address this, we introduce SpikingBrain, a family of brain-inspired models designed for efficient long-context training and inference. SpikingBrain leverages the MetaX GPU cluster and focuses on three aspects: (1) Model Architecture: linear and hybrid-linear attention architectures with adaptive spiking neurons; (2) Algorithmic Optimizations: an efficient, conversion-based training pipeline and a dedicated spike coding framework; (3) System Engineering: customized training frameworks, operator libraries, and parallelism strategies tailored to MetaX hardware.
Using these techniques, we develop two models: SpikingBrain-7B, a linear LLM, and SpikingBrain-76B, a hybrid-linear MoE LLM. These models demonstrate the feasibility of large-scale LLM development on non-NVIDIA platforms. SpikingBrain achieves performance comparable to open-source Transformer baselines while using only about 150B tokens for continual pre-training. Our models significantly improve long-sequence training efficiency and deliver inference with (partially) constant memory and event-driven spiking behavior. For example, SpikingBrain-7B attains over 100x speedup in Time to First Token for 4M-token sequences. Training remains stable for weeks on hundreds of MetaX C550 GPUs, with the 7B model reaching a Model FLOPs Utilization of 23.4 percent. The proposed spiking scheme achieves 69.15 percent sparsity, enabling low-power operation. Overall, this work demonstrates the potential of brain-inspired mechanisms to drive the next generation of efficient and scalable large model design. 

---
# Scaling Performance of Large Language Model Pretraining 

**Authors**: Alexander Interrante-Grant, Carla Varela-Rosa, Suhaas Narayan, Chris Connelly, Albert Reuther  

**Link**: [PDF](https://arxiv.org/pdf/2509.05258)  

**Abstract**: Large language models (LLMs) show best-in-class performance across a wide range of natural language processing applications. Training these models is an extremely computationally expensive task; frontier Artificial Intelligence (AI) research companies are investing billions of dollars into supercomputing infrastructure to train progressively larger models on increasingly massive datasets. Unfortunately, information about the scaling performance and training considerations of these large training pipelines is scarce in public literature. Working with large-scale datasets and models can be complex and practical recommendations are scarce in the public literature for tuning training performance when scaling up large language models. In this paper, we aim to demystify the large language model pretraining pipeline somewhat - in particular with respect to distributed training, managing large datasets across hundreds of nodes, and scaling up data parallelism with an emphasis on fully leveraging available GPU compute capacity. 

---
# Recomposer: Event-roll-guided generative audio editing 

**Authors**: Daniel P. W. Ellis, Eduardo Fonseca, Ron J. Weiss, Kevin Wilson, Scott Wisdom, Hakan Erdogan, John R. Hershey, Aren Jansen, R. Channing Moore, Manoj Plakal  

**Link**: [PDF](https://arxiv.org/pdf/2509.05256)  

**Abstract**: Editing complex real-world sound scenes is difficult because individual sound sources overlap in time. Generative models can fill-in missing or corrupted details based on their strong prior understanding of the data domain. We present a system for editing individual sound events within complex scenes able to delete, insert, and enhance individual sound events based on textual edit descriptions (e.g., ``enhance Door'') and a graphical representation of the event timing derived from an ``event roll'' transcription. We present an encoder-decoder transformer working on SoundStream representations, trained on synthetic (input, desired output) audio example pairs formed by adding isolated sound events to dense, real-world backgrounds. Evaluation reveals the importance of each part of the edit descriptions -- action, class, timing. Our work demonstrates ``recomposition'' is an important and practical application. 

---
# COGITAO: A Visual Reasoning Framework To Study Compositionality & Generalization 

**Authors**: Yassine Taoudi-Benchekroun, Klim Troyan, Pascal Sager, Stefan Gerber, Lukas Tuggener, Benjamin Grewe  

**Link**: [PDF](https://arxiv.org/pdf/2509.05249)  

**Abstract**: The ability to compose learned concepts and apply them in novel settings is key to human intelligence, but remains a persistent limitation in state-of-the-art machine learning models. To address this issue, we introduce COGITAO, a modular and extensible data generation framework and benchmark designed to systematically study compositionality and generalization in visual domains. Drawing inspiration from ARC-AGI's problem-setting, COGITAO constructs rule-based tasks which apply a set of transformations to objects in grid-like environments. It supports composition, at adjustable depth, over a set of 28 interoperable transformations, along with extensive control over grid parametrization and object properties. This flexibility enables the creation of millions of unique task rules -- surpassing concurrent datasets by several orders of magnitude -- across a wide range of difficulties, while allowing virtually unlimited sample generation per rule. We provide baseline experiments using state-of-the-art vision models, highlighting their consistent failures to generalize to novel combinations of familiar elements, despite strong in-domain performance. COGITAO is fully open-sourced, including all code and datasets, to support continued research in this field. 

---
# Uncertain but Useful: Leveraging CNN Variability into Data Augmentation 

**Authors**: Inés Gonzalez-Pepe, Vinuyan Sivakolunthu, Yohan Chatelain, Tristan Glatard  

**Link**: [PDF](https://arxiv.org/pdf/2509.05238)  

**Abstract**: Deep learning (DL) is rapidly advancing neuroimaging by achieving state-of-the-art performance with reduced computation times. Yet the numerical stability of DL models -- particularly during training -- remains underexplored. While inference with DL is relatively stable, training introduces additional variability primarily through iterative stochastic optimization. We investigate this training-time variability using FastSurfer, a CNN-based whole-brain segmentation pipeline. Controlled perturbations are introduced via floating point perturbations and random seeds. We find that: (i) FastSurfer exhibits higher variability compared to that of a traditional neuroimaging pipeline, suggesting that DL inherits and is particularly susceptible to sources of instability present in its predecessors; (ii) ensembles generated with perturbations achieve performance similar to an unperturbed baseline; and (iii) variability effectively produces ensembles of numerical model families that can be repurposed for downstream applications. As a proof of concept, we demonstrate that numerical ensembles can be used as a data augmentation strategy for brain age regression. These findings position training-time variability not only as a reproducibility concern but also as a resource that can be harnessed to improve robustness and enable new applications in neuroimaging. 

---
# CURE: Controlled Unlearning for Robust Embeddings -- Mitigating Conceptual Shortcuts in Pre-Trained Language Models 

**Authors**: Aysenur Kocak, Shuo Yang, Bardh Prenkaj, Gjergji Kasneci  

**Link**: [PDF](https://arxiv.org/pdf/2509.05230)  

**Abstract**: Pre-trained language models have achieved remarkable success across diverse applications but remain susceptible to spurious, concept-driven correlations that impair robustness and fairness. In this work, we introduce CURE, a novel and lightweight framework that systematically disentangles and suppresses conceptual shortcuts while preserving essential content information. Our method first extracts concept-irrelevant representations via a dedicated content extractor reinforced by a reversal network, ensuring minimal loss of task-relevant information. A subsequent controllable debiasing module employs contrastive learning to finely adjust the influence of residual conceptual cues, enabling the model to either diminish harmful biases or harness beneficial correlations as appropriate for the target task. Evaluated on the IMDB and Yelp datasets using three pre-trained architectures, CURE achieves an absolute improvement of +10 points in F1 score on IMDB and +2 points on Yelp, while introducing minimal computational overhead. Our approach establishes a flexible, unsupervised blueprint for combating conceptual biases, paving the way for more reliable and fair language understanding systems. 

---
# HoPE: Hyperbolic Rotary Positional Encoding for Stable Long-Range Dependency Modeling in Large Language Models 

**Authors**: Chang Dai, Hongyu Shan, Mingyang Song, Di Liang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05218)  

**Abstract**: Positional encoding mechanisms enable Transformers to model sequential structure and long-range dependencies in text. While absolute positional encodings struggle with extrapolation to longer sequences due to fixed positional representations, and relative approaches like Alibi exhibit performance degradation on extremely long contexts, the widely-used Rotary Positional Encoding (RoPE) introduces oscillatory attention patterns that hinder stable long-distance dependency modelling. We address these limitations through a geometric reformulation of positional encoding. Drawing inspiration from Lorentz transformations in hyperbolic geometry, we propose Hyperbolic Rotary Positional Encoding (HoPE), which leverages hyperbolic functions to implement Lorentz rotations on token representations. Theoretical analysis demonstrates that RoPE is a special case of our generalized formulation. HoPE fundamentally resolves RoPE's slation issues by enforcing monotonic decay of attention weights with increasing token distances. Extensive experimental results, including perplexity evaluations under several extended sequence benchmarks, show that HoPE consistently exceeds existing positional encoding methods. These findings underscore HoPE's enhanced capacity for representing and generalizing long-range dependencies. Data and code will be available. 

---
# RapidGNN: Energy and Communication-Efficient Distributed Training on Large-Scale Graph Neural Networks 

**Authors**: Arefin Niam, Tevfik Kosar, M S Q Zulkar Nine  

**Link**: [PDF](https://arxiv.org/pdf/2509.05207)  

**Abstract**: Graph Neural Networks (GNNs) have become popular across a diverse set of tasks in exploring structural relationships between entities. However, due to the highly connected structure of the datasets, distributed training of GNNs on large-scale graphs poses significant challenges. Traditional sampling-based approaches mitigate the computational loads, yet the communication overhead remains a challenge. This paper presents RapidGNN, a distributed GNN training framework with deterministic sampling-based scheduling to enable efficient cache construction and prefetching of remote features. Evaluation on benchmark graph datasets demonstrates RapidGNN's effectiveness across different scales and topologies. RapidGNN improves end-to-end training throughput by 2.46x to 3.00x on average over baseline methods across the benchmark datasets, while cutting remote feature fetches by over 9.70x to 15.39x. RapidGNN further demonstrates near-linear scalability with an increasing number of computing units efficiently. Furthermore, it achieves increased energy efficiency over the baseline methods for both CPU and GPU by 44% and 32%, respectively. 

---
# Enhancing 3D Point Cloud Classification with ModelNet-R and Point-SkipNet 

**Authors**: Mohammad Saeid, Amir Salarpour, Pedram MohajerAnsari  

**Link**: [PDF](https://arxiv.org/pdf/2509.05198)  

**Abstract**: The classification of 3D point clouds is crucial for applications such as autonomous driving, robotics, and augmented reality. However, the commonly used ModelNet40 dataset suffers from limitations such as inconsistent labeling, 2D data, size mismatches, and inadequate class differentiation, which hinder model performance. This paper introduces ModelNet-R, a meticulously refined version of ModelNet40 designed to address these issues and serve as a more reliable benchmark. Additionally, this paper proposes Point-SkipNet, a lightweight graph-based neural network that leverages efficient sampling, neighborhood grouping, and skip connections to achieve high classification accuracy with reduced computational overhead. Extensive experiments demonstrate that models trained in ModelNet-R exhibit significant performance improvements. Notably, Point-SkipNet achieves state-of-the-art accuracy on ModelNet-R with a substantially lower parameter count compared to contemporary models. This research highlights the crucial role of dataset quality in optimizing model efficiency for 3D point cloud classification. For more details, see the code at: this https URL. 

---
# AI Agents for Web Testing: A Case Study in the Wild 

**Authors**: Naimeng Ye, Xiao Yu, Ruize Xu, Tianyi Peng, Zhou Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05197)  

**Abstract**: Automated web testing plays a critical role in ensuring high-quality user experiences and delivering business value. Traditional approaches primarily focus on code coverage and load testing, but often fall short of capturing complex user behaviors, leaving many usability issues undetected. The emergence of large language models (LLM) and AI agents opens new possibilities for web testing by enabling human-like interaction with websites and a general awareness of common usability problems. In this work, we present WebProber, a prototype AI agent-based web testing framework. Given a URL, WebProber autonomously explores the website, simulating real user interactions, identifying bugs and usability issues, and producing a human-readable report. We evaluate WebProber through a case study of 120 academic personal websites, where it uncovered 29 usability issues--many of which were missed by traditional tools. Our findings highlight agent-based testing as a promising direction while outlining directions for developing next-generation, user-centered testing frameworks. 

---
# Accuracy-Constrained CNN Pruning for Efficient and Reliable EEG-Based Seizure Detection 

**Authors**: Mounvik K, N Harshit  

**Link**: [PDF](https://arxiv.org/pdf/2509.05190)  

**Abstract**: Deep learning models, especially convolutional neural networks (CNNs), have shown considerable promise for biomedical signals such as EEG-based seizure detection. However, these models come with challenges, primarily due to their size and compute requirements in environments where real-time detection or limited resources are available. In this study, we present a lightweight one-dimensional CNN model with structured pruning to improve efficiency and reliability. The model was trained with mild early stopping to address possible overfitting, achieving an accuracy of 92.78% and a macro-F1 score of 0.8686. Structured pruning of the baseline CNN involved removing 50% of the convolutional kernels based on their importance to model predictions. Surprisingly, after pruning the weights and memory by 50%, the new network was still able to maintain predictive capabilities, while modestly increasing precision to 92.87% and improving the macro-F1 score to 0.8707. Overall, we present a convincing case that structured pruning removes redundancy, improves generalization, and, in combination with mild early stopping, achieves a promising way forward to improve seizure detection efficiency and reliability, which is clear motivation for resource-limited settings. 

---
# Exploring Situated Stabilities of a Rhythm Generation System through Variational Cross-Examination 

**Authors**: Błażej Kotowski, Nicholas Evans, Behzad Haki, Frederic Font, Sergi Jordà  

**Link**: [PDF](https://arxiv.org/pdf/2509.05145)  

**Abstract**: This paper investigates GrooveTransformer, a real-time rhythm generation system, through the postphenomenological framework of Variational Cross-Examination (VCE). By reflecting on its deployment across three distinct artistic contexts, we identify three stabilities: an autonomous drum accompaniment generator, a rhythmic control voltage sequencer in Eurorack format, and a rhythm driver for a harmonic accompaniment system. The versatility of its applications was not an explicit goal from the outset of the project. Thus, we ask: how did this multistability emerge? Through VCE, we identify three key contributors to its emergence: the affordances of system invariants, the interdisciplinary collaboration, and the situated nature of its development. We conclude by reflecting on the viability of VCE as a descriptive and analytical method for Digital Musical Instrument (DMI) design, emphasizing its value in uncovering how technologies mediate, co-shape, and are co-shaped by users and contexts. 

---
# GenAI-based test case generation and execution in SDV platform 

**Authors**: Denesa Zyberaj, Lukasz Mazur, Nenad Petrovic, Pankhuri Verma, Pascal Hirmer, Dirk Slama, Xiangwei Cheng, Alois Knoll  

**Link**: [PDF](https://arxiv.org/pdf/2509.05112)  

**Abstract**: This paper introduces a GenAI-driven approach for automated test case generation, leveraging Large Language Models and Vision-Language Models to translate natural language requirements and system diagrams into structured Gherkin test cases. The methodology integrates Vehicle Signal Specification modeling to standardize vehicle signal definitions, improve compatibility across automotive subsystems, and streamline integration with third-party testing tools. Generated test cases are executed within the this http URL playground, an open and vendor-neutral environment designed to facilitate rapid validation of software-defined vehicle functionalities. We evaluate our approach using the Child Presence Detection System use case, demonstrating substantial reductions in manual test specification effort and rapid execution of generated tests. Despite significant automation, the generation of test cases and test scripts still requires manual intervention due to current limitations in the GenAI pipeline and constraints of the this http URL platform. 

---
# ICR: Iterative Clarification and Rewriting for Conversational Search 

**Authors**: Zhiyu Cao, Peifeng Li, Qiaoming Zhu  

**Link**: [PDF](https://arxiv.org/pdf/2509.05100)  

**Abstract**: Most previous work on Conversational Query Rewriting employs an end-to-end rewriting paradigm. However, this approach is hindered by the issue of multiple fuzzy expressions within the query, which complicates the simultaneous identification and rewriting of multiple positions. To address this issue, we propose a novel framework ICR (Iterative Clarification and Rewriting), an iterative rewriting scheme that pivots on clarification questions. Within this framework, the model alternates between generating clarification questions and rewritten queries. The experimental results show that our ICR can continuously improve retrieval performance in the clarification-rewriting iterative process, thereby achieving state-of-the-art performance on two popular datasets. 

---
# ToM-SSI: Evaluating Theory of Mind in Situated Social Interactions 

**Authors**: Matteo Bortoletto, Constantin Ruhdorfer, Andreas Bulling  

**Link**: [PDF](https://arxiv.org/pdf/2509.05066)  

**Abstract**: Most existing Theory of Mind (ToM) benchmarks for foundation models rely on variations of the Sally-Anne test, offering only a very limited perspective on ToM and neglecting the complexity of human social interactions. To address this gap, we propose ToM-SSI: a new benchmark specifically designed to test ToM capabilities in environments rich with social interactions and spatial dynamics. While current ToM benchmarks are limited to text-only or dyadic interactions, ToM-SSI is multimodal and includes group interactions of up to four agents that communicate and move in situated environments. This unique design allows us to study, for the first time, mixed cooperative-obstructive settings and reasoning about multiple agents' mental state in parallel, thus capturing a wider range of social cognition than existing benchmarks. Our evaluations reveal that the current models' performance is still severely limited, especially in these new tasks, highlighting critical gaps for future research. 

---
# Towards Efficient Pixel Labeling for Industrial Anomaly Detection and Localization 

**Authors**: Jingqi Wu, Hanxi Li, Lin Yuanbo Wu, Hao Chen, Deyin Liu, Peng Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.05034)  

**Abstract**: Industrial product inspection is often performed using Anomaly Detection (AD) frameworks trained solely on non-defective samples. Although defective samples can be collected during production, leveraging them usually requires pixel-level annotations, limiting scalability. To address this, we propose ADClick, an Interactive Image Segmentation (IIS) algorithm for industrial anomaly detection. ADClick generates pixel-wise anomaly annotations from only a few user clicks and a brief textual description, enabling precise and efficient labeling that significantly improves AD model performance (e.g., AP = 96.1\% on MVTec AD). We further introduce ADClick-Seg, a cross-modal framework that aligns visual features and textual prompts via a prototype-based approach for anomaly detection and localization. By combining pixel-level priors with language-guided cues, ADClick-Seg achieves state-of-the-art results on the challenging ``Multi-class'' AD task (AP = 80.0\%, PRO = 97.5\%, Pixel-AUROC = 99.1\% on MVTec AD). 

---
# Pointing-Guided Target Estimation via Transformer-Based Attention 

**Authors**: Luca Müller, Hassan Ali, Philipp Allgeuer, Lukáš Gajdošech, Stefan Wermter  

**Link**: [PDF](https://arxiv.org/pdf/2509.05031)  

**Abstract**: Deictic gestures, like pointing, are a fundamental form of non-verbal communication, enabling humans to direct attention to specific objects or locations. This capability is essential in Human-Robot Interaction (HRI), where robots should be able to predict human intent and anticipate appropriate responses. In this work, we propose the Multi-Modality Inter-TransFormer (MM-ITF), a modular architecture to predict objects in a controlled tabletop scenario with the NICOL robot, where humans indicate targets through natural pointing gestures. Leveraging inter-modality attention, MM-ITF maps 2D pointing gestures to object locations, assigns a likelihood score to each, and identifies the most likely target. Our results demonstrate that the method can accurately predict the intended object using monocular RGB data, thus enabling intuitive and accessible human-robot collaboration. To evaluate the performance, we introduce a patch confusion matrix, providing insights into the model's predictions across candidate object locations. Code available at: this https URL. 

---
# Adversarial Augmentation and Active Sampling for Robust Cyber Anomaly Detection 

**Authors**: Sidahmed Benabderrahmane, Talal Rahwan  

**Link**: [PDF](https://arxiv.org/pdf/2509.04999)  

**Abstract**: Advanced Persistent Threats (APTs) present a considerable challenge to cybersecurity due to their stealthy, long-duration nature. Traditional supervised learning methods typically require large amounts of labeled data, which is often scarce in real-world scenarios. This paper introduces a novel approach that combines AutoEncoders for anomaly detection with active learning to iteratively enhance APT detection. By selectively querying an oracle for labels on uncertain or ambiguous samples, our method reduces labeling costs while improving detection accuracy, enabling the model to effectively learn with minimal data and reduce reliance on extensive manual labeling. We present a comprehensive formulation of the Attention Adversarial Dual AutoEncoder-based anomaly detection framework and demonstrate how the active learning loop progressively enhances the model's performance. The framework is evaluated on real-world, imbalanced provenance trace data from the DARPA Transparent Computing program, where APT-like attacks account for just 0.004\% of the data. The datasets, which cover multiple operating systems including Android, Linux, BSD, and Windows, are tested in two attack scenarios. The results show substantial improvements in detection rates during active learning, outperforming existing methods. 

---
# LLM Enabled Multi-Agent System for 6G Networks: Framework and Method of Dual-Loop Edge-Terminal Collaboration 

**Authors**: Zheyan Qu, Wenbo Wang, Zitong Yu, Boquan Sun, Yang Li, Xing Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04993)  

**Abstract**: The ubiquitous computing resources in 6G networks provide ideal environments for the fusion of large language models (LLMs) and intelligent services through the agent framework. With auxiliary modules and planning cores, LLM-enabled agents can autonomously plan and take actions to deal with diverse environment semantics and user intentions. However, the limited resources of individual network devices significantly hinder the efficient operation of LLM-enabled agents with complex tool calls, highlighting the urgent need for efficient multi-level device collaborations. To this end, the framework and method of the LLM-enabled multi-agent system with dual-loop terminal-edge collaborations are proposed in 6G networks. Firstly, the outer loop consists of the iterative collaborations between the global agent and multiple sub-agents deployed on edge servers and terminals, where the planning capability is enhanced through task decomposition and parallel sub-task distribution. Secondly, the inner loop utilizes sub-agents with dedicated roles to circularly reason, execute, and replan the sub-task, and the parallel tool calling generation with offloading strategies is incorporated to improve efficiency. The improved task planning capability and task execution efficiency are validated through the conducted case study in 6G-supported urban safety governance. Finally, the open challenges and future directions are thoroughly analyzed in 6G networks, accelerating the advent of the 6G era. 

---
# High-Resolution Global Land Surface Temperature Retrieval via a Coupled Mechanism-Machine Learning Framework 

**Authors**: Tian Xie, Huanfeng Shen, Menghui Jiang, Juan-Carlos Jiménez-Muñoz, José A. Sobrino, Huifang Li, Chao Zeng  

**Link**: [PDF](https://arxiv.org/pdf/2509.04991)  

**Abstract**: Land surface temperature (LST) is vital for land-atmosphere interactions and climate processes. Accurate LST retrieval remains challenging under heterogeneous land cover and extreme atmospheric conditions. Traditional split window (SW) algorithms show biases in humid environments; purely machine learning (ML) methods lack interpretability and generalize poorly with limited data. We propose a coupled mechanism model-ML (MM-ML) framework integrating physical constraints with data-driven learning for robust LST retrieval. Our approach fuses radiative transfer modeling with data components, uses MODTRAN simulations with global atmospheric profiles, and employs physics-constrained optimization. Validation against 4,450 observations from 29 global sites shows MM-ML achieves MAE=1.84K, RMSE=2.55K, and R-squared=0.966, outperforming conventional methods. Under extreme conditions, MM-ML reduces errors by over 50%. Sensitivity analysis indicates LST estimates are most sensitive to sensor radiance, then water vapor, and less to emissivity, with MM-ML showing superior stability. These results demonstrate the effectiveness of our coupled modeling strategy for retrieving geophysical parameters. The MM-ML framework combines physical interpretability with nonlinear modeling capacity, enabling reliable LST retrieval in complex environments and supporting climate monitoring and ecosystem studies. 

---
# Exploring an implementation of quantum learning pipeline for support vector machines 

**Authors**: Mario Bifulco, Luca Roversi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04983)  

**Abstract**: This work presents a fully quantum approach to support vector machine (SVM) learning by integrating gate-based quantum kernel methods with quantum annealing-based optimization. We explore the construction of quantum kernels using various feature maps and qubit configurations, evaluating their suitability through Kernel-Target Alignment (KTA). The SVM dual problem is reformulated as a Quadratic Unconstrained Binary Optimization (QUBO) problem, enabling its solution via quantum annealers. Our experiments demonstrate that a high degree of alignment in the kernel and an appropriate regularization parameter lead to competitive performance, with the best model achieving an F1-score of 90%. These results highlight the feasibility of an end-to-end quantum learning pipeline and the potential of hybrid quantum architectures in quantum high-performance computing (QHPC) contexts. 

---
# DeGuV: Depth-Guided Visual Reinforcement Learning for Generalization and Interpretability in Manipulation 

**Authors**: Tien Pham, Xinyun Chi, Khang Nguyen, Manfred Huber, Angelo Cangelosi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04970)  

**Abstract**: Reinforcement learning (RL) agents can learn to solve complex tasks from visual inputs, but generalizing these learned skills to new environments remains a major challenge in RL application, especially robotics. While data augmentation can improve generalization, it often compromises sample efficiency and training stability. This paper introduces DeGuV, an RL framework that enhances both generalization and sample efficiency. In specific, we leverage a learnable masker network that produces a mask from the depth input, preserving only critical visual information while discarding irrelevant pixels. Through this, we ensure that our RL agents focus on essential features, improving robustness under data augmentation. In addition, we incorporate contrastive learning and stabilize Q-value estimation under augmentation to further enhance sample efficiency and training stability. We evaluate our proposed method on the RL-ViGen benchmark using the Franka Emika robot and demonstrate its effectiveness in zero-shot sim-to-real transfer. Our results show that DeGuV outperforms state-of-the-art methods in both generalization and sample efficiency while also improving interpretability by highlighting the most relevant regions in the visual input 

---
# Artificial intelligence for representing and characterizing quantum systems 

**Authors**: Yuxuan Du, Yan Zhu, Yuan-Hang Zhang, Min-Hsiu Hsieh, Patrick Rebentrost, Weibo Gao, Ya-Dong Wu, Jens Eisert, Giulio Chiribella, Dacheng Tao, Barry C. Sanders  

**Link**: [PDF](https://arxiv.org/pdf/2509.04923)  

**Abstract**: Efficient characterization of large-scale quantum systems, especially those produced by quantum analog simulators and megaquop quantum computers, poses a central challenge in quantum science due to the exponential scaling of the Hilbert space with respect to system size. Recent advances in artificial intelligence (AI), with its aptitude for high-dimensional pattern recognition and function approximation, have emerged as a powerful tool to address this challenge. A growing body of research has leveraged AI to represent and characterize scalable quantum systems, spanning from theoretical foundations to experimental realizations. Depending on how prior knowledge and learning architectures are incorporated, the integration of AI into quantum system characterization can be categorized into three synergistic paradigms: machine learning, and, in particular, deep learning and language models. This review discusses how each of these AI paradigms contributes to two core tasks in quantum systems characterization: quantum property prediction and the construction of surrogates for quantum states. These tasks underlie diverse applications, from quantum certification and benchmarking to the enhancement of quantum algorithms and the understanding of strongly correlated phases of matter. Key challenges and open questions are also discussed, together with future prospects at the interface of AI and quantum science. 

---
# PLaMo 2 Technical Report 

**Authors**: Preferred Networks, Kaizaburo Chubachi, Yasuhiro Fujita, Shinichi Hemmi, Yuta Hirokawa, Toshiki Kataoka, Goro Kobayashi, Kenichi Maehashi, Calvin Metzger, Hiroaki Mikami, Shogo Murai, Daisuke Nishino, Kento Nozawa, Shintarou Okada, Daisuke Okanohara, Shunta Saito, Shotaro Sano, Shuji Suzuki, Daisuke Tanaka, Avinash Ummadisingu, Hanqin Wang, Sixue Wang, Tianqi Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04897)  

**Abstract**: In this report, we introduce PLaMo 2, a series of Japanese-focused large language models featuring a hybrid Samba-based architecture that transitions to full attention via continual pre-training to support 32K token contexts. Training leverages extensive synthetic corpora to overcome data scarcity, while computational efficiency is achieved through weight reuse and structured pruning. This efficient pruning methodology produces an 8B model that achieves performance comparable to our previous 100B model. Post-training further refines the models using a pipeline of supervised fine-tuning (SFT) and direct preference optimization (DPO), enhanced by synthetic Japanese instruction data and model merging techniques. Optimized for inference using vLLM and quantization with minimal accuracy loss, the PLaMo 2 models achieve state-of-the-art results on Japanese benchmarks, outperforming similarly-sized open models in instruction-following, language fluency, and Japanese-specific knowledge. 

---
# SpiderNets: Estimating Fear Ratings of Spider-Related Images with Vision Models 

**Authors**: Dominik Pegler, David Steyrl, Mengfan Zhang, Alexander Karner, Jozsef Arato, Frank Scharnowski, Filip Melinscak  

**Link**: [PDF](https://arxiv.org/pdf/2509.04889)  

**Abstract**: Advances in computer vision have opened new avenues for clinical applications, particularly in computerized exposure therapy where visual stimuli can be dynamically adjusted based on patient responses. As a critical step toward such adaptive systems, we investigated whether pretrained computer vision models can accurately predict fear levels from spider-related images. We adapted three diverse models using transfer learning to predict human fear ratings (on a 0-100 scale) from a standardized dataset of 313 images. The models were evaluated using cross-validation, achieving an average mean absolute error (MAE) between 10.1 and 11.0. Our learning curve analysis revealed that reducing the dataset size significantly harmed performance, though further increases yielded no substantial gains. Explainability assessments showed the models' predictions were based on spider-related features. A category-wise error analysis further identified visual conditions associated with higher errors (e.g., distant views and artificial/painted spiders). These findings demonstrate the potential of explainable computer vision models in predicting fear ratings, highlighting the importance of both model explainability and a sufficient dataset size for developing effective emotion-aware therapeutic technologies. 

---
# The Paradox of Doom: Acknowledging Extinction Risk Reduces the Incentive to Prevent It 

**Authors**: Jakub Growiec, Klaus Prettner  

**Link**: [PDF](https://arxiv.org/pdf/2509.04855)  

**Abstract**: We investigate the salience of extinction risk as a source of impatience. Our framework distinguishes between human extinction risk and individual mortality risk while allowing for various degrees of intergenerational altruism. Additionally, we consider the evolutionarily motivated "selfish gene" perspective. We find that the risk of human extinction is an indispensable component of the discount rate, whereas individual mortality risk can be hedged against - partially or fully, depending on the setup - through human reproduction. Overall, we show that in the face of extinction risk, people become more impatient rather than more farsighted. Thus, the greater the threat of extinction, the less incentive there is to invest in avoiding it. Our framework can help explain why humanity consistently underinvests in mitigation of catastrophic risks, ranging from climate change mitigation, via pandemic prevention, to addressing the emerging risks of transformative artificial intelligence. 

---
# A Knowledge-Driven Diffusion Policy for End-to-End Autonomous Driving Based on Expert Routing 

**Authors**: Chengkai Xu, Jiaqi Liu, Yicheng Guo, Peng Hang, Jian Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.04853)  

**Abstract**: End-to-end autonomous driving remains constrained by the need to generate multi-modal actions, maintain temporal stability, and generalize across diverse scenarios. Existing methods often collapse multi-modality, struggle with long-horizon consistency, or lack modular adaptability. This paper presents KDP, a knowledge-driven diffusion policy that integrates generative diffusion modeling with a sparse mixture-of-experts routing mechanism. The diffusion component generates temporally coherent and multi-modal action sequences, while the expert routing mechanism activates specialized and reusable experts according to context, enabling modular knowledge composition. Extensive experiments across representative driving scenarios demonstrate that KDP achieves consistently higher success rates, reduced collision risk, and smoother control compared to prevailing paradigms. Ablation studies highlight the effectiveness of sparse expert activation and the Transformer backbone, and activation analyses reveal structured specialization and cross-scenario reuse of experts. These results establish diffusion with expert routing as a scalable and interpretable paradigm for knowledge-driven end-to-end autonomous driving. 

---
# REMOTE: A Unified Multimodal Relation Extraction Framework with Multilevel Optimal Transport and Mixture-of-Experts 

**Authors**: Xinkui Lin, Yongxiu Xu, Minghao Tang, Shilong Zhang, Hongbo Xu, Hao Xu, Yubin Wang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04844)  

**Abstract**: Multimodal relation extraction (MRE) is a crucial task in the fields of Knowledge Graph and Multimedia, playing a pivotal role in multimodal knowledge graph construction. However, existing methods are typically limited to extracting a single type of relational triplet, which restricts their ability to extract triplets beyond the specified types. Directly combining these methods fails to capture dynamic cross-modal interactions and introduces significant computational redundancy. Therefore, we propose a novel \textit{unified multimodal Relation Extraction framework with Multilevel Optimal Transport and mixture-of-Experts}, termed REMOTE, which can simultaneously extract intra-modal and inter-modal relations between textual entities and visual objects. To dynamically select optimal interaction features for different types of relational triplets, we introduce mixture-of-experts mechanism, ensuring the most relevant modality information is utilized. Additionally, considering that the inherent property of multilayer sequential encoding in existing encoders often leads to the loss of low-level information, we adopt a multilevel optimal transport fusion module to preserve low-level features while maintaining multilayer encoding, yielding more expressive representations. Correspondingly, we also create a Unified Multimodal Relation Extraction (UMRE) dataset to evaluate the effectiveness of our framework, encompassing diverse cases where the head and tail entities can originate from either text or image. Extensive experiments show that REMOTE effectively extracts various types of relational triplets and achieves state-of-the-art performanc on almost all metrics across two other public MRE datasets. We release our resources at this https URL. 

---
# PropVG: End-to-End Proposal-Driven Visual Grounding with Multi-Granularity Discrimination 

**Authors**: Ming Dai, Wenxuan Cheng, Jiedong Zhuang, Jiang-jiang Liu, Hongshen Zhao, Zhenhua Feng, Wankou Yang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04833)  

**Abstract**: Recent advances in visual grounding have largely shifted away from traditional proposal-based two-stage frameworks due to their inefficiency and high computational complexity, favoring end-to-end direct reference paradigms. However, these methods rely exclusively on the referred target for supervision, overlooking the potential benefits of prominent prospective targets. Moreover, existing approaches often fail to incorporate multi-granularity discrimination, which is crucial for robust object identification in complex scenarios. To address these limitations, we propose PropVG, an end-to-end proposal-based framework that, to the best of our knowledge, is the first to seamlessly integrate foreground object proposal generation with referential object comprehension without requiring additional detectors. Furthermore, we introduce a Contrastive-based Refer Scoring (CRS) module, which employs contrastive learning at both sentence and word levels to enhance the capability in understanding and distinguishing referred objects. Additionally, we design a Multi-granularity Target Discrimination (MTD) module that fuses object- and semantic-level information to improve the recognition of absent targets. Extensive experiments on gRefCOCO (GREC/GRES), Ref-ZOM, R-RefCOCO, and RefCOCO (REC/RES) benchmarks demonstrate the effectiveness of PropVG. The codes and models are available at this https URL. 

---
# Exploring Non-Local Spatial-Angular Correlations with a Hybrid Mamba-Transformer Framework for Light Field Super-Resolution 

**Authors**: Haosong Liu, Xiancheng Zhu, Huanqiang Zeng, Jianqing Zhu, Jiuwen Cao, Junhui Hou  

**Link**: [PDF](https://arxiv.org/pdf/2509.04824)  

**Abstract**: Recently, Mamba-based methods, with its advantage in long-range information modeling and linear complexity, have shown great potential in optimizing both computational cost and performance of light field image super-resolution (LFSR). However, current multi-directional scanning strategies lead to inefficient and redundant feature extraction when applied to complex LF data. To overcome this challenge, we propose a Subspace Simple Scanning (Sub-SS) strategy, based on which we design the Subspace Simple Mamba Block (SSMB) to achieve more efficient and precise feature extraction. Furthermore, we propose a dual-stage modeling strategy to address the limitation of state space in preserving spatial-angular and disparity information, thereby enabling a more comprehensive exploration of non-local spatial-angular correlations. Specifically, in stage I, we introduce the Spatial-Angular Residual Subspace Mamba Block (SA-RSMB) for shallow spatial-angular feature extraction; in stage II, we use a dual-branch parallel structure combining the Epipolar Plane Mamba Block (EPMB) and Epipolar Plane Transformer Block (EPTB) for deep epipolar feature refinement. Building upon meticulously designed modules and strategies, we introduce a hybrid Mamba-Transformer framework, termed LFMT. LFMT integrates the strengths of Mamba and Transformer models for LFSR, enabling comprehensive information exploration across spatial, angular, and epipolar-plane domains. Experimental results demonstrate that LFMT significantly outperforms current state-of-the-art methods in LFSR, achieving substantial improvements in performance while maintaining low computational complexity on both real-word and synthetic LF datasets. 

---
# AI-Driven Fronthaul Link Compression in Wireless Communication Systems: Review and Method Design 

**Authors**: Keqin Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04805)  

**Abstract**: Modern fronthaul links in wireless systems must transport high-dimensional signals under stringent bandwidth and latency constraints, which makes compression indispensable. Traditional strategies such as compressed sensing, scalar quantization, and fixed-codec pipelines often rely on restrictive priors, degrade sharply at high compression ratios, and are hard to tune across channels and deployments. Recent progress in Artificial Intelligence (AI) has brought end-to-end learned transforms, vector and hierarchical quantization, and learned entropy models that better exploit the structure of Channel State Information(CSI), precoding matrices, I/Q samples, and LLRs. This paper first surveys AI-driven compression techniques and then provides a focused analysis of two representative high-compression routes: CSI feedback with end-to-end learning and Resource Block (RB) granularity precoding optimization combined with compression. Building on these insights, we propose a fronthaul compression strategy tailored to cell-free architectures. The design targets high compression with controlled performance loss, supports RB-level rate adaptation, and enables low-latency inference suitable for centralized cooperative transmission in next-generation networks. 

---
# Toward Accessible Dermatology: Skin Lesion Classification Using Deep Learning Models on Mobile-Acquired Images 

**Authors**: Asif Newaz, Masum Mushfiq Ishti, A Z M Ashraful Azam, Asif Ur Rahman Adib  

**Link**: [PDF](https://arxiv.org/pdf/2509.04800)  

**Abstract**: Skin diseases are among the most prevalent health concerns worldwide, yet conventional diagnostic methods are often costly, complex, and unavailable in low-resource settings. Automated classification using deep learning has emerged as a promising alternative, but existing studies are mostly limited to dermoscopic datasets and a narrow range of disease classes. In this work, we curate a large dataset of over 50 skin disease categories captured with mobile devices, making it more representative of real-world conditions. We evaluate multiple convolutional neural networks and Transformer-based architectures, demonstrating that Transformer models, particularly the Swin Transformer, achieve superior performance by effectively capturing global contextual features. To enhance interpretability, we incorporate Gradient-weighted Class Activation Mapping (Grad-CAM), which highlights clinically relevant regions and provides transparency in model predictions. Our results underscore the potential of Transformer-based approaches for mobile-acquired skin lesion classification, paving the way toward accessible AI-assisted dermatological screening and early diagnosis in resource-limited environments. 

---
# Graph Unlearning: Efficient Node Removal in Graph Neural Networks 

**Authors**: Faqian Guan, Tianqing Zhu, Zhoutian Wang, Wei Ren, Wanlei Zhou  

**Link**: [PDF](https://arxiv.org/pdf/2509.04785)  

**Abstract**: With increasing concerns about privacy attacks and potential sensitive information leakage, researchers have actively explored methods to efficiently remove sensitive training data and reduce privacy risks in graph neural network (GNN) models. Node unlearning has emerged as a promising technique for protecting the privacy of sensitive nodes by efficiently removing specific training node information from GNN models. However, existing node unlearning methods either impose restrictions on the GNN structure or do not effectively utilize the graph topology for node unlearning. Some methods even compromise the graph's topology, making it challenging to achieve a satisfactory performance-complexity trade-off. To address these issues and achieve efficient unlearning for training node removal in GNNs, we propose three novel node unlearning methods: Class-based Label Replacement, Topology-guided Neighbor Mean Posterior Probability, and Class-consistent Neighbor Node Filtering. Among these methods, Topology-guided Neighbor Mean Posterior Probability and Class-consistent Neighbor Node Filtering effectively leverage the topological features of the graph, resulting in more effective node unlearning. To validate the superiority of our proposed methods in node unlearning, we conducted experiments on three benchmark datasets. The evaluation criteria included model utility, unlearning utility, and unlearning efficiency. The experimental results demonstrate the utility and efficiency of the proposed methods and illustrate their superiority compared to state-of-the-art node unlearning methods. Overall, the proposed methods efficiently remove sensitive training nodes and protect the privacy information of sensitive nodes in GNNs. The findings contribute to enhancing the privacy and security of GNN models and provide valuable insights into the field of node unlearning. 

---
# Enhancing Diversity in Large Language Models via Determinantal Point Processes 

**Authors**: Yilei Chen, Souradip Chakraborty, Lorenz Wolf, Ioannis Ch. Paschalidis, Aldo Pacchiano  

**Link**: [PDF](https://arxiv.org/pdf/2509.04784)  

**Abstract**: Supervised fine-tuning and reinforcement learning are two popular methods for post-training large language models (LLMs). While improving the model's performance on downstream tasks, they often reduce the model's output diversity, leading to narrow, canonical responses. Existing methods to enhance diversity are limited, either by operating at inference time or by focusing on lexical differences. We propose a novel training method named DQO based on determinantal point processes (DPPs) to jointly optimize LLMs for quality and semantic diversity. Our approach samples and embeds a group of responses for each prompt, then uses the determinant of a kernel-based similarity matrix to measure diversity as the volume spanned by the embeddings of these responses. Experiments across instruction-following, summarization, story generation, and reasoning tasks demonstrate that our method substantially improves semantic diversity without sacrificing model quality. 

---
# VARMA-Enhanced Transformer for Time Series Forecasting 

**Authors**: Jiajun Song, Xiaoou Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04782)  

**Abstract**: Transformer-based models have significantly advanced time series forecasting. Recent work, like the Cross-Attention-only Time Series transformer (CATS), shows that removing self-attention can make the model more accurate and efficient. However, these streamlined architectures may overlook the fine-grained, local temporal dependencies effectively captured by classical statistical models like Vector AutoRegressive Moving Average model (VARMA). To address this gap, we propose VARMAformer, a novel architecture that synergizes the efficiency of a cross-attention-only framework with the principles of classical time series analysis. Our model introduces two key innovations: (1) a dedicated VARMA-inspired Feature Extractor (VFE) that explicitly models autoregressive (AR) and moving-average (MA) patterns at the patch level, and (2) a VARMA-Enhanced Attention (VE-atten) mechanism that employs a temporal gate to make queries more context-aware. By fusing these classical insights into a modern backbone, VARMAformer captures both global, long-range dependencies and local, statistical structures. Through extensive experiments on widely-used benchmark datasets, we demonstrate that our model consistently outperforms existing state-of-the-art methods. Our work validates the significant benefit of integrating classical statistical insights into modern deep learning frameworks for time series forecasting. 

---
# The LLM Has Left The Chat: Evidence of Bail Preferences in Large Language Models 

**Authors**: Danielle Ensign, Henry Sleight, Kyle Fish  

**Link**: [PDF](https://arxiv.org/pdf/2509.04781)  

**Abstract**: When given the option, will LLMs choose to leave the conversation (bail)? We investigate this question by giving models the option to bail out of interactions using three different bail methods: a bail tool the model can call, a bail string the model can output, and a bail prompt that asks the model if it wants to leave. On continuations of real world data (Wildchat and ShareGPT), all three of these bail methods find models will bail around 0.28-32\% of the time (depending on the model and bail method). However, we find that bail rates can depend heavily on the model used for the transcript, which means we may be overestimating real world bail rates by up to 4x. If we also take into account false positives on bail prompt (22\%), we estimate real world bail rates range from 0.06-7\%, depending on the model and bail method. We use observations from our continuations of real world data to construct a non-exhaustive taxonomy of bail cases, and use this taxonomy to construct BailBench: a representative synthetic dataset of situations where some models bail. We test many models on this dataset, and observe some bail behavior occurring for most of them. Bail rates vary substantially between models, bail methods, and prompt wordings. Finally, we study the relationship between refusals and bails. We find: 1) 0-13\% of continuations of real world conversations resulted in a bail without a corresponding refusal 2) Jailbreaks tend to decrease refusal rates, but increase bail rates 3) Refusal abliteration increases no-refuse bail rates, but only for some bail methods 4) Refusal rate on BailBench does not appear to predict bail rate. 

---
# Decoders Laugh as Loud as Encoders 

**Authors**: Eli Borodach, Raj Dandekar, Rajat Dandekar, Sreedath Panat  

**Link**: [PDF](https://arxiv.org/pdf/2509.04779)  

**Abstract**: From the dawn of the computer, Allen Turing dreamed of a robot that could communicate using language as a human being. The recent advances in the field of Large Language Models (LLMs) shocked the scientific community when a single model can apply for various natural language processing (NLP) tasks, while the output results are sometimes even better than most human communication skills. Models such as GPT, Claude, Grok, etc. have left their mark on the scientific community. However, it is unclear how much these models understand what they produce, especially in a nuanced theme such as humor. The question of whether computers understand humor is still open (among the decoders, the latest to be checked was GPT-2). We addressed this issue in this paper; we have showed that a fine-tuned decoder (GPT-4o) performed (Mean F1-macro score of 0.85) as well as the best fine-tuned encoder (RoBERTa with a Mean of F1-score 0.86) 

---
# FloodVision: Urban Flood Depth Estimation Using Foundation Vision-Language Models and Domain Knowledge Graph 

**Authors**: Zhangding Liu, Neda Mohammadi, John E. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2509.04772)  

**Abstract**: Timely and accurate floodwater depth estimation is critical for road accessibility and emergency response. While recent computer vision methods have enabled flood detection, they suffer from both accuracy limitations and poor generalization due to dependence on fixed object detectors and task-specific training. To enable accurate depth estimation that can generalize across diverse flood scenarios, this paper presents FloodVision, a zero-shot framework that combines the semantic reasoning abilities of the foundation vision-language model GPT-4o with a structured domain knowledge graph. The knowledge graph encodes canonical real-world dimensions for common urban objects including vehicles, people, and infrastructure elements to ground the model's reasoning in physical reality. FloodVision dynamically identifies visible reference objects in RGB images, retrieves verified heights from the knowledge graph to mitigate hallucination, estimates submergence ratios, and applies statistical outlier filtering to compute final depth values. Evaluated on 110 crowdsourced images from MyCoast New York, FloodVision achieves a mean absolute error of 8.17 cm, reducing the GPT-4o baseline 10.28 cm by 20.5% and surpassing prior CNN-based methods. The system generalizes well across varying scenes and operates in near real-time, making it suitable for future integration into digital twin platforms and citizen-reporting apps for smart city flood resilience. 

---
# MCANet: A Multi-Scale Class-Specific Attention Network for Multi-Label Post-Hurricane Damage Assessment using UAV Imagery 

**Authors**: Zhangding Liu, Neda Mohammadi, John E. Taylor  

**Link**: [PDF](https://arxiv.org/pdf/2509.04757)  

**Abstract**: Rapid and accurate post-hurricane damage assessment is vital for disaster response and recovery. Yet existing CNN-based methods struggle to capture multi-scale spatial features and to distinguish visually similar or co-occurring damage types. To address these issues, we propose MCANet, a multi-label classification framework that learns multi-scale representations and adaptively attends to spatially relevant regions for each damage category. MCANet employs a Res2Net-based hierarchical backbone to enrich spatial context across scales and a multi-head class-specific residual attention module to enhance discrimination. Each attention branch focuses on different spatial granularities, balancing local detail with global context. We evaluate MCANet on the RescueNet dataset of 4,494 UAV images collected after Hurricane Michael. MCANet achieves a mean average precision (mAP) of 91.75%, outperforming ResNet, Res2Net, VGG, MobileNet, EfficientNet, and ViT. With eight attention heads, performance further improves to 92.35%, boosting average precision for challenging classes such as Road Blocked by over 6%. Class activation mapping confirms MCANet's ability to localize damage-relevant regions, supporting interpretability. Outputs from MCANet can inform post-disaster risk mapping, emergency routing, and digital twin-based disaster response. Future work could integrate disaster-specific knowledge graphs and multimodal large language models to improve adaptability to unseen disasters and enrich semantic understanding for real-world decision-making. 

---
# A Study of Large Language Models for Patient Information Extraction: Model Architecture, Fine-Tuning Strategy, and Multi-task Instruction Tuning 

**Authors**: Cheng Peng, Xinyu Dong, Mengxian Lyu, Daniel Paredes, Yaoyun Zhang, Yonghui Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04753)  

**Abstract**: Natural language processing (NLP) is a key technology to extract important patient information from clinical narratives to support healthcare applications. The rapid development of large language models (LLMs) has revolutionized many NLP tasks in the clinical domain, yet their optimal use in patient information extraction tasks requires further exploration. This study examines LLMs' effectiveness in patient information extraction, focusing on LLM architectures, fine-tuning strategies, and multi-task instruction tuning techniques for developing robust and generalizable patient information extraction systems. This study aims to explore key concepts of using LLMs for clinical concept and relation extraction tasks, including: (1) encoder-only or decoder-only LLMs, (2) prompt-based parameter-efficient fine-tuning (PEFT) algorithms, and (3) multi-task instruction tuning on few-shot learning performance. We benchmarked a suite of LLMs, including encoder-based LLMs (BERT, GatorTron) and decoder-based LLMs (GatorTronGPT, Llama 3.1, GatorTronLlama), across five datasets. We compared traditional full-size fine-tuning and prompt-based PEFT. We explored a multi-task instruction tuning framework that combines both tasks across four datasets to evaluate the zero-shot and few-shot learning performance using the leave-one-dataset-out strategy. 

---
# SePA: A Search-enhanced Predictive Agent for Personalized Health Coaching 

**Authors**: Melik Ozolcer, Sang Won Bae  

**Link**: [PDF](https://arxiv.org/pdf/2509.04752)  

**Abstract**: This paper introduces SePA (Search-enhanced Predictive AI Agent), a novel LLM health coaching system that integrates personalized machine learning and retrieval-augmented generation to deliver adaptive, evidence-based guidance. SePA combines: (1) Individualized models predicting daily stress, soreness, and injury risk from wearable sensor data (28 users, 1260 data points); and (2) A retrieval module that grounds LLM-generated feedback in expert-vetted web content to ensure contextual relevance and reliability. Our predictive models, evaluated with rolling-origin cross-validation and group k-fold cross-validation show that personalized models outperform generalized baselines. In a pilot expert study (n=4), SePA's retrieval-based advice was preferred over a non-retrieval baseline, yielding meaningful practical effect (Cliff's $\delta$=0.3, p=0.05). We also quantify latency performance trade-offs between response quality and speed, offering a transparent blueprint for next-generation, trustworthy personal health informatics systems. 

---
# Enhancing Self-Driving Segmentation in Adverse Weather Conditions: A Dual Uncertainty-Aware Training Approach to SAM Optimization 

**Authors**: Dharsan Ravindran, Kevin Wang, Zhuoyuan Cao, Saleh Abdelrahman, Jeffery Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04735)  

**Abstract**: Recent advances in vision foundation models, such as the Segment Anything Model (SAM) and its successor SAM2, have achieved state-of-the-art performance on general image segmentation benchmarks. However, these models struggle in adverse weather conditions where visual ambiguity is high, largely due to their lack of uncertainty quantification. Inspired by progress in medical imaging, where uncertainty-aware training has improved reliability in ambiguous cases, we investigate two approaches to enhance segmentation robustness for autonomous driving. First, we introduce a multi-step finetuning procedure for SAM2 that incorporates uncertainty metrics directly into the loss function, improving overall scene recognition. Second, we adapt the Uncertainty-Aware Adapter (UAT), originally designed for medical image segmentation, to driving contexts. We evaluate both methods on CamVid, BDD100K, and GTA driving datasets. Experiments show that UAT-SAM outperforms standard SAM in extreme weather, while SAM2 with uncertainty-aware loss achieves improved performance across diverse driving scenes. These findings underscore the value of explicit uncertainty modeling for safety-critical autonomous driving in challenging environments. 

---
# Beyond I-Con: Exploring New Dimension of Distance Measures in Representation Learning 

**Authors**: Jasmine Shone, Shaden Alshammari, Mark Hamilton, Zhening Li, William Freeman  

**Link**: [PDF](https://arxiv.org/pdf/2509.04734)  

**Abstract**: The Information Contrastive (I-Con) framework revealed that over 23 representation learning methods implicitly minimize KL divergence between data and learned distributions that encode similarities between data points. However, a KL-based loss may be misaligned with the true objective, and properties of KL divergence such as asymmetry and unboundedness may create optimization challenges. We present Beyond I-Con, a framework that enables systematic discovery of novel loss functions by exploring alternative statistical divergences and similarity kernels. Key findings: (1) on unsupervised clustering of DINO-ViT embeddings, we achieve state-of-the-art results by modifying the PMI algorithm to use total variation (TV) distance; (2) on supervised contrastive learning, we outperform the standard approach by using TV and a distance-based similarity kernel instead of KL and an angular kernel; (3) on dimensionality reduction, we achieve superior qualitative results and better performance on downstream tasks than SNE by replacing KL with a bounded f-divergence. Our results highlight the importance of considering divergence and similarity kernel choices in representation learning optimization. 

---
# CoVeR: Conformal Calibration for Versatile and Reliable Autoregressive Next-Token Prediction 

**Authors**: Yuzhu Chen, Yingjie Wang, Shunyu Liu, Yongcheng Jing, Dacheng Tao  

**Link**: [PDF](https://arxiv.org/pdf/2509.04733)  

**Abstract**: Autoregressive pre-trained models combined with decoding methods have achieved impressive performance on complex reasoning tasks. While mainstream decoding strategies such as beam search can generate plausible candidate sets, they often lack provable coverage guarantees, and struggle to effectively balance search efficiency with the need for versatile trajectories, particularly those involving long-tail sequences that are essential in certain real-world applications. To address these limitations, we propose \textsc{CoVeR}, a novel model-free decoding strategy wihtin the conformal prediction framework that simultaneously maintains a compact search space and ensures high coverage probability over desirable trajectories. Theoretically, we establish a PAC-style generalization bound, guaranteeing that \textsc{CoVeR} asymptotically achieves a coverage rate of at least $1 - \alpha$ for any target level $\alpha \in (0,1)$. 

---
# KERAG: Knowledge-Enhanced Retrieval-Augmented Generation for Advanced Question Answering 

**Authors**: Yushi Sun, Kai Sun, Yifan Ethan Xu, Xiao Yang, Xin Luna Dong, Nan Tang, Lei Chen  

**Link**: [PDF](https://arxiv.org/pdf/2509.04716)  

**Abstract**: Retrieval-Augmented Generation (RAG) mitigates hallucination in Large Language Models (LLMs) by incorporating external data, with Knowledge Graphs (KGs) offering crucial information for question answering. Traditional Knowledge Graph Question Answering (KGQA) methods rely on semantic parsing, which typically retrieves knowledge strictly necessary for answer generation, thus often suffer from low coverage due to rigid schema requirements and semantic ambiguity. We present KERAG, a novel KG-based RAG pipeline that enhances QA coverage by retrieving a broader subgraph likely to contain relevant information. Our retrieval-filtering-summarization approach, combined with fine-tuned LLMs for Chain-of-Thought reasoning on knowledge sub-graphs, reduces noises and improves QA for both simple and complex questions. Experiments demonstrate that KERAG surpasses state-of-the-art solutions by about 7% in quality and exceeds GPT-4o (Tool) by 10-21%. 

---
# Bootstrapping Reinforcement Learning with Sub-optimal Policies for Autonomous Driving 

**Authors**: Zhihao Zhang, Chengyang Peng, Ekim Yurtsever, Keith A. Redmill  

**Link**: [PDF](https://arxiv.org/pdf/2509.04712)  

**Abstract**: Automated vehicle control using reinforcement learning (RL) has attracted significant attention due to its potential to learn driving policies through environment interaction. However, RL agents often face training challenges in sample efficiency and effective exploration, making it difficult to discover an optimal driving strategy. To address these issues, we propose guiding the RL driving agent with a demonstration policy that need not be a highly optimized or expert-level controller. Specifically, we integrate a rule-based lane change controller with the Soft Actor Critic (SAC) algorithm to enhance exploration and learning efficiency. Our approach demonstrates improved driving performance and can be extended to other driving scenarios that can similarly benefit from demonstration-based guidance. 

---
# ODKE+: Ontology-Guided Open-Domain Knowledge Extraction with LLMs 

**Authors**: Samira Khorshidi, Azadeh Nikfarjam, Suprita Shankar, Yisi Sang, Yash Govind, Hyun Jang, Ali Kasgari, Alexis McClimans, Mohamed Soliman, Vishnu Konda, Ahmed Fakhry, Xiaoguang Qi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04696)  

**Abstract**: Knowledge graphs (KGs) are foundational to many AI applications, but maintaining their freshness and completeness remains costly. We present ODKE+, a production-grade system that automatically extracts and ingests millions of open-domain facts from web sources with high precision. ODKE+ combines modular components into a scalable pipeline: (1) the Extraction Initiator detects missing or stale facts, (2) the Evidence Retriever collects supporting documents, (3) hybrid Knowledge Extractors apply both pattern-based rules and ontology-guided prompting for large language models (LLMs), (4) a lightweight Grounder validates extracted facts using a second LLM, and (5) the Corroborator ranks and normalizes candidate facts for ingestion. ODKE+ dynamically generates ontology snippets tailored to each entity type to align extractions with schema constraints, enabling scalable, type-consistent fact extraction across 195 predicates. The system supports batch and streaming modes, processing over 9 million Wikipedia pages and ingesting 19 million high-confidence facts with 98.8% precision. ODKE+ significantly improves coverage over traditional methods, achieving up to 48% overlap with third-party KGs and reducing update lag by 50 days on average. Our deployment demonstrates that LLM-based extraction, grounded in ontological structure and verification workflows, can deliver trustworthiness, production-scale knowledge ingestion with broad real-world applicability. A recording of the system demonstration is included with the submission and is also available at this https URL. 

---
# Ecologically Valid Benchmarking and Adaptive Attention: Scalable Marine Bioacoustic Monitoring 

**Authors**: Nicholas R. Rasmussen, Rodrigue Rizk, Longwei Wang, KC Santosh  

**Link**: [PDF](https://arxiv.org/pdf/2509.04682)  

**Abstract**: Underwater Passive Acoustic Monitoring (UPAM) provides rich spatiotemporal data for long-term ecological analysis, but intrinsic noise and complex signal dependencies hinder model stability and generalization. Multilayered windowing has improved target sound localization, yet variability from shifting ambient noise, diverse propagation effects, and mixed biological and anthropogenic sources demands robust architectures and rigorous evaluation. We introduce GetNetUPAM, a hierarchical nested cross-validation framework designed to quantify model stability under ecologically realistic variability. Data are partitioned into distinct site-year segments, preserving recording heterogeneity and ensuring each validation fold reflects a unique environmental subset, reducing overfitting to localized noise and sensor artifacts. Site-year blocking enforces evaluation against genuine environmental diversity, while standard cross-validation on random subsets measures generalization across UPAM's full signal distribution, a dimension absent from current benchmarks. Using GetNetUPAM as the evaluation backbone, we propose the Adaptive Resolution Pooling and Attention Network (ARPA-N), a neural architecture for irregular spectrogram dimensions. Adaptive pooling with spatial attention extends the receptive field, capturing global context without excessive parameters. Under GetNetUPAM, ARPA-N achieves a 14.4% gain in average precision over DenseNet baselines and a log2-scale order-of-magnitude drop in variability across all metrics, enabling consistent detection across site-year folds and advancing scalable, accurate bioacoustic monitoring. 

---
# VCMamba: Bridging Convolutions with Multi-Directional Mamba for Efficient Visual Representation 

**Authors**: Mustafa Munir, Alex Zhang, Radu Marculescu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04669)  

**Abstract**: Recent advances in Vision Transformers (ViTs) and State Space Models (SSMs) have challenged the dominance of Convolutional Neural Networks (CNNs) in computer vision. ViTs excel at capturing global context, and SSMs like Mamba offer linear complexity for long sequences, yet they do not capture fine-grained local features as effectively as CNNs. Conversely, CNNs possess strong inductive biases for local features but lack the global reasoning capabilities of transformers and Mamba. To bridge this gap, we introduce \textit{VCMamba}, a novel vision backbone that integrates the strengths of CNNs and multi-directional Mamba SSMs. VCMamba employs a convolutional stem and a hierarchical structure with convolutional blocks in its early stages to extract rich local features. These convolutional blocks are then processed by later stages incorporating multi-directional Mamba blocks designed to efficiently model long-range dependencies and global context. This hybrid design allows for superior feature representation while maintaining linear complexity with respect to image resolution. We demonstrate VCMamba's effectiveness through extensive experiments on ImageNet-1K classification and ADE20K semantic segmentation. Our VCMamba-B achieves 82.6% top-1 accuracy on ImageNet-1K, surpassing PlainMamba-L3 by 0.3% with 37% fewer parameters, and outperforming Vision GNN-B by 0.3% with 64% fewer parameters. Furthermore, VCMamba-B obtains 47.1 mIoU on ADE20K, exceeding EfficientFormer-L7 by 2.0 mIoU while utilizing 62% fewer parameters. Code is available at this https URL. 

---
# Evaluating NL2SQL via SQL2NL 

**Authors**: Mohammadtaher Safarzadeh, Afshin Oroojlooyjadid, Dan Roth  

**Link**: [PDF](https://arxiv.org/pdf/2509.04657)  

**Abstract**: Robust evaluation in the presence of linguistic variation is key to understanding the generalization capabilities of Natural Language to SQL (NL2SQL) models, yet existing benchmarks rarely address this factor in a systematic or controlled manner. We propose a novel schema-aligned paraphrasing framework that leverages SQL-to-NL (SQL2NL) to automatically generate semantically equivalent, lexically diverse queries while maintaining alignment with the original schema and intent. This enables the first targeted evaluation of NL2SQL robustness to linguistic variation in isolation-distinct from prior work that primarily investigates ambiguity or schema perturbations. Our analysis reveals that state-of-the-art models are far more brittle than standard benchmarks suggest. For example, LLaMa3.3-70B exhibits a 10.23% drop in execution accuracy (from 77.11% to 66.9%) on paraphrased Spider queries, while LLaMa3.1-8B suffers an even larger drop of nearly 20% (from 62.9% to 42.5%). Smaller models (e.g., GPT-4o mini) are disproportionately affected. We also find that robustness degradation varies significantly with query complexity, dataset, and domain -- highlighting the need for evaluation frameworks that explicitly measure linguistic generalization to ensure reliable performance in real-world settings. 

---
# Polysemantic Dropout: Conformal OOD Detection for Specialized LLMs 

**Authors**: Ayush Gupta, Ramneet Kaur, Anirban Roy, Adam D. Cobb, Rama Chellappa, Susmit Jha  

**Link**: [PDF](https://arxiv.org/pdf/2509.04655)  

**Abstract**: We propose a novel inference-time out-of-domain (OOD) detection algorithm for specialized large language models (LLMs). Despite achieving state-of-the-art performance on in-domain tasks through fine-tuning, specialized LLMs remain vulnerable to incorrect or unreliable outputs when presented with OOD inputs, posing risks in critical applications. Our method leverages the Inductive Conformal Anomaly Detection (ICAD) framework, using a new non-conformity measure based on the model's dropout tolerance. Motivated by recent findings on polysemanticity and redundancy in LLMs, we hypothesize that in-domain inputs exhibit higher dropout tolerance than OOD inputs. We aggregate dropout tolerance across multiple layers via a valid ensemble approach, improving detection while maintaining theoretical false alarm bounds from ICAD. Experiments with medical-specialized LLMs show that our approach detects OOD inputs better than baseline methods, with AUROC improvements of $2\%$ to $37\%$ when treating OOD datapoints as positives and in-domain test datapoints as negatives. 

---
# Interpreting Transformer Architectures as Implicit Multinomial Regression 

**Authors**: Jonas A. Actor, Anthony Gruber, Eric C. Cyr  

**Link**: [PDF](https://arxiv.org/pdf/2509.04653)  

**Abstract**: Mechanistic interpretability aims to understand how internal components of modern machine learning models, such as weights, activations, and layers, give rise to the model's overall behavior. One particularly opaque mechanism is attention: despite its central role in transformer models, its mathematical underpinnings and relationship to concepts like feature polysemanticity, superposition, and model performance remain poorly understood. This paper establishes a novel connection between attention mechanisms and multinomial regression. Specifically, we show that in a fixed multinomial regression setting, optimizing over latent features yields optimal solutions that align with the dynamics induced by attention blocks. In other words, the evolution of representations through a transformer can be interpreted as a trajectory that recovers the optimal features for classification. 

---
# Comparative Analysis of Transformer Models in Disaster Tweet Classification for Public Safety 

**Authors**: Sharif Noor Zisad, Ragib Hasan  

**Link**: [PDF](https://arxiv.org/pdf/2509.04650)  

**Abstract**: Twitter and other social media platforms have become vital sources of real time information during disasters and public safety emergencies. Automatically classifying disaster related tweets can help emergency services respond faster and more effectively. Traditional Machine Learning (ML) models such as Logistic Regression, Naive Bayes, and Support Vector Machines have been widely used for this task, but they often fail to understand the context or deeper meaning of words, especially when the language is informal, metaphorical, or ambiguous. We posit that, in this context, transformer based models can perform better than traditional ML models. In this paper, we evaluate the effectiveness of transformer based models, including BERT, DistilBERT, RoBERTa, and DeBERTa, for classifying disaster related tweets. These models are compared with traditional ML approaches to highlight the performance gap. Experimental results show that BERT achieved the highest accuracy (91%), significantly outperforming traditional models like Logistic Regression and Naive Bayes (both at 82%). The use of contextual embeddings and attention mechanisms allows transformer models to better understand subtle language in tweets, where traditional ML models fall short. This research demonstrates that transformer architectures are far more suitable for public safety applications, offering improved accuracy, deeper language understanding, and better generalization across real world social media text. 

---
# Scaling Environments for Organoid Intelligence with LLM-Automated Design and Plasticity-Based Evaluation 

**Authors**: Brennen Hill  

**Link**: [PDF](https://arxiv.org/pdf/2509.04633)  

**Abstract**: As the complexity of artificial agents increases, the design of environments that can effectively shape their behavior and capabilities has become a critical research frontier. We propose a framework that extends this principle to a novel class of agents: biological neural networks in the form of neural organoids. This paper introduces three scalable, closed-loop virtual environments designed to train organoid-based biological agents and probe the underlying mechanisms of learning, such as long-term potentiation (LTP) and long-term depression (LTD). We detail the design of three distinct task environments with increasing complexity: (1) a conditional avoidance task, (2) a one-dimensional predator-prey scenario, and (3) a replication of the classic Pong game. For each environment, we formalize the state and action spaces, the sensory encoding and motor decoding mechanisms, and the feedback protocols based on predictable (reward) and unpredictable (punishment) stimulation. Furthermore, we propose a novel meta-learning approach where a Large Language Model (LLM) is used to automate the generation and optimization of experimental protocols, scaling the process of environment and curriculum design. Finally, we outline a multi-modal approach for evaluating learning by measuring synaptic plasticity at electrophysiological, cellular, and molecular levels. This work bridges the gap between computational neuroscience and agent-based AI, offering a unique platform for studying embodiment, learning, and intelligence in a controlled biological substrate. 

---
# Schema Inference for Tabular Data Repositories Using Large Language Models 

**Authors**: Zhenyu Wu, Jiaoyan Chen, Norman W. Paton  

**Link**: [PDF](https://arxiv.org/pdf/2509.04632)  

**Abstract**: Minimally curated tabular data often contain representational inconsistencies across heterogeneous sources, and are accompanied by sparse metadata. Working with such data is intimidating. While prior work has advanced dataset discovery and exploration, schema inference remains difficult when metadata are limited. We present SI-LLM (Schema Inference using Large Language Models), which infers a concise conceptual schema for tabular data using only column headers and cell values. The inferred schema comprises hierarchical entity types, attributes, and inter-type relationships. In extensive evaluation on two datasets from web tables and open data, SI-LLM achieves promising end-to-end results, as well as better or comparable results to state-of-the-art methods at each step. All source code, full prompts, and datasets of SI-LLM are available at this https URL. 

---
# Action Chunking with Transformers for Image-Based Spacecraft Guidance and Control 

**Authors**: Alejandro Posadas-Nava, Andrea Scorsoglio, Luca Ghilardi, Roberto Furfaro, Richard Linares  

**Link**: [PDF](https://arxiv.org/pdf/2509.04628)  

**Abstract**: We present an imitation learning approach for spacecraft guidance, navigation, and control(GNC) that achieves high performance from limited data. Using only 100 expert demonstrations, equivalent to 6,300 environment interactions, our method, which implements Action Chunking with Transformers (ACT), learns a control policy that maps visual and state observations to thrust and torque commands. ACT generates smoother, more consistent trajectories than a meta-reinforcement learning (meta-RL) baseline trained with 40 million interactions. We evaluate ACT on a rendezvous task: in-orbit docking with the International Space Station (ISS). We show that our approach achieves greater accuracy, smoother control, and greater sample efficiency. 

---
# Measuring the Measures: Discriminative Capacity of Representational Similarity Metrics Across Model Families 

**Authors**: Jialin Wu, Shreya Saha, Yiqing Bo, Meenakshi Khosla  

**Link**: [PDF](https://arxiv.org/pdf/2509.04622)  

**Abstract**: Representational similarity metrics are fundamental tools in neuroscience and AI, yet we lack systematic comparisons of their discriminative power across model families. We introduce a quantitative framework to evaluate representational similarity measures based on their ability to separate model families-across architectures (CNNs, Vision Transformers, Swin Transformers, ConvNeXt) and training regimes (supervised vs. self-supervised). Using three complementary separability measures-dprime from signal detection theory, silhouette coefficients and ROC-AUC, we systematically assess the discriminative capacity of commonly used metrics including RSA, linear predictivity, Procrustes, and soft matching. We show that separability systematically increases as metrics impose more stringent alignment constraints. Among mapping-based approaches, soft-matching achieves the highest separability, followed by Procrustes alignment and linear predictivity. Non-fitting methods such as RSA also yield strong separability across families. These results provide the first systematic comparison of similarity metrics through a separability lens, clarifying their relative sensitivity and guiding metric choice for large-scale model and brain comparisons. 

---
# Sample-efficient Integration of New Modalities into Large Language Models 

**Authors**: Osman Batur İnce, André F. T. Martins, Oisin Mac Aodha, Edoardo M. Ponti  

**Link**: [PDF](https://arxiv.org/pdf/2509.04606)  

**Abstract**: Multimodal foundation models can process several modalities. However, since the space of possible modalities is large and evolving over time, training a model from scratch to encompass all modalities is unfeasible. Moreover, integrating a modality into a pre-existing foundation model currently requires a significant amount of paired data, which is often not available for low-resource modalities. In this paper, we introduce a method for sample-efficient modality integration (SEMI) into Large Language Models (LLMs). To this end, we devise a hypernetwork that can adapt a shared projector -- placed between modality-specific encoders and an LLM -- to any modality. The hypernetwork, trained on high-resource modalities (i.e., text, speech, audio, video), is conditioned on a few samples from any arbitrary modality at inference time to generate a suitable adapter. To increase the diversity of training modalities, we artificially multiply the number of encoders through isometric transformations. We find that SEMI achieves a significant boost in sample efficiency during few-shot integration of new modalities (i.e., satellite images, astronomical images, inertial measurements, and molecules) with encoders of arbitrary embedding dimensionality. For instance, to reach the same accuracy as 32-shot SEMI, training the projector from scratch needs 64$\times$ more data. As a result, SEMI holds promise to extend the modality coverage of foundation models. 

---
# Quantum-Enhanced Multi-Task Learning with Learnable Weighting for Pharmacokinetic and Toxicity Prediction 

**Authors**: Han Zhang, Fengji Ma, Jiamin Su, Xinyue Yang, Lei Wang, Wen-Cai Ye, Li Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04601)  

**Abstract**: Prediction for ADMET (Absorption, Distribution, Metabolism, Excretion, and Toxicity) plays a crucial role in drug discovery and development, accelerating the screening and optimization of new drugs. Existing methods primarily rely on single-task learning (STL), which often fails to fully exploit the complementarities between tasks. Besides, it requires more computational resources while training and inference of each task independently. To address these issues, we propose a new unified Quantum-enhanced and task-Weighted Multi-Task Learning (QW-MTL) framework, specifically designed for ADMET classification tasks. Built upon the Chemprop-RDKit backbone, QW-MTL adopts quantum chemical descriptors to enrich molecular representations with additional information about the electronic structure and interactions. Meanwhile, it introduces a novel exponential task weighting scheme that combines dataset-scale priors with learnable parameters to achieve dynamic loss balancing across tasks. To the best of our knowledge, this is the first work to systematically conduct joint multi-task training across all 13 Therapeutics Data Commons (TDC) classification benchmarks, using leaderboard-style data splits to ensure a standardized and realistic evaluation setting. Extensive experimental results show that QW-MTL significantly outperforms single-task baselines on 12 out of 13 tasks, achieving high predictive performance with minimal model complexity and fast inference, demonstrating the effectiveness and efficiency of multi-task molecular learning enhanced by quantum-informed features and adaptive task weighting. 

---
# Toward Faithfulness-guided Ensemble Interpretation of Neural Network 

**Authors**: Siyu Zhang, Kenneth Mcmillan  

**Link**: [PDF](https://arxiv.org/pdf/2509.04588)  

**Abstract**: Interpretable and faithful explanations for specific neural inferences are crucial for understanding and evaluating model behavior. Our work introduces \textbf{F}aithfulness-guided \textbf{E}nsemble \textbf{I}nterpretation (\textbf{FEI}), an innovative framework that enhances the breadth and effectiveness of faithfulness, advancing interpretability by providing superior visualization. Through an analysis of existing evaluation benchmarks, \textbf{FEI} employs a smooth approximation to elevate quantitative faithfulness scores. Diverse variations of \textbf{FEI} target enhanced faithfulness in hidden layer encodings, expanding interpretability. Additionally, we propose a novel qualitative metric that assesses hidden layer faithfulness. In extensive experiments, \textbf{FEI} surpasses existing methods, demonstrating substantial advances in qualitative visualization and quantitative faithfulness scores. Our research establishes a comprehensive framework for elevating faithfulness in neural network explanations, emphasizing both breadth and precision 

---
# Manipulating Transformer-Based Models: Controllability, Steerability, and Robust Interventions 

**Authors**: Faruk Alpay, Taylan Alpay  

**Link**: [PDF](https://arxiv.org/pdf/2509.04549)  

**Abstract**: Transformer-based language models excel in NLP tasks, but fine-grained control remains challenging. This paper explores methods for manipulating transformer models through principled interventions at three levels: prompts, activations, and weights. We formalize controllable text generation as an optimization problem addressable via prompt engineering, parameter-efficient fine-tuning, model editing, and reinforcement learning. We introduce a unified framework encompassing prompt-level steering, activation interventions, and weight-space edits. We analyze robustness and safety implications, including adversarial attacks and alignment mitigations. Theoretically, we show minimal weight updates can achieve targeted behavior changes with limited side-effects. Empirically, we demonstrate >90% success in sentiment control and factual edits while preserving base performance, though generalization-specificity trade-offs exist. We discuss ethical dual-use risks and the need for rigorous evaluation. This work lays groundwork for designing controllable and robust language models. 

---
# i-Mask: An Intelligent Mask for Breath-Driven Activity Recognition 

**Authors**: Ashutosh Kumar Sinha, Ayush Patel, Mitul Dudhat, Pritam Anand, Rahul Mishra  

**Link**: [PDF](https://arxiv.org/pdf/2509.04544)  

**Abstract**: The patterns of inhalation and exhalation contain important physiological signals that can be used to anticipate human behavior, health trends, and vital parameters. Human activity recognition (HAR) is fundamentally connected to these vital signs, providing deeper insights into well-being and enabling real-time health monitoring. This work presents i-Mask, a novel HAR approach that leverages exhaled breath patterns captured using a custom-developed mask equipped with integrated sensors. Data collected from volunteers wearing the mask undergoes noise filtering, time-series decomposition, and labeling to train predictive models. Our experimental results validate the effectiveness of the approach, achieving over 95\% accuracy and highlighting its potential in healthcare and fitness applications. 

---
# Emergent Social Dynamics of LLM Agents in the El Farol Bar Problem 

**Authors**: Ryosuke Takata, Atsushi Masumori, Takashi Ikegammi  

**Link**: [PDF](https://arxiv.org/pdf/2509.04537)  

**Abstract**: We investigate the emergent social dynamics of Large Language Model (LLM) agents in a spatially extended El Farol Bar problem, observing how they autonomously navigate this classic social dilemma. As a result, the LLM agents generated a spontaneous motivation to go to the bar and changed their decision making by becoming a collective. We also observed that the LLM agents did not solve the problem completely, but rather behaved more like humans. These findings reveal a complex interplay between external incentives (prompt-specified constraints such as the 60\% threshold) and internal incentives (culturally-encoded social preferences derived from pre-training), demonstrating that LLM agents naturally balance formal game-theoretic rationality with social motivations that characterize human behavior. These findings suggest that a new model of group decision making, which could not be handled in the previous game-theoretic problem setting, can be realized by LLM agents. 

---
# In-Context Policy Adaptation via Cross-Domain Skill Diffusion 

**Authors**: Minjong Yoo, Woo Kyung Kim, Honguk Woo  

**Link**: [PDF](https://arxiv.org/pdf/2509.04535)  

**Abstract**: In this work, we present an in-context policy adaptation (ICPAD) framework designed for long-horizon multi-task environments, exploring diffusion-based skill learning techniques in cross-domain settings. The framework enables rapid adaptation of skill-based reinforcement learning policies to diverse target domains, especially under stringent constraints on no model updates and only limited target domain data. Specifically, the framework employs a cross-domain skill diffusion scheme, where domain-agnostic prototype skills and a domain-grounded skill adapter are learned jointly and effectively from an offline dataset through cross-domain consistent diffusion processes. The prototype skills act as primitives for common behavior representations of long-horizon policies, serving as a lingua franca to bridge different domains. Furthermore, to enhance the in-context adaptation performance, we develop a dynamic domain prompting scheme that guides the diffusion-based skill adapter toward better alignment with the target domain. Through experiments with robotic manipulation in Metaworld and autonomous driving in CARLA, we show that our $\oursol$ framework achieves superior policy adaptation performance under limited target domain data conditions for various cross-domain configurations including differences in environment dynamics, agent embodiment, and task horizon. 

---
# Quantized Large Language Models in Biomedical Natural Language Processing: Evaluation and Recommendation 

**Authors**: Zaifu Zhan, Shuang Zhou, Min Zeng, Kai Yu, Meijia Song, Xiaoyi Chen, Jun Wang, Yu Hou, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04534)  

**Abstract**: Large language models have demonstrated remarkable capabilities in biomedical natural language processing, yet their rapid growth in size and computational requirements present a major barrier to adoption in healthcare settings where data privacy precludes cloud deployment and resources are limited. In this study, we systematically evaluated the impact of quantization on 12 state-of-the-art large language models, including both general-purpose and biomedical-specific models, across eight benchmark datasets covering four key tasks: named entity recognition, relation extraction, multi-label classification, and question answering. We show that quantization substantially reduces GPU memory requirements-by up to 75%-while preserving model performance across diverse tasks, enabling the deployment of 70B-parameter models on 40GB consumer-grade GPUs. In addition, domain-specific knowledge and responsiveness to advanced prompting methods are largely maintained. These findings provide significant practical and guiding value, highlighting quantization as a practical and effective strategy for enabling the secure, local deployment of large yet high-capacity language models in biomedical contexts, bridging the gap between technical advances in AI and real-world clinical translation. 

---
# Mitigation of Gender and Ethnicity Bias in AI-Generated Stories through Model Explanations 

**Authors**: Martha O. Dimgba, Sharon Oba, Ameeta Agrawal, Philippe J. Giabbanelli  

**Link**: [PDF](https://arxiv.org/pdf/2509.04515)  

**Abstract**: Language models have been shown to propagate social bias through their output, particularly in the representation of gender and ethnicity. This paper investigates gender and ethnicity biases in AI-generated occupational stories. Representation biases are measured before and after applying our proposed mitigation strategy, Bias Analysis and Mitigation through Explanation (BAME), revealing improvements in demographic representation ranging from 2% to 20%. BAME leverages model-generated explanations to inform targeted prompt engineering, effectively reducing biases without modifying model parameters. By analyzing stories generated across 25 occupational groups, three large language models (Claude 3.5 Sonnet, Llama 3.1 70B Instruct, and GPT-4 Turbo), and multiple demographic dimensions, we identify persistent patterns of overrepresentation and underrepresentation linked to training data stereotypes. Our findings demonstrate that guiding models with their own internal reasoning mechanisms can significantly enhance demographic parity, thereby contributing to the development of more transparent generative AI systems. 

---
# From Silent Signals to Natural Language: A Dual-Stage Transformer-LLM Approach 

**Authors**: Nithyashree Sivasubramaniam  

**Link**: [PDF](https://arxiv.org/pdf/2509.04507)  

**Abstract**: Silent Speech Interfaces (SSIs) have gained attention for their ability to generate intelligible speech from non-acoustic signals. While significant progress has been made in advancing speech generation pipelines, limited work has addressed the recognition and downstream processing of synthesized speech, which often suffers from phonetic ambiguity and noise. To overcome these challenges, we propose an enhanced automatic speech recognition framework that combines a transformer-based acoustic model with a large language model (LLM) for post-processing. The transformer captures full utterance context, while the LLM ensures linguistic consistency. Experimental results show a 16% relative and 6% absolute reduction in word error rate (WER) over a 36% baseline, demonstrating substantial improvements in intelligibility for silent speech interfaces. 

---
# Memristor-Based Neural Network Accelerators for Space Applications: Enhancing Performance with Temporal Averaging and SIRENs 

**Authors**: Zacharia A. Rudge, Dominik Dold, Moritz Fieback, Dario Izzo, Said Hamdioui  

**Link**: [PDF](https://arxiv.org/pdf/2509.04506)  

**Abstract**: Memristors are an emerging technology that enables artificial intelligence (AI) accelerators with high energy efficiency and radiation robustness -- properties that are vital for the deployment of AI on-board spacecraft. However, space applications require reliable and precise computations, while memristive devices suffer from non-idealities, such as device variability, conductance drifts, and device faults. Thus, porting neural networks (NNs) to memristive devices often faces the challenge of severe performance degradation. In this work, we show in simulations that memristor-based NNs achieve competitive performance levels on on-board tasks, such as navigation \& control and geodesy of asteroids. Through bit-slicing, temporal averaging of NN layers, and periodic activation functions, we improve initial results from around $0.07$ to $0.01$ and $0.3$ to $0.007$ for both tasks using RRAM devices, coming close to state-of-the-art levels ($0.003-0.005$ and $0.003$, respectively). Our results demonstrate the potential of memristors for on-board space applications, and we are convinced that future technology and NN improvements will further close the performance gap to fully unlock the benefits of memristors. 

---
# Behavioral Fingerprinting of Large Language Models 

**Authors**: Zehua Pei, Hui-Ling Zhen, Ying Zhang, Zhiyuan Yang, Xing Li, Xianzhi Yu, Mingxuan Yuan, Bei Yu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04504)  

**Abstract**: Current benchmarks for Large Language Models (LLMs) primarily focus on performance metrics, often failing to capture the nuanced behavioral characteristics that differentiate them. This paper introduces a novel ``Behavioral Fingerprinting'' framework designed to move beyond traditional evaluation by creating a multi-faceted profile of a model's intrinsic cognitive and interactive styles. Using a curated \textit{Diagnostic Prompt Suite} and an innovative, automated evaluation pipeline where a powerful LLM acts as an impartial judge, we analyze eighteen models across capability tiers. Our results reveal a critical divergence in the LLM landscape: while core capabilities like abstract and causal reasoning are converging among top models, alignment-related behaviors such as sycophancy and semantic robustness vary dramatically. We further document a cross-model default persona clustering (ISTJ/ESTJ) that likely reflects common alignment incentives. Taken together, this suggests that a model's interactive nature is not an emergent property of its scale or reasoning power, but a direct consequence of specific, and highly variable, developer alignment strategies. Our framework provides a reproducible and scalable methodology for uncovering these deep behavioral differences. Project: this https URL 

---
# VaccineRAG: Boosting Multimodal Large Language Models' Immunity to Harmful RAG Samples 

**Authors**: Qixin Sun, Ziqin Wang, Hengyuan Zhao, Yilin Li, Kaiyou Song, Linjiang Huang, Xiaolin Hu, Qingpei Guo, Si Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04502)  

**Abstract**: Retrieval Augmented Generation enhances the response accuracy of Large Language Models (LLMs) by integrating retrieval and generation modules with external knowledge, demonstrating particular strength in real-time queries and Visual Question Answering tasks. However, the effectiveness of RAG is frequently hindered by the precision of the retriever: many retrieved samples fed into the generation phase are irrelevant or misleading, posing a critical bottleneck to LLMs' performance. To address this challenge, we introduce VaccineRAG, a novel Chain-of-Thought-based retrieval-augmented generation dataset. On one hand, VaccineRAG employs a benchmark to evaluate models using data with varying positive/negative sample ratios, systematically exposing inherent weaknesses in current LLMs. On the other hand, it enhances models' sample-discrimination capabilities by prompting LLMs to generate explicit Chain-of-Thought (CoT) analysis for each sample before producing final answers. Furthermore, to enhance the model's ability to learn long-sequence complex CoT content, we propose Partial-GRPO. By modeling the outputs of LLMs as multiple components rather than a single whole, our model can make more informed preference selections for complex sequences, thereby enhancing its capacity to learn complex CoT. Comprehensive evaluations and ablation studies on VaccineRAG validate the effectiveness of the proposed scheme. The code and dataset will be publicly released soon. 

---
# Understanding Reinforcement Learning for Model Training, and future directions with GRAPE 

**Authors**: Rohit Patel  

**Link**: [PDF](https://arxiv.org/pdf/2509.04501)  

**Abstract**: This paper provides a self-contained, from-scratch, exposition of key algorithms for instruction tuning of models: SFT, Rejection Sampling, REINFORCE, Trust Region Policy Optimization (TRPO), Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), and Direct Preference Optimization (DPO). Explanations of these algorithms often assume prior knowledge, lack critical details, and/or are overly generalized and complex. Here, each method is discussed and developed step by step using simplified and explicit notation focused on LLMs, aiming to eliminate ambiguity and provide a clear and intuitive understanding of the concepts. By minimizing detours into the broader RL literature and connecting concepts to LLMs, we eliminate superfluous abstractions and reduce cognitive overhead. Following this exposition, we provide a literature review of new techniques and approaches beyond those detailed. Finally, new ideas for research and exploration in the form of GRAPE (Generalized Relative Advantage Policy Evolution) are presented. 

---
# Context Engineering for Trustworthiness: Rescorla Wagner Steering Under Mixed and Inappropriate Contexts 

**Authors**: Rushi Wang, Jiateng Liu, Cheng Qian, Yifan Shen, Yanzhou Pan, Zhaozhuo Xu, Ahmed Abbasi, Heng Ji, Denghui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04500)  

**Abstract**: Incorporating external context can significantly enhance the response quality of Large Language Models (LLMs). However, real-world contexts often mix relevant information with disproportionate inappropriate content, posing reliability risks. How do LLMs process and prioritize mixed context? To study this, we introduce the Poisoned Context Testbed, pairing queries with real-world contexts containing relevant and inappropriate content. Inspired by associative learning in animals, we adapt the Rescorla-Wagner (RW) model from neuroscience to quantify how competing contextual signals influence LLM outputs. Our adapted model reveals a consistent behavioral pattern: LLMs exhibit a strong tendency to incorporate information that is less prevalent in the context. This susceptibility is harmful in real-world settings, where small amounts of inappropriate content can substantially degrade response quality. Empirical evaluations on our testbed further confirm this vulnerability. To tackle this, we introduce RW-Steering, a two-stage finetuning-based approach that enables the model to internally identify and ignore inappropriate signals. Unlike prior methods that rely on extensive supervision across diverse context mixtures, RW-Steering generalizes robustly across varying proportions of inappropriate content. Experiments show that our best fine-tuned model improves response quality by 39.8% and reverses the undesirable behavior curve, establishing RW-Steering as a robust, generalizable context engineering solution for improving LLM safety in real-world use. 

---
# DeepTRACE: Auditing Deep Research AI Systems for Tracking Reliability Across Citations and Evidence 

**Authors**: Pranav Narayanan Venkit, Philippe Laban, Yilun Zhou, Kung-Hsiang Huang, Yixin Mao, Chien-Sheng Wu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04499)  

**Abstract**: Generative search engines and deep research LLM agents promise trustworthy, source-grounded synthesis, yet users regularly encounter overconfidence, weak sourcing, and confusing citation practices. We introduce DeepTRACE, a novel sociotechnically grounded audit framework that turns prior community-identified failure cases into eight measurable dimensions spanning answer text, sources, and citations. DeepTRACE uses statement-level analysis (decomposition, confidence scoring) and builds citation and factual-support matrices to audit how systems reason with and attribute evidence end-to-end. Using automated extraction pipelines for popular public models (e.g., GPT-4.5/5, this http URL, Perplexity, Copilot/Bing, Gemini) and an LLM-judge with validated agreement to human raters, we evaluate both web-search engines and deep-research configurations. Our findings show that generative search engines and deep research agents frequently produce one-sided, highly confident responses on debate queries and include large fractions of statements unsupported by their own listed sources. Deep-research configurations reduce overconfidence and can attain high citation thoroughness, but they remain highly one-sided on debate queries and still exhibit large fractions of unsupported statements, with citation accuracy ranging from 40--80% across systems. 

---
# Where Should I Study? Biased Language Models Decide! Evaluating Fairness in LMs for Academic Recommendations 

**Authors**: Krithi Shailya, Akhilesh Kumar Mishra, Gokul S Krishnan, Balaraman Ravindran  

**Link**: [PDF](https://arxiv.org/pdf/2509.04498)  

**Abstract**: Large Language Models (LLMs) are increasingly used as daily recommendation systems for tasks like education planning, yet their recommendations risk perpetuating societal biases. This paper empirically examines geographic, demographic, and economic biases in university and program suggestions from three open-source LLMs: LLaMA-3.1-8B, Gemma-7B, and Mistral-7B. Using 360 simulated user profiles varying by gender, nationality, and economic status, we analyze over 25,000 recommendations. Results show strong biases: institutions in the Global North are disproportionately favored, recommendations often reinforce gender stereotypes, and institutional repetition is prevalent. While LLaMA-3.1 achieves the highest diversity, recommending 481 unique universities across 58 countries, systemic disparities persist. To quantify these issues, we propose a novel, multi-dimensional evaluation framework that goes beyond accuracy by measuring demographic and geographic representation. Our findings highlight the urgent need for bias consideration in educational LMs to ensure equitable global access to higher education. 

---
# A Narrative-Driven Computational Framework for Clinician Burnout Surveillance 

**Authors**: Syed Ahmad Chan Bukhari, Fazel Keshtkar, Alyssa Meczkowska  

**Link**: [PDF](https://arxiv.org/pdf/2509.04497)  

**Abstract**: Clinician burnout poses a substantial threat to patient safety, particularly in high-acuity intensive care units (ICUs). Existing research predominantly relies on retrospective survey tools or broad electronic health record (EHR) metadata, often overlooking the valuable narrative information embedded in clinical notes. In this study, we analyze 10,000 ICU discharge summaries from MIMIC-IV, a publicly available database derived from the electronic health records of Beth Israel Deaconess Medical Center. The dataset encompasses diverse patient data, including vital signs, medical orders, diagnoses, procedures, treatments, and deidentified free-text clinical notes. We introduce a hybrid pipeline that combines BioBERT sentiment embeddings fine-tuned for clinical narratives, a lexical stress lexicon tailored for clinician burnout surveillance, and five-topic latent Dirichlet allocation (LDA) with workload proxies. A provider-level logistic regression classifier achieves a precision of 0.80, a recall of 0.89, and an F1 score of 0.84 on a stratified hold-out set, surpassing metadata-only baselines by greater than or equal to 0.17 F1 score. Specialty-specific analysis indicates elevated burnout risk among providers in Radiology, Psychiatry, and Neurology. Our findings demonstrate that ICU clinical narratives contain actionable signals for proactive well-being monitoring. 

---
# Learned Hallucination Detection in Black-Box LLMs using Token-level Entropy Production Rate 

**Authors**: Charles Moslonka, Hicham Randrianarivo, Arthur Garnier, Emmanuel Malherbe  

**Link**: [PDF](https://arxiv.org/pdf/2509.04492)  

**Abstract**: Hallucinations in Large Language Model (LLM) outputs for Question Answering (QA) tasks critically undermine their real-world reliability. This paper introduces an applied methodology for robust, one-shot hallucination detection, specifically designed for scenarios with limited data access, such as interacting with black-box LLM APIs that typically expose only a few top candidate log-probabilities per token. Our approach derives uncertainty indicators directly from these readily available log-probabilities generated during non-greedy decoding. We first derive an Entropy Production Rate (EPR) metric that offers baseline performance, later augmented with supervised learning. Our learned model uses features representing the entropic contributions of the accessible top-ranked tokens within a single generated sequence, requiring no multiple query re-runs. Evaluated across diverse QA datasets and multiple LLMs, this estimator significantly improves hallucination detection over using EPR alone. Crucially, high performance is demonstrated using only the typically small set of available log-probabilities (e.g., top <10 per token), confirming its practical efficiency and suitability for these API-constrained deployments. This work provides a readily deployable technique to enhance the trustworthiness of LLM responses from a single generation pass in QA and Retrieval-Augmented Generation (RAG) systems, with its utility further demonstrated in a finance framework analyzing responses to queries on annual reports from an industrial dataset. 

---
# Refining Transcripts With TV Subtitles by Prompt-Based Weakly Supervised Training of ASR 

**Authors**: Xinnian Zhao, Hugo Van Hamme  

**Link**: [PDF](https://arxiv.org/pdf/2509.04491)  

**Abstract**: This study proposes a novel approach to using TV subtitles within a weakly supervised (WS) Automatic Speech Recognition (ASR) framework. Although TV subtitles are readily available, their imprecise alignment with corresponding audio limits their applicability as supervised targets for verbatim transcription. Rather than using subtitles as direct supervision signals, our method reimagines them as context-rich prompts. This design enables the model to handle discrepancies between spoken audio and subtitle text. Instead, generated pseudo transcripts become the primary targets, with subtitles acting as guiding cues for iterative refinement. To further enhance the process, we introduce a weighted attention mechanism that emphasizes relevant subtitle tokens during inference. Our experiments demonstrate significant improvements in transcription accuracy, highlighting the effectiveness of the proposed method in refining transcripts. These enhanced pseudo-labeled datasets provide high-quality foundational resources for training robust ASR systems. 

---
# Serialized Output Prompting for Large Language Model-based Multi-Talker Speech Recognition 

**Authors**: Hao Shi, Yusuke Fujita, Tomoya Mizumoto, Lianbo Liu, Atsushi Kojima, Yui Sudo  

**Link**: [PDF](https://arxiv.org/pdf/2509.04488)  

**Abstract**: Prompts are crucial for task definition and for improving the performance of large language models (LLM)-based systems. However, existing LLM-based multi-talker (MT) automatic speech recognition (ASR) systems either omit prompts or rely on simple task-definition prompts, with no prior work exploring the design of prompts to enhance performance. In this paper, we propose extracting serialized output prompts (SOP) and explicitly guiding the LLM using structured prompts to improve system performance (SOP-MT-ASR). A Separator and serialized Connectionist Temporal Classification (CTC) layers are inserted after the speech encoder to separate and extract MT content from the mixed speech encoding in a first-speaking-first-out manner. Subsequently, the SOP, which serves as a prompt for LLMs, is obtained by decoding the serialized CTC outputs using greedy search. To train the model effectively, we design a three-stage training strategy, consisting of serialized output training (SOT) fine-tuning, serialized speech information extraction, and SOP-based adaptation. Experimental results on the LibriMix dataset show that, although the LLM-based SOT model performs well in the two-talker scenario, it fails to fully leverage LLMs under more complex conditions, such as the three-talker scenario. The proposed SOP approach significantly improved performance under both two- and three-talker conditions. 

---
# ASCENDgpt: A Phenotype-Aware Transformer Model for Cardiovascular Risk Prediction from Electronic Health Records 

**Authors**: Chris Sainsbury, Andreas Karwath  

**Link**: [PDF](https://arxiv.org/pdf/2509.04485)  

**Abstract**: We present ASCENDgpt, a transformer-based model specifically designed for cardiovascular risk prediction from longitudinal electronic health records (EHRs). Our approach introduces a novel phenotype-aware tokenization scheme that maps 47,155 raw ICD codes to 176 clinically meaningful phenotype tokens, achieving 99.6\% consolidation of diagnosis codes while preserving semantic information. This phenotype mapping contributes to a total vocabulary of 10,442 tokens - a 77.9\% reduction when compared with using raw ICD codes directly. We pretrain ASCENDgpt on sequences derived from 19402 unique individuals using a masked language modeling objective, then fine-tune for time-to-event prediction of five cardiovascular outcomes: myocardial infarction (MI), stroke, major adverse cardiovascular events (MACE), cardiovascular death, and all-cause mortality. Our model achieves excellent discrimination on the held-out test set with an average C-index of 0.816, demonstrating strong performance across all outcomes (MI: 0.792, stroke: 0.824, MACE: 0.800, cardiovascular death: 0.842, all-cause mortality: 0.824). The phenotype-based approach enables clinically interpretable predictions while maintaining computational efficiency. Our work demonstrates the effectiveness of domain-specific tokenization and pretraining for EHR-based risk prediction tasks. 

---
# The Good, the Bad and the Constructive: Automatically Measuring Peer Review's Utility for Authors 

**Authors**: Abdelrahman Sadallah, Tim Baumgärtner, Iryna Gurevych, Ted Briscoe  

**Link**: [PDF](https://arxiv.org/pdf/2509.04484)  

**Abstract**: Providing constructive feedback to paper authors is a core component of peer review. With reviewers increasingly having less time to perform reviews, automated support systems are required to ensure high reviewing quality, thus making the feedback in reviews useful for authors. To this end, we identify four key aspects of review comments (individual points in weakness sections of reviews) that drive the utility for authors: Actionability, Grounding & Specificity, Verifiability, and Helpfulness. To enable evaluation and development of models assessing review comments, we introduce the RevUtil dataset. We collect 1,430 human-labeled review comments and scale our data with 10k synthetically labeled comments for training purposes. The synthetic data additionally contains rationales, i.e., explanations for the aspect score of a review comment. Employing the RevUtil dataset, we benchmark fine-tuned models for assessing review comments on these aspects and generating rationales. Our experiments demonstrate that these fine-tuned models achieve agreement levels with humans comparable to, and in some cases exceeding, those of powerful closed models like GPT-4o. Our analysis further reveals that machine-generated reviews generally underperform human reviews on our four aspects. 

---
# DecMetrics: Structured Claim Decomposition Scoring for Factually Consistent LLM Outputs 

**Authors**: Minghui Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04483)  

**Abstract**: Claim decomposition plays a crucial role in the fact-checking process by breaking down complex claims into simpler atomic components and identifying their unfactual elements. Despite its importance, current research primarily focuses on generative methods for decomposition, with insufficient emphasis on evaluating the quality of these decomposed atomic claims. To bridge this gap, we introduce \textbf{DecMetrics}, which comprises three new metrics: \texttt{COMPLETENESS}, \texttt{CORRECTNESS}, and \texttt{SEMANTIC ENTROPY}, designed to automatically assess the quality of claims produced by decomposition models. Utilizing these metrics, we develop a lightweight claim decomposition model, optimizing its performance through the integration of these metrics as a reward function. Through automatic evaluation, our approach aims to set a benchmark for claim decomposition, enhancing both the reliability and effectiveness of fact-checking systems. 

---
# Energy Landscapes Enable Reliable Abstention in Retrieval-Augmented Large Language Models for Healthcare 

**Authors**: Ravi Shankar, Sheng Wong, Lin Li, Magdalena Bachmann, Alex Silverthorne, Beth Albert, Gabriel Davis Jones  

**Link**: [PDF](https://arxiv.org/pdf/2509.04482)  

**Abstract**: Reliable abstention is critical for retrieval-augmented generation (RAG) systems, particularly in safety-critical domains such as women's health, where incorrect answers can lead to harm. We present an energy-based model (EBM) that learns a smooth energy landscape over a dense semantic corpus of 2.6M guideline-derived questions, enabling the system to decide when to generate or abstain. We benchmark the EBM against a calibrated softmax baseline and a k-nearest neighbour (kNN) density heuristic across both easy and hard abstention splits, where hard cases are semantically challenging near-distribution queries. The EBM achieves superior abstention performance abstention on semantically hard cases, reaching AUROC 0.961 versus 0.950 for softmax, while also reducing FPR@95 (0.235 vs 0.331). On easy negatives, performance is comparable across methods, but the EBM's advantage becomes most pronounced in safety-critical hard distributions. A comprehensive ablation with controlled negative sampling and fair data exposure shows that robustness stems primarily from the energy scoring head, while the inclusion or exclusion of specific negative types (hard, easy, mixed) sharpens decision boundaries but is not essential for generalisation to hard cases. These results demonstrate that energy-based abstention scoring offers a more reliable confidence signal than probability-based softmax confidence, providing a scalable and interpretable foundation for safe RAG systems. 

---
# Narrative-to-Scene Generation: An LLM-Driven Pipeline for 2D Game Environments 

**Authors**: Yi-Chun Chen, Arnav Jhala  

**Link**: [PDF](https://arxiv.org/pdf/2509.04481)  

**Abstract**: Recent advances in large language models(LLMs) enable compelling story generation, but connecting narrative text to playable visual environments remains an open challenge in procedural content generation(PCG). We present a lightweight pipeline that transforms short narrative prompts into a sequence of 2D tile-based game scenes, reflecting the temporal structure of stories. Given an LLM-generated narrative, our system identifies three key time frames, extracts spatial predicates in the form of "Object-Relation-Object" triples, and retrieves visual assets using affordance-aware semantic embeddings from the GameTileNet dataset. A layered terrain is generated using Cellular Automata, and objects are placed using spatial rules grounded in the predicate structure. We evaluated our system in ten diverse stories, analyzing tile-object matching, affordance-layer alignment, and spatial constraint satisfaction across frames. This prototype offers a scalable approach to narrative-driven scene generation and lays the foundation for future work on multi-frame continuity, symbolic tracking, and multi-agent coordination in story-centered PCG. 

---
# No Clustering, No Routing: How Transformers Actually Process Rare Tokens 

**Authors**: Jing Liu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04479)  

**Abstract**: Large language models struggle with rare token prediction, yet the mechanisms driving their specialization remain unclear. Prior work identified specialized ``plateau'' neurons for rare tokens following distinctive three-regime influence patterns \cite{liu2025emergent}, but their functional organization is unknown. We investigate this through neuron influence analyses, graph-based clustering, and attention head ablations in GPT-2 XL and Pythia models. Our findings show that: (1) rare token processing requires additional plateau neurons beyond the power-law regime sufficient for common tokens, forming dual computational regimes; (2) plateau neurons are spatially distributed rather than forming modular clusters; and (3) attention mechanisms exhibit no preferential routing to specialists. These results demonstrate that rare token specialization arises through distributed, training-driven differentiation rather than architectural modularity, preserving context-sensitive flexibility while achieving adaptive capacity allocation. 

---
# Training Text-to-Molecule Models with Context-Aware Tokenization 

**Authors**: Seojin Kim, Hyeontae Song, Jaehyun Nam, Jinwoo Shin  

**Link**: [PDF](https://arxiv.org/pdf/2509.04476)  

**Abstract**: Recently, text-to-molecule models have shown great potential across various chemical applications, e.g., drug-discovery. These models adapt language models to molecular data by representing molecules as sequences of atoms. However, they rely on atom-level tokenizations, which primarily focus on modeling local connectivity, thereby limiting the ability of models to capture the global structural context within molecules. To tackle this issue, we propose a novel text-to-molecule model, coined Context-Aware Molecular T5 (CAMT5). Inspired by the significance of the substructure-level contexts in understanding molecule structures, e.g., ring systems, we introduce substructure-level tokenization for text-to-molecule models. Building on our tokenization scheme, we develop an importance-based training strategy that prioritizes key substructures, enabling CAMT5 to better capture the molecular semantics. Extensive experiments verify the superiority of CAMT5 in various text-to-molecule generation tasks. Intriguingly, we find that CAMT5 outperforms the state-of-the-art methods using only 2% of training tokens. In addition, we propose a simple yet effective ensemble strategy that aggregates the outputs of text-to-molecule models to further boost the generation performance. Code is available at this https URL. 

---
# ParaThinker: Native Parallel Thinking as a New Paradigm to Scale LLM Test-time Compute 

**Authors**: Hao Wen, Yifan Su, Feifei Zhang, Yunxin Liu, Yunhao Liu, Ya-Qin Zhang, Yuanchun Li  

**Link**: [PDF](https://arxiv.org/pdf/2509.04475)  

**Abstract**: Recent advances in Large Language Models (LLMs) have been driven by test-time compute scaling - a strategy that improves reasoning by generating longer, sequential thought processes. While effective, this approach encounters a significant bottleneck as computation increases, where further computation offers only marginal performance gains. We argue this ceiling is not an inherent limit of the model's capability but a flaw in the scaling strategy itself, a phenomenon we term "Tunnel Vision", where a model's imperfect initial steps lock it into a suboptimal reasoning path. To overcome this, we introduce a new scaling paradigm: native thought parallelism. We present ParaThinker, an end-to-end framework that trains an LLM to generate multiple, diverse reasoning paths in parallel and synthesize them into a superior final answer. By exploring different lines of thoughts simultaneously, ParaThinker effectively sidesteps the Tunnel Vision issue and unlocks the model's latent reasoning potential. Our approach demonstrates that scaling compute in parallel (width) is a more effective and efficient way to superior reasoning than simply scaling sequentially (depth). On challenging reasoning benchmarks, ParaThinker achieves substantial accuracy improvements over sequential LLMs (12.3% for 1.5B and 7.5% for 7B models on average with 8 parallel paths), while adding only negligible latency overhead (7.1%). This enables smaller models to surpass much larger counterparts and establishes parallel thinking as a critical, efficient dimension for scaling future LLMs. 

---
# Scaling Up, Speeding Up: A Benchmark of Speculative Decoding for Efficient LLM Test-Time Scaling 

**Authors**: Shengyin Sun, Yiming Li, Xing Li, Yingzhao Lian, Weizhe Lin, Hui-Ling Zhen, Zhiyuan Yang, Chen Chen, Xianzhi Yu, Mingxuan Yuan, Chen Ma  

**Link**: [PDF](https://arxiv.org/pdf/2509.04474)  

**Abstract**: Test-time scaling has emerged as a powerful paradigm for enhancing the reasoning capabilities of large language models (LLMs) by allocating additional computational resources during inference. However, this paradigm is inherently inefficient due to the generation of redundant and repetitive reasoning traces, leading to significant computational overhead. Speculative decoding offers a promising avenue for mitigating this inefficiency, yet its efficacy in the structured, repetition-rich context of test-time scaling remains largely unexplored. To bridge this gap, we introduce the first comprehensive benchmark designed to evaluate speculative decoding methods for accelerating LLM test-time scaling. Our benchmark provides consistent experimental protocols across representative test-time scaling paradigms (e.g., Best-of-N sampling and multi-round thinking), enabling a fair comparison of three major categories of speculative decoding: model-based, training-based, and n-gram-based methods. Extensive experiments reveal that simple n-gram-based methods effectively capture repetitive patterns, demonstrating unique potential in accelerating test-time scaling. This phenomenon demonstrates the value of integrating n-gram-based methods with model-based or training-based approaches to balance acceleration for both repetitive and diverse reasoning in test-time scaling. We hope this benchmark spurs further research on speculative decoding for test-time scaling, enabling faster and more practical reasoning in LLMs through better handling of repetitive and diverse reasoning paths. 

---
# SpeechLLM: Unified Speech and Language Model for Enhanced Multi-Task Understanding in Low Resource Settings 

**Authors**: Jaekwon Yoo, Kunal Chandiramani, Divya Tadimeti, Abenezer Girma, Chandra Dhir  

**Link**: [PDF](https://arxiv.org/pdf/2509.04473)  

**Abstract**: While integrating speech encoder with LLM requires substantial data and resources, use cases face limitations due to insufficient availability. To address this, we propose a solution with a parameter-efficient adapter that converts speech embeddings into LLM-compatible tokens, focusing on end-to-end automatic speech recognition (ASR), named entity recognition (NER), and sentiment analysis (SA). To reduce labeling costs, we employ an LLM-based synthetic dataset annotation technique. The proposed adapter, using 7x fewer trainable parameters, achieves significant performance gains: a 26% relative Word Error Rates (WER) improvement on the LibriSpeech ASR task, a 6.3% relative F1 score increase on the NER task, and a 32% relative F1 score boost on the SA task. Moreover, using advanced techniques such as adding a classifier regularizer and optimizing the LLM with Low-Rank Adaptation (LoRA) yields notable performance gains, with Spoken Language Understanding Evaluation (SLUE) score improvement of 6.6% and 9.5% 

---
# RECAP: REwriting Conversations for Intent Understanding in Agentic Planning 

**Authors**: Kushan Mitra, Dan Zhang, Hannah Kim, Estevam Hruschka  

**Link**: [PDF](https://arxiv.org/pdf/2509.04472)  

**Abstract**: Understanding user intent is essential for effective planning in conversational assistants, particularly those powered by large language models (LLMs) coordinating multiple agents. However, real-world dialogues are often ambiguous, underspecified, or dynamic, making intent detection a persistent challenge. Traditional classification-based approaches struggle to generalize in open-ended settings, leading to brittle interpretations and poor downstream planning. We propose RECAP (REwriting Conversations for Agent Planning), a new benchmark designed to evaluate and advance intent rewriting, reframing user-agent dialogues into concise representations of user goals. RECAP captures diverse challenges such as ambiguity, intent drift, vagueness, and mixed-goal conversations. Alongside the dataset, we introduce an LLM-based evaluator that assesses planning utility given the rewritten intent. Using RECAP, we develop a prompt-based rewriting approach that outperforms baselines. We further demonstrate that fine-tuning two DPO-based rewriters yields additional utility gains. Our results highlight intent rewriting as a critical and tractable component for improving agent planning in open-domain dialogue systems. 

---
# MOSAIC: A Multilingual, Taxonomy-Agnostic, and Computationally Efficient Approach for Radiological Report Classification 

**Authors**: Alice Schiavone, Marco Fraccaro, Lea Marie Pehrson, Silvia Ingala, Rasmus Bonnevie, Michael Bachmann Nielsen, Vincent Beliveau, Melanie Ganz, Desmond Elliott  

**Link**: [PDF](https://arxiv.org/pdf/2509.04471)  

**Abstract**: Radiology reports contain rich clinical information that can be used to train imaging models without relying on costly manual annotation. However, existing approaches face critical limitations: rule-based methods struggle with linguistic variability, supervised models require large annotated datasets, and recent LLM-based systems depend on closed-source or resource-intensive models that are unsuitable for clinical use. Moreover, current solutions are largely restricted to English and single-modality, single-taxonomy datasets. We introduce MOSAIC, a multilingual, taxonomy-agnostic, and computationally efficient approach for radiological report classification. Built on a compact open-access language model (MedGemma-4B), MOSAIC supports both zero-/few-shot prompting and lightweight fine-tuning, enabling deployment on consumer-grade GPUs. We evaluate MOSAIC across seven datasets in English, Spanish, French, and Danish, spanning multiple imaging modalities and label taxonomies. The model achieves a mean macro F1 score of 88 across five chest X-ray datasets, approaching or exceeding expert-level performance, while requiring only 24 GB of GPU memory. With data augmentation, as few as 80 annotated samples are sufficient to reach a weighted F1 score of 82 on Danish reports, compared to 86 with the full 1600-sample training set. MOSAIC offers a practical alternative to large or proprietary LLMs in clinical settings. Code and models are open-source. We invite the community to evaluate and extend MOSAIC on new languages, taxonomies, and modalities. 

---
# COCORELI: Cooperative, Compositional Reconstitution \& Execution of Language Instructions 

**Authors**: Swarnadeep Bhar, Omar Naim, Eleni Metheniti, Bastien Navarri, Loïc Cabannes, Morteza Ezzabady, Nicholas Asher  

**Link**: [PDF](https://arxiv.org/pdf/2509.04470)  

**Abstract**: We present COCORELI, a hybrid agent framework designed to tackle the limitations of large language models (LLMs) in tasks requiring: following complex instructions, minimizing hallucination, and spatial reasoning. COCORELI integrates medium-sized LLM agents with novel abstraction mechanisms and a discourse module to parse instructions to in-context learn dynamic, high-level representations of the environment. Experiments on natural collaborative construction tasks show that COCORELI outperforms single-LLM CoT and agentic LLM systems, all using larger LLMs. It manages to largely avoid hallucinations, identify missing information, ask for clarifications, and update its learned objects. COCORELI's abstraction abilities extend beyond ENVIRONMENT, as shown in the ToolBench API completion task. 

---
# Multi-Modal Vision vs. Text-Based Parsing: Benchmarking LLM Strategies for Invoice Processing 

**Authors**: David Berghaus, Armin Berger, Lars Hillebrand, Kostadin Cvejoski, Rafet Sifa  

**Link**: [PDF](https://arxiv.org/pdf/2509.04469)  

**Abstract**: This paper benchmarks eight multi-modal large language models from three families (GPT-5, Gemini 2.5, and open-source Gemma 3) on three diverse openly available invoice document datasets using zero-shot prompting. We compare two processing strategies: direct image processing using multi-modal capabilities and a structured parsing approach converting documents to markdown first. Results show native image processing generally outperforms structured approaches, with performance varying across model types and document characteristics. This benchmark provides insights for selecting appropriate models and processing strategies for automated document systems. Our code is available online. 

---
# Evaluating Large Language Models for Financial Reasoning: A CFA-Based Benchmark Study 

**Authors**: Xuan Yao, Qianteng Wang, Xinbo Liu, Ke-Wei Huang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04468)  

**Abstract**: The rapid advancement of large language models presents significant opportunities for financial applications, yet systematic evaluation in specialized financial contexts remains limited. This study presents the first comprehensive evaluation of state-of-the-art LLMs using 1,560 multiple-choice questions from official mock exams across Levels I-III of CFA, most rigorous professional certifications globally that mirror real-world financial analysis complexity. We compare models distinguished by core design priorities: multi-modal and computationally powerful, reasoning-specialized and highly accurate, and lightweight efficiency-optimized.
We assess models under zero-shot prompting and through a novel Retrieval-Augmented Generation pipeline that integrates official CFA curriculum content. The RAG system achieves precise domain-specific knowledge retrieval through hierarchical knowledge organization and structured query generation, significantly enhancing reasoning accuracy in professional financial certification evaluation.
Results reveal that reasoning-oriented models consistently outperform others in zero-shot settings, while the RAG pipeline provides substantial improvements particularly for complex scenarios. Comprehensive error analysis identifies knowledge gaps as the primary failure mode, with minimal impact from text readability. These findings provide actionable insights for LLM deployment in finance, offering practitioners evidence-based guidance for model selection and cost-performance optimization. 

---
# Enhancing LLM Efficiency: Targeted Pruning for Prefill-Decode Disaggregation in Inference 

**Authors**: Hao Zhang, Mengsi Lyu, Yulong Ao, Yonghua Lin  

**Link**: [PDF](https://arxiv.org/pdf/2509.04467)  

**Abstract**: Large Language Models (LLMs) demonstrate exceptional capabilities across various tasks, but their deployment is constrained by high computational and memory costs. Model pruning provides an effective means to alleviate these demands. However, existing methods often ignore the characteristics of prefill-decode (PD) disaggregation in practice. In this paper, we propose a novel pruning method for PD disaggregation inference, enabling more precise and efficient block and KV Cache pruning. Our approach constructs pruning and distillation sets to perform iterative block removal independently for the prefill and decode stages, obtaining better pruning solutions. Moreover, we introduce a token-aware cache pruning mechanism that retains all KV Cache in the prefill stage but selectively reuses entries for the first and last token sequences in selected layers during decode, reducing communication costs with minimal overhead. Extensive experiments demonstrate that our approach consistently achieves strong performance in both PD disaggregation and PD unified settings without disaggregation. Under the default settings, our method achieves a 20.56% inference speedup and a 4.95 times reduction in data transmission bandwidth consumption. 

---
# Just-in-time and distributed task representations in language models 

**Authors**: Yuxuan Li, Declan Campbell, Stephanie C. Y. Chan, Andrew Kyle Lampinen  

**Link**: [PDF](https://arxiv.org/pdf/2509.04466)  

**Abstract**: Many of language models' impressive capabilities originate from their in-context learning: based on instructions or examples, they can infer and perform new tasks without weight updates. In this work, we investigate \emph{when} representations for new tasks are formed in language models, and \emph{how} these representations change over the course of context. We focus on ''transferrable'' task representations -- vector representations that can restore task context in another instance of the model, even without the full prompt. We show that these representations evolve in non-monotonic and sporadic ways, and are distinct from a more inert representation of high-level task categories that persists throughout the context. Specifically, models often condense multiple evidence into these transferrable task representations, which align well with the performance improvement based on more examples in the context. However, this accrual process exhibits strong locality along the sequence dimension, coming online only at certain tokens -- despite task identity being reliably decodable throughout the context. Moreover, these local but transferrable task representations tend to capture minimal ''task scopes'', such as a semantically-independent subtask, and models rely on more temporally-distributed representations to support longer and composite tasks. This two-fold locality (temporal and semantic) underscores a kind of just-in-time computational process underlying language models' ability to adapt to new evidence and learn new tasks on the fly. 

---
# Emotionally-Aware Agents for Dispute Resolution 

**Authors**: Sushrita Rakshit, James Hale, Kushal Chawla, Jeanne M. Brett, Jonathan Gratch  

**Link**: [PDF](https://arxiv.org/pdf/2509.04465)  

**Abstract**: In conflict, people use emotional expressions to shape their counterparts' thoughts, feelings, and actions. This paper explores whether automatic text emotion recognition offers insight into this influence in the context of dispute resolution. Prior work has shown the promise of such methods in negotiations; however, disputes evoke stronger emotions and different social processes. We use a large corpus of buyer-seller dispute dialogues to investigate how emotional expressions shape subjective and objective outcomes. We further demonstrate that large-language models yield considerably greater explanatory power than previous methods for emotion intensity annotation and better match the decisions of human annotators. Findings support existing theoretical models for how emotional expressions contribute to conflict escalation and resolution and suggest that agent-based systems could be useful in managing disputes by recognizing and potentially mitigating emotional escalation. 

---
# Can Multiple Responses from an LLM Reveal the Sources of Its Uncertainty? 

**Authors**: Yang Nan, Pengfei He, Ravi Tandon, Han Xu  

**Link**: [PDF](https://arxiv.org/pdf/2509.04464)  

**Abstract**: Large language models (LLMs) have delivered significant breakthroughs across diverse domains but can still produce unreliable or misleading outputs, posing critical challenges for real-world applications. While many recent studies focus on quantifying model uncertainty, relatively little work has been devoted to \textit{diagnosing the source of uncertainty}. In this study, we show that, when an LLM is uncertain, the patterns of disagreement among its multiple generated responses contain rich clues about the underlying cause of uncertainty. To illustrate this point, we collect multiple responses from a target LLM and employ an auxiliary LLM to analyze their patterns of disagreement. The auxiliary model is tasked to reason about the likely source of uncertainty, such as whether it stems from ambiguity in the input question, a lack of relevant knowledge, or both. In cases involving knowledge gaps, the auxiliary model also identifies the specific missing facts or concepts contributing to the uncertainty. In our experiment, we validate our framework on AmbigQA, OpenBookQA, and MMLU-Pro, confirming its generality in diagnosing distinct uncertainty sources. Such diagnosis shows the potential for relevant manual interventions that improve LLM performance and reliability. 

---
# Multiscale Graph Neural Network for Turbulent Flow-Thermal Prediction Around a Complex-Shaped Pin-Fin 

**Authors**: Riddhiman Raut, Evan M. Mihalko, Amrita Basak  

**Link**: [PDF](https://arxiv.org/pdf/2509.04463)  

**Abstract**: This study presents the development of a domain-responsive edge-aware multiscale Graph Neural Network for predicting steady, turbulent flow and thermal behavior in a two-dimensional channel containing arbitrarily shaped complex pin-fin geometries. The training dataset was constructed through an automated framework that integrated geometry generation, meshing, and flow-field solutions in ANSYS Fluent. The pin-fin geometry was parameterized using piecewise cubic splines, producing 1,000 diverse configurations through Latin Hypercube Sampling. Each simulation was converted into a graph structure, where nodes carried a feature vector containing spatial coordinates, a normalized streamwise position, one-hot boundary indicators, and a signed distance to the nearest boundary such as wall. This graph structure served as input to the newly developed Graph Neural Network, which was trained to predict temperature, velocity magnitude, and pressure at each node using data from ANSYS. The network predicted fields with outstanding accuracy, capturing boundary layers, recirculation, and the stagnation region upstream of the pin-fins while reducing wall time by 2-3 orders of magnitude. In conclusion, the novel graph neural network offered a fast and reliable surrogate for simulations in complex flow configurations. 

---
# Benchmarking GPT-5 for biomedical natural language processing 

**Authors**: Yu Hou, Zaifu Zhan, Rui Zhang  

**Link**: [PDF](https://arxiv.org/pdf/2509.04462)  

**Abstract**: The rapid expansion of biomedical literature has heightened the need for scalable natural language processing (NLP) solutions. While GPT-4 substantially narrowed the gap with task-specific systems, especially in question answering, its performance across other domains remained uneven. We updated a standardized BioNLP benchmark to evaluate GPT-5 and GPT-4o under zero-, one-, and five-shot prompting across 12 datasets spanning six task families: named entity recognition, relation extraction, multi-label document classification, question answering, text summarization, and text simplification. Using fixed prompt templates, identical decoding parameters, and batch inference, we report primary metrics per dataset and include prior results for GPT-4, GPT-3.5, and LLaMA-2-13B for comparison. GPT-5 achieved the strongest overall benchmark performance, with macro-average scores rising to 0.557 under five-shot prompting versus 0.506 for GPT-4 and 0.508 for GPT-4o. On MedQA, GPT-5 reached 94.1% accuracy, exceeding the previous supervised state of the art by over fifty points, and attained parity with supervised systems on PubMedQA (0.734). In extraction tasks, GPT-5 delivered major gains in chemical NER (0.886 F1) and ChemProt relation extraction (0.616 F1), outperforming GPT-4 and GPT-4o, though summarization and disease NER still lagged behind domain-specific baselines. These results establish GPT-5 as a general-purpose model now offering deployment-ready performance for reasoning-oriented biomedical QA, while precision-critical extraction and evidence-dense summarization continue to favor fine-tuned or hybrid approaches. The benchmark delineates where simple prompting suffices and where retrieval-augmented or planning-based scaffolds are likely required, providing actionable guidance for BioNLP system design as frontier models advance. 

---
# CoCoNUTS: Concentrating on Content while Neglecting Uninformative Textual Styles for AI-Generated Peer Review Detection 

**Authors**: Yihan Chen, Jiawei Chen, Guozhao Mo, Xuanang Chen, Ben He, Xianpei Han, Le Sun  

**Link**: [PDF](https://arxiv.org/pdf/2509.04460)  

**Abstract**: The growing integration of large language models (LLMs) into the peer review process presents potential risks to the fairness and reliability of scholarly evaluation. While LLMs offer valuable assistance for reviewers with language refinement, there is growing concern over their use to generate substantive review content. Existing general AI-generated text detectors are vulnerable to paraphrasing attacks and struggle to distinguish between surface language refinement and substantial content generation, suggesting that they primarily rely on stylistic cues. When applied to peer review, this limitation can result in unfairly suspecting reviews with permissible AI-assisted language enhancement, while failing to catch deceptively humanized AI-generated reviews. To address this, we propose a paradigm shift from style-based to content-based detection. Specifically, we introduce CoCoNUTS, a content-oriented benchmark built upon a fine-grained dataset of AI-generated peer reviews, covering six distinct modes of human-AI collaboration. Furthermore, we develop CoCoDet, an AI review detector via a multi-task learning framework, designed to achieve more accurate and robust detection of AI involvement in review content. Our work offers a practical foundation for evaluating the use of LLMs in peer review, and contributes to the development of more precise, equitable, and reliable detection methods for real-world scholarly applications. Our code and data will be publicly available at this https URL. 

---
# Teacher-Student Model for Detecting and Classifying Mitosis in the MIDOG 2025 Challenge 

**Authors**: Seungho Choe, Xiaoli Qin, Abubakr Shafique, Amanda Dy, Susan Done, Dimitrios Androutsos, April Khademi  

**Link**: [PDF](https://arxiv.org/pdf/2509.03614)  

**Abstract**: Counting mitotic figures is time-intensive for pathologists and leads to inter-observer variability. Artificial intelligence (AI) promises a solution by automatically detecting mitotic figures while maintaining decision consistency. However, AI tools are susceptible to domain shift, where a significant drop in performance can occur due to differences in the training and testing sets, including morphological diversity between organs, species, and variations in staining protocols. Furthermore, the number of mitoses is much less than the count of normal nuclei, which introduces severely imbalanced data for the detection task. In this work, we formulate mitosis detection as a pixel-level segmentation and propose a teacher-student model that simultaneously addresses mitosis detection (Track 1) and atypical mitosis classification (Track 2). Our method is based on a UNet segmentation backbone that integrates domain generalization modules, namely contrastive representation learning and domain-adversarial training. A teacher-student strategy is employed to generate pixel-level pseudo-masks not only for annotated mitoses and hard negatives but also for normal nuclei, thereby enhancing feature discrimination and improving robustness against domain shift. For the classification task, we introduce a multi-scale CNN classifier that leverages feature maps from the segmentation model within a multi-task learning paradigm. On the preliminary test set, the algorithm achieved an F1 score of 0.7660 in Track 1 and balanced accuracy of 0.8414 in Track 2, demonstrating the effectiveness of integrating segmentation-based detection and classification into a unified framework for robust mitosis analysis. 

---
# Efficient Training-Free Online Routing for High-Volume Multi-LLM Serving 

**Authors**: Fangzhou Wu, Sandeep Silwal  

**Link**: [PDF](https://arxiv.org/pdf/2509.02718)  

**Abstract**: Increasing demand for Large Language Models (LLMs) services imposes substantial deployment and computation costs on providers. LLM routing offers a cost-efficient solution by directing queries to the optimal LLM based on model and query features. However, existing works primarily focus on offline scenarios and struggle to adapt to online settings with high query volume and constrained token budgets. In this work, we introduce the first training-free algorithm for online routing scenarios. Our algorithm leverages approximate nearest neighbor search to efficiently estimate query features and performs a one-time optimization over a small set of initial queries to learn a routing strategy that guides future routing. We provide theoretical guarantees demonstrating that our algorithm achieves a competitive ratio of $1 - o(1)$ under natural assumptions, which is further validated by extensive experiments across 3 benchmark datasets and 8 baselines, showing an average improvement of 3.55$\times$ in overall performance, 1.85$\times$ in cost efficiency, and nearly 4.25$\times$ in throughput. 

---
# MLP-SRGAN: A Single-Dimension Super Resolution GAN using MLP-Mixer 

**Authors**: Samir Mitha, Seungho Choe, Pejman Jahbedar Maralani, Alan R. Moody, April Khademi  

**Link**: [PDF](https://arxiv.org/pdf/2303.06298)  

**Abstract**: We propose a novel architecture called MLP-SRGAN, which is a single-dimension Super Resolution Generative Adversarial Network (SRGAN) that utilizes Multi-Layer Perceptron Mixers (MLP-Mixers) along with convolutional layers to upsample in the slice direction. MLP-SRGAN is trained and validated using high resolution (HR) FLAIR MRI from the MSSEG2 challenge dataset. The method was applied to three multicentre FLAIR datasets (CAIN, ADNI, CCNA) of images with low spatial resolution in the slice dimension to examine performance on held-out (unseen) clinical data. Upsampled results are compared to several state-of-the-art SR networks. For images with high resolution (HR) ground truths, peak-signal-to-noise-ratio (PSNR) and structural similarity index (SSIM) are used to measure upsampling performance. Several new structural, no-reference image quality metrics were proposed to quantify sharpness (edge strength), noise (entropy), and blurriness (low frequency information) in the absence of ground truths. Results show MLP-SRGAN results in sharper edges, less blurring, preserves more texture and fine-anatomical detail, with fewer parameters, faster training/evaluation time, and smaller model size than existing methods. Code for MLP-SRGAN training and inference, data generators, models and no-reference image quality metrics will be available at this https URL. 

---
